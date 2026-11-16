# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Experiment-only instrumentation; never imported by default training."""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.checkpoint.dcp import DistributedCheckpointer
from cosmos_framework.utils import misc
from cosmos_framework.utils.callback import Callback


class ReadOnlyCheckpointer(DistributedCheckpointer):
    """Load the production checkpoint normally; omit benchmark output checkpoints."""

    def save(self, *args, **kwargs):
        pass


class Observer(Callback):
    def __init__(self):
        super().__init__()
        self.rank = int(os.environ["RANK"])
        self.path = Path(os.environ["UNIFIED_ATTN_RUN"]) / f"rank{self.rank}.jsonl"
        self.started = None
        self.microbatches = 0
        self.losses = []
        self.inputs = []
        self.rng_fingerprints = []
        self.tokens = []
        self.geometry_saved = False
        self.audit = os.environ.get("UNIFIED_ATTN_AUDIT") == "1"
        self.audit_before = None

    def emit(self, record):
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def on_train_start(self, model, iteration=0):
        if os.environ.get("UNIFIED_ATTN_CAPTURE") == "1":
            from tools.unified_attn.capture import install

            install(model)
        misc.set_random_seed(seed=42, by_rank=True)
        if self.audit:
            original = model._add_noise_to_input

            def record_noising(*args, **kwargs):
                result = original(*args, **kwargs)
                if self.audit:
                    hashes = {}
                    for field in ("epsilon_vision", "xt_tokens_vision", "vt_target_vision", "sigmas_vision"):
                        digest = hashlib.sha256()
                        for tensor in getattr(result, field):
                            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
                            digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
                        hashes[field] = digest.hexdigest()
                    self.emit({"event": "noising_audit", "micro": self.microbatches + 1, "hashes": hashes})
                return result

            model._add_noise_to_input = record_noising

        self.emit(
            {
                "event": "start",
                "rank": self.rank,
                "iteration": iteration,
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
            }
        )

    def on_before_dataloading(self, iteration=0):
        if self.started is None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.started = time.perf_counter()

    def on_training_step_start(self, model, data, iteration=0):
        # Seed each logical microbatch independently of compiler/autotuner RNG use.
        seed = 42 + self.rank + 8 * (2 * (iteration - 25000) + self.microbatches)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Noise/timestep sampling follows this hook. Record RNG position, not just input order.
        state = torch.cuda.get_rng_state().numpy().tobytes() + torch.get_rng_state().numpy().tobytes()
        self.rng_fingerprints.append(hashlib.sha256(state).hexdigest())

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        self.microbatches += 1
        self.losses.append(loss.detach())

        # Small text tensors and latent shapes identify replay order without copying videos.
        def encode(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, (list, tuple)):
                return [encode(x) for x in value]
            return value

        identity = {
            key: encode(data_batch[key])
            for key in (
                "__key__",
                "text_token_ids",
                "vae_latent_shapes",
                "cfg_dropped",
                "conditioning_requested_latent_frames",
                "conditioning_effective_latent_frames",
            )
            if key in data_batch
        }
        pack = data_batch.get("packed_sequence")
        if pack is not None:
            if not self.geometry_saved:
                geometry = {
                    "sample_lens": pack.sample_lens,
                    "split_lens": pack.split_lens,
                    "attn_modes": pack.attn_modes,
                }
                (self.path.parent / f"geometry_rank{self.rank}.json").write_text(json.dumps(geometry))
                self.geometry_saved = True
            identity["sample_lens"] = pack.sample_lens
            identity["split_lens"] = pack.split_lens
            self.tokens.append(
                {
                    "und": sum(n for n, m in zip(pack.split_lens, pack.attn_modes) if m == "causal"),
                    "gen": sum(n for n, m in zip(pack.split_lens, pack.attn_modes) if m == "full"),
                }
            )
        self.inputs.append(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())

    def on_after_backward(self, model, iteration=0):
        if not self.audit:
            return
        names, checks = [], []
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                names.append(name)
                checks.append(torch.isfinite(grad).all())
        finite = torch.stack(checks).cpu().tolist()
        bad = [name for name, ok in zip(names, finite) if not ok]
        self.emit(
            {
                "event": "pre_clip_finite_audit",
                "iteration": iteration,
                "micro": self.microbatches + 1,
                "nonfinite_parameters": bad,
            }
        )
        assert not bad, bad

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        if not self.audit or self.audit_before is not None:
            return
        self.audit_before = {}
        from cosmos_framework.callbacks.grad_clip import GradClip

        for cb in self.trainer.callbacks._callbacks:
            if isinstance(cb, GradClip):
                state = cb._states[cb._state_key]["global"].state
                coefficient = (cb.clip_norm / (state + 1e-6)).clamp(max=1.0)
                self.emit(
                    {
                        "event": "clip_audit",
                        "iteration": iteration,
                        "unclipped_global_norm": float(state),
                        "coefficient": float(coefficient),
                    }
                )
        missing, unexpected = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad and p.grad is not None:
                unexpected.append(name)
            if p.requires_grad and p.grad is None:
                missing.append(name)
            if p.grad is None or not self.audit_parameter(name):
                continue

            def local(t):
                return t.to_local() if hasattr(t, "to_local") else t

            self.audit_before[name] = {
                "before": local(p).detach().cpu().clone(),
                "grad_after_clip": local(p.grad).detach().cpu().clone(),
            }
        self.emit(
            {
                "event": "gradient_audit",
                "iteration": iteration,
                "trainable_missing_grad": missing,
                "frozen_unexpected_grad": unexpected,
            }
        )
        assert not unexpected, unexpected

    def audit_parameter(self, name):
        """Choose snapshots; experiments may override to audit all trainable shards."""
        return any(
            key in name for key in ("layers.0.", "layers.17.", "layers.35.", "time_embedder", "vae2llm", "llm2vae")
        )

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        self.emit(
            {
                "event": "step",
                "rank": self.rank,
                "iteration": iteration,
                "seconds": elapsed,
                "compiler_counters": {
                    group: dict(torch._dynamo.utils.counters[group]) for group in ("frames", "stats", "inductor")
                },
                "microbatches": self.microbatches,
                "losses": [float(x) for x in self.losses],
                "input_fingerprints": self.inputs,
                "rng_fingerprints": self.rng_fingerprints,
                "real_tokens": self.tokens,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        if self.audit and self.audit_before is not None:
            for name, p in model.named_parameters():
                if name in self.audit_before:
                    t = p.to_local() if hasattr(p, "to_local") else p
                    self.audit_before[name]["after"] = t.detach().cpu().clone()
            torch.save(self.audit_before, self.path.parent / f"update_rank{self.rank}.pt")
            self.audit = False
            self.audit_before = {}  # Release CPU snapshots after writing.
        self.started = None
        self.microbatches = 0
        self.losses.clear()
        self.inputs.clear()
        self.rng_fingerprints.clear()
        self.tokens.clear()

    def on_train_end(self, model, iteration=0):
        self.emit({"event": "end", "rank": self.rank, "iteration": iteration})
