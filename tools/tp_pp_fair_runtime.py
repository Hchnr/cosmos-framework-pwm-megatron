# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Enforce and audit runtime settings during the raw TP4/FSDP4 experiment."""

import json
import os
import socket
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

EXPECTED = {
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cudnn_allow_tf32": False,
    "matmul_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "deterministic_algorithms": False,
    "deterministic_warn_only": False,
}


def runtime_flags():
    return {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }


class RuntimeAudit:
    def __init__(self, output, backend):
        self.path = Path(output) / f"runtime_fairness.rank{os.environ['RANK']}.json"
        self.backend = backend
        self.events = []
        self.compile_config = None
        self.vae = None
        self.compiled_module_count = None
        self.compiled_layer_indices = None
        self.compile_calls = []
        self.model_compile_calls = []
        self.steps = []
        self.window_end_ns = None
        torch.use_deterministic_algorithms(False, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        self.wrap_compile()
        self.wrap_vae_compiler()

    def check(self, phase):
        actual = runtime_flags()
        compiler = {
            "recompile_limit": torch._dynamo.config.recompile_limit,
            "use_duck_shape": torch.fx.experimental._config.use_duck_shape,
        }
        self.events.append({"phase": phase, "flags": actual, "compiler_runtime": compiler})
        if actual != EXPECTED:
            raise RuntimeError(f"Runtime settings differ at {phase}: {actual}")
        if compiler != {
            "recompile_limit": self.compile_settings["trainer_recompile_limit"],
            "use_duck_shape": self.compile_settings["trainer_use_duck_shape"],
        }:
            raise RuntimeError(f"Runtime compiler settings differ at {phase}: {compiler}")

    def config(self, config):
        self.compile_config = OmegaConf.to_container(config.model.config.compile, resolve=True)
        trainer = config.trainer
        if not trainer.cudnn.deterministic or trainer.cudnn.benchmark:
            raise RuntimeError("cuDNN config is not the matched configuration")
        self.compile_settings = {
            "model": self.compile_config,
            "trainer_recompile_limit": trainer.compile_config.recompile_limit,
            "trainer_use_duck_shape": trainer.compile_config.use_duck_shape,
        }
        if not self.compile_config.get("enabled"):
            raise RuntimeError("Model compile must be enabled")

    def wrap_compile(self):
        original = torch.compile

        def traced(*args, **kwargs):
            self.compile_calls.append({key: repr(value) for key, value in sorted(kwargs.items())})
            if args and isinstance(args[0], torch.nn.Module):
                self.model_compile_calls.append(
                    {
                        "fullgraph": kwargs.get("fullgraph", False),
                        "dynamic": kwargs.get("dynamic"),
                        "mode": kwargs.get("mode"),
                        "backend": kwargs.get("backend", "inductor"),
                        "options": kwargs.get("options"),
                    }
                )
            return original(*args, **kwargs)

        torch.compile = traced

    def wrap_vae_compiler(self):
        from cosmos_framework.callbacks.compile_tokenizer import CompileTokenizer

        original = CompileTokenizer.on_training_step_start

        def traced(compiler, model, *args, **kwargs):
            self.check("vae_compiler_before")
            result = original(compiler, model, *args, **kwargs)
            self.check("vae_compiler_after")
            tokenizer = model.tokenizer_vision_gen
            if compiler._has_compiled:
                functions = getattr(tokenizer.model, "_aot_chunk_fns", {})
                self.vae = {
                    "enabled": compiler.enabled,
                    "compiled": compiler._has_compiled,
                    "backend": compiler.backend,
                    "mode": compiler.mode,
                    "fullgraph": compiler.fullgraph,
                    "dynamic": compiler.dynamic,
                    "require_full_coverage": compiler.require_full_coverage,
                    "variant_keys": sorted(map(str, functions)),
                }
                if not self.vae["enabled"] or not self.vae["variant_keys"]:
                    raise RuntimeError("VAE compile did not install any AOT functions")
            return result

        CompileTokenizer.on_training_step_start = traced

    def modules(self, model):
        from torch._dynamo.eval_frame import OptimizedModule

        layers = model.layers if self.backend == "tp" else model.net.language_model.model.layers
        entries = list(layers.named_children())
        indices = sorted(int(name) for name, _ in entries)
        if (
            not indices
            or indices != list(range(len(indices)))
            or any(not isinstance(module, OptimizedModule) for _, module in entries)
        ):
            raise RuntimeError(f"Compiled language layer coverage is incomplete or ambiguous: {indices}")
        count = len(indices)
        if self.compiled_layer_indices is not None and indices != self.compiled_layer_indices:
            raise RuntimeError("Compiled language layer coverage changed during training")
        self.compiled_module_count = count
        self.compiled_layer_indices = indices

    def begin_update(self, iteration):
        import torch.distributed as dist

        if len(self.steps) >= 7 or iteration != 1200 + len(self.steps):
            raise RuntimeError("Unexpected or duplicate update timing boundary")
        if len(self.steps) == 2:
            torch.cuda.synchronize()
            dist.barrier()
        self.steps.append({"iteration": iteration + 1, "start_ns": time.monotonic_ns()})

    def end_update(self, iteration):
        import torch.distributed as dist

        if not self.steps or self.steps[-1]["iteration"] != iteration + 1 or "end_ns" in self.steps[-1]:
            raise RuntimeError("Unexpected optimizer timing boundary")
        self.steps[-1]["end_ns"] = time.monotonic_ns()
        if len(self.steps) == 7:
            torch.cuda.synchronize()
            dist.barrier()
            self.window_end_ns = time.monotonic_ns()

    def finish(self, error=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "rank": int(os.environ["RANK"]),
            "backend": self.backend,
            "passed": False,
            "events": self.events,
            "compile_settings": getattr(self, "compile_settings", None),
            "vae_compile": self.vae,
            "compiled_module_count": self.compiled_module_count,
            "compiled_layer_indices": self.compiled_layer_indices,
            "torch_compile_calls": self.compile_calls,
            "model_compile_calls": self.model_compile_calls,
            "clock": "CLOCK_MONOTONIC via time.monotonic_ns",
            "hostname": socket.gethostname(),
            "steps": self.steps,
            "window_end_ns": self.window_end_ns,
            "failure": str(error) if error else None,
        }
        if error is None:
            if (
                not self.events
                or not self.compile_settings
                or not self.vae
                or not self.compiled_module_count
                or not self.compiled_layer_indices
                or not self.compile_calls
                or not self.model_compile_calls
                or len(self.steps) != 7
                or not self.window_end_ns
            ):
                raise RuntimeError("Incomplete runtime fairness audit")
            report["passed"] = True
        self.path.write_text(json.dumps(report, indent=2) + "\n")


def validate_reports(root):
    reports = []
    for backend in ("tp", "fsdp"):
        for rank in range(4):
            report = json.loads((Path(root) / backend / f"runtime_fairness.rank{rank}.json").read_text())
            if report["rank"] != rank or report["backend"] != backend or not report["passed"]:
                raise ValueError("Invalid runtime fairness report")
            if not report["events"] or any(row["flags"] != EXPECTED for row in report["events"]):
                raise ValueError("Runtime flags differ across training")
            expected_compiler = {
                "recompile_limit": report["compile_settings"]["trainer_recompile_limit"],
                "use_duck_shape": report["compile_settings"]["trainer_use_duck_shape"],
            }
            if any(row["compiler_runtime"] != expected_compiler for row in report["events"]):
                raise ValueError("Runtime compiler flags differ across training")
            if (
                not report["compiled_module_count"]
                or not report["compiled_layer_indices"]
                or not report["torch_compile_calls"]
            ):
                raise ValueError("Model compilation missing")
            if not report["model_compile_calls"]:
                raise ValueError("Actual model compile calls missing")
            if (
                len(report["steps"]) != 7
                or not report["window_end_ns"]
                or [step["iteration"] for step in report["steps"]] != list(range(1201, 1208))
                or any(step["end_ns"] <= step["start_ns"] for step in report["steps"])
            ):
                raise ValueError("Incomplete actual step timing")
            if sum(row["phase"] == "microbatch_before" for row in report["events"]) != 21:
                raise ValueError("Missing 21 runtime microbatch samples")
            if sum(row["phase"] == "microbatch_after_prepare" for row in report["events"]) != 21:
                raise ValueError("Missing 21 post-prepare runtime samples")
            reports.append(report)
    settings = reports[0]["compile_settings"]
    vae = reports[0]["vae_compile"]
    if any(row["compile_settings"] != settings or row["vae_compile"] != vae for row in reports):
        raise ValueError("Model or VAE compile settings differ across ranks/backends")
    indices = reports[0]["compiled_layer_indices"]
    if indices != list(range(len(indices))) or any(row["compiled_layer_indices"] != indices for row in reports):
        raise ValueError("Compiled language layer coverage differs across ranks/backends")
    signatures = {json.dumps(call, sort_keys=True) for call in reports[0]["model_compile_calls"]}
    if any({json.dumps(call, sort_keys=True) for call in row["model_compile_calls"]} != signatures for row in reports):
        raise ValueError("Actual torch.compile model settings differ across ranks/backends")
    if len({row["hostname"] for row in reports}) != 1:
        raise ValueError("Full-window timing requires one host clock")
    windows = {}
    for backend in ("tp", "fsdp"):
        rows = [row for row in reports if row["backend"] == backend]
        start = min(row["steps"][2]["start_ns"] for row in rows)
        end = max(row["window_end_ns"] for row in rows)
        if end <= start or any(row["window_end_ns"] < row["steps"][-1]["end_ns"] for row in rows):
            raise ValueError("Invalid synchronized measurement window")
        windows[backend] = {
            "start_ns": start,
            "end_ns": end,
            "seconds": (end - start) / 1e9,
            "per_rank_steps": {str(row["rank"]): row["steps"] for row in rows},
        }
    return {
        "passed": True,
        "flags": EXPECTED,
        "compile_settings": settings,
        "vae_compile": vae,
        "model_compile_signatures": sorted(signatures),
        "compiled_layer_indices": indices,
        "windows": windows,
        "compiled_module_counts": {f"{r['backend']}:{r['rank']}": r["compiled_module_count"] for r in reports},
    }
