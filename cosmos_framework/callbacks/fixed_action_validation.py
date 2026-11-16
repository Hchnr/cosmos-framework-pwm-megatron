"""Fixed held-out flow losses for action pilots, without changing training RNGs."""

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb

from cosmos_framework.data.generator.sequence_packing.static_layout import (
    PACKED_SEQUENCE_KEY,
    StaticPackingConfig,
    build_static_packed_sequence,
)
from cosmos_framework.utils import log, misc
from cosmos_framework.utils.callback import Callback


class FixedActionValidation(Callback):
    def __init__(self, fixture_path: str, every_n: int = 100, seed: int = 2026):
        super().__init__()
        if every_n < 1:
            raise ValueError("every_n must be positive")
        self.fixture_path = Path(fixture_path)
        self.every_n = every_n
        self.seed = seed
        self.last_iteration = None

    def on_train_start(self, model, iteration=0):
        # The fixture is a trusted local artifact produced from the Parquet loader.
        fixture = torch.load(self.fixture_path, map_location="cpu", weights_only=False)
        self.batches = fixture["reference_batches"]
        self.fixture_sha256 = hashlib.sha256(self.fixture_path.read_bytes()).hexdigest()
        if not self.batches:
            raise ValueError("Evaluation fixture has no reference batches")
        for batch in self.batches:
            for plan in batch["sequence_plan"]:
                if plan.robot_state_frame_indexes or plan.condition_frame_indexes_action:
                    raise ValueError("This no-state pilot must not condition on robot state or past action")
            # Offline sampling fixtures predate the training-only static layout.
            # Use the identical builder and geometry as the Parquet collator.
            keys = (
                "sequence_plan",
                "text_token_ids",
                "video",
                "image_size",
                "action",
                "domain_id",
                "raw_action_dim",
                "conditioning_fps",
                "conditioning_fps_action",
                "decode_metadata",
                "action_loss_mask",
                "vision_loss_mask",
            )
            if getattr(model.config, "action128_spec", None) is not None:
                required = (
                    "action_loss_mask",
                    "vision_loss_mask",
                    "normalization_domain_id",
                    "action128_contract_fingerprint",
                )
                missing = [key for key in required if key not in batch]
                if missing:
                    raise ValueError(f"Action128 validation fixture lacks required metadata: {missing}")
            samples = [{key: batch[key][i] for key in keys if key in batch} for i in range(len(batch["sequence_plan"]))]
            for sample in samples:
                # A fixture captured after training_step contains normalized
                # [1,C,T,H,W] video; only the layout proxy needs reader geometry.
                video = sample.get("video")
                while isinstance(video, (list, tuple)) and len(video) == 1:
                    video = video[0]
                if isinstance(video, torch.Tensor) and video.ndim == 5 and video.shape[0] == 1:
                    sample["video"] = video[0]
                # Fixtures contain the collated per-sample batch axis; the
                # layout builder consumes the original reader mask shapes.
                for key, ndim in (("action_loss_mask", 2), ("vision_loss_mask", 3)):
                    if key not in sample:
                        continue
                    mask = sample[key]
                    while isinstance(mask, (list, tuple)) and len(mask) == 1:
                        mask = mask[0]
                    if isinstance(mask, torch.Tensor) and mask.ndim == ndim + 1 and mask.shape[0] == 1:
                        mask = mask[0]
                    if not isinstance(mask, torch.Tensor) or mask.ndim != ndim or mask.dtype != torch.bool:
                        raise ValueError(f"Invalid validation {key}: expected a {ndim}D boolean mask")
                    sample[key] = mask
            if "video_latent" in batch:
                from cosmos_framework.data.generator.offline_vae import VaeLatentRef

                for i, sample in enumerate(samples):
                    latent = batch["video_latent"][i]
                    while isinstance(latent, (list, tuple)):
                        latent = latent[0]
                    sample.pop("video", None)
                    sample["video_latent_ref"] = VaeLatentRef(
                        path="/fixture.safetensors",
                        tensor_key=str(i),
                        latent_shape=tuple(latent.shape[-4:]),
                        target_fps=15,
                        catalog_id=str(i),
                        dataset="fixed_action_validation",
                    )
            batch[PACKED_SEQUENCE_KEY] = build_static_packed_sequence(
                samples,
                special_tokens=model.llm_special_tokens,
                config=StaticPackingConfig(**self.config.dataloader_train.static_packing_config),
            )
        counts = {}
        for name, parameter in model.net.named_parameters():
            if "language_model.model.layers." not in name:
                continue
            role = "GEN_ACTION" if "_moe_gen_action" in name else "GEN_VIDEO" if "_moe_gen" in name else "UND"
            if parameter.requires_grad != (role != "UND"):
                raise RuntimeError(f"Unexpected freeze rule: {name}")
            if parameter.dtype != torch.bfloat16:
                raise RuntimeError(f"Expected BF16 tower parameter: {name}: {parameter.dtype}")
            counts[role] = counts.get(role, 0) + parameter.numel()
        expected = {"UND", "GEN_VIDEO"}
        if model.config.tower_structure == "three_tower":
            expected.add("GEN_ACTION")
        if set(counts) != expected:
            raise RuntimeError(f"Unexpected tower parameter groups: {counts}")
        self._write("trainability", dict(iteration=iteration, parameters=counts, und_frozen=True))
        if (not dist.is_initialized() or dist.get_rank() == 0) and wandb.run:
            wandb.define_metric("validation/iteration")
            wandb.define_metric("validation/*", step_metric="validation/iteration")
        self.evaluate(model, iteration)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration % self.every_n == 0:
            self.evaluate(model, iteration)

    def on_train_end(self, model, iteration=0):
        if self.last_iteration != iteration:
            self.evaluate(model, iteration)

    def _write(self, name, report):
        if not dist.is_initialized() or dist.get_rank() == 0:
            output = Path(self.config.job.path_local) / "fixed_action_validation"
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")

    @torch.no_grad()
    def evaluate(self, model, iteration):
        # Every FSDP rank executes the same small fixture, in the same order.
        # Global averaging therefore introduces no uneven-sample weighting.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        module_modes = [(module, module.training) for module in model.modules()]
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        rows = []
        sample_counts = []
        try:
            with torch.random.fork_rng(devices=devices):
                model.eval()
                for index, original in enumerate(self.batches):
                    seed = self.seed + index
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.random.default_generator.manual_seed(seed)
                    if devices:
                        torch.cuda.default_generators[devices[0]].manual_seed(seed)
                    batch = misc.to(copy.deepcopy(original), device="cuda" if devices else "cpu")
                    # GT is the flow-matching target, never an extra policy condition.
                    # Deterministic training samplers can seed private generators
                    # from this argument. Keep it fixed per sample as well, rather
                    # than changing the validation noise with the training step.
                    output, loss = model.training_step(batch, seed)
                    metrics = {"loss": loss.detach().float()}
                    metrics.update(
                        (key, value.detach().float())
                        for key, value in output.items()
                        if key.startswith("flow_matching_loss") and isinstance(value, torch.Tensor)
                    )
                    for key, value in metrics.items():
                        # Per-instance metrics can contain multiple samples.
                        value = value.mean().clone()
                        if dist.is_initialized():
                            dist.all_reduce(value)
                            value /= dist.get_world_size()
                        if not torch.isfinite(value).all():
                            raise RuntimeError(f"Non-finite held-out metric: {key}")
                        metrics[key] = float(value.item())
                    rows.append(metrics)
                    sample_counts.append(len(original.get("sequence_plan", [None])))
                    del batch, output, loss
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, was_training in module_modes:
                module.training = was_training
        mean = {
            key: sum(row[key] * count for row, count in zip(rows, sample_counts, strict=True)) / sum(sample_counts)
            for key in rows[0]
        }
        report = dict(
            iteration=iteration,
            seed=self.seed,
            fixture_sha256=self.fixture_sha256,
            samples=rows,
            sample_counts=sample_counts,
            mean=mean,
        )
        self._write(f"iter_{iteration:09d}", report)
        if (not dist.is_initialized() or dist.get_rank() == 0) and wandb.run:
            # Use a dedicated x axis: a training callback may already have
            # committed this optimizer step. Do not advance its W&B step counter.
            wandb.log(
                {"validation/iteration": iteration, **{f"validation/{key}": value for key, value in mean.items()}},
                commit=False,
            )
        log.info(f"Fixed held-out action validation at {iteration}: {mean}")
        self.last_iteration = iteration
