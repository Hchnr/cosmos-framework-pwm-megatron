"""Structural audit for small action towers; no evaluation or RNG changes."""

import json
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from cosmos_framework.utils.callback import Callback


class SmallActionAudit(Callback):
    def __init__(self, expected_global_batch_size: int = 16):
        super().__init__()
        if expected_global_batch_size < 1:
            raise ValueError("expected_global_batch_size must be positive")
        self.expected_global_batch_size = expected_global_batch_size

    def on_train_start(self, model, iteration=0):
        world = dist.get_world_size() if dist.is_initialized() else 1
        gbs = world * self.config.trainer.grad_accum_iter * self.config.dataloader_train.max_samples_per_batch
        if gbs != self.expected_global_batch_size:
            raise ValueError(f"Expected global batch {self.expected_global_batch_size}, got {gbs}")
        counts = dict(UND=0, GEN_VIDEO=0, GEN_ACTION=0)
        action_count = 0
        for name, p in model.net.named_parameters():
            is_action = any(
                part in name
                for part in (
                    "moe_gen_action",
                    "action_time_embedder",
                    "action2llm",
                    "llm2action",
                    "action_modality_embed",
                )
            )
            if is_action:
                action_count += p.numel()
                assert p.requires_grad and p.dtype == torch.bfloat16, name
            if "language_model.model.layers." in name:
                role = "GEN_ACTION" if "_moe_gen_action" in name else "GEN_VIDEO" if "_moe_gen" in name else "UND"
                assert p.requires_grad == (role != "UND"), name
                assert p.dtype == torch.bfloat16, name
                counts[role] += p.numel()
        assert 900_000_000 <= action_count <= 1_100_000_000, action_count
        decoder = model.net.language_model.model
        report = dict(
            preset=model.config.action_tower_preset,
            global_batch_size=gbs,
            world_size=world,
            grad_accum_iter=self.config.trainer.grad_accum_iter,
            transformer_parameters=counts,
            action_parameters_including_heads_and_adapters=action_count,
            und_frozen=True,
            precision="bfloat16",
            attention_backend=decoder.small_action_backend,
            fixed_tower_lengths=decoder.small_action_fixed_lengths,
            checkpoint_wrapped_blocks=sum(isinstance(module, CheckpointWrapper) for module in decoder.layers.modules()),
        )
        self.output = Path(self.config.job.path_local) / "small_action_audit"
        self.output.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        (self.output / f"rank_{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        if rank == 0 and wandb.run:
            wandb.run.summary.update(report)

    def on_training_step_start(self, model, data, iteration=0):
        plans = data.get("sequence_plan", [])
        expected = self.config.dataloader_train.max_samples_per_batch
        if len(plans) != expected:
            raise ValueError(
                f"Fixed-GBS action training requires {expected} samples per pack, got {len(plans)}; "
                "increase token capacity or use a sample-budget-aware training recipe."
            )
        for plan in plans:
            if plan.robot_state_frame_indexes or plan.condition_frame_indexes_action:
                raise ValueError("The small-action pilot forbids robot state and action history")
