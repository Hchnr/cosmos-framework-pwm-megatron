"""Measure full production-capacity compute memory without Adam state allocation.

Uses the full model, real-batch replay, original capacity/short-UND policy,
activation checkpointing and GA=3. No optimizer update is performed. The probe
is a memory qualification aid, not a training correctness or throughput test.
"""

import json
import os
from pathlib import Path

import torch
import train_real_batch_smoke as driver

from cosmos_framework.utils.generator.optimizer import OptimizersContainer

_original_load = driver.load_experiment_from_toml


def full_capacity_config(*args, **kwargs):
    prefixes = (
        "model.config.fixed_sequence_length.",
        "model.config.action_tower_und_short_tokens=",
        "model.config.max_num_tokens_after_packing=",
        "dataloader_train.max_sequence_length=",
    )
    # Retain the production TOML's physical capacities and short-UND policy.
    if os.environ.get("MEMORY_PROBE_TOKENIZER_COMPILE") == "1":
        prefixes += ("trainer.callbacks.compile_tokenizer.enabled=",)
    kwargs["extra_overrides"] = [item for item in kwargs["extra_overrides"] if not item.startswith(prefixes)]
    if os.environ.get("MEMORY_PROBE_FORCE_FULL_UND") == "1":
        kwargs["extra_overrides"].append("model.config.action_tower_und_short_tokens=0")
    return _original_load(*args, **kwargs)


def no_optimizer_update(self):
    if self._megatron_runtime is not None:
        self._megatron_runtime.after_step()


if __name__ == "__main__":
    driver.load_experiment_from_toml = full_capacity_config
    OptimizersContainer.step = no_optimizer_update
    driver.main()
    if int(os.environ["RANK"]) == 0:
        import sys

        output = Path(sys.argv[sys.argv.index("--benchmark-json") + 1])
        report = json.loads(output.read_text())
        report.update(
            scope="full 36-layer model, original production token capacities, real-batch replay, no optimizer update",
            optimizer_updates_disabled=True,
            optimizer_state_allocated=False,
            token_capacities=[11776, 62592, 2944],
            short_und_capacity=0 if os.environ.get("MEMORY_PROBE_FORCE_FULL_UND") == "1" else 5120,
            maximum_physical_tokens=74368,
            throughput_qualification=False,
            tokenizer_compile_enabled=os.environ.get("MEMORY_PROBE_TOKENIZER_COMPILE") == "1",
        )
        # BF16 FusedAdam: FP32 master + first moment + second moment for the
        # 7,991,321,792 selected parameters. Keep the measured shard4 gradient
        # buffers rather than subtracting their excess over shard16.
        adam_bytes = 7991321792 * 12 // 16
        report["shard16_adam_state_bytes"] = adam_bytes
        report["conservative_compute_plus_shard16_adam_bytes"] = report["peak_allocated_bytes"] + adam_bytes
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["excludes"] += [
            "optimizer update workspace",
            "192-GPU communication workspaces",
            "tokenizer CUDA graphs",
        ]
        if report["tokenizer_compile_enabled"]:
            report["excludes"] = [
                item for item in report["excludes"] if item not in ("tokenizer compilation", "tokenizer CUDA graphs")
            ]
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
