# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Restore BF16 compute from TP-specific nonzero masters, including DP shards.

Run with four GPUs, --pp=2 (DP1) or --pp=1 (DP2). Replicated parameters
deliberately cross a BF16 rounding boundary between TP ranks. A deduplicated
model checkpoint cannot by itself restore both original compute copies.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.model.generator.megatron_tp_pp.payload import PayloadLayout
from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
from cosmos_framework.model.generator.mot.small_action_optimization_test import make_layer
from cosmos_framework.training_backends.megatron_tp_pp import checkpoint
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver
from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states, initialize_adam_state
from cosmos_framework.training_backends.megatron_tp_pp.runtime import initialize, transformer_config


def main():
    from megatron.core import parallel_state as ps
    from megatron.core.optimizer import OptimizerConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    topology = ModelParallelConfig(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=args.pp,
        global_packed_microbatches_per_update=12,
        logical_data_parallel_size=4,
        reference_micro_steps_per_update=3,
    )
    initialize(topology)
    torch.manual_seed(732)
    layers = [make_layer()[0] for _ in range(2)]
    config = transformer_config(topology, num_layers=2, hidden_size=64, num_attention_heads=4, dtype=torch.bfloat16)
    stage = PipelineStage(
        config,
        lambda index: copy.deepcopy(layers[index]),
        layout=PayloadLayout(((8, 64), (12, 64), (6, 32))),
    ).to(device="cuda", dtype=torch.bfloat16)
    driver = ModelParallelDriver(stage, topology, OptimizerConfig(bf16=True, use_distributed_optimizer=True, lr=1e-3))
    # The original compute and moments are intentionally unrelated to the
    # initialized module values, so reloading model values into masters fails.
    value = 0.5 + ps.get_tensor_model_parallel_rank() * 0.00390625
    for entry in adam_parameter_states(stage, driver.optimizer):
        template = torch.full_like(entry.model_parameter, value, dtype=torch.float32)
        initialize_adam_state(entry, step=1200, master=template, exp_avg=template * 0.1, exp_avg_sq=template * 0.01)
    with torch.no_grad():
        for opt in getattr(driver.optimizer, "chained_optimizers", [driver.optimizer]):
            opt._copy_main_params_to_model_params()
        driver.model.start_param_sync(force_sync=True)
    original = {
        entry.name: {
            "compute": entry.select(entry.model_parameter.detach()).clone(),
            "master": entry.master.detach().clone(),
            "exp_avg": entry.state["exp_avg"].clone(),
            "exp_avg_sq": entry.state["exp_avg_sq"].clone(),
        }
        for entry in adam_parameter_states(stage, driver.optimizer)
    }
    driver.iteration = 1200
    checkpoint.save(driver, args.checkpoint_dir, rank_state={"rank": dist.get_rank()})
    with torch.no_grad():
        for entry in adam_parameter_states(stage, driver.optimizer):
            entry.model_parameter.zero_()
            entry.master.zero_()
            entry.state["exp_avg"].zero_()
            entry.state["exp_avg_sq"].zero_()
    rank_state = checkpoint.load(driver, args.checkpoint_dir)
    assert rank_state == {"rank": dist.get_rank()}
    checked = 0
    for entry in adam_parameter_states(stage, driver.optimizer):
        expected = original[entry.name]
        assert torch.equal(entry.select(entry.model_parameter.detach()), expected["compute"]), entry.name
        assert torch.equal(entry.master, expected["master"]), entry.name
        assert torch.equal(entry.state["exp_avg"], expected["exp_avg"]), entry.name
        assert torch.equal(entry.state["exp_avg_sq"], expected["exp_avg_sq"]), entry.name
        assert entry.step == 1200
        checked += 1
    report = {
        "passed": True,
        "rank": dist.get_rank(),
        "tp": 2,
        "pp": args.pp,
        "dp": ps.get_data_parallel_world_size(),
        "fragments_checked": checked,
        "iteration": driver.iteration,
    }
    args.output.with_suffix(f".rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
