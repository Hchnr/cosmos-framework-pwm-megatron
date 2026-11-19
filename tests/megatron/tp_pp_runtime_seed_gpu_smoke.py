# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Four-rank runtime RNG witness; run old and candidate in fresh processes.

This checks actual MCore CUDA initialization and the real physical RNG
serialization helper. It does not replace a full training checkpoint restart.
"""

import argparse
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.runtime import initialize
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, set_rand_state_dict


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 4:
            raise ValueError("Requires four ranks")
        config = ModelParallelConfig(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            global_packed_microbatches_per_update=12,
            logical_data_parallel_size=4,
            reference_micro_steps_per_update=3,
        )
        groups = initialize(config, seed=42)
        tracker = get_cuda_rng_tracker()
        initial = {
            "python": hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
            "numpy": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
            "torch_cpu": digest(torch.get_rng_state()),
            "cuda": digest(torch.cuda.get_rng_state()),
            "tracker": {name: digest(state) for name, state in sorted(tracker.get_states().items())},
            "tracker_current_name": tracker._current_state_name,
            "tracker_registered_seeds": sorted(tracker.seeds_),
            "tracker_initialized": tracker.is_initialized(),
        }
        physical = get_rand_state_dict()
        tracked = {name: state.clone() for name, state in tracker.get_states().items()}

        def draw():
            result = {
                "python": random.random(),
                "numpy": float(np.random.random()),
                "torch_cpu": float(torch.rand(1).item()),
                "cuda": float(torch.rand(1, device="cuda").item()),
            }
            with tracker.fork():
                result["model_parallel_cuda"] = float(torch.rand(1, device="cuda").item())
            return result

        expected = draw()
        _ = draw()
        set_rand_state_dict(physical)
        tracker.set_states(tracked)
        restored = draw()
        if restored != expected:
            raise AssertionError("Physical and model-parallel RNG states did not restore")
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"rank{rank}.json").open("x") as handle:
            json.dump(
                {"rank": rank, "groups": groups, "initial": initial, "draw": expected, "restore_passed": True},
                handle,
                indent=2,
            )
            handle.write("\n")
        dist.barrier()
    finally:
        if ps.model_parallel_is_initialized():
            ps.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
