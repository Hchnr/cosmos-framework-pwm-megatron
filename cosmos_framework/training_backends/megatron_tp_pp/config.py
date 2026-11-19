# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Topology and logical batch contracts, independent of CUDA/Megatron imports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParallelConfig:
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 2
    global_packed_microbatches_per_update: int = 576
    logical_data_parallel_size: int = 192
    reference_micro_steps_per_update: int = 3
    sequence_parallel: bool = False
    pipeline_schedule: str = "1f1b"
    use_distributed_optimizer: bool = True
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False

    def __post_init__(self):
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "global_packed_microbatches_per_update",
            "logical_data_parallel_size",
            "reference_micro_steps_per_update",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.tensor_model_parallel_size not in (1, 2, 4, 8):
            raise ValueError("TP must be 1, 2, 4 or 8 (KV replication is unsupported)")
        if self.sequence_parallel or self.pipeline_schedule != "1f1b":
            raise ValueError("Only non-interleaved 1F1B without sequence parallelism is supported")
        if self.global_packed_microbatches_per_update != (
            self.logical_data_parallel_size * self.reference_micro_steps_per_update
        ):
            raise ValueError("Logical slots must equal logical DP * reference micro steps")

    def data_parallel_size(self, world_size):
        model_size = self.tensor_model_parallel_size * self.pipeline_model_parallel_size
        if world_size < model_size or world_size % model_size:
            raise ValueError(f"WORLD={world_size} must be divisible by TP*PP={model_size}")
        dp = world_size // model_size
        if self.global_packed_microbatches_per_update % dp:
            raise ValueError("Logical packed microbatches must be divisible by DP")
        return dp

    def microbatches(self, world_size):
        return self.global_packed_microbatches_per_update // self.data_parallel_size(world_size)


def layer_range(num_layers, pipeline_size, pipeline_rank):
    if not 0 <= pipeline_rank < pipeline_size or num_layers < pipeline_size:
        raise ValueError("Each pipeline stage must own at least one layer")
    return range(num_layers * pipeline_rank // pipeline_size, num_layers * (pipeline_rank + 1) // pipeline_size)
