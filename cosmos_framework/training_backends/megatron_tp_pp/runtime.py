# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""MCore 0.20 process groups and configuration; no FSDP mesh assumptions."""

import importlib.metadata
import os

import torch
import torch.distributed as dist


def initialize(config, *, seed=42):
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if importlib.metadata.version("megatron-core") != "0.20.0":
        raise RuntimeError("megatron_tp_pp requires the audited Megatron-Core 0.20.0 API")
    if not dist.is_initialized():
        raise RuntimeError("Initialize torch.distributed and set the local CUDA device first")
    config.data_parallel_size(dist.get_world_size())
    if ps.model_parallel_is_initialized():
        raise RuntimeError("Model parallel groups are already initialized")
    ps.initialize_model_parallel(
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        context_parallel_size=1,
        order="tp-cp-ep-dp-pp",
        distributed_timeout_minutes=max(1, int(os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")) // 60),
    )
    model_parallel_cuda_manual_seed(seed)
    return {
        "tp": dist.get_process_group_ranks(ps.get_tensor_model_parallel_group()),
        "pp": dist.get_process_group_ranks(ps.get_pipeline_model_parallel_group()),
        "dp": dist.get_process_group_ranks(ps.get_data_parallel_group()),
        "microbatches": config.microbatches(dist.get_world_size()),
    }


def transformer_config(topology, *, num_layers, hidden_size, num_attention_heads, dtype=torch.float32, **kwargs):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        tensor_model_parallel_size=topology.tensor_model_parallel_size,
        pipeline_model_parallel_size=topology.pipeline_model_parallel_size,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        sequence_parallel=False,
        perform_initialization=False,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        deallocate_pipeline_outputs=True,
        batch_p2p_comm=False,
        overlap_p2p_comm=False,
        calculate_per_token_loss=False,
        **kwargs,
    )
