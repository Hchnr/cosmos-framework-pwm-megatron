# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in native FP32 gradient-norm arithmetic for physical DP=1.

Only parameter identities and optimizer-field presence come from the canonical
manifest. No reference gradients, norms, parameter values or optimizer state
are read. The unsharded order is reconstructed from the native meta model.
"""

import hashlib
import json
import math
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist


def requested(flag, *, dtype, dp_size, clip_grad):
    if flag not in ("0", "1"):
        raise ValueError("COSMOS_TP_PP_NATIVE_FP32_NORM must be 0 or 1")
    if flag == "0":
        return False
    if dtype != torch.float32 or dp_size != 1 or not math.isfinite(clip_grad) or clip_grad <= 0:
        raise ValueError("Native FP32 norm requires FP32, physical DP=1 and finite positive gradient clipping")
    return True


def supported_optimizer_config(config):
    return (
        config.optimizer == "adam"
        and not config.fp16
        and not config.bf16
        and not config.use_precision_aware_optimizer
        and config.loss_scale is None
        and config.grad_norm_skip_threshold == float("inf")
        and config.use_distributed_optimizer
        and math.isfinite(config.clip_grad)
        and config.clip_grad > 0
    )


def agree_requested(flag, *, dtype, dp_size, optimizer_config, device):
    """Negotiate eligibility once across all ranks, including the disabled path."""
    invalid, enabled = 0, 0
    try:
        enabled = int(requested(flag, dtype=dtype, dp_size=dp_size, clip_grad=optimizer_config.clip_grad))
        if enabled and not supported_optimizer_config(optimizer_config):
            invalid = 1
    except (AttributeError, TypeError, ValueError, OverflowError):
        invalid = 1
    counts = torch.tensor([invalid, enabled], dtype=torch.int64, device=device)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    invalid_count, enabled_count = counts.tolist()
    if invalid_count:
        raise ValueError("Native FP32 norm configuration is invalid or uses an unsupported optimizer")
    if enabled_count not in (0, dist.get_world_size()):
        raise ValueError("Native FP32 norm flag must agree on every training rank")
    return bool(enabled_count)


def ordered_trainable_names(model, manifest):
    """Native state_dict registration order, filtered only by structural metadata."""
    entries = manifest["parameters"]
    parameters = dict(model.named_parameters())
    names = [
        "net." + name
        for name in model.state_dict()
        if name in parameters and entries["net." + name]["optimizer_fields"]
    ]
    expected = {name for name, entry in entries.items() if entry["optimizer_fields"]}
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("Native meta-model optimizer parameter coverage differs from canonical metadata")
    return names


@contextmanager
def preserve_training_rng():
    """Preserve CPU, Python, NumPy, current CUDA and initialized MCore RNGs."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    tracker, tracker_states, tracker_seeds, tracker_current = None, None, None, None
    if devices:
        from megatron.core.tensor_parallel import random as mcore_random

        if mcore_random._CUDA_RNG_STATE_TRACKER_INITIALIZED:
            tracker = mcore_random.get_cuda_rng_tracker()
            if not isinstance(tracker, mcore_random.CudaRNGStatesTracker) or not tracker.is_initialized():
                raise ValueError("Native FP32 norm requires the initialized MCore RNG tracker")
            tracker_states = {
                name: state.clone() if isinstance(state, torch.Tensor) else state.clone_state()
                for name, state in tracker.get_states().items()
            }
            tracker_seeds = tracker.seeds_.copy()
            tracker_current = tracker._current_state_name
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if tracker is not None:
            tracker.set_states(tracker_states)
            tracker.seeds_ = tracker_seeds
            tracker._current_state_name = tracker_current


def derive_native_parameter_order(model_config, manifest):
    with preserve_training_rng():
        return _derive_native_parameter_order(model_config, manifest)


def _derive_native_parameter_order(model_config, manifest):
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
        Cosmos3VFMNetwork,
    )
    from cosmos_framework.model.generator.omni_mot_model import (
        OmniMoTModel,
        _prepare_full_model_instance,
    )
    from cosmos_framework.utils.lazy_config import instantiate

    instance, _, _ = _prepare_full_model_instance(
        model_config.vlm_config.model_instance,
        load_vlm_visual=False,
        include_generation_tower=True,
    )
    language = instantiate(instance["config"])
    view = OmniMoTModel.__new__(OmniMoTModel)
    view.__dict__.update(config=model_config, tokenizer_vision_gen=None)
    network = view.build_network_config(language)
    with torch.device("meta"):
        model = Cosmos3VFMNetwork(instantiate(instance), network)
    return ordered_trainable_names(model, manifest)


@torch.no_grad()
def gradient_norm(driver, order):
    from megatron.core import parallel_state as ps

    from .parameter_map import stage_parameter_manifest

    if (
        driver.topology.data_parallel_size(dist.get_world_size()) != 1
        or driver.stage.config.params_dtype != torch.float32
    ):
        raise ValueError("Native FP32 norm is only supported with FP32 and physical DP=1")
    indices = {name: index for index, name in enumerate(order)}
    duplicate_order = len(indices) != len(order)
    all_parameters = list(driver.stage.named_parameters())
    stage_parameters = [(name, p) for name, p in all_parameters if p.requires_grad]
    if all_parameters:
        device = all_parameters[0][1].device
    else:
        device = (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_initialized() else torch.device("cpu")
        )
    norms = torch.zeros(len(order), dtype=torch.float32, device=device)
    used = torch.zeros(len(order), dtype=torch.int32, device=device)
    owners = torch.zeros(len(order), dtype=torch.int32, device=device)
    try:
        manifest = stage_parameter_manifest(driver.stage)
        manifest_error = False
    except Exception:
        # Other model-parallel ranks must leave preflight as well; raising
        # locally here would strand peers in the first collective.
        manifest, manifest_error = {}, True
    tp_group = ps.get_tensor_model_parallel_group()
    tp_size = ps.get_tensor_model_parallel_world_size()
    owner = ps.get_tensor_model_parallel_rank() == 0
    group = ps.get_model_parallel_group()
    # Coordinate unsupported local layouts before any rank enters a different
    # number of TP all-gathers. Otherwise a bad stage could fail locally while a
    # peer waits forever in the later model-parallel collective.
    invalid = torch.zeros(1, dtype=torch.int32, device=device)
    if not stage_parameters or duplicate_order or manifest_error:
        invalid.fill_(1)
    optimizer = getattr(driver, "optimizer", None)
    optimizer_parameters = [
        parameter for child in getattr(optimizer, "chained_optimizers", ()) for parameter in child.get_parameters()
    ]
    if any(getattr(parameter, "grad_norm_group", None) is not None for parameter in optimizer_parameters):
        # MCore clips separate groups independently. This candidate computes a
        # single native norm, so reject that configuration on every MP rank.
        invalid.fill_(1)
    for name, parameter in stage_parameters:
        spec = manifest.get(name)
        gradient = getattr(parameter, "main_grad", None)
        if (
            spec is None
            or spec.canonical_name not in indices
            or getattr(parameter, "grad_norm_group", None) is not None
            or (
                not isinstance(gradient, torch.Tensor)
                or gradient.dtype != torch.float32
                or gradient.device != device
                or gradient.shape != parameter.shape
                or (spec.axis is not None and not 0 <= spec.axis < gradient.ndim)
            )
        ):
            invalid.fill_(1)
        elif owner:
            owners[indices[spec.canonical_name]] += 1
    dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=group)
    if bool(invalid.item()):
        raise ValueError("Native norm encountered an unsupported stage, gradient, parameter identity or shard axis")
    # Every rank must interpret the vector indices identically. A mismatch
    # could otherwise make unrelated parameters occupy the same slots.
    digest = hashlib.sha256(json.dumps(order, ensure_ascii=True, separators=(",", ":")).encode()).digest()
    digest_min = torch.tensor(list(digest), dtype=torch.uint8, device=device)
    digest_max = digest_min.clone()
    dist.all_reduce(digest_min, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(digest_max, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(digest_min, digest_max):
        raise ValueError("Native norm parameter registration order differs across model-parallel ranks")
    dist.all_reduce(owners, op=dist.ReduceOp.SUM, group=group)
    if not bool(torch.all(owners == 1)):
        raise ValueError("Native norm requires exactly one model-parallel owner per parameter")
    # The per-parameter all-gathers below follow local stage registration
    # order.  Matching the global native order is not enough: TP peers must
    # agree on the *local* order, gather axes and tensor shapes as well.
    # Otherwise NCCL can hang, or equal-sized but different parameters can
    # be silently combined.  Propagate any TP-group mismatch to the entire
    # model-parallel group before any peer enters a data all-gather.
    layout = [
        (manifest[name].canonical_name, manifest[name].axis, list(parameter.main_grad.shape))
        for name, parameter in stage_parameters
    ]
    layout_digest = hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).digest()
    layout_min = torch.tensor(list(layout_digest), dtype=torch.uint8, device=device)
    layout_max = layout_min.clone()
    dist.all_reduce(layout_min, op=dist.ReduceOp.MIN, group=tp_group)
    dist.all_reduce(layout_max, op=dist.ReduceOp.MAX, group=tp_group)
    layout_invalid = torch.tensor([not torch.equal(layout_min, layout_max)], dtype=torch.int32, device=device)
    dist.all_reduce(layout_invalid, op=dist.ReduceOp.MAX, group=group)
    if bool(layout_invalid.item()):
        raise ValueError("Native norm TP ranks have different parameter order, axes or gradient shapes")
    for name, parameter in stage_parameters:
        spec = manifest[name]
        index = indices[spec.canonical_name]
        gradient = parameter.main_grad.detach()
        if gradient.dtype != torch.float32 or gradient.device != device:
            raise ValueError("Native norm requires FP32 gradients on one device")
        if spec.axis is not None and tp_size > 1:
            pieces = [torch.empty_like(gradient) for _ in range(tp_size)]
            dist.all_gather(pieces, gradient.contiguous(), group=tp_group)
            gradient = torch.cat(pieces, dim=spec.axis)
        if owner:
            norms[index] = torch._foreach_norm([gradient], 2)[0]
        used[index] = int(parameter in driver._globally_used)
    dist.all_reduce(norms, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(used, op=dist.ReduceOp.MAX, group=group)
    driver._native_fp32_norm_calls += 1
    return torch.linalg.vector_norm(norms[used.bool()], 2).item()
