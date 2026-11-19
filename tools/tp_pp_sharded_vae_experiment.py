# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Isolated frozen VAE item sharding primitive; not wired into training.

All ranks call ``sharded_vision_encode`` in the same order, on the same group.
The owner supplies the exact raw vision-item sequence that native
``_encode_vision_x0_tokens`` would have processed. Each worker calls the real
bound ``_encode_vision_item`` through ``native_item_encoder``; owner receives
latents in original order and continues the *unchanged* native preparation,
noise sampling, packing and loss path. Inference-only prefix encoding is out of
scope. Never use this prototype as a correctness or throughput claim without the
real two-/four-GPU tests, including RNG, masks, losses and checkpoint restart.
"""

import traceback
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class VisionItem:
    tensor: torch.Tensor
    num_views: int = 1
    frames_per_view: int | None = None


def assign_items(costs, workers):
    """Greedily balance static pixel counts; identity and result order stay fixed."""
    if workers < 1 or any(isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0 for cost in costs):
        raise ValueError("Expected a positive worker count and positive integer item costs")
    loads, owners = [0] * workers, [0] * len(costs)
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        worker = min(range(workers), key=lambda i: (loads[i], i))
        owners[index] = worker
        loads[worker] += costs[index]
    return owners


def native_item_encoder(preparation):
    """Adapt the production no-grad item/view logic, without touching RNG/noising."""

    def encode(item, num_views, frames_per_view):
        return preparation._encode_vision_item(item, num_views=num_views, frames_per_view=frames_per_view)

    return encode


def native_training_items(
    raw_state_vision,
    num_vision_items_per_sample,
    vision_condition_indexes,
    num_views_per_vision_item=None,
    frames_per_vision_item=None,
):
    """Map the native full-training VAE call to independently encodable items.

    The input list is already flattened in native camera/control order; the
    sample-grouping metadata is validated here and otherwise left untouched.
    Inference-only prefix encoding has a different contract and is rejected.
    """
    if vision_condition_indexes is not None:
        raise ValueError("Only native full-encode training is supported")
    count = len(raw_state_vision)
    if num_vision_items_per_sample is not None and (
        any(n <= 0 for n in num_vision_items_per_sample) or sum(num_vision_items_per_sample) != count
    ):
        raise ValueError("Flattened vision items disagree with sample grouping")
    if (num_views_per_vision_item is None) != (frames_per_vision_item is None):
        raise ValueError("Both multiview metadata lists must be provided together")
    if num_views_per_vision_item is None:
        num_views_per_vision_item, frames_per_vision_item = [1] * count, [None] * count
    if len(num_views_per_vision_item) != count or len(frames_per_vision_item) != count:
        raise ValueError("Multiview metadata disagrees with flattened vision items")
    return [
        VisionItem(tensor, views, frames)
        for tensor, views, frames in zip(
            raw_state_vision, num_views_per_vision_item, frames_per_vision_item, strict=True
        )
    ]


def _dtype(name):
    result = getattr(torch, name, None)
    if not isinstance(result, torch.dtype):
        raise ValueError(f"Unsupported tensor dtype {name!r}")
    return result


def _describe(items, size):
    if not isinstance(items, (tuple, list)) or not items:
        raise ValueError("The owner must supply a nonempty ordered list of vision items")
    descriptors = []
    for item in items:
        if not isinstance(item, VisionItem) or item.tensor.ndim not in (4, 5) or item.tensor.requires_grad:
            raise ValueError("Each item must be a detached 4-D/5-D VisionItem")
        if item.num_views < 1 or (item.frames_per_view is None and item.num_views != 1):
            raise ValueError("Invalid camera view metadata")
        if item.frames_per_view is not None and (
            item.frames_per_view <= 0 or item.tensor.shape[-3] != item.num_views * item.frames_per_view
        ):
            raise ValueError("Camera metadata disagrees with raw temporal dimension")
        descriptors.append(
            {
                "shape": tuple(item.tensor.shape),
                "dtype": str(item.tensor.dtype).removeprefix("torch."),
                "num_views": item.num_views,
                "frames_per_view": item.frames_per_view,
            }
        )
    assignment = assign_items([item.tensor.numel() for item in items], size)
    return [{**item, "worker": worker} for item, worker in zip(descriptors, assignment, strict=True)]


def sharded_vision_encode(items, encoder, group, *, owner_rank=0, device=None):
    """Scatter video items, encode once each, and gather in original order.

    ``owner_rank`` is a rank *within* ``group``; peers pass items=None. On the
    owner, ``items`` can also be a zero-argument item factory so failures in
    native input-to-item conversion are distributed before any P2P transfers. The
    metadata object collective and P2P messages all use the provided group.
    Workers' exceptions propagate collectively before any result transfers.
    Transport/process failure itself requires a distributed timeout/restart.
    This primitive does not advance, replace, or checkpoint owner noising RNG.
    """
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if not 0 <= owner_rank < size:
        raise ValueError("Invalid owner rank within the process group")
    source = dist.get_global_rank(group, owner_rank)
    owner = rank == owner_rank
    if device is None:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend(group) == "nccl"
            else torch.device("cpu")
        )
    device = torch.device(device)
    envelope = [None]
    if owner:
        try:
            if callable(items):
                items = items()
            envelope[0] = {"descriptors": _describe(items, size), "error": None}
        except Exception:
            envelope[0] = {"descriptors": None, "error": traceback.format_exc()}
    dist.broadcast_object_list(envelope, src=source, group=group, device=device)
    if envelope[0]["error"] is not None:
        raise RuntimeError("Owner rejected VAE sharding input:\n" + envelope[0]["error"])
    descriptors = envelope[0]["descriptors"]
    local_items = {}
    # Each rank executes the same ordered protocol. This implementation favors
    # correctness and bounded metadata over overlapping PCIe/NVLink transfers.
    for index, desc in enumerate(descriptors):
        worker = desc["worker"]
        peer = dist.get_global_rank(group, worker)
        if owner and worker == owner_rank:
            local_items[index] = items[index].tensor.to(device=device).contiguous()
        elif owner:
            dist.send(items[index].tensor.to(device=device).contiguous(), dst=peer, group=group)
        elif worker == rank:
            value = torch.empty(desc["shape"], dtype=_dtype(desc["dtype"]), device=device)
            dist.recv(value, src=source, group=group)
            local_items[index] = value
    local_results, local_error = {}, None
    try:
        with torch.no_grad():
            for index, value in local_items.items():
                desc = descriptors[index]
                result = encoder(value, desc["num_views"], desc["frames_per_view"])
                if not isinstance(result, torch.Tensor) or result.device != device or result.requires_grad:
                    raise ValueError("Encoder must return a detached tensor on the group's execution device")
                local_results[index] = result.contiguous()
    except Exception:
        local_error = traceback.format_exc()
    reports = [None] * size
    local_meta = {
        "rank": rank,
        "error": local_error,
        "outputs": {
            index: {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
            for index, tensor in local_results.items()
        },
    }
    dist.all_gather_object(reports, local_meta, group=group)
    failures = [f"rank {part['rank']}:\n{part['error']}" for part in reports if part["error"]]
    if failures:
        raise RuntimeError("Sharded VAE worker failed:\n" + "\n".join(failures))
    result = [None] * len(descriptors) if owner else None
    for index, desc in enumerate(descriptors):
        worker = desc["worker"]
        peer = dist.get_global_rank(group, worker)
        metadata = reports[worker]["outputs"].get(index)
        if metadata is None:
            raise RuntimeError(f"Worker {worker} omitted item {index}")
        if owner and worker == owner_rank:
            result[index] = local_results[index]
        elif owner:
            value = torch.empty(metadata["shape"], dtype=_dtype(metadata["dtype"]), device=device)
            dist.recv(value, src=peer, group=group)
            result[index] = value
        elif worker == rank:
            dist.send(local_results[index], dst=source, group=group)
    return result


def sharded_native_training_encode(
    preparation,
    group,
    *,
    raw_state_vision=None,
    num_vision_items_per_sample=None,
    vision_condition_indexes=None,
    num_views_per_vision_item=None,
    frames_per_vision_item=None,
    owner_rank=0,
    device=None,
):
    """Call at native full-encode boundary; only owner supplies raw metadata.

    Coordination at the training call site and every-rank tokenizer loading are
    deliberately left to a future opt-in integration and are not installed by
    this helper. After the collective returns, owner resumes native noising.
    """
    if dist.get_rank(group) == owner_rank:

        def items():
            return native_training_items(
                raw_state_vision,
                num_vision_items_per_sample,
                vision_condition_indexes,
                num_views_per_vision_item,
                frames_per_vision_item,
            )
    else:
        items = None
    return sharded_vision_encode(items, native_item_encoder(preparation), group, owner_rank=owner_rank, device=device)
