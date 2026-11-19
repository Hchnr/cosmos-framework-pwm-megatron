# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read canonical DCP tensor slices without constructing the source model."""

import os
from pathlib import Path

import torch
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, TensorStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import _create_read_items, create_read_items_for_chunk_list
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

from cosmos_framework.utils import log


def release_file_cache(paths):
    """Read-only advice for completed import/checkpoint files already copied to CUDA."""
    if not hasattr(os, "posix_fadvise"):
        return
    for path in paths:
        try:
            with Path(path).open("rb") as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as error:
            log.warning(f"Could not advise checkpoint read cache for {path}: {error}")


class _SliceLoadPlanner(DefaultLoadPlanner):
    def create_local_plan(self):
        requests = []
        for name, value in self.state_dict.items():
            metadata = self.metadata.state_dict_metadata[name]
            if isinstance(metadata, TensorStorageMetadata):
                requests.extend(create_read_items_for_chunk_list(name, metadata, value.__create_chunk_list__()))
            else:
                requests.extend(_create_read_items(name, metadata, value))
        return LoadPlan(requests)

    def lookup_tensor(self, index):
        return self.state_dict[index.fqn]


class CanonicalDCPReader:
    """Independent, bounded CPU reads; never participates in training collectives."""

    def __init__(self, path):
        self.path = Path(path)
        self.reader = FileSystemReader(self.path)
        metadata = self.reader.read_metadata()
        self.metadata = metadata.state_dict_metadata
        # Metadata can contain thousands of DCP chunks. Reuse the audited
        # snapshot across bounded reads instead of unpickling it per tensor.
        self.reader.read_metadata = lambda: metadata

    def read(self, name, *, axis=None, rank=0, size=1):
        metadata = self.metadata[name]
        value = None
        if isinstance(metadata, TensorStorageMetadata):
            shape = list(metadata.size)
            offset = [0] * len(shape)
            if axis is not None:
                if shape[axis] % size:
                    raise ValueError(f"TP does not divide {name} on axis {axis}")
                shape[axis] //= size
                offset[axis] = rank * shape[axis]
            value = torch.empty(shape, dtype=metadata.properties.dtype)
            # DCP's read planner uses the global chunk offset but writes into
            # local tensor offsets; no full-size placeholder is allocated.
            value.__create_chunk_list__ = lambda: [ChunkStorageMetadata(tuple(offset), tuple(shape))]
        state = {name: value}
        planner = _SliceLoadPlanner(flatten_state_dict=False, allow_partial_load=True)
        _load_state_dict(
            state,
            storage_reader=self.reader,
            no_dist=True,
            planner=planner,
        )
        result = planner.state_dict[name]
        if isinstance(result, torch.Tensor) and hasattr(result, "__create_chunk_list__"):
            del result.__create_chunk_list__
        return result

    def read_many(self, names):
        """Read a bounded set of complete leaves with one DCP planning pass.

        Intended for small rank RNG snapshots, not bulk model tensors.
        """
        state = {}
        for name in names:
            metadata = self.metadata[name]
            state[name] = (
                torch.empty(metadata.size, dtype=metadata.properties.dtype)
                if isinstance(metadata, TensorStorageMetadata)
                else None
            )
        planner = DefaultLoadPlanner(flatten_state_dict=False, allow_partial_load=True)
        _load_state_dict(state, storage_reader=self.reader, no_dist=True, planner=planner)
        return planner.state_dict


def canonical_tp_axis(name):
    """Independent Q/K/V and gate/up layouts; no fused-QKV chunk ambiguity."""
    if name.endswith((".model.embed_tokens.weight", ".lm_head.weight")):
        return 0
    if ".layers." not in name:
        # Small I/O modules start replicated; vocabulary matrices are sharded.
        return None
    leaf = name.rsplit(".", 2)[-2].removesuffix("_moe_gen")
    field = name.rsplit(".", 1)[-1]
    if leaf in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
        return 0
    if leaf in ("o_proj", "down_proj") and field == "weight":
        return 1
    return None
