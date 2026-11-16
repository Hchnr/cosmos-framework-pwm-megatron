"""Carry uneven-shard DCP metadata through CPU staging and process queues.

Tensor copies discard Megatron's local-tensor planning closures. Carry only
plain metadata across the queue, then rebuild the closures in the writer.
This module intentionally does not import Megatron or initialize CUDA.
"""

import torch.distributed as dist
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorProperties
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType
from torch.distributed.tensor import DTensor, Shard


def prepare_uneven_state(state, shard_group):
    """Batch metadata exchange, including ranks with missing empty optimizer state.

    Per-tensor collectives are unsafe: Adam initializes no state for empty local
    parameters, so different ranks may have different state keys. One exchange
    per state dictionary also avoids thousands of small object collectives.
    This backend currently uses TP=CP=1 and a single parameter shard dimension.
    """
    entries = {}

    def visit(value, path=()):
        if isinstance(value, DTensor):
            shards = [p for p in value.placements if isinstance(p, Shard)]
            if not shards:
                return
            if len(shards) != 1:
                raise ValueError("Cosmos Megatron checkpointing currently requires one shard dimension")
            entries[path] = (value, shards[0].dim)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + (key,))
        elif isinstance(value, (list, tuple)):
            for key, child in enumerate(value):
                visit(child, path + (key,))

    visit(state)
    descriptors = {path: tuple(tensor.to_local().shape) for path, (tensor, _) in entries.items()}
    gathered = [None] * dist.get_world_size(shard_group)
    dist.all_gather_object(gathered, descriptors, group=shard_group)
    rank = dist.get_rank(shard_group)
    metadata = []
    for path, (tensor, dim) in entries.items():
        offsets = [0] * tensor.ndim
        offsets[dim] = sum(peer[path][dim] if path in peer else 0 for peer in gathered[:rank])
        metadata.append((path, [ChunkStorageMetadata(offsets=tuple(offsets), sizes=tensor.to_local().shape)]))
    restore_chunk_metadata(state, metadata)
    return state


def extract_chunk_metadata(state, path=()):
    result = []
    if isinstance(state, DTensor):
        chunks = getattr(state._local_tensor, "__create_chunk_list__", None)
        if chunks is not None:
            result.append((path, chunks()))
    elif isinstance(state, dict):
        for key, value in state.items():
            result.extend(extract_chunk_metadata(value, path + (key,)))
    elif isinstance(state, (tuple, list)):
        for index, value in enumerate(state):
            result.extend(extract_chunk_metadata(value, path + (index,)))
    return result


def restore_chunk_metadata(state, metadata):
    def install(tensor, chunks):
        def create_write_items(fqn, value):
            if value.to_local().numel() == 0:
                return []
            return [
                WriteItem(
                    type=WriteItemType.SHARD,
                    index=MetadataIndex(fqn, chunk.offsets),
                    tensor_data=TensorWriteData(
                        chunk=chunk,
                        properties=TensorProperties.create_from_tensor(value.to_local()),
                        size=value.size(),
                    ),
                )
                for chunk in chunks
            ]

        tensor._local_tensor.__create_chunk_list__ = lambda: chunks
        tensor._local_tensor.__create_write_items__ = create_write_items

    for path, chunks in metadata:
        tensor = state
        for key in path:
            tensor = tensor[key]
        install(tensor, chunks)
