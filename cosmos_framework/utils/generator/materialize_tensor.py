"""Logical parameter materialization for regular and Megatron uneven DTensors."""

import torch
from torch.distributed.tensor import DTensor


def materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, DTensor):
        return tensor
    megatron_model = getattr(tensor, "_megatron_fsdp_model", None)
    if megatron_model is not None:
        from cosmos_framework.training_backends.checkpoint_metadata import prepare_uneven_state

        prepare_uneven_state({"tensor": tensor}, megatron_model.dist_index.get_fsdp_group())
    if hasattr(tensor._local_tensor, "__create_chunk_list__"):
        from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import uneven_dtensor_to_full_tensor

        return uneven_dtensor_to_full_tensor(tensor)
    return tensor.full_tensor()


@torch.no_grad()
def copy_model_parameter(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy logical tower parameters even when their local owners differ."""
    owner = getattr(destination, "_megatron_fsdp_model", None)
    chunks = (
        getattr(destination._local_tensor, "__create_chunk_list__", None) if isinstance(destination, DTensor) else None
    )
    # state_dict() detaches the Parameter's owner attribute but retains its
    # prepared uneven chunk metadata. Those views need logical copies too.
    if not isinstance(destination, DTensor) or (owner is None and chunks is None):
        destination.data.copy_(source.data)
        return
    if owner is not None:
        from cosmos_framework.training_backends.checkpoint_metadata import prepare_uneven_state

        prepare_uneven_state({"source": source, "destination": destination}, owner.dist_index.get_fsdp_group())
    full = materialize_tensor(source)
    chunks = getattr(destination._local_tensor, "__create_chunk_list__", None)
    if chunks is None:
        slices = tuple(slice(None) for _ in destination.shape)
    else:
        chunk = chunks()[0]
        slices = tuple(slice(o, o + n) for o, n in zip(chunk.offsets, chunk.sizes, strict=True))
    destination.to_local().copy_(full[slices])
