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
