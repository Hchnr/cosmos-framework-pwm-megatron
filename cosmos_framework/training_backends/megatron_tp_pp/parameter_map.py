"""Canonical model identities and TP offsets, shared by loading and validation."""

import re
from dataclasses import asdict, dataclass

import torch

from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear


def canonical_local_name(name):
    return name.replace("._orig_mod.", ".").replace("._checkpoint_wrapped_module.", ".")


@dataclass(frozen=True)
class ParameterLayout:
    canonical_name: str
    local_name: str
    shape: tuple[int, ...]
    axis: int | None
    offset: int
    local_shape: tuple[int, ...]
    pipeline_rank: int
    tensor_rank: int
    trainable: bool

    def select(self, tensor):
        if tuple(tensor.shape) != self.shape:
            raise ValueError(f"{self.canonical_name}: expected {self.shape}, got {tensor.shape}")
        if self.axis is None:
            return tensor
        return tensor.narrow(self.axis, self.offset, self.local_shape[self.axis])

    def to_dict(self):
        return asdict(self)


def stage_parameter_manifest(stage, *, decoder_prefix="net.language_model.model."):
    from megatron.core import parallel_state as ps

    result = {}
    tp = ps.get_tensor_model_parallel_world_size()
    rank = ps.get_tensor_model_parallel_rank()
    state_keys = {canonical_local_name(key) for key in stage.state_dict()}
    entries = list(stage.named_parameters()) + list(stage.named_buffers())
    for name, param in entries:
        normalized = canonical_local_name(name)
        if normalized not in state_keys:
            continue  # RoPE / timestep frequencies are reconstructed constants.
        if normalized.startswith("layers."):
            canonical = decoder_prefix + normalized
        elif normalized.startswith(("input_adapter.", "output_adapter.")):
            canonical = "net." + normalized.split(".", 1)[1]
            owner = canonical_parameter_owner(
                canonical, stage.config.num_layers, stage.config.pipeline_model_parallel_size
            )
            if owner != ps.get_pipeline_model_parallel_rank():
                raise ValueError(f"Adapter parameter has the wrong stage owner: {canonical}")
        else:
            raise ValueError(f"Missing canonical parameter mapping: {name}")
        parent_path, _, leaf = name.rpartition(".")
        parent = stage.get_submodule(parent_path)
        axis = getattr(param, "partition_dim", None) if getattr(param, "tensor_model_parallel", False) else None
        if isinstance(parent, TensorParallelLinear):
            axis = parent.axis if leaf == "weight" else (0 if parent.axis == 0 else None)
        shape = list(param.shape)
        offset = 0
        if axis is not None:
            offset = rank * shape[axis]
            shape[axis] *= tp
        result[name] = ParameterLayout(
            canonical,
            name,
            tuple(shape),
            axis,
            offset,
            tuple(param.shape),
            ps.get_pipeline_model_parallel_rank(),
            rank,
            param.requires_grad,
        )
    return result


def canonical_parameter_owner(name, num_layers, pp_size):
    """Fail closed for unknown keys, instead of losing buffers or duplicating heads."""
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match:
        index = int(match[1])
        if not 0 <= index < num_layers:
            raise ValueError(f"Invalid global layer: {name}")
        # Inverse of floor boundaries, including uneven partitions.
        return ((index + 1) * pp_size - 1) // num_layers
    output_names = (".norm.", ".norm_moe_gen.", ".norm_moe_gen_action.", ".lm_head.", ".llm2vae.", ".llm2action")
    if any(key in name for key in output_names):
        return pp_size - 1
    input_names = (
        ".embed_tokens.",
        ".time_embedder.",
        ".action_time_embedder.",
        ".vae2llm.",
        ".action2llm",
        ".action_modality_embed",
        ".action_normalization.",
    )
    if any(key in name for key in input_names):
        return 0
    raise ValueError(f"No canonical stage ownership registered for {name}")


@torch.no_grad()
def load_canonical_stage(stage, state, manifest):
    expected = {layout.canonical_name for layout in manifest.values()}
    missing = expected - state.keys()
    if missing:
        raise ValueError(f"Missing canonical parameters: {sorted(missing)}")
    parameters = dict(stage.named_parameters()) | dict(stage.named_buffers())
    for name, layout in manifest.items():
        parameters[name].copy_(layout.select(state[layout.canonical_name]))
