"""Bounded CUDA materialization for two-tower Omni models."""

import inspect
import math

import torch

from cosmos_framework.configs.base.defaults.parallelism import PRECISION_TO_TORCH_DTYPE
from cosmos_framework.training_backends.meta_init import prepare_hf_meta_initialization
from cosmos_framework.utils.generator.lora import LoraInjectedLinear


@torch.no_grad()
def prepare_omni_meta_initialization(net, dtype, *, ema=False):
    """Initialize small non-language heads and buffers, keeping the LM on meta."""
    language = net.language_model
    if any(
        getattr(module, "noisy_gating", False)
        or getattr(module, "shared_expert", None) is not None
        or getattr(getattr(module, "cosine_router", None), "enabled", False)
        or getattr(getattr(module, "aux_loss_free_load_balancing_config", None), "enabled", False)
        for module in language.modules()
    ):
        raise ValueError(
            "Omni meta initialization has not qualified generation-only router/shared-expert initialization"
        )

    def move_heads(module):
        if module is language:
            return
        for name, parameter in module._parameters.items():
            if parameter is not None:
                if parameter.is_meta:
                    raise ValueError(f"Omni meta initialization requires CPU head {type(module).__name__}.{name}")
                parameter.data = parameter.data.to(device="cuda", dtype=dtype)
        for name, buffer in module._buffers.items():
            if buffer is not None:
                if buffer.is_meta:
                    raise ValueError(f"Omni meta initialization lost head buffer {type(module).__name__}.{name}")
                module._buffers[name] = buffer.to("cuda")
        for child in module.children():
            move_heads(child)

    if not ema:
        move_heads(net)
    # Root heads retain their original policies; meta LM parameter init is a
    # no-op while RoPE/Fourier buffers are rebuilt on CUDA in their original dtype.
    net.init_weights(buffer_device=torch.device("cuda"))
    if ema:
        return None

    def reset_owner(module):
        initializer = getattr(module, "init_weights", None)
        if initializer is not None and "buffer_device" in inspect.signature(initializer).parameters:
            initializer(buffer_device=torch.device("cuda"))
        else:
            language._init_weights(module, buffer_device=torch.device("cuda"))

    precision = next(name for name, value in PRECISION_TO_TORCH_DTYPE.items() if value == dtype)
    return prepare_hf_meta_initialization(net, precision, reset_initializer=reset_owner)


@torch.no_grad()
def finish_omni_meta_initialization(runtime, *, lora):
    """Seed adapters with full small matrices before slicing uneven local views."""
    net = runtime.fsdp.module
    if lora:
        runtime.prepare_state_dict(dict(net.named_parameters()))
        for module in net.modules():
            if not isinstance(module, LoraInjectedLinear):
                continue
            parameter = module.lora_A.weight
            full = torch.empty(parameter.shape, dtype=parameter.dtype, device="cuda")
            torch.nn.init.kaiming_uniform_(full, a=math.sqrt(5))
            chunks = getattr(parameter._local_tensor, "__create_chunk_list__", None)
            if chunks is None:
                slices = tuple(slice(None) for _ in parameter.shape)
            else:
                chunk = chunks()[0]
                slices = tuple(slice(o, o + n) for o, n in zip(chunk.offsets, chunk.sizes, strict=True))
            parameter.to_local().copy_(full[slices])
            module.lora_B.weight.to_local().zero_()
    runtime.finalize_loaded_state()
    net.__dict__.pop("_cosmos_megatron_meta_pending", None)
