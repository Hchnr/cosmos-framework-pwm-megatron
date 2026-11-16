"""Prepare HF parameter owners for Megatron's bounded meta materialization."""

from types import MethodType

import torch

from cosmos_framework.configs.base.defaults.parallelism import PRECISION_TO_TORCH_DTYPE


def prepare_hf_meta_initialization(model, precision):
    """Keep weights on meta until MCore creates their local CUDA buffers.

    HF checkpoint loading runs immediately after wrapping and validates every
    required key. Temporary zero initialization avoids model-family-specific
    reset methods. Tied owners are materialized together because resetting one
    owner with ``to_empty`` would otherwise break the shared Parameter identity.
    """
    dtype = PRECISION_TO_TORCH_DTYPE[precision]
    owners = {}
    for module in model.modules():
        for name, parameter in module._parameters.items():
            if parameter is not None:
                owners.setdefault(id(parameter), []).append((module, name, parameter))
    tied_owners = {id(module) for entries in owners.values() if len(entries) > 1 for module, _, _ in entries}
    replacements = {}
    for entries in owners.values():
        materialize = any(id(module) in tied_owners for module, _, _ in entries)
        for module, name, parameter in entries:
            if id(parameter) not in replacements:
                if materialize:
                    value = torch.zeros(parameter.shape, dtype=dtype, device="cuda")
                    replacements[id(parameter)] = torch.nn.Parameter(value, requires_grad=parameter.requires_grad)
                else:
                    replacements[id(parameter)] = torch.nn.Parameter(
                        parameter.to(dtype=dtype), requires_grad=parameter.requires_grad
                    )
            module._parameters[name] = replacements[id(parameter)]

    @torch.no_grad()
    def reset_parameters(module):
        for parameter in module.parameters(recurse=False):
            parameter.zero_()

    reset_overrides = []
    for module in model.modules():
        if any(p.is_meta for p in module.parameters(recurse=False)):
            reset_overrides.append((module, module.__dict__.get("reset_parameters")))
            module.reset_parameters = MethodType(reset_parameters, module)
        for name, buffer in module._buffers.items():
            if buffer is not None:
                if buffer.is_meta:
                    raise ValueError(
                        f"HF meta initialization requires materialized buffer {type(module).__name__}.{name}"
                    )
                module._buffers[name] = buffer.to("cuda")

    def restore_reset_methods():
        for module, previous in reset_overrides:
            if previous is None:
                module.__dict__.pop("reset_parameters", None)
            else:
                module.reset_parameters = previous

    return restore_reset_methods
