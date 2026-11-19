"""Native GPU leaf parameters with CPU-owned moments and streamed frozen regions."""

import torch


def place_gpu_trainables(reference):
    """Keep native trainable leaves on CUDA; stream frozen regions per layer.

    Forward saved-tensor offload must be active in the caller so backward does
    not retain every streamed frozen matrix on the GPU at once.
    """
    frozen = {id(parameter) for parameter in reference.parameters() if not parameter.requires_grad}
    for parameter in reference.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.cuda()
    for module in reference.modules():
        for name, value in module.named_buffers(recurse=False):
            module._buffers[name] = value.cuda()
    regions = list(reference.language_model.model.layers)
    covered = {id(p) for region in regions for p in region.parameters() if not p.requires_grad}
    for name, parameter in reference.named_parameters():
        if not parameter.requires_grad and id(parameter) not in covered:
            parent, _, _ = name.rpartition(".")
            region = reference.get_submodule(parent)
            regions.append(region)
            covered.update(id(p) for p in region.parameters() if not p.requires_grad)
    if covered != frozen:
        raise ValueError("Frozen-region offload coverage differs from native model")
    handles, restorations = [], []
    for region in regions:
        mapping = []
        for name, parameter in region.named_parameters():
            if not parameter.requires_grad:
                parent, _, leaf = name.rpartition(".")
                mapping.append((region.get_submodule(parent), leaf, parameter))

        def enter(module, arguments, mapping=mapping):
            for owner, name, parameter in mapping:
                owner._parameters[name] = torch.nn.Parameter(parameter.cuda(), requires_grad=False)

        def leave(module, arguments, output, mapping=mapping):
            for owner, name, parameter in mapping:
                owner._parameters[name] = parameter

        handles.extend((region.register_forward_pre_hook(enter), region.register_forward_hook(leave, always_call=True)))
        restorations.extend(mapping)
    return handles, restorations


@torch.no_grad()
def step_gpu_leaves(optimizer):
    from cosmos_framework.utils.generator.fused_adam import FusedAdam

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if not parameter.is_cuda or parameter.dtype != torch.float32:
                raise ValueError("GPU leaf reference requires actual FP32 CUDA parameters")
            state = optimizer.state[parameter]
            fused = FusedAdam(
                [parameter],
                lr=group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                bias_correction=True,
                capturable=True,
                master_weights=False,
            )
            fused.param_groups[0]["step"] = torch.tensor([int(state["step"].item())], dtype=torch.int32, device="cuda")
            fused.state[parameter] = {field: state[field].cuda() for field in ("exp_avg", "exp_avg_sq")}
            fused.step()
            for field in ("exp_avg", "exp_avg_sq"):
                state[field].copy_(fused.state[parameter][field].cpu())
            state["step"].fill_(int(fused.param_groups[0]["step"].item()))
            del fused


def close_gpu_trainables(handles, restorations):
    for owner, name, parameter in restorations:
        owner._parameters[name] = parameter
    for handle in handles:
        handle.remove()
