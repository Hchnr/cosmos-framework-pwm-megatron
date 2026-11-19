"""Match native FP32 clipping and capturable FusedAdam arithmetic in MCore."""

import torch


def clip_parameters(parameters, max_norm, total_norm, use_decoupled_grad=False):
    from megatron.core.utils import to_local_if_dtensor

    gradients = []
    for parameter in parameters:
        gradient = getattr(parameter, "decoupled_grad", None) if use_decoupled_grad else parameter.grad
        if gradient is not None:
            gradient = to_local_if_dtensor(gradient).detach()
            if gradient.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("Native clipping candidate requires FP32/BF16 gradients")
            gradients.append(gradient)
    if not gradients:
        return
    if len({gradient.device for gradient in gradients}) != 1:
        raise ValueError("One optimizer clipping call must use a single device")
    # Native torch clipping keeps norm, epsilon addition, and reciprocal in
    # FP32. MCore's float-return path otherwise divides Python doubles first.
    norm = torch.as_tensor(total_norm, dtype=torch.float32, device=gradients[0].device)
    coefficient = (float(max_norm) / (norm + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(gradients, coefficient)


def install_native_clip_arithmetic():
    from megatron.core.optimizer import clip_grads
    from megatron.core.optimizer import optimizer as optimizer_module

    clip_grads.clip_grad_by_total_norm_fp32 = clip_parameters
    # The optimizer imports this symbol directly, so update its bound alias.
    optimizer_module.clip_grad_by_total_norm_fp32 = clip_parameters


def _normalize_native_group_scalars(optimizer, args, kwargs):
    for group in optimizer.param_groups:
        if not group["params"]:
            continue
        device = group["params"][0].device
        group["lr"] = torch.as_tensor(group["lr"], dtype=torch.float32, device=device)
        group["step"] = torch.as_tensor(group.get("step", 0), dtype=torch.int32, device=device).reshape(1)


def align_native_adam_arithmetic(optimizer):
    """Retain MCore shard ownership while using the native optimizer kernel mode."""
    install_native_clip_arithmetic()
    for wrapper in getattr(optimizer, "chained_optimizers", [optimizer]):
        implementation = wrapper.optimizer
        if not hasattr(implementation, "capturable"):
            raise RuntimeError("Native arithmetic alignment requires the pinned TE FusedAdam implementation")
        implementation.capturable = True
        if not hasattr(implementation, "_cosmos_native_arithmetic_hook"):
            implementation._cosmos_native_arithmetic_hook = implementation.register_step_pre_hook(
                _normalize_native_group_scalars
            )
