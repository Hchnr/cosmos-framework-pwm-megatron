"""Capture pre-clipping full-model gradients for a separate correctness audit.

Run via torchrun with train_real_batch_smoke arguments, steps=1, warmup=0 and
benchmark-json set to disable checkpoint I/O. These timings are not benchmarks.
"""

import dataclasses
import os
from pathlib import Path

import torch
import torch.distributed as dist
import train_real_batch_smoke as driver
from torch.distributed.tensor import DTensor, Shard

from cosmos_framework.callbacks.grad_clip import GradClip
from cosmos_framework.training_backends.checkpoint_metadata import extract_chunk_metadata, prepare_uneven_state

_original_clip = GradClip.on_before_optimizer_step


def capture_then_clip(self, model, optimizer, scheduler, grad_scaler, iteration=0):
    if os.environ.get("GRADIENT_AUDIT_INPUTS_ONLY") == "1":
        return _original_clip(self, model, optimizer, scheduler, grad_scaler, iteration)
    gradients = {}
    for name, parameter in model.net.named_parameters():
        if not parameter.requires_grad:
            continue
        name = name.replace("._checkpoint_wrapped_module", "").replace("._orig_mod", "")
        gradients[name] = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
    prepare_uneven_state(gradients, dist.group.WORLD)
    names = sorted(gradients)
    sums = []
    saved = {}
    metadata = dict(extract_chunk_metadata(gradients))
    for name in names:
        gradient = gradients[name]
        local = gradient.to_local() if isinstance(gradient, DTensor) else gradient
        square = local.float().square().sum()
        if not isinstance(gradient, DTensor) or not any(isinstance(p, Shard) for p in gradient.placements):
            square = square / dist.get_world_size()
        sums.append(square)
        saved[name] = {
            "local": local.detach().cpu(),
            "global_shape": list(gradient.shape),
            "chunks": metadata.get((name,), []),
            "placements": str(getattr(gradient, "placements", ())),
            "shard_dim": next((p.dim for p in gradient.placements if isinstance(p, Shard)), None)
            if isinstance(gradient, DTensor)
            else None,
        }
    totals = torch.stack(sums)
    dist.all_reduce(totals)
    root = Path(os.environ["GRADIENT_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "gradients": saved,
            "squared_norms": dict(zip(names, totals.cpu().tolist(), strict=True)),
        },
        root / f"rank_{dist.get_rank()}.pt",
    )
    print(f"GRADIENT_AUDIT rank={dist.get_rank()} fp32_global_norm={totals.sum().sqrt().item()}", flush=True)
    return _original_clip(self, model, optimizer, scheduler, grad_scaler, iteration)


def capture_inputs(module, args, kwargs):
    count = getattr(module, "_gradient_audit_input_count", 0)
    leaves = {}

    def visit(value, path="args"):
        if isinstance(value, torch.Tensor):
            leaves[path] = value.detach().cpu()
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name), path + "." + field.name)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + "." + str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, path + "." + str(index))

    visit(args)
    visit(kwargs, "kwargs")
    root = Path(os.environ["GRADIENT_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    torch.save(leaves, root / f"input_rank_{dist.get_rank()}_micro_{count}.pt")
    module._gradient_audit_input_count = count + 1


if __name__ == "__main__":
    if os.environ.get("GRADIENT_AUDIT_DETERMINISTIC") == "1":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        original_load_config = driver.load_experiment_from_toml

        def load_config(*args, **kwargs):
            kwargs["extra_overrides"] += ["trainer.cudnn.deterministic=true", "trainer.cudnn.benchmark=false"]
            return original_load_config(*args, **kwargs)

        driver.load_experiment_from_toml = load_config
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    original_init = OmniMoTModel.__init__

    def init_with_input_capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.net.register_forward_pre_hook(capture_inputs, with_kwargs=True)

    OmniMoTModel.__init__ = init_with_input_capture
    GradClip.on_before_optimizer_step = capture_then_clip
    driver.main()
