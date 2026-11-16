"""Numerical parity at actual Nano/action widths and production token capacities."""

import argparse
import copy
import faulthandler
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_three_tower import Stack, inputs
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from cosmos_framework.callbacks.grad_clip import GradClip
from cosmos_framework.training_backends.megatron import MegatronRuntime
from cosmos_framework.training_backends.projection_fusion import fuse_decoder_projections
from cosmos_framework.utils.generator.fused_adam import FusedAdam


def canonical(name):
    return name.replace("._checkpoint_wrapped_module", "").replace("._orig_mod", "")


def main():
    faulthandler.dump_traceback_later(60, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    parser.add_argument("--fuse-projections", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import uneven_dtensor_to_full_tensor

    torch.manual_seed(42)
    reference = Stack(1, False).cuda().to(torch.bfloat16)
    candidate = copy.deepcopy(reference)
    if args.fuse_projections:
        fuse_decoder_projections([layer._checkpoint_wrapped_module for layer in candidate.layers])
    for model in (reference, candidate):
        for index, layer in enumerate(model.layers):
            model.layers[index] = torch.compile(layer, fullgraph=True, dynamic=False)
    pack = inputs(reference.shape)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("shard",))
    for layer in reference.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(reference, mesh=mesh)
    runtime = MegatronRuntime(
        candidate,
        shard_degree=dist.get_world_size(),
        unit_types=(type(candidate.layers[0]),),
        sharding_strategy=args.sharding_strategy,
    )
    optimizers = [
        FusedAdam(
            (p for p in model.parameters() if p.requires_grad),
            lr=1e-4,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=0.05,
            capturable=True,
            master_weights=True,
        )
        for model in (reference, candidate)
    ]
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    runtime.zero_grad()
    losses = []
    for micro in range(3):
        print(f"rank={dist.get_rank()} micro={micro} reference", flush=True)
        expected = reference(pack) / 3
        expected.backward()
        print(f"rank={dist.get_rank()} micro={micro} candidate", flush=True)
        with runtime.accumulate(micro == 2):
            actual = candidate(pack) / 3
            actual.backward()
        torch.testing.assert_close(actual, expected, rtol=0.005, atol=1e-5)
        losses.append([expected.item(), actual.item()])
    refs = {canonical(name): p for name, p in reference.named_parameters()}
    parameters = {canonical(name): p for name, p in candidate.named_parameters()}
    gradients = {
        name: p.grad if p.grad is not None else torch.zeros_like(p) for name, p in parameters.items() if p.requires_grad
    }
    print(f"rank={dist.get_rank()} gradients ready", flush=True)
    runtime.prepare_state_dict(gradients)
    difference, magnitude, max_error = 0.0, 0.0, 0.0
    for name, gradient in gradients.items():
        if dist.get_rank() == 0:
            print(f"compare gradient: {name}", flush=True)
        actual = uneven_dtensor_to_full_tensor(gradient).float()
        expected = refs[name].grad.full_tensor().float()
        error = actual - expected
        difference += error.square().sum().item()
        magnitude += expected.square().sum().item()
        max_error = max(max_error, error.abs().max().item())
    print(f"rank={dist.get_rank()} gradient comparison complete", flush=True)
    relative_gradient_error = (difference / magnitude) ** 0.5
    assert relative_gradient_error < 0.02, relative_gradient_error
    clipping = [GradClip(clip_norm=0.1, error_if_nonfinite=True) for _ in range(2)]
    for clip, model, optimizer in zip(clipping, (reference, candidate), optimizers, strict=True):
        clip.on_before_optimizer_step(model, optimizer, None, None)
        optimizer.step()
    runtime.after_step()
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    expected_state = get_model_state_dict(reference)
    actual_state = get_model_state_dict(candidate)
    difference, magnitude = 0.0, 0.0
    for name, expected_tensor in expected_state.items():
        expected = expected_tensor.full_tensor().float()
        actual = uneven_dtensor_to_full_tensor(actual_state[name]).float()
        difference += (actual - expected).square().sum().item()
        magnitude += expected.square().sum().item()
    relative_weight_error = (difference / magnitude) ** 0.5
    assert relative_weight_error < 0.002, relative_weight_error
    if dist.get_rank() == 0:
        report = vars(args) | {
            "output": str(args.output),
            "passed": True,
            "torch": torch.__version__,
            "seed": 42,
            "scope": "production-width one-layer three-tower decoder",
            "losses": losses,
            "relative_gradient_error": relative_gradient_error,
            "max_gradient_error": max_error,
            "relative_weight_error": relative_weight_error,
            "gradient_error_limit": 0.02,
            "weight_error_limit": 0.002,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    faulthandler.cancel_dump_traceback_later()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
