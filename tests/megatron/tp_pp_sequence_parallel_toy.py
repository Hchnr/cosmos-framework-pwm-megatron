"""Distributed CPU/Gloo or CUDA/NCCL mathematical SP reference experiment.

This is a two-layer synthetic FP64 control, not an MCore training backend or
performance result. It checks three continuous nonzero Adam updates, GQA,
RoPE, uneven valid masks, causal UND, missing action, inputs, per-parameter
shards, first/second moments, and the replicated loss scaling rule.
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist

from tools.tp_pp_sequence_parallel_experiment import (
    ThreeStreamBlock,
    gather,
    initialize_reference,
    slice_weight,
    validate_contract,
)


def ids_for(update, device):
    return {
        "und": torch.tensor([0, 0, 0, 1, 1, 1, -1, -1], device=device),
        "video": torch.tensor([0] * 5 + [1] * 5 + [-1] * 2, device=device),
        "action": torch.tensor([] if update == 1 else [0, 1, 1, -1], device=device, dtype=torch.long),
    }


def loss(streams, ids, *, gen_targets=True):
    names = ("und", "video", "action") if gen_targets else ("und",)
    valid = [streams[name][ids[name] >= 0] for name in names]
    count = sum(tensor.numel() for tensor in valid)
    return sum(value.square().sum() for value in valid) / count


def full_reference(layers, streams, ids, positions):
    for layer in layers:
        streams = layer(streams, ids, positions)
    return streams


def sharded_candidate(layers, streams, ids, positions, group):
    for layer in layers:
        streams = layer(streams, ids, positions, group=group)
    return {name: gather(value, group) for name, value in streams.items()}


def build_models(rank, world, device, dtype):
    reference = torch.nn.ModuleList([ThreeStreamBlock().to(device=device, dtype=dtype) for _ in range(2)])
    candidate = torch.nn.ModuleList(
        [ThreeStreamBlock(rank=rank, world=world).to(device=device, dtype=dtype) for _ in range(2)]
    )
    generator = torch.Generator(device=device).manual_seed(221)
    for full, local in zip(reference, candidate, strict=True):
        initialize_reference(full, generator)
        with torch.no_grad():
            for name, param in full.weights.items():
                local.weights[name].copy_(slice_weight(name, param, rank, world))
    return reference, candidate


def build_optimizers(reference, candidate, rank, world, device):
    reference_optim = torch.optim.Adam(reference.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    candidate_optim = torch.optim.Adam(candidate.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    generator = torch.Generator(device=device).manual_seed(900)
    for full, local in zip(reference, candidate, strict=True):
        for name, param in full.weights.items():
            exp_avg = torch.randn(param.shape, generator=generator, dtype=param.dtype, device=device) * 0.001
            exp_avg_sq = torch.rand(param.shape, generator=generator, dtype=param.dtype, device=device) * 0.001 + 0.01
            reference_optim.state[param] = {"step": torch.tensor(7.0), "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
            candidate_optim.state[local.weights[name]] = {
                "step": torch.tensor(7.0),
                "exp_avg": slice_weight(name, exp_avg, rank, world).clone(),
                "exp_avg_sq": slice_weight(name, exp_avg_sq, rank, world).clone(),
            }
    return reference_optim, candidate_optim


def compare_state(reference, candidate, ref_optim, cand_optim, rank, world, *, check_grad, rtol, atol):
    maximum = 0.0
    compared = 0
    for full, local in zip(reference, candidate, strict=True):
        for name, full_param in full.weights.items():
            local_param = local.weights[name]
            values = [(slice_weight(name, full_param, rank, world), local_param)]
            if check_grad:
                if (full_param.grad is None) != (local_param.grad is None):
                    raise AssertionError(f"Gradient presence differs for {name}")
                if full_param.grad is not None:
                    values.append((slice_weight(name, full_param.grad, rank, world), local_param.grad))
            else:
                for key in ("exp_avg", "exp_avg_sq"):
                    values.append(
                        (
                            slice_weight(name, ref_optim.state[full_param][key], rank, world),
                            cand_optim.state[local_param][key],
                        )
                    )
                torch.testing.assert_close(
                    ref_optim.state[full_param]["step"], cand_optim.state[local_param]["step"], rtol=0, atol=0
                )
            for expected, actual in values:
                if expected.numel():
                    maximum = max(maximum, (expected - actual).abs().max().item())
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                compared += 1
    return {"maximum_absolute_error": maximum, "compared_tensors": compared}


def clip_and_compare(reference, candidate, group, *, rtol, atol):
    reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3, foreach=False)
    values = [parameter.grad for parameter in candidate.parameters() if parameter.grad is not None]
    total = torch.zeros((), device=values[0].device, dtype=torch.float64)
    for gradient in values:
        total += gradient.double().square().sum()
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    norm = total.sqrt()
    torch.testing.assert_close(norm, reference_norm.double(), rtol=rtol, atol=atol)
    scale = torch.clamp(0.3 / (norm + 1e-6), max=1.0)
    with torch.no_grad():
        for gradient in values:
            gradient.mul_(scale.to(dtype=gradient.dtype))
    return {"reference_norm": reference_norm.item(), "sharded_norm": norm.item(), "clip_scale": scale.item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world not in (2, 4):
        raise ValueError("The toy supports exactly two or four ranks")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    dtype = getattr(torch, args.dtype)
    rtol, atol = (1e-4, 1e-5) if dtype == torch.float32 else (1e-8, 1e-9)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=180))
    try:
        group = dist.group.WORLD
        reference, candidate = build_models(rank, world, device, dtype)
        ropt, copt = build_optimizers(reference, candidate, rank, world, device)
        records = []
        for update in range(3):
            ids = ids_for(update, device)
            lengths = {name: tensor.numel() for name, tensor in ids.items()}
            validate_contract(lengths, group)
            positions = {name: torch.arange(length, device=device) for name, length in lengths.items()}
            generator = torch.Generator(device=device).manual_seed(300 + update)
            full_inputs = {
                name: torch.randn(length, 16, dtype=dtype, device=device, generator=generator).requires_grad_()
                for name, length in lengths.items()
            }
            local_inputs = {
                name: value.detach()
                .narrow(0, rank * (lengths[name] // world), lengths[name] // world)
                .clone()
                .requires_grad_()
                for name, value in full_inputs.items()
            }
            ropt.zero_grad(set_to_none=True)
            copt.zero_grad(set_to_none=True)
            ref = full_reference(reference, full_inputs, ids, positions)
            candidate_output = sharded_candidate(candidate, local_inputs, ids, positions, group)
            for name in ids:
                torch.testing.assert_close(candidate_output[name], ref[name], rtol=rtol, atol=atol)
            reference_loss = loss(ref, ids, gen_targets=update != 2)
            candidate_loss = loss(candidate_output, ids, gen_targets=update != 2) / world
            reference_loss.backward()
            candidate_loss.backward()
            torch.testing.assert_close(candidate_loss.detach() * world, reference_loss.detach(), rtol=rtol, atol=atol)
            for name, full in full_inputs.items():
                if (full.grad is None) != (local_inputs[name].grad is None):
                    raise AssertionError(f"Input gradient presence differs for {name}")
                if full.grad is not None:
                    expected = full.grad.narrow(0, rank * (lengths[name] // world), lengths[name] // world)
                    torch.testing.assert_close(local_inputs[name].grad, expected, rtol=rtol, atol=atol)
            gradient = compare_state(
                reference, candidate, ropt, copt, rank, world, check_grad=True, rtol=rtol, atol=atol
            )
            clip = clip_and_compare(reference, candidate, group, rtol=rtol, atol=atol)
            clipped = compare_state(
                reference, candidate, ropt, copt, rank, world, check_grad=True, rtol=rtol, atol=atol
            )
            ropt.step()
            copt.step()
            optimizer = compare_state(
                reference, candidate, ropt, copt, rank, world, check_grad=False, rtol=rtol, atol=atol
            )
            records.append(
                {
                    "update": update,
                    "missing_action": update == 1,
                    "no_generation_targets": update == 2,
                    "gradient": gradient,
                    "clip": clip,
                    "clipped_gradient": clipped,
                    "optimizer": optimizer,
                }
            )
        # Reject unequal metadata and non-divisible physical lengths on all ranks.
        for lengths in ({"und": 7, "video": 12, "action": 4}, {"und": 8 + rank * 4, "video": 12, "action": 4}):
            try:
                validate_contract(lengths, group)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid or rank-mismatched metadata was not rejected")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "rank": rank,
                    "world": world,
                    "backend": args.backend,
                    "dtype": args.dtype,
                    "rtol": rtol,
                    "atol": atol,
                    "steps": records,
                    "production_acceptance": False,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
