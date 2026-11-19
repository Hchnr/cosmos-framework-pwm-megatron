# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare the backward peak of flat outer and nested MLP checkpoints.

This isolated synthetic gated MLP probe is not a full-model qualification. It
uses the same token chunks, initial weights, inputs, and upstream gradient in
both cases. CUDA peak allocated memory is reset immediately before backward;
the backward includes recomputation under the real flat outer checkpoint.
"""

import argparse
import gc
import json
from pathlib import Path

import torch

from tools.run_tp_pp_checkpointed_chunked_mlp_math import checkpointed_chunks
from tools.run_tp_pp_flat_checkpoint_math import flat_checkpoint


class GatedMLP(torch.nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.gate = torch.nn.Linear(width, intermediate, bias=False)
        self.up = torch.nn.Linear(width, intermediate, bias=False)
        self.down = torch.nn.Linear(intermediate, width, bias=False)

    def forward(self, value):
        return self.down(torch.nn.functional.silu(self.gate(value)) * self.up(value))


def measure_case(initial, input_cpu, upstream_cpu, chunk_tokens, nested, device):
    model = GatedMLP(initial["gate.weight"].shape[1], initial["gate.weight"].shape[0])
    model.load_state_dict(initial)
    model.to(device=device)
    inputs = input_cpu.to(device=device, copy=True).requires_grad_()
    upstream = upstream_cpu.to(device=device)
    block_calls = 0

    def block(pack):
        nonlocal block_calls
        block_calls += 1
        value = pack["tokens"]
        if nested:
            result = checkpointed_chunks(model, value, chunk_tokens)
        else:
            result = torch.cat([model(part) for part in value.split(chunk_tokens, dim=0)], dim=0)
        return value + result

    result = flat_checkpoint(block, {"tokens": inputs})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = None
    result.backward(upstream)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = None
    saved = {
        "output": result.detach().cpu().clone(),
        "input_grad": inputs.grad.detach().cpu().clone(),
        "parameter_grads": {name: value.grad.detach().cpu().clone() for name, value in model.named_parameters()},
    }
    return {
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_over_baseline_bytes": None if peak is None else peak - baseline,
        "outer_block_calls": block_calls,
    }, saved


def run_comparison(*, tokens, width, intermediate, chunk_tokens, device):
    if min(tokens, width, intermediate, chunk_tokens) <= 0 or chunk_tokens >= tokens:
        raise ValueError("Positive shapes and at least two token chunks are required")
    torch.manual_seed(713)
    initial = GatedMLP(width, intermediate).state_dict()
    inputs = torch.randn(tokens, width)
    upstream = torch.randn_like(inputs)
    plain, plain_values = measure_case(initial, inputs, upstream, chunk_tokens, False, device)
    # Release any checkpoint graph cycles before measuring the next case.
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    nested, nested_values = measure_case(initial, inputs, upstream, chunk_tokens, True, device)
    for name in ("output", "input_grad"):
        torch.testing.assert_close(nested_values[name], plain_values[name], rtol=1e-5, atol=1e-6)
    for name, expected in plain_values["parameter_grads"].items():
        torch.testing.assert_close(nested_values["parameter_grads"][name], expected, rtol=1e-5, atol=1e-6)
    if plain["outer_block_calls"] < 2 or nested["outer_block_calls"] < 2:
        raise AssertionError("Both cases must exercise the outer checkpoint recomputation")
    memory_reduced = None if device.type != "cuda" else nested["peak_allocated_bytes"] < plain["peak_allocated_bytes"]
    return {
        "scope": "synthetic FP32 gated MLP, flat outer non-reentrant checkpoint, backward peak",
        "device": str(device),
        "shape": {"tokens": tokens, "width": width, "intermediate": intermediate, "chunk_tokens": chunk_tokens},
        "outputs_and_all_gradients_close": True,
        "plain": plain,
        "nested": nested,
        "cuda_peak_reduced": memory_reduced,
        "passed": memory_reduced is True,
        "full_model_acceptance": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the peak-memory qualification")
    report = run_comparison(
        tokens=args.tokens,
        width=args.width,
        intermediate=args.intermediate,
        chunk_tokens=args.chunk_tokens,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not report["passed"]:
        raise RuntimeError("Nested checkpoint did not reduce the measured CUDA allocated peak")


if __name__ == "__main__":
    main()
