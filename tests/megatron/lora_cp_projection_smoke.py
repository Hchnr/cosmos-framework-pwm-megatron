"""Independent full-projection reference for CP column slices with LoRA."""

import argparse
import ast
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from cosmos_framework.model.generator.mot.unified_mot import _apply_head_sharded_o_proj
from cosmos_framework.utils.generator.lora import LoraInjectedLinear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-source", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    projection = LoraInjectedLinear(nn.Linear(64, 48, bias=True), rank=4, alpha=8).to_empty(device="cuda")
    with torch.no_grad():
        for parameter in projection.parameters():
            parameter.uniform_(-0.2, 0.2)
    reference = copy.deepcopy(projection)
    inputs = torch.randn(7, 64, device="cuda", requires_grad=True)
    expected_inputs = inputs.detach().clone().requires_grad_()
    width = 64 // dist.get_world_size()
    feature_slice = slice(dist.get_rank() * width, (dist.get_rank() + 1) * width)
    expected = reference(expected_inputs)
    legacy_error = None
    if args.legacy_source is not None:
        parsed = ast.parse(args.legacy_source.read_text())
        function = next(
            node
            for node in parsed.body
            if isinstance(node, ast.FunctionDef) and node.name == "_apply_head_sharded_o_proj"
        )
        namespace = {"torch": torch, "nn": nn, "ProcessGroup": dist.ProcessGroup}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.legacy_source), "exec"), namespace)
        legacy = namespace[function.name](inputs[:, feature_slice], projection, feature_slice, dist.group.WORLD)
        legacy_error = (legacy - expected).abs().max().item()
    actual = _apply_head_sharded_o_proj(inputs[:, feature_slice], projection, feature_slice, dist.group.WORLD)
    error = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    maximum_gradient_error = 0.0
    for (name, parameter), (_, baseline) in zip(
        projection.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None, name
        # Bias is added after the sum, so each rank already has its complete
        # gradient. Other parameters contribute one input-column partial.
        if name != "bias":
            dist.all_reduce(parameter.grad)
        torch.testing.assert_close(parameter.grad, baseline.grad, rtol=2e-5, atol=2e-6, msg=name)
        maximum_gradient_error = max(maximum_gradient_error, (parameter.grad - baseline.grad).abs().max().item())
    dist.all_reduce(inputs.grad)
    torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=2e-5, atol=2e-6)
    result = dict(
        passed=True,
        cp=dist.get_world_size(),
        nonzero_lora_A_and_B=True,
        tf32=False,
        legacy_source=str(args.legacy_source) if args.legacy_source is not None else None,
        legacy_maximum_output_error=legacy_error,
        independent_full_projection_reference=True,
        maximum_output_error=error,
        maximum_gradient_error=maximum_gradient_error,
    )
    if dist.get_rank() == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
