# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Qualify CPU saved-tensor offload with compiled full AC and nonzero Adam."""

import argparse
import copy
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from tp_pp_gpu_leaf_reference_smoke import Reference

from cosmos_framework.callbacks.grad_clip import _clip_grads_with_global_norm, _total_norm_by_mesh
from cosmos_framework.utils.generator.fused_adam import FusedAdam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preserve-layout", action="store_true")
    parser.add_argument("--selective-blocks", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite qualification evidence")
    if not 0 <= args.selective_blocks <= 2 or args.selective_blocks and not args.preserve_layout:
        raise ValueError("Selective qualification requires preserve-layout and at most two blocks")
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("Tiny activation offload qualification deferred: less than 8 GiB free")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    source = Reference()
    models = [copy.deepcopy(source).cuda() for _ in range(2)]
    parameters = [[p for p in model.parameters() if p.requires_grad] for model in models]
    optimizers = [FusedAdam(p, lr=1e-4, weight_decay=0.01, capturable=True) for p in parameters]
    for optimizer in optimizers:
        optimizer.param_groups[0]["step"] = torch.tensor([1200], dtype=torch.int32, device="cuda")
    for pair in zip(*parameters, strict=True):
        states = {"exp_avg": torch.randn_like(pair[0]) * 1e-4, "exp_avg_sq": torch.rand_like(pair[0]) * 1e-5}
        for optimizer, parameter in zip(optimizers, pair, strict=True):
            optimizer.state[parameter] = {name: value.clone() for name, value in states.items()}

    def checkpoint_layer(layer):
        if args.selective_blocks:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

            return checkpoint_wrapper(layer, preserve_rng_state=True)
        return layer

    compiled = [
        [
            torch.compile(checkpoint_layer(layer), fullgraph=True, options={"triton.cudagraphs": False})
            for layer in model.language_model.model.layers
        ]
        for model in models
    ]

    def forward(index, value):
        for block in compiled[index]:
            value = block(value) if args.selective_blocks else checkpoint(block, value, use_reentrant=False)
        return models[index].language_model.lm_head(value)

    saved_cuda_bytes = 0
    saved_calls = 0
    changed_layouts = 0
    kept_cuda_calls = 0

    def observe(tensor, saved):
        nonlocal saved_cuda_bytes, saved_calls, changed_layouts, kept_cuda_calls
        if tensor.is_cuda:
            if saved[1].device.type == "cpu":
                saved_cuda_bytes += tensor.numel() * tensor.element_size()
                saved_calls += 1
                changed_layouts += int(saved[1].stride() != tensor.stride())
            else:
                assert args.selective_blocks and saved[1].data_ptr() == tensor.data_ptr()
                kept_cuda_calls += 1

    if args.selective_blocks:
        from tools.tp_pp_selective_activation_offload import wrap_blocks

        wrap_blocks(((str(i), block) for i, block in enumerate(compiled[1])), args.selective_blocks, observer=observe)
    reports = []
    for update in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for _ in range(2):
            value = torch.randn(7, 16, device="cuda")
            expected = forward(0, value)
            (expected.square().mean() / 2).backward()
            if args.selective_blocks:
                context = nullcontext()
            elif args.preserve_layout:
                from tools.run_tp_pp_with_activation_offload import activation_offload_context

                context = activation_offload_context("preserve_layout")
            else:
                context = torch.autograd.graph.save_on_cpu(pin_memory=True)
            if not args.selective_blocks:
                original_pack = context.pack_hook

                def pack(tensor):
                    saved = original_pack(tensor)
                    observe(tensor, saved)
                    return saved

                context.pack_hook = pack
            with context:
                actual = forward(1, value)
                (actual.square().mean() / 2).backward()
            assert torch.equal(actual, expected)
        for index, (left, right) in enumerate(zip(*parameters, strict=True)):
            if not torch.equal(left.grad, right.grad):
                from tools.compare_tp_pp_runtime_checkpoints import tensor_difference

                difference = tensor_difference(
                    left.grad.cpu(), right.grad.cpu(), relative_tolerance=1e-4, absolute_tolerance=2e-6, exact=False
                )
                failure = {
                    "passed": False,
                    "scope": "strict bitexact activation offload qualification, stopped at first gradient mismatch",
                    "update": update,
                    "parameter_index": index,
                    "preserve_layout": args.preserve_layout,
                    "changed_saved_tensor_layouts": changed_layouts,
                    "gradient_difference": difference,
                    "full_model_dp2_restart_accepted": False,
                }
                with args.output.open("x") as handle:
                    json.dump(failure, handle, indent=2)
                    handle.write("\n")
                print(json.dumps(failure), flush=True)
                raise AssertionError("Activation offload changed the exact gradient; failure report retained")
        norms = []
        for params in parameters:
            groups = {"default": params}
            norm, _ = _total_norm_by_mesh(groups, error_if_nonfinite=True)
            _clip_grads_with_global_norm(groups, 0.03, norm)
            norms.append(norm)
        assert torch.equal(*norms)
        for optimizer in optimizers:
            optimizer.step()
        for left, right in zip(*parameters, strict=True):
            assert torch.equal(left, right)
            for field in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(optimizers[0].state[left][field], optimizers[1].state[right][field])
        assert all(int(optimizer.param_groups[0]["step"]) == 1201 + update for optimizer in optimizers)
        reports.append({"update": update, "outputs_gradients_norm_parameters_moments_step_bitexact": True})
    assert saved_cuda_bytes > 0 and saved_calls > 0
    if args.selective_blocks:
        assert kept_cuda_calls > 0 and changed_layouts == 0
    report = {
        "passed": True,
        "scope": "tiny mixed frozen/trainable CUDA model, compiled per-block full AC, two microbatches and three nonzero native Adam updates; CPU saved-tensor offload only",
        "updates": reports,
        "saved_cuda_bytes": saved_cuda_bytes,
        "saved_cuda_tensor_calls": saved_calls,
        "changed_saved_tensor_layouts": changed_layouts,
        "preserve_layout": args.preserve_layout,
        "selective_blocks": args.selective_blocks,
        "kept_cuda_parameter_calls": kept_cuda_calls,
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
        "full_model_dp2_restart_accepted": False,
        "production_performance_accepted": False,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
