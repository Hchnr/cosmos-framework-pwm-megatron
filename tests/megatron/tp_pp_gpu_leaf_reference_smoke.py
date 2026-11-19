# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Three native CUDA updates must remain exact with frozen-region offload."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from tp_pp_gpu_leaf_reference import close_gpu_trainables, place_gpu_trainables, step_gpu_leaves

from cosmos_framework.callbacks.grad_clip import _clip_grads_with_global_norm, _total_norm_by_mesh
from cosmos_framework.utils.generator.fused_adam import FusedAdam


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(16, 16)
        self.active = nn.Linear(16, 16)
        self.frozen.requires_grad_(False)

    def forward(self, value):
        return torch.tanh(self.frozen(value) + self.active(value))


class Reference(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([Block(), Block()])
        self.language_model.lm_head = nn.Linear(16, 4)
        self.language_model.lm_head.requires_grad_(False)

    def forward(self, value):
        for layer in self.language_model.model.layers:
            value = layer(value)
        return self.language_model.lm_head(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    source = Reference()
    baseline = copy.deepcopy(source).cuda()
    active = [parameter for parameter in source.parameters() if parameter.requires_grad]
    original_ids = [id(parameter) for parameter in active]
    native_parameters = [parameter for parameter in baseline.parameters() if parameter.requires_grad]
    holder = torch.optim.AdamW(active, lr=1e-4, foreach=False)
    native = FusedAdam(native_parameters, lr=1e-4, weight_decay=0.01, capturable=True)
    native.param_groups[0]["step"] = torch.tensor([1200], dtype=torch.int32, device="cuda")
    for parameter, expected in zip(active, native_parameters, strict=True):
        state = {"exp_avg": torch.randn_like(parameter) * 1e-4, "exp_avg_sq": torch.rand_like(parameter) * 1e-5}
        holder.state[parameter] = {"step": torch.tensor(1200.0), **state}
        native.state[expected] = {name: value.cuda() for name, value in state.items()}
    handles, restorations = place_gpu_trainables(source)
    assert [id(parameter) for parameter in active] == original_ids
    assert all(parameter.is_cuda and parameter.is_leaf for parameter in active)
    reports = []
    try:
        for update in range(3):
            holder.zero_grad(set_to_none=True)
            native.zero_grad(set_to_none=True)
            for _ in range(2):
                value = torch.randn(7, 16, device="cuda")
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    actual = source(value)
                    expected = baseline(value)
                    assert torch.equal(actual, expected)
                    loss = actual.square().mean() / 2
                    reference_loss = expected.square().mean() / 2
                loss.backward()
                reference_loss.backward()
                assert all(not parameter.is_cuda for parameter in source.parameters() if not parameter.requires_grad)
            for parameter, expected in zip(active, native_parameters, strict=True):
                assert torch.equal(parameter.grad, expected.grad)
            norms = []
            for parameters in (active, native_parameters):
                groups = {"default": parameters}
                norm, _ = _total_norm_by_mesh(groups, error_if_nonfinite=True)
                _clip_grads_with_global_norm(groups, 0.03, norm)
                norms.append(norm)
            assert torch.equal(*norms)
            step_gpu_leaves(holder)
            native.step()
            for parameter, expected in zip(active, native_parameters, strict=True):
                assert torch.equal(parameter, expected)
                for field in ("exp_avg", "exp_avg_sq"):
                    assert torch.equal(holder.state[parameter][field], native.state[expected][field].cpu())
                assert int(holder.state[parameter]["step"]) == int(native.param_groups[0]["step"]) == 1201 + update
            reports.append({"update": update, "output_gradient_norm_master_moments_exact": True})
    finally:
        close_gpu_trainables(handles, restorations)
    with args.output.open("x") as handle:
        json.dump({"passed": True, "updates": reports, "full_model_acceptance": False}, handle, indent=2)
        handle.write("\n")
    print("Frozen offload / GPU leaves / native clipping and streamed Adam: three updates exact", flush=True)


if __name__ == "__main__":
    main()
