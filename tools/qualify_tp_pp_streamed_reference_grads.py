# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare native CUDA leaf-gradient streaming against the unstreamed optimizer."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn

from tools.tp_pp_streamed_reference_grads import StreamedReferenceGradients


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 32))
        self.optional = nn.Sequential(nn.Linear(64, 32), nn.Tanh())
        self.frozen = nn.Linear(64, 32)
        self.frozen.requires_grad_(False)

    def forward(self, x, use_optional):
        result = self.main(x) + self.frozen(x)
        return result + self.optional(x) if use_optional else result


def make_optimizer(model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=0.002, betas=(0.9, 0.99), weight_decay=0.03, foreach=False
    )
    for index, parameter in enumerate(parameters):
        generator = torch.Generator(device="cpu").manual_seed(3000 + index)
        optimizer.state[parameter] = {
            "step": torch.tensor(7.0),
            "exp_avg": torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            )
            * 0.01,
            "exp_avg_sq": torch.rand(
                parameter.shape, generator=generator, dtype=torch.float32
            )
            * 0.01
            + 0.001,
        }
    return optimizer


def qualify_foreach():
    generator = torch.Generator(device="cpu").manual_seed(2727)
    sizes = (1, 31, 32, 33, 511, 512, 513, 4096, 65537, 2_097_153)
    gradients = [torch.randn((size,), generator=generator).cuda() for size in sizes]
    gradients.append(torch.randn((129, 257), generator=generator).cuda().t())
    grouped = torch._foreach_norm(gradients, 2)
    streamed = [torch._foreach_norm([gradient], 2)[0] for gradient in gradients]
    if any(
        not torch.equal(left, right)
        for left, right in zip(grouped, streamed, strict=True)
    ):
        raise AssertionError(
            "Per-tensor CUDA foreach norms differ between grouped and single calls"
        )
    grouped_total = torch.linalg.vector_norm(torch.stack(grouped), 2)
    streamed_total = torch.linalg.vector_norm(torch.stack(streamed), 2)
    if not torch.equal(grouped_total, streamed_total):
        raise AssertionError("CUDA grouped and streamed total norms differ")
    group_coefficient = torch.clamp(0.3 / (grouped_total + 1e-6), max=1.0)
    stream_coefficient = torch.clamp(0.3 / (streamed_total + 1e-6), max=1.0)
    if not torch.equal(group_coefficient, stream_coefficient):
        raise AssertionError("CUDA grouped and streamed clip coefficients differ")
    grouped_clipped = [gradient.clone() for gradient in gradients]
    streamed_clipped = [gradient.clone() for gradient in gradients]
    torch._foreach_mul_(grouped_clipped, group_coefficient)
    for gradient in streamed_clipped:
        torch._foreach_mul_([gradient], stream_coefficient)
    if any(
        not torch.equal(left, right)
        for left, right in zip(grouped_clipped, streamed_clipped, strict=True)
    ):
        raise AssertionError("CUDA grouped and streamed clipping differs")
    return {
        "passed": True,
        "element_counts": [tensor.numel() for tensor in gradients],
        "noncontiguous": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    from tp_pp_gpu_leaf_reference import step_gpu_leaves

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    foreach_report = qualify_foreach()
    torch.manual_seed(71)
    reference = TinyModel().cuda()
    streamed = copy.deepcopy(reference)
    reference_optimizer, streamed_optimizer = (
        make_optimizer(reference),
        make_optimizer(streamed),
    )
    streamer = StreamedReferenceGradients(streamed.parameters())
    rows = []
    for update in range(4):
        reference_optimizer.zero_grad(set_to_none=True)
        streamed_optimizer.zero_grad(set_to_none=True)
        streamer.reset()
        missing = 0
        optional_missing = 0
        for microbatch in range(2):
            generator = torch.Generator(device="cpu").manual_seed(
                900 + update * 2 + microbatch
            )
            x = torch.randn((7, 64), generator=generator).cuda()
            used = update != 1 and (microbatch == 0 or update != 2)
            expected_output = reference(x, used)
            actual_output = streamed(x, used)
            if not torch.equal(expected_output, actual_output):
                raise AssertionError((update, microbatch, "forward output differs"))
            (expected_output.square().mean() / 2).backward()
            (actual_output.square().mean() / 2).backward()
            for name, (expected, actual) in zip(
                dict(reference.named_parameters()),
                zip(reference.parameters(), streamed.parameters(), strict=True),
                strict=True,
            ):
                host = streamer.raw(actual)
                if expected.grad is None:
                    missing += 1
                    optional_missing += int(name.startswith("optional."))
                    if host is not None:
                        raise AssertionError(
                            (update, microbatch, name, "expected no gradient")
                        )
                elif host is None or not torch.equal(host.cuda(), expected.grad):
                    raise AssertionError(
                        (update, microbatch, name, "accumulated gradient differs")
                    )
                if actual.grad is not None:
                    raise AssertionError(
                        (update, microbatch, name, "gradient was not offloaded")
                    )
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
        actual_norm = streamer.norm(streamed.parameters())
        if not torch.equal(actual_norm, expected_norm):
            raise AssertionError(
                (update, "norm differs", expected_norm.item(), actual_norm.item())
            )
        expected_coefficient = torch.clamp(0.3 / (expected_norm + 1e-6), max=1.0)
        actual_coefficient = torch.clamp(0.3 / (actual_norm + 1e-6), max=1.0)
        if not torch.equal(actual_coefficient, expected_coefficient):
            raise AssertionError((update, "clipping coefficient differs"))
        clipped = {}

        def step_one(optimizer):
            for name, parameter in streamed.named_parameters():
                if parameter.grad is not None:
                    clipped[name] = parameter.grad.detach().clone()
            step_gpu_leaves(optimizer)

        streamer.clip_and_step(streamed_optimizer, 0.3, actual_norm, step_one)
        for name, parameter in reference.named_parameters():
            if parameter.grad is None:
                if name in clipped:
                    raise AssertionError((update, name, "unexpected clipped gradient"))
            elif name not in clipped or not torch.equal(clipped[name], parameter.grad):
                raise AssertionError((update, name, "clipped gradient differs"))
        step_gpu_leaves(reference_optimizer)
        for (name, expected), (_, actual) in zip(
            reference.named_parameters(), streamed.named_parameters(), strict=True
        ):
            if not torch.equal(expected, actual):
                raise AssertionError((update, name, "parameter differs"))
            left, right = (
                reference_optimizer.state.get(expected, {}),
                streamed_optimizer.state.get(actual, {}),
            )
            if left.keys() != right.keys() or any(
                not torch.equal(left[key], right[key]) for key in left
            ):
                raise AssertionError((update, name, "optimizer state differs"))
        rows.append(
            {
                "update": update,
                "norm": actual_norm.item(),
                "clip_coefficient": actual_coefficient.item(),
                "missing_gradient_checks": missing,
                "missing_optional_gradient_checks": optional_missing,
            }
        )
    if rows[1]["missing_optional_gradient_checks"] == 0:
        raise AssertionError(
            "The missing-modality case did not cover optional trainable gradients"
        )
    streamer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(
            {
                "passed": True,
                "scope": "single-GPU FP32 tiny CUDA leaf oracle with nonzero CPU Adam moments",
                "tf32": False,
                "updates": rows,
                "foreach_norm_and_clipping": foreach_report,
                "microbatches_per_update": 2,
                "checks": "bitwise raw accumulated and clipped gradients, native norm and clip coefficient, parameters, Adam moments and steps",
                "full_model_acceptance": False,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
