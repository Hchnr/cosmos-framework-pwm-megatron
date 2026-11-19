"""Qualify the production Flex helper against FP32 math SDPA, including GQA."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel

from cosmos_framework.model.generator.mot.small_action_attention import sample_block_mask, sparse_attend


def difference(left, right):
    a, b = left.detach().double(), right.detach().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    relative = float((a - b).norm() / a.norm().clamp_min(1e-30))
    cosine = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
    return {
        "passed": finite and relative <= 1e-4 and cosine >= 0.999999,
        "finite": finite,
        "relative_l2": relative,
        "cosine": cosine,
        "maximum_absolute_error": float((a - b).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite Flex qualification evidence")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("Small Flex qualification deferred: less than 8 GiB free")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    query_ids = torch.repeat_interleave(torch.tensor([0, 1, 2, -1]), torch.tensor([100, 80, 64, 12])).cuda()
    key_ids = torch.repeat_interleave(torch.tensor([0, 1, 2, -1]), torch.tensor([128, 100, 96, 60])).cuda()
    mask = sample_block_mask(query_ids, key_ids)
    dense = ((query_ids[:, None] >= 0) & (query_ids[:, None] == key_ids[None, :])) | (
        (query_ids[:, None] < 0) & (torch.arange(len(key_ids), device="cuda")[None, :] == 0)
    )
    tensors = [torch.randn(shape).cuda() for shape in [(256, 32, 128), (384, 8, 128), (384, 8, 128)]]
    reference = [tensor.clone().requires_grad_() for tensor in tensors]
    candidate = [tensor.clone().requires_grad_() for tensor in tensors]
    upstream = torch.randn(256, 4096).cuda()
    with sdpa_kernel(SDPBackend.MATH):
        expected = (
            functional.scaled_dot_product_attention(
                *(tensor.transpose(0, 1)[None] for tensor in reference), attn_mask=dense, enable_gqa=True
            )[0]
            .transpose(0, 1)
            .flatten(1)
        )
    expected_gradients = torch.autograd.grad((expected * upstream).sum(), reference)
    actual = sparse_attend(*candidate, mask)
    actual_gradients = torch.autograd.grad((actual * upstream).sum(), candidate)
    rows = {"output": difference(expected, actual)}
    rows.update(
        {
            key: difference(left, right)
            for key, left, right in zip(("dq", "dk", "dv"), expected_gradients, actual_gradients, strict=True)
        }
    )
    source = Path("cosmos_framework/model/generator/mot/small_action_attention.py")
    report = {
        "passed": all(row["passed"] for row in rows.values()),
        "scope": "small production Flex FP32 kernel versus math SDPA; unequal packed samples, padding and GQA32/8",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": rows,
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
        "full_model_math_accepted": False,
        "production_capacity_576_slot_acceptance": False,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("FP32 Flex kernel qualification failed")


if __name__ == "__main__":
    main()
