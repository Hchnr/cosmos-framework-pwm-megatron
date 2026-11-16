"""Poison uninitialized buffers to reproduce legacy varlen padding gradients."""

import importlib.util
import json
import os
from functools import partial
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

import cosmos_framework.model.generator.mot.attention as current
from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_mode_splits,
    get_all_seq,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.attention import attention


def run(function):
    torch.manual_seed(42)
    weights = (torch.randn(3, 256, 256, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_()
    hidden = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
    base = sequence_pack_from_packed_sequence(
        hidden,
        ["causal", "full"],
        [2, 3],
        [5],
        torch.arange(2, device="cuda"),
        torch.arange(2, 5, device="cuda"),
        cp_world_size=2,
    )
    packs = [
        from_mode_splits(
            F.linear(base["causal_seq"], weight).view(-1, 4, 64),
            F.linear(base["full_only_seq"], weight).view(-1, 4, 64),
            base,
        )
        for weight in weights
    ]
    output = get_all_seq(function(*packs))
    forward_finite = bool(torch.isfinite(output).all())
    output.float().square().mean().backward()
    nonfinite = int((~torch.isfinite(weights.grad)).sum())
    return dict(
        forward_real_tokens_finite=forward_finite, nonfinite_parameter_gradient_elements=nonfinite
    ), output.detach().cpu()


def main():
    path = Path(
        "/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm/cosmos_framework/model/generator/mot/attention.py"
    )
    spec = importlib.util.spec_from_file_location("cosmos_original_padding_attention", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.attention = partial(attention, backend="flash3", deterministic=True)
    current.attention = partial(attention, backend="flash3", deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    old, old_output = run(original.two_way_attention)
    new, new_output = run(current.two_way_attention)
    torch.testing.assert_close(old_output, new_output, rtol=0, atol=0)
    assert old["forward_real_tokens_finite"] and old["nonfinite_parameter_gradient_elements"] > 0, old
    assert new["forward_real_tokens_finite"] and new["nonfinite_parameter_gradient_elements"] == 0, new
    result = dict(
        passed=True,
        backend="flash3",
        dtype="bfloat16",
        poison_uninitialized_buffers=True,
        real_forward_bitwise_equal=True,
        original_source=str(path),
        original=old,
        fixed=new,
    )
    Path("artifacts/megatron/legacy_padding_gradient_qualification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
