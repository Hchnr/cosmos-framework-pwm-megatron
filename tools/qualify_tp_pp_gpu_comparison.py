"""Require unchanged FP64 comparison decisions on CPU and CUDA."""

import argparse
import json
import math
from pathlib import Path

import torch

from tools.compare_tp_pp_runtime_checkpoints import tensor_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(391)
    base = torch.randn(1024, 4096) * 1e-3
    rows = []
    for label, left, right in [
        ("exact", base, base.clone()),
        ("relative_pass", base, base + torch.randn_like(base) * 2e-8),
        ("relative_fail", base, base + torch.randn_like(base) * 2e-6),
        ("zero_absolute_pass", torch.zeros(32), torch.full((32,), 1e-8)),
        ("zero_absolute_fail", torch.zeros(32), torch.full((32,), 3e-6)),
    ]:
        kwargs = dict(relative_tolerance=1e-4, absolute_tolerance=2e-6, exact=False, local_windows=False)
        cpu = tensor_difference(left, right, **kwargs)
        cuda = tensor_difference(left.cuda(), right.cuda(), **kwargs)
        assert cpu["passed"] == cuda["passed"], (label, cpu, cuda)
        for field in ("relative_l2", "cosine", "maximum_absolute_error", "worst_chunk_relative_l2"):
            if math.isinf(cpu[field]):
                assert cpu[field] == cuda[field]
            else:
                assert abs(cpu[field] - cuda[field]) <= 1e-12, (label, field, cpu[field], cuda[field])
        rows.append(dict(case=label, cpu=cpu, cuda=cuda))
    with args.output.open("x") as handle:
        json.dump(dict(passed=True, scope=__doc__, rows=rows, production_math_accepted=False), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
