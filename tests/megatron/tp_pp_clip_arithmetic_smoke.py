"""Compare native/MCore clipping arithmetic on immutable real gradient samples."""

import argparse
import json
from pathlib import Path

import torch

from tools.tp_pp_native_clip_arithmetic import clip_parameters, install_native_clip_arithmetic


def main():
    from megatron.core.optimizer import clip_grads
    from megatron.core.optimizer import optimizer as optimizer_module
    from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite arithmetic evidence")
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("Tiny clipping arithmetic qualification deferred: less than 8 GiB free")
    records = torch.load(args.fixture, weights_only=False, map_location="cpu")
    reference = torch.load(args.reference / "outputs.pt", weights_only=False, map_location="cpu")
    install_native_clip_arithmetic()
    if (
        clip_grads.clip_grad_by_total_norm_fp32 is not clip_parameters
        or optimizer_module.clip_grad_by_total_norm_fp32 is not clip_parameters
    ):
        raise RuntimeError("MCore optimizer's actual clipping aliases were not replaced")
    rows = []
    for update in range(3):
        native_norm = reference[update]["norm"]
        # First pair is measured at bitexact native/TP1 raw gradients; the
        # later pairs isolate arithmetic using the same actual FP32 norm.
        provided_norm = 123.16945184252262 if update == 0 else native_norm
        groups = []
        for _ in range(3):
            parameters = [
                torch.nn.Parameter(torch.zeros_like(row["gradients"][update], device="cuda")) for row in records
            ]
            for parameter, row in zip(parameters, records, strict=True):
                parameter.grad = row["gradients"][update].cuda()
            groups.append(parameters)
        torch.nn.utils.clip_grads_with_norm_(
            groups[0], 1.0, torch.tensor(native_norm, dtype=torch.float32, device="cuda"), foreach=True
        )
        clip_grad_by_total_norm_fp32(groups[1], 1.0, provided_norm)
        optimizer_module.clip_grad_by_total_norm_fp32(groups[2], 1.0, provided_norm)
        differences = [
            sum(not torch.equal(a.grad, b.grad) for a, b in zip(groups[0], candidate, strict=True))
            for candidate in groups[1:]
        ]
        row = {
            "update": update,
            "native_fp32_norm": native_norm,
            "provided_mcore_float_norm": provided_norm,
            "legacy_different_parameter_samples": differences[0],
            "candidate_different_parameter_samples": differences[1],
            "candidate_bitexact": differences[1] == 0,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "passed": all(row["candidate_bitexact"] for row in rows),
        "scope": "elementwise clipping of actual gradient samples; first uses measured identical-gradient norm pair, later use actual native norms; no full trajectory acceptance",
        "fixture": str(args.fixture),
        "parameters": len(records),
        "mcore_optimizer_bound_alias_verified": True,
        "rows": rows,
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
        "full_model_trajectory_accepted": False,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not report["passed"]:
        raise SystemExit("Native clipping arithmetic candidate qualification failed")


if __name__ == "__main__":
    main()
