"""Qualify available TE norm/scale kernels on actual gradient samples.

Reports bounded kernel measurements only. It neither patches MCore nor
accepts the full model trajectory or production performance.
"""

import argparse
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path

import torch


def timed(call, repetitions):
    call()
    torch.cuda.synchronize()
    times = []
    for _ in range(repetitions):
        started = time.perf_counter()
        float(call())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def qualify(records, repetitions):
    import transformer_engine.pytorch.optimizers as te
    from megatron.core.optimizer import clip_grads

    rows = []
    noop = torch.zeros(1, dtype=torch.int32, device="cuda")
    for update in range(3):
        gradients = [row["gradients"][update].cuda() for row in records]

        def fallback():
            return clip_grads.multi_tensor_applier(clip_grads.l2_norm_impl, noop, [gradients], False)[0][0]

        def fused():
            return te.multi_tensor_applier(te.multi_tensor_l2norm, noop, [gradients], False)[0][0]

        def native():
            return torch.nn.utils.get_total_norm(gradients, foreach=True)

        oracle = torch.cat([row["gradients"][update].double() for row in records]).norm().item()
        norms = {
            "installed_mcore": float(fallback()),
            "te_available": float(fused()),
            "native_foreach": float(native()),
        }
        errors = {key: abs(value - oracle) / max(oracle, 1e-30) for key, value in norms.items()}
        times = {
            "installed_mcore": timed(fallback, repetitions),
            "te_available": timed(fused, repetitions),
            "native_foreach": timed(native, repetitions),
        }
        scale_results = []
        for coefficient in (0.5, 0.008118385449051857):
            left, right = ([gradient.clone() for gradient in gradients] for _ in range(2))
            torch._foreach_mul_(left, coefficient)
            te.multi_tensor_applier(te.multi_tensor_scale, noop, [right, right], coefficient)
            exact = all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
            scale_results.append({"coefficient": coefficient, "bitexact_native_foreach": exact})
        rows.append(
            {
                "update": update,
                "fp64_sample_norm": oracle,
                "norms": norms,
                "norm_relative_errors": errors,
                "median_wall_seconds_including_scalar_read": times,
                "scale": scale_results,
                "passed": max(errors.values()) <= 1e-6 and all(row["bitexact_native_foreach"] for row in scale_results),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    return rows, {
        "installed_norm_function": clip_grads.l2_norm_impl.__name__,
        "installed_norm_module": clip_grads.l2_norm_impl.__module__,
        "te_symbols": {
            name: hasattr(te, name)
            for name in (
                "multi_tensor_applier",
                "multi_tensor_l2norm",
                "multi_tensor_scale",
                "multi_tensor_scale_tensor",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1 or args.output.exists():
        raise ValueError("Require positive repetitions and a new output")
    torch.set_num_threads(1)
    records = torch.load(args.fixture, map_location="cpu", weights_only=False)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("Tiny clipping qualification deferred: less than 8 GiB free")
    rows, backend = qualify(records, args.repetitions)
    report = {
        "passed": all(row["passed"] for row in rows),
        "scope": "811-parameter actual clipped-gradient samples, norm/scale only; concurrent GPU diagnostic, not production throughput",
        "fixture": str(args.fixture),
        "parameters": len(records),
        "elements_per_update": sum(row["gradients"][0].numel() for row in records),
        "transformer_engine_version": importlib.metadata.version("transformer-engine"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "backend": backend,
        "rows": rows,
        "full_model_trajectory_accepted": False,
        "production_performance_accepted": False,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not report["passed"]:
        raise SystemExit("Clipping backend sample qualification failed")


if __name__ == "__main__":
    main()
