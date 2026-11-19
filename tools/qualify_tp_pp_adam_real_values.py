"""Compare native/TE Adam arithmetic on samples of actual nonzero source state.

This isolates kernels using identical gradients. Sampled elementwise coverage
does not replace complete-model gradient or continuous trajectory acceptance.
"""

import argparse
import json
import os
from pathlib import Path

import torch


def samples(bundle, reference, count, clip_norm):
    manifest = json.loads((bundle / "manifest.json").read_text())
    native = json.loads((reference / "manifest.json").read_text())
    names = {"net." + name: index for index, name in enumerate(native["names"])}
    outputs = torch.load(reference / "outputs.pt", weights_only=False, map_location="cpu")
    records = []
    for name, entry in manifest["parameters"].items():
        if not entry["optimizer_fields"]:
            continue
        saved = torch.load(bundle / entry["shards"][0]["path"], weights_only=False, mmap=True)
        value = saved["state"]["master_param"]
        local = torch.linspace(0, value.numel() - 1, min(count, value.numel()), dtype=torch.float64).round().long()
        coordinates = torch.unravel_index(local, value.shape)
        global_indexes = torch.zeros_like(local)
        stride = 1
        for coordinate, size in zip(reversed(coordinates), reversed(entry["global_shape"]), strict=True):
            global_indexes += coordinate * stride
            stride *= size
        hyper = saved["hyperparameters"]
        row = {
            "name": name,
            "state": {
                field: saved["state"][field].reshape(-1)[local].clone()
                for field in ("master_param", "exp_avg", "exp_avg_sq")
            },
            "hyper": {field: float(hyper[field]) for field in ("lr", "eps", "weight_decay")},
            "step": int(hyper["step"].item()),
            "global_indexes": global_indexes,
            "gradients": [],
        }
        row["hyper"].update(betas=tuple(hyper["betas"]), bias_correction=bool(hyper["bias_correction"]))
        for update in range(3):
            target = torch.load(
                reference / f"update{update}.parameter{names[name]:04d}.pt", weights_only=False, mmap=True
            )
            coefficient = (torch.tensor(clip_norm) / (torch.tensor(outputs[update]["norm"]) + 1e-6)).clamp(max=1)
            row["gradients"].append(target["gradient"].reshape(-1)[global_indexes].clone() * coefficient)
        if not all(torch.isfinite(tensor).all() for tensor in [*row["state"].values(), *row["gradients"]]):
            raise ValueError(f"Nonfinite source sample: {name}")
        records.append(row)
    return records


def qualify(records):
    from transformer_engine.pytorch.optimizers import FusedAdam as TEAdam

    from cosmos_framework.utils.generator.fused_adam import FusedAdam as NativeAdam

    def build(kind):
        parameters = [torch.nn.Parameter(row["state"]["master_param"].cuda()) for row in records]
        groups = {}
        for parameter, row in zip(parameters, records, strict=True):
            key = (*row["hyper"].values(), row["step"])
            group = groups.setdefault(key, {**row["hyper"], "params": []})
            group["params"].append(parameter)
        if kind == "streamed":
            optimizers = [
                NativeAdam([{**row["hyper"], "params": [p]}], capturable=True)
                for p, row in zip(parameters, records, strict=True)
            ]
        else:
            cls = TEAdam if kind.startswith("te") else NativeAdam
            if kind == "te_capturable":
                for group in groups.values():
                    group["lr"] = torch.tensor(group["lr"], dtype=torch.float32, device="cuda")
            optimizers = [cls(list(groups.values()), capturable=kind not in ("te", "te_adapter", "te_adapter_restart"))]
        by_parameter = {parameter: row for parameter, row in zip(parameters, records, strict=True)}
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                step = by_parameter[group["params"][0]]["step"]
                group["step"] = (
                    step
                    if kind in ("te", "te_adapter", "te_adapter_restart")
                    else torch.tensor([step], dtype=torch.int32, device="cuda")
                )
                for parameter in group["params"]:
                    optimizer.state[parameter] = {
                        field: by_parameter[parameter]["state"][field].cuda() for field in ("exp_avg", "exp_avg_sq")
                    }
        if kind.startswith("te_adapter"):
            from types import SimpleNamespace

            from cosmos_framework.training_backends.megatron_tp_pp.optimizer_math import align_native_adam_arithmetic

            for optimizer in optimizers:
                align_native_adam_arithmetic(SimpleNamespace(optimizer=optimizer))
        return (
            parameters,
            optimizers,
            {p: optimizer for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]},
        )

    implementations = {
        kind: build(kind) for kind in ("streamed", "batched", "te", "te_capturable", "te_adapter", "te_adapter_restart")
    }
    reports = []
    for update in range(3):
        for parameters, optimizers, _ in implementations.values():
            for parameter, row in zip(parameters, records, strict=True):
                parameter.grad = row["gradients"][update].cuda()
            for optimizer in optimizers:
                optimizer.step()
        if update == 0:
            import io
            from types import SimpleNamespace

            from cosmos_framework.training_backends.megatron_tp_pp.optimizer_math import align_native_adam_arithmetic

            old_parameters, old_optimizers, _ = implementations["te_adapter_restart"]
            saved = io.BytesIO()
            torch.save([optimizer.state_dict() for optimizer in old_optimizers], saved)
            saved.seek(0)
            cpu_states = torch.load(saved, map_location="cpu", weights_only=False)
            rebuilt = build("te_adapter_restart")
            new_parameters, new_optimizers, _ = rebuilt
            with torch.no_grad():
                for old, new in zip(old_parameters, new_parameters, strict=True):
                    new.copy_(old)
            for optimizer, state in zip(new_optimizers, cpu_states, strict=True):
                optimizer.load_state_dict(state)
                hooks_before = len(optimizer._optimizer_step_pre_hooks)
                assert hooks_before == 1
                align_native_adam_arithmetic(SimpleNamespace(optimizer=optimizer))
                assert len(optimizer._optimizer_step_pre_hooks) == hooks_before
            implementations["te_adapter_restart"] = rebuilt
        for kind in ("batched", "te", "te_capturable", "te_adapter", "te_adapter_restart"):
            failures = []
            expected, _, left_optimizers = implementations["streamed"]
            actual, _, right_optimizers = implementations[kind]
            left_steps = {
                parameter: int(group["step"])
                for optimizer in {id(opt): opt for opt in left_optimizers.values()}.values()
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            right_steps = {
                parameter: int(group["step"])
                for optimizer in {id(opt): opt for opt in right_optimizers.values()}.values()
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            for left, right, row in zip(expected, actual, records, strict=True):
                if left_steps[left] != right_steps[right]:
                    failures.append({"name": row["name"], "field": "step"})
                for field in ("master_param", "exp_avg", "exp_avg_sq"):
                    a = left.detach() if field == "master_param" else left_optimizers[left].state[left][field]
                    b = right.detach() if field == "master_param" else right_optimizers[right].state[right][field]
                    if not torch.equal(a, b):
                        failures.append(
                            {
                                "name": row["name"],
                                "field": field,
                                "maximum_absolute_error": float((a - b).abs().max()),
                                "different_elements": int(torch.count_nonzero(a != b)),
                            }
                        )
            reports.append(
                {
                    "update": update,
                    "comparison": f"native_streamed_vs_{kind}",
                    "bitexact": not failures,
                    "differences": failures,
                }
            )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-parameter", type=int, default=257)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--fixture", type=Path, help="Reuse a previously sampled immutable CPU fixture")
    args = parser.parse_args()
    if args.samples_per_parameter < 1 or args.clip_norm <= 0 or args.output.exists():
        raise ValueError("Require positive sampling/clipping settings and a new output")
    torch.set_num_threads(2)
    records = (
        torch.load(args.fixture, map_location="cpu", weights_only=False)
        if args.fixture is not None
        else samples(args.bundle, args.reference, args.samples_per_parameter, args.clip_norm)
    )
    fixture = args.fixture or args.output.with_suffix(".samples.pt")
    if args.fixture is None:
        with fixture.open("xb") as handle:
            torch.save(records, handle)
    print(json.dumps({"prepared_real_parameters": len(records), "fixture": str(fixture)}), flush=True)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        raise RuntimeError("Real-value kernel qualification deferred: less than 16 GiB free")
    reports = qualify(records)
    report = {
        "passed": all(row["bitexact"] for row in reports),
        "scope": "sampled actual source master/moments/step and three actual gradient value distributions; elementwise Adam arithmetic only",
        "parameters": len(records),
        "sampled_elements": sum(row["state"]["master_param"].numel() for row in records),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "reports": reports,
        "full_model_trajectory_accepted": False,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "reports"}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
