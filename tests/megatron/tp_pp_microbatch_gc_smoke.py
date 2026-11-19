# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Verify real MCore backward/compiled full AC/Adam with periodic cycle GC."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from cosmos_framework.training_backends.megatron_tp_pp.memory import collect_microbatch_cycles
from cosmos_framework.utils.generator.fused_adam import FusedAdam


def model():
    return torch.compile(
        checkpoint_wrapper(nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 32)).cuda()),
        fullgraph=True,
        dynamic=False,
    )


def run(interval, initial, inputs):
    from megatron.core.pipeline_parallel import schedules

    network = model()
    network.load_state_dict(initial)
    optimizer = FusedAdam(network.parameters(), lr=1e-4, capturable=True, master_weights=False)
    for parameter in network.parameters():
        optimizer.state[parameter] = {
            "exp_avg": torch.full_like(parameter, 0.01),
            "exp_avg_sq": torch.full_like(parameter, 0.02),
        }
    optimizer.param_groups[0]["step"] = torch.tensor([1200], dtype=torch.int32, device="cuda")
    config = SimpleNamespace(timers=None, grad_scale_func=None, deallocate_pipeline_outputs=False)
    original = schedules.backward_step
    records = []
    with collect_microbatch_cycles(interval) as collector:
        for update in range(3):
            optimizer.zero_grad(set_to_none=True)
            for batch in inputs:
                cycle = {"tensor": torch.empty(1 << 20, device="cuda")}
                cycle["self"] = cycle
                del cycle
                output = network(batch)
                schedules.backward_step(None, output.square().mean() / len(inputs), None, config)
            gradients = [p.grad.detach().cpu().clone() for p in network.parameters()]
            optimizer.step()
            records.append(
                {
                    "update": update,
                    "gradients": gradients,
                    "parameters": [p.detach().cpu().clone() for p in network.parameters()],
                    "exp_avg": [optimizer.state[p]["exp_avg"].cpu().clone() for p in network.parameters()],
                    "exp_avg_sq": [optimizer.state[p]["exp_avg_sq"].cpu().clone() for p in network.parameters()],
                    "step": int(optimizer.param_groups[0]["step"].item()),
                }
            )
    assert schedules.backward_step is original
    return records, collector


def main():
    from megatron.core.pipeline_parallel import schedules

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite GC qualification evidence")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("Small GC qualification deferred: less than 8 GiB free")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    template = model()
    initial = {key: tensor.cpu().clone() for key, tensor in template.state_dict().items()}
    del template
    inputs = [torch.randn(16, 64).cuda() for _ in range(2)]
    auto_gc = gc.isenabled()
    gc.disable()
    try:
        reference, _ = run(0, initial, inputs)
        gc.collect()
        candidate, collector = run(2, initial, inputs)
    finally:
        if auto_gc:
            gc.enable()
        gc.collect()
    rows = []
    for expected, actual in zip(reference, candidate, strict=True):
        fields = {
            key: all(torch.equal(a, b) for a, b in zip(expected[key], actual[key], strict=True))
            for key in ("gradients", "parameters", "exp_avg", "exp_avg_sq")
        }
        rows.append({"update": expected["update"], **fields, "step": expected["step"] == actual["step"]})
    original = schedules.backward_step
    try:
        with collect_microbatch_cycles(2):
            raise RuntimeError("intentional exception restoration check")
    except RuntimeError as error:
        assert str(error) == "intentional exception restoration check"
    exception_restored = schedules.backward_step is original
    reclaimed = sum(row["reclaimed_allocated_bytes"] for row in collector.collections)
    report = {
        "passed": all(all(value for key, value in row.items() if key != "update") for row in rows)
        and exception_restored
        and collector.completed_backwards == 6
        and len(collector.collections) == 3
        and reclaimed >= 6 * (1 << 20) * 4,
        "scope": "actual MCore backward, compiled full AC, three native nonzero-Adam updates; artificial CUDA cycles",
        "rows": rows,
        "collections": collector.collections,
        "exception_restored_schedule": exception_restored,
        "reclaimed_allocated_bytes": reclaimed,
        "production_capacity_576_slot_acceptance": False,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("Microbatch cycle GC qualification failed")


if __name__ == "__main__":
    main()
