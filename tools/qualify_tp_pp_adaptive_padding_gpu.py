# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Two-GPU tiny FSDP2/full vs FSDP2/adaptive vs TP2/adaptive qualification.

This is a numerical qualification, not a throughput benchmark or full-model
acceptance. All variants consume the same synthetic packed sequence on both
ranks. Each shape/case is warmed with forward and backward before any update.
Run only in an allocated two-GPU window, for example:

  torchrun --standalone --nproc-per-node=2 tools/qualify_tp_pp_adaptive_padding_gpu.py --output /tmp/adaptive.json
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path


def audit_real_pack(pack):
    """Capture real tokens and independently verify dense sample visibility."""
    import torch

    lengths = tuple(pack["_small_real_lengths"])
    ids = tuple(value[:n].detach().cpu() for value, n in zip(pack["_small_ids"], lengths, strict=True))
    tokens = tuple(
        pack[key][:n].detach().float().cpu()
        for key, n in zip(("causal_seq", "full_only_seq", "action_seq"), lengths, strict=True)
    )
    masks = {}
    for active, query in ((False, 0), (False, 1), (True, 1), (True, 2)):
        key_streams = (0,) if query == 0 else ((0, 1, 2) if active else (0, 1))
        key_ids = torch.cat([ids[i] for i in key_streams])
        full_key_ids = torch.cat([pack["_small_ids"][i] for i in key_streams])
        query_indices = torch.nonzero(pack["_small_ids"][query] >= 0).flatten()
        key_indices = torch.nonzero(full_key_ids >= 0).flatten()
        mask = pack["_small_masks"][(active, query)].index_select(0, query_indices).index_select(1, key_indices)
        mask = mask.detach().cpu()
        expected = ids[query][:, None] == key_ids[None, :]
        if query == 0:
            row = torch.arange(len(ids[query]))
            expected &= row[:, None] >= row[None, :]
        if not torch.equal(mask, expected):
            raise AssertionError(f"Incorrect valid sample/causal mask for {(active, query)}")
        masks[f"{int(active)}:{query}"] = mask
    return {"lengths": lengths, "ids": ids, "tokens": tokens, "masks": masks}


def compare_real_packs(reference, other):
    import torch

    if reference["lengths"] != other["lengths"]:
        raise AssertionError("Real stream lengths differ")
    for left, right in zip(reference["ids"], other["ids"], strict=True):
        if not torch.equal(left, right):
            raise AssertionError("Real sample IDs differ")
    for left, right in zip(reference["tokens"], other["tokens"], strict=True):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    for name in reference["masks"]:
        if not torch.equal(reference["masks"][name], other["masks"][name]):
            raise AssertionError(f"Valid mask differs: {name}")


def audit_summary(audit):
    payload = {
        "real_lengths": list(audit["lengths"]),
        "sample_ids": [item.tolist() for item in audit["ids"]],
        "valid_masks": {key: value.tolist() for key, value in audit["masks"].items()},
    }
    payload["sample_mask_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def validate_coverage(rows, compile_events, cases):
    """Fail closed if measured cases or compiler warmups are incomplete."""
    if not compile_events:
        raise ValueError("No torch.compile events were recorded")
    modes = {"fsdp_full", "fsdp_adaptive", "tp_adaptive"}
    names = {case["name"] for case in cases}
    if len(names) != len(cases) or not names:
        raise ValueError("Cases must have unique nonempty names")
    for phase in ("warmup", "update"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        combinations = {(row["mode"], row["case"]) for row in phase_rows}
        if combinations != {(mode, case) for mode in modes for case in names} or len(phase_rows) != len(combinations):
            raise ValueError(f"Incomplete or duplicate execution in {phase}")
        for case in names:
            matching = {row["mode"]: row for row in phase_rows if row["case"] == case}
            if matching["fsdp_adaptive"]["capacity"] != matching["tp_adaptive"]["capacity"]:
                raise ValueError(f"FSDP and TP adaptive capacity differ for {case}")
    initial = [row for row in rows if row["phase"] == "warmup_full"]
    if (
        {row["mode"] for row in initial} != modes
        or len(initial) != 3
        or any(row["capacity"][0] != 8 for row in initial)
    ):
        raise ValueError("All three paths must execute the initial full-UND warmup")
    warm = {(row["mode"], row["case"], tuple(row["capacity"])) for row in rows if row["phase"] == "warmup"}
    measured = [row for row in rows if row["phase"] == "update"]
    if not measured:
        raise ValueError("No optimizer updates were recorded")
    for row in measured:
        if (row["mode"], row["case"], tuple(row["capacity"])) not in warm:
            raise ValueError(f"Unwarmed mode/case/capacity: {row['mode']}/{row['case']}/{row['capacity']}")
    by_mode = {}
    for row in measured:
        by_mode.setdefault(row["mode"], set()).add(tuple(row["capacity"]))
    if set(by_mode) != modes:
        raise ValueError("Missing a required actual execution path")
    if by_mode["fsdp_adaptive"] != by_mode["tp_adaptive"]:
        raise ValueError("FSDP and TP observed different adaptive buckets")
    if len({capacity[0] for capacity in by_mode["tp_adaptive"]}) < 2:
        raise ValueError("Short and long adaptive UND buckets must both execute")
    if any(event["phase"] == "update" for event in compile_events):
        raise ValueError("An optimizer update triggered a compile variant after warmup")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("WORLD_SIZE") != "2" or not os.environ.get("RANK", "").isdecimal():
        raise ValueError("Run with torchrun --standalone --nproc-per-node=2")
    rank = int(os.environ["RANK"])
    target = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests/megatron"))
    report = {"passed": False, "rank": rank, "full_model_acceptance": False, "failure": None}
    try:
        run(report)
    except BaseException:
        report["failure"] = traceback.format_exc()
        raise
    finally:
        target.write_text(json.dumps(report, indent=2) + "\n")


def run(report):
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.optimizer import OptimizerConfig
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, distribute_tensor
    from tp_pp_network_io_test import make_native

    from cosmos_framework.model.generator.megatron_tp_pp import io as tp_io
    from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs, LastStageOutputs
    from cosmos_framework.model.generator.megatron_tp_pp.payload import PayloadLayout
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from cosmos_framework.model.generator.mot import small_action_tower
    from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver
    from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states, initialize_adam_state
    from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import (
        canonical_local_name,
        load_canonical_stage,
        stage_parameter_manifest,
    )
    from cosmos_framework.training_backends.megatron_tp_pp.runtime import initialize, transformer_config
    from cosmos_framework.training_backends.megatron_tp_pp.transport import broadcast_batch
    from cosmos_framework.utils.generator.fused_adam import FusedAdam
    from tools.run_tp_pp_adaptive_padding import install
    from tools.tp_pp_adaptive_qualification_data import make_qualification_sequence, qualification_cases

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = False
    topology = ModelParallelConfig(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        global_packed_microbatches_per_update=1,
        logical_data_parallel_size=1,
        reference_micro_steps_per_update=1,
    )
    groups = initialize(topology)
    if (len(groups["tp"]), len(groups["pp"]), len(groups["dp"])) != (2, 1, 1):
        raise ValueError("Expected TP2/PP1/DP1")
    torch.manual_seed(421)
    source, shape = make_native()
    source = source.cuda()
    source.config.fixed_sequence_length_enabled = True
    source.config.fixed_und_tokens = 8
    source.config.fixed_gen_tokens = 10
    source.config.fixed_action_tokens = 4
    source.config.action_tower_und_short_tokens = 4
    source.language_model.model.small_action_fixed_lengths = (8, 6, 4)
    source.language_model.model.small_action_und_short_tokens = 4
    source.language_model.model._small_action_warm_full_und = True
    full, adaptive = copy.deepcopy(source), copy.deepcopy(source)
    config = transformer_config(
        topology, num_layers=2, hidden_size=64, num_attention_heads=4, variable_seq_lengths=True
    )
    stage = PipelineStage(
        config,
        lambda i: copy.deepcopy(source.language_model.model.layers[i]),
        layout=PayloadLayout(((8, 64), (6, 64), (4, 32)), context_width=96),
        input_factory=lambda: FirstStageInputs(source.config, source.language_model.config, config, shape=shape),
        output_factory=lambda: LastStageOutputs(source.config, source.language_model.config, config, shape=shape),
    ).to_empty(device="cuda")
    manifest = stage_parameter_manifest(stage)
    load_canonical_stage(stage, {"net." + name: value for name, value in source.state_dict().items()}, manifest)
    stage.input_adapter._warm_full_und = True
    for key, layer in stage.layers.items():
        stage.layers[key] = torch.compile(checkpoint_wrapper(layer), fullgraph=True, dynamic=False)
    manifest = stage_parameter_manifest(stage)
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",))
    for model in (full, adaptive):
        layers = model.language_model.model.layers
        for key, layer in layers.named_children():
            compiled = torch.compile(checkpoint_wrapper(layer), fullgraph=True, dynamic=False)
            layers.register_module(key, compiled)
            fully_shard(compiled, mesh=mesh)
        fully_shard(model, mesh=mesh)
    driver = ModelParallelDriver(
        stage,
        topology,
        OptimizerConfig(
            lr=1e-3, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.99, clip_grad=0.0, use_distributed_optimizer=True
        ),
    )
    for group in driver.optimizer.param_groups:
        group["weight_decay"] = 0.01

    def named(model):
        result = {canonical_local_name(name): param for name, param in model.named_parameters()}
        if len(result) != len(list(model.named_parameters())):
            raise ValueError("Canonical native parameter names collide")
        return result

    fsdp_params = {"fsdp_full": named(full), "fsdp_adaptive": named(adaptive)}
    source_params = dict(source.named_parameters())
    trainable = sorted(name for name, p in source_params.items() if p.requires_grad)
    initial_state = {}
    generator = torch.Generator(device="cpu").manual_seed(1147)
    for name in trainable:
        param = source_params[name]
        initial_state[name] = (
            (torch.randn(param.shape, generator=generator) * 0.01).cuda(),
            (torch.rand(param.shape, generator=generator) * 0.01 + 0.001).cuda(),
        )
    optimizers = {}
    for mode, params in fsdp_params.items():
        optimizer = FusedAdam([params[name] for name in trainable], lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01)
        optimizer.param_groups[0]["step"] = 7
        for name in trainable:
            param = params[name]
            if not isinstance(param, DTensor):
                raise TypeError(f"Native FSDP parameter is not sharded DTensor: {mode}/{name}")
            optimizer.state[param] = {
                field: distribute_tensor(value, device_mesh=param.device_mesh, placements=param.placements)
                for field, value in zip(("exp_avg", "exp_avg_sq"), initial_state[name], strict=True)
            }
        optimizers[mode] = optimizer
    entries = list(adam_parameter_states(stage, driver.optimizer))
    entry_names = {manifest[entry.name].canonical_name.removeprefix("net.") for entry in entries}
    if entry_names != set(trainable):
        raise ValueError(
            f"TP optimizer coverage mismatch: missing={set(trainable) - entry_names}, extra={entry_names - set(trainable)}"
        )
    for entry in entries:
        spec = manifest[entry.name]
        name = spec.canonical_name.removeprefix("net.")
        m, v = initial_state[name]
        initialize_adam_state(
            entry, step=7, master=spec.select(source_params[name]), exp_avg=spec.select(m), exp_avg_sq=spec.select(v)
        )

    original = small_action_tower.compact_pack
    observations = []
    install((2, 4, 8), (1, 2, 4), observations)
    adaptive_compact = small_action_tower.compact_pack
    context = {"phase": "setup", "mode": "", "case": ""}
    captures = {}
    rows = []
    events = []
    checks = []
    # Keep partial progress even when a numerical check or compilation fails.
    report.update({"rows": rows, "compile_events": events, "updates_checked": checks})

    @torch._dynamo.on_compile_start
    def on_compile(args):
        events.append({**context, "compile_id": str(getattr(args, "compile_id", None))})

    def audited_compact(*a, **kw):
        mode = context["mode"]
        pack = (original if mode == "fsdp_full" else adaptive_compact)(*a, **kw)
        captures[mode] = audit_real_pack(pack)
        capacities = [len(pack[key]) for key in ("causal_seq", "full_only_seq", "action_seq")]
        rows.append({**context, "capacity": capacities, "audit": audit_summary(captures[mode])})
        return pack

    small_action_tower.compact_pack = audited_compact
    tp_io.compact_pack = audited_compact

    def loss(output):
        return sum(value.square().mean() for key in ("preds_vision", "preds_action") for value in output[key])

    def compare_output(left, right):
        for key in ("preds_vision", "preds_action"):
            for x, y in zip(left[key], right[key], strict=True):
                torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(loss(left), loss(right), atol=2e-6, rtol=2e-5)

    def fsdp_pass(mode, sequence):
        context["mode"] = mode
        model = full if mode == "fsdp_full" else adaptive
        optimizer = optimizers[mode]
        optimizer.zero_grad(set_to_none=True)
        output = model(copy.deepcopy(sequence))
        loss(output).backward()
        return output

    def tp_pass(sequence, expected):
        context["mode"] = "tp_adaptive"
        batch = (
            {"slot_id": (0, 0, 0), "packed_sequence": copy.deepcopy(sequence)}
            if ps.get_tensor_model_parallel_rank() == 0
            else None
        )
        batch = broadcast_batch(batch, ps.get_tensor_model_parallel_group())

        def loss_fn(output, _batch):
            compare_output(expected, output)
            value = loss(output)
            return value, {"loss": value.detach()}

        driver.forward_backward(iter((batch,)), loss_fn)

    def full_value(value):
        return value.full_tensor() if isinstance(value, DTensor) else value

    def snapshot(mode, field):
        result = {}
        params = fsdp_params[mode]
        for name in trainable:
            param = params[name]
            value = (
                param.grad
                if field == "grad"
                else (param if field == "parameter" else optimizers[mode].state[param][field])
            )
            result[name] = None if value is None else full_value(value).detach().clone()
        return result

    cases = qualification_cases()
    # The first call intentionally executes the existing full-UND warmup. It
    # does not warm the first case's adaptive short bucket, so replay every
    # complete case before recording optimizer updates.
    for phase in ("warmup_full", "warmup", "update"):
        context["phase"] = phase
        phase_cases = cases[:1] if phase == "warmup_full" else cases
        for index, case in enumerate(phase_cases):
            context["case"] = case["name"]
            torch.manual_seed(5100 + index)
            sequence = make_qualification_sequence(case)
            sequence.to_cuda()
            output_full = fsdp_pass("fsdp_full", sequence)
            output_adaptive = fsdp_pass("fsdp_adaptive", sequence)
            compare_output(output_full, output_adaptive)
            tp_pass(sequence, output_full)
            compare_real_packs(captures["fsdp_full"], captures["fsdp_adaptive"])
            compare_real_packs(captures["fsdp_full"], captures["tp_adaptive"])
            if phase != "update":
                continue
            full_grads = snapshot("fsdp_full", "grad")
            adaptive_grads = snapshot("fsdp_adaptive", "grad")
            for name in trainable:
                a, b = full_grads[name], adaptive_grads[name]
                if a is None or b is None:
                    if a is not b:
                        raise AssertionError(f"FSDP None gradient mismatch for {name}")
                else:
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4, msg=f"FSDP gradient {name}")
            for entry in entries:
                spec = manifest[entry.name]
                name = spec.canonical_name.removeprefix("net.")
                expected = full_grads[name]
                if (expected is None) != (entry.model_parameter not in driver._used):
                    raise AssertionError(f"TP None gradient mismatch for {name}")
                if expected is not None:
                    torch.testing.assert_close(
                        entry.select(entry.model_parameter.main_grad),
                        entry.select(spec.select(expected)),
                        atol=2e-6,
                        rtol=1e-4,
                        msg=f"TP gradient {name}",
                    )
            optimizers["fsdp_full"].step()
            optimizers["fsdp_adaptive"].step()
            success, _, _ = driver.step()
            if not success:
                raise AssertionError("TP optimizer failed")
            driver.model.start_param_sync(force_sync=True)
            for field in ("parameter", "exp_avg", "exp_avg_sq"):
                full_state, adaptive_state = snapshot("fsdp_full", field), snapshot("fsdp_adaptive", field)
                for name in trainable:
                    torch.testing.assert_close(
                        full_state[name], adaptive_state[name], atol=2e-6, rtol=1e-4, msg=f"FSDP {field} {name}"
                    )
                for entry in entries:
                    spec = manifest[entry.name]
                    name = spec.canonical_name.removeprefix("net.")
                    actual = entry.master if field == "parameter" else entry.state[field]
                    torch.testing.assert_close(
                        actual,
                        entry.select(spec.select(full_state[name])),
                        atol=2e-6,
                        rtol=1e-4,
                        msg=f"TP {field} {name}",
                    )
            if (
                optimizers["fsdp_full"].param_groups[0]["step"] != 8 + index
                or optimizers["fsdp_adaptive"].param_groups[0]["step"] != 8 + index
            ):
                raise AssertionError("FSDP optimizer step mismatch")
            if any(entry.step != 8 + index for entry in entries):
                raise AssertionError("TP optimizer step mismatch")
            checks.append(
                {
                    "case": case["name"],
                    "parameter_names": trainable,
                    "none_gradients": [name for name in trainable if full_grads[name] is None],
                    "state_fields": ["parameter", "exp_avg", "exp_avg_sq", "step"],
                }
            )
    validate_coverage(rows, events, cases)
    report.update(
        {
            "passed": True,
            "scope": "tiny FP32 SDPA synthetic inputs, native FSDP2 full/adaptive and TP2 adaptive; no throughput claim",
            "groups": groups,
            "cases": cases,
            "rows": rows,
            "compile_events": events,
            "updates": len(cases),
            "trainable_parameters": len(trainable),
            "optimizer": "FusedAdam nonzero initial moments, no gradient clipping",
        }
    )
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
