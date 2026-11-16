#!/usr/bin/env python3
"""Capture a real Nano action baseline/candidate through the Parquet dataloader.

Run captures in separate processes so the two full models need not coexist.
No optimizer is constructed. Local checkpoints and LeRobot data are required.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value, **{f.name: cpu(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}
        )
    return value


def prepare_catalog(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = Path(args.source_catalog)
    rows = pq.read_table(source).to_pylist()
    if len(rows) != 1 or rows[0]["type"] != "action":
        raise ValueError("Expected the existing one-row Action Parquet catalog contract")
    data = json.loads(rows[0]["data"])
    if args.data_root:
        data["root"] = args.data_root
    if not Path(data["root"]).is_dir():
        raise FileNotFoundError(data["root"])
    data.update(chunk_length=args.chunk_length, cfg_dropout_rate=0.0, resolution="256", use_state=False)
    rows[0]["data"] = json.dumps(data)
    provenance = json.loads(rows[0]["provenance"])
    provenance["action_tower_parity"] = {
        "source_catalog": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "chunk_length": args.chunk_length,
        "use_state": False,
    }
    rows[0]["provenance"] = json.dumps(provenance)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)
    print(output)


def capture(args):
    import pyarrow.parquet as pq

    catalog_data = json.loads(pq.read_table(args.catalog).to_pylist()[0]["data"])
    assert catalog_data.get("use_state") is False, "Both baseline and candidate require explicit use_state=false"
    from cosmos_framework.configs.base.defaults.compile import CompileConfig
    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
    from cosmos_framework.data.generator.catalog_processors import build_catalog_training_dataloader
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

    # Explicit diagnostic variants are recorded in the artifact; they are never
    # silently substituted for the production capture path.
    if args.diagnostic:
        from cosmos_framework.model.generator.mot import action_tower

        backend, _ = action_tower.resolve_execution(args.execution, args.attention_backend, args.projection_mode)
        if backend != "flex":
            raise ValueError("Diagnostic Flex variants require the resolved attention backend to be flex")
        if "128" in args.diagnostic:
            original_attention = action_tower._joint_attention

            def tiled_attention(*positional, **kwargs):
                kwargs["kernel_options"] = {"BLOCK_M": 64, "BLOCK_N": 128}
                return original_attention(*positional, **kwargs)

            action_tower._joint_attention = tiled_attention
    if args.diagnostic_full_gen or (args.diagnostic and "full_routing" in args.diagnostic):
        from tools.action_tower_diagnostics import install_full_gen_diagnostic

        install_full_gen_diagnostic()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.config_file:
        from cosmos_framework.inference.common.config import deserialize_config_dict

        config = Cosmos3OmniConfig(model=deserialize_config_dict(Path(args.config_file))["model"])
    else:
        config = Cosmos3OmniConfig.from_pretrained(args.checkpoint)
    cfg = config.model["config"]
    cfg["tower_structure"] = args.towers
    initialization = args.initialization or ("copy_gen" if args.towers == "three_tower" else "checkpoint")
    cfg["action_tower_init"] = initialization
    cfg["action_tower_execution"] = args.execution
    cfg["action_tower_attention_backend"] = args.attention_backend
    cfg["action_tower_projection_mode"] = args.projection_mode
    cfg["precision"] = "bfloat16"
    cfg["tokenizer"]["vae_path"] = args.vae
    cfg["tokenizer"]["bucket_name"] = ""
    cfg["vlm_config"]["tokenizer"]["config_variant"] = "hf"
    cfg["vlm_config"]["tokenizer"]["pretrained_model_name"] = args.tokenizer
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    print("Loading complete checkpoint", flush=True)
    wrapper = Cosmos3OmniModel.from_pretrained_dcp(
        Path(args.checkpoint),
        config=config,
        parallelism_config=ParallelismConfig(data_parallel_shard_degree=1, enable_inference_mode=True),
        compile_config=CompileConfig(enabled=False),
    )
    model = wrapper.model.eval()
    trainability = None
    if args.training_toml:
        import tomllib

        from cosmos_framework.utils.generator.optimizer import _build_params_with_metadata

        recipe = tomllib.loads(Path(args.training_toml).read_text())
        optimizer = recipe["optimizer"]
        model.net.requires_grad_(True)
        _build_params_with_metadata(
            model, optimizer["keys_to_select"], optimizer.get("lr_multipliers", {}), optimizer["lr"], True
        )
        trainability = {}
        for name, parameter in model.net.named_parameters():
            if "language_model.model.layers." not in name:
                continue
            role = "GEN_ACTION" if "_moe_gen_action" in name else "GEN_VIDEO" if "_moe_gen" in name else "UND"
            assert parameter.requires_grad == (role != "UND"), (name, parameter.requires_grad)
            entry = trainability.setdefault(role, {"parameters": 0, "tensors": 0, "trainable": parameter.requires_grad})
            entry["parameters"] += parameter.numel()
            entry["tensors"] += 1
        assert {"UND", "GEN_VIDEO"}.issubset(trainability)
        if args.towers == "three_tower":
            assert "GEN_ACTION" in trainability
        print(f"Training parameter selection verified: {trainability}", flush=True)
    from cosmos_framework.model.generator.mot.action_tower import source_key

    weights = model.net.state_dict()
    copied = 0
    independent, different = 0, 0
    for key, tensor in weights.items():
        source = source_key(key)
        if source is not None and source not in weights:
            # Heterogeneous blocks have explicit conversion mappings, not the
            # same-name copy_gen mapping used by the full-width tower.
            if cfg.get("action_tower_preset", "full") == "full":
                raise KeyError(source)
            assert initialization == "checkpoint"
            assert all(
                tensor.data_ptr() != value.data_ptr()
                for name, value in weights.items()
                if "_moe_gen" in name and "_moe_gen_action" not in name
            )
            independent += 1
            continue
        if source is not None:
            equal = torch.equal(tensor, weights[source])
            if initialization == "copy_gen":
                assert equal, f"Initialization differs: {key}"
                copied += 1
            assert tensor.data_ptr() != weights[source].data_ptr(), f"Parameters aliased: {key}"
            independent += 1
            different += int(not equal)
    del weights
    if args.towers == "three_tower":
        torch.testing.assert_close(
            model.net.action_time_embedder._timestep_frequencies,
            model.net.time_embedder._timestep_frequencies,
            atol=0,
            rtol=0,
        )
    print(f"Verified {copied} independently copied parameter tensors", flush=True)
    print(f"Independent tensors={independent}; different from GEN source={different}", flush=True)
    print("Checkpoint loaded; constructing existing catalog dataloader", flush=True)
    # Caption selection also uses Python/NumPy RNG. Isolate data randomness
    # from the different number of modules initialized by each architecture.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    loader = build_catalog_training_dataloader(
        [{"path": args.catalog, "ratio": 1}],
        processor=model.vlm_processor,
        seed=args.seed,
        max_samples_per_batch=1,
        num_workers=0,
    )
    if args.batch_file:
        fixture = torch.load(args.batch_file, map_location="cpu", weights_only=False)
        iterator = iter(fixture["batches"])
        if args.samples > len(fixture["batches"]):
            raise ValueError("Requested more samples than the frozen batch file contains")
    else:
        iterator = iter(loader)
    captures = []
    routing_audit, routing_handles, expected_rows = {}, [], {}
    if cfg.get("action_tower_preset", "full") != "full":

        def pack_rows(index):
            def record(module, inputs):
                pack = inputs[0]
                expected_rows[index] = {
                    "UND": tuple(pack["causal_seq"].shape),
                    "GEN_VIDEO": tuple(pack["full_only_seq"].shape),
                    "GEN_ACTION": tuple(pack["action_seq"].shape),
                }
                assert len(pack["full_only_seq"]) + len(pack["action_seq"]) == pack["_num_full_tokens"]

            return record

        def query_rows(index, role):
            def check(module, inputs):
                actual = tuple(inputs[0].shape)
                assert actual == expected_rows[index][role], (index, role, actual, expected_rows[index])
                routing_audit.setdefault(
                    f"layer{index}.{role}", dict(input_shape=actual, query_projection_width=module.out_features)
                )

            return check

        for index, layer in enumerate(model.net.language_model.model.layers):
            routing_handles.append(layer.register_forward_pre_hook(pack_rows(index)))
            projections = {"UND": layer.self_attn.q_proj, "GEN_VIDEO": layer.self_attn.q_proj_moe_gen}
            if hasattr(layer, "action_block_moe_gen_action"):
                projections["GEN_ACTION"] = layer.action_block_moe_gen_action.q_proj
            for role, projection in projections.items():
                routing_handles.append(projection.register_forward_pre_hook(query_rows(index, role)))
    trace = {}
    trace_handles = []
    if args.trace_layers:
        from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, get_und_seq

        def save_layer(name):
            def hook(module, inputs, output):
                if name not in trace:
                    trace[name] = cpu({"und": get_und_seq(output[0]), "gen": get_gen_seq(output[0])})

            return hook

        def save_module(name):
            def hook(module, inputs, output):
                if name not in trace:
                    trace[name] = cpu({"in": inputs[0], "out": output})

            return hook

        for i, layer in enumerate(model.net.language_model.model.layers):
            trace_handles.append(layer.register_forward_hook(save_layer(f"layer{i}")))
            if i == 0:
                for name, module in layer.named_modules():
                    if name and not list(module.children()):
                        trace_handles.append(module.register_forward_hook(save_module(name)))
    for sample in range(args.samples):
        batch = next(iterator)
        assert all(not plan.robot_state_frame_indexes for plan in batch["sequence_plan"])
        assert all(
            action.shape[0] == catalog_data["chunk_length"]
            for sample_actions in batch["action"]
            for action in sample_actions
        ), "Unexpected prepended state/history token"
        print(f"Sample {sample}: state indexes={batch['sequence_plan'][0].robot_state_frame_indexes}", flush=True)
        steps, initial, packed_steps = [], [], []
        prepared = []
        original_prepare = model._prepare_inference_data

        def freeze_prepared(*positional, **kwargs):
            values = original_prepare(*positional, **kwargs)
            prepared.append(cpu(values))
            return values

        if args.freeze_inputs:
            model._prepare_inference_data = freeze_prepared

        def pre_hook(module, positional, kwargs):
            packed = kwargs.get("packed_seq", positional[0] if positional else None)
            if packed is not None:
                assert packed.action.robot_state_token_indexes.numel() == 0, "robot_state must not reach either model"
            if args.freeze_inputs and packed is not None:
                packed_steps.append(cpu(packed))
            if initial:
                return
            if packed is not None:
                initial.append(
                    {
                        "action": cpu(packed.action.tokens),
                        "vision": cpu(packed.vision.tokens),
                        "positions": cpu(packed.position_ids),
                        "text_ids": cpu(packed.text_ids),
                        "state_indexes": cpu(packed.action.robot_state_token_indexes),
                    }
                )

        def hook(module, inputs, output):
            values = cpu({k: v for k, v in output.items() if k in ("preds_action", "preds_vision")})
            for key, tensors in values.items():
                for value in tensors:
                    assert torch.isfinite(value).all(), f"Non-finite {key} at denoiser call {len(steps)}"
            steps.append(values)

        handles = [
            model.net.register_forward_pre_hook(pre_hook, with_kwargs=True),
            model.net.register_forward_hook(hook),
        ]
        with torch.inference_mode():
            result = model.generate_samples_from_batch(
                copy.deepcopy(batch),
                seed=[args.seed + sample],
                num_steps=args.steps,
                guidance=args.guidance,
                shift=args.shift,
            )
            if args.decode_video:
                result["decoded_video"] = [cpu(model.decode(value)) for value in result["vision"]]
        for handle in handles:
            handle.remove()
        model._prepare_inference_data = original_prepare
        assert len(initial) == 1 and steps, "Capture must include actual denoiser inputs and outputs"
        captures.append(
            {
                "batch": cpu(batch),
                "initial": initial,
                "steps": steps,
                "result": cpu(result),
                "packed_steps": packed_steps,
                "prepared": prepared,
            }
        )
        for handle in trace_handles:
            handle.remove()
        print(f"Sample {sample} complete: {len(steps)} forwards; action={result['action'][0].shape}", flush=True)
    for handle in routing_handles:
        handle.remove()
    if routing_audit:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).with_suffix(".routing.json").write_text(json.dumps(routing_audit, indent=2) + "\n")
    payload = {
        "routing_audit": routing_audit,
        "args": {k: v for k, v in vars(args).items() if k != "func"},
        "config": config.to_dict(),
        "captures": captures,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "catalog_sha256": hashlib.sha256(Path(args.catalog).read_bytes()).hexdigest(),
        "batch_file_sha256": hashlib.sha256(Path(args.batch_file).read_bytes()).hexdigest()
        if args.batch_file
        else None,
        "diagnostic_only": bool(args.diagnostic_full_gen or args.diagnostic),
        "copied_parameter_tensors": copied,
        "independent_parameter_tensors": independent,
        "different_parameter_tensors": different,
        "trace": trace,
        "trainability": trainability,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    if args.freeze_inputs:
        Path(args.freeze_inputs).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.freeze_inputs)
    print(f"Saved {args.output}", flush=True)


def compare(args):
    a, b = [torch.load(p, map_location="cpu", weights_only=False) for p in (args.baseline, args.candidate)]
    assert a["catalog_sha256"] == b["catalog_sha256"]
    for key in ("seed", "steps", "guidance", "samples", "checkpoint", "tokenizer", "vae"):
        assert a["args"][key] == b["args"][key], key
    rows = []

    def check(x, y, path, exact=False):
        if isinstance(x, torch.Tensor):
            assert x.shape == y.shape, path
            assert x.dtype == y.dtype, path
            diff = (x.float() - y.float()).abs()
            close = torch.isclose(x, y, atol=0 if exact else args.atol, rtol=0 if exact else args.rtol)
            bitwise = torch.equal(
                x.contiguous().reshape(-1).view(torch.uint8), y.contiguous().reshape(-1).view(torch.uint8)
            )
            row = {
                "path": path,
                "max_abs": diff.max().item() if diff.numel() else 0,
                "mean_abs": diff.mean().item() if diff.numel() else 0,
                "passed": bitwise if exact or args.atol == args.rtol == 0 else bool(close.all()),
                "bitwise_equal": bitwise,
                "mismatched": int((~close).sum()),
                "elements": x.numel(),
            }
            rows.append(row)
        elif isinstance(x, dict):
            assert x.keys() == y.keys(), path
            for key in x:
                check(x[key], y[key], f"{path}/{key}", exact)
        elif isinstance(x, list):
            assert len(x) == len(y), path
            for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
                check(xi, yi, f"{path}/{i}", exact)

    assert len(a["captures"]) == len(b["captures"])
    for i, (x, y) in enumerate(zip(a["captures"], b["captures"], strict=True)):
        check(x["batch"]["text_token_ids"], y["batch"]["text_token_ids"], f"sample{i}/text_token_ids", exact=True)
        check(x["initial"], y["initial"], f"sample{i}/initial", exact=True)
        check(x["steps"], y["steps"], f"sample{i}/steps")
        check(x["result"], y["result"], f"sample{i}/result")
    report = {
        "passed": all(row["passed"] for row in rows),
        "baseline": args.baseline,
        "candidate": args.candidate,
        "candidate_diagnostic": b["args"].get("diagnostic"),
        "candidate_diagnostic_only": b.get("diagnostic_only", False),
        "atol": args.atol,
        "rtol": args.rtol,
        "metrics": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}))
    if not report["passed"]:
        raise SystemExit(1)


def audit_checkpoint(args):
    import torch.distributed.checkpoint as dcp

    from cosmos_framework.model.generator.mot.action_tower import source_key

    reader = dcp.FileSystemReader(args.checkpoint)
    metadata = reader.read_metadata().state_dict_metadata
    new_keys = [key for key in metadata if source_key(key) is not None]
    assert new_keys, "No independent action tower found in checkpoint"
    # Cover early/late Transformer layers, state input, embedding and time MLP
    # without materializing the whole model or any optimizer state.
    selected = [
        key
        for key in new_keys
        if any(
            part in key
            for part in (
                "layers.0.self_attn.q_proj_moe_gen_action.weight",
                "layers.35.self_attn.q_proj_moe_gen_action.weight",
                "norm_moe_gen_action.weight",
                "action_time_embedder.",
            )
        )
    ]
    keys = set(selected) | {source_key(key) for key in selected}
    state = {key: torch.empty(metadata[key].size, dtype=metadata[key].properties.dtype) for key in keys}
    dcp.load(state, storage_reader=reader, no_dist=True)
    rows = []
    for key in selected:
        a, b = state[key], state[source_key(key)]
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), key
        diff = (a.float() - b.float()).abs()
        rows.append({"key": key, "max_abs_from_source": diff.max().item(), "different": not torch.equal(a, b)})
    assert any(row["different"] for row in rows), "Trained action parameters still all equal their GEN source"
    report = {"checkpoint": args.checkpoint, "additional_tensor_count": len(new_keys), "audit": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"additional_tensor_count": len(new_keys), "checked": len(rows), "passed": True}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("prepare-catalog")
    c.add_argument("--source-catalog", required=True)
    c.add_argument("--data-root")
    c.add_argument("--chunk-length", type=int, default=4)
    c.add_argument("--output", required=True)
    c.set_defaults(func=prepare_catalog)
    c = sub.add_parser("capture")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--config-file", help="Training config.yaml for a DCP model checkpoint")
    c.add_argument("--initialization", choices=["copy_gen", "checkpoint"])
    c.add_argument("--catalog", required=True)
    c.add_argument("--vae", required=True)
    c.add_argument("--tokenizer", required=True)
    c.add_argument("--towers", choices=["two_tower", "three_tower"], required=True)
    c.add_argument("--execution", choices=["native", "joint", "nano_compatible"], default="native")
    c.add_argument("--attention-backend", choices=["auto", "nano", "flex", "sdpa"], default="auto")
    c.add_argument("--projection-mode", choices=["auto", "routed"], default="routed")
    c.add_argument(
        "--diagnostic-full-gen",
        action="store_true",
        help="DIAGNOSTIC ONLY: duplicate full GEN evaluation to isolate rounding; never a production configuration",
    )
    c.add_argument("--samples", type=int, default=2)
    c.add_argument("--steps", type=int, default=4)
    c.add_argument("--guidance", type=float, default=1.5)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--shift", type=float, default=1.0)
    c.add_argument(
        "--batch-file", help="Local trusted frozen evaluation batches produced by prepare_libero_single_task"
    )
    c.add_argument("--decode-video", action="store_true", help="Include actual VAE-decoded generated video")
    c.add_argument(
        "--training-toml", help="Audit the recipe's optimizer selection on loaded parameters, without updates"
    )
    c.add_argument("--output", required=True)
    c.add_argument("--trace-layers", action="store_true")
    c.add_argument(
        "--freeze-inputs",
        help="Save explicit packed inputs for every denoiser step, including sampled noise and timesteps",
    )
    c.add_argument("--diagnostic", choices=["flex128", "full_routing128"])
    c.set_defaults(func=capture)
    c = sub.add_parser("compare")
    c.add_argument("--baseline", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--atol", type=float, default=0.02)
    c.add_argument("--rtol", type=float, default=0.01)
    c.add_argument("--output", required=True)
    c.set_defaults(func=compare)
    c = sub.add_parser("audit-checkpoint")
    c.add_argument("--checkpoint", required=True, help="DCP model/ directory")
    c.add_argument("--output", required=True)
    c.set_defaults(func=audit_checkpoint)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
