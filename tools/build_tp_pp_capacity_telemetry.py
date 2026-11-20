# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Align seed handling and add comparison telemetry in an isolated snapshot."""

import hashlib

DECODER_DTYPE_OBSERVER = '''def install_decoder_dtype_observer(stage):
    """Observe decoder boundary tensors during this math diagnostic only."""
    observations = {}
    active_update = [None]
    handles = []
    for layer_name, layer in stage.layers.items():
        def observe(_module, _inputs, output, layer_name=layer_name):
            if active_update[0] is None or not isinstance(output, tuple) or not isinstance(output[0], dict):
                raise RuntimeError("Unexpected decoder activation observation")
            pack = output[0]
            for stream in ("causal_seq", "full_only_seq", "action_seq"):
                value = pack.get(stream)
                if not torch.is_tensor(value) or not value.is_floating_point():
                    raise RuntimeError(f"Missing floating decoder stream {stream}")
                key = (active_update[0], layer_name, stream)
                entry = observations.setdefault(key, {"dtype_counts": {}, "autocast_counts": {},
                                                       "matmul_tf32_counts": {}, "cudnn_tf32_counts": {}})
                for field, observed in (
                    ("dtype_counts", str(value.dtype)),
                    ("autocast_counts", str(torch.is_autocast_enabled("cuda"))),
                    ("matmul_tf32_counts", str(torch.backends.cuda.matmul.allow_tf32)),
                    ("cudnn_tf32_counts", str(torch.backends.cudnn.allow_tf32)),
                ):
                    entry[field][observed] = entry[field].get(observed, 0) + 1
        handles.append(layer.register_forward_hook(observe))
    return observations, active_update, handles


'''


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Unexpected diagnostic helper; refusing ambiguous telemetry transformation")
    result = source.replace(old, new, 1)
    compile(result, "<capacity-telemetry>", "exec")
    return result


def comparator(source):
    anchor = '        "relative_l2": relative_l2,\n'
    return _replace_once(source, anchor, anchor + '        "reference_squared_l2": aa,\n')


def helper(source):
    anchor = '                    "maximum_absolute_error": absolute.item(),\n'
    source = _replace_once(
        source,
        anchor,
        anchor + '                    "comparison_relative_l2": difference.get("relative_l2"),\n'
        '                    "comparison_maximum_absolute_error": difference.get("maximum_absolute_error"),\n'
        '                    "reference_squared_l2": difference.get("reference_squared_l2"),\n'
        '                    "nonfinite": difference.get("nonfinite", False),\n',
    )
    source = _replace_once(
        source,
        "for actual, target in zip(output[key], expected[key], strict=True):",
        "for sample_index, (actual, target) in enumerate(zip(output[key], expected[key], strict=True)):",
    )
    source = _replace_once(
        source,
        '"slot": batch["slot_id"], "field": key, **difference',
        '"slot": batch["slot_id"], "field": key, "sample_index": sample_index, **difference',
    )
    source = _replace_once(
        source,
        "    gate_failures, compute_parameter_checks = [], []\n",
        "    gate_failures, compute_parameter_checks = [], []\n    norm_checks, step_checks = [], []\n",
    )
    source = _replace_once(
        source,
        "        if not norm_passed:\n",
        '        norm_checks.append({"update": update, "actual": float(torch.tensor(norm).item()), "reference": float(torch.tensor(records[update]["norm"]).item()), "passed": norm_passed})\n'
        "        if not norm_passed:\n",
    )
    source = _replace_once(
        source,
        "            if not step_passed:\n",
        '            step_checks.append({"update": update, "name": spec.canonical_name, "actual": int(entry.step), "reference": int(saved["step"].item()), "passed": step_passed})\n'
        "            if not step_passed:\n",
    )
    source = _replace_once(
        source,
        '                "compute_parameter_checks": compute_parameter_checks,\n',
        '                "compute_parameter_checks": compute_parameter_checks,\n'
        '                "norm_checks": norm_checks,\n                "step_checks": step_checks,\n',
    )
    source = _replace_once(
        source,
        '                "scope": reference["scope"],',
        '                "rank": dist.get_rank(),\n                "scope": reference["scope"],',
    )
    return seed_telemetry(dtype_telemetry(source))


def seed_telemetry(source):
    """Align the legacy capacity helper with the seed-aware report gate.

    This modifies only the copied diagnostic. A requested seed is evidence of
    initialization intent, not proof that native and TP RNG states match.
    """
    source = _replace_once(
        source,
        'def configuration(toml, *, original_capacity=False, attention_backend="sdpa"):',
        "def validate_reference_seed(reference, seed):\n"
        "    if (type(seed) is not int or not 0 <= seed < 2**32\n"
        '        or type(reference.get("seed")) is not int or reference["seed"] != seed):\n'
        '        raise ValueError("Native reference must record the requested seed")\n\n\n'
        'def configuration(toml, *, original_capacity=False, attention_backend="sdpa"):',
    )
    source = _replace_once(
        source,
        "def tp(args, config, language, network, inputs):\n",
        "def tp(args, config, language, network, inputs):\n"
        "    try:\n"
        '        existing_reference = json.loads((args.reference_dir / "manifest.json").read_text())\n'
        "    except (FileNotFoundError, json.JSONDecodeError):\n"
        "        pass\n"
        "    else:\n"
        "        validate_reference_seed(existing_reference, args.seed)\n",
    )
    source = _replace_once(source, "    initialize(topology)\n", "    initialize(topology, seed=args.seed)\n")
    source = _replace_once(
        source,
        '    assert reference["input_hashes"] == [batch_fingerprint(s) for s in inputs]\n',
        "    validate_reference_seed(reference, args.seed)\n"
        '    assert reference["input_hashes"] == [batch_fingerprint(s) for s in inputs]\n',
    )
    source = _replace_once(
        source,
        '    report = {\n        "passed": True,\n        "names": names,',
        '    report = {\n        "passed": True,\n        "seed": args.seed,\n'
        '        "rng_state_evidence": "not_recorded",\n        "names": names,',
    )
    source = _replace_once(
        source,
        '                "scope": reference["scope"],',
        '                "seed": args.seed,\n                "rng_state_evidence": "not_recorded",\n'
        '                "scope": reference["scope"],',
    )
    source = _replace_once(
        source,
        '    parser.add_argument("--mode", choices=("native", "tp"), required=True)\n',
        '    parser.add_argument("--mode", choices=("native", "tp"), required=True)\n'
        '    parser.add_argument("--seed", type=int, default=42)\n',
    )
    return _replace_once(
        source,
        "    if args.deterministic:\n",
        "    from cosmos_framework.utils.misc import set_random_seed\n\n"
        "    set_random_seed(args.seed)\n"
        "    if args.deterministic:\n",
    )


def dtype_telemetry(source):
    anchor = "def tp(args, config, language, network, inputs):"
    source = _replace_once(source, anchor, DECODER_DTYPE_OBSERVER + anchor)
    anchor = "    torch.backends.cuda.matmul.allow_tf32 = False\n    topology = ModelParallelConfig("
    source = _replace_once(
        source,
        anchor,
        "    torch.backends.cuda.matmul.allow_tf32 = False\n"
        "    torch.backends.cudnn.allow_tf32 = False\n    topology = ModelParallelConfig(",
    )
    anchor = "    bundle.load_optimizer(stage, driver.optimizer)\n"
    source = _replace_once(
        source,
        anchor,
        anchor + "    activation_dtypes, observed_update, dtype_hook_handles = install_decoder_dtype_observer(stage)\n",
    )
    anchor = '    loss_module = GenerationLoss(config.model.config) if args.objective == "native_rf" else None\n    for update in range(3):\n'
    source = _replace_once(source, anchor, anchor + "        observed_update[0] = update\n")
    source = _replace_once(
        source,
        '"sample_index": sample_index, **difference',
        '"sample_index": sample_index, "actual_dtype": str(actual.dtype), "reference_dtype": str(target.dtype), **difference',
    )
    anchor = '                    "comparison_relative_l2": difference.get("relative_l2"),\n'
    source = _replace_once(
        source,
        anchor,
        '                    "gradient_dtype": str(actual.dtype),\n'
        '                    "reference_dtype": str(target.dtype),\n'
        '                    "model_parameter_dtype": str(entry.model_parameter.dtype),\n' + anchor,
    )
    source = _replace_once(
        source,
        '{"update": update, "name": spec.canonical_name, "compute_equals_master": model_master_match}',
        '{"update": update, "name": spec.canonical_name, "compute_equals_master": model_master_match, "compute_dtype": str(entry.model_parameter.dtype), "master_dtype": str(entry.master.dtype)}',
    )
    source = _replace_once(
        source,
        'state_errors.append({"update": update, "name": spec.canonical_name, "field": key, **difference})',
        'state_errors.append({"update": update, "name": spec.canonical_name, "field": key, "actual_dtype": str(actual.dtype), "reference_dtype": str(target.dtype), **difference})',
    )
    anchor = '                "step_checks": step_checks,\n'
    source = _replace_once(
        source,
        anchor,
        anchor
        + '                "activation_checks": [{"update": u, "layer": layer, "stream": stream, **details} for (u, layer, stream), details in sorted(activation_dtypes.items())],\n'
        '                "native_dtype_evidence": "historical_reference_not_recorded",\n',
    )
    return source


def apply(snapshot):
    paths = {
        snapshot / "tools/compare_tp_pp_runtime_checkpoints.py": comparator,
        snapshot / "tests/megatron/tp_pp_full_source_math_cuda_compare_streamed_reference_v2.py": helper,
    }
    transformed = {}
    provenance = {}
    for path, transform in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected regular isolated snapshot file: {path}")
        source = path.read_bytes()
        transformed[path] = transform(source.decode()).encode()
        provenance[str(path.relative_to(snapshot))] = {
            "before_sha256": hashlib.sha256(source).hexdigest(),
            "after_sha256": hashlib.sha256(transformed[path]).hexdigest(),
        }
    for path, content in transformed.items():
        path.write_bytes(content)
    return provenance
