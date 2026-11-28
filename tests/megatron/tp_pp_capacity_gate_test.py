# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only regression tests for numerical report completeness and pass flags."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.tp_pp_capacity_gate import tp_gate, validate_reports


def case(tp):
    pp = 4 // tp
    names = ["first", "second"]
    reference = {"names": names, "runtime_constants_sha256": "a" * 64, "seed": 42}
    bundle = {
        "tp": tp,
        "pp": pp,
        "complete": True,
        "parameters": {
            "net.first": {"pipeline_rank": 0, "optimizer_fields": ["master"]},
            "net.second": {"pipeline_rank": pp - 1, "optimizer_fields": ["master"]},
            "net.frozen": {"pipeline_rank": 0, "optimizer_fields": []},
        },
    }
    reports = []
    metric = {
        "passed": True,
        "relative_l2": 1e-6,
        "maximum_absolute_error": 1e-7,
        "comparison_relative_l2": 1e-6,
        "comparison_maximum_absolute_error": 1e-7,
        "reference_squared_l2": 1.0,
        "local_windows_checked": False,
        "gradient_dtype": "torch.float32",
        "model_parameter_dtype": "torch.float32",
        "actual_dtype": "torch.float32",
        "reference_dtype": "torch.float32",
    }
    for rank in range(4):
        stage = rank // tp
        local = [
            name
            for name, item in bundle["parameters"].items()
            if item["optimizer_fields"] and item["pipeline_rank"] == stage
        ]
        pairs = [(update, name) for update in range(3) for name in local]
        outputs = []
        if stage == pp - 1:
            outputs = [
                {"update": update, "slot": [update, 0, slot], "field": field, "sample_index": sample, **metric}
                for update in range(3)
                for slot in range(2)
                for field in ("preds_vision", "preds_action")
                for sample in range(2)
            ]
        reports.append(
            {
                "rank": rank,
                "seed": 42,
                "updates": 3,
                "original_capacity": True,
                "attention_backend": "flex",
                "objective": "native_rf",
                "tensor_parallel_size": tp,
                "pipeline_parallel_size": pp,
                "continuous_trajectory": True,
                "replay_native_state": False,
                "native_adam_arithmetic": True,
                "deterministic_algorithms": True,
                "activation_cpu_offload": True,
                "production_capacity_576_slot_acceptance": False,
                "runtime_constants_sha256": "a" * 64,
                "observe_all_updates": True,
                "retain_full_text": True,
                "native_dtype_evidence": "historical_reference_not_recorded",
                "passed": True,
                "pointwise_gates_passed": True,
                "gate_failures": [],
                "maximum_parameter_gradient_relative_l2": 1e-6,
                "gradient_summary": [{"relative_l2": 1e-5, "cosine": 1.0} for _ in range(3)],
                "parameter_gradient_errors": [{"update": u, "name": n, **metric} for u, n in pairs],
                "parameter_state_errors": [
                    {"update": u, "name": n, "field": field, **metric}
                    for u, n in pairs
                    for field in ("master", "exp_avg", "exp_avg_sq")
                ],
                "compute_parameter_checks": [
                    {
                        "update": u,
                        "name": n,
                        "compute_equals_master": True,
                        "compute_dtype": "torch.float32",
                        "master_dtype": "torch.float32",
                    }
                    for u, n in pairs
                ],
                "activation_checks": [
                    {
                        "update": u,
                        "layer": str(layer),
                        "stream": stream,
                        "dtype_counts": {"torch.float32": 2},
                        "autocast_counts": {"False": 2},
                        "matmul_tf32_counts": {"False": 2},
                        "cudnn_tf32_counts": {"False": 2},
                    }
                    for u in range(3)
                    for layer in range(36 * stage // pp, 36 * (stage + 1) // pp)
                    for stream in ("causal_seq", "full_only_seq", "action_seq")
                ],
                "step_checks": [
                    {"update": u, "name": n, "actual": u + 1, "reference": u + 1, "passed": True} for u, n in pairs
                ],
                "norm_checks": [{"update": u, "actual": 1.0, "reference": 1.0, "passed": True} for u in range(3)],
                "output_errors": outputs,
            }
        )
    return reports, reference, bundle


@pytest.mark.parametrize("tp", [2, 4])
def test_valid_complete_trajectory_passes(tp):
    reports, reference, bundle = case(tp)
    result = validate_reports(reports, tp, {"returncode": 0}, reference, bundle)
    assert result["complete_observation"] and result["passed"]
    assert result["rng_state_comparison_verified"] is False


def test_missing_or_different_requested_seed_cannot_pass():
    reports, reference, bundle = case(4)
    for invalid in ({}, {"seed": True}, {"seed": 41}):
        altered = dict(reference)
        if not invalid:
            altered.pop("seed")
        else:
            altered.update(invalid)
        with pytest.raises(ValueError):
            validate_reports(reports, 4, {"returncode": 0}, altered, bundle)
    reports[3]["seed"] = 43
    with pytest.raises(ValueError):
        validate_reports(reports, 4, {"returncode": 0}, reference, bundle)


def test_complete_numerical_failure_remains_a_failed_observation():
    reports, reference, bundle = case(4)
    for report in reports:
        report["passed"] = report["pointwise_gates_passed"] = False
    reports[0]["parameter_state_errors"][0].update(passed=False, relative_l2=0.1, maximum_absolute_error=1.0)
    reports[0]["gate_failures"] = ["update0.net.first.master"]
    result = validate_reports(reports, 4, {"returncode": 1}, reference, bundle)
    assert result["complete_observation"] and not result["passed"]


def test_near_zero_reference_absolute_tolerance_is_not_rejected():
    reports, reference, bundle = case(4)
    reports[0]["parameter_state_errors"][0].update(
        relative_l2=0.1, maximum_absolute_error=1e-7, reference_squared_l2=1e-17
    )
    assert validate_reports(reports, 4, {"returncode": 0}, reference, bundle)["passed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "forged_flag",
        "both_tolerances",
        "summary",
        "nonfinite",
        "missing_gradient",
        "duplicate_gradient",
        "missing_state",
        "wrong_state",
        "missing_compute",
        "failed_compute",
        "missing_output",
        "wrong_slot",
        "missing_step",
        "wrong_step",
        "missing_norm_check",
        "wrong_norm_check",
        "boundary_norm",
        "missing_activation",
        "activation_bf16",
        "autocast_on",
        "tf32_on",
        "cudnn_tf32_on",
        "gradient_bf16",
        "parameter_bf16",
        "master_bf16",
        "optimizer_bf16",
        "output_bf16",
        "native_claim",
        "missing_stage",
        "wrong_pp",
        "wrong_runtime",
        "inconsistent_rank",
        "returncode",
        "missing_failure",
        "missing_norm",
        "wrong_branch",
        "gradient_tolerance",
        "duplicated_sample",
        "wrong_rank",
    ],
)
def test_incomplete_or_contradictory_reports_cannot_pass(mutation):
    reports, reference, bundle = case(4)
    r = reports[0]
    result = {"returncode": 0}
    if mutation == "forged_flag":
        r["gate_failures"] = ["explicit_failure"]
        r["parameter_gradient_errors"][0]["passed"] = False
    elif mutation == "both_tolerances":
        r["parameter_state_errors"][0].update(relative_l2=1.0, maximum_absolute_error=1.0)
    elif mutation == "summary":
        for item in reports:
            item["gradient_summary"][0]["relative_l2"] = 1.0
    elif mutation == "nonfinite":
        r["parameter_state_errors"][0]["relative_l2"] = float("nan")
    elif mutation == "missing_gradient":
        r["parameter_gradient_errors"].pop()
    elif mutation == "duplicate_gradient":
        r["parameter_gradient_errors"][-1] = copy.deepcopy(r["parameter_gradient_errors"][0])
    elif mutation == "missing_state":
        r["parameter_state_errors"].pop()
    elif mutation == "wrong_state":
        r["parameter_state_errors"][0]["field"] = "unexpected"
    elif mutation == "missing_compute":
        r["compute_parameter_checks"].pop()
    elif mutation == "failed_compute":
        r["compute_parameter_checks"][0]["compute_equals_master"] = False
    elif mutation == "missing_output":
        r["output_errors"].pop()
    elif mutation == "wrong_slot":
        r["output_errors"][0]["slot"] = [0, 0, 99]
    elif mutation == "missing_norm":
        del r["parameter_state_errors"][0]["reference_squared_l2"]
    elif mutation == "wrong_branch":
        r["parameter_state_errors"][0].update(reference_squared_l2=1.0, relative_l2=0.1, maximum_absolute_error=1e-7)
    elif mutation == "gradient_tolerance":
        r["parameter_gradient_errors"][0]["comparison_relative_l2"] = 1.0
    elif mutation == "duplicated_sample":
        r["output_errors"][1]["sample_index"] = r["output_errors"][0]["sample_index"]
    elif mutation == "wrong_rank":
        r["rank"] = 1
    elif mutation == "missing_step":
        r["step_checks"].pop()
    elif mutation == "wrong_step":
        r["step_checks"][0]["actual"] = 999
    elif mutation == "missing_norm_check":
        r["norm_checks"].pop()
    elif mutation == "wrong_norm_check":
        r["norm_checks"][0]["actual"] = 10.0
    elif mutation == "boundary_norm":
        r["norm_checks"][0].update(reference=0.0, actual=2e-6)
    elif mutation == "missing_activation":
        r["activation_checks"].pop()
    elif mutation == "activation_bf16":
        r["activation_checks"][0]["dtype_counts"] = {"torch.bfloat16": 2}
    elif mutation == "autocast_on":
        r["activation_checks"][0]["autocast_counts"] = {"True": 2}
    elif mutation == "tf32_on":
        r["activation_checks"][0]["matmul_tf32_counts"] = {"True": 2}
    elif mutation == "cudnn_tf32_on":
        r["activation_checks"][0]["cudnn_tf32_counts"] = {"True": 2}
    elif mutation == "gradient_bf16":
        r["parameter_gradient_errors"][0]["gradient_dtype"] = "torch.bfloat16"
    elif mutation == "parameter_bf16":
        r["parameter_gradient_errors"][0]["model_parameter_dtype"] = "torch.bfloat16"
    elif mutation == "master_bf16":
        r["compute_parameter_checks"][0]["master_dtype"] = "torch.bfloat16"
    elif mutation == "optimizer_bf16":
        r["parameter_state_errors"][0]["actual_dtype"] = "torch.bfloat16"
    elif mutation == "output_bf16":
        r["output_errors"][0]["actual_dtype"] = "torch.bfloat16"
    elif mutation == "native_claim":
        r["native_dtype_evidence"] = "fp32_verified"
    elif mutation == "missing_stage":
        bundle["parameters"]["net.second"]["optimizer_fields"] = []
    elif mutation == "wrong_pp":
        r["pipeline_parallel_size"] = 2
    elif mutation == "wrong_runtime":
        r["runtime_constants_sha256"] = "b" * 64
    elif mutation == "inconsistent_rank":
        r["passed"] = r["pointwise_gates_passed"] = False
    elif mutation == "returncode":
        result["returncode"] = 1
    else:
        r["parameter_gradient_errors"][0]["passed"] = False
    with pytest.raises(ValueError):
        validate_reports(reports, 4, result, reference, bundle)


def test_gate_reads_all_four_rank_reports(tmp_path):
    reports, reference, bundle = case(4)
    ref_dir, bundle_dir = tmp_path / "reference", tmp_path / "bundle"
    ref_dir.mkdir()
    bundle_dir.mkdir()
    (ref_dir / "manifest.json").write_text(json.dumps(reference))
    (bundle_dir / "manifest.json").write_text(json.dumps(bundle))
    output = tmp_path / "tp4.json"
    for rank, report in enumerate(reports):
        output.with_name(f"tp4.rank{rank}.json").write_text(json.dumps(report))
    assert tp_gate(output, 4, {"returncode": 0}, ref_dir, bundle_dir)["passed"]
    output.with_name("tp4.rank3.json").unlink()
    with pytest.raises(FileNotFoundError):
        tp_gate(output, 4, {"returncode": 0}, ref_dir, bundle_dir)


@pytest.mark.parametrize("pp", [1, 2, 4])
def test_fp32_helper_report_pipeline_size_comes_from_runtime_topology(pp):
    # Inspect the report expression without importing the CUDA training helper.
    source = Path(__file__).with_name("tp_pp_full_source_math.py").read_text()
    module = ast.parse(source)
    values = [
        value
        for node in ast.walk(module)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "pipeline_parallel_size"
    ]
    assert len(values) == 1
    expr = compile(ast.Expression(values[0]), "<pipeline report>", "eval")
    assert eval(expr, {"topology": SimpleNamespace(pipeline_model_parallel_size=pp)}) == pp


def test_streamed_reference_builder_preserves_telemetry():
    from tools.build_tp_pp_streamed_reference_helper import transform

    source = Path(__file__).with_name("tp_pp_full_source_math.py").read_text()
    generated = transform(source)
    assert '"reference_squared_l2": difference.get("reference_squared_l2")' in generated
    assert '"sample_index": sample_index' in generated
    assert '"rank": dist.get_rank()' in generated
    assert "outputs, slot_losses = [], []" in generated
    assert '"fingerprint_version": 2' in generated


def test_telemetry_builder_rejects_wrong_input_before_writing(tmp_path):
    from tools.build_tp_pp_capacity_telemetry import apply

    compare = tmp_path / "tools/compare_tp_pp_runtime_checkpoints.py"
    helper = tmp_path / "tests/megatron/tp_pp_full_source_math_cuda_compare_streamed_reference_v2.py"
    compare.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    compare.write_text('def compare(relative_l2, aa):\n    return {\n        "relative_l2": relative_l2,\n    }\n')
    helper.write_text("# Missing the required helper anchors\n")
    before = compare.read_bytes()
    with pytest.raises(ValueError, match="ambiguous telemetry"):
        apply(tmp_path)
    assert compare.read_bytes() == before


def test_decoder_observer_records_real_torch_dtypes_without_changing_backward():
    torch = pytest.importorskip("torch")
    source = Path(__file__).with_name("tp_pp_full_source_math.py").read_text()
    module = ast.parse(source)
    func = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "install_decoder_dtype_observer")
    isolated = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(isolated, "<observer>", "exec"), namespace)

    class Layer(torch.nn.Module):
        def forward(self, x):
            return {"causal_seq": x * 2, "full_only_seq": x * 3, "action_seq": x * 4}, None, None

    stage = SimpleNamespace(layers=torch.nn.ModuleDict({"0": Layer()}))
    observations, update, handles = namespace["install_decoder_dtype_observer"](stage)
    update[0] = 0
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        x = torch.tensor([2.0], dtype=torch.float32, requires_grad=True)
        result = stage.layers["0"](x)[0]
        stage.layers["0"](x)
        sum(value.sum() for value in result.values()).backward()
        assert x.grad.item() == 9.0
        assert set(observations) == {(0, "0", key) for key in ("causal_seq", "full_only_seq", "action_seq")}
        assert observations[(0, "0", "action_seq")] == {
            "dtype_counts": {"torch.float32": 2},
            "autocast_counts": {"False": 2},
            "matmul_tf32_counts": {"False": 2},
            "cudnn_tf32_counts": {"False": 2},
        }
    finally:
        for handle in handles:
            handle.remove()
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def test_isolated_builder_observer_matches_base_diagnostic():
    from tools.build_tp_pp_capacity_telemetry import DECODER_DTYPE_OBSERVER

    source = Path(__file__).with_name("tp_pp_full_source_math.py").read_text()
    base = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "install_decoder_dtype_observer"
    )
    built = next(n for n in ast.parse(DECODER_DTYPE_OBSERVER).body if isinstance(n, ast.FunctionDef))
    assert ast.dump(base, include_attributes=False) == ast.dump(built, include_attributes=False)


def test_legacy_telemetry_rejects_current_instrumented_helper():
    from tools.build_tp_pp_capacity_telemetry import helper

    source = Path(__file__).with_name("tp_pp_full_source_math.py").read_text()
    with pytest.raises(ValueError, match="refusing ambiguous telemetry transformation"):
        helper(source)


def test_original_capacity_builder_accepts_current_helper():
    from tools.build_tp_pp_original_capacity_helper import transform

    source = Path(__file__).with_name("tp_pp_full_source_math.py").read_text()
    generated = transform(source)
    assert '"fingerprint_version": 2' in generated
    assert "initial_shapes(network, original_capacity=args.original_capacity)" in generated
