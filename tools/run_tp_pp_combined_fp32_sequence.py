# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Gate combined FP32 diagnostic on captured SDPA/row evidence and tiny GPUs."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path("/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm-megatron")
OUT = ROOT / "outputs/megatron_tp4_fsdp4_microbatch_20261119"
HERE = Path(__file__).resolve().parent
BASE = Path("/opt/cosmos-tp-pp-sdpa-replay-20261119/run_full_head_sequence.py")
SDPA = OUT / "sdpa_captured_replay_20261119T1045"
ROW = OUT / "full_width_row_capture_queue_20261119T1209"
FILES = (
    "tp_pp_native_width_column_diagnostic.py",
    "tp_pp_expand_qkv_diagnostic.py",
    "tp_pp_full_width_row_diagnostic.py",
    "tp_pp_full_head_attention_experiment.py",
    "run_tp_pp_combined_fp32_diagnostic.py",
    "qualify_tp_pp_combined_fp32_diagnostic.py",
    "run_tp_pp_combined_fp32_sequence.py",
)


def load_base():
    spec = importlib.util.spec_from_file_location("_full_head_sequence", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(base):
    paths = {str(HERE / name): hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in FILES}
    paths[str(BASE)] = hashlib.sha256(BASE.read_bytes()).hexdigest()
    return {"combined": paths, "upstream": base.tree_hash()}


def evidence(base):
    for path in (SDPA / "queue_result.json", ROW / "queue_result.json"):
        queue = json.loads(path.read_text())
        if not queue.get("completed") or queue.get("failure") or queue.get("returncode") != 0:
            return False, f"Predecessor did not complete successfully: {path}"
    ok, reason = base.sdpa_replay_decision(json.loads((SDPA / "sdpa_replay.json").read_text()))
    if not ok:
        return False, reason
    row = json.loads((ROW / "sequence/summary.json").read_text())
    if not row.get("passed") or len(row.get("cases", {})) != 3:
        return False, "Row capture or VJP replay did not pass"
    changed = any(
        not rank["native_full_vs_shard_sum_common_input"]["exact"]
        for case in row["cases"].values()
        for rank in case["ranks"]
    )
    return (
        changed,
        "SDPA and row replay passed; common-input row difference observed"
        if changed
        else "No common-input row difference observed",
    )


def tiny_gate(result, directory, world):
    if result["returncode"] != 0 or result["stop_reason"]:
        raise ValueError("Tiny process failed")
    reports = []
    for rank in range(world):
        smoke = json.loads((directory / f"tiny{world}.rank{rank}.json").read_text())
        report = json.loads(
            (directory / f"tiny{world}_reports/fp32_reconstructed_kv_oracle.rank{rank}.json").read_text()
        )
        modules = report["modules"]
        if not (
            smoke["passed"]
            and report["passed"]
            and report["full_activation_checkpointing"]
            and len(smoke["groups"]["tp"]) == world
            and report["row_forward_calls"] > 0
            and report["mlp_column_forward_calls"] > 0
            and min(report["attention"].values()) > 0
            and report["peak_allocated_bytes"] > 0
            and report["exercise"]["reference_parameter_dtypes"] == ["torch.float32"]
            and any("q_proj" in k and v["forward_calls"] for k, v in modules.items())
        ):
            raise ValueError(f"Tiny rank {rank} incomplete")
        reports.append(
            {
                "rank": rank,
                "max_relative_gradient_l2": report["max_relative_gradient_l2"],
                "peak_allocated_bytes": report["peak_allocated_bytes"],
            }
        )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "completed": False,
        "passed": False,
        "scope": "eager FP32 cropped-input diagnostic, not original-capacity or BF16 acceptance",
    }
    base = load_base()
    try:
        base.assert_lock_held()
        try:
            ok, reason = evidence(base)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.update(completed=True, skipped=f"Missing or invalid capture evidence: {exc}")
            return
        result["capture_decision"] = reason
        if not ok:
            result.update(completed=True, skipped=reason)
            return
        before = fingerprint(base)
        if before != json.loads(args.expected_sources.read_text()):
            raise RuntimeError("Source or input fingerprint changed before launch")
        result["source_sha256_before"] = before
        idle = base.queue.await_idle(timeout=3600, poll=3)
        uuids = [row["uuid"] for row in idle[-1]["inventory"]]
        python = str(ROOT / ".venv/bin/python")
        for world in (2, 4):
            command = [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={world}",
                str(HERE / "qualify_tp_pp_combined_fp32_diagnostic.py"),
                "--tp",
                str(world),
                "--full-ac",
                "--output",
                str(args.output / f"tiny{world}.json"),
                "--report-dir",
                str(args.output / f"tiny{world}_reports"),
            ]
            case = base.launch(
                command, args.output, f"tiny{world}", base.load_environment(",".join(uuids[:world])), 1800
            )
            result[f"tiny{world}"] = case
            result[f"tiny{world}_gate"] = tiny_gate(case, args.output, world)
            base.queue.await_idle(timeout=3600, poll=3)
            if fingerprint(base) != before:
                raise RuntimeError("Source changed during tiny test")
        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(HERE / "run_tp_pp_combined_fp32_diagnostic.py"),
            "--entry",
            "standard",
            "--capture-dir",
            str(args.output / "capture"),
            "--forward-indices",
            "0,1",
            "--layers",
            "0,30,31,32",
            "--run-path",
            str(base.SNAP / "tests/megatron/tp_pp_full_source_math_cuda_compare_correct_pp_report.py"),
            "--trace-dir",
            str(args.output / "layer_tp4"),
            "--native-norm-order",
            str(OUT / "native_parameter_order_from_meta.json"),
            "--mode",
            "tp",
            "--tp",
            "4",
            "--bundle",
            "/share-mixed/project/eai_pwm/cache/megatron_tp_pp/canonical_iter1200_tp4_pp1_20261119",
            "--native-adam-arithmetic",
            "--activation-cpu-offload",
            "--observe-all-updates",
            "--objective",
            "native_rf",
            "--retain-full-text",
            "--deterministic",
            "--toml",
            str(ROOT / "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"),
            "--prepared-dir",
            str(OUT / "recovered_math_inputs"),
            "--reference-dir",
            "/opt/cosmos-tp-pp-attention-native-fp32-reference-20261119",
            "--output",
            str(args.output / "tp4.json"),
        ]
        result["real36"] = base.launch(
            command, args.output, "tp4", base.load_environment(",".join(uuids), args.output / "reports"), 3 * 3600
        )
        base.queue.await_idle(timeout=3600, poll=3)
        if fingerprint(base) != before:
            raise RuntimeError("Source changed during 36-layer run")
        rows = []
        for rank in range(4):
            report = json.loads((args.output / f"tp4.rank{rank}.json").read_text())
            combined = json.loads((args.output / f"reports/combined_fp32.rank{rank}.json").read_text())
            if not (
                report.get("updates") == 3
                and report.get("continuous_trajectory")
                and not report.get("original_capacity")
                and report.get("tensor_parallel_size") == 4
                and combined.get("stage_parameter_dtypes") == ["torch.float32"]
                and combined.get("activation_dtype_checked") == "torch.float32"
                and combined.get("row_parameter_count") == 216
                and combined.get("row_forward_calls", 0) > 0
                and combined.get("mlp_column_forward_calls", 0) > 0
                and min(combined.get("attention", {}).values()) > 0
                and combined.get("peak_allocated_bytes", 0) > 0
            ):
                raise ValueError(f"Incomplete 36-layer evidence rank {rank}")
            rows.append(
                {
                    "rank": rank,
                    "passed": report["passed"],
                    "gradient_summary": report["gradient_summary"],
                    "gate_failure_count": len(report["gate_failures"]),
                    "peak_allocated_bytes": combined["peak_allocated_bytes"],
                }
            )
        result["real36_ranks"] = rows
        result["passed"] = result["real36"]["returncode"] == 0 and all(row["passed"] for row in rows)
        result["completed"] = True
    except BaseException as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
