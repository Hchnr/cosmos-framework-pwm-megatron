# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run captured UND Q diagnosis and conditional native-width intervention.

This process must be launched by a coordinator already holding the shared GPU
lock after predecessor processes terminate. It never claims full-capacity or
production acceptance. Each step writes its own isolated evidence.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = Path("/opt/cosmos-tp-pp-full-vjp-20261119")
BASE = Path("/opt/cosmos-tp-pp-kv-qualification-20261119")
REQUIRED = (
    "tools/run_tp_pp_und_first_attention_capture.py",
    "tools/compare_tp_pp_und_first_attention.py",
    "tools/run_tp_pp_full_width_und_q.py",
)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def python_hashes(root):
    return {
        str(path.relative_to(root)): sha(path)
        for directory in ("tools", "tests", "cosmos_framework")
        for path in (root / directory).rglob("*.py")
        if path.is_file()
    }


def load_queue():
    spec = importlib.util.spec_from_file_location("_und_queue_helper", DIAG / "run_after_streamed.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def case(queue, name, command, visible, directory, *, timeout=14400, extra_env=None):
    result = queue.run_case(name, command, visible, directory, timeout=timeout, extra_env=extra_env)
    if result["returncode"] != 0:
        raise RuntimeError(f"{name} failed with exit {result['returncode']}")
    return result


def same_historical_query(directory, baseline):
    import torch

    for mode, size in (("native", 1), ("tp", 4)):
        for rank in range(size):
            name = f"layer00.attention_call0.query.rank{rank}.pt"
            old = torch.load(baseline / mode / name, map_location="cpu", weights_only=False)
            new = torch.load(directory / "capture" / mode / name, map_location="cpu", weights_only=False)
            if old["shape"] != new["shape"] or not torch.equal(old["value"], new["value"]):
                raise ValueError(f"New capture does not reproduce historical query: {mode} rank {rank}")
    return True


def gate_replay(report):
    replay = report["gpu_replay"]
    projection, headwise = replay["projection"], replay["headwise"]
    if not projection["captured_native_reproduced"]["exact"] or not all(
        item["exact"] for item in projection["captured_tp_reproduced"]
    ):
        raise ValueError("GPU Q GEMM replay did not reproduce captured outputs")
    for key in ("q_norm", "rope"):
        row = headwise[key]
        if not row["captured_native_reproduced"]["exact"] or not all(
            item["exact"] for item in row["captured_tp_reproduced"]
        ):
            raise ValueError(f"GPU {key} replay did not reproduce captured outputs")
    return not projection["native_full_vs_tp_sharded"]["exact"]


def gate_tiny(directory):
    rows = []
    for rank in range(2):
        report = json.loads((directory / "tiny_reports" / f"fp32_reconstructed_kv_oracle.rank{rank}.json").read_text())
        side = json.loads((directory / "tiny_reports" / f"fp32_reconstructed_kv.rank{rank}.json").read_text())
        q = [entry for entry in side["modules"].values() if entry.get("kind") == "frozen_und_q"]
        if not (
            report["passed"]
            and side["completed"]
            and side["full_width_vjp"]
            and q
            and all(item["forward_calls"] > 0 for item in q)
        ):
            raise ValueError(f"UND Q tiny qualification failed on rank {rank}")
        rows.append(
            {"rank": rank, "und_q_modules": len(q), "max_relative_gradient_l2": report["max_relative_gradient_l2"]}
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"scope": "experimental FP32 UND Q diagnostic and conditional intervention", "passed": False}
    result_path = args.output / "summary.json"
    queue = load_queue()
    seq = queue.sequence
    clone = Path("/opt") / f"cosmos-tp-pp-und-q-{os.getpid()}"
    native_reference = Path("/opt") / f"cosmos-tp-pp-und-q-native-{os.getpid()}"
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("The outer queue must provide four distinct visible GPU UUIDs")
    try:
        result["base_snapshot"] = queue.helper.check_kv_snapshot()
        base_hashes = python_hashes(BASE)
        source_hashes = {name: sha(ROOT / name) for name in REQUIRED}
        fixed_inputs = [
            *sorted((seq.OUTPUT / "recovered_math_inputs").glob("*.pt")),
            seq.TOML,
            seq.TP2 / "manifest.json",
            seq.TP4 / "manifest.json",
            seq.OUTPUT / "native_parameter_order_from_meta.json",
        ]
        fixed_hashes = {str(path): sha(path) for path in fixed_inputs}
        result["fixed_input_sha256"] = fixed_hashes
        shutil.copytree(BASE, clone, symlinks=True)
        if python_hashes(clone) != base_hashes:
            raise RuntimeError("Isolated snapshot does not match its source")
        for name in REQUIRED:
            shutil.copy2(ROOT / name, clone / name)
        queue.SNAP = clone
        captured = args.output / "capture"
        common = [
            "--objective",
            "native_rf",
            "--retain-full-text",
            "--deterministic",
            "--toml",
            str(seq.TOML),
            "--prepared-dir",
            str(seq.OUTPUT / "recovered_math_inputs"),
        ]
        entry = [
            str(clone / REQUIRED[0]),
            "--entry",
            "standard",
            "--capture-dir",
            str(captured),
            "--forward-index",
            "0",
            "--layers",
            "0",
            "--run-path",
            str(clone / "tests/megatron/tp_pp_full_source_math_cuda_compare_correct_pp_report.py"),
        ]
        reference = Path("/opt/cosmos-tp-pp-attention-native-fp32-reference-20261119")
        result["native_capture"] = case(
            queue,
            "native_capture",
            [
                seq.PYTHON,
                *entry,
                "--trace-dir",
                str(args.output / "layer_native"),
                "--mode",
                "native",
                "--gpu-leaf-reference",
                "--preserve-saved-layout",
                "--native-device",
                "cuda",
                "--native-optimizer",
                "native_cuda",
                "--bundle",
                str(seq.TP2),
                *common,
                "--reference-dir",
                str(native_reference),
                "--output",
                str(args.output / "native.json"),
            ],
            devices[0],
            args.output,
        )
        result["tp_capture"] = case(
            queue,
            "tp_capture",
            [
                seq.PYTHON,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                *entry,
                "--trace-dir",
                str(args.output / "layer_tp4"),
                "--native-norm-order",
                str(seq.OUTPUT / "native_parameter_order_from_meta.json"),
                "--mode",
                "tp",
                "--tp",
                "4",
                "--bundle",
                str(seq.TP4),
                "--native-adam-arithmetic",
                "--activation-cpu-offload",
                *common,
                "--reference-dir",
                str(reference),
                "--output",
                str(args.output / "tp.json"),
            ],
            ",".join(devices),
            args.output,
        )
        result["historical_query_matches"] = same_historical_query(
            args.output, seq.OUTPUT / "attention_isolation_standard_fp32/forward000"
        )
        result["q_replay"] = case(
            queue,
            "q_replay",
            [
                seq.PYTHON,
                str(clone / REQUIRED[1]),
                "--capture-dir",
                str(captured),
                "--output",
                str(args.output / "q_replay.json"),
                "--gpu-replay",
            ],
            devices[0],
            args.output,
        )
        q_report = json.loads((args.output / "q_replay.json").read_text())
        if not gate_replay(q_report):
            result["skipped"] = "Captured Q difference was not reproduced by native-width versus shard-width GEMM"
            return
        tiny = [
            seq.PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(clone / REQUIRED[2]),
            "--vjp-entry",
            str(DIAG / "qualify_full_vjp_network.py"),
            "--full-ac",
            "--output",
            str(args.output / "tiny.json"),
            "--report-dir",
            str(args.output / "tiny_reports"),
        ]
        result["tiny"] = case(queue, "tiny", tiny, ",".join(devices[:2]), args.output)
        result["tiny_gate"] = gate_tiny(args.output)
        full = [
            seq.PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(clone / REQUIRED[2]),
            "--vjp-entry",
            str(DIAG / "run_full_model_vjp.py"),
            "--entry",
            "standard",
            "--capture-dir",
            str(args.output / "full_capture"),
            "--forward-indices",
            "0,1",
            "--layers",
            "0,30,31,32",
            "--run-path",
            str(clone / "tests/megatron/tp_pp_full_source_math_cuda_compare_correct_pp_report.py"),
            "--trace-dir",
            str(args.output / "full_layers"),
            "--native-norm-order",
            str(seq.OUTPUT / "native_parameter_order_from_meta.json"),
            "--mode",
            "tp",
            "--tp",
            "4",
            "--bundle",
            str(seq.TP4),
            "--native-adam-arithmetic",
            "--activation-cpu-offload",
            "--observe-all-updates",
            *common,
            "--reference-dir",
            str(reference),
            "--output",
            str(args.output / "full.json"),
        ]
        result["full"] = queue.run_case(
            "full",
            full,
            ",".join(devices),
            args.output,
            timeout=14400,
            extra_env={"FULL_VJP_REPORT_DIR": str(args.output / "full_reports")},
        )
        ranks = []
        for rank in range(4):
            report = json.loads((args.output / f"full.rank{rank}.json").read_text())
            side = json.loads((args.output / "full_reports" / f"fp32_reconstructed_kv.rank{rank}.json").read_text())
            q = [entry for entry in side["modules"].values() if entry.get("kind") == "frozen_und_q"]
            if not (
                report["updates"] == 3
                and report["continuous_trajectory"]
                and len(q) == 36
                and all(item["forward_calls"] > 0 for item in q)
            ):
                raise ValueError(f"Rank {rank} did not execute the complete UND Q experiment")
            ranks.append(
                {
                    "rank": rank,
                    "passed": report["passed"],
                    "gradient_summary": report["gradient_summary"],
                    "gate_failure_count": len(report["gate_failures"]),
                }
            )
        result["ranks"] = ranks
        result["passed"] = result["full"]["returncode"] == 0 and all(row["passed"] for row in ranks)
        result["source_unchanged"] = (
            queue.helper.check_kv_snapshot() == result["base_snapshot"]
            and {name: sha(ROOT / name) for name in REQUIRED} == source_hashes
            and {str(path): sha(path) for path in fixed_inputs} == fixed_hashes
        )
        if not result["source_unchanged"]:
            raise RuntimeError("Source or fixed inputs changed during UND Q experiment")
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
