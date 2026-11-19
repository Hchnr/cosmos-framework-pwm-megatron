"""Run recovered-input/VAE measurements after the active FP32 sequence."""

import importlib.util
import json
import time
from pathlib import Path

import psutil


def main():
    path = Path(__file__).with_name("run_fp32_and_microbatch_sequence_20261119.py")
    spec = importlib.util.spec_from_file_location("_sequence", path)
    sequence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sequence)
    identity = json.loads((sequence.OUTPUT / "gpu_validator_sequence_command.json").read_text())
    while True:
        events = [json.loads(line) for line in (sequence.OUTPUT / "sequence_events.jsonl").read_text().splitlines()]
        if any(event.get("cuda_validator_sequence_finished") for event in events):
            break
        try:
            process = psutil.Process(identity["pid"])
            if abs(process.create_time() - identity["create_time"]) > 0.01 or process.status() == psutil.STATUS_ZOMBIE:
                raise psutil.NoSuchProcess(identity["pid"])
        except psutil.NoSuchProcess:
            sequence.event(post_sequence_wait_failed=True, reason="FP32 coordinator exited without completion marker")
            return
        time.sleep(15)
    sequence.event(post_fp32_raw_performance_started=True)
    sequence.run(
        "fp32_native_block_control_gpu_corrected",
        sequence.distributed(
            "tools/run_tp_pp_traced_fp32_math.py",
            "--run-path",
            "tests/megatron/tp_pp_full_source_math_cuda_compare.py",
            "--trace-dir",
            sequence.OUTPUT / "math_traces/fp32_native_block_control_gpu_corrected",
            "--native-block-control",
            "--mode",
            "tp",
            "--tp",
            2,
            "--bundle",
            sequence.TP2,
            "--native-adam-arithmetic",
            "--activation-cpu-offload",
            "--observe-all-updates",
            "--objective",
            "native_rf",
            "--retain-full-text",
            "--deterministic",
            "--toml",
            sequence.TOML,
            "--prepared-dir",
            sequence.OUTPUT / "recovered_math_inputs",
            "--reference-dir",
            sequence.REFERENCE,
            "--output",
            sequence.OUTPUT / "fp32_native_block_control_gpu_corrected.json",
        ),
    )
    code = (
        'import sys,runpy;sys.path.insert(0,"tests/megatron");'
        "from tools.run_tp_pp_wide_linear_math import install_wide_linears,install_tp_checkpointing;"
        "install_wide_linears();install_tp_checkpointing();"
        f'sys.argv=["network","--output",{str(sequence.OUTPUT / "wide_linear_network_qualification.json")!r}];'
        'runpy.run_path("tests/megatron/tp_pp_network_smoke.py",run_name="__main__")'
    )
    sequence.run(
        "wide_linear_network_qualification",
        [
            sequence.PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "--no-python",
            sequence.PYTHON,
            "-c",
            code,
        ],
    )
    legacy = sequence.REPO / "outputs/megatron_tp_pp/cosmos3_action/action128/tp_pp_controlled12_fa3det_c"
    sequence.run(
        "recover_legacy_prepared_inputs",
        [
            sequence.PYTHON,
            "tools/recover_tp_pp_legacy_prepared_inputs.py",
            "--toml",
            str(sequence.TOML),
            "--raw-root",
            str(sequence.OUTPUT),
            "--runtime-checkpoint",
            str(legacy / "checkpoints/iter_000001203"),
            "--plan",
            str(legacy / "replay_plan.iter000001203.rank0.json"),
            "--output",
            str(sequence.OUTPUT / "legacy_prepared_inputs"),
        ],
        single_gpu=True,
    )
    assert json.loads((sequence.OUTPUT / "raw_microbatch_fixture/manifest.json").read_text())["passed"]
    codes = {}
    for backend, bundle in [("tp", sequence.TP4), ("fsdp", sequence.TP2)]:
        codes[backend] = sequence.run(
            f"raw_performance_{backend}",
            sequence.distributed(
                "tools/run_tp4_fsdp4_raw_microbatch_benchmark.py",
                "--raw-fixture",
                sequence.OUTPUT / "raw_microbatch_fixture",
                "--backend",
                backend,
                "--bundle",
                bundle,
                "--prepared-spool",
                sequence.SPOOL,
                "--output",
                sequence.OUTPUT / f"raw_{backend}",
                "--toml",
                sequence.TOML,
                "--updates",
                7,
                "--warmup-updates",
                2,
            ),
        )
    if all(code == 0 for code in codes.values()):
        from tools.compare_tp4_fsdp4_microbatch_benchmark import compare

        report = compare(sequence.OUTPUT / "raw_tp", sequence.OUTPUT / "raw_fsdp")
        report["scope"] = (
            "verified raw fixture copy, H2D, native VAE/noising, training/communication/optimizer; media decoding excluded"
        )
        (sequence.OUTPUT / "raw_performance_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    sequence.event(post_fp32_raw_performance_finished=True, production_math_acceptance=False)


if __name__ == "__main__":
    main()
