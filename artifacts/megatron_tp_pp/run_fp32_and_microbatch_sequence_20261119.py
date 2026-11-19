# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""New-node isolated validation; no changes to guarded shared training sources."""

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import psutil

REPO = Path("/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm-megatron")
WORK = Path("/opt/cosmos-tp-pp-work-20261119")
OUTPUT = REPO / "outputs/megatron_tp4_fsdp4_microbatch_20261119"
PYTHON = str(REPO / ".venv/bin/python")
TP2 = Path("/share-mixed/project/eai_pwm/cache/megatron_tp_pp/canonical_iter1200_tp2_pp2_recovered_20261119")
TP4 = Path("/share-mixed/project/eai_pwm/cache/megatron_tp_pp/canonical_iter1200_tp4_pp1_20261119")
TOML = REPO / "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"
SPOOL = (
    REPO
    / "outputs/megatron_tp_pp_default_gc16_full576_strict_restart_20261118/training/cosmos3_action/action128/uninterrupted/prepared_spool/iter000001200.t4z9_emx"
)
REFERENCE = Path("/opt/cosmos-tp-pp-native-ac-fp32-reference-20261119")
ENV = os.environ.copy()
ENV.update(
    ACTION128_EPISODE_CACHE_DIR="/share-mixed/project/eai_pwm/cache/action128_episode_metadata_v1",
    NUMPY_MADVISE_HUGEPAGE="0",
    MIMALLOC_ALLOW_LARGE_OS_PAGES="0",
    LD_LIBRARY_PATH="/opt/cosmos-megatron-ffmpeg8",
    PYTHONPATH=str(WORK),
    CUDA_VISIBLE_DEVICES="0,1,2,3",
    CUDA_DEVICE_MAX_CONNECTIONS="1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="1",
    PYTORCH_ALLOC_CONF="expandable_segments:True",
    TORCHINDUCTOR_COMPILE_THREADS="1",
    COSMOS_ATTENTION_DETERMINISTIC="1",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
    TOKENIZERS_PARALLELISM="false",
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="7200",
    COSMOS_NCCL_SUBGROUP_TIMEOUT_SEC="7200",
    COSMOS_TP_PP_MICROBATCH_GC_INTERVAL="16",
    MEGATRON_TP_PP_IMPORT_PATH=str(TP2),
    HF_MODEL_PATH="/share-mixed/project/eai_pwm/models/Cosmos3-Nano-VLM",
    WAN_VAE_PATH="/share-mixed/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth",
    BASE_CHECKPOINT_PATH="/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm/outputs/pretrain52src_state_h1_nano_192gpu_ga3_from17200_run01/cosmos3_action/action128/action128_wam_nano_52src_state_h1/checkpoints/iter_000001200",
    COSMOS_TP_PP_COLLECTIVES_CANDIDATE=str(REPO / "artifacts/megatron_tp_pp/candidates/collectives_warmup_20261119.py"),
)
for key in (
    "COSMOS_TP_PP_VERIFY_RAW_REPLAY",
    "COSMOS_TP_PP_PARALLEL_PREPARATION",
    "COSMOS_TOKENIZER_AOT_CACHE_MODE",
    "COSMOS_TP_PP_PROFILE_SLOTS",
    "COSMOS_TP_PP_PROFILE_ITERATION",
):
    ENV.pop(key, None)


def event(**data):
    with (OUTPUT / "sequence_events.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(time=time.time(), host=socket.gethostname(), **data)) + "\n")


def wait_for(path):
    event(waiting_for=str(path))
    deadline = time.monotonic() + 6 * 3600
    while not path.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(path)
        time.sleep(15)


def run(label, command, *, single_gpu=False):
    environment = ENV.copy()
    if single_gpu:
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    with (OUTPUT / f"{label}.log").open("x") as log:
        process = subprocess.Popen(
            command, cwd=WORK, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        event(launched=label, pid=process.pid, create_time=psutil.Process(process.pid).create_time(), command=command)
        peak_rss = 0
        while process.poll() is None:
            try:
                members = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
                rss = sum(member.memory_info().rss for member in members if member.is_running())
                peak_rss = max(peak_rss, rss)
                if rss > 780 * 1024**3:
                    event(terminated_for_host_memory=label, rss=rss)
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=60)
            except psutil.NoSuchProcess:
                pass
            time.sleep(15)
    event(finished=label, returncode=process.returncode, peak_process_tree_rss_bytes=peak_rss)
    return process.returncode


def distributed(entry, *arguments):
    return [PYTHON, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4", entry, *map(str, arguments)]


def main():
    assert json.loads((OUTPUT / "canonical_tp4_independent_verification.json").read_text())["passed"]
    assert json.loads((OUTPUT / "native_preserved_full_ac_qualification.json").read_text())["passed"]
    for rank in range(4):
        assert json.loads((OUTPUT / f"native_block_gather_qualification.rank{rank}.json").read_text())["passed"]
        wait_for(OUTPUT / f"contract_qualification/contract.rank{rank}.json")
    wait_for(TP2 / ".save_complete")
    common = [
        "--objective",
        "native_rf",
        "--retain-full-text",
        "--deterministic",
        "--bundle",
        str(TP2),
        "--toml",
        str(TOML),
        "--prepared-dir",
        str(OUTPUT / "recovered_math_inputs"),
        "--reference-dir",
        str(REFERENCE),
    ]
    native_code = run(
        "fp32_native_ac_reference",
        [
            PYTHON,
            "tools/run_tp_pp_traced_fp32_math.py",
            "--run-path",
            "tests/megatron/tp_pp_full_source_math.py",
            "--trace-dir",
            str(OUTPUT / "math_traces/native"),
            "--mode",
            "native",
            "--gpu-leaf-reference",
            "--preserve-saved-layout",
            "--native-device",
            "cuda",
            "--native-optimizer",
            "native_cuda",
            *common,
            "--output",
            str(OUTPUT / "fp32_native_ac_reference.json"),
        ],
        single_gpu=True,
    )
    # Performance remains independent of mathematical acceptance: a failed
    # oracle must not silently cancel the explicitly requested comparison.
    wait_for(TP4 / ".save_complete")
    codes = {}
    for backend, bundle in [("tp", TP4), ("fsdp", TP2)]:
        codes[backend] = run(
            f"performance_{backend}",
            distributed(
                "tools/run_tp4_fsdp4_microbatch_benchmark.py",
                "--backend",
                backend,
                "--bundle",
                bundle,
                "--prepared-spool",
                SPOOL,
                "--output",
                OUTPUT / backend,
                "--toml",
                TOML,
                "--updates",
                7,
                "--warmup-updates",
                2,
            ),
        )
    if all(code == 0 for code in codes.values()):
        run(
            "performance_comparison",
            [
                PYTHON,
                "tools/compare_tp4_fsdp4_microbatch_benchmark.py",
                "--tp-root",
                str(OUTPUT / "tp"),
                "--fsdp-root",
                str(OUTPUT / "fsdp"),
                "--output",
                str(OUTPUT / "performance_comparison.json"),
            ],
        )
    if native_code == 0:
        for label, extra, tp_size, bundle in [
            ("fp32_tp2_standard", [], 2, TP2),
            ("fp32_native_block_control", ["--native-block-control"], 2, TP2),
            ("fp32_tp4_standard", [], 4, TP4),
        ]:
            case_common = list(common)
            case_common[case_common.index("--bundle") + 1] = str(bundle)
            run(
                label,
                distributed(
                    "tools/run_tp_pp_traced_fp32_math.py",
                    "--run-path",
                    "tests/megatron/tp_pp_full_source_math.py",
                    "--trace-dir",
                    OUTPUT / "math_traces" / label,
                    *extra,
                    "--mode",
                    "tp",
                    "--tp",
                    tp_size,
                    "--native-adam-arithmetic",
                    "--activation-cpu-offload",
                    "--observe-all-updates",
                    *case_common,
                    "--output",
                    OUTPUT / f"{label}.json",
                ),
            )
    event(sequence_finished=True, numerical_acceptance=False, independent_production_validation_still_required=True)


if __name__ == "__main__":
    main()
