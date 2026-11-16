#!/usr/bin/env python3
"""Run the v0.0.1b Cosmos3-Edge full-optimization demo on one 8-GPU node.

This is an evidence-producing runner for
``plans/v0.0.1b_cosmos3_full_opt_demo_run.md``.  It deliberately leaves the
official recipe and framework source untouched.  Every expensive phase is
create-only and can be resumed by passing the same ``--run-id``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDITED_COMMIT = "6e37e6264c869fb33cf97204edac5c72e2270910"
DEFAULT_TASK_ROOT = Path("/share/project/eai_pwm/home/hcr/cosmos3_demo/v0.0.1b")
DEFAULT_PARENT_ROOT = Path("/share/project/eai_pwm/home/hcr/cosmos3_demo/v0.0.1")
DEFAULT_PARENT_RECORD = DEFAULT_PARENT_ROOT / "records/20260818T023233Z_964feaf"
DEFAULT_LOCAL_COMPILE_CACHE_ROOT = Path("/tmp/cosmos3_full_opt_demo")
ALLOWED_BUDGETS = (45056, 32768, 24576)
PHASES = ("preflight", "dryrun", "train", "resume", "export", "inference", "finalize")

DATASET_REVISION = "40d018ac1c1a2a4b9734f17fdb21f3d933c49a01"
WAN_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
EDGE_REVISION = "9eff1178c4e6dcbed1e27c8113109a6cd6d3706e"
CRITICAL_HASHES = {
    "dataset_manifest": "67955a402ba68de2686d7c139df96c144c9774fc83fb742600aa0b042b86ca76",
    "wan_vae": "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36",
    "edge_transformer_1": "f74b228d29f844a58bef266f3afc2d695fdc7e00f0d18b618f5889966586891b",
    "edge_transformer_2": "96e9031a940699710dd11b3b81345c22418471ad82bb838abe40f4f03eb86503",
    "edge_vision": "2180ad739ecc96b5c1e9386892d3c5c08bfa42b9cdab9aabc53b028671db89b3",
    "edge_tokenizer": "4dc692a99dca6d92a44e9529ffb53541eb79bb260eec7281bc51268c05d7311c",
}


class GateError(RuntimeError):
    """A plan gate failed without mutating protected inputs."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_descendant(path: Path, root: Path, label: str) -> Path:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise GateError(f"{label} must be below {resolved_root}, got {resolved_path}")
    return resolved_path


def validate_root_boundaries(task_root: Path, parent_root: Path) -> tuple[Path, Path]:
    task = task_root.resolve(strict=False)
    parent = parent_root.resolve(strict=True)
    if task in (Path("/"), Path.home().resolve()) or len(task.parts) < 5:
        raise GateError(f"refusing broad task root: {task}")
    if parent in (Path("/"), Path.home().resolve()) or len(parent.parts) < 5:
        raise GateError(f"refusing broad parent root: {parent}")
    if task == parent or task in parent.parents or parent in task.parents:
        raise GateError(f"task and parent roots must be disjoint: task={task}, parent={parent}")
    if task.exists() and task.is_symlink():
        raise GateError(f"task root must not be a symlink: {task}")
    if parent_root.is_symlink():
        raise GateError(f"parent root must not be a symlink: {parent_root}")
    return task, parent


def validate_repository_transition(
    preflight_status: str,
    final_status: str,
    *,
    allow_committed_transition: bool,
    preflight_commit: str,
    final_commit: str,
) -> dict[str, str]:
    if final_status == preflight_status and final_commit == preflight_commit:
        return {"mode": "unchanged", "preflight_commit": preflight_commit, "final_commit": final_commit}
    if not allow_committed_transition:
        raise GateError(
            "repository changed since preflight; pass --allow-committed-repository-transition only after "
            "reviewing and committing the in-scope implementation/documentation changes"
        )
    if final_status:
        raise GateError("committed repository transition requires a clean final worktree")
    return {
        "mode": "committed_transition",
        "preflight_commit": preflight_commit,
        "final_commit": final_commit,
        "preflight_status": preflight_status,
        "final_status": final_status,
    }


def inventory_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        stat = path.stat()
        rel = path.relative_to(root).as_posix()
        digest.update(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
        count += 1
        total += stat.st_size
    return digest.hexdigest(), count, total


def run_capture(command: Sequence[str], *, env: dict[str, str] | None = None, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and result.returncode != 0:
        raise GateError(f"command failed ({result.returncode}): {shlex.join(command)}\n{result.stdout}")
    return result.stdout


def parse_selected_gpus(raw: str) -> list[str]:
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    if len(selected) != 8 or len(set(selected)) != 8 or any(not item.isdigit() for item in selected):
        raise GateError(f"--gpus must contain exactly 8 unique numeric indices, got {raw!r}")
    return selected


def query_gpus() -> list[dict[str, str]]:
    output = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 6:
            raise GateError(f"unexpected nvidia-smi GPU row: {row!r}")
        rows.append(
            dict(
                index=row[0].strip(),
                uuid=row[1].strip(),
                name=row[2].strip(),
                memory_total_mib=row[3].strip(),
                memory_used_mib=row[4].strip(),
                utilization_gpu_pct=row[5].strip(),
            )
        )
    return rows


def query_compute_apps() -> list[dict[str, str]]:
    output = run_capture(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) == 4:
            rows.append(dict(uuid=row[0].strip(), pid=row[1].strip(), name=row[2].strip(), memory_mib=row[3].strip()))
    return rows


def assert_gpus_idle(selected: Sequence[str], max_idle_memory_mib: int) -> list[dict[str, str]]:
    all_gpus = {row["index"]: row for row in query_gpus()}
    missing = sorted(set(selected) - set(all_gpus))
    if missing:
        raise GateError(f"selected GPUs not found: {missing}")
    chosen = [all_gpus[index] for index in selected]
    non_h100 = [row for row in chosen if "H100" not in row["name"]]
    if non_h100:
        raise GateError(f"all selected GPUs must be H100: {non_h100}")
    busy_memory = [row for row in chosen if int(row["memory_used_mib"]) > max_idle_memory_mib]
    selected_uuids = {row["uuid"] for row in chosen}
    foreign_apps = [row for row in query_compute_apps() if row["uuid"] in selected_uuids]
    if busy_memory or foreign_apps:
        raise GateError(f"selected GPUs are not idle: memory={busy_memory}, compute_apps={foreign_apps}")
    return chosen


def nested_get(config: dict[str, Any], dotted: str) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise GateError(f"resolved config is missing {dotted!r} at {part!r}")
        current = current[part]
    return current


def unresolved_strings(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(unresolved_strings(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(unresolved_strings(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and ("${" in value or "???" in value):
        found.append(f"{prefix}={value}")
    return found


def job_name(budget: int, attempt_tag: str | None = None) -> str:
    if budget not in ALLOWED_BUDGETS:
        raise GateError(f"packing budget must be one of {ALLOWED_BUDGETS}, got {budget}")
    base = f"vision_sft_edge_v0_0_1b_full_opt_single_node_hsdp_r2_s4_cp2_seq{budget}_full_params"
    if attempt_tag:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", attempt_tag):
            raise GateError(f"invalid attempt tag: {attempt_tag!r}")
        return f"{base}__{attempt_tag}"
    return base


@dataclass(frozen=True)
class InputPaths:
    parent_root: Path
    parent_record: Path
    dataset: Path
    edge_hf: Path
    base_dcp: Path
    wan_vae: Path


@dataclass
class RunContext:
    args: argparse.Namespace
    task_root: Path
    inputs: InputPaths
    run_id: str
    record_dir: Path
    job_name: str
    run_dir: Path
    python: Path
    torchrun: Path
    compile_cache_root: Path

    @property
    def state_file(self) -> Path:
        return self.record_dir / "state.json"

    @property
    def commands_file(self) -> Path:
        return self.record_dir / "commands.log"

    def environment(self, output_root: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": self.args.gpus,
                "DATASET_PATH": str(self.inputs.dataset),
                "EDGE_HF_PATH": str(self.inputs.edge_hf),
                "BASE_CHECKPOINT_PATH": str(self.inputs.base_dcp),
                "WAN_VAE_PATH": str(self.inputs.wan_vae),
                "IMAGINAIRE_OUTPUT_ROOT": str(output_root),
                "HF_HOME": str(self.task_root / "cache/huggingface"),
                "TORCHINDUCTOR_CACHE_DIR": str(self.task_root / "cache/torchinductor"),
                "TRITON_CACHE_DIR": str(self.task_root / "cache/triton"),
                "COSMOS_COMPILE_CACHE_ROOT": str(self.compile_cache_root),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONHASHSEED": "42",
                "PYTHONDONTWRITEBYTECODE": "1",
                "LD_LIBRARY_PATH": "",
                "WANDB_RUN_ID": self.run_id,
            }
        )
        return env


def build_context(args: argparse.Namespace) -> RunContext:
    task_root, parent_root = validate_root_boundaries(args.task_root, args.parent_root)
    parent_record = args.parent_record.resolve(strict=True)
    assert_descendant(parent_record, parent_root, "parent record")
    inputs = InputPaths(
        parent_root=parent_root,
        parent_record=parent_record,
        dataset=assert_descendant(
            parent_root / "data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge",
            parent_root,
            "dataset",
        ),
        edge_hf=assert_descendant(parent_root / "checkpoints/hf/Cosmos3-Edge", parent_root, "Edge HF"),
        base_dcp=assert_descendant(parent_root / "checkpoints/dcp/Cosmos3-Edge", parent_root, "base DCP"),
        wan_vae=assert_descendant(parent_root / "checkpoints/wan22_vae/Wan2.2_VAE.pth", parent_root, "Wan VAE"),
    )
    for label, path in asdict(inputs).items():
        if label not in ("parent_root", "parent_record") and not Path(path).exists():
            raise GateError(f"missing immutable input {label}: {path}")

    current_commit = run_capture(["git", "rev-parse", "HEAD"]).strip()
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{current_commit[:7]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise GateError(f"invalid --run-id: {run_id!r}")
    name = job_name(args.packing_token_budget, args.attempt_tag)
    record_dir = assert_descendant(task_root / "records" / run_id, task_root, "record directory")
    run_dir = assert_descendant(task_root / "outputs/train/cosmos3/sft" / name, task_root, "training run")
    local_compile_base = args.local_compile_cache_root.resolve(strict=False)
    if local_compile_base in (Path("/"), Path("/tmp")) or len(local_compile_base.parts) < 3:
        raise GateError(f"refusing broad local compile cache root: {local_compile_base}")
    if args.local_compile_cache_root.is_symlink():
        raise GateError(f"local compile cache root must not be a symlink: {args.local_compile_cache_root}")
    compile_cache_root = assert_descendant(
        local_compile_base / run_id,
        local_compile_base,
        "per-run local compile cache",
    )
    return RunContext(
        args=args,
        task_root=task_root,
        inputs=inputs,
        run_id=run_id,
        record_dir=record_dir,
        job_name=name,
        run_dir=run_dir,
        python=REPO_ROOT / ".venv/bin/python",
        torchrun=REPO_ROOT / ".venv/bin/torchrun",
        compile_cache_root=compile_cache_root,
    )


def load_state(ctx: RunContext) -> dict[str, Any]:
    if not ctx.state_file.exists():
        return {
            "run_id": ctx.run_id,
            "budget": ctx.args.packing_token_budget,
            "job_name": ctx.job_name,
            "phases": {},
        }
    state = json.loads(ctx.state_file.read_text())
    assert_equal(state.get("run_id"), ctx.run_id, "state run_id")
    assert_equal(state.get("budget"), ctx.args.packing_token_budget, "state packing budget")
    assert_equal(state.get("job_name"), ctx.job_name, "state job name")
    return state


def save_state(ctx: RunContext, state: dict[str, Any]) -> None:
    temp = ctx.state_file.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temp.replace(ctx.state_file)


def phase_completed(ctx: RunContext, phase: str) -> bool:
    return load_state(ctx).get("phases", {}).get(phase, {}).get("status") == "PASS"


def require_phases(ctx: RunContext, *phases: str) -> None:
    state = load_state(ctx)
    missing = [phase for phase in phases if state.get("phases", {}).get(phase, {}).get("status") != "PASS"]
    if missing:
        raise GateError(f"phase prerequisite not satisfied: {missing}; use the same --run-id {ctx.run_id!r}")


def mark_phase(ctx: RunContext, phase: str, status: str, **details: Any) -> None:
    state = load_state(ctx)
    state.setdefault("phases", {})[phase] = {"status": status, "time_utc": utc_now(), **details}
    save_state(ctx, state)


def append_command(ctx: RunContext, command: Sequence[str], started: str, returncode: int) -> None:
    with ctx.commands_file.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "started_utc": started,
                    "ended_utc": utc_now(),
                    "cwd": str(REPO_ROOT),
                    "argv": list(command),
                    "returncode": returncode,
                },
                sort_keys=True,
            )
            + "\n"
        )


def training_overrides(ctx: RunContext) -> list[str]:
    budget = ctx.args.packing_token_budget
    return [
        f"job.name={ctx.job_name}",
        f"model.config.max_num_tokens_after_packing={budget}",
        "model.config.joint_attn_implementation=two_way",
        "model.config.ema.enabled=True",
        "model.config.lora_enabled=False",
        "model.config.precision=bfloat16",
        "model.config.parallelism.data_parallel_shard_degree=4",
        "model.config.parallelism.data_parallel_replicate_degree=2",
        "model.config.parallelism.context_parallel_shard_degree=2",
        "model.config.parallelism.cfg_parallel_shard_degree=1",
        "model.config.parallelism.fsdp_master_dtype=float32",
        "model.config.attention_io_layout=sequence_sharded",
        "model.config.compile.enabled=True",
        "model.config.compile.compile_dynamic=True",
        "model.config.compile.compiled_region=all",
        "model.config.compile.max_autotune_pointwise=True",
        "model.config.compile.coordinate_descent_tuning=True",
        "model.config.compile.use_cuda_graphs=False",
        "model.config.activation_checkpointing.mode=selective",
        "model.config.activation_checkpointing.save_ops_regex=[fmha,mm,bmm,addmm]",
        "model.config.activation_checkpointing.preserve_rng_state=True",
        "model.config.activation_checkpointing.determinism_check=default",
        "optimizer.fused=True",
        "optimizer.keys_to_select=[]",
        "optimizer.lr=0.0001",
        "trainer.distributed_parallelism=fsdp",
        "trainer.grad_accum_iter=2",
        "trainer.logging_iter=1",
        "trainer.callbacks.compile_tokenizer.enabled=True",
        "trainer.callbacks.compile_tokenizer.compile_after_iterations=3",
        "trainer.callbacks.compile_tokenizer.warmup_resolutions=[256]",
        "trainer.callbacks.sampled_media.enabled=True",
        "trainer.callbacks.sampled_media.flush_every_n_batches=10",
        "trainer.callbacks.sampled_media.record_caption=False",
        f"dataloader_train.max_sequence_length={budget}",
        "dataloader_train.max_samples_per_batch=null",
        "dataloader_train.dataloader.num_workers=4",
        "dataloader_train.dataloader.persistent_workers=True",
        "dataloader_train.dataloader.pin_memory=True",
        "dataloader_train.dataloader.prefetch_factor=4",
        "checkpoint.dcp_async_mode_enabled=True",
        "checkpoint.dcp_load_dedup=True",
        "checkpoint.save_iter=2",
        "checkpoint.keys_to_skip_loading=[net_ema.]",
        "checkpoint.strict_resume=True",
        "model.config.vlm_config.tokenizer.repository=null",
        "model.config.vlm_config.tokenizer.revision=null",
        "model.config.vlm_config.tokenizer.subdir=null",
        f"+model.config.vlm_config.tokenizer.tokenizer_type={ctx.inputs.edge_hf}",
    ]


def _descendant_pids(root_pid: int) -> list[int]:
    ppid_map: dict[int, int] = {}
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = {}
            for line in status.read_text().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key] = value.strip()
            ppid_map[int(status.parent.name)] = int(fields.get("PPid", "-1"))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in ppid_map.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return sorted(selected)


def _sample_processes(root_pid: int) -> tuple[int, int, int, int, str]:
    rss_kib = read_bytes = write_bytes = cpu_ticks = 0
    alive: list[str] = []
    for pid in _descendant_pids(root_pid):
        proc = Path("/proc") / str(pid)
        try:
            stat = (proc / "stat").read_text().split()
            status = (proc / "status").read_text()
            io_text = (proc / "io").read_text()
            cpu_ticks += int(stat[13]) + int(stat[14])
            rss_match = re.search(r"^VmRSS:\s+(\d+) kB", status, re.MULTILINE)
            read_match = re.search(r"^read_bytes:\s+(\d+)", io_text, re.MULTILINE)
            write_match = re.search(r"^write_bytes:\s+(\d+)", io_text, re.MULTILINE)
            rss_kib += int(rss_match.group(1)) if rss_match else 0
            read_bytes += int(read_match.group(1)) if read_match else 0
            write_bytes += int(write_match.group(1)) if write_match else 0
            alive.append(str(pid))
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
    return rss_kib, cpu_ticks, read_bytes, write_bytes, ";".join(alive)


def monitor_process(root_pid: int, stop: threading.Event, gpu_csv: Path, process_csv: Path, selected: set[str]) -> None:
    gpu_fields = [
        "timestamp",
        "index",
        "uuid",
        "memory.used",
        "utilization.gpu",
        "power.draw",
        "temperature.gpu",
    ]
    with gpu_csv.open("x", newline="") as gpu_handle, process_csv.open("x", newline="") as proc_handle:
        gpu_writer = csv.writer(gpu_handle)
        proc_writer = csv.writer(proc_handle)
        gpu_writer.writerow(gpu_fields)
        proc_writer.writerow(["timestamp_utc", "rss_kib", "cpu_ticks", "read_bytes", "write_bytes", "pids"])
        next_process_sample = 0.0
        while not stop.is_set():
            now = utc_now()
            query = [
                "nvidia-smi",
                "--query-gpu=" + ",".join(gpu_fields[1:]),
                "--format=csv,noheader,nounits",
            ]
            try:
                output = run_capture(query)
                for row in csv.reader(output.splitlines()):
                    if row and row[0].strip() in selected:
                        gpu_writer.writerow([now, *(item.strip() for item in row)])
                gpu_handle.flush()
            except GateError as error:
                gpu_writer.writerow([now, "MONITOR_ERROR", str(error)])
                gpu_handle.flush()
            monotonic = time.monotonic()
            if monotonic >= next_process_sample:
                proc_writer.writerow([now, *_sample_processes(root_pid)])
                proc_handle.flush()
                next_process_sample = monotonic + 5
            stop.wait(2)


def run_logged(
    ctx: RunContext,
    command: Sequence[str],
    log_path: Path,
    env: dict[str, str],
    *,
    monitor_prefix: str | None = None,
) -> int:
    if log_path.exists():
        raise GateError(f"create-only log already exists: {log_path}")
    started = utc_now()
    log_handle = log_path.open("x", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        raise
    stop = threading.Event()
    monitor_thread = None
    if monitor_prefix:
        gpu_csv = ctx.record_dir / f"gpu_{monitor_prefix}.csv"
        process_csv = ctx.record_dir / f"process_{monitor_prefix}.csv"
        monitor_thread = threading.Thread(
            target=monitor_process,
            args=(process.pid, stop, gpu_csv, process_csv, set(parse_selected_gpus(ctx.args.gpus))),
            daemon=True,
        )
        monitor_thread.start()
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        returncode = process.wait()
        raise
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Do not escalate to a broader/stronger signal automatically.  The
                # exact task-owned process group is left visible for diagnosis.
                pass
        raise
    finally:
        log_handle.close()
        stop.set()
        if monitor_thread:
            monitor_thread.join(timeout=10)
        append_command(ctx, command, started, process.returncode if process.returncode is not None else -1)
    return returncode


def validate_parent_inputs(ctx: RunContext) -> dict[str, Any]:
    before = inventory_digest(ctx.inputs.parent_root)
    manifest = ctx.inputs.dataset / "train/video_dataset_file.jsonl"
    assert_equal(sha256_file(manifest), CRITICAL_HASHES["dataset_manifest"], "dataset JSONL SHA256")
    assert_equal(sha256_file(ctx.inputs.wan_vae), CRITICAL_HASHES["wan_vae"], "Wan VAE SHA256")

    hf_checks = {
        "edge_transformer_1": ctx.inputs.edge_hf / "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "edge_transformer_2": ctx.inputs.edge_hf / "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "edge_vision": ctx.inputs.edge_hf / "vision_encoder/model.safetensors",
        "edge_tokenizer": ctx.inputs.edge_hf / "tokenizer.json",
    }
    for label, path in hf_checks.items():
        assert_equal(sha256_file(path), CRITICAL_HASHES[label], f"{label} SHA256")

    dcp_hash_file = ctx.inputs.parent_record / "dcp_artifacts.sha256"
    for raw_line in dcp_hash_file.read_text().splitlines():
        expected, raw_path = raw_line.split(maxsplit=1)
        path = Path(raw_path)
        assert_descendant(path, ctx.inputs.parent_root, "DCP checksum target")
        assert_equal(sha256_file(path), expected, f"DCP SHA256 {path.name}")

    parent_manifest = (ctx.inputs.parent_record / "run_manifest.md").read_text()
    for revision in (DATASET_REVISION, WAN_REVISION, EDGE_REVISION):
        if revision not in parent_manifest:
            raise GateError(f"parent manifest does not contain pinned revision {revision}")

    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            raw_path = Path(row["vision_path"])
            if raw_path.is_absolute() or "://" in row["vision_path"]:
                raise GateError(f"dataset line {line_number} has non-relative vision_path: {raw_path}")
            video = (manifest.parent / raw_path).resolve(strict=True)
            assert_descendant(video, ctx.inputs.dataset, f"dataset line {line_number} video")
            rows.append((line_number, row, video))
    assert_equal(len(rows), 1222, "dataset row count")
    sample_indexes = sorted({round(index * (len(rows) - 1) / 19) for index in range(20)})
    probes = []
    for index in sample_indexes:
        line_number, row, video = rows[index]
        output = run_capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames:format=duration",
                "-of",
                "json",
                str(video),
            ]
        )
        probe = json.loads(output)
        if not probe.get("streams") or float(probe.get("format", {}).get("duration", 0)) <= 0:
            raise GateError(f"ffprobe failed semantic validation for {video}")
        probes.append({"line": line_number, "uuid": row["uuid"], "video": str(video)})

    after = inventory_digest(ctx.inputs.parent_root)
    assert_equal(after, before, "parent input inventory before/after validation")
    return {"inventory_sha256": before[0], "file_count": before[1], "total_bytes": before[2], "probes": probes}


def phase_preflight(ctx: RunContext) -> None:
    if ctx.record_dir.exists():
        if phase_completed(ctx, "preflight"):
            print(f"preflight already PASS for run_id={ctx.run_id}")
            return
        raise GateError(f"record directory already exists without a completed preflight: {ctx.record_dir}")

    actual_commit = run_capture(["git", "rev-parse", "HEAD"]).strip()
    assert_equal(actual_commit, ctx.args.expected_commit, "audited repository commit")
    selected = parse_selected_gpus(ctx.args.gpus)
    gpu_rows = assert_gpus_idle(selected, ctx.args.max_idle_memory_mib)
    if shutil.disk_usage(ctx.task_root.parent).free < ctx.args.min_free_gib * 1024**3:
        raise GateError(f"task filesystem has less than {ctx.args.min_free_gib} GiB free")
    for executable in (ctx.python, ctx.torchrun):
        if not executable.is_file():
            raise GateError(f"missing repo-local executable: {executable}")

    ctx.record_dir.mkdir(parents=True, exist_ok=False)
    for directory in (
        ctx.task_root / "cache/huggingface",
        ctx.task_root / "cache/torchinductor",
        ctx.task_root / "cache/triton",
        ctx.task_root / "outputs/dryrun",
        ctx.task_root / "outputs/train",
        ctx.task_root / "outputs/inference",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    env = ctx.environment(ctx.task_root / "outputs/dryrun")
    versions = run_capture(
        [
            str(ctx.python),
            "-c",
            (
                "import importlib.metadata as m;import platform,torch,flash_attn,natten,transformer_engine,cosmos_framework;"
                "print('python='+platform.python_version());print('torch='+torch.__version__);"
                "print('cuda='+str(torch.version.cuda));print('flash_attn='+flash_attn.__version__);"
                "print('natten='+natten.__version__);print('transformer_engine='+transformer_engine.__version__);"
                "print('cosmos_framework='+m.version('cosmos-framework'))"
            ),
        ],
        env=env,
    )
    smoke_results = []
    for gpu in selected:
        gpu_env = dict(env)
        gpu_env["CUDA_VISIBLE_DEVICES"] = gpu
        smoke = run_capture(
            [
                str(ctx.python),
                "-c",
                (
                    "import torch;assert torch.cuda.device_count()==1;"
                    "a=torch.ones((256,256),device='cuda',dtype=torch.bfloat16);"
                    "v=float((a@a)[0,0]);assert v==256.0;"
                    "print(torch.cuda.get_device_name(0),v)"
                ),
            ],
            env=gpu_env,
        ).strip()
        smoke_results.append({"gpu": gpu, "result": smoke})

    input_result = validate_parent_inputs(ctx)
    repo_status = run_capture(["git", "status", "--short"])
    (ctx.record_dir / "versions.txt").write_text(versions)
    (ctx.record_dir / "preflight.log").write_text(
        json.dumps(
            {
                "time_utc": utc_now(),
                "commit": actual_commit,
                "repo_status": repo_status.splitlines(),
                "gpus": gpu_rows,
                "bf16_smoke": smoke_results,
                "disk_free_bytes": shutil.disk_usage(ctx.task_root).free,
                "input_validation": input_result,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (ctx.record_dir / "input_validation.log").write_text(json.dumps(input_result, indent=2) + "\n")
    mark_phase(
        ctx,
        "preflight",
        "PASS",
        commit=actual_commit,
        repo_status=repo_status,
        parent_inventory_sha256=input_result["inventory_sha256"],
        selected_gpus=gpu_rows,
    )
    print(f"RUN_ID={ctx.run_id}")


def validate_resolved_config(ctx: RunContext, config_path: Path, max_iter: int) -> list[str]:
    import yaml

    config = yaml.safe_load(config_path.read_text())
    checks = {
        "job.project": "cosmos3",
        "job.group": "sft",
        "job.name": ctx.job_name,
        "job.wandb_mode": "disabled",
        "trainer.seed": 42,
        "trainer.max_iter": max_iter,
        "trainer.grad_accum_iter": 2,
        "trainer.logging_iter": 1,
        "trainer.distributed_parallelism": "fsdp",
        "model.config.max_num_tokens_after_packing": ctx.args.packing_token_budget,
        "model.config.joint_attn_implementation": "two_way",
        "model.config.ema.enabled": True,
        "model.config.lora_enabled": False,
        "model.config.precision": "bfloat16",
        "model.config.parallelism.data_parallel_shard_degree": 4,
        "model.config.parallelism.data_parallel_replicate_degree": 2,
        "model.config.parallelism.context_parallel_shard_degree": 2,
        "model.config.parallelism.cfg_parallel_shard_degree": 1,
        "model.config.parallelism.fsdp_master_dtype": "float32",
        "model.config.attention_io_layout": "sequence_sharded",
        "model.config.compile.enabled": True,
        "model.config.compile.compile_dynamic": True,
        "model.config.compile.compiled_region": "all",
        "model.config.compile.max_autotune_pointwise": True,
        "model.config.compile.coordinate_descent_tuning": True,
        "model.config.compile.use_cuda_graphs": False,
        "model.config.activation_checkpointing.mode": "selective",
        "model.config.activation_checkpointing.save_ops_regex": ["fmha", "mm", "bmm", "addmm"],
        "model.config.activation_checkpointing.preserve_rng_state": True,
        "model.config.activation_checkpointing.determinism_check": "default",
        "optimizer.fused": True,
        "optimizer.keys_to_select": [],
        "optimizer.lr": 0.0001,
        "trainer.callbacks.compile_tokenizer.enabled": True,
        "trainer.callbacks.compile_tokenizer.compile_after_iterations": 3,
        "trainer.callbacks.compile_tokenizer.warmup_resolutions": [256],
        "trainer.callbacks.sampled_media.enabled": True,
        "trainer.callbacks.sampled_media.flush_every_n_batches": 10,
        "trainer.callbacks.sampled_media.record_caption": False,
        "dataloader_train.max_sequence_length": ctx.args.packing_token_budget,
        "dataloader_train.max_samples_per_batch": None,
        "dataloader_train.dataloader.num_workers": 4,
        "dataloader_train.dataloader.persistent_workers": True,
        "dataloader_train.dataloader.pin_memory": True,
        "dataloader_train.dataloader.prefetch_factor": 4,
        "checkpoint.dcp_async_mode_enabled": True,
        "checkpoint.dcp_load_dedup": True,
        "checkpoint.save_iter": 2,
        "checkpoint.keys_to_skip_loading": ["net_ema."],
        "checkpoint.strict_resume": True,
        "checkpoint.load_path": str(ctx.inputs.base_dcp),
        "model.config.tokenizer.vae_path": str(ctx.inputs.wan_vae),
        "model.config.vlm_config.tokenizer.repository": None,
        "model.config.vlm_config.tokenizer.revision": None,
        "model.config.vlm_config.tokenizer.subdir": None,
        "model.config.vlm_config.tokenizer.tokenizer_type": str(ctx.inputs.edge_hf),
    }
    passed = []
    for dotted, expected in checks.items():
        assert_equal(nested_get(config, dotted), expected, dotted)
        passed.append(dotted)
    jsonl_paths = nested_get(config, "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths")
    assert_equal(jsonl_paths, [str(ctx.inputs.dataset / "train/video_dataset_file.jsonl")], "dataset JSONL path")
    unresolved = unresolved_strings(config)
    if unresolved:
        raise GateError(f"unresolved config interpolations: {unresolved[:10]}")
    return passed


def train_command(ctx: RunContext, max_iter: int, dryrun: bool = False) -> list[str]:
    executable = ctx.python if dryrun else ctx.torchrun
    prefix = [str(executable)]
    if not dryrun:
        prefix.extend(["--standalone", "--nproc-per-node=8"])
    prefix.extend(
        [
            "-m",
            "cosmos_framework.scripts.train" if dryrun else "tools.cosmos3_train_rank_entry",
            "--sft-toml=examples/toml/sft_config/vision_sft_edge.toml",
        ]
    )
    if dryrun:
        prefix.append("--dryrun")
    return [*prefix, "--", *training_overrides(ctx), f"trainer.max_iter={max_iter}"]


def phase_dryrun(ctx: RunContext) -> None:
    require_phases(ctx, "preflight")
    if phase_completed(ctx, "dryrun"):
        print("dryrun already PASS")
        return
    dryrun_root = ctx.task_root / "outputs/dryrun" / ctx.run_id
    config_path = dryrun_root / "cosmos3/sft" / ctx.job_name / "config.yaml"
    recovered = False
    if dryrun_root.exists():
        # The framework command may have completed and produced an immutable
        # config before a runner-side validator/state-write bug interrupted the
        # phase.  Revalidate those exact artifacts without overwriting them.
        if not config_path.is_file() or not (ctx.record_dir / "dryrun.log").is_file():
            raise GateError(f"create-only dryrun root already exists but is incomplete: {dryrun_root}")
        recovered = True
    else:
        dryrun_root.mkdir(parents=True, exist_ok=False)
        command = train_command(ctx, 5, dryrun=True)
        returncode = run_logged(ctx, command, ctx.record_dir / "dryrun.log", ctx.environment(dryrun_root))
        if returncode != 0:
            mark_phase(ctx, "dryrun", "FAIL", returncode=returncode)
            raise GateError(f"dryrun failed with exit code {returncode}")
    if not config_path.is_file():
        raise GateError(f"dryrun did not create resolved config: {config_path}")
    checks = validate_resolved_config(ctx, config_path, 5)
    validation = {
        "config": str(config_path),
        "sha256": sha256_file(config_path),
        "checks": checks,
        "recovered_existing_artifacts": recovered,
    }
    validation_path = ctx.record_dir / "dryrun_validation.log"
    if validation_path.exists():
        existing_validation = json.loads(validation_path.read_text())
        if existing_validation.get("status") != "PASS":
            raise GateError(f"existing dryrun validation is not PASS: {validation_path}")
        for key in ("config", "sha256", "checks"):
            assert_equal(existing_validation.get(key), validation[key], f"existing dryrun validation {key}")
    else:
        validation_path.write_text(json.dumps({"status": "PASS", **validation}, indent=2) + "\n")
    mark_phase(ctx, "dryrun", "PASS", **validation)


def validate_checkpoint(run_dir: Path, iteration: int) -> dict[str, Any]:
    checkpoint_name = f"iter_{iteration:09d}"
    pointer = run_dir / "checkpoints/latest_checkpoint.txt"
    if not pointer.is_file():
        raise GateError(f"missing checkpoint pointer: {pointer}")
    raw_pointer = pointer.read_text().strip()
    if not re.fullmatch(r"iter_[0-9]{9}", raw_pointer):
        raise GateError(f"unsafe checkpoint pointer: {raw_pointer!r}")
    assert_equal(raw_pointer, checkpoint_name, "latest checkpoint pointer")
    checkpoint = run_dir / "checkpoints" / raw_pointer
    components = {}
    for component in ("model", "optim", "scheduler", "trainer"):
        directory = checkpoint / component
        metadata = directory / ".metadata"
        shards = list(directory.glob("*.distcp"))
        if (
            not metadata.is_file()
            or metadata.stat().st_size == 0
            or not shards
            or any(path.stat().st_size == 0 for path in shards)
        ):
            raise GateError(f"incomplete DCP component {component}: {directory}")
        components[component] = {"metadata_bytes": metadata.stat().st_size, "shards": len(shards)}
    dataloader_files = list((checkpoint / "dataloader").glob("rank_*.pkl"))
    components["dataloader"] = {
        "status": "present" if dataloader_files else "unsupported_by_vision_sft_edge_recipe",
        "rank_files": len(dataloader_files),
    }
    return {"checkpoint": str(checkpoint), "components": components}


def parse_losses(log_text: str) -> dict[int, float]:
    losses = {}
    for match in re.finditer(r"Iteration\s+(\d+):[^\n]*?Loss:\s*([^\s|]+)", log_text):
        try:
            losses[int(match.group(1))] = float(match.group(2))
        except ValueError as error:
            raise GateError(f"unparseable loss value: {match.group(2)!r}") from error
    if any(not math.isfinite(value) for value in losses.values()):
        raise GateError(f"non-finite losses found: {losses}")
    return losses


def parse_aot_results(log_text: str) -> dict[str, Any]:
    matches = re.findall(
        r"Rank (\d+): AOT compiled (\d+), loaded (\d+)/(\d+) chunk variants, time: ([0-9.]+)s",
        log_text,
    )
    by_rank = {
        int(rank): (int(compiled), int(loaded), int(total), float(seconds))
        for rank, compiled, loaded, total, seconds in matches
    }
    assert_equal(set(by_rank), set(range(8)), "tokenizer AOT completion ranks")
    for rank, (_compiled, loaded, total, _seconds) in by_rank.items():
        if total <= 0 or loaded != total:
            raise GateError(f"rank {rank} tokenizer AOT did not load every variant: loaded={loaded}, total={total}")
    return {
        "ranks": len(by_rank),
        "variants": next(iter(by_rank.values()))[2],
        "max_seconds": max(values[3] for values in by_rank.values()),
        "compiled_per_rank": {rank: values[0] for rank, values in sorted(by_rank.items())},
    }


def validate_sampled_media(run_dir: Path, expected_microbatches_per_rank: int) -> dict[str, Any]:
    lance = importlib.import_module("lance")
    table_path = run_dir / "sampled_media.lance"
    if not table_path.exists():
        raise GateError(f"missing sampled-media evidence: {table_path}")
    table = lance.dataset(str(table_path)).to_table(columns=["rank", "batch_index", "sample_id", "iteration"])
    records = table.to_pylist()
    if not records:
        raise GateError("sampled-media evidence is empty")
    batch_sizes = Counter((int(row["rank"]), int(row["batch_index"])) for row in records)
    ranks = {rank for rank, _ in batch_sizes}
    assert_equal(ranks, set(range(8)), "sampled-media ranks")
    for rank in ranks:
        indexes = {batch for candidate_rank, batch in batch_sizes if candidate_rank == rank}
        if not set(range(expected_microbatches_per_rank)).issubset(indexes):
            raise GateError(f"rank {rank} sampled-media misses microbatches: found {sorted(indexes)}")
    max_samples = max(batch_sizes.values())
    if max_samples <= 1:
        raise GateError("packing gate failed: no recorded microbatch contains more than one sample")
    return {"records": len(records), "ranks": len(ranks), "max_samples_in_rank_microbatch": max_samples}


def validate_train_log(ctx: RunContext, log_path: Path, expected_iteration: int, resume: bool) -> dict[str, Any]:
    text = log_path.read_text(errors="replace")
    if "Done with training." not in text:
        raise GateError("training log does not contain completion marker")
    forbidden = (
        "CUDA out of memory",
        "Traceback (most recent call last)",
        "NCCL watchdog",
        "Watchdog caught collective operation timeout",
        "DistBackendError",
        "CompilerError",
        "InductorError",
    )
    present = [pattern for pattern in forbidden if pattern in text]
    if present:
        raise GateError(f"training log contains fatal patterns: {present}")
    required_topology = (
        "Building 3-D device mesh with ['rest', 'cfgp', 'cp'], [4, 1, 2]",
        "Building 2-D device mesh with ['dp_replicate', 'dp_shard'], [2, 4]",
    )
    for pattern in required_topology:
        if pattern not in text:
            raise GateError(f"missing runtime topology evidence: {pattern}")
    if "clamping dp_shard" in text:
        raise GateError("runtime clamped dp_shard")
    tensor_matches = re.findall(r"Total tensors: (\d+), trainable tensors: (\d+), selected tensors: (\d+)", text)
    if not tensor_matches:
        raise GateError("missing optimizer tensor-selection evidence")
    totals = tuple(map(int, tensor_matches[-1]))
    if not (totals[0] == totals[1] == totals[2] > 0):
        raise GateError(f"full-parameter optimizer gate failed: total/trainable/selected={totals}")
    losses = parse_losses(text)
    if expected_iteration not in losses:
        raise GateError(f"missing finite loss for iteration {expected_iteration}: {losses}")
    aot = parse_aot_results(text)
    if "Removed directory:" not in text or "/aot_tokenizer" not in text:
        raise GateError("training log lacks tokenizer AOT cleanup evidence")
    if resume:
        for component in ("model", "optimizer", "scheduler", "trainer"):
            if f"Loading the {component}" not in text:
                raise GateError(f"resume did not load {component}")
        if f"in iteration {expected_iteration - 1}" not in text:
            raise GateError("resume log does not show iteration continuity")
        if "[DCP-LOAD-DEDUP]" not in text or "[DCP-LOAD-TIMING] broadcast" not in text:
            raise GateError("resume log lacks DCP load-dedup runtime evidence")
    else:
        if "(warm-start, local)" not in text:
            raise GateError("stage A did not identify the base DCP as a local warm start")
        if (
            "Staging the state_dict in CPU memory" not in text
            or "Submitted checkpoint to background process" not in text
        ):
            raise GateError("stage A log lacks async pinned-memory checkpoint evidence")
    checkpoint = validate_checkpoint(ctx.run_dir, expected_iteration)
    return {"losses": losses, "optimizer_tensors": totals, "tokenizer_aot": aot, **checkpoint}


def _finalize_train_phase(ctx: RunContext, *, resume: bool, log_path: Path, max_iter: int) -> dict[str, Any]:
    result = validate_train_log(ctx, log_path, max_iter, resume)
    if (ctx.run_dir / "aot_tokenizer").exists():
        raise GateError("tokenizer AOT temporary package directory was not cleaned after load")
    if not resume:
        result["sampled_media"] = validate_sampled_media(ctx.run_dir, 10)
        compiler_artifacts = [
            path for path in ctx.compile_cache_root.rglob("*") if path.is_file() and path.stat().st_size > 0
        ]
        if not compiler_artifacts:
            raise GateError("TorchInductor cache contains no compiled artifacts")
        result.update(compiler_artifacts=len(compiler_artifacts))
        compiler_hash_file = ctx.record_dir / "compile_cache_artifacts.sha256"
        if not compiler_hash_file.exists():
            compiler_files, compiler_bytes = hash_tree(ctx.compile_cache_root, compiler_hash_file)
            result.update(compiler_files=compiler_files, compiler_bytes=compiler_bytes)
    checkpoint_hash_file = (
        ctx.record_dir / "checkpoint_artifacts.sha256"
        if resume
        else ctx.record_dir / "checkpoint_iter_000000005.sha256"
    )
    checkpoint_dir = Path(result["checkpoint"])
    if not checkpoint_hash_file.exists():
        checkpoint_files, checkpoint_bytes = hash_tree(checkpoint_dir, checkpoint_hash_file)
        result.update(checkpoint_files=checkpoint_files, checkpoint_bytes=checkpoint_bytes)
    validation_file = ctx.record_dir / ("resume_validation.log" if resume else "train_stage_a_validation.log")
    if validation_file.exists():
        previous = json.loads(validation_file.read_text())
        assert_equal(
            previous.get("checkpoint"), result.get("checkpoint"), f"existing {validation_file.name} checkpoint"
        )
        result = previous
    else:
        validation_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _phase_train_common(ctx: RunContext, *, resume: bool) -> None:
    phase = "resume" if resume else "train"
    prerequisites = ("preflight", "dryrun", "train") if resume else ("preflight", "dryrun")
    require_phases(ctx, *prerequisites)
    if phase_completed(ctx, phase):
        print(f"{phase} already PASS")
        return
    selected = parse_selected_gpus(ctx.args.gpus)
    assert_gpus_idle(selected, ctx.args.max_idle_memory_mib)
    max_iter = 6 if resume else 5
    log_name = "train_stage_b_resume.log" if resume else "train_stage_a.log"
    log_path = ctx.record_dir / log_name
    checkpoint_path = ctx.run_dir / "checkpoints" / f"iter_{max_iter:09d}"
    recovering = log_path.is_file() and checkpoint_path.is_dir()
    if recovering:
        result = _finalize_train_phase(ctx, resume=resume, log_path=log_path, max_iter=max_iter)
        result["recovered_existing_artifacts"] = True
        mark_phase(ctx, phase, "PASS", **result)
        return
    if not resume and ctx.run_dir.exists():
        raise GateError(f"create-only formal run already exists and is not recoverable: {ctx.run_dir}")
    if not resume and ctx.compile_cache_root.exists():
        raise GateError(
            f"create-only local compile cache already exists and is not recoverable: {ctx.compile_cache_root}"
        )
    if resume:
        validate_checkpoint(ctx.run_dir, 5)
        if not ctx.compile_cache_root.is_dir():
            raise GateError(f"resume compile cache is missing: {ctx.compile_cache_root}")
    monitor_name = "stage_b" if resume else "stage_a"
    command = train_command(ctx, max_iter)
    returncode = run_logged(
        ctx,
        command,
        log_path,
        ctx.environment(ctx.task_root / "outputs/train"),
        monitor_prefix=monitor_name,
    )
    if returncode != 0:
        text = log_path.read_text(errors="replace")
        oom = "CUDA out of memory" in text or "OutOfMemoryError" in text
        mark_phase(ctx, phase, "FAIL", returncode=returncode, cuda_oom=oom)
        if oom and not resume:
            lower = {45056: 32768, 32768: 24576}.get(ctx.args.packing_token_budget)
            suffix = (
                f"; start a new run-id with --packing-token-budget {lower}"
                if lower
                else "; minimum legal budget also OOMed"
            )
            raise GateError(f"stage A CUDA OOM at budget {ctx.args.packing_token_budget}{suffix}")
        raise GateError(f"{phase} failed with exit code {returncode}")
    result = _finalize_train_phase(ctx, resume=resume, log_path=log_path, max_iter=max_iter)
    mark_phase(ctx, phase, "PASS", **result)


def phase_train(ctx: RunContext) -> None:
    _phase_train_common(ctx, resume=False)


def phase_resume(ctx: RunContext) -> None:
    _phase_train_common(ctx, resume=True)


def hash_tree(root: Path, output: Path, *, exclude: Iterable[Path] = ()) -> tuple[int, int]:
    excluded = {path.resolve(strict=False) for path in exclude}
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.resolve(strict=False) not in excluded),
        key=lambda path: path.as_posix(),
    )
    with output.open("x", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path}\n")
    return len(files), sum(path.stat().st_size for path in files)


def phase_export(ctx: RunContext) -> None:
    require_phases(ctx, "preflight", "dryrun", "train", "resume")
    if phase_completed(ctx, "export"):
        print("export already PASS")
        return
    checkpoint_result = validate_checkpoint(ctx.run_dir, 6)
    model_dir = ctx.run_dir / "model"
    if model_dir.exists():
        raise GateError(f"create-only export target already exists: {model_dir}")
    export_gpu = str(ctx.args.export_gpu)
    assert_gpus_idle([export_gpu], ctx.args.max_idle_memory_mib)
    config_file = ctx.run_dir / "config.yaml"
    validate_resolved_config(ctx, config_file, 6)
    command = [
        str(ctx.python),
        "-m",
        "cosmos_framework.scripts.export_model",
        "--checkpoint-path",
        checkpoint_result["checkpoint"],
        "--config-file",
        str(config_file),
        "--vit-checkpoint-path",
        str(ctx.inputs.edge_hf),
        "--dp-shard-size=1",
        "--dp-replicate-size=1",
        "--tp-size=1",
        "--cp-size=1",
        "--cfgp-size=1",
        "--no-use-torch-compile",
        "--no-use-cuda-graphs",
        "-o",
        str(model_dir),
    ]
    env = ctx.environment(ctx.task_root / "outputs/train")
    env["CUDA_VISIBLE_DEVICES"] = export_gpu
    returncode = run_logged(ctx, command, ctx.record_dir / "export.log", env)
    if returncode != 0:
        mark_phase(ctx, "export", "FAIL", returncode=returncode)
        raise GateError(f"export failed with exit code {returncode}")
    required = (
        "model.safetensors.index.json",
        "config.json",
        "export_manifest.json",
        "processor_config.json",
        "tokenizer.json",
        "vision_encoder/model.safetensors",
    )
    missing = [relative for relative in required if not (model_dir / relative).is_file()]
    if missing:
        raise GateError(f"export is missing required artifacts: {missing}")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    keys = set(index["weight_map"])
    lora = sorted(key for key in keys if re.search(r"lora_[AB](?:\.|$)", key, re.IGNORECASE))
    if lora:
        raise GateError(f"full-parameter export contains LoRA tensors: {lora[:10]}")
    categories = {
        "understanding": any("language_model.model.layers" in key and "moe_gen" not in key for key in keys),
        "generation": any("moe_gen" in key for key in keys),
        "embedding": any("embed_tokens" in key for key in keys),
        "head": any("lm_head" in key for key in keys),
    }
    if not all(categories.values()):
        raise GateError(f"export key coverage failed: {categories}")
    checksum = ctx.record_dir / "export_artifacts.sha256"
    count, total = hash_tree(model_dir, checksum)
    result = {
        "model_dir": str(model_dir),
        "files": count,
        "bytes": total,
        "key_count": len(keys),
        "categories": categories,
    }
    (ctx.record_dir / "export_validation.log").write_text(json.dumps(result, indent=2) + "\n")
    mark_phase(ctx, "export", "PASS", **result)


def validate_media_output(output_dir: Path) -> dict[str, Any]:
    from PIL import Image

    images = sorted([*output_dir.rglob("*.jpg"), *output_dir.rglob("*.jpeg"), *output_dir.rglob("*.png")])
    videos = sorted([*output_dir.rglob("*.mp4"), *output_dir.rglob("*.webm")])
    validated = []
    for path in images:
        with Image.open(path) as image:
            image.load()
            extrema = image.convert("RGB").getextrema()
            if image.width <= 0 or image.height <= 0 or not any(high > low for low, high in extrema):
                raise GateError(f"constant or invalid image output: {path}")
            validated.append({"path": str(path), "kind": "image", "size": [image.width, image.height]})
    for path in videos:
        probe = json.loads(
            run_capture(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames:format=duration",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )
        stream = (probe.get("streams") or [{}])[0]
        if (
            int(stream.get("width", 0)) <= 0
            or int(stream.get("height", 0)) <= 0
            or float(probe.get("format", {}).get("duration", 0)) <= 0
        ):
            raise GateError(f"invalid video output: {path}")
        validated.append({"path": str(path), "kind": "video"})
    if not validated:
        raise GateError(f"inference produced no readable image/video under {output_dir}")
    return {"outputs": validated}


def phase_inference(ctx: RunContext) -> None:
    require_phases(ctx, "preflight", "dryrun", "train", "resume", "export")
    if phase_completed(ctx, "inference"):
        print("inference already PASS")
        return
    output_dir = ctx.task_root / "outputs/inference" / f"{ctx.job_name}_export_iter_000000006"
    if output_dir.exists():
        raise GateError(f"create-only inference target already exists: {output_dir}")
    export_gpu = str(ctx.args.export_gpu)
    assert_gpus_idle([export_gpu], ctx.args.max_idle_memory_mib)
    command = [
        str(ctx.python),
        "-m",
        "cosmos_framework.scripts.inference",
        "-i",
        "inputs/omni/t2i.json",
        "-o",
        str(output_dir),
        "--checkpoint-path",
        str(ctx.run_dir / "model"),
        "--dp-shard-size=1",
        "--dp-replicate-size=1",
        "--cp-size=1",
        "--cfgp-size=1",
        "--no-use-torch-compile",
        "--no-use-cuda-graphs",
        "--seed=0",
        "--no-guardrails",
        "--experiment-overrides",
        "model.config.vlm_config.tokenizer.repository=nvidia/Cosmos3-Edge",
        f"model.config.vlm_config.tokenizer.revision={EDGE_REVISION}",
        "model.config.vlm_config.tokenizer.subdir=",
        f"model.config.tokenizer.vae_path={ctx.inputs.wan_vae}",
        "model.config.tokenizer.bucket_name=",
        "model.config.tokenizer.object_store_credential_path_pretrained=",
    ]
    env = ctx.environment(ctx.task_root / "outputs/inference")
    env["CUDA_VISIBLE_DEVICES"] = export_gpu
    returncode = run_logged(ctx, command, ctx.record_dir / "inference.log", env)
    if returncode != 0:
        mark_phase(ctx, "inference", "FAIL", returncode=returncode)
        raise GateError(f"inference failed with exit code {returncode}")
    text = (ctx.record_dir / "inference.log").read_text(errors="replace")
    if str(ctx.run_dir / "model") not in text:
        raise GateError("inference log does not identify the exported checkpoint")
    if "Traceback (most recent call last)" in text:
        raise GateError("inference log contains a traceback")
    result = {"output_dir": str(output_dir), **validate_media_output(output_dir)}
    (ctx.record_dir / "inference_validation.log").write_text(json.dumps(result, indent=2) + "\n")
    mark_phase(ctx, "inference", "PASS", **result)


def phase_finalize(ctx: RunContext) -> None:
    require_phases(ctx, "preflight", "dryrun", "train", "resume", "export", "inference")
    if phase_completed(ctx, "finalize"):
        print("finalize already PASS")
        return
    selected = parse_selected_gpus(ctx.args.gpus)
    final_gpus = assert_gpus_idle(selected, ctx.args.max_idle_memory_mib)
    validate_checkpoint(ctx.run_dir, 6)
    state = load_state(ctx)
    final_input_result = validate_parent_inputs(ctx)
    assert_equal(
        final_input_result["inventory_sha256"],
        state["phases"]["preflight"]["parent_inventory_sha256"],
        "parent inventory at final acceptance",
    )
    repo_status = run_capture(["git", "status", "--short"])
    final_commit = run_capture(["git", "rev-parse", "HEAD"]).strip()
    repository_provenance = validate_repository_transition(
        state["phases"]["preflight"]["repo_status"],
        repo_status,
        allow_committed_transition=ctx.args.allow_committed_repository_transition,
        preflight_commit=state["phases"]["preflight"]["commit"],
        final_commit=final_commit,
    )
    state.setdefault("phases", {})["finalize"] = {
        "status": "PASS",
        "time_utc": utc_now(),
        "repository_provenance": repository_provenance,
    }
    save_state(ctx, state)
    manifest = {
        "title": "Cosmos3 v0.0.1b full-opt demo run manifest",
        "status": "PASS_WITH_DATALOADER_CURSOR_LIMITATION",
        "run_id": ctx.run_id,
        "time_utc": utc_now(),
        "repository_provenance": repository_provenance,
        "packing_token_budget": ctx.args.packing_token_budget,
        "job_name": ctx.job_name,
        "run_dir": str(ctx.run_dir),
        "local_compile_cache": str(ctx.compile_cache_root),
        "selected_gpus": final_gpus,
        "immutable_input_revisions": {
            "dataset": DATASET_REVISION,
            "wan": WAN_REVISION,
            "edge": EDGE_REVISION,
        },
        "phases": state["phases"],
        "limitation": (
            "vision_sft_edge uses RankPartitionedDataLoader without a dataloader-state callback; "
            "current DCP therefore resumes model/optimizer/scheduler/trainer but not prefetched loader position"
        ),
    }
    (ctx.record_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifacts = ctx.record_dir / "artifacts.sha256"
    count, total = hash_tree(ctx.task_root, artifacts, exclude=[artifacts])
    for line in artifacts.read_text().splitlines():
        expected, raw_path = line.split(maxsplit=1)
        assert_equal(sha256_file(Path(raw_path)), expected, f"final artifact {raw_path}")
    print(f"FULL_OPT_DEMO=PASS_WITH_DATALOADER_CURSOR_LIMITATION files={count} bytes={total} run_id={ctx.run_id}")


PHASE_FUNCTIONS = {
    "preflight": phase_preflight,
    "dryrun": phase_dryrun,
    "train": phase_train,
    "resume": phase_resume,
    "export": phase_export,
    "inference": phase_inference,
    "finalize": phase_finalize,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*PHASES, "all"))
    parser.add_argument("--run-id", help="Reuse this ID for separately invoked phases; generated by preflight/all")
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--parent-record", type=Path, default=DEFAULT_PARENT_RECORD)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--export-gpu", type=int, default=0)
    parser.add_argument("--packing-token-budget", type=int, choices=ALLOWED_BUDGETS, default=45056)
    parser.add_argument("--attempt-tag", help="Create a distinct job name for a preserved retry/diagnostic attempt")
    parser.add_argument(
        "--allow-committed-repository-transition",
        action="store_true",
        help="At finalize only, accept a clean committed worktree transition and record both commit IDs",
    )
    parser.add_argument("--local-compile-cache-root", type=Path, default=DEFAULT_LOCAL_COMPILE_CACHE_ROOT)
    parser.add_argument("--expected-commit", default=AUDITED_COMMIT)
    parser.add_argument("--min-free-gib", type=int, default=100)
    parser.add_argument("--max-idle-memory-mib", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.phase != "preflight" and args.phase != "all" and not args.run_id:
        parser.error("--run-id is required when continuing after preflight")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ctx = build_context(args)
        phases = PHASES if args.phase == "all" else (args.phase,)
        for phase in phases:
            print(f"[{utc_now()}] phase={phase} run_id={ctx.run_id}")
            PHASE_FUNCTIONS[phase](ctx)
        return 0
    except GateError as error:
        print(f"GATE_FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
