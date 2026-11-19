"""Compare uninterrupted execution with save/exit and at least two resumed updates.

Starts both branches from a canonical bundle or a read-only completed checkpoint.
All child ranks exit between launches. No checkpoint is overwritten.
Run when four GPUs are available; checkpoint I/O is substantial.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import tomllib

from tools.compare_tp_pp_performance import inputs, updates
from tools.compare_tp_pp_runtime_checkpoints import compare


def source_hashes(toml):
    files = list(Path("cosmos_framework").rglob("*.py")) + [
        toml,
        Path(__file__),
        Path("tools/compare_tp_pp_runtime_checkpoints.py"),
        Path("tools/compare_tp_pp_performance.py"),
        Path("tools/run_tp_pp_with_activation_offload.py"),
        Path("tools/run_tp_pp_with_selective_offload.py"),
        Path("tools/tp_pp_selective_activation_offload.py"),
        Path("tools/tp_pp_activation_offload.py"),
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def start_topology(bundle, checkpoint, logical_dp, global_slots, world_size=4):
    """Validate the actual start topology before creating output or launching ranks."""
    tp, pp = bundle["tp"], bundle["pp"]
    if tp < 1 or pp < 1 or world_size % (tp * pp):
        raise ValueError("Canonical TP/PP must divide the physical world size")
    if checkpoint is not None:
        if checkpoint["format"] != "cosmos_megatron_tp_pp" or checkpoint["world_size"] != world_size:
            raise ValueError("Runtime checkpoint format or world size differs from this run")
        topology = checkpoint["topology"]
        if (topology["tensor_model_parallel_size"], topology["pipeline_model_parallel_size"]) != (tp, pp):
            raise ValueError("Runtime checkpoint topology differs from the canonical bundle")
        for supplied, key in (
            (logical_dp, "logical_data_parallel_size"),
            (global_slots, "global_packed_microbatches_per_update"),
        ):
            if supplied is not None and supplied != topology[key]:
                raise ValueError(f"Explicit {key} differs from the runtime checkpoint")
        logical_dp = topology["logical_data_parallel_size"]
        global_slots = topology["global_packed_microbatches_per_update"]
    elif logical_dp is None or global_slots is None:
        raise ValueError("Canonical starts require explicit logical DP and global packed microbatches")
    dp = world_size // (tp * pp)
    if logical_dp < 1 or global_slots < 1 or logical_dp % dp or global_slots % logical_dp:
        raise ValueError("Logical streams and global slots must be divisible by DP and logical DP respectively")
    return {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "logical_data_parallel_size": logical_dp,
        "global_packed_microbatches_per_update": global_slots,
    }


def training_entry(*, activation_cpu_offload, selective_blocks):
    if selective_blocks < 0 or activation_cpu_offload and selective_blocks:
        raise ValueError("Choose either whole-update or positive selective block offload")
    if selective_blocks:
        return "tools.run_tp_pp_with_selective_offload"
    if activation_cpu_offload:
        return "tools.run_tp_pp_with_activation_offload"
    return "cosmos_framework.scripts.train"


def validate_microbatch_gc(root, iterations, *, interval, microbatches):
    """Require actual schedule records; an environment request is not execution."""
    expected = set(iterations)
    ticks = list(range(interval, microbatches + 1, interval))
    for rank in range(4):
        path = root / f"microbatch_gc.rank{rank}.jsonl"
        if not path.is_file():
            raise ValueError(f"GC policy execution record is missing: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        indexed = {row["iteration"]: row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != expected:
            raise ValueError("GC policy records do not cover exactly the completed update window")
        for row in rows:
            if (
                row["interval"] != interval
                or not row["schedule_completed"]
                or row["completed_backwards"] != microbatches
                or [event["completed_backwards"] for event in row["collections"]] != ticks
            ):
                raise ValueError("GC policy did not execute at the requested microbatch boundaries")
    return True


def run(args):
    import torch

    # This coordinator only trains in child processes. Large CPU comparisons
    # otherwise inherit the host's 159-thread default for every 1M window.
    torch.set_num_threads(2)
    entry = training_entry(
        activation_cpu_offload=args.activation_cpu_offload, selective_blocks=args.selective_offload_blocks
    )
    if args.updates_after_restart < 2:
        raise ValueError("Strict restart acceptance requires at least two updates after restart")
    if args.microbatch_gc_interval < 0:
        raise ValueError("Microbatch GC interval must be nonnegative")
    initial = args.initial_checkpoint.resolve() if args.initial_checkpoint else None
    manifest = None
    if initial is not None:
        if not (initial / ".save_complete").is_file():
            raise ValueError("Restart acceptance requires a complete initial checkpoint")
        manifest = json.loads((initial / "manifest.json").read_text())
    bundle_manifest = json.loads((args.bundle / "manifest.json").read_text())
    if not (args.bundle / ".save_complete").is_file():
        raise ValueError("A completed canonical source bundle is required")
    topology = start_topology(
        bundle_manifest, manifest, args.logical_data_parallel_size, args.global_packed_microbatches_per_update
    )
    if manifest is None:
        from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader

        iteration = int(CanonicalDCPReader(args.bundle / "logical_state" / "trainer").read("iteration"))
    else:
        iteration = manifest["iteration"]
        if initial.name != f"iter_{iteration:09d}":
            raise ValueError("Initial checkpoint directory must retain its iteration name")
    root = args.output_root.resolve()
    if (initial is not None and root.is_relative_to(initial)) or root.is_relative_to(args.bundle.resolve()):
        raise ValueError("Acceptance output must be outside the read-only initial checkpoint and bundle")
    root.mkdir(parents=True, exist_ok=False)
    recipe = tomllib.loads(args.toml.read_text())
    paths = {}
    for branch in ("uninterrupted", "restarted"):
        path = root / "training" / recipe["job"]["project"] / recipe["job"]["group"] / branch
        checkpoint_dir = path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        if initial is not None:
            (checkpoint_dir / initial.name).symlink_to(initial, target_is_directory=True)
        paths[branch] = path

    environment = os.environ.copy()
    for key, value in {
        "HF_MODEL_PATH": "/share-mixed/project/eai_pwm/models/Cosmos3-Nano-VLM",
        "WAN_VAE_PATH": "/share-mixed/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth",
        "ACTION128_EPISODE_CACHE_DIR": "/share-mixed/project/eai_pwm/cache/action128_episode_metadata_v1",
        "NUMPY_MADVISE_HUGEPAGE": "0",
        "MIMALLOC_ALLOW_LARGE_OS_PAGES": "0",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "7200",
        "COSMOS_NCCL_SUBGROUP_TIMEOUT_SEC": "7200",
    }.items():
        environment.setdefault(key, value)
    environment["BASE_CHECKPOINT_PATH"] = bundle_manifest["source"]
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["LD_LIBRARY_PATH"] = environment.get("COSMOS_FFMPEG_LIBRARY_PATH") or (
        "/opt/cosmos-megatron-ffmpeg8" if Path("/opt/cosmos-megatron-ffmpeg8").is_dir() else ""
    )
    environment.update(
        OUTPUT_ROOT=str(root / "training"),
        IMAGINAIRE_OUTPUT_ROOT=str(root / "training"),
        COSMOS_MEGATRON_PYTHON=sys.executable,
        MEGATRON_TP_PP_IMPORT_PATH=str(args.bundle.resolve()),
        CUDA_VISIBLE_DEVICES=args.devices,
        NPROC_PER_NODE="4",
        COSMOS_TP_PP_TRACE_PREPARED="1",
        COSMOS_TP_PP_PARALLEL_PREPARATION="1" if args.parallel_preparation else "0",
        COSMOS_ATTENTION_DETERMINISTIC="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="2",
        PYTHONPATH=str(Path.cwd()) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        COSMOS_TP_PP_ACTIVATION_OFFLOAD_BLOCKS=str(args.selective_offload_blocks),
        COSMOS_TP_PP_ACTIVATION_OFFLOAD_POLICY=args.activation_offload_policy,
        COSMOS_TP_PP_MICROBATCH_GC_INTERVAL=str(args.microbatch_gc_interval),
    )
    for key in (
        "COSMOS_TP_PP_PROFILE_SLOTS",
        "COSMOS_TP_PP_PROFILE_ITERATION",
        "COSMOS_TP_PP_CAPTURE_PREPARED_DIR",
        "COSMOS_TP_PP_CAPTURE_ITERATION",
        "COSMOS_TP_PP_VERIFY_RAW_REPLAY",
    ):
        environment.pop(key, None)
    # Use the supplied TOML directly, retaining all recipe compute/optimizer flags.
    hashes = source_hashes(args.toml)
    (root / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    commands = []
    final_iteration = iteration + 1 + args.updates_after_restart
    for branch, stop, label in (
        ("uninterrupted", final_iteration, "uninterrupted"),
        ("restarted", iteration + 1, "before_exit"),
        ("restarted", final_iteration, "after_restart"),
    ):
        if source_hashes(args.toml) != hashes:
            raise ValueError("Training/comparison source changed between acceptance branches")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "-m",
            entry,
            "--sft-toml",
            str(args.toml),
            "--",
            "trainer.seed=42",
            f"job.name={branch}",
            f"trainer.max_iter={stop}",
            "checkpoint.save_iter=100000",
            "trainer.cudnn.deterministic=true",
            "trainer.cudnn.benchmark=false",
            f"trainer.callbacks.compile_tokenizer.enabled={'true' if args.compile_tokenizer else 'false'}",
            f"model.config.parallelism.megatron.tensor_model_parallel_size={topology['tensor_model_parallel_size']}",
            f"model.config.parallelism.megatron.pipeline_model_parallel_size={topology['pipeline_model_parallel_size']}",
            f"model.config.parallelism.megatron.logical_data_parallel_size={topology['logical_data_parallel_size']}",
            f"model.config.parallelism.megatron.global_packed_microbatches_per_update={topology['global_packed_microbatches_per_update']}",
        ]
        commands.append({"label": label, "command": command})
        (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(json.dumps({"launch": label, "stop_iteration": stop}), flush=True)
        with (root / f"{label}.log").open("x") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not (paths[branch] / "checkpoints" / f"iter_{stop:09d}" / ".save_complete").is_file():
            raise ValueError(f"Child exited without publishing the expected checkpoint: {label}")

    if source_hashes(args.toml) != hashes:
        raise ValueError("Training/comparison source changed during acceptance execution")
    gc_execution_verified = None
    if args.microbatch_gc_interval:
        dp = 4 // (topology["tensor_model_parallel_size"] * topology["pipeline_model_parallel_size"])
        gc_execution_verified = all(
            validate_microbatch_gc(
                path,
                range(iteration + 1, final_iteration + 1),
                interval=args.microbatch_gc_interval,
                microbatches=topology["global_packed_microbatches_per_update"] // dp,
            )
            for path in paths.values()
        )
    left, right = (updates(paths[key]) for key in ("uninterrupted", "restarted"))
    expected_updates = list(range(iteration + 1, final_iteration + 1))
    if sorted(next(iter(left.values()))) != expected_updates or sorted(next(iter(right.values()))) != expected_updates:
        raise ValueError("Both branches must complete the complete predefined update window")
    mismatches = []
    loss_errors = []
    for update in expected_updates:
        losses = [
            [rank[update]["loss"] for rank in run.values() if rank[update].get("loss") is not None]
            for run in (left, right)
        ]
        if not all(losses) or not all(math.isfinite(value) for values in losses for value in values):
            raise ValueError("Restart branches must report finite losses")
        a, b = (sum(values) / len(values) for values in losses)
        loss_errors.append(abs(a - b) / max(abs(a), 1e-20))
    fields = ("source", "samples", "image", "tokens", "sha256", "prepared_noise_sha256", "prepared_packed_sha256")
    for update in range(iteration, final_iteration):
        a, b = inputs(paths["uninterrupted"], update), inputs(paths["restarted"], update)
        if set(a) != set(b) or len(a) != topology["global_packed_microbatches_per_update"]:
            raise ValueError("Missing or unexpected logical restart slots")
        for slot in a:
            for field in fields:
                if a[slot].get(field) is None or a[slot].get(field) != b[slot].get(field):
                    mismatches.append({"slot": slot, "field": field})
    checkpoints = [path / "checkpoints" / f"iter_{final_iteration:09d}" for path in paths.values()]
    state = compare(*checkpoints, args.bundle)
    (root / "state_comparison.json").write_text(json.dumps(state, indent=2) + "\n")
    report = {
        "passed": not mismatches and state["passed"] and max(loss_errors) <= 0.01,
        "scope": "actual source updates, uninterrupted versus save/all-ranks-exit/new-process and at least two resumed updates",
        "initial_checkpoint": str(initial) if initial is not None else None,
        "initial_bundle": str(args.bundle.resolve()),
        "initial_iteration": iteration,
        "topology": topology,
        "completed_iterations": expected_updates,
        "updates_after_restart": args.updates_after_restart,
        "global_slots_per_update": topology["global_packed_microbatches_per_update"],
        "input_mismatches": mismatches,
        "maximum_bf16_loss_relative_difference": max(loss_errors),
        "complete_state_comparison_passed": state["passed"],
        "parallel_preparation": args.parallel_preparation,
        "tokenizer_compile": args.compile_tokenizer,
        "microbatch_gc_interval": args.microbatch_gc_interval,
        "microbatch_gc_execution_verified": gc_execution_verified,
        "diagnostic_activation_cpu_offload": args.activation_cpu_offload,
        "activation_cpu_offload_policy": args.activation_offload_policy if args.activation_cpu_offload else None,
        "diagnostic_selective_offload_blocks": args.selective_offload_blocks,
        "cpu_comparison_threads": torch.get_num_threads(),
        "physical_compiler_rng_excluded_by_state_comparer": True,
        "production_576_slot_acceptance": False,
        "192_gpu_performance_accepted": False,
    }
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("Strict restart acceptance failed; complete observations retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--logical-data-parallel-size", type=int)
    parser.add_argument("--global-packed-microbatches-per-update", type=int)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--toml",
        type=Path,
        default=Path("examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"),
    )
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--parallel-preparation", action="store_true")
    parser.add_argument("--compile-tokenizer", action="store_true")
    parser.add_argument(
        "--activation-cpu-offload",
        action="store_true",
        help="Diagnostic saved-tensor CPU offload; retains input capacity, changes memory/performance behavior",
    )
    parser.add_argument("--selective-offload-blocks", type=int, default=0)
    parser.add_argument(
        "--activation-offload-policy",
        choices=("preserve_layout", "torch_default"),
        default="preserve_layout",
        help="Whole-update CPU offload layout policy; torch_default reproduces the legacy diagnostic",
    )
    parser.add_argument("--microbatch-gc-interval", type=int, default=0)
    parser.add_argument("--updates-after-restart", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
