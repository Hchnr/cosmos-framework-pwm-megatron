# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run an isolated 8-GPU AC comparison through the production shell launcher."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac", choices=["full", "attention_only", "legacy_regex"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", type=Path, default=Path("/tmp/cosmos-ac-attn/runs"))
    parser.add_argument("--gen-tokens", type=int, default=98304)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--initial-step", type=int, default=25000)
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--deterministic-attention", action="store_true")
    parser.add_argument("--eager-audit", action="store_true")
    parser.add_argument("--filter-oversized", action="store_true")
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.steps < 1 or args.warmup < 0 or args.gen_tokens < 1:
        parser.error("Use a new directory basename and positive steps/nonnegative warmup")
    if args.audit and (args.nsys or args.steps > 2 or args.warmup):
        parser.error("Audit is diagnostic-only: warmup=0, steps<=2, no nsys")
    if args.deterministic_attention and not args.audit:
        parser.error("Deterministic diagnostics require --audit and cannot supply performance results")
    if args.eager_audit and not args.audit:
        parser.error("Eager diagnostics require --audit and cannot supply performance results")
    run = (args.root / args.name).resolve()
    run.mkdir(parents=True, exist_ok=False)
    # Derive before schema validation so loader budgets and fixed lengths agree.
    recipe = BASE.read_text()
    recipe = recipe.replace("101376", str(args.gen_tokens + 3072)).replace("98304", str(args.gen_tokens))
    (run / "recipe.toml").write_text(recipe)
    steps = 2 if args.nsys else args.steps
    opts = [
        f"job.name=ac-{args.name}",
        "job.wandb_mode=disabled",
        "upload_reproducible_setup=false",
        "model.config.joint_attn_backend=split_default",
        "model.config.parallelism.data_parallel_shard_degree=8",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        f"model.config.activation_checkpointing.mode={'full' if args.ac == 'full' else 'selective'}",
        "trainer.seed=42",
        "trainer.callbacks.grad_clip.error_if_nonfinite=true",
        f"trainer.max_iter={args.initial_step + args.warmup + steps}",
        "checkpoint.type._target_=tools.unified_attn.observe.ReadOnlyCheckpointer",
        "+trainer.callbacks.ac_observer._target_="
        + ("tools.ac_attn.profile_observe.ProfileObserver" if args.nsys else "tools.ac_attn.observe.Observer"),
        "+trainer.callbacks.nvtx._target_=cosmos_framework.utils.callback.NVTXCallback",
        "+trainer.callbacks.nvtx.synchronize=false",
        f"+trainer.callbacks.nvtx.granularity={'module' if args.nsys else 'coarse'}",
        "+trainer.callbacks.nvtx.detail_target_ranks=[0]",
    ]
    if args.ac == "attention_only":
        # Hydra quoted strings need an escaped backslash. Use [.] instead.
        opts += ['model.config.activation_checkpointing.save_ops_regex=["^_flash_attn_forward[.]default$"]']
    if args.eager_audit:
        opts += ["model.config.compile.enabled=false", "trainer.compile_cache.enabled=false"]
    if args.nsys:
        opts += [
            "trainer.profiling.enable_nsys=true",
            "trainer.profiling.target_ranks=[0]",
            "trainer.profiling.profile_warmup=0",
            "trainer.profiling.profile_active=2",
            f"trainer.profiling.profile_freq={args.initial_step + args.warmup + 2}",
        ]
    env = os.environ.copy()
    for key in (
        "RDZV_ENDPOINT",
        "RDZV_ID",
        "RDZV_BACKEND",
        "EXTRA_TAIL_OVERRIDES",
        "TRAIN_RESOLUTION",
        "MAX_ITER",
        "COSMOS_FA3_PROVIDER",
        "IMAGINAIRE_OUTPUT_ROOT",
        "WANDB_API_KEY",
    ):
        env.pop(key, None)
    env.update(
        VIRTUAL_ENV=str(Path(sys.executable).parent.parent),
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        LD_LIBRARY_PATH="",
        LD_LIBRARY_PATH_OVERRIDE="",
        PYTHONHASHSEED="42",
        OMP_NUM_THREADS="8",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        COSMOS_EXIT_WITHOUT_FINALIZE="1",
        NNODES="1",
        NPROC_PER_NODE="8",
        RANK="0",
        PYTHONPATH=str(ROOT),
        SFT_TOML_FILE=str(run / "recipe.toml"),
        SFT_TRAINING_MODULE="tools.ac_attn.rank",
        OUTPUT_ROOT=str(run / "output"),
        IMAGINAIRE_OUTPUT_ROOT=str(run / "output"),
        UNIFIED_ATTN_RUN=str(run),
        UNIFIED_ATTN_NSYS=str(int(args.nsys)),
        UNIFIED_ATTN_AUDIT=str(int(args.audit)),
        UNIFIED_ATTN_CAPTURE="0",
        AC_INITIAL_STEP=str(args.initial_step),
        AC_FILTER_OVERSIZED=str(int(args.filter_oversized)),
        AC_DETERMINISTIC_ATTENTION=str(int(args.deterministic_attention)),
        WANDB_MODE="disabled",
        WANDB_PERSIST_LOGIN="0",
        NCCL_DEBUG="WARN",
        USR_COMPILE_CACHE_ROOT=str(run / "compile_cache"),
    )
    if args.deterministic_attention:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    launcher = (
        "examples/launch_sft_vision_nano_usr_attn_ac.sh"
        if args.ac == "attention_only"
        else "examples/launch_sft_vision_nano_usr_128gpu.sh"
    )
    command = ["bash", launcher, *opts]
    if args.dryrun:
        command = [
            sys.executable,
            "-m",
            "cosmos_framework.scripts.train",
            f"--sft-toml={run / 'recipe.toml'}",
            "--dryrun",
            "--",
            *opts,
        ]
    manifest = {
        "args": {**vars(args), "root": str(args.root), "backend": "split_default"},
        "command": command,
        "initial_step": args.initial_step,
        "baseline": str(BASE),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "fa3_provider": "nv",
        "python": sys.executable,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "flash-attn-3-nv", "transformers", "triton", "lerobot")
        },
        "environment": {
            key: env.get(key)
            for key in (
                "VIRTUAL_ENV",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "TORCHINDUCTOR_COMPILE_THREADS",
                "USR_COMPILE_CACHE_ROOT",
                "USR_DATASET_ROOT",
                "HF_MODEL_PATH",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    (run / "source_hashes.json").write_text(
        json.dumps(
            {
                f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                for f in files
                if f.endswith(".py") and (ROOT / f).is_file()
            },
            indent=2,
            sort_keys=True,
        )
    )
    (run / "worktree.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
    print(run, flush=True)
    with (run / "train.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (run / "exit.json").write_text(json.dumps({"returncode": result.returncode}))
    print(run, "exit=", result.returncode, flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
