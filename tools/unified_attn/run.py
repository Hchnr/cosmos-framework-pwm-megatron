# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reproducible eight-GPU derivative of the online T2V training recipe."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend",
    required=True,
    choices=[
        "split_default",
        "split_fa3_fork",
        "unified_fa3_grouped",
        "split_fa4",
        "unified_fa4_mask",
        "split_flex_triton",
        "unified_flex_triton",
    ],
)
parser.add_argument("--name", required=True)
parser.add_argument("--gen-tokens", type=int, default=98304)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--nsys", action="store_true")
parser.add_argument("--dryrun", action="store_true")
parser.add_argument("--audit", action="store_true")
parser.add_argument("--capture-attention", action="store_true")
args = parser.parse_args()
if args.capture_attention and (args.nsys or args.steps > 2):
    parser.error("Attention capture is diagnostic-only; use at most two steps without nsys")
run = ROOT / "plans/v0.2.7_unified_attn/artifacts/runs" / args.name
run.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env.update(
    LD_LIBRARY_PATH="",
    PYTHONHASHSEED="42",
    OMP_NUM_THREADS="8",
    TORCHINDUCTOR_COMPILE_THREADS="4",
    COSMOS_EXIT_WITHOUT_FINALIZE="1",
    IMAGINAIRE_OUTPUT_ROOT=str(run / "output"),
    UNIFIED_ATTN_RUN=str(run),
    UNIFIED_ATTN_NSYS=str(int(args.nsys)),
    UNIFIED_ATTN_AUDIT=str(int(args.audit)),
    UNIFIED_ATTN_CAPTURE=str(int(args.capture_attention)),
    WANDB_MODE="disabled",
    NCCL_DEBUG="WARN",
    USR_COMPILE_CACHE_ROOT=str(Path("/tmp/cosmos-unified-attn") / args.name),
)
env.pop("WANDB_API_KEY", None)
env["PYTHONPATH"] = str(ROOT)
if args.backend in ("split_fa3_fork", "unified_fa3_grouped"):
    env["COSMOS_FA3_PROVIDER"] = "fork"
    env["PYTHONPATH"] += ":" + str(ROOT / "plans/v0.2.7_unified_attn/artifacts/deps/hopper")
else:
    env.pop("COSMOS_FA3_PROVIDER", None)
steps = 2 if args.nsys else args.steps
opts = [
    f"job.name=unified-{args.name}",
    "job.wandb_mode=disabled",
    f"model.config.joint_attn_backend={args.backend}",
    "model.config.parallelism.data_parallel_shard_degree=8",
    "model.config.parallelism.data_parallel_replicate_degree=1",
    f"model.config.fixed_sequence_length.gen_tokens={args.gen_tokens}",
    f"model.config.max_num_tokens_after_packing={args.gen_tokens + 3072}",
    f"dataloader_train.max_sequence_length={args.gen_tokens + 3072}",
    "trainer.seed=42",
    "trainer.callbacks.grad_clip.error_if_nonfinite=true",
    f"trainer.max_iter={25000 + args.warmup + steps}",
    "checkpoint.save_iter=1000000",
    "checkpoint.type._target_=tools.unified_attn.observe.ReadOnlyCheckpointer",
    "+trainer.callbacks.unified_observer._target_=tools.unified_attn.observe.Observer",
    "+trainer.callbacks.nvtx._target_=cosmos_framework.utils.callback.NVTXCallback",
    "+trainer.callbacks.nvtx.synchronize=false",
    f"+trainer.callbacks.nvtx.granularity={'module' if args.nsys else 'coarse'}",
    "+trainer.callbacks.nvtx.detail_target_ranks=[0]",
]
if args.nsys:
    opts += [
        "trainer.profiling.enable_nsys=true",
        "trainer.profiling.target_ranks=[0]",
        "trainer.profiling.profile_warmup=0",
        "trainer.profiling.profile_active=2",
        f"trainer.profiling.profile_freq={25000 + args.warmup + 2}",
    ]
train_args = ["--sft-toml=" + BASE]
if args.dryrun:
    train_args += ["--dryrun"]
train_args += ["--", *opts]
command = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nproc-per-node=8",
    "tools/unified_attn/rank.py",
    *train_args,
]
if args.dryrun:
    command = [sys.executable, "-m", "cosmos_framework.scripts.train", *train_args]
(run / "manifest.json").write_text(
    json.dumps(
        {
            "args": vars(args),
            "command": command,
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "baseline": BASE,
            "initial_step": 25000,
            "environment": {
                k: env[k]
                for k in [
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "PYTHONHASHSEED",
                    "OMP_NUM_THREADS",
                    "TORCHINDUCTOR_COMPILE_THREADS",
                    "USR_COMPILE_CACHE_ROOT",
                ]
            },
            "fa3_provider": env.get("COSMOS_FA3_PROVIDER", "nv"),
        },
        indent=2,
    )
)
# Capture exact tracked sources; commits containing only progress notes may differ between jobs.
tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
source_hashes = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in tracked
    if name.endswith(".py") and (ROOT / name).is_file()
}
(run / "source_hashes.json").write_text(json.dumps(source_hashes, sort_keys=True, indent=2))
(run / "worktree.diff").write_bytes(
    subprocess.check_output(["git", "diff", "HEAD", "--", "cosmos_framework", "tools/unified_attn"], cwd=ROOT)
)
with (run / "train.log").open("w") as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
(run / "exit.json").write_text(json.dumps({"returncode": result.returncode}))
print(run, "exit=", result.returncode, flush=True)
sys.exit(result.returncode)
