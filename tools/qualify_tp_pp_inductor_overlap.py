# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the four-update TP/PP network oracle with observed Inductor overlap.

Example: torchrun --standalone --nproc-per-node=2
  tools/qualify_tp_pp_inductor_overlap.py --tp 2 --pp 1
  --report-dir /tmp/tp-overlap --cache-root /tmp/tp-overlap-cache
  --output /tmp/tp-overlap-oracle.json

Each rank must observe the scheduler on a collective-containing graph. Compiler
caches are disabled for this qualification; it is not a performance benchmark.
"""

import argparse
import functools
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.run_tp_pp_inductor_overlap import Observation, candidate_options, install


def create_rank_cache(cache_root, rank, environ, *, disable_caches=True):
    if not rank.isdecimal():
        raise ValueError("Rank must be numeric")
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / f"rank{rank}"
    # Do not reuse or delete an existing cache: an explicit failure preserves
    # the evidence boundary even when a previous run was interrupted.
    cache.mkdir(exist_ok=False)
    environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    if disable_caches:
        environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        environ["TORCH_COMPILE_FORCE_DISABLE_CACHES"] = "1"
    return cache


def install_oracle_compile(observation, *, torch_module=None, wrapper_type=None):
    if torch_module is None:
        import torch as torch_module
    if wrapper_type is None:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        wrapper_type = CheckpointWrapper
    original = torch_module.compile
    config = torch_module._inductor.config
    if not torch_module.compiler.config.force_disable_caches or not config.force_disable_caches:
        raise RuntimeError("Torch did not activate force_disable_caches from the environment")

    @functools.wraps(original)
    def compile_block(model=None, *args, **kwargs):
        if isinstance(model, wrapper_type):
            options = candidate_options(
                SimpleNamespace(enabled=True),
                kwargs.get("options") or {},
                deterministic_algorithms=torch_module.are_deterministic_algorithms_enabled(),
                inductor_deterministic=config.deterministic,
            )
            kwargs["options"] = options
            observation.data["options_installed"] = True
            observation.data["effective_block_options"] = options
            observation.data["qualification_caches_disabled"] = True
            observation.save()
        return original(model, *args, **kwargs)

    torch_module.compile = compile_block


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tp", type=int, default=2, choices=(1, 2))
    parser.add_argument("--pp", type=int, default=1, choices=(1, 2))
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--shared-input-reduction", action="store_true")
    args, remainder = parser.parse_known_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != args.tp * args.pp:
        parser.error("Require physical DP1: WORLD_SIZE must equal TP times PP")
    rank = os.environ.get("RANK", "")
    if not rank.isdecimal():
        parser.error("torchrun must provide a numeric RANK")
    if "torch" in sys.modules:
        raise RuntimeError("Fresh compiler-cache environment must be configured before importing torch")
    cache = create_rank_cache(args.cache_root, rank, os.environ)
    observation = Observation(args.report_dir / f"inductor_overlap_oracle.rank{rank}.json")
    observation.data["fresh_rank_cache"] = str(cache)
    observation.data["triton_cache"] = os.environ["TRITON_CACHE_DIR"]
    observation.data["force_disable_caches_env"] = os.environ["TORCH_COMPILE_FORCE_DISABLE_CACHES"]
    observation.save()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
        import tp_pp_network_smoke as smoke

        smoke.ModelParallelConfig = functools.partial(
            smoke.ModelParallelConfig, tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=args.pp
        )
        if args.shared_input_reduction:
            from tools.tp_pp_shared_input_reduction import install as install_shared_input

            install_shared_input()
        install(observation)
        install_oracle_compile(observation)
        sys.argv = [sys.argv[0], "--compile", *remainder]
        smoke.main()
    except BaseException as error:
        observation.finish(error=f"{type(error).__name__}: {error}")
        raise
    observation.finish()
    if not observation.data["effect_verified"]:
        raise SystemExit(observation.data["failure"])


if __name__ == "__main__":
    main()
