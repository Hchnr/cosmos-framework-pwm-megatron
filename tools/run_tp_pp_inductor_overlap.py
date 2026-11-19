# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in Inductor communication scheduling experiment for TP/PP blocks.

The training implementation and optimizer are unchanged. Each rank writes the
requested options and actual scheduler observations, including an explicit
unverified result when compilation caches prevent observing a collective.
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

PASSES = (
    "reorder_communication_preserving_peak_memory",
    "sink_waits_iterative",
    "reorder_communication_preserving_peak_memory",
)
OPTION_ENABLED = "reorder_for_compute_comm_overlap"
OPTION_PASSES = "reorder_for_compute_comm_overlap_passes"


def candidate_options(compile_config, existing, *, deterministic_algorithms, inductor_deterministic):
    if not compile_config.enabled:
        raise ValueError("The overlap candidate requires compiled blocks")
    if deterministic_algorithms or inductor_deterministic:
        raise ValueError("Inductor overlap is disabled in deterministic mode")
    result = dict(existing)
    if OPTION_ENABLED in result or OPTION_PASSES in result:
        raise ValueError("Existing compile options already set the overlap policy")
    result[OPTION_ENABLED] = True
    result[OPTION_PASSES] = list(PASSES)
    return result


class Observation:
    def __init__(self, target):
        self.target = Path(target)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.data = {
            "scope": "opt-in Inductor TP/PP block scheduling; numerical and performance acceptance are separate",
            "requested_options": {OPTION_ENABLED: True, OPTION_PASSES: list(PASSES)},
            "options_installed": False,
            "scheduler_calls": 0,
            "scheduler_calls_with_collectives": 0,
            "collective_nodes_observed": 0,
            "effect_verified": False,
            "failure": None,
        }
        with self.target.open("x") as handle:
            json.dump(self.data, handle, indent=2)
            handle.write("\n")

    def save(self):
        temporary = self.target.with_name(self.target.name + f".{os.getpid()}.partial")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n")
        os.replace(temporary, self.target)

    def finish(self, *, error=None):
        self.data["effect_verified"] = bool(
            self.data["options_installed"] and self.data["scheduler_calls_with_collectives"]
        )
        self.data["failure"] = error
        if not self.data["effect_verified"] and error is None:
            self.data["failure"] = "No collective scheduler invocation observed; cache hit or unsupported path"
        self.save()


def install(observation, *, model_module=None, comms_module=None, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    if model_module is None:
        from cosmos_framework.training_backends.megatron_tp_pp import model as model_module
    if comms_module is None:
        from torch._inductor import comms as comms_module

    original_options = model_module.inductor_compile_options
    original_scheduler = comms_module.reorder_compute_and_comm_for_overlap
    config = torch_module._inductor.config

    def options(compile_config):
        result = candidate_options(
            compile_config,
            original_options(compile_config),
            deterministic_algorithms=torch_module.are_deterministic_algorithms_enabled(),
            inductor_deterministic=config.deterministic,
        )
        observation.data["options_installed"] = True
        observation.data["effective_block_options"] = result
        observation.save()
        return result

    def reorder(nodes):
        if config.deterministic or not config.reorder_for_compute_comm_overlap:
            raise RuntimeError("Scheduler reached the candidate without the requested nondeterministic overlap mode")
        if tuple(config.reorder_for_compute_comm_overlap_passes) != PASSES:
            raise RuntimeError("Scheduler received different overlap passes than requested")
        collective_count = sum(bool(comms_module.is_collective(node.node)) for node in nodes)
        result = original_scheduler(nodes)
        observation.data["scheduler_calls"] += 1
        observation.data["collective_nodes_observed"] += collective_count
        observation.data["scheduler_calls_with_collectives"] += int(collective_count > 0)
        observation.save()
        return result

    model_module.inductor_compile_options = options
    comms_module.reorder_compute_and_comm_for_overlap = reorder


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--fresh-cache-root", type=Path)
    parser.add_argument("--allow-unobserved", action="store_true")
    args, remainder = parser.parse_known_args()
    rank = os.environ.get("RANK", "standalone")
    if not rank.isdecimal() and rank != "standalone":
        parser.error("RANK must be numeric")
    if args.fresh_cache_root is not None:
        if not rank.isdecimal():
            parser.error("Fresh per-rank caches require torchrun's numeric RANK")
        from tools.qualify_tp_pp_inductor_overlap import create_rank_cache

        # A fresh cache proves that this benchmark actually compiled the
        # candidate, without changing normal steady-state cache policy.
        create_rank_cache(args.fresh_cache_root, rank, os.environ, disable_caches=False)
    observation = Observation(args.report_dir / f"inductor_overlap.rank{rank}.json")
    try:
        install(observation)
        sys.path.insert(0, str(args.run_path.resolve().parent))
        sys.argv = [str(args.run_path), *remainder]
        runpy.run_path(str(args.run_path), run_name="__main__")
    except BaseException as error:
        observation.finish(error=f"{type(error).__name__}: {error}")
        raise
    observation.finish()
    if not observation.data["effect_verified"] and not args.allow_unobserved:
        raise SystemExit(observation.data["failure"])


if __name__ == "__main__":
    main()
