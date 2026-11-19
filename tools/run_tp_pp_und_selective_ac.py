# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in row-count heuristic for checkpointing selected mm outputs.

An mm is tagged for saving when its first input has the same row count as a
declared UND bucket. This is a shape heuristic, not proof of tensor provenance:
other streams or intermediates may have the same row count. Only static
fixed-capacity three-stream models with distinct declared capacities are
eligible. Actual compiler decisions and memory use require profiling.
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def buckets(model_config):
    fixed = model_config.fixed_sequence_length
    if not fixed.enabled:
        raise ValueError("UND selective AC requires fixed sequence capacities")
    full = fixed.und_tokens
    short = model_config.action_tower_und_short_tokens
    video = fixed.gen_tokens - fixed.action_tokens
    action = fixed.action_tokens
    values = (full, video, action)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("UND, VIDEO and ACTION token counts must be positive static integers")
    if not isinstance(short, int) or short < 0 or (short and short >= full):
        raise ValueError("Short UND bucket must be zero or strictly smaller than full UND")
    und = (full, short) if short else (full,)
    if len(set((*und, video, action))) != len(und) + 2:
        raise ValueError("UND, VIDEO and ACTION token counts must not collide")
    return {"und": und, "video": video, "action": action}


def should_save_mm(func, args, lengths):
    name = getattr(func, "__name__", str(func))
    if name != "mm.default":
        return False
    if len(args) < 2 or getattr(args[0], "ndim", None) != 2 or getattr(args[1], "ndim", None) != 2:
        raise ValueError("UND selective AC expected a two-dimensional mm")
    tokens = args[0].shape[0]
    if not isinstance(tokens, int):
        raise ValueError("UND selective AC requires a static mm row dimension")
    return tokens in lengths["und"]


class Observation:
    def __init__(self, directory, rank):
        if not str(rank).isdecimal():
            raise ValueError("Numeric torchrun rank required")
        self.path = Path(directory) / f"und_selective_ac.rank{rank}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {
            "rank": int(rank),
            "scope": "checkpoint policy tracing; saved kernel behavior requires profiling",
            "classification": "mm_input_row_count_heuristic",
            "provenance_verified": False,
            "matching_non_und_ops_possible": True,
            "stage_count": 0,
            "selected_blocks": 0,
            "token_counts": None,
            "saved_mm_by_tokens": {},
            "saved_mm_by_shape": {},
            "seen_mm_by_shape": {},
            "saved_mm_by_shape_and_requires_grad": {},
            "completed": False,
            "failure": None,
        }
        with self.path.open("x") as handle:
            json.dump(self.data, handle, indent=2)
            handle.write("\n")

    def save(self):
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.partial")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n")
        os.replace(tmp, self.path)

    def finish(self, error=None):
        if error is None and not self.data["saved_mm_by_tokens"]:
            error = "No row-matched mm save was observed; compilation cache or unsupported capture path"
        self.data["completed"] = error is None
        self.data["failure"] = error
        self.save()
        if error is not None and error.startswith("No row-matched mm save"):
            raise RuntimeError(error)


def make_policy(lengths, observation):
    from torch.utils.checkpoint import CheckpointPolicy

    def policy(ctx, func, *args, **kwargs):
        selected = should_save_mm(func, args, lengths)
        if getattr(func, "__name__", str(func)) == "mm.default":
            key = f"{tuple(args[0].shape)} @ {tuple(args[1].shape)}"
            seen = observation.data["seen_mm_by_shape"]
            seen[key] = seen.get(key, 0) + 1
        if selected:
            tokens = str(args[0].shape[0])
            counts = observation.data["saved_mm_by_tokens"]
            counts[tokens] = counts.get(tokens, 0) + 1
            shapes = observation.data["saved_mm_by_shape"]
            shapes[key] = shapes.get(key, 0) + 1
            flags = tuple(bool(arg.requires_grad) for arg in args[:2])
            annotated = f"{key}; input_requires_grad={flags}"
            details = observation.data["saved_mm_by_shape_and_requires_grad"]
            details[annotated] = details.get(annotated, 0) + 1
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.MUST_RECOMPUTE

    return policy


def wrap_block(block, ac, policy):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    return checkpoint_wrapper(
        block,
        context_fn=lambda: create_selective_checkpoint_contexts(policy),
        preserve_rng_state=ac.preserve_rng_state,
        determinism_check=ac.determinism_check,
    )


def install(observation, layer_count):
    import torch
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from cosmos_framework.training_backends.megatron_tp_pp import model, train

    def optimize(stage, model_config):
        ac, cfg = model_config.activation_checkpointing, model_config.compile
        if (
            ac.mode != "full"
            or not cfg.enabled
            or cfg.compiled_region != "language"
            or cfg.use_cuda_graphs
            or cfg.compile_dynamic
        ):
            raise ValueError("UND selective AC requires the static compiled per-block full-AC baseline")
        lengths = buckets(model_config)
        if not 0 < layer_count <= len(stage.layers):
            raise ValueError("Selective layer count must fit every pipeline stage")
        names = tuple(stage.layers)[-layer_count:]
        observation.data["token_counts"] = lengths
        observation.data["stage_count"] += 1
        observation.data["selected_blocks"] += len(names)
        policy = make_policy(lengths, observation)
        options = model.inductor_compile_options(cfg)
        for name, block in list(stage.layers.items()):
            if name in names:
                block = wrap_block(block, ac, policy)
            else:
                block = checkpoint_wrapper(
                    block, preserve_rng_state=ac.preserve_rng_state, determinism_check=ac.determinism_check
                )
            stage.layers[name] = torch.compile(block, fullgraph=True, dynamic=False, options=options or None)
        observation.save()

    train.optimize_stage = optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--selective-layers", type=int, required=True)
    parser.add_argument("--shared-input-reduction", action="store_true")
    parser.add_argument(
        "--allow-row-count-heuristic",
        action="store_true",
        help="Acknowledge that matching mm rows do not prove UND provenance",
    )
    args, remainder = parser.parse_known_args()
    if not args.allow_row_count_heuristic:
        parser.error("This candidate requires --allow-row-count-heuristic; mm provenance is not verified")
    observation = Observation(args.report_dir, os.environ.get("RANK", ""))
    try:
        if args.shared_input_reduction:
            from tools.tp_pp_shared_input_reduction import install as install_shared

            install_shared()
        install(observation, args.selective_layers)
        sys.path.insert(0, str(args.entry.resolve().parent))
        sys.argv = [str(args.entry), *remainder]
        runpy.run_path(str(args.entry), run_name="__main__")
    except BaseException as error:
        observation.finish(error=f"{type(error).__name__}: {error}")
        raise
    observation.finish()


if __name__ == "__main__":
    main()
