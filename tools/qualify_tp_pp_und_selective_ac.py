# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Two-GPU, four-update, fixed-capacity UND selective-AC network oracle.

Runs production I/O and TP blocks versus a native tiny model. Both UND buckets,
missing action, and missing noisy targets are required. Never claim this tiny
qualification as full-model correctness or performance acceptance.
"""

import argparse
import functools
import importlib
import json
import os
import sys
from pathlib import Path


def configure_native_buckets(net):
    net.config.fixed_sequence_length_enabled = True
    net.config.fixed_und_tokens = 8
    net.config.fixed_gen_tokens = 8
    net.config.fixed_action_tokens = 2
    net.config.action_tower_und_short_tokens = 4
    model = net.language_model.model
    model.small_action_fixed_lengths = (8, 6, 2)
    model.small_action_und_short_tokens = 4
    model._small_action_warm_full_und = True
    return net


def validate(observation, exercise, smoke):
    if not smoke.get("passed") or not smoke.get("compiled"):
        raise ValueError("Compiled network oracle did not pass")
    if observation.data["stage_count"] != 1 or observation.data["selected_blocks"] != 2:
        raise ValueError("Expected both blocks of the single TP2 stage to use the selective policy")
    groups = smoke["groups"]
    if (len(groups["tp"]), len(groups["pp"]), len(groups["dp"])) != (2, 1, 1):
        raise ValueError("Expected true TP2/PP1/DP1")
    if not exercise["missing_action"] or not exercise["missing_noisy_targets"]:
        raise ValueError("Missing-modality test cases did not execute")
    for token in (4, 8):
        if not exercise["actual_und_tokens"].get(str(token)):
            raise ValueError(f"UND bucket {token} did not actually execute")
        if not observation.data["saved_mm_by_tokens"].get(str(token)):
            raise ValueError(f"UND bucket {token} did not trigger checkpoint policy")
    errors = smoke["relative_gradient_l2"]
    for update in range(4):
        names = [name for name in errors if name.startswith(f"{update}/")]
        if not names or not any("action_block_moe_gen_action" in name for name in names):
            raise ValueError(f"Missing per-parameter action-gradient coverage for update {update}")
    return {"max_relative_gradient_l2": max(errors.values()), "gradient_entries": len(errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--shared-input-reduction", action="store_true")
    args = parser.parse_args()
    rank = os.environ.get("RANK", "")
    if not rank.isdecimal() or int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise ValueError("Run this qualification with torchrun --nproc-per-node=2")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests/megatron"))
    import tp_pp_network_smoke as smoke
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
    from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from tools.run_tp_pp_und_selective_ac import Observation, make_policy

    target = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    if target.exists():
        raise FileExistsError(target)
    observation = Observation(args.report_dir, rank)
    summary_path = args.report_dir / f"und_selective_oracle.rank{rank}.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    exercise = {"missing_action": 0, "missing_noisy_targets": 0, "actual_und_tokens": {}}
    try:
        if args.shared_input_reduction:
            from tools.tp_pp_shared_input_reduction import install as install_shared

            install_shared()
        original_native = smoke.make_native
        original_sequence = smoke.make_sequence
        original_stage = PipelineStage.__init__
        original_first = FirstStageInputs.forward

        def make_native():
            net, shape = original_native()
            return configure_native_buckets(net), shape

        def make_sequence(**kwargs):
            exercise["missing_action"] += int(not kwargs.get("action", True))
            exercise["missing_noisy_targets"] += int(
                not kwargs.get("noisy_vision", True) and not kwargs.get("noisy_action", True)
            )
            return original_sequence(**kwargs)

        def stage_init(self, *stage_args, **stage_kwargs):
            original_stage(self, *stage_args, **stage_kwargs)
            observation.data["stage_count"] += 1
            if self.input_adapter is not None:
                self.input_adapter._warm_full_und = True

        def input_forward(self, *forward_args, **forward_kwargs):
            result = original_first(self, *forward_args, **forward_kwargs)
            actual = tuple(result[key].shape[0] for key in ("causal_seq", "full_only_seq", "action_seq"))
            if actual not in ((4, 6, 2), (8, 6, 2)):
                raise ValueError(f"Tiny three-stream capacities do not match their declared buckets: {actual}")
            key = str(actual[0])
            exercise["actual_und_tokens"][key] = exercise["actual_und_tokens"].get(key, 0) + 1
            return result

        smoke.make_native = make_native
        smoke.make_sequence = make_sequence
        smoke.ModelParallelConfig = functools.partial(
            smoke.ModelParallelConfig, tensor_model_parallel_size=2, pipeline_model_parallel_size=1
        )
        PipelineStage.__init__ = stage_init
        FirstStageInputs.forward = input_forward
        checkpoint = importlib.import_module("torch.distributed.algorithms._checkpoint.checkpoint_wrapper")
        original_checkpoint = checkpoint.checkpoint_wrapper
        ac = ActivationCheckpointingConfig(mode="full", preserve_rng_state=True, determinism_check="default")
        lengths = {"und": (8, 4), "video": 6, "action": 2}
        observation.data["token_counts"] = lengths
        policy = make_policy(lengths, observation)

        def checkpoint_with_policy(block, *a, **kw):
            observation.data["selected_blocks"] += 1
            return original_checkpoint(
                block,
                *a,
                context_fn=lambda: create_selective_checkpoint_contexts(policy),
                preserve_rng_state=ac.preserve_rng_state,
                determinism_check=ac.determinism_check,
                **kw,
            )

        checkpoint.checkpoint_wrapper = checkpoint_with_policy
        sys.argv = [sys.argv[0], "--compile", "--output", str(args.output)]
        smoke.main()
        smoke_report = json.loads(target.read_text())
        stats = validate(observation, exercise, smoke_report)
        observation.finish()
        report = {
            "passed": True,
            "full_model_acceptance": False,
            "scope": "TP2/PP1 compiled production-I/O tiny four-update oracle",
            "exercise": exercise,
            "policy_saved_mm_by_tokens": observation.data["saved_mm_by_tokens"],
            "policy_saved_mm_by_shape": observation.data["saved_mm_by_shape"],
            "smoke_report": str(target),
            "checks": "native outputs, per-parameter gradients, clipped norm, Adam masters, moments and step on all four updates",
            **stats,
        }
        with summary_path.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    except BaseException as error:
        observation.finish(error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
