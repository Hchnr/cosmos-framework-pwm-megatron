# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""TP2/PP1 eager FP32 four-update tiny oracle for K/V reconstruction.

TP ranks must follow the same K/V forward and backward usage paths. All-gather
and gradient reductions cannot safely handle rank-divergent use. This is a
numerical diagnostic, not a production fix or a performance benchmark.
"""

import argparse
import functools
import json
import os
import sys
from pathlib import Path


def validate(smoke, observation, exercise):
    if not smoke.get("passed") or smoke.get("compiled") is not False:
        raise ValueError("Expected a passed eager FP32 network oracle")
    if exercise.get("reference_parameter_dtypes") != ["torch.float32"] or observation.data.get(
        "stage_parameter_dtypes"
    ) != ["torch.float32"]:
        raise ValueError("Reference and stage floating parameters must all be FP32")
    groups = smoke["groups"]
    if (len(groups["tp"]), len(groups["pp"]), len(groups["dp"])) != (2, 1, 1):
        raise ValueError("Expected TP2/PP1/DP1")
    if not exercise["missing_action"] or not exercise["missing_noisy_targets"]:
        raise ValueError("The missing-modality cases did not execute")
    modules = observation.data["modules"]
    if not modules or not any(item["forward_calls"] for item in modules.values()):
        raise ValueError("No K/V reconstruction forward calls were observed")
    if not any("k_proj" in name and info["forward_calls"] for name, info in modules.items()):
        raise ValueError("No reconstructed K forward observed")
    if not any("v_proj" in name and info["forward_calls"] for name, info in modules.items()):
        raise ValueError("No reconstructed V forward observed")
    errors = smoke["relative_gradient_l2"]
    for update in range(4):
        names = [name for name in errors if name.startswith(f"{update}/")]
        if not names or not any("action_block_moe_gen_action" in name for name in names):
            raise ValueError(f"Missing action per-parameter gradient coverage for update {update}")
    return {"max_relative_gradient_l2": max(errors.values()), "gradient_entries": len(errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--full-ac", action="store_true")
    args = parser.parse_args()
    rank = os.environ.get("RANK", "")
    if not rank.isdecimal() or int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise ValueError("Run with torchrun --nproc-per-node=2")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests/megatron"))
    import tp_pp_network_smoke as smoke

    from tools.tp_pp_fp32_reconstructed_kv import Observation, install

    target = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    summary_path = args.report_dir / f"fp32_reconstructed_kv_oracle.rank{rank}.json"
    if target.exists() or summary_path.exists():
        raise FileExistsError("Oracle output already exists")
    observation = Observation(args.report_dir, rank)
    exercise = {"missing_action": 0, "missing_noisy_targets": 0, "reference_parameter_dtypes": []}
    try:
        original_native = smoke.make_native
        original_sequence = smoke.make_sequence

        def make_native():
            net, shape = original_native()
            exercise["reference_parameter_dtypes"] = sorted(
                {str(parameter.dtype) for parameter in net.parameters() if parameter.is_floating_point()}
            )
            if exercise["reference_parameter_dtypes"] != ["torch.float32"]:
                raise ValueError("Native model has non-FP32 floating parameters")
            return net, shape

        def make_sequence(**kwargs):
            exercise["missing_action"] += int(not kwargs.get("action", True))
            exercise["missing_noisy_targets"] += int(
                not kwargs.get("noisy_vision", True) and not kwargs.get("noisy_action", True)
            )
            return original_sequence(**kwargs)

        smoke.make_native = make_native
        smoke.make_sequence = make_sequence
        smoke.ModelParallelConfig = functools.partial(
            smoke.ModelParallelConfig, tensor_model_parallel_size=2, pipeline_model_parallel_size=1
        )
        install(observation)
        if args.full_ac:
            from tools.run_tp_pp_wide_linear_math import install_tp_checkpointing

            install_tp_checkpointing()
        from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage

        original_stage = PipelineStage.__init__

        def stage_init(self, *stage_args, **stage_kwargs):
            original_stage(self, *stage_args, **stage_kwargs)
            dtypes = sorted({str(parameter.dtype) for parameter in self.parameters() if parameter.is_floating_point()})
            if dtypes != ["torch.float32"]:
                raise ValueError("TP stage has non-FP32 floating parameters")
            observation.data["stage_parameter_dtypes"] = dtypes
            observation.save()

        PipelineStage.__init__ = stage_init
        sys.argv = [sys.argv[0], "--output", str(args.output)]
        smoke.main()
        smoke_report = json.loads(target.read_text())
        stats = validate(smoke_report, observation, exercise)
        observation.finish()
        report = {
            "passed": True,
            "scope": "TP2/PP1 eager FP32 tiny production-I/O four-update diagnostic",
            "full_model_acceptance": False,
            "performance_acceptance": False,
            "rank_consistent_kv_use_required": True,
            "full_activation_checkpointing": args.full_ac,
            "exercise": exercise,
            "modules": observation.data["modules"],
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
