# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Witness native-state replay before each TP diagnostic forward.

This wrapper is diagnostic-only. It never makes a continuous-trajectory
acceptance claim. A successful witness establishes that saved native master,
Adam moments and step were loaded before each later update, not that the
resulting TP gradients, outputs, or production training are correct.
"""

import argparse
import json
import runpy
import sys
from pathlib import Path

EXPECTED_EXIT = "Native-state replay is diagnostic only; continuous trajectory acceptance remains false"


def unique_option(arguments, option):
    if arguments.count(option) != 1:
        raise ValueError(f"Expected exactly one {option}")
    index = arguments.index(option)
    if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
        raise ValueError(f"Missing value for {option}")
    return arguments[index + 1]


def validate_arguments(arguments):
    if arguments.count("--replay-native-state") != 1 or arguments.count("--observe-all-updates") != 1:
        raise ValueError("A native-state replay must observe all three updates")
    if unique_option(arguments, "--mode") != "tp" or unique_option(arguments, "--tp") != "4":
        raise ValueError("This witness is scoped to TP4")
    reference = Path(unique_option(arguments, "--reference-dir"))
    output = Path(unique_option(arguments, "--output"))
    helper = Path(unique_option(arguments, "--run-path"))
    if not reference.is_dir() or not helper.is_file():
        raise FileNotFoundError("Native reference or target helper is missing")
    return reference.resolve(), output.resolve(), helper.resolve()


def read_report(path):
    report = json.loads(path.read_text())
    if not (
        report.get("updates") == 3
        and report.get("tensor_parallel_size") == 4
        and report.get("pipeline_parallel_size") == 1
        and report.get("replay_native_state") is True
        and report.get("continuous_trajectory") is False
        and report.get("passed") is False
        and len(report.get("gradient_summary", [])) == 3
    ):
        raise ValueError("Output is not the expected three-update replay diagnostic")
    return report


def install_state_witness(reference, witness_dir):
    import torch
    import torch.distributed as dist

    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver
    from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states
    from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import stage_parameter_manifest

    manifest = json.loads((reference / "manifest.json").read_text())
    names = manifest["names"]
    if not manifest.get("passed") or len(names) != 811 or len(set(names)) != len(names):
        raise ValueError("The native reference must have 811 unique completed parameters")
    indices = {name: index for index, name in enumerate(names)}
    witness_dir.mkdir(parents=True, exist_ok=True)
    original = ModelParallelDriver.forward_backward

    def witnessed(driver, batches, loss_fn, *, metadata=None):
        update = driver.iteration
        rank = dist.get_rank()
        witness_path = witness_dir / f"replay_state.update{update}.rank{rank}.json"
        if witness_path.exists():
            raise FileExistsError(witness_path)
        checked = 0
        if update in (1, 2):
            specs = stage_parameter_manifest(driver.stage)
            for entry in adam_parameter_states(driver.stage, driver.optimizer):
                spec = specs[entry.name]
                name = spec.canonical_name.removeprefix("net.")
                index = indices[name]
                snapshot = reference / f"update{update - 1}.parameter{index:04d}.pt"
                saved = torch.load(snapshot, map_location="cpu", weights_only=True, mmap=True)
                for field in ("master", "exp_avg", "exp_avg_sq"):
                    actual = entry.master if field == "master" else entry.state[field]
                    expected = entry.select(spec.select(saved[field])).to(actual.device)
                    if not torch.equal(actual, expected):
                        raise RuntimeError(f"rank {rank}, update {update}, {name}: {field} replay mismatch")
                if entry.step != int(saved["step"].item()):
                    raise RuntimeError(f"rank {rank}, update {update}, {name}: step replay mismatch")
                if not torch.equal(entry.select(entry.model_parameter.detach()), entry.master):
                    raise RuntimeError(f"rank {rank}, update {update}, {name}: compute/master mismatch")
                checked += 1
            if checked == 0:
                raise RuntimeError("No optimizer parameter shards were verified")
        elif update != 0:
            raise RuntimeError(f"Unexpected update {update}")
        with witness_path.open("x") as handle:
            json.dump(
                {
                    "rank": rank,
                    "update": update,
                    "verified_shards": checked,
                    "master_moments_step_compute_exact": update in (1, 2),
                    "continuous_trajectory_acceptance": False,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        return original(driver, batches, loss_fn, metadata=metadata)

    ModelParallelDriver.forward_backward = witnessed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--vjp-entry", type=Path, required=True)
    parser.add_argument("--witness-dir", type=Path, required=True)
    args, remainder = parser.parse_known_args(argv)
    reference, output, target = validate_arguments(remainder)
    vjp_entry = args.vjp_entry.resolve()
    if not vjp_entry.is_file():
        raise FileNotFoundError(vjp_entry)
    import torch.distributed as dist

    install_state_witness(reference, args.witness_dir)
    original_run_path = runpy.run_path

    def checked_run_path(path, *values, **options):
        try:
            return original_run_path(path, *values, **options)
        except SystemExit as error:
            if Path(path).resolve() != target or str(error) != EXPECTED_EXIT:
                raise
            # The target destroys its distributed process group before this
            # expected exit. torchrun still exposes the numeric rank in env.
            import os

            rank = int(os.environ["RANK"])
            report_path = output.with_name(f"{output.stem}.rank{rank}.json")
            read_report(report_path)
            for update in (1, 2):
                witness = json.loads((args.witness_dir / f"replay_state.update{update}.rank{rank}.json").read_text())
                if not witness["master_moments_step_compute_exact"] or witness["verified_shards"] <= 0:
                    raise ValueError("Replay state witness is incomplete")
            print(
                json.dumps({"rank": rank, "diagnostic_expected_exit": True, "continuous_acceptance": False}), flush=True
            )
            return {}

    if dist.is_initialized():
        raise RuntimeError("Witness must be installed before distributed initialization")
    runpy.run_path = checked_run_path
    sys.path.insert(0, str(vjp_entry.parent))
    sys.argv = [str(vjp_entry), *remainder]
    original_run_path(str(vjp_entry), run_name="__main__")


if __name__ == "__main__":
    main()
