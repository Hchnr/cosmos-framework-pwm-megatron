# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Capture one warmed TP4 or FSDP4 update under CUDA profiler/Nsight.

This is a diagnostic capture, not a throughput measurement. The caller must
run under nsys with CUDA profiler capture and check both boundary markers.
"""

import argparse
import functools
import json
import os
import sys
from pathlib import Path


def install_native_capture(benchmark, native, capture, iteration):
    original_prepare = benchmark.PreparedMicrosteps.prepare

    @functools.wraps(original_prepare)
    def prepare(self, loader, slot, *args, **kwargs):
        if slot.update == iteration and slot.micro_step == 0:
            capture.start()
        try:
            return original_prepare(self, loader, slot, *args, **kwargs)
        except BaseException:
            capture.stop(failed=True)
            raise

    benchmark.PreparedMicrosteps.prepare = prepare
    original_load = native.load_experiment_from_toml

    @functools.wraps(original_load)
    def load(*args, **kwargs):
        config = original_load(*args, **kwargs)
        trainer = config.trainer.type
        original_step = trainer.training_step

        @functools.wraps(original_step)
        def step(self, *values, **options):
            try:
                result = original_step(self, *values, **options)
            except BaseException:
                capture.stop(failed=True)
                raise
            if options.get("iteration") == iteration and result[2] == 0:
                capture.stop()
            return result

        trainer.training_step = step
        return config

    native.load_experiment_from_toml = load


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--capture-iteration", type=int, default=1202)
    parser.add_argument("--selective-layers", type=int, default=0)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--shared-input-reduction", action="store_true")
    args, remainder = parser.parse_known_args()
    if "--backend" not in remainder:
        raise ValueError("Benchmark backend is required")
    backend = remainder[remainder.index("--backend") + 1]
    if backend not in ("tp", "fsdp") or args.selective_layers < 0:
        raise ValueError("Invalid backend or selected layer count")
    rank = os.environ.get("RANK", "")
    if not rank.isdecimal():
        raise ValueError("Numeric distributed rank required")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = args.report_dir / f"profile.rank{rank}.json"
    if report.exists():
        raise FileExistsError(report)
    from cosmos_framework.utils.nvtx import configure_nvtx
    from tools import run_tp4_fsdp4_microbatch_benchmark as benchmark
    from tools.profile_tp4_microbatch import install_capture
    from tools.run_tp_pp_flash3_selective_ac import install
    from tools.run_tp_pp_full_update_profile import UpdateCapture

    observation = {"forward": 0, "recompute": 0, "native_selected_layers": 0, "tp_selected_layers": 0}
    error = None
    capture = UpdateCapture()
    try:
        if args.shared_input_reduction and backend == "tp":
            from tools.tp_pp_shared_input_reduction import install as install_shared

            install_shared()
        if args.selective_layers:
            install(args.selective_layers, observation)
        configure_nvtx("module")
        if backend == "tp":
            install_capture(capture, args.capture_iteration)
        else:
            from tools import run_tp_pp_native_reference as native

            install_native_capture(benchmark, native, capture, args.capture_iteration)
        sys.argv = [str(Path(benchmark.__file__).resolve()), *remainder]
        benchmark.main()
        if not capture.finished:
            raise RuntimeError("Training exited before the selected update was captured")
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        with report.open("x") as handle:
            json.dump(
                {
                    "rank": int(rank),
                    "backend": backend,
                    "selected_layers": args.selective_layers,
                    "policy_trace_counts": observation,
                    "capture_started": capture.started,
                    "capture_finished": capture.finished,
                    "failure": error,
                    "timing_acceptance": False,
                },
                handle,
                indent=2,
            )
            handle.write("\n")


if __name__ == "__main__":
    main()
