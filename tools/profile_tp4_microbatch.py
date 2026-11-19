# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Capture one warmed TP4 update, including all batch preparation and optimizer.

Run under nsys --capture-range=cudaProfilerApi --capture-range-end=stop
--kill=none --wait=all. Profiled times are diagnostic, never throughput evidence.
"""

import argparse
import functools
import runpy
import sys
from pathlib import Path


def install_capture(capture, iteration, driver_class=None):
    if driver_class is None:
        from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver

        driver_class = ModelParallelDriver
    original_forward = driver_class.forward_backward
    original_step = driver_class.step

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        if self.iteration == iteration:
            capture.start()
        try:
            return original_forward(self, *args, **kwargs)
        except BaseException:
            if capture.started and not capture.finished:
                capture.stop(failed=True)
            raise

    @functools.wraps(original_step)
    def step(self, *args, **kwargs):
        try:
            result = original_step(self, *args, **kwargs)
        except BaseException:
            if capture.started and not capture.finished:
                capture.stop(failed=True)
            raise
        if capture.started and not capture.finished:
            capture.stop(failed=not result[0])
            if not result[0]:
                raise RuntimeError("The captured TP4 optimizer update failed")
        return result

    driver_class.forward_backward = forward
    driver_class.step = step


def main():
    from cosmos_framework.utils.nvtx import configure_nvtx
    from tools.run_tp_pp_full_update_profile import UpdateCapture

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--capture-iteration", type=int, default=1202)
    parser.add_argument("--shared-input-reduction", action="store_true")
    args, remainder = parser.parse_known_args()
    if "--backend" not in remainder or remainder[remainder.index("--backend") + 1] != "tp":
        parser.error("This entry captures TP4; select --backend tp")
    if args.shared_input_reduction:
        from tools.tp_pp_shared_input_reduction import install

        install()
    capture = UpdateCapture()
    configure_nvtx("module")
    install_capture(capture, args.capture_iteration)
    sys.path.insert(0, str(args.run_path.resolve().parent))
    sys.argv = [str(args.run_path), *remainder]
    runpy.run_path(str(args.run_path), run_name="__main__")
    if not capture.finished:
        raise RuntimeError("Training exited without capturing the selected complete update")


if __name__ == "__main__":
    main()
