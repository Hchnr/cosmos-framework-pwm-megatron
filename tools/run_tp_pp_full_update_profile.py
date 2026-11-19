# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Capture one complete update's preparation, compute, communication and optimizer.

VAE setup before the update and checkpoint writes remain outside the capture.
Profiled wall times are diagnostic; use separate unprofiled runs for throughput.
Run under nsys with --capture-range=cudaProfilerApi --kill=none --wait=all.
"""

import argparse
import functools
import json
import os
import runpy
import sys


def rank_barrier():
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


class UpdateCapture:
    def __init__(self, cuda=None, barrier=None):
        if cuda is None:
            import torch

            cuda = torch.cuda
        self.cuda = cuda
        self.barrier = barrier or rank_barrier
        self.started = False
        self.finished = False

    def start(self):
        if not self.started:
            self.cuda.synchronize()
            self.barrier()
            self.cuda.cudart().cudaProfilerStart()
            self.started = True
            print(json.dumps({"full_update_capture": "started", "rank": os.environ.get("RANK")}), flush=True)

    def stop(self, *, failed=False):
        if self.started and not self.finished:
            self.cuda.synchronize()
            if not failed:
                # The first rank's Stop can end a torchrun-wide Nsight capture.
                # Require all ranks' optimizer kernels to finish first.
                self.barrier()
            self.cuda.cudart().cudaProfilerStop()
            self.finished = True
            print(json.dumps({"full_update_capture": "stopped", "rank": os.environ.get("RANK")}), flush=True)


def start_before(function, capture):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        capture.start()
        try:
            return function(*args, **kwargs)
        except BaseException:
            capture.stop(failed=True)
            raise

    return wrapped


def stop_after(function, capture, *, only_update_boundary=False):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except BaseException:
            capture.stop(failed=True)
            raise
        if not only_update_boundary or result[2] == 0:
            capture.stop()
        return result

    return wrapped


def install_tp(capture):
    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver
    from cosmos_framework.training_backends.megatron_tp_pp.parallel_preparation import ParallelPreparation

    ParallelPreparation.prepare = start_before(ParallelPreparation.prepare, capture)
    # Stop and retain a partial capture if compute fails before optimizer.step.
    original = ModelParallelDriver.forward_backward

    @functools.wraps(original)
    def forward_backward(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except BaseException:
            capture.stop(failed=True)
            raise

    ModelParallelDriver.forward_backward = forward_backward
    ModelParallelDriver.step = stop_after(ModelParallelDriver.step, capture)


def install_native(module, capture):
    module.prepare_slot = start_before(module.prepare_slot, capture)
    load = module.load_experiment_from_toml

    @functools.wraps(load)
    def load_config(*args, **kwargs):
        config = load(*args, **kwargs)
        trainer = config.trainer.type
        trainer.training_step = stop_after(trainer.training_step, capture, only_update_boundary=True)
        return config

    module.load_experiment_from_toml = load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--backend", choices=("tp", "native"), required=True)
    args, remainder = parser.parse_known_args()
    if os.environ.get("COSMOS_TP_PP_PROFILE_SLOTS") or os.environ.get("COSMOS_TP_PP_PROFILE_ITERATION"):
        raise ValueError("Full-update capture cannot overlap the short-window profiler")
    if args.backend == "tp" and os.environ.get("COSMOS_TP_PP_PARALLEL_PREPARATION") != "1":
        raise ValueError("Full TP capture requires the parallel preparation path")
    from cosmos_framework.utils.nvtx import configure_nvtx

    configure_nvtx("module")
    capture = UpdateCapture()
    sys.argv = [sys.argv[0], *remainder]
    if args.backend == "tp":
        install_tp(capture)
        runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
    else:
        from tools import run_tp_pp_native_reference as native

        install_native(native, capture)
        native.main()
    if not capture.finished:
        raise RuntimeError("Training exited without completing the full-update profiler boundary")


if __name__ == "__main__":
    main()
