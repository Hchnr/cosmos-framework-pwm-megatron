# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Qualify opt-in native norm with the existing production-I/O four-update oracle.

This uses a synthetic tiny native meta model and synthetic canonical optimizer
presence metadata. It never reads oracle gradients, norms, parameter values or
optimizer state to construct the order. GPU execution is an explicit later step.
"""

import argparse
import functools
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def tiny_meta_order(make_native, ordered_trainable_names):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            model, _ = make_native()
        if not all(parameter.is_meta for parameter in model.parameters()):
            raise ValueError("Qualification order must come from a meta model")
        entries = {
            "net." + name: {"optimizer_fields": ["master_param"] if parameter.requires_grad else []}
            for name, parameter in model.named_parameters()
        }
        return ordered_trainable_names(model, {"parameters": entries})
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def derive_meta_order_without_cuda(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("Meta-order subprocess requires hidden GPUs and uninitialized CUDA")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    from tp_pp_network_io_test import make_native

    from cosmos_framework.training_backends.megatron_tp_pp.native_fp32_norm import ordered_trainable_names

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized while importing meta-model dependencies")
    names = tiny_meta_order(make_native, ordered_trainable_names)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized while deriving the tiny meta-model order")
    output.write_text(json.dumps({"names": names, "cuda_initialized_before": False, "cuda_initialized_after": False}))


def meta_order_before_parent_cuda_initialization():
    if torch.cuda.is_initialized():
        raise RuntimeError("Meta ordering must be requested before parent CUDA initialization")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    with tempfile.TemporaryDirectory(prefix="native-fp32-meta-order-") as directory:
        output = Path(directory) / "order.json"
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--derive-meta-order", str(output)],
            env=environment,
            check=True,
        )
        result = json.loads(output.read_text())
    if torch.cuda.is_initialized():
        raise RuntimeError("Parent CUDA initialized before the meta order was available")
    if result["cuda_initialized_before"] or result["cuda_initialized_after"]:
        raise RuntimeError("Meta-order subprocess unexpectedly initialized CUDA")
    names = result["names"]
    if not names or len(names) != len(set(names)) or not all(name.startswith("net.") for name in names):
        raise ValueError("Meta-order subprocess returned invalid parameter identities")
    return names


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--derive-meta-order":
        derive_meta_order_without_cuda(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-native-fp32-norm", action="store_true")
    args, remainder = parser.parse_known_args()
    # Importing the real smoke/driver on a GPU-visible process can initialize
    # CUDA. Derive metadata in a GPU-hidden child while this rank is still
    # uninitialized, then load the training dependencies.
    names = meta_order_before_parent_cuda_initialization() if args.enable_native_fp32_norm else None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    import tp_pp_network_smoke as smoke

    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver

    smoke.ModelParallelConfig = functools.partial(
        smoke.ModelParallelConfig, tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=args.pp
    )
    norm_calls = []
    reference_output_dtypes = set()
    stage_parameter_dtypes = set()
    reference_parameter_dtypes = set()
    reference_forwards = []
    if args.enable_native_fp32_norm:
        if "native_fp32_norm_order" not in inspect.signature(ModelParallelDriver.__init__).parameters:
            raise RuntimeError("The driver candidate patch is not installed in this checkout")
        original_make_native = smoke.make_native

        def make_native():
            model, shape = original_make_native()
            reference_parameter_dtypes.update(str(p.dtype) for p in model.parameters() if p.is_floating_point())
            if reference_parameter_dtypes != {"torch.float32"}:
                raise RuntimeError("The reference model must have only FP32 floating-point parameters")

            def observe_output(_module, _inputs, output):
                dtypes = {
                    str(value.dtype)
                    for key in ("preds_vision", "preds_action")
                    for value in output[key]
                    if value.is_floating_point()
                }
                if dtypes != {"torch.float32"}:
                    raise RuntimeError("The reference predictions must be FP32")
                reference_output_dtypes.update(dtypes)
                reference_forwards.append(1)

            model.register_forward_hook(observe_output)
            return model, shape

        smoke.make_native = make_native
        original_init, original_step = ModelParallelDriver.__init__, ModelParallelDriver.step

        def initialize(self, *positional, **keywords):
            if "native_fp32_norm_order" in keywords:
                raise ValueError("Qualification already received a native norm order")
            keywords["native_fp32_norm_order"] = names
            original_init(self, *positional, **keywords)
            if not self._native_fp32_norm_enabled:
                raise RuntimeError("Candidate norm was not enabled")
            stage_parameter_dtypes.update(str(p.dtype) for p in self.stage.parameters() if p.is_floating_point())
            if stage_parameter_dtypes != {"torch.float32"}:
                raise RuntimeError("The TP stage must have only FP32 floating-point parameters")

        def step(self, *positional, **keywords):
            before = self._native_fp32_norm_calls
            result = original_step(self, *positional, **keywords)
            if result[0]:
                if self._native_fp32_norm_calls != before + 1:
                    raise RuntimeError("Expected one native norm per successful update")
                norm_calls.append(self._native_fp32_norm_calls)
            return result

        ModelParallelDriver.__init__, ModelParallelDriver.step = initialize, step
    sys.argv = [sys.argv[0], "--output", str(args.output), *remainder]
    smoke.main()
    if args.enable_native_fp32_norm and len(norm_calls) != 4:
        raise RuntimeError("Expected four completed updates with the candidate norm")
    if args.enable_native_fp32_norm and len(reference_forwards) != 48:
        raise RuntimeError("Expected 48 reference forwards across four 12-slot updates")
    report = {
        "passed": True,
        "native_fp32_norm_enabled": args.enable_native_fp32_norm,
        "successful_candidate_norm_calls": norm_calls,
        "reference_parameter_dtypes": sorted(reference_parameter_dtypes),
        "stage_parameter_dtypes": sorted(stage_parameter_dtypes),
        "reference_output_dtypes": sorted(reference_output_dtypes),
        "reference_forward_count": len(reference_forwards),
        "order_derived_before_parent_cuda_initialization": bool(args.enable_native_fp32_norm),
        "order_child_cuda_hidden_and_uninitialized": bool(args.enable_native_fp32_norm),
        "order_parameter_count": len(names) if names is not None else 0,
        "order_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
        if names is not None
        else None,
        "order_source": "tiny native meta model plus synthetic canonical optimizer-presence metadata",
        "full_model_acceptance": False,
    }
    target = args.output.with_name(f"{args.output.stem}.native_norm.rank{os.environ['RANK']}.json")
    with target.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
