# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Diagnostic training entry with saved-tensor CPU offload at original capacity.

This keeps the model, input, optimizer, and schedule unchanged. It is an opt-in
memory diagnostic, and must not be reported as baseline-aligned performance.
"""

import functools
import json
import os
import runpy

import torch


def activation_offload_context(policy):
    if policy == "preserve_layout":
        from tools.tp_pp_activation_offload import saved_tensor_cpu_offload

        return saved_tensor_cpu_offload()
    if policy == "torch_default":
        return torch.autograd.graph.save_on_cpu(pin_memory=True)
    raise ValueError(f"Unknown activation CPU offload policy: {policy}")


def install_activation_offload(policy="preserve_layout"):
    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver

    # Validate before mutating the driver; the legacy policy remains available
    # to reproduce its recorded full-model numerical failure.
    activation_offload_context(policy)
    original = ModelParallelDriver.forward_backward
    if getattr(original, "_cosmos_activation_cpu_offload", False):
        raise RuntimeError("Activation offload is already installed")

    @functools.wraps(original)
    def forward_backward(self, *args, **kwargs):
        with activation_offload_context(policy):
            return original(self, *args, **kwargs)

    forward_backward._cosmos_activation_cpu_offload = True
    ModelParallelDriver.forward_backward = forward_backward


def main():
    policy = os.environ.get("COSMOS_TP_PP_ACTIVATION_OFFLOAD_POLICY", "preserve_layout")
    install_activation_offload(policy)
    print(
        json.dumps(
            {
                "rank": os.environ.get("RANK"),
                "diagnostic_saved_tensor_cpu_offload": True,
                "activation_cpu_offload_policy": policy,
            }
        ),
        flush=True,
    )
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
