# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the equal-microstep TP4 benchmark with shared-input reductions enabled."""

import runpy

from tools.tp_pp_shared_input_reduction import install

if __name__ == "__main__":
    install()
    runpy.run_module("tools.run_tp4_fsdp4_microbatch_benchmark", run_name="__main__")
