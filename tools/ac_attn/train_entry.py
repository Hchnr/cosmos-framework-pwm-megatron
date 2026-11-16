# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Install explicitly requested experiment adapters before the production entry."""

import os
import runpy

if os.environ.get("AC_FILTER_OVERSIZED") == "1":
    from tools.ac_attn.capacity import install

    install()
target = (
    "tools.ac_attn.deterministic_train"
    if os.environ.get("AC_DETERMINISTIC_ATTENTION") == "1"
    else "cosmos_framework.scripts.train"
)
runpy.run_module(target, run_name="__main__")
