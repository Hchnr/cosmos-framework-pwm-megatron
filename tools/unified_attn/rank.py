# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Wrap rank zero only in Nsight; all ranks execute identical training code."""

import os
import sys
from pathlib import Path

command = [sys.executable, "-m", "cosmos_framework.scripts.train", *sys.argv[1:]]
if os.environ.get("UNIFIED_ATTN_NSYS") == "1" and int(os.environ["RANK"]) == 0:
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=false",
        "-o",
        str(Path(os.environ["UNIFIED_ATTN_RUN"]) / "rank0"),
        *command,
    ]
os.execvp(command[0], command)
