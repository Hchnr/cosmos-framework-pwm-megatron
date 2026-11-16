# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Profile rank zero while every rank uses the same experiment entry."""

import os
import sys
from pathlib import Path

command = [sys.executable, "-m", "tools.ac_attn.train_entry", *sys.argv[1:]]
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
