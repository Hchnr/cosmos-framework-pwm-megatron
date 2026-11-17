"""Run one stock-trainer rank under Nsight; preserve one report per rank."""

import os
import sys
from pathlib import Path

command = [sys.executable, "tests/megatron/train_real_batch_smoke.py", *sys.argv[1:]]
if os.environ.get("NSYS_CAPTURE", "1") == "1":
    destination = Path(os.environ["NSYS_RUN_DIR"]) / f"rank{int(os.environ['RANK'])}"
    sampling = []
    if os.environ.get("PROFILE_PYTHON_SAMPLING", "0") == "1":
        sampling = ["--python-sampling=true", "--python-sampling-frequency=250"]
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,nccl",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=false",
        *sampling,
        "--output",
        str(destination),
        *command,
    ]
os.execvp(command[0], command)
