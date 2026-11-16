# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export and validate two complete rank-zero optimizer steps in a real trace."""

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
run = args.run.resolve()
report = run / "rank0.nsys-rep"
assert report.is_file() and report.stat().st_size > 0, report
sqlite = run / "rank0.sqlite"
with (run / "nsys_export.log").open("a") as log:
    if not sqlite.exists():
        subprocess.run(
            ["nsys", "export", "--type=sqlite", "--output=" + str(sqlite), str(report)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    subprocess.run(
        [
            "nsys",
            "stats",
            "--report=cuda_gpu_kern_sum,nvtx_sum,cuda_api_sum",
            "--format=csv",
            "--output=" + str(run / "stats"),
            str(report),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
db = sqlite3.connect(str(sqlite))
tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert "CUPTI_ACTIVITY_KIND_KERNEL" in tables and "NVTX_EVENTS" in tables
kernel_count = db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
assert kernel_count > 0
ranges = db.execute("""SELECT coalesce(s.value,n.text), n.start, n.end
                       FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
                       WHERE n.end IS NOT NULL""").fetchall()
micros = [
    (name, start, end)
    for name, start, end in ranges
    if name and re.fullmatch(r"cosmos.train.rank_000.step_\d+\.micro_\d+_of_02", name)
]
optimizers = [(name, start, end) for name, start, end in ranges if name == "optimizer_step"]
assert len(micros) == 4, micros
assert len(optimizers) == 2, optimizers
manifest = json.loads((run / "manifest.json").read_text())
first = manifest["initial_step"] + manifest["args"]["warmup"] + 1
expected = {
    f"cosmos.train.rank_000.step_{step:06d}.micro_{micro:02d}_of_02" for step in (first, first + 1) for micro in (1, 2)
}
assert {m[0] for m in micros} == expected, (micros, expected)
compilation = [
    (name, start, end)
    for name, start, end in ranges
    if name and any(x in name.lower() for x in ("compile_fx", "graphlowering", "autotun"))
]
digest = hashlib.sha256()
with report.open("rb") as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
result = {
    "report": str(report),
    "sha256": digest.hexdigest(),
    "bytes": report.stat().st_size,
    "kernel_count": kernel_count,
    "optimizer_updates": len(optimizers),
    "microbatches": micros,
    "possible_compilation_ranges": compilation,
    "actual_optimizer_steps": [first, first + 1],
    "warning": "Absence of compile-labelled NVTX ranges alone is not proof of zero JIT; review CUDA API and timeline.",
}
(run / "nsys_review.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
