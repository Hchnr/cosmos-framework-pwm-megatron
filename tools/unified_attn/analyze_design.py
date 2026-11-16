# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Read existing evidence only; export kernel attribution and a reviewable plot."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/v0.2.7_unified_attn"
OUT = PLAN / "analysis"
OUT.mkdir(exist_ok=True)
BACKENDS = [
    "split_default",
    "split_fa3_fork",
    "unified_fa3_grouped",
    "split_fa4",
    "unified_fa4_mask",
    "split_flex_triton",
    "unified_flex_triton",
]
rows = []
for backend in BACKENDS:
    run = PLAN / "artifacts/runs" / f"final_l0_v1_profile_r0_{backend}"
    path = run / "trace_breakdown.json"
    data = json.loads(path.read_text())
    totals = {"forward_ms": 0.0, "backward_ms": 0.0, "forward_count": 0, "backward_count": 0}
    matched = []
    for kernel in data["attention_named_kernels"]:
        name = kernel["name"]
        forward = (
            "FlashAttnFwdSm90<" in name
            or "flash_fwd_sm90" in name
            or ("triton_tem_" in name and "flex_attention" in name and "backward" not in name)
        )
        backward = (
            "FlashAttnBwdSm90<" in name
            or "flash_bwd_sm90" in name
            or ("triton_tem_" in name and "flex_attention_backward" in name)
        )
        if forward or backward:
            phase = "forward" if forward else "backward"
            totals[phase + "_ms"] += kernel["gpu_duration_ms_sum"]
            totals[phase + "_count"] += kernel["count"]
            matched.append(kernel)
    assert totals["forward_count"] == (288 if backend.startswith("unified") else 576)
    assert totals["backward_count"] == (144 if backend.startswith("unified") else 288)
    rows.append({"backend": backend, **totals, "source": str(path.relative_to(ROOT)), "kernels": matched})

geometry = json.loads((PLAN / "artifacts/runs/final_l0_v1_profile_r0_split_default/geometry_rank0.json").read_text())
u, g = geometry["split_lens"][::2], geometry["split_lens"][1::2]
assert geometry["attn_modes"] == ["causal", "full"] * len(u)
und_pairs = sum(n * (n + 1) // 2 for n in u)
gen_pairs = sum(v * (t + v) for t, v in zip(u, g, strict=True))

cache = Path("/tmp/cosmos-unified-attn/final_l0_v1_profile_r0_unified_flex_triton/rank0")
ptx_rows = []
for path in sorted((cache / "triton").rglob("*.ptx")):
    if "triton_tem_fused_cat_flex_attention" not in path.name:
        continue
    text = path.read_text()
    ptx_rows.append(
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "static_instruction_text_counts_not_runtime_counts": {
                op: text.count(op) for op in ("wgmma.", "mma.sync", "cp.async.", "cp.async.bulk")
            },
        }
    )
assert len(ptx_rows) == 2
sources = [
    (ROOT / ".venv-unified-attn/lib/python3.13/site-packages/flash_attn/cute/mask.py", [(211, 281)]),
    (ROOT / ".venv-unified-attn/lib/python3.13/site-packages/flash_attn/cute/interface.py", [(216, 262)]),
    (
        ROOT / ".venv-unified-attn/lib/python3.13/site-packages/flash_attn/cute/flash_bwd_sm90.py",
        [(1370, 1400), (1523, 1547)],
    ),
    (
        cache / "sj/csjbbtoxhzae4zqyqccc4cz7uxjvc35imj6eilrdo56xhszxgzbq.py",
        [(8, 14), (110, 152), (450, 460), (510, 541), (655, 665), (715, 750)],
    ),
    (
        Path("/share/project/eai_pwm/home/hcr/repos/flash-attention/plans/v0.0.1_ragged_fa3/summary_v2.md"),
        [(1, 36), (38, 47)],
    ),
]
excerpts = []
for path, ranges in sources:
    text = path.read_text()
    lines = text.splitlines()
    excerpts.append(
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "excerpts": [{"start_line": a, "end_line": b, "text": "\n".join(lines[a - 1 : b])} for a, b in ranges],
        }
    )
evidence = {
    "scope": "rank0 two profiled steps; forward includes AC recomputation; GPU sums are NOT wall times",
    "profiles": rows,
    "first_micro_rank0_geometry": {
        "und_tokens": u,
        "gen_tokens": g,
        "und_pairs": und_pairs,
        "gen_pairs": gen_pairs,
        "und_pair_fraction": und_pairs / (und_pairs + gen_pairs),
        "excludes_padding": True,
    },
    "flex_ptx": ptx_rows,
    "source_excerpts": excerpts,
}
(OUT / "design_evidence.json").write_text(json.dumps(evidence, indent=2))
labels = ["FA3 default", "FA3 split", "FA3 grouped", "FA4 split", "FA4 mixed mask", "Flex split", "Flex unified"]
fig, ax = plt.subplots(figsize=(11, 5.2))
y = list(range(len(rows)))
f = [r["forward_ms"] / 1000 for r in rows]
b = [r["backward_ms"] / 1000 for r in rows]
ax.barh(y, f, color="#4c78a8", label="Forward incl. AC replay")
ax.barh(y, b, left=f, color="#e58b45", label="Backward main kernel")
for i, (fv, bv) in enumerate(zip(f, b, strict=True)):
    ax.text(fv / 2, i, f"{fv:.2f}", va="center", ha="center", color="white")
    ax.text(fv + bv / 2, i, f"{bv:.2f}", va="center", ha="center", color="black")
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("Summed GPU kernel duration (seconds), rank 0 / two optimizer steps")
ax.set_title("Attention main kernels: launch fusion does not imply less compute")
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=2, frameon=False)
fig.text(
    0.02,
    0.01,
    "Not step wall time; helpers excluded; one profile window per backend. Source: design_evidence.json",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(OUT / "attention_kernel_breakdown.svg")
fig.savefig(OUT / "attention_kernel_breakdown.png", dpi=160)
plt.close(fig)
print(json.dumps([{k: v for k, v in r.items() if k != "kernels"} for r in rows], indent=2))
