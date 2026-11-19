# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run controlled layer-0 row projection replays in one exclusive GPU slot."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-trace", "tp-trace", "native-bundle", "tp-bundle", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"scope": "captured layer0 row GEMM diagnostic; no full-model acceptance", "cases": {}, "passed": False}
    try:
        for projection in ("video_o", "action_o", "video_down"):
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                "-m",
                "tools.replay_tp_pp_full_width_row_capture",
                "--projection",
                projection,
                "--native-trace",
                str(args.native_trace),
                "--tp-trace",
                str(args.tp_trace),
                "--native-bundle",
                str(args.native_bundle),
                "--tp-bundle",
                str(args.tp_bundle),
                "--output",
                str(args.output / projection),
            ]
            with (args.output / f"{projection}.log").open("x") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=1800)
            case = {"returncode": completed.returncode, "ranks": []}
            result["cases"][projection] = case
            if completed.returncode != 0:
                raise RuntimeError(f"{projection} failed with exit {completed.returncode}")
            for rank in range(4):
                data = json.loads((args.output / projection / f"{projection}.rank{rank}.json").read_text())
                if (
                    data["rank"] != rank
                    or data["projection"] != projection
                    or data["full_model_acceptance"] is not False
                ):
                    raise ValueError("Unexpected replay report identity")
                case["ranks"].append(data)
            case["capture_reproduced"] = all(
                row["native_capture_reproduced"]["exact"] and row["tp_capture_reproduced_with_own_input"]["exact"]
                for row in case["ranks"]
            )
            case["full_width_vjp_exact"] = all(
                row["candidate_full_forward"]["exact"] and row["candidate_dx"]["exact"] and row["candidate_dw"]["exact"]
                for row in case["ranks"]
            )
        result["passed"] = all(
            case["capture_reproduced"] and case["full_width_vjp_exact"] for case in result["cases"].values()
        )
    finally:
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit("A captured GEMM replay or full-width VJP gate failed")


if __name__ == "__main__":
    main()
