"""Run the unchanged full-source math diagnostic with native FP32 clip arithmetic.

This entry is diagnostic only. It preserves the reference, input crops, optimizer
flags and all acceptance gates of the existing math helper.
"""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from tools.tp_pp_native_clip_arithmetic import install_native_clip_arithmetic


def main():
    if "--mode" not in sys.argv or sys.argv[sys.argv.index("--mode") + 1] != "tp":
        raise SystemExit("Native clipping diagnostic requires --mode tp")
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    rank = int(os.environ.get("RANK", "0"))
    helper = Path("tests/megatron/tp_pp_full_source_math.py").resolve()
    paths = [helper, Path(__file__), Path("tools/tp_pp_native_clip_arithmetic.py")]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    install_native_clip_arithmetic()
    # Match direct execution of the original helper, including its crop imports.
    sys.path.insert(0, str(helper.parent))
    sys.argv[0] = str(helper)
    completed = False
    try:
        runpy.run_path(str(helper), run_name="__main__")
        completed = True
    finally:
        after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        report = {
            "native_clip_arithmetic": True,
            "helper_completed_successfully": completed,
            "source_unchanged": before == after,
            "source_hashes": before,
            "scope": "diagnostic only; original helper gates and inputs unchanged",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_name(f"{output.stem}.native_clip.rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        if before != after:
            raise RuntimeError("Native clipping diagnostic source changed during execution")


if __name__ == "__main__":
    main()
