"""Torchrun rank entry that isolates compiler caches on node-local storage."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def configure_rank_cache() -> tuple[Path, Path]:
    root_value = os.environ.get("COSMOS_COMPILE_CACHE_ROOT")
    rank = os.environ.get("LOCAL_RANK")
    if not root_value or rank is None or not rank.isdigit():
        raise RuntimeError("COSMOS_COMPILE_CACHE_ROOT and numeric LOCAL_RANK are required")
    root = Path(root_value).resolve(strict=False)
    inductor = root / "torchinductor" / f"rank_{rank}"
    triton = root / "triton" / f"rank_{rank}"
    inductor.mkdir(parents=True, exist_ok=True)
    triton.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)
    os.environ["TRITON_CACHE_DIR"] = str(triton)
    return inductor, triton


def main() -> None:
    configure_rank_cache()
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
