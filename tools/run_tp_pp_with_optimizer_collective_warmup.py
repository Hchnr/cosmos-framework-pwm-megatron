"""Diagnose early optimizer-collective initialization without changing training math."""

import argparse
import functools
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path


def install_collective_warmup(warm):
    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver

    original = ModelParallelDriver.__init__
    if getattr(original, "_cosmos_optimizer_collective_warmup", False):
        raise RuntimeError("Optimizer collective warmup is already installed")

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if hasattr(self, "optimizer_collective_warmup_report"):
            raise RuntimeError("Default driver already initialized optimizer collectives")
        report = warm(self.optimizer)
        self.optimizer_collective_warmup_report = report
        print(json.dumps({"optimizer_collective_warmup": report, "rank": os.environ.get("RANK")}), flush=True)
        directory = os.environ.get("COSMOS_TP_PP_OPTIMIZER_WARMUP_TRACE_DIR")
        if directory:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            target = root / f"optimizer_collective_warmup.rank{os.environ['RANK']}.pid{os.getpid()}.json"
            with target.open("x") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

    initialize._cosmos_optimizer_collective_warmup = True
    ModelParallelDriver.__init__ = initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--run-path", type=Path)
    args, remainder = parser.parse_known_args()
    candidate = os.environ.get("COSMOS_TP_PP_COLLECTIVES_CANDIDATE")
    if candidate:
        spec = importlib.util.spec_from_file_location("_cosmos_collectives_warmup_candidate", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from cosmos_framework.training_backends.megatron_tp_pp import collectives as module

    install_collective_warmup(module.warm_optimizer_collectives)
    sys.argv = [str(args.run_path) if args.run_path else sys.argv[0], *remainder]
    if args.run_path:
        sys.path.insert(0, str(args.run_path.resolve().parent))
        runpy.run_path(str(args.run_path), run_name="__main__")
    else:
        runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
