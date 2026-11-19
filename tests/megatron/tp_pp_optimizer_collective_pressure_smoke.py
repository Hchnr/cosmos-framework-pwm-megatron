"""Diagnose MCore's first gradient norm under CUDA allocator cache pressure.

Run cold and warmed cases in separate four-rank processes. The cold case may
intentionally fail. A successful norm releases the artificial cache pressure,
then continues the existing compiled network/native-Adam numerical checks.
This is not a production-capacity or performance acceptance test.
"""

import argparse
import functools
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver


def reserve_cached_memory(headroom_bytes):
    blocks = []
    granularity = 2 << 20
    while True:
        free, _ = torch.cuda.mem_get_info()
        size = min(256 << 20, (free - headroom_bytes) // granularity * granularity)
        if size < granularity:
            break
        blocks.append(torch.empty(size, dtype=torch.uint8, device="cuda"))
    blocks.clear()
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-collectives", action="store_true")
    parser.add_argument("--headroom-mib", type=int, default=256)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.headroom_mib < 128:
        raise ValueError("Reserve at least 128 MiB raw CUDA headroom for the diagnostic")
    if args.warm_collectives:
        from tools.run_tp_pp_with_optimizer_collective_warmup import install_collective_warmup

        candidate = os.environ["COSMOS_TP_PP_COLLECTIVES_CANDIDATE"]
        spec = importlib.util.spec_from_file_location("_pressure_collectives_candidate", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        install_collective_warmup(module.warm_optimizer_collectives)
    original = ModelParallelDriver.__init__

    @functools.wraps(original)
    def initialize(self, *positional, **keywords):
        original(self, *positional, **keywords)
        get_norm = self.optimizer.get_grad_norm
        tested = False

        @functools.wraps(get_norm)
        def pressure_norm(*positional, **keywords):
            nonlocal tested
            if tested:
                return get_norm(*positional, **keywords)
            tested = True
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "scope": "actual MCore optimizer norm under artificial cached CUDA memory pressure; no production acceptance",
                "rank": dist.get_rank(),
                "warm_collectives": args.warm_collectives,
                "warmup_applied": hasattr(self, "optimizer_collective_warmup_report"),
                "requested_raw_headroom_mib": args.headroom_mib,
                "passed": False,
                "stage": "cache_fill",
            }
            target = args.output.with_name(f"{args.output.stem}.pressure.rank{dist.get_rank()}.json")
            try:
                reserve_cached_memory(args.headroom_mib << 20)
                free, _ = torch.cuda.mem_get_info()
                report.update(
                    stage="norm",
                    free_cuda_bytes_before_norm=free,
                    allocated_cuda_bytes_before_norm=torch.cuda.memory_allocated(),
                    reserved_cuda_bytes_before_norm=torch.cuda.memory_reserved(),
                )
                value = get_norm(*positional, **keywords)
                torch.cuda.synchronize()
                report.update(passed=True, gradient_norm=float(value))
            except BaseException as error:
                report.update(error_type=type(error).__name__, error=str(error))
                with target.open("x") as handle:
                    json.dump(report, handle, indent=2)
                    handle.write("\n")
                print(json.dumps(report), flush=True)
                # An errored NCCL communicator may hang teardown. Retain evidence
                # and let torchrun terminate the other diagnostic ranks.
                os._exit(1)
            with target.open("x") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            print(json.dumps(report), flush=True)
            # The numerical continuation is at normal pressure. This occurs only
            # after the norm under pressure succeeded, never inside the candidate.
            torch.cuda.empty_cache()
            return value

        self.optimizer.get_grad_norm = pressure_norm

    ModelParallelDriver.__init__ = initialize
    network = Path(__file__).with_name("tp_pp_network_smoke.py")
    sys.path.insert(0, str(network.parent.resolve()))
    sys.argv = [str(network), "--output", str(args.output), *(["--compile"] if args.compile else [])]
    runpy.run_path(str(network), run_name="__main__")


if __name__ == "__main__":
    main()
