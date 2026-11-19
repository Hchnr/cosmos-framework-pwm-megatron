"""Opt-in selective activation offload without changing canonical parameter names."""

import json
import os
import runpy
import tempfile
from pathlib import Path

from tools.tp_pp_selective_activation_offload import wrap_blocks


def main():
    import torch.distributed as dist

    from cosmos_framework.training_backends.megatron_tp_pp import train
    from cosmos_framework.training_backends.megatron_tp_pp.parallel_preparation import ParallelPreparation

    count = int(os.environ["COSMOS_TP_PP_ACTIVATION_OFFLOAD_BLOCKS"])
    if count < 1:
        raise ValueError("Selective activation offload requires a positive block count")
    original_optimize = train.optimize_stage

    def optimize(stage, config):
        original_optimize(stage, config)
        selected = wrap_blocks(stage.layers.items(), count)
        print(json.dumps({"rank": dist.get_rank(), "selective_activation_offload_blocks": selected}), flush=True)

    original_prepare = ParallelPreparation.prepare

    def prepare(self, slots, *args, **kwargs):
        result = original_prepare(self, slots, *args, **kwargs)
        if self.owner:
            target = self.root.parent / f"replay_plan.iter{slots[0].update:09d}.rank{dist.get_rank()}.json"
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=target.parent, suffix=".partial", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    json.dump(result[0], handle, indent=2)
                    handle.write("\n")
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return result

    train.optimize_stage = optimize
    ParallelPreparation.prepare = prepare
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
