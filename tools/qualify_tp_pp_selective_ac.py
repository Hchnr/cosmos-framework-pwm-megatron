# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the four-update MCore network oracle with native selective AC enabled."""

import argparse
import functools
import importlib
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--save-op-regex", action="append", required=True)
    args, remainder = parser.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    import tp_pp_network_smoke as smoke

    from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
    from cosmos_framework.model.generator.mot.parallelize_unified_mot import _apply_selective_ac
    from tools.run_tp_pp_selective_ac import selective_config

    ac = selective_config(ActivationCheckpointingConfig(mode="full"), args.save_op_regex)
    checkpoint_module = importlib.import_module("torch.distributed.algorithms._checkpoint.checkpoint_wrapper")
    checkpoint_module.checkpoint_wrapper = lambda module, *a, **kw: _apply_selective_ac(module, ac)
    smoke.ModelParallelConfig = functools.partial(
        smoke.ModelParallelConfig, tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=args.pp
    )
    sys.argv = [sys.argv[0], *remainder]
    smoke.main()


if __name__ == "__main__":
    main()
