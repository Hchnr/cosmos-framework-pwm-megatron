# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Four-process Gloo regression for the real VAE loader's subgroup root.

The production dtype-agreement helper allocates on CUDA. This CPU test replaces
only that precheck with a Gloo object all-gather; the real _video_vae and
sync_model_states/_broadcast_coalesced calls are used with a tiny model.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from cosmos_framework.model.generator.tokenizers import wan2pt2_vae_4x16x16 as wan
from cosmos_framework.utils import distributed as cosmos_dist


def verify_dtypes_on_cpu(group, parameters):
    local = tuple(str(parameter.dtype) for parameter in parameters)
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local, group=group)
    if any(item != local for item in gathered):
        raise RuntimeError("Gloo dtype mismatch")


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise RuntimeError("This regression requires four Gloo ranks")
    singletons = [dist.new_group([i], backend="gloo", timeout=timedelta(seconds=30)) for i in range(4)]
    pairs = [dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=30)) for ranks in ([0, 1], [2, 3])]
    wan.WanVAE_ = lambda **_kwargs: nn.Linear(2, 2)
    wan.TRAINING, wan.INTERNAL = True, True
    cosmos_dist._verify_param_dtype_across_processes = verify_dtypes_on_cpu
    loads = []

    def fake_load(_path, **_kwargs):
        loads.append(rank)
        return {"weight": torch.full((2, 2), float(rank + 1)), "bias": torch.full((2,), float(rank + 11))}

    wan.easy_io.load = fake_load
    for group, expected_source in ((singletons[rank], rank), (pairs[rank // 2], (rank // 2) * 2)):
        before = len(loads)
        model = wan._video_vae("/readonly/fake-vae.pth", device="cpu", process_group=group)
        expected_loads = int(rank == expected_source)
        if len(loads) - before != expected_loads:
            raise AssertionError("Only subgroup rank zero should load weights")
        torch.testing.assert_close(model.weight, torch.full((2, 2), float(expected_source + 1)))
        torch.testing.assert_close(model.bias, torch.full((2,), float(expected_source + 11)))
    # The former global-rank argument is invalid for nonzero singleton groups.
    if rank:
        try:
            dist._broadcast_coalesced(singletons[rank], [torch.ones(1)], 1024, rank)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Historical global-rank root unexpectedly succeeded")
    dist.barrier()
    print(f"VAE_SUBGROUP_GLOO_PASS rank={rank}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
