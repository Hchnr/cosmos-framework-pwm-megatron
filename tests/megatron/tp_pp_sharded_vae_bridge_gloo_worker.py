# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU protocol exercise; launched only by the companion pytest."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from cosmos_framework.utils.generator.rand_state import preserve_training_rng
from tools.run_tp4_sharded_vae_raw_benchmark import PreparationBridge
from tools.tp_pp_sharded_vae_experiment import sharded_native_training_encode


def main():
    dist.init_process_group("gloo")
    rank, group = dist.get_rank(), dist.group.WORLD
    torch.manual_seed(100 + rank)
    random.seed(200 + rank)
    np.random.seed(300 + rank)

    def control(flag, selected_group):
        value = torch.tensor([flag or 0], dtype=torch.int32)
        dist.broadcast(value, src=0, group=selected_group)
        return int(value.item())

    class Preparation:
        def _encode_vision_item(self, item, *, num_views, frames_per_view):
            assert num_views == 1 and frames_per_view is None
            return item.float() * 2 + 1

    prep = Preparation()
    bridge = PreparationBridge(
        group,
        prep,
        owner=rank == 0,
        send_control=control,
        shard_encode=sharded_native_training_encode,
        preserve_rng=preserve_training_rng,
    )
    payloads = [torch.full((1, 2, 2, 2), index, dtype=torch.uint8) for index in range(5)]
    initial_torch = torch.get_rng_state().clone()
    initial_python, initial_numpy = random.getstate(), np.random.get_state()
    rows = []
    for slot in range(3):

        def owner_prepare():
            if slot == 1:
                return []
            first = bridge.owner_encode(payloads, None, None)
            if slot == 2:
                first += bridge.owner_encode(payloads[:2], None, None)
            return first

        if rank == 0:
            output = bridge.prepare(owner_prepare)
            expected = [item.float() * 2 + 1 for item in payloads] if slot != 1 else []
            if slot == 2:
                expected += [item.float() * 2 + 1 for item in payloads[:2]]
            assert len(output) == len(expected)
            assert all(torch.equal(left, right) for left, right in zip(output, expected, strict=True))
        else:
            bridge.serve()
        rows.append(slot)
        dist.barrier()
    assert torch.equal(initial_torch, torch.get_rng_state())
    assert initial_python == random.getstate()
    numpy_after = np.random.get_state()
    assert initial_numpy[0] == numpy_after[0]
    assert np.array_equal(initial_numpy[1], numpy_after[1])
    assert initial_numpy[2:] == numpy_after[2:]
    try:
        if rank == 0:
            bridge.prepare(lambda: (_ for _ in ()).throw(ValueError("owner input error")))
        else:
            bridge.serve()
    except (ValueError, RuntimeError):
        caught = True
    else:
        caught = False
    observed = [None] * dist.get_world_size()
    dist.all_gather_object(observed, caught)
    assert all(observed)
    out = Path(os.environ["BRIDGE_TEST_OUTPUT"])
    (out / f"rank{rank}.json").write_text(
        json.dumps({"rank": rank, "rows": rows, "slots": bridge.slots, "calls": bridge.calls, "caught": caught}) + "\n"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
