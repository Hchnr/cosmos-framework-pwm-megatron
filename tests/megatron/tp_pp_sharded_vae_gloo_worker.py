# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU/Gloo contract worker; no real VAE, NCCL or GPU mathematics."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from tools.tp_pp_sharded_vae_experiment import (
    VisionItem,
    native_item_encoder,
    sharded_native_training_encode,
    sharded_vision_encode,
)


class MockPreparation:
    def __init__(self, fail=False):
        self.fail = fail

    def _encode_vision_item(self, item, *, num_views, frames_per_view):
        if self.fail and int(item.flatten()[0].item()) == 7:
            raise ValueError("intentional worker failure")
        # Return a tensor whose value reflects input identity and camera metadata.
        return item.float() * 2 + num_views + (frames_per_view or 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("normal", "worker_failure", "invalid_owner", "native_factory_error"), required=True
    )
    parser.add_argument("--owner", type=int, default=0)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    # Greedy assignment sends item 0 to rank 0 and item 1 to rank 1.
    items = [
        VisionItem(torch.ones((1, 1, 4, 3, 3))),
        VisionItem(torch.full((1, 1, 3, 3, 3), 7.0)),
        VisionItem(torch.full((1, 1, 2, 2, 2), 3.0), num_views=2, frames_per_view=1),
    ]
    if args.mode == "invalid_owner":
        items[2] = VisionItem(items[2].tensor, num_views=2, frames_per_view=2)
    torch.manual_seed(1234 + rank)
    rng_before = torch.get_rng_state().clone()
    error = None
    result = None
    try:
        if args.mode == "native_factory_error":
            result = sharded_native_training_encode(
                MockPreparation(),
                dist.group.WORLD,
                raw_state_vision=[item.tensor for item in items] if rank == args.owner else None,
                vision_condition_indexes=[[0]] if rank == args.owner else None,
                owner_rank=args.owner,
                device="cpu",
            )
        else:
            result = sharded_vision_encode(
                items if rank == args.owner else None,
                native_item_encoder(MockPreparation(fail=args.mode == "worker_failure")),
                dist.group.WORLD,
                owner_rank=args.owner,
                device="cpu",
            )
    except RuntimeError as exc:
        error = str(exc)
    if not torch.equal(torch.get_rng_state(), rng_before):
        raise AssertionError("VAE orchestration changed the local torch RNG state")
    if args.mode == "normal":
        assert error is None
        if rank == args.owner:
            reference = [
                MockPreparation()._encode_vision_item(
                    x.tensor, num_views=x.num_views, frames_per_view=x.frames_per_view
                )
                for x in items
            ]
            for expected, actual in zip(reference, result, strict=True):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            # Owner's subsequent original RNG draws stay byte-exact and in order.
            noise = [torch.randn(x.shape) for x in reference]
            torch.set_rng_state(rng_before)
            expected_noise = [torch.randn(x.shape) for x in reference]
            assert all(torch.equal(x, y) for x, y in zip(noise, expected_noise, strict=True))
        else:
            assert result is None
    else:
        expected = {
            "worker_failure": "intentional worker failure",
            "invalid_owner": "Camera metadata disagrees",
            "native_factory_error": "Only native full-encode training is supported",
        }[args.mode]
        assert error is not None and expected in error
    target = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    target.write_text(json.dumps({"rank": rank, "mode": args.mode, "passed": True, "error": error}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
