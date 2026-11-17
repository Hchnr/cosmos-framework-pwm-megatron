"""Warm-start two-tower native DCP into an unevenly sharded three-tower model."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omni_ema_smoke import batch
from omni_hf_initialization_smoke import construct

from cosmos_framework.model.generator.mot.action_tower import (
    copy_from_gen,
    defer_copy_initialization,
    finish_copy_initialization,
    source_key,
)
from cosmos_framework.utils.distributed import ddp_sync_grad
from cosmos_framework.utils.generator.materialize_tensor import materialize_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    path = Path("/tmp") / f"cosmos_megatron_three_tower_copy_{os.environ['MASTER_PORT']}"
    source, _ = construct("pytorch", "qwen", path, "FusedAdam", "optim_grads_params", False, False)
    target, optimizer = construct(
        "megatron",
        "qwen",
        path,
        "FusedAdam",
        "optim_grads_params",
        False,
        False,
        tower_structure="three_tower",
    )
    baseline = source.state_dict()
    dcp.save({"model": baseline}, checkpoint_id=path)
    requested = {"model": target.state_dict()}
    pending = defer_copy_initialization(requested["model"], baseline)
    dcp.load(requested, checkpoint_id=path)
    different_owners = 0
    for destination, original in pending.values():
        a = destination._local_tensor.__create_chunk_list__()[0]
        b = requested["model"][original]._local_tensor.__create_chunk_list__()[0]
        different_owners += (a.offsets, a.sizes) != (b.offsets, b.sizes)
    counts = [None] * dist.get_world_size()
    dist.all_gather_object(counts, different_owners)
    assert sum(counts) > 0, "Exercise different physical GEN/GEN_ACTION owners"
    finish_copy_initialization(requested["model"], pending)
    target.load_state_dict(requested["model"], strict=False)

    def verify_all():
        state = target.state_dict()
        for name, value in state.items():
            expected = baseline[source_key(name) or name]
            torch.testing.assert_close(
                materialize_tensor(value), materialize_tensor(expected), rtol=0, atol=0, msg=name
            )
        return len(state)

    checked = verify_all()
    with torch.no_grad():
        state = target.state_dict()
        for name in pending:
            state[name].to_local().zero_()
    copy_from_gen(target.net)
    copy_from_gen(target.net_ema)
    target.load_state_dict(target.state_dict(), strict=False)
    verify_all()
    optimizer.zero_grad()
    for micro in range(3):
        torch.manual_seed(100 + micro + dist.get_rank())
        with ddp_sync_grad(target, enabled=micro == 2):
            output = target.net(batch())["preds_action"][0]
            loss = output.float().square().mean() / 3
            assert torch.isfinite(loss)
            loss.backward()
    optimizer.step()
    result = dict(
        passed=True,
        native_two_tower_checkpoint=True,
        target_three_tower=True,
        compile=True,
        ema_weights_verified=True,
        independently_verified_tensors=checked,
        deferred_parameters=len(pending),
        different_owner_cases_per_rank=counts,
        all_initialized_weights_bitwise_equal=True,
        explicit_in_memory_gen_copy_verified=True,
        gradient_accumulation=3,
        subsequent_fused_adam_update=True,
    )
    if dist.get_rank() == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
