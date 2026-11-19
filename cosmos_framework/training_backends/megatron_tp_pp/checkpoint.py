"""Synchronous complete MCore checkpoints with atomic completion publication.

Model parameters use canonical global FQNs and TP axes. Optimizer fragments use
MCore's sharded representation. Loader/data RNG remain explicitly same-topology.
"""

import gc
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from .source import release_file_cache

FORMAT_VERSION = 1
MCORE_CHECKPOINT_VERSION = 3.1


def prune_completed(path, *, keep_last_n, keep_every_n_steps=None):
    """Apply native local retention only after a complete save is latest.

    Symlinks, incomplete writes, future steps and other formats are excluded.
    """
    if keep_last_n is None:
        return
    if isinstance(keep_last_n, bool) or not isinstance(keep_last_n, int) or keep_last_n < 1:
        raise ValueError("keep_last_n must be a positive integer or None")
    if keep_every_n_steps is not None and (
        isinstance(keep_every_n_steps, bool) or not isinstance(keep_every_n_steps, int) or keep_every_n_steps < 1
    ):
        raise ValueError("keep_every_n_steps must be a positive integer or None")
    path = Path(path)
    root = path.parent
    if (
        not re.fullmatch(r"iter_[0-9]{9,}", path.name)
        or path.is_symlink()
        or not (path / ".save_complete").is_file()
        or (root / "latest_checkpoint.txt").read_text().strip() != path.name
    ):
        raise ValueError("Retention requires the completed and published latest checkpoint")
    current = int(path.name[5:])
    candidates = []
    for candidate in root.iterdir():
        if candidate.is_symlink() or not candidate.is_dir() or not re.fullmatch(r"iter_[0-9]{9,}", candidate.name):
            continue
        manifest = candidate / "manifest.json"
        if int(candidate.name[5:]) > current or not (candidate / ".save_complete").is_file() or not manifest.is_file():
            continue
        if json.loads(manifest.read_text()).get("format") == "cosmos_megatron_tp_pp":
            candidates.append(candidate)
    for candidate in sorted(candidates, key=lambda item: int(item.name[5:]), reverse=True)[keep_last_n:]:
        step = int(candidate.name[5:])
        if candidate == path or (keep_every_n_steps is not None and step > 0 and step % keep_every_n_steps == 0):
            continue
        shutil.rmtree(candidate)


def _state(driver, *, rank_state, loading=False):
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    model = driver.stage.sharded_state_dict()
    metadata = {
        "distrib_optim_sharding_type": "dp_reshardable",
        "checkpoint_version": MCORE_CHECKPOINT_VERSION,
    }
    optimizer = driver.optimizer.sharded_state_dict(model, is_loading=loading, metadata=metadata)
    return {
        "model": model,
        "optimizer": optimizer,
        "iteration": driver.iteration,
        "scheduler": driver.scheduler.state_dict() if driver.scheduler is not None else None,
        "rank_state": ShardedObject("rank_state", rank_state, (dist.get_world_size(),), (dist.get_rank(),)),
    }


def _cpu_coordinated_strategy(group):
    """Pinned MCore translation with DCP planning collectives on CPU/Gloo."""
    from megatron.core.dist_checkpointing.strategies.torch import (
        MCoreSavePlanner,
        TorchDistSaveShardedStrategy,
        _get_filesystem_writer,
        _replace_state_dict_keys_with_sharded_keys,
        mcore_to_pyt_state_dict,
    )
    from torch.distributed import checkpoint as dcp

    class CpuCoordinatedSave(TorchDistSaveShardedStrategy):
        def save(self, sharded_state_dict, checkpoint_dir):
            grouped, _, _ = _replace_state_dict_keys_with_sharded_keys(sharded_state_dict, self.keep_only_main_replica)
            dcp.save(
                mcore_to_pyt_state_dict(grouped, False),
                storage_writer=_get_filesystem_writer(checkpoint_dir, thread_count=self.thread_count),
                planner=MCoreSavePlanner(
                    dedup_replicated_tensors=not self.keep_only_main_replica,
                    flatten_state_dict=False,
                    flatten_sharded_tensors=False,
                ),
                process_group=group,
            )

    return CpuCoordinatedSave()


def save(driver, path, *, rank_state, async_save=False):
    from megatron.core import dist_checkpointing

    if async_save:
        raise ValueError("TP/PP full-state checkpoint currently supports synchronous writes only")
    path = Path(path)
    exists = torch.tensor(int(path.exists()) if dist.get_rank() == 0 else 0, dtype=torch.int32, device="cuda")
    dist.broadcast(exists, src=0)
    if exists.item():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    if dist.get_rank() == 0:
        path.mkdir(parents=True)
    dist.barrier()
    driver.model.start_param_sync(force_sync=True)
    # NCCL object collectives can require fresh external CUDA allocations when
    # the allocator holds the original-capacity training cache. Match native DCP
    # CPU coordination, retaining MCore's tensor format/replica deduplication.
    torch.cuda.empty_cache()
    group = dist.new_group(backend="gloo")
    try:
        dist_checkpointing.save(
            _state(driver, rank_state=rank_state), str(path), sharded_strategy=_cpu_coordinated_strategy(group)
        )
    finally:
        dist.destroy_process_group(group)
    dist.barrier()
    if dist.get_rank() == 0:
        manifest = {
            "format": "cosmos_megatron_tp_pp",
            "format_version": FORMAT_VERSION,
            "mcore_checkpoint_version": MCORE_CHECKPOINT_VERSION,
            "topology": asdict(driver.topology),
            "world_size": dist.get_world_size(),
            "iteration": driver.iteration,
            "optimizer": True,
            "rank_state": True,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (path / ".save_complete").touch()
    dist.barrier()


def load(driver, path):
    from megatron.core import dist_checkpointing

    path = Path(path)
    if not (path / ".save_complete").is_file():
        raise ValueError("Incomplete TP/PP checkpoint")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported TP/PP checkpoint version")
    if manifest["topology"] != asdict(driver.topology) or manifest["world_size"] != dist.get_world_size():
        raise ValueError("Full-state restart currently requires the same topology")
    state = dist_checkpointing.load(_state(driver, rank_state=None, loading=True), str(path))
    driver.stage.load_state_dict(state["model"], strict=True)
    driver.optimizer.load_state_dict(state["optimizer"])
    # Replicated BF16 compute values are deduplicated in the model checkpoint,
    # but each TP rank retains its own FP32 optimizer master. Tiny differences
    # between replicas can cross a BF16 rounding boundary. Restore compute
    # from the actual loaded masters, as after an ordinary optimizer update;
    # otherwise restarting silently replaces rank1's compute with rank0's.
    # DistributedOptimizer writes DP-owned fragments into its parameter buffer;
    # DDP then gathers those fragments and copies them into model parameters.
    with torch.no_grad():
        for optimizer in getattr(driver.optimizer, "chained_optimizers", [driver.optimizer]):
            if hasattr(optimizer, "_copy_main_params_to_model_params"):
                optimizer._copy_main_params_to_model_params()
        driver.model.start_param_sync(force_sync=True)
    driver.iteration = state["iteration"]
    if driver.scheduler is not None:
        driver.scheduler.load_state_dict(state["scheduler"])
    elif state["scheduler"] is not None:
        raise ValueError("Checkpoint scheduler would be dropped")
    rank_state = state.pop("rank_state")
    del state
    # DCP planners and load closures can form cycles around GPU staging tensors.
    # Reclaim setup-only staging before entering the production-capacity schedule.
    gc.collect()
    dist.barrier()
    if dist.get_rank() == 0:
        release_file_cache(candidate for candidate in path.glob("*.distcp") if not candidate.is_symlink())
    dist.barrier()
    return rank_state
