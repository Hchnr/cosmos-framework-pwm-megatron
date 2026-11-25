# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synchronous complete MCore checkpoints with atomic completion publication.

Model parameters use canonical global FQNs and TP axes. Optimizer fragments use
MCore's sharded representation. Loader/data RNG remain explicitly same-topology.
"""

import gc
import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from .source import release_file_cache

FORMAT_VERSION = 1
MCORE_CHECKPOINT_VERSION = 3.1
_LEGACY_METADATA = object()


def _metadata_sha256(path):
    """Hash the saved DCP index without following a replacement symlink."""
    target = Path(path) / ".metadata"
    fd = os.open(target, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)

        def identity(value):
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

        if not stat.S_ISREG(before.st_mode) or before.st_size == 0 or identity(before) != identity(target.lstat()):
            raise ValueError("Invalid DCP metadata file")
        digest = hashlib.sha256()
        while True:
            data = os.read(fd, 1024 * 1024)
            if not data:
                break
            digest.update(data)
        if identity(before) != identity(os.fstat(fd)) or identity(before) != identity(target.lstat()):
            raise ValueError("DCP metadata changed during hashing")
        return digest.hexdigest()
    finally:
        os.close(fd)


def _require_metadata(path, expected):
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("Invalid checkpoint DCP metadata SHA256")
    if _metadata_sha256(path) != expected:
        raise ValueError("Checkpoint DCP metadata SHA256 mismatch")


def _coordinated_metadata_check(path, expected, *, device):
    error = None
    try:
        _require_metadata(path, expected)
    except Exception as exc:
        error = exc
    invalid = torch.tensor(int(error is not None), dtype=torch.int32, device=device)
    dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    if invalid.item():
        raise ValueError("At least one rank cannot verify checkpoint DCP metadata") from error


def _require_usable_driver(driver):
    if getattr(driver, "_checkpoint_restore_failed", False):
        raise RuntimeError("Checkpoint restore failed; discard and rebuild this driver")


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
    _require_usable_driver(driver)
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
    publication_error = None
    if dist.get_rank() == 0:
        try:
            # The DCP index has already been published and all writers have
            # joined the barrier above. Bind its bytes before completion.
            manifest = {
                "format": "cosmos_megatron_tp_pp",
                "format_version": FORMAT_VERSION,
                "mcore_checkpoint_version": MCORE_CHECKPOINT_VERSION,
                "topology": asdict(driver.topology),
                "world_size": dist.get_world_size(),
                "iteration": driver.iteration,
                "optimizer": True,
                "rank_state": True,
                "dcp_metadata_sha256": _metadata_sha256(path),
            }
            (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (path / ".save_complete").touch()
        except Exception as error:
            publication_error = error
    published = torch.tensor(int(publication_error is None), dtype=torch.int32, device="cuda")
    dist.broadcast(published, src=0)
    if not published.item():
        raise ValueError("Checkpoint manifest or DCP metadata publication failed") from publication_error
    dist.barrier()


def _checkpoint_range_present(metadata, key, start, end):
    """Return whether a one-dimensional DCP tensor covers a local shard."""
    saved = metadata.get(key)
    if saved is None:
        return False
    chunks = getattr(saved, "chunks", None)
    if chunks is None:
        raise ValueError(f"Expected tensor metadata for {key}")
    overlaps = []
    for chunk in chunks:
        if len(chunk.offsets) != 1 or len(chunk.sizes) != 1:
            raise ValueError(f"Expected a flat optimizer checkpoint tensor for {key}")
        left, right = int(chunk.offsets[0]), int(chunk.offsets[0] + chunk.sizes[0])
        if left < end and right > start:
            overlaps.append((max(left, start), min(right, end)))
    if not overlaps:
        return False
    covered = start
    for left, right in sorted(overlaps):
        if left > covered:
            raise ValueError(f"Incomplete checkpoint coverage for {key}")
        covered = max(covered, right)
    if covered != end:
        raise ValueError(f"Incomplete checkpoint coverage for {key}")
    return True


def _prepare_optimizer_for_load(
    optimizer, path, *, device, metadata_sha256=_LEGACY_METADATA, manifest_error=None, manifest_digest=None
):
    """Match fresh MCore optimizer templates to the actual DCP state.

    An identity lookup can populate the optimizer's defaultdict with empty
    dictionaries. MCore then omits moments, even if the checkpoint has them.
    Empty and partially trained checkpoints must retain their own templates.
    """
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from torch.distributed.checkpoint import FileSystemReader

    pending, local_error = [], None
    try:
        if manifest_error is not None:
            raise manifest_error
        if metadata_sha256 is not _LEGACY_METADATA:
            _require_metadata(path, metadata_sha256)
        metadata = FileSystemReader(path).read_metadata().state_dict_metadata
        if metadata_sha256 is not _LEGACY_METADATA:
            _require_metadata(path, metadata_sha256)
        wrappers = getattr(optimizer, "chained_optimizers", [optimizer])
        for index, wrapper in enumerate(wrappers):
            if getattr(wrapper, "is_stub_optimizer", False):
                continue
            if not isinstance(wrapper, DistributedOptimizer):
                # Preserve the previous loader behaviour for non-distributed
                # optimizers. Legacy metadata is not attested in that path.
                continue
            inner = wrapper.optimizer
            states = inner.state
            original_keys = set(states)
            try:
                shard_state = wrapper.sharded_param_state_dp_reshardable(
                    {}, is_loading=False, metadata={"checkpoint_version": MCORE_CHECKPOINT_VERSION}
                )
            finally:
                for param in set(states) - original_keys:
                    del states[param]
            prefix = f"chained_{index}." if len(wrappers) > 1 else ""
            shards = [value for value in nested_values(shard_state) if isinstance(value, ShardedTensor)]
            ranges = {
                (prefix + value.key, int(value.global_offset[0]), int(value.global_offset[0] + value.local_shape[0]))
                for value in shards
            }
            params = [param for group in inner.param_groups for param in group["params"]]
            param_shards = [value for value in shards if value.key.endswith(".param")]
            if params and not param_shards:
                raise ValueError("Distributed optimizer has parameters but no checkpoint parameter shards")
            if params and not wrapper.optimizer_state_keys:
                raise ValueError("Distributed optimizer has no checkpoint state keys")
            if not getattr(wrapper.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False):
                shard_ids = {id(value.data) for value in param_shards}
                if any(id(param) not in shard_ids for param in params):
                    raise ValueError("Distributed optimizer checkpoint omitted an owned parameter shard")
            elif len(param_shards) < len(params):
                raise ValueError("Distributed optimizer checkpoint omitted an owned parameter shard")
            comparisons = []
            for key, start, end in ranges:
                if not key.endswith(".param"):
                    continue
                if not _checkpoint_range_present(metadata, key, start, end):
                    raise ValueError(f"Missing optimizer parameter shard in checkpoint: {key}")
                stem = key.removesuffix(".param")
                flags = []
                for state_key in wrapper.optimizer_state_keys:
                    moment_key = f"{stem}.{state_key}"
                    saved = _checkpoint_range_present(metadata, moment_key, start, end)
                    flags.append(saved)
                    comparisons.append((saved, (moment_key, start, end) in ranges))
                if any(flags) and not all(flags):
                    raise ValueError(f"Incomplete checkpoint optimizer state for {key}")
            # Legacy checkpoints do not attest whether two missing moments
            # were intentionally absent or lost together. Permit them only
            # when every locally represented fragment has all saved moments.
            if metadata_sha256 is _LEGACY_METADATA and comparisons and not all(saved for saved, _ in comparisons):
                raise ValueError("Legacy checkpoint has ambiguous missing optimizer state; metadata is not attested")
            empty = all(isinstance(value, dict) and not value for value in states.values())
            mismatch = any(saved != current for saved, current in comparisons)
            if mismatch:
                if empty and comparisons and all(saved and not current for saved, current in comparisons):
                    pending.append(("clear", states, None))
                elif (
                    metadata_sha256 is not _LEGACY_METADATA
                    and not getattr(wrapper.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
                    and not isinstance(inner, (torch.optim.Adam, torch.optim.AdamW))
                ):
                    # Metadata bound at save time distinguishes a genuinely
                    # unused parameter from a corrupted legacy checkpoint.
                    # Prepare only owned FusedAdam moment shards; allocate
                    # before voting, and mutate no existing dict until all
                    # ranks agree. Unknown/partial in-memory schemas fail.
                    owned_ids = {id(param): param for param in params}
                    owned_shards = {}
                    for shard in param_shards:
                        if id(shard.data) in owned_ids:
                            if id(shard.data) in owned_shards:
                                raise ValueError("Duplicate optimizer parameter shard")
                            owned_shards[id(shard.data)] = shard
                    if set(owned_shards) != set(owned_ids) or any(id(param) not in owned_ids for param in states):
                        raise ValueError("Optimizer parameter ownership differs")
                    moment_keys = tuple(wrapper.optimizer_state_keys)
                    current_values = [states.get(param, {}) for param in params]
                    if any(bool(value) for value in current_values) and not all(
                        bool(value) for value in current_values
                    ):
                        raise ValueError("Partially initialized existing optimizer state is ambiguous")
                    changes = []
                    for param in params:
                        value = states.get(param, {})
                        if not isinstance(value, dict) or set(value) - set(moment_keys):
                            raise ValueError("Unsupported mixed optimizer state schema")
                        if value and set(value) != set(moment_keys):
                            raise ValueError("Incomplete existing optimizer state")
                        shard = owned_shards[id(param)]
                        start = int(shard.global_offset[0])
                        end = start + int(shard.local_shape[0])
                        stem = (prefix + shard.key).removesuffix(".param")
                        present = [
                            _checkpoint_range_present(metadata, f"{stem}.{key}", start, end) for key in moment_keys
                        ]
                        if any(present) and not all(present):
                            raise ValueError("Incomplete saved optimizer state")
                        required = all(present)
                        allocated = (
                            {
                                key: torch.empty_like(param, dtype=wrapper._get_state_key_dtype(key))
                                for key in moment_keys
                            }
                            if required and not value
                            else {}
                        )
                        changes.append((param, required, moment_keys, allocated))
                    pending.append(("mixed", states, changes))
                else:
                    raise ValueError("Checkpoint and existing optimizer state have incompatible fragments")
            elif not states and comparisons and not any(saved for saved, _ in comparisons):
                pending.append(("empty", states, params))
    except Exception as error:
        local_error = error

    invalid = torch.tensor(int(local_error is not None), dtype=torch.int32, device=device)
    dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    if invalid.item():
        raise ValueError("At least one rank cannot match its distributed optimizer checkpoint state") from local_error
    if manifest_digest is not None:
        local = torch.tensor(list(manifest_digest), dtype=torch.uint8, device=device)
        root = local.clone()
        dist.broadcast(root, src=0)
        different = torch.tensor(int(not torch.equal(local, root)), dtype=torch.int32, device=device)
        dist.all_reduce(different, op=dist.ReduceOp.MAX)
        if different.item():
            raise ValueError("Checkpoint manifest differs across ranks")
    # Do not change any optimizer until every rank has passed validation.
    for action, states, params in pending:
        if action == "clear":
            states.clear()
        elif action == "empty":
            for param in params:
                states.setdefault(param, {})
        else:
            for param, required, keys, allocated in params:
                state = states.setdefault(param, {})
                if required:
                    state.update(allocated)
                else:
                    for key in keys:
                        state.pop(key, None)


def load(driver, path):
    _require_usable_driver(driver)
    from megatron.core import dist_checkpointing

    path = Path(path)
    manifest, manifest_error, manifest_digest = None, None, None
    try:
        if not (path / ".save_complete").is_file():
            raise ValueError("Incomplete TP/PP checkpoint")
        raw = (path / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise ValueError("Invalid TP/PP checkpoint manifest")
        if manifest["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported TP/PP checkpoint version")
        if manifest["topology"] != asdict(driver.topology) or manifest["world_size"] != dist.get_world_size():
            raise ValueError("Full-state restart currently requires the same topology")
        manifest_digest = hashlib.sha256(raw).digest()
    except Exception as error:
        manifest_error = error
    metadata_sha256 = (
        manifest.get("dcp_metadata_sha256", _LEGACY_METADATA) if manifest_error is None else _LEGACY_METADATA
    )
    _prepare_optimizer_for_load(
        driver.optimizer,
        path,
        device="cuda",
        metadata_sha256=metadata_sha256,
        manifest_error=manifest_error,
        manifest_digest=manifest_digest,
    )
    # MCore loads directly into tensors supplied by the driver. If any later
    # check fails these tensors may already have changed; the driver must be
    # discarded, never reused or saved as though restoration had succeeded.
    driver._checkpoint_restore_failed = True
    state = dist_checkpointing.load(_state(driver, rank_state=None, loading=True), str(path))
    if metadata_sha256 is not _LEGACY_METADATA:
        _coordinated_metadata_check(path, metadata_sha256, device="cuda")
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
    driver._checkpoint_restore_failed = False
    return rank_state
