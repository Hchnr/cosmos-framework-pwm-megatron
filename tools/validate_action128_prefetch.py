# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Audit the real Action128 recipe and hash production-packed training batches.

This runs data preparation only, never starts a model or writes training state.
Run legacy/shared separately with the same recipe, logical topology and seed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import random
import time
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import psutil
import torch
from omegaconf import OmegaConf

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.catalog import ParquetCatalogDataset
from cosmos_framework.utils.lazy_config import instantiate


def payload_digest(value) -> str:
    """Hash tensor storage without copying whole multi-GiB batches into bytes."""
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            if tensor.numel():
                digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B"))
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            if item.size:
                digest.update(memoryview(np.ascontiguousarray(item)).cast("B"))
        elif dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                digest.update(field.name.encode())
                update(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for key in sorted(item):
                if key == "_cosmos_loader" or key.startswith(("_worker_", "_sample_time", "_aug_", "_pre_aug")):
                    continue
                digest.update(key.encode())
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                update(child)
        elif item is None or isinstance(item, (str, bool, int, float)):
            digest.update(repr(item).encode())
        else:
            raise TypeError(f"Unsupported payload type in exact hash: {type(item)}")

    update(value)
    return digest.hexdigest()


def process_memory():
    root = psutil.Process()
    pss = 0
    for process in [root, *root.children(recursive=True)]:
        try:
            pss += process.memory_full_info().pss
        except psutil.NoSuchProcess:
            pass
    return pss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--backend", choices=["legacy_per_source", "shared_prefetch"], default="shared_prefetch")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--legacy-workers", type=int, default=0, help="Legacy physical worker budget; 0 runs synchronously"
    )
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--logical-workers", type=int, default=None)
    parser.add_argument("--episode-columns", choices=["projected", "all"], default="projected")
    parser.add_argument("--pin-memory-mode", choices=["background", "inline"], default="background")
    parser.add_argument("--benchmark", action="store_true", help="Time fetch without payload hashing or in-loop PSS")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--compute-delay", type=float, default=0.0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.benchmark and not 2 <= args.warmup < args.batches:
        parser.error("benchmark requires 2 <= warmup < batches")
    # Benchmark-only ablation: replace this module's parquet binding, not the
    # global pyarrow module. Only episode projection is disabled; frame reads,
    # production code and the underlying dataset remain unchanged.
    if args.episode_columns == "all":
        import cosmos_framework.data.generator.action.unified_action128 as action128

        parquet = action128.pq

        def read_table(*read_args, **kwargs):
            if kwargs.get("columns") == list(action128._EPISODE_COLUMNS):
                kwargs.pop("columns")
            return parquet.read_table(*read_args, **kwargs)

        action128.pq = types.SimpleNamespace(
            **{name: getattr(parquet, name) for name in dir(parquet) if name != "read_table"},
            read_table=read_table,
        )
    # Set HF_MODEL_PATH, WAN_VAE_PATH, BASE_CHECKPOINT_PATH explicitly as with
    # the launcher. Never collect the full environment or any credentials.
    config = load_experiment_from_toml(args.recipe, extra_overrides=["job.wandb_mode=disabled"])
    loader_config = config.dataloader_train
    fixed = config.model.config.fixed_sequence_length
    catalogs = [ParquetCatalogDataset(spec.path) for spec in loader_config.dataset_specs]
    audit = {
        "recipe": str(args.recipe.resolve()),
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "experiment": config.job.name,
        "action_gen": bool(config.model.config.action_gen),
        "tower_structure": config.model.config.tower_structure,
        "action_tower_preset": config.model.config.action_tower_preset,
        "max_action_dim": config.model.config.max_action_dim,
        "fixed": OmegaConf.to_container(fixed),
        "max_samples_per_batch": loader_config.max_samples_per_batch,
        "checkpoint_mode": config.checkpoint.action128_load_mode,
        "configured_workers": loader_config.num_workers,
        "configured_prefetch": loader_config.prefetch_factor,
        "datasets": [
            {
                "path": spec.path,
                "ratio": spec.ratio,
                "type": catalog[0]["type"],
                "adapter": catalog[0]["data"].get("adapter"),
                "fingerprint": catalog.fingerprint,
            }
            for spec, catalog in zip(loader_config.dataset_specs, catalogs, strict=True)
        ],
    }
    assert audit["action_gen"] and audit["max_action_dim"] == 128
    assert audit["tower_structure"] == "three_tower"
    assert all(entry["type"] == "action" and entry["adapter"] == "unified_action128" for entry in audit["datasets"])
    assert (fixed.und_tokens, fixed.gen_tokens, fixed.action_tokens) == (11776, 62592, 2944)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"audit": audit, "batches": []}
    import cosmos_framework

    result["implementation_root"] = str(Path(cosmos_framework.__file__).resolve().parent.parent)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.audit_only:
        return
    # The actual post-OOM production baseline has no workers, i.e. one logical
    # shard per source. Keep its topology when enabling the shared pool.
    supports_shared = importlib.util.find_spec("cosmos_framework.data.generator.dataflow.shared_prefetch") is not None
    physical_workers = args.legacy_workers if args.backend == "legacy_per_source" else args.workers
    logical_workers = [0] * len(catalogs)
    if args.backend == "legacy_per_source":
        base, extra = divmod(physical_workers, len(catalogs))
        logical_workers = [base + int(index < extra) for index in range(len(catalogs))]
    if args.logical_workers is not None:
        logical_workers = [args.logical_workers] * len(catalogs)
    if supports_shared:
        loader_config.execution_backend = args.backend
        loader_config.logical_workers = logical_workers
    elif args.backend != "legacy_per_source":
        raise ValueError("This checkout does not implement shared prefetch")
    loader_config.num_workers = physical_workers
    loader_config.prefetch_factor = args.prefetch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    result["execution"] = {
        "backend": args.backend,
        "physical_workers": loader_config.num_workers,
        "prefetch": args.prefetch,
        "logical_workers": logical_workers,
        "episode_columns": args.episode_columns,
        "pin_memory_mode": args.pin_memory_mode,
        "benchmark": args.benchmark,
        "warmup": args.warmup,
        "compute_delay": args.compute_delay,
    }
    init_start = time.perf_counter()
    loader = instantiate(loader_config)
    if hasattr(loader, "pin_memory_in_background"):
        loader.pin_memory_in_background = args.pin_memory_mode == "background"
    result["initialization_seconds"] = time.perf_counter() - init_start
    result["initialized_tree_pss_bytes"] = process_memory()
    delivery_start = time.perf_counter()
    iterator = iter(loader)
    measured_start = None
    try:
        for index in range(args.batches):
            if args.benchmark and index == args.warmup:
                measured_start = time.perf_counter()
            start = time.perf_counter()
            batch = next(iterator)
            fetch_seconds = time.perf_counter() - start
            actions = batch["action"]
            assert actions and all(sample[0].shape[-1] == 128 for sample in actions)
            assert all(plan.has_action and plan.has_vision for plan in batch["sequence_plan"])
            row = {
                "global_id": index,
                "dataset": batch["dataset_name"],
                "samples": len(actions),
                "sample_keys": batch.get("__key__"),
                "action_shapes": [list(sample[0].shape) for sample in actions],
                "video_shapes": [list(sample[0].shape) for sample in batch["video"]],
                "text_lengths": [int(sample[0].numel()) for sample in batch["text_token_ids"]],
                "payload_sha256": payload_digest(batch) if not args.benchmark else None,
                "fetch_seconds": fetch_seconds,
                "tree_pss_bytes": process_memory() if not args.benchmark else None,
                "prefetch_stats": getattr(loader, "prefetch_stats", None),
            }
            if not hasattr(loader, "prefetch_stats"):
                row["legacy_outstanding_tasks"] = sum(
                    getattr(stream, "_tasks_outstanding", 0) for stream in loader._iterators
                )
            result["batches"].append(row)
            if not args.benchmark:
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"sample_keys", "video_shapes", "action_shapes"}
                    }
                ),
                flush=True,
            )
            del batch
            if args.compute_delay:
                time.sleep(args.compute_delay)
        if args.benchmark:
            measured_seconds = time.perf_counter() - measured_start
            delivery_seconds = time.perf_counter() - delivery_start
            result["final_tree_pss_bytes"] = process_memory()
            waits = [row["fetch_seconds"] for row in result["batches"][args.warmup :]]
            result["benchmark_summary"] = {
                "measured_batches": len(waits),
                "delivery_seconds_including_warmup": delivery_seconds,
                "measured_seconds": measured_seconds,
                "batches_per_second": len(waits) / measured_seconds,
                "samples_per_second": sum(row["samples"] for row in result["batches"][args.warmup :])
                / measured_seconds,
                "fetch_total_seconds": sum(waits),
                "fetch_mean_seconds": float(np.mean(waits)),
                "fetch_p95_seconds": float(np.percentile(waits, 95)),
                "sampled_peak_tree_pss_bytes": max(
                    result["final_tree_pss_bytes"],
                    result["initialized_tree_pss_bytes"],
                ),
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if hasattr(loader, "close"):
            loader.close()
        else:
            if args.benchmark:
                # Measurements are already persisted. Stop all speculative
                # legacy suffixes together instead of paying a separate join
                # timeout for each source; no measured work is interrupted.
                for stream in loader._iterators or []:
                    for worker in getattr(stream, "_workers", []):
                        if worker.is_alive():
                            worker.terminate()
            for stream in loader._iterators or []:
                shutdown = getattr(stream, "_shutdown_workers", None)
                if shutdown is not None:
                    shutdown()
                underlying = getattr(getattr(stream, "_dataset_fetcher", None), "dataset_iter", None)
                if underlying is not None:
                    underlying.close()
        for stream in loader._loaders:
            dataset = getattr(getattr(stream.dataset, "_distributor", None), "_dataset", None)
            decoder = getattr(dataset, "decoder", None)
            if decoder is not None:
                decoder.close()


if __name__ == "__main__":
    main()
