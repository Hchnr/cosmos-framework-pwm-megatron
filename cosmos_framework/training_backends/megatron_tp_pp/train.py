# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Update-level production driver for the planned dense Action128 TP/PP recipe."""

import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from cosmos_framework.utils import log, misc
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, set_rand_state_dict
from cosmos_framework.utils.lazy_config import LazyConfig, instantiate

from . import checkpoint
from .batch import logical_slots, native_sample_level_loss
from .bundle import CanonicalBundle
from .config import ModelParallelConfig
from .data_state import import_logical_state
from .driver import ModelParallelDriver
from .logical_loader import LogicalDataLoader
from .model import build_stage, optimize_stage
from .parameter_map import stage_parameter_manifest
from .planning import completed_sample_totals, preflight_update, prepare_slot
from .preparation import GenerationLoss, GenerationPreparation
from .rng import LogicalTrainingRNG
from .runtime import initialize
from .scheduler import NativeScheduler
from .source import CanonicalDCPReader
from .transport import PipelineMetadata, broadcast_batch


def _validate(config, topology):
    model, parallel = config.model.config, config.model.config.parallelism
    if (
        parallel.context_parallel_shard_degree != 1
        or parallel.cfg_parallel_shard_degree != 1
        or parallel.data_parallel_shard_degree not in (-1, 1)
        or parallel.data_parallel_replicate_degree != 1
    ):
        raise ValueError("TP/PP derives its DP group; native FSDP/HSDP/CP/CFG degrees are incompatible")
    if (
        model.tower_structure != "three_tower"
        or model.action_tower_preset != "width_only_1b"
        or model.load_vlm_visual
        or model.sound_gen
        or model.ema.enabled
        or model.lora_enabled
        or model.fp8_training.enabled
        or model.quantization.method is not None
        or model.predict_text_tokens
    ):
        raise ValueError("TP/PP requires the planned unquantized dense three-tower Nano action recipe")
    if (
        config.trainer.grad_accum_iter != 1
        or config.checkpoint.dcp_async_mode_enabled
        or not config.checkpoint.load_training_state
        or config.checkpoint.keys_to_skip_loading
        or config.checkpoint.keys_not_to_resume
        or not config.checkpoint.megatron_tp_pp_import_path
        or config.checkpoint.save_to_object_store.enabled
        or config.checkpoint.load_from_object_store.enabled
    ):
        raise ValueError("TP/PP requires one outer update, synchronous full-state loading and a canonical import path")
    clip_norm = float(config.trainer.callbacks.grad_clip.clip_norm)
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("TP/PP requires a finite positive grad_clip.clip_norm")
    dp = topology.data_parallel_size(dist.get_world_size())
    if topology.logical_data_parallel_size % dp:
        raise ValueError("Native logical streams currently require physical DP to divide logical DP")


def _latest(directory):
    candidates = []
    if directory.exists():
        for path in directory.glob("iter_*"):
            manifest = path / "manifest.json"
            if (path / ".save_complete").is_file():
                if not manifest.is_file() or json.loads(manifest.read_text()).get("format") != "cosmos_megatron_tp_pp":
                    raise ValueError(f"Refusing to mix TP/PP checkpoints with another job format: {path}")
                candidates.append(path)
    return max(candidates, key=lambda path: int(path.name.removeprefix("iter_"))) if candidates else None


def _save(driver, directory, *, loader, rng, source_state, model_config, checkpoint_config, logical_state=None):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    rank_state = {
        "version": 1,
        "physical_rng": get_rand_state_dict(),
        "model_parallel_rng": get_cuda_rng_tracker().get_states(),
        "logical_loader": logical_state["loader"]
        if logical_state is not None
        else (loader.logical_state_dict() if loader is not None else None),
        "logical_training_rng": logical_state["rng"]
        if logical_state is not None
        else (rng.state_dict() if rng is not None else None),
        "source_trainer": source_state,
        "input_adapter_warm_full_und": getattr(driver.stage.input_adapter, "_warm_full_und", None),
        "action128_contract": {
            "version": 1,
            "spec": OmegaConf.to_container(model_config.action128_spec, resolve=True),
            "vision_spatial_position_mode": model_config.diffusion_expert_config.vision_spatial_position_mode,
        },
    }
    target = directory / f"iter_{driver.iteration:09d}"
    checkpoint.save(driver, target, rank_state=rank_state)
    if dist.get_rank() == 0:
        temporary = directory / ".latest_checkpoint.tmp"
        temporary.write_text(target.name + "\n")
        temporary.replace(directory / "latest_checkpoint.txt")
        checkpoint.prune_completed(
            target,
            keep_last_n=checkpoint_config.keep_last_n,
            keep_every_n_steps=checkpoint_config.keep_every_n_steps,
        )
    dist.barrier()


def run(config, args=None, *, initialized_callback=None):
    from megatron.core import parallel_state as ps
    from megatron.core.optimizer import OptimizerConfig

    topology = ModelParallelConfig(**OmegaConf.to_container(config.model.config.parallelism.megatron, resolve=True))
    _validate(config, topology)
    config.freeze()
    misc.set_torch_compile_options(
        config.trainer.compile_config.recompile_limit, config.trainer.compile_config.use_duck_shape
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    groups = initialize(topology, seed=config.trainer.seed)
    owner = ps.is_pipeline_first_stage() and ps.get_tensor_model_parallel_rank() == 0
    model_group = ps.get_model_parallel_group()
    source_rank = min(dist.get_process_group_ranks(model_group))
    data_group = ps.get_data_parallel_group()
    dp_size, dp_rank = ps.get_data_parallel_world_size(), ps.get_data_parallel_rank()
    logical_ranks = tuple(range(dp_rank, topology.logical_data_parallel_size, dp_size))
    trace_prepared = os.environ.get("COSMOS_TP_PP_TRACE_PREPARED", "0") == "1"
    preflight_required = (
        native_sample_level_loss(config.model.config) or os.environ.get("COSMOS_TP_PP_VERIFY_RAW_REPLAY", "0") == "1"
    )
    capture_dir = os.environ.get("COSMOS_TP_PP_CAPTURE_PREPARED_DIR")
    capture_iteration_value = os.environ.get("COSMOS_TP_PP_CAPTURE_ITERATION")
    capture_iteration = int(capture_iteration_value) if capture_iteration_value is not None else None
    if capture_iteration is not None and capture_dir is None:
        raise ValueError("COSMOS_TP_PP_CAPTURE_ITERATION requires COSMOS_TP_PP_CAPTURE_PREPARED_DIR")
    root = Path(config.job.path_local)
    parallel_preparation_flag = os.environ.get("COSMOS_TP_PP_PARALLEL_PREPARATION", "0")
    if parallel_preparation_flag not in ("0", "1"):
        raise ValueError("COSMOS_TP_PP_PARALLEL_PREPARATION must be 0 or 1")
    parallel_preparation = None
    if parallel_preparation_flag == "1":
        from .parallel_preparation import ParallelPreparation

        parallel_preparation = ParallelPreparation(model_group, logical_ranks, root)
        if parallel_preparation.owner != owner:
            raise ValueError("Parallel preparation and pipeline data owner disagree")
    preparation_ranks = parallel_preparation.assigned if parallel_preparation is not None else logical_ranks
    save_directory = root / "checkpoints"
    latest = _latest(save_directory)
    if dist.get_rank() == 0:
        root.mkdir(parents=True, exist_ok=True)
        LazyConfig.save_yaml(config, str(root / "config.yaml"))
    dist.barrier()
    stage, network_config = build_stage(config.model.config, topology)
    bundle = CanonicalBundle(config.checkpoint.megatron_tp_pp_import_path, topology)
    if Path(config.checkpoint.load_path).resolve() != Path(bundle.manifest["source"]).resolve():
        raise ValueError("Canonical bundle source differs from checkpoint.load_path used for normalization continuity")
    saved_contract, changed = bundle.validate_action128_extension(
        {
            "version": 1,
            "spec": OmegaConf.to_container(config.model.config.action128_spec, resolve=True),
            "vision_spatial_position_mode": network_config.vision_spatial_position_mode,
        }
    )
    bundle.load_model(stage)
    optimize_stage(stage, config.model.config)
    optimizer_config = OptimizerConfig(
        lr=float(config.optimizer.lr),
        weight_decay=float(config.optimizer.weight_decay),
        adam_beta1=float(config.optimizer.betas[0]),
        adam_beta2=float(config.optimizer.betas[1]),
        adam_eps=float(config.optimizer.eps),
        clip_grad=float(config.trainer.callbacks.grad_clip.clip_norm),
        use_distributed_optimizer=topology.use_distributed_optimizer,
        bf16=stage.config.params_dtype == torch.bfloat16,
    )
    overrides = bundle.optimizer_overrides(stage, optimizer_config)
    from .native_fp32_norm import agree_requested

    norm_flag = os.environ.get("COSMOS_TP_PP_NATIVE_FP32_NORM", "0")
    norm_enabled = agree_requested(
        norm_flag, dtype=stage.config.params_dtype, dp_size=dp_size, optimizer_config=optimizer_config, device="cuda"
    )
    native_norm_order = None
    if norm_enabled:
        from .native_fp32_norm import derive_native_parameter_order

        derivation_error = None
        try:
            native_norm_order = derive_native_parameter_order(config.model.config, bundle.manifest)
        except Exception as error:
            derivation_error = error
        failed = torch.tensor([int(derivation_error is not None)], dtype=torch.int32, device="cuda")
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError("Native FP32 norm metadata derivation failed on at least one rank") from derivation_error
    driver = ModelParallelDriver(
        stage,
        topology,
        optimizer_config,
        config_overrides=overrides,
        gc_trace_root=root,
        native_fp32_norm_order=native_norm_order,
        _native_norm_consensus_checked=True,
    )
    bundle.load_optimizer(stage, driver.optimizer)
    dist.barrier()
    if dist.get_rank() == 0:
        bundle.release_import_cache()
    dist.barrier()
    scheduler_values = OmegaConf.to_container(config.scheduler, resolve=True)
    scheduler_type = scheduler_values.pop("lr_scheduler_type")
    scheduler_values.pop("_target_")
    scheduler_values.pop("optimizer", None)
    driver.scheduler = NativeScheduler(driver.optimizer, scheduler_type, scheduler_values)
    driver.iteration = driver.scheduler.import_source(CanonicalDCPReader(bundle.path / "logical_state" / "scheduler"))
    source_iteration = CanonicalDCPReader(bundle.path / "logical_state" / "trainer").read("iteration")
    if driver.iteration != source_iteration:
        raise ValueError("Source scheduler and trainer disagree on iteration")
    log.info(f"TP/PP import complete: allocated CUDA bytes={torch.cuda.memory_allocated()}", rank0_only=False)
    restarted = checkpoint.load(driver, latest) if latest is not None else None
    if latest is not None:
        log.info(f"TP/PP restart loaded: allocated CUDA bytes={torch.cuda.memory_allocated()}", rank0_only=False)
    if any(scheduler.last_epoch != driver.iteration for scheduler in driver.scheduler.schedulers):
        raise ValueError("Restored scheduler and trainer disagree on iteration")
    if restarted is not None and restarted.get("version") != 1:
        raise ValueError("Unknown logical training checkpoint version")
    if restarted is not None and driver.stage.input_adapter is not None:
        # Legacy runtime checkpoints were only saved after successful updates,
        # so their first-forward UND warmup had already completed. Preserve the
        # actual flag in new checkpoints rather than repeating it on restart.
        warm_full_und = restarted.get("input_adapter_warm_full_und", False)
        if not isinstance(warm_full_und, bool):
            raise ValueError("Invalid restored first-stage UND warmup state")
        driver.stage.input_adapter._warm_full_und = warm_full_und
    source_state, loader, rng, preparation, vae_compiler = {}, None, None, None, None
    if owner or parallel_preparation is not None:
        log.info("TP/PP owner: constructing native dataset catalog", rank0_only=False)
        template = instantiate(
            config.dataloader_train, parallel_dims=SimpleNamespace(dp_coord=(0, topology.logical_data_parallel_size))
        )
        log.info(
            f"TP/PP preparation worker: importing {len(preparation_ranks)} logical loader/RNG states", rank0_only=False
        )
        if restarted is None:
            states, rng, source_state = import_logical_state(
                template, bundle, topology, preparation_ranks, saved_contract, changed
            )
        else:
            if parallel_preparation is not None:
                partition = parallel_preparation.restart_partition(restarted)
                states, rng = partition["loader"], LogicalTrainingRNG(partition["rng"])
            else:
                states = restarted["logical_loader"]
                rng = LogicalTrainingRNG(restarted["logical_training_rng"])
            source_state = restarted["source_trainer"]
            if tuple(sorted(states)) != preparation_ranks:
                raise ValueError("Restart changed logical DP ownership")
        loader = LogicalDataLoader(template, states)
        log.info("TP/PP owner: constructing frozen tokenizer/VAE", rank0_only=False)
        preparation_group = data_group
        preparation_adapter = stage.input_adapter
        if parallel_preparation is not None:
            from torch import nn

            from cosmos_framework.model.generator.mot.action128 import Action128Normalization

            spec = broadcast_batch(stage.input_adapter.action_normalization.spec if owner else None, model_group)
            preparation_adapter = nn.Module()
            preparation_adapter.action_normalization = (
                stage.input_adapter.action_normalization
                if owner
                else Action128Normalization(spec, network_config.vision_spatial_position_mode).cuda()
            )
            preparation_group = dist.group.WORLD
            parallel_preparation.loader, parallel_preparation.rng = loader, rng
        preparation = GenerationPreparation(config.model.config, preparation_adapter, process_group=preparation_group)
        compile_node = config.trainer.callbacks.get("compile_tokenizer")
        if compile_node is not None:
            vae_compiler = instantiate(compile_node)
            vae_compiler.config = config
            vae_compiler.process_group = preparation_group
    loss = GenerationLoss(config.model.config) if ps.is_pipeline_last_stage() else None
    metadata = PipelineMetadata()
    manifest = stage_parameter_manifest(stage)
    initialized = {
        "backend": "megatron_tp_pp",
        "rank": dist.get_rank(),
        "groups": groups,
        "stage_layers": list(stage.layers.keys()),
        "parameters": sum(p.numel() for p in stage.parameters()),
        "canonical_tensors": len(manifest),
        "iteration": driver.iteration,
        "logical_slots": topology.global_packed_microbatches_per_update,
        "microbatches": driver.microbatches,
        "runtime_restart": str(latest) if latest else None,
        "scheduler_last_epochs": [scheduler.last_epoch for scheduler in driver.scheduler.schedulers],
        "learning_rates": [float(group["lr"]) for group in driver.optimizer.param_groups],
        "prepared_noise_trace": trace_prepared,
        "raw_data_preflight": preflight_required,
        "parallel_preparation": parallel_preparation is not None,
        "raw_data_fingerprints": preflight_required or trace_prepared or capture_dir is not None,
        "prepared_capture_directory": capture_dir,
        "prepared_capture_iteration": capture_iteration,
        "attention_deterministic": os.environ.get("COSMOS_ATTENTION_DETERMINISTIC", "0") == "1",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "profile_slots": os.environ.get("COSMOS_TP_PP_PROFILE_SLOTS"),
        "compute_precision": str(stage.config.params_dtype),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "optimizer_types": [
            type(opt.optimizer).__module__ + "." + type(opt.optimizer).__name__
            for opt in getattr(driver.optimizer, "chained_optimizers", [driver.optimizer])
        ],
        "logical_data_ranks": list(preparation_ranks) if loader is not None else [],
        "logical_loader_states": len(loader.native_states) if loader is not None else 0,
        "logical_training_rng_states": len(rng.states) if rng is not None else 0,
        "source_trainer": source_state,
    }
    if norm_enabled:
        initialized["native_fp32_norm_enabled"] = True
    (root / f"initialization.rank{dist.get_rank()}.json").write_text(json.dumps(initialized, indent=2) + "\n")
    log.info(json.dumps(initialized), rank0_only=False)
    if restarted is not None:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        set_rand_state_dict(restarted["physical_rng"])
        get_cuda_rng_tracker().set_states(restarted["model_parallel_rng"])
    try:
        if initialized_callback is not None:
            initialized_callback(driver)
        while driver.iteration < config.trainer.max_iter:
            started = time.monotonic()
            metadata_bytes_before = metadata.bytes_sent
            if (owner or parallel_preparation is not None) and vae_compiler is not None:
                vae_compiler.on_training_step_start(preparation, {}, iteration=driver.iteration)
            tokenizer_setup_seconds = time.monotonic() - started
            torch.cuda.reset_peak_memory_stats()
            slots = logical_slots(
                topology, update=driver.iteration, data_parallel_size=dp_size, data_parallel_rank=dp_rank
            )
            preflight_started = time.monotonic()
            preparation_report = {}
            if parallel_preparation is not None:
                records, totals, preparation_report = parallel_preparation.prepare(
                    slots,
                    topology,
                    preparation,
                    preflight=preflight_required,
                    trace_prepared=trace_prepared,
                    capture_dir=capture_dir,
                    capture_iteration=capture_iteration,
                )
                loader = parallel_preparation.loader
            elif preflight_required:
                loader, records, totals = preflight_update(
                    loader,
                    slots,
                    topology,
                    preparation,
                    owner=owner,
                    model_group=model_group,
                    source_rank=source_rank,
                    dp_group=data_group,
                )
            else:
                records, totals = [{} for _ in slots], None
            preflight_seconds = (
                preparation_report["local_preflight_seconds"]
                if parallel_preparation is not None
                else time.monotonic() - preflight_started
            )
            if owner and preflight_required:
                plan_path = root / f"replay_plan.iter{driver.iteration:09d}.rank{dist.get_rank()}.json"
                plan_path.write_text(json.dumps(records, indent=2) + "\n")

            def batches():
                for index, slot in enumerate(slots):
                    if ps.is_pipeline_first_stage():
                        batch = None
                        if owner:
                            if parallel_preparation is not None:
                                batch = parallel_preparation.spool.load(
                                    (slot.update, slot.micro_step, slot.logical_rank)
                                )
                            else:
                                batch = prepare_slot(
                                    loader,
                                    slot,
                                    records[index],
                                    totals,
                                    topology,
                                    preparation,
                                    rng,
                                    hash_raw=preflight_required,
                                    trace_prepared=trace_prepared,
                                    capture_dir=capture_dir
                                    if capture_iteration is None or capture_iteration == slot.update
                                    else None,
                                )
                        if owner and (index == 0 or (index + 1) % 16 == 0 or index + 1 == len(slots)):
                            log.info(
                                f"TP/PP update {driver.iteration}: prepared slot {index + 1}/{len(slots)}",
                                rank0_only=False,
                            )
                        batch = broadcast_batch(batch, ps.get_tensor_model_parallel_group())
                        batch["publish_metadata"] = metadata.send
                        yield batch
                    else:
                        yield metadata.receive(expected_slot=(slot.update, slot.micro_step, slot.logical_rank))

            forward_backward_started = time.monotonic()
            metrics = driver.forward_backward(batches(), loss, metadata=metadata)
            metadata.drain()
            forward_backward_host_seconds = time.monotonic() - forward_backward_started
            optimizer_started = time.monotonic()
            success, grad_norm, _ = driver.step()
            if not success:
                raise FloatingPointError("TP/PP update failed finiteness checks; logical data are not silently skipped")
            if any(scheduler.last_epoch != driver.iteration for scheduler in driver.scheduler.schedulers):
                raise ValueError("Scheduler must advance exactly once per optimizer update")
            torch.cuda.synchronize()
            optimizer_and_sync_host_seconds = time.monotonic() - optimizer_started
            if totals is None:
                totals = completed_sample_totals(
                    records,
                    topology,
                    owner=owner,
                    model_group=model_group,
                    source_rank=source_rank,
                    dp_group=data_group,
                )
            elif isinstance(totals, dict):
                totals = totals["samples"]
            report = {
                "iteration": driver.iteration,
                "rank": dist.get_rank(),
                "update_seconds": time.monotonic() - started,
                "denominator_preflight_seconds": preflight_seconds,
                "tokenizer_setup_seconds": tokenizer_setup_seconds,
                "raw_data_preflight": preflight_required,
                "raw_data_fingerprints": preflight_required
                or trace_prepared
                or (
                    capture_dir is not None and (capture_iteration is None or capture_iteration == driver.iteration - 1)
                ),
                "forward_backward_host_seconds": forward_backward_host_seconds,
                "optimizer_and_sync_host_seconds": optimizer_and_sync_host_seconds,
                "grad_norm": grad_norm,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "logical_slots": topology.global_packed_microbatches_per_update,
                "loss_reporting": "unscaled_microbatch_mean_v1",
                "parallel_preparation": parallel_preparation is not None,
                **preparation_report,
                "samples_by_original_microstep": totals,
                "metadata_bytes_sent": metadata.bytes_sent - metadata_bytes_before,
            }
            if owner:
                report["owner_host_seconds_are_worker_sums"] = parallel_preparation is not None
                if trace_prepared or not preflight_required:
                    plan_path = root / f"replay_plan.iter{driver.iteration - 1:09d}.rank{dist.get_rank()}.json"
                    plan_path.write_text(json.dumps(records, indent=2) + "\n")
                report["local_logical_samples"] = sum(record["samples"] for record in records)
                # Host wall times include stalls and asynchronous CUDA enqueue.
                # They are not exclusive GPU kernel durations; use Nsight for
                # communication/compute attribution and overlap.
                report["owner_host_seconds"] = {
                    name: sum(record.get("host_seconds", {}).get(name, 0.0) for record in records)
                    for name in (
                        "preflight_load",
                        "preflight_hash",
                        "replay_load",
                        "replay_hash",
                        "prepare_enqueue",
                        "acceptance_trace",
                        "acceptance_capture",
                        "prepared_spool_write",
                    )
                }
                report["local_logical_tokens"] = {
                    name: sum(record["tokens"][name] for record in records) for name in ("text", "vision", "action")
                }
            if ps.is_pipeline_last_stage():
                report["loss"] = (
                    sum(float(item["loss"]) for item in metrics) / len(metrics)
                    if metrics and "loss" in metrics[0]
                    else None
                )
            with (root / f"updates.rank{dist.get_rank()}.jsonl").open("a") as handle:
                handle.write(json.dumps(report) + "\n")
            log.info(json.dumps(report), rank0_only=False)
            if parallel_preparation is not None:
                parallel_preparation.release()
            if driver.iteration % config.checkpoint.save_iter == 0 or driver.iteration == config.trainer.max_iter:
                checkpoint_started = time.monotonic()
                logical_state = parallel_preparation.checkpoint_state() if parallel_preparation is not None else None
                _save(
                    driver,
                    save_directory,
                    loader=loader if parallel_preparation is None else None,
                    rng=rng if parallel_preparation is None else None,
                    logical_state=logical_state,
                    # Preserve the original owner-only source-trainer layout.
                    # Parallel preparation imports this common metadata on all
                    # ranks, but only the pipeline data owner checkpoints it.
                    source_state=source_state if owner else {},
                    model_config=config.model.config,
                    checkpoint_config=config.checkpoint,
                )
                with (root / f"checkpoint_timings.rank{dist.get_rank()}.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "iteration": driver.iteration,
                                "checkpoint_seconds": time.monotonic() - checkpoint_started,
                                "included_in_update_seconds": False,
                            }
                        )
                        + "\n"
                    )
    except torch.OutOfMemoryError as error:
        report = {
            "iteration": driver.iteration,
            "rank": dist.get_rank(),
            "error": str(error),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "logical_slots": topology.global_packed_microbatches_per_update,
            "compute_precision": str(stage.config.params_dtype),
            "memory_summary": torch.cuda.memory_summary(),
        }
        (root / f"oom.iter{driver.iteration:09d}.rank{dist.get_rank()}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        raise
    finally:
        if loader is not None:
            loader.close()
    ps.destroy_model_parallel()
    dist.destroy_process_group()
