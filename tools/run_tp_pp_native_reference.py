# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native PyTorch FSDP reference with the same original logical 192 x 3 update.

Keeps the native model, optimizer, clipping and Trainer.training_step. The
bounded logical loader/noising helpers supply the topology-independent slots.
Independent unvirtualized native loaders remain the data-continuity oracle.
Optional snapshots contain numerical state only, not a resumable training job.
"""

import argparse
import gc
import json
import os
import time
import types
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("COSMOS_ATTENTION_DETERMINISTIC", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist

from cosmos_framework.callbacks import grad_clip
from cosmos_framework.callbacks.compile_tokenizer import CompileTokenizer
from cosmos_framework.callbacks.cosmos_dataloader_state import attach_cosmos_dataloader_state_callback
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.training_backends.megatron_tp_pp.batch import logical_slots
from cosmos_framework.training_backends.megatron_tp_pp.bundle import CanonicalBundle
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.data_state import import_logical_state
from cosmos_framework.training_backends.megatron_tp_pp.logical_loader import LogicalDataLoader
from cosmos_framework.training_backends.megatron_tp_pp.memory import MicrobatchGarbageCollector
from cosmos_framework.training_backends.megatron_tp_pp.planning import (
    completed_sample_totals,
    preflight_update,
    prepare_slot,
)
from cosmos_framework.training_backends.megatron_tp_pp.preparation import GenerationPreparation
from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader, release_file_cache
from cosmos_framework.utils import distributed, misc
from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils.nvtx import configure_nvtx


def prepared_step(self, batch, iteration):
    outputs = self.denoise(data_batch_packed=batch["packed_sequence"])
    weight = torch.as_tensor(batch["weight"], **self.tensor_kwargs_fp32) if "weight" in batch else None
    loss, metrics = self._compute_losses(
        out_net=outputs,
        data_batch_packed=batch["packed_sequence"],
        **batch["loss_inputs"],
        sample_level_scale_override=weight,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Native reference produced nonfinite loss")
    return {
        "batch_size": batch["sample_count"],
        "is_image_batch": batch["loss_inputs"]["is_image_batch"],
        **metrics,
    }, loss


def numerical_snapshot(model, optimizer, scheduler, root, iteration):
    import torch.distributed.checkpoint as dcp

    from cosmos_framework.checkpoint.dcp import ModelWrapper

    target = root / f"math_state_{iteration:09d}"
    if dist.get_rank() == 0:
        target.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    # Native FQN optimizer state includes true masters and nonzero moments.
    # Coordinate large object metadata on CPU, as in the TP/PP checkpoint
    # writer, so a full activation cache cannot exhaust NCCL CUDA allocations.
    torch.cuda.empty_cache()
    group = dist.new_group(backend="gloo")
    try:
        dcp.save(ModelWrapper(model).state_dict(), checkpoint_id=target / "model", process_group=group)
        dcp.save(optimizer.state_dict(), checkpoint_id=target / "optim", process_group=group)
        dcp.save(scheduler.state_dict(), checkpoint_id=target / "scheduler", process_group=group)
    finally:
        dist.destroy_process_group(group)
    dist.barrier()
    if dist.get_rank() == 0:
        (target / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "cosmos_native_reference_numerical_state",
                    "iteration": iteration,
                    "resumable_training_checkpoint": False,
                },
                indent=2,
            )
            + "\n"
        )
        (target / ".math_complete").touch()
    dist.barrier()


def main():
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-name", default="tp_pp_native_fsdp_reference")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--logical-dp", type=int, default=192)
    parser.add_argument("--save-numerical-state", action="store_true")
    parser.add_argument("--save-final-numerical-state", action="store_true")
    parser.add_argument("--disable-tokenizer-compile", action="store_true")
    parser.add_argument(
        "--microbatch-gc-interval",
        type=int,
        default=16,
        help="Collect unreachable Python cycles every N local microbatches; zero disables it",
    )
    args = parser.parse_args()
    if args.updates < 1 or args.logical_dp < 1:
        parser.error("Updates and logical DP must be positive")
    if args.microbatch_gc_interval < 0:
        parser.error("Microbatch GC interval must be nonnegative")
    os.environ.setdefault("OUTPUT_ROOT", str(args.output.resolve() / "stock_trainer"))
    os.environ.setdefault("IMAGINAIRE_OUTPUT_ROOT", os.environ["OUTPUT_ROOT"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init()
    world, rank = dist.get_world_size(), dist.get_rank()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    topology = ModelParallelConfig(
        global_packed_microbatches_per_update=args.logical_dp * 3, logical_data_parallel_size=args.logical_dp
    )
    if topology.global_packed_microbatches_per_update % world:
        raise ValueError("Native DP must divide the logical update")
    microbatches = topology.global_packed_microbatches_per_update // world
    profile = None
    profile_finished = False
    if os.environ.get("COSMOS_TP_PP_PROFILE_SLOTS"):
        profile = tuple(int(value) for value in os.environ["COSMOS_TP_PP_PROFILE_SLOTS"].split(":"))
        if len(profile) != 2 or not 0 <= profile[0] < profile[1] < microbatches:
            raise ValueError("COSMOS_TP_PP_PROFILE_SLOTS must be start:end within native local microbatches")
        configure_nvtx("module")
    source_iteration = CanonicalDCPReader(args.bundle / "logical_state/trainer").read("iteration")
    config = load_experiment_from_toml(
        str(args.toml),
        extra_overrides=[
            "model.config.parallelism.backend=pytorch",
            f"model.config.parallelism.data_parallel_shard_degree={world}",
            "model.config.parallelism.data_parallel_replicate_degree=1",
            f"trainer.grad_accum_iter={microbatches}",
            f"trainer.max_iter={source_iteration + args.updates}",
            f"job.name={args.job_name}",
            "job.wandb_mode=disabled",
            "trainer.cudnn.deterministic=true",
            "trainer.cudnn.benchmark=false",
            "checkpoint.dcp_async_mode_enabled=false",
            *(["trainer.callbacks.compile_tokenizer.enabled=false"] if args.disable_tokenizer_compile else []),
        ],
    )
    config.validate()
    config.freeze()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    misc.set_torch_compile_options(
        config.trainer.compile_config.recompile_limit, config.trainer.compile_config.use_duck_shape
    )
    trainer = config.trainer.type(config)
    model = instantiate(config.model)
    template = instantiate(config.dataloader_train, parallel_dims=SimpleNamespace(dp_coord=(0, args.logical_dp)))
    # Required by the original source checkpoint's strict append-only validation.
    # The actual reference stream below is the logical loader, not this template.
    attach_cosmos_dataloader_state_callback(trainer, template)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    trainer.callbacks.on_optimizer_init_start()
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    trainer.callbacks.on_optimizer_init_end()
    iteration = trainer.checkpointer.load(model, optimizer, scheduler, scaler)
    if iteration != source_iteration:
        raise ValueError("Native reference must start from the specified original source iteration")
    bundle = CanonicalBundle(args.bundle, topology)
    saved_contract, changed = bundle.validate_action128_extension(
        {
            "version": 1,
            "spec": OmegaConf.to_container(config.model.config.action128_spec, resolve=True),
            "vision_spatial_position_mode": config.model.config.diffusion_expert_config.vision_spatial_position_mode,
        }
    )
    logical_ranks = tuple(range(rank, args.logical_dp, world))
    states, rng, source_state = import_logical_state(template, bundle, topology, logical_ranks, saved_contract, changed)
    loader = LogicalDataLoader(template, states)
    preparation = SimpleNamespace(
        config=model.config,
        is_image_batch=model.is_image_batch,
        prepare=lambda raw, step: GenerationPreparation.prepare(model, raw, step),
    )
    model.training_step = types.MethodType(prepared_step, model)
    trainer.callbacks.on_train_start(model, iteration=iteration)
    compiler = next(
        (callback for callback in trainer.callbacks._callbacks if isinstance(callback, CompileTokenizer)), None
    )
    singletons = [dist.new_group([index]) for index in range(world)]
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        release_file_cache(
            path
            for component in ("model", "optim", "trainer")
            for path in (Path(config.checkpoint.load_path) / component).glob("*.distcp")
        )
    dist.barrier()
    last_norm = {}
    original_norm = grad_clip._total_norm_by_mesh

    def record_norm(*positional, **keywords):
        result = original_norm(*positional, **keywords)
        last_norm["value"] = result[0].detach().item()
        if not bool(torch.isfinite(result[0])):
            raise FloatingPointError("Native reference produced nonfinite gradients")
        return result

    grad_clip._total_norm_by_mesh = record_norm
    preflight_required = (
        config.model.config.rectified_flow_training_config.sample_level_loss_averaging
        or os.environ.get("COSMOS_TP_PP_VERIFY_RAW_REPLAY", "0") == "1"
    )
    trace_prepared = os.environ.get("COSMOS_TP_PP_TRACE_PREPARED", "0") == "1"
    try:
        for update_index in range(args.updates):
            started = time.monotonic()
            if compiler is not None:
                compiler.on_training_step_start(model, {}, iteration=iteration)
            tokenizer_setup_seconds = time.monotonic() - started
            slots = logical_slots(topology, update=iteration, data_parallel_size=world, data_parallel_rank=rank)
            preflight_started = time.monotonic()
            if preflight_required:
                loader, records, totals = preflight_update(
                    loader,
                    slots,
                    topology,
                    preparation,
                    owner=True,
                    model_group=singletons[rank],
                    source_rank=rank,
                    dp_group=dist.group.WORLD,
                )
            else:
                records, totals = [{} for _ in slots], None
            preflight_seconds = time.monotonic() - preflight_started
            loss_sum = torch.zeros((), dtype=torch.float64, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            accumulation = 0
            microbatch_gc = MicrobatchGarbageCollector(args.microbatch_gc_interval)
            for index, slot in enumerate(slots):
                profile_iteration = os.environ.get("COSMOS_TP_PP_PROFILE_ITERATION")
                if (
                    profile
                    and not profile_finished
                    and (profile_iteration is None or iteration == int(profile_iteration))
                ):
                    if index == profile[0]:
                        torch.cuda.cudart().cudaProfilerStart()
                    if index == profile[1]:
                        torch.cuda.cudart().cudaProfilerStop()
                        profile_finished = True
                batch = prepare_slot(
                    loader,
                    slot,
                    records[index],
                    totals,
                    topology,
                    preparation,
                    rng,
                    trace_prepared=trace_prepared,
                    hash_raw=preflight_required,
                )
                for callback in trainer.callbacks._callbacks:
                    if isinstance(callback, grad_clip.GradClip):
                        callback._state_key = "image" if batch["loss_inputs"]["is_image_batch"] else "video"
                _, loss, accumulation = trainer.training_step(
                    model, optimizer, scheduler, scaler, batch, iteration=iteration, grad_accum_iter=accumulation
                )
                if args.microbatch_gc_interval:
                    microbatch_gc.after_backward()
                loss_sum += loss.detach().double() / microbatches
                if index == 0 or (index + 1) % 16 == 0 or index + 1 == len(slots):
                    print(
                        json.dumps(
                            {
                                "rank": rank,
                                "iteration": iteration,
                                "native_slots": index + 1,
                                "microbatches": microbatches,
                            }
                        ),
                        flush=True,
                    )
            if accumulation or not torch.isfinite(loss_sum) or getattr(model, "_skip_optimizer_step", False):
                raise ValueError("Native reference update was incomplete or nonfinite")
            dist.all_reduce(loss_sum)
            loss_sum /= world
            torch.cuda.synchronize()
            if totals is None:
                totals = completed_sample_totals(
                    records,
                    topology,
                    owner=True,
                    model_group=singletons[rank],
                    source_rank=rank,
                    dp_group=dist.group.WORLD,
                )
            report = {
                "iteration": iteration + 1,
                "loss": loss_sum.item(),
                "gradient_norm": last_norm.get("value"),
                "update_seconds": time.monotonic() - started,
                "preflight_seconds": preflight_seconds,
                "raw_data_preflight": preflight_required,
                "raw_data_fingerprints": preflight_required or trace_prepared,
                "prepared_noise_trace": trace_prepared,
                "tokenizer_setup_seconds": tokenizer_setup_seconds,
                "samples_by_original_microstep": totals,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "logical_slots": topology.global_packed_microbatches_per_update,
                "physical_dp": world,
                "source_trainer": source_state,
                "scope": "native PyTorch FSDP, original logical data/noising, baseline compute/optimizer/Trainer step",
                "native_unvirtualized_data_oracle_still_required": True,
                "measurement_batch_callbacks_omitted": True,
                "microbatch_gc_interval": args.microbatch_gc_interval,
                "microbatch_gc_collections": microbatch_gc.collections,
            }
            (args.output / f"replay_plan.iter{iteration:09d}.rank{rank}.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            with (args.output / f"updates.rank{rank}.jsonl").open("a") as handle:
                handle.write(json.dumps(report) + "\n")
            print(json.dumps(report), flush=True)
            iteration += 1
            if args.save_numerical_state or (args.save_final_numerical_state and update_index == args.updates - 1):
                numerical_snapshot(model, optimizer, scheduler, args.output, iteration)
    finally:
        grad_clip._total_norm_by_mesh = original_norm
        loader.close()
        template.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
