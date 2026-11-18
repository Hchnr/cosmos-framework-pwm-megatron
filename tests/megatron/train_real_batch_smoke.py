"""Full-depth real-batch integration smoke using the stock Cosmos trainer.

Defaults to reduced capacities, fresh optimizer state and real-batch replay.
Opt-in flags retain production capacities/loading and restored training state.
Benchmark mode excludes checkpoint writes and tokenizer compilation.
"""

import argparse
import copy
import faulthandler
import importlib.metadata
import itertools
import json
import os
import statistics
import time
from pathlib import Path

import torch

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.lazy_config import instantiate


class ReplayBatch:
    def __init__(self, path):
        self.batch = torch.load(path, map_location="cpu", weights_only=False)

    def __iter__(self):
        return (copy.deepcopy(self.batch) for _ in itertools.count())


class ProfileUpdateRanges(Callback):
    """Mark whole updates, including loader wait and benchmark synchronization."""

    def __init__(self):
        super().__init__()
        self.active = None

    def on_before_dataloading(self, iteration=0):
        if self.active is None:
            self.active = torch.cuda.nvtx.range_start(f"attribution.update.{iteration + 1:06d}")

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if self.active is not None:
            torch.cuda.nvtx.range_end(self.active)
            self.active = None


class StepMeasurement(Callback):
    def __init__(self, *, reset_seed=True, expected_start_iteration=None):
        super().__init__()
        self.reset_seed = reset_seed
        self.expected_start_iteration = expected_start_iteration
        self.initial_iteration = None
        self.started = None
        self.seconds = []
        self.peak_bytes = 0
        self.losses = []
        self.initialization_peak_bytes = 0
        self.inclusive_started = None
        self.inclusive_seconds = []
        self.data_wait_seconds = []
        self.data_wait = 0.0

    def on_train_start(self, model, iteration=0):
        from cosmos_framework.utils.misc import set_random_seed

        self.initial_iteration = iteration
        if self.expected_start_iteration is not None and iteration != self.expected_start_iteration:
            raise ValueError(f"Restored iteration {iteration}, expected {self.expected_start_iteration}")
        if self.reset_seed:
            set_random_seed(seed=42, by_rank=True)
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.int64)
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        self.initialization_peak_bytes = peak.item()
        torch.cuda.reset_peak_memory_stats()

    def on_before_dataloading(self, iteration=0):
        if self.inclusive_started is None:
            torch.cuda.synchronize()
            self.inclusive_started = time.perf_counter()
        self.loading_started = time.perf_counter()

    def on_after_dataloading(self, iteration=0):
        self.data_wait += time.perf_counter() - self.loading_started

    def on_training_step_start(self, model, data_batch, iteration=0):
        if self.started is None:
            torch.cuda.synchronize()
            self.started = time.perf_counter()

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        value = torch.tensor(time.perf_counter() - self.started, device="cuda", dtype=torch.float64)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
        self.seconds.append(value.item())
        inclusive = torch.tensor(
            [time.perf_counter() - self.inclusive_started, self.data_wait], device="cuda", dtype=torch.float64
        )
        torch.distributed.all_reduce(inclusive, op=torch.distributed.ReduceOp.MAX)
        self.inclusive_seconds.append(inclusive[0].item())
        self.data_wait_seconds.append(inclusive[1].item())
        self.inclusive_started, self.data_wait = None, 0.0
        mean_loss = loss.detach().float().clone()
        torch.distributed.all_reduce(mean_loss)
        mean_loss.div_(torch.distributed.get_world_size())
        self.losses.append(mean_loss.item())
        self.started = None
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.int64)
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        self.peak_bytes = max(self.peak_bytes, peak.item())
        if torch.distributed.get_rank() == 0:
            print(
                json.dumps(
                    dict(
                        benchmark_update=iteration,
                        seconds=value.item(),
                        including_first_dataload_seconds=inclusive[0].item(),
                        data_wait_seconds=inclusive[1].item(),
                        loss=mean_loss.item(),
                        peak_allocated_bytes=self.peak_bytes,
                    )
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sharding-strategy", default="optim_grads")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--checkpoint-load-mode", choices=("auto", "warm_start", "extend"))
    parser.add_argument("--preserve-source-normalization", action="store_true")
    parser.add_argument(
        "--state-normalization-paths", type=Path, help="JSON mapping dataset root to immutable State128 snapshot"
    )
    parser.add_argument("--expected-start-iteration", type=int)
    parser.add_argument("--production-capacities", action="store_true")
    parser.add_argument("--production-loader", action="store_true")
    parser.add_argument("--backend", choices=("pytorch", "megatron"), default="megatron")
    parser.add_argument("--live-data", action="store_true")
    parser.add_argument("--trace-json", type=Path)
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profile-start-iteration", type=int)
    parser.add_argument("--profile-updates", type=int, default=2)
    parser.add_argument("--disable-param-prefetch", action="store_true")
    parser.add_argument("--profile-python-gc", action="store_true")
    parser.add_argument("--profile-allocator", action="store_true")
    parser.add_argument("--fsdp-copy-in-default-stream", action="store_true")
    parser.add_argument("--disable-auto-gc-after-warmup", action="store_true")
    parser.add_argument("--und-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=8192)
    parser.add_argument("--action-tokens", type=int, default=128)
    args = parser.parse_args()
    changed_copy_in_streams = []
    if args.fsdp_copy_in_default_stream:
        if args.backend != "pytorch":
            parser.error("--fsdp-copy-in-default-stream requires the PyTorch backend")
        from allocator_trace import use_default_copy_in_stream

        changed_copy_in_streams = use_default_copy_in_stream()
    if args.production_loader and not args.live_data:
        parser.error("--production-loader requires --live-data")
    load_mode = args.checkpoint_load_mode or ("auto" if args.resume_training else "warm_start")
    if load_mode == "extend" and not args.resume_training:
        parser.error("extend requires --resume-training")
    if load_mode == "extend" and not args.live_data:
        parser.error("extend requires --live-data to bind and restore the real catalog loader")
    if args.resume_training and load_mode == "warm_start":
        parser.error("warm_start cannot restore training state")
    if args.preserve_source_normalization and load_mode != "extend":
        parser.error("Preserving source normalization requires extend")
    if args.expected_start_iteration is not None and not args.benchmark_json:
        parser.error("--expected-start-iteration requires --benchmark-json")
    if (args.profile_python_gc or args.disable_auto_gc_after_warmup) and not args.benchmark_json:
        parser.error("GC observation/control requires --benchmark-json")
    if args.benchmark_json:
        first_iteration = args.expected_start_iteration or 0
        if args.warmup_steps < 0 or args.steps - first_iteration <= args.warmup_steps:
            parser.error("Benchmark requires at least one steady optimizer update")
    os.environ.setdefault(
        "ACTION128_EPISODE_CACHE_DIR", "/share-mixed/project/eai_pwm/cache/action128_episode_metadata_v1"
    )
    os.environ["BASE_CHECKPOINT_PATH"] = str(args.checkpoint.resolve())
    os.environ["HF_MODEL_PATH"] = "/share-mixed/project/eai_pwm/models/Cosmos3-Nano-VLM"
    os.environ["WAN_VAE_PATH"] = "/share-mixed/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth"
    os.environ["OUTPUT_ROOT"] = str(args.output_root.resolve())
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init()
    overrides = [
        "job.name=megatron_full_depth_real_batch_smoke",
        "job.wandb_mode=disabled",
        f"model.config.parallelism.backend={args.backend}",
        f"model.config.parallelism.megatron_sharding_strategy={args.sharding_strategy}",
        f"model.config.parallelism.data_parallel_shard_degree={distributed.get_world_size()}",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        f"model.config.fixed_sequence_length.und_tokens={args.und_tokens}",
        f"model.config.fixed_sequence_length.gen_tokens={args.gen_tokens}",
        f"model.config.fixed_sequence_length.action_tokens={args.action_tokens}",
        "model.config.action_tower_und_short_tokens=0",
        f"model.config.max_num_tokens_after_packing={args.und_tokens + args.gen_tokens}",
        f"dataloader_train.max_sequence_length={args.und_tokens + args.gen_tokens}",
        "dataloader_train.max_samples_per_batch=1",
        "trainer.seed=42",
        f"trainer.max_iter={args.steps}",
        "trainer.grad_accum_iter=3",
        "trainer.timeout_period=7200",
        "trainer.run_validation=false",
        "trainer.callbacks.compile_tokenizer.enabled=false",
        f"checkpoint.load_training_state={str(args.resume_training).lower()}",
        f"checkpoint.action128_load_mode={load_mode}",
        "checkpoint.save_iter=1000000",
        "checkpoint.keep_last_n=1",
        "checkpoint.keep_every_n_steps=null",
    ]
    if not args.resume_training:
        overrides.extend(["optimizer.lr=1e-5", "scheduler.warm_up_steps=[0]", "scheduler.f_start=[1.0]"])
    if args.production_capacities:
        prefixes = (
            "model.config.fixed_sequence_length.",
            "model.config.action_tower_und_short_tokens=",
            "model.config.max_num_tokens_after_packing=",
            "dataloader_train.max_sequence_length=",
        )
        overrides = [item for item in overrides if not item.startswith(prefixes)]
    if args.production_loader:
        overrides.remove("dataloader_train.max_samples_per_batch=1")
    if args.live_data and not args.production_loader:
        overrides.extend(
            [
                "dataloader_train.num_workers=1",
                "dataloader_train.prefetch_factor=1",
                "dataloader_train.packing_pool_size=1",
                "dataloader_train.microbatch_timeout_sec=0",
            ]
        )
    if args.preserve_source_normalization:
        overrides.append("checkpoint.action128_preserve_source_normalization=true")
    if args.profile_start_iteration is not None:
        if args.profile_updates < 1 or args.profile_start_iteration + args.profile_updates > args.steps:
            parser.error("Profile window must contain positive complete updates within --steps")
        overrides.extend(
            [
                "trainer.profiling.enable_nsys=true",
                f"trainer.profiling.profile_freq={args.profile_start_iteration + args.profile_updates}",
                "trainer.profiling.profile_warmup=0",
                f"trainer.profiling.profile_active={args.profile_updates}",
                "trainer.profiling.target_ranks=" + str(list(range(distributed.get_world_size()))),
            ]
        )
    if args.state_normalization_paths:
        paths = json.loads(args.state_normalization_paths.read_text())
        overrides.append("dataloader_train.action_state_normalization_paths=" + json.dumps(paths))
    config = load_experiment_from_toml(
        "examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml",
        extra_overrides=overrides,
    )
    config.validate()
    config.freeze()
    fixed = config.model.config.fixed_sequence_length
    capacities = [int(fixed.und_tokens), int(fixed.gen_tokens), int(fixed.action_tokens)]
    capacity_scope = (
        "original production token capacities" if args.production_capacities else "reduced token capacities"
    )
    trainer = config.trainer.type(config)
    if args.profile_start_iteration is not None:
        from cosmos_framework.utils.callback import NVTXCallback
        from cosmos_framework.utils.nvtx import NVTXGranularity

        nvtx_callbacks = [c for c in trainer.callbacks._callbacks if isinstance(c, NVTXCallback)]
        if nvtx_callbacks:
            for callback in nvtx_callbacks:
                callback.granularity = NVTXGranularity.MODULE
        else:
            callback = NVTXCallback(granularity="module")
            callback.config = config
            callback.trainer = trainer
            trainer.callbacks._callbacks.append(callback)
    model = instantiate(config.model)
    if args.disable_param_prefetch:
        if args.backend != "megatron":
            parser.error("--disable-param-prefetch requires the Megatron backend")
        model.net._cosmos_megatron_options["param_prefetch"] = False
    if distributed.get_rank() == 0:
        print("FULL_DEPTH_MODEL_INITIALIZED", flush=True)
    measurement = None
    if args.benchmark_json:
        if args.steps <= args.warmup_steps:
            raise ValueError("Benchmark requires steps > warmup_steps")
        measurement = StepMeasurement(
            reset_seed=not args.resume_training, expected_start_iteration=args.expected_start_iteration
        )
        trainer.callbacks._callbacks.append(measurement)
        # Isolate training-step throughput; actual async saves are tested by the
        # default integration mode. This benchmark explicitly excludes save I/O.
        trainer.checkpointer.save = lambda *args, **kwargs: None
    if args.profile_start_iteration is not None:
        trainer.callbacks._callbacks.append(ProfileUpdateRanges())
    if args.trace_json:
        from live_loader_trace import LiveTrace

        trainer.callbacks._callbacks.append(LiveTrace(args.trace_json))
    if args.profile_python_gc or args.disable_auto_gc_after_warmup:
        from python_gc_trace import PythonGCTrace

        disable_iteration = None
        if args.disable_auto_gc_after_warmup:
            if args.expected_start_iteration is None or args.warmup_steps < 1:
                parser.error("GC control requires explicit restored iteration and positive warmup")
            disable_iteration = args.expected_start_iteration + args.warmup_steps
        trainer.callbacks._callbacks.append(
            PythonGCTrace(args.benchmark_json.parent / "python_gc.json", disable_iteration)
        )
    if args.profile_allocator:
        from allocator_trace import AllocatorTrace

        if args.benchmark_json is None or args.expected_start_iteration is None:
            parser.error("Allocator observation requires benchmark output and explicit restored iteration")
        trainer.callbacks._callbacks.append(
            AllocatorTrace(
                args.benchmark_json.parent / "allocator.json", args.expected_start_iteration + args.warmup_steps
            )
        )
    faulthandler.dump_traceback_later(180, repeat=True)
    loader = instantiate(config.dataloader_train) if args.live_data else ReplayBatch(args.batch)
    if args.live_data:
        from cosmos_framework.callbacks.cosmos_dataloader_state import attach_cosmos_dataloader_state_callback

        attach_cosmos_dataloader_state_callback(trainer, loader)
    trainer.train(model, loader, None)
    faulthandler.cancel_dump_traceback_later()
    # The stock trainer destroys the process group at completion.
    if int(os.environ["RANK"]) == 0:
        from cosmos_framework.training_backends.megatron import get_runtime

        runtime = get_runtime(model)
        suppressed_prefetch = runtime.suppressed_prefetch_requests if runtime is not None else 0
        if args.disable_param_prefetch and suppressed_prefetch == 0:
            raise RuntimeError("Prefetch ablation did not suppress any actual prefetch requests")
        if args.fsdp_copy_in_default_stream and not changed_copy_in_streams:
            raise RuntimeError("Copy-in stream diagnostic did not modify any initialized FSDP context")
        if measurement is not None and len(measurement.seconds) <= args.warmup_steps:
            raise ValueError("No steady updates after the restored starting iteration")
        report = {
            "passed": True,
            "scope": f"full 36-layer model, {capacity_scope}, live data"
            if args.live_data
            else f"full 36-layer model, one replayed real batch, {capacity_scope}",
            "sharding_strategy": args.sharding_strategy,
            "param_prefetch": not args.disable_param_prefetch if args.backend == "megatron" else None,
            "suppressed_prefetch_requests": suppressed_prefetch,
            "effective_overlap_param_gather": bool(runtime.fsdp.ddp_config.overlap_param_gather)
            if runtime is not None
            else None,
            "gradient_accumulation": 3,
            "optimizer_steps": len(measurement.seconds) if measurement is not None else args.steps,
            "checkpoint": str(args.checkpoint),
            "token_capacities": capacities,
            "short_und_capacity": int(config.model.config.action_tower_und_short_tokens),
            "checkpoint_load_mode": load_mode,
            "preserve_source_normalization": args.preserve_source_normalization,
            "state_normalization_paths": dict(config.dataloader_train.action_state_normalization_paths),
            "dataloader_max_samples_per_batch": int(config.dataloader_train.max_samples_per_batch),
            "dataloader_packing_pool_size": int(config.dataloader_train.packing_pool_size),
            "fresh_optimizer_state": not args.resume_training,
            "backend": args.backend,
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "allocator_observation": args.profile_allocator,
            "copy_in_stream_intervention": changed_copy_in_streams,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "megatron_core_version": importlib.metadata.version("megatron-core"),
            "gpu": torch.cuda.get_device_name(),
            "world_size": int(os.environ["WORLD_SIZE"]),
            "live_data": args.live_data,
            "output_root": str(args.output_root),
        }
        if measurement is not None:
            report.update(
                scope=f"full 36-layer stock trainer; {capacity_scope}; live data"
                if args.live_data
                else f"full 36-layer stock trainer with real-batch replay; {capacity_scope}",
                step_seconds=measurement.seconds,
                update_including_first_dataload_seconds=measurement.inclusive_seconds,
                update_data_wait_seconds=measurement.data_wait_seconds,
                content_trace_enabled=args.trace_json is not None,
                mean_final_microbatch_losses=measurement.losses,
                seed_reset_after_initialization=42 if measurement.reset_seed else None,
                initial_iteration=measurement.initial_iteration,
                requested_final_iteration=args.steps,
                warmup_steps=args.warmup_steps,
                median_seconds=statistics.median(measurement.seconds[args.warmup_steps :]),
                peak_allocated_bytes=measurement.peak_bytes,
                initialization_peak_allocated_bytes=measurement.initialization_peak_bytes,
                excludes=["checkpoint save I/O", "tokenizer compilation"]
                + ([] if args.live_data else ["live shared-prefetch data loading"]),
            )
            args.benchmark_json.parent.mkdir(parents=True, exist_ok=True)
            args.benchmark_json.write_text(json.dumps(report, indent=2) + "\n")
        report_mode = "live_data" if args.live_data else "real_batch"
        output = (
            Path("artifacts/megatron")
            / f"full_depth_{report_mode}_{'resume' if args.resume_training else 'smoke'}.json"
        )
        if measurement is None:
            output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
