"""Full-depth real-batch integration smoke using the stock Cosmos trainer.

Uses the original production TOML and checkpoint, reduced token capacities for
four GPUs, fresh optimizer state and a deterministic replay of one decoded batch.
This is an integration check, not a production throughput or loader-resume run.
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


class StepMeasurement(Callback):
    def __init__(self):
        super().__init__()
        self.started = None
        self.seconds = []
        self.peak_bytes = 0
        self.losses = []
        self.initialization_peak_bytes = 0

    def on_train_start(self, model, iteration=0):
        from cosmos_framework.utils.misc import set_random_seed

        set_random_seed(seed=42, by_rank=True)
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.int64)
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        self.initialization_peak_bytes = peak.item()
        torch.cuda.reset_peak_memory_stats()

    def on_training_step_start(self, model, data_batch, iteration=0):
        if self.started is None:
            torch.cuda.synchronize()
            self.started = time.perf_counter()

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        value = torch.tensor(time.perf_counter() - self.started, device="cuda", dtype=torch.float64)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
        self.seconds.append(value.item())
        mean_loss = loss.detach().float().clone()
        torch.distributed.all_reduce(mean_loss)
        mean_loss.div_(torch.distributed.get_world_size())
        self.losses.append(mean_loss.item())
        self.started = None
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.int64)
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        self.peak_bytes = max(self.peak_bytes, peak.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sharding-strategy", default="optim_grads")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--backend", choices=("pytorch", "megatron"), default="megatron")
    parser.add_argument("--live-data", action="store_true")
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()
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
        "model.config.fixed_sequence_length.und_tokens=512",
        "model.config.fixed_sequence_length.gen_tokens=8192",
        "model.config.fixed_sequence_length.action_tokens=128",
        "model.config.action_tower_und_short_tokens=0",
        "model.config.max_num_tokens_after_packing=8704",
        "dataloader_train.max_sequence_length=8704",
        "dataloader_train.max_samples_per_batch=1",
        "trainer.seed=42",
        f"trainer.max_iter={args.steps}",
        "trainer.grad_accum_iter=3",
        "trainer.timeout_period=7200",
        "trainer.run_validation=false",
        "trainer.callbacks.compile_tokenizer.enabled=false",
        "optimizer.lr=1e-5",
        "scheduler.warm_up_steps=[0]",
        "scheduler.f_start=[1.0]",
        f"checkpoint.load_training_state={str(args.resume_training).lower()}",
        f"checkpoint.action128_load_mode={'auto' if args.resume_training else 'warm_start'}",
        "checkpoint.save_iter=1000000",
        "checkpoint.keep_last_n=1",
        "checkpoint.keep_every_n_steps=null",
    ]
    if args.live_data:
        overrides.extend(
            [
                "dataloader_train.num_workers=1",
                "dataloader_train.prefetch_factor=1",
                "dataloader_train.packing_pool_size=1",
                "dataloader_train.microbatch_timeout_sec=0",
            ]
        )
    config = load_experiment_from_toml(
        "examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml",
        extra_overrides=overrides,
    )
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    model = instantiate(config.model)
    if distributed.get_rank() == 0:
        print("FULL_DEPTH_MODEL_INITIALIZED", flush=True)
    measurement = None
    if args.benchmark_json:
        if args.steps <= args.warmup_steps or args.resume_training:
            raise ValueError("Benchmark requires fresh state and steps > warmup_steps")
        measurement = StepMeasurement()
        trainer.callbacks._callbacks.append(measurement)
        # Isolate training-step throughput; actual async saves are tested by the
        # default integration mode. This benchmark explicitly excludes save I/O.
        trainer.checkpointer.save = lambda *args, **kwargs: None
    faulthandler.dump_traceback_later(180, repeat=True)
    loader = instantiate(config.dataloader_train) if args.live_data else ReplayBatch(args.batch)
    trainer.train(model, loader, None)
    faulthandler.cancel_dump_traceback_later()
    # The stock trainer destroys the process group at completion.
    if int(os.environ["RANK"]) == 0:
        report = {
            "passed": True,
            "scope": "full 36-layer model, reduced token capacities, live data"
            if args.live_data
            else "full 36-layer model, one replayed real batch, reduced token capacities",
            "sharding_strategy": args.sharding_strategy,
            "gradient_accumulation": 3,
            "optimizer_steps": args.steps,
            "checkpoint": str(args.checkpoint),
            "token_capacities": [512, 8192, 128],
            "fresh_optimizer_state": not args.resume_training,
            "backend": args.backend,
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
                scope="full 36-layer stock trainer; reduced token capacities; live data"
                if args.live_data
                else "full 36-layer stock trainer with real-batch replay; reduced token capacities",
                step_seconds=measurement.seconds,
                mean_final_microbatch_losses=measurement.losses,
                seed_reset_after_initialization=42,
                warmup_steps=args.warmup_steps,
                median_seconds=statistics.median(measurement.seconds[args.warmup_steps :]),
                peak_allocated_bytes=measurement.peak_bytes,
                initialization_peak_allocated_bytes=measurement.initialization_peak_bytes,
                excludes=["checkpoint save I/O", "tokenizer compilation"]
                + ([] if args.live_data else ["live shared-prefetch data loading"]),
            )
            args.benchmark_json.parent.mkdir(parents=True, exist_ok=True)
            args.benchmark_json.write_text(json.dumps(report, indent=2) + "\n")
        output = (
            Path("artifacts/megatron") / f"full_depth_real_batch_{'resume' if args.resume_training else 'smoke'}.json"
        )
        if measurement is None:
            output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
