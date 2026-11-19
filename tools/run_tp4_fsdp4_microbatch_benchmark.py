"""Compare TP4 and native FSDP4 with equal batches per rank/microstep.

Replays immutable real prepared payloads through the production training loops.
Media loading, VAE/noising and their initialization are excluded. Global batch
sizes deliberately differ by four. This is a training-compute benchmark, not
an end-to-end data-pipeline benchmark or a numerical equivalence acceptance.
"""

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path


def payload_digest(header, buffers):
    import torch

    digest = hashlib.sha256(header)
    for dtype, tensor in sorted(buffers.items()):
        digest.update(dtype.encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().data)
    return digest.hexdigest()


class PreparedMicrosteps:
    def __init__(self, directory, microsteps):
        import torch

        from cosmos_framework.training_backends.megatron_tp_pp.transport import decode

        self.payloads = []
        self.description = []
        for micro in range(microsteps):
            path = directory / f"update000001200.micro{micro:03d}.rank000.pt"
            header, buffers = torch.load(path, map_location="cpu", weights_only=False)
            batch = decode(header, buffers)
            packed = batch["packed_sequence"]
            self.payloads.append((header, buffers))
            self.description.append(
                {
                    "microstep": micro,
                    "source": str(path.resolve()),
                    "payload_sha256": payload_digest(header, buffers),
                    "samples": batch["sample_count"],
                    "image": batch["loss_inputs"]["is_image_batch"],
                    "tokens": {
                        "text": packed.text_indexes.numel(),
                        "vision": packed.vision.sequence_indexes.numel() if packed.vision is not None else 0,
                        "action": packed.action.sequence_indexes.numel() if packed.action is not None else 0,
                    },
                }
            )

    def prepare(self, loader, slot, record, totals, topology, preparation, rng, **kwargs):
        from cosmos_framework.training_backends.megatron_tp_pp.transport import decode

        if totals is not None or preparation.config.rectified_flow_training_config.sample_level_loss_averaging:
            raise ValueError("Fixed microstep benchmark requires native rank-level loss averaging")
        started = time.monotonic()
        header, buffers = self.payloads[slot.micro_step]
        batch = decode(header, {dtype: tensor.cuda() for dtype, tensor in buffers.items()})
        batch["slot_id"] = (slot.update, slot.micro_step, slot.logical_rank)
        batch.pop("weight", None)
        description = self.description[slot.micro_step]
        record.update(
            slot=batch["slot_id"],
            source="immutable_real_prepared_microbatch",
            samples=description["samples"],
            image=description["image"],
            tokens=description["tokens"],
            sha256=None,
            prepared_payload_sha256=description["payload_sha256"],
            host_seconds={"prepared_replay_h2d": time.monotonic() - started},
        )
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("tp", "fsdp"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prepared-spool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=7)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument(
        "--contract-only", action="store_true", help="Validate immutable input batches without training"
    )
    args = parser.parse_args()
    if not 0 < args.warmup_updates < args.updates or int(os.environ["WORLD_SIZE"]) != 4:
        parser.error("Require four GPUs and positive warmup/measured windows")
    for key in ("COSMOS_TP_PP_VERIFY_RAW_REPLAY", "COSMOS_TP_PP_PARALLEL_PREPARATION"):
        if os.environ.get(key, "0") != "0":
            raise ValueError(f"Remove {key}: the benchmark replays prepared inputs")
    os.environ["COSMOS_TP_PP_TRACE_PREPARED"] = "0"
    os.environ["MEGATRON_TP_PP_IMPORT_PATH"] = str(args.bundle.resolve())
    os.environ["OUTPUT_ROOT"] = str(args.output.resolve() / "stock_trainer")
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = os.environ["OUTPUT_ROOT"]
    import torch

    from cosmos_framework.configs.toml_config import sft_config
    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    fixture = PreparedMicrosteps(args.prepared_spool, 3)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": __doc__,
        "backend": args.backend,
        "tp": 4 if args.backend == "tp" else 1,
        "pp": 1,
        "dp": 1 if args.backend == "tp" else 4,
        "precision": "bfloat16",
        "microsteps_per_update": 3,
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.updates - args.warmup_updates,
        "per_rank_microbatches": fixture.description,
        "global_sample_count_matched": False,
        "numerical_acceptance": False,
        "checkpoint_writes": False,
    }
    with (args.output / f"contract.rank{os.environ['RANK']}.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if args.contract_only:
        return
    original_load = sft_config.load_experiment_from_toml

    def load_config(*positional, **keywords):
        overrides = list(keywords.pop("extra_overrides", ()))
        overrides += [
            "model.config.load_vision_tokenizer=false",
            "trainer.callbacks.compile_tokenizer.enabled=false",
            "model.config.precision=bfloat16",
            "trainer.seed=42",
        ]
        return original_load(*positional, extra_overrides=overrides, **keywords)

    sft_config.load_experiment_from_toml = load_config
    # Warm the actual MCore norm groups before activation/cache pressure.
    from tools.run_tp_pp_with_optimizer_collective_warmup import install_collective_warmup

    candidate = os.environ.get("COSMOS_TP_PP_COLLECTIVES_CANDIDATE")
    if args.backend == "tp":
        if not candidate:
            raise ValueError("Select the qualified isolated collective warmup candidate")
        import importlib.util

        spec = importlib.util.spec_from_file_location("_benchmark_collectives", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(ModelParallelDriver.__init__, "_cosmos_optimizer_collective_warmup"):
            install_collective_warmup(module.warm_optimizer_collectives)
        from cosmos_framework.training_backends.megatron_tp_pp import train

        train.prepare_slot = fixture.prepare
        # Both benchmark branches discard state. Avoid a final unmeasured
        # full-state write delaying the next GPU job; updates still run intact.
        train._save = lambda *positional, **keywords: None
        os.environ["OUTPUT_ROOT"] = str(args.output / "training")
        os.environ["IMAGINAIRE_OUTPUT_ROOT"] = os.environ["OUTPUT_ROOT"]
        os.environ["MEGATRON_TP_PP_IMPORT_PATH"] = str(args.bundle.resolve())
        sys.argv = [
            "train",
            "--sft-toml",
            str(args.toml),
            "--",
            "job.name=tp4_microbatch",
            f"trainer.max_iter={1200 + args.updates}",
            "checkpoint.save_iter=100000",
            "model.config.parallelism.megatron.tensor_model_parallel_size=4",
            "model.config.parallelism.megatron.pipeline_model_parallel_size=1",
            "model.config.parallelism.megatron.logical_data_parallel_size=1",
            "model.config.parallelism.megatron.global_packed_microbatches_per_update=3",
        ]
        runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
    else:
        from tools import run_tp_pp_native_reference as native

        native.prepare_slot = fixture.prepare
        sys.argv = [
            "native",
            "--toml",
            str(args.toml),
            "--bundle",
            str(args.bundle),
            "--output",
            str(args.output / "training"),
            "--logical-dp",
            "4",
            "--updates",
            str(args.updates),
            "--job-name",
            "fsdp4_microbatch",
            "--disable-tokenizer-compile",
        ]
        native.main()


if __name__ == "__main__":
    main()
