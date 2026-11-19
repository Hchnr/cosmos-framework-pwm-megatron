# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Equal-microbatch TP4/FSDP4 with real VAE/noising and training included.

Uses verified CPU raw fixtures. Media decode is outside timing; fixture copying,
H2D, native VAE/noising, communication, backward and optimizer are measured.
Every rank uses the same raw batches and preparation RNG trajectory. This is
a performance fixture, not a native data-continuity or restart acceptance.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

from cosmos_framework.configs.toml_config import sft_config
from cosmos_framework.training_backends.megatron_tp_pp.rng import LogicalTrainingRNG
from tools import run_tp4_fsdp4_microbatch_benchmark as benchmark


class RawMicrosteps:
    directory = None

    def __init__(self, prepared_directory, microsteps):
        root = self.directory
        manifest = json.loads((root / "manifest.json").read_text())
        if not manifest["passed"] or len(manifest["records"]) != microsteps:
            raise ValueError("Require the complete verified raw microbatch fixture")
        self.raw, self.description = [], []
        for micro, record in enumerate(manifest["records"]):
            path = root / f"micro{micro:03d}.raw.pt"
            self.raw.append(torch.load(path, map_location="cpu", weights_only=False))
            self.description.append(
                {
                    "microstep": micro,
                    "source": str(path.resolve()),
                    "payload_sha256": record["sha256"],
                    "payload_kind": "raw_before_vae_and_noising",
                    "samples": record["samples"],
                    "image": record["image"],
                    "tokens": record["tokens"],
                }
            )
        states = torch.load(root / "original_logical_rng.pt", map_location="cpu", weights_only=False)
        if set(states) != {0}:
            raise ValueError("The benchmark shares one logical preparation RNG across all physical replicas")
        self.rng = LogicalTrainingRNG(states)

    def prepare(self, loader, slot, record, totals, topology, preparation, rng, **kwargs):
        if totals is not None or preparation.config.rectified_flow_training_config.sample_level_loss_averaging:
            raise ValueError("The benchmark retains native rank-level loss averaging")
        if getattr(preparation, "tokenizer_vision_gen", True) is None:
            raise ValueError("The raw benchmark requires the native VAE")
        started = time.monotonic()
        raw = copy.deepcopy(self.raw[slot.micro_step])
        copied = time.monotonic()
        with self.rng.activate(0):
            batch = preparation.prepare(raw, slot.update)
        description = self.description[slot.micro_step]
        if (
            batch["sample_count"] != description["samples"]
            or batch["loss_inputs"]["is_image_batch"] != description["image"]
        ):
            raise ValueError("Native VAE preparation changed sample/modality counts")
        batch["slot_id"] = (slot.update, slot.micro_step, slot.logical_rank)
        batch.pop("weight", None)
        record.update(
            slot=batch["slot_id"],
            source="immutable_real_raw_microbatch",
            samples=description["samples"],
            image=description["image"],
            tokens=description["tokens"],
            sha256=description["payload_sha256"],
            host_seconds={"raw_fixture_copy": copied - started, "prepare_enqueue": time.monotonic() - copied},
        )
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--raw-fixture", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    if remainder.count("--backend") != 1 or remainder.count("--output") != 1:
        raise ValueError("Require one backend and output for runtime auditing")
    backend = remainder[remainder.index("--backend") + 1]
    output = remainder[remainder.index("--output") + 1]
    from tools.tp_pp_fair_runtime import RuntimeAudit

    audit = RuntimeAudit(output, backend)
    if backend == "tp":
        from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver

        original_forward_backward = ModelParallelDriver.forward_backward

        def forward_backward(driver, batches, *positional, **keywords):
            audit.begin_update(driver.iteration)

            def checked_batches():
                for _ in range(driver.microbatches):
                    audit.modules(driver.stage)
                    audit.check("microbatch_before")
                    batch = next(iterator)
                    audit.check("microbatch_after_prepare")
                    yield batch

            iterator = iter(batches)
            result = original_forward_backward(driver, checked_batches(), *positional, **keywords)
            audit.check("update_after_forward_backward")
            return result

        ModelParallelDriver.forward_backward = forward_backward
        original_step = ModelParallelDriver.step

        def step(driver, *positional, **keywords):
            iteration = driver.iteration
            result = original_step(driver, *positional, **keywords)
            audit.end_update(iteration)
            return result

        ModelParallelDriver.step = step
    elif backend != "fsdp":
        raise ValueError(f"Unknown backend: {backend}")
    RawMicrosteps.directory = args.raw_fixture
    benchmark.PreparedMicrosteps = RawMicrosteps
    benchmark.__doc__ = __doc__
    original = sft_config.load_experiment_from_toml

    def load_config(*positional, **keywords):
        overrides = list(keywords.pop("extra_overrides", ()) or ())
        overrides = [
            value
            for value in overrides
            if not value.startswith(
                (
                    "model.config.load_vision_tokenizer=",
                    "trainer.callbacks.compile_tokenizer.enabled=",
                )
            )
        ]
        overrides += [
            "model.config.load_vision_tokenizer=true",
            "trainer.callbacks.compile_tokenizer.enabled=true",
            "trainer.cudnn.deterministic=true",
            "trainer.cudnn.benchmark=false",
        ]
        config = original(*positional, extra_overrides=overrides, **keywords)
        audit.config(config)
        return config

    sft_config.load_experiment_from_toml = load_config
    if backend == "fsdp":
        # Import after installing the config hook: native captures this symbol
        # at import time and must see the identical VAE and cuDNN overrides.
        from cosmos_framework.trainer import ImaginaireTrainer
        from tools import run_tp_pp_native_reference as native

        original_prepared_step = native.prepared_step
        original_prepare = RawMicrosteps.prepare
        original_training_step = ImaginaireTrainer.training_step

        def prepare(*positional, **keywords):
            slot = positional[2]
            if slot.micro_step == 0:
                audit.begin_update(slot.update)
            audit.check("microbatch_before")
            result = original_prepare(*positional, **keywords)
            audit.check("microbatch_after_prepare")
            return result

        def prepared_step(model, *positional, **keywords):
            audit.modules(model)
            audit.check("model_forward_before")
            result = original_prepared_step(model, *positional, **keywords)
            audit.check("model_forward_after")
            return result

        RawMicrosteps.prepare = prepare
        native.prepared_step = prepared_step

        def training_step(trainer, *positional, **keywords):
            result = original_training_step(trainer, *positional, **keywords)
            if result[2] == 0:
                audit.end_update(keywords["iteration"])
            return result

        ImaginaireTrainer.training_step = training_step
    sys.argv = [sys.argv[0], *remainder]
    try:
        benchmark.main()
    except BaseException as error:
        audit.finish(error)
        raise
    else:
        audit.finish()


if __name__ == "__main__":
    main()
