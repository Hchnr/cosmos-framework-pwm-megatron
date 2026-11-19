# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in TP4 raw-fixture benchmark with item-sharded VAE preparation.

This experiment installs an in-process bridge around the existing TP training
and raw benchmark entry points; it does not change the training backend. It
requires a prior exact four-GPU VAE/noising qualification for the same fixture.
The other ranks run only frozen VAE encoding; owner packing, noise, logical RNG,
loss, and trainable model code remain the native benchmark path. No performance
or full-model numerical acceptance follows from using this entry point alone.
"""

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import tomllib

QUALIFICATION_DEPENDENCIES = (
    "tools/qualify_tp_pp_sharded_vae_gpu.py",
    "tools/tp_pp_sharded_vae_experiment.py",
    "cosmos_framework/model/generator/omni_mot_model.py",
    "cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py",
    "cosmos_framework/training_backends/megatron_tp_pp/preparation.py",
    "cosmos_framework/training_backends/megatron_tp_pp/rng.py",
    "cosmos_framework/training_backends/megatron_tp_pp/transport.py",
    "cosmos_framework/callbacks/compile_tokenizer.py",
    "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml",
)


def _digest(path):
    before = path.stat()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest


def _preparation_class(path):
    tree = ast.parse(path.read_text())
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GenerationPreparation"]
    if len(matches) != 1:
        raise ValueError(f"Expected one GenerationPreparation class in {path}")
    return ast.dump(matches[0], include_attributes=False)


def _python_syntax(path):
    return ast.dump(ast.parse(path.read_text()), include_attributes=False)


def validate_qualification(directory, fixture, source_root=None, recipe=None, environment=None):
    """Reject incomplete, tolerant or different-fixture VAE qualification."""
    source_root = source_root or Path(__file__).resolve().parents[1]
    queue = json.loads((directory / "queue_result.json").read_text())
    if queue.get("completed") is not True or queue.get("returncode") != 0 or queue.get("failure") is not None:
        raise ValueError("The VAE qualification coordinator did not complete successfully")
    pinned = queue.get("source_sha256_on_queue")
    if not pinned or pinned != queue.get("source_sha256_before_launch") or pinned != queue.get("source_sha256_after"):
        raise ValueError("VAE qualification source fingerprints are missing or changed")
    environment = os.environ if environment is None else environment
    for name in ("HF_MODEL_PATH", "WAN_VAE_PATH", "BASE_CHECKPOINT_PATH"):
        expected = queue.get("required_environment", {}).get(name, {}).get("sha256")
        actual = hashlib.sha256(environment.get(name, "").encode()).hexdigest()
        if not expected or not environment.get(name) or actual != expected:
            raise ValueError(f"Benchmark {name} differs from VAE qualification")
    vae_file = Path(environment["WAN_VAE_PATH"]).resolve(strict=True)
    weight_pins = [entry for entry in pinned.values() if Path(entry["resolved"]) == vae_file]
    if not weight_pins or len({entry["sha256"] for entry in weight_pins}) != 1:
        raise ValueError("VAE qualification must pin the VAE checkpoint file content")
    if _digest(vae_file) != weight_pins[0]["sha256"]:
        raise ValueError("VAE checkpoint content changed after qualification")
    for relative in QUALIFICATION_DEPENDENCIES:
        matches = [entry for label, entry in pinned.items() if label.endswith("/" + relative)]
        if len(matches) != 1:
            raise ValueError(f"Missing pinned qualification dependency: {relative}")
        old = Path(matches[0]["resolved"])
        candidate = source_root / relative
        if _digest(old) != matches[0]["sha256"]:
            raise ValueError(f"Qualified source no longer matches its pinned hash: {relative}")
        if relative.endswith("/preparation.py"):
            # PR #43 changed GenerationLoss and an import, not the preparation
            # class. Compare the actual class syntax, in addition to pinning
            # the full old file, so that fix does not invalidate this gate.
            equal = _preparation_class(old) == _preparation_class(candidate)
        elif relative.endswith(("/rng.py", "/transport.py")):
            # SPDX headers differ between the protected checkout used for the
            # qualification and main; require their complete Python syntax to
            # match, excluding comments and source locations.
            equal = _python_syntax(old) == _python_syntax(candidate)
        elif relative.endswith(".toml"):
            equal = tomllib.loads(old.read_text()) == tomllib.loads(candidate.read_text())
        else:
            equal = _digest(candidate) == matches[0]["sha256"]
        if not equal:
            raise ValueError(f"Benchmark and qualified VAE source differ: {relative}")
    if recipe is not None:
        qualified_recipe = (
            source_root / "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"
        )
        if tomllib.loads(recipe.read_text()) != tomllib.loads(qualified_recipe.read_text()):
            raise ValueError("Benchmark recipe differs from the qualified VAE recipe")
    manifest = json.loads((fixture / "manifest.json").read_text())
    if not manifest.get("passed") or len(manifest.get("records", [])) != 3:
        raise ValueError("Require the complete three-microbatch raw fixture")
    reports = []
    for rank in range(4):
        path = directory / f"qualification.rank{rank}.json"
        report = json.loads(path.read_text())
        if report.get("rank") != rank or report.get("passed") is not True:
            raise ValueError(f"Missing passing VAE qualification for rank {rank}")
        if report.get("atol") != 0 or report.get("rtol") != 0:
            raise ValueError("VAE qualification must use exact comparison")
        reports.append(report)
    rows = [row for row in reports[0].get("rows", []) if row.get("phase") == "qualification"]
    verified = []
    for index, record in enumerate(manifest["records"]):
        selected = [row for row in rows if row.get("fixture") == index]
        if not selected:
            raise ValueError(f"No exact VAE qualification for microbatch {index}")
        fixture_path = fixture / f"micro{index:03d}.raw.pt"
        fixture_hash = _digest(fixture_path)
        for row in selected:
            errors = row.get("latent_max_absolute_differences")
            if (
                row.get("fixture_file_sha256") != fixture_hash
                or row.get("original_fixture_identity") != record
                or row.get("logical_rng_all_fields_exact") is not True
                or row.get("physical_owner_rng_restored_all_fields") is not True
                or len(row.get("workers", [])) != 4
                or not row.get("complete_prepared_payload")
                or any(not item.get("exact") for item in row["complete_prepared_payload"].values())
                or not errors
                or any(error != 0 for error in errors)
            ):
                raise ValueError(f"VAE output, RNG or fixture contract failed for microbatch {index}")
        verified.append(fixture_hash)
    return verified


class PreparationBridge:
    """Per-slot control protocol before the existing TP batch broadcast."""

    ENCODE, DONE, FAILED = 1, 2, 3

    def __init__(self, group, preparation, *, owner, send_control, shard_encode, preserve_rng):
        self.group, self.preparation, self.owner = group, preparation, owner
        self.send_control, self.shard_encode, self.preserve_rng = send_control, shard_encode, preserve_rng
        self.calls = self.slots = 0
        self.in_slot = False
        self.failed_inside_collective = False

    def owner_encode(self, raw, grouping, condition, num_views=None, frames=None):
        if not self.owner or not self.in_slot:
            raise RuntimeError("Owner VAE encode occurred outside a guarded benchmark slot")
        self.send_control(self.ENCODE, self.group)
        try:
            result = self.shard_encode(
                self.preparation,
                self.group,
                raw_state_vision=raw,
                num_vision_items_per_sample=grouping,
                vision_condition_indexes=condition,
                num_views_per_vision_item=num_views,
                frames_per_vision_item=frames,
            )
        except BaseException:
            # Peer ranks receive the same error from the sharded collective;
            # do not start a second control collective after that failure.
            self.failed_inside_collective = True
            raise
        self.calls += 1
        return result

    def prepare(self, function, *args, **kwargs):
        if not self.owner or self.in_slot:
            raise RuntimeError("Invalid owner preparation nesting")
        self.in_slot, self.failed_inside_collective = True, False
        try:
            value = function(*args, **kwargs)
        except BaseException:
            if not self.failed_inside_collective:
                self.send_control(self.FAILED, self.group)
            raise
        else:
            self.send_control(self.DONE, self.group)
            self.slots += 1
            return value
        finally:
            self.in_slot = False

    def serve(self):
        if self.owner:
            raise RuntimeError("Only peers serve owner preparation")
        while True:
            flag = self.send_control(None, self.group)
            if flag == self.DONE:
                self.slots += 1
                return
            if flag == self.FAILED:
                raise RuntimeError("TP owner failed while preparing its microbatch")
            if flag != self.ENCODE:
                raise RuntimeError(f"Invalid VAE control code: {flag}")
            # Workers never consume the owner's logical noise stream. Preserve
            # their physical CPU/CUDA/Python/NumPy state around frozen VAE.
            with self.preserve_rng():
                self.shard_encode(self.preparation, self.group)
            self.calls += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--vae-qualification-dir", type=Path, required=True)
    parser.add_argument("--bridge-report-dir", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    if os.environ.get("WORLD_SIZE") != "4" or not os.environ.get("RANK", "").isdecimal():
        raise ValueError("Run with torchrun and exactly four ranks")
    if os.environ.get("COSMOS_TP_PP_PARALLEL_PREPARATION", "0") != "0":
        raise ValueError("This experiment cannot combine with parallel logical-stream preparation")
    if remainder.count("--backend") != 1 or remainder[remainder.index("--backend") + 1] != "tp":
        raise ValueError("This wrapper is limited to the TP4 raw benchmark")
    if remainder.count("--raw-fixture") != 1:
        raise ValueError("A single --raw-fixture is required")
    if remainder.count("--toml") != 1:
        raise ValueError("A single --toml is required")
    for option, expected in (("--updates", "7"), ("--warmup-updates", "2")):
        if any(value.startswith(option + "=") for value in remainder):
            raise ValueError(f"Use {option} {expected} as separate arguments")
        if (
            remainder.count(option) > 1
            or option in remainder
            and (remainder.index(option) + 1 >= len(remainder) or remainder[remainder.index(option) + 1] != expected)
        ):
            raise ValueError("The benchmark requires seven updates and two warmups")
    fixture = Path(remainder[remainder.index("--raw-fixture") + 1])
    recipe = Path(remainder[remainder.index("--toml") + 1])
    verified_hashes = validate_qualification(args.vae_qualification_dir, fixture, recipe=recipe)
    args.bridge_report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.bridge_report_dir / f"sharded_vae_bridge.rank{os.environ['RANK']}.json"
    if report_path.exists():
        raise FileExistsError(report_path)

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps

    from cosmos_framework.callbacks.compile_tokenizer import CompileTokenizer
    from cosmos_framework.training_backends.megatron_tp_pp import train
    from cosmos_framework.utils.generator.rand_state import preserve_training_rng
    from tools import run_tp4_fsdp4_raw_microbatch_benchmark as raw_benchmark
    from tools.tp_pp_sharded_vae_experiment import sharded_native_training_encode

    state = {"preparation": None, "compiler": None, "bridge": None}
    original_preparation, original_instantiate = train.GenerationPreparation, train.instantiate
    original_run, original_broadcast = train.run, train.broadcast_batch
    original_prepare = raw_benchmark.RawMicrosteps.prepare

    class CapturePreparation(original_preparation):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            if state["preparation"] is not None:
                raise RuntimeError("Multiple owner preparation instances")
            state["preparation"] = self

    def capture_instantiate(*positional, **keywords):
        value = original_instantiate(*positional, **keywords)
        if isinstance(value, CompileTokenizer):
            state["compiler"] = value
        return value

    def control(flag, group):
        source = dist.get_global_rank(group, 0)
        value = torch.tensor([flag or 0], dtype=torch.int32, device="cuda")
        dist.broadcast(value, src=source, group=group)
        return int(value.item())

    def run(config, run_args=None, *, initialized_callback=None):
        def ready(driver):
            if (
                ps.get_tensor_model_parallel_world_size() != 4
                or ps.get_pipeline_model_parallel_world_size() != 1
                or ps.get_data_parallel_world_size() != 1
                or not config.model.config.load_vision_tokenizer
            ):
                raise ValueError("Require TP4/PP1/DP1 with online vision tokenizer")
            owner = ps.get_tensor_model_parallel_rank() == 0
            group = ps.get_tensor_model_parallel_group()
            if owner:
                preparation, compiler = state["preparation"], state["compiler"]
                if preparation is None or compiler is None:
                    raise RuntimeError("Owner tokenizer or compiler was not created")
            else:
                with preserve_training_rng():
                    preparation = original_preparation(
                        config.model.config, driver.stage.input_adapter, process_group=ps.get_data_parallel_group()
                    )
                    compiler = original_instantiate(config.trainer.callbacks.get("compile_tokenizer"))
                if compiler is None:
                    raise RuntimeError("Missing worker tokenizer compiler")
            if not compiler.enabled:
                raise ValueError("Require the same production VAE AOT configuration on every rank")
            compiler.config, compiler.process_group = config, group
            # Both baseline and candidate exclude VAE compile from measured
            # updates. All candidate ranks use the same distributed AOT path.
            dist.barrier(group=group)
            compiler.on_training_step_start(preparation, {}, iteration=driver.iteration)
            if not compiler._has_compiled:
                raise RuntimeError("VAE AOT did not run on every TP rank")
            dist.barrier(group=group)
            bridge = PreparationBridge(
                group,
                preparation,
                owner=owner,
                send_control=control,
                shard_encode=sharded_native_training_encode,
                preserve_rng=preserve_training_rng,
            )
            state["bridge"] = bridge
            if owner:
                preparation._encode_vision_x0_tokens = bridge.owner_encode
            if initialized_callback is not None:
                initialized_callback(driver)

        return original_run(config, run_args, initialized_callback=ready)

    def prepare(self, *positional, **keywords):
        bridge = state["bridge"]
        if bridge is None:
            raise RuntimeError("VAE bridge was not initialized")
        return bridge.prepare(original_prepare, self, *positional, **keywords)

    def broadcast(value, group, **keywords):
        bridge = state["bridge"]
        if bridge is None or group is not bridge.group:
            raise RuntimeError("Unexpected TP batch broadcast or bridge lifecycle")
        if not bridge.owner:
            bridge.serve()
        return original_broadcast(value, group, **keywords)

    train.GenerationPreparation, train.instantiate, train.run, train.broadcast_batch = (
        CapturePreparation,
        capture_instantiate,
        run,
        broadcast,
    )
    raw_benchmark.RawMicrosteps.prepare = prepare
    report = {
        "passed": False,
        "rank": int(os.environ["RANK"]),
        "fixture_file_sha256": verified_hashes,
        "qualification_weight_file_content_hashed": True,
        "full_model_numerical_acceptance": False,
    }
    try:
        sys.argv = [sys.argv[0], *remainder]
        raw_benchmark.main()
        bridge = state["bridge"]
        if bridge is None or bridge.slots != 21 or bridge.calls != 21:
            raise RuntimeError("Expected exactly 7 updates × 3 online VAE microsteps")
        report.update(passed=True, slots=bridge.slots, vae_calls=bridge.calls)
    except BaseException as error:
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
