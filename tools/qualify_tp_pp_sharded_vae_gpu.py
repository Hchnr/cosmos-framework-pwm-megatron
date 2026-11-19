# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Real four-GPU VAE/noising qualification; isolated from training.

Every rank loads the production GenerationPreparation and AOT tokenizer.
For each saved raw microbatch, owner prepares the whole original batch twice
from the same logical RNG state. The first call uses native serial VAE; during
the second, all ranks split only the native VAE item loop and owner resumes the
unchanged native packing and noising. The comparison includes each clean
latent, complete prepared wire payload and final logical RNG state.

This does not construct the TP model, run backward/optimizer, or establish
whole-training memory fit or throughput. Times include audit overhead and must
not be promoted to an end-to-end samples/s claim. Use only in an allocated GPU
window through a coordinator with the shared cooperative lock.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def compare_wire(reference_header, reference_buffers, actual_header, actual_buffers, *, atol, rtol):
    """Compare the complete typed payload and return auditable tensor totals."""
    import torch

    if reference_header != actual_header or set(reference_buffers) != set(actual_buffers):
        raise ValueError("Prepared payload structure, metadata, aliasing or dtypes differ")
    stats = {}
    for name, reference in reference_buffers.items():
        actual = actual_buffers[name]
        if reference.shape != actual.shape or reference.dtype != actual.dtype:
            raise ValueError(f"Prepared payload shape/dtype differs: {name}")
        if reference.is_floating_point():
            if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(actual).all()):
                raise ValueError(f"Prepared payload contains nonfinite values: {name}")
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol, msg=f"prepared {name}")
            difference = (reference.float() - actual.float()).abs()
            maximum = float(difference.max()) if difference.numel() else 0.0
        else:
            if not torch.equal(reference, actual):
                raise ValueError(f"Prepared integer/bool tensor differs: {name}")
            maximum = 0.0
        stats[name] = {
            "elements": reference.numel(),
            "bytes": reference.numel() * reference.element_size(),
            "max_absolute_difference": maximum,
            "exact": bool(torch.equal(reference, actual)),
        }
    return stats


def validate_manifest(manifest):
    if manifest.get("passed") is not True or not manifest.get("records"):
        raise ValueError("Raw fixture must have a complete, passing capture manifest")
    for index, record in enumerate(manifest["records"]):
        if record["slot"][1] != index or record["samples"] <= 0 or not record.get("sha256"):
            raise ValueError(f"Invalid raw fixture record {index}")
    return len(manifest["records"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--raw-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1, help="Paired native and sharded warmups per fixture")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--collective-timeout", type=float, default=600.0)
    parser.add_argument("--deterministic-algorithms", action="store_true")
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 1 or args.atol < 0 or args.rtol < 0 or args.collective_timeout <= 0:
        parser.error("Require positive warmups/repetitions and nonnegative tolerances")
    return args


def main():
    args = parse_args()
    if os.environ.get("WORLD_SIZE") != "4" or not os.environ.get("RANK", "").isdecimal():
        raise ValueError("Run only via torchrun --standalone --nproc-per-node=4")
    rank = int(os.environ["RANK"])
    output = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    report = {"passed": False, "rank": rank, "failure": None, "full_training_acceptance": False}
    try:
        run(args, report)
    except BaseException:
        report["failure"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")


def run(args, report):
    import datetime

    import torch
    import torch.distributed as dist
    from torch import nn

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.model.generator.mot.action128 import Action128Normalization
    from cosmos_framework.training_backends.megatron_tp_pp.rng import LogicalTrainingRNG
    from cosmos_framework.training_backends.megatron_tp_pp.transport import encode
    from cosmos_framework.utils.generator.rand_state import get_rand_state_dict
    from cosmos_framework.utils.lazy_config import instantiate
    from tools.tp_pp_sharded_vae_experiment import assign_items, native_training_items, sharded_native_training_encode

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=args.collective_timeout))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(args.deterministic_algorithms)
    owner = rank == 0
    output_root = args.output.parent / "tokenizer_runtime"
    os.environ["OUTPUT_ROOT"] = str(output_root)
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(output_root)
    config = load_experiment_from_toml(str(args.toml), extra_overrides=["model.config.load_vision_tokenizer=true"])
    model_config = config.model.config
    if not model_config.load_vision_tokenizer or model_config.sound_gen:
        raise ValueError("Require the online vision-only/action training recipe")
    adapter = nn.Module()
    adapter.action_normalization = Action128Normalization(
        model_config.action128_spec, model_config.diffusion_expert_config.vision_spatial_position_mode
    ).to(device)
    from cosmos_framework.training_backends.megatron_tp_pp.preparation import GenerationPreparation

    start_init = time.monotonic()
    preparation = GenerationPreparation(model_config, adapter, process_group=dist.group.WORLD)
    tokenizer_seconds = time.monotonic() - start_init
    compiler = instantiate(config.trainer.callbacks.get("compile_tokenizer"))
    if compiler is None or not compiler.enabled:
        raise ValueError("Qualification requires the recipe's production AOT tokenizer")
    compiler.config = config
    compiler.process_group = dist.group.WORLD
    dist.barrier()
    start_compile = time.monotonic()
    compiler.on_training_step_start(preparation, {}, iteration=max(1200, compiler.compile_after_iterations))
    torch.cuda.synchronize()
    compiler_seconds = time.monotonic() - start_compile
    if not compiler._has_compiled:
        raise ValueError("Tokenizer AOT did not run")
    manifest = json.loads((args.raw_fixture / "manifest.json").read_text())
    count = validate_manifest(manifest)
    if owner:
        original_states = torch.load(
            args.raw_fixture / "original_logical_rng.pt", map_location="cpu", weights_only=False
        )
        if set(original_states) != {0}:
            raise ValueError("Expected exactly the original logical RNG state for rank 0")
        logical_trajectory = copy.deepcopy(original_states)
    else:
        original_states = None
    original_encoder = preparation._encode_vision_item
    original_encode_loop = preparation._encode_vision_x0_tokens
    context = {"phase": "setup", "mode": "setup", "fixture": -1}
    events = []
    captured = {}
    totals = {}

    def tracked_item(item, *, num_views, frames_per_view):
        start_event = torch.cuda.Event(enable_timing=True)
        finish_event = torch.cuda.Event(enable_timing=True)
        started = time.monotonic_ns()
        start_event.record()
        result = original_encoder(item, num_views=num_views, frames_per_view=frames_per_view)
        finish_event.record()
        events.append(
            {
                **context,
                "shape": list(item.shape),
                "input_bytes": item.numel() * item.element_size(),
                "output_bytes": result.numel() * result.element_size(),
                "num_views": num_views,
                "frames_per_view": frames_per_view,
                "host_start_ns": started,
                "host_enqueue_end_ns": time.monotonic_ns(),
                "_start": start_event,
                "_end": finish_event,
            }
        )
        return result

    preparation._encode_vision_item = tracked_item

    def native_loop(raw, grouping, condition, num_views=None, frames=None):
        start = time.monotonic_ns()
        result = original_encode_loop(raw, grouping, condition, num_views, frames)
        captured["native"] = result
        totals["native_vae_loop_host_ns"] = time.monotonic_ns() - start
        return result

    def sharded_loop(raw, grouping, condition, num_views=None, frames=None):
        start = time.monotonic_ns()
        result = sharded_native_training_encode(
            preparation,
            dist.group.WORLD,
            raw_state_vision=raw,
            num_vision_items_per_sample=grouping,
            vision_condition_indexes=condition,
            num_views_per_vision_item=num_views,
            frames_per_vision_item=frames,
            device=device,
        )
        totals["sharded_vae_collective_host_ns"] = time.monotonic_ns() - start
        # The collective's owner factory validates this data on all ranks
        # before transfers. Rebuilding it here is only for audit statistics.
        items = native_training_items(raw, grouping, condition, num_views, frames)
        owners = assign_items([item.tensor.numel() for item in items], dist.get_world_size())
        totals["assignment"] = owners
        totals["raw_p2p_bytes"] = sum(
            item.tensor.numel() * item.tensor.element_size()
            for item, worker in zip(items, owners, strict=True)
            if worker != 0
        )
        totals["latent_p2p_bytes"] = sum(
            item.numel() * item.element_size() for item, worker in zip(result, owners, strict=True) if worker != 0
        )
        captured["sharded"] = result
        return result

    rows = []
    for fixture in range(count):
        if owner:
            fixture_path = args.raw_fixture / f"micro{fixture:03d}.raw.pt"
            with fixture_path.open("rb") as handle:
                fixture_file_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            raw = torch.load(fixture_path, map_location="cpu", weights_only=False)
            if len(raw["packed_sequence"].sample_lens) != manifest["records"][fixture]["samples"]:
                raise ValueError(f"Fixture {fixture} sample count differs from its manifest")
        for run_index in range(args.warmups + args.repetitions):
            context.update(phase="warmup" if run_index < args.warmups else "qualification", fixture=fixture)
            context["run"] = run_index
            captures_before = len(events)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            dist.barrier()
            context["mode"] = "native"
            native_started = time.monotonic_ns()
            if owner:
                preparation._encode_vision_x0_tokens = native_loop
                ref_rng = LogicalTrainingRNG(copy.deepcopy(logical_trajectory))
                physical_before = get_rand_state_dict()
                with ref_rng.activate(0):
                    ref = preparation.prepare(copy.deepcopy(raw), int(manifest["records"][fixture]["slot"][0]))
                torch.cuda.synchronize()
                physical_after_ref = get_rand_state_dict()
                native_end = time.monotonic_ns()
            else:
                ref_rng, ref, native_end = None, None, None
            torch.cuda.synchronize()
            native_peak = {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            }
            torch.cuda.reset_peak_memory_stats()
            dist.barrier()
            context["mode"] = "sharded"
            sharded_started = time.monotonic_ns()
            if owner:
                preparation._encode_vision_x0_tokens = sharded_loop
                candidate_rng = LogicalTrainingRNG(copy.deepcopy(logical_trajectory))
                with candidate_rng.activate(0):
                    candidate = preparation.prepare(copy.deepcopy(raw), int(manifest["records"][fixture]["slot"][0]))
                physical_after_candidate = get_rand_state_dict()
            else:
                sharded_native_training_encode(preparation, dist.group.WORLD, device=device)
                candidate_rng, candidate = None, None
            torch.cuda.synchronize()
            sharded_end = time.monotonic_ns()
            current_events = []
            for item in events[captures_before:]:
                clean = {key: value for key, value in item.items() if not key.startswith("_")}
                clean["cuda_elapsed_ms"] = item["_start"].elapsed_time(item["_end"])
                current_events.append(clean)
            local = {
                "rank": rank,
                "native_wait_or_run_host_ns": None if native_end is None else native_end - native_started,
                "sharded_wait_or_run_host_ns": sharded_end - sharded_started,
                "native_cuda_peak_bytes": native_peak,
                "sharded_cuda_peak_bytes": {
                    "allocated": torch.cuda.max_memory_allocated(),
                    "reserved": torch.cuda.max_memory_reserved(),
                },
                "items": current_events,
            }
            worker_reports = [None] * dist.get_world_size()
            dist.all_gather_object(worker_reports, local)
            if owner:
                if len(captured["native"]) != len(captured["sharded"]):
                    raise ValueError("Native and sharded VAE returned different item counts")
                latent_errors = []
                for expected, actual in zip(captured["native"], captured["sharded"], strict=True):
                    torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
                    diff = (actual - expected).abs()
                    latent_errors.append(float(diff.max()) if diff.numel() else 0.0)
                ref_header, ref_buffers = encode(ref, device="cpu")
                test_header, test_buffers = encode(candidate, device="cpu")
                comparison = compare_wire(
                    ref_header, ref_buffers, test_header, test_buffers, atol=args.atol, rtol=args.rtol
                )
                for field, expected in ref_rng.state_dict()[0].items():
                    if not torch.equal(expected, candidate_rng.state_dict()[0][field]):
                        raise ValueError(f"Logical training RNG differs for field {field}")
                for field, expected in physical_before.items():
                    if not torch.equal(expected, physical_after_ref[field]) or not torch.equal(
                        expected, physical_after_candidate[field]
                    ):
                        raise ValueError(f"Physical owner RNG changed for field {field}")
                if run_index == args.warmups:
                    # Warmups and repeated qualification use the same snapshot;
                    # advance once in original microstep order for the next raw
                    # fixture, as the unsplit owner path would.
                    next_logical_trajectory = ref_rng.state_dict()
                rows.append(
                    {
                        "fixture": fixture,
                        "phase": context["phase"],
                        "run": run_index,
                        "original_fixture_identity": manifest["records"][fixture],
                        "manifest_semantic_hash_reverified": False,
                        "fixture_file_sha256": fixture_file_sha256,
                        "latent_max_absolute_differences": latent_errors,
                        "complete_prepared_payload": comparison,
                        "logical_rng_all_fields_exact": True,
                        "physical_owner_rng_restored_all_fields": True,
                        "sharded_timing_and_bytes": dict(totals),
                        "workers": worker_reports,
                    }
                )
                del ref, candidate, ref_buffers, test_buffers
                captured.clear()
                totals.clear()
            events.clear()
            dist.barrier()
        if owner:
            logical_trajectory = next_logical_trajectory
            del raw
    report.update(
        {
            "passed": True,
            "scope": __doc__,
            "atol": args.atol,
            "rtol": args.rtol,
            "torch_deterministic_algorithms": args.deterministic_algorithms,
            "collective_timeout_seconds": args.collective_timeout,
            "initialization": {"tokenizer_load_host_seconds": tokenizer_seconds, "aot_host_seconds": compiler_seconds},
            "rows": rows if owner else [],
            "performance_limit": "isolated preparation with numerical audit; native result stays resident during sharded measurement, all four ranks load VAE, and no TP model, end-to-end throughput, GPU timeline overlap or production memory acceptance is measured",
        }
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
