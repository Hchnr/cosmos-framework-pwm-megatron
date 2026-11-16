"""Single-GPU offline Action128 benchmark using JSON copies of training catalog rows.

Run from the repository with PYTHONPATH=. and COSMOS_TRAINING=0.
Catalog fields are exactly id/type/dataset/data/provenance; sample selection is
outside the catalog rows. No ground-truth future actions or frames condition WAM.
Defaults to action-only delivery for robot control; --save-video is auxiliary.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
)
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor
from cosmos_framework.inference.action_server import initialize_single_rank_distributed, model_overrides
from cosmos_framework.inference.action_serving import resolve_checkpoint
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.utils import misc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attention-backend", choices=["flash3", "flash3_video", "sdpa"], default="flash3")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--reuse-small-action-und", action="store_true")
    parser.add_argument("--reuse-small-action-metadata", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--fuse-qkv", action="store_true")
    delivery = parser.add_mutually_exclusive_group()
    delivery.add_argument(
        "--action-only", dest="action_only", action="store_true", help="Only deliver actions (default)"
    )
    delivery.add_argument(
        "--save-video", dest="action_only", action="store_false", help="Also decode/save predicted video (auxiliary)"
    )
    parser.set_defaults(action_only=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[35, 10, 5])
    parser.add_argument("--guidances", type=float, nargs="+", default=[1.5, 1.0])
    parser.add_argument("--warmup", type=int, default=2, help="Warmup calls per independent batch group")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Pack independent catalog windows, never duplicate one request"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--processor-path", type=Path)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one GPU")
    if args.warmup < 1 or args.repeats < 1 or min(args.steps) < 1:
        parser.error("warmup/repeats/steps must be positive")
    torch.cuda.set_device(0)
    initialize_single_rank_distributed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, config = resolve_checkpoint(args.checkpoint_path)
    args.vae_path = args.tokenizer_path = None
    setup = OmniSetupOverrides(
        checkpoint_path=str(checkpoint),
        config_file=str(config),
        output_dir=args.output_dir,
        experiment_overrides=model_overrides(args),
        use_ema_weights=False,
        use_torch_compile=not args.no_compile,
        compiled_region="language",
        compile_dynamic=False,
        use_cuda_graphs=args.cuda_graphs,
        guardrails=False,
        dp_shard_size=1,
        dp_replicate_size=1,
        cp_size=1,
        cfgp_size=1,
    ).build_setup(world_size=1)
    start = time.perf_counter()
    model = OmniInference.create(setup).model
    model.eval()
    fused_qkv_bytes = 0
    if args.fuse_qkv:
        from cosmos_framework.inference.action_optimization import fuse_small_action_qkv

        fused_qkv_bytes = fuse_small_action_qkv(model)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    spec = OmegaConf.to_container(model.config.action128_spec, resolve=True)
    model.net.action_normalization.validate_contract()
    model.net.action_normalization._reset_transforms()
    manifest = json.loads(args.input.read_text())
    rows = {r["id"]: r for r in manifest["catalog_rows"]}
    processor_path = args.processor_path or model.config.vlm_config.tokenizer.pretrained_model_name
    processor = Qwen3VLProcessor(name=str(processor_path))
    datasets, batches, inputs, items = {}, [], [], []
    for selection in manifest["samples"]:
        key = selection["catalog_id"]
        row = rows[key]
        if set(row) != {"id", "type", "dataset", "data", "provenance"} or row["type"] != "action":
            raise ValueError("Expected the five-column Action training catalog")
        if key not in datasets:
            data = copy.deepcopy(row["data"])
            if isinstance(data, str):
                data = json.loads(data)
            if data.pop("adapter") != "unified_action128":
                raise ValueError("Expected unified_action128 adapter")
            data.pop("format", None)
            datasets[key] = UnifiedAction128Dataset(
                **data,
                action128_spec=spec,
                tokenizer_config=processor,
                seed=args.seed,
                video_fps_max=30,
            )
        dataset = datasets[key]
        start = time.perf_counter()
        item = dataset[selection["dataset_index"]]
        # Keep only the observed first frame; remove future labels before inference.
        item["video"][:, 1:] = item["video"][:, :1].clone()
        item["action"].zero_()
        batch = VFMListCollator().collate([item])
        # Inference also packs an unconditional caption for CFG. It has a
        # different token length and never consumes training supervision masks.
        for plan in batch["sequence_plan"]:
            plan.text_loss_mask = None
        batch.update(
            make_batched_action_processing_fields(
                ActionProcessingRecord(raw_action_dim=128, action_normalizer=None),
                1,
            )
        )
        assert batch["sequence_plan"][0].condition_frame_indexes_vision == [0]
        assert not batch["sequence_plan"][0].condition_frame_indexes_action
        inputs.append(
            {
                **dataset.media_sample_identity(selection["dataset_index"]),
                "prepare_seconds": time.perf_counter() - start,
                "video_shape": list(item["video"].shape),
                "action_shape": list(item["action"].shape),
                "action_fps": item["conditioning_fps_action"],
                "video_fps": item["fps"],
                "caption": item["ai_caption"],
                "domain": int(item["normalization_domain_id"]),
            }
        )
        batches.append(batch)
        items.append(item)
    if args.batch_size < 1 or len(items) % args.batch_size:
        raise ValueError("batch-size must be positive and divide the number of selected independent windows")
    sample_groups = [list(range(i, i + args.batch_size)) for i in range(0, len(items), args.batch_size)]
    if args.batch_size > 1:
        batches = []
        for group in sample_groups:
            batch = VFMListCollator().collate([items[i] for i in group])
            for plan in batch["sequence_plan"]:
                plan.text_loss_mask = None
            batch.update(
                make_batched_action_processing_fields(
                    ActionProcessingRecord(raw_action_dim=128, action_normalizer=None), len(group)
                )
            )
            batches.append(batch)
    result = {
        "checkpoint": str(checkpoint),
        "config": str(config),
        "load_seconds": load_seconds,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "seed": args.seed,
        "command": [sys.executable, *sys.argv],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_graphs_requested": args.cuda_graphs,
        "warmup": args.warmup,
        "warmup_calls_per_run": args.warmup * len(batches),
        "repeats": args.repeats,
        "reuse_small_action_und": args.reuse_small_action_und,
        "reuse_small_action_metadata": args.reuse_small_action_metadata,
        "action_only": args.action_only,
        "batch_size": args.batch_size,
        "fused_qkv_bytes": fused_qkv_bytes,
        "sample_groups": sample_groups,
        "attention_backend": model.net.language_model.model.small_action_backend,
        "compile": OmegaConf.to_container(OmegaConf.structured(model.config.compile)),
        "inputs": inputs,
        "runs": [],
    }
    layers = model.net.language_model.model.layers
    result["compiled_layers"] = sum(hasattr(layer, "_orig_mod") for layer in layers)
    if not args.no_compile and result["compiled_layers"] != len(layers):
        raise RuntimeError("Transformer layers were not compiled")
    output = args.output_dir / "results.json"

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    def generate(batch, steps, guidance):
        return model.generate_samples_from_batch(
            batch,
            seed=[args.seed] * len(batch["sequence_plan"]),
            num_steps=steps,
            guidance=guidance,
            shift=5.0,
            reuse_small_action_und=args.reuse_small_action_und,
            reuse_small_action_metadata=args.reuse_small_action_metadata,
        )

    save()
    with torch.inference_mode():
        for steps in [] if args.profile_only else args.steps:
            for guidance in args.guidances:
                run = {
                    "steps": steps,
                    "guidance": guidance,
                    "cfg": guidance != 1,
                    "warmup_seconds": [],
                    "measurements": [],
                }
                result["runs"].append(run)
                warmup_calls = args.warmup * len(batches)
                for i in range(warmup_calls + args.repeats):
                    batch = copy.deepcopy(batches[i % len(batches)])
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    batch = misc.to(batch, device="cuda")
                    torch.cuda.synchronize()
                    transfer = time.perf_counter() - start
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    samples = generate(batch, steps, guidance)
                    torch.cuda.synchronize()
                    sample_seconds = time.perf_counter() - start
                    if i < warmup_calls:
                        run["warmup_seconds"].append(sample_seconds)
                        print(f"warmup steps={steps} cfg={guidance}: {sample_seconds:.4f}s", flush=True)
                        save()
                        continue
                    group = sample_groups[i % len(batches)]
                    actions, masks = [], []
                    for sample_offset, sample_index in enumerate(group):
                        normalized = samples["action"][sample_offset].float()
                        physical = model.net.action_normalization.denormalize(
                            normalized, inputs[sample_index]["domain"]
                        )
                        mask = batch["action_dim_mask"][sample_offset].reshape_as(physical)
                        physical = physical.masked_fill(~mask, 0)
                        if not torch.isfinite(physical).all():
                            raise RuntimeError("Nonfinite action")
                        actions.append(physical.cpu().numpy())
                        masks.append(mask.cpu().numpy())
                    torch.cuda.synchronize()
                    ready_seconds = time.perf_counter() - start
                    for offset, action in enumerate(actions):
                        suffix = "" if args.batch_size == 1 else f"_sample{offset}"
                        np.save(args.output_dir / f"action_s{steps}_g{guidance}_{i}{suffix}.npy", action)
                    video = None
                    decode_seconds = save_seconds = 0.0
                    if not args.action_only:
                        from cosmos_framework.tools.visualize.video import save_img_or_video

                        for offset, sample_index in enumerate(group):
                            start_decode = time.perf_counter()
                            video = ((model.decode(samples["vision"][offset]) + 1) / 2).clamp(0, 1).squeeze(0)
                            torch.cuda.synchronize()
                            decode_seconds += time.perf_counter() - start_decode
                            start_save = time.perf_counter()
                            suffix = "" if args.batch_size == 1 else f"_sample{offset}"
                            save_img_or_video(
                                video,
                                str(args.output_dir / f"video_s{steps}_g{guidance}_{i}{suffix}"),
                                fps=inputs[sample_index]["video_fps"],
                                quality=5,
                            )
                            torch.cuda.synchronize()
                            save_seconds += time.perf_counter() - start_save
                    record = {
                        "sample_index": group[0],
                        "sample_indices": group,
                        "chunks": len(group),
                        "transfer_seconds": transfer,
                        "sample_seconds": sample_seconds,
                        "action_ready_seconds": ready_seconds,
                        "decode_seconds": decode_seconds,
                        "save_seconds": save_seconds,
                        "total_seconds": time.perf_counter() - start + transfer,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                        "action_shape": list(actions[0].shape),
                        "action_shapes": [list(a.shape) for a in actions],
                        "active_action_slots": [np.flatnonzero(m.any(axis=0)).tolist() for m in masks],
                        "active_action_values": [int(m.sum()) for m in masks],
                        "finite": True,
                        "inactive_slots_zero": all(
                            bool((a[~m] == 0).all()) for a, m in zip(actions, masks, strict=True)
                        ),
                    }
                    run["measurements"].append(record)
                    print(json.dumps({"steps": steps, "guidance": guidance, **record}), flush=True)
                    del samples, video, batch
                    save()
                for metric in ("sample_seconds", "action_ready_seconds", "total_seconds"):
                    values = [r[metric] for r in run["measurements"]]
                    run[metric] = {
                        "median": float(np.median(values)),
                        "p95": float(np.percentile(values, 95)),
                        "min": min(values),
                        "max": max(values),
                    }
                run["chunks_per_second"] = args.batch_size / run["total_seconds"]["median"]
                run["action_ready_chunks_per_second"] = args.batch_size / run["action_ready_seconds"]["median"]
                save()
        if args.profile or args.profile_only:
            batch = misc.to(copy.deepcopy(batches[0]), device="cuda")
            # Warm the exact profiled input before collecting kernel evidence.
            generate(batch, 2, 1.5)
            torch.cuda.synchronize()
            original_denoise = model.denoise

            def profiled_denoise(*positional, **kwargs):
                with torch.profiler.record_function("action_denoiser"):
                    return original_denoise(*positional, **kwargs)

            model.denoise = profiled_denoise
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                generate(batch, 2, 1.5)
                torch.cuda.synchronize()
            events = {
                e.key: e.count
                for e in prof.key_averages()
                if any(s in e.key.lower() for s in ("flash", "scaled_dot", "compiled", "flex"))
            }
            result["profile_all_attention_events"] = events
            scoped = Counter()
            graph_launches = 0
            for event in prof.events():
                parent = event.cpu_parent
                while parent is not None and parent.name != "action_denoiser":
                    parent = parent.cpu_parent
                if parent is not None and "cudagraphlaunch" in event.name.lower():
                    graph_launches += 1
                if parent is not None and any(
                    s in event.name.lower() for s in ("flash", "scaled_dot", "compiled", "flex")
                ):
                    scoped[event.name] += 1
            result["profile_events"] = dict(scoped)
            # Graph replay hides the original FA3 CPU operator beneath a graph
            # launch. Keep CPU denoiser attribution and require actual Hopper
            # FA3 CUDA kernels as evidence; compiled-region names alone are not
            # proof that a CUDA graph or FA3 executed. CUDA kernel events do not
            # reliably carry cpu_parent links in Kineto, so kernel counts cover
            # this one generate call (the conditioning VAE may use SDPA).
            flash3_kernels = Counter(
                event.name
                for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA and "FlashAttnFwdSm90" in event.name
            )
            result["profile_graph_evidence"] = {
                "denoiser_cuda_graph_launches": graph_launches,
                "generate_flash3_cuda_kernels": dict(flash3_kernels),
            }
            model.denoise = original_denoise
            prof.export_chrome_trace(str(args.output_dir / "trace.json"))
            save()
            if args.attention_backend == "flash3":
                if args.cuda_graphs:
                    assert graph_launches > 0 and flash3_kernels, result["profile_graph_evidence"]
                else:
                    assert any("flash_attn_3" in name for name in scoped), scoped
                assert not any("scaled_dot_product" in name for name in scoped), scoped
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
