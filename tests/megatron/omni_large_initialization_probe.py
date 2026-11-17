"""Full-width/depth pretrained MoE two-tower initialization and LoRA update probe."""

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from async_checkpoint_smoke import writer
from omni_ema_smoke import TinyOmni, batch
from safetensors import safe_open
from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.model.generator.mot.action_tower_test import tiny_action128_spec
from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoeMoTConfig, Qwen3VLMoeTextForCausalLM
from cosmos_framework.training_backends.checkpoint_metadata import extract_chunk_metadata
from cosmos_framework.training_backends.megatron import get_runtime
from cosmos_framework.utils.distributed import ddp_sync_grad
from cosmos_framework.utils.generator.materialize_tensor import materialize_tensor
from cosmos_framework.utils.generator.optimizer import OptimizersContainer
from cosmos_framework.utils.lazy_config import LazyCall


def full_config(config_path, **kwargs):
    return Qwen3VLMoeMoTConfig(json.loads(Path(config_path).read_text()), **kwargs)


def state_sha256(state):
    digest = hashlib.sha256()

    def visit(value, path=()):
        if isinstance(value, torch.Tensor):
            local = value.to_local() if hasattr(value, "to_local") else value
            if not local.numel():
                return
            digest.update(str((path, tuple(local.shape), local.dtype)).encode())
            digest.update(local.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key], path + (key,))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, path + (index,))
        else:
            digest.update(str((path, value)).encode())

    visit(state)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--verify-hf", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--expected-checkpoint-summary", type=Path)
    parser.add_argument("--start-step", type=int, default=0)
    args = parser.parse_args()
    assert os.environ.get("COSMOS_MEGATRON_OMNI_META_INITIALIZATION") == "1"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    config = OmniMoTModelConfig()
    config.parallelism.backend = "megatron"
    config.parallelism.data_parallel_shard_degree = dist.get_world_size()
    config.parallelism.megatron_sharding_strategy = "optim_grads_params"
    config.compile.enabled, config.compile.compile_dynamic = args.compile, False
    config.ema.enabled = False
    config.lora_enabled, config.lora_rank, config.lora_alpha = True, 16, 32
    config.vision_gen = config.action_gen = True
    config.action128_spec = tiny_action128_spec()
    config.tower_structure = "two_tower"
    config.load_vision_tokenizer = config.load_vlm_visual = False
    config.tokenizer = SimpleNamespace(temporal_compression_factor=4)
    config.vlm_config.model_instance = LazyCall(Qwen3VLMoeTextForCausalLM)(
        config=LazyCall(full_config)(config_path=str(args.model_path / "config.json"))
    )
    pretrain = config.vlm_config.pretrained_weights
    pretrain.enabled, pretrain.backbone_path, pretrain.credentials_path = True, str(args.model_path), ""
    pretrain.checkpoint_format = "qwen3"
    config.diffusion_expert_config.load_weights_from_pretrained = True
    model = TinyOmni(config).to("cuda")
    optimizer = OptimizersContainer(model, "FusedAdam", fused=True, lr=1e-4, keys_to_select=["lora_"])
    restored_hash_verified = False
    if args.resume is not None:
        restored = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        dcp.load(restored, checkpoint_id=args.resume)
        model.load_state_dict(restored["model"], strict=False)
        optimizer.load_state_dict(restored["optimizer"])
        model.load_pretrained_model_if_needed(has_resumable_checkpoint=True, has_load_path=False)
        if args.expected_checkpoint_summary is not None:
            expected = json.loads(args.expected_checkpoint_summary.read_text())["ranks"][dist.get_rank()]["checkpoint"]
            actual_hash = state_sha256({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
            assert actual_hash == expected["restored_full_state_sha256"], (actual_hash, expected)
            restored_hash_verified = True
            print(
                json.dumps(dict(rank=dist.get_rank(), stage="new_process_resume_verified", sha256=actual_hash)),
                flush=True,
            )
    else:
        model.load_pretrained_model_if_needed(has_resumable_checkpoint=False, has_load_path=False)
    runtime = get_runtime(model)
    torch.cuda.synchronize()
    construction_seconds = time.perf_counter() - started
    total_parameters = sum(p.numel() for p in model.net.parameters())
    trainable_parameters = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    print(
        json.dumps(
            dict(
                rank=dist.get_rank(),
                stage="initialized",
                total_parameters=total_parameters,
                trainable_parameters=trainable_parameters,
                seconds=construction_seconds,
            )
        ),
        flush=True,
    )
    verified = 0
    if args.verify_hf:
        weight_map = json.loads((args.model_path / "model.safetensors.index.json").read_text())["weight_map"]
        state = model.net.language_model.state_dict()
        runtime.prepare_state_dict(state)
        for name, value in state.items():
            name = name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
            if ".lora_" in name:
                continue
            original = name.replace("_moe_gen", "")
            key = "model.language_model." + original[6:] if original.startswith("model.") else original
            chunk = value._local_tensor.__create_chunk_list__()[0]
            slices = tuple(slice(o, o + n) for o, n in zip(chunk.offsets, chunk.sizes, strict=True))
            with safe_open(str(args.model_path / weight_map[key]), framework="pt", device="cpu") as reader:
                expected = reader.get_slice(key)[slices]
            torch.testing.assert_close(value.to_local().cpu(), expected.to(value.dtype), rtol=0, atol=0, msg=name)
            verified += 1
            if verified % 200 == 0 and dist.get_rank() == 0:
                print(json.dumps(dict(stage="verified_hf", tensors=verified)), flush=True)
    observations = []
    for step in range(args.start_step, args.start_step + args.steps):
        optimizer.zero_grad()
        for micro in range(3):
            torch.manual_seed(42 + step * 100 + micro + dist.get_rank())
            with ddp_sync_grad(model, enabled=micro == 2):
                output = model.net(batch())["preds_action"][0]
                loss = output.float().square().mean() / 3
                assert torch.isfinite(loss)
                loss.backward()
        norm = runtime.total_grad_norm(error_if_nonfinite=True)
        assert norm.item() > 0
        optimizer.step()
        torch.cuda.synchronize()
        observations.append(dict(step=step, final_microbatch_loss=loss.item(), gradient_norm=norm.item()))
        print(json.dumps(dict(rank=dist.get_rank(), stage="updated", **observations[-1])), flush=True)
    checkpoint_result = None
    if args.checkpoint is not None:
        optimizer.zero_grad()
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        before_hash = state_sha256(saved)
        print(json.dumps(dict(rank=dist.get_rank(), stage="checkpoint_hash_before", sha256=before_hash)), flush=True)
        metadata = extract_chunk_metadata(saved)
        staged = _create_cpu_state_dict(saved, pin_memory=True, share_memory=True)
        _copy_state_dict(saved, staged)
        torch.cuda.synchronize()
        print(json.dumps(dict(rank=dist.get_rank(), stage="checkpoint_staged")), flush=True)
        context = torch.multiprocessing.get_context("spawn")
        queue = context.Queue()
        process = context.Process(
            target=writer,
            args=(queue, dist.get_rank(), dist.get_world_size(), int(os.environ["MASTER_PORT"]) + 2, args.checkpoint),
        )
        process.start()
        queue.put((staged, metadata))
        process.join(timeout=600)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        assert process.exitcode == 0, process.exitcode
        print(json.dumps(dict(rank=dist.get_rank(), stage="checkpoint_written")), flush=True)
        with torch.no_grad():
            for parameter in model.net.parameters():
                parameter.to_local().zero_()
            for value in saved["optimizer"].values():
                if isinstance(value, torch.Tensor):
                    (value.to_local() if hasattr(value, "to_local") else value).zero_()
        restored = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        dcp.load(restored, checkpoint_id=args.checkpoint)
        model.load_state_dict(restored["model"], strict=False)
        optimizer.load_state_dict(restored["optimizer"])
        after_hash = state_sha256({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        assert after_hash == before_hash, (before_hash, after_hash)
        checkpoint_result = dict(
            path=str(args.checkpoint),
            asynchronous_cpu_staging_spawn_queue=True,
            zeroed_model_and_optimizer=True,
            restored_full_state_sha256=after_hash,
            bitwise_equal=True,
        )
        print(json.dumps(dict(rank=dist.get_rank(), stage="checkpoint_restored", sha256=after_hash)), flush=True)
    digest = hashlib.sha256()
    for name, value in model.net.state_dict().items():
        if ".lora_" in name:
            digest.update(name.encode())
            digest.update(materialize_tensor(value).cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    row = dict(
        rank=dist.get_rank(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        independently_verified_hf_tensors=verified,
        lora_sha256=digest.hexdigest(),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        observations=observations,
        checkpoint=checkpoint_result,
    )
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, row)
    if dist.get_rank() == 0:
        result = dict(
            passed=True,
            meta_initialization=True,
            model_path=str(args.model_path),
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
            world_size=dist.get_world_size(),
            sharding_strategy="optim_grads_params",
            lora_rank=16,
            lora_alpha=32,
            ema=False,
            compile=args.compile,
            steps=args.steps,
            start_step=args.start_step,
            resume=str(args.resume) if args.resume is not None else None,
            new_process_full_state_hash_verified=restored_hash_verified,
            gradient_accumulation=3,
            construction_seconds=construction_seconds,
            peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in ranks),
            ranks=ranks,
            scope="Full two-tower MoE, pretrained UND→GEN and synthetic 2-text/3-action-token batches; no live loader/VAE or quality/throughput qualification.",
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
