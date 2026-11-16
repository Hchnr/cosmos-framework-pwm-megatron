"""Actual standalone Qwen3-VL: BF16/FP32 masters, frozen vision and tied embeddings."""

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGeneration,
)

from cosmos_framework.configs.base.reasoner.defaults.policy_config import VLMModelConfig
from cosmos_framework.model.generator.vlm_model import VLMModel
from cosmos_framework.training_backends.megatron import get_runtime
from cosmos_framework.utils.distributed import ddp_sync_grad
from cosmos_framework.utils.generator.optimizer import OptimizersContainer


def build(backend, path, args):
    config = copy.deepcopy(VLMModelConfig())
    config.policy.backbone.model_name = str(path)
    config.policy.attn_implementation = "sdpa"
    config.parallelism.backend = backend
    config.parallelism.data_parallel_shard_degree = dist.get_world_size() // args.replicas
    config.parallelism.data_parallel_replicate_degree = args.replicas
    config.parallelism.megatron_sharding_strategy = args.sharding_strategy
    config.parallelism.fsdp_reduce_dtype = getattr(args, "reduce_dtype", None)
    if args.moe:
        config.lbl.coeff = 0.01
        config.lbl.method = "global"
    config.compile.enabled = args.compile
    config.compile.compile_dynamic = False
    config.freeze.freeze_vision_encoder = True
    config.freeze.freeze_mm_projector = True
    if getattr(args, "freeze_embeddings", False):
        config.freeze.frozen_params = [r"embed_tokens", r"lm_head"]
    checkpoint = SimpleNamespace(
        load_path=getattr(args, "resume_load_path", ""),
        load_from_object_store=SimpleNamespace(credentials="", bucket=""),
    )
    model = VLMModel(config, checkpoint)
    model.to("cuda")  # Exercise the stock trainer's pre-optimizer device move.
    optimizer = OptimizersContainer(
        model.model, "AdamW", fused=True, lr=getattr(args, "learning_rate", 1e-3), weight_decay=0.05
    )
    return model, optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    parser.add_argument("--reduce-dtype", choices=("float32", "bfloat16"))
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    path = Path("/tmp/cosmos_megatron_tiny_qwen3_vl_moe" if args.moe else "/tmp/cosmos_megatron_tiny_qwen3_vl")
    if args.model_path is not None:
        path = args.model_path
    if args.model_path is None and dist.get_rank() == 0:
        config_type = Qwen3VLMoeConfig if args.moe else Qwen3VLConfig
        original = (
            "/share-mixed/project/eai_pwm/models/Qwen3-VL-30B-A3B-Instruct"
            if args.moe
            else "/share-mixed/project/eai_pwm/models/Qwen3-VL-2B-Instruct"
        )
        config = config_type.from_pretrained(original)
        if args.moe:
            config.text_config.num_experts = 8
            config.text_config.num_experts_per_tok = 2
            config.text_config.moe_intermediate_size = 64
        for name, value in dict(
            hidden_size=128,
            intermediate_size=256,
            head_dim=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
        ).items():
            setattr(config.text_config, name, value)
        config.text_config.rope_scaling["mrope_section"] = [4, 6, 6]
        config.image_token_id = 201
        config.video_token_id = 202
        config.vision_start_token_id = 203
        config.vision_end_token_id = 204
        for name, value in dict(
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            depth=2,
            out_hidden_size=128,
            deepstack_visual_indexes=[0, 1],
            num_position_embeddings=64,
        ).items():
            setattr(config.vision_config, name, value)
        torch.manual_seed(42)
        model_type = Qwen3VLMoeForConditionalGeneration if args.moe else Qwen3VLForConditionalGeneration
        model_type(config).float().save_pretrained(path)
    dist.barrier()
    source, source_opt = build("pytorch", path, args)
    target, target_opt = build("megatron", path, args)
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import uneven_dtensor_to_full_tensor

    maximum_loss_error = 0.0
    for step in range(3):
        source_opt.zero_grad()
        target_opt.zero_grad()
        for micro in range(3):
            torch.manual_seed(step * 100 + micro + dist.get_rank())
            ids = torch.randint(0, 200, (1, 17), device="cuda")
            extra_inputs = {}
            if args.vision:
                hf_config = source.hf_config
                ids[:, 1] = hf_config.vision_start_token_id
                ids[:, 2:6] = hf_config.image_token_id
                ids[:, 6] = hf_config.vision_end_token_id
                vision = hf_config.vision_config
                patch_features = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
                extra_inputs = dict(
                    pixel_values=torch.randn(16, patch_features, device="cuda"),
                    image_grid_thw=torch.tensor([[1, 4, 4]], device="cuda"),
                )
            losses = []
            for model in (source, target):
                with ddp_sync_grad(model, enabled=micro == 2):
                    _, loss = model.training_step(
                        dict(input_ids=ids, labels=ids.clone(), attention_mask=torch.ones_like(ids), **extra_inputs),
                        step,
                    )
                    (loss / 3).backward()
                losses.append(loss.detach())
            error = ((losses[1] - losses[0]).abs() / losses[0].abs()).item()
            maximum_loss_error = max(maximum_loss_error, error)
            assert error < 0.02, error
        source_opt.step()
        target_opt.step()
        assert get_runtime(target) is not None
        tied = target.model.net.get_input_embeddings().weight is target.model.net.get_output_embeddings().weight
        assert tied == bool(target.hf_config.tie_word_embeddings)
        expected, actual = source.state_dict(), target.state_dict()
        squared_error = torch.zeros((), device="cuda", dtype=torch.float64)
        squared_reference = squared_error.clone()
        for name, value in expected.items():
            expected_full = value.full_tensor() if hasattr(value, "full_tensor") else value
            target_value = actual[name]
            actual_full = (
                uneven_dtensor_to_full_tensor(target_value) if hasattr(target_value, "to_local") else target_value
            )
            if ".visual." in name or (args.freeze_embeddings and ("embed_tokens" in name or "lm_head" in name)):
                torch.testing.assert_close(
                    actual_full, expected_full, rtol=0, atol=0, msg=lambda message: f"{name}: {message}"
                )
            squared_error += (actual_full.double() - expected_full.double()).square().sum()
            squared_reference += expected_full.double().square().sum()
        weight_relative_l2 = (squared_error / squared_reference).sqrt().item()
        assert weight_relative_l2 < 0.002, weight_relative_l2
        # Test native → Megatron continuation, then Megatron → Megatron restart.
        # Include exact frozen FP32 state and optimizer moments, not weights only.
        if step < 2:
            checkpoint_path = Path("/tmp") / f"cosmos_megatron_vlm_state_{os.environ['MASTER_PORT']}_{step}"
            checkpoint_source = source if step == 0 else target
            checkpoint_optimizer = source_opt if step == 0 else target_opt
            dcp.save(
                {"model": checkpoint_source.state_dict(), "optimizer": checkpoint_optimizer.state_dict()},
                checkpoint_id=checkpoint_path,
            )
            if step == 1:
                before_restart = {
                    name: (uneven_dtensor_to_full_tensor(value) if hasattr(value, "to_local") else value).cpu().clone()
                    for name, value in target.state_dict().items()
                }
                args.resume_load_path = str(checkpoint_path)
                target, target_opt = build("megatron", path, args)
            restored = {"model": target.state_dict(), "optimizer": target_opt.state_dict()}
            dcp.load(restored, checkpoint_id=checkpoint_path)
            target.load_state_dict(restored["model"])
            target_opt.load_state_dict(restored["optimizer"])
            if step == 1:
                for name, value in target.state_dict().items():
                    full = uneven_dtensor_to_full_tensor(value) if hasattr(value, "to_local") else value
                    torch.testing.assert_close(full.cpu(), before_restart[name], rtol=0, atol=0)
    if dist.get_rank() == 0:
        result = dict(
            passed=True,
            meta_initialization=os.environ.get("COSMOS_MEGATRON_META_INITIALIZATION", "0") == "1",
            steps=3,
            gradient_accumulation=3,
            compile=args.compile,
            replicas=args.replicas,
            sharding_strategy=args.sharding_strategy,
            reduce_dtype=args.reduce_dtype,
            maximum_loss_relative_error=maximum_loss_error,
            weight_relative_l2=weight_relative_l2,
            native_checkpoint_continuation=True,
            megatron_checkpoint_restart=True,
            restart_weights_bitwise_equal=True,
            restart_skipped_hf_loading=True,
            vision_input=args.vision,
            moe=args.moe,
            global_auxiliary_loss_coefficient=0.01 if args.moe else None,
            model_path=str(path),
            frozen_vision_bitwise_equal=True,
            frozen_embeddings=args.freeze_embeddings,
            tied_embeddings_preserved=True,
        )
        print(json.dumps(result), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
