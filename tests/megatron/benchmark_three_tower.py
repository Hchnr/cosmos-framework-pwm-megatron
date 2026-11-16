"""Production-width/token-capacity MoT benchmark with precomputed synthetic inputs.

Excludes VAE and dataloading. It uses the actual three-tower decoder, frozen
UND, production Flash3-video/Flex attention and full AC, with configurable depth.
"""

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, sequence_pack_from_packed_sequence
from cosmos_framework.model.generator.mot.action_tower import prepare_joint_metadata
from cosmos_framework.model.generator.mot.small_action_tower import (
    ActionBlock,
    ActionShape,
    action_rope,
    compact_pack,
    small_decoder_inputs,
)
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, MoTDecoderLayer
from cosmos_framework.training_backends.megatron import MegatronRuntime


class Stack(nn.Module):
    def __init__(self, depth, compile_enabled, fuse_projections=False):
        super().__init__()
        config = Qwen3VLTextConfig(
            hidden_size=4096,
            intermediate_size=12288,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            rms_norm_eps=1e-6,
        )
        self.shape = ActionShape(1280, 3840, 32, 8, 128, tuple(range(depth)))
        self.layers = nn.ModuleList()
        for index in range(depth):
            layer = MoTDecoderLayer(
                config,
                layer_idx=index,
                layer_types=LayerTypes("qwen3_vl_dense"),
                qk_norm_for_text=True,
                qk_norm_for_diffusion=True,
            )
            for name, parameter in layer.named_parameters():
                parameter.requires_grad_("_moe_gen" in name)
            layer.action_block_moe_gen_action = ActionBlock(self.shape, type(layer.input_layernorm), 1e-6)
            layer.small_action_enabled = True
            if fuse_projections:
                from cosmos_framework.training_backends.projection_fusion import fuse_decoder_projections

                fuse_decoder_projections([layer])
            layer = checkpoint_wrapper(layer)
            if compile_enabled:
                layer = torch.compile(layer, fullgraph=True, dynamic=False)
            self.layers.append(layer)
        self.unused_frozen = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, pack):
        for layer in self.layers:
            pack = layer(pack, None, None)[0]
        return pack["full_only_seq"].float().square().mean() + pack["action_seq"].float().square().mean()


def inputs(shape, samples=92):
    u, v, a = 40, 640, 32
    total = u + v + a
    und, gen, action = [], [], []
    for index in range(samples):
        offset = index * total
        und.extend(range(offset, offset + u))
        gen.extend(range(offset + u, offset + total))
        action.extend(range(offset + u + v, offset + total))
    x = torch.randn(samples * total, 4096, device="cuda", dtype=torch.bfloat16)
    pack = sequence_pack_from_packed_sequence(
        x,
        ["causal", "full"] * samples,
        [u, v + a] * samples,
        [total] * samples,
        torch.tensor(und, device="cuda"),
        torch.tensor(gen, device="cuda"),
    )
    prepare_joint_metadata(pack, torch.tensor(action, device="cuda"), "native")
    ids = torch.arange(total, device="cuda").repeat(samples).expand(3, -1)
    rope = tuple(from_all_seq(t, pack) for t in action_rope(ids, x.dtype, 128))
    return small_decoder_inputs(
        compact_pack(
            pack,
            rope,
            ids,
            shape,
            fixed_lengths=(5120, 59648, 2944),
            attention_backend="flash3_video",
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("pytorch", "megatron"), required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fuse-projections", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    model = Stack(args.layers, args.compile, args.fuse_projections).cuda().to(torch.bfloat16)
    pack = inputs(model.shape)
    runtime = None
    if args.backend == "megatron":
        runtime = MegatronRuntime(
            model,
            shard_degree=dist.get_world_size(),
            unit_types=(type(model.layers[0]),),
            sharding_strategy=args.sharding_strategy,
        )
    else:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("shard",))
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
    # Match the production optimizer and its FP32 master/moment storage.
    from cosmos_framework.utils.generator.fused_adam import FusedAdam

    optimizer = FusedAdam(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-4,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.05,
        capturable=True,
        master_weights=True,
    )
    times = []
    optimizer.zero_grad(set_to_none=True)
    if runtime:
        runtime.zero_grad()
    for step in range(args.warmup + args.steps):
        dist.barrier()
        torch.cuda.synchronize()
        if step == args.warmup:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for micro in range(3):
            with runtime.accumulate(micro == 2) if runtime else nullcontext():
                loss = model(pack) / 3
                loss.backward()
        optimizer.step()
        if runtime:
            runtime.after_step()
        optimizer.zero_grad(set_to_none=True)
        if runtime:
            runtime.zero_grad()
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - start, device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if step >= args.warmup:
            times.append(elapsed.item())
        if dist.get_rank() == 0:
            print(f"step={step} seconds={elapsed.item():.4f} loss={loss.item():.6f}", flush=True)
    memory = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.int64)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        report = vars(args) | {
            "output": str(args.output),
            "world_size": dist.get_world_size(),
            "torch": torch.__version__,
            "scope": "actual MoT decoder; production widths and short-UND token capacities; synthetic inputs",
            "seed": 42,
            "dtype": "bfloat16",
            "optimizer": "FusedAdam FP32 masters",
            "gradient_accumulation": 3,
            "median_step_seconds": statistics.median(times),
            "step_seconds": times,
            "peak_allocated_bytes": memory.item(),
            "token_capacities": [5120, 59648, 2944],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
