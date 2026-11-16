"""Isolated GPU diagnostics for the production small Action transformer block.

Synthetic sample geometry and random weights; production head/hidden/physical
token shapes by default. This is not an end-to-end distributed training speedup.
No dataset, checkpoint, online tracker, or production process is modified.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import _apply_ac_to_transformer_block
from cosmos_framework.model.generator.mot.small_action_attention import (
    sample_block_mask,
    sparse_attend,
    video_varlen_attend,
    video_varlen_mask,
)
from cosmos_framework.model.generator.mot.small_action_tower import (
    ActionBlock,
    ActionShape,
    action_rope,
    attend,
    project_qkv,
    rotate,
    small_layer_forward,
)
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, MoTDecoderLayer
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig


def ids_for(capacity, samples, device, skewed=False):
    real = int(capacity * 0.9)
    if skewed:
        weights = list(range(1, samples + 1))
        cumulative = [real * sum(weights[: i + 1]) // sum(weights) for i in range(samples)]
        counts = [cumulative[0], *(cumulative[i] - cumulative[i - 1] for i in range(1, samples))]
    else:
        counts = [real // samples + (i < real % samples) for i in range(samples)]
    ids = torch.repeat_interleave(torch.arange(samples, device=device), torch.tensor(counts, device=device))
    return torch.nn.functional.pad(ids, (0, capacity - real), value=-1)


def metadata(lengths, samples, device, fa3_only=False, skewed=False):
    ids = [ids_for(n, samples, device, skewed=skewed) for n in lengths]
    kv_ids = torch.cat(ids)
    masks = (
        {}
        if fa3_only
        else {
            (False, 0): sample_block_mask(ids[0], ids[0], causal=True),
            (True, 1): video_varlen_mask(ids[1], kv_ids, samples, fixed_capacity=True),
            (True, 2): sample_block_mask(ids[2], kv_ids),
        }
    )
    if fa3_only:
        masks = {
            (False, 0): video_varlen_mask(ids[0], ids[0], samples, fixed_capacity=True, causal=True),
            (True, 1): video_varlen_mask(ids[1], kv_ids, samples, fixed_capacity=True),
            (True, 2): video_varlen_mask(ids[2], kv_ids, samples, fixed_capacity=True),
        }
    query_ids = torch.cat(ids[1:])
    order = torch.argsort(torch.where(query_ids >= 0, query_ids, samples), stable=True)
    inverse = torch.argsort(order)
    merged = video_varlen_mask(query_ids[order], kv_ids, samples, fixed_capacity=True)
    if fa3_only:
        return ids, masks, (order, inverse, merged), None
    n = len(kv_ids)
    padded_ids = torch.nn.functional.pad(kv_ids, (0, (-n) % 128), value=-1)

    def allowed(b, h, q, k):
        same = (padded_ids[q] >= 0) & (padded_ids[q] == padded_ids[k])
        visible = (q >= lengths[0]) | ((k < lengths[0]) & (q >= k))
        return (same & visible) | ((padded_ids[q] < 0) & (k == 0))

    joint = create_block_mask(allowed, None, None, n, n, device=device, _compile=True)
    return ids, masks, (order, inverse, merged), joint


class PermutationGather(torch.autograd.Function):
    """Gather a true permutation; its gradient is another gather, without scatter-add."""

    @staticmethod
    def forward(ctx, x, order, inverse):
        ctx.save_for_backward(inverse)
        return x.index_select(0, order)

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        return grad.index_select(0, inverse), None, None


def attention_forward(qs, ks, vs, masks, merged, joint, variant):
    if variant == "unified_flex":
        y = sparse_attend(torch.cat(qs), torch.cat(ks), torch.cat(vs), joint)
        return y.split([q.shape[0] for q in qs])
    yu = (video_varlen_attend if variant.startswith("fa3_") else attend)(qs[0], ks[0], vs[0], masks[(False, 0)])
    k, v = torch.cat(ks), torch.cat(vs)
    if variant in ("merge_va", "fa3_merge_va", "fa3_merge_perm"):
        order, inverse, mask = merged
        if variant == "fa3_merge_perm":
            q = PermutationGather.apply(torch.cat(qs[1:]), order, inverse)
            y = video_varlen_attend(q, k, v, mask)
            y = PermutationGather.apply(y, inverse, order)
        else:
            y = video_varlen_attend(torch.cat(qs[1:])[order], k, v, mask)[inverse]
        yv, ya = y.split([qs[1].shape[0], qs[2].shape[0]])
    elif variant in ("split", "fa3_split"):
        yv = video_varlen_attend(qs[1], k, v, masks[(True, 1)])
        ya = (video_varlen_attend if variant == "fa3_split" else attend)(qs[2], k, v, masks[(True, 2)])
    else:
        raise ValueError(variant)
    return yu, yv, ya


class DiagnosticBlock(nn.Module):
    def __init__(self, variant, hidden, intermediate, action_hidden, heads, kv_heads, head_dim):
        super().__init__()
        config = Qwen3VLTextConfig(
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=36,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            attention_bias=False,
            rms_norm_eps=1e-6,
        )
        self.layer = MoTDecoderLayer(
            config,
            layer_idx=0,
            layer_types=LayerTypes("qwen3_vl_dense"),
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
        )
        self.layer.action_block_moe_gen_action = ActionBlock(
            ActionShape(action_hidden, action_hidden * 3, heads, kv_heads, head_dim, (0,)),
            type(self.layer.input_layernorm),
            1e-6,
            base_heads=kv_heads,
            base_dim=head_dim,
        )
        self.variant = variant

    def forward(self, pack, merged, joint):
        if self.variant == "split":
            out = small_layer_forward(self.layer, pack)[0]
            return out["causal_seq"], out["full_only_seq"], out["action_seq"]
        layer = self.layer
        attn, action = layer.self_attn, layer.action_block_moe_gen_action
        streams = [pack[k] for k in ("causal_seq", "full_only_seq", "action_seq")]
        norms = [layer.input_layernorm, layer.input_layernorm_moe_gen, action.input_layernorm]
        qs, ks, vs = [], [], []
        for i, (x, norm) in enumerate(zip(streams, norms, strict=True)):
            module, suffix = (action, "") if i == 2 else (attn, "" if i == 0 else "_moe_gen")
            q, k, v = project_qkv(module, norm(x), suffix)
            q = getattr(module, "q_norm" + suffix)(q.view(-1, attn.num_attention_heads, attn.head_dim))
            k = getattr(module, "k_norm" + suffix)(k.view(-1, attn.num_key_value_heads, attn.head_dim))
            qs.append(rotate(q, pack["_small_rope"][i]))
            ks.append(rotate(k, pack["_small_rope"][i]))
            vs.append(v.view(-1, attn.num_key_value_heads, attn.head_dim))
        ys = attention_forward(qs, ks, vs, pack["_small_masks"], merged, joint, self.variant)
        u = streams[0] + attn.o_proj(ys[0])
        u = u + layer.mlp(layer.post_attention_layernorm(u))
        v = streams[1] + attn.o_proj_moe_gen(ys[1])
        v = v + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(v))
        a = streams[2] + action.o_proj(ys[2])
        a = a + action.mlp(action.post_attention_layernorm(a))
        return u, v, a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", choices=["attention", "block"], default="block")
    parser.add_argument(
        "--variant",
        choices=["split", "merge_va", "unified_flex", "fa3_split", "fa3_merge_va", "fa3_merge_perm"],
        default="split",
    )
    parser.add_argument("--ac", choices=["none", "full", "save_attention", "save_downproj"], default="full")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--reduction-split-size", choices=["default", "8", "16", "32", "64", "128"], default="default")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--skewed-samples", action="store_true", help="Use unequal sample lengths for synthetic tests")
    parser.add_argument("--und-tokens", type=int, default=5120)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--memory-probe", action="store_true", help="Synchronize forward retention separately; not performance evidence"
    )
    parser.add_argument(
        "--layers", type=int, default=1, help="Repeated shared-weight blocks, for activation-memory scaling only"
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()
    compile_options = (
        {}
        if args.reduction_split_size == "default"
        else {"triton.mix_order_reduction_split_size": int(args.reduction_split_size)}
    )
    if args.samples < 1 or args.repeats < 1 or args.warmup < 0 or args.layers < 1:
        parser.error("Positive samples/repeats and nonnegative warmup required")
    if args.layers != 1 and (not args.memory_probe or args.kind != "block"):
        parser.error("Multiple shared-weight blocks are supported only for block memory probes")
    if args.kind == "attention" and args.ac != "none":
        parser.error("Attention-only timing has no block checkpoint wrapper; use --ac=none")
    if args.write_reference and args.reference is None:
        parser.error("--write-reference requires --reference")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    device, dtype = "cuda", torch.bfloat16
    lengths = (64, 256, 64) if args.tiny else (args.und_tokens, 59648, 2944)
    hidden, intermediate, action_hidden, heads, kv_heads, head_dim = (
        (256, 768, 80, 4, 1, 64) if args.tiny else (4096, 12288, 1280, 32, 8, 128)
    )
    begin = time.perf_counter()
    ids, masks, merged, joint = metadata(
        lengths, args.samples, device, fa3_only=args.variant.startswith("fa3_"), skewed=args.skewed_samples
    )
    torch.cuda.synchronize()
    metadata_s = time.perf_counter() - begin
    params = []
    if args.kind == "block":
        block = DiagnosticBlock(args.variant, hidden, intermediate, action_hidden, heads, kv_heads, head_dim)
        block = block.to(device=device, dtype=dtype).train()
        params = list(block.parameters())
        values = [
            torch.randn(n, d, device=device, dtype=dtype, requires_grad=True)
            for n, d in zip(lengths, (hidden, hidden, action_hidden), strict=True)
        ]
        rope = tuple(action_rope(torch.arange(n, device=device), dtype, head_dim) for n in lengths)
        pack = dict(zip(("causal_seq", "full_only_seq", "action_seq"), values, strict=True))
        pack.update(_small_rope=rope, _small_action_rope=rope, _small_masks=masks)
        fn = block
        if args.ac == "save_downproj":
            down_shapes = {(intermediate, hidden), (action_hidden * 3, action_hidden)}

            def policy(ctx, func, *inputs, **kwargs):
                if func == torch.ops.aten.mm.default and tuple(inputs[1].shape) in down_shapes:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.MUST_RECOMPUTE

            fn = checkpoint_wrapper(block, context_fn=lambda: create_selective_checkpoint_contexts(policy))
        elif args.ac != "none":
            fn = _apply_ac_to_transformer_block(
                block,
                ActivationCheckpointingConfig(
                    mode="full" if args.ac == "full" else "selective",
                    save_ops_regex=[r"^_flash_attn_forward\.default$"],
                ),
            )
        if args.compile:
            fn = torch.compile(fn, fullgraph=True, dynamic=False, options=compile_options)

        def forward():
            current = pack
            for _ in range(args.layers):
                outputs = fn(current, merged, joint)
                current = {**pack, **dict(zip(("causal_seq", "full_only_seq", "action_seq"), outputs, strict=True))}
            return outputs
    else:
        values = [
            torch.randn(n, h, head_dim, device=device, dtype=dtype, requires_grad=True)
            for h in (heads, kv_heads, kv_heads)
            for n in lengths
        ]

        def forward_impl(*xs):
            return attention_forward(xs[:3], xs[3:6], xs[6:], masks, merged, joint, args.variant)

        fn = (
            torch.compile(forward_impl, fullgraph=True, dynamic=False, options=compile_options)
            if args.compile
            else forward_impl
        )

        def forward():
            return fn(*values)

    real_lengths = [int((x >= 0).sum()) for x in ids]
    output_grads = [
        torch.randn(n, d, device=device, dtype=dtype) / (n * d)
        for n, d in zip(
            lengths, (hidden, hidden, action_hidden) if args.kind == "block" else (heads * head_dim,) * 3, strict=True
        )
    ]
    for grad, real in zip(output_grads, real_lengths, strict=True):
        grad[real:] = 0

    memory_snapshots = []

    def step():
        for value in [*values, *params]:
            value.grad = None
        if args.memory_probe:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
        with torch.cuda.nvtx.range("forward"):
            outputs = forward()
        if args.memory_probe:
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated()
            memory_snapshots.append(
                {"before_forward_bytes": before, "after_forward_bytes": after, "forward_retained_bytes": after - before}
            )
        with torch.cuda.nvtx.range("backward"):
            torch.autograd.backward(outputs, output_grads)
        return outputs

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    validation = None
    if args.reference is not None:
        outputs = step()
        actual = {f"output_{i}": x[: real_lengths[i]].detach() for i, x in enumerate(outputs)}
        actual.update({f"input_grad_{i}": x.grad.detach() for i, x in enumerate(values)})
        actual.update({f"param_grad_{i}": x.grad.detach() for i, x in enumerate(params) if x.grad is not None})
        assert all(bool(torch.isfinite(x).all()) for x in actual.values()), "Non-finite outputs or gradients"
        if args.write_reference:
            args.reference.parent.mkdir(parents=True, exist_ok=True)
            if args.reference.exists():
                raise FileExistsError(args.reference)
            torch.save({k: x.cpu() for k, x in actual.items()}, args.reference)
            validation = {"reference_written": str(args.reference), "tensors": len(actual)}
        else:
            reference = torch.load(args.reference, map_location="cpu", weights_only=True)
            assert actual.keys() == reference.keys()
            assert all(bool(torch.isfinite(x).all()) for x in reference.values()), "Non-finite reference tensors"
            validation = {}
            for key, x in actual.items():
                ref = reference[key].to(device).float()
                diff = x.float() - ref
                validation[key] = {
                    "relative_l2": float(diff.norm() / ref.norm().clamp_min(1e-12)),
                    "max_abs": float(diff.abs().max()),
                    "finite": bool(torch.isfinite(x).all()),
                }
            assert all(row["finite"] for row in validation.values())
            del reference, ref, diff
        del actual, outputs
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.repeats):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.cuda.nvtx.range(f"step_{i}"):
            step()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - begin)
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    result = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scope": "Single GPU, production-shaped block(s) or attention F/B; synthetic sample geometry, random weights. No optimizer, FSDP, VAE or dataloader. Profile runs are attribution only.",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "physical_lengths": lengths,
        "real_lengths": real_lengths,
        "metadata_seconds": metadata_s,
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "validation": validation,
        "forward_retention_probes": memory_snapshots[-args.repeats :] if args.memory_probe else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("validation", "seconds")}))


if __name__ == "__main__":
    main()
