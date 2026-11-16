"""Explicit, deterministic Nano -> small-action DCP conversion (no training).

Creates self-contained checkpoints; never mutates the source. DomainAwareLinear
stores [domain,input,output], unlike nn.Linear's [output,input].
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.model.generator.mot.small_action_tower import action_shape


def evenly(source, target):
    return torch.linspace(0, source - 1, target).round().long()


def indices(preset):
    s = action_shape(preset)
    residual, ffn = evenly(4096, s.hidden), evenly(12288, s.intermediate)
    pairs = evenly(64, s.head_dim // 2)
    coords = torch.cat((pairs, pairs + 64))
    kv_heads = evenly(8, s.kv_heads)
    if s.q_heads == 12:
        q_heads = (kv_heads[:, None] * 4 + torch.arange(3)[None]).flatten()
    else:
        q_heads = torch.arange(32)
    q = (q_heads[:, None] * 128 + coords[None]).flatten()
    kv = (kv_heads[:, None] * 128 + coords[None]).flatten()
    return residual, ffn, q, kv, coords, kv_heads


def linear(w, rows, columns):
    return (w.index_select(0, rows).index_select(1, columns).float() * math.sqrt(w.shape[1] / len(columns))).to(w.dtype)


def adapter_matrix(source, target, *, paired=False):
    w = torch.zeros(target, source, dtype=torch.bfloat16)
    if target <= source:
        ix = evenly(source // 2, target // 2) if paired else evenly(source, target)
        if paired:
            ix = torch.cat((ix, ix + source // 2))
    else:
        ix = evenly(source // 2, target // 2) if paired else evenly(source, target)
        if paired:
            ix = torch.cat((ix, ix + source // 2))
    w[torch.arange(target), ix] = 1
    return w


def convert(state, preset):
    s = action_shape(preset)
    r, f, qi, ki, co, kh = indices(preset)
    out = dict(state)
    manifest = dict(
        preset=preset,
        shape=s.__dict__,
        residual_indices=r.tolist(),
        ffn_indices=f.tolist(),
        q_indices=qi.tolist(),
        kv_indices=ki.tolist(),
        coordinate_indices=co.tolist(),
        input_scaling="sqrt(source_fan_in/target_fan_in)",
        source="source checkpoint GEN",
    )
    for layer in s.layers:
        prefix = f"net.language_model.model.layers.{layer}."
        dst = prefix + "action_block_moe_gen_action."
        for name in ("input_layernorm", "post_attention_layernorm"):
            out[dst + name + ".weight"] = state[prefix + name + "_moe_gen.weight"][r].clone()
        for name in ("q_norm", "k_norm"):
            out[dst + name + ".weight"] = state[prefix + "self_attn." + name + "_moe_gen.weight"][co].clone()
        for name, row, col in (("q_proj", qi, r), ("k_proj", ki, r), ("v_proj", ki, r), ("o_proj", r, qi)):
            out[dst + name + ".weight"] = linear(state[prefix + "self_attn." + name + "_moe_gen.weight"], row, col)
        for name, row, col in (("gate_proj", f, r), ("up_proj", f, r), ("down_proj", r, f)):
            out[dst + name + ".weight"] = linear(state[prefix + "mlp_moe_gen." + name + ".weight"], row, col)
        if (s.kv_heads, s.head_dim) != (8, 128):
            for direction in ("und_to_action", "video_to_action", "action_to_video"):
                src, dstsize = (8, s.kv_heads) if preset == "width_heads_1b" else (128, s.head_dim)
                if direction == "action_to_video":
                    src, dstsize = dstsize, src
                for kind in ("k", "v"):
                    out[dst + f"adapters.{direction}_{kind}.linear.weight"] = adapter_matrix(
                        src, dstsize, paired=preset == "width_head_dim_1b"
                    )
    out["net.language_model.model.norm_moe_gen_action.weight"] = state["net.language_model.model.norm_moe_gen.weight"][
        r
    ].clone()
    for name, w in state.items():
        if name.startswith("net.time_embedder."):
            dest = name.replace("net.time_embedder.", "net.action_time_embedder.")
            if w.ndim == 1:
                out[dest] = w[r].clone()
            else:
                out[dest] = linear(w, r, torch.arange(w.shape[1]) if ".0." in name else r)
    out["net.action_modality_embed"] = state["net.action_modality_embed"][r].clone()
    # Both domain-aware maps are flattened [input,output].
    w = state["net.action2llm.fc.weight"].reshape(32, 64, 4096)
    out["net.action2llm.fc.weight"] = w[:, :, r].contiguous().flatten(1)
    out["net.action2llm.bias.weight"] = state["net.action2llm.bias.weight"][:, r].contiguous()
    w = state["net.llm2action.fc.weight"].reshape(32, 4096, 64)
    out["net.llm2action.fc.weight"] = (w[:, r, :].float() * math.sqrt(4096 / s.hidden)).to(w.dtype).flatten(1)
    manifest["action_parameters"] = sum(
        v.numel()
        for k, v in out.items()
        if any(
            x in k
            for x in ("moe_gen_action", "action_time_embedder", "action2llm", "llm2action", "action_modality_embed")
        )
    )
    if not 900_000_000 <= manifest["action_parameters"] <= 1_100_000_000:
        raise ValueError(manifest["action_parameters"])
    return out, manifest


def load_source_state(source: Path, fallback_source: Path | None = None):
    """Load regular training weights; use fallback only for absent sound heads."""
    reader = dcp.FileSystemReader(source / "model")
    metadata = reader.read_metadata().state_dict_metadata
    state = {
        k: torch.empty(tuple(v.size), dtype=v.properties.dtype)
        for k, v in metadata.items()
        if k.startswith("net.") and isinstance(v, dcp.TensorStorageMetadata)
    }
    if not state:
        raise ValueError(f"No net.* training weights in {source}")
    if any("action_block_moe_gen_action" in k for k in state):
        raise ValueError("Source must be a two-tower checkpoint")
    print(f"Loading {len(state)} regular net.* tensors from {source} on CPU", flush=True)
    dcp.load(state, storage_reader=reader)
    fallback_keys = []
    if fallback_source is not None:
        fallback_reader = dcp.FileSystemReader(fallback_source / "model")
        fallback_metadata = fallback_reader.read_metadata().state_dict_metadata
        missing = {k for k in fallback_metadata if k.startswith("net.") and k not in state}
        allowed = {
            "net.llm2sound.bias",
            "net.llm2sound.weight",
            "net.sound2llm.bias",
            "net.sound2llm.weight",
            "net.sound_modality_embed",
        }
        if missing - allowed:
            raise ValueError(f"Refusing to fill non-sound source weights: {sorted(missing - allowed)}")
        fallback = {
            k: torch.empty(tuple(fallback_metadata[k].size), dtype=fallback_metadata[k].properties.dtype)
            for k in sorted(missing)
        }
        if fallback:
            dcp.load(fallback, storage_reader=fallback_reader)
            state.update(fallback)
        fallback_keys = sorted(fallback)
    return state, fallback_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fallback-source", type=Path, help="Fill missing sound heads only; never replace trained weights"
    )
    parser.add_argument(
        "--presets", nargs="+", default=["depth_1b", "width_heads_1b", "width_head_dim_1b", "width_only_1b"]
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    for preset in args.presets:
        if (args.output / preset).exists():
            raise FileExistsError(args.output / preset)
    state, fallback_keys = load_source_state(args.source, args.fallback_source)
    source_hash = hashlib.sha256((args.source / "model/.metadata").read_bytes()).hexdigest()
    for preset in args.presets:
        dest = args.output / preset
        if dest.exists():
            raise FileExistsError(dest)
        out, manifest = convert(state, preset)
        manifest["source_metadata_sha256"] = source_hash
        manifest["source_path"] = str(args.source)
        manifest["source_weights"] = "net (non-EMA)"
        manifest["fallback_source"] = str(args.fallback_source) if args.fallback_source else None
        manifest["fallback_keys"] = fallback_keys
        dest.mkdir(parents=True)
        print("Saving", preset, manifest["action_parameters"], flush=True)
        dcp.save(out, storage_writer=dcp.FileSystemWriter(dest / "model", thread_count=8))
        (dest / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
        del out
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
