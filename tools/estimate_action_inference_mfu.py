"""Estimate forward-only three-tower matrix MFU from saved Action128 benchmarks.

This counts useful QKV/O, SwiGLU and attention QK/PV FLOPs, not hardware
instructions. VAE and elementwise FLOPs are excluded and explicitly reported.
No GPU or checkpoint weights are loaded. Run with PYTHONPATH=.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from transformers import Qwen2Tokenizer

from cosmos_framework.data.generator.sequence_packing.modality import add_special_tokens
from cosmos_framework.data.generator.sequence_packing.sequence import packed_text_span_length
from cosmos_framework.model.generator.mot.small_action_tower import action_shape
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption


def linear_flops(tokens, hidden, intermediate, heads, kv_heads, head_dim):
    q, k = heads * head_dim, kv_heads * head_dim
    # Q,K,V,O plus gate,up,down. One multiply-add = 2 FLOPs.
    return 2 * tokens * (2 * hidden * q + 2 * hidden * k + 3 * hidden * intermediate)


def forward_flops(u, v, a, text, shape):
    layers = text["num_hidden_layers"]
    h, dh = text["num_attention_heads"], text["head_dim"]
    assert len(shape.layers) == layers and (shape.q_heads, shape.head_dim) == (h, dh)
    kv = u + v + a
    return {
        "und_linear": layers
        * linear_flops(u, text["hidden_size"], text["intermediate_size"], h, text["num_key_value_heads"], dh),
        "video_linear": layers
        * linear_flops(v, text["hidden_size"], text["intermediate_size"], h, text["num_key_value_heads"], dh),
        "action_linear": layers
        * linear_flops(a, shape.hidden, shape.intermediate, shape.q_heads, shape.kv_heads, shape.head_dim),
        "und_attention": layers * 4 * h * dh * (u * (u + 1) // 2),
        "video_attention": layers * 4 * h * dh * v * kv,
        "action_attention": layers * 4 * shape.q_heads * shape.head_dim * a * kv,
    }


def trace_summary(path):
    events = json.loads(path.read_text())["traceEvents"]
    spans = [e for e in events if e.get("cat") == "gpu_user_annotation" and e["name"] == "action_denoiser"]
    if len(spans) != 4:
        raise ValueError("Expected the saved 2-step, two-CFG-branch trace")
    kernels = [
        e
        for e in events
        if e.get("cat") == "kernel"
        and any(e["ts"] >= s["ts"] and e["ts"] + e["dur"] <= s["ts"] + s["dur"] + 1 for s in spans)
    ]
    categories = defaultdict(lambda: {"count": 0, "milliseconds": 0.0})
    for event in kernels:
        name = event["name"]
        category = (
            "FA3"
            if "FlashAttn" in name
            else "GEMM"
            if any(s in name.lower() for s in ("gemm", "nvjet", "xmma", "cublas"))
            else "Triton"
            if name.startswith("triton")
            else "other"
        )
        categories[category]["count"] += 1
        categories[category]["milliseconds"] += event["dur"] / 1000
    wall_ms = sum(e["dur"] for e in spans) / 1000
    for category in categories.values():
        category["percent_of_gpu_scope"] = category["milliseconds"] / wall_ms * 100
    return {
        "trace": str(path),
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope_count": len(spans),
        "gpu_scope_seconds": wall_ms / 1000,
        "kernels": dict(categories),
        "residual_ms_including_memcpy_memset_gaps": wall_ms - sum(c["milliseconds"] for c in categories.values()),
        "note": "Summed GPU annotation scopes from a profiled 2-step request, not the unprofiled 35-step latency.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("plans/v0.3.0d_action_infer"))
    parser.add_argument("--tokenizer", default="/share/project/eai_pwm/models/Cosmos3-Nano-VLM")
    parser.add_argument("--trace", type=Path, default=Path("outputs/action_infer_baseline/profile/trace.json"))
    parser.add_argument("--peak-tflops", type=float, default=989.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    text = json.loads(config_path.read_text())["text_config"]
    baseline_path = args.baseline_dir / "flash3_compile_results.json"
    baseline = json.loads(baseline_path.read_text())
    assert baseline["compiled_layers"] == text["num_hidden_layers"] == 36
    assert len(baseline["inputs"]) == 1
    inp = baseline["inputs"][0]
    _, frames, height, width = inp["video_shape"]
    assert frames == 41 and height == 960 and width == 640 and inp["action_shape"] == [40, 128]
    # Wan2.2 4x16x16 plus 2x2 latent patchification. Clean frame 0 is
    # still processed by all 36 layers; only the output projection excludes it.
    latent_frames = (frames - 1) // 4 + 1
    spatial_tokens = ((height // 16 + 1) // 2) * ((width // 16 + 1) // 2)
    video, action = latent_frames * spatial_tokens, inp["action_shape"][0]
    tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer)
    tokenizer, special = add_special_tokens(tokenizer)
    special["eos_token_id"] = tokenizer.eos_token_id
    branches = {}
    shape = action_shape("width_only_1b")
    for name, caption in (("conditional", inp["caption"]), ("unconditional", "")):
        ids = tokenize_caption(caption, tokenizer, is_video=False, use_system_prompt=False)
        und = packed_text_span_length(len(ids), special, has_generation=True, supervise_text=False)
        parts = forward_flops(und, video, action, text, shape)
        branches[name] = {
            "caption_tokens": len(ids),
            "und_tokens": und,
            "video_tokens": video,
            "action_tokens": action,
            "components_flops": parts,
            "forward_flops": sum(parts.values()),
        }
    rows = []
    for run in baseline["runs"]:
        selected = ["conditional", "unconditional"] if run["cfg"] else ["conditional"]
        flops = run["steps"] * sum(branches[b]["forward_flops"] for b in selected)
        sample = run["sample_seconds"]["median"]
        total = run["total_seconds"]["median"]
        rows.append(
            {
                "steps": run["steps"],
                "guidance": run["guidance"],
                "forward_calls": run["steps"] * len(selected),
                "backbone_flops": flops,
                "sample_seconds": sample,
                "total_seconds": total,
                "achieved_backbone_tflops_per_second": flops / sample / 1e12,
                "backbone_mfu_sampling_pct": 100 * flops / (sample * args.peak_tflops * 1e12),
                "backbone_mfu_total_pct": 100 * flops / (total * args.peak_tflops * 1e12),
                "ideal_backbone_seconds_at_peak": flops / (args.peak_tflops * 1e12),
            }
        )
    # These extra BF16 input/output projections are not in the main numerator.
    boundary = 2 * 192 * 4096 * (video + (latent_frames - 1) * spatial_tokens) + 4 * 128 * shape.hidden * action
    result = {
        "method": "Forward-only useful three-tower matrix FLOPs / (one GPU dense BF16 peak * measured time)",
        "excluded": [
            "VAE encode/decode",
            "norm, RoPE, softmax, activations, bias, elementwise and sampler math",
            "embedding lookup",
            "FP32 timestep MLP",
            "boundary projections (estimated separately)",
        ],
        "peak_tflops": args.peak_tflops,
        "peak_source": "cosmos_framework/callbacks/mfu.py:48",
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "text_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "branches": branches,
        "boundary_projection_flops_per_forward": boundary,
        "runs": rows,
    }
    profile = trace_summary(args.trace)
    trace_flops = 2 * sum(b["forward_flops"] for b in branches.values())
    profile["backbone_mfu_gpu_scope_pct"] = 100 * trace_flops / (args.peak_tflops * 1e12 * profile["gpu_scope_seconds"])
    result["profile"] = profile
    (args.baseline_dir / "mfu_estimate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"branches": branches, "runs": rows, "profile": profile}, indent=2))


if __name__ == "__main__":
    main()
