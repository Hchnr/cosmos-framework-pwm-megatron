# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Capture the first UND attention, then intentionally stop the math oracle.

The existing recorder does not include the frozen UND Q projection and norm.
This diagnostic adds them, the exact runtime parameter bytes, and the first
RoPE input/output and factors. It is not a completed model update or an
acceptance test. Run native and TP against the same unchanged input separately.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

UND_MODULES = ("input_layernorm", "self_attn.q_proj", "self_attn.q_norm")


class FirstUndAttentionCaptured(Exception):
    """Expected early termination after attention-call zero has been recorded."""


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def add_modules(trace):
    existing = tuple(trace.MODULE_NAMES)
    if any(name in existing for name in UND_MODULES):
        raise ValueError("The trace already includes one of the UND modules")
    trace.MODULE_NAMES = existing + UND_MODULES


def install_witness(trace, tower, weight_dir):
    recorders = []
    weight_dir = Path(weight_dir)
    old_init, old_attach, old_attend, old_rotate = (
        trace.Recorder.__init__,
        trace.Recorder.attach,
        trace.Recorder.attend,
        tower.rotate,
    )

    def initialize(recorder, *args, **kwargs):
        old_init(recorder, *args, **kwargs)
        recorder._und_rotate_calls = {}
        recorder._und_weight_seen = set()
        recorders.append(recorder)

    def attach(recorder, root, layers):
        old_attach(recorder, root, layers)
        for index, layer in layers:
            if int(index) != 0 or 0 not in recorder.layers:
                continue
            for name in UND_MODULES:
                module = layer.get_submodule(name)

                def capture_weight(current, inputs, output, *, tag=name.replace(".", "_")):
                    if recorder.active_layer != 0 or recorder.sequence != 0 or tag in recorder._und_weight_seen:
                        return
                    recorder._und_weight_seen.add(tag)
                    parameter = getattr(current, "weight", None)
                    if not isinstance(parameter, torch.Tensor):
                        raise ValueError(f"UND module {tag} has no tensor weight")
                    bias = getattr(current, "bias", None)
                    if bias is not None and not isinstance(bias, torch.Tensor):
                        raise ValueError(f"UND module {tag} has an unsupported bias")
                    # The forward hook executes while the native frozen region
                    # is still resident on GPU, before its outer leave hook.
                    result = {
                        "weight": parameter.detach().cpu().clone(),
                        "bias": None if bias is None else bias.detach().cpu().clone(),
                        "weight_device": str(parameter.device),
                        "weight_stride": tuple(parameter.stride()),
                        "weight_sha256": tensor_digest(parameter),
                        "bias_sha256": None if bias is None else tensor_digest(bias),
                        "epsilon": getattr(current, "variance_epsilon", None),
                        "rank": recorder.rank,
                        "tp_rank": recorder.tp_rank,
                        "tp_size": recorder.tp_size,
                    }
                    target = weight_dir / f"{tag}.rank{recorder.rank}.pt"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as handle:
                        torch.save(result, handle)

                module.register_forward_hook(capture_weight)

    def rotate(value, position):
        active = [r for r in recorders if r.active_layer == 0 and r.sequence == 0]
        if len(active) > 1:
            raise RuntimeError("Ambiguous UND RoPE recorder")
        if not active:
            return old_rotate(value, position)
        recorder = active[0]
        key = (recorder.sequence, recorder.active_layer)
        number = recorder._und_rotate_calls.get(key, 0)
        recorder._und_rotate_calls[key] = number + 1
        result = old_rotate(value, position)
        if number == 0:
            for kind, tensor in (("input", value), ("cos", position[0]), ("sin", position[1]), ("output", result)):
                recorder.save(tensor, 0, "und_rotate0", kind)
        return result

    def attend(recorder, original, query, key, value, mask, **kwargs):
        result = old_attend(recorder, original, query, key, value, mask, **kwargs)
        if recorder.active_layer == 0 and recorder.sequence == 0 and recorder.attention_call == 1:
            raise FirstUndAttentionCaptured("first UND attention captured")
        return result

    trace.Recorder.__init__, trace.Recorder.attach, trace.Recorder.attend = initialize, attach, attend
    tower.rotate = rotate


def validate_captured(capture_dir, weight_dir, mode, rank):
    prefix = Path(capture_dir) / mode
    expected = [
        prefix / f"layer00.{name}.{kind}.rank{rank}.pt"
        for name in ("input_layernorm", "self_attn_q_proj", "self_attn_q_norm")
        for kind in ("input", "output")
    ]
    expected += [prefix / f"layer00.und_rotate0.{kind}.rank{rank}.pt" for kind in ("input", "cos", "sin", "output")]
    expected += [
        prefix / f"layer00.attention_call0.{kind}.rank{rank}.pt" for kind in ("query", "key", "value", "output")
    ]
    expected += [Path(weight_dir) / f"{name.replace('.', '_')}.rank{rank}.pt" for name in UND_MODULES]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Early-stop capture is incomplete: {missing}")
    return len(expected)


def main():
    # The exact recorder and model are imported from the caller's isolated
    # checkout. Its CLI determines source, input, mode and precision.
    from cosmos_framework.model.generator.mot import small_action_tower
    from tools import trace_tp_pp_attention_isolation as trace

    args = sys.argv[1:]
    if args.count("--capture-dir") != 1 or args.count("--mode") != 1 or args.count("--forward-index") != 1:
        raise ValueError("Specify a fresh --capture-dir, --mode and --forward-index 0")

    def get(name):
        return args[args.index(name) + 1]

    mode, capture_dir = get("--mode"), Path(get("--capture-dir"))
    if mode not in ("native", "tp") or get("--forward-index") != "0" or get("--layers") != "0":
        raise ValueError("Capture is restricted to layer zero and the first forward")
    rank = 0 if mode == "native" else int(os.environ["RANK"])
    weight_dir = capture_dir / "weights" / mode
    add_modules(trace)
    install_witness(trace, small_action_tower, weight_dir)
    try:
        trace.main()
    except FirstUndAttentionCaptured:
        count = validate_captured(capture_dir, weight_dir, mode, rank)
        target = capture_dir / f"early_stop.{mode}.rank{rank}.json"
        with target.open("x") as handle:
            json.dump(
                {
                    "passed": True,
                    "mode": mode,
                    "rank": rank,
                    "captured_files": count,
                    "completed_update": False,
                    "full_model_acceptance": False,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
    else:
        raise RuntimeError("Oracle returned without capturing its first UND attention")


if __name__ == "__main__":
    main()
