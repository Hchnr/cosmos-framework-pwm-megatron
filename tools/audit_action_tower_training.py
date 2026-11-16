#!/usr/bin/env python3
"""Audit the real training entry point without changing its tensors or kernels.

Run with torchrun, passing the normal train.py arguments. AUDIT_OUTPUT_DIR
selects the JSON report directory (one file per rank). Use compile.enabled=false:
Python hooks are instrumentation, not a throughput benchmark.
AUDIT_EXPECT_FROZEN_UND=1 additionally requires UND activations to stay outside
autograd; only use it when text/visual inputs and the entire UND tower are frozen.
"""

from __future__ import annotations

import atexit
import json
import os
import runpy
from collections import Counter
from pathlib import Path

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import get_und_seq
from cosmos_framework.model.generator.mot import action_tower
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def main():
    directory = Path(os.environ.get("AUDIT_OUTPUT_DIR", "outputs/action_tower_training_audit"))
    directory.mkdir(parents=True, exist_ok=True)
    report = {"rank": int(os.environ.get("RANK", "0")), "steps": [], "violations": []}
    modules, attentions, backward_activations = Counter(), Counter(), Counter()
    phase = "setup"
    active = None
    installed = False

    def save():
        report["module_calls"] = [{"signature": json.loads(k), "calls": v} for k, v in modules.items()]
        report["attention_calls"] = [{"signature": json.loads(k), "calls": v} for k, v in attentions.items()]
        report["activation_backward_calls"] = dict(backward_activations)
        (directory / f"rank_{report['rank']}.json").write_text(json.dumps(report, indent=2) + "\n")

    atexit.register(save)

    def module_hook(name):
        def hook(module, args, output):
            if active is None or not isinstance(output, torch.Tensor):
                return
            role = "action" if "_moe_gen_action" in name else "video" if "_moe_gen" in name else "und"
            signature = {
                "name": name,
                "phase": phase,
                "role": role,
                "rows": args[0].shape[0],
                "input_dtype": str(args[0].dtype),
                "output_dtype": str(output.dtype),
                "input_requires_grad": args[0].requires_grad,
                "output_requires_grad": output.requires_grad,
            }
            modules[json.dumps(signature, sort_keys=True)] += 1
            if args[0].shape[0] != active[role]:
                report["violations"].append({"kind": "routing", **signature, "expected": active[role]})
            if args[0].dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
                report["violations"].append({"kind": "precision", **signature})
            if phase == "forward" and output.requires_grad:

                def backward_hook(grad):
                    backward_activations[name] += 1

                output.register_hook(backward_hook)

        return hook

    original_layer = action_tower.layer_forward

    def layer_forward(layer, pack, *args, **kwargs):
        nonlocal active
        previous = active
        active = {
            "und": get_und_seq(pack).shape[0],
            "video": pack["_video_indices"].numel(),
            "action": pack["_action_indices"].numel(),
        }
        try:
            return original_layer(layer, pack, *args, **kwargs)
        finally:
            active = previous

    action_tower.layer_forward = layer_forward
    from cosmos_framework.model.generator.mot import attention as attention_module

    original_attention = attention_module.attention

    def attention(q, k, v, *args, **kwargs):
        signature = {
            "phase": phase,
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "q_dtype": str(q.dtype),
            "k_dtype": str(k.dtype),
            "v_dtype": str(v.dtype),
            "causal": kwargs.get("is_causal", False),
        }
        attentions[json.dumps(signature, sort_keys=True)] += 1
        return original_attention(q, k, v, *args, **kwargs)

    attention_module.attention = attention
    original_step = OmniMoTModel.training_step

    def training_step(self, data, iteration):
        nonlocal installed, phase
        if not installed:
            if self.config.compile.enabled:
                raise ValueError("Training audit hooks require model.compile.enabled=false")
            if self.config.tower_structure != "three_tower":
                raise ValueError("This audit requires three_tower")
            counts = Counter()
            for name, parameter in self.net.named_parameters():
                counts[f"{parameter.dtype}/trainable={parameter.requires_grad}"] += parameter.numel()
            report["parameter_elements"] = dict(counts)
            report["execution"] = self.config.action_tower_execution
            report["precision"] = self.config.precision
            report["activation_checkpointing"] = self.config.activation_checkpointing.mode
            for name, module in self.net.language_model.model.layers.named_modules():
                if isinstance(module, torch.nn.Linear) or "layernorm" in name or name.endswith(("q_norm", "k_norm")):
                    module.register_forward_hook(module_hook(name))
            installed = True
        phase = "forward"
        output, loss = original_step(self, data, iteration)
        report["steps"].append(
            {"iteration": iteration, "loss": loss.detach().float().item(), "loss_dtype": str(loss.dtype)}
        )
        phase = "backward_recompute"
        save()
        return output, loss

    OmniMoTModel.training_step = training_step
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
    report["training_completed"] = True
    forward_projection_calls = Counter()
    for key, count in modules.items():
        signature = json.loads(key)
        leaf = signature["name"].split(".")[-1]
        if signature["phase"] == "forward" and leaf.startswith(("q_proj", "k_proj", "v_proj")):
            forward_projection_calls[signature["name"]] += count
    report["forward_projection_calls"] = dict(forward_projection_calls)
    und_backward_calls = sum(count for name, count in backward_activations.items() if "_moe_gen" not in name)
    report["und_activation_backward_calls"] = und_backward_calls
    if os.environ.get("AUDIT_EXPECT_FROZEN_UND") == "1":
        for key in modules:
            signature = json.loads(key)
            if signature["role"] == "und" and signature["output_requires_grad"]:
                report["violations"].append({"kind": "frozen_und_graph", **signature})
        if und_backward_calls:
            report["violations"].append({"kind": "frozen_und_backward", "calls": und_backward_calls})
    expected_calls = len(report["steps"])
    if not forward_projection_calls:
        report["violations"].append({"kind": "missing_projections"})
    for name, calls in forward_projection_calls.items():
        if calls != expected_calls:
            report["violations"].append(
                {"kind": "projection_count", "name": name, "calls": calls, "expected": expected_calls}
            )
    save()
    if report["violations"]:
        raise RuntimeError(f"Training audit failed; see {directory}")


if __name__ == "__main__":
    main()
