"""GPU numerical validation; invokes the shared benchmark without changing it.

Validation intentionally performs extra reference forwards; its timings are not
performance results. Run with benchmark flags, one visible GPU and --action-only.
"""

import copy
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from tools import benchmark_action_inference as benchmark

MAX_ABS = 0.05
RMSE = 0.01


def metrics(actual, expected, mask=None):
    actual, expected = actual.float(), expected.float()
    if mask is not None:
        actual, expected = actual[mask], expected[mask]
    diff = actual - expected
    return {
        "max_abs": diff.abs().max().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "count": diff.numel(),
    }


@contextmanager
def unfused(model):
    removed = []
    for module in model.net.modules():
        for name in list(module._buffers):
            if name.startswith("_inference_qkv_"):
                removed.append((module, name, module._buffers.pop(name)))
    try:
        yield
    finally:
        for module, name, value in removed:
            module._buffers[name] = value


def main():
    # Validation alternates fused/unfused, cached/full and two CFG layouts.
    # These deliberately create more specializations than a deployed replica.
    torch._dynamo.config.recompile_limit = 64
    out = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "thresholds": {"max_abs": MAX_ABS, "rmse": RMSE},
        "velocity": [],
        "actions": [],
        "limitations": [
            "BF16 reference; no full-model FP32 or closed-loop quality claim",
            "Reference forwards reuse identical current noisy input; full actions use same seed",
        ],
    }
    original_generate = OmniMoTModel.generate_samples_from_batch
    original_velocity = OmniMoTModel._get_velocity
    validating = False
    request = 0
    current_mask = None
    first_memory = [None]

    def checked_velocity(self, **kwargs):
        if not validating:
            return original_velocity(self, **kwargs)
        ref_kwargs = dict(kwargs, memory=None, packed_sequence_template=None)
        # The reference owns no UND cache or compact metadata from optimized execution.
        with unfused(self):
            reference = [x.detach().clone() for x in original_velocity(self, **ref_kwargs)]
        actual = original_velocity(self, **kwargs)
        clean = kwargs["gen_data_clean"]
        action_tokens = clean.x0_tokens_action
        action_count = action_tokens[0].numel() if action_tokens else 0
        if first_memory[0] is None:
            first_memory[0] = id(kwargs.get("memory"))
        row_branch = "cond" if id(kwargs.get("memory")) == first_memory[0] else "uncond"
        row = {
            "request": request,
            "branch": row_branch,
            "timestep": kwargs["timestep"].flatten()[0].item(),
            "skip_text_tokens": kwargs["skip_text_tokens"],
            "full_velocity": metrics(torch.cat(actual), torch.cat(reference)),
        }
        if action_count:
            # Flattening contract is [vision | action | optional sound]; this WAM has no sound.
            row["action_velocity"] = metrics(actual[0][-action_count:], reference[0][-action_count:])
            if current_mask is not None:
                row["valid_action_velocity"] = metrics(
                    actual[0][-action_count:], reference[0][-action_count:], current_mask.flatten()
                )
        summary["velocity"].append(row)
        return actual

    def checked_generate(self, batch, **kwargs):
        nonlocal validating, request, current_mask
        torch._dynamo.config.recompile_limit = 64
        current_mask = batch["action_dim_mask"][0].bool()
        request += 1
        # Independent baseline requests also exercise release/recreation of optimized caches.
        with unfused(self):
            baseline = original_generate(
                self,
                copy.deepcopy(batch),
                **dict(kwargs, reuse_small_action_und=False, reuse_small_action_metadata=False),
            )
        first_memory[0] = None
        validating = request <= 2
        try:
            actual = original_generate(self, copy.deepcopy(batch), **kwargs)
        finally:
            validating = False
        masks = batch["action_dim_mask"] if "action_dim_mask" in batch else None
        for index, (a, b) in enumerate(zip(actual["action"], baseline["action"], strict=True)):
            mask = masks[index].bool().reshape_as(a) if masks is not None else None
            row = {
                "request": request,
                "sample": index,
                "seed": kwargs.get("seed"),
                "guidance": kwargs.get("guidance"),
                "all_slots": metrics(a, b),
                "valid_slots": metrics(a, b, mask),
                "mask_available": mask is not None,
            }
            summary["actions"].append(row)
            torch.save(
                {"baseline": b.cpu(), "optimized": a.cpu(), "valid_mask": mask.cpu() if mask is not None else None},
                out / f"actions_{request}_{index}.pt",
            )
        summary["passed"] = all(
            m["finite"] and m["max_abs"] <= MAX_ABS and m["rmse"] <= RMSE
            for m in [r["valid_slots"] for r in summary["actions"]]
            + [
                r.get("valid_action_velocity", r["action_velocity"])
                for r in summary["velocity"]
                if "action_velocity" in r
            ]
        )
        (out / "correctness.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("CORRECTNESS", json.dumps(summary["actions"][-1]), flush=True)
        return actual

    OmniMoTModel.generate_samples_from_batch = checked_generate
    OmniMoTModel._get_velocity = checked_velocity
    benchmark.main()
    if not summary.get("passed", False):
        raise SystemExit("Numerical validation failed; see correctness.json")


if __name__ == "__main__":
    main()
