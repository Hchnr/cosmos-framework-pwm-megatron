"""Locate first standalone VLM forward divergence without changing tolerances."""

import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten
from vlm_smoke import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    options = SimpleNamespace(replicas=2, compile=True, sharding_strategy="optim_grads", moe=False)
    source, _ = build("pytorch", args.model_path, options)
    target, _ = build("megatron", args.model_path, options)
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import uneven_dtensor_to_full_tensor

    weight_differences = []
    source_state, target_state = source.state_dict(), target.state_dict()
    for name, value in source_state.items():
        left = value.full_tensor() if hasattr(value, "full_tensor") else value
        right = target_state[name]
        right = uneven_dtensor_to_full_tensor(right) if hasattr(right, "to_local") else right
        if not torch.equal(left, right):
            weight_differences.append(dict(name=name, maximum_absolute_error=(left - right).abs().max().item()))
    captures = [{}, {}]
    handles = []

    def snapshot(value):
        return [t.detach().cpu().clone() for t in tree_flatten(value)[0] if isinstance(t, torch.Tensor)]

    for index, model in enumerate((source, target)):
        seen = set()
        for name, module in model.model.net.named_modules():
            canonical = name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
            if canonical in seen:
                continue
            if not (
                re.search(
                    r"(embed_tokens|patch_embed|pos_embed|layers\.\d+|blocks\.\d+|merger|norm|lm_head)$", canonical
                )
            ):
                continue
            seen.add(canonical)

            def hook(module, inputs, kwargs, output, key=canonical, slot=index):
                captures[slot][key] = {"inputs": snapshot((inputs, kwargs)), "outputs": snapshot(output)}

            handles.append(module.register_forward_hook(hook, with_kwargs=True))
    torch.manual_seed(dist.get_rank())
    ids = torch.randint(0, 200, (1, 17), device="cuda")
    config = source.hf_config
    ids[:, 1] = config.vision_start_token_id
    ids[:, 2:6] = config.image_token_id
    ids[:, 6] = config.vision_end_token_id
    vision = config.vision_config
    pixels = torch.randn(16, vision.in_channels * vision.temporal_patch_size * vision.patch_size**2, device="cuda")
    losses = []
    with torch.no_grad():
        for model in (source, target):
            _, loss = model.training_step(
                dict(
                    input_ids=ids,
                    labels=ids.clone(),
                    attention_mask=torch.ones_like(ids),
                    pixel_values=pixels,
                    image_grid_thw=torch.tensor([[1, 4, 4]], device="cuda"),
                ),
                0,
            )
            losses.append(loss.item())
    comparisons = []
    for name, left in captures[0].items():
        right = captures[1][name]
        fields = {}
        for key in ("inputs", "outputs"):
            values = []
            for a, b in zip(left[key], right[key], strict=True):
                difference = a.double() - b.double()
                values.append(
                    dict(
                        shape=list(a.shape),
                        source_dtype=str(a.dtype),
                        target_dtype=str(b.dtype),
                        bitwise_equal=torch.equal(a, b),
                        maximum_absolute_error=difference.abs().max().item(),
                        relative_l2=(difference.norm() / a.double().norm().clamp_min(1e-30)).item(),
                    )
                )
            fields[key] = values
        comparisons.append(dict(name=name, **fields))
    if dist.get_rank() == 0:
        args.output.write_text(
            json.dumps(
                dict(initial_weight_differences=weight_differences, losses=losses, module_comparisons=comparisons),
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(dict(initial_weight_differences=len(weight_differences), losses=losses)))
    for handle in handles:
        handle.remove()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
