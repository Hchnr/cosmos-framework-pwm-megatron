# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in attention/QK-norm trace for one real forward; never an acceptance test.

The wrapper instruments native or TP eager execution without editing model code.
It records layer 30--32 module/attention tensors and cotangents, on every TP
rank, for forward index 1 by default. Full tensors up to the element limit are
saved; larger tensors retain the first/last rows along dimension 0 when that
axis is long enough, otherwise the full tensor is kept. Each record
contains its original dtype, shape, strides, sample scope and SHA256 of the
stored bytes. Input hooks see the accumulated gradient if a tensor is shared
by branches; use attention output cotangents for an isolated operator replay.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

MODULE_NAMES = (
    "input_layernorm_moe_gen",
    "post_attention_layernorm_moe_gen",
    "self_attn.q_proj_moe_gen",
    "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen",
    "self_attn.q_norm_moe_gen",
    "self_attn.k_norm_moe_gen",
    "self_attn.o_proj_moe_gen",
    "mlp_moe_gen.gate_proj",
    "mlp_moe_gen.up_proj",
    "mlp_moe_gen.down_proj",
    "action_block_moe_gen_action.input_layernorm",
    "action_block_moe_gen_action.q_proj",
    "action_block_moe_gen_action.k_proj",
    "action_block_moe_gen_action.v_proj",
    "action_block_moe_gen_action.q_norm",
    "action_block_moe_gen_action.k_norm",
    "action_block_moe_gen_action.o_proj",
)


class Recorder:
    def __init__(self, directory, *, rank, tp_rank, tp_size, target_forward=1, max_elements=4_000_000, rows=32):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank, self.tp_rank, self.tp_size = rank, tp_rank, tp_size
        self.target_forward, self.max_elements, self.rows = target_forward, max_elements, rows
        self.sequence = -1
        self.active_layer = None
        self.attention_call = 0
        self.seen_layers = set()
        self.seen_modules = set()

    def begin(self, module, inputs):
        self.sequence += 1

    def save(self, tensor, layer, tag, kind):
        import torch

        if not isinstance(tensor, torch.Tensor):
            return
        value = tensor.detach()
        shape, stride = list(value.shape), list(value.stride())
        if value.numel() > self.max_elements and value.ndim:
            n = min(self.rows, value.shape[0])
            if value.shape[0] <= 2 * n:
                selected, scope = value, {"kind": "full"}
            else:
                selected = torch.cat((value[:n], value[-n:]), dim=0)
                scope = {"kind": "first_last_dim0", "rows_each_end": n}
        else:
            selected, scope = value, {"kind": "full"}
        selected = selected.contiguous().cpu()
        digest = hashlib.sha256(selected.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        record = {
            "forward": self.target_forward,
            "layer": layer,
            "tag": tag,
            "kind": kind,
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "dtype": str(tensor.dtype),
            "original_device": str(tensor.device),
            "shape": shape,
            "stride": stride,
            "scope": scope,
            "stored_sha256": digest,
            "value": selected,
        }
        path = self.directory / f"layer{layer:02d}.{tag}.{kind}.rank{self.rank}.pt"
        with path.open("xb") as handle:
            torch.save(record, handle)

    def watch(self, tensor, layer, tag, kind):
        import torch

        if not isinstance(tensor, torch.Tensor):
            return
        self.save(tensor, layer, tag, kind)
        if tensor.requires_grad:
            tensor.register_hook(lambda grad: self.save(grad, layer, tag, kind + "_dgrad"))

    def attach(self, root, layers):
        root.register_forward_pre_hook(self.begin)
        for index, layer in layers:
            index = int(index)
            if index not in (30, 31, 32):
                continue

            def enter(module, inputs, layer_index=index):
                identity = (self.sequence, layer_index)
                if self.sequence == self.target_forward and identity not in self.seen_layers:
                    self.seen_layers.add(identity)
                    self.active_layer, self.attention_call = layer_index, 0
                else:
                    self.active_layer = None

            def leave(module, inputs, output):
                self.active_layer = None

            layer.register_forward_pre_hook(enter)
            layer.register_forward_hook(leave)
            for name in MODULE_NAMES:
                try:
                    target = layer.get_submodule(name)
                except AttributeError:
                    continue
                tag = name.replace(".", "_")

                def hook(module, inputs, output, layer_index=index, module_tag=tag):
                    identity = (self.sequence, layer_index, module_tag)
                    if self.active_layer != layer_index or identity in self.seen_modules:
                        return
                    self.seen_modules.add(identity)
                    if inputs:
                        self.watch(inputs[0], layer_index, module_tag, "input")
                    self.watch(output, layer_index, module_tag, "output")

                target.register_forward_hook(hook)

    def attend(self, original, query, key, value, mask, **kwargs):
        import torch

        index = self.active_layer
        if index is None:
            return original(query, key, value, mask, **kwargs)
        call = self.attention_call
        self.attention_call += 1
        tag = f"attention_call{call}"
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            self.watch(tensor, index, tag, name)
        if isinstance(mask, torch.Tensor):
            self.save(mask, index, tag, "mask")
            full_mask_stored = mask.numel() <= self.max_elements or not mask.ndim or mask.shape[0] <= 2 * self.rows
        else:
            # Flex masks may be structured objects. Their metadata alone is not
            # enough to replay them: retain the real input fixture to rebuild it.
            full_mask_stored = mask is None
        metadata = {
            "forward": self.target_forward,
            "layer": index,
            "rank": self.rank,
            "mask_type": None if mask is None else f"{type(mask).__module__}.{type(mask).__qualname__}",
            "full_mask_stored": full_mask_stored,
            "kwargs": {
                key: value
                for key, value in kwargs.items()
                if value is None or isinstance(value, (bool, int, float, str))
            },
        }
        path = self.directory / f"layer{index:02d}.{tag}.mask.rank{self.rank}.json"
        with path.open("x") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        output = original(query, key, value, mask, **kwargs)
        self.watch(output, index, tag, "output")
        return output


def install(directory, *, target_forward, max_elements, rows):
    import torch.distributed as dist

    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from cosmos_framework.model.generator.mot import small_action_tower

    recorders = []
    original_stage = PipelineStage.__init__

    def initialize(stage, *args, **kwargs):
        from megatron.core import parallel_state as ps

        original_stage(stage, *args, **kwargs)
        recorder = Recorder(
            Path(directory) / "tp",
            rank=dist.get_rank(),
            tp_rank=ps.get_tensor_model_parallel_rank(),
            tp_size=ps.get_tensor_model_parallel_world_size(),
            target_forward=target_forward,
            max_elements=max_elements,
            rows=rows,
        )
        recorder.attach(stage, list(stage.layers.items()))
        recorders.append(recorder)

    PipelineStage.__init__ = initialize
    import tp_pp_gpu_leaf_reference as reference

    original_place = reference.place_gpu_trainables

    def place(model):
        handles = original_place(model)
        recorder = Recorder(
            Path(directory) / "native",
            rank=0,
            tp_rank=0,
            tp_size=1,
            target_forward=target_forward,
            max_elements=max_elements,
            rows=rows,
        )
        recorder.attach(model, list(enumerate(model.language_model.model.layers)))
        recorders.append(recorder)
        return handles

    reference.place_gpu_trainables = place
    original_attend = small_action_tower.attend

    def attend(query, key, value, mask, **kwargs):
        active = [recorder for recorder in recorders if recorder.active_layer is not None]
        if len(active) > 1:
            raise RuntimeError("Ambiguous concurrent attention trace contexts")
        return (
            active[0].attend(original_attend, query, key, value, mask, **kwargs)
            if active
            else original_attend(query, key, value, mask, **kwargs)
        )

    small_action_tower.attend = attend


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--entry", choices=("standard", "wide"), required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--forward-index", type=int, default=1)
    parser.add_argument("--max-elements", type=int, default=4_000_000)
    parser.add_argument("--rows", type=int, default=32)
    args, remainder = parser.parse_known_args()
    if args.forward_index < 0 or args.max_elements < 1 or args.rows < 1:
        parser.error("Forward index must be nonnegative and sample limits positive")
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    install(args.capture_dir, target_forward=args.forward_index, max_elements=args.max_elements, rows=args.rows)
    sys.argv = [sys.argv[0], *remainder]
    if args.entry == "wide":
        from tools.run_tp_pp_wide_linear_math import main as run
    else:
        from tools.run_tp_pp_traced_fp32_math import main as run
    run()


if __name__ == "__main__":
    main()
