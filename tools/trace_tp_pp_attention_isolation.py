# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in attention/QK-norm trace for one real forward; never an acceptance test.

The wrapper instruments native or TP eager execution without editing model code.
It records selected layer module/attention tensors and cotangents, on every TP
rank, for forward index 1 and layers 30--32 by default. Full tensors up to the element limit are
saved; larger tensors retain the first/last rows along dimension 0 when that
axis is long enough, otherwise the full tensor is kept. Each record
contains its original dtype, shape, strides, sample scope and SHA256 of the
stored bytes. Input hooks see the accumulated gradient if a tensor is shared
by branches; use attention output cotangents for an isolated operator replay.

The optional branch mode inserts identity autograd edges before the video MLP
normalization and its gate/up consumers. It records individual input cotangents;
use layer-output cotangents from the layer trace for the residual shortcut, and
account for masks. This changes the autograd graph and hooks remain attached for
the module lifetime. A separate same-configuration output/gradient/RNG test and
``verify_branch_capture`` coverage check are required before interpreting it;
no real-GPU invariance is claimed by this tool.
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
    def __init__(
        self,
        directory,
        *,
        rank,
        tp_rank,
        tp_size,
        target_forward=1,
        max_elements=4_000_000,
        rows=32,
        layers=(30, 31, 32),
        capture_branch_gradients=False,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (
            not layers
            or len(set(layers)) != len(layers)
            or any(not isinstance(index, int) or index < 0 for index in layers)
        ):
            raise ValueError("Layer indices must be nonnegative and unique")
        self.layers = frozenset(layers)
        self.rank, self.tp_rank, self.tp_size = rank, tp_rank, tp_size
        self.target_forward, self.max_elements, self.rows = target_forward, max_elements, rows
        self.sequence = -1
        self.active_layer = None
        self.attention_call = 0
        self.seen_layers = set()
        self.seen_modules = set()
        self.seen_edges = set()
        self.capture_branch_gradients = capture_branch_gradients
        self.event_order = 0

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
            "event_order": self.event_order,
            "value": selected,
        }
        self.event_order += 1
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

    def branch_input(self, tensor, layer, tag):
        """Insert an opt-in identity edge to observe this consumer's cotangent.

        This adds an autograd node and is diagnostic only; numerical invariance
        must be verified for the exact model and execution configuration.
        """
        import torch

        recorder = self

        class EdgeTap(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value

            @staticmethod
            def backward(ctx, gradient):
                recorder.save(gradient, layer, tag, "branch_input_dgrad")
                return gradient

        return EdgeTap.apply(tensor)

    def attach(self, root, layers):
        root.register_forward_pre_hook(self.begin)
        for index, layer in layers:
            index = int(index)
            if index not in self.layers:
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
                if self.capture_branch_gradients and name in (
                    "post_attention_layernorm_moe_gen",
                    "mlp_moe_gen.gate_proj",
                    "mlp_moe_gen.up_proj",
                ):

                    def edge_hook(module, inputs, layer_index=index, module_tag=tag):
                        import torch

                        identity = (self.sequence, layer_index, module_tag)
                        if self.active_layer != layer_index or identity in self.seen_edges:
                            return None
                        if (
                            not torch.is_grad_enabled()
                            or not inputs
                            or not isinstance(inputs[0], torch.Tensor)
                            or not inputs[0].requires_grad
                        ):
                            raise RuntimeError("Branch diagnostic requires a differentiable tensor input")
                        self.seen_edges.add(identity)
                        return (self.branch_input(inputs[0], layer_index, module_tag), *inputs[1:])

                    target.register_forward_pre_hook(edge_hook)

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


def verify_branch_capture(directory, *, kind, forwards, layers, ranks):
    """Fail closed unless all explicitly expected opt-in branch tensors exist.

    Call after all backward work; coverage alone does not establish numerical
    non-perturbation or production correctness. Handles live for module lifetime.
    """
    import torch

    if kind not in ("native", "tp") or any(
        not values or len(values) != len(set(values)) or any(type(value) is not int or value < 0 for value in values)
        for values in (forwards, layers, ranks)
    ):
        raise ValueError("Explicit, unique branch coverage coordinates required")
    tags = ("post_attention_layernorm_moe_gen", "mlp_moe_gen_gate_proj", "mlp_moe_gen_up_proj")
    expected = {
        Path(directory) / f"forward{forward:03d}" / kind / f"layer{layer:02d}.{tag}.branch_input_dgrad.rank{rank}.pt": (
            forward,
            layer,
            tag,
            rank,
        )
        for forward in forwards
        for layer in layers
        for rank in ranks
        for tag in tags
    }
    found = {
        path
        for forward in forwards
        for path in (Path(directory) / f"forward{forward:03d}" / kind).glob("*.branch_input_dgrad.rank*.pt")
    }
    if found != set(expected):
        raise ValueError("Branch capture coverage missing or extra files")
    sampled_files = []
    for path, coordinates in expected.items():
        item = torch.load(path, map_location="cpu", weights_only=True)
        value = item["value"]
        if (
            item.get("kind") != "branch_input_dgrad"
            or (item.get("forward"), item.get("layer"), item.get("tag"), item.get("rank")) != coordinates
            or not isinstance(value, torch.Tensor)
            or hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            != item.get("stored_sha256")
        ):
            raise ValueError(f"Invalid branch capture payload: {path}")
        scope = item.get("scope")
        if not isinstance(scope, dict) or scope.get("kind") not in ("full", "first_last_dim0"):
            raise ValueError(f"Invalid branch capture scope: {path}")
        if scope["kind"] == "full":
            if list(value.shape) != item.get("shape"):
                raise ValueError(f"Invalid full branch tensor shape: {path}")
        else:
            sampled_files.append(str(path))
    return {
        "coverage_passed": True,
        "expected_files": len(expected),
        "full_tensor_coverage": not sampled_files,
        "sampled_files": sorted(sampled_files),
        "numerical_invariance_verified": False,
    }


def attach_recorders(
    root,
    layers,
    directory,
    kind,
    *,
    rank,
    tp_rank,
    tp_size,
    target_forwards,
    split_by_forward,
    max_elements,
    rows,
    selected_layers,
    capture_branch_gradients=False,
):
    recorders = []
    for target_forward in target_forwards:
        base = Path(directory) / f"forward{target_forward:03d}" if split_by_forward else Path(directory)
        recorder = Recorder(
            base / kind,
            rank=rank,
            tp_rank=tp_rank,
            tp_size=tp_size,
            target_forward=target_forward,
            max_elements=max_elements,
            rows=rows,
            layers=selected_layers,
            capture_branch_gradients=capture_branch_gradients,
        )
        recorder.attach(root, layers)
        recorders.append(recorder)
    return recorders


def install(
    directory,
    *,
    target_forwards,
    split_by_forward,
    max_elements,
    rows,
    layers=(30, 31, 32),
    capture_branch_gradients=False,
):
    import torch.distributed as dist

    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from cosmos_framework.model.generator.mot import small_action_tower

    recorders = []
    original_stage = PipelineStage.__init__

    def initialize(stage, *args, **kwargs):
        from megatron.core import parallel_state as ps

        original_stage(stage, *args, **kwargs)
        recorders.extend(
            attach_recorders(
                stage,
                list(stage.layers.items()),
                directory,
                "tp",
                rank=dist.get_rank(),
                tp_rank=ps.get_tensor_model_parallel_rank(),
                tp_size=ps.get_tensor_model_parallel_world_size(),
                target_forwards=target_forwards,
                split_by_forward=split_by_forward,
                max_elements=max_elements,
                rows=rows,
                selected_layers=layers,
                capture_branch_gradients=capture_branch_gradients,
            )
        )

    PipelineStage.__init__ = initialize
    import tp_pp_gpu_leaf_reference as reference

    original_place = reference.place_gpu_trainables

    def place(model):
        handles = original_place(model)
        recorders.extend(
            attach_recorders(
                model,
                list(enumerate(model.language_model.model.layers)),
                directory,
                "native",
                rank=0,
                tp_rank=0,
                tp_size=1,
                target_forwards=target_forwards,
                split_by_forward=split_by_forward,
                max_elements=max_elements,
                rows=rows,
                selected_layers=layers,
                capture_branch_gradients=capture_branch_gradients,
            )
        )
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
    forward = parser.add_mutually_exclusive_group()
    forward.add_argument("--forward-index", type=int, default=None)
    forward.add_argument("--forward-indices", help="Comma-separated forward indices, captured in separate directories")
    parser.add_argument("--max-elements", type=int, default=4_000_000)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--layers", default="30,31,32", help="Comma-separated global layer indices")
    parser.add_argument(
        "--capture-branch-gradients",
        action="store_true",
        help="Diagnostic identity edges; requires separate invariance verification",
    )
    args, remainder = parser.parse_known_args()
    if args.max_elements < 1 or args.rows < 1:
        parser.error("Sample limits must be positive")
    try:
        forwards = (
            tuple(int(index) for index in args.forward_indices.split(","))
            if args.forward_indices is not None
            else (1 if args.forward_index is None else args.forward_index,)
        )
    except ValueError:
        parser.error("Forward indices must be comma-separated integers")
    if not forwards or len(set(forwards)) != len(forwards) or any(index < 0 for index in forwards):
        parser.error("Forward indices must be nonnegative and unique")
    try:
        layers = tuple(int(index) for index in args.layers.split(","))
    except ValueError:
        parser.error("Layers must be comma-separated integer indices")
    if not layers or len(set(layers)) != len(layers) or any(index < 0 for index in layers):
        parser.error("Layer indices must be nonnegative and unique")
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    install(
        args.capture_dir,
        target_forwards=forwards,
        split_by_forward=args.forward_indices is not None or args.capture_branch_gradients,
        max_elements=args.max_elements,
        rows=args.rows,
        layers=layers,
        capture_branch_gradients=args.capture_branch_gradients,
    )
    sys.argv = [sys.argv[0], *remainder]
    if args.entry == "wide":
        from tools.run_tp_pp_wide_linear_math import main as run
    else:
        from tools.run_tp_pp_traced_fp32_math import main as run
    run()


if __name__ == "__main__":
    main()
