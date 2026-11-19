"""Compare the real three-stream MCore layer with grouped row sums on CPU."""

import argparse
import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def tensor_diff(a, b):
    if a.shape != b.shape:
        raise ValueError("Compared tensors must have identical shapes")
    if a.numel() == 0:
        return {"relative_l2": 0.0, "max_abs": 0.0}
    delta = a.detach().to(torch.float64) - b.detach().to(torch.float64)
    denom = a.detach().to(torch.float64).norm().item()
    return {"relative_l2": delta.norm().item() / max(denom, 1e-30), "max_abs": delta.abs().max().item()}


from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.utils.checkpoint import checkpoint

from tools.tp_pp_full_head_attention_experiment import install as install_attention
from tools.tp_pp_full_width_row_diagnostic import full_row
from tools.tp_pp_native_width_column_diagnostic import native_width_column_linear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.worktree / "tests/megatron"))
    from tp_pp_network_io_test import make_native, make_sequence

    from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs
    from cosmos_framework.model.generator.megatron_tp_pp.layers import parallelize_layer
    from cosmos_framework.model.generator.mot import small_action_tower as tower

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (2, 4):
        raise ValueError("Only TP2 and TP4 are covered")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world, pipeline_model_parallel_size=1)
    torch.manual_seed(521)
    native, shape = make_native()
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=world,
        pipeline_model_parallel_size=1,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
    )
    first = FirstStageInputs(
        native.config,
        native.language_model.config,
        None,
        shape=shape,
        vocab_parallel=False,
    )
    first.to_empty(device="cpu")
    source = native.state_dict()
    first.load_state_dict({key: source[key] for key in first.state_dict()}, strict=True)
    source_layer = copy.deepcopy(native.language_model.model.layers[0])
    if world == 4:
        # The stock tiny fixture has only two KV heads. Increase to four to
        # exercise a valid TP4 shape without changing its three-stream logic.
        source_layer.self_attn.num_key_value_heads = 4
        source_layer.self_attn.num_key_value_groups = 1
        for suffix in ("", "_moe_gen"):
            for name in ("k_proj", "v_proj"):
                old = getattr(source_layer.self_attn, name + suffix)
                replacement = torch.nn.Linear(old.in_features, 64, bias=old.bias is not None)
                replacement.weight.requires_grad_(old.weight.requires_grad)
                setattr(source_layer.self_attn, name + suffix, replacement)
        block = source_layer.action_block_moe_gen_action
        block.shape = dataclasses.replace(block.shape, kv_heads=4)
        for name in ("k_proj", "v_proj"):
            old = getattr(block, name)
            replacement = torch.nn.Linear(old.in_features, 64, bias=old.bias is not None)
            replacement.weight.requires_grad_(old.weight.requires_grad)
            setattr(block, name, replacement)
    baseline = source_layer
    candidate = parallelize_layer(copy.deepcopy(source_layer), config, rank, world)
    from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear

    original = TensorParallelLinear.forward

    def combined(self, inputs):
        if self.bias is not None:
            raise ValueError("Only bias-free layers supported")
        if self.column:
            return native_width_column_linear(inputs, self.weight, self._impl.tp_group)
        return full_row(inputs, self.weight, self._impl.tp_group)

    TensorParallelLinear.forward = combined
    install_attention()
    oldopt = torch.optim.AdamW((p for p in baseline.parameters() if p.requires_grad), lr=0.001, weight_decay=0.01)
    newopt = torch.optim.AdamW((p for p in candidate.parameters() if p.requires_grad), lr=0.001, weight_decay=0.01)
    reports = []
    source_params = dict(baseline.named_parameters())
    target_params = dict(candidate.named_parameters())
    assert source_params.keys() == target_params.keys()

    def local_value(name, value):
        parent, _, leaf = name.rpartition(".")
        module = candidate.get_submodule(parent)
        if isinstance(module, TensorParallelLinear) and leaf == "weight":
            return value.chunk(world, dim=module.axis)[rank]
        return value

    for step, (has_action, selected) in enumerate(
        (
            (True, ("full_only_seq", "action_seq")),
            (True, ("full_only_seq",)),
            (True, ("action_seq",)),
            (False, ("full_only_seq",)),
        )
    ):
        torch.manual_seed(9800 + step)
        packed = first({"packed_sequence": make_sequence(action=has_action)})
        old, new = dict(packed), dict(packed)
        for name in ("causal_seq", "full_only_seq", "action_seq"):
            source = packed[name].detach()
            old[name] = source.clone().requires_grad_(name != "causal_seq")
            new[name] = source.clone().requires_grad_(name != "causal_seq")
        oldopt.zero_grad(set_to_none=True)
        newopt.zero_grad(set_to_none=True)

        def call(layer, pack):
            return tower.small_layer_forward(layer, pack)[0]

        rng_before = torch.random.get_rng_state()
        if os.environ.get("COMBINED_AC") == "1":
            expected = checkpoint(lambda pack: call(baseline, pack), old, use_reentrant=False)
            expected_rng = torch.random.get_rng_state()
            torch.random.set_rng_state(rng_before)
            actual = checkpoint(lambda pack: call(candidate, pack), new, use_reentrant=False)
        else:
            expected = call(baseline, old)
            expected_rng = torch.random.get_rng_state()
            torch.random.set_rng_state(rng_before)
            actual = call(candidate, new)
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        sum(expected[name].square().sum() for name in selected).backward()
        sum(actual[name].square().sum() for name in selected).backward()
        errors = []
        for name in selected:
            errors.append(tensor_diff(expected[name], actual[name]))
        for name in ("full_only_seq", "action_seq"):
            if old[name].grad is None or new[name].grad is None:
                assert old[name].grad is None and new[name].grad is None, (step, name)
            else:
                errors.append(tensor_diff(old[name].grad, new[name].grad))
        for name, q in target_params.items():
            p = source_params[name]
            if q.grad is not None and getattr(q, "_cosmos_tp_partial_grad", False):
                dist.all_reduce(q.grad, group=parallel_state.get_tensor_model_parallel_group())
            if p.grad is None or q.grad is None:
                assert p.grad is None and q.grad is None, (step, name)
            else:
                errors.append(tensor_diff(local_value(name, p.grad), q.grad))
        oldopt.step()
        newopt.step()
        for name, q in target_params.items():
            p = source_params[name]
            errors.append(tensor_diff(local_value(name, p), q))
            if p in oldopt.state or q in newopt.state:
                assert set(oldopt.state[p]) == set(newopt.state[q]), name
                for key in oldopt.state[p]:
                    v = oldopt.state[p][key]
                    errors.append(tensor_diff(local_value(name, v) if v.ndim else v, newopt.state[q][key]))
        maximum = max(e["relative_l2"] for e in errors)
        assert maximum < 1e-4, (step, maximum)
        reports.append({"step": step, "has_action": has_action, "selected": selected, "max_relative_l2": maximum})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps({"rank": rank, "cases": reports}) + "\n")
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
