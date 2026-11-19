# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Real TP adapters + Qwen three-stream blocks, isolated SP CPU/GPU control.

Tiny synthetic PackedSequence with two samples; production I/O, mask/RoPE,
MCore local GEMM and full AC are exercised. Synthetic squared-output loss is
not the training recipe's actual RF loss. Not installed into production.
"""

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from tools.tp_pp_sequence_parallel_real_io import reduce_partial_input_gradients, replicated_output, sharded_input
from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs, LastStageOutputs
from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear, parallelize_layer
from cosmos_framework.model.generator.mot.action128 import Action128DomainLinear
from cosmos_framework.model.generator.mot.action_tower_test import tiny_action128_spec
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder
from cosmos_framework.model.generator.mot.small_action_tower import ActionBlock, ActionShape
from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
from tools.tp_pp_sequence_parallel_real_decoder import SPBlock, TRACE
from tools.tp_pp_adaptive_qualification_data import Case, make_sequence


def create_net():
    config = Qwen3VLMoTConfig({"text_config": {
        "vocab_size": 32, "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 2, "num_attention_heads": 16, "num_key_value_heads": 4,
        "head_dim": 8, "rms_norm_eps": 1e-6,
    }})
    lm = Qwen3VLTextForCausalLM(config)
    net = Cosmos3VFMNetwork(lm, Cosmos3VFMNetworkConfig(
        vlm_config=lm.config, action_gen=True, action_dim=128, num_embodiment_domains=64,
        action128_spec=tiny_action128_spec(), tower_structure="two_tower",
    ))
    shape = ActionShape(32, 64, 16, 4, 8, (0, 1))
    net.config.tower_structure = "three_tower"
    net.config.action_tower_preset = "width_only_1b"
    net.config.action_tower_attention_backend = "sdpa"
    net.config.fixed_sequence_length_enabled = True
    net.config.fixed_und_tokens, net.config.fixed_gen_tokens, net.config.fixed_action_tokens = 8, 16, 4
    net.config.action_tower_und_short_tokens = 0
    net.action_hidden_size = shape.hidden
    net.action_modality_embed = nn.Parameter(torch.randn(shape.hidden) * 0.01)
    net.action_time_embedder = TimestepEmbedder(shape.hidden)
    net.action2llm = Action128DomainLinear(128, shape.hidden, 64)
    net.llm2action = Action128DomainLinear(shape.hidden, 128, 64)
    model = net.language_model.model
    model.small_action_shape, model.small_action_fixed_lengths = shape, None
    model.small_action_backend = "sdpa"
    model.norm_moe_gen_action = type(model.norm)(shape.hidden, eps=1e-6)
    for layer in model.layers:
        layer.small_action_enabled = True
        layer.action_block_moe_gen_action = ActionBlock(shape, type(model.norm), 1e-6, base_heads=4, base_dim=8)
    for name, param in net.named_parameters():
        if name.startswith("language_model."):
            param.requires_grad_("_moe_gen" in name)
    return net, shape


def build(rank, world, device):
    torch.manual_seed(2213)
    net, shape = create_net()
    first = FirstStageInputs(net.config, net.language_model.config, None, shape=shape, vocab_parallel=False).to_empty(device=device)
    last = LastStageOutputs(net.config, net.language_model.config, None, shape=shape, vocab_parallel=False).to_empty(device=device)
    source = net.state_dict()
    first.load_state_dict({name: source[name] for name in first.state_dict()}, strict=True)
    last.load_state_dict({name: source[name] for name in last.state_dict()}, strict=True)
    reference = nn.ModuleList([copy.deepcopy(layer).to(device) for layer in net.language_model.model.layers])
    candidate_first, candidate_last = copy.deepcopy(first), copy.deepcopy(last)
    config = TransformerConfig(
        num_layers=2, hidden_size=64, num_attention_heads=16, kv_channels=8,
        tensor_model_parallel_size=world, pipeline_model_parallel_size=1, params_dtype=torch.float32,
        pipeline_dtype=torch.float32, perform_initialization=False, use_cpu_initialization=True,
        gradient_accumulation_fusion=False, sequence_parallel=False,
    )
    candidate = nn.ModuleList([
        SPBlock(parallelize_layer(copy.deepcopy(layer).cpu(), config, rank, world), ps.get_tensor_model_parallel_group()).to(device)
        for layer in reference
    ])
    return first, reference, last, candidate_first, candidate, candidate_last


def params(first, layers, last, *, candidate=False):
    result = {}
    for name, value in first.named_parameters():
        result[f"first.{name}"] = (value, None, "first")
    for li, wrapper in enumerate(layers):
        layer = wrapper.layer if candidate else wrapper
        for name, value in layer.named_parameters():
            parent_name, _, leaf = name.rpartition(".")
            module = layer.get_submodule(parent_name) if parent_name else layer
            axis = (module.axis if leaf == "weight" else (0 if module.axis == 0 else None)) if isinstance(module, TensorParallelLinear) else None
            result[f"layers.{li}.{name}"] = (value, axis, "layer")
    for name, value in last.named_parameters():
        result[f"last.{name}"] = (value, None, "last")
    return result


def select(value, axis, rank, world):
    return value if axis is None else value.narrow(axis, rank * (value.shape[axis] // world), value.shape[axis] // world)


def optimizers(ref, cand, rank, world, device):
    rparams = [x[0] for x in ref.values() if x[0].requires_grad]
    cparams = [x[0] for x in cand.values() if x[0].requires_grad]
    ropt = torch.optim.AdamW(rparams, lr=0.001, weight_decay=0.01, foreach=False)
    copt = torch.optim.AdamW(cparams, lr=0.001, weight_decay=0.01, foreach=False)
    generator = torch.Generator(device=device).manual_seed(913)
    if ref.keys() != cand.keys():
        raise ValueError("Reference/candidate parameter names differ")
    for name, (r, _, _) in ref.items():
        c, axis, _ = cand[name]
        if r.requires_grad != c.requires_grad:
            raise ValueError(f"Parameter trainability differs for {name}")
        if not r.requires_grad:
            continue
        avg = torch.randn(r.shape, generator=generator, device=device, dtype=r.dtype) * 0.001
        sq = torch.rand(r.shape, generator=generator, device=device, dtype=r.dtype) * 0.001 + 0.01
        ropt.state[r] = {"step": torch.tensor(7.0), "exp_avg": avg, "exp_avg_sq": sq}
        copt.state[c] = {"step": torch.tensor(7.0), "exp_avg": select(avg, axis, rank, world).clone(), "exp_avg_sq": select(sq, axis, rank, world).clone()}
    return ropt, copt


def run(first, layers, last, sequence, group, *, candidate, scale_dummy=True, wrong_output=False):
    batch = {"packed_sequence": copy.deepcopy(sequence)}
    pack = first(batch)
    if candidate:
        pack = sharded_input(pack, group, scale_dummy=scale_dummy)
    context = pack.pop("pipeline_context")
    for layer in layers:
        pack = layer(pack) if candidate else layer(pack, None, None)[0]
    pack["pipeline_context"] = context
    if not batch["decoder_has_loss"]:
        pack = {**pack, **{name: pack[name].detach() for name in ("causal_seq", "full_only_seq", "action_seq")}}
    if candidate:
        pack = replicated_output(pack, group, incorrect_sum_backward=wrong_output)
    out = last(pack, batch)
    return out, batch


def synthetic_loss(out, batch):
    value = sum(t.float().square().mean() for key in ("preds_vision", "preds_action") for t in out[key])
    return value if batch["decoder_has_loss"] else value * 0


def compare(ref, cand, ropt, copt, rank, world, *, gradients):
    total = none = 0
    maximum = 0.0
    for name, (r, _, _) in ref.items():
        c, axis, _ = cand[name]
        tensors = [(select(r, axis, rank, world), c)]
        if gradients:
            if (r.grad is None) != (c.grad is None):
                raise AssertionError(f"Gradient presence mismatch {name}")
            if r.grad is None:
                none += 1
            else:
                tensors.append((select(r.grad, axis, rank, world), c.grad))
        elif r.requires_grad:
            for key in ("step", "exp_avg", "exp_avg_sq"):
                tensors.append((select(ropt.state[r][key], axis, rank, world) if key != "step" else ropt.state[r][key], copt.state[c][key]))
        for expected, actual in tensors:
            if expected.numel():
                maximum = max(maximum, (expected - actual).abs().max().item())
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5, msg=name)
            total += 1
    return {"compared": total, "none": none, "max_abs": maximum}


def sync_layer_grads(cand, group):
    counts = {"head_norm": 0, "sequence_norm": 0, "other": 0}
    for name, (param, axis, section) in sorted(cand.items()):
        if section != "layer" or axis is not None or not param.requires_grad:
            continue
        flags = [None] * dist.get_world_size(group)
        dist.all_gather_object(flags, param.grad is not None, group=group)
        if len(set(flags)) != 1:
            raise ValueError(f"Rank-dependent gradient presence: {name}")
        if not flags[0]:
            continue
        kind = "head_norm" if ".q_norm" in name or ".k_norm" in name else "sequence_norm" if "layernorm" in name else "other"
        dist.all_reduce(param.grad, group=group)
        counts[kind] += 1
    return counts


def make_case(update, micro):
    config = [(True, True, True), (False, True, False), (True, True, False), (False, False, False)][update]
    action, vision_target, action_target = config
    return make_sequence(Case(f"u{update}m{micro}", (2, 2), action=action, noisy_vision=vision_target, noisy_action=action_target), seed=4400 + 10 * update + micro)


def negative_controls(first, ref_run, last, cfirst, cand_run, clast, ropt, copt, group, world, device):
    # For the no-target dummy, disabling the replicated-context scaling makes
    # the subsequent input-adapter SUM multiply its gradient by TP size.
    ropt.zero_grad(set_to_none=True)
    copt.zero_grad(set_to_none=True)
    seq = make_case(3, 0)
    if device.type == "cuda":
        seq.to_cuda()
    expected, _ = run(first, ref_run, last, seq, group, candidate=False)
    actual, _ = run(cfirst, cand_run, clast, seq, group, candidate=True, scale_dummy=False)
    sum(x.float().square().sum() for x in expected["preds_action"]).backward()
    sum(x.float().square().sum() for x in actual["preds_action"]).backward()
    reduce_partial_input_gradients(cfirst, group)
    reference = first.action_modality_embed.grad
    observed = cfirst.action_modality_embed.grad
    if reference is None or reference.norm().item() == 0:
        raise AssertionError("Dummy negative control must have a nonzero gradient")
    torch.testing.assert_close(observed, world * reference, atol=2e-6, rtol=2e-5)
    dummy_norm = reference.norm().item()

    # Dividing the replicated loss by TP compensates the wrong SUM gather for
    # decoder inputs, but incorrectly divides replicated output-head gradients.
    ropt.zero_grad(set_to_none=True)
    copt.zero_grad(set_to_none=True)
    seq = make_case(0, 0)
    if device.type == "cuda":
        seq.to_cuda()
    expected, _ = run(first, ref_run, last, seq, group, candidate=False)
    actual, _ = run(cfirst, cand_run, clast, seq, group, candidate=True, wrong_output=True)
    sum(x.float().square().mean() for x in expected["preds_vision"]).backward()
    (sum(x.float().square().mean() for x in actual["preds_vision"]) / world).backward()
    reference = last.llm2vae.weight.grad
    observed = clast.llm2vae.weight.grad
    if reference is None or reference.norm().item() == 0:
        raise AssertionError("Output-head negative control must have a nonzero gradient")
    torch.testing.assert_close(observed * world, reference, atol=2e-6, rtol=2e-5)
    return {"dummy_gradient_factor": world, "dummy_reference_norm": dummy_norm,
            "replicated_output_gradient_factor": 1 / world, "output_reference_norm": reference.norm().item()}


class CollectiveRecorder:
    """Record Python distributed API order/shapes, not backend-internal calls."""
    def __init__(self):
        self.events = []
        self.active = False
        for name in ("all_gather", "all_gather_into_tensor", "reduce_scatter_tensor", "all_reduce"):
            original = getattr(dist, name)

            def wrapped(*args, _name=name, _original=original, **kwargs):
                if self.active:
                    def describe(value):
                        if isinstance(value, torch.Tensor):
                            return [list(value.shape), str(value.dtype)]
                        if isinstance(value, list):
                            return [describe(item) for item in value]
                        return type(value).__name__
                    self.events.append([_name, [describe(value) for value in args]])
                return _original(*args, **kwargs)
            setattr(dist, name, wrapped)

    def check(self, group, world):
        self.active = False
        copies = [None] * world
        dist.all_gather_object(copies, self.events, group=group)
        if any(item != copies[0] for item in copies):
            raise ValueError("Distributed API collective order/shapes differ across ranks")
        return list(self.events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=180))
    ps.initialize_model_parallel(tensor_model_parallel_size=world, pipeline_model_parallel_size=1, order="tp-cp-ep-dp-pp")
    try:
        first, layers, last, cfirst, clayers, clast = build(rank, world, device)
        ref, cand = params(first, layers, last), params(cfirst, clayers, clast, candidate=True)
        ropt, copt = optimizers(ref, cand, rank, world, device)
        ref_run = [checkpoint_wrapper(layer, preserve_rng_state=True) for layer in layers]
        cand_run = [checkpoint_wrapper(layer, preserve_rng_state=True) for layer in clayers]
        group = ps.get_tensor_model_parallel_group()
        negatives = negative_controls(first, ref_run, last, cfirst, cand_run, clast, ropt, copt, group, world, device)
        recorder = CollectiveRecorder()
        records = []
        for update in range(4):
            TRACE.clear()
            recorder.events.clear()
            recorder.active = True
            ropt.zero_grad(set_to_none=True)
            copt.zero_grad(set_to_none=True)
            for micro in range(2):
                sequence = make_case(update, micro)
                if device.type == "cuda":
                    sequence.to_cuda()
                expected, rb = run(first, ref_run, last, sequence, group, candidate=False)
                actual, cb = run(cfirst, cand_run, clast, sequence, group, candidate=True)
                if rb["decoder_has_loss"] != cb["decoder_has_loss"]:
                    raise AssertionError("Decoder target presence differs")
                for key in ("preds_vision", "preds_action"):
                    if len(expected[key]) != len(actual[key]):
                        raise AssertionError(f"Output list length mismatch: {key}")
                    for left, right in zip(expected[key], actual[key], strict=True):
                        torch.testing.assert_close(right, left, atol=2e-6, rtol=2e-5)
                rloss, closs = synthetic_loss(expected, rb) / 2, synthetic_loss(actual, cb) / 2
                torch.testing.assert_close(closs, rloss, atol=2e-6, rtol=2e-5)
                rloss.backward()
                closs.backward()
            traces = [None] * world
            dist.all_gather_object(traces, TRACE, group=group)
            if any(trace != traces[0] for trace in traces):
                raise ValueError("SP collective order/shapes differ across ranks")
            reduce_partial_input_gradients(cfirst, group)
            counts = sync_layer_grads(cand, group)
            all_collectives = recorder.check(group, world)
            raw = compare(ref, cand, ropt, copt, rank, world, gradients=True)
            if update == 3:
                if any(value.grad is not None for value, _, section in cand.values() if section == "layer"):
                    raise AssertionError("No-target decoder gradients must stay None")
            ropt.step()
            copt.step()
            state = compare(ref, cand, ropt, copt, rank, world, gradients=False)
            records.append({"update": update, "no_target": update == 3, "decoder_has_loss": rb["decoder_has_loss"], "gradients": raw, "optimizer": state, "layer_sync_once_counts": counts, "collective_trace": list(TRACE), "distributed_api_trace": all_collectives})
        args.output.mkdir(parents=True, exist_ok=True)
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        (args.output / f"rank{rank}.json").write_text(json.dumps({"passed": True, "rank": rank, "world": world, "backend": args.backend, "full_ac": True, "real_io": True, "real_decoder": True, "synthetic_loss": True, "production_acceptance": False, "cuda_peak_allocated_bytes": peak, "negative_controls": negatives, "steps": records}, indent=2) + "\n")
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
