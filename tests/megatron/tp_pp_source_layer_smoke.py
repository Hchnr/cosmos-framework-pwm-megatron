# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Actual Nano layer weights + nonzero source Adam state, TP versus native math."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, sequence_pack_from_packed_sequence
from cosmos_framework.model.generator.megatron_tp_pp.factory import nano_layer_factory
from cosmos_framework.model.generator.megatron_tp_pp.payload import PayloadLayout
from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
from cosmos_framework.model.generator.mot.action_tower import prepare_joint_metadata
from cosmos_framework.model.generator.mot.small_action_tower import (
    action_rope,
    action_shape,
    compact_pack,
    small_decoder_inputs,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver
from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states, initialize_adam_state
from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import stage_parameter_manifest
from cosmos_framework.training_backends.megatron_tp_pp.runtime import initialize, transformer_config


def make_inputs(count):
    u, v, a = 2 * count, 3 * count, count
    # Assemble real packed samples, not three unrelated attention tensors.
    total = 6 * count
    und = torch.tensor([6 * i + j for i in range(count) for j in range(2)], device="cuda")
    gen = torch.tensor([6 * i + j for i in range(count) for j in range(2, 6)], device="cuda")
    actions = torch.tensor([6 * i + 5 for i in range(count)], device="cuda")
    x = torch.randn(total, 4096, device="cuda") * 0.1
    pack = sequence_pack_from_packed_sequence(
        x, ["causal", "full"] * count, [2, 4] * count, [6] * count, und, gen, fixed_causal_len=8, fixed_full_len=18
    )
    prepare_joint_metadata(pack, actions, "native")
    positions = torch.arange(total, device="cuda").expand(3, -1)
    rope = tuple(from_all_seq(t, pack) for t in action_rope(positions, x.dtype, 128))
    return small_decoder_inputs(
        compact_pack(pack, rope, positions, action_shape("width_only_1b"), fixed_lengths=(8, 12, 6))
    )


def main():
    from megatron.core import parallel_state as ps
    from megatron.core.optimizer import OptimizerConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = False
    topology = ModelParallelConfig(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=1,
        global_packed_microbatches_per_update=2,
        logical_data_parallel_size=2,
        reference_micro_steps_per_update=1,
    )
    groups = initialize(topology)
    text_config = Qwen3VLTextConfig(
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
    )
    factory = nano_layer_factory(text_config)
    reference = factory(0).to_empty(device="cuda")
    reference.small_action_enabled = True
    layout = PayloadLayout(((8, 4096), (12, 4096), (6, 1280)))
    stage = PipelineStage(
        transformer_config(topology, num_layers=1, hidden_size=4096, num_attention_heads=32), factory, layout=layout
    ).to_empty(device="cuda")
    manifest = stage_parameter_manifest(stage)
    bundle = json.loads((args.bundle / "manifest.json").read_text())
    state_cache = {}
    # One layer is deliberately a partial conversion; select only its registered
    # source tensors and require complete coverage of every constructed weight.
    for name, param in reference.named_parameters():
        key = "net.language_model.model.layers.0." + name
        entry = bundle["parameters"][key]
        pieces = [torch.load(args.bundle / shard["path"], weights_only=False) for shard in entry["shards"]]
        state = {
            field: (
                pieces[0]["state"][field]
                if entry["axis"] is None
                else torch.cat([piece["state"][field] for piece in pieces], dim=entry["axis"])
            )
            for field in pieces[0]["state"]
        }
        # FP32 mathematical isolation starts from actual source masters for
        # trainable parameters, and exact source model values for frozen UND.
        with torch.no_grad():
            param.copy_(state["master_param"] if param.requires_grad else state["model"])
        if param.requires_grad:
            state_cache[key] = (state, pieces[0]["hyperparameters"])
    with torch.no_grad():
        for name, param in stage.named_parameters():
            entry = manifest[name]
            ref = dict(reference.named_parameters())[name.removeprefix("layers.0.")]
            param.copy_(entry.select(ref))
    driver = ModelParallelDriver(
        stage,
        topology,
        OptimizerConfig(
            lr=1e-4,
            weight_decay=0.05,
            adam_beta1=0.9,
            adam_beta2=0.99,
            adam_eps=1e-8,
            clip_grad=0.3,
            use_distributed_optimizer=True,
        ),
    )
    for group in driver.optimizer.param_groups:
        group["weight_decay"] = 0.05
    native = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05
    )
    for name, param in reference.named_parameters():
        if param.requires_grad:
            state, hyper = state_cache["net.language_model.model.layers.0." + name]
            native.state[param] = {
                "step": hyper["step"].float().reshape(()).cpu(),
                "exp_avg": state["exp_avg"].cuda(),
                "exp_avg_sq": state["exp_avg_sq"].cuda(),
            }
    for entry in adam_parameter_states(stage, driver.optimizer):
        spec = manifest[entry.name]
        state, hyper = state_cache[spec.canonical_name]
        initialize_adam_state(
            entry,
            step=hyper["step"].item(),
            master=spec.select(state["master_param"]).cuda(),
            exp_avg=spec.select(state["exp_avg"]).cuda(),
            exp_avg_sq=spec.select(state["exp_avg_sq"]).cuda(),
        )
    del state_cache
    errors = {}
    dp_size, dp_rank = ps.get_data_parallel_world_size(), ps.get_data_parallel_rank()
    for update in range(3):
        with sdpa_kernel(SDPBackend.MATH):
            native.zero_grad(set_to_none=True)
            batches = []
            for logical_rank in range(dp_rank, 2, dp_size):
                torch.manual_seed(900 + update * 2 + logical_rank)
                count = 1 + logical_rank
                pack = make_inputs(count)
                expected = reference(pack, None, None)[0]
                expected_loss = (expected["full_only_seq"].square().mean() + expected["action_seq"].square().mean()) * (
                    2 * count / 3 / driver.microbatches
                )
                expected_loss.backward()
                batches.append(
                    {"pack": pack, "count": count, "expected": {k: expected[k].detach() for k in layout.keys}}
                )
            for param in reference.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, group=ps.get_data_parallel_group())
                    param.grad.div_(dp_size)

            def loss_fn(output, batch):
                actual = layout.unpack(output)
                for key in actual:
                    torch.testing.assert_close(actual[key], batch["expected"][key], atol=2e-5, rtol=1e-4)
                loss = (actual["full_only_seq"].square().mean() + actual["action_seq"].square().mean()) * (
                    2 * batch["count"] / 3
                )
                return loss, {"loss": loss.detach()}

            driver.forward_backward(batches, loss_fn)
        for entry in adam_parameter_states(stage, driver.optimizer):
            ref = dict(reference.named_parameters())[entry.name.removeprefix("layers.0.")]
            grad = entry.select(entry.model_parameter.main_grad)
            expected_grad = entry.select(manifest[entry.name].select(ref.grad))
            error = (grad.double() - expected_grad.double()).norm() / expected_grad.double().norm().clamp_min(1e-20)
            errors[f"{update}/{entry.name}"] = error.item()
            assert error < 1e-4, (entry.name, error.item())
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
        success, norm, _ = driver.step()
        assert success
        torch.testing.assert_close(torch.tensor(norm, device="cuda"), expected_norm, atol=2e-6, rtol=1e-4)
        native.step()
        driver.model.start_param_sync(force_sync=True)
        for entry in adam_parameter_states(stage, driver.optimizer):
            ref = dict(reference.named_parameters())[entry.name.removeprefix("layers.0.")]
            expected_master = entry.select(manifest[entry.name].select(ref))
            torch.testing.assert_close(entry.master, expected_master, atol=2e-6, rtol=1e-4)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    entry.state[key],
                    entry.select(manifest[entry.name].select(native.state[ref][key])),
                    atol=2e-6,
                    rtol=1e-4,
                )
            assert entry.step == 1201 + update
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target = args.output.with_name(f"{args.output.stem}.rank{dist.get_rank()}.json")
    target.write_text(
        json.dumps(
            {
                "passed": True,
                "scope": "actual source layer 0, synthetic packed inputs, FP32",
                "groups": groups,
                "updates": 3,
                "source_step": 1200,
                "gradient_relative_l2": errors,
            },
            indent=2,
        )
        + "\n"
    )
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
