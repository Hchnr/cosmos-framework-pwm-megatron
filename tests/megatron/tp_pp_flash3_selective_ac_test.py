# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from tools.run_tp_pp_flash3_selective_ac import estimate_saved_bytes, install, make_policy, selected_layers, wrap_block


def observation():
    return {"forward": 0, "recompute": 0, "native_selected_layers": 0, "tp_selected_layers": 0}


def test_memory_report_estimates_observed_tp_and_unsharded_shapes():
    tp = estimate_saved_bytes(59648, 8, 128, 2, 36)
    fsdp = estimate_saved_bytes(59648, 32, 128, 2, 36)
    assert tp["output_bytes_per_layer"] == 122159104
    assert tp["estimated_bytes_per_active_microstep"] == 36 * (122159104 + 59648 * 8 * 4)
    assert fsdp["estimated_bytes_per_active_microstep"] == 4 * tp["estimated_bytes_per_active_microstep"]
    assert not tp["gpu_peak_measured"]
    with pytest.raises(ValueError):
        estimate_saved_bytes(59648, 8, 128, 0, 36)


def test_exact_schema_matching_does_not_save_other_operators():
    policy = make_policy(observation())
    ctx = SimpleNamespace(is_recompute=False)

    def fake(name, overload=""):
        return SimpleNamespace(_schema=SimpleNamespace(name=name, overload_name=overload))

    assert policy(ctx, fake("flash_attn_3::_flash_attn_forward")) == CheckpointPolicy.MUST_SAVE
    assert policy(ctx, fake("flash_attn_3::_flash_attn_backward")) == CheckpointPolicy.MUST_RECOMPUTE
    assert policy(ctx, fake("other::_flash_attn_forward")) == CheckpointPolicy.MUST_RECOMPUTE
    assert policy(ctx, fake("flash_attn_3::_flash_attn_forward", "other")) == CheckpointPolicy.MUST_RECOMPUTE
    assert policy(ctx, torch.ops.aten.mm.default) == CheckpointPolicy.MUST_RECOMPUTE


def test_saved_custom_attention_output_avoids_recompute_and_preserves_gradients():
    calls = {"forward": 0}

    @torch.library.custom_op("cosmos_sac_test::attention", mutates_args=())
    def attention(x: torch.Tensor) -> torch.Tensor:
        calls["forward"] += 1
        return x.sin()

    def setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], output)

    def backward(ctx, grad):
        x, _ = ctx.saved_tensors
        return grad * x.cos()

    attention.register_autograd(backward, setup_context=setup)

    @attention.register_fake
    def fake_attention(x):
        return torch.empty_like(x)

    observations = []
    for save in (False, True):
        calls["forward"] = 0
        obs = observation()
        policy = make_policy(obs, schema="cosmos_sac_test::attention" if save else "no::such_op")
        x = torch.tensor([0.2, -0.3], requires_grad=True)
        result = checkpoint(
            lambda value: attention(value * 2).square(),
            x,
            use_reentrant=False,
            context_fn=lambda: create_selective_checkpoint_contexts(policy),
        )
        result.sum().backward()
        observations.append((result.detach(), x.grad.detach().clone(), calls["forward"], obs))
    torch.testing.assert_close(observations[0][0], observations[1][0], rtol=0, atol=0)
    torch.testing.assert_close(observations[0][1], observations[1][1], rtol=0, atol=0)
    assert observations[0][2] == 2 and observations[1][2] == 1
    assert observations[1][3]["forward"] == 1 and observations[1][3]["recompute"] == 1

    compiled_obs = observation()
    compiled_policy = make_policy(compiled_obs, schema="cosmos_sac_test::attention")

    class CompiledBlock(torch.nn.Module):
        def forward(self, value):
            return attention(value * 2).square()

    wrapped = wrap_block(
        CompiledBlock(), SimpleNamespace(preserve_rng_state=True, determinism_check="default"), compiled_policy
    )

    compiled_x = torch.tensor([0.2, -0.3], requires_grad=True)
    calls["forward"] = 0
    compiled_result = torch.compile(wrapped, backend="aot_eager", fullgraph=True)(compiled_x)
    compiled_result.sum().backward()
    torch.testing.assert_close(compiled_result, observations[0][0], rtol=0, atol=0)
    torch.testing.assert_close(compiled_x.grad, observations[0][1], rtol=0, atol=0)
    assert calls["forward"] == 1


def test_same_last_layers_in_native_and_tp_preserve_other_options(monkeypatch):
    from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
    from cosmos_framework.model.generator.mot import parallelize_unified_mot as parallel
    from cosmos_framework.training_backends.megatron_tp_pp import model, train
    from tools import run_tp_pp_flash3_selective_ac as candidate

    observed = []
    monkeypatch.setattr(parallel, "apply_ac", parallel.apply_ac)
    monkeypatch.setattr(train, "optimize_stage", train.optimize_stage)
    monkeypatch.setattr(parallel, "_trainable_visual_blocks", lambda value: None)
    monkeypatch.setattr(parallel, "_apply_full_ac", lambda block, ac: observed.append(("full", block)) or block)
    monkeypatch.setattr(candidate, "wrap_block", lambda block, ac, policy: observed.append(("save", block)) or block)
    monkeypatch.setattr(model, "inductor_compile_options", lambda cfg: {"reorder_for_compute_comm_overlap": True})
    monkeypatch.setattr(torch, "compile", lambda block, **kwargs: observed.append(("compile", kwargs)) or block)
    config = ActivationCheckpointingConfig(mode="full")
    cfg = SimpleNamespace(enabled=True, compiled_region="language", use_cuda_graphs=False, compile_dynamic=False)

    def blocks():
        return torch.nn.ModuleDict({str(i): torch.nn.Identity() for i in range(3)})

    native, stage = SimpleNamespace(model=SimpleNamespace(layers=blocks())), SimpleNamespace(layers=blocks())
    obs = observation()
    install(2, obs)
    parallel.apply_ac(native, config)
    train.optimize_stage(stage, SimpleNamespace(activation_checkpointing=config, compile=cfg))
    assert [entry[0] for entry in observed] == [
        "full",
        "save",
        "save",
        "full",
        "compile",
        "save",
        "compile",
        "save",
        "compile",
    ]
    assert obs["native_selected_layers"] == obs["tp_selected_layers"] == 2
    assert all(
        entry[1]["options"] == {"reorder_for_compute_comm_overlap": True} for entry in observed if entry[0] == "compile"
    )


@pytest.mark.parametrize("count", [0, -1, 4])
def test_layer_count_must_fit(count):
    with pytest.raises(ValueError):
        selected_layers(("0", "1", "2"), count)
