"""Production input/output adapters versus the native packed network."""

import copy

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs, LastStageOutputs
from cosmos_framework.model.generator.megatron_tp_pp.payload import STREAMS, PayloadLayout
from cosmos_framework.model.generator.mot.action128 import Action128DomainLinear
from cosmos_framework.model.generator.mot.action_tower_test import tiny_net
from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder
from cosmos_framework.model.generator.mot.small_action_tower import ActionBlock, ActionShape


def make_native():
    net = tiny_net("two_tower", num_hidden_layers=2)
    shape = ActionShape(32, 64, 4, 2, 16, (0, 1))
    net.config.tower_structure = "three_tower"
    net.config.action_tower_preset = "width_only_1b"
    net.config.action_tower_attention_backend = "sdpa"
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
        layer.action_block_moe_gen_action = ActionBlock(shape, type(model.norm), 1e-6, base_heads=2, base_dim=16)
    for name, param in net.named_parameters():
        if name.startswith("language_model."):
            param.requires_grad_("_moe_gen" in name)
    return net, shape


def make_sequence(*, action=True, noisy_vision=True, noisy_action=True):
    total = 6 if action else 4
    vision = ModalityData(
        tokens=[torch.randn(1, 16, 2, 2, 2)],
        token_shapes=[(2, 1, 1)],
        sequence_indexes=torch.tensor([2, 3]),
        mse_loss_indexes=torch.tensor([2, 3] if noisy_vision else [], dtype=torch.long),
        timesteps=torch.tensor([0.4, 0.4] if noisy_vision else []),
        noisy_frame_indexes=[torch.tensor([0, 1] if noisy_vision else [], dtype=torch.long)],
        condition_mask=[torch.full((2, 1, 1), 0.0 if noisy_vision else 1.0)],
    )
    action_data = None
    if action:
        action_data = ModalityData(
            tokens=[torch.randn(2, 128)],
            token_shapes=[(2,)],
            sequence_indexes=torch.tensor([4, 5]),
            mse_loss_indexes=torch.tensor([4, 5] if noisy_action else [], dtype=torch.long),
            domain_id=[torch.tensor([0])],
            timesteps=torch.tensor([0.6, 0.6] if noisy_action else []),
            noisy_frame_indexes=[torch.tensor([0, 1] if noisy_action else [], dtype=torch.long)],
            condition_mask=[torch.full((2, 1), 0.0 if noisy_action else 1.0)],
        )
    return PackedSequence(
        sample_lens=[total],
        split_lens=[2, total - 2],
        attn_modes=["causal", "full"],
        sequence_length=total,
        text_ids=torch.tensor([1, 2]),
        text_indexes=torch.arange(2),
        position_ids=torch.arange(total).expand(3, -1),
        vision=vision,
        action=action_data,
    )


def test_first_input_uses_full_und_warmup_then_native_short_bucket():
    native, shape = make_native()
    config = native.config
    config.fixed_sequence_length_enabled = True
    config.fixed_und_tokens, config.fixed_gen_tokens, config.fixed_action_tokens = 8, 8, 4
    config.action_tower_und_short_tokens = 4
    first = FirstStageInputs(config, native.language_model.config, None, shape=shape, vocab_parallel=False).to_empty(
        device="cpu"
    )
    source = native.state_dict()
    first.load_state_dict({key: source[key] for key in first.state_dict()}, strict=True)
    first._warm_full_und = True
    with torch.no_grad():
        full = first({"packed_sequence": make_sequence()})
        short = first({"packed_sequence": make_sequence()})
    assert full["causal_seq"].shape == (8, 64)
    assert short["causal_seq"].shape == (4, 64)
    assert full["full_only_seq"].shape == short["full_only_seq"].shape


@pytest.mark.parametrize(
    "action,noisy_vision,noisy_action",
    [(True, True, True), (True, False, True), (True, True, False), (False, True, False), (False, False, False)],
)
def test_production_io_reuses_native_predictions_and_graph(action, noisy_vision, noisy_action):
    torch.manual_seed(130)
    native, shape = make_native()
    first = FirstStageInputs(
        native.config, native.language_model.config, None, shape=shape, vocab_parallel=False
    ).to_empty(device="cpu")
    last = LastStageOutputs(
        native.config, native.language_model.config, None, shape=shape, vocab_parallel=False
    ).to_empty(device="cpu")
    source = native.state_dict()
    first.load_state_dict({k: source[k] for k in first.state_dict()}, strict=True)
    last.load_state_dict({k: source[k] for k in last.state_dict()}, strict=True)
    layers = copy.deepcopy(native.language_model.model.layers)
    sequence = make_sequence(action=action, noisy_vision=noisy_vision, noisy_action=noisy_action)
    expected = native(copy.deepcopy(sequence))
    batch = {"packed_sequence": sequence}
    pack = first(batch)
    layout = PayloadLayout(tuple(tuple(pack[k].shape) for k in STREAMS), context_width=96)
    # Exercise a differentiable single-tensor boundary including the dummy context.
    pack = {**pack, **layout.unpack(layout.pack(pack), frozen_und=True)}
    context = pack.pop("pipeline_context")
    for layer in layers:
        pack = layer(pack, None, None)[0]
    pack["pipeline_context"] = context
    pack = layout.unpack(layout.pack(pack), frozen_und=True)
    actual = last(pack, batch)
    for key in ("preds_vision", "preds_action"):
        for value, ref in zip(actual[key], expected[key], strict=True):
            torch.testing.assert_close(value, ref, atol=2e-6, rtol=2e-5)
    # Dummy heads must retain the exact native graph and parameter ownership.
    expected_loss = sum(t.square().mean() for k in ("preds_vision", "preds_action") for t in expected[k])
    actual_loss = sum(t.square().mean() for k in ("preds_vision", "preds_action") for t in actual[k])
    actual_loss.backward()
    expected_loss.backward()
    all_params = dict(first.named_parameters()) | dict(last.named_parameters())
    all_params.update({"language_model.model.layers." + k: p for k, p in layers.named_parameters()})
    for name, ref in native.named_parameters():
        param = all_params[name]
        if ref.grad is None:
            assert param.grad is None, name
        else:
            torch.testing.assert_close(param.grad, ref.grad, atol=2e-6, rtol=2e-5, msg=name)
    assert not hasattr(first, "llm2vae") and not hasattr(last, "vae2llm")
    assert not hasattr(first.language_model.model, "layers")
