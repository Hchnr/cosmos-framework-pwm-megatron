# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.projection_routing import (
    NUM_PROJECTION_SLOTS,
    PROJECTION_HEAD_IDS,
    PROJECTION_ROUTING_VERSION,
)
from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.generator.mot.action_tower import (
    defer_copy_initialization,
    finish_copy_initialization,
    prepare_joint_metadata,
    resolve_execution,
    routed_gen,
    source_key,
    validate_checkpoint_initialization,
)
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM


def tiny_action128_spec():
    """Published-contract shape with identity transforms for model-only tests."""
    return {
        "registry": {
            "version": "action128-v3",
            "domains": ["franka_panda__parallel_gripper/eef20/test"],
            "embodiments": ["franka_panda__parallel_gripper"],
            "domain_embodiments": [0],
            "domain_projection_ids": [0],
        },
        "projection_routing": {
            "version": PROJECTION_ROUTING_VERSION,
            "num_slots": NUM_PROJECTION_SLOTS,
            "keys": dict(PROJECTION_HEAD_IDS),
        },
        "transforms": [
            {
                "normalization_mask": [False] * 128,
                "scale": [1.0] * 128,
                "bias": [0.0] * 128,
                "clip_min": -1.0,
                "clip_max": 1.0,
            }
        ],
        "source_digests": ["synthetic-model-test"],
    }


def tiny_net(towers, num_hidden_layers=1, head_dim=16):
    config = Qwen3VLMoTConfig(
        {
            "text_config": {
                "vocab_size": 32,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": head_dim,
                "rms_norm_eps": 1e-6,
            }
        }
    )
    lm = Qwen3VLTextForCausalLM(config)
    return Cosmos3VFMNetwork(
        lm,
        Cosmos3VFMNetworkConfig(
            vlm_config=lm.config,
            action_gen=True,
            action_dim=128,
            num_embodiment_domains=64,
            action128_spec=tiny_action128_spec(),
            tower_structure=towers,
        ),
    )


def reference_two_pass(qp, kp, vp, *args, **kwargs):
    """Independent dense reference for the original two attention calls."""
    qu, qg = get_und_seq(qp), get_gen_seq(qp)
    ku, kg = get_und_seq(kp), get_gen_seq(kp)
    vu, vg = get_und_seq(vp), get_gen_seq(vp)
    us = qp["_causal_sample_ids"]
    gs = qp["_full_only_sample_ids"]
    ui = torch.arange(len(us))
    umask = (us[:, None] == us[None, :]) & (ui[:, None] >= ui[None, :])
    gmask = gs[:, None] == torch.cat([us, gs])[None, :]

    def run(q, k, v, mask):
        return (
            F.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                k.transpose(0, 1)[None],
                v.transpose(0, 1)[None],
                attn_mask=mask,
                enable_gqa=True,
            )[0]
            .transpose(0, 1)
            .flatten(1)
        )

    return from_und_gen_splits(
        run(qu, ku, vu, umask), run(qg, torch.cat([ku, kg]), torch.cat([vu, vg]), gmask), qp
    ), None


@pytest.mark.parametrize("action_indexes", [[], [5, 6, 10], [2, 3, 4, 5, 6, 8, 9, 10]])
@pytest.mark.parametrize("execution", ["joint", "native"])
@pytest.mark.parametrize("training", [True, False])
def test_layer_parity_and_independent_gradients(action_indexes, execution, training):
    torch.manual_seed(42)
    base, new = tiny_net("two_tower"), tiny_net("three_tower")
    state = base.state_dict()
    migrated = {k: state[source_key(k) or k].clone() for k in new.state_dict()}
    new.load_state_dict(migrated, strict=True)
    old_layer = base.language_model.model.layers[0]
    old_layer.self_attn.dispatch_attention_fn = reference_two_pass
    new.language_model.model.layers[0].self_attn.dispatch_attention_fn = reference_two_pass
    x = torch.randn(11, 64)
    pack = sequence_pack_from_packed_sequence(
        x,
        ["causal", "full", "causal", "full"],
        [2, 5, 1, 3],
        [7, 4],
        torch.tensor([0, 1, 7]),
        torch.tensor([2, 3, 4, 5, 6, 8, 9, 10]),
    )
    # Identity RoPE isolates projection/mask/FFN equivalence.
    u, g = get_und_seq(pack), get_gen_seq(pack)
    cos = from_und_gen_splits(torch.ones(len(u), 16), torch.ones(len(g), 16), pack)
    sin = from_und_gen_splits(torch.zeros(len(u), 16), torch.zeros(len(g), 16), pack)
    expected, _, _ = old_layer(pack, None, (cos, sin))
    prepare_joint_metadata(pack, torch.tensor(action_indexes, dtype=torch.long), execution)
    new.train(training)
    seen = set()

    def check_rows(name):
        def hook(module, args):
            key = "_action_indices" if "_moe_gen_action" in name else "_video_indices"
            assert args[0].shape[0] == len(pack[key]), name
            seen.add(name)

        return hook

    for name, module in new.language_model.named_modules():
        if "_moe_gen" in name:
            module.register_forward_pre_hook(check_rows(name))
    actual, _, _ = new.language_model.model.layers[0](pack, None, (cos, sin))
    torch.testing.assert_close(get_und_seq(actual), get_und_seq(expected), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(get_gen_seq(actual), get_gen_seq(expected), atol=1e-5, rtol=1e-5)
    routed_gen(
        new.language_model.model.norm_moe_gen, new.language_model.model.norm_moe_gen_action, get_gen_seq(actual), pack
    )
    assert any("mlp_moe_gen_action" in name for name in seen)
    assert any("norm_moe_gen_action" in name for name in seen)
    if action_indexes:
        get_gen_seq(actual).square().sum().backward()
        layer = new.language_model.model.layers[0]
        video = layer.self_attn.q_proj_moe_gen.weight
        action = layer.self_attn.q_proj_moe_gen_action.weight
        assert video.data_ptr() != action.data_ptr()
        assert action.grad is not None and action.grad.abs().sum() > 0
        before = video.detach().clone()
        with torch.no_grad():
            action.add_(action.grad, alpha=-1e-3)
        torch.testing.assert_close(video, before, atol=0, rtol=0)


def test_copy_loading_and_roundtrip():
    base, new = tiny_net("two_tower"), tiny_net("three_tower")
    old = base.state_dict()
    target = new.state_dict()
    pending = defer_copy_initialization(target, old)
    for k in target:
        target[k].copy_(old[k])
    finish_copy_initialization(target, pending)
    new.load_state_dict(target, strict=True)
    for k, value in new.state_dict().items():
        torch.testing.assert_close(value, old[source_key(k) or k], atol=0, rtol=0)
    trained = copy.deepcopy(new.state_dict())
    trained["language_model.model.norm_moe_gen_action.weight"].add_(1)
    new.load_state_dict(trained, strict=True)
    assert not torch.equal(
        new.language_model.model.norm_moe_gen_action.weight, new.language_model.model.norm_moe_gen.weight
    )
    with pytest.raises(ValueError, match="two-tower checkpoint"):
        defer_copy_initialization(dict(new.state_dict()), trained)
    with pytest.raises(ValueError, match="missing or skipped"):
        defer_copy_initialization(dict(new.state_dict()), old, skipped=["moe_gen"])


def test_invalid_config():
    with pytest.raises(ValueError, match="copy_gen requires"):
        Cosmos3VFMNetworkConfig(action_tower_init="copy_gen")
    with pytest.raises(ValueError, match="three_tower requires"):
        Cosmos3VFMNetworkConfig(tower_structure="three_tower", action_gen=True, video_temporal_causal=True)


def test_missing_new_checkpoint_weights_fail():
    base, new = tiny_net("two_tower"), tiny_net("three_tower")
    with pytest.raises(ValueError, match="copy_gen"):
        validate_checkpoint_initialization(new.state_dict(), base.state_dict())
    validate_checkpoint_initialization(new.state_dict(), new.state_dict())


def test_action128_warm_start_skips_legacy_heads_but_resume_checks_shapes():
    key = "net.action2llm.fc.weight"
    target = {key: torch.empty(64, 128 * 8)}
    legacy = {key: SimpleNamespace(size=(32, 64 * 8))}
    validate_checkpoint_initialization(target, legacy, skipped=["action2llm"])
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate_checkpoint_initialization(target, legacy)


def test_warm_start_cannot_skip_action_tower_weights():
    key = "net.language_model.model.layers.0.action_block_moe_gen_action.q_proj.weight"
    target = {key: torch.empty(8, 4)}
    with pytest.raises(ValueError, match="Missing three-tower checkpoint parameter"):
        validate_checkpoint_initialization(target, target, skipped=["moe_gen_action"])


def test_meta_materialization_rebuilds_time_frequencies():
    with torch.device("meta"):
        net = tiny_net("three_tower")
    net.to_empty(device="cpu")
    # Poison to ensure the test does not accidentally pass with allocator data.
    net.action_time_embedder._timestep_frequencies.fill_(float("nan"))
    net.init_weights(buffer_device=torch.device("cpu"))
    torch.testing.assert_close(
        net.action_time_embedder._timestep_frequencies,
        net.time_embedder._timestep_frequencies,
        atol=0,
        rtol=0,
    )
    t = torch.tensor([0.2, 0.8])
    torch.testing.assert_close(net.action_time_embedder(t), net.time_embedder(t), atol=0, rtol=0)


def test_state_provenance_not_inferred_from_clean_tokens():
    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder

    builder = PackedSequenceBuilder()
    builder.pack_action_tokens(
        input_action_tokens=torch.randn(6, 8),
        condition_frame_indexes_action=[0, 1, 2],
        robot_state_frame_indexes=[2],
        input_timestep=0.5,
    )
    # Historical actions 0/1 are clean, but only position 2 uses the state head.
    assert builder.action.robot_state_token_indexes == [2]
    builder.pack_action_tokens(
        input_action_tokens=torch.randn(4, 8),
        condition_frame_indexes_action=[0],
        robot_state_frame_indexes=[0],
        input_timestep=0.5,
    )
    assert builder.action.robot_state_token_indexes == [2, 6]
    with pytest.raises(ValueError, match="clean"):
        builder.pack_action_tokens(
            input_action_tokens=torch.randn(4, 8),
            condition_frame_indexes_action=[],
            robot_state_frame_indexes=[0],
            input_timestep=0.5,
        )


def test_three_tower_state_uses_action_encoder_and_must_be_clean():
    from cosmos_framework.data.generator.sequence_packing.modality import ModalityData

    net = tiny_net("three_tower")
    assert not any("robot_state" in key for key in net.state_dict())
    action = ModalityData(
        robot_state_token_indexes=torch.tensor([0]),
        sequence_indexes=torch.tensor([0, 1]),
        token_shapes=[(2,)],
        tokens=[torch.zeros(2, 128)],
        domain_id=[torch.tensor([0])],
        condition_mask=[torch.tensor([[1.0], [1.0]])],
        noisy_frame_indexes=[torch.empty(0, dtype=torch.long)],
    )
    output = torch.zeros(2, 64)
    net._encode_action(SimpleNamespace(action=action), output, torch.float32)
    assert torch.isfinite(output).all()
    action.condition_mask[0][0] = 0
    with pytest.raises(RuntimeError, match="clean conditioning"):
        net._encode_action(SimpleNamespace(action=action), torch.zeros(2, 64), torch.float32)


def test_state_with_noisy_actions_supports_fullgraph_and_checks_conditioning():
    from cosmos_framework.data.generator.sequence_packing.modality import ModalityData

    torch.manual_seed(42)
    net = tiny_net("three_tower")
    action = ModalityData(
        robot_state_token_indexes=torch.tensor([0]),
        sequence_indexes=torch.tensor([0, 1, 2]),
        token_shapes=[(3,)],
        tokens=[torch.randn(3, 128)],
        domain_id=[torch.tensor([32, 0, 0])],
        condition_mask=[torch.tensor([[1.0], [0.0], [0.0]])],
        noisy_frame_indexes=[torch.tensor([1, 2])],
        mse_loss_indexes=torch.tensor([1, 2]),
        timesteps=torch.tensor([0.5, 0.5]),
    )
    packed = SimpleNamespace(action=action, uses_single_timestep=False)
    expected = torch.zeros(3, 64)
    net._encode_action(packed, expected, torch.float32)
    # Match the encode/decode compile boundary used by compiled_region="all".
    encode = torch.compile(net._encode_action, backend="eager", fullgraph=True, dynamic=True)
    actual = torch.zeros_like(expected)
    encode(packed, actual, torch.float32)
    torch.testing.assert_close(actual, expected)

    # Check runtime values on a reused graph, rather than specializing away validation.
    action.condition_mask[0][0, 0] = 0
    with pytest.raises(RuntimeError, match="clean conditioning"):
        encode(packed, torch.zeros_like(expected), torch.float32)


def test_eef_state_projection_is_input_only():
    from cosmos_framework.data.generator.sequence_packing.modality import ModalityData

    net = tiny_net("three_tower")
    action = ModalityData(
        robot_state_token_indexes=torch.tensor([0]),
        sequence_indexes=torch.tensor([0, 1, 2]),
        token_shapes=[(3,)],
        tokens=[torch.zeros(3, 128)],
        domain_id=[torch.tensor([32, 0, 0])],
        condition_mask=[torch.tensor([[1.0], [0.0], [0.0]])],
        noisy_frame_indexes=[torch.tensor([1, 2])],
        mse_loss_indexes=torch.tensor([1, 2]),
        timesteps=torch.tensor([0.5, 0.5]),
    )
    observed = {}
    h1 = net.action2llm.register_forward_pre_hook(lambda module, args: observed.update(input=args[1].tolist()))
    h2 = net.llm2action.register_forward_pre_hook(lambda module, args: observed.update(output=args[1].tolist()))
    hidden = torch.zeros(3, 64)
    net._encode_action(SimpleNamespace(action=action, uses_single_timestep=False), hidden, torch.float32)
    net._decode_action(SimpleNamespace(action=action), hidden, {})
    h1.remove()
    h2.remove()
    assert observed == {"input": [32, 0, 0], "output": [0, 0]}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for original attention backend")
@pytest.mark.parametrize("attention_backend", ["nano", "flex"])
@pytest.mark.parametrize("padded", [False, True])
def test_cuda_native_two_tower_attention_parity(attention_backend, padded):
    torch.manual_seed(42)
    base, new = tiny_net("two_tower"), tiny_net("three_tower")
    state = base.state_dict()
    new.load_state_dict({k: state[source_key(k) or k].clone() for k in new.state_dict()}, strict=True)
    base, new = [net.to(device="cuda", dtype=torch.bfloat16) for net in (base, new)]
    x = torch.randn(40, 64, device="cuda", dtype=torch.bfloat16)
    pack = sequence_pack_from_packed_sequence(
        x,
        ["causal", "full"],
        [8, 32],
        [40],
        torch.arange(8, device="cuda"),
        torch.arange(8, 40, device="cuda"),
        fixed_causal_len=16 if padded else None,
        fixed_full_len=48 if padded else None,
    )
    cos = from_und_gen_splits(
        torch.ones(len(get_und_seq(pack)), 16, device="cuda", dtype=torch.bfloat16),
        torch.ones(len(get_gen_seq(pack)), 16, device="cuda", dtype=torch.bfloat16),
        pack,
    )
    sin = from_und_gen_splits(
        torch.zeros(len(get_und_seq(pack)), 16, device="cuda", dtype=torch.bfloat16),
        torch.zeros(len(get_gen_seq(pack)), 16, device="cuda", dtype=torch.bfloat16),
        pack,
    )
    from cosmos_framework.model.generator.mot.attention import SplitInfo

    mask = SplitInfo([8, 32], ["causal", "full"], [40], 40)
    old, _, _ = base.language_model.model.layers[0](pack, mask, (cos, sin))
    prepare_joint_metadata(
        pack,
        torch.arange(35, 40, device="cuda"),
        attention_backend=attention_backend,
        projection_mode="routed",
    )
    # Audit compute, independently of approximate two/three-tower parity:
    # each role projects only its own rows, once per forward, in BF16.
    seen = {}
    handles = []
    layer = new.language_model.model.layers[0]

    def projection_hook(name, rows):
        def hook(module, args, output):
            assert args[0].shape[0] == rows
            assert args[0].dtype == output.dtype == module.weight.dtype == torch.bfloat16
            seen[name] = seen.get(name, 0) + 1

        return hook

    for suffix, rows in (
        ("", len(get_und_seq(pack))),
        ("_moe_gen", pack["_video_indices"].numel()),
        ("_moe_gen_action", pack["_action_indices"].numel()),
    ):
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = projection + suffix
            handles.append(getattr(layer.self_attn, name).register_forward_hook(projection_hook(name, rows)))
    actual, _, _ = new.language_model.model.layers[0](pack, mask, (cos, sin))
    for handle in handles:
        handle.remove()
    assert len(seen) == 9 and set(seen.values()) == {1}
    atol, rtol = 0.02, 0.01
    torch.testing.assert_close(get_gen_seq(actual), get_gen_seq(old), atol=atol, rtol=rtol)
    get_gen_seq(actual).float().square().mean().backward()
    assert torch.isfinite(new.language_model.model.layers[0].self_attn.q_proj_moe_gen_action.weight.grad).all()


def test_execution_presets_and_explicit_overrides():
    assert resolve_execution("native") == ("nano", "routed")
    assert resolve_execution("nano_compatible") == ("nano", "routed")
    assert resolve_execution("joint") == ("flex", "routed")
    assert resolve_execution("joint", "nano", "routed") == ("nano", "routed")
    for backend in ("nano", "flex"):
        for projection in ("auto", "routed"):
            config = Cosmos3VFMNetworkConfig(
                tower_structure="three_tower",
                action_gen=True,
                action_tower_attention_backend=backend,
                action_tower_projection_mode=projection,
            )
            assert resolve_execution(
                config.action_tower_execution,
                config.action_tower_attention_backend,
                config.action_tower_projection_mode,
            ) == (backend, "routed")
    with pytest.raises(ValueError, match="attention_backend"):
        Cosmos3VFMNetworkConfig(action_tower_attention_backend="unknown")
    with pytest.raises(ValueError, match="projection_mode"):
        Cosmos3VFMNetworkConfig(action_tower_projection_mode="unknown")


def test_production_rejects_full_gen_and_routes_even_with_stale_pack():
    with pytest.raises(ValueError, match="production requires routed"):
        Cosmos3VFMNetworkConfig(action_tower_projection_mode="full_gen")
    x = torch.arange(24, dtype=torch.float32).view(6, 4).requires_grad_()
    pack = {
        "_video_indices": torch.tensor([0, 2, 5]),
        "_action_indices": torch.tensor([1, 3, 4]),
        "_action_tower_projection_mode": "full_gen",
    }
    seen = []

    def video(rows):
        seen.append(rows.detach().clone())
        return rows * 2

    def action(rows):
        seen.append(rows.detach().clone())
        return rows * 3

    output = routed_gen(video, action, x, pack)
    torch.testing.assert_close(seen[0], x[pack["_video_indices"]])
    torch.testing.assert_close(seen[1], x[pack["_action_indices"]])
    output.sum().backward()
    torch.testing.assert_close(x.grad[pack["_video_indices"]], torch.full((3, 4), 2.0))
    torch.testing.assert_close(x.grad[pack["_action_indices"]], torch.full((3, 4), 3.0))


def test_standalone_full_gen_diagnostic_forbids_training():
    from tools.action_tower_diagnostics import full_gen_diagnostic

    video, action = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    x = torch.ones(2, 4)
    pack = {"_video_indices": torch.tensor([0]), "_action_indices": torch.tensor([1])}
    with torch.no_grad(), pytest.raises(RuntimeError, match="training is forbidden"):
        full_gen_diagnostic(video, action, x, pack)
    video.eval()
    action.eval()
    with pytest.raises(RuntimeError, match="training is forbidden"):
        full_gen_diagnostic(video, action, x, pack)
    with torch.no_grad():
        assert full_gen_diagnostic(video, action, x, pack).shape == x.shape


@pytest.mark.parametrize("execution", ["native", "joint"])
@pytest.mark.parametrize("trainable_condition", ["none", "embedding", "visual", "und_layer", "und_mlp", "deepstack"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_frozen_und_graph_isolation_preserves_trainable_gradients(execution, trainable_condition, checkpointed):
    """Packed GEN writes must not force a frozen conditioner through backward.

    Compare all trainable gradients to the old connected graph, including a
    trainable text input, visual feature source, or UND layer. Two layers expose
    accidental graph reconnection through joint attention's concatenated output.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder

    torch.manual_seed(42)
    net = tiny_net("three_tower", num_hidden_layers=2)
    net.config.action_tower_execution = execution
    for name, parameter in net.language_model.named_parameters():
        parameter.requires_grad_("_moe_gen" in name)
    if trainable_condition == "embedding":
        net.language_model.model.embed_tokens.requires_grad_(True)
    elif trainable_condition == "und_layer":
        net.language_model.model.layers[0].self_attn.k_proj.requires_grad_(True)
    elif trainable_condition == "und_mlp":
        net.language_model.model.layers[0].mlp.requires_grad_(True)
    elif trainable_condition in ("visual", "deepstack"):
        # A trainable visual feature inserted by _encode_text, just as packed
        # ViT features are scattered into its text/visual conditioning buffer.
        net.test_visual_feature = torch.nn.Parameter(torch.randn(64))

    reference = copy.deepcopy(net)
    seen = []
    for model in (net, reference):
        original_encode_text = model._encode_text

        def encode_text(*args, _model=model, _original=original_encode_text, **kwargs):
            packed, dtype, deepstack = _original(*args, **kwargs)
            if trainable_condition == "visual":
                packed[0] = packed[0] + _model.test_visual_feature
            elif trainable_condition == "deepstack":
                addition = torch.zeros_like(packed)
                addition[0] = _model.test_visual_feature
                deepstack = [addition]
            if _model is reference:
                # Keep the pre-fix autograd connection without changing values.
                packed = packed + torch.zeros_like(packed, requires_grad=True)
            return packed, dtype, deepstack

        model._encode_text = encode_text
        for layer in model.language_model.model.layers:
            layer.self_attn.dispatch_attention_fn = reference_two_pass
            if model is net:
                layer.input_layernorm.register_forward_hook(lambda module, args, out: seen.append(out.requires_grad))
        if checkpointed:
            for i, layer in enumerate(model.language_model.model.layers):
                model.language_model.model.layers[i] = checkpoint_wrapper(layer)

    def run(model):
        builder = PackedSequenceBuilder()
        builder.begin_sample(0)
        text_len = builder.pack_text_tokens(
            [3, 4], {"eos_token_id": 1, "start_of_generation": 2}, True, supervise_text=False
        )
        action_len = builder.pack_action_tokens(torch.ones(4, 128), [], 0.5)
        builder.finish_sample(action_len, text_len + action_len)
        packed = builder.finalize()
        packed.prepare_sequence_pack_metadata()
        result = model(packed)["preds_action"][0]
        result.square().mean().backward()
        return result.detach()

    expected, actual = run(reference), run(net)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    ref_params = dict(reference.named_parameters())
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            other = ref_params[name].grad
            if other is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, other, atol=1e-6, rtol=1e-5, msg=name)
    if trainable_condition == "none":
        assert seen and not any(seen)
    elif trainable_condition in ("embedding", "visual"):
        assert all(seen)
        source = (
            net.test_visual_feature if trainable_condition == "visual" else net.language_model.model.embed_tokens.weight
        )
        assert source.grad is not None and source.grad.abs().sum() > 0
    else:
        assert any(seen) and not all(seen)
        if trainable_condition == "deepstack":
            source = net.test_visual_feature
        elif trainable_condition == "und_mlp":
            source = net.language_model.model.layers[0].mlp.down_proj.weight
        else:
            source = net.language_model.model.layers[0].self_attn.k_proj.weight
        assert source.grad is not None and source.grad.abs().sum() > 0
