# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from tools.run_tp_pp_und_selective_ac import Observation, buckets, make_policy, should_save_mm, wrap_block


def config(full=8, short=4, video=11, action=6, enabled=True):
    return SimpleNamespace(
        fixed_sequence_length=SimpleNamespace(
            enabled=enabled, und_tokens=full, gen_tokens=video + action, action_tokens=action
        ),
        action_tower_und_short_tokens=short,
    )


def test_declared_row_buckets_match_and_other_declared_stream_sizes_do_not(tmp_path):
    lengths = buckets(config())
    observer = Observation(tmp_path, 0)
    policy = make_policy(lengths, observer)
    from torch.utils.checkpoint import CheckpointPolicy

    for rows, save in ((4, True), (8, True), (11, False), (6, False), (3, False)):
        args = (torch.randn(rows, 3), torch.randn(3, 5))
        assert should_save_mm(torch.ops.aten.mm.default, args, lengths) is save
        assert (policy(None, torch.ops.aten.mm.default, *args) == CheckpointPolicy.MUST_SAVE) is save
    assert observer.data["saved_mm_by_tokens"] == {"4": 1, "8": 1}
    assert observer.data["saved_mm_by_shape"] == {"(4, 3) @ (3, 5)": 1, "(8, 3) @ (3, 5)": 1}
    assert observer.data["provenance_verified"] is False
    assert not should_save_mm(torch.ops.aten.add.Tensor, (torch.randn(4, 3), torch.randn(4, 3)), lengths)


@pytest.mark.parametrize(
    "configuration",
    [
        config(full=8, short=4, video=4),
        config(full=8, short=4, action=8),
        config(full=8, short=4, video=6, action=6),
        config(full=8, short=8),
        config(enabled=False),
    ],
)
def test_ambiguous_or_invalid_capacity_fails_closed(configuration):
    with pytest.raises(ValueError):
        buckets(configuration)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 5, dtype=torch.float64))

    def forward(self, x):
        return torch.sin(x @ self.weight) * torch.rand((x.shape[0], 5), dtype=x.dtype)


def test_checkpoint_keeps_parameters_rng_and_full_gradient(tmp_path):
    torch.manual_seed(3)
    original = Model()
    candidate = copy.deepcopy(original)
    original_ids = {name: id(value) for name, value in candidate.named_parameters()}
    ac = ActivationCheckpointingConfig(mode="full", preserve_rng_state=True, determinism_check="default")
    observation = Observation(tmp_path, 0)
    wrapped = wrap_block(candidate, ac, make_policy(buckets(config()), observation))
    assert {name: id(value) for name, value in wrapped.named_parameters()} == original_ids
    x = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    torch.manual_seed(42)
    expected = original(x)
    expected.sum().backward()
    next_expected = torch.rand(1)
    torch.manual_seed(42)
    actual = wrapped(y)
    actual.sum().backward()
    next_actual = torch.rand(1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)
    torch.testing.assert_close(candidate.weight.grad, original.weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(next_actual, next_expected, rtol=0, atol=0)
    assert observation.data["saved_mm_by_tokens"].get("8", 0) > 0


class CountMM(TorchDispatchMode):
    def __init__(self):
        self.mm = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.mm.default:
            self.mm += 1
        return func(*args, **(kwargs or {}))


def test_policy_saves_row_matched_mm_and_avoids_recomputation(tmp_path):
    ac = ActivationCheckpointingConfig(mode="full", preserve_rng_state=True, determinism_check="default")
    counts = []
    for rows in (8, 11):
        observation = Observation(tmp_path / str(rows), 0)
        wrapped = wrap_block(Model(), ac, make_policy(buckets(config()), observation))
        mode = CountMM()
        with mode:
            wrapped(torch.randn(rows, 3, dtype=torch.float64, requires_grad=True)).sum().backward()
        counts.append(mode.mm)
    assert counts[0] < counts[1], counts


def test_policy_is_observed_in_aot_compiled_blocks(tmp_path):
    observation = Observation(tmp_path, 1)
    ac = ActivationCheckpointingConfig(mode="full", preserve_rng_state=True, determinism_check="default")
    model = torch.compile(
        wrap_block(Model(), ac, make_policy(buckets(config()), observation)),
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )
    model(torch.randn(8, 3, dtype=torch.float64, requires_grad=True)).sum().backward()
    assert observation.data["saved_mm_by_tokens"].get("8", 0) > 0
    observation.finish()


def test_unobserved_policy_does_not_pass_qualification(tmp_path):
    observation = Observation(tmp_path, 0)
    with pytest.raises(RuntimeError, match="No row-matched mm save"):
        observation.finish()


def test_non_und_intermediate_can_collide_with_an_und_row_count(tmp_path):
    lengths = buckets(config())
    observer = Observation(tmp_path, 0)
    policy = make_policy(lengths, observer)
    from torch.utils.checkpoint import CheckpointPolicy

    # The same shape could arise by flattening an unrelated VIDEO intermediate;
    # the policy cannot distinguish it from an UND projection by provenance.
    unrelated = (torch.randn(8, 3), torch.randn(3, 5))
    assert policy(None, torch.ops.aten.mm.default, *unrelated) == CheckpointPolicy.MUST_SAVE
    assert observer.data["provenance_verified"] is False
    assert observer.data["matching_non_und_ops_possible"] is True
