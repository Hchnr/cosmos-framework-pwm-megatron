"""Experimental fusion must retain actual decoder math and checkpoint keys."""

import copy

import torch

from cosmos_framework.model.generator.mot.small_action_optimization_test import make_layer, make_pack
from cosmos_framework.model.generator.mot.small_action_tower import small_layer_forward
from cosmos_framework.training_backends.projection_fusion import fuse_decoder_projections


def test_fused_decoder_outputs_gradients_and_state_names():
    torch.manual_seed(42)
    reference, shape = make_layer()
    candidate = copy.deepcopy(reference)
    fuse_decoder_projections([candidate])
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    samples = [(torch.randn(u, 64), torch.randn(v, 64), torch.randn(a, 32)) for u, v, a in ((2, 3, 2), (1, 2, 1))]
    pack = make_pack(samples, shape, fixed_lengths=(8, 12, 6))
    expected = small_layer_forward(reference, pack)[0]
    actual = small_layer_forward(candidate, pack)[0]
    for key in ("causal_seq", "full_only_seq", "action_seq"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-5)
    for output in (expected, actual):
        (output["full_only_seq"].square().mean() + output["action_seq"].square().mean()).backward()
    reference_params = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, reference_params[name].grad, atol=2e-6, rtol=2e-5, msg=name)
        else:
            assert parameter.grad is None
