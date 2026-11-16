import pytest
import torch

from tools.benchmark_action_training_compute import PermutationGather


@pytest.mark.parametrize("length", [1, 2, 17, 128])
@pytest.mark.parametrize("seed", range(5))
def test_permutation_gather_matches_indexing_and_gradient(length, seed):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(length, generator=generator)
    inverse = torch.argsort(order)
    x = torch.randn(length, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(length, 3, generator=generator, dtype=torch.float64)
    actual = PermutationGather.apply(x, order, inverse)
    actual_grad = torch.autograd.grad(actual, x, grad)[0]

    reference_x = x.detach().clone().requires_grad_()
    reference = reference_x[order]
    reference_grad = torch.autograd.grad(reference, reference_x, grad)[0]
    assert torch.equal(actual, reference)
    assert torch.equal(actual_grad, reference_grad)
