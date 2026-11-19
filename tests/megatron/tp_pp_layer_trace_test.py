import copy

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from tools.run_tp_pp_traced_fp32_math import attach_trace


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, pack):
        return ({"full_only_seq": self.linear(pack["full_only_seq"]).tanh()},)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({str(index): checkpoint_wrapper(Block()) for index in range(2)})

    def forward(self, tensor):
        pack = {"full_only_seq": tensor}
        for block in self.layers.values():
            pack = block(pack)[0]
        return pack["full_only_seq"]


def test_layer_trace_keeps_gradients_rng_and_checkpoint_recomputation(tmp_path):
    torch.manual_seed(73)
    reference = Network()
    traced = copy.deepcopy(reference)
    attach_trace(traced, traced.layers.items(), tmp_path)
    for _ in range(6):
        inputs = torch.randn(4, 8)
        before = torch.get_rng_state().clone()
        expected, actual = reference(inputs), traced(inputs)
        assert torch.equal(expected, actual)
        expected.square().sum().backward()
        actual.square().sum().backward()
        assert torch.equal(before, torch.get_rng_state())
        for original, observed in zip(reference.parameters(), traced.parameters(), strict=True):
            assert torch.equal(original.grad, observed.grad)
    assert len(list(tmp_path.glob("*.outputs.pt"))) == 12
    assert len(list(tmp_path.glob("*.gradient.pt"))) == 12
