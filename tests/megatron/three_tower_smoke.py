"""Four-GPU parity for the actual packed heterogeneous three-stream decoder."""

import argparse
import copy
import json
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils._pytree import tree_map

from cosmos_framework.model.generator.mot.small_action_optimization_test import make_layer, make_pack
from cosmos_framework.model.generator.mot.small_action_tower import small_decoder_inputs
from cosmos_framework.training_backends.megatron import MegatronRuntime


class Decoder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.unused_frozen = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, pack):
        return self.layer(pack, None, None)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fuse-projections", action="store_true")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import uneven_dtensor_to_full_tensor

    torch.manual_seed(42)
    layer, shape = make_layer()
    layer.small_action_enabled = True
    model = Decoder(layer).cuda()
    reference = copy.deepcopy(model)
    if args.fuse_projections:
        from cosmos_framework.training_backends.projection_fusion import fuse_decoder_projections

        fuse_decoder_projections([model.layer])
    model.layer = checkpoint_wrapper(model.layer)
    if args.compile:
        model.layer = torch.compile(model.layer, fullgraph=True, dynamic=False)
    runtime = MegatronRuntime(
        model,
        shard_degree=dist.get_world_size() // args.replicas,
        replicate_degree=args.replicas,
        unit_types=(type(model.layer),),
        sharding_strategy=args.sharding_strategy,
    )
    runtime.zero_grad()
    for micro in range(3):
        torch.manual_seed(42 + micro + dist.get_rank())
        samples = [(torch.randn(u, 64), torch.randn(v, 64), torch.randn(a, 32)) for u, v, a in ((2, 3, 2), (1, 2, 1))]
        pack = make_pack(samples, shape, fixed_lengths=(8, 12, 6))
        pack = tree_map(lambda x: x.cuda() if isinstance(x, torch.Tensor) else x, small_decoder_inputs(pack))
        expected = reference(pack)
        with runtime.accumulate(micro == 2):
            actual = model(pack)
            for key in ("causal_seq", "full_only_seq", "action_seq"):
                torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-5)
            loss = sum(actual[key].square().mean() for key in ("full_only_seq", "action_seq")) / 3
            loss.backward()
        (sum(expected[key].square().mean() for key in ("full_only_seq", "action_seq")) / 3).backward()
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    distributed = get_model_state_dict(model, options=None)
    max_gradient_error = 0.0
    for name, parameter in model.named_parameters():
        canonical = name.replace("._checkpoint_wrapped_module", "").replace("._orig_mod", "")
        ref = dict(reference.named_parameters())[canonical]
        if not ref.requires_grad:
            assert parameter.grad is None
            continue
        dist.all_reduce(ref.grad)
        ref.grad.div_(dist.get_world_size())
        grad = parameter.grad
        if grad is None:
            assert parameter.to_local().numel() == 0
            grad = torch.zeros_like(parameter)
        actual = uneven_dtensor_to_full_tensor(grad)
        max_gradient_error = max(max_gradient_error, (actual - ref.grad).abs().max().item())
        torch.testing.assert_close(actual, ref.grad, atol=2e-6, rtol=2e-5, msg=canonical)
    assert distributed
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "passed": True,
                    "model": "actual three-stream MoT decoder",
                    "compiled": args.compile,
                    "fused_projections": args.fuse_projections,
                    "replicas": args.replicas,
                    "max_gradient_error": max_gradient_error,
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
