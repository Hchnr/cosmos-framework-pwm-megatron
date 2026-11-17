"""Actual MoT CP=2 forward, gradients and AdamW parity against FSDP2."""

import argparse
import copy
import json
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from omni_ema_smoke import batch, tiny_nemotron_config
from torch import nn

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.model.generator.mot.action_tower_test import tiny_action128_spec, tiny_net
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.parallelize_vfm_network import parallelize_vfm_network
from cosmos_framework.model.generator.mot.unified_mot import Nemotron3DenseVLTextForCausalLM
from cosmos_framework.training_backends.megatron import MegatronRuntime
from cosmos_framework.utils.distributed import ddp_sync_grad
from cosmos_framework.utils.generator.materialize_tensor import materialize_tensor
from cosmos_framework.utils.generator.parallelism import ParallelDims


class Owner(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cp", type=int, choices=(1, 2), default=2)
    parser.add_argument("--family", choices=("qwen", "nemotron"), default="qwen")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--layout", choices=("sequence_sharded", "replicated"), default="sequence_sharded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    if args.family == "nemotron":
        language = Nemotron3DenseVLTextForCausalLM(tiny_nemotron_config())
        base = Cosmos3VFMNetwork(
            language,
            Cosmos3VFMNetworkConfig(
                vlm_config=language.config,
                action_gen=True,
                action_dim=128,
                num_embodiment_domains=64,
                action128_spec=tiny_action128_spec(),
                tower_structure="two_tower",
            ),
        )
    else:
        base = tiny_net("two_tower", num_hidden_layers=2)
    base = base.cuda().to(getattr(torch, args.dtype))
    base.init_weights(buffer_device=torch.device("cuda"))
    for name, p in base.named_parameters():
        p.requires_grad_(
            "_moe_gen" in name
            or any(k in name for k in ("action_modality_embed", "action_time", "action2llm", "llm2action"))
        )
    target = copy.deepcopy(base)
    dims = ParallelDims(
        world_size=dist.get_world_size(), dp_shard=2, dp_replicate=2, cp=args.cp, enable_inference_mode=False
    )
    dims.build_meshes(device_type="cuda")
    compilation = CompileConfig(enabled=args.compile, compile_dynamic=False)
    ac = ActivationCheckpointingConfig(mode="full")
    source = Owner(
        parallelize_vfm_network(base, dims, compilation, ac, attention_io_layout=args.layout, backend="pytorch")
    )
    target = Owner(
        parallelize_vfm_network(target, dims, compilation, ac, attention_io_layout=args.layout, backend="megatron")
    )
    runtime = MegatronRuntime(
        target.net,
        shard_degree=2,
        replicate_degree=2,
        unit_types=tuple({type(layer) for layer in target.net.language_model.model.layers}),
    )
    # Keep the upstream guard observable outside compiled forwards. Released
    # raw ZeRO-3 storage must not silently become an ordinary conversion.
    eager_guard_checked = False
    for parameter in runtime.fsdp.raw_param.values():
        if parameter.untyped_storage().nbytes() == 0:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                converted = parameter.to(torch.float32)
            assert converted.ndim == 0
            assert any("sharded by Megatron-FSDP" in str(item.message) for item in captured)
            eager_guard_checked = True
            break
    assert eager_guard_checked, "Expected released raw ZeRO-3 storage"
    optimizers = [
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, fused=True)
        for model in (source, target)
    ]
    maximum_output_error = 0.0
    maximum_gradient_error = 0.0
    maximum_parameter_error = 0.0
    for step in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad()
        runtime.zero_grad()
        for micro in range(3):
            torch.manual_seed(step * 100 + micro + dist.get_rank() // args.cp)
            data = batch()
            outputs = []
            for model in (source, target):
                with ddp_sync_grad(model, enabled=micro == 2):
                    output = model.net(copy.deepcopy(data))["preds_action"][0]
                    (output.float().square().mean() / 3).backward()
                outputs.append(output.detach())
            maximum_output_error = max(maximum_output_error, (outputs[0] - outputs[1]).abs().max().item())
            torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-4, atol=2e-5)
        expected = dict(source.net.named_parameters())
        actual = dict(target.net.named_parameters())
        gradients = {
            name: p.grad if p.grad is not None else torch.zeros_like(p) for name, p in actual.items() if p.requires_grad
        }
        runtime.prepare_state_dict(gradients)
        for name, p in expected.items():
            if not p.requires_grad:
                continue
            reference = p.grad
            candidate = gradients[name]
            if candidate is None:
                assert actual[name].to_local().numel() == 0
                candidate = torch.zeros_like(actual[name])
            a, b = materialize_tensor(reference), materialize_tensor(candidate)
            assert torch.isfinite(a).all(), f"Native FSDP2 non-finite gradient: {name}"
            assert torch.isfinite(b).all(), f"Megatron non-finite gradient: {name}"
            maximum_gradient_error = max(maximum_gradient_error, (a - b).abs().max().item())
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5, msg=name)
        for optimizer in optimizers:
            optimizer.step()
        runtime.after_step()
        expected_state = source.net.state_dict()
        for name, value in target.net.state_dict().items():
            a, b = materialize_tensor(expected_state[name]), materialize_tensor(value)
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5, msg=name)
            if a.is_floating_point():
                maximum_parameter_error = max(maximum_parameter_error, (a - b).abs().max().item())
    result = dict(
        passed=True,
        eager_cast_guard_preserved=eager_guard_checked,
        cp=args.cp,
        family=args.family,
        shards=2,
        replicas=2,
        compile=args.compile,
        attention_io_layout=args.layout,
        dtype=args.dtype,
        steps=2,
        gradient_accumulation=3,
        maximum_output_error=maximum_output_error,
        maximum_gradient_error=maximum_gradient_error,
        maximum_parameter_error=maximum_parameter_error,
    )
    if dist.get_rank() == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
