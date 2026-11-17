"""Reasoner generation through training FSDP wrappers, followed by a train step."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omni_ema_smoke import batch
from omni_hf_initialization_smoke import construct

from cosmos_framework.training_backends.megatron import get_runtime
from cosmos_framework.utils.distributed import ddp_sync_grad
from cosmos_framework.utils.generator.materialize_tensor import materialize_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen", "nemotron"), default="qwen")
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    path = Path("/tmp") / f"cosmos_megatron_reasoner_{os.environ['MASTER_PORT']}"
    source, source_opt = construct("pytorch", args.family, path, "AdamW", args.sharding_strategy, False, False)
    target, target_opt = construct("megatron", args.family, path, "AdamW", args.sharding_strategy, False, False)
    dcp.save({"model": source.state_dict()}, checkpoint_id=path)
    state = {"model": target.state_dict()}
    dcp.load(state, checkpoint_id=path)
    target.load_state_dict(state["model"], strict=False)
    ids = torch.tensor([[1, 2, 3]], device="cuda")
    results = []
    with torch.no_grad():
        for model in (source, target):
            model.eval()
            results.append(model.net.generate_reasoner_text(ids, max_new_tokens=3, do_sample=False, seed=42))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    ema_results = []
    with torch.no_grad():
        for model in (source, target):
            head = model.net_ema.language_model.lm_head.weight
            head.to_local().neg_()
            with model.ema_scope(is_cpu=True):
                ema_results.append(model.net.generate_reasoner_text(ids, max_new_tokens=3, do_sample=False, seed=42))
            restored_generation = model.net.generate_reasoner_text(ids, max_new_tokens=3, do_sample=False, seed=42)
            torch.testing.assert_close(restored_generation, results[0], rtol=0, atol=0)
    torch.testing.assert_close(ema_results[0], ema_results[1], rtol=0, atol=0)
    assert not torch.equal(ema_results[0], results[0]), "EMA head mutation must change greedy tokens"
    output_error = 0.0
    for model in (source, target):
        model.train()
    for optimizer in (source_opt, target_opt):
        optimizer.zero_grad()
    for micro in range(3):
        torch.manual_seed(100 + micro + dist.get_rank())
        data, outputs = batch(), []
        for model in (source, target):
            with ddp_sync_grad(model, enabled=micro == 2):
                output = model.net(copy.deepcopy(data))["preds_action"][0]
                (output.float().square().mean() / 3).backward()
            outputs.append(output.detach())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-4, atol=2e-5)
        output_error = max(output_error, (outputs[0] - outputs[1]).abs().max().item())

    def canonical(name):
        return name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")

    expected = {canonical(n): p for n, p in source.net.named_parameters()}
    gradients = {
        n: p.grad if p.grad is not None else torch.zeros_like(p)
        for n, p in target.net.named_parameters()
        if p.requires_grad
    }
    get_runtime(target).prepare_state_dict(gradients)
    maximum_gradient_error = 0.0
    for name, gradient in gradients.items():
        parameter = expected[canonical(name)]
        baseline = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        a, b = materialize_tensor(baseline), materialize_tensor(gradient)
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5, msg=name)
        maximum_gradient_error = max(maximum_gradient_error, (a - b).abs().max().item())
    source_opt.step()
    target_opt.step()
    expected_state = {canonical(n): value for n, value in source.net.state_dict().items()}
    maximum_parameter_error = 0.0
    for name, value in target.net.state_dict().items():
        a, b = materialize_tensor(expected_state[canonical(name)]), materialize_tensor(value)
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5, msg=name)
        if a.is_floating_point():
            maximum_parameter_error = max(maximum_parameter_error, (a - b).abs().max().item())
    result = dict(
        passed=True,
        family=args.family,
        ema_generation_bitwise_equal=True,
        original_generation_restored=True,
        generated_tokens_bitwise_equal=True,
        generation=results[1].tolist(),
        training_after_generation=True,
        gradient_accumulation=3,
        compile=True,
        sharding_strategy=args.sharding_strategy,
        maximum_training_output_error=output_error,
        maximum_training_gradient_error=maximum_gradient_error,
        maximum_updated_parameter_error=maximum_parameter_error,
    )
    if dist.get_rank() == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
