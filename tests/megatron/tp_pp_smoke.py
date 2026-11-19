"""Real TP sharding and 1F1B versus native simultaneous three-tower math.

Run with four GPUs; supports TP/PP/DP matrix, nonzero Adam state, multiple
updates, unequal sample counts, fixed padding and absent Action tokens.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils._pytree import tree_map

from cosmos_framework.model.generator.megatron_tp_pp.payload import STREAMS, PayloadLayout
from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
from cosmos_framework.model.generator.mot.small_action_optimization_test import make_layer, make_pack
from cosmos_framework.model.generator.mot.small_action_tower import small_decoder_inputs, small_layer_forward
from cosmos_framework.training_backends.megatron_tp_pp.batch import logical_slots, reference_loss_weight
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver
from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states, initialize_adam_state
from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import (
    canonical_local_name,
    stage_parameter_manifest,
)
from cosmos_framework.training_backends.megatron_tp_pp.runtime import initialize, transformer_config


class InputAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae2llm = nn.Linear(64, 64)
        self.action2llm = nn.Linear(32, 32)

    def forward(self, batch):
        pack = dict(batch["pack"])
        pack["full_only_seq"] = self.vae2llm(pack["full_only_seq"])
        pack["action_seq"] = self.action2llm(pack["action_seq"])
        return pack


class OutputAdapter(nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.llm2vae = nn.Linear(64, 64)
        self.llm2action = nn.Linear(32, 32)
        self.layout = layout

    def forward(self, pack, batch):
        pack = dict(pack)
        pack["full_only_seq"] = self.llm2vae(pack["full_only_seq"])
        pack["action_seq"] = self.llm2action(pack["action_seq"])
        return batch.get("layout", self.layout).pack(pack)


def compare(actual, expected, name, errors, *, atol=2e-6, rtol=1e-4):
    assert torch.isfinite(actual).all(), name
    diff = (actual.double() - expected.double()).norm().item()
    norm = expected.double().norm().item()
    relative = diff / max(norm, 1e-30)
    errors[name] = {"relative_l2": relative, "absolute_max": (actual - expected).abs().max().item()}
    # Acceptance uses per-parameter relative L2, with an absolute gate for
    # theoretical zero / near-zero tensors, as fixed in the implementation plan.
    if norm > 1e-5:
        assert relative <= 1e-4, (name, errors[name])
    else:
        assert errors[name]["absolute_max"] <= atol, (name, errors[name])


def main():
    from megatron.core import parallel_state as ps
    from megatron.core.optimizer import OptimizerConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint-activations", action="store_true")
    parser.add_argument("--train-io", action="store_true")
    parser.add_argument("--dynamic-layouts", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--distributed-optimizer", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    topology = ModelParallelConfig(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        global_packed_microbatches_per_update=12,
        logical_data_parallel_size=4,
        reference_micro_steps_per_update=3,
        use_distributed_optimizer=args.distributed_optimizer,
    )
    groups = initialize(topology)
    torch.manual_seed(732)
    base, action_shape = make_layer()
    # Different layer weights prevent swapped stage/layer identity passing.
    second, _ = make_layer()
    layers = nn.ModuleDict({"0": base, "1": second}).cuda()
    reference = nn.Module()
    reference.layers = copy.deepcopy(layers)
    layout = PayloadLayout(((8, 64), (12, 64), (6, 32)))
    if args.train_io:
        reference.input_adapter = InputAdapter().cuda()
        reference.output_adapter = OutputAdapter(layout).cuda()
    config = transformer_config(
        topology, num_layers=2, hidden_size=64, num_attention_heads=4, variable_seq_lengths=args.dynamic_layouts
    )
    stage = PipelineStage(
        config,
        lambda i, source=layers: copy.deepcopy(source[str(i)]),
        layout=layout,
        input_factory=(lambda: copy.deepcopy(reference.input_adapter)) if args.train_io else None,
        output_factory=(lambda: copy.deepcopy(reference.output_adapter)) if args.train_io else None,
    ).cuda()
    del layers
    if args.checkpoint_activations:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        for i, layer in stage.layers.items():
            stage.layers[i] = checkpoint_wrapper(layer)
    if args.compile:
        for i, layer in stage.layers.items():
            stage.layers[i] = torch.compile(layer, fullgraph=True, dynamic=False)
    manifest = stage_parameter_manifest(stage, decoder_prefix="")
    optim_config = OptimizerConfig(
        lr=1e-3,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_eps=1e-8,
        clip_grad=0.3,
        use_distributed_optimizer=args.distributed_optimizer,
    )
    driver = ModelParallelDriver(stage, topology, optim_config)
    # Match native all-parameter weight decay; MCore defaults exempt norms/bias.
    for group in driver.optimizer.param_groups:
        group["weight_decay"] = 0.01
        group["wd_mult"] = 1.0
    reference_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad],
        lr=1e-3,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01,
    )
    reference_scheduler = torch.optim.lr_scheduler.LambdaLR(reference_optimizer, lambda step: 1 - 0.05 * step)
    inner = getattr(driver.optimizer, "chained_optimizers", [driver.optimizer])[0].optimizer
    driver.scheduler = torch.optim.lr_scheduler.LambdaLR(inner, lambda step: 1 - 0.05 * step)
    # Exercise a genuine continuation, not just Adam's zero-moment first step.
    for index, (name, param) in enumerate(reference.named_parameters()):
        if param.requires_grad:
            torch.manual_seed(8000 + index)
            reference_optimizer.state[param] = {
                "step": torch.tensor(7.0),
                "exp_avg": torch.randn_like(param) * 0.01,
                "exp_avg_sq": torch.rand_like(param) * 0.01 + 0.001,
            }
    for entry in adam_parameter_states(stage, driver.optimizer):
        ref = dict(reference.named_parameters())[canonical_local_name(entry.name)]
        state = reference_optimizer.state[ref]
        initialize_adam_state(
            entry,
            step=7,
            master=manifest[entry.name].select(ref.detach()),
            exp_avg=manifest[entry.name].select(state["exp_avg"]),
            exp_avg_sq=manifest[entry.name].select(state["exp_avg_sq"]),
        )
    errors, norms, gradient_summary = {}, [], []
    if args.resume:
        from cosmos_framework.training_backends.megatron_tp_pp.checkpoint import load

        restored = load(driver, args.checkpoint_dir)
        reference.load_state_dict(restored["reference_model"])
        reference_optimizer.load_state_dict(restored["reference_optimizer"])
        reference_scheduler.load_state_dict(restored["reference_scheduler"])
    dp_size, dp_rank = ps.get_data_parallel_world_size(), ps.get_data_parallel_rank()
    local_parameters = dict(stage.named_parameters())
    for update in range(driver.iteration, args.updates):
        batches = []
        counts = [1, 2, 1, 3]
        for slot in logical_slots(topology, update=update, data_parallel_size=dp_size, data_parallel_rank=dp_rank):
            torch.manual_seed(1000 + update * 12 + slot.micro_step * 4 + slot.logical_rank)
            action_missing = slot.logical_rank == 0 or update == 2
            # Multiple packed samples; one logical rank has no Action this step.
            samples = [
                (torch.randn(1, 64), torch.randn(2, 64), torch.randn(0 if action_missing else 1, 32))
                for _ in range(counts[slot.logical_rank])
            ]
            current_layout = PayloadLayout(
                ((4 if args.dynamic_layouts and slot.logical_rank % 2 else 8, 64), (12, 64), (6, 32))
            )
            pack = make_pack(samples, action_shape, fixed_lengths=tuple(s[0] for s in current_layout.shapes))
            inputs = tree_map(lambda x: x.cuda() if isinstance(x, torch.Tensor) else x, small_decoder_inputs(pack))
            weight = reference_loss_weight(topology, local_samples=len(samples), reference_modality_samples=sum(counts))
            batches.append(
                {
                    "pack": inputs,
                    "metadata": {k: v for k, v in inputs.items() if k not in STREAMS},
                    "weight": weight,
                    "layout": current_layout,
                    "lengths": (len(samples), 2 * len(samples), 0 if action_missing else len(samples)),
                }
            )

        def loss_fn(tensor, batch):
            if "expected" in batch:
                compare(tensor, batch["expected"], f"update{update}/output", errors)
            streams = batch["layout"].unpack(tensor)
            loss = (
                sum(
                    streams[key][:length].square().sum() / max(1, length * streams[key].shape[-1])
                    for key, length in zip(STREAMS[1:], batch["lengths"][1:], strict=True)
                )
                * batch["weight"]
            )
            return loss, {"loss": loss.detach()}

        reference_optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            pack = reference.input_adapter(batch) if args.train_io else batch["pack"]
            for layer in reference.layers.values():
                pack = small_layer_forward(layer, pack)[0]
            output = reference.output_adapter(pack, batch) if args.train_io else batch["layout"].pack(pack)
            loss_fn(output, batch)[0].div(driver.microbatches).backward()
            batch["expected"] = output.detach()
        for param in reference.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, group=ps.get_data_parallel_group())
                param.grad.div_(dp_size)
        driver.forward_backward(batches, loss_fn)
        for name, param in local_parameters.items():
            canonical = canonical_local_name(name)
            ref = dict(reference.named_parameters())[canonical]
            if not ref.requires_grad:
                assert not param.requires_grad and param.grad is None
                continue
        # With distributed optimizer only the DP-owned fragment is reduced;
        # non-owned main_grad storage is scratch, not a global gradient.
        statistics = torch.zeros(4, dtype=torch.float64, device="cuda")
        for entry in adam_parameter_states(stage, driver.optimizer):
            ref = dict(reference.named_parameters())[canonical_local_name(entry.name)]
            actual_grad = entry.select(entry.model_parameter.main_grad).double()
            reference_grad = entry.select(manifest[entry.name].select(ref.grad)).double()
            compare(
                actual_grad,
                reference_grad,
                f"update{update}/grad/{entry.name}",
                errors,
            )
            if torch.count_nonzero(reference_grad) == 0:
                assert torch.count_nonzero(actual_grad) == 0, entry.name
            if manifest[entry.name].axis is None and ps.get_tensor_model_parallel_rank() != 0:
                continue
            if not args.distributed_optimizer and dp_rank != 0:
                continue
            statistics += torch.stack(
                (
                    (actual_grad - reference_grad).square().sum(),
                    actual_grad.square().sum(),
                    reference_grad.square().sum(),
                    (actual_grad * reference_grad).sum(),
                )
            )
        dist.all_reduce(statistics)
        relative = (statistics[0] / statistics[2]).sqrt().item()
        cosine = (statistics[3] / (statistics[1] * statistics[2]).sqrt()).item()
        assert relative <= 1e-4 and cosine >= 0.999999, (relative, cosine)
        gradient_summary.append({"relative_l2": relative, "cosine": cosine})
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
        success, actual_norm, _ = driver.step()
        assert success
        torch.testing.assert_close(torch.as_tensor(actual_norm, device="cuda"), expected_norm, atol=2e-6, rtol=1e-5)
        norms.append(float(actual_norm))
        reference_optimizer.step()
        reference_scheduler.step()
        assert driver.scheduler.last_epoch == reference_scheduler.last_epoch == update + 1
        for group in driver.optimizer.param_groups:
            assert group["lr"] == reference_optimizer.param_groups[0]["lr"]
        # Distributed optimizer gathers updated parameters before next forward.
        driver.model.start_param_sync(force_sync=True)
        for entry in adam_parameter_states(stage, driver.optimizer):
            ref = dict(reference.named_parameters())[canonical_local_name(entry.name)]
            state = reference_optimizer.state[ref]
            for key in ("exp_avg", "exp_avg_sq"):
                compare(
                    entry.state[key],
                    entry.select(manifest[entry.name].select(state[key])),
                    f"update{update}/{key}/{entry.name}",
                    errors,
                )
        for name, param in local_parameters.items():
            ref = dict(reference.named_parameters())[canonical_local_name(name)]
            compare(param, manifest[name].select(ref), f"update{update}/weight/{name}", errors)
        for entry in adam_parameter_states(stage, driver.optimizer):
            ref = dict(reference.named_parameters())[canonical_local_name(entry.name)]
            state = reference_optimizer.state[ref]
            for key in ("exp_avg", "exp_avg_sq"):
                compare(
                    entry.state[key],
                    entry.select(manifest[entry.name].select(state[key])),
                    f"update{update}/{key}/{entry.name}",
                    errors,
                )
            compare(
                entry.master,
                entry.select(manifest[entry.name].select(ref)),
                f"update{update}/master/{entry.name}",
                errors,
            )
            assert entry.step == state["step"].item() == 8 + update
        if args.stop_after is not None and driver.iteration == args.stop_after:
            from cosmos_framework.training_backends.megatron_tp_pp.checkpoint import save

            save(
                driver,
                args.checkpoint_dir,
                rank_state={
                    "reference_model": reference.state_dict(),
                    "reference_optimizer": reference_optimizer.state_dict(),
                    "reference_scheduler": reference_scheduler.state_dict(),
                },
            )
            break
    report = {
        "passed": True,
        "tp": args.tp,
        "pp": args.pp,
        "dp": dp_size,
        "groups": groups,
        "updates": driver.iteration,
        "compared_updates": len(norms),
        "global_slots": 12,
        "precision": "float32",
        "compiled": args.compile,
        "checkpoint_activations": args.checkpoint_activations,
        "trainable_io": args.train_io,
        "dynamic_layouts": args.dynamic_layouts,
        "scheduler_step": driver.scheduler.last_epoch,
        "grad_norms": norms,
        "gradient_summary": gradient_summary,
        "errors": errors,
        "local_parameter_numel": sum(p.numel() for p in stage.parameters()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target = args.output.with_name(f"{args.output.stem}.rank{dist.get_rank()}.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    if dist.get_rank() == 0:
        print(json.dumps({k: v for k, v in report.items() if k != "errors"}))
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
