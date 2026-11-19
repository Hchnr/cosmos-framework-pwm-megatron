"""Real production I/O + TP blocks + PP metadata versus the native network."""

import argparse
import copy
import faulthandler
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from tp_pp_network_io_test import make_native, make_sequence

from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs, LastStageOutputs
from cosmos_framework.model.generator.megatron_tp_pp.payload import PayloadLayout
from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
from cosmos_framework.training_backends.megatron_tp_pp.batch import logical_slots
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver
from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states, initialize_adam_state
from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import (
    load_canonical_stage,
    stage_parameter_manifest,
)
from cosmos_framework.training_backends.megatron_tp_pp.runtime import initialize, transformer_config
from cosmos_framework.training_backends.megatron_tp_pp.transport import PipelineMetadata, broadcast_batch


def main():
    from megatron.core import parallel_state as ps
    from megatron.core.optimizer import OptimizerConfig

    from cosmos_framework.utils.generator.fused_adam import FusedAdam

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    faulthandler.dump_traceback_later(45, repeat=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = False
    topology = ModelParallelConfig(
        global_packed_microbatches_per_update=12, logical_data_parallel_size=4, reference_micro_steps_per_update=3
    )
    groups = initialize(topology)
    torch.manual_seed(421)
    reference, shape = make_native()
    reference = reference.cuda()
    config = transformer_config(
        topology, num_layers=2, hidden_size=64, num_attention_heads=4, variable_seq_lengths=True
    )
    maximum = PayloadLayout(((8, 64), (12, 64), (6, 32)), context_width=96)
    stage = PipelineStage(
        config,
        lambda i: copy.deepcopy(reference.language_model.model.layers[i]),
        layout=maximum,
        input_factory=lambda: FirstStageInputs(reference.config, reference.language_model.config, config, shape=shape),
        output_factory=lambda: LastStageOutputs(reference.config, reference.language_model.config, config, shape=shape),
    ).to_empty(device="cuda")
    manifest = stage_parameter_manifest(stage)
    load_canonical_stage(stage, {"net." + name: value for name, value in reference.state_dict().items()}, manifest)
    if args.compile:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        for key, layer in stage.layers.items():
            stage.layers[key] = torch.compile(checkpoint_wrapper(layer), fullgraph=True, dynamic=False)
    manifest = stage_parameter_manifest(stage)
    driver = ModelParallelDriver(
        stage,
        topology,
        OptimizerConfig(
            lr=1e-3, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.99, clip_grad=0.3, use_distributed_optimizer=True
        ),
    )
    for group in driver.optimizer.param_groups:
        group["weight_decay"] = 0.01
    native = FusedAdam(
        [p for p in reference.parameters() if p.requires_grad],
        lr=1e-3,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        master_weights=False,
        capturable=False,
    )
    for group in native.param_groups:
        group["step"] = 7
    for index, param in enumerate(reference.parameters()):
        if param.requires_grad:
            torch.manual_seed(4000 + index)
            native.state[param] = {
                "step": torch.tensor(7.0),
                "exp_avg": torch.randn_like(param) * 0.01,
                "exp_avg_sq": torch.rand_like(param) * 0.01 + 0.001,
            }
    reference_params = dict(reference.named_parameters())
    for entry in adam_parameter_states(stage, driver.optimizer):
        spec = manifest[entry.name]
        ref = reference_params[spec.canonical_name.removeprefix("net.")]
        initialize_adam_state(
            entry,
            step=7,
            master=spec.select(ref.detach()),
            exp_avg=spec.select(native.state[ref]["exp_avg"]),
            exp_avg_sq=spec.select(native.state[ref]["exp_avg_sq"]),
        )
    metadata = PipelineMetadata()
    errors = {}

    def loss(outputs):
        return sum(t.square().mean() for key in ("preds_vision", "preds_action") for t in outputs[key])

    for update in range(4):
        native.zero_grad(set_to_none=True)
        slots = logical_slots(topology, update=update, data_parallel_size=1, data_parallel_rank=0)
        sequences, expected = {}, {}
        for slot in slots:
            identity = (slot.update, slot.micro_step, slot.logical_rank)
            torch.manual_seed(9000 + update * 12 + slot.micro_step * 4 + slot.logical_rank)
            sequence = make_sequence(
                action=not (update == 1 and slot.logical_rank % 2),
                noisy_vision=update != 2,
                noisy_action=update != 2,
            )
            sequence.to_cuda()
            sequences[identity] = sequence
            output = reference(sequence)
            (loss(output) / 12).backward()
            expected[identity] = {
                key: [tensor.detach() for tensor in output[key]] for key in ("preds_vision", "preds_action")
            }

        def batches():
            for slot in slots:
                identity = (slot.update, slot.micro_step, slot.logical_rank)
                if ps.is_pipeline_first_stage():
                    batch = (
                        {"slot_id": identity, "packed_sequence": sequences[identity]}
                        if ps.get_tensor_model_parallel_rank() == 0
                        else None
                    )
                    batch = broadcast_batch(batch, ps.get_tensor_model_parallel_group())
                    batch["publish_metadata"] = metadata.send
                    yield batch
                else:
                    yield metadata.receive(expected_slot=identity)

        def loss_fn(output, batch):
            for key in ("preds_vision", "preds_action"):
                for value, ref in zip(output[key], expected[batch["slot_id"]][key], strict=True):
                    torch.testing.assert_close(value, ref, atol=2e-6, rtol=2e-5)
            value = loss(output)
            return value, {"loss": value.detach()}

        driver.forward_backward(batches(), loss_fn, metadata=metadata)
        metadata.drain()
        for entry in adam_parameter_states(stage, driver.optimizer):
            spec = manifest[entry.name]
            ref = reference_params[spec.canonical_name.removeprefix("net.")]
            actual = entry.select(entry.model_parameter.main_grad)
            target = entry.select(spec.select(ref.grad if ref.grad is not None else torch.zeros_like(ref)))
            relative = (actual.double() - target.double()).norm() / target.double().norm().clamp_min(1e-20)
            errors[f"{update}/{spec.canonical_name}"] = relative.item()
            assert relative <= 1e-4 or (actual - target).abs().max() <= 2e-6, (entry.name, relative.item())
        ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
        success, actual_norm, _ = driver.step()
        assert success
        torch.testing.assert_close(torch.tensor(actual_norm, device="cuda"), ref_norm, atol=2e-6, rtol=1e-5)
        native.step()
        driver.model.start_param_sync(force_sync=True)
        for entry in adam_parameter_states(stage, driver.optimizer):
            spec = manifest[entry.name]
            ref = reference_params[spec.canonical_name.removeprefix("net.")]
            torch.testing.assert_close(
                entry.master,
                entry.select(spec.select(ref)),
                atol=2e-6,
                rtol=1e-4,
                msg=lambda msg: f"update={update} {spec.canonical_name} used={entry.model_parameter in driver._globally_used}\n{msg}",
            )
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    entry.state[key], entry.select(spec.select(native.state[ref][key])), atol=2e-6, rtol=1e-4
                )
            assert entry.step == 8 + update
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target = args.output.with_name(f"{args.output.stem}.rank{dist.get_rank()}.json")
    target.write_text(
        json.dumps(
            {
                "passed": True,
                "groups": groups,
                "compiled": args.compile,
                "scope": "production I/O, vocab sharding, dummy context, real metadata P2P, synthetic latent inputs",
                "metadata_bytes_sent": metadata.bytes_sent,
                "relative_gradient_l2": errors,
            },
            indent=2,
        )
        + "\n"
    )
    faulthandler.cancel_dump_traceback_later()
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
