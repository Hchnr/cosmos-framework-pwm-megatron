# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Import every real Nano/Action parameter and Adam fragment on four GPUs.

This validates runtime state ownership, not a production-capacity update.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.model.generator.megatron_tp_pp.factory import nano_layer_factory
from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs, LastStageOutputs
from cosmos_framework.model.generator.megatron_tp_pp.payload import PayloadLayout
from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, _prepare_full_model_instance
from cosmos_framework.training_backends.megatron_tp_pp.bundle import CanonicalBundle
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver
from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states
from cosmos_framework.training_backends.megatron_tp_pp.runtime import initialize, transformer_config
from cosmos_framework.training_backends.megatron_tp_pp.scheduler import NativeScheduler
from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader
from cosmos_framework.utils.lazy_config import instantiate


def main():
    from megatron.core import parallel_state as ps
    from megatron.core.optimizer import OptimizerConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    topology = ModelParallelConfig()
    groups = initialize(topology)
    config = load_experiment_from_toml(
        str(args.toml), extra_overrides=["checkpoint.action128_preserve_source_normalization=true"]
    )
    instance, _, _ = _prepare_full_model_instance(
        config.model.config.vlm_config.model_instance, load_vlm_visual=False, include_generation_tower=True
    )
    language_config = instantiate(instance["config"])
    text = getattr(language_config, "text_config", language_config)
    view = OmniMoTModel.__new__(OmniMoTModel)
    view.__dict__.update(config=config.model.config, tokenizer_vision_gen=None)
    network_config = view.build_network_config(language_config)
    bundle = CanonicalBundle(args.bundle, topology)
    _, extended = bundle.validate_action128_extension(
        {
            "version": 1,
            "spec": OmegaConf.to_container(config.model.config.action128_spec, resolve=True),
            "vision_spatial_position_mode": network_config.vision_spatial_position_mode,
        }
    )
    dtype = getattr(torch, args.dtype)
    mcore = transformer_config(
        topology,
        num_layers=text.num_hidden_layers,
        hidden_size=text.hidden_size,
        num_attention_heads=text.num_attention_heads,
        dtype=dtype,
        variable_seq_lengths=True,
    )
    stage = (
        PipelineStage(
            mcore,
            nano_layer_factory(text),
            layout=PayloadLayout(((8, 4096), (12, 4096), (6, 1280)), context_width=5376),
            input_factory=lambda: FirstStageInputs(network_config, language_config, mcore),
            output_factory=lambda: LastStageOutputs(network_config, language_config, mcore),
        )
        .to(dtype=dtype)
        .to_empty(device="cuda")
    )
    manifest = bundle.load_model(stage)
    state = dict(stage.named_parameters()) | dict(stage.named_buffers())
    model_verified = 0
    for name, entry in manifest.items():
        if extended and entry.canonical_name.startswith("net.action_normalization."):
            continue  # Validated expanded buffers intentionally differ from the old registry.
        expected = bundle.read(entry.canonical_name, entry.tensor_rank)["state"]["model"]
        torch.testing.assert_close(state[name].detach().cpu(), expected.to(state[name].dtype), rtol=0, atol=0)
        model_verified += 1
    optimizer_config = OptimizerConfig(
        lr=float(config.optimizer.lr),
        weight_decay=float(config.optimizer.weight_decay),
        adam_beta1=float(config.optimizer.betas[0]),
        adam_beta2=float(config.optimizer.betas[1]),
        adam_eps=float(config.optimizer.eps),
        use_distributed_optimizer=True,
        bf16=dtype == torch.bfloat16,
    )
    overrides = bundle.optimizer_overrides(stage, optimizer_config)
    driver = ModelParallelDriver(stage, topology, optimizer_config, config_overrides=overrides)
    bundle.load_optimizer(stage, driver.optimizer)
    fragments, elements = 0, 0
    for entry in adam_parameter_states(stage, driver.optimizer):
        layout = manifest[entry.name]
        saved = bundle.read(layout.canonical_name, layout.tensor_rank)
        for key in ("master_param", "exp_avg", "exp_avg_sq"):
            value = entry.master if key == "master_param" else entry.state[key]
            assert value.dtype == torch.float32
            torch.testing.assert_close(value.detach().cpu(), entry.select(saved["state"][key]), rtol=0, atol=0)
        assert entry.step == int(saved["hyperparameters"]["step"].item())
        fragments += 1
        elements += entry.master.numel()
    scheduler_kwargs = OmegaConf.to_container(config.scheduler, resolve=True)
    scheduler_type = scheduler_kwargs.pop("lr_scheduler_type")
    scheduler_kwargs.pop("_target_")
    scheduler_kwargs.pop("optimizer", None)
    driver.scheduler = NativeScheduler(driver.optimizer, scheduler_type, scheduler_kwargs)
    source_epoch = driver.scheduler.import_source(CanonicalDCPReader(args.bundle / "logical_state" / "scheduler"))
    iteration = CanonicalDCPReader(args.bundle / "logical_state" / "trainer").read("iteration")
    assert source_epoch == iteration
    torch.cuda.synchronize()
    report = {
        "passed": True,
        "scope": "full real model and Adam import; no training update",
        "groups": groups,
        "dtype": args.dtype,
        "local_layers": list(stage.layers.keys()),
        "model_tensors_verified": model_verified,
        "optimizer_fragments_verified": fragments,
        "optimizer_owned_elements": elements,
        "scheduler_last_epoch": source_epoch,
        "scheduler_group_lrs": [float(group["lr"]) for group in driver.optimizer.param_groups],
        "action128_append_only_extension": extended,
        "local_parameter_elements": sum(p.numel() for p in stage.parameters()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "production_capacity_576_slot_acceptance": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name(f"{args.output.stem}.rank{dist.get_rank()}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
