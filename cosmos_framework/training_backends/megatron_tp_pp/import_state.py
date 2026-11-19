"""Full-precision stage import from canonical source DCP identities."""

from pathlib import Path

import torch

from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states, initialize_adam_state
from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import stage_parameter_manifest
from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader


@torch.no_grad()
def load_stage_model(stage, source, *, decoder_prefix="net.language_model.model."):
    source = Path(source)
    reader = CanonicalDCPReader(source / "model")
    manifest = stage_parameter_manifest(stage, decoder_prefix=decoder_prefix)
    state = dict(stage.named_parameters()) | dict(stage.named_buffers())
    for name, entry in manifest.items():
        parameter = state[name]
        tensor = reader.read(
            entry.canonical_name, axis=entry.axis, rank=entry.tensor_rank, size=stage.config.tensor_model_parallel_size
        )
        parameter.copy_(tensor)
    normalization = getattr(stage.input_adapter, "action_normalization", None)
    if normalization is not None:
        normalization.validate_contract()
        normalization._reset_transforms()
    return manifest


def restore_stage_adam(stage, optimizer, source, *, decoder_prefix="net.language_model.model."):
    source = Path(source)
    reader = CanonicalDCPReader(source / "optim")
    manifest = stage_parameter_manifest(stage, decoder_prefix=decoder_prefix)
    group_steps = {}
    for entry in adam_parameter_states(stage, optimizer):
        layout = manifest[entry.name]
        canonical = layout.canonical_name
        hyper = {
            key: reader.read(f"param_groups.{canonical}.{key}")
            for key in ("lr", "initial_lr", "bias_correction", "betas", "eps", "weight_decay", "step")
        }
        if (
            not hyper["bias_correction"]
            or tuple(hyper["betas"]) != tuple(entry.group["betas"])
            or hyper["eps"] != entry.group["eps"]
        ):
            raise ValueError(f"Optimizer math is incompatible for {canonical}")
        if (
            abs(float(hyper["weight_decay"]) - entry.group["weight_decay"]) > 1e-12
            or abs(float(hyper["lr"]) - entry.group["lr"]) > 1e-10
        ):
            raise ValueError(f"Build matching source LR/weight-decay groups before import: {canonical}")
        step = int(hyper["step"].item())
        if group_steps.setdefault(id(entry.group), step) != step:
            raise ValueError("Different source steps cannot share a FusedAdam parameter group")
        values = {
            key: reader.read(
                f"state.{canonical}.{key}",
                axis=layout.axis,
                rank=layout.tensor_rank,
                size=stage.config.tensor_model_parallel_size,
            ).cuda()
            for key in ("master_param", "exp_avg", "exp_avg_sq")
        }
        initialize_adam_state(
            entry, step=step, master=values["master_param"], exp_avg=values["exp_avg"], exp_avg_sq=values["exp_avg_sq"]
        )
    return manifest
