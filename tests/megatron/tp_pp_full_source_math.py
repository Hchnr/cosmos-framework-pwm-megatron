# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Full 36-layer source/master/Adam FP32 replay against an unsharded native model.

Run native first, then four-rank tp. Uses small crops of verified real prepared
inputs and either a prediction-square diagnostic loss or native RF targets and
conditioning. Production capacity/data continuity remain separate acceptance
gates. No random replacement weights.
"""

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, _prepare_full_model_instance
from cosmos_framework.training_backends.megatron_tp_pp.planning import batch_fingerprint
from cosmos_framework.training_backends.megatron_tp_pp.source import release_file_cache
from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils.misc import set_random_seed
from tools.compare_tp_pp_runtime_checkpoints import tensor_difference


def validate_reference_seed(reference, seed):
    """Reject references that cannot prove the same requested initial seed.

    Matching seed labels alone do not prove that runtime RNG states match.
    """
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(reference.get("seed")) is not int
        or reference["seed"] != seed
    ):
        raise ValueError("Native reference must record the same requested seed as the TP diagnostic")


def diagnostic_topology(tp, pp=None):
    """Use the historical pipeline size unless the diagnostic requests one."""
    from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig

    if pp is None:
        pp = 1 if tp == 4 else 2
    return ModelParallelConfig(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        global_packed_microbatches_per_update=2,
        logical_data_parallel_size=2,
        reference_micro_steps_per_update=1,
    )


def configuration(toml, *, original_capacity=False, attention_backend="sdpa"):
    config = load_experiment_from_toml(str(toml))
    instance, _, _ = _prepare_full_model_instance(
        config.model.config.vlm_config.model_instance, load_vlm_visual=False, include_generation_tower=True
    )
    language = instantiate(instance["config"])
    view = OmniMoTModel.__new__(OmniMoTModel)
    view.__dict__.update(config=config.model.config, tokenizer_vision_gen=None)
    network = view.build_network_config(language)
    network.action_tower_attention_backend = attention_backend
    if not original_capacity:
        network.fixed_sequence_length_enabled = True
        network.fixed_und_tokens, network.fixed_gen_tokens, network.fixed_action_tokens = 8, 18, 6
        network.action_tower_und_short_tokens = 0
    return config, instance, language, network


def real_crop(path):
    """Retain real text IDs, noised latents, action domains and diffusion times.

    Reduce only the token count: two items, two text/video/action tokens each.
    This crop is explicitly not the source batch's original packing/objective.
    """
    source = torch.load(path, map_location="cpu", weights_only=False)
    if source.vision is None or source.action is None or len(source.vision.tokens) < 2:
        raise ValueError("Full-source math replay needs two real video/action items")
    if len(source.action.tokens) < 2 or source.text_ids.numel() < 4:
        raise ValueError("Insufficient real action/text items")
    videos, actions, domains = [], [], []
    for index in range(2):
        video = source.vision.tokens[index].float()
        if video.ndim != 5 or min(video.shape[-2:]) < 2:
            raise ValueError(f"Unexpected native video payload {video.shape}")
        if video.shape[2] == 1:
            video = video.repeat(1, 1, 2, 1, 1)
        videos.append(video[:, :, :2, :2, :2].contiguous())
        actions.append(source.action.tokens[index][:2].float().contiguous())
        if actions[-1].shape != (2, 128):
            raise ValueError("Expected native Action128 payload")
        domain = source.action.domain_id[index].reshape(-1)
        domains.append(domain[:2].clone() if domain.numel() > 1 else domain.clone())
    vision = ModalityData(
        tokens=videos,
        token_shapes=[(2, 1, 1)] * 2,
        sequence_indexes=torch.tensor([2, 3, 8, 9]),
        mse_loss_indexes=torch.tensor([2, 3, 8, 9]),
        timesteps=source.vision.timesteps[:1].float().repeat(4),
        noisy_frame_indexes=[torch.tensor([0, 1])] * 2,
        condition_mask=[torch.zeros(2, 1, 1)] * 2,
    )
    action = ModalityData(
        tokens=actions,
        token_shapes=[(2,)] * 2,
        sequence_indexes=torch.tensor([4, 5, 10, 11]),
        mse_loss_indexes=torch.tensor([4, 5, 10, 11]),
        domain_id=domains,
        timesteps=source.action.timesteps[:1].float().repeat(4),
        noisy_frame_indexes=[torch.tensor([0, 1])] * 2,
        condition_mask=[torch.zeros(2, 1)] * 2,
    )
    return PackedSequence(
        sample_lens=[6, 6],
        split_lens=[2, 4, 2, 4],
        attn_modes=["causal", "full"] * 2,
        sequence_length=12,
        text_ids=source.text_ids[:4].clone(),
        text_indexes=torch.tensor([0, 1, 6, 7]),
        position_ids=torch.arange(6).repeat(2).expand(3, -1).clone(),
        vision=vision,
        action=action,
    )


def objective(output):
    return sum(t.square().mean() for key in ("preds_vision", "preds_action") for t in output[key])


def parameter_file(root, update, index):
    return root / f"update{update}.parameter{index:04d}.pt"


def runtime_constants(model):
    return {
        name: value.detach().cpu()
        for name, value in model.named_buffers()
        if name.endswith(("rotary_emb.inv_freq", "_timestep_frequencies"))
    }


@torch.no_grad()
def clip_offloaded_gradients(parameters, maximum, *, norm_device):
    """Keep CPU masters while measuring the native CUDA reduction.

    Large CPU FP32 vector norms can lose appreciable summation accuracy.
    Moving one gradient at a time preserves the CUDA reference's norm without
    retaining another full model of gradients on the GPU.
    """
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    norms = [torch.linalg.vector_norm(gradient.to(norm_device), 2) for gradient in gradients]
    norm = torch.linalg.vector_norm(torch.stack(norms), 2)
    coefficient = (maximum / (norm + 1e-6)).clamp(max=1).cpu()
    torch._foreach_mul_(gradients, coefficient)
    return norm


@torch.no_grad()
def native_cuda_adam_step(optimizer):
    """Stream CPU-owned state through the original native CUDA Adam kernel.

    CPU AdamW is retained as a state/zero_grad container only. Its CPU update
    arithmetic differs from the production fused optimizer and can amplify
    small FP32 differences across a deep three-update replay.
    """
    from cosmos_framework.utils.generator.fused_adam import FusedAdam

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = optimizer.state[parameter]
            local = torch.nn.Parameter(parameter.detach().cuda())
            local.grad = parameter.grad.cuda()
            fused = FusedAdam(
                [local],
                lr=group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                bias_correction=True,
                capturable=True,
                master_weights=False,
            )
            fused.param_groups[0]["step"] = torch.tensor([int(state["step"].item())], dtype=torch.int32, device="cuda")
            fused.state[local] = {field: state[field].cuda() for field in ("exp_avg", "exp_avg_sq")}
            fused.step()
            parameter.copy_(local.cpu())
            for field in ("exp_avg", "exp_avg_sq"):
                state[field].copy_(fused.state[local][field].cpu())
            state["step"].fill_(int(fused.param_groups[0]["step"].item()))
            del fused, local


def native(args, config, instance, network, inputs):
    from tp_pp_real_rf_crop import move_batch

    from cosmos_framework.training_backends.megatron_tp_pp.preparation import GenerationLoss

    if args.objective == "native_rf" and args.native_device != "cuda":
        raise ValueError("Native RF replay uses the original CUDA loss helper")
    loss_module = GenerationLoss(config.model.config) if args.objective == "native_rf" else None
    if args.native_device == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with torch.device("meta"):
        reference = Cosmos3VFMNetwork(instantiate(instance), network)
    reference = reference.float().to_empty(device="cpu")
    # Match the production/native CUDA initialization, including nonpersistent
    # RoPE frequencies. CPU pow followed by a CUDA copy differs by a few ULPs
    # from CUDA pow and is not the same numerical input to a deep decoder.
    reference.language_model.model.rotary_emb.init_weights(buffer_device=torch.device(args.native_device))
    for embedder in (reference.time_embedder, reference.action_time_embedder):
        embedder._timestep_frequencies = embedder._build_timestep_frequencies(
            embedder.frequency_embedding_size, max_period=10000, device="cpu"
        )
    manifest = json.loads((args.bundle / "manifest.json").read_text())["parameters"]
    source = reference.state_dict()
    if {"net." + name for name in source} != manifest.keys():
        raise ValueError("Native full-model canonical key coverage mismatch")
    groups, trainable = {}, []
    parameters = dict(reference.named_parameters())
    with torch.no_grad():
        for name, value in source.items():
            entry = manifest["net." + name]
            if name.startswith("action_normalization."):
                continue
            pieces = [
                torch.load(args.bundle / shard["path"], weights_only=False, mmap=True) for shard in entry["shards"]
            ]
            active = bool(entry["optimizer_fields"])
            field = "master_param" if active else "model"
            values = [piece["state"][field] for piece in pieces]
            value.copy_(values[0] if entry["axis"] is None else torch.cat(values, dim=entry["axis"]))
            if name in parameters:
                parameters[name].requires_grad_(active)
            if active:
                hyper = pieces[0]["hyperparameters"]
                signature = (hyper["lr"], hyper["weight_decay"], tuple(hyper["betas"]), hyper["eps"])
                group = groups.setdefault(
                    signature,
                    {
                        "params": [],
                        "lr": signature[0],
                        "weight_decay": signature[1],
                        "betas": signature[2],
                        "eps": signature[3],
                    },
                )
                group["params"].append(parameters[name])
                trainable.append((name, parameters[name], entry, hyper))
            del pieces, values
    reference.action_normalization.reset_contract()
    reference.action_normalization.validate_contract()
    optimizer = torch.optim.AdamW(list(groups.values()), foreach=False)
    for name, param, entry, hyper in trainable:
        pieces = [torch.load(args.bundle / shard["path"], weights_only=False, mmap=True) for shard in entry["shards"]]
        state = {"step": hyper["step"].float().reshape(()).cpu().clone()}
        for field in ("exp_avg", "exp_avg_sq"):
            values = [piece["state"][field] for piece in pieces]
            state[field] = values[0].clone() if entry["axis"] is None else torch.cat(values, dim=entry["axis"])
        optimizer.state[param] = state
        del pieces, values, state
    release_file_cache(args.bundle / shard["path"] for entry in manifest.values() for shard in entry["shards"])
    gpu_leaf_handles = None
    if args.gpu_leaf_reference:
        from tp_pp_gpu_leaf_reference import place_gpu_trainables

        del source
        gpu_leaf_handles = place_gpu_trainables(reference)
    args.reference_dir.mkdir(parents=True, exist_ok=False)
    constants = runtime_constants(reference)
    torch.save(constants, args.reference_dir / "runtime_constants.pt")
    names = [name for name, _, _, _ in trainable]
    records = []
    for update in range(3):
        optimizer.zero_grad(set_to_none=True)
        outputs, slot_losses = [], []
        with sdpa_kernel(SDPBackend.MATH):
            for replay in inputs:
                batch = move_batch(copy.deepcopy(replay), "cuda") if args.objective == "native_rf" else None
                sequence = batch["packed_sequence"] if batch is not None else copy.deepcopy(replay)
                if args.native_device == "cuda":
                    if batch is None:
                        sequence.to_cuda()
                    # All forward/backward GEMMs and SDPA are native unsharded
                    # CUDA math. CPU masters and offloaded saved tensors keep
                    # the full FP32 reference within one 80-GB H100.
                    save_context = torch.autograd.graph.save_on_cpu(pin_memory=True)
                    if args.preserve_saved_layout:
                        from tools.tp_pp_activation_offload import saved_tensor_cpu_offload

                        save_context = saved_tensor_cpu_offload()
                    with save_context:
                        if args.gpu_leaf_reference:
                            output = reference(sequence)
                        else:
                            cuda_parameters = {name: param.cuda() for name, param in reference.named_parameters()}
                            cuda_buffers = {name: value.cuda() for name, value in reference.named_buffers()}
                            output = torch.func.functional_call(
                                reference, (cuda_parameters, cuda_buffers), (sequence,), strict=True
                            )
                        slot_loss = loss_module(output, batch)[0] if loss_module is not None else objective(output)
                        loss = slot_loss / len(inputs)
                    if not args.gpu_leaf_reference:
                        del cuda_parameters, cuda_buffers
                    loss.backward()
                else:
                    output = reference(sequence)
                    slot_loss = objective(output)
                    (slot_loss / len(inputs)).backward()
                slot_losses.append(float(slot_loss.detach().item()))
                outputs.append(
                    {key: [t.detach().cpu().clone() for t in output[key]] for key in ("preds_vision", "preds_action")}
                )
        # Save raw gradients before the native global clipping operation.
        for index, (_, param, _, _) in enumerate(trainable):
            torch.save(
                {"gradient": param.grad.detach().cpu() if param.grad is not None else None},
                parameter_file(args.reference_dir, update, index),
            )
        if args.gpu_leaf_reference:
            from cosmos_framework.callbacks.grad_clip import _clip_grads_with_global_norm, _total_norm_by_mesh

            grouped = {"default": [p for p in reference.parameters() if p.grad is not None]}
            norm, _ = _total_norm_by_mesh(grouped, error_if_nonfinite=True)
            _clip_grads_with_global_norm(grouped, float(config.trainer.callbacks.grad_clip.clip_norm), norm)
        else:
            norm = clip_offloaded_gradients(
                reference.parameters(),
                float(config.trainer.callbacks.grad_clip.clip_norm),
                norm_device=args.native_device,
            )
        if args.gpu_leaf_reference:
            from tp_pp_gpu_leaf_reference import step_gpu_leaves

            step_gpu_leaves(optimizer)
        elif args.native_optimizer == "native_cuda":
            native_cuda_adam_step(optimizer)
        else:
            optimizer.step()
        for index, (_, param, _, _) in enumerate(trainable):
            path = parameter_file(args.reference_dir, update, index)
            saved = torch.load(path, weights_only=False)
            saved.update(master=param.detach().cpu(), **optimizer.state[param])
            torch.save(saved, path)
        records.append({"norm": norm.item(), "outputs": outputs, "slot_losses": slot_losses})
        torch.save(records, args.reference_dir / "outputs.pt")
        release_file_cache(parameter_file(args.reference_dir, update, index) for index in range(len(trainable)))
        print(json.dumps({"native_completed_update": update, "grad_norm": norm.item()}), flush=True)
    report = {
        "passed": True,
        "seed": args.seed,
        "rng_state_evidence": "not_recorded",
        "names": names,
        "input_hashes": [batch_fingerprint(s) for s in inputs],
        "runtime_constants_sha256": batch_fingerprint(constants),
        "scope": (
            f"unsharded native CUDA FP32 GPU trainable leaves, CPU moments and streamed frozen weights, actual 36-layer source, {args.objective}"
            if args.gpu_leaf_reference
            else f"unsharded native {args.native_device} FP32, CPU masters/Adam, actual 36-layer source, cropped real inputs, {args.objective}"
        ),
        "objective": args.objective,
        "retain_full_text": args.retain_full_text,
        "clipping_norm_device": args.native_device,
        "native_optimizer": args.native_optimizer,
        "gpu_leaf_reference": args.gpu_leaf_reference,
        "preserve_saved_tensor_dense_layout": args.preserve_saved_layout,
        "clipping_implementation": "native_grad_clip" if args.gpu_leaf_reference else "offloaded_cuda_norm",
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated() if args.native_device == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "original_capacity": args.original_capacity,
        "attention_backend": args.attention_backend,
    }
    (args.reference_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    if gpu_leaf_handles is not None:
        from tp_pp_gpu_leaf_reference import close_gpu_trainables

        close_gpu_trainables(*gpu_leaf_handles)


def install_decoder_dtype_observer(stage):
    """Observe decoder boundary tensors during this math diagnostic only."""
    observations = {}
    active_update = [None]
    handles = []
    for layer_name, layer in stage.layers.items():

        def observe(_module, _inputs, output, layer_name=layer_name):
            if active_update[0] is None or not isinstance(output, tuple) or not isinstance(output[0], dict):
                raise RuntimeError("Unexpected decoder activation observation")
            pack = output[0]
            for stream in ("causal_seq", "full_only_seq", "action_seq"):
                value = pack.get(stream)
                if not torch.is_tensor(value) or not value.is_floating_point():
                    raise RuntimeError(f"Missing floating decoder stream {stream}")
                key = (active_update[0], layer_name, stream)
                entry = observations.setdefault(
                    key, {"dtype_counts": {}, "autocast_counts": {}, "matmul_tf32_counts": {}, "cudnn_tf32_counts": {}}
                )
                for field, observed in (
                    ("dtype_counts", str(value.dtype)),
                    ("autocast_counts", str(torch.is_autocast_enabled("cuda"))),
                    ("matmul_tf32_counts", str(torch.backends.cuda.matmul.allow_tf32)),
                    ("cudnn_tf32_counts", str(torch.backends.cudnn.allow_tf32)),
                ):
                    entry[field][observed] = entry[field].get(observed, 0) + 1

        handles.append(layer.register_forward_hook(observe))
    return observations, active_update, handles


def tp(args, config, language, network, inputs):
    from megatron.core import parallel_state as ps
    from megatron.core.optimizer import OptimizerConfig
    from tp_pp_real_rf_crop import move_batch

    from cosmos_framework.model.generator.megatron_tp_pp.factory import nano_layer_factory
    from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs, LastStageOutputs
    from cosmos_framework.model.generator.megatron_tp_pp.payload import PayloadLayout
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from cosmos_framework.training_backends.megatron_tp_pp.bundle import CanonicalBundle
    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver
    from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states
    from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import stage_parameter_manifest
    from cosmos_framework.training_backends.megatron_tp_pp.preparation import GenerationLoss
    from cosmos_framework.training_backends.megatron_tp_pp.runtime import initialize, transformer_config
    from cosmos_framework.training_backends.megatron_tp_pp.transport import PipelineMetadata, broadcast_batch

    # Reject an already-complete incompatible reference before materializing
    # the full model. A concurrently generated reference is checked again below.
    try:
        existing_reference = json.loads((args.reference_dir / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    else:
        validate_reference_seed(existing_reference, args.seed)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    topology = diagnostic_topology(args.tp, getattr(args, "pp", None))
    initialize(topology, seed=args.seed)
    text = getattr(language, "text_config", language)
    mcore = transformer_config(
        topology, num_layers=36, hidden_size=4096, num_attention_heads=32, variable_seq_lengths=True
    )
    stage = PipelineStage(
        mcore,
        nano_layer_factory(text),
        layout=PayloadLayout(((network.fixed_und_tokens, 4096), (12, 4096), (6, 1280)), context_width=5376),
        input_factory=lambda: FirstStageInputs(network, language, mcore),
        output_factory=lambda: LastStageOutputs(network, language, mcore),
    ).to_empty(device="cuda")
    if args.native_linear_autograd:
        from tp_pp_native_linear_autograd import NativeLinearAutograd

        from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear

        for module in stage.modules():
            if isinstance(module, TensorParallelLinear):
                module.__dict__["_eager_linear"] = NativeLinearAutograd
    bundle_manifest_path = args.bundle / "manifest.json"
    bundle_manifest_sha256 = hashlib.sha256(bundle_manifest_path.read_bytes()).hexdigest()
    if args.tp == 1:

        class UnshardedDiagnosticBundle(CanonicalBundle):
            """Read-only full tensors for PP/DP isolation; production import stays strict."""

            def read(self, name, tensor_rank):
                axis = self.manifest["parameters"][name]["axis"]
                left = super().read(name, 0)
                if axis is None:
                    return left
                right = super().read(name, 1)
                return {
                    **left,
                    "state": {
                        key: torch.cat((value, right["state"][key]), dim=axis) for key, value in left["state"].items()
                    },
                }

        bundle = UnshardedDiagnosticBundle(args.bundle, replace(topology, tensor_model_parallel_size=2))
    else:
        bundle = CanonicalBundle(args.bundle, topology)
    # Normalization buffers use the native current append-only registry.
    from omegaconf import OmegaConf

    bundle.validate_action128_extension(
        {
            "version": 1,
            "spec": OmegaConf.to_container(config.model.config.action128_spec, resolve=True),
            "vision_spatial_position_mode": network.vision_spatial_position_mode,
        }
    )
    bundle.load_model(stage)
    manifest = stage_parameter_manifest(stage)
    with torch.no_grad():
        for name, param in stage.named_parameters():
            if param.requires_grad:
                spec = manifest[name]
                param.copy_(bundle.read(spec.canonical_name, spec.tensor_rank)["state"]["master_param"])
    opt = OptimizerConfig(
        lr=float(config.optimizer.lr),
        weight_decay=float(config.optimizer.weight_decay),
        adam_beta1=float(config.optimizer.betas[0]),
        adam_beta2=float(config.optimizer.betas[1]),
        adam_eps=float(config.optimizer.eps),
        clip_grad=float(config.trainer.callbacks.grad_clip.clip_norm),
        use_distributed_optimizer=True,
    )
    driver = ModelParallelDriver(stage, topology, opt, config_overrides=bundle.optimizer_overrides(stage, opt))
    bundle.load_optimizer(stage, driver.optimizer)
    if hashlib.sha256(bundle_manifest_path.read_bytes()).hexdigest() != bundle_manifest_sha256:
        raise ValueError("Canonical bundle manifest changed during optimizer initialization")
    owned_parameter_fragments = [
        {
            "name": manifest[entry.name].canonical_name,
            "axis": manifest[entry.name].axis,
            "offset": manifest[entry.name].offset,
            "local_shape": manifest[entry.name].local_shape,
            "start": entry.start,
            "end": entry.end,
        }
        for entry in adam_parameter_states(stage, driver.optimizer)
    ]
    activation_dtypes, observed_update, dtype_hook_handles = install_decoder_dtype_observer(stage)
    if args.native_adam_arithmetic:
        # Candidate arithmetic alignment, without changing production defaults
        # or enabling graph capture. Native CUDA Adam uses this TE kernel.
        def native_groups(optimizer, positional, keywords):
            for group in optimizer.param_groups:
                if not group["params"]:
                    continue
                device = group["params"][0].device
                group["lr"] = torch.as_tensor(group["lr"], dtype=torch.float32, device=device)
                group["step"] = torch.as_tensor(group["step"], dtype=torch.int32, device=device).reshape(1)

        for optimizer in getattr(driver.optimizer, "chained_optimizers", [driver.optimizer]):
            optimizer.optimizer.capturable = True
            optimizer.optimizer.register_step_pre_hook(native_groups)
    dist.barrier()
    if dist.get_rank() == 0:
        bundle.release_import_cache()
    dist.barrier()
    deadline = time.monotonic() + 7200
    while True:
        try:
            reference = json.loads((args.reference_dir / "manifest.json").read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Native full-source reference did not complete within two hours")
        print(json.dumps({"rank": dist.get_rank(), "waiting_for_native_reference": True}), flush=True)
        time.sleep(30)
    validate_reference_seed(reference, args.seed)
    input_hashes = [batch_fingerprint(s) for s in inputs]
    assert reference["input_hashes"] == input_hashes
    assert reference.get("original_capacity", False) == args.original_capacity
    assert reference.get("attention_backend", "sdpa") == args.attention_backend
    if ps.is_pipeline_first_stage():
        assert reference["runtime_constants_sha256"] == batch_fingerprint(runtime_constants(stage.input_adapter)), (
            "Native and pipeline nonpersistent runtime constants differ"
        )
    indices = {name: index for index, name in enumerate(reference["names"])}
    records = torch.load(args.reference_dir / "outputs.pt", weights_only=False)
    reference_prediction_counts = [
        [{key: len(values) for key, values in output.items()} for output in record["outputs"]] for record in records
    ]
    metadata = PipelineMetadata()
    errors = []
    summaries = []
    gate_failures, compute_parameter_checks = [], []
    norm_checks, step_checks, gradient_usage_checks = [], [], []
    parameter_errors = []
    state_errors = []
    output_errors, consumed_slots, loss_checks = [], [], []
    loss_module = GenerationLoss(config.model.config) if args.objective == "native_rf" else None
    for update in range(3):
        observed_update[0] = update
        if args.replay_native_state and update > 0:
            # Diagnostic only: evaluate each subsequent gradient at exactly the
            # reference's previous master/moment state. This separates a local
            # differentiation error from amplification of prior FP32 rounding.
            # It is deliberately excluded from continuous-trajectory acceptance.
            with torch.no_grad():
                for entry in adam_parameter_states(stage, driver.optimizer):
                    spec = manifest[entry.name]
                    index = indices[spec.canonical_name.removeprefix("net.")]
                    saved = torch.load(
                        parameter_file(args.reference_dir, update - 1, index), weights_only=False, mmap=True
                    )
                    for field in ("master", "exp_avg", "exp_avg_sq"):
                        target = entry.select(spec.select(saved[field]))
                        destination = entry.master if field == "master" else entry.state[field]
                        destination.copy_(target)
                    if entry.group_step:
                        entry.group["step"] = int(saved["step"].item())
                    else:
                        entry.state["step"].fill_(saved["step"].item())
                for optimizer in getattr(driver.optimizer, "chained_optimizers", [driver.optimizer]):
                    optimizer._copy_main_params_to_model_params()
                driver.model.start_param_sync(force_sync=True)
            print(json.dumps({"rank": dist.get_rank(), "native_state_replayed_before_update": update}), flush=True)

        def batches():
            for index in range(ps.get_data_parallel_rank(), len(inputs), ps.get_data_parallel_world_size()):
                sequence = inputs[index]
                identity = (update, 0, index)
                if ps.is_pipeline_first_stage():
                    batch = None
                    if ps.get_tensor_model_parallel_rank() == 0:
                        if args.objective == "native_rf":
                            batch = move_batch(copy.deepcopy(sequence), "cuda")
                            batch["slot_id"] = identity
                        else:
                            sequence = copy.deepcopy(sequence)
                            sequence.to_cuda()
                            batch = {"slot_id": identity, "packed_sequence": sequence}
                    batch = broadcast_batch(batch, ps.get_tensor_model_parallel_group())
                    batch["publish_metadata"] = metadata.send
                else:
                    batch = metadata.receive(expected_slot=identity)
                if tuple(batch["slot_id"]) != identity:
                    raise ValueError("DP rank consumed an unexpected logical slot")
                consumed_slots.append({"update": update, "index": index, "prepared_input_sha256": input_hashes[index]})
                yield batch

        def loss_fn(output, batch):
            expected = records[update]["outputs"][batch["slot_id"][2]]
            for key in expected:
                for sample_index, (actual, target) in enumerate(zip(output[key], expected[key], strict=True)):
                    difference = tensor_difference(
                        target,
                        actual.cpu(),
                        relative_tolerance=1e-4,
                        absolute_tolerance=2e-6,
                        exact=False,
                        local_windows=False,
                    )
                    output_errors.append(
                        {
                            "update": update,
                            "slot": batch["slot_id"],
                            "field": key,
                            "sample_index": sample_index,
                            "actual_dtype": str(actual.dtype),
                            "reference_dtype": str(target.dtype),
                            **difference,
                        }
                    )
            value = loss_module(output, batch)[0] if loss_module is not None else objective(output)
            slot = batch["slot_id"][2]
            expected_losses = records[update].get("slot_losses")
            actual_loss = value.detach().cpu()
            expected_loss = None if expected_losses is None else expected_losses[slot]
            loss_checks.append(
                {
                    "update": update,
                    "index": slot,
                    "actual": actual_loss.item(),
                    "reference": expected_loss,
                    "passed": None
                    if expected_loss is None
                    else bool(torch.isclose(actual_loss, actual_loss.new_tensor(expected_loss), rtol=1e-4, atol=2e-6)),
                }
            )
            return value, {"loss": value.detach()}

        with sdpa_kernel(SDPBackend.MATH):
            save_context = nullcontext()
            if args.activation_cpu_offload:
                from tools.tp_pp_activation_offload import saved_tensor_cpu_offload

                save_context = saved_tensor_cpu_offload()
            with save_context:
                driver.forward_backward(batches(), loss_fn, metadata=metadata)
        metadata.drain()
        stats = torch.zeros(4, dtype=torch.float64, device="cuda")
        for entry in adam_parameter_states(stage, driver.optimizer):
            spec = manifest[entry.name]
            index = indices[spec.canonical_name.removeprefix("net.")]
            saved = torch.load(parameter_file(args.reference_dir, update, index), weights_only=False, mmap=True)
            actual = entry.select(entry.model_parameter.main_grad).cpu()
            reference_gradient = saved["gradient"]
            target = (
                torch.zeros_like(actual)
                if reference_gradient is None
                else entry.select(spec.select(reference_gradient))
            )
            rel = (actual.double() - target.double()).norm() / target.double().norm().clamp_min(1e-20)
            absolute = (actual - target).abs().max()
            difference = tensor_difference(
                target, actual, relative_tolerance=1e-4, absolute_tolerance=2e-6, exact=False, local_windows=False
            )
            errors.append(rel.item())
            parameter_errors.append(
                {
                    "update": update,
                    "name": spec.canonical_name,
                    "relative_l2": rel.item(),
                    "maximum_absolute_error": absolute.item(),
                    "gradient_dtype": str(actual.dtype),
                    "reference_dtype": str(target.dtype),
                    "reference_gradient_present": reference_gradient is not None,
                    "model_parameter_dtype": str(entry.model_parameter.dtype),
                    "comparison_relative_l2": difference.get("relative_l2"),
                    "comparison_maximum_absolute_error": difference.get("maximum_absolute_error"),
                    "reference_squared_l2": difference.get("reference_squared_l2"),
                    "nonfinite": difference.get("nonfinite", False),
                    "passed": difference["passed"],
                }
            )
            a, b = actual.double(), target.double()
            multiplicity = topology.tensor_model_parallel_size if spec.axis is None else 1
            stats += (
                torch.tensor(
                    [a.square().sum(), b.square().sum(), (a * b).sum(), (a - b).square().sum()],
                    dtype=torch.float64,
                    device="cuda",
                )
                / multiplicity
            )
        dist.all_reduce(stats)
        aa, bb, dot, dd = stats.cpu().tolist()
        finite = all(math.isfinite(value) for value in (aa, bb, dot, dd))
        relative_l2 = (dd / bb) ** 0.5 if finite and bb > 0 else None
        cosine = dot / (aa * bb) ** 0.5 if finite and aa > 0 and bb > 0 else None
        summary = {"relative_l2": relative_l2, "cosine": cosine}
        summaries.append(summary)
        failed_parameters = [item for item in parameter_errors if item["update"] == update and not item["passed"]]
        failed_outputs = [item for item in output_errors if item["update"] == update and not item["passed"]]
        failed_losses = [item for item in loss_checks if item["update"] == update and item["passed"] is False]
        global_gradient_passed = (
            relative_l2 is not None and cosine is not None and relative_l2 <= 1e-4 and cosine >= 0.999999
        )
        failures = torch.tensor(
            len(failed_parameters) + len(failed_outputs) + len(failed_losses) + int(not global_gradient_passed),
            device="cuda",
        )
        dist.all_reduce(failures)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        observation_path = args.output.with_name(
            f"{args.output.stem}.update{update}.observation.rank{dist.get_rank()}.json"
        )
        observation_path.write_text(
            json.dumps(
                {
                    "passed": failures.item() == 0,
                    "scope": "complete pre-clip gradient/output observation; optimizer state gate not yet checked",
                    "seed": args.seed,
                    "rng_state_evidence": "not_recorded",
                    "objective": args.objective,
                    "runtime_constants_sha256": reference["runtime_constants_sha256"],
                    "update": update,
                    "replay_native_state": args.replay_native_state,
                    "continuous_trajectory": not args.replay_native_state,
                    "gradient_summary": summary,
                    "failed_parameter_shards": failed_parameters,
                    "parameter_gradient_errors": [item for item in parameter_errors if item["update"] == update],
                    "output_errors": [item for item in output_errors if item["update"] == update],
                    "loss_checks": [item for item in loss_checks if item["update"] == update],
                    "production_capacity_576_slot_acceptance": False,
                },
                indent=2,
            )
            + "\n"
        )
        if failures.item():
            gate_failures.append(f"update{update}.gradient_or_output")
        if not args.observe_all_updates:
            assert failures.item() == 0, f"FP32 gradient/output gate failed; complete observation: {observation_path}"
        pre_step_failures = len(gate_failures)
        success, norm, _ = driver.step()
        assert success
        norm_passed = bool(
            torch.isclose(torch.tensor(norm), torch.tensor(records[update]["norm"]), rtol=1e-4, atol=2e-6)
        )
        norm_checks.append(
            {
                "update": update,
                "actual": float(torch.tensor(norm).item()),
                "reference": float(torch.tensor(records[update]["norm"]).item()),
                "passed": norm_passed,
            }
        )
        if not norm_passed:
            gate_failures.append(f"update{update}.norm")
        driver.model.start_param_sync(force_sync=True)
        for entry in adam_parameter_states(stage, driver.optimizer):
            spec = manifest[entry.name]
            index = indices[spec.canonical_name.removeprefix("net.")]
            saved = torch.load(parameter_file(args.reference_dir, update, index), weights_only=False, mmap=True)
            reference_used = saved["gradient"] is not None
            actual_used = entry.model_parameter in driver._globally_used
            gradient_usage_checks.append(
                {
                    "update": update,
                    "name": spec.canonical_name,
                    "actual_used": actual_used,
                    "reference_used": reference_used,
                    "passed": actual_used == reference_used,
                }
            )
            if actual_used != reference_used:
                gate_failures.append(f"update{update}.{spec.canonical_name}.gradient_usage")
            model_master_match = torch.equal(entry.select(entry.model_parameter.detach()), entry.master)
            compute_parameter_checks.append(
                {
                    "update": update,
                    "name": spec.canonical_name,
                    "compute_equals_master": model_master_match,
                    "compute_dtype": str(entry.model_parameter.dtype),
                    "master_dtype": str(entry.master.dtype),
                }
            )
            if not model_master_match:
                gate_failures.append(f"update{update}.{spec.canonical_name}.compute_master")
            for key in ("master", "exp_avg", "exp_avg_sq"):
                actual = entry.master if key == "master" else entry.state[key]
                target = entry.select(spec.select(saved[key]))
                difference = tensor_difference(
                    target,
                    actual.cpu(),
                    relative_tolerance=1e-4,
                    absolute_tolerance=2e-6,
                    exact=False,
                    local_windows=False,
                )
                if not difference["passed"]:
                    gate_failures.append(f"update{update}.{spec.canonical_name}.{key}")
                state_errors.append(
                    {
                        "update": update,
                        "name": spec.canonical_name,
                        "field": key,
                        "actual_dtype": str(actual.dtype),
                        "reference_dtype": str(target.dtype),
                        **difference,
                    }
                )
            step_passed = entry.step == int(saved["step"].item())
            step_checks.append(
                {
                    "update": update,
                    "name": spec.canonical_name,
                    "actual": int(entry.step),
                    "reference": int(saved["step"].item()),
                    "passed": step_passed,
                }
            )
            if not step_passed:
                gate_failures.append(f"update{update}.{spec.canonical_name}.step")
        if not args.observe_all_updates:
            post_step_failures = torch.tensor(len(gate_failures) - pre_step_failures, device="cuda")
            dist.all_reduce(post_step_failures)
            assert post_step_failures.item() == 0, f"FP32 optimizer or gradient usage gate failed in update {update}"
        print(json.dumps({"rank": dist.get_rank(), "tp_completed_update": update, "grad_norm": norm}), flush=True)
    frozen_verified = 0
    for name, parameter in stage.named_parameters():
        if not parameter.requires_grad:
            spec = manifest[name]
            expected = bundle.read(spec.canonical_name, spec.tensor_rank)["state"]["model"]
            torch.testing.assert_close(parameter.cpu(), expected.float(), rtol=0, atol=0)
            frozen_verified += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_failures = torch.tensor(len(gate_failures), dtype=torch.int64, device="cuda")
    dist.all_reduce(total_failures)
    pointwise_gates_passed = total_failures.item() == 0
    passed = pointwise_gates_passed and not args.replay_native_state
    args.output.with_name(f"{args.output.stem}.rank{dist.get_rank()}.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "rank": dist.get_rank(),
                "seed": args.seed,
                "rng_state_evidence": "not_recorded",
                "scope": reference["scope"],
                "objective": args.objective,
                "retain_full_text": args.retain_full_text,
                "original_capacity": args.original_capacity,
                "attention_backend": args.attention_backend,
                "activation_cpu_offload": args.activation_cpu_offload,
                "runtime_constants_sha256": reference["runtime_constants_sha256"],
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "updates": 3,
                "tensor_parallel_size": args.tp,
                "pipeline_parallel_size": topology.pipeline_model_parallel_size,
                "data_parallel_size": ps.get_data_parallel_world_size(),
                "parallel_ranks": {
                    "tp": ps.get_tensor_model_parallel_rank(),
                    "pp": ps.get_pipeline_model_parallel_rank(),
                    "dp": ps.get_data_parallel_rank(),
                },
                "canonical_bundle_manifest_sha256": bundle_manifest_sha256,
                "owned_parameter_fragments": owned_parameter_fragments,
                "prepared_input_hashes": input_hashes,
                "reference_prediction_counts": reference_prediction_counts,
                "microbatches_per_dp_replica": driver.microbatches,
                "consumed_slots": consumed_slots,
                "loss_checks": loss_checks,
                "maximum_parameter_gradient_relative_l2": max(errors),
                "gradient_summary": summaries,
                "parameter_gradient_errors": parameter_errors,
                "gradient_usage_checks": gradient_usage_checks,
                "parameter_state_errors": state_errors,
                "compute_parameter_checks": compute_parameter_checks,
                "norm_checks": norm_checks,
                "step_checks": step_checks,
                "activation_checks": [
                    {"update": u, "layer": layer, "stream": stream, **details}
                    for (u, layer, stream), details in sorted(activation_dtypes.items())
                ],
                "native_dtype_evidence": "historical_reference_not_recorded",
                "observe_all_updates": args.observe_all_updates,
                "replay_native_state": args.replay_native_state,
                "native_adam_arithmetic": args.native_adam_arithmetic,
                "native_linear_autograd": args.native_linear_autograd,
                "continuous_trajectory": not args.replay_native_state,
                "pointwise_gates_passed": pointwise_gates_passed,
                "gate_failures": gate_failures,
                "output_errors": output_errors,
                "frozen_parameters_exact": frozen_verified,
                "production_capacity_576_slot_acceptance": False,
            },
            indent=2,
        )
        + "\n"
    )
    ps.destroy_model_parallel()
    dist.destroy_process_group()
    if args.replay_native_state:
        raise SystemExit("Native-state replay is diagnostic only; continuous trajectory acceptance remains false")
    if not passed:
        raise SystemExit("Full-source three-update observation completed with FAILED fixed gates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "tp"), required=True)
    parser.add_argument("--seed", type=int, default=42, help="Initial seed used by both native and TP diagnostics")
    parser.add_argument("--native-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--native-optimizer", choices=("torch_cpu", "native_cuda"), default="torch_cpu")
    parser.add_argument("--gpu-leaf-reference", action="store_true")
    parser.add_argument("--preserve-saved-layout", action="store_true")
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--pp", type=int, choices=(1, 2), help="Pipeline size. Defaults to 1 for TP4 and 2 otherwise.")
    parser.add_argument("--objective", choices=("prediction_square", "native_rf"), default="prediction_square")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--retain-full-text", action="store_true")
    parser.add_argument("--original-capacity", action="store_true")
    parser.add_argument("--attention-backend", choices=("sdpa", "flex"), default="sdpa")
    parser.add_argument("--activation-cpu-offload", action="store_true")
    parser.add_argument("--native-adam-arithmetic", action="store_true")
    parser.add_argument(
        "--native-linear-autograd",
        action="store_true",
        help="Diagnostic only: isolate native local Linear autograd within unchanged MCore TP/PP/DDP",
    )
    parser.add_argument(
        "--replay-native-state",
        action="store_true",
        help="Diagnostic only: reset to native previous state before updates 1/2; continuous acceptance remains false",
    )
    parser.add_argument(
        "--observe-all-updates",
        action="store_true",
        help="Keep collecting finite updates after a gate failure; final passed remains false and exit is nonzero",
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pp is not None and args.mode != "tp":
        parser.error("--pp applies to the TP/PP diagnostic")
    if args.retain_full_text and args.objective != "native_rf":
        parser.error("Complete native text conditioning requires --objective native_rf")
    if args.original_capacity and not (
        args.objective == "native_rf" and args.retain_full_text and args.attention_backend == "flex"
    ):
        parser.error("Original FP32 capacity requires native_rf, retain-full-text and Flex attention")
    if args.activation_cpu_offload and args.mode != "tp":
        parser.error("TP activation offload applies only to the TP diagnostic")
    if args.replay_native_state and args.mode != "tp":
        parser.error("Native-state replay is a TP/PP diagnostic only")
    if args.native_adam_arithmetic and args.mode != "tp":
        parser.error("Native Adam arithmetic candidate applies only to TP/PP")
    if args.native_linear_autograd and args.mode != "tp":
        parser.error("Native local-linear isolation applies only to TP/PP")
    if args.native_optimizer == "native_cuda" and args.native_device != "cuda":
        parser.error("Native fused CUDA optimizer requires --native-device cuda")
    if args.gpu_leaf_reference and (
        args.mode != "native" or args.native_device != "cuda" or args.native_optimizer != "native_cuda"
    ):
        parser.error("GPU leaf reference requires native CUDA mode and the native CUDA optimizer")
    if args.preserve_saved_layout and (args.mode != "native" or args.native_device != "cuda"):
        parser.error("Saved-tensor layout qualification applies only to the native CUDA reference")
    set_random_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    config, instance, language, network = configuration(
        args.toml, original_capacity=args.original_capacity, attention_backend=args.attention_backend
    )
    if args.objective == "native_rf":
        from tp_pp_real_rf_crop import full_text_rf_crop, original_rf_inputs, real_rf_crop

        crop = (
            original_rf_inputs
            if args.original_capacity
            else full_text_rf_crop
            if args.retain_full_text
            else real_rf_crop
        )
        inputs = [crop(args.prepared_dir / f"rank{rank}.micro0.pt") for rank in (0, 1)]
        if args.retain_full_text and not args.original_capacity:
            maximum_text = max(batch["packed_sequence"].text_ids.numel() for batch in inputs)
            network.fixed_und_tokens = ((maximum_text + 127) // 128) * 128
        if config.model.config.rectified_flow_training_config.sample_level_loss_averaging:
            for batch in inputs:
                batch["weight"] = 1.0
    else:
        inputs = [real_crop(args.prepared_dir / f"rank{rank}.micro0.pt") for rank in (0, 1)]
    if args.mode == "native":
        native(args, config, instance, network, inputs)
    else:
        tp(args, config, language, network, inputs)


if __name__ == "__main__":
    main()
