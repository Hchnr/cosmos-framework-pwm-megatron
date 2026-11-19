# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""One MCore schedule invocation = one optimizer update (no outer GA)."""

import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils.nvtx import configure_nvtx, nvtx_range


def finalize_grads(models, num_tokens=None, **kwargs):
    from megatron.core import parallel_state as ps
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    finalize_model_grads(models, num_tokens, **kwargs)
    # MCore recognizes q_layernorm/k_layernorm names; Cosmos preserves q_norm /
    # k_norm FQNs. Explicitly SUM only their head-local partial gradients.
    partials = [
        p.main_grad
        for model in models
        for p in model.parameters()
        if p.requires_grad and getattr(p, "_cosmos_tp_partial_grad", False)
    ]
    if partials and ps.get_tensor_model_parallel_world_size() > 1:
        flat = _flatten_dense_tensors(partials)
        dist.all_reduce(flat, group=ps.get_tensor_model_parallel_group())
        for grad, reduced in zip(partials, _unflatten_dense_tensors(flat, partials), strict=True):
            grad.copy_(reduced)


class ModelParallelDriver:
    def __init__(
        self,
        stage,
        topology,
        optimizer_config,
        *,
        scheduler=None,
        config_overrides=None,
        gc_trace_root=None,
        native_fp32_norm_order=None,
        _native_norm_consensus_checked=False,
    ):
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.optimizer import get_megatron_optimizer

        if not _native_norm_consensus_checked:
            from .native_fp32_norm import agree_requested

            agree_requested(
                "1" if native_fp32_norm_order is not None else "0",
                dtype=stage.config.params_dtype,
                dp_size=topology.data_parallel_size(dist.get_world_size()),
                optimizer_config=optimizer_config,
                device="cuda",
            )
        if optimizer_config.use_distributed_optimizer != topology.use_distributed_optimizer:
            raise ValueError("DDP and optimizer must agree on distributed optimizer ownership")
        if optimizer_config.bf16 != (stage.config.params_dtype == torch.bfloat16):
            raise ValueError("Model and optimizer BF16/master configuration disagree")
        if stage.config.params_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("TP/PP currently supports FP32 and BF16 compute")
        self.stage, self.topology, self.scheduler = stage, topology, scheduler
        from .memory import MicrobatchGarbageCollector

        self.microbatch_gc_interval = int(os.environ.get("COSMOS_TP_PP_MICROBATCH_GC_INTERVAL", "16"))
        MicrobatchGarbageCollector(self.microbatch_gc_interval)
        self._microbatch_gc_trace_root = Path(gc_trace_root) if gc_trace_root is not None else None
        for name, parameter in stage.named_parameters():
            parameter._cosmos_tp_partial_grad = ".q_norm" in name or ".k_norm" in name
        self.microbatches = topology.microbatches(dist.get_world_size())
        self.model = DistributedDataParallel(
            stage.config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=topology.overlap_grad_reduce,
                overlap_param_gather=topology.overlap_param_gather,
                use_distributed_optimizer=topology.use_distributed_optimizer,
                check_for_nan_in_grad=True,
            ),
            stage,
        )
        self.optimizer = get_megatron_optimizer(
            optimizer_config,
            [self.model],
            config_overrides=config_overrides,
        )
        self._native_fp32_norm_calls = 0
        self._native_fp32_norm_enabled = native_fp32_norm_order is not None
        if self._native_fp32_norm_enabled:
            from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
            from megatron.core.optimizer.optimizer import ChainedOptimizer

            from .native_fp32_norm import gradient_norm, requested, supported_optimizer_config

            requested(
                "1",
                dtype=stage.config.params_dtype,
                dp_size=topology.data_parallel_size(dist.get_world_size()),
                clip_grad=optimizer_config.clip_grad,
            )
            # The pinned MCore ChainedOptimizer.step calls get_grad_norm;
            # other step implementations may bypass the candidate entirely.
            valid_optimizer = isinstance(self.optimizer, ChainedOptimizer)
            try:
                if valid_optimizer:
                    valid_optimizer = all(
                        getattr(child, "is_stub_optimizer", False)
                        or (
                            isinstance(child, DistributedOptimizer)
                            and child.grad_scaler is None
                            and supported_optimizer_config(child.config)
                        )
                        for child in self.optimizer.chained_optimizers
                    )
            except (AttributeError, TypeError, ValueError, OverflowError):
                valid_optimizer = False
            invalid_optimizer = torch.tensor([int(not valid_optimizer)], dtype=torch.int32, device="cuda")
            dist.all_reduce(invalid_optimizer, op=dist.ReduceOp.MAX)
            if invalid_optimizer.item():
                raise ValueError("Native FP32 norm requires the qualified MCore ChainedOptimizer")
            self.optimizer.get_grad_norm = partial(gradient_norm, self, tuple(native_fp32_norm_order))
        from .optimizer_math import align_native_adam_arithmetic

        align_native_adam_arithmetic(self.optimizer)
        # DDP allocates a zero main_grad even when autograd never visits a
        # parameter. Adam must retain native grad=None semantics: otherwise a
        # missing modality still decays its weights and nonzero moments.
        self._parameters = [p for p in stage.parameters() if p.requires_grad]
        self._used = set()
        self._usage_hooks = [p.register_hook(self._mark_used(p)) for p in self._parameters]
        from .optimizer import adam_parameter_states

        # The hook needs identities only. Do not retain Adam state dictionaries:
        # load_state_dict may replace them during a runtime restart.
        fragments = [(entry.model_parameter, entry.master) for entry in adam_parameter_states(stage, self.optimizer)]

        def mask_unused(optimizer, args, kwargs):
            for parameter, master in fragments:
                if parameter not in self._globally_used:
                    master.grad = None
                    if hasattr(master, "decoupled_grad"):
                        master.decoupled_grad = None

        self._optimizer_hooks = [
            opt.optimizer.register_step_pre_hook(mask_unused)
            for opt in getattr(self.optimizer, "chained_optimizers", [self.optimizer])
        ]
        stage.config.finalize_model_grads_func = finalize_grads
        stage.config.no_sync_func = self.model.no_sync
        stage.config.grad_scale_func = self.optimizer.scale_loss
        self.iteration = 0
        self._profile_finished = False

    def _mark_used(self, parameter):
        def hook(gradient):
            if gradient is not None:
                self._used.add(parameter)
            return gradient

        return hook

    def forward_backward(self, batches, loss_fn, *, metadata=None):
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        self.optimizer.zero_grad()
        self.model.zero_grad_buffer()
        self._used.clear()
        profile = None
        profile_iteration = os.environ.get("COSMOS_TP_PP_PROFILE_ITERATION")
        if (
            os.environ.get("COSMOS_TP_PP_PROFILE_SLOTS")
            and not self._profile_finished
            and (profile_iteration is None or self.iteration == int(profile_iteration))
        ):
            profile = tuple(int(value) for value in os.environ["COSMOS_TP_PP_PROFILE_SLOTS"].split(":"))
            if len(profile) != 2 or not 0 <= profile[0] < profile[1] < self.microbatches:
                raise ValueError("COSMOS_TP_PP_PROFILE_SLOTS must be start:end within the update's microbatches")
            configure_nvtx("module")
        forward_index = 0

        def forward_step(iterator, model):
            nonlocal forward_index
            if profile and forward_index == profile[0]:
                torch.cuda.cudart().cudaProfilerStart()
            if profile and forward_index == profile[1]:
                torch.cuda.cudart().cudaProfilerStop()
                self._profile_finished = True
            forward_index += 1
            with nvtx_range("tp_pp.batch_prepare"):
                batch = next(iterator)
            with nvtx_range("tp_pp.forward"):
                output = model(batch)

            def loss(output):
                with nvtx_range("tp_pp.loss"):
                    value, metrics = loss_fn(output, batch)
                # MCore divides exactly once by M; loss_fn supplies explicit
                # reference-slot weights when original rank counts differ.
                return value, metrics

            return output, loss

        kwargs = {}
        if metadata is not None and self.topology.pipeline_model_parallel_size > 1:
            from megatron.core.pipeline_parallel.schedules import _build_default_pg_collection

            kwargs["p2p_communicator"] = metadata.communicator(self.stage.config)
            kwargs["pg_collection"] = _build_default_pg_collection()
        if self.topology.pipeline_model_parallel_size > 1 and not self.stage.config.variable_seq_lengths:
            kwargs["adjust_tensor_shapes_fn"] = lambda recv, send: (
                [self.stage.layout.tensor_shape],
                [self.stage.layout.tensor_shape],
            )
        from .memory import collect_microbatch_cycles

        completed = False
        with collect_microbatch_cycles(self.microbatch_gc_interval) as collector:
            try:
                result = get_forward_backward_func()(
                    forward_step_func=forward_step,
                    data_iterator=iter(batches),
                    model=self.model,
                    num_microbatches=self.microbatches,
                    seq_length=1,
                    micro_batch_size=1,
                    forward_only=False,
                    **kwargs,
                )
                completed = True
                return result
            finally:
                report = {
                    "iteration": self.iteration + 1,
                    "interval": self.microbatch_gc_interval,
                    "schedule_completed": completed,
                    "completed_backwards": collector.completed_backwards if self.microbatch_gc_interval else None,
                    "collections": collector.collections,
                }
                self.microbatch_gc_report = report
                if self._microbatch_gc_trace_root is not None:
                    self._microbatch_gc_trace_root.mkdir(parents=True, exist_ok=True)
                    path = self._microbatch_gc_trace_root / f"microbatch_gc.rank{dist.get_rank()}.jsonl"
                    with path.open("a") as handle:
                        handle.write(json.dumps(report) + "\n")

    def step(self):
        from megatron.core import parallel_state as ps

        used = torch.tensor([p in self._used for p in self._parameters], dtype=torch.int32, device="cuda")
        dist.all_reduce(used, op=dist.ReduceOp.MAX, group=ps.get_data_parallel_group())
        self._globally_used = {p for p, active in zip(self._parameters, used.cpu().tolist(), strict=True) if active}
        finite = torch.ones((), dtype=torch.int32, device="cuda")
        for parameter in self.stage.parameters():
            if parameter.requires_grad:
                finite = torch.minimum(finite, torch.isfinite(parameter.main_grad).all().to(torch.int32))
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            return False, None, None
        norm_calls_before = self._native_fp32_norm_calls
        success, grad_norm, zeros = self.optimizer.step()
        if self._native_fp32_norm_enabled and success and self._native_fp32_norm_calls == norm_calls_before:
            raise RuntimeError("MCore optimizer did not consume the requested native FP32 norm")
        # MCore norm/finiteness collectives span the complete TP/PP model;
        # distributed optimizer additionally accounts for DP-owned shards.
        if success:
            self.iteration += 1
            if self.scheduler is not None:
                self.scheduler.step()
        return success, grad_norm, zeros

    def update(self, batches, loss_fn):
        metrics = self.forward_backward(batches, loss_fn)
        return metrics, self.step()
