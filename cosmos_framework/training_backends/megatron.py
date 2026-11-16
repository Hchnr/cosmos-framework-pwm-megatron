"""Megatron-Core FSDP integration preserving Cosmos module and checkpoint names.

The runtime is deliberately held outside the nn.Module tree. Megatron owns the
communication and sharded parameters; the original module remains the public
model, so data, loss, parameter selection and checkpoint FQNs stay unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


def _support_pytree_backward_hooks(module):
    """Adapt MCore 0.20's shallow output inspection to Cosmos SequencePack.

    MCore's custom backward hook only inspects Tensor or flat tuple outputs.
    MoT returns (SequencePack, metadata, cache), so without this adapter no
    pre-backward all-gather fires, and AC recomputes against released storage.
    Keep each hook's position and preserve the public output structure.
    """

    def adapt(original):
        @torch.compiler.disable
        def hook(module, inputs, output):
            leaves, spec = tree_flatten(output)
            updated = original(module, inputs, tuple(leaves))
            return tree_unflatten(list(updated), spec)

        return hook

    for child in module.modules():
        for key, hook in list(child._forward_hooks.items()):
            if getattr(hook, "__name__", "") == "forward_hook" and getattr(hook, "__module__", "").endswith(
                "megatron_fsdp.megatron_fsdp"
            ):
                child._forward_hooks[key] = adapt(hook)


class MegatronRuntime:
    def __init__(
        self,
        module,
        *,
        shard_degree,
        replicate_degree=1,
        unit_types=(),
        sharding_strategy="optim_grads_params",
        main_params_dtype=None,
        main_grads_dtype=None,
        grad_comm_dtype=None,
        initial_master_weights=None,
        compute_dtype=None,
    ):
        from megatron.core.distributed.fsdp.src.megatron_fsdp import fully_shard_model
        from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy

        if not dist.is_initialized():
            raise RuntimeError("Megatron requires an initialized torch.distributed process group")
        if shard_degree * replicate_degree != dist.get_world_size():
            raise ValueError("Megatron shard_degree * replicate_degree must equal WORLD_SIZE")
        mesh = init_device_mesh(
            "cuda", (replicate_degree, shard_degree, 1), mesh_dim_names=("replicate", "shard", "tp")
        )
        kwargs = {}
        if replicate_degree > 1:
            kwargs.update(dp_outer_dim="replicate", hybrid_fsdp_group=dist.group.WORLD)
        if any(".experts." in name for name, _ in module.named_parameters()):
            # HF fused experts use ordinary data parallelism (EP=1). MCore
            # detects their names and requires an explicit expert DP mesh.
            kwargs["expt_device_mesh"] = mesh
            if replicate_degree > 1:
                kwargs["hybrid_fsdp_expt_group"] = dist.group.WORLD
        self.fsdp = fully_shard_model(
            module,
            device_mesh=mesh,
            dp_shard_dim="shard",
            tp_dim="tp",
            fsdp_unit_modules=list(unit_types),
            zero_dp_strategy=sharding_strategy,
            mixed_precision_policy=MixedPrecisionPolicy(
                main_params_dtype=main_params_dtype,
                main_grads_dtype=main_grads_dtype,
                grad_comm_dtype=grad_comm_dtype,
            ),
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            sync_model_each_microbatch=False,
            preproc_state_dict_for_dcp_ckpt=False,
            report_nan_in_param_grad=False,
            **kwargs,
        )
        _support_pytree_backward_hooks(module)
        # Megatron's forward method only swaps optimizer DTensors to compute
        # parameters. Its communication hooks are installed on `module` itself.
        self._forward_hook = module.register_forward_pre_hook(self._before_forward, prepend=True)
        self._input_cast_hooks = []
        if compute_dtype is not None:

            def cast_inputs(module, args, kwargs):
                def cast(value):
                    return (
                        value.to(dtype=compute_dtype)
                        if isinstance(value, torch.Tensor) and value.is_floating_point()
                        else value
                    )

                return tree_map(cast, args), tree_map(cast, kwargs)

            # Match FSDP2 MixedPrecisionPolicy at the root and every wrapped
            # transformer unit. In particular, RoPE cos/sin kwargs are FP32
            # buffers but FSDP2 casts them to the compute dtype at block entry.
            for child in module.modules():
                if child is module or isinstance(child, tuple(unit_types)):
                    self._input_cast_hooks.append(
                        child.register_forward_pre_hook(cast_inputs, with_kwargs=True, prepend=True)
                    )
        module.__dict__["_cosmos_megatron_runtime"] = self

        self._frozen_masters = {}

        def prepare_dcp(module, state_dict, prefix, local_metadata):
            for name, master in self._frozen_masters.items():
                state_dict[prefix + name] = master
            self.prepare_state_dict(state_dict)

        self._state_hook = module.register_state_dict_post_hook(prepare_dcp)

        def load_frozen(module, state, prefix, *args):
            # Frozen FP32 checkpoint values must survive BF16 compute storage.
            # Keep sharded FP32 copies; never gather a full frozen tower here.
            for name, master in self._frozen_masters.items():
                key = prefix + name
                if key not in state:
                    continue
                incoming = state[key]
                master.to_local().copy_(incoming.to_local())
                state[key] = incoming.to(dtype=self._frozen_compute_dtypes[name])

        self._load_hook = module.register_load_state_dict_pre_hook(load_frozen)
        if initial_master_weights is not None:
            self._restore_initial_masters(module, initial_master_weights)

    @torch.no_grad()
    def _restore_initial_masters(self, module, weights):
        # CPU materialization/loading preserves FP32 HF values before compute
        # parameters are cast to BF16 and Megatron allocates optimizer shards.
        state = module.state_dict()
        self._frozen_compute_dtypes = {}
        for name, value in state.items():
            if not isinstance(value, DTensor) or name not in weights:
                continue
            original = weights[name]
            chunk = value.to_local().__create_chunk_list__()[0]
            slices = tuple(slice(o, o + n) for o, n in zip(chunk.offsets, chunk.sizes, strict=True))
            if value.dtype != original.dtype:
                master = value.to(dtype=original.dtype)
                self._frozen_masters[name] = master
                self._frozen_compute_dtypes[name] = value.dtype
            else:
                master = value
            master.to_local().copy_(original[slices])
            state[name] = master
        module.load_state_dict(state)

    def _before_forward(self, module, args):
        self.fsdp._replace_param_with_raw_if_needed()

    @contextmanager
    def accumulate(self, last_microbatch):
        previous = getattr(self.fsdp, "is_last_microbatch", False)
        self.fsdp.is_last_microbatch = last_microbatch
        try:
            yield
            if last_microbatch:
                # Complete reductions before Cosmos gradient clipping / NaN
                # handling. Waiting inside optimizer.step would be too late.
                self.fsdp.finish_grad_sync()
        finally:
            self.fsdp.is_last_microbatch = previous

    def prepare_state_dict(self, state_dict):
        from cosmos_framework.training_backends.checkpoint_metadata import prepare_uneven_state

        return prepare_uneven_state(state_dict, self.fsdp.dist_index.get_fsdp_group())

    def after_step(self):
        self.fsdp.install_optimized_model_weights()

    def zero_grad(self):
        self.fsdp.zero_grad_buffer()

    def _local_gradients(self):
        sharded, replicated = [], []
        for parameter in self.fsdp.module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if isinstance(gradient, DTensor):
                local = gradient.to_local()
                destination = sharded if any(isinstance(p, Shard) for p in gradient.placements) else replicated
            else:
                local, destination = gradient, replicated
            if local.numel():
                destination.append(local)
        return sharded, replicated

    @torch.no_grad()
    def total_grad_norm(self, error_if_nonfinite=False):
        # Unlike FSDP2, different ranks may own different numbers of nonempty
        # parameter shards. DTensor foreach-norm can resolve one collective per
        # parameter, which deadlocks when those list lengths differ. Reduce one
        # FP32 squared sum over the shard group; do not count replicas twice.
        sharded, replicated = self._local_gradients()

        def squared_norm(gradients):
            if not gradients:
                return torch.zeros((), device="cuda", dtype=torch.float32)
            norms = torch._foreach_norm(gradients, 2, dtype=torch.float32)
            return torch.stack(norms).square().sum()

        shard_sum = squared_norm(sharded)
        dist.all_reduce(shard_sum, group=self.fsdp.dist_index.get_fsdp_group())
        norm = (shard_sum + squared_norm(replicated)).sqrt()
        if error_if_nonfinite and not bool(torch.isfinite(norm)):
            raise RuntimeError("Megatron gradient norm is non-finite")
        return norm

    @torch.no_grad()
    def clip_gradients(self, max_norm, total_norm):
        coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        sharded, replicated = self._local_gradients()
        gradients = sharded + replicated
        if gradients:
            torch._foreach_mul_(gradients, coefficient)


def get_runtime(model):
    seen = set()
    while model is not None and id(model) not in seen and len(seen) < 8:
        seen.add(id(model))
        runtime = getattr(model, "__dict__", {}).get("_cosmos_megatron_runtime")
        if runtime is not None:
            return runtime
        nested = getattr(model, "net", None)
        model = nested if nested is not None else getattr(model, "model", None)
    return None
