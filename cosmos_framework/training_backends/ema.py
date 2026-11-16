"""FP32 EMA shadows with exactly the online Megatron optimizer-shard layout."""

import torch
from torch import nn

from cosmos_framework.utils.generator.dtensor_helper import DTensorFastEmaModelUpdater


def _canonical(name):
    return name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")


@torch.no_grad()
def shard_ema_model(runtime, ema):
    """Replace a CPU EMA's parameters with FP32 copies of matching local shards.

    EMA is never forwarded by the Cosmos trainer: ema_scope copies its values
    into the online network. It therefore needs neither gather hooks nor gradient
    buffers. Preserve its original values rather than rounding via online BF16.
    """
    online = {_canonical(name): value for name, value in runtime.fsdp.module.state_dict().items()}
    aliases = {}
    for name, parameter in ema.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for parameter in list(ema.parameters()):
        names = aliases[id(parameter)]
        reference = online[_canonical(names[0])]
        chunks = getattr(reference._local_tensor, "__create_chunk_list__", None)
        if chunks is None:
            slices = tuple(slice(None) for _ in reference.shape)
        else:
            chunk = chunks()[0]
            slices = tuple(slice(o, o + n) for o, n in zip(chunk.offsets, chunk.sizes, strict=True))
        shadow = reference.detach().to(dtype=torch.float32, copy=True)
        shadow.to_local().copy_(parameter.detach()[slices])
        replacement = nn.Parameter(shadow, requires_grad=False)
        replacement._megatron_fsdp_model = runtime.fsdp
        for name in names:
            parent, _, leaf = name.rpartition(".")
            ema.get_submodule(parent)._parameters[leaf] = replacement
    for module in ema.modules():
        for name, buffer in module._buffers.items():
            if buffer is not None:
                module._buffers[name] = buffer.to("cuda")

    def prepare(module, state, prefix, local_metadata):
        runtime.prepare_state_dict(state)

    ema.register_state_dict_post_hook(prepare)


class MegatronEmaModelUpdater(DTensorFastEmaModelUpdater):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime

    def _optimizer_view(self):
        self.runtime.fsdp.synchronize_param_gather()
        self.runtime.fsdp._replace_param_with_distributed_if_needed()

    def _refresh_compute(self):
        self.runtime.after_step()
        self.runtime.fsdp.all_gather_pipeline.reset()

    def update_average(self, src_model, tgt_model, beta=0.9999):
        self._optimizer_view()
        super().update_average(src_model, tgt_model, beta)

    def copy_to(self, src_model, tgt_model):
        self._optimizer_view()
        super().copy_to(src_model, tgt_model)
        if tgt_model is self.runtime.fsdp.module:
            self._refresh_compute()

    def cache(self, parameters, is_cpu=False):
        self._optimizer_view()
        super().cache(parameters, is_cpu)

    def restore(self, parameters):
        self._optimizer_view()
        super().restore(parameters)
        self._refresh_compute()
