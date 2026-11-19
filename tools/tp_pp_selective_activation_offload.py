"""Offload selected compiled blocks' saved activations, retaining GPU parameters."""

from tools.tp_pp_activation_offload import saved_tensor_cpu_offload


def activation_offload(parameters, observer=None):
    keep = {parameter.untyped_storage().data_ptr() for parameter in parameters if parameter.is_cuda}
    context = saved_tensor_cpu_offload()
    original_pack = context.pack_hook

    def pack(tensor):
        if tensor.is_cuda and tensor.untyped_storage().data_ptr() in keep:
            # Never retain the original autograd Tensor in a saved-tensor hook.
            result = (tensor.device, tensor.detach())
        else:
            result = original_pack(tensor)
        if observer is not None:
            observer(tensor, result)
        return result

    context.pack_hook = pack
    return context


def wrap_blocks(blocks, count, observer=None):
    blocks = list(blocks)
    if not 0 < count <= len(blocks):
        raise ValueError("Selective offload count must fit the local pipeline stage")
    selected = []
    for name, block in blocks[:count]:
        if getattr(block.forward, "_cosmos_selective_activation_offload", False):
            raise ValueError("Block already has selective activation offload")
        original = block.forward

        def forward(*args, _block=block, _original=original, **kwargs):
            # DDP may rebind parameters to flat buffers after optimize_stage.
            # Resolve live storages here rather than caching pre-DDP pointers.
            with activation_offload(_block.parameters(), observer):
                return _original(*args, **kwargs)

        forward._cosmos_selective_activation_offload = True
        block.forward = forward
        selected.append(name)
    return selected
