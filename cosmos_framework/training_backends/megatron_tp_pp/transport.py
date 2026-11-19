# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded tensor-schema transport, separate from differentiable pipeline P2P.

No pickle/object collectives or kernel objects travel on this path. A JSON
descriptor and one contiguous buffer per dtype encode read-only batch state.
The adjacent-stage channel is prewarmed before the MCore schedule, and sends
are kept alive until the next forward or schedule drain.
"""

import json
from collections import deque
from dataclasses import fields, is_dataclass

import torch
import torch.distributed as dist

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.data.generator.sequence_packing.modality import ModalitySpan
from cosmos_framework.data.generator.sequence_packing.runtime import SequencePackMetadata
from cosmos_framework.model.generator.megatron_tp_pp.payload import PayloadLayout
from cosmos_framework.model.generator.mot.small_action_tower import build_small_masks
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean, GenerationDataNoised

_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    )
}
_CLASSES = {
    cls.__name__: cls
    for cls in (
        ModalityData,
        ModalitySpan,
        PackedSequence,
        SequencePackMetadata,
        PayloadLayout,
        GenerationDataClean,
        GenerationDataNoised,
    )
}


def register_dataclass(cls):
    if not is_dataclass(cls) or cls.__name__ in _CLASSES and _CLASSES[cls.__name__] is not cls:
        raise ValueError("Invalid or ambiguous wire dataclass")
    _CLASSES[cls.__name__] = cls


def encode(value, *, device="cuda"):
    tensors, seen = [], {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                raise ValueError("Trainable activations belong in the differentiable PP payload")
            if id(value) not in seen:
                seen[id(value)] = len(tensors)
                tensors.append(value)
            return {"tensor": seen[id(value)]}
        if is_dataclass(value):
            kind = type(value).__name__
            if _CLASSES.get(kind) is not type(value):
                raise TypeError(f"Unregistered wire dataclass: {kind}")
            node = {"class": kind, "fields": {field.name: visit(getattr(value, field.name)) for field in fields(value)}}
            if isinstance(value, PackedSequence):
                node["runtime_metadata"] = visit(value.get_sequence_pack_metadata())
            return node
        if isinstance(value, torch.device):
            return {"device": value.type}
        if isinstance(value, dict):
            return {"dict": [[visit(key), visit(child)] for key, child in value.items()]}
        if isinstance(value, (tuple, list)):
            return {"tuple" if isinstance(value, tuple) else "list": [visit(child) for child in value]}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"Unsupported wire value: {type(value).__name__}")

    tree = visit(value)
    groups, descriptors, offsets = {}, [], {}
    for tensor in tensors:
        dtype = str(tensor.dtype)
        if dtype not in _DTYPES:
            raise TypeError(f"Unsupported wire dtype: {dtype}")
        group = groups.setdefault(dtype, [])
        descriptors.append(
            {
                "dtype": dtype,
                "shape": list(tensor.shape),
                "offset": offsets.get(dtype, 0),
                "cpu": tensor.device.type == "cpu",
            }
        )
        group.append(tensor.detach().to(device=device).reshape(-1))
        offsets[dtype] = offsets.get(dtype, 0) + tensor.numel()
    buffers = {dtype: torch.cat(values) for dtype, values in groups.items()}
    header = {
        "version": 1,
        "tree": tree,
        "tensors": descriptors,
        "buffers": {dtype: buffer.numel() for dtype, buffer in buffers.items()},
    }
    return json.dumps(header, separators=(",", ":")).encode(), buffers


def decode(header_bytes, buffers):
    header = json.loads(header_bytes)
    if header["version"] != 1:
        raise ValueError("Unsupported tensor schema version")
    tensors = []
    for spec in header["tensors"]:
        count = 1
        for dimension in spec["shape"]:
            count *= dimension
        value = buffers[spec["dtype"]].narrow(0, spec["offset"], count).view(spec["shape"])
        tensors.append(value.cpu() if spec["cpu"] else value)

    def visit(value):
        if not isinstance(value, dict):
            return value
        if "tensor" in value:
            return tensors[value["tensor"]]
        if "class" in value:
            result = _CLASSES[value["class"]](**{key: visit(child) for key, child in value["fields"].items()})
            if isinstance(result, PackedSequence):
                result._sequence_pack_metadata = visit(value.get("runtime_metadata"))
            return result
        if "device" in value:
            return (
                torch.device("cuda", torch.cuda.current_device())
                if value["device"] == "cuda"
                else torch.device(value["device"])
            )
        if "dict" in value:
            return {visit(key): visit(child) for key, child in value["dict"]}
        if "tuple" in value:
            return tuple(visit(child) for child in value["tuple"])
        if "list" in value:
            return [visit(child) for child in value["list"]]
        raise ValueError("Unknown tensor schema node")

    return visit(header["tree"])


def broadcast_batch(value, group, *, source_rank=0):
    """Only the TP owner prepares/VAE-encodes data; all peers receive tensors."""
    if dist.get_world_size(group) == 1:
        return value
    source = dist.get_global_rank(group, source_rank)
    owner = dist.get_rank() == source
    header_bytes, buffers = encode(value) if owner else (None, {})
    length = torch.tensor([len(header_bytes) if owner else 0], dtype=torch.int64, device="cuda")
    dist.broadcast(length, src=source, group=group)
    size = int(length.item())
    if not 0 < size <= 16 * 1024 * 1024:
        raise ValueError("Invalid TP metadata descriptor length")
    header_tensor = (
        torch.frombuffer(bytearray(header_bytes), dtype=torch.uint8).cuda()
        if owner
        else torch.empty(size, dtype=torch.uint8, device="cuda")
    )
    dist.broadcast(header_tensor, src=source, group=group)
    if not owner:
        header_bytes = header_tensor.cpu().numpy().tobytes()
        for dtype, numel in json.loads(header_bytes)["buffers"].items():
            buffers[dtype] = torch.empty(numel, dtype=_DTYPES[dtype], device="cuda")
    for buffer in buffers.values():
        if buffer.numel():
            dist.broadcast(buffer, src=source, group=group)
    return value if owner else decode(header_bytes, buffers)


_STATE_KEYS = (
    "slot_id",
    "layout",
    "packed_sequence",
    "loss_inputs",
    "restore_metadata",
    "prepared_network",
    "decoder_metadata",
    "decoder_has_loss",
    "attention_backend",
    "fixed_capacity",
    "num_samples",
    "weight",
)


class PipelineMetadata:
    def __init__(self):
        from megatron.core import parallel_state as ps

        self.ranks = dist.get_process_group_ranks(ps.get_pipeline_model_parallel_group())
        self.rank = self.ranks.index(dist.get_rank())
        # All ranks must create communicators in the same global order. Creating
        # disjoint local groups concurrently can reuse NCCL rendezvous keys and
        # deadlock at the first collective.
        local = torch.tensor(self.ranks, dtype=torch.int64, device="cuda")
        memberships = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(memberships, local)
        for ranks in sorted({tuple(value.cpu().tolist()) for value in memberships}):
            group = dist.new_group(list(ranks), backend="nccl")
            if dist.get_rank() in ranks:
                self.group = group
        dist.all_reduce(torch.zeros(1, device="cuda"), group=self.group)
        self.pending = []
        self.send_buffers = []
        self.bytes_sent = 0
        self.outgoing = deque()
        self.incoming = deque()

    def communicator(self, config):
        """Pair metadata with forward P2P before MCore receives activations.

        Sending from inside module.forward deadlocks: the next stage is still
        waiting for the activation, whereas NCCL can block posting the sideband.
        Keep MCore's schedule and differentiable P2P unchanged, and attach this
        narrow protocol at its public communicator boundary.
        """
        from megatron.core import parallel_state as ps
        from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

        channel = self

        class WithMetadata(P2PCommunicator):
            def recv_forward(self, tensor_shapes, is_first_stage):
                if not is_first_stage:
                    channel.incoming.append(channel._receive_wire())
                return super().recv_forward(tensor_shapes, is_first_stage)

            def send_forward(self, output_tensors, is_last_stage):
                if not is_last_stage:
                    channel._flush_forward()
                return super().send_forward(output_tensors, is_last_stage)

            def send_forward_recv_backward(self, output_tensors, tensor_shapes, is_last_stage):
                if not is_last_stage:
                    channel._flush_forward()
                return super().send_forward_recv_backward(output_tensors, tensor_shapes, is_last_stage)

            def send_backward_recv_forward(self, input_tensor_grads, tensor_shapes, is_first_stage):
                if not is_first_stage:
                    channel.incoming.append(channel._receive_wire())
                return super().send_backward_recv_forward(input_tensor_grads, tensor_shapes, is_first_stage)

        return WithMetadata(ps.get_pipeline_model_parallel_group(), config)

    def drain(self):
        for work in self.pending:
            work.wait()
        self.pending.clear()
        self.send_buffers.clear()

    def send(self, batch):
        if self.rank == len(self.ranks) - 1:
            return
        self.outgoing.append(batch)

    def _flush_forward(self):
        self._send_wire(self.outgoing.popleft())
        self.drain()

    def _send_wire(self, batch):
        self.drain()
        state = {key: batch[key] for key in _STATE_KEYS if key in batch}
        header, buffers = encode(state)
        header_tensor = torch.frombuffer(bytearray(header), dtype=torch.uint8).cuda()
        length = torch.tensor([len(header)], dtype=torch.int64, device="cuda")
        self.send_buffers = [length, header_tensor, *buffers.values()]
        target = self.ranks[self.rank + 1]
        tensors = [tensor for tensor in self.send_buffers if tensor.numel()]
        self.pending = dist.batch_isend_irecv(
            [dist.P2POp(dist.isend, tensor, target, self.group) for tensor in tensors]
        )
        self.bytes_sent += sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _receive_tensor(self, tensor, source):
        for work in dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, source, self.group)]):
            work.wait()

    def _receive_wire(self):
        if self.rank == 0:
            raise ValueError("First stage owns prepared batch state")
        source = self.ranks[self.rank - 1]
        length = torch.empty(1, dtype=torch.int64, device="cuda")
        self._receive_tensor(length, source)
        header_size = int(length.item())
        if not 0 < header_size <= 16 * 1024 * 1024:
            raise ValueError("Invalid pipeline metadata descriptor length")
        header_tensor = torch.empty(header_size, dtype=torch.uint8, device="cuda")
        self._receive_tensor(header_tensor, source)
        header_bytes = header_tensor.cpu().numpy().tobytes()
        header = json.loads(header_bytes)
        buffers = {}
        for dtype, numel in header["buffers"].items():
            buffers[dtype] = torch.empty(numel, dtype=_DTYPES[dtype], device="cuda")
            if numel:
                self._receive_tensor(buffers[dtype], source)
        batch = decode(header_bytes, buffers)
        return batch

    def receive(self, *, expected_slot=None):
        if not self.incoming:
            raise RuntimeError("Metadata must arrive through the schedule's P2P communicator")
        batch = self.incoming.popleft()
        if expected_slot is not None and batch["slot_id"] != expected_slot:
            raise ValueError(f"Wrong in-flight metadata: {batch['slot_id']} != {expected_slot}")
        metadata = dict(batch["decoder_metadata"])
        metadata["_small_masks"] = build_small_masks(
            metadata.pop("_small_ids"),
            batch["attention_backend"],
            batch["num_samples"],
            fixed_capacity=batch["fixed_capacity"],
        )
        batch["metadata"] = metadata
        if self.rank < len(self.ranks) - 1:
            batch["publish_metadata"] = self.send
        return batch
