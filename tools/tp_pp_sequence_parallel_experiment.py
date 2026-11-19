"""Isolated three-stream sequence/tensor-parallel mathematical prototype.

No training integration. Every rank must call every collective in the same
order. Inputs have identical global metadata and physical sequence lengths
divisible by the process group size. Q/K/V see the whole packed sequence; only
residuals and per-token norms are sequence sharded. The full output is gathered
for a replicated loss; each rank MUST divide that loss by group size.
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local, group):
        ctx.group = group
        ctx.size = dist.get_world_size(group)
        if local.ndim != 2:
            raise ValueError("Expected a 2-D sequence shard")
        if local.numel() == 0:
            return local.clone()
        output = local.new_empty((local.shape[0] * ctx.size, local.shape[1]))
        dist.all_gather_into_tensor(output, local.contiguous(), group=group)
        return output

    @staticmethod
    def backward(ctx, gradient):
        if gradient.numel() == 0:
            return gradient.clone(), None
        output = gradient.new_empty((gradient.shape[0] // ctx.size, gradient.shape[1]))
        dist.reduce_scatter_tensor(output, gradient.contiguous(), op=dist.ReduceOp.SUM, group=ctx.group)
        return output, None


class ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial, group):
        ctx.group = group
        ctx.size = dist.get_world_size(group)
        if partial.ndim != 2 or partial.shape[0] % ctx.size:
            raise ValueError("Physical token length must divide the TP degree")
        if partial.numel() == 0:
            return partial.clone()
        output = partial.new_empty((partial.shape[0] // ctx.size, partial.shape[1]))
        dist.reduce_scatter_tensor(output, partial.contiguous(), op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, gradient):
        if gradient.numel() == 0:
            return gradient.clone(), None
        output = gradient.new_empty((gradient.shape[0] * ctx.size, gradient.shape[1]))
        dist.all_gather_into_tensor(output, gradient.contiguous(), group=ctx.group)
        return output, None


def gather(local, group):
    return Gather.apply(local, group)


def reduce_scatter(partial, group):
    return ReduceScatter.apply(partial, group)


def validate_contract(lengths, group):
    size = dist.get_world_size(group)
    records = [None] * size
    dist.all_gather_object(records, lengths, group=group)
    if any(item != lengths for item in records):
        raise ValueError("All ranks must enter the same streams and lengths")
    if not isinstance(lengths, dict) or set(lengths) != {"und", "video", "action"}:
        raise ValueError("Exactly three modalities are required")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 or n % size for n in lengths.values()):
        raise ValueError("Physical sequence lengths must be nonnegative and divisible by TP")


def rotate(x, position):
    half = x.shape[-1] // 2
    angle = position.to(x.dtype)[:, None, None] * 0.03
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), dim=-1)


def attention(q, k, v, query_ids, key_ids, *, causal):
    if q.shape[0] == 0:
        return q.flatten(1)
    mask = (query_ids[:, None] >= 0) & (query_ids[:, None] == key_ids[None, :])
    if causal:
        mask = mask & (
            torch.arange(len(query_ids), device=q.device)[:, None]
            >= torch.arange(len(key_ids), device=q.device)[None, :]
        )
    # Physical padding is inert and its output is omitted from the loss, but
    # every row retains one finite key to avoid NaN softmax rows.
    mask = mask | ((query_ids[:, None] < 0) & (torch.arange(len(key_ids), device=q.device)[None, :] == 0))
    result = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        attn_mask=mask[None, None],
        enable_gqa=True,
    )
    return result.squeeze(0).transpose(0, 1).reshape(q.shape[0], -1)


class ThreeStreamBlock(torch.nn.Module):
    def __init__(self, hidden=16, qheads=8, kvheads=4, dim=4, intermediate=32, *, rank=None, world=1):
        super().__init__()
        if qheads % kvheads or qheads % world or kvheads % world or intermediate % world or dim % 2:
            raise ValueError("Projection heads and MLP width must divide TP; rotary dimension must be even")
        self.hidden, self.qheads, self.kvheads, self.dim = hidden, qheads, kvheads, dim
        self.rank, self.world, self.intermediate = rank, world, intermediate
        shapes = {
            "q": (qheads * dim, hidden),
            "k": (kvheads * dim, hidden),
            "v": (kvheads * dim, hidden),
            "o": (hidden, qheads * dim),
            "gate": (intermediate, hidden),
            "up": (intermediate, hidden),
            "down": (hidden, intermediate),
        }
        self.weights = torch.nn.ParameterDict()
        for stream in ("und", "video", "action"):
            for name, (rows, cols) in shapes.items():
                axis = 1 if name in ("o", "down") else 0
                shape = (rows // world, cols) if axis == 0 else (rows, cols // world)
                self.weights[f"{stream}_{name}"] = torch.nn.Parameter(torch.empty(shape, dtype=torch.float64))

    def forward(self, streams, ids, positions, *, group=None):
        sharded = group is not None
        expanded = {name: gather(x, group) if sharded else x for name, x in streams.items()}
        q, k, v = {}, {}, {}
        for name in ("und", "video", "action"):
            t = expanded[name].shape[0]
            q[name] = rotate(
                F.linear(expanded[name], self.weights[f"{name}_q"]).reshape(t, self.qheads // self.world, self.dim),
                positions[name],
            )
            k[name] = rotate(
                F.linear(expanded[name], self.weights[f"{name}_k"]).reshape(t, self.kvheads // self.world, self.dim),
                positions[name],
            )
            v[name] = F.linear(expanded[name], self.weights[f"{name}_v"]).reshape(
                t, self.kvheads // self.world, self.dim
            )
        result = {}
        for name in ("und", "video", "action"):
            others = ("und",) if name == "und" else ("und", "video", "action")
            joined_k = torch.cat([k[key] for key in others])
            joined_v = torch.cat([v[key] for key in others])
            joined_ids = torch.cat([ids[key] for key in others])
            y = attention(q[name], joined_k, joined_v, ids[name], joined_ids, causal=name == "und")
            partial = F.linear(y, self.weights[f"{name}_o"])
            combined = reduce_scatter(partial, group) if sharded else partial
            residual = streams[name] + combined
            normalized = F.layer_norm(residual, (self.hidden,))
            mlp_input = gather(normalized, group) if sharded else normalized
            gated = F.silu(F.linear(mlp_input, self.weights[f"{name}_gate"])) * F.linear(
                mlp_input, self.weights[f"{name}_up"]
            )
            partial = F.linear(gated, self.weights[f"{name}_down"])
            mlp = reduce_scatter(partial, group) if sharded else partial
            result[name] = residual + mlp
        return result


def slice_weight(name, value, rank, world):
    axis = 1 if name.endswith(("_o", "_down")) else 0
    if value.shape[axis] % world:
        raise ValueError("Parameter dimension must divide TP")
    return value.narrow(axis, rank * (value.shape[axis] // world), value.shape[axis] // world)


def initialize_reference(reference, generator):
    with torch.no_grad():
        for weight in reference.parameters():
            weight.copy_(
                torch.randn(weight.shape, dtype=weight.dtype, generator=generator, device=weight.device)
                / math.sqrt(weight.shape[-1])
            )
