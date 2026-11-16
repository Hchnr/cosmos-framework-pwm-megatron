"""Fresh HF language weights into raw, native FSDP2 and Megatron uneven layouts."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.distributed.fsdp import fully_shard

from cosmos_framework.model.generator.utils.safetensors_loader import load_language_model
from cosmos_framework.training_backends.megatron import MegatronRuntime
from cosmos_framework.utils.generator.materialize_tensor import materialize_tensor
from cosmos_framework.utils.generator.parallelism import ParallelDims, fsdp_mesh


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(7, 4)
        self.model.norm = nn.Module()
        self.model.norm.weight = nn.Parameter(torch.zeros(3))
        self.lm_head = nn.Linear(4, 7, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    path = Path("/tmp/cosmos_megatron_language_loader")
    expected = {
        "model.embed_tokens.weight": torch.arange(28, dtype=torch.float32).reshape(7, 4),
        "model.norm.weight": torch.arange(3, dtype=torch.float32) + 5,
        "lm_head.weight": torch.arange(28, dtype=torch.float32).reshape(7, 4) + 100,
    }
    dims = ParallelDims(world_size=dist.get_world_size(), dp_shard=2, dp_replicate=2, cp=1, enable_inference_mode=False)
    dims.build_meshes(device_type="cuda")
    for tied in (False, True):
        expected["lm_head.weight"] = (
            expected["model.embed_tokens.weight"] if tied else torch.arange(28, dtype=torch.float32).reshape(7, 4) + 100
        )
        if dist.get_rank() == 0:
            path.mkdir(exist_ok=True)
            save_file(
                {
                    name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name: value
                    for name, value in expected.items()
                    if not (tied and name == "lm_head.weight")
                },
                path / "model.safetensors",
            )
        dist.barrier()
        for backend in ("raw", "pytorch", "megatron"):
            model = Language().cuda()
            model.config.tie_word_embeddings = tied
            if backend == "pytorch":
                fully_shard(model, mesh=fsdp_mesh(dims))
            elif backend == "megatron":
                MegatronRuntime(
                    model,
                    shard_degree=2,
                    replicate_degree=2,
                    unit_types=(Language,),
                    sharding_strategy="optim_grads_params",
                )
            loaded = load_language_model(model, str(path), None, dims, checkpoint_format="qwen3")
            assert loaded == set(expected) - ({"lm_head.weight"} if tied else set()), loaded
            for name, value in model.state_dict().items():
                torch.testing.assert_close(materialize_tensor(value).cpu(), expected[name], atol=0, rtol=0)
        dist.barrier()
    result = dict(
        passed=True,
        backends=["raw", "pytorch", "megatron"],
        replicas=2,
        shards=2,
        weight_tensors=3,
        tied_embedding_cases=2,
        bitwise_equal=True,
    )
    if dist.get_rank() == 0:
        print(json.dumps(result), flush=True)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
