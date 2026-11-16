"""Standalone H100 BF16/Inductor/cache/fusion and CUDA graph correctness smoke.

Run with one visible GPU, COSMOS_TRAINING=0 and PYTHONPATH=.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from cosmos_framework.inference.action_optimization import fuse_small_action_qkv
from cosmos_framework.model.generator.mot.inference_small_action_memory import SmallActionMemoryState
from cosmos_framework.model.generator.mot.small_action_optimization_test import make_layer, make_pack
from cosmos_framework.model.generator.mot.small_action_tower import small_decoder_inputs, small_layer_forward
from cosmos_framework.utils import misc


def main():
    Path("outputs/action_infer_opt/correctness").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(79)
    layer, shape = make_layer()
    layer = layer.eval().cuda().bfloat16()
    reference = copy.deepcopy(layer)
    model = SimpleNamespace(
        training=False,
        parallel_dims=None,
        net=SimpleNamespace(
            language_model=SimpleNamespace(model=SimpleNamespace(small_action_shape=shape, layers=[layer]))
        ),
    )
    fuse_small_action_qkv(model)
    run = torch.compile(small_layer_forward, fullgraph=True, mode="reduce-overhead")
    rows = []
    with torch.inference_mode():
        for request in range(2):
            memories = [SmallActionMemoryState(1), SmallActionMemoryState(1)]
            unds = [
                torch.randn(3, 64, device="cuda", dtype=torch.bfloat16),
                torch.randn(2, 64, device="cuda", dtype=torch.bfloat16),
            ]
            for step in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                for branch in range(2):
                    pack = make_pack(
                        [
                            (
                                unds[branch].cpu(),
                                torch.randn(4, 64, dtype=torch.bfloat16),
                                torch.randn(2, 32, dtype=torch.bfloat16),
                            )
                        ],
                        shape,
                    )
                    pack = misc.to(small_decoder_inputs(pack), device="cuda")
                    expected = small_layer_forward(reference, pack)[0]
                    actual, _, stored = run(layer, pack, memory_value=memories[branch].read_for_layer(0))
                    if stored is not None:
                        memories[branch].write_for_layer(0, stored)
                    for key in ("causal_seq", "full_only_seq", "action_seq"):
                        diff = actual[key].float() - expected[key].float()
                        torch.testing.assert_close(actual[key], expected[key], atol=0.03125, rtol=0.01)
                        rows.append(
                            {
                                "request": request,
                                "step": step,
                                "branch": branch,
                                "key": key,
                                "max_abs": diff.abs().max().item(),
                                "rmse": diff.square().mean().sqrt().item(),
                            }
                        )
            assert all(m.read_for_layer(0).und_key is not None for m in memories)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                run(layer, pack, memory_value=memories[-1].read_for_layer(0))
        prof.export_chrome_trace("outputs/action_infer_opt/correctness/tiny_trace.json")
    json.dump(
        {"passed": True, "rows": rows, "torch": torch.__version__},
        open("outputs/action_infer_opt/correctness/tiny_gpu.json", "w"),
        indent=2,
    )
    print("PASSED", max(x["max_abs"] for x in rows))


if __name__ == "__main__":
    main()
