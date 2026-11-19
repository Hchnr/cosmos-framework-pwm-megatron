# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate a fresh full-source helper with opt-in native gradient streaming."""

import argparse
import hashlib
import json
from pathlib import Path


def transform(source):
    replacements = [
        (
            "        gpu_leaf_handles = place_gpu_trainables(reference)\n    args.reference_dir.mkdir",
            "        gpu_leaf_handles = place_gpu_trainables(reference)\n"
            "    streamer = None\n"
            "    if args.gpu_leaf_reference:\n"
            "        from tools.tp_pp_streamed_reference_grads import StreamedReferenceGradients\n"
            "        streamer = StreamedReferenceGradients(reference.parameters())\n"
            "    args.reference_dir.mkdir",
        ),
        (
            "    for update in range(3):\n        optimizer.zero_grad(set_to_none=True)\n        outputs = []",
            "    for update in range(3):\n        optimizer.zero_grad(set_to_none=True)\n"
            "        if streamer is not None:\n            streamer.reset()\n        outputs = []",
        ),
        (
            '{"gradient": param.grad.detach().cpu() if param.grad is not None else None}',
            '{"gradient": streamer.raw(param) if streamer is not None else '
            "(param.grad.detach().cpu() if param.grad is not None else None)}",
        ),
        (
            "        if args.gpu_leaf_reference:\n"
            "            from cosmos_framework.callbacks.grad_clip import _clip_grads_with_global_norm, _total_norm_by_mesh\n",
            "        if streamer is not None:\n"
            "            from tp_pp_gpu_leaf_reference import step_gpu_leaves\n"
            "            norm = streamer.norm(reference.parameters())\n"
            "            streamer.clip_and_step(optimizer, float(config.trainer.callbacks.grad_clip.clip_norm), "
            "norm, step_gpu_leaves)\n"
            "        elif args.gpu_leaf_reference:\n"
            "            from cosmos_framework.callbacks.grad_clip import _clip_grads_with_global_norm, _total_norm_by_mesh\n",
        ),
        (
            "        if args.gpu_leaf_reference:\n"
            "            from tp_pp_gpu_leaf_reference import step_gpu_leaves\n\n"
            "            step_gpu_leaves(optimizer)\n"
            '        elif args.native_optimizer == "native_cuda":',
            "        if streamer is not None:\n"
            "            pass  # Already clipped and stepped one GPU leaf at a time.\n"
            "        elif args.gpu_leaf_reference:\n"
            "            from tp_pp_gpu_leaf_reference import step_gpu_leaves\n\n"
            "            step_gpu_leaves(optimizer)\n"
            '        elif args.native_optimizer == "native_cuda":',
        ),
        (
            '        "gpu_leaf_reference": args.gpu_leaf_reference,',
            '        "gpu_leaf_reference": args.gpu_leaf_reference,\n'
            '        "gradient_streaming_candidate": streamer is not None,',
        ),
        (
            '        "clipping_implementation": "native_grad_clip" if args.gpu_leaf_reference else "offloaded_cuda_norm",',
            '        "clipping_implementation": "streamed_cuda_norm_and_clip_candidate" if streamer is not None '
            'else ("native_grad_clip" if args.gpu_leaf_reference else "offloaded_cuda_norm"),',
        ),
        (
            "    if gpu_leaf_handles is not None:\n        from tp_pp_gpu_leaf_reference import close_gpu_trainables",
            "    if streamer is not None:\n        streamer.close()\n"
            "    if gpu_leaf_handles is not None:\n        from tp_pp_gpu_leaf_reference import close_gpu_trainables",
        ),
    ]
    for old, _ in replacements:
        if source.count(old) != 1:
            raise ValueError(
                "Unexpected helper source; refusing ambiguous gradient-streaming transformation"
            )
    result = source
    for old, new in replacements:
        result = result.replace(old, new, 1)
    compile(result, "<streamed-reference-helper>", "exec")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    result = transform(source)
    with args.output.open("x") as handle:
        handle.write(result)
    print(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "output_sha256": hashlib.sha256(result.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
