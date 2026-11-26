# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Record layer outputs/cotangents for the unchanged FP32 mathematical gates."""

import argparse
import json
import runpy
import sys
from pathlib import Path


def attach_trace(root, layers, directory, *, dp_rank=0, dp_size=1):
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    sequence = [-1]
    seen = set()

    def begin(module, inputs):
        sequence[0] += 1

    root.register_forward_pre_hook(begin)

    def sample(tensor):
        value = tensor.detach().reshape(-1, tensor.shape[-1])
        # Fixed head/tail rows retain all channels and use no RNG. Full norms
        # reveal scale differences; sampled payloads locate their first layer.
        selected = torch.cat((value[:32], value[-32:]))
        return {"shape": list(tensor.shape), "norm": value.double().norm().item(), "sample": selected.cpu()}

    for name, layer in layers:

        def capture(module, inputs, output, index=name):
            identity = (sequence[0], index)
            if identity in seen:
                return
            seen.add(identity)
            pack = output[0]
            capture = sequence[0] * dp_size + dp_rank
            prefix = f"forward{capture:03d}.layer{int(index):02d}"
            values = {key: value for key, value in pack.items() if key in ("causal_seq", "full_only_seq", "action_seq")}
            torch.save({key: sample(value) for key, value in values.items()}, directory / f"{prefix}.outputs.pt")
            for key, value in values.items():
                if value.requires_grad:

                    def gradient_hook(gradient, field=key, stem=prefix):
                        torch.save(sample(gradient), directory / f"{stem}.{field}.gradient.pt")

                    value.register_hook(gradient_hook)

        layer.register_forward_hook(capture)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--native-block-control", action="store_true")
    parser.add_argument("--native-norm-order", type=Path)
    args, remainder = parser.parse_known_args()
    sys.path.insert(0, str(args.run_path.resolve().parent))
    mode = remainder[remainder.index("--mode") + 1]
    if args.native_norm_order is not None and mode == "tp":
        from tools.tp_pp_native_grad_norm import install_native_gradient_norm

        names = json.loads(args.native_norm_order.read_text())["names"]
        install_native_gradient_norm([name if name.startswith("net.") else "net." + name for name in names])
    if mode == "native":
        import tp_pp_gpu_leaf_reference as reference

        from tools.run_tp_pp_fp32_native_ac import install_native_checkpointing

        install_native_checkpointing()
        original = reference.place_gpu_trainables

        def place(model):
            handles = original(model)
            attach_trace(model, enumerate(model.language_model.model.layers), args.trace_dir)
            return handles

        reference.place_gpu_trainables = place
    else:
        if args.native_block_control:
            from tools.run_tp_pp_native_block_diagnostic import install

            install()
        from megatron.core import parallel_state as ps

        from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage

        original = PipelineStage.__init__

        def initialize(self, *positional, **keywords):
            original(self, *positional, **keywords)
            if ps.get_tensor_model_parallel_rank() == 0:
                dp_rank, dp_size = ps.get_data_parallel_rank(), ps.get_data_parallel_world_size()
                directory = args.trace_dir if dp_size == 1 else args.trace_dir / f"dp{dp_rank:02d}"
                attach_trace(self, self.layers.items(), directory, dp_rank=dp_rank, dp_size=dp_size)

        PipelineStage.__init__ = initialize
    sys.argv = [str(args.run_path), *remainder]
    runpy.run_path(str(args.run_path), run_name="__main__")


if __name__ == "__main__":
    main()
