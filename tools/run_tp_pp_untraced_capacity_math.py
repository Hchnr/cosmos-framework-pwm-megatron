# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the original-capacity FP32 oracle without costly per-layer trace norms.

This installs nested token-chunk and flat outer checkpointing on both paths.
The caller must validate the streamed reference, GPU tiny and memory gates and
run native before TP. This entry alone makes no numerical acceptance claim.
"""

import argparse
import json
import runpy
import sys
from pathlib import Path


def unique_option(arguments, option):
    if arguments.count(option) != 1:
        raise ValueError(f"Expected exactly one {option}")
    index = arguments.index(option)
    if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
        raise ValueError(f"Missing value for {option}")
    return arguments[index + 1]


def validate_contract(arguments, norm_order, chunk_tokens):
    if chunk_tokens <= 0:
        raise ValueError("Chunk size must be positive")
    for flag in ("--original-capacity", "--retain-full-text", "--deterministic"):
        if arguments.count(flag) != 1:
            raise ValueError(f"Expected exactly one {flag}")
    for option, expected in (("--objective", "native_rf"), ("--attention-backend", "flex")):
        if unique_option(arguments, option) != expected:
            raise ValueError(f"Expected {option}={expected}")
    mode = unique_option(arguments, "--mode")
    if mode == "native":
        if norm_order is not None:
            raise ValueError("The native reference does not use a diagnostic TP norm order")
        for flag in ("--gpu-leaf-reference", "--preserve-saved-layout"):
            if arguments.count(flag) != 1:
                raise ValueError(f"Native reference requires {flag}")
        if (
            unique_option(arguments, "--native-device") != "cuda"
            or unique_option(arguments, "--native-optimizer") != "native_cuda"
        ):
            raise ValueError("Native reference requires CUDA and native fused Adam")
    elif mode == "tp":
        if norm_order is None or arguments.count("--native-adam-arithmetic") != 1:
            raise ValueError("TP requires native norm order and native Adam arithmetic")
        if unique_option(arguments, "--tp") not in ("2", "4"):
            raise ValueError("TP degree must be 2 or 4")
        if "--replay-native-state" in arguments:
            raise ValueError("TP must use its continuous optimizer trajectory")
    else:
        raise ValueError("Mode must be native or tp")
    return mode


def install(mode, chunk_tokens, norm_order):
    from tools import run_tp_pp_checkpointed_chunked_mlp_math as chunks
    from tools import run_tp_pp_flat_checkpoint_math as flat

    # flat.install patches the imported native_ac module function. Calling
    # that same module below is necessary: runpy on its source would create a
    # separate __main__ and silently lose the patched flat checkpoint.
    flat.install()
    chunks.install(chunk_tokens)
    if mode == "native":
        from tools import run_tp_pp_fp32_native_ac as native_ac

        native_ac.install_native_checkpointing()
    else:
        from tools.tp_pp_native_grad_norm import install_native_gradient_norm

        names = json.loads(norm_order.read_text())["names"]
        install_native_gradient_norm([name if name.startswith("net.") else "net." + name for name in names])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--native-norm-order", type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    args, remainder = parser.parse_known_args(argv)
    mode = validate_contract(remainder, args.native_norm_order, args.chunk_tokens)
    if not args.run_path.is_file():
        raise FileNotFoundError(args.run_path)
    sys.path.insert(0, str(args.run_path.resolve().parent))
    install(mode, args.chunk_tokens, args.native_norm_order)
    print(json.dumps({"untraced_capacity_runner": True, "mode": mode, "chunk_tokens": args.chunk_tokens}), flush=True)
    sys.argv = [str(args.run_path), *remainder]
    runpy.run_path(str(args.run_path), run_name="__main__")


if __name__ == "__main__":
    main()
