# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Recover the exact former FP32 failing inputs, requiring old noise/pack SHA."""

import argparse
import json
import os
from pathlib import Path

import torch

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.training_backends.megatron_tp_pp.planning import batch_fingerprint
from cosmos_framework.utils.generator.rand_state import preserve_training_rng, set_rand_state_dict
from tools.verify_tp_pp_native_data_replay import native_preparation, runtime_rank_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-legacy-fingerprints", action="store_true")
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--runtime-checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_legacy_fingerprints:
        parser.error("Legacy ambiguous fingerprints require --allow-legacy-fingerprints")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["OUTPUT_ROOT"] = str(args.output.resolve() / "runtime")
    torch.cuda.set_device(0)
    config = load_experiment_from_toml(str(args.toml))
    preparation = native_preparation(config.model.config)
    _, rng = runtime_rank_states(args.runtime_checkpoint)
    expected = {tuple(row["slot"]): row for row in json.loads(args.plan.read_text())}
    iteration = json.loads((args.runtime_checkpoint / "manifest.json").read_text())["iteration"]
    rows = []
    for rank in (0, 1):
        folder = args.raw_root / f"legacy_raw_rank{rank}"
        manifest = json.loads((folder / "manifest.json").read_text())
        target = expected[(iteration, 0, rank)]
        if target.get("fingerprint_version", 1) != 1:
            raise ValueError("Legacy recovery requires a version 1 plan")
        assert manifest["passed"] and manifest["records"][0]["sha256"] == target["sha256"]
        raw = torch.load(folder / "micro000.raw.pt", map_location="cpu", weights_only=False)
        with preserve_training_rng(), torch.no_grad():
            set_rand_state_dict(rng[rank])
            prepared = preparation.prepare_generation_training_inputs(raw, iteration)
        loss_inputs = {
            "gen_data_noised": prepared["gen_data_noised"],
            "timesteps": prepared["timesteps_vision"],
            "timesteps_action": prepared["timesteps_action"],
            "timesteps_sound": prepared["timesteps_sound"],
            "is_image_batch": prepared["gen_data_clean"].is_image_batch,
        }
        noise = batch_fingerprint(loss_inputs, allow_cuda=True, version=1)
        packed = batch_fingerprint(prepared["packed_sequence"], allow_cuda=True, version=1)
        if noise != target["prepared_noise_sha256"] or packed != target["prepared_packed_sha256"]:
            raise ValueError(f"Former failing input changed at logical rank {rank}")
        torch.save(prepared["packed_sequence"], args.output / f"rank{rank}.micro0.pt")
        torch.save(loss_inputs, args.output / f"rank{rank}.micro0.loss.pt")
        rows.append(dict(rank=rank, raw_sha256=target["sha256"], noise_sha256=noise, packed_sha256=packed))
        print(json.dumps(rows[-1]), flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                passed=False,
                recorded_fingerprints_match=True,
                scope=__doc__,
                rows=rows,
                mathematical_acceptance=False,
                fingerprint_version=1,
                legacy_fingerprint_ambiguous=True,
            ),
            indent=2,
        )
        + "\n"
    )
    # Recovery completed but the legacy fingerprint cannot establish structural identity.
    raise SystemExit(1)


if __name__ == "__main__":
    main()
