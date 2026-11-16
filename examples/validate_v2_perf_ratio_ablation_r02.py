#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Strict preflight for the four Nano8B V2 ratio-ablation runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

REPO = Path(__file__).resolve().parents[1]
BASE_TOML = REPO / "examples/toml/sft_config/qwen3vl_v2_0a_understanding_nano8b.toml"
RUN_TABLE = REPO / "examples/v2_perf_ratio_ablation_runs_r02.tsv"
EXPECTED_BRANCH = "ybai_0828/understanding_verification"
BASE_CHECKPOINT = Path("/share/project/liyue/model_ckpts/Cosmos3-Nano-DCP")
HF_MODEL = Path("/share/project/eai_pwm/models/Cosmos3-Nano")
OWNER_FILE = ".v2_perf_ratio_ablation_owner"
RATIO_SUM = 30_248

DATASET_NAMES = [
    "vlm_robotics_fullscan_v2r1_p0_episodic_embodied",
    "vlm_robotics_fullscan_v2r1_p0_metric_3d_scene",
    "vlm_robotics_fullscan_v2r1_p0_multiview_memory",
    "vlm_robotics_fullscan_v2r1_p0_temporal_physical",
    "vlm_robotics_fullscan_v2r1_p1_episodic_embodied",
    "vlm_robotics_fullscan_v2r1_p1_free_space_affordance",
    "vlm_robotics_fullscan_v2r1_p1_metric_3d_scene",
    "vlm_robotics_fullscan_v2r1_p1_multiview_memory",
    "vlm_robotics_fullscan_v2r1_p1_temporal_physical",
    "vlm_robotics_affordance_support_v1_robot_placement",
    "vlm_robotics_affordance_support_v1_semantic_affordance",
    "vlm_robotics_fullscan_v2r1_p2_visual_structured_retention",
    "vlm_robotics_general_retention_v1_safe_general",
    "gqa_train_balanced_reference",
    "mindcube_train_spatial_qa_reference",
    "sharerobot_planning_qa_v1_train",
    "sharerobot_affordance_bbox_text_v1_train",
    "sharerobot_trajectory_points_text_v1_train",
    "msr3d_strict_object_image_qa_v1_train",
    "robobench_official_prompt_all_training_v1_train",
]

RATIOS = {
    "robot85_general15": [7047, 1557, 1358, 5636, 65, 70, 1009, 246, 2722, 161, 1, 2853, 1427, 257, 52, 5271, 33, 36, 416, 31],
    "robot60_general40": [4975, 1099, 958, 3978, 46, 49, 713, 174, 1921, 114, 1, 7609, 3804, 686, 36, 3720, 24, 25, 294, 22],
    "spatial_high": [5123, 2373, 2069, 4097, 48, 51, 1538, 374, 1979, 117, 1, 4944, 2472, 446, 79, 3831, 24, 26, 634, 22],
    "planning_high": [5281, 1167, 1017, 4223, 49, 92, 757, 184, 2040, 210, 2, 4944, 2472, 446, 39, 6884, 43, 46, 312, 40],
}

TAIL_OVERRIDES = [
    "+model.config.vlm_config.model_instance.config.include_visual=true",
    f"model.config.vlm_config.tokenizer.pretrained_model_name={HF_MODEL}",
    "~trainer.callbacks.every_n_sample_reg=null",
    "~trainer.callbacks.every_n_sample_ema=null",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for line in RUN_TABLE.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 7:
            fail(f"invalid run-table row: {line!r}")
        experiment, host, toml, job, output, port, session = fields
        runs.append(
            {
                "experiment": experiment,
                "host": host,
                "toml": Path(toml),
                "job": job,
                "output": Path(output),
                "port": int(port),
                "session": session,
            }
        )
    if len(runs) != 4:
        fail(f"run table must contain four rows, found {len(runs)}")
    if [run["experiment"] for run in runs] != list(RATIOS):
        fail("run-table experiment order differs from the required launch order")
    return runs


def check_branch() -> tuple[str, str]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    if branch != EXPECTED_BRANCH:
        fail(f"wrong branch: expected {EXPECTED_BRANCH!r}, got {branch!r}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    return branch, commit


def check_output_roots(runs: list[dict[str, object]], state: str) -> None:
    outputs = [Path(run["output"]).resolve(strict=False) for run in runs]
    if len(set(outputs)) != 4:
        fail("output roots are not unique")
    for left in outputs:
        for right in outputs:
            if left != right and (left in right.parents or right in left.parents):
                fail(f"overlapping output roots: {left} and {right}")
    allowed_parent = (REPO / "outputs/train").resolve(strict=False)
    for run, root in zip(runs, outputs, strict=True):
        if root.parent != allowed_parent:
            fail(f"output root is outside the exact allowed parent: {root}")
        model_dir = root / "cosmos3/understanding_verification" / str(run["job"])
        checkpoint_dir = model_dir / "checkpoints"
        if state == "absent":
            if root.exists() or model_dir.exists() or checkpoint_dir.exists():
                fail(f"new output path already exists: {root}")
        elif state == "owned":
            owner = root / OWNER_FILE
            if not root.is_dir() or not owner.is_file():
                fail(f"prepared output lacks ownership marker: {root}")
            if owner.read_text().strip() != run["experiment"]:
                fail(f"ownership marker mismatch for {root}")
        elif state != "ignore":
            fail(f"unsupported output state: {state}")


def expected_raw(base: dict, run: dict[str, object]) -> dict:
    result = copy.deepcopy(base)
    result["job"]["name"] = run["job"]
    result["job"]["wandb_mode"] = "online"
    result["model"]["max_num_tokens_after_packing"] = 65_536
    result["model"]["train_vlm_visual"] = True
    result["optimizer"]["lr"] = 2.0e-5
    result["optimizer"]["weight_decay"] = 2.0e-7
    result["scheduler"]["cycle_lengths"] = [8_500]
    result["scheduler"]["warm_up_steps"] = [200]
    result["trainer"]["grad_accum_iter"] = 1
    result["trainer"]["logging_iter"] = 1
    result["trainer"]["max_iter"] = 8_500
    result["checkpoint"]["save_iter"] = 500
    result["dataloader_train"]["max_samples_per_batch"] = 48
    result["dataloader_train"]["packing_pool_size"] = 48
    result["dataloader_train"]["max_sequence_length"] = 65_536
    result["dataloader_train"]["seed"] = 42
    for spec, ratio in zip(
        result["dataloader_train"]["datasets"], RATIOS[str(run["experiment"])], strict=True
    ):
        spec["ratio"] = ratio
    return result


def check_static_configs(runs: list[dict[str, object]]) -> tuple[dict, list[dict[str, object]]]:
    with BASE_TOML.open("rb") as handle:
        base = tomllib.load(handle)
    base_paths = [spec["path"] for spec in base["dataloader_train"]["datasets"]]
    if len(base_paths) != 20 or [Path(path).name for path in base_paths] != DATASET_NAMES:
        fail("baseline dataset paths/order no longer match the frozen 20-way V2 contract")
    reports: list[dict[str, object]] = []
    for run in runs:
        toml_path = Path(run["toml"])
        if not toml_path.is_file():
            fail(f"missing TOML: {toml_path}")
        with toml_path.open("rb") as handle:
            raw = tomllib.load(handle)
        expected = expected_raw(base, run)
        if raw != expected:
            fail(f"{toml_path} differs from baseline beyond the authorized fields")
        specs = raw["dataloader_train"]["datasets"]
        ratios = [int(spec["ratio"]) for spec in specs]
        paths = [spec["path"] for spec in specs]
        if len(specs) != 20 or paths != base_paths:
            fail(f"dataset paths/order mismatch: {toml_path}")
        if ratios != RATIOS[str(run["experiment"])] or sum(ratios) != RATIO_SUM:
            fail(f"ratio mismatch or bad sum: {toml_path}")
        model_dir = Path(run["output"]) / "cosmos3/understanding_verification" / str(run["job"])
        reports.append(
            {
                **run,
                "toml": str(toml_path),
                "output": str(run["output"]),
                "model_dir": str(model_dir),
                "checkpoint_dir": str(model_dir / "checkpoints"),
                "toml_sha256": sha256(toml_path),
                "ratio_sum": sum(ratios),
            }
        )
    return base, reports


def check_resolved_configs(runs: list[dict[str, object]]) -> None:
    os.environ["BASE_CHECKPOINT_PATH"] = str(BASE_CHECKPOINT)
    os.environ["HF_MODEL_PATH"] = str(HF_MODEL)
    sys.path.insert(0, str(REPO))
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.data.generator.catalog_processors import build_catalog_joint_dataloader

    if 'selection_mode="exact_periodic"' not in inspect.getsource(build_catalog_joint_dataloader):
        fail("catalog loader no longer hard-codes exact_periodic scheduling")
    normalized: list[dict[str, object]] = []
    for run in runs:
        os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(run["output"])
        config = load_experiment_from_toml(str(run["toml"]), TAIL_OVERRIDES)
        model = config.model.config
        checks = {
            "job.name": (str(config.job.name), str(run["job"])),
            "job.wandb_mode": (str(config.job.wandb_mode), "online"),
            "optimizer.lr": (float(config.optimizer.lr), 2.0e-5),
            "optimizer.weight_decay": (float(config.optimizer.weight_decay), 2.0e-7),
            "trainer.max_iter": (int(config.trainer.max_iter), 8_500),
            "trainer.grad_accum_iter": (int(config.trainer.grad_accum_iter), 1),
            "trainer.logging_iter": (int(config.trainer.logging_iter), 1),
            "scheduler.cycle_lengths": (list(config.scheduler.cycle_lengths), [8_500]),
            "scheduler.warm_up_steps": (list(config.scheduler.warm_up_steps), [200]),
            "checkpoint.save_iter": (int(config.checkpoint.save_iter), 500),
            "checkpoint.load_path": (str(config.checkpoint.load_path), str(BASE_CHECKPOINT)),
            "model.max_num_tokens_after_packing": (int(model.max_num_tokens_after_packing), 65_536),
            "model.train_vlm_visual": (bool(model.train_vlm_visual), True),
            "dataloader.max_sequence_length": (int(config.dataloader_train.max_sequence_length), 65_536),
            "dataloader.max_samples_per_batch": (int(config.dataloader_train.max_samples_per_batch), 48),
            "dataloader.packing_pool_size": (int(config.dataloader_train.packing_pool_size), 48),
            "dataloader.seed": (int(config.dataloader_train.seed), 42),
            "trainer.seed": (int(config.trainer.seed), 42),
        }
        for key, (actual, expected) in checks.items():
            if actual != expected:
                fail(f"resolved {key}: expected {expected!r}, got {actual!r} for {run['experiment']}")
        actual_specs = [(spec["path"], int(spec["ratio"])) for spec in config.catalog_datasets]
        expected_specs = [
            (spec["path"], ratio)
            for spec, ratio in zip(
                tomllib.loads(Path(run["toml"]).read_text())["dataloader_train"]["datasets"],
                RATIOS[str(run["experiment"])],
                strict=True,
            )
        ]
        if actual_specs != expected_specs:
            fail(f"resolved dataset order/ratio mismatch for {run['experiment']}")
        normalized.append(
            {
                "optimizer": dict(config.optimizer),
                "scheduler": {k: v for k, v in dict(config.scheduler).items() if k != "_target_"},
                "trainer": {
                    "max_iter": config.trainer.max_iter,
                    "grad_accum_iter": config.trainer.grad_accum_iter,
                    "logging_iter": config.trainer.logging_iter,
                    "seed": config.trainer.seed,
                },
                "model": {
                    "max_num_tokens_after_packing": model.max_num_tokens_after_packing,
                    "train_vlm_visual": model.train_vlm_visual,
                    "parallelism": dict(model.parallelism),
                },
                "dataloader": {
                    "max_sequence_length": config.dataloader_train.max_sequence_length,
                    "max_samples_per_batch": config.dataloader_train.max_samples_per_batch,
                    "packing_pool_size": config.dataloader_train.packing_pool_size,
                    "seed": config.dataloader_train.seed,
                },
                "checkpoint": {
                    "save_iter": config.checkpoint.save_iter,
                    "load_path": config.checkpoint.load_path,
                },
            }
        )
    if any(item != normalized[0] for item in normalized[1:]):
        fail("resolved non-ratio training parameters differ across experiments")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-state", choices=["absent", "owned", "ignore"], default="absent")
    parser.add_argument("--skip-resolved", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    branch, commit = check_branch()
    if not BASE_CHECKPOINT.is_dir() or not HF_MODEL.is_dir():
        fail("base DCP or Nano HF processor directory is missing")
    runs = read_runs()
    check_output_roots(runs, args.output_state)
    _, reports = check_static_configs(runs)
    if not args.skip_resolved:
        check_resolved_configs(runs)
    report = {
        "status": "PASS",
        "branch": branch,
        "commit": commit,
        "base_toml": str(BASE_TOML),
        "base_toml_sha256": sha256(BASE_TOML),
        "base_checkpoint": str(BASE_CHECKPOINT),
        "resolved_checked": not args.skip_resolved,
        "runs": reports,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("PASS: four TOMLs exactly match the baseline plus authorized changes")
        print("PASS: each ratio has 20 entries and sums to 30248")
        print("PASS: dataset paths/order and exact_periodic scheduling are unchanged")
        if not args.skip_resolved:
            print("PASS: resolved LR/WD/scheduler/max_iter/save_iter/packing/seed/WandB mode are exact")
        else:
            print("SKIP: resolved-config construction was not requested")
        print("PASS: output roots are unique and satisfy state", args.output_state)
        for item in reports:
            print(f"{item['experiment']}: {item['toml_sha256']} {item['output']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
