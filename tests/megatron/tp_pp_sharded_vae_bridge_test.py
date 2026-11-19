# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.run_tp4_sharded_vae_raw_benchmark import QUALIFICATION_DEPENDENCIES, validate_qualification


@pytest.mark.parametrize("world_size", [2, 4])
def test_bridge_protocol_preserves_items_rng_and_owner_failure(tmp_path, world_size):
    worker = Path(__file__).with_name("tp_pp_sharded_vae_bridge_gloo_worker.py")
    env = {**os.environ, "BRIDGE_TEST_OUTPUT": str(tmp_path), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world_size}", str(worker)],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(world_size):
        row = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert row == {"rank": rank, "rows": [0, 1, 2], "slots": 3, "calls": 3, "caught": True}


def fixture_reports(tmp_path):
    fixture, qualification = tmp_path / "fixture", tmp_path / "qualification"
    fixture.mkdir()
    qualification.mkdir()
    original, candidate = tmp_path / "original", tmp_path / "candidate"
    pins = {}
    for relative in QUALIFICATION_DEPENDENCIES:
        if relative.endswith("/preparation.py"):
            text = "class GenerationPreparation:\n    def run(self): return 1\n"
        elif relative.endswith(".py"):
            text = f"VALUE = {relative!r}\n"
        else:
            text = 'value = "fixture"\n'
        for root in (original, candidate):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        old = original / relative
        pins[str(old)] = {"resolved": str(old), "sha256": hashlib.sha256(old.read_bytes()).hexdigest()}
    vae_file = tmp_path / "vae.pth"
    vae_file.write_bytes(b"vae-weights")
    pins[str(vae_file)] = {"resolved": str(vae_file), "sha256": hashlib.sha256(vae_file.read_bytes()).hexdigest()}
    hf_root = tmp_path / "hf"
    hf_root.mkdir()
    hf_file = hf_root / "model.safetensors"
    hf_file.write_bytes(b"hf-weights")
    pins[str(hf_file)] = {"resolved": str(hf_file), "sha256": hashlib.sha256(hf_file.read_bytes()).hexdigest()}
    environment = {"HF_MODEL_PATH": str(hf_root), "WAN_VAE_PATH": str(vae_file), "BASE_CHECKPOINT_PATH": "model"}
    (qualification / "queue_result.json").write_text(
        json.dumps(
            {
                "completed": True,
                "returncode": 0,
                "failure": None,
                "source_sha256_on_queue": pins,
                "source_sha256_before_launch": pins,
                "source_sha256_after": pins,
                "required_environment": {
                    key: {"sha256": hashlib.sha256(value.encode()).hexdigest()} for key, value in environment.items()
                },
            }
        )
    )
    records, rows = [], []
    for index in range(3):
        data = f"fixture{index}".encode()
        (fixture / f"micro{index:03d}.raw.pt").write_bytes(data)
        record = {"slot": [1200, index, 0], "samples": 9 + index, "sha256": "semantic"}
        records.append(record)
        rows.append(
            {
                "fixture": index,
                "phase": "qualification",
                "fixture_file_sha256": hashlib.sha256(data).hexdigest(),
                "original_fixture_identity": record,
                "logical_rng_all_fields_exact": True,
                "physical_owner_rng_restored_all_fields": True,
                "workers": [{"rank": rank} for rank in range(4)],
                "complete_prepared_payload": {"float32": {"exact": True}},
                "latent_max_absolute_differences": [0.0],
            }
        )
    (fixture / "manifest.json").write_text(json.dumps({"passed": True, "records": records}))
    for rank in range(4):
        (qualification / f"qualification.rank{rank}.json").write_text(
            json.dumps({"rank": rank, "passed": True, "atol": 0, "rtol": 0, "rows": rows if rank == 0 else []})
        )
    return fixture, qualification, candidate, environment


def test_qualification_requires_exact_same_fixture_and_rng(tmp_path):
    fixture, qualification, source_root, environment = fixture_reports(tmp_path)
    assert len(validate_qualification(qualification, fixture, source_root, environment=environment)) == 3
    (fixture / "micro001.raw.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="contract failed"):
        validate_qualification(qualification, fixture, source_root, environment=environment)


@pytest.mark.parametrize("mutation", ["tolerance", "rng", "empty_latents", "inexact_payload", "failed_rank"])
def test_qualification_rejects_weak_or_failed_report(tmp_path, mutation):
    fixture, qualification, source_root, environment = fixture_reports(tmp_path)
    target = qualification / ("qualification.rank3.json" if mutation == "failed_rank" else "qualification.rank0.json")
    report = json.loads(target.read_text())
    if mutation == "tolerance":
        report["rtol"] = 1e-5
    elif mutation == "rng":
        report["rows"][0]["logical_rng_all_fields_exact"] = False
    elif mutation == "empty_latents":
        report["rows"][0]["latent_max_absolute_differences"] = []
    elif mutation == "inexact_payload":
        report["rows"][0]["complete_prepared_payload"]["float32"]["exact"] = False
    else:
        report["passed"] = False
    target.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_qualification(qualification, fixture, source_root, environment=environment)


def test_qualification_rejects_changed_vae_source_but_allows_unrelated_loss_change(tmp_path):
    fixture, qualification, source_root, environment = fixture_reports(tmp_path)
    preparation = source_root / "cosmos_framework/training_backends/megatron_tp_pp/preparation.py"
    preparation.write_text(preparation.read_text() + "\nclass GenerationLoss: pass\n")
    validate_qualification(qualification, fixture, source_root, environment=environment)
    preparation.write_text(preparation.read_text().replace("return 1", "return 2"))
    with pytest.raises(ValueError, match="source differ"):
        validate_qualification(qualification, fixture, source_root, environment=environment)


def test_qualification_allows_comments_but_rejects_changed_rng_and_recipe(tmp_path):
    fixture, qualification, source_root, environment = fixture_reports(tmp_path)
    rng = source_root / "cosmos_framework/training_backends/megatron_tp_pp/rng.py"
    recipe = source_root / "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"
    rng.write_text("# license\n" + rng.read_text())
    recipe.write_text("# license\n" + recipe.read_text())
    validate_qualification(qualification, fixture, source_root, environment=environment)
    rng.write_text(rng.read_text() + "CHANGED = True\n")
    with pytest.raises(ValueError, match="source differ"):
        validate_qualification(qualification, fixture, source_root, environment=environment)
    rng.write_text(rng.read_text().replace("CHANGED = True\n", ""))
    recipe.write_text(recipe.read_text().replace('"fixture"', '"different"'))
    with pytest.raises(ValueError, match="source differ"):
        validate_qualification(qualification, fixture, source_root, environment=environment)


def test_qualification_rejects_different_runtime_recipe(tmp_path):
    fixture, qualification, source_root, environment = fixture_reports(tmp_path)
    recipe = tmp_path / "runtime.toml"
    recipe.write_text('value = "different"\n')
    with pytest.raises(ValueError, match="recipe differs"):
        validate_qualification(qualification, fixture, source_root, recipe, environment)


def test_qualification_requires_same_vae_path_environment(tmp_path):
    fixture, qualification, source_root, environment = fixture_reports(tmp_path)
    environment["WAN_VAE_PATH"] = "another_vae"
    with pytest.raises(ValueError, match="WAN_VAE_PATH"):
        validate_qualification(qualification, fixture, source_root, environment=environment)


def test_qualification_rejects_changed_or_unpinned_vae_file(tmp_path):
    fixture, qualification, source_root, environment = fixture_reports(tmp_path)
    vae_file = Path(environment["WAN_VAE_PATH"])
    vae_file.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="content changed"):
        validate_qualification(qualification, fixture, source_root, environment=environment)
    vae_file.write_bytes(b"vae-weights")
    queue_path = qualification / "queue_result.json"
    queue = json.loads(queue_path.read_text())
    for key in ("source_sha256_on_queue", "source_sha256_before_launch", "source_sha256_after"):
        del queue[key][str(vae_file)]
    queue_path.write_text(json.dumps(queue))
    with pytest.raises(ValueError, match="must pin"):
        validate_qualification(qualification, fixture, source_root, environment=environment)


def test_qualification_rejects_changed_or_new_hf_model_file(tmp_path):
    fixture, qualification, source_root, environment = fixture_reports(tmp_path)
    hf_file = Path(environment["HF_MODEL_PATH"]) / "model.safetensors"
    hf_file.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="content changed"):
        validate_qualification(qualification, fixture, source_root, environment=environment)
    hf_file.write_bytes(b"hf-weights")
    (hf_file.parent / "new.safetensors").write_bytes(b"extra")
    with pytest.raises(ValueError, match="must pin"):
        validate_qualification(qualification, fixture, source_root, environment=environment)
