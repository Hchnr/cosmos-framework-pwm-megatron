# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

from tools.compare_tp4_fsdp4_microbatch_benchmark import compare


def write_run(root, backend):
    root.mkdir()
    batches = [
        {"microstep": index, "samples": count, "payload_sha256": str(index), "tokens": dict(text=3, vision=8, action=2)}
        for index, count in enumerate((2, 3, 4))
    ]
    contract = dict(
        backend=backend,
        tp=4 if backend == "tp" else 1,
        pp=1,
        dp=1 if backend == "tp" else 4,
        precision="bfloat16",
        microsteps_per_update=3,
        warmup_updates=2,
        measured_updates=5,
        per_rank_microbatches=batches,
    )
    logs = root / "training"
    if backend == "tp":
        logs /= "cosmos3_action/action128/tp4_microbatch"
    logs.mkdir(parents=True)
    for rank in range(4):
        (root / f"contract.rank{rank}.json").write_text(json.dumps(contract))
        rows = [dict(iteration=1201 + index, update_seconds=(99 if index < 2 else 6) + rank) for index in range(7)]
        (logs / f"updates.rank{rank}.jsonl").write_text("\n".join(map(json.dumps, rows)))


def test_equal_microbatches_allow_fourfold_global_batch(tmp_path):
    tp, fsdp = tmp_path / "tp", tmp_path / "fsdp"
    write_run(tp, "tp")
    write_run(fsdp, "fsdp")
    report = compare(tp, fsdp)
    assert report["tp"]["measured_mean_update_seconds"] == 9
    assert report["tp"]["amortized_microstep_seconds"] == 3
    assert report["tp"]["global_samples_per_update"] == 9
    assert report["fsdp"]["global_samples_per_update"] == 36
    assert report["tp_over_fsdp_sample_throughput"] == 0.25


def test_reject_different_microbatch_payload(tmp_path):
    tp, fsdp = tmp_path / "tp", tmp_path / "fsdp"
    write_run(tp, "tp")
    write_run(fsdp, "fsdp")
    for path in fsdp.glob("contract.rank*.json"):
        contract = json.loads(path.read_text())
        contract["per_rank_microbatches"][0]["payload_sha256"] = "different"
        path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="Per-microstep comparison mismatch"):
        compare(tp, fsdp)


def test_reject_incomplete_measured_window(tmp_path):
    tp, fsdp = tmp_path / "tp", tmp_path / "fsdp"
    write_run(tp, "tp")
    write_run(fsdp, "fsdp")
    path = fsdp / "training/updates.rank3.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="Incomplete measurement window"):
        compare(tp, fsdp)


def test_raw_scope_includes_vae_but_excludes_media_decode(tmp_path):
    tp, fsdp = tmp_path / "tp", tmp_path / "fsdp"
    write_run(tp, "tp")
    write_run(fsdp, "fsdp")
    for root in (tp, fsdp):
        for path in root.glob("contract.rank*.json"):
            contract = json.loads(path.read_text())
            for batch in contract["per_rank_microbatches"]:
                batch["payload_kind"] = "raw_before_vae_and_noising"
            path.write_text(json.dumps(contract))
    scope = compare(tp, fsdp)["scope"]
    assert "native VAE/noising" in scope
    assert "media decode and checkpoint writes excluded" in scope
