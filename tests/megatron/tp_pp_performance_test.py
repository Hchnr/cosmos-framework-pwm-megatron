"""Performance comparisons must reject mismatched inputs and repeated resumes."""

import json

import pytest

from tools.compare_tp_pp_performance import compare


def runs(root, logical_dp=4):
    tp, native = root / "tp", root / "native"
    tp.mkdir()
    native.mkdir()
    for target, seconds in ((tp, 100.0), (native, 40.0)):
        for rank in range(4):
            rows = [
                {"iteration": iteration, "update_seconds": seconds + rank, "loss": 0.02} for iteration in (1201, 1202)
            ]
            (target / f"updates.rank{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        for iteration in (1200, 1201):
            owners = [0] if target == tp else range(4)
            for owner in owners:
                records = [
                    {
                        "slot": [iteration, micro, logical],
                        "source": "video",
                        "samples": logical + 1,
                        "image": False,
                        "tokens": {"text": 10, "vision": 20, "action": 3},
                        "sha256": f"raw-{micro}-{logical}",
                        "prepared_noise_sha256": f"noise-{micro}-{logical}",
                        "prepared_packed_sha256": f"pack-{micro}-{logical}",
                    }
                    for micro in range(3)
                    for logical in range(logical_dp)
                    if target == tp or logical % 4 == owner
                ]
                (target / f"replay_plan.iter{iteration:09d}.rank{owner}.json").write_text(json.dumps(records))
    return tp, native


def test_matched_inputs_use_slowest_rank_and_exclude_warmup(tmp_path):
    tp, native = runs(tmp_path)
    report = compare(tp, native, warmup_updates=1)
    assert report["inputs_matched"] and report["bf16_loss_trajectory_passed"]
    assert report["measured_updates"] == 1
    assert report["tp_mean_update_seconds"] == 103.0
    assert report["native_mean_update_seconds"] == 43.0
    assert report["matched_tp_throughput_over_native"] == 43 / 103
    assert report["updates"][1]["samples"] == 30
    assert report["matched_samples_per_second"] == {"tp": 30 / 103, "native": 30 / 43}
    assert report["matched_tokens_per_second"]["vision"] == {"tp": 240 / 103, "native": 240 / 43}


def test_full576_comparison_requires_explicit_logical_topology(tmp_path, monkeypatch):
    from tools.compare_tp_pp_performance import main

    tp, native = runs(tmp_path, logical_dp=192)
    with pytest.raises(ValueError, match="Missing or unexpected slots"):
        compare(tp, native, warmup_updates=1)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_tp_pp_performance",
            "--tp-run",
            str(tp),
            "--native-run",
            str(native),
            "--logical-dp",
            "192",
            "--reference-microsteps",
            "3",
            "--warmup-updates",
            "1",
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["inputs_matched"]
    assert report["global_packed_microbatches_per_update"] == 576
    assert all(row["logical_slots"] == 576 for row in report["updates"])
    assert report["matched_samples_per_second"] == {"tp": 55584 / 103, "native": 55584 / 43}
    assert not report["production_576_slot_acceptance"]


def test_changed_noise_refuses_matched_throughput(tmp_path):
    tp, native = runs(tmp_path)
    path = native / "replay_plan.iter000001201.rank0.json"
    rows = json.loads(path.read_text())
    rows[0]["prepared_noise_sha256"] = "different"
    path.write_text(json.dumps(rows))
    report = compare(tp, native, warmup_updates=1)
    assert not report["inputs_matched"]
    assert report["matched_tp_throughput_over_native"] is None
    assert report["matched_samples_per_second"] is None
    assert report["matched_tokens_per_second"] is None
    assert report["input_mismatches"] == [{"slot": (1201, 0, 0), "field": "prepared_noise_sha256"}]


def test_phase_accounting_uses_one_critical_rank_instead_of_cross_rank_maxima(tmp_path):
    tp, native = runs(tmp_path)
    for rank in range(4):
        path = tp / f"updates.rank{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            prepare, backward, optimizer = (95.0, 1.0, 1.0) if rank == 0 else (20.0, 80.0, 2.0)
            row.update(
                parallel_preparation=True,
                parallel_prepare_wall_seconds=prepare,
                forward_backward_host_seconds=backward,
                optimizer_and_sync_host_seconds=optimizer,
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    row = compare(tp, native, warmup_updates=1)["updates"][1]
    phases = row["tp_critical_rank_host_phases"]
    assert phases["rank_log"] == "updates.rank3.jsonl"
    assert phases["seconds"]["parallel_prepare_wall_seconds"] == 20.0
    assert phases["unassigned_host_seconds"] == 1.0
    assert phases["accounting_consistent"]
    assert row["native_critical_rank_host_phases"] is None


def test_absent_hashes_cannot_prove_matched_inputs(tmp_path):
    tp, native = runs(tmp_path)
    for target in (tp, native):
        for path in target.glob("replay_plan.*.json"):
            records = json.loads(path.read_text())
            for row in records:
                row["sha256"] = None
            path.write_text(json.dumps(records))
    report = compare(tp, native, warmup_updates=1)
    assert not report["inputs_matched"]
    assert report["matched_tp_throughput_over_native"] is None


def test_repeated_resume_records_cannot_be_treated_as_continuous_updates(tmp_path):
    tp, native = runs(tmp_path)
    path = tp / "updates.rank0.jsonl"
    with path.open("a") as handle:
        handle.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Repeated iterations"):
        compare(tp, native, warmup_updates=1)


def test_legacy_loss_correction_is_explicit_and_rejects_corrected_metrics(tmp_path):
    tp, native = runs(tmp_path)
    for path in tp.glob("updates.rank*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["loss"] /= 12
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert not compare(tp, native, warmup_updates=1)["bf16_loss_trajectory_passed"]
    corrected = compare(tp, native, warmup_updates=1, legacy_tp_loss_scale=12)
    assert corrected["bf16_loss_trajectory_passed"]
    assert corrected["legacy_tp_loss_reporting_scale"] == 12
    assert corrected["updates"][0]["tp_loss_as_logged"] == 0.02 / 12
    path = tp / "updates.rank0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["loss_reporting"] = "unscaled_microbatch_mean_v1"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="versioned metrics"):
        compare(tp, native, warmup_updates=1, legacy_tp_loss_scale=12)
