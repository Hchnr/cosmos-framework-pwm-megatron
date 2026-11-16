from argparse import Namespace
from pathlib import Path

import pytest

from tools.cosmos3_full_opt_demo import (
    GateError,
    InputPaths,
    RunContext,
    assert_descendant,
    job_name,
    parse_aot_results,
    parse_losses,
    parse_selected_gpus,
    training_overrides,
    unresolved_strings,
    validate_checkpoint,
    validate_repository_transition,
    validate_root_boundaries,
)
from tools.cosmos3_train_rank_entry import configure_rank_cache


def _context(tmp_path: Path, budget: int = 45056) -> RunContext:
    parent = tmp_path / "parent"
    task = tmp_path / "task"
    inputs = InputPaths(
        parent_root=parent,
        parent_record=parent / "records/source",
        dataset=parent / "data/dataset",
        edge_hf=parent / "checkpoints/hf/edge",
        base_dcp=parent / "checkpoints/dcp/edge",
        wan_vae=parent / "checkpoints/vae.pth",
    )
    name = job_name(budget)
    return RunContext(
        args=Namespace(packing_token_budget=budget, gpus="0,1,2,3,4,5,6,7"),
        task_root=task,
        inputs=inputs,
        run_id="unit-test",
        record_dir=task / "records/unit-test",
        job_name=name,
        run_dir=task / "outputs/train/cosmos3/sft" / name,
        python=tmp_path / ".venv/bin/python",
        torchrun=tmp_path / ".venv/bin/torchrun",
        compile_cache_root=tmp_path / "compile",
    )


def test_gpu_list_requires_exactly_eight_unique_numeric_indices() -> None:
    assert parse_selected_gpus("7,6,5,4,3,2,1,0") == ["7", "6", "5", "4", "3", "2", "1", "0"]
    for invalid in ("0,1,2,3,4,5,6", "0,1,2,3,4,5,6,6", "0,1,2,3,4,5,6,x"):
        with pytest.raises(GateError):
            parse_selected_gpus(invalid)


def test_budget_is_encoded_in_both_packing_overrides_and_job_name(tmp_path: Path) -> None:
    ctx = _context(tmp_path, 32768)
    overrides = training_overrides(ctx)
    assert f"job.name={ctx.job_name}" in overrides
    assert "model.config.max_num_tokens_after_packing=32768" in overrides
    assert "dataloader_train.max_sequence_length=32768" in overrides
    assert "dataloader_train.max_samples_per_batch=null" in overrides
    assert "optimizer.keys_to_select=[]" in overrides
    assert "model.config.parallelism.data_parallel_shard_degree=4" in overrides
    assert "model.config.parallelism.data_parallel_replicate_degree=2" in overrides
    assert "model.config.parallelism.context_parallel_shard_degree=2" in overrides


def test_job_name_rejects_unplanned_budget() -> None:
    with pytest.raises(GateError):
        job_name(16384)


def test_attempt_tag_creates_distinct_safe_job_name() -> None:
    assert job_name(45056, "triton_local").endswith("seq45056_full_params__triton_local")
    with pytest.raises(GateError, match="invalid attempt tag"):
        job_name(45056, "../escape")


def test_rank_entry_uses_isolated_local_rank_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "compile"
    monkeypatch.setenv("COSMOS_COMPILE_CACHE_ROOT", str(root))
    monkeypatch.setenv("LOCAL_RANK", "3")
    inductor, triton = configure_rank_cache()
    assert inductor == root / "torchinductor/rank_3"
    assert triton == root / "triton/rank_3"
    assert inductor.is_dir() and triton.is_dir()
    assert __import__("os").environ["TORCHINDUCTOR_CACHE_DIR"] == str(inductor)
    assert __import__("os").environ["TRITON_CACHE_DIR"] == str(triton)


def test_path_gates_reject_root_overlap_and_escape(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    task = tmp_path / "task"
    assert validate_root_boundaries(task, parent) == (task.resolve(), parent.resolve())
    with pytest.raises(GateError):
        validate_root_boundaries(parent / "child", parent)
    with pytest.raises(GateError):
        assert_descendant(tmp_path / "elsewhere", parent, "escape")


def test_repository_transition_is_explicit_and_requires_clean_final_tree() -> None:
    unchanged = validate_repository_transition(
        " M plan.md\n", " M plan.md\n", allow_committed_transition=False, preflight_commit="a", final_commit="a"
    )
    assert unchanged["mode"] == "unchanged"
    with pytest.raises(GateError, match="repository changed since preflight"):
        validate_repository_transition(
            "?? runner.py\n", "", allow_committed_transition=False, preflight_commit="a", final_commit="b"
        )
    with pytest.raises(GateError, match="clean final worktree"):
        validate_repository_transition(
            "?? runner.py\n",
            " M runner.py\n",
            allow_committed_transition=True,
            preflight_commit="a",
            final_commit="b",
        )
    transition = validate_repository_transition(
        "?? runner.py\n", "", allow_committed_transition=True, preflight_commit="a", final_commit="b"
    )
    assert transition == {
        "mode": "committed_transition",
        "preflight_commit": "a",
        "final_commit": "b",
        "preflight_status": "?? runner.py\n",
        "final_status": "",
    }


def test_unresolved_interpolations_are_reported_with_paths() -> None:
    value = {"model": {"path": "${oc.env:MISSING}"}, "items": ["ok", "???"]}
    assert unresolved_strings(value) == ["model.path=${oc.env:MISSING}", "items[1]=???"]


def test_loss_parser_requires_finite_values() -> None:
    text = "Iteration 1: Hit counter: 1/50 | Loss: 3.1250 | Time: 1s\nIteration 2: Loss: 2.5e-1"
    assert parse_losses(text) == {1: 3.125, 2: 0.25}
    with pytest.raises(GateError):
        parse_losses("Iteration 1: Loss: NaN")


def test_aot_parser_requires_every_rank_and_every_variant() -> None:
    complete = "\n".join(
        f"[RANK {rank}] Rank {rank}: AOT compiled 2, loaded 16/16 chunk variants, time: {10 + rank}.5s"
        for rank in range(8)
    )
    result = parse_aot_results(complete)
    assert result["ranks"] == 8
    assert result["variants"] == 16
    assert result["max_seconds"] == 17.5

    with pytest.raises(GateError, match="completion ranks"):
        parse_aot_results("\n".join(complete.splitlines()[:-1]))
    with pytest.raises(GateError, match="did not load every variant"):
        parse_aot_results(complete.replace("loaded 16/16", "loaded 15/16", 1))


def test_checkpoint_validator_checks_pointer_and_required_components(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints/iter_000000006"
    for component in ("model", "optim", "scheduler", "trainer"):
        directory = checkpoint / component
        directory.mkdir(parents=True)
        (directory / ".metadata").write_bytes(b"metadata")
        (directory / "__0_0.distcp").write_bytes(b"shard")
    pointer = run_dir / "checkpoints/latest_checkpoint.txt"
    pointer.write_text("iter_000000006\n")

    result = validate_checkpoint(run_dir, 6)
    assert result["components"]["model"]["shards"] == 1
    assert result["components"]["dataloader"]["status"] == "unsupported_by_vision_sft_edge_recipe"

    pointer.write_text("../../outside\n")
    with pytest.raises(GateError, match="unsafe checkpoint pointer"):
        validate_checkpoint(run_dir, 6)
