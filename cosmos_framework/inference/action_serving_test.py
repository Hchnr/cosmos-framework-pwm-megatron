"""CPU checks for the Action128 contract, HTTP admission and rank coordination."""

import base64
import io
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fastapi.testclient import TestClient
from PIL import Image

from cosmos_framework.data.generator.sequence_packing.spatial import action128_seam_position
from cosmos_framework.inference.action_server import DistributedActionService, deployment_plan, parser
from cosmos_framework.inference.action_serving import (
    PreparedAction128Request,
    build_batch,
    create_app,
    prepare_wam_request,
    resolve_checkpoint,
)
from cosmos_framework.inference.args import OmniSampleOverrides
from cosmos_framework.inference.common.args import SampleOutput, SampleOutputs


def spec():
    return {
        "registry": {
            "domains": ["robot/source-a", "robot/source-b"],
            "embodiments": ["robot"],
            "domain_embodiments": [0, 0],
            "domain_projection_ids": [3, 3],
        },
        "transforms": [],
    }


def payload(roles=("main",)):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (12, 34, 56)).save(buffer, "PNG")
    image = base64.b64encode(buffer.getvalue()).decode()
    return {
        "action_domain": "robot/source-b",
        "task_title": "pick",
        "prompt": "Pick up the object",
        "cameras": [{"role": role, "image_base64": image, "fps": 15} for role in roles],
        "action_fps": 50,
        "action_dim_mask": [True] * 10 + [False] * 118,
    }


def checkpoint(tmp_path, step=200):
    run = tmp_path / "project/group/run"
    model = run / f"checkpoints/iter_{step:09d}/model"
    model.mkdir(parents=True, exist_ok=True)
    (model / ".metadata").touch()
    (model / "__0_0.distcp").touch()
    (run / "config.yaml").write_text("model: {}\n")
    (model.parent.parent / "latest_checkpoint.txt").write_text(model.parent.name)
    return model, run / "config.yaml"


def test_latest_ignores_inflight_and_snapshot_is_immutable(tmp_path):
    model, config = checkpoint(tmp_path)
    (model.parent.parent / "iter_000000400/model").mkdir(parents=True)
    assert resolve_checkpoint(tmp_path) == (model, config)
    assert resolve_checkpoint(config.parent) == (model, config)
    assert resolve_checkpoint(model.parent.parent) == (model, config)
    assert resolve_checkpoint(model) == (model, config)
    new_model, _ = checkpoint(tmp_path, 400)
    assert model.name == "model" and model.parent.name == "iter_000000200"
    assert resolve_checkpoint(tmp_path)[0] == new_model
    with pytest.raises(ValueError, match="completion evidence"):
        resolve_checkpoint(model)
    (model.parent / ".save_complete").touch()
    assert resolve_checkpoint(model)[0] == model


def test_bad_pointer_missing_files_and_ambiguous_runs(tmp_path):
    model, _ = checkpoint(tmp_path)
    pointer = model.parent.parent / "latest_checkpoint.txt"
    pointer.write_text("../elsewhere")
    with pytest.raises(ValueError, match="Invalid latest"):
        resolve_checkpoint(tmp_path)
    pointer.write_text("iter_000000201")
    with pytest.raises(ValueError, match="Missing DCP"):
        resolve_checkpoint(tmp_path)
    pointer.write_text(model.parent.name)
    checkpoint(tmp_path / "another")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_checkpoint(tmp_path)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8, 16])
def test_deployment_card_count(tmp_path, monkeypatch, world_size):
    checkpoint(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    flags = ["--no-reuse-small-action-metadata"] if world_size > 1 else []
    plan = deployment_plan(parser().parse_args(["--checkpoint-root", str(tmp_path), *flags]))
    assert plan["dp_shard_size"] == world_size
    assert plan["cp_size"] == plan["cfgp_size"] == 1
    assert plan["ema_weights"] is False
    assert plan["reuse_small_action_metadata"] is (world_size == 1)


@pytest.mark.parametrize(
    "flags,backend,compile_enabled,reuse_metadata",
    [
        ([], "flash3", True, True),
        (["--no-compile"], "flash3", False, True),
        (["--no-reuse-small-action-metadata"], "flash3", True, False),
        (["--attention-backend", "sdpa", "--no-compile", "--no-reuse-small-action-metadata"], "sdpa", False, False),
        (
            ["--attention-backend", "flash3_video", "--compile", "--reuse-small-action-metadata"],
            "flash3_video",
            True,
            True,
        ),
    ],
)
def test_serving_profile_defaults_and_explicit_overrides(tmp_path, flags, backend, compile_enabled, reuse_metadata):
    from cosmos_framework.inference.action_server import model_overrides

    checkpoint(tmp_path)
    args = parser().parse_args(["--checkpoint-path", str(tmp_path), *flags])
    plan = deployment_plan(args)
    assert plan["attention_backend"] == backend
    assert plan["compile"] is compile_enabled
    assert plan["reuse_small_action_metadata"] is reuse_metadata
    assert not plan["cuda_graphs"] and not plan["reuse_small_action_und"] and not plan["fuse_qkv"]
    assert f"model.config.action_tower_attention_backend={backend}" in model_overrides(args)


def test_multirank_dry_run_requires_explicit_metadata_opt_out(tmp_path, monkeypatch, capsys):
    import sys

    from cosmos_framework.inference.action_server import main

    checkpoint(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "2")
    argv = ["action128_server", "--checkpoint-path", str(tmp_path), "--dry-run"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="--no-reuse-small-action-metadata"):
        main()
    monkeypatch.setattr(sys, "argv", [*argv, "--no-reuse-small-action-metadata"])
    main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["world_size"] == 2 and plan["reuse_small_action_metadata"] is False


@pytest.mark.parametrize(
    "roles", [("main",), ("main", "left_wrist"), ("right_wrist", "main", "left_wrist"), ("auxiliary_0", "left_wrist")]
)
def test_training_camera_contract_and_native_timestamps(roles):
    request = PreparedAction128Request.model_validate(payload(roles))
    batch = build_batch(request, spec())
    video = batch["video"][0][0]
    assert video.dtype == torch.uint8
    assert video.shape[1] == 33  # 2 sec at 15 Hz, VAE rounds to 4k+1
    assert batch["action"][0][0].shape == (100, 128)  # native 50 Hz is independent
    assert batch["normalization_domain_id"] == [1]
    assert batch["domain_id"][0].item() == 3  # same projection, different normalization source
    assert batch["action_processing_record"][0].action_normalizer is None
    plan = batch["sequence_plan"][0]
    assert plan.condition_frame_indexes_vision == [0]
    assert not plan.condition_frame_indexes_action and not plan.robot_state_frame_indexes
    assert plan.action_timestamps_seconds[-1] == pytest.approx(1.98)
    meta = batch["decode_metadata"][0]
    seam = action128_seam_position(meta, tuple(video.shape[-2:]), 32)
    if len(roles) == 1:
        assert seam is None
    else:
        assert seam == meta["camera_layout"][0][3] / 32
    assert video[:, 0].max() == 56
    assert not video[:, 1:].any()  # future pixels are never supplied as policy conditions
    caption = json.loads(batch["ai_caption"][0])
    assert caption["fps"] == 15 and caption["action_frequency_hz"] == 50
    assert caption["command"] == request.prompt


def test_build_batch_preserves_sparse_slot_mask_independently_of_normalization():
    active_slots = {0, 6, 9, 14, 63, 101, 127}
    request = PreparedAction128Request.model_validate(
        payload() | {"action_dim_mask": [slot in active_slots for slot in range(128)]}
    )
    source_spec = spec()
    normalization_mask = [slot in {0, 6, 33} for slot in range(128)]
    source_spec["transforms"] = [{"normalization_mask": normalization_mask} for _ in range(2)]

    batch = build_batch(request, source_spec)

    assert len(batch["action_dim_mask"]) == len(batch["action"]) == 1
    assert len(batch["action_dim_mask"][0]) == len(batch["action"][0]) == 1
    slot_mask = batch["action_dim_mask"][0][0]
    assert slot_mask.dtype == torch.bool and slot_mask.shape == (128,)
    assert torch.equal(slot_mask, torch.tensor(request.action_dim_mask, dtype=torch.bool))
    assert slot_mask[63] and not normalization_mask[63]  # Active identity slots must stay active.
    assert normalization_mask[33] and not slot_mask[33]  # A transform does not enable a caller's slot.
    assert batch["raw_action_dim"][0].item() == 128
    assert "action_loss_mask" not in batch  # Static slot validity is not temporal supervision.


def test_build_batch_slot_masks_do_not_leak_between_requests():
    first_request = PreparedAction128Request.model_validate(
        payload() | {"action_dim_mask": [slot in {0, 19, 96, 127} for slot in range(128)]}
    )
    second_request = PreparedAction128Request.model_validate(
        payload() | {"action_dim_mask": [slot in {2, 14, 70} for slot in range(128)]}
    )
    first_batch = build_batch(first_request, spec())
    second_batch = build_batch(second_request, spec())
    first_mask = first_batch["action_dim_mask"][0][0]
    second_mask = second_batch["action_dim_mask"][0][0]

    assert torch.equal(first_mask, torch.tensor(first_request.action_dim_mask, dtype=torch.bool))
    assert torch.equal(second_mask, torch.tensor(second_request.action_dim_mask, dtype=torch.bool))
    first_mask.zero_()
    assert first_request.action_dim_mask[0]
    assert torch.equal(second_mask, torch.tensor(second_request.action_dim_mask, dtype=torch.bool))
    rebuilt = build_batch(first_request, spec())["action_dim_mask"][0][0]
    assert torch.equal(rebuilt, torch.tensor(first_request.action_dim_mask, dtype=torch.bool))


def wam_payload(tmp_path):
    image = tmp_path / "observation.png"
    Image.new("RGB", (64, 48), (12, 34, 56)).save(image)
    return {
        "name": "policy",
        "model_mode": "wam",
        "prompt": "Pick up the object",
        "vision_path": str(image),
        "domain_name": "robot",
        "action_chunk_size": 100,
        "fps": 15,
        "seed": 42,
        "num_steps": 35,
        "guidance": 1.5,
        "shift": 5,
        "extra": {
            "action128": {
                "action_domain": "robot/source-b",
                "action_dim_mask": payload()["action_dim_mask"],
                "action_fps": 50,
            }
        },
    }


class FakeService:
    spec = spec()
    video_key = "video"

    def __init__(self, tmp_path):
        from types import SimpleNamespace

        self.calls = 0
        self.info = {"checkpoint_path": "iter_200/model", "world_size": 2, "output_dir": str(tmp_path / "outputs")}
        self.config = SimpleNamespace(
            resolution="720",
            action_gen=True,
            sound_gen=False,
            tokenizer=SimpleNamespace(temporal_compression_factor=4),
            vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
        )

    def prepare(self, request, output_dir):
        return prepare_wam_request(request, self.spec, self.config, output_dir, self.video_key)

    def predict(self, request, batch):
        self.calls += 1
        return SampleOutputs(args=request.sample_args, outputs=[SampleOutput(content={"action": [[1.0] * 128]})])


def test_http_validation_health_and_domain_discovery(tmp_path):
    service = FakeService(tmp_path)
    client = TestClient(create_app(service))
    request = wam_payload(tmp_path)
    assert client.get("/healthz").json() == {"status": "ready"}
    assert client.get("/info").json()["world_size"] == 2
    assert client.get("/models").json() == ["action128"]
    assert client.get("/domains").json()["domains"] == spec()["registry"]["domains"]
    response = client.post("/generate", json=request)
    assert response.status_code == 200, response.text
    assert SampleOutputs.model_validate(response.json()).outputs[0].content["action"] == [[1.0] * 128]
    for patch in [
        {"model_mode": "t2v"},
        {"action_fps": 0},
        {"duration_seconds": 3},
        {"state": [0] * 128},
        {"extra": {"action128": {"action_domain": "wrong", "action_dim_mask": [True] * 128}}},
        {"extra": {"action128": {"action_domain": "robot/source-b", "action_dim_mask": [False] * 128}}},
    ]:
        response = client.post("/generate", json=request | patch)
        assert response.status_code == 422, response.text
    assert client.post("/generate", json=request | {"model": "missing"}).status_code == 404
    assert client.post("/predict", json=payload()).status_code == 404
    assert service.calls == 1


def test_busy_request_is_rejected(tmp_path):
    import threading

    service = FakeService(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def blocked(request, batch):
        entered.set()
        assert release.wait(20)
        return SampleOutputs(args=request.sample_args)

    service.predict = blocked
    client = TestClient(create_app(service))
    request = wam_payload(tmp_path)
    thread = threading.Thread(target=lambda: client.post("/generate", json=request))
    thread.start()
    try:
        assert entered.wait(20)
        response = client.post("/generate", json=request)
        assert response.status_code == 503 and response.headers["Retry-After"] == "1"
        assert client.get("/healthz").status_code == 200
    finally:
        release.set()
        thread.join(20)
    assert not thread.is_alive()


def test_original_request_type_and_submit_file_loader_preserve_action128_extensions(tmp_path):
    request = wam_payload(tmp_path)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(request))
    loaded = OmniSampleOverrides.from_files([path])[0]
    assert loaded.extra == request["extra"]
    service = FakeService(tmp_path)
    prepared, batch = service.prepare(loaded, tmp_path / "prepared")
    assert prepared.sample_args["action_chunk_size"] == 100
    assert prepared.sample_args["fps"] == 15
    assert prepared.sample_args["num_frames"] == 33
    assert batch["video"][0][0].shape == (3, 33, 48, 64)
    assert batch["action"][0][0].shape == (100, 128)
    assert batch["sequence_plan"][0].action_timestamps_seconds[-1] == pytest.approx(1.98)


class ProtocolService(DistributedActionService):
    def execute(self, request, batch):
        # Exercise real object broadcast of tensors + SequencePlan across ranks.
        assert batch["action"][0][0].shape == (100, 128)
        self.seeds.append(request.seed)
        return {"seed": request.seed}


def distributed_worker(rank, world_size, directory):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=40),
    )
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=40))
        service = ProtocolService.__new__(ProtocolService)
        service.rank, service.group, service.seeds = rank, group, []
        if rank == 0:
            for seed in [42, 43]:
                request = PreparedAction128Request.model_validate(payload() | {"seed": seed})
                assert service.predict(request, build_batch(request, spec())) == {"seed": seed}
            dist.broadcast_object_list([("stop", None, None)], src=0, group=group)
        else:
            service.worker_loop()
        Path(directory, f"rank{rank}.json").write_text(json.dumps(service.seeds))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3])
def test_real_multirank_dispatch_and_shutdown(tmp_path, world_size):
    mp.spawn(distributed_worker, args=(world_size, str(tmp_path)), nprocs=world_size, join=True)
    for rank in range(world_size):
        assert json.loads((tmp_path / f"rank{rank}.json").read_text()) == [42, 43]


def test_execute_uses_saved_inverse_transform_once_and_masks_slots(monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.utils import misc

    request = PreparedAction128Request.model_validate(payload())
    received = {}

    def generate(batch, **kwargs):
        received.update(kwargs)
        return {"action": [torch.full((100, 128), 0.5)]}

    def denormalize(action, domain):
        assert domain == 1
        assert torch.equal(action, torch.full((100, 128), 0.5))
        return action * 2 + 3

    service = DistributedActionService.__new__(DistributedActionService)
    service.model = SimpleNamespace(
        generate_samples_from_batch=generate,
        net=SimpleNamespace(action_normalization=SimpleNamespace(denormalize=denormalize)),
    )
    service.rank, service.spec, service.info = 0, spec(), {"checkpoint_path": "iter_200/model"}
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(misc, "to", lambda batch, **_: batch)
    # Keep response masking on CPU while exercising execute's output contract.
    result = service.execute(request, build_batch(request, spec()))
    assert len(result["action"]) == 100
    assert result["action"][0] == [4.0] * 10 + [0.0] * 118
    assert result["normalized_action"][0] == [0.5] * 10 + [0.0] * 118
    assert result["action_units"] == "raw_physical_or_source_units"
    assert received == {"seed": [42], "num_steps": 35, "guidance": 1.5, "shift": 5.0}


def test_inference_overrides_disable_training_buckets_without_changing_contract(tmp_path):
    from cosmos_framework.inference.action_server import model_overrides

    args = parser().parse_args(
        [
            "--checkpoint-root",
            str(tmp_path),
            "--vae-path",
            str(tmp_path / "vae.pth"),
            "--tokenizer-path",
            str(tmp_path / "processor"),
        ]
    )
    overrides = model_overrides(args)
    assert "model.config.fixed_sequence_length.enabled=false" in overrides
    assert "model.config.action_tower_und_short_tokens=0" in overrides
    assert "model.config.action_tower_init=checkpoint" in overrides
    assert not any("action128_spec" in override or "vision_spatial_position_mode" in override for override in overrides)
    assert any("tokenizer.vae_path=" in override for override in overrides)


@pytest.mark.parametrize("gpus", [1, 2, 4, 8])
def test_launcher_ignores_training_card_counts_and_forwards_args(tmp_path, gpus):
    import os
    import subprocess
    import sys

    capture = tmp_path / "command.json"
    fake = tmp_path / "torchrun"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({'argv':sys.argv[1:],"
        "'training':os.environ['COSMOS_TRAINING'],'ld':os.environ['LD_LIBRARY_PATH']}))\n"
    )
    fake.chmod(0o755)
    repo = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "CHECKPOINT_ROOT": "/shared/training",
        "CHECKPOINT_PATH": "/shared/training/checkpoints/iter_000006000/model",
        "NNODES": "18",
        "NPROC_PER_NODE": "8",
        "MASTER_PORT": "50012",
        "SERVING_GPUS": str(gpus),
        "SERVING_NNODES": "1",
    }
    subprocess.run(
        ["bash", str(repo / "examples/launch_action128_serving.sh"), "--dry-run"],
        env=env,
        check=True,
        capture_output=True,
        timeout=10,
    )
    result = json.loads(capture.read_text())
    assert f"--nproc-per-node={gpus}" in result["argv"]
    assert "--nnodes=1" in result["argv"]
    assert "--master-port=29571" in result["argv"]
    assert result["argv"][-1] == "--dry-run"
    assert result["argv"][result["argv"].index("--checkpoint-path") + 1] == env["CHECKPOINT_PATH"]
    assert result["training"] == "0" and result["ld"] == ""


@pytest.mark.parametrize("structured", [False, True])
def test_service_accepts_saved_contract_as_dict_or_omegaconf(monkeypatch, structured):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    calls = []
    normalizer = SimpleNamespace(
        validate_contract=lambda: calls.append("validate"), _reset_transforms=lambda: calls.append("restore_fp32")
    )
    model = SimpleNamespace(
        eval=lambda: calls.append("eval"),
        input_video_key="video",
        config=SimpleNamespace(action128_spec=OmegaConf.create(spec()) if structured else spec(), action_gen=True),
        net=SimpleNamespace(action_normalization=normalizer),
    )
    service = DistributedActionService(SimpleNamespace(model=model), None, {})
    assert service.spec == spec()
    assert calls == ["eval", "validate", "restore_fp32"]


def test_one_rank_torchrun_uses_existing_agent_store(monkeypatch):
    from cosmos_framework.inference.action_server import initialize_single_rank_distributed

    calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
    initialize_single_rank_distributed()
    assert calls == [(("nccl",), {"init_method": "env://"})]


def test_legacy_domain_label_and_json_prompt_are_preserved(tmp_path):
    service = FakeService(tmp_path)
    request = wam_payload(tmp_path)
    caption = {
        "task_title": "Pick",
        "command": "Pick up the object",
        "task_plan": ["approach", "grasp"],
        "current_subtask": "grasp",
        "duration": "stale",
        "resolution": {"H": 1, "W": 1},
    }
    request.update(domain_name="libero", prompt=json.dumps(caption))
    prepared, batch = service.prepare(OmniSampleOverrides.model_validate(request), tmp_path / "prepared")
    formatted = json.loads(batch["ai_caption"][0])
    assert formatted["task_plan"] == caption["task_plan"]
    assert formatted["current_subtask"] == "grasp"
    assert formatted["duration"] == "2s" and formatted["resolution"] == {"H": 48, "W": 64}
    assert prepared.sample_args["domain_name"] == "libero"
    assert batch["domain_id"][0].item() == 3


def test_image_size_resizes_stitched_canvas_and_actual_seam(tmp_path):
    service = FakeService(tmp_path)
    request = wam_payload(tmp_path)
    Image.new("RGB", (64, 96), (12, 34, 56)).save(request["vision_path"])
    request["image_size"] = 48
    request["extra"]["action128"].update(
        selected_roles=["main", "left_wrist"], camera_layout=[["main", "top", 64, 48], ["left_wrist", "bottom", 64, 48]]
    )
    _, batch = service.prepare(OmniSampleOverrides.model_validate(request), tmp_path / "prepared")
    video = batch["video"][0][0]
    assert video.shape == (3, 33, 48, 32)
    meta = batch["decode_metadata"][0]
    assert meta["camera_layout"] == [["main", "top", 32, 24], ["left_wrist", "bottom", 32, 24]]
    assert action128_seam_position(meta, (48, 32), 32) == 0.75


def test_caption_frequency_matches_training_float_format(tmp_path):
    service = FakeService(tmp_path)
    request = wam_payload(tmp_path)
    _, batch = service.prepare(OmniSampleOverrides.model_validate(request), tmp_path / "prepared")
    caption = json.loads(batch["ai_caption"][0])
    assert type(caption["fps"]) is float
    assert caption["fps"] == 15.0


def test_explicit_checkpoint_path_cli_alias(tmp_path):
    model, _ = checkpoint(tmp_path)
    args = parser().parse_args(["--checkpoint-path", str(model)])
    assert deployment_plan(args)["checkpoint_path"] == str(model)


@pytest.mark.parametrize("reuse_und,reuse_metadata", [(False, False), (True, False), (False, True), (True, True)])
def test_action_only_http_result_skips_decode_and_preserves_output_contract(
    monkeypatch, tmp_path, reuse_und, reuse_metadata
):
    from types import SimpleNamespace

    from cosmos_framework.utils import misc

    request = PreparedAction128Request.model_validate(payload())
    request.return_video = False
    request.sample_args = {
        "guidance_interval": None,
        "normalize_cfg": False,
        "sigma_max": 80.0,
        "output_dir": str(tmp_path / "request"),
        "name": "action_only",
        "fps": 20,
        "video_save_quality": 5,
    }
    Path(request.sample_args["output_dir"]).mkdir()
    calls = []

    def generate(batch, **kwargs):
        calls.append(kwargs)
        # Deliberately no vision output: decoding must never be attempted.
        return {"action": [torch.ones(100, 128)]}

    service = DistributedActionService.__new__(DistributedActionService)
    service.rank, service.spec = 0, spec()
    checkpoint(tmp_path / "checkpoint")
    flags = ["--reuse-small-action-und"] if reuse_und else []
    if not reuse_metadata:
        flags.append("--no-reuse-small-action-metadata")
    service.info = deployment_plan(
        parser().parse_args(["--checkpoint-path", str(tmp_path / "checkpoint"), "--output-dir", str(tmp_path), *flags])
    )
    service.model = SimpleNamespace(
        generate_samples_from_batch=generate,
        net=SimpleNamespace(action_normalization=SimpleNamespace(denormalize=lambda x, _: x)),
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(misc, "to", lambda batch, **_: batch)
    result = service.execute(request, build_batch(request, spec()))
    assert result.outputs[0].files == []
    assert len(result.outputs[0].content["action128"]["conditioning_caption_sha256"]) == 64
    assert result.outputs[0].content["action"][0] == [1.0] * 10 + [0.0] * 118
    assert calls[0].get("reuse_small_action_und", False) is reuse_und
    assert calls[0].get("reuse_small_action_metadata", False) is reuse_metadata
    assert (Path(request.sample_args["output_dir"]) / "sample_outputs.json").is_file()


@pytest.mark.parametrize("as_list", [False, True])
def test_explicit_startup_warmup_repeats_requests_and_records_readiness(tmp_path, as_list):
    from cosmos_framework.inference.action_server import warmup_service

    service = FakeService(tmp_path)
    request = wam_payload(tmp_path)
    request["vision_path"] = "observation.png"  # Relative to warmup JSON, as in the normal file loader.
    requests = [request, {**request, "seed": 43}] if as_list else request
    input_path = tmp_path / "warmup.json"
    input_path.write_text(json.dumps(requests))
    warmup_service(service, input_path, tmp_path / "outputs")
    status = service.info["warmup"]
    assert status["completed"] and status["requested"]
    assert len(status["input_sha256"]) == 64
    assert service.calls == (4 if as_list else 2)
    assert len(status["requests"]) == (2 if as_list else 1)
    assert all(len(item["repeat_seconds"]) == 2 for item in status["requests"])
    assert status["requests"][0]["seed"] == 42
    assert status["requests"][0]["return_video"] is True
    assert len(status["requests"][0]["conditioning_caption_sha256"]) == 64


@pytest.mark.parametrize("requests", [[], [1]])
def test_startup_warmup_rejects_empty_or_non_request_lists(tmp_path, requests):
    from cosmos_framework.inference.action_server import warmup_service

    input_path = tmp_path / "warmup.json"
    input_path.write_text(json.dumps(requests))
    service = FakeService(tmp_path)
    with pytest.raises(ValueError, match="nonempty Cosmos request"):
        warmup_service(service, input_path, tmp_path / "outputs")
    assert service.calls == 0
