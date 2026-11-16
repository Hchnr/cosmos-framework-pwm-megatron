# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
    _load_tokenizer_aot_artifact,
    _publish_tokenizer_aot_artifact,
)


def _packages(tmp_path: Path) -> dict[tuple[int, int, int, int], str]:
    first = tmp_path / "chunk_ct0_1f_128x128.pt2"
    second = tmp_path / "chunk_ct2_8f_128x128.pt2"
    first.write_bytes(b"package-zero")
    second.write_bytes(b"package-two")
    return {
        (1, 128, 128, 0): str(first),
        (8, 128, 128, 2): str(second),
    }


def test_tokenizer_artifact_publish_and_load_round_trip(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    packages = _packages(tmp_path)
    _publish_tokenizer_aot_artifact(artifact, "fingerprint", packages)

    assert (artifact / "READY").read_text().strip() == "fingerprint"
    loaded = _load_tokenizer_aot_artifact(artifact, "fingerprint")
    assert set(loaded) == set(packages)
    assert {Path(path).read_bytes() for path in loaded.values()} == {b"package-zero", b"package-two"}


def test_tokenizer_artifact_is_create_only(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    packages = _packages(tmp_path)
    _publish_tokenizer_aot_artifact(artifact, "fingerprint", packages)
    with pytest.raises(RuntimeError, match="already exists"):
        _publish_tokenizer_aot_artifact(artifact, "fingerprint", packages)


def test_tokenizer_artifact_rejects_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _publish_tokenizer_aot_artifact(artifact, "fingerprint", _packages(tmp_path))
    package = next((artifact / "packages").iterdir())
    package.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _load_tokenizer_aot_artifact(artifact, "fingerprint")


def test_tokenizer_artifact_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _publish_tokenizer_aot_artifact(artifact, "fingerprint", _packages(tmp_path))
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        _load_tokenizer_aot_artifact(artifact, "different")


@pytest.mark.parametrize("version", [1, 2])
def test_tokenizer_artifact_rejects_previous_numerical_policies(tmp_path: Path, version: int) -> None:
    artifact = tmp_path / "artifact"
    _publish_tokenizer_aot_artifact(artifact, "fingerprint", _packages(tmp_path))
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = version
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="unsupported tokenizer AOT schema"):
        _load_tokenizer_aot_artifact(artifact, "fingerprint")


def test_rare_cache_rounding_requires_propagation_check() -> None:
    import torch

    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import _check_aot_tensor

    reference = torch.full((1_000_000,), 2.0, dtype=torch.bfloat16)
    actual = reference.clone()
    actual[0] = 1.7734375  # Historical 0.2265625 outlier; not a latent relaxation.
    accepted, followup, _ = _check_aot_tensor(actual, reference, allow_rare_cache_outliers=True)
    assert accepted and followup
    assert not _check_aot_tensor(actual, reference)[0]
    # Two outliers exceed the conservative sparsity budget.
    actual[1] = actual[0]
    assert not _check_aot_tensor(actual, reference, allow_rare_cache_outliers=True)[0]
    # Sparse absolute error can still dominate a near-zero reference tensor.
    reference.fill_(0.001)
    actual.copy_(reference)
    actual[0] += 0.2265625
    assert not _check_aot_tensor(actual, reference, allow_rare_cache_outliers=True)[0]


@pytest.mark.parametrize("kind", ["nan", "inf", "dense_drift", "large_outlier", "shape", "dtype", "missing"])
def test_aot_numerical_policy_rejects_corruption(kind: str) -> None:
    import torch

    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import _check_aot_tensor

    reference = torch.ones(1_000_000, dtype=torch.bfloat16)
    actual = reference.clone()
    if kind == "nan":
        actual[0] = float("nan")
    elif kind == "inf":
        actual[0] = float("inf")
    elif kind == "dense_drift":
        actual += 0.25
    elif kind == "large_outlier":
        actual[0] = 3
    elif kind == "shape":
        actual = actual[:10]
    elif kind == "dtype":
        actual = actual.float()
    elif kind == "missing":
        actual = None
    assert not _check_aot_tensor(actual, reference, allow_rare_cache_outliers=True)[0]


def test_tokenizer_artifact_rejects_unsafe_package_name(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _publish_tokenizer_aot_artifact(artifact, "fingerprint", _packages(tmp_path))
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    first_entry = next(iter(manifest["variants"].values()))
    first_entry["package"] = "../escape.pt2"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="unsafe"):
        _load_tokenizer_aot_artifact(artifact, "fingerprint", strict_hashes=False)


@pytest.mark.parametrize("sensitive", [False, True])
def test_cache_outlier_must_not_corrupt_next_latent(sensitive: bool) -> None:
    import torch

    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import _check_aot_cache_propagation

    class Encoder:
        def _encode_chunk_impl(self, x, cache, scale):
            assert x.shape[2] == 4 and x.is_contiguous()
            value = cache[0][0] if sensitive else cache[0].mean()
            return value.reshape(1), cache

    reference = torch.full((1_000_000,), 2.0)
    actual = reference.clone()
    actual[0] = 1.7734375
    rng_before = torch.get_rng_state().clone()
    accepted, _ = _check_aot_cache_propagation(Encoder(), torch.ones(1, 12, 1, 2, 2), [actual], [reference], None)
    assert accepted is not sensitive
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_fused_bfloat16_aot_accuracy_against_fp32(tmp_path: Path) -> None:
    import copy

    import torch

    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
        _TOKENIZER_AOT_INDUCTOR_CONFIGS,
        RMS_norm,
    )

    if not torch.cuda.is_available():
        pytest.skip("AOT numerical regression requires CUDA")
    model = torch.nn.Sequential(RMS_norm(160, images=False), torch.nn.SiLU()).cuda().bfloat16().eval()
    fp32_model = copy.deepcopy(model).float()
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(129)
        x = torch.rand(1, 160, 2, 60, 40, device="cuda", dtype=torch.bfloat16)
        package = tmp_path / "fused_bf16.pt2"
        torch._inductor.aoti_compile_and_package(
            torch.export.export(model, (x,), strict=False),
            package_path=str(package),
            inductor_configs=_TOKENIZER_AOT_INDUCTOR_CONFIGS.copy(),
        )
        runner = torch._inductor.aoti_load_package(str(package))
        required = set(runner.get_constant_fqns())
        runner.load_constants(
            {key: value for key, value in model.state_dict().items() if key in required},
            check_full_update=True,
            user_managed=True,
        )
        reference = fp32_model(x.float())
        actual, eager = runner(x).float(), model(x).float()
        torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
        # Eager BF16 is not the accuracy oracle: normal fusion can eliminate
        # unnecessary intermediate rounding and improve agreement with FP32.
        assert (actual - reference).square().mean() <= (eager - reference).square().mean()


def test_cache_propagation_rejects_delayed_latent_corruption() -> None:
    import torch

    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import _check_aot_cache_propagation

    class Encoder:
        def _encode_chunk_impl(self, x, cache, scale):
            # First latent agrees; the incoming discrepancy becomes visible
            # only when the next temporal-cache state is consumed.
            latent = torch.zeros(1) if cache[1].item() == 0 else cache[0][:1]
            return latent, [cache[0], torch.ones(1)]

    accepted, diagnostic = _check_aot_cache_propagation(
        Encoder(),
        torch.ones(1, 12, 4, 2, 2),
        [torch.tensor([1.7734375]), torch.zeros(1)],
        [torch.tensor([2.0]), torch.zeros(1)],
        None,
    )
    assert not accepted
    assert diagnostic.startswith("followup 2:")


def test_action_recipe_covers_every_view_layout_canvas() -> None:
    from types import SimpleNamespace

    import tomllib

    from cosmos_framework.configs.toml_config.sft_config import CompileTokenizerCallback
    from cosmos_framework.data.generator.action.unified_action128 import view_layout
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import _collect_warmup_shapes

    recipe = tomllib.loads(Path("examples/toml/sft_config/action128_wam_nano_160gpu.toml").read_text())
    config = CompileTokenizerCallback.model_validate(recipe["trainer"]["callbacks"]["compile_tokenizer"])
    assert config.enabled and config.require_full_coverage
    expected = set()
    for w, h in [(16, 9), (3, 2), (4, 3)]:
        for count in (1, 2, 3):
            roles = ["main", "left_wrist", "right_wrist"][:count]
            episode = {
                "view_mapping": {role: role for role in roles},
                "video_views": [{"role": role, "source_key": role, "width": w, "height": h} for role in roles],
            }
            _, _, width, height = view_layout(episode)
            expected.add((height, width))
    assert set(config.spatial_sizes) == expected
    tokenizer = SimpleNamespace(encode_chunk_frames={"256": 68, "480": 24, "720": 8})
    shapes = _collect_warmup_shapes(tokenizer, [], spatial_sizes=config.spatial_sizes)
    assert set(shapes) == {(24, h // 2, w // 2) for h, w in expected}


@pytest.mark.parametrize("size", [[0, 640], [481, 640], [480], [480, True]])
def test_explicit_warmup_rejects_invalid_pixel_dimensions(size) -> None:
    from types import SimpleNamespace

    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import _collect_warmup_shapes

    with pytest.raises(ValueError, match="positive pixel"):
        _collect_warmup_shapes(SimpleNamespace(encode_chunk_frames=24), [], spatial_sizes=[size])


def test_strict_runtime_rejects_missing_aot_chunk() -> None:
    import torch

    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import WanVAE_

    model = WanVAE_.__new__(WanVAE_)
    torch.nn.Module.__init__(model)
    model.temporal_window = 24
    model._encode_exact_durations = set()
    model._new_enc_cache = lambda: [None]
    model._aot_chunk_fns = {}
    model._require_aot_coverage = True
    with pytest.raises(RuntimeError, match="missing runtime chunk"):
        model.encode(torch.zeros(1, 3, 1, 16, 16), (torch.zeros(48), torch.ones(48)))
