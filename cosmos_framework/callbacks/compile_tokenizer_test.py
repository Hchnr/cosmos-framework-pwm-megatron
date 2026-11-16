import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.callbacks.compile_tokenizer import CompileTokenizer


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str, dict]] = []

    def compile_encode(self, resolutions, output_dir: str, **kwargs) -> None:
        self.calls.append((list(resolutions), output_dir, kwargs))


def _callback(after: int = 3) -> tuple[CompileTokenizer, _FakeTokenizer, SimpleNamespace]:
    callback = CompileTokenizer(enabled=True, compile_after_iterations=after, warmup_resolutions=[256])
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local="/tmp/test-job"))
    tokenizer = _FakeTokenizer()
    model = SimpleNamespace(tokenizer_vision_gen=tokenizer)
    return callback, tokenizer, model


def test_compile_trigger_uses_optimizer_iteration_not_microbatch_count() -> None:
    callback, tokenizer, model = _callback(after=3)

    # Gradient accumulation and CP can produce multiple hook calls at the same
    # optimizer iteration.  They must not advance a rank-local trigger counter.
    for iteration in (0, 0, 1, 1, 2, 2):
        callback.on_training_step_start(model, {}, iteration=iteration)
    assert tokenizer.calls == []

    callback.on_training_step_start(model, {}, iteration=3)
    callback.on_training_step_start(model, {}, iteration=3)
    callback.on_training_step_start(model, {}, iteration=4)
    assert tokenizer.calls == [
        (
            ["256"],
            "/tmp/test-job",
            {
                "aspect_ratio": None,
                "backend": "inductor",
                "mode": "reduce-overhead",
                "fullgraph": False,
                "dynamic": False,
            },
        )
    ]


def test_resume_past_trigger_compiles_once_on_first_hook() -> None:
    callback, tokenizer, model = _callback(after=3)
    callback.on_training_step_start(model, {}, iteration=5)
    callback.on_training_step_start(model, {}, iteration=5)
    assert len(tokenizer.calls) == 1


def test_negative_compile_iteration_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CompileTokenizer(enabled=True, compile_after_iterations=-1, warmup_resolutions=["256"])


def test_aspect_ratio_is_forwarded_to_tokenizer_compile() -> None:
    callback = CompileTokenizer(
        enabled=True,
        compile_after_iterations=0,
        warmup_resolutions=["256"],
        aspect_ratio="1,1",
    )
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local="/tmp/test-job"))
    tokenizer = _FakeTokenizer()
    model = SimpleNamespace(tokenizer_vision_gen=tokenizer)

    callback.on_training_step_start(model, {}, iteration=0)

    assert tokenizer.calls[0][2]["aspect_ratio"] == "1,1"


def test_missing_tokenizer_disables_callback() -> None:
    callback = CompileTokenizer(enabled=True, compile_after_iterations=0, warmup_resolutions=["256"])
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local="/tmp/test-job"))
    model = SimpleNamespace(tokenizer_vision_gen=None)

    callback.on_training_step_start(model, {}, iteration=0)

    assert callback.enabled is False
    assert callback._has_compiled is False


def test_explicit_canvases_without_resolution_buckets() -> None:
    callback = CompileTokenizer(
        enabled=True, compile_after_iterations=0, spatial_sizes=[[480, 640]], require_full_coverage=True
    )
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local="/tmp/test-job"))
    tokenizer = _FakeTokenizer()
    callback.on_training_step_start(SimpleNamespace(tokenizer_vision_gen=tokenizer), {}, iteration=0)
    resolutions, _, kwargs = tokenizer.calls[0]
    assert resolutions == []
    assert kwargs["spatial_sizes"] == [[480, 640]]
    assert kwargs["require_full_coverage"] is True


@pytest.mark.parametrize("sizes,aspect", [([], None), ([[480, 640]], "1,1")])
def test_explicit_canvases_reject_empty_or_conflicting_filter(sizes, aspect) -> None:
    with pytest.raises(ValueError, match="spatial_sizes"):
        CompileTokenizer(enabled=True, spatial_sizes=sizes, aspect_ratio=aspect)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpus(1))])
@pytest.mark.parametrize("fails", [False, True])
def test_compile_preserves_restored_training_rng_even_on_failure(monkeypatch, device, fails):
    callback, tokenizer, model = _callback()

    def consuming_compile(*args, **kwargs):
        random.random()
        np.random.standard_normal(11)
        torch.randn(13)
        if device == "cuda":
            torch.randn(17, device=device)
        if fails:
            raise RuntimeError("injected AOT failure")

    monkeypatch.setattr(tokenizer, "compile_encode", consuming_compile)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    if fails:
        with pytest.raises(RuntimeError, match="injected AOT failure"):
            callback.on_training_step_start(model, {}, iteration=11200)
    else:
        callback.on_training_step_start(model, {}, iteration=11200)
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    if cuda_rng is not None:
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    assert random.getstate() == python_rng
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_rng[0] and restored_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(restored_numpy[1], numpy_rng[1])
    assert callback._has_compiled
