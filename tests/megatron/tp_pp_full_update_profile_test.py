"""Full-update profiling must include preparation and all accumulated backwards."""

import sys
from types import SimpleNamespace

import pytest

from tools.run_tp_pp_full_update_profile import UpdateCapture, install_native, install_tp, start_before, stop_after


def capture(events):
    cuda = SimpleNamespace(
        cudart=lambda: SimpleNamespace(
            cudaProfilerStart=lambda: events.append("start"), cudaProfilerStop=lambda: events.append("stop")
        ),
        synchronize=lambda: events.append("synchronize"),
    )
    return UpdateCapture(cuda, barrier=lambda: events.append("rank_barrier"))


def test_native_capture_spans_preparation_all_microbatches_and_optimizer():
    events = []
    state = capture(events)

    class Trainer:
        def training_step(self, index):
            events.append(f"backward{index}")
            if index == 143:
                events.append("optimizer")
            return "output", "loss", (index + 1) % 144

    module = SimpleNamespace(
        prepare_slot=lambda index: events.append(f"prepare{index}"),
        load_experiment_from_toml=lambda: SimpleNamespace(trainer=SimpleNamespace(type=Trainer)),
    )
    install_native(module, state)
    trainer = module.load_experiment_from_toml().trainer.type()
    for index in range(144):
        module.prepare_slot(index)
        assert trainer.training_step(index) == ("output", "loss", (index + 1) % 144)
        assert state.finished == (index == 143)
    assert events[:4] == ["synchronize", "rank_barrier", "start", "prepare0"]
    assert events[-5:] == ["backward143", "optimizer", "synchronize", "rank_barrier", "stop"]
    assert events.count("start") == events.count("stop") == 1
    assert events.count("rank_barrier") == 2


def test_tp_capture_includes_parallel_preparation_and_closes_after_optimizer(monkeypatch):
    events = []
    state = capture(events)

    class Preparation:
        def prepare(self):
            events.extend(["vae", "spool_write"])
            return "prepared"

    class Driver:
        def forward_backward(self):
            events.extend(["forward", "backward", "gc"])
            return "metrics"

        def step(self):
            events.append("optimizer")
            return True, 0.1, None

    monkeypatch.setitem(
        sys.modules,
        "cosmos_framework.training_backends.megatron_tp_pp.parallel_preparation",
        SimpleNamespace(ParallelPreparation=Preparation),
    )
    monkeypatch.setitem(
        sys.modules,
        "cosmos_framework.training_backends.megatron_tp_pp.driver",
        SimpleNamespace(ModelParallelDriver=Driver),
    )
    install_tp(state)
    assert Preparation().prepare() == "prepared"
    driver = Driver()
    assert driver.forward_backward() == "metrics" and not state.finished
    assert driver.step() == (True, 0.1, None)
    assert events == [
        "synchronize",
        "rank_barrier",
        "start",
        "vae",
        "spool_write",
        "forward",
        "backward",
        "gc",
        "optimizer",
        "synchronize",
        "rank_barrier",
        "stop",
    ]


@pytest.mark.parametrize("failing_phase", ["prepare", "optimizer"])
def test_failed_phase_flushes_capture_and_propagates_exception(failing_phase):
    events = []
    state = capture(events)

    def fail():
        events.append(failing_phase)
        raise RuntimeError("intentional failure")

    if failing_phase == "prepare":
        function = start_before(fail, state)
    else:
        state.start()
        function = stop_after(fail, state)
    with pytest.raises(RuntimeError, match="intentional failure"):
        function()
    assert events == ["synchronize", "rank_barrier", "start", failing_phase, "synchronize", "stop"]
