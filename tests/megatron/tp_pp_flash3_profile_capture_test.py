# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

from tools.profile_tp_fsdp_flash3_ac import install_native_capture


def test_native_capture_starts_at_first_slot_and_stops_after_target_update():
    calls = []

    class Capture:
        def start(self):
            calls.append("start")

        def stop(self, *, failed=False):
            calls.append(("stop", failed))

    class Fixture:
        def prepare(self, loader, slot, *args, **kwargs):
            calls.append(("prepare", slot.update, slot.micro_step))

    class Trainer:
        def training_step(self, *, iteration, grad_accum_iter):
            calls.append(("step", iteration, grad_accum_iter))
            return None, None, (grad_accum_iter + 1) % 3

    config = SimpleNamespace(trainer=SimpleNamespace(type=Trainer))
    benchmark = SimpleNamespace(PreparedMicrosteps=Fixture)
    native = SimpleNamespace(load_experiment_from_toml=lambda: config)
    install_native_capture(benchmark, native, Capture(), 1202)
    native.load_experiment_from_toml()
    fixture, trainer = Fixture(), Trainer()
    for iteration in (1201, 1202):
        for micro in range(3):
            fixture.prepare(None, SimpleNamespace(update=iteration, micro_step=micro))
            trainer.training_step(iteration=iteration, grad_accum_iter=micro)
    assert calls.count("start") == 1
    assert calls.count(("stop", False)) == 1
    assert calls.index("start") < calls.index(("prepare", 1202, 0))
    assert calls[-1] == ("stop", False)
