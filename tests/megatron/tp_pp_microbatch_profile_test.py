# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest

from tools.profile_tp4_microbatch import install_capture


@pytest.mark.parametrize("failure", [None, "forward", "optimizer"])
def test_capture_excludes_warmup_and_covers_preparation_through_optimizer(failure):
    events = []
    capture = SimpleNamespace(started=False, finished=False)

    def start():
        capture.started = True
        events.append("start")

    def stop(*, failed=False):
        capture.finished = True
        events.append(("stop", failed))

    capture.start, capture.stop = start, stop

    class Driver:
        iteration = 1200

        def forward_backward(self, batches):
            for value in batches:
                events.append(("compute", self.iteration, value))
            if self.iteration == 1202 and failure == "forward":
                raise RuntimeError("forward failed")

        def step(self):
            events.append(("optimizer", self.iteration))
            if self.iteration == 1202 and failure == "optimizer":
                raise RuntimeError("optimizer failed")
            self.iteration += 1
            return True, 1.0, None

    install_capture(capture, 1202, Driver)
    driver = Driver()

    def batches():
        for index in range(3):
            events.append(("prepare", driver.iteration, index))
            yield index

    for _ in range(2):
        driver.forward_backward(batches())
        driver.step()
    assert not capture.started
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            driver.forward_backward(batches())
            driver.step()
    else:
        driver.forward_backward(batches())
        driver.step()
    assert capture.finished
    assert events.index("start") < events.index(("prepare", 1202, 0))
    assert events.index(("compute", 1202, 2)) < events.index(("stop", bool(failure)))
    if failure != "forward":
        assert events.index(("optimizer", 1202)) < events.index(("stop", bool(failure)))
