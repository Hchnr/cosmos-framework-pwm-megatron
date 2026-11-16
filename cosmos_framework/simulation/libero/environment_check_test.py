"""Reject the observed MuJoCo / NumPy ABI failure before collecting rollouts."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cosmos_framework.simulation.libero.closed_loop_eval import _check_mujoco_array_layout


@pytest.mark.parametrize("broken", [False, True])
def test_mujoco_array_layout(monkeypatch, broken):
    groups = np.ones(6, dtype=np.uint8)
    if broken:
        groups = np.broadcast_to(groups[:1], (6,))
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(MjvOption=lambda: SimpleNamespace(geomgroup=groups)))
    if broken:
        with pytest.raises(RuntimeError, match="uv --no-config"):
            _check_mujoco_array_layout()
    else:
        _check_mujoco_array_layout()
