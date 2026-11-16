"""Vector runner geometry and chunk lengths, without starting a simulator."""

import copy
import multiprocessing
import signal
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cosmos_framework.simulation.libero import closed_loop_eval as evaluator


@pytest.fixture
def vector_runner(monkeypatch):
    class Vec:
        actions = []

        def __init__(self, fns):
            self.obs = [
                {
                    "agentview_image": np.full((8, 8, 3), slot, dtype=np.uint8),
                    "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
                    "robot0_eef_quat": Rotation.from_euler("z", angle, degrees=True).as_quat(),
                }
                for slot, angle in enumerate((90, 180)[: len(fns)])
            ]

        def seed(self, seed):
            pass

        def reset(self, id):
            pass

        def set_init_state(self, states, id):
            return [copy.deepcopy(self.obs[s]) for s in id]

        def step(self, action, id):
            self.actions.append((list(id), np.asarray(action).copy()))
            for s, value in zip(id, action, strict=True):
                self.obs[s]["robot0_eef_pos"] += value[:3]
            return [copy.deepcopy(self.obs[s]) for s in id], [0.0] * len(id), [False] * len(id), [{} for _ in id]

        def close(self):
            pass

    module = ModuleType("libero.libero.envs.venv")
    module.SubprocVectorEnv = Vec
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(evaluator, "get_libero_path", lambda key: "/tmp", raising=False)
    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)

    def run(lengths, *, horizon=2, action_space="framewise", max_steps=2):
        class Client:
            def predict_batch(self, observations):
                return [[[0.1, 0, 0, 0, 0, 0, 1]] * lengths[int(np.asarray(obs[0])[0, 0, 0])] for obs in observations]

        kwargs = dict(
            num_trials=2,
            num_envs=2,
            env_image_size=8,
            seed=42,
            render_gpu_device_id=0,
            client=Client(),
            cameras=["agentview"],
            flip_images=False,
            rotate_180=False,
            action_horizon=horizon,
            action_dim=7,
            rotation_space="3d",
            gripper_mode="pm_one",
            max_steps=max_steps,
            warmup_steps=0,
            init_states=[np.zeros(1), np.zeros(1)],
        )
        if action_space == "relative":
            kwargs["action_space"] = action_space

        def expired(signum, frame):
            raise TimeoutError("Vectorized evaluation made no progress on an empty chunk")

        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, 2)
        try:
            results = evaluator._run_task_vectorized(
                SimpleNamespace(problem_folder="test", bddl_file="test.bddl"), "pick", **kwargs
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        return results, Vec.actions

    return run


@pytest.mark.parametrize("space", ["framewise", "relative"])
def test_vectorized_conversion_uses_each_env_anchor_and_current_pose(vector_runner, space):
    results, actions = vector_runner([2, 2], action_space=space)
    assert len(actions) == 2
    expected = np.array([[0.1, 0, 0], [0.1, 0, 0]])
    if space == "relative":
        expected = np.array([[0, 0.1, 0], [-0.1, 0, 0]])
    np.testing.assert_allclose(actions[0][1][:, :3], expected, atol=1e-6)
    np.testing.assert_allclose(actions[1][1][:, :3], 0 if space == "relative" else expected, atol=1e-6)
    assert [item["steps"] for item in results] == [2, 2]


@pytest.mark.parametrize("horizon", [0, 2, 5])
@pytest.mark.parametrize("lengths", [[1, 1], [1, 2], [0, 1], [0, 0]])
def test_vectorized_horizon_handles_short_uneven_and_empty_chunks(vector_runner, horizon, lengths):
    results, actions = vector_runner(lengths, horizon=horizon)
    assert len(results) == 2
    for slot, length in enumerate(lengths):
        if length == 0:
            assert results[slot]["error"] is not None
            assert "empty" in results[slot]["error"].lower()
        else:
            assert results[slot]["error"] is None
    assert len(actions) <= 2
