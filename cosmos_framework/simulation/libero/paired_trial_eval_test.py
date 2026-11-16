from types import SimpleNamespace

import numpy as np

from cosmos_framework.simulation.libero.paired_trial_eval import RecordingClient, RecordingEnv


def test_recording_preserves_environment_step_and_copies_state():
    class Env:
        def __init__(self):
            self.sim = SimpleNamespace(data=SimpleNamespace(qpos=np.array([1.0, 2.0])))
            self.obs = {"robot0_eef_pos": np.array([1.0, 2.0, 3.0]), "agentview_image": np.zeros((2, 2, 3))}

        def set_init_state(self, state):
            return self.obs

        def step(self, action):
            self.sim.data.qpos += action
            return self.obs, 1.0, False, {"success": False}

    raw = Env()
    env = RecordingEnv(raw)
    assert env.set_init_state(None) is raw.obs
    action = np.array([0.1, 0.2])
    obs, reward, done, info = env.step(action)
    action[:] = 99
    raw.obs["robot0_eef_pos"][:] = 9
    assert obs is raw.obs and reward == 1 and not done and not info["success"]
    assert env.actions == [[0.1, 0.2]]
    assert env.trace[0]["qpos"] == [1.0, 2.0]
    assert env.trace[1]["robot0_eef_pos"] == [1.0, 2.0, 3.0]
    assert "agentview_image" not in env.trace[1]
    assert env.trace[1]["step"] == 1
    env.set_init_state(None)
    assert len(env.trace) == 1 and not env.actions


def test_action_only_keeps_views_and_records_action(monkeypatch):
    from cosmos_framework.simulation.libero import closed_loop_eval

    client = RecordingClient("http://localhost:1234", "libero", "task", 2, 10)
    client.action_only = True
    client.requests = []
    views = [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)]
    actions = [[0.1] * 10] * 16
    seen = []

    def predict_batch(self, observations):
        seen.extend(observations)
        return [actions]

    def unexpected_video_request(*args):
        raise AssertionError("Action-only evaluation must not request decoded prediction video")

    monkeypatch.setattr(closed_loop_eval.ActionEnvironmentClient, "predict_batch", predict_batch)
    monkeypatch.setattr(closed_loop_eval.ActionEnvironmentClient, "predict", unexpected_video_request)
    assert client.predict(views) == {"action": actions}
    assert len(seen) == 1 and seen[0] is views
    assert client.requests[0]["action"] == actions
    assert len(client.requests[0]["observation_sha256"]) == 64
