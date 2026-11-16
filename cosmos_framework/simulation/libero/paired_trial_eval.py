"""Run official LIBERO states or matched dataset Episode starts with paired evidence.

Simulator state is saved for evaluation only; policy requests contain images and
task text through the existing ActionEnvironmentClient.
"""

import argparse
import hashlib
import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cosmos_framework.simulation.libero import closed_loop_eval as rollout


class RecordingEnv:
    def __init__(self, env, model_xml=None):
        self.env = env
        self.model_xml = model_xml
        self.trace = []
        self.actions = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def record(self, obs):
        row = {"step": len(self.actions), "qpos": np.asarray(self.env.sim.data.qpos).tolist()}
        for key, value in obs.items():
            if key.endswith(("_pos", "_quat", "_qpos")) or key == "object-state":
                row[key] = np.asarray(value).tolist()
        self.trace.append(row)

    def reset(self):
        obs = self.env.reset()
        if self.model_xml is not None:
            from libero.libero.utils.utils import postprocess_model_xml

            xml = ET.fromstring(postprocess_model_xml(self.model_xml, {}))
            for node in xml.iter():
                path = node.get("file")
                if path and "/assets/" in path and not Path(path).is_file():
                    local = Path(rollout.get_libero_path("assets")) / path.split("/assets/", 1)[1]
                    if not local.is_file():
                        raise FileNotFoundError(local)
                    node.set("file", str(local))
            self.env.reset_from_xml_string(ET.tostring(xml, encoding="unicode"))
        return obs

    def set_init_state(self, state):
        obs = self.env.set_init_state(state)
        self.trace, self.actions = [], []
        self.record(obs)
        return obs

    def step(self, action):
        result = self.env.step(action)
        self.actions.append(np.asarray(action).tolist())
        self.record(result[0])
        return result


class RecordingClient(rollout.ActionEnvironmentClient):
    action_only = False

    def predict(self, observation):
        image = (
            self.concatenate_images(observation) if isinstance(observation, list) else self.resize_image(observation)
        )
        if self.action_only:
            views = observation if isinstance(observation, list) else [observation]
            result = {"action": self.predict_batch([views])[0]}
        else:
            result = super().predict(observation)
        self.requests.append(
            {
                "observation_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                "action": result.get("action"),
            }
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--trials", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--initial-states", type=Path, help="Frozen dataset Episode states and model XML, indexed by trial"
    )
    parser.add_argument(
        "--action-only", action="store_true", help="Use batch-one action API; skip unused VAE video decoding"
    )
    args = parser.parse_args()
    if not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("Require 0 <= warmup_steps < max_steps")
    trials = [int(value) for value in args.trials.split(",")]
    if len(set(trials)) != len(trials) or any(t < 0 for t in trials):
        raise ValueError("Trial IDs must be distinct and nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    rollout._configure_mujoco_env("osmesa")
    rollout._import_libero()
    suite = rollout.benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(8)
    initial_states = suite.get_task_init_states(8)
    frozen = json.loads(args.initial_states.read_text()) if args.initial_states else None
    if frozen:
        initial_states = {r["trial"]: r["initial_state"] for r in frozen["episodes"]}
        if not set(trials).issubset(initial_states):
            raise ValueError("Trial absent from frozen Episode manifest")
    if not frozen and max(trials) >= len(initial_states):
        raise ValueError("Requested initial state does not exist")
    client = RecordingClient(args.server_url, "libero", str(task.language), image_size=256, timeout=180)
    client.action_only = args.action_only
    rollout._wait_for_server(client, 600)
    info = client.get_info()
    protocol = dict(
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        action_only=args.action_only,
        initial_states_sha256=hashlib.sha256(args.initial_states.read_bytes()).hexdigest() if frozen else None,
    )
    results = []
    for trial in trials:
        path = args.output / f"trial_{trial:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["server_info"] != info or saved.get("protocol") != protocol:
                raise ValueError("Existing trial used different server configuration")
            results.append(saved)
            continue
        random.seed(args.seed + trial)
        np.random.seed(args.seed + trial)
        env, _ = rollout._get_libero_env(task, resolution=256, seed=args.seed + trial, render_gpu_device_id=-1)
        episode = next(r for r in frozen["episodes"] if r["trial"] == trial) if frozen else None
        env = RecordingEnv(env, episode["model_xml"] if episode else None)
        initial_state = np.asarray(initial_states[trial])
        client.requests = []
        started = time.perf_counter()
        try:
            result = rollout._run_episode(
                env,
                client,
                cameras=["agentview", "wrist"],
                flip_images=False,
                rotate_180=True,
                action_horizon=8,
                action_dim=10,
                action_space="frame_wise_relative",
                rotation_space="6d",
                gripper_mode="zero_one",
                max_steps=args.max_steps,
                warmup_steps=args.warmup_steps,
                initial_state=initial_state,
                gif_path=args.output / f"trial_{trial:03d}.gif",
                gif_fps=20,
            )
            success, error = result.success, result.error
        except Exception as exc:
            success, error = False, repr(exc)
        finally:
            env.close()
        row = dict(
            trial=trial,
            seed=args.seed + trial,
            initial_state_sha256=hashlib.sha256(initial_state.tobytes()).hexdigest(),
            server_info=info,
            protocol=protocol,
            dataset_episode=episode["episode"] if episode else None,
            success=success,
            error=error,
            steps=len(env.actions),
            elapsed_s=time.perf_counter() - started,
            actions=env.actions,
            observations=env.trace,
            requests=client.requests,
        )
        path.write_text(json.dumps(row) + "\n")
        results.append(row)
        summary = {
            "trial_ids": trials,
            "completed": len(results),
            "successes": sum(r["success"] for r in results),
            "errors": sum(r["error"] is not None for r in results),
            "results": [{k: r[k] for k in ("trial", "success", "steps", "error", "elapsed_s")} for r in results],
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary["results"][-1]), flush=True)
        if error:
            raise RuntimeError(f"Trial {trial} failed: {error}")
        client.notify_next_episode()


if __name__ == "__main__":
    main()
