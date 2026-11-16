"""Run real serving acceptance on free GPUs with a caller-provided observation JSON.

Example: python tools/validate_action128_serving.py --checkpoint-root <training-output>
  --request-file <request.json> --gpu-counts 1 2 4 8 --output-dir <artifacts>
This starts and stops only its own launcher process groups. It does not free GPUs.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
import yaml


def check_response(response, request, config):
    assert response["status"] == "success", response.get("message")
    content = response["outputs"][0]["content"]
    response = {"action": content["action"], **content["action128"]}
    extension = request["extra"]["action128"]
    action_fps = extension.get("action_fps", request["fps"])
    action = np.asarray(response["action"], dtype=np.float64)
    normalized = np.asarray(response["normalized_action"], dtype=np.float64)
    expected_steps = request["action_chunk_size"]
    assert action.shape == normalized.shape == (expected_steps, 128)
    assert np.isfinite(action).all() and np.isfinite(normalized).all()
    active = np.asarray(extension["action_dim_mask"], dtype=bool)
    assert not action[:, ~active].any() and not normalized[:, ~active].any()
    spec = config["model"]["config"]["action128_spec"]
    source = spec["registry"]["domains"].index(extension["action_domain"])
    transform = spec["transforms"][source]
    mask = np.asarray(transform["normalization_mask"], dtype=bool)
    physical = normalized.copy()
    physical[:, mask] = (physical[:, mask] - np.asarray(transform["bias"])[mask]) / np.asarray(transform["scale"])[mask]
    physical[:, ~active] = 0
    np.testing.assert_allclose(action, physical, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(response["action_timestamps_seconds"], np.arange(expected_steps) / action_fps)
    assert response["action_units"] == "raw_physical_or_source_units"
    return {
        "shape": list(action.shape),
        "finite": True,
        "inactive_slots_zero": True,
        "inverse_transform_verified": True,
        "elapsed_seconds": response["elapsed_seconds"],
        "active_action_min": float(action[:, active].min()),
        "active_action_max": float(action[:, active].max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", "--checkpoint-root", dest="checkpoint_root", required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--gpu-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--original-client", action="store_true", help="Use the unmodified Cosmos ray.submit client")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    repo = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = json.loads(args.request_file.read_text())
    manifest = {"request_file": str(args.request_file.resolve()), "deployments": []}
    manifest_path = args.output_dir / "results.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for count in args.gpu_counts:
        if count < 1:
            raise ValueError("GPU counts must be positive")
        dest = args.output_dir / f"{count}gpu"
        dest.mkdir(parents=True, exist_ok=True)
        row = {"gpu_count": count, "status": "starting", "responses": [], "sampled_peak_gpu_memory_mib": {}}
        manifest["deployments"].append(row)
        save()
        env = {
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
            "CHECKPOINT_PATH": args.checkpoint_root,
            "SERVING_GPUS": str(count),
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(count))),
            "SERVING_NNODES": "1",
            "SERVING_NODE_RANK": "0",
            "SERVING_PORT": str(args.port),
            "SERVING_MASTER_PORT": str(args.port + 21000),
            "SERVING_HOST": "127.0.0.1",
            "SERVING_OUTPUT_DIR": str(dest.resolve()),
        }
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                try:
                    output = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                        text=True,
                        timeout=5,
                    )
                    for line in output.splitlines():
                        index, memory = map(int, line.split(","))
                        if index < count:
                            peaks = row["sampled_peak_gpu_memory_mib"]
                            peaks[str(index)] = max(peaks.get(str(index), 0), memory)
                except (subprocess.SubprocessError, ValueError):
                    pass
                stop.wait(0.5)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        process = None
        try:
            print(f"Starting {count} GPU deployment", flush=True)
            start = time.monotonic()
            with (dest / "server.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        "bash",
                        "examples/launch_action128_serving.sh",
                        *(["--no-reuse-small-action-metadata"] if count > 1 else []),
                    ],
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            url = f"http://127.0.0.1:{args.port}"
            session = requests.Session()
            session.trust_env = False
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; see {dest}/server.log")
                try:
                    if session.get(f"{url}/healthz", timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() - start > args.startup_timeout:
                    raise TimeoutError("Model startup timed out")
                time.sleep(1)
            row["startup_seconds"] = time.monotonic() - start
            row["info"] = session.get(f"{url}/info", timeout=5).json()
            assert row["info"]["world_size"] == count
            config = yaml.safe_load(Path(row["info"]["config_file"]).read_text())
            row["status"] = "generating"
            save()
            print(f"{count} GPU ready in {row['startup_seconds']:.2f}s", flush=True)
            previous = None
            for number in range(args.repeats):
                if args.original_client:
                    from cosmos_framework.inference.ray.submit import Args as SubmitArgs
                    from cosmos_framework.inference.ray.submit import submit

                    client_dir = dest / f"original_client_{number}"
                    os.environ["NO_PROXY"] = ",".join(
                        filter(None, (os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"))
                    )
                    submit(
                        SubmitArgs(
                            output_dir=client_dir.resolve(),
                            input_files=[args.request_file.resolve()],
                            host="127.0.0.1",
                            port=args.port,
                            timeout=args.request_timeout,
                        )
                    )
                    data = json.loads((client_dir / request["name"] / "sample_outputs.json").read_text())
                else:
                    response = session.post(f"{url}/generate", json=request, timeout=args.request_timeout)
                    response.raise_for_status()
                    data = response.json()
                (dest / f"response_{number}.json").write_text(json.dumps(data) + "\n")
                summary = check_response(data, request, config)
                assert data["outputs"][0]["files"]
                video_path = data["outputs"][0]["files"][0]
                saved_video = dest / f"vision_{number}.mp4"
                if args.original_client:
                    saved_video.write_bytes(Path(video_path).read_bytes())
                    summary["original_submit_client"] = True
                else:
                    video = session.get(f"{url}/outputs/{video_path}", timeout=args.request_timeout)
                    video.raise_for_status()
                    saved_video.write_bytes(video.content)
                probe = json.loads(
                    subprocess.check_output(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=width,height,nb_frames,r_frame_rate",
                            "-of",
                            "json",
                            str(saved_video),
                        ],
                        text=True,
                    )
                )["streams"][0]
                assert int(probe["nb_frames"]) == data["args"]["num_frames"]
                summary["video"] = probe
                summary["video_bytes"] = saved_video.stat().st_size
                if previous is not None:
                    summary["repeat_max_abs_difference"] = float(
                        np.max(np.abs(np.asarray(data["outputs"][0]["content"]["action"]) - previous))
                    )
                    np.testing.assert_allclose(data["outputs"][0]["content"]["action"], previous, atol=1e-5, rtol=1e-5)
                previous = np.asarray(data["outputs"][0]["content"]["action"])
                row["responses"].append(summary)
                save()
                print(f"{count} GPU request {number}: {summary}", flush=True)
            assert session.get(f"{url}/healthz", timeout=5).status_code == 200
            row["status"] = "passed"
        except Exception as exc:
            row["status"], row["error"] = "failed", str(exc)
            print(f"{count} GPU failed: {exc}", flush=True)
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            stop.set()
            thread.join(10)
            save()
    if any(row["status"] != "passed" for row in manifest["deployments"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
