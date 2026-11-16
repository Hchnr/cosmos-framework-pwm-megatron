"""HTTP smoke benchmark against already running Action128 servers.

No subprocesses are started. Requests must describe distinct real observations,
with identical camera layout/action shape. Timings include the complete HTTP path.
"""

import argparse
import concurrent.futures
import copy
import hashlib
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(
    description="Validate two independent loopback Action128 HTTP replicas using real WAM request JSONs."
)
parser.add_argument("--requests", type=Path, nargs=2, required=True)
parser.add_argument("--ports", type=int, nargs=2, default=[18104, 18105])
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--ready-timeout", type=float, default=300)
args = parser.parse_args()
ROOT = args.output_dir
ROOT.mkdir(parents=True, exist_ok=True)
OUT = args.summary
OUT.parent.mkdir(parents=True, exist_ok=True)
requests = [json.loads(path.read_text()) for path in args.requests]
summary = {
    "command": sys.argv,
    "request_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.requests],
    "num_steps": requests[0]["num_steps"],
    "guidance": requests[0]["guidance"],
    "records": [],
}

summary_lock = threading.Lock()


def read(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/{path}", timeout=5) as r:
        return json.load(r)


def run(port, request, label):
    start = time.perf_counter()
    body = json.dumps(request).encode()
    with urllib.request.urlopen(
        urllib.request.Request(
            f"http://127.0.0.1:{port}/generate", data=body, headers={"Content-Type": "application/json"}
        ),
        timeout=600,
    ) as r:
        response = json.load(r)
    elapsed = time.perf_counter() - start
    (ROOT / f"{label}.json").write_text(json.dumps(response))
    output = response["outputs"][0]
    content = output["content"]
    meta = content["action128"]
    actions = np.array(content["action"])
    normal = np.array(meta["normalized_action"])
    mask = np.array(request["extra"]["action128"]["action_dim_mask"])
    assert actions.shape == (request["action_chunk_size"], 128) and normal.shape == actions.shape
    assert np.isfinite(actions).all() and np.isfinite(normal).all()
    assert (actions[:, ~mask] == 0).all() and (normal[:, ~mask] == 0).all()
    assert meta["action_domain"] == request["extra"]["action128"]["action_domain"]
    assert meta["seed"] == request["seed"] and meta["action_dim_mask"] == mask.tolist()
    assert np.allclose(
        meta["action_timestamps_seconds"],
        np.arange(request["action_chunk_size"]) / request["extra"]["action128"]["action_fps"],
    )
    return_video = request["extra"]["action128"].get("return_video", True)
    assert bool(output["files"]) == return_video
    for file in output["files"]:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/outputs/{file}", timeout=10) as r:
            assert len(r.read()) > 1000
    record = {
        "label": label,
        "port": port,
        "client_seconds": elapsed,
        "model_elapsed_seconds": meta["elapsed_seconds"],
        "shape": list(actions.shape),
        "finite": True,
        "inactive_slots_zero": True,
        "domain_mask_timestamps_valid": True,
        "files": output["files"],
        "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
    }
    with summary_lock:
        summary["records"].append(record)
        OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(record), flush=True)
    return actions


for port in args.ports:
    ready_deadline = time.monotonic() + args.ready_timeout
    while True:
        try:
            read(port, "healthz")
            break
        except (OSError, ValueError):
            if time.monotonic() > ready_deadline:
                raise TimeoutError(f"HTTP replica {port} did not become ready") from None
            time.sleep(2)
    summary[f"info_{port}"] = read(port, "info")
    summary[f"domains_{port}"] = read(port, "domains")
# Cold/compilation requests happen simultaneously on isolated GPUs.
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    futures = [pool.submit(run, p, requests[0], f"initial_{p}") for p in args.ports]
    [f.result() for f in futures]
a = run(args.ports[0], requests[0], "warm_A")
b = run(args.ports[0], requests[1], "warm_B")
again = run(args.ports[0], requests[0], "replay_A")
summary["isolation"] = {
    "A_replay_equal": bool(np.array_equal(a, again)),
    "A_B_different": bool(not np.array_equal(a, b)),
    "max_A_replay_abs": float(abs(a - again).max()),
}
assert summary["isolation"]["A_replay_equal"] and summary["isolation"]["A_B_different"]
# Each replica must replay its own request after another observation/seed.
a_other = run(args.ports[1], requests[0], "other_A")
b_other = run(args.ports[1], requests[1], "other_B")
again_other = run(args.ports[1], requests[0], "other_replay_A")
summary["other_isolation"] = {
    "A_replay_equal": bool(np.array_equal(a_other, again_other)),
    "A_B_different": bool(not np.array_equal(a_other, b_other)),
}
assert all(summary["other_isolation"].values())
summary["cross_replica_physical_action"] = {
    "A_max_abs": float(abs(a - a_other).max()),
    "A_mean_abs": float(abs(a - a_other).mean()),
    "B_max_abs": float(abs(b - b_other).max()),
    "B_mean_abs": float(abs(b - b_other).mean()),
    "note": "Cross-replica differences are measured separately; no closed-loop equivalence claim.",
}
summary["cross_replica_normalized_active_slots"] = {
    "thresholds": {"max_abs": 0.05, "rmse": 0.01},
    "cases": {},
}
for left, right in (("warm_A", "other_A"), ("warm_B", "other_B")):
    left_meta = json.loads((ROOT / f"{left}.json").read_text())["outputs"][0]["content"]["action128"]
    right_meta = json.loads((ROOT / f"{right}.json").read_text())["outputs"][0]["content"]["action128"]
    mask = np.array(left_meta["action_dim_mask"])
    diff = np.array(left_meta["normalized_action"])[:, mask] - np.array(right_meta["normalized_action"])[:, mask]
    max_abs = float(abs(diff).max())
    rmse = float(np.sqrt((diff * diff).mean()))
    summary["cross_replica_normalized_active_slots"]["cases"][left] = {
        "max_abs": max_abs,
        "rmse": rmse,
        "passed": max_abs <= 0.05 and rmse <= 0.01,
    }
summary["cross_replica_numerical_equivalence_passed"] = all(
    case["passed"] for case in summary["cross_replica_normalized_active_slots"]["cases"].values()
)
for i in range(3):
    run(args.ports[0], requests[0], f"steady_action_{i}")
video = copy.deepcopy(requests[0])
video["extra"]["action128"].pop("return_video")
run(args.ports[0], video, "default_video")
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    futures = [pool.submit(run, p, requests[i], f"parallel_{p}") for i, p in enumerate(args.ports)]
    [f.result() for f in futures]
summary["parallel_wall_seconds"] = time.perf_counter() - start
summary["http_contract_passed"] = True
summary["limitations"] = [
    "Short HTTP smoke benchmark, no closed-loop quality or p99/SLA validation.",
    "Health readiness only means weights loaded; first generation performs compilation/warmup.",
    "HTTP includes request preparation, serialization, local filesystem and optional video decode/save; offline sampling timings use different boundaries.",
]
OUT.write_text(json.dumps(summary, indent=2))
