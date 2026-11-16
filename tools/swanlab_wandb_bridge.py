"""Read-only W&B snapshots -> SwanLab scalar history and supported media.

Runs separately from trainers. Credentials and proxy URLs are never logged.
"""

import argparse
import bisect
import fcntl
import json
import math
import os
import re
import signal
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEP_LOG = None
STOP = False


def stop(*_):
    global STOP
    STOP = True


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def snapshot(source, destination):
    remaining = source.stat().st_size
    with source.open("rb") as src, destination.open("wb") as dst:
        while remaining:
            data = src.read(min(remaining, 1024 * 1024))
            if not data:
                raise RuntimeError("source_shrank")
            dst.write(data)
            remaining -= len(data)


def extract(source):
    from swanlab.converter.wb.wbdatastore import DataStoreTruncatedError, WBDataStoreReader
    from wandb.proto import wandb_internal_pb2

    copy = ROOT / "latest_prefix.wandb"
    snapshot(source, copy)
    ds = WBDataStoreReader()
    ds.open(str(copy))
    histories = {}
    system_stats = []
    types = Counter()
    partial = False
    skipped = Counter()
    try:
        while True:
            try:
                raw = ds.scan()
            except DataStoreTruncatedError:
                partial = True
                break
            if raw is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            kind = record.WhichOneof("record_type")
            types[kind] += 1
            if kind == "stats":
                metrics = {}
                for item in record.stats.item:
                    value = json.loads(item.value_json)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        metrics["system/" + item.key] = value
                system_stats.append((record.stats.timestamp.ToNanoseconds() / 1e9, metrics))
            if kind != "history":
                continue
            values = {}
            for item in record.history.item:
                key = item.key or "/".join(item.nested_key)
                if key:
                    values[key] = json.loads(item.value_json)
            step = values.get("_step", values.get("iteration"))
            if not isinstance(step, (int, float)):
                skipped["history_without_step"] += 1
                continue
            step = int(step)
            metrics = {}
            media = {}
            for name, kind in list(values.items()):
                if name.endswith("/_type") and kind in ["table-file", "image-file", "audio-file"]:
                    base = name[:-6]
                    props = {k[len(base) + 1 :]: v for k, v in values.items() if k.startswith(base + "/")}
                    if "path" in props:
                        media[base] = props
                        for k in list(values):
                            if k.startswith(base + "/"):
                                values.pop(k)
            for key, value in values.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[key] = value
                elif isinstance(value, dict) and value.get("_type") in ["table-file", "image-file", "audio-file"]:
                    media[key] = value
                else:
                    skipped["unsupported_values"] += 1
            # W&B can have multiple records at one step. Merge disjoint keys.
            row = histories.setdefault(
                step, {"step": step, "scalars": {}, "media": {}, "source_timestamp": values.get("_timestamp", 0)}
            )
            row["scalars"].update(metrics)
            row["media"].update(media)
    finally:
        ds.close()
    rows = [histories[s] for s in sorted(histories)]
    system_stats.sort(key=lambda r: r[0])
    stat_times = [r[0] for r in system_stats]
    for row in rows:
        i = bisect.bisect_right(stat_times, row["source_timestamp"]) - 1
        if i >= 0:
            row["scalars"].update(system_stats[i][1])
    # Rank0 step wall time; exclude startup step1201.
    log = STEP_LOG if STEP_LOG is not None else ROOT.parent.parent / "node0.log"
    if log.exists():
        for line in log.open():
            m = re.search(
                r"Iteration (\d+): Loss: [\d.]+ \| (?:Iter speed: ([\d.]+) seconds per iteration|Time: ([\d.]+)s)", line
            )
            if m:
                step = int(m[1])
                if step in histories and step > min(histories) + 1:
                    histories[step]["scalars"]["performance/step_seconds"] = float(m[2] or m[3])
    with (ROOT / "history_export.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    audit = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "snapshot_bytes": copy.stat().st_size,
        "partial_tail_ignored": partial,
        "record_types": dict(types),
        "history_steps": len(rows),
        "first_step": rows[0]["step"] if rows else None,
        "last_step": rows[-1]["step"] if rows else None,
        "scalar_points": sum(len(r["scalars"]) for r in rows),
        "metric_keys": len({k for r in rows for k in r["scalars"]}),
        "media_references": sum(len(r["media"]) for r in rows),
        "skipped": dict(skipped),
        "network_upload_attempted": False,
    }
    save(ROOT / "extraction.json", audit)
    return rows, audit


def media_value(value, files):
    import swanlab

    path = (files / value["path"]).resolve()
    if not path.is_relative_to(files.resolve()) or not path.is_file():
        raise ValueError("missing_or_invalid_media_path")
    kind = value["_type"]
    if kind == "table-file":
        from html import escape

        data = json.loads(path.read_text())
        header = "<tr>" + "".join("<th>" + escape(str(v)) + "</th>" for v in data["columns"]) + "</tr>"
        body = "".join(
            "<tr>" + "".join("<td>" + escape(str(v)) + "</td>" for v in row) + "</tr>" for row in data["data"]
        )
        html = (
            '<!doctype html><meta charset="utf-8"><style>body{font:12px system-ui;margin:0}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:5px;white-space:nowrap}th{position:sticky;top:0;background:#eee}</style><table><thead>'
            + header
            + "</thead><tbody>"
            + body
            + "</tbody></table>"
        )
        return swanlab.Html(html, caption="Original W&B table: " + path.name)
    if kind == "image-file":
        return swanlab.Image(str(path))
    return swanlab.Audio(str(path))


def pending_values(row, sent, files, media_errors):
    step = row["step"]
    values = {key: value for key, value in row["scalars"].items() if step > sent.get(key, -1)}
    for key, value in row["media"].items():
        if step <= sent.get(key, -1):
            continue
        try:
            values[key] = media_value(value, files)
        except Exception as error:
            # Local optional-media conversion must not stop scalar delivery.
            # Leave its per-key cursor unchanged so the next cycle retries it.
            # Do not record exception messages: they may contain private paths.
            media_errors.append({"step": step, "key": key, "error_type": type(error).__name__})
    return values


def main():
    global ROOT, STEP_LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["export", "offline", "online"], default="export")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--duration", type=float, default=28800)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--workspace")
    ap.add_argument("--launch", type=Path)
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--name")
    ap.add_argument("--step-log", type=Path)
    ap.add_argument("--project", default="cosmos3_action")
    ap.add_argument("--key-file", type=Path, default=Path("/share-mixed/project/eai_pwm/.secrets/swanlab_api_key"))
    ap.add_argument("--proxy-file", type=Path)
    args = ap.parse_args()
    STEP_LOG = args.step_log.resolve() if args.step_log else None
    if args.state_dir:
        ROOT = args.state_dir.resolve()
        ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "bridge.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    source = args.source.resolve()
    launch = args.launch.resolve() if args.launch else None
    manifest = (
        json.loads((launch / "manifest.json").read_text())
        if launch
        else {
            "wandb_id": source.stem.removeprefix("run-"),
            "resume_iteration": None,
        }
    )
    state_path = ROOT / ("online_state.json" if args.mode == "online" else "offline_state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("source_wandb_run_id") not in (None, manifest["wandb_id"]):
        state_path.rename(state_path.with_name("previous_state_" + str(int(time.time())) + ".json"))
        state = {}
    run = None
    sent = {}
    confirmed = {}
    start = time.monotonic()
    try:
        while not STOP and time.monotonic() - start < args.duration:
            rows, audit = extract(source)
            if args.mode == "export":
                print(json.dumps(audit), flush=True)
                return
            if run is None:
                # Disable probes: the bridge host is not the GPU training host.
                for name in ["https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "all_proxy"]:
                    os.environ.pop(name, None)
                if args.proxy_file:
                    proxy = args.proxy_file.read_text().strip()
                    os.environ.update({k: proxy for k in ["https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"]})
                os.environ["SWANLAB_API_KEY"] = args.key_file.read_text().strip() if args.mode == "online" else ""
                import swanlab

                settings = swanlab.Settings(
                    interactive=False,
                    probe=swanlab.Settings.Probe(
                        hardware=False,
                        runtime=False,
                        requirements=False,
                        conda=False,
                        git=False,
                        swanlab=False,
                        monitor=False,
                    ),
                )
                swanlab.merge_settings(settings)
                if args.mode == "online":
                    swanlab.login(api_key=os.environ["SWANLAB_API_KEY"], save=False, timeout=10)
                run = swanlab.init(
                    mode=args.mode,
                    project=args.project,
                    workspace=args.workspace,
                    name=args.name or source.parent.name,
                    id=state.get("run_id") if args.mode == "online" else None,
                    resume="allow" if args.mode == "online" and state.get("run_id") else "never",
                    public=False,
                    log_dir=str(ROOT / ("swanlog_" + args.mode)),
                    config={
                        "source_wandb_run_id": manifest["wandb_id"],
                        "bridge_mode": "read_only_sidecar",
                        "resume_iteration": manifest["resume_iteration"],
                    },
                )
                state.update(
                    run_id=run.id,
                    mode=args.mode,
                    source_wandb_run_id=manifest["wandb_id"],
                    source_launch=str(launch) if launch else None,
                    source_wandb_file=str(source),
                )
                if args.mode == "online":
                    state.update(url=run.url, run_path=run.path)
                    # Server acknowledgements, not the previous local enqueue cursor.
                    summary = swanlab.Api().run(run.path.strip("/")).summary()
                    confirmed = {k: int(v["step"]) for k, v in summary.items() if isinstance(v, dict) and "step" in v}
                    sent = dict(confirmed)
                save(state_path, state)
            count = 0
            media_errors = []
            for row in rows:
                step = row["step"]
                values = pending_values(row, sent, source.parent / "files", media_errors)
                if values:
                    run.log(values, step=step)
                    sent.update({k: step for k in values})
                    count += len(values)
            if args.mode == "online":
                import swanlab

                summary = swanlab.Api().run(run.path.strip("/")).summary(keys=["iteration", "train/loss", "optim/lr"])
                confirmed = {k: int(v["step"]) for k, v in summary.items() if isinstance(v, dict) and "step" in v}
            state.pop("error_type", None)
            state.pop("error_frames", None)
            state.update(
                utc=datetime.now(timezone.utc).isoformat(),
                local_source_last_step=audit["last_step"],
                enqueued_points_this_cycle=count,
                latest_enqueued_iteration=sent.get("iteration"),
                server_confirmed_steps=confirmed,
                source_history_steps=audit["history_steps"],
                media_conversion_error_count=len(media_errors),
                media_conversion_errors=media_errors[:100],
                state="streaming" if args.mode == "online" else "offline_validation",
                pid=os.getpid(),
            )
            save(state_path, state)
            print(json.dumps(state), flush=True)
            if args.once:
                break
            until = time.monotonic() + args.interval
            while not STOP and time.monotonic() < until:
                time.sleep(1)
    except Exception as error:
        state.update(
            utc=datetime.now(timezone.utc).isoformat(),
            state="error",
            error_type=type(error).__name__,
            error_frames=[
                {"file": f.filename, "line": f.lineno, "function": f.name}
                for f in traceback.extract_tb(error.__traceback__)
            ][-8:],
        )
        save(state_path, state)
        print(json.dumps({"state": "error", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1) from None
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
