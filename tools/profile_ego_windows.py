"""Read-only real-window performance diagnosis; no training/data mutations."""

import argparse
import bisect
import cProfile
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from cosmos_framework.data.generator.action import unified_action128 as ua
from cosmos_framework.data.generator.action.media_timeline import MediaTimeline
from cosmos_framework.data.generator.action.native_video import NativeVideoReader
from cosmos_framework.data.generator.media_decode import SupervisedVideoDecoder, decode_video_at_timestamps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/ego_profile")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare helper-resized pixels to the old worker resize; cold timing includes verification",
    )
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cases = json.loads(
        Path(
            "/share-mixed/projset/eai_pwm/datasets/action_processed_r2/catalogs/ego/framework_handoff_cases_20261030.json"
        ).read_text()
    )["cases"]
    groups = {}
    for case in cases:
        groups.setdefault(case["dataset"], []).append(case)
    reports = []
    original_copy = ua._copy_resized_camera
    for name, group in groups.items():
        if args.dataset and name != args.dataset:
            continue
        case = group[len(group) // 2]
        root = Path(case["root"])
        t = time.perf_counter()
        parts = sorted(json.loads((root / "meta/catalog.json").read_text())["parts"], key=lambda p: p["episode_from"])
        part = parts[bisect.bisect_right([p["episode_from"] for p in parts], case["episode_index"]) - 1]
        episode = next(
            e for e in pq.read_table(root / part["path"]).to_pylist() if e["episode_index"] == case["episode_index"]
        )
        metadata_seconds = time.perf_counter() - t
        reader = ua.BoundedRows(max_row_group_bytes=2 * 1024**3)
        t = time.perf_counter()
        rows = reader.episode_rows(root / episode["data_path"], episode, use_state=True)
        row_seconds = time.perf_counter() - t
        decoder = SupervisedVideoDecoder(timeout_seconds=300, reuse_seek_groups=True)
        dataset = ua.UnifiedAction128Dataset.__new__(ua.UnifiedAction128Dataset)
        dataset.root, dataset.duration = root, 2.0
        dataset.media_timeline = MediaTimeline()
        dataset.video_fps = dataset.video_fps_max = None
        dataset.zarr_tar_decoder = type("UnusedLegacy", (), {"close": lambda _: None})()
        report = dict(
            dataset=name,
            episode=case["episode_index"],
            metadata_seconds=metadata_seconds,
            numeric_read_seconds=row_seconds,
            windows=[],
        )
        calls = []
        copies = []
        captured = []
        verified = set()

        class TimedDecoder:
            def decode(self, path, **kwargs):
                started = time.perf_counter()
                frames, meta = decoder.decode(path, **kwargs)
                calls.append(
                    dict(
                        path=path,
                        seconds=time.perf_counter() - started,
                        frames=len(frames),
                        bytes=sum(f.nbytes for f in frames),
                        shape=list(frames[0].shape),
                        native=kwargs.get("native_view", {}).get("container_format", "mp4"),
                        decoded_frames=meta.get("decoded_frames_per_seek"),
                    )
                )
                captured.append((path, kwargs))
                key = (path, kwargs.get("native_view", {}).get("mcap_topic"))
                if args.verify and key not in verified and "camera_output_size" in kwargs:
                    raw_options = dict(kwargs)
                    height, width = raw_options.pop("camera_output_size")
                    raw, _ = decoder.decode(path, **raw_options)
                    expected = torch.empty((3, len(raw), height, width), dtype=torch.uint8)
                    original_copy(raw, expected)
                    import numpy as np

                    np.testing.assert_array_equal(np.stack(frames), expected.permute(1, 2, 3, 0).numpy())
                    verified.add(key)
                    print("PIXEL EXACT", name, key, flush=True)
                return frames, meta

            def close(self):
                decoder.close()

        def timed_copy(*a, **kw):
            started = time.perf_counter()
            result = original_copy(*a, **kw)
            copies.append(time.perf_counter() - started)
            return result

        dataset.decoder = TimedDecoder()
        ua._copy_resized_camera = timed_copy
        duration = episode["duration_seconds"]
        start = max(0, (duration - 2) / 2)
        try:
            for repeat in range(3):
                calls.clear()
                copies.clear()
                captured.clear()
                t = time.perf_counter()
                result = dataset._video(
                    episode, start, min(start + 2, duration), dataset.sampling_video_fps(episode), rows=rows
                )
                total = time.perf_counter() - t
                report["windows"].append(
                    dict(
                        repeat=repeat,
                        total_seconds=total,
                        decode_seconds=sum(x["seconds"] for x in calls),
                        resize_seconds=sum(copies),
                        output_shape=list(result[0].shape),
                        views=list(calls),
                    )
                )
                print(name, repeat, json.dumps(report["windows"][-1]), flush=True)
                del result
            # Separate in-process decode to locate codec/IO costs vs helper IPC.
            native = NativeVideoReader()
            profile = cProfile.Profile()
            for i, (path, kwargs) in enumerate(captured):
                kwargs = dict(kwargs)
                kwargs.pop("camera_output_size", None)
                limits = dict(
                    max_sampled_frames=256,
                    max_frame_pixels=16777216,
                    max_window_bytes=1073741824,
                    max_decode_frames_per_seek=2048,
                )
                for repeat in range(2):
                    t = time.perf_counter()
                    if "native_view" in kwargs:
                        frames, _ = profile.runcall(native.decode, path, **kwargs, **limits)
                    else:
                        frames, _ = profile.runcall(
                            decode_video_at_timestamps, path, **kwargs, **limits, reuse_seek_groups=True
                        )
                    print("DIRECT", name, i, repeat, time.perf_counter() - t, flush=True)
                    del frames
            native.close()
            profile.dump_stats(str(out / f"{name}.prof"))
        finally:
            decoder.close()
            ua._copy_resized_camera = original_copy
        reports.append(report)
        (out / "windows.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
