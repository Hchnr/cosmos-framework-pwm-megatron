"""Validate published first/middle/last/cut-boundary cases with framework decoding."""

import argparse
import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from cosmos_framework.data.generator.action.media_timeline import MediaTimeline, episode_views
from cosmos_framework.data.generator.action.pretrain_state import current_state, load_state_transforms
from cosmos_framework.data.generator.action.unified_action128 import (
    BoundedRows,
    UnifiedAction128Dataset,
    build_action128_spec,
)
from cosmos_framework.data.generator.media_decode import SupervisedVideoDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="/share-mixed/projset/eai_pwm/datasets/action_processed_r2/catalogs/ego/framework_handoff_cases_20261030.json",
    )
    parser.add_argument("--output", default="outputs/ego6_validation/cases.json")
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--numbers-only", action="store_true")
    parser.add_argument(
        "--reference-code", help="Optional action-preprocess root; validation only, never a training dependency"
    )
    args = parser.parse_args()
    reference_frame = None
    if args.reference_code:
        sys.path.insert(0, args.reference_code)
        from egocentric.bulk_media import raw_frame

        reference_frame = raw_frame
    torch.set_num_threads(1)
    cases = json.loads(Path(args.cases).read_text())["cases"]
    spec = build_action128_spec(sorted({c["root"] for c in cases}))
    assert len(spec["registry"]["domains"]) == 6
    assert set(spec["registry"]["domain_projection_ids"]) == {63}
    assert set(spec["registry"]["domain_state_projection_ids"]) == {62}
    reader, timeline = BoundedRows(max_row_group_bytes=2 * 1024**3), MediaTimeline()
    decoder = SupervisedVideoDecoder(timeout_seconds=300, reuse_seek_groups=True)
    reports = []
    try:
        for case in cases[args.start_case :]:
            started = time.monotonic()
            root = Path(case["root"])
            parts = sorted(
                json.loads((root / "meta/catalog.json").read_text())["parts"], key=lambda p: p["episode_from"]
            )
            part = parts[bisect.bisect_right([p["episode_from"] for p in parts], case["episode_index"]) - 1]
            episode = next(
                e for e in pq.read_table(root / part["path"]).to_pylist() if e["episode_index"] == case["episode_index"]
            )
            rows = reader.episode_rows(root / episode["data_path"], episode, use_state=True)
            report = {"dataset": case["dataset"], "episode": case["episode_index"], "views": []}
            transform = load_state_transforms(root)["domains"][episode["action_domain"]]["transform"]
            for expected in case["expected_rows"]:
                index = expected["numeric_row"]
                row = rows.slice(index, 1).to_pylist()[0]
                assert abs(row["timestamp"] - expected["published_timestamp"]) < 1e-7
                assert row["action_valid"] == expected["action_valid"]
                assert sum(row["action_dim_mask"]) == expected["active_action_slots"]
                assert sum(row["observation.state_mask"]) == expected["active_state_slots"]
                state, mask, relative = current_state(rows, start=row["timestamp"], transform=transform)
                assert torch.isfinite(state).all()
                assert len(state) == int(expected["active_state_slots"] > 0)
                assert int(mask.sum()) == expected["active_state_slots"]
                if not len(state):
                    continue
                assert relative.item() <= 1e-7
                identity = ~np.array(transform["normalization_mask"], dtype=bool) & np.array(
                    row["observation.state_mask"]
                )
                np.testing.assert_allclose(
                    state.numpy()[0, identity], np.array(row["observation.state"])[identity], rtol=0, atol=0
                )
            report["numeric_contract_pass"] = True
            report["action_head"], report["state_head"] = 63, 62
            if args.numbers_only:
                reports.append(report)
                print("NUMERIC PASS", case["dataset"], case["episode_index"], flush=True)
                continue
            for view in episode_views(episode):
                times = [r["published_timestamp"] for r in case["expected_rows"]]
                query, indexes, meta = timeline.resolve(root, episode, view, times, rows)
                expected = [
                    next(v for v in r["expected_views"] if v["role"] == view["role"]) for r in case["expected_rows"]
                ]
                np.testing.assert_allclose(query, [v["decoder_seconds"] for v in expected], atol=1e-7, rtol=0)
                if view["container_format"] == "mcap" and not view.get("timeline_key"):
                    # The reference reports the numeric row's main-camera ID
                    # here. MCAP has no declared per-topic frame-index mapping;
                    # its decoder must use topic + epoch timestamp, not that ID.
                    assert indexes is None
                    meta["reference_main_frame_ids_not_used_for_mcap"] = [v["source_frame_index"] for v in expected]
                else:
                    assert indexes.tolist() == [v["source_frame_index"] for v in expected]
                options = {"timestamps_sec": query.tolist()}
                if view["container_format"] in {"mcap", "zarr"}:
                    options.update(native_view=view, source_frame_indexes=None if indexes is None else indexes.tolist())
                frames, decoded = decoder.decode(str(root / view["path"]), **options)
                assert len(frames) == len(times) and all(f.ndim == 3 and f.shape[-1] == 3 for f in frames)
                tolerance = 2.0 / view["source_fps"] if view["container_format"] == "mcap" else 1e-6
                assert max(decoded["timestamp_error_seconds"]) <= tolerance
                if reference_frame is not None:
                    native_view = dict(view, fps=view["source_fps"])
                    for i, frame in enumerate(frames):
                        reference = reference_frame(
                            root / view["path"],
                            native_view,
                            float(query[i]),
                            None if indexes is None else int(indexes[i]),
                        )
                        np.testing.assert_array_equal(frame, reference)
                    meta["reference_pixels_equal"] = True
                report["views"].append({**meta, "decode": decoded, "shapes": [list(f.shape) for f in frames]})
                print("FRAME PASS", case["dataset"], case["episode_index"], view["role"], query.tolist(), flush=True)
            # Exercise the actual canvas path at the start and the final partial window.
            dataset = UnifiedAction128Dataset.__new__(UnifiedAction128Dataset)
            dataset.root, dataset.duration, dataset.decoder, dataset.media_timeline = root, 2.0, decoder, timeline
            dataset.video_fps = dataset.video_fps_max = None
            dataset.zarr_tar_decoder = type("UnusedLegacy", (), {"close": lambda _: None})()
            duration = episode["duration_seconds"]
            for start in sorted({0.0, 2.0 * max(0, int(np.ceil(duration / 2)) - 1)}):
                video, valid, cameras, metadata, width, height = dataset._video(
                    episode, start, min(start + 2, duration), dataset.sampling_video_fps(episode), rows=rows
                )
                assert video.shape == (3, video.shape[1], height, width) and video.shape[1] % 4 == 1
                assert valid[0].all() and cameras[0].any()
                print("WINDOW PASS", case["dataset"], case["episode_index"], start, tuple(video.shape), flush=True)
            report["elapsed_seconds"] = time.monotonic() - started
            reports.append(report)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(reports, indent=2))
    finally:
        decoder.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(reports, indent=2))
    print("PASS", len(reports), "real cases", flush=True)


if __name__ == "__main__":
    main()
