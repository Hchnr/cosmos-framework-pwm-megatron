# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.tools.load_balance_report import build_report, render_markdown

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _microbatch(rank: int, cp_group: int, cp_rank: int, forward_ms: float) -> dict:
    return {
        "record_type": "microbatch",
        "rank": rank,
        "world_size": 4,
        "iteration": 0,
        "cp_window_index": 0,
        "cp_window_slot": 0,
        "cp_group_id": cp_group,
        "cp_rank": cp_rank,
        "cp_size": 2,
        "is_active_payload_owner": cp_rank == 0,
        "forward_cuda_ms": forward_ms,
        "output": {
            "batch_size": 2,
            "und_token_length": 10,
            "gen_token_length": 20,
            "split_lens": [4, 8, 6, 12],
            "attn_modes": ["causal", "full", "causal", "full"],
            "vae_pixel_shapes": [[3, 4, 5]],
        },
        "data": {
            "video_shapes": [[[3, 93, 256, 256]], [[1, 3, 397, 256, 256]]],
            "sequence_plans": [
                {"vision_conditioning": "t2v", "vision_condition_frames": 0},
                {"vision_conditioning": "i2v", "vision_condition_frames": 1},
            ],
        },
    }


def test_build_report_collapses_cp_peers_before_comparing_groups() -> None:
    records = [
        *[{"record_type": "trace_start", "rank": rank, "world_size": 4} for rank in range(4)],
        _microbatch(0, 0, 0, 10.0),
        _microbatch(1, 0, 1, 10.0),
        _microbatch(2, 2, 0, 20.0),
        _microbatch(3, 2, 1, 20.0),
        *[{"record_type": "trace_end", "rank": rank, "world_size": 4} for rank in range(4)],
    ]

    report = build_report(records, file_count=4)

    assert report["integrity"]["rank_record_coverage"] == 1.0
    assert report["integrity"]["cp_rank_coverage"] == 1.0
    assert report["integrity"]["payload_owner_coverage"] == 1.0
    assert report["integrity"]["cp_output_consistency"] == 1.0
    timing = report["timings"]["forward_cuda_ms"]
    assert timing["total_mean_work_ms"] == 15.0
    assert timing["total_critical_path_ms"] == 20.0
    assert timing["weighted_bubble_proxy"] == 0.25
    assert timing["zero_overhead_balance_ratio_proxy"] == pytest.approx(4 / 3)
    assert timing["cp_peer_relative_spread"]["max"] == 0.0
    assert report["timings_by_cp_slot"]["0"]["forward_cuda_ms"] == timing
    assert report["correlations_by_cp_slot"]["0"]["forward_cuda_ms"]["observations"] == 2
    assert report["features"]["conditioning_counts"] == {"i2v": 2, "t2v": 2}
    sequence_lengths = report["features"]["conditioning_sequence_lengths"]
    assert sequence_lengths["t2v"]["und_token_length"]["p50"] == 4
    assert sequence_lengths["t2v"]["gen_token_length"]["p50"] == 8
    assert sequence_lengths["i2v"]["total_token_length"]["p50"] == 18
    frame_buckets = report["features"]["frame_bucket_sequence_lengths"]
    assert frame_buckets["93"]["total_token_length"]["p50"] == 12
    assert frame_buckets["397"]["gen_token_length"]["p50"] == 12
    markdown = render_markdown(report)
    assert "observed elapsed time" in markdown
    assert "Sequence length by vision conditioning" in markdown
    assert "Sequence length by pixel-frame bucket" in markdown
