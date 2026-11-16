# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

_INSTITUTIONS = [
    "AUTOLab",
    "CLVR",
    "GuptaLab",
    "ILIAD",
    "IPRL",
    "IRIS",
    "PennPAL",
    "RAD",
    "RAIL",
    "REAL",
    "RPL",
    "TRI",
    "WEIRD",
]

LEROBOT_ROOTS = {
    "droid_lerobot_20260115_no_noops": None,
    "droid_plus_lerobot_320x180_20260406_sharded": [f"success/{x}" for x in _INSTITUTIONS]
    + [f"failure/{x}" for x in _INSTITUTIONS],
    "droid_plus_lerobot_320x180_20260406": ["success", "failure"],
    "droid_plus_lerobot_640x360_20260412_sharded": [f"success/{x}" for x in _INSTITUTIONS]
    + [f"failure/{x}" for x in _INSTITUTIONS],
    "droid_plus_lerobot_640x360_20260412": ["success", "failure"],
}

IMAGE_FEATURES = {
    "droid_lerobot_20260115_no_noops": {
        "wrist": "observation.images.wrist_image_left",
        "left": "observation.images.exterior_image_1_left",
        "right": "observation.images.exterior_image_2_left",
    },
    "droid_plus_lerobot_320x180_20260406_sharded": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_320x180_20260406": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_640x360_20260412_sharded": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_640x360_20260412": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
}

STATE_FEATURES = {
    "droid_lerobot_20260115_no_noops": "observation.state",
    "droid_plus_lerobot_320x180_20260406_sharded": "observation.state.cartesian_position",
    "droid_plus_lerobot_320x180_20260406": "observation.state.cartesian_position",
    "droid_plus_lerobot_640x360_20260412_sharded": "observation.state.cartesian_position",
    "droid_plus_lerobot_640x360_20260412": "observation.state.cartesian_position",
}

ACTION_FEATURES = {
    "droid_lerobot_20260115_no_noops": "action",
    "droid_plus_lerobot_320x180_20260406_sharded": "action.gripper_position",
    "droid_plus_lerobot_320x180_20260406": "action.gripper_position",
    "droid_plus_lerobot_640x360_20260412_sharded": "action.gripper_position",
    "droid_plus_lerobot_640x360_20260412": "action.gripper_position",
}

IS_FLAT_ACTION = {
    "droid_lerobot_20260115_no_noops": True,
    "droid_plus_lerobot_320x180_20260406_sharded": False,
    "droid_plus_lerobot_320x180_20260406": False,
    "droid_plus_lerobot_640x360_20260412_sharded": False,
    "droid_plus_lerobot_640x360_20260412": False,
}

HAS_MULTI_LANGUAGE_ANNOTATIONS = {
    "droid_lerobot_20260115_no_noops": False,
    "droid_plus_lerobot_320x180_20260406_sharded": True,
    "droid_plus_lerobot_320x180_20260406": True,
    "droid_plus_lerobot_640x360_20260412_sharded": True,
    "droid_plus_lerobot_640x360_20260412": True,
}

IS_GRIPPER_ACTION_FLIPPED = {
    "droid_lerobot_20260115_no_noops": False,
    "droid_plus_lerobot_320x180_20260406_sharded": True,
    "droid_plus_lerobot_320x180_20260406": True,
    "droid_plus_lerobot_640x360_20260412_sharded": True,
    "droid_plus_lerobot_640x360_20260412": True,
}

_JOINT_ACTION_FEATURE = "action.joint_position"
_JOINT_STATE_FEATURE = "observation.state.joint_positions"
_GRIPPER_STATE_FEATURE = "observation.state.gripper_position"

CANONICAL_DROID_FEATURES = frozenset(
    {
        "observation.image.wrist_image_left",
        "observation.image.exterior_image_1_left",
        "observation.image.exterior_image_2_left",
        "observation.state.joint_positions",
        "observation.state.cartesian_position",
        "observation.state.gripper_position",
        "action.joint_position",
        "action.gripper_position",
    }
)


def resolve_droid_profile(root: str | Path) -> tuple[list[str] | None, dict[str, str], str, str, bool, bool, bool]:
    """Resolve a legacy named profile or a canonical metadata signature."""

    root_path = Path(root).expanduser().resolve()
    version = root_path.name
    if version in LEROBOT_ROOTS:
        return (
            LEROBOT_ROOTS[version],
            IMAGE_FEATURES[version],
            STATE_FEATURES[version],
            ACTION_FEATURES[version],
            IS_FLAT_ACTION[version],
            HAS_MULTI_LANGUAGE_ANNOTATIONS[version],
            IS_GRIPPER_ACTION_FLIPPED[version],
        )

    candidates: list[tuple[str | None, Path]] = []
    direct = root_path / "meta" / "info.json"
    if direct.is_file():
        candidates.append((None, direct))
    for split in ("success", "failure"):
        info_path = root_path / split / "meta" / "info.json"
        if info_path.is_file():
            candidates.append((split, info_path))
    if not candidates:
        raise ValueError(f"Unknown DROID root {str(root_path)!r}: no known basename and no meta/info.json")

    roots: list[str] = []
    for relative_root, info_path in candidates:
        with info_path.open(encoding="utf-8") as handle:
            info = json.load(handle)
        features = info.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"DROID metadata has no features object: {info_path}")
        missing = sorted(CANONICAL_DROID_FEATURES.difference(features))
        if missing:
            raise ValueError(f"DROID metadata is not canonical at {info_path}; missing features: {missing}")
        if relative_root is not None:
            roots.append(relative_root)

    return (
        roots or None,
        {
            "wrist": "observation.image.wrist_image_left",
            "left": "observation.image.exterior_image_1_left",
            "right": "observation.image.exterior_image_2_left",
        },
        "observation.state.cartesian_position",
        "action.gripper_position",
        False,
        True,
        True,
    )
