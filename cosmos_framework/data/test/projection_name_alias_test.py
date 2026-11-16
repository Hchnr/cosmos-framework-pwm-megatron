"""Robot-model projection aliases for published tactile pretraining data."""

from cosmos_framework.data.generator.action.projection_routing import (
    PUBLISHED_PROJECTION_FAMILY_IDS,
    projection_head_id,
)


def test_openneo_uses_existing_joint_robot_heads():
    assert projection_head_id("joint128", "agilex_aloha__parallel_gripper") == 2
    assert projection_head_id("joint128", "arx_x5__parallel_gripper") == 7
    for name, expected in (
        ("agilex_aloha__parallel_gripper", 2),
        ("arx_x5__parallel_gripper", 7),
    ):
        key = f"joint::{name}/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1"
        assert PUBLISHED_PROJECTION_FAMILY_IDS[key] == expected


def test_univtac_franka_uses_existing_franka_joint_head():
    assert projection_head_id("joint128", "franka_panda__parallel_gripper") == 8
    key = (
        "joint::franka_panda__parallel_gripper/joint128/"
        "observed_absolute_state_t_plus_1/"
        "univtac-franka9-two-finger-mean04-next-state-v1"
    )
    assert PUBLISHED_PROJECTION_FAMILY_IDS[key] == 8


def test_new_joint_families_are_compact_without_splitting_bench2dex():
    new_families = {
        key: value
        for key, value in PUBLISHED_PROJECTION_FAMILY_IDS.items()
        if key.startswith("joint::") and 33 <= value < 62
    }
    assert sorted(set(new_families.values())) == list(range(33, 42))
    assert new_families["joint::bench2dex_all_pretrain_v1"] == 33
