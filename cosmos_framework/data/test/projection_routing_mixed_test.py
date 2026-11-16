"""Projection policy for the mixed dexterous pretraining collection."""

from cosmos_framework.data.generator.action.projection_routing import (
    EEF_ABSOLUTE_STATE_HEAD_ID,
    MIXED_EEF_ACTION_HEAD_ID,
    MIXED_EEF_STATE_HEAD_ID,
    PUBLISHED_PROJECTION_FAMILY_IDS,
    projection_head_id,
)


def test_mixed_eef_heads_do_not_change_robot_action_eef():
    assert (MIXED_EEF_STATE_HEAD_ID, MIXED_EEF_ACTION_HEAD_ID) == (62, 63)
    assert (EEF_ABSOLUTE_STATE_HEAD_ID, projection_head_id("eef20", "franka_panda__parallel_gripper")) == (32, 0)
    for key, head in PUBLISHED_PROJECTION_FAMILY_IDS.items():
        if key.startswith("action::"):
            assert head == MIXED_EEF_ACTION_HEAD_ID, key
        elif key.startswith("state::"):
            assert head == MIXED_EEF_STATE_HEAD_ID, key


def test_joint_families_keep_distinct_shared_state_action_heads():
    assert PUBLISHED_PROJECTION_FAMILY_IDS["joint::paxini_df2_bimanual_v1"] == 29
    assert PUBLISHED_PROJECTION_FAMILY_IDS["joint::sharpa_north_wave_bimanual_v1"] == 30
    assert PUBLISHED_PROJECTION_FAMILY_IDS["joint::trex_vega1_sharpa_wave_bimanual_v1"] == 40
    assert PUBLISHED_PROJECTION_FAMILY_IDS["joint::bench2dex_all_pretrain_v1"] == 33
    assert len({MIXED_EEF_ACTION_HEAD_ID, MIXED_EEF_STATE_HEAD_ID, 40}) == 3
