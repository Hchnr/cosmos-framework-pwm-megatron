import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cosmos_framework.data.generator.action.physical_normalization import apply_affine, validate_physical_document
from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset, build_action128_spec
from cosmos_framework.data.generator.catalog import CatalogSchemaError
from cosmos_framework.data.test.action128_test import media_stub as media_stub
from cosmos_framework.data.test.action128_test import source as source
from cosmos_framework.data.test.pretrain_state_test import state_source as state_source


def physical_document(document, field):
    document = copy.deepcopy(document)
    document["policy_version"] = "robot-limit-affine-no-clip-v1"
    document["fit_scope"] = {
        "data_role": "robot_specification",
        "training_eligible": False,
        "row_mask": "not_applicable",
        "slot_mask": "action_dim_mask" if field == "action" else "observation.state_mask",
        "control_mask_used_for_fit": False,
    }
    document["application"]["field"] = field
    for record in document["domains"].values():
        mask = record["transform"]["normalization_mask"]
        record["transform"] = {
            "kind": ["robot_limit_affine" if active else "inactive" for active in mask],
            "normalization_mask": mask,
            "scale": [0.5 if active else 1.0 for active in mask],
            "bias": [0.0] * 128,
            "clip_policy": "none",
            "nominal_min": -1.0,
            "nominal_max": 1.0,
        }
        record["statistics"] = {
            "lower_bound": [-2.0 if active else 0 for active in mask],
            "upper_bound": [2.0 if active else 0 for active in mask],
            "active_slot_count": sum(mask),
            "source": "robot_urdf_limits",
        }
    return document


def test_physical_action_and_current_state_keep_overshoot_and_mask_absent_values(state_source, media_stub):
    root, _ = state_source
    legacy = UnifiedAction128Dataset(root, use_state=True, tokenizer_config=object(), video_decoder=media_stub)
    old_spec = build_action128_spec([str(root)])
    for name, field in [("action_normalization.json", "action"), ("state_normalization.json", "observation.state")]:
        path = root / "meta" / name
        path.write_text(json.dumps(physical_document(json.loads(path.read_text()), field)))
    dataset = UnifiedAction128Dataset(root, use_state=True, tokenizer_config=object(), video_decoder=media_stub)
    sample = dataset[0]
    assert legacy[0]["action"][1, 0] == -1
    assert sample["action"][1, 0] == -1.5
    assert torch.equal(sample["action_dim_mask"], legacy[0]["action_dim_mask"])
    assert torch.equal(sample["action_loss_mask"], legacy[0]["action_loss_mask"])
    assert not sample["action"][:, ~sample["action_dim_mask"].any(0)].any()
    assert build_action128_spec([str(root)]) != old_spec
    path = root / "data/frames.parquet"
    table = pq.read_table(path)
    values = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    values[0, 0] = 6.0
    table = table.set_column(
        table.schema.get_field_index("observation.state"),
        "observation.state",
        pa.array(values.tolist(), type=pa.list_(pa.float32(), 128)),
    )
    pq.write_table(table, path)
    dataset = UnifiedAction128Dataset(root, use_state=True, tokenizer_config=object(), video_decoder=media_stub)
    assert dataset[0]["action"][0, 0] == 3.0


@pytest.mark.parametrize("corruption", ["scale", "clip", "scope", "kind"])
def test_physical_bounds_and_clip_contract_are_checked(source, corruption):
    root, _, _ = source
    document = physical_document(json.loads((root / "meta/action_normalization.json").read_text()), "action")
    validate_physical_document(document, field="action")
    record = next(iter(document["domains"].values()))
    if corruption == "scale":
        record["transform"]["scale"][0] *= 2
    if corruption == "clip":
        record["transform"]["clip_policy"] = "clamp"
    if corruption == "scope":
        document["fit_scope"]["data_role"] = "training"
    if corruption == "kind":
        record["transform"]["kind"][0] = "robust_quantile"
    with pytest.raises(CatalogSchemaError):
        validate_physical_document(document, field="action")


def test_legacy_affine_retains_clipping():
    transform = {
        "normalization_mask": [True] + [False] * 127,
        "scale": [0.5] + [1.0] * 127,
        "bias": [0.0] * 128,
        "clip_min": -1.0,
        "clip_max": 1.0,
    }
    values = torch.zeros(2, 128)
    values[:, 0] = torch.tensor([-6.0, 6.0])
    assert apply_affine(values, transform)[:, 0].tolist() == [-1.0, 1.0]
