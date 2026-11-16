import json

import pytest
import torch
import torch.distributed.checkpoint as dcp

from tools.migrate_action128_physical_checkpoint import overlay_component


def test_overlay_replaces_only_requested_buffers_and_retains_source_hardlinks(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    original = {
        "learned.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "normalization.scale": torch.tensor([1.0, 2.0]),
        "iteration": 5200,
        "contract": json.dumps({"version": "old"}),
    }
    dcp.save(original, storage_writer=dcp.FileSystemWriter(source), no_dist=True)
    replacement = {"normalization.scale": torch.tensor([3.0, 4.0]), "contract": json.dumps({"version": "new"})}
    overlay_component(source, target, replacement)
    restored = {
        key: torch.empty_like(value) if isinstance(value, torch.Tensor) else (0 if isinstance(value, int) else "")
        for key, value in original.items()
    }
    dcp.load(restored, storage_reader=dcp.FileSystemReader(target), no_dist=True)
    assert torch.equal(restored["learned.weight"], original["learned.weight"])
    assert torch.equal(restored["normalization.scale"], replacement["normalization.scale"])
    assert restored["contract"] == replacement["contract"]
    assert restored["iteration"] == 5200
    old = {"contract": "", "normalization.scale": torch.empty(2)}
    dcp.load(old, storage_reader=dcp.FileSystemReader(source), no_dist=True)
    assert old["contract"] == original["contract"]
    assert torch.equal(old["normalization.scale"], original["normalization.scale"])
    assert (source / ".metadata").stat().st_ino != (target / ".metadata").stat().st_ino
    for path in source.glob("*.distcp"):
        assert path.stat().st_ino == (target / path.name).stat().st_ino
    with pytest.raises(ValueError, match="existing checkpoint keys"):
        overlay_component(source, tmp_path / "bad", {"unknown": torch.ones(1)})
