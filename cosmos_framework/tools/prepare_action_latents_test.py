"""Publication must preserve the public five-column catalog contract."""

import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_framework.data.generator.catalog import CATALOG_COLUMNS, ParquetCatalogDataset
from cosmos_framework.tools.prepare_action_latents import finalize, source_spec, write_json


def test_finalize_strips_runtime_catalog_fields(tmp_path):
    source = tmp_path / "source.parquet"
    row = dict(
        id="droid",
        type="action",
        dataset="droid",
        data=json.dumps(dict(adapter="droid", root="/droid")),
        provenance="{}",
    )
    pq.write_table(pa.Table.from_pylist([row]), source)
    assert "_catalog_index" in ParquetCatalogDataset(source)[0]
    assert tuple(source_spec(source)[0]) == CATALOG_COLUMNS
    write_json(tmp_path / "selection.json", dict(source_catalog=str(source), indices=[3], blocks=[[0, 1]]))
    write_json(tmp_path / "rank0.json", [dict(source_index=3, key="3")])
    finalize(SimpleNamespace(output=str(tmp_path)))
    for name in ("online", "offline"):
        assert tuple(pq.read_schema(tmp_path / f"{name}.parquet").names) == CATALOG_COLUMNS
        catalog = ParquetCatalogDataset(tmp_path / f"{name}.parquet")
        assert len(catalog) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["complete"] and manifest["records"][0]["source_index"] == 3


def test_finalize_rejects_missing_windows(tmp_path):
    write_json(tmp_path / "selection.json", dict(indices=[0, 1]))
    write_json(tmp_path / "rank0.json", [dict(source_index=0)])
    with pytest.raises(ValueError, match="Incomplete"):
        finalize(SimpleNamespace(output=str(tmp_path)))
    assert not (tmp_path / "manifest.json").exists()
