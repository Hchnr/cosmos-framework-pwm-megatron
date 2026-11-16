from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from tools.prepare_offline_vae_demo_catalogs import CATALOG_COLUMNS, main


def _write_source(root: Path, dataset: str) -> None:
    source = root / "latents" / f"{dataset}_wan2pt2_tokenizer-t480-test"
    output = source / "out"
    output.mkdir(parents=True)
    sha = "a" * 64
    (source / "MANIFEST.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "vae_name": "wan2pt2_tokenizer",
                "vae_tag": "wan2pt2_tokenizer-t480-test",
                "vae_weights_sha256": sha,
            }
        )
    )
    tensors = {
        f"{dataset}/accepted": torch.zeros(48, 2, 4, 6, dtype=torch.bfloat16),
        f"{dataset}/filtered": torch.zeros(48, 3, 4, 6, dtype=torch.bfloat16),
    }
    save_file(tensors, output / "chunk_00001.safetensors")
    metadata = []
    for suffix, shape in (("accepted", [48, 2, 4, 6]), ("filtered", [48, 3, 4, 6])):
        metadata.append(
            {
                "dataset": dataset,
                "record_id": f"{dataset}/{suffix}",
                "tensor_key": f"{dataset}/{suffix}",
                "shard_relpath": "out/chunk_00001.safetensors",
                "latent_shape": shape,
                "latent_dtype": "bfloat16",
                "latent_nbytes": torch.tensor(shape).prod().item() * 2,
                "target_fps": 24.0,
                "clip_id": suffix,
                "window_index": 0,
                "vae_tag": "wan2pt2_tokenizer-t480-test",
                "vae_weights_sha256": sha,
            }
        )
    (output / "chunk_00001.jsonl").write_text("".join(json.dumps(item) + "\n" for item in metadata))


def test_builds_valid_five_column_catalogs_and_filters_token_budget(tmp_path: Path) -> None:
    producer_root = tmp_path / "producer"
    output_root = tmp_path / "catalogs"
    for dataset in ("action100m", "koala-36m"):
        _write_source(producer_root, dataset)

    assert (
        main(
            [
                "--producer-root",
                str(producer_root),
                "--output-root",
                str(output_root),
                "--datasets",
                "action100m",
                "koala-36m",
                "--max-gen-tokens",
                "12",
            ]
        )
        == 0
    )

    for dataset in ("action100m", "koala-36m"):
        catalog_path = output_root / dataset / "catalog.parquet"
        parquet = pq.ParquetFile(catalog_path)
        assert tuple(parquet.schema_arrow.names) == CATALOG_COLUMNS
        assert parquet.metadata.num_rows == 1
        row = parquet.read().to_pylist()[0]
        assert row["id"] == f"{dataset}/accepted"
        assert row["dataset"] == dataset
        target = json.loads(row["data"])["messages"][1]["content"][0]
        assert target["conditioning_latent_frames"] == 0
        assert target["vae_latent"]["latent_shape"] == [48, 2, 4, 6]
