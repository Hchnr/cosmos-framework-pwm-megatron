#!/usr/bin/env python3
"""Build five-column offline-VAE smoke catalogs from producer demo artifacts.

This tool intentionally creates smoke-test catalogs with a fixed generic text
prompt.  It is not a replacement for the production caption/catalog pipeline.
By default it prepares the existing Action100M and Koala-36M 10k demo outputs
for the Nano T2V fixed-static recipe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from safetensors import safe_open

CATALOG_COLUMNS = ("id", "type", "dataset", "data", "provenance")
DEFAULT_PRODUCER_ROOT = Path("/share/project/eai_pwm/pipeline_store/demo/vae_demo_10k")
DEFAULT_OUTPUT_ROOT = Path("/share/project/eai_pwm/data/parquet-data/offline-vae-demo-10k")
DEFAULT_DATASETS = ("action100m", "koala-36m")
DEFAULT_PROMPT = "Generate a video that follows the described motion."
REQUIRED_MANIFEST_FIELDS = ("dataset", "vae_name", "vae_tag", "vae_weights_sha256")
EXPECTED_DTYPE = "BF16"


@dataclass(frozen=True)
class DatasetSource:
    name: str
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LatentRecord:
    row: dict[str, str]
    path: Path
    tensor_key: str
    latent_shape: tuple[int, int, int, int]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def discover_sources(producer_root: Path, datasets: Sequence[str]) -> list[DatasetSource]:
    latents_root = producer_root.expanduser().resolve() / "latents"
    if not latents_root.is_dir():
        raise FileNotFoundError(f"producer latent root does not exist: {latents_root}")

    discovered: dict[str, DatasetSource] = {}
    for manifest_path in sorted(latents_root.glob("*/MANIFEST.json")):
        manifest = _read_json_object(manifest_path)
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if not manifest.get(field)]
        if missing:
            raise ValueError(f"manifest {manifest_path} is missing required fields {missing}")
        dataset = str(manifest["dataset"])
        if dataset in discovered:
            raise ValueError(
                f"multiple producer roots declare dataset {dataset!r}: "
                f"{discovered[dataset].root} and {manifest_path.parent}"
            )
        discovered[dataset] = DatasetSource(
            name=dataset,
            root=manifest_path.parent.resolve(),
            manifest_path=manifest_path.resolve(),
            manifest=manifest,
        )

    missing_datasets = [dataset for dataset in datasets if dataset not in discovered]
    if missing_datasets:
        raise ValueError(f"producer root {latents_root} has no manifests for datasets {missing_datasets}")

    sources = [discovered[dataset] for dataset in datasets]
    lineage_fields = ("vae_name", "vae_tag", "vae_weights_sha256")
    reference = tuple(sources[0].manifest[field] for field in lineage_fields)
    for source in sources[1:]:
        actual = tuple(source.manifest[field] for field in lineage_fields)
        if actual != reference:
            raise ValueError(
                f"offline VAE lineage mismatch between {sources[0].manifest_path} and {source.manifest_path}: "
                f"expected {dict(zip(lineage_fields, reference, strict=True))}, "
                f"got {dict(zip(lineage_fields, actual, strict=True))}"
            )
    return sources


def _positive_shape(value: Any, *, metadata_path: Path, line_number: int) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in value)
    ):
        raise ValueError(f"{metadata_path}:{line_number}: latent_shape must be four positive integers [C,T,H,W]")
    return tuple(value)  # type: ignore[return-value]


def _resolve_latent_path(root: Path, shard_relpath: Any, *, metadata_path: Path, line_number: int) -> Path:
    if not isinstance(shard_relpath, str) or not shard_relpath:
        raise ValueError(f"{metadata_path}:{line_number}: shard_relpath must be a non-empty string")
    if Path(shard_relpath).is_absolute():
        raise ValueError(f"{metadata_path}:{line_number}: shard_relpath must be relative")
    path = (root / shard_relpath).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{metadata_path}:{line_number}: shard_relpath escapes producer root: {shard_relpath}")
    if path.suffix != ".safetensors":
        raise ValueError(f"{metadata_path}:{line_number}: latent shard must end in .safetensors: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{metadata_path}:{line_number}: latent shard does not exist: {path}")
    return path


def build_records(
    source: DatasetSource,
    *,
    prompt: str,
    expected_channels: int,
    latent_patch_size: int,
    max_gen_tokens: int,
) -> tuple[list[LatentRecord], int]:
    metadata_paths = sorted((source.root / "out").glob("chunk_*.jsonl"))
    if not metadata_paths:
        raise FileNotFoundError(f"no chunk_*.jsonl producer metadata found under {source.root / 'out'}")

    records_by_id: dict[str, LatentRecord] = {}
    filtered = 0
    for metadata_path in metadata_paths:
        with metadata_path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    metadata = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {metadata_path}:{line_number}: {error}") from error
                if not isinstance(metadata, dict):
                    raise ValueError(f"{metadata_path}:{line_number}: producer metadata must be a JSON object")
                if metadata.get("dataset") != source.name:
                    raise ValueError(
                        f"{metadata_path}:{line_number}: dataset={metadata.get('dataset')!r}, expected {source.name!r}"
                    )
                if metadata.get("vae_tag") != source.manifest["vae_tag"]:
                    raise ValueError(f"{metadata_path}:{line_number}: vae_tag does not match {source.manifest_path}")
                if metadata.get("vae_weights_sha256") != source.manifest["vae_weights_sha256"]:
                    raise ValueError(
                        f"{metadata_path}:{line_number}: vae_weights_sha256 does not match {source.manifest_path}"
                    )
                if metadata.get("latent_dtype") not in {"bfloat16", "BF16"}:
                    raise ValueError(f"{metadata_path}:{line_number}: latent_dtype must be bfloat16")

                row_id = metadata.get("record_id")
                tensor_key = metadata.get("tensor_key")
                target_fps = metadata.get("target_fps")
                if not isinstance(row_id, str) or not row_id:
                    raise ValueError(f"{metadata_path}:{line_number}: record_id must be a non-empty string")
                if not isinstance(tensor_key, str) or not tensor_key:
                    raise ValueError(f"{metadata_path}:{line_number}: tensor_key must be a non-empty string")
                if (
                    not isinstance(target_fps, (int, float))
                    or isinstance(target_fps, bool)
                    or not math.isfinite(float(target_fps))
                    or float(target_fps) <= 0
                ):
                    raise ValueError(f"{metadata_path}:{line_number}: target_fps must be finite and positive")

                latent_shape = _positive_shape(
                    metadata.get("latent_shape"), metadata_path=metadata_path, line_number=line_number
                )
                if latent_shape[0] != expected_channels:
                    raise ValueError(
                        f"{metadata_path}:{line_number}: latent C={latent_shape[0]}, expected {expected_channels}"
                    )
                expected_nbytes = math.prod(latent_shape) * 2
                if metadata.get("latent_nbytes") != expected_nbytes:
                    raise ValueError(
                        f"{metadata_path}:{line_number}: latent_nbytes={metadata.get('latent_nbytes')}, "
                        f"expected {expected_nbytes}"
                    )
                gen_tokens = (
                    latent_shape[1]
                    * math.ceil(latent_shape[2] / latent_patch_size)
                    * math.ceil(latent_shape[3] / latent_patch_size)
                )
                if gen_tokens > max_gen_tokens:
                    filtered += 1
                    continue

                latent_path = _resolve_latent_path(
                    source.root,
                    metadata.get("shard_relpath"),
                    metadata_path=metadata_path,
                    line_number=line_number,
                )
                data = {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "gen_video",
                                    "vae_latent": {
                                        "path": str(latent_path),
                                        "tensor_key": tensor_key,
                                        "latent_shape": list(latent_shape),
                                        "target_fps": float(target_fps),
                                    },
                                    "conditioning_latent_frames": 0,
                                }
                            ],
                        },
                    ]
                }
                row = {
                    "id": row_id,
                    "type": "visual_generation",
                    "dataset": source.name,
                    "data": json.dumps(data, separators=(",", ":")),
                    "provenance": json.dumps(
                        {
                            "source_id": metadata.get("clip_id"),
                            "window_index": metadata.get("window_index"),
                        },
                        separators=(",", ":"),
                    ),
                }
                record = LatentRecord(row=row, path=latent_path, tensor_key=tensor_key, latent_shape=latent_shape)
                existing = records_by_id.get(row_id)
                if existing is not None and existing != record:
                    raise ValueError(f"conflicting producer metadata for record_id {row_id!r}")
                records_by_id[row_id] = record

    records = list(records_by_id.values())
    if not records:
        raise ValueError(f"all {source.name} producer rows exceeded max_gen_tokens={max_gen_tokens}")
    return records, filtered


def validate_tensors(records: Sequence[LatentRecord]) -> None:
    by_path: dict[Path, list[LatentRecord]] = defaultdict(list)
    for record in records:
        by_path[record.path].append(record)
    for path, shard_records in sorted(by_path.items()):
        with safe_open(path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for record in shard_records:
                if record.tensor_key not in available:
                    raise ValueError(f"tensor key {record.tensor_key!r} is missing from {path}")
                tensor_slice = handle.get_slice(record.tensor_key)
                actual_shape = tuple(int(dim) for dim in tensor_slice.get_shape())
                actual_dtype = tensor_slice.get_dtype()
                if actual_shape != record.latent_shape:
                    raise ValueError(
                        f"tensor {record.tensor_key!r} in {path} has shape {actual_shape}, "
                        f"expected {record.latent_shape}"
                    )
                if actual_dtype != EXPECTED_DTYPE:
                    raise ValueError(
                        f"tensor {record.tensor_key!r} in {path} has dtype {actual_dtype}, expected {EXPECTED_DTYPE}"
                    )


def write_catalog(records: Sequence[LatentRecord], output_path: Path, *, row_group_size: int, overwrite: bool) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output_path}")

    schema = pa.schema([(name, pa.string()) for name in CATALOG_COLUMNS])
    table = pa.Table.from_pylist([record.row for record in records], schema=schema)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        pq.write_table(table, temporary_path, compression="zstd", row_group_size=row_group_size)
        written = pq.ParquetFile(temporary_path)
        if tuple(written.schema_arrow.names) != CATALOG_COLUMNS:
            raise RuntimeError(f"unexpected catalog schema in {temporary_path}: {written.schema_arrow.names}")
        if written.metadata.num_rows != len(records):
            raise RuntimeError(
                f"catalog row count mismatch in {temporary_path}: {written.metadata.num_rows} != {len(records)}"
            )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-root", type=Path, default=DEFAULT_PRODUCER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--expected-channels", type=int, default=48)
    parser.add_argument("--latent-patch-size", type=int, default=2)
    parser.add_argument("--max-gen-tokens", type=int, default=25_600)
    parser.add_argument("--row-group-size", type=int, default=256)
    parser.add_argument("--skip-tensor-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for name in ("expected_channels", "latent_patch_size", "max_gen_tokens", "row_group_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = discover_sources(args.producer_root, args.datasets)
    for source in sources:
        records, filtered = build_records(
            source,
            prompt=args.prompt,
            expected_channels=args.expected_channels,
            latent_patch_size=args.latent_patch_size,
            max_gen_tokens=args.max_gen_tokens,
        )
        if not args.skip_tensor_validation:
            validate_tensors(records)
        output_path = args.output_root / source.name / "catalog.parquet"
        write_catalog(records, output_path, row_group_size=args.row_group_size, overwrite=args.overwrite)
        print(
            f"dataset={source.name} rows={len(records)} filtered_over_token_budget={filtered} "
            f"catalog={output_path.resolve()} manifest={source.manifest_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
