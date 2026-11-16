"""Prepare window-local DROID Offline VAE data (prepare -> torchrun encode -> finalize)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def source_spec(path):
    from cosmos_framework.data.generator.catalog import CATALOG_COLUMNS, ParquetCatalogDataset

    row = ParquetCatalogDataset(path)[0]
    return {key: row[key] for key in CATALOG_COLUMNS}, dict(row["data"])


def source_dataset(path):
    from cosmos_framework.data.generator.catalog_processors import build_action_dataset_from_catalog

    return build_action_dataset_from_catalog(path, tokenizer_config=None)


class RawWindows(Dataset):
    def __init__(self, source, indices):
        self.source, self.indices = source, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        raw = self.source._dataset
        fetch = raw._fetch_sample
        captions = []

        def capture(idx):
            result = fetch(idx)
            if raw._has_multi_language_annotations:
                captions.extend(result[3]["task"].split(" | "))
            return result

        raw._fetch_sample = capture
        try:
            sample = raw[index]
        finally:
            raw._fetch_sample = fetch
        if captions:
            sample["_offline_caption_choices"] = captions
        sample = self.source._transform.video_resize(sample, self.source._resolution)
        return index, sample


def identity(x):
    return x


def prepare(args):
    _, spec = source_spec(args.catalog)
    if spec.get("adapter") != "droid" or spec.get("use_image_augmentation", False):
        raise ValueError("Producer currently requires deterministic DROID media without image augmentation")
    if spec.get("mode", "wam") == "joint":
        raise ValueError("Random task selection must not be frozen into an offline dataset")
    dataset = source_dataset(args.catalog)
    raw = dataset._dataset
    if getattr(raw, "_max_num_history_actions", 0):
        raise ValueError("Random caption/history sampling needs an explicit offline contract")
    if getattr(raw, "_video_mode", None) is not None:
        raise ValueError("Only deterministic default camera composition is supported")
    blocks = dataset.get_shuffle_blocks()
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(blocks), generator=generator).tolist() if args.max_samples else range(len(blocks))
    indices, selected_blocks = [], []
    for b in order:
        start, length = blocks[b]
        if args.max_samples:
            length = min(length, args.max_samples - len(indices), args.windows_per_episode)
        if not length:
            break
        selected_blocks.append((len(indices), length))
        indices.extend(range(start, start + length))
    defaults = dict(
        cfg_dropout_rate=0.1,
        max_action_dim=64,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
        format_prompt_as_json=False,
    )
    transform = {k: spec.get(k, v) for k, v in defaults.items()}
    plan = dict(
        format="cosmos_action_latents_v1",
        source_catalog=str(Path(args.catalog).resolve()),
        source_catalog_sha256=digest(args.catalog),
        source_spec=spec,
        source_length=len(dataset),
        vae_path=str(Path(args.vae).resolve()),
        vae_weights_sha256=digest(args.vae),
        encode_chunk_frames={"256": 68, "480": 24, "720": 8, "768": 8},
        encode_exact_durations=[33],
        seed=args.seed,
        indices=indices,
        blocks=selected_blocks,
        transform=transform,
        resolution=spec.get("resolution", "256"),
        complete=False,
        subset=bool(args.max_samples),
    )
    path = Path(args.output) / "selection.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError("Existing selection differs; use a new output directory")
    write_json(path, plan)
    print(
        json.dumps(
            {
                "source_windows": len(dataset),
                "selected_windows": len(indices),
                "blocks": len(selected_blocks),
                "subset": plan["subset"],
            }
        ),
        flush=True,
    )


def encode(args):
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    root = Path(args.output)
    plan = json.loads((root / "selection.json").read_text())
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    if digest(plan["source_catalog"]) != plan["source_catalog_sha256"]:
        raise ValueError("Source catalog changed after selection")
    if digest(plan["vae_path"]) != plan["vae_weights_sha256"]:
        raise ValueError("VAE weights changed after selection")
    vae = Wan2pt2VAEInterface(
        vae_path=plan["vae_path"],
        encode_chunk_frames=plan["encode_chunk_frames"],
        encode_exact_durations=plan["encode_exact_durations"],
    )
    indices = plan["indices"][rank::world]
    result_path = root / f"rank{rank}.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if [r["source_index"] for r in previous] == indices:
            print(f"rank {rank}: already complete", flush=True)
            return
        raise ValueError("Completed shard does not match selection/world size")
    selection_hash = digest(root / "selection.json")
    records = []
    parts = sorted(root.glob(f"part_rank{rank}_*.json"))
    for path in parts:
        part = json.loads(path.read_text())
        if part["selection_sha256"] != selection_hash or part["world_size"] != world:
            raise ValueError("Partial encode belongs to a different selection or world size")
        records.extend(part["records"])
    if [r["source_index"] for r in records] != indices[: len(records)]:
        raise ValueError("Partial encode is not a contiguous prefix of this rank's windows")
    dataset = RawWindows(source_dataset(plan["source_catalog"]), indices[len(records) :])
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers,
        collate_fn=identity,
        prefetch_factor=2 if args.workers else None,
    )
    tensors, chunk = {}, []
    shard_no = len(parts)

    def flush():
        nonlocal tensors, chunk, shard_no
        name = f"rank{rank}_{shard_no:05d}.safetensors"
        tmp = root / (name + ".tmp")
        save_file(tensors, str(tmp), metadata={"selection_sha256": digest(root / "selection.json")})
        tmp.replace(root / name)
        for record in chunk:
            record["shard"] = name
        write_json(
            root / f"part_rank{rank}_{shard_no:05d}.json",
            {
                "selection_sha256": selection_hash,
                "world_size": world,
                "records": chunk,
            },
        )
        records.extend(chunk)
        tensors, chunk = {}, []
        shard_no += 1

    with torch.inference_mode():
        for index, sample in loader:
            video = sample.pop("video")
            pixels = video.unsqueeze(0).cuda().float() / 127.5 - 1.0
            latent = vae.encode(pixels)
            size = sample["image_size"].flatten()
            latent = latent[0, :, :, : int(size[2]) // 16, : int(size[3]) // 16].to(torch.bfloat16).cpu().contiguous()
            prefix = str(index)
            tensors[prefix + ".latent"] = latent
            meta, tensor_keys = {}, []
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    tensors[prefix + "." + key] = value.cpu().contiguous()
                    tensor_keys.append(key)
                else:
                    meta[key] = value
            chunk.append(
                dict(
                    source_index=index,
                    key=prefix,
                    metadata=meta,
                    tensor_keys=tensor_keys,
                    video_shape=list(video.shape),
                    latent_shape=list(latent.shape),
                )
            )
            if len(chunk) >= args.shard_size:
                flush()
                print(f"rank {rank}: {len(records)}/{len(indices)}", flush=True)
        if chunk:
            flush()
    write_json(result_path, records)


def finalize(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(args.output).resolve()
    plan = json.loads((root / "selection.json").read_text())
    records = {}
    for path in sorted(root.glob("rank*.json")):
        for record in json.loads(path.read_text()):
            if record["source_index"] in records:
                raise ValueError("Duplicate encoded source index")
            records[record["source_index"]] = record
    if set(records) != set(plan["indices"]):
        raise ValueError(f"Incomplete data: {len(records)} / {len(plan['indices'])}")
    manifest = dict(plan, records=[records[i] for i in plan["indices"]], complete=True)
    manifest.pop("indices")
    write_json(root / "manifest.json", manifest)
    row, source = source_spec(plan["source_catalog"])
    for name, data in (
        ("offline", dict(adapter="droid_offline", manifest_path=str(root / "manifest.json"))),
        ("online", dict(source, selection_manifest=str(root / "manifest.json"))),
    ):
        out = dict(row, data=json.dumps(data))
        if isinstance(out.get("provenance"), dict):
            out["provenance"] = json.dumps(out["provenance"])
        pq.write_table(pa.Table.from_pylist([out]), root / f"{name}.parquet")
    print(f"Published {len(records)} windows: {root}", flush=True)


def evaluation(args):
    """Convert a trusted local fixed-action torch fixture using the same VAE contract."""
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    if not args.fixture:
        raise ValueError("evaluation requires --fixture pointing to a trusted local torch fixture")
    root = Path(args.output)
    plan = json.loads((root / "selection.json").read_text())
    if digest(plan["vae_path"]) != plan["vae_weights_sha256"]:
        raise ValueError("VAE weights changed after selection")
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    vae = Wan2pt2VAEInterface(
        vae_path=plan["vae_path"],
        encode_chunk_frames=plan["encode_chunk_frames"],
        encode_exact_durations=plan["encode_exact_durations"],
    )
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    with torch.inference_mode():
        for batch in fixture["reference_batches"]:
            latents = []
            for items, size in zip(batch.pop("video"), batch["image_size"], strict=True):
                video = items[0] if isinstance(items, list) else items
                size = size[0] if isinstance(size, list) else size
                size = size.flatten()
                latent = vae.encode(video.unsqueeze(0).cuda().float() / 127.5 - 1.0)
                latent = latent[0, :, :, : int(size[2]) // 16, : int(size[3]) // 16]
                latents.append([latent.to(torch.bfloat16).cpu().contiguous()])
            batch["video_latent"] = latents
    fixture["online_fixture_sha256"] = digest(args.fixture)
    fixture["vae_weights_sha256"] = plan["vae_weights_sha256"]
    path = root / "flow_eval_offline.pt"
    torch.save(fixture, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    print(f"Converted {len(fixture['reference_batches'])} fixed validation batches: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "encode", "finalize", "evaluation"])
    parser.add_argument(
        "--catalog",
        default="/share/project/eai_pwm/datasets/cosmos3_action_tower_training/v0.2.1_more_data/droid_success/train.parquet",
    )
    parser.add_argument("--vae", default="/share/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture", help="Trusted local fixed-action evaluation .pt file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means the entire source dataset")
    parser.add_argument("--windows-per-episode", type=int, default=184)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=128)
    args = parser.parse_args()
    if args.max_samples < 0 or args.workers < 0 or min(args.shard_size, args.windows_per_episode) < 1:
        parser.error("sample/worker counts must be nonnegative and shard/window sizes positive")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
