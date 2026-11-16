"""Create a model-only DCP with EEF absolute-State input slot 32 initialized from 0."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source, output = args.source.resolve(), args.output.absolute()
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    reader = dcp.FileSystemReader(source / "model")
    metadata = reader.read_metadata().state_dict_metadata
    if not all(isinstance(v, TensorStorageMetadata) for v in metadata.values()):
        raise TypeError("Expected model-only tensor metadata")
    state = {k: torch.empty(v.size, dtype=v.properties.dtype) for k, v in metadata.items()}
    print(f"Loading {len(state)} tensors", flush=True)
    dcp.load(state, storage_reader=reader, no_dist=True)
    changed = []
    for prefix in ("net", "net_ema"):
        keys = [f"{prefix}.action2llm.{suffix}" for suffix in ("fc.weight", "bias.weight")]
        if prefix == "net_ema" and not any(k.startswith("net_ema.") for k in state):
            continue
        for key in keys:
            tensor = state[key]
            if tensor.ndim != 2 or tensor.shape[0] != 64:
                raise ValueError(f"Unexpected projection shape: {key} {tensor.shape}")
            before = digest(tensor)
            tensor[32].copy_(tensor[0])
            changed.append({"key": key, "source_slot": 0, "target_slot": 32, "before_sha256": before})
    expected = {k: digest(v) for k, v in state.items()}
    print(f"Saving; updated {[entry['key'] for entry in changed]}", flush=True)
    dcp.save(state, storage_writer=dcp.FileSystemWriter(temporary / "model", thread_count=8))
    print("Reloading and verifying every tensor", flush=True)
    for tensor in state.values():
        tensor.zero_()
    dcp.load(state, checkpoint_id=temporary / "model", no_dist=True)
    assert set(dcp.FileSystemReader(temporary / "model").read_metadata().state_dict_metadata) == set(metadata)
    assert all(digest(v) == expected[k] for k, v in state.items())
    for entry in changed:
        assert torch.equal(state[entry["key"]][32], state[entry["key"]][0])
    # Copy configuration only, never optimizer, scheduler, cursor or step state.
    for src, dest in (
        (source / "model/config.json", temporary / "model/config.json"),
        (source.parent.parent / "config.yaml", temporary / "config.yaml"),
    ):
        if src.is_file():
            shutil.copy2(src, dest)
    report = {
        "source": str(source),
        "kind": "model_only_warm_start",
        "changes": changed,
        "verified_tensors": len(state),
        "tensor_sha256": expected,
        "output_projection": "unchanged",
        "ema_present": any(k.startswith("net_ema.") for k in state),
        "verification": "all tensors reloaded and bytewise SHA256 matched",
        "config_note": "config.yaml is the original source configuration; enable State in the new training TOML",
    }
    (temporary / "initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    temporary.rename(output)
    print(f"PASS: {output}", flush=True)


if __name__ == "__main__":
    main()
