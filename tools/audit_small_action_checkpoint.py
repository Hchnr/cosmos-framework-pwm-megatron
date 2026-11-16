"""CPU-only audit of saved pilot checkpoint structure and sampled weight updates."""

import argparse
import collections
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.model.generator.mot.small_action_tower import action_shape


def metadata(path):
    return dcp.FileSystemReader(path).read_metadata().state_dict_metadata


def read_tensors(path, keys, info):
    state = {key: torch.empty(tuple(info[key].size), dtype=info[key].properties.dtype) for key in keys}
    dcp.load(state, checkpoint_id=path)
    return state


def check_storage(path):
    info = dcp.FileSystemReader(path).read_metadata()
    required_sizes = collections.defaultdict(int)
    for location in info.storage_data.values():
        required_sizes[location.relative_path] = max(
            required_sizes[location.relative_path], location.offset + location.length
        )
    assert required_sizes, (path, "Empty checkpoint storage")
    for name, size in required_sizes.items():
        assert (path / name).stat().st_size >= size, (path / name, "Truncated checkpoint shard")
    return len(required_sizes)


def audit(root, initial_root, preset, step=1000):
    run = root / "train/cosmos3_action/action_tower_v022_libero_1b" / f"libero_bowl_plate_{preset}"
    checkpoints = run / "checkpoints"
    expected_steps = list(range(200, step + 1, 200))
    saved_steps = sorted(int(p.name.removeprefix("iter_")) for p in checkpoints.glob("iter_*"))
    assert all(saved_step in range(200, 1001, 200) for saved_step in saved_steps), saved_steps
    assert set(expected_steps).issubset(saved_steps), (expected_steps, saved_steps)
    final = checkpoints / f"iter_{step:09d}"
    shard_counts = {part: check_storage(final / part) for part in ("model", "optim", "scheduler", "trainer")}
    trainer_state = {"iteration": 0}
    dcp.load(trainer_state, checkpoint_id=final / "trainer")
    assert trainer_state["iteration"] == step, trainer_state
    source = initial_root / preset / "model"
    before, after = metadata(source), metadata(final / "model")
    # Action-only construction omits the unused audio heads present in Nano.
    unused_audio = {
        "net.sound2llm.bias",
        "net.sound2llm.weight",
        "net.llm2sound.weight",
        "net.llm2sound.bias",
        "net.sound_modality_embed",
    }
    omitted = before.keys() - after.keys()
    assert omitted <= unused_audio and not (after.keys() - before.keys()), (omitted, after.keys() - before.keys())
    for key, item in after.items():
        assert isinstance(item, dcp.TensorStorageMetadata), key
        assert item.size == before[key].size, key
        assert item.properties.dtype == torch.bfloat16, (key, item.properties.dtype)
    action_count = sum(
        math.prod(item.size)
        for key, item in after.items()
        if any(
            x in key
            for x in ("moe_gen_action", "action_time_embedder", "action2llm", "llm2action", "action_modality_embed")
        )
    )
    assert 900_000_000 <= action_count <= 1_100_000_000
    shape = action_shape(preset)
    frozen, updated = [], []
    for layer in (0, 18, 35):
        prefix = f"net.language_model.model.layers.{layer}."
        frozen.extend(prefix + x for x in ("self_attn.q_proj.weight", "mlp.gate_proj.weight"))
        updated.extend(prefix + x for x in ("self_attn.q_proj_moe_gen.weight", "mlp_moe_gen.gate_proj.weight"))
    for layer in (shape.layers[0], shape.layers[-1]):
        prefix = f"net.language_model.model.layers.{layer}.action_block_moe_gen_action."
        updated.extend(prefix + x for x in ("q_proj.weight", "gate_proj.weight"))
    frozen.append("net.language_model.model.embed_tokens.weight")
    updated.extend(("net.action2llm.fc.weight", "net.llm2action.fc.weight"))
    adapters = [key for key in after if ".adapters." in key]
    keys = frozen + updated + adapters
    a, b = read_tensors(source, keys, before), read_tensors(final / "model", keys, after)
    changes = {}
    for key in keys:
        assert torch.isfinite(b[key]).all(), key
        count = int(torch.count_nonzero(a[key] != b[key]))
        if key in frozen:
            assert count == 0, ("Frozen UND changed", key)
        elif key in updated:
            assert count > 0, ("Trainable sample unchanged", key)
        changes[key] = dict(changed_elements=count, total_elements=b[key].numel())
    if adapters:
        for direction in ("und_to_action", "video_to_action", "action_to_video"):
            assert any(changes[key]["changed_elements"] for key in adapters if direction in key), direction
    optim = metadata(final / "optim")
    optim_dtypes = collections.Counter(
        str(item.properties.dtype) for item in optim.values() if isinstance(item, dcp.TensorStorageMetadata)
    )
    return dict(
        checkpoint=str(final),
        model_tensor_count=len(after),
        omitted_unused_audio_keys=sorted(omitted),
        model_dtype="bfloat16",
        action_parameters_including_adapters=action_count,
        adapter_parameters=sum(math.prod(after[key].size) for key in adapters),
        audited_step=step,
        saved_steps=saved_steps,
        checkpoint_save_interval=200,
        checkpoint_shard_counts=shard_counts,
        trainer_iteration=trainer_state["iteration"],
        sampled_und_unchanged=True,
        sampled_video_and_action_updated=True,
        adapter_directions_updated=bool(adapters),
        optimizer_tensor_dtypes=dict(optim_dtypes),
        sampled_weight_changes=changes,
        limitation="Weight update/finiteness checks sample backbone and action weights; all adapter matrices are checked.",
    )


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/action_tower_1b"))
    parser.add_argument("--step", type=int, choices=range(200, 1001, 200), default=1000)
    parser.add_argument(
        "--initial-root", type=Path, default=Path("/share/project/eai_pwm/models/Cosmos3-Nano-action-1b-v022")
    )
    parser.add_argument(
        "--presets", nargs="+", default=["depth_1b", "width_heads_1b", "width_head_dim_1b", "width_only_1b"]
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    for preset in args.presets:
        report = audit(args.root, args.initial_root, preset, args.step)
        suffix = "" if args.step == 1000 else f"_step{args.step}"
        path = args.root / f"checkpoint_audit_{preset}{suffix}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(preset, "PASS", path, flush=True)


if __name__ == "__main__":
    main()
