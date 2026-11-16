"""Diagnostic CPU-batch replay, explicitly not a production training loader.

Capture one real catalog pack per rank without worker processes, then replay it.
Host-to-device transfer, online VAE, forward/backward and optimizer work remain.
Only catalog access, decode, transforms and background workers are removed.
Fixtures are trusted local torch.save artifacts; do not load downloaded files.
"""

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data._utils.pin_memory import pin_memory


class ActionInputReplay:
    def __init__(self, batch):
        self.batch = batch
        # The trainer transfers tensors to CUDA before model code sees them.
        # Share immutable CPU tensors, but isolate all mutable batch metadata.
        self.tensor_memo = {}

        def visit(value):
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise ValueError("Replay fixtures must contain CPU tensors")
                self.tensor_memo[id(value)] = value
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)

        visit(batch)

    def set_start_iteration(self, iteration):
        if iteration != 0:
            raise ValueError("Diagnostic replay does not support training resume")

    def __iter__(self):
        while True:
            yield copy.deepcopy(self.batch, self.tensor_memo.copy())


def build_action_input_replay(*, replay_root: str, **kwargs):
    from cosmos_framework.data.generator.catalog_processors import build_catalog_training_dataloader
    from cosmos_framework.utils import misc

    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    root = Path(replay_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"rank_{rank}.pt"
    keys = (
        "dataset_specs",
        "max_samples_per_batch",
        "max_sequence_length",
        "fixed_und_tokens",
        "fixed_gen_tokens",
        "static_packing_config",
        "seed",
    )
    signature = json.loads(json.dumps({k: kwargs.get(k) for k in keys}, default=str))
    signature["world_size"] = world
    signature["rank"] = rank
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError(f"Replay fixture configuration mismatch: {path}")
        batch = saved["batch"]
    else:
        # Capture with no worker subprocesses, so none can contend with training.
        capture_kwargs = {**kwargs, "num_workers": 0, "prefetch_factor": None, "pin_memory": False}
        loader = build_catalog_training_dataloader(**capture_kwargs)
        iterator = iter(loader)
        batch = next(iterator)
        iterator.close()
        del iterator, loader
        with path.open("xb") as output:
            torch.save({"signature": signature, "batch": batch}, output)
    if len(batch["sequence_plan"]) != kwargs["max_samples_per_batch"]:
        raise ValueError("Replay fixture has an unexpected sample count")
    hasher = hashlib.sha256()
    with path.open("rb") as fixture:
        for chunk in iter(lambda: fixture.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    (root / f"rank_{rank}.json").write_text(
        json.dumps({"signature": signature, "sha256": digest, "diagnostic_only": True}, indent=2) + "\n"
    )
    if kwargs.get("pin_memory", False):
        batch = pin_memory(batch)
    # Capture and cache-hit paths must leave the same RNG state for AC comparisons.
    misc.set_random_seed(seed=kwargs.get("seed", 42), by_rank=True)
    return ActionInputReplay(batch)


def configure_replay(config, replay_root):
    """Replace the validated catalog factory before config freezing/serialization."""
    from omegaconf import open_dict

    with open_dict(config.dataloader_train):
        config.dataloader_train._target_ = "tools.action_input_replay.build_action_input_replay"
        config.dataloader_train.replay_root = str(Path(replay_root).resolve())
    # Replayed batches are not a resumable catalog cursor.
    with open_dict(config.trainer.callbacks):
        config.trainer.callbacks.pop("dataloader_state", None)


def main():
    import argparse
    import sys

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.scripts.train import launch
    from cosmos_framework.utils.lazy_config import LazyConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-toml", required=True)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    replay_root = os.environ.get("ACTION_REPLAY_ROOT")
    if not replay_root:
        parser.error("ACTION_REPLAY_ROOT must identify a trusted local fixture directory")
    config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)
    configure_replay(config, replay_root)
    args.config = args.sft_toml
    args.deterministic = False
    args.attach_vscode_debugger = False
    if args.dryrun:
        Path(config.job.path_local).mkdir(parents=True, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        return
    launch(config, args)
    if os.environ.get("COSMOS_EXIT_WITHOUT_FINALIZE", "").lower() in ("1", "true"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
