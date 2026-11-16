"""Read real pretraining samples and traverse the main shared-worker loader."""

import argparse

import torch

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.drop_sample_contract import DROP_SAMPLE_KEY, DROP_SAMPLE_REASON_KEY
from cosmos_framework.utils.lazy_config import instantiate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/toml/sft_config/action128_pretrain_state_h1_smoke.toml")
    parser.add_argument(
        "--batches", type=int, default=None, help="Shared-worker batches; zero checks each source only."
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_experiment_from_toml(args.config)
    assert config.dataloader_train.action_use_state is True
    config.dataloader_train.pin_memory = False
    loader = instantiate(config.dataloader_train)
    try:
        for stream in loader._loaders:
            dataset = stream.dataset._distributor._dataset
            for index in range(min(100, len(dataset))):
                item = dataset[index]
                if not item.get(DROP_SAMPLE_KEY):
                    break
                print("DROP", dataset.root, index, item.get(DROP_SAMPLE_REASON_KEY), flush=True)
            else:
                raise AssertionError(f"No admitted sample among first 100 windows: {dataset.root}")
            state_indexes = item["sequence_plan"].robot_state_frame_indexes
            assert state_indexes in ([], [0])
            assert item["sequence_plan"].condition_frame_indexes_action == state_indexes
            if state_indexes:
                assert not item["action_loss_mask"][0].any()
                action_head = dataset.registry["domain_projection_ids"][item["normalization_domain_id"]]
                state_head = dataset.registry["domain_state_projection_ids"][item["normalization_domain_id"]]
                expected = [state_head] + [action_head] * (len(item["action"]) - 1)
                assert item["domain_id"].tolist() == expected
            assert torch.isfinite(item["action"]).all()
            augmented = stream.dataset._processor.process(item)
            assert augmented["action"] is item["action"]
            assert torch.equal(augmented["video"][:, 1:], item["video"][:, 1:])
            assert not torch.equal(augmented["video"][:, 0], item["video"][:, 0])
            print(
                "SAMPLE",
                dataset.root,
                item["action"].shape,
                "state_dims",
                int(item["action_dim_mask"][0].sum()) if state_indexes else 0,
                "times",
                item["action_timestamps"][:3].tolist(),
                "projection_ids",
                item["domain_id"][:3].tolist(),
                flush=True,
            )
            dataset.decoder.close()
        batch_count = len(loader._loaders) * 2 if args.batches is None else args.batches
        iterator = iter(loader) if batch_count else None
        for _ in range(batch_count):
            batch = next(iterator)
            packed = batch["packed_sequence"]
            action = packed.action
            state = action.robot_state_token_indexes
            assert not torch.cat(action.loss_valid_mask)[state].any()
            assert torch.cat(action.condition_mask).flatten()[state].bool().all()
            assert not torch.isin(action.sequence_indexes[state], action.mse_loss_indexes).any()
            print(
                "BATCH",
                batch.get("dataset_name"),
                "states",
                state.tolist(),
                "action_shapes",
                action.token_shapes,
                flush=True,
            )
    finally:
        loader.close()
    print("PASS real EEF/Joint + shared workers + packing + clean/loss masks", flush=True)


if __name__ == "__main__":
    main()
