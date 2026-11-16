"""Sample processing remains part of Action128 append-only source validation."""

import hashlib

import pytest

from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
from cosmos_framework.data.generator.catalog_processors import build_catalog_joint_dataloader
from cosmos_framework.data.generator.dataflow.batchers import SimpleBatcher
from cosmos_framework.data.generator.dataflow.collators import DefaultBatchCollator
from cosmos_framework.data.test.action128_test import source as source


@pytest.fixture(params=["legacy_per_source", "shared_prefetch"])
def make_loader(request):
    loaders = []

    def build(catalog, *, jitter, seed=42, appended=False):
        loader = build_catalog_joint_dataloader(
            [{"path": str(catalog), "ratio": 1}] * (2 if appended else 1),
            components={"action": {"batcher": SimpleBatcher(1), "collator": DefaultBatchCollator()}},
            first_frame_color_jitter=jitter,
            seed=seed,
            num_workers=0,
            execution_backend=request.param,
        )
        loaders.append(loader)
        return loader

    yield build
    for loader in loaders:
        if hasattr(loader, "close"):
            loader.close()


@pytest.mark.parametrize("jitter", [False, True])
def test_extend_retains_processing_fingerprint_for_unchanged_old_sources(source, make_loader, jitter):
    _, catalog, _ = source
    original = make_loader(catalog, jitter=jitter)
    distributor = original._loaders[0].dataset._distributor
    dataset = distributor._dataset
    expected = dataset.fingerprint
    if jitter:
        expected = hashlib.sha256(f"{expected}:first-frame-color-jitter-v1:42".encode()).hexdigest()
    assert original.dataset_fingerprints == {"robot:0": expected}
    saved = JointCosmosDataLoaderStateCallback(original).state_dict()

    extended = make_loader(catalog, jitter=jitter, appended=True)
    assert extended.dataset_fingerprints_for_action128(dataset.action128_spec)["robot:0"] == expected
    callback = JointCosmosDataLoaderStateCallback(extended)
    assert callback.validate_action128_extension(saved, dataset.action128_spec)


@pytest.mark.parametrize(
    ("old_jitter", "new_jitter", "new_seed"),
    [(True, False, 42), (False, True, 42), (True, True, 43)],
)
def test_extend_rejects_changed_processing_configuration(source, make_loader, old_jitter, new_jitter, new_seed):
    _, catalog, _ = source
    original = make_loader(catalog, jitter=old_jitter)
    dataset = original._loaders[0].dataset._distributor._dataset
    saved = JointCosmosDataLoaderStateCallback(original).state_dict()
    changed = make_loader(catalog, jitter=new_jitter, seed=new_seed, appended=True)
    with pytest.raises(ValueError, match="every old dataset unchanged"):
        JointCosmosDataLoaderStateCallback(changed).validate_action128_extension(saved, dataset.action128_spec)
