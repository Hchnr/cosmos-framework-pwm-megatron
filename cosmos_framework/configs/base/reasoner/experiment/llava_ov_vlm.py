# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VLM training on lmms-lab/LLaVA-OneVision-Data via CosmosDataLoader.

Self-contained — inlines the Phase 2 VLMModel/FSDP2 base (formerly the
``pre_exp012_000_phase2_vlm_smoke_4gpu_8b`` smoke recipe in
``pre_exp012_phase2_vlm_smoke.py``) and replaces the dataloader with the
OSS-facing CosmosDataLoader + four-role dataflow pattern. Hydra defaults
below pin the VLM model (``vlm_fsdp`` / ``qwen3_vl_8b_instruct``), the
checkpoint backend, and callbacks.

The dataset is loaded in streaming mode from the HuggingFace Hub so no local
download is required.  Each record is converted from ShareGPT conversation
format to the OpenAI message format expected by Qwen3-VL's processor, then
tokenized in the DataLoader worker via ``processor.apply_chat_template``.

Resume semantics
----------------
The streaming HF dataset is a ``datasets.IterableDataset``, which
CosmosDataLoader's IterableDistributor flags with no stateful resume
(streaming has no meaningful position to record). On checkpoint save the
dataloader shard stores placeholder ``(epoch=0, index=0)`` per worker.
When resuming with ``checkpoint.load_training_state=true``:

  - model / optim / scheduler / trainer state restore correctly (iter
    counter, optimizer momentum, LR schedule position all continue).
  - dataloader stream position does NOT restore; the streamed dataset
    re-yields from the beginning, so the first N resumed iters see the
    same samples as the first N iters of the original run.

For a true position-stateful resume, swap the data_source to a map-style
dataset (``load_dataset(..., streaming=False)``).

Usage (smoke test)::

    torchrun --nproc_per_node=4 --master_port=12344 -m cosmos_framework.scripts.train \\
        --config=cosmos_framework/configs/base/reasoner/config.py -- \\
        experiment=pre_exp012_llava_ov \\
        "model.config.policy.backbone.model_name=/path/to/Siglip2-Qwen3-1.7B-BF16-Alignment" \\
        trainer.max_iter=10 trainer.logging_iter=1 \\
        job.wandb_mode=disabled ckpt_type=dummy

See ``launch_vlm_llava_ov.sh`` for a ready-to-run shell script.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


# ---------------------------------------------------------------------------
# LLaVA-OneVision-Data source factory
#
# Loads lmms-lab/LLaVA-OneVision-Data in streaming mode so no local download
# is needed.  streaming=True returns an IterableDataset which CosmosDataLoader
# wraps directly via IterableDistributor.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Map-style data source factory (for the resumable variant below)
#
# Loads a subset as a real on-disk ``datasets.Dataset`` (streaming=False —
# random-access), filters it, and caps it to ``n`` rows so ``MapDistributor``
# can checkpoint exact ``(epoch, index)`` positions per worker.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Experiment registration
# ---------------------------------------------------------------------------


pre_exp012_llava_ov = LazyDict(
    dict(
        # Hydra defaults — inlined from the former pre_exp012_000_phase2_vlm_smoke_4gpu_8b
        # smoke recipe. data_train/data_val intentionally omitted because the
        # dataloader_train below is a self-contained CosmosDataLoader; pulling in
        # the smoke's s3 webdataset defaults would let storage_type schema bleed into
        # our CosmosDataLoader config.
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "basic_log"]},
            "_self_",
        ],
        job=dict(
            name="pre_exp012_llava_ov_${now:%Y-%m-%d}_${now:%H-%M-%S}",
            group="vlm_llava_ov_demo",
            wandb_mode="disabled",
        ),
        trainer=dict(
            max_iter=10,
            logging_iter=1,
            run_validation=False,
        ),
        optimizer=dict(
            lr=1e-5,
            fused=True,
        ),
        model=dict(
            config=dict(
                # Phase 2 requires a trainable_params regex; ".*" = full fine-tune.
                freeze=dict(
                    trainable_params=[".*"],
                ),
                parallelism=dict(
                    data_parallel_shard_degree=4,
                    data_parallel_replicate_degree=-1,
                ),
            ),
        ),
        # Local-only mode: disable the parent's object-store IO and clear the
        # S3 credentials/bucket so maybe_download_hf_model_from_s3 falls back
        # to HuggingFace Hub (avoids opening credentials/s3_training.secret in
        # OSS smoke runs). Pattern mirrors vision_sft_nano.py.
        checkpoint=dict(
            # Don't save checkpoints during smoke runs.
            save_iter=100000,
            load_from_object_store=dict(enabled=False, credentials="", bucket=""),
            save_to_object_store=dict(enabled=False, credentials="", bucket=""),
        ),
        dataloader_train=dict(num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=2),
        dataloader_val=None,
        # Suppress S3 uploads in callbacks (iter_speed.save_s3, param_count.save_s3,
        # wandb_*.save_s3 all interpolate from ${upload_reproducible_setup}). Mirrors
        # the VFM SFT experiments under cosmos/configs/base/experiment/sft/.
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

cs.store(
    group="experiment",
    package="_global_",
    name="pre_exp012_llava_ov",
    node=pre_exp012_llava_ov,
)
