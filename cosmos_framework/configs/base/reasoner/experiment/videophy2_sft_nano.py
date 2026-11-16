# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VideoPhy-2 understanding SFT recipe using the TOML Parquet catalog."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


videophy2_sft_nano = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "local"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "basic_log", "hf_export"]},
            "_self_",
        ],
        job=dict(
            project="cosmos3_reasoner",
            group="sft",
            wandb_mode="disabled",
        ),
        trainer=dict(
            callbacks=dict(
                log_tensor_shape=dict(num_log=2),
            ),
            max_iter=50,
            logging_iter=1,
            run_validation=True,
            validation_iter=10,
            max_val_iter=50,
            grad_accum_iter=8,
        ),
        optimizer=dict(
            # Uniform 1e-6 across both tiers. Nano is stable at 1e-5 too, but the 32B
            # super run spikes transiently (loss->6.6 @ iter 23) at 1e-5 before
            # recovering to the same endpoint; 1e-6 reaches the identical final loss
            # with no instability, so both tiers ship at 1e-6. (Super deepcopies this
            # block, so setting it here covers both.)
            lr=1e-6,
            fused=True,
            weight_decay=0.05,
            betas=[0.9, 0.999],
            # NOTE: the ViT is FROZEN (freeze_vision_encoder=True below), so the only
            # trainable "model.visual.*" params are the projector (model.visual.merger
            # + deepstack_merger_list). This key therefore sets the PROJECTOR LR only,
            # to 1.0x optimizer.lr -- giving a uniform LR across projector + LLM +
            # lm_head. It is NOT touching the frozen ViT backbone.
            #
            # Why the key is "model.visual" (broad) and not "model.visual.merger":
            # the reasoner default optimizer (configs/base/reasoner/defaults/optimizer.py)
            # ships lr_multipliers={"model.visual": 0.1}. Hydra merges dicts base-first
            # (so that key stays PINNED FIRST) and _build_params_with_metadata
            # (utils/generator/optimizer.py) is first-substring-match-wins -- so a
            # narrower "model.visual.merger" key is shadowed by "model.visual" and never
            # matches (verified: the merger still resolves to 0.1x). Overriding the SAME
            # "model.visual" key to 1.0 is the only way to cancel the 0.1x default; with
            # the ViT frozen that reaches only the merger. (To give an UNFROZEN ViT its
            # own LR you'd likewise override this "model.visual" value -- a narrower key
            # can't out-prioritize it.)
            lr_multipliers={"model.visual": 1.0},
        ),
        scheduler=dict(
            warm_up_steps=[5],
            cycle_lengths=[50],
            f_min=[0.1],
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=1,
            distributor_seed=1993,
        ),
        model=dict(
            config=dict(
                freeze=dict(
                    freeze_vision_encoder=True,
                    freeze_mm_projector=False,
                ),
                parallelism=dict(
                    data_parallel_shard_degree=8,
                    data_parallel_replicate_degree=-1,
                ),
                policy=dict(
                    monkey_patch_for_text_only_data=True,
                ),
            ),
        ),
        # hf_export so eval_videophy2 can read each save as HF safetensors.
        checkpoint=dict(
            save_iter=100,
            hf_export=dict(enabled=True),
        ),
        upload_reproducible_setup=False,
        dataloader_train=dict(num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=2),
        dataloader_val=None,
    ),
    flags={"allow_objects": True},
)


for _item in [videophy2_sft_nano]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    if "job" not in _item:
        _item["job"] = dict(name=experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}")
    else:
        _item["job"]["name"] = experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    cs.store(group="experiment", package="_global_", name=experiment_name, node=_item)
