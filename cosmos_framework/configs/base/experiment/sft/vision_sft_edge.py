# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``vision_sft_edge`` — Cosmos3-Edge vision SFT recipe.

Nemotron-2B-Dense-VL / Cosmos3-Edge backbone with the structured-TOML
Parquet dataloader for T2V/TI2V/V2V SFT. EMA enabled.

Sibling of ``vision_sft_nano``: same Hydra defaults, same Parquet dataloader wiring, same
optimizer / scheduler / trainer / checkpoint blocks. The only differences are the
model config (``EDGE_MODEL_CONFIG`` — dense Nemotron backbone, no audio) and the
recipe name. Edge has no sound tokenizer, matching nano's
``{"override /sound_tokenizer": None}`` default.

Notes:
    * ``_target_`` references for ``model``, ``optimizer``, ``scheduler``,
      ``checkpoint``, ``callbacks``, ``ema``, ``tokenizer``, ``cluster``,
      ``vlm_config``, and ``ckpt_type`` flow from the ``defaults:`` group
      choices, matching the YAML's ``defaults:`` list (``_self_`` is placed
      LAST per prerelease convention so the experiment overrides the
      defaults, but no setting changes semantically).
    * The YAML's giant ``model_parallel`` block, ``trainer.profiling``,
      ``trainer.straggler_detection`` and ``trainer.type`` are populated by
      the base ``Config`` (``cosmos_framework/configs/base/config.py``) defaults and
      are therefore omitted here.

``checkpoint.load_path`` is left as ``???`` (a Hydra MISSING marker); supply
via CLI / a downstream experiment that inherits from this one.

Usage::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 \\
        --master_port=12341 -m cosmos_framework.scripts.train \\
        --config=cosmos_framework/configs/base/config.py -- \\
        experiment=vision_sft_edge \\
        checkpoint.load_path=<path>
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.defaults.reasoner import Qwen3VLMoT_VLM_4b_Instruct_Config
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

# Vision SFT trains no action tokens, so drop the action head (recipe semantics).
# Historical: pre-renewal (< 2026-07-16) Edge checkpoints shipped no action weights,
# so action_gen=True NaN'd (uninitialized frozen head made the graph-consistency
# dummy forward's 0.0 * preds_action NaN).
_EDGE_VISION_MODEL_CONFIG = copy.deepcopy(EDGE_MODEL_CONFIG)
_EDGE_VISION_MODEL_CONFIG["action_gen"] = False


vision_sft_edge = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            # YAML used `scheduler: warmup_cosine_lr` but that group is only
            # registered in cosmos_framework/configs/base/reasoner/defaults/optimizer.py
            # (reachable from the vlm config tree). The base vfm config path
            # only knows `lambdacosine`, which also sets
            # lr_scheduler_type="LambdaCosine" — behaviorally identical.
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                    "generation",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="sft",
            name="vision_sft_edge",
            wandb_mode="disabled",
        ),
        model=dict(
            config=copy.deepcopy(_EDGE_VISION_MODEL_CONFIG),
        ),
        optimizer=dict(
            betas=[0.9, 0.95],
            eps=1.0e-06,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "k_norm_und_for_gen",  # und-K norm trains with the gen pathway
            ],
            lr=5.0e-04,
            lr_multipliers={},
            optimizer_type="AdamW",
            weight_decay=0,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaCosine",
            cycle_lengths=[1000],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[50],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=2,
            logging_iter=1,
            max_iter=500,
            max_val_iter=None,
            # YAML had `memory_format: preserve_format` as a string, but the
            # prerelease trainer passes this verbatim to model.to(memory_format=…)
            # which requires a torch.memory_format enum (not a string).
            # Omit and let the framework default apply, matching how
            # mixed_modality_sft_nano.py handles it.
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(
                    compile_after_iterations=3,
                    enabled=False,
                    warmup_resolutions=None,
                ),
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200,
                    log_memory_detail=True,
                    save_s3=False,
                    step_size=1,
                    upload_every_n_mul=5,
                ),
                expert_heatmap=dict(every_n=1000),
                grad_clip=dict(clip_norm=0.1, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                norm_monitor=dict(
                    every_n=100,
                    layer_norm_only=False,
                    log_stat_wandb=True,
                    model_key=None,
                    save_s3=False,
                    step_size=1,
                    track_activations=True,
                ),
                param_count=dict(save_s3=False),
                sequence_packing_padding=dict(every_n=50),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                wandb_2x=dict(
                    logging_iter_multipler=2,
                    save_logging_iter_multipler=1,
                    save_s3=False,
                ),
                wandb_val=dict(save_s3=False),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_path="???",  # supply via CLI / downstream experiment
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=True,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(
                    bucket="",
                    credentials="",
                    enabled=False,
                ),
            ),
            jit=dict(
                device="cuda",
                dtype="bfloat16",
                enabled=False,
                input_shape=None,
                strict=True,
            ),
            load_from_object_store=dict(
                bucket="",
                credentials="",
                enabled=False,
            ),
            save_to_object_store=dict(
                bucket="",
                credentials="",
                enabled=False,
            ),
        ),
        dataloader_train=dict(num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=4),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# Pure Understanding uses the same Omni/MoT model and packed catalog dataflow
# as generation.  It keeps Edge's native visual tower but skips the generation
# VAE and diffusion/action heads.
understanding_sft_edge = copy.deepcopy(vision_sft_edge)
understanding_sft_edge.job.group = "understanding_sft"
understanding_sft_edge.job.name = "understanding_sft_edge"
understanding_sft_edge.model.config.load_vlm_visual = True
understanding_sft_edge.model.config.load_vision_tokenizer = False
understanding_sft_edge.model.config.vision_gen = False
understanding_sft_edge.model.config.action_gen = False
understanding_sft_edge.model.config.predict_text_tokens = True
understanding_sft_edge.model.config.ema.enabled = False
understanding_sft_edge.optimizer.keys_to_select = []
understanding_sft_edge.optimizer.lr = 1.0e-6
understanding_sft_edge.optimizer.weight_decay = 0.1


# GEN-driven UND unfreezing keeps the official Cosmos GEN data path and loss;
# only optimizer selection changes which shared reasoner parameters are updated.
vision_sft_edge_gen_unfreeze = copy.deepcopy(vision_sft_edge)
vision_sft_edge_gen_unfreeze.job.name = "vision_sft_edge_gen_unfreeze"
vision_sft_edge_gen_unfreeze.optimizer.keys_to_select = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "k_norm_und_for_gen",
    "language_model.model.layers",
    "language_model.model.embed_tokens",
    "language_model.model.norm",
    "language_model.lm_head",
]

# Qwen 4+4 is the same full Omni/Joint-MoT class as Nano with a different
# native Qwen config/checkpoint identity. Keep the Understanding pathway
# frozen: this recipe trains only the generation expert and its generation-
# specific timestep/latent adapters, matching the pure-GEN Edge recipe above.
vision_sft_qwen4b_gen = copy.deepcopy(vision_sft_edge)
vision_sft_qwen4b_gen.job.name = "vision_sft_qwen4b_gen"
_qwen4b_vlm = copy.deepcopy(Qwen3VLMoT_VLM_4b_Instruct_Config)
_qwen4b_vlm.tokenizer.tokenizer_type = "${oc.env:QWEN3_VL_4B_PATH}"
_qwen4b_vlm.tokenizer.config_variant = "hf"
_qwen4b_vlm.pretrained_weights.backbone_path = "${oc.env:QWEN3_VL_4B_PATH}"
_qwen4b_vlm.pretrained_weights.credentials_path = ""
vision_sft_qwen4b_gen.model.config.vlm_config = _qwen4b_vlm
vision_sft_qwen4b_gen.optimizer.keys_to_select = ["moe_gen", "time_embedder", "vae2llm", "llm2vae"]

# Qwen3-VL-4B can also start pure Understanding SFT directly from a local
# Hugging Face snapshot.  It uses the same slim UND-only model construction as
# Nano/Edge and therefore needs neither a Generation DCP nor a Wan VAE.
understanding_sft_qwen4b = copy.deepcopy(understanding_sft_edge)
understanding_sft_qwen4b.job.name = "understanding_sft_qwen4b"
understanding_sft_qwen4b.model.config.vlm_config = copy.deepcopy(_qwen4b_vlm)
understanding_sft_qwen4b.checkpoint.load_path = ""


for _item in [
    understanding_sft_edge,
    understanding_sft_qwen4b,
    vision_sft_edge,
    vision_sft_edge_gen_unfreeze,
    vision_sft_qwen4b_gen,
]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
