# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""TOML-readable pydantic schema for SFT experiment knobs.

Sibling of ``toml_config_helper.py`` — this file holds the pydantic models
(the schema the TOML must conform to). All conversion logic (TOML →
override list, ``PATH_REMAPS``, etc.) lives in ``toml_config_helper.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import tomllib
from omegaconf import OmegaConf, open_dict
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_framework.configs.toml_config.toml_config_helper import (
    TASK_TO_BASE_CONFIG,
    build_hydra_overrides,
)

# Common config for every model in this file:
# - ``extra="forbid"``        → unknown TOML keys raise ValidationError (typo guard).
# - ``protected_namespaces=()`` → silence the ``model_*`` field-name warning so
#                                 the ``model:`` field on SFTExperimentConfig is allowed.
_PYDANTIC_MODEL_CONFIG = ConfigDict(extra="forbid", protected_namespaces=())


# ---------------------------------------------------------------- job
class JobConfig(BaseModel):
    """Run identity + meta-fields that pick the Hydra config tree to load."""

    model_config = _PYDANTIC_MODEL_CONFIG

    task: str = Field(
        default="vfm",
        description=(
            "META — chooses which make_config() to call: "
            "'vfm' → cosmos_framework/configs/base/config.py (video foundation model), "
            "'vlm' → cosmos_framework/configs/base/reasoner/config.py (vision-language model). "
            "Also picks the path-remap rules in toml_config_helper.PATH_REMAPS."
        ),
    )
    experiment: str = Field(
        default="",
        description=(
            "META — names the Hydra experiment LazyDict registered in "
            "ConfigStore under experiment/<name>. Resolved at load time via "
            "the 'experiment=<name>' Hydra-CLI override "
            "(e.g. 'vision_sft_nano')."
        ),
    )
    project: str = Field(
        default="",
        description=(
            "Wandb project (team-level bucket). Flows to config.job.project "
            "and is what shows up under 'Projects' in the wandb UI."
        ),
    )
    group: str = Field(
        default="",
        description=(
            "Wandb group — sub-label under <project> for clustering related "
            "runs (e.g. 'sft', 'action_bridge'). Flows to config.job.group."
        ),
    )
    name: str = Field(
        default="",
        description=(
            "Wandb run name. Flows to config.job.name and forms part of the "
            "output-dir path: $IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>. "
            "Leave empty (or use Hydra ${now:%Y-%m-%d}_${now:%H-%M-%S}) to "
            "get an auto-timestamped subdir."
        ),
    )
    wandb_mode: str = Field(
        default="disabled",
        description=(
            "Wandb upload mode: 'online' (real-time, needs WANDB_API_KEY), "
            "'offline' (log locally, sync later with `wandb sync`), or "
            "'disabled' (no wandb at all)."
        ),
    )
    upload_reproducible_setup: bool = Field(
        default=False,
        description=(
            "Upload the reproducible-setup bundle and wandb save_s3 artifacts "
            "to S3. Defaults False (OSS: no S3 access). Set True only when S3 "
            "upload is configured. Remapped out of [job] to the top-level "
            "config.upload_reproducible_setup, overriding the base config value "
            "(the VLM base defaults it True). Always emitted by "
            "load_experiment_from_toml, so omitting it forces False."
        ),
    )


# ---------------------------------------------------------------- model
class EMAConfig(BaseModel):
    """Exponential Moving Average of the generation-pathway weights.

    Lands at ``model.config.ema.*`` on both VFM and VLM. When enabled the
    trainer keeps a second fp32 copy of the trainable params updated as
    ``ema_w = (1 - rate^k) · w_curr + rate^k · ema_w_prev``. EMA weights
    are used for inference; the live weights keep training.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    enabled: bool = Field(
        default=True,
        description=(
            "Turn EMA tracking on/off. Full fine-tunes typically enable it; "
            "LoRA recipes leave it off because the adapter weights are tiny."
        ),
    )
    rate: float = Field(
        default=0.1,
        description=(
            "Base EMA decay rate. Lower = slower decay = EMA tracks the live "
            "weights more tightly. Effective per-step rate is ramped by the "
            "iteration counter so the EMA 'warms up' from init."
        ),
    )
    iteration_shift: int = Field(
        default=0,
        description=(
            "Step offset added before computing the warmup ramp. Use a "
            "positive value when resuming so the EMA doesn't reset to "
            "'early-iter' decay strength."
        ),
    )


class ParallelismConfig(BaseModel):
    """FSDP / context-parallel / classifier-free-guidance topology.

    Lands at ``model.config.parallelism.*`` on both VFM and VLM.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    data_parallel_shard_degree: int = Field(
        default=-1,
        description=(
            "FSDP shard degree. -1 = auto-fit WORLD_SIZE from torchrun. "
            "Set explicitly when you want the run to fail loudly on the "
            "wrong GPU count."
        ),
    )
    data_parallel_replicate_degree: int = Field(
        default=1,
        description=(
            "FSDP replicate degree (HSDP). >1 adds an outer replicate loop "
            "so the same shard topology runs N times in parallel; usually "
            "only needed for very large clusters."
        ),
    )
    context_parallel_shard_degree: int = Field(
        default=1,
        description=(
            "Context-parallel shard degree. >1 splits the sequence dimension "
            "across this many ranks, which lets long-context models fit in "
            "memory. CP overlays the DP mesh and does not multiply world size."
        ),
    )
    cfg_parallel_shard_degree: int = Field(
        default=1,
        description=(
            "Classifier-free-guidance parallel shard degree. Splits the "
            "duplicated conditional/unconditional forward across ranks. "
            "Almost always 1 for SFT."
        ),
    )


class CompileConfig(BaseModel):
    """torch.compile knobs.

    Lands at ``model.config.compile.*`` on both VFM and VLM.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    enabled: bool = Field(
        default=False,
        description=(
            "torch.compile the network (was ``parallelism.use_torch_compile``). "
            "Big speedup on stable shapes; conflicts with some custom CUDA "
            "kernels and deterministic modes."
        ),
    )
    compile_dynamic: bool = Field(
        default=True,
        description=(
            "Generate a symbolic-shape graph that can reuse kernels across "
            "sequence lengths. VFM compile with this disabled requires "
            "model.fixed_sequence_length.enabled=true."
        ),
    )
    compiled_region: str = Field(
        default="language",
        pattern="^(language|all)$",
        description=(
            "VFM compile scope. 'language' compiles every unified MoT decoder "
            "layer; 'all' additionally compiles VFM encode/decode helpers."
        ),
    )
    mix_order_reduction_split_size: Literal[8, 16, 32, 64, 128] | None = Field(
        default=None,
        description=(
            "Override TorchInductor's mixed-order reduction split size. "
            "Use 32 when a long static-sequence RMSNorm backward kernel "
            "exceeds the GPU per-block shared-memory limit."
        ),
    )


class FixedSequenceLengthConfig(BaseModel):
    """Task-aware fixed physical token lengths for unified MoT decoder layers."""

    model_config = _PYDANTIC_MODEL_CONFIG

    enabled: bool = False
    understanding_tokens: int = Field(default=0, ge=0)
    und_tokens: int = Field(default=0, ge=0)
    gen_tokens: int = Field(default=0, ge=0)
    action_tokens: int = Field(
        default=0, ge=0, description="Small action tower physical rows reserved within gen_tokens."
    )


class ActivationCheckpointingConfig(BaseModel):
    """Recompute activations during backward to trade FLOPs for memory.

    Lands at ``model.config.activation_checkpointing.*`` on both VFM and VLM.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    mode: str = Field(
        default="full",
        description=(
            "AC mode: 'selective' (per-op SAC; save matmuls/FMHA, recompute "
            "the rest — MoT path only), 'full' (checkpoint each whole "
            "transformer block), or 'none' (no recomputation, higher memory). "
            "End-to-end throughput also depends on the sequence length/batch "
            "that fits in memory and the input pipeline."
        ),
    )
    save_ops_regex: list[str] = Field(
        default_factory=lambda: ["fmha"],
        description=(
            "Regex patterns for ops to KEEP saved when mode='selective'. "
            "Ignored in 'full'/'none' mode. Default keeps flash/multi-head-"
            "attention outputs."
        ),
    )
    preserve_rng_state: bool = Field(
        default=True,
        description=(
            "Stash and restore CUDA RNG across recompute boundaries. Required "
            "for deterministic results vs. non-checkpointed runs; small slowdown."
        ),
    )
    determinism_check: str = Field(
        default="default",
        description=(
            "Forwarded to torch.utils.checkpoint. 'default' disables the "
            "extra determinism check; 'match' cross-checks recomputed "
            "activations against the original (debug-only, very slow)."
        ),
    )


class ModelTokenizerConfig(BaseModel):
    """Video tokenizer (VAE) settings. VFM only — VLM skips this sub-tree."""

    model_config = _PYDANTIC_MODEL_CONFIG

    vae_path: str = Field(
        default="pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
        description=(
            "Path to Wan2.2_VAE.pth. SFT recipes typically pass this via "
            "env interpolation: vae_path = '${oc.env:WAN_VAE_PATH}'."
        ),
    )


class BackboneConfig(BaseModel):
    """Foundation backbone settings. VLM only — VFM keeps its backbone
    wiring inline in the experiment Python (vlm_config.model_instance)
    and skips this sub-tree.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    model_name: str = Field(
        default="???",
        description=(
            "HF repo ID or local snapshot path of the VLM backbone "
            "(e.g. 'Qwen/Qwen3-VL-8B-Instruct'). Drives AutoConfig + "
            "AutoModel selection (architecture). Remapped to "
            "'model.config.policy.backbone.model_name' on VLM; skipped on "
            "VFM. Default '???' is the OmegaConf MISSING sentinel — "
            "recognized by build_hydra_overrides and skipped, so the "
            "experiment Python's default takes effect when the TOML omits "
            "[model.backbone]."
        ),
    )
    safetensors_path: str = Field(
        default="???",
        description=(
            "Optional local path to a .safetensors file (or directory) used "
            "for weight loading. When set, overrides the auto-downloaded "
            "snapshot under model_name; the architecture is still driven by "
            "model_name. Useful for pointing at a converted/finetuned "
            "checkpoint while keeping the public HF model_name for tokenizer "
            "and architecture discovery. Remapped to "
            "'model.config.policy.backbone.safetensors_path' on VLM; "
            "skipped on VFM. Default '???' = MISSING sentinel (omitted from "
            "overrides; falls back to '' from VLMConfig, which means "
            "'use the auto-downloaded model_name snapshot')."
        ),
    )


class FP8TrainingConfig(BaseModel):
    """Training GEMMs only; disabled by default, independent of inference PTQ."""

    model_config = _PYDANTIC_MODEL_CONFIG

    enabled: bool = False
    backend: Literal["torchao"] = "torchao"
    recipe: Literal["rowwise", "rowwise_with_gw_hp"] = "rowwise"
    include_regex: list[str] = Field(
        default_factory=lambda: [r"^language_model\.model\.layers\.\d+\.mlp_moe_gen\.(gate_proj|up_proj|down_proj)$"]
    )
    exclude_regex: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection(self) -> "FP8TrainingConfig":
        from cosmos_framework.configs.base.defaults.fp8_training import validate_fp8_training_config

        validate_fp8_training_config(self)
        return self


class RectifiedFlowLossConfig(BaseModel):
    """Optional modality weights; omitted fields retain the experiment defaults."""

    model_config = _PYDANTIC_MODEL_CONFIG

    loss_scale: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    action_loss_weight: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ModelConfig(BaseModel):
    """Top-level model knobs.

    Lands at ``model.config.*`` on both VFM and VLM. After the ParallelismConfig
    split, the training-infra surface (parallelism, compile, activation_
    checkpointing, precision) lives at the same depth on both tasks; per-task
    leaf skips are handled by ``PATH_REMAPS``.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    precision: str = Field(
        default="bfloat16",
        description=(
            "Compute dtype for the network forward/backward "
            "(``MixedPrecisionPolicy.param_dtype``). Was "
            "``parallelism.precision`` before the ParallelismConfig split; "
            "lands at ``model.config.precision`` now."
        ),
    )
    resolution: Optional[str] = Field(
        default=None, description="VFM resolution/schedule preset. Omitted values retain the base experiment default."
    )
    vlm_tokenizer_path: Optional[str] = Field(
        default=None, description="VFM text tokenizer/processor path or HF identifier; does not select model weights."
    )
    max_num_tokens_after_packing: int = Field(
        default=13312,
        description=(
            "Token-packing target: max number of tokens after sequence "
            "packing. -1 disables the cap. VFM-only — VLM uses "
            "data_setting.max_tokens and policy.qwen_max_video_token_length."
        ),
    )
    gen_loss_scale: float | None = Field(
        default=None,
        gt=0,
        description="Optional scale applied to a generation/action microbatch loss.",
    )
    und_loss_scale: float | None = Field(
        default=None,
        gt=0,
        description="Optional scale applied to an Understanding microbatch loss.",
    )
    load_vlm_visual: bool = Field(
        default=True,
        description=(
            "Build and restore the VLM visual tower from the unified DCP. Set false for pure text-conditioned "
            "generation runs that never consume Understanding or user image/video inputs."
        ),
    )
    train_vlm_visual: bool = Field(
        default=False,
        description=(
            "Optimize the family-native VLM visual tower and include it in FSDP/activation checkpointing. "
            "Requires load_vlm_visual=true; existing unified-DCP recipes keep it frozen by default."
        ),
    )
    load_vision_tokenizer: bool = Field(
        default=True,
        description=(
            "Instantiate the generation VAE. Disable it for pure Understanding runs; "
            "Generation and Action runs require it."
        ),
    )
    vision_gen: bool = Field(
        default=True,
        description="Build the image/video generation pathway. Disable it for pure Understanding runs.",
    )
    action_gen: bool = Field(
        default=False,
        description="Build the action-generation pathway. Action training recipes enable it.",
    )
    text_loss_scale: float | None = Field(
        default=None,
        description=(
            "Deprecated input-only compatibility key. The loader accepts only "
            "numeric 0 and discards it; positive values are rejected."
        ),
    )
    vision_spatial_position_mode: Literal["canvas", "action128_seam"] = "canvas"
    tower_structure: Literal["two_tower", "three_tower"] = "two_tower"
    action_tower_preset: Literal["full", "depth_1b", "width_heads_1b", "width_head_dim_1b", "width_only_1b"] = "full"
    action_tower_init: Literal["checkpoint", "copy_gen"] = "checkpoint"
    action_tower_execution: Literal["native", "joint", "nano_compatible"] = "native"
    action_tower_attention_backend: Literal["auto", "nano", "flex", "sdpa", "flash3_video", "flash3"] = "auto"
    action_tower_und_short_tokens: int = Field(default=0, ge=0)
    action_tower_projection_mode: Literal["auto", "routed"] = "routed"
    joint_attn_backend: Literal[
        "split_default",
        "split_fa3_fork",
        "unified_fa3_grouped",
        "split_fa4",
        "unified_fa4_mask",
        "split_flex_triton",
        "unified_flex_triton",
    ] = "split_default"
    joint_attn_implementation: str = Field(
        default="two_way",
        description=(
            "VFM attention layout: 'two_way' (separate U/G blocks with "
            "cross-attention), 'three_way' (adds a sparsity-aware third "
            "block — NATTEN), or 'flex' (legacy). Used when "
            "[job].task='vfm'; skipped on VLM."
        ),
    )
    predict_text_tokens: bool = Field(
        default=False,
        description=(
            "VFM only. Enable the LM head for assistant text targets in T2I/T2V samples. "
            "Samples without assistant text carry an empty mask and skip CE."
        ),
    )
    text_loss_weight: float = Field(
        default=1.0,
        ge=0,
        description="VFM-only multiplier for assistant text cross-entropy loss.",
    )
    attn_implementation: str = Field(
        default="cosmos",
        description=(
            "VLM HF attention impl: 'cosmos' (cosmos NATTEN/Blackwell-FMHA "
            "wrapper), 'flash_attention_2' (HF flash-attn-2), 'sdpa' "
            "(torch SDPA), or 'eager' (pure-python fallback). Used when "
            "[job].task='vlm'; skipped on VFM."
        ),
    )
    lora_enabled: bool = Field(
        default=False,
        description=(
            "Inject LoRA adapters into the generation pathway BEFORE FSDP "
            "wraps the network. Pair with optimizer.keys_to_select=['lora_'] "
            "(train only adapters) and checkpoint.keys_to_skip_loading=["
            "..., 'lora_'] (don't load missing adapter tensors). Used by "
            "Nano/Edge recipes leave it off by default. Skipped on VLM."
        ),
    )
    lora_rank: int = Field(
        default=16,
        description=(
            "LoRA rank `r`. Adapter shape is (rank × hidden_dim) per target module. Standard values are 4, 8, 16, 32."
        ),
    )
    lora_alpha: int = Field(
        default=32,
        description=(
            "LoRA scaling factor. Effective magnitude of the adapter update "
            "is alpha/rank; rank=16 alpha=32 gives a 2× scale."
        ),
    )
    lora_target_modules: str = Field(
        default="q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen",
        description=(
            "Comma-separated substrings of param names that get a LoRA "
            "adapter. Defaults target the four MoE-gen projection matrices."
        ),
    )

    rectified_flow_training_config: RectifiedFlowLossConfig = Field(default_factory=RectifiedFlowLossConfig)
    ema: EMAConfig = Field(default_factory=EMAConfig)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    compile: CompileConfig = Field(default_factory=CompileConfig)
    fp8_training: FP8TrainingConfig = Field(default_factory=FP8TrainingConfig)
    fixed_sequence_length: FixedSequenceLengthConfig = Field(default_factory=FixedSequenceLengthConfig)
    activation_checkpointing: ActivationCheckpointingConfig = Field(default_factory=ActivationCheckpointingConfig)
    tokenizer: ModelTokenizerConfig = Field(default_factory=ModelTokenizerConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)


# ---------------------------------------------------------------- optimizer
class OptimizerConfig(BaseModel):
    """AdamW-family optimizer parameters. Same shape on VFM and VLM (eps skipped on VLM)."""

    model_config = _PYDANTIC_MODEL_CONFIG

    betas: list[float] = Field(
        default_factory=lambda: [0.9, 0.99],
        description=(
            "Adam β1, β2 — gradient and squared-gradient EMAs. Standard pair "
            "is (0.9, 0.999); SFT recipes commonly use (0.9, 0.99) or "
            "(0.9, 0.95) for tighter tracking of recent gradients."
        ),
    )
    eps: float = Field(
        default=1.0e-8,
        description=(
            "Adam numerical stability epsilon. 1e-8 is the PyTorch default; "
            "1e-6 is sometimes used in bf16 to avoid underflow in the "
            "squared-gradient denominator. Skipped on VLM (no eps field)."
        ),
    )
    fused: bool = Field(
        default=True,
        description=(
            "Use the fused AdamW kernel. Faster on modern GPUs; slightly "
            "different numerical behavior vs. the foreach implementation."
        ),
    )
    keys_to_select: list[str] = Field(
        default_factory=list,
        description=(
            "Substring allowlist for params that the optimizer trains. "
            "Empty list = train everything. ['lora_'] = LoRA-only fine-tune "
            "(freezes everything except adapters)."
        ),
    )
    lr: float = Field(
        default=2.0e-4,
        description="Base learning rate.",
    )
    lr_multipliers: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-param-group LR multipliers (substring → multiplier). Used "
            "by action recipes to e.g. give 'action_modality_embed' 5× the "
            "base lr. Substrings not in the dict default to 1.0."
        ),
    )
    weight_decay: float = Field(
        default=0.0,
        description="AdamW decoupled weight decay. 0 disables.",
    )


# ---------------------------------------------------------------- scheduler
class SchedulerConfig(BaseModel):
    """LambdaLinear / LambdaCosine LR scheduler knobs.

    All four ``f_*`` values are **ratios of the optimizer's base lr** —
    effective lr at the corresponding milestone = ``lr × f_x``. Each list
    has one entry per scheduler cycle.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    cycle_lengths: list[int] = Field(
        default_factory=lambda: [20000],
        description=(
            "Length of each cycle in optimizer steps. With one entry, the "
            "scheduler completes one full warmup→peak→trough cycle over "
            "that many iterations."
        ),
    )
    f_max: list[float] = Field(
        default_factory=lambda: [1.0],
        description="Peak LR multiplier reached at the end of warmup.",
    )
    f_min: list[float] = Field(
        default_factory=lambda: [0.0],
        description=(
            "Final LR multiplier at the end of each cycle (the 'floor'). "
            "For LambdaCosine the LR decays toward lr × f_min."
        ),
    )
    f_start: list[float] = Field(
        default_factory=lambda: [1.0e-6],
        description=("Initial LR multiplier at step 0, before warmup ramps up."),
    )
    verbosity_interval: int = Field(
        default=0,
        description=(
            "How often the scheduler logs the current LR (in optimizer steps). 0 = silent. VFM only — skipped on VLM."
        ),
    )
    warm_up_steps: list[int] = Field(
        default_factory=lambda: [100],
        description=(
            "Linear warmup duration in optimizer steps. LR ramps from "
            "lr × f_start to lr × f_max linearly over this many iters "
            "before the cosine/linear decay begins."
        ),
    )


# ---------------------------------------------------------------- trainer
class CompileTokenizerCallback(BaseModel):
    """Compile the VAE tokenizer once shapes stabilize.

    The concrete tokenizer selects its implementation: Wan uses chunk-level
    AOTInductor packages, while DCAE and UniAE use online ``torch.compile``.
    VFM only — skipped on VLM (no tokenizer to compile).
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    compile_after_iterations: int = Field(
        default=3,
        description=(
            "Wait this many training iterations after start before triggering "
            "the compile (lets one-shot init / dataloader settle)."
        ),
    )
    enabled: bool = Field(
        default=True,
        description="Master switch for the callback.",
    )
    warmup_resolutions: Optional[list[str]] = Field(
        default=None,
        description=(
            "Resolutions to 'prime' the compile cache with. The callback "
            "runs the tokenizer once per listed resolution so the compiled "
            "graph for each is ready before training hits it. None = use "
            "whatever resolutions the tokenizer's encode_chunk_frames knows."
        ),
    )
    aspect_ratio: Optional[str] = Field(
        default=None,
        description=(
            "Optional single aspect-ratio bucket to compile, such as '1,1'. "
            "None compiles every aspect ratio for each warmup resolution."
        ),
    )

    spatial_sizes: Optional[list[tuple[int, int]]] = Field(
        default=None,
        description="Explicit VAE pixel (H, W) canvases replacing standard resolution buckets.",
    )
    require_full_coverage: bool = Field(
        default=False,
        description="Fail if any requested VAE variant fails validation or runtime falls back to eager.",
    )


class GradClipCallback(BaseModel):
    """Gradient clipping callback. Present on both VFM and VLM."""

    model_config = _PYDANTIC_MODEL_CONFIG

    clip_norm: float = Field(
        default=1.0,
        description=(
            "Maximum global L2 norm of the gradient. Steps with a larger norm are rescaled so ||grad|| ≤ clip_norm."
        ),
    )
    force_finite: bool = Field(
        default=True,
        description=(
            "When True, replace NaN/Inf grads with zero before the step "
            "(treats them as no-op rather than crashing). VFM defaults to "
            "True; VLM defaults to False."
        ),
    )


class DrawSampleCallback(BaseModel):
    """Frequency and S3 upload controls for online generation previews."""

    model_config = _PYDANTIC_MODEL_CONFIG

    every_n: int = Field(default=5000, ge=0, description="Sampling interval; 0 disables training-step sampling.")
    run_at_start: bool = Field(default=False, description="Sample on the first training step when every_n is nonzero.")
    save_s3: bool = Field(default=True, description="Upload generated sample artifacts to S3.")


class TrainerCallbacksConfig(BaseModel):
    """Callbacks exposed by the TOML schema. The full
    callbacks dict (norm_monitor, mfu, heart_beat, …) stays in the
    experiment Python.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    compile_tokenizer: CompileTokenizerCallback = Field(default_factory=CompileTokenizerCallback)
    grad_clip: GradClipCallback = Field(default_factory=GradClipCallback)
    every_n_sample_reg: DrawSampleCallback = Field(default_factory=DrawSampleCallback)
    every_n_sample_ema: DrawSampleCallback = Field(default_factory=DrawSampleCallback)


class CompileCacheConfig(BaseModel):
    """Direct PyTorch compiler cache configuration."""

    model_config = _PYDANTIC_MODEL_CONFIG

    enabled: bool = Field(default=False, description="Enable direct PyTorch native cache reuse.")
    path: str = Field(
        default=".compile_cache",
        description="Cache directory; relative paths resolve from the repository root.",
    )


class TrainerCompileConfig(BaseModel):
    """Dynamo guard-cache limits, independent of the model's compile scope."""

    model_config = _PYDANTIC_MODEL_CONFIG

    recompile_limit: int = Field(default=8, ge=1)
    use_duck_shape: bool = True


class TrainerConfig(BaseModel):
    """Trainer-level knobs that the TOML drives directly."""

    model_config = _PYDANTIC_MODEL_CONFIG

    distributed_parallelism: str = Field(
        default="fsdp",
        description=(
            "Distributed strategy. 'fsdp' (the only supported value today) routes through cosmos's FSDP wrapper."
        ),
    )
    grad_accum_iter: int = Field(
        default=1,
        description=(
            "Number of micro-batches accumulated before each "
            "optimizer.step(). Effective global batch = grad_accum_iter × "
            "per-rank batch × world_size."
        ),
    )
    logging_iter: int = Field(
        default=50,
        description="Console / wandb log frequency (in optimizer steps).",
    )
    max_iter: int = Field(
        default=500,
        description="Total number of optimizer steps the run will execute.",
    )
    callbacks: TrainerCallbacksConfig = Field(default_factory=TrainerCallbacksConfig)
    compile_config: TrainerCompileConfig = Field(default_factory=TrainerCompileConfig)
    compile_cache: CompileCacheConfig = Field(default_factory=CompileCacheConfig)


# ---------------------------------------------------------------- checkpoint
class CheckpointConfig(BaseModel):
    """Resume + save policy. Lands at ``config.checkpoint.*``."""

    model_config = _PYDANTIC_MODEL_CONFIG

    dcp_async_mode_enabled: bool = Field(
        default=False,
        description="Stage a consistent CPU snapshot, then write DCP in background processes.",
    )
    keep_last_n: int | None = Field(
        default=None,
        ge=1,
        description="Keep the newest N completed local DCP checkpoints in this job after a successful save.",
    )
    keep_every_n_steps: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description=(
            "Also retain completed local DCP checkpoints at positive multiples of this optimizer-step interval. "
            "Must be a multiple of save_iter; omitted disables periodic retention."
        ),
    )

    action128_load_mode: Literal["auto", "reset_heads", "extend", "warm_start"] = Field(
        default="auto",
        description="extend preserves full state after append-only changes; warm_start loads weights with new normalization.",
    )

    strict_resume: bool = Field(
        default=False,
        description="Require all non-skipped checkpoint tensors instead of allowing partial loads.",
    )

    keys_to_skip_loading: list[str] = Field(
        default_factory=list,
        description=(
            "Substring blocklist applied at load time. Any tensor whose FQN "
            "contains one of these substrings is skipped (kept at fresh-"
            "init). Used to mask EMA + LoRA + action layers when "
            "warm-starting from a base checkpoint without them."
        ),
    )
    keys_not_to_resume: list[Literal["model", "optim", "scheduler", "trainer", "dataloader"]] = Field(
        default_factory=list,
        description="Checkpoint components excluded when resuming full training state.",
    )
    load_path: str = Field(
        default="???",
        description=(
            "Path to the checkpoint directory to load. '???' is the "
            "OmegaConf MISSING sentinel — recognized by build_hydra_overrides "
            "and skipped, so the user must provide a real path at runtime "
            "(via env interpolation or CLI extra-override)."
        ),
    )
    load_training_state: bool = Field(
        default=False,
        description=(
            "Restore optimizer, scheduler, trainer iteration, and dataloader state "
            "in addition to model weights when loading checkpoint.load_path."
        ),
    )
    save_iter: int = Field(
        default=100,
        description="Save a new checkpoint every N optimizer steps.",
    )


# ---------------------------------------------------------------- dataloader_train
class CatalogDatasetConfig(BaseModel):
    """One five-column Parquet catalog and its integer microbatch ratio."""

    model_config = _PYDANTIC_MODEL_CONFIG

    path: str
    ratio: int = Field(default=1, gt=0)
    vae_manifest_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional dataset-level offline VAE lineage manifest, validated when supplied. Used with "
            "dataloader_train.visual_generation.vae_mode='offline'."
        ),
    )


class DataloaderMediaConfig(BaseModel):
    """Bounded Parquet/video materialization settings."""

    model_config = _PYDANTIC_MODEL_CONFIG

    decode_timeout_sec: float = Field(default=120.0, gt=0)
    max_sampled_frames: int = Field(default=256, gt=0)
    max_frame_pixels: int = Field(default=16_777_216, gt=0)
    max_window_bytes: int = Field(default=1_073_741_824, gt=0)
    max_decode_frames_per_seek: int = Field(default=2_048, gt=0)
    reuse_seek_groups: bool = False
    # TEMPORARY WORKAROUND, NOT DATA REPAIR: dropping windows biases the data mix.
    action_skip_media_errors: bool = False
    action_max_consecutive_skips: int = Field(default=100, gt=0)
    action_failure_log_dir: str | None = None
    action_exclusion_jsonl: str | None = None
    # Parquet metadata's uncompressed row-group size; not a peak-RAM bound.
    action_max_row_group_bytes: int = Field(default=268_435_456, gt=0)
    parquet_handle_cache_size: int = Field(default=8, gt=0)
    video_handle_cache_size: int = Field(default=8, gt=0)
    video_decode_threads: Annotated[int, Field(strict=True, ge=1, le=4)] = Field(
        default=1,
        description="Bounded FFmpeg threads per cached video stream; values >1 enable frame threading in the helper.",
    )
    safetensors_handle_cache_size: int = Field(default=8, gt=0)
    max_latent_tensor_bytes: int = Field(default=1_073_741_824, gt=0)


class VisualGenerationDataloaderConfig(BaseModel):
    """Target-prefix mixture used when a text-only gen_video row omits conditioning."""

    model_config = _PYDANTIC_MODEL_CONFIG

    vae_mode: Literal["online", "offline"] = Field(
        default="online",
        description=(
            "Target materialization mode. 'online' consumes pixel gen_video paths and runs the VAE encoder; "
            "'offline' consumes gen_video.vae_latent references and bypasses VAE construction/encoding."
        ),
    )
    expected_vae_name: str = Field(default="wan2pt2_tokenizer", min_length=1)
    expected_vae_weights_sha256: Optional[str] = Field(
        default=None,
        description="Optional expected dataset-level VAE weights SHA256 for offline manifest validation.",
    )
    cfg_drop_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of dropping generation text conditioning; visual conditions are preserved.",
    )
    conditioning_config: dict[int, float] = Field(default_factory=lambda: {0: 0.7, 1: 0.2, 2: 0.1})

    @field_validator("conditioning_config")
    @classmethod
    def validate_conditioning_config(cls, value: dict[int, float]) -> dict[int, float]:
        if set(value) != {0, 1, 2}:
            raise ValueError("conditioning_config must contain exactly the keys 0, 1, and 2")
        if any(weight < 0 for weight in value.values()) or sum(value.values()) <= 0:
            raise ValueError("conditioning_config weights must be non-negative with a positive total")
        total = sum(value.values())
        return {key: value[key] / total for key in (0, 1, 2)}


class DataloaderTrainConfig(BaseModel):
    """Top-level dataloader scalars only. The dataloader's class (LazyCall)
    and full pipeline wiring (datasets, packers, …) stay in the experiment
    Python — they vary too much between VFM IterativeJointDataLoader,
    PackingDataLoader, and VLM CosmosDataLoader to model uniformly.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    action_use_state: bool | None = Field(
        default=None,
        description="Unified Action128: add one current State128 clean token when valid; None follows catalog.",
    )
    first_frame_color_jitter: bool = Field(
        default=False, description="Ordinary color jitter on the Action RGB condition frame only."
    )

    action_video_fps_default: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Legacy name: maximum Action128 video sampling FPS, applied to video_views[].fps; "
            "lower rates remain unchanged. None applies no cap. Missing view FPS is an error."
        ),
    )

    max_samples_per_batch: Optional[int] = Field(
        default=None,
        description=(
            "Cap on samples per micro-batch. Remapped to "
            "'dataloader_train.batcher.max_batch_size' on the VLM CosmosDataLoader "
            "(its PoolPackingBatcher). None = no per-count cap "
            "(the packer's token budget is what limits batch size)."
        ),
    )
    packing_pool_size: int = Field(
        default=16,
        ge=1,
        description=(
            "Number of processed samples buffered for Understanding pool packing. "
            "This must be at least as large as the desired multi-sample packed batch; "
            "the token budget and max_samples_per_batch remain the final limits."
        ),
    )
    max_sequence_length: Optional[int] = Field(
        default=None,
        description=(
            "Cap on tokens per packed sequence. Remapped to "
            "'dataloader_train.batcher.max_tokens' on the VLM CosmosDataLoader "
            "(its PoolPackingBatcher). None = no per-token cap."
        ),
    )
    enable_map_style_resume: bool = Field(
        default=True,
        description=(
            "Preserve per-worker map-distributor positions in the DCP. The behavior for "
            "non-contiguous pool-packed batches is selected by map_style_resume_mode."
        ),
    )
    map_style_resume_mode: Literal["exact", "at_most_once"] = Field(
        default="exact",
        description=(
            "Resume policy when enable_map_style_resume=true. 'exact' rejects a "
            "non-contiguous reordering-pool batch because a scalar cursor cannot preserve "
            "its buffered holes. 'at_most_once' checkpoints the delivered high-watermark "
            "and resumes after it: already-trained rows are never replayed, while a real "
            "restart may skip unselected candidates that were below the watermark in the "
            "old packing pool."
        ),
    )
    max_caption_tokens: Optional[int] = Field(
        default=None,
        description=(
            "VFM only. Per-caption token cap before truncation — remapped to the SFT "
            "dataset's 'max_caption_tokens'. Structured-JSON captions are longer than dense "
            "prose, so the example recipes set 2048 (measured max ~1790). None = keep the "
            "recipe default. Skipped on VLM (the data packer caps via max_sequence_length)."
        ),
    )
    seed: int = Field(
        default=42,
        description=("Dataloader RNG seed. Skipped on VLM (CosmosDataLoader has no seed ctor kwarg there)."),
    )
    num_workers: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total worker-process budget shared across all catalog data streams.",
    )
    prefetch_factor: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Global number of mixed-dataset batches prefetched in shared_prefetch mode (default 1); "
            "per-worker factor only in legacy_per_source mode. Never multiplied by workers in shared mode."
        ),
    )
    execution_backend: Literal["shared_prefetch", "legacy_per_source"] = "shared_prefetch"
    logical_workers: Optional[list[Annotated[int, Field(strict=True, ge=0)]]] = Field(
        default=None,
        description="Per-source legacy logical worker counts, in datasets order; pin these before changing num_workers.",
    )
    worker_assignment: Optional[list[Annotated[int, Field(strict=True, ge=0)]]] = Field(
        default=None,
        description="Physical worker id for each active logical shard, in source/shard order (shared mode).",
    )
    worker_timeout: float = Field(default=300.0, gt=0, description="Shared batch result timeout in seconds.")
    microbatch_timeout_sec: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
        description="Worker microbatch preparation deadline; 0 disables skipping.",
    )
    microbatch_timeout_max_skips: int = Field(
        default=2,
        ge=0,
        description="Maximum timeout replacements per scheduled microbatch; 0 retries until worker_timeout bounds consumer waiting.",
    )
    microbatch_skip_log_dir: str | None = Field(
        default=None, description="Defaults to OUTPUT_ROOT/logs/skipped_samples; one unique JSONL per run/rank/worker."
    )
    pin_memory: Optional[bool] = Field(
        default=None,
        description="Whether catalog DataLoaders use a PyTorch pin-memory transfer thread.",
    )
    receive_in_background: bool = Field(
        default=True,
        strict=True,
        description="Drain shared-prefetch worker results in a CPU thread within the existing global task budget.",
    )
    datasets: list[CatalogDatasetConfig] = Field(default_factory=list)
    media: DataloaderMediaConfig = Field(default_factory=DataloaderMediaConfig)
    visual_generation: VisualGenerationDataloaderConfig = Field(default_factory=VisualGenerationDataloaderConfig)


# ---------------------------------------------------------------- top
class SFTExperimentConfig(BaseModel):
    """Top-level structured-TOML schema. Each field corresponds to a
    top-level ``[<section>]`` block in the TOML and to a sub-model above.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    job: JobConfig = Field(default_factory=JobConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    dataloader_train: DataloaderTrainConfig = Field(default_factory=DataloaderTrainConfig)
    custom: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form, project-owned escape hatch. Arbitrary nested content "
            "passes through verbatim — the framework never validates inside it. "
            "Injected onto the loaded config as ``config.custom`` after Hydra "
            "resolution; specify concrete values here (no ${...} interpolation)."
        ),
    )

    @model_validator(mode="after")
    def validate_vfm_fixed_sequence_length_contract(self) -> "SFTExperimentConfig":
        """Reject invalid VFM static compile combinations before Hydra load."""

        from cosmos_framework.configs.base.defaults.fp8_training import validate_fp8_training_model_config

        validate_fp8_training_model_config(self.model, task=self.job.task)
        if self.job.task != "vfm":
            if self.dataloader_train.visual_generation.vae_mode != "online":
                raise ValueError("offline VAE mode is supported only for VFM training")
            return self

        if self.dataloader_train.visual_generation.vae_mode == "offline":
            if not self.dataloader_train.datasets:
                raise ValueError("offline VAE mode requires dataloader_train.datasets")

        from cosmos_framework.configs.base.defaults.fixed_sequence_length import (
            fixed_sequence_total,
            validate_fixed_sequence_length_contract,
        )

        fixed = self.model.fixed_sequence_length
        compile_config = self.model.compile
        validate_fixed_sequence_length_contract(self.model)
        if not fixed.enabled:
            return self
        del compile_config
        fixed_total = fixed_sequence_total(fixed)
        if self.dataloader_train.max_sequence_length is None:
            raise ValueError("Fixed sequence length requires dataloader_train.max_sequence_length token-budget mode")
        if self.dataloader_train.max_sequence_length != fixed_total:
            raise ValueError(
                "dataloader_train.max_sequence_length must equal the largest enabled fixed profile; "
                f"got {self.dataloader_train.max_sequence_length} != {fixed_total}"
            )
        return self


# ---------------------------------------------------------------------------
# End-to-end loader: TOML → validate → Hydra overrides → merged Config.
# ---------------------------------------------------------------------------
def _normalize_legacy_text_loss_scale(raw: dict[str, Any], source: str | Path) -> None:
    """Consume the sole supported legacy task-control spelling in one place."""
    model_raw = raw.get("model", {})
    if "text_loss_scale" not in model_raw:
        return

    legacy_scale = model_raw.pop("text_loss_scale")
    if isinstance(legacy_scale, bool) or not isinstance(legacy_scale, (int, float)) or legacy_scale != 0:
        raise ValueError(
            f"{source}: legacy [model].text_loss_scale accepts only numeric 0. "
            "Route batches as generation or understanding microbatches and use "
            "und_loss_scale/gen_loss_scale for task-loss weighting."
        )
    warnings.warn(
        "[model].text_loss_scale=0 is deprecated and was ignored; "
        "route batches through the generation/understanding schema.",
        DeprecationWarning,
        stacklevel=2,
    )


def _resolve_catalog_datasets_for_inspection(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve env interpolation before eagerly inspecting catalog rows.

    Keep the original payload unchanged so the injected LazyCall retains the
    established interpolation behavior at instantiation time.
    """
    resolved = OmegaConf.to_container(OmegaConf.create(datasets), resolve=True)
    assert isinstance(resolved, list)
    return resolved


def _get_action128_roots(datasets: list[dict[str, Any]]) -> list[str]:
    """Inspect resolved catalog metadata before constructing the model contract."""
    from cosmos_framework.data.generator.catalog import ParquetCatalogDataset

    roots = []
    for dataset in _resolve_catalog_datasets_for_inspection(datasets):
        row = ParquetCatalogDataset(dataset["path"])[0]
        if row["type"] == "action" and row["data"].get("adapter") == "unified_action128":
            roots.append(row["data"]["root"])
    return roots


def load_experiment_from_toml(
    toml_path: str | Path,
    extra_overrides: list[str] | None = None,
) -> Any:
    """End-to-end loader for the SFT structured-TOML schema.

    The base config module is picked from ``[job].task`` in the TOML:

    - ``task = "vfm"`` → ``cosmos_framework/configs/base/config.py``
    - ``task = "vlm"`` → ``cosmos_framework/configs/base/reasoner/config.py``

    ``extra_overrides`` takes precedence over TOML values. Catalog dataloader
    overrides and model fixed-shape/token-budget overrides are merged into the
    validated TOML tree before the lazy loader is built; all other entries are
    appended to the Hydra overrides. Each entry must be
    Hydra dotted-path syntax (``key.path=value``); the ``--`` separator token
    is filtered out. Examples::

        ["optimizer.lr=1e-5", "trainer.max_iter=200"]
        ["model.config.parallelism.data_parallel_shard_degree=4"]

    The load then:

    1. Runs ``load_config`` — imports the base config module, runs
       ``make_config()`` (registers config groups + experiment modules), and
       lets Hydra ``compose`` resolve the ``experiment=<name>`` selector and
       apply the dotted-path overrides, followed by ``extra_overrides``.
    2. Injects the TOML's ``[custom]`` table (if any) verbatim onto
       ``config.custom`` *after* loading — kept out of ``build_hydra_overrides``
       so it lands as-is, not per-leaf-remapped. Because this happens after
       Hydra resolution, ``[custom]`` must hold concrete values; ``${...}``
       interpolation against ``custom`` is not supported.

    Returns the merged ``Config`` instance, ready for ``launch()``.
    """
    with open(toml_path, "rb") as fh:
        raw = tomllib.load(fh)
    _normalize_legacy_text_loss_scale(raw, toml_path)

    hydra_extra_overrides: list[str] = []
    if extra_overrides:
        catalog_extra_overrides: list[str] = []
        has_catalog = bool(raw.get("dataloader_train", {}).get("datasets"))
        for override in extra_overrides:
            if not override or override == "--":
                continue
            if "=" not in override:
                raise ValueError(
                    f"extra override {override!r} must be Hydra dotted-path syntax (e.g. 'optimizer.lr=1e-5')."
                )
            normalized = override.removeprefix("+")
            key = normalized.split("=", 1)[0]
            if has_catalog and key.startswith("dataloader_train."):
                catalog_extra_overrides.append(normalized)
            elif key.startswith("model.config.fixed_sequence_length.") or key == (
                "model.config.max_num_tokens_after_packing"
            ):
                # Admission and model execution must see the same CLI shape.
                # Merge before cross-field validation and loader construction.
                catalog_extra_overrides.append(normalized.replace("model.config.", "model.", 1))
            else:
                hydra_extra_overrides.append(override)
        if catalog_extra_overrides:
            merged = OmegaConf.merge(OmegaConf.create(raw), OmegaConf.from_dotlist(catalog_extra_overrides))
            raw = OmegaConf.to_container(merged, resolve=False)
            assert isinstance(raw, dict)

    # Validate structure against the pydantic schema (raises ValidationError on
    # unknown keys because of ``extra="forbid"``).
    cfg = SFTExperimentConfig.model_validate(raw)

    vae_mode = cfg.dataloader_train.visual_generation.vae_mode
    if raw.get("job", {}).get("task", "vfm") == "vfm" and vae_mode == "offline":
        # ``vae_mode`` is the single user-facing switch. Avoid requiring a second
        # load flag and avoid resolving WAN_VAE_PATH when no tokenizer is built.
        raw.setdefault("model", {})["load_vision_tokenizer"] = False
        raw["model"].pop("tokenizer", None)
        cfg.model.load_vision_tokenizer = False

    # ``build_hydra_overrides`` walks the *raw* dict, so an omitted field emits
    # no override and the base config's value wins. ``upload_reproducible_setup``
    # must instead default to False (OSS: no S3), overriding the VLM base's True.
    # Inject the pydantic-resolved value back into raw so the override is always
    # emitted — the default lives in exactly one place (JobConfig.Field).
    raw.setdefault("job", {})["upload_reproducible_setup"] = cfg.job.upload_reproducible_setup

    task = raw.get("job", {}).get("task", "vfm")
    try:
        base_config_path = TASK_TO_BASE_CONFIG[task]
    except KeyError as e:
        raise ValueError(
            f"{toml_path}: [job].task={task!r} is not supported. Valid values: {sorted(TASK_TO_BASE_CONFIG)}"
        ) from e

    overrides = build_hydra_overrides(raw)

    overrides.extend(hydra_extra_overrides)

    # Import lazily so this module stays cheap to import in non-training contexts.
    from cosmos_framework.utils.config import load_config

    config = load_config(base_config_path, overrides)

    from cosmos_framework.configs.base.defaults.fp8_training import validate_fp8_training_model_config

    validate_fp8_training_model_config(config.model.config, task=task)
    fp8 = getattr(config.model.config, "fp8_training", None)
    if fp8 is not None and fp8.enabled:
        target = config.model.get("_target_")
        target_name = target if isinstance(target, str) else getattr(target, "__name__", "")
        if target_name.split(".")[-1] != "OmniMoTModel":
            raise ValueError("FP8 training currently supports OmniMoTModel SFT, not distillation or other model owners")

    if task == "vfm":
        config.model.config.vision_vae_mode = vae_mode
        if vae_mode == "offline":
            temporal_mode = str(config.model.config.diffusion_expert_config.vision_temporal_position_mode)
            if temporal_mode != "latent_index":
                raise ValueError(
                    f"offline VAE mode requires vision_temporal_position_mode='latent_index', got {temporal_mode!r}"
                )
            # These callbacks require a live generation tokenizer and/or raw
            # target pixels. Derive their absence from the same run-level
            # switch instead of asking users to maintain extra booleans.
            with open_dict(config.trainer.callbacks):
                for callback_name in ("compile_tokenizer", "every_n_sample_reg", "every_n_sample_ema"):
                    config.trainer.callbacks.pop(callback_name, None)

    # Inject [custom] verbatim after Hydra resolution. Kept off the base config
    # schema so the framework-owned hydra configs stay untouched; lands as a
    # plain dict reachable via config.custom.
    config.custom = raw.get("custom", {})
    # Catalog wiring is task-specific (Understanding/T2V/Action use different
    # processors), so expose the validated settings without forcing them into
    # legacy Hydra recipe nodes.  Catalog loader factories consume these fields.
    # Preserve the established online catalog payload shape. Optional fields
    # such as vae_manifest_path are emitted only when configured.
    config.catalog_datasets = [item.model_dump(exclude_none=True) for item in cfg.dataloader_train.datasets]
    config.dataloader_media = cfg.dataloader_train.media.model_dump()
    action128_roots = _get_action128_roots(config.catalog_datasets) if cfg.dataloader_train.datasets else []
    if action128_roots:
        from cosmos_framework.data.generator.action.unified_action128 import build_action128_spec

        if task != "vfm" or not config.model.config.action_gen:
            raise ValueError("Action128 catalogs require a VFM recipe with action_gen=true")
        spec = build_action128_spec(action128_roots)
        config.model.config.action128_spec = spec
        config.model.config.max_action_dim = 128
        config.model.config.num_embodiment_domains = spec["projection_routing"]["num_slots"]
        config.model.config.video_temporal_causal = False
        config.model.config.diffusion_expert_config.enable_fps_modulation = True
        # The DCP loader examines the source contract before deciding whether
        # new heads are needed. Do not inherit a legacy recipe's head blocklist.
        explicit_skip_override = any(
            value.lstrip("+").startswith("checkpoint.keys_to_skip_loading=") for value in extra_overrides or []
        )
        if "keys_to_skip_loading" not in cfg.checkpoint.model_fields_set and not explicit_skip_override:
            config.checkpoint.keys_to_skip_loading = []
        stats = dict(config.trainer.callbacks.get("training_stats", {}))
        stats.setdefault("_target_", "cosmos_framework.callbacks.training_stats.TrainingStatsCallback")
        stats.setdefault("log_freq", 100)
        config.trainer.callbacks.training_stats = stats
    if cfg.dataloader_train.datasets:
        from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
        from cosmos_framework.data.generator.catalog_processors import build_catalog_training_dataloader
        from cosmos_framework.data.generator.processors import build_processor_lazy
        from cosmos_framework.utils.lazy_config import LazyCall as L

        legacy_loader = config.dataloader_train
        worker_loader = legacy_loader.get("dataloader", legacy_loader)
        if task == "vfm":
            vlm_config = config.model.config.vlm_config
            processor_config = vlm_config.get("tokenizer") or {}
            processor_model_name = str(
                processor_config.get("tokenizer_type")
                or processor_config.get("repository")
                or processor_config.get("pretrained_model_name")
                or vlm_config.model_name
            )
            resolution = str(config.model.config.resolution)
        else:
            processor_model_name = str(config.model.config.policy.backbone.model_name)
            resolution = "256"

        max_sequence_length = cfg.dataloader_train.max_sequence_length
        if max_sequence_length is None:
            max_sequence_length = legacy_loader.get("max_sequence_length")
        if max_sequence_length is None and "batcher" in legacy_loader:
            max_sequence_length = legacy_loader.batcher.get("max_tokens")
        max_sequence_length = int(max_sequence_length or 45_056)

        num_workers = cfg.dataloader_train.num_workers
        if num_workers is None:
            num_workers = int(worker_loader.get("num_workers", 0) or 0)
        prefetch_factor = cfg.dataloader_train.prefetch_factor
        if prefetch_factor is None:
            prefetch_factor = (
                1
                if cfg.dataloader_train.execution_backend == "shared_prefetch"
                else worker_loader.get("prefetch_factor")
            )
        if prefetch_factor is not None:
            prefetch_factor = int(prefetch_factor)
        pin_memory = cfg.dataloader_train.pin_memory
        if pin_memory is None:
            pin_memory = bool(worker_loader.get("pin_memory", False))
        config.dataloader_train = L(build_catalog_training_dataloader)(
            dataset_specs=config.catalog_datasets,
            action_video_fps_default=cfg.dataloader_train.action_video_fps_default,
            action_use_state=cfg.dataloader_train.action_use_state,
            first_frame_color_jitter=cfg.dataloader_train.first_frame_color_jitter,
            processor=L(build_processor_lazy)(
                tokenizer_type=processor_model_name,
                config_variant="hf",
            ),
            conditioning_config=cfg.dataloader_train.visual_generation.conditioning_config,
            cfg_drop_ratio=cfg.dataloader_train.visual_generation.cfg_drop_ratio,
            vae_mode=vae_mode,
            expected_latent_channels=int(config.model.config.state_ch),
            expected_vae_name=cfg.dataloader_train.visual_generation.expected_vae_name,
            expected_vae_weights_sha256=(cfg.dataloader_train.visual_generation.expected_vae_weights_sha256),
            seed=cfg.dataloader_train.seed,
            max_sequence_length=max_sequence_length,
            max_samples_per_batch=cfg.dataloader_train.max_samples_per_batch,
            packing_pool_size=cfg.dataloader_train.packing_pool_size,
            enable_map_style_resume=cfg.dataloader_train.enable_map_style_resume,
            map_style_resume_mode=cfg.dataloader_train.map_style_resume_mode,
            max_caption_tokens=cfg.dataloader_train.max_caption_tokens or 2_048,
            resolution=resolution,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            execution_backend=cfg.dataloader_train.execution_backend,
            logical_workers=cfg.dataloader_train.logical_workers,
            worker_assignment=cfg.dataloader_train.worker_assignment,
            worker_timeout=cfg.dataloader_train.worker_timeout,
            microbatch_timeout_sec=cfg.dataloader_train.microbatch_timeout_sec,
            microbatch_timeout_max_skips=cfg.dataloader_train.microbatch_timeout_max_skips,
            microbatch_skip_log_dir=cfg.dataloader_train.microbatch_skip_log_dir,
            pin_memory=pin_memory,
            receive_in_background=cfg.dataloader_train.receive_in_background,
            media_config=config.dataloader_media,
            static_packing_config={
                "spatial_compression_factor": int(config.model.config.tokenizer.spatial_compression_factor),
                "temporal_compression_factor": int(config.model.config.tokenizer.temporal_compression_factor),
                "latent_patch_size": int(config.model.config.diffusion_expert_config.patch_spatial),
                "unified_3d_mrope_reset_spatial_ids": bool(
                    config.model.config.diffusion_expert_config.unified_3d_mrope_reset_spatial_ids
                ),
                "unified_3d_mrope_temporal_modality_margin": int(
                    config.model.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin
                ),
                "enable_fps_modulation": bool(config.model.config.diffusion_expert_config.enable_fps_modulation),
                "base_fps": float(config.model.config.diffusion_expert_config.base_fps),
                "vision_temporal_position_mode": str(
                    config.model.config.diffusion_expert_config.vision_temporal_position_mode
                ),
                "vision_spatial_position_mode": str(
                    config.model.config.diffusion_expert_config.vision_spatial_position_mode
                ),
                "video_temporal_causal": bool(config.model.config.video_temporal_causal),
                "action_dim": int(config.model.config.max_action_dim),
            },
            fixed_sequence_length_enabled=cfg.model.fixed_sequence_length.enabled,
            fixed_understanding_tokens=cfg.model.fixed_sequence_length.understanding_tokens,
            fixed_und_tokens=cfg.model.fixed_sequence_length.und_tokens,
            fixed_gen_tokens=cfg.model.fixed_sequence_length.gen_tokens,
            fixed_action_tokens=cfg.model.fixed_sequence_length.action_tokens,
        )
        with open_dict(config.trainer.callbacks):
            if cfg.dataloader_train.enable_map_style_resume:
                config.trainer.callbacks.dataloader_state = L(JointCosmosDataLoaderStateCallback)()
            else:
                config.trainer.callbacks.pop("dataloader_state", None)
    return config
