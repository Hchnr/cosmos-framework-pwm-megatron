# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Semantic rendering and media materialization for Parquet catalog rows."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_droid_merged_lerobot_sft_dataset,
    get_action_droid_sft_dataset,
    get_action_libero_sft_dataset,
)
from cosmos_framework.data.generator.action.utils.transforms import VideoResize, find_closest_target_size
from cosmos_framework.data.generator.catalog import (
    CatalogSchemaError,
    EpisodeBlockDistributor,
    ParquetBlockDistributor,
    ParquetCatalogDataset,
)
from cosmos_framework.data.generator.dataflow.base import RawItemProcessor
from cosmos_framework.data.generator.dataflow.batchers import (
    PoolPackingBatcher,
    SequentialPackingBatcher,
    SequentialTokenBudgetBatcher,
)
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.dataflow.loader import CosmosDataLoader, JointCosmosDataLoader
from cosmos_framework.data.generator.dataflow.processors import IdentityProcessor
from cosmos_framework.data.generator.drop_sample_contract import DROP_SAMPLE_KEY, DROP_SAMPLE_REASON_KEY
from cosmos_framework.data.generator.media_decode import MediaDecodeError, SupervisedVideoDecoder
from cosmos_framework.data.generator.offline_vae import (
    SafetensorsLatentReader,
    parse_vae_latent_ref,
    validate_offline_vae_manifests,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.modality import add_special_tokens
from cosmos_framework.data.generator.sequence_packing.static_layout import StaticPackingConfig
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_generation_caption
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.reasoner.constant import PROCESSOR_KEYS_TO_ADD

_VAE_TEMPORAL_COMPRESSION_FACTOR = 4


def _messages(row: dict[str, Any], expected_type: str) -> list[dict[str, Any]]:
    if row.get("type") != expected_type:
        raise CatalogSchemaError(
            f"catalog row {row.get('id')!r} has type {row.get('type')!r}, expected {expected_type!r}"
        )
    messages = row.get("data", {}).get("messages")
    if not isinstance(messages, list) or not messages:
        raise CatalogSchemaError(f"catalog row {row.get('id')!r} requires data.messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise CatalogSchemaError(f"invalid chat message in catalog row {row.get('id')!r}")
    return copy.deepcopy(messages)


class CatalogMediaMaterializer:
    """Resolve paths embedded in ``data.messages`` without a side ``media`` column."""

    def __init__(self, video_decoder: SupervisedVideoDecoder | None = None) -> None:
        self._video_decoder = video_decoder or SupervisedVideoDecoder()
        self.last_decode_metadata: list[dict[str, Any]] = []

    def materialize_chat(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.last_decode_metadata = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    raise CatalogSchemaError("chat content entries must be objects")
                kind = item.get("type")
                if kind == "image" and isinstance(item.get("image"), str):
                    image_path = Path(item["image"]).expanduser().resolve()
                    try:
                        with Image.open(image_path) as image:
                            item["image"] = image.convert("RGB").copy()
                    except OSError as error:
                        raise MediaDecodeError(f"image decode failed for {image_path}: {error}") from error
                elif kind == "video" and isinstance(item.get("video"), str):
                    count = min(int(item.get("num_frames", 32)), 256)
                    frames, metadata = self._video_decoder.decode(item["video"], uniform_sample_count=count)
                    item["video"] = [Image.fromarray(frame) for frame in frames]
                    item.setdefault("fps", metadata.get("fps") or 1.0)
                    self.last_decode_metadata.append(metadata)
        return messages


class CatalogUnderstandingProcessor(RawItemProcessor):
    """Render one Qwen3-VL chat sample for causal sequence packing."""

    def __init__(self, processor: Any, video_decoder: SupervisedVideoDecoder | None = None) -> None:
        self._processor = processor
        self._materializer = CatalogMediaMaterializer(video_decoder)
        tokenizer = getattr(processor, "tokenizer", processor)
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
        self._special_token_ids: set[int] = set()
        for token in (
            "<|im_start|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<think>",
            "</think>",
        ):
            if vocab is not None and token not in vocab:
                continue
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except (KeyError, ValueError):
                continue
            if token_id is not None:
                self._special_token_ids.add(int(token_id))

    def process(self, item: dict[str, Any]) -> dict[str, Any]:
        messages = self._materializer.materialize_chat(_messages(item, "understanding"))
        inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        input_ids = inputs["input_ids"]
        token_mask = self._processor.add_assistant_tokens_mask(input_ids).bool()
        for token_id in self._special_token_ids:
            token_mask &= input_ids != token_id
        result: dict[str, Any] = {
            "text_token_ids": input_ids,
            "sequence_plan": SequencePlan(
                has_text=True,
                supervise_text=bool(token_mask.any()),
                text_loss_mask=token_mask.tolist(),
            ),
            "catalog_id": item["id"],
            "decode_metadata": list(self._materializer.last_decode_metadata),
        }
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]
        return result


def _generation_parts(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, str | None, str, dict[str, Any], bool]:
    """Split chat conditioning from the assistant's DiT target declaration."""

    messages = _messages(row, "visual_generation")
    rendered_messages: list[dict[str, Any]] = []
    prompt_parts: list[str] = []
    assistant_parts: list[str] = []
    generation_kind: str | None = None
    generation: dict[str, Any] | None = None
    has_user_media = False
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            content = [{"type": "text", "text": content}]
        rendered_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise CatalogSchemaError(f"catalog row {row['id']!r} has non-object chat content")
            kind = part.get("type")
            if message["role"] == "user" and kind == "text":
                prompt_parts.append(str(part.get("text", "")))
            if message["role"] == "user" and kind in {"image", "video"}:
                has_user_media = True
            if message["role"] == "assistant" and kind == "text":
                text = str(part.get("text", "")).strip()
                if text:
                    assistant_parts.append(text)
            if kind in {"gen_image", "gen_video"}:
                if message["role"] != "assistant":
                    raise CatalogSchemaError(f"catalog row {row['id']!r} places {kind!r} outside the assistant message")
                if generation is not None:
                    raise CatalogSchemaError(f"catalog row {row['id']!r} has multiple generation items")
                generation_kind = str(kind)
                generation = part
                continue
            rendered_content.append(part)
        if rendered_content:
            rendered_messages.append({**message, "content": rendered_content})
    if generation is None or generation_kind is None:
        raise CatalogSchemaError(f"catalog row {row['id']!r} requires one assistant gen_image/gen_video item")
    assistant_text = "\n".join(assistant_parts) if assistant_parts else None
    return (
        rendered_messages,
        "\n".join(prompt_parts).strip(),
        assistant_text,
        generation_kind,
        generation,
        has_user_media,
    )


def _normalized_conditioning_config(values: dict[int | str, float] | None) -> dict[int, float]:
    raw = {0: 0.7, 1: 0.2, 2: 0.1} if values is None else values
    converted: dict[int, float] = {}
    for key, value in raw.items():
        if isinstance(key, bool):
            raise ValueError("conditioning_config keys must be 0, 1, or 2")
        try:
            frame_count = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError("conditioning_config keys must be 0, 1, or 2") from error
        weight = float(value)
        if frame_count not in {0, 1, 2} or weight < 0:
            raise ValueError("conditioning_config requires non-negative weights for keys 0, 1, and 2")
        converted[frame_count] = weight
    if set(converted) != {0, 1, 2} or sum(converted.values()) <= 0:
        raise ValueError("conditioning_config must contain keys 0, 1, and 2 with a positive total weight")
    total = sum(converted.values())
    return {key: converted[key] / total for key in (0, 1, 2)}


def _select_conditioning_latents(
    weights: dict[int, float], *, seed: int, epoch: int, dataset_fingerprint: str, sample_id: str
) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"cosmoscond")
    digest.update(f"target-prefix\0{seed}\0{epoch}\0{dataset_fingerprint}\0{sample_id}".encode())
    draw = int.from_bytes(digest.digest(), "big") / 2**64
    cumulative = 0.0
    for frame_count in (0, 1, 2):
        cumulative += weights[frame_count]
        if draw < cumulative:
            return frame_count
    return 2


def _explicit_conditioning_latent_frames(generation: dict[str, Any], *, row_id: str) -> int | None:
    if "conditioning" in generation:
        raise CatalogSchemaError(
            f"catalog row {row_id!r}: nested conditioning is unsupported; use conditioning_latent_frames"
        )
    value = generation.get("conditioning_latent_frames")
    if value is None:
        return None
    if generation.get("type") != "gen_video":
        raise CatalogSchemaError(f"catalog row {row_id!r}: conditioning_latent_frames is valid only for gen_video")
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1, 2}:
        raise CatalogSchemaError(f"catalog row {row_id!r}: conditioning_latent_frames must be 0, 1, 2, or null")
    return value


def _video_clip_contract(generation: dict[str, Any], *, row_id: str) -> tuple[int, int, float]:
    missing = [field for field in ("trim_head_frames", "valid_frames", "fps") if field not in generation]
    if missing:
        raise CatalogSchemaError(f"catalog row {row_id!r}: gen_video is missing required clip fields {missing}")
    trim_head_frames = generation["trim_head_frames"]
    valid_frames = generation["valid_frames"]
    fps = generation["fps"]
    if isinstance(trim_head_frames, bool) or not isinstance(trim_head_frames, int) or trim_head_frames < 0:
        raise CatalogSchemaError(f"catalog row {row_id!r}: trim_head_frames must be a non-negative integer")
    if isinstance(valid_frames, bool) or not isinstance(valid_frames, int) or valid_frames <= 0:
        raise CatalogSchemaError(f"catalog row {row_id!r}: valid_frames must be a positive integer")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise CatalogSchemaError(f"catalog row {row_id!r}: fps must be a positive target sampling rate")
    return trim_head_frames, valid_frames, float(fps)


def _center_crop_temporal_window(
    frames: tuple[np.ndarray, ...],
    metadata: dict[str, Any],
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Center-crop a decoded window to the VAE-compatible ``4 * n + 1`` shape."""
    compatible_frames = 1 + ((len(frames) - 1) // _VAE_TEMPORAL_COMPRESSION_FACTOR) * _VAE_TEMPORAL_COMPRESSION_FACTOR
    removed = len(frames) - compatible_frames
    crop_start = removed // 2
    crop_end = crop_start + compatible_frames
    if removed == 0:
        return frames, metadata

    cropped_metadata = dict(metadata)
    for key in ("requested_timestamps_sec", "timestamps_sec", "pts"):
        values = cropped_metadata.get(key)
        if isinstance(values, (list, tuple)) and len(values) == len(frames):
            cropped_metadata[key] = list(values[crop_start:crop_end])
    cropped_metadata["temporal_center_crop"] = {
        "input_frames": len(frames),
        "output_frames": compatible_frames,
        "removed_head_frames": crop_start,
        "removed_tail_frames": removed - crop_start,
        "temporal_compression_factor": _VAE_TEMPORAL_COMPRESSION_FACTOR,
    }
    return frames[crop_start:crop_end], cropped_metadata


class CatalogVisualGenerationProcessor(RawItemProcessor):
    """Render Qwen3-VL chat conditioning followed by an image/video DiT target."""

    def __init__(
        self,
        processor: Any,
        *,
        video_decoder: SupervisedVideoDecoder | None = None,
        resolution: str | int = "256",
        max_caption_tokens: int = 2_048,
        conditioning_config: dict[int | str, float] | None = None,
        cfg_drop_ratio: float = 0.0,
        seed: int = 42,
        vae_mode: str = "online",
        expected_latent_channels: int = 48,
        max_latent_tensor_bytes: int = 1_073_741_824,
    ) -> None:
        if vae_mode not in {"online", "offline"}:
            raise ValueError(f"vae_mode must be 'online' or 'offline', got {vae_mode!r}")
        if not 0.0 <= cfg_drop_ratio <= 1.0:
            raise ValueError("cfg_drop_ratio must be between 0 and 1")
        self._cfg_drop_ratio = cfg_drop_ratio
        self._processor = processor
        self._tokenizer = getattr(processor, "tokenizer", processor)
        self._decoder = video_decoder or SupervisedVideoDecoder()
        self._materializer = CatalogMediaMaterializer(self._decoder)
        self._resolution = resolution
        self._max_caption_tokens = max_caption_tokens
        self._conditioning_config = _normalized_conditioning_config(conditioning_config)
        self._seed = seed
        self._resize = VideoResize()
        self._vae_mode = vae_mode
        self._expected_latent_channels = int(expected_latent_channels)
        self._max_latent_tensor_bytes = int(max_latent_tensor_bytes)

    def process(self, item: dict[str, Any]) -> dict[str, Any]:
        messages, caption, assistant_text, generation_kind, generation, has_user_media = _generation_parts(item)
        requested_conditioning = _explicit_conditioning_latent_frames(generation, row_id=str(item["id"]))
        is_video = generation_kind == "gen_video"
        has_pixel_target = generation.get(generation_kind) is not None
        has_latent_target = generation.get("vae_latent") is not None
        if has_pixel_target == has_latent_target:
            raise CatalogSchemaError(
                f"catalog row {item['id']!r}: {generation_kind} must contain exactly one of "
                f"{generation_kind!r} or 'vae_latent'"
            )

        offline_ref = None
        decode_metadata: dict[str, Any] | None = None
        if self._vae_mode == "offline":
            if not is_video:
                raise CatalogSchemaError(f"catalog row {item['id']!r}: offline VAE mode supports gen_video only")
            if has_user_media:
                raise CatalogSchemaError(
                    f"catalog row {item['id']!r}: offline VAE mode does not support user image/video media"
                )
            if not has_latent_target:
                raise CatalogSchemaError(f"catalog row {item['id']!r}: offline VAE mode requires gen_video.vae_latent")
            offline_ref = parse_vae_latent_ref(
                generation["vae_latent"],
                fps=generation.get("fps"),
                row_id=str(item["id"]),
                dataset=str(item.get("dataset", "")),
                expected_channels=self._expected_latent_channels,
                max_tensor_bytes=self._max_latent_tensor_bytes,
            )
            generation_path = offline_ref.path
            target_fps = offline_ref.target_fps
            target_latents = offline_ref.latent_shape[1]
            result: dict[str, Any] = {"video_latent_ref": offline_ref}
        else:
            if not has_pixel_target or not isinstance(generation[generation_kind], str):
                raise CatalogSchemaError(
                    f"catalog row {item['id']!r}: online VAE mode requires string field {generation_kind!r}"
                )
            generation_path = str(generation[generation_kind])
            if is_video:
                trim_head_frames, valid_frames, target_fps = _video_clip_contract(generation, row_id=str(item["id"]))
                decode_resize: dict[str, int] = {}
                metadata_width, metadata_height = generation.get("width"), generation.get("height")
                if (
                    isinstance(metadata_width, int)
                    and not isinstance(metadata_width, bool)
                    and metadata_width > 0
                    and isinstance(metadata_height, int)
                    and not isinstance(metadata_height, bool)
                    and metadata_height > 0
                ):
                    canvas_width, canvas_height = find_closest_target_size(
                        metadata_height, metadata_width, self._resolution
                    )
                    decode_resize = {
                        "output_canvas_width": canvas_width,
                        "output_canvas_height": canvas_height,
                    }
                frames, decode_metadata = self._decoder.decode(
                    generation_path,
                    trim_head_frames=trim_head_frames,
                    valid_frames=valid_frames,
                    target_fps=target_fps,
                    **decode_resize,
                )
                frames, decode_metadata = _center_crop_temporal_window(frames, decode_metadata)
                media_array = np.stack(frames, axis=0)
            else:
                image_path = Path(generation_path).expanduser().resolve()
                try:
                    with Image.open(image_path) as image:
                        media_array = np.asarray(image.convert("RGB")).copy()[None]
                except OSError as error:
                    raise MediaDecodeError(f"image decode failed for {image_path}: {error}") from error
                decode_metadata = {"path": str(image_path), "num_frames": 1}
                target_fps = 1.0
            video = torch.from_numpy(np.ascontiguousarray(media_array.transpose(3, 0, 1, 2))).to(torch.uint8)
            result = {"video": video}
            self._resize(result, self._resolution)
            if not is_video:
                result["images"] = result.pop("video")[:, 0]
            media = result["video"] if is_video else result["images"]
            target_latents = 1 + (int(media.shape[1]) - 1) // _VAE_TEMPORAL_COMPRESSION_FACTOR if is_video else 1

        # Stateless, epoch-scoped sampling preserves exact map-style resume and
        # does not couple CFG dropout to the target-prefix mixture draw.
        digest = hashlib.blake2b(digest_size=8, person=b"cosmoscfg")
        digest.update(
            f"{self._seed}\0{item.get('_dp_epoch', 0)}\0"
            f"{item.get('_dp_dataset_fingerprint', item.get('dataset', ''))}\0{item['id']}".encode()
        )
        cfg_dropped = int.from_bytes(digest.digest(), "big") / 2**64 < self._cfg_drop_ratio
        if cfg_dropped:
            caption = ""
            assistant_text = None
            # Assistant explanations can leak the caption into the DiT context.
            # Keep system instructions and visual inputs, including clean prefixes.
            messages = [message for message in messages if message["role"] != "assistant"]
            for message in messages:
                if message["role"] == "user":
                    for part in message["content"]:
                        if part.get("type") == "text":
                            part["text"] = ""

        materialized_messages = self._materializer.materialize_chat(messages)
        processor_inputs: dict[str, Any] = {}
        rendered = self._processor.apply_chat_template(
            materialized_messages,
            tokenize=True,
            add_generation_prompt=assistant_text is None,
            enable_thinking=False,
        )
        if isinstance(rendered, Mapping) and "input_ids" in rendered:
            processor_inputs = dict(rendered)
            text_ids_tensor = rendered["input_ids"].clone()
            attention_mask = rendered.get("attention_mask")
            if assistant_text is None:
                empty_think = self._tokenizer.encode("<think></think>", add_special_tokens=False)
                if empty_think and text_ids_tensor[-len(empty_think) :].tolist() == empty_think:
                    text_ids_tensor = text_ids_tensor[: -len(empty_think)]
                    if attention_mask is not None:
                        attention_mask = attention_mask[: -len(empty_think)]
            else:
                im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
                positions = (text_ids_tensor == im_end_id).nonzero(as_tuple=False).flatten()
                if positions.numel() == 0:
                    raise CatalogSchemaError(f"catalog row {item['id']!r}: assistant text lacks <|im_end|>")
                end = int(positions[-1]) + 1
                text_ids_tensor = text_ids_tensor[:end]
                if attention_mask is not None:
                    attention_mask = attention_mask[:end]
            if assistant_text is None:
                text_loss_mask = [False] * int(text_ids_tensor.numel())
            else:
                text_loss_mask = self._processor.add_assistant_tokens_mask(text_ids_tensor).bool().tolist()
            text_ids = text_ids_tensor.tolist()
            processor_inputs["input_ids"] = text_ids_tensor
            if attention_mask is not None:
                processor_inputs["attention_mask"] = attention_mask
        else:
            if has_user_media:
                raise CatalogSchemaError(
                    f"catalog row {item['id']!r}: user image/video requires a full Qwen3-VL processor"
                )
            text_ids, text_loss_mask = tokenize_generation_caption(
                caption,
                self._tokenizer,
                assistant_text=assistant_text,
                is_video=is_video,
                use_system_prompt=False,
            )
        if len(text_ids) > self._max_caption_tokens:
            raise CatalogSchemaError(
                f"catalog row {item['id']!r} chat template has {len(text_ids)} tokens, "
                f"exceeding max_caption_tokens={self._max_caption_tokens}; refusing to truncate assistant EOS"
            )
        if not is_video:
            requested_conditioning = 0
        elif requested_conditioning is None and has_user_media:
            # VLM media conditioning and DiT clean-prefix conditioning are
            # independent.  A multimodal user prompt without an explicit
            # target-prefix remains a T2V target (zero clean target latents).
            requested_conditioning = 0
        elif requested_conditioning is None:
            requested_conditioning = _select_conditioning_latents(
                self._conditioning_config,
                seed=self._seed,
                epoch=int(item.get("_dp_epoch", 0)),
                dataset_fingerprint=str(item.get("_dp_dataset_fingerprint", item.get("dataset", ""))),
                sample_id=str(item["id"]),
            )
        effective_conditioning = min(requested_conditioning, max(0, target_latents - 1))
        result.update(
            {
                "__key__": item["id"],
                "__url__": generation_path,
                "ai_caption": caption,
                "cfg_dropped": cfg_dropped,
                "text_token_ids": torch.tensor(text_ids, dtype=torch.long),
                "conditioning_fps": float(target_fps),
                "num_frames": (_VAE_TEMPORAL_COMPRESSION_FACTOR * (target_latents - 1) + 1 if is_video else 1),
                "sequence_plan": SequencePlan(
                    has_text=True,
                    supervise_text=any(text_loss_mask),
                    text_loss_mask=text_loss_mask,
                    has_vision=True,
                    condition_frame_indexes_vision=list(range(effective_conditioning)),
                ),
                "conditioning_requested_latent_frames": requested_conditioning,
                "conditioning_effective_latent_frames": effective_conditioning,
                "vision_input_mode": self._vae_mode,
            }
        )
        if offline_ref is None:
            assert decode_metadata is not None
            media = result["video"] if is_video else result["images"]
            result["padding_mask"] = torch.zeros(media.shape[-2:], dtype=torch.float32).unsqueeze(0)
            result["num_frames"] = int(media.shape[1]) if is_video else 1
            result["decode_metadata"] = decode_metadata
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in {"input_ids", "attention_mask"}:
                continue
            if key in processor_inputs and processor_inputs[key] is not None:
                result[key] = processor_inputs[key]
        return result


class DropFailedMediaProcessor(RawItemProcessor):
    """Mark ordinary media failures for loader-side drop; propagate structural errors."""

    def __init__(self, processor: RawItemProcessor) -> None:
        self._processor = processor

    def process(self, item: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._processor.process(item)
        except MediaDecodeError as error:
            sample_id = str(item.get("id", "unknown"))
            reason = f"{type(error).__name__}: {error}"
            return {
                DROP_SAMPLE_KEY: True,
                DROP_SAMPLE_REASON_KEY: reason,
                "catalog_id": sample_id,
            }


class OfflineVaeLineageProcessor(RawItemProcessor):
    """Enforce the dataset sidecar identity on every row before processing."""

    def __init__(self, processor: RawItemProcessor, *, expected_dataset: str) -> None:
        self._processor = processor
        self._expected_dataset = expected_dataset

    def process(self, item: dict[str, Any]) -> dict[str, Any]:
        if item.get("dataset") != self._expected_dataset:
            raise CatalogSchemaError(
                f"catalog row {item.get('id')!r} has dataset={item.get('dataset')!r}, "
                f"but offline VAE manifest declares dataset={self._expected_dataset!r}"
            )
        return self._processor.process(item)


def build_video_decoder_from_config(media: dict[str, Any] | None = None) -> SupervisedVideoDecoder:
    """Construct the killable decoder from ``dataloader_train.media`` values."""

    values = dict(media or {})
    return SupervisedVideoDecoder(
        timeout_seconds=float(values.get("decode_timeout_sec", 120.0)),
        video_handle_cache_size=int(values.get("video_handle_cache_size", 8)),
        max_sampled_frames=int(values.get("max_sampled_frames", 256)),
        max_frame_pixels=int(values.get("max_frame_pixels", 16_777_216)),
        max_window_bytes=int(values.get("max_window_bytes", 1_073_741_824)),
        max_decode_frames_per_seek=int(values.get("max_decode_frames_per_seek", 2_048)),
        reuse_seek_groups=bool(values.get("reuse_seek_groups", False)),
        video_decode_threads=values.get("video_decode_threads", 1),
    )


def build_action_dataset_from_catalog(
    path: str | Path,
    *,
    tokenizer_config: dict | None = None,
    parquet_handle_cache_size: int = 8,
    action128_spec: dict | None = None,
    seed: int = 42,
    media_config: dict | None = None,
    action_video_fps_default: float | None = None,
    action_use_state: bool | None = None,
) -> torch.utils.data.Dataset:
    """Expand the one-row Action catalog into the existing LeRobot window dataset."""

    catalog = ParquetCatalogDataset(
        path,
        expected_type="action",
        handle_cache_size=parquet_handle_cache_size,
    )
    if len(catalog) != 1:
        raise CatalogSchemaError(f"an Action catalog must contain exactly one dataset row, got {len(catalog)}")
    data = dict(catalog[0]["data"])
    adapter = data.pop("adapter", None)
    data.pop("format", None)
    if adapter == "droid_offline":
        from cosmos_framework.data.generator.action.offline_vae import OfflineActionDataset

        return OfflineActionDataset(data["manifest_path"], tokenizer_config)
    selection_manifest = data.pop("selection_manifest", None)
    if not isinstance(data.get("root"), str):
        raise CatalogSchemaError("Action catalog data.root must be a path string")
    data.setdefault("tokenizer_config", tokenizer_config)
    if adapter == "unified_action128":
        from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset

        if action_use_state is not None:
            data["use_state"] = action_use_state

        # Legacy config name: this is now a ceiling on view FPS, NOT a
        # missing-value replacement. Dataset metadata cleaning is a separate,
        # explicit operation; training workers never mutate shared Parquet.
        if action_video_fps_default is not None:
            data["video_fps_max"] = action_video_fps_default
        data["max_row_group_bytes"] = (media_config or {}).get("action_max_row_group_bytes", 256 * 1024**2)
        return UnifiedAction128Dataset(
            **data,
            action128_spec=action128_spec,
            seed=seed,
            video_decoder=build_video_decoder_from_config(media_config),
        )
    if adapter == "droid":
        dataset = get_action_droid_sft_dataset(**data)
        if selection_manifest:
            from cosmos_framework.data.generator.action.offline_vae import SelectedActionDataset

            dataset = SelectedActionDataset(dataset, selection_manifest)
        return dataset
    if adapter == "droid_merged":
        return get_action_droid_merged_lerobot_sft_dataset(**data)
    if adapter == "libero":
        return get_action_libero_sft_dataset(**data)
    raise CatalogSchemaError(f"unsupported Action catalog adapter: {adapter!r}")


def build_catalog_joint_dataloader(
    dataset_specs: list[dict[str, Any]],
    *,
    components: dict[str, dict[str, Any]],
    seed: int = 42,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    execution_backend: str = "shared_prefetch",
    logical_workers: list[int] | None = None,
    worker_assignment: list[int] | None = None,
    worker_timeout: float = 300.0,
    microbatch_timeout_sec: float = 0.0,
    microbatch_timeout_max_skips: int = 2,
    microbatch_skip_log_dir: str | None = None,
    pin_memory: bool = False,
    receive_in_background: bool = True,
    parallel_dims: Any = None,
    tokenizer_config: dict | None = None,
    parquet_handle_cache_size: int = 8,
    media_config: dict[str, Any] | None = None,
    action_video_fps_default: float | None = None,
    action_use_state: bool | None = None,
    first_frame_color_jitter: bool = False,
    enable_map_style_resume: bool = True,
    map_style_resume_mode: str = "exact",
) -> JointCosmosDataLoader:
    """Build the exact-ratio joint loader from validated TOML dataset specs.

    ``components`` supplies the existing Cosmos batcher/collator for each task
    type and, for non-Action tasks, its task-specific ``RawItemProcessor``.
    This preserves the public four-role dataflow rather than introducing a
    second loader type system.
    """

    if microbatch_timeout_sec and execution_backend != "shared_prefetch":
        raise ValueError("microbatch_timeout_sec requires execution_backend=shared_prefetch")
    media_values = dict(media_config or {})
    if execution_backend not in {"shared_prefetch", "legacy_per_source"}:
        raise ValueError(f"Unknown execution_backend: {execution_backend!r}")
    if execution_backend == "legacy_per_source" and worker_assignment is not None:
        raise ValueError("worker_assignment is only supported by shared_prefetch")
    if logical_workers is not None and (
        len(logical_workers) != len(dataset_specs)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in logical_workers)
    ):
        raise ValueError("logical_workers needs one nonnegative legacy worker count per catalog source")
    parquet_handle_cache_size = int(media_values.get("parquet_handle_cache_size", parquet_handle_cache_size))
    loaders: dict[str, dict[str, Any]] = {}
    from cosmos_framework.data.generator.action.unified_action128 import build_action128_spec

    action128_roots = []
    for spec in dataset_specs:
        row = ParquetCatalogDataset(spec["path"])[0]
        if row["type"] == "action" and row["data"].get("adapter") == "unified_action128":
            action128_roots.append(row["data"]["root"])
    action128_spec = build_action128_spec(action128_roots) if action128_roots else None
    # ``num_workers`` is a budget for the whole joint loader.  Replicating it
    # for every stream multiplies worker processes and tensor-sharing file
    # descriptors by the dataset count, which is especially costly for packed
    # VLM/video batches.
    worker_base, worker_remainder = divmod(max(0, num_workers), max(1, len(dataset_specs)))
    for position, spec in enumerate(dataset_specs):
        path = spec.get("path")
        ratio = spec.get("ratio")
        if not isinstance(path, str) or isinstance(ratio, bool) or not isinstance(ratio, int) or ratio <= 0:
            raise ValueError(f"invalid catalog dataset spec: {spec!r}")
        catalog = ParquetCatalogDataset(path, handle_cache_size=parquet_handle_cache_size)
        first = catalog[0]
        offline_lineage = spec.get("_offline_vae_lineage")
        if offline_lineage is not None and offline_lineage.get("dataset") != first["dataset"]:
            raise ValueError(
                f"offline VAE manifest dataset={offline_lineage.get('dataset')!r} does not match "
                f"catalog dataset={first['dataset']!r} for {path}"
            )
        task_type = first["type"]
        if offline_lineage is not None and task_type != "visual_generation":
            raise ValueError(
                "offline VAE mode supports pure visual_generation catalogs only; "
                f"catalog {path} starts with task type {task_type!r}"
            )
        try:
            task_components = components[task_type]
            batcher = task_components["batcher"]
            collator = task_components["collator"]
        except KeyError as error:
            raise ValueError(f"missing dataflow component {error} for task {task_type!r}") from error
        if task_components.get("fixed_profile_available") is False:
            raise ValueError(
                f"catalog {path!r} dataset {first['dataset']!r} task {task_type!r} "
                "has no configured fixed-sequence profile; "
                "configure understanding_tokens for Understanding or und_tokens/gen_tokens for Generation/Action"
            )
        name = f"{first['dataset']}:{position}"
        if task_type == "action":
            is_offline_catalog = first["data"].get("adapter") == "droid_offline"
            if is_offline_catalog != (getattr(collator, "latent_reader", None) is not None):
                raise ValueError("Action catalog adapter and visual_generation.vae_mode must agree")
            dataset = build_action_dataset_from_catalog(
                path,
                action_video_fps_default=action_video_fps_default,
                action_use_state=action_use_state,
                tokenizer_config=tokenizer_config,
                parquet_handle_cache_size=parquet_handle_cache_size,
                action128_spec=action128_spec,
                seed=seed,
                media_config=media_config,
            )
            expected_hash = task_components.get("expected_vae_weights_sha256")
            if is_offline_catalog and expected_hash and dataset.manifest["vae_weights_sha256"] != expected_hash:
                raise ValueError("Offline Action VAE weights SHA256 does not match the configured expectation")
            fingerprint = getattr(dataset, "fingerprint", catalog.fingerprint)
            distributor = EpisodeBlockDistributor(
                dataset,
                seed=seed,
                name=name,
                dataset_fingerprint=fingerprint,
                fingerprint_suffix=f"first-frame-color-jitter-v1:{seed}" if first_frame_color_jitter else None,
                media_config=media_config,
            )
            processor: RawItemProcessor = task_components.get("processor", IdentityProcessor())
            if first_frame_color_jitter:
                if is_offline_catalog:
                    raise ValueError("first_frame_color_jitter requires online RGB pixels")
                from cosmos_framework.data.generator.first_frame_color_jitter import FirstFrameColorJitterProcessor

                processor = FirstFrameColorJitterProcessor(processor, seed=seed)
        else:
            distributor = ParquetBlockDistributor(catalog, seed=seed, name=name)
            processor = task_components.get("processor")
            if not isinstance(processor, RawItemProcessor):
                raise TypeError(f"task {task_type!r} requires a RawItemProcessor")
            if offline_lineage is not None:
                processor = OfflineVaeLineageProcessor(
                    processor,
                    expected_dataset=str(offline_lineage["dataset"]),
                )
            processor = DropFailedMediaProcessor(processor)
        stream_workers = worker_base + int(position < worker_remainder)
        if logical_workers is not None:
            stream_workers = logical_workers[position]
        flow_kwargs = dict(
            distributor=distributor,
            processor=processor,
            batcher=batcher,
            collator=collator,
            enable_map_style_resume=enable_map_style_resume,
            map_style_resume_mode=map_style_resume_mode,
        )
        if execution_backend == "shared_prefetch":
            from cosmos_framework.data.generator.dataflow.loader import (
                _DataflowIterableDataset,
                data_parallel_coordinates,
            )
            from cosmos_framework.data.generator.dataflow.shared_prefetch import CosmosDataStream

            dp_rank, dp_world_size = data_parallel_coordinates(parallel_dims)
            stream = CosmosDataStream(
                _DataflowIterableDataset(**flow_kwargs, dp_rank=dp_rank, dp_world_size=dp_world_size), stream_workers
            )
        else:
            stream = CosmosDataLoader(
                **flow_kwargs,
                num_workers=stream_workers,
                prefetch_factor=prefetch_factor if stream_workers > 0 else None,
                persistent_workers=stream_workers > 0,
                pin_memory=pin_memory,
                parallel_dims=parallel_dims,
            )
        loaders[name] = {"dataloader": stream, "ratio": ratio}
    if execution_backend == "shared_prefetch":
        from cosmos_framework.data.generator.dataflow.shared_prefetch import SharedPrefetchDataLoader

        return SharedPrefetchDataLoader(
            loaders,
            seed=seed,
            selection_mode="exact_periodic",
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if prefetch_factor is not None else 1,
            pin_memory=pin_memory,
            receive_in_background=receive_in_background,
            worker_assignment=worker_assignment,
            worker_timeout=worker_timeout,
            microbatch_timeout_sec=microbatch_timeout_sec,
            microbatch_timeout_max_skips=microbatch_timeout_max_skips,
            microbatch_skip_log_dir=microbatch_skip_log_dir,
        )
    return JointCosmosDataLoader(loaders, seed=seed, selection_mode="exact_periodic")


def build_catalog_training_dataloader(
    dataset_specs: list[dict[str, Any]],
    *,
    processor: Any,
    action_video_fps_default: float | None = None,
    action_use_state: bool | None = None,
    first_frame_color_jitter: bool = False,
    conditioning_config: dict[int | str, float] | None = None,
    cfg_drop_ratio: float = 0.0,
    vae_mode: str = "online",
    expected_latent_channels: int = 48,
    expected_vae_name: str = "wan2pt2_tokenizer",
    expected_vae_weights_sha256: str | None = None,
    seed: int = 42,
    max_sequence_length: int = 45_056,
    max_samples_per_batch: int | None = None,
    packing_pool_size: int = 16,
    max_caption_tokens: int = 2_048,
    resolution: str | int = "256",
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    execution_backend: str = "shared_prefetch",
    logical_workers: list[int] | None = None,
    worker_assignment: list[int] | None = None,
    worker_timeout: float = 300.0,
    microbatch_timeout_sec: float = 0.0,
    microbatch_timeout_max_skips: int = 2,
    microbatch_skip_log_dir: str | None = None,
    pin_memory: bool = False,
    receive_in_background: bool = True,
    parallel_dims: Any = None,
    media_config: dict[str, Any] | None = None,
    static_packing_config: dict[str, Any] | None = None,
    fixed_sequence_length_enabled: bool = False,
    fixed_understanding_tokens: int = 0,
    fixed_und_tokens: int = 0,
    fixed_gen_tokens: int = 0,
    fixed_action_tokens: int = 0,
    enable_map_style_resume: bool = True,
    map_style_resume_mode: str = "exact",
) -> JointCosmosDataLoader:
    """Build every supported training task from the structured TOML surface."""

    if vae_mode not in {"online", "offline"}:
        raise ValueError(f"vae_mode must be 'online' or 'offline', got {vae_mode!r}")
    dataset_specs = [dict(spec) for spec in dataset_specs]
    manifest_specs = [spec for spec in dataset_specs if spec.get("vae_manifest_path")]
    if vae_mode == "offline" and manifest_specs:
        lineages = validate_offline_vae_manifests(
            manifest_specs,
            expected_vae_name=expected_vae_name,
            expected_vae_weights_sha256=expected_vae_weights_sha256,
        )
        for spec, lineage in zip(manifest_specs, lineages, strict=True):
            spec["_offline_vae_lineage"] = lineage
        lineage = lineages[0]
        log.info(
            "Offline VAE preflight passed for "
            f"{len(lineages)} dataset(s): vae_name={lineage['vae_name']}, "
            f"vae_tag={lineage['vae_tag']}, vae_weights_sha256={lineage['vae_weights_sha256']}"
        )

    media_values = dict(media_config or {})
    packing_config = StaticPackingConfig(**static_packing_config) if static_packing_config is not None else None
    packing_geometry = packing_config or StaticPackingConfig()
    special_tokens: dict[str, int] | None = None
    static_collator_kwargs: dict[str, Any] = {}
    if packing_config is not None:
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer, special_tokens = add_special_tokens(tokenizer)
        special_tokens["eos_token_id"] = int(tokenizer.eos_token_id)
        static_collator_kwargs = {
            "static_packing_config": packing_config,
            "special_tokens": special_tokens,
        }
    latent_reader = (
        SafetensorsLatentReader(
            handle_cache_size=int(media_values.get("safetensors_handle_cache_size", 8)),
            max_tensor_bytes=int(media_values.get("max_latent_tensor_bytes", 1_073_741_824)),
        )
        if vae_mode == "offline"
        else None
    )

    def _understanding_sample_size(sample: dict[str, Any]) -> int:
        from cosmos_framework.data.generator.sequence_packing.sequence import packed_text_span_length

        if special_tokens is None:
            tokens = sample.get("text_token_ids", sample.get("input_ids"))
            if tokens is None:
                raise KeyError("Understanding packing requires text_token_ids or input_ids")
            return len(tokens)
        plan = sample.get("sequence_plan")
        while isinstance(plan, list) and len(plan) == 1:
            plan = plan[0]
        if plan is None:
            raise ValueError("Understanding fixed admission requires sequence_plan")
        plan_get = plan.get if isinstance(plan, dict) else lambda key, default=None: getattr(plan, key, default)
        tokens = sample["text_token_ids"]
        while isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], (list, torch.Tensor)):
            tokens = tokens[0]
        num_tokens = int(tokens.numel()) if isinstance(tokens, torch.Tensor) else len(tokens)
        return packed_text_span_length(
            num_tokens,
            special_tokens,
            has_generation=False,
            supervise_text=bool(plan_get("supervise_text", True)),
            text_loss_mask=plan_get("text_loss_mask"),
        )

    understanding_capacity = (
        fixed_understanding_tokens - 1
        if fixed_sequence_length_enabled and fixed_understanding_tokens > 0
        else max_sequence_length
    )
    understanding_batcher = (
        SequentialTokenBudgetBatcher(
            max_tokens=understanding_capacity,
            size_fn=_understanding_sample_size,
            max_samples_per_batch=max_samples_per_batch,
            task_type="understanding",
            physical_target=fixed_understanding_tokens,
        )
        if fixed_sequence_length_enabled
        else PoolPackingBatcher(
            max_tokens=understanding_capacity,
            pool_size=packing_pool_size,
            max_batch_size=max_samples_per_batch or 1,
        )
    )
    # Parquet distributors are resumable map-style streams. A value above one
    # lets the batcher bypass an overflowing sample and emit a later position,
    # which cannot be represented by the per-worker checkpoint watermark.
    # One still packs consecutive samples up to the configured budget, but
    # carries the first overflow into the next batch without reordering.
    vfm_batcher = (
        SequentialPackingBatcher(
            max_sequence_length=max_sequence_length,
            max_samples_per_batch=max_samples_per_batch,
            tokenizer_spatial_compression_factor=packing_geometry.spatial_compression_factor,
            tokenizer_temporal_compression_factor=packing_geometry.temporal_compression_factor,
            patch_spatial=packing_geometry.latent_patch_size,
            lookahead_limit=1,
            fixed_sequence_length_enabled=True,
            fixed_understanding_tokens=fixed_understanding_tokens,
            fixed_und_tokens=fixed_und_tokens,
            fixed_gen_tokens=fixed_gen_tokens,
            fixed_action_tokens=fixed_action_tokens,
            special_tokens=special_tokens,
        )
        if fixed_sequence_length_enabled
        else SequentialPackingBatcher(
            max_samples_per_batch=max_samples_per_batch,
            tokenizer_spatial_compression_factor=packing_geometry.spatial_compression_factor,
            tokenizer_temporal_compression_factor=packing_geometry.temporal_compression_factor,
            patch_spatial=packing_geometry.latent_patch_size,
            lookahead_limit=1,
            fixed_sequence_length_enabled=fixed_sequence_length_enabled,
            fixed_understanding_tokens=fixed_understanding_tokens,
            fixed_und_tokens=fixed_und_tokens,
            fixed_gen_tokens=fixed_gen_tokens,
            fixed_action_tokens=fixed_action_tokens,
            special_tokens=special_tokens,
        )
        if max_samples_per_batch is not None
        else SequentialPackingBatcher(
            max_sequence_length=max_sequence_length,
            tokenizer_spatial_compression_factor=packing_geometry.spatial_compression_factor,
            tokenizer_temporal_compression_factor=packing_geometry.temporal_compression_factor,
            patch_spatial=packing_geometry.latent_patch_size,
            lookahead_limit=1,
            fixed_sequence_length_enabled=fixed_sequence_length_enabled,
            fixed_understanding_tokens=fixed_understanding_tokens,
            fixed_und_tokens=fixed_und_tokens,
            fixed_gen_tokens=fixed_gen_tokens,
            fixed_action_tokens=fixed_action_tokens,
            special_tokens=special_tokens,
        )
    )

    components = {
        "understanding": {
            "processor": CatalogUnderstandingProcessor(
                processor,
                video_decoder=build_video_decoder_from_config(media_config),
            ),
            "batcher": understanding_batcher,
            "collator": VFMListCollator(**static_collator_kwargs),
            "fixed_profile_available": not fixed_sequence_length_enabled or fixed_understanding_tokens > 1,
        },
        "visual_generation": {
            "processor": CatalogVisualGenerationProcessor(
                processor,
                video_decoder=build_video_decoder_from_config(media_config),
                resolution=resolution,
                max_caption_tokens=max_caption_tokens,
                conditioning_config=conditioning_config,
                cfg_drop_ratio=cfg_drop_ratio,
                seed=seed,
                vae_mode=vae_mode,
                expected_latent_channels=expected_latent_channels,
                max_latent_tensor_bytes=int(media_values.get("max_latent_tensor_bytes", 1_073_741_824)),
            ),
            "batcher": vfm_batcher,
            "collator": VFMListCollator(latent_reader=latent_reader, **static_collator_kwargs),
            "fixed_profile_available": not fixed_sequence_length_enabled
            or (fixed_und_tokens > 1 and fixed_gen_tokens > 1),
        },
        "action": {
            "batcher": vfm_batcher,
            "collator": VFMListCollator(latent_reader=latent_reader, **static_collator_kwargs),
            "expected_vae_weights_sha256": expected_vae_weights_sha256,
            "fixed_profile_available": not fixed_sequence_length_enabled
            or (fixed_und_tokens > 1 and fixed_gen_tokens > 1),
        },
    }
    if vae_mode == "offline":
        # The run-level offline contract also applies without producer manifests.
        components = {key: components[key] for key in ("visual_generation", "action")}
    return build_catalog_joint_dataloader(
        dataset_specs,
        action_video_fps_default=action_video_fps_default,
        action_use_state=action_use_state,
        first_frame_color_jitter=first_frame_color_jitter,
        components=components,
        seed=seed,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        execution_backend=execution_backend,
        logical_workers=logical_workers,
        worker_assignment=worker_assignment,
        worker_timeout=worker_timeout,
        microbatch_timeout_sec=microbatch_timeout_sec,
        microbatch_timeout_max_skips=microbatch_timeout_max_skips,
        microbatch_skip_log_dir=microbatch_skip_log_dir,
        pin_memory=pin_memory,
        receive_in_background=receive_in_background,
        parallel_dims=parallel_dims,
        tokenizer_config=processor,
        media_config=media_config,
        enable_map_style_resume=enable_map_style_resume,
        map_style_resume_mode=map_style_resume_mode,
    )


__all__ = [
    "CatalogMediaMaterializer",
    "CatalogUnderstandingProcessor",
    "CatalogVisualGenerationProcessor",
    "DropFailedMediaProcessor",
    "build_action_dataset_from_catalog",
    "build_catalog_joint_dataloader",
    "build_catalog_training_dataloader",
    "build_video_decoder_from_config",
]
