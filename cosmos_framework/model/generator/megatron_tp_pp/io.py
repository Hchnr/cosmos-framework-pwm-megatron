# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Production first/last-stage modules sharing the native network mathematics."""

import torch
from torch import nn

from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq
from cosmos_framework.model.generator.mot.action128 import Action128DomainLinear, Action128Normalization
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder, has_noisy_tokens
from cosmos_framework.model.generator.mot.small_action_tower import (
    action_shape,
    compact_pack,
    expand_pack,
    small_decoder_inputs,
)
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes

from .layers import TensorParallelLinear
from .payload import STREAMS, PayloadLayout


class _NetworkIO(nn.Module):
    # Reuse the native methods rather than maintaining a second packing/loss
    # implementation. This class owns no decoder blocks and no frozen VAE.
    prepare_decoder_inputs = Cosmos3VFMNetwork.prepare_decoder_inputs
    patchify_and_pack_latents = Cosmos3VFMNetwork.patchify_and_pack_latents
    unpatchify_and_unpack_latents = Cosmos3VFMNetwork.unpatchify_and_unpack_latents
    pack_action = Cosmos3VFMNetwork.pack_action
    unpack_action = Cosmos3VFMNetwork.unpack_action
    _embed_packed_timesteps = Cosmos3VFMNetwork._embed_packed_timesteps
    _encode_text = Cosmos3VFMNetwork._encode_text
    _encode_vision = Cosmos3VFMNetwork._encode_vision
    _encode_action = Cosmos3VFMNetwork._encode_action
    _decode_vision = Cosmos3VFMNetwork._decode_vision
    _decode_action = Cosmos3VFMNetwork._decode_action
    decode_decoder_outputs = Cosmos3VFMNetwork.decode_decoder_outputs

    def __init__(self, config, text_config, shape):
        super().__init__()
        text_config = getattr(text_config, "text_config", text_config)
        if config.tower_structure != "three_tower" or config.sound_gen or config.video_temporal_causal:
            raise ValueError("TP/PP I/O requires the planned dense three-tower configuration")
        self.config, self.text_config, self.shape = config, text_config, shape
        self.hidden_size, self.action_hidden_size = text_config.hidden_size, shape.hidden
        self.num_heads, self.head_dim = text_config.num_attention_heads, text_config.head_dim
        self.num_hidden_layers = text_config.num_hidden_layers
        self.latent_patch_size, self.latent_channel = config.latent_patch_size, config.latent_channel_size
        self.patch_latent_dim = config.latent_patch_size**2 * config.latent_channel_size
        self.timestep_scale = config.timestep_scale
        self.action_dim = config.action_dim
        self.parallel_dims, self.flex_backend = None, None
        self.attention_io_layout = "sequence_sharded"
        self.natten_parameter_list = config.natten_parameter_list
        self.video_temporal_causal, self.pad_for_cuda_graphs = False, False
        self.predict_text_tokens = config.predict_text_tokens
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()

    def _forward_packed_vit(self, *args, **kwargs):
        raise ValueError("The TP/PP action recipe requires load_vlm_visual=false")


class FirstStageInputs(_NetworkIO):
    def __init__(self, config, text_config, mcore_config, *, shape=None, vocab_parallel=True):
        shape = shape or action_shape("width_only_1b")
        super().__init__(config, text_config, shape)
        text_config = self.text_config
        with torch.device("meta"):
            self.time_embedder = TimestepEmbedder(self.hidden_size, bias=config.enable_input_bias)
            self.action_time_embedder = TimestepEmbedder(shape.hidden, bias=config.enable_input_bias)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size, bias=config.enable_input_bias)
            self.action2llm = Action128DomainLinear(128, shape.hidden, 64)
            self.action_modality_embed = nn.Parameter(torch.empty(shape.hidden))
            self.action_normalization = Action128Normalization(
                config.action128_spec, config.vision_spatial_position_mode
            )
            if not vocab_parallel:
                self.language_model.model.embed_tokens = nn.Embedding(text_config.vocab_size, self.hidden_size)
        if vocab_parallel:
            from megatron.core.tensor_parallel.layers import VocabParallelEmbedding

            self.language_model.model.embed_tokens = VocabParallelEmbedding(
                text_config.vocab_size,
                self.hidden_size,
                init_method=lambda t: t,
                config=mcore_config,
            )
        self.language_model.model.embed_tokens.weight.requires_grad_(False)
        self.language_model.model.rotary_emb = LayerTypes("qwen3_vl_dense").rotary_embedding(text_config)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        device = self.vae2llm.weight.device
        if device.type != "meta":
            self.language_model.model.rotary_emb.init_weights(buffer_device=device)
            for embedder in (self.time_embedder, self.action_time_embedder):
                embedder._timestep_frequencies = embedder._build_timestep_frequencies(
                    embedder.frequency_embedding_size,
                    max_period=10000,
                    device=device,
                )
        return self

    def forward(self, batch):
        sequence = batch["packed_sequence"]
        batch["decoder_has_loss"] = (
            has_noisy_tokens(sequence.vision)
            or has_noisy_tokens(sequence.action)
            or sequence.ce_loss_indexes is not None
            and sequence.ce_loss_indexes.numel() > 0
        )
        prepared = self.prepare_decoder_inputs(batch["packed_sequence"], vlm_inputs=batch.get("vlm_inputs"))
        if prepared["deepstack_und"]:
            raise ValueError("TP/PP action recipe does not use a visual reasoner/deepstack")
        pack, positions = prepared["input_pack"], prepared["position_ids"]
        cos, sin = self.language_model.model.rotary_emb(
            pack["full_only_seq"],
            positions.unsqueeze(0) if positions.ndim == 1 else positions.unsqueeze(1),
        )
        rope = tuple(from_all_seq(t.squeeze(0), pack) for t in (cos, sin))
        capacities = None
        if self.config.fixed_sequence_length_enabled:
            capacities = (
                self.config.fixed_und_tokens,
                self.config.fixed_gen_tokens - self.config.fixed_action_tokens,
                self.config.fixed_action_tokens,
            )
        pack = compact_pack(
            pack,
            rope,
            positions,
            self.shape,
            fixed_lengths=capacities,
            attention_backend=self.config.action_tower_attention_backend,
            und_short_tokens=0 if getattr(self, "_warm_full_und", False) else self.config.action_tower_und_short_tokens,
        )
        self._warm_full_und = False
        # These exact tiny activations are needed by native dummy heads when a
        # modality has no targets. Carry their graph to the last stage, keeping
        # every input parameter exclusively on the first stage.
        dummy_v = self.vae2llm(pack["full_only_seq"].new_zeros(1, self.patch_latent_dim))
        dummy_a = (
            self.action2llm(
                pack["action_seq"].new_zeros(1, 128), torch.zeros(1, dtype=torch.long, device=dummy_v.device)
            )
            + self.action_modality_embed
        )
        context = torch.cat((dummy_v, dummy_a), dim=-1)
        batch["restore_metadata"] = {
            key: value
            for key, value in pack.items()
            if key
            not in ("causal_seq", "full_only_seq", "action_seq", "_small_masks", "_small_rope", "_small_action_rope")
        }
        batch["prepared_network"] = {
            key: prepared[key] for key in ("original_latent_shapes", "sequence_shard_parallel_dims")
        }
        result = small_decoder_inputs(pack)
        batch["decoder_metadata"] = {key: pack[key] for key in ("_small_ids", "_small_rope", "_small_action_rope")}
        batch["attention_backend"] = self.config.action_tower_attention_backend
        batch["fixed_capacity"] = self.config.fixed_sequence_length_enabled
        batch["num_samples"] = len(pack["sample_offsets"]) - 1
        batch["layout"] = PayloadLayout(
            tuple(tuple(result[key].shape) for key in STREAMS), context_width=self.hidden_size + self.action_hidden_size
        )
        result["pipeline_context"] = context
        return result


class _OutputView:
    """Per-microbatch native decoder view; no mutable current-batch module state."""

    _decode_vision = Cosmos3VFMNetwork._decode_vision
    _decode_action = Cosmos3VFMNetwork._decode_action

    def __init__(self, adapter, context):
        self.adapter, self.context = adapter, context

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def vae2llm(self, x):
        return self.context[:, : self.adapter.hidden_size]

    def action2llm(self, x, domain_ids):
        return self.context[:, self.adapter.hidden_size :]

    @property
    def action_modality_embed(self):
        return self.context.new_zeros(self.adapter.action_hidden_size)


class LastStageOutputs(_NetworkIO):
    def __init__(self, config, text_config, mcore_config, *, shape=None, vocab_parallel=True):
        from megatron.core import parallel_state as ps

        shape = shape or action_shape("width_only_1b")
        super().__init__(config, text_config, shape)
        text_config = self.text_config
        norm = LayerTypes("qwen3_vl_dense").rms_norm
        with torch.device("meta"):
            self.language_model.model.norm = norm(self.hidden_size, eps=text_config.rms_norm_eps)
            self.language_model.model.norm_moe_gen = norm(self.hidden_size, eps=text_config.rms_norm_eps)
            self.language_model.model.norm_moe_gen_action = norm(shape.hidden, eps=text_config.rms_norm_eps)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.llm2action = Action128DomainLinear(shape.hidden, 128, 64)
            head = nn.Linear(self.hidden_size, text_config.vocab_size, bias=False)
        head.weight.requires_grad_(False)
        self.language_model.model.norm.weight.requires_grad_(False)
        self.language_model.lm_head = (
            TensorParallelLinear(
                head,
                mcore_config,
                column=True,
                tp_rank=ps.get_tensor_model_parallel_rank(),
                tp_size=mcore_config.tensor_model_parallel_size,
            )
            if vocab_parallel
            else head
        )

    def forward(self, pack, batch):
        context = pack["pipeline_context"]
        packed_outputs = expand_pack({**batch["restore_metadata"], **pack}, self.language_model.model)
        sequence = batch["packed_sequence"]
        if self.predict_text_tokens or (sequence.ce_loss_indexes is not None and sequence.ce_loss_indexes.numel()):
            raise ValueError("Nonempty text CE requires a vocab-parallel loss adapter")
        # Native decode is reused for real targets and dummy outputs alike.
        view = _OutputView(self, context)
        outputs = Cosmos3VFMNetwork.decode_decoder_outputs(
            view,
            sequence,
            packed_outputs,
            {},
            batch["prepared_network"],
        )
        if "ce_preds" in outputs and outputs["ce_preds"].numel() == 0:
            outputs["ce_preds"] = outputs["ce_preds"].new_empty((0, self.text_config.vocab_size))
        return outputs
