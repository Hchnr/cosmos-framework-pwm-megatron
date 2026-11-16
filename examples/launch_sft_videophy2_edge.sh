#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Packed Understanding SFT with the unified Cosmos3-Edge Omni/MoT model.
# The language model and native Edge visual tower are restored from one DCP.
# Pure Understanding does not load Wan VAE.

TOML_FILE="examples/toml/sft_config/videophy2_sft_edge.toml"
: "${DATASET_PATH:?Set DATASET_PATH to an Understanding Parquet catalog}"
: "${BASE_CHECKPOINT_PATH:=/share/project/liyue/model_ckpts/Cosmos3-Edge-DCP}"
: "${HF_MODEL_PATH:=/share/project/liyue/model_ckpts/Cosmos3-Edge}"
REQUIRE_WAN_VAE=0

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

# Optional local Edge snapshot for processor files; weights still come from DCP.
TAIL_OVERRIDES+=(
    "model.config.vlm_config.tokenizer.repository=null"
    "model.config.vlm_config.tokenizer.revision=null"
    "+model.config.vlm_config.tokenizer.tokenizer_type=$HF_MODEL_PATH"
)
if [[ "${SMOKE_RUN:-0}" == "1" ]]; then
    TAIL_OVERRIDES+=(
        "job.name=videophy2_sft_edge_smoke"
        "trainer.max_iter=2"
        "trainer.grad_accum_iter=1"
        "checkpoint.save_iter=2"
    )
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
