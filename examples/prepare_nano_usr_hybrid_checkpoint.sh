#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Run once before the distributed training job. Never overwrite either source.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f /root/.venv/bin/activate ]]; then
        source /root/.venv/bin/activate
    elif [[ -f /opt/venv/bin/activate ]]; then
        source /opt/venv/bin/activate
    fi
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_OVERRIDE:-}"
GENERATION_CHECKPOINT_PATH="${GENERATION_CHECKPOINT_PATH:-/share/project/eai_pwm/models/Cosmos3-Nano-DCP}"
UNDERSTANDING_CHECKPOINT_PATH="${UNDERSTANDING_CHECKPOINT_PATH:-/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct-DCP}"
HYBRID_CHECKPOINT_PATH="${HYBRID_CHECKPOINT_PATH:-/share/project/eai_pwm/models/Cosmos3-Nano-Qwen3-VL-8B-Instruct-Hybrid-DCP}"
python -m cosmos_framework.scripts.merge_understanding_dcp \
    --generation-checkpoint-path "$GENERATION_CHECKPOINT_PATH" \
    --understanding-checkpoint-path "$UNDERSTANDING_CHECKPOINT_PATH" \
    --output-path "$HYBRID_CHECKPOINT_PATH"
