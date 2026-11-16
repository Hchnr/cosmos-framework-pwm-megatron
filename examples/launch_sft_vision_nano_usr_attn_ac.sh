#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Opt-in FA3 attention-only SAC. Inherit the online recipe and all launch controls.
# Validated backend: split_default / flash_attn_3_nv; see tools/ac_attn/README.md.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${COSMOS_FA3_PROVIDER:-nv}" != "nv" ]]; then
    echo "Attention-only AC launcher requires COSMOS_FA3_PROVIDER=nv" >&2
    exit 1
fi
exec bash "$SCRIPT_DIR/launch_sft_vision_nano_usr_128gpu.sh" \
    model.config.joint_attn_backend=split_default \
    model.config.activation_checkpointing.mode=selective \
    'model.config.activation_checkpointing.save_ops_regex=["^_flash_attn_forward[.]default$"]' \
    "$@"
