#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Keep the latest completed DCP plus explicitly requested milestone DCPs.
# A directory newer than latest_checkpoint.txt may still be receiving an async
# save and is therefore never removed.

set -euo pipefail

usage() {
    echo "Usage: CHECKPOINT_RETENTION_ROOT=/home/... $0 <checkpoint-dir> [poll-seconds] [keep-steps]" >&2
}

if (( $# < 1 || $# > 3 )); then
    usage
    exit 2
fi

CHECKPOINT_DIR="$(realpath -m "$1")"
POLL_SECONDS="${2:-60}"
KEEP_STEPS="${3:-}"
ALLOWED_ROOT_INPUT="${CHECKPOINT_RETENTION_ROOT:-}"
RETENTION_ONCE="${RETENTION_ONCE:-0}"

[[ -n "$ALLOWED_ROOT_INPUT" ]] || {
    echo "ERROR: CHECKPOINT_RETENTION_ROOT is required" >&2
    exit 2
}
ALLOWED_ROOT="$(realpath -m "$ALLOWED_ROOT_INPUT")"

[[ "$ALLOWED_ROOT" == /home/* && "$ALLOWED_ROOT" != /home ]] || {
    echo "ERROR: retention root must be a scoped directory below /home: $ALLOWED_ROOT" >&2
    exit 2
}
[[ "$CHECKPOINT_DIR" == "$ALLOWED_ROOT"/* && "$(basename -- "$CHECKPOINT_DIR")" == "checkpoints" ]] || {
    echo "ERROR: checkpoint directory must be a checkpoints/ descendant of $ALLOWED_ROOT: $CHECKPOINT_DIR" >&2
    exit 2
}
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: poll-seconds must be a positive integer: $POLL_SECONDS" >&2
    exit 2
}
[[ "$RETENTION_ONCE" == "0" || "$RETENTION_ONCE" == "1" ]] || {
    echo "ERROR: RETENTION_ONCE must be 0 or 1" >&2
    exit 2
}

declare -A KEEP_ITERATIONS=()
if [[ -n "$KEEP_STEPS" ]]; then
    IFS=',' read -r -a keep_step_values <<< "$KEEP_STEPS"
    for keep_step in "${keep_step_values[@]}"; do
        [[ "$keep_step" =~ ^(0|[1-9][0-9]*)$ ]] || {
            echo "ERROR: keep-steps must be comma-separated non-negative integers: $KEEP_STEPS" >&2
            exit 2
        }
        KEEP_ITERATIONS[$((10#$keep_step))]=1
    done
fi

log() {
    echo ">>> $(date -u '+%Y-%m-%dT%H:%M:%SZ') $*"
}

checkpoint_is_complete() {
    local checkpoint_dir="$1"
    local state_key
    for state_key in model optim scheduler trainer; do
        [[ -f "$checkpoint_dir/$state_key/.metadata" ]] || return 1
    done
}

prune_once() {
    local latest_file="$CHECKPOINT_DIR/latest_checkpoint.txt"
    [[ -f "$latest_file" ]] || return 0

    local latest_name
    latest_name="$(sed -n '1{s/[[:space:]]*$//;p;q;}' "$latest_file")"
    if [[ ! "$latest_name" =~ ^iter_([0-9]{9})$ ]]; then
        log "Skipping retention: invalid latest pointer '$latest_name'"
        return 0
    fi
    local latest_iteration="${BASH_REMATCH[1]}"
    local latest_dir
    latest_dir="$(realpath -m "$CHECKPOINT_DIR/$latest_name")"
    if [[ "$latest_dir" != "$CHECKPOINT_DIR/$latest_name" ]] || ! checkpoint_is_complete "$latest_dir"; then
        log "Skipping retention: pointed checkpoint is absent or incomplete: $latest_name"
        return 0
    fi

    local removed=0
    local candidate candidate_name candidate_iteration candidate_path
    shopt -s nullglob
    for candidate in "$CHECKPOINT_DIR"/iter_[0-9]*; do
        [[ -d "$candidate" ]] || continue
        candidate_name="$(basename -- "$candidate")"
        [[ "$candidate_name" =~ ^iter_([0-9]{9})$ ]] || continue
        candidate_iteration="${BASH_REMATCH[1]}"

        # Keep the pointer target and any higher iteration that may still be
        # written by the asynchronous checkpointer.
        if (( 10#$candidate_iteration >= 10#$latest_iteration )); then
            continue
        fi
        if [[ -n "${KEEP_ITERATIONS[$((10#$candidate_iteration))]+x}" ]]; then
            continue
        fi

        candidate_path="$(realpath -m "$candidate")"
        [[ "$candidate_path" == "$CHECKPOINT_DIR/$candidate_name" ]] || {
            log "Refusing to remove non-canonical candidate: $candidate"
            continue
        }
        log "Removing superseded checkpoint: $candidate_name (latest: $latest_name)"
        rm -rf -- "$candidate_path"
        removed=$((removed + 1))
    done
    shopt -u nullglob

    if (( removed > 0 )); then
        log "Retention complete: removed $removed checkpoint(s), retained latest $latest_name and milestones ${KEEP_STEPS:-none}"
    fi
}

while true; do
    prune_once
    if [[ "$RETENTION_ONCE" == "1" ]]; then
        break
    fi
    sleep "$POLL_SECONDS"
done
