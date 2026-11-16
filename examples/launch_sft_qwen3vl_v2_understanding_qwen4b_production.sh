#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Production supervisor for the frozen V2 Qwen3-VL-4B Understanding recipe.
# It leaves model/trainer source untouched, saves every 500 steps by default,
# retains the latest completed DCP plus evaluation milestones, and records a
# 20-minute status log.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_LAUNCHER="$REPO_ROOT/examples/launch_sft_qwen3vl_v2_understanding_qwen4b.sh"
RETENTION_SCRIPT="$REPO_ROOT/examples/retain_latest_dcp_checkpoint.sh"

: "${OUTPUT_ROOT:=$REPO_ROOT/outputs/train/qwen3vl_v2_0a_understanding_qwen4b_production}"
: "${JOB_NAME:=qwen3vl_v2_0a_understanding_qwen4b_production}"
: "${CHECKPOINT_SAVE_ITER:=500}"
: "${RETENTION_POLL_SECONDS:=60}"
: "${CHECKPOINT_KEEP_STEPS:=1500,2000,3000,5000,8000,10000,20000,40000}"
: "${MONITOR_INTERVAL_SECONDS:=1200}"
: "${SUPERVISOR_POLL_SECONDS:=30}"
: "${LOG_FILENAME:=qwen3vl_v2_0a_understanding_qwen4b_production.log}"
: "${ALLOW_RESUME:=0}"

OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
[[ "$OUTPUT_ROOT" == /home/* && "$OUTPUT_ROOT" != /home ]] || {
    echo "ERROR: OUTPUT_ROOT must be a scoped directory below /home: $OUTPUT_ROOT" >&2
    exit 2
}
[[ "$JOB_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: JOB_NAME contains unsupported characters: $JOB_NAME" >&2
    exit 2
}
for value_name in CHECKPOINT_SAVE_ITER RETENTION_POLL_SECONDS MONITOR_INTERVAL_SECONDS SUPERVISOR_POLL_SECONDS; do
    value="${!value_name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: $value_name must be a positive integer: $value" >&2
        exit 2
    }
done
[[ "$ALLOW_RESUME" == "0" || "$ALLOW_RESUME" == "1" ]] || {
    echo "ERROR: ALLOW_RESUME must be 0 or 1" >&2
    exit 2
}
[[ "$CHECKPOINT_KEEP_STEPS" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || {
    echo "ERROR: CHECKPOINT_KEEP_STEPS must be comma-separated non-negative integers: $CHECKPOINT_KEEP_STEPS" >&2
    exit 2
}

JOB_DIR="$OUTPUT_ROOT/cosmos3/understanding_verification/$JOB_NAME"
CHECKPOINT_DIR="$JOB_DIR/checkpoints"
LATEST_POINTER="$CHECKPOINT_DIR/latest_checkpoint.txt"
if [[ -f "$LATEST_POINTER" && "$ALLOW_RESUME" != "1" ]]; then
    echo "ERROR: resumable checkpoint already exists; set ALLOW_RESUME=1 to resume: $LATEST_POINTER" >&2
    exit 2
fi
if [[ -d "$JOB_DIR" && ! -f "$LATEST_POINTER" && -n "$(find "$JOB_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: non-empty job directory has no resumable checkpoint; choose a new OUTPUT_ROOT/JOB_NAME: $JOB_DIR" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs"
TRAINING_LOG="$OUTPUT_ROOT/logs/$LOG_FILENAME"
RETENTION_LOG="$OUTPUT_ROOT/logs/checkpoint_retention.log"
MONITOR_LOG="$OUTPUT_ROOT/logs/monitor_20m.log"

export PATH="$REPO_ROOT/.venv/bin:/usr/local/bin:/usr/bin:/bin"
export OUTPUT_ROOT LOG_FILENAME
EXTRA_TAIL_OVERRIDES="${EXTRA_TAIL_OVERRIDES:+$EXTRA_TAIL_OVERRIDES }checkpoint.save_iter=$CHECKPOINT_SAVE_ITER job.name=$JOB_NAME"
export EXTRA_TAIL_OVERRIDES

training_pid=""
retention_pid=""

stop_process_group() {
    local pid="$1"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || true
    fi
}

cleanup() {
    stop_process_group "$training_pid"
    stop_process_group "$retention_pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

snapshot() {
    {
        echo ">>> $(date -u '+%Y-%m-%dT%H:%M:%SZ') production status"
        echo "training_pid=${training_pid:-none}"
        if [[ -f "$LATEST_POINTER" ]]; then
            echo -n "latest_checkpoint="
            sed -n '1p' "$LATEST_POINTER"
        else
            echo "latest_checkpoint=none"
        fi
        if [[ -f "$TRAINING_LOG" ]]; then
            grep -E "Iteration [0-9]+:|Done with training|Traceback|RuntimeError|AssertionError|\|ERROR\|" \
                "$TRAINING_LOG" | tail -8 || true
        fi
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
        df -h "$OUTPUT_ROOT" | tail -1 || true
    } >> "$MONITOR_LOG"
}

echo ">>> $(date -u '+%Y-%m-%dT%H:%M:%SZ') Starting retention: keep latest plus ${CHECKPOINT_KEEP_STEPS}, poll ${RETENTION_POLL_SECONDS}s" \
    >> "$RETENTION_LOG"
setsid env CHECKPOINT_RETENTION_ROOT="$OUTPUT_ROOT" \
    bash "$RETENTION_SCRIPT" "$CHECKPOINT_DIR" "$RETENTION_POLL_SECONDS" "$CHECKPOINT_KEEP_STEPS" \
    >> "$RETENTION_LOG" 2>&1 &
retention_pid=$!

setsid env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    bash "$BASE_LAUNCHER" &
training_pid=$!

echo ">>> $(date -u '+%Y-%m-%dT%H:%M:%SZ') Production supervisor started"
echo ">>> output_root=$OUTPUT_ROOT"
echo ">>> job_name=$JOB_NAME"
echo ">>> checkpoint_save_iter=$CHECKPOINT_SAVE_ITER keep_latest=1 keep_steps=$CHECKPOINT_KEEP_STEPS"
echo ">>> monitor_interval_seconds=$MONITOR_INTERVAL_SECONDS"

last_snapshot=0
while kill -0 "$training_pid" 2>/dev/null; do
    if ! kill -0 "$retention_pid" 2>/dev/null; then
        echo "ERROR: checkpoint retention process exited; stopping training to prevent unbounded checkpoint growth" >&2
        stop_process_group "$training_pid"
        wait "$training_pid" 2>/dev/null || true
        exit 1
    fi

    now="$(date +%s)"
    if (( now - last_snapshot >= MONITOR_INTERVAL_SECONDS )); then
        snapshot
        last_snapshot="$now"
    fi
    sleep "$SUPERVISOR_POLL_SECONDS"
done

set +e
wait "$training_pid"
training_status=$?
set -e
training_pid=""

stop_process_group "$retention_pid"
wait "$retention_pid" 2>/dev/null || true
retention_pid=""

if ! RETENTION_ONCE=1 CHECKPOINT_RETENTION_ROOT="$OUTPUT_ROOT" \
    bash "$RETENTION_SCRIPT" "$CHECKPOINT_DIR" "$RETENTION_POLL_SECONDS" "$CHECKPOINT_KEEP_STEPS" \
    >> "$RETENTION_LOG" 2>&1; then
    echo "ERROR: final checkpoint retention pass failed" >&2
    training_status=1
fi
snapshot

echo ">>> $(date -u '+%Y-%m-%dT%H:%M:%SZ') Production supervisor exited with status $training_status"
exit "$training_status"
