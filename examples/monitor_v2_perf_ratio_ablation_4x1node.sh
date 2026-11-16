#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Supervise four independent single-node jobs, retain DCPs, and recover ordinary failures.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TABLE="$REPO_ROOT/examples/v2_perf_ratio_ablation_runs.tsv"
LAUNCHER="$REPO_ROOT/examples/launch_v2_perf_ratio_ablation_4x1node.sh"
VALIDATOR="$REPO_ROOT/examples/validate_v2_perf_ratio_ablation.py"
RETENTION="$REPO_ROOT/examples/retain_v2_perf_ratio_ablation_checkpoint.py"
EXPECTED_BRANCH="ybai_0828/understanding_verification"
BASE_CHECKPOINT="/share/project/liyue/model_ckpts/Cosmos3-Nano-DCP"
GLOBAL_ALERT="$REPO_ROOT/outputs/train/v2_perf_ratio_ablation_4x1node_20260901_r01_ALERT"
POLL_SECONDS="${POLL_SECONDS:-60}"

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: POLL_SECONDS must be positive" >&2; exit 2; }

load_run() {
    local experiment="$1"
    local row
    row="$(awk -F "|" -v experiment="$experiment" '$1 == experiment {print; found++} END {if (found != 1) exit 3}' "$RUN_TABLE")" || return 2
    IFS='|' read -r RUN_EXPERIMENT RUN_HOST RUN_TOML RUN_JOB RUN_OUTPUT RUN_PORT RUN_SESSION <<< "$row"
    RUN_MODEL_DIR="$RUN_OUTPUT/cosmos3/understanding_verification/$RUN_JOB"
    RUN_CHECKPOINT_DIR="$RUN_MODEL_DIR/checkpoints"
}

remote_shell() {
    local host="$1"
    local command="$2"
    if [[ "$host" == "local" ]]; then
        bash -lc "$command"
    else
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "$command"
    fi
}

log_task() {
    local message="$1"
    printf ">>> %s [%s] %s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$RUN_EXPERIMENT" "$message" | tee -a "$RUN_OUTPUT/logs/monitor.log"
}

alert_task() {
    local message="$1"
    local blocked="$RUN_OUTPUT/monitor_state/blocked"
    if [[ ! -f "$blocked" ]]; then
        printf "%s\n" "$message" > "$blocked"
    fi
    printf ">>> %s [%s] %s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$RUN_EXPERIMENT" "$message" >> "$GLOBAL_ALERT"
    log_task "BLOCKED $message"
}

latest_step_from_log() {
    local log_file="$1"
    [[ -f "$log_file" ]] || { echo 0; return; }
    grep -Eo 'Iteration [0-9]+:|\[RANK 0\] [0-9]+ : iter_speed|global_id=[0-9]+' "$log_file" 2>/dev/null |
        grep -Eo '[0-9]+' | sort -n | tail -n 1
}

current_attempt_log() {
    local pointer="$RUN_OUTPUT/monitor_state/current_attempt_log"
    if [[ -f "$pointer" ]]; then sed -n '1p' "$pointer"; else echo "$RUN_OUTPUT/logs/train.log"; fi
}

latest_complete_checkpoint() {
    local latest_file="$RUN_CHECKPOINT_DIR/latest_checkpoint.txt"
    [[ -f "$latest_file" ]] || return 1
    local latest_name check_output
    latest_name="$(sed -n '1{s/[[:space:]]*$//;p;q;}' "$latest_file")"
    [[ "$latest_name" =~ ^iter_[0-9]{9}$ ]] || return 1
    check_output="$("$REPO_ROOT/.venv/bin/python" "$RETENTION" "$RUN_EXPERIMENT" --check-only 2>&1)" || return 1
    grep -q "CHECK complete latest=$latest_name" <<< "$check_output" || return 1
    printf "%s\n" "$RUN_CHECKPOINT_DIR/$latest_name"
}

check_config_drift() {
    local manifest="$RUN_OUTPUT/run_manifest.yaml"
    [[ -f "$manifest" ]] || return 1
    local expected_sha actual_sha
    expected_sha="$(awk '/^toml_sha256:/ {print $2; exit}' "$manifest")"
    actual_sha="$(sha256sum -- "$RUN_TOML" | awk '{print $1}')"
    [[ -n "$expected_sha" && "$expected_sha" == "$actual_sha" ]]
}

blocking_error() {
    local attempt_log="$1"
    [[ -f "$attempt_log" ]] || return 1
    grep -Eiq 'Loss:[[:space:]]*(-?nan|[-+]?inf)|loss[^[:alnum:]]*(is|=|:)?[[:space:]]*(-?nan|[-+]?inf)|CUDA out of memory|ValidationError|ConfigKeyError|FileNotFoundError|No space left on device|Input/output error|DataLoader worker.*(exited unexpectedly|killed)|map-style resume cannot|dataset fingerprint mismatch|schedule mismatch' "$attempt_log"
}

session_alive() {
    remote_shell "$RUN_HOST" "tmux has-session -t $(printf '%q' "$RUN_SESSION")" >/dev/null 2>&1
}

node_snapshot() {
    local gpu disk_output disk_tmp
    gpu="$(remote_shell "$RUN_HOST" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,pstate --format=csv,noheader,nounits" 2>/dev/null | tr '\n' ';' || true)"
    disk_output="$(remote_shell "$RUN_HOST" "df -Pk $(printf '%q' "$RUN_OUTPUT") | awk 'NR==2 {print \$4\"KB_free/\"\$2\"KB_total\"}'" 2>/dev/null || true)"
    disk_tmp="$(remote_shell "$RUN_HOST" "df -Pk /tmp | awk 'NR==2 {print \$4\"KB_free/\"\$2\"KB_total\"}'" 2>/dev/null || true)"
    printf "gpu=[%s] shared_disk=%s local_tmp=%s" "$gpu" "${disk_output:-unknown}" "${disk_tmp:-unknown}"
}

recover_task() {
    local checkpoint attempt_count=0
    if [[ -f "$RUN_OUTPUT/monitor_state/attempt_count" ]]; then
        attempt_count="$(sed -n '1p' "$RUN_OUTPUT/monitor_state/attempt_count")"
    fi
    [[ "$attempt_count" =~ ^[0-9]+$ ]] || attempt_count=0
    if (( attempt_count >= 3 )); then
        alert_task "ordinary failure repeated across $attempt_count attempts; preserving state for review"
        return
    fi
    if checkpoint="$(latest_complete_checkpoint)"; then
        :
    elif [[ -e "$RUN_CHECKPOINT_DIR/latest_checkpoint.txt" ]] ||
        find "$RUN_CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]' -print -quit 2>/dev/null | grep -q .; then
        alert_task "process stopped but checkpoint artifacts exist without a complete latest DCP; preserving state for review"
        return
    else
        checkpoint="$BASE_CHECKPOINT"
    fi
    log_task "RECOVERY_START checkpoint=$checkpoint prior_attempts=$attempt_count"
    local recovery_output
    if recovery_output="$(WORKTREE_DIFF_HASH="$(awk '/^worktree_diff_sha256:/ {print $2; exit}' "$RUN_OUTPUT/run_manifest.yaml")" "$LAUNCHER" --start-one "$RUN_EXPERIMENT" "$checkpoint" 2>&1)"; then
        printf "%s\n" "$recovery_output" | tee -a "$RUN_OUTPUT/logs/recovery.log" "$RUN_OUTPUT/logs/monitor.log"
    else
        log_task "RECOVERY_DEFERRED launcher refused or node/GPU not ready: $recovery_output"
    fi
}

check_task() {
    local experiment="$1"
    load_run "$experiment" || return
    mkdir -p -- "$RUN_OUTPUT/logs" "$RUN_OUTPUT/monitor_state"
    if ! remote_shell "$RUN_HOST" "hostname" >/dev/null 2>&1; then
        log_task "NODE_UNREACHABLE host=$RUN_HOST"
        return
    fi
    if ! check_config_drift; then
        alert_task "TOML hash/config drift detected; no automatic restart"
        return
    fi

    local train_log="$RUN_OUTPUT/logs/train.log"
    local step=0 age=-1 latest=none snapshot attempt_log
    step="$(latest_step_from_log "$train_log")"
    step="${step:-0}"
    if [[ -f "$train_log" ]]; then age=$(( $(date +%s) - $(stat -c %Y "$train_log") )); fi
    if [[ -f "$RUN_CHECKPOINT_DIR/latest_checkpoint.txt" ]]; then latest="$(sed -n '1p' "$RUN_CHECKPOINT_DIR/latest_checkpoint.txt")"; fi
    snapshot="$(node_snapshot)"
    attempt_log="$(current_attempt_log)"

    "$REPO_ROOT/.venv/bin/python" "$RETENTION" "$RUN_EXPERIMENT" >> "$RUN_OUTPUT/logs/monitor.log" 2>&1 || {
        alert_task "retention validation failed; no further automatic deletion/restart"
        return
    }

    if blocking_error "$attempt_log"; then
        alert_task "NaN/Inf, OOM, config drift, storage, or systemic data error found in $attempt_log"
        return
    fi

    if session_alive; then
        if (( age > 900 )); then
            log_task "STALE_LOG step=$step age_seconds=$age latest=$latest $snapshot"
        else
            log_task "RUNNING step=$step log_age_seconds=$age latest=$latest $snapshot"
        fi
        return
    fi

    if (( step >= 8500 )) && [[ "$latest" == "iter_000008500" ]]; then
        local final_output
        if final_output="$("$REPO_ROOT/.venv/bin/python" "$RETENTION" "$RUN_EXPERIMENT" 2>&1)" && grep -q "FINAL_OK" <<< "$final_output"; then
            printf "%s\n" "$final_output" >> "$RUN_OUTPUT/logs/monitor.log"
            printf "%s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_OUTPUT/monitor_state/done_utc"
            log_task "DONE step=8500 latest=iter_000008500"
            return
        fi
        alert_task "training reached 8500 but final DCP/retention validation failed"
        return
    fi

    if [[ -f "$RUN_OUTPUT/monitor_state/blocked" ]]; then
        log_task "STOPPED_BLOCKED step=$step latest=$latest"
        return
    fi
    log_task "PROCESS_DOWN step=$step latest=$latest; attempting same-task recovery"
    recover_task
}

all_done() {
    local experiment host toml job output port session count=0
    while IFS='|' read -r experiment host toml job output port session; do
        [[ -n "$experiment" && "$experiment" != \#* ]] || continue
        [[ -f "$output/monitor_state/done_utc" ]] && count=$((count + 1))
    done < "$RUN_TABLE"
    [[ "$count" == "4" ]]
}

echo ">>> $(date -u '+%Y-%m-%dT%H:%M:%SZ') monitor started poll_seconds=$POLL_SECONDS"
while true; do
    if [[ "$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null)" != "$EXPECTED_BRANCH" ]]; then
        printf ">>> %s GLOBAL_BLOCKED branch drift\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$GLOBAL_ALERT"
    else
        while IFS='|' read -r experiment host toml job output port session; do
            [[ -n "$experiment" && "$experiment" != \#* ]] || continue
            check_task "$experiment"
        done < "$RUN_TABLE"
    fi
    if all_done; then
        echo ">>> $(date -u '+%Y-%m-%dT%H:%M:%SZ') ALL_DONE all four experiments completed and validated"
        exit 0
    fi
    sleep "$POLL_SECONDS"
done
