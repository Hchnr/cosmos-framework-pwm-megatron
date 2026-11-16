#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Orchestrate four independent 1-node x 8-GPU Nano8B ratio-ablation runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TABLE="$REPO_ROOT/examples/v2_perf_ratio_ablation_runs_r02.tsv"
VALIDATOR="$REPO_ROOT/examples/validate_v2_perf_ratio_ablation_r02.py"
RETENTION="$REPO_ROOT/examples/retain_v2_perf_ratio_ablation_checkpoint_r02.py"
MONITOR="$REPO_ROOT/examples/monitor_v2_perf_ratio_ablation_4x1node_r02.sh"
THIS_SCRIPT="$REPO_ROOT/examples/launch_v2_perf_ratio_ablation_4x1node_r02.sh"
EXPECTED_BRANCH="ybai_0828/understanding_verification"
BASE_CHECKPOINT="/share/project/liyue/model_ckpts/Cosmos3-Nano-DCP"
HF_MODEL="/share/project/eai_pwm/models/Cosmos3-Nano"
OWNER_FILE=".v2_perf_ratio_ablation_owner"
MONITOR_SESSION="v2_perf_ratio_ablation_monitor_20260901_r02"
GLOBAL_MONITOR_LOG="$REPO_ROOT/outputs/train/v2_perf_ratio_ablation_4x1node_20260901_r02_monitor.log"

usage() {
    echo "Usage:" >&2
    echo "  $0                         # preflight, prepare, stagger-launch all four, start monitor" >&2
    echo "  $0 --worker <experiment> <base-or-own-checkpoint>" >&2
    echo "  $0 --start-one <experiment> <base-or-own-checkpoint>" >&2
}

load_run() {
    local experiment="$1"
    local row
    row="$(awk -F "|" -v experiment="$experiment" '$1 == experiment {print; found++} END {if (found != 1) exit 3}' "$RUN_TABLE")" || {
        echo "ERROR: experiment is not registered exactly once: $experiment" >&2
        return 2
    }
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
        ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$host" "$command"
    fi
}

shell_join() {
    local destination="$1"
    shift
    local joined
    printf -v joined "%q " "$@"
    printf -v "$destination" "%s" "${joined% }"
}

worktree_diff_hash() {
    (
        git -C "$REPO_ROOT" diff --binary HEAD
        git -C "$REPO_ROOT" ls-files --others --exclude-standard -z | sort -z |
            while IFS= read -r -d '' path; do
                printf "UNTRACKED %s\0" "$path"
                sha256sum -- "$REPO_ROOT/$path"
            done
) | sha256sum | awk '{print $1}'
}

prepare_roots() {
    local experiment host toml job output port session
    while IFS='|' read -r experiment host toml job output port session; do
        [[ -n "$experiment" && "$experiment" != \#* ]] || continue
        if [[ -e "$output" ]]; then
            echo "ERROR: refusing to overwrite existing output root: $output" >&2
            return 1
        fi
        mkdir -p -- "$output/logs" "$output/monitor_state"
        printf "%s\n" "$experiment" > "$output/$OWNER_FILE"
    done < "$RUN_TABLE"
}

write_manifest() {
    local start_checkpoint="$1"
    local start_utc="$2"
    local hostname_value="$3"
    local attempt="$4"
    local toml_sha="$5"
    local diff_hash="$6"
    local manifest="$RUN_OUTPUT/run_manifest.yaml"
    local branch commit command_text
    branch="$(git -C "$REPO_ROOT" branch --show-current)"
    commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    command_text="IMAGINAIRE_OUTPUT_ROOT=$RUN_OUTPUT BASE_CHECKPOINT_PATH=$start_checkpoint CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. WANDB_MODE=online env -u MASTER_ADDR -u MASTER_PORT -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u ROLE_RANK -u PET_MASTER_ADDR -u PET_MASTER_PORT -u PET_NNODES -u PET_NODE_RANK -u PET_NPROC_PER_NODE -u PET_RDZV_BACKEND -u PET_RDZV_ENDPOINT -u PET_WORLD_SIZE -u TORCHELASTIC_RUN_ID -u RDZV_ENDPOINT torchrun --nnodes=1 --node-rank=0 --nproc-per-node=8 --rdzv-backend=static --rdzv-endpoint=127.0.0.1:$RUN_PORT --rdzv-id=$RUN_SESSION --master-addr=127.0.0.1 --master-port=$RUN_PORT -m cosmos_framework.scripts.train --sft-toml=$RUN_TOML -- +model.config.vlm_config.model_instance.config.include_visual=true model.config.vlm_config.tokenizer.pretrained_model_name=$HF_MODEL ~trainer.callbacks.every_n_sample_reg=null ~trainer.callbacks.every_n_sample_ema=null"
    if [[ -e "$manifest" ]]; then
        return 0
    fi
    [[ "$start_checkpoint" == "$BASE_CHECKPOINT" ]] || {
        echo "ERROR: first attempt must start from the common base DCP" >&2
        return 1
    }
    {
        echo "schema_version: 1"
        printf "experiment: %s\n" "$RUN_EXPERIMENT"
        printf "toml: %s\n" "$RUN_TOML"
        printf "toml_sha256: %s\n" "$toml_sha"
        printf "git_branch: %s\n" "$branch"
        printf "git_commit: %s\n" "$commit"
        printf "worktree_diff_sha256: %s\n" "$diff_hash"
        printf "host_alias: %s\n" "$RUN_HOST"
        printf "hostname: %s\n" "$hostname_value"
        printf "start_utc: %s\n" "$start_utc"
        printf "initial_attempt: %s\n" "$attempt"
        printf "launcher_pid: %s\n" "$$"
        printf "initial_checkpoint: %s\n" "$BASE_CHECKPOINT"
        echo "topology:"
        echo "  nnodes: 1"
        echo "  nproc_per_node: 8"
        echo "  cuda_visible_devices: 0,1,2,3,4,5,6,7"
        echo "resolved:"
        echo "  optimizer_lr: 2.0e-5"
        echo "  optimizer_weight_decay: 2.0e-7"
        echo "  trainer_max_iter: 8500"
        echo "  trainer_grad_accum_iter: 1"
        echo "  trainer_logging_iter: 1"
        echo "  scheduler_cycle_lengths: [8500]"
        echo "  scheduler_warm_up_steps: [200]"
        echo "  checkpoint_save_iter: 500"
        echo "  max_num_tokens_after_packing: 65536"
        echo "  dataloader_max_sequence_length: 65536"
        echo "  max_samples_per_batch: 48"
        echo "  packing_pool_size: 48"
        echo "  train_vlm_visual: true"
        echo "  seed: 42"
        echo "  dataset_selection_mode: exact_periodic"
        echo "  wandb_mode: online"
        printf "output_root: %s\n" "$RUN_OUTPUT"
        printf "model_dir: %s\n" "$RUN_MODEL_DIR"
        printf "checkpoint_dir: %s\n" "$RUN_CHECKPOINT_DIR"
        printf "train_log: %s\n" "$RUN_OUTPUT/logs/train.log"
        printf "monitor_log: %s\n" "$RUN_OUTPUT/logs/monitor.log"
        printf "checkpoint_retention_log: %s\n" "$RUN_OUTPUT/logs/checkpoint_retention.log"
        echo "launch_command: >-"
        printf "  %s\n" "$command_text"
        printf "recovery_log: %s\n" "$RUN_OUTPUT/logs/recovery.log"
    } > "$manifest"
}

validate_resume_checkpoint() {
    local checkpoint="$1"
    if [[ "$checkpoint" == "$BASE_CHECKPOINT" ]]; then
        [[ -d "$BASE_CHECKPOINT" ]] || { echo "ERROR: base DCP missing" >&2; return 1; }
        return 0
    fi
    local checkpoint_real expected_prefix latest_name check_output
    checkpoint_real="$(realpath -m -- "$checkpoint")"
    expected_prefix="$(realpath -m -- "$RUN_CHECKPOINT_DIR")"
    [[ "$checkpoint_real" == "$expected_prefix"/iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9] ]] || {
        echo "ERROR: resume checkpoint is outside this task: $checkpoint_real" >&2
        return 1
    }
    [[ -f "$expected_prefix/latest_checkpoint.txt" ]] || { echo "ERROR: latest pointer missing" >&2; return 1; }
    latest_name="$(sed -n '1{s/[[:space:]]*$//;p;q;}' "$expected_prefix/latest_checkpoint.txt")"
    [[ "$checkpoint_real" == "$expected_prefix/$latest_name" ]] || {
        echo "ERROR: resume checkpoint is not the task latest pointer: $checkpoint_real vs $latest_name" >&2
        return 1
    }
    check_output="$("$REPO_ROOT/.venv/bin/python" "$RETENTION" "$RUN_EXPERIMENT" --check-only 2>&1)" || return 1
    grep -q "CHECK complete latest=$latest_name" <<< "$check_output" || {
        echo "ERROR: resume checkpoint failed full DCP validation: $check_output" >&2
        return 1
    }
}

worker_main() {
    local experiment="$1"
    local start_checkpoint="$2"
    load_run "$experiment"
    [[ "$(git -C "$REPO_ROOT" branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
        echo "ERROR: wrong branch; refusing to train" >&2
        return 1
    }
    [[ -f "$RUN_OUTPUT/$OWNER_FILE" && "$(sed -n '1p' "$RUN_OUTPUT/$OWNER_FILE")" == "$RUN_EXPERIMENT" ]] || {
        echo "ERROR: output ownership marker mismatch" >&2
        return 1
    }
    "$REPO_ROOT/.venv/bin/python" "$VALIDATOR" --output-state owned --skip-resolved >/dev/null
    validate_resume_checkpoint "$start_checkpoint"

    local state_dir="$RUN_OUTPUT/monitor_state"
    local attempt_file="$state_dir/attempt_count"
    local prior_attempt=0 attempt attempt_log current_log start_utc hostname_value toml_sha diff_hash
    mkdir -p -- "$RUN_OUTPUT/logs" "$state_dir"
    if [[ -f "$attempt_file" ]]; then prior_attempt="$(sed -n '1p' "$attempt_file")"; fi
    [[ "$prior_attempt" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid attempt counter" >&2; return 1; }
    attempt=$((prior_attempt + 1))
    printf "%s\n" "$attempt" > "$attempt_file.tmp"
    mv -- "$attempt_file.tmp" "$attempt_file"
    attempt_log="$RUN_OUTPUT/logs/train_attempt_$(printf '%03d' "$attempt").log"
    current_log="$state_dir/current_attempt_log"
    printf "%s\n" "$attempt_log" > "$current_log.tmp"
    mv -- "$current_log.tmp" "$current_log"
    start_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    hostname_value="$(hostname)"
    toml_sha="$(sha256sum -- "$RUN_TOML" | awk '{print $1}')"
    diff_hash="${WORKTREE_DIFF_HASH:-$(worktree_diff_hash)}"

    if [[ -f "$RUN_OUTPUT/run_manifest.yaml" ]]; then
        local recorded_sha
        recorded_sha="$(awk '/^toml_sha256:/ {print $2; exit}' "$RUN_OUTPUT/run_manifest.yaml")"
        [[ "$recorded_sha" == "$toml_sha" ]] || { echo "ERROR: TOML hash drift" >&2; return 1; }
    fi
    write_manifest "$start_checkpoint" "$start_utc" "$hostname_value" "$attempt" "$toml_sha" "$diff_hash"
    if [[ ! -f "$RUN_OUTPUT/logs/start_utc" ]]; then printf "%s\n" "$start_utc" > "$RUN_OUTPUT/logs/start_utc"; fi
    printf "%s\n" "$start_utc" > "$RUN_OUTPUT/logs/last_start_utc"
    printf ">>> %s START experiment=%s attempt=%s checkpoint=%s hostname=%s launcher_pid=%s\n" "$start_utc" "$RUN_EXPERIMENT" "$attempt" "$start_checkpoint" "$hostname_value" "$$" | tee -a "$RUN_OUTPUT/logs/train.log" "$attempt_log"
    if (( attempt > 1 )); then
        printf ">>> %s RECOVERY attempt=%s checkpoint=%s\n" "$start_utc" "$attempt" "$start_checkpoint" >> "$RUN_OUTPUT/logs/recovery.log"
    fi

    unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
    unset PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NODE_RANK PET_NPROC_PER_NODE PET_RDZV_BACKEND PET_RDZV_ENDPOINT PET_WORLD_SIZE
    unset TORCHELASTIC_RUN_ID RDZV_ENDPOINT
    export PATH="$REPO_ROOT/.venv/bin:$PATH"
    export WANDB_MODE="online"
    export BASE_CHECKPOINT_PATH="$start_checkpoint"
    export HF_MODEL_PATH="$HF_MODEL"
    export IMAGINAIRE_OUTPUT_ROOT="$RUN_OUTPUT"
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    export PYTHONPATH="."
    export PYTHONUNBUFFERED=1
    local project_cuda_lib
    for project_cuda_lib in "$REPO_ROOT"/.venv/lib/python*/site-packages/nvidia/cu13/lib; do
        if [[ -d "$project_cuda_lib" ]]; then
            export LD_LIBRARY_PATH="$project_cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            break
        fi
    done
    cd "$REPO_ROOT"
    printf ">>> %s TORCHRUN_START port=%s toml=%s output=%s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$RUN_PORT" "$RUN_TOML" "$RUN_OUTPUT" | tee -a "$RUN_OUTPUT/logs/train.log" "$attempt_log"
    set +e
    torchrun --nnodes=1 --node-rank=0 --nproc-per-node=8 --rdzv-backend=static \
        --rdzv-endpoint="127.0.0.1:$RUN_PORT" --rdzv-id="$RUN_SESSION" \
        --master-addr=127.0.0.1 --master-port="$RUN_PORT" -m cosmos_framework.scripts.train \
        --sft-toml="$RUN_TOML" -- \
        "+model.config.vlm_config.model_instance.config.include_visual=true" \
        "model.config.vlm_config.tokenizer.pretrained_model_name=$HF_MODEL" \
        "~trainer.callbacks.every_n_sample_reg=null" \
        "~trainer.callbacks.every_n_sample_ema=null" \
        2>&1 | tee -a "$RUN_OUTPUT/logs/train.log" "$attempt_log"
    local exit_code=${PIPESTATUS[0]}
    set -e
    printf ">>> %s TORCHRUN_END experiment=%s attempt=%s exit=%s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$RUN_EXPERIMENT" "$attempt" "$exit_code" | tee -a "$RUN_OUTPUT/logs/train.log" "$attempt_log"
    return "$exit_code"
}

start_one() {
    local experiment="$1"
    local start_checkpoint="$2"
    load_run "$experiment"
    local check_session gpu_apps worker_command tmux_command
    shell_join check_session tmux has-session -t "$RUN_SESSION"
    if remote_shell "$RUN_HOST" "$check_session" >/dev/null 2>&1; then
        echo "ERROR: tmux session already exists on $RUN_HOST: $RUN_SESSION" >&2
        return 1
    fi
    gpu_apps="$(remote_shell "$RUN_HOST" "nvidia-smi --query-compute-apps=pid --format=csv,noheader" 2>/dev/null || true)"
    [[ -z "${gpu_apps//[[:space:]]/}" ]] || {
        echo "ERROR: GPU compute processes already exist on $RUN_HOST: $gpu_apps" >&2
        return 1
    }
    local env_args=(env "WORKTREE_DIFF_HASH=${WORKTREE_DIFF_HASH:-$(worktree_diff_hash)}" bash "$THIS_SCRIPT" --worker "$RUN_EXPERIMENT" "$start_checkpoint")
    shell_join worker_command "${env_args[@]}"
    worker_command="cd $(printf '%q' "$REPO_ROOT") && $worker_command"
    shell_join tmux_command tmux new-session -d -s "$RUN_SESSION" bash -lc "$worker_command"
    remote_shell "$RUN_HOST" "$tmux_command"

    local tries alive=0
    for tries in $(seq 1 12); do
        sleep 5
        if remote_shell "$RUN_HOST" "$check_session" >/dev/null 2>&1 && [[ -s "$RUN_OUTPUT/logs/train.log" ]] && grep -q "TORCHRUN_START" "$RUN_OUTPUT/logs/train.log"; then
            alive=1
            break
        fi
        if ! remote_shell "$RUN_HOST" "$check_session" >/dev/null 2>&1; then
            echo "ERROR: launcher exited immediately for $RUN_EXPERIMENT" >&2
            tail -n 80 "$RUN_OUTPUT/logs/train.log" 2>/dev/null || true
            return 1
        fi
    done
    [[ "$alive" == "1" ]] || { echo "ERROR: no initial torchrun log after 60 seconds" >&2; return 1; }

    local hostname_value pane_pid torchrun_pid start_utc
    hostname_value="$(remote_shell "$RUN_HOST" "hostname")"
    pane_pid="$(remote_shell "$RUN_HOST" "tmux list-panes -t $(printf '%q' "$RUN_SESSION") -F '#{pane_pid}'" | head -n 1)"
    torchrun_pid="$(remote_shell "$RUN_HOST" "pgrep -f 'torchrun.*$(basename "$RUN_TOML")' | head -n 1" || true)"
    start_utc="$(sed -n '1p' "$RUN_OUTPUT/logs/last_start_utc")"
    echo "LAUNCHED|$RUN_EXPERIMENT|$RUN_HOST|$hostname_value|$start_utc|$RUN_SESSION|$pane_pid|${torchrun_pid:-unknown}|$RUN_OUTPUT/logs/train.log"
}

orchestrate() {
    cd "$REPO_ROOT"
    [[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
        echo "ERROR: current branch is not $EXPECTED_BRANCH" >&2
        return 1
    }
    "$REPO_ROOT/.venv/bin/python" "$VALIDATOR" --output-state absent
    local diff_hash preflight_output
    diff_hash="$(worktree_diff_hash)"
    export WORKTREE_DIFF_HASH="$diff_hash"
    prepare_roots
    if ! preflight_output="$("$REPO_ROOT/.venv/bin/python" "$VALIDATOR" --output-state owned --json 2>&1)"; then
        printf "%s\n" "$preflight_output" >&2
        return 1
    fi
    local experiment host toml job output port session
    while IFS='|' read -r experiment host toml job output port session; do
        [[ -n "$experiment" && "$experiment" != \#* ]] || continue
        printf "%s\n" "$preflight_output" > "$output/logs/preflight.log"
    done < "$RUN_TABLE"

    local launch_index=0 previous check_session
    local -a launched_experiments=()
    while IFS='|' read -r experiment host toml job output port session; do
        [[ -n "$experiment" && "$experiment" != \#* ]] || continue
        if (( launch_index > 0 )); then
            echo ">>> $(date -u '+%Y-%m-%dT%H:%M:%SZ') sleeping required 120 seconds before $experiment"
            sleep 120
            for previous in "${launched_experiments[@]}"; do
                load_run "$previous"
                shell_join check_session tmux has-session -t "$RUN_SESSION"
                if ! remote_shell "$RUN_HOST" "$check_session" >/dev/null 2>&1; then
                    echo "ERROR: previously launched task exited during stagger wait: $previous" >&2
                    return 1
                fi
            done
        fi
        start_one "$experiment" "$BASE_CHECKPOINT"
        launched_experiments+=("$experiment")
        launch_index=$((launch_index + 1))
    done < "$RUN_TABLE"

    [[ ! -e "$GLOBAL_MONITOR_LOG" ]] || { echo "ERROR: global monitor log already exists" >&2; return 1; }
    local monitor_command
    shell_join monitor_command bash "$MONITOR"
    monitor_command="cd $(printf '%q' "$REPO_ROOT") && $monitor_command >> $(printf '%q' "$GLOBAL_MONITOR_LOG") 2>&1"
    tmux new-session -d -s "$MONITOR_SESSION" bash -lc "$monitor_command"
    echo "MONITOR_STARTED|local|$MONITOR_SESSION|$GLOBAL_MONITOR_LOG"
}

case "${1:-}" in
    --worker)
        (( $# == 3 )) || { usage; exit 2; }
        worker_main "$2" "$3"
        ;;
    --start-one)
        (( $# == 3 )) || { usage; exit 2; }
        start_one "$2" "$3"
        ;;
    "")
        orchestrate
        ;;
    *)
        usage
        exit 2
        ;;
esac
