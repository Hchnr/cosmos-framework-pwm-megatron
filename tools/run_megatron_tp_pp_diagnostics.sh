#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Four-GPU mechanism diagnostics. This does not substitute for production acceptance.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${COSMOS_MEGATRON_PYTHON:-$PWD/.venv/bin/python}"
RUN_ROOT="${COSMOS_TP_PP_DIAGNOSTIC_ROOT:-$PWD/artifacts/megatron_tp_pp/diagnostics_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite diagnostic run: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT/logs"
export LD_LIBRARY_PATH=''
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m pytest -q tests/megatron/tp_pp_contract_test.py tests/megatron/tp_pp_source_test.py \
    tests/megatron/tp_pp_conversion_test.py tests/megatron/tp_pp_loss_test.py \
    tests/megatron/tp_pp_network_io_test.py tests/megatron/tp_pp_transport_test.py \
    tests/megatron/tp_pp_bundle_test.py tests/megatron/tp_pp_logical_loader_test.py \
    tests/megatron/tp_pp_rng_test.py tests/megatron/tp_pp_scheduler_test.py \
    tests/megatron/tp_pp_preparation_test.py tests/megatron/tp_pp_schema_test.py \
    tests/megatron/tp_pp_planning_test.py tests/megatron/tp_pp_retention_test.py \
    tests/megatron/tp_pp_checkpoint_compare_test.py > "$RUN_ROOT/logs/unit.log" 2>&1
run_case() {
    local name="$1"
    shift
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/tp_pp_smoke.py \
        --output "$RUN_ROOT/$name.json" "$@" > "$RUN_ROOT/logs/$name.log" 2>&1
    echo "Passed: $name"
}
run_case tp1_pp1 --tp 1 --pp 1
run_case tp2_pp1 --tp 2 --pp 1
run_case tp1_pp2 --tp 1 --pp 2
run_case tp2_pp2 --tp 2 --pp 2
run_case tp2_pp2_compile_ac --tp 2 --pp 2 --compile --checkpoint-activations
run_case tp2_pp2_compile_ac_io --tp 2 --pp 2 --compile --checkpoint-activations --train-io
run_case tp2_pp2_dynamic_io --tp 2 --pp 2 --compile --checkpoint-activations --train-io --dynamic-layouts
run_case checkpoint_save --tp 2 --pp 2 --checkpoint-dir "$RUN_ROOT/checkpoint" --stop-after 1
# All training ranks from the save invocation have exited before this new launch.
run_case checkpoint_resume --tp 2 --pp 2 --checkpoint-dir "$RUN_ROOT/checkpoint" --resume
"$PYTHON" - "$RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
reports = [json.loads(path.read_text()) for path in root.glob('*.rank*.json')]
assert len(reports) == 36 and all(report['passed'] for report in reports)
summary = {
    'passed': True,
    'scope': 'FP32 small-capacity three-tower mechanism diagnostics, 12 logical slots',
    'rank_reports': len(reports),
    'maximum_global_gradient_relative_l2': max(s['relative_l2'] for r in reports for s in r['gradient_summary']),
    'minimum_global_gradient_cosine': min(s['cosine'] for r in reports for s in r['gradient_summary']),
    'source_full_checkpoint_resume_accepted': False,
    'production_capacity_576_slot_acceptance': False,
    'performance_accepted': False,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
PY
