#!/usr/bin/env bash
# Reuse production CUDA dependencies without modifying the reference environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REFERENCE_ENV="${COSMOS_REFERENCE_ENV:-/share-mixed/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train}"
TARGET_ENV="${COSMOS_MEGATRON_ENV:-$PWD/.venv}"
[[ -x "$REFERENCE_ENV/bin/python" ]] || { echo "Missing reference Python: $REFERENCE_ENV" >&2; exit 1; }
[[ "$TARGET_ENV" != "$REFERENCE_ENV" ]] || { echo "Target must differ from reference environment" >&2; exit 1; }
if [[ ! -x "$TARGET_ENV/bin/python" ]]; then
    uv venv --python "$REFERENCE_ENV/bin/python" "$TARGET_ENV"
fi
export LD_LIBRARY_PATH=''
REFERENCE_SITE="$("$REFERENCE_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
TARGET_SITE="$("$TARGET_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "$REFERENCE_SITE" > "$TARGET_SITE/cosmos-production-dependencies.pth"
if [[ "${COSMOS_STAGE_PYTHON:-1}" == "1" ]]; then
    "$TARGET_ENV/bin/python" tools/stage_megatron_python.py \
        --source-site "$REFERENCE_SITE" --target-site "$TARGET_SITE" \
        --cache-root "${COSMOS_LOCAL_PYTHON_ROOT:-/opt/cosmos-megatron-runtime}"
fi
uv pip install --python "$TARGET_ENV/bin/python" --no-deps megatron-core==0.20.0
PYTHONPATH="$PWD" "$TARGET_ENV/bin/python" - <<'PY'
import importlib.metadata
import torch
from cosmos_framework.model.attention.flash3 import FLASH3_SUPPORTED

assert importlib.metadata.version("megatron-core") == "0.20.0"
assert FLASH3_SUPPORTED, "Production configuration requires FlashAttention 3"
print(f"Megatron environment ready: torch={torch.__version__}, FlashAttention3={FLASH3_SUPPORTED}")
PY
