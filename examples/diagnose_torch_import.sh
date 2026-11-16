#!/usr/bin/env bash
# Collect the native loader error without starting training or changing packages.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TRAIN_ENV="${COSMOS_TRAIN_VENV:-/share-mixed/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train}"
DIAG_ROOT="${OUTPUT_ROOT:-$PWD/outputs/torch_import_diagnostics}/env_diagnostics"
DIAG_DIR="$DIAG_ROOT/$(hostname)_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$DIAG_DIR"
exec > >(tee "$DIAG_DIR/report.log") 2>&1
echo ">>> Torch import diagnostics: $DIAG_DIR"

export LD_LIBRARY_PATH=''
source "$TRAIN_ENV/bin/activate"
python - "$DIAG_DIR" <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys

out = Path(sys.argv[1])
spec = importlib.util.find_spec("torch")
library = Path(spec.origin).parent / "lib/libtorch_cpu.so" if spec else None
keys = (
    "LD_LIBRARY_PATH", "LD_LIBRARY_PATH_OVERRIDE", "LD_PRELOAD", "LD_AUDIT",
    "MKL_THREADING_LAYER", "MKL_INTERFACE_LAYER", "MKL_SERVICE_FORCE_INTEL",
    "MKL_ENABLE_INSTRUCTIONS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "RANK", "NODE_RANK",
)
report = dict(
    hostname=socket.gethostname(), executable=sys.executable, prefix=sys.prefix,
    platform=platform.platform(), libc=platform.libc_ver(),
    environment={key: os.environ.get(key) for key in keys},
    limits={key: resource.getrlimit(getattr(resource, key)) for key in
            ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_NOFILE", "RLIMIT_NPROC", "RLIMIT_STACK")},
    torch_library=str(library),
)
if library is not None:
    try:
        stat = library.stat()
        report["library_stat"] = dict(size=stat.st_size, mode=oct(stat.st_mode), realpath=str(library.resolve()))
        with library.open("rb") as stream:
            report["library_header"] = stream.read(16).hex()
    except OSError as error:
        report["library_error"] = str(error)
    (out / "library_path.txt").write_text(str(library))
for name in ("/proc/meminfo", "/proc/self/limits", "/proc/self/cgroup", "/proc/cpuinfo", "/etc/os-release"):
    try:
        (out / Path(name).name).write_text(Path(name).read_text())
    except OSError:
        pass
(out / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
PY

if timeout 90 python -c 'import torch; print("torch=" + torch.__version__); print("MKL=" + str(torch.backends.mkl.is_available())); print("CPU matmul=" + str((torch.ones(2, 2) @ torch.ones(2, 2)).tolist()))' > "$DIAG_DIR/import.log" 2>&1; then
    cat "$DIAG_DIR/import.log"
    echo ">>> PASS: Torch import and CPU matmul succeeded."
    exit 0
else
    IMPORT_STATUS=$?
fi
cat "$DIAG_DIR/import.log"
if [[ -f "$DIAG_DIR/library_path.txt" ]]; then
    TORCH_CPU_LIBRARY="$(cat "$DIAG_DIR/library_path.txt")"
    ldd "$TORCH_CPU_LIBRARY" > "$DIAG_DIR/ldd.log" 2>&1 || echo "ldd exit=$?"
    timeout 90 python - "$TORCH_CPU_LIBRARY" > "$DIAG_DIR/dlopen.log" 2>&1 <<'PY' || echo "dlopen exit=$?"
import ctypes
import os
import sys

ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW | os.RTLD_GLOBAL)
print("Direct dlopen succeeded", flush=True)
PY
    cat "$DIAG_DIR/dlopen.log"
fi
timeout 90 env LD_DEBUG=libs,files python -c 'import torch' > "$DIAG_DIR/loader.log" 2>&1 || echo "loader trace exit=$?"
echo ">>> FAIL: import exit=$IMPORT_STATUS; native loader trace: $DIAG_DIR/loader.log"
exit "$IMPORT_STATUS"
