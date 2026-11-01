#!/usr/bin/env bash
# scripts/cuda/kaggle_gpu_check.sh — one-command CUDA gate on Kaggle's T4.
#
# Needs no CUDA toolkit, no PyTorch, no Rust: it compiles the driver loader and
# the standalone harness with the system C/C++ compilers, downloads nothing,
# and loads the checked-in src/cuda/kernels.ptx through libcuda.so.1.
#
# Kaggle notebook:
#   1. Settings -> Accelerator = GPU T4 x2 (or P100)
#   2. Settings -> Internet = On (only needed to clone/fetch the bundle)
#   3. !bash scripts/cuda/kaggle_gpu_check.sh
#
# Exit codes: 0 all kernels pass · 1 parity failure · 2 no driver/device.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CUDA="$ROOT/src/cuda"
CC_BIN="${CC:-cc}"
CXX_BIN="${CXX:-c++}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sofuu-cuda-gpu.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

echo "== Sofuu CUDA GPU parity gate (Kaggle T4) =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader || true
else
    echo "[warn] nvidia-smi not found; libcuda.so.1 may still be present"
fi

if [[ ! -f "$CUDA/kernels.ptx" ]]; then
    echo "ERROR: $CUDA/kernels.ptx missing (upload the whole bundle)" >&2
    exit 1
fi

echo "-- building harness"
"$CC_BIN" -std=c11 -O2 -I "$CUDA" -c "$CUDA/cuda_loader.c" \
    -o "$TMP/cuda_loader.o"
"$CXX_BIN" -std=c++17 -O2 -I "$CUDA" -I "$CUDA/parity" \
    "$CUDA/parity/gpu_check.cpp" "$TMP/cuda_loader.o" -ldl \
    -o "$TMP/gpu_check"

echo "-- running kernels"
set +e
"$TMP/gpu_check" "$CUDA/kernels.ptx"
rc=$?
set -e

if [[ "$rc" -eq 2 ]]; then
    echo "" >&2
    echo "No CUDA device was visible to the driver." >&2
    echo "Check: Kaggle sidebar -> Session options -> Accelerator = GPU T4 x2;" >&2
    echo "then restart the session. 'nvidia-smi' should list a Tesla T4." >&2
fi
exit "$rc"
