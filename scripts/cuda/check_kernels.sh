#!/usr/bin/env bash
# scripts/cuda/check_kernels.sh — the local (no-GPU) CUDA gate.
#
# Runs everywhere:
#   1. regenerates the PTX and fails if the checked-in artifact is stale
#   2. verifies every expected .entry is present
#   3. verifies kernels.ptx.gz is the deterministic gzip of kernels.ptx
#   4. compiles kernels.cu in SOFUU_CUDA_HOST mode and compares its math
#      against src/cuda/parity/cpu_ref.h (q4_K/q6_K/q8_0/rmsnorm/swiglu/rope/kv)
#   5. compiles cuda_loader.c under strict C11 warnings
#
# The actual GPU launches are gated by Kaggle T4:
#   bash scripts/cuda/kaggle_gpu_check.sh
#
# Usage: bash scripts/cuda/check_kernels.sh [--no-ptx]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CUDA="$ROOT/src/cuda"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sofuu-cuda-check.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

pass=0
warn=0

echo "== Sofuu CUDA kernel checks =="

# 1+2. PTX freshness / entries
if [[ "${1:-}" == "--no-ptx" ]]; then
    echo "[SKIP] PTX regeneration (--no-ptx)"
    warn=$((warn + 1))
elif "$ROOT/scripts/cuda/gen_ptx.sh" --compiler >/dev/null 2>&1; then
    "$ROOT/scripts/cuda/gen_ptx.sh" "$TMP/kernels.ptx" >/dev/null
    if cmp -s "$TMP/kernels.ptx" "$CUDA/kernels.ptx"; then
        echo "[PASS] kernels.ptx is in sync with kernels.cu"
        pass=$((pass + 1))
    else
        echo "[FAIL] kernels.ptx is STALE — run: scripts/cuda/gen_ptx.sh" >&2
        diff -u "$CUDA/kernels.ptx" "$TMP/kernels.ptx" | head -40 >&2 || true
        exit 1
    fi
    if cmp -s "$TMP/kernels.ptx.gz" "$CUDA/kernels.ptx.gz"; then
        echo "[PASS] kernels.ptx.gz matches (deterministic gzip -9n)"
        pass=$((pass + 1))
    else
        echo "[FAIL] kernels.ptx.gz is stale" >&2
        exit 1
    fi
else
    echo "[SKIP] no nvcc/NVPTX clang++ on this host; using the checked-in PTX"
    warn=$((warn + 1))
fi

# 3. Entry manifest on the artifact that actually ships.
missing=0
for k in sofuu_gemv_q8_0 sofuu_gemv_q4_k sofuu_gemv_q6_k sofuu_gemv_f32 \
         sofuu_quantize_q8_1 sofuu_gemv_q4_k_q8 sofuu_gemv_q4_k_q8_x2 sofuu_gemv_q6_k_q8 \
         sofuu_rmsnorm sofuu_rmsnorm_q8 sofuu_rmsnorm_add_q8 sofuu_swiglu sofuu_swiglu_q8 sofuu_rope_f32 sofuu_rope_kv_f32 sofuu_kv_write_f32 \
         sofuu_attn_decode_f32 sofuu_attn_decode_split sofuu_attn_prefill_split sofuu_attn_combine sofuu_add_f32 \
         sofuu_gemm_q4_k_b8 sofuu_gemm_q4_k_b8_q8 sofuu_gemm_q6_k_b8 sofuu_gemm_q8_0_b8 \
         sofuu_gemm_f32_b8 sofuu_rmsnorm_batch sofuu_rope_batch \
         sofuu_kv_write_batch sofuu_attn_prefill_f32; do
    grep -Eq "\.entry[[:space:]]+$k\(" "$CUDA/kernels.ptx" || {
        echo "[FAIL] missing PTX entry $k" >&2
        missing=1
    }
done
if [[ "$missing" -eq 0 ]]; then
    echo "[PASS] 30 kernel entries present in kernels.ptx"
    pass=$((pass + 1))
else
    exit 1
fi

# 4. Host algorithm parity
CXX_BIN="${CXX:-c++}"
"$CXX_BIN" -std=c++17 -O2 -Wall -Wextra -DSOFUU_CUDA_HOST \
    "$CUDA/parity/host_check.cpp" -o "$TMP/host_check"
"$TMP/host_check"

# 5. The GPU harness must compile cleanly even where it cannot run. This is
#    where the `!sofuu_cuda_ctx_alloc(...)` mistake (0 means success) would
#    have been caught before a Kaggle round-trip.
"$CXX_BIN" -std=c++17 -O2 -Wall -Wextra -Werror -I "$CUDA" -I "$CUDA/parity" \
    -fsyntax-only "$CUDA/parity/gpu_check.cpp"
echo "[PASS] gpu_check.cpp compiles clean under -Wall -Wextra -Werror"
pass=$((pass + 1))

# 6. Loader portability
CC_BIN="${CC:-cc}"
"$CC_BIN" -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror -I "$CUDA" \
    -c "$CUDA/cuda_loader.c" -o "$TMP/cuda_loader.o"
echo "[PASS] cuda_loader.c compiles clean under -Wall -Wextra -Wpedantic -Werror"
pass=$((pass + 1))

echo "== RESULT: PASS ($pass checks; $warn compiler-dependent skips) =="
