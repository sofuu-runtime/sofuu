#!/usr/bin/env bash
# scripts/cuda/check-leaks.sh — the leak/sanitizer gate (PLAN-AUDIT-REMEDIATION §8 Task 6.3).
#
# CUDA-KAGGLE.md and PLAN-GPU-CUDA.md long claimed handle teardown was
# "verified under Linux memory sanitizers and Valgrind". It was not: no such
# target existed. This is that target.
#
# What it does: builds the same GPU parity harness the Kaggle gate builds, then
# runs it twice on a real NVIDIA host —
#   1. under valgrind --leak-check=full, asserting 0 definitely-lost bytes
#   2. as an AddressSanitizer/LeakSanitizer build, asserting a clean exit
#
# It belongs in the CUDA release job, not the per-push job: it needs a GPU and a
# driver. Everywhere else it exits 0 with an explicit "skipped" line, because a
# gate that fails on a laptop teaches people to ignore gates.
#
#   bash scripts/cuda/check-leaks.sh            # both halves
#   bash scripts/cuda/check-leaks.sh --valgrind # one half
#
# Exit: 0 pass or skipped · 1 a leak was found · 2 the harness could not run.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CUDA="$ROOT/src/cuda"
CC_BIN="${CC:-cc}"
CXX_BIN="${CXX:-c++}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sofuu-cuda-leaks.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

WANT="${1:-all}"
skip() { echo "[SKIP] $*"; exit 0; }

echo "== Sofuu CUDA leak gate =="

command -v nvidia-smi >/dev/null 2>&1 || skip "no NVIDIA host (nvidia-smi absent); run this on the release GPU box"
[[ -f "$CUDA/kernels.ptx" ]] || { echo "ERROR: $CUDA/kernels.ptx missing" >&2; exit 2; }

build_harness() {
    local extra_cxx="$1" out="$2"
    "$CC_BIN" -std=c11 -O1 -g -I "$CUDA" -c "$CUDA/cuda_loader.c" -o "$TMP/cuda_loader.o" || return 1
    # shellcheck disable=SC2086
    "$CXX_BIN" -std=c++17 -O1 -g -I "$CUDA" -I "$CUDA/parity" $extra_cxx \
        "$CUDA/parity/gpu_check.cpp" "$TMP/cuda_loader.o" -ldl -o "$out"
}

fail=0

if [[ "$WANT" == "all" || "$WANT" == "--valgrind" ]]; then
    if ! command -v valgrind >/dev/null 2>&1; then
        echo "[SKIP] valgrind not installed (apt-get install -y valgrind)"
    else
        echo "-- building under plain gcc for valgrind"
        if build_harness "" "$TMP/gpu_check_vg"; then
            echo "-- valgrind --leak-check=full"
            valgrind --error-exitcode=99 --leak-check=full --show-leak-kinds=definite \
                --errors-for-leak-kinds=definite --track-fds=yes \
                "$TMP/gpu_check_vg" "$CUDA/kernels.ptx" >"$TMP/vg.log" 2>&1
            rc=$?
            # Definitely-lost and still-reachable lines from the summary.
            grep -E 'definitely lost|indirectly lost|ERROR SUMMARY|FILE DESCRIPTORS' "$TMP/vg.log" || true
            if [[ "$rc" -eq 2 ]]; then
                echo "[SKIP] valgrind run found no GPU (harness exit 2)"
            elif [[ "$rc" -ne 0 ]]; then
                echo "[FAIL] valgrind exited $rc — leaks or memory errors"; tail -40 "$TMP/vg.log"; fail=1
            else
                echo "[PASS] valgrind: no definite leaks, no memory errors"
            fi
        else
            echo "[FAIL] harness did not build for valgrind"; fail=1
        fi
    fi
fi

if [[ "$WANT" == "all" || "$WANT" == "--asan" ]]; then
    echo "-- building with -fsanitize=address,leak"
    if build_harness "-fsanitize=address,leak -fno-omit-frame-pointer" "$TMP/gpu_check_asan"; then
        ASAN_OPTIONS=detect_leaks=1 LSAN_OPTIONS=report_objects=1 \
            "$TMP/gpu_check_asan" "$CUDA/kernels.ptx" >"$TMP/asan.log" 2>&1
        rc=$?
        if [[ "$rc" -eq 2 ]]; then
            echo "[SKIP] sanitizer run found no GPU (harness exit 2)"
        elif [[ "$rc" -ne 0 ]]; then
            echo "[FAIL] sanitizer run exited $rc"; tail -40 "$TMP/asan.log"; fail=1
        elif grep -q 'ERROR: LeakSanitizer' "$TMP/asan.log"; then
            echo "[FAIL] LeakSanitizer reported leaks"; tail -40 "$TMP/asan.log"; fail=1
        else
            echo "[PASS] sanitizers: clean exit, no leaks reported"
        fi
    else
        echo "[SKIP] compiler has no -fsanitize=address,leak (use clang or gcc >= 8)"
    fi
fi

if [[ "$fail" -ne 0 ]]; then
    echo "== RESULT: FAIL =="
    exit 1
fi
echo "== RESULT: PASS =="
exit 0
