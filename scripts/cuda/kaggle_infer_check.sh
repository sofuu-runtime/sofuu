#!/usr/bin/env bash
# scripts/cuda/kaggle_infer_check.sh — one-command GPU inference gate on a T4.
#
# Runs the same prompt through the CPU and the CUDA decoder, compares the full
# last-token logit vectors, then does a 32-token GPU decode for the rate.
#
# The binary MUST be the glibc build (dist/sofuu-linux-x86_64-gnu). The static
# musl release cannot dlopen libcuda.so.1; the script refuses it with a named
# reason instead of quietly measuring the CPU twice.
#
# Usage:
#   bash scripts/cuda/kaggle_infer_check.sh /kaggle/input/.../model.gguf ["prompt"]
#
# Overrides: SOFUU_BIN (binary path), SOFUU_WORK (scratch dir).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${SOFUU_BIN:-$ROOT/sofuu}"
WORK="${SOFUU_WORK:-/kaggle/working}"
MODEL="${1:-}"
PROMPT="${2:-The capital of France is}"

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "usage: $0 /kaggle/input/<dataset>/<model>.gguf [\"prompt\"]" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "no executable at $BIN (set SOFUU_BIN=...)" >&2
    exit 2
fi
mkdir -p "$WORK"

echo "== Sofuu CUDA inference gate =="
echo "binary: $BIN"
echo "model:  $MODEL"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

# Refuse a static-musl binary before spending a GPU run on it. `file` is not
# guaranteed on Kaggle, so the check is the presence of the glibc interpreter
# string in the ELF header area, which a static binary does not carry.
if ! grep -aq "ld-linux-x86-64.so.2" "$BIN"; then
    echo "" >&2
    echo "REFUSED: $BIN looks static (no ld-linux interpreter)." >&2
    echo "Static musl cannot dlopen libcuda.so.1. Use the glibc build (dist/kaggle/sofuu-linux-x86_64-gnu)" >&2
    echo "(the sofuu-gpu-kit artifact)." >&2
    exit 2
fi

CPU_LOG="$WORK/cpu.sflog"
GPU_LOG="$WORK/gpu.sflog"
CPU_ERR="$WORK/cpu.stderr"
GPU_ERR="$WORK/gpu.stderr"
rm -f "$CPU_LOG" "$GPU_LOG"

echo ""
echo "-- CPU reference (f32 KV, 1 token)"
SOFUU_CUDA=0 SOFUU_KV_F16=0 SOFUU_DUMP_LOGITS="$CPU_LOG" \
    "$BIN" infer "$MODEL" "$PROMPT" -n 1 -t 4 \
    >"$WORK/cpu.stdout" 2>"$CPU_ERR"

echo "-- GPU decoder (1 token)"
SOFUU_CUDA=1 SOFUU_CUDA_DEBUG=1 SOFUU_DUMP_LOGITS="$GPU_LOG" \
    "$BIN" infer "$MODEL" "$PROMPT" -n 1 -t 4 \
    >"$WORK/gpu.stdout" 2>"$GPU_ERR"

if ! grep -q "GPU decoder armed" "$GPU_ERR"; then
    echo "" >&2
    echo "FAIL: the CUDA decoder did not arm. Last lines:" >&2
    tail -25 "$GPU_ERR" >&2
    exit 1
fi
grep -E "^\[cuda\]" "$GPU_ERR" | head -10

if [ ! -s "$CPU_LOG" ] || [ ! -s "$GPU_LOG" ]; then
    echo "FAIL: a logits receipt is missing ($CPU_LOG / $GPU_LOG)" >&2
    exit 1
fi

echo ""
echo "-- logits comparison"
python3 - "$CPU_LOG" "$GPU_LOG" <<'PY'
import struct, sys
import numpy as np

def load(p):
    b = open(p, 'rb').read()
    assert b[:8] == b'SFLOG01\0', (p, b[:8])
    n = struct.unpack('<Q', b[8:16])[0]
    return np.frombuffer(b[16:16 + 4 * n], dtype='<f4')

a, b = load(sys.argv[1]), load(sys.argv[2])
if a.shape != b.shape:
    print(f'FAIL: vocab mismatch {a.shape} vs {b.shape}')
    raise SystemExit(1)
d = np.abs(a - b)
amax_a, amax_b = int(a.argmax()), int(b.argmax())
top_a, top_b = set(a.argsort()[-10:].tolist()), set(b.argsort()[-10:].tolist())
print(f'n={len(a)} maxabs={float(d.max()):.6g} meanabs={float(d.mean()):.6g}')
print(f'argmax cpu={amax_a} gpu={amax_b} top10_overlap={len(top_a & top_b)}/10')
ok = amax_a == amax_b and float(d.max()) <= 0.5
print('RESULT:', 'PASS' if ok else 'FAIL')
raise SystemExit(0 if ok else 1)
PY
cmp_rc=$?

echo ""
echo "-- GPU phase split (graph off, one sync per phase)"
SOFUU_CUDA_PROFILE=1 SOFUU_CUDA=1 "$BIN" infer "$MODEL" "$PROMPT" -n 4 -t 4 \
    >"$WORK/gpuprof.stdout" 2>"$WORK/gpuprof.stderr" || true
grep -E "cuda-profile" "$WORK/gpuprof.stderr" | tail -2 || true

echo ""
echo "-- GPU decode rate (32 tokens)"
SOFUU_CUDA=1 "$BIN" infer "$MODEL" "$PROMPT" -n 32 -t 4 \
    >"$WORK/gpu32.stdout" 2>"$WORK/gpu32.stderr" || true
grep -E "prefill:|decode:" "$WORK/gpu32.stderr" | tail -2 || tail -10 "$WORK/gpu32.stderr"
grep -q "GPU decoder armed" "$WORK/gpu32.stderr" || {
    echo "WARN: 32-token run did not report an armed decoder" >&2
}

echo ""
if [ "$cmp_rc" -eq 0 ]; then
    echo "RESULT: PASS (GPU logits match the CPU argmax)"
    exit 0
fi
echo "RESULT: FAIL (logits diverge — send cpu.stderr/gpu.stderr and both .sflog)" >&2
exit 1
