#!/usr/bin/env bash
# scripts/cuda/kaggle_longctx_check.sh — one command: how far does decode fall
# off as context grows?
#
# Runs the same sentence at two context lengths through the CUDA decoder and
# prints the rate at each, then repeats the long one with SOFUU_CUDA_PROFILE=1
# (which turns the graph off and syncs per phase) because the graph is exactly
# what hides where the time goes.
#
# Kaggle notebook (after build_gpu_from_clone.sh):
#   %%bash
#   bash scripts/cuda/kaggle_longctx_check.sh
#
# Usage: bash scripts/cuda/kaggle_longctx_check.sh [model.gguf] [reps]
#   reps = how many times the sentence is repeated, default 240 (~2600-3400
#   tokens depending on the tokenizer). Keep prompt + 32 below the KV window. Override the binary with SOFUU_BIN, the KV window with
#   SOFUU_CUDA_CTX (default 4096, which must exceed the prompt).
#
# Exit codes: 0 both runs produced a rate · 1 a run failed · 2 setup problem.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${SOFUU_BIN:-$ROOT/target/release/sofuu}"
CTX="${SOFUU_CUDA_CTX:-4096}"
REPS="${2:-240}"
SENT="The quick brown fox jumps over the lazy dog. "

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
    # Kaggle mounts datasets under /kaggle/input/<dataset>/; take the first
    # GGUF. `find` rather than a glob so an empty mount is not an error under
    # `set -e`.
    MODEL="$(find /kaggle/input -maxdepth 3 -name '*.gguf' 2>/dev/null | head -1 || true)"
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "usage: $0 <model.gguf> [reps]" >&2
    echo "no GGUF given and none found under /kaggle/input" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "no executable at $BIN (set SOFUU_BIN=...); run" >&2
    echo "  bash scripts/kaggle/build_gpu_from_clone.sh" >&2
    echo "first" >&2
    exit 2
fi

LONG=""
for _ in $(seq 1 "$REPS"); do LONG+="$SENT"; done
WORK="${SOFUU_WORK:-/kaggle/working}"
mkdir -p "$WORK"

echo "== Sofuu CUDA long-context check =="
echo "binary: $BIN"
echo "model:  $MODEL"
echo "kv window: $CTX rows"
echo "prompt: ~$((REPS * 11))-$((${REPS} * 14)) tokens (${REPS}x the sentence; exact count is in the footer)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader || true
fi

# One decode run, footer only. The rate line is on stderr; stdout is the
# generated text and is discarded so the numbers stay readable.
run() { # run <label> <prompt> <n> <extra-env...>
    local label="$1" prompt="$2" n="$3"
    shift 3
    local err="$WORK/longctx-$label.stderr"
    echo ""
    echo "-- $label (n=$n)"
    if ! env SOFUU_CUDA=1 "$@" "$BIN" infer "$MODEL" "$prompt" \
            -n "$n" -t 4 --ctx "$CTX" >"$WORK/longctx-$label.stdout" 2>"$err"; then
        echo "FAILED: see $err" >&2
        tail -12 "$err" >&2
        return 1
    fi
    grep -E "GPU decoder armed" "$err" | head -1 || true
    grep -E "prefill:|decode:" "$err" | tail -1 || {
        echo "no rate line in $err" >&2
        tail -12 "$err" >&2
        return 1
    }
}

Short_err="$WORK/longctx-short.stderr"

run short "$SENT" 32 >/dev/null || exit 1
run long "$LONG" 32 >/dev/null || exit 1
# The graph is one launch per token, so a phase split needs it off.
run profile "$LONG" 4 SOFUU_CUDA_PROFILE=1 >/dev/null || true
echo "   phase split:"
grep -E "cuda-profile" "$WORK/longctx-profile.stderr" | tail -2 | sed 's/^/   /' || true

echo ""
echo "== summary =="
short_line="$(grep -E "prefill:|decode:" "$WORK/longctx-short.stderr" | tail -1)"
long_line="$(grep -E "prefill:|decode:" "$WORK/longctx-long.stderr" | tail -1)"
short_dec="$(printf '%s' "$short_line" | sed -n 's/.*decode: [0-9]* tk in [^(]*(\([0-9.]*\) tk\/s).*/\1/p')"
long_dec="$(printf '%s' "$long_line" | sed -n 's/.*decode: [0-9]* tk in [^(]*(\([0-9.]*\) tk\/s).*/\1/p')"
printf 'short: %s\n' "$short_line"
printf 'long:  %s\n' "$long_line"
if [ -n "$short_dec" ] && [ -n "$long_dec" ] &&
   awk -v a="$short_dec" 'BEGIN{exit !(a>0)}'; then
    awk -v s="$short_dec" -v l="$long_dec" \
        'BEGIN{printf "decode long/short: %.1f%%  (%.1f -> %.1f tk/s)\n", 100*l/s, s, l}'
fi

if ! grep -qE "GPU decoder armed" "$WORK/longctx-long.stderr"; then
    echo "WARN: the long run did not report an armed CUDA decoder" >&2
fi
echo ""
echo "Read it as: decode flat with context = the weights dominate; decode"
echo "collapsing = the KV walk (attn bucket) dominates."
