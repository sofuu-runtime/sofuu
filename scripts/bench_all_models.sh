#!/usr/bin/env bash
# Benchmarks every model on disk through `sofuu serve --infer` (CPU) and
# appends one JSON line per model: {model, prefill_tps, decode_tps, ...}.
# Reports land in docs/bench/all-models.jsonl (append-only history).
#
# Usage: scripts/bench_all_models.sh [binary]    (default: target/release/sofuu)
set -u
BIN="${1:-target/release/sofuu}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/docs/bench/all-models.jsonl"
mkdir -p "$REPO/docs/bench"
MODELS=(
  "$HOME/Downloads/liqiud-350"
  "$HOME/Downloads/liquid"
  "$HOME/Downloads/gpt-x3-150m"
  "$HOME/Downloads/gemma-2b"
  "$HOME/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf"
  "$HOME/Downloads/Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf"
  "$HOME/Downloads/Spark-X2.5-4B-Q4_K_M.gguf"
  "$HOME/Downloads/Qwen3-8B-Q4_K_M.gguf"
  "$HOME/Downloads/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
)
pkill -f "sofuu serve --infer" 2>/dev/null; sleep 2
for M in "${MODELS[@]}"; do
  [ -e "$M" ] || { echo "SKIP (missing): $M"; continue; }
  NAME=$(basename "$M")
  PORT=$((8940 + RANDOM % 40))
  "$BIN" serve --infer -m "$M" --cpu --port "$PORT" >"/tmp/sofuu-bench-$NAME.log" 2>&1 &
  SPID=$!
  sleep 6
  R=$(python3 "$REPO/scripts/bench_serve.py" "$PORT" 16 2>/dev/null)
  if [ -z "$R" ]; then
    R='{"error": "no response (load timeout or unsupported backend)"}'
    sleep 240  # give big-model loads one last chance to fail loudly
    R=$(python3 "$REPO/scripts/bench_serve.py" "$PORT" 16 2>/dev/null)
  fi
  python3 -c "
import json, sys
r = json.loads('''$R''')
r['model'] = '$NAME'
r['ts'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
print(json.dumps(r))
" >> "$OUT"
  tail -1 "$OUT"
  kill "$SPID" 2>/dev/null
  pkill -f "sofuu serve --infer -m $M" 2>/dev/null
  sleep 3
done
echo "=== done: $OUT ==="
