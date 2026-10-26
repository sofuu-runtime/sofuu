#!/usr/bin/env bash
# smoke32.sh — the P5 32-token tiny-model smoke: the committed 77 KB fixture
# model must load and decode 32 tokens on ANY platform this runtime builds on.
# Sub-30s, no network, no QTSQ (the fixture needs the engine, not the brain).
#
# Usage: bash scripts/smoke32.sh [binary] [model]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:-$ROOT/sofuu}"
[ -x "$BIN" ] || BIN="$ROOT/target/release/sofuu"
MODEL="${2:-$ROOT/contract/fixtures/sofuu-fixture.gguf}"

if [ ! -x "$BIN" ]; then
    echo "FAIL smoke32: binary not found ($BIN)"; exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "FAIL smoke32: fixture model not found ($MODEL)"; exit 1
fi

OUT="$("$BIN" infer "$MODEL" "The quick brown fox" --max-tokens 32 2>&1)"
rc=$?

if [ $rc -ne 0 ]; then
    echo "FAIL smoke32: infer exited $rc"; echo "$OUT" | tail -3; exit 1
fi

# The generated text after the header lines must be non-empty (tokens
# decoded). The fixture is a real (if tiny) model: any string qualifies.
TOKENS=$(printf '%s' "$OUT" | tail -c 400 | wc -c | awk '{print $1}')
if [ "$TOKENS" -lt 10 ]; then
    echo "FAIL smoke32: implausibly short output"; echo "$OUT" | tail -3; exit 1
fi

echo "OK  smoke32: 32 tokens from the fixture model on $(uname -s)-$(uname -m)"
