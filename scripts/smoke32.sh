#!/usr/bin/env bash
# smoke32.sh — exercise local inference with an existing real-weight GGUF.
# A maximum of 32 tokens is requested; the model may stop earlier. The
# synthetic contract fixture is a format fixture, not an inference model.
#
# Usage: bash scripts/smoke32.sh [binary] [model]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:-$ROOT/sofuu}"
[ -x "$BIN" ] || BIN="$ROOT/target/release/sofuu"
MODEL="${2:-${SOFUU_CONTRACT_MODEL:-}}"

if [ ! -x "$BIN" ]; then
    echo "FAIL smoke32: binary not found ($BIN)"; exit 1
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "BLOCKED smoke32: supply an existing compatible real-weight GGUF as argument 2 or SOFUU_CONTRACT_MODEL"; exit 2
fi
case "$MODEL" in
    */sofuu-fixture.gguf|sofuu-fixture.gguf)
        echo "BLOCKED smoke32: the synthetic contract fixture is not real inference coverage"; exit 2 ;;
esac

OUT="$("$BIN" infer "$MODEL" "The quick brown fox" --max-tokens 32 2>&1)"
rc=$?

if [ $rc -ne 0 ]; then
    echo "FAIL smoke32: infer exited $rc"; exit 1
fi

# The inference summary reports actual decoded tokens; banner and diagnostics
# alone must not count as output. Do not print generated content in CI logs.
if ! printf '%s\n' "$OUT" | grep -Eq 'decode: [1-9][0-9]* tk'; then
    echo "FAIL smoke32: inference did not report any decoded tokens"; exit 1
fi

echo "OK  smoke32: real inference decoded at least one token (limit 32) on $(uname -s)-$(uname -m)"
