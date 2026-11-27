#!/usr/bin/env bash
# scripts/check_ui_c_abi.sh — Phase C2 C integration gate.
#
# Compiles scripts/ui_c_abi_smoke.c against the REAL dist/libsofu.dylib and
# runs it: session create/atlas/insets, attach_runtime + JS mount, DrawList
# polling, click -> semantic event, springs settle (needs_next_frame == 0),
# session-first free tombstone, detach, rt_free.
#
# Mode B surface symbols must answer SOFUU_ERR_NOT_IMPL (-2).
#
# Usage: bash scripts/check_ui_c_abi.sh   (runs `make libsofuu` first)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

make libsofuu >/dev/null

LIB=""
for candidate in dist/libsofuu.dylib dist/libsofuu.so; do
    if [ -f "$candidate" ]; then
        LIB="$candidate"
        break
    fi
done
if [ -z "$LIB" ]; then
    echo "  ✗ no UI-enabled libsofuu in dist/" >&2
    exit 1
fi

OUT="target/ui_c_abi_smoke"
mkdir -p target
cc scripts/ui_c_abi_smoke.c -Idist -Ldist -lsofuu \
    -Wl,-rpath,"$REPO_ROOT/dist" -o "$OUT"

set +e
"$OUT"
rc=$?
set -e
if [ $rc -ne 0 ]; then
    echo "  ✗ C ABI smoke failed (exit $rc)" >&2
    exit $rc
fi
