#!/usr/bin/env bash
# scripts/check_abi.sh — H6 CI gate: ABI export guard for libsofuu
#
# Diffs the exported symbols of the current build against a checked-in
# baseline. A new public symbol = deliberate PR change (update the baseline).
# A missing symbol = accidental removal (breaking change).
#
# Usage: bash scripts/check_abi.sh [lib-path]
#   lib-path: path to the .a or .dylib/.so (default: dist/libsofuu.a)
#
# CI: runs after `make libsofuu`; fails the build on unexpected symbol changes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="${1:-$REPO_ROOT/dist/libsofuu.a}"
BASELINE="$REPO_ROOT/scripts/abi_symbols.txt"

if [ ! -f "$LIB" ]; then
    echo "  \033[31m✗ Library not found: $LIB\033[0m"
    echo "  Run 'make libsofuu' first."
    exit 1
fi

# Extract exported symbols (T = text segment = exported functions).
# On macOS, symbols are prefixed with _; on Linux they're not.
# We normalize by stripping the leading _ and filtering to the PUBLIC API
# surface only (sofuu_rt_*, sofuu_embed_*, sofuu_free, and the legacy
# sofuu_init/eval/run/destroy/loop symbols). Internal helpers (sofuu_cma_*,
# sofuu_js_*, sofuu_tui_*, sofuu_uv_*, etc.) are NOT part of the ABI contract.
PUBLIC_RE='^(sofuu_rt_|sofuu_embed_|sofuu_free|sofuu_init|sofuu_destroy|sofuu_eval|sofuu_run_jobs|sofuu_get_engine|sofuu_engine_ctx|sofuu_flush_jobs|sofuu_loop_)'

SYMBOLS_FILE=$(mktemp)
if [[ "$(uname -s)" == "Darwin" ]]; then
    # .dylib: use -gU (extern only, defined). .a archive: use plain nm.
    if [[ "$LIB" == *.dylib ]] || [[ "$LIB" == *.so ]]; then
        nm -gU "$LIB" 2>/dev/null | grep ' T ' | awk '{print $NF}' | sed 's/^_//' | grep -E "$PUBLIC_RE" | sort -u > "$SYMBOLS_FILE"
    else
        # .a archive — nm lists symbols per object file; T = text (exported).
        nm "$LIB" 2>/dev/null | grep ' T ' | awk '{print $NF}' | sed 's/^_//' | grep -E "$PUBLIC_RE" | sort -u > "$SYMBOLS_FILE"
    fi
else
    nm -D "$LIB" 2>/dev/null | grep ' T ' | awk '{print $NF}' | grep -E "$PUBLIC_RE" | sort -u > "$SYMBOLS_FILE"
fi

if [ ! -s "$SYMBOLS_FILE" ]; then
    echo "  \033[33m⚠ No sofuu_ symbols found in $LIB\033[0m"
    echo "  (This may be expected if the library wasn't built with the capi crate.)"
    rm -f "$SYMBOLS_FILE"
    exit 0
fi

echo "  \033[36mABI symbol check\033[0m"
echo "  Library: $LIB"
echo "  Symbols found: $(wc -l < "$SYMBOLS_FILE")"
echo ""

# If no baseline exists, create one (first run).
if [ ! -f "$BASELINE" ]; then
    echo "  \033[33m⚠ No baseline at $BASELINE — creating one.\033[0m"
    echo "  Review and commit scripts/abi_symbols.txt."
    cp "$SYMBOLS_FILE" "$BASELINE"
    cat "$BASELINE"
    rm -f "$SYMBOLS_FILE"
    exit 0
fi

# Diff against the baseline.
NEW_SYMBOLS=$(comm -13 "$BASELINE" "$SYMBOLS_FILE")
MISSING_SYMBOLS=$(comm -23 "$BASELINE" "$SYMBOLS_FILE")

if [ -n "$MISSING_SYMBOLS" ]; then
    echo "  \033[31m✗ ABI BREAK: missing symbols (removed from library):\033[0m"
    echo "$MISSING_SYMBOLS" | sed 's/^/    /'
    echo ""
    echo "  These symbols were in the baseline but not the current build."
    echo "  If this is intentional (breaking change), update scripts/abi_symbols.txt."
    rm -f "$SYMBOLS_FILE"
    exit 1
fi

if [ -n "$NEW_SYMBOLS" ]; then
    echo "  \033[33m⚠ New symbols (not in baseline):\033[0m"
    echo "$NEW_SYMBOLS" | sed 's/^/    /'
    echo ""
    echo "  If these are intentional (new public API), update the baseline:"
    echo "    cp $SYMBOLS_FILE $BASELINE"
    echo "  Then commit scripts/abi_symbols.txt."
    rm -f "$SYMBOLS_FILE"
    # Don't fail — new symbols are additive, not breaking. But warn loudly.
    exit 0
fi

echo "  \033[32m✓ ABI symbols match baseline exactly.\033[0m"
echo ""
cat "$SYMBOLS_FILE"
rm -f "$SYMBOLS_FILE"
