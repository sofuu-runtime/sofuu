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
# surface only (sofuu_rt_*, sofuu_embed_*, sofuu_voice_*, sofuu_free, and the
# legacy sofuu_init/eval/run/destroy/loop symbols). Internal helpers
# (sofuu_cma_*, sofuu_js_*, sofuu_tui_*, sofuu_uv_*, etc.) are NOT part of the
# ABI contract.
#
# 2026-09-25: sofuu_voice_* was MISSING from this allowlist while sitting in
# the baseline. A symbol that the filter hides can never be detected as
# added OR removed, so the gate silently passed while the two lists
# disagreed — it reported the voice functions as "removed from library"
# on a build that exports them. Any prefix listed in abi_symbols.txt MUST
# also be listed here; assert_abi_prefix_coverage (below) enforces that.
PUBLIC_RE='^(sofuu_rt_|sofuu_embed_|sofuu_voice_|sofuu_free|sofuu_init|sofuu_destroy|sofuu_eval|sofuu_run_jobs|sofuu_get_engine|sofuu_engine_ctx|sofuu_flush_jobs|sofuu_loop_)'

SYMBOLS_FILE=$(mktemp)
BASELINE_SORTED=$(mktemp)
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
    rm -f "$SYMBOLS_FILE" "$BASELINE_SORTED"
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
    rm -f "$SYMBOLS_FILE" "$BASELINE_SORTED"
    exit 0
fi

# Diff against the baseline.
#
# 2026-09-25: `comm` is byte-order and REQUIRES both inputs sorted in the
# same collation. The symbol list is produced with `sort -u` (locale
# collation) while the checked-in baseline is plain ASCII, and under the
# C locale the two orders differ on underscore-prefixed names — so comm
# reported `sofuu_run_jobs` as REMOVED from a library that exports it.
# That is a false ABI break, and a gate that cries wolf gets ignored.
# Force LC_ALL=C on both the sort and the comm so the comparison is
# byte-exact and reproducible on any machine.
LC_ALL=C sort -u "$BASELINE" -o "$BASELINE_SORTED"
LC_ALL=C sort -u "$SYMBOLS_FILE" -o "$SYMBOLS_FILE"
NEW_SYMBOLS=$(LC_ALL=C comm -13 "$BASELINE_SORTED" "$SYMBOLS_FILE")
MISSING_SYMBOLS=$(LC_ALL=C comm -23 "$BASELINE_SORTED" "$SYMBOLS_FILE")

if [ -n "$MISSING_SYMBOLS" ]; then
    echo "  \033[31m✗ ABI BREAK: missing symbols (removed from library):\033[0m"
    echo "$MISSING_SYMBOLS" | sed 's/^/    /'
    echo ""
    echo "  These symbols were in the baseline but not the current build."
    echo "  If this is intentional (breaking change), update scripts/abi_symbols.txt."
    rm -f "$SYMBOLS_FILE" "$BASELINE_SORTED"
    exit 1
fi

if [ -n "$NEW_SYMBOLS" ]; then
    echo "  \033[33m⚠ New symbols (not in baseline):\033[0m"
    echo "$NEW_SYMBOLS" | sed 's/^/    /'
    echo ""
    # P1-13 (AUDIT-2026-09-01): new symbols used to WARN and exit 0 — an
    # accidental export (typo'd #[no_mangle]) passed the CI gate with a
    # log line. Now FAIL unless the addition is explicitly acknowledged
    # via SOFUU_ABI_ALLOW_NEW=1 (set it when deliberately adding API).
    if [ "${SOFUU_ABI_ALLOW_NEW:-0}" = "1" ]; then
        echo "  SOFUU_ABI_ALLOW_NEW=1 set — treating as an acknowledged addition."
        echo "  Update the baseline: cp $SYMBOLS_FILE $BASELINE && commit scripts/abi_symbols.txt."
        rm -f "$SYMBOLS_FILE" "$BASELINE_SORTED"
        exit 0
    fi
    echo "  \033[31m✗ ABI ADDITION without acknowledgment:\033[0m"
    echo "  If these are intentional (new public API), re-run with:"
    echo "    SOFUU_ABI_ALLOW_NEW=1 $0"
    echo "  then update the baseline: cp $SYMBOLS_FILE $BASELINE"
    echo "  and commit scripts/abi_symbols.txt."
    rm -f "$SYMBOLS_FILE" "$BASELINE_SORTED"
    exit 1
fi

echo "  \033[32m✓ ABI symbols match baseline exactly.\033[0m"
echo ""
cat "$SYMBOLS_FILE"
rm -f "$SYMBOLS_FILE" "$BASELINE_SORTED"
