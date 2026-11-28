#!/usr/bin/env bash
# scripts/check_ui_js_sync.sh — the JS/TS drift gate for the sofuu-ui split.
#
# PLAN-SPLIT-UI-EFFECTS-EXTERNAL-REPO §5 requires:
#
#   diff -u src/js/ui.js deps/sofuu-ui/js/ui.js
#
# so an edit cannot land in the host copy without upstreaming it to the UI
# repository (or vice versa after cutover). `SOFUU_UI_DIR` supports a sibling
# checkout, exactly like `SOFUU_QTSQ_DIR` elsewhere in this Makefile.
#
# Pre-split the same check compares the runtime copy with the split staging
# area at `crates/sofuu-ui/js/`. Both directions are byte comparisons; the
# sync script owns the copy.
#
# Usage: bash scripts/check_ui_js_sync.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

UI_DIR="${SOFUU_UI_DIR:-}"
if [ -z "$UI_DIR" ] && [ -f deps/sofuu-ui/js/ui.js ]; then
    UI_DIR="deps/sofuu-ui"
fi
if [ -z "$UI_DIR" ] && [ -f crates/sofuu-ui/js/ui.js ]; then
    UI_DIR="crates/sofuu-ui"
fi

if [ -z "$UI_DIR" ] || [ ! -d "$UI_DIR/js" ]; then
    echo "ui-js-sync: no sofuu-ui checkout found (deps/sofuu-ui, SOFUU_UI_DIR, crates/sofuu-ui/js)" >&2
    exit 1
fi
# `SOFUU_UI_DIR` is commonly absolute; keep the displayed path relative to the
# workspace but address files through one absolute root.
case "$UI_DIR" in
    /*) UI_ABS="$UI_DIR" ;;
    *)  UI_ABS="$ROOT/$UI_DIR" ;;
esac

echo "ui-js-sync: $UI_DIR/js <-> src/js"
fail=0
checked=0
for name in ui.js ui_kit.js fx.js; do
    src="$ROOT/src/js/$name"
    up="$UI_ABS/js/$name"
    if [ ! -f "$src" ] && [ ! -f "$up" ]; then
        continue
    fi
    checked=$((checked + 1))
    if [ ! -f "$src" ] || [ ! -f "$up" ]; then
        echo "  ✗ $name exists on only one side (src/js vs $UI_DIR/js)" >&2
        fail=1
        continue
    fi
    if ! cmp -s "$src" "$up"; then
        echo "  ✗ $name differs:" >&2
        diff -u "$up" "$src" | sed -n '1,24p' | sed 's/^/    /' >&2 || true
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "  Run 'make sync-ui-js' to reconcile (direction follows the checkout):" >&2
    echo "    pre-split  src/js -> crates/sofuu-ui/js" >&2
    echo "    post-split deps/sofuu-ui/js -> src/js (+ bytecode)" >&2
    exit 1
fi

echo "  ✓ $checked JS source(s) identical"

# The standalone package also carries the UI/FX type declarations extracted
# from the host `sofuu.d.ts`. Resolve the same checkout the JS gate resolved,
# so the check survives the cutover from `crates/sofuu-ui` to `deps/sofuu-ui`.
DTS="$UI_ABS/js/index.d.ts"
if [ -f "$DTS" ]; then
    bash "$ROOT/scripts/sync_ui_dts.sh" --check
    # Typecheck the module-shaped declarations when the host has a TS
    # toolchain; a fresh clone without node_modules still runs the byte gate.
    TSC="$ROOT/sofuu-desktop/node_modules/.bin/tsc"
    if [ -x "$TSC" ]; then
        if "$TSC" --noEmit --strict --target ES2022 --lib ES2022,DOM \
            "$DTS" >/dev/null 2>&1; then
            echo "  ✓ js/index.d.ts typechecks"
        else
            echo "  ✗ js/index.d.ts does not typecheck" >&2
            "$TSC" --noEmit --strict --target ES2022 --lib ES2022,DOM \
                "$DTS" >&2 || true
            exit 1
        fi
    fi
fi
