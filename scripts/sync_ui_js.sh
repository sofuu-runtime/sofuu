#!/usr/bin/env bash
# scripts/sync_ui_js.sh — move the JS SDK across the sofuu-ui repo boundary.
#
# The authoritative UI/effects JS lives in the external `sofuu-ui` repository
# (`js/ui.js`, `js/ui_kit.js`). The host keeps a generated copy at `src/js/`
# because `crates/sofuu-core/src/shipped.rs` embeds it into the binary and the
# headless `--no-default-features` build must work without a submodule
# (PLAN-SPLIT-UI-EFFECTS-EXTERNAL-REPO §4.4, Option A).
#
# Direction:
#   post-split  deps/sofuu-ui/js/ (or $SOFUU_UI_DIR/js/) -> src/js/
#               and rebuilds ui.bc / ui.bc.gz / ui_kit.bc / ui_kit.bc.gz
#   pre-split   src/js/ -> crates/sofuu-ui/js/   (stage the split snapshot)
#
# An explicit --to-ui / --from-ui overrides the inferred direction.
#
# Usage:
#   bash scripts/sync_ui_js.sh [--to-ui|--from-ui] [--check]
#   SOFUU_UI_DIR=~/projects/sofuu-ui bash scripts/sync_ui_js.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mode="auto"
if [ "${1:-}" = "--to-ui" ]; then
    mode="to-ui"
elif [ "${1:-}" = "--from-ui" ]; then
    mode="from-ui"
elif [ "${1:-}" = "--check" ]; then
    exec bash scripts/check_ui_js_sync.sh
elif [ -n "${1:-}" ]; then
    echo "sync-ui-js: unknown argument: $1" >&2
    exit 2
fi

UI_DIR="${SOFUU_UI_DIR:-}"
if [ -z "$UI_DIR" ] && [ -d deps/sofuu-ui/js ]; then
    UI_DIR="deps/sofuu-ui"
fi
if [ -z "$UI_DIR" ] && [ -d crates/sofuu-ui/js ]; then
    UI_DIR="crates/sofuu-ui"
fi
if [ -z "$UI_DIR" ] || [ ! -d "$UI_DIR/js" ]; then
    echo "sync-ui-js: no UI JS source found" >&2
    echo "  expected deps/sofuu-ui/js, \$SOFUU_UI_DIR/js, or crates/sofuu-ui/js" >&2
    exit 1
fi
case "$UI_DIR" in
    /*) UI_ABS="$UI_DIR" ;;
    *)  UI_ABS="$ROOT/$UI_DIR" ;;
esac

if [ "$mode" = auto ]; then
    case "$UI_DIR" in
        crates/sofuu-ui) mode="to-ui" ;;  # pre-split: the runtime copy is canonical
        *) mode="from-ui" ;;
    esac
fi

names=()
for name in ui.js ui_kit.js fx.js; do
    if [ -f "$UI_DIR/js/$name" ] || [ -f "src/js/$name" ]; then
        names+=("$name")
    fi
done

changed=0
copy() { # <from-dir> <to-dir>
    local from="$1" to="$2" name
    mkdir -p "$to"
    for name in "${names[@]}"; do
        if [ ! -f "$from/$name" ]; then
            continue
        fi
        if cmp -s "$from/$name" "$to/$name"; then
            echo "  = $name"
        else
            cp -p "$from/$name" "$to/$name"
            changed=1
            echo "  -> $name"
        fi
    done
}

if [ "$mode" = to-ui ]; then
    echo "sync-ui-js: src/js -> $UI_DIR/js"
    copy src/js "$UI_ABS/js"
    # Type declarations are derived from the host `sofuu.d.ts` while it owns
    # the global `sofuu.ui` surface (pre-split direction).
    if [ -f scripts/sync_ui_dts.sh ]; then
        bash scripts/sync_ui_dts.sh
    fi
    echo "  staged; run 'make bc-check' if you also changed the runtime copy"
else
    echo "sync-ui-js: $UI_DIR/js -> src/js"
    copy "$UI_ABS/js" src/js

    # The binary embeds bytecode snapshots, not the source (QTSQ Phase 6).
    # A no-op sync (the usual post-submodule-update case) reuses the committed
    # snapshots, so a fresh clone with no built binary can still run it.
    needs_bc=0
    for name in "${names[@]}"; do
        [ -f "src/js/$name" ] || continue
        out="src/js/${name%.js}.bc"
        if [ "$changed" -eq 1 ] || [ ! -f "$out" ] || [ ! -f "$out.gz" ]; then
            needs_bc=1
        fi
    done
    if [ "$needs_bc" -eq 0 ]; then
        echo "  bytecode already current (sources unchanged)"
        echo "  verify with 'make bc-check'"
    else
        BIN="${SOFUU_BIN:-}"
        if [ -z "$BIN" ]; then
            for candidate in ./sofuu target/debug/sofuu target/release/sofuu; do
                if [ -x "$candidate" ]; then
                    BIN="$candidate"
                    break
                fi
            done
        fi
        if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
            echo "  ✗ no sofuu binary to compile bytecode; run 'make' first" >&2
            exit 1
        fi
        for name in "${names[@]}"; do
            [ -f "src/js/$name" ] || continue
            out="src/js/${name%.js}.bc"
            "$BIN" compile-bc "src/js/$name" "$out" >/dev/null
            gzip -9 -n -k -f "$out"
            echo "  bc ${out#"$ROOT/"} ($(wc -c <"$out" | tr -d ' ') B)"
        done
        echo "  verify with 'make bc-check'"
    fi
    if [ -f "$UI_ABS/js/index.d.ts" ]; then
        bash scripts/sync_ui_dts.sh --check
    fi
fi
