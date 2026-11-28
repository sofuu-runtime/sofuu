#!/usr/bin/env bash
# scripts/sync_ui_dts.sh — derive the standalone UI declarations.
#
# The host's `sofuu.d.ts` still owns the global `sofuu.ui` / `sofuu.fx`
# declarations. The external `sofuu-ui` package needs the same types as a
# module (`import type { SofuuUi, UiNode } from 'sofuu-ui'`), so this extracts
# the contiguous UI/FX section into the resolved checkout's `js/index.d.ts`
# (`SOFUU_UI_DIR`, `deps/sofuu-ui`, or `crates/sofuu-ui`).
#
# The extraction is marker-based, not line-number-based, so unrelated edits to
# `sofuu.d.ts` do not churn the copy. Run with --check in CI (the JS drift gate
# does) and without it after changing a UI declaration.
#
# Usage: bash scripts/sync_ui_dts.sh [--check]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Same checkout resolution as scripts/sync_ui_js.sh: an explicit SOFUU_UI_DIR
# wins, then the post-split submodule, then the pre-split in-tree crate.
UI_DIR="${SOFUU_UI_DIR:-}"
if [ -z "$UI_DIR" ] && [ -d deps/sofuu-ui/js ]; then
    UI_DIR="deps/sofuu-ui"
fi
if [ -z "$UI_DIR" ] && [ -d crates/sofuu-ui/js ]; then
    UI_DIR="crates/sofuu-ui"
fi
if [ -z "$UI_DIR" ]; then
    echo "sync-ui-dts: no sofuu-ui checkout found" >&2
    exit 1
fi
OUT="$UI_DIR/js/index.d.ts"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

python3 - "$ROOT/sofuu.d.ts" "$TMP" <<'PY'
import sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()

start_marker = ("/* ==================================================================\n"
                " * Native 2D GPU UI (`sofuu.ui`, enabled with `--ui`)\n")
end_marker = "/** What the running binary was compiled with"

start = text.find(start_marker)
if start < 0:
    raise SystemExit("sync-ui-dts: UI section start marker not found in sofuu.d.ts")
end = text.find(end_marker, start)
if end < 0:
    raise SystemExit("sync-ui-dts: UI section end marker not found in sofuu.d.ts")

section = text[start:end].rstrip() + "\n"

header = '''/**
 * `sofuu-ui` — declarative scene, widgets and effects surface.
 *
 * Generated from `sofuu.d.ts` in the host while the UI declarations still
 * live there (PLAN-SPLIT-UI-EFFECTS-EXTERNAL-REPO §2.1). The host owns the
 * global `sofuu.ui` / `sofuu.fx` types; this module is what a standalone
 * consumer imports.
 *
 * Do not edit by hand: `bash scripts/sync_ui_dts.sh`.
 */

// ---8<--- generated from sofuu.d.ts (do not edit below) ---

'''
open(dst, "w").write(header + section)
PY

if [ "$CHECK" -eq 1 ]; then
    if cmp -s "$TMP" "$OUT"; then
        echo "  ✓ js/index.d.ts matches sofuu.d.ts"
        exit 0
    fi
    echo "  ✗ js/index.d.ts drifted from sofuu.d.ts:" >&2
    diff -u "$OUT" "$TMP" | sed -n '1,24p' | sed 's/^/    /' >&2 || true
    echo "    run 'make sync-ui-dts' to regenerate" >&2
    exit 1
fi

cp "$TMP" "$OUT"
echo "sync-ui-dts: wrote $OUT"
