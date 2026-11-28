#!/usr/bin/env bash
# scripts/check-ui-kit-filmstrip.sh - F4 gate for the Tier-2 app kit.
#
# Builds the gallery example once and renders it through `sofuu-emu`'s
# filmstrip matrix (iOS, Android, macOS, Windows, Linux). The gate fails when
# any profile reports a layout/contrast **error**; warnings are reported for
# design review but do not fail (the plan's acceptance bar is
# `audit.errors == 0`).
#
#   scripts/check-ui-kit-filmstrip.sh
#   SOFUU_BIN=/path/to/sofuu scripts/check-ui-kit-filmstrip.sh
#
# The binary must embed the current `src/js/ui*.bc.gz` snapshots; build with
# `make` after editing the kit.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

OUT="$ROOT/target/ui-kit-filmstrip"
mkdir -p "$OUT"

BIN="${SOFUU_BIN:-}"
if [ -z "$BIN" ]; then
    if [ -x "$ROOT/target/debug/sofuu" ]; then
        BIN="$ROOT/target/debug/sofuu"
    elif [ -x "$ROOT/sofuu" ]; then
        BIN="$ROOT/sofuu"
    else
        echo "ui-kit-filmstrip: no sofuu binary (run 'make' or set SOFUU_BIN)" >&2
        exit 1
    fi
fi

echo "ui-kit-filmstrip: $BIN"

# Preflight: the binary must embed the current `src/js/ui_kit.bc.gz`. A stale
# snapshot shows up as `kit.RangeSlider is not a function`, which reads like an
# example bug rather than a build one, so name it here.
PREFLIGHT="$(mktemp -t kit-preflight)"
cat >"$PREFLIGHT" <<'JS'
var missing = [];
if (!sofuu.uiKit || typeof sofuu.uiKit.RangeSlider !== 'function') missing.push('uiKit.RangeSlider');
if (!sofuu.uiKit || typeof sofuu.uiKit.Progress !== 'function') missing.push('uiKit.Progress');
if (typeof sofuu.ui.theme !== 'function') missing.push('ui.theme');
if (missing.length) { console.log('STALE ' + missing.join(',')); }
JS
if "$BIN" run "$PREFLIGHT" 2>/dev/null | grep -q '^STALE '; then
    echo "ui-kit-filmstrip: $BIN embeds a stale ui_kit snapshot" >&2
    echo "  rebuild the binary (make) so it carries src/js/ui_kit.bc.gz" >&2
    rm -f "$PREFLIGHT"
    exit 1
fi
rm -f "$PREFLIGHT"

"$BIN" ui snapshot examples/ui_kit_gallery.js \
    --platforms ios,android,macos,windows,linux \
    --mode filmstrip \
    --out "$OUT/kit_filmstrip.png" \
    --audit-json "$OUT/kit_audit.json" \
    --fail-on-error

python3 - "$OUT/kit_audit.json" <<'PY'
import json
import sys

path = sys.argv[1]
report = json.load(open(path))
summary = report.get("summary", {})
errors = int(summary.get("errors", 0))
warnings = int(summary.get("warnings", 0))
print(f"ui-kit-filmstrip: errors={errors} warnings={warnings} "
      f"interactive={summary.get('interactive_nodes', 0)}")
for issue in report.get("issues", []):
    if issue.get("severity") == "error":
        print(f"  ERROR {issue.get('kind')} {issue.get('node_id')}: {issue.get('detail')}")
if errors:
    sys.exit(1)
PY

echo "ui-kit-filmstrip: OK -> $OUT/kit_filmstrip.png"
