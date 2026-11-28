#!/usr/bin/env bash
# Capture the real sofuu-android-v-2 assets through sofuu-emu and compare the
# light/dark frames with the v1 goldens.
#
# The bundle is generated from the same files the Android host evaluates
# (theme.js, icons.js, bridge.js, screens/home.js) plus emu_env.js, which maps
# __sofuu_emu_device to __v2. Captures land in sofuu-android-v-2/captures/emu/.
#
# Usage:
#   scripts/capture_v2_emu.sh
#   SOFUU_BIN=/path/to/sofuu scripts/capture_v2_emu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOFUU_BIN="${SOFUU_BIN:-$ROOT/target/debug/sofuu}"
BUNDLE="$ROOT/build/v2_ui.bundle.js"
OUT="$ROOT/sofuu-android-v-2/captures/emu"
GOLDENS="$ROOT/sofuu-android-v-2/captures"

if [[ ! -x "$SOFUU_BIN" ]]; then
  echo "error: $SOFUU_BIN is missing; build it first (cargo build -p sofuu-core)" >&2
  exit 1
fi
if ! command -v node >/dev/null 2>&1; then
  echo "error: node is required to generate the v2 bundle" >&2
  exit 1
fi

node "$ROOT/build/v2_ui_bundle.js" --out "$BUNDLE" >/dev/null
mkdir -p "$OUT"

shot() {
  local name="$1"
  local device="$2"
  shift 2
  echo "== $name ($device) =="
  "$SOFUU_BIN" ui snapshot "$BUNDLE" \
    --platform android \
    --device "$device" \
    "$@" \
    --out "$OUT/$name.png" \
    --audit-json "$OUT/$name.audit.json" \
    --fail-on-error
}

shot v2_home_light sofuu-phone-411 --theme light --mode clean --props '{"workspace":"workspace"}'
shot v2_home_dark sofuu-phone-411 --theme dark --mode clean --props '{"workspace":"workspace"}'
shot v2_home_light_annotated sofuu-phone-411 --theme light --mode annotated --props '{"workspace":"workspace"}'
shot v2_home_light_keyboard sofuu-phone-411 --theme light --mode clean --keyboard-open --props '{"workspace":"workspace"}'
shot v2_home_narrow budget-small --theme light --mode clean --props '{"workspace":"workspace"}'

echo
echo "== v1 golden comparison =="
for theme in light dark; do
  golden="$GOLDENS/v1_home_$theme.png"
  capture="$OUT/v2_home_$theme.png"
  if [[ -f "$golden" ]]; then
    python3 "$ROOT/scripts/compare_v2_goldens.py" "$golden" "$capture"
  else
    echo "missing v1 golden: $golden (skipping $theme comparison)"
  fi
done

echo
echo "captures: $OUT"
