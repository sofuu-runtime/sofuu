#!/usr/bin/env bash
# scripts/gen-driver-bc.sh - recompile every shipped driver snapshot.
#
# Run after editing anything in src/js/, then rebuild (make): a release
# build embeds the .bc snapshots and NOT the sources, so the snapshot IS the
# shipped driver code.
#
#   scripts/gen-driver-bc.sh && make && make bc-check
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
if [ ! -f "$SOFUU" ]; then
    echo "sofuu binary not found - run 'make' first" >&2
    exit 1
fi

cd "$ROOT" || exit 1
for js in src/js/*.js; do
    # Keep source-only files out of the snapshot lane (same marker the check
    # script uses; see memory_api.js).
    if grep -q '@sofuu-source-only' "$js"; then
        continue
    fi
    name="$(basename "$js" .js)"
    "$SOFUU" compile-bc "$js" "src/js/$name.bc" >/dev/null
    gzip -9 -n -k -f "src/js/$name.bc"
    echo "  compiled $name.js -> src/js/$name.bc ($(wc -c < "src/js/$name.bc" | tr -d ' ') B, gz: $(wc -c < "src/js/$name.bc.gz" | tr -d ' ') B)"
done
echo "GEN-DRIVER-BC: done - now run 'make' so the binary embeds them"
