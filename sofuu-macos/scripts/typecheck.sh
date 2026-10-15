#!/usr/bin/env bash
# sofuu-macos/scripts/typecheck.sh — fast full-app typecheck without the engine
#
# Mirrors the iOS flow (scripts/mobile/typecheck_ios.sh): `import Sofuu`
# resolves to the header-only stub in .devstub (refreshed from
# include/sofuu_embed.h), so a Swift error shows in seconds instead of
# waiting on the multi-minute Rust cdylib build. Type-only: nothing links.
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"   # sofuu-macos/
REPO="$DIR/.."

cp "$REPO/include/sofuu_embed.h" "$DIR/.devstub/sofuu_embed.h"

cd "$REPO"
SOURCES=$(find "$DIR/SofuuMac" -name '*.swift' | sort)

xcrun -sdk macosx swiftc -typecheck \
    -sdk "$(xcrun -sdk macosx -show-sdk-path)" \
    -target arm64-apple-macos13.0 \
    -I "$DIR/.devstub" \
    -swift-version 5 \
    $SOURCES

echo "  \033[32m✓\033[0m sofuu-macos typecheck clean ($(echo "$SOURCES" | wc -l | tr -d ' ') files)"
