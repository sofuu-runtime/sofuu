#!/usr/bin/env bash
# scripts/dist/all.sh — H4: build all platform packs for libsofuu
#
# Runs every platform pack that can build on the current host:
#   - macOS: macos.sh arm64 + x86_64
#   - Linux: linux.sh x86_64 + arm64 (requires Zig)
#   - iOS:   ios.sh all (requires Xcode — macOS only)
#
# Android is NOT included here (needs the NDK) — run android.sh separately.
# Windows/WASM are v2 — not yet supported.
#
# Usage: bash scripts/dist/all.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST_OS="$(uname -s)"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Sofuu libsofuu Platform Packs — H4                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

case "$HOST_OS" in
    Darwin)
        echo "  Host: macOS — building macOS + iOS packs"
        bash "$REPO_ROOT/scripts/dist/macos.sh" all
        if command -v xcodebuild &>/dev/null; then
            bash "$REPO_ROOT/scripts/dist/ios.sh" all
        else
            echo "  \033[33m⚠ Xcode not found — skipping iOS pack\033[0m"
        fi
        # Linux cross via Zig (if installed).
        if [ -x "$REPO_ROOT/tools/zig/zig" ]; then
            echo ""
            echo "  Zig found — cross-compiling Linux packs"
            bash "$REPO_ROOT/scripts/dist/linux.sh" all
        else
            echo "  \033[33m⚠ Zig not found — skipping Linux cross packs (run: make zig-install)\033[0m"
        fi
        ;;
    Linux)
        echo "  Host: Linux — building Linux pack (native)"
        bash "$REPO_ROOT/scripts/dist/linux.sh" "$(uname -m)"
        ;;
    *)
        echo "  \033[33m⚠ Unsupported host OS: $HOST_OS\033[0m"
        echo "  Run individual platform scripts manually."
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Platform packs complete                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Artifacts in dist/:"
ls -lh "$REPO_ROOT"/dist/libsofuu-* "$REPO_ROOT/dist/sofuu_embed.h" 2>/dev/null || true
if [ -d "$REPO_ROOT/dist/libsofuu.xcframework" ]; then
    echo ""
    echo "  xcframework: $(du -sh "$REPO_ROOT/dist/libsofuu.xcframework" | cut -f1)"
fi
echo ""
