#!/usr/bin/env bash
# scripts/cross/cross_compile.sh
# ─────────────────────────────────────────────────────────────────────────────
# Sofuu Cross-Compilation Orchestrator
#
# Produces fully-static Linux binaries from macOS (or CI) using Zig as CC.
# No Docker, no sysroot, no package manager needed — just Zig.
#
# Usage:
#   bash scripts/cross/cross_compile.sh              # build all targets
#   bash scripts/cross/cross_compile.sh x86_64       # one target
#   bash scripts/cross/cross_compile.sh arm64        # one target
#
# First-time setup (one-off):
#   bash scripts/cross/install_zig.sh
#
# Output artefacts in dist/:
#   dist/sofuu-linux-x86_64   (runs on any x86_64 Linux including AWS/GCP)
#   dist/sofuu-linux-arm64    (runs on Graviton / RPi / Oracle ARM)
#   dist/sofuu-linux-x86_64.tar.gz
#   dist/sofuu-linux-arm64.tar.gz
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$REPO_ROOT/dist"
ZIG="$REPO_ROOT/tools/zig/zig"

ARG="${1:-all}"

# ── Prerequisite check ────────────────────────────────────────
if [ ! -x "$ZIG" ]; then
    echo ""
    echo "  Zig compiler not found at: $ZIG"
    echo ""
    echo "  Install it with:"
    echo "    bash scripts/cross/install_zig.sh"
    echo ""
    exit 1
fi
echo "  Using Zig: $($ZIG version)"

# ── Helper: build one target ──────────────────────────────────
build_target() {
    local name="$1"   # x86_64 | arm64
    local script="$REPO_ROOT/scripts/cross/build_linux_${name}.sh"
    if [ ! -f "$script" ]; then
        echo "No build script for: $name"
        return 1
    fi
    bash "$script"
}

# ── Build ──────────────────────────────────────────────────────
mkdir -p "$DIST"

if [ "$ARG" = "all" ]; then
    build_target x86_64
    build_target arm64
else
    build_target "$ARG"
fi

# ── Create compressed archives ─────────────────────────────────
echo "→ Creating release archives..."
cd "$DIST"
for bin in sofuu-linux-*; do
    [ -f "$bin" ] || continue
    [ -x "$bin" ] || continue
    chmod +x "$bin"
    tar -czf "${bin}.tar.gz" "$bin"
    echo "   ✓ ${bin}.tar.gz ($(du -sh "${bin}.tar.gz" | cut -f1))"
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Cross-compilation complete                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
ls -lh "$DIST"/sofuu-linux-* 2>/dev/null || true
echo ""
echo "  Test on Linux with:"
echo "    rsync dist/sofuu-linux-x86_64 user@server:~/sofuu"
echo "    ssh user@server './sofuu version'"
echo ""
