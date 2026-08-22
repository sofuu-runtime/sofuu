#!/usr/bin/env bash
# scripts/cross/build_linux_x86_64.sh
#
# Cross-compile Sofuu for Linux x86_64 from macOS (or CI) using Zig as CC.
# Zig bundles its own libc and linker — no sysroot or musl package needed.
#
# M10: the runtime is Rust (crates/) over the C engine layer, so this now
# runs a cargo cross-build: rustc links with zig (static musl), the cc
# crate compiles QuickJS/SIMD/http-parser with `zig cc`, and libuv +
# libcurl are prebuilt for the target (dist/). QTSQ is skipped — the
# proprietary checkout is macOS-only; CI covers the same degraded build.
#
# Usage:
#   bash scripts/cross/install_zig.sh     # once
#   bash scripts/cross/build_linux_x86_64.sh
#
# Output: dist/sofuu-linux-x86_64  (statically linked)
#
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIG="$REPO_ROOT/tools/zig/zig"
DIST="$REPO_ROOT/dist"

if [ ! -x "$ZIG" ]; then
    echo "Zig not found at $ZIG"
    echo "Run: bash scripts/cross/install_zig.sh"
    exit 1
fi

TARGET="x86_64-linux-musl"
RUST_TARGET="x86_64-unknown-linux-musl"
OUT="$DIST/sofuu-linux-x86_64"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Cross-compiling Sofuu → Linux x86_64 (musl static)        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$DIST"
cd "$REPO_ROOT"

# ── rustup target (std for musl) ─────────────────────────────────
if ! rustup target list --installed | grep -q "^${RUST_TARGET}$"; then
    echo "→ Adding rustup target ${RUST_TARGET}..."
    rustup target add "$RUST_TARGET"
fi

# ── Build libuv for the target ──────────────────────────────────
echo "→ Building libuv for ${TARGET}..."
UV_DIR="deps/libuv"
UV_BUILD="$UV_DIR/build-linux-x86_64"

# Use zig cc as CMake toolchain via environment variables
export CC="$ZIG cc -target $TARGET"
export CXX="$ZIG c++ -target $TARGET"
export AR="$ZIG ar"
export RANLIB="$ZIG ranlib"

cmake -S "$UV_DIR" -B "$UV_BUILD" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_C_COMPILER="$ZIG" \
    -DCMAKE_C_COMPILER_ARG1="cc -target $TARGET" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
    -DCMAKE_C_FLAGS="-target $TARGET" \
    -DCMAKE_EXE_LINKER_FLAGS="-target $TARGET -static" \
    2>&1 | tail -3
cmake --build "$UV_BUILD" --target uv_a -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) 2>&1 | tail -5

UV_LIB="$UV_BUILD/libuv.a"
echo "✓ libuv built: $UV_LIB"

# ── Build libcurl for the target ────────────────────────────────
echo "→ Building static libcurl for ${TARGET}..."
bash "$REPO_ROOT/scripts/cross/build_libcurl_static.sh" "$TARGET"
CURL_INSTALL="$DIST/libcurl-$TARGET"
CURL_LIB="$CURL_INSTALL/lib"

# ── Cargo cross-build (compiles QuickJS/SIMD/http-parser via zig cc) ──
echo "→ cargo build --release --target ${RUST_TARGET}"
echo "   (cc crate → zig cc; linker → zig; QTSQ-free; see PLAN-RUST-MIGRATION M10)"
export CC="$ZIG cc -target $TARGET"
export CXX="$ZIG c++ -target $TARGET"
export AR="$ZIG ar"
export RANLIB="$ZIG ranlib"
export SOFUU_UV_DIR="$UV_BUILD"
export SOFUU_CURL_STATIC_DIR="$CURL_LIB"
# QTSQ is macOS-only — point at an empty dir so the probe fails cleanly.
mkdir -p "$DIST/no-qtsq"
export SOFUU_QTSQ_DIR="$DIST/no-qtsq"
export RUSTFLAGS="-C linker=$ZIG -C link-arg=cc -C link-arg=-target -C link-arg=$TARGET"

cargo build --release --target "$RUST_TARGET" 2>&1 | tail -20

cp "target/${RUST_TARGET}/release/sofuu" "$OUT"
chmod +x "$OUT"

SIZE=$(du -sh "$OUT" 2>/dev/null | cut -f1)
echo ""
echo "✅ Built: $OUT ($SIZE)"
echo "   Target: $TARGET (static musl, runs on any Linux x86_64)"
echo "   (QTSQ-free — the codec checkout is macOS-only)"
echo ""
