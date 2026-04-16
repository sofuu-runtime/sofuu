#!/usr/bin/env bash
# scripts/cross/build_linux_x86_64.sh
#
# Cross-compile Sofuu for Linux x86_64 from macOS arm64 using Zig as CC.
# Zig bundles its own libc and linker — no sysroot or musl package needed.
#
# Usage:
#   bash scripts/cross/install_zig.sh     # once
#   bash scripts/cross/build_linux_x86_64.sh
#
# Output: dist/sofuu-linux-x86_64  (statically linked, ~1.1MB)
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
OUT="$DIST/sofuu-linux-x86_64"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Cross-compiling Sofuu → Linux x86_64 (musl static)        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$DIST"
cd "$REPO_ROOT"

# ── Build libuv for the target ──────────────────────────────────
echo "→ Building libuv for ${TARGET}..."
UV_DIR="deps/libuv"
UV_BUILD="$UV_DIR/build-linux-x86_64"

# Use zig cc as CMake toolchain via environment variables
export CC="$ZIG cc -target $TARGET"
export CXX="$ZIG c++ -target $TARGET"
export AR="$ZIG ar"
export RANLIB="$ZIG ranlib"

/opt/homebrew/bin/cmake -S "$UV_DIR" -B "$UV_BUILD" \
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

/opt/homebrew/bin/cmake --build "$UV_BUILD" --target uv_a -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) 2>&1 | tail -5

UV_LIB="$UV_BUILD/libuv.a"
echo "✓ libuv built: $UV_LIB"

# ── Build libcurl for the target ────────────────────────────────
echo "→ Building static libcurl for ${TARGET}..."
bash "$REPO_ROOT/scripts/cross/build_libcurl_static.sh" "$TARGET"
CURL_INSTALL="$DIST/libcurl-$TARGET"
CURL_LIB="$CURL_INSTALL/lib/libcurl.a"
CURL_INC="$CURL_INSTALL/include"

# ── Collect sources ─────────────────────────────────────────────
QJS_DIR="deps/quickjs"
SRC="src"

QJS_SRCS=(
    "$QJS_DIR/quickjs.c"
    "$QJS_DIR/libregexp.c"
    "$QJS_DIR/libunicode.c"
    "$QJS_DIR/cutils.c"
    "$QJS_DIR/libbf.c"
    "$QJS_DIR/quickjs-libc.c"
)

SOFUU_SRCS=(
    "$SRC/main.c"
    "$SRC/sofuu.c"
    "$SRC/engine/engine.c"
    "$SRC/modules/mod_console.c"
    "$SRC/modules/mod_process.c"
    "$SRC/modules/mod_ai.c"
    "$SRC/io/promises.c"
    "$SRC/io/loop.c"
    "$SRC/io/timer.c"
    "$SRC/io/fs.c"
    "$SRC/io/subprocess.c"
    "$SRC/http/client.c"
    "$SRC/http/server.c"
    "$SRC/http/sse.c"
    "$SRC/mcp/mcp.c"
    "$SRC/ts/stripper.c"
    "$SRC/npm/resolver.c"
    "$SRC/npm/cjs.c"
    "$SRC/repl/repl.c"
    "$SRC/bundler/bundler.c"
    "$SRC/simd/avx.c"        # x86_64 — AVX2
    "deps/http-parser/http_parser.c"
)

QJS_VERSION=$(cat "$QJS_DIR/VERSION" 2>/dev/null || echo "2024-01-13")

CFLAGS=(
    -O2
    -target "$TARGET"
    -I"$QJS_DIR"
    -I"deps/libuv/include"
    -I"$SRC"
    -I"$SRC/engine"
    -I"$SRC/modules"
    -I"$SRC/io"
    -I"$SRC/http"
    -I"$SRC/mcp"
    -I"$SRC/ts"
    -I"$SRC/simd"
    -I"$SRC/npm"
    -I"$SRC/repl"
    -I"$SRC/bundler"
    -I"$CURL_INC"
    -Ideps/http-parser
    -D_GNU_SOURCE
    -DCONFIG_VERSION="\"$QJS_VERSION\""
    -mavx2 -mfma
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-cast-function-type
)

LDFLAGS=(
    -target "$TARGET"
    -static
    "$UV_LIB"
    "$CURL_LIB"
    -lm
    -lpthread
    -ldl
)

echo ""
echo "→ Compiling QJS sources (${#QJS_SRCS[@]} files) with -O2..."
QJS_OBJS=()
for f in "${QJS_SRCS[@]}"; do
    obj="/tmp/sofuu_cross_$(basename ${f%.c}).o"
    "$ZIG" cc "${CFLAGS[@]}" -O2 -c "$f" -o "$obj"
    QJS_OBJS+=("$obj")
done

echo "→ Compiling Sofuu sources (${#SOFUU_SRCS[@]} files)..."
SOFUU_OBJS=()
for f in "${SOFUU_SRCS[@]}"; do
    obj="/tmp/sofuu_cross_$(echo "$f" | tr '/' '_').o"
    "$ZIG" cc "${CFLAGS[@]}" -c "$f" -o "$obj"
    SOFUU_OBJS+=("$obj")
done

echo "→ Linking..."
"$ZIG" cc "${CFLAGS[@]}" -o "$OUT" \
    "${QJS_OBJS[@]}" "${SOFUU_OBJS[@]}" \
    "${LDFLAGS[@]}" 2>&1

SIZE=$(du -sh "$OUT" 2>/dev/null | cut -f1)
echo ""
echo "✅ Built: $OUT ($SIZE)"
echo "   Target: $TARGET (static musl, runs on any Linux x86_64)"
echo ""
