#!/usr/bin/env bash
# scripts/cross/build_linux_arm64.sh
# Cross-compile Sofuu for Linux arm64 (aarch64) from macOS arm64 using Zig.
# Output: dist/sofuu-linux-arm64
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIG="$REPO_ROOT/tools/zig/zig"
DIST="$REPO_ROOT/dist"

if [ ! -x "$ZIG" ]; then
    echo "Zig not found. Run: bash scripts/cross/install_zig.sh"
    exit 1
fi

TARGET="aarch64-linux-musl"
OUT="$DIST/sofuu-linux-arm64"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Cross-compiling Sofuu → Linux arm64 (musl static)         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$DIST"
cd "$REPO_ROOT"

# ── Build libuv for aarch64-linux-musl ─────────────────────────
echo "→ Building libuv for ${TARGET}..."
UV_DIR="deps/libuv"
UV_BUILD="$UV_DIR/build-linux-arm64"

cmake -S "$UV_DIR" -B "$UV_BUILD" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_C_COMPILER="$ZIG" \
    -DCMAKE_C_COMPILER_ARG1="cc -target $TARGET" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
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
CURL_LIB="$CURL_INSTALL/lib/libcurl.a"
CURL_INC="$CURL_INSTALL/include"

QJS_DIR="deps/quickjs"
SRC="src"
QJS_VERSION=$(cat "$QJS_DIR/VERSION" 2>/dev/null || echo "2024-01-13")

QJS_SRCS=(
    "$QJS_DIR/quickjs.c"
    "$QJS_DIR/libregexp.c"
    "$QJS_DIR/libunicode.c"
    "$QJS_DIR/cutils.c"
    "$QJS_DIR/libbf.c"
    "$QJS_DIR/quickjs-libc.c"
)

SOFUU_SRCS=(
    "$SRC/main.c" "$SRC/sofuu.c" "$SRC/engine/engine.c"
    "$SRC/modules/mod_console.c" "$SRC/modules/mod_process.c" "$SRC/modules/mod_ai.c"
    "$SRC/io/promises.c" "$SRC/io/loop.c" "$SRC/io/timer.c"
    "$SRC/io/fs.c" "$SRC/io/subprocess.c"
    "$SRC/http/client.c" "$SRC/http/server.c" "$SRC/http/sse.c"
    "$SRC/mcp/mcp.c"
    "$SRC/ts/stripper.c"
    "$SRC/npm/resolver.c" "$SRC/npm/cjs.c"
    "$SRC/repl/repl.c"
    "$SRC/simd/neon.c"          # arm64 — NEON
    "deps/http-parser/http_parser.c"
)

CFLAGS=(
    -O2 -target "$TARGET"
    -I"$QJS_DIR" -I"deps/libuv/include"
    -I"$SRC" -I"$SRC/engine" -I"$SRC/modules" -I"$SRC/io"
    -I"$SRC/http" -I"$SRC/mcp" -I"$SRC/ts"
    -I"$SRC/simd" -I"$SRC/npm" -I"$SRC/repl"
    -I"$SRC/bundler"
    -I"$CURL_INC" -Ideps/http-parser
    -D_GNU_SOURCE -DCONFIG_VERSION="\"$QJS_VERSION\""
    -Wno-unused-parameter -Wno-sign-compare -Wno-cast-function-type
)

echo "→ Compiling..."
ALL_OBJS=()
for f in "${QJS_SRCS[@]}" "${SOFUU_SRCS[@]}"; do
    obj="/tmp/sofa64_$(echo "$f" | tr '/' '_').o"
    "$ZIG" cc "${CFLAGS[@]}" -c "$f" -o "$obj"
    ALL_OBJS+=("$obj")
done

echo "→ Linking..."
"$ZIG" cc "${CFLAGS[@]}" -o "$OUT" \
    "${ALL_OBJS[@]}" \
    -target "$TARGET" -static \
    "$UV_LIB" "$CURL_LIB" -lm -lpthread -ldl

SIZE=$(du -sh "$OUT" 2>/dev/null | cut -f1)
echo ""
echo "✅ Built: $OUT ($SIZE)"
echo "   Target: Linux arm64 (musl static)"
echo ""
