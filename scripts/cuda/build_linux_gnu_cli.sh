#!/usr/bin/env bash
# scripts/cuda/build_linux_gnu_cli.sh — build the CUDA-capable Linux CLI.
#
# WHY THIS EXISTS: the release CLI is static musl. musl's dlopen is a stub
# ("Dynamic loading not supported"), so a static musl binary cannot load
# libcuda.so.1 and the CUDA decoder is not compiled into it. The GPU backend
# therefore lives only in this glibc (target_env = "gnu") build, where the
# system loader can open the driver.
#
# This is the CLI counterpart of scripts/cross/build_linux_capi_so.sh: same
# zig/glibc toolchain, same gnu QTSQ/libuv/libcurl archives, one binary.
#
# Usage: bash scripts/cuda/build_linux_gnu_cli.sh
# Output: dist/sofuu-linux-x86_64-gnu

set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIG="$REPO_ROOT/tools/zig/zig"
DIST="$REPO_ROOT/dist"
RUST_TARGET="x86_64-unknown-linux-gnu"
ZIG_TARGET="x86_64-linux-gnu"
OUT="$DIST/kaggle/sofuu-linux-x86_64-gnu"

[ -x "$ZIG" ] || { echo "Zig not found at $ZIG — run scripts/cross/install_zig.sh" >&2; exit 1; }
cd "$REPO_ROOT"
mkdir -p "$DIST/kaggle"

if ! rustup target list --installed 2>/dev/null | grep -q "^${RUST_TARGET}$"; then
    rustup target add "$RUST_TARGET"
fi

# libuv: the glibc/PIC archive the C-ABI build already stages.
UV_BUILD="$REPO_ROOT/deps/libuv/build-linux-gnu-x86_64"
if [ ! -f "$UV_BUILD/libuv.a" ]; then
    echo "missing $UV_BUILD/libuv.a — run scripts/cross/build_linux_capi_so.sh once" >&2
    exit 1
fi

bash "$REPO_ROOT/scripts/cross/build_qtsq_cross.sh" >/dev/null
QTSQ_DIR="$DIST/qtsq-linux-x86_64-gnu"
[ -f "$QTSQ_DIR/libqtsq.a" ] || { echo "missing glibc QTSQ ($QTSQ_DIR)" >&2; exit 1; }
bash "$REPO_ROOT/scripts/cross/build_libcurl_static.sh" "$ZIG_TARGET" >/dev/null
CURL_LIB="$DIST/libcurl-$ZIG_TARGET/lib"

export CC="$REPO_ROOT/scripts/cross/zig-cc.sh"
export CXX="$REPO_ROOT/scripts/cross/zig-cxx.sh --cxx"
export AR="$ZIG ar"
export RANLIB="$ZIG ranlib"
export SOFUU_UV_DIR="$UV_BUILD"
export SOFUU_CURL_STATIC_DIR="$CURL_LIB"
export SOFUU_QTSQ_DIR="$QTSQ_DIR"
export SOFUU_ZLIB_DIR="$QTSQ_DIR"

# The glibc binary is not size-gated; it is a test/GPU artifact, not the
# 5,767,168 B release. Keep the build-host path remaps so a staged binary does
# not leak the producing machine.
if [ -z "${SOFUU_BUILD:-}" ]; then
    _rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
    if ! git -C "$REPO_ROOT" diff --quiet --untracked-files=no 2>/dev/null; then
        _rev="${_rev}-dirty"
    fi
    export SOFUU_BUILD="$_rev"
fi
export RUSTFLAGS="$(bash "$REPO_ROOT/scripts/cross/remap_flags.sh") \
 -C linker=$REPO_ROOT/scripts/cross/zig-ld.sh \
 -C link-arg=-target -C link-arg=$ZIG_TARGET"

echo "→ cargo build --release --target $RUST_TARGET -p sofuu-core (CUDA decoder included)"
cargo build --release --target "$RUST_TARGET" -p sofuu-core

cp "$REPO_ROOT/target/$RUST_TARGET/release/sofuu" "$OUT"
chmod +x "$OUT"

READELF="${READELF:-}"
if [ -z "$READELF" ] && command -v llvm-readelf >/dev/null 2>&1; then
    READELF=llvm-readelf
fi
echo ""
echo "✅ Built: $OUT ($(wc -c < "$OUT" | tr -d ' ') B)"
file "$OUT"
if [ -n "$READELF" ]; then
    "$READELF" -l "$OUT" | grep -i interpreter || true
    "$READELF" -d "$OUT" | grep NEEDED || true
fi
echo "   This binary needs glibc (Kaggle Ubuntu is fine) and loads libcuda at runtime."
echo "   Package for Kaggle: bash scripts/cuda/make_gpu_cli_kit.sh"
