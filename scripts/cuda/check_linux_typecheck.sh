#!/usr/bin/env bash
# scripts/cuda/check_linux_typecheck.sh — typecheck the Linux-only CUDA model
# from a macOS host, with no GPU and no CUDA toolkit.
#
# Why this exists: `domains/infer/cuda_model.rs` is gated on
# `target_os = "linux"`, so a native macOS `cargo build` never compiles it. A
# missing struct field or a changed kernel signature there is invisible until
# the T4 build, which is a slow round trip through Kaggle. This cross-checks
# `sofuu-core` for aarch64-unknown-linux-gnu using zig as the C/C++ cross
# compiler (it ships its own libc headers, so no Linux sysroot is needed).
#
# It proves types and field lists, not behaviour: kernels still need
# `make cuda-check` for algebra and the T4 for launches.
#
# Usage: bash scripts/cuda/check_linux_typecheck.sh
# Requires: rustup target aarch64-unknown-linux-gnu, zig on PATH.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

command -v zig >/dev/null || {
    echo "SKIP: zig is not on PATH (brew install zig)" >&2
    exit 0
}
rustc --print target-list >/dev/null
if ! rustc --print sysroot >/dev/null; then
    echo "ERROR: rustc sysroot is unreadable" >&2
    exit 1
fi

TARGET=aarch64-unknown-linux-gnu
if ! rustup target list --installed 2>/dev/null | grep -qx "$TARGET"; then
    echo "SKIP: $TARGET std is not installed (rustup target add $TARGET)" >&2
    exit 0
fi

# cc-rs appends its own --target=<rust triple>, which zig cannot parse (and zig
# spells the triple for its own -target differently: aarch64-linux-gnu), and the
# SIMD sources pass -march=armv8.2-a+..., which zig spells -mcpu=. ampere1 has
# dotprod, so the ARM kernel paths compile instead of erroring on missing
# intrinsics.
WRAP="$(mktemp -d)"
trap 'rm -rf "$WRAP"' EXIT
for driver in cc c++; do
    cat > "$WRAP/zig$driver.sh" <<EOF
#!/bin/bash
args=()
for a in "\$@"; do
  case "\$a" in
    --target=*|-march=*) continue ;;
  esac
  args+=("\$a")
done
exec zig $driver -target aarch64-linux-gnu -mcpu=ampere1 "\${args[@]}"
EOF
    chmod +x "$WRAP/zig$driver.sh"
done

# build.rs treats a SOFUU_QTSQ_DIR whose basename is "no-qtsq" as the
# codec-less shape (same marker scripts/kaggle/build_gpu_from_clone.sh uses).
MARKER="${TMPDIR:-/tmp}/no-qtsq"
mkdir -p "$MARKER"

SOFUU_QTSQ_DIR="$MARKER" \
CC_aarch64_unknown_linux_gnu="$WRAP/zigcc.sh" \
CXX_aarch64_unknown_linux_gnu="$WRAP/zigc++.sh" \
AR_aarch64_unknown_linux_gnu="zig ar" \
    cargo check -p sofuu-core --lib --target "$TARGET" 2>&1 | tail -3

echo "OK: sofuu-core typechecks for $TARGET (cuda_model.rs included)"
