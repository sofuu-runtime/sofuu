#!/usr/bin/env bash
# scripts/dist/macos-engine.sh — engine artifact behind the sofuu-macos app
#
# Produces (in sofuu-macos/engine/, gitignored):
#     sofuu-macos/engine/Sofuu.framework       macOS arm64 framework slice
#       ├── Sofuu            (QTSQ-linked cdylib, install name @rpath/…)
#       ├── Headers/sofuu_embed.h
#       ├── Modules/module.modulemap   (`import Sofuu`, autolinks Metal)
#       └── Resources/default.metallib (GPU kernels fast path)
#
# Same framework layout scripts/dist/ios.sh ships in the xcframework, so the
# app consumes it identically (XcodeGen: - framework: …, embed: true). A
# bare .dylib does NOT work: XcodeGen links it as `-lsofuu -Fengine` and
# the link dies with "library 'sofuu' not found".
#
# Unlike scripts/dist/macos.sh (CI, QTSQ-free), this links QTSQ when the
# checkout is present so sessions/brain persist exactly like the Tauri app.
# The lookup mirrors sofuu-ffi/build.rs's probed fallback so an explicit env
# var is never required locally — but we EXPORT it explicitly, so the build
# log states which shape was produced instead of silently probing.
#
# Usage: bash scripts/dist/macos-engine.sh [arm64|x86_64]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

APP_ENGINE_DIR="$REPO_ROOT/sofuu-macos/engine"
FRAMEWORK="$APP_ENGINE_DIR/Sofuu.framework"
ARCH_ARG="${1:-$(uname -m)}"
if [ "$ARCH_ARG" = "aarch64" ]; then ARCH_ARG="arm64"; fi

case "$ARCH_ARG" in
    arm64)  RUST_TARGET="aarch64-apple-darwin" ;;
    x86_64) RUST_TARGET="x86_64-apple-darwin" ;;
    *) echo "Usage: $0 [arm64|x86_64]"; exit 1 ;;
esac

if ! rustup target list --installed 2>/dev/null | grep -q "^${RUST_TARGET}$"; then
    rustup target add "$RUST_TARGET"
fi

# ── QTSQ: same probe build.rs does, but stated out loud ────────────
if [ -z "${SOFUU_QTSQ_DIR:-}" ] && [ -f "$HOME/projects/black-hole-disk/libqtsq.a" ]; then
    export SOFUU_QTSQ_DIR="$HOME/projects/black-hole-disk"
fi
if [ -z "${SOFUU_QTSQ_DIR:-}" ]; then
    mkdir -p "$APP_ENGINE_DIR/no-qtsq"
    export SOFUU_QTSQ_DIR="$APP_ENGINE_DIR/no-qtsq"
    QTSQ_SHAPE="QTSQ-free (fail-closed: chats/brain do NOT persist) — set SOFUU_QTSQ_DIR=<checkout> to link it"
else
    if [ ! -f "$SOFUU_QTSQ_DIR/libqtsq.a" ]; then
        echo "  \033[31m✗ SOFUU_QTSQ_DIR=$SOFUU_QTSQ_DIR has no libqtsq.a\033[0m"
        exit 1
    fi
    QTSQ_SHAPE="QTSQ-linked (encrypted sessions/brain persist)"
fi

# ── libuv for the host arch (PIC) ──────────────────────────────────
# Default generator (Unix Makefiles) — no Ninja dependency, unlike the CI
# release script which brew-installs one. ABSOLUTE path required: cargo runs
# build scripts with the package root (crates/sofuu-ffi) as cwd, so a
# relative SOFUU_UV_DIR resolves to the wrong tree (iOS README pitfall #1).
UV_BUILD="$REPO_ROOT/deps/libuv/build-macos-$ARCH_ARG"
cmake -S "$REPO_ROOT/deps/libuv" -B "$UV_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_OSX_ARCHITECTURES="$ARCH_ARG"
cmake --build "$UV_BUILD" --target uv_a
export SOFUU_UV_DIR="$UV_BUILD"

# ── cdylib (system libcurl; no static fold needed on macOS) ────────
echo "  \033[36mENGINE\033[0m cargo rustc -p sofuu-capi --target $RUST_TARGET ($QTSQ_SHAPE)"
cargo rustc --release -p sofuu-capi --target "$RUST_TARGET" --crate-type cdylib

LIB_DIR="target/$RUST_TARGET/release"
DYLIB="$LIB_DIR/libsofuu_capi.dylib"
[ -f "$DYLIB" ] || DYLIB="$LIB_DIR/libsofuu.dylib"

# ── Assemble Sofuu.framework (ios.sh's layout, desktop slice) ──────
rm -rf "$FRAMEWORK"
FWROOT="$FRAMEWORK/Versions/A"
mkdir -p "$FWROOT/Headers" "$FWROOT/Modules" "$FWROOT/Resources"
cp "$DYLIB" "$FWROOT/Sofuu"
cp include/sofuu_embed.h "$FWROOT/Headers/sofuu_embed.h"
cp "$APP_ENGINE_DIR/sofuu_embed.h" "$FWROOT/Headers/sofuu_embed.h" 2>/dev/null || true
cat > "$FWROOT/Modules/module.modulemap" <<'EOF'
framework module Sofuu {
    header "sofuu_embed.h"
    export *
    link framework "Metal"
}
EOF

# GPU kernels (fast path; the engine falls back to runtime compile).
TMP="$(mktemp -d)"
xcrun -sdk macosx metal -c "$REPO_ROOT/src/metal/kernels.metal" \
    -o "$TMP/kernels.air" -std=metal3.0
xcrun -sdk macosx metallib "$TMP/kernels.air" -o "$FWROOT/Resources/default.metallib"
rm -rf "$TMP"
for fn in gemv_bf16 gemv_f16 gemv_f32 gemv_mlx4_affine_f16 \
          gemv_mlx4_affine_bf16 rmsnorm swiglu conv1d_3; do
    if ! strings "$FWROOT/Resources/default.metallib" | grep -qx "$fn"; then
        echo "  \033[31m✗ default.metallib is missing kernel $fn\033[0m"
        exit 1
    fi
done

cat > "$FWROOT/Resources/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>com.sofuu.engine</string>
    <key>CFBundleExecutable</key><string>Sofuu</string>
    <key>CFBundleName</key><string>Sofuu</string>
    <key>CFBundlePackageType</key><string>FMWK</string>
    <key>CFBundleShortVersionString</key><string>0.2.0</string>
    <key>CFBundleVersion</key><string>0.2.0</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict>
</plist>
EOF

# Versioned symlinks (the framework layout dyld expects).
ln -sfn A "$FRAMEWORK/Versions/Current"
ln -sfn Versions/Current/Sofuu "$FRAMEWORK/Sofuu"
ln -sfn Versions/Current/Headers "$FRAMEWORK/Headers"
ln -sfn Versions/Current/Modules "$FRAMEWORK/Modules"
ln -sfn Versions/Current/Resources "$FRAMEWORK/Resources"

# Relocatable install name + ad-hoc sign (any modification invalidates the
# old signature).
install_name_tool -id "@rpath/Sofuu.framework/Versions/A/Sofuu" "$FWROOT/Sofuu"
codesign --force --sign - "$FWROOT/Sofuu"

{
    echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "arch: $ARCH_ARG ($(uname -m))"
    echo "target: $RUST_TARGET"
    echo "qtsq: $QTSQ_SHAPE"
    echo "uv: $UV_BUILD"
} > "$APP_ENGINE_DIR/STAMP"

rm -rf "$APP_ENGINE_DIR/no-qtsq"
echo "  \033[32m✓\033[0m sofuu-macos/engine/Sofuu.framework ($(du -sh "$FWROOT/Sofuu" | cut -f1) binary) — $QTSQ_SHAPE"
