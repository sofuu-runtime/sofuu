#!/usr/bin/env bash
# scripts/mobile/typecheck_ios.sh — Swift type-check the iOS app WITHOUT the
# engine framework.
#
# Why this exists: the app `import Sofuu`, which only resolves once
# dist/libsofuu.xcframework has been built. That makes a plain
# `xcodebuild` a chicken-and-egg problem — you cannot check a single line of
# Swift until a multi-minute Rust cross-build has finished.
#
# This script breaks the cycle. sofuu-ios/.devstub/ carries a header stub
# with the same declarations as dist/sofuu_embed.h and a modulemap that
# declares the `Sofuu` module, so `swiftc -typecheck` can resolve the import
# and check every file in a couple of seconds. It catches every type error,
# missing argument, bad ViewBuilder construct and signature drift long
# before a device build would.
#
# It does NOT link, and it does NOT compile the .metal shader — for those
# you need the real framework and `make ios-app`.
#
# Usage: bash scripts/mobile/typecheck_ios.sh [device|sim]   (default: sim)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP_DIR="$REPO_ROOT/sofuu-ios"
WHICH="${1:-sim}"

case "$WHICH" in
    sim)    SDK_NAME="iphonesimulator"; TARGET="arm64-apple-ios16.0-simulator" ;;
    device) SDK_NAME="iphoneos";        TARGET="arm64-apple-ios16.0" ;;
    *) echo "Usage: $0 [device|sim]"; exit 1 ;;
esac

if [ ! -f "$APP_DIR/.devstub/module.modulemap" ]; then
    echo "  \033[31m✗ $APP_DIR/.devstub/module.modulemap is missing\033[0m"
    echo "    It is the stub module that lets Swift resolve 'import Sofuu'."
    exit 1
fi

SDK="$(xcrun --sdk "$SDK_NAME" --show-sdk-path)"

echo "  \033[36mType-checking $(find "$APP_DIR/SofuuApp" -name '*.swift' | wc -l | tr -d ' ') Swift files against $SDK_NAME\033[0m"

cd "$APP_DIR"
# shellcheck disable=SC2046
xcrun swiftc -typecheck \
    -parse-as-library \
    -sdk "$SDK" \
    -target "$TARGET" \
    -I .devstub \
    -module-name SofuuApp \
    $(find SofuuApp -name '*.swift' | sort)

echo "  \033[32m✓\033[0m Swift type-check clean"

# The shader is independent of the Swift, so check it here too — a shader
# that fails to compile ships as a silently missing default.metallib, which
# is exactly the "the background vanished" failure mode.
echo "  \033[36mCompiling SofuuCloud.metal against $SDK_NAME\033[0m"
xcrun -sdk "$SDK_NAME" metal -c SofuuApp/Shaders/SofuuCloud.metal -o /dev/null -std=metal3.0
echo "  \033[32m✓\033[0m Metal shader compiles"

# The engine's kernels too (PLAN-EMBED-SDK E3). This is the fail-fast for
# the GPU path: if kernels.metal does not compile for the iOS SDK, the
# shipped default.metallib would be missing (or, worse, half-built) and the
# engine would be silently CPU-only on device. -std=metal3.1 because the
# bf16 kernels use `bfloat`, which MSL 3.0 rejects for every SDK.
echo "  \033[36mCompiling src/metal/kernels.metal against $SDK_NAME\033[0m"
xcrun -sdk "$SDK_NAME" metal -c "$REPO_ROOT/src/metal/kernels.metal" -o /dev/null -std=metal3.1
echo "  \033[32m✓\033[0m Metal engine kernels compile"
