#!/usr/bin/env bash
# scripts/dist/ios.sh — H4 platform pack: iOS (arm64 device + simulator)
#
# Produces: dist/libsofuu-ios-arm64.a (device)
#           dist/libsofuu-ios-sim.a   (simulator, arm64)
#           dist/libsofuu.xcframework (combined device + sim; contains Sofuu.framework)
#           dist/sofuu_embed.h
#
# Usage: bash scripts/dist/ios.sh [target]
#   target: device | sim | deps | xcframework | all (default: all)
#
#   deps  builds ONLY the third-party stack (OpenSSL + libcurl + libuv) for
#         both slices. Useful once, up front, so a later engine rebuild is
#         a plain cargo invocation.
#
# Prerequisites: Xcode, Rust toolchain with iOS targets.
#   rustup target add aarch64-apple-ios aarch64-apple-ios-sim
#
# QuickJS is a pure interpreter (no JIT) → iOS App Store executable-memory
# policy is satisfied by design. This is the key iOS architectural advantage.
#
# ── Why this script builds dependencies and android.sh does not ──────
# Android links a `libsofuu.so` against the NDK's sysroot, which already
# carries libz, liblog and friends, so the Android driver only has to add
# OpenSSL + curl + QTSQ. iOS ships NO system libcurl and no system OpenSSL,
# and `crates/sofuu-ffi/build.rs` links the *system* curl (`-lcurl`) when
# SOFUU_CURL_STATIC_DIR is unset — a link that dies on iOS with
# `ld: library 'curl' not found`. So the TLS stack has to be cross-built and
# merged into the engine archive first. That is what `ensure_deps` does, and
# it is why a cold `make ios-engine` takes a few minutes instead of seconds.
#
# ── Why QTSQ is deliberately absent ─────────────────────────────────
# The proprietary codec checkout is macOS-only, so iOS packs ship pure MIT.
# This script points SOFUU_QTSQ_DIR at the repo's empty `dist/no-qtsq` marker
# directory — deliberately NOT `unset`, because build.rs has a hardcoded
# fallback that fires when the variable is missing and would drag a macOS
# QTSQ build into the iOS archive. See the long note in build_target().
# Consequence: has_qtsq = false, so the `-lz` branch in build.rs is never
# entered and the archive needs no zlib.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$REPO_ROOT/dist"
TARGET_ARG="${1:-all}"

# Toolchain plumbing (SDK, CC/AR, per-slice identity, CMake flags).
source "$REPO_ROOT/scripts/mobile/ios_env.sh"

CURL_DIR_FOR() { echo "$DIST/libcurl-ios-$1"; }

# ── Third-party stack ────────────────────────────────────────────────
# Idempotent: each builder short-circuits when its archive is already on
# disk, so this is cheap on every run after the first. Set
# SOFUU_IOS_SKIP_DEPS=1 to assert the deps are present and fail loudly if not.
ensure_deps() {
    local slice="$1"
    local curl_dir; curl_dir="$(CURL_DIR_FOR "$slice")"

    if [ -f "$curl_dir/lib/libcurl.a" ]; then
        echo "  ✓ libcurl-ios-$slice already built"
        return 0
    fi

    if [ -n "${SOFUU_IOS_SKIP_DEPS:-}" ]; then
        echo ""
        echo "  \033[31m✗ SOFUU_IOS_SKIP_DEPS is set but $curl_dir/lib/libcurl.a is missing\033[0m"
        echo "    Unset it, or run: bash scripts/dist/ios.sh deps"
        exit 1
    fi

    echo ""
    echo "  \033[36mBuilding the iOS TLS stack for $slice (one-time, a few minutes)\033[0m"
    # curl needs OpenSSL first, and the curl builder asserts that — run them
    # in order rather than letting the assertion do the explaining.
    bash "$REPO_ROOT/scripts/mobile/build_openssl_ios.sh" "$slice"
    bash "$REPO_ROOT/scripts/mobile/build_libcurl_ios.sh" "$slice"

    if [ ! -f "$curl_dir/lib/libcurl.a" ]; then
        echo "  \033[31m✗ libcurl for iOS $slice still missing after the build\033[0m"
        exit 1
    fi
}

build_target() {
    local NAME="$1"
    local RUST_TARGET="$2"
    local OUT_NAME="$3"

    # Resolve the slice: IOS_SDK_NAME, IOS_SYSROOT, IOS_CC, IOS_CMAKE_FLAGS,
    # the CARGO_TARGET_*_LINKER export, …
    ios_slice_env "$NAME"

    echo ""
    echo "  \033[36mBuilding libsofuu for iOS $NAME ($RUST_TARGET)\033[0m"

    cd "$REPO_ROOT"

    if ! rustup target list --installed 2>/dev/null | grep -q "^${RUST_TARGET}$"; then
        echo "  Installing rustup target $RUST_TARGET..."
        rustup target add "$RUST_TARGET"
    fi

    ensure_deps "$NAME"

    # ── libuv ────────────────────────────────────────────────────────
    local UV_DIR="deps/libuv"
    local UV_BUILD="deps/libuv/build-ios-$NAME"

    # The sysroot alone is NOT enough: without CMAKE_SYSTEM_NAME=iOS,
    # CMake configures for Darwin/macOS and emits -mmacosx-version-min
    # flags, producing a libuv that cannot link into an iOS binary.
    # (Verified by inspecting LC_BUILD_VERSION: platform 2 = iOS, not 1.)
    # shellcheck disable=SC2086
    cmake -S "$UV_DIR" -B "$UV_BUILD" $IOS_CMAKE_GEN \
        $IOS_CMAKE_FLAGS \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF 2>&1 | tail -3
    cmake --build "$UV_BUILD" --target uv_a 2>&1 | tail -3

    if [ ! -f "$UV_BUILD/libuv.a" ] && [ ! -f "$UV_BUILD/libuv_a.a" ]; then
        echo "  \033[31m✗ libuv archive not found in $UV_BUILD\033[0m"
        exit 1
    fi

    # libuv is the archive that actually burned us: CMAKE_OSX_SYSROOT alone
    # was not enough, CMake quietly produced a macOS libuv, and the failure
    # only surfaced much later at link time. Assert the platform here.
    local UV_LIB
    UV_LIB="$UV_BUILD/libuv.a"
    [ -f "$UV_LIB" ] || UV_LIB="$UV_BUILD/libuv_a.a"
    echo "  libuv $NAME:"
    ios_assert_platform "$UV_LIB" "libuv.a ($NAME)"
    ios_assert_no_macos "$UV_LIB" "libuv.a ($NAME)"

    # ABSOLUTE, deliberately. cargo runs build scripts with the cwd set to
    # the crate directory (crates/sofuu-ffi), so a relative SOFUU_UV_DIR
    # resolves against the wrong root and build.rs panics with a misleading
    # "libuv not built" even though the archive is sitting right there.
    export SOFUU_UV_DIR="$REPO_ROOT/$UV_BUILD"

    # ── libcurl + OpenSSL ────────────────────────────────────────────
    # The `lib` subdirectory, matching scripts/mobile/build_engine_android.sh
    # and build.rs's `link-search=native=<dir>` + `link-lib=static=curl`.
    # The merged archive inside already carries libssl + libcrypto, so a
    # single `-lcurl` resolves the whole TLS stack.
    export SOFUU_CURL_STATIC_DIR="$(CURL_DIR_FOR "$NAME")/lib"

    # ── QTSQ: force the codec-less shape ─────────────────────────────
    # The proprietary codec checkout is macOS-only, so iOS packs ship pure
    # MIT. This uses the repo-wide convention — an empty `dist/no-qtsq`
    # directory — and NOT `unset SOFUU_QTSQ_DIR`.
    #
    # That distinction is load-bearing. Unsetting the variable looks like the
    # stronger statement, but crates/sofuu-ffi/build.rs has a fallback that
    # fires precisely when the variable is *absent*:
    #
    #     let qtsq_dir = env::var("SOFUU_QTSQ_DIR").map(PathBuf::from).ok()
    #         .or_else(|| {
    #             let candidate = PathBuf::from(
    #                 "/Users/priyanshuboruah/projects/black-hole-disk");
    #             candidate.join("libqtsq.a").exists().then_some(candidate)
    #         });
    #
    # On this machine that path exists and holds an **arm64 macOS** build, so
    # unsetting silently pulled 102 macOS `qtsq_*.o` objects into the iOS
    # archive — the archive grew 10 MB → 20 MB and would have failed the app
    # link. Pointing at an empty directory instead makes qtsq_dir Some but
    # qtsq_lib non-existent, so has_qtsq = false and the fallback never runs.
    # (That fallback contradicts the comment two lines above it in build.rs,
    # which says there is deliberately no developer-machine absolute path.
    # It is an uncommitted local change from another workstream; left alone
    # here, but worth reverting.)
    #
    # Bonus: with has_qtsq = false the `-lz` branch in build.rs is never
    # entered, so the archive needs no zlib — which is why the iOS libcurl
    # is built with CMAKE_DISABLE_FIND_PACKAGE_ZLIB.
    mkdir -p "$DIST/no-qtsq"
    export SOFUU_QTSQ_DIR="$DIST/no-qtsq"
    unset SOFUU_QTSQ_LIB
    unset SOFUU_ZLIB_DIR

    # ── Rust ─────────────────────────────────────────────────────────
    # `--crate-type staticlib` deliberately, NOT a plain `cargo build`.
    # sofuu-capi declares ["staticlib", "cdylib", "rlib"], so a plain build
    # also tries to emit a .dylib — which is pointless on iOS (nothing may
    # dlopen it) and is the step that drags in -lcurl and fails the whole
    # build even when the archive we actually want was produced fine.
    # An xcframework takes the static archive and nothing else.
    cargo rustc --release -p sofuu-capi --target "$RUST_TARGET" --crate-type staticlib

    local RUST_A=""
    if [ -f "target/${RUST_TARGET}/release/libsofuu_capi.a" ]; then
        RUST_A="target/${RUST_TARGET}/release/libsofuu_capi.a"
    elif [ -f "target/${RUST_TARGET}/release/libsofuu.a" ]; then
        RUST_A="target/${RUST_TARGET}/release/libsofuu.a"
    else
        echo "  \033[31m✗ no static archive produced for $RUST_TARGET\033[0m"
        exit 1
    fi

    # ── Make the archive self-contained ──────────────────────────────
    # A staticlib does NOT link. cargo archives the Rust objects plus the
    # ones the `cc` crate compiled (QuickJS — which is why `JS_NewRuntime`
    # is *defined* in the archive) and then drops every
    # `cargo:rustc-link-lib` on the floor. The result leaves ~371 `_uv_*`
    # and ~24 `_curl_*` symbols UNDEFINED.
    #
    # Android never sees this: libsofuu.so is a *shared* object, so its
    # libuv/curl are resolved when the .so itself is linked. iOS ships a
    # static xcframework, so those archives have to travel inside it or the
    # app link dies in a wall of undefined symbols.
    #
    # `libtool -static` merges them (and rebuilds __.SYMDEF). This is the
    # same fold-the-dependency-in trick used one level down to put OpenSSL
    # inside libcurl.a — just applied to the final archive. The member-name
    # collisions libtool would otherwise choke on are already avoided:
    # libcurl.a's OpenSSL objects were prefixed ssl_/crypto_ by the curl
    # builder, and Rust's object names are hash-suffixed.
    mkdir -p "$DIST"
    local CURL_A
    CURL_A="$(CURL_DIR_FOR "$NAME")/lib/libcurl.a"
    echo "  merging engine + libuv + libcurl (+OpenSSL) → $OUT_NAME"
    rm -f "$DIST/$OUT_NAME"
    libtool -static -o "$DIST/$OUT_NAME" "$RUST_A" "$UV_LIB" "$CURL_A"

    # Prove the fold actually worked. NOTE: `nm -u` on the merged archive is
    # the WRONG check — it reports undefined references per object, so it
    # counts curl's own internal cross-references and can never reach zero
    # (it goes *up*, 371 → 684, purely because more objects are present).
    # The meaningful question is whether every uv/curl symbol the engine
    # needs is now DEFINED somewhere in the archive. The helpers return
    # global-defined vs undefined symbol name sets, and are pipefail-safe.
    local NEEDED DEFINED missing needed_count
    NEEDED="$(mktemp)"; DEFINED="$(mktemp)"
    ios_undefined_symbols "$RUST_A" | grep -E '^_(uv_|curl_)' > "$NEEDED" || true
    ios_defined_symbols "$DIST/$OUT_NAME" > "$DEFINED"
    missing="$(comm -23 "$NEEDED" "$DEFINED" | wc -l | tr -d ' ' || true)"
    needed_count="$(wc -l < "$NEEDED" | tr -d ' ')"
    if [ "$missing" != "0" ]; then
        echo "  \033[31m✗ archive is not self-contained: $missing of $needed_count uv/curl symbols unresolved\033[0m"
        comm -23 "$NEEDED" "$DEFINED" | head -10 | sed 's/^/      /'
        rm -f "$NEEDED" "$DEFINED"
        exit 1
    fi
    rm -f "$NEEDED" "$DEFINED"
    echo "  self-contained: all $needed_count _uv_*/_curl_* symbols resolved"

    # Prove the archive is iOS and not a stray macOS build — the failure mode
    # that cost an afternoon on libuv. Architecture alone proves nothing
    # (both are arm64), so check the LC_BUILD_VERSION platform too.
    local plat
    plat="$(lipo -info "$DIST/$OUT_NAME" 2>/dev/null | sed 's/.*architecture[s]*: //')"
    local SIZE
    SIZE="$(du -sh "$DIST/$OUT_NAME" 2>/dev/null | cut -f1)"
    echo "  \033[32m✓\033[0m $OUT_NAME ($SIZE, $plat)"
    ios_assert_platform "$DIST/$OUT_NAME" "$OUT_NAME"
    # The check that would have caught the macOS QTSQ leak: census every
    # member, not just the first one. A single-member probe reported
    # "platform: IOS ✓" while 102 macOS objects sat inside the archive.
    ios_assert_no_macos "$DIST/$OUT_NAME" "$OUT_NAME"

    # Size check. The 5 MB repo-wide shipping cap targets the *linked app
    # binary*; this archive is a different artifact entirely. It is
    # deliberately self-contained — QuickJS + libuv + libcurl with OpenSSL
    # merged in — so it lands near 20 MB before the app linker dead-strips
    # everything unreferenced. 32 MB is the ceiling; it is loose enough not
    # to trip on a healthy build and tight enough to catch the things that
    # actually go wrong (QTSQ leaking in, a debug profile, bundled debug
    # info). Override with SOFUU_IOS_MAX_A_SIZE.
    local MAX_A_SIZE="${SOFUU_IOS_MAX_A_SIZE:-33554432}"
    local A_SIZE
    A_SIZE=$(stat -f%z "$DIST/$OUT_NAME" 2>/dev/null || echo 0)
    if [ "$A_SIZE" -gt "$MAX_A_SIZE" ]; then
        echo "  \033[31m✗ Static lib exceeds cap ($A_SIZE > $MAX_A_SIZE bytes)\033[0m"
        echo "    Re-run with SOFUU_IOS_MAX_A_SIZE=$A_SIZE if that is expected,"
        echo "    or trim the archive before shipping."
        exit 1
    fi
}

# ── Framework assembly ───────────────────────────────────────────────
# ios_make_framework <archive> <outdir> → <outdir>/Sofuu.framework
#
# WHY A .framework AND NOT `-library … -headers …`
# ------------------------------------------------
# `xcodebuild -create-xcframework -library X.a -headers DIR` produces an
# xcframework that is fine for *linking* but useless for Swift `import`.
# Verified by building the app: Xcode copies the resolved slice's .a into
# Build/Products/Debug-iphonesimulator and passes `-F` there, so the linker
# finds the archive — but there is no `.framework` bundle anywhere, so
# Clang has nothing to resolve `import Sofuu` against, and every call dies
# with "cannot find 'sofuu_rt_call' in scope". Adding the Headers dir with
# `-I` does work (that is what .devstub/ exploits for `make ios-typecheck`),
# but it means shipping the modulemap twice and keeping the copies in sync.
# A real framework bundle is the mechanism Apple intends, and `-F` finds it
# with no extra build settings at all.
#
# The modulemap MUST say `framework module`, not `module`. Inside a bundle
# Clang requires the qualifier; with a plain `module Sofuu` it warns
# "skipping 'sofuu_embed.h' because module declaration of 'Sofuu' lacks the
# 'framework' qualifier" and the module ends up EMPTY — which looks like a
# missing header rather than a modulemap problem.
#
# The binary is named `Sofuu` (not libsofuu.a) because a framework's
# executable name, its bundle name and its module name have to agree for
# `-F` + `import Sofuu` to line up.
ios_make_framework() {
    local archive="$1" outdir="$2"
    local fw="$outdir/Sofuu.framework"
    rm -rf "$fw"
    mkdir -p "$fw/Headers" "$fw/Modules"
    cp "$archive" "$fw/Sofuu"
    cp "$REPO_ROOT/include/sofuu_embed.h" "$fw/Headers/"

    cat > "$fw/Modules/module.modulemap" << 'EOF'
framework module Sofuu {
    umbrella header "sofuu_embed.h"
    export *
    module * { export * }
}
EOF

    cat > "$fw/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleExecutable</key><string>Sofuu</string>
    <key>CFBundleIdentifier</key><string>com.sofuu.engine</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>Sofuu</string>
    <key>CFBundlePackageType</key><string>FMWK</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleVersion</key><string>1</string>
</dict>
</plist>
EOF

    if [ ! -f "$fw/Sofuu" ] || [ ! -f "$fw/Modules/module.modulemap" ]; then
        echo "  \033[31m✗ framework assembly incomplete at $fw\033[0m"
        exit 1
    fi
}

build_xcframework() {
    echo ""
    echo "  \033[36mCreating xcframework...\033[0m"

    local DEVICE_LIB="$DIST/libsofuu-ios-arm64.a"
    local SIM_LIB="$DIST/libsofuu-ios-sim.a"

    if [ ! -f "$DEVICE_LIB" ] || [ ! -f "$SIM_LIB" ]; then
        echo "  \033[33m⚠ Missing device or sim lib — building both first...\033[0m"
        build_target "device" "aarch64-apple-ios" "libsofuu-ios-arm64.a"
        build_target "sim" "aarch64-apple-ios-sim" "libsofuu-ios-sim.a"
    fi

    # Assemble a real .framework per slice, then combine. See ios_make_framework.
    #
    # The OUTER name stays `libsofuu.xcframework` on purpose: that is the
    # published artifact name — .github/workflows/release.yml zips it,
    # scripts/stamp-swiftpm.sh pins a SwiftPM binaryTarget URL ending in
    # libsofuu.xcframework.zip, and docs/EMBEDDING-DIST.md documents it.
    # The name that has to match the module is the one INSIDE (Sofuu.framework),
    # because that is what `-F` + `import Sofuu` resolves. The outer container
    # name is arbitrary; the xcframework's Info.plist carries the real
    # LibraryPath.
    local STAGE="$DIST/.xcframework-stage"
    rm -rf "$STAGE"
    mkdir -p "$STAGE/device" "$STAGE/sim"
    ios_make_framework "$DEVICE_LIB" "$STAGE/device"
    ios_make_framework "$SIM_LIB"    "$STAGE/sim"

    rm -rf "$DIST/libsofuu.xcframework"
    xcodebuild -create-xcframework \
        -framework "$STAGE/device/Sofuu.framework" \
        -framework "$STAGE/sim/Sofuu.framework" \
        -output "$DIST/libsofuu.xcframework" 2>&1 | tail -5
    rm -rf "$STAGE"

    # The inner framework name is load-bearing: `import Sofuu` resolves by
    # looking for Sofuu.framework, so a mismatch here reproduces the original
    # "cannot find 'sofuu_rt_call' in scope" failure. Assert it rather than
    # let the app find out.
    if [ -f "$DIST/libsofuu.xcframework/ios-arm64/Sofuu.framework/Modules/module.modulemap" ] \
       && [ -f "$DIST/libsofuu.xcframework/ios-arm64-simulator/Sofuu.framework/Modules/module.modulemap" ]; then
        echo "  \033[32m✓\033[0m libsofuu.xcframework (Sofuu.framework, device + simulator)"
        du -sh "$DIST/libsofuu.xcframework"
    else
        echo "  \033[31m✗ xcframework creation failed or is missing a slice\033[0m"
        find "$DIST/libsofuu.xcframework" -maxdepth 3 2>/dev/null | head -12
        exit 1
    fi

    # Stale byproduct of the earlier `-library … -headers …` form. That shape
    # linked fine but could never satisfy `import Sofuu`, and leaving the
    # loose header dir around invites someone to wire it back up.
    rm -rf "$DIST/libsofuu-headers"
}

# Copy header.
mkdir -p "$DIST"
cp "$REPO_ROOT/include/sofuu_embed.h" "$DIST/sofuu_embed.h" 2>/dev/null || true

case "$TARGET_ARG" in
    device) build_target "device" "aarch64-apple-ios" "libsofuu-ios-arm64.a" ;;
    sim)    build_target "sim" "aarch64-apple-ios-sim" "libsofuu-ios-sim.a" ;;
    deps)
        # Third-party stack only — no Rust, no xcframework.
        ensure_deps device
        ensure_deps sim
        echo ""
        echo "  \033[32m✓ iOS dependency stack ready\033[0m"
        ls -lh "$DIST"/libcurl-ios-*/lib/libcurl.a 2>/dev/null || true
        exit 0
        ;;
    xcframework) build_xcframework ;;
    all)
        build_target "device" "aarch64-apple-ios" "libsofuu-ios-arm64.a"
        build_target "sim" "aarch64-apple-ios-sim" "libsofuu-ios-sim.a"
        build_xcframework
        ;;
    *) echo "Usage: $0 [device|sim|deps|xcframework|all]"; exit 1 ;;
esac

echo ""
echo "  \033[32mDone. iOS artifacts in dist/:\033[0m"
ls -lh "$DIST"/libsofuu-ios-* "$DIST/sofuu_embed.h" 2>/dev/null || true
if [ -d "$DIST/libsofuu.xcframework" ]; then
    echo "  xcframework: $(du -sh "$DIST/libsofuu.xcframework" | cut -f1)"
fi
