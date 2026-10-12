#!/usr/bin/env bash
# scripts/mobile/ios_env.sh — shared iOS toolchain plumbing for the engine
# cross-builds. Source it:  source scripts/mobile/ios_env.sh <device|sim>
#
# The iOS counterpart of ndk_env.sh. Resolves the SDK, exports the toolchain
# and defines the per-slice identity (rust triple, OpenSSL target, arch) used
# by the OpenSSL / curl / Rust steps.
#
# Slices, not ABIs: iOS has exactly one architecture per SDK, so the only
# axis is device vs simulator.

IOS_MIN="${IOS_DEPLOYMENT_TARGET:-16.0}"

# Ninja if we can find one, else CMake's default generator (Unix Makefiles).
# `make` always exists with Xcode, so the iOS builds never hard-depend on
# ninja the way the NDK ones do — this machine has ninja only inside a venv.
IOS_CMAKE_GEN=""
if [ -n "${NINJA:-}" ] && [ -x "${NINJA}" ]; then
    IOS_CMAKE_GEN="-G Ninja"
elif command -v ninja >/dev/null 2>&1; then
    IOS_CMAKE_GEN="-G Ninja"
fi

ios_slice_env() {
    local slice="$1"
    case "$slice" in
        device|iphoneos)
            IOS_SLICE_NAME="device"
            IOS_SDK_NAME="iphoneos"
            IOS_RUST_TARGET="aarch64-apple-ios"
            IOS_OPENSSL_TARGET="ios64-xcrun"
            IOS_ARCH="arm64"
            IOS_EXPECTED_PLATFORM="IOS"
            # Device flag. Verified: yields `platform IOS`, `minos 16.0`.
            IOS_MIN_FLAG="-mios-version-min=$IOS_MIN"
            ;;
        sim|simulator|iphonesimulator)
            IOS_SLICE_NAME="sim"
            IOS_SDK_NAME="iphonesimulator"
            IOS_RUST_TARGET="aarch64-apple-ios-sim"
            IOS_OPENSSL_TARGET="iossimulator-arm64-xcrun"
            IOS_ARCH="arm64"
            IOS_EXPECTED_PLATFORM="IOSSIMULATOR"
            # MUST be the simulator spelling. Verified: `-mios-version-min`
            # against an iPhoneSimulator sysroot silently compiles for
            # `platform IOS` — i.e. it flips the slice back to device. Only
            # `-mios-simulator-version-min` yields `platform IOSSIMULATOR`.
            IOS_MIN_FLAG="-mios-simulator-version-min=$IOS_MIN"
            ;;
        *) echo "✗ unknown iOS slice: $slice (device|sim)"; exit 1 ;;
    esac

    IOS_SYSROOT="$(xcrun --sdk "$IOS_SDK_NAME" --show-sdk-path 2>/dev/null || true)"
    [ -n "$IOS_SYSROOT" ] && [ -d "$IOS_SYSROOT" ] || {
        echo "✗ no SDK for $IOS_SDK_NAME — is Xcode installed and selected?"
        exit 1
    }

    IOS_CC="$(xcrun --sdk "$IOS_SDK_NAME" -f clang)"
    IOS_CXX="$(xcrun --sdk "$IOS_SDK_NAME" -f clang++)"
    IOS_AR="$(xcrun -f ar)"
    IOS_RANLIB="$(xcrun -f ranlib)"

    # The cc crate (libsofuu's C sources) reads CC/CXX; rustc passes --target
    # itself for Apple triples, so no CFLAGS are needed here.
    export CC="$IOS_CC" CXX="$IOS_CXX" AR="$IOS_AR" RANLIB="$IOS_RANLIB"

    # Common CMake flags for a cross build against this SDK.
    IOS_CMAKE_FLAGS="-DCMAKE_SYSTEM_NAME=iOS \
-DCMAKE_OSX_ARCHITECTURES=$IOS_ARCH \
-DCMAKE_OSX_SYSROOT=$IOS_SYSROOT \
-DCMAKE_OSX_DEPLOYMENT_TARGET=$IOS_MIN \
-DCMAKE_POSITION_INDEPENDENT_CODE=ON"

    # Rust links through clang, and cargo's default "cc" would be the host
    # compiler — name the linker explicitly for the target triple.
    local linker_var="CARGO_TARGET_$(echo "$IOS_RUST_TARGET" | tr 'a-z-' 'A-Z_')_LINKER"
    export "$linker_var=$IOS_CC"

    # ── Deployment target for cc-compiled C/C++ (QuickJS) ────────────
    # The `cc` crate has NO deployment-target logic: cc-1.4.7's src/ never
    # mentions IPHONEOS_DEPLOYMENT_TARGET, -mios-version-min, or anything
    # similar (verified by grepping the crate source). So when it compiles
    # QuickJS it passes no min-version flag and clang records the *SDK*
    # version as the object minimum — `minos 26.2` on Xcode 26.2. Every app
    # link then emits "object file was built for newer iOS version (26.2)
    # than being linked (16.0)" for each of the ~200 QuickJS objects.
    #
    # cc-rs resolves flags through target_envs(), which tries
    # `CFLAGS_<target>` first, then `CFLAGS_<target_underscored>`, then
    # `TARGET_CFLAGS`, then `CFLAGS` — so the underscored spelling below is
    # the precise, non-leaking place to put the flag. Both C and C++
    # spellings, since either may be compiled.
    local tu="${IOS_RUST_TARGET//-/_}"
    export "CFLAGS_$tu=$IOS_MIN_FLAG"
    export "CXXFLAGS_$tu=$IOS_MIN_FLAG"
}

# ── Platform verification ────────────────────────────────────────────
# `lipo -info` only reports the *architecture*, so a macOS arm64 archive
# looks identical to an iOS arm64 one — which is exactly how a silently
# macOS libuv survived a build and only failed much later, at link time.
#
# These helpers therefore census EVERY member of the archive in a single
# `otool -l` pass (~1s for 3000 members) instead of probing one member.
# That distinction is not academic: an earlier single-member version of
# this check happily reported "platform: IOS ✓" for an archive that also
# contained 102 macOS `qtsq_*.o` objects, because the member it happened to
# probe was a legitimate iOS one. Probing one member proves nothing about
# the other 3015.
#
# Mach-O platform numbers: 1=macOS 2=iOS 3=tvOS 4=watchOS
# 6=MacCatalyst 7=iOS-simulator 8=tvOS-sim 9=watchOS-sim.
#
# Requires ios_slice_env to have run (uses $IOS_EXPECTED_PLATFORM).
#
# PIPEFAIL NOTE — every helper here survives `set -euo pipefail`, which all
# three iOS scripts set. Two traps bit during development:
#
#   * `ar t … | grep … | head -1` — head closes the pipe after one line, so
#     grep dies on SIGPIPE (141) and pipefail propagates it. awk's
#     `{print; exit}` reads one line and exits *by itself*, no SIGPIPE.
#   * `nm …` exits 1 on an archive that contains a symbol-table member even
#     though it printed every symbol correctly. Its status must never reach
#     the caller, so it is captured through a file with `|| true`.
#
# awk is used for all counting because it always exits 0 and prints a
# number, so `grep -c`'s exit-1-on-zero-match never reaches the caller.
ios_platform_name() {
    case "$1" in
        1) echo "MACOS" ;;          2) echo "IOS" ;;
        3) echo "TVOS" ;;           4) echo "WATCHOS" ;;
        6) echo "MACCATALYST" ;;    7) echo "IOSSIMULATOR" ;;
        8) echo "TVOSSIMULATOR" ;;  9) echo "WATCHOSSIMULATOR" ;;
        *) echo "platform-$1" ;;
    esac
}

# ios_archive_platform <archive> → dominant platform name across all members.
# Falls back to the legacy LC_VERSION_MIN_IPHONEOS form, which is what rustc
# emits for iOS (it carries no `platform N` field at all).
ios_archive_platform() {
    local archive="$1"
    [ -f "$archive" ] || { echo "missing"; return 0; }
    local best
    best="$(otool -l "$archive" 2>/dev/null | awk '
        /^ *platform [0-9]+$/ { c[$2]++; next }
        /LC_VERSION_MIN_IPHONEOS/ { legacy_ios = 1 }
        /LC_VERSION_MIN_MACOSX/   { legacy_mac = 1 }
        END {
            best = 0; bestn = 0
            for (p in c) if (c[p] > bestn) { bestn = c[p]; best = p }
            if (best != 0)      print best
            else if (legacy_mac) print 1
            else if (legacy_ios) print 2
            else                 print "unknown"
        }')"
    case "$best" in
        ''|unknown) echo "unknown" ;;
        *) ios_platform_name "$best" ;;
    esac
}

# Number of macOS-platform members in an archive. Must be 0 for anything
# that ships in an iOS app.
ios_archive_macos_count() {
    otool -l "$1" 2>/dev/null | awk '
        /^ *platform 1$/        { n++ }
        /LC_VERSION_MIN_MACOSX/ { n++ }
        END { print n + 0 }'
}

# ios_assert_no_macos <archive> <label> → hard fail on any macOS member.
ios_assert_no_macos() {
    local archive="$1" label="$2" n
    n="$(ios_archive_macos_count "$archive")"
    if [ "$n" = "0" ]; then
        echo "    no macOS objects ✓"
        return 0
    fi
    echo "  \033[31m✗ $label contains $n macOS (platform 1) object(s)\033[0m"
    echo "    They will link into the iOS binary and fail at runtime, or be"
    echo "    silently dead-stripped. Usual cause: a dependency resolved to a"
    echo "    host build — e.g. SOFUU_QTSQ_DIR falling back to a developer"
    echo "    checkout (see the note in scripts/dist/ios.sh)."
    echo "    Offending members:"
    otool -l "$archive" 2>/dev/null | awk '
        /^[^ ]/ && /:$/ { m = $0; sub(/:$/, "", m) }
        /^ *platform 1$/ { print m }
        /LC_VERSION_MIN_MACOSX/ { print m }' | sort -u | head -8 | sed 's/^/      /'
    exit 1
}

# Sorted, unique symbol names. See the PIPEFAIL NOTE above.
ios_undefined_symbols() {   # <archive> → undefined symbol names
    local archive="$1" raw
    raw="$(mktemp)"
    nm -u "$archive" > "$raw" 2>/dev/null || true
    awk '{print $NF}' "$raw" | sort -u
    rm -f "$raw"
}

ios_defined_symbols() {     # <archive> → global defined symbol names
    local archive="$1" raw
    raw="$(mktemp)"
    nm -gU "$archive" > "$raw" 2>/dev/null || true
    awk '{print $NF}' "$raw" | sort -u
    rm -f "$raw"
}

# ios_assert_platform <archive> <label> → exits 1 unless it matches the
# slice resolved by ios_slice_env. Set SOFUU_IOS_ALLOW_PLATFORM_MISMATCH=1
# to downgrade to a warning (e.g. when reusing a prebuilt archive).
ios_assert_platform() {
    local archive="$1" label="$2"
    local got; got="$(ios_archive_platform "$archive")"
    if [ "$got" = "$IOS_EXPECTED_PLATFORM" ]; then
        echo "    platform: $got ✓"
        return 0
    fi
    echo "  \033[31m✗ $label is platform '$got', expected '$IOS_EXPECTED_PLATFORM'\033[0m"
    echo "    A macOS archive links into a simulator build and then dies at"
    echo "    runtime or at app link — it never fails where it was built."
    if [ -n "${SOFUU_IOS_ALLOW_PLATFORM_MISMATCH:-}" ]; then
        echo "    SOFUU_IOS_ALLOW_PLATFORM_MISMATCH set — continuing anyway."
        return 0
    fi
    exit 1
}

# Sourced WITH a slice argument → resolve it immediately.
#
# Guarded on the *value*, not on `$#`, because `source` inherits the caller's
# positional parameters: `bash scripts/dist/ios.sh deps` would otherwise reach
# here with $1=deps and die on "unknown iOS slice: deps". Same trap for
# `build_libcurl_ios.sh all`. Matching only real slice names makes the
# convenience safe for every caller. Callers that want a specific slice
# should still call ios_slice_env explicitly.
case "${1:-}" in
    device|iphoneos|sim|simulator|iphonesimulator) ios_slice_env "$1" ;;
esac
