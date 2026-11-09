#!/usr/bin/env bash
# scripts/mobile/verify_android_aar.sh — reject an AAR that compiles but
# cannot load on Android.
#
# The checks cover the three packaging failures that shipped in 0.3.0:
#   * libc++_shared.so missing even though libsofuu.so NEEDs it
#   * absolute build paths in libsofuu_jni.so's DT_NEEDED (no SONAME)
#   * 4 KiB LOAD alignment on devices with 16 KiB pages
#
# Usage: bash scripts/mobile/verify_android_aar.sh <path-to.aar>
set -euo pipefail

AAR="${1:?usage: verify_android_aar.sh <path-to.aar>}"
[ -f "$AAR" ] || { echo "✗ AAR not found: $AAR" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

find_readelf() {
    if [ -n "${READELF:-}" ] && [ -x "$READELF" ]; then
        printf '%s\n' "$READELF"
        return
    fi
    local ndk="${ANDROID_NDK_HOME:-${ANDROID_NDK:-$HOME/Library/Android/sdk/ndk/27.0.12077973}}"
    local host_tag
    case "$(uname -s)" in
        Darwin) host_tag="darwin-x86_64" ;;
        Linux) host_tag="linux-x86_64" ;;
        *) echo "✗ unsupported host for NDK readelf" >&2; exit 1 ;;
    esac
    local candidate="$ndk/toolchains/llvm/prebuilt/$host_tag/bin/llvm-readelf"
    if [ -x "$candidate" ]; then
        printf '%s\n' "$candidate"
        return
    fi
    if command -v llvm-readelf >/dev/null 2>&1; then
        command -v llvm-readelf
        return
    fi
    echo "✗ llvm-readelf not found; set READELF or ANDROID_NDK_HOME" >&2
    exit 1
}

READELF="$(find_readelf)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() {
    echo "✗ $*" >&2
    exit 1
}

# --- native payload -------------------------------------------------------
for abi in arm64-v8a x86_64; do
    for lib in libsofuu.so libsofuu_jni.so libc++_shared.so; do
        entry="jni/$abi/$lib"
        unzip -l "$AAR" "$entry" >/dev/null 2>&1 \
            || fail "AAR missing $entry"
        unzip -p "$AAR" "$entry" > "$TMP/$abi-$lib"
        [ -s "$TMP/$abi-$lib" ] || fail "AAR entry is empty: $entry"
    done

    engine="$TMP/$abi-libsofuu.so"
    jni="$TMP/$abi-libsofuu_jni.so"

    "$READELF" -d "$engine" > "$TMP/$abi-engine.dynamic"
    grep -q 'Library soname: \[libsofuu.so\]' "$TMP/$abi-engine.dynamic" \
        || fail "$abi/libsofuu.so has no SONAME"

    "$READELF" -d "$jni" > "$TMP/$abi-jni.dynamic"
    grep -q 'Shared library: \[libsofuu.so\]' "$TMP/$abi-jni.dynamic" \
        || fail "$abi/libsofuu_jni.so does not NEED libsofuu.so"
    if grep 'Shared library:' "$TMP/$abi-jni.dynamic" | grep -q '/'; then
        fail "$abi/libsofuu_jni.so has an absolute DT_NEEDED path"
    fi

    for lib in libsofuu.so libsofuu_jni.so; do
        "$READELF" -l "$TMP/$abi-$lib" > "$TMP/$abi-$lib.program"
        bad_align="$(awk '/^[[:space:]]*LOAD/ {print $NF}' "$TMP/$abi-$lib.program" \
            | grep -v '^0x4000$' || true)"
        [ -z "$bad_align" ] \
            || fail "$abi/$lib has a LOAD segment not aligned to 0x4000: $bad_align"
    done
done

# --- legal notices --------------------------------------------------------
unzip -p "$AAR" classes.jar > "$TMP/classes.jar"
for notice in META-INF/LICENSES/QTSQ-FORMAT.txt META-INF/SOFUU-NOTICE.txt; do
    unzip -l "$TMP/classes.jar" "$notice" >/dev/null 2>&1 \
        || fail "classes.jar missing $notice"
done

echo "✓ AAR native payload verified: SONAME, relative NEEDED, 16 KiB alignment, libc++, notices"
