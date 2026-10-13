#!/usr/bin/env bash
# scripts/mobile/sdk_jvm_check.sh — exercise the Android SDK wrapper on the
# host JVM. No device, no emulator, no network (after the first fetch).
#
# The AAR's Kotlin wrapper is thin, but the code under it is not: a JNI shim,
# the C ABI, and the runtime's event loop. This script closes the gap between
# "the Kotlin compiles" and "it works on a phone":
#
#   1. compiles bindings/android/sofuu/src/main/cpp/jni_bridge.c — the same
#      file the NDK builds for Android — for the HOST OS, linked against
#      dist/libsofuu
#   2. compiles the real Kotlin wrapper (against the Android SDK's android.jar)
#   3. runs sdk_jvm_check/Check.kt through it: streaming deltas, mid-flight
#      cancel, the UTF-8 boundary, streamLocal's positional contract, host
#      tools (sync + deferred from another thread), and close() semantics
#
# Skips cleanly (exit 0, a yellow line saying what is missing) when a host
# toolchain piece is absent — that is "not verifiable here", not a pass.
# Needs `make libsofuu` first (dist/libsofuu.{dylib,so}).
set -euo pipefail

RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; RST=$'\033[0m'

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECK_DIR="$REPO_ROOT/scripts/mobile/sdk_jvm_check"
BUILD="$REPO_ROOT/dist/sdk-jvm-check"

skip() { printf '%s— %s%s\n' "$YEL" "$1" "$RST"; exit 0; }

# ── the runtime must already be built ────────────────────────────────
case "$(uname -s)" in
    Darwin) RUNTIME_LIB="$REPO_ROOT/dist/libsofuu.dylib"; SHIM_NAME="libsofuu_jni.dylib"; JDK_OS=darwin ;;
    Linux)  RUNTIME_LIB="$REPO_ROOT/dist/libsofuu.so";    SHIM_NAME="libsofuu_jni.so";    JDK_OS=linux ;;
    *)      skip "sdk-jvm-check: unsupported host OS $(uname -s)" ;;
esac
if [ ! -f "$RUNTIME_LIB" ]; then
    printf '%s✗%s dist runtime missing (%s) — run `make libsofuu` first\n' \
        "$RED" "$RST" "$RUNTIME_LIB"
    exit 1
fi

# ── C compiler ───────────────────────────────────────────────────────
CC="${CC:-clang}"
command -v "$CC" >/dev/null 2>&1 || skip "sdk-jvm-check: no C compiler ($CC not found)"

# ── JDK (with JNI headers — java alone is not enough) ────────────────
JDK_HOME="${JAVA_HOME:-}"
if [ -z "$JDK_HOME" ] && command -v /usr/libexec/java_home >/dev/null 2>&1; then
    JDK_HOME="$(/usr/libexec/java_home 2>/dev/null || true)"
fi
if [ -z "$JDK_HOME" ] || [ ! -f "$JDK_HOME/include/jni.h" ]; then
    skip "sdk-jvm-check: no JDK with JNI headers (set JAVA_HOME)"
fi
JAVA="$JDK_HOME/bin/java"

# ── android.jar (the wrapper uses org.json + android.util): compile-only ──
SDK_DIR="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
ANDROID_JAR=""
for candidate in "$SDK_DIR"/platforms/android-*/android.jar; do
    [ -f "$candidate" ] && ANDROID_JAR="$candidate"   # last one wins → highest API
done
if [ -z "$ANDROID_JAR" ]; then
    skip "sdk-jvm-check: no android.jar under $SDK_DIR/platforms (set ANDROID_HOME)"
fi

# ── org.json for the JVM (android.jar's copy is compile-only stubs) ──
JSON_JAR=""
if [ -d "$HOME/.gradle/caches/modules-2" ]; then
    JSON_JAR="$(find "$HOME/.gradle/caches/modules-2" -path '*org.json*' -name 'json-*.jar' \
        ! -name '*-sources.jar' 2>/dev/null | sort | tail -1 || true)"
fi
if [ -z "$JSON_JAR" ]; then
    JSON_JAR="$BUILD/json.jar"
    if [ ! -f "$JSON_JAR" ]; then
        mkdir -p "$BUILD"
        curl -sSfL -o "$JSON_JAR" \
            https://repo1.maven.org/maven2/org/json/json/20240303/json-20240303.jar 2>/dev/null \
            || skip "sdk-jvm-check: org.json jar not in the gradle cache and not fetchable"
    fi
fi

# ── Kotlin compiler (from the gradle cache — the repo already has one
#    after any Android build) ─────────────────────────────────────────
GRADLE_CACHE="$HOME/.gradle/caches/modules-2"
find_jar() {
    [ -d "$GRADLE_CACHE" ] || return 0
    find "$GRADLE_CACHE" -name "$1" ! -name '*-sources.jar' 2>/dev/null | sort | tail -1 || true
}
KC="$(find_jar 'kotlin-compiler-embeddable-*.jar')"
if [ -z "$KC" ]; then
    skip "sdk-jvm-check: kotlin-compiler-embeddable not in ~/.gradle/caches (run the Android build once to populate it)"
fi
# The compiler's version decides the stdlib we may read: a NEWER stdlib in
# the cache (another project's) has newer metadata and the compiler refuses
# it, so take the exact match first and otherwise the oldest.
KC_VER="$(printf '%s' "$KC" | sed -n 's|.*/kotlin-compiler-embeddable/\([^/]*\)/.*|\1|p')"
kotlin_jar() { # prefix, e.g. kotlin-stdlib
    local exact
    exact="$(find_jar "${1}-${KC_VER}.jar")"
    if [ -n "$exact" ] && [ "$exact" != "$KC" ]; then
        printf '%s' "$exact"
        return
    fi
    [ -d "$GRADLE_CACHE" ] || return 0
    find "$GRADLE_CACHE" -name "${1}-[0-9]*.jar" ! -name '*-sources.jar' 2>/dev/null | sort | head -1 || true
}
STDLIB="$(kotlin_jar kotlin-stdlib)"
if [ -z "$STDLIB" ]; then
    skip "sdk-jvm-check: kotlin-stdlib $KC_VER not in ~/.gradle/caches"
fi
# The compiler's own dependencies; trove4j and annotations are fetched if the
# cache lacks them (both are small and stable).
REFLECT="$(kotlin_jar kotlin-reflect)"
SCRIPT_RT="$(kotlin_jar kotlin-script-runtime)"
DAEMON="$(kotlin_jar kotlin-daemon-embeddable)"
COROUTINES="$(find_jar 'kotlinx-coroutines-core-jvm-*.jar')"
TROVE="$(find_jar 'trove4j-*.jar')"
ANNOTATIONS="$(find_jar 'annotations-13*.jar')"
fetch_if_missing() { # url dest label
    [ -f "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    curl -sSfL -o "$2" "$1" 2>/dev/null || skip "sdk-jvm-check: $3 not available (cache miss + no network)"
}
if [ -z "$TROVE" ]; then
    TROVE="$BUILD/trove4j.jar"
    fetch_if_missing https://repo1.maven.org/maven2/org/jetbrains/intellij/deps/trove4j/1.0.20200330/trove4j-1.0.20200330.jar "$TROVE" "trove4j"
fi
if [ -z "$ANNOTATIONS" ]; then
    ANNOTATIONS="$BUILD/annotations.jar"
    fetch_if_missing https://repo1.maven.org/maven2/org/jetbrains/annotations/24.1.0/annotations-24.1.0.jar "$ANNOTATIONS" "org.jetbrains annotations"
fi
COMPILER_CP="$KC:$STDLIB:$REFLECT:$SCRIPT_RT:$COROUTINES:$DAEMON:$TROVE:$ANNOTATIONS"

# ── 1. the JNI shim, host OS ─────────────────────────────────────────
SHIM="$BUILD/$SHIM_NAME"
mkdir -p "$BUILD/classes" "$BUILD/checkclasses" "$BUILD/home"
printf '  %sCC %s jni_bridge.c → %s\n' "$DIM" "$RST" "$SHIM_NAME"
"$CC" -shared -fPIC -O2 -Wall -Wextra \
    -o "$SHIM" \
    "$REPO_ROOT/bindings/android/sofuu/src/main/cpp/jni_bridge.c" \
    -I "$REPO_ROOT/bindings/android/sofuu/src/main/cpp/include" \
    -I "$CHECK_DIR/android_stub" \
    -I "$JDK_HOME/include" -I "$JDK_HOME/include/$JDK_OS" \
    -L "$(dirname "$RUNTIME_LIB")" -lsofuu \
    -Wl,-rpath,"$(dirname "$RUNTIME_LIB")"

# ── 2. the Kotlin wrapper, then the check ────────────────────────────
SOFUU_KT="$REPO_ROOT/bindings/android/sofuu/src/main/java/com/sofuu/runtime"
printf '  %sKTC%s SofuuBridge.kt → classes/\n' "$DIM" "$RST"
"$JAVA" -cp "$COMPILER_CP" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler \
    -no-stdlib -jvm-target 17 -classpath "$ANDROID_JAR:$STDLIB" \
    -d "$BUILD/classes" \
    "$SOFUU_KT/SofuuBridge.kt" "$SOFUU_KT/SofuuVoice.kt"
printf '  %sKTC%s Check.kt\n' "$DIM" "$RST"
"$JAVA" -cp "$COMPILER_CP" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler \
    -no-stdlib -jvm-target 17 -classpath "$ANDROID_JAR:$STDLIB:$BUILD/classes:$JSON_JAR" \
    -d "$BUILD/checkclasses" "$CHECK_DIR/Check.kt"

# ── 3. run it (scratch HOME keeps ~/.sofuu untouched) ────────────────
printf '\n'
HOME="$BUILD/home" "$JAVA" \
    -cp "$BUILD/classes:$BUILD/checkclasses:$STDLIB:$JSON_JAR" \
    -Djava.library.path="$BUILD" \
    CheckKt
