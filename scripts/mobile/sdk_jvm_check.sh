#!/usr/bin/env bash
# scripts/mobile/sdk_jvm_check.sh — exercise the Android SDK wrapper on the
# host JVM. No device, no emulator, no network (after the first fetch).
#
# The AAR's Kotlin wrapper is thin, but the code under it is not: a JNI shim,
# the C ABI, and the runtime's event loop. This script closes the gap between
# "the Kotlin compiles" and "it works on a phone":
#
#   1. builds the runtime (libsofuu) from THIS source tree
#   2. compiles bindings/android/sofuu/src/main/cpp/jni_bridge.c — the same
#      file the NDK builds for Android — for the HOST OS, linked against
#      that runtime
#   3. compiles the real Kotlin wrapper (against the Android SDK's android.jar)
#   4. runs sdk_jvm_check/Check.kt through it: real local inference with an
#      existing GGUF, streaming deltas, mid-flight cancel, the UTF-8
#      boundary, streamLocal's positional contract, and host tools (sync +
#      deferred from another thread) reading a real temporary file
#
# REAL EXECUTION (RULES.md §1.9): the harness drives the real engine
# against a real local GGUF — no provider is stubbed, no answer is canned.
# The camera is NOT covered here: it needs a physical Android device, so the
# harness records camera coverage as unverified instead of faking it.
#
# Reports BLOCKED (exit 2) when a real prerequisite is missing — that is
# "not verifiable here", not a pass. The result is HOST-JVM coverage: this
# is not Android device coverage.
#
# Usage: bash scripts/mobile/sdk_jvm_check.sh [model.gguf]
#   model   an existing real-weight GGUF (default: $SOFUU_JVM_CHECK_MODEL,
#           else ~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf)
#
# Environment:
#   SOFUU_JVM_CHECK_MODEL   same as the model argument
#   SOFUU_JVM_CHECK_LIB     use this prebuilt libsofuu instead of building one
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECK_DIR="$REPO_ROOT/scripts/mobile/sdk_jvm_check"
BUILD="$REPO_ROOT/dist/sdk-jvm-check"

RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; RST=$'\033[0m'

blocked() { printf '%sBLOCKED%s sdk-jvm-check: %s\n' "$YEL" "$RST" "$1" >&2; exit 2; }
fail()    { printf '%sFAIL%s sdk-jvm-check: %s\n' "$RED" "$RST" "$1" >&2; exit 1; }

# ── the real local model the inference cases run against ─────────────
MODEL="${1:-${SOFUU_JVM_CHECK_MODEL:-}}"
if [ -z "$MODEL" ]; then
    for candidate in \
        "$HOME/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf" \
        "$HOME/Downloads/SmolLM2-360M-Instruct-Q4_K_M.gguf"
    do
        if [ -f "$candidate" ]; then MODEL="$candidate"; break; fi
    done
fi
if [ -n "$MODEL" ]; then
    [ -f "$MODEL" ] || blocked "model file not found: $MODEL"
    # A synthetic or vocabulary-only fixture is a parser fixture, never
    # inference coverage.
    case "$(basename "$MODEL")" in
        sofuu-fixture.gguf|ggml-vocab-*.gguf)
            blocked "the synthetic/vocab-only fixture is not real inference coverage: $MODEL" ;;
    esac
    MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
fi

# ── the runtime, built from this source tree ────────────────────────
case "$(uname -s)" in
    Darwin) LIB_NAME="libsofuu.dylib"; SHIM_NAME="libsofuu_jni.dylib"; JDK_OS=darwin ;;
    Linux)  LIB_NAME="libsofuu.so";    SHIM_NAME="libsofuu_jni.so";    JDK_OS=linux ;;
    *)      blocked "unsupported host OS $(uname -s)" ;;
esac

mkdir -p "$BUILD"
if [ -n "${SOFUU_JVM_CHECK_LIB:-}" ]; then
    RUNTIME_SRC="$SOFUU_JVM_CHECK_LIB"
    [ -f "$RUNTIME_SRC" ] || blocked "SOFUU_JVM_CHECK_LIB not found: $RUNTIME_SRC"
else
    CARGO="${CARGO:-cargo}"
    command -v "$CARGO" >/dev/null 2>&1 || blocked "no cargo ($CARGO not found) — the runtime cannot be built from source"
    printf '  %sCARGO%s build sofuu-capi (release) — the check runs THIS source tree\n' "$DIM" "$RST"
    REMAP_FLAGS="$(bash "$REPO_ROOT/scripts/cross/remap_flags.sh" 2>/dev/null || true)"
    if ! ( cd "$REPO_ROOT" && \
        CARGO_PROFILE_RELEASE_PANIC=unwind RUSTFLAGS="$REMAP_FLAGS ${RUSTFLAGS:-}" \
        "$CARGO" build --offline --release -p sofuu-capi ); then
        blocked "cargo build --offline --release -p sofuu-capi failed — the compiler output above is the reason"
    fi
    case "$(uname -s)" in
        Darwin) RUNTIME_SRC="$REPO_ROOT/target/release/libsofuu_capi.dylib" ;;
        Linux)  RUNTIME_SRC="$REPO_ROOT/target/release/libsofuu_capi.so" ;;
    esac
fi
[ -f "$RUNTIME_SRC" ] || blocked "runtime library not found after the build: $RUNTIME_SRC"

# Provenance of the library the check runs against: a receipt has to name
# the binary, not just "the runtime".
printf '  %sLIB%s %s\n' "$DIM" "$RST" "$RUNTIME_SRC"
if command -v shasum >/dev/null 2>&1; then
    printf '  %sSHA%s %s\n' "$DIM" "$RST" "$(shasum -a 256 "$RUNTIME_SRC" | cut -d' ' -f1)"
fi
if command -v git >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    printf '  %sREV%s %s (%s)\n' "$DIM" "$RST" \
        "$(git -C "$REPO_ROOT" rev-parse --short HEAD)" \
        "$(git -C "$REPO_ROOT" status --porcelain | grep -q . && echo dirty || echo clean)"
fi

# The build-tree dylib records its absolute path as its install name; the
# host-JVM shim loads it by @rpath, so copy it next to the shim and give it
# the relocatable id (the same treatment `make libsofuu` applies).
RUNTIME_LIB="$BUILD/$LIB_NAME"
if [ "$RUNTIME_SRC" != "$RUNTIME_LIB" ]; then
    cp "$RUNTIME_SRC" "$RUNTIME_LIB"
fi
if [ "$(uname -s)" = Darwin ]; then
    install_name_tool -id "@rpath/$LIB_NAME" "$RUNTIME_LIB" 2>/dev/null || true
    codesign -s - -f "$RUNTIME_LIB" 2>/dev/null || true
fi

# ── C compiler ───────────────────────────────────────────────────────
CC="${CC:-clang}"
command -v "$CC" >/dev/null 2>&1 || blocked "no C compiler ($CC not found)"

# ── JDK (with JNI headers — java alone is not enough) ────────────────
JDK_HOME="${JAVA_HOME:-}"
if [ -z "$JDK_HOME" ] && command -v /usr/libexec/java_home >/dev/null 2>&1; then
    JDK_HOME="$(/usr/libexec/java_home 2>/dev/null || true)"
fi
if [ -z "$JDK_HOME" ] || [ ! -f "$JDK_HOME/include/jni.h" ]; then
    blocked "no JDK with JNI headers (set JAVA_HOME)"
fi
JAVA="$JDK_HOME/bin/java"

# ── android.jar (the wrapper uses org.json + android.util): compile-only ──
SDK_DIR="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
ANDROID_JAR=""
for candidate in "$SDK_DIR"/platforms/android-*/android.jar; do
    [ -f "$candidate" ] && ANDROID_JAR="$candidate"   # last one wins → highest API
done
if [ -z "$ANDROID_JAR" ]; then
    blocked "no android.jar under $SDK_DIR/platforms (set ANDROID_HOME)"
fi

# ── org.json for the JVM (android.jar's copy is compile-only stubs) ──
JSON_JAR=""
if [ -d "$HOME/.gradle/caches/modules-2" ]; then
    JSON_JAR="$(find "$HOME/.gradle/caches/modules-2" -path '*org.json*' -name 'json-*.jar' \
        ! -name '*-sources.jar' 2>/dev/null | sort | tail -1 || true)"
fi
if [ -z "$JSON_JAR" ] && [ -d "$HOME/.m2/repository/org/json" ]; then
    JSON_JAR="$(find "$HOME/.m2/repository/org/json" -name 'json-*.jar' \
        ! -name '*-sources.jar' 2>/dev/null | sort | tail -1 || true)"
fi
if [ -z "$JSON_JAR" ]; then
    JSON_JAR="$BUILD/json.jar"
    if [ ! -f "$JSON_JAR" ]; then
        mkdir -p "$BUILD"
        curl -sSfL -o "$JSON_JAR" \
            https://repo1.maven.org/maven2/org/json/json/20240303/json-20240303.jar 2>/dev/null \
            || blocked "org.json jar not in the gradle cache, the maven cache or $BUILD (and not fetchable)"
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
    blocked "kotlin-compiler-embeddable not in ~/.gradle/caches (run the Android build once to populate it)"
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
    blocked "kotlin-stdlib $KC_VER not in ~/.gradle/caches"
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
    curl -sSfL -o "$2" "$1" 2>/dev/null || blocked "$3 not available (cache miss + no network)"
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
    -L "$BUILD" -lsofuu \
    -Wl,-rpath,"$BUILD" \
    || fail "the JNI shim did not compile (see the compiler output above)"

# ── 2. the Kotlin wrapper, then the check ────────────────────────────
SOFUU_KT="$REPO_ROOT/bindings/android/sofuu/src/main/java/com/sofuu/runtime"
printf '  %sKTC%s SofuuBridge.kt → classes/\n' "$DIM" "$RST"
"$JAVA" -cp "$COMPILER_CP" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler \
    -no-stdlib -jvm-target 17 -classpath "$ANDROID_JAR:$STDLIB" \
    -d "$BUILD/classes" \
    "$SOFUU_KT/SofuuBridge.kt" "$SOFUU_KT/SofuuVoice.kt" \
    || fail "the Kotlin wrapper did not compile (see the compiler output above)"
printf '  %sKTC%s Check.kt\n' "$DIM" "$RST"
"$JAVA" -cp "$COMPILER_CP" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler \
    -no-stdlib -jvm-target 17 -classpath "$ANDROID_JAR:$STDLIB:$BUILD/classes:$JSON_JAR" \
    -d "$BUILD/checkclasses" "$CHECK_DIR/Check.kt" \
    || fail "Check.kt did not compile (see the compiler output above)"

# ── 3. run it (scratch HOME keeps ~/.sofuu untouched) ────────────────
printf '\n'
set +e
HOME="$BUILD/home" "$JAVA" \
    -cp "$BUILD/classes:$BUILD/checkclasses:$STDLIB:$JSON_JAR" \
    -Djava.library.path="$BUILD" \
    -Dsofuu.jvm.check.model="$MODEL" \
    CheckKt
rc=$?
set -e

printf '\n'
case "$rc" in
    0) printf '%sOK%s   sdk-jvm-check: host-JVM coverage only — NOT Android device coverage\n' "$GRN" "$RST" ;;
    2) printf '%sBLOCKED%s sdk-jvm-check: a real prerequisite was unavailable (see the BLOCKED lines above)\n' "$YEL" "$RST" ;;
    *) printf '%sFAIL%s  sdk-jvm-check: the host-JVM check failed (rc=%s)\n' "$RED" "$RST" "$rc" ;;
esac
exit "$rc"
