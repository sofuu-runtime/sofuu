#!/usr/bin/env bash
# scripts/dist-dryrun.sh — F3 (funnel plan): everything that CAN be verified
# about the distribution channels without registry credentials, in one gate.
#
#   make dist-dryrun     (or: scripts/dist-dryrun.sh)
#
# Per channel, this runs the local half of the publish path and FAILS on a
# broken package (the live pushes — `pod trunk push`, Maven Central upload,
# `npm publish` — stay owner-gated: they need credentials and they are
# outward-facing). Skips a channel with a yellow note when its toolchain is
# absent on this machine ("not verifiable here" is not a pass).
set -u

RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; RST=$'\033[0m'
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail=0

note_ok()   { printf '%s✓ %s%s\n' "$GRN" "$1" "$RST"; }
note_skip() { printf '%s— %s (skipped: %s)%s\n' "$YEL" "$1" "$2" "$RST"; }
note_fail() { printf '%s✗ %s%s\n' "$RED" "$1" "$RST"; fail=1; }

# ── prerequisites: the shared library ────────────────────────────────
if [ ! -f dist/libsofuu.dylib ] && [ ! -f dist/libsofuu.so ]; then
    note_fail "dist/libsofuu missing — run 'make libsofuu' first"
    exit 1
fi
note_ok "dist library present"

# ── channel 1: SwiftPM (Package.swift + checksum stamp) ─────────────
if command -v swift >/dev/null 2>&1; then
    if swift package dump-package > /dev/null 2>&1; then
        note_ok "SwiftPM manifest parses (swift package dump-package)"
    else
        note_fail "Package.swift does not parse"
    fi
    if [ -f dist/libsofuu.xcframework.zip ]; then
        want="$(shasum -a 256 dist/libsofuu.xcframework.zip | awk '{print $1}')"
        got="$(grep -o '[a-f0-9]\{64\}' Package.swift | head -1 || true)"
        if [ "$want" = "$got" ]; then
            note_ok "SwiftPM checksum matches the xcframework zip"
        else
            note_fail "Package.swift checksum stale vs dist/libsofuu.xcframework.zip (run scripts/stamp-swiftpm.sh)"
        fi
    else
        note_skip "SwiftPM checksum" "no dist/libsofuu.xcframework.zip (built by release.yml)"
    fi
else
    note_skip "SwiftPM" "swift toolchain absent"
fi

# ── channel 2: CocoaPods (podspec lint via the minimal shim) ────────
if [ -f Sofuu.podspec ]; then
    if command -v ruby >/dev/null 2>&1 && [ -f scripts/check_podspec.rb ]; then
        if ruby scripts/check_podspec.rb > /dev/null 2>&1; then
            note_ok "podspec passes the manifest check"
        else
            note_fail "Sofuu.podspec fails scripts/check_podspec.rb"
        fi
    else
        note_skip "CocoaPods" "ruby or the check shim absent"
    fi
    if command -v pod >/dev/null 2>&1; then
        note_ok "pod CLI present — live 'pod trunk push' stays owner-gated"
    else
        note_skip "CocoaPods trunk" "pod CLI absent"
    fi
else
    note_skip "CocoaPods" "Sofuu.podspec absent"
fi

# ── channel 3: Android (Gradle AAR → Maven Local) ───────────────────
if [ -f bindings/android/build.gradle ]; then
    if command -v gradle >/dev/null 2>&1 || [ -f bindings/android/gradlew ]; then
        cd bindings/android
        out="$( (./gradlew tasks --no-daemon 2>&1 || gradle tasks --no-daemon 2>&1) )"
        cd "$ROOT"
        if echo "$out" | grep -q "BUILD SUCCESSFUL"; then
            note_ok "Gradle project evaluates (bindings/android)"
        elif echo "$out" | grep -q "SDK location not found"; then
            note_skip "Android/Gradle" "no Android SDK on this machine (gradle eval needs ANDROID_HOME)"
        else
            note_fail "Gradle cannot evaluate bindings/android (unexpected — see gradle output)"
        fi
        note_ok "Maven path = publishToMavenLocal first; Central upload stays owner-gated"
    else
        note_skip "Android/Gradle" "gradle + gradlew absent"
    fi
else
    note_skip "Android/Gradle" "bindings/android absent"
fi

# ── channel 4: npm CLI pack (dry — never publishes) ─────────────────
if command -v npm >/dev/null 2>&1; then
    npm_dir=""
    if [ -f npm/package.json ]; then npm_dir=npm; fi
    if [ -n "$npm_dir" ]; then
        if (cd "$npm_dir" && npm pack --dry-run > /dev/null 2>&1); then
            note_ok "npm pack --dry-run OK ($npm_dir/ builds a packable tarball)"
        else
            note_fail "npm pack --dry-run failed in $npm_dir/"
        fi
    else
        note_fail "npm CLI package missing (npm/package.json)"
    fi
else
    note_skip "npm" "npm absent"
fi

if [ "$fail" -ne 0 ]; then
    printf '%sdist-dryrun FAILED — fix the channels above before release%s\n' "$RED" "$RST"
    exit 1
fi
printf '%sdist-dryrun passed — all present channels verified; live pushes remain owner-gated%s\n' "$GRN" "$RST"
