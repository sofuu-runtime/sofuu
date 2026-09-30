#!/usr/bin/env bash
# scripts/cross/zig-cc.sh — cc-rs compatibility wrapper for `zig cc`.
#
# cc-rs (the build-dependency that compiles QuickJS/SIMD/http-parser)
# injects the *rust-style* triple as `--target=x86_64-unknown-linux-musl`
# when cross-compiling. zig cc's target parser only understands zig-style
# triples (`x86_64-linux-musl`) and rejects the rust spelling with
# "UnknownOperatingSystem" — and the LAST --target on the command line
# wins, so pre-seeding our own `-target` in $CC does not help.
#
# This wrapper rewrites any rust-style `--target=<triple>` argument into
# the zig spelling and forwards everything else. Same recipe cargo-zigbuild
# uses. `zig-cxx.sh` is the same wrapper for `zig c++`.
#
# Usage (from the cross build scripts):
#   export CC="$REPO_ROOT/scripts/cross/zig-cc.sh"
#   export CXX="$REPO_ROOT/scripts/cross/zig-cxx.sh --cxx"
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIG="$DIR/../../tools/zig/zig"
SUB="cc"

if [ "${1:-}" = "--cxx" ]; then
    SUB="c++"
    shift
fi

declare -a ARGS=()
for a in "$@"; do
    case "$a" in
        --target=x86_64-unknown-linux-musl)   ARGS+=("-target" "x86_64-linux-musl") ;;
        --target=aarch64-unknown-linux-musl)  ARGS+=("-target" "aarch64-linux-musl") ;;
        --target=x86_64-unknown-linux-gnu)    ARGS+=("-target" "x86_64-linux-gnu") ;;
        --target=aarch64-unknown-linux-gnu)   ARGS+=("-target" "aarch64-linux-gnu") ;;
        --target=x86_64-pc-windows-gnu)       ARGS+=("-target" "x86_64-windows-gnu") ;;
        --target=aarch64-pc-windows-gnullvm)  ARGS+=("-target" "aarch64-windows-gnu") ;;
        --target=x86_64-apple-darwin)         ARGS+=("-target" "x86_64-macos") ;;
        --target=aarch64-apple-darwin)        ARGS+=("-target" "aarch64-macos") ;;
        *)                                    ARGS+=("$a") ;;
    esac
done

exec "$ZIG" "$SUB" "${ARGS[@]}"
