#!/usr/bin/env bash
# scripts/cross/zig-cc-win.sh — zig cc with the Windows mingw target baked in.
# For cmake's CMAKE_C_COMPILER, which must be a single executable with no
# prepended args (CMAKE_C_COMPILER_ARG1 is unreliable across generators —
# the Linux path got away with it, the Windows path did not).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/../../tools/zig/zig" cc -target x86_64-windows-gnu "$@"
