#!/usr/bin/env bash
# scripts/cross/zig-ranlib.sh — zig ranlib for cmake's CMAKE_RANLIB.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/../../tools/zig/zig" ranlib "$@"
