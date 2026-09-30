#!/usr/bin/env bash
# scripts/cross/zig-ar.sh — zig ar for cmake's CMAKE_AR (a bare `zig` can't
# be CMAKE_AR: cmake prepends ar-style flags like `qc`, which zig parses as
# subcommands).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/../../tools/zig/zig" ar "$@"
