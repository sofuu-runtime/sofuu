#!/usr/bin/env bash
# scripts/cross/zig-cxx.sh — cc-rs wrapper for `zig c++` (see zig-cc.sh).
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/zig-cc.sh" --cxx "$@"
