#!/usr/bin/env bash
# Run one explicit Rust library area. Cross-area and integration checks are
# selected separately, according to the change's actual dependencies.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
    echo "usage: bash scripts/test-area.sh {kaze|robotics|memory|inference|compiler|mcp|policy|sandbox|runtime|diagnostics|rlm}" >&2
}
[ "$#" -eq 1 ] || { usage; exit 2; }
case "$1" in
    kaze)        filter='domains::kaze::' ;;
    robotics)    filter='domains::robotics::' ;;
    memory)      filter='domains::memory::' ;;
    inference)   filter='domains::infer::' ;;
    compiler)    filter='compiler::' ;;
    mcp)         filter='mcp::' ;;
    policy)      filter='rt::policy::' ;;
    sandbox)     filter='rt::sandbox::' ;;
    runtime)     filter='rt::' ;;
    diagnostics) filter='rt::diag_contract::' ;;
    rlm)         filter='rlm::' ;;
    *) usage; exit 2 ;;
esac
printf 'Focused area: %s (Rust name filter %s)\n' "$1" "$filter"
printf 'Cross-area, integration, device and release checks are not included.\n'
SOFUU_RUST_FILTER="$filter" bash "$ROOT/scripts/preflight.sh" rust
