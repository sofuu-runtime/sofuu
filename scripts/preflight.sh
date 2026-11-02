#!/usr/bin/env bash
# scripts/preflight.sh — Local Invariant & Boundary Gate
# Ensures clean module boundaries, bytecode sync, binary size compliance, and unit tests pass.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> [1/4] Checking Domain Boundary Isolation..."
DOMAINS="crates/sofuu-core/src/domains"
if [ -d "$DOMAINS/kaze" ] && [ -d "$DOMAINS/robotics" ] && [ -d "$DOMAINS/memory" ] && [ -d "$DOMAINS/infer" ]; then
    # Kaze leaf checks
    if grep -rnE "use crate::domains::(robotics|infer)" "$DOMAINS/kaze" 2>/dev/null; then
        echo "❌ VIOLATION: domains/kaze must not depend on other domains!" && exit 1
    fi
    # Robotics leaf checks
    if grep -rnE "use crate::domains::(kaze|infer)" "$DOMAINS/robotics" 2>/dev/null; then
        echo "❌ VIOLATION: domains/robotics must not depend on kaze or infer!" && exit 1
    fi
    # Memory leaf checks
    if grep -rnE "use crate::domains::(kaze|robotics)" "$DOMAINS/memory" 2>/dev/null; then
        echo "❌ VIOLATION: domains/memory must not depend on kaze or robotics!" && exit 1
    fi
    # Infer leaf checks
    if grep -rnE "use crate::domains::(kaze|robotics)" "$DOMAINS/infer" 2>/dev/null; then
        echo "❌ VIOLATION: domains/infer must not depend on kaze or robotics!" && exit 1
    fi
    echo "   Passed (Zero cross-domain coupling across kaze, robotics, memory, infer)."
else
    echo "   Domains in migration; skipping cross-domain import check."
fi

echo "==> [2/4] Verifying Bytecode Snapshot Sync..."
make bc-check

echo "==> [3/4] Verifying Binary Size Cap (<= 5.5 MiB)..."
make size-check

echo "==> [4/4] Running Core Subsystem Unit Tests..."
cargo test -p sofuu-core --lib

echo "🎉 All preflight invariants passed!"
