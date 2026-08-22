#!/usr/bin/env bash
# tests/run_js_tests.sh — consolidated JS test runner (Track A, A0.2).
#
# Runs every assert-based JS test in tests/ and reports pass/fail with
# proper exit codes. Each test file MUST exit 0 on success or non-zero
# on failure (the assert() helper calls process.exit(1) on failure).
#
# Usage:
#   make test           # runs the full suite (Makefile calls this)
#   ./tests/run_js_tests.sh  # run directly
#
# To add a new test: drop a .js file in tests/, add it to the list below.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
TESTS_DIR="$ROOT/tests"

if [ ! -f "$SOFUU" ]; then
    echo "✗ sofuu binary not found — run 'make' first" >&2
    exit 1
fi

# ── Test registry ─────────────────────────────────────────────────
# Each entry: "file|description"
TESTS=(
    "priority1_test.js|GC clean exit + fetch API + rejection handler"
    "priority2_test.js|ESM import + process API + stdin"
    "ts_test.ts|TypeScript stripper (interfaces/types/generics/classes)"
    "simd_test.js|SIMD vector operations (dot/cosine/L2)"
    "fs_test.js|Async file I/O (read/write/append/mkdir/readdir/rm)"
    "process_test.js|Process module (argv/env/cwd/exit/arch/uptime)"
    "spawn_test.js|Child process spawn (stdout/stderr/exit code)"
    "sse_test.js|SSE parser (data/event/id/retry/multi-line)"
    "npm_test.js|npm safety (spec validation/tar extraction)"
    "ai_complete_test.js|ai.complete (mock provider, tool calls)"
    "ai_stream_test.js|ai.stream (mock SSE streaming)"
    "fetch_test.js|fetch() Response API (status/headers/text/json)"
    "mcp_client_test.js|MCP client (connect/tools/call)"
    "mcp_server_test.js|MCP server (tool registration/handling)"
    "server_test.js|HTTP server (GET/POST/routes)"
    "agent_test.js|Agent loop (define/run/delegate/budgets/cancel)"
    "rlm_mock_test.js|RLM long-context (mock provider, chunk/recurse)"
    "web_test.js|Web search (mock DDG, tool end-to-end)"
    "live_provider_test.js|Live-provider E2E (skips unless SOFUU_LIVE_TEST=1 + key)"
)

# ── Helpers ───────────────────────────────────────────────────────
GREEN='\033[32m'
RED='\033[31m'
CYAN='\033[36m'
BOLD='\033[1m'
RESET='\033[0m'

PASS=0
FAIL=0
FAILED_TESTS=()

for entry in "${TESTS[@]}"; do
    file="${entry%%|*}"
    desc="${entry##*|}"
    path="$TESTS_DIR/$file"

    if [ ! -f "$path" ]; then
        echo -e "${RED}  MISSING: $file${RESET}"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$file (missing)")
        continue
    fi

    echo -e "\n${CYAN}--- $file: $desc ---${RESET}"

    if "$SOFUU" run "$path" 2>&1; then
        echo -e "${GREEN}✓ PASSED: $file${RESET}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}✗ FAILED: $file${RESET}"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$file")
    fi
done

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}=== JS Test Suite Results ===${RESET}"
echo -e "Passed: ${GREEN}$PASS${RESET}  Failed: ${RED}$FAIL${RESET}  Total: $((PASS + FAIL))"

if [ "$FAIL" -gt 0 ]; then
    echo -e "\n${RED}Failed tests:${RESET}"
    for t in "${FAILED_TESTS[@]}"; do
        echo "  - $t"
    done
    exit 1
fi

echo -e "\n${GREEN}✅ All JS tests passed!${RESET}"
