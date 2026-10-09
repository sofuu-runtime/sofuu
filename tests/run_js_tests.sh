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
    "ai_stream_error_frame_test.js|ai.stream error surfacing (in-stream error frame, 429, empty)"
    "fetch_test.js|fetch() Response API (status/headers/text/json)"
    "mcp_client_test.js|MCP client (connect/tools/call)"
    "mcp_server_test.js|MCP server (tool registration/handling)"
    "server_test.js|HTTP server (GET/POST/routes)"
    "agent_test.js|Agent loop (define/run/delegate/budgets/cancel)"
    "long_horizon_test.js|Long-horizon endurance (429 storm/mid-answer cuts/empty recovery/25-round loop/delegate budget/bash self-timing/wall honesty)"
    "rlm_mock_test.js|RLM long-context (mock provider, chunk/recurse)"
    "rlm_latch_test.js|js-10 concurrent-episode abort latch (first teardown must not clear the shared signal, last-out re-arms it)"
    "verify_memory_test.js|CMA brain suite (recall/dedup/entity/persist/decay/GMF/dream/resonance)"
    "multimodal_test.js|Multimodal e2e (embedImage + img1 store + text-query image recall)"
    "voice_test.js|ai.transcribe + ai.speak (multipart/bytes over local mock provider)"
    "caps_test.js|Context-window ladder (explicit /ctx obeys, inherited global shrinks, endpoint caps detected)"
    "compaction_guard_test.js|Compaction safety (per-pass cap, model-independent protected-turn floor)"
    "auto_continue_test.js|Step budget renews instead of stopping (long task finishes, bounded renewals, 0 disables)"
    "abi_script_test.js|ABI gate soundness (allowlist coverage, comm collation, set -u, baseline vs library)"
    "web_test.js|Web search (mock DDG, tool end-to-end)"
    "web_guard_test.js|SSRF guard host extraction (userinfo/IPv6/port, P1-15)"
    "fetch_deadline_test.js|js-8 per-request fetch timeoutMs (stalled endpoint dies at the caller deadline, fast endpoint unaffected)"
    "fetch_maxbody_test.js|js-9 per-request fetch maxBodyBytes (over-cap transfer aborts naming the cap, web.open transfer cap e2e)"
    "live_provider_test.js|Live-provider E2E (skips unless SOFUU_LIVE_TEST=1 + key)"
)

# ── Shell e2e suites (P1-16/P1-17, AUDIT-2026-09-01) ───────────────
# These drive the REAL chat/serve binaries end-to-end and assert on
# user-visible output; they run via bash, not `sofuu run`.
E2E_TESTS=(
    "brain_auth_e2e.sh|Brain-server auth (remote-bind guard, urandom token, 401s, Bearer 200)"
    "bundle_e2e.sh|Bundler re-exports (P1-13: export * as ns SyntaxError + missing re-export-target module)"
    "chat_no_response_e2e.sh|Chat failure modes (429/errframe/empty/silent/loop/close never (no response))"
    "chat_paste_e2e.sh|Bracketed-paste input (30KB paste waits for Enter, lands whole, typed input intact)"
    "chat_copy_e2e.sh|Mouse-drag selection + Ctrl-K copy (inverse-video paint, clipboard holds clean text)"
    "chat_chip_e2e.sh|Usage-chip persistence (in→out tk survives the 2s timer, reflects the request's real ptk)"
    "chat_models_e2e.sh|Model picker e2e (/model lists ALL live models per provider, stored models survive an unreachable provider, typed manual model persists, live listing cached)"
    "chat_sessions_e2e.sh|Sessions browser e2e (/sessions opens the keyboard picker titled with its project, arrows move, Enter resumes)"
    "chat_resume_replay_e2e.sh|Resume replay e2e (/resume prints the previous turns, not just a notice)"
    "chat_transcript_e2e.sh|Transcript hierarchy e2e (markdown levels render, blank row opens tool groups, live todo checklist replaces the one-row summary)"
    "chat_theme_e2e.sh|Theme e2e (25 dark+light themes listed, slate applies live to transcript+panel, persists, unknown rejected)"
    "chat_mode_no_wipe_e2e.sh|Settings-change e2e (/mode mid-conversation keeps the transcript — panel logged once, not re-logged)"
    "chat_empty_session_e2e.sh|Empty-session e2e (open+quit with no turns saves nothing; one turn is kept once, ended)"
    "chat_compact_role_e2e.sh|Strict-gateway compact e2e (summarize payload ends with role=user — a 400-on-assistant-last mock compacts instead of [Compact failed])"
    "chat_retention_e2e.sh|Claude-style transcript retention (tool transcript persists into history, no orphaned tool messages)"
    "chat_modes_e2e.sh|Permission modes (welcome Mode row, /mode /plan /edit /full, footer chip, bad arg refused, persists + boot-applied across restart)"
    "chat_prompt_dedupe_e2e.sh|Prompt-event dedupe (one prompt per turn across THINK/CAPACITY retries, both drivers, no over-suppression)"
    "chat_remember_e2e.sh|/remember brain-path fix (pin lands project-local, HOME file never created, cross-session recall injected into the request, brain_path override honored)"
    "chat_nothink_failover_e2e.sh|js-2 noThink-retry failover (the retry's own 429 re-enters the failover chain instead of killing the turn, done event still fires)"
    "chat_approval_timer_e2e.sh|js-3 approval ceiling-timer leak (resolved approvals clearTimeout, 1120 gated calls past the 1024-timer registry cap no longer fill it — zero tool errors, all files written)"
    "chat_state_copy_e2e.sh|js-4 state({history:true}) live tool_calls reference (deep copy — a host vandalizing the returned history cannot corrupt the next turn's request)"
    "chat_secrets_guard_e2e.sh|js-5 grep/glob sensitive-file guard (canary config/.ssh/.pem fixtures never reach the mock transcript via auto-approved scans; read_file control still works)"
    "tools_edit_literal_e2e.sh|js-6 edit_file literal splice (replacement-pattern sequences in new_string land literally: single-match, replace_all, zero-match throw; scratch-project fixtures cleaned up)"
    "tools_diff_e2e.sh|TUI edit/write diff (bounded unified diff stashed per call, return strings unchanged, agent event carries diff, errors carry no diff)"
)

# ── Helpers ───────────────────────────────────────────────────────
GREEN='\033[32m'
RED='\033[31m'
CYAN='\033[36m'
BOLD='\033[1m'
RESET='\033[0m'

PASS=0
FAIL=0
SKIP=0
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

    # (set -e-safe: the if consumes the exit code; 77 = opt-in skip)
    rc=0
    if [ "$file" = "caps_test.js" ]; then
        # The caps ladder test feeds SYNTHETIC model listings into the
        # discovered-caps store, which persists to $HOME/.sofuu/ml. Point
        # HOME at a scratch dir so a test run can never pollute (or race)
        # the user's real harvested caps.
        CAPS_HOME="$(mktemp -d)"
        HOME="$CAPS_HOME" "$SOFUU" run "$path" 2>&1 || rc=$?
        rm -rf "$CAPS_HOME"
    else
        "$SOFUU" run "$path" 2>&1 || rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED: $file${RESET}"
        PASS=$((PASS + 1))
    elif [ "$rc" -eq 77 ]; then
        # 77 = the test itself chose to skip (opt-in gate not met). Counted
        # honestly as SKIP, never as PASSED (P2-26, AUDIT-2026-09-01).
        echo -e "${CYAN}⊘ SKIPPED: $file${RESET}"
        SKIP=$((SKIP + 1))
    else
        echo -e "${RED}✗ FAILED: $file (exit $rc)${RESET}"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$file")
    fi
done

# Shell e2e suites — same accounting, invoked with bash.
for entry in "${E2E_TESTS[@]}"; do
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

    rc=0
    bash "$path" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED: $file${RESET}"
        PASS=$((PASS + 1))
    elif [ "$rc" -eq 77 ]; then
        # 77 = the test itself chose to skip (opt-in gate not met, e.g. no
        # QTSQ codec so sofuu.memory does not exist). Counted honestly as
        # SKIP, never as PASSED — same convention as the .js suites.
        echo -e "${CYAN}⊘ SKIPPED: $file${RESET}"
        SKIP=$((SKIP + 1))
    else
        echo -e "${RED}✗ FAILED: $file (exit $rc)${RESET}"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$file")
    fi
done

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}=== JS Test Suite Results ===${RESET}"
echo -e "Passed: ${GREEN}$PASS${RESET}  Failed: ${RED}$FAIL${RESET}  Skipped: ${CYAN}$SKIP${RESET}  Total: $((PASS + FAIL + SKIP))"

if [ "$FAIL" -gt 0 ]; then
    echo -e "\n${RED}Failed tests:${RESET}"
    for t in "${FAILED_TESTS[@]}"; do
        echo "  - $t"
    done
    exit 1
fi

echo -e "\n${GREEN}✅ All JS tests passed!${RESET}"
