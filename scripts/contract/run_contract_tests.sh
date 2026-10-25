#!/usr/bin/env bash
# run_contract_tests.sh — conformance-suite skeleton (P0).
#
# Runs the 52 cases in cases.tsv against the built binary and reports
# PASS / FAIL / BLOCKED per case. This is deliberately a skeleton: in P0 most
# cases are BLOCKED on fixtures that do not exist yet, and the ones that can
# run are expected to FAIL where the contract is not implemented.
#
# Exit code: 0 only when every case PASSes. P0 is expected to exit non-zero.
#
# Usage:
#   bash scripts/contract/run_contract_tests.sh
#   CONTRACT_HTTP_URL=http://127.0.0.1:8080 \
#   CONTRACT_LIBSOFUU=dist/libsofuu.dylib \
#   SOFUU_BIN=target/release/sofuu \
#     bash scripts/contract/run_contract_tests.sh
#
# No dependency beyond bash, grep, tail. Cases never mutate the repository.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASES="$ROOT/scripts/contract/cases.tsv"
SOFUU="${SOFUU_BIN:-$ROOT/sofuu}"
HTTP_URL="${CONTRACT_HTTP_URL:-}"
LIBSOFUU="${CONTRACT_LIBSOFUU:-}"

if [ ! -x "$SOFUU" ]; then
    echo "✗ binary not executable: $SOFUU (build with make, or set SOFUU_BIN)" >&2
    exit 2
fi
if [ ! -f "$CASES" ]; then
    echo "✗ cases file missing: $CASES" >&2
    exit 2
fi

pass=0; fail=0; blocked=0
declare -a failures=() blockers=()

report_pass()  { pass=$((pass + 1));  printf 'PASS     %-6s %s\n' "$1" "$2"; }
report_fail()  { fail=$((fail + 1));  failures+=("$1: $2"); printf 'FAIL     %-6s %s\n' "$1" "$2"; }
report_block() { blocked=$((blocked + 1)); blockers+=("$1: $2"); printf 'BLOCKED  %-6s %s\n' "$1" "$2"; }

run_file_case() {
    local path="$1" expect="$2"
    if [ ! -f "$ROOT/$path" ]; then
        report_block "$id" "file not found: $path"
        return
    fi
    case "$expect" in
        has:*)
            if grep -qF -- "${expect#has:}" "$ROOT/$path"; then report_pass "$id" "$path contains '${expect#has:}'"
            else report_fail "$id" "$path does not contain '${expect#has:}'"; fi ;;
        hasnot:*)
            if grep -qF -- "${expect#hasnot:}" "$ROOT/$path"; then report_fail "$id" "$path still contains '${expect#hasnot:}'"
            else report_pass "$id" "$path has no '${expect#hasnot:}'"; fi ;;
        *) report_fail "$id" "bad file expectation: $expect" ;;
    esac
}

run_js_case() {
    local envspec="$1" expr="$2" expect="$3" out last
    if [ "$envspec" = "-" ]; then
        out="$("$SOFUU" eval "console.log($expr)" 2>&1)"
    else
        out="$(env "$envspec" "$SOFUU" eval "console.log($expr)" 2>&1)"
    fi
    last="$(printf '%s\n' "$out" | tail -n 1)"
    if [ "$last" = "$expect" ]; then
        report_pass "$id" "expr => $last"
    else
        report_fail "$id" "expected '$expect', got '$last'"
    fi
}

run_noqts_case() {
    local expr="$1" expect="$2" bin="${CONTRACT_NOQTSQ_BIN:-}" out last
    if [ -z "$bin" ] || [ ! -x "$bin" ]; then
        report_block "$id" "build a QTSQ-free binary (SOFUU_ALLOW_NO_QTSQ=1) and set CONTRACT_NOQTSQ_BIN"
        return
    fi
    out="$("$bin" eval "console.log($expr)" 2>&1)"
    last="$(printf '%s\n' "$out" | tail -n 1)"
    if [ "$last" = "$expect" ]; then
        report_pass "$id" "no-QTSQ binary: expr => $last"
    else
        report_fail "$id" "expected '$expect', got '$last'"
    fi
}

run_cli_case() {
    local envspec="$1" cmd="$2" expect="$3" out code
    if [ "$envspec" = "-" ]; then
        out="$(cd "$ROOT" && bash -c "$cmd" 2>&1)"; code=$?
    else
        out="$(cd "$ROOT" && env "$envspec" bash -c "$cmd" 2>&1)"; code=$?
    fi
    case "$expect" in
        exit:*)
            if [ "$code" = "${expect#exit:}" ]; then report_pass "$id" "exit $code"
            else report_fail "$id" "expected exit ${expect#exit:}, got $code"; fi ;;
        stdout_has:*)
            if printf '%s' "$out" | grep -qF -- "${expect#stdout_has:}"; then report_pass "$id" "stdout contains '${expect#stdout_has:}'"
            else report_fail "$id" "stdout missing '${expect#stdout_has:}'"; fi ;;
        stdout_absent:*)
            if printf '%s' "$out" | grep -qF -- "${expect#stdout_absent:}"; then report_fail "$id" "stdout still contains '${expect#stdout_absent:}'"
            else report_pass "$id" "stdout free of '${expect#stdout_absent:}'"; fi ;;
        *) report_fail "$id" "bad cli expectation: $expect" ;;
    esac
}

run_http_case() {
    local route="$1" expect="$2" method path code body
    if [ -z "$HTTP_URL" ]; then
        report_block "$id" "set CONTRACT_HTTP_URL to enable HTTP cases ($route)"
        return
    fi
    method="${route%% *}"; path="${route#* }"
    if [ "$method" = "GET" ]; then
        body="$(curl -sS -m 10 "$HTTP_URL$path" 2>&1)"; code=$?
    else
        body="$(curl -sS -m 10 -X "$method" -H 'Content-Type: application/json' \
              -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":false}' \
              "$HTTP_URL$path" 2>&1)"; code=$?
    fi
    if [ "$code" != "0" ]; then
        report_fail "$id" "curl failed: $body"
        return
    fi
    case "$expect" in
        200) if printf '%s' "$body" | grep -q '"'; then report_pass "$id" "route responded"
             else report_fail "$id" "route returned no JSON"; fi ;;
        sse) if printf '%s' "$body" | grep -q 'data:'; then report_pass "$id" "SSE frames present"
             else report_fail "$id" "no SSE frames"; fi ;;
        json) if printf '%s' "$body" | grep -q '"choices"'; then report_pass "$id" "OpenAI JSON shape"
              else report_fail "$id" "no choices object"; fi ;;
        *) report_fail "$id" "bad http expectation: $expect" ;;
    esac
}

run_cabi_case() {
    local symbol="$1" expect="$2"
    if [ -z "$LIBSOFUU" ] || [ ! -f "$ROOT/$LIBSOFUU" ]; then
        report_block "$id" "set CONTRACT_LIBSOFUU to a built library to enable C ABI cases"
        return
    fi
    case "$expect" in
        present)
            if nm -gU "$ROOT/$LIBSOFUU" 2>/dev/null | grep -q " _$symbol$"; then
                report_pass "$id" "symbol exported"
            else
                report_fail "$id" "symbol not exported"
            fi ;;
        *) report_fail "$id" "bad cabi expectation: $expect" ;;
    esac
}

while IFS='|' read -r id rule clause kind envspec cmd expect prereq; do
    [ -z "${id:-}" ] && continue
    case "$id" in \#*) continue ;; esac
    case "$kind" in
        file)   run_file_case "$cmd" "$expect" ;;
        js)     run_js_case "$envspec" "$cmd" "$expect" ;;
        noqts)  run_noqts_case "$cmd" "$expect" ;;
        cli)    run_cli_case "$envspec" "$cmd" "$expect" ;;
        http)   run_http_case "$cmd" "$expect" ;;
        cabi)   run_cabi_case "$cmd" "$expect" ;;
        manual) report_block "$id" "${prereq:-manual prerequisite missing}" ;;
        *)      report_fail "$id" "unknown case kind '$kind'" ;;
    esac
done < "$CASES"

total=$((pass + fail + blocked))
echo
printf 'summary: %d cases — PASS %d · FAIL %d · BLOCKED %d\n' "$total" "$pass" "$fail" "$blocked"
if [ "$fail" -gt 0 ]; then
    echo "contract violations:"
    for f in "${failures[@]}"; do echo "  - $f"; done
fi
if [ "$blocked" -gt 0 ]; then
    echo "blocked prerequisites:"
    for b in "${blockers[@]}"; do echo "  - $b"; done
fi
if [ "$pass" -eq "$total" ]; then
    echo "contract suite GREEN"
    exit 0
fi
echo "contract suite NOT green (expected in P0 — see contract/README.md)"
exit 1
