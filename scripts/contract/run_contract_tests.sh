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
# R-23: isolate every home-relative side effect. The runtime writes sessions,
# storage and brain keys under SOFUU_HOME; without this the suite left real
# `.sofuu/` directories inside the checkout and in $HOME.
CONTRACT_TMP="$(mktemp -d "${TMPDIR:-/tmp}/sofuu-contract.XXXXXX" 2>/dev/null || echo "")"
if [ -n "$CONTRACT_TMP" ]; then
    export SOFUU_HOME="${SOFUU_HOME:-$CONTRACT_TMP/home}"
    mkdir -p "$SOFUU_HOME"
    trap 'rm -rf "$CONTRACT_TMP"' EXIT INT TERM
fi
GIT_BEFORE="$(cd "$ROOT" && git status --porcelain 2>/dev/null | sort)"
CASES="$ROOT/scripts/contract/cases.tsv"
SOFUU="${SOFUU_BIN:-$ROOT/sofuu}"
HTTP_URL="${CONTRACT_HTTP_URL:-}"
if [ -z "${CONTRACT_LIBSOFUU:-}" ]; then
    for cand in dist/libsofuu.dylib dist/libsofuu.so dist/libsofuu.a; do
        if [ -f "$ROOT/$cand" ]; then CONTRACT_LIBSOFUU="$cand"; break; fi
    done
fi
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


run_cli_case() {
    local envspec="$1" cmd="$2" expect="$3" out code out_tail
    if [ "$envspec" = "-" ]; then
        out="$(cd "$ROOT" && bash -c "$cmd" 2>&1)"; code=$?
    else
        out="$(cd "$ROOT" && env "$envspec" bash -c "$cmd" 2>&1)"; code=$?
    fi
    # A failed CLI case must say WHICH assertion failed. The harness prints
    # its checks to the captured stream; keep a short tail in the FAIL line so
    # the CI gate's report is diagnosable without a local rerun.
    out_tail="$(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -n 6 | tr '\n' '~' | cut -c1-400)"
    case "$expect" in
        exit:*)
            if [ "$code" = "${expect#exit:}" ]; then report_pass "$id" "exit $code"
            else report_fail "$id" "expected exit ${expect#exit:}, got $code [out: $out_tail]"; fi ;;
        stdout_has:*)
            if printf '%s' "$out" | grep -qF -- "${expect#stdout_has:}"; then report_pass "$id" "stdout contains '${expect#stdout_has:}'"
            else report_fail "$id" "stdout missing '${expect#stdout_has:}' [out: $out_tail]"; fi ;;
        stdout_absent:*)
            if printf '%s' "$out" | grep -qF -- "${expect#stdout_absent:}"; then report_fail "$id" "stdout still contains '${expect#stdout_absent:}' [out: $out_tail]"
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

# symbol_exported <lib> <symbol> — is <symbol> in the DYNAMIC export table?
# Mach-O spells exports with a leading underscore and lists extern-defined
# symbols with `nm -gU`; ELF has no underscore and needs `nm -D --defined-only`
# (`-U` is Darwin-only; GNU nm rejects it). Some Linux hosts ship no binutils at
# all (the AWS test box has no `nm`), so the last resort is a dlopen probe
# through python3 — the same lookup a real host performs, i.e. stronger than nm.
symbol_exported() {
    local lib="$1" sym="$2"
    if command -v nm >/dev/null 2>&1; then
        case "$(uname -s)" in
            Darwin)
                nm -gU "$lib" 2>/dev/null | grep -qE "[[:space:]]_${sym}$" && return 0 ;;
            *)
                nm -D --defined-only "$lib" 2>/dev/null | grep -qE "[[:space:]]${sym}$" && return 0
                nm -g --defined-only "$lib" 2>/dev/null | grep -qE "[[:space:]]${sym}$" && return 0 ;;
        esac
    fi
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$lib" "$sym" >/dev/null 2>&1 <<'PY'
import ctypes, sys
getattr(ctypes.CDLL(sys.argv[1]), sys.argv[2])
PY
        return $?
    fi
    return 1
}

run_cabi_case() {
    local symbol="$1" expect="$2" lib="$LIBSOFUU"
    # CONTRACT_LIBSOFUU may be absolute (an out-of-tree build) or repo-relative.
    case "$lib" in /*) ;; *) lib="$ROOT/$lib" ;; esac
    if [ -z "$LIBSOFUU" ] || [ ! -f "$lib" ]; then
        report_block "$id" "set CONTRACT_LIBSOFUU to a built library to enable C ABI cases"
        return
    fi
    case "$expect" in
        present)
            if symbol_exported "$lib" "$symbol"; then
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
else
    echo "contract suite NOT green (expected in P0 — see contract/README.md)"
fi

# R-23 (AUDIT-2026-10-29): the header claimed "cases never mutate the
# repository", but the CLI/JS cases ran with cwd=$ROOT and no isolated home, so
# a suite run left `.sofuu/sessions/*` and `.sofuu/storage/*.qtsq` behind. The
# home is now a throwaway directory and this assertion makes the claim
# verifiable instead of aspirational: any tracked-file change fails the run.
if [ -n "${GIT_BEFORE:-}" ] || [ -d "$ROOT/.git" ]; then
    GIT_AFTER="$(cd "$ROOT" && git status --porcelain 2>/dev/null | sort)"
    if [ "$GIT_AFTER" != "${GIT_BEFORE:-}" ]; then
        echo "✗ the contract suite mutated the repository (R-23):"
        printf '%s\n' "$GIT_AFTER" | sed 's/^/    /'
        NEW_MUT=1
    fi
fi
[ "${NEW_MUT:-0}" = "1" ] && exit 1

if [ "$pass" -eq "$total" ]; then
    exit 0
fi
exit 1
