#!/usr/bin/env bash
# run_contract_tests.sh — conformance suite for the cases in cases.tsv.
#
# Runs every case against the built binary and reports PASS / FAIL / BLOCKED
# per case. A case either runs against a REAL local resource — the running
# binary, a real local MCP file server, a real local store, a real child
# process, or an existing local GGUF — or it BLOCKS with the reason. Nothing
# is answered from a substitute: a harness that reports "BLOCKED:" is
# recorded as a coverage gap, never as a pass, and a case that cannot run
# compliantly keeps the suite out of green.
#
# Every harness case declares the resource it uses ("// RESOURCE: real|pure:")
# and the case table must agree with that declaration; a case without one
# fails loudly instead of running unreviewed.
#
# Exit code: 0 only when every case PASSes (or when ALLOW_BLOCKED=1 covers
# blocked RUNNER prerequisites — never a real-execution coverage gap).
#
# Usage:
#   bash scripts/contract/run_contract_tests.sh
#   CONTRACT_HTTP_URL=http://127.0.0.1:8080 \
#   CONTRACT_LIBSOFUU=dist/libsofuu.dylib \
#   SOFUU_BIN=target/release/sofuu \
#   SOFUU_CONTRACT_MODEL=/path/to/existing/real-model.gguf \
#     bash scripts/contract/run_contract_tests.sh
#   SOFUU_CONTRACT_CASES=/absolute/path/to/cases.tsv  # optional subset
#
# SOFUU_CONTRACT_MODEL is exported to every case, so all model-dependent
# cases run against the SAME existing local weights. No weights are ever
# downloaded (AGENTS.md).
#
# No dependency beyond bash, grep, tail, sed. Cases never mutate the
# repository.

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
CASES="${SOFUU_CONTRACT_CASES:-$ROOT/scripts/contract/cases.tsv}"
SOFUU="${SOFUU_BIN:-$ROOT/sofuu}"
# A case that resolves its own binary must measure the SAME artifact this
# suite reports on. Without the export, text_gate.py fell back to a
# different binary than the suite's.
export SOFUU_BIN="$SOFUU"
HTTP_URL="${CONTRACT_HTTP_URL:-}"
if [ -z "${CONTRACT_LIBSOFUU:-}" ]; then
    for cand in dist/libsofuu.dylib dist/libsofuu.so dist/libsofuu.a; do
        if [ -f "$ROOT/$cand" ]; then CONTRACT_LIBSOFUU="$cand"; break; fi
    done
fi
LIBSOFUU="${CONTRACT_LIBSOFUU:-}"

# The model-dependent cases run against an existing local GGUF that the
# operator selects. The value is exported so every case sees the SAME file:
# a case must never quietly pick some other weights. An unset value leaves
# those cases BLOCKED — a visible coverage gap, never a silent pass — and a
# value that is not usable real weights is a broken prerequisite, so the
# suite stops before it can report anything green.
CONTRACT_MODEL="${SOFUU_CONTRACT_MODEL:-}"
if [ -n "$CONTRACT_MODEL" ]; then
    if [ ! -f "$CONTRACT_MODEL" ]; then
        echo "✗ SOFUU_CONTRACT_MODEL does not exist: $CONTRACT_MODEL" >&2
        exit 2
    fi
    case "$CONTRACT_MODEL" in
        *.gguf) ;;
        *) echo "✗ SOFUU_CONTRACT_MODEL must name a .gguf weights file: $CONTRACT_MODEL" >&2; exit 2 ;;
    esac
    case "$CONTRACT_MODEL" in
        *sofuu-fixture.gguf|*ggml-vocab-*.gguf)
            echo "✗ SOFUU_CONTRACT_MODEL must name existing real weights, not a parser fixture: $CONTRACT_MODEL" >&2
            exit 2 ;;
    esac
else
    echo "note: SOFUU_CONTRACT_MODEL is unset — the model-dependent cases report BLOCKED" >&2
fi
export SOFUU_CONTRACT_MODEL="$CONTRACT_MODEL"

if [ ! -x "$SOFUU" ]; then
    echo "✗ binary not executable: $SOFUU (build with make, or set SOFUU_BIN)" >&2
    exit 2
fi
if [ ! -f "$CASES" ]; then
    echo "✗ cases file missing: $CASES" >&2
    exit 2
fi

pass=0; fail=0; blocked=0; real_execution_blocked=0
declare -a failures=() blockers=()

# The runtime colours its stderr; the block record has to be recognisable
# whatever the colour codes around it.
strip_ansi() { sed $'s/\033\\[[0-9;]*m//g'; }

report_pass()  { pass=$((pass + 1));  printf 'PASS     %-6s %s\n' "$1" "$2"; }
report_fail()  { fail=$((fail + 1));  failures+=("$1: $2"); printf 'FAIL     %-6s %s\n' "$1" "$2"; }
report_block() { blocked=$((blocked + 1)); blockers+=("$1: $2"); printf 'BLOCKED  %-6s %s\n' "$1" "$2"; }

# Every harness case declares the resource it runs against, and the case
# table must agree with that declaration. A harness without one is a review
# gap, not a pass: the case fails loudly instead of running unreviewed.
check_case_resource() {
    local cmd="$1" prereq="$2" file want got path
    file="$(printf '%s\n' "$cmd" | sed -n 's|.*scripts/contract/harness/\([^ ]*\.js\).*|\1|p')"
    [ -n "$file" ] || return 0
    path="$ROOT/scripts/contract/harness/$file"
    [ -f "$path" ] || return 0
    got="$(grep -m1 -E '^// RESOURCE: (real|pure):' "$path" | sed -E 's#^// RESOURCE: (real|pure):.*#\1#')"
    if [ -z "$got" ]; then
        report_fail "$id" "harness $file declares no resource (// RESOURCE: real|pure:)"
        return 1
    fi
    want="$(printf '%s\n' "$prereq" | sed -nE 's/^(real|pure):.*/\1/p')"
    if [ -n "$want" ] && [ "$want" != "$got" ]; then
        report_fail "$id" "the case table says '$want' but $file declares '$got'"
        return 1
    fi
    return 0
}

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
    local envspec="$1" cmd="$2" expect="$3" out code out_tail blockline
    if [ "$envspec" = "-" ]; then
        out="$(cd "$ROOT" && bash -c "$cmd" 2>&1)"; code=$?
    else
        out="$(cd "$ROOT" && env "$envspec" bash -c "$cmd" 2>&1)"; code=$?
    fi
    # A failed CLI case must say WHICH assertion failed. The harness prints
    # its checks to the captured stream; keep a short tail in the FAIL line so
    # the CI gate's report is diagnosable without a local rerun.
    out_tail="$(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -n 6 | tr '\n' '~' | cut -c1-400)"
    # A harness that reports a block has NOT exercised its scenario. That
    # record wins over the exit status: a block must never be recorded as a
    # pass, and a block that does not use exit 2 is a broken harness, so it
    # fails loudly instead of passing the expectation.
    blockline="$(printf '%s\n' "$out" | strip_ansi | grep -m1 '^BLOCKED:' | cut -c1-300)"
    if [ -n "$blockline" ]; then
        if [ "$code" -ne 2 ]; then
            report_fail "$id" "the harness reported a block but exited $code instead of 2: $blockline"
        else
            report_block "$id" "$blockline"
            real_execution_blocked=$((real_execution_blocked + 1))
        fi
        return
    fi
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
    # A case that runs a harness must agree with that harness's own
    # declaration of the resource it uses (see check_case_resource). The
    # scripted-provider cases that used to live here were migrated to real
    # resources; a case without a declaration now fails loudly instead of
    # running unreviewed.
    case "$kind" in
        cli|js) check_case_resource "$cmd" "$prereq" || continue ;;
    esac
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
if [ "$fail" -gt 0 ]; then
    echo "contract suite FAILED ($fail failure(s))"
    exit 1
fi

if [ "$real_execution_blocked" -gt 0 ]; then
    echo "contract suite NOT green: $real_execution_blocked real-execution coverage gap(s) cannot be waived with ALLOW_BLOCKED"
    exit 1
fi

if [ "$pass" -eq "$total" ]; then
    echo "contract suite GREEN"
    exit 0
elif [ "${ALLOW_BLOCKED:-0}" = "1" ] && [ "$((pass + blocked))" -eq "$total" ]; then
    echo "contract suite GREEN (0 failures, $blocked blocked runner prerequisite(s) allowed)"
    exit 0
else
    echo "contract suite NOT green (blocked cases are recorded gaps, not passes — see contract/README.md)"
    exit 1
fi
