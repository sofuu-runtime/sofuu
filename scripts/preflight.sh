#!/usr/bin/env bash
# scripts/preflight.sh — the repository's gate runner (RULES §3.4, §1.9).
#
# Every gate prints PASS / FAIL / BLOCKED and the reason it reached that
# verdict. A blocked gate is a coverage gap, never a pass: this runner never
# substitutes a model, a tool, a store or a sink for the real one, and it
# never reports a skipped check as a successful one.
#
# Sections (all of them by default; name one or more to run a subset):
#
#   isolation  domain boundary isolation: no cross-imports between the domain
#              capsules under crates/sofuu-core/src/domains (RULES §3.4)
#   bytecode   shipped QuickJS snapshots match their sources (make bc-check)
#   drift      Kaze product surface vs the prototype copy (make kaze-drift-check)
#   size       release binary within SIZE_LIMIT (make size-check)
#   rust       core unit tests; SOFUU_RUST_FILTER narrows the test names
#   real       real-resource gates: real MCP, real memory, Kaze memory, real
#              alert sinks, real local-GGUF inference, the event-loop benchmark
#   serve      serve acceptance: names the scenarios that are blocked
#   js         the local-only JS corpus: names the scenarios that are blocked
#
# Exit code: 0 when every gate that ran passed, 1 when a gate failed, 2 when
# the only outstanding outcome is a blocked prerequisite. ALLOW_BLOCKED=1
# downgrades BLOCKED to a warning for a runner that deliberately lacks a
# resource; it never turns a FAIL into a pass and never substitutes a backend.
#
# Usage:
#   bash scripts/preflight.sh
#   bash scripts/preflight.sh real serve
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

pass=0
fail=0
blocked=0
declare -a failures=()
declare -a blockers=()

report_pass()  { pass=$((pass + 1));   printf 'PASS     %-22s %s\n' "$1" "$2"; }
report_fail()  { fail=$((fail + 1));   failures+=("$1: $2");  printf 'FAIL     %-22s %s\n' "$1" "$2"; }
report_block() { blocked=$((blocked + 1)); blockers+=("$1: $2"); printf 'BLOCKED  %-22s %s\n' "$1" "$2"; }
detail()       { printf '         %s\n' "$*"; }

# Strip the ANSI colours the Makefile gates print.
plain() { sed -E $'s/\033\\[[0-9;]*m//g'; }

# ── isolation ───────────────────────────────────────────────────────────────
# RULES §3.4: the domain capsules under crates/sofuu-core/src/domains must
# not import one another. This half of the preflight had not been re-run
# since the asynchronous Kaze bridge landed, so it runs here as its own
# gate rather than as a silent side effect of another step.
section_isolation() {
    local id=isolation
    local domains="crates/sofuu-core/src/domains"
    local pair capsule others hits violations=0
    if [ ! -d "$domains" ]; then
        report_block "$id" "$domains is absent — the domain capsules are not in this tree"
        return
    fi
    for pair in "kaze:robotics|infer" "robotics:kaze|infer" "memory:kaze|robotics" "infer:kaze|robotics"; do
        capsule="${pair%%:*}"
        others="${pair##*:}"
        if [ ! -d "$domains/$capsule" ]; then
            report_block "$id" "capsule $domains/$capsule is absent — nothing to isolate"
            return
        fi
        # Any mention of another capsule's path counts, not just a `use`
        # statement: an inline path is a coupling too.
        hits=$(grep -rnE "crate::domains::($others)" "$domains/$capsule" 2>/dev/null)
        if [ -n "$hits" ]; then
            violations=$((violations + 1))
            printf '%s\n' "$hits" | while IFS= read -r line; do detail "$line"; done
        fi
    done
    if [ "$violations" -gt 0 ]; then
        report_fail "$id" "$violations capsule(s) import another domain (RULES §3.4)"
    else
        report_pass "$id" "no cross-domain imports between kaze, robotics, memory and infer"
    fi
}

# ── bytecode ───────────────────────────────────────────────────────────────
section_bytecode() {
    local id=bytecode out rc summary
    out=$(make bc-check 2>&1); rc=$?
    summary=$(printf '%s\n' "$out" | grep -E 'BC-CHECK|MISSING|STALE|COMPILE' | tail -2 | plain | tr '\n' ' ')
    if [ "$rc" -eq 0 ]; then
        report_pass "$id" "${summary:-bytecode snapshots match their sources}"
    elif printf '%s\n' "$out" | grep -qE 'SOFUU_QTSQ_DIR is not set|no libqtsq'; then
        report_block "$id" "the release build cannot run here (no QTSQ checkout) — the shipped snapshots were NOT verified"
    elif printf '%s\n' "$out" | grep -qE '^error(\[|:)|could not compile'; then
        report_fail "$id" "the release build did not compile, so the shipped snapshots were not checked — $(printf '%s\n' "$out" | grep -E '^error(\[|:)|could not compile' | head -2 | plain | tr '\n' ' ')"
    else
        report_fail "$id" "${summary:-make bc-check exited $rc}"
    fi
}

# ── kaze drift ─────────────────────────────────────────────────────────────
section_drift() {
    local id=drift out rc
    out=$(make kaze-drift-check 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        if printf '%s\n' "$out" | grep -q 'nothing to compare'; then
            report_block "$id" "the Kaze prototype copy is gitignored and absent — the drift gate has nothing to compare"
        else
            report_pass "$id" "the Kaze product surface has not fallen behind the prototype"
        fi
    else
        report_fail "$id" "the Kaze product surface and the prototype have drifted"
        printf '%s\n' "$out" | grep -E '^(FAIL|  (verb|opcode) only)' | head -12 | while IFS= read -r line; do detail "$line"; done
    fi
}

# ── binary size ────────────────────────────────────────────────────────────
section_size() {
    local id=size out rc line
    out=$(make size-check 2>&1); rc=$?
    line=$(printf '%s\n' "$out" | grep -E 'Size [0-9]+ bytes' | tail -1 | plain)
    if [ "$rc" -eq 0 ]; then
        report_pass "$id" "${line:-the release binary is within the size cap}"
    elif printf '%s\n' "$out" | grep -qE 'SOFUU_QTSQ_DIR is not set|no libqtsq'; then
        report_block "$id" "the release build cannot run here (no QTSQ checkout) — the size cap was NOT measured"
    elif printf '%s\n' "$out" | grep -qE '^error(\[|:)|could not compile'; then
        report_fail "$id" "the release build did not compile, so the size cap was not measured — $(printf '%s\n' "$out" | grep -E '^error(\[|:)|could not compile' | head -2 | plain | tr '\n' ' ')"
    else
        report_fail "$id" "${line:-make size-check exited $rc} — RULES §3.1 caps the release binary at 5767168 bytes"
    fi
}

# ── core unit tests ────────────────────────────────────────────────────────
# These tests drive the RLM episode loop with a canned model response. A
# scripted model is a simulated AI response (RULES §1.9), so they are not
# executed here; their names are reported so the gap stays visible instead of
# disappearing into a green run.
#
# The list is derived from the source instead of being written out here: a
# test that calls the episode runner with a closure is a scripted-model test,
# whatever it is called today. A hardcoded name list went stale within a day
# of being written.
scripted_model_tests() {
    python3 - "$ROOT/crates/sofuu-core/src" <<'PY'
import pathlib, re, sys

# A test is scripted-model only when it drives a SUBSTITUTE PROVIDER: a
# canned model answer that impersonates the backend. Feeding fixed replies
# into a pure state machine is a static-input test (RULES §1.9 rule 2) and
# must keep running — skipping it would drop real coverage, not protect it.
SUBSTITUTE = re.compile(
    r'provider\s*[:=]\s*"?mock'
    r'|"model"\s*:\s*"mock'
    r'|\bmock-stream\b|\bmock-tool\b|\bmock-ap\b'
    r'|\bfake_stream\b|\bfakeStream\b'
    r'|\bscripted_model\b|\bfake model\b|\bmock model\b'
    r'|\bsetProvider\b|\binstall_mock\b',
    re.I)

root = pathlib.Path(sys.argv[1])
for path in sorted(root.rglob("*.rs")):
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in re.finditer(r"#\[test\]\s*\n\s*fn ([a-z0-9_]+)\(\)\s*\{", text):
        start = match.end()
        end = text.find("\n    }", start)
        body = text[start:end if end != -1 else len(text)]
        if SUBSTITUTE.search(body):
            module = path.relative_to(root).with_suffix("").as_posix().replace("/", "::")
            print("%s::tests::%s" % (module, match.group(1)))
PY
}

section_rust() {
    local id=rust out rc summary t filter="${SOFUU_RUST_FILTER:-}"
    local -a skip=() selection=()
    local scripted
    scripted=$(scripted_model_tests)
    while IFS= read -r t; do
        [ -n "$t" ] || continue
        if [ -n "$filter" ]; then
            case "$t" in *"$filter"*) ;; *) continue ;; esac
        fi
        skip+=(--skip "$t")
    done <<< "$scripted"
    [ -z "$filter" ] || selection+=("$filter")
    detail "Rust scope: ${filter:-all sofuu-core library tests}; integration, release and other areas are not included"
    if [ "${#skip[@]}" -gt 0 ]; then
        out=$(SOFUU_ALLOW_LOOPBACK=1 cargo test --offline -p sofuu-core --lib "${selection[@]}" -- "${skip[@]}" 2>&1); rc=$?
    else
        out=$(SOFUU_ALLOW_LOOPBACK=1 cargo test --offline -p sofuu-core --lib "${selection[@]}" 2>&1); rc=$?
    fi
    summary=$(printf '%s\n' "$out" | grep -E '^test result:' | tail -1)
    if [ "$rc" -eq 0 ]; then
        if [ -z "$summary" ] || printf '%s\n' "$summary" | grep -q '^test result: ok\. 0 passed; 0 failed;'; then
            report_block "$id" "no passing tests in Rust scope '${filter:-all}'; check the filter and ignored tests"
        else
            report_pass "$id" "selected core tests — $summary"
        fi
        if [ "${#skip[@]}" -gt 0 ]; then
            detail "not executed (scripted model response, RULES §1.9):"
            for ((t=1; t<${#skip[@]}; t+=2)); do detail "  ${skip[$t]}"; done
            report_block "$id" "$((${#skip[@]} / 2)) selected scripted-model test(s) are not executed; real-backend coverage is still needed"
        fi
    else
        report_fail "$id" "${summary:-cargo test exited $rc}"
        printf '%s\n' "$out" | grep -E '^(test .*FAILED|failures:|error)' | head -20 | while IFS= read -r line; do detail "$line"; done
    fi
}

# ── real-resource gates ────────────────────────────────────────────────────
# Every check below runs against a real backend: a real MCP server reading a
# real file, a real QTSQ store, a real CMA store, a real alert sink, or real
# local GGUF weights. A missing prerequisite is reported BLOCKED; it is never
# replaced by a substitute.
# run_bounded <seconds> <command...> — run a command with a wall-clock
# budget. 124 means the budget expired. A hung case must not hold the gate.
run_bounded() {
    local secs="$1"; shift
    "$@" & local pid=$!
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$secs" ]; do
        sleep 5
        waited=$((waited + 5))
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null
        sleep 2
        kill -KILL "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        return 124
    fi
    wait "$pid"
}

run_check() {
    local id="$1" desc="$2"
    shift 2
    local out rc
    out=$("$@" 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        report_pass "$id" "$desc"
    elif [ "$rc" -eq 2 ] && printf '%s\n' "$out" | grep -q 'BLOCKED'; then
        report_block "$id" "$desc — $(printf '%s\n' "$out" | grep -m1 'BLOCKED' | cut -c1-200)"
    else
        report_fail "$id" "$desc — exit $rc: $(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -3 | tr '\n' ' ' | cut -c1-260)"
    fi
}

# Why an attempt did not ground, in the words of the check itself.
gguf_failure_reason() {
    local out="$1"
    case "$out" in
        *"timed out"*) echo "the run timed out (the host is loaded or the path hangs)"; return ;;
        *E_TEST_MARKER_ABSENT*) echo "the model's answer did not contain the tool's marker"; return ;;
        *E_TEST_NO_REPORT*) echo "the Kaze bridge returned no report"; return ;;
        *E_TEST_POLICY*) echo "the policy did not load"; return ;;
    esac
    printf '%s\n' "$out" | grep -m1 -E 'AssertionError|FAIL' | cut -c1-160
}

# The pure-Kaze milestone needs a real model to answer with the marker's tool
# result in it. The only local weights on this machine are a 360M model, whose
# instruction-following is stochastic (measured 10 of 14 runs grounded), so the
# gate runs several attempts: it fails when NO attempt grounds (the path is
# broken) and reports the grounded count when it does, because a single
# grounded run already proves the real tool result reached a real model.
gguf_check() {
    local attempts="${SOFUU_GGUF_ATTEMPTS:-3}"
    local i ok=0 out rc
    for ((i = 1; i <= attempts; i++)); do
        out=$(python3 "$ROOT/scripts/test-kaze-local-gguf-real.py" 2>&1); rc=$?
        if [ "$rc" -eq 0 ]; then
            ok=$((ok + 1))
            detail "attempt $i/$attempts: grounded"
        elif [ "$rc" -eq 2 ] && printf '%s\n' "$out" | grep -q 'BLOCKED'; then
            report_block "kaze-local-gguf" "pure-Kaze real inference over local GGUF weights — $(printf '%s\n' "$out" | grep -m1 'BLOCKED' | cut -c1-200)"
            return
        else
            detail "attempt $i/$attempts: not grounded — $(gguf_failure_reason "$out")"
        fi
    done
    if [ "$ok" -eq 0 ]; then
        report_fail "kaze-local-gguf" "the local model never grounded the tool result — 0/$attempts independent runs"
    elif [ "$ok" -eq "$attempts" ]; then
        report_pass "kaze-local-gguf" "pure-Kaze real inference over local GGUF weights with real MCP grounding — $ok/$attempts independent runs"
    else
        report_pass "kaze-local-gguf" "pure-Kaze real inference over local GGUF weights with real MCP grounding — $ok/$attempts independent runs grounded (the local model is stochastic)"
    fi
}

# A vocabulary-only or synthetic GGUF is a parser fixture, not inference
# evidence (RULES §1.9, AGENTS.md). A gate that accepts one would report
# coverage it does not have.
is_real_weight_model() {
    case "$1" in
        */sofuu-fixture.gguf|sofuu-fixture.gguf) return 1 ;;
        */ggml-vocab-*.gguf|ggml-vocab-*.gguf) return 1 ;;
        */contract/fixtures/*|contract/fixtures/*) return 1 ;;
    esac
    return 0
}

# The event-loop benchmark serves a real local model over loopback. Without a
# real-weight GGUF it is blocked, never measured against a substitute stream.
event_loop_check() {
    local out rc
    if [ -z "${SOFUU_LOOP_MODEL:-}" ]; then
        report_block "event-loop" "event-loop latency during real local inference — SOFUU_LOOP_MODEL is not set to an existing real-weight GGUF"
        return
    fi
    if ! is_real_weight_model "$SOFUU_LOOP_MODEL"; then
        report_block "event-loop" "event-loop latency during real local inference — $SOFUU_LOOP_MODEL is a vocabulary-only or synthetic fixture, not real weights"
        return
    fi
    out=$(SOFUU_ALLOW_LOOPBACK=1 "$SOFUU_BIN" run scripts/bench/event_loop_latency.js 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        report_pass "event-loop" "event-loop latency during real local inference — $(printf '%s\n' "$out" | grep -iE 'max.*delay|budget' | tail -1 | cut -c1-160)"
    elif [ "$rc" -eq 2 ] && printf '%s\n' "$out" | grep -q 'BLOCKED'; then
        report_block "event-loop" "event-loop latency during real local inference — $(printf '%s\n' "$out" | grep -m1 'BLOCKED' | cut -c1-200)"
    else
        report_fail "event-loop" "event-loop latency during real local inference — exit $rc: $(printf '%s\n' "$out" | tail -3 | tr '\n' ' ' | cut -c1-260)"
    fi
}

section_real() {
    local id=real bin="" cand
    if [ -n "${SOFUU_BIN:-}" ]; then
        bin="$SOFUU_BIN"
    else
        for cand in "$ROOT/target/debug/sofuu" "$ROOT/target/release/sofuu" "$ROOT/sofuu"; do
            if [ -x "$cand" ]; then bin="$cand"; break; fi
        done
    fi
    if [ -z "$bin" ] || [ ! -x "$bin" ]; then
        report_block "$id" "no built Sofuu binary to exercise — build one (make) or set SOFUU_BIN"
        return
    fi
    export SOFUU_BIN="$bin"
    local digest=""
    if command -v shasum >/dev/null 2>&1; then
        digest=$(shasum -a 256 "$bin" 2>/dev/null | cut -d' ' -f1)
    elif command -v sha256sum >/dev/null 2>&1; then
        digest=$(sha256sum "$bin" 2>/dev/null | cut -d' ' -f1)
    fi
    detail "artifact: $bin${digest:+ (sha256 $digest)}"

    # A QTSQ-free build cannot open a real store at all; that is a missing
    # resource, not a failing gate.
    local qtsq
    qtsq=$("$bin" capabilities 2>/dev/null | python3 -c 'import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("unknown"); raise SystemExit
missing = d.get("build", {}).get("missing") or []
print("absent" if "qtsq" in missing else "present")' 2>/dev/null)

    if [ "$qtsq" = "absent" ]; then
        report_block "real-memory" "real QTSQ store: write, separate-process reopen, isolation, readonly and flush failure — this build is QTSQ-free"
    else
        run_check "real-memory" "real QTSQ store: write, separate-process reopen, isolation, readonly and flush failure" \
            python3 "$ROOT/scripts/test-real-memory.py"
    fi

    run_check "real-mcp" "real stdio + HTTP MCP server reading a real temporary file" \
        python3 "$ROOT/scripts/test-real-mcp-file.py"
    run_check "kaze-memory" "Kaze per-run CMA store: isolation, policy, limits, source, bytecode, JS, standalone" \
        python3 "$ROOT/scripts/test-kaze-memory-real.py" "$bin"
    run_check "kaze-memory-durable" "durable Kaze memory: write, process reopen, isolation, bytecode, standalone" \
        python3 "$ROOT/scripts/test-kaze-memory-durable.py" "$bin"
    run_check "kaze-alert" "real alert sinks (stdout, stderr, file): delivery, policy denial, fail-closed" \
        python3 "$ROOT/scripts/test-kaze-alert-real.py" "$bin"
    run_check "kaze-fail-closed" "Kaze gates through real local execution surfaces (no model, no device)" \
        python3 "$ROOT/scripts/test-kaze-fail-closed.py" "$bin"
    gguf_check
    event_loop_check
}

# ── serve acceptance ───────────────────────────────────────────────────────
# tests/serve_test_run.sh starts a REAL `sofuu serve` on loopback with
# existing local weights and drives it over real HTTP: the model list, a
# streamed SSE completion, a non-streaming completion, real error bodies and
# a real client disconnect. The previous half only analysed the stale mock
# probe, so the gate could not see a green or a red run.
section_serve() {
    local id=serve runner="$ROOT/tests/serve_test_run.sh"
    if [ ! -f "$runner" ]; then
        report_block "$id" "the serve acceptance corpus is local-only (tests/ is gitignored) — a clean checkout has neither the runner nor the probe"
        return
    fi
    local bin="" cand
    if [ -n "${SOFUU_BIN:-}" ]; then
        bin="$SOFUU_BIN"
    else
        for cand in "$ROOT/target/debug/sofuu" "$ROOT/target/release/sofuu" "$ROOT/sofuu"; do
            if [ -x "$cand" ]; then bin="$cand"; break; fi
        done
    fi
    if [ -z "$bin" ] || [ ! -x "$bin" ]; then
        report_block "$id" "no built Sofuu binary to exercise — build one (make) or set SOFUU_BIN"
        return
    fi
    local out rc
    out="$(SOFUU="$bin" bash "$runner" 2>&1)"; rc=$?
    printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -12 | while IFS= read -r line; do detail "$line"; done
    case "$rc" in
        0) report_pass "$id" "real serve acceptance over loopback: model list, SSE stream, non-streaming completion, error bodies, client disconnect" ;;
        2) report_block "$id" "real serve acceptance — $(printf '%s\n' "$out" | grep -m1 'BLOCKED' | cut -c1-200)" ;;
        3) report_block "$id" "real serve acceptance ran against the real backend; one scenario is unverified — $(printf '%s\n' "$out" | grep -i -m1 'unverified' | cut -c1-200)" ;;
        *) report_fail "$id" "real serve acceptance — exit $rc: $(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -3 | tr '\n' ' ' | cut -c1-260)" ;;
    esac
}

# ── local-only JS corpus ───────────────────────────────────────────────────
# The corpus runs: the provider-backed cases resolve an existing local GGUF
# and the scripted ones are reported as coverage gaps by the runner itself,
# never as passes. It is bounded so a hung case cannot hold the gate.
section_js() {
    local id=js runner="$ROOT/tests/run_js_tests.sh"
    if [ ! -f "$runner" ]; then
        report_block "$id" "the JS corpus is local-only (tests/ is gitignored) — a clean checkout has no runner, so the aggregate cannot run there"
        return
    fi
    local out rc
    out="$(run_bounded 2700 bash "$runner" 2>&1)"; rc=$?
    printf '%s\n' "$out" | grep -E '^(Passed:|.*(BLOCKED|FAILED): )' | tail -40 \
        | while IFS= read -r line; do detail "$line"; done
    case "$rc" in
        0) report_pass "$id" "the JS corpus ran against real backends" ;;
        2) report_block "$id" "the JS corpus ran; the cases listed above are coverage gaps (scripted-provider substitutes or absent real resources), not passes" ;;
        124) report_fail "$id" "the JS corpus did not finish inside its 45-minute budget" ;;
        *) report_fail "$id" "the JS corpus failed — exit $rc: $(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -3 | tr '\n' ' ' | cut -c1-260)" ;;
    esac
}

# ── runner ─────────────────────────────────────────────────────────────────
sections=("$@")
if [ "${#sections[@]}" -eq 0 ]; then
    sections=(isolation bytecode drift size rust real serve js)
fi

for s in "${sections[@]}"; do
    case "$s" in
        isolation) section_isolation ;;
        bytecode)  section_bytecode ;;
        drift)     section_drift ;;
        size)      section_size ;;
        rust)      section_rust ;;
        real)      section_real ;;
        serve)     section_serve ;;
        js)        section_js ;;
        *)         echo "unknown section: $s" >&2; exit 2 ;;
    esac
done

echo
printf 'preflight: %d check(s) — PASS %d · FAIL %d · BLOCKED %d\n' \
    "$((pass + fail + blocked))" "$pass" "$fail" "$blocked"
if [ "$fail" -gt 0 ]; then
    echo "failed checks:"
    for f in "${failures[@]}"; do echo "  - $f"; done
fi
if [ "$blocked" -gt 0 ]; then
    echo "blocked checks (coverage gaps, not passes):"
    for b in "${blockers[@]}"; do echo "  - $b"; done
fi

if [ "$fail" -gt 0 ]; then
    echo "preflight FAILED ($fail failure(s))"
    exit 1
fi
if [ "$blocked" -gt 0 ]; then
    if [ "${ALLOW_BLOCKED:-0}" = "1" ]; then
        echo "preflight: $blocked blocked check(s) allowed by ALLOW_BLOCKED=1 — the coverage gaps above are NOT closed"
        exit 0
    fi
    echo "preflight NOT GREEN ($blocked blocked check(s))"
    exit 2
fi
if [ "${#sections[@]}" -eq 1 ] && [ "${sections[0]}" = rust ] && [ -n "${SOFUU_RUST_FILTER:-}" ]; then
    echo "preflight GREEN for the selected Rust scope only (not full preflight)"
else
    echo "preflight GREEN for the selected sections"
fi
exit 0
