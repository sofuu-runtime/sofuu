#!/usr/bin/env bash
# embedded-kaze-c-gates.sh — Plan 14 acceptance gates.
#
# The four gates from docs/kaze-plan/14-EMBEDDED-KAZE-IN-C-AND-KAZE.md, plus
# the ABI/snapshot check that makes the C library reproducible:
#
#   1. Behavioral regression: unit, integration, robotics and JS-boundary tests.
#   2. Crash immunity: hostile inputs through the C VM under AddressSanitizer.
#   3. Bytecode parity: Formats 1, 2, 4 and 5 load with the right diagnostics.
#   4. Rust elimination: domains/kaze_engine/ and all 12 duplicated engine
#      modules are retired; domains/kaze/ holds only caps/FFI/JS-bridge glue.
#   5. ABI/snapshot: the header exports the promised surface, the active
#      embedded compiler path has no process exit, and deps/kaze/ matches
#      kaze/dist/ byte-for-byte.
#
#   make embedded-kaze-gates
#   scripts/embedded-kaze-c-gates.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CC="${CC:-cc}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/embedded-kaze-gates.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
SKIP=0

section() { printf '\n== %s ==\n' "$1"; }
ok()      { printf 'PASS  %s\n' "$1"; PASS=$((PASS + 1)); }
bad()     { printf 'FAIL  %s\n' "$1"; FAIL=$((FAIL + 1)); }
skip()    { printf 'SKIP  %s\n' "$1"; SKIP=$((SKIP + 1)); }

# Run a command, keep its output out of the way on success, and show the tail
# on failure. The tests are useful when they fail, not when they pass.
run_quiet() {
    local label="$1"; shift
    local log="$TMP/$(printf '%s' "$label" | tr ' /' '__').log"
    if "$@" >"$log" 2>&1; then
        ok "$label"
    else
        bad "$label"
        tail -20 "$log" | sed 's/^/      /'
    fi
}

KAZE="$ROOT/kaze/bin/kaze"
run_quiet "build standalone kaze CLI" make kaze-build

section "Gate 1 — behavioral regression"
run_quiet "unit: domains::kaze" cargo test -p sofuu-core --lib domains::kaze
run_quiet "unit: rt::kaze_memory" cargo test -p sofuu-core --lib kaze_memory
run_quiet "unit: robotics::hardware" cargo test -p sofuu-core --lib robotics::hardware
run_quiet "unit: sofuu-ffi kaze ABI" cargo test -p sofuu-ffi --lib kaze
run_quiet "unit: robotics kaze_live" cargo test -p sofuu-core --lib kaze_live
run_quiet "integration: kaze_language" cargo test -p sofuu-core --test kaze_language
run_quiet "integration: kaze_live" cargo test -p sofuu-core --test kaze_live
run_quiet "integration: diagnostics_contract" cargo test -p sofuu-core --test diagnostics_contract
run_quiet "integration: parser_nesting" cargo test -p sofuu-core --test parser_nesting
run_quiet "drift: surface + vendored C snapshot" make kaze-drift-check
run_quiet "JS boundary and argument isolation" make kaze-js-boundary-check

# The gate needs a real binary. Build it even when one exists: a stale
# `target/debug/sofuu` would test the previous engine, not this tree.
run_quiet "build sofuu binary" cargo build -p sofuu-core --bin sofuu
SOFUU="$ROOT/target/debug/sofuu"

# Shared C harness. It is built twice: plain for parity, ASAN for hostile input.
build_probe() {
    local out="$1"; shift
    "$CC" -std=c99 -O1 -g "$@" -I "$ROOT/kaze/dist" \
        "$ROOT/scripts/embedded-kaze-vm-probe.c" \
        "$ROOT/kaze/dist/kaze_vm.c" "$ROOT/kaze/dist/kaze_rt.c" \
        -lm -o "$out"
}

if ! build_probe "$TMP/probe" 2>"$TMP/probe-build.log"; then
    bad "C VM probe builds"
    tail -20 "$TMP/probe-build.log" | sed 's/^/      /'
    printf '\nembedded-kaze-c-gates: %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
    exit 1
fi
ok "C VM probe builds"

section "Gate 2 — hostile input, ASAN, no process termination"

build_probe "$TMP/probe-asan" -fsanitize=address -fno-omit-frame-pointer \
    2>"$TMP/probe-asan-build.log"
ASAN_RC=$?
if [ "$ASAN_RC" -eq 0 ]; then
    ok "ASAN VM probe builds"
else
    bad "ASAN VM probe builds"
    tail -20 "$TMP/probe-asan-build.log" | sed 's/^/      /'
fi

# The compiler half of the ABI is large; it still has to survive the same
# hostile-input run, so build a second ASAN probe with it linked in.
"$CC" -std=c99 -O1 -g -fsanitize=address -fno-omit-frame-pointer \
    -DPROBE_WITH_COMPILER -I "$ROOT/kaze/dist" \
    "$ROOT/scripts/embedded-kaze-vm-probe.c" \
    "$ROOT/kaze/dist/kaze_vm.c" "$ROOT/kaze/dist/kaze_rt.c" \
    "$ROOT/kaze/dist/kaze_embed.c" -lm -o "$TMP/probe-compiler-asan" \
    2>"$TMP/probe-compiler-asan-build.log"
COMPILER_ASAN_RC=$?
if [ "$COMPILER_ASAN_RC" -eq 0 ]; then
    ok "ASAN compiler probe builds"
else
    bad "ASAN compiler probe builds"
    tail -20 "$TMP/probe-compiler-asan-build.log" | sed 's/^/      /'
fi

export ASAN_OPTIONS="abort_on_error=1:detect_leaks=0"

cat >"$TMP/loop.kaze" <<'KAZE'
needs: log
action main():
    let n = 0
    while n < 10000 bounded 10000:
        set n to n + 1
KAZE
cat >"$TMP/heap.kaze" <<'KAZE'
needs: log
action main():
    let i = 0
    while i < 100000 bounded 100000:
        let xs = [1, 2, 3, 4, 5, 6, 7, 8]
        set i to i + 1
KAZE
cat >"$TMP/loop_fallback.kaze" <<'KAZE'
needs: log
action main():
    let z = 0
    let y = 1 / z otherwise 0
    return y
KAZE
cat >"$TMP/pause.kaze" <<'KAZE'
needs: memory
action main():
    memory.save("k", "v")
    return 1
KAZE

compile_v2() {
    "$SOFUU" script compile "$1" -o "$2" >/dev/null 2>&1
}
python3 "$ROOT/scripts/embedded-kaze-format-convert.py" --synth-div0 "$TMP/div0-v2.kazebc" >/dev/null &&
    compile_v2 "$TMP/loop.kaze" "$TMP/loop-v2.kazebc" &&
    compile_v2 "$TMP/heap.kaze" "$TMP/heap-v2.kazebc" &&
    compile_v2 "$TMP/pause.kaze" "$TMP/pause-v2.kazebc" ||
    bad "compile hostile fixtures"

# A trap, a fuel exhaustion, a collection-budget refusal and every truncation
# of a valid header must all come back as statuses, never a signal.
probe_case() {
    local label="$1"; shift
    local expected="$1"; shift
    local got
    if ! got=$("$@" 2>"$TMP/probe-err.log"); then
        bad "$label (probe exited non-zero)"
        cat "$TMP/probe-err.log" | tail -5 | sed 's/^/      /'
        return
    fi
    if [ "$got" = "$expected" ]; then
        ok "$label"
    else
        bad "$label"
        printf '      expected: %s\n      got:      %s\n' "$expected" "$got"
    fi
}

if [ "$ASAN_RC" -eq 0 ]; then
    probe_case "ASAN: div0 is a bounded trap" "ERR E_KAZE_DIV0 4" \
        "$TMP/probe-asan" run "$TMP/div0-v2.kazebc" main "" 1000
    probe_case "ASAN: fuel exhaustion is a bounded trap" "ERR E_TIMEOUT 4" \
        "$TMP/probe-asan" run "$TMP/loop-v2.kazebc" main "" 1000
    probe_case "ASAN: collection budget refuses rather than clamps" \
        "ERR E_KAZE_MEMORY 4" \
        "$TMP/probe-asan" heap "$TMP/heap-v2.kazebc" main 64
    probe_case "ASAN: truncated and corrupt headers do not crash" "CORRUPT_OK" \
        "$TMP/probe-asan" corrupt "$TMP/div0-v2.kazebc"
    pause_out=$("$TMP/probe-asan" pause "$TMP/pause-v2.kazebc" main "" 2>/dev/null | tr '\n' '|')
    case "$pause_out" in
        *"PAUSED memory save"*"OK 1 "*)
            ok "ASAN: capability pause resumes exactly once" ;;
        *)
            bad "ASAN: capability pause resumes exactly once"
            printf '      got: %s\n' "$pause_out" ;;
    esac
fi

if [ "$COMPILER_ASAN_RC" -eq 0 ]; then
    "$TMP/probe-compiler-asan" compiler "$TMP/loop_fallback.kaze" >"$TMP/compiler.out" 2>"$TMP/compiler.err"
    if grep -q '^CHECK 1 ' "$TMP/compiler.out" && grep -q '^COMPILE 1 ' "$TMP/compiler.out"; then
        ok "ASAN: embedded compiler checks and compiles"
    else
        bad "ASAN: embedded compiler checks and compiles"
        head -5 "$TMP/compiler.out" | sed 's/^/      /'
    fi
    cat >"$TMP/bad.kaze" <<'KAZE'
filter main(:
    allow "ok"
KAZE
    "$TMP/probe-compiler-asan" compiler "$TMP/bad.kaze" >"$TMP/compiler-bad.out" 2>"$TMP/compiler-bad.err"
    if grep -q '^CHECK 0 ' "$TMP/compiler-bad.out"; then
        ok "ASAN: embedded compiler returns a diagnostic for bad input"
    else
        bad "ASAN: embedded compiler returns a diagnostic for bad input"
        head -5 "$TMP/compiler-bad.out" | sed 's/^/      /'
    fi
fi

section "Gate 3 — bytecode format parity (1, 2, 4, 5)"

python3 "$ROOT/scripts/embedded-kaze-format-convert.py" \
    "$TMP/div0-v2.kazebc" "$TMP/div0" >"$TMP/convert.log" 2>&1
if [ -f "$TMP/div0-v1.kazebc" ] && [ -f "$TMP/div0-v4.kazebc" ]; then
    ok "Format 2 converts to legacy Formats 1 and 4"
else
    bad "Format 2 converts to legacy Formats 1 and 4"
    cat "$TMP/convert.log" | sed 's/^/      /'
fi

probe_case "Format 1 loads (item-line fallback)" "ERR E_KAZE_DIV0 2" \
    "$TMP/probe" run "$TMP/div0-v1.kazebc" main ""
probe_case "Format 2 preserves the exact statement line" "ERR E_KAZE_DIV0 4" \
    "$TMP/probe" run "$TMP/div0-v2.kazebc" main ""
probe_case "Format 4 loads (item-line fallback)" "ERR E_KAZE_DIV0 2" \
    "$TMP/probe" run "$TMP/div0-v4.kazebc" main ""

"$KAZE" compile "$TMP/loop_fallback.kaze" >"$TMP/div0-v5.hex" 2>"$TMP/v5.log"
if [ -s "$TMP/div0-v5.hex" ]; then
    python3 - "$TMP/div0-v5.hex" "$TMP/div0-v5.kazebc" <<'PY'
import sys
text = open(sys.argv[1]).read().strip()
open(sys.argv[2], "wb").write(bytes.fromhex(text))
PY
    probe_case "Format 5 loads and executes" "OK 0 13 13" \
        "$TMP/probe" run "$TMP/div0-v5.kazebc" main ""
else
    bad "standalone compiler emits Format 5"
    tail -5 "$TMP/v5.log" | sed 's/^/      /'
fi

section "Gate 4 — Rust engine elimination"

if [ ! -e "$ROOT/crates/sofuu-core/src/domains/kaze_engine" ]; then
    ok "domains/kaze_engine/ is removed"
else
    bad "domains/kaze_engine/ still exists"
fi

GLUE_FILES=$(find "$ROOT/crates/sofuu-core/src/domains/kaze" -name '*.rs' | wc -l | tr -d ' ')
EXTRA_KAZE_FILES=$(find "$ROOT/crates/sofuu-core/src/domains/kaze" -name '*.rs' \
    ! -name 'mod.rs' ! -name 'caps.rs' ! -name 'js_bridge.rs' | wc -l | tr -d ' ')
if [ "$GLUE_FILES" -eq 3 ] && [ "$EXTRA_KAZE_FILES" -eq 0 ]; then
    ok "domains/kaze/ contains only {mod.rs, caps.rs, js_bridge.rs}"
else
    bad "domains/kaze/ has unexpected .rs files ($GLUE_FILES total, $EXTRA_KAZE_FILES extra)"
fi

stale_modules=""
for module in arena ast c_codegen checker compiler geo isa lexer machine parser value vm; do
    if find "$ROOT/crates/sofuu-core/src/domains" -name "$module.rs" | grep -q .; then
        stale_modules="$stale_modules $module.rs"
    fi
done
if [ -z "$stale_modules" ]; then
    ok "none of the 12 duplicated engine modules exist under domains/"
else
    bad "duplicated engine module(s) still present:$stale_modules"
fi

if "$KAZE" --help 2>&1 | grep -q 'patch' && "$KAZE" --help 2>&1 | grep -q 'ast-hash'; then
    ok "kaze/bin/kaze exposes unified CLI subcommands after make kaze-build"
else
    bad "kaze/bin/kaze missing unified CLI subcommands (clobbered by make kaze-build?)"
fi

V1="$("$SOFUU" script version 2>&1 | head -1)"
V2="$("$KAZE" version 2>&1 | head -1)"
if [ "$V1" = "kaze 1.0" ] && [ "$V2" = "kaze 1.0" ]; then
    ok "sofuu and standalone kaze both report kaze 1.0"
else
    bad "version parity (sofuu: '$V1', kaze: '$V2')"
fi

section "Gate 5 — ABI surface, zero exit, vendored snapshot"

HEADER="$ROOT/kaze/dist/kaze_embed.h"
missing=""
for symbol in kz_embed_compiler_new kz_embed_compiler_free kz_embed_check \
              kz_embed_compile kz_embed_emit_c kz_embed_to_machine \
              kz_embed_to_human kz_vm_create kz_vm_destroy kz_vm_set_host \
              kz_vm_start kz_vm_step_slice kz_vm_supply_cap_result kz_vm_finish \
              kz_vm_error_code kz_vm_error_line kz_vm_value_json; do
    grep -q "$symbol" "$HEADER" || missing="$missing $symbol"
done
if [ -z "$missing" ]; then
    ok "kaze_embed.h exports the promised ABI"
else
    bad "kaze_embed.h is missing:$missing"
fi

# The VM has no process-termination path. kaze_embed.c is the compiler
# translation unit; in KAZE_EMBEDDED mode its KZ_EXIT records a status instead
# of leaving the process, and the source is generated from the same
# pure-Kaze compiler the standalone CLI uses.
if python3 - "$ROOT/kaze/dist/kaze_vm.c" <<'PY'
import re
import sys
source = open(sys.argv[1]).read()
source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
source = re.sub(r"//[^\n]*", "", source)
bad = re.findall(r"\b(?:exit|abort|longjmp)\s*\(", source)
raise SystemExit(1 if bad else 0)
PY
then
    ok "kaze_vm.c contains no exit(), abort() or longjmp()"
else
    bad "kaze_vm.c still calls exit(), abort() or longjmp()"
fi

if grep -q '^#define KAZE_EMBEDDED 1' "$ROOT/kaze/dist/kaze_embed.c" &&
   grep -q '#define KZ_EXIT(code) ((void)(kz_exit_code_g = (code)))' "$ROOT/kaze/dist/kaze_embed.c"; then
    ok "kaze_embed.c is built in KAZE_EMBEDDED mode with a status trap"
else
    bad "kaze_embed.c is not the embedded-mode translation unit"
fi

snapshot_bad=0
for f in kaze_rt.c kaze_rt.h kaze_vm.c kaze_vm.h kaze_embed.c kaze_embed.h; do
    if ! cmp -s "$ROOT/kaze/dist/$f" "$ROOT/deps/kaze/$f"; then
        bad "deps/kaze/$f differs from kaze/dist/$f (run make kaze-embed)"
        snapshot_bad=1
    fi
done
[ "$snapshot_bad" -eq 0 ] && ok "deps/kaze/ is byte-identical to kaze/dist/"

printf '\nembedded-kaze-c-gates: %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
