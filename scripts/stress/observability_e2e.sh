#!/usr/bin/env bash
# scripts/stress/observability_e2e.sh — E2E for the observability CLI surface
# (OBSERVABILITY_PLAN.md Component 5, docs/OBSERVABILITY-SPEC.md §4):
#
#   sofuu logs                    table of runs (and a friendly empty state)
#   sofuu logs show <run_id>      rendered event trace
#   sofuu logs --last-error       formatted post-mortem for the newest crash
#   sofuu resume <run_id>         refuses a run with no checkpoint, exit != 0
#   sofuu --help / sofuu help     both list the new commands
#
# The whole run happens in a throwaway project (mktemp -d): the logs and crash
# cards are seeded by hand, so no model, no network and no downloads are
# involved. A command that has not landed yet is a clean FAIL, not a crash.
# Run ids follow the documented mapping: run id <id> lives at
# .sofuu/logs/agent-<id>.jsonl (OBSERVABILITY_PLAN.md Component 4), which is
# why "sofuu resume bot-01" reads agent-bot-01.jsonl.
#
# Exits non-zero on any failure.
# Run:  bash scripts/stress/observability_e2e.sh   (or: make observability-stress)
# Keep the scratch project: OBS_E2E_KEEP=1 bash scripts/stress/observability_e2e.sh

set -u

# Lives at scripts/stress/, two levels below the checkout root.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOFUU="${SOFUU_BIN:-$ROOT/sofuu}"

if [ ! -x "$SOFUU" ]; then
  echo "FATAL: no sofuu binary at $SOFUU (build it, or set SOFUU_BIN)" >&2
  exit 2
fi

PASS=0
FAIL=0
OUT=""
RC=0

ok()  { PASS=$((PASS+1)); echo "PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "FAIL $1${2:+  [$2]}"; }
first_line() { printf '%s\n' "$1" | sed -n '1p' | cut -c1-200; }

TMP="$(mktemp -d /tmp/sofuu_obs_e2e_XXXXXX)"
PROJ="$TMP/proj"
HOMEDIR="$TMP/home"
mkdir -p "$PROJ" "$HOMEDIR"
if [ "${OBS_E2E_KEEP:-0}" = "1" ]; then
  trap 'echo "  kept: $TMP"' EXIT
else
  trap 'rm -rf "$TMP"' EXIT
fi

# Run the CLI inside the scratch project and capture stdout+stderr (ANSI
# removed) plus the exit code. OUT/RC are the function's results.
run_sofuu() {
  OUT="$(cd "$PROJ" && HOME="$HOMEDIR" "$SOFUU" "$@" 2>&1)"
  RC=$?
  OUT="$(printf '%s\n' "$OUT" | sed $'s/\033\[[0-9;]*m//g')"
}

# Run ids, not file names: the log for <id> is .sofuu/logs/agent-<id>.jsonl.
RUN="e2e-ok"
NOCK="e2e-nock"
HALT="e2e-halt"
CRASH_ID="crash_e2e_1"

# ── case 1: an empty project has an explicit no-runs state ─────────────
echo "--- 1. sofuu logs with no runs ---"
run_sofuu logs
print_rc=$RC
printf '%s' "$OUT" | grep -Eqi 'no ([a-z-]+ )*runs?|none|empty|no agent' \
  && said_empty=yes || said_empty=no
if [ "$print_rc" -eq 0 ] && [ "$said_empty" = "yes" ]; then
  ok "sofuu logs on an empty project prints a no-runs message (exit 0)"
else
  bad "sofuu logs on an empty project prints a no-runs message (exit 0)" \
      "exit=$print_rc no_runs_message=$said_empty out=$(first_line "$OUT")"
fi

# ── seed: one completed run, one halted run + crash card, one live run ─
mkdir -p "$PROJ/.sofuu/logs" "$PROJ/.sofuu/crashes"

cat > "$PROJ/.sofuu/logs/agent-$RUN.jsonl" <<EOF
{"kind":"run_start","schema_version":1,"sid":"$RUN","turn_id":"t-1","seq":1,"ts":"2026-10-29T00:00:00.000Z","surface":"js","payload":{"task":"e2e seed"}}
{"kind":"turn_start","schema_version":1,"sid":"$RUN","turn_id":"t-1","seq":2,"ts":"2026-10-29T00:00:00.100Z","surface":"js","payload":{"step":1}}
{"kind":"tool_call","schema_version":1,"sid":"$RUN","turn_id":"t-1","seq":3,"ts":"2026-10-29T00:00:00.200Z","surface":"js","payload":{"tool":"read_file","args":{"path":"notes.txt"}}}
{"kind":"model_complete","schema_version":1,"sid":"$RUN","turn_id":"t-1","seq":4,"ts":"2026-10-29T00:00:00.300Z","surface":"js","payload":{"finish_reason":"stop","prompt_tokens":11,"completion_tokens":7}}
{"kind":"checkpoint","schema_version":1,"sid":"$RUN","turn_id":"t-1","seq":5,"ts":"2026-10-29T00:00:00.400Z","surface":"js","payload":{"step":1,"messages":[{"role":"user","content":"e2e seed"}],"task":"e2e seed","model":"local","provider":"local","tokens":18,"last_tool":"read_file"}}
{"kind":"run_complete","schema_version":1,"sid":"$RUN","turn_id":"t-1","seq":6,"ts":"2026-10-29T00:00:00.500Z","surface":"js","payload":{"steps":1}}
EOF

# A run that started but never checkpointed: sofuu resume must refuse it.
cat > "$PROJ/.sofuu/logs/agent-$NOCK.jsonl" <<EOF
{"kind":"run_start","schema_version":1,"sid":"$NOCK","turn_id":"t-1","seq":1,"ts":"2026-10-29T00:00:00.000Z","surface":"js","payload":{"task":"no checkpoint"}}
{"kind":"turn_start","schema_version":1,"sid":"$NOCK","turn_id":"t-1","seq":2,"ts":"2026-10-29T00:00:00.100Z","surface":"js","payload":{"step":1}}
EOF

# A halted run plus the crash card a fatal error would have written.
cat > "$PROJ/.sofuu/logs/agent-$HALT.jsonl" <<EOF
{"kind":"run_start","schema_version":1,"sid":"$HALT","turn_id":"t-1","seq":1,"ts":"2026-10-29T00:00:01.000Z","surface":"js","payload":{"task":"halt"}}
{"kind":"fatal_error","schema_version":1,"sid":"$HALT","turn_id":"t-1","seq":2,"ts":"2026-10-29T00:00:01.100Z","surface":"js","payload":{"code":"E_SCHEMA","message":"tool transfer_funds failed argument validation"}}
{"kind":"halt","schema_version":1,"sid":"$HALT","turn_id":"t-1","seq":3,"ts":"2026-10-29T00:00:01.200Z","surface":"js","payload":{"reason":"fatal_error"}}
EOF

cat > "$PROJ/.sofuu/crashes/$CRASH_ID.json" <<EOF
{"crash_id":"$CRASH_ID","run_id":"$HALT","timestamp":"2026-10-29T00:00:01.000Z","failed_step":14,"error":{"code":"E_SCHEMA","message":"tool transfer_funds failed argument validation","details":{"missing_fields":["recipient_account"]}},"context_snapshot":{"active_model":"local","context_tokens_used":4210,"last_user_message":"Send 100 to Alice","last_tool_in_flight":"transfer_funds"},"recovery_hint":"Run sofuu resume $HALT after fixing the tool schema."}
EOF

# ── case 2: the table lists the seeded run with its derived status ─────
echo "--- 2. sofuu logs lists the seeded run ---"
run_sofuu logs
list_rc=$RC
printf '%s' "$OUT" | grep -q "$RUN" && has_run=yes || has_run=no
printf '%s' "$OUT" | grep -Eq '[[:space:]]OK([[:space:]]|$)' && has_ok=yes || has_ok=no
if [ "$list_rc" -eq 0 ] && [ "$has_run" = "yes" ] && [ "$has_ok" = "yes" ]; then
  ok "sofuu logs lists $RUN with status OK"
else
  bad "sofuu logs lists $RUN with status OK" \
      "exit=$list_rc lists_run=$has_run lists_status_OK=$has_ok out=$(first_line "$OUT")"
fi

# ── case 3: logs show renders the trace ────────────────────────────────
echo "--- 3. sofuu logs show <run> ---"
run_sofuu logs show "$RUN"
show_rc=$RC
printf '%s' "$OUT" | grep -q 'run_start' && has_start=yes || has_start=no
printf '%s' "$OUT" | grep -q 'checkpoint' && has_ck=yes || has_ck=no
printf '%s' "$OUT" | grep -q 'run_complete' && has_done=yes || has_done=no
if [ "$show_rc" -eq 0 ] && [ "$has_start" = "yes" ] && [ "$has_ck" = "yes" ] && [ "$has_done" = "yes" ]; then
  ok "sofuu logs show $RUN renders run_start/checkpoint/run_complete"
else
  bad "sofuu logs show $RUN renders run_start/checkpoint/run_complete" \
      "exit=$show_rc run_start=$has_start checkpoint=$has_ck run_complete=$has_done out=$(first_line "$OUT")"
fi

# ── case 4: logs --last-error renders the crash card ───────────────────
echo "--- 4. sofuu logs --last-error ---"
run_sofuu logs --last-error
err_rc=$RC
printf '%s' "$OUT" | grep -q 'E_SCHEMA' && has_code=yes || has_code=no
printf '%s' "$OUT" | grep -q "$HALT" && has_halt=yes || has_halt=no
printf '%s' "$OUT" | grep -Eqi 'recovery|resume' && has_hint=yes || has_hint=no
if [ "$err_rc" -eq 0 ] && [ "$has_code" = "yes" ] && [ "$has_halt" = "yes" ] && [ "$has_hint" = "yes" ]; then
  ok "sofuu logs --last-error renders the $HALT post-mortem"
else
  bad "sofuu logs --last-error renders the $HALT post-mortem" \
      "exit=$err_rc shows_code=$has_code shows_run=$has_halt shows_hint=$has_hint out=$(first_line "$OUT")"
fi

# ── case 5: resume refuses a run with no checkpoint ────────────────────
echo "--- 5. sofuu resume <run> with no checkpoint ---"
run_sofuu resume "$NOCK"
res_rc=$RC
if [ "$res_rc" -ne 0 ]; then nonzero=yes; else nonzero=no; fi
printf '%s' "$OUT" | grep -Eqi 'checkpoint' && says_ck=yes || says_ck=no
printf '%s' "$OUT" | grep -Eqi 'panicked|backtrace|segmentation fault' && panicked=yes || panicked=no
if [ "$nonzero" = "yes" ] && [ "$says_ck" = "yes" ] && [ "$panicked" = "no" ]; then
  ok "sofuu resume $NOCK exits non-zero with a clean no-checkpoint error"
else
  bad "sofuu resume $NOCK exits non-zero with a clean no-checkpoint error" \
      "exit=$res_rc non_zero=$nonzero names_checkpoint=$says_ck panicked=$panicked out=$(first_line "$OUT")"
fi

# ── case 6/7: both help spellings list the new commands ────────────────
# The command lines are the contract: a bare word match would also hit the
# prose in the banner, so anchor on the "sofuu <cmd>" line shape.
help_lists_new() {
  printf '%s\n' "$1" | grep -Eq '^[[:space:]]*sofuu[[:space:]]+logs([[:space:]]|$)' || return 1
  printf '%s\n' "$1" | grep -Eq '^[[:space:]]*sofuu[[:space:]]+resume([[:space:]]|$)' || return 1
  return 0
}

echo "--- 6. sofuu --help ---"
run_sofuu --help
help1_rc=$RC
if [ "$help1_rc" -eq 0 ] && help_lists_new "$OUT"; then
  ok "sofuu --help lists logs and resume"
else
  bad "sofuu --help lists logs and resume" "exit=$help1_rc"
fi

echo "--- 7. sofuu help ---"
run_sofuu help
help2_rc=$RC
if [ "$help2_rc" -eq 0 ] && help_lists_new "$OUT"; then
  ok "sofuu help lists logs and resume"
else
  bad "sofuu help lists logs and resume" "exit=$help2_rc"
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "OBSERVABILITY E2E: ALL PASSED ($PASS cases)"
else
  echo "OBSERVABILITY E2E: $FAIL FAILURE(S), $PASS passed"
  exit 1
fi
