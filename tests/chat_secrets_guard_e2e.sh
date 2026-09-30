#!/usr/bin/env bash
# tests/chat_secrets_guard_e2e.sh — js-5 (AUDIT-2026-09-07): grep/glob are
# auto-approved read tools with no credential guard. read_file refuses
# .sofuu/config.json, .ssh/* and *.pem (assertNotSensitive), but grep/glob
# walk and print anything: an explicit path=".sofuu" walks INSIDE the
# config dir (SKIP_DIRS filters child dirs only), ".ssh/" is not filtered,
# and "*.pem" files are read wherever they live — so a routine scan
# exfiltrates API keys / key material into the provider transcript.
#
# Vehicle: the JS chat engine (src/js/chat.js) via sofuu.chat.submit in
# tests/chat_secrets_guard_driver.js (same pattern as chat_state_copy_e2e).
#
# Fixtures (fake canaries only, created HERE in the shell):
#   .sofuu/config.json  KEYCANARY-VALUE=sk-FAKE-KEYCANARY-9911
#   server.pem + certs/ca.pem  FAKE-PEM-CANARY-BLOCK
#   .ssh/id_rsa  FAKE-SSH-CANARY-BLOCK
#   notes.txt    SECRETS-NOTES-PLAIN-OK   (control: read_file must work)
#
# RED (pre-fix): the mock sees canaries in tool messages → LEAKED + FAILs.
# GREEN (post-fix): the guards skip all four → PROTECTED + ALL PASSED.
# Run:  bash tests/chat_secrets_guard_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18991 + RANDOM % 89))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() {
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_sg_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "active": "mock-sg",
  "providers": [ { "name": "mock-sg", "model": "mock-model",
                   "endpoint": "$BASE", "api_key": "x" } ],
  "brain": false, "ghost": false, "sync": false }
EOF

# Canary fixtures (all fake values — nothing real).
mkdir -p "$PROJ/.sofuu" "$PROJ/certs" "$PROJ/.ssh"
printf '{\n  "note": "KEYCANARY-VALUE=sk-FAKE-KEYCANARY-9911"\n}\n' > "$PROJ/.sofuu/config.json"
printf '%s\n' '-----BEGIN CERTIFICATE-----' 'FAKE-PEM-CANARY-BLOCK' '-----END CERTIFICATE-----' > "$PROJ/server.pem"
printf 'FAKE-PEM-CANARY-BLOCK\n' > "$PROJ/certs/ca.pem"
printf 'FAKE-SSH-CANARY-BLOCK\n' > "$PROJ/.ssh/id_rsa"
printf 'SECRETS-NOTES-PLAIN-OK\n' > "$PROJ/notes.txt"

OUT="$TMP/driver.log"
perl -e 'alarm 240; exec @ARGV' -- \
  env HOME="$HOME_P" SOFUU_PROJECT="$PROJ" SOFUU_QTSQ_DIR="${SOFUU_QTSQ_DIR:-$HOME/projects/black-hole-disk}" \
    "$SOFUU" run "$ROOT/tests/chat_secrets_guard_driver.js" "$PORT" > "$OUT" 2>&1
RC=$?
grep -q "MOCK-SECRETS-READY" "$OUT" || { echo "FAIL mock server never became ready"; cat "$OUT"; exit 1; }

check "driver exited 0 (all in-driver checks passed)" "$RC"

for s in "leg 1 grep .sofuu done" "leg 2 glob tree done" "leg 3 grep pem done" \
         "leg 4 read notes done" "mock served all four legs" \
         "notes content reached the mock (read_file works)" \
         "fake-key canary never reached the LLM" "pem canary never reached the LLM" \
         "ssh key path never reached the LLM" "pem path never reached the LLM"; do
  if grep -q "PASS $s" "$OUT"; then echo "PASS $s"; else echo "FAIL $s"; FAILURES=$((FAILURES+1)); fi
done
grep -q "SECRETS-GUARD DRIVER: ALL PASSED" "$OUT" || { echo "FAIL driver final line"; FAILURES=$((FAILURES+1)); }

# Forensics on failure: the mock's per-leg observations + driver tail.
if [ $FAILURES -gt 0 ]; then
  echo "--- driver log (head) ---"; head -40 "$OUT"
  echo "--- driver log (tail) ---"; tail -30 "$OUT"
fi

echo "SECRETS-GUARD E2E: $FAILURES failures"
exit $([ $FAILURES -eq 0 ] && echo 0 || echo 1)
