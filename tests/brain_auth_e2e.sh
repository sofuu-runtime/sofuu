#!/usr/bin/env bash
# tests/brain_auth_e2e.sh — P1-16 (AUDIT-2026-09-01): regression test for the
# 2026-08-22 P0-5 brain-server auth fix. Exercises the REAL `sofuu serve`
# binary end-to-end and hard-asserts the four shipped behaviors:
#   1. remote binding (--host != loopback) without an explicit --token is
#      refused (exit 1) — never silently auto-tokens a public listener
#   2. the auto-generated token is 16 urandom bytes ("sofuu-" + 32 hex)
#      and is persisted to ~/.sofuu/serve_token with mode 0600
#   3. no/wrong Bearer token -> 401 {"error":"unauthorized"} on every route
#   4. the correct token gets 200 — including an UPPERCASE header name
#      (regression for the lowercased req.headers normalization the
#      server's bearer auth depends on)
# No network beyond loopback, no API keys. Exits non-zero on any failure.
# Run:  bash tests/brain_auth_e2e.sh

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT=$((18800 + RANDOM % 200))

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_brain_auth_XXXXXX)"
HOME_DIR="$TMP/home"
mkdir -p "$HOME_DIR/.sofuu" "$TMP/proj"
trap 'kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null; rm -rf "$TMP"' EXIT

# ── 1. remote bind without a token is refused ─────────────────────
HOME="$HOME_DIR" SOFUU_PROJECT="$TMP/proj" perl -e 'alarm 15; exec @ARGV' \
  "$SOFUU" serve --host 0.0.0.0 --port "$PORT" >"$TMP/remote.log" 2>&1
RC=$?
grep -q "remote binding requires an explicit --token" "$TMP/remote.log"
check "remote bind without --token refused with the loud error" $?
[ "$RC" -eq 1 ]; check "remote bind without --token exits 1 (not 0, not hanging)" $?

# ── 2-4. live server: auto-token + 401s + authorized 200 ───────────
HOME="$HOME_DIR" SOFUU_PROJECT="$TMP/proj" "$SOFUU" serve --brain "$TMP/brain.qtsq" --port "$PORT" \
  >"$TMP/serve.log" 2>&1 &
SERVE_PID=$!
for _ in $(seq 1 50); do
  grep -q "brain server listening" "$TMP/serve.log" 2>/dev/null && break
  kill -0 "$SERVE_PID" 2>/dev/null || break
  sleep 0.1
done
grep -q "brain server listening" "$TMP/serve.log"
check "server started listening" $?

TOKEN="$(cat "$HOME_DIR/.sofuu/serve_token" 2>/dev/null)"
[ -n "$TOKEN" ]; check "auto-generated token persisted to ~/.sofuu/serve_token" $?
echo "$TOKEN" | grep -Eq '^sofuu-[0-9a-f]{32}$'
check "token shape: urandom 16 bytes (sofuu- + 32 lowercase hex)" $?
MODE="$(stat -f%Lp "$HOME_DIR/.sofuu/serve_token" 2>/dev/null || stat -c%a "$HOME_DIR/.sofuu/serve_token" 2>/dev/null)"
[ "$MODE" = "600" ]
check "serve_token file created with mode 0600" $?

probe() { # probe <expected-status> <curl-args...> — echoes body to $TMP/body
  local want="$1"; shift
  local got
  got="$(curl -s -o "$TMP/body" --max-time 5 -w '%{http_code}' "$@")"
  [ "$got" = "$want" ]
}

# 3a. no auth -> 401 with the unauthorized body
probe 401 "http://127.0.0.1:$PORT/health"
check "GET /health without auth -> 401" $?
grep -q '"unauthorized"' "$TMP/body"
check "401 body is the JSON error object" $?

# 3b. wrong token -> 401 (also POST route: auth runs before routing)
probe 401 -H "Authorization: Bearer sofuu-deadbeef" "http://127.0.0.1:$PORT/health"
check "GET /health with wrong token -> 401" $?
probe 401 -X POST -d '{"text":"x"}' "http://127.0.0.1:$PORT/remember"
check "POST /remember without auth -> 401 (auth precedes routing)" $?

# 4. correct token -> 200 on both routes; uppercase header name must
#    normalize into req.headers.authorization (the parsing regression).
probe 200 -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:$PORT/health"
check "GET /health with correct token -> 200" $?
grep -q '"ok":true' "$TMP/body"
check "authorized /health returns ok:true" $?
probe 200 -H "AUTHORIZATION: Bearer $TOKEN" "http://127.0.0.1:$PORT/health"
check "UPPERCASE header name still authorizes (lowercase normalization)" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "BRAIN AUTH E2E: ALL PASSED"
  exit 0
fi
echo "BRAIN AUTH E2E: $FAILURES FAILURE(S)"
cat "$TMP/serve.log" 2>/dev/null
exit 1
