#!/usr/bin/env bash
# tests/chat_models_e2e.sh — E2E for the /model picker's live listing +
# per-provider model memory (stored models survive an unreachable endpoint).
#
# Bugs this locks down:
#   1. LIVE-LISTING — the picker used to show only the provider's saved
#      default model. It must fetch EVERY model from each provider's own
#      endpoint (/v1/models, /api/tags for local) and list them all.
#   2. STORED FALLBACK — when a provider's endpoint is unreachable, its
#      remembered models (everything the user ever set there + the last
#      successful live listing) must still be offered instead of hiding
#      them behind a single current model.
#   3. TYPED PERSISTENCE — `/model <name>` (typed manually or picked) must
#      be remembered on that provider entry, and a successful live listing
#      must be cached the same way via __chat_models_cache.
#
# Harness: two providers in an isolated home — "deadprov" (ACTIVE) against
# a refused port with a seeded models array, and "mockprov" against the
# scripted mock LLM (tests/mock_llm_server.js, which now serves a /models
# catalog). Drives the chat TUI on a real pty (python pty.fork):
#   /model typed-manual → typed model persisted on the active provider
#   /model              → picker opens: the unreachable active tab shows its
#                         stored rows + "endpoint unreachable — enter it
#                         manually"; Tab → mockprov tab lists live models
# Then asserts the on-screen rows AND the written config.json (typed model
# + the live listing cached via __chat_models_cache).
#
# No network, no real API keys. Exits non-zero on any failure.
# Run:  bash tests/chat_models_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18801 + RANDOM % 89))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

# ── isolated environment ─────────────────────────────────────────
TMP="$(mktemp -d /tmp/sofuu_chat_models_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
MOCK_LOG="$TMP/mock.log"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "providers": [
    { "name": "deadprov", "endpoint": "http://127.0.0.1:9/v1/chat/completions",
      "api_key": "x", "model": "dead-default", "profile": "",
      "models": ["stored-one", "stored-two"] },
    { "name": "mockprov", "endpoint": "$BASE", "api_key": "x",
      "model": "mock-default", "profile": "openai" } ],
  "active": "deadprov",
  "provider": "deadprov", "model": "dead-default",
  "base_url": "http://127.0.0.1:9/v1/chat/completions", "api_key": "x", "profile": "",
  "brain": false, "ghost": false, "sync": false }
EOF

# ── mock LLM (serves /v1/models + completions) ───────────────────
# exec: the subshell replaces itself with sofuu, so $! is the SERVER pid
# (see chat_paste_e2e.sh — a plain subshell leaked the listening sofuu).
( cd "$ROOT" && exec env HOME="$HOME_P" "$SOFUU" run tests/mock_llm_server.js "$PORT" ) > "$MOCK_LOG" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 200); do
  grep -q "MOCK-LLM-READY" "$MOCK_LOG" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-LLM-READY" "$MOCK_LOG" || { echo "mock LLM failed to start"; cat "$MOCK_LOG"; exit 1; }

# ── drive the chat TUI on a pty ──────────────────────────────────
export HOME_P SOFUU
python3 - <<'PYEOF'
import os, pty, sys, time, select

home = os.environ["HOME_P"]
sofuu = os.environ["SOFUU"]
proj = os.path.dirname(home) + "/proj"

pid, fd = pty.fork()
if pid == 0:
    os.environ["HOME"] = home
    os.environ["TERM"] = "xterm-256color"
    os.chdir(proj)
    os.execv(sofuu, [sofuu, "chat"])

out = b""
start = time.time()
typed_model = opened_picker = switched_tab = escaped = False
while time.time() - start < 25:
    r, _, _ = select.select([fd], [], [], 0.2)
    if r:
        try: d = os.read(fd, 65536)
        except OSError: break
        if not d: break
        out += d
    # 1) type a model manually — must persist onto the ACTIVE provider
    #    (deadprov, whose endpoint is unreachable: the user's scenario)
    if not typed_model and time.time() - start > 4:
        typed_model = True
        os.write(fd, b"/model typed-manual\r")
    # 2) open the picker — deadprov tab shows stored rows + manual hint
    if typed_model and not opened_picker and time.time() - start > 7:
        opened_picker = True
        os.write(fd, b"/model\r")
    # 3) Tab → mockprov tab: the live catalog must be listed
    if opened_picker and not switched_tab and time.time() - start > 9.5:
        switched_tab = True
        os.write(fd, b"\t")
    # 4) esc closes the picker
    if switched_tab and not escaped and time.time() - start > 11:
        escaped = True
        os.write(fd, b"\x1b")
    if time.time() - start > 12.5:
        break
try:
    os.write(fd, b"\x03")
    time.sleep(0.3)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
except Exception:
    pass

text = out.decode("utf-8", "replace")
with open(os.path.dirname(home) + "/pty.txt", "w") as f:
    f.write(text)
print("PTY-DONE")
PYEOF

PTY_TXT="$TMP/pty.txt"
CFG="$HOME_P/.sofuu/config.json"

# ── on-screen assertions ─────────────────────────────────────────
check "picker fetches from both providers" \
  $(grep -q "Fetching models from 2 providers" "$PTY_TXT" && echo 0 || echo 1)
check "live listing succeeded for mockprov (1/2)" \
  $(grep -q "live models from 1/2 providers" "$PTY_TXT" && echo 0 || echo 1)
check "live models listed: live-alpha" \
  $(grep -q "live-alpha" "$PTY_TXT" && echo 0 || echo 1)
check "live models listed: live-beta" \
  $(grep -q "live-beta" "$PTY_TXT" && echo 0 || echo 1)
check "live models listed: live-gamma" \
  $(grep -q "live-gamma" "$PTY_TXT" && echo 0 || echo 1)
check "typed /model applied (Model → typed-manual)" \
  $(grep -q "Model → typed-manual" "$PTY_TXT" && echo 0 || echo 1)
check "deadprov stored row: stored-one" \
  $(grep -q "stored-one" "$PTY_TXT" && echo 0 || echo 1)
check "deadprov stored row: stored-two" \
  $(grep -q "stored-two" "$PTY_TXT" && echo 0 || echo 1)
check "unreachable endpoint keeps manual-entry hint" \
  $(grep -q "endpoint unreachable — enter it manually" "$PTY_TXT" && echo 0 || echo 1)

# ── config assertions (persistence) ──────────────────────────────
python3 - "$CFG" <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
provs = {p["name"]: p for p in cfg.get("providers", [])}
dp, mp = provs.get("deadprov", {}), provs.get("mockprov", {})
fails = []
def ck(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok: fails.append(name)
ck("active provider is deadprov", cfg.get("active") == "deadprov")
ck("typed model became the active model", cfg.get("model") == "typed-manual")
ck("typed model persisted on the provider entry", dp.get("model") == "typed-manual")
ck("typed model stored alongside the seeded models",
   dp.get("models") == ["stored-one", "stored-two", "typed-manual"])
mp_models = mp.get("models", [])
for m in ("live-alpha", "live-beta", "live-gamma"):
    ck("live listing cached: %s" % m, m in mp_models)
sys.exit(1 if fails else 0)
PYEOF
[ $? -eq 0 ] || FAILURES=$((FAILURES+1))

# ── failure transcript for debugging ─────────────────────────────
if [ "$FAILURES" -gt 0 ]; then
  echo "── pty transcript tail ──"
  tail -c 3000 "$PTY_TXT" | cat -v | tail -40
fi

echo "models e2e: $FAILURES failure(s)"
exit $([ "$FAILURES" -eq 0 ] && echo 0 || echo 1)
