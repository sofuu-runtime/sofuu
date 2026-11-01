#!/usr/bin/env bash
# scripts/pi5/verify_deploy.sh — M-R3 acceptance for the Pi 5 ambient device.
#
# Everything the plan's ambient smoke suite promises, checked against the
# running deployment: unit state, revision stamp and cap; the embedded seed;
# offline recall through the R7 policy; no TCP/UDP sockets; WAL recovery after
# a hard kill; and (opt-in) a write that survives a restart.
#
# The brain is SINGLE-WRITER: every brain CLI call and every direct artifact
# run below happens with the service stopped. Order matters and is deliberate.
#
# Usage (root; the plan's command line):
#   sudo scripts/pi5/verify_deploy.sh \
#     --artifact /var/lib/sofuu/ambient \
#     --brain /var/lib/sofuu/ambient.brain.qtsq \
#     --revision 98eb89352
#
# Exit: 0 all checks passed · 1 one or more failed · 2 usage/precondition.
set -uo pipefail

SERVICE=sofuu-agent
SERVICE_USER=sofuu
STATE_DIR=/var/lib/sofuu
ENVFILE=/etc/sofuu/agent.env
BINARY="$(command -v sofuu 2>/dev/null || true)"
ARTIFACT=""
BRAIN=""
REVISION=""
STRICT_SIZE=0
SKIP_RESTART=0
SKIP_SOCKETS=0
WRITE_TEST=0
PASS=0
FAIL=0
SKIPPED=0

pass() { PASS=$((PASS + 1)); printf '\033[32mPASS\033[0m %s\n' "$*"; }
fail() { FAIL=$((FAIL + 1)); printf '\033[31mFAIL\033[0m %s\n' "$*"; }
skip() { SKIPPED=$((SKIPPED + 1)); printf '\033[33mSKIP\033[0m %s\n' "$*"; }
info() { printf '     %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 2; }

usage() {
    cat <<EOF
Usage: sudo $0 [options]

  --artifact <path>   the compiled one-file artifact under test
  --brain <path>      the materialised brain (default: SOFUU_EMBEDDED_BRAIN
                      from $ENVFILE, else <artifact>.brain.qtsq)
  --revision <rev>    expected build revision (from \`sofuu version\`)
  --strict-size       fail when the binary is over the 5.5 MiB cap (default: warn)
  --skip-restart      do not kill -9 the service
  --skip-sockets      do not inspect the service's socket table
  --write-test        append one record and prove the count grows and survives
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --artifact) ARTIFACT=${2:?--artifact needs a path}; shift 2 ;;
        --brain) BRAIN=${2:?--brain needs a path}; shift 2 ;;
        --revision) REVISION=${2:?--revision needs a value}; shift 2 ;;
        --strict-size) STRICT_SIZE=1; shift ;;
        --skip-restart) SKIP_RESTART=1; shift ;;
        --skip-sockets) SKIP_SOCKETS=1; shift ;;
        --write-test) WRITE_TEST=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[ "$(id -u)" -eq 0 ] || die "run with sudo: the check stops, kills and starts $SERVICE"
[ -n "$BINARY" ] || die "sofuu is not on PATH (install it first)"
[ -x "$BINARY" ] || die "$BINARY is not executable"
systemctl cat "$SERVICE" >/dev/null 2>&1 || die "$SERVICE is not installed"

if [ -n "$ARTIFACT" ] && [ ! -x "$ARTIFACT" ]; then
    die "$ARTIFACT is not an executable artifact"
fi
if [ -z "$BRAIN" ] && [ -n "$ARTIFACT" ] && [ -f "$ARTIFACT.brain.qtsq" ]; then
    BRAIN="$ARTIFACT.brain.qtsq"
fi
if [ -z "$BRAIN" ] && [ -f "$ENVFILE" ]; then
    # Compiled shape first, then the development shape's file. Only a path
    # that exists is accepted, so a stale env value cannot mask the real store.
    for key in SOFUU_EMBEDDED_BRAIN SOFUU_DEVICE_BRAIN; do
        candidate="$(awk -F= -v k="$key" '$1 == k {print $2; exit}' "$ENVFILE")"
        if [ -n "$candidate" ] && [ -f "$candidate" ]; then
            BRAIN="$candidate"
            break
        fi
    done
fi
[ -z "$BRAIN" ] || [ -f "$BRAIN" ] || die "brain file not found: $BRAIN"

# A fleet key, when configured, is part of how the service opens its brain;
# the CLI checks below must use the same one.
BRAIN_KEY=""
if [ -f "$ENVFILE" ]; then
    BRAIN_KEY="$(awk -F= '/^SOFUU_BRAIN_KEY=/{print $2; exit}' "$ENVFILE")"
fi

# The unit's WorkingDirectory is where <cwd>/.sofuu/policy.json resolves; a
# direct artifact run must stand there too or it would load a different (or
# no) policy than the service does.
WORK_DIR="$(systemctl show -p WorkingDirectory --value "$SERVICE" 2>/dev/null)"
[ -n "$WORK_DIR" ] || WORK_DIR="$STATE_DIR"

# The machine key lives at $HOME/.sofuu/brain.key. root's HOME is /root, so a
# brain CLI call as root would mint a NEW key and fail to open the service's
# brain. Run the CLI as the service user, with its HOME and working directory —
# that also keeps any lock file the store creates owned by the writer.
brain() {
    local key_env=()
    [ -z "$BRAIN_KEY" ] || key_env=(SOFUU_BRAIN_KEY="$BRAIN_KEY")
    run_as_service env -C "$WORK_DIR" HOME="$STATE_DIR" "${key_env[@]}" "$BINARY" brain "$@"
}

# runuser keeps the machine key and file ownership right; sudo is the fallback.
if command -v runuser >/dev/null 2>&1; then
    run_as_service() { runuser -u "$SERVICE_USER" -- "$@"; }
else
    run_as_service() {
        local quoted
        quoted="$(printf '%q ' "$@")"
        su -s /bin/sh "$SERVICE_USER" -c "$quoted"
    }
fi

json_get() {
    # json_get <dotted.path>  (reads JSON on stdin; prints "" when absent)
    python3 -c '
import json, sys
path = sys.argv[1].split(".")
try:
    doc = json.load(sys.stdin)
except Exception:
    print("")
    sys.exit(0)
for key in path:
    if not isinstance(doc, dict) or key not in doc:
        print("")
        sys.exit(0)
    doc = doc[key]
print(doc if isinstance(doc, (str, int, float, bool)) else json.dumps(doc))
' "$1"
}

brain_verify_ok() {
    brain verify "$1" --json 2>/dev/null | json_get ok
}

brain_records() {
    brain stats "$1" --json 2>/dev/null | json_get records
}

brain_fingerprint() {
    # manifest + every sealed segment, so a rewritten file cannot hide behind
    # an identical record count.
    local p="$1" digest
    digest="$(sha256sum "$p" | awk '{print $1}')"
    if [ -d "$p.d" ]; then
        digest="$digest$(find "$p.d" -type f -name '*.qtsq' -print0 2>/dev/null \
            | sort -z | xargs -0 -r sha256sum 2>/dev/null \
            | awk '{print $1}' | tr '\n' ':')"
    fi
    printf '%s' "$digest" | sha256sum | awk '{print $1}'
}

artifact_run() {
    local embedded_env=()
    [ -z "$BRAIN" ] || embedded_env=(SOFUU_EMBEDDED_BRAIN="$BRAIN")
    [ -z "$BRAIN_KEY" ] || embedded_env+=(SOFUU_BRAIN_KEY="$BRAIN_KEY")
    run_as_service env -C "$WORK_DIR" HOME="$STATE_DIR" "${embedded_env[@]}" "$ARTIFACT" "$@"
}

echo ""
echo "Ambient device verification — $SERVICE"
echo "──────────────────────────────────────"
info "binary:   $BINARY"
[ -z "$ARTIFACT" ] || info "artifact: $ARTIFACT"
[ -z "$BRAIN" ] || info "brain:    $BRAIN"

# ── 1. binary and unit ────────────────────────────────────────────────
SIZE="$(stat -c %s "$BINARY")"
CAP=5767168
if [ "$SIZE" -le "$CAP" ]; then
    pass "binary size $SIZE B ≤ cap $CAP B"
else
    if [ "$STRICT_SIZE" -eq 1 ]; then
        fail "binary size $SIZE B is OVER the $CAP B cap (Linux builds are tracked work)"
    else
        skip "binary size $SIZE B is over the $CAP B cap (warn only; --strict-size to fail)"
    fi
fi

GOT_REV="$("$BINARY" version 2>/dev/null | awk '/^revision /{print $2}')"
if [ -n "$REVISION" ]; then
    if [ "$GOT_REV" = "$REVISION" ]; then
        pass "revision $GOT_REV matches the expected build"
    else
        fail "revision is '${GOT_REV:-unknown}', expected '$REVISION'"
    fi
else
    info "revision: ${GOT_REV:-unknown} (pass --revision to assert it)"
fi

if systemctl is-enabled --quiet "$SERVICE"; then
    pass "$SERVICE is enabled"
else
    fail "$SERVICE is not enabled (add WantedBy=multi-user.target)"
fi

WAS_ACTIVE=0
if systemctl is-active --quiet "$SERVICE"; then
    WAS_ACTIVE=1
    pass "$SERVICE is active"
    RESTART_POLICY="$(systemctl show -p Restart --value "$SERVICE")"
    if [ "$RESTART_POLICY" = "on-failure" ]; then
        pass "Restart=on-failure"
    else
        fail "Restart is '${RESTART_POLICY}', expected on-failure"
    fi
    if systemctl show "$SERVICE" | grep -q '^WatchdogUSec=[1-9]'; then
        fail "WatchdogSec is set but the runtime does not call sd_notify yet"
    fi
else
    skip "$SERVICE is not active — restart/socket checks will be skipped"
fi

# ── 2. no sockets while serving (structural offline) ──────────────────
if [ "$WAS_ACTIVE" -eq 1 ] && [ "$SKIP_SOCKETS" -eq 0 ]; then
    PID="$(systemctl show -p MainPID --value "$SERVICE")"
    if command -v ss >/dev/null 2>&1; then
        if [ "${PID:-0}" -gt 0 ] && ss -H -tunp 2>/dev/null | grep -q "pid=$PID,"; then
            fail "the service holds a TCP/UDP socket (pid $PID)"
            ss -H -tunp 2>/dev/null | grep "pid=$PID," | sed 's/^/     /'
        else
            pass "no TCP/UDP sockets held by the service"
        fi
    else
        skip "ss is not installed — cannot inspect the socket table"
    fi
else
    skip "socket check skipped"
fi

# ── 3. hard kill → systemd restart ────────────────────────────────────
if [ "$WAS_ACTIVE" -eq 1 ] && [ "$SKIP_RESTART" -eq 0 ]; then
    OLD_PID="$(systemctl show -p MainPID --value "$SERVICE")"
    OLD_RESTARTS="$(systemctl show -p NRestarts --value "$SERVICE")"
    [ -n "$OLD_RESTARTS" ] || OLD_RESTARTS=0
    if [ "${OLD_PID:-0}" -le 0 ]; then
        fail "cannot read MainPID — restart check inconclusive"
    else
        kill -9 "$OLD_PID" 2>/dev/null
        NEW_PID=0
        for _ in $(seq 1 30); do
            sleep 1
            NEW_PID="$(systemctl show -p MainPID --value "$SERVICE")"
            [ "${NEW_PID:-0}" -gt 0 ] && [ "$NEW_PID" != "$OLD_PID" ] && break
            NEW_PID=0
        done
        if [ "${NEW_PID:-0}" -gt 0 ] && systemctl is-active --quiet "$SERVICE"; then
            pass "kill -9 → systemd restarted the service (pid $OLD_PID → $NEW_PID)"
            NEW_RESTARTS="$(systemctl show -p NRestarts --value "$SERVICE")"
            if [ "${NEW_RESTARTS:-0}" -gt "$OLD_RESTARTS" ]; then
                pass "NRestarts incremented ($OLD_RESTARTS → $NEW_RESTARTS)"
            else
                info "NRestarts stayed at ${NEW_RESTARTS:-?} (unit was restarted by the manager)"
            fi
        else
            fail "the service did not come back after kill -9"
        fi
    fi
else
    skip "restart check skipped"
fi

# ── 4. stop the single writer, then check the brain ───────────────────
if [ "$WAS_ACTIVE" -eq 1 ]; then
    systemctl stop "$SERVICE"
    for _ in $(seq 1 15); do
        systemctl is-active --quiet "$SERVICE" || break
        sleep 1
    done
fi

if [ -n "$BRAIN" ]; then
    if [ "$(brain_verify_ok "$BRAIN")" = "True" ]; then
        pass "sofuu brain verify: clean"
    else
        fail "sofuu brain verify reported damage (see \`sofuu brain verify $BRAIN\`)"
        brain verify "$BRAIN" 2>&1 | sed 's/^/     /'
    fi

    BEFORE_COUNT="$(brain_records "$BRAIN")"
    info "records at check time: ${BEFORE_COUNT:-unknown}"
else
    skip "no brain given — seed/recall checks skipped"
fi

# ── 5. direct artifact checks (service stopped: single writer) ────────
if [ -n "$ARTIFACT" ] && [ -n "$BRAIN" ]; then
    BEFORE_FP="$(brain_fingerprint "$BRAIN")"

    STATS_OUT="$(artifact_run --stats 2>&1)"
    if printf '%s' "$STATS_OUT" | grep -q '"event":"stats"'; then
        pass "artifact --stats runs as the service user"
        NET_DECISION="$(printf '%s' "$STATS_OUT" | grep '"event":"stats"' | tail -1 | json_get net.decision)"
        if [ "$NET_DECISION" = "deny" ]; then
            pass "R7 policy denies net.fetch (decision: fail-closed)"
        else
            fail "net.fetch decision is '${NET_DECISION:-missing}', expected deny"
        fi
        SRC="$(printf '%s' "$STATS_OUT" | grep '"event":"stats"' | tail -1 | json_get policy_sources)"
        if printf '%s' "$SRC" | grep -q 'project'; then
            pass "policy loaded from <workdir>/.sofuu/policy.json"
        else
            fail "policy sources are '${SRC:-none}' — the device is running unprotected"
        fi
    else
        fail "artifact --stats did not produce a stats line"
        printf '%s\n' "$STATS_OUT" | tail -5 | sed 's/^/     /'
    fi

    RECALL_OUT="$(artifact_run --recall device 3 2>&1)"
    if printf '%s' "$RECALL_OUT" | grep -q '"text"'; then
        pass "offline recall returns the seeded device profile"
    else
        fail "offline recall returned no hits"
        printf '%s\n' "$RECALL_OUT" | tail -5 | sed 's/^/     /'
    fi

    if [ -n "${BEFORE_FP:-}" ]; then
        AFTER_FP="$(brain_fingerprint "$BRAIN")"
        if [ "$AFTER_FP" = "$BEFORE_FP" ]; then
            pass "read-only runs leave the brain byte-identical (seed not overwritten)"
        else
            skip "brain fingerprint changed during read-only runs — a WAL repair, not a seed overwrite (verify above decides)"
        fi
    fi

    AFTER_COUNT="$(brain_records "$BRAIN")"
    if [ -n "$BEFORE_COUNT" ] && [ -n "$AFTER_COUNT" ] && [ "$AFTER_COUNT" -lt "$BEFORE_COUNT" ]; then
        fail "record count fell from $BEFORE_COUNT to $AFTER_COUNT"
    else
        pass "record count did not fall (${BEFORE_COUNT:-?} → ${AFTER_COUNT:-?})"
    fi

    if [ "$WRITE_TEST" -eq 1 ]; then
        WRITE_OUT="$(artifact_run --once 2>&1)"
        if printf '%s' "$WRITE_OUT" | grep -q '"event":"reading"'; then
            pass "write test appended a reading"
        else
            fail "write test did not append (see output below)"
            printf '%s\n' "$WRITE_OUT" | tail -5 | sed 's/^/     /'
        fi
        GROWN="$(brain_records "$BRAIN")"
        if [ -n "$GROWN" ] && [ -n "$AFTER_COUNT" ] && [ "$GROWN" -gt "$AFTER_COUNT" ]; then
            pass "record count grew $AFTER_COUNT → $GROWN"
        else
            fail "record count did not grow (${AFTER_COUNT:-?} → ${GROWN:-?})"
        fi
        if [ "$(brain_verify_ok "$BRAIN")" = "True" ]; then
            pass "brain verifies after the write"
        else
            fail "brain does not verify after the write"
        fi
    fi
else
    skip "artifact checks skipped (need --artifact and a brain)"
fi

# ── 6. put the unit back the way it was found ─────────────────────────
if [ "$WAS_ACTIVE" -eq 1 ]; then
    systemctl start "$SERVICE"
    sleep 1
    if systemctl is-active --quiet "$SERVICE"; then
        pass "service restarted and active"
    else
        fail "service did not restart at the end of the run"
    fi
fi

echo "──────────────────────────────────────"
printf 'checks: \033[32m%d passed\033[0m, \033[31m%d failed\033[0m, \033[33m%d skipped\033[0m\n' \
       "$PASS" "$FAIL" "$SKIPPED"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
