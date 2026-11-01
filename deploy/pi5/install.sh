#!/usr/bin/env bash
# deploy/pi5/install.sh — install the Sofuu ambient-device service on a Pi 5.
#
# Two shapes, one program:
#
#   sudo ./install.sh --binary ~/Downloads/sofuu-linux-arm64
#       development shape: ExecStart=`sofuu run .../agent.js`, brain on disk.
#
#   sudo ./install.sh --binary /usr/local/bin/sofuu --artifact /var/lib/sofuu/ambient
#       production shape: ExecStart is the compiled one-file artifact.
#
# The script is idempotent. It never overwrites an existing brain, an existing
# /etc/sofuu/agent.env, or an operator-edited policy unless --force is given —
# a redeploy must not throw away what the device has remembered.
#
# It does NOT build the binary, seed the brain, or compile the artifact: those
# are explicit steps in README.md, because `sofuu compile` must run as the
# service user for the embedded brain's machine key to match.
set -euo pipefail

SERVICE=sofuu-agent
SERVICE_USER=sofuu
PREFIX=/usr/local
STATE_DIR=/var/lib/sofuu
CONF_DIR=/etc/sofuu
UNIT_DIR=/etc/systemd/system
UNIT="$UNIT_DIR/$SERVICE.service"
APP_DIR="$PREFIX/lib/sofuu/ambient"
HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$HERE/sofuu-agent.service.in"
POLICY_SRC="$HERE/ambient/policy.json"

BINARY=""
ARTIFACT=""
FLEET_KEY=""
NO_START=0
FORCE=0

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 2; }
say() { printf '  %s\n' "$*"; }

usage() {
    cat <<EOF
Usage: sudo $0 --binary <path-to-sofuu> [options]

Required
  --binary <path>      a built sofuu (QTSQ-linked for brain support)

Shape (choose one)
  --artifact <path>    a \`sofuu compile --embed-brain\` artifact; installed to
                       $STATE_DIR/ambient and run as the service
  (default)            run $APP_DIR/agent.js through the installed sofuu

Options
  --brain-key <hex>    write SOFUU_BRAIN_KEY (64 hex chars) to the env file;
                       for fleets that must share one brain key
  --no-start           install and enable, do not start/restart now
  --force              replace an existing env file / policy / artifact
  -h, --help           this page
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --binary)   BINARY=${2:?--binary needs a path}; shift 2 ;;
        --artifact) ARTIFACT=${2:?--artifact needs a path}; shift 2 ;;
        --brain-key) FLEET_KEY=${2:?--brain-key needs 64 hex chars}; shift 2 ;;
        --no-start) NO_START=1; shift ;;
        --force)    FORCE=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[ "$(id -u)" -eq 0 ] || die "run with sudo: this writes $UNIT_DIR and $STATE_DIR"
[ -n "$BINARY" ] || die "--binary is required"
[ -f "$BINARY" ] || die "$BINARY does not exist"
[ -x "$BINARY" ] || die "$BINARY is not executable"
[ -f "$TEMPLATE" ] || die "$TEMPLATE is missing (run from the repo checkout)"
[ -f "$POLICY_SRC" ] || die "$POLICY_SRC is missing (run from the repo checkout)"

if [ -n "$ARTIFACT" ]; then
    [ -f "$ARTIFACT" ] || die "$ARTIFACT does not exist"
    [ -x "$ARTIFACT" ] || die "$ARTIFACT is not executable"
    # A normal `sofuu` binary would run as the CLI (and start guided setup on a
    # tty-less unit). On the Pi's ELF artifact the 32-byte trailer sits at EOF,
    # so its first 12 bytes are the magic. (A signed Mach-O keeps the trailer
    # before the signature blob — this installer is Linux/systemd only.)
    if [ "$(tail -c 32 "$ARTIFACT" | head -c 12)" != "SOFUU_BIN_V1" ]; then
        die "$ARTIFACT is not a \`sofuu compile\` artifact (no SOFUU_BIN_V1 trailer)"
    fi
fi

if [ -n "$FLEET_KEY" ]; then
    case "$FLEET_KEY" in
        *[!0-9a-fA-F]*) die "--brain-key must be hex (64 characters)" ;;
    esac
    [ "${#FLEET_KEY}" -eq 64 ] || die "--brain-key must be 64 hex characters (got ${#FLEET_KEY})"
fi

echo ""
echo "Sofuu ambient device — install"
echo "───────────────────────────────"

# ── 1. service user ───────────────────────────────────────────────
if id -u "$SERVICE_USER" >/dev/null 2>&1; then
    say "user $SERVICE_USER: exists"
else
    useradd --system --home-dir "$STATE_DIR" --create-home \
            --shell /usr/sbin/nologin "$SERVICE_USER"
    say "user $SERVICE_USER: created (home $STATE_DIR)"
fi
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$STATE_DIR"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$STATE_DIR/.sofuu"

# ── 2. runtime binary ─────────────────────────────────────────────
if [ "$(readlink -f "$BINARY")" = "$(readlink -f "$PREFIX/bin/sofuu" 2>/dev/null || true)" ]; then
    say "binary: already at $PREFIX/bin/sofuu"
else
    install -m 0755 "$BINARY" "$PREFIX/bin/sofuu"
    say "binary: $PREFIX/bin/sofuu"
fi
REVISION="$("$PREFIX/bin/sofuu" version 2>/dev/null | awk '/^revision /{print $2}')" || true
say "revision: ${REVISION:-unknown}"

# ── 3. the program and its policy ─────────────────────────────────
install -d -m 0755 "$APP_DIR"
for f in agent.js seed.js README.md; do
    if [ -f "$HERE/ambient/$f" ]; then
        install -m 0644 "$HERE/ambient/$f" "$APP_DIR/$f"
    fi
done
say "app: $APP_DIR"

if [ -n "$ARTIFACT" ]; then
    EXEC_START="$STATE_DIR/ambient"
    if [ "$(readlink -f "$ARTIFACT")" = "$(readlink -f "$EXEC_START" 2>/dev/null || true)" ]; then
        say "artifact: already at $EXEC_START"
    else
        install -m 0755 "$ARTIFACT" "$EXEC_START"
        say "artifact: $EXEC_START"
    fi
else
    EXEC_START="$PREFIX/bin/sofuu run $APP_DIR/agent.js"
    say "exec: $EXEC_START"
fi

# The unit's cwd is the writable state dir in BOTH shapes: it is where
# <cwd>/.sofuu/policy.json is discovered, and where the runtime's own writes
# (brain key, run logs) are allowed by ProtectSystem=strict.
WORK_DIR="$STATE_DIR"

# Do not re-mode an existing .sofuu/ — in the artifact shape it is the
# 0750, service-owned directory created above, not a fresh app-dir folder.
if [ ! -d "$WORK_DIR/.sofuu" ]; then
    install -d -m 0755 "$WORK_DIR/.sofuu"
fi
if [ -f "$WORK_DIR/.sofuu/policy.json" ] && [ "$FORCE" -ne 1 ]; then
    say "policy: kept existing $WORK_DIR/.sofuu/policy.json (--force to replace)"
else
    install -m 0644 "$POLICY_SRC" "$WORK_DIR/.sofuu/policy.json"
    say "policy: $WORK_DIR/.sofuu/policy.json"
fi

# ── 4. environment file ───────────────────────────────────────────
install -d -m 0755 "$CONF_DIR"
ENVFILE="$CONF_DIR/agent.env"
if [ -f "$ENVFILE" ] && [ "$FORCE" -ne 1 ]; then
    say "env: kept existing $ENVFILE (--force to replace)"
else
    umask 077
    {
        echo "# generated by deploy/pi5/install.sh — edit in place, then restart"
        echo "SOFUU_EMBEDDED_BRAIN=$STATE_DIR/ambient.brain.qtsq"
        echo "SOFUU_DEVICE_BRAIN=$STATE_DIR/device.brain.qtsq"
        echo "SOFUU_DEVICE_DIM=768"
        echo "SOFUU_DEVICE_NAME=ambient-01"
        echo "SOFUU_DEVICE_SITE=bench"
        echo "SOFUU_READING_INTERVAL_MS=60000"
        if [ -n "$FLEET_KEY" ]; then
            echo "SOFUU_BRAIN_KEY=$FLEET_KEY"
        fi
    } > "$ENVFILE"
    umask 022
    chmod 0600 "$ENVFILE"
    say "env: $ENVFILE (0600)"
fi

# ── 5. systemd unit ───────────────────────────────────────────────
# sed-safe substitution: the only two placeholders in the template.
TMP_UNIT="$(mktemp)"
sed -e "s|@WORK_DIR@|$WORK_DIR|g" \
    -e "s|@EXEC_START@|$EXEC_START|g" \
    "$TEMPLATE" > "$TMP_UNIT"
install -m 0644 "$TMP_UNIT" "$UNIT"
rm -f "$TMP_UNIT"
say "unit: $UNIT"

systemctl daemon-reload
systemctl enable "$SERVICE" >/dev/null
say "enabled: $SERVICE"

if [ "$NO_START" -eq 1 ]; then
    say "not started (--no-start)"
else
    systemctl restart "$SERVICE"
    sleep 1
    if systemctl is-active --quiet "$SERVICE"; then
        say "active: $SERVICE"
    else
        printf '\033[31m  service did not stay active\033[0m\n' >&2
        systemctl status "$SERVICE" --no-pager -l || true
        exit 1
    fi
fi

cat <<EOF

Next steps
──────────
1. If the service is not healthy yet, read the app's JSON log lines:
     journalctl -u $SERVICE -n 50 --no-pager

2. Production shape only — seed the brain and build the one-file artifact AS
   THE SERVICE USER, so the machine key matches the one the service reads:
     sudo -u $SERVICE_USER -H $PREFIX/bin/sofuu run $APP_DIR/seed.js \\
          $STATE_DIR/seed.brain.qtsq

     sudo -u $SERVICE_USER -H $PREFIX/bin/sofuu compile $APP_DIR/agent.js \\
          --embed-brain $STATE_DIR/seed.brain.qtsq -o $STATE_DIR/ambient

   then re-run this installer with the same --binary plus
     --artifact $STATE_DIR/ambient --force
   (--force replaces the artifact only; the env file and brain are kept).

3. Verify the deployment:
     sudo scripts/pi5/verify_deploy.sh --brain $STATE_DIR/ambient.brain.qtsq \\
          --artifact $STATE_DIR/ambient --revision ${REVISION:-<rev>}

4. Record the M-R0 baseline:
     python3 scripts/pi5/bench_baseline.py --out /tmp/pi5-baseline.json
EOF
