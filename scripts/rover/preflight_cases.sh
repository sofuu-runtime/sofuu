#!/usr/bin/env bash
# scripts/rover/preflight_cases.sh — the M-R2 pre-flight half, with no SITL.
#
# Three cases through the REAL mission path (`sofuu drone mission --dry-run`):
#
#   1. mission.patrol.json                must pass   ("ok":true, exit 0)
#   2. mission.geofence-violation.json    must refuse (E_SAFETY_VIOLATION /
#                                         R-HW/bounds/geofenceRadiusM)
#   3. mission.speed-violation.json       must refuse (E_SAFETY_VIOLATION /
#                                         R-HW/bounds/maxSpeedMs)
#
# Nothing is sent: --dry-run stops after pre-flight, before a link is opened.
# The optional --sitl flag additionally flies the patrol, which needs an
# ArduPilot Rover SITL endpoint (see deploy/rover/README.md).
#
#   scripts/rover/preflight_cases.sh --binary ./sofuu
#   scripts/rover/preflight_cases.sh --binary ./sofuu --kaze
#   scripts/rover/preflight_cases.sh --binary ./sofuu --sitl \
#       --device udpout:192.168.1.10:14550
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="${SOFUU_BIN:-$(command -v sofuu 2>/dev/null || true)}"
POLICY="$ROOT/deploy/rover/rover.policy.json"
DIR="$ROOT/deploy/rover"
DEVICE=""
RUN_KAZE=0
RUN_SITL=0
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); printf '\033[32mPASS\033[0m %s\n' "$*"; }
fail() { FAIL=$((FAIL + 1)); printf '\033[31mFAIL\033[0m %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 2; }

usage() {
    cat <<EOF
Usage: $0 [options]

  --binary <path>   sofuu binary (default: on PATH, or \$SOFUU_BIN)
  --policy <path>   hardware policy (default: $POLICY)
  --dir <path>      mission directory (default: $DIR)
  --device <dev>    override the policy's first devices.allow entry
  --kaze            also run examples/kaze/rover.kaze on the deterministic sim
  --sitl            also fly mission.patrol.json for real (needs Rover SITL)
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --binary) BIN=${2:?--binary needs a path}; shift 2 ;;
        --policy) POLICY=${2:?--policy needs a path}; shift 2 ;;
        --dir) DIR=${2:?--dir needs a path}; shift 2 ;;
        --device) DEVICE=${2:?--device needs an endpoint}; shift 2 ;;
        --kaze) RUN_KAZE=1; shift ;;
        --sitl) RUN_SITL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[ -n "$BIN" ] || die "sofuu is not on PATH: pass --binary"
[ -x "$BIN" ] || die "$BIN is not executable"
[ -f "$POLICY" ] || die "policy not found: $POLICY"
for f in mission.patrol.json mission.geofence-violation.json mission.speed-violation.json; do
    [ -f "$DIR/$f" ] || die "missing $DIR/$f"
done

DEVICE_ARGS=()
[ -z "$DEVICE" ] || DEVICE_ARGS=(--device "$DEVICE")

# run_dry <mission>  → sets OUT and RC
run_dry() {
    OUT="$("$BIN" drone mission "$1" --policy "$POLICY" ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} --dry-run 2>&1)"
    RC=$?
}

echo ""
echo "Rover pre-flight cases"
echo "──────────────────────"
echo "  binary: $BIN"
echo "  policy: $POLICY"
"$BIN" version 2>/dev/null | sed 's/^/  /' || true
echo ""

# ── 0. the policy itself must validate (catches a pre-hardware binary) ─
PC_OUT="$("$BIN" policy check "$POLICY" 2>&1)"
PC_RC=$?
if [ "$PC_RC" -eq 0 ]; then
    pass "rover.policy.json validates against this build's schema"
else
    fail "the policy does not validate (exit $PC_RC) — this binary may predate the hardware section"
    printf '%s\n' "$PC_OUT" | tail -8 | sed 's/^/     /' >&2
fi

# ── 1. the patrol must pass ──────────────────────────────────────────
run_dry "$DIR/mission.patrol.json"
printf '%s\n' "$OUT" | sed 's/^/     /' >&2
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q '"ok":true'; then
    pass "patrol mission passes pre-flight"
else
    fail "patrol mission was not accepted (exit $RC)"
fi

# ── 2. geofence violation must be refused ────────────────────────────
run_dry "$DIR/mission.geofence-violation.json"
if [ "$RC" -ne 0 ] \
   && printf '%s' "$OUT" | grep -q 'E_SAFETY_VIOLATION' \
   && printf '%s' "$OUT" | grep -q 'geofenceRadiusM'; then
    pass "geofence violation refused before anything was sent"
else
    fail "geofence violation was not refused as expected (exit $RC)"
    printf '%s\n' "$OUT" | sed 's/^/     /' >&2
fi

# ── 3. speed violation must be refused ───────────────────────────────
run_dry "$DIR/mission.speed-violation.json"
if [ "$RC" -ne 0 ] \
   && printf '%s' "$OUT" | grep -q 'E_SAFETY_VIOLATION' \
   && printf '%s' "$OUT" | grep -q 'maxSpeedMs'; then
    pass "speed violation refused before anything was sent"
else
    fail "speed violation was not refused as expected (exit $RC)"
    printf '%s\n' "$OUT" | sed 's/^/     /' >&2
fi

# ── 4. Kaze rover skill body (deterministic sim, optional) ───────────
if [ "$RUN_KAZE" -eq 1 ]; then
    KAKE="$ROOT/examples/kaze/rover.kaze"
    if [ ! -f "$KAKE" ]; then
        fail "examples/kaze/rover.kaze not found"
    else
        KOUT="$("$BIN" script run "$KAKE" --cap rover=sim --cap lidar=sim \
                --ticks 5 --sim distance=0.3 2>&1)"
        KRC=$?
        if [ "$KRC" -eq 0 ]; then
            pass "Kaze rover obstacle loop runs on the sim backend"
        else
            fail "Kaze rover loop failed (exit $KRC)"
            printf '%s\n' "$KOUT" | sed 's/^/     /' >&2
        fi
    fi
fi

# ── 5. full SITL flight (optional) ───────────────────────────────────
if [ "$RUN_SITL" -eq 1 ]; then
    SOUT="$("$BIN" drone mission "$DIR/mission.patrol.json" --policy "$POLICY" \
            "${DEVICE_ARGS[@]}" 2>&1)"
    SRC=$?
    if [ "$SRC" -eq 0 ]; then
        pass "patrol flew end-to-end through the gate"
    else
        fail "patrol flight failed (exit $SRC)"
        printf '%s\n' "$SOUT" | tail -10 | sed 's/^/     /' >&2
    fi
fi

echo "──────────────────────"
printf 'cases: \033[32m%d passed\033[0m, \033[31m%d failed\033[0m\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
