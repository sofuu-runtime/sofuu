#!/usr/bin/env bash
# conformance_report.sh — attach the P5 per-release conformance artifact:
# the 52-case contract board (its summary parsed into the report) + the
# runtime's own capability posture + the install-integrity gate, merged
# into one JSON next to the release.
#
# Usage: bash scripts/conformance_report.sh [out-file]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/dist/conformance-report.json}"
BIN="${SOFUU_BIN:-$ROOT/sofuu}"
[ -x "$BIN" ] || BIN="$ROOT/target/release/sofuu"
mkdir -p "$(dirname "$OUT")"

SUITE_LOG="$(mktemp)"
# Two case families execute only when their targets exist: T5.5 (C-ABI
# parity — a built libsofuu) and T1.3 (the typed memory refusal — a
# QTSQ-free binary). A release report should carry the strongest board the
# artifact set can support, so detect both instead of leaving them BLOCKED.
if [ -z "${CONTRACT_LIBSOFUU:-}" ]; then
    # The runner resolves this against the repo root, so hand it the
    # relative form.
    for cand in dist/libsofuu.dylib dist/libsofuu.so dist/libsofuu.a; do
        if [ -f "$ROOT/$cand" ]; then CONTRACT_LIBSOFUU="$cand"; break; fi
    done
fi
if [ -z "${CONTRACT_NOQTSQ_BIN:-}" ]; then
    miss=$("$BIN" eval 'console.log((JSON.parse(sofuu.capabilities()).build.missing||[]).join(","))' 2>/dev/null | tail -1)
    case ",$miss," in
        *,qtsq,*) CONTRACT_NOQTSQ_BIN="$BIN" ;;
    esac
fi
[ -n "${CONTRACT_LIBSOFUU:-}" ] && export CONTRACT_LIBSOFUU
[ -n "${CONTRACT_NOQTSQ_BIN:-}" ] && export CONTRACT_NOQTSQ_BIN
bash "$ROOT/scripts/contract/run_contract_tests.sh" > "$SUITE_LOG" 2>&1
suite_rc=$?

python3 - "$SUITE_LOG" "${SUITE_LOG}.json" "$suite_rc" <<'PY'
import json, re, sys
log = open(sys.argv[1]).read()
summary = re.search(r"summary: (\d+) cases — PASS (\d+) · FAIL (\d+) · BLOCKED (\d+)", log)
d = {
    "suite_rc": int(sys.argv[3]),
    "cases": int(summary.group(1)) if summary else None,
    "pass": int(summary.group(2)) if summary else None,
    "fail": int(summary.group(3)) if summary else None,
    "blocked": int(summary.group(4)) if summary else None,
    "violations": re.findall(r"^  - (\S+): (.+)$", log, re.M)[:20],
}
json.dump(d, open(sys.argv[2], "w"), indent=1)
PY

"$BIN" conformance --suite "${SUITE_LOG}.json" > "$OUT" 2>/dev/null \
    || "$BIN" conformance > "$OUT"

if bash "$ROOT/scripts/verify-dist.sh" "$ROOT/dist" "$BIN" >> "$SUITE_LOG" 2>&1; then
    python3 - "$OUT" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["install_integrity"]["verify_dist"] = "ok"
json.dump(d, open(p, "w"), indent=1)
PY
fi
rm -f "$SUITE_LOG" "${SUITE_LOG}.json"
echo "conformance report: $OUT (suite rc=$suite_rc)"
