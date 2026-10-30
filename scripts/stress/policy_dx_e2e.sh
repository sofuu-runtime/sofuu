#!/usr/bin/env bash
# scripts/stress/policy_dx_e2e.sh — R7.P3 policy developer-experience gates.
#
# Exercises the commands an operator actually types, end to end:
#
#   policy init          default/strict/offline profiles parse and carry the
#                        documented keys; overwrite is refused without --force
#   policy check <json>  document validation still works (probe table)
#   policy check <js>    static audit; exits 1 when a requirement is denied
#   policy dry-run <js>  prints would-be decisions; exits 0; denies say why
#   policy why           answers under the LAYERED on-disk policy
#   run --read-only      fs.write gate denies with E_PERMISSION_DENIED, and
#                        the file is not created
#   run --no-net         outbound network gate denies with E_PERMISSION_DENIED
#   run --confirm-all    tool.run evaluates to "ask"
#
# Usage:  bash scripts/stress/policy_dx_e2e.sh [path-to-sofuu]
# Exit:   0 all gates pass, 1 a gate failed.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOFUU="${1:-${SOFUU:-$ROOT/sofuu}}"
case "$SOFUU" in
    /*) ;;
    *) SOFUU="$(cd "$(dirname "$SOFUU")" 2>/dev/null && pwd)/$(basename "$SOFUU")" ;;
esac
if [ ! -x "$SOFUU" ]; then
    echo "✗ sofuu binary not found: $SOFUU (run make, or pass a path)" >&2
    exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sofuu-policy-dx.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
cd "$WORK"

PASS=0
FAIL=0
ok()   { PASS=$((PASS+1)); printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31m✗\033[0m %s\n' "$1"; }
check(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }

echo ""
echo "=== R7.P3 policy DX ==="

# ── init: all three profiles ────────────────────────────────────
for prof in default strict offline; do
    rm -rf .sofuu
    if "$SOFUU" policy init --profile "$prof" >/dev/null 2>&1; then
        ok "policy init --profile $prof writes .sofuu/policy.json"
    else
        bad "policy init --profile $prof failed"
        continue
    fi
    case "$prof" in
        default)
            check "default: fs.roots + shell.deny + per-run approval" \
              "python3 -c \"import json,sys;d=json.load(open('.sofuu/policy.json'));sys.exit(0 if d['fs']['roots'] and d['shell']['deny'] and d['approval']['mode']=='per-run' else 1)\""
            ;;
        strict)
            check "strict: fs.write off + shell off + net off + approval always" \
              "python3 -c \"import json,sys;d=json.load(open('.sofuu/policy.json'));sys.exit(0 if d['fs']['write']=='off' and d['shell']['enabled'] is False and d['net']['enabled'] is False and d['approval']['mode']=='always' else 1)\""
            ;;
        offline)
            check "offline: net.enabled false + fs.roots" \
              "python3 -c \"import json,sys;d=json.load(open('.sofuu/policy.json'));sys.exit(0 if d['net']['enabled'] is False and d['fs']['roots'] else 1)\""
            ;;
    esac
done
check "init refuses to overwrite without --force" "! $SOFUU policy init >/dev/null 2>&1"
check "init --force overwrites" "$SOFUU policy init --force >/dev/null 2>&1"

# ── check <json> keeps the probe-table contract ────────────────
rm -rf .sofuu && "$SOFUU" policy init --profile default >/dev/null 2>&1
OUT=$("$SOFUU" policy check .sofuu/policy.json 2>&1); RC=$?
check "policy check <json> exits 0" "[ $RC -eq 0 ]"
check "policy check <json> prints the probe table" "echo \"\$OUT\" | grep -q 'probes denied'"
check "policy check <json> shows shell.deny firing" "echo \"\$OUT\" | grep -q 'R7.6/shell.deny'"

# ── the audit script ───────────────────────────────────────────
cat > audit.js <<'JS'
const cfg = await sofuu.fs.readFile('/etc/passwd');
await sofuu.fs.writeFile('./out.txt', 'x');
await fetch('https://api.openai.com/v1/models');
const key = process.env.OPENAI_API_KEY;
await sofuu.spawn('rm -rf /');
JS
OUT=$("$SOFUU" policy check audit.js 2>&1); RC=$?
check "policy check <script> exits 1 when a requirement is denied" "[ $RC -eq 1 ]"
check "policy check <script> reports the fs root denial" "echo \"\$OUT\" | grep -q 'R7.6/fs.roots'"
check "policy check <script> reports the shell.deny denial" "echo \"\$OUT\" | grep -q 'R7.6/shell.deny'"
check "policy check <script> lists the env read" "echo \"\$OUT\" | grep -q 'OPENAI_API_KEY'"
check "policy check <script> --json emits decisions" \
  "($SOFUU policy check audit.js --json || true) | python3 -c \"import json,sys;a=json.load(sys.stdin);sys.exit(0 if any(r['decision']=='deny' for r in a) else 1)\""

OUT=$("$SOFUU" policy dry-run audit.js 2>&1); RC=$?
check "policy dry-run exits 0 (advisory)" "[ $RC -eq 0 ]"
check "policy dry-run prints [POLICY DRY-RUN] lines" "echo \"\$OUT\" | grep -q '\\[POLICY DRY-RUN\\]'"
check "policy dry-run explains a denial" "echo \"\$OUT\" | grep -q 'would fail closed'"
check "policy dry-run says the script was NOT executed" "echo \"\$OUT\" | grep -q 'NOT executed'"

# ── policy why reads the layered on-disk policy ────────────────
OUT=$("$SOFUU" policy why shell.run command='rm -rf /' 2>&1)
check "policy why answers deny under .sofuu/policy.json" "echo \"\$OUT\" | grep -q 'R7.6/shell.deny'"

# ── guard flags, enforced at the real gates ────────────────────
rm -rf .sofuu
cat > probe.js <<'JS'
function dec(req) { return JSON.parse(sofuu.policy.evaluate(req)); }
console.log('fs=' + dec({op:'fs.write', path: process.cwd() + '/x.txt'}).decision);
console.log('net=' + dec({op:'net.fetch', host:'example.com'}).decision);
console.log('tool=' + dec({op:'tool.run', tool:'read_file'}).decision);
console.log('shell=' + dec({op:'shell.run', command:'echo hi'}).decision);
JS
OUT=$("$SOFUU" run --read-only --no-net --confirm-all probe.js 2>&1)
check "--read-only + --no-net + --confirm-all reach the evaluator" \
  "echo \"\$OUT\" | grep -q 'fs=deny' && echo \"\$OUT\" | grep -q 'net=deny' && echo \"\$OUT\" | grep -q 'tool=ask' && echo \"\$OUT\" | grep -q 'shell=deny'"

cat > write.js <<'JS'
(async function () {
  try {
    await sofuu.fs.writeFile(process.cwd() + '/nope.txt', 'x');
    console.log('wrote');
  } catch (e) {
    console.log('denied:' + (e && e.code));
  }
})();
JS
OUT=$("$SOFUU" run --read-only write.js 2>&1)
check "--read-only refuses a real fs write (E_PERMISSION_DENIED)" "echo \"\$OUT\" | grep -q 'denied:E_PERMISSION_DENIED'"
check "--read-only created no file" "[ ! -f nope.txt ]"

cat > net.js <<'JS'
(async function () {
  try { await fetch('http://127.0.0.1:9/none'); console.log('allowed'); }
  catch (e) { console.log('denied:' + (e && e.code)); }
})();
JS
OUT=$("$SOFUU" run --no-net net.js 2>&1)
check "--no-net refuses a real fetch (E_PERMISSION_DENIED)" "echo \"\$OUT\" | grep -q 'denied:E_PERMISSION_DENIED'"

echo ""
if [ "$FAIL" -eq 0 ]; then
    printf '\033[32mpolicy DX: %d passed, 0 failed\033[0m\n\n' "$PASS"
    exit 0
fi
printf '\033[31mpolicy DX: %d passed, %d failed\033[0m\n\n' "$PASS" "$FAIL"
exit 1
