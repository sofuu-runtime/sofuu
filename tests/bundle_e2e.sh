#!/usr/bin/env bash
# tests/bundle_e2e.sh — P1-13 (AUDIT-2026-09-07): `sofuu bundle` end-to-end over
# the committed tests/bundle_ns/ fixture chain. Two shipped behaviors:
#   1. `export * as ns from './dep.js'` no longer lands verbatim in the __d
#      factory (`export` is illegal in a function body -> SyntaxError); it
#      becomes a namespace binding on the re-exporting module's exports
#   2. the re-export target actually JOINS the module graph — before the fix a
#      module reachable only through a re-export was a "missing module" stub
#      at runtime, so ns came back empty
# entry.js hard-asserts ns.alpha/beta/default (via star-as) and ns.gamma (via
# the plain-star control) and exits 1 on any mismatch.
# Run:  bash tests/bundle_e2e.sh

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
FIX="$ROOT/tests/bundle_ns"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu-bundle-e2e.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# 1. Bundle the fixture chain.
"$SOFUU" bundle "$FIX/entry.js" -o "$TMP/out.js" >/dev/null 2>"$TMP/err"
check "bundle: sofuu bundle exits 0" $?

# 2. All three modules joined the graph (deps emit before dependents).
grep -q "__d('./dep.js'" "$TMP/out.js";    check "bundle: dep.js in graph" $?
grep -q "__d('./extra.js'" "$TMP/out.js";  check "bundle: extra.js in graph (plain-star control)" $?
grep -q "__d('./mid.js'" "$TMP/out.js";    check "bundle: mid.js in graph" $?

# 3. The star-as transform emitted the namespace binding, not verbatim source.
grep -q "var __sx=require('./dep.js');exports.ns=__sx;" "$TMP/out.js"
check "bundle: star-as emitted as namespace binding" $?
# Strip comments first: fixture comment text may legitimately contain "export *".
sed 's://.*$::' "$TMP/out.js" | grep -q "export \*"
[ $? -ne 0 ]; check "bundle: no verbatim export * left (comments stripped)" $?

# 4. The bundle RUNS and entry.js's assertions hold end-to-end.
OUT="$("$SOFUU" run "$TMP/out.js" 2>"$TMP/err")"
RC=$?
check "run: bundled output executes cleanly" "$RC"
printf '%s' "$OUT" | grep -q 'bundle_ns_ok'; check "run: ns.alpha/beta/default/gamma all correct" $?

if [ "$FAILURES" -gt 0 ]; then
    echo "--- bundle_e2e: $FAILURES failure(s) ---"
    [ -f "$TMP/err" ] && sed -n '1,15p' "$TMP/err"
    exit 1
fi
echo "bundle_e2e: all checks passed"
