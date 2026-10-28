#!/usr/bin/env bash
# verify-dist.sh — P4 install-integrity gate (R7 + the plan's P4 deliverable).
#
# 1. checksums: (re)generate dist/SHA256SUMS when missing; verify when present.
# 2. capabilities-vs-claim: the built binary's live surface must match what
#    the contract manifest documents for this build (QTSQ present → memory
#    primitive available and the session document surface exported).
# 3. doctor: exits 0 on a complete build.
# 4. version parity: Cargo, the CLI banner, and npm agree.
#
# Usage: bash scripts/verify-dist.sh [dist-dir] [binary]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="${1:-$ROOT/dist}"
BIN="${2:-$ROOT/sofuu}"
rc=0

say()  { printf '%s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*"; rc=1; }

# ── 1. checksums ─────────────────────────────────────────────────────────────
if [ ! -d "$DIST" ] || ! ls "$DIST"/libsofuu.* >/dev/null 2>&1; then
    fail "dist artifacts missing under $DIST (run make libsofuu)"
else
    if [ -f "$DIST/SHA256SUMS" ]; then
        if (cd "$DIST" && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1); then
            say "OK  dist checksums verify"
        else
            fail "dist checksums DO NOT verify — the artifact set changed without a re-stamp"
        fi
    else
        (cd "$DIST" && shasum -a 256 libsofuu.* *.h 2>/dev/null > SHA256SUMS)
        say "OK  dist checksums stamped (first run — commit SHA256SUMS with the archive)"
    fi
fi

# ── 2. capabilities-vs-claim ─────────────────────────────────────────────────
if [ ! -x "$BIN" ]; then
    fail "binary not executable: $BIN"
else
    # Classify from the build's own manifest, not typeof: since P1 a
    # QTSQ-free build exposes typed-refusal stubs, so `remember` is a
    # function there too and typeof cannot tell the two apart.
    has_qtsq=$("$BIN" eval 'console.log(sofuu.capabilities())' 2>/dev/null | tail -1 | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("unknown"); raise SystemExit
print("no" if "qtsq" in (d.get("build", {}).get("missing") or []) else "yes")' 2>/dev/null)
    [ -n "$has_qtsq" ] || has_qtsq=unknown
    case "$has_qtsq" in
        yes)
            mem=$("$BIN" eval 'console.log(typeof sofuu.memory.remember)' 2>/dev/null | tail -1)
            [ "$mem" = "function" ] \
                && say "OK  live capabilities: memory available (QTSQ core component linked)" \
                || fail "a QTSQ build must expose sofuu.memory.remember, got '${mem:-no output}'"
            ;;
        no)
            fail "QTSQ is a mandatory core component; binary is missing QTSQ"
            ;;
        *)
            fail "cannot read the build manifest (sofuu capabilities)"
            ;;
    esac
    sess=$("$BIN" eval 'console.log(typeof sofuu.session.export)' 2>/dev/null | tail -1)
    [ "$sess" = "function" ] && say "OK  live capabilities: session document surface exported" \
        || fail "session document surface missing (P3 regression)"
    doc=$("$BIN" eval 'console.log(typeof sofuu.events.append && typeof sofuu.events.render)' 2>/dev/null | tail -1)
    [ "$doc" = "function" ] && say "OK  live capabilities: R6 event surface exported" \
        || fail "event surface missing (P2 regression)"

    # ── 3. doctor ─────────────────────────────────────────────────────────
    # QTSQ is mandatory; doctor must exit 0 on every valid build.
    if "$BIN" doctor >/dev/null 2>&1; then
        say "OK  doctor exits 0 on this build"
    else
        fail "doctor exits non-zero — the build's declared features are not all present"
    fi

    # ── 4. version parity (Cargo ↔ npm ↔ CLI) ─────────────────────────────
    cargo_v=$(grep -m1 '^version' "$ROOT/Cargo.toml" 2>/dev/null | sed 's/.*"\(.*\)".*/\1/')
    cli_v=$("$BIN" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+[^ ]*' | head -1)
    npm_v=""
    for pj in "$ROOT/package.json" "$ROOT"/npm/package.json "$ROOT"/js/package.json; do
        if [ -f "$pj" ]; then
            npm_v=$(grep -m1 '"version"' "$pj" | sed 's/.*"\([0-9][^"]*\)".*/\1/')
            break
        fi
    done
    say "    versions: cargo=$cargo_v cli=$cli_v npm=${npm_v:-none}"
    if [ -n "$cargo_v" ] && [ -n "$cli_v" ] && [ "$cargo_v" = "$cli_v" ]; then
        say "OK  Cargo and CLI versions agree"
    else
        fail "Cargo ($cargo_v) and CLI ($cli_v) versions disagree"
    fi
    if [ -n "$npm_v" ] && [ -n "$cargo_v" ] && [ "$npm_v" != "$cargo_v" ]; then
        say "WARN npm version ($npm_v) differs from Cargo ($cargo_v) — align at the publish step"
    fi
fi

exit $rc
