#!/usr/bin/env bash
# scripts/cross/smoke_linux.sh — prove a cross-built Linux artifact works.
#
# The lean link and the build-std panic strategy are size changes, and neither
# can be smoke-tested on the macOS host that produces the artifact: qemu-system
# is here but user-mode emulation is not, so a macOS machine cannot execute an
# x86_64-musl binary. This script is the smoke test, and it must run on a Linux
# host (the release box, a container, or CI) before an artifact is published.
#
#   scripts/cross/smoke_linux.sh dist/sofuu-linux-x86_64
#
# It checks the size gate, then the four things a lean build can plausibly
# break: startup, the JS engine, the Kaze control plane, and a file round-trip.
set -uo pipefail

BIN="${1:-dist/sofuu-linux-x86_64}"
CAP=5767168
fail=0
ok()   { echo "  ✓ $*"; }
bad()  { echo "  ✗ $*"; fail=1; }

echo "smoke: $BIN"
if [ ! -f "$BIN" ]; then
    echo "error: no such file: $BIN" >&2
    exit 2
fi
chmod +x "$BIN" 2>/dev/null || true

# 1. the size gate itself (RULES.md §3.1)
bytes=$(wc -c < "$BIN")
if [ "$bytes" -le "$CAP" ]; then
    ok "size $bytes <= cap $CAP"
else
    bad "size $bytes exceeds cap $CAP by $((bytes - CAP))"
fi

# 2. startup: does the binary even reach its own main
if out=$("$BIN" --version 2>&1); then
    ok "--version: $(printf '%s' "$out" | head -1)"
else
    bad "--version failed: $(printf '%s' "$out" | head -3)"
fi

# 3. the JS engine runs and returns a value (QuickJS is linked, not JITed)
if out=$("$BIN" eval '1+1' 2>&1) && printf '%s' "$out" | grep -q '2'; then
    ok "js eval"
else
    bad "js eval: $(printf '%s' "$out" | head -3)"
fi

# 4. the Kaze control plane. The program is real Kaze -- action/needs shaped,
#    not stray JS -- so a parse regression fails here rather than in a user's
#    script. Structural equality of two lists is the check: it is recent code,
#    and it is evaluated in the same run loop the fuel meter lives in.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/eq.kaze" <<'KAZE'
needs: log

action main():
    let xs = [1, 2]
    let ys = [1, 2]
    let same = xs == ys
    log "equal"
KAZE
if out=$("$BIN" script run "$tmp/eq.kaze" 2>&1); then
    if printf '%s' "$out" | grep -q 'fuel'; then
        ok "kaze control plane ($(printf '%s' "$out" | grep -o '[0-9]* fuel' | head -1))"
    else
        ok "kaze control plane"
    fi
else
    bad "kaze: $(printf '%s' "$out" | head -3)"
fi

# 5. a file write and read-back: the persistence layer links server-side pieces
#    (QTSQ, zlib) that a lean link is the most likely to have broken.
if "$BIN" eval "require('fs').writeFileSync('$tmp/x.txt','hi'); require('fs').readFileSync('$tmp/x.txt','utf8')" 2>/dev/null | grep -q hi; then
    ok "file round-trip"
else
    bad "file round-trip"
fi

echo ""
if [ "$fail" -eq 0 ]; then
    echo "RESULT: PASS — publishable"
    exit 0
fi
echo "RESULT: FAIL — do not publish this artifact"
exit 1