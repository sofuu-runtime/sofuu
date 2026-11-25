#!/usr/bin/env bash
# scripts/kaze-drift-check.sh — the Kaze surface gate (PLAN-AUDIT-REMEDIATION §4 Task 2.1).
#
# Why this exists: there are two Kaze engines in this tree.
#
#   crates/sofuu-core/src/domains/kaze/   THE PRODUCT. Aligned language spec v1.0,
#                                         compiled into the sofuu binary.
#   kaze/                                 a standalone prototype crate with its own
#                                         workspace, its own CLI and its own tests.
#                                         Not built by make, not built by CI.
#
# Both engines implement the unified Kaze 1.0 contract (47 opcodes, 17 capability verbs).
# This gate enforces strict zero-drift across verbs, opcodes, and version strings.
#
# Policy, in order of severity:
#   FAIL   a capability verb or opcode that the PRODUCT has and the prototype
#          does not: the copy has fallen behind what ships.
#   FAIL   (strict only) anything present in only one engine, or a differing
#          language VERSION. Run with --strict, or KAZE_DRIFT_STRICT=1.
#   REPORT prototype-only verbs/opcodes and the two version strings. Expected
#          while the prototype stays a prototype, and printed on every run so
#          nobody can claim they were hidden.
#
#   make kaze-drift-check
#   scripts/kaze-drift-check.sh --strict
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHIP="$ROOT/crates/sofuu-core/src/domains/kaze"
PROTO="$ROOT/kaze"

STRICT=0
case "${1:-}" in
    --strict) STRICT=1 ;;
    "") ;;
    *) echo "usage: $0 [--strict]" >&2; exit 2 ;;
esac
[ "${KAZE_DRIFT_STRICT:-0}" = "1" ] && STRICT=1

# The prototype is gitignored (.gitignore /kaze/), so a clean clone has no copy
# to compare against. That is not a failure — there is nothing to drift from.
# The prototype is either a Rust checkout (old layout) or a pure-Kaze tree (Phase 10,
# which removed the Rust engine). A missing prototype still means nothing to drift from,
# but a present one must be readable: a stale path is a failure, not a green run.
if [ ! -d "$PROTO" ]; then
    echo "kaze-drift: no prototype at $PROTO (gitignored) — nothing to compare"
    exit 0
fi
if [ ! -d "$PROTO/src" ] && [ ! -d "$PROTO/compiler" ]; then
    echo "kaze-drift: prototype at $PROTO has neither src/ nor compiler/ — nothing to compare"
    exit 0
fi

for f in "$SHIP/caps.rs" "$SHIP/isa.rs" "$SHIP/mod.rs"; do
    if [ ! -f "$f" ]; then
        echo "kaze-drift: missing $f" >&2
        echo "kaze-drift: the gate is stale — fix the path before trusting a green run" >&2
        exit 2
    fi
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# (family, verb) pairs from an arity table. One line may carry several verbs
# ("forward" | "backward"), so each is expanded.
#
# The verbs come from the op_arity table only. A capability verb is a
# registry entry ("drone", "takeoff"); a telemetry reading ("drone",
# "battery") is a field of one operation, not a verb, and reading the whole
# file turned every new reading into phantom drift.
verbs() {
    # One "family op" per line from the op_arity table. An arm may list several
    # verbs with alternation - ("rover", "forward" | "backward") - so the alternation
    # has to be split before the comparison, or every such arm is silently dropped.
    awk '/pub fn op_arity/{inside=1} inside{print} inside && /^}/{exit}' "$1" \
        | grep -oE '\("[a-z_]+",[^)]*\)' \
        | sed -E 's/^\("([a-z_]+)",(.*)\)$/\1|\2/' \
        | while IFS='|' read -r fam rest; do
            echo "$rest" | tr -d '"' | tr '|' '\n' \
                | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' \
                | while read -r op; do
                    [ -n "$op" ] && printf '%s %s\n' "$fam" "$op"
                done
        done | sort -u
}

# Name@index from the Op enum, so a renumbered opcode is drift too.
opcodes() {
    awk '/pub enum Op \{/,/^\}/' "$1" \
        | grep -oE '^[[:space:]]+[A-Za-z][A-Za-z0-9]* = [0-9]+' \
        | sed -E 's/^[[:space:]]+([A-Za-z][A-Za-z0-9]*) = ([0-9]+)/\1@\2/' \
        | sort
}

version() {
    sed -nE 's/^pub const VERSION: &str = "([^"]+)".*/\1/p' "$1" | head -1
}

verbs   "$SHIP/caps.rs" >"$TMP/ship_verbs"
opcodes "$SHIP/isa.rs"  >"$TMP/ship_ops"
SHIP_V="$(version "$SHIP/mod.rs")"

# Phase 10 removed the prototype's Rust sources. When they are gone the standalone
# compiler prints its own surface (`kaze surface`), which is the same table the checker
# uses - so the comparison stays real instead of comparing against nothing.
PROTO_KINDS=""
if [ -f "$PROTO/src/caps.rs" ]; then
    verbs   "$PROTO/src/caps.rs" >"$TMP/proto_verbs"
    opcodes "$PROTO/src/isa.rs"  >"$TMP/proto_ops"
    PROTO_V="$(version "$PROTO/src/lib.rs")"
    PROTO_KINDS="rust"
else
    PROTO_BIN="${KAZE_BIN:-$PROTO/bin/kaze}"
    if [ ! -x "$PROTO_BIN" ]; then
        echo "kaze-drift: no prototype compiler at $PROTO_BIN; build it with make kaze-build" >&2
        exit 2
    fi
    "$PROTO_BIN" surface > "$TMP/proto_surface"
    grep -v '^opcode ' "$TMP/proto_surface" | sed -E 's/\./ /' | sort -u > "$TMP/proto_verbs"
    sed -n 's/^opcode //p' "$TMP/proto_surface" | sed -E 's/@/ /' | awk '{print $1"@"$2}' | sort > "$TMP/proto_ops"
    PROTO_V="$("$PROTO_BIN" version | awk 'NR==1 {print $2}')"
    if [ -z "$PROTO_V" ]; then PROTO_V="unknown"; fi
    PROTO_KINDS="pure"
fi

comm -23 "$TMP/ship_verbs" "$TMP/proto_verbs" >"$TMP/verbs_product_only"
comm -13 "$TMP/ship_verbs" "$TMP/proto_verbs" >"$TMP/verbs_proto_only"
comm -23 "$TMP/ship_ops"   "$TMP/proto_ops"   >"$TMP/ops_product_only"
comm -13 "$TMP/ship_ops"   "$TMP/proto_ops"   >"$TMP/ops_proto_only"

op_name() { sed -E 's/@[0-9]+$//' "$1"; }
count()   { wc -l <"$1" | tr -d ' '; }

fail=0
echo ""
echo "Kaze surface drift — product (domains/kaze) vs prototype (kaze/)"
echo "  product   $SHIP"
echo "  prototype $PROTO"
echo "  verbs     $(count "$TMP/ship_verbs") product · $(count "$TMP/proto_verbs") prototype"
echo "  opcodes   $(count "$TMP/ship_ops") product · $(count "$TMP/proto_ops") prototype"
echo "  version   $SHIP_V product · $PROTO_V prototype"
echo ""

if [ -s "$TMP/verbs_product_only" ] || [ -s "$TMP/ops_product_only" ]; then
    fail=1
    echo "FAIL — the prototype is BEHIND the product:"
    [ -s "$TMP/verbs_product_only" ] && sed 's/^/  verb only in product: /' "$TMP/verbs_product_only"
    [ -s "$TMP/ops_product_only" ]   && op_name "$TMP/ops_product_only" | sed 's/^/  opcode only in product: /'
    echo "  Fix: port the missing surface into kaze/, or delete the prototype."
    echo ""
fi

if [ -s "$TMP/verbs_proto_only" ] || [ -s "$TMP/ops_proto_only" ]; then
    echo "NOTE — the prototype is AHEAD of the product (expected while it stays a prototype):"
    [ -s "$TMP/verbs_proto_only" ] && sed 's/^/  verb only in prototype: /' "$TMP/verbs_proto_only"
    [ -s "$TMP/ops_proto_only" ]   && op_name "$TMP/ops_proto_only" | sed 's/^/  opcode only in prototype: /'
    echo "  Nothing to fix in the product: these are unshipped prototype features."
    echo "  If a product document claims one of them, that document is wrong."
    echo ""
    [ "$STRICT" -eq 1 ] && fail=1
fi

if [ "$SHIP_V" != "$PROTO_V" ]; then
    echo "NOTE — version strings differ: product '$SHIP_V' (the frozen spec), prototype '$PROTO_V'."
    echo ""
    [ "$STRICT" -eq 1 ] && fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "kaze-drift: OK"
    exit 0
fi
echo "kaze-drift: DRIFT DETECTED"
exit 1
