#!/usr/bin/env bash
# scripts/kaze-drift-check.sh — the Kaze surface gate (PLAN-AUDIT-REMEDIATION §4 Task 2.1).
#
# Why this exists: there are two Kaze engines in this tree.
#
#   crates/sofuu-core/src/domains/kaze/   THE PRODUCT. Frozen language spec v0.1,
#                                         compiled into the sofuu binary.
#   kaze/                                 a standalone prototype crate with its own
#                                         workspace, its own CLI and its own tests.
#                                         Not built by make, not built by CI.
#
# Two copies of one language drift silently. They already did: the prototype grew
# device.set_duty and nine vector/trig opcodes that the product never shipped,
# while the documents pointed at the prototype. This gate makes the drift loud.
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
if [ ! -d "$PROTO/src" ]; then
    echo "kaze-drift: no prototype at $PROTO (gitignored) — nothing to compare"
    exit 0
fi

for f in "$SHIP/caps.rs" "$SHIP/isa.rs" "$SHIP/mod.rs" "$PROTO/src/caps.rs" "$PROTO/src/isa.rs" "$PROTO/src/lib.rs"; do
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
verbs() {
    grep -oE '\("[a-z_]+", "[^"]+"\)' "$1" \
        | sed -E 's/^\("([a-z_]+)", "(.*)"\)$/\1|\2/' \
        | while IFS='|' read -r fam rest; do
            echo "$rest" | tr '|' '\n' \
                | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/"//g' \
                | while read -r op; do
                    [ -n "$op" ] && printf '%s %s\n' "$fam" "$op"
                done
        done | sort -u
}

# Name@index from the Op enum, so a renumbered opcode is drift too.
opcodes() {
    awk '/pub enum Op \{/,/^\}/' "$1" \
        | grep -oE '^[[:space:]]+[A-Za-z]+ = [0-9]+' \
        | sed -E 's/^[[:space:]]+([A-Za-z]+) = ([0-9]+)/\1@\2/' \
        | sort
}

version() {
    sed -nE 's/^pub const VERSION: &str = "([^"]+)".*/\1/p' "$1" | head -1
}

verbs   "$SHIP/caps.rs"      >"$TMP/ship_verbs"
verbs   "$PROTO/src/caps.rs" >"$TMP/proto_verbs"
opcodes "$SHIP/isa.rs"       >"$TMP/ship_ops"
opcodes "$PROTO/src/isa.rs"  >"$TMP/proto_ops"
SHIP_V="$(version "$SHIP/mod.rs")"
PROTO_V="$(version "$PROTO/src/lib.rs")"

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
