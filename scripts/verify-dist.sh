#!/usr/bin/env bash
# verify-dist.sh — P4 install-integrity gate (R7 + the plan's P4 deliverable).
#
# 1. checksums: (re)generate dist/SHA256SUMS when missing; verify when present.
# 2. capabilities-vs-claim: the built binary's live surface must match what
#    the contract manifest documents for this build (QTSQ present → memory
#    primitive available and the session document surface exported).
# 3. doctor: exits 0 on a complete build.
# 4. version parity: Cargo, the CLI banner, and npm agree.
# 5. provenance (PC-07): no absolute build-host path in the binary or in the
#    dist CLI artifacts, and every one of them carries the source revision.
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

    # ── 3b. archive contents (H-06) ───────────────────────────────────────
    # MIT requires the licence text to travel with the binary. v0.3.0-beta
    # shipped tarballs containing the bare executable and nothing else, so the
    # gate now fails any archive that omits LICENSE or NOTICE.
    for tgz in "$ROOT"/dist/sofuu-*.tar.gz; do
        [ -f "$tgz" ] || continue
        listing=$(tar -tzf "$tgz" 2>/dev/null)
        for want in LICENSE NOTICE; do
            if printf '%s\n' "$listing" | grep -qE "(^|/)$want\$"; then
                say "OK  $(basename "$tgz") contains $want"
            else
                fail "$(basename "$tgz") is missing $want — rebuild with scripts/pack/stage_legal.sh"
            fi
        done
    done

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

    # ── 5. provenance: build paths + source revision (PC-07) ──────────────
    # AUDIT-2026-10-29: a published binary embedded 87 absolute build-host
    # paths — <home>/.cargo/registry/…, <home>/.rustup/…, <checkout>/crates/…
    # — and carried no source revision at all, so a binary could not be mapped
    # back to what produced it. Paths are remapped at compile time by
    # scripts/cross/remap_flags.sh (wired into the Makefile and the cross
    # scripts); the revision comes from crates/sofuu-core/build.rs and is
    # queryable as build.revision in `sofuu capabilities`.
    scan_text() { # strings(1), or a printable-run fallback where it is absent
        if command -v strings >/dev/null 2>&1; then
            strings "$1" 2>/dev/null
        else
            LC_ALL=C tr -c '[:print:]' '\n' < "$1" 2>/dev/null
        fi
    }
    # An absolute home root, not a URL path: the leading "/" must not be
    # preceded by a word character, "." "_" ":" or another "/", so
    # https://host/home/page does not trip the gate.
    leak_re='(^|[^[:alnum:]._:/])/(Users|home)/'
    check_paths() {
        hits=$(scan_text "$1" | grep -E "$leak_re" | grep -oE '/(Users|home)/[^[:space:]]*' | sort -u)
        n=$(printf '%s\n' "$hits" | grep -c . || true)
        if [ "$n" -eq 0 ]; then
            say "OK  $(basename "$1"): no absolute build-host paths"
        else
            fail "$(basename "$1") embeds $n absolute build-host path(s), e.g. $(printf '%s\n' "$hits" | head -1)"
            say "    rebuild it with the remap wired in (make / make dist / the cross scripts)"
        fi
    }
    stamp_of() { # the revision a binary was built from ("" when missing)
        "$1" capabilities 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin).get("build", {}).get("revision", ""))
except Exception:
    print("")' 2>/dev/null
    }
    check_paths "$BIN"
    # The shipped CLI artifacts are the ones this gate exists for; a stale
    # one in dist/ is reported by name so it can be rebuilt.
    for art in "$DIST"/sofuu-*; do
        [ -f "$art" ] || continue
        case "$art" in *.tar.gz | *.sha256 | *.txt) continue ;; esac
        [ "$art" = "$BIN" ] && continue
        check_paths "$art"
    done

    # ── 5b. source revision stamp ─────────────────────────────────────────
    rev=$(stamp_of "$BIN")
    case "$rev" in
        "")
            fail "no source revision stamp — capabilities.build.revision is empty (build.rs sets SOFUU_BUILD)"
            ;;
        unknown)
            fail "source revision is 'unknown' — built outside a git checkout; set SOFUU_BUILD=<short-sha> for those builds"
            ;;
        *)
            say "OK  build stamp: $rev"
            if scan_text "$BIN" | grep -qF -- "$rev"; then
                say "OK  stamp is embedded in the binary's strings"
            else
                fail "build stamp '$rev' does not appear in the binary's strings"
            fi
            case "$rev" in
                *-dirty) say "WARN built from a dirty tree ($rev) — release artifacts must come from a clean checkout" ;;
            esac
            base=${rev%-dirty}
            # Compare against the full sha by prefix: git's abbreviation width
            # grows with the object count, so a 7-char stamp is correct for a
            # 9-char `--short` HEAD of the same commit.
            head_full=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)
            head_short=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || true)
            if [ -z "$head_full" ]; then
                say "WARN no git checkout — cannot compare the stamp against HEAD"
            elif [ "${head_full#"$base"}" != "$head_full" ]; then
                say "OK  stamp matches HEAD ($head_short)"
            else
                fail "stamp ($base) does not match HEAD ($head_short) — the binary is stale, rebuild it"
            fi
            # A mixed dist (one artifact from an older revision) is a
            # release-integrity bug, so every shipped CLI binary must carry it.
            for art in "$DIST"/sofuu-*; do
                [ -f "$art" ] || continue
                case "$art" in *.tar.gz | *.sha256 | *.txt) continue ;; esac
                [ "$art" = "$BIN" ] && continue
                if scan_text "$art" | grep -qF -- "$base"; then
                    say "OK  $(basename "$art") carries stamp $base"
                else
                    fail "$(basename "$art") carries no stamp for $base — rebuild it (make dist / the cross scripts)"
                fi
            done
            ;;
    esac
fi

exit $rc
