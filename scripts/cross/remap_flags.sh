#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# scripts/cross/remap_flags.sh — make a build independent of its host paths
# ─────────────────────────────────────────────────────────────────────
#
# Printing: `bash scripts/cross/remap_flags.sh` prints the rustc flags on one
# line, ready to prepend to RUSTFLAGS:
#
#   RUSTFLAGS="$(bash scripts/cross/remap_flags.sh) $RUSTFLAGS" cargo build --release
#
# PC-07 (AUDIT-2026-10-29): a published binary carried 87 absolute build-host
# paths — `<home>/.cargo/registry/...`, `<home>/.rustup/...` and
# `<checkout>/crates/...` — so every artifact published the producing
# machine's directory layout. rustc records the path it is *handed* into panic
# locations and `file!()`, and cargo hands it absolute paths, so the prefixes
# have to be remapped at compile time.
#
# Why a script instead of `.cargo/config.toml`: rustflags in a cargo config
# cannot expand `$HOME`/`$PWD`, and writing this developer's home directory
# into a committed file is exactly the leak being fixed. The Makefile
# (`make`, `make libsofuu`, `make dist`) and the cross scripts call this, so
# every artifact built from this tree gets the same remap. `.cargo/config.toml`
# keeps the checkout-independent half (shared temp roots) for a bare
# `cargo build`; when RUSTFLAGS is set it replaces that list, which is why the
# generic entries are mirrored below.
#
# Ordering: rustc applies the LAST matching prefix (not the longest), so this
# emits broadest → most specific. `$HOME` first, the workspace root last;
# otherwise `$HOME` would swallow the checkout and every crate path would read
# `./ai-native-js-runtime/crates/...` instead of `./crates/...`.
#
# No network, no dependencies, no side effects. Paths containing whitespace
# cannot be expressed in RUSTFLAGS (cargo splits it on whitespace) and are
# skipped with a warning.
set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
home="${HOME:-}"

flags=()
seen=" "

# add <absolute-prefix> <placeholder>
add() {
    local from="${1:-}" to="${2:-}"
    [ -n "$from" ] || return 0
    case "$from" in
        "/" | ".") return 0 ;;        # never remap the filesystem root
    esac
    case "$from" in
        *[[:space:]]*)
            printf 'remap_flags: %s contains whitespace — cannot be passed through RUSTFLAGS, skipped\n' "$from" >&2
            return 0
            ;;
    esac
    from="${from%/}"                  # rustc matches textually; no trailing slash
    case "$seen" in
        *" $from "*) return 0 ;;      # already covered
    esac
    seen="$seen$from "
    flags+=("--remap-path-prefix=$from=$to")
}

# under <path> <parent> — is path already covered by a prefix we emitted?
under() {
    [ -n "${1:-}" ] && [ -n "${2:-}" ] || return 1
    case "$1" in
        "$2"/*) return 0 ;;
    esac
    return 1
}

# ── 1. the user's home directory (registry, toolchain, sibling checkouts) ──
add "$home" "."

# ── 2. cargo/rustup homes when they live outside $HOME ────────────────────
under "${CARGO_HOME:-}" "$home" || add "${CARGO_HOME:-}" ".cargo"
under "${RUSTUP_HOME:-}" "$home" || add "${RUSTUP_HOME:-}" ".rustup"

# ── 3. shared temp roots: machine-specific but username-free ──────────────
# Mirrored in .cargo/config.toml (the part that needs no environment).
add "/private/var/folders" "/tmp"
add "/var/folders" "/tmp"
add "/private/tmp" "/tmp"
add "/var/tmp" "/tmp"
add "${TMPDIR:-}" "/tmp"

# ── 4. this checkout — last, so it wins over $HOME ───────────────────────
add "$root" "."

printf '%s' "${flags[*]:-}"
