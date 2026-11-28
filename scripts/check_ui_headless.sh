#!/usr/bin/env bash
# scripts/check_ui_headless.sh — Phase C5 gate for the embedded UI ABI.
#
# Builds libsofuu WITHOUT default features (no `sofuu-ui` in the graph) and
# proves the headless artifact exports none of the additive `sofuu_ui_*`
# symbols. This is the build a server / edge host links; it must stay exactly
# the frozen v2 surface.
#
# Usage: bash scripts/check_ui_headless.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Compile-time isolation, before the (slower) release build: the JS/AI
# runtime and the C ABI's non-UI half must not pull the external sofuu-ui
# crate into their graph (PLAN-SPLIT-UI-EFFECTS-EXTERNAL-REPO §1.3). This is
# also what lets a clone without `--recursive` build the headless CLI.
for crate in sofuu-core sofuu-ffi; do
    if cargo tree -p "$crate" -e normal --prefix none 2>/dev/null | grep -q '^sofuu-ui v'; then
        echo "  ✗ $crate compile-depends on sofuu-ui; keep the ui feature boundary" >&2
        exit 1
    fi
done
echo "  ✓ sofuu-core/sofuu-ffi have no sofuu-ui compile dependency"

echo "  building libsofuu with --no-default-features"
CARGO_PROFILE_RELEASE_PANIC=unwind \
    cargo build --release -p sofuu-capi --no-default-features >/dev/null

LIB=""
for candidate in target/release/libsofuu_capi.dylib \
                 target/release/libsofuu_capi.so \
                 target/release/libsofuu_capi.dll; do
    if [ -f "$candidate" ]; then
        LIB="$candidate"
        break
    fi
done

if [ -z "$LIB" ]; then
    echo "  ✗ no headless cdylib found under target/release/" >&2
    exit 1
fi

# nm flags are format-specific: Mach-O needs -gU, ELF needs -D.
if [[ "$LIB" == *.dylib ]]; then
    SYMS=$(nm -gU "$LIB" 2>/dev/null || true)
else
    SYMS=$(nm -D "$LIB" 2>/dev/null || true)
fi

if printf '%s\n' "$SYMS" | grep -q 'sofuu_ui_'; then
    echo "  ✗ headless libsofuu still exports sofuu_ui_*:"
    printf '%s\n' "$SYMS" | grep 'sofuu_ui_' | sed 's/^/    /'
    exit 1
fi

if ! printf '%s\n' "$SYMS" | grep -q 'sofuu_rt_new'; then
    echo "  ✗ headless libsofuu is missing the frozen v2 surface (sofuu_rt_new)" >&2
    exit 1
fi

echo "  ✓ headless libsofuu exports the v2 surface and no sofuu_ui_* symbols"
echo "    $LIB"
