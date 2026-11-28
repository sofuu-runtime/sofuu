#!/usr/bin/env bash
# scripts/check_ui_shaders.sh — shader syntax gate for the canonical UI kernels.
#
# The draw instance structs and the shader sources are one contract
# (`sofuu_ui::draw::SOFUU_DRAW_ABI_VERSION` + `sofuu_ui::shaders`). This gate
# compiles every pipeline branch of every backend family that the host may
# upload:
#
#   shaders/msl/ui_shaders.metal      Metal (xcrun metal)
#   shaders/glsl/ui_shaders_330.glsl  desktop OpenGL 3.3 (glslangValidator)
#   shaders/glsl/ui_shaders_es3.glsl  OpenGL ES 3.0 (glslangValidator)
#   shaders/glsl/ui_shaders_450.glsl  Vulkan / GLSL 4.5 (glslangValidator -V)
#
# Committed Vulkan SPIR-V is additionally checked with `spirv-val` when it is
# installed. Missing toolchains skip their family with a note (the plan's gate
# says "when installed"), so a Linux CI box without Xcode still runs the GLSL
# half and a Mac without glslang still runs Metal.
#
# Usage: bash scripts/check_ui_shaders.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Same checkout resolution as the JS/dts gates: an explicit SOFUU_UI_DIR wins,
# then the post-split submodule, then the pre-split in-tree crate.
UI_DIR="${SOFUU_UI_DIR:-}"
if [ -z "$UI_DIR" ] && [ -d deps/sofuu-ui/shaders ]; then
    UI_DIR="deps/sofuu-ui"
fi
if [ -z "$UI_DIR" ] && [ -d crates/sofuu-ui/shaders ]; then
    UI_DIR="crates/sofuu-ui"
fi
if [ -z "$UI_DIR" ]; then
    echo "check-ui-shaders: no sofuu-ui checkout found" >&2
    exit 1
fi
case "$UI_DIR" in
    /*) SHADERS="$UI_DIR/shaders" ;;
    *)  SHADERS="$ROOT/$UI_DIR/shaders" ;;
esac
SPIRV="$ROOT/crates/sofuu-gpu/src/shaders/spirv"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PIPELINES=(QUAD RIBBON STROKE GLYPH EFFECT PARTICLE BACKDROP BLUR)
# The Vulkan source only defines the six instance pipelines it ships SPIR-V
# for; backdrop/blur are unused GLSL-only branches there.
PIPELINES_450=(QUAD RIBBON STROKE GLYPH EFFECT PARTICLE)
checked=0
skipped=0

fail() {
    echo "  ✗ $1" >&2
    exit 1
}

# ── Metal ─────────────────────────────────────────────────────────
if command -v xcrun >/dev/null 2>&1 && xcrun -sdk macosx --find metal >/dev/null 2>&1; then
    if ! xcrun -sdk macosx metal -c "$SHADERS/msl/ui_shaders.metal" \
        -o "$TMP/ui_shaders.air" >"$TMP/metal.log" 2>&1; then
        cat "$TMP/metal.log" >&2
        fail "Metal source did not compile"
    fi
    echo "  ✓ Metal: ${#PIPELINES[@]} pipelines"
    checked=$((checked + 1))
else
    echo "  · Metal: skipped (xcrun metal not installed)"
    skipped=$((skipped + 1))
fi

# ── GLSL ──────────────────────────────────────────────────────────
if command -v glslangValidator >/dev/null 2>&1; then
    compile_glsl() { # <label> <file> <extra args...>
        local label="$1" file="$2"
        shift 2
        local p stage
        local list=("${PIPELINES[@]}")
        [ "$label" = 450 ] && list=("${PIPELINES_450[@]}")
        for p in "${list[@]}"; do
            for stage in vert frag; do
                local defines=("-DSOFUU_PIPELINE_$p")
                [ "$stage" = vert ] && defines+=("-DSOFUU_VERTEX")
                if ! glslangValidator -S "$stage" "${defines[@]}" "$@" "$file" \
                    -o "$TMP/$label.$p.$stage.spv" >"$TMP/glsl.log" 2>&1; then
                    cat "$TMP/glsl.log" >&2
                    fail "GLSL $label $p/$stage did not compile"
                fi
            done
        done
    }

    # Desktop GL 3.3 and ES 3.0 are valid driver GLSL but not SPIR-V-legal as
    # written (plain uniforms need explicit locations, and ES SPIR-V requires
    # >= 310 es). Auto-map for the check, and bump only the validation copy's
    # version line; the shipped source keeps 300 es for every Android driver.
    compile_glsl 330 "$SHADERS/glsl/ui_shaders_330.glsl" \
        -G --auto-map-locations --auto-map-bindings

    sed '1s/#version 300 es/#version 310 es/' \
        "$SHADERS/glsl/ui_shaders_es3.glsl" >"$TMP/ui_shaders_es3_310.glsl"
    compile_glsl es3 "$TMP/ui_shaders_es3_310.glsl" \
        -G --auto-map-locations --auto-map-bindings

    compile_glsl 450 "$SHADERS/glsl/ui_shaders_450.glsl" -V

    echo "  ✓ GLSL: 330/ES3 × ${#PIPELINES[@]} + 450 × ${#PIPELINES_450[@]} pipelines × 2 stages"
    checked=$((checked + 1))
else
    echo "  · GLSL: skipped (glslangValidator not installed)"
    skipped=$((skipped + 1))
fi

# ── Committed SPIR-V ───────────────────────────────────────────────
if command -v spirv-val >/dev/null 2>&1; then
    n=0
    for spv in "$SPIRV"/*.spv; do
        [ -e "$spv" ] || continue
        spirv-val "$spv" >"$TMP/val.log" 2>&1 || {
            cat "$TMP/val.log" >&2
            fail "invalid SPIR-V: ${spv#"$ROOT/"}"
        }
        n=$((n + 1))
    done
    echo "  ✓ SPIR-V: $n committed modules validate"
    checked=$((checked + 1))
else
    echo "  · SPIR-V: skipped (spirv-val not installed)"
    skipped=$((skipped + 1))
fi

if [ "$checked" -eq 0 ]; then
    echo "  · no shader toolchain installed; nothing to validate"
fi
echo "  sources: $UI_DIR/shaders (checked=$checked skipped=$skipped)"
