#!/usr/bin/env bash
# Regenerate the committed Vulkan SPIR-V binaries from the canonical
# `sofuu_ui::shaders::GLSL_450_SRC` (crates/sofuu-ui/shaders/glsl).
# Requires glslangValidator (glslang) on PATH; runtime builds never invoke it.
set -euo pipefail
cd "$(dirname "$0")"
GLSLANG="${GLSLANG:-glslangValidator}"

# Pre-split the crate is at crates/sofuu-ui; after the repo split it is the
# deps/sofuu-ui submodule. SOFUU_UI_DIR wins for a sibling checkout.
if [ -n "${SOFUU_UI_DIR:-}" ]; then
    SRC="$SOFUU_UI_DIR/shaders/glsl"
else
    SRC=""
    for cand in ../../../sofuu-ui ../../../../deps/sofuu-ui; do
        if [ -f "$cand/shaders/glsl/ui_shaders_450.glsl" ]; then
            SRC="$cand/shaders/glsl"
            break
        fi
    done
fi
if [ -z "$SRC" ] || [ ! -f "$SRC/ui_shaders_450.glsl" ]; then
    echo "generate_spirv: canonical ui_shaders_450.glsl not found" >&2
    echo "  set SOFUU_UI_DIR or check out crates/sofuu-ui / deps/sofuu-ui" >&2
    exit 1
fi
mkdir -p spirv
for pipeline in QUAD RIBBON STROKE GLYPH EFFECT PARTICLE; do
  name=$(printf '%s' "$pipeline" | tr '[:upper:]' '[:lower:]')
  "$GLSLANG" -V -S vert -DSOFUU_PIPELINE_$pipeline -DSOFUU_VERTEX "$SRC/ui_shaders_450.glsl" -o "spirv/$name.vert.spv"
  "$GLSLANG" -V -S frag -DSOFUU_PIPELINE_$pipeline "$SRC/ui_shaders_450.glsl" -o "spirv/$name.frag.spv"
done
echo "wrote $(ls spirv/*.spv | wc -l | tr -d ' ') SPIR-V modules"
