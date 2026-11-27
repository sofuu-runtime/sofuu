#!/usr/bin/env bash
# Regenerate the committed Vulkan SPIR-V binaries from ui_shaders_450.glsl.
# Requires glslangValidator (glslang) on PATH; runtime builds never invoke it.
set -euo pipefail
cd "$(dirname "$0")"
GLSLANG="${GLSLANG:-glslangValidator}"
mkdir -p spirv
for pipeline in QUAD RIBBON STROKE GLYPH; do
  name=$(printf '%s' "$pipeline" | tr '[:upper:]' '[:lower:]')
  "$GLSLANG" -V -S vert -DSOFUU_PIPELINE_$pipeline -DSOFUU_VERTEX glsl/ui_shaders_450.glsl -o "spirv/$name.vert.spv"
  "$GLSLANG" -V -S frag -DSOFUU_PIPELINE_$pipeline glsl/ui_shaders_450.glsl -o "spirv/$name.frag.spv"
done
echo "wrote $(ls spirv/*.spv | wc -l | tr -d ' ') SPIR-V modules"
