#!/usr/bin/env bash
# scripts/cuda/make_kaggle_bundle.sh — build the tiny self-contained archive the
# Kaggle T4 gate needs. No git checkout, no Rust, no CUDA toolkit.
#
# Output: dist/sofuu-cuda-kaggle.tar.gz
# Kaggle: upload as a Dataset, add it to the notebook, then
#   !tar xzf /kaggle/input/<dataset>/sofuu-cuda-kaggle.tar.gz -C /kaggle/working
#   !bash /kaggle/working/scripts/cuda/kaggle_gpu_check.sh
#   (or: cd /kaggle/working && bash scripts/cuda/kaggle_gpu_check.sh)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/dist/sofuu-cuda-kaggle.tar.gz}"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/sofuu-cuda-bundle.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/src" "$STAGE/scripts" "$STAGE/docs"
cp -R "$ROOT/src/cuda" "$STAGE/src/cuda"
cp -R "$ROOT/scripts/cuda" "$STAGE/scripts/cuda"
# Strip any local build debris (never expected, but be safe).
find "$STAGE" -name '*.o' -delete
find "$STAGE" -name '.DS_Store' -delete

if [[ -f "$ROOT/docs/CUDA-KAGGLE.md" ]]; then
    cp "$ROOT/docs/CUDA-KAGGLE.md" "$STAGE/docs/CUDA-KAGGLE.md"
fi

mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" -C "$STAGE" .
bytes=$(wc -c < "$OUT" | tr -d ' ')
echo "[cuda] wrote $OUT ($bytes B)"
echo "[cuda] contents:"
tar -tzf "$OUT" | sed -n '1,40p'
