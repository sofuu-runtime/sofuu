#!/usr/bin/env bash
# scripts/seed_sofuu_ui_repo.sh — seed the standalone `sofuu-ui` repository.
#
# PLAN-SPLIT-UI-EFFECTS-EXTERNAL-REPO §6 Phase 2. This copies the current
# working tree (including uncommitted UI work, which is why `git subtree split`
# alone is not enough yet) into a self-contained Cargo package:
#
#   <target>/Cargo.toml      standalone metadata + the same dependencies
#   <target>/src/            ir, layout, text, draw, compiler, audit, fx, ...
#   <target>/assets/         SFD font atlases
#   <target>/shaders/        canonical MSL + GLSL sources
#   <target>/js/             ui.js + ui_kit.js + index.d.ts
#   <target>/tests/          pure wire-parity tests + fixtures
#
# Nothing in the seed reaches back into the host checkout: the crate's tests
# read `js/` and `tests/fixtures/` relative to CARGO_MANIFEST_DIR.
#
# Usage:
#   bash scripts/seed_sofuu_ui_repo.sh [--target DIR] [--repo-url URL]
#                                      [--commit] [--history] [--no-test] [--force]
#
#   --target DIR   destination (default: $SOFUU_UI_REPO_DIR or ~/projects/sofuu-ui)
#   --repo-url URL canonical remote written to Cargo.toml (default:
#                  $SOFUU_UI_REPO_URL or https://github.com/sofuu-runtime/sofuu-ui)
#   --commit       git init + commit + tag v0.1.0 (the developer pushes)
#   --history      preserve committed history with `git subtree split` first
#   --no-test      skip the standalone `cargo test` verification
#   --force        allow seeding over an existing non-empty directory
#
# The default is a dry-ish seed: files + standalone manifest, no git commands
# and no commit. The developer owns pushes and tags.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TARGET="${SOFUU_UI_REPO_DIR:-$HOME/projects/sofuu-ui}"
repo_url="${SOFUU_UI_REPO_URL:-https://github.com/sofuu-runtime/sofuu-ui}"
do_commit=0
do_history=0
do_test=1
force=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --target) TARGET="${2:?--target needs a directory}"; shift 2 ;;
        --repo-url) repo_url="${2:?--repo-url needs a URL}"; shift 2 ;;
        --commit) do_commit=1; shift ;;
        --history) do_history=1; shift ;;
        --no-test) do_test=0; shift ;;
        --force) force=1; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "seed-sofuu-ui: unknown argument: $1" >&2; exit 2 ;;
    esac
done

SRC="$ROOT/crates/sofuu-ui"
if [ ! -f "$SRC/Cargo.toml" ] || [ ! -d "$SRC/shaders" ] || [ ! -d "$SRC/js" ]; then
    echo "seed-sofuu-ui: $SRC is not the prepared crate (shaders/ and js/ required)" >&2
    exit 1
fi

if [ -e "$TARGET" ] && [ -n "$(ls -A "$TARGET" 2>/dev/null || true)" ]; then
    if [ "$force" -ne 1 ]; then
        echo "seed-sofuu-ui: $TARGET is not empty (use --force to overlay it)" >&2
        exit 1
    fi
fi

if [ "$do_history" -eq 1 ]; then
    echo "seed-sofuu-ui: splitting committed history (git subtree split)"
    git subtree split -P crates/sofuu-ui -b split/sofuu-ui >/dev/null
    rm -rf "$TARGET"
    # --no-local + --single-branch: copy only objects reachable from the
    # subtree branch, never the host's full object database (a local clone
    # would otherwise hardlink everything). --no-tags: host tags name host-wide
    # releases, not sofuu-ui releases; the seed creates its own baseline below.
    git clone --quiet --no-local --no-tags --single-branch \
        --branch split/sofuu-ui "$ROOT" "$TARGET"
    git -C "$TARGET" remote remove origin || true
    # Defensive: local/transport quirks have copied host tags before.
    git -C "$TARGET" tag -l | while IFS= read -r tag; do
        git -C "$TARGET" tag -d "$tag" >/dev/null
    done
    git -C "$TARGET" branch -m main
    echo "  cloned the subtree branch into $TARGET (branch: main)"
fi

mkdir -p "$TARGET"
for part in src assets shaders js tests; do
    mkdir -p "$TARGET/$part"
    cp -R "$SRC/$part/." "$TARGET/$part/"
done

if [ -f "$ROOT/LICENSE" ]; then
    cp -p "$ROOT/LICENSE" "$TARGET/LICENSE"
fi
if [ -f "$ROOT/rust-toolchain.toml" ]; then
    cp -p "$ROOT/rust-toolchain.toml" "$TARGET/rust-toolchain.toml"
fi

# Standalone Cargo.toml: no `workspace = true` inheritance, same dependency
# set as the in-tree crate. Read the dependency block so it cannot drift.
python3 - "$SRC/Cargo.toml" "$TARGET/Cargo.toml" "$repo_url" <<'PY'
import re
import sys

src, dst, repo_url = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src).read()

deps = ''
m = re.search(r'^\[dependencies\]\s*$(.*?)(?=^\[|\Z)', text, re.S | re.M)
if m:
    deps = m.group(1).strip()

manifest = f'''[package]
name = "sofuu-ui"
version = "0.1.0"
edition = "2021"
license = "MIT"
description = "Sofuu custom 2D GPU UI compositor core — layout, springs, SDF text, effects, audit and packed DrawList"
repository = "{repo_url}"

[lib]
name = "sofuu_ui"
path = "src/lib.rs"

[dependencies]
{deps}
'''
open(dst, 'w').write(manifest)
print(f"  wrote {dst}")
PY

# Profiles live in `.cargo/config.toml`, not Cargo.toml: a bare `cargo test`
# in the standalone checkout still gets opt-level 1 (the 50k-row 1 ms budget is
# meaningless at -O0), while consuming this package as a workspace member does
# not emit Cargo's "profiles for the non root package will be ignored" warning.
mkdir -p "$TARGET/.cargo"
cat >"$TARGET/.cargo/config.toml" <<'EOF'
# Standalone sofuu-ui build profile. Ignored when this crate is consumed as a
# path dependency/member of the host workspace (the host root owns profiles).
[profile.dev]
opt-level = 1

[profile.release]
opt-level = "z"
lto = "fat"
codegen-units = 1
EOF

cat >"$TARGET/.gitignore" <<'EOF'
/target
Cargo.lock
.DS_Store
EOF

cat >"$TARGET/README.md" <<'EOF'
# sofuu-ui

Domain-agnostic 2D GPU UI compositor core for [Sofuu](https://github.com/sofuu-runtime/sofuu):
Scene IR, responsive layout, springs, SDF typography, icons/images, effects
choreography, accessibility semantics, geometric audit and a flat `#[repr(C)]`
`DrawList` consumed by Metal, OpenGL/GLES and Vulkan hosts.

This repository was split out of `sofuu` by
`PLAN-SPLIT-UI-EFFECTS-EXTERNAL-REPO.md`. The host (`sofuu`) keeps the OS
windowing/GPU drivers (`sofuu-gpu`), the C ABI (`sofuu-capi`), the JS/Kaze
runtime and the GPU Platform UI Emulator (`sofuu-emu`).

Canonical remote: <https://github.com/sofuu-runtime/sofuu-ui>.

## Contract

* `src/draw.rs` owns the instance structs and `SOFUU_DRAW_ABI_VERSION`.
* `shaders/` owns the canonical MSL/GLSL sources; `src/shaders.rs` bundles
  them with `include_str!`, so a host cannot compile a shader that disagrees
  with the structs it uploads.
* `js/ui.js` + `js/ui_kit.js` are the declarative `sofuu.ui` / `sofuu.fx`
  surface; `js/index.d.ts` is the module-shaped UI/FX type surface.
  The host keeps generated copies under `src/js/` for headless builds
  (Option A in the plan) and checks them with `make ui-js-sync-check`.

## Build

```bash
cargo test          # pure Rust, no Kaze/QuickJS/GPU dependency
cargo build
```

Shader syntax (when the toolchains are installed):

```bash
xcrun -sdk macosx metal -c shaders/msl/ui_shaders.metal -o /tmp/ui.air
glslangValidator -V -DSOFUU_PIPELINE_QUAD -DSOFUU_VERTEX \
  shaders/glsl/ui_shaders_450.glsl -o /tmp/quad.vert.spv
```
EOF

# Keep the generated README's link in lock-step with the manifest's repository.
python3 - "$TARGET/README.md" "$repo_url" <<'PY'
import sys
path, repo_url = sys.argv[1], sys.argv[2]
text = open(path).read()
text = text.replace("https://github.com/sofuu-runtime/sofuu-ui", repo_url)
open(path, "w").write(text)
PY

echo "seed-sofuu-ui: staged $TARGET"
echo "  src=$(find "$TARGET/src" -name '*.rs' | wc -l | tr -d ' ') rust files, shaders=$(find "$TARGET/shaders" -type f | wc -l | tr -d ' '), js=$(find "$TARGET/js" -type f | wc -l | tr -d ' ')"

if [ "$do_test" -eq 1 ]; then
    echo "seed-sofuu-ui: cargo test (standalone)"
    # Run from inside the seed so Cargo also discovers `.cargo/config.toml`
    # (the opt-level 1 dev profile): `--manifest-path` from the host would not.
    ( cd "$TARGET" && cargo test --quiet )
    echo "seed-sofuu-ui: standalone tests OK"
fi

if [ "$do_commit" -eq 1 ]; then
    if [ ! -d "$TARGET/.git" ]; then
        git -C "$TARGET" init --quiet
    fi
    git -C "$TARGET" add -A
    git -C "$TARGET" commit --quiet -m "Seed sofuu-ui from ai-native-js-runtime @ $(git rev-parse --short HEAD)"
    git -C "$TARGET" branch -M main
    git -C "$TARGET" tag -f v0.1.0 >/dev/null
    echo "seed-sofuu-ui: committed + tagged v0.1.0 (push it yourself)"
else
    echo "seed-sofuu-ui: no git/commit/tag written (pass --commit to create them)"
fi

cat <<EOF

Next (PLAN-SPLIT-UI-EFFECTS-EXTERNAL-REPO Phase 2/3), run by the maintainer:

  # Skip these four lines if you passed --commit (already committed + tagged):
  git -C "$TARGET" init && git -C "$TARGET" add -A
  git -C "$TARGET" commit -m "Seed sofuu-ui"
  git -C "$TARGET" branch -M main
  git -C "$TARGET" tag -f v0.1.0

  git -C "$TARGET" remote add origin "$repo_url"
  git -C "$TARGET" push -u origin main --tags

Then, in this repository:

  git rm -r crates/sofuu-ui
  git submodule add "$repo_url" deps/sofuu-ui
  # Cargo.toml (two lines):
  #   [workspace] members: "crates/sofuu-ui" -> "deps/sofuu-ui"
  #   [workspace.dependencies] sofuu-ui = { path = "deps/sofuu-ui" }
  make sync-ui-js       # now pulls deps/sofuu-ui/js -> src/js + bytecode
  make ui-gates
EOF
