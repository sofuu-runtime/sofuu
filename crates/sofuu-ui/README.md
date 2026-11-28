# sofuu-ui

Domain-agnostic 2D GPU UI compositor core: Scene IR, responsive layout,
springs, SDF typography, icons/images, effects choreography, accessibility
semantics, geometric audit and a flat `#[repr(C)]` `DrawList`.

This crate is deliberately free of OS windowing, GPU drivers, Kaze/QuickJS and
network code. `sofuu-gpu` (Metal/GL/GLES/Vulkan hosts), `sofuu-capi` (the C
ABI) and `sofuu-emu` (the GPU Platform UI Emulator) consume it.

## Layout

| Path | Contract |
|---|---|
| `src/draw.rs` | Instance structs, draw-kind ids, `SOFUU_DRAW_ABI_VERSION` |
| `src/shaders.rs` | Canonical MSL/GLSL sources bundled with `include_str!` |
| `shaders/` | `msl/ui_shaders.metal`, `glsl/ui_shaders_{330,450,es3}.glsl` |
| `src/ir.rs`, `src/layout.rs` | Scene IR, responsive solver, layout hints |
| `src/text.rs`, `assets/` | SFD font atlas parsing, rich text, measurement |
| `src/compiler.rs` | `UiEngine`, patch application, frame scheduling |
| `src/audit.rs`, `src/a11y.rs` | Geometric audit and semantic tree |
| `src/fx.rs` | `sofuu.fx` motion/effect descriptors |
| `js/ui.js`, `js/ui_kit.js` | Declarative `sofuu.ui` / `sofuu.fx` surface |
| `js/index.d.ts` | UI/FX types as a module for standalone consumers |
| `tests/` | Pure wire-parity tests + JSON fixtures |

## ABI rule

`src/draw.rs` and `shaders/` are one contract. A field insertion, a changed
`DRAW_KIND_*` id or a shader uniform change must:

1. bump `SOFUU_DRAW_ABI_VERSION`,
2. update `EXPECTED_GPU_DRAW_ABI_VERSION` in `crates/sofuu-gpu/src/lib.rs`,
3. update the `SOFUU_UI_DRAW_ABI_VERSION` macro in `include/sofuu_embed.h`,
4. keep `scripts/check_ui_shaders.sh` green.

The Rust side pins this with a `const _: () = assert!(...)`, so a mismatch is a
`cargo check` failure, never corrupted geometry.

## Tests

```bash
cargo test                 # unit + wire parity, no Kaze/GPU dependency
bash ../../scripts/check_ui_shaders.sh   # Metal/GLSL/SPIR-V syntax
```

The crate is split-ready: nothing under `tests/` reads outside the crate. When
the standalone repository is cut over, `../../src/js/` remains a generated copy
for the host's headless build (Option A in the migration plan).
