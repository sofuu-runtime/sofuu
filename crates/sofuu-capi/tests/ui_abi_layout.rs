//! Phase C1 gate — C header / Rust `#[repr(C)]` layout parity.
//!
//! The host compiles `ui_shaders.metal` (or GLSL/SPIR-V) once and uploads the
//! session's flat arrays directly, so a single field drifting by four bytes is
//! a corrupted frame, not a compile error. This test pins:
//!
//!   1. every `SofuuUi*` Rust definition to the exact size/alignment/offset,
//!   2. the C header to the same numbers through real `_Static_assert`s,
//!   3. the header compiling cleanly under `-Wall -Wextra -Werror -pedantic`.
//!
//! It runs on whatever host `cargo test` targets; the 32-bit offset table is
//! included so a future i686/armv7 runner keeps the same guarantee.

use std::mem::{align_of, offset_of, size_of};
use std::path::PathBuf;
use std::process::Command;

use sofuu_capi::ui::{
    SofuuUiClipRect, SofuuUiDrawCmd, SofuuUiDrawList, SofuuUiFrameUniforms, SofuuUiGlyphInstance,
    SofuuUiInputEvent, SofuuUiQuadInstance, SofuuUiRibbonInstance, SofuuUiStrokeInstance,
};

#[test]
fn rust_structs_match_sofuu_ui_draw_layout() {
    // Instance structs are type aliases of sofuu-ui::draw, so these also prove
    // the public Rust types and the C typedefs describe the same bytes.
    assert_eq!(size_of::<SofuuUiFrameUniforms>(), 32);
    assert_eq!(size_of::<SofuuUiClipRect>(), 16);
    assert_eq!(size_of::<SofuuUiDrawCmd>(), 16);
    assert_eq!(size_of::<SofuuUiQuadInstance>(), 80);
    assert_eq!(size_of::<SofuuUiRibbonInstance>(), 48);
    assert_eq!(size_of::<SofuuUiStrokeInstance>(), 48);
    assert_eq!(size_of::<SofuuUiGlyphInstance>(), 48);

    assert_eq!(offset_of!(SofuuUiQuadInstance, rect), 0);
    assert_eq!(offset_of!(SofuuUiQuadInstance, radii), 16);
    assert_eq!(offset_of!(SofuuUiQuadInstance, color_top), 32);
    assert_eq!(offset_of!(SofuuUiQuadInstance, params), 48);
    assert_eq!(offset_of!(SofuuUiQuadInstance, clip_id), 64);
    assert_eq!(offset_of!(SofuuUiQuadInstance, _pad), 68);

    assert_eq!(offset_of!(SofuuUiRibbonInstance, color_top), 24);
    assert_eq!(offset_of!(SofuuUiRibbonInstance, clip_id), 32);

    assert_eq!(offset_of!(SofuuUiStrokeInstance, p0), 0);
    assert_eq!(offset_of!(SofuuUiStrokeInstance, p1), 8);
    assert_eq!(offset_of!(SofuuUiStrokeInstance, width), 16);
    assert_eq!(offset_of!(SofuuUiStrokeInstance, color), 32);
    assert_eq!(offset_of!(SofuuUiStrokeInstance, clip_id), 36);

    assert_eq!(offset_of!(SofuuUiGlyphInstance, pos), 0);
    assert_eq!(offset_of!(SofuuUiGlyphInstance, size), 8);
    assert_eq!(offset_of!(SofuuUiGlyphInstance, uv), 16);
    assert_eq!(offset_of!(SofuuUiGlyphInstance, color), 32);
    assert_eq!(offset_of!(SofuuUiGlyphInstance, px_range), 36);
    assert_eq!(offset_of!(SofuuUiGlyphInstance, clip_id), 40);

    // Self-describing frame header. The exact pointer offsets depend on the
    // host word size; both tables are pinned so CI on either width is honest.
    assert_eq!(offset_of!(SofuuUiDrawList, uniforms), 0);
    if size_of::<usize>() == 8 {
        assert_eq!(size_of::<SofuuUiDrawList>(), 128);
        assert_eq!(align_of::<SofuuUiDrawList>(), 8);
        assert_eq!(offset_of!(SofuuUiDrawList, quads), 32);
        assert_eq!(offset_of!(SofuuUiDrawList, quad_count), 40);
        assert_eq!(offset_of!(SofuuUiDrawList, ribbons), 48);
        assert_eq!(offset_of!(SofuuUiDrawList, ribbon_count), 56);
        assert_eq!(offset_of!(SofuuUiDrawList, strokes), 64);
        assert_eq!(offset_of!(SofuuUiDrawList, stroke_count), 72);
        assert_eq!(offset_of!(SofuuUiDrawList, glyphs), 80);
        assert_eq!(offset_of!(SofuuUiDrawList, glyph_count), 88);
        assert_eq!(offset_of!(SofuuUiDrawList, cmds), 96);
        assert_eq!(offset_of!(SofuuUiDrawList, cmd_count), 104);
        assert_eq!(offset_of!(SofuuUiDrawList, clips), 112);
        assert_eq!(offset_of!(SofuuUiDrawList, clip_count), 120);
        assert_eq!(offset_of!(SofuuUiDrawList, recompiled), 124);
        assert_eq!(offset_of!(SofuuUiDrawList, needs_next_frame), 125);
        assert_eq!(offset_of!(SofuuUiDrawList, theme_is_dark), 126);
        assert_eq!(offset_of!(SofuuUiDrawList, _reserved), 127);
    } else {
        assert_eq!(size_of::<SofuuUiDrawList>(), 84);
        assert_eq!(align_of::<SofuuUiDrawList>(), 4);
        assert_eq!(offset_of!(SofuuUiDrawList, quads), 32);
        assert_eq!(offset_of!(SofuuUiDrawList, quad_count), 36);
        assert_eq!(offset_of!(SofuuUiDrawList, clips), 72);
        assert_eq!(offset_of!(SofuuUiDrawList, clip_count), 76);
        assert_eq!(offset_of!(SofuuUiDrawList, recompiled), 80);
    }

    // Input event: six f32/u32 lanes, then the borrowed text pointer.
    assert_eq!(offset_of!(SofuuUiInputEvent, kind), 0);
    assert_eq!(offset_of!(SofuuUiInputEvent, x), 4);
    assert_eq!(offset_of!(SofuuUiInputEvent, scale_factor), 20);
    assert_eq!(offset_of!(SofuuUiInputEvent, button), 24);
    assert_eq!(offset_of!(SofuuUiInputEvent, modifiers), 28);
    assert_eq!(offset_of!(SofuuUiInputEvent, text_utf8), 32);
    assert_eq!(size_of::<SofuuUiInputEvent>(), if size_of::<usize>() == 8 { 40 } else { 36 });
}

#[test]
fn c_header_compiles_and_static_asserts_the_same_layout() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include_dir = manifest.join("../../include");
    let header = include_dir.join("sofuu_embed.h");
    assert!(
        header.is_file(),
        "header not found at {} (run from the repo workspace)",
        header.display()
    );

    let source = r#"
#include <stddef.h>
#include <stdint.h>
#include "sofuu_embed.h"

#define SOFUU_SA(cond, msg) _Static_assert((cond), msg)

SOFUU_SA(sizeof(SofuuUiFrameUniforms) == 32, "uniforms");
SOFUU_SA(sizeof(SofuuUiClipRect) == 16, "clip");
SOFUU_SA(sizeof(SofuuUiDrawCmd) == 16, "cmd");
SOFUU_SA(sizeof(SofuuUiQuadInstance) == 80, "quad");
SOFUU_SA(sizeof(SofuuUiRibbonInstance) == 48, "ribbon");
SOFUU_SA(sizeof(SofuuUiStrokeInstance) == 48, "stroke");
SOFUU_SA(sizeof(SofuuUiGlyphInstance) == 48, "glyph");

SOFUU_SA(offsetof(SofuuUiQuadInstance, rect) == 0, "quad.rect");
SOFUU_SA(offsetof(SofuuUiQuadInstance, radii) == 16, "quad.radii");
SOFUU_SA(offsetof(SofuuUiQuadInstance, color_top) == 32, "quad.color_top");
SOFUU_SA(offsetof(SofuuUiQuadInstance, params) == 48, "quad.params");
SOFUU_SA(offsetof(SofuuUiQuadInstance, clip_id) == 64, "quad.clip_id");
SOFUU_SA(offsetof(SofuuUiQuadInstance, _pad) == 68, "quad._pad");
SOFUU_SA(offsetof(SofuuUiRibbonInstance, color_top) == 24, "ribbon.color_top");
SOFUU_SA(offsetof(SofuuUiRibbonInstance, clip_id) == 32, "ribbon.clip_id");
SOFUU_SA(offsetof(SofuuUiStrokeInstance, p1) == 8, "stroke.p1");
SOFUU_SA(offsetof(SofuuUiStrokeInstance, width) == 16, "stroke.width");
SOFUU_SA(offsetof(SofuuUiStrokeInstance, color) == 32, "stroke.color");
SOFUU_SA(offsetof(SofuuUiGlyphInstance, uv) == 16, "glyph.uv");
SOFUU_SA(offsetof(SofuuUiGlyphInstance, px_range) == 36, "glyph.px_range");
SOFUU_SA(offsetof(SofuuUiGlyphInstance, clip_id) == 40, "glyph.clip_id");

#if UINTPTR_MAX == 0xffffffffffffffffULL
SOFUU_SA(sizeof(SofuuUiDrawList) == 128, "drawlist-64");
SOFUU_SA(offsetof(SofuuUiDrawList, quads) == 32, "dl.quads");
SOFUU_SA(offsetof(SofuuUiDrawList, quad_count) == 40, "dl.quad_count");
SOFUU_SA(offsetof(SofuuUiDrawList, ribbons) == 48, "dl.ribbons");
SOFUU_SA(offsetof(SofuuUiDrawList, strokes) == 64, "dl.strokes");
SOFUU_SA(offsetof(SofuuUiDrawList, glyphs) == 80, "dl.glyphs");
SOFUU_SA(offsetof(SofuuUiDrawList, cmds) == 96, "dl.cmds");
SOFUU_SA(offsetof(SofuuUiDrawList, clips) == 112, "dl.clips");
SOFUU_SA(offsetof(SofuuUiDrawList, recompiled) == 124, "dl.recompiled");
SOFUU_SA(offsetof(SofuuUiDrawList, needs_next_frame) == 125, "dl.needs");
SOFUU_SA(sizeof(SofuuUiInputEvent) == 40, "input-64");
#else
SOFUU_SA(sizeof(SofuuUiDrawList) == 84, "drawlist-32");
SOFUU_SA(offsetof(SofuuUiDrawList, quads) == 32, "dl.quads32");
SOFUU_SA(offsetof(SofuuUiDrawList, quad_count) == 36, "dl.quad_count32");
SOFUU_SA(offsetof(SofuuUiDrawList, clips) == 72, "dl.clips32");
SOFUU_SA(offsetof(SofuuUiDrawList, recompiled) == 80, "dl.recompiled32");
SOFUU_SA(sizeof(SofuuUiInputEvent) == 36, "input-32");
#endif

/* Compile every prototype: a signature drift is a C compile error. */
int probe_calls(SofuuRuntime *rt, SofuuUiSession *ui,
                SofuuUiDrawList *dl, const SofuuUiInputEvent *evt) {
    char *err = 0;
    const uint8_t *pixels = 0;
    uint32_t w = 0, h = 0;
    float range = 0.0f;
    SofuuUiSession *created = sofuu_ui_session_new(0, &err);
    (void)created;
    sofuu_ui_session_free(ui);
    (void)sofuu_ui_session_set_wake_cb(ui, 0, 0);
    (void)sofuu_ui_session_set_safe_insets(ui, 0.0f, 0.0f);
    (void)sofuu_ui_session_get_atlas(ui, &pixels, &w, &h, &range);
    (void)sofuu_ui_attach_runtime(rt, ui);
    (void)sofuu_ui_detach_runtime(rt);
    (void)sofuu_ui_push_patch_json(ui, "{}", 2);
    (void)sofuu_ui_ingest_agent_event_json(ui, "{}");
    (void)sofuu_ui_dispatch_input(ui, evt);
    (void)sofuu_ui_poll_event_json(ui, &err);
    (void)sofuu_ui_poll_draw_list(ui, 0.016f, 800.0f, 600.0f, 2.0f, dl);
    (void)sofuu_ui_surface_attach_metal(ui, 0, 0);
    (void)sofuu_ui_surface_attach_egl(ui, 0, 0, 0);
    (void)sofuu_ui_surface_render(ui, 0.016f, 800u, 600u, 2.0f);
    (void)sofuu_ui_surface_detach(ui);
    return (pixels != 0) || (w + h == 0) || (range == 0.0f);
}

int main(void) { return 0; }
"#;

    let tmp = std::env::temp_dir().join(format!("sofuu_ui_abi_{}.c", std::process::id()));
    std::fs::write(&tmp, source).expect("write C layout probe");

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let output = Command::new(&cc)
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-pedantic")
        .arg("-fsyntax-only")
        .arg("-I")
        .arg(&include_dir)
        .arg(&tmp)
        .output();

    let _ = std::fs::remove_file(&tmp);

    let output = match output {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("skipping C header gate: compiler `{cc}` not found");
            return;
        }
        Err(e) => panic!("failed to run {cc}: {e}"),
    };

    if !output.status.success() {
        panic!(
            "C header failed to compile under {cc} -std=c11 -Wall -Wextra -Werror -pedantic:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
