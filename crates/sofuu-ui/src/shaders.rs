//! Canonical GPU UI shader sources.
//!
//! The draw structs in [`crate::draw`] and the shader sources here are one
//! contract. A `#[repr(C)]` layout change (or a draw-kind renumbering) needs a
//! matching shader change and a [`crate::draw::SOFUU_DRAW_ABI_VERSION`] bump.
//!
//! Hosts consume these constants instead of keeping a second copy:
//! `sofuu-gpu` uploads them to Metal/GL/GLES, and the Vulkan SPIR-V generator
//! compiles [`GLSL_450_SRC`]. Keeping the source next to the instance structs
//! is what makes the split boundary auditable: the external `sofuu-ui` crate
//! ships both the layout and the shaders that read it.
//!
//! The compiled Vulkan SPIR-V modules deliberately stay in `sofuu-gpu`
//! (`src/shaders/spirv/`): they are generated artifacts for one backend, not a
//! second source of truth.

/// Metal Shading Language source for the quad, ribbon, stroke, glyph, effect,
/// particle and backdrop-blur pipelines.
pub const METAL_SRC: &str = include_str!("../shaders/msl/ui_shaders.metal");

/// [`METAL_SRC`] with a trailing NUL, ready for `newLibraryWithSource:`.
pub const METAL_SRC_NUL: &str = concat!(include_str!("../shaders/msl/ui_shaders.metal"), "\0");

/// OpenGL 3.3 core source (desktop Linux / Windows).
pub const GLSL_330_SRC: &str = include_str!("../shaders/glsl/ui_shaders_330.glsl");

/// OpenGL 4.5 source; also the input to
/// `sofuu-gpu/src/shaders/generate_spirv.sh`.
pub const GLSL_450_SRC: &str = include_str!("../shaders/glsl/ui_shaders_450.glsl");

/// OpenGL ES 3.0 source (Android, and the GL fallback on Apple platforms).
pub const GLSL_ES3_SRC: &str = include_str!("../shaders/glsl/ui_shaders_es3.glsl");

/// Every bundled source with the path a compile error should name.
pub const SOURCES: [(&str, &str); 4] = [
    ("msl/ui_shaders.metal", METAL_SRC),
    ("glsl/ui_shaders_330.glsl", GLSL_330_SRC),
    ("glsl/ui_shaders_450.glsl", GLSL_450_SRC),
    ("glsl/ui_shaders_es3.glsl", GLSL_ES3_SRC),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shader_source_is_bundled_and_nonempty() {
        for (name, src) in SOURCES {
            assert!(src.len() > 1024, "{name} is suspiciously short");
            assert!(
                src.contains("FrameUniforms") || src.contains("uniform"),
                "{name} lost its frame uniform block"
            );
        }
        assert!(METAL_SRC_NUL.ends_with('\0'));
        assert_eq!(&METAL_SRC_NUL[..METAL_SRC.len()], METAL_SRC);
    }

    /// The shader family is the other half of the draw ABI: if a source loses
    /// a pipeline the GPU host would silently draw nothing for that kind.
    #[test]
    fn every_backend_declares_every_pipeline() {
        for kernel in ["quad", "ribbon", "stroke", "glyph", "effect", "particle"] {
            assert!(
                METAL_SRC.contains(&format!("{kernel}_vert")),
                "MSL lost {kernel}_vert"
            );
            assert!(
                METAL_SRC.contains(&format!("{kernel}_frag")),
                "MSL lost {kernel}_frag"
            );
        }
        for (name, src) in [
            ("330", GLSL_330_SRC),
            ("450", GLSL_450_SRC),
            ("es3", GLSL_ES3_SRC),
        ] {
            for kind in 0..=6u32 {
                assert!(
                    src.contains(&format!("v_kind == {kind}u")),
                    "GLSL {name} lost draw kind branch {kind}"
                );
            }
        }
    }
}
