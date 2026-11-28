//! Media primitives: crisp SDF icons and textured bitmap sprites.
//!
//! Icons sample the process-wide [`IconAtlas`] (`tex_id = 0`) through the sprite
//! pipeline with `SPRITE_FLAG_SDF`, so tinting and scaling happen entirely in the
//! fragment shader. Images sample a decoded [`crate::images::ImageAtlas`] page
//! (`tex_id >= 1`) and support cover/contain/fill/scale-down crops plus rounded
//! avatar clipping without any CPU copy.

use crate::draw::{DrawList, QuadInstance, SpriteInstance, SPRITE_FLAG_SDF};
use crate::icons::IconAtlas;
use crate::images::{fit_image, ImageFit, ImageHandle};
use crate::layout::ComputedRect;
use crate::theme::Theme;

/// Emit one vector icon centered inside `rect`.
///
/// Returns `false` when the icon name is not registered (callers can then draw
/// a placeholder instead of leaving a hole).
pub fn emit_icon(
    icons: &IconAtlas,
    name: &str,
    rect: ComputedRect,
    size: f32,
    color: u32,
    dl: &mut DrawList,
) -> bool {
    let Some(uv) = icons.uv(name) else {
        return false;
    };
    if rect.w <= 0.0 || rect.h <= 0.0 {
        return false;
    }
    let s = size.max(6.0).min(rect.w).min(rect.h);
    let x = rect.x + (rect.w - s) * 0.5;
    let y = rect.y + (rect.h - s) * 0.5;
    let tint = if color == 0 { 0xffff_ffff } else { color };
    dl.push_sprite(
        SpriteInstance::new([x, y, s, s], uv)
            .with_tint(tint)
            .with_tex(crate::images::ICON_TEX_ID)
            .with_flags(SPRITE_FLAG_SDF),
    );
    true
}

/// Emit a decoded image sprite, or a shimmer/placeholder when it is still loading.
#[allow(clippy::too_many_arguments)]
pub fn emit_image(
    handle: Option<ImageHandle>,
    rect: ComputedRect,
    fit: ImageFit,
    radii: [f32; 4],
    tint: Option<u32>,
    placeholder: Option<u32>,
    theme: &Theme,
    dl: &mut DrawList,
) -> bool {
    if rect.w <= 0.0 || rect.h <= 0.0 {
        return false;
    }
    let Some(handle) = handle else {
        let base = placeholder.unwrap_or(theme.surface_2);
        dl.push_quad(QuadInstance {
            rect: [rect.x, rect.y, rect.w, rect.h],
            radii,
            color_top: base,
            color_bottom: crate::draw::scale_alpha(base, 0.72),
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
        return false;
    };
    let fitted = fit_image(
        fit,
        [rect.x, rect.y, rect.w, rect.h],
        handle.width,
        handle.height,
    );
    let [u0, v0, u1, v1] = handle.uv;
    let c = fitted.uv;
    let uv = [
        u0 + (u1 - u0) * c[0],
        v0 + (v1 - v0) * c[1],
        u0 + (u1 - u0) * c[2],
        v0 + (v1 - v0) * c[3],
    ];
    dl.push_sprite(
        SpriteInstance::new(fitted.rect, uv)
            .with_radii(radii)
            .with_tint(tint.unwrap_or(0xffff_ffff))
            .with_tex(handle.tex_id),
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::{DecodedImage, ImageAtlas};

    fn rect() -> ComputedRect {
        ComputedRect {
            x: 10.0,
            y: 20.0,
            w: 40.0,
            h: 30.0,
            content_h: 30.0,
            folded: false,
        }
    }

    #[test]
    fn icons_emit_sdf_sprites_with_tint_and_centering() {
        let icons = IconAtlas::builtin();
        let mut dl = DrawList::new();
        assert!(emit_icon(&icons, "search", rect(), 16.0, 0xff00ff00, &mut dl));
        assert_eq!(dl.sprites.len(), 1);
        let s = dl.sprites[0];
        assert_eq!(s.flags, SPRITE_FLAG_SDF);
        assert_eq!(s.tex_id, 0);
        assert_eq!(s.tint, 0xff00ff00);
        // Square, centered inside the 40x30 rect.
        assert_eq!(s.rect[2], 16.0);
        assert_eq!(s.rect[3], 16.0);
        assert!((s.rect[0] - 22.0).abs() < 0.001);
        assert!((s.rect[1] - 27.0).abs() < 0.001);
        assert!(!emit_icon(&icons, "definitely-not-an-icon", rect(), 16.0, 0, &mut dl));
    }

    #[test]
    fn missing_image_emits_placeholder_and_present_image_emits_sprite() {
        let theme = Theme::dark();
        let mut dl = DrawList::new();
        assert!(!emit_image(
            None,
            rect(),
            ImageFit::Cover,
            [8.0; 4],
            None,
            Some(0xff334455),
            &theme,
            &mut dl
        ));
        assert_eq!(dl.quads.len(), 1);
        assert_eq!(dl.quads[0].radii, [8.0; 4]);

        let mut atlas = ImageAtlas::with_page_size(64, 1);
        let handle = atlas
            .insert(1, DecodedImage::new(8, 8, vec![255; 8 * 8 * 4]).unwrap())
            .unwrap();
        assert!(emit_image(
            Some(handle),
            rect(),
            ImageFit::Contain,
            [4.0; 4],
            Some(0xffffffff),
            None,
            &theme,
            &mut dl
        ));
        assert_eq!(dl.sprites.len(), 1);
        assert_eq!(dl.sprites[0].tex_id, crate::images::FIRST_IMAGE_TEX_ID);
        // Contain in a 4:3 box for a square source -> letterboxed to 30x30.
        assert!((dl.sprites[0].rect[2] - 30.0).abs() < 0.001);
        assert!((dl.sprites[0].rect[0] - 15.0).abs() < 0.001);
    }
}
