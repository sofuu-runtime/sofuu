// Sofuu Custom 2D GPU UI Engine — Base Metal Shading Language (MSL) Kernels
//
// Four instanced shaders covering 100% of precision 2D GPU UI & charting:
// 1. sdf_quad:   Rounded boxes, candles, pills, vertical gradients, opt-in borders & soft shadows
// 2. ribbon:     Smooth sloped trapezoid/band fills (area charts, stacked bands, oscillator spreads)
// 3. aa_stroke:  Antialiased solid & GPU-dashed vector line capsules (lines, wicks, crosshairs, grids)
// 4. sdf_text:   Resolution-independent signed-distance typography (Geist Mono R8 atlas)

#include <metal_stdlib>
using namespace metal;

struct FrameUniforms {
    float2 viewport_size;
    float2 atlas_size;
    float scale_factor;
    float time_sec;
    float2 _pad;
};

struct ClipRect {
    float x;
    float y;
    float w;
    float h;
};

struct QuadInstance {
    float4 rect;         // x, y, w, h
    float4 radii;        // tl, tr, br, bl
    uint color_top;
    uint color_bottom;
    uint border_color;
    uint shadow_color;
    float4 params;       // border_width, shadow_blur, shadow_offset_y, opacity
    uint clip_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct RibbonInstance {
    float x0;
    float y0_top;
    float y0_bot;
    float x1;
    float y1_top;
    float y1_bot;
    uint color_top;
    uint color_bottom;
    uint clip_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct StrokeInstance {
    float2 p0;
    float2 p1;
    float width;
    float feather;
    float dash_len;
    float gap_len;
    uint color;
    uint clip_id;
    uint _pad0;
    uint _pad1;
};

struct GlyphInstance {
    float2 pos;
    float2 size;
    float4 uv;
    uint color;
    float px_range;
    uint clip_id;
    uint _pad;
};

inline float4 unpack_rgba8(uint c) {
    return float4(
        float(c & 0xffu),
        float((c >> 8u) & 0xffu),
        float((c >> 16u) & 0xffu),
        float((c >> 24u) & 0xffu)
    ) / 255.0;
}

inline float2 to_ndc(float2 pos, float2 viewport) {
    return float2(
        (pos.x / max(viewport.x, 1.0)) * 2.0 - 1.0,
        1.0 - (pos.y / max(viewport.y, 1.0)) * 2.0
    );
}

inline bool clip_discard(float2 logical_pos, constant ClipRect* clips, uint clip_id) {
    ClipRect c = clips[clip_id];
    return logical_pos.x < c.x || logical_pos.x > (c.x + c.w) ||
           logical_pos.y < c.y || logical_pos.y > (c.y + c.h);
}

inline float sd_rounded_box(float2 p, float2 half_size, float4 radii) {
    float2 r_side = (p.x > 0.0) ? float2(radii.y, radii.z) : float2(radii.x, radii.w);
    float r = (p.y > 0.0) ? r_side.y : r_side.x;
    r = clamp(r, 0.0, min(half_size.x, half_size.y));
    float2 q = abs(p) - half_size + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

inline float2 unit_quad_corner(uint vid) {
    constexpr float2 corners[6] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(0.0, 1.0),
        float2(1.0, 0.0),
        float2(1.0, 1.0),
    };
    return corners[vid];
}

// ── 1. SDF Quad Shader ──────────────────────────────────────────────────────────────────

struct QuadOut {
    float4 position [[position]];
    float2 logical_pos;
    float4 rect;
    float4 radii;
    float4 color_top;
    float4 color_bottom;
    float4 border_color;
    float4 shadow_color;
    float4 params;
    uint clip_id [[flat]];
};

vertex QuadOut quad_vert(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant FrameUniforms& u [[buffer(0)]],
    const device QuadInstance* quads [[buffer(1)]]
) {
    QuadInstance q = quads[iid];
    float blur = max(q.params.y, 0.0);
    float off_y = q.params.z;
    float margin = (blur > 0.01 ? blur + abs(off_y) : 0.0) + 1.5;

    float2 corner = unit_quad_corner(vid);
    float2 origin = q.rect.xy - float2(margin);
    float2 extent = q.rect.zw + float2(margin * 2.0);
    float2 pos = origin + corner * extent;

    QuadOut out;
    out.position = float4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    out.logical_pos = pos;
    out.rect = q.rect;
    out.radii = q.radii;
    out.color_top = unpack_rgba8(q.color_top);
    out.color_bottom = unpack_rgba8(q.color_bottom);
    out.border_color = unpack_rgba8(q.border_color);
    out.shadow_color = unpack_rgba8(q.shadow_color);
    out.params = q.params;
    out.clip_id = q.clip_id;
    return out;
}

fragment float4 quad_frag(
    QuadOut in [[stage_in]],
    constant FrameUniforms& u [[buffer(0)]],
    constant ClipRect* clips [[buffer(1)]]
) {
    if (clip_discard(in.logical_pos, clips, in.clip_id)) {
        discard_fragment();
    }

    float2 half_size = in.rect.zw * 0.5;
    float2 center = in.rect.xy + half_size;
    float2 p = in.logical_pos - center;

    float aa = 0.5 / max(u.scale_factor, 1.0);
    float border_w = max(in.params.x, 0.0);
    float shadow_blur = max(in.params.y, 0.0);
    float shadow_off_y = in.params.z;
    float opacity = clamp(in.params.w, 0.0, 1.0);

    float dist = sd_rounded_box(p, half_size, in.radii);

    float4 accum = float4(0.0);
    if (shadow_blur > 0.01 && in.shadow_color.a > 0.001) {
        float sd_shadow = sd_rounded_box(p - float2(0.0, shadow_off_y), half_size, in.radii);
        float sh = 1.0 - smoothstep(-shadow_blur * 0.25, shadow_blur, sd_shadow);
        float sh_a = in.shadow_color.a * sh * sh * opacity;
        accum = float4(in.shadow_color.rgb * sh_a, sh_a);
    }

    float fill_cov = 1.0 - smoothstep(-aa, aa, dist);
    if (fill_cov > 0.001) {
        float grad_t = clamp((in.logical_pos.y - in.rect.y) / max(in.rect.w, 1.0), 0.0, 1.0);
        float4 fill = mix(in.color_top, in.color_bottom, grad_t);

        if (border_w > 0.01 && in.border_color.a > 0.001) {
            float inner_cov = 1.0 - smoothstep(-aa, aa, dist + border_w);
            float border_cov = clamp(fill_cov - inner_cov, 0.0, 1.0);
            float b_a = in.border_color.a * border_cov;
            fill.rgb = mix(fill.rgb, in.border_color.rgb, b_a);
            fill.a = max(fill.a * fill_cov, b_a);
        } else {
            fill.a *= fill_cov;
        }

        fill.a *= opacity;
        float4 src = float4(fill.rgb * fill.a, fill.a);
        accum = src + accum * (1.0 - src.a);
    }

    if (accum.a <= 0.001) {
        discard_fragment();
    }
    return accum;
}

// ── 2. Smooth Sloped Trapezoid / Ribbon Shader ──────────────────────────────────────────

struct RibbonOut {
    float4 position [[position]];
    float2 logical_pos;
    float v_t;
    float4 color_top;
    float4 color_bottom;
    uint clip_id [[flat]];
};

vertex RibbonOut ribbon_vert(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant FrameUniforms& u [[buffer(0)]],
    const device RibbonInstance* ribbons [[buffer(1)]]
) {
    RibbonInstance r = ribbons[iid];
    float2 corner = unit_quad_corner(vid);
    float x = mix(r.x0, r.x1, corner.x);
    float y_top = mix(r.y0_top, r.y1_top, corner.x);
    float y_bot = mix(r.y0_bot, r.y1_bot, corner.x);
    float y = mix(y_top, y_bot, corner.y);

    RibbonOut out;
    out.position = float4(to_ndc(float2(x, y), u.viewport_size), 0.0, 1.0);
    out.logical_pos = float2(x, y);
    out.v_t = corner.y;
    out.color_top = unpack_rgba8(r.color_top);
    out.color_bottom = unpack_rgba8(r.color_bottom);
    out.clip_id = r.clip_id;
    return out;
}

fragment float4 ribbon_frag(
    RibbonOut in [[stage_in]],
    constant ClipRect* clips [[buffer(1)]]
) {
    if (clip_discard(in.logical_pos, clips, in.clip_id)) {
        discard_fragment();
    }
    float4 col = mix(in.color_top, in.color_bottom, clamp(in.v_t, 0.0, 1.0));
    if (col.a <= 0.001) {
        discard_fragment();
    }
    return float4(col.rgb * col.a, col.a);
}

// ── 3. Antialiased Solid & Dashed Stroke Capsule Shader ─────────────────────────────────

struct StrokeOut {
    float4 position [[position]];
    float2 logical_pos;
    float2 p0;
    float2 p1;
    float half_w;
    float feather;
    float dash_len;
    float gap_len;
    float4 color;
    uint clip_id [[flat]];
};

vertex StrokeOut stroke_vert(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant FrameUniforms& u [[buffer(0)]],
    const device StrokeInstance* strokes [[buffer(1)]]
) {
    StrokeInstance s = strokes[iid];
    float half_w = max(s.width * 0.5, 0.35);
    float pad = half_w + max(s.feather, 1.0) + 2.0;

    float2 bmin = min(s.p0, s.p1) - float2(pad);
    float2 bmax = max(s.p0, s.p1) + float2(pad);
    float2 corner = unit_quad_corner(vid);
    float2 pos = mix(bmin, bmax, corner);

    StrokeOut out;
    out.position = float4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    out.logical_pos = pos;
    out.p0 = s.p0;
    out.p1 = s.p1;
    out.half_w = half_w;
    out.feather = max(s.feather, 0.75 / max(u.scale_factor, 1.0));
    out.dash_len = s.dash_len;
    out.gap_len = s.gap_len;
    out.color = unpack_rgba8(s.color);
    out.clip_id = s.clip_id;
    return out;
}

fragment float4 stroke_frag(
    StrokeOut in [[stage_in]],
    constant ClipRect* clips [[buffer(1)]]
) {
    if (clip_discard(in.logical_pos, clips, in.clip_id)) {
        discard_fragment();
    }
    float2 pa = in.logical_pos - in.p0;
    float2 ba = in.p1 - in.p0;
    float len_sq = max(dot(ba, ba), 1e-6);
    float h = clamp(dot(pa, ba) / len_sq, 0.0, 1.0);
    float d = length(pa - ba * h) - in.half_w;

    float cov = 1.0 - smoothstep(-in.feather, in.feather, d);
    if (in.dash_len > 0.01 && in.gap_len > 0.01) {
        float seg_len = sqrt(len_sq);
        float along = h * seg_len;
        float period = in.dash_len + in.gap_len;
        float phase = fmod(along, period);
        float dash_cov = smoothstep(0.0, 0.75, phase) *
                         (1.0 - smoothstep(in.dash_len - 0.75, in.dash_len + 0.25, phase));
        cov *= dash_cov;
    }

    float a = in.color.a * cov;
    if (a <= 0.001) {
        discard_fragment();
    }
    return float4(in.color.rgb * a, a);
}

// ── 4. SDF Text Shader ──────────────────────────────────────────────────────────────────

struct GlyphOut {
    float4 position [[position]];
    float2 logical_pos;
    float2 uv;
    float4 color;
    float px_range;
    uint clip_id [[flat]];
};

vertex GlyphOut glyph_vert(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant FrameUniforms& u [[buffer(0)]],
    const device GlyphInstance* glyphs [[buffer(1)]]
) {
    GlyphInstance g = glyphs[iid];
    float2 corner = unit_quad_corner(vid);
    float2 pos = g.pos + corner * g.size;
    float2 uv = float2(
        mix(g.uv.x, g.uv.z, corner.x),
        mix(g.uv.y, g.uv.w, corner.y)
    );

    GlyphOut out;
    out.position = float4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    out.logical_pos = pos;
    out.uv = uv;
    out.color = unpack_rgba8(g.color);
    out.px_range = g.px_range;
    out.clip_id = g.clip_id;
    return out;
}

fragment float4 glyph_frag(
    GlyphOut in [[stage_in]],
    constant FrameUniforms& u [[buffer(0)]],
    constant ClipRect* clips [[buffer(1)]],
    texture2d<float> atlas [[texture(0)]],
    sampler atlas_sampler [[sampler(0)]]
) {
    if (clip_discard(in.logical_pos, clips, in.clip_id)) {
        discard_fragment();
    }
    float sd = atlas.sample(atlas_sampler, in.uv).r - 0.5;
    float2 unit_range = float2(in.px_range) / max(u.atlas_size, float2(1.0));
    float2 screen_tex_size = float2(1.0) / max(fwidth(in.uv), float2(1e-5));
    float screen_px_range = max(0.5 * dot(unit_range, screen_tex_size), 1.0);
    // Natural contour-centered Regular-weight SDF AA without stroke bloat or low-opacity fading.
    float px_dist = sd * screen_px_range;
    float cov = smoothstep(-0.48, 0.44, px_dist);
    float alpha = cov * in.color.a;
    if (alpha <= 0.002) {
        discard_fragment();
    }
    return float4(in.color.rgb * alpha, alpha);
}
