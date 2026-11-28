// Sofuu Custom 2D GPU UI Engine — Base Metal Shading Language (MSL) Kernels
//
// Four instanced pipelines cover 100% of precision 2D GPU UI & charting:
// 1. sdf_quad:   Rounded boxes, candles, pills, vertical gradients, opt-in borders & soft shadows
// 2. ribbon:     Smooth sloped trapezoid/band fills (area charts, stacked bands, oscillator spreads)
// 3. aa_stroke:  Antialiased solid & GPU-dashed vector line capsules (lines, wicks, crosshairs, grids)
// 4. sdf_text:   Resolution-independent signed-distance typography (Geist Mono R8 atlas)
//
// plus the `sofuu.fx` layer:
// 5. effect:     parameterized SDF effect quads (shimmer, ripple, border beam, cursor glow,
//                aurora, pulse ring, voice orb)
// 6. particle:   instanced confetti/spark bursts (circle, ribbon and spark shapes)
// 7. backdrop:   SDF glass compositing over a blurred snapshot (tint + specular rim)
// 8. blur:       half-resolution dual-filter Kawase downsample + compute passes

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

// ── 5. Parameterized UI Effect Quad ─────────────────────────────────────────────────────
//
// One fragment branch table covers shimmer, ripples, border beams, cursor glow, aurora
// fields, pulse rings and the amplitude-reactive voice orb. Every mode is masked by the same
// analytic rounded-rect SDF so effects never escape their node bounds.

struct EffectInstance {
    float4 rect;         // x, y, w, h
    float4 radii;        // tl, tr, br, bl
    float4 params0;
    float4 params1;
    uint color0;
    uint color1;
    uint color2;
    uint kind_and_clip;  // low 16: effect kind, high 16: clip id
};

struct EffectOut {
    float4 position [[position]];
    float2 logical_pos;
    float4 rect;
    float4 radii;
    float4 params0;
    float4 params1;
    float4 color0;
    float4 color1;
    float4 color2;
    uint kind [[flat]];
    uint clip_id [[flat]];
};

vertex EffectOut effect_vert(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant FrameUniforms& u [[buffer(0)]],
    const device EffectInstance* effects [[buffer(1)]]
) {
    EffectInstance e = effects[iid];
    float2 corner = unit_quad_corner(vid);
    float2 pos = e.rect.xy - float2(2.0) + corner * (e.rect.zw + float2(4.0));

    EffectOut out;
    out.position = float4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    out.logical_pos = pos;
    out.rect = e.rect;
    out.radii = e.radii;
    out.params0 = e.params0;
    out.params1 = e.params1;
    out.color0 = unpack_rgba8(e.color0);
    out.color1 = unpack_rgba8(e.color1);
    out.color2 = unpack_rgba8(e.color2);
    out.kind = e.kind_and_clip & 0xffffu;
    out.clip_id = e.kind_and_clip >> 16u;
    return out;
}

inline float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

inline float4 effect_shimmer(EffectOut in) {
    float2 p = in.logical_pos - in.rect.xy;
    float band = max(in.params0.x, 8.0);
    float intensity = in.params0.y;
    float phase = in.params1.x;
    float2 dir = normalize(float2(0.82, 0.57));
    float span = in.rect.z * dir.x + in.rect.w * dir.y;
    float travel = phase * (span + band * 2.0) - band;
    float d = abs(dot(p, dir) - travel) / band;
    float band_a = pow(max(1.0 - d, 0.0), 2.2);
    float sheen = 0.035 * intensity;
    float3 rgb = mix(in.color1.rgb, in.color0.rgb, band_a);
    float alpha = (band_a * in.color0.a * 0.9 + sheen) * intensity;
    return float4(rgb, alpha);
}

inline float4 effect_ripple(EffectOut in) {
    float2 origin = in.rect.xy + in.params0.xy;
    float max_r = max(in.params0.z, 1.0);
    float ring_w = max(in.params0.w, 0.6);
    float progress = clamp(in.params1.x, 0.0, 1.0);
    float r = progress * max_r;
    float dist = length(in.logical_pos - origin);
    float aa = 0.75;
    float ring = 1.0 - smoothstep(ring_w * 0.5 - aa, ring_w * 0.5 + aa, abs(dist - r));
    float fill = (1.0 - smoothstep(0.0, max(r, 1.0), dist)) * (1.0 - progress);
    float fade = pow(1.0 - progress, 1.5);
    float alpha = (ring * fade + fill * 0.10) * in.color0.a;
    float3 rgb = mix(in.color0.rgb, in.color1.rgb, fill * 0.6);
    return float4(rgb, alpha);
}

inline float4 effect_beam(EffectOut in) {
    float2 center = in.rect.xy + in.rect.zw * 0.5;
    float2 p = in.logical_pos - center;
    float half_w = in.rect.z * 0.5;
    float half_h = in.rect.w * 0.5;
    float dist = sd_rounded_box(p, float2(half_w, half_h), in.radii);
    float width = max(in.params0.y, 1.0);
    float border = 1.0 - smoothstep(0.0, width, -dist);
    float angle = atan2(p.y, p.x);
    float time = in.params1.x * in.params0.x;
    float comet = pow(max(cos(angle - time * 2.0), 0.0), 10.0);
    float alpha = border * (0.22 + 0.78 * comet) * in.color0.a * in.params0.z;
    float3 rgb = mix(in.color2.rgb, mix(in.color0.rgb, in.color1.rgb, comet), 0.85);
    return float4(rgb, alpha);
}

inline float4 effect_cursor_glow(EffectOut in) {
    float2 local = in.logical_pos - in.rect.xy - in.params0.xy;
    float radius = max(in.params0.z, 1.0);
    float d = length(local) / radius;
    float glow = exp(-d * d * 2.2) * in.params0.w;
    float2 center = in.rect.xy + in.rect.zw * 0.5;
    float sd = sd_rounded_box(in.logical_pos - center, in.rect.zw * 0.5, in.radii);
    float rim = (1.0 - smoothstep(0.0, 1.5, abs(sd))) * 0.35;
    float alpha = (glow + rim) * in.color0.a;
    float3 rgb = mix(in.color1.rgb, in.color0.rgb, clamp(1.0 - d * 0.8, 0.0, 1.0));
    return float4(rgb, alpha);
}

inline float4 effect_aurora(EffectOut in) {
    float2 uv = (in.logical_pos - in.rect.xy) / max(in.rect.zw, float2(1.0));
    float t = in.params1.x * 0.22 + in.params0.x;
    float intensity = in.params0.y;
    float2 c0 = float2(0.25, 0.30) + 0.18 * float2(sin(t * 0.9), cos(t * 1.13));
    float2 c1 = float2(0.75, 0.32) + 0.20 * float2(cos(t * 0.7 + 2.0), sin(t * 1.07));
    float2 c2 = float2(0.45, 0.78) + 0.22 * float2(sin(t * 1.21 + 4.0), cos(t * 0.8 + 1.0));
    float2 c3 = float2(0.70, 0.70) + 0.16 * float2(cos(t * 1.4 + 1.0), sin(t * 0.6 + 3.0));
    float f0 = exp(-dot(uv - c0, uv - c0) * 7.0);
    float f1 = exp(-dot(uv - c1, uv - c1) * 7.0);
    float f2 = exp(-dot(uv - c2, uv - c2) * 8.0);
    float f3 = exp(-dot(uv - c3, uv - c3) * 10.0);
    float field = clamp((f0 + f1 + f2 + f3) * 0.9, 0.0, 1.5);
    float3 rgb = in.color0.rgb * f0 + in.color1.rgb * f1 + in.color2.rgb * f2 +
                 mix(in.color1.rgb, in.color2.rgb, 0.5) * f3;
    rgb /= max(f0 + f1 + f2 + f3, 0.001);
    rgb = mix(in.color2.rgb, rgb, clamp(field * 0.8, 0.0, 1.0));
    float dither = (hash21(floor(in.logical_pos * 1.5)) - 0.5) * 0.008;
    float alpha = clamp(field * 0.55, 0.0, 1.0) * in.color0.a * intensity;
    return float4(max(rgb + dither, float3(0.0)), alpha);
}

inline float4 effect_pulse_ring(EffectOut in) {
    float2 center = in.rect.xy + in.rect.zw * 0.5;
    float progress = clamp(in.params0.x, 0.0, 1.0);
    float max_r = max(in.params0.y, 1.0);
    float width = max(in.params0.z, 0.8);
    float dist = length(in.logical_pos - center);
    float ring = 1.0 - smoothstep(width * 0.5, width * 0.5 + 1.0, abs(dist - progress * max_r));
    float fade = pow(1.0 - progress, 1.3);
    float alpha = ring * fade * in.color0.a * in.params0.w;
    float3 rgb = mix(in.color0.rgb, in.color1.rgb, progress);
    return float4(rgb, alpha);
}

inline float4 effect_voice_orb(EffectOut in) {
    float2 half_size = in.rect.zw * 0.5;
    float2 center = in.rect.xy + half_size;
    float2 uv = (in.logical_pos - center) / max(min(half_size.x, half_size.y), 1.0);
    float amp = clamp(in.params0.x, 0.0, 1.0);
    float phase = in.params0.y;
    float angle = atan2(uv.y, uv.x);
    float r = length(uv);
    float wobble = 0.055 * sin(angle * 3.0 + phase * 1.9) +
                   0.035 * sin(angle * 5.0 - phase * 1.3) +
                   amp * 0.085 * sin(angle * 2.0 + phase * 2.7);
    float body = 1.0 - smoothstep(0.82 + wobble, 0.86 + wobble, r);
    float glow = exp(-max(r - 0.80, 0.0) * 5.5) * (0.35 + amp * 0.65);
    float alpha = (body + glow * 0.45) * in.color0.a * in.params0.z;
    float3 base = mix(in.color1.rgb, in.color0.rgb, clamp(1.0 - r * 0.9, 0.0, 1.0));
    float3 rgb = mix(base, in.color2.rgb, clamp((r - 0.35) * 0.9, 0.0, 1.0) * 0.5);
    rgb += in.color2.rgb * pow(max(1.0 - r, 0.0), 3.0) * 0.18;
    return float4(rgb, alpha);
}

fragment float4 effect_frag(
    EffectOut in [[stage_in]],
    constant FrameUniforms& u [[buffer(0)]],
    constant ClipRect* clips [[buffer(1)]]
) {
    if (clip_discard(in.logical_pos, clips, in.clip_id)) {
        discard_fragment();
    }

    float2 half_size = in.rect.zw * 0.5;
    float2 center = in.rect.xy + half_size;
    float dist = sd_rounded_box(in.logical_pos - center, half_size, in.radii);
    float aa = 0.5 / max(u.scale_factor, 1.0);
    float cov = 1.0 - smoothstep(-aa, aa, dist);
    if (cov <= 0.001) {
        discard_fragment();
    }

    float4 c;
    switch (in.kind) {
        case 0u: c = effect_shimmer(in); break;
        case 1u: c = effect_ripple(in); break;
        case 2u: c = effect_beam(in); break;
        case 3u: c = effect_cursor_glow(in); break;
        case 4u: c = effect_aurora(in); break;
        case 5u: c = effect_pulse_ring(in); break;
        case 6u: c = effect_voice_orb(in); break;
        default: c = float4(0.0); break;
    }

    float a = clamp(c.a, 0.0, 1.0) * cov;
    if (a <= 0.001) {
        discard_fragment();
    }
    return float4(c.rgb * a, a);
}

// ── 6. Instanced Particles ──────────────────────────────────────────────────────────────

struct ParticleInstance {
    float2 pos;
    float2 size;
    float angle_rad;
    uint color;
    uint shape_clip;   // low 4: shape, high: clip id
    uint _pad;
};

struct ParticleOut {
    float4 position [[position]];
    float2 logical_pos;
    float2 local;
    float2 half_size;
    float angle;
    float4 color;
    uint shape [[flat]];
    uint clip_id [[flat]];
};

vertex ParticleOut particle_vert(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant FrameUniforms& u [[buffer(0)]],
    const device ParticleInstance* particles [[buffer(1)]]
) {
    ParticleInstance p = particles[iid];
    float2 corner = unit_quad_corner(vid);
    float2 half_size = max(p.size * 0.5, float2(1.0));
    float pad = 2.0;
    float2 local = (corner - 0.5) * (half_size * 2.0 + pad * 2.0);
    float c = cos(p.angle_rad);
    float s = sin(p.angle_rad);
    float2 rotated = float2(local.x * c - local.y * s, local.x * s + local.y * c);
    float2 pos = p.pos + rotated;

    ParticleOut out;
    out.position = float4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    out.logical_pos = pos;
    out.local = local;
    out.half_size = half_size;
    out.angle = p.angle_rad;
    out.color = unpack_rgba8(p.color);
    out.shape = p.shape_clip & 0xfu;
    out.clip_id = p.shape_clip >> 4u;
    return out;
}

fragment float4 particle_frag(
    ParticleOut in [[stage_in]],
    constant FrameUniforms& u [[buffer(0)]],
    constant ClipRect* clips [[buffer(1)]]
) {
    if (clip_discard(in.logical_pos, clips, in.clip_id)) {
        discard_fragment();
    }
    // Rotate back into the particle's local frame for analytic shape coverage.
    float c = cos(-in.angle);
    float s = sin(-in.angle);
    float2 p = float2(in.local.x * c - in.local.y * s, in.local.x * s + in.local.y * c);
    float aa = 1.0;
    float cov = 0.0;
    if (in.shape == 1u) {
        float2 q = abs(p) - in.half_size;
        float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 1.0;
        cov = 1.0 - smoothstep(-aa, aa, d);
    } else if (in.shape == 2u) {
        float d = (abs(p.x) + abs(p.y)) * 0.7071 - min(in.half_size.x, in.half_size.y) * 1.05;
        cov = 1.0 - smoothstep(-aa, aa, d);
        cov += exp(-length(p) * 2.5) * 0.25;
    } else {
        float d = length(p) - min(in.half_size.x, in.half_size.y);
        cov = 1.0 - smoothstep(-aa, aa, d);
    }
    float a = clamp(cov, 0.0, 1.0) * in.color.a;
    if (a <= 0.002) {
        discard_fragment();
    }
    return float4(in.color.rgb * a, a);
}

// ── 7. Backdrop Blur Composite ─────────────────────────────────────────────────────────
//
// The host renders the scene into an offscreen texture, downsamples + dual-filters it at half
// resolution, then draws this SDF-clipped pane that samples the blurred texture.

struct BackdropInstance {
    float4 rect;
    float4 radii;
    uint tint;
    uint rim;
    float4 params;   // blur_radius, rim_width, grain, opacity
    uint clip_id;
    uint _pad;
};

struct BackdropOut {
    float4 position [[position]];
    float2 logical_pos;
    float4 rect;
    float4 radii;
    float4 tint;
    float4 rim;
    float4 params;
    uint clip_id [[flat]];
};

vertex BackdropOut backdrop_vert(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant FrameUniforms& u [[buffer(0)]],
    const device BackdropInstance* regions [[buffer(1)]]
) {
    BackdropInstance b = regions[iid];
    float2 corner = unit_quad_corner(vid);
    float2 pos = b.rect.xy + corner * b.rect.zw;

    BackdropOut out;
    out.position = float4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    out.logical_pos = pos;
    out.rect = b.rect;
    out.radii = b.radii;
    out.tint = unpack_rgba8(b.tint);
    out.rim = unpack_rgba8(b.rim);
    out.params = b.params;
    out.clip_id = b.clip_id;
    return out;
}

fragment float4 backdrop_frag(
    BackdropOut in [[stage_in]],
    constant FrameUniforms& u [[buffer(0)]],
    constant ClipRect* clips [[buffer(1)]],
    texture2d<float> blur_tex [[texture(0)]],
    sampler blur_sampler [[sampler(0)]]
) {
    if (clip_discard(in.logical_pos, clips, in.clip_id)) {
        discard_fragment();
    }
    float2 half_size = in.rect.zw * 0.5;
    float2 center = in.rect.xy + half_size;
    float dist = sd_rounded_box(in.logical_pos - center, half_size, in.radii);
    float aa = 0.5 / max(u.scale_factor, 1.0);
    float cov = 1.0 - smoothstep(-aa, aa, dist);
    if (cov <= 0.001) {
        discard_fragment();
    }

    float2 uv = in.logical_pos / max(u.viewport_size, float2(1.0));
    float4 blur = blur_tex.sample(blur_sampler, uv);
    float3 rgb = in.tint.rgb * in.tint.a + blur.rgb * (1.0 - in.tint.a);
    float alpha = in.tint.a + blur.a * (1.0 - in.tint.a);

    float grain = (hash21(floor(in.logical_pos * 1.5)) - 0.5) * in.params.z;
    rgb += grain;

    float rim_w = max(in.params.y, 1.0);
    float rim_mask = 1.0 - smoothstep(0.0, rim_w, abs(dist));
    rgb = mix(rgb, in.rim.rgb, in.rim.a * rim_mask);

    float a = cov * clamp(in.params.w, 0.0, 1.0);
    a *= clamp(alpha + in.tint.a * 0.35, 0.0, 1.0);
    if (a <= 0.001) {
        discard_fragment();
    }
    return float4(rgb * a, a);
}

// ── 8. Kawase Blur Passes (compute) + Fullscreen Blit ──────────────────────────────────

kernel void blur_downsample(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    sampler samp [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    float2 texel = 1.0 / float2(src.get_width(), src.get_height());
    float4 c = src.sample(samp, uv + texel * float2(-0.5, -0.5));
    c += src.sample(samp, uv + texel * float2(0.5, -0.5));
    c += src.sample(samp, uv + texel * float2(-0.5, 0.5));
    c += src.sample(samp, uv + texel * float2(0.5, 0.5));
    dst.write(c * 0.25, gid);
}

kernel void blur_kawase(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    sampler samp [[sampler(0)]],
    constant float& offset [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    float2 texel = 1.0 / float2(src.get_width(), src.get_height());
    float2 o = texel * max(offset, 0.5);
    float4 c = src.sample(samp, uv + float2(-o.x, -o.y));
    c += src.sample(samp, uv + float2(o.x, -o.y));
    c += src.sample(samp, uv + float2(-o.x, o.y));
    c += src.sample(samp, uv + float2(o.x, o.y));
    dst.write(c * 0.25, gid);
}

struct BlitOut {
    float4 position [[position]];
    float2 uv;
};

vertex BlitOut blit_vert(uint vid [[vertex_id]]) {
    float2 corner = unit_quad_corner(vid);
    BlitOut out;
    out.position = float4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.uv = corner;
    return out;
}

fragment float4 blit_frag(
    BlitOut in [[stage_in]],
    texture2d<float> scene [[texture(0)]],
    sampler scene_sampler [[sampler(0)]]
) {
    return scene.sample(scene_sampler, in.uv);
}
