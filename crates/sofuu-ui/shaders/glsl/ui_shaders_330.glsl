#version 330 core
// __SOFUU_PIPELINE_DEFINES__

// Sofuu 2D GPU UI — Desktop OpenGL 3.3 Core parities of the Metal kernels.
//
// The renderer compiles this single source eight times (one shader pair per pipeline) by
// replacing __SOFUU_PIPELINE_DEFINES__ with:
//   #define SOFUU_PIPELINE_QUAD | RIBBON | STROKE | GLYPH
//   #define SOFUU_PIPELINE_EFFECT | PARTICLE | BACKDROP | BLUR
//   #define SOFUU_VERTEX            (or SOFUU_FRAGMENT)
//
// Instance data arrives through instanced vertex attributes (GL 3.3 core supports
// glVertexAttribDivisor); clip rectangles live in a std140 UBO. Geometry and alpha math are
// kept line-for-line equivalent to `sofuu_ui::shaders::METAL_SRC`.

#ifdef SOFUU_PIPELINE_QUAD

#ifdef SOFUU_VERTEX
layout(location = 0) in vec2 a_corner;
layout(location = 1) in vec4 i_rect;
layout(location = 2) in vec4 i_radii;
layout(location = 3) in uvec4 i_colors;
layout(location = 4) in vec4 i_params;
layout(location = 5) in uint i_clip_id;

uniform vec2 u_viewport_size;

out vec2 v_logical_pos;
out vec4 v_rect;
out vec4 v_radii;
out vec4 v_color_top;
out vec4 v_color_bottom;
out vec4 v_border_color;
out vec4 v_shadow_color;
out vec4 v_params;
flat out uint v_clip_id;

vec4 unpack_rgba8(uint c) {
    return vec4(
        float(c & 0xffu),
        float((c >> 8u) & 0xffu),
        float((c >> 16u) & 0xffu),
        float((c >> 24u) & 0xffu)
    ) / 255.0;
}

vec2 to_ndc(vec2 pos, vec2 viewport) {
    return vec2(
        (pos.x / max(viewport.x, 1.0)) * 2.0 - 1.0,
        1.0 - (pos.y / max(viewport.y, 1.0)) * 2.0
    );
}

void main() {
    float blur = max(i_params.y, 0.0);
    float off_y = i_params.z;
    float margin = (blur > 0.01 ? blur + abs(off_y) : 0.0) + 1.5;

    vec2 origin = i_rect.xy - vec2(margin);
    vec2 extent = i_rect.zw + vec2(margin * 2.0);
    vec2 pos = origin + a_corner * extent;

    gl_Position = vec4(to_ndc(pos, u_viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_rect = i_rect;
    v_radii = i_radii;
    v_color_top = unpack_rgba8(i_colors.x);
    v_color_bottom = unpack_rgba8(i_colors.y);
    v_border_color = unpack_rgba8(i_colors.z);
    v_shadow_color = unpack_rgba8(i_colors.w);
    v_params = i_params;
    v_clip_id = i_clip_id;
}
#else
in vec2 v_logical_pos;
in vec4 v_rect;
in vec4 v_radii;
in vec4 v_color_top;
in vec4 v_color_bottom;
in vec4 v_border_color;
in vec4 v_shadow_color;
in vec4 v_params;
flat in uint v_clip_id;

uniform vec2 u_viewport_size;
uniform float u_scale_factor;

layout(std140) uniform ClipsBlock {
    vec4 u_clips[256];
};

out vec4 o_color;

bool clip_discard(vec2 logical_pos, uint clip_id) {
    vec4 c = u_clips[clip_id];
    return logical_pos.x < c.x || logical_pos.x > (c.x + c.z) ||
           logical_pos.y < c.y || logical_pos.y > (c.y + c.w);
}

float sd_rounded_box(vec2 p, vec2 half_size, vec4 radii) {
    vec2 r_side = (p.x > 0.0) ? vec2(radii.y, radii.z) : vec2(radii.x, radii.w);
    float r = (p.y > 0.0) ? r_side.y : r_side.x;
    r = clamp(r, 0.0, min(half_size.x, half_size.y));
    vec2 q = abs(p) - half_size + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r;
}

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) {
        discard;
    }

    vec2 half_size = v_rect.zw * 0.5;
    vec2 center = v_rect.xy + half_size;
    vec2 p = v_logical_pos - center;

    float aa = 0.5 / max(u_scale_factor, 1.0);
    float border_w = max(v_params.x, 0.0);
    float shadow_blur = max(v_params.y, 0.0);
    float shadow_off_y = v_params.z;
    float opacity = clamp(v_params.w, 0.0, 1.0);

    float dist = sd_rounded_box(p, half_size, v_radii);

    vec4 accum = vec4(0.0);
    if (shadow_blur > 0.01 && v_shadow_color.a > 0.001) {
        float sd_shadow = sd_rounded_box(p - vec2(0.0, shadow_off_y), half_size, v_radii);
        float sh = 1.0 - smoothstep(-shadow_blur * 0.25, shadow_blur, sd_shadow);
        float sh_a = v_shadow_color.a * sh * sh * opacity;
        accum = vec4(v_shadow_color.rgb * sh_a, sh_a);
    }

    float fill_cov = 1.0 - smoothstep(-aa, aa, dist);
    if (fill_cov > 0.001) {
        float grad_t = clamp((v_logical_pos.y - v_rect.y) / max(v_rect.w, 1.0), 0.0, 1.0);
        vec4 fill = mix(v_color_top, v_color_bottom, grad_t);

        if (border_w > 0.01 && v_border_color.a > 0.001) {
            float inner_cov = 1.0 - smoothstep(-aa, aa, dist + border_w);
            float border_cov = clamp(fill_cov - inner_cov, 0.0, 1.0);
            float b_a = v_border_color.a * border_cov;
            fill.rgb = mix(fill.rgb, v_border_color.rgb, b_a);
            fill.a = max(fill.a * fill_cov, b_a);
        } else {
            fill.a *= fill_cov;
        }

        fill.a *= opacity;
        vec4 src = vec4(fill.rgb * fill.a, fill.a);
        accum = src + accum * (1.0 - src.a);
    }

    if (accum.a <= 0.001) {
        discard;
    }
    o_color = accum;
}
#endif
#endif  // SOFUU_PIPELINE_QUAD

#ifdef SOFUU_PIPELINE_RIBBON

#ifdef SOFUU_VERTEX
layout(location = 0) in vec2 a_corner;
layout(location = 1) in vec4 i_x0_y0top_y0bot_x1;
layout(location = 2) in vec2 i_y1top_y1bot;
layout(location = 3) in uvec4 i_colors_clip;

uniform vec2 u_viewport_size;

out vec2 v_logical_pos;
out float v_t;
out vec4 v_color_top;
out vec4 v_color_bottom;
flat out uint v_clip_id;

vec4 unpack_rgba8(uint c) {
    return vec4(
        float(c & 0xffu),
        float((c >> 8u) & 0xffu),
        float((c >> 16u) & 0xffu),
        float((c >> 24u) & 0xffu)
    ) / 255.0;
}

vec2 to_ndc(vec2 pos, vec2 viewport) {
    return vec2(
        (pos.x / max(viewport.x, 1.0)) * 2.0 - 1.0,
        1.0 - (pos.y / max(viewport.y, 1.0)) * 2.0
    );
}

void main() {
    float x0 = i_x0_y0top_y0bot_x1.x;
    float y0_top = i_x0_y0top_y0bot_x1.y;
    float y0_bot = i_x0_y0top_y0bot_x1.z;
    float x1 = i_x0_y0top_y0bot_x1.w;
    float y1_top = i_y1top_y1bot.x;
    float y1_bot = i_y1top_y1bot.y;

    float x = mix(x0, x1, a_corner.x);
    float y_top = mix(y0_top, y1_top, a_corner.x);
    float y_bot = mix(y0_bot, y1_bot, a_corner.x);
    float y = mix(y_top, y_bot, a_corner.y);

    gl_Position = vec4(to_ndc(vec2(x, y), u_viewport_size), 0.0, 1.0);
    v_logical_pos = vec2(x, y);
    v_t = a_corner.y;
    v_color_top = unpack_rgba8(i_colors_clip.x);
    v_color_bottom = unpack_rgba8(i_colors_clip.y);
    v_clip_id = i_colors_clip.z;
}
#else
in vec2 v_logical_pos;
in float v_t;
in vec4 v_color_top;
in vec4 v_color_bottom;
flat in uint v_clip_id;

layout(std140) uniform ClipsBlock {
    vec4 u_clips[256];
};

out vec4 o_color;

bool clip_discard(vec2 logical_pos, uint clip_id) {
    vec4 c = u_clips[clip_id];
    return logical_pos.x < c.x || logical_pos.x > (c.x + c.z) ||
           logical_pos.y < c.y || logical_pos.y > (c.y + c.w);
}

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) {
        discard;
    }
    vec4 col = mix(v_color_top, v_color_bottom, clamp(v_t, 0.0, 1.0));
    if (col.a <= 0.001) {
        discard;
    }
    o_color = vec4(col.rgb * col.a, col.a);
}
#endif
#endif  // SOFUU_PIPELINE_RIBBON

#ifdef SOFUU_PIPELINE_STROKE

#ifdef SOFUU_VERTEX
layout(location = 0) in vec2 a_corner;
layout(location = 1) in vec4 i_p0p1;
layout(location = 2) in vec4 i_width_feather_dash_gap;
layout(location = 3) in uvec4 i_color_clip;

uniform vec2 u_viewport_size;
uniform float u_scale_factor;

out vec2 v_logical_pos;
out vec2 v_p0;
out vec2 v_p1;
out float v_half_w;
out float v_feather;
out float v_dash_len;
out float v_gap_len;
out vec4 v_color;
flat out uint v_clip_id;

vec4 unpack_rgba8(uint c) {
    return vec4(
        float(c & 0xffu),
        float((c >> 8u) & 0xffu),
        float((c >> 16u) & 0xffu),
        float((c >> 24u) & 0xffu)
    ) / 255.0;
}

vec2 to_ndc(vec2 pos, vec2 viewport) {
    return vec2(
        (pos.x / max(viewport.x, 1.0)) * 2.0 - 1.0,
        1.0 - (pos.y / max(viewport.y, 1.0)) * 2.0
    );
}

void main() {
    vec2 p0 = i_p0p1.xy;
    vec2 p1 = i_p0p1.zw;
    float width = i_width_feather_dash_gap.x;
    float feather = i_width_feather_dash_gap.y;
    float dash_len = i_width_feather_dash_gap.z;
    float gap_len = i_width_feather_dash_gap.w;

    float half_w = max(width * 0.5, 0.35);
    float pad = half_w + max(feather, 1.0) + 2.0;

    vec2 bmin = min(p0, p1) - vec2(pad);
    vec2 bmax = max(p0, p1) + vec2(pad);
    vec2 pos = mix(bmin, bmax, a_corner);

    gl_Position = vec4(to_ndc(pos, u_viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_p0 = p0;
    v_p1 = p1;
    v_half_w = half_w;
    v_feather = max(feather, 0.75 / max(u_scale_factor, 1.0));
    v_dash_len = dash_len;
    v_gap_len = gap_len;
    v_color = unpack_rgba8(i_color_clip.x);
    v_clip_id = i_color_clip.y;
}
#else
in vec2 v_logical_pos;
in vec2 v_p0;
in vec2 v_p1;
in float v_half_w;
in float v_feather;
in float v_dash_len;
in float v_gap_len;
in vec4 v_color;
flat in uint v_clip_id;

layout(std140) uniform ClipsBlock {
    vec4 u_clips[256];
};

out vec4 o_color;

bool clip_discard(vec2 logical_pos, uint clip_id) {
    vec4 c = u_clips[clip_id];
    return logical_pos.x < c.x || logical_pos.x > (c.x + c.z) ||
           logical_pos.y < c.y || logical_pos.y > (c.y + c.w);
}

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) {
        discard;
    }
    vec2 pa = v_logical_pos - v_p0;
    vec2 ba = v_p1 - v_p0;
    float len_sq = max(dot(ba, ba), 1e-6);
    float h = clamp(dot(pa, ba) / len_sq, 0.0, 1.0);
    float d = length(pa - ba * h) - v_half_w;

    float cov = 1.0 - smoothstep(-v_feather, v_feather, d);
    if (v_dash_len > 0.01 && v_gap_len > 0.01) {
        float seg_len = sqrt(len_sq);
        float along = h * seg_len;
        float period = v_dash_len + v_gap_len;
        float phase = mod(along, period);
        float dash_cov = smoothstep(0.0, 0.75, phase) *
                         (1.0 - smoothstep(v_dash_len - 0.75, v_dash_len + 0.25, phase));
        cov *= dash_cov;
    }

    float a = v_color.a * cov;
    if (a <= 0.001) {
        discard;
    }
    o_color = vec4(v_color.rgb * a, a);
}
#endif
#endif  // SOFUU_PIPELINE_STROKE

#ifdef SOFUU_PIPELINE_GLYPH

#ifdef SOFUU_VERTEX
layout(location = 0) in vec2 a_corner;
layout(location = 1) in vec4 i_pos_size;
layout(location = 2) in vec4 i_uv;
layout(location = 3) in uint i_color;
layout(location = 4) in float i_px_range;
layout(location = 5) in uint i_clip_id;

uniform vec2 u_viewport_size;

out vec2 v_logical_pos;
out vec2 v_uv;
out vec4 v_color;
out float v_px_range;
flat out uint v_clip_id;

vec4 unpack_rgba8(uint c) {
    return vec4(
        float(c & 0xffu),
        float((c >> 8u) & 0xffu),
        float((c >> 16u) & 0xffu),
        float((c >> 24u) & 0xffu)
    ) / 255.0;
}

vec2 to_ndc(vec2 pos, vec2 viewport) {
    return vec2(
        (pos.x / max(viewport.x, 1.0)) * 2.0 - 1.0,
        1.0 - (pos.y / max(viewport.y, 1.0)) * 2.0
    );
}

void main() {
    vec2 pos = i_pos_size.xy + a_corner * i_pos_size.zw;
    vec2 uv = vec2(
        mix(i_uv.x, i_uv.z, a_corner.x),
        mix(i_uv.y, i_uv.w, a_corner.y)
    );

    gl_Position = vec4(to_ndc(pos, u_viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_uv = uv;
    v_color = unpack_rgba8(i_color);
    v_px_range = i_px_range;
    v_clip_id = i_clip_id;
}
#else
in vec2 v_logical_pos;
in vec2 v_uv;
in vec4 v_color;
in float v_px_range;
flat in uint v_clip_id;

uniform vec2 u_atlas_size;

layout(std140) uniform ClipsBlock {
    vec4 u_clips[256];
};

uniform sampler2D u_atlas;

out vec4 o_color;

bool clip_discard(vec2 logical_pos, uint clip_id) {
    vec4 c = u_clips[clip_id];
    return logical_pos.x < c.x || logical_pos.x > (c.x + c.z) ||
           logical_pos.y < c.y || logical_pos.y > (c.y + c.w);
}

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) {
        discard;
    }
    float sd = texture(u_atlas, v_uv).r - 0.5;
    vec2 unit_range = vec2(v_px_range) / max(u_atlas_size, vec2(1.0));
    vec2 screen_tex_size = vec2(1.0) / max(fwidth(v_uv), vec2(1e-5));
    float screen_px_range = max(0.5 * dot(unit_range, screen_tex_size), 1.0);
    float px_dist = sd * screen_px_range;
    float cov = smoothstep(-0.48, 0.44, px_dist);
    float alpha = cov * v_color.a;
    if (alpha <= 0.002) {
        discard;
    }
    o_color = vec4(v_color.rgb * alpha, alpha);
}
#endif
#endif  // SOFUU_PIPELINE_GLYPH
#ifdef SOFUU_PIPELINE_EFFECT

#ifdef SOFUU_VERTEX
layout(location = 0) in vec2 a_corner;
layout(location = 1) in vec4 i_rect;
layout(location = 2) in vec4 i_radii;
layout(location = 3) in vec4 i_params0;
layout(location = 4) in vec4 i_params1;
layout(location = 5) in uvec4 i_colors_kind;

uniform vec2 u_viewport_size;

out vec2 v_logical_pos;
out vec4 v_rect;
out vec4 v_radii;
out vec4 v_params0;
out vec4 v_params1;
out vec4 v_color0;
out vec4 v_color1;
out vec4 v_color2;
flat out uint v_kind;
flat out uint v_clip_id;

vec4 unpack_rgba8(uint c) {
    return vec4(
        float(c & 0xffu),
        float((c >> 8u) & 0xffu),
        float((c >> 16u) & 0xffu),
        float((c >> 24u) & 0xffu)
    ) / 255.0;
}

vec2 to_ndc(vec2 pos, vec2 viewport) {
    return vec2(
        (pos.x / max(viewport.x, 1.0)) * 2.0 - 1.0,
        1.0 - (pos.y / max(viewport.y, 1.0)) * 2.0
    );
}

void main() {
    vec2 pos = i_rect.xy - vec2(2.0) + a_corner * (i_rect.zw + vec2(4.0));
    gl_Position = vec4(to_ndc(pos, u_viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_rect = i_rect;
    v_radii = i_radii;
    v_params0 = i_params0;
    v_params1 = i_params1;
    v_color0 = unpack_rgba8(i_colors_kind.x);
    v_color1 = unpack_rgba8(i_colors_kind.y);
    v_color2 = unpack_rgba8(i_colors_kind.z);
    v_kind = i_colors_kind.w & 0xffffu;
    v_clip_id = i_colors_kind.w >> 16u;
}
#else
in vec2 v_logical_pos;
in vec4 v_rect;
in vec4 v_radii;
in vec4 v_params0;
in vec4 v_params1;
in vec4 v_color0;
in vec4 v_color1;
in vec4 v_color2;
flat in uint v_kind;
flat in uint v_clip_id;

uniform vec2 u_viewport_size;
uniform float u_scale_factor;

layout(std140) uniform ClipsBlock {
    vec4 u_clips[256];
};

out vec4 o_color;

bool clip_discard(vec2 logical_pos, uint clip_id) {
    vec4 c = u_clips[clip_id];
    return logical_pos.x < c.x || logical_pos.x > (c.x + c.z) ||
           logical_pos.y < c.y || logical_pos.y > (c.y + c.w);
}

float sd_rounded_box(vec2 p, vec2 half_size, vec4 radii) {
    vec2 r_side = (p.x > 0.0) ? vec2(radii.y, radii.z) : vec2(radii.x, radii.w);
    float r = (p.y > 0.0) ? r_side.y : r_side.x;
    r = clamp(r, 0.0, min(half_size.x, half_size.y));
    vec2 q = abs(p) - half_size + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r;
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec4 effect_shimmer(vec2 pos) {
    vec2 p = pos - v_rect.xy;
    float band = max(v_params0.x, 8.0);
    float intensity = v_params0.y;
    float phase = v_params1.x;
    vec2 dir = normalize(vec2(0.82, 0.57));
    float span = v_rect.z * dir.x + v_rect.w * dir.y;
    float travel = phase * (span + band * 2.0) - band;
    float d = abs(dot(p, dir) - travel) / band;
    float band_a = pow(max(1.0 - d, 0.0), 2.2);
    float sheen = 0.035 * intensity;
    return vec4(mix(v_color1.rgb, v_color0.rgb, band_a),
                (band_a * v_color0.a * 0.9 + sheen) * intensity);
}

vec4 effect_ripple(vec2 pos) {
    vec2 origin = v_rect.xy + v_params0.xy;
    float max_r = max(v_params0.z, 1.0);
    float ring_w = max(v_params0.w, 0.6);
    float progress = clamp(v_params1.x, 0.0, 1.0);
    float r = progress * max_r;
    float dist = length(pos - origin);
    float ring = 1.0 - smoothstep(ring_w * 0.5 - 0.75, ring_w * 0.5 + 0.75, abs(dist - r));
    float fill = (1.0 - smoothstep(0.0, max(r, 1.0), dist)) * (1.0 - progress);
    float fade = pow(1.0 - progress, 1.5);
    return vec4(mix(v_color0.rgb, v_color1.rgb, fill * 0.6),
                (ring * fade + fill * 0.10) * v_color0.a);
}

vec4 effect_beam(vec2 pos) {
    vec2 center = v_rect.xy + v_rect.zw * 0.5;
    vec2 p = pos - center;
    float dist = sd_rounded_box(p, v_rect.zw * 0.5, v_radii);
    float width = max(v_params0.y, 1.0);
    float border = 1.0 - smoothstep(0.0, width, -dist);
    float angle = atan(p.y, p.x);
    float time = v_params1.x * v_params0.x;
    float comet = pow(max(cos(angle - time * 2.0), 0.0), 10.0);
    return vec4(mix(v_color2.rgb, mix(v_color0.rgb, v_color1.rgb, comet), 0.85),
                border * (0.22 + 0.78 * comet) * v_color0.a * v_params0.z);
}

vec4 effect_cursor_glow(vec2 pos) {
    vec2 local = pos - v_rect.xy - v_params0.xy;
    float radius = max(v_params0.z, 1.0);
    float d = length(local) / radius;
    float glow = exp(-d * d * 2.2) * v_params0.w;
    vec2 center = v_rect.xy + v_rect.zw * 0.5;
    float sd = sd_rounded_box(pos - center, v_rect.zw * 0.5, v_radii);
    float rim = (1.0 - smoothstep(0.0, 1.5, abs(sd))) * 0.35;
    return vec4(mix(v_color1.rgb, v_color0.rgb, clamp(1.0 - d * 0.8, 0.0, 1.0)),
                (glow + rim) * v_color0.a);
}

vec4 effect_aurora(vec2 pos) {
    vec2 uv = (pos - v_rect.xy) / max(v_rect.zw, vec2(1.0));
    float t = v_params1.x * 0.22 + v_params0.x;
    float intensity = v_params0.y;
    vec2 c0 = vec2(0.25, 0.30) + 0.18 * vec2(sin(t * 0.9), cos(t * 1.13));
    vec2 c1 = vec2(0.75, 0.32) + 0.20 * vec2(cos(t * 0.7 + 2.0), sin(t * 1.07));
    vec2 c2 = vec2(0.45, 0.78) + 0.22 * vec2(sin(t * 1.21 + 4.0), cos(t * 0.8 + 1.0));
    vec2 c3 = vec2(0.70, 0.70) + 0.16 * vec2(cos(t * 1.4 + 1.0), sin(t * 0.6 + 3.0));
    float f0 = exp(-dot(uv - c0, uv - c0) * 7.0);
    float f1 = exp(-dot(uv - c1, uv - c1) * 7.0);
    float f2 = exp(-dot(uv - c2, uv - c2) * 8.0);
    float f3 = exp(-dot(uv - c3, uv - c3) * 10.0);
    float field = clamp((f0 + f1 + f2 + f3) * 0.9, 0.0, 1.5);
    vec3 rgb = (v_color0.rgb * f0 + v_color1.rgb * f1 + v_color2.rgb * f2 +
                mix(v_color1.rgb, v_color2.rgb, 0.5) * f3) / max(f0 + f1 + f2 + f3, 0.001);
    rgb = mix(v_color2.rgb, rgb, clamp(field * 0.8, 0.0, 1.0));
    float dither = (hash21(floor(pos * 1.5)) - 0.5) * 0.008;
    return vec4(max(rgb + dither, vec3(0.0)),
                clamp(field * 0.55, 0.0, 1.0) * v_color0.a * intensity);
}

vec4 effect_pulse_ring(vec2 pos) {
    vec2 center = v_rect.xy + v_rect.zw * 0.5;
    float progress = clamp(v_params0.x, 0.0, 1.0);
    float max_r = max(v_params0.y, 1.0);
    float width = max(v_params0.z, 0.8);
    float dist = length(pos - center);
    float ring = 1.0 - smoothstep(width * 0.5, width * 0.5 + 1.0, abs(dist - progress * max_r));
    return vec4(mix(v_color0.rgb, v_color1.rgb, progress),
                ring * pow(1.0 - progress, 1.3) * v_color0.a * v_params0.w);
}

vec4 effect_voice_orb(vec2 pos) {
    vec2 half_size = v_rect.zw * 0.5;
    vec2 center = v_rect.xy + half_size;
    vec2 uv = (pos - center) / max(min(half_size.x, half_size.y), 1.0);
    float amp = clamp(v_params0.x, 0.0, 1.0);
    float phase = v_params0.y;
    float angle = atan(uv.y, uv.x);
    float r = length(uv);
    float wobble = 0.055 * sin(angle * 3.0 + phase * 1.9) +
                   0.035 * sin(angle * 5.0 - phase * 1.3) +
                   amp * 0.085 * sin(angle * 2.0 + phase * 2.7);
    float body = 1.0 - smoothstep(0.82 + wobble, 0.86 + wobble, r);
    float glow = exp(-max(r - 0.80, 0.0) * 5.5) * (0.35 + amp * 0.65);
    float alpha = (body + glow * 0.45) * v_color0.a * v_params0.z;
    vec3 base = mix(v_color1.rgb, v_color0.rgb, clamp(1.0 - r * 0.9, 0.0, 1.0));
    vec3 rgb = mix(base, v_color2.rgb, clamp((r - 0.35) * 0.9, 0.0, 1.0) * 0.5);
    rgb += v_color2.rgb * pow(max(1.0 - r, 0.0), 3.0) * 0.18;
    return vec4(rgb, alpha);
}

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) {
        discard;
    }
    vec2 half_size = v_rect.zw * 0.5;
    vec2 center = v_rect.xy + half_size;
    float dist = sd_rounded_box(v_logical_pos - center, half_size, v_radii);
    float aa = 0.5 / max(u_scale_factor, 1.0);
    float cov = 1.0 - smoothstep(-aa, aa, dist);
    if (cov <= 0.001) {
        discard;
    }

    vec4 c = vec4(0.0);
    if (v_kind == 0u) {
        c = effect_shimmer(v_logical_pos);
    } else if (v_kind == 1u) {
        c = effect_ripple(v_logical_pos);
    } else if (v_kind == 2u) {
        c = effect_beam(v_logical_pos);
    } else if (v_kind == 3u) {
        c = effect_cursor_glow(v_logical_pos);
    } else if (v_kind == 4u) {
        c = effect_aurora(v_logical_pos);
    } else if (v_kind == 5u) {
        c = effect_pulse_ring(v_logical_pos);
    } else if (v_kind == 6u) {
        c = effect_voice_orb(v_logical_pos);
    }

    float a = clamp(c.a, 0.0, 1.0) * cov;
    if (a <= 0.001) {
        discard;
    }
    o_color = vec4(c.rgb * a, a);
}
#endif
#endif  // SOFUU_PIPELINE_EFFECT

#ifdef SOFUU_PIPELINE_PARTICLE

#ifdef SOFUU_VERTEX
layout(location = 0) in vec2 a_corner;
layout(location = 1) in vec4 i_pos_size;
layout(location = 2) in float i_angle;
layout(location = 3) in uint i_color;
layout(location = 4) in uint i_shape_clip;

uniform vec2 u_viewport_size;

out vec2 v_logical_pos;
out vec2 v_local;
out vec2 v_half_size;
out float v_angle;
out vec4 v_color;
flat out uint v_shape;
flat out uint v_clip_id;

vec4 unpack_rgba8(uint c) {
    return vec4(
        float(c & 0xffu),
        float((c >> 8u) & 0xffu),
        float((c >> 16u) & 0xffu),
        float((c >> 24u) & 0xffu)
    ) / 255.0;
}

vec2 to_ndc(vec2 pos, vec2 viewport) {
    return vec2(
        (pos.x / max(viewport.x, 1.0)) * 2.0 - 1.0,
        1.0 - (pos.y / max(viewport.y, 1.0)) * 2.0
    );
}

void main() {
    vec2 half_size = max(i_pos_size.zw * 0.5, vec2(1.0));
    vec2 local = (a_corner - 0.5) * (half_size * 2.0 + 4.0);
    float c = cos(i_angle);
    float s = sin(i_angle);
    vec2 pos = i_pos_size.xy + vec2(local.x * c - local.y * s, local.x * s + local.y * c);

    gl_Position = vec4(to_ndc(pos, u_viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_local = local;
    v_half_size = half_size;
    v_angle = i_angle;
    v_color = unpack_rgba8(i_color);
    v_shape = i_shape_clip & 0xfu;
    v_clip_id = i_shape_clip >> 4u;
}
#else
in vec2 v_logical_pos;
in vec2 v_local;
in vec2 v_half_size;
in float v_angle;
in vec4 v_color;
flat in uint v_shape;
flat in uint v_clip_id;

layout(std140) uniform ClipsBlock {
    vec4 u_clips[256];
};

out vec4 o_color;

bool clip_discard(vec2 logical_pos, uint clip_id) {
    vec4 c = u_clips[clip_id];
    return logical_pos.x < c.x || logical_pos.x > (c.x + c.z) ||
           logical_pos.y < c.y || logical_pos.y > (c.y + c.w);
}

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) {
        discard;
    }
    float c = cos(-v_angle);
    float s = sin(-v_angle);
    vec2 p = vec2(v_local.x * c - v_local.y * s, v_local.x * s + v_local.y * c);
    float cov = 0.0;
    if (v_shape == 1u) {
        vec2 q = abs(p) - v_half_size;
        float d = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - 1.0;
        cov = 1.0 - smoothstep(-1.0, 1.0, d);
    } else if (v_shape == 2u) {
        float d = (abs(p.x) + abs(p.y)) * 0.7071 - min(v_half_size.x, v_half_size.y) * 1.05;
        cov = 1.0 - smoothstep(-1.0, 1.0, d);
        cov += exp(-length(p) * 2.5) * 0.25;
    } else {
        float d = length(p) - min(v_half_size.x, v_half_size.y);
        cov = 1.0 - smoothstep(-1.0, 1.0, d);
    }
    float a = clamp(cov, 0.0, 1.0) * v_color.a;
    if (a <= 0.002) {
        discard;
    }
    o_color = vec4(v_color.rgb * a, a);
}
#endif
#endif  // SOFUU_PIPELINE_PARTICLE

#ifdef SOFUU_PIPELINE_BACKDROP

#ifdef SOFUU_VERTEX
layout(location = 0) in vec2 a_corner;
layout(location = 1) in vec4 i_rect;
layout(location = 2) in vec4 i_radii;
layout(location = 3) in uvec2 i_tint_rim;
layout(location = 4) in vec4 i_params;
layout(location = 5) in uint i_clip_id;

uniform vec2 u_viewport_size;

out vec2 v_logical_pos;
out vec4 v_rect;
out vec4 v_radii;
out vec4 v_tint;
out vec4 v_rim;
out vec4 v_params;
flat out uint v_clip_id;

vec4 unpack_rgba8(uint c) {
    return vec4(
        float(c & 0xffu),
        float((c >> 8u) & 0xffu),
        float((c >> 16u) & 0xffu),
        float((c >> 24u) & 0xffu)
    ) / 255.0;
}

vec2 to_ndc(vec2 pos, vec2 viewport) {
    return vec2(
        (pos.x / max(viewport.x, 1.0)) * 2.0 - 1.0,
        1.0 - (pos.y / max(viewport.y, 1.0)) * 2.0
    );
}

void main() {
    vec2 pos = i_rect.xy + a_corner * i_rect.zw;
    gl_Position = vec4(to_ndc(pos, u_viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_rect = i_rect;
    v_radii = i_radii;
    v_tint = unpack_rgba8(i_tint_rim.x);
    v_rim = unpack_rgba8(i_tint_rim.y);
    v_params = i_params;
    v_clip_id = i_clip_id;
}
#else
in vec2 v_logical_pos;
in vec4 v_rect;
in vec4 v_radii;
in vec4 v_tint;
in vec4 v_rim;
in vec4 v_params;
flat in uint v_clip_id;

uniform vec2 u_viewport_size;
uniform float u_scale_factor;
uniform sampler2D u_blur_tex;

layout(std140) uniform ClipsBlock {
    vec4 u_clips[256];
};

out vec4 o_color;

bool clip_discard(vec2 logical_pos, uint clip_id) {
    vec4 c = u_clips[clip_id];
    return logical_pos.x < c.x || logical_pos.x > (c.x + c.z) ||
           logical_pos.y < c.y || logical_pos.y > (c.y + c.w);
}

float sd_rounded_box(vec2 p, vec2 half_size, vec4 radii) {
    vec2 r_side = (p.x > 0.0) ? vec2(radii.y, radii.z) : vec2(radii.x, radii.w);
    float r = (p.y > 0.0) ? r_side.y : r_side.x;
    r = clamp(r, 0.0, min(half_size.x, half_size.y));
    vec2 q = abs(p) - half_size + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r;
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) {
        discard;
    }
    vec2 half_size = v_rect.zw * 0.5;
    vec2 center = v_rect.xy + half_size;
    float dist = sd_rounded_box(v_logical_pos - center, half_size, v_radii);
    float aa = 0.5 / max(u_scale_factor, 1.0);
    float cov = 1.0 - smoothstep(-aa, aa, dist);
    if (cov <= 0.001) {
        discard;
    }

    vec2 uv = v_logical_pos / max(u_viewport_size, vec2(1.0));
    vec4 blur = texture(u_blur_tex, uv);
    vec3 rgb = v_tint.rgb * v_tint.a + blur.rgb * (1.0 - v_tint.a);
    float alpha = v_tint.a + blur.a * (1.0 - v_tint.a);
    rgb += (hash21(floor(v_logical_pos * 1.5)) - 0.5) * v_params.z;

    float rim_w = max(v_params.y, 1.0);
    float rim_mask = 1.0 - smoothstep(0.0, rim_w, abs(dist));
    rgb = mix(rgb, v_rim.rgb, v_rim.a * rim_mask);

    float a = cov * clamp(v_params.w, 0.0, 1.0);
    a *= clamp(alpha + v_tint.a * 0.35, 0.0, 1.0);
    if (a <= 0.001) {
        discard;
    }
    o_color = vec4(rgb * a, a);
}
#endif
#endif  // SOFUU_PIPELINE_BACKDROP

#ifdef SOFUU_PIPELINE_BLUR

#ifdef SOFUU_VERTEX
out vec2 v_uv;

void main() {
    vec2 corner;
    if (gl_VertexID == 0) {
        corner = vec2(0.0, 0.0);
    } else if (gl_VertexID == 1) {
        corner = vec2(1.0, 0.0);
    } else if (gl_VertexID == 2) {
        corner = vec2(0.0, 1.0);
    } else {
        corner = vec2(1.0, 1.0);
    }
    v_uv = corner;
    gl_Position = vec4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
}
#else
in vec2 v_uv;

uniform sampler2D u_blur_src;
uniform vec2 u_texel;
uniform float u_offset;

out vec4 o_color;

void main() {
    vec2 o = u_texel * max(u_offset, 0.5);
    vec4 c = texture(u_blur_src, v_uv + vec2(-o.x, -o.y));
    c += texture(u_blur_src, v_uv + vec2(o.x, -o.y));
    c += texture(u_blur_src, v_uv + vec2(-o.x, o.y));
    c += texture(u_blur_src, v_uv + vec2(o.x, o.y));
    o_color = c * 0.25;
}
#endif
#endif  // SOFUU_PIPELINE_BLUR
