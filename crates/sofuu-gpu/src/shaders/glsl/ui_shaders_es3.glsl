#version 300 es
// __SOFUU_PIPELINE_DEFINES__
precision highp float;
precision highp int;
precision highp sampler2D;

// Sofuu 2D GPU UI — OpenGL ES 3.0 parities of the Metal kernels.
//
// The renderer compiles this single source four times (one shader pair per pipeline) by
// replacing __SOFUU_PIPELINE_DEFINES__ with:
//   #define SOFUU_PIPELINE_QUAD
//   #define SOFUU_VERTEX            (or SOFUU_FRAGMENT)
//
// Instance data arrives through instanced vertex attributes (GLES 3.0 core supports
// glVertexAttribDivisor); clip rectangles live in a std140 UBO. Geometry and alpha math are
// kept line-for-line equivalent to shaders/msl/ui_shaders.metal.

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
