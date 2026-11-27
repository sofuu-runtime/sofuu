#version 450

// Sofuu 2D GPU UI — Vulkan 1.1 SPIR-V kernels.
//
// Compile one vertex + one fragment module per pipeline with:
//   glslangValidator -V -S vert -DSOFUU_PIPELINE_QUAD   -DSOFUU_VERTEX ui_shaders_450.glsl -o quad.vert.spv
//   glslangValidator -V -S frag -DSOFUU_PIPELINE_QUAD                  ui_shaders_450.glsl -o quad.frag.spv
// (see shaders/generate_spirv.sh). Instance arrays and clip rectangles are std430 SSBOs indexed
// by gl_InstanceIndex; per-frame values arrive as a 32-byte push constant block.
// Geometry/SDF/alpha math matches the Metal and GLSL 330 kernels line-for-line.

#ifdef SOFUU_PIPELINE_QUAD
struct QuadInstance {
    vec4 rect;
    vec4 radii;
    uvec4 colors;
    vec4 params;
    uint clip_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
layout(std430, binding = 0) readonly buffer InstanceBlock { QuadInstance instances[]; };
#elif defined(SOFUU_PIPELINE_RIBBON)
struct RibbonInstance {
    vec4 x0_y0top_y0bot_x1;
    vec2 y1top_y1bot;
    uint color_top;
    uint color_bottom;
    uint clip_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
layout(std430, binding = 0) readonly buffer InstanceBlock { RibbonInstance instances[]; };
#elif defined(SOFUU_PIPELINE_STROKE)
struct StrokeInstance {
    vec4 p0p1;
    vec4 width_feather_dash_gap;
    uint color;
    uint clip_id;
    uint _pad0;
    uint _pad1;
};
layout(std430, binding = 0) readonly buffer InstanceBlock { StrokeInstance instances[]; };
#elif defined(SOFUU_PIPELINE_GLYPH)
struct GlyphInstance {
    vec4 pos_size;
    vec4 uv;
    uint color;
    float px_range;
    uint clip_id;
    uint _pad;
};
layout(std430, binding = 0) readonly buffer InstanceBlock { GlyphInstance instances[]; };
#else
#error "Define one SOFUU_PIPELINE_* macro"
#endif

layout(push_constant) uniform Push {
    vec2 viewport_size;
    vec2 atlas_size;
    float scale_factor;
    float time_sec;
    uint clip_count;
    uint base_instance;
} u;

layout(std430, binding = 1) readonly buffer ClipsBlock { vec4 clips[]; };

#ifdef SOFUU_PIPELINE_GLYPH
layout(binding = 2) uniform sampler2D atlas;
#endif

const vec2 kCorners[4] = vec2[](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0));

vec4 unpack_rgba8(uint c) {
    return vec4(
        float(c & 0xffu),
        float((c >> 8u) & 0xffu),
        float((c >> 16u) & 0xffu),
        float((c >> 24u) & 0xffu)
    ) / 255.0;
}

// Vulkan NDC has +Y down; logical (0,0) is the top-left of the window.
vec2 to_ndc(vec2 pos, vec2 viewport) {
    return vec2(
        (pos.x / max(viewport.x, 1.0)) * 2.0 - 1.0,
        (pos.y / max(viewport.y, 1.0)) * 2.0 - 1.0
    );
}

bool clip_discard(vec2 logical_pos, uint clip_id) {
    if (clip_id >= u.clip_count) return true;
    vec4 c = clips[clip_id];
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

#ifdef SOFUU_VERTEX

#ifdef SOFUU_PIPELINE_QUAD
layout(location = 0) out vec2 v_logical_pos;
layout(location = 1) out vec4 v_rect;
layout(location = 2) out vec4 v_radii;
layout(location = 3) out vec4 v_color_top;
layout(location = 4) out vec4 v_color_bottom;
layout(location = 5) out vec4 v_border_color;
layout(location = 6) out vec4 v_shadow_color;
layout(location = 7) out vec4 v_params;
layout(location = 8) flat out uint v_clip_id;

void main() {
    QuadInstance q = instances[u.base_instance + gl_InstanceIndex];
    float blur = max(q.params.y, 0.0);
    float off_y = q.params.z;
    float margin = (blur > 0.01 ? blur + abs(off_y) : 0.0) + 1.5;

    vec2 origin = q.rect.xy - vec2(margin);
    vec2 extent = q.rect.zw + vec2(margin * 2.0);
    vec2 pos = origin + kCorners[gl_VertexIndex] * extent;

    gl_Position = vec4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_rect = q.rect;
    v_radii = q.radii;
    v_color_top = unpack_rgba8(q.colors.x);
    v_color_bottom = unpack_rgba8(q.colors.y);
    v_border_color = unpack_rgba8(q.colors.z);
    v_shadow_color = unpack_rgba8(q.colors.w);
    v_params = q.params;
    v_clip_id = q.clip_id;
}
#elif defined(SOFUU_PIPELINE_RIBBON)
layout(location = 0) out vec2 v_logical_pos;
layout(location = 1) out float v_t;
layout(location = 2) out vec4 v_color_top;
layout(location = 3) out vec4 v_color_bottom;
layout(location = 4) flat out uint v_clip_id;

void main() {
    RibbonInstance r = instances[u.base_instance + gl_InstanceIndex];
    vec2 corner = kCorners[gl_VertexIndex];
    float x = mix(r.x0_y0top_y0bot_x1.x, r.x0_y0top_y0bot_x1.w, corner.x);
    float y_top = mix(r.x0_y0top_y0bot_x1.y, r.y1top_y1bot.x, corner.x);
    float y_bot = mix(r.x0_y0top_y0bot_x1.z, r.y1top_y1bot.y, corner.x);
    float y = mix(y_top, y_bot, corner.y);

    gl_Position = vec4(to_ndc(vec2(x, y), u.viewport_size), 0.0, 1.0);
    v_logical_pos = vec2(x, y);
    v_t = corner.y;
    v_color_top = unpack_rgba8(r.color_top);
    v_color_bottom = unpack_rgba8(r.color_bottom);
    v_clip_id = r.clip_id;
}
#elif defined(SOFUU_PIPELINE_STROKE)
layout(location = 0) out vec2 v_logical_pos;
layout(location = 1) out vec2 v_p0;
layout(location = 2) out vec2 v_p1;
layout(location = 3) out float v_half_w;
layout(location = 4) out float v_feather;
layout(location = 5) out float v_dash_len;
layout(location = 6) out float v_gap_len;
layout(location = 7) out vec4 v_color;
layout(location = 8) flat out uint v_clip_id;

void main() {
    StrokeInstance s = instances[u.base_instance + gl_InstanceIndex];
    vec2 p0 = s.p0p1.xy;
    vec2 p1 = s.p0p1.zw;
    float half_w = max(s.width_feather_dash_gap.x * 0.5, 0.35);
    float feather = s.width_feather_dash_gap.y;
    float dash_len = s.width_feather_dash_gap.z;
    float gap_len = s.width_feather_dash_gap.w;
    float pad = half_w + max(feather, 1.0) + 2.0;

    vec2 bmin = min(p0, p1) - vec2(pad);
    vec2 bmax = max(p0, p1) + vec2(pad);
    vec2 pos = mix(bmin, bmax, kCorners[gl_VertexIndex]);

    gl_Position = vec4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_p0 = p0;
    v_p1 = p1;
    v_half_w = half_w;
    v_feather = max(feather, 0.75 / max(u.scale_factor, 1.0));
    v_dash_len = dash_len;
    v_gap_len = gap_len;
    v_color = unpack_rgba8(s.color);
    v_clip_id = s.clip_id;
}
#elif defined(SOFUU_PIPELINE_GLYPH)
layout(location = 0) out vec2 v_logical_pos;
layout(location = 1) out vec2 v_uv;
layout(location = 2) out vec4 v_color;
layout(location = 3) out float v_px_range;
layout(location = 4) flat out uint v_clip_id;

void main() {
    GlyphInstance g = instances[u.base_instance + gl_InstanceIndex];
    vec2 corner = kCorners[gl_VertexIndex];
    vec2 pos = g.pos_size.xy + corner * g.pos_size.zw;
    vec2 uv = vec2(
        mix(g.uv.x, g.uv.z, corner.x),
        mix(g.uv.y, g.uv.w, corner.y)
    );

    gl_Position = vec4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_uv = uv;
    v_color = unpack_rgba8(g.color);
    v_px_range = g.px_range;
    v_clip_id = g.clip_id;
}
#endif

#else  // fragment

layout(location = 0) in vec2 v_logical_pos;
layout(location = 0) out vec4 o_color;

#ifdef SOFUU_PIPELINE_QUAD
layout(location = 1) in vec4 v_rect;
layout(location = 2) in vec4 v_radii;
layout(location = 3) in vec4 v_color_top;
layout(location = 4) in vec4 v_color_bottom;
layout(location = 5) in vec4 v_border_color;
layout(location = 6) in vec4 v_shadow_color;
layout(location = 7) in vec4 v_params;
layout(location = 8) flat in uint v_clip_id;

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) discard;

    vec2 half_size = v_rect.zw * 0.5;
    vec2 center = v_rect.xy + half_size;
    vec2 p = v_logical_pos - center;

    float aa = 0.5 / max(u.scale_factor, 1.0);
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

    if (accum.a <= 0.001) discard;
    o_color = accum;
}
#elif defined(SOFUU_PIPELINE_RIBBON)
layout(location = 1) in float v_t;
layout(location = 2) in vec4 v_color_top;
layout(location = 3) in vec4 v_color_bottom;
layout(location = 4) flat in uint v_clip_id;

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) discard;
    vec4 col = mix(v_color_top, v_color_bottom, clamp(v_t, 0.0, 1.0));
    if (col.a <= 0.001) discard;
    o_color = vec4(col.rgb * col.a, col.a);
}
#elif defined(SOFUU_PIPELINE_STROKE)
layout(location = 1) in vec2 v_p0;
layout(location = 2) in vec2 v_p1;
layout(location = 3) in float v_half_w;
layout(location = 4) in float v_feather;
layout(location = 5) in float v_dash_len;
layout(location = 6) in float v_gap_len;
layout(location = 7) in vec4 v_color;
layout(location = 8) flat in uint v_clip_id;

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) discard;
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
    if (a <= 0.001) discard;
    o_color = vec4(v_color.rgb * a, a);
}
#elif defined(SOFUU_PIPELINE_GLYPH)
layout(location = 1) in vec2 v_uv;
layout(location = 2) in vec4 v_color;
layout(location = 3) in float v_px_range;
layout(location = 4) flat in uint v_clip_id;

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) discard;
    float sd = texture(atlas, v_uv).r - 0.5;
    vec2 unit_range = vec2(v_px_range) / max(u.atlas_size, vec2(1.0));
    vec2 screen_tex_size = vec2(1.0) / max(fwidth(v_uv), vec2(1e-5));
    float screen_px_range = max(0.5 * dot(unit_range, screen_tex_size), 1.0);
    float px_dist = sd * screen_px_range;
    float cov = smoothstep(-0.48, 0.44, px_dist);
    float alpha = cov * v_color.a;
    if (alpha <= 0.002) discard;
    o_color = vec4(v_color.rgb * alpha, alpha);
}
#endif

#endif
