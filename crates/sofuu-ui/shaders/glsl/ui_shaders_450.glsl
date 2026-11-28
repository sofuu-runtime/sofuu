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
#elif defined(SOFUU_PIPELINE_EFFECT)
struct EffectInstance {
    vec4 rect;
    vec4 radii;
    vec4 params0;
    vec4 params1;
    uvec4 colors_kind;  // c0, c1, c2, [kind: u16, clip: u16]
};
layout(std430, binding = 0) readonly buffer InstanceBlock { EffectInstance instances[]; };
#elif defined(SOFUU_PIPELINE_PARTICLE)
struct ParticleInstance {
    vec4 pos_size;
    float angle_rad;
    uint color;
    uint shape_clip;
    uint _pad;
};
layout(std430, binding = 0) readonly buffer InstanceBlock { ParticleInstance instances[]; };
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
#elif defined(SOFUU_PIPELINE_EFFECT)
layout(location = 0) out vec2 v_logical_pos;
layout(location = 1) out vec4 v_rect;
layout(location = 2) out vec4 v_radii;
layout(location = 3) out vec4 v_params0;
layout(location = 4) out vec4 v_params1;
layout(location = 5) out vec4 v_color0;
layout(location = 6) out vec4 v_color1;
layout(location = 7) out vec4 v_color2;
layout(location = 8) flat out uint v_kind;
layout(location = 9) flat out uint v_clip_id;

void main() {
    EffectInstance e = instances[u.base_instance + gl_InstanceIndex];
    vec2 pos = e.rect.xy - vec2(2.0) +
               kCorners[gl_VertexIndex] * (e.rect.zw + vec2(4.0));
    gl_Position = vec4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_rect = e.rect;
    v_radii = e.radii;
    v_params0 = e.params0;
    v_params1 = e.params1;
    v_color0 = unpack_rgba8(e.colors_kind.x);
    v_color1 = unpack_rgba8(e.colors_kind.y);
    v_color2 = unpack_rgba8(e.colors_kind.z);
    v_kind = e.colors_kind.w & 0xffffu;
    v_clip_id = e.colors_kind.w >> 16u;
}
#elif defined(SOFUU_PIPELINE_PARTICLE)
layout(location = 0) out vec2 v_logical_pos;
layout(location = 1) out vec2 v_local;
layout(location = 2) out vec2 v_half_size;
layout(location = 3) out float v_angle;
layout(location = 4) out vec4 v_color;
layout(location = 5) flat out uint v_shape;
layout(location = 6) flat out uint v_clip_id;

void main() {
    ParticleInstance p = instances[u.base_instance + gl_InstanceIndex];
    vec2 half_size = max(p.pos_size.zw * 0.5, vec2(1.0));
    vec2 local = (kCorners[gl_VertexIndex] - 0.5) * (half_size * 2.0 + 4.0);
    float c = cos(p.angle_rad);
    float s = sin(p.angle_rad);
    vec2 pos = p.pos_size.xy + vec2(local.x * c - local.y * s,
                                    local.x * s + local.y * c);
    gl_Position = vec4(to_ndc(pos, u.viewport_size), 0.0, 1.0);
    v_logical_pos = pos;
    v_local = local;
    v_half_size = half_size;
    v_angle = p.angle_rad;
    v_color = unpack_rgba8(p.color);
    v_shape = p.shape_clip & 0xfu;
    v_clip_id = p.shape_clip >> 4u;
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
#elif defined(SOFUU_PIPELINE_EFFECT)
layout(location = 1) in vec4 v_rect;
layout(location = 2) in vec4 v_radii;
layout(location = 3) in vec4 v_params0;
layout(location = 4) in vec4 v_params1;
layout(location = 5) in vec4 v_color0;
layout(location = 6) in vec4 v_color1;
layout(location = 7) in vec4 v_color2;
layout(location = 8) flat in uint v_kind;
layout(location = 9) flat in uint v_clip_id;

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
    if (clip_discard(v_logical_pos, v_clip_id)) discard;
    vec2 half_size = v_rect.zw * 0.5;
    vec2 center = v_rect.xy + half_size;
    float dist = sd_rounded_box(v_logical_pos - center, half_size, v_radii);
    float aa = 0.5 / max(u.scale_factor, 1.0);
    float cov = 1.0 - smoothstep(-aa, aa, dist);
    if (cov <= 0.001) discard;

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
    if (a <= 0.001) discard;
    o_color = vec4(c.rgb * a, a);
}
#elif defined(SOFUU_PIPELINE_PARTICLE)
layout(location = 1) in vec2 v_local;
layout(location = 2) in vec2 v_half_size;
layout(location = 3) in float v_angle;
layout(location = 4) in vec4 v_color;
layout(location = 5) flat in uint v_shape;
layout(location = 6) flat in uint v_clip_id;

void main() {
    if (clip_discard(v_logical_pos, v_clip_id)) discard;
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
    if (a <= 0.002) discard;
    o_color = vec4(v_color.rgb * a, a);
}
#endif

#endif
