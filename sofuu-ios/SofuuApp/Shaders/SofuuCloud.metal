//
//  SofuuCloud.metal
//  Sofuu
//
//  The desktop cloud shader, as Metal Shading Language.
//
//  Direct port of the WebGL fragment shader in sofuu-desktop's
//  CloudBackground.tsx, carried through the Android app's AGSL twin
//  (sofuu-android/.../CloudBackground.kt). The maths is unchanged — the
//  same Ashima/Gustavson 3D simplex noise, the same three-octave `lobes3`
//  for the giant slow bodies, the same four-octave `fbm` for the wispy
//  detail, the same domain warp, the same palette and alpha curve.
//
//  Translation notes (AGSL → MSL):
//    * AGSL's `precision` qualifiers do not exist here and are dropped.
//    * `half4 main(float2 fragCoord)` becomes a real fragment function;
//      `fragCoord` is `[[position]].xy`, which in Metal is already in
//      pixels with a top-left origin — exactly AGSL's convention, so the
//      aspect correction and the edge fade need no flip.
//    * Uniforms arrive as a `constant` buffer whose layout must match
//      `CloudUniforms` in CloudBackground.swift byte for byte.
//

#include <metal_stdlib>
using namespace metal;

// Must match CloudUniforms in CloudBackground.swift:
//   float2 iResolution;  // offset 0,  8 bytes (8-byte aligned)
//   float  iTime;        // offset 8,  4 bytes
//   float  iDark;        // offset 12, 4 bytes
//                          total 16, stride 16
struct CloudUniforms {
    float2 iResolution;
    float  iTime;
    float  iDark;
};

struct CloudVertexOut {
    float4 position [[position]];
};

/// One oversized triangle covering the clip square. Cheaper than a quad and
/// it removes the diagonal seam a two-triangle quad would risk on the very
/// low-frequency lobes.
vertex CloudVertexOut cloud_vertex(uint vid [[vertex_id]]) {
    const float2 pts[3] = {
        float2(-1.0, -3.0),
        float2(-1.0,  1.0),
        float2( 3.0,  1.0)
    };
    CloudVertexOut out;
    out.position = float4(pts[vid], 0.0, 1.0);
    return out;
}

// ── Simplex noise (Ashima Arts / Stefan Gustavson, MIT) ─────────────

static float3 mod289(float3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
static float4 mod289(float4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
static float4 permute(float4 x) { return mod289(((x * 34.0) + 1.0) * x); }
static float4 taylorInvSqrt(float4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

static float snoise(float3 v) {
    const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);
    const float4 D = float4(0.0, 0.5, 1.0, 2.0);

    float3 i  = floor(v + dot(v, C.yyy));
    float3 x0 = v - i + dot(i, C.xxx);

    float3 g = step(x0.yzx, x0.xyz);
    float3 l = 1.0 - g;
    float3 i1 = min(g.xyz, l.zxy);
    float3 i2 = max(g.xyz, l.zxy);

    float3 x1 = x0 - i1 + C.xxx;
    float3 x2 = x0 - i2 + C.yyy;
    float3 x3 = x0 - D.yyy;

    i = mod289(i);
    float4 p = permute(permute(permute(
                 i.z + float4(0.0, i1.z, i2.z, 1.0))
               + i.y + float4(0.0, i1.y, i2.y, 1.0))
               + i.x + float4(0.0, i1.x, i2.x, 1.0));

    const float n_ = 0.142857142857;
    float3 ns = n_ * D.wyz - D.xzx;

    float4 j = p - 49.0 * floor(p * ns.z * ns.z);

    float4 x_ = floor(j * ns.z);
    float4 y_ = floor(j - 7.0 * x_);

    float4 x = x_ * ns.x + ns.yyyy;
    float4 y = y_ * ns.x + ns.yyyy;
    float4 h = 1.0 - abs(x) - abs(y);

    float4 b0 = float4(x.xy, y.xy);
    float4 b1 = float4(x.zw, y.zw);

    float4 s0 = floor(b0) * 2.0 + 1.0;
    float4 s1 = floor(b1) * 2.0 + 1.0;
    float4 sh = -step(h, float4(0.0));

    float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    float3 p0 = float3(a0.xy, h.x);
    float3 p1 = float3(a0.zw, h.y);
    float3 p2 = float3(a1.xy, h.z);
    float3 p3 = float3(a1.zw, h.w);

    float4 norm = taylorInvSqrt(float4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    float4 m = max(0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, float4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

/// 3-octave variant for the giant slow lobes — those are low-frequency by
/// nature, so the extra octaves were never visible.
static float lobes3(float3 p) {
    float f = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 3; i++) {
        f += amp * snoise(p);
        p = p * 2.03 + float3(9.7, 3.1, 6.3);
        amp *= 0.5;
    }
    return f;
}

static float fbm(float3 p) {
    float f = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; i++) {
        f += amp * snoise(p);
        p = p * 2.03 + float3(9.7, 3.1, 6.3);
        amp *= 0.5;
    }
    return f;
}

fragment half4 cloud_fragment(CloudVertexOut in [[stage_in]],
                              constant CloudUniforms &u [[buffer(0)]]) {
    float2 fragCoord = in.position.xy;
    float2 uv = (fragCoord - 0.5 * u.iResolution) / u.iResolution.y;

    /* The field drifts slowly while z = time evolves the noise slice
       itself, so every frame is a fresh sample — no replay, no loop. */
    float2 p = uv * 1.15 + float2(u.iTime * 0.03, -u.iTime * 0.014);
    float3 seed = float3(p, u.iTime * 0.16);

    /* Domain warp — the source of the turbulent, wispy limbs. */
    float3 q = float3(
        fbm(seed * 0.6),
        fbm(seed * 0.6 + float3(5.2, 1.3, 2.8)),
        fbm(seed * 0.6 + float3(1.7, 9.2, 8.1))
    );
    float detail = fbm(seed + q * 0.9);

    float lobes = lobes3(seed * 0.42);
    float i = clamp(
        (lobes * 0.5 + 0.5) * 0.62 + (detail * 0.5 + 0.5) * 0.58 - 0.28,
        0.0, 1.0
    );
    i = i * i * (3.0 - 2.0 * i); /* softstep: pale gaps, dense cores */

    float2 dEdge = min(fragCoord, u.iResolution - fragCoord);
    float edge = smoothstep(0.0, 0.10 * u.iResolution.y, min(dEdge.x, dEdge.y));

    /* Light blue with gradient: pale wash → periwinkle bodies → deeper
       sky blue in the dense patches → near-white glow. */
    float3 c_gap  = mix(float3(0.882, 0.925, 0.973), float3(0.129, 0.118, 0.102), u.iDark);
    float3 c_mid  = mix(float3(0.663, 0.800, 0.945), float3(0.196, 0.176, 0.149), u.iDark);
    float3 c_deep = mix(float3(0.518, 0.702, 0.910), float3(0.275, 0.247, 0.204), u.iDark);
    float3 c_glow = mix(float3(0.992, 0.996, 1.000), float3(0.353, 0.314, 0.255), u.iDark);

    float3 col = mix(c_gap, c_mid, smoothstep(0.05, 0.45, i));
    col = mix(col, c_deep, smoothstep(0.45, 0.80, i));
    col = mix(col, c_glow, smoothstep(0.85, 1.00, i));

    float a = (0.28 + 0.60 * i) * (0.35 + 0.65 * edge);

    /* Paint the window ground INSIDE the shader and composite the cloud on
       it at full opacity. Relying on a translucent film meant the cloud
       vanished whenever the blend landed near the background value — and a
       flat screen is indistinguishable from "no cloud at all". */
    float3 ground = mix(float3(1.0, 0.984, 0.941), float3(0.129, 0.114, 0.098), u.iDark); /* #fffbf0 / #211d19 */
    float3 outc = mix(ground, col, clamp(a * 1.25, 0.0, 1.0));
    return half4(half3(outc), 1.0h);
}
