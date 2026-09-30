// CloudBackground.tsx — full-bleed animated cloud behind the main window
// (never the sidebar). A WebGL fragment shader samples domain-warped fBm
// simplex noise each frame; because time advances monotonically (never
// modulo-wrapped) and the noise is re-sampled rather than replayed, the
// cloud churns probabilistically forever with no visible loop point.
import { useEffect, useRef } from "react";

const VERT = `
attribute vec2 a_pos;
void main() {
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

const FRAG = `
precision mediump float;

uniform vec2 u_res;
uniform float u_time;
uniform float u_dark;

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

/* 3D simplex noise (Ashima Arts / Stefan Gustavson, MIT). */
float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289(i);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

float fbm(vec3 p) {
  float f = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 5; i++) {
    f += amp * snoise(p);
    p = p * 2.03 + vec3(9.7, 3.1, 6.3);
    amp *= 0.5;
  }
  return f;
}

void main() {
  /* Aspect-corrected uv, centered at (0, 0), scaled to window height. */
  vec2 uv = (gl_FragCoord.xy - 0.5 * u_res) / u_res.y;

  /* Field drifts slowly while z = time evolves the noise slice itself,
     so every frame is a fresh sample — no replay, no loop point. */
  vec2 p = uv * 1.15 + vec2(u_time * 0.03, -u_time * 0.014);
  vec3 seed = vec3(p, u_time * 0.16);

  /* Domain warp: sample noise at a noise-displaced position — the source
     of the turbulent, wispy limbs in the reference cloud shape. */
  vec3 q = vec3(
    fbm(seed * 0.6),
    fbm(seed * 0.6 + vec3(5.2, 1.3, 2.8)),
    fbm(seed * 0.6 + vec3(1.7, 9.2, 8.1))
  );
  float detail = fbm(seed + q * 0.9);

  /* Giant slow lobes decide where the cloud gathers and thins; the
     warped detail adds the organic wisps on top. */
  float lobes = fbm(seed * 0.42);
  float i = clamp(
    (lobes * 0.5 + 0.5) * 0.62 + (detail * 0.5 + 0.5) * 0.58 - 0.28,
    0.0, 1.0
  );
  i = i * i * (3.0 - 2.0 * i); /* softstep: cream gaps, dense cores */

  /* Edge-to-edge coverage: only the outermost sliver fades so the
     corners stay soft like the reference. */
  vec2 dEdge = min(gl_FragCoord.xy, u_res - gl_FragCoord.xy);
  float edge = smoothstep(0.0, 0.10 * u_res.y, min(dEdge.x, dEdge.y));

  /* Light blue with gradient: pale wash -> periwinkle bodies ->
     deeper sky blue in the dense patches -> near-white glow.
     Dark: warm-brown wisps on the dark app ground, same structure —
     subtle contrast so the cloud reads as texture, not a spotlight. */
  vec3 c_gap  = mix(vec3(0.882, 0.925, 0.973), vec3(0.129, 0.118, 0.102), u_dark); /* #e1ecf8 / #211e1a */
  vec3 c_mid  = mix(vec3(0.663, 0.800, 0.945), vec3(0.196, 0.176, 0.149), u_dark); /* #a9ccf1 / #322d26 */
  vec3 c_deep = mix(vec3(0.518, 0.702, 0.910), vec3(0.275, 0.247, 0.204), u_dark); /* #84b3e8 / #463f34 */
  vec3 c_glow = mix(vec3(0.992, 0.996, 1.000), vec3(0.353, 0.314, 0.255), u_dark); /* near-white / #5a5041 */
  vec3 col = mix(c_gap, c_mid, smoothstep(0.05, 0.45, i));
  col = mix(col, c_deep, smoothstep(0.45, 0.80, i));
  col = mix(col, c_glow, smoothstep(0.85, 1.00, i));

  float a = (0.28 + 0.60 * i) * (0.35 + 0.65 * edge);
  /* Premultiplied alpha compositing over the app background. */
  gl_FragColor = vec4(col * a, a);
}
`;

export function CloudBackground({ dark = false }: { dark?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  /* Animation clock lives outside the GL effect: a theme flip re-runs the
     effect (new u_dark) but the field keeps drifting from where it was —
     no reset, no replay. */
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    const gl =
      canvas.getContext("webgl2", { alpha: true, premultipliedAlpha: true }) ||
      canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true });
    if (!gl) return;

    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        // eslint-disable-next-line no-console
        console.error("cloud shader:", gl.getShaderInfoLog(s));
        return null;
      }
      return s;
    };
    const vs = compile(gl.VERTEX_SHADER, VERT);
    const fs = compile(gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return;
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      // eslint-disable-next-line no-console
      console.error("cloud program:", gl.getProgramInfoLog(prog));
      return;
    }
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    const uRes = gl.getUniformLocation(prog, "u_res");
    const uTime = gl.getUniformLocation(prog, "u_time");
    const uDark = gl.getUniformLocation(prog, "u_dark");

    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const resize = () => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w === 0 || h === 0) return;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      gl.viewport(0, 0, canvas.width, canvas.height);
    };

    let raf = 0;
    let last = performance.now();
    const redraw = () => {
      gl.uniform2f(uRes, canvas.width, canvas.height);
      gl.uniform1f(uTime, timeRef.current);
      gl.uniform1f(uDark, dark ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    const draw = () => {
      const now = performance.now();
      /* Clamp the leap so a backgrounded/unfocus pause doesn't jump. */
      timeRef.current += Math.min((now - last) / 1000, 0.05);
      last = now;
      redraw();
      raf = requestAnimationFrame(draw);
    };

    resize();
    const ro = new ResizeObserver(() => {
      resize();
      redraw();
    });
    ro.observe(canvas);

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      redraw(); /* static frame, still resize-aware */
    } else {
      raf = requestAnimationFrame(draw);
    }

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      gl.deleteBuffer(buf);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
    };
  }, [dark]);

  return <canvas ref={ref} className="cloud-bg" aria-hidden="true" />;
}
