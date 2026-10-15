package com.sofuu.ui

import android.content.Context
import android.graphics.Paint
import android.graphics.RuntimeShader
import android.os.Build
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.ImageShader
import androidx.compose.ui.graphics.ShaderBrush
import androidx.compose.ui.graphics.TileMode
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.platform.LocalContext
import com.sofuu.ui.theme.LocalSofuuColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.floor

/**
 * The procedural cloud background — a port of the desktop's
 * CloudBackground.tsx (a domain-warped fBm simplex-noise shader, same
 * palette, same drift).
 *
 * Two paths, one visual language:
 *  - **API 33+**: the desktop's fragment shader runs as an AGSL
 *    [RuntimeShader] — the field is re-sampled from a monotonic clock
 *    every frame, so it churns forever with no loop point, exactly like
 *    the WebGL original.
 *  - **Below API 33** (no AGSL): the same noise is baked once into a
 *    seamless tile on a background thread and drifts under two counter-
 *    moving layers, so it still animates at zero per-frame cost.
 *
 * The palette is the desktop's light-blue set — pale wash → periwinkle
 * bodies → deeper sky blue in the dense patches → near-white glow.
 * Dark mode keeps the desktop's warm-brown-on-dark-ground variant so
 * the warm-dark identity survives.
 */
@Composable
fun CloudBackground(modifier: Modifier = Modifier, animate: Boolean = true) {
    val colors = LocalSofuuColors.current
    val dark = colors.dark
    LaunchedEffect(dark) {
        android.util.Log.i(
            "sofuu-cloud",
            "cloud composing: dark=$dark agsl=${Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU} api=${Build.VERSION.SDK_INT}",
        )
    }

    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        ShaderCloud(dark, animate, modifier)
    } else {
        TileCloud(dark, animate, modifier)
    }
}

// ── Path 1: the shader (API 33+) ────────────────────────────────

@Composable
private fun ShaderCloud(dark: Boolean, animate: Boolean, modifier: Modifier) {
    val shader = remember(dark) {
        try {
            RuntimeShader(CLOUD_AGSL).apply {
                setFloatUniform("iDark", if (dark) 1f else 0f)
            }
        } catch (e: Exception) {
            // A silent null here is exactly how the background disappeared:
            // log the AGSL compile error so the fallback is a decision, not
            // a mystery.
            android.util.Log.e("sofuu-cloud", "AGSL shader unavailable, using tile cloud: ${e.message}", e)
            null
        }
    }
    if (shader == null) {
        TileCloud(dark, animate, modifier)
        return
    }

    val brush = remember(shader) { ShaderBrush(shader) }
    // One-off health line: which path the cloud is actually running on.
    LaunchedEffect(shader) {
        android.util.Log.i("sofuu-cloud", "cloud path = AGSL RuntimeShader (light=${!dark})")
    }
    val start = remember { System.nanoTime() }
    val context = LocalContext.current
    val reduced = remember { animationsDisabled(context) }

    // 30fps is plenty for a slow cloud and halves the fragment cost of a
    // full-screen fBm on a phone; the field itself is re-sampled each
    // draw, so the slower cadence is invisible.
    //
    // animate gates it entirely: the shader samples 4 domain-warped fBm
    // chains of 3D simplex noise over every pixel — a constant heavy GPU
    // load. Driving it only while a turn is in flight makes the cloud
    // double as the activity indicator and drops idle GPU+CPU cost to a
    // single frame.
    var frame by remember { mutableStateOf(0L) }
    if (animate && !reduced) {
        LaunchedEffect(dark, animate) {
            var last = 0L
            while (true) {
                withFrameNanos { now ->
                    if (now - last >= 33_000_000L) {
                        last = now
                        frame = now
                    }
                }
            }
        }
    }
    val elapsed = if (reduced || !animate) 12_000f else ((frame.takeIf { it > 0 } ?: System.nanoTime()) - start) / 1_000_000_000f

    var logged by remember { mutableStateOf(false) }
    Canvas(modifier.fillMaxSize()) {
        // iResolution MUST match the draw space or the shader's aspect
        // correction and edge fade are wrong.
        shader.setFloatUniform("iResolution", size.width, size.height)
        shader.setFloatUniform("iTime", elapsed)
        if (!logged) {
            logged = true
            android.util.Log.i(
                "sofuu-cloud",
                "first frame: canvas=${size.width}x${size.height} t=$elapsed dark=$dark",
            )
        }
        drawRect(brush = brush, size = size)
    }
}

/** Developer options / battery saver can zero the animator scale — then
 *  the cloud renders one static frame instead of animating. */
private fun animationsDisabled(context: Context): Boolean = runCatching {
    android.provider.Settings.Global.getFloat(
        context.contentResolver,
        android.provider.Settings.Global.ANIMATOR_DURATION_SCALE,
        1f,
    ) == 0f
}.getOrDefault(false)

/**
 * The desktop's fragment shader, as AGSL. `precision` qualifiers are
 * removed (AGSL has none), `gl_FragCoord` becomes main()'s parameter and
 * `gl_FragColor` becomes the half4 return.
 */
private val CLOUD_AGSL = """
uniform float2 iResolution;
uniform float iTime;
uniform float iDark;

float3 mod289(float3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
float4 mod289(float4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
float4 permute(float4 x) { return mod289(((x * 34.0) + 1.0) * x); }
float4 taylorInvSqrt(float4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

/* 3D simplex noise (Ashima Arts / Stefan Gustavson, MIT). */
float snoise(float3 v) {
  float2 C = float2(1.0 / 6.0, 1.0 / 3.0);
  float4 D = float4(0.0, 0.5, 1.0, 2.0);

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

  float n_ = 0.142857142857;
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
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

  float4 m = max(0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, float4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

/* 3-octave variant for the giant slow lobes — those are low-frequency by
   nature, so the extra octaves were never visible. */
float lobes3(float3 p) {
  float f = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 3; i++) {
    f += amp * snoise(p);
    p = p * 2.03 + float3(9.7, 3.1, 6.3);
    amp *= 0.5;
  }
  return f;
}

float fbm(float3 p) {
  float f = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 4; i++) {
    f += amp * snoise(p);
    p = p * 2.03 + float3(9.7, 3.1, 6.3);
    amp *= 0.5;
  }
  return f;
}

half4 main(float2 fragCoord) {
  float2 uv = (fragCoord - 0.5 * iResolution) / iResolution.y;

  /* The field drifts slowly while z = time evolves the noise slice
     itself, so every frame is a fresh sample — no replay, no loop. */
  float2 p = uv * 1.15 + float2(iTime * 0.03, -iTime * 0.014);
  float3 seed = float3(p, iTime * 0.16);

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

  float2 dEdge = min(fragCoord, iResolution - fragCoord);
  float edge = smoothstep(0.0, 0.10 * iResolution.y, min(dEdge.x, dEdge.y));

  /* Light blue with gradient: pale wash → periwinkle bodies → deeper
     sky blue in the dense patches → near-white glow. */
  float3 c_gap  = mix(float3(0.882, 0.925, 0.973), float3(0.129, 0.118, 0.102), iDark);
  float3 c_mid  = mix(float3(0.663, 0.800, 0.945), float3(0.196, 0.176, 0.149), iDark);
  float3 c_deep = mix(float3(0.518, 0.702, 0.910), float3(0.275, 0.247, 0.204), iDark);
  float3 c_glow = mix(float3(0.992, 0.996, 1.000), float3(0.353, 0.314, 0.255), iDark);

  float3 col = mix(c_gap, c_mid, smoothstep(0.05, 0.45, i));
  col = mix(col, c_deep, smoothstep(0.45, 0.80, i));
  col = mix(col, c_glow, smoothstep(0.85, 1.00, i));

  float a = (0.28 + 0.60 * i) * (0.35 + 0.65 * edge);

  /* Paint the window ground INSIDE the shader and composite the cloud on
     it at full opacity. Relying on a translucent film meant the cloud
     vanished whenever the blend landed near the background value — and a
     flat screen is indistinguishable from "no cloud at all". */
  float3 ground = mix(float3(1.0, 0.984, 0.941), float3(0.129, 0.114, 0.098), iDark); /* #fffbf0 / #211d19 */
  float3 outc = mix(ground, col, clamp(a * 1.25, 0.0, 1.0));
  return half4(half3(outc), half(1.0));
}
""".trimIndent()

// ── Path 2: the baked tile (below API 33) ───────────────────────

@Composable
private fun TileCloud(dark: Boolean, animate: Boolean, modifier: Modifier) {
    val size = 192
    var tile by remember(dark, size) { mutableStateOf<ImageBitmap?>(null) }
    LaunchedEffect(dark, size) {
        tile = withContext(Dispatchers.Default) { bakeCloudTile(size, dark) }
    }
    LaunchedEffect(Unit) {
        android.util.Log.i("sofuu-cloud", "cloud path = baked tile (no AGSL on this device)")
    }
    val bitmap = tile ?: return

    val start = remember { System.nanoTime() }
    val context = LocalContext.current
    var frame by remember { mutableStateOf(0L) }
    if (animate && !animationsDisabled(context)) {
        LaunchedEffect(dark, animate) {
            while (true) {
                withFrameNanos { now -> frame = now }
            }
        }
    }
    val t = if (!animate) 12_000f else ((frame.takeIf { it > 0 } ?: System.nanoTime()) - start) / 1_000_000_000f

    val brushA = remember(bitmap) {
        ShaderBrush(ImageShader(bitmap, TileMode.Repeated, TileMode.Repeated))
    }
    val brushB = remember(bitmap) {
        ShaderBrush(ImageShader(bitmap, TileMode.Repeated, TileMode.Repeated))
    }

    val ground = if (dark) Color(0xFF211D19) else Color(0xFFFFFBF0)
    Canvas(modifier.fillMaxSize()) {
        val w = this.size.width
        val h = this.size.height
        val tilePx = bitmap.width.toFloat()
        val span = androidx.compose.ui.geometry.Size(w + 2 * tilePx, h + 2 * tilePx)
        // Ground first (opaque), cloud over it — same contract as the
        // shader path, so the fallback can never read as "no cloud".
        drawRect(color = ground)
        // Two counter-drifting layers of the same tile read as a slowly
        // churning cloud rather than a sliding texture.
        translate(left = (t * 14f) % tilePx, top = (-t * 7f) % tilePx) {
            drawRect(brush = brushA, size = span, alpha = 0.85f)
        }
        translate(left = (-t * 9f) % tilePx, top = (t * 5f) % tilePx) {
            drawRect(brush = brushB, size = span, alpha = 0.55f)
        }
    }
}

/** Bake one seamless tile of the same field: domain-warped 2D fBm value
 *  noise, mapped through the desktop's exact palette + alpha curve. */
private fun bakeCloudTile(n: Int, dark: Boolean): ImageBitmap {
    val pixels = IntArray(n * n)
    val gap = if (dark) Color(0xFF211E1A) else Color(0xFFDCECFD)
    val mid = if (dark) Color(0xFF322D26) else Color(0xFFA9CCF1)
    val deep = if (dark) Color(0xFF463F34) else Color(0xFF84B3E8)
    val glow = if (dark) Color(0xFF5A5041) else Color(0xFFFEFEFF)
    // Present, not ghostly: the desktop cloud reads at a clear but soft
    // strength over the app ground.
    val alphaScale = if (dark) 0.45f else 0.85f

    for (y in 0 until n) {
        for (x in 0 until n) {
            // Tileable domain: sample noise on a torus so edges match.
            val u = x.toFloat() / n
            val v = y.toFloat() / n
            val a = (u * 2f * Math.PI.toFloat())
            val b = (v * 2f * Math.PI.toFloat())
            val qx = valueFbm(a * 1.7f + 11.3f, b * 1.7f + 4.1f, 4)
            val qy = valueFbm(a * 1.9f + 3.7f, b * 1.9f + 9.2f, 4)
            val detail = valueFbm(a * 3.1f + qx * 0.9f + 2.5f, b * 3.1f + qy * 0.9f + 6.7f, 5)
            val lobes = valueFbm(a * 0.9f + qx * 0.35f, b * 0.9f + qy * 0.35f, 3)
            var i = ((lobes * 0.5f + 0.5f) * 0.62f + (detail * 0.5f + 0.5f) * 0.58f - 0.28f)
            i = i.coerceIn(0f, 1f)
            i = i * i * (3f - 2f * i)

            val col = lerpColor(gap, mid, smoothstep(0.05f, 0.45f, i))
            val col2 = lerpColor(col, deep, smoothstep(0.45f, 0.80f, i))
            val final = lerpColor(col2, glow, smoothstep(0.85f, 1f, i))
            val alpha = ((0.28f + 0.60f * i) * alphaScale).coerceIn(0f, 1f)
            pixels[y * n + x] = (final.copy(alpha = alpha).toArgbSafe())
        }
    }
    return android.graphics.Bitmap.createBitmap(pixels, n, n, android.graphics.Bitmap.Config.ARGB_8888).asImageBitmap()
}

// ── helpers ─────────────────────────────────────────────────────

private fun Color.toArgbSafe(): Int =
    android.graphics.Color.argb(
        (this.alpha * 255).toInt().coerceIn(0, 255),
        (this.red * 255).toInt().coerceIn(0, 255),
        (this.green * 255).toInt().coerceIn(0, 255),
        (this.blue * 255).toInt().coerceIn(0, 255),
    )

private fun lerpColor(a: Color, b: Color, t: Float): Color = Color(
    red = a.red + (b.red - a.red) * t,
    green = a.green + (b.green - a.green) * t,
    blue = a.blue + (b.blue - a.blue) * t,
    alpha = 1f,
)

private fun smoothstep(e0: Float, e1: Float, x: Float): Float {
    val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f)
    return t * t * (3f - 2f * t)
}

/** Tileable 2D value noise on a torus + fBm octaves. */
private fun valueFbm(a: Float, b: Float, octaves: Int): Float {
    var sum = 0f
    var amp = 0.5f
    var freq = 1f
    for (o in 0 until octaves) {
        sum += amp * valueNoise(a * freq + o * 17.3f, b * freq + o * 9.1f)
        freq *= 2.03f
        amp *= 0.5f
    }
    return sum
}

private fun valueNoise(x: Float, y: Float): Float {
    val xi = floor(x).toInt()
    val yi = floor(y).toInt()
    val xf = x - xi
    val yf = y - yi
    val u = xf * xf * (3f - 2f * xf)
    val v = yf * yf * (3f - 2f * yf)
    val n00 = hash2(xi, yi)
    val n10 = hash2(xi + 1, yi)
    val n01 = hash2(xi, yi + 1)
    val n11 = hash2(xi + 1, yi + 1)
    val nx0 = n00 + (n10 - n00) * u
    val nx1 = n01 + (n11 - n01) * u
    return (nx0 + (nx1 - nx0) * v) * 2f - 1f
}

private fun hash2(xi: Int, yi: Int): Float {
    var h = xi * 374761393 + yi * 668265263
    h = (h xor (h shr 13)) * 1274126177
    h = h xor (h shr 16)
    return ((h and 0x7fffffff) / 1073741824f - 0.5f).coerceIn(0f, 1f)
}
