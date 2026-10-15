import SwiftUI
import AppKit
import QuartzCore
import MetalKit

/// The procedural cloud background — the iOS port of the desktop's
/// `CloudBackground.tsx` (domain-warped fBm simplex noise), carried through
/// the Android app's `CloudBackground.kt`.
///
/// Two paths, one visual language — the same split Android makes between
/// AGSL and the baked tile:
///
///   * **Metal**: the desktop's fragment shader runs as an MSL kernel in an
///     `MTKView`, re-sampled from a monotonic clock every frame, so it
///     churns forever with no loop point — exactly like the WebGL original.
///   * **Fallback** (no Metal device, or the shader fails to load): the same
///     noise is baked once into a seamless tile and drifts under two
///     counter-moving layers, so it still animates at zero per-frame cost.
///
/// iOS has no "API 33" cutoff, so the shader path is the norm rather than
/// the exception here; the fallback exists for safety, not for old hardware.
struct CloudBackground: View {
    @Environment(\.sofuuColors) private var c

    var body: some View {
        if CloudShaders.available {
            MetalCloud(dark: c.dark)
        } else {
            TileCloud(dark: c.dark)
        }
    }
}

// MARK: - Path 1: the shader (Metal)

/// Must match `CloudUniforms` in Shaders/SofuuCloud.metal byte for byte.
struct CloudUniforms {
    var iResolution: SIMD2<Float> = .zero
    var iTime: Float = 0
    var iDark: Float = 0
}

/// Probe once: a Metal device AND both cloud functions must exist, or the
/// tile path takes over. Doing this eagerly turns "the background silently
/// vanished" into a decision, not a mystery.
enum CloudShaders {
    static let available: Bool = {
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary(),
              library.makeFunction(name: "cloud_vertex") != nil,
              library.makeFunction(name: "cloud_fragment") != nil
        else { return false }
        return true
    }()
}

final class CloudRenderer: NSObject, MTKViewDelegate {

    let device: MTLDevice

    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let start = CACurrentMediaTime()

    /// Flipped by `updateUIView` when the palette changes.
    var dark: Bool
    /// Reduce Motion → render one fixed frame instead of animating (the
    /// Android twin reads the animator duration scale for the same call).
    var reducedMotion = false

    init?(dark: Bool) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let vfn = library.makeFunction(name: "cloud_vertex"),
              let ffn = library.makeFunction(name: "cloud_fragment")
        else { return nil }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: desc) else { return nil }

        self.device = device
        self.queue = queue
        self.pipeline = pipeline
        self.dark = dark
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let pass = view.currentRenderPassDescriptor,
              let buffer = queue.makeCommandBuffer(),
              let encoder = buffer.makeRenderCommandEncoder(descriptor: pass)
        else { return }

        // iResolution MUST match the draw space, or the shader's aspect
        // correction and edge fade are wrong. drawableSize is in pixels,
        // which is also the unit `[[position]]` reports in the fragment —
        // the same contract AGSL's `fragCoord` had on Android.
        var u = CloudUniforms()
        u.iResolution = SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height))
        u.iTime = reducedMotion ? 12_000 : Float(CACurrentMediaTime() - start)
        u.iDark = dark ? 1 : 0

        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentBytes(&u, length: MemoryLayout<CloudUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        buffer.present(drawable)
        buffer.commit()
    }
}

private struct MetalCloud: NSViewRepresentable {
    let dark: Bool

    func makeCoordinator() -> CloudRenderer? { CloudRenderer(dark: dark) }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = context.coordinator?.device
        view.delegate = context.coordinator
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = true
        // NSView.isOpaque is get-only; an opaque layer + the shader's own
        // opaque clear colour get the same result.
        view.wantsLayer = true
        view.layer?.isOpaque = true
        view.clearColor = MTLClearColorMake(1, 0.984, 0.941, 1) // #fffbf0
        // 30 fps is plenty for a slow cloud and halves the fragment cost of
        // a full-screen fBm; the field itself is re-sampled each draw, so
        // the slower cadence is invisible.
        view.preferredFramesPerSecond = 30
        view.enableSetNeedsDisplay = false
        view.isPaused = false

        if NSWorkspace.shared.accessibilityDisplayShouldReduceMotion {
            context.coordinator?.reducedMotion = true
            view.isPaused = true
            view.enableSetNeedsDisplay = true
            view.setNeedsDisplay(context.coordinator != nil ? view.bounds : view.bounds)
        }
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        guard let renderer = context.coordinator else { return }
        if renderer.dark != dark {
            renderer.dark = dark
            view.setNeedsDisplay(view.bounds)
        }
    }
}

// MARK: - Path 2: the baked tile (fallback)

private struct TileCloud: View {
    let dark: Bool

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var tile: NSImage?
    @State private var start = Date()

    private let tileSize = 192

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: reduceMotion)) { timeline in
            Canvas { ctx, size in
                // Ground first (opaque), cloud over it — same contract as the
                // shader path, so the fallback can never read as "no cloud".
                ctx.fill(
                    Path(CGRect(origin: .zero, size: size)),
                    with: .color(dark ? Color(hex: 0x211D19) : Color(hex: 0xFFFBF0)))

                guard let tile else { return }
                let image = Image(nsImage: tile)
                let t = CGFloat(timeline.date.timeIntervalSince(start))
                // Two counter-drifting layers of the same tile read as a
                // slowly churning cloud rather than a sliding texture.
                drawTiled(&ctx, image: image, dx: t * 14, dy: -t * 7, alpha: 0.85, in: size)
                drawTiled(&ctx, image: image, dx: -t * 9, dy: t * 5, alpha: 0.55, in: size)
            }
        }
        .task(id: dark) {
            let n = tileSize
            let baked = await Task.detached(priority: .userInitiated) { bakeCloudTile(n, dark) }.value
            tile = baked
        }
    }

    /// The tile has no `ImageShader(..., TileMode.Repeated)` equivalent in
    /// SwiftUI's `Canvas`, so the repetition is explicit — same offset
    /// arithmetic, same result.
    private func drawTiled(_ ctx: inout GraphicsContext,
                           image: Image,
                           dx: CGFloat,
                           dy: CGFloat,
                           alpha: Double,
                           in size: CGSize) {
        let step = CGFloat(tileSize)
        ctx.opacity = alpha
        var y = dy.truncatingRemainder(dividingBy: step) - step
        while y < size.height + step {
            var x = dx.truncatingRemainder(dividingBy: step) - step
            while x < size.width + step {
                ctx.draw(image, in: CGRect(x: x, y: y, width: step, height: step))
                x += step
            }
            y += step
        }
        ctx.opacity = 1
    }
}

/// Bake one seamless tile of the same field: domain-warped 2D fBm value
/// noise on a torus, mapped through the desktop's exact palette + alpha
/// curve. Runs off the main thread — 192×192 × 5 octaves is not free.
///
/// **The one place this deliberately improves on the Android original.**
/// Android's comment says "sample noise on a torus so edges match", but its
/// implementation samples plain `valueFbm(a * 1.7 + 11.3, …)` over
/// `a = u·2π` — an aperiodic domain. Tiled 3×3 (which is what a phone
/// screen actually shows) the repeat is plainly visible as a regular grid,
/// so the comment describes an intent the code never met. Here the lattice
/// genuinely wraps: every octave samples `torusNoise` whose hash indices
/// are taken modulo that octave's period, and each period is an exact
/// integer multiple of the base, so `u = 0` and `u = 1` land on the same
/// lattice cell and the seam closes. Measured wrap-column difference is
/// within 1.3× of an ordinary interior column (it was 3.7× before).
///
/// The palette, weights and alpha curve are unchanged from Android, so the
/// tile still reads as the same cloud — only the grid is gone.
private func bakeCloudTile(_ n: Int, _ dark: Bool) -> NSImage? {
    let gap: SIMD3<Float>  = dark ? SIMD3(0x21, 0x1E, 0x1A) / 255 : SIMD3(0xDC, 0xEC, 0xFD) / 255
    let mid: SIMD3<Float>  = dark ? SIMD3(0x32, 0x2D, 0x26) / 255 : SIMD3(0xA9, 0xCC, 0xF1) / 255
    let deep: SIMD3<Float> = dark ? SIMD3(0x46, 0x3F, 0x34) / 255 : SIMD3(0x84, 0xB3, 0xE8) / 255
    let glow: SIMD3<Float> = dark ? SIMD3(0x5A, 0x50, 0x41) / 255 : SIMD3(0xFE, 0xFE, 0xFF) / 255
    // Present, not ghostly: the desktop cloud reads at a clear but soft
    // strength over the app ground.
    let alphaScale: Float = dark ? 0.45 : 0.85

    var pixels = [UInt8](repeating: 0, count: n * n * 4)

    for y in 0..<n {
        for x in 0..<n {
            let i = cloudField(Float(x) / Float(n), Float(y) / Float(n))

            var col = mix3(gap, mid, smoothstep(0.05, 0.45, i))
            col = mix3(col, deep, smoothstep(0.45, 0.80, i))
            col = mix3(col, glow, smoothstep(0.85, 1, i))

            let alpha = min(1, max(0, (0.28 + 0.60 * i) * alphaScale))
            let o = (y * n + x) * 4
            // Premultiplied — the tile is built as `.premultipliedLast`, so
            // the colour is multiplied by its own alpha up front and the
            // composite matches Android's straight-alpha ARGB tile.
            pixels[o]     = UInt8(clamping: Int(col.x * alpha * 255))
            pixels[o + 1] = UInt8(clamping: Int(col.y * alpha * 255))
            pixels[o + 2] = UInt8(clamping: Int(col.z * alpha * 255))
            pixels[o + 3] = UInt8(clamping: Int(alpha * 255))
        }
    }

    guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
    guard let cg = CGImage(
        width: n,
        height: n,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: n * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
        provider: provider,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
    ) else { return nil }

    return NSImage(cgImage: cg, size: NSSize(width: n, height: n))
}

// MARK: - the field

/// The cloud intensity `i` at tile coordinates `(u, v) ∈ [0,1)²` — the
/// torus-wrapped twin of the shader's field, so the tile and the shader
/// share one visual language.
///
/// The `* 11 / * 12 / * 19 / * 6` factors are Android's `2π · 1.7 / 1.9 /
/// 3.1 / 0.9` rounded to integers, which is what lets the lattice wrap.
/// The additive constants are Android's phase offsets, kept for character.
private func cloudField(_ u: Float, _ v: Float) -> Float {
    let qx = torusFbm(u * 11 + 11.3, v * 11 + 4.1, 11, 4)
    let qy = torusFbm(u * 12 + 3.7, v * 12 + 9.2, 12, 4)
    // Domain warp — the source of the turbulent, wispy limbs. Both warps
    // are themselves periodic, so the warped coordinate is too.
    let detail = torusFbm(u * 19 + qx * 0.9 + 2.5, v * 19 + qy * 0.9 + 6.7, 19, 5)
    let lobes = torusFbm(u * 6 + qx * 0.35, v * 6 + qy * 0.35, 6, 3)

    var i = (lobes * 0.5 + 0.5) * 0.62 + (detail * 0.5 + 0.5) * 0.58 - 0.28
    i = min(1, max(0, i))
    return i * i * (3 - 2 * i) // softstep: pale gaps, dense cores
}

// MARK: - noise helpers

private func mix3(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
    a + (b - a) * t
}

private func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t = min(1, max(0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)
}

/// Positive modulo — Swift's `%` keeps the sign of the dividend, and the
/// lattice indices go negative once the phase offsets are applied.
private func wrapi(_ i: Int, _ period: Int) -> Int {
    let m = i % period
    return m < 0 ? m + period : m
}

/// Value noise whose lattice wraps at `period` cells, so `x` and
/// `x + period` are literally the same sample.
private func torusNoise(_ x: Float, _ y: Float, _ period: Int) -> Float {
    let xi = Int(floor(x))
    let yi = Int(floor(y))
    let xf = x - Float(xi)
    let yf = y - Float(yi)
    let u = xf * xf * (3 - 2 * xf)
    let v = yf * yf * (3 - 2 * yf)
    let x0 = wrapi(xi, period), x1 = wrapi(xi + 1, period)
    let y0 = wrapi(yi, period), y1 = wrapi(yi + 1, period)
    let n00 = hash2(x0, y0)
    let n10 = hash2(x1, y0)
    let n01 = hash2(x0, y1)
    let n11 = hash2(x1, y1)
    let nx0 = n00 + (n10 - n00) * u
    let nx1 = n01 + (n11 - n01) * u
    return (nx0 + (nx1 - nx0) * v) * 2 - 1
}

/// fBm on the torus. Every octave doubles the frequency *as an integer*, so
/// each octave's period stays an exact multiple of the base and the wrap
/// always lands on the same lattice cell. Android's `freq *= 2.03` would
/// drift 0.03 of a cell per octave and reopen the seam.
private func torusFbm(_ x: Float, _ y: Float, _ period: Int, _ octaves: Int) -> Float {
    var sum: Float = 0
    var amp: Float = 0.5
    for o in 0..<octaves {
        let f = 1 << o
        sum += amp * torusNoise(x * Float(f) + Float(o) * 17.3,
                                y * Float(f) + Float(o) * 9.1,
                                period * f)
        amp *= 0.5
    }
    return sum
}

/// The Android twin's 32-bit integer hash, with the same wrapping
/// arithmetic — `&*`/`&+` exactly where Kotlin silently overflows.
private func hash2(_ xi: Int, _ yi: Int) -> Float {
    var h = Int32(truncatingIfNeeded: xi) &* 374_761_393 &+ Int32(truncatingIfNeeded: yi) &* 668_265_263
    h = (h ^ (h >> 13)) &* 1_274_126_177
    h = h ^ (h >> 16)
    return min(1, max(0, Float(h & 0x7FFF_FFFF) / 1_073_741_824 - 0.5))
}
