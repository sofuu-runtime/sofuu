// GpuUiMetalView.swift — embedded 120Hz Sofuu canvas over the `sofuu_ui_*` C ABI.
//
// PLAN-EMBEDDED-NATIVE-HOST-C-ABI Phase C3 (Mode A: zero-copy DrawList pull).
//
// The host owns the Metal device (MTKView), compiles the four ui shaders once,
// and every display-link tick calls `sofuu_ui_poll_draw_list`, uploads the flat
// instance arrays into ring buffers, and issues instanced draws in the exact
// order of `cmds[]`. When the engine reports `needs_next_frame == 0` the view
// pauses (0% GPU/CPU idle); the session's wake callback resumes it on the main
// queue when JS pushes a patch or input starts a spring.
//
// This file is shared verbatim by sofuu-macos and sofuu-ios (the platform
// adapters are behind `#if os(macOS)` / `#if os(iOS)`). Keep the two copies in
// sync; both compile into the app's default.metallib from `ui_shaders.metal`.
//
// The C surface is imported through the `Sofuu` Clang module. All session
// polling happens on the main/UI thread; `sofuu_ui_attach_runtime` must be
// called on the runtime's owning thread (see SofuuEngine.attachUISession).

#if os(macOS)
import AppKit
#else
import UIKit
#endif
import Metal
import MetalKit
import QuartzCore
import Sofuu
import SwiftUI

// ── Session configuration ────────────────────────────────────────────

/// C `sofuu_ui_wake_cb`: fires from any thread; the actual MTKView resume is
/// hopped to the main queue because MetalKit's paused flag is main-thread state.
private let gpuUiWakeCallback: sofuu_ui_wake_cb = { opaque in
    guard let opaque else { return }
    let view = Unmanaged<GpuUiMetalView>.fromOpaque(opaque).takeUnretainedValue()
    DispatchQueue.main.async {
        view.resumeRendering()
    }
}

// ── Ring buffer (one slot per in-flight frame) ───────────────────────

/// Three shared-storage buffers rotate per frame so the GPU can still be
/// reading frame N-1 while frame N is encoded. Capacity is retained; a steady
/// scene never allocates after the first few frames.
private final class BufferRing {
    private var buffers: [MTLBuffer?]
    private var capacities: [Int]

    init(slots: Int = 3) {
        buffers = Array(repeating: nil, count: slots)
        capacities = Array(repeating: 0, count: slots)
    }

    func upload<T>(_ pointer: UnsafePointer<T>?, count: Int, slot: Int, device: MTLDevice) -> MTLBuffer? {
        guard let pointer, count > 0 else { return nil }
        let bytes = count * MemoryLayout<T>.stride
        if buffers[slot] == nil || capacities[slot] < bytes {
            let length = max(bytes, 16)
            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                return nil
            }
            buffers[slot] = buffer
            capacities[slot] = length
        }
        guard let buffer = buffers[slot] else { return nil }
        buffer.contents().copyMemory(from: pointer, byteCount: bytes)
        return buffer
    }
}

// ── The view ─────────────────────────────────────────────────────────

public final class GpuUiMetalView: MTKView, MTKViewDelegate {
    // Session (owned by this view; the runtime only holds a Weak bridge).
    public private(set) var uiSession: OpaquePointer?
    private var attachedRuntime: OpaquePointer?

    // Config consumed by `commonInit`. Public so a SwiftUI wrapper can set
    // them in `GpuUiMetalView(frame:title:theme:)`.
    public var uiTitle: String = "Sofuu"
    public var uiTheme: String = "light"
    /// Extra top clearance for a transparent titlebar host, in points.
    public var topInsetOverride: CGFloat = 0

    // Metal state.
    private var commandQueue: MTLCommandQueue?
    private var quadPipeline: MTLRenderPipelineState?
    private var ribbonPipeline: MTLRenderPipelineState?
    private var strokePipeline: MTLRenderPipelineState?
    private var glyphPipeline: MTLRenderPipelineState?
    private var atlasTexture: MTLTexture?
    private var atlasSampler: MTLSamplerState?
    private let quadRing = BufferRing()
    private let ribbonRing = BufferRing()
    private let strokeRing = BufferRing()
    private let glyphRing = BufferRing()
    private let clipRing = BufferRing()
    private var ringSlot = 0
    private var lastTime: CFTimeInterval = CACurrentMediaTime()
    private var loggedFailure = false

    // Touch/drag state (iOS).
    #if os(iOS)
    private var lastTouchPoint: CGPoint?
    #endif

    // MARK: Init

    public init(frame: CGRect, title: String = "Sofuu", theme: String = "light") {
        self.uiTitle = title
        self.uiTheme = theme
        super.init(frame: frame, device: MTLCreateSystemDefaultDevice())
        commonInit()
    }

    public required init(coder: NSCoder) {
        super.init(coder: coder)
        if device == nil { device = MTLCreateSystemDefaultDevice() }
        commonInit()
    }

    deinit {
        if let session = uiSession {
            _ = sofuu_ui_session_set_wake_cb(session, nil, nil)
            sofuu_ui_session_free(session)
        }
    }

    private func commonInit() {
        delegate = self
        colorPixelFormat = .bgra8Unorm
        framebufferOnly = true
        isPaused = true
        enableSetNeedsDisplay = false
        preferredFramesPerSecond = 120
        clearColor = MTLClearColorMake(0, 0, 0, 1)
        #if os(macOS)
        wantsLayer = true
        layer?.isOpaque = true
        #endif

        createSession()
        setupMetal()
        applySafeInsets()
        resumeRendering()
    }

    private func createSession() {
        var config: [String: Any] = ["title": uiTitle, "theme": uiTheme]
        config["auto_title"] = "Sofuu // \(uiTitle)"

        let json: String?
        if let data = try? JSONSerialization.data(withJSONObject: config) {
            json = String(data: data, encoding: .utf8)
        } else {
            json = nil
        }

        var error: UnsafeMutablePointer<CChar>?
        let created: OpaquePointer?
        if let json {
            created = json.withCString { sofuu_ui_session_new($0, &error) }
        } else {
            created = sofuu_ui_session_new(nil, &error)
        }
        if let error {
            NSLog("[sofuu-ui] session_new failed: %@", String(cString: error))
            sofuu_free(error)
        }
        guard let created else { return }
        uiSession = created
        _ = sofuu_ui_session_set_wake_cb(
            created,
            gpuUiWakeCallback,
            Unmanaged.passUnretained(self).toOpaque()
        )
    }

    // MARK: Metal setup

    private func setupMetal() {
        guard let device else {
            NSLog("[sofuu-ui] no Metal device")
            return
        }
        commandQueue = device.makeCommandQueue()

        guard let library = device.makeDefaultLibrary() else {
            NSLog("[sofuu-ui] default.metallib missing — add ui_shaders.metal to the target")
            return
        }

        func pipeline(vertex: String, fragment: String) -> MTLRenderPipelineState? {
            guard let vfn = library.makeFunction(name: vertex),
                  let ffn = library.makeFunction(name: fragment) else {
                NSLog("[sofuu-ui] shader function missing: %@ / %@", vertex, fragment)
                return nil
            }
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = vfn
            descriptor.fragmentFunction = ffn
            descriptor.colorAttachments[0].pixelFormat = colorPixelFormat
            let attachment = descriptor.colorAttachments[0]!
            attachment.isBlendingEnabled = true
            attachment.rgbBlendOperation = .add
            attachment.alphaBlendOperation = .add
            attachment.sourceRGBBlendFactor = .one
            attachment.sourceAlphaBlendFactor = .one
            attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
            attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            do {
                return try device.makeRenderPipelineState(descriptor: descriptor)
            } catch {
                NSLog("[sofuu-ui] pipeline %@ failed: %@", vertex, String(describing: error))
                return nil
            }
        }

        quadPipeline = pipeline(vertex: "quad_vert", fragment: "quad_frag")
        ribbonPipeline = pipeline(vertex: "ribbon_vert", fragment: "ribbon_frag")
        strokePipeline = pipeline(vertex: "stroke_vert", fragment: "stroke_frag")
        glyphPipeline = pipeline(vertex: "glyph_vert", fragment: "glyph_frag")

        let sampler = MTLSamplerDescriptor()
        sampler.minFilter = .linear
        sampler.magFilter = .linear
        sampler.sAddressMode = .clampToEdge
        sampler.tAddressMode = .clampToEdge
        atlasSampler = device.makeSamplerState(descriptor: sampler)

        makeAtlasTexture(device: device)
    }

    private func makeAtlasTexture(device: MTLDevice) {
        guard let session = uiSession else { return }
        var pixels: UnsafePointer<UInt8>?
        var width: UInt32 = 0
        var height: UInt32 = 0
        var pxRange: Float = 0
        let rc = sofuu_ui_session_get_atlas(session, &pixels, &width, &height, &pxRange)
        guard rc == 0, let pixels, width > 0, height > 0 else {
            NSLog("[sofuu-ui] atlas unavailable (rc=%d)", rc)
            return
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: Int(width),
            height: Int(height),
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        guard let texture = device.makeTexture(descriptor: descriptor) else { return }
        texture.replace(
            region: MTLRegionMake2D(0, 0, Int(width), Int(height)),
            mipmapLevel: 0,
            withBytes: pixels,
            bytesPerRow: Int(width)
        )
        atlasTexture = texture
    }

    // MARK: Runtime attachment

    /// Bind the engine runtime to this canvas. MUST run on the runtime's owning
    /// thread; `SofuuEngine.attachUISession` marshals it for you.
    public func attachRuntime(_ runtime: OpaquePointer) {
        guard let session = uiSession else { return }
        attachedRuntime = runtime
        let rc = sofuu_ui_attach_runtime(runtime, session)
        if rc != 0 {
            NSLog("[sofuu-ui] attach_runtime failed (rc=%d)", rc)
        } else {
            resumeRendering()
        }
    }

    /// Unbind without freeing either side. Must run on the runtime thread.
    public func detachRuntime() {
        guard let runtime = attachedRuntime else { return }
        _ = sofuu_ui_detach_runtime(runtime)
        attachedRuntime = nil
    }

    // MARK: Frame loop

    public func resumeRendering() {
        lastTime = CACurrentMediaTime()
        if isPaused {
            isPaused = false
        }
    }

    public func pauseRendering() {
        isPaused = true
    }

    #if os(macOS)
    public override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window == nil {
            isPaused = true
        } else {
            applySafeInsets()
            resumeRendering()
        }
    }

    public override func layout() {
        super.layout()
        applySafeInsets()
    }
    #else
    public override func didMoveToWindow() {
        super.didMoveToWindow()
        if window == nil {
            isPaused = true
        } else {
            applySafeInsets()
            resumeRendering()
        }
    }

    public override func layoutSubviews() {
        super.layoutSubviews()
        applySafeInsets()
    }
    #endif

    private func applySafeInsets() {
        guard let session = uiSession else { return }
        #if os(macOS)
        let top = max(safeAreaInsets.top, topInsetOverride)
        let bottom = safeAreaInsets.bottom
        #else
        let top = max(safeAreaInsets.top, topInsetOverride)
        let bottom = safeAreaInsets.bottom
        #endif
        _ = sofuu_ui_session_set_safe_insets(session, Float(top), Float(bottom))
        // The inset marks the scene dirty; resume a paused display link so
        // the reflow is actually compiled (keyboard/rotation while idle).
        resumeRendering()
    }

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // The next draw passes the logical bounds; nothing to cache here.
    }

    public func draw(in view: MTKView) {
        guard let session = uiSession,
              let device,
              let commandQueue,
              let quadPipeline,
              let ribbonPipeline,
              let strokePipeline,
              let glyphPipeline,
              let atlasTexture,
              let atlasSampler else {
            return
        }
        guard view.window != nil else {
            isPaused = true
            return
        }

        let now = CACurrentMediaTime()
        let dt = min(max(now - lastTime, 0), 0.25)
        lastTime = now

        let logical = view.bounds.size
        guard logical.width > 0, logical.height > 0 else { return }
        let scale = uiScale

        var drawList = SofuuUiDrawList()
        let rc = sofuu_ui_poll_draw_list(
            session,
            Float(dt),
            Float(logical.width),
            Float(logical.height),
            scale,
            &drawList
        )
        guard rc == 0, drawList.cmd_count > 0 else {
            if !loggedFailure {
                loggedFailure = true
                NSLog("[sofuu-ui] poll_draw_list rc=%d cmds=%u", rc, drawList.cmd_count)
            }
            return
        }

        guard let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
            return
        }

        ringSlot = (ringSlot + 1) % 3
        let quads = quadRing.upload(drawList.quads, count: Int(drawList.quad_count), slot: ringSlot, device: device)
        let ribbons = ribbonRing.upload(drawList.ribbons, count: Int(drawList.ribbon_count), slot: ringSlot, device: device)
        let strokes = strokeRing.upload(drawList.strokes, count: Int(drawList.stroke_count), slot: ringSlot, device: device)
        let glyphs = glyphRing.upload(drawList.glyphs, count: Int(drawList.glyph_count), slot: ringSlot, device: device)
        let clips = clipRing.upload(drawList.clips, count: Int(drawList.clip_count), slot: ringSlot, device: device)

        var uniforms = drawList.uniforms
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<SofuuUiFrameUniforms>.size, index: 0)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<SofuuUiFrameUniforms>.size, index: 0)
        if let clips {
            encoder.setFragmentBuffer(clips, offset: 0, index: 1)
        }
        encoder.setFragmentTexture(atlasTexture, index: 0)
        encoder.setFragmentSamplerState(atlasSampler, index: 0)

        guard let cmds = drawList.cmds else { return }
        for index in 0..<Int(drawList.cmd_count) {
            let cmd = cmds[index]
            switch cmd.kind {
            case 0: // quads
                guard let quads, cmd.count > 0 else { continue }
                encoder.setRenderPipelineState(quadPipeline)
                encoder.setVertexBuffer(
                    quads,
                    offset: Int(cmd.start) * MemoryLayout<SofuuUiQuadInstance>.stride,
                    index: 1
                )
                encoder.drawPrimitives(
                    type: .triangle,
                    vertexStart: 0,
                    vertexCount: 6,
                    instanceCount: Int(cmd.count)
                )
            case 1: // ribbons
                guard let ribbons, cmd.count > 0 else { continue }
                encoder.setRenderPipelineState(ribbonPipeline)
                encoder.setVertexBuffer(
                    ribbons,
                    offset: Int(cmd.start) * MemoryLayout<SofuuUiRibbonInstance>.stride,
                    index: 1
                )
                encoder.drawPrimitives(
                    type: .triangle,
                    vertexStart: 0,
                    vertexCount: 6,
                    instanceCount: Int(cmd.count)
                )
            case 2: // strokes
                guard let strokes, cmd.count > 0 else { continue }
                encoder.setRenderPipelineState(strokePipeline)
                encoder.setVertexBuffer(
                    strokes,
                    offset: Int(cmd.start) * MemoryLayout<SofuuUiStrokeInstance>.stride,
                    index: 1
                )
                encoder.drawPrimitives(
                    type: .triangle,
                    vertexStart: 0,
                    vertexCount: 6,
                    instanceCount: Int(cmd.count)
                )
            case 3: // glyphs
                guard let glyphs, cmd.count > 0 else { continue }
                encoder.setRenderPipelineState(glyphPipeline)
                encoder.setVertexBuffer(
                    glyphs,
                    offset: Int(cmd.start) * MemoryLayout<SofuuUiGlyphInstance>.stride,
                    index: 1
                )
                encoder.drawPrimitives(
                    type: .triangle,
                    vertexStart: 0,
                    vertexCount: 6,
                    instanceCount: Int(cmd.count)
                )
            default:
                continue
            }
        }

        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()

        // 0% idle: the wake callback (patch/input/spring) resumes the view.
        if drawList.needs_next_frame == 0 {
            isPaused = true
        }
    }

    // MARK: Input helpers

    /// Backing scale on the current platform. AppKit exposes it on the
    /// window (NSView has no `contentScaleFactor`); UIKit on the view.
    private var uiScale: Float {
        #if os(macOS)
        return Float(window?.backingScaleFactor ?? 2.0)
        #else
        return Float(contentScaleFactor)
        #endif
    }

    private func dispatchPointer(_ kind: UInt32, x: CGFloat, y: CGFloat, button: UInt32 = 0) {
        guard let session = uiSession else { return }
        var event = SofuuUiInputEvent()
        event.kind = kind
        event.x = Float(x)
        event.y = Float(y)
        event.scale_factor = uiScale
        event.button = button
        _ = sofuu_ui_dispatch_input(session, &event)
    }

    private func dispatchText(_ text: String) {
        guard let session = uiSession, !text.isEmpty else { return }
        text.withCString { pointer in
            var event = SofuuUiInputEvent()
            event.kind = UInt32(SOFUU_INPUT_TEXT_INPUT)
            event.text_utf8 = pointer
            event.scale_factor = uiScale
            _ = sofuu_ui_dispatch_input(session, &event)
        }
    }

    private func dispatchKey(_ key: String, modifiers: UInt32 = 0) {
        guard let session = uiSession else { return }
        key.withCString { pointer in
            var event = SofuuUiInputEvent()
            event.kind = UInt32(SOFUU_INPUT_KEY_DOWN)
            event.modifiers = modifiers
            event.text_utf8 = pointer
            event.scale_factor = uiScale
            _ = sofuu_ui_dispatch_input(session, &event)
        }
    }

    private func dispatchScroll(x: CGFloat, y: CGFloat, dx: CGFloat, dy: CGFloat) {
        guard let session = uiSession else { return }
        var event = SofuuUiInputEvent()
        event.kind = UInt32(SOFUU_INPUT_SCROLL)
        event.x = Float(x)
        event.y = Float(y)
        event.dx = Float(dx)
        event.dy = Float(dy)
        event.scale_factor = uiScale
        _ = sofuu_ui_dispatch_input(session, &event)
    }

    #if os(macOS)
    // ── macOS adapters ───────────────────────────────────────────────

    public override var acceptsFirstResponder: Bool { true }

    public override func becomeFirstResponder() -> Bool { true }

    public override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas { removeTrackingArea(area) }
        addTrackingArea(
            NSTrackingArea(
                rect: .zero,
                options: [.activeInKeyWindow, .mouseMoved, .inVisibleRect, .mouseEnteredAndExited],
                owner: self,
                userInfo: nil
            )
        )
    }

    private func logicalPoint(_ event: NSEvent) -> CGPoint {
        let point = convert(event.locationInWindow, from: nil)
        // AppKit is bottom-left; the Sofuu canvas is top-left (logical px).
        return CGPoint(x: point.x, y: bounds.height - point.y)
    }

    private func flags(_ event: NSEvent) -> UInt32 {
        var value: UInt32 = 0
        if event.modifierFlags.contains(.shift) { value |= 1 << 0 }
        if event.modifierFlags.contains(.control) { value |= 1 << 1 }
        if event.modifierFlags.contains(.option) { value |= 1 << 2 }
        if event.modifierFlags.contains(.command) { value |= 1 << 3 }
        return value
    }

    public override func mouseMoved(with event: NSEvent) {
        let p = logicalPoint(event)
        dispatchPointer(UInt32(SOFUU_INPUT_POINTER_MOVE), x: p.x, y: p.y)
    }

    public override func mouseDragged(with event: NSEvent) {
        let p = logicalPoint(event)
        dispatchPointer(UInt32(SOFUU_INPUT_POINTER_MOVE), x: p.x, y: p.y, button: 0)
    }

    public override func mouseDown(with event: NSEvent) {
        _ = window?.makeFirstResponder(self)
        let p = logicalPoint(event)
        dispatchPointer(UInt32(SOFUU_INPUT_POINTER_DOWN), x: p.x, y: p.y, button: 0)
    }

    public override func mouseUp(with event: NSEvent) {
        let p = logicalPoint(event)
        dispatchPointer(UInt32(SOFUU_INPUT_POINTER_UP), x: p.x, y: p.y, button: 0)
    }

    public override func scrollWheel(with event: NSEvent) {
        let p = logicalPoint(event)
        dispatchScroll(
            x: p.x,
            y: p.y,
            dx: event.scrollingDeltaX,
            dy: event.scrollingDeltaY
        )
    }

    public override func keyDown(with event: NSEvent) {
        let modifiers = flags(event)
        switch event.keyCode {
        case 51, 117:
            dispatchKey("Backspace", modifiers: modifiers)
        case 36, 76:
            dispatchKey("Enter", modifiers: modifiers)
        case 53:
            dispatchKey("Escape", modifiers: modifiers)
        case 123:
            dispatchKey("ArrowLeft", modifiers: modifiers)
        case 124:
            dispatchKey("ArrowRight", modifiers: modifiers)
        case 125:
            dispatchKey("ArrowDown", modifiers: modifiers)
        case 126:
            dispatchKey("ArrowUp", modifiers: modifiers)
        default:
            // Printable/IME text: the engine appends it only while an Input is
            // focused, so non-focused keystrokes are harmless.
            if let characters = event.characters,
               !characters.isEmpty,
               !event.modifierFlags.contains(.command),
               !event.modifierFlags.contains(.control) {
                dispatchText(characters)
            } else {
                super.keyDown(with: event)
            }
        }
    }
    #else
    // ── iOS adapters ─────────────────────────────────────────────────

    public override var canBecomeFirstResponder: Bool { true }

    public override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        let p = touch.location(in: self)
        lastTouchPoint = p
        _ = becomeFirstResponder()
        dispatchPointer(UInt32(SOFUU_INPUT_POINTER_DOWN), x: p.x, y: p.y)
    }

    public override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first, let previous = lastTouchPoint else { return }
        let p = touch.location(in: self)
        dispatchPointer(UInt32(SOFUU_INPUT_POINTER_MOVE), x: p.x, y: p.y)
        // A vertical drag is a scroll gesture when it lands on a scroll
        // container or plot; the engine hit-tests and ignores it elsewhere.
        let dy = p.y - previous.y
        if abs(dy) > 0.5 {
            dispatchScroll(x: p.x, y: p.y, dx: 0, dy: dy)
        }
        lastTouchPoint = p
    }

    public override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        let p = touch.location(in: self)
        dispatchPointer(UInt32(SOFUU_INPUT_POINTER_UP), x: p.x, y: p.y)
        lastTouchPoint = nil
    }

    public override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        let p = touch.location(in: self)
        dispatchPointer(UInt32(SOFUU_INPUT_POINTER_UP), x: p.x, y: p.y)
        lastTouchPoint = nil
    }
    #endif
}

// ── UIKeyInput (iOS soft keyboard) ───────────────────────────────────

#if os(iOS)
extension GpuUiMetalView: UIKeyInput {
    public var hasText: Bool { true }

    public func insertText(_ text: String) {
        dispatchText(text)
    }

    public func deleteBackward() {
        dispatchKey("Backspace")
    }
}
#endif

// ── SwiftUI wrappers ─────────────────────────────────────────────────

#if os(macOS)
/// Drop-in AppKit canvas. Attach the engine runtime from its worker thread:
///     canvas.onReady = { view in store.engine.attachUISession(view.uiSession) }
public struct GpuUiCanvas: NSViewRepresentable {
    public var title: String
    public var theme: String
    public var topInset: CGFloat
    public var onReady: ((GpuUiMetalView) -> Void)?

    public init(
        title: String = "Sofuu",
        theme: String = "light",
        topInset: CGFloat = 0,
        onReady: ((GpuUiMetalView) -> Void)? = nil
    ) {
        self.title = title
        self.theme = theme
        self.topInset = topInset
        self.onReady = onReady
    }

    public func makeNSView(context: Context) -> GpuUiMetalView {
        let view = GpuUiMetalView(frame: .zero, title: title, theme: theme)
        view.topInsetOverride = topInset
        onReady?(view)
        return view
    }

    public func updateNSView(_ nsView: GpuUiMetalView, context: Context) {}
}
#else
/// Drop-in UIKit canvas for a SwiftUI screen or a UIKit view controller.
public struct GpuUiCanvas: UIViewRepresentable {
    public var title: String
    public var theme: String
    public var topInset: CGFloat
    public var onReady: ((GpuUiMetalView) -> Void)?

    public init(
        title: String = "Sofuu",
        theme: String = "light",
        topInset: CGFloat = 0,
        onReady: ((GpuUiMetalView) -> Void)? = nil
    ) {
        self.title = title
        self.theme = theme
        self.topInset = topInset
        self.onReady = onReady
    }

    public func makeUIView(context: Context) -> GpuUiMetalView {
        let view = GpuUiMetalView(frame: .zero, title: title, theme: theme)
        view.topInsetOverride = topInset
        onReady?(view)
        return view
    }

    public func updateUIView(_ uiView: GpuUiMetalView, context: Context) {}
}
#endif
