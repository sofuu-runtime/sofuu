// Direct macOS AppKit + CAMetalLayer + Metal 2D GPU compositor host.
// Zero wgpu, zero winit, zero WebView, zero third-party Objective-C wrappers.

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <stdint.h>
#include <string.h>

typedef enum {
    SOFUU_EV_POINTER_MOVE = 1,
    SOFUU_EV_POINTER_DOWN = 2,
    SOFUU_EV_POINTER_UP = 3,
    SOFUU_EV_SCROLL = 4,
    SOFUU_EV_RESIZE = 5,
    SOFUU_EV_CLOSE = 6,
    SOFUU_EV_KEY_DOWN = 7,
} SofuuEvKind;

typedef void (*SofuuEventCallback)(
    void* user_ctx,
    uint32_t kind,
    float x,
    float y,
    float a,
    float b,
    uint32_t u
);

typedef struct {
    float viewport_size[2];
    float atlas_size[2];
    float scale_factor;
    float time_sec;
    float _pad[2];
} SofuuFrameUniforms;

@interface SofuuMetalView : NSView
@property (nonatomic, assign) SofuuEventCallback callback;
@property (nonatomic, assign) void* userCtx;
@property (nonatomic, strong) NSTrackingArea* trackingArea;
@end

@implementation SofuuMetalView

+ (Class)layerClass {
    return [CAMetalLayer class];
}

- (CALayer*)makeBackingLayer {
    return [CAMetalLayer layer];
}

- (BOOL)wantsUpdateLayer {
    return YES;
}

- (BOOL)isFlipped {
    return YES;
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)updateTrackingAreas {
    [super updateTrackingAreas];
    if (self.trackingArea != nil) {
        [self removeTrackingArea:self.trackingArea];
    }
    NSTrackingAreaOptions opts = NSTrackingMouseMoved | NSTrackingActiveAlways |
                                 NSTrackingInVisibleRect | NSTrackingMouseEnteredAndExited;
    self.trackingArea = [[NSTrackingArea alloc] initWithRect:self.bounds
                                                     options:opts
                                                       owner:self
                                                    userInfo:nil];
    [self addTrackingArea:self.trackingArea];
}

- (void)emitResize {
    if (!self.callback) return;
    CGSize sz = self.bounds.size;
    CGFloat scale = self.window ? self.window.backingScaleFactor : 1.0;
    if (scale < 1.0) scale = 1.0;
    CAMetalLayer* layer = (CAMetalLayer*)self.layer;
    layer.contentsScale = scale;
    layer.drawableSize = CGSizeMake(MAX(1.0, sz.width * scale), MAX(1.0, sz.height * scale));
    self.callback(self.userCtx, SOFUU_EV_RESIZE, (float)sz.width, (float)sz.height, (float)scale, 0.0f, 0);
}

- (void)setFrameSize:(NSSize)newSize {
    [super setFrameSize:newSize];
    [self emitResize];
}

- (void)viewDidChangeBackingProperties {
    [super viewDidChangeBackingProperties];
    [self emitResize];
}

- (void)mouseMoved:(NSEvent*)event {
    if (!self.callback) return;
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    self.callback(self.userCtx, SOFUU_EV_POINTER_MOVE, (float)p.x, (float)p.y, 0.0f, 0.0f, 0);
}

- (void)mouseDragged:(NSEvent*)event {
    [self mouseMoved:event];
}

- (void)mouseDown:(NSEvent*)event {
    if (!self.callback) return;
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    self.callback(self.userCtx, SOFUU_EV_POINTER_DOWN, (float)p.x, (float)p.y, 0.0f, 0.0f, 0);
}

- (void)mouseUp:(NSEvent*)event {
    if (!self.callback) return;
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    self.callback(self.userCtx, SOFUU_EV_POINTER_UP, (float)p.x, (float)p.y, 0.0f, 0.0f, 0);
}

- (void)rightMouseDown:(NSEvent*)event {
    if (!self.callback) return;
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    self.callback(self.userCtx, SOFUU_EV_POINTER_DOWN, (float)p.x, (float)p.y, 0.0f, 0.0f, 1);
}

- (void)rightMouseUp:(NSEvent*)event {
    if (!self.callback) return;
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    self.callback(self.userCtx, SOFUU_EV_POINTER_UP, (float)p.x, (float)p.y, 0.0f, 0.0f, 1);
}

- (void)scrollWheel:(NSEvent*)event {
    if (!self.callback) return;
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    float dx = (float)event.scrollingDeltaX;
    float dy = (float)event.scrollingDeltaY;
    if (!event.hasPreciseScrollingDeltas) {
        dx *= 24.0f;
        dy *= 24.0f;
    }
    self.callback(self.userCtx, SOFUU_EV_SCROLL, (float)p.x, (float)p.y, dx, dy, 0);
}

- (void)keyDown:(NSEvent*)event {
    if (!self.callback) return;
    self.callback(self.userCtx, SOFUU_EV_KEY_DOWN, 0.0f, 0.0f, 0.0f, 0.0f, (uint32_t)event.keyCode);
}

@end

@interface SofuuWindowDelegate : NSObject<NSWindowDelegate>
@property (nonatomic, assign) BOOL closed;
@property (nonatomic, assign) SofuuEventCallback callback;
@property (nonatomic, assign) void* userCtx;
@end

@implementation SofuuWindowDelegate
- (void)windowWillClose:(NSNotification*)notification {
    self.closed = YES;
    if (self.callback) {
        self.callback(self.userCtx, SOFUU_EV_CLOSE, 0.0f, 0.0f, 0.0f, 0.0f, 0);
    }
}
@end

typedef struct SofuuMacWindow {
    NSWindow* window;
    SofuuMetalView* view;
    SofuuWindowDelegate* delegate;
    CAMetalLayer* metalLayer;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLRenderPipelineState> quadPipeline;
    id<MTLRenderPipelineState> strokePipeline;
    id<MTLRenderPipelineState> glyphPipeline;
    id<MTLTexture> atlasTexture;
    id<MTLSamplerState> atlasSampler;
    id<MTLBuffer> clipBuf;
    id<MTLBuffer> quadBuf;
    id<MTLBuffer> strokeBuf;
    id<MTLBuffer> glyphBuf;
} SofuuMacWindow;

static id<MTLRenderPipelineState> make_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> lib,
    NSString* vertName,
    NSString* fragName,
    NSError** errOut
) {
    id<MTLFunction> vf = [lib newFunctionWithName:vertName];
    id<MTLFunction> ff = [lib newFunctionWithName:fragName];
    if (!vf || !ff) return nil;

    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vf;
    desc.fragmentFunction = ff;
    MTLRenderPipelineColorAttachmentDescriptor* att = desc.colorAttachments[0];
    att.pixelFormat = MTLPixelFormatBGRA8Unorm;
    att.blendingEnabled = YES;
    att.rgbBlendOperation = MTLBlendOperationAdd;
    att.alphaBlendOperation = MTLBlendOperationAdd;
    att.sourceRGBBlendFactor = MTLBlendFactorOne;
    att.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    att.sourceAlphaBlendFactor = MTLBlendFactorOne;
    att.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

    return [device newRenderPipelineStateWithDescriptor:desc error:errOut];
}

static id<MTLBuffer> ensure_upload_buffer(
    id<MTLDevice> device,
    __strong id<MTLBuffer>* slot,
    const void* data,
    NSUInteger bytes
) {
    if (bytes == 0 || data == NULL) return nil;
    id<MTLBuffer> buf = *slot;
    if (!buf || buf.length < bytes) {
        NSUInteger cap = 4096;
        while (cap < bytes) cap *= 2;
        buf = [device newBufferWithLength:cap options:MTLResourceStorageModeShared];
        *slot = buf;
    }
    memcpy(buf.contents, data, bytes);
    return buf;
}

SofuuMacWindow* sofuu_macos_create_window(
    const char* title_utf8,
    float width,
    float height,
    float min_w,
    float min_h,
    const char* msl_source_utf8,
    const uint8_t* atlas_pixels,
    uint32_t atlas_w,
    uint32_t atlas_h,
    SofuuEventCallback cb,
    void* user_ctx
) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp finishLaunching];

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return NULL;

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:msl_source_utf8];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            NSLog(@"[sofuu-gpu] MSL compile error: %@", err);
            return NULL;
        }

        id<MTLRenderPipelineState> quadPipe = make_pipeline(device, lib, @"quad_vert", @"quad_frag", &err);
        id<MTLRenderPipelineState> strokePipe = make_pipeline(device, lib, @"stroke_vert", @"stroke_frag", &err);
        id<MTLRenderPipelineState> glyphPipe = make_pipeline(device, lib, @"glyph_vert", @"glyph_frag", &err);
        if (!quadPipe || !strokePipe || !glyphPipe) {
            NSLog(@"[sofuu-gpu] pipeline creation failed: %@", err);
            return NULL;
        }

        MTLTextureDescriptor* texDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                         width:atlas_w
                                        height:atlas_h
                                     mipmapped:NO];
        texDesc.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> atlasTex = [device newTextureWithDescriptor:texDesc];
        [atlasTex replaceRegion:MTLRegionMake2D(0, 0, atlas_w, atlas_h)
                    mipmapLevel:0
                      withBytes:atlas_pixels
                    bytesPerRow:atlas_w];

        MTLSamplerDescriptor* sampDesc = [[MTLSamplerDescriptor alloc] init];
        sampDesc.minFilter = MTLSamplerMinMagFilterLinear;
        sampDesc.magFilter = MTLSamplerMinMagFilterLinear;
        sampDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
        sampDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
        id<MTLSamplerState> atlasSampler = [device newSamplerStateWithDescriptor:sampDesc];

        NSRect frame = NSMakeRect(0, 0, MAX(200.0f, width), MAX(160.0f, height));
        NSWindowStyleMask style = NSWindowStyleMaskTitled |
                                  NSWindowStyleMaskClosable |
                                  NSWindowStyleMaskMiniaturizable |
                                  NSWindowStyleMaskResizable;
        NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:style
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        window.title = title_utf8 ? [NSString stringWithUTF8String:title_utf8] : @"Sofuu UI";
        window.contentMinSize = NSMakeSize(MAX(200.0f, min_w), MAX(160.0f, min_h));
        window.acceptsMouseMovedEvents = YES;
        [window center];

        SofuuMetalView* view = [[SofuuMetalView alloc] initWithFrame:frame];
        view.wantsLayer = YES;
        view.callback = cb;
        view.userCtx = user_ctx;

        CAMetalLayer* metalLayer = (CAMetalLayer*)view.layer;
        metalLayer.device = device;
        metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        metalLayer.framebufferOnly = YES;
        metalLayer.presentsWithTransaction = NO;
        CGFloat scale = window.backingScaleFactor > 0.0 ? window.backingScaleFactor : 1.0;
        metalLayer.contentsScale = scale;
        metalLayer.drawableSize = CGSizeMake(frame.size.width * scale, frame.size.height * scale);

        SofuuWindowDelegate* delegate = [[SofuuWindowDelegate alloc] init];
        delegate.closed = NO;
        delegate.callback = cb;
        delegate.userCtx = user_ctx;
        window.delegate = delegate;
        window.contentView = view;

        [window makeKeyAndOrderFront:nil];
        [window makeFirstResponder:view];
        [NSApp activateIgnoringOtherApps:YES];

        SofuuMacWindow* handle = (SofuuMacWindow*)calloc(1, sizeof(SofuuMacWindow));
        handle->window = window;
        handle->view = view;
        handle->delegate = delegate;
        handle->metalLayer = metalLayer;
        handle->device = device;
        handle->queue = [device newCommandQueue];
        handle->quadPipeline = quadPipe;
        handle->strokePipeline = strokePipe;
        handle->glyphPipeline = glyphPipe;
        handle->atlasTexture = atlasTex;
        handle->atlasSampler = atlasSampler;
        return handle;
    }
}

void sofuu_macos_destroy_window(SofuuMacWindow* win) {
    if (!win) return;
    @autoreleasepool {
        win->view.callback = NULL;
        win->delegate.callback = NULL;
        [win->window close];
        win->clipBuf = nil;
        win->quadBuf = nil;
        win->strokeBuf = nil;
        win->glyphBuf = nil;
        win->atlasSampler = nil;
        win->atlasTexture = nil;
        win->quadPipeline = nil;
        win->strokePipeline = nil;
        win->glyphPipeline = nil;
        win->queue = nil;
        win->device = nil;
        win->metalLayer = nil;
        win->view = nil;
        win->delegate = nil;
        win->window = nil;
        free(win);
    }
}

void sofuu_macos_wake(void) {
    @autoreleasepool {
        if (!NSApp) return;
        NSEvent* wake = [NSEvent otherEventWithType:NSEventTypeApplicationDefined
                                           location:NSZeroPoint
                                      modifierFlags:0
                                          timestamp:0
                                       windowNumber:0
                                            context:nil
                                            subtype:0
                                              data1:0
                                              data2:0];
        [NSApp postEvent:wake atStart:NO];
    }
}

void sofuu_macos_get_metrics(SofuuMacWindow* win, float* out_w, float* out_h, float* out_scale) {
    if (!win) return;
    CGSize sz = win->view.bounds.size;
    CGFloat scale = win->window.backingScaleFactor;
    if (scale < 1.0) scale = 1.0;
    *out_w = (float)sz.width;
    *out_h = (float)sz.height;
    *out_scale = (float)scale;
}

void sofuu_macos_render(
    SofuuMacWindow* win,
    const SofuuFrameUniforms* uniforms,
    const void* clips,
    uint32_t clips_count,
    const void* quads,
    uint32_t quads_count,
    const void* strokes,
    uint32_t strokes_count,
    const void* glyphs,
    uint32_t glyphs_count
) {
    if (!win || !win->metalLayer) return;
    @autoreleasepool {
        id<CAMetalDrawable> drawable = [win->metalLayer nextDrawable];
        if (!drawable) return;

        id<MTLBuffer> clipB = ensure_upload_buffer(win->device, &win->clipBuf, clips, (NSUInteger)clips_count * 16);
        id<MTLBuffer> quadB = ensure_upload_buffer(win->device, &win->quadBuf, quads, (NSUInteger)quads_count * 80);
        id<MTLBuffer> strokeB = ensure_upload_buffer(win->device, &win->strokeBuf, strokes, (NSUInteger)strokes_count * 32);
        id<MTLBuffer> glyphB = ensure_upload_buffer(win->device, &win->glyphBuf, glyphs, (NSUInteger)glyphs_count * 48);

        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.043, 0.059, 0.078, 1.0);

        id<MTLCommandBuffer> cmd = [win->queue commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];

        [enc setVertexBytes:uniforms length:sizeof(SofuuFrameUniforms) atIndex:0];
        [enc setFragmentBytes:uniforms length:sizeof(SofuuFrameUniforms) atIndex:0];
        if (clipB) {
            [enc setFragmentBuffer:clipB offset:0 atIndex:1];
        }

        if (quadB && quads_count > 0) {
            [enc setRenderPipelineState:win->quadPipeline];
            [enc setVertexBuffer:quadB offset:0 atIndex:1];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:6
                  instanceCount:quads_count];
        }

        if (strokeB && strokes_count > 0) {
            [enc setRenderPipelineState:win->strokePipeline];
            [enc setVertexBuffer:strokeB offset:0 atIndex:1];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:6
                  instanceCount:strokes_count];
        }

        if (glyphB && glyphs_count > 0) {
            [enc setRenderPipelineState:win->glyphPipeline];
            [enc setVertexBuffer:glyphB offset:0 atIndex:1];
            [enc setFragmentTexture:win->atlasTexture atIndex:0];
            [enc setFragmentSamplerState:win->atlasSampler atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:6
                  instanceCount:glyphs_count];
        }

        [enc endEncoding];
        [cmd presentDrawable:drawable];
        [cmd commit];
    }
}

int sofuu_macos_poll_or_wait(SofuuMacWindow* win, int wait_forever, double timeout_sec) {
    if (!win || win->delegate.closed) return 0;
    @autoreleasepool {
        NSDate* limit = wait_forever
            ? [NSDate distantFuture]
            : [NSDate dateWithTimeIntervalSinceNow:MAX(0.0, timeout_sec)];

        NSEvent* ev = [NSApp nextEventMatchingMask:NSEventMaskAny
                                         untilDate:limit
                                            inMode:NSDefaultRunLoopMode
                                           dequeue:YES];
        if (ev) {
            if (ev.type != NSEventTypeApplicationDefined) {
                [NSApp sendEvent:ev];
            }
            // Drain all already-queued events in the same batch for minimum input latency.
            while ((ev = [NSApp nextEventMatchingMask:NSEventMaskAny
                                            untilDate:nil
                                               inMode:NSDefaultRunLoopMode
                                              dequeue:YES])) {
                if (ev.type != NSEventTypeApplicationDefined) {
                    [NSApp sendEvent:ev];
                }
            }
        }
        [NSApp updateWindows];
        return win->delegate.closed ? 0 : 1;
    }
}
