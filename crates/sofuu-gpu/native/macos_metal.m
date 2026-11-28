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
    SOFUU_EV_TEXT_CHAR = 8,
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

@class SofuuStripController;
@protocol SofuuStripActions <NSObject>
- (void)send:(NSString*)action;
- (void)toggleCollapse:(id)sender;
@end
static id<SofuuStripActions> g_stripController = nil;

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
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    if (p.y <= 30.0 && self.window) {
        if (event.clickCount == 2) {
            [self.window performZoom:nil];
            return;
        }
        [self.window performWindowDragWithEvent:event];
        return;
    }
    if (!self.callback) return;
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
    NSEventModifierFlags mods = event.modifierFlags;
    if ((mods & NSEventModifierFlagCommand) && g_stripController) {
        NSString* key = event.charactersIgnoringModifiers.lowercaseString;
        if ([key isEqualToString:@"."]) {
            [(id)g_stripController toggleCollapse:nil];
            return;
        }
        if ([key isEqualToString:@"1"]) { [(id)g_stripController send:@"platform:ios"]; return; }
        if ([key isEqualToString:@"2"]) { [(id)g_stripController send:@"platform:android"]; return; }
        if ([key isEqualToString:@"3"]) { [(id)g_stripController send:@"platform:macos"]; return; }
        if ([key isEqualToString:@"4"]) { [(id)g_stripController send:@"platform:windows"]; return; }
        if ([key isEqualToString:@"5"]) { [(id)g_stripController send:@"platform:linux"]; return; }
        if ([key isEqualToString:@"r"]) { [(id)g_stripController send:@"rotate"]; return; }
        if ([key isEqualToString:@"k"]) { [(id)g_stripController send:@"keyboard:toggle"]; return; }
        if ([key isEqualToString:@"t"]) { [(id)g_stripController send:@"theme:toggle"]; return; }
        if ([key isEqualToString:@"i"]) { [(id)g_stripController send:@"bounds:toggle"]; return; }
        if ([key isEqualToString:@"s"]) { [(id)g_stripController send:@"snapshot"]; return; }
    }
    if (!self.callback) return;
    self.callback(self.userCtx, SOFUU_EV_KEY_DOWN, 0.0f, 0.0f, 0.0f, 0.0f, (uint32_t)event.keyCode);
    if ((mods & (NSEventModifierFlagCommand | NSEventModifierFlagControl)) == 0) {
        NSString* chars = event.characters;
        NSUInteger n = chars.length;
        for (NSUInteger i = 0; i < n; i++) {
            unichar c = [chars characterAtIndex:i];
            if (c >= 0x20 && c != 0x7F && !(c >= 0xF700 && c <= 0xF8FF)) {
                self.callback(self.userCtx, SOFUU_EV_TEXT_CHAR, 0.0f, 0.0f, 0.0f, 0.0f, (uint32_t)c);
            }
        }
    }
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

// ── E6: attached, non-activating emulator control strip ────────────────────
//
// The strip is host chrome: it is a separate NSPanel so clicking it never
// steals keyboard focus from the screen window, and it is never part of the
// app Scene/DrawList, so captures cannot contain it.

typedef void (*SofuuStripCallback)(void* user_ctx, const char* action);

@interface SofuuStripPanel : NSPanel
@end

@implementation SofuuStripPanel
- (BOOL)canBecomeKeyWindow { return NO; }
- (BOOL)canBecomeMainWindow { return NO; }
@end

@interface SofuuStripController : NSObject<SofuuStripActions>
@property (nonatomic, strong) SofuuStripPanel* panel;
@property (nonatomic, weak) NSWindow* mainWindow;
@property (nonatomic, assign) SofuuStripCallback callback;
@property (nonatomic, assign) void* userCtx;
@property (nonatomic, assign) BOOL collapsed;
@property (nonatomic, assign) BOOL desktop;
@property (nonatomic, assign) BOOL landscape;
@property (nonatomic, strong) NSStackView* stack;
@property (nonatomic, strong) NSPopUpButton* platformPopup;
@property (nonatomic, strong) NSPopUpButton* devicePopup;
@property (nonatomic, strong) NSButton* collapseButton;
@property (nonatomic, strong) NSButton* pauseButton;
@property (nonatomic, strong) NSButton* stepButton;
@property (nonatomic, assign) BOOL paused;
@property (nonatomic, strong) NSArray* profiles;
@property (nonatomic, copy) NSString* currentPlatform;
@property (nonatomic, copy) NSString* currentDevice;
- (void)send:(NSString*)action;
- (void)toggleCollapse:(id)sender;
- (void)setCollapsed:(BOOL)collapsed;
@end

@implementation SofuuStripController

- (instancetype)initWithMainWindow:(NSWindow*)mainWindow
                     profilesJSON:(const char*)profilesJSON
                         platform:(const char*)platformUTF8
                           device:(const char*)deviceUTF8
                         callback:(SofuuStripCallback)callback
                          userCtx:(void*)userCtx
                          desktop:(BOOL)desktop {
    self = [super init];
    if (!self) return nil;
    _mainWindow = mainWindow;
    _callback = callback;
    _userCtx = userCtx;
    _desktop = desktop;
    _collapsed = NO;
    _profiles = @[];
    if (profilesJSON) {
        NSData* data = [NSData dataWithBytes:profilesJSON length:strlen(profilesJSON)];
        id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if ([root isKindOfClass:[NSDictionary class]] &&
            [root[@"profiles"] isKindOfClass:[NSArray class]]) {
            _profiles = root[@"profiles"];
        }
    }
    if (platformUTF8 && platformUTF8[0]) {
        _currentPlatform = [NSString stringWithUTF8String:platformUTF8];
    }
    if (deviceUTF8 && deviceUTF8[0]) {
        _currentDevice = [NSString stringWithUTF8String:deviceUTF8];
    }
    if (!_currentPlatform && _profiles.count > 0) {
        _currentPlatform = _profiles[0][@"id"] ?: @"ios";
    }
    if (!_currentDevice) {
        NSDictionary* profile = [self currentProfile];
        NSArray* devices = profile[@"devices"];
        if ([devices isKindOfClass:[NSArray class]] && devices.count > 0) {
            _currentDevice = devices[0][@"id"] ?: @"";
        }
    }
    [self buildPanel];
    [self layoutPanel];
    NSNotificationCenter* nc = [NSNotificationCenter defaultCenter];
    [nc addObserver:self selector:@selector(mainWindowMoved:)
               name:NSWindowDidMoveNotification object:mainWindow];
    [nc addObserver:self selector:@selector(mainWindowMoved:)
               name:NSWindowDidResizeNotification object:mainWindow];
    [nc addObserver:self selector:@selector(mainWindowMoved:)
               name:NSWindowDidChangeScreenNotification object:mainWindow];
    return self;
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)send:(NSString*)action {
    if (self.callback) self.callback(self.userCtx, action.UTF8String);
}

- (NSButton*)button:(NSString*)title action:(SEL)sel {
    NSButton* b = [NSButton buttonWithTitle:title target:self action:sel];
    b.bezelStyle = NSBezelStyleRounded;
    b.controlSize = NSControlSizeSmall;
    b.font = [NSFont systemFontOfSize:11.0 weight:NSFontWeightMedium];
    return b;
}

- (void)buildPanel {
    CGFloat w = 176.0;
    CGFloat h = 404.0;
    SofuuStripPanel* panel = [[SofuuStripPanel alloc]
        initWithContentRect:NSMakeRect(0, 0, w, h)
                  styleMask:NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
                    backing:NSBackingStoreBuffered
                      defer:NO];
    panel.level = NSFloatingWindowLevel;
    panel.opaque = YES;
    panel.hasShadow = YES;
    panel.hidesOnDeactivate = NO;
    panel.becomesKeyOnlyIfNeeded = YES;
    panel.backgroundColor = [NSColor windowBackgroundColor];
    panel.collectionBehavior = NSWindowCollectionBehaviorCanJoinAllSpaces |
                               NSWindowCollectionBehaviorFullScreenAuxiliary;
    self.panel = panel;

    NSView* content = panel.contentView;
    content.wantsLayer = YES;

    NSTextField* header = [NSTextField labelWithString:@"EMU"];
    header.font = [NSFont systemFontOfSize:11.0 weight:NSFontWeightBold];
    header.textColor = [NSColor secondaryLabelColor];
    self.collapseButton = [self button:@"»" action:@selector(toggleCollapse:)];
    self.collapseButton.toolTip = @"Collapse strip (⌘.)";
    NSStackView* head = [NSStackView stackViewWithViews:@[header, self.collapseButton]];
    head.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    head.alignment = NSLayoutAttributeCenterY;
    head.distribution = NSStackViewDistributionFill;
    [head setHuggingPriority:NSLayoutPriorityDefaultLow forOrientation:NSLayoutConstraintOrientationHorizontal];

    self.platformPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(0, 0, w - 24, 24) pullsDown:NO];
    self.platformPopup.controlSize = NSControlSizeSmall;
    self.platformPopup.font = [NSFont systemFontOfSize:11.0];
    [self.platformPopup setTarget:self];
    [self.platformPopup setAction:@selector(platformChanged:)];
    for (NSDictionary* p in self.profiles) {
        NSString* name = p[@"name"] ?: p[@"id"] ?: @"Platform";
        [self.platformPopup addItemWithTitle:name];
        self.platformPopup.lastItem.representedObject = p[@"id"];
    }
    if (self.currentPlatform) {
        for (NSMenuItem* item in self.platformPopup.itemArray) {
            if ([item.representedObject isEqual:self.currentPlatform]) {
                [self.platformPopup selectItem:item];
                break;
            }
        }
    }

    self.devicePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(0, 0, w - 24, 24) pullsDown:NO];
    self.devicePopup.controlSize = NSControlSizeSmall;
    self.devicePopup.font = [NSFont systemFontOfSize:11.0];
    [self.devicePopup setTarget:self];
    [self.devicePopup setAction:@selector(deviceChanged:)];
    [self reloadDevices];

    NSArray* rows = @[
        [self button:@"Theme" action:@selector(toggleTheme:)],
        [self button:@"Rotate" action:@selector(rotate:)],
        [self button:@"Keyboard" action:@selector(toggleKeyboard:)],
        [self button:@"Bounds" action:@selector(toggleBounds:)],
        [self button:@"Snapshot" action:@selector(snapshot:)],
        [self button:@"Watch" action:@selector(toggleWatch:)],
    ];
    self.pauseButton = [self button:@"Pause" action:@selector(togglePause:)];
    self.pauseButton.toolTip = @"Freeze the engine clock (step-frame)";
    self.stepButton = [self button:@"Step +16.6ms" action:@selector(stepFrame:)];
    self.stepButton.toolTip = @"Advance one 60 Hz frame while paused";
    NSStackView* frameRow = [NSStackView stackViewWithViews:@[
        self.pauseButton, self.stepButton,
    ]];
    frameRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    frameRow.distribution = NSStackViewDistributionFillEqually;
    frameRow.spacing = 6.0;
    self.stack = [NSStackView stackViewWithViews:@[
        head, self.platformPopup, self.devicePopup,
        rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], frameRow,
    ]];
    self.stack.orientation = NSUserInterfaceLayoutOrientationVertical;
    self.stack.alignment = NSLayoutAttributeLeading;
    self.stack.spacing = 6.0;
    self.stack.edgeInsets = NSEdgeInsetsMake(8, 8, 8, 8);
    self.stack.translatesAutoresizingMaskIntoConstraints = NO;
    [content addSubview:self.stack];
    [NSLayoutConstraint activateConstraints:@[
        [self.stack.leadingAnchor constraintEqualToAnchor:content.leadingAnchor],
        [self.stack.trailingAnchor constraintEqualToAnchor:content.trailingAnchor],
        [self.stack.topAnchor constraintEqualToAnchor:content.topAnchor],
    ]];
    for (NSView* v in self.stack.arrangedSubviews) {
        [v.widthAnchor constraintEqualToConstant:w - 16].active = YES;
    }
    [panel orderFront:nil];
}

- (void)reloadDevices {
    [self.devicePopup removeAllItems];
    NSDictionary* profile = [self currentProfile];
    NSArray* devices = profile[@"devices"];
    if (![devices isKindOfClass:[NSArray class]]) return;
    for (NSDictionary* d in devices) {
        NSString* name = d[@"name"] ?: d[@"id"] ?: @"Device";
        [self.devicePopup addItemWithTitle:name];
        self.devicePopup.lastItem.representedObject = d[@"id"];
    }
    if (self.currentDevice) {
        for (NSMenuItem* item in self.devicePopup.itemArray) {
            if ([item.representedObject isEqual:self.currentDevice]) {
                [self.devicePopup selectItem:item];
                break;
            }
        }
    }
}

- (NSDictionary*)currentProfile {
    for (NSDictionary* p in self.profiles) {
        if ([p[@"id"] isEqual:self.currentPlatform]) return p;
    }
    return self.profiles.count ? self.profiles[0] : nil;
}

- (NSDictionary*)currentDeviceDict {
    NSDictionary* profile = [self currentProfile];
    NSArray* devices = profile[@"devices"];
    if (![devices isKindOfClass:[NSArray class]]) return nil;
    for (NSDictionary* d in devices) {
        if ([d[@"id"] isEqual:self.currentDevice]) return d;
    }
    return devices.count ? devices[0] : nil;
}

- (void)platformChanged:(NSPopUpButton*)sender {
    NSString* pid = sender.selectedItem.representedObject;
    if (!pid) return;
    self.currentPlatform = pid;
    self.currentDevice = nil;
    [self reloadDevices];
    [self resizeForCurrentDevice];
    [self send:[NSString stringWithFormat:@"platform:%@", pid]];
}

- (void)deviceChanged:(NSPopUpButton*)sender {
    NSString* did = sender.selectedItem.representedObject;
    if (!did) return;
    self.currentDevice = did;
    [self resizeForCurrentDevice];
    [self send:[NSString stringWithFormat:@"device:%@", did]];
}

- (void)resizeForCurrentDevice {
    NSDictionary* d = [self currentDeviceDict];
    if (!d) return;
    CGFloat w = [d[@"width"] doubleValue];
    CGFloat h = [d[@"height"] doubleValue];
    if (self.landscape) {
        CGFloat t = w;
        w = h;
        h = t;
    }
    if (w > 100.0 && h > 100.0) {
        [self.mainWindow setContentSize:NSMakeSize(w, h)];
        [self layoutPanel];
    }
}

- (void)toggleTheme:(id)sender { [self send:@"theme:toggle"]; }
- (void)rotate:(id)sender {
    self.landscape = !self.landscape;
    [self resizeForCurrentDevice];
    [self send:@"rotate"];
}
- (void)toggleKeyboard:(id)sender { [self send:@"keyboard:toggle"]; }
- (void)toggleBounds:(id)sender { [self send:@"bounds:toggle"]; }
- (void)snapshot:(id)sender { [self send:@"snapshot"]; }
- (void)toggleWatch:(id)sender { [self send:@"watch:toggle"]; }
- (void)togglePause:(id)sender {
    self.paused = !self.paused;
    self.pauseButton.title = self.paused ? @"Resume" : @"Pause";
    [self send:self.paused ? @"pause" : @"resume"];
}
- (void)stepFrame:(id)sender {
    if (!self.paused) {
        self.paused = YES;
        self.pauseButton.title = @"Resume";
        [self send:@"pause"];
    }
    [self send:@"step"];
}

- (void)toggleCollapse:(id)sender {
    [self setCollapsed:!self.collapsed];
}

- (void)setCollapsed:(BOOL)collapsed {
    if (self.collapsed == collapsed) {
        [self layoutPanel];
        return;
    }
    self.collapsed = collapsed;
    if (collapsed) {
        [self.stack setHidden:YES];
        [self.panel setContentSize:NSMakeSize(28, 28)];
        self.collapseButton = [self button:@"«" action:@selector(toggleCollapse:)];
        self.collapseButton.frame = NSMakeRect(0, 0, 28, 28);
        [self.panel.contentView addSubview:self.collapseButton];
        [self send:@"collapse"];
    } else {
        [self.collapseButton removeFromSuperview];
        [self.stack setHidden:NO];
        [self.panel setContentSize:NSMakeSize(176, 404)];
        [self send:@"expand"];
    }
    [self.panel orderFront:nil];
    [self layoutPanel];
}

- (void)mainWindowMoved:(NSNotification*)note {
    [self layoutPanel];
}

- (void)layoutPanel {
    NSWindow* main = self.mainWindow;
    if (!main) return;
    NSRect mf = main.frame;
    NSSize ps = self.panel.frame.size;
    NSScreen* screen = main.screen ?: [NSScreen mainScreen];
    NSRect visible = screen.visibleFrame;
    const CGFloat gap = 2.0;

    /* Adjacent slots, in preference order. The strip must never overlap the
     * screen: if the preferred slot is off-screen we try the other three
     * sides before collapsing to the 28 pt tab. */
    NSPoint slots[4];
    int count = 0;
    if (self.desktop) {
        slots[count++] = NSMakePoint(NSMinX(mf), NSMaxY(mf) + gap);
        slots[count++] = NSMakePoint(NSMinX(mf), NSMinY(mf) - ps.height - gap);
        slots[count++] = NSMakePoint(NSMaxX(mf) + gap, NSMaxY(mf) - ps.height);
        slots[count++] = NSMakePoint(NSMinX(mf) - ps.width - gap, NSMaxY(mf) - ps.height);
    } else {
        slots[count++] = NSMakePoint(NSMaxX(mf) + gap, NSMaxY(mf) - ps.height);
        slots[count++] = NSMakePoint(NSMinX(mf) - ps.width - gap, NSMaxY(mf) - ps.height);
        slots[count++] = NSMakePoint(NSMinX(mf), NSMaxY(mf) + gap);
        slots[count++] = NSMakePoint(NSMinX(mf), NSMinY(mf) - ps.height - gap);
    }
    for (int i = 0; i < count; i++) {
        NSRect r = NSMakeRect(slots[i].x, slots[i].y, ps.width, ps.height);
        if (NSContainsRect(visible, r)) {
            [self.panel setFrameOrigin:r.origin];
            return;
        }
    }
    if (!self.collapsed) {
        [self setCollapsed:YES];
        return;
    }
    /* No adjacent slot fits even for the tab: hide rather than paint over the
     * simulated screen. The hotkey still brings it back. */
    [self.panel orderOut:nil];
}

@end

typedef struct {
    uint32_t kind;
    uint32_t start;
    uint32_t count;
    uint32_t _pad;
} SofuuDrawCmd;

// Mirrors `sofuu_ui::BackdropRegion` (64 B). Only the blur radius is read here; the
// attributes are consumed by the backdrop pipeline.
typedef struct {
    float rect[4];
    float radii[4];
    uint32_t tint;
    uint32_t rim;
    float params[4];
    uint32_t clip_id;
    uint32_t _pad;
} SofuuBackdropRegion;

@interface SofuuMacWindow : NSObject
@property (nonatomic, strong) NSWindow* window;
@property (nonatomic, strong) SofuuMetalView* view;
@property (nonatomic, strong) SofuuWindowDelegate* delegate;
@property (nonatomic, strong) CAMetalLayer* metalLayer;
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLRenderPipelineState> quadPipeline;
@property (nonatomic, strong) id<MTLRenderPipelineState> ribbonPipeline;
@property (nonatomic, strong) id<MTLRenderPipelineState> strokePipeline;
@property (nonatomic, strong) id<MTLRenderPipelineState> glyphPipeline;
@property (nonatomic, strong) id<MTLRenderPipelineState> effectPipeline;
@property (nonatomic, strong) id<MTLRenderPipelineState> particlePipeline;
@property (nonatomic, strong) id<MTLRenderPipelineState> backdropPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> blurDownPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> blurKawasePipeline;
@property (nonatomic, strong) id<MTLTexture> atlasTexture;
@property (nonatomic, strong) id<MTLSamplerState> atlasSampler;
@property (nonatomic, strong) id<MTLTexture> snapshotTexture;
@property (nonatomic, strong) id<MTLTexture> blurTextureA;
@property (nonatomic, strong) id<MTLTexture> blurTextureB;
@property (nonatomic, assign) NSUInteger blurWidth;
@property (nonatomic, assign) NSUInteger blurHeight;
@property (nonatomic, strong) id<MTLBuffer> clipBuf;
@property (nonatomic, strong) id<MTLBuffer> quadBuf;
@property (nonatomic, strong) id<MTLBuffer> ribbonBuf;
@property (nonatomic, strong) id<MTLBuffer> strokeBuf;
@property (nonatomic, strong) id<MTLBuffer> glyphBuf;
@property (nonatomic, strong) id<MTLBuffer> effectBuf;
@property (nonatomic, strong) id<MTLBuffer> particleBuf;
@property (nonatomic, strong) id<MTLBuffer> backdropBuf;
@end

@implementation SofuuMacWindow
@end

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

static id<MTLComputePipelineState> make_compute_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> lib,
    NSString* name,
    NSError** errOut
) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [device newComputePipelineStateWithFunction:fn error:errOut];
}

static id<MTLTexture> make_texture(
    id<MTLDevice> device,
    NSUInteger width,
    NSUInteger height,
    MTLTextureUsage usage
) {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:MAX(1, width)
                                    height:MAX(1, height)
                                 mipmapped:NO];
    desc.usage = usage;
    desc.storageMode = MTLStorageModePrivate;
    return [device newTextureWithDescriptor:desc];
}

static void ensure_blur_textures(SofuuMacWindow* win, NSUInteger width, NSUInteger height) {
    if (win.snapshotTexture && win.blurWidth == width && win.blurHeight == height) return;
    win.blurWidth = width;
    win.blurHeight = height;
    win.snapshotTexture = make_texture(win.device, width, height, MTLTextureUsageShaderRead);
    win.blurTextureA = make_texture(win.device, width / 2, height / 2,
                                    MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    win.blurTextureB = make_texture(win.device, width / 2, height / 2,
                                    MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
}

static id<MTLBuffer> ensure_upload_buffer(
    id<MTLDevice> device,
    id<MTLBuffer> buf,
    const void* data,
    NSUInteger bytes
) {
    if (bytes == 0 || data == NULL) return buf;
    if (!buf || buf.length < bytes) {
        NSUInteger cap = 4096;
        while (cap < bytes) cap *= 2;
        buf = [device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    }
    memcpy(buf.contents, data, bytes);
    return buf;
}

static SofuuMacWindow* create_renderer(
    const char* msl_source_utf8,
    const uint8_t* atlas_pixels,
    uint32_t atlas_w,
    uint32_t atlas_h
) {
    @autoreleasepool {
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
        id<MTLRenderPipelineState> ribbonPipe = make_pipeline(device, lib, @"ribbon_vert", @"ribbon_frag", &err);
        id<MTLRenderPipelineState> strokePipe = make_pipeline(device, lib, @"stroke_vert", @"stroke_frag", &err);
        id<MTLRenderPipelineState> glyphPipe = make_pipeline(device, lib, @"glyph_vert", @"glyph_frag", &err);
        id<MTLRenderPipelineState> effectPipe = make_pipeline(device, lib, @"effect_vert", @"effect_frag", &err);
        id<MTLRenderPipelineState> particlePipe = make_pipeline(device, lib, @"particle_vert", @"particle_frag", &err);
        id<MTLRenderPipelineState> backdropPipe = make_pipeline(device, lib, @"backdrop_vert", @"backdrop_frag", &err);
        id<MTLComputePipelineState> blurDownPipe = make_compute_pipeline(device, lib, @"blur_downsample", &err);
        id<MTLComputePipelineState> blurKawasePipe = make_compute_pipeline(device, lib, @"blur_kawase", &err);
        if (!quadPipe || !ribbonPipe || !strokePipe || !glyphPipe ||
            !effectPipe || !particlePipe || !backdropPipe ||
            !blurDownPipe || !blurKawasePipe) {
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

        SofuuMacWindow* r = [[SofuuMacWindow alloc] init];
        r.device = device;
        r.queue = [device newCommandQueue];
        r.quadPipeline = quadPipe;
        r.ribbonPipeline = ribbonPipe;
        r.strokePipeline = strokePipe;
        r.glyphPipeline = glyphPipe;
        r.effectPipeline = effectPipe;
        r.particlePipeline = particlePipe;
        r.backdropPipeline = backdropPipe;
        r.blurDownPipeline = blurDownPipe;
        r.blurKawasePipeline = blurKawasePipe;
        r.atlasTexture = atlasTex;
        r.atlasSampler = atlasSampler;
        return r;
    }
}

void* sofuu_macos_create_window_ex(
    const char* title_utf8,
    float width,
    float height,
    float min_w,
    float min_h,
    int decorations,
    int resizable,
    float scale_cap,
    float corner_radius,
    int strip,
    const char* platform_utf8,
    const char* device_utf8,
    const char* profiles_json,
    const char* msl_source_utf8,
    const uint8_t* atlas_pixels,
    uint32_t atlas_w,
    uint32_t atlas_h,
    SofuuEventCallback cb,
    void* user_ctx,
    SofuuStripCallback strip_cb,
    void* strip_ctx
) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp finishLaunching];

        SofuuMacWindow* handle = create_renderer(
            msl_source_utf8, atlas_pixels, atlas_w, atlas_h);
        if (!handle) return NULL;
        id<MTLDevice> device = handle.device;

        NSRect frame = NSMakeRect(0, 0, MAX(200.0f, width), MAX(160.0f, height));
        NSWindowStyleMask style;
        if (decorations) {
            style = NSWindowStyleMaskTitled |
                    NSWindowStyleMaskClosable |
                    NSWindowStyleMaskMiniaturizable |
                    NSWindowStyleMaskFullSizeContentView;
            if (resizable) style |= NSWindowStyleMaskResizable;
        } else {
            style = NSWindowStyleMaskBorderless;
        }
        NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:style
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        window.releasedWhenClosed = NO;
        window.title = title_utf8 ? [NSString stringWithUTF8String:title_utf8] : @"";
        window.titleVisibility = NSWindowTitleHidden;
        window.titlebarAppearsTransparent = YES;
        if (@available(macOS 11.0, *)) {
            window.titlebarSeparatorStyle = NSTitlebarSeparatorStyleNone;
        }
        window.backgroundColor = [NSColor clearColor];
        window.opaque = NO;
        window.hasShadow = YES;
        if (resizable) {
            window.contentMinSize = NSMakeSize(MAX(200.0f, min_w), MAX(160.0f, min_h));
        }
        window.acceptsMouseMovedEvents = YES;
        [window center];
        NSLog(@"[sofuu-gpu] screen window %.0fx%.0f decorations=%d resizable=%d scale_cap=%.2f",
              frame.size.width, frame.size.height, decorations, resizable, scale_cap);

        SofuuMetalView* view = [[SofuuMetalView alloc] initWithFrame:frame];
        view.wantsLayer = YES;
        view.callback = cb;
        view.userCtx = user_ctx;

        CAMetalLayer* metalLayer = (CAMetalLayer*)view.layer;
        metalLayer.device = device;
        metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        // Backdrop snapshots require blitting from the drawable into the blur chain.
        metalLayer.framebufferOnly = NO;
        metalLayer.presentsWithTransaction = NO;
        metalLayer.opaque = NO;
        metalLayer.cornerRadius = MAX(0.0f, corner_radius);
        metalLayer.masksToBounds = YES;
        if (@available(macOS 10.15, *)) {
            metalLayer.cornerCurve = kCACornerCurveContinuous;
        }
        CGFloat scale = window.backingScaleFactor > 0.0 ? window.backingScaleFactor : 1.0;
        if (scale_cap > 0.0f) scale = MIN(scale, (CGFloat)scale_cap);
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

        if (strip && strip_cb) {
            NSString* platform = platform_utf8 ? [NSString stringWithUTF8String:platform_utf8] : @"";
            BOOL desktop = [platform isEqualToString:@"macos"] ||
                           [platform isEqualToString:@"windows"] ||
                           [platform isEqualToString:@"linux"];
            g_stripController = [[SofuuStripController alloc]
                initWithMainWindow:window
                     profilesJSON:profiles_json
                         platform:platform_utf8
                           device:device_utf8
                         callback:strip_cb
                          userCtx:strip_ctx
                          desktop:desktop];
            NSLog(@"[sofuu-gpu] strip attached (%@) window=%@",
                  desktop ? @"top" : @"right", NSStringFromRect(window.frame));
        }

        handle.window = window;
        handle.view = view;
        handle.delegate = delegate;
        handle.metalLayer = metalLayer;
        return (__bridge_retained void*)handle;
    }
}

void* sofuu_macos_create_window(
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
    return sofuu_macos_create_window_ex(
        title_utf8, width, height, min_w, min_h,
        1, 1, 0.0f, 18.0f, 0,
        NULL, NULL, NULL,
        msl_source_utf8, atlas_pixels, atlas_w, atlas_h,
        cb, user_ctx, NULL, NULL);
}

void sofuu_macos_destroy_window(void* winPtr) {
    if (!winPtr) return;
    @autoreleasepool {
        SofuuMacWindow* win = (__bridge_transfer SofuuMacWindow*)winPtr;
        win.view.callback = NULL;
        win.delegate.callback = NULL;
        if (win.window) {
            if (!win.delegate.closed) {
                [win.window close];
            }
            win.window.delegate = nil;
            win.window.contentView = nil;
        }
        win.clipBuf = nil;
        win.quadBuf = nil;
        win.ribbonBuf = nil;
        win.strokeBuf = nil;
        win.glyphBuf = nil;
        win.effectBuf = nil;
        win.particleBuf = nil;
        win.backdropBuf = nil;
        win.snapshotTexture = nil;
        win.blurTextureA = nil;
        win.blurTextureB = nil;
        win.atlasSampler = nil;
        win.atlasTexture = nil;
        win.quadPipeline = nil;
        win.ribbonPipeline = nil;
        win.strokePipeline = nil;
        win.glyphPipeline = nil;
        win.effectPipeline = nil;
        win.particlePipeline = nil;
        win.backdropPipeline = nil;
        win.blurDownPipeline = nil;
        win.blurKawasePipeline = nil;
        win.queue = nil;
        win.device = nil;
        win.metalLayer = nil;
        win.view = nil;
        win.delegate = nil;
        win.window = nil;
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

void sofuu_macos_get_metrics(void* winPtr, float* out_w, float* out_h, float* out_scale) {
    if (!winPtr) return;
    SofuuMacWindow* win = (__bridge SofuuMacWindow*)winPtr;
    CGSize sz = win.view.bounds.size;
    CGFloat scale = win.window.backingScaleFactor;
    if (scale < 1.0) scale = 1.0;
    *out_w = (float)sz.width;
    *out_h = (float)sz.height;
    *out_scale = (float)scale;
}

static int render_drawlist(
    SofuuMacWindow* win,
    id<MTLTexture> target,
    id<CAMetalDrawable> drawable,
    int wait_for_completion,
    id<MTLBuffer> readback,
    const SofuuFrameUniforms* uniforms,
    const void* clips,
    uint32_t clips_count,
    const void* quads,
    uint32_t quads_count,
    const void* ribbons,
    uint32_t ribbons_count,
    const void* strokes,
    uint32_t strokes_count,
    const void* glyphs,
    uint32_t glyphs_count,
    const void* effects,
    uint32_t effects_count,
    const void* particles,
    uint32_t particles_count,
    const void* backdrops,
    uint32_t backdrops_count,
    const SofuuDrawCmd* cmds,
    uint32_t cmds_count
) {
    if (!win || !target) return -1;
    @autoreleasepool {
        if (clips_count > 0) win.clipBuf = ensure_upload_buffer(win.device, win.clipBuf, clips, (NSUInteger)clips_count * 16);
        if (quads_count > 0) win.quadBuf = ensure_upload_buffer(win.device, win.quadBuf, quads, (NSUInteger)quads_count * 80);
        if (ribbons_count > 0) win.ribbonBuf = ensure_upload_buffer(win.device, win.ribbonBuf, ribbons, (NSUInteger)ribbons_count * 48);
        if (strokes_count > 0) win.strokeBuf = ensure_upload_buffer(win.device, win.strokeBuf, strokes, (NSUInteger)strokes_count * 48);
        if (glyphs_count > 0) win.glyphBuf = ensure_upload_buffer(win.device, win.glyphBuf, glyphs, (NSUInteger)glyphs_count * 48);
        if (effects_count > 0) win.effectBuf = ensure_upload_buffer(win.device, win.effectBuf, effects, (NSUInteger)effects_count * 80);
        if (particles_count > 0) win.particleBuf = ensure_upload_buffer(win.device, win.particleBuf, particles, (NSUInteger)particles_count * 32);
        if (backdrops_count > 0) win.backdropBuf = ensure_upload_buffer(win.device, win.backdropBuf, backdrops, (NSUInteger)backdrops_count * 64);

        id<MTLBuffer> clipB = clips_count > 0 ? win.clipBuf : nil;
        id<MTLBuffer> quadB = quads_count > 0 ? win.quadBuf : nil;
        id<MTLBuffer> ribbonB = ribbons_count > 0 ? win.ribbonBuf : nil;
        id<MTLBuffer> strokeB = strokes_count > 0 ? win.strokeBuf : nil;
        id<MTLBuffer> glyphB = glyphs_count > 0 ? win.glyphBuf : nil;
        id<MTLBuffer> effectB = effects_count > 0 ? win.effectBuf : nil;
        id<MTLBuffer> particleB = particles_count > 0 ? win.particleBuf : nil;
        id<MTLBuffer> backdropB = backdrops_count > 0 ? win.backdropBuf : nil;

        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

        id<MTLCommandBuffer> cmd = [win.queue commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];

        void (^bind_common)(id<MTLRenderCommandEncoder>) = ^(id<MTLRenderCommandEncoder> e) {
            [e setVertexBytes:uniforms length:sizeof(SofuuFrameUniforms) atIndex:0];
            [e setFragmentBytes:uniforms length:sizeof(SofuuFrameUniforms) atIndex:0];
            if (clipB) {
                [e setFragmentBuffer:clipB offset:0 atIndex:1];
            }
            [e setFragmentTexture:win.atlasTexture atIndex:0];
            [e setFragmentSamplerState:win.atlasSampler atIndex:0];
        };
        bind_common(enc);

        uint32_t last_kind = 0xffffffffu;
        for (uint32_t i = 0; i < cmds_count; i++) {
            SofuuDrawCmd dc = cmds[i];
            if (dc.count == 0) continue;

            if (dc.kind == 7) {
                if (!backdropB || !win.blurDownPipeline || !win.blurKawasePipeline) {
                    continue;
                }
                [enc endEncoding];
                ensure_blur_textures(win, target.width, target.height);

                id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                [blit copyFromTexture:target
                          sourceSlice:0
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(target.width, target.height, 1)
                            toTexture:win.snapshotTexture
                     destinationSlice:0
                     destinationLevel:0
                    destinationOrigin:MTLOriginMake(0, 0, 0)];
                [blit endEncoding];

                NSUInteger hw = MAX(1, target.width / 2);
                NSUInteger hh = MAX(1, target.height / 2);
                MTLSize threads = MTLSizeMake(8, 8, 1);
                MTLSize groups = MTLSizeMake((hw + 7) / 8, (hh + 7) / 8, 1);
                float blur_scale = 1.0f;
                if (backdrops && dc.start < backdrops_count) {
                    const SofuuBackdropRegion* regions =
                        (const SofuuBackdropRegion*)backdrops;
                    blur_scale = regions[dc.start].params[0] / 18.0f;
                    if (blur_scale < 0.5f) blur_scale = 0.5f;
                    if (blur_scale > 3.0f) blur_scale = 3.0f;
                }

                id<MTLComputeCommandEncoder> ce = [cmd computeCommandEncoder];
                [ce setComputePipelineState:win.blurDownPipeline];
                [ce setTexture:win.snapshotTexture atIndex:0];
                [ce setTexture:win.blurTextureA atIndex:1];
                [ce setSamplerState:win.atlasSampler atIndex:0];
                [ce dispatchThreadgroups:groups threadsPerThreadgroup:threads];

                float offset = 1.5f * blur_scale;
                [ce setComputePipelineState:win.blurKawasePipeline];
                [ce setTexture:win.blurTextureA atIndex:0];
                [ce setTexture:win.blurTextureB atIndex:1];
                [ce setBytes:&offset length:sizeof(offset) atIndex:0];
                [ce dispatchThreadgroups:groups threadsPerThreadgroup:threads];

                offset = 2.5f * blur_scale;
                [ce setTexture:win.blurTextureB atIndex:0];
                [ce setTexture:win.blurTextureA atIndex:1];
                [ce setBytes:&offset length:sizeof(offset) atIndex:0];
                [ce dispatchThreadgroups:groups threadsPerThreadgroup:threads];
                [ce endEncoding];

                MTLRenderPassDescriptor* resume = [MTLRenderPassDescriptor renderPassDescriptor];
                resume.colorAttachments[0].texture = target;
                resume.colorAttachments[0].loadAction = MTLLoadActionLoad;
                resume.colorAttachments[0].storeAction = MTLStoreActionStore;
                enc = [cmd renderCommandEncoderWithDescriptor:resume];
                bind_common(enc);

                [enc setRenderPipelineState:win.backdropPipeline];
                [enc setVertexBuffer:backdropB offset:(NSUInteger)dc.start * 64 atIndex:1];
                [enc setFragmentTexture:win.blurTextureA atIndex:0];
                [enc setFragmentSamplerState:win.atlasSampler atIndex:0];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:6
                      instanceCount:1];
                [enc setFragmentTexture:win.atlasTexture atIndex:0];
                last_kind = 0xffffffffu;
                continue;
            }

            if (dc.kind == 5 && effectB) {
                if (last_kind != 5) {
                    [enc setRenderPipelineState:win.effectPipeline];
                    last_kind = 5;
                }
                [enc setVertexBuffer:effectB offset:(NSUInteger)dc.start * 80 atIndex:1];
            } else if (dc.kind == 6 && particleB) {
                if (last_kind != 6) {
                    [enc setRenderPipelineState:win.particlePipeline];
                    last_kind = 6;
                }
                [enc setVertexBuffer:particleB offset:(NSUInteger)dc.start * 32 atIndex:1];
            } else if (dc.kind == 0 && quadB) {
                if (last_kind != 0) {
                    [enc setRenderPipelineState:win.quadPipeline];
                    last_kind = 0;
                }
                [enc setVertexBuffer:quadB offset:(NSUInteger)dc.start * 80 atIndex:1];
            } else if (dc.kind == 1 && ribbonB) {
                if (last_kind != 1) {
                    [enc setRenderPipelineState:win.ribbonPipeline];
                    last_kind = 1;
                }
                [enc setVertexBuffer:ribbonB offset:(NSUInteger)dc.start * 48 atIndex:1];
            } else if (dc.kind == 2 && strokeB) {
                if (last_kind != 2) {
                    [enc setRenderPipelineState:win.strokePipeline];
                    last_kind = 2;
                }
                [enc setVertexBuffer:strokeB offset:(NSUInteger)dc.start * 48 atIndex:1];
            } else if (dc.kind == 3 && glyphB) {
                if (last_kind != 3) {
                    [enc setRenderPipelineState:win.glyphPipeline];
                    last_kind = 3;
                }
                [enc setVertexBuffer:glyphB offset:(NSUInteger)dc.start * 48 atIndex:1];
            } else {
                continue;
            }

            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:6
                  instanceCount:dc.count];
        }

        [enc endEncoding];
        if (readback) {
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromTexture:target
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(target.width, target.height, 1)
                         toBuffer:readback
                destinationOffset:0
           destinationBytesPerRow:(NSUInteger)target.width * 4
         destinationBytesPerImage:(NSUInteger)target.width * 4 * target.height];
            [blit endEncoding];
        }
        if (drawable) {
            [cmd presentDrawable:drawable];
        }
        [cmd commit];
        if (wait_for_completion) {
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) return -2;
        }
        return 0;
    }
}

void sofuu_macos_render(
    void* winPtr,
    const SofuuFrameUniforms* uniforms,
    const void* clips,
    uint32_t clips_count,
    const void* quads,
    uint32_t quads_count,
    const void* ribbons,
    uint32_t ribbons_count,
    const void* strokes,
    uint32_t strokes_count,
    const void* glyphs,
    uint32_t glyphs_count,
    const void* effects,
    uint32_t effects_count,
    const void* particles,
    uint32_t particles_count,
    const void* backdrops,
    uint32_t backdrops_count,
    const SofuuDrawCmd* cmds,
    uint32_t cmds_count
) {
    if (!winPtr) return;
    SofuuMacWindow* win = (__bridge SofuuMacWindow*)winPtr;
    if (!win.metalLayer || win.delegate.closed) return;
    @autoreleasepool {
        id<CAMetalDrawable> drawable = [win.metalLayer nextDrawable];
        if (!drawable) return;
        render_drawlist(
            win, drawable.texture, drawable, 0, nil, uniforms,
            clips, clips_count, quads, quads_count,
            ribbons, ribbons_count, strokes, strokes_count,
            glyphs, glyphs_count, effects, effects_count,
            particles, particles_count, backdrops, backdrops_count,
            cmds, cmds_count);
    }
}

void* sofuu_macos_create_offscreen(
    const char* msl_source_utf8,
    const uint8_t* atlas_pixels,
    uint32_t atlas_w,
    uint32_t atlas_h
) {
    @autoreleasepool {
        SofuuMacWindow* r = create_renderer(msl_source_utf8, atlas_pixels, atlas_w, atlas_h);
        if (!r) return NULL;
        return (__bridge_retained void*)r;
    }
}

int sofuu_macos_render_offscreen(
    void* ptr,
    uint32_t width,
    uint32_t height,
    const SofuuFrameUniforms* uniforms,
    const void* clips,
    uint32_t clips_count,
    const void* quads,
    uint32_t quads_count,
    const void* ribbons,
    uint32_t ribbons_count,
    const void* strokes,
    uint32_t strokes_count,
    const void* glyphs,
    uint32_t glyphs_count,
    const void* effects,
    uint32_t effects_count,
    const void* particles,
    uint32_t particles_count,
    const void* backdrops,
    uint32_t backdrops_count,
    const SofuuDrawCmd* cmds,
    uint32_t cmds_count,
    uint8_t* out_rgba,
    size_t out_len
) {
    SofuuMacWindow* win = (__bridge SofuuMacWindow*)ptr;
    if (!win || !out_rgba) return -1;
    if (width == 0 || height == 0) return -2;
    size_t need = (size_t)width * (size_t)height * 4;
    if (out_len < need) return -3;
    @autoreleasepool {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                         width:width
                                        height:height
                                     mipmapped:NO];
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        id<MTLTexture> target = [win.device newTextureWithDescriptor:desc];
        if (!target) return -4;
        id<MTLBuffer> readback = [win.device newBufferWithLength:need
                                                        options:MTLResourceStorageModeShared];
        if (!readback) return -5;
        int rc = render_drawlist(
            win, target, nil, 1, readback, uniforms,
            clips, clips_count, quads, quads_count,
            ribbons, ribbons_count, strokes, strokes_count,
            glyphs, glyphs_count, effects, effects_count,
            particles, particles_count, backdrops, backdrops_count,
            cmds, cmds_count);
        if (rc != 0) return -6;
        memcpy(out_rgba, readback.contents, need);
        for (size_t i = 0; i + 3 < need; i += 4) {
            uint8_t b = out_rgba[i];
            out_rgba[i] = out_rgba[i + 2];
            out_rgba[i + 2] = b;
        }
        return 0;
    }
}

int sofuu_macos_window_focused(void* winPtr) {
    if (!winPtr) return 1;
    SofuuMacWindow* win = (__bridge SofuuMacWindow*)winPtr;
    return (win.window && win.window.isKeyWindow) ? 1 : 0;
}

int sofuu_macos_reduce_motion(void) {
    if (@available(macOS 10.12, *)) {
        return NSWorkspace.sharedWorkspace.accessibilityDisplayShouldReduceMotion ? 1 : 0;
    }
    return 0;
}

int sofuu_macos_poll_or_wait(void* winPtr, int wait_forever, double timeout_sec) {
    if (!winPtr) return 0;
    SofuuMacWindow* win = (__bridge SofuuMacWindow*)winPtr;
    if (win.delegate.closed) return 0;
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
        return win.delegate.closed ? 0 : 1;
    }
}

void sofuu_macos_strip_set_collapsed(int collapsed) {
    if (!g_stripController) return;
    [(id)g_stripController setCollapsed:collapsed ? YES : NO];
}
