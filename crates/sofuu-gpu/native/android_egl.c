// Sofuu Android window + EGL shim.
//
// Android does not let a plain shared library create an Activity/Window on its own. The host
// (Java/Kotlin SurfaceView, GameActivity, or sofuu-android Mode B) owns the ANativeWindow and
// forwards input through the additive `sofuu_android_*` helpers below. All EGL/GL objects stay
// on the thread that calls sofuu_android_create_window (normally the UI thread or a dedicated
// render thread).
//
// Idle behaviour is identical to desktop: Thread 0 blocks in poll() on an eventfd/pipe and
// consumes 0% CPU/GPU until the host pushes an event or a ScenePatch wakes it.

#define _GNU_SOURCE 1

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/eventfd.h>
#endif

#define EV_POINTER_MOVE 1
#define EV_POINTER_DOWN 2
#define EV_POINTER_UP 3
#define EV_SCROLL 4
#define EV_RESIZE 5
#define EV_CLOSE 6
#define EV_KEY_DOWN 7
#define EV_TEXT_CHAR 8

typedef void (*SofuuEventCb)(void* ctx, uint32_t kind, float x, float y, float a, float b, uint32_t u);

typedef void* EGLDisplay;
typedef void* EGLConfig;
typedef void* EGLSurface;
typedef void* EGLContext;
typedef int32_t EGLint;
typedef uint32_t EGLBoolean;
typedef uint32_t EGLenum;

#define EGL_PLATFORM_ANDROID_KHR 0x3141
#define EGL_OPENGL_ES_API 0x30A0
#define EGL_SURFACE_TYPE 0x3033
#define EGL_WINDOW_BIT 0x0004
#define EGL_RENDERABLE_TYPE 0x3040
#define EGL_OPENGL_ES3_BIT 0x0040
#define EGL_RED_SIZE 0x3024
#define EGL_GREEN_SIZE 0x3023
#define EGL_BLUE_SIZE 0x3022
#define EGL_ALPHA_SIZE 0x3021
#define EGL_NONE 0x3038
#define EGL_CONTEXT_CLIENT_VERSION 0x3098
#define EGL_NO_CONTEXT ((EGLContext)0)
#define EGL_NO_SURFACE ((EGLSurface)0)
#define EGL_NO_DISPLAY ((EGLDisplay)0)
#define EGL_BAD_SURFACE 0x300D
#define EGL_CONTEXT_LOST 0x300E
#define EGL_DEFAULT_DISPLAY ((void*)0)

typedef struct {
    void* lib;
    EGLDisplay (*get_platform_display)(EGLenum, void*, const intptr_t*);
    EGLDisplay (*get_display)(void*);
    EGLBoolean (*initialize)(EGLDisplay, EGLint*, EGLint*);
    EGLBoolean (*choose_config)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*);
    EGLSurface (*create_window_surface)(EGLDisplay, EGLConfig, void*, const EGLint*);
    EGLContext (*create_context)(EGLDisplay, EGLConfig, EGLContext, const EGLint*);
    EGLBoolean (*make_current)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
    EGLBoolean (*swap_buffers)(EGLDisplay, EGLSurface);
    EGLBoolean (*bind_api)(EGLenum);
    EGLint (*get_error)(void);
    EGLBoolean (*destroy_surface)(EGLDisplay, EGLSurface);
    EGLBoolean (*destroy_context)(EGLDisplay, EGLContext);
    EGLBoolean (*terminate)(EGLDisplay);
    EGLBoolean (*swap_interval)(EGLDisplay, EGLint);
    void* (*get_proc_address)(const char*);
} EglApi;

typedef struct SofuuAndroidWindow {
    EglApi egl;
    EGLDisplay display;
    EGLConfig config;
    EGLSurface surface;
    EGLContext context;
    void* native_window;
    void* gles_lib;
    SofuuEventCb cb;
    void* user_ctx;
    float logical_w;
    float logical_h;
    float scale;
    int closed;
    int surface_lost;
    int api_kind;
    int queue_fd;
    int queue_write_fd;
} SofuuAndroidWindow;

typedef struct {
    uint32_t kind;
    float x;
    float y;
    float a;
    float b;
    uint32_t u;
} SofuuAndroidEvent;

static SofuuAndroidWindow* g_window = NULL;
static char g_last_error[512];
static void* g_attached_window = NULL;
// Bumped on every host attach. Vulkan uses it to notice a SurfaceView recreation even when
// the allocator hands back the same ANativeWindow address.
static uint64_t g_attached_generation = 1;
static float g_attached_w = 1.0f;
static float g_attached_h = 1.0f;
static float g_attached_scale = 1.0f;
static int g_queue_fd = -1;
static int g_queue_write_fd = -1;

static void set_error(const char* msg) {
    snprintf(g_last_error, sizeof(g_last_error), "%s", msg ? msg : "");
}

static void* load_first(const char** names, int count) {
    for (int i = 0; i < count; ++i) {
        void* h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (h) return h;
    }
    return NULL;
}

static int ensure_queue(void) {
    if (g_queue_fd >= 0) return 1;
    // A pipe doubles as the event queue and the wake channel; every record is 24 bytes and
    // the render thread drains whole records in poll_or_wait().
    int fds[2];
    if (pipe(fds) != 0) return 0;
    for (int i = 0; i < 2; ++i) {
        fcntl(fds[i], F_SETFL, fcntl(fds[i], F_GETFL, 0) | O_NONBLOCK);
        fcntl(fds[i], F_SETFD, FD_CLOEXEC);
    }
    g_queue_fd = fds[0];
    g_queue_write_fd = fds[1];
    return 1;
}

static int egl_load(EglApi* e) {
    const char* names[] = {"libEGL.so", "libEGL.so.1"};
    e->lib = load_first(names, 2);
    if (!e->lib) return 0;
#define SYM(field, name) do { *(void**)(&e->field) = dlsym(e->lib, name); } while (0)
    SYM(get_platform_display, "eglGetPlatformDisplay");
    SYM(get_display, "eglGetDisplay");
    SYM(initialize, "eglInitialize");
    SYM(choose_config, "eglChooseConfig");
    SYM(create_window_surface, "eglCreateWindowSurface");
    SYM(create_context, "eglCreateContext");
    SYM(make_current, "eglMakeCurrent");
    SYM(swap_buffers, "eglSwapBuffers");
    SYM(bind_api, "eglBindAPI");
    SYM(get_error, "eglGetError");
    SYM(destroy_surface, "eglDestroySurface");
    SYM(destroy_context, "eglDestroyContext");
    SYM(terminate, "eglTerminate");
    SYM(swap_interval, "eglSwapInterval");
    SYM(get_proc_address, "eglGetProcAddress");
#undef SYM
    return e->initialize && e->choose_config && e->create_window_surface && e->create_context &&
           e->make_current && e->swap_buffers;
}

static void destroy_egl(SofuuAndroidWindow* w) {
    if (!w) return;
    if (w->egl.make_current && w->display)
        w->egl.make_current(w->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (w->egl.destroy_surface && w->surface) w->egl.destroy_surface(w->display, w->surface);
    if (w->egl.destroy_context && w->context) w->egl.destroy_context(w->display, w->context);
    if (w->egl.terminate && w->display) w->egl.terminate(w->display);
    w->surface = EGL_NO_SURFACE;
    w->context = EGL_NO_CONTEXT;
    w->display = EGL_NO_DISPLAY;
}

static int create_surface_and_context(SofuuAndroidWindow* w) {
    if (!w->native_window) return 0;
    if (!egl_load(&w->egl)) return 0;
    const char* gl_names[] = {"libGLESv2.so", "libGLESv2.so.2"};
    w->gles_lib = load_first(gl_names, 2);
    const intptr_t attribs[] = {0};
    w->display = w->egl.get_platform_display
        ? w->egl.get_platform_display(EGL_PLATFORM_ANDROID_KHR, EGL_DEFAULT_DISPLAY, attribs)
        : EGL_NO_DISPLAY;
    if (w->display == EGL_NO_DISPLAY && w->egl.get_display) {
        w->display = w->egl.get_display(EGL_DEFAULT_DISPLAY);
    }
    if (w->display == EGL_NO_DISPLAY) return 0;
    EGLint major = 0, minor = 0;
    if (!w->egl.initialize(w->display, &major, &minor)) return 0;
    if (w->egl.bind_api) w->egl.bind_api(EGL_OPENGL_ES_API);
    const EGLint cfg_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE,
    };
    EGLint count = 0;
    if (!w->egl.choose_config(w->display, cfg_attribs, &w->config, 1, &count) || count < 1) return 0;
    const EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    w->context = w->egl.create_context(w->display, w->config, EGL_NO_CONTEXT, ctx_attribs);
    if (w->context == EGL_NO_CONTEXT) {
        destroy_egl(w);
        return 0;
    }
    w->surface = w->egl.create_window_surface(w->display, w->config, w->native_window, NULL);
    if (w->surface == EGL_NO_SURFACE) {
        destroy_egl(w);
        return 0;
    }
    if (!w->egl.make_current(w->display, w->surface, w->surface, w->context)) {
        destroy_egl(w);
        return 0;
    }
    if (w->egl.swap_interval) w->egl.swap_interval(w->display, 1);
    w->api_kind = 0;
    w->surface_lost = 0;
    return 1;
}

void sofuu_android_attach_window(void* native_window, float width, float height, float scale) {
    ++g_attached_generation;
    g_attached_window = native_window;
    // A NULL window means the host backgrounded the app; keep the last logical size so the
    // scene does not reflow to 1x1 while there is nothing to draw into.
    if (native_window) {
        g_attached_w = width > 1.0f ? width : 1.0f;
        g_attached_h = height > 1.0f ? height : 1.0f;
        if (!(scale >= 1.0f)) scale = 1.0f; // also catches NaN
        if (scale > 4.0f) scale = 4.0f;
        g_attached_scale = scale;
    }
    if (g_window) {
        g_window->native_window = native_window;
        g_window->surface_lost = 1;
        if (native_window) {
            g_window->logical_w = g_attached_w;
            g_window->logical_h = g_attached_h;
            g_window->scale = g_attached_scale;
        }
    }
    if (native_window && g_queue_write_fd >= 0) {
        SofuuAndroidEvent ev = {EV_RESIZE, g_attached_w, g_attached_h, g_attached_scale, 0, 0};
        ssize_t ignored = write(g_queue_write_fd, &ev, sizeof(ev));
        (void)ignored;
    }
}

void sofuu_android_surface_changed(void* native_window, float width, float height, float scale) {
    sofuu_android_attach_window(native_window, width, height, scale);
}

void* sofuu_android_create_window(const char* title, float width, float height,
                                  float min_w, float min_h, SofuuEventCb cb, void* user_ctx) {
    (void)title; (void)min_w; (void)min_h;
    set_error("");
    if (!g_attached_window) {
        set_error("E_UI_NO_DISPLAY: Android requires the host to attach an ANativeWindow via sofuu_android_attach_window() before opening the UI");
        return NULL;
    }
    if (!ensure_queue()) {
        set_error("E_UI_NO_DISPLAY: failed to create the Android input wake channel");
        return NULL;
    }
    SofuuAndroidWindow* w = (SofuuAndroidWindow*)calloc(1, sizeof(SofuuAndroidWindow));
    if (!w) {
        set_error("E_UI_NO_MEMORY: failed to allocate Android window state");
        return NULL;
    }
    w->cb = cb;
    w->user_ctx = user_ctx;
    w->native_window = g_attached_window;
    // The attached ANativeWindow dictates the drawable size; `WindowConfig` defaults are not a
    // meaningful size on Android. Fall back to them only when the host attached no dimensions.
    w->logical_w = g_attached_w > 1.0f ? g_attached_w
                                       : (width > 1.0f ? width : 1.0f);
    w->logical_h = g_attached_h > 1.0f ? g_attached_h
                                       : (height > 1.0f ? height : 1.0f);
    w->scale = g_attached_scale;
    w->queue_fd = g_queue_fd;
    w->queue_write_fd = g_queue_write_fd;
    if (!create_surface_and_context(w)) {
        destroy_egl(w);
        free(w);
        set_error("E_UI_NO_DISPLAY: failed to create an Android EGL/GLES 3.0 surface");
        return NULL;
    }
    g_window = w;
    return w;
}

void sofuu_android_destroy_window(void* handle) {
    SofuuAndroidWindow* w = (SofuuAndroidWindow*)handle;
    if (!w) return;
    destroy_egl(w);
    if (g_window == w) g_window = NULL;
    free(w);
}

void sofuu_android_get_metrics(void* handle, float* out_w, float* out_h, float* out_scale) {
    SofuuAndroidWindow* w = (SofuuAndroidWindow*)handle;
    if (!w) return;
    if (out_w) *out_w = w->logical_w;
    if (out_h) *out_h = w->logical_h;
    if (out_scale) *out_scale = w->scale;
}

int sofuu_android_poll_or_wait(void* handle, int wait_forever, double timeout_sec) {
    SofuuAndroidWindow* w = (SofuuAndroidWindow*)handle;
    if (!w || w->closed) return 0;
    for (;;) {
        struct pollfd fd = {w->queue_fd, POLLIN, 0};
        int timeout_ms = wait_forever ? -1 : (int)ceil(timeout_sec * 1000.0);
        if (timeout_ms < 0 && !wait_forever) timeout_ms = 0;
        int r = poll(&fd, 1, timeout_ms);
        if (r < 0) {
            if (errno == EINTR) continue;
            w->closed = 1;
            break;
        }
        if (r == 0) break;
        SofuuAndroidEvent ev;
        int drained = 0;
        while (read(w->queue_fd, &ev, sizeof(ev)) == (ssize_t)sizeof(ev)) {
            drained = 1;
            if (w->cb) w->cb(w->user_ctx, ev.kind, ev.x, ev.y, ev.a, ev.b, ev.u);
        }
        if (drained) break;
    }
    return w->closed ? 0 : 1;
}

int sofuu_android_make_current(void* handle) {
    SofuuAndroidWindow* w = (SofuuAndroidWindow*)handle;
    if (!w || !w->egl.make_current) return 0;
    return w->egl.make_current(w->display, w->surface, w->surface, w->context) ? 1 : 0;
}

void sofuu_android_swap_buffers(void* handle) {
    SofuuAndroidWindow* w = (SofuuAndroidWindow*)handle;
    if (!w || !w->egl.swap_buffers) return;
    if (!w->egl.swap_buffers(w->display, w->surface)) {
        EGLint err = w->egl.get_error ? w->egl.get_error() : 0;
        if (err == EGL_BAD_SURFACE || err == EGL_CONTEXT_LOST) w->surface_lost = 1;
    }
}

void* sofuu_android_get_proc_address(const char* name) {
    if (!name) return NULL;
    SofuuAndroidWindow* w = g_window;
    if (w && w->egl.get_proc_address) {
        void* p = w->egl.get_proc_address(name);
        if (p) return p;
    }
    if (w && w->gles_lib) {
        void* p = dlsym(w->gles_lib, name);
        if (p) return p;
    }
    return NULL;
}

void sofuu_android_wake(void) {
    if (g_queue_write_fd < 0) ensure_queue();
    if (g_queue_write_fd < 0) return;
    SofuuAndroidEvent noop = {0, 0, 0, 0, 0, 0};
    ssize_t ignored = write(g_queue_write_fd, &noop, sizeof(noop));
    (void)ignored;
}

void sofuu_android_push_event(uint32_t kind, float x, float y, float a, float b, uint32_t u) {
    if (g_queue_write_fd < 0) ensure_queue();
    if (g_queue_write_fd < 0) return;
    SofuuAndroidEvent ev = {kind, x, y, a, b, u};
    ssize_t ignored = write(g_queue_write_fd, &ev, sizeof(ev));
    (void)ignored;
}

void sofuu_android_push_motion(int action, float x, float y) {
    switch (action & 0xFF) {
        case 0: sofuu_android_push_event(EV_POINTER_DOWN, x, y, 0, 0, 0); break;
        case 1: sofuu_android_push_event(EV_POINTER_UP, x, y, 0, 0, 0); break;
        case 2: sofuu_android_push_event(EV_POINTER_MOVE, x, y, 0, 0, 0); break;
        default: break;
    }
}

void sofuu_android_push_scroll(float x, float y, float dx, float dy) {
    sofuu_android_push_event(EV_SCROLL, x, y, dx, dy, 0);
}

void sofuu_android_push_key(int keycode, int modifiers) {
    uint32_t norm = 0;
    switch (keycode) {
        case 67: case 112: norm = 1; break; // DEL / FORWARD_DEL
        case 66: norm = 2; break;          // ENTER
        case 111: case 4: norm = 3; break; // ESCAPE / BACK
        case 21: norm = 4; break;
        case 22: norm = 5; break;
        case 19: norm = 6; break;
        case 20: norm = 7; break;
        case 122: norm = 8; break;
        case 123: norm = 9; break;
        case 61: norm = 10; break;
        default: norm = 1000u + (uint32_t)keycode; break;
    }
    // `modifiers` is the SOFUU_MOD_* bitmask (Shift=1, Ctrl=2, Alt=4, Meta=8); the host
    // translates Android KeyEvent metaState before calling this entry point.
    sofuu_android_push_event(EV_KEY_DOWN, 0, 0, (float)modifiers, 0, norm);
}

void sofuu_android_push_text(const char* utf8) {
    if (!utf8) return;
    const unsigned char* s = (const unsigned char*)utf8;
    while (*s) {
        uint32_t cp = 0;
        int len = 0;
        if (*s < 0x80) { cp = *s; len = 1; }
        else if ((*s & 0xE0) == 0xC0) { cp = *s & 0x1F; len = 2; }
        else if ((*s & 0xF0) == 0xE0) { cp = *s & 0x0F; len = 3; }
        else if ((*s & 0xF8) == 0xF0) { cp = *s & 0x07; len = 4; }
        else { ++s; continue; }
        int valid = 1;
        for (int i = 1; i < len; ++i) {
            if (s[i] == 0 || (s[i] & 0xC0) != 0x80) { valid = 0; break; }
            cp = (cp << 6) | (s[i] & 0x3F);
        }
        if (!valid) { ++s; continue; }
        if (cp >= 0x20 && cp != 0x7F) sofuu_android_push_event(EV_TEXT_CHAR, 0, 0, 0, 0, cp);
        s += len;
    }
}

void sofuu_android_push_close(void) {
    sofuu_android_push_event(EV_CLOSE, 0, 0, 0, 0, 0);
}

int sofuu_android_context_lost(void* handle) {
    SofuuAndroidWindow* w = (SofuuAndroidWindow*)handle;
    return w ? w->surface_lost : 0;
}

int sofuu_android_recover(void* handle) {
    SofuuAndroidWindow* w = (SofuuAndroidWindow*)handle;
    if (!w) return 0;
    destroy_egl(w);
    w->native_window = g_attached_window;
    if (!create_surface_and_context(w)) {
        w->surface_lost = 1;
        return 0;
    }
    return 1;
}

void sofuu_android_set_theme_dark(void* handle, int dark) {
    (void)handle; (void)dark;
}

int sofuu_android_api_kind(void* handle) {
    (void)handle;
    return 0;
}

int sofuu_android_surface_kind(void* handle) {
    (void)handle;
    return 3;
}

void* sofuu_android_display_handle(void* handle) {
    (void)handle;
    return NULL;
}

void* sofuu_android_window_handle(void* handle) {
    SofuuAndroidWindow* w = (SofuuAndroidWindow*)handle;
    return w ? w->native_window : g_attached_window;
}

uint64_t sofuu_android_window_generation(void* handle) {
    (void)handle;
    return g_attached_generation;
}

const char* sofuu_android_last_error(void) { return g_last_error; }
