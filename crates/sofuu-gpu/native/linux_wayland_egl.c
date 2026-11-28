// Sofuu Linux window + EGL shim.
//
// Wayland (xdg_wm_base) is the primary path; X11 is the fallback. Every system library
// (libwayland-client, libwayland-egl, libEGL, libGLESv2/libGL, libxkbcommon, libX11) is opened
// with dlopen() at run time, so libsofuu_gpu.so never hard-links a display server or a GPU
// driver. The Rust side owns all GL objects through dynamically resolved entry points.
//
// Threading: window creation, event pumping, EGL and every GL call happen on Thread 0. Thread 1+
// only calls sofuu_linux_wake(), which writes one byte to an eventfd/pipe.

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
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/eventfd.h>
#endif

// ── Event kinds (kept byte-for-byte in sync with macOS/Windows/Android) ───────────────────
#define EV_POINTER_MOVE 1
#define EV_POINTER_DOWN 2
#define EV_POINTER_UP 3
#define EV_SCROLL 4
#define EV_RESIZE 5
#define EV_CLOSE 6
#define EV_KEY_DOWN 7
#define EV_TEXT_CHAR 8

typedef void (*SofuuEventCb)(void* ctx, uint32_t kind, float x, float y, float a, float b, uint32_t u);

static char g_last_error[512];
static int g_wake_fd = -1;
static int g_wake_write_fd = -1;

static void set_error(const char* msg) {
    snprintf(g_last_error, sizeof(g_last_error), "%s", msg ? msg : "");
}

// ── Wake channel: eventfd on Linux, self-pipe elsewhere ────────────────────────────────────
static int g_wake_pipe[2] = {-1, -1};

static void init_wake(void) {
    if (g_wake_fd >= 0) return;
#ifdef __linux__
    int fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (fd >= 0) {
        g_wake_fd = fd;
        g_wake_write_fd = fd;
        return;
    }
#endif
    int fds[2];
#if defined(__linux__) || defined(__FreeBSD__)
    if (pipe2(fds, O_NONBLOCK | O_CLOEXEC) != 0) return;
#else
    if (pipe(fds) != 0) return;
    for (int i = 0; i < 2; ++i) {
        fcntl(fds[i], F_SETFL, fcntl(fds[i], F_GETFL, 0) | O_NONBLOCK);
        fcntl(fds[i], F_SETFD, FD_CLOEXEC);
    }
#endif
    g_wake_pipe[0] = fds[0];
    g_wake_pipe[1] = fds[1];
    g_wake_fd = fds[0];
    g_wake_write_fd = fds[1];
}

static void drain_wake(void) {
    if (g_wake_fd < 0) return;
    uint8_t buf[64];
    while (read(g_wake_fd, buf, sizeof(buf)) > 0) {
    }
}

void sofuu_linux_wake(void) {
    int fd = g_wake_write_fd;
    if (fd < 0) return;
#ifdef __linux__
    uint64_t one = 1;
    ssize_t ignored = write(fd, &one, sizeof(one));
#else
    uint8_t one = 1;
    ssize_t ignored = write(fd, &one, 1);
#endif
    (void)ignored;
}

// ── Wayland ABI (no wayland-client.h required) ─────────────────────────────────────────────
struct wl_display;
struct wl_proxy;
struct wl_registry;
struct wl_compositor;
struct wl_surface;
struct wl_seat;
struct wl_pointer;
struct wl_keyboard;
struct wl_output;
struct wl_egl_window;
struct xdg_wm_base;
struct xdg_surface;
struct xdg_toplevel;
struct xdg_positioner;
struct xdg_popup;

struct wl_interface;
struct wl_message {
    const char* name;
    const char* signature;
    const struct wl_interface** types;
};

struct wl_interface {
    const char* name;
    int version;
    int method_count;
    const struct wl_message* methods;
    int event_count;
    const struct wl_message* events;
};

// Only the xdg-shell interfaces need local definitions; core interfaces are resolved from
// libwayland-client's exported wl_*_interface symbols.
static const struct wl_message xdg_wm_base_requests[] = {
    {"destroy", "", NULL},
    {"create_positioner", "n", NULL},
    {"get_xdg_surface", "no", NULL},
    {"pong", "u", NULL},
};
static const struct wl_message xdg_wm_base_events[] = {
    {"ping", "u", NULL},
};
static const struct wl_interface xdg_wm_base_interface = {
    "xdg_wm_base", 4, 4, xdg_wm_base_requests, 1, xdg_wm_base_events,
};

static const struct wl_message xdg_surface_requests[] = {
    {"destroy", "", NULL},
    {"get_toplevel", "n", NULL},
    {"get_popup", "n?oo", NULL},
    {"set_window_geometry", "iiii", NULL},
    {"ack_configure", "u", NULL},
};
static const struct wl_message xdg_surface_events[] = {
    {"configure", "u", NULL},
};
static const struct wl_interface xdg_surface_interface = {
    "xdg_surface", 4, 5, xdg_surface_requests, 1, xdg_surface_events,
};

static const struct wl_message xdg_toplevel_requests[] = {
    {"destroy", "", NULL},
    {"set_parent", "?o", NULL},
    {"set_title", "s", NULL},
    {"set_app_id", "s", NULL},
    {"show_window_menu", "ouii", NULL},
    {"move", "ou", NULL},
    {"resize", "ouu", NULL},
    {"set_max_size", "ii", NULL},
    {"set_min_size", "ii", NULL},
    {"set_maximized", "", NULL},
    {"unset_maximized", "", NULL},
    {"set_fullscreen", "?o", NULL},
    {"unset_fullscreen", "", NULL},
    {"set_minimized", "", NULL},
};
static const struct wl_message xdg_toplevel_events[] = {
    {"configure", "iia", NULL},
    {"close", "", NULL},
    {"configure_bounds", "ii", NULL},
};
static const struct wl_interface xdg_toplevel_interface = {
    "xdg_toplevel", 4, 14, xdg_toplevel_requests, 3, xdg_toplevel_events,
};

// wp_fractional_scale_v1: opt-in protocol for non-integer HiDPI scales. Without it the client
// falls back to the integer wl_output.scale.
struct wp_fractional_scale_manager_v1;
struct wp_fractional_scale_v1;
static const struct wl_message fractional_scale_manager_requests[] = {
    {"destroy", "", NULL},
    {"get_fractional_scale", "no", NULL},
};
static const struct wl_interface wp_fractional_scale_manager_v1_interface = {
    "wp_fractional_scale_manager_v1", 1, 2, fractional_scale_manager_requests, 0, NULL,
};
static const struct wl_message fractional_scale_requests[] = {
    {"destroy", "", NULL},
};
static const struct wl_message fractional_scale_events[] = {
    {"preferred_scale", "u", NULL},
};
static const struct wl_interface wp_fractional_scale_v1_interface = {
    "wp_fractional_scale_v1", 1, 1, fractional_scale_requests, 1, fractional_scale_events,
};

// wp_viewporter: lets a fractional-scale client render a `logical * scale` buffer and still
// declare the surface's logical destination size, so the compositor downsamples/upsamples it
// instead of treating the larger buffer as a window resize.
struct wp_viewporter;
struct wp_viewport;
static const struct wl_message viewporter_requests[] = {
    {"destroy", "", NULL},
    {"get_viewport", "no", NULL},
};
static const struct wl_interface wp_viewporter_interface = {
    "wp_viewporter", 1, 2, viewporter_requests, 0, NULL,
};
static const struct wl_message viewport_requests[] = {
    {"destroy", "", NULL},
    {"set_source", "ffff", NULL},
    {"set_destination", "ii", NULL},
};
static const struct wl_interface wp_viewport_interface = {
    "wp_viewport", 1, 3, viewport_requests, 0, NULL,
};

struct wl_array;

// Listener signatures (opaque object pointers).
typedef void (*wl_registry_global_fn)(void*, struct wl_registry*, uint32_t, const char*, uint32_t);
typedef void (*wl_registry_remove_fn)(void*, struct wl_registry*, uint32_t);
struct wl_registry_listener {
    wl_registry_global_fn global;
    wl_registry_remove_fn global_remove;
};

typedef void (*xdg_wm_base_ping_fn)(void*, struct xdg_wm_base*, uint32_t);
struct xdg_wm_base_listener {
    xdg_wm_base_ping_fn ping;
};

typedef void (*xdg_surface_configure_fn)(void*, struct xdg_surface*, uint32_t);
struct xdg_surface_listener {
    xdg_surface_configure_fn configure;
};

typedef void (*xdg_toplevel_configure_fn)(void*, struct xdg_toplevel*, int32_t, int32_t, struct wl_array*);
typedef void (*xdg_toplevel_close_fn)(void*, struct xdg_toplevel*);
typedef void (*xdg_toplevel_bounds_fn)(void*, struct xdg_toplevel*, int32_t, int32_t);
struct xdg_toplevel_listener {
    xdg_toplevel_configure_fn configure;
    xdg_toplevel_close_fn close;
    xdg_toplevel_bounds_fn configure_bounds;
};

typedef void (*wl_seat_capabilities_fn)(void*, struct wl_seat*, uint32_t);
typedef void (*wl_seat_name_fn)(void*, struct wl_seat*, const char*);
struct wl_seat_listener {
    wl_seat_capabilities_fn capabilities;
    wl_seat_name_fn name;
};

typedef void (*wl_pointer_enter_fn)(void*, struct wl_pointer*, uint32_t, struct wl_surface*, int32_t, int32_t);
typedef void (*wl_pointer_leave_fn)(void*, struct wl_pointer*, uint32_t, struct wl_surface*);
typedef void (*wl_pointer_motion_fn)(void*, struct wl_pointer*, uint32_t, int32_t, int32_t);
typedef void (*wl_pointer_button_fn)(void*, struct wl_pointer*, uint32_t, uint32_t, uint32_t, uint32_t);
typedef void (*wl_pointer_axis_fn)(void*, struct wl_pointer*, uint32_t, uint32_t, int32_t);
typedef void (*wl_pointer_frame_fn)(void*, struct wl_pointer*);
typedef void (*wl_pointer_axis_source_fn)(void*, struct wl_pointer*, uint32_t);
typedef void (*wl_pointer_axis_stop_fn)(void*, struct wl_pointer*, uint32_t, uint32_t);
typedef void (*wl_pointer_axis_discrete_fn)(void*, struct wl_pointer*, uint32_t, int32_t);
struct wl_pointer_listener {
    wl_pointer_enter_fn enter;
    wl_pointer_leave_fn leave;
    wl_pointer_motion_fn motion;
    wl_pointer_button_fn button;
    wl_pointer_axis_fn axis;
    wl_pointer_frame_fn frame;
    wl_pointer_axis_source_fn axis_source;
    wl_pointer_axis_stop_fn axis_stop;
    wl_pointer_axis_discrete_fn axis_discrete;
};

typedef void (*wl_keyboard_keymap_fn)(void*, struct wl_keyboard*, uint32_t, int32_t, uint32_t);
typedef void (*wl_keyboard_enter_fn)(void*, struct wl_keyboard*, uint32_t, struct wl_surface*, struct wl_array*);
typedef void (*wl_keyboard_leave_fn)(void*, struct wl_keyboard*, uint32_t, struct wl_surface*);
typedef void (*wl_keyboard_key_fn)(void*, struct wl_keyboard*, uint32_t, uint32_t, uint32_t, uint32_t);
typedef void (*wl_keyboard_modifiers_fn)(void*, struct wl_keyboard*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
typedef void (*wl_keyboard_repeat_fn)(void*, struct wl_keyboard*, int32_t, int32_t);
struct wl_keyboard_listener {
    wl_keyboard_keymap_fn keymap;
    wl_keyboard_enter_fn enter;
    wl_keyboard_leave_fn leave;
    wl_keyboard_key_fn key;
    wl_keyboard_modifiers_fn modifiers;
    wl_keyboard_repeat_fn repeat_info;
};

typedef void (*wl_output_geometry_fn)(void*, struct wl_output*, int32_t, int32_t, int32_t, int32_t, int32_t, const char*, const char*, int32_t);
typedef void (*wl_output_mode_fn)(void*, struct wl_output*, uint32_t, int32_t, int32_t, int32_t);
typedef void (*wl_output_done_fn)(void*, struct wl_output*);
typedef void (*wl_output_scale_fn)(void*, struct wl_output*, int32_t);
struct wl_output_listener {
    wl_output_geometry_fn geometry;
    wl_output_mode_fn mode;
    wl_output_done_fn done;
    wl_output_scale_fn scale;
};

typedef void (*wp_fractional_scale_preferred_fn)(void*, struct wp_fractional_scale_v1*, uint32_t);
struct wp_fractional_scale_listener {
    wp_fractional_scale_preferred_fn preferred_scale;
};

typedef struct wl_display* (*fn_wl_display_connect)(const char*);
typedef void (*fn_wl_display_disconnect)(struct wl_display*);
typedef int (*fn_wl_display_dispatch)(struct wl_display*);
typedef int (*fn_wl_display_dispatch_pending)(struct wl_display*);
typedef int (*fn_wl_display_flush)(struct wl_display*);
typedef int (*fn_wl_display_roundtrip)(struct wl_display*);
typedef int (*fn_wl_display_get_fd)(struct wl_display*);
typedef struct wl_proxy* (*fn_wl_proxy_marshal_flags)(struct wl_proxy*, uint32_t, const struct wl_interface*, uint32_t, uint32_t, ...);
typedef void (*fn_wl_proxy_destroy)(struct wl_proxy*);
typedef int (*fn_wl_proxy_add_listener)(struct wl_proxy*, void (**)(void), void*);
typedef uint32_t (*fn_wl_proxy_get_version)(struct wl_proxy*);
typedef struct wl_proxy* (*fn_wl_registry_bind)(struct wl_registry*, uint32_t, const struct wl_interface*, uint32_t);
typedef struct wl_egl_window* (*fn_wl_egl_window_create)(struct wl_surface*, int, int);
typedef void (*fn_wl_egl_window_destroy)(struct wl_egl_window*);
typedef void (*fn_wl_egl_window_resize)(struct wl_egl_window*, int, int, int, int);

typedef struct {
    void* lib;
    void* egl_lib;
    struct wl_display* (*display_connect)(const char*);
    void (*display_disconnect)(struct wl_display*);
    int (*display_dispatch)(struct wl_display*);
    int (*display_dispatch_pending)(struct wl_display*);
    int (*display_flush)(struct wl_display*);
    int (*display_roundtrip)(struct wl_display*);
    int (*display_get_fd)(struct wl_display*);
    struct wl_proxy* (*proxy_marshal_flags)(struct wl_proxy*, uint32_t, const struct wl_interface*, uint32_t, uint32_t, ...);
    void (*proxy_destroy)(struct wl_proxy*);
    int (*proxy_add_listener)(struct wl_proxy*, void (**)(void), void*);
    uint32_t (*proxy_get_version)(struct wl_proxy*);
    struct wl_proxy* (*registry_bind)(struct wl_registry*, uint32_t, const struct wl_interface*, uint32_t);
    struct wl_egl_window* (*egl_window_create)(struct wl_surface*, int, int);
    void (*egl_window_destroy)(struct wl_egl_window*);
    void (*egl_window_resize)(struct wl_egl_window*, int, int, int, int);
    const struct wl_interface* registry_interface;
    const struct wl_interface* compositor_interface;
    const struct wl_interface* surface_interface;
    const struct wl_interface* seat_interface;
    const struct wl_interface* pointer_interface;
    const struct wl_interface* keyboard_interface;
    const struct wl_interface* output_interface;
} WlApi;

// ── xkbcommon ABI ──────────────────────────────────────────────────────────────────────────
struct xkb_context;
struct xkb_keymap;
struct xkb_state;
#define XKB_KEYMAP_FORMAT_TEXT_V1 1
#define XKB_STATE_MODS_EFFECTIVE 1

typedef struct {
    void* lib;
    struct xkb_context* (*context_new)(int);
    struct xkb_keymap* (*keymap_new_from_string)(struct xkb_context*, const char*, int, int);
    struct xkb_state* (*state_new)(struct xkb_keymap*);
    void (*state_unref)(struct xkb_state*);
    void (*keymap_unref)(struct xkb_keymap*);
    void (*context_unref)(struct xkb_context*);
    int (*state_update_mask)(struct xkb_state*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    uint32_t (*state_key_get_one_sym)(struct xkb_state*, uint32_t);
    int (*state_key_get_utf8)(struct xkb_state*, uint32_t, char*, size_t);
    uint32_t (*keysym_to_utf32)(uint32_t);
    int (*keymap_mod_get_index)(struct xkb_keymap*, const char*);
    int (*state_mod_index_is_active)(struct xkb_state*, uint32_t, int);
} XkbApi;

// ── EGL ABI ────────────────────────────────────────────────────────────────────────────────
typedef void* EGLDisplay;
typedef void* EGLConfig;
typedef void* EGLSurface;
typedef void* EGLContext;
typedef int32_t EGLint;
typedef uint32_t EGLBoolean;
typedef uint32_t EGLenum;

#define EGL_PLATFORM_WAYLAND_KHR 0x31D8
#define EGL_PLATFORM_X11_KHR 0x31D5
#define EGL_OPENGL_ES_API 0x30A0
#define EGL_OPENGL_API 0x30A2
#define EGL_SURFACE_TYPE 0x3033
#define EGL_WINDOW_BIT 0x0004
#define EGL_RENDERABLE_TYPE 0x3040
#define EGL_OPENGL_ES3_BIT 0x0040
#define EGL_OPENGL_BIT 0x0008
#define EGL_RED_SIZE 0x3024
#define EGL_GREEN_SIZE 0x3023
#define EGL_BLUE_SIZE 0x3022
#define EGL_ALPHA_SIZE 0x3021
#define EGL_NONE 0x3038
#define EGL_CONTEXT_CLIENT_VERSION 0x3098
#define EGL_CONTEXT_MAJOR_VERSION 0x3098
#define EGL_CONTEXT_MINOR_VERSION 0x30FB
#define EGL_CONTEXT_OPENGL_PROFILE_MASK 0x30FD
#define EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT 0x00000001
#define EGL_NO_CONTEXT ((EGLContext)0)
#define EGL_NO_SURFACE ((EGLSurface)0)
#define EGL_NO_DISPLAY ((EGLDisplay)0)
#define EGL_SUCCESS 0x3000
#define EGL_BAD_SURFACE 0x300D
#define EGL_CONTEXT_LOST 0x300E
#define EGL_BAD_DISPLAY 0x3008
#define EGL_NOT_INITIALIZED 0x3001

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

// ── X11 ABI (raw offsets, no Xlib headers) ────────────────────────────────────────────────
typedef struct _XDisplay Display;
typedef unsigned long Window;
typedef unsigned long Atom;
typedef unsigned long KeySym;
typedef struct _XIM* XIM;
typedef struct _XIC* XIC;

#define X11_SIZE_LONG 24
typedef struct {
    long pad[X11_SIZE_LONG];
} SofuuXEvent;

typedef struct {
    int type;
    long serial;
    int send_event;
    Display* display;
    Window window;
    Atom message_type;
    int format;
    union {
        char b[20];
        short s[10];
        long l[5];
    } data;
} SofuuXClientMessageEvent;

#define XEV_CLIENT_MESSAGE 33
#define XEV_CONFIGURE_NOTIFY 22
#define XEV_DESTROY_NOTIFY 17
#define XEV_EXPOSE 12
#define XEV_MOTION_NOTIFY 6
#define XEV_BUTTON_PRESS 4
#define XEV_BUTTON_RELEASE 5
#define XEV_KEY_PRESS 2
#define XEV_FOCUS_IN 9
#define X11_STRUCTURE_NOTIFY_MASK (1L << 17)
#define X11_KEY_PRESS_MASK (1L << 0)
#define X11_KEY_RELEASE_MASK (1L << 1)
#define X11_BUTTON_PRESS_MASK (1L << 2)
#define X11_BUTTON_RELEASE_MASK (1L << 3)
#define X11_POINTER_MOTION_MASK (1L << 6)
#define X11_EXPOSURE_MASK (1L << 15)
#define X11_FOCUS_CHANGE_MASK (1L << 21)
#define X11_FALSE 0
#define X11_TRUE 1
#define X11_PROP_MODE_REPLACE 0

typedef struct {
    void* lib;
    Display* (*open_display)(const char*);
    int (*default_screen)(Display*);
    Window (*root_window)(Display*, int);
    Window (*create_simple_window)(Display*, Window, int, int, unsigned, unsigned, unsigned, unsigned long, unsigned long);
    int (*select_input)(Display*, Window, long);
    int (*map_window)(Display*, Window);
    int (*store_name)(Display*, Window, const char*);
    Atom (*intern_atom)(Display*, const char*, int);
    int (*set_wm_protocols)(Display*, Window, Atom*, int);
    int (*change_property)(Display*, Window, Atom, Atom, int, int, const unsigned char*, int);
    int (*flush)(Display*);
    int (*pending)(Display*);
    int (*next_event)(Display*, SofuuXEvent*);
    int (*filter_event)(void*, unsigned long);
    int (*connection_number)(Display*);
    int (*send_event)(Display*, Window, int, long, void*);
    int (*move_resize_window)(Display*, Window, int, int, unsigned, unsigned);
    int (*display_width)(Display*, int);
    int (*display_height)(Display*, int);
    int (*destroy_window)(Display*, Window);
    int (*close_display)(Display*);
    int (*lookup_string)(void*, char*, int, KeySym*, void*);
    int (*utf8_lookup_string)(void*, void*, char*, int, KeySym*, int*);
    XIM (*open_im)(Display*, void*, void*, void*);
    XIC (*create_ic)(XIM, ...);
    void (*set_ic_focus)(XIC);
    void (*destroy_ic)(XIC);
    void (*close_im)(XIM);
    char* (*get_default)(Display*, const char*, const char*);
} X11Api;

// ── Window state ───────────────────────────────────────────────────────────────────────────
typedef struct SofuuLinuxWindow {
    int kind; // 0 = Wayland, 1 = X11
    int closed;
    int context_lost;
    int api_kind; // 0 = GLES, 1 = desktop GL
    float logical_w;
    float logical_h;
    float scale;
    SofuuEventCb cb;
    void* user_ctx;

    EglApi egl;
    EGLDisplay egl_display;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    void* gles_lib;
    void* gl_lib;
    void* gles2_lib;

    // Wayland
    WlApi wl;
    struct wl_display* wl_display;
    struct wl_registry* wl_registry;
    struct wl_compositor* wl_compositor;
    struct xdg_wm_base* wl_wm_base;
    struct wl_seat* wl_seat;
    struct wl_pointer* wl_pointer;
    struct wl_keyboard* wl_keyboard;
    struct wl_output* wl_output;
    struct wp_fractional_scale_manager_v1* wl_fractional_manager;
    struct wp_fractional_scale_v1* wl_fractional;
    struct wp_viewporter* wl_viewporter;
    struct wp_viewport* wl_viewport;
    struct wl_surface* wl_surface;
    struct xdg_surface* wl_xdg_surface;
    struct xdg_toplevel* wl_toplevel;
    struct wl_egl_window* wl_egl_window;
    int output_scale;
    int configured_w;
    int configured_h;
    int pointer_focus;
    float pointer_x;
    float pointer_y;
    int wl_surface_created;
    XkbApi xkb;
    struct xkb_context* xkb_ctx;
    struct xkb_keymap* xkb_keymap;
    struct xkb_state* xkb_state;

    // X11
    X11Api x11;
    Display* x11_display;
    Window x11_root;
    int x11_screen;
    Window x11_window;
    Atom x11_wm_delete;
    XIM x11_im;
    XIC x11_ic;
    int x11_has_xim;
} SofuuLinuxWindow;

static SofuuLinuxWindow* g_window = NULL;

static void emit_event(uint32_t kind, float x, float y, float a, float b, uint32_t u) {
    if (g_window && g_window->cb) g_window->cb(g_window->user_ctx, kind, x, y, a, b, u);
}

// ── EGL ────────────────────────────────────────────────────────────────────────────────────
static void* load_first(const char** names, int count) {
    for (int i = 0; i < count; ++i) {
        void* h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (h) return h;
    }
    return NULL;
}

static void load_gl_libs(SofuuLinuxWindow* w) {
    if (!w->gles_lib) {
        const char* names[] = {"libGLESv2.so.2", "libGLESv2.so"};
        w->gles_lib = load_first(names, 2);
    }
    if (!w->gl_lib) {
        const char* names[] = {"libGL.so.1", "libGL.so", "libOpenGL.so.0"};
        w->gl_lib = load_first(names, 3);
    }
}

static int egl_load(EglApi* e) {
    const char* names[] = {"libEGL.so.1", "libEGL.so"};
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
    return e->initialize && e->choose_config && e->create_window_surface &&
           e->create_context && e->make_current && e->swap_buffers;
}

static int egl_choose_and_create(SofuuLinuxWindow* w, void* native_window) {
    // Try GLES 3.0 then desktop GL 3.3, each with and without an alpha channel. Some
    // compositor/driver combinations expose no 8-bit-alpha window config; rendering opaque is
    // better than failing to open the UI.
    for (int attempt = 0; attempt < 4; ++attempt) {
        int use_es = (attempt % 2) == 0;
        int with_alpha = attempt < 2;
        EGLenum api = use_es ? EGL_OPENGL_ES_API : EGL_OPENGL_API;
        if (!w->egl.bind_api || !w->egl.bind_api(api)) continue;
        EGLint attribs[16];
        int n = 0;
        attribs[n++] = EGL_SURFACE_TYPE;
        attribs[n++] = EGL_WINDOW_BIT;
        attribs[n++] = EGL_RED_SIZE;
        attribs[n++] = 8;
        attribs[n++] = EGL_GREEN_SIZE;
        attribs[n++] = 8;
        attribs[n++] = EGL_BLUE_SIZE;
        attribs[n++] = 8;
        if (with_alpha) {
            attribs[n++] = EGL_ALPHA_SIZE;
            attribs[n++] = 8;
        }
        attribs[n++] = EGL_RENDERABLE_TYPE;
        attribs[n++] = use_es ? EGL_OPENGL_ES3_BIT : EGL_OPENGL_BIT;
        attribs[n++] = EGL_NONE;
        EGLConfig config = NULL;
        EGLint count = 0;
        if (!w->egl.choose_config(w->egl_display, attribs, &config, 1, &count) || count < 1) continue;

        EGLint ctx_attribs[9];
        int cn = 0;
        if (use_es) {
            ctx_attribs[cn++] = EGL_CONTEXT_CLIENT_VERSION;
            ctx_attribs[cn++] = 3;
        } else {
            ctx_attribs[cn++] = EGL_CONTEXT_MAJOR_VERSION;
            ctx_attribs[cn++] = 3;
            ctx_attribs[cn++] = EGL_CONTEXT_MINOR_VERSION;
            ctx_attribs[cn++] = 3;
            ctx_attribs[cn++] = EGL_CONTEXT_OPENGL_PROFILE_MASK;
            ctx_attribs[cn++] = EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT;
        }
        ctx_attribs[cn++] = EGL_NONE;
        EGLContext ctx = w->egl.create_context(w->egl_display, config, EGL_NO_CONTEXT, ctx_attribs);
        if (ctx == EGL_NO_CONTEXT) continue;

        EGLSurface surface = w->egl.create_window_surface(w->egl_display, config, native_window, NULL);
        if (surface == EGL_NO_SURFACE) {
            w->egl.destroy_context(w->egl_display, ctx);
            continue;
        }
        if (!w->egl.make_current(w->egl_display, surface, surface, ctx)) {
            w->egl.destroy_surface(w->egl_display, surface);
            w->egl.destroy_context(w->egl_display, ctx);
            continue;
        }
        w->egl_config = config;
        w->egl_context = ctx;
        w->egl_surface = surface;
        w->api_kind = use_es ? 0 : 1;
        if (w->egl.swap_interval) w->egl.swap_interval(w->egl_display, 1);
        return 1;
    }
    return 0;
}

// ── xkbcommon ──────────────────────────────────────────────────────────────────────────────
static void xkb_load(SofuuLinuxWindow* w) {
    const char* names[] = {"libxkbcommon.so.0", "libxkbcommon.so"};
    w->xkb.lib = load_first(names, 2);
    if (!w->xkb.lib) return;
#define SYM(field, name) do { *(void**)(&w->xkb.field) = dlsym(w->xkb.lib, name); } while (0)
    SYM(context_new, "xkb_context_new");
    SYM(keymap_new_from_string, "xkb_keymap_new_from_string");
    SYM(state_new, "xkb_state_new");
    SYM(state_unref, "xkb_state_unref");
    SYM(keymap_unref, "xkb_keymap_unref");
    SYM(context_unref, "xkb_context_unref");
    SYM(state_update_mask, "xkb_state_update_mask");
    SYM(state_key_get_one_sym, "xkb_state_key_get_one_sym");
    SYM(state_key_get_utf8, "xkb_state_key_get_utf8");
    SYM(keysym_to_utf32, "xkb_keysym_to_utf32");
    SYM(keymap_mod_get_index, "xkb_keymap_mod_get_index");
    SYM(state_mod_index_is_active, "xkb_state_mod_index_is_active");
#undef SYM
    if (!w->xkb.context_new) return;
    w->xkb_ctx = w->xkb.context_new(0);
}

static uint32_t key_enum_from_keysym(uint32_t sym) {
    switch (sym) {
        case 0xff08: return 1;  // BackSpace
        case 0xff0d: case 0xff8d: return 2;  // Return / KP_Enter
        case 0xff1b: return 3;  // Escape
        case 0xff51: return 4;  // Left
        case 0xff53: return 5;  // Right
        case 0xff52: return 6;  // Up
        case 0xff54: return 7;  // Down
        case 0xff50: return 8;  // Home
        case 0xff57: return 9;  // End
        case 0xff09: return 10; // Tab
        case 0xffff: return 11; // Delete
        default: return 0;
    }
}

static int xkb_mod_active(SofuuLinuxWindow* w, const char* name) {
    if (!w->xkb_state || !w->xkb.keymap_mod_get_index || !w->xkb.state_mod_index_is_active) return 0;
    int idx = w->xkb.keymap_mod_get_index(w->xkb_keymap, name);
    if (idx < 0) return 0;
    return w->xkb.state_mod_index_is_active(w->xkb_state, (uint32_t)idx, XKB_STATE_MODS_EFFECTIVE);
}

// SOFUU_MOD_* bits shared with include/sofuu_embed.h: Shift=1, Ctrl=2, Alt=4, Meta=8.
static uint32_t wayland_modifiers(SofuuLinuxWindow* w) {
    uint32_t m = 0;
    if (xkb_mod_active(w, "Shift")) m |= 1u;
    if (xkb_mod_active(w, "Control")) m |= 2u;
    if (xkb_mod_active(w, "Alt") || xkb_mod_active(w, "Mod1")) m |= 4u;
    if (xkb_mod_active(w, "Super") || xkb_mod_active(w, "Logo")) m |= 8u;
    return m;
}

static void emit_utf8_codepoints(const char* s) {
    while (s && *s) {
        unsigned char c = (unsigned char)*s;
        uint32_t cp = 0;
        int len = 0;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { ++s; continue; }
        int valid = 1;
        for (int i = 1; i < len; ++i) {
            if (s[i] == 0 || ((unsigned char)s[i] & 0xC0) != 0x80) { valid = 0; break; }
            cp = (cp << 6) | ((unsigned char)s[i] & 0x3F);
        }
        if (!valid) { ++s; continue; }
        if (cp >= 0x20 && cp != 0x7F && cp != 0x80) emit_event(EV_TEXT_CHAR, 0, 0, 0, 0, cp);
        s += len;
    }
}

// ── Wayland callbacks ──────────────────────────────────────────────────────────────────────
static void wl_registry_global(void* data, struct wl_registry* registry, uint32_t name, const char* iface, uint32_t version) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    if (strcmp(iface, "wl_compositor") == 0 && !w->wl_compositor) {
        uint32_t v = version < 4 ? version : 4;
        w->wl_compositor = (struct wl_compositor*)w->wl.registry_bind(registry, name, w->wl.compositor_interface, v);
    } else if (strcmp(iface, "xdg_wm_base") == 0 && !w->wl_wm_base) {
        uint32_t v = version < 4 ? version : 4;
        w->wl_wm_base = (struct xdg_wm_base*)w->wl.registry_bind(registry, name, &xdg_wm_base_interface, v);
    } else if (strcmp(iface, "wl_seat") == 0 && !w->wl_seat) {
        uint32_t v = version < 5 ? version : 5;
        w->wl_seat = (struct wl_seat*)w->wl.registry_bind(registry, name, w->wl.seat_interface, v);
    } else if (strcmp(iface, "wl_output") == 0 && !w->wl_output) {
        uint32_t v = version < 3 ? version : 3;
        w->wl_output = (struct wl_output*)w->wl.registry_bind(registry, name, w->wl.output_interface, v);
    } else if (strcmp(iface, "wp_viewporter") == 0 && !w->wl_viewporter) {
        uint32_t v = version < 1 ? version : 1;
        w->wl_viewporter = (struct wp_viewporter*)w->wl.registry_bind(
            registry, name, &wp_viewporter_interface, v);
    } else if (strcmp(iface, "wp_fractional_scale_manager_v1") == 0 && !w->wl_fractional_manager) {
        uint32_t v = version < 1 ? version : 1;
        w->wl_fractional_manager = (struct wp_fractional_scale_manager_v1*)w->wl.registry_bind(
            registry, name, &wp_fractional_scale_manager_v1_interface, v);
    }
}

static void wl_registry_remove(void* data, struct wl_registry* registry, uint32_t name) {
    (void)data; (void)registry; (void)name;
}

static const struct wl_registry_listener registry_listener = {
    wl_registry_global, wl_registry_remove,
};

static void xdg_wm_base_ping(void* data, struct xdg_wm_base* base, uint32_t serial) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    w->wl.proxy_marshal_flags((struct wl_proxy*)base, 3, NULL, w->wl.proxy_get_version((struct wl_proxy*)base), 0, serial);
}

static const struct xdg_wm_base_listener wm_base_listener = {xdg_wm_base_ping};

// Wayland buffers cover `logical * scale` device pixels. With fractional-scale-v1 the protocol
// requires buffer_scale=1, so the viewport destination declares the logical size and the
// compositor performs the fractional downsample. Without it, integer scales use
// set_buffer_scale for 1:1 sampling.
static int wayland_scale_is_integer(float scale) {
    // Exact comparisons: 1.999 must use the viewport path, otherwise the rounded
    // buffer_scale would not match the `logical * 1.999` buffer size.
    return fabsf(scale - (float)lroundf(scale)) < 1e-6f;
}

static void wayland_set_buffer_scale(SofuuLinuxWindow* w, int32_t value) {
    if (!w->wl_surface || !w->wl.proxy_marshal_flags || !w->wl.proxy_get_version) return;
    uint32_t version = w->wl.proxy_get_version((struct wl_proxy*)w->wl_surface);
    if (version < 3) return;
    if (value < 1) value = 1;
    w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_surface, 8, NULL, version, 0, value);
}

// Pass -1,-1 to unset the viewport destination; set_destination is double-buffered.
static void wayland_set_destination(SofuuLinuxWindow* w, int32_t width, int32_t height) {
    if (!w->wl_viewport || !w->wl.proxy_marshal_flags || !w->wl.proxy_get_version) return;
    uint32_t version = w->wl.proxy_get_version((struct wl_proxy*)w->wl_viewport);
    w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_viewport, 2, NULL, version, 0,
        width, height);
}

static void wayland_apply_scale(SofuuLinuxWindow* w) {
    if (!w->wl_viewport) {
        wayland_set_buffer_scale(w, (int32_t)lroundf(w->scale));
        return;
    }
    // A surface with a wp_fractional_scale_v1 object must keep buffer_scale=1; the viewport
    // destination carries the logical size for both integer and fractional preferred scales.
    if (w->wl_fractional || !wayland_scale_is_integer(w->scale)) {
        int32_t lw = (int32_t)lroundf(w->logical_w);
        int32_t lh = (int32_t)lroundf(w->logical_h);
        if (lw < 1) lw = 1;
        if (lh < 1) lh = 1;
        wayland_set_buffer_scale(w, 1);
        wayland_set_destination(w, lw, lh);
    } else {
        wayland_set_buffer_scale(w, (int32_t)lroundf(w->scale));
        wayland_set_destination(w, -1, -1);
    }
}

static float wayland_env_scale_override(float fallback) {
    const char* scale_env = getenv("SOFUU_UI_SCALE");
    if (scale_env && *scale_env) {
        float s = (float)atof(scale_env);
        if (!(s >= 1.0f)) s = 1.0f; // also catches NaN
        if (s > 4.0f) s = 4.0f;
        return s;
    }
    return fallback;
}

static void apply_configured_size(SofuuLinuxWindow* w) {
    if (w->configured_w <= 0 || w->configured_h <= 0) return;
    float lw = (float)w->configured_w;
    float lh = (float)w->configured_h;
    if (fabsf(lw - w->logical_w) < 0.5f && fabsf(lh - w->logical_h) < 0.5f) return;
    w->logical_w = lw;
    w->logical_h = lh;
    if (w->wl_egl_window && w->wl.egl_window_resize) {
        w->wl.egl_window_resize(w->wl_egl_window, (int)lroundf(lw * w->scale), (int)lroundf(lh * w->scale), 0, 0);
        wayland_apply_scale(w);
    }
    emit_event(EV_RESIZE, lw, lh, w->scale, 0, 0);
}

static void xdg_surface_configure(void* data, struct xdg_surface* surface, uint32_t serial) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    w->wl.proxy_marshal_flags((struct wl_proxy*)surface, 4, NULL, w->wl.proxy_get_version((struct wl_proxy*)surface), 0, serial);
    apply_configured_size(w);
}

static const struct xdg_surface_listener xdg_surface_listener = {xdg_surface_configure};

static void xdg_toplevel_configure(void* data, struct xdg_toplevel* toplevel, int32_t width, int32_t height, struct wl_array* states) {
    (void)toplevel; (void)states;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    if (width > 0) w->configured_w = width;
    if (height > 0) w->configured_h = height;
}

static void xdg_toplevel_close(void* data, struct xdg_toplevel* toplevel) {
    (void)toplevel;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    w->closed = 1;
    emit_event(EV_CLOSE, 0, 0, 0, 0, 0);
}

static void xdg_toplevel_bounds(void* data, struct xdg_toplevel* toplevel, int32_t width, int32_t height) {
    (void)data; (void)toplevel; (void)width; (void)height;
}

static const struct xdg_toplevel_listener toplevel_listener = {
    xdg_toplevel_configure, xdg_toplevel_close, xdg_toplevel_bounds,
};

static const struct wl_pointer_listener pointer_listener;
static const struct wl_keyboard_listener keyboard_listener;

static void wl_seat_capabilities(void* data, struct wl_seat* seat, uint32_t caps) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    if ((caps & 1) && !w->wl_pointer) {
        w->wl_pointer = (struct wl_pointer*)w->wl.proxy_marshal_flags((struct wl_proxy*)seat, 0,
            w->wl.pointer_interface, w->wl.proxy_get_version((struct wl_proxy*)seat), 0, NULL);
        if (w->wl_pointer) {
            w->wl.proxy_add_listener((struct wl_proxy*)w->wl_pointer, (void (**)(void))&pointer_listener, w);
        }
    }
    if ((caps & 2) && !w->wl_keyboard) {
        w->wl_keyboard = (struct wl_keyboard*)w->wl.proxy_marshal_flags((struct wl_proxy*)seat, 1,
            w->wl.keyboard_interface, w->wl.proxy_get_version((struct wl_proxy*)seat), 0, NULL);
        if (w->wl_keyboard) {
            w->wl.proxy_add_listener((struct wl_proxy*)w->wl_keyboard, (void (**)(void))&keyboard_listener, w);
        }
    }
}

static void wl_seat_name(void* data, struct wl_seat* seat, const char* name) {
    (void)data; (void)seat; (void)name;
}

static const struct wl_seat_listener seat_listener = {wl_seat_capabilities, wl_seat_name};

static void wl_pointer_enter(void* data, struct wl_pointer* pointer, uint32_t serial, struct wl_surface* surface, int32_t sx, int32_t sy) {
    (void)pointer; (void)serial; (void)surface;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    w->pointer_focus = 1;
    w->pointer_x = (float)sx / 256.0f;
    w->pointer_y = (float)sy / 256.0f;
    emit_event(EV_POINTER_MOVE, w->pointer_x, w->pointer_y, 0, 0, 0);
}

static void wl_pointer_leave(void* data, struct wl_pointer* pointer, uint32_t serial, struct wl_surface* surface) {
    (void)pointer; (void)serial; (void)surface;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    w->pointer_focus = 0;
}

static void wl_pointer_motion(void* data, struct wl_pointer* pointer, uint32_t time, int32_t sx, int32_t sy) {
    (void)pointer; (void)time;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    w->pointer_x = (float)sx / 256.0f;
    w->pointer_y = (float)sy / 256.0f;
    emit_event(EV_POINTER_MOVE, w->pointer_x, w->pointer_y, 0, 0, 0);
}

static void wl_pointer_button(void* data, struct wl_pointer* pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state) {
    (void)pointer; (void)serial; (void)time;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    uint32_t b = button == 0x110 ? 0 : button == 0x111 ? 1 : button == 0x112 ? 2 : 3;
    emit_event(state ? EV_POINTER_DOWN : EV_POINTER_UP, w->pointer_x, w->pointer_y, 0, 0, b);
}

static void wl_pointer_axis(void* data, struct wl_pointer* pointer, uint32_t time, uint32_t axis, int32_t value) {
    (void)pointer; (void)time;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    float v = (float)value / 256.0f;
    if (axis == 0) emit_event(EV_SCROLL, w->pointer_x, w->pointer_y, 0, -v, 0);
    else emit_event(EV_SCROLL, w->pointer_x, w->pointer_y, -v, 0, 0);
}

static void wl_pointer_frame(void* data, struct wl_pointer* pointer) { (void)data; (void)pointer; }
static void wl_pointer_axis_source(void* data, struct wl_pointer* pointer, uint32_t source) { (void)data; (void)pointer; (void)source; }
static void wl_pointer_axis_stop(void* data, struct wl_pointer* pointer, uint32_t time, uint32_t axis) { (void)data; (void)pointer; (void)time; (void)axis; }
static void wl_pointer_axis_discrete(void* data, struct wl_pointer* pointer, uint32_t axis, int32_t discrete) { (void)data; (void)pointer; (void)axis; (void)discrete; }

static const struct wl_pointer_listener pointer_listener = {
    wl_pointer_enter, wl_pointer_leave, wl_pointer_motion, wl_pointer_button, wl_pointer_axis,
    wl_pointer_frame, wl_pointer_axis_source, wl_pointer_axis_stop, wl_pointer_axis_discrete,
};

static void wl_keyboard_keymap(void* data, struct wl_keyboard* keyboard, uint32_t format, int32_t fd, uint32_t size) {
    (void)keyboard;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    if (format != 1 || size == 0 || !w->xkb.keymap_new_from_string) {
        close(fd);
        return;
    }
    char* map = (char*)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return;
    if (map[size - 1] != '\0') {
        // Ensure NUL termination without modifying beyond the mapping.
        char* copy = (char*)malloc((size_t)size + 1);
        if (copy) {
            memcpy(copy, map, size);
            copy[size] = '\0';
            w->xkb_keymap = w->xkb.keymap_new_from_string(w->xkb_ctx, copy, XKB_KEYMAP_FORMAT_TEXT_V1, 0);
            free(copy);
        }
    } else {
        w->xkb_keymap = w->xkb.keymap_new_from_string(w->xkb_ctx, map, XKB_KEYMAP_FORMAT_TEXT_V1, 0);
    }
    munmap(map, size);
    if (w->xkb_keymap && w->xkb.state_new) w->xkb_state = w->xkb.state_new(w->xkb_keymap);
}

static void wl_keyboard_enter(void* data, struct wl_keyboard* keyboard, uint32_t serial, struct wl_surface* surface, struct wl_array* keys) {
    (void)data; (void)keyboard; (void)serial; (void)surface; (void)keys;
}
static void wl_keyboard_leave(void* data, struct wl_keyboard* keyboard, uint32_t serial, struct wl_surface* surface) {
    (void)data; (void)keyboard; (void)serial; (void)surface;
}

static void wl_keyboard_key(void* data, struct wl_keyboard* keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state) {
    (void)keyboard; (void)serial; (void)time;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    if (state != 1) return;
    uint32_t sym = w->xkb_state && w->xkb.state_key_get_one_sym ? w->xkb.state_key_get_one_sym(w->xkb_state, key) : 0;
    uint32_t norm = key_enum_from_keysym(sym);
    emit_event(EV_KEY_DOWN, 0, 0, (float)wayland_modifiers(w), 0,
        norm ? norm : (1000u + (key > 8 ? key - 8 : key)));
    if (w->xkb_state && w->xkb.state_key_get_utf8 && sym != 0) {
        if (xkb_mod_active(w, "Control") || xkb_mod_active(w, "Alt") || xkb_mod_active(w, "Super")) return;
        char buf[32];
        int n = w->xkb.state_key_get_utf8(w->xkb_state, key, buf, sizeof(buf));
        if (n > 0) emit_utf8_codepoints(buf);
    } else if (sym) {
        uint32_t cp = w->xkb.keysym_to_utf32 ? w->xkb.keysym_to_utf32(sym) : 0;
        if (cp >= 0x20 && cp != 0x7f) emit_event(EV_TEXT_CHAR, 0, 0, 0, 0, cp);
    }
}

static void wl_keyboard_modifiers(void* data, struct wl_keyboard* keyboard, uint32_t serial, uint32_t depressed, uint32_t latched, uint32_t locked, uint32_t group) {
    (void)keyboard; (void)serial;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    if (w->xkb_state && w->xkb.state_update_mask) {
        w->xkb.state_update_mask(w->xkb_state, depressed, latched, locked, 0, 0, group);
    }
}

static void wl_keyboard_repeat(void* data, struct wl_keyboard* keyboard, int32_t rate, int32_t delay) {
    (void)data; (void)keyboard; (void)rate; (void)delay;
}

static const struct wl_keyboard_listener keyboard_listener = {
    wl_keyboard_keymap, wl_keyboard_enter, wl_keyboard_leave, wl_keyboard_key,
    wl_keyboard_modifiers, wl_keyboard_repeat,
};

static void wl_output_geometry(void* data, struct wl_output* output, int32_t x, int32_t y, int32_t pw, int32_t ph, int32_t subpixel, const char* make, const char* model, int32_t transform) {
    (void)data; (void)output; (void)x; (void)y; (void)pw; (void)ph; (void)subpixel; (void)make; (void)model; (void)transform;
}
static void wl_output_mode(void* data, struct wl_output* output, uint32_t flags, int32_t width, int32_t height, int32_t refresh) {
    (void)data; (void)output; (void)flags; (void)width; (void)height; (void)refresh;
}
static void wl_output_done(void* data, struct wl_output* output) { (void)data; (void)output; }
static void wl_output_scale(void* data, struct wl_output* output, int32_t factor) {
    (void)output;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    if (factor <= 0 || factor == w->output_scale) return;
    w->output_scale = factor;
    // With fractional-scale-v1 + wp_viewporter the preferred-scale event is exact; the
    // integer output scale is only a fallback when viewporter is unavailable.
    if (w->wl_fractional && w->wl_viewport) return;
    if (w->wl_egl_window && w->wl.egl_window_resize) {
        float next = wayland_env_scale_override((float)factor);
        if (!w->wl_viewport) {
            // Without viewporter the buffer size is the surface size; an integer buffer_scale
            // must match the render scale exactly.
            next = (float)lroundf(next);
        }
        if (next < 1.0f) next = 1.0f;
        w->scale = next;
        wayland_apply_scale(w);
        w->wl.egl_window_resize(w->wl_egl_window,
            (int)lroundf(w->logical_w * w->scale), (int)lroundf(w->logical_h * w->scale), 0, 0);
        emit_event(EV_RESIZE, w->logical_w, w->logical_h, w->scale, 0, 0);
    }
}

static const struct wl_output_listener output_listener = {
    wl_output_geometry, wl_output_mode, wl_output_done, wl_output_scale,
};

static void fractional_scale_preferred(void* data, struct wp_fractional_scale_v1* scale, uint32_t scale_120) {
    (void)scale;
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)data;
    float factor = (float)scale_120 / 120.0f;
    if (factor < 0.5f || factor > 8.0f) return;
    factor = wayland_env_scale_override(factor);
    if (!w->wl_viewport) {
        // Without viewporter the buffer stays at scale 1 but cannot describe its logical size,
        // which can cause a configure/resize feedback loop. Fall back to an integer scale.
        factor = (float)(w->output_scale > 0 ? w->output_scale : 1);
        factor = wayland_env_scale_override(factor);
        factor = (float)lroundf(factor);
    }
    if (factor < 1.0f) factor = 1.0f;
    w->scale = factor;
    if (w->wl_egl_window && w->wl.egl_window_resize) {
        wayland_apply_scale(w);
        w->wl.egl_window_resize(w->wl_egl_window,
            (int)lroundf(w->logical_w * w->scale), (int)lroundf(w->logical_h * w->scale), 0, 0);
        emit_event(EV_RESIZE, w->logical_w, w->logical_h, w->scale, 0, 0);
    }
}

static const struct wp_fractional_scale_listener fractional_scale_listener = {
    fractional_scale_preferred,
};

static int fullscreen_requested(void) {
    const char* env = getenv("SOFUU_UI_FULLSCREEN");
    return env && *env && strcmp(env, "0") != 0;
}

static int wayland_init(SofuuLinuxWindow* w, const char* title, float width, float height, float min_w, float min_h) {
    const char* names[] = {"libwayland-client.so.0", "libwayland-client.so"};
    w->wl.lib = load_first(names, 2);
    if (!w->wl.lib) return 0;
#define SYM(field, name) do { *(void**)(&w->wl.field) = dlsym(w->wl.lib, name); } while (0)
    SYM(display_connect, "wl_display_connect");
    SYM(display_disconnect, "wl_display_disconnect");
    SYM(display_dispatch, "wl_display_dispatch");
    SYM(display_dispatch_pending, "wl_display_dispatch_pending");
    SYM(display_flush, "wl_display_flush");
    SYM(display_roundtrip, "wl_display_roundtrip");
    SYM(display_get_fd, "wl_display_get_fd");
    SYM(proxy_marshal_flags, "wl_proxy_marshal_flags");
    SYM(proxy_destroy, "wl_proxy_destroy");
    SYM(proxy_add_listener, "wl_proxy_add_listener");
    SYM(proxy_get_version, "wl_proxy_get_version");
    SYM(registry_bind, "wl_registry_bind");
#undef SYM
    const char* egl_names[] = {"libwayland-egl.so.1", "libwayland-egl.so"};
    w->wl.egl_lib = load_first(egl_names, 2);
    if (w->wl.egl_lib) {
        *(void**)(&w->wl.egl_window_create) = dlsym(w->wl.egl_lib, "wl_egl_window_create");
        *(void**)(&w->wl.egl_window_destroy) = dlsym(w->wl.egl_lib, "wl_egl_window_destroy");
        *(void**)(&w->wl.egl_window_resize) = dlsym(w->wl.egl_lib, "wl_egl_window_resize");
    }
    w->wl.registry_interface = (const struct wl_interface*)dlsym(w->wl.lib, "wl_registry_interface");
    w->wl.compositor_interface = (const struct wl_interface*)dlsym(w->wl.lib, "wl_compositor_interface");
    w->wl.surface_interface = (const struct wl_interface*)dlsym(w->wl.lib, "wl_surface_interface");
    w->wl.seat_interface = (const struct wl_interface*)dlsym(w->wl.lib, "wl_seat_interface");
    w->wl.pointer_interface = (const struct wl_interface*)dlsym(w->wl.lib, "wl_pointer_interface");
    w->wl.keyboard_interface = (const struct wl_interface*)dlsym(w->wl.lib, "wl_keyboard_interface");
    w->wl.output_interface = (const struct wl_interface*)dlsym(w->wl.lib, "wl_output_interface");
    if (!w->wl.display_connect || !w->wl.display_disconnect || !w->wl.display_dispatch ||
        !w->wl.display_dispatch_pending || !w->wl.display_flush || !w->wl.display_roundtrip ||
        !w->wl.display_get_fd || !w->wl.registry_bind || !w->wl.proxy_marshal_flags ||
        !w->wl.proxy_add_listener || !w->wl.proxy_get_version ||
        !w->wl.compositor_interface || !w->wl.surface_interface || !w->wl.seat_interface ||
        !w->wl.output_interface || !w->wl.egl_window_create) {
        return 0;
    }
    w->wl_display = w->wl.display_connect(NULL);
    if (!w->wl_display) return 0;

    struct wl_proxy* registry_proxy = w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_display, 1,
        w->wl.registry_interface, w->wl.proxy_get_version((struct wl_proxy*)w->wl_display), 0, NULL);
    w->wl_registry = (struct wl_registry*)registry_proxy;
    if (!w->wl_registry) return 0;
    if (w->wl.proxy_add_listener(registry_proxy, (void (**)(void))&registry_listener, w) != 0) return 0;
    if (w->wl.display_roundtrip(w->wl_display) < 0) return 0;
    if (!w->wl_compositor || !w->wl_wm_base) return 0;

    w->wl.proxy_add_listener((struct wl_proxy*)w->wl_wm_base, (void (**)(void))&wm_base_listener, w);
    if (w->wl_seat) w->wl.proxy_add_listener((struct wl_proxy*)w->wl_seat, (void (**)(void))&seat_listener, w);
    if (w->wl_output) w->wl.proxy_add_listener((struct wl_proxy*)w->wl_output, (void (**)(void))&output_listener, w);
    w->wl.display_roundtrip(w->wl_display);

    w->wl_surface = (struct wl_surface*)w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_compositor, 0,
        w->wl.surface_interface, w->wl.proxy_get_version((struct wl_proxy*)w->wl_compositor), 0, NULL);
    if (!w->wl_surface) return 0;
    w->wl_surface_created = 1;
    if (w->wl_viewporter) {
        w->wl_viewport = (struct wp_viewport*)w->wl.proxy_marshal_flags(
            (struct wl_proxy*)w->wl_viewporter, 1, &wp_viewport_interface,
            w->wl.proxy_get_version((struct wl_proxy*)w->wl_viewporter), 0, NULL, w->wl_surface);
    }

    w->wl_xdg_surface = (struct xdg_surface*)w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_wm_base, 2,
        &xdg_surface_interface, w->wl.proxy_get_version((struct wl_proxy*)w->wl_wm_base), 0, NULL, w->wl_surface);
    if (!w->wl_xdg_surface) return 0;
    w->wl.proxy_add_listener((struct wl_proxy*)w->wl_xdg_surface, (void (**)(void))&xdg_surface_listener, w);
    w->wl_toplevel = (struct xdg_toplevel*)w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_xdg_surface, 1,
        &xdg_toplevel_interface, w->wl.proxy_get_version((struct wl_proxy*)w->wl_xdg_surface), 0, NULL);
    if (!w->wl_toplevel) return 0;
    w->wl.proxy_add_listener((struct wl_proxy*)w->wl_toplevel, (void (**)(void))&toplevel_listener, w);
    w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_toplevel, 2, NULL,
        w->wl.proxy_get_version((struct wl_proxy*)w->wl_toplevel), 0, title);
    w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_toplevel, 3, NULL,
        w->wl.proxy_get_version((struct wl_proxy*)w->wl_toplevel), 0, "dev.sofuu.ui");
    w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_toplevel, 8, NULL,
        w->wl.proxy_get_version((struct wl_proxy*)w->wl_toplevel), 0, (int32_t)min_w, (int32_t)min_h);
    if (fullscreen_requested()) {
        // xdg_toplevel.set_fullscreen(NULL) = kiosk mode on the compositor-chosen output.
        w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_toplevel, 11, NULL,
            w->wl.proxy_get_version((struct wl_proxy*)w->wl_toplevel), 0, NULL);
    }

    if (w->wl_fractional_manager) {
        w->wl_fractional = (struct wp_fractional_scale_v1*)w->wl.proxy_marshal_flags(
            (struct wl_proxy*)w->wl_fractional_manager, 1, &wp_fractional_scale_v1_interface,
            w->wl.proxy_get_version((struct wl_proxy*)w->wl_fractional_manager), 0, NULL,
            w->wl_surface);
        if (w->wl_fractional) {
            w->wl.proxy_add_listener((struct wl_proxy*)w->wl_fractional,
                (void (**)(void))&fractional_scale_listener, w);
        }
    }
    w->scale = wayland_env_scale_override((float)(w->output_scale > 0 ? w->output_scale : 1));
    if (!w->wl_viewport) w->scale = (float)lroundf(w->scale);
    if (w->scale < 1.0f) w->scale = 1.0f;
    w->logical_w = width;
    w->logical_h = height;
    wayland_apply_scale(w);
    w->wl_egl_window = w->wl.egl_window_create(w->wl_surface,
        (int)lroundf(width * w->scale), (int)lroundf(height * w->scale));
    if (!w->wl_egl_window) return 0;

    if (!egl_load(&w->egl)) return 0;
    const intptr_t attribs[] = {0};
    w->egl_display = w->egl.get_platform_display
        ? w->egl.get_platform_display(EGL_PLATFORM_WAYLAND_KHR, w->wl_display, attribs)
        : EGL_NO_DISPLAY;
    if (w->egl_display == EGL_NO_DISPLAY && w->egl.get_display) {
        w->egl_display = w->egl.get_display(w->wl_display);
    }
    if (w->egl_display == EGL_NO_DISPLAY) return 0;
    load_gl_libs(w);
    EGLint major = 0, minor = 0;
    if (!w->egl.initialize(w->egl_display, &major, &minor)) return 0;
    if (!egl_choose_and_create(w, w->wl_egl_window)) return 0;

    w->wl.proxy_marshal_flags((struct wl_proxy*)w->wl_surface, 6, NULL,
        w->wl.proxy_get_version((struct wl_proxy*)w->wl_surface), 0);
    w->wl.display_roundtrip(w->wl_display);
    apply_configured_size(w);
    return 1;
}

static int wayland_poll(SofuuLinuxWindow* w, int wait_forever, double timeout_sec) {
    w->wl.display_flush(w->wl_display);
    struct pollfd fds[2];
    fds[0].fd = w->wl.display_get_fd(w->wl_display);
    fds[0].events = POLLIN;
    fds[1].fd = g_wake_fd;
    fds[1].events = POLLIN;
    int timeout_ms = wait_forever ? -1 : (int)ceil(timeout_sec * 1000.0);
    if (timeout_ms < 0 && !wait_forever) timeout_ms = 0;
    int r = poll(fds, 2, timeout_ms);
    if (r < 0) {
        if (errno == EINTR) return 1;
        w->closed = 1;
        return 0;
    }
    if (fds[1].revents & POLLIN) drain_wake();
    // A vanished compositor reports POLLHUP/POLLERR without POLLIN; treat it as a closed
    // window instead of spinning on an always-ready fd.
    if (fds[0].revents & (POLLHUP | POLLERR | POLLNVAL)) {
        w->closed = 1;
        return 0;
    }
    if (fds[0].revents & POLLIN) {
        if (w->wl.display_dispatch(w->wl_display) < 0) {
            w->closed = 1;
            return 0;
        }
    } else {
        w->wl.display_dispatch_pending(w->wl_display);
    }
    return !w->closed;
}

// ── X11 callbacks ──────────────────────────────────────────────────────────────────────────
static int x11_ev_type(const SofuuXEvent* e) { return *(const int*)e; }

// Xlib's event ABI is fixed for a given pointer width. These offsets cover the fields this
// shim reads; they are identical on every Xlib implementation (XCB included).
#if UINTPTR_MAX == 0xFFFFFFFFu
#define X11_OFF_FIELDS 32
#define X11_OFF_STATE 48
#define X11_OFF_BUTTON 52
#define X11_OFF_CONFIGURE_WIDTH 32
#define X11_OFF_CONFIGURE_HEIGHT 36
#define X11_OFF_MESSAGE_TYPE 20
#else
#define X11_OFF_FIELDS 64
#define X11_OFF_STATE 80
#define X11_OFF_BUTTON 84
#define X11_OFF_CONFIGURE_WIDTH 56
#define X11_OFF_CONFIGURE_HEIGHT 60
#define X11_OFF_MESSAGE_TYPE 40
#endif

static int x11_scale_from_display(SofuuLinuxWindow* w) {
    const char* scale_env = getenv("SOFUU_UI_SCALE");
    if (scale_env && *scale_env) {
        float s = (float)atof(scale_env);
        if (!(s >= 1.0f)) s = 1.0f; // also catches NaN
        if (s > 4.0f) s = 4.0f;
        return (int)lroundf(s * 96.0f);
    }
    if (w->x11.get_default) {
        char* dpi = w->x11.get_default(w->x11_display, "Xft", "dpi");
        if (dpi && *dpi) {
            float v = (float)atof(dpi);
            if (v >= 48.0f && v <= 480.0f) return (int)lroundf(v);
        }
    }
    return 96;
}

static void x11_handle_event(SofuuLinuxWindow* w, const SofuuXEvent* e) {
    const char* p = (const char*)e;
    int type = x11_ev_type(e);
    if (type == XEV_CLIENT_MESSAGE) {
        const SofuuXClientMessageEvent* cm = (const SofuuXClientMessageEvent*)e;
        if (cm->message_type == w->x11_wm_delete) {
            w->closed = 1;
            emit_event(EV_CLOSE, 0, 0, 0, 0, 0);
        }
        return;
    }
    if (type == XEV_CONFIGURE_NOTIFY) {
        int32_t width = *(const int32_t*)(p + X11_OFF_CONFIGURE_WIDTH);
        int32_t height = *(const int32_t*)(p + X11_OFF_CONFIGURE_HEIGHT);
        if (width > 0 && height > 0) {
            float lw = (float)width / w->scale;
            float lh = (float)height / w->scale;
            if (fabsf(lw - w->logical_w) >= 0.5f || fabsf(lh - w->logical_h) >= 0.5f) {
                w->logical_w = lw;
                w->logical_h = lh;
                emit_event(EV_RESIZE, lw, lh, w->scale, 0, 0);
            }
        }
        return;
    }
    if (type == XEV_DESTROY_NOTIFY) {
        w->closed = 1;
        emit_event(EV_CLOSE, 0, 0, 0, 0, 0);
        return;
    }
    if (type == XEV_MOTION_NOTIFY || type == XEV_BUTTON_PRESS || type == XEV_BUTTON_RELEASE || type == XEV_KEY_PRESS) {
        // Button/Motion/Key events share x,y/state/keycode field positions.
        static const size_t field_base = X11_OFF_FIELDS;
        float x = (float)*(const int32_t*)(p + field_base) / w->scale;
        float y = (float)*(const int32_t*)(p + field_base + 4) / w->scale;
        if (type == XEV_MOTION_NOTIFY) {
            emit_event(EV_POINTER_MOVE, x, y, 0, 0, 0);
            return;
        }
        if (type == XEV_BUTTON_PRESS || type == XEV_BUTTON_RELEASE) {
            uint32_t button = *(const uint32_t*)(p + X11_OFF_BUTTON);
            if (button == 4 || button == 5 || button == 6 || button == 7) {
                // Scroll wheel button press/release pairs are not real pointer buttons.
                if (type == XEV_BUTTON_PRESS) {
                    float d = 40.0f;
                    if (button == 4) emit_event(EV_SCROLL, x, y, 0, d, 0);
                    else if (button == 5) emit_event(EV_SCROLL, x, y, 0, -d, 0);
                    else if (button == 6) emit_event(EV_SCROLL, x, y, -d, 0, 0);
                    else emit_event(EV_SCROLL, x, y, d, 0, 0);
                }
                return;
            }
            uint32_t b = button == 1 ? 0 : button == 2 ? 1 : button == 3 ? 2 : 3;
            emit_event(type == XEV_BUTTON_PRESS ? EV_POINTER_DOWN : EV_POINTER_UP, x, y, 0, 0, b);
            return;
        }
        // KeyPress
        if (w->x11_has_xim && w->x11.filter_event && w->x11.filter_event((void*)e, w->x11_window)) {
            return; // consumed by XIM (compose/dead keys/IME preedit)
        }
        uint32_t keycode = *(const uint32_t*)(p + X11_OFF_BUTTON);
        KeySym sym = 0;
        char text[64] = {0};
        int n = 0;
        if (w->x11_has_xim && w->x11.utf8_lookup_string) {
            int status = 0;
            n = w->x11.utf8_lookup_string(w->x11_ic, (void*)e, text, sizeof(text) - 1, &sym, &status);
        } else if (w->x11.lookup_string) {
            n = w->x11.lookup_string((void*)e, text, sizeof(text) - 1, &sym, NULL);
        }
        uint32_t norm = key_enum_from_keysym((uint32_t)sym);
        uint32_t state_mask = *(const uint32_t*)(p + X11_OFF_STATE);
        uint32_t mods = 0;
        if (state_mask & (1u << 0)) mods |= 1u; // ShiftMask
        if (state_mask & (1u << 2)) mods |= 2u; // ControlMask
        if (state_mask & (1u << 3)) mods |= 4u; // Mod1Mask
        if (state_mask & (1u << 6)) mods |= 8u; // Mod4Mask
        emit_event(EV_KEY_DOWN, 0, 0, (float)mods, 0, norm ? norm : (1000u + keycode));
        if (n > 0) {
            text[n] = '\0';
            emit_utf8_codepoints(text);
        } else if (sym) {
            uint32_t cp = (uint32_t)sym;
            if (cp >= 0x20 && cp < 0x7f) emit_event(EV_TEXT_CHAR, 0, 0, 0, 0, cp);
        }
        return;
    }
    if (type == XEV_FOCUS_IN) {
        if (w->x11_has_xim && w->x11.set_ic_focus) w->x11.set_ic_focus(w->x11_ic);
    }
}

static void x11_request_fullscreen(SofuuLinuxWindow* w) {
    if (!fullscreen_requested()) return;
    int screen = w->x11_screen;
    if (w->x11.intern_atom && w->x11.change_property) {
        Atom state = w->x11.intern_atom(w->x11_display, "_NET_WM_STATE", X11_FALSE);
        Atom full = w->x11.intern_atom(w->x11_display, "_NET_WM_STATE_FULLSCREEN", X11_FALSE);
        if (state && full) {
            long atom = (long)full;
            w->x11.change_property(w->x11_display, w->x11_window, state, 4 /* XA_ATOM */, 32,
                X11_PROP_MODE_REPLACE, (const unsigned char*)&atom, 1);
            if (w->x11.send_event && w->x11_root) {
                SofuuXClientMessageEvent cm;
                memset(&cm, 0, sizeof(cm));
                cm.type = XEV_CLIENT_MESSAGE;
                cm.window = w->x11_window;
                cm.message_type = state;
                cm.format = 32;
                cm.data.l[0] = 1; // _NET_WM_STATE_ADD
                cm.data.l[1] = (long)full;
                cm.data.l[2] = 0;
                cm.data.l[3] = 1; // normal application
                w->x11.send_event(w->x11_display, w->x11_root, X11_FALSE,
                    (1L << 20) | (1L << 19) /* SubstructureRedirect|Notify */, &cm);
            }
        }
    }
    // Also cover bare X servers / kiosk sessions without an EWMH window manager.
    if (w->x11.move_resize_window && w->x11.display_width && w->x11.display_height) {
        int sw = w->x11.display_width(w->x11_display, screen);
        int sh = w->x11.display_height(w->x11_display, screen);
        if (sw > 0 && sh > 0) {
            w->x11.move_resize_window(w->x11_display, w->x11_window, 0, 0,
                (unsigned)sw, (unsigned)sh);
            w->logical_w = (float)sw / w->scale;
            w->logical_h = (float)sh / w->scale;
        }
    }
    w->x11.flush(w->x11_display);
}

static int x11_init(SofuuLinuxWindow* w, const char* title, float width, float height) {
    const char* names[] = {"libX11.so.6", "libX11.so"};
    w->x11.lib = load_first(names, 2);
    if (!w->x11.lib) return 0;
#define SYM(field, name) do { *(void**)(&w->x11.field) = dlsym(w->x11.lib, name); } while (0)
    SYM(open_display, "XOpenDisplay");
    SYM(default_screen, "XDefaultScreen");
    SYM(root_window, "XRootWindow");
    SYM(create_simple_window, "XCreateSimpleWindow");
    SYM(select_input, "XSelectInput");
    SYM(map_window, "XMapWindow");
    SYM(store_name, "XStoreName");
    SYM(intern_atom, "XInternAtom");
    SYM(set_wm_protocols, "XSetWMProtocols");
    SYM(change_property, "XChangeProperty");
    SYM(flush, "XFlush");
    SYM(pending, "XPending");
    SYM(next_event, "XNextEvent");
    SYM(filter_event, "XFilterEvent");
    SYM(connection_number, "XConnectionNumber");
    SYM(send_event, "XSendEvent");
    SYM(move_resize_window, "XMoveResizeWindow");
    SYM(display_width, "XDisplayWidth");
    SYM(display_height, "XDisplayHeight");
    SYM(destroy_window, "XDestroyWindow");
    SYM(close_display, "XCloseDisplay");
    SYM(lookup_string, "XLookupString");
    SYM(utf8_lookup_string, "Xutf8LookupString");
    SYM(open_im, "XOpenIM");
    SYM(create_ic, "XCreateIC");
    SYM(set_ic_focus, "XSetICFocus");
    SYM(destroy_ic, "XDestroyIC");
    SYM(close_im, "XCloseIM");
    SYM(get_default, "XGetDefault");
#undef SYM
    if (!w->x11.open_display || !w->x11.create_simple_window || !w->x11.next_event) return 0;
    w->x11_display = w->x11.open_display(NULL);
    if (!w->x11_display) return 0;
    int screen = w->x11.default_screen(w->x11_display);
    Window root = w->x11.root_window(w->x11_display, screen);
    w->x11_screen = screen;
    w->x11_root = root;
    int scale_dpi = x11_scale_from_display(w);
    w->scale = (float)scale_dpi / 96.0f;
    if (w->scale < 1.0f || w->scale > 4.0f) w->scale = 1.0f;
    w->logical_w = width;
    w->logical_h = height;
    unsigned pw = (unsigned)lroundf(width * w->scale);
    unsigned ph = (unsigned)lroundf(height * w->scale);
    w->x11_window = w->x11.create_simple_window(w->x11_display, root, 0, 0, pw, ph, 0, 0, 0);
    if (!w->x11_window) return 0;
    w->x11.select_input(w->x11_display, w->x11_window,
        X11_STRUCTURE_NOTIFY_MASK | X11_KEY_PRESS_MASK | X11_KEY_RELEASE_MASK |
        X11_BUTTON_PRESS_MASK | X11_BUTTON_RELEASE_MASK | X11_POINTER_MOTION_MASK |
        X11_EXPOSURE_MASK | X11_FOCUS_CHANGE_MASK);
    w->x11.store_name(w->x11_display, w->x11_window, title);
    if (w->x11.intern_atom && w->x11.change_property) {
        Atom net_name = w->x11.intern_atom(w->x11_display, "_NET_WM_NAME", X11_FALSE);
        Atom utf8 = w->x11.intern_atom(w->x11_display, "UTF8_STRING", X11_FALSE);
        if (net_name && utf8) {
            w->x11.change_property(w->x11_display, w->x11_window, net_name, utf8, 8,
                X11_PROP_MODE_REPLACE, (const unsigned char*)title, (int)strlen(title));
        }
    }
    w->x11_wm_delete = w->x11.intern_atom(w->x11_display, "WM_DELETE_WINDOW", X11_FALSE);
    if (w->x11_wm_delete) w->x11.set_wm_protocols(w->x11_display, w->x11_window, &w->x11_wm_delete, 1);
    if (w->x11.open_im && w->x11.create_ic) {
        w->x11_im = w->x11.open_im(w->x11_display, NULL, NULL, NULL);
        if (w->x11_im) {
            w->x11_ic = w->x11.create_ic(w->x11_im,
                "inputStyle", (unsigned long)(0x0008 | 0x0400),
                "clientWindow", (unsigned long)w->x11_window,
                "focusWindow", (unsigned long)w->x11_window,
                (void*)0);
            w->x11_has_xim = w->x11_ic != NULL;
        }
    }
    x11_request_fullscreen(w);
    w->x11.map_window(w->x11_display, w->x11_window);
    w->x11.flush(w->x11_display);

    if (!egl_load(&w->egl)) return 0;
    const intptr_t attribs[] = {0};
    w->egl_display = w->egl.get_platform_display
        ? w->egl.get_platform_display(EGL_PLATFORM_X11_KHR, w->x11_display, attribs)
        : EGL_NO_DISPLAY;
    if (w->egl_display == EGL_NO_DISPLAY && w->egl.get_display) {
        w->egl_display = w->egl.get_display(w->x11_display);
    }
    if (w->egl_display == EGL_NO_DISPLAY) return 0;
    load_gl_libs(w);
    EGLint major = 0, minor = 0;
    if (!w->egl.initialize(w->egl_display, &major, &minor)) return 0;
    if (!egl_choose_and_create(w, (void*)(uintptr_t)w->x11_window)) return 0;
    return 1;
}

static int x11_poll(SofuuLinuxWindow* w, int wait_forever, double timeout_sec) {
    while (w->x11.pending(w->x11_display) > 0) {
        SofuuXEvent ev;
        w->x11.next_event(w->x11_display, &ev);
        x11_handle_event(w, &ev);
        if (w->closed) return 0;
    }
    w->x11.flush(w->x11_display);
    struct pollfd fds[2];
    fds[0].fd = w->x11.connection_number(w->x11_display);
    fds[0].events = POLLIN;
    fds[1].fd = g_wake_fd;
    fds[1].events = POLLIN;
    int timeout_ms = wait_forever ? -1 : (int)ceil(timeout_sec * 1000.0);
    if (timeout_ms < 0 && !wait_forever) timeout_ms = 0;
    int r = poll(fds, 2, timeout_ms);
    if (r < 0) {
        if (errno == EINTR) return 1;
        w->closed = 1;
        emit_event(EV_CLOSE, 0, 0, 0, 0, 0);
        return 0;
    }
    if (fds[1].revents & POLLIN) drain_wake();
    // The X connection died (server shutdown / kill): report closure rather than spinning.
    if (fds[0].revents & (POLLHUP | POLLERR | POLLNVAL)) {
        w->closed = 1;
        emit_event(EV_CLOSE, 0, 0, 0, 0, 0);
        return 0;
    }
    if (fds[0].revents & POLLIN) {
        while (w->x11.pending(w->x11_display) > 0) {
            SofuuXEvent ev;
            w->x11.next_event(w->x11_display, &ev);
            x11_handle_event(w, &ev);
            if (w->closed) return 0;
        }
    }
    return !w->closed;
}

// ── Public C ABI ───────────────────────────────────────────────────────────────────────────
static void linux_destroy(SofuuLinuxWindow* w) {
    if (!w) return;
    if (w->egl.make_current && w->egl_display) w->egl.make_current(w->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (w->egl.destroy_surface && w->egl_surface) w->egl.destroy_surface(w->egl_display, w->egl_surface);
    if (w->egl.destroy_context && w->egl_context) w->egl.destroy_context(w->egl_display, w->egl_context);
    if (w->egl.terminate && w->egl_display) w->egl.terminate(w->egl_display);
    if (w->wl.egl_window_destroy && w->wl_egl_window) w->wl.egl_window_destroy(w->wl_egl_window);
    if (w->xkb_state && w->xkb.state_unref) w->xkb.state_unref(w->xkb_state);
    if (w->xkb_keymap && w->xkb.keymap_unref) w->xkb.keymap_unref(w->xkb_keymap);
    if (w->xkb_ctx && w->xkb.context_unref) w->xkb.context_unref(w->xkb_ctx);
    if (w->wl_display && w->wl.display_disconnect) w->wl.display_disconnect(w->wl_display);
    if (w->kind == 1) {
        if (w->x11_has_xim && w->x11.destroy_ic) w->x11.destroy_ic(w->x11_ic);
        if (w->x11_im && w->x11.close_im) w->x11.close_im(w->x11_im);
        if (w->x11_window && w->x11.destroy_window) w->x11.destroy_window(w->x11_display, w->x11_window);
        if (w->x11_display && w->x11.close_display) w->x11.close_display(w->x11_display);
    }
    if (g_window == w) g_window = NULL;
    free(w);
}

void* sofuu_linux_create_window(const char* title, float width, float height,
                                float min_w, float min_h, SofuuEventCb cb, void* user_ctx) {
    set_error("");
    if (g_window) {
        set_error("E_UI_ALREADY_RUNNING: a Sofuu UI window is already open in this process");
        return NULL;
    }
    init_wake();
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)calloc(1, sizeof(SofuuLinuxWindow));
    if (!w) {
        set_error("E_UI_NO_MEMORY: failed to allocate window state");
        return NULL;
    }
    w->cb = cb;
    w->user_ctx = user_ctx;
    w->logical_w = width;
    w->logical_h = height;
    w->scale = 1.0f;
    w->output_scale = 0;
    w->configured_w = (int)lroundf(width);
    w->configured_h = (int)lroundf(height);
    w->kind = 0;
    g_window = w;
    xkb_load(w);

    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    const char* x11_display = getenv("DISPLAY");
    int ok = 0;
    if (wayland_display && *wayland_display) {
        ok = wayland_init(w, title && *title ? title : "Sofuu", width, height,
                          min_w > 1 ? min_w : 360, min_h > 1 ? min_h : 280);
        if (!ok) {
            // A broken Wayland connection must not prevent the X11 fallback.
            linux_destroy(w);
            w = NULL;
            g_window = NULL;
        }
    }
    if (!ok && x11_display && *x11_display) {
        if (!w) {
            w = (SofuuLinuxWindow*)calloc(1, sizeof(SofuuLinuxWindow));
            if (!w) {
                set_error("E_UI_NO_MEMORY: failed to allocate window state");
                return NULL;
            }
            w->cb = cb;
            w->user_ctx = user_ctx;
            w->logical_w = width;
            w->logical_h = height;
            w->scale = 1.0f;
            w->kind = 1;
            g_window = w;
            xkb_load(w);
        }
        ok = x11_init(w, title && *title ? title : "Sofuu", width, height);
    }
    if (!ok) {
        if (w) linux_destroy(w);
        if (!g_last_error[0]) {
            set_error("E_UI_NO_DISPLAY: no Wayland or X11 display server available; omit --ui to run headless");
        }
        return NULL;
    }
    return w;
}

void sofuu_linux_destroy_window(void* handle) { linux_destroy((SofuuLinuxWindow*)handle); }

void sofuu_linux_get_metrics(void* handle, float* out_w, float* out_h, float* out_scale) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    if (!w) return;
    if (out_w) *out_w = w->logical_w;
    if (out_h) *out_h = w->logical_h;
    if (out_scale) *out_scale = w->scale;
}

int sofuu_linux_poll_or_wait(void* handle, int wait_forever, double timeout_sec) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    if (!w || w->closed) return 0;
    if (w->kind == 0) return wayland_poll(w, wait_forever, timeout_sec);
    return x11_poll(w, wait_forever, timeout_sec);
}

int sofuu_linux_make_current(void* handle) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    if (!w || !w->egl.make_current) return 0;
    return w->egl.make_current(w->egl_display, w->egl_surface, w->egl_surface, w->egl_context) ? 1 : 0;
}

void sofuu_linux_swap_buffers(void* handle) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    if (!w || !w->egl.swap_buffers) return;
    if (!w->egl.swap_buffers(w->egl_display, w->egl_surface)) {
        EGLint err = w->egl.get_error ? w->egl.get_error() : 0;
        if (err == EGL_BAD_SURFACE || err == EGL_CONTEXT_LOST) w->context_lost = 1;
    }
}

void* sofuu_linux_get_proc_address(const char* name) {
    if (!name) return NULL;
    SofuuLinuxWindow* w = g_window;
    if (w && w->egl.get_proc_address) {
        void* p = w->egl.get_proc_address(name);
        if (p) return p;
    }
    void* libs[3] = {w ? w->gles_lib : NULL, w ? w->gl_lib : NULL, w ? w->gles2_lib : NULL};
    for (int i = 0; i < 3; ++i) {
        if (libs[i]) {
            void* p = dlsym(libs[i], name);
            if (p) return p;
        }
    }
    return NULL;
}

void sofuu_linux_set_theme_dark(void* handle, int dark) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    if (!w || w->kind != 1 || !w->x11.change_property) return;
    Atom variant = w->x11.intern_atom(w->x11_display, "_GTK_THEME_VARIANT", X11_FALSE);
    if (!variant) return;
    Atom utf8 = w->x11.intern_atom(w->x11_display, "UTF8_STRING", X11_FALSE);
    const char* value = dark ? "dark" : "light";
    w->x11.change_property(w->x11_display, w->x11_window, variant, utf8, 8,
        X11_PROP_MODE_REPLACE, (const unsigned char*)value, (int)strlen(value));
    if (w->x11.flush) w->x11.flush(w->x11_display);
}

int sofuu_linux_context_lost(void* handle) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    return w ? w->context_lost : 0;
}

int sofuu_linux_recover(void* handle) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    if (!w) return 0;
    if (w->egl.make_current) w->egl.make_current(w->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (w->egl.destroy_surface && w->egl_surface) w->egl.destroy_surface(w->egl_display, w->egl_surface);
    if (w->egl.destroy_context && w->egl_context) w->egl.destroy_context(w->egl_display, w->egl_context);
    w->egl_surface = EGL_NO_SURFACE;
    w->egl_context = EGL_NO_CONTEXT;
    void* native = w->kind == 0 ? (void*)w->wl_egl_window : (void*)(uintptr_t)w->x11_window;
    if (!egl_choose_and_create(w, native)) {
        w->context_lost = 1;
        return 0;
    }
    w->context_lost = 0;
    return 1;
}

int sofuu_linux_api_kind(void* handle) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    return w ? w->api_kind : 0;
}

int sofuu_linux_surface_kind(void* handle) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    return w ? w->kind : 0;
}

void* sofuu_linux_display_handle(void* handle) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    if (!w) return NULL;
    return w->kind == 0 ? (void*)w->wl_display : (void*)w->x11_display;
}

void* sofuu_linux_window_handle(void* handle) {
    SofuuLinuxWindow* w = (SofuuLinuxWindow*)handle;
    if (!w) return NULL;
    return w->kind == 0 ? (void*)w->wl_surface : (void*)(uintptr_t)w->x11_window;
}

const char* sofuu_linux_last_error(void) { return g_last_error; }
