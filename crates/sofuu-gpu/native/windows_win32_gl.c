// Sofuu Windows window + WGL shim.
//
// Win32 owns the HWND and message pump; WGL provides an OpenGL 3.3 Core context. The Rust
// GlesRenderer resolves every GL 3.3 entry point through sofuu_windows_get_proc_address, so the
// DLL only links the classic opengl32/gdi32/user32 surface that ships with Windows.
//
// Threading: the HWND, HDC, HGLRC and every GL call are Thread 0 only. Thread 1+ wakes Thread 0
// by posting WM_APP + 1 to the window.

#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#include <windowsx.h>
#include <GL/gl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#define EV_POINTER_MOVE 1
#define EV_POINTER_DOWN 2
#define EV_POINTER_UP 3
#define EV_SCROLL 4
#define EV_RESIZE 5
#define EV_CLOSE 6
#define EV_KEY_DOWN 7
#define EV_TEXT_CHAR 8

#define SOFUU_WAKE_MESSAGE (WM_APP + 1)

typedef void (*SofuuEventCb)(void* ctx, uint32_t kind, float x, float y, float a, float b, uint32_t u);

// WGL extension constants (avoid depending on wglext.h).
#define WGL_DRAW_TO_WINDOW_ARB 0x2001
#define WGL_SUPPORT_OPENGL_ARB 0x2010
#define WGL_DOUBLE_BUFFER_ARB 0x2011
#define WGL_PIXEL_TYPE_ARB 0x2013
#define WGL_TYPE_RGBA_ARB 0x202B
#define WGL_COLOR_BITS_ARB 0x2014
#define WGL_ALPHA_BITS_ARB 0x201B
#define WGL_DEPTH_BITS_ARB 0x2022
#define WGL_STENCIL_BITS_ARB 0x2023
#define WGL_SAMPLE_BUFFERS_ARB 0x2041
#define WGL_SAMPLES_ARB 0x2042
#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#define WGL_CONTEXT_FLAGS_ARB 0x2094

typedef HGLRC(WINAPI* PFN_wglCreateContextAttribsARB)(HDC, HGLRC, const int*);
typedef BOOL(WINAPI* PFN_wglChoosePixelFormatARB)(HDC, const int*, const FLOAT*, UINT, int*, UINT*);
typedef BOOL(WINAPI* PFN_wglSwapIntervalEXT)(int);

typedef struct SofuuWindowsWindow {
    HWND hwnd;
    HDC hdc;
    HGLRC hglrc;
    HINSTANCE instance;
    HMODULE opengl32;
    HMODULE dwmapi;
    PFN_wglCreateContextAttribsARB create_context_attribs;
    PFN_wglChoosePixelFormatARB choose_pixel_format;
    PFN_wglSwapIntervalEXT swap_interval;
    SofuuEventCb cb;
    void* user_ctx;
    float logical_w;
    float logical_h;
    float scale;
    float min_w;
    float min_h;
    int closed;
    int context_lost;
    int minimized;
    WCHAR pending_high_surrogate;
    int has_pending_high;
} SofuuWindowsWindow;

static SofuuWindowsWindow* g_window = NULL;
static const wchar_t* kClassName = L"SofuuGpuWindowClass";
static int g_class_registered = 0;

static void emit_event(uint32_t kind, float x, float y, float a, float b, uint32_t u) {
    if (g_window && g_window->cb) g_window->cb(g_window->user_ctx, kind, x, y, a, b, u);
}

static float current_scale(HWND hwnd, HDC hdc) {
    typedef UINT(WINAPI* PFN_GetDpiForWindow)(HWND);
    static PFN_GetDpiForWindow get_dpi = NULL;
    static int resolved = 0;
    if (!resolved) {
        HMODULE user32 = GetModuleHandleW(L"user32.dll");
        if (user32) get_dpi = (PFN_GetDpiForWindow)GetProcAddress(user32, "GetDpiForWindow");
        resolved = 1;
    }
    UINT dpi = 96;
    if (get_dpi && hwnd) dpi = get_dpi(hwnd);
    else if (hdc) dpi = (UINT)GetDeviceCaps(hdc, LOGPIXELSX);
    if (dpi < 48) dpi = 96;
    return (float)dpi / 96.0f;
}

static float current_system_scale(void) {
    typedef UINT(WINAPI* PFN_GetDpiForSystem)(void);
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    if (user32) {
        PFN_GetDpiForSystem get_dpi = (PFN_GetDpiForSystem)GetProcAddress(user32, "GetDpiForSystem");
        if (get_dpi) {
            UINT dpi = get_dpi();
            if (dpi >= 48) return (float)dpi / 96.0f;
        }
    }
    HDC screen = GetDC(NULL);
    UINT dpi = screen ? (UINT)GetDeviceCaps(screen, LOGPIXELSX) : 96;
    if (screen) ReleaseDC(NULL, screen);
    if (dpi < 48) dpi = 96;
    return (float)dpi / 96.0f;
}

// Resize the outer window so the GL client area is exactly `logical * scale` device pixels.
static void apply_client_size(SofuuWindowsWindow* w, float logical_w, float logical_h) {
    int cw = (int)(logical_w * w->scale + 0.5f);
    int ch = (int)(logical_h * w->scale + 0.5f);
    if (cw < 1) cw = 1;
    if (ch < 1) ch = 1;
    RECT rc = {0, 0, cw, ch};
    AdjustWindowRectEx(&rc, WS_OVERLAPPEDWINDOW, FALSE, 0);
    SetWindowPos(w->hwnd, NULL, 0, 0, rc.right - rc.left, rc.bottom - rc.top,
                 SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
}

static int fullscreen_requested(void) {
    const char* env = getenv("SOFUU_UI_FULLSCREEN");
    return env && *env && strcmp(env, "0") != 0;
}

static void apply_fullscreen(SofuuWindowsWindow* w) {
    if (!w || !w->hwnd || !fullscreen_requested()) return;
    HMONITOR mon = MonitorFromWindow(w->hwnd, MONITOR_DEFAULTTONEAREST);
    MONITORINFO mi;
    memset(&mi, 0, sizeof(mi));
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(mon, &mi)) return;
    SetWindowLongPtrW(w->hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
    SetWindowPos(w->hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                 mi.rcMonitor.right - mi.rcMonitor.left,
                 mi.rcMonitor.bottom - mi.rcMonitor.top,
                 SWP_FRAMECHANGED | SWP_NOOWNERZORDER);
}

static void update_logical_size_ex(SofuuWindowsWindow* w, int force) {
    RECT rc;
    if (!GetClientRect(w->hwnd, &rc)) return;
    float scale = w->scale > 0.1f ? w->scale : 1.0f;
    float lw = (float)(rc.right - rc.left) / scale;
    float lh = (float)(rc.bottom - rc.top) / scale;
    if (lw < 1.0f) lw = 1.0f;
    if (lh < 1.0f) lh = 1.0f;
    // A DPI change can keep the logical size while changing physical pixels, so `force`
    // re-emits Resize with the new scale for the engine's FrameUniforms.
    if (force || lw != w->logical_w || lh != w->logical_h) {
        w->logical_w = lw;
        w->logical_h = lh;
        emit_event(EV_RESIZE, lw, lh, scale, 0, 0);
    }
}

static void update_logical_size(SofuuWindowsWindow* w) {
    update_logical_size_ex(w, 0);
}

static void emit_utf16(SofuuWindowsWindow* w, WCHAR unit) {
    if (unit >= 0xD800 && unit <= 0xDBFF) {
        w->pending_high_surrogate = unit;
        w->has_pending_high = 1;
        return;
    }
    uint32_t cp = unit;
    if (unit >= 0xDC00 && unit <= 0xDFFF) {
        if (!w->has_pending_high) return;
        cp = 0x10000u + (((uint32_t)w->pending_high_surrogate - 0xD800u) << 10) +
             ((uint32_t)unit - 0xDC00u);
        w->has_pending_high = 0;
    } else if (w->has_pending_high) {
        w->has_pending_high = 0;
    }
    if (cp >= 0x20 && cp != 0x7F) emit_event(EV_TEXT_CHAR, 0, 0, 0, 0, cp);
}

static uint32_t key_enum_from_vk(WPARAM vk) {
    switch (vk) {
        case VK_BACK: return 1;
        case VK_RETURN: return 2;
        case VK_ESCAPE: return 3;
        case VK_LEFT: return 4;
        case VK_RIGHT: return 5;
        case VK_UP: return 6;
        case VK_DOWN: return 7;
        case VK_HOME: return 8;
        case VK_END: return 9;
        case VK_TAB: return 10;
        case VK_DELETE: return 11;
        default: return 0;
    }
}

// SOFUU_MOD_* bits shared with include/sofuu_embed.h: Shift=1, Ctrl=2, Alt=4, Meta=8.
static uint32_t key_modifiers(void) {
    uint32_t m = 0;
    if (GetKeyState(VK_SHIFT) & 0x8000) m |= 1u;
    if (GetKeyState(VK_CONTROL) & 0x8000) m |= 2u;
    if (GetKeyState(VK_MENU) & 0x8000) m |= 4u;
    if ((GetKeyState(VK_LWIN) | GetKeyState(VK_RWIN)) & 0x8000) m |= 8u;
    return m;
}

static void handle_key(SofuuWindowsWindow* w, WPARAM vk) {
    uint32_t norm = key_enum_from_vk(vk);
    emit_event(EV_KEY_DOWN, 0, 0, (float)key_modifiers(), 0, norm ? norm : (1000u + (uint32_t)vk));
}

static LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)GetWindowLongPtrW(hwnd, GWLP_USERDATA);
    if (msg == WM_NCCREATE) {
        CREATESTRUCTW* cs = (CREATESTRUCTW*)lp;
        w = (SofuuWindowsWindow*)cs->lpCreateParams;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)w);
        if (w) w->hwnd = hwnd;
    }
    if (!w) w = g_window;
    switch (msg) {
        case WM_CLOSE:
            if (w) {
                w->closed = 1;
                emit_event(EV_CLOSE, 0, 0, 0, 0, 0);
            }
            return 0;
        case WM_DESTROY:
            if (w) w->closed = 1;
            PostQuitMessage(0);
            return 0;
        case WM_ERASEBKGND:
            return 1;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            BeginPaint(hwnd, &ps);
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_SIZE:
            if (w) {
                // A minimized window has a 0x0 client area; SwapBuffers failures there are
                // not context loss and must not trigger recovery every frame.
                if (wp == SIZE_MINIMIZED) {
                    w->minimized = 1;
                    return 0;
                }
                w->minimized = 0;
                if (w->scale <= 0.1f) w->scale = current_scale(hwnd, w->hdc);
                update_logical_size(w);
            }
            return 0;
        case WM_DPICHANGED:
            if (w) {
                w->scale = (float)HIWORD(wp) / 96.0f;
                if (w->scale < 0.5f) w->scale = 1.0f;
                RECT* suggested = (RECT*)lp;
                if (suggested) {
                    SetWindowPos(hwnd, NULL, suggested->left, suggested->top,
                                 suggested->right - suggested->left,
                                 suggested->bottom - suggested->top,
                                 SWP_NOZORDER | SWP_NOACTIVATE);
                }
                // The suggested rect usually preserves the logical size; force a Resize event
                // so the engine picks up the new DPI scale.
                update_logical_size_ex(w, 1);
            }
            return 0;
        case WM_GETMINMAXINFO:
            if (w) {
                MINMAXINFO* mmi = (MINMAXINFO*)lp;
                float scale = w->scale > 0.1f ? w->scale : 1.0f;
                mmi->ptMinTrackSize.x = (LONG)(w->min_w * scale);
                mmi->ptMinTrackSize.y = (LONG)(w->min_h * scale);
            }
            return 0;
        case WM_MOUSEMOVE:
            if (w) {
                float s = w->scale > 0.1f ? w->scale : 1.0f;
                emit_event(EV_POINTER_MOVE, (float)(short)LOWORD(lp) / s, (float)(short)HIWORD(lp) / s, 0, 0, 0);
            }
            return 0;
        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
            if (w) {
                float s = w->scale > 0.1f ? w->scale : 1.0f;
                uint32_t b = msg == WM_LBUTTONDOWN ? 0 : msg == WM_RBUTTONDOWN ? 1 : 2;
                SetCapture(hwnd);
                emit_event(EV_POINTER_DOWN, (float)(short)LOWORD(lp) / s, (float)(short)HIWORD(lp) / s, 0, 0, b);
            }
            return 0;
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
            if (w) {
                float s = w->scale > 0.1f ? w->scale : 1.0f;
                uint32_t b = msg == WM_LBUTTONUP ? 0 : msg == WM_RBUTTONUP ? 1 : 2;
                ReleaseCapture();
                emit_event(EV_POINTER_UP, (float)(short)LOWORD(lp) / s, (float)(short)HIWORD(lp) / s, 0, 0, b);
            }
            return 0;
        case WM_CAPTURECHANGED:
            return 0;
        case WM_MOUSEWHEEL:
            if (w) {
                POINT pt = {(short)LOWORD(lp), (short)HIWORD(lp)};
                ScreenToClient(hwnd, &pt);
                float s = w->scale > 0.1f ? w->scale : 1.0f;
                float steps = (float)GET_WHEEL_DELTA_WPARAM(wp) / (float)WHEEL_DELTA;
                emit_event(EV_SCROLL, (float)pt.x / s, (float)pt.y / s, 0, steps * 40.0f, 0);
            }
            return 0;
        case WM_MOUSEHWHEEL:
            if (w) {
                POINT pt = {(short)LOWORD(lp), (short)HIWORD(lp)};
                ScreenToClient(hwnd, &pt);
                float s = w->scale > 0.1f ? w->scale : 1.0f;
                float steps = (float)GET_WHEEL_DELTA_WPARAM(wp) / (float)WHEEL_DELTA;
                emit_event(EV_SCROLL, (float)pt.x / s, (float)pt.y / s, steps * 40.0f, 0, 0);
            }
            return 0;
        case WM_KEYDOWN:
            if (w) handle_key(w, wp);
            return 0;
        case WM_CHAR:
            if (w) emit_utf16(w, (WCHAR)wp);
            return 0;
        case WM_SYSKEYDOWN:
        case WM_SYSCHAR:
            // Let DefWindowProc handle system accelerators (Alt+F4, Alt+Space, menu keys)
            // instead of swallowing them as text/key events.
            return DefWindowProcW(hwnd, msg, wp, lp);
        case WM_UNICHAR:
            if (wp == UNICODE_NOCHAR) return TRUE;
            if (w && wp >= 0x20 && wp != 0x7F) emit_event(EV_TEXT_CHAR, 0, 0, 0, 0, (uint32_t)wp);
            return 0;
        case SOFUU_WAKE_MESSAGE:
            return 0;
        default:
            return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

static int register_class(HINSTANCE instance) {
    if (g_class_registered) return 1;
    WNDCLASSEXW wc;
    memset(&wc, 0, sizeof(wc));
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = wndproc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorW(NULL, (LPCWSTR)IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName = kClassName;
    if (RegisterClassExW(&wc)) {
        g_class_registered = 1;
        return 1;
    }
    return GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
}

static void enable_dpi_awareness(void) {
    typedef BOOL(WINAPI* PFN_SetProcessDpiAwarenessContext)(HANDLE);
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    if (user32) {
        PFN_SetProcessDpiAwarenessContext set_ctx =
            (PFN_SetProcessDpiAwarenessContext)GetProcAddress(user32, "SetProcessDpiAwarenessContext");
        if (set_ctx) {
            // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (HANDLE)-4
            if (set_ctx((HANDLE)(intptr_t)-4)) return;
        }
        typedef BOOL(WINAPI* PFN_SetProcessDPIAware)(void);
        PFN_SetProcessDPIAware set_aware = (PFN_SetProcessDPIAware)GetProcAddress(user32, "SetProcessDPIAware");
        if (set_aware) {
            set_aware();
            return;
        }
    }
}

static int choose_pixel_format(SofuuWindowsWindow* w) {
    if (w->choose_pixel_format) {
        const int attribs[] = {
            WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
            WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
            WGL_DOUBLE_BUFFER_ARB, GL_TRUE,
            WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
            WGL_COLOR_BITS_ARB, 32,
            WGL_ALPHA_BITS_ARB, 8,
            WGL_DEPTH_BITS_ARB, 0,
            WGL_STENCIL_BITS_ARB, 0,
            WGL_SAMPLE_BUFFERS_ARB, 0,
            WGL_SAMPLES_ARB, 0,
            0,
        };
        int format = 0;
        UINT count = 0;
        if (w->choose_pixel_format(w->hdc, attribs, NULL, 1, &format, &count) && count > 0) {
            if (SetPixelFormat(w->hdc, format, NULL)) return 1;
        }
    }
    PIXELFORMATDESCRIPTOR pfd;
    memset(&pfd, 0, sizeof(pfd));
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cAlphaBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int format = ChoosePixelFormat(w->hdc, &pfd);
    if (format == 0) return 0;
    return SetPixelFormat(w->hdc, format, &pfd) ? 1 : 0;
}

static int bootstrap_wgl(SofuuWindowsWindow* w) {
    HWND hidden = CreateWindowExW(0, kClassName, L"SofuuWGLBootstrap", WS_OVERLAPPEDWINDOW,
                                  0, 0, 16, 16, NULL, NULL, w->instance, NULL);
    if (!hidden) return 0;
    HDC dc = GetDC(hidden);
    if (!dc) {
        DestroyWindow(hidden);
        return 0;
    }
    PIXELFORMATDESCRIPTOR pfd;
    memset(&pfd, 0, sizeof(pfd));
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cAlphaBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int format = ChoosePixelFormat(dc, &pfd);
    if (format == 0 || !SetPixelFormat(dc, format, &pfd)) {
        ReleaseDC(hidden, dc);
        DestroyWindow(hidden);
        return 0;
    }
    HGLRC ctx = wglCreateContext(dc);
    if (!ctx || !wglMakeCurrent(dc, ctx)) {
        if (ctx) wglDeleteContext(ctx);
        ReleaseDC(hidden, dc);
        DestroyWindow(hidden);
        return 0;
    }
    w->choose_pixel_format = (PFN_wglChoosePixelFormatARB)wglGetProcAddress("wglChoosePixelFormatARB");
    w->create_context_attribs = (PFN_wglCreateContextAttribsARB)wglGetProcAddress("wglCreateContextAttribsARB");
    w->swap_interval = (PFN_wglSwapIntervalEXT)wglGetProcAddress("wglSwapIntervalEXT");
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(ctx);
    ReleaseDC(hidden, dc);
    DestroyWindow(hidden);
    return w->create_context_attribs != NULL;
}

static int create_gl_context(SofuuWindowsWindow* w) {
    if (!choose_pixel_format(w)) return 0;
    if (w->create_context_attribs) {
        const int attribs[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
            WGL_CONTEXT_MINOR_VERSION_ARB, 3,
            WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
            WGL_CONTEXT_FLAGS_ARB, 0,
            0,
        };
        w->hglrc = w->create_context_attribs(w->hdc, NULL, attribs);
    }
    if (!w->hglrc) w->hglrc = wglCreateContext(w->hdc);
    if (!w->hglrc) return 0;
    if (!wglMakeCurrent(w->hdc, w->hglrc)) return 0;
    if (w->swap_interval) w->swap_interval(1);
    return 1;
}

void* sofuu_windows_create_window(const char* title_utf8, float width, float height,
                                  float min_w, float min_h, SofuuEventCb cb, void* user_ctx) {
    enable_dpi_awareness();
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)calloc(1, sizeof(SofuuWindowsWindow));
    if (!w) return NULL;
    w->instance = GetModuleHandleW(NULL);
    w->cb = cb;
    w->user_ctx = user_ctx;
    w->logical_w = width > 1.0f ? width : 1024.0f;
    w->logical_h = height > 1.0f ? height : 680.0f;
    w->min_w = min_w > 1.0f ? min_w : 360.0f;
    w->min_h = min_h > 1.0f ? min_h : 280.0f;
    // DPI awareness is enabled before the window exists, so the system DPI approximates the
    // monitor the window will open on. WM_DPICHANGED refines it after the window is placed.
    w->scale = current_system_scale();
    if (!register_class(w->instance)) {
        free(w);
        return NULL;
    }
    if (!bootstrap_wgl(w)) {
        free(w);
        return NULL;
    }

    WCHAR title_w[512];
    if (title_utf8 && *title_utf8) {
        MultiByteToWideChar(CP_UTF8, 0, title_utf8, -1, title_w, 512);
    } else {
        wcscpy(title_w, L"Sofuu");
    }
    int pw = (int)(w->logical_w * w->scale);
    int ph = (int)(w->logical_h * w->scale);
    g_window = w;
    w->hwnd = CreateWindowExW(0, kClassName, title_w, WS_OVERLAPPEDWINDOW,
                              CW_USEDEFAULT, CW_USEDEFAULT, pw, ph,
                              NULL, NULL, w->instance, w);
    if (!w->hwnd) {
        g_window = NULL;
        free(w);
        return NULL;
    }
    w->hdc = GetDC(w->hwnd);
    if (!w->hdc) {
        DestroyWindow(w->hwnd);
        g_window = NULL;
        free(w);
        return NULL;
    }
    w->scale = current_scale(w->hwnd, w->hdc);
    if (w->scale < 0.5f) w->scale = 1.0f;
    if (!create_gl_context(w)) {
        ReleaseDC(w->hwnd, w->hdc);
        DestroyWindow(w->hwnd);
        g_window = NULL;
        free(w);
        return NULL;
    }
    w->dwmapi = LoadLibraryW(L"dwmapi.dll");
    // Client area is the logical viewport: a 1024x680 request stays 1024x680 logical pixels at
    // every DPI, so typography and layout are identical at 100%, 150% and 200%.
    w->logical_w = width > 1.0f ? width : 1024.0f;
    w->logical_h = height > 1.0f ? height : 680.0f;
    apply_client_size(w, w->logical_w, w->logical_h);
    ShowWindow(w->hwnd, SW_SHOW);
    apply_fullscreen(w);
    UpdateWindow(w->hwnd);
    SetFocus(w->hwnd);
    return w;
}

void sofuu_windows_destroy_window(void* handle) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)handle;
    if (!w) return;
    if (w->hglrc) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(w->hglrc);
    }
    if (w->hdc && w->hwnd) ReleaseDC(w->hwnd, w->hdc);
    if (w->hwnd) DestroyWindow(w->hwnd);
    if (w->dwmapi) FreeLibrary(w->dwmapi);
    if (g_window == w) g_window = NULL;
    free(w);
}

void sofuu_windows_get_metrics(void* handle, float* out_w, float* out_h, float* out_scale) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)handle;
    if (!w) return;
    if (out_w) *out_w = w->logical_w;
    if (out_h) *out_h = w->logical_h;
    if (out_scale) *out_scale = w->scale;
}

int sofuu_windows_poll_or_wait(void* handle, int wait_forever, double timeout_sec) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)handle;
    if (!w || w->closed) return 0;
    DWORD timeout = INFINITE;
    if (!wait_forever) {
        double ms = timeout_sec * 1000.0;
        if (ms < 0.0) ms = 0.0;
        if (ms > 1000.0) ms = 1000.0;
        timeout = (DWORD)(ms + 0.5);
    }
    MsgWaitForMultipleObjectsEx(0, NULL, timeout, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
    MSG msg;
    while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            w->closed = 1;
            break;
        }
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return w->closed ? 0 : 1;
}

int sofuu_windows_make_current(void* handle) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)handle;
    if (!w || !w->hdc || !w->hglrc) return 0;
    return wglMakeCurrent(w->hdc, w->hglrc) ? 1 : 0;
}

void sofuu_windows_swap_buffers(void* handle) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)handle;
    if (!w || !w->hdc) return;
    if (w->minimized) return;
    RECT rc;
    if (GetClientRect(w->hwnd, &rc) && (rc.right <= rc.left || rc.bottom <= rc.top)) return;
    if (!SwapBuffers(w->hdc)) w->context_lost = 1;
}

void* sofuu_windows_get_proc_address(const char* name) {
    if (!name) return NULL;
    void* p = (void*)wglGetProcAddress(name);
    if (p && p != (void*)1 && p != (void*)2 && p != (void*)3 && p != (void*)-1) return p;
    SofuuWindowsWindow* w = g_window;
    HMODULE gl = w && w->opengl32 ? w->opengl32 : GetModuleHandleW(L"opengl32.dll");
    if (!gl) gl = LoadLibraryW(L"opengl32.dll");
    if (gl) return (void*)GetProcAddress(gl, name);
    return NULL;
}

void sofuu_windows_wake(void) {
    if (g_window && g_window->hwnd) PostMessageW(g_window->hwnd, SOFUU_WAKE_MESSAGE, 0, 0);
}

void sofuu_windows_set_theme_dark(void* handle, int dark) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)handle;
    if (!w || !w->hwnd) return;
    if (!w->dwmapi) w->dwmapi = LoadLibraryW(L"dwmapi.dll");
    if (!w->dwmapi) return;
    typedef HRESULT(WINAPI* PFN_DwmSetWindowAttribute)(HWND, DWORD, LPCVOID, DWORD);
    PFN_DwmSetWindowAttribute set_attr =
        (PFN_DwmSetWindowAttribute)GetProcAddress(w->dwmapi, "DwmSetWindowAttribute");
    if (!set_attr) return;
    BOOL value = dark ? TRUE : FALSE;
    // DWMWA_USE_IMMERSIVE_DARK_MODE (20 on Windows 10 20H1+, 19 on earlier builds).
    if (FAILED(set_attr(w->hwnd, 20, &value, sizeof(value)))) set_attr(w->hwnd, 19, &value, sizeof(value));
    // Repaint the non-client frame so the title bar updates immediately.
    SetWindowPos(w->hwnd, NULL, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
}

int sofuu_windows_context_lost(void* handle) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)handle;
    return w ? w->context_lost : 0;
}

int sofuu_windows_recover(void* handle) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)handle;
    if (!w || !w->hdc) return 0;
    if (w->hglrc) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(w->hglrc);
        w->hglrc = NULL;
    }
    if (w->create_context_attribs) {
        const int attribs[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
            WGL_CONTEXT_MINOR_VERSION_ARB, 3,
            WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
            WGL_CONTEXT_FLAGS_ARB, 0,
            0,
        };
        w->hglrc = w->create_context_attribs(w->hdc, NULL, attribs);
    }
    if (!w->hglrc) w->hglrc = wglCreateContext(w->hdc);
    if (!w->hglrc || !wglMakeCurrent(w->hdc, w->hglrc)) {
        w->context_lost = 1;
        return 0;
    }
    if (w->swap_interval) w->swap_interval(1);
    w->context_lost = 0;
    return 1;
}

int sofuu_windows_api_kind(void* handle) {
    (void)handle;
    return 1; // desktop OpenGL
}

int sofuu_windows_surface_kind(void* handle) {
    (void)handle;
    return 2; // Win32
}

void* sofuu_windows_display_handle(void* handle) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)handle;
    return w ? (void*)w->instance : NULL;
}

void* sofuu_windows_window_handle(void* handle) {
    SofuuWindowsWindow* w = (SofuuWindowsWindow*)handle;
    return w ? (void*)w->hwnd : NULL;
}

const char* sofuu_windows_last_error(void) {
    return "E_UI_NO_DISPLAY: failed to create a Win32 window / WGL OpenGL 3.3 context";
}
