/*
 * mod_process.c — process object bindings
 *
 * Provides:
 *   process.argv          — array of command-line args
 *   process.env           — object of environment variables
 *   process.exit(n)       — exit the process
 *   process.version       — runtime version string
 *   process.platform      — 'darwin' | 'linux' | 'win32'
 *   process.pid           — current process ID
 *   process.cwd()         — current working directory
 *   process.chdir(dir)    — change directory
 *   process.stdin         — readable stdin EventEmitter-style object
 *   process.stdout        — writable stdout object  (.write)
 *   process.stderr        — writable stderr object  (.write)
 *   process.on(event, fn) — 'SIGINT' | 'SIGTERM' | 'exit' | 'unhandledRejection'
 *   process.off(event)    — unregister handler
 *
 * Signal handlers are registered LAZILY with libuv — only when
 * process.on('SIGINT'/'SIGTERM', fn) is called.  This keeps the
 * libuv loop unreffed by default and avoids blocking the event loop
 * from exiting normally.
 */
#include "mod_process.h"
#include "loop.h"
#include "promises.h"
#include "uv.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>


extern char **environ;

static int   g_argc = 0;
static char **g_argv = NULL;

void mod_process_set_args(int argc, char **argv) {
    g_argc = argc;
    g_argv = argv;
}

/* ------------------------------------------------------------------ */
/* Signal callback state                                                */
/* ------------------------------------------------------------------ */
/*
 * We store signal callbacks as hidden properties on the process object
 * rather than in C-static JSValue variables.  This lets QuickJS's GC
 * track the references correctly and avoids gc_decref assertion failures
 * during shutdown.
 *
 * Hidden property names (not enumerable / accessible from JS normally):
 *   __sigint_handler, __sigterm_handler, __exit_handler
 */

static JSContext *g_ctx    = NULL;    /* set in mod_process_register */

/* Flag variables set by raw POSIX signal handlers (async-signal-safe) */
static volatile sig_atomic_t g_got_sigint  = 0;
static volatile sig_atomic_t g_got_sigterm = 0;

/* No uv_check_t handle — we call process_dispatch_pending_signals()
 * directly from sofuu_loop_run after each I/O poll to avoid the
 * handle lifecycle problems that a uv_check_t causes during shutdown. */

/* Retrieve a signal callback stored on the process object */
static JSValue get_process_cb(JSContext *ctx, const char *propname) {
    JSValue global  = JS_GetGlobalObject(ctx);
    JSValue process = JS_GetPropertyStr(ctx, global, "process");
    JSValue cb      = JS_GetPropertyStr(ctx, process, propname);
    JS_FreeValue(ctx, process);
    JS_FreeValue(ctx, global);
    return cb;
}

/* Store a signal callback on the process object (GC-tracked there) */
static void set_process_cb(JSContext *ctx, const char *propname, JSValue fn) {
    JSValue global  = JS_GetGlobalObject(ctx);
    JSValue process = JS_GetPropertyStr(ctx, global, "process");
    JS_SetPropertyStr(ctx, process, propname, fn);
    JS_FreeValue(ctx, process);
    JS_FreeValue(ctx, global);
}

static void dispatch_signal_cb(JSContext *ctx, const char *propname, const char *name) {
    JSValue cb = get_process_cb(ctx, propname);
    if (!JS_IsFunction(ctx, cb)) {
        JS_FreeValue(ctx, cb);
        return;
    }
    JSValue arg = JS_NewString(ctx, name);
    JSValue ret = JS_Call(ctx, cb, JS_UNDEFINED, 1, &arg);
    JS_FreeValue(ctx, arg);
    JS_FreeValue(ctx, cb);
    if (JS_IsException(ret)) {
        JSValue exc = JS_GetException(ctx);
        JSValue msg = JS_ToString(ctx, exc);
        const char *s = JS_ToCString(ctx, msg);
        fprintf(stderr, "[sofuu] Error in %s handler: %s\n", name, s ? s : "?");
        if (s) JS_FreeCString(ctx, s);
        JS_FreeValue(ctx, msg);
        JS_FreeValue(ctx, exc);
    }
    JS_FreeValue(ctx, ret);
    sofuu_flush_jobs(ctx);
}


/* Raw POSIX signal handlers — only set flags, never call JS */
static void posix_sigint_handler(int sig)  { (void)sig; g_got_sigint  = 1; }
static void posix_sigterm_handler(int sig) { (void)sig; g_got_sigterm = 1; }

static void install_sigaction(int signum, void (*handler)(int)) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(signum, &sa, NULL);
}

/*
 * process_dispatch_pending_signals:
 *
 * Called from sofuu_loop_run after each UV_RUN_ONCE iteration.
 * Dispatches any OS signals that fired since the last check.
 * Safe to call from the main JS thread.
 */
void process_dispatch_pending_signals(void) {
    if (!g_ctx) return;

    if (g_got_sigint) {
        g_got_sigint = 0;
        JSValue cb = get_process_cb(g_ctx, "__sigint_handler");
        if (JS_IsFunction(g_ctx, cb)) {
            dispatch_signal_cb(g_ctx, "__sigint_handler", "SIGINT");
        } else {
            fprintf(stderr, "\n");
            JS_FreeValue(g_ctx, cb);
            exit(130);
        }
        JS_FreeValue(g_ctx, cb);
    }
    if (g_got_sigterm) {
        g_got_sigterm = 0;
        JSValue cb = get_process_cb(g_ctx, "__sigterm_handler");
        if (JS_IsFunction(g_ctx, cb)) {
            dispatch_signal_cb(g_ctx, "__sigterm_handler", "SIGTERM");
        } else {
            JS_FreeValue(g_ctx, cb);
            exit(143);
        }
        JS_FreeValue(g_ctx, cb);
    }
}

/* ------------------------------------------------------------------ */
/* process.stdin — libuv pipe on fd 0                                   */
/* ------------------------------------------------------------------ */

static uv_pipe_t g_stdin_pipe;
static int       g_stdin_pipe_open = 0;

static JSContext *g_stdin_ctx    = NULL;
static JSValue    g_stdin_on_data  = JS_UNDEFINED;
static JSValue    g_stdin_on_end   = JS_UNDEFINED;
static JSValue    g_stdin_on_error = JS_UNDEFINED;

static void stdin_alloc_cb(uv_handle_t *h, size_t sug, uv_buf_t *buf) {
    (void)h;
    buf->base = malloc(sug);
    buf->len  = buf->base ? sug : 0;
}

static void stdin_read_cb(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf) {
    JSContext *ctx = g_stdin_ctx;

    if (nread > 0) {
        if (JS_IsFunction(ctx, g_stdin_on_data)) {
            JSValue chunk = JS_NewStringLen(ctx, buf->base, (size_t)nread);
            JSValue ret   = JS_Call(ctx, g_stdin_on_data, JS_UNDEFINED, 1, &chunk);
            JS_FreeValue(ctx, chunk);
            JS_FreeValue(ctx, ret);
            sofuu_flush_jobs(ctx);
        }
    } else if (nread == UV_EOF) {
        uv_read_stop(stream);
        if (JS_IsFunction(ctx, g_stdin_on_end)) {
            JSValue ret = JS_Call(ctx, g_stdin_on_end, JS_UNDEFINED, 0, NULL);
            JS_FreeValue(ctx, ret);
            sofuu_flush_jobs(ctx);
        }
    } else if (nread < 0 && nread != UV_EOF) {
        if (JS_IsFunction(ctx, g_stdin_on_error)) {
            JSValue err = JS_NewString(ctx, uv_strerror((int)nread));
            JSValue ret = JS_Call(ctx, g_stdin_on_error, JS_UNDEFINED, 1, &err);
            JS_FreeValue(ctx, err);
            JS_FreeValue(ctx, ret);
            sofuu_flush_jobs(ctx);
        }
    }

    if (buf->base) free(buf->base);
}

/* Open the stdin pipe once and start reading */
static void ensure_stdin_open(JSContext *ctx) {
    if (g_stdin_pipe_open) return;
    g_stdin_ctx = ctx;
    uv_pipe_init(sofuu_loop_get(), &g_stdin_pipe, 0);
    uv_pipe_open(&g_stdin_pipe, 0); /* wrap fd 0 */
    g_stdin_pipe_open = 1;
    /* The pipe IS reffed — it keeps loop alive until EOF/close, which is
     * exactly what we want when a user calls process.stdin.on('data', ...) */
}

static JSValue js_stdin_on(JSContext *ctx, JSValueConst this_val,
                           int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 2)
        return JS_ThrowTypeError(ctx, "stdin.on requires (event, callback)");

    const char *event = JS_ToCString(ctx, argv[0]);
    if (!event) return JS_EXCEPTION;

    if (!JS_IsFunction(ctx, argv[1])) {
        JS_FreeCString(ctx, event);
        return JS_ThrowTypeError(ctx, "stdin.on: handler must be a function");
    }

    JSValue fn = JS_DupValue(ctx, argv[1]);

    if (strcmp(event, "data") == 0) {
        JS_FreeValue(ctx, g_stdin_on_data);
        g_stdin_on_data = fn;
        ensure_stdin_open(ctx);
        uv_read_start((uv_stream_t*)&g_stdin_pipe, stdin_alloc_cb, stdin_read_cb);
    } else if (strcmp(event, "end") == 0) {
        JS_FreeValue(ctx, g_stdin_on_end);
        g_stdin_on_end = fn;
    } else if (strcmp(event, "error") == 0) {
        JS_FreeValue(ctx, g_stdin_on_error);
        g_stdin_on_error = fn;
    } else {
        JS_FreeValue(ctx, fn);
    }

    JS_FreeCString(ctx, event);
    return JS_UNDEFINED;
}

static JSValue js_stdin_resume(JSContext *ctx, JSValueConst this_val,
                               int argc, JSValueConst *argv) {
    (void)this_val; (void)argc; (void)argv;
    ensure_stdin_open(ctx);
    uv_read_start((uv_stream_t*)&g_stdin_pipe, stdin_alloc_cb, stdin_read_cb);
    return JS_UNDEFINED;
}

static JSValue js_stdin_pause(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv) {
    (void)ctx; (void)this_val; (void)argc; (void)argv;
    if (g_stdin_pipe_open)
        uv_read_stop((uv_stream_t*)&g_stdin_pipe);
    return JS_UNDEFINED;
}

static JSValue js_stdin_set_encoding(JSContext *ctx, JSValueConst this_val,
                                     int argc, JSValueConst *argv) {
    /* Always UTF-8 — no-op */
    (void)ctx; (void)this_val; (void)argc; (void)argv;
    return JS_UNDEFINED;
}

/* ------------------------------------------------------------------ */
/* process.stdout / process.stderr                                      */
/* ------------------------------------------------------------------ */

static JSValue js_stdout_write(JSContext *ctx, JSValueConst this_val,
                               int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1) return JS_FALSE;
    const char *s = JS_ToCString(ctx, argv[0]);
    if (s) { fputs(s, stdout); fflush(stdout); JS_FreeCString(ctx, s); }
    return JS_TRUE;
}

static JSValue js_stderr_write(JSContext *ctx, JSValueConst this_val,
                               int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1) return JS_FALSE;
    const char *s = JS_ToCString(ctx, argv[0]);
    if (s) { fputs(s, stderr); fflush(stderr); JS_FreeCString(ctx, s); }
    return JS_TRUE;
}

/* ------------------------------------------------------------------ */
/* process.exit(code)                                                   */
/* ------------------------------------------------------------------ */

static JSValue js_process_exit(JSContext *ctx, JSValueConst this_val,
                               int argc, JSValueConst *argv) {
    (void)this_val;
    int code = 0;
    if (argc >= 1) JS_ToInt32(ctx, &code, argv[0]);

    if (g_ctx) {
        JSValue exit_cb = get_process_cb(g_ctx, "__exit_handler");
        if (JS_IsFunction(g_ctx, exit_cb))
            dispatch_signal_cb(g_ctx, "__exit_handler", "exit");
        JS_FreeValue(g_ctx, exit_cb);
    }

    exit(code);
    return JS_UNDEFINED;
}


/* ------------------------------------------------------------------ */
/* process.cwd() / process.chdir()                                      */
/* ------------------------------------------------------------------ */

static JSValue js_process_cwd(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv) {
    (void)this_val; (void)argc; (void)argv;
    char buf[4096];
    size_t size = sizeof(buf);
    if (uv_cwd(buf, &size) != 0)
        return JS_ThrowTypeError(ctx, "cwd() failed");
    return JS_NewString(ctx, buf);
}

static JSValue js_process_chdir(JSContext *ctx, JSValueConst this_val,
                                int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1) return JS_ThrowTypeError(ctx, "chdir requires 1 arg");
    const char *dir = JS_ToCString(ctx, argv[0]);
    if (!dir) return JS_EXCEPTION;
    int r = uv_chdir(dir);
    JS_FreeCString(ctx, dir);
    if (r != 0) return JS_ThrowTypeError(ctx, "chdir failed: %s", uv_strerror(r));
    return JS_UNDEFINED;
}

/* ------------------------------------------------------------------ */
/* process.on / process.off                                             */
/* ------------------------------------------------------------------ */

static JSValue js_process_on(JSContext *ctx, JSValueConst this_val,
                             int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 2)
        return JS_ThrowTypeError(ctx, "process.on requires (event, fn)");
    if (!JS_IsFunction(ctx, argv[1]))
        return JS_ThrowTypeError(ctx, "process.on: handler must be a function");

    const char *event = JS_ToCString(ctx, argv[0]);
    if (!event) return JS_EXCEPTION;

    if (strcmp(event, "SIGINT") == 0) {
        /* Store via process object property — GC tracks it there */
        set_process_cb(ctx, "__sigint_handler", JS_DupValue(ctx, argv[1]));
        install_sigaction(SIGINT, posix_sigint_handler);

    } else if (strcmp(event, "SIGTERM") == 0) {
        set_process_cb(ctx, "__sigterm_handler", JS_DupValue(ctx, argv[1]));
        install_sigaction(SIGTERM, posix_sigterm_handler);

    } else if (strcmp(event, "exit") == 0) {
        set_process_cb(ctx, "__exit_handler", JS_DupValue(ctx, argv[1]));

    } else if (strcmp(event, "uncaughtException") == 0 ||
               strcmp(event, "unhandledRejection") == 0) {
        /* Acknowledged — rejection tracker in engine.c handles this */
    }
    /* else: silently ignore unknown events */

    JS_FreeCString(ctx, event);
    return JS_DupValue(ctx, this_val); /* chainable */
}


static JSValue js_process_off(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1) return JS_UNDEFINED;
    const char *event = JS_ToCString(ctx, argv[0]);
    if (!event) return JS_EXCEPTION;

    if (strcmp(event, "SIGINT") == 0) {
        set_process_cb(ctx, "__sigint_handler", JS_UNDEFINED);
    } else if (strcmp(event, "SIGTERM") == 0) {
        set_process_cb(ctx, "__sigterm_handler", JS_UNDEFINED);
    } else if (strcmp(event, "exit") == 0) {
        set_process_cb(ctx, "__exit_handler", JS_UNDEFINED);
    }

    JS_FreeCString(ctx, event);
    return JS_UNDEFINED;
}

/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* Registration                                                         */
/* ------------------------------------------------------------------ */

void mod_process_register(JSContext *ctx) {
    g_ctx = ctx;

    JSValue global  = JS_GetGlobalObject(ctx);
    JSValue process = JS_NewObject(ctx);

    /* -- Metadata -- */
    JS_SetPropertyStr(ctx, process, "version",
                      JS_NewString(ctx, "0.1.0-alpha"));
    JS_SetPropertyStr(ctx, process, "runtime",
                      JS_NewString(ctx, "sofuu"));

    const char *platform = "linux";
#if defined(__APPLE__)
    platform = "darwin";
#elif defined(_WIN32)
    platform = "win32";
#endif
    JS_SetPropertyStr(ctx, process, "platform",
                      JS_NewString(ctx, platform));
    JS_SetPropertyStr(ctx, process, "pid",
                      JS_NewInt32(ctx, (int32_t)getpid()));

    /* -- process.argv -- */
    JSValue argv_arr = JS_NewArray(ctx);
    for (int i = 0; i < g_argc; i++)
        JS_SetPropertyUint32(ctx, argv_arr, (uint32_t)i,
                             JS_NewString(ctx, g_argv[i]));
    JS_SetPropertyStr(ctx, process, "argv", argv_arr);

    /* -- process.env -- */
    JSValue env_obj = JS_NewObject(ctx);
    if (environ) {
        for (int i = 0; environ[i]; i++) {
            char *eq = strchr(environ[i], '=');
            if (!eq) continue;
            size_t klen = (size_t)(eq - environ[i]);
            char *key   = strndup(environ[i], klen);
            JS_SetPropertyStr(ctx, env_obj, key,
                              JS_NewString(ctx, eq + 1));
            free(key);
        }
    }
    JS_SetPropertyStr(ctx, process, "env", env_obj);

    /* -- Functions -- */
    JS_SetPropertyStr(ctx, process, "exit",
                      JS_NewCFunction(ctx, js_process_exit,  "exit",   1));
    JS_SetPropertyStr(ctx, process, "cwd",
                      JS_NewCFunction(ctx, js_process_cwd,   "cwd",    0));
    JS_SetPropertyStr(ctx, process, "chdir",
                      JS_NewCFunction(ctx, js_process_chdir, "chdir",  1));
    JS_SetPropertyStr(ctx, process, "on",
                      JS_NewCFunction(ctx, js_process_on,    "on",     2));
    JS_SetPropertyStr(ctx, process, "off",
                      JS_NewCFunction(ctx, js_process_off,   "off",    1));

    /* -- process.stdin -- */
    g_stdin_on_data  = JS_UNDEFINED;
    g_stdin_on_end   = JS_UNDEFINED;
    g_stdin_on_error = JS_UNDEFINED;

    JSValue stdin_obj = JS_NewObject(ctx);
    JS_SetPropertyStr(ctx, stdin_obj, "on",
                      JS_NewCFunction(ctx, js_stdin_on,           "on",          2));
    JS_SetPropertyStr(ctx, stdin_obj, "resume",
                      JS_NewCFunction(ctx, js_stdin_resume,       "resume",      0));
    JS_SetPropertyStr(ctx, stdin_obj, "pause",
                      JS_NewCFunction(ctx, js_stdin_pause,        "pause",       0));
    JS_SetPropertyStr(ctx, stdin_obj, "setEncoding",
                      JS_NewCFunction(ctx, js_stdin_set_encoding, "setEncoding", 1));
    JS_SetPropertyStr(ctx, stdin_obj, "fd",    JS_NewInt32(ctx, 0));
    JS_SetPropertyStr(ctx, stdin_obj, "isTTY", JS_NewBool(ctx, isatty(0)));
    JS_SetPropertyStr(ctx, process, "stdin", stdin_obj);

    /* -- process.stdout -- */
    JSValue stdout_obj = JS_NewObject(ctx);
    JS_SetPropertyStr(ctx, stdout_obj, "write",
                      JS_NewCFunction(ctx, js_stdout_write, "write", 1));
    JS_SetPropertyStr(ctx, stdout_obj, "fd",    JS_NewInt32(ctx, 1));
    JS_SetPropertyStr(ctx, stdout_obj, "isTTY", JS_NewBool(ctx, isatty(1)));
    JS_SetPropertyStr(ctx, process, "stdout", stdout_obj);

    /* -- process.stderr -- */
    JSValue stderr_obj = JS_NewObject(ctx);
    JS_SetPropertyStr(ctx, stderr_obj, "write",
                      JS_NewCFunction(ctx, js_stderr_write, "write", 1));
    JS_SetPropertyStr(ctx, stderr_obj, "fd",    JS_NewInt32(ctx, 2));
    JS_SetPropertyStr(ctx, stderr_obj, "isTTY", JS_NewBool(ctx, isatty(2)));
    JS_SetPropertyStr(ctx, process, "stderr", stderr_obj);

    /* Attach to global */
    JS_SetPropertyStr(ctx, global, "process", process);
    JS_FreeValue(ctx, global);

    /*
     * NOTE: SIGINT/SIGTERM handlers use POSIX sigaction + a uv_check_t to
     * dispatch JS callbacks safely on the JS thread.  The uv_check_t is only
     * started lazily when process.on('SIGINT'/'SIGTERM', fn) is first called.
     * It is unreffed so it never keeps the event loop alive on its own.
     */
}

/*
 * mod_process_cleanup — call BEFORE JS_FreeContext.
 * Releases C-static JSValue references held by stdin's event handlers.
 * Signal callbacks are stored on the process JS object and freed
 * automatically when JS_FreeContext destroys the global scope.
 */
void mod_process_cleanup(JSContext *ctx) {
    JS_FreeValue(ctx, g_stdin_on_data);  g_stdin_on_data  = JS_UNDEFINED;
    JS_FreeValue(ctx, g_stdin_on_end);   g_stdin_on_end   = JS_UNDEFINED;
    JS_FreeValue(ctx, g_stdin_on_error); g_stdin_on_error = JS_UNDEFINED;
    g_ctx = NULL;
}
