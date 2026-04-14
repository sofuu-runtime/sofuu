/*
 * engine.c — QuickJS engine lifecycle and file evaluation
 */
#include "engine.h"
#include "quickjs-libc.h"
#include "mod_console.h"
#include "mod_process.h"
#include "loop.h"
#include "timer.h"
#include "fs.h"
#include "subprocess.h"
#include "client.h"
#include "server.h"
#include "sse.h"
#include "mod_ai.h"
#include "mcp.h"
#include "stripper.h"
#include "resolver.h"
#include "cjs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <libgen.h>   /* dirname, basename */
#include <limits.h>   /* PATH_MAX */
#include <unistd.h>   /* getcwd */


/* ------------------------------------------------------------------ */
/* Unhandled Promise Rejection Tracker                                  */
/* ------------------------------------------------------------------ */

static void rejection_tracker(JSContext *ctx, JSValueConst promise,
                               JSValueConst reason, JS_BOOL is_handled,
                               void *opaque) {
    (void)promise; (void)opaque;
    if (!is_handled) {
        /* Print the unhandled rejection to stderr */
        JSValue str = JS_ToString(ctx, reason);
        const char *msg = JS_ToCString(ctx, str);
        fprintf(stderr, "\033[31m[sofuu] UnhandledPromiseRejection:\033[0m %s\n",
                msg ? msg : "(unknown)");

        /* Print stack trace if available */
        JSValue stack = JS_GetPropertyStr(ctx, reason, "stack");
        if (!JS_IsUndefined(stack) && !JS_IsNull(stack)) {
            const char *s = JS_ToCString(ctx, stack);
            if (s) { fprintf(stderr, "%s\n", s); JS_FreeCString(ctx, s); }
        }
        JS_FreeValue(ctx, stack);

        if (msg) JS_FreeCString(ctx, msg);
        JS_FreeValue(ctx, str);
    }
}

/* ------------------------------------------------------------------ */
/* Module Loader                                                        */
/* ------------------------------------------------------------------ */

/*
 * resolve_module_path:
 *
 * Given the path of the importer (base_name) and a module specifier
 * (module_name), return an absolute path to the target file.
 *
 * Rules (in order):
 *  1. Relative ('./' or '../') → join with importer dir, add .js if needed,
 *     then try as directory/index.js if file not found.
 *  2. Absolute path → use as-is.
 *  3. Everything else → pass through (future: node_modules resolution).
 */
static char *resolve_module_path(const char *base_name, const char *module_name) {
    char resolved[PATH_MAX];

    if (module_name[0] == '.' || module_name[0] == '/') {
        char full[PATH_MAX];

        if (module_name[0] == '.') {
            /* Relative import: derive directory from importer path */
            char base_copy[PATH_MAX];
            strncpy(base_copy, base_name, PATH_MAX - 1);
            base_copy[PATH_MAX - 1] = '\0';
            char *dir = dirname(base_copy);

            snprintf(full, sizeof(full), "%s/%s", dir, module_name);
        } else {
            /* Absolute path */
            strncpy(full, module_name, PATH_MAX - 1);
            full[PATH_MAX - 1] = '\0';
        }

        /* Canonicalize path (resolves .., symlinks, etc.) */
        if (realpath(full, resolved)) {
            return strdup(resolved);
        }

        /* Try adding .js extension */
        char with_ext[PATH_MAX];
        snprintf(with_ext, sizeof(with_ext), "%s.js", full);
        if (realpath(with_ext, resolved)) {
            return strdup(resolved);
        }

        /* Try as directory/index.js */
        char as_index[PATH_MAX];
        snprintf(as_index, sizeof(as_index), "%s/index.js", full);
        if (realpath(as_index, resolved)) {
            return strdup(resolved);
        }

        /* Could not resolve — return the best guess anyway so the error
         * message includes a useful filename */
        if (strstr(module_name, ".js") || strstr(module_name, ".mjs")) {
            return strdup(full);
        }
        return strdup(with_ext);
    }

    /* Non-relative, non-absolute: look in node_modules */
    {
        /* Use the importer's directory as the start for resolution */
        char base_copy[PATH_MAX];
        strncpy(base_copy, base_name, PATH_MAX - 1);
        base_copy[PATH_MAX - 1] = '\0';
        char *dir = dirname(base_copy);

        char *npm_path = npm_resolve(dir, module_name);
        if (npm_path) return npm_path;

        /* Also try from cwd in case base is not set correctly */
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd))) {
            npm_path = npm_resolve(cwd, module_name);
            if (npm_path) return npm_path;
        }
    }

    /* Fallback: return unchanged (will produce a useful error) */
    return strdup(module_name);
}


/*
 * g_loader_base — persistent heap string for the current module base path.
 * The JS runtime stores an opaque void* to this; we must never free it
 * while it is live.  We own and manage the storage here.
 */
static char *g_loader_base = NULL;

static void update_loader_base(JSRuntime *rt, const char *new_base) {
    char *prev = g_loader_base;
    g_loader_base = strdup(new_base);
    JS_SetModuleLoaderFunc(rt, NULL, sofuu_module_loader,
                           (void *)g_loader_base);
    if (prev) free(prev);
}

JSModuleDef *sofuu_module_loader(JSContext *ctx, const char *module_name, void *opaque) {
    const char *base = opaque ? (const char *)opaque : "./";
    char *path = resolve_module_path(base, module_name);

    size_t len = 0;
    char *source = engine_read_file(path, &len);
    if (!source) {
        JS_ThrowReferenceError(ctx, "Cannot find module '%s' (resolved: '%s')",
                               module_name, path);
        free(path);
        return NULL;
    }

    if (strstr(path, ".ts") || strstr(path, ".mts")) {
        char *stripped = ts_strip(source, len, &len);
        free(source);
        source = stripped;
    } else if (is_cjs(source)) {
        char *wrapped = cjs_to_esm(source, len, &len);
        free(source);
        source = wrapped;
    }

    /* Compile module — pass resolved path as filename for stack traces */
    JSValue func = JS_Eval(ctx, source, len, path,
                           JS_EVAL_TYPE_MODULE | JS_EVAL_FLAG_COMPILE_ONLY);
    free(source);

    if (JS_IsException(func)) {
        js_std_dump_error(ctx);
        free(path);
        return NULL;
    }

    JSModuleDef *mod = JS_VALUE_GET_PTR(func);

    /*
     * Update opaque to this module's resolved path so that transitive imports
     * inside it resolve relative to ITS directory (not the entry point).
     * update_loader_base() strdup's the path, so we can free it safely here.
     */
    update_loader_base(JS_GetRuntime(ctx), path);
    free(path);

    JS_FreeValue(ctx, func);
    return mod;
}


/* ------------------------------------------------------------------ */
/* Engine lifecycle                                                     */
/* ------------------------------------------------------------------ */

SofuuEngine *engine_create(void) {
    SofuuEngine *eng = calloc(1, sizeof(SofuuEngine));
    if (!eng) return NULL;

    eng->rt = JS_NewRuntime();
    if (!eng->rt) {
        free(eng);
        return NULL;
    }

    /* Set a generous memory limit (512MB) */
    JS_SetMemoryLimit(eng->rt, 512 * 1024 * 1024);
    /* GC threshold */
    JS_SetGCThreshold(eng->rt, 64 * 1024 * 1024);

    eng->ctx = JS_NewContext(eng->rt);
    if (!eng->ctx) {
        JS_FreeRuntime(eng->rt);
        free(eng);
        return NULL;
    }

    /* Add standard helpers (eval, parseInt, etc.) */
    js_init_module_std(eng->ctx, "std");
    js_init_module_os(eng->ctx, "os");

    /* Set up ESM module loader */
    JS_SetModuleLoaderFunc(eng->rt, NULL, sofuu_module_loader, NULL);

    /* Track unhandled promise rejections */
    JS_SetHostPromiseRejectionTracker(eng->rt, rejection_tracker, NULL);

    /* Init libuv event loop */
    sofuu_loop_init();

    return eng;
}

void engine_register_builtins(SofuuEngine *eng) {
    mod_console_register(eng->ctx);
    mod_process_register(eng->ctx);
    mod_timer_register(eng->ctx);
    mod_fs_register(eng->ctx);
    mod_subprocess_register(eng->ctx);
    mod_http_client_register(eng->ctx);
    mod_http_server_register(eng->ctx);
    mod_http_sse_register(eng->ctx);
    mod_ai_register(eng->ctx);
    mod_mcp_register(eng->ctx);
    mod_cjs_register(eng->ctx);

#if SOFUU_MEMORY
    extern void mod_memory_register(JSContext *ctx);
    mod_memory_register(eng->ctx);
    
    extern void mod_kv_register(JSContext *ctx);
    mod_kv_register(eng->ctx);

    extern void mod_agent_register(JSContext *ctx);
    mod_agent_register(eng->ctx);
#endif

    /* ── Build capital-S 'Sofuu' convenience global ─────────────────── */
    JSContext *ctx = eng->ctx;
    JSValue global = JS_GetGlobalObject(ctx);
    JSValue Sofuu  = JS_NewObject(ctx);

    /* Helper: copy fn from src_obj.key onto Sofuu.key (transfers ownership) */
#define COPY_FN(src_obj, key) do { \
    JSValue _v = JS_GetPropertyStr(ctx, (src_obj), (key)); \
    if (JS_IsFunction(ctx, _v)) \
        JS_SetPropertyStr(ctx, Sofuu, (key), _v); \
    else \
        JS_FreeValue(ctx, _v); \
} while(0)

    /* sofuu.fs.* → Sofuu.* */
    {
        JSValue sofuu_obj = JS_GetPropertyStr(ctx, global, "sofuu");
        JSValue fs_obj    = JS_GetPropertyStr(ctx, sofuu_obj, "fs");
        COPY_FN(fs_obj, "readFile");
        COPY_FN(fs_obj, "writeFile");
        COPY_FN(fs_obj, "appendFile");
        COPY_FN(fs_obj, "exists");
        COPY_FN(fs_obj, "readdir");
        JS_FreeValue(ctx, fs_obj);

        COPY_FN(sofuu_obj, "sleep");
        COPY_FN(sofuu_obj, "exec");

        JSValue ai = JS_GetPropertyStr(ctx, sofuu_obj, "ai");
        if (!JS_IsUndefined(ai) && !JS_IsNull(ai))
            JS_SetPropertyStr(ctx, Sofuu, "ai", ai);
        else
            JS_FreeValue(ctx, ai);

#if SOFUU_MEMORY
        JSValue mem = JS_GetPropertyStr(ctx, sofuu_obj, "memory");
        if (!JS_IsUndefined(mem) && !JS_IsNull(mem))
            JS_SetPropertyStr(ctx, Sofuu, "memory", mem);
        else
            JS_FreeValue(ctx, mem);
            
        JSValue kv = JS_GetPropertyStr(ctx, sofuu_obj, "kv");
        if (!JS_IsUndefined(kv) && !JS_IsNull(kv))
            JS_SetPropertyStr(ctx, Sofuu, "kv", kv);
        else
            JS_FreeValue(ctx, kv);

        JSValue agent_mod = JS_GetPropertyStr(ctx, sofuu_obj, "agent");
        if (!JS_IsUndefined(agent_mod) && !JS_IsNull(agent_mod))
            JS_SetPropertyStr(ctx, Sofuu, "agent", agent_mod);
        else
            JS_FreeValue(ctx, agent_mod);
#endif

        JS_FreeValue(ctx, sofuu_obj);
    }

    /* global.createServer, createSSEServer → Sofuu.* */
    COPY_FN(global, "createServer");
    COPY_FN(global, "createSSEServer");
    COPY_FN(global, "createMCPServer");

#undef COPY_FN

    JS_SetPropertyStr(ctx, global, "Sofuu", Sofuu);
    JS_FreeValue(ctx, global);
}

/* ------------------------------------------------------------------ */
/* File I/O helper                                                      */
/* ------------------------------------------------------------------ */

char *engine_read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size < 0) {
        fclose(f);
        return NULL;
    }

    char *buf = malloc((size_t)size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    size_t n = fread(buf, 1, (size_t)size, f);
    fclose(f);
    buf[n] = '\0';

    if (out_len) *out_len = n;
    return buf;
}

/* ------------------------------------------------------------------ */
/* Evaluation                                                           */
/* ------------------------------------------------------------------ */

static void dump_error(JSContext *ctx) {
    JSValue exception = JS_GetException(ctx);
    JSValue str = JS_ToString(ctx, exception);

    const char *msg = JS_ToCString(ctx, str);
    fprintf(stderr, "\033[31mError:\033[0m %s\n", msg ? msg : "(unknown error)");

    /* If it's an Error object, print the stack trace */
    JSValue stack = JS_GetPropertyStr(ctx, exception, "stack");
    if (!JS_IsUndefined(stack)) {
        const char *stack_str = JS_ToCString(ctx, stack);
        if (stack_str) {
            fprintf(stderr, "%s\n", stack_str);
            JS_FreeCString(ctx, stack_str);
        }
        JS_FreeValue(ctx, stack);
    }

    if (msg) JS_FreeCString(ctx, msg);
    JS_FreeValue(ctx, str);
    JS_FreeValue(ctx, exception);
}

int engine_eval_file(SofuuEngine *eng, const char *path) {
    size_t len = 0;
    char *source = engine_read_file(path, &len);
    if (!source) {
        fprintf(stderr, "\033[31mError:\033[0m Cannot open file '%s'\n", path);
        return 1;
    }

    /* TypeScript: strip types before evaluating */
    char *eval_src = source;
    size_t eval_len = len;
    int is_ts = 0;
    {
        size_t plen = strlen(path);
        if (plen >= 3 &&
                path[plen-3] == '.' &&
                path[plen-2] == 't' &&
                path[plen-1] == 's') {
            is_ts = 1;
        }
    }
    char *stripped = NULL;
    if (is_ts) {
        stripped = ts_strip(source, len, &eval_len);
        if (stripped) eval_src = stripped;
    }

    /* Update module loader opaque: record this file's path as the base for
     * all relative imports it makes.  update_loader_base owns the memory. */
    update_loader_base(eng->rt, path);

    JSValue result = JS_Eval(eng->ctx, eval_src, eval_len, path,
                             JS_EVAL_TYPE_MODULE);
    free(source);
    if (stripped) free(stripped);

    if (JS_IsException(result)) {
        dump_error(eng->ctx);
        JS_FreeValue(eng->ctx, result);
        return 1;
    }

    JS_FreeValue(eng->ctx, result);
    /* Run the libuv event loop — drains timers, I/O, and Promise chains */
    sofuu_loop_run(eng->ctx);
    return 0;
}

int engine_eval_string(SofuuEngine *eng, const char *source, const char *filename) {
    JSValue result = JS_Eval(eng->ctx, source, strlen(source),
                             filename ? filename : "<eval>",
                             JS_EVAL_TYPE_GLOBAL);

    if (JS_IsException(result)) {
        dump_error(eng->ctx);
        JS_FreeValue(eng->ctx, result);
        return 1;
    }

    JS_FreeValue(eng->ctx, result);
    sofuu_loop_run(eng->ctx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* REPL Evaluation                                                      */
/* ------------------------------------------------------------------ */

static char *repl_inspect(JSContext *ctx, JSValueConst val) {
    char buf[8192];

    if (JS_IsUndefined(val))  return strdup("\033[90mundefined\033[0m");
    if (JS_IsNull(val))       return strdup("\033[1mnull\033[0m");

    if (JS_IsBool(val))
        return strdup(JS_ToBool(ctx, val) ? "\033[33mtrue\033[0m" : "\033[33mfalse\033[0m");

    if (JS_IsNumber(val)) {
        double d; JS_ToFloat64(ctx, &d, val);
        if (d == (long long)d && d >= -1e15 && d <= 1e15)
            snprintf(buf, sizeof(buf), "\033[33m%lld\033[0m", (long long)d);
        else
            snprintf(buf, sizeof(buf), "\033[33m%g\033[0m", d);
        return strdup(buf);
    }

    if (JS_IsString(val)) {
        const char *s = JS_ToCString(ctx, val);
        snprintf(buf, sizeof(buf), "\033[32m'%s'\033[0m", s ? s : "");
        JS_FreeCString(ctx, s);
        return strdup(buf);
    }

    if (JS_IsFunction(ctx, val)) {
        JSValue name = JS_GetPropertyStr(ctx, val, "name");
        const char *n = JS_ToCString(ctx, name);
        snprintf(buf, sizeof(buf), "\033[36m[Function: %s]\033[0m", n && *n ? n : "(anonymous)");
        JS_FreeCString(ctx, n);
        JS_FreeValue(ctx, name);
        return strdup(buf);
    }

    /* Array / Object — JSON.stringify */
    if (JS_IsObject(val)) {
        JSValue json = JS_JSONStringify(ctx, val, JS_UNDEFINED, JS_UNDEFINED);
        if (!JS_IsException(json)) {
            const char *s = JS_ToCString(ctx, json);
            snprintf(buf, sizeof(buf), "\033[0m%s\033[0m", s ? s : "{}");
            JS_FreeCString(ctx, s);
            JS_FreeValue(ctx, json);
            return strdup(buf);
        }
        JS_FreeValue(ctx, json);
        return strdup("\033[0m[Object]\033[0m");
    }

    JSValue str = JS_ToString(ctx, val);
    const char *s = JS_ToCString(ctx, str);
    snprintf(buf, sizeof(buf), "%s", s ? s : "");
    JS_FreeCString(ctx, s);
    JS_FreeValue(ctx, str);
    return strdup(buf);
}

char *engine_eval_repl(SofuuEngine *eng, const char *source, int *is_error) {
    if (is_error) *is_error = 0;
    JSValue result = JS_Eval(eng->ctx, source, strlen(source),
                             "<repl>", JS_EVAL_TYPE_GLOBAL);
    /* Drain microtasks */
    JSContext *ctx1;
    for (;;) {
        int e = JS_ExecutePendingJob(eng->rt, &ctx1);
        if (e <= 0) break;
    }
    if (JS_IsException(result)) {
        if (is_error) *is_error = 1;
        JSValue exc = JS_GetException(eng->ctx);
        JSValue estr = JS_ToString(eng->ctx, exc);
        const char *msg = JS_ToCString(eng->ctx, estr);
        char buf[4096];
        snprintf(buf, sizeof(buf), "\033[31m%s\033[0m", msg ? msg : "Unknown error");
        JSValue stack = JS_GetPropertyStr(eng->ctx, exc, "stack");
        if (!JS_IsUndefined(stack)) {
            const char *st = JS_ToCString(eng->ctx, stack);
            if (st) {
                size_t bl = strlen(buf);
                snprintf(buf + bl, sizeof(buf) - bl, "\n\033[90m%s\033[0m", st);
                JS_FreeCString(eng->ctx, st);
            }
        }
        JS_FreeValue(eng->ctx, stack);
        if (msg) JS_FreeCString(eng->ctx, msg);
        JS_FreeValue(eng->ctx, estr);
        JS_FreeValue(eng->ctx, exc);
        JS_FreeValue(eng->ctx, result);
        return strdup(buf);
    }
    char *display = repl_inspect(eng->ctx, result);
    JS_FreeValue(eng->ctx, result);
    return display;
}

void engine_run_jobs(SofuuEngine *eng) {
    JSContext *ctx1;
    int err;
    for (;;) {
        err = JS_ExecutePendingJob(eng->rt, &ctx1);
        if (err <= 0) {
            if (err < 0) dump_error(ctx1);
            break;
        }
    }
}


void engine_destroy(SofuuEngine *eng) {
    if (!eng) return;
    sofuu_loop_close();
    mod_process_cleanup(eng->ctx);
    JS_RunGC(eng->rt);
    JS_FreeContext(eng->ctx);
    JS_FreeRuntime(eng->rt);
    free(eng);
}
