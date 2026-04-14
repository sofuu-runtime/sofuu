/*
 * timer.c — setTimeout, setInterval, clearTimeout, clearInterval
 *
 * Each timer creates a uv_timer_t. The callback:
 *   1. Fires the JS function (or resolves a "sleep" promise)
 *   2. For one-shot (setTimeout): stops and frees the timer
 *   3. For repeating (setInterval): keeps the timer alive
 */
#include "timer.h"
#include "promises.h"
#include "loop.h"
#include <stdlib.h>
#include <string.h>

/* Per-timer context */
typedef struct {
    JSContext *ctx;
    JSValue    callback;    /* JS function to call */
    int        repeat;      /* 1 = setInterval, 0 = setTimeout */
    int        closing;     /* 1 = clearInterval called, deferred close pending */
    uint32_t   id;          /* numeric ID for clearTimeout/clearInterval */
    uv_timer_t handle;
} sofuu_timer_t;

/* Global ID counter + a simple linked list for clearTimeout lookup */
static uint32_t g_next_id = 1;

/* Registry: array of active timers (simple linear scan is fine for Phase 2) */
#define MAX_TIMERS 1024
static sofuu_timer_t *g_timers[MAX_TIMERS];
static int g_timer_count = 0;

static void timer_register(sofuu_timer_t *t) {
    if (g_timer_count < MAX_TIMERS)
        g_timers[g_timer_count++] = t;
}

static void timer_unregister(sofuu_timer_t *t) {
    for (int i = 0; i < g_timer_count; i++) {
        if (g_timers[i] == t) {
            g_timers[i] = g_timers[--g_timer_count];
            return;
        }
    }
}

static sofuu_timer_t *timer_find(uint32_t id) {
    for (int i = 0; i < g_timer_count; i++)
        if (g_timers[i]->id == id) return g_timers[i];
    return NULL;
}

/* Close callback: free the sofuu_timer_t after libuv releases the handle */
static void on_timer_close(uv_handle_t *handle) {
    sofuu_timer_t *t = (sofuu_timer_t *)handle->data;
    JS_FreeValue(t->ctx, t->callback);
    free(t);
}

/* libuv timer callback */
static void on_timer(uv_timer_t *handle) {
    sofuu_timer_t *t = (sofuu_timer_t *)handle->data;
    JSContext *ctx = t->ctx;

    /* If clearInterval was called before this tick, skip the callback */
    if (t->closing) return;

    /* Call the JS callback with no arguments */
    JSValue ret = JS_Call(ctx, t->callback, JS_UNDEFINED, 0, NULL);
    if (JS_IsException(ret)) {
        JSValue exc = JS_GetException(ctx);
        JSValue str = JS_ToString(ctx, exc);
        const char *s = JS_ToCString(ctx, str);
        fprintf(stderr, "[timer] Uncaught: %s\n", s ? s : "?");
        if (s) JS_FreeCString(ctx, s);
        JS_FreeValue(ctx, str);
        JS_FreeValue(ctx, exc);
    }
    JS_FreeValue(ctx, ret);
    sofuu_flush_jobs(ctx);

    /* Check if the callback called clearInterval on THIS timer */
    if (t->closing) return;  /* already queued for close inside the callback */

    /* One-shot: schedule close — free happens in on_timer_close */
    if (!t->repeat) {
        uv_timer_stop(&t->handle);
        timer_unregister(t);
        uv_close((uv_handle_t *)&t->handle, on_timer_close);
    }
}

static uint32_t sofuu_timer_create(JSContext *ctx, JSValue cb, uint64_t delay_ms, int repeat) {
    sofuu_timer_t *t = calloc(1, sizeof(sofuu_timer_t));
    t->ctx      = ctx;
    t->callback = JS_DupValue(ctx, cb);
    t->repeat   = repeat;
    t->id       = g_next_id++;
    t->handle.data = t;

    uv_timer_init(sofuu_loop_get(), &t->handle);
    uv_timer_start(&t->handle, on_timer,
                   delay_ms,
                   repeat ? delay_ms : 0);   /* repeat=0 means one-shot */

    timer_register(t);
    return t->id;
}

/* ------------------------------------------------------------------ */
/* setTimeout(fn, ms)                                                  */
/* ------------------------------------------------------------------ */
static JSValue js_setTimeout(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1 || !JS_IsFunction(ctx, argv[0]))
        return JS_ThrowTypeError(ctx, "setTimeout: first argument must be a function");

    uint32_t delay = 0;
    if (argc >= 2) JS_ToUint32(ctx, &delay, argv[1]);

    uint32_t id = sofuu_timer_create(ctx, argv[0], delay, 0);
    return JS_NewUint32(ctx, id);
}

/* ------------------------------------------------------------------ */
/* setInterval(fn, ms)                                                 */
/* ------------------------------------------------------------------ */
static JSValue js_setInterval(JSContext *ctx, JSValueConst this_val,
                               int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1 || !JS_IsFunction(ctx, argv[0]))
        return JS_ThrowTypeError(ctx, "setInterval: first argument must be a function");

    uint32_t delay = 0;
    if (argc >= 2) JS_ToUint32(ctx, &delay, argv[1]);

    uint32_t id = sofuu_timer_create(ctx, argv[0], delay, 1);
    return JS_NewUint32(ctx, id);
}

/* ------------------------------------------------------------------ */
/* clearTimeout / clearInterval                                        */
/* ------------------------------------------------------------------ */
static JSValue js_clearTimer(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1) return JS_UNDEFINED;

    uint32_t id = 0;
    JS_ToUint32(ctx, &id, argv[0]);

    sofuu_timer_t *t = timer_find(id);
    if (t && !t->closing) {
        t->closing = 1;              /* mark: do not re-enter callback */
        uv_timer_stop(&t->handle);
        timer_unregister(t);
        /* Safe close: libuv calls on_timer_close when handle is fully done */
        uv_close((uv_handle_t *)&t->handle, on_timer_close);
    }
    return JS_UNDEFINED;
}

/* ------------------------------------------------------------------ */
/* sofuu.sleep(ms) — Promise-based delay                               */
/* ------------------------------------------------------------------ */
typedef struct {
    sofuu_promise_t *promise;
    uv_timer_t       handle;
} sleep_req_t;

static void on_sleep_close(uv_handle_t *handle) {
    /* Timer handle fully closed — safe to free the container */
    sleep_req_t *req = (sleep_req_t *)handle->data;
    free(req);
}

static void on_sleep_timer(uv_timer_t *handle) {
    sleep_req_t     *req = (sleep_req_t *)handle->data;
    sofuu_promise_t *p   = req->promise;
    JSContext       *ctx = p->ctx;  /* save before resolve frees p */

    uv_timer_stop(handle);
    sofuu_promise_resolve(p, JS_UNDEFINED);  /* frees p */
    sofuu_flush_jobs(ctx);

    /* Schedule handle close — frees req in on_sleep_close */
    uv_close((uv_handle_t *)handle, on_sleep_close);
}

static JSValue js_sleep(JSContext *ctx, JSValueConst this_val,
                         int argc, JSValueConst *argv) {
    (void)this_val;
    uint32_t ms = 0;
    if (argc >= 1) JS_ToUint32(ctx, &ms, argv[0]);

    sofuu_promise_t *p;
    JSValue promise = sofuu_promise_new(ctx, &p);

    sleep_req_t *req = calloc(1, sizeof(sleep_req_t));
    req->promise     = p;
    req->handle.data = req;

    uv_timer_init(sofuu_loop_get(), &req->handle);
    uv_timer_start(&req->handle, on_sleep_timer, ms, 0);

    return promise;
}

/* ------------------------------------------------------------------ */
/* Registration                                                         */
/* ------------------------------------------------------------------ */
void mod_timer_register(JSContext *ctx) {
    JSValue global = JS_GetGlobalObject(ctx);

    JS_SetPropertyStr(ctx, global, "setTimeout",
        JS_NewCFunction(ctx, js_setTimeout, "setTimeout", 2));
    JS_SetPropertyStr(ctx, global, "setInterval",
        JS_NewCFunction(ctx, js_setInterval, "setInterval", 2));
    JS_SetPropertyStr(ctx, global, "clearTimeout",
        JS_NewCFunction(ctx, js_clearTimer, "clearTimeout", 1));
    JS_SetPropertyStr(ctx, global, "clearInterval",
        JS_NewCFunction(ctx, js_clearTimer, "clearInterval", 1));

    /* sofuu.sleep(ms) — bonus utility */
    JSValue sofuu_obj = JS_GetPropertyStr(ctx, global, "sofuu");
    if (JS_IsUndefined(sofuu_obj)) {
        sofuu_obj = JS_NewObject(ctx);
    }
    JS_SetPropertyStr(ctx, sofuu_obj, "sleep",
        JS_NewCFunction(ctx, js_sleep, "sleep", 1));
    JS_SetPropertyStr(ctx, global, "sofuu", sofuu_obj);

    JS_FreeValue(ctx, global);
}
