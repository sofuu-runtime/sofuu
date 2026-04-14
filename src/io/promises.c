/*
 * promises.c — QuickJS Promise ↔ libuv async bridge implementation
 */
#include "promises.h"
#include <stdlib.h>
#include <stdio.h>

JSValue sofuu_promise_new(JSContext *ctx, sofuu_promise_t **out) {
    JSValue resolvers[2];
    JSValue promise = JS_NewPromiseCapability(ctx, resolvers);

    sofuu_promise_t *p = malloc(sizeof(sofuu_promise_t));
    p->ctx     = ctx;
    p->resolve = resolvers[0];
    p->reject  = resolvers[1];

    *out = p;
    return promise;
}

void sofuu_promise_resolve(sofuu_promise_t *p, JSValue value) {
    JSValue ret = JS_Call(p->ctx, p->resolve, JS_UNDEFINED, 1, &value);
    if (JS_IsException(ret)) {
        /* Log but don't crash — the promise was already settled or handler threw */
        JSValue exc = JS_GetException(p->ctx);
        JSValue msg = JS_ToString(p->ctx, exc);
        const char *s = JS_ToCString(p->ctx, msg);
        fprintf(stderr, "[sofuu] Promise resolve error: %s\n", s ? s : "?");
        if (s) JS_FreeCString(p->ctx, s);
        JS_FreeValue(p->ctx, msg);
        JS_FreeValue(p->ctx, exc);
    }
    JS_FreeValue(p->ctx, ret);
    JS_FreeValue(p->ctx, p->resolve);
    JS_FreeValue(p->ctx, p->reject);
    free(p);
}

void sofuu_promise_reject(sofuu_promise_t *p, JSValue error) {
    JSValue ret = JS_Call(p->ctx, p->reject, JS_UNDEFINED, 1, &error);
    JS_FreeValue(p->ctx, ret);
    JS_FreeValue(p->ctx, error);
    JS_FreeValue(p->ctx, p->resolve);
    JS_FreeValue(p->ctx, p->reject);
    free(p);
}

void sofuu_promise_reject_str(sofuu_promise_t *p, const char *msg) {
    JSValue err = JS_NewError(p->ctx);
    JS_SetPropertyStr(p->ctx, err, "message", JS_NewString(p->ctx, msg));
    sofuu_promise_reject(p, err);
}

void sofuu_flush_jobs(JSContext *ctx) {
    JSRuntime *rt = JS_GetRuntime(ctx);
    JSContext *ctx2;
    int err;
    for (;;) {
        err = JS_ExecutePendingJob(rt, &ctx2);
        if (err <= 0) break;
    }
}
