/*
 * promises.h — QuickJS Promise ↔ libuv async bridge
 *
 * PATTERN: Every async libuv operation follows this flow:
 *   1. JS calls a native function → creates a Promise via sofuu_promise_new()
 *   2. libuv starts the async op, receives sofuu_promise_t* as req->data
 *   3. libuv callback fires on completion → calls sofuu_promise_resolve/reject()
 *   4. sofuu_loop_flush_jobs() pumps the QuickJS microtask queue
 */
#ifndef SOFUU_PROMISES_H
#define SOFUU_PROMISES_H

#include "quickjs.h"

/* Opaque promise handle — passed as libuv req->data */
typedef struct sofuu_promise_t {
    JSContext *ctx;
    JSValue resolve;
    JSValue reject;
} sofuu_promise_t;

/*
 * Create a new pending Promise and return its JS value.
 * *out receives the sofuu_promise_t* to store in req->data.
 * Caller must NOT free *out — sofuu_promise_resolve/reject does that.
 */
JSValue sofuu_promise_new(JSContext *ctx, sofuu_promise_t **out);

/* Resolve the promise with a JS value. Frees the promise handle. */
void sofuu_promise_resolve(sofuu_promise_t *p, JSValue value);

/* Reject the promise with a JS Error. Frees the promise handle. */
void sofuu_promise_reject(sofuu_promise_t *p, JSValue error);

/* Convenience: reject with a plain C string message */
void sofuu_promise_reject_str(sofuu_promise_t *p, const char *msg);

/* Pump all pending QuickJS microtasks/jobs — call after each libuv callback */
void sofuu_flush_jobs(JSContext *ctx);

#endif /* SOFUU_PROMISES_H */
