/*
 * loop.c — libuv event loop + QuickJS microtask interleaving
 *
 * The key insight: after every libuv I/O callback we must pump
 * QuickJS pending jobs (Promise .then chains, async/await continuations).
 * We do this by running the loop in UV_RUN_ONCE mode in a tight loop
 * until both the libuv loop AND the JS job queue are both drained.
 */
#include "loop.h"
#include "promises.h"
#include <stdlib.h>
#include <stdio.h>

static uv_loop_t g_loop;

void sofuu_loop_init(void) {
    uv_loop_init(&g_loop);
}

uv_loop_t *sofuu_loop_get(void) {
    return &g_loop;
}

/*
 * sofuu_loop_run:
 *
 * Runs until both are true:
 *   1. No pending libuv handles/requests (UV_RUN_NOWAIT returns 0)
 *   2. No pending QuickJS jobs
 */
void sofuu_loop_run(JSContext *ctx) {
    int has_pending;
    do {
        /* First: pump any immediately-available JS microtasks */
        sofuu_flush_jobs(ctx);

        /* Then: run one round of libuv I/O (blocking with timeout) */
        has_pending = uv_run(&g_loop, UV_RUN_ONCE);

        /* After libuv fires callbacks, flush JS microtasks again */
        sofuu_flush_jobs(ctx);

    } while (has_pending || uv_loop_alive(&g_loop));

    /* Final flush after the loop fully drains */
    sofuu_flush_jobs(ctx);
}

/* Walk callback: closes any handle that is still active */
static void walk_close_cb(uv_handle_t *handle, void *arg) {
    (void)arg;
    if (!uv_is_closing(handle)) {
        uv_close(handle, NULL);
    }
}

/*
 * sofuu_loop_close:
 *
 * Drains all remaining handles before closing the loop.
 * This prevents the GC assertion by ensuring no I/O callbacks
 * can fire after JS_FreeContext().
 */
void sofuu_loop_close(void) {
    /* Walk all active handles and close them */
    uv_walk(&g_loop, walk_close_cb, NULL);

    /* Run the loop briefly to let close callbacks fire */
    uv_run(&g_loop, UV_RUN_DEFAULT);

    uv_loop_close(&g_loop);
}
