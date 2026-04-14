/*
 * loop.h — libuv event loop lifecycle for Sofuu
 */
#ifndef SOFUU_LOOP_H
#define SOFUU_LOOP_H

#include "uv.h"
#include "quickjs.h"

/* Initialize the global libuv loop (call once at startup) */
void sofuu_loop_init(void);

/* Get the global loop — used everywhere we need uv_default_loop() */
uv_loop_t *sofuu_loop_get(void);

/*
 * Run the event loop until no more pending handles/timers.
 * This is the main "drain" call after JS evaluation — it processes
 * all pending I/O, timers, and Promise microtasks in interleaved fashion.
 */
void sofuu_loop_run(JSContext *ctx);

/* Tear down the loop */
void sofuu_loop_close(void);

#endif /* SOFUU_LOOP_H */
