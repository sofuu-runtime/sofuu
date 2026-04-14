/*
 * engine.h — QuickJS engine layer for Sofuu
 */
#ifndef SOFUU_ENGINE_H
#define SOFUU_ENGINE_H

#include "quickjs.h"

typedef struct {
    JSRuntime *rt;
    JSContext *ctx;
} SofuuEngine;

/* Create and initialize the JS engine. */
SofuuEngine *engine_create(void);

/* Register all built-in native modules (console, process, etc.) */
void engine_register_builtins(SofuuEngine *eng);

/* Load and execute a JS file. Returns 0 on success. */
int engine_eval_file(SofuuEngine *eng, const char *path);

/* Evaluate a JS string. Returns 0 on success. */
int engine_eval_string(SofuuEngine *eng, const char *source, const char *filename);

/*
 * REPL eval: evaluates source, returns a malloc'd string with the pretty-printed
 * result (or NULL on undefined). Sets *is_error=1 if an exception occurred.
 * Caller must free() the returned string.
 */
char *engine_eval_repl(SofuuEngine *eng, const char *source, int *is_error);

/* Pump the microtask/job queue until empty. */
void engine_run_jobs(SofuuEngine *eng);

/* Destroy the engine and free resources. */
void engine_destroy(SofuuEngine *eng);

/* Helper: read a whole file into a heap-allocated string. Caller must free(). */
char *engine_read_file(const char *path, size_t *out_len);

/* Module loader function registered with QuickJS */
JSModuleDef *sofuu_module_loader(JSContext *ctx, const char *module_name, void *opaque);

#endif /* SOFUU_ENGINE_H */
