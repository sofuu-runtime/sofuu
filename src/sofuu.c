/*
 * sofuu.c — Sofuu Runtime public API implementation
 */
#include "sofuu.h"
#include "engine.h"
#include "mod_process.h"
#include <stdlib.h>

struct SofuuRuntime {
    SofuuEngine *engine;
};

SofuuRuntime *sofuu_init(void) {
    SofuuRuntime *rt = calloc(1, sizeof(SofuuRuntime));
    if (!rt) return NULL;

    rt->engine = engine_create();
    if (!rt->engine) {
        free(rt);
        return NULL;
    }

    engine_register_builtins(rt->engine);
    return rt;
}

int sofuu_eval_file(SofuuRuntime *rt, const char *path) {
    return engine_eval_file(rt->engine, path);
}

int sofuu_eval_string(SofuuRuntime *rt, const char *source, const char *filename) {
    return engine_eval_string(rt->engine, source, filename);
}

void sofuu_run_jobs(SofuuRuntime *rt) {
    engine_run_jobs(rt->engine);
}

void sofuu_destroy(SofuuRuntime *rt) {
    if (!rt) return;
    engine_destroy(rt->engine);
    free(rt);
}

void *sofuu_get_engine(SofuuRuntime *rt) {
    return rt ? (void *)rt->engine : NULL;
}
