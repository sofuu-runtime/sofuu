/*
 * sofuu.h — Public API for the Sofuu JS Runtime
 * (素風) — Simple/Pure Wind
 */
#ifndef SOFUU_H
#define SOFUU_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque runtime handle */
typedef struct SofuuRuntime SofuuRuntime;

/* Initialize a new runtime. Returns NULL on failure. */
SofuuRuntime *sofuu_init(void);

/* Execute a JavaScript or TypeScript file by path.
 * Returns 0 on success, non-zero on error. */
int sofuu_eval_file(SofuuRuntime *rt, const char *path);

/* Execute a JS string snippet (useful for REPL).
 * Returns 0 on success. */
int sofuu_eval_string(SofuuRuntime *rt, const char *source, const char *filename);

/* Pump pending jobs (promises/microtasks) until queue is empty. */
void sofuu_run_jobs(SofuuRuntime *rt);

/* Access the underlying engine (internal use only — cast to SofuuEngine*) */
void *sofuu_get_engine(SofuuRuntime *rt);

/* Tear down the runtime and free all resources. */
void sofuu_destroy(SofuuRuntime *rt);

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_H */
