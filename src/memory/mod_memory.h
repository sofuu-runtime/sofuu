#ifndef SOFUU_MOD_MEMORY_H
#define SOFUU_MOD_MEMORY_H

#include "quickjs.h"
#include <uv.h>
#include "formats.h"
#include "hnsw.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char          mind_path[512];
    
    /* Massive flat float array of all embeddings across all memories */
    float        *vec_store;
    size_t        vec_capacity;
    size_t        n_memories;
    size_t        vec_dim;       /* e.g., 768 for nomic-embed-text */
    
    /* JSON metadata */
    char         *meta_json;
    
    /* Parallel array matching vec_store index */
    cma_record_t *records;
    size_t        record_capacity;
    
    /* Fast vector search index */
    hnsw_t       *index;
    
    /* Background operations */
    uv_idle_t     consolidation_idle;
    JSContext    *ctx;
    
    /* QJS Native Object reference */
    JSValue       obj;
} cma_t;

/* Standard module registration hooked in engine.c */
void mod_memory_register(JSContext *ctx);

/* 
 * CMA low level C APIs that the JS wrapper will call
 */

cma_t *cma_open(JSContext *ctx, const char *path, size_t vec_dim);

int cma_remember(cma_t *cma, const float *vec, const char *text, const char *role, uint32_t kv_page_id);

typedef struct {
    uint32_t *kv_page_ids;
    size_t n;
} cma_kv_hints_t;

cma_kv_hints_t cma_kv_hints(cma_t *cma, const float *query_vec, size_t n_hints);

int cma_flush(cma_t *cma);

void cma_close(cma_t *cma);

int cma_forget(cma_t *cma, uint32_t vector_index);

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_MOD_MEMORY_H */
