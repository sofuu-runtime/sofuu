#ifndef SOFUU_MOD_AGENT_H
#define SOFUU_MOD_AGENT_H

#include <stdint.h>
#include <stddef.h>
#include "quickjs.h"
#include "mod_memory.h"
#include "mod_kv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* An LLM-powered Agent that automatically orchestrates Memory & KV caches */
typedef struct {
    cma_t      *cma; /* Cognitive Memory Architecture reference */
    kv_store_t *kv;  /* SSD KV cache reference */
    
    JSContext  *ctx;
    JSValue    obj;
    
    char       name[64];
} agent_t;

/* Instantiate an agent tying together CMA and KV */
agent_t *agent_create(JSContext *ctx, const char *name, cma_t *cma, kv_store_t *kv);

/* 
 * Async KV prefetch operation.
 * Finds related pages based on semantic memory recall and pulls them into LRU cache.
 */
int agent_prefetch_context(agent_t *agent, const float *query_vec, size_t n_memories);

/* Destroy agent reference (doesn't destroy the underlying cma/kv unless refcount drops) */
void agent_destroy(agent_t *agent);

/* Register Sofuu.agent JS namespace */
void mod_agent_register(JSContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_MOD_AGENT_H */
