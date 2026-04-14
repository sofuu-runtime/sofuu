#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "quickjs.h"
#include "mod_agent.h"

/* QJS Class ID for Agent handle */
static JSClassID agent_class_id;

extern JSClassID cma_class_id;
extern JSClassID kv_class_id;
                               
static void agent_finalizer(JSRuntime *rt, JSValue val) {
    agent_t *agent = (agent_t *)JS_GetOpaque(val, agent_class_id);
    if (agent) {
        agent_destroy(agent);
    }
}

static JSClassDef agent_class = {
    "Agent",
    .finalizer = agent_finalizer,
};

agent_t *agent_create(JSContext *ctx, const char *name, cma_t *cma, kv_store_t *kv) {
    agent_t *agent = (agent_t *)calloc(1, sizeof(agent_t));
    if (!agent) return NULL;
    
    strncpy(agent->name, name, sizeof(agent->name) - 1);
    agent->cma = cma;
    agent->kv  = kv;
    agent->ctx = ctx;
    
    return agent;
}

int agent_prefetch_context(agent_t *agent, const float *query_vec, size_t n_memories) {
    if (!agent || !agent->cma || !agent->kv || !agent->cma->index) return -1;
    
    /* 1. Recall top episodic / semantic memories from CMA */
    hnsw_result_t *results = (hnsw_result_t *)malloc(n_memories * sizeof(hnsw_result_t));
    uint32_t out_count = 0;
    
    /* ef search parameter should be at least as large as the requested memories */
    uint32_t ef = n_memories * 2;
    if (ef < 16) ef = 16;
    
    hnsw_search(agent->cma->index, query_vec, ef, results, &out_count);
    
    if (out_count == 0) {
        free(results);
        return 0;
    }
    
    uint32_t prefetched = 0;
    for (size_t i = 0; i < n_memories && i < out_count; i++) {
        uint32_t mem_idx = results[i].id;
        if (mem_idx < agent->cma->n_memories) {
            uint32_t kv_page = agent->cma->records[mem_idx].kv_page_id;
            if (kv_page > 0) {
                /* 2. Prefetch the KV container by bringing it into the active RAM LRU cache */
                active_kv_page_t *page = kv_get_page(agent->kv, kv_page);
                if (page) {
                    prefetched++;
                }
            }
        }
    }
    
    free(results);
    return prefetched;
}

void agent_destroy(agent_t *agent) {
    if (!agent) return;
    free(agent);
}

/* ──────────────────────────────────────────────────────────────────
 * JS Bindings
 * ────────────────────────────────────────────────────────────────── */

/* Workaround for reading opaque from another class without exposing class ID universally.
 * We'll just define JS wrapper that fetches internal structs...
 * BUT we can also just rely on pure JS encapsulation. We'll do it natively for speed.
 * Actually, passing JS instances of CMA/KV might be tricky if we don't have their class IDs exported.
 * Instead, let's keep references to them in JS land, and when we create an agent, we pass the raw pointers if possible.
 */

static JSValue js_agent_prefetch(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 2) return JS_ThrowTypeError(ctx, "Expected query(Float32Array), topK");
    
    agent_t *agent = (agent_t *)JS_GetOpaque2(ctx, this_val, agent_class_id);
    if (!agent) return JS_EXCEPTION;
    
    size_t q_len=0, q_off=0;
    JSValue abQ = JS_GetTypedArrayBuffer(ctx, argv[0], &q_off, &q_len, NULL);
    if (JS_IsException(abQ)) return JS_ThrowTypeError(ctx, "Query must be Float32Array");
    size_t asz; uint8_t *b = JS_GetArrayBuffer(ctx, &asz, abQ); JS_FreeValue(ctx, abQ);
    const float *query = (const float *)(b + q_off);
    
    uint32_t topK = 0;
    JS_ToUint32(ctx, &topK, argv[1]);
    
    int n = agent_prefetch_context(agent, query, topK);
    return JS_NewInt32(ctx, n);
}

/* To avoid exporting class IDs, we will handle object construction without enforcing C-level type safety on dependencies, 
   or we just do a dirty cast. Since QuickJS uses standard class IDs, we can just grab Opaque blindly and trust JS layer.
   BUT a better way is to attach the inner instances. Wait... the easiest way is to let JS build the `agent` abstraction, 
   since calling CMA.recall() and kv.get() in JS is only ~1ms overhead.
   However, we use C for performance. Let's do `agent_create` accepting JSObjects but we'll cheat to just read memory layout of generic opaques. */

static JSValue js_agent_create(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 3) return JS_ThrowTypeError(ctx, "Expected name, cma_obj, kv_obj");
    
    const char *name = JS_ToCString(ctx, argv[0]);
    
    /* VERY dirty cheat to extract opaques from any object. 
       In production, export the class_ids globally in an engine.h block. */
    cma_t *cma = (cma_t *)JS_GetOpaque(argv[1], cma_class_id);
    kv_store_t *kv = (kv_store_t *)JS_GetOpaque(argv[2], kv_class_id);
    
    agent_t *agent = agent_create(ctx, name, cma, kv);
    JS_FreeCString(ctx, name);
    
    if (!agent) return JS_ThrowInternalError(ctx, "Failed to initialize Agent");
    
    JSValue obj = JS_NewObjectClass(ctx, agent_class_id);
    JS_SetOpaque(obj, agent);
    
    /* Attach Cma/KV objects so they don't get GC'd */
    JSValue prop_cma = JS_DupValue(ctx, argv[1]);
    JSValue prop_kv  = JS_DupValue(ctx, argv[2]);
    JS_SetPropertyStr(ctx, obj, "memory", prop_cma);
    JS_SetPropertyStr(ctx, obj, "kv", prop_kv);
    
    return obj;
}

static const JSCFunctionListEntry agent_proto_funcs[] = {
    JS_CFUNC_DEF("prefetch", 2, js_agent_prefetch),
};

static const JSCFunctionListEntry agent_module_funcs[] = {
    JS_CFUNC_DEF("create", 3, js_agent_create),
};

#ifndef countof
#define countof(x) (sizeof(x) / sizeof((x)[0]))
#endif

void mod_agent_register(JSContext *ctx) {
    JS_NewClassID(&agent_class_id);
    JS_NewClass(JS_GetRuntime(ctx), agent_class_id, &agent_class);
    
    JSValue proto = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, proto, agent_proto_funcs, countof(agent_proto_funcs));
    JS_SetClassProto(ctx, agent_class_id, proto);
    
    JSValue global_obj = JS_GetGlobalObject(ctx);
    JSValue sofuu_obj = JS_GetPropertyStr(ctx, global_obj, "sofuu");
    if (JS_IsUndefined(sofuu_obj)) {
        sofuu_obj = JS_NewObject(ctx);
        JS_SetPropertyStr(ctx, global_obj, "sofuu", sofuu_obj);
        sofuu_obj = JS_GetPropertyStr(ctx, global_obj, "sofuu");
    }
    
    JSValue agent_mod = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, agent_mod, agent_module_funcs, countof(agent_module_funcs));
    JS_SetPropertyStr(ctx, sofuu_obj, "agent", agent_mod);
    
    JS_FreeValue(ctx, sofuu_obj);
    JS_FreeValue(ctx, global_obj);
}
