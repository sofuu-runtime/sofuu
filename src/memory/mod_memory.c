#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "quickjs.h"
#include "mod_memory.h"
#include "qtsq_adapter.h"
#include "simd.h"

#ifndef countof
#define countof(x) (sizeof(x) / sizeof((x)[0]))
#endif

/* QJS Class ID for CMA handle */
JSClassID cma_class_id;

static void cma_finalizer(JSRuntime *rt, JSValue val) {
    cma_t *cma = (cma_t *)JS_GetOpaque(val, cma_class_id);
    if (cma) {
        cma_close(cma);
    }
}

static JSClassDef cma_class = {
    "CMA",
    .finalizer = cma_finalizer,
};

static JSContext *get_ctx() {
    /* For simple memory macros when ctx isn't passed directly by QJS */
    return NULL; 
}

/* ──────────────────────────────────────────────────────────────────
 * C Internal Implementation
 * ────────────────────────────────────────────────────────────────── */

cma_t *cma_open(JSContext *ctx, const char *path, size_t vec_dim) {
    cma_t *cma = (cma_t *)calloc(1, sizeof(cma_t));
    if (!cma) return NULL;
    
    strncpy(cma->mind_path, path, sizeof(cma->mind_path) - 1);
    cma->vec_dim = vec_dim;
    cma->ctx = ctx;
    
    float *loaded_vecs = NULL;
    size_t loaded_n = 0;
    size_t loaded_dim = 0;
    char *loaded_meta = NULL;
    
    int r = cma_qtsq_load(path, &loaded_vecs, &loaded_n, &loaded_dim, &loaded_meta);
    if (r == QTSQ_OK && loaded_vecs && loaded_n > 0) {
        cma->n_memories = loaded_n;
        cma->vec_capacity = loaded_n + 1024;
        cma->vec_store = (float *)malloc(cma->vec_capacity * vec_dim * sizeof(float));
        memcpy(cma->vec_store, loaded_vecs, loaded_n * vec_dim * sizeof(float));
        free(loaded_vecs);
        
        cma->record_capacity = cma->vec_capacity;
        cma->records = (cma_record_t *)calloc(cma->record_capacity, sizeof(cma_record_t));
        
        /* Reconstruct metadata from JSON using QuickJS */
        if (loaded_meta) {
            JSValue meta_obj = JS_ParseJSON(ctx, loaded_meta, strlen(loaded_meta), "metadata.json");
            if (!JS_IsException(meta_obj)) {
                JSValue recs_arr = JS_GetPropertyStr(ctx, meta_obj, "records");
                if (JS_IsArray(ctx, recs_arr)) {
                    uint32_t arr_len = 0;
                    JSValue len_val = JS_GetPropertyStr(ctx, recs_arr, "length");
                    JS_ToUint32(ctx, &arr_len, len_val);
                    JS_FreeValue(ctx, len_val);
                    
                    for (uint32_t i = 0; i < arr_len && i < cma->n_memories; i++) {
                        JSValue item = JS_GetPropertyUint32(ctx, recs_arr, i);
                        cma_record_t *rec = &cma->records[i];
                        
                        JSValue v_idx = JS_GetPropertyStr(ctx, item, "vector_index");
                        JS_ToUint32(ctx, &rec->vector_index, v_idx);
                        JS_FreeValue(ctx, v_idx);
                        
                        JSValue v_kvid = JS_GetPropertyStr(ctx, item, "kv_page_id");
                        JS_ToUint32(ctx, &rec->kv_page_id, v_kvid);
                        JS_FreeValue(ctx, v_kvid);
                        
                        JSValue v_str = JS_GetPropertyStr(ctx, item, "strength");
                        double strength = 1.0;
                        JS_ToFloat64(ctx, &strength, v_str);
                        rec->strength = (float)strength;
                        JS_FreeValue(ctx, v_str);
                        
                        JSValue v_tier = JS_GetPropertyStr(ctx, item, "tier");
                        uint32_t tier = 0;
                        JS_ToUint32(ctx, &tier, v_tier);
                        rec->tier = (uint8_t)tier;
                        JS_FreeValue(ctx, v_tier);
                        
                        JSValue v_text = JS_GetPropertyStr(ctx, item, "text");
                        const char *txt = JS_ToCString(ctx, v_text);
                        if (txt) {
                            rec->text = strdup(txt);
                            JS_FreeCString(ctx, txt);
                        }
                        JS_FreeValue(ctx, v_text);
                        
                        JSValue v_role = JS_GetPropertyStr(ctx, item, "role");
                        const char *role = JS_ToCString(ctx, v_role);
                        if (role) {
                            rec->role = strdup(role);
                            JS_FreeCString(ctx, role);
                        }
                        JS_FreeValue(ctx, v_role);
                        
                        JS_FreeValue(ctx, item);
                    }
                }
                JS_FreeValue(ctx, recs_arr);
            }
            JS_FreeValue(ctx, meta_obj);
            free(loaded_meta);
        }
    } else {
        /* New empty mind */
        cma->vec_capacity = 1024;
        cma->vec_store = (float *)malloc(cma->vec_capacity * vec_dim * sizeof(float));
        cma->record_capacity = 1024;
        cma->records = (cma_record_t *)calloc(cma->record_capacity, sizeof(cma_record_t));
        cma->n_memories = 0;
    }
    
    /* Build HNSW Index */
    cma->index = hnsw_create(vec_dim, cma->vec_capacity);
    hnsw_set_vec_store(cma->index, cma->vec_store);
    for (size_t i = 0; i < cma->n_memories; i++) {
        hnsw_add(cma->index, i);
    }
    
    return cma;
}

int cma_remember(cma_t *cma, const float *vec, const char *text, const char *role, uint32_t kv_page_id) {
    if (!cma || !vec || !text) return -1;
    
    if (cma->n_memories >= cma->vec_capacity) {
        cma->vec_capacity *= 2;
        cma->vec_store = (float *)realloc(cma->vec_store, cma->vec_capacity * cma->vec_dim * sizeof(float));
        cma->record_capacity *= 2;
        cma_record_t *new_recs = (cma_record_t *)realloc(cma->records, cma->record_capacity * sizeof(cma_record_t));
        memset(new_recs + cma->n_memories, 0, (cma->record_capacity - cma->n_memories) * sizeof(cma_record_t));
        cma->records = new_recs;
        hnsw_set_vec_store(cma->index, cma->vec_store);
    }
    
    uint32_t idx = cma->n_memories;
    memcpy(&cma->vec_store[idx * cma->vec_dim], vec, cma->vec_dim * sizeof(float));
    
    cma_record_t *rec = &cma->records[idx];
    rec->vector_index = idx;
    rec->kv_page_id = kv_page_id;
    rec->strength = 1.0f;
    rec->age_seconds = 0;
    rec->half_life = 86400; /* 1 day */
    rec->tier = CMA_TIER_EPISODIC;
    rec->text = strdup(text);
    rec->role = role ? strdup(role) : strdup("unknown");
    
    cma->n_memories++;
    
    /* Add to index immediately */
    hnsw_add(cma->index, idx);
    
    return idx;
}

int cma_flush(cma_t *cma) {
    if (!cma) return -1;
    
    /* Serialize records to JSON using QuickJS */
    JSValue root = JS_NewObject(cma->ctx);
    JSValue arr = JS_NewArray(cma->ctx);
    
    for (size_t i = 0; i < cma->n_memories; i++) {
        cma_record_t *rec = &cma->records[i];
        JSValue item = JS_NewObject(cma->ctx);
        JS_SetPropertyStr(cma->ctx, item, "vector_index", JS_NewUint32(cma->ctx, rec->vector_index));
        JS_SetPropertyStr(cma->ctx, item, "kv_page_id", JS_NewUint32(cma->ctx, rec->kv_page_id));
        JS_SetPropertyStr(cma->ctx, item, "strength", JS_NewFloat64(cma->ctx, rec->strength));
        JS_SetPropertyStr(cma->ctx, item, "tier", JS_NewUint32(cma->ctx, rec->tier));
        JS_SetPropertyStr(cma->ctx, item, "text", JS_NewString(cma->ctx, rec->text ? rec->text : ""));
        JS_SetPropertyStr(cma->ctx, item, "role", JS_NewString(cma->ctx, rec->role ? rec->role : ""));
        JS_SetPropertyUint32(cma->ctx, arr, (uint32_t)i, item);
    }
    JS_SetPropertyStr(cma->ctx, root, "records", arr);
    
    /* Stringify */
    JSValue str_val = JS_JSONStringify(cma->ctx, root, JS_UNDEFINED, JS_UNDEFINED);
    const char *json_c = JS_ToCString(cma->ctx, str_val);
    
    int r = cma_qtsq_save(cma->mind_path, cma->vec_store, cma->n_memories, cma->vec_dim, json_c);
    
    JS_FreeCString(cma->ctx, json_c);
    JS_FreeValue(cma->ctx, str_val);
    JS_FreeValue(cma->ctx, root);
    
    return (r == QTSQ_OK) ? 0 : -1;
}

cma_kv_hints_t cma_kv_hints(cma_t *cma, const float *query_vec, size_t n_hints) {
    cma_kv_hints_t result = {0};
    if (!cma || !query_vec || n_hints == 0) return result;
    
    hnsw_result_t *hnsw_res = (hnsw_result_t *)malloc(n_hints * sizeof(hnsw_result_t));
    uint32_t count = 0;
    hnsw_search(cma->index, query_vec, n_hints, hnsw_res, &count);
    
    uint32_t *page_ids = (uint32_t *)malloc(count * sizeof(uint32_t));
    uint32_t valid_hints = 0;
    
    for (uint32_t i = 0; i < count; i++) {
        uint32_t v_idx = hnsw_res[i].id;
        if (v_idx < cma->n_memories) {
            uint32_t pg = cma->records[v_idx].kv_page_id;
            if (pg > 0) {
                page_ids[valid_hints++] = pg;
                /* Reinforce memory on recall */
                cma->records[v_idx].strength = 1.0f;
                cma->records[v_idx].half_life *= 1.2f; /* decays slightly slower next time */
            }
        }
    }
    
    free(hnsw_res);
    result.kv_page_ids = page_ids;
    result.n = valid_hints;
    return result;
}

int cma_forget(cma_t *cma, uint32_t vector_index) {
    if (!cma || vector_index >= cma->n_memories) return -1;
    cma->records[vector_index].tier = CMA_TIER_FORGOTTEN;
    return 0;
}

void cma_close(cma_t *cma) {
    if (!cma) return;
    cma_flush(cma);
    
    if (cma->index) hnsw_free(cma->index);
    if (cma->vec_store) free(cma->vec_store);
    if (cma->records) {
        for (size_t i = 0; i < cma->n_memories; i++) {
            if (cma->records[i].text) free(cma->records[i].text);
            if (cma->records[i].role) free(cma->records[i].role);
        }
        free(cma->records);
    }
    free(cma);
}

/* ──────────────────────────────────────────────────────────────────
 * Decay and Consolidation (K-Means)
 * ────────────────────────────────────────────────────────────────── */

void cma_decay_tick(cma_t *cma, uint32_t dt_seconds) {
    if (!cma) return;
    for (size_t i = 0; i < cma->n_memories; i++) {
        cma_record_t *rec = &cma->records[i];
        if (rec->tier == CMA_TIER_FORGOTTEN) continue;
        
        rec->age_seconds += dt_seconds;
        /* Exponential decay: S = S0 * exp(-t / tau) */
        float decay_factor = expf(-((float)dt_seconds) / (float)rec->half_life);
        rec->strength *= decay_factor;
    }
}

int cma_consolidate(cma_t *cma) {
    if (!cma || cma->n_memories == 0) return 0;
    
    /* 1. Find weak episodic memories to cluster */
    uint32_t *weak_indices = (uint32_t *)malloc(cma->n_memories * sizeof(uint32_t));
    uint32_t n_weak = 0;
    for (size_t i = 0; i < cma->n_memories; i++) {
        cma_record_t *rec = &cma->records[i];
        if (rec->tier == CMA_TIER_EPISODIC && rec->strength < 0.25f && rec->age_seconds > 86400) {
            weak_indices[n_weak++] = (uint32_t)i;
        }
    }
    
    if (n_weak < 10) { /* Need at least 10 facts to form abstract clusters */
        free(weak_indices);
        return 0;
    }
    
    /* 2. Simple K-Means (K = n_weak / 10) */
    uint32_t K = n_weak / 10;
    if (K > 64) K = 64; /* cap max semantic clusters per pass */
    
    float *centroids = (float *)calloc(K, cma->vec_dim * sizeof(float));
    uint32_t *counts = (uint32_t *)calloc(K, sizeof(uint32_t));
    
    /* Init centroids randomly from the weak pool */
    for (uint32_t i = 0; i < K; i++) {
        uint32_t idx = weak_indices[rand() % n_weak];
        memcpy(&centroids[i * cma->vec_dim], &cma->vec_store[idx * cma->vec_dim], cma->vec_dim * sizeof(float));
    }
    
    /* 10 iterations of Lloyd's algorithm */
    for (int iter = 0; iter < 10; iter++) {
        memset(counts, 0, K * sizeof(uint32_t));
        float *new_centroids = (float *)calloc(K, cma->vec_dim * sizeof(float));
        
        for (uint32_t i = 0; i < n_weak; i++) {
            uint32_t v_idx = weak_indices[i];
            const float *vec = &cma->vec_store[v_idx * cma->vec_dim];
            
            float best_dist = -1.0f;
            uint32_t best_k = 0;
            for (uint32_t k = 0; k < K; k++) {
                float dist = sofuu_l2_f32(vec, &centroids[k * cma->vec_dim], cma->vec_dim);
                if (best_dist < 0 || dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            
            /* Add to new centroid */
            for (size_t d = 0; d < cma->vec_dim; d++) {
                new_centroids[best_k * cma->vec_dim + d] += vec[d];
            }
            counts[best_k]++;
        }
        
        /* Average and update */
        for (uint32_t k = 0; k < K; k++) {
            if (counts[k] > 0) {
                for (size_t d = 0; d < cma->vec_dim; d++) {
                    centroids[k * cma->vec_dim + d] = new_centroids[k * cma->vec_dim + d] / (float)counts[k];
                }
            }
        }
        free(new_centroids);
    }
    
    /* 3. Re-inject centroids as semantic tier facts */
    uint32_t consolidated = 0;
    for (uint32_t k = 0; k < K; k++) {
        if (counts[k] > 0) {
            char summary[64];
            snprintf(summary, sizeof(summary), "Consolidated Semantic Fact (%u episodes)", counts[k]);
            
            int new_idx = cma_remember(cma, &centroids[k * cma->vec_dim], summary, "system", 0);
            if (new_idx >= 0) {
                cma->records[new_idx].tier = CMA_TIER_SEMANTIC;
                cma->records[new_idx].half_life = 86400 * 30; /* Semantic facts last 1 month instead of 1 day */
                consolidated++;
            }
        }
    }
    
    /* 4. Mark originals as forgotten (they will be garbage collected eventually) */
    for (uint32_t i = 0; i < n_weak; i++) {
        cma->records[weak_indices[i]].tier = CMA_TIER_FORGOTTEN;
    }
    
    free(centroids);
    free(counts);
    free(weak_indices);
    
    return consolidated;
}

/* ──────────────────────────────────────────────────────────────────
 * JS Bindings
 * ────────────────────────────────────────────────────────────────── */

static JSValue js_cma_open(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 2) return JS_ThrowTypeError(ctx, "Expected path and vector dimension");
    
    const char *path = JS_ToCString(ctx, argv[0]);
    uint32_t dim = 0;
    JS_ToUint32(ctx, &dim, argv[1]);
    
    cma_t *cma = cma_open(ctx, path, dim);
    JS_FreeCString(ctx, path);
    if (!cma) return JS_ThrowInternalError(ctx, "Failed to initialize CMA");
    
    JSValue obj = JS_NewObjectClass(ctx, cma_class_id);
    JS_SetOpaque(obj, cma);
    cma->obj = obj;
    return obj;
}

static JSValue js_cma_remember(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 4) return JS_ThrowTypeError(ctx, "Expected vec(Float32Array), text, role, kv_page_id");
    
    cma_t *cma = (cma_t *)JS_GetOpaque2(ctx, this_val, cma_class_id);
    if (!cma) return JS_EXCEPTION;
    
    size_t byte_length = 0;
    size_t byte_offset = 0;
    JSValue obj = argv[0];
    JSValue ab = JS_GetTypedArrayBuffer(ctx, obj, &byte_offset, &byte_length, NULL);
    if (JS_IsException(ab)) return JS_ThrowTypeError(ctx, "Argument 1 must be Float32Array");
    size_t ab_size;
    uint8_t *buf = JS_GetArrayBuffer(ctx, &ab_size, ab);
    JS_FreeValue(ctx, ab);
    if (!buf) return JS_ThrowTypeError(ctx, "Argument 1 must be Float32Array");
    const float *vec = (const float *)(buf + byte_offset);
    if (byte_length / sizeof(float) != cma->vec_dim) {
        return JS_ThrowTypeError(ctx, "Vector dimension mismatch");
    }
    
    const char *text = JS_ToCString(ctx, argv[1]);
    const char *role = JS_ToCString(ctx, argv[2]);
    uint32_t page_id = 0;
    JS_ToUint32(ctx, &page_id, argv[3]);
    
    int idx = cma_remember(cma, vec, text, role, page_id);
    
    JS_FreeCString(ctx, text);
    JS_FreeCString(ctx, role);
    
    return JS_NewInt32(ctx, idx);
}

static JSValue js_cma_flush(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    cma_t *cma = (cma_t *)JS_GetOpaque2(ctx, this_val, cma_class_id);
    if (!cma) return JS_EXCEPTION;
    int r = cma_flush(cma);
    return JS_NewBool(ctx, r == 0);
}

static JSValue js_cma_kv_hints(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 2) return JS_ThrowTypeError(ctx, "Expected queryVec(Float32Array) and n_hints");
    
    cma_t *cma = (cma_t *)JS_GetOpaque2(ctx, this_val, cma_class_id);
    if (!cma) return JS_EXCEPTION;
    
    size_t byte_length = 0;
    size_t byte_offset = 0;
    JSValue ab = JS_GetTypedArrayBuffer(ctx, argv[0], &byte_offset, &byte_length, NULL);
    if (JS_IsException(ab)) return JS_ThrowTypeError(ctx, "Argument 1 must be Float32Array");
    size_t ab_size;
    uint8_t *buf = JS_GetArrayBuffer(ctx, &ab_size, ab);
    JS_FreeValue(ctx, ab);
    if (!buf) return JS_ThrowTypeError(ctx, "Argument 1 must be Float32Array");
    const float *vec = (const float *)(buf + byte_offset);
    
    uint32_t n_hints = 0;
    JS_ToUint32(ctx, &n_hints, argv[1]);
    
    cma_kv_hints_t hints = cma_kv_hints(cma, vec, n_hints);
    
    JSValue arr = JS_NewArray(ctx);
    for (size_t i = 0; i < hints.n; i++) {
        JS_SetPropertyUint32(ctx, arr, (uint32_t)i, JS_NewUint32(ctx, hints.kv_page_ids[i]));
    }
    
    if (hints.kv_page_ids) free(hints.kv_page_ids);
    
    return arr;
}

static JSValue js_cma_recall(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 2) return JS_ThrowTypeError(ctx, "Expected queryVec(Float32Array) and topK");
    
    cma_t *cma = (cma_t *)JS_GetOpaque2(ctx, this_val, cma_class_id);
    if (!cma) return JS_EXCEPTION;
    
    size_t byte_length = 0;
    size_t byte_offset = 0;
    JSValue ab = JS_GetTypedArrayBuffer(ctx, argv[0], &byte_offset, &byte_length, NULL);
    if (JS_IsException(ab)) return JS_ThrowTypeError(ctx, "Argument 1 must be Float32Array");
    size_t ab_size;
    uint8_t *buf = JS_GetArrayBuffer(ctx, &ab_size, ab);
    JS_FreeValue(ctx, ab);
    if (!buf) return JS_ThrowTypeError(ctx, "Argument 1 must be Float32Array");
    const float *vec = (const float *)(buf + byte_offset);
    
    uint32_t topK = 0;
    JS_ToUint32(ctx, &topK, argv[1]);
    
    hnsw_result_t *hnsw_res = (hnsw_result_t *)malloc(topK * sizeof(hnsw_result_t));
    uint32_t count = 0;
    hnsw_search(cma->index, vec, topK, hnsw_res, &count);
    
    JSValue arr = JS_NewArray(ctx);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t v_idx = hnsw_res[i].id;
        if (v_idx < cma->n_memories) {
            cma_record_t *rec = &cma->records[v_idx];
            if (rec->tier == CMA_TIER_FORGOTTEN) continue;
            
            JSValue item = JS_NewObject(ctx);
            JS_SetPropertyStr(ctx, item, "id", JS_NewUint32(ctx, v_idx));
            JS_SetPropertyStr(ctx, item, "distance", JS_NewFloat64(ctx, hnsw_res[i].distance));
            JS_SetPropertyStr(ctx, item, "role", JS_NewString(ctx, rec->role ? rec->role : ""));
            JS_SetPropertyStr(ctx, item, "text", JS_NewString(ctx, rec->text ? rec->text : ""));
            JS_SetPropertyStr(ctx, item, "tier", JS_NewUint32(ctx, rec->tier));
            JS_SetPropertyUint32(ctx, arr, i, item);
        }
    }
    free(hnsw_res);
    return arr;
}

static JSValue js_cma_decay_tick(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 1) return JS_ThrowTypeError(ctx, "Expected dt_seconds");
    cma_t *cma = (cma_t *)JS_GetOpaque2(ctx, this_val, cma_class_id);
    if (!cma) return JS_EXCEPTION;
    uint32_t dt = 0;
    JS_ToUint32(ctx, &dt, argv[0]);
    cma_decay_tick(cma, dt);
    return JS_UNDEFINED;
}

static JSValue js_cma_forget(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 1) return JS_ThrowTypeError(ctx, "Expected vector_index");
    cma_t *cma = (cma_t *)JS_GetOpaque2(ctx, this_val, cma_class_id);
    if (!cma) return JS_EXCEPTION;
    uint32_t v_idx = 0;
    JS_ToUint32(ctx, &v_idx, argv[0]);
    int r = cma_forget(cma, v_idx);
    return JS_NewBool(ctx, r == 0);
}

static JSValue js_cma_consolidate(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    cma_t *cma = (cma_t *)JS_GetOpaque2(ctx, this_val, cma_class_id);
    if (!cma) return JS_EXCEPTION;
    int n = cma_consolidate(cma);
    return JS_NewInt32(ctx, n);
}

static const JSCFunctionListEntry cma_proto_funcs[] = {
    JS_CFUNC_DEF("remember", 4, js_cma_remember),
    JS_CFUNC_DEF("recall", 2, js_cma_recall),
    JS_CFUNC_DEF("kvHints", 2, js_cma_kv_hints),
    JS_CFUNC_DEF("forget", 1, js_cma_forget),
    JS_CFUNC_DEF("flush", 0, js_cma_flush),
    JS_CFUNC_DEF("decayTick", 1, js_cma_decay_tick),
    JS_CFUNC_DEF("consolidate", 0, js_cma_consolidate),
};

static const JSCFunctionListEntry module_funcs[] = {
    JS_CFUNC_DEF("open", 2, js_cma_open),
};

void mod_memory_register(JSContext *ctx) {
    JS_NewClassID(&cma_class_id);
    JS_NewClass(JS_GetRuntime(ctx), cma_class_id, &cma_class);
    
    JSValue proto = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, proto, cma_proto_funcs, countof(cma_proto_funcs));
    JS_SetClassProto(ctx, cma_class_id, proto);
    
    JSValue global_obj = JS_GetGlobalObject(ctx);
    JSValue sofuu_obj = JS_GetPropertyStr(ctx, global_obj, "sofuu");
    
    /* If 'sofuu' object doesn't exist, create it (engine.c usually does this) */
    if (JS_IsUndefined(sofuu_obj)) {
        sofuu_obj = JS_NewObject(ctx);
        JS_SetPropertyStr(ctx, global_obj, "sofuu", sofuu_obj);
        sofuu_obj = JS_GetPropertyStr(ctx, global_obj, "sofuu");
    }
    
    JSValue mem_obj = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, mem_obj, module_funcs, countof(module_funcs));
    JS_SetPropertyStr(ctx, sofuu_obj, "memory", mem_obj);
    
    JS_FreeValue(ctx, sofuu_obj);
    JS_FreeValue(ctx, global_obj);
}
