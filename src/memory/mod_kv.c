#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>
#include "quickjs.h"
#include "mod_kv.h"
#include "qtsq_adapter.h"
#include "simd.h"

#ifndef countof
#define countof(x) (sizeof(x) / sizeof((x)[0]))
#endif

/* QJS Class ID for KV store handle */
JSClassID kv_class_id;

static void kv_finalizer(JSRuntime *rt, JSValue val) {
    kv_store_t *kv = (kv_store_t *)JS_GetOpaque(val, kv_class_id);
    if (kv) {
        kv_store_close(kv);
    }
}

static JSClassDef kv_class = {
    "KVStore",
    .finalizer = kv_finalizer,
};

/* ──────────────────────────────────────────────────────────────────
 * Helper: Make directory if not exists
 * ────────────────────────────────────────────────────────────────── */
static void ensure_dir(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        mkdir(path, 0755);
    }
}

/* ──────────────────────────────────────────────────────────────────
 * C Internal Implementation
 * ────────────────────────────────────────────────────────────────── */

kv_store_t *kv_store_open(JSContext *ctx, const char *path, const char *model_id,
                          size_t n_layers, size_t n_heads, size_t head_dim,
                          size_t max_size_gb, int k_prec, int v_prec, size_t max_ram_pages) {
    kv_store_t *kv = (kv_store_t *)calloc(1, sizeof(kv_store_t));
    if (!kv) return NULL;
    
    strncpy(kv->file_path, path, sizeof(kv->file_path) - 1);
    if (model_id) strncpy(kv->model_id, model_id, sizeof(kv->model_id) - 1);
    
    ensure_dir(path);
    
    kv->n_layers = n_layers;
    kv->n_heads = n_heads;
    kv->head_dim = head_dim;
    kv->max_size_gb = max_size_gb;
    kv->k_precision = k_prec;
    kv->v_precision = v_prec;
    kv->max_active = max_ram_pages > 0 ? max_ram_pages : 2; /* Extremely strict by default: hold at most 2 decompressed pages to protect RAM */
    
    kv->active_cache = (active_kv_page_t *)calloc(kv->max_active, sizeof(active_kv_page_t));
    kv->n_active = 0;
    kv->access_counter = 1;
    
    kv->ctx = ctx;
    
    /* Load index if it exists */
    char idx_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s/index.json", path);
    FILE *f = fopen(idx_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (len > 0) {
            char *json_buf = (char *)malloc(len + 1);
            if (fread(json_buf, 1, len, f) == len) {
                json_buf[len] = '\0';
                
                JSValue meta_obj = JS_ParseJSON(ctx, json_buf, len, "index.json");
                if (!JS_IsException(meta_obj)) {
                    JSValue pages_arr = JS_GetPropertyStr(ctx, meta_obj, "pages");
                    if (JS_IsArray(ctx, pages_arr)) {
                        uint32_t arr_len = 0;
                        JSValue len_val = JS_GetPropertyStr(ctx, pages_arr, "length");
                        JS_ToUint32(ctx, &arr_len, len_val);
                        JS_FreeValue(ctx, len_val);
                        
                        kv->page_capacity = arr_len + 256;
                        kv->pages = (kv_page_summary_t *)calloc(kv->page_capacity, sizeof(kv_page_summary_t));
                        
                        for (uint32_t i = 0; i < arr_len; i++) {
                            JSValue item = JS_GetPropertyUint32(ctx, pages_arr, i);
                            kv_page_summary_t *p = &kv->pages[kv->n_pages++];
                            
                            JSValue v_id = JS_GetPropertyStr(ctx, item, "page_id");
                            JS_ToUint32(ctx, &p->page_id, v_id);
                            JS_FreeValue(ctx, v_id);
                            
                            JSValue v_str = JS_GetPropertyStr(ctx, item, "strength");
                            double st = 1.0;
                            JS_ToFloat64(ctx, &st, v_str);
                            p->strength = (float)st;
                            JS_FreeValue(ctx, v_str);
                            
                            JSValue v_time = JS_GetPropertyStr(ctx, item, "created_at");
                            JS_ToUint32(ctx, &p->created_at, v_time);
                            JS_FreeValue(ctx, v_time);
                            
                            /* Assuming k_summary is stored as base64 or array. 
                             * Here we simplify. A robust implementation would parse Float32Array JSON. */
                            JS_FreeValue(ctx, item);
                        }
                    }
                    JS_FreeValue(ctx, pages_arr);
                }
                JS_FreeValue(ctx, meta_obj);
            }
            free(json_buf);
        }
        fclose(f);
    } else {
        kv->page_capacity = 256;
        kv->pages = (kv_page_summary_t *)calloc(kv->page_capacity, sizeof(kv_page_summary_t));
    }
    
    return kv;
}

static void evict_lru_page(kv_store_t *kv, uint32_t exclude_index) {
    if (kv->n_active == 0) return;
    
    uint32_t oldest_acc = 0xFFFFFFFF;
    int oldest_idx = -1;
    
    for (size_t i = 0; i < kv->n_active; i++) {
        if (i == exclude_index) continue;
        if (kv->active_cache[i].last_accessed < oldest_acc) {
            oldest_acc = kv->active_cache[i].last_accessed;
            oldest_idx = (int)i;
        }
    }
    
    if (oldest_idx >= 0) {
        if (kv->active_cache[oldest_idx].K) free(kv->active_cache[oldest_idx].K);
        if (kv->active_cache[oldest_idx].V) free(kv->active_cache[oldest_idx].V);
        kv->active_cache[oldest_idx].K = NULL;
        kv->active_cache[oldest_idx].V = NULL;
        
        /* Shift array */
        for (size_t i = oldest_idx; i < kv->n_active - 1; i++) {
            kv->active_cache[i] = kv->active_cache[i + 1];
        }
        kv->n_active--;
    }
}

uint32_t kv_page_save(kv_store_t *kv, const float *K, const float *V, 
                      size_t n_tokens, float strength) {
    if (!kv || !K || !V) return 0;
    
    uint32_t new_id = kv->n_pages + 1;
    char page_path[1024];
    
    /* Create a mini container just for this single KV page to keep file sizes small and GC easy */
    snprintf(page_path, sizeof(page_path), "%s/page_%08u.qtsq", kv->file_path, new_id);
    
    qtsq_context_t ctx;
    qtsq_init(&ctx);
    qtsq_container_create(&ctx);
    
    int r = kv_qtsq_save_page(&ctx, K, V, kv->n_layers, kv->n_heads, n_tokens, kv->head_dim, 
                              new_id, (qtsq_tensor_precision_t)kv->k_precision, (qtsq_tensor_precision_t)kv->v_precision);
    if (r == QTSQ_OK) {
        kv_qtsq_flush(&ctx, page_path);
    }
    qtsq_free(&ctx);
    
    if (r != QTSQ_OK) return 0;
    
    if (kv->n_pages >= kv->page_capacity) {
        kv->page_capacity *= 2;
        kv->pages = (kv_page_summary_t *)realloc(kv->pages, kv->page_capacity * sizeof(kv_page_summary_t));
    }
    
    kv_page_summary_t *p = &kv->pages[kv->n_pages++];
    p->page_id = new_id;
    p->strength = strength;
    p->created_at = (uint32_t)time(NULL);
    memset(p->k_summary, 0, KV_SUMMARY_DIM * sizeof(float)); /* simplified */
    
    /* Also load it into RAM cache immediately since it was just generated */
    if (kv->n_active >= kv->max_active) evict_lru_page(kv, -1);
    
    active_kv_page_t *ac = &kv->active_cache[kv->n_active++];
    ac->page_id = new_id;
    ac->n_layers = kv->n_layers;
    ac->n_tokens = n_tokens;
    ac->last_accessed = kv->access_counter++;
    
    size_t count = kv->n_layers * kv->n_heads * n_tokens * kv->head_dim;
    ac->K = (float *)malloc(count * sizeof(float));
    ac->V = (float *)malloc(count * sizeof(float));
    memcpy(ac->K, K, count * sizeof(float));
    memcpy(ac->V, V, count * sizeof(float));
    
    return new_id;
}

active_kv_page_t *kv_get_page(kv_store_t *kv, uint32_t page_id) {
    if (!kv || page_id == 0) return NULL;
    
    /* Check cache */
    for (size_t i = 0; i < kv->n_active; i++) {
        if (kv->active_cache[i].page_id == page_id) {
            kv->active_cache[i].last_accessed = kv->access_counter++;
            return &kv->active_cache[i];
        }
    }
    
    /* Load from SSD */
    char page_path[1024];
    snprintf(page_path, sizeof(page_path), "%s/page_%08u.qtsq", kv->file_path, page_id);
    
    float *K = NULL, *V = NULL;
    size_t l = 0, tok = 0;
    
    int r = kv_qtsq_load_page(page_path, page_id, &K, &V, &l, &tok);
    if (r != QTSQ_OK || !K || !V) return NULL; /* Error or page missing */
    
    if (kv->n_active >= kv->max_active) evict_lru_page(kv, -1);
    
    active_kv_page_t *ac = &kv->active_cache[kv->n_active++];
    ac->page_id = page_id;
    ac->K = K;
    ac->V = V;
    ac->n_layers = l;
    ac->n_tokens = tok;
    ac->last_accessed = kv->access_counter++;
    
    return ac;
}

kv_search_hints_t kv_search(kv_store_t *kv, const float *query_summary_64, size_t n_hints) {
    kv_search_hints_t res = {0};
    if (!kv || !query_summary_64 || n_hints == 0 || kv->n_pages == 0) return res;
    
    /* Naive linear scan over the small summary index. 
     * Since summaries are just 64 floats, we can scan thousands in <1ms. */
    typedef struct { uint32_t id; float sim; } match_t;
    match_t *matches = (match_t *)malloc(kv->n_pages * sizeof(match_t));
    
    for (size_t i = 0; i < kv->n_pages; i++) {
        kv_page_summary_t *p = &kv->pages[i];
        float sim = sofuu_cosine_f32(query_summary_64, p->k_summary, KV_SUMMARY_DIM);
        /* Apply decay/strength weighting */
        sim = sim * p->strength;
        matches[i].id = p->page_id;
        matches[i].sim = sim;
    }
    
    /* Sort matches, simplest way for small counts is qsort or insertion. Here we use insertion for top N */
    uint32_t *top_ids = (uint32_t *)malloc(n_hints * sizeof(uint32_t));
    size_t found = 0;
    
    for (size_t k = 0; k < n_hints && k < kv->n_pages; k++) {
        float best_sim = -2.0f;
        size_t best_idx = 0;
        for (size_t i = 0; i < kv->n_pages; i++) {
            if (matches[i].sim > best_sim) {
                best_sim = matches[i].sim;
                best_idx = i;
            }
        }
        if (best_sim == -2.0f) break;
        top_ids[found++] = matches[best_idx].id;
        matches[best_idx].sim = -3.0f; /* Exclude from next pass */
    }
    
    free(matches);
    res.page_ids = top_ids;
    res.n = found;
    return res;
}

int kv_flush(kv_store_t *kv) {
    if (!kv) return -1;
    
    char idx_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s/index.json", kv->file_path);
    
    JSValue root = JS_NewObject(kv->ctx);
    JSValue arr = JS_NewArray(kv->ctx);
    
    for (size_t i = 0; i < kv->n_pages; i++) {
        kv_page_summary_t *p = &kv->pages[i];
        JSValue item = JS_NewObject(kv->ctx);
        JS_SetPropertyStr(kv->ctx, item, "page_id", JS_NewUint32(kv->ctx, p->page_id));
        JS_SetPropertyStr(kv->ctx, item, "strength", JS_NewFloat64(kv->ctx, p->strength));
        JS_SetPropertyStr(kv->ctx, item, "created_at", JS_NewUint32(kv->ctx, p->created_at));
        JS_SetPropertyUint32(kv->ctx, arr, (uint32_t)i, item);
    }
    JS_SetPropertyStr(kv->ctx, root, "pages", arr);
    
    JSValue str_val = JS_JSONStringify(kv->ctx, root, JS_UNDEFINED, JS_UNDEFINED);
    const char *json_c = JS_ToCString(kv->ctx, str_val);
    
    FILE *f = fopen(idx_path, "wb");
    if (f) {
        fwrite(json_c, 1, strlen(json_c), f);
        fclose(f);
    }
    
    JS_FreeCString(kv->ctx, json_c);
    JS_FreeValue(kv->ctx, str_val);
    JS_FreeValue(kv->ctx, root);
    
    return 0;
}

void kv_store_close(kv_store_t *kv) {
    if (!kv) return;
    kv_flush(kv);
    for (size_t i = 0; i < kv->n_active; i++) {
        if (kv->active_cache[i].K) free(kv->active_cache[i].K);
        if (kv->active_cache[i].V) free(kv->active_cache[i].V);
    }
    if (kv->active_cache) free(kv->active_cache);
    if (kv->pages) free(kv->pages);
    free(kv);
}

/* ──────────────────────────────────────────────────────────────────
 * JS Bindings
 * ────────────────────────────────────────────────────────────────── */

static JSValue js_kv_open(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 2) return JS_ThrowTypeError(ctx, "Expected path and config object");
    
    const char *path = JS_ToCString(ctx, argv[0]);
    JSValue cfg = argv[1];
    
    JSValue v_mod = JS_GetPropertyStr(ctx, cfg, "modelId");
    const char *model_id = JS_ToCString(ctx, v_mod);
    
    uint32_t n_l=0, n_h=0, h_d=0;
    JSValue v_nl = JS_GetPropertyStr(ctx, cfg, "nLayers"); JS_ToUint32(ctx, &n_l, v_nl); JS_FreeValue(ctx, v_nl);
    JSValue v_nh = JS_GetPropertyStr(ctx, cfg, "nHeads");  JS_ToUint32(ctx, &n_h, v_nh); JS_FreeValue(ctx, v_nh);
    JSValue v_hd = JS_GetPropertyStr(ctx, cfg, "headDim"); JS_ToUint32(ctx, &h_d, v_hd); JS_FreeValue(ctx, v_hd);
    
    /* default f8/f16 */
    kv_store_t *kv = kv_store_open(ctx, path, model_id, n_l, n_h, h_d, 256, 
                                   QTSQ_TENSOR_F8, QTSQ_TENSOR_F16, 2);
    
    if (model_id) JS_FreeCString(ctx, model_id);
    JS_FreeValue(ctx, v_mod);
    JS_FreeCString(ctx, path);
    
    if (!kv) return JS_ThrowInternalError(ctx, "Failed to initialize KV");
    
    JSValue obj = JS_NewObjectClass(ctx, kv_class_id);
    JS_SetOpaque(obj, kv);
    kv->obj = obj;
    return obj;
}

static JSValue js_kv_save(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 3) return JS_ThrowTypeError(ctx, "Expected K(Float32Array), V(Float32Array), n_tokens");
    
    kv_store_t *kv = (kv_store_t *)JS_GetOpaque2(ctx, this_val, kv_class_id);
    if (!kv) return JS_EXCEPTION;
    
    size_t k_len=0, k_off=0, v_len=0, v_off=0;
    
    JSValue abK = JS_GetTypedArrayBuffer(ctx, argv[0], &k_off, &k_len, NULL);
    if (JS_IsException(abK)) return JS_ThrowTypeError(ctx, "K must be Float32Array");
    size_t aszK; uint8_t *bK = JS_GetArrayBuffer(ctx, &aszK, abK); JS_FreeValue(ctx, abK);
    const float *K = (const float *)(bK + k_off);
    
    JSValue abV = JS_GetTypedArrayBuffer(ctx, argv[1], &v_off, &v_len, NULL);
    if (JS_IsException(abV)) return JS_ThrowTypeError(ctx, "V must be Float32Array");
    size_t aszV; uint8_t *bV = JS_GetArrayBuffer(ctx, &aszV, abV); JS_FreeValue(ctx, abV);
    const float *V = (const float *)(bV + v_off);
    
    uint32_t n_tok = 0;
    JS_ToUint32(ctx, &n_tok, argv[2]);
    
    uint32_t pid = kv_page_save(kv, K, V, n_tok, 1.0f);
    return JS_NewUint32(ctx, pid);
}

static JSValue js_kv_flush(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    kv_store_t *kv = (kv_store_t *)JS_GetOpaque2(ctx, this_val, kv_class_id);
    if (!kv) return JS_EXCEPTION;
    return JS_NewInt32(ctx, kv_flush(kv));
}

static JSValue js_kv_search(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 2) return JS_ThrowTypeError(ctx, "Expected query(Float32Array[64]) and n_hints");
    
    kv_store_t *kv = (kv_store_t *)JS_GetOpaque2(ctx, this_val, kv_class_id);
    if (!kv) return JS_EXCEPTION;
    
    size_t q_len=0, q_off=0;
    JSValue abQ = JS_GetTypedArrayBuffer(ctx, argv[0], &q_off, &q_len, NULL);
    if (JS_IsException(abQ)) return JS_ThrowTypeError(ctx, "Query must be Float32Array");
    size_t asz; uint8_t *b = JS_GetArrayBuffer(ctx, &asz, abQ); JS_FreeValue(ctx, abQ);
    const float *query = (const float *)(b + q_off);
    
    uint32_t n_hints = 0;
    JS_ToUint32(ctx, &n_hints, argv[1]);
    
    kv_search_hints_t hints = kv_search(kv, query, n_hints);
    
    JSValue arr = JS_NewArray(ctx);
    for (size_t i = 0; i < hints.n; i++) {
        JS_SetPropertyUint32(ctx, arr, (uint32_t)i, JS_NewUint32(ctx, hints.page_ids[i]));
    }
    if (hints.page_ids) free(hints.page_ids);
    
    return arr;
}

static const JSCFunctionListEntry kv_proto_funcs[] = {
    JS_CFUNC_DEF("save", 3, js_kv_save),
    JS_CFUNC_DEF("search", 2, js_kv_search),
    JS_CFUNC_DEF("flush", 0, js_kv_flush),
};

static const JSCFunctionListEntry kv_module_funcs[] = {
    JS_CFUNC_DEF("open", 2, js_kv_open),
};

void mod_kv_register(JSContext *ctx) {
    JS_NewClassID(&kv_class_id);
    JS_NewClass(JS_GetRuntime(ctx), kv_class_id, &kv_class);
    
    JSValue proto = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, proto, kv_proto_funcs, countof(kv_proto_funcs));
    JS_SetClassProto(ctx, kv_class_id, proto);
    
    JSValue global_obj = JS_GetGlobalObject(ctx);
    JSValue sofuu_obj = JS_GetPropertyStr(ctx, global_obj, "sofuu");
    if (JS_IsUndefined(sofuu_obj)) {
        sofuu_obj = JS_NewObject(ctx);
        JS_SetPropertyStr(ctx, global_obj, "sofuu", sofuu_obj);
        sofuu_obj = JS_GetPropertyStr(ctx, global_obj, "sofuu");
    }
    
    JSValue kv_mod = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, kv_mod, kv_module_funcs, countof(kv_module_funcs));
    JS_SetPropertyStr(ctx, sofuu_obj, "kv", kv_mod);
    
    JS_FreeValue(ctx, sofuu_obj);
    JS_FreeValue(ctx, global_obj);
}
