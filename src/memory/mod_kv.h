#ifndef SOFUU_MOD_KV_H
#define SOFUU_MOD_KV_H

#include <stdint.h>
#include <stddef.h>
#include "quickjs.h"
#include "formats.h"

#ifdef __cplusplus
extern "C" {
#endif

/* An active decompressed page held in RAM.
 * Mallocs from qtsq_decompress_tensor().
 * We keep only a few of these in an LRU cache to avoid RAM explosion. */
typedef struct active_kv_page_s {
    uint32_t page_id;
    float    *K;
    float    *V;
    size_t   n_layers;
    size_t   n_tokens;
    uint32_t last_accessed; /* for LRU eviction */
} active_kv_page_t;

/* KV Store Context */
typedef struct {
    char      file_path[512];
    char      model_id[128];
    
    size_t    n_layers;
    size_t    n_heads;
    size_t    head_dim;
    
    /* Memory capacities / limits */
    size_t    max_size_gb;
    int       k_precision; /* QTSQ precision enum */
    int       v_precision; /* QTSQ precision enum */
    
    /* In-memory index of all pages (very fast to search) */
    kv_page_summary_t *pages;
    size_t    n_pages;
    size_t    page_capacity;
    
    /* In-memory LRU cache of decompressed tensors */
    active_kv_page_t *active_cache;
    size_t    n_active;
    size_t    max_active;
    uint32_t  access_counter;
    
    JSContext *ctx;
    JSValue   obj;
} kv_store_t;

void mod_kv_register(JSContext *ctx);

/* 
 * KV Store low level C APIs
 */

kv_store_t *kv_store_open(JSContext *ctx, const char *path, const char *model_id,
                          size_t n_layers, size_t n_heads, size_t head_dim,
                          size_t max_size_gb, int k_prec, int v_prec, size_t max_ram_pages);

/* Compress and append a page. Updates index K-summary. */
uint32_t kv_page_save(kv_store_t *kv, const float *K, const float *V, 
                      size_t n_tokens, float strength);

/* Search summaries for closest K-vector match */
typedef struct {
    uint32_t *page_ids;
    size_t n;
} kv_search_hints_t;

kv_search_hints_t kv_search(kv_store_t *kv, const float *query_summary_64, size_t n_hints);

/* Prefetch and get a page into LRU cache */
active_kv_page_t *kv_get_page(kv_store_t *kv, uint32_t page_id);

/* Flush the index stream changes out */
int kv_flush(kv_store_t *kv);

void kv_store_close(kv_store_t *kv);

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_MOD_KV_H */
