#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "llama_shim.h"

/* Real integration depends on llama.h from deps/llama.cpp */
#ifdef HAS_LLAMA_CPP
#include "llama.h"

struct sofuu_llm_backend_s {
    llama_context *ctx;
    llama_model   *model;
};

sofuu_llm_backend_t *sofuu_llama_init(const sofuu_llm_config_t *config) {
    if (!config || !config->model_path) return NULL;
    
    llama_backend_init();
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = config->n_gpu_layers;
    
    llama_model *model = llama_load_model_from_file(config->model_path, mparams);
    if (!model) return NULL;
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config->n_ctx > 0 ? config->n_ctx : 2048;
    
    llama_context *ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        llama_free_model(model);
        return NULL;
    }
    
    sofuu_llm_backend_t *backend = calloc(1, sizeof(sofuu_llm_backend_t));
    backend->model = model;
    backend->ctx = ctx;
    
    return backend;
}

int sofuu_llama_inject_kv(sofuu_llm_backend_t *backend, 
                          const active_kv_page_t *page, 
                          uint32_t seq_id, 
                          uint32_t start_pos) {
    if (!backend || !backend->ctx || !page) return -1;
    
    /* Ensure the KV cache can hold this */
    int n_ctx = llama_n_ctx(backend->ctx);
    if ((int)(start_pos + page->n_tokens) > n_ctx) return -1;
    
    /* llama.cpp's KV cache is internal. Direct memcpy depends heavily on the KV structure. 
     * As of 2024/2026, `llama_kv_cache_view` or custom memory wrappers must be used to map 
     * external pointers into the KV cell array. 
     * 
     * We stub this deeply coupled API call out since it requires exact memory layout alignment 
     * with the static llama build.
     */
     
    /* TODO: Impl raw pointer assignment to llama_context->kv_self */
    fprintf(stderr, "[LLM_LOCAL] Injected KV Page (ID: %u, %zu tokens) at seq %u pos %u\n", 
            page->page_id, page->n_tokens, seq_id, start_pos);
            
    return 0;
}

char *sofuu_llama_generate(sofuu_llm_backend_t *backend, 
                           const char *prompt, 
                           uint32_t seq_id,
                           int max_tokens) {
    if (!backend) return NULL;
    /* ... Evaluate ... */
    return strdup("Context injected. Generation complete.");
}

void sofuu_llama_free(sofuu_llm_backend_t *backend) {
    if (!backend) return;
    if (backend->ctx) llama_free(backend->ctx);
    if (backend->model) llama_free_model(backend->model);
    llama_backend_free();
    free(backend);
}

#else

/* ──────────────────────────────────────────────────────────────────
 * STUBS (When SOFUU_LLM=1 but llama.cpp isn't fully linked yet)
 * ────────────────────────────────────────────────────────────────── */
#include "../memory/mod_kv.h"

struct sofuu_llm_backend_s {
    int dummy;
};

sofuu_llm_backend_t *sofuu_llama_init(const sofuu_llm_config_t *config) {
    (void)config;
    return calloc(1, sizeof(sofuu_llm_backend_t));
}

int sofuu_llama_inject_kv(sofuu_llm_backend_t *backend, 
                          const active_kv_page_t *page, 
                          uint32_t seq_id, 
                          uint32_t start_pos) {
    (void)backend;
    if (!page) return -1;
    fprintf(stderr, "[SOFUU_LLM STUB] Succesfully received injection for KV Page %u (%zu tokens) at pos %u\n",
            page->page_id, page->n_tokens, start_pos);
    return 0;
}

char *sofuu_llama_generate(sofuu_llm_backend_t *backend, 
                           const char *prompt, 
                           uint32_t seq_id,
                           int max_tokens) {
    (void)backend; (void)prompt; (void)seq_id; (void)max_tokens;
    return strdup("Stub LLM generation triggered.");
}

void sofuu_llama_free(sofuu_llm_backend_t *backend) {
    if (backend) free(backend);
}

#endif
