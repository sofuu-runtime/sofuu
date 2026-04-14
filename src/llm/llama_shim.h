#ifndef SOFUU_LLAMA_SHIM_H
#define SOFUU_LLAMA_SHIM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations for Sofuu memory models */
typedef struct active_kv_page_s active_kv_page_t; 

/* Opaque handle to the LLM backend context */
typedef struct sofuu_llm_backend_s sofuu_llm_backend_t;

/* Optional config to initialize the shim */
typedef struct {
    const char *model_path;
    int         n_ctx;
    int         n_gpu_layers;
} sofuu_llm_config_t;

/* Initialize an LLM backend */
sofuu_llm_backend_t *sofuu_llama_init(const sofuu_llm_config_t *config);

/*
 * sofuu_llama_inject_kv
 *
 * Core architectural glue:
 * Injects our `active_kv_page_t` (containing f8/f16 decompressed tensors) 
 * directly into the target llama.cpp context at a specified sequence ID.
 *
 * This completely circumvents default LLM KV generation, enabling
 * instant context swaps for the Tsubu/Cognitive engine.
 */
int sofuu_llama_inject_kv(sofuu_llm_backend_t *backend, 
                          const active_kv_page_t *page, 
                          uint32_t seq_id, 
                          uint32_t start_pos);

/* Generic prompt execution utilizing injected context */
char *sofuu_llama_generate(sofuu_llm_backend_t *backend, 
                           const char *prompt, 
                           uint32_t seq_id,
                           int max_tokens);

void sofuu_llama_free(sofuu_llm_backend_t *backend);

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_LLAMA_SHIM_H */
