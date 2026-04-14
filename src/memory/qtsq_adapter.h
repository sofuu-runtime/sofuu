#ifndef SOFUU_QTSQ_ADAPTER_H
#define SOFUU_QTSQ_ADAPTER_H

#include <stddef.h>
#include <stdint.h>
#include "qtsq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────────────
 * CMA (Cognitive Memory Architecture) Operations
 * Uses qtsq_brain_save (lossless f32) for precise embedding vectors
 * ────────────────────────────────────────────────────────────────── */

int cma_qtsq_save(const char *path,
                  const float *vecs, size_t n_memories, size_t vec_dim,
                  const char *metadata_json);

int cma_qtsq_load(const char *path,
                  float **out_vecs, size_t *out_n, size_t *out_dim,
                  char **out_metadata_json);

/* ──────────────────────────────────────────────────────────────────
 * SSD KV Cache Operations
 * Uses quantized f8/f16 precision to maximize storage density
 * ────────────────────────────────────────────────────────────────── */

int kv_qtsq_save_page(qtsq_context_t *container,
                      const float *K, const float *V,
                      size_t n_layers, size_t n_heads,
                      size_t n_tokens, size_t head_dim,
                      uint32_t page_id,
                      qtsq_tensor_precision_t k_precision,
                      qtsq_tensor_precision_t v_precision);

int kv_qtsq_load_page(const char *kv_file, uint32_t page_id,
                      float **out_K, float **out_V,
                      size_t *out_n_layers, size_t *out_n_tokens);

int kv_qtsq_flush(qtsq_context_t *container, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_QTSQ_ADAPTER_H */
