#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qtsq_adapter.h"

/* ──────────────────────────────────────────────────────────────────
 * CMA (Cognitive Memory Architecture) Operations
 * ────────────────────────────────────────────────────────────────── */

int cma_qtsq_save(const char *path,
                  const float *vecs, size_t n_memories, size_t vec_dim,
                  const char *metadata_json) {
    /* CMA uses qtsq_brain_save without weights/biases.
     * The vectors are saved losslessly (f32). */
    return qtsq_brain_save(path,
                           NULL, 0,  /* weights */
                           NULL, 0,  /* biases */
                           vecs, n_memories, vec_dim,
                           metadata_json);
}

int cma_qtsq_load(const char *path,
                  float **out_vecs, size_t *out_n, size_t *out_dim,
                  char **out_metadata_json) {
    /* CMA brain load */
    return qtsq_brain_load(path,
                           NULL, NULL, /* out_weights */
                           NULL, NULL, /* out_biases */
                           out_vecs, out_n, out_dim,
                           out_metadata_json);
}

/* ──────────────────────────────────────────────────────────────────
 * SSD KV Cache Operations
 * ────────────────────────────────────────────────────────────────── */

int kv_qtsq_save_page(qtsq_context_t *container,
                      const float *K, const float *V,
                      size_t n_layers, size_t n_heads,
                      size_t n_tokens, size_t head_dim,
                      uint32_t page_id,
                      qtsq_tensor_precision_t k_precision,
                      qtsq_tensor_precision_t v_precision) {
    if (!container || !K || !V) return -1;

    size_t count = n_layers * n_heads * n_tokens * head_dim;
    if (count == 0) return 0;

    /* Combine dimensions for tensor shape: [layers, heads, tokens, dim] */
    uint32_t dims[4] = {
        (uint32_t)n_layers,
        (uint32_t)n_heads,
        (uint32_t)n_tokens,
        (uint32_t)head_dim
    };

    char stream_name[32];
    
    /* 1. Add K tensor */
    qtsq_context_t ctx_k;
    qtsq_init(&ctx_k);
    int r = qtsq_compress_tensor_quantized(&ctx_k, K, count, dims, 4, k_precision);
    if (r == QTSQ_OK) {
        snprintf(stream_name, sizeof(stream_name), "kv_%08u_K", page_id);
        r = qtsq_container_add_stream(container, &ctx_k, stream_name);
    }
    qtsq_free(&ctx_k);
    if (r != QTSQ_OK) return r;

    /* 2. Add V tensor */
    qtsq_context_t ctx_v;
    qtsq_init(&ctx_v);
    r = qtsq_compress_tensor_quantized(&ctx_v, V, count, dims, 4, v_precision);
    if (r == QTSQ_OK) {
        snprintf(stream_name, sizeof(stream_name), "kv_%08u_V", page_id);
        r = qtsq_container_add_stream(container, &ctx_v, stream_name);
    }
    qtsq_free(&ctx_v);

    return r;
}

int kv_qtsq_load_page(const char *kv_file, uint32_t page_id,
                      float **out_K, float **out_V,
                      size_t *out_n_layers, size_t *out_n_tokens) {
    if (!kv_file || !out_K || !out_V) return -1;

    qtsq_context_t container;
    qtsq_init(&container);
    int r = qtsq_read(&container, kv_file);
    if (r != QTSQ_OK) {
        qtsq_free(&container);
        return r;
    }

    uint32_t num_streams = 0;
    r = qtsq_container_get_count(&container, &num_streams);
    if (r != QTSQ_OK) {
        qtsq_free(&container);
        return r;
    }

    char name_k[32];
    char name_v[32];
    snprintf(name_k, sizeof(name_k), "kv_%08u_K", page_id);
    snprintf(name_v, sizeof(name_v), "kv_%08u_V", page_id);

    float *k_data = NULL;
    float *v_data = NULL;
    size_t k_count = 0, v_count = 0;

    for (uint32_t i = 0; i < num_streams; i++) {
        qtsq_context_t sub;
        char name[256];
        if (qtsq_container_get_stream(&container, i, &sub, name, sizeof(name)) == QTSQ_OK) {
            if (strcmp(name, name_k) == 0) {
                if (qtsq_decompress_tensor(&sub, &k_data, &k_count) == QTSQ_OK) {
                    if (sub.schema.num_dims >= 3 && out_n_layers && out_n_tokens) {
                        *out_n_layers = sub.schema.dimensions[0];
                        *out_n_tokens = sub.schema.dimensions[2]; /* [l, h, tok, d] */
                    }
                }
            } else if (strcmp(name, name_v) == 0) {
                qtsq_decompress_tensor(&sub, &v_data, &v_count);
            }
            qtsq_free(&sub);
        }
    }

    qtsq_free(&container);

    if (k_data && v_data && k_count > 0 && k_count == v_count) {
        *out_K = k_data;
        *out_V = v_data;
        return QTSQ_OK;
    }

    if (k_data) free(k_data);
    if (v_data) free(v_data);
    return QTSQ_ERR_FORMAT; /* Page not fully found or corrupted */
}

int kv_qtsq_flush(qtsq_context_t *container, const char *path) {
    if (!container || !path) return -1;
    /* Direct stream saving without a massive ram-buffer */
    return qtsq_container_write_direct(container, path);
}
