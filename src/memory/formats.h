#ifndef SOFUU_MEMORY_FORMATS_H
#define SOFUU_MEMORY_FORMATS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────────────
 * CMA Metadata Layout
 * Serialized directly to JSON in the 'metadata' stream of QTSQ
 * ────────────────────────────────────────────────────────────────── */

/* Memory tiers */
#define CMA_TIER_WORKING    0  /* Ring buffer, not saved to disk */
#define CMA_TIER_EPISODIC   1  /* Recent individual interactions */
#define CMA_TIER_SEMANTIC   2  /* Clustered facts after consolidation */
#define CMA_TIER_FORGOTTEN  3  /* Marked for garbage collection */

/* CMA memory single record.
 * The high-dimensional float array lives in the QTSQ vector stream.
 * This metadata lives in the QTSQ JSON stream matching by 'vector_index'.
 */
typedef struct {
    uint32_t vector_index;   /* Array index in the massive float array (f32 stream) */
    uint32_t kv_page_id;     /* Which SSD KV page this memory originated from (0 = none) */
    
    float    strength;       /* Current Ebbinghaus retention score [0.0 - 1.0] */
    uint32_t age_seconds;    /* Relative age or absolute seconds since creation */
    uint32_t half_life;      /* How fast it decays (seconds) */
    uint8_t  tier;           /* CMA_TIER_* */
    
    char    *text;           /* The actual text (null-terminated, malloced) */
    char    *role;           /* "user", "assistant", "system", etc */
} cma_record_t;


/* ──────────────────────────────────────────────────────────────────
 * SSD KV Index Headers
 * Serialized to the 'index' stream of the KV container
 * ────────────────────────────────────────────────────────────────── */

#define KV_SUMMARY_DIM 64

/* Summary lookup entry per page.
 * Avoids loading a 500MB tensor just to measure vector similarity. */
typedef struct {
    uint32_t page_id;
    float    k_summary[KV_SUMMARY_DIM]; /* Mean-pooled projection of the page's Keys */
    float    strength;
    uint32_t created_at;
} kv_page_summary_t;

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_MEMORY_FORMATS_H */
