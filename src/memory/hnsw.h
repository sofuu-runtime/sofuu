#ifndef SOFUU_MEMORY_HNSW_H
#define SOFUU_MEMORY_HNSW_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HNSW_MAX_LEVELS 16
#define HNSW_MAX_M      16  /* max outbound links per node on lower levels */
#define HNSW_MAX_M0     32  /* max outbound links per node on level 0 */

/* Forward declaration of the vector store (flat array managed by mod_memory) */
typedef struct hnsw_s hnsw_t;

typedef struct {
    uint32_t *data;
    uint32_t size;
    uint32_t capacity;
} hnsw_link_list_t;

typedef struct {
    uint32_t id;      /* Same as cma_record_t.vector_index */
    uint8_t  level;
    /* links[0] is level 0 connections, links[1] is level 1... up to level */
    hnsw_link_list_t *links;
} hnsw_node_t;

struct hnsw_s {
    hnsw_node_t *nodes;
    uint32_t     node_count;
    uint32_t     node_capacity;
    
    uint32_t     enter_node;    /* Entry point for search (highest level node) */
    uint8_t      max_level;     /* Max level currently in graph */
    
    const float *vec_store;     /* Read-only pointer to the flat float array */
    size_t       vec_dim;       /* Dimensionality (e.g., 768) */
};

/* Result structure for search */
typedef struct {
    uint32_t id;
    float    distance;
} hnsw_result_t;

/* Initialize an empty HNSW graph */
hnsw_t *hnsw_create(size_t vec_dim, uint32_t initial_capacity);

/* Set the flat float array where actual vectors live */
void hnsw_set_vec_store(hnsw_t *hnsw, const float *vec_store);

/* Add a new vector index to the HNSW graph */
int hnsw_add(hnsw_t *hnsw, uint32_t vec_index);

/* Search the graph for the closest `ef` nodes */
int hnsw_search(hnsw_t *hnsw, const float *query, uint32_t ef, 
                hnsw_result_t *out_results, uint32_t *out_count);

/* Free the graph structure */
void hnsw_free(hnsw_t *hnsw);

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_MEMORY_HNSW_H */
