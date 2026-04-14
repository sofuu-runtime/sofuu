#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../simd/simd.h"
#include "hnsw.h"

#define HNSW_EF_CONSTRUCTION 32

/* Simple max heap for nearest neighbor searches.
 * We need to maintain a priority queue of elements by distance. */
typedef struct {
    uint32_t id;
    float dist;
} max_heap_item_t;

typedef struct {
    max_heap_item_t *data;
    uint32_t size;
    uint32_t capacity;
} max_heap_t;

static max_heap_t *max_heap_create(uint32_t capacity) {
    max_heap_t *h = (max_heap_t *)malloc(sizeof(max_heap_t));
    if (!h) return NULL;
    h->data = (max_heap_item_t *)malloc(sizeof(max_heap_item_t) * capacity);
    if (!h->data) { free(h); return NULL; }
    h->size = 0;
    h->capacity = capacity;
    return h;
}

static void max_heap_free(max_heap_t *h) {
    if (h) {
        if (h->data) free(h->data);
        free(h);
    }
}

static void max_heap_push(max_heap_t *h, uint32_t id, float dist) {
    if (h->size >= h->capacity) return; /* should not happen if properly sized */
    uint32_t i = h->size++;
    while (i > 0) {
        uint32_t p = (i - 1) / 2;
        if (h->data[p].dist >= dist) break;
        h->data[i] = h->data[p];
        i = p;
    }
    h->data[i].id = id;
    h->data[i].dist = dist;
}

static void max_heap_pop(max_heap_t *h, uint32_t *out_id, float *out_dist) {
    if (h->size == 0) return;
    if (out_id) *out_id = h->data[0].id;
    if (out_dist) *out_dist = h->data[0].dist;
    
    max_heap_item_t temp = h->data[--h->size];
    uint32_t i = 0;
    while (2 * i + 1 < h->size) {
        uint32_t left = 2 * i + 1;
        uint32_t right = 2 * i + 2;
        uint32_t largest = left;
        if (right < h->size && h->data[right].dist > h->data[left].dist) {
            largest = right;
        }
        if (temp.dist >= h->data[largest].dist) break;
        h->data[i] = h->data[largest];
        i = largest;
    }
    h->data[i] = temp;
}

static float calc_dist(const hnsw_t *hnsw, uint32_t id1, uint32_t id2) {
    const float *v1 = hnsw->vec_store + (size_t)id1 * hnsw->vec_dim;
    const float *v2 = hnsw->vec_store + (size_t)id2 * hnsw->vec_dim;
    return sofuu_l2_f32(v1, v2, hnsw->vec_dim);
}

static float calc_query_dist(const hnsw_t *hnsw, const float *query, uint32_t id2) {
    const float *v2 = hnsw->vec_store + (size_t)id2 * hnsw->vec_dim;
    return sofuu_l2_f32(query, v2, hnsw->vec_dim);
}

/* LCG Random for generating node levels */
static uint32_t lcg_seed = 12345;
static float random_uniform() {
    lcg_seed = 1664525 * lcg_seed + 1013904223;
    return (float)lcg_seed / (float)0xFFFFFFFF;
}

static uint8_t get_random_level() {
    double mult = 1.0 / log(2.0); /* parameter roughly setting branching to 2 */
    double r = random_uniform();
    if (r == 0) r = 0.0000001; 
    int level = (int)(-log(r) * mult);
    if (level >= HNSW_MAX_LEVELS) level = HNSW_MAX_LEVELS - 1;
    return (uint8_t)level;
}

hnsw_t *hnsw_create(size_t vec_dim, uint32_t initial_capacity) {
    hnsw_t *h = (hnsw_t *)calloc(1, sizeof(hnsw_t));
    if (!h) return NULL;
    h->node_capacity = initial_capacity > 0 ? initial_capacity : 1024;
    h->nodes = (hnsw_node_t *)calloc(h->node_capacity, sizeof(hnsw_node_t));
    if (!h->nodes) { free(h); return NULL; }
    h->node_count = 0;
    h->vec_dim = vec_dim;
    h->enter_node = 0xFFFFFFFF; /* Invalid */
    h->max_level = 0;
    return h;
}

void hnsw_set_vec_store(hnsw_t *hnsw, const float *vec_store) {
    if (hnsw) hnsw->vec_store = vec_store;
}

static void search_layer(hnsw_t *hnsw, const float *query, uint32_t ep_id, 
                         uint32_t ef, uint8_t level, max_heap_t *top_k) {
    /* Boolean visited logic: bitset for uint32 indices */
    uint32_t bitset_size = (hnsw->node_count / 32) + 1;
    uint32_t *visited = (uint32_t *)calloc(bitset_size, sizeof(uint32_t));
    
    #define SET_VISITED(id) visited[(id) >> 5] |= (1U << ((id) & 31))
    #define IS_VISITED(id)  (visited[(id) >> 5] & (1U << ((id) & 31)))
    
    max_heap_t *candidates = max_heap_create(hnsw->node_count);
    
    float initial_dist = calc_query_dist(hnsw, query, ep_id);
    SET_VISITED(ep_id);
    
    /* In candidate queue, we want to pop the *closest* element first (min-heap). 
     * Since max_heap_t pops max, we store negative distances. */
    max_heap_push(candidates, ep_id, -initial_dist);
    max_heap_push(top_k, ep_id, initial_dist);
    
    while (candidates->size > 0) {
        uint32_t c_id;
        float c_dist_neg;
        max_heap_pop(candidates, &c_id, &c_dist_neg);
        float c_dist = -c_dist_neg;
        
        float f_dist = top_k->data[0].dist; /* Farthest in top_k */
        if (c_dist > f_dist) break;
        
        /* Evaluate neighbors */
        hnsw_node_t *node = &hnsw->nodes[c_id];
        hnsw_link_list_t *links = &node->links[level];
        
        for (uint32_t i = 0; i < links->size; i++) {
            uint32_t e_id = links->data[i];
            if (IS_VISITED(e_id)) continue;
            SET_VISITED(e_id);
            
            float e_dist = calc_query_dist(hnsw, query, e_id);
            f_dist = top_k->data[0].dist;
            
            if (top_k->size < ef || e_dist < f_dist) {
                max_heap_push(candidates, e_id, -e_dist);
                max_heap_push(top_k, e_id, e_dist);
                if (top_k->size > ef) {
                    max_heap_pop(top_k, NULL, NULL);
                }
            }
        }
    }
    
    free(visited);
    max_heap_free(candidates);
}

int hnsw_add(hnsw_t *hnsw, uint32_t vec_index) {
    if (!hnsw || !hnsw->vec_store) return -1;
    
    if (hnsw->node_count >= hnsw->node_capacity) {
        uint32_t new_cap = hnsw->node_capacity * 2;
        hnsw_node_t *new_nodes = (hnsw_node_t *)realloc(hnsw->nodes, new_cap * sizeof(hnsw_node_t));
        if (!new_nodes) return -1;
        memset(new_nodes + hnsw->node_capacity, 0, (new_cap - hnsw->node_capacity) * sizeof(hnsw_node_t));
        hnsw->nodes = new_nodes;
        hnsw->node_capacity = new_cap;
    }
    
    uint32_t new_id = vec_index; 
    uint8_t level = get_random_level();
    
    hnsw_node_t *new_node = &hnsw->nodes[new_id];
    new_node->id = new_id;
    new_node->level = level;
    new_node->links = (hnsw_link_list_t *)calloc(level + 1, sizeof(hnsw_link_list_t));
    for (uint8_t i = 0; i <= level; i++) {
        uint32_t M = (i == 0) ? HNSW_MAX_M0 : HNSW_MAX_M;
        new_node->links[i].data = (uint32_t *)malloc(M * sizeof(uint32_t));
        new_node->links[i].capacity = M;
        new_node->links[i].size = 0;
    }
    
    hnsw->node_count++;
    
    /* First node */
    if (hnsw->enter_node == 0xFFFFFFFF) {
        hnsw->enter_node = new_id;
        hnsw->max_level = level;
        return 0;
    }
    
    const float *query = hnsw->vec_store + (size_t)new_id * hnsw->vec_dim;
    uint32_t ep = hnsw->enter_node;
    uint8_t max_lc = hnsw->max_level;
    
    if (level < max_lc) {
        /* Search down to level without adding links */
        for (uint8_t lc = max_lc; lc > level; lc--) {
            float dist = calc_query_dist(hnsw, query, ep);
            // Optimization: traverse towards query
            hnsw_node_t *ep_node = &hnsw->nodes[ep];
            hnsw_link_list_t *links = &ep_node->links[lc];
            int changed = 1;
            while(changed) {
                changed = 0;
                for (uint32_t i = 0; i < links->size; i++) {
                    uint32_t candidate = links->data[i];
                    float c_dist = calc_query_dist(hnsw, query, candidate);
                    if (c_dist < dist) {
                        ep = candidate;
                        dist = c_dist;
                        changed = 1;
                        links = &hnsw->nodes[ep].links[lc];
                    }
                }
            }
        }
    }
    
    /* Add links at applicable levels */
    uint8_t start_level = (level < max_lc) ? level : max_lc;
    
    max_heap_t *W = max_heap_create(HNSW_EF_CONSTRUCTION + 1);
    
    for (int lc = start_level; lc >= 0; lc--) {
        max_heap_push(W, ep, calc_query_dist(hnsw, query, ep));
        search_layer(hnsw, query, ep, HNSW_EF_CONSTRUCTION, (uint8_t)lc, W);
        
        /* Select M nearest elements from W to connect to */
        uint32_t M = (lc == 0) ? HNSW_MAX_M0 : HNSW_MAX_M;
        
        uint32_t count = W->size > M ? M : W->size;
        /* we have a max heap, easiest is to extract all into an array, then add to links */
        /* W currently has EF items, pop until we have M smallest items. */
        while(W->size > M) max_heap_pop(W, NULL, NULL);
        
        hnsw_link_list_t *new_node_links = &new_node->links[lc];
        for (uint32_t i = 0; i < count; i++) {
            uint32_t neighbor_id = W->data[i].id;
            new_node_links->data[new_node_links->size++] = neighbor_id;
            
            /* Add reverse link from neighbor to new_id */
            hnsw_link_list_t *nr_links = &hnsw->nodes[neighbor_id].links[lc];
            if (nr_links->size < nr_links->capacity) {
                nr_links->data[nr_links->size++] = new_id;
            } else {
                /* Naive link pruning for neighbor: just replace worst link */
                float worst_dist = 0;
                uint32_t worst_idx = 0;
                for (uint32_t j = 0; j < nr_links->size; j++) {
                    float d = calc_dist(hnsw, neighbor_id, nr_links->data[j]);
                    if (d > worst_dist) {
                        worst_dist = d;
                        worst_idx = j;
                    }
                }
                float new_dist = calc_dist(hnsw, neighbor_id, new_id);
                if (new_dist < worst_dist) {
                    nr_links->data[worst_idx] = new_id;
                }
            }
        }
        
        /* ep for next level down is nearest item in W */
        if (lc > 0) {
            float min_dist = -1.0f;
            for (uint32_t i = 0; i < W->size; i++) {
                if (min_dist < 0 || W->data[i].dist < min_dist) {
                    min_dist = W->data[i].dist;
                    ep = W->data[i].id;
                }
            }
        }
        /* Reset W for next level */
        W->size = 0;
    }
    
    max_heap_free(W);
    
    if (level > hnsw->max_level) {
        hnsw->max_level = level;
        hnsw->enter_node = new_id;
    }
    
    return 0;
}

int hnsw_search(hnsw_t *hnsw, const float *query, uint32_t ef, 
                hnsw_result_t *out_results, uint32_t *out_count) {
    if (!hnsw || !query || !out_results || !out_count) return -1;
    if (hnsw->node_count == 0 || hnsw->enter_node == 0xFFFFFFFF) {
        *out_count = 0;
        return 0;
    }
    
    uint32_t ep = hnsw->enter_node;
    uint8_t max_lc = hnsw->max_level;
    
    /* Search down to level 0 */
    for (uint8_t lc = max_lc; lc > 0; lc--) {
        float dist = calc_query_dist(hnsw, query, ep);
        hnsw_node_t *ep_node = &hnsw->nodes[ep];
        hnsw_link_list_t *links = &ep_node->links[lc];
        int changed = 1;
        while(changed) {
            changed = 0;
            for (uint32_t i = 0; i < links->size; i++) {
                uint32_t candidate = links->data[i];
                float c_dist = calc_query_dist(hnsw, query, candidate);
                if (c_dist < dist) {
                    ep = candidate;
                    dist = c_dist;
                    changed = 1;
                    links = &hnsw->nodes[ep].links[lc];
                }
            }
        }
    }
    
    /* Final search at level 0 */
    max_heap_t *W = max_heap_create(ef + 1);
    search_layer(hnsw, query, ep, ef, 0, W);
    
    *out_count = W->size;
    /* Move results to output array (W contains max-heap of distances) */
    /* Pop everything to get it from largest to smallest distance */
    /* Write to output backwards to get smallest first */
    for (int i = (int)W->size - 1; i >= 0; i--) {
        max_heap_pop(W, &out_results[i].id, &out_results[i].distance);
    }
    
    max_heap_free(W);
    return 0;
}

void hnsw_free(hnsw_t *hnsw) {
    if (!hnsw) return;
    for (uint32_t i = 0; i < hnsw->node_capacity; i++) {
        hnsw_node_t *node = &hnsw->nodes[i];
        if (node->links) {
            for (uint8_t j = 0; j <= node->level; j++) {
                if (node->links[j].data) free(node->links[j].data);
            }
            free(node->links);
        }
    }
    if (hnsw->nodes) free(hnsw->nodes);
    free(hnsw);
}
