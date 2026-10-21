/*
 * sofuu_qtsq_table.c — Sofuu-Optimized Strict Schema Columnar Table Engine
 *
 * Implements high-speed columnar buffer packing, bounded block chunking,
 * and checkpoint indexing using native QTSQ container primitives.
 *
 * Author: Haruhito
 */

#include "sofuu_qtsq_table.h"
#include "qtsq.h"
#include "qtsq_format.h"
#include "qtsq_codec.h"
#include "qtc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOFUU_BLOCK_MAGIC 0x5346424B /* 'SFBK' */

/* Size threshold below which data is stored raw to avoid framing overhead */
#define SOFUU_RAW_THRESHOLD 1024

/* ══════ Robust Dual-Strategy Compressor with Fallback Ladder ══════ */

static int sofuu_pack_substream(qtsq_context_t *sub, const uint8_t *data, size_t size) {
    if (!sub || !data) return QTSQ_ERR_NULL;
    if (size == 0) return QTSQ_OK;

    int r = qtsq_init(sub);
    if (r != QTSQ_OK) return r;

    /* For data below threshold (bit/byte/KB data): store raw white dwarf with 0 error risk */
    if (size < SOFUU_RAW_THRESHOLD) {
        sub->header.strategy = QTSQ_STRATEGY_NONE;
        sub->header.data_type = QTSQ_TYPE_RAW;
        sub->header.original_size = size;
        sub->singularity = (uint8_t *)malloc(size);
        if (!sub->singularity) return QTSQ_ERR_ALLOC;
        memcpy(sub->singularity, data, size);
        sub->singularity_size = size;
        sub->header.singularity_size = size;
        return QTSQ_OK;
    }

    /* For larger blocks: try QTC compression */
    uint8_t *comp = NULL;
    size_t comp_len = 0;
    int qrc = qtsq_codec_compress(data, size, &comp, &comp_len);

    /* Fallback Check: If QTC failed OR did not save space, use raw white dwarf */
    if (qrc != QTSQ_OK || !comp || comp_len >= size) {
        if (comp) free(comp);
        sub->header.strategy = QTSQ_STRATEGY_NONE;
        sub->header.data_type = QTSQ_TYPE_RAW;
        sub->header.original_size = size;
        sub->singularity = (uint8_t *)malloc(size);
        if (!sub->singularity) return QTSQ_ERR_ALLOC;
        memcpy(sub->singularity, data, size);
        sub->singularity_size = size;
        sub->header.singularity_size = size;
        return QTSQ_OK;
    }

    /* QTC compression won! Store compressed singularity */
    sub->header.strategy = QTSQ_STRATEGY_DICTIONARY;
    sub->header.data_type = QTSQ_TYPE_RAW;
    sub->header.original_size = size;
    sub->codec = QTSQ_CODEC_QTC;
    sub->singularity = comp;
    sub->singularity_size = comp_len;
    sub->header.singularity_size = comp_len;
    return QTSQ_OK;
}

static int sofuu_unpack_substream(qtsq_context_t *sub, uint8_t **out_data, size_t *out_size) {
    if (!sub || !out_data || !out_size) return QTSQ_ERR_NULL;
    *out_data = NULL;
    *out_size = 0;

    /* 1. If strategy is NONE (raw white dwarf): direct zero-decode copy */
    if (sub->header.strategy == QTSQ_STRATEGY_NONE) {
        if (!sub->singularity || sub->singularity_size == 0) return QTSQ_ERR_FORMAT;
        uint8_t *copy = (uint8_t *)malloc(sub->singularity_size);
        if (!copy) return QTSQ_ERR_ALLOC;
        memcpy(copy, sub->singularity, sub->singularity_size);
        *out_data = copy;
        *out_size = sub->singularity_size;
        return QTSQ_OK;
    }

    /* 2. If QTC compressed: decode via qtsq_codec_decompress */
    if (sub->singularity && qtsq_codec_is_qtc(sub->singularity, sub->singularity_size)) {
        uint8_t *dec = NULL;
        size_t dec_len = 0;
        uint64_t max_out = sub->header.original_size ? (sub->header.original_size * 2 + 1024) : (1024 * 1024);
        int rc = qtsq_codec_decompress(sub->singularity, sub->singularity_size, max_out, &dec, &dec_len);
        if (rc == QTSQ_OK && dec) {
            *out_data = dec;
            *out_size = dec_len;
            return QTSQ_OK;
        }
        if (dec) free(dec);
    }

    /* 3. Fallback Ladder: try standard qtsq_decompress */
    int r = qtsq_decompress(sub, out_data, out_size);
    if (r == QTSQ_OK && *out_data) return QTSQ_OK;

    /* 4. Fallback Ladder: try horizon decompress */
    r = qtsq_decompress_horizon(sub, out_data, out_size);
    if (r == QTSQ_OK && *out_data) return QTSQ_OK;

    /* 5. Ultimate Fallback: if all decoders failed, return raw singularity bytes */
    if (sub->singularity && sub->singularity_size > 0) {
        uint8_t *copy = (uint8_t *)malloc(sub->singularity_size);
        if (copy) {
            memcpy(copy, sub->singularity, sub->singularity_size);
            *out_data = copy;
            *out_size = sub->singularity_size;
            return QTSQ_OK;
        }
    }

    return QTSQ_ERR_FORMAT;
}

static size_t col_type_size(sofuu_col_type_t t) {
    switch (t) {
        case SOFUU_TYPE_INT64:
        case SOFUU_TYPE_TIMESTAMP:
        case SOFUU_TYPE_FLOAT64:
            return 8;
        case SOFUU_TYPE_INT32:
        case SOFUU_TYPE_FLOAT32:
            return 4;
        case SOFUU_TYPE_BOOL:
            return 1;
        default:
            return 0; /* variable length */
    }
}

/* ══════ Schema Implementation ══════ */

void sofuu_schema_init(sofuu_schema_t *schema, const char *table_name) {
    if (!schema) return;
    memset(schema, 0, sizeof(*schema));
    if (table_name) {
        strncpy(schema->table_name, table_name, sizeof(schema->table_name) - 1);
    }
    schema->pk_col_idx = -1;
    schema->timestamp_col_idx = -1;
}

int sofuu_schema_add_col(sofuu_schema_t *schema, const char *name, sofuu_col_type_t type, uint8_t flags) {
    if (!schema || !name || schema->num_cols >= SOFUU_MAX_COLS) {
        return -1;
    }
    uint32_t idx = schema->num_cols;
    sofuu_col_def_t *c = &schema->cols[idx];
    memset(c, 0, sizeof(*c));
    strncpy(c->name, name, sizeof(c->name) - 1);
    c->type = (uint8_t)type;
    c->flags = flags;

    if (flags & SOFUU_COL_PRIMARY_KEY) {
        schema->pk_col_idx = (int32_t)idx;
    }
    if (type == SOFUU_TYPE_TIMESTAMP || strcmp(name, "timestamp") == 0 || strcmp(name, "created_at") == 0) {
        if (schema->timestamp_col_idx < 0) {
            schema->timestamp_col_idx = (int32_t)idx;
        }
    }

    schema->num_cols++;
    return (int)idx;
}

int sofuu_schema_find_col(const sofuu_schema_t *schema, const char *name) {
    if (!schema || !name) return -1;
    for (uint32_t i = 0; i < schema->num_cols; i++) {
        if (strcmp(schema->cols[i].name, name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/* ══════ Block Implementation ══════ */

sofuu_block_t *sofuu_block_create(const sofuu_schema_t *schema, uint32_t block_idx) {
    if (!schema) return NULL;
    sofuu_block_t *b = (sofuu_block_t *)calloc(1, sizeof(sofuu_block_t));
    if (!b) return NULL;

    b->block_index = block_idx;
    b->row_count = 0;
    b->min_id = INT64_MAX;
    b->max_id = INT64_MIN;
    b->min_timestamp = INT64_MAX;
    b->max_timestamp = INT64_MIN;

    for (uint32_t i = 0; i < schema->num_cols; i++) {
        sofuu_col_type_t t = (sofuu_col_type_t)schema->cols[i].type;
        size_t sz = col_type_size(t);
        if (sz > 0) {
            b->col_data[i] = calloc(SOFUU_ROWS_PER_BLOCK, sz);
        } else {
            b->str_offsets[i] = (uint32_t *)calloc(SOFUU_ROWS_PER_BLOCK + 1, sizeof(uint32_t));
            b->str_heap_cap[i] = 1024;
            b->str_heap[i] = (uint8_t *)malloc(b->str_heap_cap[i]);
            b->str_heap_len[i] = 0;
            if (b->str_offsets[i]) {
                b->str_offsets[i][0] = 0;
            }
        }
        b->null_map[i] = (bool *)calloc(SOFUU_ROWS_PER_BLOCK, sizeof(bool));
    }
    return b;
}

void sofuu_block_free(sofuu_block_t *b, const sofuu_schema_t *schema) {
    if (!b) return;
    uint32_t num_cols = schema ? schema->num_cols : SOFUU_MAX_COLS;
    for (uint32_t i = 0; i < num_cols; i++) {
        if (b->col_data[i]) free(b->col_data[i]);
        if (b->str_offsets[i]) free(b->str_offsets[i]);
        if (b->str_heap[i]) free(b->str_heap[i]);
        if (b->null_map[i]) free(b->null_map[i]);
    }
    free(b);
}

int sofuu_block_append_row(sofuu_block_t *b, const sofuu_schema_t *schema, const sofuu_val_t *vals, uint32_t num_vals) {
    if (!b || !schema || !vals || b->row_count >= SOFUU_ROWS_PER_BLOCK) {
        return -1;
    }
    uint32_t r = b->row_count;

    for (uint32_t i = 0; i < schema->num_cols; i++) {
        const sofuu_col_def_t *c = &schema->cols[i];
        const sofuu_val_t *v = (i < num_vals) ? &vals[i] : NULL;

        if (!v || v->type == SOFUU_TYPE_NULL) {
            if (c->flags & SOFUU_COL_REQUIRED) {
                return -2; /* missing required field */
            }
            b->null_map[i][r] = true;
            if (col_type_size((sofuu_col_type_t)c->type) == 0 && b->str_offsets[i]) {
                b->str_offsets[i][r + 1] = b->str_offsets[i][r];
            }
            continue;
        }

        b->null_map[i][r] = false;

        switch (c->type) {
            case SOFUU_TYPE_INT64: {
                int64_t val = (v->type == SOFUU_TYPE_INT64) ? v->as.i64 : (int64_t)v->as.i32;
                ((int64_t *)b->col_data[i])[r] = val;
                if ((int32_t)i == schema->pk_col_idx) {
                    if (val < b->min_id) b->min_id = val;
                    if (val > b->max_id) b->max_id = val;
                }
                break;
            }
            case SOFUU_TYPE_INT32: {
                int32_t val = (v->type == SOFUU_TYPE_INT32) ? v->as.i32 : (int32_t)v->as.i64;
                ((int32_t *)b->col_data[i])[r] = val;
                if ((int32_t)i == schema->pk_col_idx) {
                    if (val < b->min_id) b->min_id = val;
                    if (val > b->max_id) b->max_id = val;
                }
                break;
            }
            case SOFUU_TYPE_FLOAT64: {
                double val = (v->type == SOFUU_TYPE_FLOAT64) ? v->as.f64 : (double)v->as.f32;
                ((double *)b->col_data[i])[r] = val;
                break;
            }
            case SOFUU_TYPE_FLOAT32: {
                float val = (v->type == SOFUU_TYPE_FLOAT32) ? v->as.f32 : (float)v->as.f64;
                ((float *)b->col_data[i])[r] = val;
                break;
            }
            case SOFUU_TYPE_BOOL: {
                ((bool *)b->col_data[i])[r] = v->as.b;
                break;
            }
            case SOFUU_TYPE_TIMESTAMP: {
                int64_t val = (v->type == SOFUU_TYPE_TIMESTAMP || v->type == SOFUU_TYPE_INT64) ? v->as.i64 : (int64_t)v->as.i32;
                ((int64_t *)b->col_data[i])[r] = val;
                if ((int32_t)i == schema->timestamp_col_idx || schema->timestamp_col_idx < 0) {
                    if (val < b->min_timestamp) b->min_timestamp = val;
                    if (val > b->max_timestamp) b->max_timestamp = val;
                }
                break;
            }
            case SOFUU_TYPE_STRING:
            case SOFUU_TYPE_BYTES:
            case SOFUU_TYPE_JSON: {
                char conv_buf[64];
                const uint8_t *src_bytes = NULL;
                uint32_t len = 0;

                if (v->type == SOFUU_TYPE_STRING || v->type == SOFUU_TYPE_JSON) {
                    src_bytes = (const uint8_t *)v->as.str.ptr;
                    len = v->as.str.len;
                } else if (v->type == SOFUU_TYPE_BYTES) {
                    src_bytes = v->as.bytes.ptr;
                    len = v->as.bytes.len;
                } else if (schema->cols[i].type == SOFUU_TYPE_JSON) {
                    // Auto-coercion of scalar primitives into JSON text
                    if (v->type == SOFUU_TYPE_INT64 || v->type == SOFUU_TYPE_TIMESTAMP) {
                        snprintf(conv_buf, sizeof(conv_buf), "%lld", (long long)v->as.i64);
                    } else if (v->type == SOFUU_TYPE_INT32) {
                        snprintf(conv_buf, sizeof(conv_buf), "%d", v->as.i32);
                    } else if (v->type == SOFUU_TYPE_FLOAT64) {
                        snprintf(conv_buf, sizeof(conv_buf), "%f", v->as.f64);
                    } else if (v->type == SOFUU_TYPE_FLOAT32) {
                        snprintf(conv_buf, sizeof(conv_buf), "%f", (double)v->as.f32);
                    } else if (v->type == SOFUU_TYPE_BOOL) {
                        snprintf(conv_buf, sizeof(conv_buf), "%s", v->as.b ? "true" : "false");
                    } else {
                        snprintf(conv_buf, sizeof(conv_buf), "null");
                    }
                    src_bytes = (const uint8_t *)conv_buf;
                    len = (uint32_t)strlen(conv_buf);
                }

                if (!src_bytes && len > 0) return -3;

                size_t needed = b->str_heap_len[i] + len;
                if (needed > b->str_heap_cap[i]) {
                    size_t new_cap = b->str_heap_cap[i] * 2;
                    if (new_cap < needed) new_cap = needed + 1024;
                    uint8_t *new_heap = (uint8_t *)realloc(b->str_heap[i], new_cap);
                    if (!new_heap) return -4;
                    b->str_heap[i] = new_heap;
                    b->str_heap_cap[i] = new_cap;
                }

                if (len > 0) {
                    memcpy(b->str_heap[i] + b->str_heap_len[i], src_bytes, len);
                }
                b->str_heap_len[i] += len;
                b->str_offsets[i][r + 1] = (uint32_t)b->str_heap_len[i];
                break;
            }
            default:
                break;
        }
    }

    b->row_count++;
    return 0;
}

int sofuu_block_serialize(const sofuu_block_t *b, const sofuu_schema_t *schema, uint8_t **out_buf, size_t *out_size) {
    if (!b || !schema || !out_buf || !out_size) return -1;

    size_t est_size = 64; /* header */
    for (uint32_t i = 0; i < schema->num_cols; i++) {
        est_size += b->row_count; /* null_map */
        size_t sz = col_type_size((sofuu_col_type_t)schema->cols[i].type);
        if (sz > 0) {
            est_size += b->row_count * sz;
        } else {
            est_size += (b->row_count + 1) * sizeof(uint32_t) + b->str_heap_len[i];
        }
    }

    uint8_t *buf = (uint8_t *)malloc(est_size);
    if (!buf) return -1;
    uint8_t *p = buf;

    /* Write Block Header */
    uint32_t magic = SOFUU_BLOCK_MAGIC;
    memcpy(p, &magic, 4); p += 4;
    memcpy(p, &b->block_index, 4); p += 4;
    memcpy(p, &b->row_count, 4); p += 4;
    memcpy(p, &b->min_id, 8); p += 8;
    memcpy(p, &b->max_id, 8); p += 8;
    memcpy(p, &b->min_timestamp, 8); p += 8;
    memcpy(p, &b->max_timestamp, 8); p += 8;

    /* Write Column Payloads */
    for (uint32_t i = 0; i < schema->num_cols; i++) {
        /* null map */
        for (uint32_t r = 0; r < b->row_count; r++) {
            *p++ = b->null_map[i][r] ? 1 : 0;
        }

        size_t sz = col_type_size((sofuu_col_type_t)schema->cols[i].type);
        if (sz > 0) {
            size_t bytes = b->row_count * sz;
            memcpy(p, b->col_data[i], bytes);
            p += bytes;
        } else {
            size_t off_bytes = (b->row_count + 1) * sizeof(uint32_t);
            memcpy(p, b->str_offsets[i], off_bytes);
            p += off_bytes;

            uint32_t heap_len = (uint32_t)b->str_heap_len[i];
            memcpy(p, &heap_len, 4); p += 4;
            if (heap_len > 0) {
                memcpy(p, b->str_heap[i], heap_len);
                p += heap_len;
            }
        }
    }

    *out_buf = buf;
    *out_size = (size_t)(p - buf);
    return 0;
}

sofuu_block_t *sofuu_block_deserialize(const uint8_t *buf, size_t size, const sofuu_schema_t *schema) {
    if (!buf || size < 44 || !schema) return NULL;
    const uint8_t *p = buf;

    uint32_t magic = 0;
    memcpy(&magic, p, 4); p += 4;
    if (magic != SOFUU_BLOCK_MAGIC) return NULL;

    uint32_t block_index = 0, row_count = 0;
    memcpy(&block_index, p, 4); p += 4;
    memcpy(&row_count, p, 4); p += 4;

    sofuu_block_t *b = sofuu_block_create(schema, block_index);
    if (!b) return NULL;
    b->row_count = row_count;

    memcpy(&b->min_id, p, 8); p += 8;
    memcpy(&b->max_id, p, 8); p += 8;
    memcpy(&b->min_timestamp, p, 8); p += 8;
    memcpy(&b->max_timestamp, p, 8); p += 8;

    for (uint32_t i = 0; i < schema->num_cols; i++) {
        for (uint32_t r = 0; r < row_count; r++) {
            b->null_map[i][r] = (*p++ != 0);
        }

        size_t sz = col_type_size((sofuu_col_type_t)schema->cols[i].type);
        if (sz > 0) {
            size_t bytes = row_count * sz;
            memcpy(b->col_data[i], p, bytes);
            p += bytes;
        } else {
            size_t off_bytes = (row_count + 1) * sizeof(uint32_t);
            memcpy(b->str_offsets[i], p, off_bytes);
            p += off_bytes;

            uint32_t heap_len = 0;
            memcpy(&heap_len, p, 4); p += 4;
            if (heap_len > b->str_heap_cap[i]) {
                b->str_heap[i] = (uint8_t *)realloc(b->str_heap[i], heap_len + 64);
                b->str_heap_cap[i] = heap_len + 64;
            }
            b->str_heap_len[i] = heap_len;
            if (heap_len > 0) {
                memcpy(b->str_heap[i], p, heap_len);
                p += heap_len;
            }
        }
    }

    return b;
}

/* ══════ Table Implementation ══════ */

sofuu_table_t *sofuu_table_create(const sofuu_schema_t *schema) {
    if (!schema) return NULL;
    sofuu_table_t *t = (sofuu_table_t *)calloc(1, sizeof(sofuu_table_t));
    if (!t) return NULL;

    memcpy(&t->schema, schema, sizeof(sofuu_schema_t));
    t->total_rows = 0;
    t->next_auto_inc = 1;
    t->blocks_cap = 16;
    t->blocks = (sofuu_block_t **)calloc(t->blocks_cap, sizeof(sofuu_block_t *));
    t->checkpoints = (sofuu_block_checkpoint_t *)calloc(t->blocks_cap, sizeof(sofuu_block_checkpoint_t));
    return t;
}

void sofuu_table_free(sofuu_table_t *tbl) {
    if (!tbl) return;
    for (uint32_t i = 0; i < tbl->num_blocks; i++) {
        if (tbl->blocks[i]) {
            sofuu_block_free(tbl->blocks[i], &tbl->schema);
        }
    }
    if (tbl->blocks) free(tbl->blocks);
    if (tbl->checkpoints) free(tbl->checkpoints);
    free(tbl);
}

int sofuu_table_insert(sofuu_table_t *tbl, const sofuu_val_t *vals, uint32_t num_vals) {
    if (!tbl || !vals) return -1;

    sofuu_val_t local_vals[SOFUU_MAX_COLS];
    uint32_t copy_count = num_vals < SOFUU_MAX_COLS ? num_vals : SOFUU_MAX_COLS;
    memcpy(local_vals, vals, copy_count * sizeof(sofuu_val_t));

    /* Auto-increment primary key assignment if not provided */
    if (tbl->schema.pk_col_idx >= 0 && (uint32_t)tbl->schema.pk_col_idx < SOFUU_MAX_COLS) {
        uint32_t pki = (uint32_t)tbl->schema.pk_col_idx;
        if (pki >= copy_count || local_vals[pki].type == SOFUU_TYPE_NULL ||
            (local_vals[pki].type == SOFUU_TYPE_INT64 && local_vals[pki].as.i64 == 0)) {
            local_vals[pki].type = SOFUU_TYPE_INT64;
            local_vals[pki].as.i64 = tbl->next_auto_inc++;
            if (pki >= copy_count) copy_count = pki + 1;
        } else if (local_vals[pki].as.i64 >= tbl->next_auto_inc) {
            tbl->next_auto_inc = local_vals[pki].as.i64 + 1;
        }
    }

    /* Find or create active block */
    sofuu_block_t *active = NULL;
    if (tbl->num_blocks > 0) {
        active = tbl->blocks[tbl->num_blocks - 1];
        if (active->row_count >= SOFUU_ROWS_PER_BLOCK) {
            active = NULL;
        }
    }

    if (!active) {
        if (tbl->num_blocks >= tbl->blocks_cap) {
            uint32_t new_cap = tbl->blocks_cap * 2;
            tbl->blocks = (sofuu_block_t **)realloc(tbl->blocks, new_cap * sizeof(sofuu_block_t *));
            tbl->checkpoints = (sofuu_block_checkpoint_t *)realloc(tbl->checkpoints, new_cap * sizeof(sofuu_block_checkpoint_t));
            tbl->blocks_cap = new_cap;
        }
        active = sofuu_block_create(&tbl->schema, tbl->num_blocks);
        if (!active) return -2;
        tbl->blocks[tbl->num_blocks++] = active;
    }

    int rc = sofuu_block_append_row(active, &tbl->schema, local_vals, copy_count);
    if (rc == 0) {
        tbl->total_rows++;
        /* Update checkpoint entry */
        uint32_t bidx = active->block_index;
        sofuu_block_checkpoint_t *cp = &tbl->checkpoints[bidx];
        cp->block_index = bidx;
        cp->row_count = active->row_count;
        cp->min_id = active->min_id;
        cp->max_id = active->max_id;
        cp->min_timestamp = active->min_timestamp;
        cp->max_timestamp = active->max_timestamp;
    }
    return rc;
}

int64_t sofuu_table_row_count(const sofuu_table_t *tbl) {
    return tbl ? (int64_t)tbl->total_rows : 0;
}

int sofuu_table_get_row(const sofuu_table_t *tbl, uint32_t row_idx, sofuu_val_t *out_vals, uint32_t max_vals) {
    if (!tbl || !out_vals) return -1;
    if (row_idx >= tbl->total_rows) return -2;

    uint32_t bidx = row_idx / SOFUU_ROWS_PER_BLOCK;
    uint32_t r = row_idx % SOFUU_ROWS_PER_BLOCK;
    if (bidx >= tbl->num_blocks) return -2;

    const sofuu_block_t *b = tbl->blocks[bidx];
    if (!b || r >= b->row_count) return -2;

    uint32_t cols_to_copy = tbl->schema.num_cols < max_vals ? tbl->schema.num_cols : max_vals;
    for (uint32_t ci = 0; ci < cols_to_copy; ci++) {
        if (b->null_map[ci] && b->null_map[ci][r]) {
            out_vals[ci].type = SOFUU_TYPE_NULL;
            continue;
        }
        sofuu_col_type_t ct = (sofuu_col_type_t)tbl->schema.cols[ci].type;
        out_vals[ci].type = (uint8_t)ct;
        switch (ct) {
            case SOFUU_TYPE_INT64:
                out_vals[ci].as.i64 = ((int64_t *)b->col_data[ci])[r];
                break;
            case SOFUU_TYPE_INT32:
                out_vals[ci].as.i32 = ((int32_t *)b->col_data[ci])[r];
                break;
            case SOFUU_TYPE_FLOAT64:
                out_vals[ci].as.f64 = ((double *)b->col_data[ci])[r];
                break;
            case SOFUU_TYPE_FLOAT32:
                out_vals[ci].as.f32 = ((float *)b->col_data[ci])[r];
                break;
            case SOFUU_TYPE_BOOL:
                out_vals[ci].as.b = ((bool *)b->col_data[ci])[r];
                break;
            case SOFUU_TYPE_TIMESTAMP:
                out_vals[ci].as.i64 = ((int64_t *)b->col_data[ci])[r];
                break;
            case SOFUU_TYPE_STRING:
            case SOFUU_TYPE_BYTES:
            case SOFUU_TYPE_JSON: {
                uint32_t o0 = b->str_offsets[ci][r];
                uint32_t o1 = b->str_offsets[ci][r + 1];
                out_vals[ci].as.str.ptr = (const char *)(b->str_heap[ci] + o0);
                out_vals[ci].as.str.len = o1 - o0;
                break;
            }
            default:
                out_vals[ci].type = SOFUU_TYPE_NULL;
                break;
        }
    }
    return (int)cols_to_copy;
}

/* ══════ Query & Checkpoint Seek Engine ══════ */

void sofuu_query_init(sofuu_query_t *q) {
    if (!q) return;
    memset(q, 0, sizeof(*q));
}

sofuu_table_t *sofuu_table_query(const sofuu_table_t *tbl, const sofuu_query_t *q) {
    if (!tbl) return NULL;
    sofuu_table_t *res = sofuu_table_create(&tbl->schema);
    if (!res) return NULL;

    uint32_t skipped = 0;
    uint32_t matched = 0;

    int filter_col_idx = -1;
    if (q && q->filter_col_name) {
        filter_col_idx = sofuu_schema_find_col(&tbl->schema, q->filter_col_name);
    }

    bool desc = q ? q->order_desc : false;
    int bi_start = desc ? ((int)tbl->num_blocks - 1) : 0;
    int bi_end = desc ? -1 : (int)tbl->num_blocks;
    int bi_step = desc ? -1 : 1;

    for (int bi = bi_start; bi != bi_end; bi += bi_step) {
        const sofuu_block_t *b = tbl->blocks[bi];
        if (!b || b->row_count == 0) continue;

        /* Checkpoint Index Skip Evaluation */
        if (q) {
            if (q->min_id != 0 && b->max_id < q->min_id) continue;
            if (q->max_id != 0 && b->min_id > q->max_id) continue;
            if (q->min_timestamp != 0 && b->max_timestamp < q->min_timestamp) continue;
            if (q->max_timestamp != 0 && b->min_timestamp > q->max_timestamp) continue;
        }

        /* Scan rows within this block */
        int r_start = desc ? ((int)b->row_count - 1) : 0;
        int r_end = desc ? -1 : (int)b->row_count;
        int r_step = desc ? -1 : 1;

        for (int r = r_start; r != r_end; r += r_step) {
            if (q && q->min_id != 0 && tbl->schema.pk_col_idx >= 0) {
                int64_t val = ((int64_t *)b->col_data[tbl->schema.pk_col_idx])[r];
                if (val < q->min_id) continue;
            }
            if (q && q->max_id != 0 && tbl->schema.pk_col_idx >= 0) {
                int64_t val = ((int64_t *)b->col_data[tbl->schema.pk_col_idx])[r];
                if (val > q->max_id) continue;
            }
            if (q && q->min_timestamp != 0 && tbl->schema.timestamp_col_idx >= 0) {
                int64_t val = ((int64_t *)b->col_data[tbl->schema.timestamp_col_idx])[r];
                if (val < q->min_timestamp) continue;
            }
            if (q && q->max_timestamp != 0 && tbl->schema.timestamp_col_idx >= 0) {
                int64_t val = ((int64_t *)b->col_data[tbl->schema.timestamp_col_idx])[r];
                if (val > q->max_timestamp) continue;
            }

            /* Custom string equality filter */
            if (filter_col_idx >= 0 && q && q->filter_val.type == SOFUU_TYPE_STRING) {
                uint32_t o0 = b->str_offsets[filter_col_idx][r];
                uint32_t o1 = b->str_offsets[filter_col_idx][r + 1];
                uint32_t len = o1 - o0;
                const char *str = (const char *)(b->str_heap[filter_col_idx] + o0);
                if (len != q->filter_val.as.str.len ||
                    strncmp(str, q->filter_val.as.str.ptr, len) != 0) {
                    continue;
                }
            }

            /* Offset check */
            if (q && q->offset > 0 && skipped < q->offset) {
                skipped++;
                continue;
            }

            /* Reconstruct row values */
            sofuu_val_t row_vals[SOFUU_MAX_COLS];
            for (uint32_t ci = 0; ci < tbl->schema.num_cols; ci++) {
                if (b->null_map[ci][r]) {
                    row_vals[ci].type = SOFUU_TYPE_NULL;
                    continue;
                }
                sofuu_col_type_t ct = (sofuu_col_type_t)tbl->schema.cols[ci].type;
                row_vals[ci].type = (uint8_t)ct;
                switch (ct) {
                    case SOFUU_TYPE_INT64:
                        row_vals[ci].as.i64 = ((int64_t *)b->col_data[ci])[r];
                        break;
                    case SOFUU_TYPE_INT32:
                        row_vals[ci].as.i32 = ((int32_t *)b->col_data[ci])[r];
                        break;
                    case SOFUU_TYPE_FLOAT64:
                        row_vals[ci].as.f64 = ((double *)b->col_data[ci])[r];
                        break;
                    case SOFUU_TYPE_FLOAT32:
                        row_vals[ci].as.f32 = ((float *)b->col_data[ci])[r];
                        break;
                    case SOFUU_TYPE_BOOL:
                        row_vals[ci].as.b = ((bool *)b->col_data[ci])[r];
                        break;
                    case SOFUU_TYPE_TIMESTAMP:
                        row_vals[ci].as.i64 = ((int64_t *)b->col_data[ci])[r];
                        break;
                    case SOFUU_TYPE_STRING:
                    case SOFUU_TYPE_BYTES:
                    case SOFUU_TYPE_JSON: {
                        uint32_t o0 = b->str_offsets[ci][r];
                        uint32_t o1 = b->str_offsets[ci][r + 1];
                        row_vals[ci].as.str.ptr = (const char *)(b->str_heap[ci] + o0);
                        row_vals[ci].as.str.len = o1 - o0;
                        break;
                    }
                    default:
                        row_vals[ci].type = SOFUU_TYPE_NULL;
                        break;
                }
            }

            sofuu_table_insert(res, row_vals, tbl->schema.num_cols);
            matched++;

            if (q && q->limit > 0 && matched >= q->limit) {
                return res;
            }
        }
    }

    return res;
}

/* ══════ QTSQ Container Pack / Unpack ══════ */

int sofuu_table_write_qtsq(const sofuu_table_t *tbl, const char *filepath, const char *password) {
    if (!tbl || !filepath) return -1;

    qtsq_context_t *container = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
    if (!container) return -2;

    int r = qtsq_container_create(container);
    if (r != QTSQ_OK) {
        free(container);
        return r;
    }
    container->codec = QTSQ_CODEC_QTC;

    /* Stream 1: Schema Definition & Column Metadata */
    qtsq_context_t *sub_meta = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
    if (sub_meta) {
        if (sofuu_pack_substream(sub_meta, (const uint8_t *)&tbl->schema, sizeof(sofuu_schema_t)) == QTSQ_OK) {
            qtsq_container_add_stream(container, sub_meta, "meta/schema");
        }
        qtsq_free(sub_meta);
        free(sub_meta);
    }

    /* Stream 2: Checkpoint Seek Index */
    if (tbl->num_blocks > 0) {
        size_t chk_size = sizeof(sofuu_block_checkpoint_t) * tbl->num_blocks;
        sofuu_block_checkpoint_t *chk_arr = (sofuu_block_checkpoint_t *)malloc(chk_size);
        if (chk_arr) {
            for (uint32_t i = 0; i < tbl->num_blocks; i++) {
                const sofuu_block_t *b = tbl->blocks[i];
                chk_arr[i].block_index = i;
                chk_arr[i].row_count = b ? b->row_count : 0;
                chk_arr[i].min_id = b ? b->min_id : 0;
                chk_arr[i].max_id = b ? b->max_id : 0;
                chk_arr[i].min_timestamp = b ? b->min_timestamp : 0;
                chk_arr[i].max_timestamp = b ? b->max_timestamp : 0;
                chk_arr[i].payload_bytes = 0;
            }
            qtsq_context_t *sub_chk = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
            if (sub_chk) {
                if (sofuu_pack_substream(sub_chk, (const uint8_t *)chk_arr, chk_size) == QTSQ_OK) {
                    qtsq_container_add_stream(container, sub_chk, "index/checkpoints");
                }
                qtsq_free(sub_chk);
                free(sub_chk);
            }
            free(chk_arr);
        }
    }

    /* Streams 3..N: Serialized Columnar Blocks */
    for (uint32_t i = 0; i < tbl->num_blocks; i++) {
        const sofuu_block_t *b = tbl->blocks[i];
        if (!b) continue;

        uint8_t *block_buf = NULL;
        size_t block_size = 0;
        if (sofuu_block_serialize(b, &tbl->schema, &block_buf, &block_size) == 0 && block_buf) {
            char blk_name[64];
            snprintf(blk_name, sizeof(blk_name), "table/%s/blk_%06u", tbl->schema.table_name, i);

            qtsq_context_t *sub_blk = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
            if (sub_blk) {
                if (sofuu_pack_substream(sub_blk, block_buf, block_size) == QTSQ_OK) {
                    qtsq_container_add_stream(container, sub_blk, blk_name);
                }
                qtsq_free(sub_blk);
                free(sub_blk);
            }
            free(block_buf);
        }
    }

    if (password && strlen(password) > 0) {
        r = qtsq_container_pack(container);
        if (r == QTSQ_OK) {
            r = qtsq_vault_encrypt_password(container, password);
        }
        if (r == QTSQ_OK) {
            r = qtsq_write(container, filepath);
        }
    } else {
        r = qtsq_container_pack_write(container, filepath, 1);
    }

    qtsq_free(container);
    free(container);
    return r;
}

sofuu_table_t *sofuu_table_read_qtsq(const char *filepath, const char *table_name, const char *password) {
    if (!filepath) return NULL;

    qtsq_context_t *container = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
    if (!container) return NULL;

    int r = qtsq_init(container);
    if (r == QTSQ_OK) {
        r = qtsq_read(container, filepath);
    }
    if (r == QTSQ_OK && container->is_encrypted) {
        if (password) {
            r = qtsq_vault_decrypt_password(container, password);
        } else {
            r = -1;
        }
    }
    if (r != QTSQ_OK) {
        qtsq_free(container);
        free(container);
        return NULL;
    }

    uint32_t stream_count = 0;
    qtsq_container_get_count(container, &stream_count);

    sofuu_schema_t schema;
    sofuu_schema_init(&schema, table_name ? table_name : "default");

    /* First pass: read metadata schema */
    for (uint32_t i = 0; i < stream_count; i++) {
        char name[256];
        if (qtsq_container_get_stream_name(container, i, name, sizeof(name)) == QTSQ_OK) {
            if (strcmp(name, "meta/schema") == 0) {
                qtsq_context_t *sub = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
                if (sub && qtsq_container_get_stream(container, i, sub, name, sizeof(name)) == QTSQ_OK) {
                    uint8_t *raw = NULL;
                    size_t raw_size = 0;
                    if (sofuu_unpack_substream(sub, &raw, &raw_size) == QTSQ_OK && raw) {
                        if (raw_size >= sizeof(sofuu_schema_t)) {
                            memcpy(&schema, raw, sizeof(sofuu_schema_t));
                        }
                        free(raw);
                    }
                    qtsq_free(sub);
                    free(sub);
                }
                break;
            }
        }
    }

    sofuu_table_t *tbl = sofuu_table_create(&schema);
    if (!tbl) {
        qtsq_free(container);
        free(container);
        return NULL;
    }

    /* Second pass: read block streams */
    for (uint32_t i = 0; i < stream_count; i++) {
        char name[256];
        if (qtsq_container_get_stream_name(container, i, name, sizeof(name)) == QTSQ_OK) {
            if (strstr(name, "blk_") != NULL) {
                qtsq_context_t *sub = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
                if (sub && qtsq_container_get_stream(container, i, sub, name, sizeof(name)) == QTSQ_OK) {
                    uint8_t *raw = NULL;
                    size_t raw_size = 0;
                    if (sofuu_unpack_substream(sub, &raw, &raw_size) == QTSQ_OK && raw) {
                        sofuu_block_t *b = sofuu_block_deserialize(raw, raw_size, &tbl->schema);
                        if (b) {
                            if (tbl->num_blocks >= tbl->blocks_cap) {
                                uint32_t new_cap = tbl->blocks_cap * 2;
                                tbl->blocks = (sofuu_block_t **)realloc(tbl->blocks, new_cap * sizeof(sofuu_block_t *));
                                tbl->checkpoints = (sofuu_block_checkpoint_t *)realloc(tbl->checkpoints, new_cap * sizeof(sofuu_block_checkpoint_t));
                                tbl->blocks_cap = new_cap;
                            }
                            tbl->blocks[tbl->num_blocks++] = b;
                            tbl->total_rows += b->row_count;
                        }
                        free(raw);
                    }
                    qtsq_free(sub);
                    free(sub);
                }
            }
        }
    }

    /* Third pass: read checkpoint index stream if present */
    for (uint32_t i = 0; i < stream_count; i++) {
        char name[256];
        if (qtsq_container_get_stream_name(container, i, name, sizeof(name)) == QTSQ_OK) {
            if (strcmp(name, "index/checkpoints") == 0) {
                qtsq_context_t *sub = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
                if (sub && qtsq_container_get_stream(container, i, sub, name, sizeof(name)) == QTSQ_OK) {
                    uint8_t *raw = NULL;
                    size_t raw_size = 0;
                    if (sofuu_unpack_substream(sub, &raw, &raw_size) == QTSQ_OK && raw) {
                        uint32_t count = raw_size / sizeof(sofuu_block_checkpoint_t);
                        if (count > 0) {
                            if (tbl->blocks_cap < count) {
                                tbl->checkpoints = (sofuu_block_checkpoint_t *)realloc(tbl->checkpoints, count * sizeof(sofuu_block_checkpoint_t));
                            }
                            memcpy(tbl->checkpoints, raw, count * sizeof(sofuu_block_checkpoint_t));
                            tbl->checkpoints_count = count;
                        }
                        free(raw);
                    }
                    qtsq_free(sub);
                    free(sub);
                }
                break;
            }
        }
    }

    qtsq_free(container);
    free(container);
    return tbl;
}

/* Direct fast checkpoint-indexed query on a QTSQ container file without loading full table */
sofuu_table_t *sofuu_table_query_qtsq(const char *filepath, const char *table_name, const char *password, const sofuu_query_t *q) {
    if (!filepath) return NULL;

    qtsq_context_t *container = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
    if (!container) return NULL;

    int r = qtsq_init(container);
    if (r == QTSQ_OK) {
        r = qtsq_read(container, filepath);
    }
    if (r == QTSQ_OK && container->is_encrypted) {
        if (password) {
            r = qtsq_vault_decrypt_password(container, password);
        } else {
            r = -1;
        }
    }
    if (r != QTSQ_OK) {
        qtsq_free(container);
        free(container);
        return NULL;
    }

    uint32_t stream_count = 0;
    qtsq_container_get_count(container, &stream_count);

    sofuu_schema_t schema;
    sofuu_schema_init(&schema, table_name ? table_name : "default");

    /* Step 1: Read metadata schema */
    for (uint32_t i = 0; i < stream_count; i++) {
        char name[256];
        if (qtsq_container_get_stream_name(container, i, name, sizeof(name)) == QTSQ_OK) {
            if (strcmp(name, "meta/schema") == 0) {
                qtsq_context_t *sub = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
                if (sub && qtsq_container_get_stream(container, i, sub, name, sizeof(name)) == QTSQ_OK) {
                    uint8_t *raw = NULL;
                    size_t raw_size = 0;
                    if (sofuu_unpack_substream(sub, &raw, &raw_size) == QTSQ_OK && raw) {
                        if (raw_size >= sizeof(sofuu_schema_t)) {
                            memcpy(&schema, raw, sizeof(sofuu_schema_t));
                        }
                        free(raw);
                    }
                    qtsq_free(sub);
                    free(sub);
                }
                break;
            }
        }
    }

    sofuu_table_t *res = sofuu_table_create(&schema);
    if (!res) {
        qtsq_free(container);
        free(container);
        return NULL;
    }

    /* Step 2: Read Checkpoints */
    sofuu_block_checkpoint_t *checkpoints = NULL;
    uint32_t num_checkpoints = 0;

    for (uint32_t i = 0; i < stream_count; i++) {
        char name[256];
        if (qtsq_container_get_stream_name(container, i, name, sizeof(name)) == QTSQ_OK) {
            if (strcmp(name, "index/checkpoints") == 0) {
                qtsq_context_t *sub = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
                if (sub && qtsq_container_get_stream(container, i, sub, name, sizeof(name)) == QTSQ_OK) {
                    uint8_t *raw = NULL;
                    size_t raw_size = 0;
                    if (sofuu_unpack_substream(sub, &raw, &raw_size) == QTSQ_OK && raw) {
                        uint32_t count = raw_size / sizeof(sofuu_block_checkpoint_t);
                        if (count > 0) {
                            checkpoints = (sofuu_block_checkpoint_t *)malloc(raw_size);
                            if (checkpoints) {
                                memcpy(checkpoints, raw, raw_size);
                                num_checkpoints = count;
                            }
                        }
                        free(raw);
                    }
                    qtsq_free(sub);
                    free(sub);
                }
                break;
            }
        }
    }

    int filter_col_idx = -1;
    if (q && q->filter_col_name) {
        filter_col_idx = sofuu_schema_find_col(&schema, q->filter_col_name);
    }

    uint32_t rows_skipped_offset = 0;
    uint32_t target_offset = q ? q->offset : 0;
    uint32_t target_limit = q ? q->limit : 0;
    bool desc = q ? q->order_desc : false;

    if (num_checkpoints > 0) {
        int start_idx = desc ? ((int)num_checkpoints - 1) : 0;
        int end_idx = desc ? -1 : (int)num_checkpoints;
        int step = desc ? -1 : 1;

        for (int ci = start_idx; ci != end_idx; ci += step) {
            const sofuu_block_checkpoint_t *chk = &checkpoints[ci];

            /* Checkpoint Skip Evaluation: Skip block WITHOUT decompressing if out of range */
            if (q) {
                if (q->min_id != 0 && chk->max_id < q->min_id) continue;
                if (q->max_id != 0 && chk->min_id > q->max_id) continue;
                if (q->min_timestamp != 0 && chk->max_timestamp < q->min_timestamp) continue;
                if (q->max_timestamp != 0 && chk->min_timestamp > q->max_timestamp) continue;
            }

            /* Block qualifies! Locate stream in container */
            char target_blk_name[64];
            snprintf(target_blk_name, sizeof(target_blk_name), "blk_%06u", chk->block_index);

            for (uint32_t si = 0; si < stream_count; si++) {
                char sname[256];
                if (qtsq_container_get_stream_name(container, si, sname, sizeof(sname)) == QTSQ_OK) {
                    if (strstr(sname, target_blk_name) != NULL) {
                        /* Decompress ONLY this matching block */
                        qtsq_context_t *sub = (qtsq_context_t *)calloc(1, sizeof(qtsq_context_t));
                        if (sub && qtsq_container_get_stream(container, si, sub, sname, sizeof(sname)) == QTSQ_OK) {
                            uint8_t *raw = NULL;
                            size_t raw_size = 0;
                            if (sofuu_unpack_substream(sub, &raw, &raw_size) == QTSQ_OK && raw) {
                                sofuu_block_t *b = sofuu_block_deserialize(raw, raw_size, &schema);
                                if (b) {
                                    int r_start = desc ? ((int)b->row_count - 1) : 0;
                                    int r_end = desc ? -1 : (int)b->row_count;
                                    int r_step = desc ? -1 : 1;

                                    for (int r = r_start; r != r_end; r += r_step) {
                                        if (q && q->min_id != 0 && schema.pk_col_idx >= 0) {
                                            int64_t val = ((int64_t *)b->col_data[schema.pk_col_idx])[r];
                                            if (val < q->min_id) continue;
                                        }
                                        if (q && q->max_id != 0 && schema.pk_col_idx >= 0) {
                                            int64_t val = ((int64_t *)b->col_data[schema.pk_col_idx])[r];
                                            if (val > q->max_id) continue;
                                        }
                                        if (q && q->min_timestamp != 0 && schema.timestamp_col_idx >= 0) {
                                            int64_t val = ((int64_t *)b->col_data[schema.timestamp_col_idx])[r];
                                            if (val < q->min_timestamp) continue;
                                        }
                                        if (q && q->max_timestamp != 0 && schema.timestamp_col_idx >= 0) {
                                            int64_t val = ((int64_t *)b->col_data[schema.timestamp_col_idx])[r];
                                            if (val > q->max_timestamp) continue;
                                        }
                                        if (filter_col_idx >= 0 && q && q->filter_val.type == SOFUU_TYPE_STRING) {
                                            uint32_t o0 = b->str_offsets[filter_col_idx][r];
                                            uint32_t o1 = b->str_offsets[filter_col_idx][r + 1];
                                            uint32_t len = o1 - o0;
                                            const char *str = (const char *)(b->str_heap[filter_col_idx] + o0);
                                            if (len != q->filter_val.as.str.len ||
                                                (len > 0 && memcmp(str, q->filter_val.as.str.ptr, len) != 0)) {
                                                continue;
                                            }
                                        }

                                        // Apply offset
                                        if (rows_skipped_offset < target_offset) {
                                            rows_skipped_offset++;
                                            continue;
                                        }

                                        // Extract row values and append to result table
                                        sofuu_val_t row_vals[SOFUU_MAX_COLS];
                                        for (uint32_t c = 0; c < schema.num_cols; c++) {
                                            row_vals[c].type = schema.cols[c].type;
                                            if (b->null_map[c] && b->null_map[c][r]) {
                                                row_vals[c].type = SOFUU_TYPE_NULL;
                                                continue;
                                            }
                                            switch (schema.cols[c].type) {
                                                case SOFUU_TYPE_INT64:
                                                case SOFUU_TYPE_TIMESTAMP:
                                                    row_vals[c].as.i64 = ((int64_t *)b->col_data[c])[r];
                                                    break;
                                                case SOFUU_TYPE_INT32:
                                                    row_vals[c].as.i32 = ((int32_t *)b->col_data[c])[r];
                                                    break;
                                                case SOFUU_TYPE_FLOAT64:
                                                    row_vals[c].as.f64 = ((double *)b->col_data[c])[r];
                                                    break;
                                                case SOFUU_TYPE_FLOAT32:
                                                    row_vals[c].as.f32 = ((float *)b->col_data[c])[r];
                                                    break;
                                                case SOFUU_TYPE_BOOL:
                                                    row_vals[c].as.b = ((bool *)b->col_data[c])[r];
                                                    break;
                                                case SOFUU_TYPE_STRING:
                                                case SOFUU_TYPE_JSON: {
                                                    uint32_t o0 = b->str_offsets[c][r];
                                                    uint32_t o1 = b->str_offsets[c][r + 1];
                                                    row_vals[c].as.str.ptr = (const char *)(b->str_heap[c] + o0);
                                                    row_vals[c].as.str.len = o1 - o0;
                                                    break;
                                                }
                                                case SOFUU_TYPE_BYTES: {
                                                    uint32_t o0 = b->str_offsets[c][r];
                                                    uint32_t o1 = b->str_offsets[c][r + 1];
                                                    row_vals[c].as.bytes.ptr = (const uint8_t *)(b->str_heap[c] + o0);
                                                    row_vals[c].as.bytes.len = o1 - o0;
                                                    break;
                                                }
                                                default:
                                                    row_vals[c].type = SOFUU_TYPE_NULL;
                                                    break;
                                            }
                                        }

                                        sofuu_table_insert(res, row_vals, schema.num_cols);

                                        // Check limit
                                        if (target_limit > 0 && res->total_rows >= target_limit) {
                                            sofuu_block_free(b, &schema);
                                            free(raw);
                                            qtsq_free(sub);
                                            free(sub);
                                            goto query_done;
                                        }
                                    }

                                    sofuu_block_free(b, &schema);
                                }
                                free(raw);
                            }
                            qtsq_free(sub);
                            free(sub);
                        }
                        break;
                    }
                }
            }
        }
    } else {
        // Fallback: no checkpoints stream found, read entire table and run in-memory query
        sofuu_table_free(res);
        qtsq_free(container);
        free(container);
        sofuu_table_t *full = sofuu_table_read_qtsq(filepath, table_name, password);
        if (!full) return NULL;
        res = sofuu_table_query(full, q);
        sofuu_table_free(full);
        return res;
    }

query_done:
    if (checkpoints) free(checkpoints);
    qtsq_free(container);
    free(container);
    return res;
}
