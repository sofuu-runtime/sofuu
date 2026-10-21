/*
 * sofuu_qtsq_table.h — Sofuu-Optimized Strict Schema Columnar Table Engine
 * 
 * Part of the Quantum Tensor Sequence (QTSQ) suite tailored for Sofuu.
 * Provides high-speed columnar buffer packing, bounded block chunking,
 * and checkpoint indexing for zero-dependency persistent structured tables.
 *
 * Author: Haruhito
 */

#ifndef SOFUU_QTSQ_TABLE_H
#define SOFUU_QTSQ_TABLE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SOFUU_TABLE_MAGIC      0x53465554 /* 'SFUT' */
#define SOFUU_TABLE_VERSION    1
#define SOFUU_MAX_COLS         32
#define SOFUU_MAX_NAME_LEN     64
#define SOFUU_ROWS_PER_BLOCK   128

/* Column Data Types */
typedef enum {
    SOFUU_TYPE_NULL      = 0,
    SOFUU_TYPE_INT64     = 1,
    SOFUU_TYPE_INT32     = 2,
    SOFUU_TYPE_FLOAT64   = 3,
    SOFUU_TYPE_FLOAT32   = 4,
    SOFUU_TYPE_BOOL      = 5,
    SOFUU_TYPE_TIMESTAMP = 6, /* Milliseconds since Unix epoch */
    SOFUU_TYPE_STRING    = 7, /* UTF-8 text */
    SOFUU_TYPE_BYTES     = 8, /* Raw binary payload */
    SOFUU_TYPE_JSON      = 9  /* Serialized JSON document */
} sofuu_col_type_t;

/* Column Flags */
#define SOFUU_COL_PRIMARY_KEY  (1 << 0)
#define SOFUU_COL_REQUIRED     (1 << 1)
#define SOFUU_COL_INDEXED      (1 << 2)
#define SOFUU_COL_AUTO_INC     (1 << 3)

/* Column Definition */
typedef struct {
    char            name[SOFUU_MAX_NAME_LEN];
    uint8_t         type;       /* sofuu_col_type_t */
    uint8_t         flags;      /* SOFUU_COL_* */
    uint16_t        reserved;
} sofuu_col_def_t;

/* Table Schema */
typedef struct {
    char            table_name[SOFUU_MAX_NAME_LEN];
    uint32_t        num_cols;
    sofuu_col_def_t cols[SOFUU_MAX_COLS];
    int32_t         pk_col_idx;        /* -1 if no primary key */
    int32_t         timestamp_col_idx; /* -1 if no timestamp column */
} sofuu_schema_t;

/* Typed Value Representation */
typedef struct {
    uint8_t type; /* sofuu_col_type_t */
    union {
        int64_t  i64;
        int32_t  i32;
        double   f64;
        float    f32;
        bool     b;
        struct {
            const char *ptr;
            uint32_t    len;
        } str;
        struct {
            const uint8_t *ptr;
            uint32_t       len;
        } bytes;
    } as;
} sofuu_val_t;

/* Checkpoint Index Entry for a single Block */
typedef struct {
    uint32_t block_index;
    uint32_t row_count;
    int64_t  min_id;
    int64_t  max_id;
    int64_t  min_timestamp;
    int64_t  max_timestamp;
    uint64_t payload_bytes;
} sofuu_block_checkpoint_t;

/* Columnar Storage for one Block (Up to SOFUU_ROWS_PER_BLOCK rows) */
typedef struct {
    uint32_t block_index;
    uint32_t row_count;
    int64_t  min_id;
    int64_t  max_id;
    int64_t  min_timestamp;
    int64_t  max_timestamp;

    /* Column data arrays */
    void     *col_data[SOFUU_MAX_COLS];     /* Contiguous arrays for fixed-width */
    uint32_t *str_offsets[SOFUU_MAX_COLS];  /* Array of row_count + 1 offsets */
    uint8_t  *str_heap[SOFUU_MAX_COLS];     /* Packed byte pool */
    size_t    str_heap_len[SOFUU_MAX_COLS];
    size_t    str_heap_cap[SOFUU_MAX_COLS];
    bool     *null_map[SOFUU_MAX_COLS];     /* true if null */
} sofuu_block_t;

/* Complete In-Memory Table */
typedef struct {
    sofuu_schema_t            schema;
    uint32_t                  total_rows;
    int64_t                   next_auto_inc;
    
    uint32_t                  num_blocks;
    uint32_t                  blocks_cap;
    sofuu_block_t           **blocks;

    uint32_t                  checkpoints_count;
    sofuu_block_checkpoint_t *checkpoints;
} sofuu_table_t;

/* Query Options */
typedef struct {
    int64_t  min_id;
    int64_t  max_id;
    int64_t  min_timestamp;
    int64_t  max_timestamp;
    const char *filter_col_name;
    sofuu_val_t filter_val;
    uint32_t limit;
    uint32_t offset;
    bool     order_desc;
} sofuu_query_t;

/* ══════ Schema Lifecycle ══════ */
void sofuu_schema_init(sofuu_schema_t *schema, const char *table_name);
int  sofuu_schema_add_col(sofuu_schema_t *schema, const char *name, sofuu_col_type_t type, uint8_t flags);
int  sofuu_schema_find_col(const sofuu_schema_t *schema, const char *name);

/* ══════ Block Operations ══════ */
sofuu_block_t *sofuu_block_create(const sofuu_schema_t *schema, uint32_t block_idx);
void           sofuu_block_free(sofuu_block_t *block, const sofuu_schema_t *schema);
int            sofuu_block_append_row(sofuu_block_t *block, const sofuu_schema_t *schema, const sofuu_val_t *vals, uint32_t num_vals);
int            sofuu_block_serialize(const sofuu_block_t *block, const sofuu_schema_t *schema, uint8_t **out_buf, size_t *out_size);
sofuu_block_t *sofuu_block_deserialize(const uint8_t *buf, size_t size, const sofuu_schema_t *schema);

/* ══════ Table Lifecycle & Operations ══════ */
sofuu_table_t *sofuu_table_create(const sofuu_schema_t *schema);
void           sofuu_table_free(sofuu_table_t *tbl);
int            sofuu_table_insert(sofuu_table_t *tbl, const sofuu_val_t *vals, uint32_t num_vals);
int64_t        sofuu_table_row_count(const sofuu_table_t *tbl);
int            sofuu_table_get_row(const sofuu_table_t *tbl, uint32_t row_idx, sofuu_val_t *out_vals, uint32_t max_vals);

/* ══════ Query & Checkpoint Seek Engine ══════ */
void           sofuu_query_init(sofuu_query_t *q);
sofuu_table_t *sofuu_table_query(const sofuu_table_t *tbl, const sofuu_query_t *q);

/* ══════ QTSQ Container Pack / Unpack ══════ */
/* Packs the complete table into an encrypted or plaintext QTSQ container.
 * If password is NULL or empty, writes using qtsq_write_plaintext.
 * If password is provided, encrypts with AES-256-GCM vault. */
int sofuu_table_write_qtsq(const sofuu_table_t *tbl, const char *filepath, const char *password);

/* Reads a table from an existing QTSQ container */
sofuu_table_t *sofuu_table_read_qtsq(const char *filepath, const char *table_name, const char *password);

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_QTSQ_TABLE_H */
