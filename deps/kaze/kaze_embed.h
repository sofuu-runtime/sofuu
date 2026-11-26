/* kaze_embed.h — the public C99 surface of libsofuu_kaze.a.
 *
 * One header covers the two halves of the embedded engine:
 *
 *   kz_embed_*  a stateful compiler handle (parse/check/compile/emit-c), and
 *   kz_vm_*     a resumable, fuel-metered VM over the resulting bytecode.
 *
 * There is no ambient I/O and no process termination in either half. A trap is
 * returned as a code + line; the caller decides what to do next. This is what
 * lets an embedded script fail without taking the QuickJS event loop with it.
 */
#ifndef KAZE_EMBED_H
#define KAZE_EMBED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kaze_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KZ_SLICE_DONE    = 0,  /* entry returned or hooks ticked cleanly */
    KZ_SLICE_YIELD   = 1,  /* instruction slice elapsed; call again */
    KZ_SLICE_CAP_REQ = 2,  /* paused waiting on an external capability */
    KZ_SLICE_ERR     = 3   /* bounded trap (OOM, fuel, div0, safety, ...) */
} kz_slice_status_t;

/* The result shape shared by the compiler and the VM. `text` points into the
 * handle (or VM) and stays valid until the next call or destroy. */
typedef struct kz_embed_result_t {
    bool        ok;
    const char *text;
    size_t      len;
    const char *err_code;
    int         err_line;
} kz_embed_result_t;

/* ── compiler handle ─────────────────────────────────────────────────── */

typedef struct kz_compiler_t kz_compiler_t;

/* A handle owns its arena, but the generated compiler's file-scope scratch is
 * process-global: callers must serialize kz_embed_* calls (the Rust door does).
 * VMs have no such global and may run concurrently. */
/* `arena_bytes` is the compiler's private bump arena. Zero selects a 256 MiB
 * default, enough for the self-hosted compiler plus a program. */
kz_compiler_t *kz_embed_compiler_new(size_t arena_bytes);
void           kz_embed_compiler_free(kz_compiler_t *c);

kz_embed_result_t kz_embed_check(kz_compiler_t *c, const char *src, size_t len);
/* Lowercase hex, exactly the bytes `kaze compile` prints. */
kz_embed_result_t kz_embed_compile(kz_compiler_t *c, const char *src, size_t len);
kz_embed_result_t kz_embed_emit_c(kz_compiler_t *c, const char *src, size_t len, bool standalone);
kz_embed_result_t kz_embed_to_machine(kz_compiler_t *c, const char *src, size_t len);
kz_embed_result_t kz_embed_to_human(kz_compiler_t *c, const char *src, size_t len);
kz_embed_result_t kz_embed_version(kz_compiler_t *c);

/* ── resumable VM ────────────────────────────────────────────────────── */

typedef struct kz_vm_t kz_vm_t;

/* What the host may answer while a capability call is in flight. */
typedef enum {
    KZ_CAP_OK     = 0,  /* out_json holds a JSON value */
    KZ_CAP_ASYNC  = 1,  /* an external driver will answer later */
    KZ_CAP_FAILED = 2   /* out_err holds "CODE: message" */
} kz_cap_status_t;

#define KZ_CAP_JSON_MAX 1048576u

typedef struct kz_host_cbs_t {
    void *ctx;

    /* Ask the host to serve one capability call. `args_json` is a JSON array.
     * Return KZ_CAP_ASYNC when the answer will arrive through
     * kz_vm_supply_cap_result(). */
    kz_cap_status_t (*cap_dispatch)(
        void *ctx,
        const char *family,
        const char *verb,
        const char *args_json,
        char *out_json,
        size_t out_cap,
        size_t *out_len,
        char *out_err,
        size_t err_cap);

    /* R7 / argument validation before dispatch. Return 1 when allowed; write
     * "CODE: message" to out_err and return 0 when denied. Optional. */
    int (*authorise)(
        void *ctx,
        const char *family,
        const char *verb,
        const char *args_json,
        char *out_err,
        size_t err_cap);

    /* Called at every FUEL_TICK. Non-zero means the host deadline fired. */
    int (*poll_deadline)(void *ctx);

    /* Optional typed freshness gate. Non-zero means the reading is fresh. */
    int (*freshness_check)(void *ctx, const char *family, const char *field);

    /* Optional `drone.link_silent_ms()` source. Return 1 and set *out_ms when
     * the link exists, 0 when there is no live link. */
    int (*link_silent_ms)(void *ctx, uint64_t *out_ms);
} kz_host_cbs_t;

kz_vm_t *kz_vm_create(const uint8_t *bc, size_t bc_len, size_t arena_bytes, uint64_t fuel_limit);
void     kz_vm_destroy(kz_vm_t *vm);
int      kz_vm_set_host(kz_vm_t *vm, const kz_host_cbs_t *cbs);
/* The collection budget (arrays, maps, points), separate from the arena. */
int      kz_vm_set_heap_limit(kz_vm_t *vm, size_t heap_bytes);

size_t      kz_vm_item_count(const kz_vm_t *vm);
const char *kz_vm_item_name(const kz_vm_t *vm, size_t index);
const char *kz_vm_item_tag(const kz_vm_t *vm, size_t index);
size_t      kz_vm_item_param_count(const kz_vm_t *vm, size_t index);
const char *kz_vm_item_param_name(const kz_vm_t *vm, size_t index, size_t param);
double      kz_vm_item_period_ms(const kz_vm_t *vm, size_t index);
uint32_t    kz_vm_item_line(const kz_vm_t *vm, size_t index);
size_t      kz_vm_needs_count(const kz_vm_t *vm);
const char *kz_vm_need(const kz_vm_t *vm, size_t index);
size_t      kz_vm_const_count(const kz_vm_t *vm);
size_t      kz_vm_cap_count(const kz_vm_t *vm);
size_t      kz_vm_code_bytes(const kz_vm_t *vm);

/* Start one item. `args_json` is a JSON array whose length must equal the
 * item's parameter count. It may be NULL for a zero-parameter item. */
int kz_vm_start(kz_vm_t *vm, size_t item_index, const char *args_json);

/* Step at most max_steps bytecode instructions. A wide instruction is one
 * step, and a pausing capability is never replayed when the caller resumes. */
kz_slice_status_t kz_vm_step_slice(kz_vm_t *vm, size_t max_steps,
                                   const char **req_family, const char **req_verb,
                                   const char **req_args_json);

/* Supply the answer to a paused KZ_SLICE_CAP_REQ. Exactly one of result_json
 * and err_msg is used. Returns false when the VM is not waiting. */
bool kz_vm_supply_cap_result(kz_vm_t *vm, const char *result_json, const char *err_msg);

/* The final rendered value (for a normal return) or "nothing". */
kz_embed_result_t kz_vm_finish(const kz_vm_t *vm);
/* The final value as JSON, for a host that keeps its own value model. */
const char *kz_vm_value_json(const kz_vm_t *vm);

const char *kz_vm_error_code(const kz_vm_t *vm);
const char *kz_vm_error_message(const kz_vm_t *vm);
int         kz_vm_error_line(const kz_vm_t *vm);

uint64_t    kz_vm_instructions(const kz_vm_t *vm);
uint64_t    kz_vm_fuel_used(const kz_vm_t *vm);
const char *kz_vm_verdict_kind(const kz_vm_t *vm);   /* "allow", "reject" or NULL */
const char *kz_vm_verdict_text(const kz_vm_t *vm);
/* The capability that paused the last KZ_SLICE_CAP_REQ. */
uint32_t    kz_vm_pending_line(const kz_vm_t *vm);
const char *kz_vm_pending_family(const kz_vm_t *vm);
const char *kz_vm_pending_verb(const kz_vm_t *vm);
const char *kz_vm_pending_args_json(const kz_vm_t *vm);
const char *kz_vm_value_text(const kz_vm_t *vm);
const char *kz_vm_root(const kz_vm_t *vm);

/* The diagnostic left by a failed kz_vm_create (process-local, not per-VM). */
const char *kz_vm_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* KAZE_EMBED_H */
