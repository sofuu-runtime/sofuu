/* kaze_rt.h — the Kaze C99 runtime substrate (roadmap Phase 1).
 *
 * Why this file exists: Phase 7 lowers Kaze to portable C99 and Phase 10 retires
 * the Rust VM. Everything a lowered program needs while it runs lives here: one
 * linear arena per run, checked arithmetic, value representations, a monotonic
 * clock, a heartbeat watchdog, and signal-time actuator de-assert. The Rust
 * reference model this mirrors is src/arena.rs (linear arena, bulk free) and
 * src/value.rs (tagged value enum).
 *
 * Invariants this file keeps:
 *   - ANSI/ISO C99. Compiler intrinsics are used when present but never required.
 *   - Includes only <stdint.h> <stdbool.h> <stddef.h> <string.h> <stdio.h>
 *     <math.h> <time.h> and, unless KAZE_NO_SIGNALS is defined, <signal.h>.
 *   - Nothing in the runtime aborts or longjmps. A trap is a code plus a
 *     message delivered to a registered fallback handler; the handler decides
 *     what a caller sees next. There is exactly one heap entry point,
 *     kaze_arena_init_heap, and it is compiled out by KAZE_NO_HEAP.
 *
 * Escape hatches (all documented in docs/KAZE-RT.md):
 *   KAZE_USE_DOUBLE  force kaze_float_t to double on a 32-bit target.
 *   KAZE_NO_SIGNALS  drop the SIGINT/SIGTERM/SIGSEGV cleanup handler entirely
 *                    (freestanding and embedded builds; no <signal.h>).
 *   KAZE_NO_HEAP     drop kaze_arena_init_heap/kaze_arena_free (no malloc).
 */
#ifndef KAZE_RT_H
#define KAZE_RT_H

/* Canonical Kaze language version implemented by both standalone and embedded engines. */
#define KAZE_VERSION "1.0"

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>   /* strtod, in kz_format_number */
#include <stdio.h>
#include <math.h>
#include <time.h>
#ifndef KAZE_NO_SIGNALS
#include <signal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* Cross-architecture types                                                   */
/* ------------------------------------------------------------------------- */

/* The Rust VM pins one integer width per pointer width (64-bit hosts get
 * i64/f64, 32-bit targets get i32/f32). MCUs pay for 64-bit arithmetic in
 * cycles and code size, so the narrow form is the default on 32-bit targets.
 * MSVC does not define __SIZEOF_POINTER__; its 32-bit branch is spelled out so
 * a Windows x86 build does not silently get the 64-bit types. */
#if defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 4
#define KAZE_PTR32 1
#elif defined(_WIN32) && !defined(_WIN64) && !defined(__SIZEOF_POINTER__)
#define KAZE_PTR32 1
#endif

#if defined(KAZE_PTR32)
typedef int32_t kaze_int_t;
typedef uint32_t kaze_uint_t;
#if defined(KAZE_USE_DOUBLE)
/* A 32-bit target with KAZE_USE_DOUBLE trades f32's 4-byte footprint for f64's
 * precision. Geodesy and control loops need the mantissa; the cost is a
 * software double on cores without an FPU. */
typedef double kaze_float_t;
#else
typedef float kaze_float_t;
#endif
#define KAZE_INT_BITS 32
#else
typedef int64_t kaze_int_t;
typedef uint64_t kaze_uint_t;
typedef double kaze_float_t;
#define KAZE_INT_BITS 64
#endif

typedef size_t kaze_size_t;
typedef ptrdiff_t kaze_ptrdiff_t;

/* Fixed-width aliases. Lowered code spells these, so they are deliberately
 * short; a translation unit that already uses u8/i32 as identifiers must
 * include this header last or rename its own. */
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef float f32;
typedef double f64;
/* In C99 the bare name bool is a <stdbool.h> macro, not a type name, so the
 * typedef has to carry the kaze_ prefix. */
typedef bool kaze_bool_t;

#define KAZE_INT_MAX ((kaze_int_t)(KAZE_INT_BITS == 32 ? INT32_MAX : INT64_MAX))
#define KAZE_INT_MIN ((kaze_int_t)(KAZE_INT_BITS == 32 ? INT32_MIN : INT64_MIN))
#define KAZE_UINT_MAX ((kaze_uint_t)(KAZE_INT_BITS == 32 ? UINT32_MAX : UINT64_MAX))

/* C99 has no _Static_assert (C11). A negative array length is a hard error on
 * every compiler this runtime claims, which is the behaviour we need at build
 * time. The assert name must be unique per translation unit. */
#define KAZE_STATIC_ASSERT(cond, name) typedef char kaze_assert_##name[(cond) ? 1 : -1]

KAZE_STATIC_ASSERT(sizeof(kaze_int_t) == sizeof(void *), int_matches_pointer_width);
KAZE_STATIC_ASSERT(sizeof(float) == 4 && sizeof(double) == 8, ieee754_binary32_binary64);
KAZE_STATIC_ASSERT(sizeof(u8) == 1 && sizeof(u64) == 8, fixed_width_widths);

/* ------------------------------------------------------------------------- */
/* Trap codes and the fallback handler                                        */
/* ------------------------------------------------------------------------- */

/* The code strings are the wire contract: they are what the handler prints,
 * what a log line carries, and what a host branches on. They are also the
 * diagnostic vocabulary the Phase 2 checker names in its own errors. */
typedef enum kaze_code_t {
    KAZE_OK = 0,
    KAZE_ERR_OOM,              /* "E_KAZE_OOM" — the arena has no room left */
    KAZE_ERR_OVERFLOW,         /* "E_KAZE_OVERFLOW" — checked integer overflow */
    KAZE_ERR_DIV0,             /* "E_KAZE_DIV0" — integer or float divide by zero */
    KAZE_ERR_MATH,             /* "E_KAZE_MATH" — a non-finite float result */
    KAZE_ERR_SAFETY_VIOLATION, /* "E_SAFETY_VIOLATION" — watchdog fail-closed */
    KAZE_ERR_BADARG,           /* "E_KAZE_BADARG" — NULL/invalid runtime argument */
    KAZE_ERR_TYPE,             /* "E_KAZE_TYPE" — an operation on a value of the wrong type */
    KAZE_ERR_SHAPE,            /* "E_KAZE_SHAPE_MISMATCH" — tensor shapes do not unify */
    KAZE_ERR_QUANT,            /* "E_KAZE_QUANT" — a malformed or unsupported quant block */
    KAZE_ERR_IO,               /* "E_KAZE_IO" — a model file could not be opened or read */
    KAZE_ERR_FORMAT            /* "E_KAZE_FORMAT" — a weight file is malformed */
} kaze_code_t;

/* A handler receives the code string and a human message and returns an int
 * that the trapping wrapper hands back to its caller. It must not abort unless
 * the embedder wants a crash: the runtime keeps running after a trap. */
typedef int (*kaze_fallback_fn)(const char *code, const char *message);

/* Install a handler; NULL restores the default. Global by design: a trap can
 * fire on any path and threading a context pointer through every wrapper would
 * cost more than it buys for a single-run VM. */
void kaze_set_fallback_handler(kaze_fallback_fn fn);
kaze_fallback_fn kaze_get_fallback_handler(void);

/* The default handler: prints "CODE: message" to stderr and returns 1. */
int kaze_default_fallback(const char *code, const char *message);

/* The canonical string for a code ("E_KAZE_OOM", ...); never NULL. */
const char *kaze_code_name(kaze_code_t code);

/* Structured trap: forwards to the registered handler and returns its status.
 * Public so lowered code and hosts can raise the same codes the runtime does. */
int kaze_trap(kaze_code_t code, const char *message);

/* ------------------------------------------------------------------------- */
/* Linear bump arena                                                          */
/* ------------------------------------------------------------------------- */

/* One arena per run, made of one or more blocks. Allocation bumps a byte
 * cursor inside the current block; there is no per-object free and no
 * fragmentation inside a block. When a request does not fit the current
 * block, a heap arena chains another block instead of failing the run, so the
 * initial capacity is a starting point rather than an upper bound. Reset
 * returns the run's memory at once, which is the property a 10 Hz controller
 * needs: allocation is bounded work and the reset is a chain walk, not a
 * per-object sweep.
 *
 * A caller-owned arena (kaze_arena_init) is the exception: it owns exactly the
 * storage the caller handed it, never allocates another block, and reports
 * E_KAZE_OOM when that buffer is full. */
typedef struct kaze_arena_block_t {
    struct kaze_arena_block_t *next; /* next (newer) block; NULL at the current one */
    void *raw;                       /* malloc pointer for this block: what free() takes */
    uint8_t *data;                   /* 8-byte aligned usable bytes */
    size_t cap;                      /* usable bytes at data */
    size_t used;                     /* bytes handed out from this block */
} kaze_arena_block_t;

typedef struct kaze_arena_t {
    uint8_t *base;   /* current block's first allocatable byte (8-byte aligned) */
    size_t cap;      /* current block's usable bytes */
    size_t offset;   /* current block's next free byte; returned pointers are aligned */
    bool failed;     /* sticky: some allocation in this run did not fit */
    bool owned;      /* blocks came from kaze_arena_init_heap */
    void *raw;       /* first block's malloc pointer; init_heap only */
    kaze_arena_block_t *first; /* anchor block of the chain; NULL for caller storage */
    kaze_arena_block_t *cur;   /* block that base/cap/offset describe */
} kaze_arena_t;

/* Chained blocks grow geometrically from the initial capacity up to this many
 * bytes each. A single request larger than the cap still succeeds: it gets a
 * dedicated block of its own size. Override at build time with
 * -DKAZE_ARENA_BLOCK_MAX_BYTES=<bytes> (e.g. on a memory-constrained target). */
#ifndef KAZE_ARENA_BLOCK_MAX_BYTES
#define KAZE_ARENA_BLOCK_MAX_BYTES ((size_t)64u * 1024u * 1024u)
#endif

/* Bind an arena to storage the caller owns (stack buffer, static array, or a
 * slice of a larger allocation). A misaligned buffer costs at most 7 leading
 * bytes; every allocation out of the arena is still 8-byte aligned. This
 * arena cannot grow: when the buffer is full, kmalloc traps E_KAZE_OOM. */
void kaze_arena_init(kaze_arena_t *arena, void *buffer, size_t capacity);

#ifndef KAZE_NO_HEAP
/* Create a heap arena whose first block holds exactly capacity usable bytes.
 * capacity is also the growth hint for later blocks: each new block takes
 * twice the current one, capped at KAZE_ARENA_BLOCK_MAX_BYTES, and is always
 * large enough for the request that triggered it. Re-initialising an arena
 * that still owns blocks leaks them; call kaze_arena_free first. For a
 * fixed-memory device, use kaze_arena_init with a static buffer instead and
 * compile KAZE_NO_HEAP. */
bool kaze_arena_init_heap(kaze_arena_t *arena, size_t capacity);

/* Release every block in the chain and clear the header. */
void kaze_arena_free(kaze_arena_t *arena);

/* Adopt/transfer all memory blocks from src into dst in O(1). */
void kaze_arena_adopt(kaze_arena_t *dst, kaze_arena_t *src);
#else
/* With KAZE_NO_HEAP there is no malloc, so the emitted program's arena comes from the board
 * layer: it provides kaze_arena_init_heap over a static buffer, and kaze_arena_free is a
 * no-op. Declaring them here keeps the generated C identical on host and board. */
bool kaze_arena_init_heap(kaze_arena_t *arena, size_t capacity);
void kaze_arena_free(kaze_arena_t *arena);
void kaze_arena_adopt(kaze_arena_t *dst, kaze_arena_t *src);
#endif

/* Bump allocation from the current block, 8-byte aligned. When the request
 * does not fit, a heap arena chains a new block sized max(size, 2 x the current
 * block), capped at KAZE_ARENA_BLOCK_MAX_BYTES, and allocates there;
 * caller-owned storage has nowhere to grow, so it returns NULL and sets
 * arena->failed in that case. A failed request leaves the arena state
 * unchanged, so a smaller request afterwards still succeeds. size == 0 consumes
 * no bytes and never triggers growth. */
void *kmalloc(kaze_arena_t *arena, size_t size);

/* Bulk free: every chained block is released, the first block is kept with an
 * empty cursor, so the growth hint is the original capacity again. The failed
 * flag is cleared, because the run that ran out of memory is over. Pointers
 * into the arena are dead after this call. */
void kaze_arena_reset(kaze_arena_t *arena);

/* Pre-flight: true when size more bytes (plus worst-case alignment padding)
 * can be served. On a growable heap arena that does not fit the current block,
 * require reserves the block that will serve the plan, so true means the
 * storage exists; on caller-owned storage it only checks the bound buffer. On
 * false it records E_KAZE_OOM and sets arena->failed, so a caller can choose a
 * smaller plan before allocating anything. */
bool kaze_arena_require(kaze_arena_t *arena, size_t size);

size_t kaze_arena_used(const kaze_arena_t *arena);      /* bytes consumed across the chain */
size_t kaze_arena_capacity(const kaze_arena_t *arena);  /* bytes usable across the chain */
bool kaze_arena_failed(const kaze_arena_t *arena);      /* sticky OOM flag */

/* ------------------------------------------------------------------------- */
/* Checked arithmetic                                                         */
/* ------------------------------------------------------------------------- */

/* The *_checked family never traps: it returns false on failure and leaves
 * *out at 0 so a caller that ignores the result cannot read a torn value. The
 * bare names below (kaze_add, ...) are the trapping wrappers: they report the
 * failure through the handler and return 0 (or 0.0) as the fallback value.
 * The Phase 7 code generator only emits the checked forms plus an otherwise
 * branch; the wrappers exist so hosts and tests can exercise the trap path. */
bool kaze_add_checked(kaze_int_t a, kaze_int_t b, kaze_int_t *out);
bool kaze_sub_checked(kaze_int_t a, kaze_int_t b, kaze_int_t *out);
bool kaze_mul_checked(kaze_int_t a, kaze_int_t b, kaze_int_t *out);
bool kaze_div_checked(kaze_int_t a, kaze_int_t b, kaze_int_t *out); /* b == 0 -> E_KAZE_DIV0 */
bool kaze_mod_checked(kaze_int_t a, kaze_int_t b, kaze_int_t *out); /* b == 0 -> E_KAZE_DIV0 */
bool kaze_neg_checked(kaze_int_t a, kaze_int_t *out);
/* Float division traps E_KAZE_DIV0 for a zero divisor and E_KAZE_MATH when the
 * result is not finite (IEEE inf/NaN). Both are runtime facts, never UB. */
bool kaze_float_div_checked(kaze_float_t a, kaze_float_t b, kaze_float_t *out);

kaze_int_t kaze_add(kaze_int_t a, kaze_int_t b);
kaze_int_t kaze_sub(kaze_int_t a, kaze_int_t b);
kaze_int_t kaze_mul(kaze_int_t a, kaze_int_t b);
kaze_int_t kaze_div(kaze_int_t a, kaze_int_t b);
kaze_int_t kaze_mod(kaze_int_t a, kaze_int_t b);
kaze_int_t kaze_neg(kaze_int_t a);
kaze_float_t kaze_float_div(kaze_float_t a, kaze_float_t b);

/* ------------------------------------------------------------------------- */
/* Value representations                                                      */
/* ------------------------------------------------------------------------- */

/* An immutable byte slice. Strings are copied into the arena at construction,
 * so they stay valid until kaze_arena_reset. len == 0 means ptr is NULL. */
typedef struct kaze_str_t {
    const uint8_t *ptr;
    size_t len;
} kaze_str_t;

/* A contiguous array header in arena memory. Grown by copying into a fresh
 * arena block; the old block is abandoned to the bump cursor, which is why a
 * run should size its vectors up front when it can (cap_hint). */
typedef struct kaze_vec_t {
    void *data;
    size_t len;
    size_t cap;
    size_t elem_size;
} kaze_vec_t;

/* The tag mirrors the Rust Value enum in src/value.rs one for one:
 * Unit, Bool, Num, Str, Arr, Map, Vec. */
typedef enum kaze_kind_t {
    KAZE_KIND_UNIT = 0,
    KAZE_KIND_BOOL = 1,
    KAZE_KIND_NUMBER = 2,
    KAZE_KIND_TEXT = 3,
    KAZE_KIND_LIST = 4,
    KAZE_KIND_MAP = 5,
    KAZE_KIND_POINT = 6
} kaze_kind_t;

/* A point or offset, [x, y, z], kept inline: unlike the Rust side table this
 * costs three floats in every value slot, but it removes an indirection from
 * the hottest geometry path. */
typedef struct kaze_point_t {
    kaze_float_t x;
    kaze_float_t y;
    kaze_float_t z;
} kaze_point_t;

/* The payload is the largest of {text slice (two words), inline point (three
 * kaze_float_t), number (one kaze_float_t)}: 12 bytes on a 32-bit host with
 * f32, 16 where the text slice is two 64-bit words, 24 wherever the point uses
 * doubles. docs/KAZE-RT.md carries the measured table. */
typedef union kaze_val_payload_t {
    bool b;
    kaze_float_t num;
    kaze_str_t text;
    kaze_vec_t *list;
    struct kaze_map_t *map;
    kaze_point_t point;
} kaze_val_payload_t;

/* A tagged value. The tag is an enum rather than a bit-field so a byte dump of
 * a kaze_val_t array is inspectable in a debugger; the padding after it is
 * unavoidable given the payload's alignment. */
typedef struct kaze_val_t {
    kaze_kind_t kind;
    kaze_val_payload_t as;
} kaze_val_t;

/* Entry state: probing treats EMPTY as "stop", TOMBSTONE as "keep looking". */
typedef enum kaze_slot_t {
    KAZE_SLOT_EMPTY = 0,
    KAZE_SLOT_USED = 1,
    KAZE_SLOT_TOMBSTONE = 2
} kaze_slot_t;

typedef struct kaze_map_entry_t {
    kaze_str_t key;
    kaze_val_t val;
    uint32_t hash;
    kaze_slot_t slot;
} kaze_map_entry_t;

/* Linear-probing open addressing over arena memory. No chaining, no malloc per
 * entry, no pointer chasing beyond the probe run. Deletion leaves a tombstone
 * so probe chains survive; growth rehashes into a fresh arena block. */
typedef struct kaze_map_t {
    kaze_map_entry_t *entries;
    size_t cap;   /* always a power of two, >= KAZE_MAP_MIN_CAP */
    size_t len;   /* live keys */
    size_t tombs; /* tombstone slots */
} kaze_map_t;

#define KAZE_MAP_MIN_CAP 8u

/* Strings. All bytes are copied into the arena; an OOM returns {NULL, 0} after
 * trapping E_KAZE_OOM. */
kaze_str_t str_from_cstr(kaze_arena_t *arena, const char *cstr);
kaze_str_t str_from_bytes(kaze_arena_t *arena, const void *bytes, size_t len);
kaze_str_t str_concat(kaze_arena_t *arena, kaze_str_t head, kaze_str_t tail);
bool str_eq(kaze_str_t a, kaze_str_t b);
/* NUL-terminated copy in the arena, for printf-style diagnostics only. */
const char *str_cstr(kaze_arena_t *arena, kaze_str_t s);

/* Vectors. vec_new returns NULL when the header or the initial block does not
 * fit. vec_push copies the element; growth doubles the capacity. vec_get
 * returns NULL for an out-of-range index instead of trapping, because
 * out-of-range reads are a Phase 2 checker error, not a runtime crash. */
kaze_vec_t *vec_new(kaze_arena_t *arena, size_t elem_size, size_t cap_hint);
bool vec_push(kaze_arena_t *arena, kaze_vec_t *v, const void *elem);
void *vec_get(const kaze_vec_t *v, size_t index);
size_t vec_len(const kaze_vec_t *v);
size_t vec_cap(const kaze_vec_t *v);

/* Convenience for the common case: a vector of kaze_val_t. */
#define vec_new_vals(arena, cap_hint) vec_new((arena), sizeof(kaze_val_t), (cap_hint))

/* Maps. Keys are kaze_str_t and the bytes are copied into the arena by
 * map_set, so a caller may pass a literal, a stack buffer, or an existing
 * arena slice. map_get/map_del report absence with false. */
kaze_map_t *map_new(kaze_arena_t *arena);
bool map_get(const kaze_map_t *m, kaze_str_t key, kaze_val_t *out);
bool map_set(kaze_arena_t *arena, kaze_map_t *m, kaze_str_t key, kaze_val_t val);
bool map_del(kaze_map_t *m, kaze_str_t key);
size_t map_len(const kaze_map_t *m);
bool map_contains(const kaze_map_t *m, kaze_str_t key);

/* Value constructors and the predicates the VM branches on. */
kaze_val_t kaze_val_unit(void);
kaze_val_t kaze_val_bool(bool b);
kaze_val_t kaze_val_number(kaze_float_t n);
kaze_val_t kaze_val_text(kaze_str_t s);
kaze_val_t kaze_val_list(kaze_vec_t *v);
kaze_val_t kaze_val_map(kaze_map_t *m);
kaze_val_t kaze_val_point(kaze_float_t x, kaze_float_t y, kaze_float_t z);
bool kaze_val_truthy(kaze_val_t v);
/* "nothing", "true/false", "number", "text", "list", "record", "point" — the
 * same words the Rust VM prints in a type error. */
const char *kaze_val_kind_name(kaze_val_t v);

/* ------------------------------------------------------------------------- */
/* Monotonic clock                                                            */
/* ------------------------------------------------------------------------- */

/* Milliseconds from a monotonic source (CLOCK_MONOTONIC where available,
 * clock() otherwise). Only differences are meaningful; the epoch is arbitrary.
 * Returns 0 if no clock can be read at all. */
uint64_t kaze_time_ms(void);

/* ------------------------------------------------------------------------- */
/* Heartbeat watchdog                                                         */
/* ------------------------------------------------------------------------- */

/* 1500 ms is the vehicle-safety cadence from the roadmap: a link that has not
 * produced a packet in that window is treated as lost. */
#define KAZE_HEARTBEAT_TIMEOUT_MS 1500u

typedef struct kaze_watchdog_t {
    uint64_t last_packet_ms;
    uint64_t timeout_ms; /* 0 means KAZE_HEARTBEAT_TIMEOUT_MS */
} kaze_watchdog_t;

/* Start the window now. A timeout_ms of 0 selects the 1500 ms default. */
void kaze_watchdog_init(kaze_watchdog_t *w, uint64_t timeout_ms);
/* Record a heartbeat at the current time. */
void kaze_watchdog_packet(kaze_watchdog_t *w);
/* True when now - last_packet >= timeout. A clock that appears to run
 * backwards is also stale: this check fails closed. */
bool kaze_watchdog_stale(const kaze_watchdog_t *w);
/* KAZE_OK while fresh. When stale it traps E_SAFETY_VIOLATION and returns that
 * code, so a caller can propagate it without string comparison. */
kaze_code_t kaze_watchdog_abort_if_stale(const kaze_watchdog_t *w);

/* ------------------------------------------------------------------------- */
/* Signal-time actuator de-assert                                             */
/* ------------------------------------------------------------------------- */

#ifndef KAZE_NO_SIGNALS
/* 16 slots is the roadmap's effector budget: enough for every actuator a
 * vehicle declares, few enough to fit in one cache line and to walk inside a
 * signal handler without allocating. */
#define KAZE_MAX_DEASSERTS 16u

/* De-assert callbacks run inside a signal handler, so they may only do what is
 * async-signal-safe: write a memory-mapped register or a pre-opened fd, set a
 * volatile flag. No malloc, no stdio, no locks. */
typedef void (*kaze_cleanup_fn)(void);

/* Register a callback. Returns false when fn is NULL or all 16 slots are used;
 * registration is not thread-safe (do it during startup). */
bool kaze_register_deassert(kaze_cleanup_fn fn);
size_t kaze_deassert_count(void);

/* Run every registered callback in reverse registration order (de-assert the
 * newest effector first) then return. The signal path calls this and exits;
 * hosts that unwind without a signal call it directly. */
void kaze_run_cleanups(void);

/* Install handlers for SIGINT, SIGTERM and SIGSEGV. The handler runs the
 * callbacks and then _exit(128 + signal), so a stuck or crashing process still
 * leaves the actuators safe. Returns false if any install fails. */
bool kaze_install_cleanup_signal_handlers(void);
#endif /* KAZE_NO_SIGNALS */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KAZE_RT_H */

