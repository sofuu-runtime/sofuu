/* kaze_rt.c — the Kaze C99 runtime substrate (roadmap Phase 1).
 *
 * Implementation notes that matter to a reader auditing this file:
 *   - No dynamic allocation except the arena's heap blocks. kaze_arena_init_heap
 *     creates the first block and a growing kmalloc chains later ones through
 *     the same single helper. Containers take the arena they allocate from as
 *     a parameter, so a lowered program has one memory story, not two.
 *   - No abort(), no longjmp(), no exit() on a trap path. Traps are codes the
 *     embedder handles; the process keeps running with a documented fallback
 *     value (0, false, NULL).
 *   - Signed overflow, division by zero and INT_MIN/-1 are never executed.
 *     The checked helpers either use the compiler intrinsic (which is defined
 *     to wrap into the out-parameter and report) or the C99 range recipe.
 */

/* Ask for POSIX.1-2008 so clock_gettime and CLOCK_MONOTONIC are declared.
 * This must come before any system header: glibc gates those declarations on
 * the feature macro, and -std=c99 alone selects a strict ISO namespace. */
#if !defined(_POSIX_C_SOURCE) && !defined(_MSC_VER)
#define _POSIX_C_SOURCE 200809L
#endif

#include "kaze_rt.h"

#ifndef KAZE_NO_HEAP
/* Declared here rather than by including <stdlib.h>: the runtime's include
 * allowlist is deliberate (embedded targets frequently have no hosted stdlib),
 * and these two symbols are referenced only from the arena block helpers,
 * which KAZE_NO_HEAP removes entirely. */
extern void *malloc(size_t size);
extern void free(void *ptr);
#endif

/* 8 bytes: the widest scalar any lowered Kaze value needs is a double or a
 * pointer, and both are 8-byte aligned on every target this runtime supports.
 * Keeping one alignment constant means a value slot never straddles a cache
 * line boundary differently on 32-bit and 64-bit builds. */
#define KAZE_ALIGN 8u

#if defined(__clang__)
#define KAZE_HAVE_BUILTIN_OVERFLOW 1
#elif defined(__GNUC__) && (__GNUC__ >= 5)
#define KAZE_HAVE_BUILTIN_OVERFLOW 1
#elif defined(_MSC_VER)
/* cl.exe has no __builtin_*_overflow (through VS2022 17.x). The portable
 * recipes below are the CERT C integer-overflow rules, so MSVC gets the same
 * answers without intrinsics. */
#define KAZE_HAVE_BUILTIN_OVERFLOW 0
#else
#define KAZE_HAVE_BUILTIN_OVERFLOW 0
#endif

/* ------------------------------------------------------------------------- */
/* Traps                                                                      */
/* ------------------------------------------------------------------------- */

static int kaze_stderr_handler(const char *code, const char *message)
{
    if (code == NULL) {
        code = "E_KAZE_BADARG";
    }
    if (message == NULL) {
        message = "";
    }
    /* One line, no allocation: a trap may fire while the arena is exhausted,
     * so the diagnostic path must not depend on the arena. */
    fprintf(stderr, "%s: %s\n", code, message);
    return 1;
}

/* Zero-initialised on purpose. An initialised pointer here is file-backed
 * writable data, which on macOS costs a whole 16 KB __DATA page in every
 * binary that links this runtime. NULL means "the stderr handler" and is
 * resolved where the handler is called, so the default behaves exactly as
 * before while the segment stays zero-fill. */
static kaze_fallback_fn kaze_active_handler;

static kaze_fallback_fn kaze_resolve_handler(void)
{
    return (kaze_active_handler != NULL) ? kaze_active_handler : kaze_stderr_handler;
}

void kaze_set_fallback_handler(kaze_fallback_fn fn)
{
    kaze_active_handler = fn;
}

kaze_fallback_fn kaze_get_fallback_handler(void)
{
    return kaze_resolve_handler();
}

int kaze_default_fallback(const char *code, const char *message)
{
    return kaze_stderr_handler(code, message);
}

const char *kaze_code_name(kaze_code_t code)
{
    switch (code) {
    case KAZE_OK:
        return "KAZE_OK";
    case KAZE_ERR_OOM:
        return "E_KAZE_OOM";
    case KAZE_ERR_OVERFLOW:
        return "E_KAZE_OVERFLOW";
    case KAZE_ERR_DIV0:
        return "E_KAZE_DIV0";
    case KAZE_ERR_MATH:
        return "E_KAZE_MATH";
    case KAZE_ERR_SAFETY_VIOLATION:
        return "E_SAFETY_VIOLATION";
    case KAZE_ERR_BADARG:
        return "E_KAZE_BADARG";
    case KAZE_ERR_TYPE:
        return "E_KAZE_TYPE";
    case KAZE_ERR_SHAPE:
        return "E_KAZE_SHAPE_MISMATCH";
    case KAZE_ERR_QUANT:
        return "E_KAZE_QUANT";
    case KAZE_ERR_IO:
        return "E_KAZE_IO";
    case KAZE_ERR_FORMAT:
        return "E_KAZE_FORMAT";
    default:
        return "E_KAZE_BADCODE";
    }
}

int kaze_trap(kaze_code_t code, const char *message)
{
    return kaze_resolve_handler()(kaze_code_name(code), message);
}

/* A trap message that names the operands. snprintf into a small stack buffer
 * is the cheapest way to make a failing gate explain itself without the arena;
 * every trap path is cold. */
static void kaze_trap_int(kaze_code_t code, const char *what, kaze_int_t a, char op, kaze_int_t b)
{
    char buf[112];
    snprintf(buf, sizeof buf, "%s %lld %c %lld does not fit kaze_int_t (%d-bit)",
             what, (long long)a, op, (long long)b, (int)KAZE_INT_BITS);
    (void)kaze_trap(code, buf);
}

/* ------------------------------------------------------------------------- */
/* Arena                                                                      */
/* ------------------------------------------------------------------------- */

/* A heap arena is a chain of blocks. Each block's header sits at the start of
 * the malloc'd region and the usable bytes follow it, advanced to the 8-byte
 * boundary the rest of the runtime promises. Every block goes through one
 * allocation helper, so the runtime still has exactly one malloc call site;
 * free() has one per link and is only reached from reset and release. */
static size_t kaze_align_pad(size_t offset)
{
    return (size_t)((KAZE_ALIGN - (offset % KAZE_ALIGN)) % KAZE_ALIGN);
}

/* The single out-of-memory path: mark the run, report the code, return NULL.
 * The cursor is not touched, so the arena is still usable for a smaller
 * request and a caller can retry with a tighter plan. */
static void *kaze_arena_oom(kaze_arena_t *arena, size_t size)
{
    char buf[112];
    size_t free_bytes = (arena->offset <= arena->cap) ? (arena->cap - arena->offset) : 0;
    arena->failed = true;
    snprintf(buf, sizeof buf, "arena exhausted: %llu bytes requested, %llu available",
             (unsigned long long)size, (unsigned long long)free_bytes);
    (void)kaze_trap(KAZE_ERR_OOM, buf);
    return NULL;
}

void kaze_arena_init(kaze_arena_t *arena, void *buffer, size_t capacity)
{
    uintptr_t start;
    size_t pad;

    if (arena == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_arena_init: arena header is NULL");
        return;
    }
    /* Caller-owned storage: one fixed block, no chain, nothing to free later.
     * Zero first so an uninitialised header never looks like an owned one. */
    memset(arena, 0, sizeof *arena);
    if (buffer == NULL || capacity == 0) {
        return;
    }
    /* Align the base once instead of on every allocation: the caller may hand
     * us a char array, a struct field, or a slice of a larger block, and we
     * still owe every allocation 8-byte alignment. */
    start = (uintptr_t)buffer;
    pad = (size_t)((KAZE_ALIGN - (start % KAZE_ALIGN)) % KAZE_ALIGN);
    if (pad >= capacity) {
        return; /* not even one aligned byte is available */
    }
    arena->base = ((uint8_t *)buffer) + pad;
    arena->cap = capacity - pad;
}

#ifndef KAZE_NO_HEAP
/* The runtime's one malloc call site. Every block, first or chained, is taken
 * here so the "single allocation point" audit in scripts/rt-check.sh stays a
 * one-line fact. */
static void *kaze_heap_alloc(size_t bytes)
{
    return malloc(bytes);
}

/* One block with exactly capacity usable bytes, or NULL when the request is
 * not representable or the host refused it. The caller reports E_KAZE_OOM. */
static kaze_arena_block_t *kaze_arena_block_new(size_t capacity)
{
    size_t header = sizeof(kaze_arena_block_t);
    size_t request;
    kaze_arena_block_t *block;
    uint8_t *raw;
    size_t pad;

    if (capacity == 0) {
        capacity = 1; /* a zero-sized block could never serve an allocation */
    }
    /* header + up to 7 alignment bytes + capacity, without wrapping. */
    if (capacity > (SIZE_MAX - header - (KAZE_ALIGN - 1u))) {
        return NULL;
    }
    request = header + (KAZE_ALIGN - 1u) + capacity;
    raw = (uint8_t *)kaze_heap_alloc(request);
    if (raw == NULL) {
        return NULL;
    }
    block = (kaze_arena_block_t *)raw;
    pad = (size_t)((KAZE_ALIGN - (((uintptr_t)(raw + header)) % KAZE_ALIGN)) % KAZE_ALIGN);
    block->next = NULL;
    block->raw = raw;
    block->data = raw + header + pad;
    block->cap = capacity;
    block->used = 0;
    return block;
}

/* Size of the block to chain for a request of size bytes: twice the current
 * block, capped, but never smaller than the request that must fit. Doubling is
 * what keeps a compiler's hundreds of megabytes at a logarithmic number of
 * blocks. The current block's capacity is the whole growth state, so reset
 * restores the original hint by making the first block current again. */
static size_t kaze_arena_growth_size(const kaze_arena_t *arena, size_t size)
{
    size_t limit = (size_t)KAZE_ARENA_BLOCK_MAX_BYTES;
    size_t current = (arena->cur != NULL) ? arena->cur->cap : 0u;
    size_t target;

    if (limit < 1u) {
        limit = 1u;
    }
    if (current >= limit || current > (SIZE_MAX / 2u)) {
        target = limit;
    } else {
        target = current * 2u;
        if (target > limit) {
            target = limit;
        }
    }
    if (target < 1u) {
        target = 1u;
    }
    if (size > target) {
        target = size; /* a dedicated block for an oversized request */
    }
    return target;
}

/* Chain a block that can serve size bytes and make it current. The previous
 * block keeps its contents; nothing is freed, because pointers into it are
 * still live until reset. */
static void *kaze_arena_grow(kaze_arena_t *arena, size_t size)
{
    kaze_arena_block_t *block = kaze_arena_block_new(kaze_arena_growth_size(arena, size));

    if (block == NULL) {
        return NULL;
    }
    arena->cur->next = block;
    arena->cur = block;
    arena->base = block->data;
    arena->cap = block->cap;
    arena->offset = 0;
    return arena->base;
}

bool kaze_arena_init_heap(kaze_arena_t *arena, size_t capacity)
{
    kaze_arena_block_t *first;

    if (arena == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_arena_init_heap: arena header is NULL");
        return false;
    }
    memset(arena, 0, sizeof *arena);
    first = kaze_arena_block_new(capacity);
    if (first == NULL) {
        char buf[96];
        snprintf(buf, sizeof buf, "arena heap allocation of %llu bytes failed",
                 (unsigned long long)capacity);
        (void)kaze_trap(KAZE_ERR_OOM, buf);
        return false;
    }
    arena->base = first->data;
    arena->cap = first->cap;
    arena->offset = 0;
    arena->failed = false;
    arena->owned = true;
    arena->raw = first->raw;
    arena->first = first;
    arena->cur = first;
    return true;
}

void kaze_arena_free(kaze_arena_t *arena)
{
    kaze_arena_block_t *block;

    if (arena == NULL) {
        return;
    }
    block = arena->first;
    while (block != NULL) {
        kaze_arena_block_t *next = block->next;
        free(block->raw);
        block = next;
    }
    memset(arena, 0, sizeof *arena);
}

void kaze_arena_adopt(kaze_arena_t *dst, kaze_arena_t *src)
{
    if (dst == NULL || src == NULL || src->first == NULL) {
        return;
    }
    if (dst->cur != NULL) {
        dst->cur->next = src->first;
        dst->cur = src->cur ? src->cur : src->first;
    } else {
        dst->first = src->first;
        dst->cur = src->cur ? src->cur : src->first;
    }
    memset(src, 0, sizeof *src);
}
#else
void kaze_arena_adopt(kaze_arena_t *dst, kaze_arena_t *src)
{
    (void)dst;
    (void)src;
}
#endif /* KAZE_NO_HEAP */

void *kmalloc(kaze_arena_t *arena, size_t size)
{
    size_t pad;
    size_t remaining;
    size_t start;

    if (arena == NULL || arena->base == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kmalloc: arena is not bound to storage");
        return NULL;
    }
    if (arena->offset > arena->cap) {
        return kaze_arena_oom(arena, size); /* defensive: only a corrupt header gets here */
    }
    remaining = arena->cap - arena->offset;
    pad = kaze_align_pad(arena->offset);
    start = arena->offset + pad;
    if (size == 0) {
        /* A zero-size request consumes nothing and still yields a pointer the
         * caller may compare or pass along. It never triggers growth. */
        if (pad > remaining) {
            return kaze_arena_oom(arena, size);
        }
        return arena->base + start;
    }
    if (pad <= remaining && size <= (remaining - pad)) {
        arena->offset = start + size;
        if (arena->cur != NULL) {
            arena->cur->used = arena->offset;
        }
        return arena->base + start;
    }
    /* The current block cannot serve this request. A heap arena chains another
     * block (old blocks stay live); caller-owned storage has nowhere to grow
     * and reports E_KAZE_OOM like it always did. */
#ifndef KAZE_NO_HEAP
    if (arena->owned && kaze_arena_grow(arena, size) != NULL) {
        arena->offset = size; /* a new block's data is already 8-byte aligned */
        arena->cur->used = size;
        return arena->base;
    }
#endif
    return kaze_arena_oom(arena, size);
}

void kaze_arena_reset(kaze_arena_t *arena)
{
    if (arena == NULL) {
        return;
    }
#ifndef KAZE_NO_HEAP
    if (arena->owned && arena->first != NULL) {
        kaze_arena_block_t *block = arena->first->next;

        while (block != NULL) {
            kaze_arena_block_t *next = block->next;
            free(block->raw);
            block = next;
        }
        arena->first->next = NULL;
        arena->first->used = 0;
        arena->cur = arena->first;
        arena->base = arena->first->data;
        arena->cap = arena->first->cap;
        arena->offset = 0;
        arena->failed = false;
        return;
    }
#endif
    /* Caller-owned storage or an unbound header: the bulk free is a cursor
     * move. Capacity, ownership and the base pointer survive. */
    arena->offset = 0;
    arena->failed = false;
}

bool kaze_arena_require(kaze_arena_t *arena, size_t size)
{
    size_t pad;
    size_t remaining;

    if (arena == NULL || arena->base == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_arena_require: arena is not bound to storage");
        return false;
    }
    if (arena->offset > arena->cap) {
        (void)kaze_arena_oom(arena, size);
        return false;
    }
    remaining = arena->cap - arena->offset;
    pad = kaze_align_pad(arena->offset);
    if (size == 0) {
        if (pad <= remaining) {
            return true;
        }
        (void)kaze_arena_oom(arena, size);
        return false;
    }
    if (pad <= remaining && size <= (remaining - pad)) {
        return true;
    }
    /* A growable arena can back the plan with a new block, so reserve it now:
     * true means the storage exists, not merely that it could be obtained. */
#ifndef KAZE_NO_HEAP
    if (arena->owned && kaze_arena_grow(arena, size) != NULL) {
        return true;
    }
#endif
    (void)kaze_arena_oom(arena, size);
    return false;
}

size_t kaze_arena_used(const kaze_arena_t *arena)
{
    const kaze_arena_block_t *block;
    size_t total = 0;

    if (arena == NULL) {
        return 0;
    }
    if (!arena->owned) {
        return arena->offset;
    }
    for (block = arena->first; block != NULL; block = block->next) {
        size_t used = (block == arena->cur) ? arena->offset : block->used;
        if (used > (SIZE_MAX - total)) {
            return SIZE_MAX;
        }
        total += used;
    }
    return total;
}

size_t kaze_arena_capacity(const kaze_arena_t *arena)
{
    const kaze_arena_block_t *block;
    size_t total = 0;

    if (arena == NULL) {
        return 0;
    }
    if (!arena->owned) {
        return arena->cap;
    }
    for (block = arena->first; block != NULL; block = block->next) {
        if (block->cap > (SIZE_MAX - total)) {
            return SIZE_MAX;
        }
        total += block->cap;
    }
    return total;
}

bool kaze_arena_failed(const kaze_arena_t *arena)
{
    return (arena != NULL) && arena->failed;
}

/* ------------------------------------------------------------------------- */
/* Checked arithmetic                                                         */
/* ------------------------------------------------------------------------- */

#if !KAZE_HAVE_BUILTIN_OVERFLOW
/* CERT C INT32-C / INT64-C recipes. Written in terms of the range bounds so
 * the compiler cannot see a signed overflow, and so MSVC and a pre-GCC-5
 * compiler agree with the intrinsic path. */
static bool kaze_add_overflows(kaze_int_t a, kaze_int_t b, kaze_int_t *r)
{
    if (b > 0 && a > (kaze_int_t)(KAZE_INT_MAX - b)) {
        return true;
    }
    if (b < 0 && a < (kaze_int_t)(KAZE_INT_MIN - b)) {
        return true;
    }
    *r = a + b;
    return false;
}

static bool kaze_sub_overflows(kaze_int_t a, kaze_int_t b, kaze_int_t *r)
{
    if (b > 0 && a < (kaze_int_t)(KAZE_INT_MIN + b)) {
        return true;
    }
    if (b < 0 && a > (kaze_int_t)(KAZE_INT_MAX + b)) {
        return true;
    }
    *r = a - b;
    return false;
}

static bool kaze_mul_overflows(kaze_int_t a, kaze_int_t b, kaze_int_t *r)
{
    if (a > 0) {
        if (b > 0) {
            if (a > (kaze_int_t)(KAZE_INT_MAX / b)) {
                return true;
            }
        } else if (b < (kaze_int_t)(KAZE_INT_MIN / a)) {
            return true;
        }
    } else if (a < 0) {
        if (b > 0) {
            if (a < (kaze_int_t)(KAZE_INT_MIN / b)) {
                return true;
            }
        } else if (a != 0 && b < (kaze_int_t)(KAZE_INT_MAX / a)) {
            return true;
        }
    }
    *r = a * b;
    return false;
}
#endif /* !KAZE_HAVE_BUILTIN_OVERFLOW */

bool kaze_add_checked(kaze_int_t a, kaze_int_t b, kaze_int_t *out)
{
    kaze_int_t r = 0;
    if (out == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_add_checked: out is NULL");
        return false;
    }
#if KAZE_HAVE_BUILTIN_OVERFLOW
    if (__builtin_add_overflow(a, b, &r)) {
        *out = 0;
        return false;
    }
#else
    if (kaze_add_overflows(a, b, &r)) {
        *out = 0;
        return false;
    }
#endif
    *out = r;
    return true;
}

bool kaze_sub_checked(kaze_int_t a, kaze_int_t b, kaze_int_t *out)
{
    kaze_int_t r = 0;
    if (out == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_sub_checked: out is NULL");
        return false;
    }
#if KAZE_HAVE_BUILTIN_OVERFLOW
    if (__builtin_sub_overflow(a, b, &r)) {
        *out = 0;
        return false;
    }
#else
    if (kaze_sub_overflows(a, b, &r)) {
        *out = 0;
        return false;
    }
#endif
    *out = r;
    return true;
}

bool kaze_mul_checked(kaze_int_t a, kaze_int_t b, kaze_int_t *out)
{
    kaze_int_t r = 0;
    if (out == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_mul_checked: out is NULL");
        return false;
    }
#if KAZE_HAVE_BUILTIN_OVERFLOW
    if (__builtin_mul_overflow(a, b, &r)) {
        *out = 0;
        return false;
    }
#else
    if (kaze_mul_overflows(a, b, &r)) {
        *out = 0;
        return false;
    }
#endif
    *out = r;
    return true;
}

bool kaze_div_checked(kaze_int_t a, kaze_int_t b, kaze_int_t *out)
{
    if (out == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_div_checked: out is NULL");
        return false;
    }
    /* Two exits: a zero divisor is E_KAZE_DIV0, and INT_MIN / -1 is the one
     * quotient that does not fit. Both are checked before the divide, so the
     * operation itself is always defined. */
    if (b == 0 || (a == KAZE_INT_MIN && b == -1)) {
        *out = 0;
        return false;
    }
    *out = a / b;
    return true;
}

bool kaze_mod_checked(kaze_int_t a, kaze_int_t b, kaze_int_t *out)
{
    if (out == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_mod_checked: out is NULL");
        return false;
    }
    if (b == 0 || (a == KAZE_INT_MIN && b == -1)) {
        *out = 0;
        return false;
    }
    *out = a % b;
    return true;
}

bool kaze_neg_checked(kaze_int_t a, kaze_int_t *out)
{
    if (out == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_neg_checked: out is NULL");
        return false;
    }
    if (a == KAZE_INT_MIN) {
        *out = 0;
        return false;
    }
    *out = -a;
    return true;
}

/* IEEE-754 finiteness without math.h's isfinite: NaN fails x == x, and any
 * infinity fails x - x == 0. This is also the one form MSVC has always had. */
static bool kaze_float_is_finite(kaze_float_t x)
{
    return (x == x) && ((x - x) == (kaze_float_t)0);
}

bool kaze_float_div_checked(kaze_float_t a, kaze_float_t b, kaze_float_t *out)
{
    kaze_float_t r;
    if (out == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_float_div_checked: out is NULL");
        return false;
    }
    if (b == (kaze_float_t)0) {
        *out = (kaze_float_t)0;
        return false;
    }
    r = a / b;
    if (!kaze_float_is_finite(r)) {
        *out = (kaze_float_t)0;
        return false;
    }
    *out = r;
    return true;
}

/* Trapping wrappers. A caller that ignores the handler's return value still
 * gets a defined result: 0 for integers, 0.0 for floats. */

kaze_int_t kaze_add(kaze_int_t a, kaze_int_t b)
{
    kaze_int_t r = 0;
    if (!kaze_add_checked(a, b, &r)) {
        kaze_trap_int(KAZE_ERR_OVERFLOW, "integer addition", a, '+', b);
        return 0;
    }
    return r;
}

kaze_int_t kaze_sub(kaze_int_t a, kaze_int_t b)
{
    kaze_int_t r = 0;
    if (!kaze_sub_checked(a, b, &r)) {
        kaze_trap_int(KAZE_ERR_OVERFLOW, "integer subtraction", a, '-', b);
        return 0;
    }
    return r;
}

kaze_int_t kaze_mul(kaze_int_t a, kaze_int_t b)
{
    kaze_int_t r = 0;
    if (!kaze_mul_checked(a, b, &r)) {
        kaze_trap_int(KAZE_ERR_OVERFLOW, "integer multiplication", a, '*', b);
        return 0;
    }
    return r;
}

kaze_int_t kaze_div(kaze_int_t a, kaze_int_t b)
{
    kaze_int_t r = 0;
    if (!kaze_div_checked(a, b, &r)) {
        if (b == 0) {
            kaze_trap_int(KAZE_ERR_DIV0, "integer division", a, '/', b);
        } else {
            kaze_trap_int(KAZE_ERR_OVERFLOW, "integer division", a, '/', b);
        }
        return 0;
    }
    return r;
}

kaze_int_t kaze_mod(kaze_int_t a, kaze_int_t b)
{
    kaze_int_t r = 0;
    if (!kaze_mod_checked(a, b, &r)) {
        if (b == 0) {
            kaze_trap_int(KAZE_ERR_DIV0, "integer modulo", a, '%', b);
        } else {
            kaze_trap_int(KAZE_ERR_OVERFLOW, "integer modulo", a, '%', b);
        }
        return 0;
    }
    return r;
}

kaze_int_t kaze_neg(kaze_int_t a)
{
    kaze_int_t r = 0;
    if (!kaze_neg_checked(a, &r)) {
        kaze_trap_int(KAZE_ERR_OVERFLOW, "integer negation", 0, '-', a);
        return 0;
    }
    return r;
}

kaze_float_t kaze_float_div(kaze_float_t a, kaze_float_t b)
{
    kaze_float_t r = (kaze_float_t)0;
    if (!kaze_float_div_checked(a, b, &r)) {
        if (b == (kaze_float_t)0) {
            (void)kaze_trap(KAZE_ERR_DIV0, "float division by zero");
        } else {
            char buf[112];
            snprintf(buf, sizeof buf, "float division produced a non-finite result (%g / %g)",
                     (double)a, (double)b);
            (void)kaze_trap(KAZE_ERR_MATH, buf);
        }
        return (kaze_float_t)0;
    }
    return r;
}

/* ------------------------------------------------------------------------- */
/* Strings                                                                    */
/* ------------------------------------------------------------------------- */

kaze_str_t str_from_bytes(kaze_arena_t *arena, const void *bytes, size_t len)
{
    kaze_str_t s;
    uint8_t *dst;

    s.ptr = NULL;
    s.len = 0;
    if (bytes == NULL || len == 0) {
        /* The empty string is the null slice; no allocation, no pointer. */
        return s;
    }
    dst = (uint8_t *)kmalloc(arena, len);
    if (dst == NULL) {
        return s; /* E_KAZE_OOM already reported by the arena */
    }
    memcpy(dst, bytes, len);
    s.ptr = dst;
    s.len = len;
    return s;
}

kaze_str_t str_from_cstr(kaze_arena_t *arena, const char *cstr)
{
    if (cstr == NULL) {
        kaze_str_t empty;
        empty.ptr = NULL;
        empty.len = 0;
        return empty;
    }
    return str_from_bytes(arena, cstr, strlen(cstr));
}

kaze_str_t str_concat(kaze_arena_t *arena, kaze_str_t head, kaze_str_t tail)
{
    kaze_str_t s;
    uint8_t *dst;
    size_t total;

    s.ptr = NULL;
    s.len = 0;
    if (head.len > (SIZE_MAX - tail.len)) {
        (void)kaze_trap(KAZE_ERR_OOM, "str_concat: combined length overflows size_t");
        return s;
    }
    total = head.len + tail.len;
    if (total == 0) {
        return s;
    }
    dst = (uint8_t *)kmalloc(arena, total);
    if (dst == NULL) {
        return s;
    }
    if (head.len > 0) {
        memcpy(dst, head.ptr, head.len);
    }
    if (tail.len > 0) {
        memcpy(dst + head.len, tail.ptr, tail.len);
    }
    s.ptr = dst;
    s.len = total;
    return s;
}

bool str_eq(kaze_str_t a, kaze_str_t b)
{
    if (a.len != b.len) {
        return false;
    }
    if (a.len == 0) {
        return true;
    }
    return memcmp(a.ptr, b.ptr, a.len) == 0;
}

const char *str_cstr(kaze_arena_t *arena, kaze_str_t s)
{
    char *dst;
    if (s.len == 0 || s.ptr == NULL) {
        return "";
    }
    dst = (char *)kmalloc(arena, s.len + 1u);
    if (dst == NULL) {
        return "";
    }
    memcpy(dst, s.ptr, s.len);
    dst[s.len] = '\0';
    return dst;
}

/* ------------------------------------------------------------------------- */
/* Vectors                                                                    */
/* ------------------------------------------------------------------------- */

kaze_vec_t *vec_new(kaze_arena_t *arena, size_t elem_size, size_t cap_hint)
{
    kaze_vec_t *v;

    if (arena == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "vec_new: arena is NULL");
        return NULL;
    }
    if (elem_size == 0) {
        elem_size = 1; /* a zero-width element would make len meaningless */
    }
    v = (kaze_vec_t *)kmalloc(arena, sizeof *v);
    if (v == NULL) {
        return NULL;
    }
    v->data = NULL;
    v->len = 0;
    v->cap = 0;
    v->elem_size = elem_size;
    if (cap_hint > 0) {
        if (cap_hint > (SIZE_MAX / elem_size)) {
            (void)kaze_trap(KAZE_ERR_OOM, "vec_new: capacity overflows size_t");
            return NULL;
        }
        v->data = kmalloc(arena, cap_hint * elem_size);
        if (v->data == NULL) {
            return NULL;
        }
        v->cap = cap_hint;
    }
    return v;
}

bool vec_push(kaze_arena_t *arena, kaze_vec_t *v, const void *elem)
{
    if (arena == NULL || v == NULL || elem == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "vec_push: NULL arena, vector or element");
        return false;
    }
    if (v->len == v->cap) {
        size_t new_cap = (v->cap == 0) ? 4u : (v->cap * 2u);
        void *fresh;

        if (new_cap < v->cap || new_cap > (SIZE_MAX / v->elem_size)) {
            (void)kaze_trap(KAZE_ERR_OOM, "vec_push: growth overflows size_t");
            return false;
        }
        fresh = kmalloc(arena, new_cap * v->elem_size);
        if (fresh == NULL) {
            return false;
        }
        if (v->len > 0) {
            /* The old block is not freed: arena memory is reclaimed in bulk.
             * That is the accepted cost of growth without a free list. */
            memcpy(fresh, v->data, v->len * v->elem_size);
        }
        v->data = fresh;
        v->cap = new_cap;
    }
    memcpy(((uint8_t *)v->data) + (v->len * v->elem_size), elem, v->elem_size);
    v->len++;
    return true;
}

void *vec_get(const kaze_vec_t *v, size_t index)
{
    if (v == NULL || v->data == NULL || index >= v->len) {
        return NULL;
    }
    return ((uint8_t *)v->data) + (index * v->elem_size);
}

size_t vec_len(const kaze_vec_t *v)
{
    return (v != NULL) ? v->len : 0;
}

size_t vec_cap(const kaze_vec_t *v)
{
    return (v != NULL) ? v->cap : 0;
}

/* ------------------------------------------------------------------------- */
/* Maps                                                                       */
/* ------------------------------------------------------------------------- */

/* FNV-1a, 32-bit. Chosen for four lines of code and a good spread on short
 * identifier-sized keys; the table is power-of-two sized, so the low bits of
 * the hash pick the first probe slot. */
static uint32_t kaze_str_hash(kaze_str_t key)
{
    uint32_t h = 2166136261u;
    size_t i;

    for (i = 0; i < key.len; i++) {
        h ^= (uint32_t)key.ptr[i];
        h *= 16777619u;
    }
    return h;
}

/* Returns the slot holding the key, or the slot where it should be inserted.
 * *found says which. SIZE_MAX means the probe run hit every slot without
 * finding an empty one, which only a corrupted table can produce; the caller
 * treats it as "grow". */
static size_t kaze_map_lookup(const kaze_map_t *m, kaze_str_t key, uint32_t hash, bool *found)
{
    size_t mask;
    size_t i;
    size_t first_tombstone = SIZE_MAX;
    size_t steps;

    *found = false;
    if (m == NULL || m->entries == NULL || m->cap == 0) {
        return SIZE_MAX;
    }
    mask = m->cap - 1u;
    i = (size_t)hash & mask;
    for (steps = 0; steps < m->cap; steps++) {
        const kaze_map_entry_t *e = &m->entries[i];

        if (e->slot == KAZE_SLOT_EMPTY) {
            return (first_tombstone != SIZE_MAX) ? first_tombstone : i;
        }
        if (e->slot == KAZE_SLOT_TOMBSTONE) {
            if (first_tombstone == SIZE_MAX) {
                first_tombstone = i;
            }
        } else if (e->hash == hash && e->key.len == key.len &&
                   (key.len == 0 || memcmp(e->key.ptr, key.ptr, key.len) == 0)) {
            *found = true;
            return i;
        }
        i = (i + 1u) & mask;
    }
    return (first_tombstone != SIZE_MAX) ? first_tombstone : SIZE_MAX;
}

static bool kaze_map_grow(kaze_arena_t *arena, kaze_map_t *m)
{
    size_t new_cap = (m->cap == 0) ? (size_t)KAZE_MAP_MIN_CAP : (m->cap * 2u);
    kaze_map_entry_t *fresh;
    size_t i;

    if (new_cap > (SIZE_MAX / sizeof(kaze_map_entry_t))) {
        (void)kaze_trap(KAZE_ERR_OOM, "map: capacity overflows size_t");
        return false;
    }
    fresh = (kaze_map_entry_t *)kmalloc(arena, new_cap * sizeof *fresh);
    if (fresh == NULL) {
        return false;
    }
    memset(fresh, 0, new_cap * sizeof *fresh);
    for (i = 0; i < m->cap; i++) {
        if (m->entries[i].slot == KAZE_SLOT_USED) {
            size_t mask = new_cap - 1u;
            size_t j = (size_t)m->entries[i].hash & mask;

            while (fresh[j].slot == KAZE_SLOT_USED) {
                j = (j + 1u) & mask;
            }
            fresh[j] = m->entries[i];
        }
    }
    /* The previous entry array is abandoned in the arena, exactly like a grown
     * vector's block: bulk reclaim at reset, no per-object free. */
    m->entries = fresh;
    m->cap = new_cap;
    m->tombs = 0;
    return true;
}

kaze_map_t *map_new(kaze_arena_t *arena)
{
    kaze_map_t *m;

    if (arena == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "map_new: arena is NULL");
        return NULL;
    }
    m = (kaze_map_t *)kmalloc(arena, sizeof *m);
    if (m == NULL) {
        return NULL;
    }
    m->entries = NULL;
    m->cap = 0;
    m->len = 0;
    m->tombs = 0;
    if (!kaze_map_grow(arena, m)) {
        return NULL;
    }
    return m;
}

bool map_get(const kaze_map_t *m, kaze_str_t key, kaze_val_t *out)
{
    bool found;
    size_t idx;
    uint32_t hash;

    if (m == NULL || m->entries == NULL || m->cap == 0) {
        return false;
    }
    hash = kaze_str_hash(key);
    idx = kaze_map_lookup(m, key, hash, &found);
    if (!found || idx == SIZE_MAX) {
        return false;
    }
    if (out != NULL) {
        *out = m->entries[idx].val;
    }
    return true;
}

bool map_contains(const kaze_map_t *m, kaze_str_t key)
{
    return map_get(m, key, NULL);
}

bool map_set(kaze_arena_t *arena, kaze_map_t *m, kaze_str_t key, kaze_val_t val)
{
    uint32_t hash;
    bool found;
    size_t idx;

    if (arena == NULL || m == NULL || m->entries == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "map_set: NULL arena or uninitialised map");
        return false;
    }
    hash = kaze_str_hash(key);
    idx = kaze_map_lookup(m, key, hash, &found);
    if (found && idx != SIZE_MAX) {
        m->entries[idx].val = val;
        return true;
    }
    /* Grow at 70% occupancy, or immediately if the probe run found no empty
     * slot. Both conditions are checked before the key bytes are copied so an
     * OOM cannot leave a half-inserted entry behind. */
    if (idx == SIZE_MAX || ((m->len + m->tombs + 1u) * 10u) >= (m->cap * 7u)) {
        if (!kaze_map_grow(arena, m)) {
            return false;
        }
        idx = kaze_map_lookup(m, key, hash, &found);
        if (idx == SIZE_MAX) {
            (void)kaze_trap(KAZE_ERR_OOM, "map_set: table cannot grow further");
            return false;
        }
    }
    if (key.len > 0) {
        uint8_t *copy = (uint8_t *)kmalloc(arena, key.len);

        if (copy == NULL) {
            return false;
        }
        memcpy(copy, key.ptr, key.len);
        key.ptr = copy;
    }
    if (m->entries[idx].slot == KAZE_SLOT_TOMBSTONE) {
        m->tombs--;
    }
    m->entries[idx].key = key;
    m->entries[idx].val = val;
    m->entries[idx].hash = hash;
    m->entries[idx].slot = KAZE_SLOT_USED;
    m->len++;
    return true;
}

bool map_del(kaze_map_t *m, kaze_str_t key)
{
    uint32_t hash;
    bool found;
    size_t idx;

    if (m == NULL || m->entries == NULL || m->cap == 0) {
        return false;
    }
    hash = kaze_str_hash(key);
    idx = kaze_map_lookup(m, key, hash, &found);
    if (!found || idx == SIZE_MAX) {
        return false;
    }
    /* Tombstone rather than EMPTY: clearing the slot would cut every probe
     * chain that passes through it. */
    m->entries[idx].slot = KAZE_SLOT_TOMBSTONE;
    m->entries[idx].key.ptr = NULL;
    m->entries[idx].key.len = 0;
    m->entries[idx].val = kaze_val_unit();
    m->entries[idx].hash = 0;
    m->tombs++;
    m->len--;
    return true;
}

size_t map_len(const kaze_map_t *m)
{
    return (m != NULL) ? m->len : 0;
}

/* ------------------------------------------------------------------------- */
/* Values                                                                     */
/* ------------------------------------------------------------------------- */

kaze_val_t kaze_val_unit(void)
{
    kaze_val_t v;
    v.kind = KAZE_KIND_UNIT;
    v.as.text.ptr = NULL;
    v.as.text.len = 0;
    return v;
}

kaze_val_t kaze_val_bool(bool b)
{
    kaze_val_t v;
    v.kind = KAZE_KIND_BOOL;
    v.as.b = b;
    return v;
}

kaze_val_t kaze_val_number(kaze_float_t n)
{
    kaze_val_t v;
    v.kind = KAZE_KIND_NUMBER;
    v.as.num = n;
    return v;
}

kaze_val_t kaze_val_text(kaze_str_t s)
{
    kaze_val_t v;
    v.kind = KAZE_KIND_TEXT;
    v.as.text = s;
    return v;
}

kaze_val_t kaze_val_list(kaze_vec_t *list)
{
    kaze_val_t v;
    v.kind = KAZE_KIND_LIST;
    v.as.list = list;
    return v;
}

kaze_val_t kaze_val_map(kaze_map_t *map)
{
    kaze_val_t v;
    v.kind = KAZE_KIND_MAP;
    v.as.map = map;
    return v;
}

kaze_val_t kaze_val_point(kaze_float_t x, kaze_float_t y, kaze_float_t z)
{
    kaze_val_t v;
    v.kind = KAZE_KIND_POINT;
    v.as.point.x = x;
    v.as.point.y = y;
    v.as.point.z = z;
    return v;
}

bool kaze_val_truthy(kaze_val_t v)
{
    switch (v.kind) {
    case KAZE_KIND_UNIT:
        return false;
    case KAZE_KIND_BOOL:
        return v.as.b;
    case KAZE_KIND_NUMBER:
        return v.as.num != (kaze_float_t)0;
    case KAZE_KIND_TEXT:
        return v.as.text.len > 0;
    case KAZE_KIND_LIST:
    case KAZE_KIND_MAP:
    case KAZE_KIND_POINT:
        return true;
    default:
        return false;
    }
}

const char *kaze_val_kind_name(kaze_val_t v)
{
    switch (v.kind) {
    case KAZE_KIND_UNIT:
        return "nothing";
    case KAZE_KIND_BOOL:
        return "true/false";
    case KAZE_KIND_NUMBER:
        return "number";
    case KAZE_KIND_TEXT:
        return "text";
    case KAZE_KIND_LIST:
        return "list";
    case KAZE_KIND_MAP:
        return "record";
    case KAZE_KIND_POINT:
        return "point";
    default:
        return "unknown";
    }
}

/* ------------------------------------------------------------------------- */
/* Clock, watchdog                                                            */
/* ------------------------------------------------------------------------- */

uint64_t kaze_time_ms(void)
{
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;

    /* The monotonic clock cannot jump when the wall clock is corrected, which
     * is the whole point of a heartbeat window. */
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        if (ts.tv_sec < 0) {
            return 0;
        }
        return ((uint64_t)ts.tv_sec * 1000u) + ((uint64_t)ts.tv_nsec / 1000000u);
    }
#endif
    /* Fallback for targets without POSIX timers: clock() measures processor
     * time, so it under-reports while a controller is blocked on I/O. It is
     * still monotonic, which is what the watchdog needs, and it exists in
     * hosted C99 on every compiler. */
    {
        clock_t ticks = clock();

        if (ticks == (clock_t)-1) {
            return 0;
        }
        return ((uint64_t)ticks * 1000u) / (uint64_t)CLOCKS_PER_SEC;
    }
}

void kaze_watchdog_init(kaze_watchdog_t *w, uint64_t timeout_ms)
{
    if (w == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_watchdog_init: watchdog is NULL");
        return;
    }
    w->last_packet_ms = kaze_time_ms();
    w->timeout_ms = (timeout_ms != 0) ? timeout_ms : (uint64_t)KAZE_HEARTBEAT_TIMEOUT_MS;
}

void kaze_watchdog_packet(kaze_watchdog_t *w)
{
    if (w == NULL) {
        return;
    }
    w->last_packet_ms = kaze_time_ms();
}

bool kaze_watchdog_stale(const kaze_watchdog_t *w)
{
    uint64_t now;
    uint64_t timeout;

    if (w == NULL) {
        /* No watchdog state means no heartbeat evidence: fail closed. */
        return true;
    }
    now = kaze_time_ms();
    timeout = (w->timeout_ms != 0) ? w->timeout_ms : (uint64_t)KAZE_HEARTBEAT_TIMEOUT_MS;
    if (now < w->last_packet_ms) {
        /* A backwards clock means the window cannot be trusted; treat it as
         * stale rather than granting a fresh window to a link that may be
         * dead. */
        return true;
    }
    return (now - w->last_packet_ms) >= timeout;
}

kaze_code_t kaze_watchdog_abort_if_stale(const kaze_watchdog_t *w)
{
    if (!kaze_watchdog_stale(w)) {
        return KAZE_OK;
    }
    {
        char buf[128];
        uint64_t now = kaze_time_ms();
        uint64_t timeout = (w != NULL && w->timeout_ms != 0) ? w->timeout_ms
                                                            : (uint64_t)KAZE_HEARTBEAT_TIMEOUT_MS;
        uint64_t age = (w != NULL && now >= w->last_packet_ms) ? (now - w->last_packet_ms) : timeout;
        snprintf(buf, sizeof buf, "heartbeat stale: %llu ms since the last packet, limit %llu ms",
                 (unsigned long long)age, (unsigned long long)timeout);
        (void)kaze_trap(KAZE_ERR_SAFETY_VIOLATION, buf);
    }
    return KAZE_ERR_SAFETY_VIOLATION;
}

/* ------------------------------------------------------------------------- */
/* Signal-time cleanup                                                        */
/* ------------------------------------------------------------------------- */

#ifndef KAZE_NO_SIGNALS
/* POSIX _exit, declared here for the same include-allowlist reason as malloc.
 * It is the only way out of the handler: returning would resume the crashing
 * or interrupted instruction with actuators still armed. */
extern void _exit(int status);

static kaze_cleanup_fn kaze_deasserts[KAZE_MAX_DEASSERTS];
static size_t kaze_deassert_count_value = 0;
static volatile sig_atomic_t kaze_cleanup_running = 0;

bool kaze_register_deassert(kaze_cleanup_fn fn)
{
    if (fn == NULL) {
        (void)kaze_trap(KAZE_ERR_BADARG, "kaze_register_deassert: callback is NULL");
        return false;
    }
    if (kaze_deassert_count_value >= (size_t)KAZE_MAX_DEASSERTS) {
        (void)kaze_trap(KAZE_ERR_OOM, "kaze_register_deassert: all 16 de-assert slots are used");
        return false;
    }
    kaze_deasserts[kaze_deassert_count_value] = fn;
    kaze_deassert_count_value++;
    return true;
}

size_t kaze_deassert_count(void)
{
    return kaze_deassert_count_value;
}

void kaze_run_cleanups(void)
{
    size_t i = kaze_deassert_count_value;

    /* Reverse order: the most recently armed effector is de-asserted first,
     * which mirrors how a bring-up sequence is unwound. */
    while (i > 0) {
        i--;
        if (kaze_deasserts[i] != NULL) {
            kaze_deasserts[i]();
        }
    }
}

static void kaze_cleanup_handler(int sig)
{
    /* Async-signal-safe: the callbacks plus _exit. A callback that faults
     * re-enters here; the guard then goes straight to _exit instead of
     * looping on a broken stack. */
    if (kaze_cleanup_running) {
        _exit(128 + sig);
    }
    kaze_cleanup_running = 1;
    kaze_run_cleanups();
    _exit(128 + sig);
}

bool kaze_install_cleanup_signal_handlers(void)
{
    /* SIGSEGV is included deliberately: a crash is exactly the moment an
     * actuator must not stay armed. SIGKILL cannot be handled, so a watchdog
     * (or a hardware one) remains the only backstop there. */
    bool ok = true;

    if (signal(SIGINT, kaze_cleanup_handler) == SIG_ERR) {
        ok = false;
    }
    if (signal(SIGTERM, kaze_cleanup_handler) == SIG_ERR) {
        ok = false;
    }
    if (signal(SIGSEGV, kaze_cleanup_handler) == SIG_ERR) {
        ok = false;
    }
    return ok;
}
#endif /* KAZE_NO_SIGNALS */
