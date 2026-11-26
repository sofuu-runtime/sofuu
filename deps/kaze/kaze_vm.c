/* kaze_vm.c — the C99 resumable Kaze bytecode VM.
 *
 * Design notes a reader should be able to check:
 *   - One kaze_arena_t per VM, supplied by kz_vm_create and released by
 *     kz_vm_destroy. No static mutable state is used by execution.
 *   - The instruction pointer, operand stack, local slots, try frames and fuel
 *     live in kz_vm_t, so a slice can pause and resume at an exact instruction.
 *   - CALL_CAP is the only effect. The host callback may answer inline or ask
 *     the VM to pause; a paused call is resumed by kz_vm_supply_cap_result().
 *   - No exit(), abort() or longjmp() is used. A trap becomes a code + line in
 *     the VM and is reported to the caller.
 *   - Formats 1, 2, 4 and 5 are accepted. Format 2's line table is retained and
 *     used for diagnostics; formats 4/5 keep the root origin, and format 5 adds
 *     the per-item recursion ceiling.
 */

#include "kaze_embed.h"

#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <string.h>

#define KZ_VM_MAX_STACK 256u
#define KZ_VM_MAX_FRAMES 128u
#define KZ_VM_MAX_HANDLERS 64u
#define KZ_VM_MAX_ITEM_ARGS 255u
#define KZ_VM_MAX_LOCALS 256u
#define KZ_ERR_CODE_CAP 48u
#define KZ_ERR_MSG_CAP 1024u
#define KZ_MAX_JSON_DEPTH 64u
#define KZ_MAX_JSON_NODES 16384u

typedef struct kz_str {
    const char *p;
    size_t len;
} kz_str;

typedef struct kz_arr kz_arr;
typedef struct kz_map kz_map;

typedef enum {
    KZV_UNIT = 0,
    KZV_BOOL,
    KZV_NUM,
    KZV_STR,
    KZV_ARR,
    KZV_MAP,
    KZV_POINT
} kzv_kind;

typedef struct kz_value {
    kzv_kind kind;
    union {
        bool b;
        double n;
        kz_str s;
        kz_arr *arr;
        kz_map *map;
        struct { double x, y, z; } pt;
    } u;
} kz_value;

typedef struct kz_pair { kz_value key, value; } kz_pair;

struct kz_arr {
    size_t len;
    size_t cap;
    kz_value *items;
};

struct kz_map {
    size_t len;
    size_t cap;
    kz_pair *items;
};

typedef struct kz_const {
    uint8_t kind; /* 0 number, 1 text, 2 bool */
    double num;
    kz_str text;
    bool boolean;
} kz_const;

typedef struct kz_capref {
    kz_str family;
    kz_str op;
    uint8_t argc;
} kz_capref;

typedef struct kz_item {
    uint8_t tag;
    double period_ms;
    uint32_t line;
    uint8_t slots;
    uint16_t recursion_bound;
    kz_str name;
    kz_str origin;
    kz_str *params;
    size_t nparams;
    uint32_t *code;
    size_t ncode;
    uint32_t *line_map;
    size_t nline_map;
} kz_item;

typedef struct kz_frame {
    const kz_item *item;
    kz_value *locals;
    size_t nlocals;
    size_t local_base;
    size_t pc;
    uint32_t current_line;
    size_t stack_base;
    size_t handler_base;
} kz_frame;

typedef struct kz_handler {
    size_t target;
    size_t stack_len;
    uint8_t slot;
    const char *verdict_kind;
    const char *verdict_text;
    size_t frame_depth;
} kz_handler;

typedef enum {
    KZ_STATE_NEW = 0,
    KZ_STATE_RUNNING,
    KZ_STATE_PAUSED,
    KZ_STATE_DONE,
    KZ_STATE_ERR
} kz_state;

typedef struct kz_buf {
    kz_vm_t *vm;
    char *data;
    size_t len;
    size_t cap;
    int failed;
} kz_buf;

struct kz_vm_t {
    kaze_arena_t arena;
    bool arena_ready;

    uint8_t *bc;
    size_t bc_len;

    kz_str root;
    kz_str *needs;
    size_t nneeds;
    kz_const *consts;
    size_t nconsts;
    kz_capref *caps;
    size_t ncaps;
    kz_item *items;
    size_t nitems;

    kz_value stack[KZ_VM_MAX_STACK];
    size_t sp;
    /* Frame locals share one bounded pool. A return rewinds local_top, so a
     * routine called in a loop reuses its slots instead of leaking them. */
    kz_value *local_pool;
    size_t local_cap;
    size_t local_top;
    kz_frame frames[KZ_VM_MAX_FRAMES];
    size_t fp; /* number of frames; the active frame is fp-1 */
    kz_handler handlers[KZ_VM_MAX_HANDLERS];
    size_t nhandlers;

    uint64_t fuel_limit;
    uint64_t fuel_used;
    uint64_t instructions;
    size_t heap_limit;
    size_t heap_used;

    kz_state state;
    size_t entry_pc;
    kz_value last_value;

    const char *verdict_kind;
    const char *verdict_text;

    bool pending;
    kz_str pending_family;
    kz_str pending_verb;
    kz_str pending_args;
    uint32_t pending_line;

    char err_code[KZ_ERR_CODE_CAP];
    char err_msg[KZ_ERR_MSG_CAP];
    int err_line;

    kz_host_cbs_t host;
    bool has_host;

    char *cap_out;
    char *cap_err;
};

static char g_vm_last_error[256];

static bool kz_raise(kz_vm_t *vm, const char *code, int line, const char *fmt, ...);

static void kz_copy(char *dst, size_t cap, const char *src)
{
    size_t n;
    if (cap == 0) return;
    if (src == NULL) src = "";
    n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = 0;
}

static void kz_vm_set_error(kz_vm_t *vm, const char *code, int line, const char *fmt, ...)
{
    va_list ap;
    if (vm == NULL) return;
    kz_copy(vm->err_code, sizeof vm->err_code, code ? code : "E_KAZE_TRAP");
    vm->err_line = line;
    va_start(ap, fmt);
    vsnprintf(vm->err_msg, sizeof vm->err_msg, fmt ? fmt : "", ap);
    va_end(ap);
}

static void *kz_vm_alloc(kz_vm_t *vm, size_t n)
{
    void *p;
    if (vm == NULL || n == 0) return NULL;
    p = kmalloc(&vm->arena, n);
    if (p == NULL) {
        kz_vm_set_error(vm, "E_KAZE_MEMORY", 0,
                        "the VM arena (%llu bytes) could not satisfy a %llu-byte request",
                        (unsigned long long)vm->arena.cap,
                        (unsigned long long)n);
    }
    return p;
}

/* Rust's Heap meter: arrays charge 16 bytes per element + 24 for the
 * vector header, maps 32 per pair + 24, points 24. A push that would cross
 * the budget is refused, never clamped. */
static bool kz_heap_charge(kz_vm_t *vm, size_t bytes, int line)
{
    size_t next = vm->heap_used + bytes;
    if (next < vm->heap_used) next = (size_t)-1;
    if (next > vm->heap_limit) {
        vm->heap_used = next;
        return kz_raise(vm, "E_KAZE_MEMORY", line,
                        "the run's collection budget of %llu bytes was exceeded (arrays and maps are rejected, never clamped)",
                        (unsigned long long)vm->heap_limit) || false;
    }
    vm->heap_used = next;
    return true;
}

static char *kz_vm_strdup_len(kz_vm_t *vm, const char *p, size_t len)
{
    char *out = (char *)kz_vm_alloc(vm, len + 1);
    if (out == NULL) return NULL;
    if (len > 0) memcpy(out, p, len);
    out[len] = 0;
    return out;
}

static kz_str kz_str_from_cstr(const char *s)
{
    kz_str out;
    out.p = s ? s : "";
    out.len = s ? strlen(s) : 0;
    return out;
}

static kz_value kz_unit(void)
{
    kz_value v;
    memset(&v, 0, sizeof v);
    v.kind = KZV_UNIT;
    return v;
}

static kz_value kz_bool_v(bool b)
{
    kz_value v = kz_unit();
    v.kind = KZV_BOOL;
    v.u.b = b;
    return v;
}

static kz_value kz_num_v(double n)
{
    kz_value v = kz_unit();
    v.kind = KZV_NUM;
    v.u.n = n;
    return v;
}

static kz_value kz_str_v(kz_str s)
{
    kz_value v = kz_unit();
    v.kind = KZV_STR;
    v.u.s = s;
    return v;
}

static kz_value kz_point_v(double x, double y, double z)
{
    kz_value v = kz_unit();
    v.kind = KZV_POINT;
    v.u.pt.x = x;
    v.u.pt.y = y;
    v.u.pt.z = z;
    return v;
}

static bool kz_is_truthy(kz_value v)
{
    switch (v.kind) {
    case KZV_BOOL: return v.u.b;
    case KZV_NUM: return v.u.n != 0.0;
    case KZV_STR: return v.u.s.len > 0;
    case KZV_ARR: return v.u.arr != NULL && v.u.arr->len > 0;
    case KZV_MAP: return true;
    case KZV_POINT: return true;
    case KZV_UNIT:
    default: return false;
    }
}

static const char *kz_type_name(kz_value v)
{
    switch (v.kind) {
    case KZV_UNIT: return "nothing";
    case KZV_BOOL: return "true/false";
    case KZV_NUM: return "number";
    case KZV_STR: return "text";
    case KZV_ARR: return "list";
    case KZV_MAP: return "record";
    case KZV_POINT: return "point";
    default: return "value";
    }
}

static bool kz_buf_init(kz_vm_t *vm, kz_buf *b, size_t cap)
{
    if (cap < 64) cap = 64;
    b->vm = vm;
    b->data = (char *)kz_vm_alloc(vm, cap);
    b->len = 0;
    b->cap = cap;
    b->failed = b->data == NULL;
    if (!b->failed) b->data[0] = 0;
    return !b->failed;
}

static bool kz_buf_reserve(kz_vm_t *vm, kz_buf *b, size_t need)
{
    char *next;
    size_t cap;
    if (b->failed) return false;
    if (need <= b->cap) return true;
    cap = b->cap;
    while (cap < need) {
        if (cap > (size_t)-1 / 2) { b->failed = 1; return false; }
        cap *= 2;
    }
    next = (char *)kz_vm_alloc(vm, cap);
    if (next == NULL) { b->failed = 1; return false; }
    if (b->len > 0) memcpy(next, b->data, b->len);
    next[b->len] = 0;
    b->data = next;
    b->cap = cap;
    return true;
}

static bool kz_buf_putn(kz_buf *b, const char *s, size_t n)
{
    if (b->failed) return false;
    if (b->len + n + 1 > b->cap && !kz_buf_reserve(b->vm, b, b->len + n + 1)) return false;
    if (n > 0) memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = 0;
    return true;
}

static bool kz_buf_puts(kz_buf *b, const char *s)
{
    return kz_buf_putn(b, s, strlen(s));
}

static bool kz_buf_putc(kz_buf *b, char c)
{
    return kz_buf_putn(b, &c, 1);
}

static bool kz_buf_printf(kz_buf *b, const char *fmt, ...)
{
    char tmp[128];
    va_list ap;
    int n;
    va_start(ap, fmt);
    n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0) { b->failed = 1; return false; }
    if ((size_t)n >= sizeof tmp) { b->failed = 1; return false; }
    return kz_buf_putn(b, tmp, (size_t)n);
}

static bool kz_buf_put_json_string(kz_buf *b, const char *s, size_t len)
{
    size_t i;
    if (!kz_buf_putc(b, '"')) return false;
    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': if (!kz_buf_putn(b, "\\\"", 2)) return false; break;
        case '\\': if (!kz_buf_putn(b, "\\\\", 2)) return false; break;
        case '\b': if (!kz_buf_putn(b, "\\b", 2)) return false; break;
        case '\f': if (!kz_buf_putn(b, "\\f", 2)) return false; break;
        case '\n': if (!kz_buf_putn(b, "\\n", 2)) return false; break;
        case '\r': if (!kz_buf_putn(b, "\\r", 2)) return false; break;
        case '\t': if (!kz_buf_putn(b, "\\t", 2)) return false; break;
        default:
            if (c < 0x20) {
                if (!kz_buf_printf(b, "\\u%04x", (unsigned)c)) return false;
            } else if (!kz_buf_putc(b, (char)c)) {
                return false;
            }
        }
    }
    return kz_buf_putc(b, '"');
}

static bool kz_buf_put_render(kz_vm_t *vm, kz_buf *b, kz_value v, unsigned depth);

static bool kz_buf_put_json(kz_vm_t *vm, kz_buf *b, kz_value v, unsigned depth)
{
    size_t i;
    if (depth > KZ_MAX_JSON_DEPTH) { b->failed = 1; return false; }
    switch (v.kind) {
    case KZV_UNIT: return kz_buf_puts(b, "null");
    case KZV_BOOL: return kz_buf_puts(b, v.u.b ? "true" : "false");
    case KZV_NUM:
        if (!isfinite(v.u.n)) { b->failed = 1; return false; }
        if (v.u.n == 0.0) return kz_buf_putc(b, '0');
        return kz_buf_printf(b, "%.17g", v.u.n);
    case KZV_STR:
        return kz_buf_put_json_string(b, v.u.s.p, v.u.s.len);
    case KZV_ARR:
        if (!kz_buf_putc(b, '[')) return false;
        if (v.u.arr != NULL) {
            for (i = 0; i < v.u.arr->len; i++) {
                if (i > 0 && !kz_buf_putc(b, ',')) return false;
                if (!kz_buf_put_json(vm, b, v.u.arr->items[i], depth + 1)) return false;
            }
        }
        return kz_buf_putc(b, ']');
    case KZV_MAP:
        if (!kz_buf_putc(b, '{')) return false;
        if (v.u.map != NULL) {
            for (i = 0; i < v.u.map->len; i++) {
                kz_pair *p = &v.u.map->items[i];
                if (i > 0 && !kz_buf_putc(b, ',')) return false;
                if (p->key.kind != KZV_STR) { b->failed = 1; return false; }
                if (!kz_buf_put_json_string(b, p->key.u.s.p, p->key.u.s.len)) return false;
                if (!kz_buf_putc(b, ':')) return false;
                if (!kz_buf_put_json(vm, b, p->value, depth + 1)) return false;
            }
        }
        return kz_buf_putc(b, '}');
    case KZV_POINT: {
        char tmp[64];
        snprintf(tmp, sizeof tmp, "[%.17g,%.17g,%.17g]", v.u.pt.x, v.u.pt.y, v.u.pt.z);
        return kz_buf_puts(b, tmp);
    }
    default: b->failed = 1; return false;
    }
}

static bool kz_buf_put_num(kz_buf *b, double n)
{
    char tmp[64];
    if (!isfinite(n)) return false;
    if (n == 0.0) return kz_buf_putc(b, '0');
    if (n == (double)(long long)n && n > -1e15 && n < 1e15) {
        snprintf(tmp, sizeof tmp, "%lld", (long long)n);
    } else {
        snprintf(tmp, sizeof tmp, "%.17g", n);
    }
    return kz_buf_puts(b, tmp);
}

static bool kz_buf_put_render(kz_vm_t *vm, kz_buf *b, kz_value v, unsigned depth)
{
    size_t i;
    if (depth > KZ_MAX_JSON_DEPTH) { b->failed = 1; return false; }
    switch (v.kind) {
    case KZV_UNIT: return kz_buf_puts(b, "nothing");
    case KZV_BOOL: return kz_buf_puts(b, v.u.b ? "true" : "false");
    case KZV_NUM: return kz_buf_put_num(b, v.u.n);
    case KZV_STR:
        if (!kz_buf_putc(b, '"')) return false;
        if (!kz_buf_putn(b, v.u.s.p, v.u.s.len)) return false;
        return kz_buf_putc(b, '"');
    case KZV_ARR:
        if (!kz_buf_putc(b, '[')) return false;
        if (v.u.arr != NULL) {
            for (i = 0; i < v.u.arr->len; i++) {
                if (i > 0 && !kz_buf_puts(b, ", ")) return false;
                if (!kz_buf_put_render(vm, b, v.u.arr->items[i], depth + 1)) return false;
            }
        }
        return kz_buf_putc(b, ']');
    case KZV_MAP:
        if (!kz_buf_putc(b, '{')) return false;
        if (v.u.map != NULL) {
            for (i = 0; i < v.u.map->len; i++) {
                kz_pair *p = &v.u.map->items[i];
                if (i > 0 && !kz_buf_puts(b, ", ")) return false;
                if (p->key.kind == KZV_STR) {
                    if (!kz_buf_putn(b, p->key.u.s.p, p->key.u.s.len)) return false;
                } else if (!kz_buf_put_render(vm, b, p->key, depth + 1)) {
                    return false;
                }
                if (!kz_buf_puts(b, ": ")) return false;
                if (!kz_buf_put_render(vm, b, p->value, depth + 1)) return false;
            }
        }
        return kz_buf_putc(b, '}');
    case KZV_POINT:
        if (!kz_buf_puts(b, "vector(")) return false;
        if (!kz_buf_put_num(b, v.u.pt.x) || !kz_buf_puts(b, ", ") ||
            !kz_buf_put_num(b, v.u.pt.y) || !kz_buf_puts(b, ", ") ||
            !kz_buf_put_num(b, v.u.pt.z) || !kz_buf_putc(b, ')')) return false;
        return true;
    default: b->failed = 1; return false;
    }
}

/* ── JSON reader ─────────────────────────────────────────────────────── */

typedef struct kz_json {
    kz_vm_t *vm;
    const char *p;
    size_t len;
    size_t at;
    size_t nodes;
} kz_json;

static void kz_json_ws(kz_json *j)
{
    while (j->at < j->len) {
        char c = j->p[j->at];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') j->at++;
        else break;
    }
}

static int kz_json_hex(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool kz_json_read_string(kz_json *j, kz_str *out)
{
    kz_buf b;
    if (!kz_buf_init(j->vm, &b, 64)) return false;
    if (j->at >= j->len || j->p[j->at] != '"') return false;
    j->at++;
    while (j->at < j->len) {
        unsigned char c = (unsigned char)j->p[j->at++];
        if (c == '"') {
            *out = (kz_str){ b.data, b.len };
            return true;
        }
        if (c != '\\') {
            if (!kz_buf_putc(&b, (char)c)) return false;
            continue;
        }
        if (j->at >= j->len) return false;
        c = (unsigned char)j->p[j->at++];
        switch (c) {
        case '"': if (!kz_buf_putc(&b, '"')) return false; break;
        case '\\': if (!kz_buf_putc(&b, '\\')) return false; break;
        case '/': if (!kz_buf_putc(&b, '/')) return false; break;
        case 'b': if (!kz_buf_putc(&b, '\b')) return false; break;
        case 'f': if (!kz_buf_putc(&b, '\f')) return false; break;
        case 'n': if (!kz_buf_putc(&b, '\n')) return false; break;
        case 'r': if (!kz_buf_putc(&b, '\r')) return false; break;
        case 't': if (!kz_buf_putc(&b, '\t')) return false; break;
        case 'u': {
            unsigned cp = 0;
            int k;
            if (j->at + 4 > j->len) return false;
            for (k = 0; k < 4; k++) {
                int h = kz_json_hex(j->p[j->at + (size_t)k]);
                if (h < 0) return false;
                cp = cp * 16u + (unsigned)h;
            }
            j->at += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF && j->at + 6 <= j->len &&
                j->p[j->at] == '\\' && j->p[j->at + 1] == 'u') {
                unsigned lo = 0, kk;
                bool ok = true;
                for (kk = 0; kk < 4; kk++) {
                    int h = kz_json_hex(j->p[j->at + 2 + (size_t)kk]);
                    if (h < 0) { ok = false; break; }
                    lo = lo * 16u + (unsigned)h;
                }
                if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                    j->at += 6;
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                }
            }
            if (cp < 0x80) {
                if (!kz_buf_putc(&b, (char)cp)) return false;
            } else if (cp < 0x800) {
                char x[2];
                x[0] = (char)(0xC0 | (cp >> 6));
                x[1] = (char)(0x80 | (cp & 0x3F));
                if (!kz_buf_putn(&b, x, 2)) return false;
            } else if (cp < 0x10000) {
                char x[3];
                x[0] = (char)(0xE0 | (cp >> 12));
                x[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                x[2] = (char)(0x80 | (cp & 0x3F));
                if (!kz_buf_putn(&b, x, 3)) return false;
            } else {
                char x[4];
                x[0] = (char)(0xF0 | (cp >> 18));
                x[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
                x[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
                x[3] = (char)(0x80 | (cp & 0x3F));
                if (!kz_buf_putn(&b, x, 4)) return false;
            }
            break;
        }
        default: return false;
        }
    }
    return false;
}

static kz_value kz_json_value(kz_json *j, unsigned depth);

static bool kz_arr_push(kz_vm_t *vm, kz_arr *a, kz_value v)
{
    if (!kz_heap_charge(vm, 16u, 0)) return false;
    if (a->len >= a->cap) {
        size_t cap = a->cap == 0 ? 4 : a->cap * 2;
        kz_value *items = (kz_value *)kz_vm_alloc(vm, cap * sizeof *items);
        if (items == NULL) return false;
        if (a->len > 0) memcpy(items, a->items, a->len * sizeof *items);
        a->items = items;
        a->cap = cap;
    }
    a->items[a->len++] = v;
    return true;
}

static bool kz_map_push(kz_vm_t *vm, kz_map *m, kz_value key, kz_value value)
{
    if (!kz_heap_charge(vm, 32u, 0)) return false;
    if (m->len >= m->cap) {
        size_t cap = m->cap == 0 ? 4 : m->cap * 2;
        kz_pair *items = (kz_pair *)kz_vm_alloc(vm, cap * sizeof *items);
        if (items == NULL) return false;
        if (m->len > 0) memcpy(items, m->items, m->len * sizeof *items);
        m->items = items;
        m->cap = cap;
    }
    m->items[m->len].key = key;
    m->items[m->len].value = value;
    m->len++;
    return true;
}

static kz_value kz_json_value(kz_json *j, unsigned depth)
{
    kz_value out = kz_unit();
    if (depth > KZ_MAX_JSON_DEPTH || j->nodes++ > KZ_MAX_JSON_NODES) return out;
    kz_json_ws(j);
    if (j->at >= j->len) return out;
    {
        char c = j->p[j->at];
        if (c == 'n') {
            if (j->at + 4 <= j->len && memcmp(j->p + j->at, "null", 4) == 0) {
                j->at += 4;
                return kz_unit();
            }
        } else if (c == 't') {
            if (j->at + 4 <= j->len && memcmp(j->p + j->at, "true", 4) == 0) {
                j->at += 4;
                return kz_bool_v(true);
            }
        } else if (c == 'f') {
            if (j->at + 5 <= j->len && memcmp(j->p + j->at, "false", 5) == 0) {
                j->at += 5;
                return kz_bool_v(false);
            }
        } else if (c == '"') {
            kz_str s;
            if (kz_json_read_string(j, &s)) return kz_str_v(s);
        } else if (c == '[') {
            kz_arr *a = (kz_arr *)kz_vm_alloc(j->vm, sizeof *a);
            if (a == NULL) return out;
            memset(a, 0, sizeof *a);
            j->at++;
            kz_json_ws(j);
            if (j->at < j->len && j->p[j->at] == ']') {
                j->at++;
            } else {
                for (;;) {
                    kz_value item = kz_json_value(j, depth + 1);
                    if (!kz_arr_push(j->vm, a, item)) return out;
                    kz_json_ws(j);
                    if (j->at < j->len && j->p[j->at] == ',') { j->at++; continue; }
                    if (j->at < j->len && j->p[j->at] == ']') { j->at++; break; }
                    return out;
                }
            }
            out.kind = KZV_ARR;
            out.u.arr = a;
            return out;
        } else if (c == '{') {
            kz_map *m = (kz_map *)kz_vm_alloc(j->vm, sizeof *m);
            if (m == NULL) return out;
            memset(m, 0, sizeof *m);
            j->at++;
            kz_json_ws(j);
            if (j->at < j->len && j->p[j->at] == '}') {
                j->at++;
            } else {
                for (;;) {
                    kz_str key;
                    kz_value val;
                    kz_json_ws(j);
                    if (!kz_json_read_string(j, &key)) return out;
                    kz_json_ws(j);
                    if (j->at >= j->len || j->p[j->at] != ':') return out;
                    j->at++;
                    val = kz_json_value(j, depth + 1);
                    if (!kz_map_push(j->vm, m, kz_str_v(key), val)) return out;
                    kz_json_ws(j);
                    if (j->at < j->len && j->p[j->at] == ',') { j->at++; continue; }
                    if (j->at < j->len && j->p[j->at] == '}') { j->at++; break; }
                    return out;
                }
            }
            out.kind = KZV_MAP;
            out.u.map = m;
            return out;
        } else {
            char *end = NULL;
            double n;
            n = strtod(j->p + j->at, &end);
            if (end != NULL && end != j->p + j->at) {
                j->at = (size_t)(end - j->p);
                return kz_num_v(n);
            }
        }
    }
    return out;
}

static bool kz_json_parse_value(kz_vm_t *vm, const char *text, kz_value *out)
{
    kz_json j;
    j.vm = vm;
    j.p = text;
    j.len = strlen(text);
    j.at = 0;
    j.nodes = 0;
    *out = kz_json_value(&j, 0);
    kz_json_ws(&j);
    return j.at == j.len;
}

/* ── bytecode reader ─────────────────────────────────────────────────── */

typedef struct kz_reader {
    kz_vm_t *vm;
    const uint8_t *p;
    size_t len;
    size_t at;
    bool bad;
} kz_reader;

static uint8_t rd_u8(kz_reader *r)
{
    if (r->at + 1 > r->len) { r->bad = true; return 0; }
    return r->p[r->at++];
}

static uint16_t rd_u16(kz_reader *r)
{
    uint16_t v;
    if (r->at + 2 > r->len) { r->bad = true; return 0; }
    v = (uint16_t)(r->p[r->at] | ((uint16_t)r->p[r->at + 1] << 8));
    r->at += 2;
    return v;
}

static uint32_t rd_u32(kz_reader *r)
{
    uint32_t v;
    if (r->at + 4 > r->len) { r->bad = true; return 0; }
    v = (uint32_t)r->p[r->at] | ((uint32_t)r->p[r->at + 1] << 8) |
        ((uint32_t)r->p[r->at + 2] << 16) | ((uint32_t)r->p[r->at + 3] << 24);
    r->at += 4;
    return v;
}

static double rd_f64(kz_reader *r)
{
    uint64_t bits = 0;
    double v;
    int i;
    if (r->at + 8 > r->len) { r->bad = true; return 0.0; }
    for (i = 0; i < 8; i++) bits |= (uint64_t)r->p[r->at + (size_t)i] << (8 * i);
    r->at += 8;
    memcpy(&v, &bits, sizeof v);
    return v;
}

static kz_str rd_text(kz_reader *r)
{
    kz_str s;
    uint32_t n = rd_u32(r);
    if (r->bad) return (kz_str){ "", 0 };
    if ((size_t)n > r->len - r->at) { r->bad = true; return (kz_str){ "", 0 }; }
    {
        char *copy = kz_vm_strdup_len(r->vm, (const char *)(r->p + r->at), n);
        if (copy == NULL) { r->bad = true; return (kz_str){ "", 0 }; }
        s.p = copy;
        s.len = n;
    }
    r->at += n;
    return s;
}

static bool kz_parse_module(kz_vm_t *vm)
{
    kz_reader r;
    uint16_t version;
    uint32_t n, i, j;
    static const uint8_t magic[8] = { 'K','A','Z','E','B','C','1',0 };

    memset(&r, 0, sizeof r);
    r.vm = vm;
    r.p = vm->bc;
    r.len = vm->bc_len;
    if (r.len < 10 || memcmp(r.p, magic, 8) != 0) {
        kz_vm_set_error(vm, "E_KAZE_BADCODE", 0, "not a Kaze module: the magic number does not match");
        return false;
    }
    r.at = 8;
    version = rd_u16(&r);
    if (version != 1 && version != 2 && version != 4 && version != 5) {
        kz_vm_set_error(vm, "E_KAZE_BADCODE", 0,
                        "unsupported Kaze bytecode version %u; supported versions are 1, 2, 4, and 5",
                        (unsigned)version);
        return false;
    }
    if (version >= 4) vm->root = rd_text(&r);
    if (r.bad) goto truncated;

    n = rd_u32(&r);
    if (r.bad || n > 65536u) goto truncated;
    vm->nneeds = n;
    if (n > 0) {
        vm->needs = (kz_str *)kz_vm_alloc(vm, (size_t)n * sizeof *vm->needs);
        if (vm->needs == NULL) return false;
    }
    for (i = 0; i < n; i++) {
        vm->needs[i] = rd_text(&r);
        if (r.bad) goto truncated;
    }

    n = rd_u32(&r);
    if (r.bad || n > 1000000u) goto truncated;
    vm->nconsts = n;
    if (n > 0) {
        vm->consts = (kz_const *)kz_vm_alloc(vm, (size_t)n * sizeof *vm->consts);
        if (vm->consts == NULL) return false;
        memset(vm->consts, 0, (size_t)n * sizeof *vm->consts);
    }
    for (i = 0; i < n; i++) {
        kz_const *c = &vm->consts[i];
        c->kind = rd_u8(&r);
        if (c->kind == 0) c->num = rd_f64(&r);
        else if (c->kind == 1) c->text = rd_text(&r);
        else if (c->kind == 2) c->boolean = rd_u8(&r) != 0;
        else { r.bad = true; }
        if (r.bad) goto truncated;
    }

    n = rd_u32(&r);
    if (r.bad || n > 1000000u) goto truncated;
    vm->ncaps = n;
    if (n > 0) {
        vm->caps = (kz_capref *)kz_vm_alloc(vm, (size_t)n * sizeof *vm->caps);
        if (vm->caps == NULL) return false;
    }
    for (i = 0; i < n; i++) {
        vm->caps[i].family = rd_text(&r);
        vm->caps[i].op = rd_text(&r);
        vm->caps[i].argc = rd_u8(&r);
        if (r.bad) goto truncated;
    }

    n = rd_u32(&r);
    if (r.bad || n > 1000000u) goto truncated;
    vm->nitems = n;
    if (n > 0) {
        vm->items = (kz_item *)kz_vm_alloc(vm, (size_t)n * sizeof *vm->items);
        if (vm->items == NULL) return false;
        memset(vm->items, 0, (size_t)n * sizeof *vm->items);
    }
    for (i = 0; i < n; i++) {
        kz_item *it = &vm->items[i];
        it->tag = rd_u8(&r);
        it->period_ms = rd_f64(&r);
        it->line = rd_u32(&r);
        it->slots = rd_u8(&r);
        if (version >= 5) it->recursion_bound = rd_u16(&r);
        it->name = rd_text(&r);
        if (version >= 4) it->origin = rd_text(&r);
        if (r.bad || it->tag > 4) goto truncated;
        j = rd_u32(&r);
        if (r.bad || j > 255u) goto truncated;
        it->nparams = j;
        if (j > 0) {
            it->params = (kz_str *)kz_vm_alloc(vm, (size_t)j * sizeof *it->params);
            if (it->params == NULL) return false;
        }
        {
            uint32_t k;
            for (k = 0; k < j; k++) {
                it->params[k] = rd_text(&r);
                if (r.bad) goto truncated;
            }
        }
        j = rd_u32(&r);
        if (r.bad || j > 67108864u) goto truncated;
        it->ncode = j;
        if (j > 0) {
            uint32_t k;
            it->code = (uint32_t *)kz_vm_alloc(vm, (size_t)j * sizeof *it->code);
            if (it->code == NULL) return false;
            for (k = 0; k < j; k++) {
                it->code[k] = rd_u32(&r);
                if (r.bad) goto truncated;
            }
        }
        if (version == 2) {
            j = rd_u32(&r);
            if (r.bad || (j != 0 && j != it->ncode) || j > 67108864u) goto truncated;
            it->nline_map = j;
            if (j > 0) {
                uint32_t k;
                it->line_map = (uint32_t *)kz_vm_alloc(vm, (size_t)j * sizeof *it->line_map);
                if (it->line_map == NULL) return false;
                for (k = 0; k < j; k++) {
                    it->line_map[k] = rd_u32(&r);
                    if (r.bad) goto truncated;
                }
            }
        }
    }
    return true;

truncated:
    kz_vm_set_error(vm, "E_KAZE_BADCODE", 0, "truncated or invalid Kaze module");
    return false;
}

/* ── VM lifecycle ────────────────────────────────────────────────────── */

kz_vm_t *kz_vm_create(const uint8_t *bc, size_t bc_len, size_t arena_bytes, uint64_t fuel_limit)
{
    kz_vm_t *vm;
    kaze_arena_t arena;
    size_t bytes = arena_bytes == 0 ? (size_t)64 * 1024 * 1024 : arena_bytes;

    g_vm_last_error[0] = 0;
    if (bc == NULL || bc_len == 0) {
        kz_copy(g_vm_last_error, sizeof g_vm_last_error, "the module is empty");
        return NULL;
    }
    if (!kaze_arena_init_heap(&arena, bytes)) {
        kz_copy(g_vm_last_error, sizeof g_vm_last_error, "cannot reserve the VM arena");
        return NULL;
    }
    vm = (kz_vm_t *)kmalloc(&arena, sizeof *vm);
    if (vm == NULL) {
        kaze_arena_free(&arena);
        kz_copy(g_vm_last_error, sizeof g_vm_last_error, "cannot allocate the VM header");
        return NULL;
    }
    memset(vm, 0, sizeof *vm);
    vm->arena = arena;
    vm->arena_ready = true;
    vm->fuel_limit = fuel_limit;
    vm->heap_limit = (size_t)4 * 1024u * 1024u;
    vm->state = KZ_STATE_NEW;
    vm->last_value = kz_unit();
    vm->bc = (uint8_t *)kz_vm_alloc(vm, bc_len);
    if (vm->bc == NULL) {
        kz_copy(g_vm_last_error, sizeof g_vm_last_error, vm->err_msg);
        kz_vm_destroy(vm);
        return NULL;
    }
    memcpy(vm->bc, bc, bc_len);
    vm->bc_len = bc_len;
    vm->cap_out = (char *)kz_vm_alloc(vm, KZ_CAP_JSON_MAX + 1u);
    vm->cap_err = (char *)kz_vm_alloc(vm, 4096);
    vm->local_cap = (size_t)KZ_VM_MAX_FRAMES * KZ_VM_MAX_LOCALS;
    vm->local_pool = (kz_value *)kz_vm_alloc(vm, vm->local_cap * sizeof *vm->local_pool);
    if (vm->cap_out == NULL || vm->cap_err == NULL || vm->local_pool == NULL) {
        kz_copy(g_vm_last_error, sizeof g_vm_last_error, vm->err_msg);
        kz_vm_destroy(vm);
        return NULL;
    }
    vm->cap_out[0] = 0;
    vm->cap_err[0] = 0;
    if (!kz_parse_module(vm)) {
        kz_copy(g_vm_last_error, sizeof g_vm_last_error,
                vm->err_msg[0] ? vm->err_msg : "invalid module");
        kz_vm_destroy(vm);
        return NULL;
    }
    return vm;
}

void kz_vm_destroy(kz_vm_t *vm)
{
    kaze_arena_t arena;
    if (vm == NULL) return;
    if (!vm->arena_ready) return;
    /* The VM header itself lives inside the arena's first block, so the
     * header must be copied out before the blocks are released. Freeing
     * through &vm->arena and then touching it would be a use-after-free. */
    arena = vm->arena;
    kaze_arena_free(&arena);
}

int kz_vm_set_heap_limit(kz_vm_t *vm, size_t heap_bytes)
{
    if (vm == NULL) return 0;
    vm->heap_limit = heap_bytes;
    return 1;
}

int kz_vm_set_host(kz_vm_t *vm, const kz_host_cbs_t *cbs)
{
    if (vm == NULL) return 0;
    if (cbs == NULL) {
        memset(&vm->host, 0, sizeof vm->host);
        vm->has_host = false;
    } else {
        vm->host = *cbs;
        vm->has_host = true;
    }
    return 1;
}

size_t kz_vm_item_count(const kz_vm_t *vm) { return vm ? vm->nitems : 0; }

const char *kz_vm_item_name(const kz_vm_t *vm, size_t index)
{
    if (vm == NULL || index >= vm->nitems) return "";
    return vm->items[index].name.p;
}

const char *kz_vm_item_tag(const kz_vm_t *vm, size_t index)
{
    static const char *names[] = { "action", "check", "filter", "when", "every" };
    if (vm == NULL || index >= vm->nitems || vm->items[index].tag > 4) return "";
    return names[vm->items[index].tag];
}

size_t kz_vm_item_param_count(const kz_vm_t *vm, size_t index)
{
    if (vm == NULL || index >= vm->nitems) return 0;
    return vm->items[index].nparams;
}

const char *kz_vm_item_param_name(const kz_vm_t *vm, size_t index, size_t param)
{
    if (vm == NULL || index >= vm->nitems || param >= vm->items[index].nparams) return "";
    return vm->items[index].params[param].p;
}

double kz_vm_item_period_ms(const kz_vm_t *vm, size_t index)
{
    return (vm && index < vm->nitems) ? vm->items[index].period_ms : 0.0;
}

uint32_t kz_vm_item_line(const kz_vm_t *vm, size_t index)
{
    return (vm && index < vm->nitems) ? vm->items[index].line : 0;
}

size_t kz_vm_needs_count(const kz_vm_t *vm) { return vm ? vm->nneeds : 0; }

const char *kz_vm_need(const kz_vm_t *vm, size_t index)
{
    if (vm == NULL || index >= vm->nneeds) return "";
    return vm->needs[index].p;
}

size_t kz_vm_const_count(const kz_vm_t *vm) { return vm ? vm->nconsts : 0; }
size_t kz_vm_cap_count(const kz_vm_t *vm) { return vm ? vm->ncaps : 0; }

size_t kz_vm_code_bytes(const kz_vm_t *vm)
{
    size_t i, total = 0;
    if (vm == NULL) return 0;
    for (i = 0; i < vm->nitems; i++) total += vm->items[i].ncode * sizeof(uint32_t);
    return total;
}

const char *kz_vm_root(const kz_vm_t *vm) { return vm ? vm->root.p : ""; }
const char *kz_vm_last_error(void) { return g_vm_last_error; }

/* ── errors and stack ────────────────────────────────────────────────── */

static bool kz_is_catchable(const char *code)
{
    return code != NULL && (
        strcmp(code, "E_KAZE_LINK") == 0 ||
        strcmp(code, "E_KAZE_LINK_TIMEOUT") == 0 ||
        strcmp(code, "E_COMMAND_REJECTED") == 0 ||
        strcmp(code, "E_ARMING_DENIED") == 0 ||
        strcmp(code, "E_KAZE_HOST") == 0);
}

static kz_str kz_alloc_str(kz_vm_t *vm, const char *s)
{
    kz_str out;
    size_t n = s ? strlen(s) : 0;
    char *p = kz_vm_strdup_len(vm, s ? s : "", n);
    out.p = p ? p : "";
    out.len = p ? n : 0;
    return out;
}

static bool kz_error_record(kz_vm_t *vm, kz_value *out, const char *code, const char *msg, int line)
{
    kz_map *m = (kz_map *)kz_vm_alloc(vm, sizeof *m);
    kz_str k_code, k_msg, k_line;
    if (m == NULL) return false;
    memset(m, 0, sizeof *m);
    k_code = kz_alloc_str(vm, "code");
    k_msg = kz_alloc_str(vm, "message");
    k_line = kz_alloc_str(vm, "line");
    if (!kz_map_push(vm, m, kz_str_v(k_code), kz_str_v(kz_alloc_str(vm, code)))) return false;
    if (!kz_map_push(vm, m, kz_str_v(k_msg), kz_str_v(kz_alloc_str(vm, msg)))) return false;
    if (!kz_map_push(vm, m, kz_str_v(k_line), kz_num_v((double)line))) return false;
    out->kind = KZV_MAP;
    out->u.map = m;
    return true;
}

/* Raise a trap. Returns true when a try frame caught it and execution may
 * continue. */
static bool kz_raise(kz_vm_t *vm, const char *code, int line, const char *fmt, ...)
{
    char msg[KZ_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt ? fmt : "", ap);
    va_end(ap);
    if (kz_is_catchable(code) && vm->nhandlers > 0) {
        kz_handler h = vm->handlers[--vm->nhandlers];
        kz_frame *f2 = &vm->frames[vm->fp - 1];
        if (h.stack_len <= vm->sp) vm->sp = h.stack_len;
        vm->verdict_kind = h.verdict_kind;
        vm->verdict_text = h.verdict_text;
        if (h.slot != 255) {
            if (h.slot >= f2->nlocals) {
                kz_vm_set_error(vm, "E_KAZE_BADCODE", line, "the error slot in TRY_BEGIN does not exist");
                vm->state = KZ_STATE_ERR;
                return false;
            }
            if (!kz_error_record(vm, &f2->locals[h.slot], code, msg, line)) {
                vm->state = KZ_STATE_ERR;
                return false;
            }
        }
        f2->pc = h.target;
        vm->err_code[0] = 0;
        vm->err_msg[0] = 0;
        vm->err_line = 0;
        return true;
    }
    kz_vm_set_error(vm, code, line, "%s", msg);
    vm->state = KZ_STATE_ERR;
    return false;
}

static bool kz_push(kz_vm_t *vm, kz_value v, int line)
{
    if (vm->sp >= KZ_VM_MAX_STACK) {
        return kz_raise(vm, "E_KAZE_STACK", line,
                        "this expression nests too deeply; split it into smaller steps") || false;
    }
    vm->stack[vm->sp++] = v;
    return true;
}

static bool kz_pop(kz_vm_t *vm, kz_value *out, int line)
{
    if (vm->sp == 0) {
        return kz_raise(vm, "E_KAZE_BADCODE", line, "the value stack underflowed") || false;
    }
    *out = vm->stack[--vm->sp];
    return true;
}

/* ── values and operations ───────────────────────────────────────────── */

static bool kz_num_of(kz_value v, double *out)
{
    if (v.kind != KZV_NUM) return false;
    *out = v.u.n;
    return true;
}

static kz_value kz_const_value(kz_vm_t *vm, size_t idx)
{
    if (idx >= vm->nconsts) return kz_unit();
    {
        kz_const *c = &vm->consts[idx];
        if (c->kind == 0) return kz_num_v(c->num);
        if (c->kind == 1) return kz_str_v(c->text);
        return kz_bool_v(c->boolean);
    }
}

static bool kz_values_equal(kz_value a, kz_value b, unsigned depth)
{
    size_t i;
    if (depth > KZ_MAX_JSON_DEPTH) return false;
    if (a.kind != b.kind) {
        /* Numbers and booleans are never equal across kinds. */
        return false;
    }
    switch (a.kind) {
    case KZV_UNIT: return true;
    case KZV_BOOL: return a.u.b == b.u.b;
    case KZV_NUM: return a.u.n == b.u.n;
    case KZV_STR: return a.u.s.len == b.u.s.len && memcmp(a.u.s.p, b.u.s.p, a.u.s.len) == 0;
    case KZV_ARR:
        if (a.u.arr == NULL || b.u.arr == NULL) return a.u.arr == b.u.arr;
        if (a.u.arr->len != b.u.arr->len) return false;
        for (i = 0; i < a.u.arr->len; i++) {
            if (!kz_values_equal(a.u.arr->items[i], b.u.arr->items[i], depth + 1)) return false;
        }
        return true;
    case KZV_MAP:
        if (a.u.map == NULL || b.u.map == NULL) return a.u.map == b.u.map;
        if (a.u.map->len != b.u.map->len) return false;
        for (i = 0; i < a.u.map->len; i++) {
            if (!kz_values_equal(a.u.map->items[i].key, b.u.map->items[i].key, depth + 1)) return false;
            if (!kz_values_equal(a.u.map->items[i].value, b.u.map->items[i].value, depth + 1)) return false;
        }
        return true;
    case KZV_POINT:
        return a.u.pt.x == b.u.pt.x && a.u.pt.y == b.u.pt.y && a.u.pt.z == b.u.pt.z;
    default: return false;
    }
}

static bool kz_str_append(kz_vm_t *vm, kz_buf *b, kz_value v)
{
    return kz_buf_put_render(vm, b, v, 0);
}

static bool kz_add(kz_vm_t *vm, kz_value a, kz_value b, kz_value *out, int line)
{
    if (a.kind == KZV_NUM && b.kind == KZV_NUM) {
        double n = a.u.n + b.u.n;
        if (!isfinite(n)) return kz_raise(vm, "E_KAZE_OVERFLOW", line, "a number became non-finite") || false;
        *out = kz_num_v(n);
        return true;
    }
    if (a.kind == KZV_STR || b.kind == KZV_STR) {
        kz_buf buf;
        if (!kz_buf_init(vm, &buf, a.kind == KZV_STR ? a.u.s.len + 64 : 64)) return false;
        if (a.kind == KZV_STR) {
            if (!kz_buf_putn(&buf, a.u.s.p, a.u.s.len)) return false;
        } else if (!kz_str_append(vm, &buf, a)) return false;
        if (b.kind == KZV_STR) {
            if (!kz_buf_putn(&buf, b.u.s.p, b.u.s.len)) return false;
        } else if (!kz_str_append(vm, &buf, b)) return false;
        *out = kz_str_v((kz_str){ buf.data, buf.len });
        return true;
    }
    if (a.kind == KZV_POINT && b.kind == KZV_POINT) {
        double x = a.u.pt.x + b.u.pt.x, y = a.u.pt.y + b.u.pt.y, z = a.u.pt.z + b.u.pt.z;
        if (!isfinite(x) || !isfinite(y) || !isfinite(z))
            return kz_raise(vm, "E_KAZE_OVERFLOW", line, "a point component became non-finite") || false;
        *out = kz_point_v(x, y, z);
        return true;
    }
    if (a.kind == KZV_ARR && b.kind == KZV_ARR) {
        size_t n1 = a.u.arr ? a.u.arr->len : 0;
        size_t n2 = b.u.arr ? b.u.arr->len : 0;
        kz_arr *arr;
        if (!kz_heap_charge(vm, (n1 + n2) * 16u + 24u, line)) return false;
        arr = (kz_arr *)kz_vm_alloc(vm, sizeof *arr);
        if (arr == NULL) return false;
        memset(arr, 0, sizeof *arr);
        if (n1 + n2 > 0) {
            arr->items = (kz_value *)kz_vm_alloc(vm, (n1 + n2) * sizeof *arr->items);
            if (arr->items == NULL) return false;
            arr->cap = n1 + n2;
            if (n1) memcpy(arr->items, a.u.arr->items, n1 * sizeof *arr->items);
            if (n2) memcpy(arr->items + n1, b.u.arr->items, n2 * sizeof *arr->items);
            arr->len = n1 + n2;
        }
        out->kind = KZV_ARR;
        out->u.arr = arr;
        return true;
    }
    kz_vm_set_error(vm, "E_KAZE_TYPE", line, "cannot add %s to %s", kz_type_name(a), kz_type_name(b));
    vm->state = KZ_STATE_ERR;
    return false;
}

static bool kz_arith(kz_vm_t *vm, int op, kz_value a, kz_value b, kz_value *out, int line)
{
    double x, y, n;
    if (a.kind == KZV_POINT || b.kind == KZV_POINT) {
        kz_value pt;
        if (op == 8 && a.kind == KZV_POINT && b.kind == KZV_POINT) {
            pt = kz_point_v(a.u.pt.x - b.u.pt.x, a.u.pt.y - b.u.pt.y, a.u.pt.z - b.u.pt.z);
        } else if (op == 9 && a.kind == KZV_POINT && b.kind == KZV_NUM) {
            pt = kz_point_v(a.u.pt.x * b.u.n, a.u.pt.y * b.u.n, a.u.pt.z * b.u.n);
        } else if (op == 9 && a.kind == KZV_NUM && b.kind == KZV_POINT) {
            pt = kz_point_v(a.u.n * b.u.pt.x, a.u.n * b.u.pt.y, a.u.n * b.u.pt.z);
        } else if (op == 10 && a.kind == KZV_POINT && b.kind == KZV_NUM) {
            if (b.u.n == 0.0) return kz_raise(vm, "E_KAZE_DIV0", line, "division by zero") || false;
            pt = kz_point_v(a.u.pt.x / b.u.n, a.u.pt.y / b.u.n, a.u.pt.z / b.u.n);
        } else {
            kz_vm_set_error(vm, "E_KAZE_TYPE", line, "cannot apply arithmetic to %s and %s",
                            kz_type_name(a), kz_type_name(b));
            vm->state = KZ_STATE_ERR;
            return false;
        }
        if (!isfinite(pt.u.pt.x) || !isfinite(pt.u.pt.y) || !isfinite(pt.u.pt.z))
            return kz_raise(vm, "E_KAZE_OVERFLOW", line, "a point component became non-finite") || false;
        *out = pt;
        return true;
    }
    if (a.kind != KZV_NUM || b.kind != KZV_NUM) {
        kz_vm_set_error(vm, "E_KAZE_TYPE", line, "cannot do arithmetic on %s and %s",
                        kz_type_name(a), kz_type_name(b));
        vm->state = KZ_STATE_ERR;
        return false;
    }
    x = a.u.n;
    y = b.u.n;
    if (op == 8) n = x - y;
    else if (op == 9) n = x * y;
    else if (op == 10) {
        if (y == 0.0) return kz_raise(vm, "E_KAZE_DIV0", line, "division by zero") || false;
        n = x / y;
    } else {
        if (y == 0.0) return kz_raise(vm, "E_KAZE_DIV0", line, "remainder by zero") || false;
        n = fmod(x, y);
    }
    if (!isfinite(n)) return kz_raise(vm, "E_KAZE_OVERFLOW", line, "a number became non-finite") || false;
    *out = kz_num_v(n);
    return true;
}

static bool kz_compare(kz_vm_t *vm, int op, kz_value a, kz_value b, bool *out, int line)
{
    int cmp;
    if (a.kind == KZV_NUM && b.kind == KZV_NUM) {
        cmp = a.u.n < b.u.n ? -1 : (a.u.n > b.u.n ? 1 : 0);
    } else if (a.kind == KZV_STR && b.kind == KZV_STR) {
        size_t n = a.u.s.len < b.u.s.len ? a.u.s.len : b.u.s.len;
        int c = n ? memcmp(a.u.s.p, b.u.s.p, n) : 0;
        cmp = c < 0 ? -1 : (c > 0 ? 1 : (a.u.s.len < b.u.s.len ? -1 : (a.u.s.len > b.u.s.len ? 1 : 0)));
    } else {
        kz_vm_set_error(vm, "E_KAZE_TYPE", line, "cannot compare %s with %s",
                        kz_type_name(a), kz_type_name(b));
        vm->state = KZ_STATE_ERR;
        return false;
    }
    if (op == 13) *out = cmp < 0;
    else if (op == 14) *out = cmp <= 0;
    else if (op == 15) *out = cmp > 0;
    else *out = cmp >= 0;
    return true;
}

static bool kz_field(kz_vm_t *vm, kz_value base, kz_value field, kz_value *out, int line)
{
    size_t i;
    if (field.kind != KZV_STR) {
        kz_vm_set_error(vm, "E_KAZE_TYPE", line, "a field name must be text, not %s", kz_type_name(field));
        vm->state = KZ_STATE_ERR;
        return false;
    }
    if (base.kind != KZV_MAP) {
        kz_vm_set_error(vm, "E_KAZE_TYPE", line, "'.%.*s' reads a field, but the value is %s",
                        (int)field.u.s.len, field.u.s.p, kz_type_name(base));
        vm->state = KZ_STATE_ERR;
        return false;
    }
    if (base.u.map != NULL) {
        for (i = 0; i < base.u.map->len; i++) {
            if (base.u.map->items[i].key.kind == KZV_STR &&
                kz_values_equal(base.u.map->items[i].key, field, 0)) {
                *out = base.u.map->items[i].value;
                return true;
            }
        }
    }
    *out = kz_unit();
    return true;
}

static bool kz_length(kz_vm_t *vm, kz_value v, double *out, int line)
{
    if (v.kind == KZV_STR) {
        size_t i, n = 0;
        for (i = 0; i < v.u.s.len; i++) {
            if (((unsigned char)v.u.s.p[i] & 0xC0u) != 0x80u) n++;
        }
        *out = (double)n;
        return true;
    }
    if (v.kind == KZV_ARR) { *out = (double)(v.u.arr ? v.u.arr->len : 0); return true; }
    if (v.kind == KZV_MAP) { *out = (double)(v.u.map ? v.u.map->len : 0); return true; }
    kz_vm_set_error(vm, "E_KAZE_TYPE", line, "'length of' does not apply to %s", kz_type_name(v));
    vm->state = KZ_STATE_ERR;
    return false;
}

static bool kz_index(kz_vm_t *vm, kz_value coll, kz_value idx, kz_value *out, int line)
{
    double d;
    size_t i;
    if (idx.kind != KZV_NUM) {
        kz_vm_set_error(vm, "E_KAZE_TYPE", line, "a position must be a number, not %s", kz_type_name(idx));
        vm->state = KZ_STATE_ERR;
        return false;
    }
    d = idx.u.n;
    if (d < 0.0 || d != (double)(size_t)d) {
        kz_vm_set_error(vm, "E_KAZE_BOUNDS", line, "%g is not a valid position", d);
        vm->state = KZ_STATE_ERR;
        return false;
    }
    i = (size_t)d;
    if (coll.kind == KZV_ARR) {
        if (coll.u.arr == NULL || i >= coll.u.arr->len) {
            kz_vm_set_error(vm, "E_KAZE_BOUNDS", line, "position %llu is past the end of the list",
                            (unsigned long long)i);
            vm->state = KZ_STATE_ERR;
            return false;
        }
        *out = coll.u.arr->items[i];
        return true;
    }
    if (coll.kind == KZV_STR) {
        size_t pos = 0, chars = 0;
        while (pos < coll.u.s.len) {
            size_t start = pos;
            unsigned char c = (unsigned char)coll.u.s.p[pos++];
            if (c >= 0x80) {
                while (pos < coll.u.s.len && (((unsigned char)coll.u.s.p[pos] & 0xC0u) == 0x80u)) pos++;
            }
            if (chars == i) {
                *out = kz_str_v((kz_str){ coll.u.s.p + start, pos - start });
                return true;
            }
            chars++;
        }
        kz_vm_set_error(vm, "E_KAZE_BOUNDS", line, "position %llu is past the end of the text",
                        (unsigned long long)i);
        vm->state = KZ_STATE_ERR;
        return false;
    }
    kz_vm_set_error(vm, "E_KAZE_TYPE", line, "[ ] does not apply to %s", kz_type_name(coll));
    vm->state = KZ_STATE_ERR;
    return false;
}

/* ── geometry ────────────────────────────────────────────────────────── */

#define KZ_PI 3.14159265358979323846
#define KZ_EARTH_RADIUS_M 6371008.8

static double kz_rad(double deg) { return deg * KZ_PI / 180.0; }
static double kz_deg(double rad) { return rad * 180.0 / KZ_PI; }

static double kz_normalise_deg(double deg)
{
    double d = fmod(deg, 360.0);
    if (d < 0.0) d += 360.0;
    return d;
}

static double kz_haversine(double a[3], double b[3])
{
    double lat1 = kz_rad(a[0]), lon1 = kz_rad(a[1]);
    double lat2 = kz_rad(b[0]), lon2 = kz_rad(b[1]);
    double dlat = lat2 - lat1, dlon = lon2 - lon1;
    double h = sin(dlat / 2.0) * sin(dlat / 2.0) +
               cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0);
    double clamped = h < 0.0 ? 0.0 : (h > 1.0 ? 1.0 : h);
    return 2.0 * KZ_EARTH_RADIUS_M * asin(sqrt(clamped));
}

static bool kz_point_of(kz_value v, double out[3])
{
    if (v.kind != KZV_POINT) return false;
    out[0] = v.u.pt.x;
    out[1] = v.u.pt.y;
    out[2] = v.u.pt.z;
    return true;
}

static bool kz_place_of(kz_value v, double out[3], int line, kz_vm_t *vm)
{
    if (!kz_point_of(v, out)) {
        kz_vm_set_error(vm, "E_KAZE_TYPE", line, "this needs a point, but it is %s", kz_type_name(v));
        vm->state = KZ_STATE_ERR;
        return false;
    }
    if (out[0] < -90.0 || out[0] > 90.0 || out[1] < -180.0 || out[1] > 180.0) {
        kz_vm_set_error(vm, "E_KAZE_RANGE", line,
                        "(%g, %g, %g) is not a latitude and longitude; for a local offset use 'length of' or 'bearing of'",
                        out[0], out[1], out[2]);
        vm->state = KZ_STATE_ERR;
        return false;
    }
    return true;
}

/* ── host capability dispatch ────────────────────────────────────────── */

static bool kz_parse_host_error(kz_vm_t *vm, const char *message, const char **code, const char **rest)
{
    static const char *known[] = {
        "E_CAP_MISSING", "E_CAP_DENIED", "E_KAZE_MEMORY", "E_KAZE_RANGE",
        "E_KAZE_LINK", "E_KAZE_LINK_TIMEOUT", "E_COMMAND_REJECTED",
        "E_ARMING_DENIED", "E_SAFETY_VIOLATION", "E_KAZE_HOST",
        "E_CAP_UNDECLARED", "E_CAP_INVALID", "E_CAP_APPROVAL_REQUIRED",
        "E_KAZE_TYPE", "E_KAZE_PARAM"
    };
    size_t i;
    const char *colon = strchr(message, ':');
    (void)vm;
    for (i = 0; i < sizeof known / sizeof known[0]; i++) {
        size_t n = strlen(known[i]);
        if (strncmp(message, known[i], n) == 0 && colon != NULL && (size_t)(colon - message) == n) {
            *code = known[i];
            *rest = colon + 1;
            while (**rest == ' ') (*rest)++;
            return true;
        }
    }
    *code = "E_KAZE_HOST";
    *rest = message;
    return true;
}

static bool kz_call_cap(kz_vm_t *vm, size_t cap_idx, size_t argc, int line,
                        kz_slice_status_t *status)
{
    kz_capref *cap;
    kz_value args[KZ_VM_MAX_ITEM_ARGS];
    kz_buf json;
    size_t i;
    size_t out_len = 0;

    if (cap_idx >= vm->ncaps) {
        return kz_raise(vm, "E_KAZE_BADCODE", line, "missing capability") || false;
    }
    cap = &vm->caps[cap_idx];
    if (argc > KZ_VM_MAX_ITEM_ARGS || argc != cap->argc) {
        return kz_raise(vm, "E_KAZE_BADCODE", line, "CALL_CAP argument count does not match the capability table") || false;
    }
    for (i = argc; i > 0; i--) {
        kz_value v;
        if (!kz_pop(vm, &v, line)) return false;
        args[i - 1] = v;
    }

    if (!kz_buf_init(vm, &json, 64)) return false;
    if (!kz_buf_putc(&json, '[')) return false;
    for (i = 0; i < argc; i++) {
        if (i > 0 && !kz_buf_putc(&json, ',')) return false;
        if (!kz_buf_put_json(vm, &json, args[i], 0)) return false;
    }
    if (!kz_buf_putc(&json, ']')) return false;

    /* The host's R7 gate runs for every call, including the in-process
       families and the VM-owned verdicts, exactly once. */
    if (vm->has_host && vm->host.authorise != NULL) {
        char family[256], verb[256];
        char auth_err[4096];
        size_t fl = cap->family.len < sizeof family - 1 ? cap->family.len : sizeof family - 1;
        size_t vl = cap->op.len < sizeof verb - 1 ? cap->op.len : sizeof verb - 1;
        memcpy(family, cap->family.p, fl); family[fl] = 0;
        memcpy(verb, cap->op.p, vl); verb[vl] = 0;
        auth_err[0] = 0;
        if (!vm->host.authorise(vm->host.ctx, family, verb, json.data, auth_err, sizeof auth_err)) {
            const char *code, *rest;
            const char *message = auth_err[0] ? auth_err : "E_CAP_DENIED: the capability gate refused the call";
            kz_parse_host_error(vm, message, &code, &rest);
            kz_raise(vm, code, line, "%s", rest);
            vm->state = KZ_STATE_ERR;
            *status = KZ_SLICE_ERR;
            return false;
        }
    }

    /* Verdicts are VM control flow, not host effects. */
    if (cap->family.len == 6 && memcmp(cap->family.p, "policy", 6) == 0 &&
        (cap->op.len == 5 && memcmp(cap->op.p, "allow", 5) == 0)) {
        kz_value v = argc > 0 ? args[0] : kz_unit();
        kz_buf rendered;
        if (!kz_buf_init(vm, &rendered, 64) || !kz_buf_put_render(vm, &rendered, v, 0)) return false;
        vm->verdict_kind = "allow";
        vm->verdict_text = rendered.data;
        return kz_push(vm, v, line);
    }
    if (cap->family.len == 6 && memcmp(cap->family.p, "policy", 6) == 0 &&
        cap->op.len == 6 && memcmp(cap->op.p, "reject", 6) == 0) {
        kz_buf rendered;
        if (!kz_buf_init(vm, &rendered, 64)) return false;
        if (argc > 0) {
            if (args[0].kind == KZV_STR && args[0].u.s.p != NULL) {
                if (!kz_buf_putn(&rendered, args[0].u.s.p, args[0].u.s.len)) return false;
            } else if (!kz_buf_put_render(vm, &rendered, args[0], 0)) {
                return false;
            }
        }
        vm->verdict_kind = "reject";
        vm->verdict_text = rendered.data;
        return kz_push(vm, kz_unit(), line);
    }

    if (!vm->has_host || vm->host.cap_dispatch == NULL) {
        const char *msg = "embedded Kaze has no available backend for this capability";
        kz_vm_set_error(vm, "E_CAP_MISSING", line, "%.*s.%.*s: %s (no capability host is installed)",
                        (int)cap->family.len, cap->family.p,
                        (int)cap->op.len, cap->op.p, msg);
        vm->state = KZ_STATE_ERR;
        *status = KZ_SLICE_ERR;
        return false;
    }

    vm->cap_out[0] = 0;
    vm->cap_err[0] = 0;
    {
        char family[256], verb[256];
        size_t fl = cap->family.len < sizeof family - 1 ? cap->family.len : sizeof family - 1;
        size_t vl = cap->op.len < sizeof verb - 1 ? cap->op.len : sizeof verb - 1;
        memcpy(family, cap->family.p, fl); family[fl] = 0;
        memcpy(verb, cap->op.p, vl); verb[vl] = 0;
        {
            kz_cap_status_t st = vm->host.cap_dispatch(
                vm->host.ctx, family, verb, json.data,
                vm->cap_out, KZ_CAP_JSON_MAX, &out_len,
                vm->cap_err, 4096);
            if (st == KZ_CAP_ASYNC) {
                vm->pending = true;
                vm->pending_family = cap->family;
                vm->pending_verb = cap->op;
                vm->pending_args = (kz_str){ json.data, json.len };
                vm->pending_line = (uint32_t)line;
                vm->state = KZ_STATE_PAUSED;
                *status = KZ_SLICE_CAP_REQ;
                return true;
            }
            if (st == KZ_CAP_FAILED) {
                const char *code, *rest;
                const char *message = vm->cap_err[0] ? vm->cap_err : "capability call failed";
                kz_parse_host_error(vm, message, &code, &rest);
                if (!kz_raise(vm, code, line, rest)) {
                    *status = KZ_SLICE_ERR;
                    return false;
                }
                return true;
            }
        }
    }
    if (out_len == 0) out_len = strlen(vm->cap_out);
    if (out_len > KZ_CAP_JSON_MAX) {
        *status = kz_raise(vm, "E_KAZE_MEMORY", line, "capability result exceeded the 1 MiB limit")
            ? KZ_SLICE_YIELD : KZ_SLICE_ERR;
        return *status != KZ_SLICE_ERR;
    }
    {
        kz_value result;
        if (!kz_json_parse_value(vm, vm->cap_out, &result)) {
            *status = kz_raise(vm, "E_KAZE_HOST", line, "capability returned invalid JSON")
                ? KZ_SLICE_YIELD : KZ_SLICE_ERR;
            return *status != KZ_SLICE_ERR;
        }
        return kz_push(vm, result, line);
    }
}

/* ── frames ──────────────────────────────────────────────────────────── */

static bool kz_frame_enter(kz_vm_t *vm, const kz_item *item, size_t *out_frame)
{
    kz_frame *f;
    size_t nlocals;
    if (vm->fp >= KZ_VM_MAX_FRAMES) {
        return kz_raise(vm, "E_KAZE_STACK", (int)item->line, "call stack overflow") || false;
    }
    nlocals = item->slots;
    if (item->nparams > nlocals) nlocals = item->nparams;
    if (nlocals == 0) nlocals = 1;
    if (nlocals > KZ_VM_MAX_LOCALS) {
        return kz_raise(vm, "E_KAZE_STACK", (int)item->line, "a routine declares more than 256 local slots") || false;
    }
    if (vm->local_top + nlocals > vm->local_cap) {
        return kz_raise(vm, "E_KAZE_STACK", (int)item->line, "call stack overflow") || false;
    }
    f = &vm->frames[vm->fp];
    f->item = item;
    f->nlocals = nlocals;
    f->local_base = vm->local_top;
    f->locals = vm->local_pool + vm->local_top;
    vm->local_top += nlocals;
    for (size_t i = 0; i < nlocals; i++) f->locals[i] = kz_unit();
    f->pc = 0;
    f->current_line = 0;
    f->stack_base = vm->sp;
    f->handler_base = vm->nhandlers;
    vm->fp++;
    *out_frame = vm->fp - 1;
    return true;
}

static bool kz_start_item(kz_vm_t *vm, size_t item_index, const char *args_json)
{
    kz_item *item;
    kz_value args[KZ_VM_MAX_ITEM_ARGS];
    size_t nargs = 0;
    size_t i;

    if (item_index >= vm->nitems) {
        kz_vm_set_error(vm, "E_KAZE_ENTRY", 0, "there is no such item in this script");
        vm->state = KZ_STATE_ERR;
        return false;
    }
    item = &vm->items[item_index];
    if (args_json != NULL && args_json[0] != 0) {
        kz_json j;
        j.vm = vm;
        j.p = args_json;
        j.len = strlen(args_json);
        j.at = 0;
        j.nodes = 0;
        kz_json_ws(&j);
        if (j.at < j.len && j.p[j.at] == '[') {
            j.at++;
            kz_json_ws(&j);
            if (j.at < j.len && j.p[j.at] == ']') j.at++;
            else {
                for (;;) {
                    kz_value v = kz_json_value(&j, 0);
                    if (nargs < KZ_VM_MAX_ITEM_ARGS) args[nargs++] = v;
                    kz_json_ws(&j);
                    if (j.at < j.len && j.p[j.at] == ',') { j.at++; continue; }
                    if (j.at < j.len && j.p[j.at] == ']') { j.at++; break; }
                    break;
                }
            }
        } else {
            kz_value v = kz_json_value(&j, 0);
            args[nargs++] = v;
        }
    }
    if (nargs != item->nparams) {
        kz_vm_set_error(vm, "E_KAZE_PARAM", (int)item->line,
                        "routine '%.*s' takes %llu argument(s), but the host supplied %llu",
                        (int)item->name.len, item->name.p,
                        (unsigned long long)item->nparams, (unsigned long long)nargs);
        vm->state = KZ_STATE_ERR;
        return false;
    }

    vm->sp = 0;
    vm->fp = 0;
    vm->local_top = 0;
    vm->nhandlers = 0;
    vm->pending = false;
    vm->last_value = kz_unit();
    vm->err_code[0] = 0;
    vm->err_msg[0] = 0;
    vm->err_line = 0;
    vm->state = KZ_STATE_RUNNING;

    {
        size_t frame = 0;
        if (!kz_frame_enter(vm, item, &frame)) return false;
        for (i = 0; i < nargs; i++) vm->frames[frame].locals[i] = args[i];
    }
    return true;
}

/* ── the interpreter ─────────────────────────────────────────────────── */

static int kz_current_line(const kz_frame *f)
{
    if (f->item->nline_map > 0 && f->pc < f->item->nline_map && f->item->line_map[f->pc] > 0)
        return (int)f->item->line_map[f->pc];
    if (f->current_line > 0)
        return (int)f->current_line;
    return (int)f->item->line;
}

static bool kz_jump(kz_vm_t *vm, kz_frame *f, int32_t imm, int line)
{
    int64_t target = (int64_t)f->pc + imm;
    if (target < 0 || (uint64_t)target > f->item->ncode) {
        return kz_raise(vm, "E_KAZE_BADCODE", line, "a jump left the code") || false;
    }
    f->pc = (size_t)target;
    return true;
}

static bool kz_do_return(kz_vm_t *vm, kz_value value)
{
    kz_frame *f = &vm->frames[vm->fp - 1];
    size_t base = f->stack_base;
    vm->nhandlers = f->handler_base;
    vm->local_top = f->local_base;
    vm->fp--;
    vm->sp = base;
    if (vm->fp == 0) {
        vm->last_value = value;
        vm->state = KZ_STATE_DONE;
        return true;
    }
    return kz_push(vm, value, (int)f->item->line);
}

static bool kz_step(kz_vm_t *vm, kz_slice_status_t *status)
{
    uint32_t w, imm;
    int op, line;
    kz_frame *f;
    kz_value a, b, out;

    *status = KZ_SLICE_YIELD;
    if (vm->state == KZ_STATE_DONE) { *status = KZ_SLICE_DONE; return true; }
    if (vm->state == KZ_STATE_ERR) { *status = KZ_SLICE_ERR; return false; }
    if (vm->state == KZ_STATE_PAUSED) { *status = KZ_SLICE_CAP_REQ; return true; }
    if (vm->state != KZ_STATE_RUNNING) { *status = KZ_SLICE_ERR; return false; }

    f = &vm->frames[vm->fp - 1];
    if (f->pc >= f->item->ncode) {
        vm->last_value = kz_unit();
        vm->state = KZ_STATE_DONE;
        *status = KZ_SLICE_DONE;
        return true;
    }
    w = f->item->code[f->pc];
    op = (int)(w & 0xffu);
    line = kz_current_line(f);

    if (vm->fuel_limit != 0 && vm->fuel_used >= vm->fuel_limit) {
        *status = kz_raise(vm, "E_TIMEOUT", line,
                           "this run used all of its fuel and was stopped (fuel, not wall time); "
                           "a loop that never ends or too much work in one tick is the usual cause")
            ? KZ_SLICE_YIELD : KZ_SLICE_ERR;
        return *status != KZ_SLICE_ERR;
    }
    vm->fuel_used++;
    vm->instructions++;
    f->pc++;

    switch (op) {
    case 0: /* Const */
        imm = f->item->code[f->pc++];
        return kz_push(vm, kz_const_value(vm, imm), line);
    case 1: { /* LoadLocal */
        size_t slot = (w >> 8) & 0xffu;
        if (slot >= f->nlocals) return kz_raise(vm, "E_KAZE_BADCODE", line, "LOAD from a missing slot") || false;
        return kz_push(vm, f->locals[slot], line);
    }
    case 2: { /* StoreLocal */
        size_t slot = (w >> 8) & 0xffu;
        if (!kz_pop(vm, &a, line)) return false;
        if (slot >= f->nlocals) return kz_raise(vm, "E_KAZE_BADCODE", line, "STORE to a missing slot") || false;
        f->locals[slot] = a;
        return true;
    }
    case 3: { /* NewArr */
        size_t n = (w >> 8) & 0xffu;
        kz_arr *arr;
        if (!kz_heap_charge(vm, n * 16u + 24u, line)) { *status = KZ_SLICE_ERR; return false; }
        arr = (kz_arr *)kz_vm_alloc(vm, sizeof *arr);
        if (arr == NULL) { vm->state = KZ_STATE_ERR; *status = KZ_SLICE_ERR; return false; }
        memset(arr, 0, sizeof *arr);
        if (n > 0) {
            if (vm->sp < n) return kz_raise(vm, "E_KAZE_BADCODE", line, "the value stack underflowed") || false;
            arr->items = (kz_value *)kz_vm_alloc(vm, n * sizeof *arr->items);
            if (arr->items == NULL) { vm->state = KZ_STATE_ERR; *status = KZ_SLICE_ERR; return false; }
            arr->cap = n;
            arr->len = n;
            for (size_t i = n; i > 0; i--) arr->items[i - 1] = vm->stack[--vm->sp];
        }
        out = kz_unit(); out.kind = KZV_ARR; out.u.arr = arr;
        return kz_push(vm, out, line);
    }
    case 4: /* GetIndex */
        if (!kz_pop(vm, &b, line) || !kz_pop(vm, &a, line)) return false;
        if (!kz_index(vm, a, b, &out, line)) { *status = KZ_SLICE_ERR; return false; }
        return kz_push(vm, out, line);
    case 5: { /* Len */
        double n;
        if (!kz_pop(vm, &a, line)) return false;
        if (!kz_length(vm, a, &n, line)) { *status = KZ_SLICE_ERR; return false; }
        return kz_push(vm, kz_num_v(n), line);
    }
    case 6: /* Pop */
        return kz_pop(vm, &a, line);
    case 7: /* Add */
        if (!kz_pop(vm, &b, line) || !kz_pop(vm, &a, line)) return false;
        if (!kz_add(vm, a, b, &out, line)) { *status = KZ_SLICE_ERR; return false; }
        return kz_push(vm, out, line);
    case 8:
    case 9:
    case 10:
    case 11:
        if (!kz_pop(vm, &b, line) || !kz_pop(vm, &a, line)) return false;
        if (!kz_arith(vm, op, a, b, &out, line)) { *status = KZ_SLICE_ERR; return false; }
        return kz_push(vm, out, line);
    case 12: /* Neg */
        if (!kz_pop(vm, &a, line)) return false;
        if (a.kind != KZV_NUM) {
            return kz_raise(vm, "E_KAZE_TYPE", line, "cannot negate %s", kz_type_name(a)) || false;
        }
        if (!isfinite(-a.u.n)) return kz_raise(vm, "E_KAZE_OVERFLOW", line, "a number became non-finite") || false;
        return kz_push(vm, kz_num_v(-a.u.n), line);
    case 13:
    case 14:
    case 15:
    case 16: {
        bool cmp;
        if (!kz_pop(vm, &b, line) || !kz_pop(vm, &a, line)) return false;
        if (!kz_compare(vm, op, a, b, &cmp, line)) { *status = KZ_SLICE_ERR; return false; }
        return kz_push(vm, kz_bool_v(cmp), line);
    }
    case 17:
    case 18:
        if (!kz_pop(vm, &b, line) || !kz_pop(vm, &a, line)) return false;
        out = kz_bool_v(op == 17 ? kz_values_equal(a, b, 0) : !kz_values_equal(a, b, 0));
        return kz_push(vm, out, line);
    case 19:
        if (!kz_pop(vm, &a, line)) return false;
        return kz_push(vm, kz_bool_v(!kz_is_truthy(a)), line);
    case 20: /* Jmp */
        imm = f->item->code[f->pc++];
        return kz_jump(vm, f, (int32_t)imm, line);
    case 21:
    case 29: /* JmpIfFalse / JmpIfTrue */
        imm = f->item->code[f->pc++];
        if (!kz_pop(vm, &a, line)) return false;
        if ((op == 29) == kz_is_truthy(a)) return kz_jump(vm, f, (int32_t)imm, line);
        return true;
    case 22: { /* CallCap */
        size_t argc = (w >> 8) & 0xffu;
        imm = f->item->code[f->pc++];
        if (!kz_call_cap(vm, imm, argc, line, status)) return *status != KZ_SLICE_ERR;
        if (vm->state == KZ_STATE_PAUSED) { *status = KZ_SLICE_CAP_REQ; return true; }
        if (vm->state == KZ_STATE_ERR) { *status = KZ_SLICE_ERR; return false; }
        return true;
    }
    case 23: /* PolicyStop */
    case 24: /* Ret */
        if (!kz_pop(vm, &a, line)) return false;
        return kz_do_return(vm, a);
    case 25: /* RetUnit */
        return kz_do_return(vm, kz_unit());
    case 26: /* FuelTick */
        imm = f->item->code[f->pc++];
        if (imm > 0) f->current_line = imm;
        if (vm->has_host && vm->host.poll_deadline != NULL && vm->host.poll_deadline(vm->host.ctx) != 0) {
            char msg[256];
            snprintf(msg, sizeof msg, "the call was interrupted after its deadline elapsed (line %u)", imm);
            return kz_raise(vm, "E_TIMEOUT", (int)imm, msg) ? true : (*status = KZ_SLICE_ERR, false);
        }
        return true;
    case 27: /* Trap */
        imm = f->item->code[f->pc++];
        {
            const char *msg = "the script stopped";
            if (imm < vm->nconsts && vm->consts[imm].kind == 1) {
                /* Const text is not NUL-terminated; copy a bounded slice. */
                static char trap_buf[512];
                size_t n = vm->consts[imm].text.len;
                if (n >= sizeof trap_buf) n = sizeof trap_buf - 1;
                memcpy(trap_buf, vm->consts[imm].text.p, n);
                trap_buf[n] = 0;
                msg = trap_buf;
            }
            return kz_raise(vm, "E_KAZE_TRAP", line, msg);
        }
    case 28: /* Halt */
        return kz_do_return(vm, kz_unit());
    case 30: /* GetField */
        if (!kz_pop(vm, &b, line) || !kz_pop(vm, &a, line)) return false;
        if (!kz_field(vm, a, b, &out, line)) { *status = KZ_SLICE_ERR; return false; }
        return kz_push(vm, out, line);
    case 31:
    case 32: { /* Jeq/Jne */
        bool eq;
        imm = f->item->code[f->pc++];
        if (!kz_pop(vm, &b, line) || !kz_pop(vm, &a, line)) return false;
        eq = kz_values_equal(a, b, 0);
        if ((op == 31) == eq) return kz_jump(vm, f, (int32_t)imm, line);
        return true;
    }
    case 33: { /* VecNew */
        size_t n = (w >> 8) & 0xffu;
        if (n != 2 && n != 3) return kz_raise(vm, "E_KAZE_BADCODE", line, "VECNEW with a bad width") || false;
        if (!kz_heap_charge(vm, 24u, line)) { *status = KZ_SLICE_ERR; return false; }
        out = kz_point_v(0.0, 0.0, 0.0);
        if (vm->sp < n) return kz_raise(vm, "E_KAZE_BADCODE", line, "the value stack underflowed") || false;
        for (size_t i = n; i > 0; i--) {
            kz_value v = vm->stack[--vm->sp];
            double d;
            if (!kz_num_of(v, &d)) return kz_raise(vm, "E_KAZE_TYPE", line, "a point component must be a number") || false;
            if (i == 3) out.u.pt.x = d;
            else if (i == 2) out.u.pt.y = d;
            else out.u.pt.z = d;
        }
        if (n == 2) out.u.pt.z = 0.0;
        if (!isfinite(out.u.pt.x) || !isfinite(out.u.pt.y) || !isfinite(out.u.pt.z))
            return kz_raise(vm, "E_KAZE_OVERFLOW", line, "a point component became non-finite") || false;
        return kz_push(vm, out, line);
    }
    case 34:
    case 36:
    case 43: {
        double p[3], q[3], n;
        if (!kz_pop(vm, &b, line) || !kz_pop(vm, &a, line)) return false;
        if (!kz_place_of(a, p, line, vm) || !kz_place_of(b, q, line, vm)) { *status = KZ_SLICE_ERR; return false; }
        if (op == 34) n = kz_haversine(p, q);
        else if (op == 43) {
            double g = kz_haversine(p, q), dz = q[2] - p[2];
            n = sqrt(g * g + dz * dz);
        } else {
            double lat1 = kz_rad(p[0]), lat2 = kz_rad(q[0]);
            double dlon = kz_rad(q[1] - p[1]);
            double y = sin(dlon) * cos(lat2);
            double x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon);
            n = kz_normalise_deg(kz_deg(atan2(y, x)));
        }
        if (!isfinite(n)) return kz_raise(vm, "E_KAZE_OVERFLOW", line, "a number became non-finite") || false;
        return kz_push(vm, kz_num_v(n), line);
    }
    case 35: {
        double p[3], n;
        if (!kz_pop(vm, &a, line)) return false;
        if (!kz_point_of(a, p)) return kz_raise(vm, "E_KAZE_TYPE", line, "this needs a point, but it is %s", kz_type_name(a)) || false;
        n = kz_normalise_deg(kz_deg(atan2(p[0], p[1])));
        return kz_push(vm, kz_num_v(n), line);
    }
    case 37:
    case 38:
    case 40:
    case 41:
    case 42: {
        double x, n;
        if (!kz_pop(vm, &a, line)) return false;
        if (!kz_num_of(a, &x)) return kz_raise(vm, "E_KAZE_TYPE", line, "this needs a number, but it is %s", kz_type_name(a)) || false;
        if (op == 37) n = sin(x);
        else if (op == 38) n = cos(x);
        else if (op == 40) {
            if (x < 0.0) return kz_raise(vm, "E_KAZE_MATH", line, "sqrt of %g has no real answer", x) || false;
            n = sqrt(x);
        } else if (op == 41) n = kz_rad(x);
        else n = kz_deg(x);
        if (!isfinite(n)) return kz_raise(vm, "E_KAZE_OVERFLOW", line, "a number became non-finite") || false;
        return kz_push(vm, kz_num_v(n), line);
    }
    case 39: {
        double x, y, n;
        if (!kz_pop(vm, &a, line) || !kz_pop(vm, &b, line)) return false;
        if (!kz_num_of(a, &x) || !kz_num_of(b, &y))
            return kz_raise(vm, "E_KAZE_TYPE", line, "atan2 needs two numbers") || false;
        n = atan2(y, x);
        if (!isfinite(n)) return kz_raise(vm, "E_KAZE_OVERFLOW", line, "a number became non-finite") || false;
        return kz_push(vm, kz_num_v(n), line);
    }
    case 44: { /* TryBegin */
        size_t slot = (w >> 8) & 0xffu;
        int32_t rel;
        imm = f->item->code[f->pc++];
        rel = (int32_t)imm;
        if (vm->nhandlers >= KZ_VM_MAX_HANDLERS)
            return kz_raise(vm, "E_KAZE_STACK", line, "more than 64 nested 'try' blocks") || false;
        {
            size_t target = f->pc + (size_t)rel;
            if ((int64_t)f->pc + rel < 0 || target > f->item->ncode)
                return kz_raise(vm, "E_KAZE_BADCODE", line, "a jump left the code") || false;
            vm->handlers[vm->nhandlers].target = target;
            vm->handlers[vm->nhandlers].stack_len = vm->sp;
            vm->handlers[vm->nhandlers].slot = (uint8_t)slot;
            vm->handlers[vm->nhandlers].verdict_kind = vm->verdict_kind;
            vm->handlers[vm->nhandlers].verdict_text = vm->verdict_text;
            vm->handlers[vm->nhandlers].frame_depth = vm->fp;
            vm->nhandlers++;
        }
        return true;
    }
    case 45: /* TryEnd */
        if (vm->nhandlers == 0) return kz_raise(vm, "E_KAZE_BADCODE", line, "TRY_END ran without an active try block") || false;
        vm->nhandlers--;
        return true;
    case 46: { /* LinkLost */
        uint64_t after_ms, silent = 0;
        bool have = false;
        imm = f->item->code[f->pc++];
        after_ms = imm;
        if (after_ms == 0) return kz_raise(vm, "E_KAZE_BADCODE", line, "LINK_LOST with a zero window would always be true") || false;
        if (vm->has_host && vm->host.link_silent_ms != NULL) {
            if (vm->host.link_silent_ms(vm->host.ctx, &silent)) have = true;
        } else if (vm->has_host && vm->host.cap_dispatch != NULL) {
            size_t out_len = 0;
            char *out = vm->cap_out;
            out[0] = 0;
            vm->cap_err[0] = 0;
            if (vm->host.cap_dispatch(vm->host.ctx, "drone", "link_silent_ms", "[]",
                                      out, KZ_CAP_JSON_MAX, &out_len, vm->cap_err, 4096) == KZ_CAP_OK) {
                char *end = NULL;
                if (out_len == 0) out_len = strlen(out);
                if (out_len > 0) out[out_len] = 0;
                silent = (uint64_t)strtod(out, &end);
                have = end != out;
            }
        }
        return kz_push(vm, kz_bool_v(have && silent >= after_ms), line);
    }
    case 47: { /* CallRoutine */
        size_t argc = (w >> 8) & 0xffu;
        size_t target, frame;
        kz_value args[KZ_VM_MAX_ITEM_ARGS];
        imm = f->item->code[f->pc++];
        target = imm;
        if (target >= vm->nitems) return kz_raise(vm, "E_KAZE_BADCODE", line, "routine item out of range") || false;
        if (argc > KZ_VM_MAX_ITEM_ARGS || vm->sp < argc)
            return kz_raise(vm, "E_KAZE_BADCODE", line, "the value stack underflowed") || false;
        for (size_t i = argc; i > 0; i--) args[i - 1] = vm->stack[--vm->sp];
        if (!kz_frame_enter(vm, &vm->items[target], &frame)) { *status = KZ_SLICE_ERR; return false; }
        for (size_t i = 0; i < argc && i < vm->frames[frame].nlocals; i++)
            vm->frames[frame].locals[i] = args[i];
        return true;
    }
    case 48: { /* NewMap */
        size_t n = (w >> 8) & 0xffu;
        kz_map *m;
        if (!kz_heap_charge(vm, n * 32u + 24u, line)) { *status = KZ_SLICE_ERR; return false; }
        m = (kz_map *)kz_vm_alloc(vm, sizeof *m);
        if (m == NULL) { vm->state = KZ_STATE_ERR; *status = KZ_SLICE_ERR; return false; }
        memset(m, 0, sizeof *m);
        if (n > 0) {
            if (vm->sp < n * 2) return kz_raise(vm, "E_KAZE_BADCODE", line, "the value stack underflowed") || false;
            m->items = (kz_pair *)kz_vm_alloc(vm, n * sizeof *m->items);
            if (m->items == NULL) { vm->state = KZ_STATE_ERR; *status = KZ_SLICE_ERR; return false; }
            m->cap = n;
            m->len = n;
            for (size_t i = n; i > 0; i--) {
                m->items[i - 1].value = vm->stack[--vm->sp];
                m->items[i - 1].key = vm->stack[--vm->sp];
            }
        }
        out = kz_unit(); out.kind = KZV_MAP; out.u.map = m;
        return kz_push(vm, out, line);
    }
    default:
        return kz_raise(vm, "E_KAZE_BADCODE", line, "this module has an unknown instruction") || false;
    }
}

kz_slice_status_t kz_vm_step_slice(kz_vm_t *vm, size_t max_steps,
                                   const char **req_family, const char **req_verb,
                                   const char **req_args_json)
{
    size_t steps = 0;
    if (vm == NULL) return KZ_SLICE_ERR;
    if (req_family) *req_family = NULL;
    if (req_verb) *req_verb = NULL;
    if (req_args_json) *req_args_json = NULL;
    if (vm->state == KZ_STATE_PAUSED) {
        if (req_family) *req_family = vm->pending_family.p;
        if (req_verb) *req_verb = vm->pending_verb.p;
        if (req_args_json) *req_args_json = vm->pending_args.p;
        return KZ_SLICE_CAP_REQ;
    }
    while (steps < max_steps) {
        kz_slice_status_t status = KZ_SLICE_DONE;
        if (!kz_step(vm, &status)) return KZ_SLICE_ERR;
        if (status == KZ_SLICE_CAP_REQ) {
            if (req_family) *req_family = vm->pending_family.p;
            if (req_verb) *req_verb = vm->pending_verb.p;
            if (req_args_json) *req_args_json = vm->pending_args.p;
            return KZ_SLICE_CAP_REQ;
        }
        if (status == KZ_SLICE_DONE) return KZ_SLICE_DONE;
        if (vm->state == KZ_STATE_DONE) return KZ_SLICE_DONE;
        if (vm->state == KZ_STATE_ERR) return KZ_SLICE_ERR;
        steps++;
    }
    return KZ_SLICE_YIELD;
}

bool kz_vm_supply_cap_result(kz_vm_t *vm, const char *result_json, const char *err_msg)
{
    kz_value value;
    if (vm == NULL || !vm->pending || vm->state != KZ_STATE_PAUSED || vm->fp == 0) return false;
    vm->pending = false;
    vm->state = KZ_STATE_RUNNING;
    if (err_msg != NULL && err_msg[0] != 0) {
        const char *code, *rest;
        kz_parse_host_error(vm, err_msg, &code, &rest);
        return kz_raise(vm, code, (int)vm->pending_line, rest);
    }
    if (result_json == NULL) result_json = "null";
    if (!kz_json_parse_value(vm, result_json, &value)) {
        return kz_raise(vm, "E_KAZE_HOST", (int)vm->pending_line, "capability returned invalid JSON");
    }
    return kz_push(vm, value, (int)vm->pending_line);
}

int kz_vm_start(kz_vm_t *vm, size_t item_index, const char *args_json)
{
    return kz_start_item(vm, item_index, args_json) ? 1 : 0;
}

kz_embed_result_t kz_vm_finish(const kz_vm_t *vm)
{
    kz_embed_result_t out;
    kz_buf b;
    memset(&out, 0, sizeof out);
    if (vm == NULL) {
        out.ok = false;
        out.text = "";
        out.err_code = "E_KAZE_STATE";
        return out;
    }
    if (vm->state == KZ_STATE_ERR) {
        out.ok = false;
        out.text = vm->err_msg;
        out.len = strlen(vm->err_msg);
        out.err_code = vm->err_code[0] ? vm->err_code : "E_KAZE_TRAP";
        out.err_line = vm->err_line;
        return out;
    }
    {
        kz_vm_t *mutable_vm = (kz_vm_t *)vm;
        if (!kz_buf_init(mutable_vm, &b, 64)) {
            out.ok = false;
            out.text = "E_KAZE_MEMORY";
            out.err_code = "E_KAZE_MEMORY";
            return out;
        }
        if (!kz_buf_put_render(mutable_vm, &b, vm->state == KZ_STATE_DONE ? vm->last_value : kz_unit(), 0)) {
            out.ok = false;
            out.text = "E_KAZE_MEMORY";
            out.err_code = "E_KAZE_MEMORY";
            return out;
        }
        out.ok = true;
        out.text = b.data;
        out.len = b.len;
        out.err_code = "";
        out.err_line = 0;
        return out;
    }
}

const char *kz_vm_error_code(const kz_vm_t *vm) { return vm ? vm->err_code : "E_KAZE_STATE"; }
const char *kz_vm_error_message(const kz_vm_t *vm) { return vm ? vm->err_msg : ""; }
int kz_vm_error_line(const kz_vm_t *vm) { return vm ? vm->err_line : 0; }
uint64_t kz_vm_instructions(const kz_vm_t *vm) { return vm ? vm->instructions : 0; }
uint64_t kz_vm_fuel_used(const kz_vm_t *vm) { return vm ? vm->fuel_used : 0; }
const char *kz_vm_verdict_kind(const kz_vm_t *vm) { return vm ? vm->verdict_kind : NULL; }
const char *kz_vm_verdict_text(const kz_vm_t *vm) { return vm ? (vm->verdict_text ? vm->verdict_text : "") : ""; }
uint32_t kz_vm_pending_line(const kz_vm_t *vm) { return vm ? vm->pending_line : 0; }
const char *kz_vm_pending_family(const kz_vm_t *vm) { return vm && vm->pending ? (vm->pending_family.p ? vm->pending_family.p : "") : ""; }
const char *kz_vm_pending_verb(const kz_vm_t *vm) { return vm && vm->pending ? (vm->pending_verb.p ? vm->pending_verb.p : "") : ""; }
const char *kz_vm_pending_args_json(const kz_vm_t *vm) { return vm && vm->pending ? (vm->pending_args.p ? vm->pending_args.p : "") : ""; }
const char *kz_vm_value_json(const kz_vm_t *vm)
{
    static char fallback[] = "";
    if (vm == NULL || vm->state != KZ_STATE_DONE) return fallback;
    {
        kz_vm_t *mutable_vm = (kz_vm_t *)vm;
        kz_buf b;
        if (!kz_buf_init(mutable_vm, &b, 64)) return fallback;
        if (!kz_buf_put_json(mutable_vm, &b, vm->last_value, 0)) return fallback;
        return b.data;
    }
}

const char *kz_vm_value_text(const kz_vm_t *vm)
{
    static char fallback[] = "";
    if (vm == NULL || vm->state != KZ_STATE_DONE) return fallback;
    {
        kz_vm_t *mutable_vm = (kz_vm_t *)vm;
        kz_buf b;
        if (!kz_buf_init(mutable_vm, &b, 64)) return fallback;
        if (!kz_buf_put_render(mutable_vm, &b, vm->last_value, 0)) return fallback;
        return b.data;
    }
}
