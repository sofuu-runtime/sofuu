/* embedded-kaze-vm-probe.c — the C99 harness used by
 * scripts/embedded-kaze-c-gates.sh.
 *
 * It is deliberately tiny: load a .kazebc, step the resumable VM, and print
 * either `OK <value> <instructions> <fuel>` or `ERR <code> <line>`. The same
 * binary is built with AddressSanitizer for the hostile-input gate, so the
 * probe must free every VM it creates and never call exit() from the engine.
 *
 *   probe run <bytecode> <entry> [args-json] [fuel]
 *   probe heap <bytecode> <entry> <heap-bytes> [args-json]
 *   probe pause <bytecode> <entry> [args-json]
 *   probe corrupt <bytecode> [truncate-to]
 *   probe compiler <source-file>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kaze_embed.h"

static int g_async;
static int g_paused;

static kz_cap_status_t serve(void *ctx, const char *family, const char *verb,
                             const char *args_json, char *out_json, size_t out_cap,
                             size_t *out_len, char *out_err, size_t err_cap)
{
    (void)ctx;
    (void)args_json;
    (void)err_cap;
    out_err[0] = 0;
    if (g_async) {
        g_async = 0;
        return KZ_CAP_ASYNC;
    }
    if (strcmp(family, "log") == 0) {
        snprintf(out_json, out_cap, "null");
    } else if (strcmp(family, "policy") == 0) {
        snprintf(out_json, out_cap, "true");
    } else if (strcmp(family, "memory") == 0) {
        snprintf(out_json, out_cap, "null");
    } else if (strcmp(family, "rover") == 0 || strcmp(family, "sensor") == 0) {
        snprintf(out_json, out_cap, "0.3");
    } else {
        snprintf(out_json, out_cap, "null");
    }
    *out_len = strlen(out_json);
    (void)verb;
    return KZ_CAP_OK;
}

static unsigned char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    unsigned char *buf;
    long n;
    if (f == NULL) {
        perror(path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        perror(path);
        return NULL;
    }
    buf = (unsigned char *)malloc((size_t)n + 1u);
    if (buf == NULL) {
        fclose(f);
        return NULL;
    }
    if (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        perror(path);
        return NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static int find_item(kz_vm_t *vm, const char *name, size_t *out)
{
    size_t i;
    for (i = 0; i < kz_vm_item_count(vm); i++) {
        if (strcmp(kz_vm_item_name(vm, i), name) == 0) {
            *out = i;
            return 1;
        }
    }
    return 0;
}

static int run(const char *path, const char *entry, const char *args, uint64_t fuel,
               size_t heap, int heap_set)
{
    size_t len = 0;
    unsigned char *bc = read_file(path, &len);
    kz_vm_t *vm;
    kz_host_cbs_t cbs;
    size_t item = 0;
    const char *family = NULL, *verb = NULL, *req_args = NULL;
    int i;

    if (bc == NULL) {
        return 2;
    }
    vm = kz_vm_create(bc, len, 4u << 20, fuel);
    free(bc);
    if (vm == NULL) {
        printf("CREATE_ERR %s\n", kz_vm_last_error());
        return 3;
    }
    memset(&cbs, 0, sizeof cbs);
    cbs.cap_dispatch = serve;
    kz_vm_set_host(vm, &cbs);
    if (heap_set) {
        kz_vm_set_heap_limit(vm, heap);
    }
    if (!find_item(vm, entry, &item)) {
        printf("NO_ITEM %s\n", entry);
        kz_vm_destroy(vm);
        return 4;
    }
    if (!kz_vm_start(vm, item, args)) {
        printf("ERR %s %d\n", kz_vm_error_code(vm), kz_vm_error_line(vm));
        kz_vm_destroy(vm);
        return 0;
    }
    for (i = 0; i < 100000; i++) {
        kz_slice_status_t st = kz_vm_step_slice(vm, 256, &family, &verb, &req_args);
        if (st == KZ_SLICE_YIELD) {
            continue;
        }
        if (st == KZ_SLICE_CAP_REQ) {
            char out[KZ_CAP_JSON_MAX + 1];
            char err[512];
            size_t out_len = 0;
            if (!g_paused) {
                printf("PAUSED %s %s\n", family ? family : "", verb ? verb : "");
                g_paused = 1;
            } else {
                kz_cap_status_t cap = serve(NULL, family, verb, req_args, out, sizeof out,
                                            &out_len, err, sizeof err);
                if (cap == KZ_CAP_ASYNC) {
                    /* The host will answer with kz_vm_supply_cap_result. */
                } else if (cap == KZ_CAP_OK) {
                    kz_vm_supply_cap_result(vm, out, NULL);
                } else {
                    kz_vm_supply_cap_result(vm, NULL, err);
                }
                continue;
            }
            if (!kz_vm_supply_cap_result(vm, "null", NULL)) {
                printf("SUPPLY_FAILED\n");
                kz_vm_destroy(vm);
                return 0;
            }
            continue;
        }
        if (st == KZ_SLICE_ERR) {
            printf("ERR %s %d\n", kz_vm_error_code(vm), kz_vm_error_line(vm));
            kz_vm_destroy(vm);
            return 0;
        }
        if (st == KZ_SLICE_DONE) {
            kz_embed_result_t r = kz_vm_finish(vm);
            printf("OK %s %llu %llu\n", r.text ? r.text : "nothing",
                   (unsigned long long)kz_vm_instructions(vm),
                   (unsigned long long)kz_vm_fuel_used(vm));
            kz_vm_destroy(vm);
            return 0;
        }
        printf("BAD_STATUS %d\n", (int)st);
        kz_vm_destroy(vm);
        return 5;
    }
    printf("STEP_LIMIT\n");
    kz_vm_destroy(vm);
    return 6;
}

static int corrupt(const char *path, size_t truncate_to)
{
    size_t len = 0;
    size_t i;
    unsigned char *bc = read_file(path, &len);
    if (bc == NULL) {
        return 2;
    }
    if (truncate_to < len) {
        len = truncate_to;
    }
    for (i = 0; i <= 16 && i <= len; i++) {
        kz_vm_t *vm = kz_vm_create(bc, len - i, 1u << 20, 10000);
        if (vm != NULL) {
            kz_vm_destroy(vm);
        }
    }
    /* A header with an impossible version and a count that runs off the end. */
    if (len >= 10) {
        unsigned char saved[10];
        memcpy(saved, bc, sizeof saved);
        bc[8] = 0xff;
        bc[9] = 0xff;
        {
            kz_vm_t *vm = kz_vm_create(bc, len, 1u << 20, 10000);
            if (vm != NULL) {
                kz_vm_destroy(vm);
            }
        }
        memcpy(bc, saved, sizeof saved);
    }
    free(bc);
    printf("CORRUPT_OK\n");
    return 0;
}

#ifdef PROBE_WITH_COMPILER
static int compiler(const char *path)
{
    size_t len = 0;
    char *src = (char *)read_file(path, &len);
    kz_compiler_t *c;
    kz_embed_result_t r;
    if (src == NULL) {
        return 2;
    }
    c = kz_embed_compiler_new(0);
    if (c == NULL) {
        printf("COMPILER_CREATE_ERR\n");
        free(src);
        return 3;
    }
    r = kz_embed_check(c, src, len);
    printf("CHECK %d %s %d\n", r.ok ? 1 : 0, r.err_code ? r.err_code : "-", r.err_line);
    r = kz_embed_compile(c, src, len);
    printf("COMPILE %d %zu %s %d\n", r.ok ? 1 : 0, r.len, r.err_code ? r.err_code : "-", r.err_line);
    r = kz_embed_emit_c(c, src, len, 1);
    printf("EMIT_C %d %zu %s %d\n", r.ok ? 1 : 0, r.len, r.err_code ? r.err_code : "-", r.err_line);
    kz_embed_compiler_free(c);
    free(src);
    return 0;
}
#else
static int compiler(const char *path)
{
    (void)path;
    printf("PROBE_NO_COMPILER\n");
    return 64;
}
#endif

int main(int argc, char **argv)
{
    uint64_t fuel = 500000;
    if (argc < 3) {
        fprintf(stderr, "usage: %s run|heap|corrupt|compiler <path> ...\n", argv[0]);
        return 64;
    }
    if (strcmp(argv[1], "run") == 0) {
        if (argc >= 6) {
            fuel = strtoull(argv[5], NULL, 10);
        }
        return run(argv[2], argv[3], argc >= 5 ? argv[4] : NULL, fuel, 0, 0);
    }
    if (strcmp(argv[1], "heap") == 0) {
        size_t heap = argc >= 4 ? (size_t)strtoull(argv[4], NULL, 10) : 0;
        if (argc >= 6) {
            fuel = strtoull(argv[5], NULL, 10);
        }
        return run(argv[2], argv[3], argc >= 6 ? argv[5] : NULL, fuel, heap, 1);
    }
    if (strcmp(argv[1], "corrupt") == 0) {
        return corrupt(argv[2], argc >= 4 ? (size_t)strtoull(argv[3], NULL, 10) : 0);
    }
    if (strcmp(argv[1], "compiler") == 0) {
        return compiler(argv[2]);
    }
    if (strcmp(argv[1], "pause") == 0) {
        g_async = 1;
        g_paused = 0;
        return run(argv[2], argv[3], argc >= 5 ? argv[4] : NULL, 500000, 0, 0);
    }
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 64;
}
