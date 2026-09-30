/* examples/headless/c_embed_vec.c — H-E1 vector embedding ABI test.
 *
 * Exercises sofuu_embed_local / sofuu_embed_batch / sofuu_embed_info
 * directly (no JSON, no eval): default + named spaces, unknown-space
 * refusal, NULL-arg guards, batch row-major layout, manifest contents.
 *
 * The embedders are stateless — rt may be NULL. Both paths are probed.
 *
 * Compile (macOS):
 *   cc c_embed_vec.c -I../../dist -L../../dist -lsofuu -o c_embed_vec
 *
 * Run:
 *   ./c_embed_vec
 */

#include "sofuu_embed.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); \
        failures++; \
    } \
} while (0)

int main(void) {
    printf("libsofuu ABI version: %u\n", sofuu_embed_abi_version());

    /* 1. Default space (NULL) with NULL rt — stateless path. */
    float *v = NULL;
    size_t dim = 0;
    int rc = sofuu_embed_local(NULL, "hello world", NULL, &v, &dim);
    CHECK(rc == 0, "default embed rc=%d, want 0", rc);
    /* E0-5: the default space is hash-768 — the same space the brain
     * stores in, so embed→remember works without a space argument. */
    CHECK(dim == 768, "default dim=%zu, want 768", dim);
    if (rc == 0 && dim == 768) {
        double sumsq = 0;
        for (size_t i = 0; i < dim; i++) sumsq += (double)v[i] * v[i];
        CHECK(fabs(sumsq - 1.0) < 1e-6, "default vector not unit (sumsq=%f)", sumsq);
        sofuu_free(v);
    }

    /* 2. Same call through a real runtime (reserved param ignored). */
    SofuuRuntime *rt = sofuu_rt_new(NULL);
    CHECK(rt != NULL, "sofuu_rt_new failed");
    v = NULL; dim = 0;
    rc = sofuu_embed_local(rt, "hello world", "sem1-64", &v, &dim);
    CHECK(rc == 0 && dim == 64, "sem1 rc=%d dim=%zu", rc, dim);
    if (v) sofuu_free(v);

    /* 3. hash-768 named space. */
    v = NULL; dim = 0;
    rc = sofuu_embed_local(NULL, "hello world", "hash-768", &v, &dim);
    CHECK(rc == 0 && dim == 768, "hash rc=%d dim=%zu", rc, dim);
    if (v) sofuu_free(v);

    /* 4. Unknown space refuses (no buffer). */
    v = (float *)0x1; dim = 999;
    rc = sofuu_embed_local(NULL, "x", "nope", &v, &dim);
    CHECK(rc == SOFUU_ERR_UNKNOWN_SPACE, "unknown space rc=%d", rc);

    /* 5. NULL-arg guards. */
    float *nv = NULL; size_t nd = 0;
    CHECK(sofuu_embed_local(NULL, NULL, NULL, &nv, &nd) == SOFUU_ERR_INVALID_ARG,
          "NULL text must be invalid-arg");
    CHECK(sofuu_embed_batch(NULL, NULL, 0, NULL, &nv, &nd, &nd) == SOFUU_ERR_INVALID_ARG,
          "empty batch must be invalid-arg");
    CHECK(sofuu_embed_info(NULL, NULL) == SOFUU_ERR_INVALID_ARG,
          "NULL info out must be invalid-arg");

    /* 6. Batch of 3 → 3 rows × 768 dims, row-major (default space). */
    const char *texts[3] = { "alpha", "beta", "gamma" };
    float *b = NULL; size_t bn = 0, bdim = 0;
    rc = sofuu_embed_batch(NULL, texts, 3, NULL, &b, &bn, &bdim);
    CHECK(rc == 0 && bn == 3 && bdim == 768, "batch rc=%d n=%zu dim=%zu", rc, bn, bdim);
    if (rc == 0) {
        /* Rows differ (distinct inputs) and each is unit length. */
        CHECK(memcmp(b, b + 768, 768 * sizeof(float)) != 0, "batch rows unexpectedly identical");
        double s0 = 0;
        for (size_t i = 0; i < 768; i++) s0 += (double)b[i] * b[i];
        CHECK(fabs(s0 - 1.0) < 1e-6, "batch row 0 not unit (%f)", s0);
        sofuu_free(b);
    }

    /* 7. Manifest names the default space and all three spaces. */
    char *info = NULL;
    rc = sofuu_embed_info(NULL, &info);
    CHECK(rc == 0 && info != NULL, "info rc=%d", rc);
    if (info) {
        CHECK(strstr(info, "sem2-64") != NULL, "manifest missing sem2-64");
        CHECK(strstr(info, "sem1-64") != NULL, "manifest missing sem1-64");
        CHECK(strstr(info, "hash-768") != NULL, "manifest missing hash-768");
        sofuu_free(info);
    }

    /* 8. Image path guards (positive path covered by the JS e2e with
     * real PNG fixtures — C has no fixture encoder here). */
    float *iv = NULL; size_t idim = 0;
    CHECK(sofuu_embed_image(NULL, NULL, 0, &iv, &idim) == SOFUU_ERR_INVALID_ARG,
          "image NULL must be invalid-arg");
    static const unsigned char garbage[16] = "not an image....";
    CHECK(sofuu_embed_image(NULL, garbage, sizeof garbage, &iv, &idim) == SOFUU_ERR_MODEL_UNAVAIL,
          "image garbage must report unavailable");
    CHECK(iv == NULL, "failed image call must not hand out a buffer");

    if (rt) sofuu_rt_free(rt);

    if (failures == 0) {
        printf("c_embed_vec: all checks passed.\n");
        return 0;
    }
    printf("c_embed_vec: %d FAILURES\n", failures);
    return 1;
}
