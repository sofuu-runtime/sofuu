/* examples/headless/c_embed.c — the five-minute embedding quickstart.
 *
 * This is the FIRST thing a developer runs, so it asserts instead of
 * printing-and-hoping: every call is checked and a non-zero exit means
 * something is actually broken. (It used to call a funnel method "version"
 * that does not exist and print the resulting ok:false envelope as if it
 * were success — the exact "docs promise more than ships" failure this
 * sample exists to prevent.)
 *
 * Compile + run (macOS/Linux, no API key required):
 *   make headless-test          # builds libsofuu, compiles + runs this
 *   cc c_embed.c -I../../dist -L../../dist -lsofuu -o c_embed && ./c_embed
 *
 * Static linking (needs the QTSQ libs on the link line):
 *   cc c_embed.c -I../../dist ../../dist/libsofuu.a -L/path/to/qtsq -lqtsq
 *
 * See docs/EMBEDDING.md for the full contract. To call a real LLM see
 * c_llm.c (needs an API key or a local endpoint).
 */

#include "sofuu_embed.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;

/* The funnel answers with a JSON envelope: {"ok":true,"result":…} on
 * success, {"ok":false,"error":{…}} on failure. Every funnel call below
 * checks for the success marker, so a silent error can never pass CI. */
static int envelope_ok(const char *json) {
    return json != NULL && strstr(json, "\"ok\":true") != NULL;
}

static void check(int cond, const char *what) {
    if (cond) {
        printf("  ok   %s\n", what);
    } else {
        printf("  FAIL %s\n", what);
        failures++;
    }
}

/* Build {"vec":[1,0,…,0], …} — a 768-dim unit vector at axis 0. */
static char *unit_vec_args(void) {
    enum { DIM = 768 };
    char *buf = malloc(16 * 1024);
    if (!buf) return NULL;
    int n = snprintf(buf, 16 * 1024, "{\"vec\":[1");
    for (int i = 1; i < DIM; i++) n += snprintf(buf + n, 16 * 1024 - n, ",0");
    snprintf(buf + n, 16 * 1024 - n,
             "],\"text\":\"the sky is blue\",\"role\":\"user\",\"kv_page_id\":0}");
    return buf;
}

int main(void) {
    printf("libsofuu ABI version: %u\n", sofuu_embed_abi_version());

    /* 1. Create a runtime (NULL config → all defaults). */
    SofuuRuntime *rt = sofuu_rt_new(NULL);
    if (!rt) {
        fprintf(stderr, "sofuu_rt_new failed\n");
        return 1;
    }

    char *out = NULL;
    int rc;

    /* 2. The eval escape hatch. */
    rc = sofuu_rt_eval(rt, "1 + 1", &out);
    check(rc == 0 && envelope_ok(out), "eval: 1 + 1");
    sofuu_free(out);

    /* 3. Embed text locally — zero network, zero API key, zero model file.
     *    This is the headline capability, so it is the first real proof. */
    float *vec = NULL;
    size_t dim = 0;
    rc = sofuu_embed_local(rt, "hello from C", NULL, &vec, &dim);
    check(rc == 0 && vec != NULL && dim > 0, "embed_local: a non-empty vector");
    if (vec) {
        printf("       dim=%zu\n", dim);
        sofuu_free(vec);
    }

    /* 4. The same embedder through the JSON funnel. */
    out = NULL;
    rc = sofuu_rt_call(rt, "ai.embedInfo", NULL, &out);
    check(rc == 0 && envelope_ok(out), "funnel: ai.embedInfo resolves ok:true");
    sofuu_free(out);

    /* 5. The private brain, end to end over the funnel — the differentiator:
     *    encrypted local memory the app owns, no cloud in the loop. */
    char tmpl[] = "/tmp/sofuu-c_embed-XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd >= 0) close(fd);

    char args[512];
    snprintf(args, sizeof(args),
             "{\"path\":\"%s\",\"dim\":768,\"embed_id\":\"hash-v1\"}", tmpl);
    out = NULL;
    rc = sofuu_rt_call(rt, "memory.open", args, &out);
    check(rc == 0 && envelope_ok(out) && strstr(out, "\"handle\"") != NULL,
          "funnel: memory.open returns a usable handle");
    sofuu_free(out);

    char *remember_args = unit_vec_args();
    out = NULL;
    rc = sofuu_rt_call(rt, "memory.remember", remember_args, &out);
    check(rc == 0 && envelope_ok(out), "funnel: memory.remember stores a vector");
    sofuu_free(out);
    free(remember_args);

    /* Recall it back and confirm the right text comes out. */
    char recall_args[16 * 1024];
    {
        int n = snprintf(recall_args, sizeof(recall_args), "{\"vec\":[1");
        for (int i = 1; i < 768; i++) n += snprintf(recall_args + n, sizeof(recall_args) - n, ",0");
        snprintf(recall_args + n, sizeof(recall_args) - n, "],\"k\":1}");
    }
    out = NULL;
    rc = sofuu_rt_call(rt, "memory.recall", recall_args, &out);
    check(rc == 0 && envelope_ok(out) && strstr(out, "the sky is blue") != NULL,
          "funnel: memory.recall returns the stored memory");
    sofuu_free(out);

    out = NULL;
    rc = sofuu_rt_call(rt, "memory.count", NULL, &out);
    check(rc == 0 && envelope_ok(out), "funnel: memory.count responds");
    sofuu_free(out);

    out = NULL;
    rc = sofuu_rt_call(rt, "memory.flush", NULL, &out);
    check(rc == 0 && envelope_ok(out), "funnel: memory.flush persists");
    sofuu_free(out);

    /* 6. Clean up. */
    sofuu_rt_free(rt);
    remove(tmpl);

    if (failures) {
        printf("\n%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("done — all checks passed\n");
    return 0;
}
