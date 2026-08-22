/* examples/headless/c_embed.c — minimal libsofuu embedding example.
 *
 * Compile (macOS):
 *   cc c_embed.c -I../../dist -L../../dist -lsofuu -o c_embed
 *
 * Static linking (requires QTSQ libs present; without QTSQ use dylib):
 *   cc c_embed.c -I../../dist ../../dist/libsofuu.a -L/path/to/qtsq -lqtsq -o c_embed
 *
 * Run:
 *   ./c_embed
 *
 * No API key needed — demonstrates eval + funnel with built-in JS only.
 * See docs/EMBEDDING.md for the full embedding contract.
 */

#include "sofuu_embed.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("libsofuu ABI version: %u\n", sofuu_embed_abi_version());

    /* 1. Create a runtime (NULL config → all defaults). */
    SofuuRuntime *rt = sofuu_rt_new(NULL);
    if (!rt) {
        fprintf(stderr, "sofuu_rt_new failed\n");
        return 1;
    }

    /* 2. Evaluate arbitrary JS — the eval escape hatch. */
    char *out = NULL;
    int rc = sofuu_rt_eval(rt, "1 + 1", &out);
    printf("eval 1+1: rc=%d  result=%s\n", rc, out ? out : "(null)");
    sofuu_free(out);

    /* 3. Evaluate a more complex expression. */
    out = NULL;
    rc = sofuu_rt_eval(rt,
        "JSON.stringify({ hello: 'world', ts: Date.now() })",
        &out);
    printf("eval JSON: rc=%d  result=%s\n", rc, out ? out : "(null)");
    sofuu_free(out);

    /* 4. Call a sync method via the funnel (no API key needed). */
    out = NULL;
    rc = sofuu_rt_call(rt, "version", NULL, &out);
    printf("sofuu.version: rc=%d  result=%s\n", rc, out ? out : "(null)");
    sofuu_free(out);

    /* 5. Clean up. */
    sofuu_rt_free(rt);
    printf("done.\n");
    return 0;
}
