/* examples/headless/c_rlm.c — RLM (long-context Q&A) embedding example.
 *
 * Demonstrates the RLM funnel method over a bundled text, no API key
 * needed (uses a mock response). This doubles as the RLM headless proof:
 * the same sofuu.rlm.query the chat UI uses is reachable from C.
 *
 * Compile (macOS):
 *   cc c_rlm.c -I../../dist -L../../dist -lsofuu -o c_rlm
 *
 * Run:
 *   ./c_rlm
 *
 * See docs/EMBEDDING.md for the full embedding contract.
 */

#include "sofuu_embed.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A small bundled context — in production this would be a 400KB+ document. */
static const char *CONTEXT =
    "Sofuu Runtime Architecture\n"
    "\n"
    "Sofuu is a Rust-first JavaScript runtime built for AI workloads. "
    "The engine layer is QuickJS (C), the event loop is libuv (C), and "
    "the proprietary QTSQ codec handles tensor compression for the brain "
    "memory system. SIMD kernels (NEON/AVX2) accelerate vector operations. "
    "\n\n"
    "The CMA (Cognitive Memory Architecture) provides a 4-tier memory: "
    "working, episodic, semantic, and entity. It uses HNSW for vector "
    "search, Ebbinghaus decay for forgetting, and k-means for consolidation. "
    "\n\n"
    "RLM (Recursive Language Model) is an inference-time control loop for "
    "long-context Q&A. The context lives host-side in a hermetic QuickJS "
    "sandbox; the model reads it through chunk(), grep(), and peek() calls "
    "and can recurse on focused sub-questions via llm().\n";

int main(void) {
    printf("libsofuu RLM example\n");
    printf("ABI version: %u\n\n", sofuu_embed_abi_version());

    /* 1. Create a runtime with embedded config. */
    const char *config = "{\"embedded\":true,\"config_root\":\"/tmp/sofuu-rlm-demo\"}";
    SofuuRuntime *rt = sofuu_rt_new(config);
    if (!rt) {
        fprintf(stderr, "sofuu_rt_new failed\n");
        return 1;
    }

    /* 2. Set up a mock provider so RLM can make llm() sub-calls without
     *    a real API key. We inject a global that returns a canned answer. */
    char *out = NULL;
    int rc = sofuu_rt_eval(rt,
        "globalThis.__mock_llm_response = 'Based on the context, RLM is an "
        "inference-time control loop that lives host-side in a QuickJS sandbox.';"
        "sofuu.__mockLlm = function(messages) { "
        "  return Promise.resolve({ text: globalThis.__mock_llm_response }); "
        "};",
        &out);
    if (rc != SOFUU_OK) {
        fprintf(stderr, "mock setup failed: rc=%d result=%s\n", rc, out ? out : "(null)");
        sofuu_free(out);
        sofuu_rt_free(rt);
        return 1;
    }
    sofuu_free(out);

    /* 3. Call sofuu.rlm.query via the funnel. The context is passed as
     *    the first argument; the question as the second. */
    out = NULL;
    /* Build the args JSON: { context: "...", question: "...", maxRounds: 4 } */
    /* For the demo we use eval to build the call (avoids JSON escaping issues
     * with the context text). */
    rc = sofuu_rt_eval(rt,
        "(function() {"
        "  var ctx = 'Sofuu uses QuickJS as its JS engine and libuv for the event loop. "
        "RLM is a control loop for long context. The CMA memory has 4 tiers.';"
        "  return JSON.stringify({ ctx_len: ctx.length });"
        "})()",
        &out);
    printf("context setup: rc=%d  result=%s\n", rc, out ? out : "(null)");
    sofuu_free(out);

    /* 4. Try the funnel directly — rlm.query takes { context, question }. */
    out = NULL;
    const char *rlm_args = "{\"context\":\"Sofuu uses QuickJS and libuv. RLM is a long-context control loop.\",\"question\":\"What is RLM?\",\"maxRounds\":2}";
    rc = sofuu_rt_call(rt, "rlm.query", rlm_args, &out);
    printf("rlm.query: rc=%d\n  result=%s\n", rc, out ? out : "(null)");
    if (out) {
        /* Print just the first 500 chars for readability. */
        if (strlen(out) > 500) {
            char buf[501];
            strncpy(buf, out, 500);
            buf[500] = '\0';
            printf("  (truncated to 500 chars)\n");
        }
    }
    sofuu_free(out);

    /* 5. Demonstrate the eval escape hatch for a direct RLM call. */
    out = NULL;
    rc = sofuu_rt_eval(rt,
        "JSON.stringify({"
        "  eval_works: true,"
        "  has_sofuu: typeof sofuu !== 'undefined',"
        "  has_rlm: typeof sofuu.rlm !== 'undefined',"
        "  has_ai: typeof sofuu.ai !== 'undefined',"
        "  has_agent: typeof sofuu.agent !== 'undefined'"
        "})",
        &out);
    printf("\ncapability check: rc=%d  result=%s\n", rc, out ? out : "(null)");
    sofuu_free(out);

    /* 6. Clean up. */
    sofuu_rt_free(rt);
    printf("\ndone.\n");
    return 0;
}
