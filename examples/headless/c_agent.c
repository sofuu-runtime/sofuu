/* examples/headless/c_agent.c — agent loop embedding example (A7).
 *
 * Demonstrates the full agent loop — define, run (with streaming events),
 * cancel — reachable from C with zero TUI. This is the A7 headless surface
 * proof: the same sofuu.agent.run the chat UI uses is reachable from C.
 *
 * Compile (macOS):
 *   cc c_agent.c -I../../dist -L../../dist -lsofuu -o c_agent
 *
 * Run:
 *   ./c_agent
 *
 * See docs/EMBEDDING.md for the full embedding contract.
 */

#include "sofuu_embed.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Called for each streaming event from agent.run. */
static void on_event(const char *event_json, void *opaque) {
    (void)opaque;
    /* Parse the kind from the JSON for pretty-printing. */
    const char *kind = strstr(event_json, "\"kind\":\"");
    if (kind) {
        kind += 8; /* skip "kind":" */
        const char *end = strchr(kind, '"');
        if (end) {
            printf("  [event] %.*s\n", (int)(end - kind), kind);
            return;
        }
    }
    printf("  [event] %s\n", event_json);
}

int main(void) {
    printf("libsofuu Agent example (A7)\n");
    printf("ABI version: %u\n\n", sofuu_embed_abi_version());

    /* 1. Create a runtime with per-process agent definitions (A7.3).
     *    No filesystem needed — agents are supplied in config. */
    const char *config =
        "{\"embedded\":true,"
        "\"agents\":[{"
          "\"name\":\"echo-agent\","
          "\"system\":\"You are a helpful test agent.\","
          "\"provider\":\"mock\","
          "\"model\":\"mock-c-agent\","
          "\"budget\":{\"maxSteps\":3}"
        "}]}";
    SofuuRuntime *rt = sofuu_rt_new(config);
    if (!rt) {
        fprintf(stderr, "sofuu_rt_new failed\n");
        return 1;
    }

    /* 2. Verify the agent was registered via the funnel. */
    char *out = NULL;
    int rc = sofuu_rt_call(rt, "agent.list", NULL, &out);
    printf("agent.list: rc=%d\n  %s\n", rc, out ? out : "(null)");
    if (out) sofuu_free(out);

    /* 3. Set up a mock provider so agent.run works without a real API key. */
    out = NULL;
    rc = sofuu_rt_eval(rt,
        "var origComplete = sofuu.ai.complete;"
        "sofuu.ai.complete = function(opts) {"
        "  if (opts.model === 'mock-c-agent') {"
        "    return Promise.resolve({ text: 'Hello from the agent! The answer is 42.' });"
        "  }"
        "  return origComplete.call(sofuu.ai, opts);"
        "}; 'mock ready'",
        &out);
    printf("mock setup: rc=%d  result=%s\n", rc, out ? out : "(null)");
    if (out) sofuu_free(out);

    /* 4. Run the agent with streaming events (A7.1). */
    printf("\n--- agent.run (streaming) ---\n");
    uint64_t cancel_id = 0;
    const char *run_args = "{\"target\":\"echo-agent\",\"task\":\"What is the answer?\"}";
    rc = sofuu_rt_call_stream(rt, "agent.run", run_args, on_event, NULL, &cancel_id);
    printf("agent.run: rc=%d  cancel_id=%llu\n", rc, (unsigned long long)cancel_id);

    /* 5. Read the result via eval (the streaming call ran the agent;
     *    we can inspect the result via a second call). */
    out = NULL;
    rc = sofuu_rt_eval(rt,
        "JSON.stringify({"
        "  agent_available: typeof sofuu.agent !== 'undefined',"
        "  has_run: typeof sofuu.agent.run === 'function',"
        "  has_define: typeof sofuu.agent.define === 'function',"
        "  has_cancel: typeof sofuu.agent.cancel === 'function',"
        "  has_list: typeof sofuu.agent.list === 'function',"
        "  has_renderTrace: typeof sofuu.agent.renderTrace === 'function'"
        "})",
        &out);
    printf("\ncapability check: rc=%d\n  %s\n", rc, out ? out : "(null)");
    if (out) sofuu_free(out);

    /* 6. Test cancellation (A7.2) — cancel an unknown ID (returns -1). */
    rc = sofuu_rt_cancel(rt, 99999);
    printf("\ncancel(unknown): rc=%d (expected -1)\n", rc);

    /* 7. Clean up. */
    sofuu_rt_free(rt);
    printf("\ndone.\n");
    return 0;
}
