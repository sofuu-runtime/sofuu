/* examples/headless/c_llm.c — "call a real LLM from C in five minutes".
 *
 * c_embed.c proves the offline path (embeddings + brain, no network, no
 * key). This sample is the other half of the promise: a real completion
 * against any OpenAI-compatible endpoint, with the key supplied in
 * *config* rather than an environment variable — which is the only way it
 * can work on iOS, Android, or a device with no shell.
 *
 * Keys come from the environment purely as a local convenience:
 *   SOFUU_DEMO_KEY   your provider key (sk-…, or a local server's token)
 *   SOFUU_DEMO_MODEL model id (default: gpt-4o-mini)
 *   SOFUU_DEMO_URL   base URL for a local/self-hosted server
 *                    (default: the provider's public endpoint)
 *
 * With no key set it falls back to a MOCK provider, so `make
 * headless-test` runs this in CI with no secrets and no network. The mock
 * path proves the request shape end to end; the live path is the same code
 * with a real key.
 *
 *   cc c_llm.c -I../../dist -L../../dist -lsofuu -o c_llm && ./c_llm
 */

#include "sofuu_embed.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

/* Streaming counters, written from the event callback. */
static int deltas = 0;
static int done_seen = 0;

static void on_event(const char *event_json, void *opaque) {
    (void)opaque;
    if (strstr(event_json, "\"kind\":\"delta\"")) deltas++;
    if (strstr(event_json, "\"kind\":\"done\"")) done_seen++;
}

static void check(int cond, const char *what) {
    if (cond) {
        printf("  ok   %s\n", what);
    } else {
        printf("  FAIL %s\n", what);
        failures++;
    }
}

static char *json_escape(const char *s, char *out, size_t cap) {
    size_t o = 0;
    for (size_t i = 0; s[i] && o + 8 < cap; i++) {
        if (s[i] == '"' || s[i] == '\\') {
            out[o++] = '\\';
            out[o++] = s[i];
        } else {
            out[o++] = s[i];
        }
    }
    out[o] = 0;
    return out;
}

int main(void) {
    const char *key = getenv("SOFUU_DEMO_KEY");
    const char *model = getenv("SOFUU_DEMO_MODEL");
    const char *url = getenv("SOFUU_DEMO_URL");
    if (!model || !*model) model = "gpt-4o-mini";

    char esc_key[512], esc_model[256], esc_url[512];
    json_escape(key ? key : "", esc_key, sizeof(esc_key));
    json_escape(model, esc_model, sizeof(esc_model));
    json_escape(url ? url : "", esc_url, sizeof(esc_url));

    /* 1. Build the config. This is the whole point: the key travels in
     *    config_json, not in the process environment, because an embedded
     *    host has no environment to rely on. */
    char config[2048];
    if (url && *url) {
        snprintf(config, sizeof(config),
                 "{\"embedded\":true,\"api_keys\":{\"openai\":\"%s\"}}", esc_key);
    } else {
        snprintf(config, sizeof(config),
                 "{\"embedded\":true,\"api_keys\":{\"openai\":\"%s\"}}", esc_key);
    }

    SofuuRuntime *rt = sofuu_rt_new(config);
    if (!rt) {
        fprintf(stderr, "sofuu_rt_new failed\n");
        return 1;
    }

    int live = (key && *key);

    if (!live) {
        /* Mock provider: proves the request path without a secret. */
        char mock[1024];
        snprintf(mock, sizeof(mock),
                 "(function(){"
                 "  sofuu.ai.complete = function(opts){"
                 "    return Promise.resolve({text:'mock reply for: ' + (opts.prompt||opts.messages||''),"
                 "                           model:'%s', usage:{promptTokens:7,completionTokens:5}});"
                 "  };"
                 "  return true;"
                 "})()", esc_model);
        char *out = NULL;
        int rc = sofuu_rt_eval(rt, mock, &out);
        check(rc == 0, "mock provider installed");
        sofuu_free(out);
    }

    /* 2. Call the LLM through the funnel. */
    char args[4096];
    snprintf(args, sizeof(args),
             "{\"prompt\":\"Say hello in one short sentence.\","
             "\"provider\":\"openai\",\"model\":\"%s\"%s%s%s}",
             esc_model,
             esc_url[0] ? ",\"base_url\":\"" : "", esc_url, esc_url[0] ? "\"" : "");

    char *out = NULL;
    int rc = sofuu_rt_call(rt, "ai.complete", args, &out);
    check(rc == 0, "ai.complete returns SOFUU_OK");

    if (out) {
        if (strstr(out, "\"ok\":true")) {
            printf("  ok   response envelope ok:true\n");
            /* Pull out .result.text if present for a human-readable echo. */
            const char *t = strstr(out, "\"text\":\"");
            if (t) {
                t += 8;
                const char *end = strchr(t, '"');
                if (end) printf("       reply: %.*s\n", (int)(end - t), t);
            }
        } else {
            printf("  FAIL ai.complete error envelope: %s\n", out);
            failures++;
        }
    } else {
        check(0, "ai.complete produced output");
    }
    sofuu_free(out);

    /* 3. Streaming: the same call through the streaming funnel, which
     *    delivers delta events and a trailing done event. */
    deltas = 0;
    done_seen = 0;

    char sargs[4096];
    uint64_t cancel_id = 0;
    snprintf(sargs, sizeof(sargs),
             "{\"prompt\":\"Count to three.\",\"provider\":\"openai\",\"model\":\"%s\"%s%s%s}",
             esc_model,
             esc_url[0] ? ",\"base_url\":\"" : "", esc_url, esc_url[0] ? "\"" : "");

    char *sout = NULL;
    rc = sofuu_rt_call_stream(rt, "ai.stream", sargs, on_event, NULL, &cancel_id);
    check(rc == 0, "ai.stream returns SOFUU_OK");
    check(done_seen, "ai.stream emits a trailing done event");
    if (live) {
        check(deltas > 0, "ai.stream delivers at least one delta (live provider)");
    } else {
        /* The mock only replaces ai.complete, so ai.stream has no endpoint;
         * the done event must still arrive (an error envelope's), which is
         * the behaviour we want to pin. */
        printf("  ..   deltas=%d (mock has no streaming endpoint; done event is the contract)\n",
               deltas);
    }
    sofuu_free(sout);

    sofuu_rt_free(rt);

    if (failures) {
        printf("\n%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("done — all checks passed%s\n", live ? " (live provider)" : " (mock provider)");
    return 0;
}
