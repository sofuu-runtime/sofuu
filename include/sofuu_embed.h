/* sofuu_embed.h — the embeddable Sofuu C ABI (v2).
 *
 * PLAN-HEADLESS H1: libsofuu lets any host (iOS, Android, desktop, server,
 * edge) embed Sofuu with all features and zero TUI. This header extends —
 * never breaks — the historical six-function `sofuu.h` surface. The full
 * embedding contract (process rule, threading, error model, memory
 * ownership, config, stability tiers) lives in docs/EMBEDDING.md (H0).
 *
 * Memory ownership: every buffer the library hands out through a
 * `char **out_json` is malloc-owned; the host frees it with sofuu_free().
 * Never free() these buffers with the host's own allocator assumptions and
 * never hand a host buffer to the library to free. Mirrors the proven
 * ffi_exports.rs contract.
 *
 * Error model: every fallible call returns an int (0 = ok, negative =
 * SOFUU_ERR_*) AND, where an out_json is provided, a JSON envelope:
 *   {"ok":true,"result":...}
 *   {"ok":false,"error":{"code":"<stable_code>","message":"<human text>"}}
 * Stable string codes (never errno-style integers) are the contract.
 *
 * Threading: one SofuuRuntime is owned by one thread; all calls on a
 * runtime must come from that thread (QuickJS + libuv are single-threaded).
 * Multiple runtimes per process are supported, each on its own thread; they
 * share the centralized libuv loop, so concurrent loop-driven work is
 * serialized on it (see docs/EMBEDDING.md).
 */
#ifndef SOFUU_EMBED_H
#define SOFUU_EMBED_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque runtime handle (created by sofuu_rt_new, freed by sofuu_rt_free). */
typedef struct SofuuRuntime SofuuRuntime;

/* Streaming event callback: invoked once per event with a NUL-terminated
 * JSON string (valid only for the duration of the call — copy if needed)
 * and the opaque pointer passed to sofuu_rt_call_stream. */
typedef void (*sofuu_event_cb)(const char *event_json, void *opaque);

/* ── Return codes ──────────────────────────────────────────────── */
#define SOFUU_OK                 0
#define SOFUU_ERR_INVALID_ARG   -1   /* NULL runtime / malformed input    */
#define SOFUU_ERR_NOT_IMPL      -2   /* method/variant not yet implemented */
#define SOFUU_ERR_UNKNOWN_METHOD -3  /* funnel method does not resolve     */
#define SOFUU_ERR_JS_EXCEPTION  -4   /* JS threw; details in out_json      */
#define SOFUU_ERR_BAD_CONFIG    -5   /* config_json failed to parse        */
#define SOFUU_ERR_INTERNAL      -6   /* engine/alloc failure               */

/* ── Versioning ────────────────────────────────────────────────── */
/* Returns the ABI version this library implements (v2 = 2). The ABI is
 * `experimental` for one release, then frozen with symbol versioning. */
uint32_t sofuu_embed_abi_version(void);

/* ── Host log callback ─────────────────────────────────────────── */
/* Console output routing: when a callback is installed, console.log/info/
 * warn/error (and friends) are delivered here instead of stdout/stderr.
 * `level` is "log"/"info"/"warn"/"error"; both strings are NUL-terminated
 * and valid only for the duration of the call — copy what you keep.
 * Process-global: applies to every runtime in this process. Passing NULL
 * clears the callback (output goes back to stdio). */
typedef void (*sofuu_log_cb)(const char *level, const char *message, void *opaque);
void sofuu_embed_set_log_cb(sofuu_log_cb cb, void *opaque);

/* ── Lifecycle ─────────────────────────────────────────────────── */
/* Create a runtime. config_json is a NUL-terminated JSON object or NULL
 * for defaults (NULL-safe). Recognized keys (all optional):
 *   { "embedded": true,            // hosted-library mode (default true)
 *     "config_root": "/path",      // replaces $HOME/.sofuu derivations
 *     "enable_signals": false,     // install signal handlers (default off)
 *     "qtsq": true,                // QTSQ brain/KV persistence (build-gated)
 *     "brain_path": "/path.qtsq",  // explicit brain file
 *     "api_keys": { "openai": "…" },
 *     "agents": [ { "name":"researcher", "system":"…", "provider":"…",
 *                   "model":"…", "tools":[…], "budget":{…} } ] }
 * In embedded mode process.exit(N) throws a catchable ExitError with an
 * `exitCode` property instead of terminating the host process, and the
 * config is visible to JS as globalThis.__sofuu_embed_config.
 * Returns NULL on failure. The returned handle is owned by ONE thread. */
SofuuRuntime *sofuu_rt_new(const char *config_json);

/* Destroy a runtime and release its engine. NULL-tolerant. */
void sofuu_rt_free(SofuuRuntime *rt);

/* ── The generic funnel ────────────────────────────────────────── */
/* Call a feature by name without a new exported symbol per feature.
 * `method` is a dot-path under the `sofuu` JS namespace (e.g.
 * "ai.complete", "ai.embed", "ai.embedLocal", "ai.similarity",
 * "memory.open", "rlm.query", "agent.run", "mcp.connect").
 *
 * `args_json` is either a JSON object (passed as a single options arg) or
 * a JSON array (spread as positional args); NULL/"" means no arguments.
 *
 * On return, *out_json (when non-NULL) is a malloc'd JSON envelope the host
 * frees with sofuu_free(). The call blocks until the feature's promise
 * settles (the event loop is drained). Returns SOFUU_OK or a SOFUU_ERR_*. */
int sofuu_rt_call(SofuuRuntime *rt, const char *method,
                  const char *args_json, char **out_json);

/* Streaming / cancellable variant (agent.run, ai.stream, rlm, …).
 *
 * Calls a method that accepts an `onStep` callback option (e.g.
 * `agent.run`) and forwards each event to the host's `on_event` callback
 * on the calling thread. The method is called with args_json merged with
 * { onStep: <internal callback>, signal: <cancel_id> }.
 *
 * For `agent.run`, args_json should be: {"target":"<name>","task":"…",…opts}.
 * Events are JSON objects: {"runId":"…","name":"…","depth":0,"t":<ms>,
 *                           "kind":"start|plan|tool|delegate|answer|stop",
 *                           "payload":{…}}
 *
 * *out_cancel_id receives a token usable with sofuu_rt_cancel.
 * Returns SOFUU_OK or a SOFUU_ERR_*. */
int sofuu_rt_call_stream(SofuuRuntime *rt, const char *method,
                         const char *args_json,
                         sofuu_event_cb on_event, void *opaque,
                         uint64_t *out_cancel_id);

/* Request cancellation of a streaming call. Calls sofuu.agent.cancel(id).
 * Returns SOFUU_OK if the cancel ID was recognized, -1 if not found. */
int sofuu_rt_cancel(SofuuRuntime *rt, uint64_t cancel_id);

/* ── Eval escape hatch ─────────────────────────────────────────── */
/* Evaluate arbitrary JS (full power when the funnel isn't enough). The
 * source is evaluated as a global script, the event loop is drained, and
 * the synchronous completion value is JSON-stringified into *out_json.
 * For async work, drive it through sofuu_rt_call or manage your own
 * globals + a follow-up read. Returns SOFUU_OK or a SOFUU_ERR_*. */
int sofuu_rt_eval(SofuuRuntime *rt, const char *source, char **out_json);

/* ── Memory ────────────────────────────────────────────────────── */
/* Free a buffer handed out by this library (out_json strings). NULL-safe. */
void sofuu_free(void *ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SOFUU_EMBED_H */
