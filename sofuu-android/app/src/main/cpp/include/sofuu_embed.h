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
/* H-E1 vector-embedding codes (do not overlap the funnel codes above). */
#define SOFUU_ERR_UNKNOWN_SPACE -7   /* space id not recognized            */
#define SOFUU_ERR_MODEL_UNAVAIL -8   /* baked embedding artifact unusable  */
#define SOFUU_ERR_NOMEM         -9   /* host-visible malloc failed         */

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
 *     "brain_path": "/path.qtsq",  // explicit brain file
 *     "api_keys": { "openai": "…" },
 *     "provider": "openai",        // request DEFAULT, used only when a
 *     "model": "gpt-4o-mini",      //   call omits it (per-call wins)
 *     "base_url": "http://host/v1",//  e.g. pin to a local LLM server
 *     "agents": [ { "name":"researcher", "system":"…", "provider":"…",
 *                   "model":"…", "tools":[…], "budget":{…} } ] }
 * provider/model/base_url exist because an embedded host has no interactive
 * `/model` picker: set the endpoint once here instead of on every call.
 * They are read only from the creating runtime's settings, so one runtime's
 * defaults never retarget another. The CLI sets none, so chat's
 * "no model configured" guidance is unchanged.
 *
 * NOTE: there is no "qtsq" config key. Whether the encrypted-brain codec is
 * present is a BUILD-time property of the library, not a runtime toggle.
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

/* Streaming / cancellable variant (agent.run, ai.stream, …).
 *
 * Two shapes are supported, and one driver handles both:
 *
 *  1. Methods taking an `onStep` option (agent.run) forward each step
 *     event to the host. args_json should be
 *     {"target":"<name>","task":"…",…opts}.
 *  2. Methods returning an async iterable (ai.stream) are drained: each
 *     chunk is forwarded as {"kind":"delta", …}.
 *
 * The stream ALWAYS ends with exactly one terminal event
 * {"kind":"done", …}:
 *   - on success: {"kind":"done","deltas":<n>,"aborted":<bool>,"usage":…}
 *     (for the onStep shape the final value is in "result")
 *   - on failure: {"kind":"done","error":{"code":…,"message":…}}
 * A host that waits for `kind:"done"` is never left hanging, and this
 * signature has no out_json — the final value is delivered as that event.
 *
 * *out_cancel_id receives a token usable with sofuu_rt_cancel; it is
 * written BEFORE the stream starts, so a callback may cancel in-flight.
 * Returns SOFUU_OK or a SOFUU_ERR_*. */
int sofuu_rt_call_stream(SofuuRuntime *rt, const char *method,
                         const char *args_json,
                         sofuu_event_cb on_event, void *opaque,
                         uint64_t *out_cancel_id);

/* Request cancellation of a streaming call. Works for both shapes: a
 * stream in flight on this thread is flagged (the ai.stream pump checks
 * it once per chunk), and agent.cancel(id) is nudged when present.
 * Returns SOFUU_OK if the cancel ID was recognized, -1 if not found. */
int sofuu_rt_cancel(SofuuRuntime *rt, uint64_t cancel_id);

/* ── Eval escape hatch ─────────────────────────────────────────── */
/* Evaluate arbitrary JS (full power when the funnel isn't enough). The
 * source is evaluated as a global script, the event loop is drained, and
 * the synchronous completion value is JSON-stringified into *out_json.
 * For async work, drive it through sofuu_rt_call or manage your own
 * globals + a follow-up read. Returns SOFUU_OK or a SOFUU_ERR_*. */
int sofuu_rt_eval(SofuuRuntime *rt, const char *source, char **out_json);

/* ── Vector embeddings (H-E1 text, M1 image) ─────────────────────── */
/* Direct float-vector entry points so hosts never JSON-encode embeddings.
 * The embedders are stateless pure functions; `rt` is reserved for future
 * per-runtime config and may be NULL on all four calls.
 *
 * Space ids (stable strings): "hash-768" (default when NULL/"" — and the
 * same space the brain stores in, so embed→remember needs no space
 * argument), "sem1-64", "sem2-64". One space per index — never mix spaces
 * in a single store. Query sofuu_embed_info() for the live manifest and
 * its default_space.
 * *out is malloc-owned (dim / n*dim floats); the host frees it with
 * sofuu_free(). Error codes: SOFUU_OK, SOFUU_ERR_INVALID_ARG,
 * SOFUU_ERR_UNKNOWN_SPACE, SOFUU_ERR_MODEL_UNAVAIL, SOFUU_ERR_NOMEM. */
int sofuu_embed_local(SofuuRuntime *rt, const char *text, const char *space,
                      float **out, size_t *out_dim);
int sofuu_embed_batch(SofuuRuntime *rt, const char **texts, size_t n,
                      const char *space,
                      float **out, size_t *out_n, size_t *out_dim);
/* Image bytes (PNG/JPEG) → img1-64 vector (joint with sem2-64 text
 * geometry, so text queries retrieve images). *out is malloc-owned floats;
 * host frees with sofuu_free(). rt may be NULL. Undecodable input reports
 * SOFUU_ERR_MODEL_UNAVAIL (no vector exists for it). */
int sofuu_embed_image(SofuuRuntime *rt, const uint8_t *bytes, size_t len,
                      float **out, size_t *out_dim);
/* Space manifest: malloc'd {"ok":true,"result":{...}} (default_space,
 * per-space model ids, dims, artifact hashes). Host frees with sofuu_free. */
int sofuu_embed_info(SofuuRuntime *rt, char **out_json);

/* ── Provider voice (M2C) ──────────────────────────────────────── */
/* Thin funnel routing to ai.transcribe / ai.speak (same implementation
 * the JS surface uses — no HTTP duplication). Hosts base64 audio
 * themselves; the alphabet is validated before embedding (no JS string
 * escape possible). opts_json NULL/"" selects {}. Speak resolves
 * {audio, format} — over JSON transport the Uint8Array travels as an
 * indexed-byte object; hosts reassemble it. Returns the funnel rc with
 * a malloc'd envelope (host frees with sofuu_free()). */
int sofuu_voice_transcribe(SofuuRuntime *rt, const char *audio_b64,
                           const char *opts_json, char **out_json);
int sofuu_voice_speak(SofuuRuntime *rt, const char *text,
                      const char *opts_json, char **out_json);

/* ── Memory ────────────────────────────────────────────────────── */
/* Free a buffer handed out by this library (out_json strings). NULL-safe. */
void sofuu_free(void *ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SOFUU_EMBED_H */
