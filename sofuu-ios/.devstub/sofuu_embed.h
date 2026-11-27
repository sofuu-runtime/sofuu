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
 *
 * ── Threading contract, per entry (v2 freeze, 2026-10-16) ──────────────
 * The owning-thread rule above is the base. Per entry, the truth is:
 *
 *   sofuu_rt_new                owning thread (thread becomes the owner)
 *   sofuu_rt_free               owning thread
 *   sofuu_rt_call               owning thread (blocks; drains the loop)
 *   sofuu_rt_call_stream        owning thread (blocks; pumps the loop)
 *   sofuu_rt_cancel             owning thread (it RUNS JS to flag the cancel,
 *                               and reads the stream registry thread-locals —
 *                               cross-thread cancel is undefined today;
 *                               cross-thread STOPPING still works through
 *                               host_tool_resolve-side cancellation and the
 *                               opts.signal token flowing in JS)
 *   sofuu_rt_eval               owning thread (blocks)
 *   sofuu_rt_eval_nb            owning thread (queues; never waits on I/O)
 *   sofuu_rt_register_host_tool owning thread
 *   sofuu_rt_host_tool_resolve  ANY thread (header-documented exception; the
 *                               result is marshalled onto the owning thread
 *                               through the runtime's event loop)
 *   sofuu_embed_set_log_cb      any thread BEFORE runtimes exist; process glob
 *   sofuu_embed_local/batch/
 *   image/info + voice        any thread (stateless pure functions)
 * Contained-panic note: after SOFUU_ERR_PANIC, the runtime is POISONED —
 * rt_free it and create a fresh one on the owning thread; the process
 * survives by contract.
 *
 * ── Stability: v2 FROZEN (2026-10-16) ─────────────────────────────────
 * v2 is frozen: existing signatures, existing semantic guarantees (one-done
 * streaming, error envelope, handle tombstones), and existing code numbering
 * do not change. The amendment ladder, in increasing severity:
 *   1. additive: new error ENUM values, new optional config keys, new funnel
 *      METHODS via the existing string funnel (no new symbols) — any minor.
 *   2. additive symbol: a genuinely new export — any minor, abi-check
 *      baseline acknowledged in the same commit.
 *   3. wire-shape change to an existing envelope/event — requires a bump to
 *      v3 and a migration note; never silently.
 * Enforcement = scripts/abi_symbols.txt + scripts/check_abi.sh (the dylib's
 * exported symbol set must equal the acknowledged baseline); linker-level
 * symbol versioning is deliberately NOT used (ELF/MachO divergence, and the
 * baseline gate already fails any un-acknowledged export).
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
/* Embedded builds unwind at the ABI boundary (the `sofuu` CLI binary keeps
 * panic=abort): a contained Rust panic returns this, with details in
 * out_json's error envelope — or, for sofuu_rt_call_stream, in a synthesized
 * terminal {"kind":"done","error":{...}} so the one-terminal contract holds
 * even through a panic. The runtime stays usable for the next call. */
#define SOFUU_ERR_PANIC         -10  /* Rust panic contained at the ABI    */

/* ── Versioning ────────────────────────────────────────────────── */
/* Returns the ABI version this library implements (v2 = 2). v2 is FROZEN
 * (2026-10-16): semantics and signatures above are stable under the
 * amendment ladder in the header preamble — additive enum values and new
 * funnel methods are minor; any change to an existing wire shape or
 * signature bumps to v3 and requires a migration note. */
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
 * NOTE: there is no "qtsq" config key. QTSQ is a core runtime component and
 * every published Android SDK artifact links it; a deliberate codec-less
 * development build is a build-time property, not a runtime toggle.
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
 * settles. While no UI session is attached, that means draining libuv until
 * it is idle; with a session attached (sofuu_ui_attach_runtime), `sofuu.ui`
 * keeps a 16 ms event pump alive, so the call instead runs one libuv round at
 * a time and returns as soon as the feature's envelope lands. Returns
 * SOFUU_OK or a SOFUU_ERR_*. */
int sofuu_rt_call(SofuuRuntime *rt, const char *method,
                  const char *args_json, char **out_json);

/* Streaming / cancellable variant (agent.run, ai.stream, ai.streamLocal).
 *
 * `args_json` follows the SAME contract as sofuu_rt_call: a JSON object is
 * passed as one options argument, a JSON array is spread as positional
 * arguments, and NULL/"" means none. The array form is what reaches
 * positional methods — `ai.streamLocal` takes
 * (modelPath, prompt, options?) — so a host streams a local model with
 *   ["/path/to/model.gguf", "hello", {"maxTokens": 128}]
 * and a local SafeTensors/PyTorch/NumPy model directory the same way (the
 * path alone routes in-process; no provider hint is needed).
 *
 * Two shapes are supported, and one driver handles both:
 *
 *  1. Methods taking an `onStep` option (agent.run) forward each step
 *     event to the host. args_json should be
 *     {"target":"<name>","task":"…",…opts}.
 *  2. Methods returning an async iterable (ai.stream, ai.streamLocal) are
 *     drained: each chunk is forwarded as {"kind":"delta", …}.
 *
 * In BOTH forms the driver injects `onStep` and the cancel token into the
 * options object — for the array form that is the trailing object, and one is
 * appended when the caller passed none — so sofuu_rt_cancel works either way.
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

/* ── Local-first modules through the funnel ────────────────────── */
/* The generic funnel already reaches the modules an embedded host needs;
 * no extra exported symbol is involved. A JSON array is spread as
 * positional arguments, a JSON object is one options argument.
 *
 * Auth / crypto (no network; the host supplies the JWKS):
 *
 *   sofuu_rt_call(rt, "crypto.sha256", "[\"abc\"]", &out);
 *   sofuu_rt_call(rt, "crypto.base64urlEncode", "[\"hello\"]", &out);
 *   sofuu_rt_call(rt, "crypto.verifyJWT",
 *       "[\"<jwt>\", {\"keys\":[...]}, {\"issuer\":\"https://id.example\"}]",
 *       &out);
 *
 * The outer envelope is {"ok":true,"result":...}; verifyJWT's own result is
 * {"ok":true,"payload":{...}} or {"ok":false,"error":{...}}. A missing
 * `azp` is accepted (native SDK tokens have no browser origin); a present
 * `azp` must be in authorizedParties.
 *
 * Local storage and agents:
 *
 *   sofuu_rt_call(rt, "storage.open", "{\"base_path\":\"/data/app\"}", &out);
 *   sofuu_rt_call(rt, "agent.run", "[\"researcher\",\"task\",{}]", &out);
 *   sofuu_rt_call_stream(rt, "ai.streamLocal",
 *       "[\"/models/model.gguf\",\"prompt\",{}]", on_event, opaque, &cancel);
 *
 * `storage.*` works in every build. `memory.*` needs QTSQ, which is linked in
 * every published Android SDK artifact; a deliberate codec-less development
 * build refuses memory with a typed E_CAPABILITY instead of silently no-oping.
 * Check `sofuu.capabilities()` for the loaded build. `ai.streamLocal` is
 * in-process and needs no provider or network.
 *
 * A host that wants the JS implementation of a pipeline can load a single
 * bundled script with sofuu_rt_eval (ES modules must be bundled first) and
 * then drive it through call/call_stream. */

/* ── Host-native tools (PLAN-EMBED-SDK E2) ─────────────────────── */
/* Register a native host function so an in-app agent can call it as a tool.
 *
 * `name` is what the model sees; `description` and `parameters_json` (a JSON
 * Schema object, NULL/"" for {"type":"object","properties":{}}) are how the
 * model decides to call it. `cb` runs ON THE RUNTIME'S THREAD (the same
 * thread that drives sofuu_rt_call_stream), with the arguments the model
 * produced as a JSON object string.
 *
 * The callback answers in one of two ways:
 *   - return a NUL-terminated JSON string → synchronous result. The string is
 *     host-owned and only needs to live for the duration of the call; the
 *     library copies it. A JSON string result reaches the model as its text,
 *     any other JSON value as JSON text.
 *   - return NULL → the host has taken the call and will complete it later
 *     with sofuu_rt_host_tool_resolve(rt, call_id, …). Use this when the work
 *     cannot finish inline (camera, location, permissions, I/O).
 *
 * The async path is bounded by the agent's tool timeout (30 s per tool by
 * default, `toolTimeoutMs` per agent), which is also what keeps the event
 * loop alive while a call is outstanding: a host that never answers gets a
 * tool-timeout error rather than a hung turn.
 *
 * The tool is visible to every agent of this runtime, including agents
 * supplied through the `agents` config key — those are defined during
 * sofuu_rt_new, before a host can register anything, so registration must be
 * resolvable after the fact (it is). Registering a name twice replaces the
 * earlier registration. Tools are scoped to the runtime's thread, per the
 * one-runtime-per-thread rule above. Returns SOFUU_OK or a SOFUU_ERR_*. */
typedef const char *(*sofuu_host_tool_fn)(const char *args_json,
                                          uint64_t call_id, void *opaque);

int sofuu_rt_register_host_tool(SofuuRuntime *rt, const char *name,
                                const char *description,
                                const char *parameters_json,
                                sofuu_host_tool_fn cb, void *opaque);

/* Complete a host tool call whose callback returned NULL.
 *
 * `result_json` is a JSON value the agent sees as the tool result; to fail
 * the call instead, pass a message in `error` (result_json may be NULL then).
 * Safe to call from ANY thread — the completion is delivered to the runtime's
 * thread through its event loop — but the runtime must outlive the call.
 * Returns SOFUU_OK, or -1 when the id is unknown (never pending, already
 * settled, or a zero id). */
int sofuu_rt_host_tool_resolve(SofuuRuntime *rt, uint64_t call_id,
                               const char *result_json, const char *error);

/* Non-blocking eval: same JS semantics as sofuu_rt_eval, but after the
 * eval it performs exactly ONE non-blocking loop tick (microtasks +
 * UV_RUN_NOWAIT + microtasks) instead of draining until idle. Use it to
 * drive async features across repeated calls (a chat turn) while the
 * runtime thread stays schedulable; the eval itself never waits on I/O.
 * Returns SOFUU_OK or a SOFUU_ERR_*. */
int sofuu_rt_eval_nb(SofuuRuntime *rt, const char *source, char **out_json);

/* ── Eval escape hatch ─────────────────────────────────────────── */
/* Evaluate arbitrary JS (full power when the funnel isn't enough). The
 * source is evaluated as a global script, the event loop is drained, and
 * the synchronous completion value is JSON-stringified into *out_json.
 * When a UI session is attached (sofuu_ui_attach_runtime), a mounted
 * `sofuu.ui` view keeps a 16 ms event pump alive, so "drained" means ONE
 * non-blocking tick (the sofuu_rt_eval_nb contract) instead of waiting for
 * an idle loop that never arrives; drive subsequent work with
 * sofuu_rt_call / sofuu_rt_eval_nb. For async work, drive it through
 * sofuu_rt_call or manage your own globals + a follow-up read. Returns
 * SOFUU_OK or a SOFUU_ERR_*. */
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

/* ── Embedded GPU UI (PLAN-EMBEDDED-NATIVE-HOST-C-ABI) ──────────────
 *
 * Tier 2 additive symbols: `sofuu_ui_*` extends the frozen v2 ABI without
 * changing a single existing signature. A host that links a headless build
 * (`--no-default-features`) simply has no UI symbols; the rest of this header
 * is unchanged.
 *
 * Mode A (always available): the host owns the Metal/GLES/Vulkan device and
 * pulls a zero-copy `SofuuUiDrawList` every display-link/Choreographer tick.
 * The six instance pointers borrow the session's double-buffered `DrawList`
 * vectors and stay valid until the next `sofuu_ui_poll_draw_list()` (or
 * `sofuu_ui_session_free()`) on the SAME session.
 *
 * Mode B (build with Cargo feature `ui-surface`): `sofuu_ui_surface_*` draws
 * directly onto a host `MTLDevice`/`CAMetalLayer`/`EGLSurface`. On builds
 * without the backend those calls return SOFUU_ERR_NOT_IMPL (-2); Mode A is
 * never affected.
 *
 * Threading (SofuuRuntime is still owned by ONE thread, Thread R):
 *   sofuu_ui_session_new/free,
 *   sofuu_ui_session_get_atlas,
 *   sofuu_ui_session_set_safe_insets,
 *   sofuu_ui_poll_draw_list,
 *   sofuu_ui_surface_*            host UI/render thread (Thread U)
 *   sofuu_ui_session_set_wake_cb  any thread; stored atomically
 *   sofuu_ui_push_patch_json,
 *   sofuu_ui_ingest_agent_event_json,
 *   sofuu_ui_dispatch_input,
 *   sofuu_ui_poll_event_json      ANY thread (lock-protected queues)
 *   sofuu_ui_attach_runtime       Thread R (it registers JS globals)
 *   sofuu_ui_detach_runtime       Thread R
 *
 * Every entry contains Rust panics and returns SOFUU_ERR_PANIC (-10) instead
 * of unwinding into the host. `sofuu_ui_session_new` returns NULL on panic.
 */

/* Packed GPU instance structs — byte-identical to sofuu-ui::draw and the
 * Metal/GLSL/SPIR-V vertex layouts (verified by crates/sofuu-capi tests). */

typedef struct {
    float viewport_size[2];
    float atlas_size[2];
    float scale_factor;
    float time_sec;
    float _pad[2];
} SofuuUiFrameUniforms; /* 32 bytes */

typedef struct {
    float x, y, w, h;
} SofuuUiClipRect; /* 16 bytes */

#define SOFUU_DRAW_KIND_QUAD   0u
#define SOFUU_DRAW_KIND_RIBBON 1u
#define SOFUU_DRAW_KIND_STROKE 2u
#define SOFUU_DRAW_KIND_GLYPH  3u

typedef struct {
    uint32_t kind;   /* SOFUU_DRAW_KIND_* */
    uint32_t start;  /* instance start index */
    uint32_t count;  /* instance count */
    uint32_t _pad;
} SofuuUiDrawCmd; /* 16 bytes */

typedef struct {
    float    rect[4];       /* x, y, w, h (logical px) */
    float    radii[4];      /* tl, tr, br, bl */
    uint32_t color_top;
    uint32_t color_bottom;
    uint32_t border_color;
    uint32_t shadow_color;
    float    params[4];     /* border_width, shadow_blur, shadow_offset_y, opacity */
    uint32_t clip_id;
    uint32_t _pad[3];
} SofuuUiQuadInstance; /* 80 bytes */

typedef struct {
    float    x0, y0_top, y0_bot;
    float    x1, y1_top, y1_bot;
    uint32_t color_top;
    uint32_t color_bottom;
    uint32_t clip_id;
    uint32_t _pad[3];
} SofuuUiRibbonInstance; /* 48 bytes */

typedef struct {
    float    p0[2];
    float    p1[2];
    float    width;
    float    feather;
    float    dash_len;
    float    gap_len;
    uint32_t color;
    uint32_t clip_id;
    uint32_t _pad[2];
} SofuuUiStrokeInstance; /* 48 bytes */

typedef struct {
    float    pos[2];
    float    size[2];
    float    uv[4];
    uint32_t color;
    float    px_range;
    uint32_t clip_id;
    uint32_t _pad;
} SofuuUiGlyphInstance; /* 48 bytes */

/* One frame, zero-copy. `quads`..`clips` point into the session and are
 * borrowed, not owned — never free them and never read them after the next
 * poll/free. `recompiled` says geometry was regenerated this tick (upload and
 * draw); `needs_next_frame` is 1 while springs/kinetic scroll or queued
 * mutations remain, 0 when the host may pause its display link (0% GPU idle). */
typedef struct {
    SofuuUiFrameUniforms          uniforms;
    const SofuuUiQuadInstance*    quads;
    uint32_t                      quad_count;
    const SofuuUiRibbonInstance*  ribbons;
    uint32_t                      ribbon_count;
    const SofuuUiStrokeInstance*  strokes;
    uint32_t                      stroke_count;
    const SofuuUiGlyphInstance*   glyphs;
    uint32_t                      glyph_count;
    const SofuuUiDrawCmd*         cmds;
    uint32_t                      cmd_count;
    const SofuuUiClipRect*        clips;
    uint32_t                      clip_count;
    uint8_t                       recompiled;       /* 1 if geometry regenerated this tick */
    uint8_t                       needs_next_frame; /* 1 while springs/kinetic scroll active */
    uint8_t                       theme_is_dark;    /* 0 = Light, 1 = Dark */
    uint8_t                       _reserved;
} SofuuUiDrawList;

/* ── Input events (`SofuuUiInputEvent`) ──────────────────────────── */
#define SOFUU_INPUT_POINTER_MOVE    1u
#define SOFUU_INPUT_POINTER_DOWN    2u
#define SOFUU_INPUT_POINTER_UP      3u
#define SOFUU_INPUT_SCROLL          4u
#define SOFUU_INPUT_KEY_DOWN        5u
#define SOFUU_INPUT_TEXT_INPUT      6u
#define SOFUU_INPUT_RESIZE          7u
#define SOFUU_INPUT_CLOSE_REQUESTED 8u

#define SOFUU_MOD_SHIFT (1u << 0)
#define SOFUU_MOD_CTRL  (1u << 1)
#define SOFUU_MOD_ALT   (1u << 2)
#define SOFUU_MOD_META  (1u << 3)

/* Pointer/scroll/resize calls never allocate. `text_utf8` is borrowed for the
 * duration of the call: a key name for KEY_DOWN, a UTF-8 string for
 * TEXT_INPUT, NULL otherwise. Coordinates are logical (not physical) pixels. */
typedef struct {
    uint32_t    kind;         /* SOFUU_INPUT_* */
    float       x;            /* logical px */
    float       y;            /* logical px */
    float       dx;           /* scroll delta_x or resize width */
    float       dy;           /* scroll delta_y or resize height */
    float       scale_factor; /* display DPI factor on RESIZE */
    uint32_t    button;       /* pointer button index (0 = primary/touch) */
    uint32_t    modifiers;    /* SOFUU_MOD_* bitmask */
    const char* text_utf8;    /* key name (KEY_DOWN) or UTF-8 string (TEXT_INPUT), else NULL */
} SofuuUiInputEvent;

/* ── Session handle & function surface ──────────────────────────── */
typedef struct SofuuUiSession SofuuUiSession;

/* Host wake callback: invoked (possibly from background runtime threads!)
 * whenever ScenePatches arrive or input triggers animation, so the host can
 * resume CADisplayLink/CVDisplayLink/Choreographer immediately from 0% idle
 * sleep. Pass cb = NULL to clear. The callback must be quick and must not
 * call back into this session on the same thread that is pushing (queue
 * locks are released before the callback runs). */
typedef void (*sofuu_ui_wake_cb)(void *opaque);

/* Create a UI session. config_json (NULL/"" = defaults) is a JSON object:
 *   { "title": "Sofuu",        // initial window/root title
 *     "auto_title": "…",       // Auto-Composer title (default: title)
 *     "theme": "light"|"dark",
 *     "width": 1024, "height": 680 }
 * On success returns a session; on failure returns NULL and, when out_err_json
 * is non-NULL, writes a malloc'd {"ok":false,...} envelope (sofuu_free it). */
SofuuUiSession *sofuu_ui_session_new(const char *config_json,
                                     char **out_err_json);
void            sofuu_ui_session_free(SofuuUiSession *ui);
int             sofuu_ui_session_set_wake_cb(SofuuUiSession *ui,
                                             sofuu_ui_wake_cb cb,
                                             void *opaque);

/* Top inset (macOS transparent titlebar traffic lights, iOS Dynamic Island /
 * notch, Android status-bar cutouts) while keeping root background full-bleed
 * from (0,0). `bottom_px` is recorded for hosts/backends; the current layout
 * engine consumes the top inset. Negative values clamp to 0. */
int sofuu_ui_session_set_safe_insets(SofuuUiSession *ui,
                                     float top_px, float bottom_px);

/* Font atlas descriptor: a single-channel R8 signed-distance atlas to upload
 * once as an R8Unorm GPU texture. *out_pixels_r8 borrows session memory and
 * is valid until sofuu_ui_session_free() (the atlas never changes). Any out
 * pointer may be NULL. */
int sofuu_ui_session_get_atlas(SofuuUiSession *ui,
                               const uint8_t **out_pixels_r8,
                               uint32_t *out_width,
                               uint32_t *out_height,
                               float *out_px_range);

/* Bind/unbind an embedded SofuuRuntime on the runtime's owning thread.
 * Installs JS host globals (`__sofuu_ui_is_attached`, `__sofuu_ui_push_patch`,
 * `__sofuu_ui_poll_event`, plus the five `__sofuu_ui_auto_*` Auto-Composer
 * funnels) pointing to lock-protected queues/state inside `ui`, so
 * `sofuu.ui.*` and the JS Auto-Composer work identically to `sofuu run --ui`
 * without loading libsofuu_gpu.
 * sofuu_rt_free(rt) automatically unbinds; a freed session is a tombstone that
 * makes the JS globals report detached (-2 / false) with no use-after-free. */
int sofuu_ui_attach_runtime(SofuuRuntime *rt, SofuuUiSession *ui);
int sofuu_ui_detach_runtime(SofuuRuntime *rt);

/* Scene mutation / telemetry ingestion (thread-safe). `patch_json` is a
 * `ScenePatch` object (`{"op":"set_text",...}` or a full
 * `{"op":"replace_scene",...}`). len == 0 means NUL-terminated. A malformed
 * patch is rejected with SOFUU_ERR_INVALID_ARG and never enters the queue.
 * `event_json` is a generic `AutoAgentEvent`; the first event seeds the
 * session's Rust Auto-Composer dashboard, later events patch it in place. */
int sofuu_ui_push_patch_json(SofuuUiSession *ui, const char *patch_json,
                             size_t len);
int sofuu_ui_ingest_agent_event_json(SofuuUiSession *ui,
                                     const char *event_json);

/* Queue an OS input event / pop the next semantic event (thread-safe).
 * poll_event_json writes a malloc'd JSON object (`{"type":"click",...}`,
 * freeing via sofuu_free) and returns 1; it returns 0 with *out_event_json =
 * NULL when the queue is empty. */
int sofuu_ui_dispatch_input(SofuuUiSession *ui, const SofuuUiInputEvent *evt);
int sofuu_ui_poll_event_json(SofuuUiSession *ui, char **out_event_json);

/* Mode A: render-thread frame evaluation. Drains queued patches/input, advances
 * 120Hz springs, and fills `out_list` (zero heap alloc in steady state). Call
 * with the display-link delta in seconds and the logical viewport. The first
 * call compiles geometry; subsequent idle calls are O(1). */
int sofuu_ui_poll_draw_list(SofuuUiSession *ui,
                            float dt_sec,
                            float width_px,
                            float height_px,
                            float scale_factor,
                            SofuuUiDrawList *out_list);

/* Mode B (Cargo feature `ui-surface`): Sofuu-managed GPU rendering onto a
 * host layer/surface. Without the backend these return SOFUU_ERR_NOT_IMPL. */
int sofuu_ui_surface_attach_metal(SofuuUiSession *ui, void *mtl_device,
                                  void *ca_metal_layer);
int sofuu_ui_surface_attach_egl(SofuuUiSession *ui, void *egl_display,
                                void *egl_surface, void *egl_context);
int sofuu_ui_surface_render(SofuuUiSession *ui, float dt_sec,
                            uint32_t width_px, uint32_t height_px,
                            float scale_factor);
int sofuu_ui_surface_detach(SofuuUiSession *ui);

/* ── Memory ────────────────────────────────────────────────────── */
/* Free a buffer handed out by this library (out_json strings). NULL-safe. */
void sofuu_free(void *ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SOFUU_EMBED_H */
