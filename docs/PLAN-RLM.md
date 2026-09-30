# PLAN — RLM Scaffold, Bundled Embeddings & Headless Runtime

> **✅ STATUS (2026-08-13): R0–R3 landed and verified.** Sandbox
> (`sofuu-ffi/src/qjs_rt.rs` + `sofuu-core/src/rlm/sandbox.rs`), bootstrap
> (`crates/sofuu-core/js/rlm_bootstrap.js`), episode state machine
> (`rlm/episode.rs`), router + JSONL log (`rlm/router.rs`), JS surface
> (`rlm/js_api.rs` + `src/js/rlm.js` → `sofuu.rlm.query/route/logRoute`),
> `/rlm` chat command (on/off/auto gate in `turn()`), `examples/rlm_demo.js`,
> and the acceptance test `examples/rlm_mock_test.js`.
> **Evidence:** 33 rlm unit tests green (of 121 total), mock-server E2E
> 15/15 assertions pass (needle-in-context via llm sub-call → final, trace
> kinds llm/final/chunk present, router matrix correct), binary 1.8MB
> (≤5MB cap green).
> **Deviations from this document, all benign:** (a) zero new C — not even an
> ffi_shim addition: `JS_NewCFunctionData` is a real exported symbol, and the
> only C touched is the 3-line `SOFUU_RUST_CORE`-guarded registration seam in
> `engine.c`; (b) v1 answers `llm()` sub-prompts with plain completions —
> nested episodes (depth>0) come with the agent/Track-B integration;
> (c) char (not byte) offsets are exposed to JS.
> **Still open here (the "fully done" delta):**
> - Real-provider proof — all green runs are vs. the mock server; first live
>   run (`examples/rlm_demo.js` vs Ollama) may need driver-prompt tuning.
> - Nested-episode recursion (v2): v1 answers `llm()` sub-prompts with plain
>   completions, so `depthReached` is always 0.
> - Chat E2E for the `/rlm` flow in a real pty (unit-tested, not yet
>   session-tested).
> - Esc aborts *between* rounds; mid-request abort of an in-flight sub-call.
>   **DONE (2026-08-21):** asks run stream-first with a 100ms watcher that
>   kills in-flight streams the moment `__rlm_aborted` is set — Esc now
>   disconnects mid-request (`tests/rlm_mock_test.js` drip probe).
> - Router window sizes are a static per-provider table (128k/32k) until
>   per-model windows are wired in.
>   **DONE (2026-08-21):** per-model windows via the `MODEL_CTX` table in
>   `src/js/agent.js` (`sofuu.agent.contextWindow(model)`); `/ctx` override
>   still wins.
> - Real `cargo-fuzz` runs (targets compile; cargo-fuzz not installed here).
> - The E-track (E0–E5 embeddings) and R4 (KV fusion hook, rides Track B).

 **Scope (decided in discussion, 2026-08-12):**
> 1. **RLM (Recursive Language Model scaffolding)** for long-context work —
>    an inference-time control loop (a *program*, not a model): the context
>    lives as data in a sandboxed environment; the model inspects it with
>    small JS snippets and can call itself recursively on sub-chunks.
> 2. **Custom bundled embeddings** — a distilled hashed-MLP embedder
>    (~500KB–1MB) replacing TF-IDF as the default brain/memory embedder,
>    plus a *deferred* ≤300KB routing model (heuristics first).
> 3. **Headless/embeddable runtime** — Sofuu is a runtime for AI
>    applications first: every feature (RLM, embeddings, brain/memory, HTTP,
>    MCP) must be consumable by any device/app **without the TUI layer**.
>
> **Verified in-tree anchors (2026-08-12):**
> - QuickJS has everything a hermetic sandbox needs: `JS_NewRuntime`
>   (quickjs.h:334), `JS_SetMemoryLimit` (:337), `JS_SetMaxStackSize` (:340),
>   `JS_NewContext` (:353), `JS_SetInterruptHandler` (:855).
> - Engine API: `engine_create/register_builtins/eval_string/run_jobs`
>   (`src/engine/engine.h:15–34`); public C API `sofuu.h`
>   (`sofuu_init/eval_file/eval_string/run_jobs/get_engine/destroy`).
> - CMA is Rust behind C-ABI exports (`ffi_exports.rs:311–599`):
>   `sofuu_cma_new/free/hydrate/remember/recall/kv_hints/entities/
>   remember_entity/mark_positive/forget/decay/consolidate/count/
>   records_json/vectors`. Brain vectors are stored **with their raw text**
>   — so re-embedding on embedder swap is possible offline.
> - `sofuu.ai.estimateTokens` exists (`mod_ai.c:2319`); `embedLocal` (768-dim
>   trigram TF-IDF, pure C) is bound (`js_ai_embed_local`); the chat driver's
>   `embedText` falls back Ollama → embedLocal (`chat.rs:1203–1216`).
> - Binary today ≈1.8MB; hard cap 5MB enforced by `make size-check` in CI.
> - KV page search uses 64-dim mean-pooled K summaries — no learned embedding
>   in that path at all (`mod_kv.c`). Small embeddings are native to the
>   design, not a compromise.
>
> **Frozen design decisions (from the discussion — do not re-litigate):**
> - RLM sandbox = **nested QuickJS context**, never a Python env. In-process,
>   ~µs spin-up, works on phones.
> - Routing between RLM and a plain call = **heuristics first**, with every
>   decision logged; a learned ≤300KB router is only built if the logs prove
>   the heuristics misroute.
> - Brain embedder = **distilled hashed-MLP** (hash-embedding table + pooling
>   + tiny MLP, int8), **not** a tiny transformer. Bar: clearly beat TF-IDF
>   on paraphrase recall; nomic is a reference point, not a target.
> - Models bundle via `include_bytes!`; binary stays ≤5MB; first-run download
>   is the escape hatch only.
> - None of the new subsystems may depend on `chat.rs`/the TUI.
 - **No new handwritten C for RLM or embeddings.** All logic is Rust
   (`sofuu-core`) + a fixed JS driver program in the sandbox. The only C
   involved is *pre-existing, unmodified*: QuickJS engine internals
   executing sandbox code (the agreed "C stays for the engine" category),
   the provider path as an unchanged callee, and existing `ffi_shim.c`
   wrappers. JS-visible APIs are registered from Rust via
   `register_global_fn` (`sofuu-ffi/src/bridge.rs:100`) — the same pattern
   `chat.rs` already uses for its ten `__chat_*` globals. Proof this needs
   no new C: the sandbox primitives (`JS_NewRuntime`, `JS_SetMemoryLimit`,
   `JS_SetMaxStackSize`, `JS_NewContext`, `JS_SetInterruptHandler`) are
   plain `extern "C"` functions in quickjs.h — Rust FFI declares and calls
   them directly.

### Security posture & trust model (shipped with R0/R1/R3 + the guardrails batch)

- **Hermetic sandbox.** Snippets run in a nested QuickJS context holding only
  the frozen `__hx_*` whitelist (7 natives) plus the frozen helper layer
  (`len/chunk/peek/grep/lines/llm/llmBatch/final/emit`): every name is
  `Object.defineProperty`-locked (`writable:false, configurable:false`), so
  model code cannot redefine the API mid-episode. No `std/os/quickjs-libc`
  intrinsics, no `fetch/process/require/sofuu` reach the sandbox; the context
  string lives host-side and is only ever served out in slices.
- **Deterministic execution.** `Date` and `Math.random` are counter/seed
  stubs reset before every snippet attempt, so an llm() suspension re-run
  replays bit-for-bit; nothing wall-clock or random can fork a re-run.
- **Budgets as cost guardrails.** Memory cap (32MB) + stack cap (1MB) +
  per-eval time slice (`eval_slice_ms`, default 5s) nested inside the
  wall-clock episode budget; `max_llm_calls`/`max_rounds`; 64KB answer cap;
  512-event trace cap with `dropped_events` accounting; 256KB per-eval cache
  injection cap with loud truncation markers; identical suspension batch 3×
  in a row ⇒ `stopped="loop_detected"`.
- **What the provider sees (the trust boundary).** The model provider
  receives the transcript, which contains *slices of the context* the model
  chose to print via snippet results — so **never run RLM over secrets you
  wouldn't send that provider**. Conversely: API credentials
  (`api_key`/`base_url`) stay in the JS provider layer; Rust never
  deserializes them, so they cannot appear in traces, `RlmResult`, or the
  router JSONL (log lines carry only `ts_ms/ctx_tokens/window_tokens/
  question(≤200c)/route/latency_ms/rlm_calls` — never prompts, never keys;
  regression-tested in `js_api::tests::api_credentials_never_reach_results_or_logs`).
- **Sentinel spoofing.** `\0RLM_SUSPEND:` / `\0RLM_FINAL:` only fire from the
  native throw path; cache answers or context text containing those exact
  byte strings round-trip as inert data (tests:
  `cache_answers_with_hostile_bytes_round_trip_verbatim`,
  `context_sentinel_strings_do_not_spoof`).
- **Residual risks (bounded).** Budget spend within the caps is possible by
  design (a cooperative model can still burn `max_llm_calls` × provider
  cost), and hostile context text can try to *talk the model into a bad
  strategy* (prompt injection via document) — bounded by budgets + trace,
  not eliminated. Fuzz targets `rlm_snippet`/`rlm_reply` keep eval/step
  panic-free on garbage input.
- **Cross-ref.** When PLAN-AGENTS A4 (per-agent tool whitelists + budgets)
  lands, tool access from inside the RLM sandbox inherits those
  whitelists/budgets — the RLM sandbox adds no new tool surface of its own.

---

## Part 1 — RLM

### R0 — The sandbox (`crates/sofuu-core/src/rlm/sandbox.rs`)

A hermetic, nested QuickJS environment. This is the component other CLIs
solve with a Python subprocess; Sofuu does it in-process.

1. **Construction:** per RLM episode, create `JS_NewRuntime()` →
   `JS_SetMemoryLimit(rt, 32MB)` → `JS_SetMaxStackSize(rt, 1MB)` →
   `JS_NewContext(rt)` → minimal intrinsics only (no std/os modules). All
   calls through new `sofuu-ffi` wrappers (unsafe stays in the FFI crate).
2. **Termination guards:** `JS_SetInterruptHandler` → callback checks a
   deadline (wall-clock μs) + an instruction-ish step counter → returning
   non-zero aborts evaluation. Host sets deadline per episode.
3. **Whitelisted host API** (the *only* globals registered in the sandbox —
   no `sofuu.*`, no fetch/fs/spawn/process):
   ```
   len()                          → total context chars
   count()                        → number of chunks
   chunk(i[, startChar, endChar]) → chunk i text (optionally a slice)
   grep(regex[, maxHits])         → [{chunk, index, line}...] across context
   peek(charStart, charEnd)       → raw slice of the full context
   llm(subPrompt)                 → async recursive model call (R1)
   emit(tag, text)                → observation log (trace)
   ```
4. The context string is injected as an immutable host-side buffer
   (`c externally-owned; sandbox reads via `chunk`/`peek`, never copied in
   whole into JS-visible memory unless the model asks).
5. **Loop sharing:** sandbox promises resolve through the host's libuv loop +
   job pump (`engine_run_jobs` pattern). The sandbox never owns a loop; the
   host drives every step. `llm()` returns a JS promise fulfilled by the
   host after the recursive call finishes.

**Verify:** unit tests in `rlm/sandbox.rs` — eval `chunk(0)` round-trip;
memory limit kills an `new Array(1e9)` bomb with a clean error; interrupt
handler kills `while(1){}` inside the deadline; sandbox cannot see `sofuu`,
`fetch`, `process`, `require`, or host globals.

### R1 — The RLM loop (`crates/sofuu-core/src/rlm/mod.rs`)

1. **Episode shape:**
   ```rust
   pub struct RlmRequest {
       pub context: String,          // the long context (never windowed whole)
       pub question: String,
       pub opts: RlmOpts,            // budgets below
   }
   pub struct RlmOpts {
       pub provider: String, pub model: String,   // recursion target
       pub max_depth: u32,           // default 3
       pub max_llm_calls: u32,       // default 24
       pub max_wall_ms: u64,         // default 120_000
       pub chunk_chars: usize,       // default 16_000 (~4k tok)
       pub trace: bool,
   }
   pub struct RlmResult {
       pub answer: String,
       pub calls: u32, pub depth_reached: u32, pub ms: u64,
       pub trace: Vec<RlmEvent>,     // every chunk read / grep / llm call
   }
   ```
2. **Driver program** (fixed, shipped — the model only fills the strategy):
   the sandbox receives a *system program* JS text teaching the API plus the
   question, e.g. "context: N chunks; find the answer; use `final(answer)`
   when done". The model's top-level task runs as an async sandbox function;
   `final()` signals completion. Depth guard: `llm()` at `depth ==
   max_depth` runs with the whitelist minus `llm`.
3. **Recursion target:** `llm()` bridges to the existing provider layer
   (`sofuu.ai.complete` path in `mod_ai.c`) via a host callback with
   `{prompt, depth}`; result text returns into the sandbox promise. Remote
   providers only for v1 (Track B makes it local later — same seam).
4. **Chunking:** UTF-8-safe boundaries, paragraph-preferred split, overlap
   200 chars between chunks; chunk map kept host-side (Rust `Vec<Range>`), so
   `chunk(i)` is O(1).
5. **Budgets & failure:** hitting any budget → graceful partial answer with
   `trace` annotated `stopped: budget=<name>`; provider error inside
   recursion → surfaces to the model as a thrown JS error it may catch.
6. **Trace:** every `chunk/grep/peek/llm/final` call logged with timing —
   this *is* the routing training data for the deferred learned router (R3).

**Verify:** scripted — 400KB lorem-with-needle corpus ("the code word is
X" planted in chunk 173) → model finds it without the corpus ever entering a
prompt window whole; depth cap respected (recursive bomb stopped at 3);
budget stop produces a partial answer; trace file lists every call.

### R2 — JS API (`sofuu.rlm`) + chat integration

1. **`sofuu.rlm.query(context, question, opts)`** — async JS API registered
   **from Rust** via `register_global_fn` (`bridge.rs:100`, the `__chat_*`
   precedent — `engine.c` is not touched), backed by `sofuu-core::rlm` over
   FFI. This is the **headless entry point** — any app embedding the runtime
   gets RLM with zero chat/TUI code (Part 3).
2. **Chat hook:** in the driver `turn()` path, when the router (R3) says
   RLM, the prompt + attached context route to `sofuu.rlm.query` and the
   answer streams out as the turn's answer with a one-line trace summary
   (`⏺ rlm · 9 calls · depth 2 · 41s`). Esc aborts via the existing
   `currentStream.abort` seam → cancels the episode deadline.

**Verify:** `sofuu run examples/rlm_demo.js` works with no chat involved;
pty chat test with a 300KB pasted log file + question → RLM path engages and
answers.

### R3 — Router v0 (heuristics + logging) and the deferred learned router

1. **Heuristic v0** (`rlm/router.rs`):
   ```rust
   fn route(ctx_tokens: usize, window: usize, task_hint: TaskHint, q: &str) -> Route
   ```
   - `ctx_tokens > 0.8 × effective_window` → `Rlm`
   - else if `ctx_tokens > 0.4 × window` AND question looks *holistic*
     ("summarize all", "across the document", "every occurrence") → `Rlm`
   - else → `Plain`
   - hard opt-outs per turn: `@file` attachments already fit, `/verify`
     second pass, tool-planning rounds → always `Plain`.
2. **Decision log:** every route decision appends one JSON line to
   `~/.sofuu/routing_log.jsonl` (`{ctx_tokens, window, task, route, latency,
   rlm_calls, user_accepted?}` — `user_accepted` = user didn't immediately
   re-ask/abort). Rotation at 5MB.
3. **Learned router — deferred by design.** Only when the log shows
   heuristic misroute clusters: distill a tiny classifier over *structural
   features* (token counts, chunk count, hint flags, question unigrams), int8,
   ≤300KB, bundled like the embedder. Acceptance gate: ≥8-point routing
   accuracy gain over heuristics on the logged eval split. If that gate
   isn't met, heuristics stay forever — that is a fine outcome.

**Verify:** unit tests for the heuristic matrix; log lines appended in a pty
session; rotation works.

### R4 — Deep fusion hooks (rides on Track B, documented not built)

- When B1 (QTSQ KV bridge) lands: RLM chunks already indexed in the CMA can
  reference **KV pages** instead of text — inject prior KV state for
  re-visited chunks via `sofuu_cma_kv_hints` (export exists,
  `ffi_exports.rs:423`) instead of re-reading raw text.
- This is the differentiation no Python-REPL CLI can copy. One paragraph in
  ROADMAP §2 when R1 ships; no code now.

---

## Part 2 — The bundled brain embedder

### E0 — Recall-quality benchmark first (`benchmarks/embedding_bench/`, ~2–3 days)

Without this, E2 has no acceptance gate. Build it before the model.
1. **Corpus:** ~2–4k (query, relevant-memory, distractor-pool) triples mined
   from: (a) dogfood chat logs (`.sofuu/sessions/*.qtsq` — we record prompts
   and answers), (b) public paraphrase sets (Quora QP / STS-B style pairs
   re-licensed cleanly), (c) synthetic memory items from our own domain
   (deploy configs, file paths, error strings, "how do I X" asks).
2. **Metrics:** recall@1/@5/@20, MRR against a 1k-record distractor pool.
3. **Baselines:** current TF-IDF `embedLocal`, plus nomic-embed-text via
   Ollama as an *oracle reference* (never a shipping target).
4. **Ship gate (frozen):** custom embedder recall@5 ≥ TF-IDF recall@5
   **+15 points** on this suite, and ≥ 0.90 × nomic recall@5. TF-IDF stays
   the fallback if the gate fails — the plan survives a training miss.

**Verify:** `node benchmarks/embedding_bench/run.js` (sofuu-run) prints the
baseline matrix; CI artifact stores the JSON.

### E1 — Model architecture & on-disk format (~2 days design, fold into E2)

1. **Architecture (target ≤ 1MB int8):**
   - Tokenizer: lowercase → word + char-trigram tokens (mirrors TF-IDF
     tokenization habits so OOV degrades gracefully).
   - Hash-embedding: murmur3(token) mod `V` (V ≈ 50k) → 48-dim table
     (~2.4M params fp32 ≈ 2.4MB fp32 → **~600KB int8**) — bloom-style
     collisions are fine (semantic dedup by collision is acceptable noise).
   - Pooling: mean + max concat (96-dim) → MLP 96→256→256 GELU → L2-norm
     → **256-dim unit vector**.
2. **On-disk format (`crates/sofuu-core/src/embed/format.rs`):** documented
   packed int8 format: header (`magic "SOFEMB1"`, version, dims, V, layer
   shapes, per-layer scales) + raw int8 blocks. Not QTSQ — QTSQ is optional
   in builds and the embedder must work in degraded CI builds too.
3. **Output dim note:** brain `VEC_DIM` is 768 today (`chat.rs:1187`) — the
   dim is a CMA constructor param (`sofuu_cma_new(vec_dim)`), so 256-dim
   vectors work; the dim is recorded in the brain header (E4 versioning
   covers it).

### E2 — Training pipeline (`tools/embedder/`, ~1 week; *offline, not shipped*)

1. **Teacher-student distillation:** teacher = nomic-embed-text (or bge-small
   via HF) over the E0 corpus + extra unlabeled memory-style text (~200k
   short texts); loss = cosine similarity matrix mimicry (student's pairwise
   sims ≈ teacher's) + light InfoNCE on paraphrase pairs.
2. **Stack constraint:** PyTorch in a venv under `tools/embedder/` — training
   never ships; the runtime only needs the exported int8 blob.
3. **Quantization:** post-training int8 per-channel for the hash table,
   per-tensor for MLP scales; quantize-then-evaluate on E0 (the gate is
   measured on the *quantized* model, not the fp32 one).
4. Export → `crates/sofuu-core/assets/embedder_v1.sofemb` +
   `EMBEDDER_ID = "sof-emb-1"`.

**Verify:** quantized model passes the E0 ship gate; artifact + training
script committed; README in `tools/embedder/` (one command reproduces).

### E3 — Runtime inference (`crates/sofuu-core/src/embed/infer.rs`, ~2–3 days)

1. Pure-Rust inference: hash → gather → mean/max pool → two int8 matmuls
   (dequant per-channel) → GELU → L2-norm. At ≤1M int8 MACs per query this
   is **microseconds-scale**; start scalar/`iter().map()` Rust, no
   hand-intrinsics — the existing C SIMD kernels stay untouched. (Profile
   once; if a chat turn's embed ever shows >2ms, revisit with the NEON/AVX2
   dot path via FFI.)
2. API: `Embedder::embed(&str) -> [f32; 256]`, `Embedder::id() -> &str`,
   built-in instance loaded from `include_bytes!("../assets/embedder_v1.sofemb")`.
3. Exports in `ffi_exports.rs` following the established memory contract
   (malloc-owned buffers, NULL-tolerant): `sofuu_embed_builtin(text, out_dim,
   out_vec*)`, `sofuu_embed_builtin_id()` → `"sof-emb-1"`.

**Verify:** Rust tests — unit norm; deterministic; paraphrase pair
("restart the server" / "reboot the backend") similarity > unrelated pair by
a margin; 10k-embed smoke under 1s total.

### E4 — Integration, versioning, migration (~2 days)

1. **`embedLocal` switches — from Rust, `mod_ai.c` untouched:** at runtime
   startup, Rust registers `sofuu.ai.embedLocal` via `register_global_fn`
   (shadowing the C binding when the bundled asset is present). The C
   `js_ai_embed_local` / TF-IDF code remains compiled, unmodified, as the
   degraded fallback (no behavior cliff if the asset is ever absent).
   Return type stays `Float32Array`; the driver's `embedText` fallback chain
   is untouched.
2. **Brain versioning:** brain file header gains `embedder_id` + `vec_dim`
   (mod_memory.c shell writes it; legacy brains default
   `embedder_id="tfidf-768"`). On open with mismatched id: **loud** path —
   refuse to mix; offer `sofuu brain reembed <file>` which walks
   `records_json`, re-embeds each record's stored raw text with the new
   embedder, and rewrites the file (text is always stored, so this is
   offline-safe). Never silently mix vector spaces.
3. **Ollama/nomic stays opt-in:** exact current behavior for users who set
   it; bundled embedder becomes the default for everyone else.

**Verify:** `verify_memory.js` extended — old TF-IDF brain opens (legacy id),
recall works; `brain reembed` migrates it; recall@5 on E0 queries improves
post-migration; mismatch without the reembed command errors clearly.

### E5 — Size budget (frozen)

| Component | Bytes |
|---|---|
| Binary today (measured) | ~1.8MB |
| + embedder asset (int8) | ~600KB–1MB |
| + RLM code (sandbox + loop + router) | ~50–150KB (over-estimate) |
| + learned router (only if E-gate triggers) | ≤300KB |
| **Worst case** | **~3.3MB** |

- `include_bytes!` adds ~1:1; `make size-check` (5MB cap) gates every PR —
  a breach fails CI loudly.
- If any future model needs >2MB, it goes to `~/.sofuu/models/` with SHA-256
  first-run download instead. Language models never bundle — that preserves
  the tiny/full artifact split for Track B (llama.cpp = full artifact).

---

## Part 3 — Headless / embeddable runtime (no TUI, all features)

> *Execution of this part moved to **`PLAN-HEADLESS.md`** (H0–H6), which
> expands and supersedes it. Kept here for context.*

**Principle:** the TUI (`chat.rs`, `tui.c`) is *one consumer* of the runtime,
not the runtime. Every feature above and below is reachable from a plain C
embed without any chat code.

1. **Feature placement rule (enforced in review):** RLM, embedder, brain,
   routing logic live in `sofuu-core` as `pub` modules with a JS API
   registered by the engine and a C-ABI surface in `ffi_exports.rs`. Chat
   only *calls* them. New compile-time check: `rlm/` and `embed/` must not
   `use crate::chat` — add a `#[cfg(test)]` static assertion or a CI grep.
2. **Rust library API** (`crates/sofuu-core/src/lib.rs`): re-export
   `pub use rlm::{rlm_query, RlmRequest, RlmResult};` and
   `pub use embed::Embedder;` so Rust apps link `sofuu-core` directly
   (desktop app candidate: Tauri-style embedding).
3. **C API v2 (`include/sofuu_embed.h`, new):** extends `sofuu.h` without
   breaking it:
   ```c
   /* one-call conveniences over a SofuuRuntime* */
   int  sofuu_embed_text(SofuuRuntime*, const char *text, float *out, int dim);
   int  sofuu_rlm_query_json(SofuuRuntime*, const char *req_json,
                             char **out_json /* malloc'd */);
   void sofuu_free_string(char *);
   ```
   (Sync-blocking variants; async stays "eval JS that uses `sofuu.rlm`".)
4. **Build:** new `make libsofuu` → `target/release/libsofuu.a` (+ `.h`) —
   cargo `crate-type = ["staticlib"]` on a *thin* wrapper crate
   (`crates/sofuu-capi`) so `sofuu-core` stays binary-first. This is the
   iOS/Android/desktop entry point ROADMAP §4 already promises ("Embeddable
   libsofuu (.h/.c)").
5. **Docs + samples:** `examples/embed/` — `c_embed.c` (10-line: init →
   rlm_query_json → print → destroy), `headless_brain.js` (open brain,
   remember, recall via `sofuu run`), and a paragraph in README "Embed
   Sofuu" pointing at both. TUI is never initialized in these paths
   (`sofuu_init` only starts TUI when chat is invoked — verify and document).

**Verify:** `make libsofuu` produces `.a` ≤5MB; `c_embed.c` compiles with
`cc c_embed.c libsofuu.a -lcurl -lpthread` and answers an RLM query against a
mock provider; headless brain JS works with QTSQ disabled build falling back
cleanly; CI green job compiling the C sample on macOS+Linux.

---

## Sequencing & effort

| # | Item | Depends on | Effort |
|---|---|---|---|
| E0 | Embed benchmark harness | — | 2–3 days |
| R0 | Sandbox | — | 2–3 days |
| E2+E1 | Train embedder + format | E0 | ~1 week |
| R1 | RLM loop | R0 | 3–4 days |
| R3 | Heuristic router + logging | R1 (chat hook R2 optional) | 1 day |
| E3 | Runtime inference | E2 | 2–3 days |
| E4 | Versioning + migration | E3 | 2 days |
| R2 | JS API + chat hook | R1, R3 | 1–2 days |
| P3 | Headless API + libsofuu | R2, E3 | 2 days |

E-track and R-track are parallelizable (E0→E2 is the long pole). Suggested
order: **E0 → R0 → E2 → R1 → E3 → R3 → E4 → R2 → P3**.
Learned router (R3b): scheduled only by routing-log evidence.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Distilled embedder misses the E0 gate | TF-IDF stays default; ship RLM without embedding swap; iterate training data (most likely fix: more domain chat text). |
| Small models write bad RLM probe code ("strategy collapse") | Fixed driver program does chunking/grep; model only chooses what to read; few-shot examples in the sandbox program; floor models noted in docs (test with Qwen3-style 4B+). |
| Recursive cost blowup | Hard `max_llm_calls` + wall-clock deadline, both logged; chat shows the call count in the footer line. |
| Sandbox escape via prototype pollution of the whitelist | Sandbox context registers only the 7-function whitelist on a fresh global; no host objects cross (strings/numbers only); fuzz the sandbox wrapper in `crates/fuzz`. |
| Brain vector-space mixing on embedder swap | `embedder_id` + loud refuse + `brain reembed`; never silent. |
| Binary size creep | Frozen budget table above; `size-check` CI; assets ≤1MB each or download-not-bundle rule. |
| libsofuu API surface ossifies too early | Mark C API v2 `experimental` in the header for one release cycle; JS API is the stable path. |

## Done-state definition

- `sofuu run examples/rlm_demo.js` answers questions about a ≥400KB context
  that never fully enters a prompt window, with a persisted trace.
- Chat routes long-context turns to RLM via heuristics; every decision
  logged to `~/.sofuu/routing_log.jsonl`.
- `sofuu.ai.embedLocal` returns the bundled embedder's vectors;
  `EMBEDDER_ID="sof-emb-1"`; old brains migrate via `sofuu brain reembed`;
  E0 gate numbers in the commit message.
- `cc examples/embed/c_embed.c libsofuu.a` produces a working binary with
  RLM + embeddings + brain and **zero TUI symbols required**.
- `make size-check` green (≤5MB); `cargo test` + pty batteries green;
  TASKS.md and ROADMAP.md updated (ROADMAP §4's "embeddable libsofuu" box
  gets checked by P3).
