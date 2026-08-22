# Sofuu Roadmap — The Edge-AI Operating System

> **North Star:** The complete edge-AI OS in **one self-contained binary** — a local SLM
> with *compressed, infinite, persistent memory*, agents, and MCP — running **unchanged** on a
> phone, a Raspberry Pi, a Cloudflare Worker, or a cloud server.
>
> *Decided 2026-06-18. Strategy: fuse local inference + memory/KV layer; **llama.cpp as the default
> backend AND a from-scratch custom C+QTSQ inference engine** behind one backend interface; build
> stability and features as parallel tracks; target phones, ARM SBCs/laptops, serverless, cloud.*
>
> *Updated 2026-08-06 (Track D): **Rust-first migration.** The CLI, chat UI, config, memory/CMA,
> HTTP/SSE parsing, MCP JSON-RPC, TS stripper, bundler, and npm safety core are ported to Rust
> (see Track D). C remains for extreme low-level work only: QuickJS (JS engine), libuv (event
> loop), SIMD kernels, QTSQ codec. The custom inference engine (Track C) will be built Rust-first
> where possible.*

---

## 1. The Indispensability Thesis

Everyone can run `llama.cpp`. Nobody else gives a local SLM **infinite context that survives
restarts and costs almost nothing**, via QTSQ KV-cache compression + a cognitive memory architecture.

The lock-in is the **brain file**: once an agent's memory (semantic facts + compressed KV pages)
lives in Sofuu's proprietary **QTSQ** format, the model becomes portable and the *history* doesn't.
You can swap Phi for Qwen for Llama and keep the same accumulated memory. That is the thing companies
can't rebuild and can't leave.

What we already have that nobody else does (verified in-tree, not aspirational):
- **CMA** — HNSW vector store, 4-tier memory (working→episodic→semantic→entity), Ebbinghaus decay,
  k-means consolidation, co-recall "gravity," resonance scoring, TF-IDF "dream" summarization.
  *Ported to Rust (`crates/sofuu-core/src/memory/`) — the C version still serves the JS API until
  the FFI wiring lands (Track D).*
- **QTSQ** — tensor compression at F4/F8/F16/F32 with SSD paging + 2-page RAM LRU + cosine search
  over 64-dim page summaries. Proprietary; the moat. (C, stays C — extreme low-level codec.)
- **SIMD** f32 dot/cosine/L2 (NEON + AVX2), offline TF-IDF embeddings, MCP client+server, HTTP, TS.
- **Rust core** — CLI + chat UI + config + HNSW/CMA + SSE parser + MCP JSON-RPC + TS stripper +
  ESM bundler + npm safety core (spec validation, SHA-1, safe tar extraction) are ported to Rust
  (`crates/sofuu-core/`), with 177 tests and the binary still ~2MB.
- **Chat features (F1–F11)** — brain pin + recall explainability (`/remember` + `/why`),
  session resume (`/resume`), `@file` mentions, brain share/import (`/share` + `/import`),
  cost tracking + budget caps (`/cost`), second-model verification (`/verify`),
  filesystem watcher (`/watch`), `~/.sofuu/hooks.js` middleware, ghost completion
  (`/ghost`), brain HTTP server (`sofuu serve --brain`). All landed 2026-08-20.

The gap that makes it all real: **local inference is stubbed** (`src/llm/llm_local.c`). The
compression engine has no local transformer feeding it KV. Closing that gap is the whole game.

---

## 2. The Fusion Loop (the keystone architecture)

```
                 ┌──────────────── one turn ────────────────┐
 user prompt ──▶ │ 1. embed(prompt)  ──▶ CMA.retrieve(top-k) │
                 │      semantic facts + relevant KV pages    │
                 │ 2. decompress KV pages (QTSQ → blob)       │
                 │ 3. llama_state_seq_set_data(blob)  ◀───────┼── inject past context (no recompute)
                 │ 4. llama.cpp generate (stream tokens)      │
                 │ 5. llama_state_seq_get_data() → blob       │
                 │ 6. QTSQ compress → CMA.store(page + facts) ┼──▶ brain.qtsq (survives restart)
                 └────────────────────────────────────────────┘
```

**Critical correction to the current stub:** do **not** raw-poke `llama_context->kv_self`
(the `llm_local.c:60` TODO) — that is internal and breaks across llama.cpp versions. Use the
**supported** API `llama_state_seq_get_data()` / `llama_state_seq_set_data()` to move a sequence's
KV cache as a documented byte buffer, and have QTSQ compress *that buffer*. This converts the
riskiest task from "reverse-engineer memory layout" into "compress a byte blob."

**Backend interface (the key abstraction):** the fusion loop must not know which engine it runs on.
Define one vtable both backends implement — in **Rust** (a trait + FFI shims over the C engines),
since the runtime shell is Rust-first (Track D):

```c
typedef struct sofuu_infer_backend {
    void* (*load)(const char *model_path, sofuu_infer_cfg *cfg);
    int   (*prefill)(void *ctx, const int *tokens, size_t n, uint32_t seq);
    int   (*decode)(void *ctx, int token, float *logits_out);   /* one step */
    int   (*sample)(void *ctx, const float *logits, sofuu_sampler *s);
    int   (*kv_export)(void *ctx, uint32_t seq, uint8_t **blob, size_t *len);  /* → QTSQ */
    int   (*kv_inject)(void *ctx, uint32_t seq, const uint8_t *blob, size_t len);
    void  (*free)(void *ctx);
} sofuu_infer_backend;
```

- `llamacpp` backend: `kv_export/inject` wrap `llama_state_seq_get/set_data`.
- `sofuu` (custom) backend: native QTSQ KV in/out, no conversion. Same QTSQ adapter path.
- `provider:"local"` picks the backend via `engine:"llamacpp"` (default) or `engine:"sofuu"`.
The QTSQ KV-bridge and CMA fusion are written **once** against this vtable (in Rust).

---

## 3. Two Parallel Tracks

### Track A — Stability ("very stable")  ·  owner-focus: correctness, never regress

**A0 — Safety net & instrumentation** *(do this before touching anything else)*
- [ ] ASAN + UBSAN build in CI on every platform; valgrind on Linux x64.
- [ ] Real test harness: convert the ~17 ad-hoc `test_*.js` (they live in `examples/`) into
      assert-based suites with exit codes and golden outputs; `make test` fails loudly. Move them
      into `tests/`. *(The Rust side already has 66 assert-based tests via `cargo test`; the
      C-side JS tests are not yet consolidated.)*
- [ ] Fuzz every parser (these are the crash/RCE surface): SSE (`http/sse.c`), HTTP server
      (`http/server.c`), MCP JSON-RPC (`mcp/mcp.c`), **QTSQ decoder** (untrusted brain files!),
      TS stripper (`ts/`), npm resolver (`npm/resolver.c`). *(The Rust ports — SSE, JSON-RPC,
      npm tar — are ready for `cargo-fuzz` targets.)*
- [ ] JSValue leak audit — the recurring risk flagged at `http/client.c:548`. Add a debug ref-count
      assert pass.

**A1 — Fix known issues**
- [x] npm resolver `/tmp/sofuu_pkg_XXXXXX` race → `mkstemp`. *(Done in `npm/resolver.c:397`.)*
- [ ] `fetch` streams response bodies (currently buffers the whole body — blocks large model
      downloads and big API responses). `http/client.c`.
- [ ] MCP global-state HACK → opaque/hidden property instead of globals. `mcp/mcp.c:601,606`.
      *(The Rust JSON-RPC port is stateless-safe; wiring it in removes the HACK.)*
- [ ] Promise double-settlement guards. `io/promises.c`.

**A2 — Resource, concurrency, crash-safety**
- [ ] Per-HW-tier memory budgets + backpressure; long-running soak tests (agents that run for days).
- [ ] **Power-loss / corruption recovery for the QTSQ on-disk store** — edge devices lose power
      mid-write. Atomic page writes, checksums, journal/recovery on load.
- [ ] Concurrency stress: many simultaneous HTTP connections + active inference.

### Track B — The Moat (local inference + fusion)

**B0 — Wire local inference** (default backend = llama.cpp, behind `SOFUU_LLM=1`)
- [ ] Define the `sofuu_infer_backend` vtable (§2) — *both* engines target this from day one.
- [ ] Vendor + pin a specific llama.cpp version into `deps/` (pinning matters for the state API).
- [ ] Implement the `llamacpp` backend against the vtable.
- [ ] `sofuu.ai.complete/stream` route to local when `provider:"local"`, `model:"<path>.gguf"`.
- [ ] Token streaming flows through the existing async-iterator path (same JS API, local or remote).
- [ ] Binary stays ~1MB with the flag off; grows only when enabled.

**B1 — The keystone: implement the QTSQ KV bridge** *(replaces the `llm_local.c:60` stub)*
- [ ] `sofuu_llama_kv_export(seq)` → `llama_state_seq_get_data()` → QTSQ compress → page.
- [ ] `sofuu_llama_kv_inject(page)` → QTSQ decompress → `llama_state_seq_set_data()`.
- [ ] Round-trip correctness test: prefill → export → free → inject → continue == no-compression baseline
      (within quant tolerance). This test is the project's heartbeat.

**B2 — Fuse CMA with inference**
- [ ] Auto-memory: each turn embeds + stores; before generation, retrieve top-k semantic facts +
      relevant KV pages and inject them.
- [ ] Flagship demo: **1M-token conversation on a laptop / Pi** with near-flat per-turn cost.

**B3 — Developer API polish**
- [ ] `Agent` class over the fusion loop; MCP tools; structured output; local embeddings.
- [ ] One-screen "agent with infinite memory in 20 lines" example.

### Track C — The Custom Inference Engine (`engine:"sofuu"`, from scratch)

> The proprietary endgame: a tiny, self-contained C+SIMD transformer that runs **QTSQ-quantized
> weights** directly and produces **QTSQ KV** natively — one format, end to end. llama.cpp is the
> **correctness oracle**: every op is validated against its output on the same model/prompt.
>
> **→ Full detailed plan: [`ROADMAP-CUSTOM-ENGINE.md`](ROADMAP-CUSTOM-ENGINE.md)** (file layout,
> quant block formats, oracle harness, per-phase exit criteria). Summary below.

**C0 — Foundations & oracle harness**
- [ ] Numeric parity harness: run a tiny model through llama.cpp, dump per-layer activations as the
      golden reference; the custom engine must match within tolerance at each stage.
- [ ] Pick **one architecture first** (Llama/Qwen-style decoder: RMSNorm + RoPE + SwiGLU + GQA).
- [ ] Tensor core in C: matmul, RMSNorm, RoPE, softmax, SwiGLU, attention — scalar reference first.

**C1 — Quantized kernels (this is where the SIMD work becomes central)**
- [ ] int8 and int4 matmul/dequant kernels: NEON + AVX2 + scalar fallback (extends `src/simd/`).
- [ ] QTSQ **weight** loader: load model tensors directly from QTSQ at F4/F8/F16 (not just KV).
- [ ] Block-wise dequant matching QTSQ's quantization scheme; verify against oracle.

**C2 — End-to-end generation**
- [ ] Full forward pass + sampling; greedy parity with llama.cpp on the chosen arch, then top-p/temp.
- [ ] Native `kv_export/kv_inject` producing QTSQ KV with no conversion step.
- [ ] KV-cache management, GQA, batching.

**C3 — Breadth & perf**
- [ ] Add architectures (Phi, Gemma, etc.); a converter `gguf → qtsq weights`.
- [ ] Perf parity target: within ~1.5–2× of llama.cpp tokens/sec on CPU per HW tier; then close gap.
- [ ] Threading + SIMD tuning per target (phone/SBC/server).

**C4 — Promote to default**
- [ ] Once parity + perf + breadth hold, flip default `engine` to `"sofuu"`; llama.cpp stays as the
      broad-compat fallback. Restores the fully-proprietary, tiny, self-contained binary.

---

## 3b. Track D — Rust-First Migration (the runtime shell)

> *Decided 2026-08-06. The interactive shell and all security-sensitive parsing move to Rust;
> C stays only for extreme low-level work (QuickJS, libuv, SIMD, QTSQ). Binary size stays ≤ 5MB
> (currently ~2MB). Full plan: `~/.commandcode/plans/sofuu-rust-migration.md`.*

**D0 — Scaffold (done)**
- [x] Rust workspace (`Cargo.toml`, `crates/sofuu-core` + `crates/sofuu-ffi`), size-optimized
      profiles (`opt-level=z`, `lto=fat`, `strip`, `panic=abort`).
- [x] `sofuu-ffi` — safe bindings to the C core (`sofuu_init/eval_*`, QuickJS bridge via
      `src/ffi_shim.c`, bundle/npm). All `unsafe` isolated in the FFI crate.
- [x] Rust entrypoint (`crates/sofuu-core/src/main.rs`) — drop-in for C `main()`.
- [x] `make` → `cargo build --release`; `make c-only` keeps the legacy C binary; `make size-check`
      enforces the 5MB cap.

**D1 — Rust owns the shell (done)**
- [x] CLI dispatch (version/help/run/eval/repl/chat/bundle/install/add/licenses) in Rust.
- [x] Chat UI in Rust (`src/cli/chat.c` retired): slash commands (`/models /providers /model
      /provider /effort /compact /clear /brain /exit`), config `~/.sofuu/config.json`, JS driver
      via the QuickJS bridge. `sofuu repl` is the JS-eval REPL; bare `sofuu` is chat.
- [x] `sofuu.ai` effort/reasoning support (`reasoning_effort` for OpenAI-compat, `thinking.
      budget_tokens` for Anthropic) + `listModels`/`listProviders` APIs.

**D2 — Rust owns the parsers & memory logic (done)**
- [x] CMA memory core (`memory/hnsw.rs`, `memory/cma.rs`) — HNSW, 4-tier decay, k-means
      consolidation, dream/TF-IDF, gravity/resonance.
- [x] SSE parser (`http/sse.rs`) and MCP JSON-RPC (`mcp/jsonrpc.rs`) — safe parsing.
- [x] TS stripper (`ts.rs`), ESM bundler (`bundler.rs`), npm safety core (`npm.rs` — spec
      validation, SHA-1, safe tar extraction rejecting traversal/symlinks).
- [x] 177 assert-based Rust tests; binary still ~2MB.

**D3 — Wire Rust in, delete C (done 2026-08-12)**
- [x] Route the JS-visible APIs to the Rust implementations via FFI: `sofuu.memory` → Rust CMA
      (`sofuu_cma_*` exports; `mod_memory.c` is a thin shell over the QTSQ brain-file I/O;
      brain files stay format-compatible both ways), MCP inbound JSON-RPC → Rust parser
      (server side; client response routing keeps the C scan — see TASKS.md), bundler → Rust
      (`sofuu bundle` is pure Rust), TS strip → Rust (`sofuu_ts_strip_rs`), SSE → Rust
      (`sofuu_sse_*`), npm safety + walk-up resolve → Rust (`sofuu_npm_resolve_rs`).
- [x] Then delete the C duplicates from the cargo build: `memory/hnsw.c`, `memory/dream.c`,
      `ts/stripper.c`, `bundler/bundler.c` are out of the cargo build (they remain for the
      deprecated `make c-only`); `src/mcp/mcp.c` inbound parse + `src/http/sse.c` + `src/npm/*`
      logic are Rust-served in the cargo build.
- [x] End state: C = QuickJS, libuv, SIMD, QTSQ, curl/tar I/O only.

**D4 — Cross-compile + CI + fuzz**
- [x] `build.rs` degrades gracefully without a QTSQ checkout (`SOFUU_QTSQ_DIR`); CI runs
      `cargo test` + `make size-check` on all runners (QTSQ-independent).
- [x] `cargo-fuzz` scaffold (`crates/fuzz/`) with targets for the Rust SSE/JSON-RPC/npm/TS parsers.
- [ ] `build.rs` handles the Zig/musl cross-build (or wrap `scripts/cross/*.sh` around cargo);
      install musl targets; add Rust jobs to `release.yml`.
- [ ] Evaluate `rquickjs`/`boa` (Rust JS engine) once the shell is proven — size/compat tradeoff.

---

## 4. Cross-Cutting: Hardware Portability (all four targets chosen)

- [ ] **Embeddable `libsofuu` (.h/.c)** for iOS / Android in-app inference (the embeddable story).
- [ ] **int8 / int4 SIMD kernels** — today SIMD is f32 only; quantized memory/KV ops need NEON + AVX2
      integer paths for phone/SBC RAM budgets. `src/simd/`. *(Shared with Track C1 — the same kernels
      power both quantized memory ops and the custom engine's matmuls.)*
- [ ] **Build matrix:** macOS arm64/x64, Linux x64/arm64 (musl static), iOS, Android NDK,
      WASM for Workers. Extend the existing Zig cross-compile setup.
- [ ] **Per-tier config:** RAM caps, RAM-page LRU depth, default quant precision per device class.
- [ ] Cold-start budget for Workers: keep startup ~3ms with inference off; lazy-load model.

---

## 5. Milestones (merge points for the two tracks)

| # | Name | Definition of done |
|---|------|--------------------|
| **M1** | Stable core + local inference | `provider:"local"` runs Phi/Qwen GGUF via the `llamacpp` backend behind the vtable; ASAN-clean; real `make test` harness green on all platforms; A1 bugs fixed. |
| **M2** | Fusion alpha | KV export/inject via state API + QTSQ implemented; infinite-context demo on a laptop; round-trip correctness test passing. |
| **M3** | Edge-ready | Embeddable lib; int4/int8 kernels; runs on Pi + phone + Worker; soak-tested; power-loss recovery for QTSQ store. |
| **M4** | 1.0 stable | Fuzzed parsers; documented; benchmarked (long-context cost vs raw llama.cpp; recall quality); GTM assets. |
| **M5** | Custom engine parity | `engine:"sofuu"` generates correct output for one architecture, matching llama.cpp within tolerance (greedy); QTSQ weights + native QTSQ KV. |
| **M6** | Custom engine default | Perf + breadth hold across HW tiers; `engine:"sofuu"` becomes default; fully-proprietary tiny binary; llama.cpp = fallback. |

---

## 6. Top Risks

| Risk | Mitigation |
|------|------------|
| KV bridge fragility | Use `llama_state_seq_get/set_data` (supported) not `kv_self` poking; pin llama.cpp version. |
| QTSQ quant degrades KV quality | Default KV to F8/F16; expose precision knob; correctness test gates every change. |
| Binary-size moat dies with llama.cpp | Strictly behind `SOFUU_LLM=1`; ship two artifacts (tiny / full). |
| C memory bugs at scale | ASAN/UBSAN/valgrind/fuzz in CI from day one (Track A0 is the prerequisite). Rust ports remove most of this surface. |
| Scope (full fusion is large) | Two tracks + milestones; M1 ships value even if M2+ slips. |
| **Custom engine correctness** (subtle numerical bugs) | llama.cpp is the **oracle** — per-layer activation parity harness (C0) gates every kernel; one architecture before breadth. |
| **Custom engine is a long arc** | It rides behind the same vtable as llama.cpp, so it never blocks shipping; promote to default only at M6 when parity+perf proven. |
| **Rust FFI safety / dual-maintenance** | All `unsafe` isolated in `sofuu-ffi`; `make c-only` fallback; migrate subsystem-by-subsystem; size cap in CI. |

---

## 7. Immediate Next Actions (current)

1. **Execute the plan documents (§8)** — the near-term feature/product work is planned in
   detail there. Updated pull order (2026-08-20): PLAN-RUST-MIGRATION M0–M10 ✅ done,
   PLAN-AGENTS A1–A9 ✅ ALL LANDED (A7 landed 2026-08-20 post-H1) incl. F4a, and web search shipped with
   it — PLAN-CHAT-FEATURES F1–F11 ✅ **ALL LANDED** (9 remaining features shipped
   2026-08-20), so the front of the queue is now PLAN-RLM E0 (embedder benchmark) +
   live-model polish, then PLAN-HEADLESS H0–H1 (which unblocks agents' A7 funnel).
2. **D4:** Rust cross-compile (musl via Zig) + CI jobs + `cargo-fuzz` targets for the Rust parsers.
3. **A0:** add ASAN/UBSAN CI job + convert `examples/test_*.js` into a real `tests/` harness
   (also an enabler in PLAN-CHAT-FEATURES before its F4).
4. **B0:** vendor + pin llama.cpp; get `provider:"local"` doing a plain generation (no KV bridge yet).

---

## 8. Active Plan Documents (added 2026-08-12)

Standalone execution plans at the repo root. Each is independently landable; cross-plan
dependencies are documented inside each file.

| Doc | Scope |
|---|---|
| [`PLAN-CHAT-FEATURES.md`](PLAN-CHAT-FEATURES.md) | Chat/brain features F1–F11: `/remember`+`/why`, `/resume`, `@file`, streaming tool-calls + multi-server MCP routing, `/share` brain cards, `/cost`, `/verify`, `/watch`, hooks.js, ghost completion, `serve --brain` — **ALL F1–F11 LANDED 2026-08-20** (177 tests green, 2.0MB ≤ 5MB). F4a/F4b landed 2026-08-16; the 9 remaining features landed 2026-08-20 |
| [`PLAN-RLM.md`](PLAN-RLM.md) | RLM long-context scaffold (nested QuickJS sandbox, no Python), bundled distilled embedder (hashed-MLP ≤1MB, bar = beat TF-IDF), heuristic router + decision log, embedder versioning — **R0–R3 landed 2026-08-13** (33 tests + mock E2E green); E-track + R4 open; **`recurseVia` added 2026-08-16** (sandbox llm() routes through tool-using agents) |
| [`PLAN-HEADLESS.md`](PLAN-HEADLESS.md) | Full headless/embeddable runtime: `libsofuu` ABI (`sofuu-capi` crate, JSON funnel), hostile-host audit (no exit/signals/stdout from library paths), platform packs (iOS/Android/Linux/macOS; Windows/WASM v2), samples (C + Swift + Kotlin), CI gates (headless test + ABI guard + fuzz) — **H0–H6 ALL ✅ DONE** (2026-08-20) |
| [`PLAN-AGENTS.md`](PLAN-AGENTS.md) | Agents + sub-agents: agents-as-tools delegation, budgets/cancel tree, memory scopes, `mapContext` (agents ↔ RLM both directions) — **A1–A9 ALL LANDED + verified** (A7 headless surface landed 2026-08-20: `sofuu_rt_call_stream` + `sofuu_rt_cancel` + per-process agent config + `c_agent.c`; 28-check mock battery ALL PASSED, incl. two real MCP child servers); shipped with **web search** (`sofuu.web.search/open`, keyless DDG default + Brave/Tavily keys, built-in `web_search`/`web_open` agent tools) |
| [`PLAN-RUST-MIGRATION.md`](PLAN-RUST-MIGRATION.md) | M0–M11: migrate the remaining C glue to Rust; end-state C = QuickJS, libuv, SIMD, QTSQ, curl only (Track D completion) |
| [`PLAN-MEMORY-TOKENS.md`](PLAN-MEMORY-TOKENS.md) | Token efficiency + RAM hygiene: minimal byte-stable system prompt (`sofuu.agent.CORE_PROMPT`), recall gating (0.30 floor + 1024-tok budget + dedupe), tool-result truncation (4k head+tail), distilled memory writes, auto-compaction at 70% budget, provider-neutral prefix caching (Anthropic `cache_control` + cache usage in `/cost`), RAM hygiene M1–M6 — **P1–P7 ALL LANDED + AUDITED 2026-08-21** (185 cargo tests, agent/rlm/chat E2E batteries green, 2.0MB ≤ 5MB) |
