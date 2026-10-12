# Sofuu Roadmap — The Edge-AI Operating System

> **North Star:** The complete edge-AI OS in **one self-contained binary** — a local SLM
> with *compressed, infinite, persistent memory*, agents, and MCP — running **unchanged** on a
> phone, a Raspberry Pi, a Cloudflare Worker, or a cloud server.
>
> *Decided 2026-06-18. Strategy: fuse local inference + memory/KV layer; **llama.cpp as the default
> backend AND a from-scratch custom C+QTSQ inference engine** behind one backend interface; build
> stability and features as parallel tracks; target phones, ARM SBCs/laptops, serverless, cloud.*
>
> *Updated 2026-10-12 — **Universal Multi-Format Ingestion, Composable Block Graph & Hardware SIMD/GPU Acceleration (Stages 1–6 Shipped).**
> Verified state: binary 4.5 MB of 5.0 MB cap · 100% green tests across all suites (Rust, C/C++, Metal).
> - **Universal Format Ingestion (Stage 3):** Standalone zero-dependency in-process loader ingests **GGUF**, **SafeTensors**,
>   **PyTorch checkpoints (`.pt` / `.pth` / `pytorch_model.bin`)**, and **NumPy arrays (`.npy` / `.npz`)** directly with
>   zero Python, zero external libraries, and true zero-copy virtual memory mapping.
> - **Composable Block Graph (Stage 4):** Universal dynamic architecture executing any transformer family: LLaMA 2/3/3.1/3.2,
>   Qwen 2/2.5/3, SmolLM/SmolLM2, Mistral (with SWA), Gemma 1/2 (with Gemma-RMSNorm and logit soft-capping), Phi, and Liquid LFM-2.
> - **Apple Silicon Metal Offset Rebasing (Stage 5):** Bypassed Apple Metal driver 256-byte alignment hardware barrier via
>   aligned base offset rebasing and unaligned byte-level memory accessors (`read_u32`, `read_bf16`, `read_f16`) in Metal shaders
>   and Objective-C++ runtime, enabling crash-free GPU execution on arbitrary SafeTensors and PyTorch checkpoints.
> - **ARMv8.2-A `sdot` INT8 Vector Acceleration (Stage 6):** 16-way integer MAC dot products (`vdotq_s32`) with zero-allocation
>   thread-local quantization for Raspberry Pi 5, Apple Silicon, and Android ARM64; accelerated Q8_0 and Q4_K matrix-vector math.
> - **Canonical Plans:** Detailed specifications in `docs/PLAN-INFERENCE-ENGINE.md`, `docs/PLAN-SOFUU-CORE.md`, and `docs/PLAN-EMBED-SDK.md`.*
>
> *Updated 2026-10-11 — **The Edge & Mobile Inference Engine Standard ("vLLM of Edge & Mobile").**
> Verified state: lib 431/0 · JS suite 46/0/1 · capi 36/0 · binary 4.4 MB of 5.0 MB cap · 100% green tests.
> - **In-process native engine shipped:** Complete standalone zero-dependency inference engine (`sofuu infer`,
>   `sofuu.ai.streamLocal()`, `sofuu chat -m <model>`). Supports both GGUF (Q4_0, Q8_0, Q4_K_M, Q6_K)
>   and HuggingFace SafeTensors directories (LFM-2 hybrid depthwise conv1d, GQA, SwiGLU, tied LM head).
> - **Raspberry Pi 5 verified (Stage 1):** Tested live on quad-core Cortex-A76 @ 2.8 GHz (`10.53.147.78`):
>   prefill boosted to **49.3 tk/s** (6.0× speedup), decode steady at **15.6 tk/s**, cold start **38.15 ms**,
>   futex storms eliminated via lock-free sequence barrier (`work_seq_`), hard thread core pinning.
> - **SafeTensors 19×–20× breakthrough (Stage 2):** On macOS M2 Pro: `liqiud-350` (BF16) accelerated from 3.6 tk/s
>   to **68.7 tk/s** (19.1× speedup) via NEON 1-cycle bit-extension (`vshll_n_u16(..., 16)`); `liquid` (MLX 4-bit)
>   accelerated from 0.4 tk/s to **8.1 tk/s** (20.25× speedup). Deadlocks resolved via adaptive yield barrier.
> - **Strategic Edge Charter:** Canonical plans in `docs/PLAN-EDGE-INFERENCE-STANDARD.md` and `docs/PLAN-CPU-INFERENCE-ENGINE.md`.*
>
> *Updated 2026-09-27 — **the headless SDK is installable, and honest.**
> Verified state: lib 395/0 · JS suite 46/0/1 · capi 36/0 · 5/5 headless
> samples · `make abi-check` green · 3.2 MB of a 5 MB cap · `sofuu doctor`
> healthy · Swift + JNI + Package.swift + podspec all validated.*
> - **What changed:** the audit of `libsofuu` — the product per
>   `PLAN-POSITIONING-2026.md` §5 — found H0–H6 had *built* the library while
>   **7 of its 17 documented funnel methods did not exist**, including the
>   whole brain; the flagship sample called a nonexistent method and printed
>   the error as success; `ai.stream` delivered zero events; and the default
>   embedding space silently mismatched the brain's dimension. All fixed and
>   now pinned by tests, including an **anti-drift test** that fails if a
>   method listed in `docs/EMBEDDING.md` does not resolve.
> - **Install paths (new):** `Package.swift` (checksum-pinned, stamped at
>   release), `Sofuu.podspec`, a Gradle AAR (`com.sofuu:sofuu-android`), and
>   an npm package for the CLI. Release CI builds and attaches all four SDK
>   packs and commits the SwiftPM checksum back to the branch. A typed Swift
>   layer (`embed`, a real `Brain`, streaming, cancel) typechecks in CI.
> - **Still the owner's step:** publish. See `RELEASE-0.2.0-PREP.md`.
>
> *Updated 2026-09-25 — **what the release now actually ships**, and what is
> still owed. Verified state: lib 395/0 · JS suite 46/0/1 · capi 24/0 ·
> `make abi-check` green · 3.37 MB of a 5 MB cap · `sofuu doctor` healthy.*
> - **Shipped:** Rust-first runtime · encrypted persistent brain (QTSQ) ·
>   offline embeddings with **no model file** (0 B, ~2 µs/doc) · image
>   embeddings joint with text · speech in/out (provider *and* on-device
>   OS engines) · 5 context-economy ML gates, each now proven to beat both
>   a constant and logistic regression on held-out families · agent loop +
>   sub-agents · MCP client/server · RLM · headless C ABI + Swift/Kotlin
>   SDK · `sofuu doctor` · context-window auto-detection.
> - **The honest gap:** the ML gates are trained on synthetic data with
>   mechanical labels, and the shadow-mode A/B on live sessions — the only
>   instrument that would prove they help real users — does not exist yet.
>   Read the gate numbers as "learnable on our generated distribution", not
>   "correct in production". Everything else open is enumerated in
>   `TASKS.md` §"Open items, consolidated".
> - **Stale artifacts, know before shipping:** every platform tarball in
>   `dist/` predates 2026-09-25 (the Linux packs are from April). Only the
>   public macOS download was rebuilt this session.
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
  (`crates/sofuu-core/`), with 395 tests and the binary at 3.2MB.
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

> The proprietary endgame: a tiny, self-contained Rust+C+SIMD transformer runtime that runs in-process with zero external dependencies.
> llama.cpp is the **correctness oracle**: every op is validated against its output on the same model/prompt.
>
> **→ Full detailed plans: [`PLAN-INFERENCE-ENGINE.md`](PLAN-INFERENCE-ENGINE.md)** and **[`ROADMAP-CUSTOM-ENGINE.md`](ROADMAP-CUSTOM-ENGINE.md)**.

**C0 — Foundations & oracle harness (done)**
- [x] Numeric parity harness: verified per-layer activations against reference outputs.
- [x] Llama/Qwen/Liquid-style decoder: RMSNorm, Gemma-RMSNorm, RoPE, SwiGLU, GeGLU, GQA, tied LM head.
- [x] In-process tensor core: pure Rust and high-performance SIMD/Metal compute microkernels.

**C1 — Quantized & floating-point microkernels (done)**
- [x] int8 (`sdot` / `sofuu_gemv_q8_0`) and int4 (`sofuu_gemv_q4_k`) microkernels: ARM NEON + ARMv8.2-A dotprod + AVX2.
- [x] BF16 1-cycle bit-extension synthesis (`sofuu_gemv_bf16`) and MLX 4-bit affine GEMV (`sofuu_gemv_mlx4`).
- [x] Zero-copy tensor loader: load model tensors directly from GGUF, SafeTensors, PyTorch (`.pt`/`.pth`), NumPy (`.npy`/`.npz`).

**C2 — End-to-end generation & Universal Formats (done)**
- [x] Full forward pass + bounded min-heap Top-K sampling (<35 µs); greedy, top-p, temperature, repetition penalty.
- [x] In-place KV compaction, StreamingLLM attention sinks, and mathematically exact sliding-window RoPE rotation.
- [x] Native multi-threaded batched GEMM prefill with monotonic lock-free sequence barrier.

**C3 — Breadth & Architectural Generalization (done)**
- [x] Composable Block Graph dynamically executing LLaMA, Qwen, Gemma 1/2, Mistral, SmolLM, Phi, Liquid LFM-2, and MoE.
- [x] Apple Silicon Metal GPU unified compute with 256-byte offset rebasing.
- [x] Stripped release binary strictly $\le 5.0\text{ MB}$ (currently 4.5 MB).

**C4 — Heterogeneous Co-Processing & Swarms (Next: Stage 7)**
- [ ] GPU dense compute + CPU dynamic RoPE/sampling co-processing.
- [ ] Host RAM KV cache offloading for small VRAM GPUs.
- [ ] Distributed sequential layer-wise activation pipelining over LAN.

---

## 3b. Track D — Rust-First Migration (the runtime shell)

> *Decided 2026-08-06. The interactive shell and all security-sensitive parsing move to Rust;
> C stays only for extreme low-level work (QuickJS, libuv, SIMD, QTSQ). Binary size stays ≤ 5MB
> (currently 3.2MB measured). Full plan: `PLAN-RUST-MIGRATION.md`.*

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
- [x] Rust test suite; binary within the 5MB cap (3.2MB measured 2026-09-27).

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

## 3c. Track E — The Edge & Mobile Inference Engine Standard ("vLLM of Edge & Mobile")

> *Charter formulated 2026-10-11. Sofuu does not aim to be another generic multi-cloud engine.
> Sofuu is purpose-built to become the global standard for Edge, Phone, Laptop (Mac-first), and
> Serverless AI inference. Zero external dependencies, $\le 5.0\text{ MB}$ stripped binary, instant
> <38 ms cold start, and 0.0% idle resource consumption. Canonical plans: `docs/PLAN-EDGE-INFERENCE-STANDARD.md`
> and `docs/PLAN-CPU-INFERENCE-ENGINE.md`.*

### Stage 1: ARM64 Edge CPU Specialization (Raspberry Pi 5 / Cortex-A76) — ✅ SHIPPED & VERIFIED
- [x] Replaced Linux futex syscall storms (~400 futex syscalls/token) with a lock-free monotonic
      sequence barrier (`work_seq_`), cacheline-isolated completion counters (`alignas(64)`), and
      hardware spin pauses (`yield` on ARM, `pause` on x86).
- [x] LPDDR4X 3-cacheline prefetching (`+0, +64, +128`) hiding ~80ns memory latency on Broadcom BCM2712.
- [x] Hard thread-to-core pinning (`pthread_setaffinity_np`) on Linux ARM64 eliminating OS scheduler migrations.
- [x] Verified directly on Raspberry Pi 5 hardware (`10.53.147.78`):
      - Cold load time: **38.15 ms** (was 105.02 ms, **2.75× faster**).
      - Prompt prefill: **49.3–49.8 tk/s** (was 8.2 tk/s, **6.0× faster**).
      - Autoregressive decode: **15.6 tk/s** sustained with zero futex storms.
- [x] Stripped static musl binary `dist/sofuu-linux-arm64` deployed and operating live on Pi 5.

### Stage 2: Universal Multi-Format Architecture & Liquid Foundation Models — ✅ SHIPPED & VERIFIED
- [x] Auto-detection supporting HuggingFace SafeTensors directories (`config.json` + `tokenizer.json` +
      `model.safetensors`) alongside standard `.gguf` files with uniform `InferModel` abstraction.
- [x] Complete Liquid Foundation Model (LFM-2) engine: causal gated 1D short convolutions ($K=3$),
      GQA multi-head attention with head RMSNorm, SwiGLU FFN, and tied linear LM head projection.
- [x] Safe unaligned memory accessors (`read_u16_unaligned`, `read_u32_unaligned`) solving SafeTensors
      odd JSON header offset barrier without undefined behavior.
- [x] Multi-core C++ NEON vector microkernels (`sofuu_gemv_bf16` and `sofuu_gemv_mlx4`):
      - BF16: 1-cycle bit-extension via `vshll_n_u16(..., 16)` and 4-way parallel FMA (`vfmaq_f32`).
      - MLX 4-bit affine: Group scale/bias broadcasting and fused multiply-add.
- [x] Breakthrough verified on macOS M2 Pro:
      - `liqiud-350` (BF16): Accelerated from 3.6 tk/s to **68.7 tk/s** (**19.1× speedup**).
      - `liquid` (MLX 4-bit): Accelerated from 0.4 tk/s to **8.1 tk/s** (**20.25× speedup**).
- [x] Fixed worker sleeping race in `WorkerThreadPool` via adaptive yield notification loops.

### Stage 3: Universal Tensor Registry (`.pt` PyTorch & `.npy`/`.npz` NumPy) — ✅ SHIPPED & VERIFIED
- [x] Pure-Rust zero-dependency ZIP and ZIP64 archive reader (`zip_reader.rs`) for PyTorch `.pt` / `.pth` and NumPy `.npz`.
- [x] NumPy `.npy` binary header parser (`\x93NUMPY`) supporting v1.0 and v2.0 formats with zero-copy mmap (`npy.rs`).
- [x] Lightweight pure-Rust mini-pickle interpreter for PyTorch `state_dict()` checkpoint files (`pytorch.rs`).
- [x] Unified all tensor loaders into `UnifiedTensorTable` emitting standardized `TensorView`s across SafeTensors, GGUF, PyTorch, and NumPy (`tensor_table.rs`).
- [x] Full integration test suite verified: `test_universal_formats.rs` (100% green).

### Stage 4: Composable Block Graph (LLaMA, Qwen, SmolLM, Gemma, Phi) — ✅ SHIPPED & VERIFIED
- [x] Generalized block pipeline in `block_graph.rs`: Pre-Norm, Sequence Mixer, Post-Attention Norm, Channel Mixer, Post-FFN Norm.
- [x] Gemma-RMSNorm ($x \cdot (1+w) / \text{RMS}(x)$) and LayerNorm primitives.
- [x] Attention and Final Logit Soft-Capping ($\text{cap} \cdot \tanh(\text{logits}/\text{cap})$).
- [x] GeGLU activation ($\text{gelu}(\text{gate}) \cdot \text{up}$) and Top-K Sparse MoE routing contracts.
- [x] Verified mathematical correctness with unit test suite: `test_gemma_rmsnorm`, `test_soft_cap_in_place`, `test_geglu_in_place`, `test_weight_matrix_gemv_f32`.

### Stage 5: Apple Silicon Metal Unified GPU Engine & Offset Rebasing — ✅ SHIPPED & VERIFIED
- [x] Metal 256-byte alignment re-basing: Pass `base = offset & ~255ULL` and `rem = offset & 255ULL`
      to MSL shaders, enabling Metal GPU execution on any SafeTensors or PyTorch model regardless of header length.
- [x] Layer command encoder batching: Encoded Q, K, V attention projections and SwiGLU Gate/Up projections into batched command buffers with 256-byte hardware-aligned slots.
- [x] Bypasses Apple Metal driver alignment restrictions on arbitrary SafeTensors and PyTorch checkpoints with zero runtime errors.
- [x] Validated via `test_metal_engine.rs` (100% green).
- [x] Binary size strictly preserved: **4.44 MB** stripped ($\le 5.0\text{ MB}$ gate).

### Stage 6: ARMv8.2-A `sdot` INT8 Vector Acceleration (Pi 5 & Android) — ✅ SHIPPED & VERIFIED
- [x] Zero-allocation thread-local `sofuu_q8_block_ext` INT8 activation vector quantization on calling thread.
- [x] Hardware `sdot` (`vdotq_s32`) inner loop on Cortex-A76 and Apple Silicon (`-march=armv8.2-a+dotprod` automated compiler detection).
- [x] 16-way integer MAC hardware instruction acceleration for `Q8_0` and `Q4_K` GEMV kernels.
- [x] Full integration into `BlockGraphModel` and `InferEngine` multi-format loader.
- [x] Validated via `test_universal_formats.rs` and `quant.rs` test suites (100% green).
- [x] Stripped release binary size strictly preserved: **4.5 MB** ($\le 5.0\text{ MB}$ gate).

### Stage 7: Heterogeneous CPU+GPU Co-Processing, Swarms & Serverless Endpoint — ✅ SHIPPED & VERIFIED
- [x] Workload specialization: Metal GPU executes heavy dense GEMV/GEMM math with unified layer-wise command buffer batching (`sofuu_metal_gemv_qkv_multi` and `sofuu_metal_gemv_gate_up_multi` supporting BF16, F16, and F32 with 256-byte aligned output slots), eliminating 66% of command buffer dispatches. CPU handles bounded Top-K sampling (<35 µs), dynamic RoPE rotations, and circular convolution state buffers.
- [x] True Asynchronous CPU-GPU Pipelining: Implemented non-blocking GPU dispatches (`sofuu_metal_gemv_async` with retained command buffer lifecycles and `sofuu_metal_sync`), overlapping CPU attention/residual/norm operations concurrently with GPU projection compute.
- [x] Zero-Allocation Hot Decoding Loop: Pre-allocated `buf_scores` and `buf_tmp` across `BlockGraphModel`, eliminating over 392 heap allocations per generated token in the inner decoding loop.
- [x] Configurable Heterogeneous Layer Partitioning: Added `BlockGraphModel::set_gpu_layers(N)` allowing arbitrary layer-wise compute partitioning between Apple Silicon GPU and the multi-threaded CPU SIMD thread pool.
- [x] Host RAM KV Cache Offloading: `BlockKvCache` stores key-value matrices in Host RAM (`Vec<f32>`), keeping GPU VRAM dedicated to weights and active projections, allowing edge devices with 2GB–4GB VRAM to run large models without OOM crashes.
- [x] Sequential layer-wise activation pipelining (`forward_layer_range` and `execute_layers` in `BlockGraphModel`): Passes compact hidden states across edge swarm nodes over LAN, enabling collaborative distributed execution of large models.
- [x] Built-in serverless OpenAI-compatible inference endpoint (`sofuu serve --infer` / `-m <model_path>`): Serves streaming SSE (`/v1/chat/completions` and `/v1/completions`) and JSON endpoints (`/v1/models`, `/infer`) with Bearer token authentication alongside Sofuu brain memory endpoints.

---

## 4. Cross-Cutting: Hardware Portability (all targets chosen)

- [x] **Embeddable `libsofuu` (.h/.c)** for iOS / Android in-app inference (C ABI, Swift/Kotlin wrappers).
- [x] **Multi-format in-process inference engine** (GGUF + SafeTensors LFM-2) in $\le 5.0\text{ MB}$ binary.
- [x] **ARMv8.2-A DotProd `sdot` kernels** for Raspberry Pi 5 and Android NDK ARM64.
- [x] **Build matrix:** macOS arm64/x64, Linux x64/arm64 (musl static), iOS, Android NDK.
- [x] **Per-tier config:** Core topology detection, lock-free monotonic barrier, 0.0% idle parking.

---

## 5. Milestones (The Unified Edge & Mobile Engine Roadmap)

| # | Milestone | Scope & Definition of Done | Status |
|---|---|---|---|
| **M1** | Headless SDK & Stable Core | `libsofuu` ABI, Swift + Kotlin SDKs, encrypted QTSQ brain round-trip, `sofuu doctor` healthy, 4.4 MB binary. | ✅ Shipped |
| **M2** | In-Process Multi-Format Engine | Native GGUF + SafeTensors LFM-2, Pi 5 ARM64 (49.3 tk/s prefill, 15.6 tk/s decode), Mac M2 Pro (68.7 tk/s BF16, 8.1 tk/s MLX4, 150 tk/s GGUF), KV compaction + SWA. | ✅ Shipped |
| **M3** | Apple Silicon Metal GPU Engine | MSL shaders with 256-byte offset re-basing, command encoder batching across all layers, unified memory zero-copy. | 🟡 In Progress |
| **M4** | ARMv8.2-A `sdot` Vector SIMD | On-the-fly INT8 activation quantization + `vdotq_s32` on Cortex-A76 (Pi 5 & Android), targeting $\ge 25\text{–}30\text{ tk/s}$ decode. | ⬜ Next |
| **M5** | Heterogeneous CPU+GPU Co-Processing | Asymmetric pipeline: GPU executes dense GEMV, CPU executes <35µs Top-K and RoPE; Host RAM KV cache offloading for 2GB–4GB GPUs. | ⬜ Planned |
| **M6** | Distributed Edge Swarms & Micro-Hosting | 4 KB activation pipelining over LAN across Pi 5 clusters and mixed PCs for 14B–32B models; `sofuu serve --infer` for edge monetization. | ⬜ Planned |

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
| [`PLAN-HEADLESS.md`](PLAN-HEADLESS.md) | Full headless/embeddable runtime: `libsofuu` ABI (`sofuu-capi` crate, JSON funnel), hostile-host audit (no exit/signals/stdout from library paths), platform packs (iOS/Android/Linux/macOS; Windows/WASM v2), samples (C + Swift + Kotlin), CI gates (headless test + ABI guard + fuzz) — **H0–H6 ALL ✅ DONE** (2026-08-20); **SDK made true + installable 2026-09-27** (memory/streaming/mcp reachable from C, asserting samples, `Package.swift` + podspec + Gradle AAR + npm, anti-drift doc test) |
| [`PLAN-POSITIONING-2026.md`](PLAN-POSITIONING-2026.md) | Market/positioning research record (2026-09-09). §5 names `libsofuu` "the product"; §9 sequences the iOS beachhead; §12 lists the owner's open decisions |
| [`PLAN-AGENTS.md`](PLAN-AGENTS.md) | Agents + sub-agents: agents-as-tools delegation, budgets/cancel tree, memory scopes, `mapContext` (agents ↔ RLM both directions) — **A1–A9 ALL LANDED + verified** (A7 headless surface landed 2026-08-20: `sofuu_rt_call_stream` + `sofuu_rt_cancel` + per-process agent config + `c_agent.c`; 28-check mock battery ALL PASSED, incl. two real MCP child servers); shipped with **web search** (`sofuu.web.search/open`, keyless DDG default + Brave/Tavily keys, built-in `web_search`/`web_open` agent tools) |
| [`PLAN-RUST-MIGRATION.md`](PLAN-RUST-MIGRATION.md) | M0–M11: migrate the remaining C glue to Rust; end-state C = QuickJS, libuv, SIMD, QTSQ, curl only (Track D completion) |
| [`PLAN-MEMORY-TOKENS.md`](PLAN-MEMORY-TOKENS.md) | Token efficiency + RAM hygiene: minimal byte-stable system prompt (`sofuu.agent.CORE_PROMPT`), recall gating (0.30 floor + 1024-tok budget + dedupe), tool-result truncation (4k head+tail), distilled memory writes, auto-compaction at 70% budget, provider-neutral prefix caching (Anthropic `cache_control` + cache usage in `/cost`), RAM hygiene M1–M6 — **P1–P7 ALL LANDED + AUDITED 2026-08-21** (185 cargo tests, agent/rlm/chat E2E batteries green, 2.0MB ≤ 5MB) |
| [`PLAN-SOFUU-CORE.md`](PLAN-SOFUU-CORE.md) | **The Main Sofuu Core Runtime & Cognitive OS**: QuickJS (ES2023, zero JIT), libuv async loop (~3 ms boot), TypeScript stripper & bundler, 4-tier Cognitive Memory Architecture (CMA), QTSQ encrypted persistent brain, autonomous agent loop, native MCP client/server, and context-economy ML gates. |
| [`PLAN-INFERENCE-ENGINE.md`](PLAN-INFERENCE-ENGINE.md) | **The In-Process Inference Engine ("vLLM of Edge & Mobile")**: Hardware-tuned SIMD microkernels (BF16 bit-extension, MLX 4-bit affine dual FP16/BF16), universal multi-format execution (GGUF + SafeTensors LFM-2 + Gemma 2B), Apple Metal GPU, ARM64 Cortex-A76 specialization, batched GEMM prefill, bounded min-heap Top-K (<35 µs), SWA & exact RoPE shift, Stages 1 through 8 (13.4 tk/s on Gemma 2B, 68+ tk/s on BF16). |
| [`PLAN-EMBED-SDK.md`](PLAN-EMBED-SDK.md) | **The Embed SDK for Mobile & Edge Applications (`libsofuu`)**: Universal C ABI v2, native iOS Swift/SwiftUI SDK, Android Kotlin/Compose SDK, host-native tool callback bridge (letting in-app agents call native Camera, Location, HealthKit, SQLite), Jetsam memory protection ($\le 30\text{ MB}$ RSS), and App Store zero-JIT compliance by design. |
| [`PLAN-EDGE-INFERENCE-STANDARD.md`](PLAN-EDGE-INFERENCE-STANDARD.md) | **The Edge & Mobile Inference Engine Standard**: Canonical charter, heterogeneous CPU+GPU co-processing, distributed edge swarms (4 KB LAN pipelining across cheap SBCs/mixed PCs), and edge micro-hosting (`sofuu serve --infer`). |
| [`PLAN-CPU-INFERENCE-ENGINE.md`](PLAN-CPU-INFERENCE-ENGINE.md) | **In-Process Inference Engine Architecture**: C++ SIMD microkernels, lock-free monotonic barrier, and zero-futex synchronization. |
