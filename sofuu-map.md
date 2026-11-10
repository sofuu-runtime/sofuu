# SOFUU ARCHITECTURAL MAP & DIVISION REGISTRY

> Master blueprint for compartmentalization, ownership, and cross-subsystem isolation across Sofuu (`v0.3.0-beta`).
> This document is the single source of truth (SSOT) for division boundaries, owned files, testing harnesses, and implementation roadmaps.

---

## 1. Global Invariants & Non-Negotiable Rules

Every division operates under strict architectural boundaries to guarantee that changes in one domain never cascade or corrupt others:

1. **Binary Size Ceiling**: The primary release binary (`./sofuu`) must remain $\le 5,767,168\text{ bytes}$ ($5.5\text{ MiB}$). Every division adding code to `sofuu-core` must pass `make size-check`.
2. **Core Compiler Generalization**: The compiler (`crates/sofuu-core/src/compiler.rs`) must remain 100% domain-agnostic. Never hardcode domain parameters (`entry_price`, `heart_rate`, `leverage`) or external APIs. All configuration enters via 12-factor hierarchy (CLI flags > ENV > Config file > Routine signature).
3. **No Auxiliary Wrappers**: Standalone agent binaries are 100% pure Kaze (`agent.kaze`). Zero auxiliary `.js` sidecars or `.sh` script wrappers.
4. **Local Models Only**: Test and benchmark harnesses must run exclusively against local GGUFs already on disk. Never initiate remote model downloads.
5. **Fail-Closed Security**: Every capability check (network, filesystem, hardware, memory) must fail closed (`E_CAP_DENIED` / reject) on missing, malformed, or empty policies.
6. **Beta Positioning**: Sofuu is in active beta (`v0.3.0-beta`). Features must be described accurately as Core or Experimental Preview.
7. **Zero Decorative Emojis**: Documentation, CLI banners, log outputs, and web interfaces must remain strictly professional without decorative emojis.

---

## 2. Division Index & Matrix

| Division | Name | Focus & Scope | Primary Locations | Maturity | Verification Command |
|---|---|---|---|---|---|
| **D01** | **Core Substrate & Event Loop** | QuickJS, libuv, process lifecycle, C-API, memory allocator | `crates/sofuu-ffi/`, `crates/sofuu-capi/`, `crates/sofuu-core/src/rt/` | Core | `cargo test -p sofuu-ffi -p sofuu-core --lib rt` |
| **D02** | **Cognitive Vector Memory** | In-process QTSQ engine, SIMD cosine recall, CMA memory, deterministic embedder | `crates/sofuu-core/src/domains/memory/`, `crates/sofuu-core/src/embedding/` | Core | `cargo test -p sofuu-core --lib domains::memory` |
| **D03** | **Autonomous AI Agents** | Turn-block context retention, sub-agent delegation, reflection, agent routines | `src/js/agent.js`, `crates/sofuu-core/src/chat.rs` | Core | `make bc-check` |
| **D04** | **Tool Belt & Protocol Gates** | Sandboxed in-process tools, SSRF firewall, native MCP client & host | `crates/sofuu-core/src/mcp/`, `crates/sofuu-core/src/rt/policy.rs` | Core | `cargo test -p sofuu-core --lib mcp` |
| **D05** | **Kaze Language & Compiler** | Declarative `.kaze` AST, linear arena VM, capability gates, supervisor daemon | `kaze/`, `crates/sofuu-core/src/compiler.rs`, `example-agents/` | Core | `cargo test --manifest-path kaze/Cargo.toml` |
| **D06** | **Hardware Inference Engine** | GGUF/SafeTensors loaders, Metal GPU shaders, NEON/AVX2 kernels, quantization | `crates/sofuu-core/src/domains/infer/`, `crates/sofuu-core/src/ml/` | Experimental Preview | `cargo test -p sofuu-core --lib domains::infer` |
| **D07** | **Robotics, Drones & Edge** | MAVLink 2.0 protocol, GPIO pulse timers, physical safety gates, telemetry | `crates/sofuu-core/src/domains/robotics/`, `kaze/src/hardware.rs` | Core | `cargo test -p sofuu-core --lib domains::robotics` |
| **D08** | **CLI, Interactive TUI & REPL** | `sofuu` CLI flags, full-screen `chat` TUI, interactive REPL, inspector | `src/js/chat_tui.js`, `crates/sofuu-core/src/main.rs` | Core | `./sofuu doctor` |
| **D09** | **Mobile & Native Embeddings** | Android C-API/APK, Swift Package Manager (iOS/macOS), CocoaPods | `sofuu-android/`, `sofuu-ios/`, `crates/sofuu-capi/` | Core / Edge | `cargo test -p sofuu-capi` |
| **D10** | **Desktop App Presentation** | Tauri GUI shell, isolated GTK/WebKit presentation layer | `sofuu-desktop/` | Experimental Preview | `cd sofuu-desktop && npm run build` |
| **D11** | **Public Surface & Operations** | Documentation, landing page (`sofuu.xyz`), npm distribution, Cloudflare Pages | `sofuu-landing/`, `npm/`, `docs/`, `README.md` | Public Operations | `cd sofuu-landing && npm run build` |

---

## 3. Detailed Division Specifications

### Division 01: Core Substrate & Event Loop (D01)
* **Mission**: Provide the deterministic, low-latency foundation for all JavaScript, TypeScript, and native execution.
* **Owned Files**:
  * `crates/sofuu-ffi/` (QuickJS C bindings, libuv integration, build scripts)
  * `crates/sofuu-capi/` (Exported C ABI for foreign runtimes)
  * `crates/sofuu-core/src/rt/` (Engine bootstrap, microtask queues, timer loops, worker threads)
* **Architectural Boundaries**:
  * Must never depend on high-level agent routines or domain-specific logic.
  * Subsystem initialization must fail fast on error (`init_subsystems(ctx).expect(...)`).
  * Binary size impact must be monitored on every dependency addition.
* **Active Tasks**:
  - [x] Integrate CMake fallback for vendored libuv builds on clean clones (F-15).
  - [x] Enforce fail-fast runtime panic on subsystem initialization failure (F-06).
  - [x] Optimize QuickJS microtask queue drain latency under saturated event loops.
  - [x] PLAN-UNIFIED-DIAGNOSTICS: versioned diagnostic contract, uncaught/rejection rendering, and shared event-string redaction (`rt/diag_contract.rs`).

---

### Division 02: Cognitive Vector Memory Subsystem (D02)
* **Mission**: Deliver microsecond-level in-process vector storage and associative memory without running external database daemons.
* **Owned Files**:
  * `crates/sofuu-core/src/domains/memory/` (QTSQ memory substrate, CMA allocator, memory index)
  * `crates/sofuu-core/src/embedding/` (Local deterministic 768-dim embedder, embedding cache)
  * `crates/sofuu-core/src/brain_cmd.rs` (Memory CLI operations: export, recall, vacuum)
* **Architectural Boundaries**:
  * Zero external database servers required (no Milvus, Chroma, Pinecone, or Redis).
  * Offline embeddings (`sofuu.ai.embedLocal`) must remain deterministic, zero-model-weights, and microsecond fast.
  * In-process vector recall must execute with SIMD acceleration (ARM NEON / x86_64 AVX2).
* **Active Tasks**:
  - [x] Standardize offline deterministic 768-dimension embedding API.
  - [x] Implement memory compaction and retention decay curve algorithms.
  - [ ] Add zero-copy memory-mapped QTSQ persistence validation for multi-process concurrency.

---

### Division 03: Autonomous AI Agents & Sub-Agents (D03)
* **Mission**: Provide lightweight, multi-turn autonomous agent loops with turn-block context retention and hierarchical delegation.
* **Owned Files**:
  * `src/js/agent.js` and snapshots `src/js/agent.bc`, `src/js/agent.bc.gz`
  * `crates/sofuu-core/src/agent_main.rs` (Agent runtime entry point)
  * `crates/sofuu-core/src/chat.rs` (Agent context retention and session state)
* **Architectural Boundaries**:
  * Bytecode snapshots must stay strictly in sync with driver JS (`make bc-check`).
  * Sub-agent execution must remain sandboxed with scoped memory access.
  * Context retention must prioritize semantic turn-blocks over unbounded raw history.
* **Active Tasks**:
  - [x] Purge outdated branding comments in favor of "turn-block context retention" (F-08).
  - [x] Recompile and lock QuickJS bytecode snapshots (11/11 match verified).
  - [ ] Expand sub-agent result synthesis protocol for parallel delegation trees.

---

### Division 04: Native Tool Belt & Protocol Gates (D04)
* **Mission**: Host in-process execution tools and native Model Context Protocol (MCP) clients and servers behind strict security firewalls.
* **Owned Files**:
  * `crates/sofuu-core/src/mcp/` (Native MCP protocol implementation, JSON-RPC transport)
  * `crates/sofuu-core/src/rt/policy.rs` (SSRF firewall, jailroot enforcer, capability arbiter)
  * `crates/sofuu-core/src/http/` (Internal HTTP server and client with connection pooling)
* **Architectural Boundaries**:
  * SSRF checks must deny private/loopback subnets unless explicitly permitted by policy.
  * Native tools must never bypass the policy evaluator (`evaluate_policy`).
  * Process execution tools must enforce strict timeout limits and resource quotas.
* **Active Tasks**:
  - [x] Fix SSRF private subnet check ordering when loopback testing mode is active (F-14).
  - [x] Eliminate unchecked binary payload decodes in tool communication (F-05).
  - [ ] Add streaming MCP tool response pipes for large file operations.

---

### Division 05: Kaze Language, VM & Standalone Compiler (D05)
* **Mission**: Pure declarative agent routine language and linear arena VM for mission-critical, zero-GC edge intelligence.
* **Owned Files**:
  * `kaze/` (Standalone parser, AST, bytecompiler, arena VM, CLI)
  * `crates/sofuu-core/src/compiler.rs` (Runtime standalone binary compiler)
  * `crates/sofuu-core/src/domains/kaze/` (In-process Kaze runtime bridge)
  * `example-agents/` (`quant_sentinel/`, `med_protocol/`, `agri_sentinel/`, `paper_trader/`)
* **Architectural Boundaries**:
  * Compiler must remain 100% domain-agnostic; zero hardcoding of parameter names or external endpoints.
  * Standalone compiled binaries must support production 12-factor configuration resolution.
  * Agents must run on pure Kaze with zero auxiliary `.js` or `.sh` wrapper scripts.
  * Missing parameters must trigger structured diagnostic guidance, not runtime crashes.
* **Active Tasks**:
  - [x] Enforce fail-closed capability gates on missing or empty policies (`E_CAP_DENIED`) (F-04).
  - [x] Build and verify 4 showcase agents in `example-agents/` with automated build harness.
  - [x] PLAN-UNIFIED-DIAGNOSTICS: verified Kaze parser/checker/VM locations and an additive versioned `diagnostic` in `script run --json` (excerpts deferred under the size cap).
  - [ ] Implement incremental AST caching for sub-millisecond Kaze routine compilation.

---

### Division 06: Hardware Inference Engine (D06 - Experimental Preview)
* **Mission**: In-process neural network inference on CPU and Apple Silicon Metal shaders for local fallback and edge nodes.
* **Owned Files**:
  * `crates/sofuu-core/src/domains/infer/` (GGUF parser, SafeTensors loader, tensor operators)
  * `crates/sofuu-core/src/ml/` (Quantization kernels, NEON/AVX2 math, Metal shader bindings)
* **Architectural Boundaries**:
  * Designated strictly as **[Experimental - Preview]**. Heavy production workloads route to `llamacpp` plugin or remote APIs.
  * Zero remote model downloads allowed during tests or benchmarks. Use on-disk weights only.
  * Decoupled from the primary release binary size ceiling via conditional compilation flags where necessary.
* **Active Tasks**:
  - [x] Maintain parity sweep receipts comparing in-process engine against llama.cpp.
  - [x] Retain experimental preview positioning across all documentation and landing pages.
  - [ ] Benchmark KV-cache memory pressure on Apple Silicon Unified Memory architectures.

---

### Division 07: Robotics, Drones & Edge Subsystems (D07)
* **Mission**: Hardware interfacing, GPIO telemetry, and MAVLink 2.0 flight protocol integration for autonomous physical systems.
* **Owned Files**:
  * `crates/sofuu-core/src/domains/robotics/` (MAVLink protocol encoder/decoder, flight safety gates)
  * `kaze/src/hardware.rs` & `kaze/src/gpio.rs` (Hardware policy enforcement and pulse timer pins)
  * `crates/sofuu-core/src/drone_cmd.rs` (Robotics CLI dispatch commands)
* **Architectural Boundaries**:
  * Hardware actuations (motor throttle, servo PWM, relay switches) must fail closed on heartbeat timeout.
  * GPIO de-assert failures must log structured errors to prevent stuck actuators (F-09).
  * Geofence policy breaches must trigger immediate automatic return-to-launch (RTL) rejection.
* **Active Tasks**:
  - [x] Add explicit error logging for pulse timer pin de-assert failures (F-09).
  - [x] Formalize MAVLink hardware policy gates with fail-closed verdicts.
  - [ ] Implement simulation mock harness for SITL (Software In The Loop) automated regression tests.

---

### Division 08: CLI, Interactive TUI & Developer Experience (D08)
* **Mission**: Provide intuitive, zero-configuration developer interfaces from one-line scripts to interactive terminal chat.
* **Owned Files**:
  * `src/js/chat_tui.js` and snapshots `src/js/chat.bc`, `src/js/chat.bc.gz`
  * `crates/sofuu-core/src/main.rs` (CLI command dispatcher and flag parsing)
  * `crates/sofuu-core/src/doctor.rs` (System diagnostic verification command)
* **Architectural Boundaries**:
  * Brand voice must describe Sofuu as a "runtime for AI", avoiding legacy phrases (F-07).
  * Zero decorative emojis in banners, progress bars, or error messages.
  * Fast path execution (`sofuu eval`, `sofuu run`) must preserve sub-25ms cold start.
* **Active Tasks**:
  - [x] Align TUI banner with brand voice ("runtime for AI") (F-07).
  - [x] Verify CLI argument parsing matches 12-factor configuration hierarchy.
  - [x] PLAN-UNIFIED-DIAGNOSTICS: concise/full diagnostic rendering on stderr, `SOFUU_DIAGNOSTICS=full`, and unchanged exit status/argument passthrough.
  - [ ] Add interactive debugging hooks (`--inspect-brk`) for Kaze VM step execution.

---

### Division 09: Mobile & Native Embeddings (D09)
* **Mission**: Distribute the embeddable Sofuu core runtime to mobile ecosystems (Android APK/AAR, iOS/macOS frameworks).
* **Owned Files**:
  * `sofuu-android/` (Gradle project, JNI bindings, sample Android application)
  * `sofuu-ios/` (Xcode project, Swift Package Manager manifest, CocoaPods spec)
  * `bindings/android/` & `dist/` (Cross-compiled static archives for mobile targets)
* **Architectural Boundaries**:
  * Mobile runtime must compile through the clean C-ABI (`sofuu-capi`).
  * Android builds must target both `arm64-v8a` and `x86_64`.
  * iOS builds must compile clean for both physical devices (`aarch64-apple-ios`) and simulators (`aarch64-apple-ios-sim`).
* **Active Tasks**:
  - [x] Package Android standalone universal APK and arm64-v8a APK.
  - [x] Maintain Swift Package Manager (`Package.swift`) distribution artifacts.
  - [ ] Add automated Gradle build verification to continuous integration pipeline.

---

### Division 10: Desktop App Presentation (D10 - Experimental Preview)
* **Mission**: Visual presentation shell and local GUI dashboard built with Tauri.
* **Owned Files**:
  * `sofuu-desktop/` (Tauri configuration, Vite frontend, GTK presentation layer)
* **Architectural Boundaries**:
  * Excluded from default workspace members (`default-members` in root `Cargo.toml`).
  * System LGPL-2.1+ dependencies (GTK, WebKitGTK) are isolated to this presentation shell (F-11).
  * The desktop application communicates exclusively via standard CLI invocation or local socket IPC.
* **Active Tasks**:
  - [x] Formally document Tauri LGPL-2.1+ GTK linkage isolation in RULES.md (F-11).
  - [ ] Complete cross-platform desktop installer packaging (macOS `.dmg`, Linux `.deb`).

---

### Division 11: Public Surface, Documentation & Operations (D11)
* **Mission**: Maintain the public website, official documentation, npm package, and release delivery pipeline.
* **Owned Files**:
  * `sofuu-landing/` (Next.js public site at `sofuu.xyz`, Cloudflare Pages deployment)
  * `npm/` (Official `sofuu` CLI distribution package on the npm registry)
  * `docs/` (Architecture blueprints, audit reports, remediation plans)
  * `README.md` (Repository documentation and quickstart guides)
* **Architectural Boundaries**:
  * Hero section must frankly communicate active beta status (`v0.3.0-beta`) with support email `hello@sofuu.xyz`.
  * Transparently communicate active beta status (`v0.3.0-beta`) without claiming finished status.
  * Zero competitor brand mentions or disparaging comparisons (F-12).
  * Cloudflare Pages deployments must pass static generation builds without uncommitted diffs.
* **Active Tasks**:
  - [x] Add non-generic beta status notice and support email to landing page hero section.
  - [x] Update README with `quickstart.js` visual snippet and pure Kaze example agents demo.
  - [x] Release and publish `sofuu@0.3.0-beta.3` to npm registry.
  - [x] Deploy updated release tarballs and static site to Cloudflare Pages (`sofuu.xyz`).
  - [x] Optimize landing and documentation responsive layout for mobile viewports (360px-480px).
  - [x] PLAN-UNIFIED-DIAGNOSTICS: publish `docs/DIAGNOSTICS.md` (contract, mapping, privacy, division gaps) and `sofuu.d.ts` diagnostic types.

---

## 4. Development Workflow & Change Discipline

When working on any task:

1. **Identify the Division**: Determine which division (D01 – D11) owns the task.
2. **Consult Invariants**: Check the division's architectural boundaries before writing code.
3. **Verify Locally**: Run that division's independent verification command.
4. **Run Global Gatekeepers**: Before finalizing any cross-cutting change:
   ```bash
   make bc-check      # Asserts bytecode snapshots match JS sources
   make size-check    # Asserts release binary remains <= 5.5 MiB
   ```
5. **Update Division Checklist**: Mark the corresponding task complete in this map.
