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
| **D05** | **Kaze Language & Compiler** | Declarative `.kaze` AST, linear arena VM, capability gates, supervisor daemon | `kaze/`, `crates/sofuu-core/src/domains/kaze/`, `crates/sofuu-core/src/compiler.rs`, `example-agents/` | Core | Policy-compliant Kaze checks (see D05) |
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
  - [x] AUDIT-REMEDIATION-2026-11-08: Complete core substrate tasks — microtask queue drainage on 1M chained promises (CLI-054), native `node:crypto` hashing (SHA-1/MD5/SHA-256/SHA-512/HMAC via `rt/digest.rs`) (JR-053), and `sofuu.spawn()` handle contract with ENOENT rejection (JR-073, JR-077).

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
  - [x] AUDIT-REMEDIATION-2026-11-08: Complete memory integrity tasks — path validation rejecting empty/directory paths (MEM-003, MEM-004), corrupt file preservation (MEM-013), post-close access `E_LOCKED` protection (MEM-024), and immediate 65KB record ceiling validation.
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
  - [x] AUDIT-REMEDIATION-2026-11-08: Complete agent loop hardening — turn-block context retention with consolidated parallel tool call emission (AG-025b), `maxDepth=0` preservation (AG-044/049), provider latency accounting against `maxWallMs` (AG-043), and sensitive credential / home directory path redaction in run logs (AG-101/102).
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
  - [x] AUDIT-REMEDIATION-2026-11-08: Complete tool belt and protocol gates — SSRF loopback alignment across URL filters and socket gates with strict private/metadata blocking (JR-090, JR-092), MCP disconnect idempotency and process-group signal isolation (MC-017), and dynamic MCP stdio line buffer up to 32MB (MC-019).
  - [ ] Add streaming MCP tool response pipes for large file operations.

---

### Division 05: Kaze Language, VM & Standalone Compiler (D05)
* **Mission**: Pure declarative agent routine language and linear arena VM for mission-critical, zero-GC edge intelligence.
* **Owned Files**:
  * `kaze/` (Separate standalone Kaze 1.0 repository, CLI, compiler, VM, and documentation)
  * `crates/sofuu-core/src/domains/kaze/` (Shipped parser, checker, bytecode compiler, arena VM, capability gates, and runtime bridge; aliased as `crate::sfs`)
  * `crates/sofuu-core/src/script_cmd.rs` (`sofuu script` CLI, including `.kazebc` compilation and execution)
  * `crates/sofuu-core/src/compiler.rs` (Runtime standalone binary compiler)
  * `example-agents/` (`quant_sentinel/`, `med_protocol/`, `agri_sentinel/`, `paper_trader/`)
* **Implementation and Verification**:
  * Standalone Kaze reports language version `1.0` and uses bytecode format `4`; its dated, per-item tagged [implementation inventory](https://github.com/sofuu-runtime/kaze/blob/main/docs/KAZE-1.0-REFERENCE.md) is in the separate Kaze repository. The separate engine embedded in Sofuu reports language version `0.1`. Its current source writes format `2` with statement-line metadata and can read legacy format `1`; the dated format-1 local inventory is [`docs/KAZE-CURRENT-SURFACE.md`](docs/KAZE-CURRENT-SURFACE.md). The engines have different features and incompatible artifacts. `docs/KAZE-SPEC.md` is a historical embedded-engine specification with known stale claims. Published binaries need separate verification.
  * **Planned version alignment:** Adopt one Kaze 1.0 language contract across standalone and Sofuu, preferably with a shared language core and separate host integrations. This is not implemented; preserve current version reporting until contract, fail-closed behavior, compatibility, and release checks pass. Runtime and bytecode format versions remain separate. The local plan at `kaze/docs/VERSION-ALIGNMENT.md` is gitignored in this checkout.
  * The landing site's [`/docs/kaze`](https://sofuu.xyz/docs/kaze) page renders a tagged snapshot of the standalone 1.0 inventory at Kaze commit `5f2475a`, with examples and links from `/kaze` and `/docs`. The local site source also records the planned version alignment; that note is not in the deployed snapshot. Update `sofuu-landing/src/app/docs/kaze/reference.md` and its examples when the standalone inventory changes. Production Cloudflare Pages deployment `83536b13` was verified on 2026-11-11; published Kaze binaries remain separately unverified.
  * The standalone source now requires explicit grants and a valid policy, refuses unavailable non-robotics backends, and requires labelled opt-in for drone/rover simulation. The [standalone roadmap](https://github.com/sofuu-runtime/kaze/blob/main/docs/ROADMAP.md) tracks current progress and remaining verification. Targeted safety and migrated compatibility/language tests pass. Live drone actuation is disabled until the required 1.5-second heartbeat and per-field telemetry freshness veto is implemented; physical hardware and release compatibility remain unverified.
  * **Embedded Kaze fail-closed work, 2026-11-11:** Current source requires grants and an R7 policy with explicit `tools.allow` or `tools.deny` rules for calls. An explicitly selected per-run CMA memory backend is available; unavailable AI, tool, persistent Kaze memory, alert, sensor and device operations refuse instead of returning fabricated success. Drone/rover simulation requires an explicit test selection and is labelled. Live drone and GPIO paths are disabled pending verified policy and safety propagation. The local debug build passed Kaze unit and cross-surface checks; published artifacts remain unverified. See the tracked [embedded capability status](KAZE-EMBEDDED.md).
  * **Embedded Kaze async bridge, 2026-11-13:** `sofuu.kaze.runAsync` and its `sofuu.sfs` alias suspend at authorized AI/tool requests, apply active host policy before dispatch, bound serialized requests, validate `ai.classify` results against requested labels, and report missing-model/unknown-outcome failures with Kaze diagnostics. `sofuu script run` now serves a source or `.kazebc` file that declares `ai`/`tool` through the same bridge; every other script keeps the engine-free synchronous runner and fails closed for unavailable backends. Approval-required requests are denied without a verifiable approval receipt. Debug artifact `cd0a0abef7835aad1393a4d75e82362231d2768c1f18e92e1324b5d1749b0a21` passed 99 focused embedded Kaze tests, 246 broader Kaze/compiler/memory/robotics tests, the fail-closed, per-run-memory and JS-boundary scripts, and a real model-plus-MCP journey through `sofuu.kaze.runAsync`, a compiled agent, and `sofuu script run` (source and `.kazebc`) with a configured remote provider. Local-GGUF weights, cancellation behavior, published binaries and physical hardware remain unverified.
  * `sofuu script compile` emits Kaze bytecode, while `sofuu compile` creates a standalone executable. The in-process APIs are exposed as `sofuu.kaze` and `sofuu.sfs`.
  * The embedded Kaze unit suite has been migrated for these changes. The ignored local `kaze_language` integration suite and repository-wide mandatory gates still include non-robotics stand-ins and require migration before they can be run under `RULES.md` §1.9. Run `make kaze-drift-check` to report differences from the prototype.
  * `kaze/`, both local Kaze documents under `docs/`, `crates/sofuu-core/tests/`, and `examples/kaze/` are gitignored local resources and may be absent in a clean checkout.
* **Architectural Boundaries**:
  * Compiler must remain 100% domain-agnostic; zero hardcoding of parameter names or external endpoints.
  * Standalone compiled binaries must support production 12-factor configuration resolution.
  * Agents must run on pure Kaze with zero auxiliary `.js` or `.sh` wrapper scripts.
  * Missing parameters must trigger structured diagnostic guidance, not runtime crashes.
* **Active Tasks**:
  - [ ] Align standalone and Sofuu on the shared Kaze 1.0 contract; define the contract, shared-core integration, compatibility and migration decisions, and cross-surface conformance before changing the embedded version string. See `kaze/docs/VERSION-ALIGNMENT.md`.
  - [ ] Complete standalone Kaze capability-safety verification and release compatibility decision; targeted policy, grant, backend, link-loss, and argument-gate checks are in `kaze/tests/capability_safety.rs`. Close the live-drone heartbeat and telemetry freshness gap before physical use, and migrate non-robotics stand-ins before running full release suites.
  - [x] K0 inventory in progress: local Kaze source and binary behavior documented; published artifact, real backend/policy paths, and applicable model coverage remain to be verified.
  - [x] Address missing/empty policy handling in the standalone prototype (F-04).
  - [ ] K1: embedded source, bytecode, JS aliases, REPL and compiled one-shot paths refuse unavailable synchronous non-robotics calls and require explicit grants and applicable policy; `sofuu.kaze.runAsync`, a compiled agent that declares `needs: ai`/`needs: tool`, and `sofuu script run` for source or `.kazebc` dispatch authorized AI/tool requests through bounded continuations. A real model-plus-MCP journey has passed on a debug build with a configured remote provider. Continue auditing supervisor/interactive execution, cancellation, local-GGUF coverage and released binaries; migrate integration/gate tests and finish the host split before claiming the milestone complete. Local debug coverage: `scripts/test-kaze-fail-closed.py`, `scripts/test-kaze-memory-real.py`, `scripts/test-kaze-js-boundaries.py`, `scripts/test-kaze-real-ai.py`, and `scripts/test-kaze-cli-real-ai.py`.
  - [x] AUDIT-REMEDIATION-2026-11-08: Complete Kaze VM and Compiler remediation — recursion depth ceiling $\le 256$ (KZ-095/099), modulo-by-zero trap `error[E_KAZE_DIV0]` (KZ-035), 12-factor configuration resolution for routine parameters (KZ-079/082/084), standalone compiler unresolved import failure scrubbing (CP-075/068), and standalone top-level await (CP-007/008).
  - [x] Build and verify 4 showcase agents in `example-agents/` with automated build harness.
  - [ ] PLAN-UNIFIED-DIAGNOSTICS: Kaze parser/checker and statement-level VM locations, version-2 bytecode mapping with version-1 reader, and run/check JSON diagnostics are in source. A local release-profile standalone arithmetic failure uses the shared renderer. Expression spans, Kaze call frames, source excerpts and broader standalone/release evidence remain open.
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
  * Fast path execution (`sofuu eval`, `sofuu run`): target sub-25ms for a specified workload and hardware; track measured results and conditions in BENCHMARKS.md.
* **Active Tasks**:
  - [x] Align TUI banner with brand voice ("runtime for AI") (F-07).
  - [x] Verify CLI argument parsing matches 12-factor configuration hierarchy.
  - [ ] PLAN-UNIFIED-DIAGNOSTICS: concise/full error rendering and bounded reported JS frames are implemented in source; audit remaining TUI/REPL, ABI, protocol and release surfaces before claiming complete coverage.
  - [x] AUDIT-REMEDIATION-2026-11-08: Complete CLI polish — read-only `--help` with zero disk side-effects (CLI-008/009), strict ANSI scrubbing under `NO_COLOR=1` / `TERM=dumb` (CLI-023/024), `sofuu run -` / `eval <<EOF` stdin execution (CLI-055–057), and `sofuu bundle` stdout streaming (CP-069).
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
  - [x] Deploy the tagged Kaze 1.0 reference page and links from the Kaze and documentation pages to Cloudflare Pages (`83536b13`); verify `/docs/kaze`, `/kaze`, `/docs`, and the existing `/map`. The clean deployment source is landing commit `20d24d3` on the local `codex/kaze-reference-deploy` branch; it has not been pushed.
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
