# 🗺️ Sofuu Master Roadmap

> **The Living Engineering & Product Roadmap for Sofuu.**  
> *This document tracks our historical breakthroughs, current active sprint, and future milestones toward industry-standard adoption and enterprise readiness.*

---

## 📊 High-Level Status Dashboard

| Milestone | Focus Area | Status | Target Date |
| :--- | :--- | :--- | :--- |
| **v0.1 – Foundation** | QuickJS + libuv Event Loop & Embedded C ABI | 🟢 **Completed** | Q2 2026 |
| **v0.2 – Cognitive Memory** | Continuous Memory Architecture (QTSQ v1) & SIMD | 🟢 **Completed** | Q3 2026 |
| **v0.3 – Autonomous Agent Core** | Sandbox v2, MCP Client/Server, Session Mesh, CMA v2 | 🟢 **Completed (Shipped)** | Oct 2026 |
| **v0.3.5 – Hardening & DX** | Zero-Crash Fuzzing, Global `sofuu.z` Schemas, `sofuu compile` | 🟡 **In Progress (Active Sprint)** | Nov 2026 |
| **v0.4 – Enterprise & Edge** | Production llama.cpp Plugin Bridge, Enterprise Blueprints | 🔵 **Planned** | Nov/Dec 2026 |
| **v1.0 – Ecosystem Expansion** | Standalone Agent Ecosystem, WinterCG Edge Standard | 🟣 **Future Vision** | Q1 2027 |

---

## 🟢 1. Completed & Shipped (Production Ready)

### ⚡ Runtime Substrate & Event Loop
- [x] **QuickJS (ES2023) + libuv core**: JIT-free, lightweight JS execution; measured 22.5 ms median cold start wall-clock incl. process spawn (Apple M2 Pro, 2026-10-31 — see BENCHMARKS.md; the 3.8 ms figure from an earlier methodology is historical).
- [x] **Cooperative Task Scheduler**: Microtask starvation protection with `scheduler.yield()`.
- [x] **Zero-Dependency TypeScript**: In-memory lexical type-stripping (`ts.rs`) running `.ts` files with no build step.
- [x] **ESM Tree-Shaking Bundler**: Fast pure-Rust bundler (`bundler.rs`) emitting self-contained JavaScript files.
- [x] **Binary Size Strict Discipline**: Hard $\le 5.5\text{ MiB}$ cap, enforced by `make size-check` — macOS arm64 build is 5.30 MiB (5,559,904 B, 202 KB headroom); Linux arm64/x86_64 remain over the cap and are tracked release work.

### 🧠 Cognitive Vector Memory (CMA v2)
- [x] **Encrypted `.qtsq` Containers**: On-device vector storage with zero external database dependencies.
- [x] **Hardware SIMD Acceleration**: ARM NEON and x86 AVX2 vector math for sub-2ms top-$K$ cosine similarity.
- [x] **Cross-Session Recall**: Automatic memory persistence across runs with `/remember` and `/recall`.
- [x] **Portable Brain Cards**: Shareable vector cards (`/share` and `/import`) with cryptographic checksums.

### 🤖 Autonomous Agents & Security
- [x] **Autonomous Agent Loops**: Multi-step reasoning loops with prompt sovereignty and budget limits.
- [x] **Model Context Protocol (MCP)**: Native zero-dependency client and server implementation over `stdio` and `SSE`.
- [x] **Sandbox v2 (R7 Policy)**: Strict filesystem jailing, domain network allowlisting, credential shielding (`.ssh`, `.aws`, `.env`), and instruction fuel metering.
- [x] **Multi-Terminal Session Mesh**: Local IPC task synchronization (`/work`, `/done`) across simultaneous developer terminals.
- [x] **Observability Stream**: Structured `.jsonl` audit logging with crash-resilient write-ahead logging (WAL).

### 🛸 Robotics, Autonomous Drones & Ambient Computing
- [x] **Zero-Dependency MAVLink v2 Codec**: Pure-Rust streaming parser and serializer with CRC-16/MCRF4XX, pinned CRC_EXTRA table, and sliding-window resync.
- [x] **R7 Hardware Safety Gate**: 6-layer veto chain with reject-never-clamp bounds, dynamic geofencing, rate limiting, and fail-closed GPS dropout protection (`E_NO_GPS_FIX`).
- [x] **Autonomous Dead-Man Switch**: Runtime-scheduled 1 Hz heartbeat triggering ArduPilot native GCS failsafes upon interruption.
- [x] **Drone Agent Tools & Mission DSL**: High-level `sofuu.drone`, `sofuu.mavlink`, `PACKS.drone` (6 agent tools), pre-flight mission validator, and survey grid generators.
- [x] **Ground Rover & Linux GPIO**: Portable `/sys/class/gpio` and Raspberry Pi 5 chardev `/dev/gpiochip*` v2 line ABI with pulse timers; ground rover pre-flight test suite passing 5/5 cases.
- [x] **Kaze Live Hardware Bridge**: Direct coupling between Kaze natural scripting and active flight links/GPIO buses.
- [x] **Field CLI Toolchain**: `sofuu drone observe / status / send / mission / why / init`.

### 🌐 Ecosystem & Public Launch
- [x] **Official Documentation & Website**: Live at [sofuu.xyz](https://sofuu.xyz) and [sofuu.xyz/docs](https://sofuu.xyz/docs).
- [x] **Official AI Coding Skill**: Machine-readable agent guide deployed at [sofuu.xyz/SKILL.md](https://sofuu.xyz/SKILL.md).
- [x] **Multi-Provider BYOK Streaming**: Unified streaming for OpenAI, Anthropic, Gemini, Groq, and custom endpoints.

---

## 🟡 2. In Progress (Active Sprint — Weeks 1–2)

### ✅ The "Killer Wedge": Standalone Agent Compiler (`sofuu compile`) — SHIPPED 2026-10-30
*See detailed specification in [`docs/STANDALONE_COMPILER_PLAN.md`](docs/STANDALONE_COMPILER_PLAN.md). Shipped and verified — see CHANGELOG.*
- [x] **Trailer Format Implementation**: 32-byte `SOFUU_BIN_V1` binary trailer with CRC32 verification.
- [x] **Fast-Path Boot Detection**: Two reads (trailer + payload) before CLI dispatch; macOS finds the trailer before `LC_CODE_SIGNATURE`.
- [x] **AOT Bytecode Bundler**: Bundling TypeScript + QuickJS bytecode into a single executable.
- [x] **OS Signing & Integrity**: Mach-O `__LINKEDIT` extension + automated ad-hoc signing; ELF/PE plain append.
- [x] **Embedded Memory Support**: The whole `.qtsq` store (manifest + segments) packed into the binary (`--embed-brain`).

### 🛡️ Zero-Crash Hardening & Concurrency Fuzzing — JavaScript level [COMPLETED]
*See detailed specification in [`docs/ENTERPRISE_HARDENING_PLAN.md`](docs/ENTERPRISE_HARDENING_PLAN.md). Phase 1 delivered the global rejection sink (`process.on('unhandledRejection')` and exit 1 on unhandled), timer scaling (65,536 limit), the 10,000-concurrency torture harness (`tests/stress/concurrency_stress.js`), QuickJS `-Os` binary size reduction, and strict compliance with the $\le 5.5\text{ MiB}$ binary limit.*

**What "zero-crash" covers here: the JavaScript level.** Unhandled promise
rejections, timer exhaustion, event-loop starvation and QuickJS allocation
exhaustion are all handled, tested and terminated cleanly. It does **not** mean a
Rust panic is survivable — see the open item below.
- [x] **Promise Rejection Trap**: Unhandled rejection sink terminating with exit code 1 instead of Rust panics; `process.on('unhandledRejection')` support.
- [x] **Handle Lifecycle Auditing**: Guaranteeing `uv_loop_alive() == 0` when work drains to eliminate hanging processes.
- [x] **10,000-Concurrency Stress Harness**: High-concurrency test (`tests/stress/concurrency_stress.js`) passing 10k timers, 1k files, 500 SIMD ops, 100 subprocesses, and 5k ALS continuations in 668ms.
- [x] **OOM Memory Guardrails & Binary Size Gate**: Graceful JavaScript `RangeError` on QuickJS allocation exhaustion; release binary stripped & signed at **5,611,328 B** ($\le 5,767,168\text{ B}$ ceiling).

### ⚠️ Rust panic paths [OPEN]
- [ ] **Panic is abort, not recovery.** `Cargo.toml` sets `panic = "abort"`, so any Rust panic ends the process immediately: no unwinding, no `Drop`, no cleanup, no error object for the host to catch. That is a deliberate choice and the right one for a panic inside a QuickJS extern-C callback, where unwinding across the FFI boundary would be undefined behaviour — but it means "zero-crash" at the JS level does not extend to the Rust level.
- [ ] **`unwrap`/`expect` on input-reachable paths.** Roughly 11 panic sites exist on production paths, concentrated in `domains/infer/metal.rs`. Converting them to typed errors and asserting the invariants at load time is tracked in `docs/PLAN-AUDIT-REMEDIATION-2026-11-02.md` §8 (Task 6.2).

### 🧰 Developer Experience: Global Schema Tool Calling [COMPLETED]
*See detailed specification in [`docs/ENTERPRISE_HARDENING_PLAN.md`](docs/ENTERPRISE_HARDENING_PLAN.md). Phase 2 delivered zero-install schema tools (`sofuu.z` / `globalThis.z`), declarative `sofuu.tools.define`, multi-format exports (OpenAI, Anthropic, MCP, JSON Schema draft-07), in-process argument validation with self-correction feedback, and full TypeScript autocompletion (`types/sofuu.d.ts`).*
- [x] **Promote `microZod` to Global**: Expose `sofuu.z` and `globalThis.z` out of the box with zero npm packages (string, number, boolean, array, enum, object, record, literal, any with `.min()`, `.max()`, `.describe()`, `.default()`).
- [x] **Declarative Tool Builder (`sofuu.tools.define`)**: Unified helper generating OpenAI, Anthropic, MCP, and JSON Schema draft-07 specs automatically.
- [x] **Self-Correcting Tool Validation**: Invalid parameters return path-precise structured error messages (`-32602`) directly to the agent without crashing the loop, enabling LLM self-correction.
- [x] **TypeScript Declarations**: Comprehensive `types/sofuu.d.ts`, `npm/sofuu.d.ts`, and root `sofuu.d.ts` for instant Cursor / VS Code IntelliSense with 100% parity (`make plugin-check`).

### 🏛️ Zero-Risk Codebase Decoupling & In-Tree Capsule Modularization [COMPLETED]
*See detailed specification in [`docs/PLAN-ZERO-RISK-MODULARIZATION.md`](docs/PLAN-ZERO-RISK-MODULARIZATION.md). Encapsulates domain logic into clean capsules with zero binary size inflation and zero linker churn.*
- [x] **In-Tree Domain Capsules**: Reorganized `crates/sofuu-core/src/domains/` (`kaze/`, `robotics/`, `memory/`, `infer/`) with strict visibility boundaries and backward-compatible re-exports.
- [x] **Subsystem Registration Trait**: Standardized `Subsystem` trait (`name()`, `register_js()`, `health_check()`) decoupling domain initialization from `rt/engine.rs`.
- [x] **Local Invariant Preflight Gate (`scripts/preflight.sh`)**: Automated gate verifying domain boundary isolation, bytecode sync, size cap ($\le 5.5\text{ MiB}$), and unit test suites.

---

## 🔵 3. Planned Next (Weeks 3–4)

### 🔌 Production-Grade Local Inference Plugin
- [x] **`sofuu-plugin-llamacpp` Bridge**: Dynamic C-ABI provider plugin connecting `sofuu.ai.streamLocal()` to `libllama`.
- [x] **Full Hardware Acceleration**: Native Metal (Apple Silicon) and CUDA/Vulkan (Linux) acceleration.
- [x] **Comprehensive Quantization Support**: Q4_K_M, Q5_K_M, Q8_0, IQ3, and multi-model context switching.
- [x] **One-Command Installation**: `sofuu plugin add llamacpp` with automatic precompiled binary distribution.

### 🏢 Enterprise Reference Recipes
- [ ] **Enterprise RAG Pipeline (`examples/enterprise-rag-pipeline/`)**: In-process document chunking, SIMD vector storage, and cited Q&A.
- [ ] **Autonomous Code Reviewer (`examples/autonomous-code-reviewer/`)**: Jailed Git diff inspection, security linting, and SARIF report generation.
- [ ] **Collaborative Swarm Mesh (`examples/multi-terminal-mesh/`)**: Two autonomous terminal workers coordinating tasks over the Session Mesh.

---

## 🟣 4. Future Vision (Enterprise & Ecosystem Expansion)

- [ ] **WinterCG Compliance**: Full compatibility with Web Standard `fetch`, `Request`, `Response`, and `ReadableStream` for instant deployment on Cloudflare Workers / Vercel Edge.
- [ ] **Cloud Session Mesh Relay**: Optional zero-config encrypted peer-to-peer task mesh connecting remote developer teams.
- [ ] **Cross-Platform Compilation Targets**: `sofuu compile --target linux-x86_64` / `darwin-arm64` cross-compilation from any host.
- [ ] **Mobile & Embedded SDKs**: Complete Swift (iOS/macOS) and Kotlin (Android) wrappers around `libsofuu`.

---

## 🔄 Roadmap Maintenance Protocol

Whenever a task or milestone is completed:
1. Update the checkbox `[ ]` $\rightarrow$ `[x]`.
2. Move completed items to **§ 1. Completed & Shipped**.
3. Re-verify the binary size invariant:
   ```bash
   make size-check
   ```
4. Push updates to `main` to keep community and prospective acquirers synchronized with our development velocity.
