# PLAN-MASTER-STRATEGY — Project Sofuu: The AI-Native Operating Environment

> **Document Type:** Master Product Strategy, Architecture, & Growth Plan  
> **Target:** Y Combinator / Pre-Seed Acceleration & Developer Adoption  
> **Founder / Architect:** Haruhito  
> **Repository Baseline:** `sofuu v0.2.0-beta` (Rust-first shell, QuickJS/libuv core, SIMD kernels, QTSQ persistent brain, in-process CPU inference shipped 2026-10-11)
>
> **Vision update 2026-10-11 — Edge & Mobile Standard (canonical):** Sofuu is the
> vLLM of Edge & Mobile — Edge, Phone, Mac, Distributed Edge Swarms in one ≤5 MB
> binary (4.7 MB shipped). This doc's "AI-native operating environment" thesis
> stands and is now executed through [`PLAN-EDGE-INFERENCE-STANDARD.md`](PLAN-EDGE-INFERENCE-STANDARD.md)
> (milestones, no timelines); engine status in [`PLAN-CPU-INFERENCE-ENGINE.md`](PLAN-CPU-INFERENCE-ENGINE.md).

---

## 1. Executive Summary: What is Sofuu?

### The Core Premise
**Sofuu (素風 — "Simple/Pure Wind") is the world's first AI-native JavaScript/TypeScript operating environment.**

It is not just an inference engine, not just a JavaScript runtime, and not just an agent framework. **It is all of them unified into a single, zero-dependency, ≤5 MB native binary (4.7 MB shipped with in-process CPU inference).**

```
┌────────────────────────────────────────────────────────────────────────┐
│                        WHAT SOFUU GIVES DEVELOPERS                     │
├────────────────────────────────┬───────────────────────────────────────┤
│ The Old Broken AI Stack        │ The Unified Sofuu Environment         │
├────────────────────────────────┼───────────────────────────────────────┤
│ • 5 GB Python / PyTorch / CUDA │ • 5 MB single native binary           │
│ • 15 npm packages per project  │ • Built-in `sofuu.ai` & `sofuu.agent` │
│ • External vector databases    │ • Built-in 320ns SIMD vector memory   │
│ • External daemons (Ollama)    │ • Built-in CPU inference engine       │
│ • Temporal / Kafka for state   │ • Built-in crash-durable journal      │
│ • Cloud token waste & 429s     │ • Built-in 5 tiny context-budget MLs  │
└────────────────────────────────┴───────────────────────────────────────┘
```

### The Vision
Developers should be able to download a single file, write clean JavaScript or TypeScript, and immediately have **autonomous agents, long-term cognitive memory, local & cloud LLMs, tool sandboxes, and interactive developer interfaces** that work anywhere—on laptops, phones, edge devices, and serverless containers—completely offline and with zero setup.

> **2026-10-11 execution:** the "runs anywhere" clause is now the Edge & Mobile
> Standard — phone / Pi / Mac / LAN swarm / cloud, same binary, zero daemons,
> <30 ms cold start, 0.0% idle (`PLAN-EDGE-INFERENCE-STANDARD.md`). The embedded
> engine row below is delivered: 17–26 ms mmap load, 138–150 tok/s prefill,
> 30–33.5 tok/s decode, <0.2 ms KV compaction, O(W+S) SWA.

---

## 2. The Five Architectural Pillars of Sofuu

Sofuu is organized into five tightly integrated, high-leverage architectural pillars:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      THE FIVE PILLARS OF SOFUU                         │
├────────────────────────────────────────────────────────────────────────┤
│ 1. THE JAVASCRIPT / TYPESCRIPT RUNTIME                                 │
│    • QuickJS C core + libuv event loop + safe Rust shell (sofuu-core). │
│    • 3 ms cold start, ~3 MB memory footprint, zero JIT (iOS compliant).│
│    • Built-in TypeScript stripper & ESM bundler (`sofuu bundle`).      │
├────────────────────────────────────────────────────────────────────────┤
│ 2. THE BATTERIES-INCLUDED AI STANDARD LIBRARY                          │
│    • `sofuu.ai`: Zero-copy SSE streaming for cloud AND local models.   │
│    • `sofuu.agent`: Multi-step agent loops with tool budgets & cancel. │
│    • `sofuu.mcp`: Model Context Protocol client AND server built-in.   │
│    • `sofuu.tools`: Native file editing, diff previews, bash, and web. │
├────────────────────────────────────────────────────────────────────────┤
│ 3. THE COGNITIVE MEMORY SYSTEM (The Brain)                             │
│    • Long-term persistence across sessions via encrypted QTSQ codec.   │
│    • CMA (Cognitive Memory Architecture) with decay and consolidation. │
│    • Native SIMD vector ops executing at 320 nanoseconds (10x V8 speed)│
├────────────────────────────────────────────────────────────────────────┤
│ 4. THE ON-DEVICE CONTEXT ECONOMY (5 Tiny ML Gates)                     │
│    • Five ~8.5k parameter micro-models running on CPU in < 20 µs.      │
│    • Freshness, Compaction, Relevance, Alloc, and Supervisor gates.    │
│    • Saves 30%–50% of tokens by compacting history before model calls. │
├────────────────────────────────────────────────────────────────────────┤
│ 5. DUAL EXECUTION ENGINE (Cloud Gateway + Embedded CPU Engine)         │
│    • Remote Gateway: OpenAI, Anthropic, DeepSeek with dynamic backoff. │
│    • Embedded Engine (`sofuu-infer`): Zero-daemon quantized CPU engine │
│      with mmap zero-copy loading & gigatoken BPE tokenization.         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. How Developers Actually Use Sofuu

Sofuu supports four distinct, seamless developer workflows:

### Workflow A: The Interactive Terminal Workspace (`sofuu chat` / bare `sofuu`)
A developer types `sofuu` in their terminal:
- Boots instantly in 3 ms into an interactive AI terminal workspace.
- **25 built-in dark & light themes** (Slate, Paper, Nord, Monokai, etc.).
- Direct tool execution with interactive permission modes (`/plan`, `/edit`, `/full`).
- Visual unified diffs and markdown rendering for code edits.
- Slash commands: `/model`, `/provider`, `/compact`, `/remember`, `/sessions`, `/cost`, `/why`.
- Real-time project-scoped session mesh sharing context across open terminals.

### Workflow B: Headless Autonomous Agents (`sofuu agent run`)
Developers automate complex repo-wide tasks from the CLI or CI/CD pipelines:
```bash
# Runs an autonomous multi-step agent without opening a UI:
sofuu agent run coder "Refactor database queries in src/ to use prepared statements" --json
```
The agent executes with automatic step budgets, tool sandboxing, and crash-durable journal checkpoints.

### Workflow C: Modern JavaScript / TypeScript Scripting (`sofuu run`)
Developers write clean scripts with zero npm dependencies:
```javascript
// index.ts — Runs with `sofuu run index.ts`
import { ai, agent, memory } from "sofuu";

// 1. Query persistent brain:
const context = await memory.recall("architecture decisions");

// 2. Stream tokens seamlessly (local or remote):
for await (const chunk of ai.stream("Draft the release notes", { model: "claude-sonnet-4-6" })) {
    process.stdout.write(chunk.text);
}
```

### Workflow D: Embedded Mobile & Desktop Apps (`libsofuu` & `sofuu-desktop`)
- **iOS & Android:** Via SwiftPM, CocoaPods, and Gradle AAR, app developers embed `libsofuu` directly into mobile apps. Because QuickJS has no JIT, it complies with Apple's strict executable memory policy.
- **Desktop:** The native macOS Tauri v2 app (`sofuu-desktop`) wraps the engine in-process with macOS Keychain credential storage and system permission prompts.

---

## 4. The Value Matrix: Why Developers Switch to Sofuu

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COMPETITIVE VALUE MATRIX                        │
├─────────────────┬────────────────────┬─────────────────────────────────┤
│ Incumbent Tool  │ What It Misses     │ Why Sofuu Wins                  │
├─────────────────┼────────────────────┼─────────────────────────────────┤
│ Node.js / Bun   │ General JS only;   │ Sofuu includes native AI, MCP,  │
│                 │ requires 20 npm    │ memory, vectors, and local      │
│                 │ packages for AI.   │ models out of the box.          │
├─────────────────┼────────────────────┼─────────────────────────────────┤
│ LangChain /     │ Heavy Python/TS    │ Sofuu is a single 5 MB binary   │
│ LangGraph       │ abstractions; high │ with native agent loops and     │
│                 │ breaking changes.  │ sub-millisecond execution.      │
├─────────────────┼────────────────────┼─────────────────────────────────┤
│ Ollama          │ Desktop background │ Sofuu is embeddable in-process; │
│                 │ daemon only; can't │ has integrated memory, agents,  │
│                 │ embed in iOS/apps. │ and a full JavaScript runtime.  │
├─────────────────┼────────────────────┼─────────────────────────────────┤
│ Temporal        │ Requires heavy Go/ │ Sofuu provides zero-infra,      │
│                 │ JVM servers & DB   │ single-file durable journals    │
│                 │ clusters.          │ for agent workflows.            │
└─────────────────┴────────────────────┴─────────────────────────────────┘
```

---

## 5. The Phased Master Roadmap

```
┌────────────────────────────────────────────────────────────────────────┐
│                        THE 4-PHASE MASTER ROADMAP                      │
├─────────┬───────────────────────────────┬──────────────────────────────┤
│ Phase   │ Primary Deliverable           │ Exit Gate / Metric           │
├─────────┼───────────────────────────────┼──────────────────────────────┤
│ PHASE 1 │ CPU-First Text Inference      │ • `sofuu-infer` operational  │
│ (Now)   │ Engine (`sofuu-infer`)        │ • gigatoken BPE integrated   │
│         │ + Clean Test Suite            │ • 320ns SIMD vector speed    │
│         │ + YC Application Submission   │ • 50+ active alpha users     │
├─────────┼───────────────────────────────┼──────────────────────────────┤
│ PHASE 2 │ Real-Data Micro-Model Tuning  │ • Compaction saves ≥30% tk   │
│ (Post-YC│ + Agent Crash-Durable Journal │ • Zero duplicate tool runs   │
│  Prep)  │ (WAL State Engine)            │ • Safe path/SSRF firewall    │
├─────────┼───────────────────────────────┼──────────────────────────────┤
│ PHASE 3 │ Hosted Cloud API Models       │ • Memory distillation API    │
│ (Growth)│ (Distilled Memory & Dense RAG)│ • Serverless GPU pool (Modal)│
│         │                               │ • First SaaS revenue ($29/mo)│
├─────────┼───────────────────────────────┼──────────────────────────────┤
│ PHASE 4 │ Multimodal Expansion          │ • On-device Vision & Audio   │
│ (Scale) │ (Image + Audio on Edge)       │ • Enterprise team governance │
│         │ + Enterprise Team Security    │ • $1M+ ARR trajectory        │
└─────────┴───────────────────────────────┴──────────────────────────────┘
```

---

## 6. Y Combinator (YC) Application Narrative

When submitting Project Sofuu to Y Combinator, this is the exact, high-conviction narrative:

### 1. What is your company going to make?
*Sofuu is the AI-native operating environment for JavaScript. We replace the bloated, fragmented AI stack—Python environments, vector databases, agent frameworks, and background daemons—with a single 5 MB native binary that runs local models, autonomous agents, and persistent memory with zero setup.*

### 2. What is new about what you make?
*Today, building an AI agent requires stitching together 15 npm packages, managing a vector database, running background daemons like Ollama, and risking process crashes that burn tokens and duplicate tool actions.  
Sofuu unifies everything: a sub-3ms JavaScript engine, native SIMD vector memory operating at 320 nanoseconds, an embedded CPU inference engine, and five on-device context-budget micro-models that eliminate token waste.*

### 3. Why now?
*Sub-1.5B models (Qwen 2.5, Llama 3.2) now possess extraordinary reasoning capabilities that execute at 50+ tokens/second on standard CPUs. The bottleneck is no longer GPU compute—it is runtime and deployment friction. Developers need the "SQLite of AI": an embeddable, single-file environment that runs anywhere.*

### 4. Founder Advantage:
*Solo-builder execution speed: Built a 99,000-line codebase in Rust and C with SIMD intrinsics, custom FFI bridges, and embedded engines with $0 capital.*

---

## 7. Open-Core Business Strategy & Monetization

```
┌────────────────────────────────────────────────────────────────────────┐
│                         OPEN-CORE MONETIZATION                         │
├────────────────────────────────────────────────────────────────────────┤
│ 100% FREE & OPEN SOURCE (MIT LICENSE)                                  │
│ • Full native runtime & CLI (`sofuu`, `sofuu run`, `sofuu chat`).     │
│ • Local CPU inference engine (`sofuu-infer`).                          │
│ • Local 320ns SIMD vector math & CMA memory persistence.               │
│ • Local agent framework (`sofuu.agent`) & MCP client/server.           │
│ • 5 tiny on-device context-economy ML gates.                           │
├────────────────────────────────────────────────────────────────────────┤
│ COMMERCIAL / ENTERPRISE SAAS TIERS                                     │
│ • Developer Pro ($19/mo): Hosted high-density cognitive memory models  │
│   and dense RAG code embeddings.                                       │
│ • Team Plan ($29/dev/mo): Cross-device real-time session mesh sync,    │
│   shared prompt policies, and webhook approval traps.                  │
│ • Enterprise ($25k+/yr): Centralized RBAC agent firewall, SOC 2 audit  │
│   forensics, tamper-proof execution logs, and air-gapped deployments.  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Strategic Summary

Sofuu is **not** a piece-meal tool:
- It gives developers the ease of **JavaScript**.
- The raw speed of **native C/Rust SIMD** (320ns vector operations).
- The autonomy of **native agent loops and tools**.
- The intelligence of **embedded CPU text inference**.
- And the durability of **long-term cognitive memory**.

All packaged into a single, beautiful, lightning-fast binary that runs anywhere on Earth.
