# Master Architectural Plan: Sofuu JavaScript Runtime Substrate & Async Event Loop Modernization

> **Document:** `docs/PLAN-RUNTIME-SUBSTRATE-AND-EVENT-LOOP.md`  
> **Author:** Haruhito  
> **Target:** Sofuu AI-Native Edge JavaScript Runtime (`ai-native-js-runtime`)  
> **Objective:** Modernize and harden Sofuu's JavaScript execution engine, microtask scheduler, native memory bridges, context propagation, and deterministically metered sandboxes.

---

## 1. Architectural Vision & Invariants

Sofuu is engineered as the lightweight, embeddable "SQLite of AI Agents & Edge Inference".
While V8 or Node.js runtimes require tens to hundreds of megabytes, Sofuu pairs **QuickJS** with **libuv** in a zero-dependency Rust binary under **5.0 MB**.

```
                        MODERNIZATION ARCHITECTURE
                        
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. Cooperative Scheduler & Starvation Protection                      │
  │    W3C scheduler.yield() • Microtask Bounding • Responsive UI Tick    │
  ├────────────────────────────────────────────────────────────────────────┤
  │ 2. AOT Bytecode Snapshotting for Shipped Drivers                       │
  │    Pre-compiled Bytecode • Sub-200µs Engine Cold Start                │
  ├────────────────────────────────────────────────────────────────────────┤
  │ 3. Zero-Copy TypedArray & Native Memory Bridge                         │
  │    AnonBuffer ↔ Float32Array ↔ QTSQ Columnar Direct Memory Views       │
  ├────────────────────────────────────────────────────────────────────────┤
  │ 4. Native AsyncLocalStorage & Agent Context Propagation                │
  │    Promise Capability Context Propagation • Universal AbortSignals     │
  ├────────────────────────────────────────────────────────────────────────┤
  │ 5. Deterministic Agent Sandboxing & Fuel Meters                       │
  │    Interrupt-based Instruction & Memory Caps for Untrusted Tools       │
  ├────────────────────────────────────────────────────────────────────────┤
  │ 6. Full Integration Torture Suite & Release Budget Audit               │
  │    Concurrent Stress Tests • Latency Benchmarks • Budget Gate          │
  └────────────────────────────────────────────────────────────────────────┘
```

### Strict Architectural Invariants
1. **Binary Size Budget**: Stripped release binary must remain strictly $\le 5.0\text{ MB}$ (current: 4.5 MB; budget headroom: 500 KB).
2. **Deterministic Non-Blocking Execution**: Heavy CPU tasks or deep recursion must never starve the libuv I/O loop or freeze host UI threads (macOS Swift IDE and Android JNI).
3. **Memory Safety & Zero-Leak Multi-Tenancy**: The process-global libuv loop must remain safely shareable across multiple engine instances without handle leakage or use-after-free.
4. **Non-Inference Isolation**: All modifications remain strictly confined to runtime substrate crates (`crates/sofuu-core/src/rt/`, `crates/sofuu-ffi/`, `src/js/`), never altering inference kernels (`metal.rs`, `simd/infer_kernels.cpp`).

---

## 2. Phase-by-Phase Roadmap

### Phase 1: Cooperative Scheduler & Microtask Starvation Protection
- **Problem**: Microtasks (`Promise.then()`, `async/await` continuations) are drained in a tight `while (JS_ExecutePendingJob)` loop in `rt/loop.rs`. Heavy recursive microtasks starve the libuv loop (timers don't fire, abort signals stall, UI threads freeze).
- **Deliverables**:
  - `sofuu_flush_jobs_bounded(ctx, max_jobs)`: Yields after $N = 1024$ jobs to run `uv_run(loop, UV_RUN_NOWAIT)` and dispatch OS signals before resuming remaining microtasks.
  - W3C `scheduler.yield()` and `sofuu.yield()` API: Breaks current microtask turn, schedules immediate tick, and returns a promise resolving on the next turn.
- **Verification**: `tests/test_scheduler_starvation.js` measuring timer jitter $\le 2\text{ ms}$ under heavy microtask load.

### Phase 2: AOT Bytecode Snapshotting for Shipped Drivers
- **Problem**: Cold-start latency parses, lexes, and byte-compiles raw JS strings (`chat.js`, `storage.js`, `extractor.js`, `npm.js`) on every startup (~5.2 ms).
- **Deliverables**:
  - Build-time bytecode compilation via `crates/sofuu-core/build.rs` using `JS_WriteObject(..., JS_WRITE_OBJ_BYTECODE)`.
  - Instant engine startup via `JS_ReadObject` instead of `JS_Eval`.
- **Verification**: Cold start latency drops to $\le 0.5\text{ ms}$.

### Phase 3: Zero-Copy TypedArray & Native Memory Bridge
- **Problem**: Vector embeddings and QTSQ columnar blocks copy memory when passed to JavaScript.
- **Deliverables**:
  - Safe FFI wrapper for `JS_NewArrayBuffer` with external deallocation callbacks.
  - Direct mapping between Rust `AnonBuffer` and JavaScript `Float32Array` / `Uint8Array`.
- **Verification**: `tests/test_zerocopy_buffer.js` verifying 0 allocations and 0 memory leaks upon GC.

### Phase 4: Native AsyncLocalStorage & Agent Context Propagation
- **Problem**: Autonomous agent execution across nested async tools requires manual passing of `sessionId`, `traceId`, and `AbortSignal`.
- **Deliverables**:
  - Promise capability hook in `sofuu_promise_new` snapshots active context.
  - `sofuu.context.AsyncLocalStorage` implementation with `.run()`, `.getStore()`, `.set()`.
- **Verification**: `tests/test_async_context.js` validating context propagation across deep async trees.

### Phase 5: Deterministic Agent Sandboxing & Fuel Meters
- **Problem**: Agent-generated tools can execute infinite loops or exhaust memory.
- **Deliverables**:
  - QuickJS interrupt handler (`JS_SetInterruptHandler`) tracking remaining instruction fuel and wall-clock deadline.
  - `sofuu.sandbox.run(code, { fuel, timeoutMs, maxMemoryMB, globals })`.
- **Verification**: `tests/test_sandbox_fuel.js` ensuring infinite loops terminate within $\le 5\text{ ms}$ safely.

### Phase 6: Integration Torture Suite & Release Audit
- **Deliverables**:
  - Comprehensive stress test combining all 5 features under 100 concurrent tasks.
  - Binary size audit strictly under $5.0\text{ MB}$.
  - OpenQodex review gate verification and clean git push.
