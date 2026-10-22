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

### Phase 1: Cooperative Scheduler & Microtask Starvation Protection [COMPLETE]
- **Status**: Implemented, verified, reviewed, and pushed (`9088c7143`).
- **Deliverables**:
  - `sofuu_flush_jobs_bounded(ctx, max_jobs)`: Yields after $N = 1024$ jobs to run `uv_run(loop, UV_RUN_NOWAIT)` and dispatch OS signals before resuming remaining microtasks.
  - W3C `scheduler.yield()`, `sofuu.yield()`, `setImmediate`, `clearImmediate`, `queueMicrotask`: Breaks current microtask turn, schedules immediate tick, and returns a promise resolving on the next turn.
- **Verification**: `tests/test_scheduler_starvation.js` executing 21,595 continuous microtasks while servicing timer ticks with 0 event loop lockup (timer jitter $\le 2\text{ ms}$).

### Phase 2: AOT Bytecode Snapshotting for Shipped Drivers [COMPLETE]
- **Status**: Implemented, verified, and audited.
- **Deliverables**:
  - QuickJS bytecode C-FFI serialization/deserialization exports (`JS_WriteObject`, `JS_ReadObject`, `JS_EvalFunction`) in `crates/sofuu-ffi/src/qjs.rs`.
  - Bytecode compiler CLI subcommand `sofuu compile-bc <in.js> <out.bc>` in `crates/sofuu-core/src/main.rs`.
  - Fast-path bytecode hydration and source fallback in `crates/sofuu-core/src/shipped.rs` and `crates/sofuu-core/src/rlm/js_api.rs`.
  - Precompiled AOT bytecode for all 7 shipped drivers: `agent.bc`, `web.bc`, `tools.bc`, `extractor.bc`, `storage.bc`, `chat.bc`, `rlm.bc`.
- **Verification**: `tests/test_bytecode_hydration.js` verifying instantaneous hydration of all 7 subsystems; engine cold start time dropped to $\le 16\text{ ms}$; stripped binary size is $4.7\text{ MB}$ (well within $\le 5.0\text{ MB}$ budget).

### Phase 3: Zero-Copy TypedArray & Native Memory Bridge [COMPLETE]
- **Status**: Implemented, verified, audited.
- **Deliverables**:
  - Bound `JS_DetachArrayBuffer` in `crates/sofuu-ffi/src/qjs.rs` and patched QuickJS detach double-free in `deps/quickjs/quickjs.c`.
  - Implemented `crates/sofuu-core/src/rt/zerocopy.rs`:
    - `AlignedBuffer`: 64-byte aligned SIMD memory allocation for AVX-512 / ARM NEON.
    - Safe C trampoline deallocators (`free_aligned_buffer_cb`, `free_vec_u8_cb`, `free_vec_f32_cb`, `free_arc_vec_f32_cb`) linking QuickJS ArrayBuffer lifecycle directly to Rust memory.
    - Zero-copy constructors: `new_zerocopy_aligned_buffer`, `new_zerocopy_u8_buffer`, `new_zerocopy_f32_array`, `new_shared_f32_array`.
    - Safe buffer reflection and borrow: `extract_buffer_info`, `create_typed_view`.
  - Exposed modern JavaScript API on both `sofuu.memory` and `sofuu.buffer`:
    - `allocAligned(bytes, alignment = 64)` -> `ArrayBuffer`
    - `allocFloat32(length)` -> `Float32Array` (64-byte aligned)
    - `allocUint8(length)` -> `Uint8Array` (64-byte aligned)
    - `viewFloat32(arrayBuffer, byteOffset?, length?)` -> `Float32Array`
    - `viewUint8(arrayBuffer, byteOffset?, length?)` -> `Uint8Array`
    - `detach(arrayBuffer)` -> `boolean`
    - `isDetached(arrayBuffer)` -> `boolean`
    - `addressOf(typedArrayOrBuffer)` -> `string`
    - `isAligned(typedArrayOrBuffer, alignment = 64)` -> `boolean`
  - Integrated zero-copy buffers into SIMD pipeline (`sofuu.ai.dot`, `sofuu.ai.similarity`, `sofuu.ai.l2`).
- **Verification**:
  - `tests/test_zerocopy_buffer.js` verifying 64-byte alignment, dual TypedArray coherent in-place mutation, SIMD dot product and cosine similarity, detachment lifecycle, and 10,000 GC cycle torture test with 0 memory leaks.
  - Full JS test suite 59 passed, 0 failed.
  - Stripped release binary size: **4.8 MB** ($\le 5.0\text{ MB}$ budget ceiling).

### Phase 4: Native AsyncLocalStorage & Agent Context Propagation [COMPLETE]
- **Status**: Implemented, verified, audited.
- **Problem**: Autonomous agent execution across nested async tools requires manual passing of `sessionId`, `traceId`, and `AbortSignal`.
- **Deliverables**:
  - Bound native async context slots in QuickJS runtime (`rt->current_async_context`), promise reactions (`rd->async_context`), and microtask job entries (`e->async_context`).
  - Implemented `JS_GetAsyncContext` and `JS_SetAsyncContext` C-FFI exports in `deps/quickjs` and `crates/sofuu-ffi/src/qjs.rs`.
  - Bound `PromiseHandle.async_context` in `crates/sofuu-core/src/rt/promise.rs` so `sofuu_promise_new` snapshots active context and `sofuu_promise_resolve` / `sofuu_promise_reject` restores it.
  - Linked timer callbacks (`setTimeout`, `setImmediate`, `setInterval`, `sofuu.sleep`) to capture and restore active async context.
  - Implemented `crates/sofuu-core/src/rt/async_context.rs` providing:
    - `AsyncLocalStorage` class with `.run(store, callback, ...args)`, `.getStore()`, `.enterWith(store)`, `.set(keyOrStore, value)`, `.exit(callback, ...args)`, `.disable()`, `static snapshot()`, and `static bind(fn)`.
    - `sofuu.context` namespace with `.run()`, `.getStore()`, `.get()`, `.set()`, `.snapshot()`, `.bind()`, and dynamic getters/setters for `.traceId`, `.sessionId`, and `.signal`.
    - Built-in `AbortController` and `AbortSignal` for cooperative cancellation across agent execution chains.
    - Zero-config built-in module resolution for `node:async_hooks` and `async_hooks`.
- **Verification**:
  - `tests/test_async_context.js`: All 9 verification steps passed covering sync execution, microtasks, promise reactions, timers, native file I/O & sleep, deep nested hierarchies (L1->L2->L3->L2->L1), concurrent interleaved tasks with zero crosstalk, and agent sugar.
  - `tests/test_async_hooks_module.mjs`: ESM module import of `node:async_hooks` verified.
  - Full JS test suite 59 passed, 0 failed, 1 skipped.
  - Stripped release binary size: **4.8 MB** ($\le 5.0\text{ MB}$ budget ceiling).

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
