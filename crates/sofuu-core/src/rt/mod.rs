// rt — the Rust event-loop substrate (PLAN-RUST-MIGRATION M1).
//
// Port of the retired `src/io/loop.c` + `src/io/promises.c`, plus the
// deferred unhandled-rejection tracker that used to live in `engine.c`
// ("its semantics move verbatim" — plan §2 M1).
//
// The remaining C core stays the caller: the same `sofuu_loop_*`,
// `sofuu_promise_*` and `sofuu_flush_jobs` symbols are now provided by this
// module, so engine.c / mod_*.c / http/ / mcp/ link against the Rust
// implementations without any caller-side changes. The rejection tracker
// installs through `sofuu_rt_install_rejection_tracker` /
// `sofuu_rt_report_pending_rejections` under SOFUU_RUST_CORE (c-only keeps
// its own frozen C copy).

#[path = "loop.rs"]
pub mod event_loop; // `loop` is a Rust keyword; the file stays loop.rs per the plan.
pub mod promise;
pub mod timer; // M2: setTimeout/setInterval/clearX/sofuu.sleep
pub mod fs; // M2: sofuu.fs.* (async libuv file I/O)
pub mod process_spawn; // M2: sofuu.spawn + sofuu.exec
pub mod http_client; // M4: fetch() over libcurl (multi ↔ libuv bridge)
pub mod http_sse; // M4: sofuu.SSEParser shell over the Rust parser
pub mod http_server; // M5: createServer/serve over uv_tcp + http-parser
pub mod mcp; // M6: sofuu.mcp (client + stdio server)
pub mod npm; // M7: npm_resolve + package installer (resolver.c)
pub mod cjs; // M7: CommonJS shim (cjs.c)
pub mod ai; // M8: sofuu.ai.* (mod_ai.c + tfidf_embed.c; SIMD kernels stay C)
pub mod model_caps; // per-model capability registry (ctx/max-output/thinking)
pub mod memory; // M9: sofuu.memory/kv/agent (mod_memory.c + mod_kv.c + mod_agent.c + qtsq_adapter.c)
pub mod engine; // M10: engine.c + sofuu.c + repl.c (boot, loader, Sofuu aliases, eval, destroy)
pub mod host_poke; // PLAN-DESKTOP A: thread-safe host→engine wakeup (uv_async poke)
pub mod session_js; // PLAN-DESKTOP C: sync session-mesh primitives for shipped chat.js
pub mod tui; // M10: io/tui.c — the chat alt-screen renderer (tui_* exports)

/// Tests that drive the process-global libuv loop (m1/m2 proofs) must be
/// serialized: the loop is one process-wide instance (same as C), so two
/// tests pumping it from different threads would schedule each other's
/// timers on the wrong context. Acquire this lock around loop usage.
#[cfg(test)]
pub(crate) static TEST_LOOP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
