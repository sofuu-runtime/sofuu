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
pub mod local_tools; // C1: local-model tool calling (prompt protocol + stream parser)
pub mod wal; // C2: durable atomic publish + write-ahead journal (brain/KV persistence)
#[cfg(has_qtsq)]
pub mod storage; // Phase 3: Native QTSQ Structured Storage Driver (sofuu.storage)
#[cfg(has_qtsq)]
pub mod brain_store; // B3: segmented brain storage (manifest + bounded segments)
pub mod brain_key; // B4: the machine-local brain key (key file + legacy fallback)
pub mod model_caps; // per-model capability registry (ctx/max-output/thinking)
pub mod model_caps_discovered; // caps harvested from endpoint model listings (provider-agnostic)
pub mod memory; // M9: sofuu.memory/kv/agent (mod_memory.c + mod_kv.c + mod_agent.c + qtsq_adapter.c)
pub mod zerocopy; // Phase 3: Zero-Copy TypedArray & Native Memory Bridge
pub mod view_guard; // R7.P2: the validated JS-view → native-pointer window (bounds/alignment/dim)
pub mod async_context; // Phase 4: Native AsyncLocalStorage & Agent Context Propagation
pub mod sandbox; // Phase 5: Deterministic Agent Sandboxing & Fuel Meters
pub mod policy; // S4 (R7): one policy evaluator for tools/shell/fs/net/MCP
pub mod events; // P2 (R6): the unified event envelope + session event log
pub mod observability; // OBSERVABILITY: run logs, checkpoints, crash cards (reads `events`)
pub mod async_contract; // P2 (R2): one error taxonomy, nested deadlines, queue bounds
pub mod crypto; // sofuu.crypto: SHA-256/base64url + RS256 JWT verification
pub mod session_doc; // P3 (R4): the §5.2 session document — one export/import format
pub mod syntax; // Phase 5 (agent plan): sofuu.parse - parse-only gate for file edits
pub mod bytecode; // Phase 6 (agent plan): sofuu.aotEval - AOT bytecode snapshots for agent scripts
pub mod engine; // M10: engine.c + sofuu.c + repl.c (boot, loader, Sofuu aliases, eval, destroy)
pub mod host_poke; // PLAN-DESKTOP A: thread-safe host→engine wakeup (uv_async poke)
pub mod deadline; // sofuu.host.armDeadline/disarmDeadline: JS-callable interrupt deadline
pub mod session_js; // PLAN-DESKTOP C: sync session-mesh primitives for shipped chat.js
pub mod tui; // M10: io/tui.c — the chat alt-screen renderer (tui_* exports)

/// End the process the way a default SIGPIPE disposition would have: quietly,
/// with status 141 and no output.
///
/// Only for a BROKEN PIPE on our OWN output streams (`sofuu … | head`) — never
/// for a socket. SIGPIPE is ignored process-wide (see `main`), because a peer
/// hanging up mid-response is routine for `run`/`serve` and must not be able to
/// kill the process: libuv's writev() raises SIGPIPE, and with the default
/// disposition the first abandoned SSE response killed the server outright.
/// The EPIPE that libuv returns is handled there; this is the other half — an
/// EPIPE from our own stdout/stderr still ends the process, as it always has.
pub fn broken_pipe_exit() -> ! {
    #[cfg(unix)]
    unsafe {
        /* Restore the conventional disposition for THIS exit alone, so the
         * shell still sees a SIGPIPE death ($? = 141, WIFSIGNALED) rather
         * than an invented status. raise() cannot return. */
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
        libc::raise(libc::SIGPIPE);
    }
    std::process::exit(141)
}

/// Tests that drive the process-global libuv loop (m1/m2 proofs) must be
/// serialized: the loop is one process-wide instance (same as C), so two
/// tests pumping it from different threads would schedule each other's
/// timers on the wrong context. Acquire this lock around loop usage.
#[cfg(test)]
pub(crate) static TEST_LOOP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire [`TEST_LOOP_LOCK`], tolerating poisoning.
///
/// A test that panics while holding this lock poisons it, and every LATER
/// test that takes it then dies with `PoisonError` — turning one real
/// failure into a cascade that hides the actual bug. That happened for real:
/// a single arch-dependent golden vector made ~28 unrelated http/mcp/
/// spawn tests report `PoisonError` in CI. The lock only serializes
/// execution; the data it guards (the libuv loop) carries nothing between
/// tests, so recovering the guard is safe and keeps the next failure
/// legible.
#[cfg(test)]
pub(crate) fn test_loop_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOOP_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
