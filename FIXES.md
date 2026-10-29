# Sofuu — what the AWS extreme-stress run found, and what was fixed

Baseline: **0.3.0-beta**, tested on the rented box (2 vCPU / 3.9 GB, Ubuntu, x86_64-musl)
against commit **4a903dcbc** with the Linux artifact. HEAD at the time of the fixes is
**028878309** (`sandbox: unclamped budgets, host bridges, TS/JSX ingest and console routing`),
with **dd14f7ca9** (observability) in between. Every fix below is on top of that HEAD.

The issue register (16 items) is §7 of `AWS_STRESS_REPORT.pdf`. This file records the
disposition of each one: what changed, where, and the evidence that it now behaves.

---

## 0. Disposition table

| # | Finding (severity) | Status | Where |
|---|---|---|---|
| R1 | A plugin that never yields hangs the process forever (critical) | **Fixed** | `rt/deadline.rs` (new), `rt/engine.rs`, `src/js/plugins.js` |
| R2 | A synchronous 5 s burn returns OK under a 300 ms deadline (critical) | **Fixed** | same |
| R3 | Work kept running after a reported `E_TIMEOUT` (high) | **Fixed** | `src/js/plugins.js` |
| R4 | 512 in-flight calls was a hard ceiling — the 513th+ failed (high) | **Fixed** | `src/js/plugins.js` (shared ticker) |
| R5 | After a 600-call burst, later calls failed on timer exhaustion (high) | **Fixed** | same |
| R6 | SIGINT did not stop a blocked runtime — it kept pumping output (high) | **Fixed** | `modules/process.rs`, `main.rs` |
| R8 | Ctrl-C during a long tool call did nothing until the tool returned (medium) | **Fixed** | same |
| R7 | `/health` answered without the token while two shipped tests disagreed (medium) | **Fixed (contract change)** | `src/js/serve.js`, docs, tests |
| P1 | A streamed HTTP 4xx/5xx degraded to a transport error, hiding the status (medium) | **Partially fixed** — status is now reported; see §5 and §10 | `rt/ai.rs` |
| P2 | M4 "timeline past cap" looked like a payload-cap breach (low) | **Not a defect** — reclassified, §6 | — |
| R9 | `help` did not list `logs`/`resume` (low) | **Already fixed upstream** (`dd14f7ca9`) | — |
| C1 | T2.3 tool-subprocess case red on Linux only (medium) | **Open** — chat client / mock-server cluster, §10 | — |
| C2 | T2.6 error-parity case red on Linux only (medium) | **Open** — same cluster, §10 | — |
| C3 | No `libsofuu.so` shipped, so C-ABI rows are blocked (informational) | Out of scope (no artifact) | — |
| C4 | Live MCP-over-SSE unverified (informational) | **Fixed** — local SSE fixture + e2e test | `tests/mcp_sse_e2e.sh` |
| C5 | `fetch().json()` failed on valid bodies ("unexpected data at the end") | **Fixed** — §12 | `rt/http_client.rs`, `rt/ai.rs` |
| C6 | a server created inside a function was garbage-collected, so its listener never accepted | **Fixed** — §13 | `rt/http_server.rs` |

Net: **10 of 16 closed**, one partially (P1), three reclassified or out of scope, and
**one Linux-only cluster** narrowing P1/C1/C2 (§10).

---

## 1. The root cause behind R1/R2/R3: the watchdog was a timer, not an interrupter

`defineTool({timeoutMs})` armed a JavaScript `setTimeout`. That is enough to *report* a
timeout, and nothing more:

* A tool that never returns never reaches the `setTimeout` callback — the loop is the tool.
  The process hung (R1), and no timer could ever fire.
* A tool that burns 5 s of CPU under a 300 ms deadline returned `ok` (R2): the timer fired
  late and the call had already produced a result.
* After reporting `E_TIMEOUT` the abandoned promise kept running to completion (R3).

A timer cannot preempt JavaScript. Only the engine can, so the fix has to live host-side.

### 1.1 New: a process-global interrupt deadline

`crates/sofuu-core/src/rt/deadline.rs` (new, 268 lines) keeps one deadline for the whole
process and drives QuickJS's own interrupt mechanism:

* `sofuu.host.armDeadline(ms)` — installs the interrupt handler and arms `now + ms`.
* `sofuu.host.disarmDeadline()` — removes the arm; at zero arms the handler is uninstalled
  and the previous interrupt counter is restored, so a clean `eval` behaves exactly as before.
* Nested arms keep the **earliest** deadline (a short tool deadline is never pushed out by a
  longer outer one), and a new arm after a fired one supersedes it.
* Limits: 16 concurrent arms, deadline granularity is the QuickJS poll batch (10,000
  instructions), maximum arm 24 h.

Registered for every context in `engine_register_builtins` (`rt/engine.rs:447`), so
`run`, `eval`, `repl`, `chat` and `serve` all have it.

### 1.2 Why the JavaScript side cannot translate the error itself

The vendored QuickJS (`deps/quickjs/quickjs.c`) raises the deadline as
`JS_ThrowInternalError(ctx, "interrupted")` and then calls
`JS_SetUncatchableError(ctx, TRUE)` on it. **The interrupt is uncatchable**: no `catch`,
no `finally`, no promise rejection handler runs. Any design that expected
`runWithWatchdog` to map `InternalError: interrupted` to `E_TIMEOUT` in JavaScript is
unimplementable — the wrapper is never entered again.

That is also why the synchronous half of the fix is observed at the *process* boundary, and
why the in-process test cannot contain a synchronous case.

### 1.3 Host-side classification

`deadline.rs` records whether an arm actually fired (`FIRED_US`/`FIRED_MS`) *before* the
handler returns 1, and exposes `classify_interrupted(msg)`. `report_uncaught`
(`rt/engine.rs:1255`) — the single choke point for `run`/`eval`/`repl`/chat-turn/serve
failures — uses it to print a typed line instead of an internal engine error:

```
[sofuu] E_TIMEOUT: plugin deadline exceeded — the call was interrupted after 60 ms
        (a synchronous call cannot be cancelled; it was aborted)
```

The fired-arm guard matters: an unrelated `new InternalError("interrupted by peer")` keeps
printing as `UncaughtException`. Verified.

### 1.4 The JavaScript side

`src/js/plugins.js` now arms the host deadline around every tool call
(`hostArmFor`/`hostDisarmFor`; one arm covers all concurrent entries, the earliest deadline
wins, and the set is re-armed with `disarmDeadline`+`armDeadline` whenever it changes) and
keeps a short **grace** arm (`min(max(ms, 1000), 5000)`) after a timeout, so a tool that
ignores its abort signal is interrupted instead of running to completion (R3).

Because the clock that fires is now the engine, a stalled `await` and a stalled loop are
finally the same problem.

### Two defects that only the stress run could expose

Running the 25-case plugin stress on the 2-vCPU box kept dying at one random case with
`UnhandledPromiseRejection: InternalError: interrupted`, while the same file was green on the Mac.
Both causes were in the new bookkeeping, and both matter:

* **The fired deadline stayed armed.** QuickJS polls the interrupt handler on every back-edge, so
  an expired deadline kept matching and every later evaluation in the process died with
  `interrupted` — the runtime never recovered. The arm is now one-shot: it disarms itself as it
  fires, and `FIRED_US` still tells the host how to classify the error it caused.
* **Arms outlived their call on error paths.** `hostDisarmFor` ran only on success, so a call that
  timed out (or threw) left its entry in `wdArms`; the next refresh then armed that stale entry as
  "a few milliseconds from now" and interrupted whatever unrelated work happened to be running.
  Every exit path now drops the arm.

The second one explains the random victim: the stress deliberately runs a 60 ms tool that hangs,
several times, and the interrupt landed on the next case instead of on the hang.

---

## 2. R4/R5: one timer per call was the real concurrency ceiling

Each plugin call created a `setTimeout` and a `clearTimeout`. QuickJS/libuv caps timers, so
the 513th concurrent call failed with `E_TOOL`, and the table stayed poisoned afterwards.

`plugins.js` now runs **one** unref'd ticker (`WD_TICK_MS = 15`) over a plain array of
pending entries (`wdAdd`/`wdDrop`/`wdPump`). No per-call timer exists any more, so the
ceiling is gone and so is the poisoning.

| Case | Before (Linux) | After |
|---|---|---|
| 600 concurrent calls, 5 s deadline | 512 ok / 88 `E_TOOL` | **600 ok / 0** |
| a call after the burst | failed (`E_TOOL`) | **ok** |

---

## 3. R6/R8: Ctrl-C now stops the runtime

`modules/process.rs:335` installs the SIGINT action (`cli_install_sigint_stop`, called from
`main.rs:2314`). It only sets a flag; a 50 ms watchdog `_exit(130)`s if the flag is still
set 200 ms later, whether the main thread is blocked in a syscall or pumping output. SIGTERM
is untouched (`main.rs` still exits 143 on its own path).

Measured: `rc 130` in under a second, both while blocked and while streaming.

---

## 4. R7: `/health` obeys the token; `/healthz` and `/livez` are the probes

`tests/serve_test.js` asserted an unauthenticated `/health`; `tests/brain_auth_e2e.sh`
asserted 401 for the same request. Both cannot be right, and the bigger problem was the
contract itself: one route on an authenticated server answered without the token.

New contract (`src/js/serve.js`, `docs/HTTP-SERVER.md`):

* `GET /healthz`, `GET /livez` — public, data-free (`{"ok":true,"status":"ok"}`), for probes.
* `GET /health` — requires `Authorization: Bearer <token>` when a token is configured;
  returns `{ok,status,uptime_ms,version}`. Unauthenticated: the standard OpenAI 401 body.

Ripples fixed with it:

* `tests/serve_test.js` H2 rewritten (public `/healthz` **and** 401 `/health` without a token).
* `tests/brain_auth_e2e.sh` — its "401 body" assertion looked for a literal `"unauthorized"`
  that no route has ever emitted (it could never pass, on any platform); it now asserts the
  documented `authentication_error`/`invalid_api_key` envelope.
* `scripts/contract/harness/t12_capabilities_parity.js` and `t58_error_parity.js` probed
  `/health` to decide "serve came up"; they now probe `/healthz`.
* `docs/SPEC-RUNTIME-CONTRACT.md` deviation row 8 marked resolved.

---

## 5. P1: a streamed HTTP error is reported as an HTTP error

`rt/ai.rs`: a transport-level failure (curl write error, closed connection mid-body) used to
mask a perfectly good HTTP status. The error path now reads `CURLINFO_RESPONSE_CODE` and a
received status of 400 or above wins over the curl wording (`stream_status_error`,
`ai.rs:2533`, used at `ai.rs:3248` and in the clean-4xx branch at `ai.rs:3285` — about 35
lines of duplicated status handling collapsed into one builder).

Effect on Linux: the first failing chat turn now prints
`✗ HTTP 429 — Provider returned error (rate limited …)` instead of a bare
`Couldn't connect to server`.

The *suite* is still red — five assertions — because of the server defect in §10, not
because of the mapping.

---

## 6. P2: "timeline past cap" was not a payload-cap breach

M4 asserted four things. The cap assertion held exactly (`withPayload === 512`), and 5,881
events had been emitted; only `stopped === "budget_steps"` differed — on a 2 vCPU box the
120 s wall budget ends the run before the step budget does. A timing-sensitive assertion in
the test, not a defect in the runtime. No code change.

---

## 7. R9: stale finding

`help` listing only 19 of 21 commands was true of the Linux artifact built from
`4a903dcbc`. `dd14f7ca9` already regenerated the help printer; every advertised command
dispatches at HEAD. No change needed.

---

## 8. Tests added

| File | Covers |
|---|---|
| `tests/watchdog_test.js` (new) | async stall → `E_TIMEOUT` promptly; 600 concurrent calls; no poisoning after the burst |
| `tests/watchdog_sync_e2e.sh` (new) | process-level: sync infinite loop aborts (2 s, rc≠0), sync burn honours a 150 ms deadline, nested disarm keeps the earliest arm, an unrelated `InternalError` is untouched, a clean eval still prints 42 |
| `tests/fetch_json_body_test.js` (new) | `json()` parses its own body: big body, an unconsumed 404 before the call, 48 body lengths, `text()` and `json()` agreement — all against a local server, so it runs offline |
| `tests/mcp_sse_server.js` + `mcp_sse_client.js` + `mcp_sse_e2e.sh` (new) | legacy MCP HTTP+SSE end to end: mount from `mcp.json`, endpoint frame, notifications, two tools, an `isError` tool, disconnect, orphan check (19 assertions) |

All are registered in `tests/run_js_tests.sh`. The synchronous deadline cases live in a shell
script on purpose: the interrupt is uncatchable, so a case that triggers it inside the
harness would kill the harness.

---

## 9. Verification

Final artifacts: macOS stripped release 5,753,784 B (cap 5,767,168), Linux x86_64-musl
static-pie 7,362,456 B. Both were built from the same tree.

| Gate | macOS | Linux box |
|---|---|---|
| full JS suite (`tests/run_js_tests.sh`) | **exit 0** | **exit 0** |
| `serve_test.js` | 34 / 0 | 34 / 0 |
| `brain_auth_e2e.sh` | 13 / 0 | ALL PASSED |
| `chat_no_response_e2e.sh` | ALL PASSED | **ALL PASSED** (was 5 failures) |
| `watchdog_test.js` | 4 / 0 | 4 / 0 |
| `watchdog_sync_e2e.sh` | 8 / 0 | 0 failures |
| `fetch_json_body_test.js` (new) | 5 / 0 | 5 / 0 |
| `mcp_sse_e2e.sh` (new) | 19 / 0 | 19 / 0 |
| `mcp_client_test.js` (stdio regression) | 9 / 0 | 9 / 0 |
| `abi_script_test.js` | 14 / 0 | 12 / 0 (dylib rows skip) |
| `plugin_stress.js` | 25 / 25 (7.3 s) | 25 / 25 (6.9 s) |
| contract suite | 51 / 0 / 1 blocked | **51 / 0 / 1 blocked** |
| `priority1_test.js` | 15 / 0 | 0 failures (HTTPS rows skipped) |

`T4.2` is the only blocked row on both platforms: it needs macOS **and** Android runners.

Two differences are the environment, not the code, and both now say so instead of going red:

* the Linux artifact links a static, TLS-less curl, so `https://` is unavailable there. The
  fetch contract is covered offline by `fetch_json_body_test.js`, and `priority1_test.js`
  skips its httpbin block with a printed reason.
* `agent_test.js`'s M4 accepts `budget_wall` as well as `budget_steps`: on 2 vCPU the wall
  clock runs out before the step counter, and the property under test — events preserved
  past the payload cap — holds either way.

For comparison, the same gates before this round: serve 32/2 → 34/0, brain_auth 12/3 → 13/0,
contract on Linux 45/5/2 → 51/0/1, `chat_no_response_e2e.sh` 5 failures → 0.

---

## 10. The Linux-only chat no-response cluster: SIGPIPE

The first two theories were wrong and are recorded so nobody re-derives them: it was not a chunked
response killing the listener, and it was not the chat client wedging. It was the process dying.

### What actually happened

`main.rs` restored SIGPIPE to `SIG_DFL` process-wide so that `sofuu … | head` exits quietly
instead of ignoring the broken pipe. That disposition also applies to servers: when the chat
client aborts an SSE stream (it returns 0 from the write callback on an in-stream error frame,
`rt/ai.rs`), the server's next `writev()` raises SIGPIPE, and libuv does not mask it. `strace` on
the mock on the Linux box:

```
write(16, "HTTP/1.1 200 OK…", 99)      = 99
write(16, "7e\r\ndata: {\"error\"…", 132) = 132
write(16, "e\r\ndata: [DONE]…", 19)    = 19
write(16, "0\r\n\r\n", 5)             = -1 EPIPE
--- SIGPIPE --- +++ killed by SIGPIPE +++
```

So the mock exits 141, every later connection is *genuinely* refused, and `Couldn't connect to
server` was literally true. macOS passes the same test because the whole response is flushed
before the client's RST arrives — the abort happens on both platforms, only the post-RST write
differs. That is why the failure looked like a server bug on Linux only.

### The fix

* `main.rs` keeps Rust's `SIG_IGN`, and a panic hook (`quiet_broken_pipe_exit`) turns std's
  "failed printing to stdout" panic back into the old silent status-141 exit, so piping into
  `head` still behaves.
* `rt/mod.rs` gains `broken_pipe_exit()`, which restores `SIG_DFL` and raises SIGPIPE for that
  one exit only — the signal is never inherited by a server.
* `modules/console.rs` routes the JS console writer's `BrokenPipe` through it (it used to swallow
  the error, which would have made `| head` spin instead of exit).

### Two contract rows that were hiding behind it

* **T2.6** — a chat turn that dies emitted no terminal event (success emits `done`, and the agent's
  own `stop` was swallowed at the step bridge), so a host UI keyed on a terminal sat on
  "in progress" forever. `src/js/chat.js` now emits one `stop{reason:'error'}` before rethrowing.
  `scripts/contract/cases.tsv` pins `SOFUU_STREAM_RETRY_DELAYS=300,300` for that row: the
  "Couldn't connect" text matches the transient-retry regex on Linux but not on macOS, so the
  shipping 6-step ladder ran ~118 s and blew the harness's 35 s watchdog.
* **T2.3** — the harness's bash tool ran `/bin/sh -c`, which is dash on Linux and has no
  `exec -a`, so the marker command exited 127 and no child existed to be live. The command is now
  spelled `bash -c 'exec -a …'`. No assertion was weakened.

### Evidence

* `tests/chat_no_response_e2e.sh`: **ALL PASSED** on macOS and on the box (10/10 checks), where it
  had been 5 failures.
* Linux contract went from 45 PASS / 5 FAIL / 2 BLOCKED (no C ABI) to **49 PASS / 2 FAIL /
  1 BLOCKED** before the JS fix, and T2.3/T2.6 flip green with it.
* `serve_test` stays 34/0 on both platforms.

---

## 11. Found while fixing P1: `Response.json()` over-read the body buffer

`fetch()`'s `json()` handed QuickJS a bare slice of the response buffer:

```rust
let parsed = JS_ParseJSON(ctx, src.as_ptr(), src.len(), c"<fetch response>".as_ptr());
```

The vendored tokenizer reads **one byte past `buf_len`** to decide end of input
(`js_parse_init` keeps the caller's buffer and never terminates it), so whether a perfectly valid
body parsed depended on whatever heap byte happened to follow it. Against httpbin: a 384-byte echo
failed with `unexpected data at the end` while the identical bytes plus a NUL parsed fine — and
`text()` was always correct, which is what made it look like a network problem.

Fixed with `parse_json_bytes()` in `rt/http_client.rs`, a NUL-terminating helper now used by
`json()`. The same pattern existed twice in `rt/ai.rs` (`<ai_response>`, `<embed_response>`:
`CString::new(body)` together with `body.len()`, which over-reads whenever the body contains an
interior NUL, because `CString::new` then yields an empty buffer) — both go through the helper now.

Evidence: a six-body matrix (7 to 46 bytes, one and two keys) that used to fail for any body with
two keys now parses every one. `tests/fetch_json_body_test.js` (new, local server, no network) pins
it with five assertions including the original trigger — an unconsumed 404 response followed by a
JSON POST — and runs on both platforms.

---

## 12. A server created inside a function never listened

This came out of writing that regression test, and it is the kind of defect that makes a runtime
feel haunted: the same code works at top level and silently does nothing inside a helper, an IIFE
or a module-scope factory.

```js
(function () {
  var s = sofuu.http.createServer(handler);
  s.listen(18837, '127.0.0.1');   // prints 'listening', nothing binds
})();
```

`listen()` returned the server and `do_listen` succeeded, but no socket ever appeared: `lsof`
showed nothing on the port and every connect was refused, with no error raised. The `Server` struct
holds the handler `JSValue`, but nothing held the *JS server object*, so when the IIFE returned the
last reference was gone, QuickJS ran the class finalizer, and `srv_finalizer` closed the listener it
had just opened. Node does not work that way — a listening server is a live root.

Fix: `Server` now carries an `anchor: JSValue` root, taken in `server.listen()` (idempotent, so a
second `listen` does not double-hold) and released with the object at teardown.

Anchoring alone was not enough, and the first attempt proved it: a rooted server keeps the event
loop alive exactly like Node's, so scripts that start a fixture and finish their assertions hung
the suite forever. The listener is therefore **unref'd by default** — a finished script still exits,
which is what every embedder and every test in this tree relies on — with `server.ref()` and
`server.unref()` exposed for Node's semantics. `sofuu serve` calls `ref()`, and so do the mock
provider fixtures, which are the scripts whose whole job is to stay up.

Evidence: the program above now binds and answers 200; the MCP-over-SSE fixture and the C5 test
both build their servers inside an IIFE, and both pass on macOS and Linux.

---

## 13. The C ABI: a library that could not exist, and the rows it hid

`T5.5`, `T7.1`, `T7.2` and `T7.3` were red because no `libsofuu` shared library was ever produced,
for three independent reasons:

1. `crates/sofuu-capi` asks for `cdylib`, which rustc **drops for `*-linux-musl`**
   (`crt_static_allows_dylibs=false`), and both the `Makefile` target and `scripts/dist/linux.sh`
   swallowed the missing file with `|| true`.
2. The only possible Linux cdylib (`x86_64-unknown-linux-gnu`) failed to load:
   `undefined symbol: qtsq_decompress`. A static binary tolerates dangling references because the
   member is never pulled in; a shared object must resolve every relocation. Caused by
   `scripts/cross/build_qtsq_cross.sh` building a reduced POSIX QTSQ subset instead of the full set
   the shipped archives already contained — fixed there.
3. macOS cannot read ELF, and `scripts/check_abi.sh` died silently with rc=1 under `set -e` because
   its `nm` pipeline aborted before the "no symbols" branch — it now has a stdlib ELF `.dynsym`
   reader and verifies a `.so` from either host.

Also fixed on the way: `sofuu_rt_suppress_rejections` was exported by an accidental `#[no_mangle]`
on a private Rust helper (`rt/promise.rs`), which the ABI gate correctly flagged as an
unacknowledged addition. The attribute is gone (the function keeps its Rust visibility); the
baseline was *not* extended, because `include/sofuu_embed.h` never declared it.

Evidence: ABI gate green on both platforms (34/34 symbols); a real C host program links and calls
`sofuu_rt_call` on both (`crypto.sha256` → `ba7816bf…15ad`); `T5.5`, `T7.1`, `T7.2`, `T7.3` PASS.
The dylib also got a relocatable `@rpath/libsofuu.dylib` id. **Limits:** the Linux library is
glibc, not musl (rustc cannot emit a musl cdylib — musl hosts use `dist/libsofuu.a`), and its curl
is the static TLS-less build, so HTTPS from it is unavailable there, same as the musl binary.

---

## 14. Files touched

```
crates/sofuu-core/src/rt/deadline.rs          NEW  interrupt deadline + classification
crates/sofuu-core/src/rt/mod.rs                mod deadline; broken_pipe_exit
crates/sofuu-core/src/rt/engine.rs             register sofuu.host; classify uncaught interrupts
crates/sofuu-core/src/rt/ai.rs                 stream_status_error; NUL-terminated JSON parses
crates/sofuu-core/src/rt/http_client.rs        parse_json_bytes (C5)
crates/sofuu-core/src/rt/http_server.rs        server anchor + ref()/unref() (C6)
crates/sofuu-core/src/rt/promise.rs            drop the accidental #[no_mangle] export
crates/sofuu-core/src/main.rs                  SIGINT stop; quiet broken-pipe exit
crates/sofuu-core/src/modules/process.rs       cli_install_sigint_stop
crates/sofuu-core/src/modules/console.rs       BrokenPipe -> broken_pipe_exit
src/js/plugins.js                              shared ticker, host arm scheduler, grace
src/js/serve.js                                /healthz, /livez, authenticated /health
src/js/tools.js                                MCP legacy HTTP+SSE transport, url configs
src/js/chat.js                                 terminal stop{reason:'error'} on a dead turn
src/js/*.bc, src/js/*.bc.gz                    regenerated snapshots (plugins, serve, tools, chat)
tests/watchdog_test.js                         NEW  async stall, 600 concurrent, no poisoning
tests/watchdog_sync_e2e.sh                     NEW  process-level sync deadlines
tests/fetch_json_body_test.js                  NEW  json() body regression + IIFE server
tests/mcp_sse_server.js, mcp_sse_client.js, mcp_sse_e2e.sh   NEW  MCP over SSE, 19 assertions
tests/serve_test.js, brain_auth_e2e.sh, run_js_tests.sh      H2, 401 body, registration
scripts/contract/harness/t12,t58                       probe /healthz
scripts/contract/harness/t23_cancel_tool.js            bash -c 'exec -a'
scripts/contract/cases.tsv                             T2.6 retry ladder
scripts/cross/build_qtsq_cross.sh, build_linux_capi_so.sh (NEW), check_abi.sh, Makefile
docs/HTTP-SERVER.md, docs/SPEC-RUNTIME-CONTRACT.md, docs/MCP.md
```

No new dependencies. `cargo fmt` was not run (the tree has pre-existing formatting drift). The
shipped bytecode snapshots must be regenerated with `scripts/gen-driver-bc.sh` after any
`src/js/*.js` change — forgetting that step silently ships the old driver, which is how the first
Linux run of these fixes was verified against stale bytecode and had to be repeated.
