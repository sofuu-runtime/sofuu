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
| C4 | Live MCP-over-SSE unverified (informational) | Out of scope (no remote server) | — |

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

Both are registered in `tests/run_js_tests.sh`. The synchronous cases live in a shell
script on purpose: the interrupt is uncatchable, so a case that triggers it inside the
harness would kill the harness.

---

## 9. Verification

### Local (macOS, stripped release artifact)

| Gate | Before | After |
|---|---|---|
| `tests/serve_test.js` | 32 passed / 2 failed | **34 / 0** |
| `tests/brain_auth_e2e.sh` | 12 passed / 3 failed | **13 / 0** |
| `tests/chat_no_response_e2e.sh` | all passed | all passed |
| `tests/watchdog_test.js` | — | **4 / 0** |
| `tests/watchdog_sync_e2e.sh` | — | **8 / 0** |
| `scripts/stress/plugin_stress.js` | 25 / 25 | **25 / 25** (7.0 s) |
| contract suite | 51 / 0 / 1 blocked | **51 / 0 / 1 blocked** |
| artifact size | 5,755,312 B | **5,753,736 B** (cap 5,767,168 → 13,432 B spare) |

### Linux (the rented box, freshly built x86_64-musl artifact, 7,356,760 B)

| Gate | Before | After |
|---|---|---|
| watchdog: 600 concurrent | 512 / 600 | **600 / 600** |
| watchdog: sync loop | hangs forever | **`E_TIMEOUT`, rc 1, 2 s** |
| `tests/serve_test.js` | 32 / 2 | **34 / 0** |
| `tests/brain_auth_e2e.sh` | 12 / 3 | **ALL PASSED** |
| `scripts/stress/plugin_stress.js` | 25 / 25 (6.6 s) | **25 / 25 (6.9 s)** |
| contract suite | 46 / 4 / 2 | **46 / 4 / 2** (different rows, see below) |
| `tests/chat_no_response_e2e.sh` | 5 failures | **5 failures** |

Contract rows: T1.2 and T5.8 ("serve came up") are now green; the four that remain are
T2.3, T2.6 (below) and T7.1/T7.2 (C-ABI rows that need `CONTRACT_LIBSOFUU`; no
`libsofuu.so` ships — unchanged, environment-gated).

An earlier Linux run showed `plugin_stress` timing out at 300 s with a stray
`sofuu serve` from a previous test still holding a port. On a clean box the same binary
finishes in 6.9 s. Stray processes after a killed test are a known nuisance, not a hang.

---

## 10. Still open: the Linux-only chat no-response cluster

Verifying P1/C1/C2 turned up something narrower than the first guess, and the first guess was
wrong, so both are recorded here.

### What was measured

| Step (on the box, mock running) | Result |
|---|---|
| plain JSON `POST` (429 path) | `code=429 connect=0.0001s` |
| `POST` the SSE / errframe path | `code=200 total=0.0003s` |
| plain `POST` again, after the SSE | `code=429 connect=0.0001s` — still healthy |
| mock process | alive (`STAT Ssl`), nothing in its log past the last request |

Conclusion: a chunked (SSE) response does **not** break the built-in server. The listener is
never closed by `res_start_chunked` and the server keeps answering plain requests
afterwards. The client is what stops reaching it.

### What actually reproduces

In `tests/chat_no_response_e2e.sh`, turn 1 (429) prints the right error, turn 2 (errframe)
already prints `Couldn't connect to server`, and turns 3-6 never reach the mock. On macOS
all six modes pass with the same binary and the same script. Two candidates remain, and
telling them apart needs an strace or a client-side URL log:

1. after the first hard failure the chat switches to another endpoint (a failover that
   rewrites `base_url`), so `Couldn't connect` is literally true — for a different
   host, which also explains why the mock log simply stops;
2. the `MODESILENT` turn (a server that never responds) wedges the connection handling
   inside the runtime's client on Linux only.

Either way the misleading part is on the chat side: it reports a transport failure for a
turn whose response it actually received. That message should say what happened to the
stream (empty, stalled, in-stream error) instead of `Couldn't connect to server`, and the
same is true of the `MODEMPTY`/`MODESILENT` paths.

### Impact

This one cluster is what keeps `tests/chat_no_response_e2e.sh` (5 assertions) and contract
T2.3/T2.6 red **on Linux only**. It is separate from the deadline/watchdog work and was not
attempted here: fixing it means chasing the chat client's endpoint state machine (or the
socket pool in `rt/event_loop.rs`), which is its own investigation.

Everything else in this file is done and verified on both platforms.

## 11. Files touched

```
crates/sofuu-core/src/rt/deadline.rs          NEW  interrupt deadline + classification
crates/sofuu-core/src/rt/mod.rs                mod deadline;
crates/sofuu-core/src/rt/engine.rs             register sofuu.host; classify uncaught interrupts
crates/sofuu-core/src/rt/ai.rs                 stream_status_error (status >= 400 wins)
crates/sofuu-core/src/modules/process.rs       cli_install_sigint_stop
crates/sofuu-core/src/main.rs                  install SIGINT stop
src/js/plugins.js                              shared ticker, host arm scheduler, grace
src/js/serve.js                                /healthz, /livez, authenticated /health
src/js/plugins.bc, src/js/serve.bc             regenerated bytecode snapshots
tests/watchdog_test.js                         NEW
tests/watchdog_sync_e2e.sh                     NEW
tests/run_js_tests.sh                          register both
tests/serve_test.js                            H2 rewritten
tests/brain_auth_e2e.sh                        401 body assertion corrected
scripts/contract/harness/t12_capabilities_parity.js, t58_error_parity.js   probe /healthz
docs/HTTP-SERVER.md, docs/SPEC-RUNTIME-CONTRACT.md                         auth contract
```

No new dependencies. `cargo fmt` was not run (the tree has pre-existing formatting drift).
The shipped bytecode snapshots must be regenerated with `scripts/gen-driver-bc.sh` after any
`src/js/*.js` change — forgetting that step silently ships the old driver, which is how the
first Linux run of these fixes was verified against stale bytecode and had to be repeated.
