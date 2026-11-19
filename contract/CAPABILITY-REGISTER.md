# Capability register — real execution status

Updated 2026-11-16. This is the tracked (clean-checkout-available) part of the
real-execution register. The working narrative and dated receipts live in
`docs/plans/REAL-EXECUTION-STATUS.md` (local-only, Git-ignored in this
checkout); the acceptance criteria live in
`docs/plans/PLAN-REAL-EXECUTION-REMEDIATION.md`.

`contract/` is tracked; `docs/` is not. This file therefore carries the part
of the register that must survive a clean checkout.

## What a status means

| Status | Meaning |
|---|---|
| `real` | A configured, authorized backend performed the operation on the named artifact and the receipt is recorded. |
| `pure` | A deterministic in-process computation with no external effect. It is a real implementation, but it is not inference, not a tool call and not a network exchange. |
| `physical-test` | An explicitly selected, labelled robotics/drone simulator. It is never evidence about real hardware. |
| `blocked` | The scenario needs a real resource that is absent here. The gap is recorded with its reason; it is not a pass and not a failure of the product. |
| `unverified` | Not exercised. No claim is made either way. |

A status is never inferred from a label in the source. It names the artifact,
the command and the receipt.

## Evidence rules

- A real-model claim needs an existing local GGUF (`~/Downloads/*.gguf`). No
  weights are ever downloaded (`AGENTS.md`).
- `contract/fixtures/sofuu-fixture.gguf` and `~/llama.cpp/models/ggml-vocab-*.gguf`
  are fixtures. They may validate parsers and pure computations. They are never
  evidence of inference, and `scripts/text_gate.py` refuses them by default
  (exit 2, `BLOCKED:`) unless `--allow-synthetic-fixture` is passed, which
  stamps the receipt `inference_evidence: false`.
- A real-tool claim needs the tool to have actually run against a real
  resource (a temporary file, a real MCP server, a real store) and its
  independent receipt to be recorded.
- A real-hardware claim needs a physical device. None was attached for any
  receipt in this register, so every hardware row is `unverified`.
- A receipt applies to the artifact and configuration named in it. It does not
  transfer to another model, platform, build or release.

## Division register

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D01 substrate, event loop | Timers, cancellation, process and network execution | `real` (measured configuration) | `scripts/bench/event_loop_latency.js` drives the real local server and a real local model; `make event-loop-check` requires `SOFUU_LOOP_MODEL` and blocks when it is unset. Not a hard real-time guarantee and not a release-wide figure. |
| D02 memory | `memory=ephemeral` (per-run CMA), `memory=durable`, `memory=file:<path>` | `real` | `scripts/test-kaze-memory-real.py` and `scripts/test-kaze-memory-durable.py`: write in one process, exit, recall in a second process; store isolation from the ephemeral store; policy denial; bytecode and standalone parity. The separate QTSQ check covers write, reopen, isolation, read-only and a real filesystem failure. |
| D03 agents | Pure-Kaze routine: real MCP tool read → real model request → model answer containing the file's fresh marker | `real` | `scripts/test-kaze-local-gguf-real.py`: 3/3 independent runs on both the source path (`sofuu.kaze.runAsync`) and the compiled standalone path, using the pre-existing `SmolLM2-360M-Instruct-Q8_0.gguf` on Metal, with independent MCP and model receipts. `scripts/test-kaze-real-ai.py` and `scripts/test-kaze-cli-real-ai.py` cover the remote-endpoint variants. |
| D03 agents | Agent loop fallback: canned final answer when the provider or a tool is unavailable | `real` (removed) | The Rust provider registry no longer contains a `"mock"` provider; the JS runtime layers and the C-ABI test harness no longer inject one. A refused capability is a typed error, and an agent with no reachable backend fails closed. |
| D04 tools, MCP, policy | Registered tool execution, MCP stdio/HTTP, allow/deny, SSRF and allowlist boundaries | `real` | `scripts/test-real-mcp-file.py`: an actual file-reading MCP server, policy denial, deadlines, a lost response, reconnect policy and no automatic replay of a possibly completed call. The contract T5/T7 cases now run against the same real servers. |
| D04 tools, MCP, policy | A denied tool receiving a request anyway | `real` (absent) | Dispatch is gated before the backend is reached; the denial cases assert zero dispatch, not only a denial message. |
| D05 Kaze and compiler | Host separation, per-operation backend resolution, policy enforcement, configuration propagation | `real` | `crates/sofuu-core/src/domains/kaze/caps.rs` splits the runtime host from the physical test host: the runtime host has no path where `ai`, `tool`, `memory` or `alert` succeeds without a configured backend, and the physical host answers only the physical families it was explicitly selected for. `crates/sofuu-core/src/compiler.rs` carries the backend behind each grant, the policy and the configuration into the generated bootstrap. `scripts/test-kaze-fail-closed.py`, `scripts/test-kaze-js-boundaries.py`, `make kaze-drift-check`. |
| D05 Kaze and compiler | A bare or empty `--cap` meaning simulation | `real` (absent) | Simulation needs an explicit `drone=sim` / `rover=sim` selection, and `--sim` accepts physical machine state only, never AI or tool responses. |
| D06 inference | Model load and decode from existing local weights | `real` | `scripts/text_gate.py` measures two-route inference on `SmolLM2-360M-Instruct-Q8_0.gguf` (290 tensors) and writes a receipt carrying the model path, SHA-256, GGUF metadata, both route environments and the decoded token counts. Exact tier (`--require-identical`) and tolerant bound (`--bound 2e-2`) both exercised; the FORK path exercised on `Qwen-3-0.6B…Q4_K_M`. |
| D06 inference | Deterministic embeddings, classifiers and other in-process algorithms | `pure` | A deterministic algorithm is not inference. These keep their real implementations and state their limits; they are never presented as a model result. |
| D07 robotics and drones | Physical test simulation | `physical-test` | Reachable only through an explicit, labelled simulator selection, with the applicable capability, policy, arming, geofence and heartbeat gates applied. The 1500 ms telemetry freshness window is unified between `kaze_live.rs` and `mavlink_gate.rs`. |
| D07 robotics and drones | Real vehicle hardware | `unverified` | No physical device was attached. Live drone and GPIO dispatch is disabled until hardware verification exists. No receipt in this register claims hardware execution. |
| D08 CLI, TUI, REPL | Modes, errors and exit status | `real` | `crates/sofuu-core/src/script_cmd.rs` and `doctor.rs` report unavailable and denied states rather than success. The JS surfaces fail closed: the HTTP `/scouter` route returns `503 scouter_unavailable` without a substrate, and the TUI prints "unavailable" instead of a health verdict it did not measure. |
| D09 mobile and native embeddings | Host-JVM SDK check through the AAR's JNI shim | `real` (host JVM only) | `scripts/mobile/sdk_jvm_check.sh` drives the real runtime; streaming, cancellation, the UTF-8 boundary and the host-tool bridge run against a real temporary file. Host-JVM coverage is not Android device coverage. |
| D09 mobile and native embeddings | Camera and device behaviour | `unverified` | No device was attached and no camera exists on the host JVM. The camera assertion is recorded as a device-only gap, not simulated. |
| D10 desktop | Desktop shell state | `unverified` | See `sofuu-desktop/` — the shell is excluded from the workspace default members and its state must come from a real runtime operation. No device or platform receipt exists for this surface. |
| D11 public and release | README, CLI help, examples, landing content, CI | `real` for the corrected claims | `example-agents/` no longer ships the simulated paper-trading example; it is replaced by a read-only example that uses real local data, a real MCP tool and real model inference. `example-agents/paper_trader` was retired. Public claims that described unexercised effects were corrected. |
| D11 public and release | Release gates | see below | A green gate means real coverage or an explicit blocked record. No release artifact was published by this work. |

## Capability family register

| Family | Success means | When it cannot be performed | Status |
|---|---|---|---|
| `ai.prompt`, `ai.complete` | A configured, authorized backend performed the request and the returned value came from that backend. | Typed configuration, policy, model, timeout or backend error. No substitute answer is ever constructed. | `real` |
| `ai.classify` | The model returned a category inside the caller's label set. | `E_KAZE_RESULT` when the model answers outside the label set; unavailable when no backend is configured. | `real` |
| `tool.call`, `tool.run` | A real, named, authorized implementation executed the operation with validated arguments and returned its actual output. | Missing tool, denial, invalid arguments, transport failure and remote error all remain errors. A denied tool receives no request. | `real` |
| `memory.save`, `memory.recall` | The selected store performed the documented write or read. Durable claims are proven by reopening the store in a separate process. | Unavailable or denied store, failed write, corrupt data and failed read are errors. An empty result is distinct from a store failure. | `real` |
| `log.*` | The local log sink actually received the log, under the per-run record and byte limits. | The documented logging error contract. Logs are not delivery receipts for external notifications. | `pure` / `real` |
| `alert.send` | The configured sink acknowledged the notification. `stdout`, `stderr` and `file:<path>` are the supported sinks; `file:` commits atomically. | Missing or unconfigured sink fails closed with `E_CAP_MISSING`; a denied sink fails with `E_CAP_DENIED`. | `real` |
| `policy.detect` | The detector's documented pattern scan ran. | An unsupported detector name is rejected rather than reported as a clean verdict. `policy.allow` / `policy.reject` are the VM's own control flow and are not served by a backend. | `real` / `pure` |
| Robotics families | An authorized hardware backend performed the operation, or an explicitly selected simulator performed it and the result says so. | Denial, stale or missing telemetry, unavailable hardware and unknown readings stop safely. An unknown reading is never reported as `0`. | `physical-test`; hardware `unverified` |

## Release surface register

| Surface | Status | Note |
|---|---|---|
| `make bc-check` | tracked gate — **currently failing** | The shipped bytecode snapshots must match their sources. `src/js/tools.js`, `src/js/providers.js` and `src/js/plugins.js` were edited in the 2026-11-16 second sweep and their snapshots were not regenerated, so the gate would fail until `scripts/gen-driver-bc.sh && make && make bc-check` runs. |
| `make size-check` | tracked gate — **currently failing** | The binary size ceiling is 5,767,168 bytes. The last measured shipping binary was 5,801,024 bytes (+33,856). The `kaze_async.js` gzip change in the second sweep is expected to remove ≈13.3 KB of that; the remainder has not been re-measured, and closing it needs either a trim or a maintainer decision on the ceiling. |
| `make kaze-drift-check` | tracked gate | The product Kaze engine and the `kaze/` prototype must not drift. |
| `scripts/preflight.sh` | tracked gate | Includes the division-isolation scan that the remediation plan had left open. |
| `scripts/verify-dist.sh` | tracked gate | Distribution verification. |
| `make test` / `make test-rust` / `make serve-test` | tracked gates | Each reports blocked scenarios explicitly and never prints a green end-to-end banner while a case is blocked. |
| Contract conformance | tracked gate | `scripts/contract/cases.tsv` plus `run_contract_tests.sh`. Cases that need a real resource block with an actionable message instead of executing a substitute. |
| `scripts/contract/run_contract_tests.sh` | tracked gate | Requires `SOFUU_CONTRACT_MODEL` for the inference-gate cases. |
| Published release binaries | `unverified` | No published artifact was replaced by this work. A receipt from a local build is not a receipt for a published binary. |

## Blocked coverage

A blocked case is a coverage gap, not a pass. These are the known gaps on
this machine:

| Gap | Reason | How to close it |
|---|---|---|
| Real model coverage for the contract inference cases | `SOFUU_CONTRACT_MODEL` must name an existing local GGUF | Export the variable; the gate then runs against that weight |
| Android device and camera coverage | No device attached | Attach a device and run the SDK check on it |
| Real vehicle hardware | No physical drone or rover attached | Attach hardware; until then the robotics rows stay unverified |
| The local-tool-call protocol (`<tool_call>` with a structured payload) | No local weight emits a protocol-conforming call: `SmolLM2-360M` answers in prose and `Qwen3-0.6B` answers in prose | Use a model that emits the protocol; the suite asserts the negative meanwhile, so no tool call is fabricated |
| The RLM code-execution protocol against small local models | `SmolLM2-360M`, `Llama-3.2-1B`, `Qwen3-0.6B` and `Qwen3-8B` do not follow it | Use a model that does; a failure is reported as a failure, never as a pass |
| The local JS corpus's substitute-provider cases | 30 entries in `tests/run_js_tests.sh` still drive a scripted OpenAI-compatible endpoint or a scripted search endpoint. The runner reports them as BLOCKED, but the substitute endpoints still run — labelling is not removal | Re-point each case at a real local model or a real local tool, or remove the substitute endpoint and keep the case as a pure check |
| The staged `dist/` release artifacts | Built from `15ad2e0be`, before the fail-closed fix; they still contain the pre-fix fabricated strings | Rebuild or quarantine them, and teach `scripts/verify-dist.sh` to compare `build.revision` |
| The desktop app (D10) | No Tauri build and no display on this machine; only the runtime side it talks to was verified | Build and drive the app on a desktop host |
| The release gates after the 2026-11-16 second sweep | The sweep's Rust edits are type-checked only; the release build, the Rust suites, `make size-check` and `make bc-check` were not re-run | See "Gaps this sweep did not close" above |

## 2026-11-16 second sweep — per-division rows

A second sweep gave one worker per division of `sofuu-map.md`, each with
disjoint file ownership, each producing a per-area report under
`contract/register/`. This section carries the rows those reports produced;
the per-area reports keep the full evidence, the command that was run, and the
gaps each worker recorded.

| Report | Area | Worker outcome |
|---|---|---|
| `contract/register/kaze.md` | D05 Kaze host, policy, provenance, shipped-asset size | completed |
| `contract/register/robotics.md` | D07 robotics — **not written**; the worker's edits landed, its report did not | timed out |
| `contract/register/memory.md` | D02 memory stores | completed (report written after the harness timeout) |
| `contract/register/ai.md` | D03/D06/D08 providers, agent loop, chat, Scouter | completed (report written after the harness timeout) |
| `contract/register/tools-mcp.md` | D04 tools, MCP, policy | completed |
| `contract/register/compiler-rlm.md` | D05 compiler, script runner, RLM | completed (report written after the harness timeout) |
| `contract/register/js-tests.md` | R5 JS corpus — **not written**; the worker's edits landed, its report did not | aborted |
| `contract/register/bench-examples.md` | R5 benchmarks and examples — **not written**; the worker's edits landed, its report did not | timed out |
| `contract/register/mobile-desktop.md` | D09 mobile, D10 desktop | completed |
| `contract/register/cli.md` | D08 CLI, doctor, `cli/README.md` | completed |
| `contract/register/contract.md` | D11 contract conformance harness | completed (report written after the harness timeout) |
| `contract/register/public-claims.md` | D11 public claims — **not written**; the worker's edits landed, its report did not | completed, report missing |
| `contract/register/repo-scan.md` | D11 whole-repository sweep (tracked and ignored sources, generated artifacts) | completed |

**Verification state of these rows.** The tree type-checks after the sweep
(`cargo check --offline -p sofuu-core --lib --tests`, clean). The release
build, the Rust suites, `make size-check` and `make bc-check` were **not**
re-run after the sweep, so a `real` row below means the code path was audited
against the plan and, where the report says so, exercised by that worker's own
harness — not that the shipping binary was re-verified. A row whose evidence
says "type-checked, not executed" is exactly that and is marked as such.

### D02 memory

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D02 memory | Kaze `memory.save` / `memory.recall` on the ephemeral CMA (`memory=ephemeral`) | `real` | `crates/sofuu-core/src/rt/kaze_memory.rs` (`EphemeralMemoryHost`) over Sofuu's real CMA. `scripts/test-kaze-memory-real.py` → PASS. Run-scoped: not durable, and the trace says so. |
| D02 memory | Kaze `memory.save` / `memory.recall` on the durable store (`memory=durable`, `memory=file:<path>`) | `real` | `DurableMemoryHost`: every save is staged, fsynced and renamed over the target before the call returns. `scripts/test-kaze-memory-durable.py` → PASS. |
| D02 memory | The durable claim across a process boundary | `real` | `scripts/test-kaze-memory-durable.py` writes in process 1, exits, recalls in process 2; the same store is re-read through `.kazebc` and through a compiled standalone binary. The Rust equivalent spawns a child process and reads the record back — type-checked, not executed. |
| D02 memory | Store isolation (path ↔ path, durable ↔ ephemeral, run ↔ run) | `real` | `scripts/test-kaze-memory-durable.py` and `scripts/test-kaze-memory-real.py`; Rust isolation tests exist and are type-checked. |
| D02 memory | Authorization of the memory family (declaration, grant, policy) | `real` | `domains/kaze/vm.rs` runs `hardware::authorise` for every capability call. Observed live: `E_CAP_MISSING` without a grant, `E_CAP_DENIED` under `tools.deny`, `E_CAP_APPROVAL_REQUIRED` under `approval.mode: always`. |
| D02 memory | Missing configuration / denied access on `memory.*` | `real` | No `--cap memory=…` → `E_CAP_MISSING: embedded Kaze has no available backend for memory.save`; a denied policy never reaches the store. |
| D02 memory | An absent, unreadable, corrupt or oversized store | `real` | Absent → a genuine empty result; unreadable → `E_KAZE_MEMORY: cannot read durable memory store …`; unparsable → `E_KAZE_MEMORY: corrupt storage data in …`; over the 256 KiB bound → `E_KAZE_MEMORY: … larger than the … byte limit`. All four observed live except the bound case (type-checked test). |
| D02 memory | A write that cannot reach the disk | `real` | `write_records` stages, publishes atomically, and returns `E_KAZE_MEMORY: cannot commit durable memory store …` when the publish fails; the staging file is removed. Observed live (a directory at the store path): `ok: false`. |
| D02 memory | Ephemeral vs durable provenance in the API, the CLI output and the trace | `real` | The resolved backend appears in the report's `backends` array and in the trace line as `[real: memory-ephemeral]` / `[real: memory-durable]`. |
| D02 memory | WAL / storage commit path (atomic publish, write-ahead journal, directory fsync) | `real` | `crates/sofuu-core/src/rt/wal.rs`: `publish_paths` = fsync the staged bytes → rename → fsync the directory; `Journal` records the intent and `recover` rolls a crashed commit forward or back. The journal is now cleared in both outcomes of a publish attempt. |
| D02 memory | Native QTSQ brain store (`open` / `remember` / `recall` / `flush` / `close`) | `real` | `scripts/test-real-memory.py` → PASS (write, separate-process reopen, isolation, readonly refusal, a real filesystem flush failure). `rt/memory.rs` throws coded errors (`E_CORRUPT`, `E_LOCKED`, `E_ENOSPC`, `E_STORE`, `E_INVALID_PATH`, `E_ACCES`). |
| D02 memory | Native structured store (`sofuu.storage`: open, insert, query, count, flush, close) | `real` | A corrupt manifest, an unreadable segment and an unlistable directory fail the open with `E_STORAGE:`; `count()` for an unknown table is a typed error, not `0`; a failed flush keeps the manager dirty; a malformed query is refused instead of widening the read. |
| D02 memory | Timeout and cancellation **during** a memory operation | `unverified` | The VM polls the host deadline at every loop head and the fuel budget stops a runaway script with `E_TIMEOUT`. A memory operation is a bounded, synchronous, local call with no internal retry and therefore no mid-operation cancellation point; a store on a hung network filesystem would block past the deadline. Not exercised on this machine. |
| D02 memory | Two **concurrent** writers on one durable path | `unverified` | The writer lock is taken on the first save and an unarbitrated writer is refused, but two live processes racing on one store were not exercised here. The durable harness runs its processes sequentially. |
| D02 memory | A store on a filesystem that refuses the directory fsync | `unverified` | The publish path treats a failed directory fsync as best-effort: the bytes are durable and the rename landed, so the write is reported as done, but the directory entry's own durability across a power loss is not guaranteed. Recorded as a limitation, not a claim. |

### D03 agents and D06 inference

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D03 agents | Agent loop: an unavailable provider, tool or substrate must fail with a typed error, never a plausible-looking answer | `real` | No substitute provider in the Rust registry (`rt/ai.rs`) or the JS registry (`src/js/plugins.js`); `src/js/agent.js` escalates an empty provider stream to `E_EMPTY_STREAM` instead of answering. |
| D03 agents | Chat turn engine: an empty agent answer replaced by the literal `'(no response)'` and persisted as the turn's answer | `unverified` | `src/js/chat.js:1904` — `String((res && res.answer) \|\| '(no response)')`, then the answer is logged and pushed to the history. `src/js/chat_tui.js` has the equivalent guard so the placeholder stays out of the transcript; `chat.js` has none. **Open**: the chat turn engine still needs the same guard. |
| D03 agents | `ai.classify`: a real model call whose category is validated against the caller's label set | `real` | `src/js/kaze_async.js` performs the provider call and validates the result, rejecting an out-of-set category with `E_KAZE_RESULT`; no keyword match remains. |
| D06 inference | Provider response path: an error body behind a 200 is a failure; an empty completion stays an empty success; a body with no completion is malformed; provider-reported usage is used when present | `real` | `rt/ai.rs` — `parse_completion_body`, `parse_usage`, and the embed/audio error-body paths. |
| D06 inference | A local (in-process) completion returns the model's own answer, never filled in | `real` | A real answer containing a NUL is no longer replaced by an empty one. |
| D06 inference | A streamed answer is not truncated | `real` | `js_string_bytes` and the stream write paths. |
| D06 inference | Unknown completion (request sent, result lost) is reported as unknown; no automatic replay of a provider request | `real` | The provider layer contains no retry or replay path. |
| D06 inference | The request body is always a real, well-formed body | `real` | `request_body_cstring`, with model/role escaping. |
| D06 inference | Per-operation deadlines and response size bounds on the provider paths | `real` | 64 MiB response cap with write callbacks and `CURLOPT_MAXFILESIZE`; 30 s connect timeout; the caller's `timeout_ms`; a 300 s stall watchdog. |
| D06 inference | Explicit cancellation on the provider paths | `real` (streams) / `unverified` (non-streaming) | Streams expose an abort handle. **Gap:** `ai.complete` exposes no abort handle, so an in-flight non-streaming completion can only be ended by the stall watchdog or the caller's deadline. |
| D03 agents (RLM) | The RLM episode state machine, the sandbox and the wire protocol | `pure` | Deterministic in-process computation over fixed reply scripts — the replies are *inputs* to the transition function, never a simulated provider. A tool suspension without an executor stops the episode with `tool_unavailable` instead of resuming with a placeholder. |
| D03 agents (RLM) | The RLM routing heuristic and its decision log | `pure` | A token-ratio + holistic-word heuristic and a rotating JSONL log. Deterministic, no external effect. |
| D03 agents (RLM) | The RLM episode loop driven by a real local model | `unverified` | The coverage exists and is `#[ignore]`d by default and drives the shipped inference engine over an existing local GGUF. It was **not executed**: `cargo test` is owned by the coordinator. Command: `cargo test --offline -p sofuu-core --lib -- rlm::real_model -- --ignored`. |
| D03 agents (RLM) | The RLM code-execution protocol against small local models | `blocked` | `SmolLM2-360M`, `Llama-3.2-1B`, `Qwen3-0.6B` and `Qwen3-8B` do not follow the snippet protocol. A failure is reported as a failure; nothing is fabricated to make the loop look successful. |

### D04 tools, MCP and policy

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D04 tools, MCP, policy | Built-in tool execution (`sofuu.tools.execute` → the tool's real implementation) | `real` | The executor resolves the implementation and returns its real output; the policy verdict is resolved before dispatch. |
| D04 tools, MCP, policy | Tool argument contract (the declared schema is enforced) | `real` | `schemaWalk` raises `E_SCHEMA` with `.path`/`.tool` before dispatch; a `grep` without a pattern is rejected with `path='$.pattern'`. |
| D04 tools, MCP, policy | Plugin-registered tool reachable through `sofuu.tools.execute` | `real` | Late-registered tools fall back to their own `execute()` instead of "unknown tool". |
| D04 tools, MCP, policy | MCP stdio client (real child process, JSON-RPC over pipes) | `real` | `scripts/test-real-mcp-file.py` phase 1 PASS against the pre-sweep artifact; the native client is `uv_spawn` + pipes. |
| D04 tools, MCP, policy | MCP Streamable-HTTP client | `real` | `scripts/test-real-mcp-file.py` phase 2 PASS against the pre-sweep artifact: real file read, policy denial, ping, deadlines, circuit breaker. |
| D04 tools, MCP, policy | MCP server transports (in-process HTTP and direct) validate arguments like the stdio server | `real` | `mcpArgsViolation` + `-32602` on both transports, handler not reached. **No contract harness drives these two transports**, so the end-to-end path is unverified. |
| D04 tools, MCP, policy | MCP server spawn bound by the process policy (`SOFUU_NO_SHELL`, `shell.enabled`, `shell.allow`) | `real` | `gate_process_spawn` runs before the child exists → zero dispatch; refusal code `-32006` → `E_POLICY_DENIED`. **Behaviour unverified end to end** — the unit test is type-checked only and the shipping binary predates the change. |
| D04 tools, MCP, policy | A denied operation produces zero dispatch (connection denial, tool denial) | `real` | The harness asserts the **absence** of the side-effect receipt and the start marker, not the presence of a denial message. |
| D04 tools, MCP, policy | Unknown completion after a disconnect, and no automatic replay | `real` | `E_MCP_OUTCOME_UNKNOWN` on a lost response, on a queued call with a lost response and on a timeout; the side-effecting call ran exactly once. |
| D04 tools, MCP, policy | SSRF guard on the network boundary | `real` | The native fetch enforces `gate_net_host` + `ip_is_internal` at the resolved address; `src/js/web.js` adds a JS-level guard with no bypass; every non-200 is `ERR_WEB_SEARCH_HTTP`. |
| D04 tools, MCP, policy | Provider adapters return real answers, never canned output | `real` | Real inference against a local `llama-server` serving `~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf`: the adapter returned the model's own answer with real token counters, corroborated server-side. Unreachable host → `E_TOOL`; missing API key → `E_DENIED`; HTTP 404 → `E_TOOL`. |
| D04 tools, MCP, policy | The provider registry as shipped in the binary | `unverified` | The shipped binary embeds an **older** provider driver: it ignores `LLAMA_CPP_HOST` and fell back to `http://127.0.0.1:8080`. The current source honours the canonical name (verified). A rebuilt binary is needed to close this row. |
| D04 tools, MCP, policy | One provider per real-service name | `unverified` | Two registrations exist under the name `llamacpp`: the first-party JS adapter and the discovered `plugins/llamacpp_provider.ts`, which overwrites the entry when `plugins/` is discovered. Neither returns canned output, but the two disagree about the endpoint name. `plugins/llamacpp_provider.ts` is outside the swept scope — an owner must reconcile it. |
| D04 tools, MCP, policy | `policy.detect` with an unsupported detector kind | `real` | `E_CAP_INVALID: unsupported policy detector '<kind>'` — a detector that does not implement a kind never returns a clean verdict for it. |
| D04 tools, MCP, policy | `policy.allow` / `policy.reject` | `pure` | VM control flow: the verdict is consumed in the VM before any host is consulted, and the host refuses to answer one. |

### D05 Kaze and compiler

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D05 Kaze and compiler | Runtime host: `ai`, `tool`, `memory`, `alert` without a configured backend | `real` | No path reports success: `ai`/`tool`/`memory` fall through to `E_CAP_MISSING`; `alert.send` needs a configured sink. |
| D05 Kaze and compiler | Physical test host: only explicitly selected `drone`/`rover` families | `physical-test` | `SimHost::with_simulated` filters to the physical families and `serves` requires the explicit selection; `hardware::validate_grant` refuses `sim` for any other family. A simulator result is never evidence about hardware. |
| D05 Kaze and compiler | Unknown telemetry field on a simulated vehicle | `physical-test` | `E_CAP_MISSING: the simulated <family> has no '<field>' reading` — never `0`/`false`. The live path refuses unknown and stale readings the same way. |
| D05 Kaze and compiler | Provenance of a simulated-vs-real action in the trace, the CLI output and the structured result | `real` | Every action line carries `[<mode>: <backend>]`; the JS result also carries the resolved `backends` array. The mixed-run test is type-checked, not executed. |
| D05 Kaze and compiler | The compiled-agent bootstrap carries the backend behind each grant, the policy, the pins and the validated configuration | `real` | `compiler.rs` → `opts.caps` (backend per family), `opts.policy`/`opts.policyJson`, `opts.pins`, the fuel/entry/timeout/heap/tick bounds, `opts.sim`, `opts.ai`/`opts.mcp`. Verified by executing the generated bootstrap in a Node sandbox (22/22 assertions) and by the Rust unit tests. |
| D05 Kaze and compiler | `--no-policy` as an authorisation bypass | `real` (absent) | Both surfaces install an explicit deny-all policy; the runtime refuses every capability call with `E_CAP_DENIED`. Evidence: a Node harness and the Rust tests `no_policy_is_a_deny_all_policy_and_nothing_reaches_a_backend` and `a_no_policy_bootstrap_carries_a_deny_all_policy`. |
| D05 Kaze and compiler | `--sim` carrying an AI, tool, memory, alert, policy or log result | `real` (absent) | `script_cmd.rs` and `compiler.rs` refuse the non-physical families at parse time (`E_CAP_INVALID`); the simulator itself refuses any key it does not model. |
| D05 Kaze and compiler | A bare or empty `--cap` meaning simulation | `real` (absent) | A bare grant carries no backend, an empty backend is refused (`E_CAP_INVALID`), and the simulator needs an explicit `drone=sim`/`rover=sim` selection. |
| D05 Kaze and compiler | A malformed option that is silently ignored | `real` | The supervisor interval now fails instead of defaulting; two policy forms in one configuration file are refused; a backend configuration that is not a JSON object is `E_CONFIG`. |
| D05 Kaze and compiler | The 12-factor configuration hierarchy of a compiled agent (CLI > environment > configuration file > routine signature) | `real` | Every option resolves through the same tiers, the environment tier also carries `ai`/`mcp`, and the command line can override them. A missing routine parameter is reported as structured guidance by the bootstrap and as `E_KAZE_PARAM` by the source path — never a low-level arithmetic crash. |
| D05 Kaze and compiler | The compiler and the runtime bootstrap carry no domain parameter, default or external endpoint | `pure` | A static check over `compiler.rs` finds no domain parameter name (`entry_price`, `sentiment`, `volume_24h`, `heart_rate`, `altitude`, `ticker`), no provider URL and no API name. The only names the bootstrap contains are generic capability families and the operator's own values. |
| D05 Kaze and compiler | Shipped JS asset size: `kaze_async.js` was embedded raw in release builds | `real` | Staged gzipped by `build.rs`, embedded as `.gz`, inflated once behind a `OnceLock` and evaluated at the same point as before. 18,884 B → 5,535 B in `.rodata`, a saving of ≈13.3 KB. The inflated bytes are identical to the committed source. **Not yet re-measured in a release build.** |
| D07 robotics and drones | Real vehicle hardware behind the Kaze host | `unverified` | No device attached; live drone/GPIO dispatch stays disabled and `LiveHost` is not wired into `run_prepared`. |

### D08 CLI, TUI and REPL

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D08 CLI, TUI, REPL | `sofuu agent run` health line (Scouter context/brain score) | `real` | A score is printed only when its measured flag is `true`; an unmeasured context omits the segment and an unmeasured brain side prints "not measured". Not exercised in this worker (build policy). |
| D08 CLI, TUI, REPL | Scouter: an unmeasured score is absent/null and flagged as unmeasured in the TUI and the CLI health line | `real` | `rt/scouter.rs` publishes `contextScore: null` with `contextMeasured: false`; `src/js/chat_tui.js` honours the flag; the CLI health line requires `contextMeasured === true`. An evaluation with nothing to evaluate no longer publishes a score. |
| D08 CLI, TUI, REPL | The TUI prints the Scouter's entropy/efficiency diagnostics of an unmeasured context report as `0.00` / `100.0%` | `unverified` | `src/js/chat_tui.js:310` — `r.max_entropy \|\| 0` / `r.efficiency_ratio \|\| 1`. Only reachable when the brain score exists while the context report is unmeasured; the brain surface has no caller today. |
| D08 CLI, TUI, REPL | `sofuu doctor` — grant → backend table | `real` | The backend names are probed from `hardware::validate_grant`, the function that enforces `--cap family=backend`. A bare grant is reported as resolving to no implementation, and `sim` is reported as the explicit drone/rover physical test only. Type-checked, not executed. |
| D08 CLI, TUI, REPL | `sofuu doctor` — policy presence, validity and non-emptiness | `real` | Reads the two layers the runtime reads and merges them with `layered_policy_from_paths`; an invalid layer is reported as **not** in force. Type-checked, not executed. |
| D08 CLI, TUI, REPL | `sofuu doctor` — model endpoint and weights | `real` | The endpoint is the configured override or the provider's own default; the weights file is checked on disk and a missing file is reported as unavailable. The doctor does not run inference and says so. |
| D08 CLI, TUI, REPL | `sofuu doctor` — Scouter state | `pure` | The doctor reads the process-local Scouter state. The scores are measured-only: absent (`null` in JSON) when nothing was measured. |
| D08 CLI, TUI, REPL | `sofuu serve` endpoint banner vs the implemented auth boundary | `real` | The banner matches `src/js/serve.js`: `/healthz` and `/livez` are tokenless, `/health` needs the bearer token unless `--public-health`. |
| D08 CLI, TUI, REPL | `sofuu plugin test` help text | `real` | The mode executes the real tool; the help no longer describes a mock. |
| D08 CLI, TUI, REPL | `cli/README.md` — the modes the prebuilt binaries actually have | `real` | Verified against `cli/sofuu-darwin-arm64 --help` and by running `version` (exit 0) and `doctor` (exit 1, unknown command) on 2026-11-16. |
| D08 CLI, TUI, REPL | `cli/README.md` — the brain round-trip claim on `cli/sofuu-darwin-arm64` | `real` | Re-run on 2026-11-16: the first `eval` printed `true` for the flush, the second process recalled `persisted-value`. The write needed the explicit flush. |
| D08 CLI, TUI, REPL | `cli/README.md` — the networking / TLS description | `real` | Read from the binaries: macOS links the system libcurl, Linux is static with libcurl 8.6.0, Windows uses Schannel. The CA-bundle overrides are in the current runtime, not in these binaries. |
| D08 CLI, TUI, REPL | `cli/README.md` — the build instructions | `real` | The `Makefile` targets, the QTSQ guard, `scripts/cross/install_zig.sh` (Zig 0.13.0) and `scripts/cross/build_qtsq_cross.sh` with the `dist/qtsq-*` libraries all exist as documented. |
| D08 CLI, TUI, REPL | The Linux and Windows binaries in `cli/` | `unverified` | They were not executed on this machine (a macOS host). Their brain round-trip and their TLS behaviour are not receipts here — only their static contents were inspected. |

### D09 mobile and D10 desktop

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D09 mobile and native embeddings | Host-JVM SDK check through the AAR's JNI shim (real runtime, real local weights) | `real` (host JVM only) | `scripts/mobile/sdk_jvm_check.sh` on `target/release/libsofuu_capi.dylib` against `~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf`: 51/51 checks, real streaming deltas and token counts from the engine, a real host-tool read of a temporary file, mid-flight cancel, the UTF-8 boundary, a real agent turn. **Not** Android device coverage. |
| D09 mobile and native embeddings | AAR helper values (`similarity`) | `pure` | The value is the runtime's own cosine similarity over caller-supplied vectors; the harness pins 1.0 for v·v and 0.0 for orthogonal vectors. `SofuuBridge.kt` no longer answers a hardcoded `0f` — a runtime that cannot compute it now answers `null`. **API change:** `similarity()` returns `Float?` instead of `Float`; no in-repo caller, and `CHANGELOG.md` must note the AAR break. |
| D09 mobile and native embeddings | Camera and other device-only capabilities | `unverified` | No Android device was attached and the host JVM has no camera. The harness prints `UNVERIFIED: camera capture — needs a physical Android device`; nothing simulates it. |
| D09 mobile and native embeddings | Android AAR packaging (SONAME, relative DT_NEEDED, 16 KiB alignment, legal notices) | `unverified` | `scripts/mobile/verify_android_aar.sh` is a real packaging check, but no AAR was built or verified in this worker. |
| D10 desktop | Desktop shell state (sessions, engine card, usage, tools, permissions) | `unverified` | The shell was **not run**: no Tauri build was possible and no display was exercised. What was verified instead is the runtime side the shell talks to: the release runtime answers every call the shell makes with the shapes the shell expects, and the shell now renders an unavailable/error state with the engine's own reason instead of a placeholder. Building and driving the app is still open. |
| D10 desktop | Session list provenance (engine registry vs on-disk registry mirror) | `unverified` (code path verified) | The backend returns the source with the list and the shell names it. The code path was verified by rendering the exact JS the backend sends against the real runtime; the app itself was not run. |
| D10 desktop | Desktop memory write (`/remember`) | `unverified` (runtime path `real`) | The command now routes through the runtime's own writer (it previously could never work — "no open brain"); the exact script returned `{"ok":true,"id":2}` against the real runtime and the fact was read back with similarity 1.0. The desktop command itself was not exercised. |
| D10 desktop | Keychain credential storage (macOS) | `unverified` | `src-tauri/src/keychain.rs` uses the real macOS Keychain; a platform without an OS credential store now refuses the write instead of keeping an in-process map that would be lost at exit. Not exercised here. |
| D10 desktop | Provider model list (SSRF-guarded engine fetch) | `unverified` | The command fetches the list inside the engine with a loopback/private-address guard and caches the outcome. The unit tests exist in the crate but were not run. |

### D11 public and release

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D11 | Whole-repository label and return-pattern sweep, tracked sources | `real` | All 1363 tracked files were swept for the seven labels plus the return patterns (constant replies, ignored errors, unconditional success, default telemetry, simulator fallback). No prohibited substitute remains in the tracked runtime source. |
| D11 | Whole-repository sweep, Git-ignored sources | `real` | `tests/`, `examples/`, `bench/`, `benchmarks/`, `docs/`, `landing-page-2/`, `.mimosa/`, `local-only/`, `npm/`, `new-model-arch./` and `emb-m/` were swept. Five substitute-driven leads were found and closed during the sweep. The remaining ignored-source lead is the documentation set below. |
| D11 | Release artifacts in `dist/` — the staged CLI binaries and their release tarballs | `unverified` | `dist/sofuu-darwin-arm64` reports `build.revision = 15ad2e0be-dirty` (2026-11-08), an ancestor of the fail-closed fix `db090b419` (2026-11-11), and contains the pre-fix fabricated strings `OK: tool '{name}' executed ({arg})` and `Synthesized autonomous decision for: "{brief}"`. `dist/sofuu-linux-x86_64` carries the same strings. **These staged artifacts must not be published.** The CI release workflow rebuilds before uploading, so the published path is not proven to ship them; a manual upload of the staged tarball would. |
| D11 | Release artifacts in `dist/` — the staged embedded libraries | `unverified` | `dist/libsofuu.{a,so,dylib}` and the Android/iOS libraries (2026-10-28…30) carry no `E_CAP_MISSING` / `E_KAZE_LINK` surface: they predate the Kaze capability feature. They cannot support the D05 capability rows. |
| D11 | `scripts/verify-dist.sh` staleness detection | `real` (absent) | The `dist/sofuu-*` loop runs only the build-host path check; a `dist/` artifact built from an older revision is not detected, although the comment there says a stale one "is reported by name so it can be rebuilt". |
| D11 | `tests/run_js_tests.sh` case typing (local-only corpus) | `real` | The registry types every case `real` / `scripted` / `model` with a mandatory reason for a gap; a `scripted` case is reported BLOCKED and a `model` case is skipped when no local GGUF is present. 74 entries re-verified (42 real / 26 scripted / 6 model). **Local-only:** `tests/` is Git-ignored, so a clean checkout has no runner and the preflight reports `BLOCKED js` for it. |
| D11 | Tracked contract conformance suite | `real` | The cases that previously ran a substitute are either migrated to the real local model or explicitly blocked by the runner, and the `cases.tsv` prereq column names the real backend and its block condition. |
| D11 | `benchmarks/agent_bench.js` | `real` | The substitute provider and its artificial per-request latency were removed; the benchmark states that its earlier substitute-provider figures measured a canned response and are not comparable. `bench/` and `benchmarks/` are Git-ignored. |
| D11 | Landing content | `pure` | `landing-page-2/src/components/ProductSections.tsx` renders static product visuals with figures that read like receipts ("3 files changed", "214 passing"). No execution path; the page is Git-ignored. Whether it needs an "illustration" label is a disclosure decision, not an execution gap. |
| D11 | Documentation | `blocked` | `docs/` is Git-ignored (`.gitignore:164`) and superseded substitute-provider instructions remain in eight plan/roadmap files (plan R5 bullet 5). |
| D11 | Contract conformance runner: blocked is a recorded gap, never a pass | `real` | A harness that prints `BLOCKED:` with exit 0 FAILs; a well-formed block is counted as a real-execution gap that `ALLOW_BLOCKED` cannot waive. |
| D11 | Contract case table: every case names the resource it runs against | `real` | `scripts/contract/cases.tsv` (53 cases, each prereq prefixed `real:` or `pure:`) is cross-checked against each harness's `// RESOURCE:` declaration; a mismatch FAILs the case. |
| D11 | Model-dependent cases select their weights through `SOFUU_CONTRACT_MODEL` and block when it is unset | `real` | The runner exports the value after validating it; with it unset 18 cases BLOCK with the reason and 0 PASS. No hardcoded per-developer model paths remain in the harness. |
| D11 | T2.5 — bounded chunk queue: typed `E_BACKPRESSURE` + provider abort | `real` | On `SmolLM2-360M-Instruct-Q8_0.gguf`: the queue overflowed after 7250 ms of not draining, `E_BACKPRESSURE` named the 1024 bound, `aborted === true` (5 passed, 0 failed). |
| D11 | T2.6 — terminal-once, total order and no seq gaps on failure paths | `real` | On `Qwen-3-0.6B…Q4_K_M`: the real `read_file` tool really failed (ENOENT) and the chat turn against a refused connection ended with `stop(reason:error)` / `E_AI_UNKNOWN_COMPLETION` — one terminal each, no seq gap (10 passed, 0 failed). |
| D11 | Static-fixture cases (session fold, migration, redaction, golden renderer, event log, deadlines, error shape, capability manifests, net policy) | `pure` | Each harness carries a `PURE CHECK:` note and a `// RESOURCE: pure:` declaration and is not presented as inference or tool evidence. |
| D11 | T2.3 — cancel mid-tool kills the tool's subprocess | `blocked` | The case needs a model that issues the tool call. `SmolLM2-360M` and `Qwen-3-0.6B…Q4_K_M` did not issue it in 3 attempts, so the case reports BLOCKED. The harness does not fabricate the call. |
| D11 | T2.6 / T7.3 — agent-leg tool and approval events with `SmolLM2-360M` | `blocked` | The same reason. Both cases pass with `Qwen-3-0.6B…Q4_K_M`. |
| D11 | T3.3 — exact-tier determinism gate across the two routes | `blocked` | The case PASSES on `SmolLM2-360M-Instruct-Q8_0.gguf` and FAILS on `Qwen-3-0.6B…Q4_K_M` on this machine (the CPU and GPU routes diverge after step 5, `\|dtop1\| = 0.0200`, verdict `NEAR-TIE-CERTIFIED` against a required `IDENTICAL`). The register's D06 row names SmolLM2 as the exact-tier model. |
| D11 | T6.4 / T6.7 — usage reconciliation against the provider's own report | `blocked` | With `Qwen-3-0.6B…Q4_K_M` the cancelled-stream accounting and the cloud-stream frame count do not match the client's observation. Both PASS with `SmolLM2-360M-Instruct-Q8_0.gguf`: the measurement differs, not the harness. |
| D11 | T7.8 — provider-echo leak route | `unverified` | The harness checks the event, event-log and session-document surfaces against a real `sofuu serve` with a real key, then ends the case BLOCKED: the echo route needs a provider that answers 401 with the `Authorization` header in its body, and the runtime surfaces a provider error body verbatim. |
| D11 | T5.6 — real agent + real MCP file server with a configured provider | `blocked` | `scripts/test-real-agent-mcp.py` blocks on `SOFUU_TEST_CONTEXT_WINDOW` (it also needs `SOFUU_TEST_LOCAL_MODEL` or the remote provider variables). |
| D11 | T4.2 — mobile/device contract surface | `blocked` | Declared `manual` in the case table: it needs macOS + Android runners. No device was attached. |

### Gaps this sweep did not close

| Gap | Why it is still open | How to close it |
|---|---|---|
| The JS corpus still executes substitute providers | 30 entries in `tests/run_js_tests.sh` are typed `scripted`: the case drives a scripted OpenAI-compatible endpoint or a scripted search endpoint. The runner now reports them as BLOCKED instead of PASS, but the substitute endpoints still exist and still run. Labelling is not removal (`RULES.md` §1.9). | Re-point each case at a real local model or a real local tool, or delete the substitute endpoint and keep the case as a pure check. `tests/` is Git-ignored, so this is local-only coverage. |
| The chat turn engine substitutes `'(no response)'` for an empty answer | `src/js/chat.js:1904` persists the literal placeholder as the turn's answer. `src/js/chat_tui.js` has the guard; `chat.js` does not. | Apply the same guard in `chat.js` and cover it with the chat e2e cases. |
| The release binary is over the size ceiling | The measured shipping binary is 5,801,024 B against the 5,767,168 B ceiling (+33,856 B). The `kaze_async.js` gzip change in this sweep is expected to remove ≈13.3 KB of that, and it has **not** been re-measured in a release build. | Rebuild, measure, and either trim the remaining ≈20 KB or record the ceiling change as a maintainer decision. |
| The shipped driver snapshots are stale | `src/js/tools.js`, `src/js/providers.js` and `src/js/plugins.js` were edited in this sweep; their `.bc` / `.bc.gz` snapshots were not regenerated, so `make bc-check` would fail. | `scripts/gen-driver-bc.sh && make && make bc-check`. |
| The staged `dist/` artifacts predate the fail-closed fix | `dist/sofuu-darwin-arm64` and `dist/sofuu-linux-x86_64` are built from `15ad2e0be` and still contain the pre-fix fabricated strings. | Rebuild or quarantine the staged artifacts; teach `scripts/verify-dist.sh` to compare `build.revision`. |
| The desktop app was never built or run | No Tauri build and no display; only the runtime side it talks to was verified. | Build and drive the app on a desktop host. |
| The RLM real-model coverage was not executed | `rlm/real_model.rs` is `#[ignore]`d by default and the coordinator owns `cargo test`. | `cargo test --offline -p sofuu-core --lib -- rlm::real_model -- --ignored` with an existing local GGUF. |
| The new Rust tests were not executed | The sweep's Rust edits are type-checked only (`cargo check --offline -p sofuu-core --lib --tests`, clean). | Run the Rust suites and the release gates. |
| Superseded substitute-provider instructions remain in eight plan/roadmap files | `docs/**` is Git-ignored and was outside every worker's editable scope. | Retire the superseded sections (plan R5 bullet 5). |

## Maintenance

Update this file whenever a capability, a gate or a receipt changes. Record
the artifact and the configuration with every receipt; keep dated historical
records labelled as historical. Unknowns stay `unverified` — they are not
implicitly real, and not implicitly broken.
