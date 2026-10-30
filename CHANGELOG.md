# Changelog

All notable changes to the `sofuu` runtime are documented here.
Versions follow the Cargo crate (`0.3.0-beta` line); npm and the CLI banner
carry the same version (checked by `scripts/verify-dist.sh`).

## Boot latency: the GPU probe is off the start path — 2026-10-30

`sofuu eval "1+1"` spent ~29 ms of its 33 ms building the engine, and ~20 of
those milliseconds were a Metal device. The chain was
`engine_register_builtins` → `mod_memory_register` → `eval(src/js/memory_api.js)`
→ `sofuu.capabilities()` → `manifest_json()` → `metal_available()` →
`MTLCreateSystemDefaultDevice()`: the flat memory API asked the whole
capability manifest for one boolean, and the manifest asks the OS for a GPU
device — which costs 18–20 ms the first time in a process.

- **`sofuu.build`** — the manifest's `build` block (features, missing,
  platform_absent, revision) is now registered at startup from `build_block()`,
  a pure function of the compilation with no probe behind it. `manifest()` is
  built from the same function and a unit test pins the two equal, so the
  boot-time facts and the manifest cannot drift apart.
- **`memory_api.js`** reads `sofuu.build.missing` instead of
  `JSON.parse(sofuu.capabilities())`. The guard's intent is unchanged (do not
  install the flat API over typed-refusal stubs on a QTSQ-free build); it just
  no longer creates a GPU device to learn a compile-time fact.
- **The Metal probe is cached** (`OnceLock`): a device cannot appear or vanish
  mid-process, so a second manifest build in the same process is now free
  instead of re-creating it.

Measured on Apple M2 Pro / macOS 15.7.3 — same script in both arms, 40
interleaved runs so both engines see the same machine load:

| Command | Before | After | |
|---|---|---|---|
| `hello.js` (as a compiled artifact) | 34.4 ms | **17.8 ms** | 1.93× |
| `sofuu eval "1+1"` | 33.3 ms | **17.1 ms** | 1.95× |
| `sofuu version` (no engine) | 4.4 ms | 5.0 ms | unchanged (noise) |
| `sofuu capabilities` | 26.8 ms | 26.3 ms | unchanged — it *should* probe the GPU |

Compiled `sofuu compile` artifacts run the same boot path and inherit the win.

What is left, and why it was not taken: ~4.2 ms evaluating the eleven shipped
drivers and ~1.8 ms parsing `compat.js` from source, on top of ~2.3 ms of
QuickJS setup and native registration. Deferring the drivers needs a lazy
accessor for every name they install — about 30 top-level properties
(`sofuu.agent`, `sofuu.tools`, plus `S.cfg`/`S.session`/`S.info`-style
aliases), several colliding with native or Rust-set names such as `VERSION` —
where one mis-installed accessor would shadow a native and break chat/serve
silently. `compat.js` belongs to the compatibility lane; the cheap follow-up
there is to ship it as `compat.bc.gz` like every other driver (~1.5 ms).

## Standalone Agent Compiler (`sofuu compile`) — 2026-10-30

Implementation of `STANDALONE_COMPILER_PLAN.md`: an agent becomes one
self-contained executable — the `sofuu` binary with its bundled program and,
optionally, its brain appended behind a 32-byte trailer.

- **Command** — `sofuu compile <entry.ts|js> [-o <out>] [--embed-brain
  <file.qtsq>] [--no-strip]`. The entry is bundled (`bundler`), compiled to
  QuickJS bytecode with a bare compile-only runtime, gzipped when that saves
  bytes, and staged as `payload = header + entry name + bytecode + brain`.
- **Boot** — the CLI inspects the trailer before any dispatch: a plain binary
  pays two reads, a compiled one runs its program immediately with
  `process.argv` forwarded and `sofuu.args` set to the words after the binary.
  `SOFUU_PLAIN=1` opts back into CLI mode (inspect or recompile an artifact).
- **Exit semantics** — identical to `sofuu run`: a synchronous throw, an
  unhandled rejection and `process.exitCode` mean what they mean there, because
  the compiled path reuses the same precedence and loop drain.
- **Embedded brain** — a store is a manifest plus `.d/` segments, so
  `--embed-brain` packs both (graph snapshot included) into one archive,
  records the vector dimension at compile time, extracts the seed beside the
  binary on first launch, and exposes
  `sofuu.memory.embedded = {path, dim, open()}`. An existing extracted store is
  never overwritten: the payload is a seed, not a reset.
- **Mach-O** — appending bytes to an arm64 binary invalidates its signature and
  fails strict validation, so the artifact is assembled as: remove signature →
  strip any earlier payload (recompiles cannot nest; three generations are
  byte-identical) → append → extend `__LINKEDIT` over the payload → ad-hoc
  re-sign. The trailer therefore does not sit at EOF on macOS; boot finds it
  just before `LC_CODE_SIGNATURE.dataoff`, tolerating codesign's 16-byte
  alignment padding. ELF and PE keep the simple append.
- **Verified** — hello and TypeScript/async parity with `sofuu run`; an
  argv/exit-code matrix (0/1/7); embedded-brain recall from an unrelated cwd;
  corrupt-payload refusal (CRC32, exit 1); size-stable recompiles; cold start
  33.5 ms vs 33.3 ms for `sofuu run` (payload overhead below noise). 16 unit
  tests in `compiler.rs`; conformance stays 51 PASS / 0 FAIL / 1 BLOCKED.
- **Known gate** — `make size-check` was already red before this change
  (5,938,720 B vs the 5,767,168 B cap; this feature adds ~18 KB). Reducing the
  binary is a separate work item, not something this feature regressed.

## Universal Sandbox v2 — 2026-10-29

Implementation of `SANDBOX_UPGRADE_PLAN.md`: the in-process execution sandbox
stops clamping developer budgets, gains callable host bridges, TypeScript/JSX
ingestion, prototype hardening and configurable console routing.

- **Unclamped budgets (I2)** — `SandboxConfig::clamp()` now enforces only
  floors (1 MB memory, 16 KB stack); a stated `timeoutMs`, `fuel`, `maxMemoryMB`
  or `maxStackKB` is never reduced. `fuel: 'unlimited' | 0 | null` disables the
  instruction meter and polling drops to `i32::MAX` between deadline checks;
  `timeoutMs: 0 | null` disables the deadline. The misleading
  `SANDBOX_MAX_FUEL` / `SANDBOX_MAX_TIMEOUT_MS` constants are gone.
- **Presets** — `strict` (2 s, 5M fuel, 32 MB, frozen prototypes), `standard`
  (10 s, 20M, 64 MB), `compute` (300 s, unmetered, 512 MB) and `worker`
  (no deadline, unmetered, 256 MB). Explicit options override the preset; an
  unknown preset name is refused. Numeric options are type-checked rather than
  coerced (`timeoutMs: "soon"` used to become a zero budget through
  `ToNumber`).
- **Callable host bridges** — a top-level function in `globals` is installed as
  a native function in the sandbox realm. Arguments and results cross as JSON;
  a sync host function returns directly, a promise-returning one gives the
  sandbox a real promise. Microtask-only promises settle inside the run; a
  promise that needs the host event loop (fetch, fs) **parks** the run —
  `run`/`eval` then return a Promise, the session reports `E_BUSY` until it
  resumes, and `close()` is deferred to completion. Host throws become
  catchable sandbox errors; a non-serializable result is a typed error.
- **TypeScript without a build step** — `ts: 'auto'` evaluates the source as
  written and retries the stripped source only when the first parse failed, so
  a stripper bug cannot change valid JavaScript. The Rust stripper gained
  **enum** support (rewritten to the same frozen object literal the plugin
  driver emits, including `const`/`export`/`declare` forms), which also fixes
  enums in `sofuu run file.ts`. Top-level `await` is supported through an
  async wrapper (documented: the value comes from `return`).
- **JSX** — `jsx: true` transpiles to `h()` calls without rewriting imports
  and installs `h`/`Fragment` in the realm, so components evaluate to a
  `{type, props, children}` tree.
- **Prototype hardening** — `freezePrototypes` freezes the intrinsic
  constructors and their prototypes at realm creation (`strict` preset
  default), so `Object.prototype.x = ...` cannot stick across turns.
- **Console routing** — `console: 'capture' | 'inherit' | 'ignore' | fn` plus
  `maxLogs` (0 = unbounded, 8 MB total / 64 KB per line ceilings). The old
  `captureConsole` boolean still maps onto it.
- **Typings** — `sofuu.d.ts` declares the `sofuu.sandbox` namespace,
  `SandboxOptions`, `SandboxResult`, `SandboxSession`, `SandboxPreset` and
  `SandboxErrorCode`. Two codes join the taxonomy: `E_BUSY` and `E_BRIDGE`.
- **`code_eval`** awaits the sandbox run (a run only becomes a Promise when a
  bridge parks) and therefore accepts TypeScript by default.
- **Gates** — `make sandbox-stress` runs 26 adversarial cases
  (`scripts/stress/sandbox_stress.js`: budget escapes, memory/stack bombs,
  catastrophic regex, host-global probes, prototype pollution, TS/JSX, bridges
  over real I/O, parked-run lifecycle, presets, 2000-run soak). Plus
  `tests/test_sandbox_v2.js` (10 acceptance groups) and 21 Rust unit tests.
  Docs: `docs/SANDBOX.md`.

## 24/7 Agent Observability, Tracing & Recovery — 2026-10-29

Implementation of `OBSERVABILITY_PLAN.md`: a durable per-run trace, a declared
failure policy, crash post-mortems, and checkpointed resume.

- **One envelope, one writer** — agent runs append the §7.2 envelope to
  `.sofuu/logs/agent-<run_id>.jsonl`; chat keeps
  `.sofuu/sessions/<sid>/events.ndjson`. Same validator, same secret
  redactor, same `rt/events.rs` append path — only the directory differs.
  `run_id` is accepted as an alias for `sid`, so a plan-shaped event
  validates exactly as written. Thirteen kinds join the registry
  (`run_start`, `turn_start`, `tool_call`, `model_complete`,
  `memory_recall`, `memory_store`, `checkpoint`, `warning`,
  `fatal_error`, `retry`, `halt`, `run_complete`, `stream_chunk`).
- **Durability** — every batch is flushed, and a batch carrying a
  `checkpoint`, `halt`, `fatal_error`, `run_complete` or `stop` is
  `fsync`ed and reports `synced: true`. A `kill -9` loses nothing already
  written; a power loss loses only events after the last checkpoint.
- **Concurrent writers are safe** — found by the stress suite (case A11). The
  writer issued two `write_all` calls (body, then newline) and `O_APPEND`
  only makes a *single* write atomic, so a second process could land between
  them and glue two records onto one line — 19 of 200 appends were lost while
  every writer reported success. Writes are now one syscall per line, and a
  per-batch `flock` makes read-tail-then-append atomic, so `seq` stays unique
  across processes. `last_seq` reads a 64 KB tail instead of the whole log,
  which removes a quadratic re-read from every append.
- **Error policy** — `--on-error=auto-retry|fail-stop|interactive` (also
  `SOFUU_ON_ERROR`), defaulting to `interactive` on a TTY and `auto-retry`
  otherwise; an unknown mode is refused rather than silently meaning the
  default. `auto-retry` reuses the existing stream ladder with full jitter and
  now also covers tool faults; five consecutive classified errors trip the
  circuit (`E_CIRCUIT_OPEN`); `interactive` degrades to `fail-stop` with a
  `warning` event when there is no TTY. `[e]dit` in `$EDITOR` is out of scope
  for v1 and documented as such.
- **Post-mortems** — `.sofuu/crashes/<crash_id>.json` carries the plan's card
  shape (failed step, error code and message, stack, context snapshot,
  recovery hint, trace path); `sofuu logs --last-error` renders it.
- **Checkpoint and resume** — a checkpoint is written after every completed
  turn, including turns that never entered the tool loop (otherwise resume has
  nothing to rehydrate for the most common run shape), and it is capped so a
  long run checkpoints its recent window. `sofuu resume <run_id> [--step=N]
  [--force]` rehydrates the newest *valid* checkpoint — skipping a torn tail
  line — reloads `~/.sofuu/agents`, adopts the checkpoint's model and
  provider, and continues from the next step. A completed run is refused
  without `--force`.
- **CLI** — `sofuu logs`, `logs show <id>`, `logs --follow`,
  `logs --last-error`, `sofuu resume`. Rendering lives in
  `src/js/observability.js` and ships as compressed bytecode; `main.rs` stays
  a thin shim, which is what keeps the feature inside the size cap.
- **Typings** — `sofuu.d.ts` declares `sofuu.observability.*`,
  `sofuu.agent.classifyError` and the `OnErrorMode` union. The plan's proposed
  codes `E_NETWORK_TIMEOUT`, `E_SERVER_ERROR` and `E_UNCAUGHT_THROW` are
  deliberately absent: they are mapped onto the normative taxonomy
  (`E_TIMEOUT`, `E_TOOL`, `E_SCHEMA`, `E_DENIED`) so cross-surface error
  parity (R5.1) still holds.
- **Gates** — `make observability-stress` runs 13 adversarial cases
  (`scripts/stress/observability_stress.js`: redaction, seq total order,
  backpressure, `kill -9` durability, four concurrent writers, torn lines,
  crash cards) plus a 7-case CLI e2e (`tests/observability_e2e.sh`).
  Conformance stays 51 PASS / 0 FAIL / 1 BLOCKED.

## Universal Plugin & Extensibility System — 2026-10-29

Implementation of PLUGIN_PLAN.md: five plugin categories behind one runtime,
one safety model, one error taxonomy.

- **Five categories** — action tools (`defineTool`), inference providers
  (`defineProvider`), storage and vector stores (`defineStorage` /
  `defineVectorStore`), lifecycle services (`defineService`, including an
  `onToolCall` interceptor) and description-driven plugins (`useOpenAPI` for
  OpenAPI 3.x/Swagger 2.0, `ingestMcp` for MCP servers). All of them are
  reachable as `import { ... } from 'sofuu'`.
- **Five shields** — schema validation (input and output), policy and SSRF
  network jailing plus a jailed `ctx.fs`, a CLOSED/OPEN/HALF_OPEN circuit
  breaker with single-trial admission, a watchdog with abort signal, and a
  panic guard. A policy denial no longer trips the breaker, so a blocked
  target keeps reporting `E_DENIED` instead of decaying into
  `E_CIRCUIT_OPEN`. Two codes join the taxonomy: `E_CAPABILITY` and
  `E_BACKPRESSURE`.
- **Engine modularised** — `src/js/plugins.js` is now eleven modules (schema,
  security, breaker, watchdog, registry, provider, store, service, openapi,
  discovery, cli) with every 0.2.0 public name preserved.
- **Native provider adapters** — new `src/js/providers.js` ships `ollama`
  (`/api/chat` NDJSON, tool calls, usage), `llamacpp` (OpenAI wire, keyless,
  GBNF `grammar`), `openai` (Bearer key required) and the `lmstudio` /
  `vllm` local aliases, all through one shared wire parser. `completeVia` /
  `streamVia` apply the provider's own breaker and watchdog;
  `sofuu.providers.list()` reports capability, and adapter streaming is
  explicitly `streaming: 'buffered'`. Rust gains a first-class
  `Provider::LlamaCpp` (no `OPENAI_API_KEY` leakage to a local server, no
  `.gguf` hijack of an explicit llama.cpp call) and `-p ollama|llamacpp`
  works end to end.
- **SDK typings** — `sofuu.d.ts` declares the whole surface (tools,
  providers, storage/vector, services, hooks, CLI helpers, the micro-Zod
  shape) and type-checks clean under `tsc --strict`. Shipped in the npm
  package.
- **TypeScript plugins without a compiler** — a scanner-based stripper
  handles imports/exports, interfaces, type aliases, enums, generics,
  parameter/return annotations, optional parameters, parameter properties and
  class fields. It is construct-aware rather than regex-based: an object
  literal `{ units: defaultUnits }` is no longer eaten, and a regex literal
  containing quotes no longer desynchronises the scanner. Unsupported
  constructs are left intact so they fail loudly instead of silently.
- **CLI and discovery** — `sofuu plugin list|validate|run|test|init`,
  `sofuu.json` manifests, `./plugins` plus `~/.sofuu/plugins`, `init`
  templates for all six categories, and a per-file discovery report that
  isolates a broken plugin instead of aborting startup.
- **Dual-write mirror** — `attachVectorStore` / `attachStorage` mirror QTSQ
  writes into an external store behind a bounded, batched, back-off queue; the
  local write stays synchronous and `mirrorStats()`/`flushMirror()` expose
  the queue.
- **Agent seam** — `onTurnStart` may rewrite the prompt, `onToolCall` may
  veto or mutate a call, and every hook is isolated and time-boxed.
- **Cookbook and docs** — four runnable plugins in `plugins/` (`fetch_web`,
  `sqlite_query`, `ollama_provider`, `token_logger`), a user guide in
  `docs/PLUGINS.md`, a README section, and two gates: `make plugin-check`
  (runtime/typings parity plus cookbook strip proof) and `make
  plugin-stress` (25 adversarial cases across all five categories).

## 0.3.0-beta — 2026-10-28 (in development)

Mandatory QTSQ core integration, runtime conformance program completion, and general-purpose AI runtime architectural alignment:

- **General-Purpose AI Runtime Alignment & Decoupling**:
  - *Prompt Sovereignty*: Developer-provided agent system prompts are 100% sacred and never prepended with terminal coding instructions (`CORE_PROMPT`). Custom agents with omitted prompts default to a neutral assistant persona; `CORE_PROMPT` and `AGENTS.md` context injection are strictly isolated to `preset: 'coding'` (used by `sofuu chat`).
  - *Tool-Aware & Opt-In Task-Gate*: Task checklist nagging is strictly tool-aware (checked via `todo_write` in the active toolset) and opt-in via `budget.taskGate` (defaults to `false` for general agents, `true` for coding preset).
  - *Decoupled Cognitive Memory (`remember`)*: Built-in `remember` tool persists facts directly into the on-device QTSQ Cognitive Vector Memory (`cma.remember(...)` with dual indexing in `semCma`); writing to project `AGENTS.md` is strictly guarded by `preset: 'coding'` or explicit `opts.agentsMd`.
  - *Domain Toolpacks (`sofuu.tools.PACKS`)*: Reorganized tools into modular domain packs: `coding` (alias `code`), `web`, `memory`, `system`. Introduced first-class general tools: `system_info` (runtime version, OS platform, CPU architecture, GPU engine, capabilities), `memory_search` (natural language query of QTSQ vector memory), and `memory_remember`. Added `category` metadata and category filtering to `sofuu.tools.list()` and HTTP `GET /v1/tools?category=...`. Enabled direct multi-domain tool string resolution in `agent.define`.
  - *Generalized Session Documents*: Expanded canonical session document top-level keys (`DOC_KEYS`) to include `state` (application/workflow state container) and `metadata` (custom annotations). General session headers support `kind` (`'chat'`, `'agent'`, `'workflow'`, `'task'`, `'pipeline'`), `agent`, `status`, and metadata. Generalized step records support `input` and `output` aliases and structured `data` payloads. `sofuu.session` JS surface accepts direct JS objects in addition to JSON strings.
- **Mandatory QTSQ Core** — Made QTSQ memory a permanent core component of the
  runtime across all targets; retired the legacy QTSQ-free build variant.
- **Contract Conformance Unblocking (51 PASS · 0 FAIL · 1 BLOCKED)** —
  Unblocked 9 previously blocked conformance test cases across capabilities
  manifest parity, single terminal event enforcement, tier and bound declarations,
  deterministic tolerant bounds, opt-in best effort decode, forward schema migration,
  MCP envelope indistinguishability, cross-surface error taxonomy parity, and token
  usage reconciliation.
- **Single Terminal Event Fix** — Resolved agent runtime double-stop bug under
  step budget exhaustion.
- **Cross-Surface Capabilities** — Added `sofuu capabilities` CLI subcommand and
  `/v1/capabilities` HTTP routes for 100% manifest parity with JS and C ABI.

## 0.2.0-beta — 2026-10-26

The runtime-contract program, phases P0–P4:

- **P0** — the normative contract (`docs/SPEC-RUNTIME-CONTRACT.md`, local-only
  by repo policy), the 48-cell surface matrix, the 52-case conformance suite,
  and the tracking artifacts under `contract/`.
- **P1** — the capability model: `sofuu.capabilities()`, `sofuu.model.describe`,
  `sofuu doctor` hard-fails on a missing declared feature, a typed
  `E_CAPABILITY` refusal on QTSQ-free builds, determinism receipts
  (`scripts/text_gate.py`) with the R3 CI gate, and the committed fixture
  model the gate runs (`contract/fixtures/sofuu-fixture.gguf`).
- **P2** — one async contract (`sofuu.async.*`: the §1.4 taxonomy, nested
  deadlines, queue bounds) and one event envelope (`sofuu.events.*`: the §7.2
  schema, validator, redaction, renderer, the per-session `events.ndjson`
  log). Live-path chaos harnesses: deadline kill, mid-SSE cancel, mid-tool
  cancel (subprocess killed, no orphan), server-side disconnect visibility,
  bounded-pump backpressure (`E_BACKPRESSURE`), usage reconciliation
  (`sofuu.events.canonicalUsage`; `ai.complete` reports provider usage).
- **P3** — the session document (`rt/session_doc.rs`, §5.2 v1): one
  export/import format behind `sofuu.session.*` and
  `sofuu session export|import`; the canonical
  `<project>/.sofuu/sessions/<id>/session.json` written by chat;
  restart-stable per-turn ids; `schema_version` on sessions, the event
  envelope, and the config; import-side format allowlisting (R4.5) and
  export redaction (R4.6); serve `/v1/*` OpenAI-wire conformance
  (models / SSE / JSON); the C-ABI symbol gate executable in the suite
  and CI (`CONTRACT_LIBSOFUU`).
- **P4** — one security policy everywhere (R7): the four-surface policy
  matrix (JS / CLI / HTTP / C ABI — same decision, same rule id, no bypass
  by entry point), approval deadline-deny and registration-order
  guarantees in the trace, the shared shell/fs refusal shape
  (`SOFUU_NO_SHELL` strips every surface), net-policy suffix anchoring with
  normalization, secrets-never-emitted checks, the MCP allowlist gate,
  project-policy auto-install at serve boot, and install integrity:
  checksummed `dist/` artifacts + the capabilities-vs-claim + version-parity
  gate (`scripts/verify-dist.sh`).

## 0.2.0-alpha — 2026-09

Engine substrate: in-process inference (GGUF/Metal), the agent loop,
encrypted QTSQ memory, MCP client/server, the C-ABI embed surface.
