# Progress — runtime contract

Updated **2026-10-26**. This is the live status; the normative text lives in
[`docs/SPEC-RUNTIME-CONTRACT.md`](../docs/SPEC-RUNTIME-CONTRACT.md) and the
sequencing in [`docs/PLAN-AI-APP-RUNTIME-STANDARD.md`](../docs/PLAN-AI-APP-RUNTIME-STANDARD.md)
(both local-only by repo policy). Everything in `contract/` is tracked.

## Phase status

| Phase | State | Evidence |
|---|---|---|
| P0 — inventory + spec verification | ✅ **complete** (2026-10-23, refreshed 2026-10-25) | 48/48 matrix cells resolved, zero unknowns; 132 tiered symbols; 125 knobs with determinism tiers (54 `unsure` open items); 52 conformance cases stubbed |
| P1 — R1 capability model + R3 tiers | ✅ **complete** (2026-10-25) | `sofuu.capabilities()`, `sofuu.model.describe()`, doctor assertion, typed memory refusal on QTSQ-free builds, `text_gate` receipts + `--require-identical`, the R3 CI gate, and the committed fixture model (T3.3) — all landed and verified; suite 12 PASS / 4 FAIL / 36 BLOCKED, and the FORK enforcement demonstrated (exit 3, `|dtop1|=0.2770`) |
| P2 — R2 async contract + R6 observability | ✅ **complete** (2026-10-26) | `sofuu.async.*` + `sofuu.events.*` substrates; envelope stamped at both emit choke points; exactly-one-terminal stop; chaos harnesses T2.1–T2.5 + T2.7/T2.8 all live (mid-tool cancel kills the subprocess — no orphan; server-side disconnect visible; pump backpressure is typed); usage reconciled through `sofuu.events.canonicalUsage` (T6.7, `ai.complete` now reports provider usage); chat appends the turn's envelopes to `events.ndjson`; deterministic renderer pinned byte-for-byte by the committed golden fixtures (T6.3); suite 26 PASS / 2 FAIL / 24 BLOCKED |
| P3 — R4 portability + R5 surface parity | ✅ **complete (2026-10-26)** | the §5.2 session document written once in the engine (`rt/session_doc.rs`) behind `sofuu.session.{list,export,import,sync}` AND `sofuu session export|import` — **T4.1/T4.3 round-trip + idempotence, T4.5 namespace, T4.6 redaction, T4.7 version stamp all executable and green**; chat persists `session.json` with restart-stable `<sid>-t<N>` turn ids; config carries `schema_version` (v0 accepted, forward-only); serve `/v1/*` conformance **T5.2/T5.3/T5.4 executable** against the committed fixture model; **T5.5 C-ABI symbol check executable** (suite + CI via CONTRACT_LIBSOFUU); matrix regenerated (session×js, model×http → full, 0 unknowns). Residue by design: device round-trip (T4.2, needs runners), v0.2 migration fixture (T4.4), MCP fixture server (T5.6), four-surface error parity (T5.8), CHANGELOG/semver (P4), app stores as caches (app lanes) |
| P4 — R7 security + install integrity | ✅ **complete (2026-10-26)** | one policy object, evaluated identically on JS/CLI/HTTP/C ABI (`t72` matrix: same decision + same rule id everywhere, no entry-point bypass); the evaluator now normalizes `scheme://`, `user@`, `:port`, paths, and `[v6]` before the host match (`t77`); approval deadline→typed-deny and the registration-order resolve, both in the trace (`t73`); shell strip + fs jail with the shared refusal shape on every surface (`t75`); secrets never reach trace/log/doc/error (`t78`); MCP allowlist gate (`t79`); serve boots with the project policy; install integrity (`scripts/verify-dist.sh`: checksums, capabilities-vs-claim, doctor, version parity — in CI); CHANGELOG + npm/Cargo/CLI semver aligned. Residue by design: a behavioral C-ABI policy sample (`examples/` is local-only — the S5 config-policy path + the capi suite carry it), an MCP fixture server (T5.6), the four-surface FAILURE-injection harness family (T2.6/T5.8) |
| P5 — platform matrix + conformance + benchmark | ✅ **complete (2026-10-26)** | **Green matrix on `main`: run `40373322014` @ `42eab7318` — macOS-arm64 ✅ (full build + 670/0 Rust unit tests + JS suite + contract suite + 32-token smoke + conformance + ABI/public-API/verify-dist/SDK checks), Linux-x86_64 ✅, Linux-arm64 ✅ (build, report-mode suite, **32-token tiny-model smoke**, `sofuu conformance` artifact each), iOS SDK typecheck ✅.** Windows x86_64 stays the recorded fix-forward: vendored QuickJS does not compile under MSVC (vendored-deps lane), `continue-on-error` so it never masks the rest. The attached macOS report is the release artifact: **52 cases — PASS 42 · FAIL 0 · BLOCKED 10**, `capability_posture_ok: true` (memory `capability-gated / typed-refusal`), `install_integrity.verify_dist: ok`, i.e. the §8 claim and the conformance claim are one file. Getting here fixed four latent defects (details below): ARM-only guards in `src/simd/infer_kernels.cpp` + `infer/lfm2.rs`/`qwen35.rs` (`d18bb0335`), capi tests asserting pre-P1 QTSQ-free behavior — which also caused the 35-minute macOS timeout via leaked runtimes (`d18bb0335`), `typeof`-based capability claims in the conformance probe (`42fa4abda`) and verify-dist (`6848e5046`), and the public-API step running Python through bash (`7ef1fb263`). Deliverables: `sofuu conformance` per-release JSON (manifest + live capability posture + suite summary), `scripts/conformance_report.sh` merged report (auto-enables T5.5/T1.3 when the artifact set supports them, `42eab7318`), the **§8 benchmark** (`benchmark/PROTOCOL.md`, Build A measured → `benchmark/table-build-a.json`: 49 lines, 0 deps, 1 process, 0 model-swap lines, offline-yes; Build B compose repro), and the blocking public-API audit (`scripts/check_public_api.py`; `examples/headless` is the C-ABI lane's boundary fixtures). The QTSQ-enabled job remains the documented opt-in (`vars.SOFUU_QTSQ_CI` + secret checkout) |
| P6 — external adoption | ⬜ not started | no external deployment yet |

## Conformance suite

**52 cases — PASS 42 · FAIL 0 · BLOCKED 10.**
(`CONTRACT_LIBSOFUU=dist/libsofuu.dylib` now enables the C-ABI parity check.)
(`CONTRACT_NOQTSQ_BIN=target-noqts/release/sofuu` enables T1.3.)

Passing:

| Test | What it proves |
|---|---|
| T1.1 | `sofuu.capabilities()` returns a manifest with `schema: 1` |
| T1.3 | a QTSQ-free build refuses `sofuu.memory.remember()` with typed `E_CAPABILITY` |
| T1.4 | `sofuu doctor` prints the capability section and exits non-zero on a missing declared feature |
| T1.5 | `sofuu.model.describe(path)` exists |
| T1.6 | the matrix has no unknown cells |
| T1.7 | `engine.paths` in the manifest declares a tier for every path (`cpu-tiles` exact, `metal-mma` tolerant, `mmq` best-effort) |
| T3.2 | CI runs the determinism gate (`scripts/text_gate.py`) |
| T3.3 | the committed fixture model runs the full two-route exact-tier gate (exit 0, IDENTICAL) |
| T3.4 | the gate emits `tier`/`backend`/`sha256`/`knobs` receipt fields |
| T5.1 | the generated surface matrix has no unknowns |
| T5.7 | the ABI gate (`abi-check`) is wired in CI |
| T7.6 | `SOFUU_NO_SHELL=1` removes `bash` from the advertised tool group |
| T2.1 | a silent child is killed at its deadline (±2 s) with no orphan |
| T2.2 | cancel terminates a real SSE stream promptly; usage accounted |
| T2.3 | cancel mid-tool kills the tool's live subprocess via the registered signal — run settles promptly, no orphan |
| T2.4 | a client disconnect is visible server-side (`res.closed()`) |
| T2.5 | a non-draining consumer gets typed `E_BACKPRESSURE` (bound named) and the provider request is aborted |
| T2.7 | nested deadlines inherit (parent − margin) and refuse exhausted parents |
| T2.8 | one error shape + the ten-code taxonomy, defaults per code |
| T6.1 | the §7.2 identity fields ride on every real agent AND chat event; the runtime validator reports zero violations |
| T6.2 | the session event log round-trips: append → read → reconstruct (answer, tools, usage, terminal) |
| T6.3 | `sofuu.events.render` is deterministic and matches the committed golden bytes exactly |
| T6.5 | seq strictly increasing; exactly one terminal `stop` per run |
| T6.6 | redaction strips secret-bearing fields recursively before any log/write |
| T6.7 | usage reconciles: `ai.complete` reports `{promptTokens, completionTokens}`, stream matches the same provider totals, agent shape normalizes identically, local cost 0 / cloud cost carried |
| T4.1 | the §5.2 session document round-trips CLI⇄JS: same turns, tools, usage, ids; one builder serves both surfaces |
| T4.3 | turn ids are stable shapes and re-import is idempotent (`overwritten_existing` false on a same-content re-import) |
| T4.5 | `sofuu.session` is a first-class namespace (`schemaVersion/list/export/import/sync`); an app-private top-level section is rejected, not tolerated |
| T4.6 | an export carrying secrets ships none (`api_key`/`Authorization`/tool tokens stripped), while lossless fields survive |
| T5.2 | `GET /v1/models` returns the OpenAI `{object:'list', data:[…id, object:'model'…]}` shape for the loaded fixture model |
| T5.3 | `POST /v1/chat/completions` `stream:true` yields OpenAI delta-frame SSE chunks terminated by `[DONE]` |
| T5.4 | `POST /v1/chat/completions` JSON matches the OpenAI completion shape (choices[].message.{role,content}, usage) |
| T5.5 | the C ABI keeps the versioned symbol contract: `sofuu_rt_call` exported from the built library (suite + CI) |
| T7.1 | one policy object denies the same operation with the same rule id on JS, CLI (`sofuu policy why`), HTTP (serve 403 + rule), and the C ABI |
| T7.2 | no bypass by entry point: the real tool belt refuses under policy on JS while serve refuses the same call over HTTP |
| T7.3 | a pending approval whose deadline passes resolves to a typed deny, recorded in the trace with the tool named |
| T7.4 | resolving immediately out of the request event is allowed — never raced to deny (pre-registration guarantee) |
| T7.5 | `SOFUU_NO_SHELL`/shell-strip and the fs jail answer identically on JS/CLI/HTTP; a direct HTTP bash call is refused, never executed |
| T7.7 | net-policy wildcards are suffix-anchored and `scheme://`, `user@`, `:port`, paths, and `[v6]` normalize before the match |
| T7.8 | a provider key never reaches the trace, the event log, the session document, or error text (even a 401 that echoes the key) |
| T7.9 | the MCP allowlist refuses non-allowlisted servers; allowlisted tools still answer through the same tools policy |

No failing cases. `T4.1`/`T4.5` (P3) and the eight T7 cases (P4) all run in the suite and CI.

## What landed in P2 (2026-10-25/26)

- **`sofuu.async.*`** (`rt/async_contract.rs`) — the §1.4 error taxonomy with
  per-code retryability, `makeError` (unknown codes wrap as `E_INTERNAL`),
  `childDeadline` (parent − margin inheritance; exhausted parents report
  `E_TIMEOUT`), and `limits()` (bounded queues + the §3.2 default timeouts).
- **`sofuu.events.*`** (`rt/events.rs`) — the §7.2 envelope: schema (10
  required kinds with payload rules + 23 known driver kinds), a validator,
  recursive redaction, a deterministic renderer, and the per-session
  `events.ndjson` log with seq-order enforcement and an `E_BACKPRESSURE`
  overflow bound.
- **Envelope stamped at both emit choke points** — `agent.js` and `chat.js`
  additively carry `schema_version/seq/ts/surface/sid/turn_id` on every event;
  existing consumers are unaffected (JS suite 59/1, the 1 pre-existing).
- **Exactly-one-terminal** — `agent.js finish()` emits `stop {reason:
  done|budget|cancelled}` once; a real run's trace shows the total order.
- **Live-path chaos harnesses** (tracked, `scripts/contract/harness/`) —
  deadline kill + orphan check, cancel mid-SSE with usage accounting, **cancel
  mid-tool through the real agent loop** (the approved `sleep` subprocess dies
  with the run — the approval decider is a run-opts `onApproval` callback, and
  `sofuu.tools.killActive()` tracks/kills the spawned process), **server-side
  disconnect visibility** (`peer_closed` on the client + `res.closed()` for
  handlers), and **bounded-pump backpressure** (1024 undelivered chunks →
  `E_BACKPRESSURE`; the abort is deferred out of the curl write callback —
  calling `stream_abort_one` in-callback spins the multi loop and starves every
  timer).
- **Usage reconcile (T6.7)** — `ai.complete` parses `usage` out of the response
  body (`extract_json_field` learned non-string JSON values — the string-only
  parser silently dropped every numeric field); `sofuu.events.canonicalUsage`
  unifies the chat/agent/complete shapes to `{in,out,model,provider,cost_usd}`.
- **Golden render (T6.3)** — `sofuu.events.render` output is pinned by the
  committed fixtures `contract/fixtures/golden-trace.{ndjson,txt}`; the harness
  re-renders the committed events and compares bytes.
- **Chat adopts the ndjson log** — `chat.js sessionLog` additionally appends
  the turn's envelopes to `<project>/.sofuu/sessions/<sid>/events.ndjson`
  (usage normalized to the canonical shape; additive, never breaks the turn).

## What landed in P3 (2026-10-26)

- **The §5.2 session document, written once** (`rt/session_doc.rs`) —
  build, validate, redact, persist, import, list. The CLI and the JS
  namespace call the SAME functions, so the two surfaces cannot drift.
  `session.json` lives at the canonical location next to `events.ndjson`.
- **`sofuu.session`** (`schemaVersion/list/export/import/sync`) — the
  session primitive on the JS surface (T4.5). Import enforces the format's
  top-level allowlist (R4.5: an app-private section is a typed refusal).
- **`sofuu session export|import`** — the CLI arm (T4.1): export prints or
  `--out`s the document (secrets stripped first, R4.6); import validates
  (typed `E_SCHEMA` refusal, R4.4), persists under its own id, and reports
  whether it replaced an existing, different doc.
- **Chat writes the canonical document** — `sessionLog` now emits
  restart-stable per-turn ids (`<sid>-t<N>`, seeded from the restored turn
  count, R4.3) and regenerates `session.json` at each turn terminal via
  `sofuu.session.sync`; the fold reconstructs turns, tools, and usage from
  the event log alone.
- **Config carries `schema_version: 1`** — every save stamps it; the
  reader accepts absent (v0) and normalizes on the next save; the field
  set is identical, so v0→v1 is lossless by construction (R4.1/R4.4).
- **serve `/v1/*` conformance (T5.2–T5.4)** — a harness boots `sofuu
  serve` with the committed 77 KB fixture model and asserts the OpenAI
  wire shapes: models list, SSE delta frames + `[DONE]`, and the JSON
  completion. Requires the bearing token — the harness sets one.
- **C-ABI parity gate executable (T5.5)** — the suite (and CI) pass
  `CONTRACT_LIBSOFUU=dist/libsofuu.dylib`; the symbol contract check
  replaced a manual-blocked case.
- **Matrix regenerated** — session×js and model×http moved to `full` with
  evidence + test ids; 17 `full` cells, 0 unknowns.

Harness notes of record: `t61`'s chat leg is raced with a dead-local
`base_url` (a host's active real provider must never leak a network call
into a contract harness) and kills its tool children so later orphan
checks see a clean process table.

## What landed in P5 (2026-10-26)

- **The platform matrix** — a `platform` CI job over Linux x86_64,
  Linux arm64 (native runners), and Windows x86_64 (the long-plumbed
  build.rs Windows path: cmake builds libuv's `uv_a.lib`, `SOFUU_UV_DIR`
  points at it). Every platform: QTSQ-free capability-gated build, the
  contract suite in report mode, the **32-token tiny-model smoke**
  (`scripts/smoke32.sh` — the committed 77 KB fixture model decodes 32
  tokens in seconds, no network), and a `sofuu conformance` artifact.
  The macOS job runs the same smoke and uploads the merged report.
- **`sofuu conformance`** (`src/conformance.rs`) — the per-release report:
  version, platform, the FULL capability manifest, and the LIVE
  capability posture (memory full-or-typed-refusal, the P2/P3/P4
  surfaces present) with `capability_posture_ok`. The probe is honest
  about the engine-lane memory finding (a clobbered namespace reports
  `not-a-function`, not a claimed pass) — the posture stays visible
  until the engine lane lands the namespace fix.
- **`scripts/conformance_report.sh`** — one release artifact: the
  52-case board's summary parsed into the runtime's report + the
  verify-dist gate verdict. Attach it to every release.
- **The QTSQ-enabled job (opt-in)** — `vars.SOFUU_QTSQ_CI=1` + a secret
  checkout URL; the documented QTSQ/publish posture stays
  capability-gated CI otherwise.
- **The §8 benchmark, committed** — `benchmark/PROTOCOL.md` (the table
  the claim lives or dies by), **Build A** (`benchmark/build-a/app.js`,
  measured: 49 app lines, 0 dependencies, 1 process, 0 daemons, model
  swap = 0 lines — one env var, offline = yes with every feature,
  resume via the R4 session document; `scripts/bench_build_a.sh`
  generates `benchmark/table-build-a.json` per release) and **Build B**
  (`benchmark/build-b/` — the fragmented stack committed as a compose
  repro: 3 daemons, two supply chains). A stranger reproduces both from
  the directory alone.
- **Public-API audit** — `scripts/check_public_api.py` (blocking CI
  step): app references (React frontend, benchmark) may not touch the
  runtime-internal `__qtsq_/__sofuu_/__session_/__chat_/__ai_` seam;
  `examples/headless/*` is the C-ABI lane's own boundary fixtures and is
  excluded by scope, not by exception.
- **The matrix's first x86_64 compile found real latent breakage**
  (`d18bb0335`). `src/simd/infer_kernels.cpp` had grown ARM-only
  tile/GQA entry points and an aarch64-only parity test with no guards:
  every x86_64 build died on undeclared `tile_block_*` types, the x86
  fallback of the rows8 attention kernel read the NEON-only `lens[]`
  (fixed to derive lengths from `counts[]`), and Rust references three
  ARM symbols (`sofuu_tile_warm`, `sofuu_tile_scale_sidecar_mode`, the
  GQA counters) unconditionally, so the non-ARM build needs inert stubs.
  Same class in `infer/lfm2.rs` (pointer-form `dot_f32_neon` + the NEON
  4×4 tile called from the fused dual-panel path) and `infer/qwen35.rs`
  (file-direct packed dots are aarch64-only). ARM code is byte-for-byte
  unchanged; the zig `x86_64-linux-musl` build now completes (5.7 MB
  static-pie) and CI proves it by running the fixture inference.
- **The macOS job's 35-minute timeout was test fallout, not a hang in
  the runtime.** Two capi tests still asserted pre-P1 behavior (a
  QTSQ-free build has NO `memory.*`); since the typed-refusal stubs
  landed, they resolve and throw, so the assertions panicked *before*
  `sofuu_rt_free`, leaking QuickJS runtimes that wedged the next
  cross-thread test on the process-global libuv loop. Tests updated to
  the P1 contract (resolution everywhere; `E_CAPABILITY` at
  `memory.open`; the capi facade's "no open brain" TypeError for
  handle-taking methods); a QTSQ-free `--test-threads=1` run is 54/54
  in ~3.5 s.
- **The conformance report's memory row was initially dishonest**
  (`42fa4abda`): `typeof sofuu.memory.remember === "function"` is true
  for the typed-refusal stubs too, so a QTSQ-free artifact claimed
  `live: "full"`. The probe now reads `capabilities().build.missing`
  to know which class the binary must be, then proves the gated class
  by calling `remember` and requiring the `E_CAPABILITY` refusal (a
  silent no-op is reported as a defect). Verified both ways on one
  commit: QTSQ-free → `{capability-gated, typed-refusal, ok:true}`;
  QTSQ-on → the engine-lane namespace clobber stays visible as
  `{full, undefined, ok:false}`.
- **verify-dist told the same typeof lie, and failed the declared
  posture** (`6848e5046`): on the QTSQ-free CI build the stubs made
  "memory available (QTSQ build)" true, while doctor — correctly —
  exited non-zero for the declared qtsq gap, so the gate failed the
  very posture the manifest documents. It now classifies from the
  manifest and requires the behavioural refusal on a gated build; with
  `SOFUU_ALLOW_NO_QTSQ=1` doctor is accepted only when every failing ✗
  is the qtsq gap. A QTSQ-bearing build now gets the truthful FAIL
  naming finding #13 (the old check silently passed it as "absent but a
  typed refusal").
- **A timing-flake test, not a runtime defect** (`eb219a761`): the
  sandbox deadline test's 10 ms timeout raced the default 5M fuel
  budget on a loaded runner and reported `fuel`. Fuel is pinned at the
  clamp ceiling so only the deadline can fire, and the promptness bound
  is 2 s. (Separately, the lfm2 bf16 prefill parity test loaded a
  hardcoded `/Users/.../Downloads` checkpoint and could never pass on a
  runner — `#[ignore]`d like its mlx4 sibling, `0d03116a8`.)
- **The release report under-counted its own board** (`42eab7318`):
  only the workflow's suite step exported `CONTRACT_LIBSOFUU` /
  `CONTRACT_NOQTSQ_BIN`, so the artifact's re-run showed T5.5/T1.3
  BLOCKED (40/0/12) right after the job had executed them. The report
  now auto-detects a built `dist/libsofuu.*` (relative — the runner
  resolves it against the repo root) and a binary whose manifest
  declares qtsq missing. The macOS artifact now carries the same
  42/0/10 board the suite step proves.
- **One CI wiring bug**: the public-API audit ran a Python script
  through bash (`7ef1fb263`), so the step died with `__qtsq_*: command
  not found` after the unit suite had gone green. Now
  `python3 scripts/check_public_api.py`.

## What landed in P4 (2026-10-26)

- **The deny matrix that proves R7.1/R7.2** (`t72_policy_matrix.js`) — ONE
  policy object installed by value; JS (`sofuu.policy.evaluate` + the real
  tool belt), CLI (`sofuu policy why`), HTTP (`serve` 403 `policy_denied`
  carrying the rule), and the C ABI (the S5 create-config policy path +
  the symbol gate) all return the same decision with the same rule id.
- **The evaluator now normalizes hosts** — `norm_host` strips `scheme://`,
  `user@`, `:port`, path suffixes, and reduces `[v6]` literals before the
  suffix anchor match. This closes a real R7.1 drift where the agent's JS
  netPolicy normalized but the Rust evaluator did not (`t77`).
- **Approvals in the trace** (`t73_approval_harness.js`) — the deadline
  resolves to a typed deny with the tool named (T7.3), and a resolve that
  arrives with the request event is allowed, never raced to deny (T7.4).
- **The shared refusal shape** (`t75_shared_shape.js`) — `SOFUU_NO_SHELL`
  strips the belt everywhere; serve boots with the project policy; the
  fs jail's escape refusal is one text on every surface.
- **Secrets-never-emitted** (`t78_secrets_never_emitted.js`) — a live key
  survives neither a successful trace, a 401 that echoes the Authorization
  header, the event log, nor the session document.
- **MCP allowlist** (`t79_mcp_allowlist.js`) — non-allowlisted servers
  refused with the rule; allowlisted tools still pass the tools policy.
- **Install integrity** — `scripts/verify-dist.sh` stamps/verifies
  `dist/SHA256SUMS`, asserts the live capabilities against the build's
  claim (memory full vs typed-refusal, session/R6 surfaces), runs doctor,
  and checks Cargo↔CLI↔npm version parity; a CI step runs it after
  `make libsofuu`. npm aligned to `0.2.0-beta`; `CHANGELOG.md` added.
- **serve joins the policy story** — `sofuu.serve.start` installs
  `<project>/.sofuu/policy.json` at boot, so HTTP cannot bypass a deny
  made for chat/CLI.

Harness lessons of record: a JS-created mock server must be held by a
module-scope reference for the run's lifetime — an unreachable listener is
garbage-collected and the run sees connection-refused (`t73`'s KEEPERS);
port ranges for contract harnesses rely on `Date.now()` offsets and collide
with concurrently-benching lanes — the P4 family moved to 38–39k.

## The fixture model and the FORK demonstration (T3.3)

`contract/fixtures/sofuu-fixture.gguf` is a **77 KB, 1-layer llama-arch model**
generated by `scripts/contract/make_fixture_gguf.py` and committed. It is a
real model the engine loads and runs: q8_0 weights (the loader slices per row,
and q4_K needs row lengths that are multiples of 256), a 256-byte-token vocab
so any prompt encodes deterministically, and all-decode-to-zero q4_K-style
bounds that keep every value finite. CI regenerates it and asserts the
committed bytes match the generator, then runs the gate:

```
[ci] VERDICT=IDENTICAL (2 steps token-equal)      → exit 0
```

The enforcement is demonstrated on the real q4_K default pair (0.6B, depth 32):

```
[enforce-deep] VERDICT=FORK first-fork-step=2 |dtop1|=0.2770 (bound 0.05)
[enforce-deep] exact-tier gate FAILED: verdict=FORK (required IDENTICAL, …)
→ exit 3
```

The `|dtop1|=0.2770` matches the parity ledger's historical receipt for the
same pair bit-for-bit — the q4_K default pair still forks at depth on the GPU
route, and the gate now catches it automatically in CI.

## What landed in P1 (2026-10-25)

- **`sofuu.capabilities()`** — the manifest: runtime/build/engine/model/surfaces,
  with the surface matrix embedded from `contract/matrix.json` at build time
  (`crates/sofuu-core/src/capabilities.rs`).
- **`sofuu.model.describe(path)`** — per-model capability block read straight
  from the GGUF header (arch, ctx, blocks, heads, quant types, size), with a
  typed `E_CAPABILITY` envelope for unreadable files. Verified against a real
  1B model: `{"arch":"llama","ctx":131072,"quant":"q8_0","size_mb":1259,…}`.
- **Doctor asserts the manifest** — a `capabilities` section that fails doctor
  (exit 1) when a declared feature is missing; the JSON report carries the full
  manifest.
- **Typed memory refusal** — on QTSQ-free builds, `sofuu.memory`/`sofuu.brain`/
  `sofuu.kv`/`sofuu.storage` register as throwing stubs: `E_CAPABILITY` with the
  rebuild remediation, instead of the silent no-op that shipped the brain-less
  2026-09-22 tarball.
- **Determinism receipts** — `scripts/text_gate.py` emits the §1.6 receipt
  (kind, schema_version, model sha256, backend, tier, knobs, prompt_id,
  verdict, numbers, ts), validates it (`--self-check` in CI on every push), and
  `--require-identical` fails the exact-tier gate on any non-IDENTICAL verdict.
- **CI gate** — `.github/workflows/ci.yml` gained the R3 step (schema check
  always; full two-route gate when `SOFUU_CONTRACT_MODEL` is set; FORK or a
  non-identical exact run fails the job).
- **Knob tiers** — `contract/knobs.json` now maps every numeric knob to a
  determinism tier (R3.5): 5 receipt-backed `exact`, 1 receipt-backed
  `tolerant`, 54 honest `unsure` open items listed in `KNOBS.md`.

Verification: main release build clean; 5/5 capability unit tests pass; a
genuine QTSQ-free build (`SOFUU_QTSQ_DIR=/nonexistent SOFUU_ALLOW_NO_QTSQ=1`,
`target-noqts/`) shows doctor exit 1 + the typed refusal.

## New P0-class finding (2026-10-25)

**`sofuu.memory` and `sofuu.brain` are clobbered by the zerocopy bridge on
every build.** `rt/zerocopy.rs` registers `sofuu.memory` (the alloc/view buffer
bridge) after `rt/memory.rs` registers the CMA brain on the same names, so on
the release build `typeof sofuu.memory.remember` is `undefined` and the CMA
brain is unreachable from JS. Chat's brain keeps working via internal
`__qtsq_*` globals, which is why the loss is silent. Recorded in
`matrix.json` (memory × js → `partial`) and in the spec's findings table. The
fix belongs to the engine lane: distinct namespaces (e.g. `sofuu.buffer`) or
merged registration — not done here to avoid crossing lanes.

## Next actions

1. **Windows x86_64** — vendored QuickJS under MSVC (`quickjs.c` C2146 —
   the vendored-deps lane's port; the platform job is `continue-on-error`
   until it lands). **Android compile job** — the opt-in
   `vars.SOFUU_ANDROID_CI` path exists but has not been exercised.
   **macOS runner availability** — two pushes today were cancelled while
   queued for a `macos-latest` runner; a targeted job re-run acquired one
   and went green, so treat a queued-then-cancelled macOS job as
   infrastructure, not a code verdict.
2. **Local-load flakes worth knowing** — `t21_timeout` (T2.1) and
   `t23_cancel_tool` (T2.3) can fail their wall-clock/orphan bounds when
   the machine is saturated by concurrent builds (seen once each locally;
   green on CI and when re-run alone). They are report-mode cases, but a
   future blocking flip should give them load-aware bounds.
3. **Flip the suite to blocking** — the board is 42/0/10 and the FAIL list
   is empty, but the runner exits 1 when cases are BLOCKED, so every CI
   step still runs it in report mode. The flip needs a FAIL-only exit code
   (0 with blocked>0) before `|| echo` can be removed.
4. **Residual harness families** — failure injection (T2.6), the v0.2
   migration fixture (T4.4), an MCP fixture server (T5.6), the four-surface
   error-parity harness (T5.8), cloud+local fixtures (T6.4), and a
   behavioral C-ABI policy sample (examples/ is local-only — the S5
   create-config policy path and the capi suite carry it behaviorally).
5. **App-lane residue from P3** — the device round-trip (T4.2) on
   macOS+Android runners, the v0.2 migration fixture (T4.4), an MCP fixture
   server (T5.6), the four-surface error-parity harness (T5.8), the app
   stores becoming caches of `session.json` (Android `SessionMessageStore`,
   macOS), and the `/v1/*` README section.
6. **Memory clobber** — engine-lane fix for the `sofuu.memory`/`sofuu.brain`
   namespace collision; then matrix memory × js returns to `full`, and
   verify-dist's memory row stops failing on QTSQ-bearing builds.
7. Refresh the remaining `partial` cells (tool/usage on the app lanes) as the
   protocol work lands; several are stale in the good direction.
8. Investigate the live FORK the gate caught (0.6B q4_K default pair, depth 32,
   `|dtop1|=0.2770`): it matches the parity ledger's known class, and the gate
   will keep it out of defaults automatically.

## Reproduce

```bash
python3 scripts/contract/gen_contract_docs.py     # validate + regenerate
bash scripts/contract/run_contract_tests.sh       # current binary
CONTRACT_NOQTSQ_BIN=target-noqts/release/sofuu \
  bash scripts/contract/run_contract_tests.sh     # + the T1.3 refusal gate
SOFUU_QTSQ_DIR=/nonexistent SOFUU_ALLOW_NO_QTSQ=1 \
  CARGO_TARGET_DIR=target-noqts cargo build --release   # the no-QTSQ binary
```