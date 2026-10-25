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
| P3 — R4 portability + R5 surface parity | ⬜ not started | `T4.1`/`T4.5` failing; 21 remaining `partial` cells |
| P4 — R7 security + install integrity | ⬜ not started | policy evaluator + approval modes landed by peers (`0b753686c`, `cdd16835b`); four-surface denial matrix still missing |
| P5 — platform matrix + conformance + benchmark | ⬜ not started | CI is macOS-only; suite is local |
| P6 — external adoption | ⬜ not started | no external deployment yet |

## Conformance suite

**52 cases — PASS 26 · FAIL 2 · BLOCKED 24.**
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

Failing (P3 backlog): `T4.1` session export, `T4.5` `sofuu.session`.

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

1. **P3** — session primitive v1 (`T4.1`/`T4.5` — the two remaining suite
   FAILs) and the parity matrix in CI. Golden renders on the macOS/Android
   lanes (T6.3's cross-surface half) ride with those app-lane phases.
2. **Memory clobber** — engine-lane fix for the `sofuu.memory`/`sofuu.brain`
   namespace collision; then matrix memory × js returns to `full`.
3. Refresh the matrix cells affected by the Oct-24 protocol commits (tool,
   transport, policy) — several `partial`/`unavailable` cells are likely stale
   in the good direction.
4. Investigate the live FORK the gate caught (0.6B q4_K default pair, depth 32,
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
