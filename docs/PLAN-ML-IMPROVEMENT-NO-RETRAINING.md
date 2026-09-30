# PLAN-ML-IMPROVEMENT-NO-RETRAINING — Maximize Sofuu ML Without New Training

> Status: partially implemented (Phase 0 + Phase 1 shipped 2026-08-30)
>
> Shipped so far (all verified by `cargo test`):
> - Phase 0: the baseline freeze lives in `crates/sofuu-core/ml-baseline.json`
>   (dims, param counts, blob byte lengths, CRC32/FNV-1a/SHA-256 identities,
>   feature-slot tables, thresholds, mechanical-rule constants) and is
>   verified against the live code by
>   `crates/sofuu-core/tests/ml_baseline_freeze.rs` on every test run.
> - Phase 1.1: allocation-policy tests are parallel-safe — the evidence
>   ladder is now a pure `resolve_with(Evidence)` seam, store-touching
>   tests serialize on a shared `ml::TEST_LOCK`, and `cargo test` is green
>   with default parallel threads (previously 4 failures).
> - Phase 1.1 fix: test runs no longer persist/load the discovered-caps
>   disk file (`cfg!(test)` isolation in `model_caps_discovered.rs`) —
>   tests were clobbering the user's real harvested caps
>   (`~/.sofuu/ml/model_caps.json`) with example.com fixtures; that
>   residue was cleaned out (588 real models kept).
> - Phase 1.2: every native ML JS boundary is hardened (bounded strings,
>   arrays, and numerics; non-finite sanitization; escaped output fields)
>   and proven end-to-end by `crates/sofuu-core/tests/ml_shim_hardening.rs`
>   driving the real QuickJS engine with malformed/oversized/Unicode/
>   adversarial inputs.
> - Phase 1.3: feature extractors survive hostile inputs — saturating
>   i64→f32 casts (alloc), NaN-proof clamps (freshness/relevance
>   strength, age), a saturating token sum (compaction overflow fix), and
>   hostile-input tests on all five extractors.
> - Phase 1.4: blob integrity extended — non-finite weight refusal and
>   trailing-data refusal in `net.rs::from_blob` (plus tests).
>
> Not yet implemented: Phases 2–8 (§6–§12). The order below is unchanged.
>
> Goal: improve the useful behavior, safety, speed, and measurable reliability
> of Sofuu's existing ML layer without increasing parameter counts and without
> retraining or obtaining a better training set.
>
> This document's remaining phases are planning only. A phase is
> "authorized" for implementation only phase by phase, one shippable unit
> at a time, each verified before the next begins.

## 1. Scope and constraints

This plan covers the main Sofuu runtime only:

- `crates/sofuu-core/src/ml/`
- the ML call sites in `src/js/agent.js`, `src/js/chat.js`, and the Rust chat driver
- the dev-only evaluation and test harness in `crates/ml-train/`

The following constraints are hard requirements:

1. Do not increase any model's parameter count.
2. Do not change hidden-layer widths, output-layer shape, or the meaning/order of
   existing feature slots while the current weight blobs remain installed.
3. Do not retrain, generate new training data, or re-bake weights in this plan.
4. Do not turn an advisor into a silent filter. The LLM remains the final decision
   maker for relevance, freshness, and supervisor advice.
5. Keep all safety clamps, hard provider-limit handling, and compaction protections
   outside the learned score.
6. Do not touch the desktop app.
7. Do not touch the QTSQ repository. This plan does not require QTSQ changes.
8. Preserve the offline, deterministic, no-API runtime contract.

### What is allowed

- Better input hygiene and bounded preprocessing that preserves feature semantics.
- Stronger deterministic rules around the existing score.
- Threshold, confidence-band, notice, and rollout policy changes.
- Better runtime placement, deduplication, caching, and failure handling.
- Better evaluation, replay, instrumentation, and test isolation.
- Safe operational improvements to the existing output-layer online-learning
  controls, without changing its learning algorithm or enabling it by default.

### What is deliberately deferred

Replacing a learned feature with a genuinely new feature, adding input dimensions,
changing the network architecture, or changing the semantics of a feature slot
would require a compatible weight re-bake. Those are future work after training is
available and are not part of this plan's implementation path.

## 2. Current baseline

The shipped models currently use the shared scalar `TinyMlp` and CRC-checked baked
`SML1` weight blobs:

| Model | Architecture | Parameters | Runtime role |
|---|---:|---:|---|
| freshness | `28 → 112 → 48 → 1` | 8,721 | advises whether material may be stale |
| relevance | `37 → 104 → 44 → 1` | 8,617 | advises which recalled candidates look useful |
| supervisor | `33 → 104 → 44 → 1` | 8,201 | advises before calls and at loop boundaries |
| compaction | `33 → 104 → 44 → 1` | 8,201 | selects safe candidates for compaction |
| alloc | `24 → 96 → 40 → 1` | 6,321 | allocates context/output/tool budgets |

The current code already has several high-value protections:

- exact repeat and unchanged-file reread rules;
- relevance duplicate and never-use rules;
- compaction keep-window, duplicate, boilerplate, and retrievable tiers;
- provider capability discovery and error-learned limits;
- model-aware hard clamps around the allocation output;
- output-layer-only online adaptation with trust region, replay anchors, batch
  floor, adoption gate, and pretrained-weight hashing;
- `/ml off` and `SOFUU_NO_ML=1` fallbacks;
- try/catch boundaries around JavaScript ML calls.

Validation observed during the source review:

- The complete ML unit suite passed serially: 96/96.
- A default parallel ML run had 4 failures in allocation-policy tests because
  tests mutate shared discovered-cap and learned-limit state concurrently. The
  same 16 policy tests pass with one test thread. This is a test-isolation defect,
  not evidence that the model scores are wrong, but it prevents a trustworthy
  default green build.
- `cargo run --release -p ml-train -- eval` currently evaluates the freshness
  model only. The other committed models have in-crate fixture tests, but the
  trainer command does not yet provide one aggregate report for all models.

## 3. Success criteria

The work is successful only when all of the following are true:

- Parameter counts, feature dimensions, weight blobs, and inference math remain
  unchanged.
- No new network call or model download is introduced.
- Malformed, oversized, empty, Unicode-heavy, and adversarial inputs cannot panic,
  emit invalid JSON, exceed bounds, or break a chat turn.
- The default test command is deterministic and green; it does not require a
  special serial-test flag.
- Advice is shorter, less repetitive, more evidence-carrying, and reaches the
  model at the correct boundary.
- Mechanical high-confidence cases are handled before the net; ambiguous cases
  remain advisory and do not cause silent data loss.
- ML-disabled behavior remains equivalent to the pre-ML path within existing
  compatibility expectations.
- p50/p95 latency and temporary allocation costs are measured for every gate and
  stay within an agreed budget.
- Every claimed quality improvement has a before/after replay or fixture result;
  threshold changes are never tuned against the final test fold.

## 4. Phase 0 — freeze the baseline and invariants

### 4.1 Record immutable model facts

- Record each model's input dimension, hidden dimensions, parameter count, weight
  blob byte length, CRC, and hash.
- Add a single machine-readable baseline report for those values so a future
  change cannot silently alter the shipped model.
- Record the exact feature-slot names and order from each `features.rs` file.
- Record each model's threshold and mechanical-rule constants.
- Record the current ML API JSON contracts and fallback behavior.

### 4.2 Record behavior and resource baselines

Build a deterministic replay corpus from existing fixtures and hand-authored
counterfactuals. Do not use it as training data. Capture:

- score and verdict for every model;
- reason, source, threshold, and nudge for every flagged case;
- false-advice and missed-advice counts by model;
- notice count and approximate notice tokens per turn;
- model invocation count per turn;
- p50/p95/p99 inference latency;
- heap allocation count/bytes where measurable;
- behavior with ML enabled, `/ml off`, and `SOFUU_NO_ML=1`.

### 4.3 Freeze compatibility rules

Document that an existing weight blob is valid only with its matching feature
schema. Any future schema change must either:

- be a semantics-preserving bug fix with evidence that the old weights still hold;
  or
- wait for a later re-bake and versioned blob.

## 5. Phase 1 — correctness and reliability first

This phase changes no learned behavior. It removes defects that can make a good
model appear bad or can make a correct score unsafe to consume.

### 5.1 Make allocation-policy tests isolated and parallel-safe

The discovered-cap and learned-limit stores are process-global by design, while
the tests currently clear and repopulate them independently. Choose one explicit
test-only isolation mechanism:

- a shared test mutex held across each stateful test; or
- a scoped state snapshot/restore guard that cannot interleave; or
- a pure policy seam that accepts an injected evidence store, with production
  lookup remaining global.

Requirements:

- the production concurrency model must not be weakened;
- tests must pass with the normal Cargo test-thread setting;
- serial execution must continue to pass;
- no test may leak discovered or learned values into another test.

### 5.2 Harden every native ML boundary

For every registered API:

- validate argument count and string type;
- handle malformed JSON as a safe no-op or conservative result;
- cap input text, candidate count, segment count, and array sizes before feature
  extraction;
- reject or sanitize non-finite numeric values;
- guarantee scores remain in `[0, 1]`;
- guarantee IDs in `use`, `skip`, `keep`, and `compact` are valid and unique;
- guarantee all returned strings are valid JSON and contain no embedded NUL;
- ensure a poisoned or unavailable mutex cannot abort a chat turn.

### 5.3 Harden feature extraction

Test and enforce behavior for:

- empty and whitespace-only text;
- very long text;
- invalid UTF-8 at FFI conversion boundaries;
- mixed Unicode case and punctuation;
- control characters and embedded NULs;
- repeated separators and path-like strings;
- giant candidate menus and segment histories;
- zero denominators and overflows in ratios, BM25, age, token, and budget math.

### 5.4 Weight and architecture integrity

Keep and extend the existing blob checks:

- magic/version/CRC validation;
- architecture and parameter-count match;
- corruption refusal;
- bounded finite weights;
- explicit failure if a blob is truncated or has trailing unexpected data.

## 6. Phase 2 — maximize value from existing signals without new parameters

The current weights must continue to see the feature meanings they were trained
against. The work in this phase is therefore a safe preprocessing and composition
layer, not a new feature schema.

### 6.1 Canonicalize inputs without changing feature meaning

- Use one shared bounded text-normalization path for case, whitespace, line
  endings, and control characters.
- Preserve URLs, versions, dates, file extensions, symbols, and identifiers that
  the current features intentionally inspect.
- Make truncation evidence-preserving: retain both a bounded head and tail when
  dates, versions, errors, or conclusions may appear near the end.
- Include an explicit truncation marker so the caller can avoid treating a partial
  sample as complete evidence.
- Apply the same canonicalization in replay tests and runtime calls.

Before shipping, compare score distributions and fixture margins against the
frozen baseline. If a normalization change shifts a model's learned semantics,
revert it until a compatible weight re-bake is available.

### 6.2 Reuse computed work

- Cache anchor vectors and other immutable embeddings through the existing lazy
  initialization path.
- Within one relevance or compaction call, compute shared embeddings once and
  reuse them across candidates/segments.
- Within one agent boundary, avoid rescoring identical text/task/kind tuples.
- Cache only bounded, content-keyed, non-sensitive feature intermediates; never
  persist raw user content merely for an ML optimization.

### 6.3 Improve information composition outside the net

Use the existing scalar score together with existing context metadata rather than
adding neural inputs:

- freshness: preserve source kind, age, explicit-date, version, and task-time
  sensitivity evidence in the notice;
- relevance: include candidate path/kind and whether a candidate is already kept
  when explaining a score;
- supervisor: distinguish exact duplicate, unchanged reread, broad action, and
  loop-level spin in the reason;
- compaction: retain tier and protection reason alongside the selected segment;
- alloc: expose per-side window/output evidence so the caller knows which limit
  was actually applied.

## 7. Phase 3 — deterministic policy layer around the existing models

This phase is expected to provide the largest immediate quality improvement
without changing weights. Rules must be conservative, auditable, and advisory
unless they protect a hard safety invariant.

### 7.1 Shared policy order

Every gate should follow this order:

1. Validate and bound input.
2. Apply a high-confidence mechanical rule.
3. Otherwise run the frozen model.
4. Apply risk-aware threshold/confidence policy.
5. Produce one concise evidence-carrying notice.
6. Record a privacy-safe metric and continue the data path.

### 7.2 Freshness

Add only high-confidence evidence composition:

- separate explicit dates, version strings, and relative-time claims in the
  explanation;
- distinguish old material from a timeless task;
- distinguish stale content from an author explicitly warning that it may be
  outdated;
- deduplicate equivalent freshness notices from the same source/target;
- suppress notices for too-short or clearly non-material tool output;
- include a bounded statement when the inspected text was truncated.

Never mark content unusable or prevent the LLM from seeing it.

### 7.3 Relevance

Preserve and strengthen information-preserving skips:

- exact content duplicates;
- near-verbatim duplicates of already-kept candidates;
- generated/minified/lockfile/license boilerplate when the current marker is
  unambiguous;
- identical candidate IDs or paths with identical content;
- candidates already visible in the current context.

Add a confidence distinction between `likely useful`, `likely tangential`, and
`uncertain`. Only the first two should be rendered when the notice stays within
the notice-token budget; uncertain candidates remain visible without a negative
instruction.

### 7.4 Supervisor

Retain the rule layer as authoritative for:

- exact repeated calls;
- unchanged-file rereads;
- rereads after a write to the same target;
- repeated identical calls that returned an error;
- obvious broad scans after a narrower result already answered the target.

Add trajectory-safe explanations for:

- target streaks;
- repeated failed attempts;
- no-progress loop boundaries;
- calls outside the task's lexical/structural target;
- calls that are broad only because the model has not yet received a useful
  result.

The call must still run. The message should teach a narrower recovery action, not
simply say that the call was wasteful.

### 7.5 Compaction

Keep the safety hierarchy explicit:

1. newest keep window;
2. user instructions and constraints;
3. settled decisions;
4. unresolved errors and failing evidence;
5. recent task-specific answers;
6. exact/near duplicates, boilerplate, and re-fetchable material.

The net may select only within this safe policy. The caller must never compact a
half-flagged user/assistant turn, a load-bearing decision, or the newest turn.
Free-tier candidates should be removed before any LLM summarization is requested.

### 7.6 Allocation

Keep hard limits mechanical and outside the learned score:

- learned provider errors outrank endpoint discovery;
- endpoint discovery outranks the registry;
- the registry outranks conservative defaults;
- window and output sides resolve independently;
- unknown output caps must not be invented and sent over the wire;
- explicit user values may shrink a plan but cannot exceed known hard limits;
- each limit kind retries at most once after a parsed provider error.

Expose the source of every clamp in diagnostics and make the plan self-contained:
the caller should not have to reconstruct safety bounds after receiving it.

## 8. Phase 4 — threshold and confidence policy without weight updates

This phase changes decisions around the frozen scores, not the scores themselves.

### 8.1 Threshold governance

- Keep recall-first thresholds for freshness and relevance where missed useful or
  stale evidence is more expensive than an extra notice.
- Keep precision/keep-safety thresholds for compaction.
- Keep a higher bar for supervisor nudges that can interrupt tool flow.
- Treat allocation pressure as a bounded policy signal; never allow the model to
  exceed resolved hard limits.
- Store threshold rationale next to each constant.

### 8.2 Calibration using existing evaluation data only

If current validation partitions are available, perform threshold-only sweeps:

- no gradient updates;
- no new labels;
- no test-fold tuning;
- report precision, recall, F1, false-advice rate, and notice cost;
- choose a threshold once, then lock it in a versioned policy record.

If a threshold change does not improve the risk-weighted result, keep the current
threshold.

### 8.3 Confidence bands and abstention

Use a small fixed policy around each score:

- high-confidence positive: render the advice;
- ambiguous: record the score but avoid a noisy notice;
- high-confidence negative: remain silent, without hiding the data.

The bands must be fixed constants or derived from the existing threshold and must
not introduce a learned parameter.

### 8.4 Notice budget and credibility

- At most one notice per gate class per context boundary.
- Deduplicate notices by model, source, target, and reason.
- Cap snippets and evidence text independently from the underlying data.
- Prefer one concrete action over several generic warnings.
- Never repeat the same loop warning after it has already been acknowledged in
  the current run.

## 9. Phase 5 — runtime integration and token economics

### 9.1 Call each gate at the right boundary

Verify and enforce this schedule:

- allocation: before request construction and again only after a learned limit;
- relevance: once over the recalled candidate menu before it enters the next
  generation context;
- freshness: once per material result, with per-source deduplication;
- supervisor: before a tool, after its result for accounting, and once per loop
  boundary;
- compaction: only when the usage/pressure policy says a pass can help.

### 9.2 Avoid redundant inference

- Do not score short or empty material when the model cannot make a useful
  judgment.
- Do not call relevance when there are no candidates or no tangential candidates
  to mention.
- Do not call compaction when the history is below its protected pressure band.
- Do not call loop supervision before its minimum trajectory evidence exists.
- Reuse the same task/recent/feature context within one boundary.

### 9.3 Preserve the no-ML path

For every integration point, add a paired replay assertion:

- ML disabled produces no ML notice or event;
- existing data and tool calls still flow;
- the legacy compaction cliff remains the safety net;
- a native gate error falls through without aborting the turn.

### 9.4 Improve advice quality in the prompt channel

- Keep notices in the ephemeral context channel where possible.
- Put supervisor recovery guidance in the tool result only when it is about the
  immediately completed call.
- Keep evidence and advice separated so the model can inspect the source itself.
- Mark every message as advisory and avoid wording that falsely claims a filter
  occurred.

## 10. Phase 6 — safe online-learning operations, without new training work

No online-learning algorithm or label-generation strategy is expanded here. The
existing adaptation remains off by default and output-layer-only.

The allowed operational work is:

- verify `/ml info`, `/ml learn`, `/ml adopt`, `/ml discard`, `/ml reset`, and
  `/ml wrong` state transitions;
- make persistence failures visible without exposing raw content;
- show pretrained hash, adaptation count, pending/adopted status, and reset path;
- retain trust-region, replay-anchor, batch-floor, and adoption-gate checks;
- ensure a changed baked supervisor blob invalidates an old adaptation;
- ensure a process restart never activates an unadopted candidate;
- add a clear user-facing warning that online state is optional and reversible.

Do not automatically enable adaptation, silently label ambiguous tool results, or
expand adaptation to hidden layers or the other models.

## 11. Phase 7 — performance and memory efficiency

### 11.1 Measurement

Benchmark each gate with:

- empty input;
- normal short input;
- maximum accepted input;
- maximum candidate/segment menu;
- Unicode-heavy and path-heavy input;
- repeated calls that should hit a cache.

Report p50/p95/p99 latency, allocations, peak temporary memory, and output size.

### 11.2 Safe optimizations

- reuse buffers for feature vectors where ownership permits;
- avoid repeated `String`/JSON conversions at the same boundary;
- cache immutable anchors and bounded per-call intermediates;
- score only the candidates/segments that can affect the current notice or plan;
- keep inference scalar and deterministic across platforms;
- avoid persisting raw text for caching or metrics.

### 11.3 Performance guardrails

Add regression limits for:

- per-gate inference latency;
- maximum temporary allocations;
- maximum notice length;
- maximum working-set growth;
- maximum number of ML calls per agent turn.

## 12. Phase 8 — evaluation and observability

### 12.1 Aggregate committed-weight evaluation

Extend the read-only trainer reporting surface so one command reports all five
committed models:

- architecture and blob integrity;
- fixture results;
- existing held-out metrics where already recorded;
- threshold and risk orientation;
- deterministic replay result;
- no weight writes.

The command must clearly distinguish:

- measured held-out results;
- hand-built fixture results;
- runtime replay results;
- unmeasured or unavailable claims.

### 12.2 Counterfactual and metamorphic tests

For each model, add paired tests where one fact changes and everything else stays
constant:

- freshness: old/current date, time-sensitive/timeless task;
- relevance: same candidate with on-task/off-task task, duplicate/non-duplicate;
- supervisor: same call before/after a write, healthy/spinning trajectory;
- compaction: old/new segment, protected/free-tier segment;
- alloc: known/unknown model, discovered/learned limit, config below/above cap.

These tests validate policy behavior without training a new model.

### 12.3 Privacy-safe runtime metrics

Expose counts and bounded numeric diagnostics, never raw prompt or file content:

- calls by gate and reason;
- silent/flagged/ambiguous counts;
- fallback/error counts;
- notice deduplication counts;
- compaction freeable tokens;
- allocation clamp source;
- score histograms or coarse bands;
- online state counters.

Metrics must be opt-in or local-only according to existing Sofuu behavior.

### 12.4 Fix the documentation truth gap

Update stale plan/status text so it agrees with the active implementation:

- `PLAN-ML-GATES.md` should no longer say that nothing is built if the shipped
  implementation remains active;
- separate implemented behavior from proposed future work;
- record the parallel-test isolation issue until it is fixed;
- record which evaluation commands cover which models.

## 13. Recommended implementation order

The order is chosen for impact without retraining:

1. Freeze hashes, schemas, contracts, and baseline metrics.
2. Fix allocation-policy test isolation and native-boundary robustness.
3. Add aggregate read-only evaluation and counterfactual replay coverage.
4. Improve bounded preprocessing and reuse computed feature work only where score
   margins remain compatible.
5. Strengthen deterministic supervisor, relevance, compaction, freshness, and
   allocation policy composition.
6. Tune thresholds/confidence bands using existing validation evidence only.
7. Reduce redundant runtime calls and improve notice/token economics.
8. Add performance limits and privacy-safe observability.
9. Re-run the full workspace and JS/E2E gates with ML on and off.
10. Update the status documentation and publish the before/after report.

## 14. Final acceptance checklist

- [ ] No model parameter count changed.
- [ ] No weight blob was retrained or replaced.
- [ ] No feature dimension or feature-slot meaning changed.
- [ ] No desktop source or build target was touched.
- [ ] No QTSQ source was touched.
- [ ] Default parallel tests pass without state leakage.
- [ ] Serial ML tests still pass.
- [ ] Aggregate evaluation distinguishes measured, fixture, and replay evidence.
- [ ] Malformed and oversized inputs are bounded and non-fatal.
- [ ] Scores and JSON outputs remain valid and deterministic.
- [ ] Advisor-only behavior is preserved.
- [ ] ML-off and no-ML fallback paths remain intact.
- [ ] Provider hard limits remain mechanical and cannot be exceeded.
- [ ] Compaction cannot remove protected or load-bearing material.
- [ ] Notice count and token cost are bounded.
- [ ] p50/p95 performance and memory regressions have explicit results.
- [ ] Online adaptation remains opt-in, reversible, and hash-keyed.
- [ ] Documentation matches the actual implementation.

## 15. Expected impact and honest limit

The highest-probability gains without training are expected from policy correctness,
runtime placement, duplicate/repetition suppression, hard-limit handling, safer
compaction composition, input bounding, and reduced notice/inference overhead.

These changes can make the existing model substantially more reliable and useful,
but they cannot create new semantic knowledge that the current weights and feature
semantics do not contain. New learned capability, new feature dimensions, and a
fundamental improvement in generalization remain gated on a future compatible
training/re-bake cycle.
