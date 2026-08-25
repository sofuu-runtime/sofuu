# PLAN-ML-GATES — Four Tiny On-Device Models for the Context Economy

> Status: proposed (2026-08-24). Nothing here is built yet.
> Companion to `TASKS.md`, `ROADMAP.md`, and the existing `PLAN-*.md` docs.
>
> One line: add **four separate ~8.5k-parameter offline models** plus one shared
> **in-memory context working set**, so the CLI spends tokens only on what a task
> actually needs — deciding what comes *in*, what actions get *taken*, what *stays*,
> and what gets *trusted*. Pretraining is the product; a bounded online loop adapts
> each model to the user's habits on top.

---

## 1. The problem

Sofuu's agent loop spends context greedily:

- Recalled memories, tool results, and attachments enter the prompt up to a flat cap,
  with no notion of whether a given item is worth its tokens.
- The biggest waste is **retrieval that should never have happened** — reading a
  3,000-line file, grepping too broadly, re-reading an unchanged file, repeating a call.
- Compaction is a **cliff**: at 70% of the context budget (`chat.rs:2991 COMPACT_AT`)
  the whole history is summarized into one blob in a single LLM call (`summarizeHistory(0)`),
  with no idea which turns were load-bearing.
- **The model is never told today's date.** `Date.now()` appears in `src/js/agent.js`
  and `crates/sofuu-core/src/chat.rs` only for timing, brain decay, and log timestamps —
  no date ever reaches a prompt, so the model cannot judge whether a 2021 page is stale.

All four gaps are *selection* problems. Small networks are good at exactly that.

---

## 2. Core principles (the decisions that shape everything)

These were settled in design review and are load-bearing. Do not revisit them lightly.

1. **Advisors, not filters.** The models never touch the data path. They emit guidance into
   the prompt (the same ephemeral context channel as recall/shared/watched notes) and the
   LLM decides what to do with it. Rationale: bad *advice* costs ~20 tokens and the LLM
   ignores it; bad *filtering* silently destroys the one thing the agent needed and nothing
   in the transcript reveals it. That asymmetry is the whole argument. Consequence: **no
   capping, no reordering, no deletion.** `capToolResult` (`agent.js:665`) stays as-is.

2. **Accuracy is the priority; the hard path is acceptable.** Where accuracy and ease
   conflict, choose accuracy. Concretely this drives the data strategy (§9) and the
   acceptance bar (§10).

3. **The four models stay fully separate.** Separate features, architectures, datasets,
   weights, thresholds, tests, and JS namespaces. Each can ship or be rejected alone.
   They share only the generic network math and the weight-blob format.

4. **Pretraining is the main part.** Online learning is a bounded add-on with hard rules
   (§12) so it cannot drift or hallucinate. It never carries the load.

5. **Offline, deterministic, zero-API.** The models are TF-IDF-feature + small-MLP, run in
   Rust, need no network, no key, no model files at runtime beyond baked weights. One
   forward pass is ~8.5k multiply-adds (sub-millisecond) against LLM round-trips of
   500ms–seconds. They run while the process is already blocked on the network.

6. **RAM is a structured working set, not a speed hack** (§8). Disk `.qtsq` stays the
   source of truth; RAM is the hot layer the models reason over.

---

## 3. Architecture overview — four gates over the context lifecycle

| Model | Question it answers | Lifecycle stage | Reads |
|---|---|---|---|
| **relevance** | is this content worth the tokens? | what comes **in** | task + candidate menu |
| **supervisor** | is this action right, right now? | what actions get **taken** | task + proposed call + trajectory |
| **compaction** | when and what to compact? | what **stays** | structured history segments |
| **freshness** | is this material current? | what gets **trusted** | collected material + task |

All four:

- are the same generic network (§4) with different weights,
- **advise, never block** (Principle 1),
- read from the one shared **in-memory context working set** (§8),
- are **pretrained** first (§9) and optionally **online-adapted** later (§12),
- expose a `sofuu.ml.<model>.*` JS surface (§13).

Plus two non-model pieces that ship first:

- the **date fix** (§7) — zero parameters,
- the **structured RAM working set** (§8) — the substrate all four need.

---

## 4. The shared network

```
in(F) → h1 tanh → h2 tanh → out(1) sigmoid        scalar f32, no SIMD
```

- Scalar-only math so scores are **bit-identical on every platform** — tests assert exact
  values, and the trainer and runtime agree by construction.
- Forward pass ~8.5k multiply-adds → microseconds.
- Weights ship as `include_bytes!` blobs with a header (magic, version, layer dims, param
  count, CRC32) then f32 LE. A cargo test asserts the header matches the compiled-in
  architecture and refuses to load on mismatch — a stale/truncated blob is a build failure,
  not a silent wrong answer.

**Why this size.** At 5k–10k params you cannot afford 768-dim embedding input — a single
dense layer over embeddings would consume the whole budget and leave nothing to learn. So
the input is a **compact computed feature vector** (each feature costs zero parameters,
reusing `sofuu.ai.embedLocal` / `similarity` / BM25 / lexical scans), and the budget is
spent on depth. Capacity is *not* the binding constraint on accuracy — data quality is
(§9). Both sit in the upper half of the 5k–10k band because accuracy is the priority and
the extra capacity is free against the 5MB size cap (~67KB of weights on a 2.1MB binary).

**Train/serve skew is structurally impossible.** Feature extractors live in
`crates/sofuu-core/src/ml/` and the trainer crate depends on them, so training and
inference run the *same* code. Feature skew is the classic silent killer for models this
small; sharing the code eliminates it.

---

## 5. Model 1 — freshness

**Answers:** *should the agent verify this material is current before relying on it?*
The easiest of the four — dates and version strings are extractable, staleness vocabulary
is real signal, and the answer doesn't depend on who's reading.

- **Arch:** `28 → 112 → 48 → 1` = **8,721 params**.
- **Features (28):** cosines to stale / fresh / changelog / hedging anchor vocabularies;
  cosine of the *task* to a time-sensitivity anchor set; newest & oldest year found, year
  span, years-behind-now, distinct-year count, has-explicit-date, has-version-string,
  has-future-year; relative-time words ("recently", "currently"); URL shape
  (docs/news/blog/year-in-path); source kind one-hots (web/memory/tool/file); brain
  `strength` and record age; content & task length; digit density; sim(content, task).
- **What the parameters earn:** if an explicit date fully decided it, a rule would do. So
  the dataset is majority **implicit** staleness — no date present, the signal is
  deprecation vocabulary, old version numbers, archived-project language — crossed against
  time-sensitive vs timeless tasks. "What is a B-tree" from a 2019 source is fine;
  "latest QuickJS release" from the same source is not. That interaction + the implicit
  cases are the learnable content.
- **Threshold tuned for recall, not accuracy:** a spurious "verify this" costs ~25 tokens;
  a missed one costs correctness.
- **Output use:** one notice per turn in the ephemeral context message, carrying the
  evidence ("the material above references 2019, 2021; current year 2026"), **not** a
  marker appended to every tool result.

---

## 6. Model 2 — relevance

**Answers:** *within a fixed budget, which candidates deserve the space?* The hardest of
the four — there is no ground truth anywhere; whether a block "helped" depends on the model
consuming it, and a false drop is invisible. Hence it ships last and gets the most care.

- **Arch:** `36 → 104 → 44 → 1` = **8,513 params**.
- **Features (36):** sim(content, task); **BM25 over the candidate set** (the strong
  classical baseline for exactly this retrieval task, zero params, and the biggest accuracy
  win available given the bundled embedder is only hashed char-trigram TF-IDF); task-term
  overlap ratio; IDF-weighted overlap; verbatim 3-gram hit; rank and set size; length and
  length ratio; kind one-hots; brain `strength` and role; max-sim to already-kept
  (near-duplicate detection); redundancy vs candidate centroid; path-like /
  extension-matches-task / code-symbol density; uppercase, punctuation and digit density;
  distinct-token ratio; boilerplate-anchor cosine (license headers, generated files,
  `node_modules`); `alreadyVisible` in recent history (reusing `agent.js:629`).
- **Role is pre-retrieval advisor (Principle 1).** It reads the task + the menu of
  available things (files, memories, search results) and emits guidance — "likely needed:
  X; probably not needed: Y; prefer grep over a full read" — rendered by the CLI into the
  ephemeral context message. The LLM decides. Tokens are saved *before* they're spent.
- **An 8.5k net cannot write those sentences.** It scores a small fixed action space —
  necessity per candidate, a scalar for how much work the task warrants, a small
  classification over read strategies (full read / grep / targeted line window). The model
  supplies the judgment; the CLI supplies the phrasing. This keeps output auditable.

---

## 7. The date fix (ships first, zero parameters)

Add `Current date: 2026-08-24.` to the per-turn ephemeral context message
(`agent.js:787` `ctxParts`), keeping the system prompt byte-stable for prefix caching
(P1/P6). ~8 tokens/turn. This is the actual root of the freshness problem and does more for
"is this info current" than any model — the freshness model builds on it, not substitutes
for it. When the freshness gate fires, it appends concrete evidence (see §5), not a vague nag.

---

## 8. The shared in-memory context working set (the "RAM" piece)

**Honest framing:** this is built because intelligent compaction and supervision *require* a
structured representation of the context that does not exist yet — **not** because disk is
slow. Within a running chat the history is already RAM-resident (the driver's `history`
array); the `.qtsq` session files (`session.rs`) and brain are small, sub-millisecond,
OS-page-cached reads, and the dominant latency is the LLM call. Fast revival comes along as
a side benefit, not the motive.

**What it is:** a structured in-memory object the models reason over — history as **segments
with metadata** (type: instruction/answer/tool-call/tool-result, token size, age, what it
references, whether already compacted), plus the supervisor's trajectory (calls made, files
read, results usefulness) and recently-recalled memories. Today the history is a flat array
of `{role, content}` with none of that.

**The one rule that keeps it safe:** **RAM is the hot layer, disk stays the source of
truth.** The `.qtsq` files remain the durable store; the RAM working set is rebuilt from
them on resume. RAM is volatile — on a crash you lose only what wasn't persisted. Durability
is unchanged.

---

## 9. Training — data is where the work goes

**Rejected: git-history mining.** Training on "which lines did this commit change" fails two
ways. It's repo-specific (teaches sofuu's vocabulary, not transferable judgment), and the
label is wrong in the dangerous direction — a commit doesn't touch the type definition, the
caller, or the test you'd want in context, so those land in the negatives, teaching the model
to drop useful context. That is noise aimed straight at the worst failure mode.

**What replaces it — the user's actual ask is knowledge of *what to use and what not to*,
which is a set of transferable decision rules, not corpus statistics.** So the dataset is
organized around the decisions, each example's label following from its construction rather
than case-by-case opinion:

- the definition site of the symbol the task names → use
- a mention of that symbol in an unrelated call site carrying no task-relevant info → don't
- license headers, generated files, lockfiles, minified blobs → never
- a near-duplicate of something already kept, or text already visible recently → don't re-send
- high lexical overlap, wrong topic (this codebase's "token"/"context"/"memory" trap) → don't
- low lexical overlap but it actually contains the answer, phrased in synonyms → use
  (the class that punishes a pure-BM25 gate)
- a changelog entry → use when the task is about versions, not otherwise
- a test for the function in question → use for behaviour questions, not typo fixes

The last group is where parameters earn their keep: the label flips based on the *task*, not
the candidate — no threshold on any single feature can express it.

**Two constructions make labels mechanical, not judgmental:**
- **Needle-in-haystack** for the semantic class — plant a known answer phrased in synonyms so
  lexical features provably fail and only semantic ones fire.
- **Rule-based augmentation** — each variant's label follows deterministically from the
  transformation: paraphrase the task, pad the block 10×, inject boilerplate, duplicate it.
  Multiplies clean data without noise and directly teaches invariances (length must not flip
  relevance; duplication must).

**On volume:** with ~36 features and ~8.5k params, sample complexity is low and label noise
dominates the error. A few thousand clean, discriminative examples beat fifty thousand noisy
rows for a model this size. That is the technical argument for the curated path, not a fallback.

**Generalization is proven, not assumed:** hold out entire decision classes and entire source
domains — train on code and docs, test on web-result-shaped and memory-shaped candidates.
Git-history data would have failed this test, which is the tell it was wrong. An uncommitted
realism check against actual local projects runs as a generalization *measurement* only,
never as training data, so the committed trainer stays reproducible.

**Trainer:** new dev-only crate `crates/ml-train/` — plain Adam + early stopping on a ~8.5k
net, grouped k-fold CV, seed sweep, calibration report. Added to workspace `members` but
**excluded from `default-members`**, so `make` / `cargo build --release` never builds it and
the shipped binary is untouched. Reproduce with `cargo run -p ml-train --release -- <model>`.

**Honest limit, documented:** curated labels encode careful judgment, not ground truth from
live sessions. Retraining on real labels is a follow-up; the feature layout and blob format
are versioned so new weights drop in without an architecture change.

---

## 10. The accuracy bar — what "good enough to ship" means

Hard asserts in `cargo test`, so a weak retrain fails the build:

1. Each model must beat, on the held-out test fold, **(a)** the production heuristic it
   replaces (the 0.30 cosine floor for relevance; the 70% cliff for compaction; nothing for
   freshness/supervisor) and **(b)** logistic regression on the identical features. If it
   cannot beat plain logistic regression, the honest outcome is to ship the logistic
   regression, and the plan says so rather than shipping a net for the story.
2. Grouped k-fold CV with per-class precision and recall reported — not a single split, not
   bare accuracy.
3. Thresholds selected on a validation fold, then measured once on the test fold. No
   threshold tuning against the test fold.
4. Feature-group ablations — prove each group earns its place and catch the net collapsing
   into "just the cosine feature".
5. Seed sweep, best-on-validation kept; reliability of output probabilities reported.
6. Behavioural gates per model (e.g. freshness recall-on-stale ≥ 90%, deliberately
   asymmetric because a missed stale costs correctness while a spurious one costs ~25 tokens).
7. Bit-exact determinism, in-process and across processes.

The trainer prints this as a report (`cargo run -p ml-train -- eval`) and the numbers go into
`TASKS.md` as measured figures, not adjectives.

**The end metric that actually matters:** tokens spent per completed task, measured on a fixed
task set against today's behaviour as the baseline. If the advice doesn't move that number, it
isn't earning its place — better to find that out on a benchmark than ship it on a story.

---

## 11. Model 3 — supervisor (real-time)

**Answers:** *is this action the right one right now?* Runs **at every action**, not every
streamed token — scoring per token would be thousands of passes per turn for no added signal.
The decision points are where actions happen.

- **Arch:** ~32 trajectory/action features, target shape `32 → 104 → 44 → 1` ≈ **8,097 params**
  (feature set finalized in its phase).
- **Checkpoints** (all in `src/js/agent.js`):
  - **Before a tool runs** — `execOneTool` (`agent.js:866`), after `name`/`args` are known but
    before `fn(args)`. The valuable one: the only place waste can be *prevented*, not observed.
  - **After a result returns** — `agent.js:888`, where the result is already capped. Judges
    whether the call paid off — a nudge opportunity *and* the cleanest online-learning label.
  - **At each loop boundary** — before the next model call (~`agent.js:934`). Trajectory-level:
    step count vs predicted budget, progress being made or not, drift from the original task.
- **Features available at call time, all cheap:** similarity of call args to the task; whether
  this call is an exact/near-duplicate of an earlier one; whether this file was already read this
  turn; calls-so-far vs the pre-turn budget; whether previous calls of this tool returned anything
  useful; how broad the call is (`grep .`, `list_dir` on root, `read_file` on a 3,000-line file);
  whether the target was in the "probably not needed" set from the pre-turn relevance advice.
- **Two of those are rule-detectable with no model at all** — exact repeat calls and re-reading an
  unchanged file are the most common ways coding agents burn tokens. **That subset ships
  immediately**, before any training.
- **Correction channel:** advise, never block (Principle 1) — refusing a tool call would re-enter
  the silent-failure hole at another layer. The natural channel is the tool result itself (already
  flows back into the message array), in-band, ~15 tokens:
  `[supervisor: you read this file at step 2 and it hasn't changed]`. Trajectory nudges go in at
  the loop boundary via the ephemeral channel. If the LLM ignores a nudge and turns out right,
  that's an environment-generated negative label the online loop picks up automatically.

---

## 12. Model 4 — compaction

**Answers:** *when and what to compact?* Replaces the current cliff (`chat.rs:2985-3090`).

- **Arch:** ~32 segment features, target shape `32 → 104 → 44 → 1` ≈ **8,097 params**
  (feature set finalized in its phase).
- **Features (~32):** segment age; type one-hots (instruction/answer/tool-call/tool-result);
  token size; **is-it-referenced-by-recent-turns** (the load-bearing signal); on-task
  similarity; redundancy with the already-compacted summary; still-retrievable-or-not (a
  file-read can be re-read cheaply, a unique derivation can't); verbose-boilerplate-ness;
  within-keep-recent-window.
- **When:** not a fixed 70% trigger, but "is there enough low-value material accumulated to be
  worth a pass." Compact opportunistically between turns, not as a blocking emergency.
- **What:** keep verbatim what's load-bearing (recent turns, decisions, active task, file paths
  and symbols the agent will need again); compact what's disposable (old verbose tool output,
  superseded reasoning, dead ends).
- **Slowly — the key change from today.** Each pass frees a *small* budget (~10–15% of the
  window), compacts only the highest-priority segments that fit, then stops and re-scores.
  Context degrades gracefully instead of cliff-dropping to a single summary.
- **Honest constraint:** an 8k net can *select* but not *write prose*. The model is the policy
  (what/when/how much); summarization still uses the LLM, but only for flagged segments and
  incrementally, instead of one call over everything. There's a **free tier below that** needing
  no LLM — truncating already-capped verbose tool output, deduplicating repeats, dropping pure
  boilerplate. Prefer the free tier first; spend an LLM call only where real summarization is
  needed. This makes compaction *cheaper* than today, not just smarter.
- **Keep the drop-oldest guard** (`chat.rs:3082`) as the final safety net.

---

## 13. Online learning — bounded by construction, off by default

Pretraining is the product; this only nudges the decision boundary toward the user's habits.
The user's loop — *watch its actions, compute the loss from what actually happened, apply it,
repeat* — is legitimate **because the label comes from the environment, not from the model's own
output.** That distinction is the whole design.

**What is forbidden:** self-training — the model says "irrelevant", that prediction is treated as
ground truth, and it retrains on its own opinion. That drifts. It stays off-limits.

**What is allowed — learn from outcomes.** This architecture generates unusually clean signals,
and credit assignment (which usually breaks these loops) doesn't here because every prediction is
per-item and each item's outcome is separately observable in the trace:

- relevance said a file probably wasn't needed and the agent read it anyway then cited it → direct
  negative on that item;
- relevance flagged five candidates likely-needed and the trace shows two ever used → per-item
  labels for all five;
- supervisor said three tool calls and the turn took nine → an unambiguous regression target;
- the agent re-called a tool with narrower args right after following advice → the advice was too tight.

**Seven guardrails, each closing a specific failure mode:**

1. **Frozen backbone.** Only the output layer adapts — 45–49 params per model. An update can move
   the boundary; it cannot corrupt learned features.
2. **Trust region.** After every update, weights are projected back into `‖w − w₀‖ ≤ 0.1·‖w₀‖`
   around the pretrained values. Bounded by construction, not by hoping the learning rate is small.
3. **Never learns from its own output.** Labels only from verifiable signals (above) or an explicit
   `/ml wrong`. No self-training, so no feedback loop into its own errors.
4. **Minimum evidence, batched.** ≥16 labelled examples buffered before any update; never a
   per-event step.
5. **Replay anchors.** Every online batch mixes in held-out pretraining examples, so an update
   can't quietly forget the pretrained task.
6. **Adoption gate.** A computed update is evaluated in-process against the committed held-out
   fixtures using **the same acceptance gates as CI** (§10); adopted only if accuracy and the
   behavioural gates hold. Otherwise discarded and logged. This is the concrete "cannot get worse"
   guarantee.
7. **Separate, reversible state.** Deltas live in `~/.sofuu/ml/<model>_online.f32`, keyed to the
   pretrained blob's hash so a Sofuu upgrade invalidates stale deltas. Shipped weights are never
   modified. `/ml reset` restores pretrained instantly. Off until `/ml learn on`; every adoption and
   rejection is logged for audit.

---

## 14. Runtime surface — `sofuu.ml`

```js
sofuu.ml.freshness.score(text, task, opts?)  // {"score":..,"stale":bool,"years":[..],"reason":".."}
sofuu.ml.relevance.score(task, text, opts?)  // {"score":..}
sofuu.ml.relevance.rank(task, items, opts?)  // {"order":[..],"scores":[..]}  batch, bounded
sofuu.ml.supervisor.check(action, ctx)       // {"ok":bool,"nudge":".."|null}
sofuu.ml.compaction.plan(segments, budget)   // {"compact":[ids..],"keep":[ids..],"freeable":tok}
sofuu.ml.info()                              // per-model params, arch, weights hash, calls, online state
sofuu.ml.feedback(model, verdict)            // explicit label from /ml wrong
```

Registered as a nested object per model under `sofuu.ml`, following `mod_ai_register`
(`rt/ai.rs:3451-3478`), wired in `engine_register_builtins` (`rt/engine.rs:227`).
`rt/ai.rs:789` `sofuu_tfidf_embed` becomes `pub(crate)`.

---

## 15. File layout

```
crates/sofuu-core/src/ml/
  net.rs                    generic TinyMlp: forward, blob load/verify, trust-region clamp
  mod.rs                    registration of sofuu.ml.*
  context.rs                the shared in-memory context working set (§8)
  freshness/
    features.rs             28 features + anchor vocabularies
    model.rs                arch, thresholds, weights include_bytes!, JS shims
    weights_v1.f32          28 -> 112 -> 48 -> 1   =  8,721 params  (~34 KiB)
    eval.rs                 held-out fixtures + acceptance gates
  relevance/
    features.rs             36 features + BM25 + anchor vocabularies
    model.rs                arch, thresholds, weights include_bytes!, JS shims
    weights_v1.f32          36 -> 104 -> 44 -> 1   =  8,513 params  (~33 KiB)
    eval.rs                 held-out fixtures + acceptance gates
  supervisor/
    features.rs             ~32 trajectory/action features
    model.rs                arch, thresholds, weights, JS shims + rule-based fast path
    weights_v1.f32          32 -> 104 -> 44 -> 1   ≈  8,097 params
    eval.rs
  compaction/
    features.rs             ~32 segment features
    model.rs                arch, thresholds, weights, JS shims + free-tier mechanical compaction
    weights_v1.f32          32 -> 104 -> 44 -> 1   ≈  8,097 params
    eval.rs
  online.rs                 bounded output-layer adaptation (shared mechanism, per-model state)

crates/ml-train/            dev-only crate, excluded from default-members
  src/main.rs               cargo run -p ml-train -- freshness|relevance|supervisor|compaction|eval|ablate
  src/data_freshness.rs
  src/data_relevance.rs
  src/data_supervisor.rs
  src/data_compaction.rs
  src/train.rs              Adam, grouped k-fold CV, seed sweep, calibration report
```

---

## 16. Wiring — how each integrates

- **Date fix:** `agent.js:787` `ctxParts` — `Current date: …` (+ freshness evidence when the
  gate fires). System prompt stays byte-stable (P1/P6).
- **Freshness:** one notice per turn in the ephemeral context message, evidence-carrying.
- **Relevance:** pre-retrieval guidance rendered into the ephemeral context message from the
  model's scores. No data-path changes.
- **Supervisor:** three checkpoints in `agent.js` (§11); nudges ride the tool result or the
  loop-boundary ephemeral message. Rule-based repeat/re-read detection ships first.
- **Compaction:** replaces the `COMPACT_AT` cliff with model-driven incremental passes (§12);
  drop-oldest guard kept as the final net.
- **Chat config:** `ChatConfig.ml: bool` (default on) persisted like `brain` (`chat.rs:276/336`);
  `/ml [on|off]`, `/ml learn [on|off]`, `/ml reset`, `/ml wrong`, `/ml info` mirroring the
  `/brain` arm (`chat.rs:749`); `mlgate` branch in `onStep` (`chat.rs:3317`) rendering dim
  **ASCII-only** lines (TUI rule). `SOFUU_NO_ML=1` and `ml:'off'` disable everything; every call
  is in try/catch so a gate failure can never break a turn.

---

## 17. Build order (phased — each delivers value, hardest gets the most care)

1. **Date fix + structured RAM working set.** Zero/low model risk; the substrate everything
   else needs. Ships value immediately.
2. **Freshness.** Easiest model; proves out the whole train→ship→wire pipeline end to end on
   the simplest case.
3. **Compaction.** Builds directly on the working set from step 1; replaces the current cliff;
   labels are mechanical.
4. **Relevance.** The hard one; gets the most data and evaluation care; ships last when the
   pipeline is proven.
5. **Supervisor + online learning.** Last, because its training labels come from traces the
   earlier models help produce. (The rule-based repeat/re-read subset of the supervisor ships
   in step 1 regardless.)

Each model can ship or be rejected on its own. The date fix ships regardless of everything.

---

## 18. Verification

- `cargo test --workspace` — includes the per-model acceptance gates (§10) and the weight-blob
  header/architecture match.
- `cargo run -p ml-train -- eval` — reproduces the committed weights bit-for-bit and prints the
  full CV/ablation/calibration report.
- `./sofuu run tests/agent_test.js` — E2E battery: `sofuu.ml` surface present, params in band,
  determinism, `Current date:` in the wire request, freshness notice on old-dated material,
  `SOFUU_NO_ML=1` reproduces current behaviour exactly.
- `make test` · `make size-check` (weights are baked and small; binary stays ≤ 5MB).
- pty smoke of chat confirming the dim `⏺ ml ·` lines render with panel alignment intact (ASCII).
- **Tokens-per-completed-task** on a fixed task set vs today's baseline (§10 end metric).

---

## 19. Honest constraints & explicitly out of scope

**Constraints (documented, not hidden):**
- Small nets **score/classify; they do not write prose.** Compaction summarization and the
  phrasing of relevance/supervisor guidance still use the LLM/CLI. The models supply judgment;
  the CLI supplies words.
- Curated/synthetic labels encode careful judgment, **not ground truth from live sessions.**
  Online learning on real labels is the path to closing that gap.
- The RAM working set is a **structural need and a cache**, not a read-speed win — disk reads of
  these small files were already fast and OS-cached; the LLM call dominates latency.

**Out of scope (documented follow-ups):**
- Retraining any model on real user labels (the online loop's adoption gate is the on-ramp).
- Persisted per-user weight overrides beyond the online-delta mechanism.
- Letting a model *block* a tool call or *delete* content — deliberately excluded by Principle 1.
- Any remote/hosted inference for these gates (they are offline by design; `provider:"local"`
  llama.cpp work in `ROADMAP.md` Track B is a separate arc).

---

## 20. Docs to update on delivery

- `README.md` — one `sofuu.ml` section documenting the four models separately, the `/ml`
  commands, and honest framing (param counts, measured held-out numbers, what the labels are and
  are not).
- `TASKS.md` — a new section with the measured report (accuracy, ablations, tokens-per-task),
  plus the online-learning rules as a stated contract.
