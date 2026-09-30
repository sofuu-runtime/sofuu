# PLAN — Memory Management & Token Efficiency (the "use tokens correctly" batch)

> **✅ STATUS (2026-08-21): P1–P7 ALL LANDED AND VERIFIED, then AUDITED and
> the six audit findings fixed.** `cargo test` 185 green (5 new units:
> `retain_drops_dead_keeps_pinned_and_entities`,
> `system_prompt_contract_survives_compression`,
> `anthropic_prefix_cache_markers_and_system_hoist`,
> `anthropic_merges_consecutive_same_role_turns`,
> `prefix_is_byte_identical_across_turns`), `agent_test.js` ALL PASSED
> (17 new checks), `rlm_mock_test.js` green, `make test` 19/19,
> `make size-check` 2.0MB ≤ 5MB, chat smoke clean.
>
> **Audit fixes (2026-08-21, post-landing):**
> 1. **Anthropic consecutive-role merge** — `build_anthropic_body_v2` now
>    merges adjacent plain same-role messages (ephemeral ctx + task on an
>    empty history, compaction summary + user turn) into one turn so the
>    wire is valid (the Messages API rejects non-alternating roles).
>    Unit-tested.
> 2. **`def.system` composes AFTER CORE_PROMPT** — a custom system no longer
>    replaces the core's discipline lines; passing CORE_PROMPT itself (the
>    chat def) is a no-op, not a duplication. Battery-tested.
> 3. **`/cost` surfaces cache hit tokens** — agent.js usage carries
>    `cacheReadTokens`/`cacheWriteTokens`; `__chat_report_usage` accepts
>    them; `/cost` shows per-turn cache hits + a session total when non-zero.
> 4. **M2 fan-out GC** — new global `__sofuu_gc` bridge (engine.rs); agent.js
>    calls it after `runMany`/`mapContext` batches.
> 5. **M4 trace stubs** — evicted middle events are demoted to `{kind,t}`
>    stubs (timeline preserved for `renderTrace`) instead of dropped; heavy
>    payloads bounded to 512. Battery-tested.
> 6. **Verification gaps closed** — P5 auto-compaction E2E
>    (`tests/chat_compact_e2e.sh`: fires at 70%, summarizer reaches the mock,
>    one-shot guard holds), P6 prefix-identical Rust test, and
>    `recall_min`/`recall_budget` config wiring (config.json → chat def).
>
> **STATUS (2026-08-21, earlier): NOT STARTED.** Plan written after a full audit of the
> current token/memory flow (chat driver, agent loop, RLM episode, CMA, HTTP
> client, TUI, session mesh).
>
> **Thesis:** token efficiency is NOT about limiting tokens — it is about
> *spending them on the right things*. Every token in the window should be
> either (a) the user's actual question, (b) evidence the model asked for, or
> (c) genuinely relevant memory. Today a meaningful share of every turn is
> spent on: un-gated memory dumps, untruncated tool output, re-paid static
> prefixes, and raw noise written back into the brain. Fix the allocation,
> don't cap the budget.
>
> **Second thesis:** the CLI must use *as much RAM as it needs and no more* —
> no leaks, no monotonic growth in long sessions, no unbounded buffers. The
> audit found six real issues (M1–M6 below); all are fixable without new FFI.
>
> **Provider stance:** Sofuu is provider-neutral — OpenAI-compatible,
> Anthropic, Gemini, OpenRouter, Ollama/local, custom endpoints are all equal
> citizens. Nothing in this plan is Anthropic-specific. The caching work (P6)
> is about **prefix stability**, which pays off on every backend:
> - OpenAI/OpenRouter/Ollama: automatic prefix caching picks up a stable prefix
>   with zero API changes.
> - Anthropic: needs explicit `cache_control` markers — one small opt-in code
>   path, not a focus.
> - Local (future llama.cpp/QTSQ, Track B): a stable prefix is exactly what
>   makes KV-cache reuse across turns possible in the fusion loop.
>
> **Scope:** chat + agent loop + RLM (all three prompt surfaces), the CMA
> memory core, and process-level RAM hygiene. Explicitly out of scope: the
> QTSQ codec, local inference (Track B/C), RLM sandbox internals beyond the
> driver prompt.

---

## 0. Audit findings (verified against the tree, 2026-08-21)

### Token waste

| # | Where | Waste |
|---|---|---|
| T1 | `src/js/agent.js:535-543` | Recall injects top-15 hits × 300 chars **unconditionally** — no similarity threshold, no token budget, no dedupe against what's already in context |
| T2 | `src/js/agent.js:534` + `:777` | Task embedded **twice per turn** — once for recall, again at store time |
| T3 | `src/js/agent.js:286-297` | Stores **raw** task + raw answer (4k chars each) into the brain — noise accumulates, recall quality degrades over time |
| T4 | `src/js/agent.js:735` | Tool results enter context **untruncated** — a `web_open` page (~20k chars ≈ 5k tokens) lands raw in the loop messages |
| T5 | `crates/sofuu-core/src/chat.rs:2973` | History trim **drops** oldest turns (pure info loss); `/compact` is manual-only |
| T6 | `crates/sofuu-core/src/rt/ai.rs:425-433` | No prompt-cache markers anywhere; worse, the chat system prompt embeds the dynamic `lastShared` mesh text (`chat.rs:3172`), so even the static prefix is byte-unstable turn-to-turn |
| T7 | `crates/sofuu-core/src/rlm/episode.rs:56-80` | RLM driver prompt is a protocol contract (must stay) but ~40% of it is prose that can be merged into the API lines |

### Memory / RAM issues

| # | Where | Issue |
|---|---|---|
| M1 | `crates/sofuu-core/src/chat.rs:2719,3858` | `watchChangesPending` pushed every poll tick, **never drained** → unbounded growth in long `/watch` sessions |
| M2 | `crates/sofuu-core/src/rt/engine.rs:736` | QuickJS full GC (`JS_RunGC`) only runs at process exit — long chat sessions accumulate JS garbage up against the 512MB `JS_SetMemoryLimit` (`engine.rs:187`) |
| M3 | `crates/sofuu-core/src/memory/cma.rs:355` + `rt/memory.rs:864` | `decayTick` is exposed but **never called** by any caller — weak memories never decay; `forget()` marks records but never removes them from the Vec, so brain memory grows monotonically |
| M4 | `src/js/agent.js:485-493` | Agent `trace[]` unbounded per run (RLM caps at 512 events, agents don't) |
| M5 | `src/js/agent.js:60` | `RUN_COUNTS` claims "last 24h" but never resets |
| M6 | `crates/sofuu-core/src/chat.rs:2041` | No RSS guardrail — the footer shows RAM but nothing acts on runaway growth |

### Audited and already fine (no action)

- TUI scrollback: `MAX_LINES = 2048` cap (`rt/tui.rs:26`).
- Chat history: token-budgeted trim (`chat.rs:2973`) + 2000-entry safety net.
- HTTP bodies: 256MB hard caps (`rt/http_client.rs:30-31`); `ResponseData` freed via JS finalizer with live-pointer validation.
- Session events: `EVENT_CAP = 300` (`session.rs:28`).
- MCP pool: idle-disconnect when no runs active (`agent.js:327`).
- `BRAINS` map: one handle per brain file, by design (bounded by distinct paths).
- `ACTIVE`/`SIGNALS` run tables: cleaned on every finish path including throws (`agent.js:517,787`).

---

## P1 — Minimal system prompt (all surfaces, model-agnostic)

**Design principle:** one byte-stable static core (~40 tokens) + dynamic
blocks appended *after* it. No persona fluff, no "you are a helpful
assistant" padding, no capability lectures — modern models (GPT, Claude,
Gemini, Qwen, Llama, small local models) all know how to be assistants; the
prompt only carries what they can't infer: identity, tool discipline, and
output style. Works identically on every wire format because it is just a
short string.

### P1.1 — The core (shared constant)

Lives once in `src/js/agent.js`, exported as `sofuu.agent.CORE_PROMPT`:

```
Sofuu coding agent. Be direct and concise. Verify with tools before asserting. If unsure, say so.
```

~25 tokens. Rationale per clause:
- `Sofuu coding agent.` — identity, 3 tokens.
- `Be direct and concise.` — output style; the single highest-leverage
  instruction for token efficiency (shorter answers = fewer output tokens =
  less cost AND less history bloat next turn).
- `Verify with tools before asserting.` — tool discipline; only meaningful
  when tools exist, but harmless without (small models ignore it cleanly).
- `If unsure, say so.` — anti-hallucination in 4 tokens.

### P1.2 — Composition rules

Assembly order in `agent.js` `run()` (replaces `agent.js:547-555`):

```
system  = CORE_PROMPT
        + (def.system !== default ? '\n' + def.system : '')   // user override
        + identityLine                                         // existing
context = recallBlock (+ lastShared mesh text)                 // NEW: separate user-role message
messages = [system] + [context (if non-empty)] + history + [user task]
```

- The **system message is byte-stable** for a given agent def — nothing
  dynamic ever enters it. This is what makes P6 caching work.
- Recall + mesh move to a **user-role context message** right before the
  task (or merged into the task message as a `--- memory ---` block —
  decided at impl time based on small-model behavior; both keep the system
  stable).
- `chat.rs:3172` chat def: `system: sofuu.agent.CORE_PROMPT` (the driver
  already runs through `sofuu.agent.run`, so it inherits everything).
  `lastShared` stops being concatenated into the system string
  (`chat.rs:3172`) — the driver passes it via a new `opts.shared` field
  that `agent.js` folds into the context message.
- Default agent fallback (`agent.js:133`, "You are a helpful agent named X")
  → just `CORE_PROMPT`; the name/identity line already covers identity.

### P1.3 — RLM driver prompt compression (`rlm/episode.rs:56-80`)

The RLM prompt is a **protocol contract**, not persona — every semantic rule
must survive, but the prose can be merged. Target ~150 tokens from ~250
(~40% cut). Shape:

```
Sofuu RLM loop: the full context lives in a sandboxed JS env, not this prompt.
Reply with exactly ONE ```js block per turn; its last expression's value (or
thrown error) returns as the next user message. No prose outside the block.

API: len() count() chunk(i[,a,b]) peek(a,b) grep(re,maxHits)→[{chunk,index,
line,text}] lines(i,f,t) llm(p) llmBatch([p]) emit(tag,text) final(answer)

Rules: work from data (grep/chunk/peek), never guess; keep snippets small,
never print the whole context; llm(p) for sub-questions over large slices;
final(answer) as soon as you can answer.
```

`tool_prompt()` (`episode.rs:84-110`) unchanged — it is already compact and
data-driven.

### P1.4 — Verify

- Unit test: chat def system string == `CORE_PROMPT` (no `lastShared`).
- Unit test: `buildSystem(def)` output for (a) bare def, (b) def with custom
  system, (c) def with identity — all start with the core, deterministic.
- Unit test: RLM `SYSTEM_PROMPT` token estimate ≤ 170 (chars/4 ≤ 680) and
  still contains every API name (`len`, `chunk`, `peek`, `grep`, `lines`,
  `llm`, `llmBatch`, `emit`, `final`) and the one-block rule.
- `examples/rlm_mock_test.js` stays green (the mock provider doesn't parse
  the prompt, but the episode state machine must not regress).

---

## P2 — Recall gating (`src/js/agent.js`)

Recall today (`agent.js:531-545`): embed task → top-15 → inject all of them
clipped to 300 chars each. Worst case ≈ 4.5k chars ≈ 1.1k tokens of memory
noise per turn, even when nothing is relevant.

### Changes (in `run()`, recall block)

1. **Similarity threshold:** drop hits with `score < RECALL_MIN` (default
   **0.30** — calibrated for the TF-IDF `embedLocal` cosine scale where
   unrelated text typically scores < 0.2 and paraphrases > 0.4; remote
   embedders behave similarly on cosine). Configurable per-def:
   `def.recallMin` (normalized in `normalizeDef`).
2. **Token budget:** after thresholding, fill highest-score-first until the
   block reaches `RECALL_BUDGET_TOK` (default **1024 tokens**, measured with
   the existing `estTok`). Configurable: `def.recallBudget`.
3. **Dedupe against context:** skip a hit if its text (first 120 chars)
   already appears in the last 4 history messages or in the task itself —
   re-injecting what the model can already see is pure waste.
4. **Zero-hit = zero bytes:** no header, no block, no empty-line padding
   when nothing survives gating.
5. **Observability:** the existing `recall` emit gains `dropped` (count
   below threshold) and `budgetCut` (count cut by the token budget) so
   `/why` and traces stay honest.

### Verify

- Unit tests (JS battery via `examples/agent_test.js` pattern with the mock
  provider): seeded brain with 3 relevant + 10 irrelevant memories →
  injected block contains only relevant, ≤ budget; empty brain → system
  message has no recall header.
- `/why` still lists the hits that were actually injected.

---

## P3 — Tool result truncation (`src/js/agent.js`)

Tool results are the single largest uncontrolled token source: `web_open`
returns up to ~20k chars, MCP servers can return anything. Today the raw
string goes straight into `loopMsgs` (`agent.js:735`) and is re-sent on
every subsequent loop round AND stored in history.

### Changes (in `execOneTool`, before the result enters any message)

1. **Cap:** `TOOL_RESULT_CAP` default **4000 chars (~1000 tokens)**.
   Per-def override: `def.budget.maxToolResultChars`.
2. **Head+tail preservation** (middle is usually boilerplate): keep first
   2800 + last 800 chars; replace the middle with
   `\n…[truncated N chars — re-call the tool with narrower args if needed]…\n`.
   The hint teaches the model to narrow its query instead of hallucinating
   about the missing middle.
3. **Trace honesty:** the `tool_result` trace event records
   `{ chars: original, kept: capped }` so runs stay auditable.
4. **Delegate results exempt from double-truncation:** child answers are
   already capped by `ANSWER_CAP_CHARS`; the delegate path applies the same
   cap once at the parent boundary (no special case needed if the cap is
   applied uniformly in `execOneTool`).

### Verify

- Unit test: 20k-char result → exactly 4000 kept, head/tail intact, marker
  present with correct N.
- Unit test: 3999-char result → untouched.
- `examples/agent_test.js`: add a scripted tool returning 50k chars; assert
  the loop's second-round prompt (mock provider captures messages) contains
  the truncated form.

---

## P4 — Smarter memory writes (`src/js/agent.js` `storeAndFinish`)

Today every turn stores `clip(task, 4000)` + `clip(answer, 4000)` raw
(`agent.js:286-297`), and the task is embedded a second time just for
storage (`agent.js:777`). The brain fills with noise → recall quality drops
→ P2's threshold has to fight harder. Store less, store better.

### Changes

1. **Reuse the recall embedding:** `storeAndFinish` takes the task vector
   already computed at recall time (`agent.js:534`) instead of re-embedding.
   Kills one full embed per turn (TF-IDF is cheap but remote embedders are
   a network call + cost).
2. **Skip trivial turns** (nothing worth remembering):
   - answer < 40 chars, or `(no response)` / `(cancelled)` (existing),
   - answer is only tool errors,
   - task < 20 chars (e.g. "ok", "next", "continue").
3. **Distill, don't dump:** store `clip(answer, 1200)` instead of 4000 —
   the first ~1200 chars of an answer carry the conclusion; the tail is
   usually elaboration/code that recall can't use anyway. Task stays at
   `clip(task, 4000)` → `clip(task, 1200)` for symmetry.
4. **No schema change:** CMA's near-duplicate gate (`cma.rs:236`) already
   reinforces instead of duplicating — with cleaner inputs it does its job.

### Verify

- Unit test: trivial-turn matrix (short answer / short task / cancelled /
  tool-error-only) → `scopeStore` never called.
- Unit test: stored text length ≤ 1200 for a 10k answer.
- Unit test: embed call count for one turn == 1 (mock `embedTextFor`
  counter), down from 2.

---

## P5 — Auto-compaction (`crates/sofuu-core/src/chat.rs`)

History trim today **drops** oldest turns past 85% of the window
(`chat.rs:2973-2990`) — silent info loss, and the warning tells the user to
run `/compact` manually. Compaction should be automatic and lossless-ish.

### Changes

1. **Trigger:** in `trimHistory()`, when `historyTokens() > 0.70 × ctxBudget()`
   → auto-compact instead of dropping. (70% leaves room for the compaction
   call itself + the next turn.)
2. **Shape:** reuse the existing `compact()` summarizer (`chat.rs:3468`)
   but keep the **last 4 turns verbatim** and summarize only the older
   prefix:
   ```
   history = [ {role:'system', content:'Prior conversation summary: ' + summary},
               ...last 4 turns... ]
   ```
   (The summary entry uses role `system` exactly like manual `/compact`
   already does — `agent.js` passes history through unchanged, and all
   providers accept a mid-array system message or fold it harmlessly;
   verify small-model behavior at impl time and fall back to a `user`-role
   `[summary]` message if any wire format rejects it.)
3. **Guard:** one auto-compact per threshold crossing (boolean flag, reset
   when history shrinks below 50%) — prevents compact-of-compact loops when
   a single giant turn keeps the window hot.
4. **User signal:** one dim line `⚙ auto-compacted N turns → summary (M tk saved)`.
5. **Manual `/compact` unchanged.**

### Verify

- Unit test (JS driver level, mock `complete`): seed history at 75% of
  budget → after `trimHistory()`, history = summary + last 4 turns,
  summarizer called exactly once.
- Unit test: second crossing without shrink → no second compaction.
- Existing trim-fallback test (drop path) still green for the case where
  compaction itself fails (network error → fall back to dropping, never
  block the turn).

---

## P6 — Prefix stability & prompt caching (provider-neutral)

**The point is the ordering, not any one provider.** A byte-stable prefix
means every backend reuses what it already processed:

| Backend | Mechanism | What Sofuu must do |
|---|---|---|
| OpenAI / OpenRouter / together / etc. | Automatic prefix caching (≥1024 tokens) | Nothing API-side — just keep the prefix stable (P1 does this) |
| Ollama / local servers | KV-cache reuse on identical prefix | Same — stable prefix; this is also the Track B fusion-loop precondition |
| Anthropic | Explicit `cache_control` breakpoints | Emit markers (small, one code path) |
| Gemini | Implicit context caching | Stable prefix helps; no API change |

### Changes

1. **Ordering enforcement** (done by P1.2): `[static system] → [dynamic
   context msg] → history → user`. Nothing dynamic may precede the static
   system string. `lastShared` mesh text moves out of the system prompt
   (`chat.rs:3172`) — this alone fixes the biggest stability bug.
2. **Anthropic markers** (`rt/ai.rs` `build_anthropic_body_v2`, ~:425):
   emit `system` as a content-block array with
   `cache_control: {type:"ephemeral"}` on the last system block, and the
   same on the last tool definition in `append_tools_anthropic`. Only when
   tools or a system prompt are present; no-op otherwise. This is the ONLY
   provider-specific code in the plan — it is a wire-format detail, not a
   design focus.
3. **Usage accounting** (`rt/ai.rs` usage extraction, ~:1719-1737): parse
   `cache_creation_input_tokens` / `cache_read_input_tokens` from Anthropic
   `message_start`/`message_delta` usage into `stream.usage.cacheWrite` /
   `usage.cacheRead` (0 for providers that don't report). OpenAI's
   `prompt_tokens_details.cached_tokens` parsed into `cacheRead` too.
4. **Surface:** `/cost` (`chat.rs` F6) shows cache hit tokens when non-zero:
   `… · cache 12.3k tk`. No new config.

### Verify

- Unit test: Anthropic body for a chat-shaped request contains exactly one
  `cache_control` on the system block and one on the last tool; OpenAI body
  is byte-identical to today except nothing (no accidental changes).
- Unit test: usage parser extracts cache fields from recorded Anthropic +
  OpenAI SSE fixtures; absent fields → 0.
- Integration: two consecutive identical turns against the mock provider →
  request bodies share a byte-identical prefix up to the last user message
  (assert with a byte diff in the test).

---

## P7 — Memory hygiene / RAM discipline (M1–M6)

**Rule:** the CLI uses as much RAM as the work needs and no more. Long
sessions must be flat-lined, not climbing. Each fix is small; together they
close every unbounded structure found in the audit.

### M1 — Drain `watchChangesPending` (`chat.rs:2719,3858`)

- After surfacing changes to the chat, clear the array (it is only used as
  a pending queue — verify no other reader at impl time; if `/watch` list
  reads it, move to a capped ring of the last 100 entries).
- Hard cap 100 entries regardless (drop oldest).

### M2 — Per-turn GC (`rt/engine.rs`)

- After each completed chat turn (driver `turn()` finally-block →
  new bridge `__chat_gc()` or reuse of an existing engine hook), call
  `JS_RunGC`. QuickJS full GC is ~ms at chat heap sizes; doing it once per
  turn (not per message, not per timer tick) bounds garbage to one turn's
  worth.
- Also run GC after `runMany`/`mapContext` fan-outs complete (agent.js
  can't call GC directly — expose via the same bridge; headless runs get it
  at engine teardown as today).

### M3 — CMA decay + physical prune (`memory/cma.rs`, `rt/memory.rs`, `chat.rs`)

- **Call decay:** the chat's existing 2s metrics tick (`chat.rs:3796`)
  calls `brain.decayTick(dt)` for the open brain (agent brains decay on
  their next `run()` entry — add a `lastDecay` timestamp per `BRAINS` entry
  in `agent.js` and tick on open).
- **Physical prune:** new `Cma::retain()` — drops records whose strength
  fell below the forget floor (the same threshold `recall` already filters)
  AND which are not pinned (`/remember` pins survive — verify the pin flag
  path at impl time). Called opportunistically: after `flush()`, at most
  once per 100 stores, so the Vec actually shrinks instead of growing
  forever behind tombstones.
- Brain-file format unchanged (QTSQ adapter writes surviving records; old
  files load fine — ids may renumber, recall is by vector so no caller
  depends on stable ids across restarts EXCEPT `/why` within one session,
  which is fine).

### M4 — Cap agent trace (`agent.js:485-493`)

- `TRACE_CAP = 512` events (RLM parity, `episode.rs:123`). On overflow:
  keep first 16 (`start`/`plan` skeleton) + last 496, drop middle payloads
  (keep `{kind, t}` stubs so `renderTrace` still shows the shape).
- `subRuns` unaffected (bounded by `maxDepth` × `maxSteps` already).

### M5 — Honest `RUN_COUNTS` (`agent.js:60`)

- Store `{n, day}` (day = `Date.now()/86400000 | 0`); on increment, reset
  `n` when the day rolled over. `list().runs24h` becomes truthful.

### M6 — RSS soft guardrail (`chat.rs`)

- The footer already computes RSS (`chat.rs:2041`). Add: past
  `RSS_WARN_BYTES` (default **1GB**, config `rss_warn_mb`) the footer RAM
  metric turns amber (like the ctx metric at 80%) and a one-time dim notice
  is printed. **Inform, never kill** — the user's work is not ours to abort.
- This is a tripwire for regressions, not a limiter: if it fires, something
  above leaked.

### Verify

- Unit test M1: simulate 200 watch polls → pending ≤ 100, drained after read.
- Unit test M3: seed CMA with weak + pinned records, run decay past
  half-life, `retain()` → weak gone, pinned present, `count()` shrank.
- Unit test M4: run with 600 trace events → result.trace.length == 512,
  first event is `start`, last is `answer`.
- Unit test M5: fake day rollover → counter resets.
- Soak check (manual, recorded in TASKS.md): 50-turn scripted chat session
  with `/watch` on and brain on → RSS sampled per turn via `__chat_rss` →
  flat line (±10%) after turn 5.

---

## Implementation order

```
P1 (prompt core + ordering)   ← foundation; P6 depends on its ordering
P2 (recall gating)            ← independent, agent.js
P3 (tool truncation)          ← independent, agent.js
P4 (memory writes)            ← independent, agent.js (touches same recall
                                 code as P2 — land together or sequentially)
P5 (auto-compaction)          ← independent, chat.rs
P6 (caching markers + usage)  ← after P1; ai.rs only
P7 (M1–M6 hygiene)            ← independent; can parallelize with anything
```

Suggested commits: `P1`, `P2+P4` (recall/store are one code path),
`P3`, `P5`, `P6`, `P7`.

## Files touched

| File | Changes |
|---|---|
| `src/js/agent.js` | P1 core + composition, P2 gating, P3 truncation, P4 store, M3 agent-brain decay, M4 trace cap, M5 run counts |
| `crates/sofuu-core/src/chat.rs` | P1 chat def + `lastShared` move, P5 auto-compact, M1 watch drain, M2 GC hook, M3 decay tick call, M6 RSS tripwire |
| `crates/sofuu-core/src/rt/ai.rs` | P6 Anthropic `cache_control`, cache usage parsing (Anthropic + OpenAI) |
| `crates/sofuu-core/src/rlm/episode.rs` | P1.3 driver prompt compression |
| `crates/sofuu-core/src/memory/cma.rs` | M3 `retain()` physical prune |
| `crates/sofuu-core/src/rt/memory.rs` | M3 expose `retain` on the CMA shell (if not already reachable) |
| `crates/sofuu-core/src/rt/engine.rs` | M2 GC hook for the bridge |

## Global verification wall

1. `cargo test` — 177 existing + new unit tests per section above, all green.
2. `./sofuu run examples/agent_test.js` — 46-check battery green (add P3
   truncation + P2 gating checks).
3. `./sofuu run examples/rlm_mock_test.js` — green after P1.3.
4. `make test` — JS parity suite green.
5. `make size-check` — binary ≤ 5MB (current 2.0MB; this plan adds ~0).
6. Soak: 50-turn session RSS flat line (P7 verify).
7. Token accounting proof: with brain on, a 10-turn scripted session reports
   strictly lower cumulative `promptTokens` than the same session on the
   pre-plan build (mock provider, same prompts) — the plan's own acceptance
   metric.

## Config fields added (all optional, all defaulted)

| Field | Default | Meaning |
|---|---|---|
| `recall_min` | 0.30 | Recall similarity floor (per-def `recallMin` too) |
| `recall_budget` | 1024 | Recall block token budget (per-def `recallBudget` too) |
| `budget.maxToolResultChars` | 4000 | Per-tool-result context cap |
| `rss_warn_mb` | 1024 | Footer RAM tripwire |

## Risks

| Risk | Mitigation |
|---|---|
| Tiny system prompt degrades small/local models (they sometimes need more hand-holding) | The core is deliberately plain-English with no abbreviations; per-def `system` override always composes after it — users of small models can add their own scaffolding without touching the runtime |
| Recall threshold 0.30 too aggressive for remote embedders (different cosine scale) | Per-def override + the dedupe/budget layers still apply; threshold is one constant, easy to retune once E0 (PLAN-RLM embedder benchmark) lands real numbers |
| Tool truncation hides the fact a model needed | Head+tail + explicit "re-call with narrower args" marker teaches recovery; per-def cap override for tools that legitimately return large payloads |
| Auto-compaction spends a turn summarizing at the wrong moment | Only triggers at 70% (not mid-flow), one per crossing, failure falls back to the existing drop path — never blocks |
| Anthropic `cache_control` on non-cacheable models errors | Markers are valid on all current Anthropic models; if a future model rejects them the API error is surfaced like any other (no silent retry logic) |
| `retain()` renumbers record ids mid-session | Only `/why` holds ids within a session and it reads them fresh each turn; cross-restart id stability is not a documented contract |
