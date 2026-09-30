# AUDIT — "(no response)" Root Cause Report

**Date:** 2026-08-24 · **Status:** root cause identified, fix pending approval
**Scope:** `sofuu` chat CLI — intermittent `(no response)` after tool-using turns
**Model repro'd on:** `openrouter/stealth/ox-alpha` · effort high · ctx 1M · coding tasks

---

## TL;DR

`(no response)` is **one symptom with two independent root causes**:

| # | Cause | Status |
|---|---|---|
| (a) | **Genuinely empty provider streams** — HTTP 200 with 0 text + 0 tool_calls | fixed (finish_reason capture + retry + cause surfaced) |
| (b) | **Silent `maxSteps` budget breach** — chat caps planning rounds at 8; breach exits with an empty answer and prints nothing | **found in this audit, NOT yet fixed** |

Every `(no response)` the user has seen is one of these two. Cause (b) is the
one in the latest screenshot and the most frequent on real coding tasks.

---

## Cause (b) — silent budget breach (the critical finding)

### Proof chain

1. **Chat def caps planning rounds at 8** — `crates/sofuu-core/src/chat.rs:3355`
   ```js
   budget: { maxSteps: 8, maxDepth: 1, maxTokens: 1e9, maxWallMs: 1e9 },
   ```
2. **Breach check at the top of every tool-loop round** — `src/js/agent.js:1127-1128`
   ```js
   var breach0 = budgetBreach();
   if (breach0) { stopped = breach0; break; }   // ← no final answer attempted
   ```
   `budgetBreach()` at `src/js/agent.js:790-796` returns `'budget_steps'` when
   `steps >= d.budget.maxSteps`.
3. **`answer` is initialized `''`** — `src/js/agent.js:958` — and on the breach
   path it is never set. `finish('')` produces `res.answer: ''`
   (`src/js/agent.js:802`: `answer: answer || ''`).
4. **Chat renders the empty answer as the fallback string** — chat driver `turn()`:
   `answer = String((res && res.answer) || '(no response)')` — `'' ||` → `(no response)`.
5. **The breach itself is never reported** — `crates/sofuu-core/src/chat.rs:3476`
   renders only `res.stopped === 'cancelled'`; `'budget_steps'` (and
   `'budget_wall'` / `'budget_tokens'`) are silently dropped. The user sees
   `(no response)` with zero explanation.

### Why the model burns its 8 rounds (aggravators)

- **Tool-result truncation forces re-reads.** `read_file` on a 4,655-line file
  returns head+tail only (dynamic cap ≈ 24k chars at a 1M window,
  `toolResultCapChars` in `src/js/agent.js`). The model literally says
  *"The middle got truncated. Let me read the specific sections…"* — then the
  section reads are truncated too → it re-reads again. Each re-read burns one
  of the 8 rounds.
- **Reasoning loop.** ox-alpha re-emitted the *identical* thinking text at
  steps 7, 8 and 9 (screenshot evidence) — rounds consumed without progress.
- **The ML layer detects but does not act.** The new `sofuu.ml` supervisor
  labels the tool lines `ml · reread_unchanged · read_file (step 7/8/9)`
  (`crates/sofuu-core/src/ml/context.rs`) — purely informational; it neither
  injects its nudge into the next prompt nor breaks the loop.

### Net effect

Any real coding task that needs ≥ 9 planning rounds (very common: read file →
read sections → grep → edit → verify) **deterministically ends in a bare
`(no response)`** after round 8, indistinguishable from a dead provider.

---

## Cause (a) — empty provider streams (already fixed, kept for the record)

- **Signature:** provider returns HTTP 200; the SSE stream carries zero text,
  zero tool_calls, zero error frame → old code fell through to
  `'(no response)'` (`src/js/agent.js` tool-loop + `finalAnswer`).
- **Why it was undiagnosable:** `finish_reason` was never captured on any
  wire. Fixed in `crates/sofuu-core/src/rt/ai.rs`: `capture_finish_reason()`
  (OpenAI `choices[0].finish_reason`, Anthropic `delta.stop_reason`) → stored
  on `AiStreamReq` → surfaced as `usage.finishReason` in the stream's done()
  stats.
- **Recovery now in `src/js/agent.js`** (both the tool loop and `finalAnswer`):
  - `finish_reason: "length"` → retry once with a dynamic
    `max_tokens = ⌊ctxWindow × 0.125⌋` (pure ratio; no flat constant).
  - otherwise → retry once without effort (never persisted to
    `no_think_models` — a transient empty must not disable thinking forever).
  - still empty → **throws** `provider returned an empty stream
    (finish_reason: …)` so the chat shows ✗ + the cause instead of a silent
    fallback.
- **Verified** with a scripted mock (tool round → empty `finish_reason:"length"`
  stream): the cause is surfaced, not swallowed.
- **Note:** `max_tokens` is a ceiling, not a target — sending an explicit
  `/maxout 384000` is honored as-is; well-behaved endpoints clamp. The
  lower-cap retry exists only as a workaround for gateways that empty on
  oversized caps.

---

## Distinguishing the two in the wild

| Symptom | Cause |
|---|---|
| `(no response)` after ~8+ tool rounds on a long task, no ✗ line | **(b) budget breach** |
| `(no response)` on the first round, or mid-task with few tool calls | **(a) empty stream** — new build prints `✗ provider returned an empty stream (finish_reason: …)` instead |

⚠️ Both fixes only exist in the repo binary `./sofuu`. `/usr/local/bin/sofuu`
was last installed 2026-08-15 and contains none of this — always launch
`./sofuu` (or `sudo make install`) before judging results.

---

## Recommended fixes (pending approval — NOT implemented)

1. **Report the breach** (chat driver): render e.g.
   `⏹ stopped: step budget (8 rounds) — asked to stop before answering` for
   `stopped: 'budget_steps' | 'budget_wall' | 'budget_tokens'`, mirroring the
   existing `cancelled` line (`chat.rs:3476`).
2. **Raise / dynamize the chat `maxSteps`** (flat `8` today): scale with task
   complexity or default coding chat to 24–32; expose `/maxsteps <n>`.
3. **Anti-loop guard:** when `sofuu.ml` flags `reread_unchanged`, inject the
   nudge into the next round's context (it already computes one) or force the
   model to answer from what it has — stops the round-burning at its source.
4. **Breach → salvage answer:** on breach, spend ONE final round asking the
   model to summarize its findings as the answer instead of returning `''`.

---

## Reproduction artifacts

- Screenshot 2026-08-24: steps 7/8/9 visible, identical thinking ×3,
  `ml · reread_unchanged` labels, `(no response)` at ctx 96.7k/1000k.
- Mock repro for (a): `/tmp/probe_empty3.js` pattern — tool round OK → empty
  `finish_reason:"length"` stream → cause thrown (passed).
- Tool write/edit sanity on new binary: `write_file` → `read_file` →
  `edit_file` all OK (cwd jail correctly rejects escapes). Not a tools bug.
