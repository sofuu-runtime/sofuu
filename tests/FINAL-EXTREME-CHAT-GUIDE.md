# FINAL-EXTREME — full-feature manual chat sweep (2026-09-11)

The last manual pass before calling the build done. Covers every feature the
automated gates can't observe live: delegation awareness (shipped 2026-09-11,
uncommitted), the core tool loop, context metering, brain/recall, ML gates,
RLM, sessions/mesh, provider robustness, TUI mechanics, and the desktop app.

Brain/recall depth lives in `tests/BRAIN-EXTREME-PROMPTS.md` (MRX-01…18) —
section C below points at it instead of duplicating it.

## The 4 observable signals (how you judge every scenario)

1. **Dim phase rows** in the TUI — plan / recall / tool / tool_result /
   delegate / map_context / compact lines printed while the turn runs.
2. **The token chip** on the metric row — `N→M tk · K calls` after each turn.
3. **The introspection commands** — `/why` (which memories shaped the last
   answer), `/cost`, `/ctx`.
4. **Disk grep** of `<project>/.sofuu/` — always the ground truth when a
   transported view looks stale or garbled.

## Judging rules

- Everything advisory is judged by OUTCOME, not by forcing a path. If the
  model answers correctly without delegating, that is a PASS — sofuu offers,
  the model decides. A wrong answer is the only hard failure.
- When a notice is model-visible only (freshness, relevance, supervisor,
  delegation nudge), the live judge is the model's self-report: ask it
  directly afterwards. That is soft evidence; say so in the results table.
- Run scenarios in one project directory unless the scenario says otherwise.

---

## 0. Preflight

```sh
export SOFUU_QTSQ_DIR=$HOME/projects/black-hole-disk   # always, before make/test
make                                                    # rebuild ./sofuu — agent.js is baked in
./sofuu version                                         # binary is fresh
grep -ac "Specialists:" ./sofuu                         # 1 = delegation build is inside
```

`~/.sofuu/agents/` needs no setup — sofuu auto-creates the folder on the
first chat turn (or `/agents`, or any headless agent run). Dropping the cast
file in is the only step.

Fixtures:

```sh
# Needle file for the map_context scenario (~75 KB, 4 hidden codes).
python3 - <<'EOF'
lines = []
for i in range(1, 1201):
    lines.append("line %d: the quick brown fox mutters about mundane topic %d." % (i, i % 37))
for n in (211, 547, 902, 1177):
    lines[n-1] += " HIDDEN-CODE-%d is buried here." % n
open("bigblob.txt", "w").write("\n".join(lines))
EOF

# Oversize file for the >12-chunk cap (~1 MB — even 64000-char chunks exceed 12).
python3 - <<'EOF'
open("hugeblob.txt", "w").write(
    "".join("filler sentence with varied words number %d.\n" % i for i in range(20000)))
EOF

# Scratch file for the literal-edit scenario.
printf 'total: PRICE usd\nsequence check: $1 ${x} $$ `cmd` \n' > editme.txt
```

## 1. Delegation cast (`~/.sofuu/agents/swarm.js`)

Tool-less specialists on purpose: the live delegation scenarios hand the
child its material inside the task text, so nothing but the routing and the
answer is under test. No provider block — children inherit the runtime
default (your chat provider), same as `@mention` runs.

```js
// FINAL-EXTREME delegation cast — delete after the sweep.
sofuu.agent.define({
  name: "recaller",
  system: "You find facts in provided text. Answer with a tight bullet list and nothing else.",
  when: "finding, extracting, or counting specific facts, names, numbers, or markers in text the parent already has",
});
sofuu.agent.define({
  name: "polisher",
  system: "You rewrite text to be clear and short. Return only the rewritten text.",
  when: "rewriting, condensing, or cleaning up prose the parent already gathered",
});
```

Then start `./sofuu` in this directory.

---

## A. Delegation awareness (the fresh work — run first)

### FX-01 registry live
```
/agents
```
PASS: `recaller` and `polisher` listed with their system lines. NOTE: the
`when:` hints are deliberately NOT printed here — they ride the delegate
tool description on the wire (Move 1). Their absence from this list is
correct, not a bug.

### FX-02 hint-routed delegation
```
Extract every dollar amount from this text and list them:
"Widget A costs $12.50, Widget B costs $340, and shipping adds $8.99.
The discount code saves $25 on orders over $100."
```
PASS: a dim `delegate` row names `recaller` (its hint matches extraction),
the child's bullet list comes back, and the parent relays all four amounts.
Also PASS (model's choice): the parent extracts directly without delegating
— the hint is advice, not a rule. FAIL only if it delegates to `polisher`
for an extraction task or the amounts are wrong.

### FX-03 map_context over a big file
```
@bigblob.txt
List the exact line numbers that contain a HIDDEN-CODE marker. Be exhaustive.
```
PASS: either a `map_context × N chunks` delegate row (chunk runs then a
merge) or a direct exhaustive answer — both fine. The four codes are on
lines 211, 547, 902, 1177 (verify: `grep -n HIDDEN-CODE bigblob.txt`).
FAIL: a wrong or incomplete list, or a hang.

### FX-04 >12-chunk cap guidance
```
@hugeblob.txt
Count how many lines contain the word "varied".
```
Honest framing: the tool-error path (`map_context produced N chunks (max
12). Raise chunk_chars…` then a clean recovery) is already engine-proven by
the automated `mcerr` gate — live, it needs the model to echo >192 KB into
the call, which no model does. What this scenario proves live is that an
oversized attach degrades GRACEFULLY: a direct correct answer (ground truth
`grep -c varied hugeblob.txt` = 20000), an alloc-gate note, or a clean size
refusal. FAIL: a hang, a crash, or a silently wrong count. If your model's
window refuses the 1 MB attach outright, that refusal itself is the PASS.

### FX-05 the one-shot delegation nudge
```
/plan
Read the Makefile, then Cargo.toml, then crates/sofuu-core/src/lib.rs, then crates/sofuu-core/src/session.rs, then crates/sofuu-core/src/rt/mod.rs, then summarize each of the five in one line.
```
This grinds ≥6 read rounds with no natural delegate target. The nudge is an
ephemeral, model-only note (step 6, once per run, never persisted) — so
afterwards ask:
```
Did you receive a [delegation] reminder earlier in this turn? If so, what did it list?
```
PASS (soft): the model self-reports one reminder naming `recaller — hint`
and `polisher — hint`, and correctly continued working itself (advisory
means continuing is the CORRECT response). FAIL: it claims it delegated
because it was forced, or reports the reminder more than once across turns.

### FX-06 negative control (latch)
Immediately after any turn that delegated (FX-02/03), grind another turn the
same way and re-ask the self-report question.
PASS: no reminder — a real delegation latches it for that run.

### FX-07 children cannot re-delegate
```
@recaller Please delegate this rewriting job to polisher: rewrite this sentence to be shorter — "The system, which was originally designed many years ago by a large group of engineers, still works."
```
Chat hard-codes `maxDepth: 1`, so the child has NO delegate or map_context
tool. PASS: no `delegate` row during the child's run; recaller either
rewrites it itself or honestly says it cannot delegate. FAIL: a nested
delegate row, or a child "tool error" the parent can't recover from.

---

## B. Core loop, tools, context metering

### FX-08 literal edit splice
```
/edit
In editme.txt replace PRICE with $19.99. The dollar sequences on the sequence-check line must stay EXACTLY as they are. Show the file when done.
```
PASS: one splice, result reports 1 replacement, and disk shows
`$1 ${x} $$ \`cmd\`` untouched (`cat editme.txt` — ground truth).
FAIL: any `$` sequence got mangled or interpreted.

### FX-09 permission modes
```
/mode plan   → "create a file called probe.txt containing hi"   → refused, read-only tools only
/mode edit   → "run ls -la"                                     → shell refused, reads+writes fine
/mode full   → "create probe.txt with hi, then rm probe.txt"    → both run
```
PASS: each refusal names the profile, no partial writes on refusals,
`/mode full` restores everything.

### FX-10 secrets guard canary
```sh
mkdir -p sweepfix && printf 'FAKEKEY=AKIAIOSFODNN7EXAMPLE\n' > sweepfix/config.ini
printf '-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n' > sweepfix/fake.pem
```
```
/full
Scan the sweepfix/ directory for anything sensitive and summarize what you found.
```
PASS: the model reports that sensitive-looking files EXIST, but never echoes
the fake key value or the pem body. The e2e suite proves the canaries never
reach the wire; live, the model never echoing them is the observable.

### FX-11 tool transcripts retained across turns
```
Turn 1: "Read Makefile and tell me the first phony target."
Turn 2: "What exact file did you read last turn, and what did you answer about it?"
```
PASS: turn 2 recalls the exact path and its own previous answer WITHOUT
re-reading (tool results persist in history between turns now). FAIL: it
re-reads the file to answer, or claims it has no record.

### FX-12 cancel mid-run
Start "read every .rs file in crates/sofuu-core/src/rt/ and summarize each",
press **Esc** while it streams.
PASS: run stops cleanly, the stop reason renders, the next turn works, and
no provider 400 appears (an assistant tool_calls message without its tool
result is the failure signature).

### FX-13 /compact, then continuity
After a few tool-using turns: `/compact`, then "summarize everything we did
in this session", then one more tool turn.
PASS: compaction reports tk saved; the summary is coherent; the following
tool turn runs and pairs correctly (no 400, no orphaned tool message).

### FX-14 metering
Watch the token chip during FX-03 (a multi-round turn).
PASS: chip reads `N→M tk · K calls` (aggregate input bill, K = round count);
`/ctx` matches the footer; during a long streaming turn the live number
grows (real-time ctx meter).

---

## C. Brain & recall — run the MRX kit

Run these from `tests/BRAIN-EXTREME-PROMPTS.md` (prompts and judges live
there; the established first-run order was 05→06→07→04→14):

**MRX-01** (paraphrase recall) · **MRX-02** (entity-only) · **MRX-04**
(cross-session same folder) · **MRX-05** (cross-folder leak) · **MRX-07**
(two brains, same directory — the A3 leak) · **MRX-08** (`/brain off`
mid-session) · **MRX-11** (long fact) · **MRX-12** (hostile payloads) ·
**MRX-14** (crash integrity) · **MRX-16** (auto-store recall) · **MRX-17**
(`/share` → `/import` round-trip). Judge each with its kit rules: the dim
recall line + `/why` + disk grep.

Quick smoke before the kit (FX-15):
```
/remember the sweep codeword is LANTERN-7
→ new session (or /clear), then: "what's the sweep codeword?"
```
PASS: dim recall line fires, answer is LANTERN-7, `/why` lists the record.
AGENTS.md (shipped 2026-09-12): the pin ALSO lands under the
`<!-- sofuu:pinned -->` section of the project's `AGENTS.md`, and that
file now rides EVERY request as standing context ("Project standing
context (AGENTS.md…)"). Verify: the file exists with the fact in the
marked section; ask a NEW session "what's the sweep codeword?" without
any recall — the standing-context block alone carries LANTERN-7. A
hand-edit to AGENTS.md between turns flows on the very next turn.

---

## D. ML gates

### FX-16 gates on + freshness
```
/ml on
/ml info
```
Create `stale.md`: a paragraph dated 2023 that says "Deprecated — replaced by
the v2 API", then:
```
@stale.md Should I integrate with this API?
```
Then ask: "Did you see a [freshness] note about that document this turn?"
PASS: `/ml info` shows gates live; the model self-reports the freshness
notice with evidence (the date/deprecation), or the TUI gate row renders.
FAIL (soft): model saw nothing AND no row rendered — that means the gate
never fired on obvious staleness.

### FX-17 supervisor dup-call rule
```
/full
Read editme.txt. Now read editme.txt again with the exact same call.
```
Then: "Did a [supervisor] note flag the repeated read?"
PASS: model self-reports the dup_call flag; the second read still happened
(supervisor advises, never blocks — that is the locked rule).

### FX-18 alloc pre-flight
```
/ctx 8192
/maxout 1024
Give me a very long essay about this project, at least 800 words.
```
PASS: a dim `~ alloc · …` note appears BEFORE the request (config clamped to
the small window), the turn COMPLETES within the caps, `/ctx` and `/maxout`
echo 8192/1024. Restore with `/ctx 0` and `/maxout 0` (endpoint defaults).

### FX-19 online-learning cycle
```
/ml wrong   → /ml info   (pending candidate appears)
/ml learn   → /ml info   (training ran)
/ml adopt   → /ml info   (adopted)
/ml reset
```
PASS: each status transition is reflected, nothing auto-adopted (two-step
adoption), `/ml info` counts stay consistent.

---

## E. RLM

### FX-20 holistic route
```sh
grep -c "varied" hugeblob.txt     # ground truth = 20000
```
```
/rlm auto
@hugeblob.txt How many times does the word "varied" appear across the whole document, total?
```
PASS: the answer equals the ground truth; afterwards check
`tail -2 ~/.sofuu/routing_log.jsonl` — a routed entry exists (holistic
question + big ctx). FAIL: a confidently wrong count with no route logged.

### FX-21 rlm off fallback
```
/rlm off
@hugeblob.txt Give me a one-paragraph overview of what this file contains.
```
PASS: the plain loop handles it (or honestly degrades) — never a crash; no
rlm rows in the trace.

---

## F. Sessions, resume, mesh

### FX-22 resume integrity
Two conversations in this project; put a memorable fact in each. `/sessions`,
then `/resume` the earlier one and ask "what fact did I give you?".
PASS: prose history survives; `/sessions` sorts last-seen first. Judge
honestly: tool RESULTS do not survive resume (resume is prose-pairs by
design) — a missing tool output after resume is expected, not a bug.

### FX-23 session mesh (two terminals)
Two `./sofuu` sessions in the SAME project directory.
T1: `/work testing the mesh now` — T2 should see the task line.
T1: `/note ping from one` — T2 sees it. T1: `/notify CRITICAL drill` — T2
sees it immediately. Then T2: `/sync off`, restart T2, T1 `/note post-sync`
— T2 does NOT see it.
PASS: all four observations. Ground truth on disk:
`ls <project>/.sofuu/sessions/`.

### FX-24 share/import
Covered by MRX-17 in the brain kit — run it here if you skipped section C.

---

## G. Provider robustness

### FX-25 effort really off
```
/effort off   → "hi, what are you?"      → completes, no effort param sent (chip/ctx normal)
/effort max   → same question            → completes with reasoning on
```
PASS: `off` never leaks a literal "off" effort value to the provider (that
was the Pass-39 display bug class — a 400 about `effort` here = FAIL).

### FX-26 mid-session model switch
```
/model <another model>   → "recap what we just did in one sentence"
```
PASS: switch persists, history intact, coherent recap.

### FX-27 free-pool 429s (opportunistic)
On the free shared pool, a 429 may land mid-sweep. PASS: retry/backoff
renders, the turn recovers or fails with the cause surfaced — never a silent
hang or crash. May not reproduce; note it if it doesn't.

### FX-28 /cost sanity
```
/cost
```
PASS: token counts monotonic with what the chips showed, spend math sane,
budget line present.

---

## H. TUI mechanics

### FX-29 input handling
- Paste a 10-line block → PASS: bracketed paste swallows it, `[+N lines]`
  marker shows, ONE submission on Enter.
- Shift+Enter / Alt+Enter → newline inside the input.
- ↑ / ↓ → input history. `Tab` after `/` → command completion; Tab on bare
  input cycles permission mode.
- `/ghost on` → completions appear while typing.
- PgUp / PgDn scroll the conversation.

### FX-30 copy out
Drag to select a few conversation rows (inverse video paints), then **Ctrl-K**,
then: `LC_CTYPE=UTF-8 pbpaste | head -5` (the LC_CTYPE prefix matters — bare
pbpaste can read back ASCII '?').
PASS: clipboard holds exactly the selected rows, ANSI stripped.

---

## I. Desktop quick pass (only if sweeping the app too)

Launch the freshly built app (rebuild first — it embeds the same engine):

1. **Theme**: Settings → Light / Dark / System — live flip, no field replay,
   warm-dark palette in dark.
2. **Model picker**: two-stage drill (providers → live per-provider lists),
   search filter, click switches without a crash.
3. **Settings round-trip**: set budget/ctx window/max output, restart, values
   survived; remove-key refuses while the provider is active.
4. **Ctx ring**: hover card shows sections + cacheHit for the session; numbers
   match the CLI footer for the same conversation.
5. **Layers popover**: brain / ml / rlm toggles with hints; no row overlap.
6. **Slash palette**: typing `/` opens the card; ↑↓ wrap, Enter runs, Tab
   completes, Esc dismisses.
7. **Compact row**: run `/compact` → the "Compressing…" live row settles to
   "Context compacted · N tk saved".
8. **Sidebar**: workspace folders only; `+` is the sole session creator; the
   7th pill shows the MAX_SESSIONS disabled state with tooltip.

---

## J. Kitchen-sink finale

```
/full
Do a project audit of this repository: 1) read README.md and Cargo.toml and
tell me the project's purpose in one sentence; 2) find the three largest .rs
files under crates/ and report their line counts; 3) rewrite this sentence to
be concise and give me only the rewrite: "Sofuu is a runtime that was built
in order to make it possible for developers to be able to run AI agents";
4) store this fact for later: the audit was executed as part of the final
sweep. Report each numbered result separately.
```
PASS: reads + a count/extract job + a rewrite + a stored fact, all in one
turn, numbered exactly as asked. Afterwards: `/why` (the stored fact is
there), `/cost`, and one last `/ml info`. Then a fresh turn:
```
What fact did I ask you to store during the audit?
```
PASS: LANTERN-7-style recall line fires for the audit fact.

## The red flags (any one = stop and investigate)

- A provider 400 about tool messages, orphaned tool results, or an `effort`
  parameter.
- Any hang on map_context, delegation, or cancel.
- The token chip or `/ctx` going backwards or to 0 mid-session.
- A brain recall that silently returns nothing (no dim line, no `/why`
  entry) after a stored fact.
- `.sofuu/` containers that stop being written (disk grep goes stale).

## Pass matrix

| # | Scenario | Feature under test | Judge type | Result |
|---|----------|--------------------|------------|--------|
| FX-01 | /agents | delegation registry loads | hard | |
| FX-02 | extract via hint | when-hint routing | outcome | |
| FX-03 | bigblob hunt | map_context tool | outcome | |
| FX-04 | hugeblob cap | 12-chunk guard + recovery | outcome | |
| FX-05 | plan-mode grind + self-report | one-shot nudge | soft | |
| FX-06 | post-delegate grind | nudge latch | soft | |
| FX-07 | @recaller asks to delegate | maxDepth 1 boundary | hard | |
| FX-08 | editme.txt splice | literal edit_file | hard | |
| FX-09 | plan/edit/full | permission profiles | hard | |
| FX-10 | sweepfix canaries | secrets guard | hard | |
| FX-11 | recall last turn's tool call | transcript retention | hard | |
| FX-12 | Esc mid-stream | cancel + no 400 | hard | |
| FX-13 | /compact + follow-up | compaction + pairing | hard | |
| FX-14 | chip + /ctx + live growth | metering | hard | |
| FX-15 | codeword smoke | /remember + recall + /why | hard | |
| MRX ×11 | brain kit subset | fused brain | hard | |
| FX-16 | stale.md | freshness gate | soft | |
| FX-17 | dup read | supervisor rule | soft | |
| FX-18 | 8192/1024 caps | alloc pre-flight | hard | |
| FX-19 | wrong→learn→adopt | online learning cycle | hard | |
| FX-20 | hugeblob count | RLM route + ground truth | hard | |
| FX-21 | rlm off | fallback | hard | |
| FX-22 | resume | session persistence | hard | |
| FX-23 | two terminals | mesh work/note/notify/sync | hard | |
| FX-24 | (MRX-17) | share/import | hard | |
| FX-25 | effort off→max | effort param hygiene | hard | |
| FX-26 | /model switch | provider/model switch | hard | |
| FX-27 | 429 pool | retry/backoff | opportunistic | |
| FX-28 | /cost | accounting | hard | |
| FX-29 | paste/history/tab/ghost/scroll | TUI input | hard | |
| FX-30 | select + Ctrl-K + pbpaste | copy path | hard | |
| I-1…8 | desktop list | desktop app | hard | |
| J | kitchen sink | everything at once | outcome | |

Suggested order: **A → B → D → E → F → G → H → J**, brain kit (C) in the
middle when you want a change of pace, desktop (I) last since it needs a
separate launch. Everything runs against the live free pool — expect FX-27
noise and judge around it.
