# PLAN — Chat & Brain Features (the "novel features" batch)

> **✅ STATUS (2026-08-19): F1–F11 ALL LANDED AND VERIFIED.** All 9
> remaining features implemented in `crates/sofuu-core/src/chat.rs` +
> `main.rs`. `cargo test` 177 green (3 new tests: slash_dispatch_f1_f11,
> fs_watcher_basic, pricing_defaults_seeded), `make test` green, size
> 2.0MB ≤ 5MB. F4a/F4b were already landed (2026-08-16). Config fields
> added: pricing, budget_usd, spend_total_usd, verify, verify_model, ghost.
> 12 new slash commands: /remember /why /resume /share /import /cost
> /verify /watch /hooks /ghost /at /serve. 12 new bridge functions.
>
> **Scope:** the feature list from the 2026-08 review — `/resume`,
> `/remember` + `/why`, `/share` brain cards, `@file` mentions, tool-call
> streaming, `/verify`, `/cost`, `/watch`, ghost completion,
> `sofuu serve --brain`, `~/.sofuu/hooks.js`.
>
> **Corrections to the review (verified against the tree 2026-08-12):**
> - "Tool-calling chat over MCP" is **already shipped** (P3 agent loop:
>   `~/.sofuu/mcp.json` auto-connect, `/tools`, plan-with-`complete()` → tool
>   calls → final streamed answer). The real remaining work is the two deltas
>   already on TASKS.md's not-started list: **streaming tool-call accumulation
>   in `ai.stream`** and **multi-server routing** (`callMcpTool` is
>   first-server-wins). That is F4 below.
> - "Wire embedLocal" is **already shipped** (`js_ai_embed_local` in
>   `src/modules/mod_ai.c`; `/brain on` works with Ollama down). Removed.
>
> **Architecture primer (the hook points everything uses):**
> - Slash commands: `ALL_COMMANDS` table (`chat.rs:186`), normalization +
>   dispatch in `handle_slash` (`chat.rs:238`) returning op tokens, executed
>   by the JS driver dispatch (~`chat.rs:1750+`, pickers at `:1801–1807`).
> - Rust↔JS bridge: `__chat_*` globals registered in `register_bridge`
>   (`chat.rs:1066`) — e.g. `__chat_slash`, `__chat_getcfg`, `__chat_poll`,
>   `__chat_past_turns` (:1080), `__chat_mcpservers` (:1082).
> - Turn pipeline (driver const `DRIVER`, `chat.rs:1088`): `turn(text)` at
>   :1264 — `__chat_log('prompt')` → `recallContext` (brain recall :1227,
>   tracks `lastRecalledIds` :1234) → agent loop (plan `ai.complete` +
>   `callMcpTool` :1158, **first-server-wins** :1159–1163) → final
>   `sofuu.ai.stream` :1306 → usage accounting :1360 → `history.push` :1367 →
>   `storeTurn` :1373 → `markPositive` :1374.
> - Session mesh (`session.rs`): per-project `.sofuu/sessions/<id>.qtsq`;
>   `log_prompt`/`log_answer` already record every turn (:470/:478);
>   `SessionRegistry::read` (:313), `load_session_data` (:496),
>   `past_turns(project, id)` (:632), `short_id` (:163). `EVENT_CAP = 300`
>   (:28) — long sessions record only the most recent events.
> - `sofuu.ai.estimateTokens` exists (`mod_ai.c:2319`).
> - `extract_tool_calls` exists for **`ai.complete` only** (`mod_ai.c:851`,
>   attached :1519) — the streaming path has no tool-call handling.
> - No `uv_fs_event`/watch infra exists anywhere in `src/`.
> - Interactive selector overlay for pickers lives in the driver (:1564); C
>   routes keys, JS owns state. New pickers reuse this.

---

## F1 — `/remember "<fact>"` + `/why`  (smallest, do first)

### `/remember` — pin a fact directly to the brain
1. `chat.rs`: add `"/remember"` to `ALL_COMMANDS` (:186); in `handle_slash`
   return op `"remember"` when an arg is present; empty arg prints usage.
   Empty-arg and enabled-state cases are pure Rust (mirror `/brain` :480).
2. Driver op handler (extend the op dispatch ~:1750): parse the raw command
   for the fact text,then:
   ```js
   if (!ensureBrain()) return;
   const emr = await embedText(fact);            // Ollama → embedLocal fallback (exists :1203)
   if (!emr) { out('…embed unavailable'); return; }
   brain.remember(emr.vec, fact, 'user_pin', 0); // CMA remember API (storeTurn pattern :1244)
   brain.flush();
   out('  ⏺ remembered · ' + brain.count() + ' memories');
   ```
3. Confirm CMA tier/strength args at impl time (pinned facts should survive
   decay: verify against `memory/cma.rs` `remember` signature; if the current
   shell can't pin, bump the record's `strength` after insert via a small
   `sofuu_cma_pin` export — **only** if needed).

### `/why` — explain which memories shaped the last answer
1. Driver: change `lastRecalledIds` (array of ids, :1093/:1234) into
   `lastRecallHits` — full hit objects `{id, score, distance, role, tier,
   strength, text}` (recall already returns all of these; see recall-shape
   parity in TASKS.md). `markPositive` then uses `lastRecallHits.map(h=>h.id)`.
2. `/why` op → print table: rank · score (resonance) · tier/role · first ~90
   chars of text. No recall this turn / brain off → one-line explanation.
3. Clear `lastRecallHits` where `lastRecalledIds` is cleared today (:1376).

**Verify:** pty script — `/remember my deploy bucket is sofuu-dist` → ask
"where do I deploy?" in a fresh session → fact appears in the answer → `/why`
lists the pinned fact with its score. Add `handle_slash` unit tests mirroring
:1922–1939. Brain-off path prints cleanly.

---

## F2 — `/resume` — picker over the session registry

1. New bridge `__chat_sessions()` in `register_bridge` (:1066): JSON array of
   `{id, short, started_at, model, task, ended, turns}` from
   `SessionRegistry::read` (:313) + `past_turns` count. Exclude the *current*
   session id.
2. `ALL_COMMANDS` + `handle_slash`: `"/resume"` → op `"resume"`.
3. Driver: `resume` op → new picker kind reusing the selector overlay (:1564):
   rows = session short id + task/model/date; type-to-filter. On select:
   - `const turns = JSON.parse(__chat_past_turns(id))` (bridge exists, :1080 →
     `session::past_turns` :632),
   - replace `history` with `[{user…},{assistant…}, …]` from those turns,
   - print a dimmed transcript header (`⏺ resumed <short> · N turns · <task>`),
   - if events hit `EVENT_CAP` (300) note "partial history".
4. Safety: if the resume would exceed `MAX_HISTORY_ENTRIES` (60), keep the
   most recent 30 turns and say so. `lastShared`/brain untouched.
5. Edge: resuming an *active* (non-`ended`) session is allowed but warns
   ("session still live on host X").

**Verify:** pty — session A: two turns mentioning "project falcon", `/exit`.
Session B: `/resume` → pick A → ask "what project were we discussing?" →
answer says falcon. Confirm `history` bounds respected (61+-turn session A).
Unit-test the op token + a Rust test for empty-registry.

---

## F3 — `@file` mentions with budget trimming

1. Driver: `expandMentions(text)` called at the top of `turn()` (:1264),
   before `recallContext`:
   - tokenize `@path` / `@path:start-end` (allow quoted paths with spaces),
   - resolve against `process.cwd()`; reject paths escaping the project root
     (after realpath) — mirrors the npm safety posture,
   - `sofuu.fs.readFile(path, 'utf8')`, slice `[start,end]` lines,
   - `let tok = sofuu.ai.estimateTokens(chunk)` (exists, `mod_ai.c:2319`),
     cumulative cap `ATTACH_TOKEN_BUDGET` (default 8192, config-overridable):
     over-budget attachments are tail-trimmed with a `…[trimmed N lines, M tk]`
     marker,
   - the user message becomes: original text + `\n\nAttached files:\n`
     + one fenced block per file (`### path:Ls-Le\n```\n…\n````).
2. History discipline: history stores the *manifest* only
   (`@a.js:1-40 (1.9k tk)`), not the full text — otherwise every follow-up
   re-sends every file. Current turn gets full text. (Aligns with the
   `MAX_HISTORY_ENTRIES` philosophy at :1188.)
3. Errors (missing file, bad range, binary-looking content) → inline warning,
   mention removed, turn continues.

**Verify:** script — `explain @examples/hello.js` → answer references its
contents; two-mention budget-overflow → trim marker present, total ≤ budget;
`@../../etc/passwd` → rejected; follow-up turn doesn't re-attach.

---

## F4 — Streaming tool calls + multi-server routing (the real Tier-1 delta)

### F4a — routing fix (independent, land first) — ✅ LANDED 2026-08-16 (via PLAN-AGENTS A6)
Implemented exactly as specified: a name→owner map (built while merging
`mcpTools`) routes `callMcpTool` to the OWNING server; unknown tools and
disconnected owners produce readable errors. Extension beyond spec: on a
name clash, inline/builtin tools win over the MCP copy (A6 namespace rule)
with a visible warning — instead of "first wins" — in both the chat driver
and `src/js/agent.js`'s shared resolver. Verified with two REAL MCP child
servers (`examples/mcp_echo_server.js`): `srv2_tool` lands on srv2 (the old
code would have called srv1), and the clashing `shared_tool` resolves to the
inline tool with a warning event in the trace.
1. Driver `connectMcpServers` (:1104): build `toolOwner = {toolName → client}`
   while merging `mcpTools`. Collision → first wins + one-line warning.
2. `callMcpTool` (:1158): look up `toolOwner[name]`; unknown tool → current
   error. Remove the first-client-unconditional-return loop.

**Verify:** two mock MCP servers, different tools each; a planning round that
calls both servers' tools in one turn; collision warning prints.

### F4b — streaming tool-call accumulation in `ai.stream` (`mod_ai.c`)
> **Status note (2026-08-16):** the OpenAI-compat HALF effectively shipped with
> PLAN-AGENTS A1 — `ai.stream` already surfaces `delta.tool_calls` via
> `setToolCalls`, and `src/js/agent.js` merges the index-addressed fragments
> (concatenating `function.arguments`) before execution, so OpenAI-wire runs
> stream-first with tools at no extra generation. Still open here: the
> Anthropic content-block (`tool_use` / `input_json_delta`) accumulator in the
> Rust stream parser — until it lands, Anthropic-wire agent runs use the
> `complete()`-planning loop automatically.
Today: `ai.stream` yields `{text}` / `{think}` chunks; tool calls exist only
in `ai.complete` via `extract_tool_calls` (:851). Goal: after the stream ends,
`stream.toolCalls` has the same shape as complete's, and (optional) the
iterator emits `{toolStart:"name"}` markers so the TUI can show live activity.
1. OpenAI-compatible streaming: `choices[].delta.tool_calls[]` fragments are
   index-addressed — `id`/`function.name` arrive once, `function.arguments`
   JSON strings concatenate. Add an accumulator to the stream state:
   `vec<idx → {id, name, argsJson}>`, append-delta on each SSE event, finalize
   + `JSON.parse` arguments at stream end → `JS_SetPropertyStr(stream,
   "toolCalls", …)` alongside `usage`.
2. Anthropic streaming: `content_block_start` (`tool_use`) opens a block,
   `input_json_delta` appends partial JSON, `content_block_stop` closes it.
   Same accumulator shape; convert to the OpenAI-shaped `{name, arguments}`
   the driver already consumes.
3. Gemini streams function calls inline in `candidates[0].content.parts[]`
   (`functionCall`) — extract per chunk, no accumulation needed. Ollama's
   `/api/chat` streams tool calls fully-formed per chunk — pass through.
4. Driver `turn()`: after the streamed answer loop, `const tcs =
   stream.toolCalls; if (tcs?.length) → execute + append `toolMsgs` →
   re-stream. This collapses the current "plan with complete, answer with
   stream" two-phase into one loop — but **keep the complete-based loop as
   fallback for one release** (flag `cfg.agent_stream`), because providers
   differ wildly in streaming-tool compliance.

**Verify:** mock SSE server emitting *fragmented* tool_calls (args split
across 3 chunks) → driver executes the tool and streams the final answer;
`examples/ai_stream_test.js` extended with a tool-call fixture; Anthropic-shape
fixture; E2E rerun of the existing agent-loop battery; toggle fallback path
tested.

---

## F5 — `/share` — portable encrypted brain cards

1. **Export surface:** `mod_memory.c` shell: expose `brain.export()` → JSON
   `{records, vectors}` (CMA already has `records_json` + `vectors` exports in
   `ffi_exports.rs` — check what's bound in the shell; add the thin binding if
   missing).
2. **Card format:** reuse the QTSQ session-file machinery
   (`sofuu_qtsq_session_save/load` in `ffi_shim.c`, already bound for the
   mesh). Payload:
   `{"schema":"sofuu-brain-card@1","exported_at":…,"brain":{…export…},"session":{"task":…,"turns":[…]}}`.
   Password: **user-supplied** (prompt twice on export; stdin-env override
   `SOFUU_CARD_PASSWORD` for scripting) — *not* the project-derived mesh
   password.
3. `/share [path.qtsq] [label]` driver op → export → write card → confirm
   size/record count.
4. `/import <path>` op → password prompt → card parse → merge: for each
   record, skip if an identical text hash already exists in the current brain
   (build a Set from the local `records_json`), else
   `brain.remember(vec, text, role, tier)`; `flush()`; report N added / M
   skipped.
5. Document clearly: the card inherits QTSQ's proprietary-format standing
   (LICENSES/QTSQ-FORMAT.txt) — a line in README's share section.

**Verify:** brain A with 5 facts → `/share card.qtsq`; fresh brain path →
`/import` → recall returns the facts in order; wrong password → loud error;
import twice → second run adds 0, skips all.

---

## F6 — `/cost` + budget caps

1. Pricing table: `~/.sofuu/config.json` → `"pricing": {"openai/gpt-4o":
   [2.50, 10.00], …}` ($ per 1M input/output tokens); seed defaults for the
   popular models; unknown model → track tokens, `$` shows `?`.
2. Driver: the streamed answer already exposes `stream.usage` (:1360).
   Accumulate `sessionCost(promptTokens, completionTokens)` per turn;
   `refreshStatus` (:1366) footer gains `$0.0042` next to `ctx:`.
3. `/cost` op → breakdown table: per-turn (model, in/out tk, $) + session
   total. Persist all-time spend in config (`"spend_total_usd"`) so `/cost`
   shows lifetime too.
4. Budget: `"budget_usd": 0` (off) in config. Preflight at turn start: if
   session spend ≥ budget → block turn with a clear message; warn at 80%.

**Verify:** mock provider returning fixed usage; scripted two turns → `/cost`
arithmetically exact; budget `0.000001` blocks the second turn; footer shows
running spend.

---

## F7 — `/verify <provider>` — second-model pass with diff

1. Config: `"verify": "off" | "<provider>"` (+ optional `"verify_model"`;
   default = provider's configured model).
2. Driver: after the final answer in `turn()`, if verify is on →
   `sofuu.ai.complete` with the verify provider, same message list (reuse
   `complete()` helper :1257) → `diffWords(answer, check)` (~60-line LCS over
   word tokens) → print only disagreement spans (dim context, red ours, green
   theirs); identical-ish (>95% agreement) → `✓ verified`.
3. Print the one-time cost notice on `/verify <provider>` enable ("each turn
   costs one extra completion"); shown in `/cost` as its own line item.

**Verify:** two mock providers with a planted disagreement → spans shown;
agreeing pair → `✓ verified`; verify provider down → warn, answer unaffected.

---

## F8 — `/watch <path>` — filesystem changes in context

1. Rust, no new FFI: add `watch.rs` (or a section in `chat.rs`) — a
   `Watcher { paths: Vec<(PathBuf, HashMap<PathBuf, (u64 mtime, u64 len)>)> }`
   with recursive walk (skip `.git`, `node_modules`, `target`; cap 2k files,
   note the cap when exceeded).
2. Hook it into the existing 1-second poll: `__chat_poll` (:1076) already
   ticks for session-mesh notices — extend its JSON return with
   `"watch":[{"path":…,"kind":"modified|created|deleted"}]`. Drain + snapshot
   swap each tick.
3. `/watch <path>` op → Rust adds to the watcher (op carries arg via the
   raw-command pattern as in F1); `/watch` bare lists; `/watch off` clears.
   v1 is session-scoped (not persisted to config).
4. Driver: watch notices print as dim one-liners immediately, and a compact
   `Files changed since last turn: a.js (modified), b.js (new)` line is
   injected into the next message's system context (alongside `lastShared`).

**Verify:** pty — `/watch /tmp/w` → `echo x > /tmp/w/f.txt` from another
shell → notice within ~1.5s → next turn's answer references the change;
cap-behavior test on a big tree.

---

## F9 — `~/.sofuu/hooks.js` — user middleware

1. Driver init (next to `connectMcpServers`): if
   `~/.sofuu/hooks.js` exists → dynamic `import()` it (file URL); failure →
   one-line warn, hooks disabled.
2. Hook surface (documented, versioned):
   `export async function pre({ text, cfg })` → replacement prompt string
   (or `{ text, skip }`);
   `export async function post({ text, answer, usage })` → replacement answer
   or `void`.
3. Wiring: `pre` at the top of `turn()` after mention expansion (F3); `post`
   right before `history.push` (:1367). Hard guards: try/catch each call,
   timeout 3s; a hook that throws 3 turns running is disabled with a notice —
   hooks must never wedge the chat loop.
4. Trust model stays simple: hooks are the user's own file, full `sofuu.*`
   power, no sandbox. README gets a safety paragraph.

**Verify:** hook that prepends "answer in ≤3 sentences" visibly changes
answers; post-hook appends a signature; a throwing hook warns once and is
ignored; missing file = zero overhead path.

---

## F10 — Ghost prompt completion (Tier 3)

1. Store-side exists: user prompts land in the brain each turn (`storeTurn`
   role `'user'`, :1244) when `/brain on`.
2. Suggestion path must be **synchronous** (keystroke path): use
   `sofuu.ai.embedLocal` (sync FFI, no network) + `brain.recall(vec, 1)`
   filtered to `role === 'user'` — never the async Ollama embed. New bridge
   `__chat_ghost(prefix)` following the `__chat_complete` precedent (:1078 —
   the C readline already calls into JS mid-input), returning a suffix string
   or `''`. Threshold: cosine ≥ 0.9 (tunable), prefix ≥ 8 chars, debounce
   250–300ms, exclude same-session near-dupes.
3. Render: dim ghost suffix inside the input box (new TUI affordance; reuse
   overlay redraw discipline — never leave stale rows). Tab or → accepts;
   any other key dismisses.
4. Config gate `"ghost": true|false` (default off for v1) — it's the most
   "surprising" feature here.

**Verify:** pty — session A: long distinctive prompt; session B: retype its
first 12 chars → ghost suggests the tail; Tab accepts; threshold rejects
low-similarity noise; brain off → no callbacks fire.

---

## F11 — `sofuu serve --brain <path>` — the brain over HTTP (Tier 3)

1. `main.rs`: new CLI op `serve` with flags `--brain <path>` (default
   `~/.sofuu_brain.qtsq`), `--port 7707`, `--host 127.0.0.1` (localhost
   default — remote binding requires `--host` + `--token` explicitly),
   `--token <t>` (else generated once, printed, stored `chmod 600`).
2. Server is a shipped JS file (same ship pattern as `src/js/think.js` —
   embed via `include_str!` or add to `lib/`; decide at impl) run through the
   normal runtime:
   - `GET  /health` → `{ok, memories}`
   - `POST /remember {text, role?, tier?}` → embed (Ollama → embedLocal) +
     store + flush
   - `GET  /recall?q=…&k=…` (or `POST /recall {vector}` for raw vectors)
   - `POST /share` (later) — export card over the wire
   Auth: `Authorization: Bearer <token>`; single brain handle, serialize through
   one mutex-ish request queue (the CMA is not thread-safe — the HTTP server
   is single-looped so this falls out naturally).
3. This is the desktop/IDE integration point — `sofuu-desktop` can point at
   `http://127.0.0.1:7707` with the token instead of opening the file itself.

**Verify:** curl round-trip remember/recall from the shell; two concurrent
`sofuu chat` instances see shared memory via the server; bad token → 401;
no-token remote bind → refuses to start.

---

## Cross-cutting requirements

**Adding a slash command — the 5-touch checklist** (every feature here):
1. `ALL_COMMANDS` entry (`chat.rs:186`) — feeds TAB completion + did-you-mean.
2. `handle_slash` arm (`chat.rs:238`) — Rust-side arg validation, op token.
3. Driver op dispatch (~:1750+) — the actual behavior.
4. `print_help` line + README's slash-command list.
5. Unit test in chat.rs's test block (:1922+) — op-token contract.

**Testing bar per feature:** scripted pty test (join the existing battery:
welcome panel, pickers, Esc, Ctrl-C, history recall), `cargo test` green,
`make size-check` ≤ 5MB, and the feature's Verify section above. Update
TASKS.md ("parked feature ideas" → done) and README as items ship.

**Sequencing note (updated 2026-08-16):** the old "do F4b/F6/F7 before M8"
warning is obsolete — `mod_ai.c` is deleted; the target file is
`crates/sofuu-core/src/rt/ai.rs` (and the chat driver lives in
`crates/sofuu-core/src/chat.rs`, now running the shared `sofuu.agent.run`
loop). All chat-side work (F1–F3, F5, F8–F10) is in Rust/driver already and
is conflict-free.

---

## Build order & rough effort

> **All items ✅ LANDED 2026-08-19.** F4a/F4b were already done (2026-08-16).

| # | Feature | Status | Effort |
|---|---|---|---|
| F1 | `/remember` + `/why` | ✅ done | ~½ day |
| F2 | `/resume` | ✅ done | ~1 day |
| F3 | `@file` | ✅ done | ~1 day |
| F4a | routing fix | ✅ done (2026-08-16) | ~½ day |
| F4b | streaming tool calls | ✅ done (2026-08-16) | ~2–3 days |
| F5 | `/share` | ✅ done | ~1–2 days |
| F6 | `/cost` + budgets | ✅ done | ~1 day |
| F7 | `/verify` | ✅ done | ~1 day |
| F8 | `/watch` | ✅ done | ~1 day |
| F9 | hooks.js | ✅ done | ~½ day |
| F10 | ghost completion | ✅ done | ~2 days |
| F11 | `serve --brain` | ✅ done | ~1–2 days |

**Total ≈ 12–15 working days.** F1–F4 is the "make the chat feel like the
product it claims to be" batch; F5+ are differentiation.

## Risks

| Risk | Mitigation |
|---|---|
| Streaming `tool_calls` semantics differ per provider (Anthropic block events vs OpenAI index fragments) | Per-provider fixtures from day one; keep the `complete()`-planning fallback flag for one release. |
| Driver JS string keeps growing (already ~2010-line file) | Accept for this batch; if F11 lands, split `DRIVER` into `src/js/chat_*.js` modules as its own cleanup. |
| `/resume` of 300-event-capped sessions misleads | Detect cap, print "partial history", offer `/compact` after resume. |
| Brain-card passwords: weak UX kills the feature | Double-prompt + `SOFUU_CARD_PASSWORD` env override; loud wrong-password error (QTSQ decrypt failure message must be human). |
| `/watch` on huge trees burns CPU | 2k-file cap, skip-lists, 1s tick coalescing. |
| Ghost completion feels like autocorrect when wrong | Default off; high threshold; only `role:'user'` prompts. |
| hooks.js wedges the loop | 3-strike disable + 3s timeout + full docs on the trust model. |
