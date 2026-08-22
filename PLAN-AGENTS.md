# PLAN — Agents & Sub-Agents (with RLM utilization)

> **STATUS (2026-08-20, third pass): ✅ A1–A6, A7, A8, A9 LANDED AND VERIFIED ·
> A4 full form + F4b + all tracked follow-ups landed.**
> First pass: shipped runtime JS (`src/js/agent.js`) + `src/js/web.js`,
> eval'd per engine context by the `shipped.rs` seam; chat runs the same
> loop; verified by `examples/agent_test.js` + `examples/rlm_mock_test.js`.
> Second pass landed the previously-open items:
> - **A4 full form** — the agent tool whitelist is now injected INTO the RLM
>   sandbox: 8th native `__hx_tool` (new `\0RLM_TOOL_SUSPEND:` sentinel),
>   bootstrap `tool()`/`toolBatch()` with a host-injected `__TOOL_CACHE`
>   (same suspension model as llm), `EpisodeAction::ResolveTool` +
>   `feed_tool_results` (maxToolCalls budget, anti-loop, trace), the
>   `__rlm_feed_tools` host fn, and `agent.js` wiring (toolset resolved
>   before the RLM gate; execTool runs through `execOneTool` — timeout,
>   trace, budget accounting — so tool code never runs inside the sandbox;
>   `delegate` excluded, agents-inside-RLM stays `recurseVia`).
> - **F4b** — Anthropic streamed `tool_use` accumulation in `ai.stream`
>   (content_block + input_json_delta → the shared setToolCalls seam; usage
>   input/output-token mapping incl. message_start nesting); every provider
>   runs the stream-first loop now. Bonus fix: the no-`[DONE]` ensure-done
>   path froze usage at {0,0}.
> - **A8.3 completed** — chat turns offer `~/.sofuu/agents/*.js` via the
>   `delegate` tool (maxDepth:1), so the `⏺ agent ← task` line is reachable
>   from chat.
> - `examples/web_test.js` (39-check engine-level web battery), chat
>   verification (piped smoke + pty `/agents` + delegation line + make
>   test), `benchmarks/agent_bench.js` A9.3 (agent-layer overhead ~0.1–0.2
>   ms/step, run-twice registry check).
> Verification: `cargo test --release` 158 green; `examples/agent_test.js`
> **53 checks ALL PASSED** (now incl. A4 sandbox tools + F4b over an
> Anthropic-wire SSE mock and two real MCP child servers);
> `examples/web_test.js` 39/39; `examples/rlm_mock_test.js` green;
> `make test` green; size 1.9MB ≤ 5MB. **Still open:** live-provider agent
> run (mocks only, by design for now).

> **Scope:** first-class agents and sub-agents for Sofuu, headless-first,
> RLM-integrated. An *agent* = named definition (system, tools, model,
> memory scope, budget). A *sub-agent* = an agent another agent may delegate
> to. Delegation follows the **agents-as-tools** model: the parent sees a
> `delegate` tool, the LLM decides when to call it, the child's answer is the
> tool result.
>
> *Written 2026-08-12. Composes with, and does not duplicate:*
> - `PLAN-HEADLESS.md` — **A1 subsumes H3** (`sofuu.agent.run` down-level);
>   A7 uses the H1 funnel (`sofuu_rt_call` + cancel handles). If this plan
>   executes first, close H3 as folded-in.
> - `PLAN-RLM.md` — A4 wires agents ↔ RLM both directions (R1/R2 provide
>   `sofuu.rlm.query`, R3 the router). A4 tolerates RLM being absent
>   (graceful fallbacks) so the two tracks can ship in any order.
> - `PLAN-CHAT-FEATURES.md` — **F4a (multi-server MCP routing) is a hard
>   prerequisite** for correct multi-server agents; F4b (streaming tool
>   calls) is optional UX polish the agent loop inherits automatically.
> - `PLAN-RUST-MIGRATION.md` — **zero new handwritten C**: agents ship as a
>   runtime JS module (`src/js/agent.js`, the `think.js` pattern) over the
>   existing primitives (`ai.complete/stream`, `mcp`, `memory`, `rlm`).
>
> **Today's starting point (verified):** a single-level agent loop exists
> only inside the chat driver (`chat.rs:1273–1302`, cap 8 tool rounds);
> `think.js` has a ReAct loop + reasoning-pattern recall; MCP tools merge
> from `~/.sofuu/mcp.json` (`callMcpTool` first-server-wins — F4a fixes);
> CMA has entity support (`sofuu_cma_remember_entity`, `entities`,
> `mark_positive`); the session mesh already links concurrent *chat*
> sessions.

---

## Design principles (frozen)

1. **Agents-as-tools:** no bespoke orchestration DSL. Delegation is a tool
   call; composition is the model's job.
2. **Runtime-level, headless-first:** `agent.js` registers for every engine
   context at init (like `sofuu.memory`), not just chat. Chat becomes a
   renderer over it.
3. **One loop, structured concurrency:** sub-agents run interleaved on the
   same single-threaded libuv loop — their waits are network/tool I/O, so
   parallelism = `Promise` pools with a concurrency cap. No threads.
4. **Budgets and cancellation are not optional:** depth caps, cycle guard,
   per-run token/wall/step budgets, and a cancel tree (cancel parent ⇒
   children die) are part of the v1 API, not add-ons.
5. **Own memory per agent, sharing on purpose:** memory scopes `shared` /
   `agent` (private namespace) / `off`; cross-agent learning flows through
   the CMA, never through globals.
6. **No new C.** Everything here is `src/js/agent.js` + at most small Rust
   plumbing (cancel handles); primitives already exist.

---

## A1 — Core agent runtime (`src/js/agent.js`) — ~1.5 days

Subsumes PLAN-HEADLESS H3. The chat driver's hand-rolled loop (`chat.rs:1273+`)
migrates to call this; one implementation, every consumer.

1. **API:**
   ```js
   sofuu.agent.define({
     name: "researcher",                 // required, unique per registry
     system: "You research precisely.",
     tools: [                            // inline tool specs OR mcp sets:
       { name, description, parameters }, // JSON-schema (MCP-shaped, like toolSpecs())
       { mcp: [{ name:"fs", command:"npx @modelcontextprotocol/server-filesystem /tmp" }] },
     ],
     provider, model, effort,            // default: caller's session config
     memory: "shared" | "agent" | "off", // default "agent"
     brainPath,                          // override; default from config root
     budget: { maxSteps: 12, maxDepth: 2, maxTokens: 200_000, maxWallMs: 300_000 },
     agents: ["librarian"],              // sub-agents visible via delegate (A2)
   }) → AgentHandle

   sofuu.agent.run(handleOrName, task, opts) → Promise<AgentResult>
   // opts: { context?, history?, signal(cancelId), onStep(evt), plain?:true }
   ```
2. **`AgentResult`:**
   ```js
   { answer, steps, subRuns: [AgentResult], // tree (A2)
     usage: { promptTokens, completionTokens, llmCalls, toolCalls, wallMs },
     trace: [ {t, kind:"plan|tool|delegate|answer", payload} ],
     stopped: null | "budget_steps" | "budget_tokens" | "budget_wall" | "cancelled" }
   ```
3. **The loop** (port of the driver loop, generalized): messages = system +
   optional `opts.context` + history + task → per step: `ai.complete` (with
   tools; F4b lands → use streaming when available) → if `toolCalls`,
   execute (A6 tool routing; results appended as `tool` role messages) →
   else produce final answer (streamed when a consumer passed `onStep` —
   chat keeps its spinner UX via events).
4. **Recall augmentation:** when `memory != "off"`, before planning, recall
   top-k from the scoped brain (port of `recallContext`, with score capture —
   `/why` parity) into the system message; after the answer, store the turn +
   `markPositive` on used hits (port of `storeTurn`, :1241). One
   implementation; chat stops owning this logic.
5. **Chat migration:** driver's `turn()` calls `sofuu.agent.run` with
   `onStep` rendering the existing UX (spinner, tool lines, footer tokens).
   Driver keeps: input handling, slash ops, pickers, `/why`, history trim,
   session logging. **Parity gate:** the existing agent-loop E2E battery
   (mock LLM + real sofuu MCP server) must pass unchanged.

**Verify:** `sofuu run examples/agent_test.js` (headless, no chat): an agent
with one inline tool answers via a mock provider; result tree/usage fields
populated; recall augmentation stores + recalls via CMA; chat pty battery
green after migration.

## A2 — Sub-agents: delegation + parallelism — ~2 days

1. **`delegate` tool injection:** if a definition has non-empty `agents`,
   the planner's tool list gains:
   ```js
   { name: "delegate",
     description: "Hand a well-scoped subtask to a specialist agent.",
     parameters: { type:"object", properties:{
       agent:{type:"string", enum: allowedSubAgentNames},
       task:{type:"string"}, why:{type:"string"} }, required:["agent","task"] } }
   ```
   Executing it = `sofuu.agent.run(childDef, task, {depth+1, parentRunId})`;
   the child's `answer` (plus a compact usage line) becomes the tool result.
2. **Safety rails (v1, non-negotiable):**
   - `budget.maxDepth` (default 2) — a run at max depth gets no `delegate`
     tool at all (the model *can't* recurse).
   - **Cycle guard:** run context carries the delegation chain
     (`["orchestrator","researcher"]`); delegating to an ancestor name fails
     with a tool error the model can read.
   - Child gets **its own** definition's tools — never an implicit superset
     of the parent's (least-privilege by default; sharing is explicit).
3. **Parallel fan-out:**
   ```js
   sofuu.agent.runMany([{agent, task, opts}…], { concurrency: 4 }) → AgentResult[]
   ```
   A tiny promise pool; interleaved on the single loop. Default concurrency
   4; configurable; **in-flight usage aggregates to the parent run** so
   budgets see the whole tree.
4. **Isolation choice (documented):** v1 = same JSContext, separate run
   state closures; `agent.js` keeps zero mutable module-level state besides
   the (immutable-once-defined) registry. v2 option: run untrusted tool-
   code agents inside the PLAN-RLM sandbox (`rlm/sandbox.rs`) — noted here so
   the shape doesn't preclude it; not built now.

**Verify:** deterministic mock-provider fixtures — parent emits
`delegate` call → child fixture answers → assert trace tree matches;
parallel: 4 slow children, assert max in-flight ≤ cap and usage summed;
depth: at maxDepth the child receives no delegate tool (assert in trace);
cycle: A→B→A yields a tool error surfaced to the parent, not a hang.

## A3 — Memory scopes & agent identity — ~1 day

1. **Scope implementation** over the existing CMA (no CMA changes):
   - `shared`: the session/default brain path — current behavior.
   - `agent`: records written with `sofuu_cma_remember_entity`
     (`ffi_exports.rs:487`) with `entity = "agent:"+name`; recall at run
     start filters to this entity + role-agnostic top-k. Enables per-agent
     private long-term learning inside one physical brain file.
   - `off`: no recall, no store (tool-deterministic agents stay clean).
2. **Identity record (optional nicety):** `define` may include
   `identity: {role, expertise}` persisted as an entity record — later
   sessions can ask "what agents exist and what are they good at" via
   `sofuu_cma_entities` (export exists, :449).
3. **KV hints hook (Track-B ready, no Track-B work):** when
   `sofuu_cma_kv_hints` becomes KV-bridged, agent recall gains KV pages with
   zero API change — one comment + a test stub.

**Verify:** two agents, one brain file — private scopes don't cross-appear
in recall; `shared` agent sees both; `off` agent writes nothing
(`count()` unchanged); entity listing returns identities.

## A4 — Agents ↔ RLM, both directions — ~1.5 days

Requires PLAN-RLM R1/R2 (`sofuu.rlm.query`) *or* its absence → fallbacks.

1. **Agents use RLM (automatic):** inside a run, the effective context
   (task + `opts.context`) is token-estimated (`ai.estimateTokens`). If the
   R3 router says `Rlm`, the turn routes to `sofuu.rlm.query` **with the
   agent's tool whitelist injected into the sandbox** — the RLM driver
   program gains tool access per agent definition (same schema; sandbox
   executes them via host callback through the normal tool path). Model/key
   = the agent's own.
2. **RLM uses agents (recursion target):** `sofuu.rlm.query(..., { recurseVia:
   { agent: "librarian" } })` — the sandbox's `llm()` call becomes
   `agent.run(librarian, subPrompt)` instead of a raw completion, so
   recursive steps can use tools and private memory.
3. **Explicit map-reduce helper** (the headliner):
   ```js
   sofuu.agent.mapContext(context, task, {
     agent: "librarian", chunkChars: 16_000, concurrency: 4, reduceAgent? })
   → AgentResult /* answer = merged; subRuns = per-chunk runs */
   ```
   Chunking reuses the RLM chunker (paragraph-preferred, 200-char overlap);
   each chunk → one sub-agent run (`task` + chunk); a reduce turn merges the
   sub-answers. This is the "1M-token document Q&A with tool-using workers"
   demo.
4. **Trace merge:** RLM events (`chunk/grep/peek/llm` from R1's trace) fold
   into the agent run's `trace` with `kind:"rlm:*"` — one explainable tree.

**Verify:** 400KB needle corpus via `mapContext` — sub-runs stay in window,
merged answer finds the needle, sub-runs count == expected chunks;
agent-with-tool inside RLM sandbox: mock tool called during recursion;
RLM-absent fallback: same test asserts plain path works and `trace` notes
the fallback.

## A5 — Budgets, cancellation, timeouts — ~1 day

1. **Budget enforcement points** in the loop: before each planning call
   (`maxTokens`, tree-summed), each wall check (`maxWallMs`), each step
   (`maxSteps`): breach → stop with result marked `stopped:*`, returning the
   best partial answer. Children inherit *remaining* parent budget on
   delegate (parent 200k, parent used 80k ⇒ child cap 120k unless its own is
   lower).
2. **Cancel tree:** `sofuu.agent.run` returns/accepts a `cancelId`;
   `sofuu.agent.cancel(id)` aborts: own in-flight provider call (via the
   existing stream `abort()` seam) + all live children recursively.
   Headless: the H1 `sofuu_rt_cancel` funnels here.
3. **Per-tool timeout:** tool executions get `toolTimeoutMs` (default 30s);
   timeout → tool error into the loop (same as current driver behavior on
   MCP failure), never a stuck run.

**Verify:** token cap stops a scripted loop at the right call count;
cancel mid-child terminates both (mock provider sees connection drop);
tool-timeout covered by a hanging mock tool.

## A6 — Tool routing prereq + MCP correctness — ~0.5 day

- **Execute F4a first** (`PLAN-CHAT-FEATURES.md`): `callMcpTool`
  first-server-wins is dangerously wrong under multiple agents (two agents,
  two servers, same tool name = silent cross-calls). The fix is the
  name→owner map; agent.js consumes the shared resolver.
- Agent tool namespaces: inline tools take precedence; on name clash with
  MCP, inline wins + `onStep` warning event.

**Verify:** F4a's two-mock-server scenario + an agent with a clashing inline
tool (inline wins, warning in trace).

## A7 — Headless surface — ~0.5 day ✅ DONE (2026-08-20)

Landed after PLAN-HEADLESS H1 (the C ABI) shipped. Agents are now fully
reachable from C with streaming events, cancellation, and per-process config.

1. Funnel methods: `agent.define`, `agent.run` (streamed via
   `sofuu_rt_call_stream` events: `plan/tool/delegate/answer_delta/done`),
   `agent.cancel`, `agent.list`.
2. `examples/headless/c_agent.c`: define one agent with one inline tool,
   run a task vs mock provider, print result JSON + tree depth.
3. Agent definitions may be supplied **per-process** in `sofuu_rt_new`
   config — no filesystem needed; `~/.sofuu/agents/*.js` is the convenience
   layer, never the only way.

**Verify:** CI compiles + runs `c_agent.c`; stream events arrive ordered;
cancel event surfaces.

## A8 — Config, CLI, chat hooks — ~1 day

1. **Config dir:** `config_root/agents/*.js` — each file exports a definition
   object (uses the runtime's own module system; sandboxed no more than any
   user script — documented trust model). Invalid file → listed as broken
   with its error, never fatal.
2. **CLI:** `sofuu agent run <name> "task"` (headless one-shot, JSON
   `--json` flag) + `sofuu agent list` — new dispatch op in `main.rs`
   (Rust, trivial).
3. **Chat:** `/agents` lists registry (name, model, memory scope, last-24h
   run count); the driver gains nothing else — when the model delegates,
   the existing tool-line UX renders `agent−researcher(…)` automatically via
   `onStep`. (`@agent` mentions parked.) **UPDATE 2026-08-21:** `@agent`
   mentions LANDED in chat — `@name <task>` / `@agent:name <task>` runs the
   loaded agent directly (own definition, Esc-cancellable via
   `signal:'chat'`); agents win over same-named files, unknown names fall
   through to `@file`. E2E: `tests/chat_mention_e2e.sh`.
4. **think.js relationship (no rebuild):** its reasoning-pattern recall
   becomes optional agent middleware (`define({middleware:["reasoningRecall"]})`)
   in a later pass; noted so the two thinking systems converge, diverge no
   further.

**Verify:** `sofuu agent run` headless E2E vs mock; `/agents` pty render;
broken agent file listed with error, chat unaffected.

## A9 — Observability: run trees — ~1 day (partially in A1)

1. Every run returns the full trace tree (A1/A2 fields) + a compact
   `renderTrace(result)` text/JSON summary (spans indented by depth with
   per-span tokens/wall/tools).
2. Optional `logs: true` on run opts → appends one JSON line per run to
   `config_root/logs/agent_runs.jsonl` (5MB rotation) — dogfood source,
   later feeds routing/cost analysis alongside `routing_log.jsonl`.
3. `benchmarks/agent_bench.js` extended: agents-vs-single-loop numbers so
   regressions in step latency are measurable.

**Verify:** tree render for a 2-depth run matches golden sample; log line
valid JSON; bench runs.

---

## Sequencing & effort

| # | Item | Effort | Depends on |
|---|---|---|---|
| A6 | F4a routing fix + namespace rules | 0.5 day | — (**do first**) |
| A1 | Core agent runtime + chat migration | 1.5 day | A6 |
| A2 | Delegation + parallelism + rails | 2 days | A1 |
| A3 | Memory scopes + identity | 1 day | A1 |
| A4 | Agents↔RLM wiring + mapContext | 1.5 days | A1 (soft: PLAN-RLM R1/R2) |
| A5 | Budgets/cancel/timeout | 1 day | A2 |
| A9 | Run-tree observability + logs | 1 day | A1 |
| A8 | Config dir + CLI verb + `/agents` | 1 day | A1 |
| A7 | Headless funnel + C sample | 0.5 day | A1, PLAN-HEADLESS H1 |

**Total ≈ 9–10 working days.** Suggested order: A6 → A1 → A2 → A5 → A3 →
A4 → A9 → A8 → A7. A4 is the only external dependency and degrades
gracefully if RLM hasn't shipped.

## Risks

| Risk | Mitigation |
|---|---|
| Small/weak models delegate badly or loop | Delegate tool injected only when `agents` is set; few-shot text in the synthetic tool description; floor-model note in docs (same standard as RLM); cycle guard + depth cap make loops structurally impossible. |
| Cost blowup via parallel sub-agents | Tree-summed budgets enforced before every planning call (A5); default concurrency 4; usage visible in every tool result line. |
| One-loop starvation with many agents | Concurrency cap; document tools-must-be-async etiquette (sync CPU tools starve everyone — same rule as Node). |
| Cross-agent memory leaks (wrong scope) | A3 tests; recall filter is definition-derived, not model-controlled; `off` agents provably write nothing. |
| Registry/global pollution across runs | Immutable defs after `define`; zero mutable module state except registry + id counters; `agent_bench` run-twice test. |
| Headless consumers need no-fs operation | Agents definable in init config (A7); fs config dir is optional sugar. |
| Divergence between chat and headless behavior | One loop (A1 mandate); chat parity gate; C sample mirrors the chat scenario in CI. |

## Done-state definition

- `sofuu.agent.define/run/runMany/mapContext/cancel` work headless
  (`sofuu run`, no chat), with tests over deterministic mock providers.
- Parent → sub-agent delegation demonstrated end-to-end with budget
  inheritance, depth cap, cycle guard, and a surviving trace tree.
- A long-context task (≥400KB) completed by parallel tool-using sub-agents
  via `mapContext`; RLM engaged automatically past the router threshold
  when present, clean fallback trace when absent.
- Memory scopes proven (shared/agent/off) on one brain file.
- Chat runs the same loop with pty battery green; `/agents` lists
  definitions; `sofuu agent run <name>` works from a shell.
- Headless funnel exposes agent methods with cancel; `c_agent.c` green in
  CI. Zero new C files added for any of it.
