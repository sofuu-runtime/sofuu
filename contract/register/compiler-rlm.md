# compiler-rlm — Compiler bootstrap, `sofuu script` runner and the RLM driver

Scope of this worker: `crates/sofuu-core/src/compiler.rs`,
`crates/sofuu-core/src/script_cmd.rs`, `crates/sofuu-core/src/rlm/**`.
Division D05 (Kaze/compiler) plus the RLM rows.

Working tree state when this work started: already dirty with the earlier
swarm's remediation change (compiler, script runner, Kaze host, RLM episode
restructuring). Nothing was reverted, reset, committed or pushed. No model
weights were downloaded (`AGENTS.md`); the weights this worker names were
already on disk.

## Checks performed

| Check | How | Result |
|---|---|---|
| The Rust edits compile | `cargo check -p sofuu-core --lib --offline` | **pass** — no errors; the 52 warnings are pre-existing and in other files (`domains/infer/model.rs`, `modules/process.rs`, `rt/ai.rs`). |
| The new unit tests compile | `cargo check -p sofuu-core --lib --tests --offline`, repeated after `touch` on both edited files | **pass** — no errors after a forced recompile of `compiler.rs` and `script_cmd.rs`. One earlier invocation of the same command printed a transient `E0277` and `could not compile (lib test)`; it did not reproduce across eight subsequent runs (including a forced recompile), and it was concurrent with the coordinator's release build on the same cargo target directory. Treated as a build race, not a defect in this change; the coordinator should re-run the suite. |
| The generated bootstrap is valid JavaScript | The bootstrap string was extracted from `compiler.rs` (raw-string body, `format!` brace unescaping, placeholder substitution) and run through `node --check` (node v22.23.2) | **pass** — `SYNTAX OK`. |
| The generated bootstrap behaves as intended | The extracted bootstrap was executed in a Node `vm` sandbox with a recording `sofuu.kaze.run`/`runAsync`, a fake `process` (argv/env/exit/stderr) and a fake `sofuu.fs`; 22 assertions over the option resolution | **pass** — 22/22 (see the table below). Harness: `/tmp/bootstrap_harness.js`, extractor `/tmp/extract_bootstrap.py`, extracted bootstrap `/tmp/bootstrap_check.js` (all outside the repository; the same assertions are now encoded as Rust unit tests in `compiler.rs`). |
| `--no-policy` is a deny-all policy, not "no policy" | Harness: `--no-policy` → `opts.policy == {"tools":{"deny":["kaze.*"]}}`, no `policyJson` key | **pass** |
| A malformed interval fails closed | Harness: `--every abc`, `--every=0`, `--every 10x` → exit 1, `E_KAZE_ARGS`, no run | **pass** |
| A valid interval still works | Harness: `--every 500ms` → no error, supervisor banner carries `500ms` | **pass** |
| `--sim` cannot carry a non-physical response | Harness: `--sim ai=hello`, `--sim tool=ok`, `AGENT_SIM=memory=42`, config `sim:{alert:…}` → exit 1, `E_CAP_INVALID`, no run | **pass** |
| A physical `--sim` key still travels | Harness: `--sim battery=42` → `opts.sim.battery == "42"` | **pass** |
| The backend configuration travels the hierarchy | Harness: `--ai` > `AGENT_AI` > config-file `ai`; `--ai not-json` → exit 1, `E_CONFIG` | **pass** |
| Two policy forms in one configuration file are refused | Harness: config with `policy` **and** `policyJson` → exit 1, `E_CAP_DENIED` | **pass** |
| A bare grant carries no backend | Harness: `--cap drone` → `opts.caps.drone === true` (granted, no device); `--cap drone=` → exit 1, `E_CAP_INVALID` | **pass** |
| The compiler holds no domain parameter or endpoint | `rg` over `crates/sofuu-core/src/compiler.rs` for `entry_price\|sentiment\|volume_24h\|heart_rate\|altitude\|ticker\|binance\|coinbase\|yahoo\|weather\|api.openai\|api.anthropic` | **pass** — no matches. The only capability names in the bootstrap are the generic families (`ai`, `tool`, `memory`, `alert`, `policy`, `log`, `drone`, `rover`). |
| The deny-all document is refused by the runtime, not merely absent | Read `domains/robotics/hardware.rs:195-221` (`authorise`: `policy.ok_or_else(…)` → `E_CAP_DENIED`), `rt/policy.rs:346` (`validate_policy` accepts `{"tools":{"deny":[…]}}`), `rt/policy.rs:592-596` (`tools.deny` is consulted before `tools.allow`; `glob_match("kaze.*", "kaze.drone.arm")` is true), `rt/policy.rs:467` (`glob_match`) | **pass** (read) |
| The source path reports a missing parameter as structured guidance, not a crash | Read `domains/kaze/mod.rs:1035-1077` (`bind_args` → `E_KAZE_PARAM` + `missing_param_message` naming the three tiers) | **pass** (read; that file is outside this worker's scope) |
| A malformed option is not swallowed by the Rust option parser | Read `domains/kaze/mod.rs:1382-1560` (`opts_from_json` returns `Err` for malformed `caps`/`policy`/`policyJson`/`pins`/`sim`) | **pass** (read) |
| `--sim` keys are validated by the simulator itself | Read `domains/kaze/caps.rs:202-246` (`SimState::apply_overrides`: every key outside the modelled set is `E_CAP_INVALID`) | **pass** (read) |
| The RLM unit tests are pure state-machine tests over a fixed reply script | Read `rlm/episode.rs` (`mod tests` header at :615-620, `full_episode_flow_over_a_fixed_reply_script` at :721, `tool_suspension_without_an_executor_stops_the_episode` at :983) and `rlm/js_api.rs` (`mod tests` header at :634-640) | **pass** — the replies are fixed inputs to a deterministic transition function; nothing in the module executes a model. `Episode::run` is used only from tests (`rg '\.run\(&mut'` → test call sites only). |
| The RLM real-model coverage lives in one `#[ignore]`d module driven by a local GGUF | Read `rlm/mod.rs:14-19` (`#[cfg(test)] mod real_model;`) and `rlm/real_model.rs` (`#:279`, `#:325` both `#[ignore = "…needs an existing local GGUF…"]`; `resolve_model_path` reads `SOFUU_RLM_TEST_MODEL` → `SOFUU_CONTRACT_MODEL` → `~/Downloads/<name>` and rejects `sofuu-fixture.gguf` / `ggml-vocab-*` as parser fixtures) | **pass** (read) |
| The local weights the real-model test would use exist here | `ls -la ~/Downloads/*.gguf` | **pass** — `SmolLM2-360M-Instruct-Q8_0.gguf` (386 MB, first candidate in `real_model.rs:31`), plus `Llama-3.2-1B-Instruct-Q8_0.gguf`, `Qwen-3-0.6B-…Q4_K_M.gguf`, `Phi-4-mini-reasoning-UD-Q6_K_XL.gguf`, `K2-Horizon-4B-Q8_0.gguf`. Nothing was downloaded. |
| The RLM tree contains no simulated provider | `rg -i 'mock\|fake\|stub\|canned\|fabricat\|placeholder\|simulat'` over `crates/sofuu-core/src/rlm/` | **pass** — only the sandbox's QuickJS `Date.now` stub and the two deliberate "no placeholder result" comments. |
| The RLM router is a pure computation | Read `rlm/router.rs` (whole file: a token-ratio + holistic-word heuristic and a rotating JSONL decision log) | **pass** — deterministic, no external effect. |

## Gaps found and fixed

| Gap | Fix | Where |
|---|---|---|
| `--no-policy` produced **no policy** (`None`), so the refusal depended on "a missing document happens to be read as a refusal" in `hardware::authorise`. A different reader of the same option could have treated the absence as "no restrictions". | `--no-policy` now yields an explicit deny-all document (`{"tools":{"deny":["kaze.*"]}}`) on both surfaces. The runtime still refuses every capability call (`E_CAP_DENIED`, `R7.6/tools.deny`), and the refusal is now a rule the evaluator applies rather than a missing input. | `crates/sofuu-core/src/script_cmd.rs:473-486` (`deny_all_policy`, `policy_layer`); `crates/sofuu-core/src/compiler.rs:1640-1646` (`denyAllPolicy`), `compiler.rs:1833` (`--no-policy` arm) |
| A malformed supervisor interval was silently replaced by the 5 s default: `--every abc`, `--every 0`, `--every 10x` all ran every 5 s without a word. | `parseInterval` now fails with `E_KAZE_ARGS` on a value that is not a positive duration; the CLI, the configuration file and the environment all propagate the refusal instead of defaulting. Unit suffixes (`ms`/`s`/`m`/`h`/`d`) and the plain-number form keep their previous meaning. | `crates/sofuu-core/src/compiler.rs:1683-1709` (`parseInterval`, `durationAmount`); callers at `compiler.rs:1785-1799`, `compiler.rs:2085-2092`, `compiler.rs:2186-2194` |
| The compiled agent's `--sim` accepted a non-physical key and only failed later, inside the runtime, with "unknown key". A `--sim ai=…` was one step away from looking like a response override. | The bootstrap refuses a `--sim` key that names a non-physical capability family at parse time (`E_CAP_INVALID`, "a simulated … result is not permitted"), for the command line, the environment and the configuration file alike. | `crates/sofuu-core/src/compiler.rs:1648-1663` (`nonPhysicalSimKeys`, `physicalSimOnly`); `compiler.rs:1851`, `compiler.rs:1857`, `compiler.rs:2097`, `compiler.rs:2234` |
| A configuration file could carry `policy` **and** `policyJson`/`policyPath`; the bootstrap silently preferred one, so an operator's guard could be the half that was dropped. | The three forms are mutually exclusive in the configuration file, exactly as the runtime already refuses "policy or policyJson, not both". | `crates/sofuu-core/src/compiler.rs:2061-2068` |
| The backend configuration (`ai`, `mcp`) was reachable from the configuration file only: the environment tier could not set it, and `AGENT_AI` was silently taken as a *routine argument* named `ai`. | `ai` and `mcp` are now reserved option names in the environment tier as well. `AGENT_AI`/`SOFUU_AI` and `AGENT_MCP`/`SOFUU_MCP` carry a JSON object and beat the configuration file; `--ai`/`--mcp` (JSON) beat the environment. A value that is not a JSON object is an `E_CONFIG` error, not an ignored value. | `crates/sofuu-core/src/compiler.rs:1555-1557` (`envKeys`), `compiler.rs:1665-1681` (`jsonObjectOption`), `compiler.rs:1860-1884` (CLI arms), `compiler.rs:2240-2253` (env tier) |
| `script_cmd.rs` carried a second, hand-maintained list of the simulator's state keys (`PHYSICAL_SIM_KEYS`) that had already drifted from the simulator's own contract (it listed `temperature`/`humidity`, which the simulator does not model). Two lists of the same thing in two files. | The duplicated list is gone. The simulator owns the list of the keys it models (`SimState::apply_overrides`); the CLI keeps only the rule it must state itself — the capability families `--sim` may never answer for — because that rule needs a message the simulator does not have. | `crates/sofuu-core/src/script_cmd.rs:514-521` (`NON_PHYSICAL_SIM_KEYS`), `script_cmd.rs:528-560` (`sim_state`) |
| The RLM episode's convenience driver fed a placeholder string back as if a tool had run (`"[rlm: tool not available on this driver]"`), which a transcript could not distinguish from a real tool result. | The episode stops with an explicit `tool_unavailable` outcome instead of resuming with a placeholder. | `crates/sofuu-core/src/rlm/episode.rs:361-392` (`Episode::run`), `episode.rs:541-577` (`done_result`); test at `episode.rs:983` |

## Register rows

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D05 Kaze and compiler | The compiled-agent bootstrap carries the backend behind each grant, the policy, the pins and the validated configuration to the Kaze host | `real` | `crates/sofuu-core/src/compiler.rs:2207-2223` (`runOptions`) → `opts.caps` (backend per family), `opts.policy`/`opts.policyJson`, `opts.pins`, `opts.fuel`/`opts.entry`/`opts.timeoutMs`/`opts.heapBytes`/`opts.ticks`/`opts.tickMs`, `opts.sim`, `opts.ai`/`opts.mcp`. Verified by executing the generated bootstrap in a Node sandbox (22/22 assertions) and by the Rust unit tests `a_compiled_agent_carries_the_backend_policy_and_configuration` and `a_compiled_agent_carries_the_backend_configuration_through_the_hierarchy` (`compiler.rs:2832`, `compiler.rs:3140`). |
| D05 Kaze and compiler | `--no-policy` as an authorisation bypass | `real` (absent) | Both surfaces now install an explicit deny-all policy: `script_cmd.rs:473-486` and `compiler.rs:1640-1646`/`compiler.rs:1833`. The runtime refuses every capability call with `E_CAP_DENIED` (`domains/robotics/hardware.rs:195-221` + `rt/policy.rs:592-596`). Evidence: Node harness (`--no-policy` → `{"tools":{"deny":["kaze.*"]}}`, no `policyJson`) and the Rust tests `no_policy_is_a_deny_all_policy_and_nothing_reaches_a_backend` (`script_cmd.rs:1423`) and `a_no_policy_bootstrap_carries_a_deny_all_policy` (`compiler.rs:3072`). |
| D05 Kaze and compiler | `--sim` carrying an AI, tool, memory, alert, policy or log result | `real` (absent) | `script_cmd.rs:521`/`script_cmd.rs:539` and `compiler.rs:1653`/`compiler.rs:1655` refuse the non-physical families at parse time (`E_CAP_INVALID`); the simulator itself refuses any key it does not model (`domains/kaze/caps.rs:202-246`). Evidence: Node harness (4 refusal cases) + `a_simulator_cannot_answer_for_ai_or_tools` (`script_cmd.rs:1516`) and `a_sim_bootstrap_refuses_a_non_physical_response` (`compiler.rs:3116`). |
| D05 Kaze and compiler | A bare or empty `--cap` meaning simulation | `real` (absent) | A bare grant carries no backend (`compiler.rs:1851` region → `opts.caps.drone === true`; `script_cmd.rs:343-358`) and an empty backend is refused (`E_CAP_INVALID`); the simulator needs an explicit `drone=sim`/`rover=sim` selection (`script_cmd.rs:544-552`, `domains/kaze/mod.rs:1524-1528`). Evidence: Node harness + `a_bare_grant_never_selects_a_simulator` (`script_cmd.rs:1478`). |
| D05 Kaze and compiler | A malformed option that is silently ignored | `real` | The supervisor interval now fails instead of defaulting (`compiler.rs:1683-1709`); two policy forms in one configuration file are refused (`compiler.rs:2061-2068`); a backend configuration that is not a JSON object is `E_CONFIG` (`compiler.rs:1665-1681`); the Rust option parser already returns `Err` for malformed `caps`/`policy`/`pins`/`sim` (`domains/kaze/mod.rs:1382-1560`). Evidence: Node harness (7 refusal cases) + `a_malformed_interval_fails_closed_instead_of_defaulting` (`compiler.rs:3094`) and the pre-existing `a_malformed_option_fails_closed_instead_of_being_ignored` (`compiler.rs:2916`). |
| D05 Kaze and compiler | The 12-factor configuration hierarchy of a compiled agent (CLI > environment > configuration file > routine signature) | `real` | Every option resolves through the same three tiers (`compiler.rs:1584-1590` `claim`, `compiler.rs:1592-1600` `grantCap`); the environment tier now also carries `ai`/`mcp` (`compiler.rs:2240-2253`) and the command line can override them (`compiler.rs:1860-1884`). A missing routine parameter is reported as structured guidance by the bootstrap (`compiler.rs:2284-2312`) and as `E_KAZE_PARAM` by the source path (`domains/kaze/mod.rs:1035-1077`) — never a low-level arithmetic crash. Evidence: Node harness (CLI > env > file for `ai`) + `a_compiled_agent_keeps_the_configuration_hierarchy` (`compiler.rs:2866`) and `a_compiled_agent_carries_the_backend_configuration_through_the_hierarchy` (`compiler.rs:3140`). |
| D05 Kaze and compiler | The compiler and the runtime bootstrap carry no domain parameter, default or external endpoint | `pure` | Static check over `crates/sofuu-core/src/compiler.rs`: no `entry_price`/`sentiment`/`volume_24h`/`heart_rate`/`altitude`/`ticker`/provider URL or API name. The only names the bootstrap contains are generic capability families (`ai`, `tool`, `memory`, `alert`, `policy`, `log`, `drone`, `rover`) and the operator's own values. No domain parameter is hardcoded in `script_cmd.rs` either after the simulator-key list was removed. |
| D03 agents (RLM) | The RLM episode state machine, the sandbox and the wire protocol | `pure` | Deterministic in-process computation over fixed reply scripts — the replies are *inputs* to the transition function, never a simulated provider. `rlm/episode.rs:615-620`, `rlm/js_api.rs:634-640`; `full_episode_flow_over_a_fixed_reply_script` (`episode.rs:721`); a tool suspension without an executor stops the episode with `tool_unavailable` instead of resuming with a placeholder (`episode.rs:983`). `Episode::run` has no non-test caller. |
| D03 agents (RLM) | The RLM routing heuristic and its decision log | `pure` | `rlm/router.rs` — a token-ratio + holistic-word heuristic and a rotating JSONL log. Deterministic, no external effect. |
| D03 agents (RLM) | The RLM episode loop driven by a real local model | `unverified` | The coverage exists and is `#[ignore]`d by default (`rlm/mod.rs:14-19`, `rlm/real_model.rs:279`, `rlm/real_model.rs:325`) and drives the shipped inference engine over an existing local GGUF (`~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf`, present on this machine; no download). It was **not executed in this worker**: `cargo test` is owned by the coordinator. Command for the coordinator: `cargo test --offline -p sofuu-core --lib -- rlm::real_model -- --ignored`. |
| D03 agents (RLM) | The RLM code-execution protocol against small local models | `blocked` | Unchanged from `contract/CAPABILITY-REGISTER.md`: `SmolLM2-360M`, `Llama-3.2-1B`, `Qwen3-0.6B` and `Qwen3-8B` do not follow the snippet protocol. A failure is reported as a failure; nothing is fabricated to make the loop look successful. Closing it needs a weight that emits the protocol, which may not be downloaded. |

## Blocked or unverified

- **The Rust test suite was not executed.** Build/test ownership sits with the
  coordinator (`cargo test` / `make` are out of this worker's hands). What was
  run: `cargo check -p sofuu-core --lib --offline` and
  `cargo check -p sofuu-core --lib --tests --offline` — both compile the
  changed code and the new tests without errors. The behavioural assertions
  were executed against the *generated bootstrap itself* in a Node sandbox
  (22/22); the Rust tests encode the same assertions and still need the
  coordinator's run.
- **The RLM real-model test was not executed** (same reason). It is
  `#[ignore]`d by design and needs `cargo test … -- --ignored` plus a local
  GGUF. The weights are present; the run is outstanding.
- **The one-shot compiled-agent artifact was not built and run.** Building the
  standalone binary is the coordinator's step; the bootstrap's option
  resolution was verified by executing the generated JavaScript, not by
  running a compiled artifact.
- **A transient `E0277` during one `cargo check --lib --tests` invocation**
  could not be reproduced in eight later runs, including a forced recompile of
  both edited files. It was concurrent with the coordinator's release build on
  the same cargo target directory. If the coordinator's suite hits it, it is
  most likely a build race, not this change.

## Notes for the coordinator

- **Files changed by this worker** (all inside the assigned scope):
  - `crates/sofuu-core/src/compiler.rs` — bootstrap: deny-all `--no-policy`,
    failing `--every`/`--interval`, `--sim` non-physical refusal, mutually
    exclusive policy forms in the configuration file, `ai`/`mcp` in the
    environment and on the command line, plus five new unit tests.
  - `crates/sofuu-core/src/script_cmd.rs` — deny-all `--no-policy`, one
    simulator-key list, plus two new unit tests and an updated
    `bridge_options` comment.
  - `crates/sofuu-core/src/rlm/**` — read and verified only; no edit was
    needed in this worker (the episode/tool-placement fix and the
    `real_model.rs` split were already in the working tree).
- **No shared file was edited**: `Makefile`, `scripts/preflight.sh`,
  `scripts/text_gate.py`, `contract/CAPABILITY-REGISTER.md`, `docs/**`, every
  `Cargo.toml` and every `src/js/*.bc`/`*.bc.gz` are untouched. The register
  rows above are for the coordinator to merge into
  `contract/CAPABILITY-REGISTER.md`.
- **`src/js/kaze_async.js` — cross-file ordering (needs a decision).** The
  JS coordinator applies `options.ai` *over* the runtime's `SOFUU_AI_*`
  environment channel (`src/js/kaze_async.js:243-250`), so a configuration
  file's `ai.model` shadows `SOFUU_AI_MODEL`. The compiler now offers the
  artifact's own env tier (`AGENT_AI`/`SOFUU_AI`) and CLI tier (`--ai`),
  which sit above the file, but the runtime's own channel is below the
  artifact's explicit configuration. Making `SOFUU_AI_MODEL` outrank the
  configuration file would have to happen in the JS layer — it is the only
  place that knows the runtime's environment variable names — and the
  compiler deliberately does not duplicate them. Flagged, not changed
  (out of scope).
- **`src/js/rlm.js:297`** has a stale comment ("a fast mock (or a fast
  provider round)"). The code there is the real provider path
  (`sofuu.ai.complete` / `sofuu.ai.stream`); only the word is stale. One-word
  fix, but the file is outside this worker's scope.
- **`tests/rlm_latch_test.js` and `tests/mock_llm_server.js`** are scripted
  HTTP servers used by the chat/RLM E2E harnesses. They are R5 material, not
  D05; this worker did not touch them. `tests/rlm_mock_test.js` (named in the
  plan's R08) no longer exists.
- **The simulator's `--sim` key list now lives in one place**
  (`SimState::apply_overrides`, `crates/sofuu-core/src/domains/kaze/caps.rs:202`).
  If the simulator ever models another key, that is the only list to change.
