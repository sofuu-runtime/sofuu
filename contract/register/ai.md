# ai-providers-chat-scouter — AI providers, agent loop, chat turn and the Scouter health surface (D03, D06, D08)

Worker scope (edits): `crates/sofuu-core/src/rt/ai.rs`,
`crates/sofuu-core/src/rt/scouter.rs`. Read-only for this worker:
`crates/sofuu-core/src/rt/engine.rs`, `crates/sofuu-core/src/chat.rs`,
`src/js/chat_tui.js`, `src/js/chat.js`, `src/js/kaze_async.js`,
`src/js/agent.js`, `src/js/plugins.js`, `crates/sofuu-core/src/main.rs`.

No weights were downloaded, read or deleted. No cargo build/test/make target
was run: `cargo check -p sofuu-core --lib` was the only cargo invocation
(compile evidence only). `contract/CAPABILITY-REGISTER.md` was read, not
edited.

## Checks performed

| # | What was run or read | Result | Evidence |
|---|---|---|---|
| 1 | Provider registry scan for a substitute/canned provider: `grep -rn '"mock"' crates/ src/js/` and a read of `parse_provider` | **pass** — no substitute provider in the Rust registry or the JS registry | `crates/sofuu-core/src/rt/ai.rs:62` (`parse_provider`: openai/anthropic/local/ollama/llamacpp/custom only), `src/js/plugins.js:1925` (`NATIVE_AI_PROVIDERS`, no substitute), `src/js/agent.js` has no `"mock"` provider |
| 2 | Provider response path, end to end (read): JSON error body behind HTTP 200 → failure; empty completion → success; body with no completion → malformed; provider usage used when present | **pass** | `crates/sofuu-core/src/rt/ai.rs:1367` `parse_completion_body` (top-level `error` → `E_AI_PROVIDER`; empty/absent content → empty success; missing `choices`/`content` → `E_AI_MALFORMED`), `:1575` `parse_usage`, `:3580` the complete branch (usage attached only when the provider reported it) |
| 3 | Same check on the embed and audio surfaces (read) | **fail → fixed** — an error body behind a 200 was reported as `E_AI_MALFORMED` ("no usable embedding vector") instead of the provider's own error | fixed at `crates/sofuu-core/src/rt/ai.rs:1454` (`provider_error_in_body`), used at `:3693` (embed) and `:3733` (audio) |
| 4 | Local (in-process) completion fidelity (read) | **fail → fixed** — the model's own answer was replaced by an EMPTY string when it contained an interior NUL (`CString::new(...).unwrap_or_default()`) | fixed at `crates/sofuu-core/src/rt/ai.rs:4640` (`JS_NewStringLen`, the same treatment the HTTP path already used at `:3591`) |
| 5 | Streamed answer fidelity (read) | **fail → fixed** — a chunk carrying an interior NUL was truncated at that byte (`CStr::from_ptr(..).to_bytes()` stops at the first NUL) and the rest of the answer was lost | fixed at `crates/sofuu-core/src/rt/ai.rs:1840` (`js_string_bytes`, length-preserving), `:1914`, `:1982`, `:3473`, `:3489` |
| 6 | Request-body construction (read) | **fail → fixed** — the model name and the message roles were inserted unescaped: a quote produced malformed JSON, and an interior NUL made the whole body copy fail and turned the request into an EMPTY body | fixed at `crates/sofuu-core/src/rt/ai.rs:563` (model), `:650` (role), `:705` (anthropic model), `:1036` (embed model); the unrepresentable body is now a typed failure at `:1066` (`request_body_cstring`), used at `:5178`, `:5492`, `:5818` |
| 7 | Unknown completion (request sent, result lost) and automatic retries (read + `grep -n -i "retry\|replay\|idempot" crates/sofuu-core/src/rt/ai.rs`) | **pass** — a lost result is `E_AI_UNKNOWN_COMPLETION`; the provider layer contains no retry at all (the only "retry" hits are message wording) | `crates/sofuu-core/src/rt/ai.rs:2256` (`ai_transport_failure`), `:3869` (stream truncation), `:5024` (local engine channel closed with no completion signal) |
| 8 | `ai.classify` is a real model call with the label set validated (read) | **pass** — the call goes to the provider; a category outside the caller's label set is rejected with `E_KAZE_RESULT`; no keyword match remains | `src/js/kaze_async.js:254` (the provider call), `:189` `validateClassifyLabels`, `:203` `validateClassifyResult`; exercised by `crates/sofuu-core/src/domains/kaze/mod.rs:2046` |
| 9 | Deadlines, response size bound, cancellation (read) | **partial** — size bound and deadlines present; explicit cancellation present for streams only | bound `crates/sofuu-core/src/rt/ai.rs:2184` (`AI_MAX_RESPONSE` 64 MiB) + the write callbacks + `CURLOPT_MAXFILESIZE`; deadlines `:2292` (30 s connect, caller's `timeout_ms`) and `:2195`/`:2452` (300 s stall watchdog); cancellation `:5325` (`__ai_abort`) for streams and `:4713` (local abort flag). GAP: `ai.complete` has no caller-visible abort handle — recorded below |
| 10 | Scouter health surface (read) | **pass → fixed** — the global score was already null/flagged as unmeasured; but an evaluation with NOTHING to evaluate still published a perfect 100/"A" and marked the global state measured | `crates/sofuu-core/src/rt/scouter.rs:208`/`:214` (context `measured`), `:344`/`:350` (brain `measured`), `:591` (`record_scouter_threats` takes `Option<u32>` and leaves an existing reading alone) |
| 11 | Scouter consumers (read-only files) | **pass** — both consumers already treat a missing score as "not measured"; the CLI health line requires `contextMeasured === true`, so an unmeasured evaluation prints no health figure at all | `src/js/chat_tui.js:293-310` ("unavailable — not measured", and an early return when neither score exists), `crates/sofuu-core/src/main.rs:1241` (`r.scouter.contextMeasured === true && typeof r.scouter.contextScore === 'number'`) |
| 12 | Chat turn engine + agent loop: a canned answer for an unavailable provider/tool/substrate (read) | **pass for the agent loop, gap for the non-TUI chat engine** — the agent loop escalates an empty stream to a typed `E_EMPTY_STREAM` instead of answering, and a failed provider/tool throws; `src/js/chat.js:1904` still substitutes the literal `'(no response)'` for an empty answer and persists it | `src/js/agent.js:6181` (`throw makeError('E_EMPTY_STREAM', …)`), `src/js/chat_tui.js:3967` (the `noResponse` guard that keeps the placeholder out of the transcript); gap reported below |
| 13 | Compile check: `cargo check -p sofuu-core --lib` | **pass** — no errors; 51 warnings, all pre-existing (the one in the edited file, a no-op `.clone()` on a `&str`, was removed) | cargo output: `Finished \`dev\` profile … in 1.92s`, `warning: \`sofuu-core\` (lib) generated 51 warnings` |
| 14 | New unit tests | **unverified** — added and compile-checked, but NOT executed: the coordinator owns `cargo test` (the cargo target directory is busy with the release build) | `crates/sofuu-core/src/rt/ai.rs:9470-9552`, `crates/sofuu-core/src/rt/scouter.rs:937` |
| 15 | Live provider / local-GGUF exercise | **not attempted** — no build, no test run, no weights loaded; a live call would also have needed an API key or a local server this worker may not start | see "Blocked or unverified" |

## Gaps found and fixed

| Gap | Fix | Path:line |
|---|---|---|
| The local model's own answer was replaced by an EMPTY string when it contained an interior NUL — a fabricated completion | `JS_NewStringLen` (byte-wise) instead of `CString::new(..).unwrap_or_default()` | `crates/sofuu-core/src/rt/ai.rs:4640` |
| A streamed answer chunk was truncated at an interior NUL and the rest of the answer was lost | `js_string_bytes` (length-preserving `JS_ToCStringLen2`) + `JS_NewStringLen` at the push sites | `crates/sofuu-core/src/rt/ai.rs:1840`, `:1914`, `:1982`, `:3473`, `:3489` |
| The SSE frames were parsed through a NUL-terminated copy that silently became an empty frame | the frames are parsed with their own length | `crates/sofuu-core/src/rt/ai.rs:1858`, `:1928`, `:2000`, `:3281`, `:3334`, `:3435`, `:3448` |
| The model name and the message roles were inserted into the request body unescaped: a quote produced malformed JSON, an interior NUL produced an EMPTY request body | `json_escape` on the model name and the role; `request_body_cstring` turns an unrepresentable body into a typed `E_AI_MALFORMED` instead of an empty POST | `crates/sofuu-core/src/rt/ai.rs:563`, `:650`, `:705`, `:1036`, `:1066`, `:5178`, `:5492`, `:5818` |
| A provider error carried in the body behind a 200 was reported as "malformed" on the embed and audio surfaces | `provider_error_in_body` → `E_AI_PROVIDER` with the provider's own message | `crates/sofuu-core/src/rt/ai.rs:1454`, `:3693`, `:3733` |
| An evaluation with nothing to evaluate published a perfect 100/"A" and marked the global Scouter state measured | `measured` flag on both reports; the JSON carries `score:null`/`grade:null`/`measured:false`, and an unmeasured pass no longer overwrites a reading | `crates/sofuu-core/src/rt/scouter.rs:208`, `:214`, `:223`, `:328`, `:344`, `:350`, `:359`, `:420`, `:591`, `:700`, `:770` |
| The curl easy handle leaked on the early-return paths of `ai.complete` / `ai.stream` / `ai.embed` | `curl_easy_cleanup` on each early return | `crates/sofuu-core/src/rt/ai.rs:5170`, `:5181`, `:5458`, `:5510`, `:5793`, `:5797`, `:5803`, `:5810` |
| A no-op `.clone()` on a `&str` produced a compiler warning in the edited file | removed | `crates/sofuu-core/src/rt/ai.rs:3812` |

## Register rows

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D03 agents | Agent loop: an unavailable provider, tool or substrate must fail with a typed error, never a plausible-looking answer | `real` | `crates/sofuu-core/src/rt/ai.rs:62` (no substitute provider in the Rust registry); `src/js/plugins.js:1925` (no substitute in the JS registry — a plugin adapter must be registered by the user and rejects with its typed code); `src/js/agent.js:6181` escalates an empty provider stream to `E_EMPTY_STREAM` instead of answering |
| D03 agents | Chat turn engine: an empty agent answer replaced by the literal `'(no response)'` and persisted as the turn's answer | `unverified` | `src/js/chat.js:1904` — `String((res && res.answer) \|\| '(no response)')`, then `sessionLog('answer', answer)` and `S.history.push`. `src/js/chat_tui.js:3967` has the `noResponse` guard that keeps the placeholder out of the transcript; `chat.js` has no equivalent. Outside this worker's edit scope — needs a JS change |
| D03 agents | `ai.classify`: a real model call whose category is validated against the caller's label set | `real` | `src/js/kaze_async.js:254` performs the provider call; `:203` `validateClassifyResult` rejects an out-of-set category with `E_KAZE_RESULT`; no keyword match remains. Exercised by `crates/sofuu-core/src/domains/kaze/mod.rs:2046` |
| D06 inference | Provider response path: an error body behind a 200 is a failure; an empty completion stays an empty success; a body with no completion is malformed; provider-reported usage is used when present | `real` | `crates/sofuu-core/src/rt/ai.rs:1367` (`parse_completion_body`), `:1575` (`parse_usage`), `:1454` + `:3693` + `:3733` (`provider_error_in_body` for embed/audio) |
| D06 inference | Local (in-process) completion returns the model's own answer, never filled in | `real` | `crates/sofuu-core/src/rt/ai.rs:4640` — a real answer containing a NUL is no longer replaced by an empty one |
| D06 inference | A streamed answer is not truncated | `real` | `crates/sofuu-core/src/rt/ai.rs:1840` (`js_string_bytes`), `:3473`, `:3489` |
| D06 inference | Unknown completion (request sent, result lost) is reported as unknown; no automatic replay of a provider request | `real` | `crates/sofuu-core/src/rt/ai.rs:2256`, `:3869`, `:5024`; `grep -n -i "retry\|replay\|idempot"` finds no retry in the provider layer |
| D06 inference | The request body is always a real, well-formed body | `real` | `crates/sofuu-core/src/rt/ai.rs:1066` (`request_body_cstring`), model/role escaping at `:563`, `:650`, `:705`, `:1036` |
| D06 inference | Per-operation deadlines and response size bounds on the provider paths | `real` | `crates/sofuu-core/src/rt/ai.rs:2184` (64 MiB cap + write callbacks + `CURLOPT_MAXFILESIZE`), `:2292` (30 s connect, caller's `timeout_ms`), `:2195`/`:2452` (300 s stall watchdog) |
| D06 inference | Explicit cancellation on the provider paths | `real` (streams) / `unverified` (non-streaming) | `crates/sofuu-core/src/rt/ai.rs:5325` (`__ai_abort` → `stream_abort_one`), `:4713` (the local engine's abort flag). GAP: `ai.complete` exposes no abort handle, so an in-flight non-streaming completion can only be ended by the stall watchdog or the caller's deadline |
| D08 CLI/TUI/REPL | Scouter: an unmeasured score is absent/null and flagged as unmeasured in the TUI and the CLI health line | `real` | `crates/sofuu-core/src/rt/scouter.rs:777` (`contextScore:null`, `contextMeasured:false`); `src/js/chat_tui.js:293-310`; `crates/sofuu-core/src/main.rs:1241` (the health line requires `contextMeasured === true`). An evaluation with nothing to evaluate no longer publishes a score (`scouter.rs:208`, `:328`, `:591`); a real run still measures it — `src/js/agent.js:6298` evaluates the context health on every step |
| D08 CLI/TUI/REPL | The TUI prints the scouter's entropy/efficiency diagnostics of an unmeasured context report as `0.00` / `100.0%` | `unverified` | `src/js/chat_tui.js:310` — `r.max_entropy \|\| 0` / `r.efficiency_ratio \|\| 1`. Only reachable when the brain score exists while the context report is unmeasured (the brain surface has no caller today); the TUI is read-only for this worker |

## Blocked or unverified

- **The new unit tests were not executed.** `crates/sofuu-core/src/rt/ai.rs`
  (`a_js_string_with_an_interior_nul_survives_the_round_trip`,
  `the_request_body_escapes_the_model_name`, `a_representable_body_is_required`,
  `an_error_body_behind_a_200_is_a_provider_failure`) and
  `crates/sofuu-core/src/rt/scouter.rs`
  (`an_unmeasured_evaluation_reports_no_score`) compile under
  `cargo check -p sofuu-core --lib`, but `cargo test` is the coordinator's.
  Their assertions are **unverified**.
- **No live provider call and no local-GGUF inference from this worker.** The
  provider-response-path claims above are a source review plus compile
  evidence, not a receipt. A live receipt needs a configured provider or an
  existing local GGUF (`~/Downloads/*.gguf`) and a built binary.
- **Explicit cancellation of `ai.complete`** is absent (streams only). It is
  **unverified** as a feature and **absent** as an implementation; closing it
  needs a JS-visible abort handle, i.e. a change to the `sofuu.ai.complete`
  contract and to its callers (chat, agent, Kaze bridge).
- **The non-TUI chat engine's empty-answer substitution**
  (`src/js/chat.js:1904`) is **unverified** — outside this worker's edit
  scope and not exercised by any test in this tree.
- No model weights were downloaded, opened or deleted; no weight file was
  used as evidence for any row above.

## Notes for the coordinator

1. **Shared-file change wanted (JS, owner of `src/js/chat.js`):**
   `src/js/chat.js:1904` replaces an empty agent answer with the literal
   `'(no response)'` and then persists it (`sessionLog('answer', answer)`,
   `S.history.push`). An empty provider completion is a genuine empty answer
   and must not be filled in — the TUI already refuses to persist that
   placeholder (`src/js/chat_tui.js:3967`), `chat.js` does not. Mirror the
   guard, and keep the placeholder out of the persisted transcript.
2. **Shared-file change wanted (JS):** `src/js/chat.js:420` and
   `src/js/chat_tui.js:2361` — `complete()` returns `'(no response)'` for an
   empty provider text. Today only the summarizer uses that helper (and
   detects the sentinel), so it is not presented as an answer, but the
   sentinel should not be reachable as an answer value at all.
3. **Shared-file change wanted (JS, read-only here):** `src/js/chat_tui.js:310`
   prints the entropy/zipf/efficiency line of a report whose `measured` flag
   is false (the values are then `0.00` / `100.0%` placeholders). Skip the
   line when `r.measured === false`. The Rust side now emits that flag
   (`crates/sofuu-core/src/rt/scouter.rs:208`).
4. **Shared-file change wanted (Rust, outside this worker's four files):**
   `crates/sofuu-core/src/rt/engine.rs` and `crates/sofuu-core/src/chat.rs`
   were reviewed and needed no change — the REPL now reports a rejected
   promise as an error (`engine.rs:2224`) and the chat config write reports
   a failed save (`chat.rs:574-620`).
5. **`ai.complete` cancellation** (see "Blocked or unverified"): adding it
   needs a JS-visible abort handle; I did not add one because it changes the
   public `sofuu.ai.complete` contract and the chat/agent/Kaze callers.
6. **Coordinator action:** run `cargo test -p sofuu-core` for the AI and
   Scouter suites — my four new tests are compile-checked only. No bytecode
   snapshot (`src/js/*.bc`) changed, so `make bc-check` is unaffected; the
   Rust additions are a few hundred bytes, far below the 5,767,168-byte
   ceiling.
7. **Register rows to merge** into `contract/CAPABILITY-REGISTER.md`
   (not edited by me): the D03 chat-turn empty-answer row, the D06
   provider-response-path / local-answer-fidelity / streamed-chunk-fidelity /
   request-body / cancellation rows, and the D08 Scouter rows from the table
   above.
