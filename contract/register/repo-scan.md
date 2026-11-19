# repo-scan — whole-repository sweep for prohibited substitutes (R0 read-only audit)

Scope of this worker: **read-only**. No source file, test, script, artifact or
document was modified. The only file written is this report. Everything below
is a *lead* with a classification, never a claim of absence: a string search is
a lead, and every lead reported here was opened and read before it was
classified.

Audit window: 2026-11-16, working tree dirty, HEAD `853d0132`-era. The tree
was **changing underneath this audit** — see "Gaps found and fixed", where five
leads were closed by concurrent workers while the sweep was running.

## Checks performed

| Check | Command / source read | Result | Evidence |
|---|---|---|---|
| Label sweep over **tracked** sources for `sim`, `mock`, `fake`, `stub`, `placeholder`, `canned`, `dummy`, `fixture` | `git ls-files -z \| xargs -0 grep -l -i -E '…'` over 1363 tracked files, binary extensions excluded | pass (leads triaged) | 221 tracked files matched; 200 of them are vendored (`deps/libuv`, `deps/quickjs`), binary/icon assets, or prose. The 21 in first-party source are classified in "Register rows". |
| Label sweep over **Git-ignored** sources | `Grep` tool with `include_ignored=true` over `crates/`, `src/js/`, `scripts/`, `tests/`, `examples/`, `bench/`, `benchmarks/`, `bindings/`, `sofuu-android/`, `sofuu-ios/`, `sofuu-macos/`, `sofuu-desktop/`, `docs/`, `local-only/`, `npm/`, `new-model-arch./`, `emb-m/`, `landing-page-2/`, `docs-page-2/` | pass (leads triaged) | `tests/` (`.gitignore:146`), `examples/` (`:147`), `bench/` (`:140`), `benchmarks/` (`:141`), `docs/` (`:164`), `landing-page-2/` (`:168`), `.mimosa/` (`:86`) are ignored; all were swept. |
| Rust runtime core: no non-physical path where `ai`/`tool`/`memory`/`alert` succeeds without a configured backend | read `crates/sofuu-core/src/domains/kaze/caps.rs:426-530` (`RuntimeHost::call`) | pass | `ai`, `tool`, `memory` fall through to the `_` arm → `E_CAP_MISSING: embedded Kaze has no available backend for {family}.{op}` (`caps.rs:512`); `alert.send` requires a non-empty configured sink (`caps.rs:502-503`). No `ai_responses` / `tool_responses` / `ai_reply` / `tool_exec` symbol remains in the tree (`grep` over `crates/`: 0 hits). |
| Physical simulator is explicit and family-scoped | read `crates/sofuu-core/src/domains/robotics/hardware.rs:109-120` (`validate_grant`), `caps.rs:455-470` | pass | `sim` is accepted only for `drone`/`rover`; an unselected family returns `E_CAP_MISSING: no backend selected for {family}.{op}; drone/rover simulation is available for tests with an explicit sim grant` (`caps.rs:468`). |
| Live telemetry never invents a reading | read `crates/sofuu-core/src/domains/robotics/kaze_live.rs:71-91` (`DroneTelemetry::read`), `:92-99` (`require_fresh`), `:120-140` (`apply`) | pass | An unknown field → `E_CAP_MISSING`; a field not observed within the 1500 ms window → `E_KAZE_LINK: drone {field} is missing or stale`. `DroneTelemetry::default()` is only reachable through a poisoned mutex (`kaze_live.rs:316`), and its `observed` map is empty, so the default can never satisfy `require_fresh`. No `0`/`false` fallback path was found on the live path. |
| Physical simulator telemetry never invents a reading | read `caps.rs:584-605` (`SimHost::telemetry`) | pass | An unmodelled field returns `E_CAP_MISSING: the simulated {family} has no '{field}' reading`. |
| No "mock" provider in either provider registry | read `crates/sofuu-core/src/rt/ai.rs:60-190` (`Provider` enum / dispatch), `src/js/providers.js:1043-1096` | pass | The Rust `Provider` enum is `OpenAi / Anthropic / Custom / Local`; the JS `PROVIDERS` map registers `openai`, `lmstudio`, `vllm` and plugin-registered adapters. No `mock`/`sim`/`fake` provider exists. |
| No substitute provider left in the tracked **contract** harness | `grep -i 'mock\|canned\|scripted'` over `scripts/contract/harness/*.js`, cross-checked against `scripts/contract/cases.tsv` and the blocked list in `scripts/contract/run_contract_tests.sh:210-212` | pass | `t22_cancel_stream.js`, `t23_cancel_tool.js`, `t24_disconnect.js` now drive the real local engine (`t22:5-11`, `t23:6-9`, `t24:6-11`). `t25_backpressure.js:6-8` and `t26_terminal_fuzz.js:5-9` were migrated to the real local model during this audit (0 `mock` hits remaining). `t41_session_roundtrip.js:31,36` uses `"model": "mock"` **inside a synthetic session document** — static data for a serialization round-trip, not an executed provider. `t52_serve_v1.js:154` uses the phrase to assert the *negative* (an unconfigured model must be refused). |
| No substitute provider left in the tracked **event-loop benchmark** | read `scripts/bench/event_loop_latency.js:1-40` | pass | Requires `SOFUU_LOOP_MODEL` naming an existing GGUF; a missing model exits 2 (`blocked`), never a substitute. `make event-loop-check` cannot run without it. |
| **Bytecode driver snapshots** contain no substitute provider | `strings src/js/*.bc` over all 12 snapshots; `gunzip -c src/js/serve.bc.gz \| cmp - src/js/serve.bc` | pass | 0 `mock`/`MOCK-LLM`/`OK: tool`/`Synthesized autonomous` hits in any `.bc`; `serve.bc.gz` inflates byte-identical to `serve.bc`. |
| **Shipped release artifacts under `dist/`** inspected separately from sources | `strings` + `sofuu capabilities` on every binary and library in `dist/` (611 MB, 40+ artifacts) | **fail** | `dist/sofuu-darwin-arm64` and `dist/sofuu-linux-x86_64` (both 2026-11-08) carry the **pre-remediation fabricated** strings `OK: tool '{name}' executed ({arg})` and `Synthesized autonomous decision for: "{brief}"`. `dist/sofuu-darwin-arm64 capabilities` reports `build.revision = 15ad2e0be-dirty`; `15ad2e0be` (2026-11-08 22:54) is an ancestor of `db090b419` "fix(kaze): enforce fail-closed embedded capabilities" (2026-11-11 12:28) — `git merge-base --is-ancestor 15ad2e0be db090b419` → true. The current working-tree build reports `853d01325-dirty` and has **neither** string. See Lead D1. |
| `dist/` embedded libraries carry the Kaze capability host at all | `strings` for `E_CAP_MISSING` / `E_KAZE_LINK` / `ai.prompt` over `dist/libsofuu.{a,so,dylib}`, `dist/libsofuu-android-*.so`, `dist/libsofuu-ios-*.a` | **fail** | All are 2026-10-28…10-30 builds: 0 hits for `E_CAP_MISSING`, `E_KAZE_LINK` and `ai.prompt`. They predate the Kaze capability feature entirely, so the published embedded SDK cannot do what the current README describes. See Lead D2. |
| `dist/` binaries that carry the capability host are all pre-remediation | `strings` for the pre-fix wording `the host did not grant a backend for` vs the post-fix `no backend selected for` | **fail** | `dist/sofuu-linux-arm64` (2026-11-05) and `dist/kaggle/sofuu-linux-x86_64-gnu` (2026-11-07) carry the **pre-fix** wording only; `dist/sofuu-darwin-arm64` and `dist/sofuu-linux-x86_64` carry the pre-fix wording *and* the fabricated replies. |
| `scripts/verify-dist.sh` detects a stale `dist/` artifact | read `scripts/verify-dist.sh:169-176` (the `dist/sofuu-*` loop) and `:178-200` (the revision stamp check) | **fail** | The loop over `$DIST/sofuu-*` runs **only** `check_paths` (absolute build-host path leak, `:150-160`). The revision stamp is read from `$BIN` alone (`:180`). The comment at `:169-171` claims "a stale one in `dist/` is reported by name so it can be rebuilt", but no staleness comparison exists. See Lead D3. |
| The tracked **local-only JS corpus** types every case by what it may claim | `tests/run_js_tests.sh` re-read at 13:20 (the file was rewritten during this audit) | pass (after a concurrent fix) | The registry is now `file\|description\|kind\|reason` with `kind ∈ {real, scripted, model}`; `registry_kind()` (`:255-279`) rejects an unknown kind or a `scripted`/`model` entry without a reason; `run_entry` (`:300-330`) reports `scripted` as `⊘ BLOCKED` and skips `model` cases when `MODEL_STATE != ok`. Re-verified: 74 entries, 42 `real` / 26 `scripted` / 6 `model`; **no** entry with a mock-provider signature is left untyped except `chat_sessions_e2e.sh`, which makes no model request (its provider is `http://127.0.0.1:9/…`, unreachable, and the case asserts "Session has no turns to resume"). |
| No substitute provider in `bench/` / `benchmarks/` | read `benchmarks/agent_bench.js:14,264` | pass (after a concurrent fix) | The mock provider and its artificial 8 ms latency were removed during this audit; the benchmark now states that "earlier figures from this benchmark's mock provider measured a canned response" and refuses to compare against them. |
| `examples/` and the headless C samples | read `examples/headless/README.md:3-4`, `examples/headless/c_agent.c:3`, `examples/headless/c_llm.c:9`, `examples/headless/demo_config.h:4` | pass | "Every sample here drives the real runtime. None of them substitutes a mock provider, a scripted model or a canned answer (RULES.md §1.9)". No `mock`/`fake`/`canned` token in any example source. |
| `example-agents/` public claims | read `example-agents/README.md:8-33`, `example-agents/local_brief/README.md:4-95`, `example-agents/quant_sentinel/README.md:23`, `example-agents/med_protocol/README.md:22` | pass | The agents state their real dependency and that a missing tool fails closed with `E_CAP_MISSING (E_MCP_TOOL_NOT_FOUND)`; `paper_trader` is retired. No claim of a verified integration that was not exercised. |
| Bindings, mobile and desktop surfaces | `grep -i 'mock\|simulat\|fake\|canned\|dummy'` over `bindings/`, `sofuu-android/app/src/`, `sofuu-ios/SofuuApp/`, `sofuu-macos/SofuuMac/`, `sofuu-desktop/src/`, `sofuu-desktop/src-tauri/src/` | pass | No substitute provider. The desktop hits are React input `placeholder=` props; the mobile hits are the honest comment "unavailable, rather than a fake zero vector" (`SofuuBridge.kt:172`, `jni_bridge.c:424`, `SofuuBridge.swift:106`, `:122`). `sofuu-android/.../network_security_config.xml:1-4` allows cleartext to `10.0.2.2` "so the turn smoke test can reach a local mock" — a dev-only emulator allowance, not a runtime path. |
| Landing content | `grep` over `landing-page-2/src/`, `docs-page-2/src/`, `sofuu-landing/` | pass (with a note) | `landing-page-2/src/components/ProductSections.tsx` renders `styles.mock` product visuals (skeleton bars, "3 files changed", "214 passing", "200 OK"). These are **static illustrations with no execution path**; the page is Git-ignored. See Lead D6. |
| Public claims in `README.md`, `cli/README.md`, `CHANGELOG.md` | read `README.md:28,111,146,347,366`, `cli/README.md:1-35` | pass | The README states "published binaries remain separately unverified"; `cli/README.md:3-8` honestly labels `cli/sofuu-*` as "local build outputs of the 0.2.0-beta line … They predate the Kaze engine". No false claim found. |
| CI workflow | read `.github/workflows/ci.yml:25,88,140,251-264,453` | pass | The CI refuses the synthetic GGUF fixture as inference evidence (`ci.yml:264`, `exit 2`), names the scripted-model RLM tests (`:88`), and regenerates the fixture deterministically (`:257-258`). No mock provider is executed by CI. |
| `contract/` tracked register and fixtures | `grep -i 'mock\|fake\|canned\|simulat'` over `contract/` | pass | Only the register's own description of the removed mock provider (`CAPABILITY-REGISTER.md:49`) and a historical harness lesson (`contract/PROGRESS.md:287`). |
| Ignored **stale source copies** inside the runtime tree | `find . -name '*.source'` + `git check-ignore` | pass with a note | 15,134 `.source` files live under `.mimosa/hook-state/`: 15,000 in the repo-root `.mimosa/`, 117 in `sofuu-desktop/`, **10 in `crates/sofuu-core/src/`**, 5 in `crates/ml-train/`, 2 in `deps/quickjs/`. The `.mimosa/` directories total 94 MB, 90 MB of it in the repo root. All are Git-ignored (`.gitignore:86`), none is tracked, **none** carries the pre-remediation `OK: tool` marker, and none ends in `.rs` so Cargo cannot pick it up. The ten that sit inside `crates/sofuu-core/src/` do inflate every repo-wide text sweep of the runtime tree. See Lead D5. |
| `docs/` (ignored) still contains superseded mock-provider instructions | `grep -rn 'mock LLM\|mock provider\|canned answer\|rlm_mock_test'` over `docs/` | **fail** | `docs/plans/PLAN-AGENTS.md:21,47,149,152,268,292,374`, `docs/plans/PLAN-MEMORY-TOKENS.md:10,189,258,378,493,499`, `docs/plans/PLAN-AGENT-TESTING.md:37,216`, `docs/plans/PLAN-CHAT-FEATURES.md:248,266`, `docs/plans/PLAN-ML-GATES.md:650`, `docs/plans/PLAN-DESKTOP.md:273`, `docs/plans/PLAN-EMBED-SDK.md:335`, `docs/roadmaps/ROADMAP-BRAIN-CMA.md:200`. R5 bullet 5 requires removing superseded mock instructions from active docs and plans. See Lead D7. |
| `local-only/`, `npm/`, `new-model-arch./`, `emb-m/`, `patch/`, `tools/` | `grep` over each | pass | No substitute provider. `local-only/tests-removed-from-repo.rs.txt` is a local copy of removed GPU-kernel tests (0 mock hits). `new-model-arch./spf_v1/tests/test_spf_credit.c:186` fakes a firing in a research credit-assignment unit test, unrelated to the runtime. `emb-m/` is a standalone research project that nothing in the runtime references. |
| `crates/sofuu-core/src/rt/ai.rs:7983` doc comment | read the surrounding test | **fail** (stale text) | "/// (Transfer paths are covered by `tests/voice_test.js` against a mock.)" — `tests/voice_test.js` is now an explicitly-labelled **protocol fixture** that reports its live half as `BLOCKED` (`tests/voice_test.js:13-27`). The comment is stale and understates the file. See Lead D4. |

## Gaps found and fixed

This worker edited **no** source file. Five gaps found by this sweep were
closed by concurrent workers *while the sweep was running*; they are listed
here with the closing evidence so the register is accurate about who changed
what. The remaining gaps are **open** and need a change outside this worker's
read-only scope.

| Gap | Fix | Path:line |
|---|---|---|
| The local-only JS corpus counted 14 mock-driven cases as `PASSED`: `tests/run_js_tests.sh` tagged only 17 of its entries `scripted`, so `agent_test.js`, `long_horizon_test.js`, `rlm_latch_test.js`, `auto_continue_test.js`, `chat_transcript_e2e.sh`, `chat_theme_e2e.sh`, `chat_mode_no_wipe_e2e.sh`, `chat_compact_role_e2e.sh`, `chat_select_e2e.sh`, `chat_nothink_failover_e2e.sh`, `chat_approval_timer_e2e.sh`, `chat_state_copy_e2e.sh`, `chat_secrets_guard_e2e.sh` and `web_test.js` ran against a canned OpenAI-compatible endpoint and were counted as passes | **closed by a concurrent worker** — the registry became `file\|description\|kind\|reason`; every substitute-driven entry is now `kind=scripted` and is reported `⊘ BLOCKED: … (coverage gap)`, `kind=model` entries are skipped when no local GGUF is present, and `registry_kind()` rejects an unknown kind or a missing reason | `tests/run_js_tests.sh:59-63` (registry format), `:255-279` (`registry_kind`), `:300-330` (`run_entry` accounting) |
| `scripts/contract/harness/t25_backpressure.js` flooded a **mock** SSE server with synthetic frames and `t26_terminal_fuzz.js` served a **mock** server that fabricated a `tool_calls` frame for `broken_tool` | **closed by a concurrent worker** — t25 now drives the real local engine and reaches the 1024-chunk bound with real model output; t26 drives the real model against the real file tool and lets a genuinely missing file produce the failure | `scripts/contract/harness/t25_backpressure.js:6-18`, `scripts/contract/harness/t26_terminal_fuzz.js:5-15` |
| `scripts/contract/cases.tsv` described T2.2/T2.3 as "mock-provider cancel + usage accounting" and "cancel mid-tool kills the subprocess (mock provider, no orphan)" while the harnesses had already moved to the real local model | **closed by a concurrent worker** — the prereq column now names the real backend and the block condition | `scripts/contract/cases.tsv:32-33` |
| `crates/sofuu-core/src/main.rs:220` CLI help advertised `sofuu plugin test <tool> Test a plugin tool in isolation with mock parameters`, implying a mocked run | **closed by a concurrent worker** — the help now reads "Alias of `plugin run`: execute the tool with the arguments you supply" (the implementation was always a real plugin execution: `main.rs:1686-1704` → `sofuu.plugins.cliTest` → `cliRun`, `src/js/plugins.js:3855-3861`) | `crates/sofuu-core/src/main.rs:220` |
| `benchmarks/agent_bench.js` measured a mock provider with a hard-coded 8 ms artificial per-request latency and printed it as a benchmark figure | **closed by a concurrent worker** — the mock provider and the artificial delay are gone and the benchmark states that its earlier figures measured a canned response and are not comparable | `benchmarks/agent_bench.js:14,264` |

**Open gaps that this worker cannot close (read-only scope) — for the coordinator:**

| Gap | Required change | Where |
|---|---|---|
| **D1** The staged release artifacts under `dist/` were built **before** the fail-closed remediation and still contain the fabricated Kaze replies `OK: tool '{name}' executed ({arg})` and `Synthesized autonomous decision for: "{brief}"` | Rebuild `dist/` from the current tree (`make dist`), or delete/quarantine the stale artifacts so no one uploads them. `dist/sofuu-darwin-arm64.tar.gz` is the exact asset name `.github/workflows/release.yml:100-105` uploads. | `dist/sofuu-darwin-arm64`, `dist/sofuu-linux-x86_64`, `dist/sofuu-darwin-arm64.tar.gz`, `dist/sofuu-linux-x86_64.tar.gz` |
| **D2** The staged embedded libraries in `dist/` (`libsofuu.{a,so,dylib}`, `libsofuu-android-*.so`, `libsofuu-ios-*.a`, 2026-10-28…10-30) predate the Kaze capability feature entirely and carry no `E_CAP_MISSING`/`E_KAZE_LINK`/`ai.prompt` surface at all | Rebuild them from the current tree, or record them as a pre-Kaze lineage that cannot support the D05 capability rows | `dist/libsofuu.dylib`, `dist/libsofuu.so`, `dist/libsofuu.a`, `dist/libsofuu-android-*.so`, `dist/libsofuu-ios-*.a` |
| **D3** `scripts/verify-dist.sh` does **not** detect a stale `dist/` artifact — the loop over `$DIST/sofuu-*` runs only `check_paths` (build-host path leak), and the revision stamp is read from `$BIN` alone, while the comment claims a stale artifact "is reported by name so it can be rebuilt" | Compare each `dist/sofuu-*` artifact's `capabilities.build.revision` against the revision the gate expects, and fail when it differs | `scripts/verify-dist.sh:169-176` (loop), `:178-200` (stamp) |
| **D4** `crates/sofuu-core/src/rt/ai.rs:7983` still describes `tests/voice_test.js` as "against a mock"; the file is an explicitly-labelled protocol fixture that reports its live half as BLOCKED | Update the doc comment to name the fixture and its BLOCKED live coverage | `crates/sofuu-core/src/rt/ai.rs:7983` |
| **D5** 15,134 stale `.source` baseline copies live under `.mimosa/hook-state/` (94 MB of `.mimosa/` in total; 10 of the files sit inside `crates/sofuu-core/src/`, 117 in `sofuu-desktop/`) | Consider moving `.mimosa/` out of the runtime source tree (or adding it to a sweep-exclusion list) so a repository-wide text sweep is not swamped by tool state | `crates/sofuu-core/src/.mimosa/`, `crates/sofuu-core/src/rt/.mimosa/`, `sofuu-desktop/**/.mimosa/`, repo-root `.mimosa/` |
| **D6** `landing-page-2/src/components/ProductSections.tsx` shows "mock UI" product visuals with figures that read like real receipts ("3 files changed", "214 passing", "200 OK") | If the landing page is published, label the visuals as illustrations (or use figures that exist in the register) | `landing-page-2/src/components/ProductSections.tsx:9-42,76-179` |
| **D7** `docs/plans/*` and `docs/roadmaps/*` still carry superseded mock-provider instructions (R5 bullet 5) | Retire or annotate the superseded sections | `docs/plans/PLAN-AGENTS.md:21,47,149,152,268,292,374`; `docs/plans/PLAN-MEMORY-TOKENS.md:10,189,258,378,493,499`; `docs/plans/PLAN-AGENT-TESTING.md:37,216`; `docs/plans/PLAN-CHAT-FEATURES.md:248,266`; `docs/plans/PLAN-ML-GATES.md:650`; `docs/plans/PLAN-DESKTOP.md:273`; `docs/plans/PLAN-EMBED-SDK.md:335`; `docs/roadmaps/ROADMAP-BRAIN-CMA.md:200` |

## Register rows

Division `D11` (public/release) unless stated. Statuses are the ones defined
in `contract/CAPABILITY-REGISTER.md`.

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D11 | Whole-repository label and return-pattern sweep, tracked sources | `real` | Swept all 1363 tracked files for the seven labels plus the return patterns (constant replies, ignored errors, unconditional success, default telemetry, simulator fallback). No prohibited substitute remains in the tracked runtime source: the Rust and JS provider registries have no `mock` provider, the Kaze runtime host fails closed on `ai`/`tool`/`memory`/`alert`, and the tracked contract harnesses drive the real local engine. |
| D11 | Whole-repository sweep, Git-ignored sources | `real` | `tests/`, `examples/`, `bench/`, `benchmarks/`, `docs/`, `landing-page-2/`, `.mimosa/`, `local-only/`, `npm/`, `new-model-arch./`, `emb-m/` were swept. Five substitute-driven leads found; all five were closed by concurrent workers during the sweep (see "Gaps found and fixed"). The remaining ignored-source lead is D7/D8 below. |
| D11 | Release artifacts in `dist/` — the staged CLI binaries and their `.tar.gz` release assets | `unverified` | `dist/sofuu-darwin-arm64` reports `build.revision = 15ad2e0be-dirty` (2026-11-08), an ancestor of the fail-closed fix `db090b419` (2026-11-11). It contains the pre-fix fabricated strings `OK: tool '{name}' executed ({arg})` and `Synthesized autonomous decision for: "{brief}"`. `dist/sofuu-linux-x86_64` (same build window) carries the same strings. **These staged artifacts must not be published.** The CI release workflow rebuilds before uploading (`.github/workflows/release.yml:76-78`), so the published path is not proven to ship them; a manual upload of the staged tarball would. |
| D11 | Release artifacts in `dist/` — the staged embedded libraries | `unverified` | `dist/libsofuu.{a,so,dylib}` and `dist/libsofuu-android-*.so` / `dist/libsofuu-ios-*.a` (2026-10-28…10-30) carry no `E_CAP_MISSING` / `E_KAZE_LINK` / `ai.prompt` surface: they predate the Kaze capability feature. They cannot support the D05 capability rows and are not evidence for any published SDK behaviour. |
| D11 | `scripts/verify-dist.sh` staleness detection | `real` (absent) | The `dist/sofuu-*` loop runs only `check_paths` (build-host path leak); the revision stamp is read from `$BIN` alone. A `dist/` artifact built from an older revision is not detected, even though the comment at `verify-dist.sh:169-171` says a stale one "is reported by name so it can be rebuilt". |
| D11 | `tests/run_js_tests.sh` case typing (local-only corpus) | `real` | The registry now types every case `real` / `scripted` / `model` with a mandatory reason for a gap; `scripted` is reported `⊘ BLOCKED` and `model` is skipped when no local GGUF is present. Re-verified 74 entries (42 real / 26 scripted / 6 model). **Local-only:** `tests/` is Git-ignored (`.gitignore:146`), so a clean checkout has no runner and `scripts/preflight.sh:405-410` reports `BLOCKED js` for it. |
| D11 | Tracked contract conformance suite | `real` | `scripts/contract/cases.tsv` + `run_contract_tests.sh`. The cases that previously ran a substitute (`T2.2`–`T2.6`) are either migrated to the real local model (`t22`, `t23`, `t24`, `t25`, `t26`) or explicitly blocked by the runner (`run_contract_tests.sh:210-212`), and the `cases.tsv` prereq column now names the real backend and its block condition. |
| D11 | `benchmarks/agent_bench.js` | `real` | The mock provider and its artificial 8 ms per-request latency were removed during this audit; the benchmark states that its earlier mock-provider figures measured a canned response and are not comparable. `bench/` and `benchmarks/` are Git-ignored (`.gitignore:140-141`). |
| D03 | Agent loop fallback (canned final answer) | `real` | Confirmed: no `mock` provider in `crates/sofuu-core/src/rt/ai.rs:60-190` or `src/js/providers.js:1043-1096`; the Kaze runtime host returns `E_CAP_MISSING` for `ai`/`tool`/`memory` (`caps.rs:512`). Already covered by the register row at `CAPABILITY-REGISTER.md:49`; re-verified here. |
| D04 | Tool / MCP / policy execution | `real` | No substitute provider or mock tool registry in `crates/sofuu-core/src/rt/mcp.rs`, `src/js/tools.js` or the contract harness. Already covered by `CAPABILITY-REGISTER.md:50-51`; re-verified here. |
| D05 | Kaze host separation, backend resolution, fail-closed policy | `real` | `crates/sofuu-core/src/domains/kaze/caps.rs:426-530`: `ai`/`tool`/`memory` → `E_CAP_MISSING`; `alert.send` requires a configured sink; the physical simulator is family-scoped and explicitly selected. Already covered by `CAPABILITY-REGISTER.md:52-53`; re-verified here. |
| D06 | Inference | `pure` | `src/js/*.bc` driver snapshots (12 files) contain no substitute provider and `serve.bc.gz` inflates byte-identical to `serve.bc`. The synthetic `contract/fixtures/sofuu-fixture.gguf` remains a parser fixture only and CI refuses it as inference evidence (`.github/workflows/ci.yml:264`). Already covered by `CAPABILITY-REGISTER.md:54-55`. |
| D07 | Robotics / drones | `physical-test` | `dist/sofuu-linux-arm64` and `dist/kaggle/sofuu-linux-x86_64-gnu` carry the **pre-fix** `E_CAP_MISSING` wording (`the host did not grant a backend for …`), so the staged artifacts do not have the current fail-closed message. The live telemetry path has no default reading (`kaze_live.rs:71-99`: unknown field → `E_CAP_MISSING`, stale field → `E_KAZE_LINK`). Hardware remains `unverified` (no device attached). |
| D08 | CLI help | `real` | `crates/sofuu-core/src/main.rs:220` no longer advertises "mock parameters" for `sofuu plugin test`; the command is an alias of `plugin run` and executes the plugin for real (`src/js/plugins.js:3855-3861`). One stale doc comment remains at `crates/sofuu-core/src/rt/ai.rs:7983`. |
| D09 | Mobile / native bindings | `real` (host JVM only) | No substitute provider in `sofuu-android/`, `sofuu-ios/`, `sofuu-macos/` or `bindings/`. The bindings report an unavailable embedding as an error, not a fake zero vector (`SofuuBridge.kt:172`, `jni_bridge.c:424`, `SofuuBridge.swift:106,122`). Device coverage stays `unverified`. Already covered by `CAPABILITY-REGISTER.md:59-60`. |
| D10 | Desktop shell | `unverified` | No substitute provider in `sofuu-desktop/src/` or `src-tauri/src/`; the hits are React `placeholder=` props. No device or platform receipt exists for this surface. Already covered by `CAPABILITY-REGISTER.md:61`. |
| D11 | Landing content | `pure` | `landing-page-2/src/components/ProductSections.tsx` renders static product visuals with figures that read like receipts ("3 files changed", "214 passing"). No execution path; the page is Git-ignored. Whether it needs an "illustration" label is a D11 disclosure decision, not an execution gap. |
| D11 | Documentation | `blocked` | `docs/` is Git-ignored (`.gitignore:164`) and superseded mock-provider instructions remain in eight plan/roadmap files (R5 bullet 5). Owner: the coordinator — `docs/**` is outside every worker's editable scope here. |

## Blocked or unverified

- **Published release binaries: `unverified`.** No published artifact was
  inspected — this environment has no access to the GitHub release assets and
  the developer is the sole maintainer. What *is* established is that the
  binaries staged in `dist/` on this machine predate the fail-closed fix
  (receipt: `dist/sofuu-darwin-arm64` → `build.revision 15ad2e0be-dirty`,
  `git merge-base --is-ancestor 15ad2e0be db090b419` → true, plus the two
  fabricated strings in `strings` output). Whether a published release carries
  them is `unverified` and needs either a fresh `make dist` or an inspection
  of the published asset by the maintainer.
- **`dist/sofuu-linux-x86_64` revision: `unverified`.** It cannot be executed
  on this macOS host, so its `capabilities.build.revision` could not be read.
  Its pre-remediation state is inferred from its 2026-11-08 23:14 build
  window and the identical pre-fix strings it carries; the same inference
  applies to `dist/sofuu-linux-arm64` (2026-11-05) and
  `dist/kaggle/sofuu-linux-x86_64-gnu` (2026-11-07), which carry the pre-fix
  `E_CAP_MISSING` wording but not the fabricated replies.
- **Rust execution: `unverified` for this worker.** Build policy forbids
  `cargo build` / `cargo test` here, so every Rust-side statement in this
  report is a **source read**, not an executed result. The coordinator owns
  the build and the suite run.
- **The local-only JS corpus was not executed.** `tests/run_js_tests.sh` was
  read and its accounting logic verified statically; no case was run, so the
  "26 scripted entries are reported BLOCKED" claim is a source-read, not an
  observed run.
- **Hardware: `unverified`.** No drone, rover, GPIO or Android/iOS device was
  attached; nothing here is evidence about physical hardware.
- **Landing page: `unverified`.** `landing-page-2/` is Git-ignored
  (`.gitignore:168`) and was not built or rendered; the finding is a source
  read of the JSX.
- **`docs/` is Git-ignored** (`.gitignore:164`), so the superseded
  mock-provider instructions in the plan files are local-only and are not
  published. They still need retirement under R5 bullet 5, but they are not a
  live public-claim defect.

## Notes for the coordinator

1. **Merge note.** Rows in "Register rows" are proposed additions to
   `contract/CAPABILITY-REGISTER.md`. Two of them contradict the current
   wording of the register and need the coordinator to reconcile them:
   - `CAPABILITY-REGISTER.md:90` says "Published release binaries |
     `unverified` | No published artifact was replaced by this work." That is
     true, but it does not record that the **staged** artifacts in `dist/` are
     pre-remediation. Suggested extra row: "Staged release artifacts in
     `dist/` | `unverified` | built from `15ad2e0be` (pre-`db090b419`); they
     contain the fabricated Kaze tool/AI replies and must not be published."
   - `CAPABILITY-REGISTER.md:88` says contract cases "block with an actionable
     message instead of executing a substitute". That now holds, but it only
     became true during this audit: `t25`/`t26` were still substitute-driven
     when this sweep began and were migrated by a concurrent worker.
2. **`scripts/verify-dist.sh` needs a staleness check (D3).** It is a tracked
   release gate and it currently cannot tell a fresh `dist/` artifact from a
   three-week-old one. The smallest honest fix is to read
   `capabilities.build.revision` from each `$DIST/sofuu-*` and fail when it
   differs from the revision the gate expects. `scripts/verify-dist.sh` is not
   in this worker's editable scope, and it is adjacent to the coordinator's
   `scripts/` ownership.
3. **`dist/` needs a decision (D1, D2).** Either rebuild it from the current
   tree or quarantine it. The release workflow overwrites `dist/` from a fresh
   build before uploading, so the CI path is safe; the risk is a manual
   upload of `dist/sofuu-darwin-arm64.tar.gz` or a developer handing someone
   a staged `dist/` binary. Note also that `dist/sofuu-linux-arm64` and
   `dist/kaggle/sofuu-linux-x86_64-gnu` are a *different*, older lineage that
   carries the pre-fix wording without the fabricated replies — a rebuild has
   to cover every `dist/sofuu-*`, not just the darwin one.
4. **`docs/**` (D7) is outside every worker's editable scope here.** R5
   bullet 5 requires the superseded mock-provider instructions in
   `docs/plans/PLAN-AGENTS.md`, `PLAN-MEMORY-TOKENS.md`, `PLAN-AGENT-TESTING.md`,
   `PLAN-CHAT-FEATURES.md`, `PLAN-ML-GATES.md`, `PLAN-DESKTOP.md`,
   `PLAN-EMBED-SDK.md` and `docs/roadmaps/ROADMAP-BRAIN-CMA.md` to be retired
   or annotated. They are local-only (Git-ignored), so they are not a live
   public-claim defect, but they are the last place in the tree where a reader
   can still find "mock provider" described as a way to test.
5. **The `tests/` corpus is local-only.** The `run_js_tests.sh` typing
   remediation is correct and verified by source read, but `tests/` is
   Git-ignored, so a clean checkout has no runner and the D03/D04 coverage it
   provides is not reproducible from a clean tree. The register's
   release-surface row for `make test` (`CAPABILITY-REGISTER.md:87`) should
   say that the JS corpus behind it is local-only, matching the way
   `scripts/preflight.sh:405-410` reports it.
6. **One stale doc comment left in the Rust tree (D4):**
   `crates/sofuu-core/src/rt/ai.rs:7983` still says the transfer paths are
   covered "against a mock", referring to a file that is now an explicitly
   labelled protocol fixture. Cosmetic, but it is a claim about a substitute
   in a file another worker may be editing.
7. **Sweep hygiene for the next R0 pass (D5).** `.mimosa/hook-state/`
   contributes 15,134 stale `.source` copies (94 MB of `.mimosa/` in total),
   10 of them inside `crates/sofuu-core/src/`. None carries pre-remediation
   runtime code and none can be compiled (the extension is `.source`, not
   `.rs`), but they make any repo-wide text sweep noisy and are a standing
   false-positive source. A sweep exclusion for `.mimosa/` would be worth
   adding to the plan's search procedure.
8. **This worker changed no file except this report.** No `git` mutating
   command was run; nothing was reverted, reset, stashed, cleaned or
   committed.
