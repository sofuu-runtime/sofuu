# cli-doctor-honest-state — CLI and doctor surfaces

Scope of this worker: `crates/sofuu-core/src/main.rs`,
`crates/sofuu-core/src/doctor.rs`, `cli/README.md`. Plan R6, division D08
(CLI/TUI/REPL: honest mode, errors and exit status).

## Checks performed

| Check | Command / source read | Result | Evidence |
|---|---|---|---|
| R6: the CLI health line prints a score only when it was measured | read `crates/sofuu-core/src/main.rs:1241-1247` (the `sofuu agent run` summary) against `crates/sofuu-core/src/rt/scouter.rs:557-568` (`ScouterState.context_measured` / `brain_measured`) and `scouter.rs:773-800` (`js_scouter_status` emits `null` for an unmeasured score) | pass (after this change) | Before: `contextMeasured !== false` accepted any payload that merely lacked the field, and the brain number was printed on `typeof === 'number'` alone. Now the line requires `contextMeasured === true` for the context score and `brainMeasured === true` for the brain score; an unmeasured side prints `not measured` and the whole segment is omitted. |
| R6: `doctor` reports which grant resolves to which backend | read `crates/sofuu-core/src/domains/robotics/hardware.rs:97-146` (`validate_grant`) — the same function that judges `--cap family=backend` at `crates/sofuu-core/src/script_cmd.rs:560` and `crates/sofuu-core/src/domains/kaze/mod.rs:482-497` | pass (after this change) | New `doctor.rs:159-182` (`accepted_backends`) probes the runtime's own validator with one representative of every backend spelling, so the printed table cannot drift from the enforcing code. New section `doctor.rs:1053-1097` prints the accepted names per family, marks `sim` as the explicit physical test, and states that a bare grant resolves to **no** implementation (`E_CAP_MISSING`). `sensor`/`device` are reported `unavailable` — they have no backend at all. |
| R6: `doctor` reports whether the policy is present and non-empty | read `crates/sofuu-core/src/rt/policy.rs:1304-1357` (`sofuu.policy.load()`: a malformed layer installs **nothing** and leaves the previous posture in force) and `policy.rs:969-1013` (`layered_policy_from_paths`) | pass (after this change) | New section `doctor.rs:859-937` reads the two layers the runtime reads (`~/.sofuu/policy.json`, `<cwd>/.sofuu/policy.json` — `policy.rs:1314-1319`), reports presence, size, section count and whether `tools.allow`/`tools.deny` exist (an empty `tools.allow` authorizes nothing — `domains/robotics/hardware.rs:203-216`), and says explicitly that an invalid layer is **not** in force and the default ALLOW posture applies. Before: doctor printed only the installed posture and nothing about the document on disk. |
| R6: `doctor` reports the model endpoint and the weights | read `crates/sofuu-core/src/rt/ai.rs:115-141` (`provider_api_url` — every provider carries a default URL) and `crates/sofuu-core/src/chat.rs:462-486` (the GGUF caps path) | pass (after this change) | `doctor.rs:146-152` (`endpoint_text`) names the provider's default endpoint instead of the old `(no endpoint)`, which read like a missing configuration. `doctor.rs:1032-1049`: a path-like model name is checked wherever it is configured (the old code required `provider == "local"`), a missing file is `unavailable`, a present file is reported as present but not loaded, and a local provider with a model **name** is reported as "cannot tell whether the weights are present" instead of being silently skipped. |
| R6: the Scouter scores are measured-only in `doctor` | read `crates/sofuu-core/src/rt/scouter.rs:518-539` (`context_measured` / `brain_measured` flags) and `scouter.rs:736-745` (`null` for an unmeasured score) | pass (after this change) | New section `doctor.rs:1099-1129` prints the two scores only when the measured flag is set, and the JSON report carries `"contextScore": null` / `"brainScore": null` for an unmeasured score (`doctor.rs:1123-1129`, `doctor.rs:1163`) — never `0`, never `100`. |
| R6: the `serve` banner described the `/health` route wrongly | read `src/js/serve.js:820-842` (`/healthz` and `/livez` are the tokenless liveness routes; `/health` needs the bearer token unless `--public-health`) against the banner at `crates/sofuu-core/src/main.rs:2918-2920` | pass (after this change) | The banner said `GET /health (liveness, no auth)` — false for every run with a token (the token is auto-generated or supplied, `main.rs:2852-2895`). Now it names `/healthz, /livez` as the tokenless routes and says that `/health` needs the bearer token unless `--public-health`. The `sofuu serve --help` text was already correct. |
| R6: the `plugin test` help text described a mock | read `src/js/plugins.js:3855-3861` (`cliTest` resets the tool's circuit breaker and calls `cliRun` — it executes the real tool with the arguments supplied) | pass (after this change) | The top-level help said "Test a plugin tool in isolation with mock parameters" (`main.rs:220`). Nothing is mocked: the mode runs the tool. Now: "Alias of `plugin run`: execute the tool with the arguments you supply". |
| R6: the CLI never prints a success banner for an operation that did not happen | grepped every `✓` / "success" literal in `main.rs` (lines 1128, 2999, 3854, 4485, 4696, 4778) and read the `serve` bind path (`crates/sofuu-core/src/rt/http_server.rs:1239-1284`, `1325-1327`) | pass | Every `✓` follows the operation it reports (a written bundle, a loaded model, a real discovery result). A failed `server.listen()` throws in `js_server_listen`, so `serve.js` never reaches its "brain server listening" line and `cmd_serve` returns non-zero (`main.rs:4349-4354`). No success banner for a bind that did not happen. |
| R6: an unavailable capability exits non-zero with the typed reason | read the `doctor` arm (`main.rs:3420-3433`), the `script` arm (`main.rs:4359-4369`), `crates/sofuu-core/src/script_cmd.rs:44-96` (`Outcome::exit_code`) and the unknown-command arm (`main.rs:4436-4440`) | pass | `doctor` returns 1 on any finding; `script` maps `E_CAP_MISSING`/`E_CAP_INVALID`/`E_CONFIG`/`E_KAZE_RUNTIME` to a non-zero exit with the typed code; an unknown command exits 1. No path returns 0 for an operation that was refused. |
| `cli/README.md` matches the binaries in the folder | ran `cli/sofuu-darwin-arm64 version` → `sofuu 0.2.0-beta` (exit 0); `cli/sofuu-darwin-arm64 doctor` → `Unknown command 'doctor'` (exit 1); read `cli/sofuu-darwin-arm64 --help` | pass (after this change) | The README now lists the modes these binaries actually have (`chat`, `repl`, `run`, `eval`, `bundle`, `install`, `add`, `agent list\|run`, `serve`, `session …`, `outputs …`, `version`, `licenses`, `help`) and names the commands they do **not** have, with the exit status of an unknown command (1). |
| `cli/README.md` — the brain round-trip claim is a real receipt | ran the two documented `sofuu eval` commands in `/tmp/sofuu-cli-check` against `cli/sofuu-darwin-arm64` | pass | First `eval` printed `true` for the flush, the second process printed `1 persisted-value` — the claim holds and is now dated in the README (2026-11-16). The write really needed the explicit flush. |
| `cli/README.md` — the networking claims match the binaries | `otool -L cli/sofuu-darwin-arm64` → links `/usr/lib/libcurl.4.dylib` (**the system library**); `file` on the Linux binaries → `static-pie linked` / `statically linked`; `strings` → `CLIENT libcurl 8.6.0` in both Linux binaries and `CALG_SCHANNEL_ENC_KEY` in the Windows binary; no `OpenSSL` version string is recorded in any of them | pass (after this change) | The README's blanket "statically linked libcurl 8.6.0" was false for macOS. It now states, per platform, what was read from the binaries: macOS links the system libcurl (OS TLS), Linux is static with libcurl 8.6.0 and an OpenSSL TLS backend (the exact OpenSSL version is **not** recorded inside these binaries; the current cross-build uses 3.3.2, `dist/openssl-*-linux-musl`), Windows uses Schannel. |
| `cli/README.md` — the CA-bundle claim matches the binaries | `strings` on all four binaries: no `SOFUU_CA_BUNDLE`, `SOFUU_CA_PATH`, `SSL_CERT_FILE` or `/etc/ssl/certs/ca-certificates.crt` | pass (after this change) | Those overrides and the Linux trust-store auto-probe exist in the **current** runtime (`crates/sofuu-ffi/src/curl.rs:227-275`), not in these 0.2.0-beta binaries. The README now says so and points at the current release. |
| `cli/README.md` — the build instructions match the tree | read `Makefile:490-517` (`zig-install`, `linux`, `linux-x86_64`, `linux-arm64`, `windows-x86_64`), `Makefile:79-87` (the QTSQ guard), `scripts/cross/install_zig.sh:7` (Zig 0.13.0), `scripts/cross/build_qtsq_cross.sh` (produces `dist/qtsq-*/libqtsq.a`) and `ls dist/qtsq-*` | pass | Every target, script and version the README names exists. The QTSQ guard really refuses to build without `SOFUU_QTSQ_DIR`. |
| The edits compile | `cargo check -p sofuu-core --bin sofuu` and `cargo check -p sofuu-core --bin sofuu --tests` | pass | The bin target: `Finished dev profile`, 0 errors, only pre-existing warnings (`crates/sofuu-core/src/chat.rs:3150` deprecated `mach_task_self`, unused `eprint!` macro in `main.rs:79`). The `--tests` variant also passed with the new unit test included when it was run at 13:36; it now fails on `crates/sofuu-core/src/rt/storage.rs:1183` (`SofuQuery` needs `Debug`), a file another worker was editing concurrently — **not** in this worker's scope, and it does not affect the non-test build. |
| The Rust test suite | — | **not run** (build policy: the coordinator owns the build) | The new test `the_backend_table_names_only_real_backends` (`doctor.rs`) is type-checked only. |

## Gaps found and fixed

| Gap | Fix | Path:line |
|---|---|---|
| The `sofuu agent run` health line printed a score from a payload that merely lacked the `contextMeasured` field (`!== false`), and printed the brain number without checking its own measured flag | both numbers now require their measured flag to be `true`; an unmeasured side prints `not measured` and an unmeasured context omits the whole segment | `crates/sofuu-core/src/main.rs:1241-1247` |
| `doctor` never said which grant resolves to which backend, so an operator could not tell which capability families have a backend at all | new `kaze backends` section; the backend names are probed from `hardware::validate_grant`, the enforcing code, so the table cannot drift | `crates/sofuu-core/src/doctor.rs:159-182`, `crates/sofuu-core/src/doctor.rs:1053-1097` |
| `doctor` said nothing about the policy document: not whether it exists, not whether it is non-empty, and not that an invalid layer leaves the default ALLOW posture in force | new `policy` section: per-layer presence and size, merged section count, `tools.allow`/`tools.deny` presence (an empty allowlist authorizes nothing), and the typed reason an invalid layer is not in force | `crates/sofuu-core/src/doctor.rs:859-937` |
| `doctor` printed `(no endpoint)` for a model whose provider simply uses its default URL — that reads as a missing configuration | `endpoint_text()` names the provider's default endpoint; the configured override is printed when there is one | `crates/sofuu-core/src/doctor.rs:146-152` |
| The weights check was skipped for a path-like model that was not configured under the `local` provider, so a missing GGUF was never reported | the check no longer depends on the provider name; a missing file is `unavailable`, a present file is "present, not loaded", and a local provider with a model *name* reports that the presence is unknown | `crates/sofuu-core/src/doctor.rs:1032-1049` |
| `doctor` reported no Scouter state at all, so an unmeasured score was silently absent rather than explicitly unmeasured | new `scouter` section; the scores appear only when measured, and the JSON report carries `null` (never `0`, never `100`) | `crates/sofuu-core/src/doctor.rs:1099-1129`, `crates/sofuu-core/src/doctor.rs:1123-1129` |
| The `serve` banner claimed `GET /health` was a no-auth liveness route; with a token configured (always) it returns 401 | the banner names `/healthz, /livez` as the tokenless routes and says `/health` needs the bearer token unless `--public-health` | `crates/sofuu-core/src/main.rs:2918-2920` |
| The top-level help described `sofuu plugin test` as running "with mock parameters"; it actually executes the real tool | the help text now describes the alias for what it does | `crates/sofuu-core/src/main.rs:220` |
| `cli/README.md` claimed a "statically linked libcurl 8.6.0" for all four binaries; the macOS binary links the system libcurl | per-platform networking facts, each read from the binary itself | `cli/README.md:110-134` |
| `cli/README.md` documented CA-bundle overrides and the Linux trust-store auto-probe for binaries that do not contain them | the paragraph now states that they are in the current runtime (`crates/sofuu-ffi/src/curl.rs`) and not in these 0.2.0-beta binaries | `cli/README.md:130-134` |
| `cli/README.md` did not say which modes the folder's binaries actually have, so "the commands in the repository README do not all exist in them" was open-ended | the status banner now lists the modes these binaries have and the ones they do not, with the exit status of an unknown command | `cli/README.md:9-16` |
| `cli/README.md` documented `sofuu doctor` without its exit-status contract | the contract is stated (0 = every check passed, 1 = a problem was found; capabilities that were not exercised do not by themselves make it non-zero) | `cli/README.md:80-84` |
| `cli/README.md` said "All four binaries link QTSQ static libs built per-target" while the next paragraph says the macOS library is not in the folder | the lead sentence now says the cross builds use the per-target static libs | `cli/README.md:28-36` |

## Register rows

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D08 CLI, TUI, REPL | `sofuu agent run` health line (Scouter context/brain score) | `real` | A score is printed only when its measured flag is `true` (`main.rs:1241-1247`); an unmeasured context omits the segment and an unmeasured brain side prints `not measured`. The scores come from the real Scouter evaluation (`rt/scouter.rs:593-720`). Not exercised in this worker (build policy). |
| D08 CLI, TUI, REPL | `sofuu doctor` — grant → backend table | `real` | The backend names are probed from `hardware::validate_grant`, the function that enforces `--cap family=backend` (`domains/robotics/hardware.rs:97-146`); `doctor.rs:159-182`, `doctor.rs:1053-1097`. A bare grant is reported as resolving to no implementation, and `sim` is reported as the explicit drone/rover physical test only. Unit test added (`doctor.rs`), type-checked, not executed here. |
| D08 CLI, TUI, REPL | `sofuu doctor` — policy presence, validity and non-emptiness | `real` | `doctor.rs:859-937` reads the two layers the runtime reads and merges them with `rt::policy::layered_policy_from_paths`; an invalid layer is reported as **not** in force, matching `sofuu.policy.load()` (`rt/policy.rs:1344-1349`). Not exercised in this worker (build policy). |
| D08 CLI, TUI, REPL | `sofuu doctor` — model endpoint and weights | `real` | The endpoint is the configured override or the provider's own default (`rt/ai.rs:115-141`); the weights file is checked on disk and a missing file is reported as unavailable (`doctor.rs:1032-1049`). The doctor does not run inference and says so. |
| D08 CLI, TUI, REPL | `sofuu doctor` — Scouter state | `pure` | The doctor reads the process-local Scouter state (`rt/scouter.rs:518-539`). The scores are measured-only: absent (`null` in JSON) when nothing was measured. The measurement itself belongs to a real conversation, which this check does not perform. |
| D08 CLI, TUI, REPL | `sofuu serve` endpoint banner vs the implemented auth boundary | `real` | The banner now matches `src/js/serve.js:820-842`: `/healthz` and `/livez` are tokenless, `/health` needs the bearer token unless `--public-health` (`main.rs:2918-2920`). |
| D08 CLI, TUI, REPL | `sofuu plugin test` help text | `real` | The mode executes the real tool (`src/js/plugins.js:3855-3861`); the help no longer describes a mock (`main.rs:220`). |
| D08 CLI, TUI, REPL | `cli/README.md` — the modes the prebuilt binaries actually have | `real` | Verified against `cli/sofuu-darwin-arm64 --help` and by running `version` (exit 0) and `doctor` (exit 1, unknown command) on 2026-11-16. The README lists the modes that exist and the ones that do not. |
| D08 CLI, TUI, REPL | `cli/README.md` — the brain round-trip claim on `cli/sofuu-darwin-arm64` | `real` | Re-run on 2026-11-16 in `/tmp/sofuu-cli-check`: the first `eval` printed `true` for the flush, the second process recalled `persisted-value` (`1 persisted-value`). The write needed the explicit flush. |
| D08 CLI, TUI, REPL | `cli/README.md` — the networking / TLS description | `real` | Read from the binaries: macOS links the system libcurl (`otool -L`), Linux is static with libcurl 8.6.0 (`strings`), Windows uses Schannel. The CA-bundle overrides are in the current runtime, not in these binaries. |
| D08 CLI, TUI, REPL | `cli/README.md` — the build instructions | `real` | `Makefile` targets `zig-install` / `linux` / `windows-x86_64`, the QTSQ guard (`Makefile:79-87`), `scripts/cross/install_zig.sh` (Zig 0.13.0) and `scripts/cross/build_qtsq_cross.sh` with the `dist/qtsq-*` libraries all exist as documented. |
| D08 CLI, TUI, REPL | The Linux and Windows binaries in `cli/` | `unverified` | They were not executed on this machine (a macOS host). Their brain round-trip and their TLS behaviour are not receipts here — only their static contents were inspected. |

## Blocked or unverified

- **The Rust test suite was not executed in this worker** (build policy: the
  coordinator owns the build). The new test
  `the_backend_table_names_only_real_backends` in
  `crates/sofuu-core/src/doctor.rs` is type-checked only
  (`cargo check -p sofuu-core --bin sofuu --tests` → pass). The coordinator
  must run it before the change is considered green.
- **The runtime output of the new `doctor` sections was not observed.** No
  binary was built here (build policy), so `sofuu doctor` and
  `sofuu doctor --json` must be run once after the build: the new sections are
  `policy`, `kaze backends` and `scouter`, and the JSON document gains a
  `scouter` object whose unmeasured scores are `null`.
- **`cli/sofuu-linux-x86_64`, `cli/sofuu-linux-arm64` and
  `cli/sofuu-windows-x86_64.exe` were not executed** — this is a macOS host.
  Their contents were inspected statically only (`file`, `strings`,
  `otool -L`); their brain round-trip and their TLS behaviour are
  `unverified` here.
- **`sofuu serve` prints its configuration banner before the socket is
  bound.** The honest confirmation is `src/js/serve.js`'s "brain server
  listening" line, which is printed only after a successful bind, and a bind
  error throws (`rt/http_server.rs:1325-1327`) so the command exits
  non-zero. Moving the CLI banner behind the bind would need a change in
  `src/js/serve.js`, which is outside this worker's scope.
- **The `doctor` verdict still says "install healthy"** when every check it
  ran passed, even though the run also lists capabilities that are
  unavailable or were not exercised. That is deliberate — the verdict names
  exactly what was operational and the counts name the rest — but it is a
  wording the coordinator may want to review.

## Notes for the coordinator

- **`Makefile:492` says "cross builds are QTSQ-free (the local checkout is
  macOS-only)", which contradicts `scripts/cross/build_qtsq_cross.sh`** (it
  compiles QTSQ for the three cross targets and points the cross scripts at
  `dist/qtsq-*`) and the README. The Makefile is outside this worker's scope;
  the comment looks stale.
- **`crates/sofuu-core/src/rt/scouter.rs` and
  `crates/sofuu-core/src/script_cmd.rs` were being edited by another worker
  while this one ran** (a `cargo check --tests` at 13:33 saw
  `BrainHealthReport` missing its new `measured` field and a
  `PHYSICAL_SIM_KEYS` / `NON_PHYSICAL_SIM_KEYS` mismatch). Both were resolved
  within a minute and the tree compiles now, but the coordinator should be
  aware that the tree was briefly uncompilable under `--tests`.
- **`crates/sofuu-core/src/rt/storage.rs:1183` currently breaks
  `cargo check --tests`** (`SofuQuery` does not implement `Debug`, so
  `parse_query(bad).unwrap_err()` does not compile). That file is being
  edited by another worker and is outside this worker's scope; the
  non-test build is unaffected.
- **`crates/sofuu-core/src/doctor.rs` compiles into the binary crate only**
  (`main.rs:116` declares `mod doctor;`; `lib.rs` does not). The comment at
  `doctor.rs:953-954` claiming otherwise is stale, but the `sofuu_core::`
  prefix it asks for is correct and is used throughout the file.
- **No shared file outside this worker's scope needs a change for the D08
  rows above.** The one optional follow-up is `src/js/serve.js` +
  `crates/sofuu-core/src/rt/http_server.rs` if the `serve` banner should move
  behind the bind.
