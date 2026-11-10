# Changelog

All notable changes to the `sofuu` runtime are documented here.
Versions follow the Cargo crate (`0.3.0-beta` line); npm and the CLI banner
carry the same version (checked by `scripts/verify-dist.sh`).

## Unified Diagnostics Review Fixes — 2026-11-10

- Sanitization handles ordered and quoted key/value pairs, common token
  prefixes, private-key blocks and URL userinfo. JS error locations and frames
  are bounded/sanitized, and inspecting Errors avoids getters and proxies.
  Pattern-based redaction cannot guarantee discovery of every secret.
- Event redaction preserves multiline and long non-secret content. Checkpoints
  altered by redaction or known producer truncation are marked and rejected
  for automatic resume. Historical checkpoints are not retroactively repaired.
- Full-detail JS diagnostics retain bounded reported frames. Kaze errors avoid
  reopening inputs; run/check JSON errors sanitize the legacy message, and
  check failures honor `--json`.
- The embedded Kaze engine writes bytecode format 2 with statement-line maps;
  its reader accepts format 1 with a labelled routine-start fallback. Older
  runtimes that only read format 1 reject format 2. Original source identity,
  expression spans and Kaze call frames remain incomplete.
- Kaze JS failure objects now carry a diagnostic, which the headless standalone
  executable renders for runtime failures; its legacy JS `error` remains raw.
  Native QuickJS OOM detection avoids executing JavaScript getters.
- Focused checks passed, including Kaze source/bytecode, a standalone
  arithmetic failure, redaction and checkpoint recovery. The macOS arm64
  release-profile primary binary measured 5,668,432 bytes against the
  5,767,168-byte cap. This is source-tree evidence, not a published release.
- The product-wide plan remains incomplete, including source excerpts, Kaze
  call frames, broader privacy coverage and real agent/model/tool integration.
  See local `docs/DIAGNOSTICS.md` for details and unverified surfaces.

## Audit Remediation — 2026-11-08/10

All 32 tasks of `docs/PLAN-AUDIT-REMEDIATION-2026-11-08.md` (150 findings across the P0 security gates, Kaze/QuickJS isolation, CLI truthfulness and the 12-factor surface) are implemented, with the plan's four gates green:

- **Fail-closed policy everywhere (KZ-051/KZ-052):** a policy document that exists but cannot be parsed is fatal. This round closed the last hole — `sofuu script run|check|repl --policy <file>` advertised the flag but never parsed it, so `--policy bad.json` became a positional and the run used the default-allow document. It now fails with `E_CAP_DENIED` and exit 1 (array/numeric/malformed documents alike), honours `--policy=<file>` and `SOFUU_POLICY`, and refuses `--policy` + `--no-policy` as a usage error.
- **SSRF escape hatch is consistent (JR-092):** `SOFUU_ALLOW_LOOPBACK` / `net.allow_loopback` exempted the policy evaluator but not the RESOLVED-address gate in `opensocket`, so the documented hatch was a no-op for every name resolving to 127.0.0.1. Both gates now agree, and the exemption covers loopback only — metadata, private, CGNAT and ULA addresses stay refused.
- **Binary budget recovered without dropping a feature:** the five learned gates keep their trained weights, now stored as IEEE half precision (build-time conversion in `crates/sofuu-core/build.rs`, version-2 container decoded by `ml::net::from_blob`). Committed `.f32` blobs remain the source of truth; `shipped_f16_blobs_track_the_committed_f32_blobs` asserts every weight and a forward pass against them.
- **CLI polish:** `sofuu-agent --help` prints usage instead of trying to open a file named `--help`, and points Kaze agents at `sofuu script run <agent.kaze>`.

Verification on the final tree: `make size-check` → `Size 5668400 bytes <= 5767168 bytes`; `scripts/check-driver-bc.sh` → `BC-CHECK: OK (11 shipped snapshots match their source)`; `cargo test --release --all` (with `SOFUU_ALLOW_LOOPBACK=1`, as `make test-rust` runs it) → 1395 passed, 0 failed, 48 ignored; `tests/run_js_tests.sh` → 72 passed, 0 failed, 1 skipped (the live-provider test, which needs a real API key).
## Unified Diagnostics — 2026-11-10

One versioned diagnostic contract across the CLI, JavaScript, Kaze, events and crash cards (`docs/PLAN-UNIFIED-DIAGNOSTICS.md`):

- **Shared model (`rt/diag_contract.rs`):** `diagnostic_version`, stable `code`, `severity`, bounded `message`/`hints`, optional verified location, and a known `reason`. Unknown failures stay `UNCLASSIFIED`; known codes gain one safe explanation and next step, never a guessed cause. `sofuu.diagnostics.*` exposes `render`, `fromError`, `redact` and `version`.
- **Human and machine adapters:** `sofuu eval`/`run` uncaught exceptions and unhandled rejections now print one concise block (location, reason, help); `SOFUU_DIAGNOSTICS=full` expands the hints. `sofuu script run --json` keeps its `{ok, code, error}` fields and adds the versioned `diagnostic`.
- **Kaze source attribution:** parser errors keep line + column, and checker/compiler/VM errors keep their line. The VM's item-level line limitation is reported honestly; instruction-level spans would require a versioned bytecode migration.
- **Privacy:** the event writer’s existing redaction seam now sanitizes every free-form string (ANSI/OSC, C0/C1/bidi controls, Bearer/Basic credentials, secret-shaped key/value pairs, private home paths). Crash cards are redacted before disk.
- **Compatibility:** existing codes, event kinds, exit status, retries, approvals, C ABI status/envelope and protocol stdout are unchanged. Legacy codes map into the event taxonomy with the original retained in `legacy_code`.
- **Context-limit finding:** regression tests prove the configured short model name resolves a local listing's full-path `.gguf` id, and that learned-limit keys are case- and endpoint-normalized.

Verification: `cargo test -p sofuu-core --lib rt::diag_contract`, `rt::events`, `rt::observability`, `domains::kaze`, plus `cargo test -p sofuu-core --test diagnostics_contract` (all passing locally), and real `sofuu eval` / `sofuu script run|check` receipts. Binary size: this contract measured 5,750,928 B (cap 5,767,168 B) before concurrent audit edits to `main.rs`/`engine.rs` landed; the combined tree reached 5,810,928 B after the audit remediation landed, and `make size-check` is green again at 5,668,400 B: the five TinyMlp weight blobs now ship as version-2 (f16) containers, which halved ~250 KB of committed f32 weights without changing any gate's scores (`blob_integrity` and a new shipped-vs-committed drift test pin both directions).

## Enterprise Hardening & Linux Cross-Compile Static TLS Integration — 2026-11-09

Reliability, networking, and cross-platform transport hardening across Linux, enterprise schemas, and zero-crash invariants:

**Linux Standalone Static TLS & HTTPS Support.**
- **Static OpenSSL 3.3.2 Cross-Compilation (`scripts/cross/build_openssl_static.sh`):** Added automated cross-compilation pipeline for static `libcrypto.a` and `libssl.a` targeting `x86_64-linux-musl` and `aarch64-linux-musl` using `zig cc` (pure C with `no-asm` and deterministic symbols).
- **TLS-Enabled Static Libcurl (`scripts/cross/build_libcurl_static.sh`):** Built libcurl with `-DCURL_USE_OPENSSL=ON` and merged OpenSSL static archives into `libcurl.a` using response-file bundling (`@objlist.txt`). Automatically detects and replaces stale TLS-free static archives.
- **Root CA Trust Store Auto-Discovery (`crates/sofuu-ffi/src/curl.rs`):** Implemented multi-distribution root certificate bundle probing on Linux (`/etc/ssl/certs/ca-certificates.crt`, `/etc/pki/tls/certs/ca-bundle.crt`, `/etc/ssl/ca-bundle.pem`, `/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`, `/etc/ssl/cert.pem`). Honors standard `SSL_CERT_FILE` and `SSL_CERT_DIR` environment variables in addition to `SOFUU_CA_BUNDLE` and `SOFUU_CA_PATH`. Resolves `CURLE_UNSUPPORTED_PROTOCOL` (`✖ Unsupported protocol`) when communicating with cloud LLM endpoints from cross-compiled Linux binaries.

**Enterprise Hardening Phase 1 & 2 (Zero-Crash & Schema DX).**
- **Promise Rejection Containment:** Implemented global unhandled rejection trapping in `promise.rs` and `process.on('unhandledRejection')` lifecycle handling in `process.rs`.
- **Global Schema Engine (`sofuu.z` / `globalThis.z`):** Built-in zero-dependency schema validation and conversion to Claude, OpenAI, and MCP tool definitions.

## MCP Tooling, In-Memory Direct Transport, Middleware & Bundler Hardening — 2026-11-07

Post-stabilization freeze enhancements across Model Context Protocol (MCP), standalone bundler resolution, cross-compilation tooling, and CUDA kernel decoders:

**MCP Pillar 3 — In-Memory Direct Transport & Middleware Onion Pipeline (`51fa5a23f`).**
- **In-Memory Direct Transport (`transport: "direct"`):** Connect in-process MCP servers directly to runtime clients (`sofuu.mcp.connect(server, { transport: 'direct' })`) with zero IPC overhead, eliminating process-spawn and JSON serialization bottlenecks (sub-millisecond invocation latency). Directly accepted by `sofuu.agent.define({ tools: [directClient] })`.
- **Middleware Onion Pipeline (`client.use` / `server.use`):** Added composable async middleware interceptors (`(ctx, next) => { await next(); }`) executing in standard onion order across both client and server lifecycles, enabling cross-cutting concerns (tool call tracing, validation, authorization, retries).
- **Standalone HTTP MCP Server:** Added HTTP server mode (`sofuu.mcp.serve({ transport: 'http', port })`) with graceful connection teardown (`server.close()`).
- **MCP CLI Inspection:** Added `sofuu mcp inspect <name | command line | http(s)://url> [--call <tool> [--args <json>]]` to introspect server capabilities, list tools/resources, and invoke tools directly from the command line.

**MCP Tooling, Scaffolding & IDE Integration (`8514a2f76`).**
- **Server Scaffolding (`sofuu mcp init <name>`):** Scaffolds new MCP servers with idiomatic schemas, tools, and entrypoints.
- **IDE Linking (`sofuu mcp link <claude-desktop | cursor>`):** Automatically provisions runtime server entries and configurations into Claude Desktop (`claude_desktop_config.json`) and Cursor (`cursor_settings.json`).
- **Module Exports:** Shipped top-level `mcp` export in the `sofuu` module interface (`import { mcp } from 'sofuu'`).

**Bundler Built-in Virtual Module Resolution (`e64344b7c`).**
- Resolved virtual module identifiers (`sofuu`, `@sofuu/*`, and built-in modules) during bundling and standalone compilation (`sofuu bundle`, `sofuu compile`), preventing erroneous external-resolution fallbacks when packaging self-contained agents.

**Cross-Compilation Tooling & Release Hygiene (`592dec45c`, `9adbb511e`).**
- Replaced invalid `git diff --untracked-files=no` option in `scripts/cross/build_linux_arm64.sh`, `build_linux_x86_64.sh`, and `build_linux_gnu_cli.sh` with `git status --porcelain --untracked-files=no`, stopping spurious `-dirty` suffixes on clean cross-builds.
- Fixed UTF-8 default reading in `scripts/check_podspec.rb`, whitespace formatting in `E_KAZE_MEMORY`, and directional npm/Cargo version check in `scripts/verify-dist.sh`.

## Stabilization Freeze — RAM, Concurrency & Error-Precision Closeout — 2026-11-05

[`docs/PLAN-STABILIZATION-FREEZE.md`](docs/PLAN-STABILIZATION-FREEZE.md) is implemented and verified pass by pass. Pass 1 (F-01…F-04) was already landed in the audit remediation; this entry covers the remaining work.

**Pass 2.1 — the 64 MiB Argon2id spike is verified gone from steady state.**
- QTSQ KDF v2 (process-wide master derivation + per-file SHA3 PRF) is what the linked `libqtsq.a` already ships; the runtime now **pins** it. `sofuu-ffi` gained `security_version()` (offset 400, header-probed) and `KDF_VERSION_MASTER`; `vault_containers_use_the_process_wide_master_kdf` fails if a new container regresses to per-container Argon2id.
- The stale `brain_store.rs` cost comment now describes the real behaviour: the 64 MiB stretch runs once per process/password; legacy v1 files keep their per-file derivation as a one-time migration.

**Pass 2.2 — explicit zero-KDF plaintext mode for edge devices.**
- `sofuu.memory.open(path, dim, { plaintext: true })` (aliases `directKey` / `direct_key`) writes segments, manifests and legacy single-file saves through the codec's plaintext streaming writer. No password vault, no Argon2, no KDF.
- The mode is persisted as `Manifest.plaintext`; a later open without the option adopts it instead of half-rekeying the store. An explicit plaintext open of an encrypted store marks the next flush as a full rewrite.
- Measured on macOS arm64: first encrypted flush peaks at **79,839,232 B** RSS; the same plaintext flush peaks at **12,632,064 B** — the 64 MiB transient is gone.
- New tests: `plaintext_segments_and_manifest_round_trip_without_the_vault`, `plaintext_open_writes_unencrypted_files_and_a_silent_reopen_adopts_it`, `legacy_single_file_plaintext_save_round_trips`, plus a JS end-to-end open/flush/reopen check.

**Pass 2.3 — bounded streaming text buffers.**
- `agent.js` and `serve.js` coalesce token streams into ~64 KiB chunks (`textChunkBuffer`) instead of growing one string per token; `serve.js` also stopped accumulating thinking text that no reader consumed.
- The agent's mid-stream continuation path now reads the same buffer (`parts.text()`), so a cut answer still resumes exactly once with overlap trimmed.

**Pass 2.4 — turn-boundary reclamation.**
- `agent.js finish()` runs one full collection at the root turn boundary; `serve.js` runs one in the request `finally` for alloc-heavy routes only (health probes stay allocation-free); `runMany` keeps its batch collection.
- `rt/engine.rs`'s GC bridge now also calls `malloc_trim(0)` on glibc, so the RSS curve follows the JS heap down. `sofuu.gc()` is exposed for hosts that want an explicit boundary.

**Pass 3.1 — concurrency ceiling measured, not assumed.**
- `scripts/bench/event_loop_latency.js` (`make event-loop-check`, wired into `make test`) drives the shipped `sofuu.serve` handler with a macrotask-yielding fake stream: 32 concurrent turns × 128 chunks measured **max loop lag 1 ms**, p95 1 ms, against the 16 ms hard-realtime ceiling. The 64×256 stress row is also 1 ms.

**Extreme cold-start characterization.**
- `scripts/bench/extreme_cold_start.py` (`make cold-start-bench`) runs ~7,000 process launches: 1,000 spawn-floor runs, 500 engine boots, 300-run env variants, and concurrent bursts to 200 processes.
- Current build: **14.58 ms p50 / 15.33 ms p95 / 15.53 ms p99** for `sofuu eval "1+1"`; spawn floor 1.48 ms; CLI-only 4.35 ms; idle RSS 10.91 MB.
- Pre-stabilization artifact `51fa5a23f`: 14.47 ms p50 — **no regression**. `SOFUU_NO_ML` and `SOFUU_NO_TOOLS` do not move boot. `capabilities` pays a deliberate Metal probe (~31.9 ms).
- Concurrent throughput saturates at **~400 process starts/s** on 10 cores; beyond one process per core, latency rises with queuing (documented in BENCHMARKS.md §2). Warm page cache only: macOS `purge` requires root.

**Pass 3.2 / 4.1 — gates.**
- Clippy ratchet green (619 / 6 / 51, at baseline).
- Both Linux artifacts rebuilt from this tree and under the cap: arm64 **5,117,296 B**, x86_64 **5,751,616 B** (≤ 5,767,168 B).

**Verification receipts:** 1,154 lib tests passed / 0 failed / 47 ignored · `bash scripts/preflight.sh` green · `make test` green (full JS suite 72 passed / 0 failed / 1 skipped) · `make bc-check` 11/11 · `make plugin-check` 5/5 · `make size-check` 5,628,208 B · `make cold-start-bench` 14.58 ms p50 · `make event-loop-check` 1 ms max lag · serve suite 34/34 · agent suite all passed · memory JS suites 101/30/22/40/69 assertions passed · clippy ratchet 619/6/51 at baseline. `verify-dist` remains red only on release hygiene (dirty tree; the darwin dist archive predates this HEAD), not on size.

## Full Source Audit Remediation, Zero-Catch Retiring & Linux Binary Cap Compliance — 2026-11-04

[`docs/PLAN-AUDIT-REMEDIATION-2026-11-02.md`](docs/PLAN-AUDIT-REMEDIATION-2026-11-02.md) is fully implemented and verified across all 7 phases, remediating all 16 findings from the 2026-11-02 independent source audit:

**Phase 1: Physical safety gates & resource bounding.**
- **H-4 Drone Goto Altitude Range Gate (`domains/robotics/hardware.rs`):** Switched coordinate checking to arity-based selection (index 2 for 3-argument `goto(lat, lon, alt)`, index 1 for 2-argument). Western-hemisphere negative longitudes are now correctly accepted, and altitudes exceeding 500m are refused with typed `E_KAZE_RANGE` (reject-never-clamp).
- **Fail-Closed Policy Firewall:** In `hardware.rs`, unreadable or missing policy verdicts fail closed with `E_CAP_DENIED`. The `"ask"` verdict explicitly fails closed with `E_CAP_APPROVAL_REQUIRED` to prevent headless autonomous control loops from executing without human approval.
- **H-4 Non-Numeric Argument Rejection:** `range()` rejects missing or non-float arguments with `E_KAZE_RANGE` instead of passing.
- **Bounded Kaze Collection Memory (`domains/kaze/value.rs`):** Introduced metered `Heap` with a 4 MiB default limit (`Limits.heap_bytes`, CLI `--heap-kb`, `RunOptions.heap_bytes`). Allocations in `push_array` and `push_map` charge 16 bytes per `Value` plus `Vec` overhead, returning typed `E_KAZE_MEMORY` when the budget is crossed. Documented in `KAZE-SPEC.md` and `README.md`.
- **Wholesale Deallocation & Equality:** `free_wholesale` is now invoked at run completion to reset both arena and collection heap meters. Array and map comparison (`==`) is now structural (recursive content equality up to depth 64), matching string equality semantics.

**Phase 2: Kaze engine reconciliation & drift prevention.**
- Shipped `scripts/kaze-drift-check.sh` (`make kaze-drift-check`, wired into `make test`) detecting drift between `domains/kaze/` and the prototype `kaze/`.
- Ported `device.set_duty` (PWM) into the core checker, `SimHost`, and `rt/hardware.rs` with 0..100% duty cycle validation.

**Phase 3 & 6: Tooling health, clippy unblock & robustness.**
- **Clippy Gate Unblocked:** Fixed all 5 `.unwrap()` calls in `crates/sofuu-ffi/src/qtsq_table.rs` with typed `Result` error mapping, satisfying `#![deny(clippy::unwrap_used)]`.
- **CI Ratchet:** Shipped `scripts/check-clippy-ratchet.sh` enforcing that warnings never exceed the committed baseline (`.ci/clippy-baseline.txt`).
- **Events FFI Guard:** Added `argc < 1 || argv.is_null()` validation to all QuickJS event callbacks in `rt/events.rs`.
- **Zero-Crash Hardening Scope:** Refined roadmap language to clarify that promise rejections and event-loop unhandled errors are hardened, while Rust runtime panics deliberately retain `panic = "abort"` semantics. Hostile zip archives in `zip_reader.rs` now return typed `Err` instead of aborting.

**Phase 4: CUDA kernel launch hardening.**
- Declared `.maxntid 256,1,1` launch bounds across CUDA reduction, RMSNorm, and quantization kernels in `kernels.cu`.
- Guarded `kv_write` with pre-launch buffer capacity checks (`pos + n_tokens <= cap`) in Rust (`validate_kv_write`) and CUDA.
- Fixed driver table leak on failed initialization in `cuda_loader.c`. Added Valgrind leak check script `scripts/cuda/check-leaks.sh`.
- Aligned documentation (`SOFUU_CUDA` is documented as a kill-switch, not an enable flag; unified `SOFUU_CUDA_DEBUG`).

**Phase 5: Hygiene sweep & zero-catch backlog retirement.**
- **Zero-Catch Backlog Retired (337 $\rightarrow$ 0):** Every catch block across all JavaScript runtime files (`src/js/*.js`) that previously swallowed errors now explicitly comments what error is ignored and why nothing depends on it. CI ratchet `scripts/check-js-swallow.py` enforces zero undocumented swallows.
- **Dead Code Cleanup:** Triaged all 33 unused functions/constants and 10 unused struct fields with explicit dispositions. Removed redundant `unsafe` blocks.
- **Stale Knobs Frozen:** Frozen documented-only environment variables under `scripts/check-env-knobs.py`.

**Phase 7: Linux binary size compliance under 5.5 MiB.**
- Closed the multi-megabyte Linux binary overage without raising the 5.5 MiB (5,767,168 B) ceiling.
- Enabled `-Z build-std` with `panic=immediate-abort` in cross-compile scripts (-1.16 MB) and optimized C dependencies (libuv `MinSizeRel`, QTSQ `-Oz`, kernels `-Oz`).
- **macOS arm64:** 5,595,072 B ($\le 5,767,168$ B, 172 KB headroom).
- **Linux arm64:** 5,094,896 B ($\le 5,767,168$ B, 672 KB headroom).
- **Linux x86_64:** 5,726,720 B ($\le 5,767,168$ B, 40 KB headroom).
- Verified execution under QEMU Alpine VM (`scripts/cross/vm-smoke.sh`); documented x86_64 AVX2 baseline requirement.
- Fixed SwiftPM and CocoaPods release manifest URL formatting (`libsofuu-xcframework.zip`) and checksum pinning.


## Zero-Risk Codebase Decoupling & In-Tree Capsule Modularization — 2026-11-02

[`PLAN-ZERO-RISK-MODULARIZATION.md`](docs/PLAN-ZERO-RISK-MODULARIZATION.md) is implemented and verified end to end: encapsulates core systems substrate and domain expansions into isolated capsules with zero binary size inflation and zero linker churn.

**Domain capsule encapsulation (`crates/sofuu-core/src/domains/`).**
- **Kaze Domain (`domains/kaze/`):** Full 115 KB clean-room VM, lexer, parser, checker, compiler, and capabilities isolated from `sfs/`. Backward-compatible alias `pub use domains::kaze as sfs;` in `lib.rs`.
- **Robotics & Hardware Domain (`domains/robotics/`):** Consolidates MAVLink v2 streaming, 6-stage hardware veto gate, transports, 1 Hz dead-man switch, GPIO, and mission operations out of `rt/`.
- **Cognitive Memory Domain (`domains/memory/`):** Encapsulates CMA v2, HNSW, and decay evaluators under clean domain boundaries.
- **Inference Domain (`domains/infer/`):** Encapsulates tensor graph execution and model architectures under clean domain boundaries.

**Subsystem registration trait (`domains/subsystem.rs`).**
Introduced unified `Subsystem` trait (`name()`, `register_js()`, `health_check()`) and `init_subsystems(ctx)` dispatch, cleanly decoupling domain global initialization from `rt/engine.rs`.

**Local invariant preflight gate (`scripts/preflight.sh`).**
Shipped 4-stage automated gate verifying zero cross-domain coupling, QuickJS bytecode snapshot synchronization (`make bc-check`), strict binary cap compliance ($\le 5.5\text{ MiB}$), and test execution.

**Verification and test receipts.**
- 1,138 unit tests passing (0 failed).
- `make size-check` passing at 5,594,992 B ($\le 5,767,168$ B cap, 172 KB headroom).
- All Kaze tests (15/15 language, 3/3 live) and MAVLink tests (91/91) passing green.
- Full contract test suite (`scripts/contract/run_contract_tests.sh`) passing 51/52 (0 failed, 1 runner-blocked).

## Drones, MAVLink v2 & Robotics Ground / Ambient Subsystem — 2026-11-01

[`PLAN-DRONES-MAVLINK.md`](docs/PLAN-DRONES-MAVLINK.md) and [`PLAN-ROBOTICS-PI5-EDGE.md`](docs/PLAN-ROBOTICS-PI5-EDGE.md) are implemented and verified: a full companion-computer brain for aerial drones, ground rovers, and Raspberry Pi 5 ambient edge devices, governed by the deterministic R7 hardware safety firewall.

**Zero-dependency MAVLink v2 codec (`domains/robotics/mavlink.rs`).** Hand-rolled streaming parser and serializer with CRC-16/MCRF4XX checksum calculation, pinned `CRC_EXTRA` table, sliding-window resynchronization over EMI/motor noise, and PreArm diagnostic extraction from `STATUSTEXT` (msgid 253).

**Deterministic R7 hardware safety gate (`domains/robotics/mavlink_gate.rs`, `domains/robotics/hardware.rs`).** Enforces a 6-stage veto chain on every outgoing frame (device allowlist → deny-list → send-whitelist → MAV_CMD whitelist → non-clamping payload bounds → rate limits → approvals). GPS fix quality drops (< 3 sats or HDOP > 2.0) or telemetry loss (> 1.5s) immediately veto coordinate commands with typed `E_NO_GPS_FIX`.

**Transports & heartbeat dead-man switch (`domains/robotics/mavlink_link.rs`).** Implemented dual UDP transport (`udp:` bind-and-learn + `udpout:` direct peer) and UART serial over PTY pairs. Runtime-scheduled 1 Hz `HEARTBEAT` loop triggers ArduPilot's native GCS failsafes upon interruption.

**JavaScript surface, drone skills & mission DSL (`src/js/drone.js`, `domains/robotics/drone_ops.rs`).** Exposes `sofuu.drone` and `sofuu.mavlink`, along with `PACKS.drone` registering 6 agent tools (`drone_arm`, `drone_takeoff`, `drone_goto`, `drone_rtl`, `drone_hold`, `drone_telemetry`). Static pre-flight validator checks entire missions before any frame exists.

**Ground rover & Linux GPIO (`domains/robotics/gpio.rs`, `deploy/rover/`).** Zero-dependency GPIO driver supporting `/sys/class/gpio` and Raspberry Pi 5 `/dev/gpiochip*` v2 line ABI with non-blocking kernel pulse timers. Ground rover test harness (`scripts/rover/preflight_cases.sh`) verifies 5/5 passing cases against geofence and speed violations.

**Kaze live hardware bridge (`domains/robotics/kaze_live.rs`).** Couples Kaze natural-phrasing scripts directly to live MAVLink flight links and GPIO buses behind `CapHost`.

**CLI toolchain (`sofuu drone`).** Shipped subcommands: `observe`, `status`, `send`, `mission`, `why`, and `init`.

**Verification and test receipts.**
- 123 unit and integration tests passing across MAVLink, drone operations, GPIO, and Kaze live backends.
- Fuzzing target `crates/fuzz/fuzz_targets/mavlink.rs` active.
- Live ArduPilot SITL flight over Wi-Fi verified (`takeoff -> goto -> rtl`).


## Production Local Inference Plugin Bridge & Plugin Add CLI — 2026-11-01

Enterprise Hardening Phase 3 is implemented end to end: high-throughput local
inference bridging, automated CLI plugin installation, and zero-overhead
provider execution while maintaining the $\le 5.5\text{ MiB}$ standalone binary
envelope.

**Local inference provider (`plugins/llamacpp_provider.ts`).** Implemented a
production Category 2 `defineProvider` plugin connecting Sofuu to local
`llama.cpp` server instances (or any OpenAI-compatible local runtime such as
vLLM or LocalAI). Features SSE stream decoding, completion mapping, micro-tool
call parsing, token usage metrics, model metadata extraction, and active health
probing (`/health`). Internal symbols are scoped to prevent namespace collisions
across QuickJS evaluation frames.

**CLI plugin management (`sofuu plugin add <name>`).** Added turnkey plugin
installation (`P.cliAdd` in `src/js/plugins.js` wired to `cmd_plugin` in
`crates/sofuu-core/src/main.rs`). Supports official presets (`sofuu plugin add llamacpp`,
`sofuu plugin add ollama`) as well as custom scaffolds with flags `--global` (installing
to `~/.sofuu/plugins/`) and `--force` (overwriting existing implementations). All
templates strip cleanly under Sofuu's built-in TypeScript parser without requiring
`node_modules` or bundlers.

**Fallback guidance and acceleration tips.** Integrated acceleration suggestions
into `print_infer_help()` and runtime error paths in `sofuu infer` and `sofuu chat`,
guiding users to activate hardware-accelerated local inference plugins without
obscuring native fallback paths.

**Verification and constraints.**
- Zero model weights downloaded (`AGENTS.md` compliant).
- Recompiled QuickJS bytecode snapshot (`src/js/plugins.bc`).
- Binary size checked at 5,659,776 B ($\le 5,767,168$ B cap, 107 KB of headroom).
- All 5 cookbook plugins validated cleanly (`make plugin-check`).


## Kaze v0.1 — the device scripting language — 2026-11-01

[`PLAN-KAZE-LANGUAGE.md`](docs/PLAN-KAZE-LANGUAGE.md) is implemented end to
end: a new domain language, its own bytecode engine, a command surface and a JS
bridge. Kaze is additive — QuickJS, the shipped drivers and every existing
`sofuu.*` API are untouched.

**The engine** (`crates/sofuu-core/src/domains/kaze/`, with backward-compatible `crate::sfs` alias) is a clean-room pipeline written
with no new crates: lexer (indentation-aware, tabs refused) → parser
(natural-phrasing verbs, unit normalization) → checker (declared capabilities,
op arity, unit conflicts, `filter` termination on every path) → compiler
(fixed 4-byte ISA, 33 opcodes) → VM (linear arena, fuel per instruction, host
deadline poll at every loop head). Compiling the same source twice produces
identical bytes, and `sofuu script compile` writes a `.kazebc` module that
reloads and runs without the front end (`sofuu script run patrol.kazebc` executes
the artifact from its bytes, with the same trace as the source). Equality branches compile to the fused
`JEQ`/`JNE` opcodes, so a decision costs one instruction and materialises no
boolean.

**Safety is structural, not advisory.** A script has no instruction that can
reach the outside world: `CALL_CAP` is the single door, and it passes the R7
hardware gate in `rt/hardware.rs` first — `E_CAP_UNDECLARED` (H-1),
`E_CAP_MISSING` (H-2), `E_CAP_DENIED` (H-3) through the one policy evaluator,
and `E_KAZE_RANGE` (H-4). Arguments are rejected, never clamped: `takeoff to
900 meters` stops the run instead of flying somewhere else. Fuel bounds every
run, and a JS caller's deadline interrupts it: both report `E_TIMEOUT` as the
plan requires, and the message says which limit fired.

**The capability surface is the spec's, not a subset.** `drone.goto(lat, lon,
alt)` takes the three arguments its own signature names, `sensor.read(pin)` and
`device.read_gpio(pin)` read a named telemetry field (or a deterministic `0` for
a numeric pin), and records — the maps JSON arguments already produce — are read
with `record.field`, which compiles to `GET_FIELD`. A missing field reads as
`nothing`; a field read on something that is not a record is `E_KAZE_TYPE`.

**Surfaces.** `sofuu script run|check|compile|repl` — the command never
initialises QuickJS, so `check` answers with no model, config or network — and
`sofuu.sfs.run/check/compile/version` from JS, which returns a report object
(`ok`, `actions`, `logs`, `result`, `verdict`, `fuel`) instead of throwing,
so a tool loop can branch on `ok`.

**Examples and docs.** `examples/kaze/` ships the four canonical scripts from
the plan (drone patrol with a low-battery hook, rover obstacle loop, greenhouse
climate check, prompt guardrail), a minimal one, and
[`patrol_tool.mjs`](examples/kaze/patrol_tool.mjs) — the §6.2 agent-tool shape,
wired with `import { defineTool, sfs } from "sofuu"` (the `sofuu` module now
exports `sfs`);
`crates/sofuu-core/tests/kaze_language.rs` exercises them, including the
refusal paths. `docs/KAZE-SPEC.md` freezes the grammar, unit table, capability
schema, opcode table, module layout and error codes; `README.md` gains a Kaze
section and a `sofuu.sfs` API entry.

**Verified on a release build.**
`cargo test -p sofuu-core --lib sfs::` and `hardware::` pass (64 tests), and
`cargo test -p sofuu-core --test kaze_language` passes (15 tests, one of them
comparing every example against the byte-exact trace in
`examples/kaze/golden/`). The traced
patrol run, the `E_CAP_MISSING` refusal, the fuel `E_TIMEOUT` stop, the guardrail
verdicts, the `triage` run that reads `record.kind` and `record.level`, and the
JS round-trip quoted above were produced by `./sofuu`, not by inspection.

**Size.** A differential build of the same tree with the Kaze modules disabled
measures the engine at 115,520 B of binary. `make size-check` passes after the
final change: 5,759,808 B stripped against the 5,767,168 B cap — 7,360 B of
headroom, and shrinking while the in-flight M-D1 work lands in the same tree.

---

## Audit remediation: 36 defect classes closed — 2026-10-31

A full stress audit of the macOS and AWS builds produced 54 findings across the
runtime and the site. This entry records what changed in the engine. Every claim
below was reproduced against a built binary first and re-verified after the fix;
the verification set is listed at the end.

**Release integrity (P0-1, P1-8).** `sofuu version` now prints the build
revision, so two binaries from a 53-commit gap are no longer indistinguishable.
The Makefile copied the release binary onto the `sofuu` symlink with `cp`,
which aborts as "identical (not copied)" — and because it was chained with
`&&`, every strip, codesign and size check after it silently stopped running,
shipping an unsigned binary that Apple Silicon kills at exec. It now removes the
target first. `scripts/verify-dist.sh` fails a release that ships a dirty-tree
build, an artifact with no HEAD stamp, or one over the 5.5 MiB cap.

**Binary size (P1-8).** Four driver payloads were embedded as raw JavaScript and
four Metal kernels as text. Bytecode-only embedding in release was already the
rule for eleven drivers; the stragglers now follow it and the kernels ship
gzip-compressed and are inflated at init. The host build is 5,559,904 B against
the 5,767,168 B cap — 207 KB of headroom where the tree was 320 KB over.

**HTTP server (P0-2, P1-12).** `listen()` unrefed its handle, so the README's
server example exited immediately and the MCP HTTP transport was dead on
arrival. It now refs like Node; `server.unref()` and `SOFUU_SERVERS_UNREF=1`
are the opt-outs, and embedded hosts keep the old behaviour.

**fetch (P1-2, P1-3, P1-4).** `AbortSignal` is honoured mid-body and mid-headers
through a dedicated poll timer (never removing a handle from inside a curl
callback), orphaned responses no longer reject armed readers — 60 concurrent
readers now return 60 bodies where 55 failed — and responses are gzip-decoded
because `CURLOPT_ACCEPT_ENCODING` was being handed a NULL.

**HTTP handlers (P1-11).** A handler that threw — synchronously, or inside an
`async` function — left the client to time out. The returned promise was freed
without a rejection handler, so the throw surfaced as a process-wide
`UnhandledPromiseRejection` and the connection sat until its own deadline. The
rejection handler is now bound to that request's response
(`JS_NewCFunctionData`, so there is no global "current response" to race on and
no per-request eval), answers 500 exactly once, terminates a streaming response
instead of hanging it, and never puts the script's error text on the wire.
`req.on('data'|'end'|'close')` exists as well; unknown events and late
registration throw a `TypeError` rather than silently never firing.

**TypeScript erasure (P0-3, P1-1).** Two silent miscompiles: constructor
parameter properties were dropped, and — worse — every object literal whose
value was a bare identifier lost it (`{ k: x }` compiled to `{ k }`), which is
a `ReferenceError` in the most common construct in JavaScript. The stripper now
tracks brace and paren context, so a `: ` is only treated as a type annotation
where a key cannot appear. The same pass fixed three neighbours: a ternary colon
after an object literal was eaten as an annotation, `a < b > (c)` was blanked as
type parameters, and generic arrows (`const f = <T>(x: T) => x`) were not
recognised. `implements`, `abstract` classes and members, and
`export type`/`export interface` are now erased instead of throwing. `namespace`
and decorators still refuse with a `SyntaxError` — deliberately, because they
need a transform this stripper does not do.

**Brain (P1-5, P1-13, P2-3, P2-4).** Vector scores are normalised into 0…1, a
dimension mismatch or non-finite value is rejected with `E_DIM`/`E_NONFINITE`
instead of being stored, and `remember({vector})` without text no longer throws
`E_TYPE`.

Opening and closing a brain also stopped leaking. `brain_store` released a
scanned-past stream with `qtsq_ctx_free`, which frees the 5 KB context block but
not the payload `qtsq_container_get_stream` had just allocated into it — and
because a segment carries `[memories, meta]` and the reader asks for `meta`
first, every segment read orphaned its whole packed tensor (81,647 B per scan on
a 65 KB stream; 1,092 B after the fix, and `stream_names` had the same bug).
The other half was not a leak at all: a closed handle keeps a whole loaded brain
reachable until a collection runs, and the only trigger was a 64 MB JS-heap
budget that a Rust-side reopen loop never reaches, so retention grew without
bound (0.5 MB/cycle over 900 cycles, no plateau). `close()` now runs a
collection: the handle being closed is still referenced, so the sweep reclaims
the previous cycles — which is what caps the live set at one brain instead of
N.

**Policy (P1-6, P1-15, P1-14).** `policy.set()` validates against a schema and
fails closed: a misspelt key, a string where a list belongs, or malformed JSON
is refused and the running policy is left untouched, instead of silently
clearing it. `evaluate()` answers from the enforcement path, so it can no longer
report a decision the kernel would not make. A permanent QuickJS interrupt
handler bounds each regular-expression call (`SOFUU_REGEX_BUDGET_MS`, default
1000, `0` disables) so a catastrophic backtrack costs a caught error rather
than the process.

**Runtime surface (P1-9, P2-1, P2-2, P2-10, P2-13).** `structuredClone`,
`Blob`, `File`, `FormData`, `Event`/`EventTarget`/`CustomEvent`,
`MessageChannel`/`MessagePort`, `navigator` and a persistent `localStorage` are
now defined; `process.memoryUsage()` and `process.stdin.pipe()` exist; `eval`
accepts top-level `await`; an import cycle evaluates each module once; and an
empty prompt exits 0 with a message instead of 1.

**MCP and plugins (P2-7, P2-6).** `mcp.connect({command, args})` works — the
finding's premise was wrong — but the failure was opaque because the child's
stderr went to `/dev/null`; it is now captured and the rejection reads
`MCP server exited before initialize (exit code N): <stderr>`. A plugin whose
default export is `{ handler }` registers as a tool with `run` aliased.

**argv (P1-10).** `process.argv` leaked the `run` subcommand, so the standard
`argv.slice(2)` returned the script name; a compiled artifact omitted the script
slot entirely. Both are now `[exec, script, ...args]`.

**Build (P1-7, P2-12).** A fresh clone can build: the missing `SOFUU_QTSQ_DIR`
and `pkg-config` prerequisites are documented, and the panic messages name the
fix.

**Docs and site (P2-9, P0-2).** The installer verified checksums by delegating
to `shasum -c`, which is satisfied by a `.sha256` file that contains nothing
but a hash, so a tampered archive with a one-line checksum file installed
cleanly; it now compares an expected digest explicitly. `scripts/check-doc-metrics.sh`
reads prose, so the retired 5.2 MB and sub-4 ms figures are gone from every live
document. The embedding pages no longer advertise SwiftPM and CocoaPods
coordinates that cannot resolve, the four thin pages carry JSON-LD, and
`public/_headers` ships a CSP and HSTS measured against the actual export.

## Modern npm resolution, WHATWG streams, and import diagnostics — 2026-10-30

Three friction points that made modern AI SDKs fail on Sofuu, closed:

**Conditional `exports` (`crates/sofuu-core/src/npm.rs`).** The resolver now
reads `package.json#exports` before `module`/`main`: exact subpaths, `./*`
wildcards (longest prefix wins), string/array/nested-condition targets, and a
condition order that differs per loader — `import` → `module` → `default` →
`node` for ESM, `require` → `module` → `default` → `node` → `import` for
`require()`. That last list is the fix for dual-published packages, which used
to hand `require()` an `.mjs` file. Legacy `module`/`main`/`index.js` and deep
imports still resolve; escape targets (`..`, absolute) are refused.

**Web streams (`src/js/compat.js` + `rt/http_client.rs`).** `ReadableStream`,
`WritableStream`, `TransformStream`, `TextDecoderStream` and
`TextEncoderStream` are implemented in plain JS (no dependency, no new native
code), and `Response.body` is now a stable `ReadableStream` backed by the same
curl write-callback chunks the runtime already delivered — `getReader()`,
`pipeThrough(new TextDecoderStream())`, `for await`, `tee()` and `locked` all
work, so official SDK streaming loops run unmodified. The previous async
iterator remains the fallback when no `ReadableStream` is installed.

**Diagnostics (`rt/diagnostics.rs`).** A `.node` addon (direct, as a package
`main`, or imported from ESM), an absent Node built-in (`child_process`, `v8`,
`worker_threads`, `cluster`, `inspector`), `require()` of an ES module, and a
package that resolves to nothing now throw coded errors
(`ERR_SOFUU_NATIVE_ADDON`, `ERR_SOFUU_UNSUPPORTED_NODE_API`,
`ERR_SOFUU_REQUIRE_ESM`, `ERR_SOFUU_RESOLUTION`) whose second line says what
to use instead — including a small table of native packages with pure-JS or
in-runtime alternatives.

`make modern-npm` runs 28 acceptance checks against a real chunked
SSE server (chunk arrival time proves streaming, not a resolved promise).
`COMPAT.md` records the new surface, including what is deliberately absent.

## Pluggable memory adapters — 2026-10-30

`sofuu.memory` was hard-wired to the embedded QTSQ engine. Deployments that
already run a vector database had to move their data into a `.qtsq` container
(or keep a second copy), which is friction at review time and a blocker for
serverless swarms that need one shared store. `open()` now routes to three
handle kinds behind the *same* flat API:

| Backend | Selected with | Transport |
|---|---|---|
| native QTSQ (default) | `memory.open(path, dim)` | in-process, unchanged |
| custom adapter | `memory.open({remember, recall})` | whatever the developer writes |
| `qdrant` / `pinecone` / `pgvector` | `memory.open(uri, {adapter, dimensions, …})` | `fetch` only |

- **One contract.** A remote handle takes the same calls as a native brain:
  `remember(vec, text, role, page, source)` and `remember(text, {metadata, tags})`,
  `recall("text", 5)` and `recall({text, vector, limit, threshold, filter})`.
  Remote methods return promises, so `await brain.remember(…)` reads the same
  on every backend; the native methods stay synchronous.
- **Scores are clamped cosine in [0, 1]** (an exact hit is 1.0), `threshold`
  is applied client-side on that normalized score for every backend, and
  `filter` is flat equality on metadata keys (payload for qdrant, metadata
  for pinecone, the RPC `filter` argument for pgvector).
- **Embedding is local by default** — `sofuu.ai.embedLocal` at `dimensions`
  width (deterministic, no key). `embed: fn`, `embedSpace: 'semantic' |
  'semantic-v2'`, or `embedProvider`/`embedModel` override it. A custom
  adapter gets raw text unless it declares `embed`, so the router never
  embeds behind a backend's back.
- **No native client**: all three backends are JavaScript over `fetch`.
  There is no `libpq`, no driver crate, and nothing new to link.
- **R7 firewall inherited.** Every request asks `sofuu.policy.evaluate` for
  `net.fetch` on the host first (typed `E_PERMISSION_DENIED` carrying the
  rule id); `net.enabled:false` stops adapter traffic; a read-only policy
  (`--read-only` narrows it to `fs.write:"off"`) refuses `remember`, `delete`
  and `flush` with `E_PERMISSION_DENIED` unless the policy says
  `memory.write:"on"`. Keys come from `apiKeyEnv`/`apiKeySecret` through
  `sofuu.policy.secret()`, and no error message ever echoes a header.
- **Typed failures**: `E_NETWORK`, `E_AUTH` (401/403), `E_NOT_FOUND`,
  `E_HTTP`, `E_ADAPTER`, `E_ADAPTER_UNSUPPORTED`, `E_DIM`, `E_EMBED`,
  `E_PERMISSION_DENIED`, `E_RECORD`. A dead endpoint rejects; it never
  crashes the runtime.
- **Boot is unchanged.** The adapter layer lives in
  `src/js/memory_adapters.js` and is compiled on the first non-native
  `open()` (`rt/memory_adapters.rs` installs the loader; nothing is evaluated
  at boot). Keeping it inside `memory_api.js` cost **1.56 ms** of parse on
  every `sofuu eval`; lazy loading measured **+0.17 ms** (inside noise) on an
  80-run interleaved A/B against the pre-change binary.

Conformance: `crates/sofuu-core/tests/memory_adapters.rs` (local-only, like
the rest of `crates/sofuu-core/tests/` per `.gitignore`) boots the real
runtime against an in-process fake backend that speaks all three dialects and
runs 47 checks — round-trip and exact-hit scoring on each adapter, tag and
metadata survival, equality filtering, flat-API routing, the custom-adapter
contract (it must receive `{text, metadata}` and no forced vector), every
typed failure above, the three policy gates, and a native QTSQ round-trip
proving the default path did not change. `make bc-check` is 11/11 (both
memory sources carry `@sofuu-source-only`; they are evaluated as source, not
shipped as bytecode).

Not in this pass: `agent.define({memory: brain})` (the plan's DX sketch) is
not wired — the agent's memory tools still drive the on-device brain, and an
injected async adapter would need an awaited recall inside the loop.
pgvector counting uses `GET` + `Prefer: count=exact` + `Content-Range` rather
than `HEAD`, because the fetch bridge drives curl through `CUSTOMREQUEST` and
a HEAD answers CURLE_GOT_NOTHING (52).

## R7 security firewall: honest threat model, hardened FFI, policy DX, supply chain — 2026-10-30

The R7 evaluator already existed; this pass makes it *verifiable* and closes
the gaps an independent review found.

**Trust documents.** `THREAT_MODEL.md` states what R7 does and — explicitly —
what it does not: plugins and sandbox bridges are trusted code, prompt
injection can still cause *allowed* actions, SSRF guards are layered and
configurable per surface, redaction is key-based, symlink TOCTOU remains.
`SECURITY.md` gains scope, response SLAs, an honest "no PGP key published yet"
statement and a no-bounty statement. The landing page's "preventing agent
hallucination escapes" claim is replaced with what the evaluator actually
guarantees, and its sandbox snippet now shows the real `sofuu.sandbox.run`
options (the old example had invented `fs`/`net` keys).

**FFI/SIMD boundary (`rt/view_guard.rs`).** One validator re-derives every
`Float32Array` window before a pointer reaches a kernel: 4-byte alignment,
`offset + length ≤ buffer size` with checked arithmetic, and an optional exact
dimension. `sofuu.ai.similarity/dot/l2` (`rt/ai.rs::extract_f32`) and the
memory/kv lane (`rt/memory.rs::cma_vec_view|arg`) now both go through it;
zero-length views are accepted on the vector path and refused, with a coded
error, where a bare pointer would read as "already threw". A new cargo-fuzz
target (`fuzz_simd_bridge`) asserts the accepted window is always a subset of
the buffer, and `.github/workflows/asan.yml` runs the boundary + policy tests
and a fuzz smoke under AddressSanitizer on Linux nightly.

**Policy DX.** `sofuu policy init [--profile default|strict|offline]` writes a
profile (absolute project root, atomic refusal to overwrite without `--force`);
`policy check <script>` statically audits fetch/fs/shell/env/process.env/MCP
requirements against the active policy and exits non-zero on a denial (a
`.json` argument still runs the historical probe table); `policy dry-run`
prints the would-be decisions; `policy why` now answers under the layered
on-disk policy instead of an empty one. Two evaluator keys were added:
`shell.deny` (full-command globs, checked before `shell.allow`) and
`net.enabled` (the `--no-net` floor). `sofuu run` and `sofuu chat` accept
`--read-only`, `--no-net`, `--confirm-all`; `run` accepts `--policy` for the
first time. `scripts/stress/policy_dx_e2e.sh` (25 gates) proves each flag at
the real gate, not just in the evaluator.

**Supply chain.** `scripts/sbom.py` emits a CycloneDX 1.5 SBOM from
`cargo metadata` (stdlib only, 500+ components with licences) and
`make sbom` writes it to `dist/`; the release workflow validates and attaches
it. `scripts/sign_release.sh` signs `SHA256SUMS` with minisign and verifies
its own output against the published key; `install.sh` verifies the signed
manifest before unpacking — but only once a public key is published. Until
then it says so out loud, and no unsigned signature file is ever emitted.

## Boot latency: HTTP initialization moved to the first request — 2026-10-30

With the GPU probe gone, the largest item left in engine boot was libcurl.
Two modules paid for it unconditionally: `mod_http_client_register` ran
`curl_global_init` (which loads the TLS backend and trust store, 2.7–4.6 ms
on macOS) and built a multi handle that no script might ever use, and
`mod_ai_register` called `ai_ensure_init()` — doing the same work a second
time — so every `sofuu eval` and `sofuu run` carried ~6 ms of HTTP setup
before user code ran.

- **`curl::ensure_global_init()`** — a `Once`-guarded `curl_global_init`.
  `curl_easy_init()` calls it, which covers all three users of easy handles
  (fetch, providers, npm), and both module initializers call it too.
- **`http_client`** creates its multi handle and timeout timer in
  `ensure_multi()`, invoked at the top of `js_sofuu_fetch` — the only entry
  point that starts a transfer.
- **`ai`** — `mod_ai_register` no longer calls `ai_ensure_init()`; the four
  request entry points (`complete`, `stream`, `embed`, audio) already did.
- **Teardown** — `ai_abort_requests_for_ctx` used to end with an
  unconditional `curl_multi_socket_action`, which with AI initialization
  deferred is `NULL` and faults. It now returns early when no multi exists
  (the abort list is empty on exactly that path). Regression test:
  `abort_requests_without_curl_init_is_a_no_op`.

Measured on Apple M2 Pro / macOS 15.7.3, same scripts in both arms,
interleaved runs so both binaries see the same machine load:

| Command | Before | After | |
|---|---|---|---|
| `sofuu eval "1+1"` (quiet window) | 17.3 ms | **11.4 ms** | 1.52× |
| `sofuu eval "1+1"` (loaded window) | 20.8 ms | **14.3 ms** | 1.46× |
| `hello.js` (as a compiled artifact) | 21.5 ms | **14.7 ms** | 1.46× |
| `sofuu version` (no engine) | 4.9 ms | 4.9 ms | unchanged — already the floor |
| `sofuu capabilities` | 38.2 ms | 38.9 ms | unchanged — it *should* probe the GPU |

The absolute medians track machine load (two windows the same day differ by
~3 ms); the ~6 ms delta is consistent across both. Together with the GPU
entry below, `sofuu eval "1+1"` went from ~33 ms to ~11 ms on an idle
machine. Verified after the change: `fetch` over TLS (200 + body), a DNS
failure, an AI request to a dead endpoint, a compiled artifact doing
`fetch`, and `bc-check` 11/11. The remaining boot cost is the eleven driver
snapshots (~4.2 ms) plus native registration (~1.6 ms); deferring driver
evaluation was evaluated and rejected — the drivers install ~30 top-level
names that collide with native ones, so lazy loading would change what
`sofuu.*` resolves to at boot.

## Boot latency: the GPU probe is off the start path — 2026-10-30

`sofuu eval "1+1"` spent ~29 ms of its 33 ms building the engine, and ~20 of
those milliseconds were a Metal device. The chain was
`engine_register_builtins` → `mod_memory_register` → `eval(src/js/memory_api.js)`
→ `sofuu.capabilities()` → `manifest_json()` → `metal_available()` →
`MTLCreateSystemDefaultDevice()`: the flat memory API asked the whole
capability manifest for one boolean, and the manifest asks the OS for a GPU
device — which costs 18–20 ms the first time in a process.

- **`sofuu.build`** — the manifest's `build` block (features, missing,
  platform_absent, revision) is now registered at startup from `build_block()`,
  a pure function of the compilation with no probe behind it. `manifest()` is
  built from the same function and a unit test pins the two equal, so the
  boot-time facts and the manifest cannot drift apart.
- **`memory_api.js`** reads `sofuu.build.missing` instead of
  `JSON.parse(sofuu.capabilities())`. The guard's intent is unchanged (do not
  install the flat API over typed-refusal stubs on a QTSQ-free build); it just
  no longer creates a GPU device to learn a compile-time fact.
- **The Metal probe is cached** (`OnceLock`): a device cannot appear or vanish
  mid-process, so a second manifest build in the same process is now free
  instead of re-creating it.

Measured on Apple M2 Pro / macOS 15.7.3 — same script in both arms, 40
interleaved runs so both engines see the same machine load:

| Command | Before | After | |
|---|---|---|---|
| `hello.js` (as a compiled artifact) | 34.4 ms | **17.8 ms** | 1.93× |
| `sofuu eval "1+1"` | 33.3 ms | **17.1 ms** | 1.95× |
| `sofuu version` (no engine) | 4.4 ms | 5.0 ms | unchanged (noise) |
| `sofuu capabilities` | 26.8 ms | 26.3 ms | unchanged — it *should* probe the GPU |

Compiled `sofuu compile` artifacts run the same boot path and inherit the win.

What is left, and why it was not taken: ~4.2 ms evaluating the eleven shipped
drivers and ~1.8 ms parsing `compat.js` from source, on top of ~2.3 ms of
QuickJS setup and native registration. Deferring the drivers needs a lazy
accessor for every name they install — about 30 top-level properties
(`sofuu.agent`, `sofuu.tools`, plus `S.cfg`/`S.session`/`S.info`-style
aliases), several colliding with native or Rust-set names such as `VERSION` —
where one mis-installed accessor would shadow a native and break chat/serve
silently. `compat.js` belongs to the compatibility lane; the cheap follow-up
there is to ship it as `compat.bc.gz` like every other driver (~1.5 ms).

## Standalone Agent Compiler (`sofuu compile`) — 2026-10-30

Implementation of `STANDALONE_COMPILER_PLAN.md`: an agent becomes one
self-contained executable — the `sofuu` binary with its bundled program and,
optionally, its brain appended behind a 32-byte trailer.

- **Command** — `sofuu compile <entry.ts|js> [-o <out>] [--embed-brain
  <file.qtsq>] [--no-strip]`. The entry is bundled (`bundler`), compiled to
  QuickJS bytecode with a bare compile-only runtime, gzipped when that saves
  bytes, and staged as `payload = header + entry name + bytecode + brain`.
- **Boot** — the CLI inspects the trailer before any dispatch: a plain binary
  pays two reads, a compiled one runs its program immediately with
  `process.argv` forwarded and `sofuu.args` set to the words after the binary.
  `SOFUU_PLAIN=1` opts back into CLI mode (inspect or recompile an artifact).
- **Exit semantics** — identical to `sofuu run`: a synchronous throw, an
  unhandled rejection and `process.exitCode` mean what they mean there, because
  the compiled path reuses the same precedence and loop drain.
- **Embedded brain** — a store is a manifest plus `.d/` segments, so
  `--embed-brain` packs both (graph snapshot included) into one archive,
  records the vector dimension at compile time, extracts the seed beside the
  binary on first launch, and exposes
  `sofuu.memory.embedded = {path, dim, open()}`. An existing extracted store is
  never overwritten: the payload is a seed, not a reset.
- **Mach-O** — appending bytes to an arm64 binary invalidates its signature and
  fails strict validation, so the artifact is assembled as: remove signature →
  strip any earlier payload (recompiles cannot nest; three generations are
  byte-identical) → append → extend `__LINKEDIT` over the payload → ad-hoc
  re-sign. The trailer therefore does not sit at EOF on macOS; boot finds it
  just before `LC_CODE_SIGNATURE.dataoff`, tolerating codesign's 16-byte
  alignment padding. ELF and PE keep the simple append.
- **Verified** — hello and TypeScript/async parity with `sofuu run`; an
  argv/exit-code matrix (0/1/7); embedded-brain recall from an unrelated cwd;
  corrupt-payload refusal (CRC32, exit 1); size-stable recompiles; cold start
  33.5 ms vs 33.3 ms for `sofuu run` (payload overhead below noise). 16 unit
  tests in `compiler.rs`; conformance stays 51 PASS / 0 FAIL / 1 BLOCKED.
- **Known gate** — `make size-check` was already red before this change
  (5,938,720 B vs the 5,767,168 B cap; this feature adds ~18 KB). Reducing the
  binary is a separate work item, not something this feature regressed.

## Universal Sandbox v2 — 2026-10-29

Implementation of `SANDBOX_UPGRADE_PLAN.md`: the in-process execution sandbox
stops clamping developer budgets, gains callable host bridges, TypeScript/JSX
ingestion, prototype hardening and configurable console routing.

- **Unclamped budgets (I2)** — `SandboxConfig::clamp()` now enforces only
  floors (1 MB memory, 16 KB stack); a stated `timeoutMs`, `fuel`, `maxMemoryMB`
  or `maxStackKB` is never reduced. `fuel: 'unlimited' | 0 | null` disables the
  instruction meter and polling drops to `i32::MAX` between deadline checks;
  `timeoutMs: 0 | null` disables the deadline. The misleading
  `SANDBOX_MAX_FUEL` / `SANDBOX_MAX_TIMEOUT_MS` constants are gone.
- **Presets** — `strict` (2 s, 5M fuel, 32 MB, frozen prototypes), `standard`
  (10 s, 20M, 64 MB), `compute` (300 s, unmetered, 512 MB) and `worker`
  (no deadline, unmetered, 256 MB). Explicit options override the preset; an
  unknown preset name is refused. Numeric options are type-checked rather than
  coerced (`timeoutMs: "soon"` used to become a zero budget through
  `ToNumber`).
- **Callable host bridges** — a top-level function in `globals` is installed as
  a native function in the sandbox realm. Arguments and results cross as JSON;
  a sync host function returns directly, a promise-returning one gives the
  sandbox a real promise. Microtask-only promises settle inside the run; a
  promise that needs the host event loop (fetch, fs) **parks** the run —
  `run`/`eval` then return a Promise, the session reports `E_BUSY` until it
  resumes, and `close()` is deferred to completion. Host throws become
  catchable sandbox errors; a non-serializable result is a typed error.
- **TypeScript without a build step** — `ts: 'auto'` evaluates the source as
  written and retries the stripped source only when the first parse failed, so
  a stripper bug cannot change valid JavaScript. The Rust stripper gained
  **enum** support (rewritten to the same frozen object literal the plugin
  driver emits, including `const`/`export`/`declare` forms), which also fixes
  enums in `sofuu run file.ts`. Top-level `await` is supported through an
  async wrapper (documented: the value comes from `return`).
- **JSX** — `jsx: true` transpiles to `h()` calls without rewriting imports
  and installs `h`/`Fragment` in the realm, so components evaluate to a
  `{type, props, children}` tree.
- **Prototype hardening** — `freezePrototypes` freezes the intrinsic
  constructors and their prototypes at realm creation (`strict` preset
  default), so `Object.prototype.x = ...` cannot stick across turns.
- **Console routing** — `console: 'capture' | 'inherit' | 'ignore' | fn` plus
  `maxLogs` (0 = unbounded, 8 MB total / 64 KB per line ceilings). The old
  `captureConsole` boolean still maps onto it.
- **Typings** — `sofuu.d.ts` declares the `sofuu.sandbox` namespace,
  `SandboxOptions`, `SandboxResult`, `SandboxSession`, `SandboxPreset` and
  `SandboxErrorCode`. Two codes join the taxonomy: `E_BUSY` and `E_BRIDGE`.
- **`code_eval`** awaits the sandbox run (a run only becomes a Promise when a
  bridge parks) and therefore accepts TypeScript by default.
- **Gates** — `make sandbox-stress` runs 26 adversarial cases
  (`scripts/stress/sandbox_stress.js`: budget escapes, memory/stack bombs,
  catastrophic regex, host-global probes, prototype pollution, TS/JSX, bridges
  over real I/O, parked-run lifecycle, presets, 2000-run soak). Plus
  `tests/test_sandbox_v2.js` (10 acceptance groups) and 21 Rust unit tests.
  Docs: `docs/SANDBOX.md`.

## 24/7 Agent Observability, Tracing & Recovery — 2026-10-29

Implementation of `OBSERVABILITY_PLAN.md`: a durable per-run trace, a declared
failure policy, crash post-mortems, and checkpointed resume.

- **One envelope, one writer** — agent runs append the §7.2 envelope to
  `.sofuu/logs/agent-<run_id>.jsonl`; chat keeps
  `.sofuu/sessions/<sid>/events.ndjson`. Same validator, same secret
  redactor, same `rt/events.rs` append path — only the directory differs.
  `run_id` is accepted as an alias for `sid`, so a plan-shaped event
  validates exactly as written. Thirteen kinds join the registry
  (`run_start`, `turn_start`, `tool_call`, `model_complete`,
  `memory_recall`, `memory_store`, `checkpoint`, `warning`,
  `fatal_error`, `retry`, `halt`, `run_complete`, `stream_chunk`).
- **Durability** — every batch is flushed, and a batch carrying a
  `checkpoint`, `halt`, `fatal_error`, `run_complete` or `stop` is
  `fsync`ed and reports `synced: true`. A `kill -9` loses nothing already
  written; a power loss loses only events after the last checkpoint.
- **Concurrent writers are safe** — found by the stress suite (case A11). The
  writer issued two `write_all` calls (body, then newline) and `O_APPEND`
  only makes a *single* write atomic, so a second process could land between
  them and glue two records onto one line — 19 of 200 appends were lost while
  every writer reported success. Writes are now one syscall per line, and a
  per-batch `flock` makes read-tail-then-append atomic, so `seq` stays unique
  across processes. `last_seq` reads a 64 KB tail instead of the whole log,
  which removes a quadratic re-read from every append.
- **Error policy** — `--on-error=auto-retry|fail-stop|interactive` (also
  `SOFUU_ON_ERROR`), defaulting to `interactive` on a TTY and `auto-retry`
  otherwise; an unknown mode is refused rather than silently meaning the
  default. `auto-retry` reuses the existing stream ladder with full jitter and
  now also covers tool faults; five consecutive classified errors trip the
  circuit (`E_CIRCUIT_OPEN`); `interactive` degrades to `fail-stop` with a
  `warning` event when there is no TTY. `[e]dit` in `$EDITOR` is out of scope
  for v1 and documented as such.
- **Post-mortems** — `.sofuu/crashes/<crash_id>.json` carries the plan's card
  shape (failed step, error code and message, stack, context snapshot,
  recovery hint, trace path); `sofuu logs --last-error` renders it.
- **Checkpoint and resume** — a checkpoint is written after every completed
  turn, including turns that never entered the tool loop (otherwise resume has
  nothing to rehydrate for the most common run shape), and it is capped so a
  long run checkpoints its recent window. `sofuu resume <run_id> [--step=N]
  [--force]` rehydrates the newest *valid* checkpoint — skipping a torn tail
  line — reloads `~/.sofuu/agents`, adopts the checkpoint's model and
  provider, and continues from the next step. A completed run is refused
  without `--force`.
- **CLI** — `sofuu logs`, `logs show <id>`, `logs --follow`,
  `logs --last-error`, `sofuu resume`. Rendering lives in
  `src/js/observability.js` and ships as compressed bytecode; `main.rs` stays
  a thin shim, which is what keeps the feature inside the size cap.
- **Typings** — `sofuu.d.ts` declares `sofuu.observability.*`,
  `sofuu.agent.classifyError` and the `OnErrorMode` union. The plan's proposed
  codes `E_NETWORK_TIMEOUT`, `E_SERVER_ERROR` and `E_UNCAUGHT_THROW` are
  deliberately absent: they are mapped onto the normative taxonomy
  (`E_TIMEOUT`, `E_TOOL`, `E_SCHEMA`, `E_DENIED`) so cross-surface error
  parity (R5.1) still holds.
- **Gates** — `make observability-stress` runs 13 adversarial cases
  (`scripts/stress/observability_stress.js`: redaction, seq total order,
  backpressure, `kill -9` durability, four concurrent writers, torn lines,
  crash cards) plus a 7-case CLI e2e (`tests/observability_e2e.sh`).
  Conformance stays 51 PASS / 0 FAIL / 1 BLOCKED.

## Universal Plugin & Extensibility System — 2026-10-29

Implementation of PLUGIN_PLAN.md: five plugin categories behind one runtime,
one safety model, one error taxonomy.

- **Five categories** — action tools (`defineTool`), inference providers
  (`defineProvider`), storage and vector stores (`defineStorage` /
  `defineVectorStore`), lifecycle services (`defineService`, including an
  `onToolCall` interceptor) and description-driven plugins (`useOpenAPI` for
  OpenAPI 3.x/Swagger 2.0, `ingestMcp` for MCP servers). All of them are
  reachable as `import { ... } from 'sofuu'`.
- **Five shields** — schema validation (input and output), policy and SSRF
  network jailing plus a jailed `ctx.fs`, a CLOSED/OPEN/HALF_OPEN circuit
  breaker with single-trial admission, a watchdog with abort signal, and a
  panic guard. A policy denial no longer trips the breaker, so a blocked
  target keeps reporting `E_DENIED` instead of decaying into
  `E_CIRCUIT_OPEN`. Two codes join the taxonomy: `E_CAPABILITY` and
  `E_BACKPRESSURE`.
- **Engine modularised** — `src/js/plugins.js` is now eleven modules (schema,
  security, breaker, watchdog, registry, provider, store, service, openapi,
  discovery, cli) with every 0.2.0 public name preserved.
- **Native provider adapters** — new `src/js/providers.js` ships `ollama`
  (`/api/chat` NDJSON, tool calls, usage), `llamacpp` (OpenAI wire, keyless,
  GBNF `grammar`), `openai` (Bearer key required) and the `lmstudio` /
  `vllm` local aliases, all through one shared wire parser. `completeVia` /
  `streamVia` apply the provider's own breaker and watchdog;
  `sofuu.providers.list()` reports capability, and adapter streaming is
  explicitly `streaming: 'buffered'`. Rust gains a first-class
  `Provider::LlamaCpp` (no `OPENAI_API_KEY` leakage to a local server, no
  `.gguf` hijack of an explicit llama.cpp call) and `-p ollama|llamacpp`
  works end to end.
- **SDK typings** — `sofuu.d.ts` declares the whole surface (tools,
  providers, storage/vector, services, hooks, CLI helpers, the micro-Zod
  shape) and type-checks clean under `tsc --strict`. Shipped in the npm
  package.
- **TypeScript plugins without a compiler** — a scanner-based stripper
  handles imports/exports, interfaces, type aliases, enums, generics,
  parameter/return annotations, optional parameters, parameter properties and
  class fields. It is construct-aware rather than regex-based: an object
  literal `{ units: defaultUnits }` is no longer eaten, and a regex literal
  containing quotes no longer desynchronises the scanner. Unsupported
  constructs are left intact so they fail loudly instead of silently.
- **CLI and discovery** — `sofuu plugin list|validate|run|test|init`,
  `sofuu.json` manifests, `./plugins` plus `~/.sofuu/plugins`, `init`
  templates for all six categories, and a per-file discovery report that
  isolates a broken plugin instead of aborting startup.
- **Dual-write mirror** — `attachVectorStore` / `attachStorage` mirror QTSQ
  writes into an external store behind a bounded, batched, back-off queue; the
  local write stays synchronous and `mirrorStats()`/`flushMirror()` expose
  the queue.
- **Agent seam** — `onTurnStart` may rewrite the prompt, `onToolCall` may
  veto or mutate a call, and every hook is isolated and time-boxed.
- **Cookbook and docs** — four runnable plugins in `plugins/` (`fetch_web`,
  `sqlite_query`, `ollama_provider`, `token_logger`), a user guide in
  `docs/PLUGINS.md`, a README section, and two gates: `make plugin-check`
  (runtime/typings parity plus cookbook strip proof) and `make
  plugin-stress` (25 adversarial cases across all five categories).

## 0.3.0-beta — 2026-10-28 (in development)

Mandatory QTSQ core integration, runtime conformance program completion, and general-purpose AI runtime architectural alignment:

- **General-Purpose AI Runtime Alignment & Decoupling**:
  - *Prompt Sovereignty*: Developer-provided agent system prompts are 100% sacred and never prepended with terminal coding instructions (`CORE_PROMPT`). Custom agents with omitted prompts default to a neutral assistant persona; `CORE_PROMPT` and `AGENTS.md` context injection are strictly isolated to `preset: 'coding'` (used by `sofuu chat`).
  - *Tool-Aware & Opt-In Task-Gate*: Task checklist nagging is strictly tool-aware (checked via `todo_write` in the active toolset) and opt-in via `budget.taskGate` (defaults to `false` for general agents, `true` for coding preset).
  - *Decoupled Cognitive Memory (`remember`)*: Built-in `remember` tool persists facts directly into the on-device QTSQ Cognitive Vector Memory (`cma.remember(...)` with dual indexing in `semCma`); writing to project `AGENTS.md` is strictly guarded by `preset: 'coding'` or explicit `opts.agentsMd`.
  - *Domain Toolpacks (`sofuu.tools.PACKS`)*: Reorganized tools into modular domain packs: `coding` (alias `code`), `web`, `memory`, `system`. Introduced first-class general tools: `system_info` (runtime version, OS platform, CPU architecture, GPU engine, capabilities), `memory_search` (natural language query of QTSQ vector memory), and `memory_remember`. Added `category` metadata and category filtering to `sofuu.tools.list()` and HTTP `GET /v1/tools?category=...`. Enabled direct multi-domain tool string resolution in `agent.define`.
  - *Generalized Session Documents*: Expanded canonical session document top-level keys (`DOC_KEYS`) to include `state` (application/workflow state container) and `metadata` (custom annotations). General session headers support `kind` (`'chat'`, `'agent'`, `'workflow'`, `'task'`, `'pipeline'`), `agent`, `status`, and metadata. Generalized step records support `input` and `output` aliases and structured `data` payloads. `sofuu.session` JS surface accepts direct JS objects in addition to JSON strings.
- **Mandatory QTSQ Core** — Made QTSQ memory a permanent core component of the
  runtime across all targets; retired the legacy QTSQ-free build variant.
- **Contract Conformance Unblocking (51 PASS · 0 FAIL · 1 BLOCKED)** —
  Unblocked 9 previously blocked conformance test cases across capabilities
  manifest parity, single terminal event enforcement, tier and bound declarations,
  deterministic tolerant bounds, opt-in best effort decode, forward schema migration,
  MCP envelope indistinguishability, cross-surface error taxonomy parity, and token
  usage reconciliation.
- **Single Terminal Event Fix** — Resolved agent runtime double-stop bug under
  step budget exhaustion.
- **Cross-Surface Capabilities** — Added `sofuu capabilities` CLI subcommand and
  `/v1/capabilities` HTTP routes for 100% manifest parity with JS and C ABI.

## 0.2.0-beta — 2026-10-26

The runtime-contract program, phases P0–P4:

- **P0** — the normative contract (`docs/SPEC-RUNTIME-CONTRACT.md`, local-only
  by repo policy), the 48-cell surface matrix, the 52-case conformance suite,
  and the tracking artifacts under `contract/`.
- **P1** — the capability model: `sofuu.capabilities()`, `sofuu.model.describe`,
  `sofuu doctor` hard-fails on a missing declared feature, a typed
  `E_CAPABILITY` refusal on QTSQ-free builds, determinism receipts
  (`scripts/text_gate.py`) with the R3 CI gate, and the committed fixture
  model the gate runs (`contract/fixtures/sofuu-fixture.gguf`).
- **P2** — one async contract (`sofuu.async.*`: the §1.4 taxonomy, nested
  deadlines, queue bounds) and one event envelope (`sofuu.events.*`: the §7.2
  schema, validator, redaction, renderer, the per-session `events.ndjson`
  log). Live-path chaos harnesses: deadline kill, mid-SSE cancel, mid-tool
  cancel (subprocess killed, no orphan), server-side disconnect visibility,
  bounded-pump backpressure (`E_BACKPRESSURE`), usage reconciliation
  (`sofuu.events.canonicalUsage`; `ai.complete` reports provider usage).
- **P3** — the session document (`rt/session_doc.rs`, §5.2 v1): one
  export/import format behind `sofuu.session.*` and
  `sofuu session export|import`; the canonical
  `<project>/.sofuu/sessions/<id>/session.json` written by chat;
  restart-stable per-turn ids; `schema_version` on sessions, the event
  envelope, and the config; import-side format allowlisting (R4.5) and
  export redaction (R4.6); serve `/v1/*` OpenAI-wire conformance
  (models / SSE / JSON); the C-ABI symbol gate executable in the suite
  and CI (`CONTRACT_LIBSOFUU`).
- **P4** — one security policy everywhere (R7): the four-surface policy
  matrix (JS / CLI / HTTP / C ABI — same decision, same rule id, no bypass
  by entry point), approval deadline-deny and registration-order
  guarantees in the trace, the shared shell/fs refusal shape
  (`SOFUU_NO_SHELL` strips every surface), net-policy suffix anchoring with
  normalization, secrets-never-emitted checks, the MCP allowlist gate,
  project-policy auto-install at serve boot, and install integrity:
  checksummed `dist/` artifacts + the capabilities-vs-claim + version-parity
  gate (`scripts/verify-dist.sh`).

## 0.2.0-alpha — 2026-09

Engine substrate: in-process inference (GGUF/Metal), the agent loop,
encrypted QTSQ memory, MCP client/server, the C-ABI embed surface.
