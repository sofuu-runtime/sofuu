# kaze-host-containment-provenance-size — Kaze host containment, provenance, and the `kaze_async.js` size fix

Scope of this worker: `crates/sofuu-core/src/domains/kaze/caps.rs`,
`crates/sofuu-core/src/domains/kaze/checker.rs`,
`crates/sofuu-core/src/domains/kaze/mod.rs`, `crates/sofuu-core/build.rs`,
`crates/sofuu-core/src/shipped.rs`. Division D05 (Kaze/compiler) plus the D04
policy rows that this host serves.

## Checks performed

| Check | Command / source read | Result | Evidence |
|---|---|---|---|
| R1: the runtime host has no path where `ai`, `tool`, `memory` or `alert` succeeds without a configured backend | read `crates/sofuu-core/src/domains/kaze/caps.rs:426-512` (`RuntimeHost::call`) | pass | `ai`/`tool`/`memory` fall through to the `_` arm → `E_CAP_MISSING`; `alert.send` requires a non-empty sink (`caps.rs:487-496`) else `E_CAP_MISSING`. The real memory stores live in `crates/sofuu-core/src/rt/kaze_memory.rs` (`EphemeralMemoryHost` → real CMA, `DurableMemoryHost` → real file store); neither fabricates a recall. |
| R1: the physical test host answers only the families it was explicitly selected for | read `caps.rs:546-606` (`SimHost::with_simulated`, `serves`) and `caps.rs:617-643` | pass | `with_simulated` filters to `PHYSICAL_TEST_FAMILIES` (`drone`,`rover`); `call` refuses a non-physical family and a family that was not selected (`E_CAP_MISSING`). Grant-time filter: `crates/sofuu-core/src/domains/robotics/hardware.rs:109-120` refuses `sim` for any other family. |
| R1: an unknown telemetry field is a typed error, never `0`/`false` | read `caps.rs:584-605` (`SimHost::telemetry`) and `caps.rs:695-697` | pass | An unmodelled field returns `E_CAP_MISSING: the simulated <family> has no '<field>' reading`. The live path does the same (`domains/robotics/kaze_live.rs:70-100`, unknown field → `E_CAP_MISSING`, stale field → `E_KAZE_LINK`). |
| R1: `policy.detect` refuses an unsupported detector kind | read `caps.rs:730-765` (`detect`) and `caps.rs:459-468` | pass | An unknown kind returns `E_CAP_INVALID: unsupported policy detector '<kind>'`; it is never a clean `false`. |
| R1: `policy.allow`/`policy.reject` stay the VM's own control flow | read `caps.rs:470-474` and `domains/kaze/vm.rs:930-946` | pass | The VM consumes the verdict before any host is consulted; the host refuses to answer one (`E_CAP_INVALID: policy verdicts are the VM's own control flow`). |
| R1: no absent backend and no unmatched family delegates to a general simulator | read `domains/kaze/mod.rs:625-674` (`build_cap_host`), `domains/kaze/mod.rs:482-523` (`resolve_backends`), `domains/kaze/mod.rs:527-533` (`simulated_families`) | pass | The simulator is handed only `simulated_families(selection)`, i.e. families whose backend is literally `sim`; `validate_grant` already refuses `sim` outside `drone`/`rover`. `domains/robotics/kaze_live.rs:534` builds its fallback host with an empty simulator list. |
| R1: a physical test grant cannot hand a fake device to unrelated workloads | read `caps.rs:445-458`, `caps.rs:563-568`, `domains/kaze/mod.rs:537-551` | pass | `sensor` and `device` have no simulator arm at all; a live `drone`/`rover`/`sensor`/`device` grant is listed in `blocked_live_families` and refused with `E_CAP_DENIED` rather than answered. |
| R1: the checker does not open a simulation door | read `crates/sofuu-core/src/domains/kaze/checker.rs` in full | pass | The checker only reasons about declarations, arity, units and filter verdicts; it has no backend concept, so it cannot grant one. No change needed in this file. |
| R4.1: the mode and the backend appear in the trace, the CLI output and the structured result | ran the render path; read `caps.rs:151-175` (`Action::render`), `domains/kaze/mod.rs:1243-1278` (`report_to_js`) | pass (after this change) | Every trace line now ends in `[<mode>: <backend>]`, e.g. `drone.arm -> nothing [simulated: sim]`, `log.info(<redacted>) -> nothing [real: local-log]`, `drone.arm -> error: E_CAP_MISSING: … [real: none]`. `report.actions` feeds the CLI text output (`script_cmd.rs:1120`), the `--plain` output (`mod.rs:573-599`), the CLI JSON `actions` array (`script_cmd.rs:1149`) and the JS `actions` array (`mod.rs:1260-1265`). The JS result additionally carries the resolved `backends` array (`family`/`backend`/`simulated`, `mod.rs:1251-1259`). |
| R4.1: a MIXED run names the mode and the backend of every action | new test `domains/kaze/mod.rs:204` (`a_mixed_run_names_the_mode_and_the_backend_of_every_action`) | type-checked, **not executed** (build policy) | The test runs `needs: drone, log` with `drone=sim` (simulator + real local log sink) and then `needs: drone, alert` with no alert sink, asserting `E_CAP_MISSING` for `alert.send` — the simulator never answers another family. |
| Size fix: `kaze_async.js` is staged gzipped and embedded compressed | `crates/sofuu-core/build.rs:369` (added to the shipped-asset list); `crates/sofuu-core/src/shipped.rs:137` (`KAZE_ASYNC_JS_GZ`); `crates/sofuu-core/src/shipped.rs:143` (`kaze_async_js()`, `OnceLock` cache) | pass | The staged payload exists at `target/debug/build/sofuu-core-401f1da4b8b4b844/out/kaze_async.js.gz` (5,535 B); inflating it reproduces `src/js/kaze_async.js` byte for byte (18,884 B). Expected binary saving ≈ **13,349 B** (18,884 → 5,535). |
| Size fix: the driver still evaluates into the Kaze namespace with the same error handling | read `domains/kaze/mod.rs:1914-1930` | pass | Same `JS_Eval` with the same `<kaze-async-driver>` name and the same `js_std_dump_error` on exception; only the source acquisition changed (inflated once instead of embedded raw). `src/js/kaze_async.js` was not modified. |
| Size fix: the staged gzip cannot drift from the source | added `("kaze_async.js", include_str!("../../../src/js/kaze_async.js"))` to the existing drift test in `crates/sofuu-core/src/shipped.rs:397` | type-checked, **not executed** | The existing test `gzipped_shipped_assets_match_their_sources` now covers the new asset. |
| The edits compile | `cargo check -p sofuu-core --lib --tests` | pass | `Finished dev profile … in 9.63s`; only pre-existing warnings in `domains/infer/metal.rs`, `domains/infer/model.rs`, `domains/memory/cma.rs`. No new warnings from the changed files. |
| The Rust test suite | — | **not run** (the coordinator owns the build; `cargo test` is out of policy for this worker) | The new and updated unit tests are type-checked only. They must be executed by the coordinator. |

## Gaps found and fixed

| Gap | Fix | Path:line |
|---|---|---|
| `kaze_async.js` (18,884 B) was embedded raw with `include_str!` in release builds, while every other shipped JS asset is gzipped by `build.rs` and inflated once | `build.rs` stages `kaze_async.js` gzipped into `OUT_DIR`; `shipped.rs` embeds the `.gz` and exposes `kaze_async_js()` behind a `OnceLock` (process-lifetime cache, inflated at the evaluation site); `mod.rs` evaluates the inflated source with the same name, the same `JS_Eval` flags and the same error handling | `crates/sofuu-core/build.rs:369`, `crates/sofuu-core/src/shipped.rs:137-152`, `crates/sofuu-core/src/domains/kaze/mod.rs:1916-1928` |
| The trace line carried no mode and no backend, so a simulated `drone.arm` and a real `drone.arm` were indistinguishable in the trace, in the CLI output and in the CLI JSON | `Action::render()` appends `[<mode>: <backend>]`; an empty backend renders as `none`, so a refused call says that nothing answered | `crates/sofuu-core/src/domains/kaze/caps.rs:151-175` |
| The provenance change was not covered by a test, and no test covered a mixed run | new test for the rendered provenance of a real and a simulated action, a test for the "no backend reached" line, and a mixed-run test (simulator + real log sink, then a family with no backend that must fail) | `crates/sofuu-core/src/domains/kaze/caps.rs:1097-1131`, `crates/sofuu-core/src/domains/kaze/mod.rs:204-236` |
| The JS structured result asserted the family-level provenance but not the per-action provenance | the JS test now also asserts that the action line names the mode and the backend | `crates/sofuu-core/src/domains/kaze/mod.rs:2683-2712` (inside `the_js_report_names_the_backend_that_served_the_run`) |

## Register rows

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D05 Kaze and compiler | Runtime host (`crates/sofuu-core/src/domains/kaze/caps.rs`): `ai`, `tool`, `memory`, `alert` without a configured backend | `real` | No path reports success: `ai`/`tool`/`memory` fall through to `E_CAP_MISSING` (`caps.rs:497-501`); `alert.send` needs a configured sink (`caps.rs:487-496`). Verified by reading the dispatch, not by a fresh run in this worker. |
| D05 Kaze and compiler | Physical test host (`SimHost`): only explicitly selected `drone`/`rover` families | `physical-test` | `SimHost::with_simulated` filters to `PHYSICAL_TEST_FAMILIES` and `serves` requires the explicit selection (`caps.rs:562-579`); `hardware::validate_grant` refuses `sim` for any other family (`domains/robotics/hardware.rs:109-120`). A simulator result is never evidence about hardware. |
| D05 Kaze and compiler | Unknown telemetry field on a simulated vehicle | `physical-test` | `E_CAP_MISSING: the simulated <family> has no '<field>' reading` (`caps.rs:600-604`) — never `0`/`false`. The live path refuses unknown and stale readings the same way (`domains/robotics/kaze_live.rs:70-100`). |
| D04 tools, MCP, policy | `policy.detect` with an unsupported detector kind | `real` | `E_CAP_INVALID: unsupported policy detector '<kind>'` (`caps.rs:760-763`) — a detector that does not implement a kind never returns a clean verdict for it. |
| D04 tools, MCP, policy | `policy.allow` / `policy.reject` | `pure` | VM control flow: the verdict is consumed in `domains/kaze/vm.rs:930-946` before any host is consulted, and the host refuses to answer one (`caps.rs:470-474`). |
| D05 Kaze and compiler | Provenance of a simulated-vs-real action in the trace, the CLI output and the structured result | `real` | Every action line carries `[<mode>: <backend>]` (`caps.rs:151-175`); the JS result also carries the resolved `backends` array (`mod.rs:1251-1259`). Mixed-run coverage: `mod.rs:204-236` (type-checked, not executed here). |
| D05 Kaze and compiler | Shipped JS asset size: `kaze_async.js` embedded raw in release builds | `real` | Staged gzipped by `build.rs:369`, embedded as `.gz` (`shipped.rs:137`), inflated once behind a `OnceLock` (`shipped.rs:143`) and evaluated at `mod.rs:1916`. 18,884 B → 5,535 B in `.rodata`, a saving of ≈13.3 KB. The inflated bytes are identical to the committed source (verified against the staged `OUT_DIR` payload). |
| D07 robotics and drones | Real vehicle hardware behind the Kaze host | `unverified` | No device attached; live drone/GPIO dispatch stays disabled (`E_CAP_DENIED` from `blocked_live_families`), and `LiveHost` is not wired into `run_prepared` — it is exercised only by its own unit test. |

## Blocked or unverified

- **The Rust test suite was not executed in this worker** (build policy: the
  coordinator owns the build). The new and updated tests in `caps.rs` and
  `mod.rs` are type-checked only (`cargo check -p sofuu-core --lib --tests`
  → pass). The coordinator must run them before the change is considered
  green.
- **Real hardware coverage is unverified** — no drone, rover or GPIO device was
  attached. The live paths stay disabled; nothing in this change enables them.
- **The `ai`/`tool` asynchronous path names the serving host, not the
  provider.** `RuntimeHost::note_async_completion` records those calls with
  the backend `local` (`caps.rs:422-429`), because the resume protocol carries
  only the result value: the provider identity lives in the JS bridge
  (`src/js/kaze_async.js`, out of scope). The mode is `real` and no simulator
  can serve that path, so the simulated-vs-real provenance is intact; naming
  the provider needs a protocol change (bridge → resume → host note).
- **The `memory` family records its actions with the generic backend name
  `local`** (`crates/sofuu-core/src/rt/kaze_memory.rs` calls
  `RuntimeHost::note`); the store identity is in the result text ("per-run
  ephemeral CMA store" / the durable store path). Naming the store in the
  backend field needs a one-line change in that file, which is outside this
  worker's scope.
- **The golden corpus is stale and is local-only.** `examples/` is
  Git-ignored (`.gitignore:147`), so `examples/kaze/golden/*.txt` and the
  example scripts are untracked. Two independent reasons make the current
  goldens unobtainable: (a) the trace format now carries `[mode: backend]`,
  and (b) the fail-closed host means `patrol.kaze` needs an explicit
  `drone=sim` backend, while `greenhouse.kaze` (sensor/device/alert) has **no**
  selectable backend at all — `hardware::live_backends` only knows `drone`, and
  `sim` is drone/rover only — so that example cannot run today and its golden
  trace cannot be regenerated. See the notes below.

## Notes for the coordinator

1. **Shared-file changes needed (not made here).**
   - `crates/sofuu-core/src/script_cmd.rs`: the CLI JSON report has no
     `backends` array, because `RunReport` has no field for the resolved
     backends. Adding `pub backends: Vec<BackendSelection>` to
     `crates/sofuu-core/src/domains/kaze/mod.rs:556` requires updating the
     struct literal at `script_cmd.rs:989`, `report_from_bridge` (which already
     receives them from the JS report), `report_json` (`script_cmd.rs:1144`)
     and `print_report` (`script_cmd.rs:1106`). The per-action provenance is
     already in `report.actions`, so this is an addition, not a fix.
   - `crates/sofuu-core/tests/kaze_language.rs` (out of scope): the golden
     test needs `backend: vec![("drone".into(), "sim".into())]` (and the same
     for `rover`) in the `RunOptions` it builds, because a bare grant no longer
     selects a simulator. `greenhouse.kaze` has no runnable backend at all, so
     that example must either be retired from the golden corpus or recorded as
     a blocked coverage gap.
   - `examples/kaze/golden/*.txt`: regenerate with the rebuilt binary
     (`sofuu script run … --plain`) once the above is settled. Because
     `/examples/` is Git-ignored, this corpus cannot be a clean-checkout gate
     — an R6 item, not a regression from this change.
2. **Trace format change is intentional.** `Action::render()` now appends
   `[<mode>: <backend>]`. Every consumer I found uses `contains`, not equality
   (`crates/sofuu-core/src/daemon_cmd.rs:1163-1170`,
   `crates/sofuu-core/src/compiler.rs:2406-2521`,
   `crates/sofuu-core/tests/kaze_language.rs`), so the shape is compatible; the
   only equality assertions were the three updated in this change
   (`caps.rs:856-857`, `mod.rs:2338-2344`, `mod.rs:2460`).
3. **The `kaze_async.js` bytecode snapshot was left alone.**
   `src/js/kaze_async.bc.gz` exists (6,073 B) but the driver is still
   evaluated from source, as it was before this change: using the snapshot
   would alter behaviour (and the snapshot could be stale against the source).
   The source path now costs 5,535 B instead of 18,884 B, so the snapshot is
   not needed for the size goal.
4. **No files outside my scope were touched.** `git status` shows only
   `crates/sofuu-core/build.rs`, `crates/sofuu-core/src/shipped.rs`,
   `crates/sofuu-core/src/domains/kaze/caps.rs` and
   `crates/sofuu-core/src/domains/kaze/mod.rs` changed by this worker
   (`checker.rs` needed no change: it has no backend concept and opens no
   simulation door). No commit, no push, no PR; `contract/CAPABILITY-REGISTER.md`
   was read but not edited.
