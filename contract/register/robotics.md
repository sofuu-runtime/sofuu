# robotics-safety-gate-and-host-containment — Robotics safety gate, simulator containment, and telemetry freshness

Scope of this audit: `crates/sofuu-core/src/domains/robotics/safety.rs`,
`crates/sofuu-core/src/domains/robotics/drone_ops.rs`,
`crates/sofuu-core/src/domains/robotics/hardware.rs`,
`crates/sofuu-core/src/domains/robotics/kaze_live.rs`,
`crates/sofuu-core/src/domains/robotics/mavlink_gate.rs`,
`crates/sofuu-core/src/domains/robotics/mavlink_link.rs`,
`crates/sofuu-core/src/domains/robotics/mavlink_js.rs`,
`crates/sofuu-core/src/domains/robotics/gpio.rs`,
`crates/sofuu-core/src/domains/robotics/mod.rs`. Division D07 (robotics and drones).

## Checks performed

| Check | Command / source read | Result | Evidence |
|---|---|---|---|
| R4: Extraction and execution of unified physical safety rules in `safety.rs` | `cargo test -p sofuu-core --lib -- domains::robotics::safety` | pass | 11 passed, 0 failed. Covers link-alive, disarmed refusal, unknown-arming distinction, stale telemetry (>1500 ms), missing telemetry, geofence radius bounds, missing home position, recovery command safety bypass, and backend-agnostic facts. |
| R4: Telemetry freshness unified at 1500 ms | read `safety.rs:25` (`TELEMETRY_MAX_AGE_MS = 1_500`) and `mavlink_gate.rs:27` (`TELEMETRY_MAX_AGE_MS = 1_500`) | pass | Unified window across Kaze capability host and MAVLink protocol gate. Both reject stale telemetry exceeding 1500 ms. |
| R4: Safe recovery commands not blocked by telemetry loss | `cargo test -p sofuu-core --lib -- domains::robotics::safety::tests::a_recovery_command_is_not_blocked_by_stale_telemetry` | pass | `land`, `rtl`, `stop`, and `hold` still reach the vehicle when telemetry is stale or unknown so vehicles are brought down safely. |
| R4: Physical test simulator explicitly selected and labelled | read `hardware.rs:109-120` (`validate_grant`) and `caps.rs:562-579` | pass | Only `drone` and `rover` accept `sim`. Any bare `--cap` or empty backend is rejected. Non-robotics overrides (`--sim` AI/tools) are removed. |
| R4: Simulator answers only physical families; unmatched families refuse | `cargo test -p sofuu-core --lib -- domains::robotics::kaze_live` | pass | Live and simulated hosts refuse unmatched families with `E_CAP_MISSING` and never route back to a general-purpose simulator. |
| R4: Mixed runs carry real backends for non-robotics operations | read `domains/kaze/mod.rs:204-236` | pass | In mixed runs (`drone=sim` + `log=local-log`), physical commands reach the simulator while non-robotics operations (`alert`, `ai`, `tool`) require their real backends or fail closed. |
| R4: Provenance in traces and structured results | read `Action::render()` in `caps.rs:151-175` | pass | Every action carries `[<mode>: <backend>]` (`[simulated: sim]` or `[real: <backend>]`). |
| R4: Full robotics suite executed | `cargo test -p sofuu-core --lib -- domains::robotics` | pass | 151 passed; 0 failed; 0 ignored; finished in 0.21s. |
| R4: Real vehicle hardware coverage | read `hardware.rs:135` (`blocked_live_families`) and `sofuu-map.md` | unverified | No physical vehicle or drone was attached. Live hardware dispatch stays disabled (`E_CAP_DENIED`). Zero claims of physical hardware validation. |

## Suite Summary

- Executed `cargo test -p sofuu-core --lib -- domains::robotics`:
  - `domains::robotics::safety`: 11 passed, 0 failed.
  - `domains::robotics::mavlink_gate`: 56 passed, 0 failed.
  - `domains::robotics::mavlink_link`: 41 passed, 0 failed.
  - `domains::robotics::mavlink_js`: 19 passed, 0 failed.
  - `domains::robotics::kaze_live`: 13 passed, 0 failed.
  - `domains::robotics::gpio`: 6 passed, 0 failed.
  - `domains::robotics::drone_ops`: 5 passed, 0 failed.
  - **Total:** 151 passed, 0 failed.

## Register rows

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D07 robotics and drones | Physical safety gate (`domains/robotics/safety.rs`) | `real` / `pure` | 11 unit tests in `safety.rs` passed. Unified 1500 ms telemetry window, link status check, arming gate, and geofence boundary checks executed against unified `VehicleState`. |
| D07 robotics and drones | Physical test simulation (`drone=sim`, `rover=sim`) | `physical-test` | Explicit selection required; answers only physical families; provenance tagged as `[simulated: sim]`. |
| D07 robotics and drones | Telemetry freshness window | `physical-test` / `real` | 1500 ms window strictly enforced; stale or missing telemetry produces `E_KAZE_LINK` before flight actuation. |
| D07 robotics and drones | Recovery commands under dead link or stale telemetry | `real` / `pure` | `land`, `rtl`, `stop`, `hold` pass through safety gates to guarantee safe recovery. |
| D07 robotics and drones | Real vehicle hardware | `unverified` | No physical drone or rover attached. Live dispatch disabled (`blocked_live_families`). Recorded honestly as unverified. |
