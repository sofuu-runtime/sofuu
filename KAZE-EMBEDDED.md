# Embedded Kaze: current capability status

This document describes the embedded Kaze source in Sofuu, not a published
binary. The embedded engine reports language version **0.1**, writes bytecode
format 2, and reads legacy format 1. The separate standalone Kaze project
reports language version 1.0 and uses bytecode format 4. Aligning them on one
language contract remains planned; their artifacts are not interchangeable.

## Capability authorization

Each capability call must pass all of the following:

1. The script declares its family with `needs:`.
2. The host explicitly grants the family, for example `--cap log`.
3. A valid R7 policy contains `tools.allow` or `tools.deny`, and its decision
   permits the operation. An empty `tools.allow` permits nothing. A
   `tools.deny: []` policy is permissive unless another R7 rule denies or
   requires approval.
4. Applicable physical argument checks pass; out-of-range values are rejected.

The policy names calls as `kaze.<family>.<operation>`. For example, this
policy permits matching log calls through the tool allowlist:

```json
{"tools":{"allow":["kaze.log.*"]}}
```

`sofuu script run program.kaze --cap log --policy policy.json` passes the
grant and policy explicitly. Without `--policy`, the script CLI checks the
user and project `.sofuu/policy.json` layers. `--no-policy` disables loading;
capability calls then fail closed. Pure calculations need no capability grant
or policy. Malformed policy and duplicate JSON keys are refused.

`sofuu.kaze.run` and its `sofuu.sfs.run` alias accept `grants` or `caps`,
`policy` or strict `policyJson`, and optional backend, pin and simulation
settings. Compiled Kaze agents accept `--cap`, `--policy`, `--pin`, and
`--sim`; a newly compiled agent refuses an older runtime base that lacks the
required capability gate.

## Available and unavailable operations

- Local logging and the supported policy detectors execute in process.
  Unknown detector names fail; `allow` and `reject` produce VM verdicts.
- Embedded AI, tool, memory, alert, sensor and device calls have no enabled
  backend and return `E_CAP_MISSING` after authorization. They do not perform
  model inference, MCP calls, storage, alert delivery or sensor reads.
- Drone and rover simulation is available only when explicitly selected for
  testing, for example `--cap drone=sim`. Reports label simulated families.
  A bare `--cap drone` does not select a simulator.
- Embedded live drone and GPIO dispatch is disabled until policy propagation
  and physical safety checks are verified. No physical-hardware result is
  claimed.

Existing published binaries have not been verified against these changes and
may behave differently. This work does not establish a real Kaze model-and-tool
agent, a shared Kaze 1.0 core, or bytecode compatibility with standalone Kaze.

## Local verification and remaining work

On 2026-11-11, a macOS debug build passed 72 embedded Kaze unit tests, 15
hardware-gate unit tests, and the cross-surface checks below. They cover pure
computation, refusals and policy, source and bytecode, both JS aliases, REPL,
compiled one-shot agents, an incompatible older runtime base and explicitly
selected drone simulation. No model, external tool, or physical hardware was
used.

```sh
cargo build --offline -p sofuu-core --bin sofuu --bin sofuu-agent
cargo test --offline -p sofuu-core --lib domains::kaze
cargo test --offline -p sofuu-core --lib domains::robotics::hardware::tests
python3 scripts/test-kaze-fail-closed.py target/debug/sofuu
```

The ignored local Kaze integration suite and repository-wide mandatory gates
still contain non-robotics stand-ins that need migration before compliant
execution. Supervisor and interactive paths, release artifacts, real adapters,
and physical hardware need further verification. Follow the remaining Kaze
tasks in [sofuu-map.md](sofuu-map.md).
