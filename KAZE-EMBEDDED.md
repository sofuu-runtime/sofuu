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
settings. `sofuu.kaze.runAsync` has the same Kaze authorization gates and
suspends for asynchronous AI and tool operations. Capability calls must also
pass the embedding runtime's active policy; Kaze options cannot override a host
deny or approval requirement. Script files passed to `run`, `runAsync`, `check`
or `compile` use the runtime filesystem policy and native root. JS callers
supply routine arguments explicitly; they do not implicitly read
`agent.config.json` or environment variables. For `runAsync`, AI settings may be
passed in `options.ai`; compiled-agent bootstraps also resolve documented
`SOFUU_AI_*` settings.

For `sofuu script run`, named `--arg` and positional values take precedence
over environment variables and the optional default `agent.config.json`.
`--config <file>` selects another JSON configuration. An unreadable selected
file or a malformed existing configuration is an error. Compiled agents apply
their configuration tiers in their own bootstrap.

Compiled Kaze agents accept `--cap`, `--policy`, `--pin`, and
`--sim`; a newly compiled agent refuses an older runtime base that lacks the
required capability and embedded-boundary gates.

## Available and unavailable operations

- Local logging and the supported policy detectors execute in process.
  Unknown detector names fail; `allow` and `reject` produce VM verdicts.
- `--cap memory=ephemeral` selects an in-process CMA store for that one run.
  Without this explicit selection, Kaze memory calls return `E_CAP_MISSING`.
  The store is isolated per run and is discarded on return; it is not durable
  and is not shared with Sofuu's QTSQ stores or other agents.
- `sofuu.kaze.runAsync`, a compiled agent that declares `needs: ai` or
  `needs: tool`, and `sofuu script run` for a source or `.kazebc` file that
  declares either capability dispatch `ai.prompt`, `ai.complete`, and
  `ai.classify` through the real `sofuu.ai.complete` API, and `tool.call` /
  `tool.run` through Sofuu's built-in tool executor or configured MCP client.
  The adapter validates returned classification labels and converts a missing
  model to an actionable `E_CAP_MISSING`. Approval-required calls fail closed
  because no verifiable approval receipt is available.
- A real model-plus-MCP journey has been exercised on a debug build through all
  three entry points; see the local verification section. A script or
  `.kazebc` file that declares neither `ai` nor `tool` keeps the engine-free
  synchronous runner. Alert, sensor, device, and persistent memory operations
  remain unavailable unless explicitly documented above.
- Drone and rover simulation is available only when explicitly selected for
  testing, for example `--cap drone=sim`. Reports label simulated families.
  A bare `--cap drone` does not select a simulator.
- Embedded live drone and GPIO dispatch is disabled until policy propagation
  and physical safety checks are verified. No physical-hardware result is
  claimed.

### Per-run Kaze memory

For the current source build, choose a policy such as
`{"tools":{"allow":["kaze.memory.*"]}}` and run with
`--cap memory=ephemeral --policy policy.json`. The JS API accepts
`{caps: {memory: "ephemeral"}, policy: {tools: {allow: ["kaze.memory.*"]}}}`.

`memory.save(label, value)` stores a JSON-compatible value under a nonempty
text label; `memory.recall(label)` returns a list of values with that exact,
case-sensitive label, in insertion order. Identical label/value pairs are
deduplicated. Matching is exact, not semantic search. Values may include
lists and text-keyed records, with finite numbers; labels are limited to 1,024
bytes. A run accepts at most 256 records and 256 KiB of serialized writes;
values are limited to 16 nesting levels and 4,096 nodes. A rejected write
reports `E_KAZE_MEMORY`. The trace redacts memory labels and values; the
script's returned result can still contain the values it requests.

Each invocation creates its own in-memory CMA. Calls within that invocation
and its supervisor ticks share the store; independent invocations do not.
There is no disk reopen, cross-process sharing or storage encryption claim.

Existing published binaries have not been verified against these changes and
may behave differently. The async bridge is implemented and has passed a real
model-plus-MCP journey on a debug build with a configured remote provider; local
GGUF weights and released artifacts remain unverified. This work does not
establish a shared Kaze 1.0 core or bytecode compatibility with standalone Kaze.

## Local verification and remaining work

On 2026-11-13, the macOS debug build with SHA-256
`cd0a0abef7835aad1393a4d75e82362231d2768c1f18e92e1324b5d1749b0a21` passed the
99 focused embedded Kaze tests, 246 broader Kaze/compiler/memory/robotics tests,
and the cross-surface checks below. They cover pure
computation, refusals and policy, source and bytecode, both JS aliases, REPL,
compiled one-shot agents, per-run CMA storage, explicitly selected drone
simulation, missing-model classification, and output-label validation. A
separate opt-in run exercised a configured remote model and a real stdio MCP
server through `sofuu script run` (source and `.kazebc`), `sofuu.kaze.runAsync`
and a compiled agent. It needs a provider configuration, is not part of the
offline suite below, and downloaded no model weights.

```sh
cargo build --offline -p sofuu-core --bin sofuu --bin sofuu-agent
cargo test --offline -p sofuu-core --lib domains::kaze
cargo test --offline -p sofuu-core --lib domains::robotics::hardware::tests
python3 scripts/test-kaze-fail-closed.py target/debug/sofuu
python3 scripts/test-kaze-memory-real.py target/debug/sofuu
python3 scripts/test-kaze-js-boundaries.py target/debug/sofuu
# Opt-in, requires SOFUU_TEST_* provider configuration:
python3 scripts/test-kaze-real-ai.py
python3 scripts/test-kaze-cli-real-ai.py
```

The same boundary, fail-closed and ephemeral-memory checks also passed on a
temporary release candidate before the async bridge was added; that candidate
does not contain the bridge. The real-provider journey above is a local
debug-build result with one configured provider; it does not establish
published-artifact, local-GGUF, supervisor, interactive, or cancellation
coverage. The ignored local Kaze integration suite and repository-wide
mandatory gates still contain non-robotics stand-ins that need migration.
Follow the remaining Kaze tasks in [sofuu-map.md](sofuu-map.md).
