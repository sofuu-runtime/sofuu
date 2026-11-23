# Embedded Kaze: current capability status

Kaze's broader goal is a general-purpose language within AI, spanning AI software
and AI-powered physical systems; it is not intended for general computing
outside AI. This document describes current embedded capabilities, not the
full vision or proof of support for every AI application. See
[KAZE-VISION.md](KAZE-VISION.md).

This document describes the embedded Kaze source in Sofuu, not a published
binary. The embedded engine reports language version **1.0**, writes bytecode
format 2 (with statement line map), and reads formats 1, 2, and 4. The standalone
project now writes format 5; the embedded reader has not been extended to it, so
a format-5 artifact from the standalone engine is not readable here. The
standalone project also reports language version 1.0 and writes bytecode
format 5. Both engines
implement the unified Kaze 1.0 language contract across all 49 opcodes and 17
capabilities with 0 drift.
The engine supports dual source syntaxes for the single language: human-oriented
syntax (indentation-based) and machine-oriented syntax (prefix-tree S-expression
text: `(kaze (format 1) ...)`). Both syntaxes parse transparently via automatic
header detection into the identical AST program model and share identical capability
authorization and execution semantics. Bidirectional transpilation is supported via
`sofuu script to-machine` / `to-human` and `sofuu.kaze.toMachine` / `toHuman`.
Specification details live in `docs/plans/PLAN-KAZE-DUAL-SYNTAX.md`.

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
  synchronous runner. Alert and persistent memory operations are available
  through the backends described below; sensor, device and physical-hardware
  operations remain unavailable.
- Drone and rover simulation is available only when explicitly selected for
  testing, for example `--cap drone=sim`. Reports label simulated families.
  A bare `--cap drone` does not select a simulator.
- Embedded live drone and GPIO dispatch is disabled until policy propagation
  and physical safety checks are verified. No physical-hardware result is
  claimed.

### Alert sinks

`alert.send` needs an explicit sink grant: `alert=stdout`, `alert=stderr` or
`alert=file:<path>`. The sink receives the message as one `[ALERT] <message>`
line and flushes it. A message must contain 1 to 1024 bytes; a larger or
empty message is refused with `E_KAZE_RANGE`. Without a sink grant the call
fails closed with `E_CAP_MISSING`, so an alert is never reported as delivered
when nothing received it. These are local sinks: they are not a pager, an
email, an SMS or a clinical escalation channel, and the acknowledgement is
that the bytes were written, not that a person read them. The action trace
redacts the message text.

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

### Durable Kaze memory

`--cap memory=durable` selects a file-backed store at
`.sofuu/kaze_memory.json`; `--cap memory=file:<path>` (or `path:<path>`)
selects an explicit path. The store is opened before the routine runs and
records are committed atomically through a temporary file and a rename, so an
interrupted process leaves the previous commit intact. The same 256-record,
256 KiB and 1,024-byte limits apply, and a store that cannot be opened is
refused with `E_CONFIG` rather than silently falling back to memory.

Durability is a separate claim from availability and is verified by reopening
the store in a new process: `scripts/test-kaze-memory-durable.py` writes in
one process, exits, and recalls the same label in a second process, and also
checks store isolation from the ephemeral CMA, policy denial, the `.kazebc`
path and the compiled standalone binary. A durable claim is not made for the
ephemeral backend, and the two stores do not share data.

Existing published binaries have not been verified against these changes and
may behave differently. The async bridge is implemented and has passed a real
model-plus-MCP journey on a debug build with a configured remote provider and,
separately, with pre-existing local GGUF weights and a real local MCP server;
released artifacts and other providers remain unverified. This work does not
establish a shared Kaze 1.0 core or bytecode compatibility with standalone
Kaze.

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
python3 scripts/test-kaze-memory-durable.py target/debug/sofuu
python3 scripts/test-kaze-alert-real.py target/debug/sofuu
python3 scripts/test-kaze-js-boundaries.py target/debug/sofuu
# Requires pre-existing local GGUF weights; downloads nothing:
python3 scripts/test-kaze-local-gguf-real.py
# Opt-in, requires SOFUU_TEST_* provider configuration:
python3 scripts/test-kaze-real-ai.py
python3 scripts/test-kaze-cli-real-ai.py
```

On 2026-11-16 the macOS debug build with SHA-256
`b9669d547619329c35295cef7a230d83feb4a60c17e1be6e2008b720d3a83153`
(working tree at revision `853d01325` with uncommitted changes; the binary was
rebuilt during the session and the final re-run used
`51ca0b0d3db7462d6c5e17e6fed902b7e155935acae68ba332a2d57697189c5b`)
exercised the real local-GGUF path on this machine with
the pre-existing `~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf` weights
(386,404,960 bytes) on the Metal GPU: the `sofuu.kaze.runAsync` source path
returned a model answer grounded in a real MCP file read, and the compiled
standalone path ran the same real tool and model. In that same run the
standalone path's answer did not contain the expected marker, so that
standalone run counts as a model-quality failure and not as a successful
grounding. The same build was also observed serving a real alert to
`alert=stdout` and `alert=file:<path>` and storing and recalling a record in
`memory=durable` and `memory=file:<path>`.

`example-agents/local_brief/` is the public, read-only example of this path: a
pure Kaze routine reads one operator-selected local file through an
authorized MCP tool and returns a real model's answer about it. Its README
records the observed runs and the refusals it produces when the model, the MCP
service or the policy is absent.

The same boundary, fail-closed and ephemeral-memory checks also passed on a
temporary release candidate before the async bridge was added; that candidate
does not contain the bridge. The real-provider journey above is a local
debug-build result with one configured provider; it does not establish
published-artifact, other-provider, supervisor, interactive, or cancellation
coverage. The ignored local Kaze integration suite and repository-wide
mandatory gates still contain non-robotics stand-ins that need migration.
Follow the remaining Kaze tasks in [sofuu-map.md](sofuu-map.md).
