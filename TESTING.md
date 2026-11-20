# Testing

Choose checks based on the changed behavior and the interfaces it affects. A
focused run is fast feedback, not a substitute for affected integration tests
or a claim that the full suite or a release passed. The maintainer remains
responsible for choosing the relevant tests; no automatic file mapping can
infer every dependency.

## Focused Rust Tests

```sh
make test-area AREA=kaze
make test-area AREA=memory
make test-rust-filter FILTER=domains::kaze::parser::
```

The explicit area names and Rust name filters are:

| Area | `sofuu-core` library filter |
| --- | --- |
| `kaze` | `domains::kaze::` |
| `robotics` | `domains::robotics::` |
| `memory` | `domains::memory::` |
| `inference` | `domains::infer::` |
| `compiler` | `compiler::` |
| `mcp` | `mcp::` |
| `policy` | `rt::policy::` |
| `sandbox` | `rt::sandbox::` |
| `runtime` | `rt::` |
| `diagnostics` | `rt::diag_contract::` |
| `rlm` | `rlm::` |

These use the existing offline preflight Rust gate and its scripted-model
exclusions. They compile the `sofuu-core` library test binary and execute only
matching test names; `runtime` is intentionally broad. Cargo can still compile
shared dependencies. The output includes the selected filter, passed, failed,
ignored and filtered counts. No passing tests, an unknown area, or a missing
filter is not a successful check. A skipped scripted-model test is a reported
coverage gap, not a pass. Direct Cargo filters are also available when an even
narrower or different crate test is needed; use `SOFUU_ALLOW_LOOPBACK=1` for
tests that require the local transport policy. `SOFUU_RUST_FILTER` is accepted
only for an explicit `preflight.sh rust` run, preventing an inherited filter
from silently narrowing a full or multi-section preflight.

Names do not follow dependencies automatically. For example, a Kaze change
affecting standalone compilation also requires `AREA=compiler`, and possibly
`script_cmd` tests (`make test-rust-filter FILTER=script_cmd::`). A memory
change touching its JS binding, WAL, C ABI, or agent consumer needs those
additional tests. Shared runtime, FFI, protocol and security changes can need
several areas or the complete suite. Add a regression test for the actual bug.

## Integration And Contracts

```sh
make test-contract-case CASE=T1.1
SOFUU_BIN=/path/to/candidate make real-mcp-check
SOFUU_BIN=/path/to/candidate make real-memory-check
SOFUU_BIN=/path/to/candidate make real-kaze-fail-closed-check
```

Contract case IDs are exact IDs from `scripts/contract/cases.tsv`; an unknown
ID fails instead of running zero cases. A case exercises its declared real or
pure resources, and a missing prerequisite is a coverage gap. Use an explicitly
built `SOFUU_BIN` where the selected check supports it; some contract commands
name `./sofuu` directly, so check the row to establish the artifact tested.
Individual `real-*` targets and the local JS corpus are additional options in
the Makefile. `tests/` is mostly Git-ignored: its local runner and coverage are
not guaranteed in a clean checkout, and individual test provenance must be
checked before relying on its result.

For driver edits run `make bc-check` and the relevant driver/consumer tests.
For domain-boundary edits run `bash scripts/preflight.sh isolation`. For Kaze
surface changes consider `make kaze-drift-check`; a missing local prototype is
not proof of compatibility. For the C ABI, platform bindings, inference or
hardware, run the corresponding crate, platform or real-resource checks and
record unavailable environments explicitly.

After changing the selection or result-reporting logic itself, run
`python3 scripts/test-focused-runner.py`. It exercises the real Rust and
contract runners, so it needs an executable `./sofuu` and an offline Cargo
build environment; it does not substitute model or tool backends.

## Full Gates

`make test-rust`, `make test`, and `bash scripts/preflight.sh` retain their
existing broader scope. CI remains broad. Before release, run the full
preflight, contract suite, distribution verification, size and relevant
platform/device gates on the actual candidate. See `RULES.md` and
`contract/CAPABILITY-REGISTER.md` for evidence requirements.

Preflight exits 0 on success, 1 on failure, and 2 when blocked. Its optional
`ALLOW_BLOCKED=1` only permits an explicitly incomplete run and never closes
the gap; contract checks have their own stricter blocked handling. Report
which commands, artifacts and resources were used, along with failures,
blocked tests and checks not run. Use only models already present locally;
never download weights or substitute non-robotics backend results.
