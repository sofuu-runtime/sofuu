# Runtime contract artifacts

This directory is the **machine-readable half** of the runtime contract. The
prose spec lives in [`docs/SPEC-RUNTIME-CONTRACT.md`](../docs/SPEC-RUNTIME-CONTRACT.md)
(local-only by repo policy); everything here is tracked.

Created in **P0** (2026-10-23 audit). Nothing here changes runtime behavior.

## Files

| File | Role |
|---|---|
| [`PROGRESS.md`](PROGRESS.md) | Live phase status, suite results, verification provenance |
| [`matrix.json`](matrix.json) | Source of truth for the 48-cell surface matrix (8 primitives × 6 surfaces), with per-cell status, missing operations, evidence and test ids |
| [`SURFACE-MATRIX.md`](SURFACE-MATRIX.md) | Generated, readable matrix + per-cell detail |
| [`stability.json`](stability.json) | Source of truth for the T1/T2/T3 stability index (34 C ABI exports, 21 CLI commands, 11 HTTP routes, 23 JS namespaces, 43 config keys — 132 entries) |
| [`STABILITY-INDEX.md`](STABILITY-INDEX.md) | Generated stability tables |
| [`knobs.json`](knobs.json) | Generated inventory of every directly-read `SOFUU_*` env knob (123), with class and numeric-effect decision |
| [`KNOBS.md`](KNOBS.md) | Generated knob table |

**Rule:** edit the JSON, then regenerate. Never hand-edit the generated `.md`.

## Regenerate

```bash
python3 scripts/contract/gen_contract_docs.py
```

The generator validates before writing:

- all 48 (primitive × surface) cells exist exactly once;
- every status is one of `full | partial | unavailable | n/a`;
- `partial` names its missing operations, `unavailable` names its reason,
  `full` cites evidence, every non-`n/a` cell names its test ids;
- every evidence path exists;
- every declared ABI symbol matches `scripts/abi_symbols.txt` (drift fails the run);
- every stability entry carries a valid tier.

It exits non-zero and lists every failing item.

## Conformance skeleton

```bash
bash scripts/contract/run_contract_tests.sh
```

52 cases (`T1.1` … `T7.9`), one per spec clause. Each case reports one of:

- **PASS** — the clause's observable check holds today;
- **FAIL** — the check ran and the contract is violated;
- **BLOCKED** — the check needs a prerequisite that does not exist yet
  (a fixture model, `libsofuu`, a running server, a chaos harness). The reason
  is printed.

Optional environment for the blocked kinds:

```bash
CONTRACT_HTTP_URL=http://127.0.0.1:8080   # enables the HTTP cases
CONTRACT_LIBSOFUU=dist/libsofuu.dylib     # enables the C ABI cases
SOFUU_BIN=target/release/sofuu            # override the binary
```

**P0 expected state:** mostly `FAIL`/`BLOCKED`; a `PASS` is legitimate only for
artifact checks that already hold (matrix completeness, ABI gate presence).
The runner exits non-zero until the contract is implemented — that is the
intended signal, not a defect in the runner.

Current state (2026-10-25): **12 PASS · 4 FAIL · 36 BLOCKED** — P1 is fully
complete, including the committed fixture model running the two-route exact
gate in CI (T3.3) and the demonstrated FORK enforcement (exit 3). See
[`PROGRESS.md`](PROGRESS.md) for the per-test detail, what landed in P1, and
the memory-clobber finding.

## P0 findings (2026-10-23)

The audit resolved every cell and surfaced these contract violations, recorded
in `matrix.json` and the spec:

1. `SOFUU_NO_SHELL=1` still advertises `bash` in `TOOLS` — **fixed 2026-10-23
   (`0b753686c`)**; `GROUP` is now filtered (`src/js/tools.js:65-77,1265-1271`).
2. The C ABI funnel is dynamic (no method whitelist), and
   `sofuu_rt_call_stream` injects `onStep` while `chat.submit` reads `opts.onEvent`
   — chat streaming events are unreachable through the C stream API.
3. Cancellation is thread-local in the C ABI (`lib.rs:1491`); Swift's
   `StreamHandle.cancel` silently no-ops off-thread.
4. A C host cannot settle a mid-run approval; only host-tool resolution exists.
5. `mcp.serve` returns `{}` through `sofuu_rt_call` (not serializable).
6. HTTP: no session/tool/agent routes; no usage in completion responses;
   `/v1/models` can return an id that completions reject; `serve --infer` is a
   no-op flag; `/health` requires the bearer token.
7. HTTP `serve` and chat/agent use different default brain files
   (`~/.sofuu_brain.qtsq` vs `<cwd>/.sofuu/brain/brain.qtsq`).
8. `sofuu.chat.deleteSession` removes the registry entry, not the files.
9. Usage has three shapes (`pt/ct/cr/cw/cost`, `promptTokens/completionTokens`,
   none) and `ai.complete` reports none.
10. Event vocabularies differ across chat, agent, and the persisted session log
    (`start` has different payloads; terminal events differ).
11. `SOFUU_NO_TILES`/`SOFUU_NO_ROPE_TABLE` are determinism-reference switches but
    are not declared as tiers anywhere.
12. The public distribution story cannot ship memory: CI builds QTSQ-free, the
    four QTSQ-bearing binaries in `cli/` are unpublished ("keep them private"),
    and a QTSQ-free build silently no-ops `sofuu.memory`.

## Provenance

- Audit date: 2026-10-23. Sources: direct code reads across `src/js`,
  `crates/*/src`, `bindings/`, `include/`, `scripts/abi_symbols.txt`,
  `examples/headless`, and `.github/workflows/ci.yml`.
- This directory and `scripts/contract/` are new files only. No source file,
  CI workflow, ABI header, or peer-owned lane was modified.
