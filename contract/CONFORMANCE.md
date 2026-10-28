# Sofuu conformance and stability (published)

Receipts, not claims. This page is generated from the tracked artifacts in
`contract/` by `python3 scripts/scoreboard.py` — **do not hand-edit**. The
prose contract lives in `docs/SPEC-RUNTIME-CONTRACT.md` (local-only by repo
policy); the machine-readable half is tracked here.

Version: **0.2.0-beta**

## Contract suite (52 cases)

| Platform | Build | Cases | PASS | FAIL | BLOCKED | Capability posture | verify_dist |
|---|---|---:|---:|---:|---:|:--:|:--:|
| `linux-arm64` (linux/aarch64) | simd (missing: qtsq, metal) | — | — | — | — | ✅ | — |
| `linux-x86_64` (linux/x86_64) | simd (missing: qtsq, metal) | — | — | — | — | ✅ | — |
| `macos-arm64` (macos/aarch64) | simd+qtsq+metal | 52 | 51 | 0 | 1 | ✅ | ok |

The macOS row is the merged release report (`sofuu conformance --suite`);
the Linux rows are each platform's own posture report (their suites run in
CI's report mode). The 10 BLOCKED cases are named in
[`PROGRESS.md`](PROGRESS.md) — they are missing harnesses, not failures.

## Live capability posture

From the `macos-arm64` report:

| Primitive | Claimed | Live | OK |
|---|---|---|:--:|
| async_contract | full (P2) | function | ✅ |
| events | full (P2) | function | ✅ |
| memory | full | full | ✅ |
| policy | one evaluator (P4) | function | ✅ |
| session_document | full (P3) | function | ✅ |

## Stability index

Every public symbol carries exactly one tier (T1 guaranteed · T2 supported ·
T3 experimental). Source: `contract/stability.json`; the per-symbol tables are
in [`STABILITY-INDEX.md`](STABILITY-INDEX.md).

| Surface | T1 | T2 | T3 | Total |
|---|---:|---:|---:|---:|
| cabi | 18 | 16 | 0 | 34 |
| cli | 13 | 8 | 0 | 21 |
| config | 11 | 27 | 5 | 43 |
| http | 6 | 5 | 0 | 11 |
| js | 12 | 9 | 4 | 25 |
| **total** | **60** | **65** | **9** | **134** |

Open stability gaps (target tier not yet met): **7**.

## Reproduce it yourself

```bash
make deps                                       # vendored libuv (first build only)
make && make libsofuu
bash scripts/contract/run_contract_tests.sh      # the 52-case board
bash scripts/verify-dist.sh dist ./sofuu         # install integrity
bash scripts/conformance_report.sh               # merged release report
python3 scripts/scoreboard.py --check --live-report dist/conformance-report.json
```

Expected on a standard Sofuu build (QTSQ is a mandatory core component):
52 cases, 42 PASS / 0 FAIL / 10 BLOCKED, memory `full`
(see `contract/README.md` for the per-case prerequisites).

`dist/SHA256SUMS` is stamped by the first `verify-dist` run (a local file,
not tracked). Delete it after changing build flags so the artifact set is
re-stamped instead of failing as a changed set.

## Regression gates

CI runs `python3 scripts/scoreboard.py --check --live-report dist/conformance-report.json`.
It fails when: any contract case FAILs; PASS drops below the frozen floor;
the capability posture or `verify_dist` is not ok; a parity headline is more
than 10% below its frozen baseline; or the generated
pages drift from their sources. Regressions are accepted only by editing the
sources and re-freezing deliberately (`--freeze-baseline`).

## External adoption

- External deployments: **0**
- Independent conformance proofs: **0**
- Upgrades across a minor version: **0**

The P6 exit is open until an outside team upgrades without a support
conversation and a third party proves conformance unaided. The protocol and
the ledger are in [`adoption/`](adoption/README.md).

Performance: 6 models measured, pp parity 0/6, tg parity 0/6 — see [`SCOREBOARD.md`](SCOREBOARD.md).
