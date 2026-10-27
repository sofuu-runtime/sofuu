# Scoreboard sources (tracked receipts)

These are the **inputs** to the published pages:

| Path | What it is |
|---|---|
| `parity.jsonl` | One JSON receipt per `scripts/parity_sweep.sh` measurement. Append-only. |
| `conformance/*.json` | `sofuu conformance` reports captured from the platform matrix. |
| `baseline.json` | The frozen acceptance floor enforced by `scripts/scoreboard.py --check`. |

Generated from these (never edit by hand): `../SCOREBOARD.md`, `../CONFORMANCE.md`,
`../scoreboard.json`, `../adoption/LEDGER.md`.

## Current provenance

- Conformance snapshots: CI run **40457020037** @ `d92ff9599` (macOS arm64
  merged report; Linux x86_64 and arm64 posture reports). The macOS report is
  52 cases — 42 PASS / 0 FAIL / 10 BLOCKED, posture ok, `verify_dist: ok`.
- Parity ledger: the sweep receipts accumulated through 2026-10-22. The
  headline is the latest **valid** receipt per model; zero-metric rows stay in
  the ledger but are listed as unmeasured.

Refresh the snapshots from a run:

```bash
gh run download <run-id> -n conformance-macos-arm64 -D /tmp/c
cp /tmp/c/dist/conformance-report.json contract/scoreboard/conformance/macos-arm64.json
gh run download <run-id> -n conformance-Linux-x86_64 -D /tmp/cx
cp /tmp/cx/conformance-Linux-x86_64.json contract/scoreboard/conformance/linux-x86_64.json
gh run download <run-id> -n conformance-Linux-arm64 -D /tmp/ca
cp /tmp/ca/conformance-Linux-arm64.json contract/scoreboard/conformance/linux-arm64.json
python3 scripts/scoreboard.py
```

A new sweep appends to `parity.jsonl` (`scripts/parity_sweep.sh` writes here
and to the local-only `docs/bench/parity.jsonl`):

```bash
bash scripts/parity_sweep.sh
python3 scripts/scoreboard.py
```

`python3 scripts/scoreboard.py --check` fails on a metric more than 10% below
`baseline.json`, on lost parity, on any conformance FAIL, or on generated-page
drift. Accept a new floor only deliberately: `--freeze-baseline`.
