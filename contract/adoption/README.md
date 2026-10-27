# External adoption loop (P6)

P6 is the standing loop that turns the published receipts into outside use.
The loop is deliberately mechanical: nothing enters it without a public-surface
deployment and a receipt, and nothing leaves it without evidence.

## What counts as a deployment

A team outside this repository running a **hard requirement** on the public
surface:

- a shipped device app (Android/iOS/macOS/desktop) using the public API only,
- a service or fleet (e.g. a Pi fleet) running `sofuu serve` or the library,
- a regulated use case where failure is not cosmetic.

The bar is that Sofuu is load-bearing, not evaluated. Internal reference apps
(`benchmark/`, `examples/`, the in-repo macOS/Android apps) do not count as
external deployments; they are the fixtures the public surface is audited
against.

## Onboarding (what the team does, unaided)

1. Pin a release tag and its `CHECKSUMS.txt`.
2. Build from the documented path (README install, or `make` + `make libsofuu`).
3. Run the contract + install gates:

   ```bash
   bash scripts/contract/run_contract_tests.sh
   bash scripts/verify-dist.sh dist ./sofuu
   sofuu conformance > conformance.json
   ```

   Expected on a release (QTSQ-free) build: 52 cases — 42 PASS / 0 FAIL /
   10 BLOCKED, posture ok, `verify_dist: ok`. A discrepancy is a P6 finding:
   record it in the ledger before working around it.
4. Keep the app on the public surface only. A private path (an undocumented
   global, a forked driver, a hand-built session file) is an API defect and is
   recorded as one.

## Exit evidence (P6 is done only when both exist)

1. **Upgrade drill** — the team moves from one release to the next minor
   version using only the changelog, with **zero support conversations**.
2. **Independent conformance** — a third party runs the suite and produces the
   expected board **without help from the maintainers**.

Record each in [`ledger.json`](ledger.json):

```json
{
 "deployments": [
  {"name": "…", "surface": "…", "requirement": "…", "started": "YYYY-MM-DD",
   "status": "running", "evidence": "link to their report/issue"}
 ],
 "independent_conformance_proofs": [
  {"team": "…", "platform": "…", "report": "link", "date": "YYYY-MM-DD"}
 ],
 "upgrade_drills": [
  {"team": "…", "from_version": "0.2.0", "to_version": "0.2.1",
   "support_conversations": 0, "date": "YYYY-MM-DD", "notes": "…"}
 ]
}
```

Then regenerate the published ledger and pages:

```bash
python3 scripts/scoreboard.py
```

Outside teams do not need repository write access: open an **Adoption report**
issue (the template in `.github/ISSUE_TEMPLATE/adoption-report.yml`) with the
release tag, the conformance summary and the upgrade evidence; accepted
entries are mirrored into `ledger.json` here.

## Current status

**Open.** No external deployment yet. The publishing half of P6 is live
(`../CONFORMANCE.md`, `../SCOREBOARD.md`); the loop now needs a real outside
team to enter it. Recruiting one is the next action, not a code change.
