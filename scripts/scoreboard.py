#!/usr/bin/env python3
"""scoreboard.py — publish + gate the P6 conformance and parity receipts.

Sources (tracked, hand-fed):
  contract/scoreboard/parity.jsonl           one JSON receipt per sweep row
  contract/scoreboard/conformance/*.json     `sofuu conformance` reports
  contract/stability.json                    the T1/T2/T3 stability index
  contract/adoption/ledger.json              external deployments and exit evidence

Generated (never hand-edit):
  contract/scoreboard.json                   machine-readable digest
  contract/SCOREBOARD.md                     the public parity scoreboard
  contract/CONFORMANCE.md                    conformance + stability + adoption
  contract/adoption/LEDGER.md                the adoption ledger

Usage:
  python3 scripts/scoreboard.py                 # regenerate the published files
  python3 scripts/scoreboard.py --freeze-baseline
                                                # accept the current headline as
                                                # the regression floor (deliberate)
  python3 scripts/scoreboard.py --check [--live-report FILE]
                                                # CI gate: generated files must be
                                                # current, no conformance FAIL, no
                                                # parity regression beyond tolerance

Exit codes: 0 clean, 1 validation/regression/drift, 2 usage/environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PARITY = ROOT / "contract/scoreboard/parity.jsonl"
CONF_DIR = ROOT / "contract/scoreboard/conformance"
BASELINE = ROOT / "contract/scoreboard/baseline.json"
STABILITY = ROOT / "contract/stability.json"
LEDGER = ROOT / "contract/adoption/ledger.json"
OUT_JSON = ROOT / "contract/scoreboard.json"
OUT_SCOREBOARD = ROOT / "contract/SCOREBOARD.md"
OUT_CONFORMANCE = ROOT / "contract/CONFORMANCE.md"
OUT_LEDGER = ROOT / "contract/adoption/LEDGER.md"

PARITY_TOLERANCE_PCT = 10.0


# ── helpers ────────────────────────────────────────────────────────────────


def die(msg: str, code: int = 1) -> None:
    print(f"scoreboard: {msg}", file=sys.stderr)
    raise SystemExit(code)


def num(x: Any) -> str:
    """Compact deterministic number (605.0 -> 605, 23.45 -> 23.5)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.1f}"


def mark(ok: bool | None) -> str:
    if ok is None:
        return "—"
    return "✅" if ok else "❌"


def dump_json(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


# ── parity ─────────────────────────────────────────────────────────────────


def load_parity() -> list[dict[str, Any]]:
    if not PARITY.is_file():
        die(f"missing {PARITY.relative_to(ROOT)}", 2)
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(PARITY.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            die(f"{PARITY.relative_to(ROOT)}:{lineno}: invalid JSON ({e})", 2)
        if not isinstance(row, dict) or not row.get("model"):
            die(f"{PARITY.relative_to(ROOT)}:{lineno}: row needs a model", 2)
        row["_line"] = lineno
        rows.append(row)
    if not rows:
        die("parity ledger is empty", 2)
    return rows


def metrics(row: dict[str, Any]) -> tuple[float, float, float, float]:
    def f(key: str) -> float:
        try:
            return float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return f("ll_pp"), f("ll_tg"), f("sofuu_pp"), f("sofuu_tg")


def summarize_parity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]).strip(), []).append(row)

    headline: dict[str, Any] = {}
    unmeasured: list[dict[str, Any]] = []
    for model in sorted(by_model):
        valid: list[dict[str, Any]] = []
        for row in by_model[model]:
            ll_pp, ll_tg, s_pp, s_tg = metrics(row)
            if ll_pp > 0 and ll_tg > 0 and s_pp > 0 and s_tg > 0:
                valid.append(row)
        if not valid:
            last = by_model[model][-1]
            unmeasured.append(
                {
                    "model": model,
                    "receipts": len(by_model[model]),
                    "last_ts": str(last.get("ts") or ""),
                    "note": str(last.get("note") or ""),
                }
            )
            continue
        # Latest by RFC3339 timestamp; file order breaks ties.
        best = max(valid, key=lambda r: (str(r.get("ts") or ""), int(r["_line"])))
        ll_pp, ll_tg, s_pp, s_tg = metrics(best)
        headline[model] = {
            "ll_pp": ll_pp,
            "ll_tg": ll_tg,
            "sofuu_pp": s_pp,
            "sofuu_tg": s_tg,
            "pass_pp": s_pp >= ll_pp,
            "pass_tg": s_tg >= ll_tg,
            "text_exact": best.get("text_exact") if isinstance(best.get("text_exact"), bool) else None,
            "ts": str(best.get("ts") or ""),
            "note": str(best.get("note") or ""),
            "receipts": len(by_model[model]),
        }

    pass_pp = sum(1 for v in headline.values() if v["pass_pp"])
    pass_tg = sum(1 for v in headline.values() if v["pass_tg"])
    return {
        "receipts": len(rows),
        "models": len(by_model),
        "headline": headline,
        "unmeasured": unmeasured,
        "pass_pp": pass_pp,
        "pass_tg": pass_tg,
        "measured": len(headline),
    }


# ── conformance ────────────────────────────────────────────────────────────


def load_conformance() -> list[dict[str, Any]]:
    if not CONF_DIR.is_dir():
        die(f"missing {CONF_DIR.relative_to(ROOT)}", 2)
    reports: list[dict[str, Any]] = []
    for path in sorted(CONF_DIR.glob("*.json")):
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            die(f"{path.relative_to(ROOT)}: invalid JSON ({e})", 2)
        if report.get("kind") != "sofuu-conformance":
            die(f"{path.relative_to(ROOT)}: not a sofuu-conformance report", 2)
        report["_platform"] = path.stem
        reports.append(report)
    if not reports:
        die("no conformance reports under contract/scoreboard/conformance/", 2)
    return reports


def suite_of(report: dict[str, Any]) -> dict[str, Any] | None:
    suite = report.get("contract_suite")
    return suite if isinstance(suite, dict) else None


def posture_ok(report: dict[str, Any]) -> bool:
    return report.get("capability_posture_ok") is True


def verify_dist(report: dict[str, Any]) -> str | None:
    integ = report.get("install_integrity")
    if not isinstance(integ, dict):
        return None
    value = integ.get("verify_dist")
    return value if isinstance(value, str) else None


def build_of(report: dict[str, Any]) -> str:
    manifest = report.get("manifest")
    build = manifest.get("build") if isinstance(manifest, dict) else None
    if not isinstance(build, dict):
        return "—"
    features = [str(x) for x in (build.get("features") or [])]
    missing = [str(x) for x in (build.get("missing") or [])]
    text = "+".join(features) if features else "—"
    if missing:
        text += f" (missing: {', '.join(missing)})"
    return text


def summarize_conformance(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    versions = sorted({str(r.get("version") or "") for r in reports})
    for report in reports:
        suite = suite_of(report)
        rows.append(
            {
                "platform": report["_platform"],
                "os": str((report.get("platform") or {}).get("os") or ""),
                "arch": str((report.get("platform") or {}).get("arch") or ""),
                "version": str(report.get("version") or ""),
                "build": build_of(report),
                "cases": suite.get("cases") if suite else None,
                "pass": suite.get("pass") if suite else None,
                "fail": suite.get("fail") if suite else None,
                "blocked": suite.get("blocked") if suite else None,
                "posture_ok": posture_ok(report),
                "verify_dist": verify_dist(report),
                "posture": report.get("capability_posture") or {},
                "generated_at": report.get("generated_at"),
            }
        )
    return {"rows": rows, "versions": versions}


# ── stability + adoption ───────────────────────────────────────────────────


def summarize_stability() -> dict[str, Any]:
    if not STABILITY.is_file():
        die(f"missing {STABILITY.relative_to(ROOT)}", 2)
    data = json.loads(STABILITY.read_text())
    out: dict[str, Any] = {}
    totals = {"T1": 0, "T2": 0, "T3": 0}
    for surface in sorted((data.get("surfaces") or {})):
        block = data["surfaces"][surface]
        counts = {"T1": 0, "T2": 0, "T3": 0}
        if isinstance(block, dict):
            # Each surface has one group (symbols / commands / routes /
            # namespaces / keys); a JS namespace entry carries its tier in
            # `tier` and its members underneath, but the index counts the
            # namespace once, not every member.
            for group in block.values():
                if not isinstance(group, dict):
                    continue
                for entry in group.values():
                    tier = entry if isinstance(entry, str) else (
                        entry.get("tier") if isinstance(entry, dict) else None
                    )
                    if tier in counts:
                        counts[tier] += 1
        out[surface] = {**counts, "total": sum(counts.values())}
        for key in totals:
            totals[key] += counts[key]
    gaps = data.get("gaps") or []
    return {"surfaces": out, "total": {**totals, "total": sum(totals.values())}, "gaps": len(gaps)}


def load_adoption() -> dict[str, Any]:
    if not LEDGER.is_file():
        die(f"missing {LEDGER.relative_to(ROOT)}", 2)
    data = json.loads(LEDGER.read_text())
    for key in ("deployments", "upgrade_drills", "independent_conformance_proofs"):
        if not isinstance(data.get(key, []), list):
            die(f"{LEDGER.relative_to(ROOT)}: {key} must be a list", 2)
    return data


# ── rendering ──────────────────────────────────────────────────────────────


def render_scoreboard(parity: dict[str, Any], conf: dict[str, Any]) -> str:
    head = parity["headline"]
    lines = [
        "# Sofuu performance scoreboard (generated)",
        "",
        "**This is the inference-parity engineering bar, not the product claim.**",
        "The bar is llama.cpp's default build (Metal offload on macOS) on the same",
        "machine; the app-level comparison is the two-build benchmark in",
        "`benchmark/`. A failing row is a measurement, not a marketing result — the",
        "parity program that closes it is tracked in `docs/PLAN-LLAMA-CPP-PARITY.md`",
        "(local-only by repo policy).",
        "",
        "Every row is a measured receipt; the headline is the **latest valid receipt",
        "per model**, i.e. the newest row that recorded both pp and tg (zero-metric",
        "rows are listed under *unmeasured* instead of being scored). The ledger is",
        "append-only and receipts carry the configuration they were taken under, so",
        "the table is a chronological status, not a normalized benchmark. Generated",
        "by `python3 scripts/scoreboard.py` — **do not hand-edit**.",
        "",
        f"**{parity['measured']} models measured · {parity['receipts']} receipts · "
        f"pp parity {parity['pass_pp']}/{parity['measured']} · "
        f"tg parity {parity['pass_tg']}/{parity['measured']}**",
        "",
        "| Model | llama.cpp pp | Sofuu pp | pp | llama.cpp tg | Sofuu tg | tg | text | measured |",
        "|---|---:|---:|:--:|---:|---:|:--:|:--:|---|",
    ]
    for model in sorted(head):
        row = head[model]
        lines.append(
            f"| `{model}` | {num(row['ll_pp'])} | {num(row['sofuu_pp'])} | {mark(row['pass_pp'])} "
            f"| {num(row['ll_tg'])} | {num(row['sofuu_tg'])} | {mark(row['pass_tg'])} "
            f"| {mark(row['text_exact'])} | {row['ts'][:10] or '—'} |"
        )
    lines += [
        "",
        "`text` = the exact-text gate ran and matched; `—` = not recorded on that receipt.",
        "",
        "## Configuration of the headline receipts",
        "",
    ]
    for model in sorted(head):
        row = head[model]
        note = row["note"] or "no note recorded"
        lines.append(f"- `{model}` ({row['ts'][:16] or '—'}): {note}")
    lines += [
        "",
        "## Unmeasured receipts",
        "",
    ]
    if parity["unmeasured"]:
        for row in parity["unmeasured"]:
            note = f" — {row['note']}" if row["note"] else ""
            lines.append(
                f"- `{row['model']}`: {row['receipts']} receipt(s), no valid pp+tg pair "
                f"(last {row['last_ts'][:10] or '—'}){note}"
            )
    else:
        lines.append("None — every model in the ledger has a valid measurement.")
    lines += [
        "",
        "## How this is fed and gated",
        "",
        "- `scripts/parity_sweep.sh` measures each local model against `llama-bench`",
        "  and appends receipts to `contract/scoreboard/parity.jsonl` (and the",
        "  local-only `docs/bench/parity.jsonl`).",
        "- `python3 scripts/scoreboard.py` regenerates this page, `contract/scoreboard.json`",
        "  and the conformance page from the tracked sources.",
        "- `python3 scripts/scoreboard.py --check --live-report <report>` is the CI",
        "  gate: it fails on drift, on a conformance FAIL/posture regression, and on",
        f"  any metric more than {num(PARITY_TOLERANCE_PCT)}% below the frozen baseline.",
        "  Accepting a new floor is deliberate: `--freeze-baseline`.",
        "",
        f"Measured alongside: {len(conf['rows'])} conformance snapshots "
        f"({', '.join(r['platform'] for r in conf['rows'])}) — see `contract/CONFORMANCE.md`.",
        "",
    ]
    return "\n".join(lines)


def render_conformance(
    parity: dict[str, Any],
    conf: dict[str, Any],
    stability: dict[str, Any],
    adoption: dict[str, Any],
) -> str:
    lines = [
        "# Sofuu conformance and stability (published)",
        "",
        "Receipts, not claims. This page is generated from the tracked artifacts in",
        "`contract/` by `python3 scripts/scoreboard.py` — **do not hand-edit**. The",
        "prose contract lives in `docs/SPEC-RUNTIME-CONTRACT.md` (local-only by repo",
        "policy); the machine-readable half is tracked here.",
        "",
        f"Version: **{', '.join(conf['versions']) or '—'}**",
        "",
        "## Contract suite (52 cases)",
        "",
        "| Platform | Build | Cases | PASS | FAIL | BLOCKED | Capability posture | verify_dist |",
        "|---|---|---:|---:|---:|---:|:--:|:--:|",
    ]
    for row in conf["rows"]:
        suite = (
            f"{row['cases']} | {row['pass']} | {row['fail']} | {row['blocked']}"
            if row["cases"] is not None
            else "— | — | — | —"
        )
        lines.append(
            f"| `{row['platform']}` ({row['os']}/{row['arch']}) | {row['build']} | {suite} "
            f"| {mark(row['posture_ok'])} | {row['verify_dist'] or '—'} |"
        )
    lines += [
        "",
        "The macOS row is the merged release report (`sofuu conformance --suite`);",
        "the Linux rows are each platform's own posture report (their suites run in",
        "CI's report mode). Any BLOCKED cases are named in",
        "[`PROGRESS.md`](PROGRESS.md) — reserved multi-device runners, not failures.",
        "",
        "## Live capability posture",
        "",
    ]
    merged = next((r for r in conf["rows"] if suite_of_report(r)), conf["rows"][0])
    posture = merged.get("posture") or {}
    if posture:
        lines += [
            f"From the `{merged['platform']}` report:",
            "",
            "| Primitive | Claimed | Live | OK |",
            "|---|---|---|:--:|",
        ]
        for name in sorted(posture):
            entry = posture[name] if isinstance(posture[name], dict) else {}
            lines.append(
                f"| {name} | {entry.get('claimed', '—')} | {entry.get('live', '—')} "
                f"| {mark(bool(entry.get('ok')))} |"
            )
    else:
        lines.append("No posture entries recorded.")
    lines += [
        "",
        "## Stability index",
        "",
        "Every public symbol carries exactly one tier (T1 guaranteed · T2 supported ·",
        "T3 experimental). Source: `contract/stability.json`; the per-symbol tables are",
        "in [`STABILITY-INDEX.md`](STABILITY-INDEX.md).",
        "",
        "| Surface | T1 | T2 | T3 | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for surface in sorted(stability["surfaces"]):
        row = stability["surfaces"][surface]
        lines.append(f"| {surface} | {row['T1']} | {row['T2']} | {row['T3']} | {row['total']} |")
    total = stability["total"]
    lines += [
        f"| **total** | **{total['T1']}** | **{total['T2']}** | **{total['T3']}** | **{total['total']}** |",
        "",
        f"Open stability gaps (target tier not yet met): **{stability['gaps']}**.",
        "",
        "## Reproduce it yourself",
        "",
        "```bash",
        "make deps                                       # vendored libuv (first build only)",
        "make && make libsofuu",
        "bash scripts/contract/run_contract_tests.sh      # the 52-case board",
        "bash scripts/verify-dist.sh dist ./sofuu         # install integrity",
        "bash scripts/conformance_report.sh               # merged release report",
        "python3 scripts/scoreboard.py --check --live-report dist/conformance-report.json",
        "```",
        "",
        "Expected on a standard Sofuu build (QTSQ is a mandatory core component):",
        "52 cases, 42 PASS / 0 FAIL / 10 BLOCKED, memory `full`",
        "(see `contract/README.md` for the per-case prerequisites).",
        "",
        "`dist/SHA256SUMS` is stamped by the first `verify-dist` run (a local file,",
        "not tracked). Delete it after changing build flags so the artifact set is",
        "re-stamped instead of failing as a changed set.",
        "",
        "## Regression gates",
        "",
        "CI runs `python3 scripts/scoreboard.py --check --live-report dist/conformance-report.json`.",
        "It fails when: any contract case FAILs; PASS drops below the frozen floor;",
        "the capability posture or `verify_dist` is not ok; a parity headline is more",
        f"than {num(PARITY_TOLERANCE_PCT)}% below its frozen baseline; or the generated",
        "pages drift from their sources. Regressions are accepted only by editing the",
        "sources and re-freezing deliberately (`--freeze-baseline`).",
        "",
        "## External adoption",
        "",
        f"- External deployments: **{len(adoption.get('deployments', []))}**",
        f"- Independent conformance proofs: **{len(adoption.get('independent_conformance_proofs', []))}**",
        f"- Upgrades across a minor version: **{len(adoption.get('upgrade_drills', []))}**",
        "",
        "The P6 exit is open until an outside team upgrades without a support",
        "conversation and a third party proves conformance unaided. The protocol and",
        "the ledger are in [`adoption/`](adoption/README.md).",
        "",
        f"Performance: {parity['measured']} models measured, "
        f"pp parity {parity['pass_pp']}/{parity['measured']}, "
        f"tg parity {parity['pass_tg']}/{parity['measured']} — "
        "see [`SCOREBOARD.md`](SCOREBOARD.md).",
        "",
    ]
    return "\n".join(lines)


def suite_of_report(row: dict[str, Any]) -> bool:
    return row.get("cases") is not None


def render_ledger(adoption: dict[str, Any]) -> str:
    lines = [
        "# External adoption ledger (generated)",
        "",
        "Source of truth: [`ledger.json`](ledger.json). Add entries there, then run",
        "`python3 scripts/scoreboard.py`. The P6 protocol is in",
        "[`README.md`](README.md).",
        "",
        "## Deployments",
        "",
        "| Deployment | Surface | Hard requirement | Started | Status | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    deployments = adoption.get("deployments", [])
    if deployments:
        for d in deployments:
            lines.append(
                f"| {d.get('name', '—')} | {d.get('surface', '—')} | {d.get('requirement', '—')} "
                f"| {d.get('started', '—')} | {d.get('status', '—')} | {d.get('evidence', '—')} |"
            )
    else:
        lines.append("| — | — | — | — | **none yet** | P6 exit open |")
    lines += ["", "## Independent conformance proofs", ""]
    proofs = adoption.get("independent_conformance_proofs", [])
    if proofs:
        lines += ["| Team | Platform | Report | Date |", "|---|---|---|---|"]
        for p in proofs:
            lines.append(
                f"| {p.get('team', '—')} | {p.get('platform', '—')} "
                f"| {p.get('report', '—')} | {p.get('date', '—')} |"
            )
    else:
        lines.append("None yet — no third party has run the suite unaided.")
    lines += ["", "## Upgrades across one minor version", ""]
    drills = adoption.get("upgrade_drills", [])
    if drills:
        lines += ["| Team | From → to | Support conversations | Date | Notes |", "|---|---|---:|---|---|"]
        for d in drills:
            lines.append(
                f"| {d.get('team', '—')} | {d.get('from_version', '—')} → {d.get('to_version', '—')} "
                f"| {d.get('support_conversations', '—')} | {d.get('date', '—')} | {d.get('notes', '')} |"
            )
    else:
        lines.append("None yet.")
    lines.append("")
    return "\n".join(lines)


# ── documents ──────────────────────────────────────────────────────────────


def build_documents() -> dict[Path, str]:
    parity = summarize_parity(load_parity())
    reports = load_conformance()
    conf = summarize_conformance(reports)
    stability = summarize_stability()
    adoption = load_adoption()
    digest = {
        "schema": 1,
        "kind": "sofuu-scoreboard",
        "parity": parity,
        "conformance": {
            "versions": conf["versions"],
            "rows": [
                {k: v for k, v in row.items() if k not in ("posture",)}
                for row in conf["rows"]
            ],
        },
        "stability": stability,
        "adoption": {
            "deployments": len(adoption.get("deployments", [])),
            "independent_conformance_proofs": len(adoption.get("independent_conformance_proofs", [])),
            "upgrade_drills": len(adoption.get("upgrade_drills", [])),
        },
    }
    return {
        OUT_JSON: dump_json(digest),
        OUT_SCOREBOARD: render_scoreboard(parity, conf),
        OUT_CONFORMANCE: render_conformance(parity, conf, stability, adoption),
        OUT_LEDGER: render_ledger(adoption),
    }


def baseline_from(parity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": 1,
        "note": "Frozen acceptance floor. Update deliberately with --freeze-baseline.",
        "parity_tolerance_pct": PARITY_TOLERANCE_PCT,
        "conformance": {
            "min_reports": 3,
            "min_cases": 52,
            "min_pass": 42,
            "max_fail": 0,
            "require_posture_ok": True,
            "require_verify_dist_ok": True,
        },
        "parity": {
            model: {
                "pp": round(row["sofuu_pp"], 3),
                "tg": round(row["sofuu_tg"], 3),
                "pass_pp": row["pass_pp"],
                "pass_tg": row["pass_tg"],
                "ts": row["ts"],
            }
            for model, row in sorted(parity["headline"].items())
        },
    }


# ── check ──────────────────────────────────────────────────────────────────


def check_report(
    report: dict[str, Any],
    rules: dict[str, Any],
    label: str,
    problems: list[str],
    require_suite: bool,
) -> None:
    if rules.get("require_posture_ok", True) and not posture_ok(report):
        problems.append(f"{label}: capability posture is not ok")
    suite = suite_of(report)
    if suite is None:
        # Platform jobs report posture only; the merged release report (and a
        # live CI report) must carry the board.
        if require_suite:
            problems.append(f"{label}: no contract_suite block")
        return
    cases = suite.get("cases")
    passed = suite.get("pass")
    failed = suite.get("fail")
    if failed is None or passed is None or cases is None:
        problems.append(f"{label}: suite summary is incomplete")
        return
    if failed > rules.get("max_fail", 0):
        problems.append(f"{label}: {failed} contract FAIL(s)")
    floor = rules.get("min_pass", 42)
    if passed < floor:
        problems.append(f"{label}: {passed} PASS is below the frozen floor {floor}")
    if cases < rules.get("min_cases", 0):
        problems.append(f"{label}: only {cases} cases ran (floor {rules['min_cases']})")


def run_check(live_reports: list[Path]) -> list[str]:
    problems: list[str] = []
    docs = build_documents()

    if not BASELINE.is_file():
        die(f"missing {BASELINE.relative_to(ROOT)}; run --freeze-baseline once", 2)
    baseline = json.loads(BASELINE.read_text())
    rules = baseline.get("conformance") or {}

    # 1. Generated files must be current (no hand edits, no forgotten regen).
    for path, text in docs.items():
        current = path.read_text() if path.is_file() else None
        if current != text:
            problems.append(f"{path.relative_to(ROOT)} is stale — run python3 scripts/scoreboard.py")

    # 2. Conformance snapshots.
    reports = load_conformance()
    if len(reports) < rules.get("min_reports", 0):
        problems.append(
            f"only {len(reports)} conformance report(s); floor is {rules.get('min_reports')}"
        )
    for report in reports:
        check_report(report, rules, f"conformance/{report['_platform']}", problems, require_suite=False)
    if not any(suite_of(report) is not None for report in reports):
        problems.append("no conformance snapshot carries the contract suite")
    if rules.get("require_verify_dist_ok", True) and not any(
        verify_dist(report) == "ok" for report in reports
    ):
        problems.append("no conformance report carries install_integrity.verify_dist == ok")

    # 3. The freshly produced report (CI runs this alongside the gate).
    for live in live_reports:
        if not live.is_file():
            problems.append(f"live report {live} is missing")
            continue
        try:
            report = json.loads(live.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"live report {live}: invalid JSON ({e})")
            continue
        check_report(report, rules, f"live:{live.name}", problems, require_suite=True)
        if rules.get("require_verify_dist_ok", True) and verify_dist(report) != "ok":
            problems.append(f"live:{live.name}: install_integrity.verify_dist is not ok")

    # 4. Parity regressions against the frozen floor.
    parity = summarize_parity(load_parity())
    tolerance = float(baseline.get("parity_tolerance_pct", PARITY_TOLERANCE_PCT)) / 100.0
    for model, base in sorted((baseline.get("parity") or {}).items()):
        current = parity["headline"].get(model)
        if current is None:
            problems.append(f"parity: {model} disappeared from the ledger")
            continue
        if base.get("pass_pp") and not current["pass_pp"]:
            problems.append(f"parity: {model} lost pp parity ({num(current['sofuu_pp'])} < {num(current['ll_pp'])})")
        if base.get("pass_tg") and not current["pass_tg"]:
            problems.append(f"parity: {model} lost tg parity ({num(current['sofuu_tg'])} < {num(current['ll_tg'])})")
        for metric in ("pp", "tg"):
            floor = float(base.get(metric, 0.0)) * (1.0 - tolerance)
            if floor > 0 and current[f"sofuu_{metric}"] < floor:
                problems.append(
                    f"parity: {model} {metric} regressed to {num(current[f'sofuu_{metric}'])} "
                    f"(floor {num(floor)} = baseline {num(base.get(metric))} - {num(PARITY_TOLERANCE_PCT)}%)"
                )
    return problems


# ── main ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish and gate the Sofuu conformance/parity receipts.")
    ap.add_argument("--check", action="store_true", help="validate drift + regressions (CI gate)")
    ap.add_argument(
        "--live-report",
        action="append",
        default=[],
        type=Path,
        help="a freshly generated conformance report to gate (repeatable)",
    )
    ap.add_argument(
        "--freeze-baseline",
        action="store_true",
        help="accept the current headline as the regression floor, then regenerate",
    )
    args = ap.parse_args()

    if args.check:
        problems = run_check(list(args.live_report))
        if problems:
            print("SCOREBOARD: FAILED", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print("SCOREBOARD: OK (sources current, no conformance or parity regression)")
        return 0

    if args.freeze_baseline:
        parity = summarize_parity(load_parity())
        baseline = baseline_from(parity)
        BASELINE.write_text(dump_json(baseline))
        print(
            f"scoreboard: baseline frozen ({len(parity['headline'])} models) -> "
            f"{BASELINE.relative_to(ROOT)}"
        )

    docs = build_documents()
    for path, text in docs.items():
        path.write_text(text)
        print(f"scoreboard: wrote {path.relative_to(ROOT)}")
    if not BASELINE.is_file():
        parity = summarize_parity(load_parity())
        BASELINE.write_text(dump_json(baseline_from(parity)))
        print(f"scoreboard: wrote {BASELINE.relative_to(ROOT)} (initial freeze)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
