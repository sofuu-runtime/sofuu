#!/usr/bin/env python3
"""Generate and validate the runtime-contract artifacts.

Reads:
  contract/matrix.json       (audited surface matrix, source of truth)
  contract/stability.json    (per-symbol stability tiers, source of truth)
  scripts/abi_symbols.txt    (live ABI baseline, for drift detection)

Writes:
  contract/SURFACE-MATRIX.md
  contract/STABILITY-INDEX.md
  contract/knobs.json        (rebuilt from the tree)
  contract/KNOBS.md

Exit codes:
  0  artifacts generated and valid
  1  validation failed (the message names every failing item)

This script only reads source files; it never edits them.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contract"

SURFACES = ["js", "cli", "http", "cabi", "swift_kotlin", "mcp"]
PRIMITIVES = ["session", "stream", "tool", "usage", "memory", "model", "agent", "transport"]
STATUSES = {"full", "partial", "unavailable", "n/a"}
TIERS = {"T1", "T2", "T3"}

# ── knob scan ────────────────────────────────────────────────────────────────
SCAN_DIRS = [ROOT / "src", ROOT / "bindings"] + sorted(ROOT.glob("crates/*/src"))

PATTERNS = [
    re.compile(r'getenv\s*\(\s*"(SOFUU_[A-Z0-9_]+)"'),
    re.compile(r'env::var\s*\(\s*"(SOFUU_[A-Z0-9_]+)"'),
    re.compile(r'process\.env\.(SOFUU_[A-Z0-9_]+)'),
]

BUILD = {"SOFUU_QTSQ_DIR", "SOFUU_QTSQ_LIB", "SOFUU_CURL_STATIC_DIR", "SOFUU_UV_DIR",
         "SOFUU_ZLIB_DIR", "SOFUU_ABI_ALLOW_NEW", "SOFUU_PROJECT"}
TEST_PREFIX = ("SOFUU_TEST_", "SOFUU_BENCH_", "SOFUU_EMB_", "SOFUU_EMBED_EVAL_", "SOFUU_IMG_")
TEST = {"SOFUU_SPARKTEST_VARIANT", "SOFUU_DEMO_KEY", "SOFUU_DET_GGUF"}
DEBUG_MARKERS = ("DEBUG", "PROFILE", "TRACE", "CAPTURE", "TIMING", "METRICS", "SNAP",
                 "PROBE", "_PROF", "HAZARD")
PUBLIC = {"SOFUU_NO_SHELL", "SOFUU_STORAGE", "SOFUU_BRAIN_KEY", "SOFUU_BRAIN_MAX_RECORDS",
          "SOFUU_BRAIN_EF_SEARCH", "SOFUU_SERVE_TOKEN", "SOFUU_HTTP_HEADERS_MS",
          "SOFUU_HTTP_IDLE_MS", "SOFUU_STALL_TIMEOUT_SECS", "SOFUU_NUM_THREADS",
          "SOFUU_KIVI", "SOFUU_KIVI_BITS", "SOFUU_DYNAMIC_FFN", "SOFUU_RESIDENT_WEIGHTS",
          "SOFUU_NO_ML", "SOFUU_NO_TILES", "SOFUU_KV_F16", "SOFUU_KV_DTYPE", "SOFUU_MCP_CONFIG",
          "SOFUU_POLICY"}
NON_NUMERIC = {"SOFUU_AFFINITY", "SOFUU_NO_TOOLS"}
NUMERIC_RE = re.compile(
    r"TILE|GEMM|GEMV|COALESCED|FUSE|PRE8|GQA|Q4|Q6|Q3|NIBBLE|SCALE_SIDECAR|SPEC_|"
    r"LOOKAHEAD|FA2|MMQ|DECODE_|PREFILL|ROPE|KV_F16|KV_DTYPE|KIVI|RESIDENT_WEIGHTS|"
    r"GPU_ATTN|GPU_SKIP|GPU_SPEC|ATTN_|NO_BARRIER|NO_TILES|GEMM_TYPES|T143|PHASE|ENTROPY"
)

# R3.5: a numeric knob MUST map to a determinism tier. Only receipts-backed
# assignments live here; everything else is `unsure` and stays an open item.
DETERMINISM_EXACT = {
    "SOFUU_NO_TILES",       # the reference route itself
    "SOFUU_ATTN_SERIAL",    # opt-in deterministic prefill attention (a77d4540c)
    "SOFUU_GEMM_I8MM",      # bit-identical to the SDOT kernel (PLAN-TARGET-1200-1500 §23)
    "SOFUU_Q4K_COALESCED",  # fleet receipts: full-text bit-identical (battery 11.5)
}
DETERMINISM_TOLERANT = {
    "SOFUU_GPU_GEMM_MMA",   # fp16 tiles, published 2e-2 abs/scale bound (parity LEDGER #7)
    # Quantized KV on the GPU decode path is OPT-IN (the default is the f32
    # cache). --kv-dtype 8 holds the ~1% ppl band on the local fleet (max
    # +0.23%) at 3.2x less KV; --kv-dtype 4 does not (SmolLM2 +7.7%).
    # f32 is the exact contract (byte-identical dumps); see
    # contract/KV4-OPERATOR-NOTE.md for every receipt.
    "SOFUU_KV_DTYPE",
}
DETERMINISM_BEST_EFFORT_PREFIX = "SOFUU_ENTROPY"  # sampler params: opt-in, change output


def determinism_tier(name: str, numeric: str):
    """exact | tolerant | best-effort | unsure | None (non-numeric knob)."""
    if numeric != "yes":
        return None
    if name in DETERMINISM_EXACT or name.startswith("SOFUU_SPEC_"):
        # speculative decode verifies exactly — token-identical receipts
        return "exact"
    if name in DETERMINISM_TOLERANT:
        return "tolerant"
    if name.startswith(DETERMINISM_BEST_EFFORT_PREFIX):
        return "best-effort"
    return "unsure"


def scan_knobs() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in {".c", ".cc", ".cpp", ".h", ".hpp",
                                                         ".m", ".mm", ".rs", ".js"}:
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            rel = str(path.relative_to(ROOT))
            for lineno, line in enumerate(text.splitlines(), 1):
                for pat in PATTERNS:
                    for name in pat.findall(line):
                        found.setdefault(name, [])
                        ev = f"{rel}:{lineno}"
                        if ev not in found[name]:
                            found[name].append(ev)
    return found


def classify(name: str) -> tuple[str, str, str]:
    """Return (class, numeric_effect, stability_tier)."""
    if name in BUILD:
        return "build", "no", "T3"
    if name in TEST or name.startswith(TEST_PREFIX):
        return "test", "no", "T3"
    if name in PUBLIC:
        numeric = "yes" if NUMERIC_RE.search(name) else "no"
        tier = "T2"
        return "public", numeric, tier
    if any(m in name for m in DEBUG_MARKERS):
        return "debug", "no", "T3"
    if name in NON_NUMERIC:
        return "tuning", "no", "T3"
    numeric = "yes" if NUMERIC_RE.search(name) else "unknown"
    return "tuning", numeric, "T3"


def build_knobs() -> dict:
    raw = scan_knobs()
    entries = []
    for name in sorted(raw):
        cls, numeric, tier = classify(name)
        entries.append({
            "name": name, "class": cls, "numeric_effect": numeric,
            "stability_tier": tier,
            "determinism_tier": determinism_tier(name, numeric),
            "evidence": raw[name][:3],
        })
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    unknown_numeric = [e["name"] for e in entries if e["numeric_effect"] == "unknown"]
    unsure_tier = [e["name"] for e in entries if e["determinism_tier"] == "unsure"]
    return {
        "schema": 1,
        "generated_from": "scan of src/, crates/*/src/, bindings/ (getenv | env::var | process.env)",
        "count": len(entries),
        "counts_by_class": counts,
        "unclassified_numeric_effect": unknown_numeric,
        "unsure_determinism_tier": unsure_tier,
        "knobs": entries,
    }


# ── validation ───────────────────────────────────────────────────────────────
def load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def validate_matrix(matrix: dict) -> list[str]:
    errs: list[str] = []
    if sorted(matrix.get("surfaces", [])) != sorted(SURFACES):
        errs.append(f"matrix.surfaces must be {SURFACES}")
    if sorted(matrix.get("primitives", [])) != sorted(PRIMITIVES):
        errs.append(f"matrix.primitives must be {PRIMITIVES}")

    seen: dict[tuple[str, str], dict] = {}
    for cell in matrix.get("cells", []):
        key = (cell.get("primitive"), cell.get("surface"))
        if key in seen:
            errs.append(f"duplicate cell {key}")
        seen[key] = cell
        status = cell.get("status")
        if status not in STATUSES:
            errs.append(f"{key}: invalid status {status!r} (must be one of {sorted(STATUSES)})")
        if status == "partial" and not cell.get("missing"):
            errs.append(f"{key}: partial without a `missing` list")
        if status == "unavailable" and not cell.get("reason"):
            errs.append(f"{key}: unavailable without a `reason`")
        if status == "full" and not cell.get("evidence"):
            errs.append(f"{key}: full without evidence")
        if status != "n/a" and not cell.get("tests"):
            errs.append(f"{key}: no test ids")
        for ev in cell.get("evidence", []):
            rel = ev.split(":")[0]
            if not (ROOT / rel).exists():
                errs.append(f"{key}: evidence path missing: {rel}")

    for p in PRIMITIVES:
        for s in SURFACES:
            if (p, s) not in seen:
                errs.append(f"missing cell {p} x {s}")
    return errs


def validate_stability(stability: dict) -> list[str]:
    errs: list[str] = []
    surfaces = stability.get("surfaces", {})

    for group, payload in surfaces.items():
        for section, entries in payload.items():
            if group == "js" and section == "namespaces":
                for ns, ns_payload in entries.items():
                    if ns_payload.get("tier") not in TIERS:
                        errs.append(f"stability.js.namespaces.{ns}: invalid tier "
                                    f"{ns_payload.get('tier')!r}")
                    for member, tier in ns_payload.get("members", {}).items():
                        if tier not in TIERS:
                            errs.append(f"stability.js.{ns}.{member}: invalid tier {tier!r}")
                continue
            for key, tier in entries.items():
                if tier not in TIERS:
                    errs.append(f"stability.{group}.{section}.{key}: invalid tier {tier!r}")

    abi_file = ROOT / "scripts" / "abi_symbols.txt"
    if not abi_file.exists():
        errs.append("scripts/abi_symbols.txt missing")
        return errs
    live = {ln.strip() for ln in abi_file.read_text().splitlines() if ln.strip()}
    declared = set(surfaces.get("cabi", {}).get("symbols", {}))
    for sym in sorted(live - declared):
        errs.append(f"ABI symbol not in stability.json: {sym}")
    for sym in sorted(declared - live):
        errs.append(f"stability.json declares a non-exported ABI symbol: {sym}")
    return errs


# ── rendering ────────────────────────────────────────────────────────────────
def render_matrix(matrix: dict) -> str:
    cells = {(c["primitive"], c["surface"]): c for c in matrix["cells"]}
    counts = {s: 0 for s in STATUSES}
    for c in matrix["cells"]:
        counts[c["status"]] += 1

    out = [
        "# Surface matrix (generated)",
        "",
        f"Generated from [`contract/matrix.json`](matrix.json) — audit date {matrix['audit_date']}.",
        "Do not edit this file by hand; edit the JSON and run",
        "`python3 scripts/contract/gen_contract_docs.py`.",
        "",
        "**Legend:** " + " · ".join(f"`{k}` = {v}" for k, v in matrix["status_legend"].items()),
        "",
        f"Cells: {len(matrix['cells'])} — "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        "",
        "| primitive | " + " | ".join(SURFACES) + " |",
        "|---|" + "---|" * len(SURFACES),
    ]
    for p in PRIMITIVES:
        row = [f"**{p}**"]
        for s in SURFACES:
            c = cells[(p, s)]
            cell = c["status"]
            if cell == "partial":
                cell = "partial*"
            row.append(cell)
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("`partial*` — the missing operations are listed per cell below.")
    out.append("")

    out.append("## Cell detail")
    out.append("")
    for p in PRIMITIVES:
        out.append(f"### {p}")
        out.append("")
        for s in SURFACES:
            c = cells[(p, s)]
            line = f"- **{s}** — `{c['status']}`"
            if c.get("missing"):
                line += "; missing: " + ", ".join(c["missing"])
            if c.get("reason"):
                line += "; reason: " + c["reason"]
            if c.get("note"):
                line += "; note: " + c["note"]
            if c.get("evidence"):
                line += "; evidence: " + ", ".join(f"`{e}`" for e in c["evidence"])
            if c.get("tests"):
                line += "; tests: " + ", ".join(c["tests"])
            out.append(line)
        out.append("")
    return "\n".join(out) + "\n"


def render_stability(stability: dict) -> str:
    surfaces = stability["surfaces"]
    out = [
        "# Stability index (generated)",
        "",
        f"Generated from [`contract/stability.json`](stability.json) — audit date {stability['audit_date']}.",
        "Every public symbol carries exactly one tier. Do not edit by hand.",
        "",
        "**Legend:** " + " · ".join(f"`{k}` = {v}" for k, v in stability["tier_legend"].items()),
        "",
    ]

    def tier_table(title: str, entries: dict[str, str]) -> None:
        counts = {t: sum(1 for v in entries.values() if v == t) for t in sorted(TIERS)}
        out.append(f"## {title} ({len(entries)} entries — " +
                   ", ".join(f"{t} {n}" for t, n in counts.items()) + ")")
        out.append("")
        out.append("| symbol | tier |")
        out.append("|---|---|")
        for key in sorted(entries):
            out.append(f"| `{key}` | {entries[key]} |")
        out.append("")

    tier_table("C ABI symbols", surfaces["cabi"]["symbols"])
    tier_table("CLI commands", surfaces["cli"]["commands"])
    tier_table("HTTP routes", surfaces["http"]["routes"])
    tier_table("Config keys", surfaces["config"]["keys"])

    out.append("## JS namespaces")
    out.append("")
    out.append("| namespace | tier | members | T1 members |")
    out.append("|---|---|---|---|")
    namespaces = surfaces["js"]["namespaces"]
    for ns in sorted(namespaces):
        payload = namespaces[ns]
        members = payload["members"]
        t1 = sum(1 for v in members.values() if v == "T1")
        out.append(f"| `{ns}` | {payload['tier']} | {len(members)} | {t1} |")
    out.append("")

    for ns in sorted(namespaces):
        payload = namespaces[ns]
        if len(payload["members"]) <= 3:
            continue
        out.append(f"### `{ns}` members")
        out.append("")
        out.append("| member | tier |")
        out.append("|---|---|")
        for m in sorted(payload["members"]):
            out.append(f"| `{m}` | {payload['members'][m]} |")
        out.append("")

    if stability.get("gaps"):
        out.append("## Known gaps (planned, not yet public)")
        out.append("")
        out.append("| symbol | surface | target tier | phase |")
        out.append("|---|---|---|---|")
        for g in stability["gaps"]:
            out.append(f"| `{g['symbol']}` | {g['surface']} | {g['target_tier']} | {g['phase']} |")
        out.append("")
    return "\n".join(out) + "\n"


def render_knobs(knobs: dict) -> str:
    out = [
        "# Runtime env knobs (generated)",
        "",
        f"Generated from a source scan — {knobs['count']} knobs. Class counts: "
        + ", ".join(f"{k} {v}" for k, v in sorted(knobs["counts_by_class"].items())) + ".",
        "Classes: `public` (user-facing), `tuning` (performance knobs), `debug`,",
        "`test`, `build`. `numeric_effect=yes` means the knob changes results and",
        "therefore must map to a determinism tier (R3.5). `unknown` is an open item.",
        "",
    ]
    if knobs["unclassified_numeric_effect"]:
        out.append(f"**Open items:** {len(knobs['unclassified_numeric_effect'])} knobs have "
                   "`numeric_effect=unknown` and need a tier decision.")
        out.append("")
    if knobs.get("unsure_determinism_tier"):
        out.append(f"**Open items (R3.5):** {len(knobs['unsure_determinism_tier'])} numeric knobs "
                   "have `determinism_tier=unsure` — each needs a receipt before it can map to a "
                   "declared path tier.")
        out.append("")
    out.append("| knob | class | numeric effect | determinism tier | stability tier | first evidence |")
    out.append("|---|---|---|---|---|---|")
    for e in knobs["knobs"]:
        ev = e["evidence"][0] if e["evidence"] else ""
        det = e.get("determinism_tier") or "—"
        out.append(f"| `{e['name']}` | {e['class']} | {e['numeric_effect']} | {det} | "
                   f"{e['stability_tier']} | `{ev}` |")
    out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    matrix = load(CONTRACT / "matrix.json")
    stability = load(CONTRACT / "stability.json")

    errors = validate_matrix(matrix) + validate_stability(stability)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    knobs = build_knobs()
    (CONTRACT / "knobs.json").write_text(json.dumps(knobs, indent=2) + "\n")
    (CONTRACT / "SURFACE-MATRIX.md").write_text(render_matrix(matrix))
    (CONTRACT / "STABILITY-INDEX.md").write_text(render_stability(stability))
    (CONTRACT / "KNOBS.md").write_text(render_knobs(knobs))

    print(f"OK matrix: {len(matrix['cells'])} cells "
          f"({', '.join(f'{k}={v}' for k, v in sorted({s: sum(1 for c in matrix['cells'] if c['status'] == s) for s in STATUSES}.items()))})")
    print(f"OK stability: {sum(len(v) for g in stability['surfaces'].values() for v in g.values())} entries")
    print(f"OK knobs: {knobs['count']} ({knobs['counts_by_class']})")
    if knobs["unclassified_numeric_effect"]:
        print(f"OPEN: {len(knobs['unclassified_numeric_effect'])} knobs with numeric_effect=unknown")
    if knobs.get("unsure_determinism_tier"):
        print(f"OPEN: {len(knobs['unsure_determinism_tier'])} numeric knobs with determinism_tier=unsure")
    print("wrote contract/SURFACE-MATRIX.md, contract/STABILITY-INDEX.md, contract/knobs.json, contract/KNOBS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
