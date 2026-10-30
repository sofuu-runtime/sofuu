#!/usr/bin/env python3
"""Generate a CycloneDX 1.5 SBOM for a Sofuu release from the Cargo graph.

Why this exists: enterprise reviewers ask for a machine-readable dependency
inventory with every release. `cargo-cyclonedx` would do it, but it is another
tool to install and pin in CI; `cargo metadata` already knows every package,
version and licence, so the SBOM is a short transform away.

    python3 scripts/sbom.py [--out dist/sofuu-sbom.json] [--root .]

The output is deterministic apart from the serial number and timestamp:
components are sorted by name then version, so two runs over the same lockfile
diff cleanly.

No third-party Python packages are used (stdlib only), and nothing is
downloaded except what `cargo metadata` already has in its index cache.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

BOM_FORMAT = "CycloneDX"
SPEC_VERSION = "1.5"


def cargo_metadata(root: str) -> dict:
    """Resolve the full dependency graph (offline-friendly after `cargo fetch`)."""
    cmd = ["cargo", "metadata", "--format-version", "1", "--locked"]
    proc = subprocess.run(
        cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        sys.stderr.write(
            "sbom: `cargo metadata` failed (is Cargo.lock present and readable?)\n"
            + proc.stderr
        )
        raise SystemExit(1)
    return json.loads(proc.stdout)


def git_revision(root: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def purl(name: str, version: str) -> str:
    return "pkg:cargo/{}@{}".format(quote(name, safe=""), quote(version, safe=""))


def licence_entry(raw: str) -> dict:
    """CycloneDX wants either a single SPDX id or an SPDX expression."""
    text = raw.strip()
    if not text:
        return {}
    if " " in text or "/" in text or text.startswith("("):
        return {"expression": text}
    return {"license": {"id": text}}


def component(pkg: dict) -> dict:
    name = pkg.get("name") or "unknown"
    version = pkg.get("version") or "0.0.0"
    entry = {
        "type": "library",
        "bom-ref": purl(name, version),
        "name": name,
        "version": version,
        "purl": purl(name, version),
    }
    lic = pkg.get("license")
    if lic:
        entry["licenses"] = [licence_entry(lic)]
    # crates.io registry packages carry a source; path/git members do not.
    source = pkg.get("source")
    if isinstance(source, str) and source.startswith("registry+"):
        entry["properties"] = [{"name": "sofuu:cargo_source", "value": "crates.io"}]
    elif isinstance(source, str) and source.startswith("git+"):
        entry["properties"] = [{"name": "sofuu:cargo_source", "value": source}]
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a CycloneDX SBOM from Cargo metadata")
    ap.add_argument("--out", default="dist/sofuu-sbom.json", help="output path")
    ap.add_argument("--root", default=".", help="repository root (default: .)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    meta = cargo_metadata(root)
    packages = meta.get("packages") or []
    if not packages:
        sys.stderr.write("sbom: cargo metadata returned no packages\n")
        return 1

    by_id = {p.get("id"): p for p in packages if p.get("id")}
    root_pkg = by_id.get((meta.get("resolve") or {}).get("root"))
    if root_pkg is None:
        members = meta.get("workspace_members") or []
        root_pkg = by_id.get(members[0]) if members else packages[0]

    root_component = component(root_pkg)
    root_component["type"] = "application"
    root_component["name"] = "sofuu"

    components = {}
    for pkg in packages:
        comp = component(pkg)
        components[comp["bom-ref"]] = comp
    # The root is metadata.component, not a component of itself.
    components.pop(root_component["bom-ref"], None)

    revision = git_revision(root)
    properties = []
    if revision:
        properties.append({"name": "sofuu:git_revision", "value": revision})

    doc = {
        "bomFormat": BOM_FORMAT,
        "specVersion": SPEC_VERSION,
        "serialNumber": "urn:uuid:{}".format(uuid.uuid4()),
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tools": {
                "components": [
                    {"type": "application", "name": "sofuu-sbom", "version": "1"}
                ]
            },
            "component": root_component,
            "properties": properties,
        },
        "components": sorted(
            components.values(), key=lambda c: (c["name"], c["version"])
        ),
    }

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")

    print(
        "sbom: wrote {} ({} components, CycloneDX {})".format(
            args.out, len(doc["components"]), SPEC_VERSION
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
