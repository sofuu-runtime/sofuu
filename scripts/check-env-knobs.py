#!/usr/bin/env python3
"""scripts/check-env-knobs.py - documented knobs that no code reads (5.5).

A SOFUU_* name that appears in docs/ and nowhere in the source is either a
feature that was removed (the doc is a dated receipt and keeps its history) or a
knob somebody believes in and cannot set. Either way the list may not grow:
every new documented-only name is a promise to a reader that the binary does not
keep.

  python3 scripts/check-env-knobs.py            # check against the baseline
  python3 scripts/check-env-knobs.py --list     # print the whole inventory
  python3 scripts/check-env-knobs.py --update   # rewrite the baseline

Exit: 0 at or below baseline - 1 when a new stale name appears - 2 usage error.
"""
import pathlib
import re
import subprocess
import sys

NL = chr(10)
BASELINE = pathlib.Path(".ci/env-knobs-baseline.txt")
NAME = re.compile(r"SOFUU_[A-Z0-9_]+")
DOES_NOT_EXIST = "generated, not a knob"  # SOFUU_CUDA_PTX_GZ is a C symbol


def grep_names(roots):
    """Every SOFUU_* token under the given roots.

    Shelling out to grep -rI rather than walking with rglob: the first version
    filtered by file extension and silently skipped .mm and .metal, which is
    where a chunk of this project's env reads actually live. A suffix allowlist
    that misses a language reports false positives, and a false positive here
    accuses a live knob of being dead.
    """
    names = set()
    for root in roots:
        if not pathlib.Path(root).exists():
            continue
        out = subprocess.run(
            ["grep", "-rIhoE", r"SOFUU_[A-Z0-9_]+", root],
            capture_output=True,
            text=True,
        ).stdout
        names.update(NAME.findall(out))
    return names


def main(argv):
    update = "--update" in argv
    listing = "--list" in argv
    if any(a not in ("--update", "--list") for a in argv):
        print(__doc__.splitlines()[0])
        return 2

    source = grep_names(["src", "crates", "scripts", "Makefile"])
    documented = grep_names(["docs"])
    stale = sorted(n for n in documented - source if n not in source)

    if listing:
        for n in stale:
            print(n)
        print(f"{len(stale)} documented-only names")
        return 0

    if update:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# SOFUU_* names that appear in docs/ and nowhere in the source.",
            "# A ratchet: this list may shrink, never grow. Every entry is either a",
            "# dated receipt (docs/ is history, not a promise) or a mistake worth",
            "# fixing. Add a name here only with a comment saying which it is.",
        ]
        lines += stale
        BASELINE.write_text(NL.join(lines) + NL)
        print(f"env-knobs: baseline written ({len(stale)})")
        return 0

    known = set()
    if BASELINE.exists():
        for line in BASELINE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                known.add(line)
    new = [n for n in stale if n not in known]
    retired = len(known - set(stale))
    if new:
        print(f"env-knobs: FAIL - {len(new)} documented-only knob(s) not in the baseline:")
        for n in new:
            print(f"  {n}")
        print("Either wire the knob, fix the name in the doc, or mark the doc entry"
              "as removed, then run --update.")
        return 1
    note = f"{retired} below baseline - lower it with --update" if retired else "at baseline"
    print(f"env-knobs: OK - {len(stale)} documented-only names, {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
