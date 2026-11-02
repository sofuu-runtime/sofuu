#!/usr/bin/env python3
"""scripts/check-js-swallow.py - the empty-catch ratchet (PLAN-AUDIT-REMEDIATION 5.6).

A swallow is a catch block with nothing in it: no log, no rethrow, no fallback.
Some are correct - a best-effort status read whose failure genuinely does not
change the answer - but silence is indistinguishable from a forgotten handler,
so the rule is:

  An empty catch is allowed only when a comment says what is being ignored and
  why nothing depends on it.  catch (_) { /* status probe: null means unknown */ }

This gate is a RATCHET, not a ban: the existing ones are a backlog to retire
deliberately, and every fix lowers the baseline. It counts per file as well as
in total, because a single global number lets one file gain a swallow while
another retires one, and the count never moves. A new file with a swallow in it
fails even when the total is unchanged.

  python3 scripts/check-js-swallow.py            # check against the baseline
  python3 scripts/check-js-swallow.py --list     # print every offending site
  python3 scripts/check-js-swallow.py --update   # lower the baseline to today

Exit: 0 at or below baseline - 1 above it - 2 usage error.
"""
import pathlib
import re
import sys

NL = chr(10)
BASELINE = pathlib.Path(".ci/js-swallow-baseline.txt")
ROOT = pathlib.Path("src/js")
# catch (e) {} plus catch {} - an optional binding, then nothing at all inside.
EMPTY = re.compile(r"catch\s*(?:\([^)]*\))?\s*\{[\s\n]*\}", re.MULTILINE)
# Only .js source; vendored files are somebody else's backlog.
SKIP = ("vendor", "node_modules", "dist")
# The intentional shape this gate teaches: a comment inside the block.
COMMENTED = re.compile(r"catch\s*(?:\([^)]*\))?\s*\{\s*(?://[^\n]*|/\*.*?\*/)\s*\}", re.DOTALL)


def scan():
    """Return (per-file counts, list of (path, line))."""
    counts = {}
    sites = []
    for path in sorted(ROOT.rglob("*.js")):
        if any(s in path.parts for s in SKIP):
            continue
        text = path.read_text(errors="ignore")
        for m in EMPTY.finditer(text):
            if COMMENTED.match(text, m.start()):
                continue
            line = text.count(NL, 0, m.start()) + 1
            sites.append((str(path), line))
            counts[str(path)] = counts.get(str(path), 0) + 1
    return counts, sites


def read_baseline():
    """Return (total, {path: count}). Understands the old single-number file."""
    total = 0
    per_file = {}
    if not BASELINE.exists():
        return 0, per_file
    for line in BASELINE.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        try:
            n = int(value.strip())
        except ValueError:
            continue
        if key in ("swallows", "total"):
            total = n
        else:
            per_file[key] = n
    return total, per_file


def main(argv):
    update = "--update" in argv
    listing = "--list" in argv
    if any(a not in ("--update", "--list") for a in argv):
        print(__doc__.splitlines()[0])
        return 2
    if not ROOT.is_dir():
        print(f"js-swallow: {ROOT} does not exist", file=sys.stderr)
        return 0

    counts, sites = scan()
    total = sum(counts.values())

    if listing:
        for path, line in sites:
            print(f"{path}:{line}")
        print(f"-- {total} un-commented swallow(s)")

    if update:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        out = [
            "# Un-commented empty catch blocks in src/js. A ratchet, not a target:",
            "# lower this whenever a swallow is fixed or explained. The per-file counts",
            "# exist so one file cannot gain a swallow while another retires one.",
            "total: " + str(total),
        ]
        for path in sorted(counts):
            out.append(f"{path}: {counts[path]}")
        out.append("")
        BASELINE.write_text(NL.join(out))
        print(f"js-swallow: baseline written ({total})")
        return 0

    base_total, base_files = read_baseline()
    problems = []
    if total > base_total:
        problems.append(f"total rose {base_total} -> {total}")
    for path in sorted(counts):
        grew = counts[path] > base_files.get(path, 0)
        if grew:
            was = base_files.get(path, "new file")
            problems.append(f"{path}: {was} -> {counts[path]}")

    if problems:
        print(f"js-swallow: FAIL - {total} un-commented swallows (baseline {base_total})")
        for p in problems:
            print(f"  {p}")
        # Only the sites that explain the failure. Dumping all 305 buries the
        # one that matters; a whole-file dump is worth it only for a new file.
        blame = {p.split(":")[0] for p in problems}
        shown = [(path, line) for path, line in sites if path in blame]
        if not shown:
            shown = sites[:40]
        for path, line in shown[:200]:
            print(f"  {path}:{line}")
        if len(shown) > 200:
            print(f"  ... and {len(shown) - 200} more")
        return 1

    slack = base_total - total
    if slack > 0:
        print(f"js-swallow: OK - {total} un-commented swallows, baseline {base_total} "
              f"({slack} below baseline - lower it with --update)")
    else:
        print(f"js-swallow: OK - {total} un-commented swallows, baseline {base_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))