#!/usr/bin/env python3
"""scripts/check-js-swallow.py - the empty-catch ratchet (PLAN-AUDIT-REMEDIATION 5.6).

A swallow is a catch block with nothing in it: no log, no rethrow, no fallback.
Some are correct - a best-effort status read whose failure genuinely does not
change the answer - but silence is indistinguishable from a forgotten handler,
so the rule is:

  An empty catch is allowed only when a comment says what is being ignored and
  why nothing depends on it.  catch (_) { /* status probe: null means unknown */ }

This gate counts un-commented swallows, compares the count to
.ci/js-swallow-baseline.txt, and fails when the number grows. It is a ratchet,
not a ban: the existing ones are a backlog to retire deliberately, and every fix
lowers the baseline.

  python3 scripts/check-js-swallow.py            # check against the baseline
  python3 scripts/check-js-swallow.py --list     # print every offending site
  python3 scripts/check-js-swallow.py --update   # lower the baseline to today

Exit: 0 at or below baseline · 1 above it · 2 usage error.
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


def main(argv):
    update = "--update" in argv
    listing = "--list" in argv
    if any(a not in ("--update", "--list") for a in argv):
        print(__doc__.splitlines()[0])
        return 2
    if not ROOT.is_dir():
        print(f"js-swallow: {ROOT} does not exist", file=sys.stderr)
        return 0

    total = 0
    sites = []
    for path in sorted(ROOT.rglob("*.js")):
        if any(s in path.parts for s in SKIP):
            continue
        text = path.read_text(errors="ignore")
        for m in EMPTY.finditer(text):
            commented = COMMENTED.match(text, m.start())
            if commented:
                continue
            line = text.count("\n", 0, m.start()) + 1
            sites.append((str(path), line))
            total += 1

    baseline = 0
    if BASELINE.exists():
        for line in BASELINE.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                # The file reads "swallows: 337" so a human can see what it is.
                baseline = int(line.split(":", 1)[1].strip())

    if listing:
        for path, line in sites:
            print(f"{path}:{line}")
        print(f"-- {total} un-commented swallow(s)")

    if update:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        lines_out = [
            "# Un-commented empty catch blocks in src/js. A ratchet, not a target:",
            "# lower this whenever a swallow is fixed or explained.",
            "swallows: " + str(total),
            "",
        ]
        BASELINE.write_text(NL.join(lines_out))
        print(f"js-swallow: baseline written ({total})")
        return 0

    if total > baseline:
        print(f"js-swallow: FAIL - {total} un-commented swallows, baseline is {baseline}")
        for path, line in sites:
            print(f"  {path}:{line}")
        print("Explain the swallow with a comment or handle the error; then rerun.")
        return 1

    note = f" ({baseline - total} below baseline - lower it with --update)" if total < baseline else ""
    print(f"js-swallow: OK - {total} un-commented swallows, baseline {baseline}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
