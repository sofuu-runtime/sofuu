#!/usr/bin/env python3
"""P5 (R5 / plan exit): reference apps and benchmark apps must use only the
PUBLIC runtime API — the `__qtsq_*` / `__sofuu_*` / `__session_*` /
`__chat_*` C seam is runtime-internal and may not appear in app code."""
import glob, re, sys

PATTERNS = re.compile(r"__qtsq_[a-z_]+|__sofuu_[a-z_]+|__session_[a-z_]+|__chat_[a-z_]+|__ai_[a-z_]+")
# examples/headless/* are the C-ABI lane's OWN boundary fixtures — they exist
# to test the internal seam and are runtime-owned, not user-facing app code.
# The audit governs app references (React frontend, benchmark apps).
ROOTS = ["examples/", "benchmark/"]
SKIP_PREFIX = ("examples/headless/",)
violations = []
for root in ROOTS:
    for path in glob.glob(root + "**/*.*", recursive=True):
        if path.endswith((".md", ".json", ".yml", ".yaml", ".txt", ".lock", ".metal", ".h", ".rs", ".kt", ".swift")):
            continue
        if any(path.startswith(pref) for pref in SKIP_PREFIX):
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if PATTERNS.search(line):
                violations.append(f"{path}:{i}: {line.strip()[:90]}")

if violations:
    print("PUBLIC-API violations (runtime-internal seam used by app code):")
    for v in violations[:20]:
        print("  " + v)
    sys.exit(1)
print("OK  public-api: examples/ and benchmark/ carry no runtime-internal seam usage")
