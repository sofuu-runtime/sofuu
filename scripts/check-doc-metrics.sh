#!/usr/bin/env bash
# check-doc-metrics.sh — fail when a live doc asserts a retired measurement.
#
# The project's measured numbers live in exactly two places:
#   size  → the release archives themselves: macOS arm64 5.30 MiB (tree) /
#           5.39 MiB (published archive), Linux arm64 5.88 MiB, Linux x86_64
#           6.80 MiB, against the hard cap in RULES.md §3.1
#           (5.5 MiB = 5,767,168 bytes).
#   speed → BENCHMARKS.md (22.5 ms median wall clock incl. process spawn,
#           `sofuu eval "1+1"`, Apple M2 Pro, 2026-10-31).
#
# Every figure below was true once and then went false as the engine grew. Two of them
# reached the public marketing site before anyone noticed, because nothing in CI read
# prose. This check exists so the next one fails here instead of there.
#
# Exit 0 = clean, 1 = a live doc carries a retired number.
#
# Scope: tracked markdown (what CI sees) plus docs/ when present locally. docs/ is
# gitignored (see .gitignore), so CI never sees it but a local run does.
#
# Waiver: any line containing the marker `doc-metrics:allow` is skipped. Keep the reason
# in an HTML comment so it stays invisible when rendered:
#   <!-- doc-metrics:allow dated snapshot, superseded above -->
set -u

# Dated records, archives, and the sources of truth themselves are exempt by design.
# Matched with case globs (not regex: '^' is a literal character in a case pattern).
skip() {
    case "$1" in
        BENCHMARKS.md|CHANGELOG.md) return 0 ;;
        docs/archive/*)             return 0 ;;
        *AUDIT*|*SESSION*)          return 0 ;;
        docs/*AUDIT*|docs/*SESSION*|docs/*PLAN*) return 0 ;;
    esac
    return 1
}

# <regex><TAB><what to write instead>
PATTERNS=$(cat <<'EOF'
sub-?4 ?ms	The sub-4 ms cold-start claim is retired — the measured median is 22.5 ms wall clock (BENCHMARKS.md).
sub4ms	The sub-4 ms cold-start claim is retired — the measured median is 22.5 ms wall clock (BENCHMARKS.md).
< ?4 ?ms	The sub-4 ms cold-start claim is retired — the measured median is 22.5 ms wall clock (BENCHMARKS.md).
4\\text\{ ?ms\}	Retired cold-start figure written in LaTeX so the gate could not see it — rewrite in plain text (BENCHMARKS.md: 22.5 ms median).
5[.]2 MB	Retired binary size. macOS arm64 (this tree) is 5.30 MiB (5,559,904 bytes).
5[.]2\\text\{ ?MB\}	Retired binary size written in LaTeX — rewrite in plain text. macOS arm64 (this tree) is 5.30 MiB.
5[.]0 MB cap	Retired size cap. RULES.md §3.1 sets 5.5 MiB = 5,767,168 bytes.
5 MB cap	Retired size cap. RULES.md §3.1 sets 5.5 MiB = 5,767,168 bytes.
5MB cap	Retired size cap. RULES.md §3.1 sets 5.5 MiB = 5,767,168 bytes.
4[.][45] MB of	Retired binary size. macOS arm64 (this tree) is 5.30 MiB (5,559,904 bytes).
~?3[.]8 ?ms startup	Retired cold-start figure. The measured median is 22.5 ms wall clock (BENCHMARKS.md).
~4 ?ms	The ~4 ms cold-start claim is retired — the measured median is 22.5 ms wall clock (BENCHMARKS.md).
EOF
)

FILES=$(git ls-files '*.md' 2>/dev/null || true)
if [ -d docs ]; then
    FILES="$FILES
$(find docs -name '*.md' -type f 2>/dev/null)"
fi
FILES=$(printf '%s\n' "$FILES" | sed '/^[[:space:]]*$/d' | sort -u)

checked=0
hits=0
while IFS= read -r f; do
    [ -n "$f" ] || continue
    [ -f "$f" ] || continue
    skip "$f" && continue
    checked=$((checked + 1))
    while IFS="$(printf '\t')" read -r re fix; do
        [ -n "$re" ] || continue
        found=$(grep -nE "$re" "$f" 2>/dev/null | grep -v 'doc-metrics:allow' || true)
        [ -n "$found" ] || continue
        if [ "$hits" -eq 0 ]; then
            printf '✗ Retired numbers found in live documentation:\n\n'
        fi
        while IFS= read -r line; do
            printf '  %s:%s\n      → %s\n' "$f" "$(printf '%s' "$line" | cut -d: -f1)" "$fix"
            hits=$((hits + 1))
        done <<< "$found"
    done <<< "$PATTERNS"
done <<< "$FILES"

if [ "$hits" -gt 0 ]; then
    printf '\n  %d occurrence(s) in %d file(s) checked.\n' "$hits" "$checked"
    printf '  Fix the number, or add <!-- doc-metrics:allow reason --> on a dated record.\n'
    printf '  Sources of truth: RULES.md §3.1 (size cap), BENCHMARKS.md (speed).\n'
    exit 1
fi

printf '✓ doc metrics clean — %d live markdown file(s) checked, 0 retired numbers\n' "$checked"
