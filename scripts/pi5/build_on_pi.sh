#!/usr/bin/env bash
# scripts/pi5/build_on_pi.sh — shape B: the native, QTSQ-linked build on a Pi 5.
#
# The cross build (scripts/cross/build_linux_arm64.sh) skips QTSQ, so the
# portable static binary has no brain. The ambient device needs the brain, so
# it needs a native build on the device, from a QTSQ checkout.
#
# Why this script exists instead of a plain `make`: the upstream QTSQ Makefile
# compiles src/qtsq_metal.mm on every target and needs Apple's Objective-C++
# toolchain. CONTRIBUTING.md documents the Linux workaround — compile the C
# translation units directly and archive them. This script is that recipe with
# the guards the procedure needs (arch, toolchain, and a build directory that
# is not the checkout).
#
# Usage:
#   bash scripts/pi5/build_on_pi.sh --qtsq-dir ~/projects/qtsq
#   bash scripts/pi5/build_on_pi.sh --qtsq-dir ~/projects/qtsq --jobs 4
#
# It builds; it does not run the binary or benchmark it (see
# scripts/pi5/bench_baseline.py for the M-R0 receipt).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QTSQ_DIR="${SOFUU_QTSQ_DIR:-}"
JOBS="$(nproc 2>/dev/null || echo 4)"
OUT_DIR="$HOME/sofuu-build"

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 2; }
say() { printf '  %s\n' "$*"; }

usage() {
    cat <<EOF
Usage: bash $0 --qtsq-dir <path> [options]

  --qtsq-dir <path>   a QTSQ checkout (contains src/*.c); SOFUU_QTSQ_DIR is
                      honoured when the flag is omitted
  --jobs <n>          parallel compile jobs (default: $(nproc 2>/dev/null || echo 4))
  --out <dir>         where the built binary is copied (default: $OUT_DIR)
  -h, --help          this page
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --qtsq-dir) QTSQ_DIR=${2:?--qtsq-dir needs a path}; shift 2 ;;
        --jobs) JOBS=${2:?--jobs needs a number}; shift 2 ;;
        --out) OUT_DIR=${2:?--out needs a path}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[ -n "$QTSQ_DIR" ] || die "--qtsq-dir is required (or set SOFUU_QTSQ_DIR)"
[ -d "$QTSQ_DIR" ] || die "$QTSQ_DIR is not a directory"
ls "$QTSQ_DIR"/src/*.c >/dev/null 2>&1 || die "$QTSQ_DIR/src has no .c translation units"

ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    printf '\033[33mwarning:\033[0m this is %s, not a Pi 5 (aarch64) — build it on the device\n' "$ARCH" >&2
fi

for tool in cc ar make cargo rustc; do
    command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool (apt install build-essential; rustup for cargo)"
done

mkdir -p "$OUT_DIR/qtsq-obj"
OBJ="$OUT_DIR/qtsq-obj"

echo ""
echo "Native QTSQ build — $ARCH"
echo "─────────────────────────"
say "checkout: $QTSQ_DIR"
say "objects:  $OBJ"

# ── 1. QTSQ C translation units (CONTRIBUTING.md Linux workaround) ──
# A stale archive silently links the previous checkout's objects, so it is
# rebuilt from scratch every run. The object dir is outside the checkout: a
# build must not modify what a `git status` there would report.
rm -f "$OBJ"/*.o
count=0
for src in "$QTSQ_DIR"/src/*.c; do
    cc -O2 -fPIC -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE \
       -I "$QTSQ_DIR/src" -I "$QTSQ_DIR" \
       -c "$src" -o "$OBJ/$(basename "${src%.c}").o"
    count=$((count + 1))
done
say "compiled: $count translation unit(s)"

rm -f "$QTSQ_DIR/libqtsq.a"
ar rcs "$QTSQ_DIR/libqtsq.a" "$OBJ"/*.o
say "archive:  $QTSQ_DIR/libqtsq.a ($(stat -c %s "$QTSQ_DIR/libqtsq.a") B)"

# ── 2. the runtime ────────────────────────────────────────────────
cd "$REPO_ROOT"
say "cargo:    building (this is the slow part)"
SOFUU_QTSQ_DIR="$QTSQ_DIR" make -j"$JOBS"

# ── 3. the size verdict (recorded, not hidden) ───────────────────
SIZE="$(stat -c %s ./sofuu)"
CAP=5767168
say "binary:   ./sofuu = $SIZE B (cap $CAP)"
if [ "$SIZE" -le "$CAP" ]; then
    say "cap:      passes ($((CAP - SIZE)) B headroom)"
else
    printf '\033[33m  cap:      OVER by %s B — a tracked release gap, not an exemption (RULES §3.1)\033[0m\n' "$((SIZE - CAP))" >&2
fi

install -m 0755 ./sofuu "$OUT_DIR/sofuu-linux-arm64-qtsq"
say "copied:   $OUT_DIR/sofuu-linux-arm64-qtsq"

cat <<EOF

Next (run these yourself — this script stops at a built binary):
  ./sofuu version                     # revision stamp must be this commit
  sudo deploy/pi5/install.sh --binary $OUT_DIR/sofuu-linux-arm64-qtsq
  python3 scripts/pi5/bench_baseline.py --out ~/pi5-baseline.json
EOF
