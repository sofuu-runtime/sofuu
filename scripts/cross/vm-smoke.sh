#!/usr/bin/env bash
# scripts/cross/vm-smoke.sh — execute a cross-built Linux artifact on a real
# Linux kernel, on a machine that cannot run ELF binaries.
#
# The cross builds produce static musl binaries this host cannot execute: there
# is no user-mode emulation here, and the artifact must be proven to run before
# it is published. This boots it as PID 1 of a minimal Linux kernel under
# qemu-system, with an initramfs that contains nothing but the binary and two
# test programs — so a pass means it started, parsed and ran JavaScript, ran a
# Kaze program through the fuel meter, and wrote and read a file.
#
#   scripts/cross/vm-smoke.sh dist/sofuu-linux-x86_64 dist/sofuu-linux-arm64
#
# Requirements: qemu-system-x86_64 and/or qemu-system-aarch64, python3, and curl
# for the one-time kernel download. Kernels cache in ${SOFUU_VM_CACHE:-$HOME/.cache/sofuu-vm}.
#
# x86_64 note: the artifact is built with -mavx2/-mfma for its SIMD kernels and
# needs an AVX2 CPU. qemu's default qemu64 model has none and the binary dies on
# SIGILL; this boots -cpu max, which is what shipping hardware looks like. That
# requirement was undocumented until this script existed to reveal it.
set -uo pipefail

CACHE="${SOFUU_VM_CACHE:-$HOME/.cache/sofuu-vm}"
mkdir -p "$CACHE"
fail=0

die() { echo "error: $*" >&2; exit 2; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing $1"; }
need python3
need curl

cat > "$CACHE/mk_initramfs.py" <<'PY'
import gzip, pathlib, stat, sys

def h(n):
    return b"%08X" % n

def entry(name, data, mode, ino):
    out = bytearray(b"070701")
    for v in (ino, mode, 0, 0, 1, 0, len(data), 0, 0, 0, 0, len(name) + 1, 0):
        out += h(v)
    out += name.encode() + b"\x00"
    while len(out) % 4:
        out += b"\x00"
    out += data
    while len(out) % 4:
        out += b"\x00"
    return bytes(out)

out = pathlib.Path(sys.argv[1])
blob = b""
ino = 1
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    blob += entry(name, pathlib.Path(path).read_bytes(), stat.S_IFREG | 0o755, ino)
    ino += 1
trailer = bytearray(b"070701")
for _ in range(13):
    trailer += h(0)
trailer += b"TRAILER!!!\x00"
while len(trailer) % 512:
    trailer += b"\x00"
out.write_bytes(gzip.compress(blob + bytes(trailer), 9))
PY

cat > "$CACHE/kaze_probe.kaze" <<'KAZE'
needs: log

action main():
    let xs = [1, 2]
    let ys = [1, 2]
    let same = xs == ys
    log "equal"
KAZE

cat > "$CACHE/fs_probe.js" <<'JS'
const fs = require('fs');
fs.writeFileSync('/x.txt', 'hi');
console.log(fs.readFileSync('/x.txt', 'utf8'));
JS

kernel_spec() {
    case "$1" in
        x86_64)  echo "vmlinuz-lts|https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/x86_64/netboot/vmlinuz-lts" ;;
        aarch64) echo "vmlinuz-virt|https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/aarch64/netboot/vmlinuz-virt" ;;
        *) return 1 ;;
    esac
}

fetch_kernel() {
    local arch="$1" name="$2" url="$3"
    local out="$CACHE/kernel-$arch"
    if [ ! -s "$out" ]; then
        echo "→ fetching $name for $arch (one time, into $CACHE)" >&2
        curl -fsSL --max-time 300 "$url" -o "$out.part" || die "kernel download failed"
        mv "$out.part" "$out"
    fi
    echo "$out"
}

boot() {
    local arch="$1" kernel="$2" initrd="$3" cmd="$4"
    case "$arch" in
        x86_64)
            qemu-system-x86_64 -m 1024 -smp 2 -cpu max -kernel "$kernel" -initrd "$initrd" \
                -append "console=ttyS0 panic=1 quiet -- $cmd" -nographic -no-reboot ;;
        aarch64)
            qemu-system-aarch64 -machine virt -cpu cortex-a76 -m 1024 -smp 2 \
                -kernel "$kernel" -initrd "$initrd" \
                -append "console=ttyAMA0 panic=1 quiet -- $cmd" -nographic -no-reboot ;;
    esac 2>&1 | tr -d '\r' > "$CACHE/boot-$arch.log"
    # The guest drives the serial console as a terminal: reset (c), the
    # wrap flag ([?7l), clear-screen ([2J) and SGR colors all arrive in
    # front of the program's own output. Stripping only colors left the line
    # starting with ESC, so an exact-match check on "2" could never pass.
    # BSD sed has no \x escapes, so this cleanup is Python.
    python3 - "$CACHE/boot-$arch.log" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)
text = re.sub(r"\x1b.", "", text)  # ESC + one byte: RIS (\x1bc), keypad (\x1b=), etc.
text = text.replace("Booting from ROM..", "")
keep = []
for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith("["):
        continue
    if any(s in line for s in ("iPXE", "Press Ctrl-B", "SeaBIOS")):
        continue
    keep.append(line)
print("\n".join(keep[-8:]))
PY
}

smoke_one() {
    local bin="$1"
    [ -f "$bin" ] || die "no such file: $bin"
    local arch
    arch=$(python3 -c 'import sys
d = open(sys.argv[1], "rb").read(20)
print({0x3e: "x86_64", 0xb7: "aarch64"}.get(d[18], "unknown") if d[:4] == b"\x7fELF" else "unknown")' "$bin")
    [ "$arch" != unknown ] || die "$bin is not an ELF binary"
    local spec; spec=$(kernel_spec "$arch") || die "unsupported arch $arch"
    local kname kurl
    IFS='|' read -r kname kurl <<<"$spec"
    local kernel; kernel=$(fetch_kernel "$arch" "$kname" "$kurl")
    local initrd="$CACHE/initramfs-$arch.cpio.gz"
    python3 "$CACHE/mk_initramfs.py" "$initrd" \
        "init=$bin" "kaze_probe.kaze=$CACHE/kaze_probe.kaze" "fs_probe.js=$CACHE/fs_probe.js" >/dev/null

    echo ""
    echo "smoke: $bin ($arch, $(wc -c < "$bin") B)"
    local out

    out=$(boot "$arch" "$kernel" "$initrd" "--version")
    if printf '%s' "$out" | grep -q 'sofuu 0'; then
        echo "  OK  --version: $(printf '%s' "$out" | grep -m1 'sofuu 0')"
    else
        echo "  FAIL --version"; printf '%s\n' "$out" | sed 's/^/       /'; fail=1
    fi

    out=$(boot "$arch" "$kernel" "$initrd" "eval 1+1")
    if printf '%s' "$out" | grep -qx '2'; then
        echo "  OK  js eval"
    else
        echo "  FAIL js eval"; printf '%s\n' "$out" | sed 's/^/       /'; fail=1
    fi

    out=$(boot "$arch" "$kernel" "$initrd" "script run /kaze_probe.kaze")
    if printf '%s' "$out" | grep -q 'fuel'; then
        echo "  OK  kaze control plane ($(printf '%s' "$out" | grep -o '[0-9]* fuel' | head -1))"
    else
        echo "  FAIL kaze"; printf '%s\n' "$out" | sed 's/^/       /'; fail=1
    fi

    out=$(boot "$arch" "$kernel" "$initrd" "run /fs_probe.js")
    if printf '%s' "$out" | grep -qx 'hi'; then
        echo "  OK  file round-trip"
    else
        echo "  FAIL file round-trip"; printf '%s\n' "$out" | sed 's/^/       /'; fail=1
    fi
}

[ "$#" -gt 0 ] || die "usage: $0 <linux-artifact> [more artifacts]"
for bin in "$@"; do
    smoke_one "$bin"
done

echo ""
if [ "$fail" -eq 0 ]; then
    echo "RESULT: PASS — every artifact booted and ran"
    exit 0
fi
echo "RESULT: FAIL — do not publish these artifacts"
exit 1
