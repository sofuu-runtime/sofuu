#!/usr/bin/env bash
# scripts/kaggle/build_gpu_from_clone.sh — build the CUDA-capable CLI on a
# Kaggle (or any Ubuntu/NVIDIA) box from a plain `git clone`, with no binary
# upload and no CUDA toolkit.
#
# Why a build is needed: dist/ (the Linux binaries, QTSQ, libuv, libcurl
# archives) and tools/ (zig) are gitignored, so a clone cannot link the CLI.
# This script builds natively with the host gcc and the codec-less QTSQ shape
# that sofuu-ffi/build.rs explicitly supports (SOFUU_QTSQ_DIR basename
# "no-qtsq"). Inference does not need QTSQ; brain/session persistence is what
# this test build omits. `sofuu doctor` will report the codec absent — expected.
#
# The static-musl release is NOT usable for CUDA (musl's dlopen is a stub).
# Building natively gives target_env=gnu, where the CUDA decoder is compiled.
#
# Usage (from the clone root, on Kaggle):
#   bash scripts/kaggle/build_gpu_from_clone.sh
#   SOFUU_BIN="$PWD/target/release/sofuu" bash scripts/cuda/kaggle_infer_check.sh "$MODEL"
#
# Requirements: Linux x86_64/aarch64, git, cmake, gcc, Internet for rustup and
# (only if the linker cannot see libcurl.so) libcurl4-openssl-dev.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

say() { printf '\033[36m[kaggle]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[kaggle] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || die "this script builds on Linux; run it in the Kaggle notebook, not on macOS"
case "$(uname -m)" in
    x86_64 | aarch64) ;;
    *) die "unsupported arch $(uname -m) (x86_64 or aarch64 expected)" ;;
esac

command -v git >/dev/null || die "git is missing"
command -v cmake >/dev/null || die "cmake is missing (Kaggle images normally have it)"
command -v cc >/dev/null || die "cc/gcc is missing"

if ! grep -q avx2 /proc/cpuinfo 2>/dev/null && [ "$(uname -m)" = "x86_64" ]; then
    say "WARNING: this CPU does not advertise AVX2; the x86_64 engine may SIGILL"
fi

# ── Rust ────────────────────────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1 || ! command -v rustc >/dev/null 2>&1; then
    say "Rust not found — installing a minimal stable toolchain via rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable
    # shellcheck disable=SC1090
    . "$HOME/.cargo/env"
fi
say "rustc: $(rustc --version)"
say "cargo: $(cargo --version)"

# ── libuv (tracked source; build.rs looks in deps/libuv/build) ──────────
UV_BUILD="$REPO/deps/libuv/build"
if [ ! -f "$UV_BUILD/libuv.a" ]; then
    [ -f "$REPO/deps/libuv/CMakeLists.txt" ] || die "deps/libuv is missing from the clone"
    say "building libuv (static, Release) → deps/libuv/build/libuv.a"
    cmake -S "$REPO/deps/libuv" -B "$UV_BUILD" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF >/dev/null
    cmake --build "$UV_BUILD" --target uv_a -j"$(nproc 2>/dev/null || echo 4)" >/dev/null
fi
[ -f "$UV_BUILD/libuv.a" ] || die "libuv built but $UV_BUILD/libuv.a is absent"

# ── libcurl: build.rs emits -lcurl on Linux when no static dir is set ──
if ! (ldconfig -p 2>/dev/null | grep -Eq 'libcurl\.so( |$)'); then
    if command -v apt-get >/dev/null 2>&1; then
        SUDO=""
        [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
        say "libcurl.so is not on the linker path — installing libcurl4-openssl-dev"
        DEBIAN_FRONTEND=noninteractive $SUDO apt-get update -qq
        DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq libcurl4-openssl-dev
    fi
fi
ldconfig -p 2>/dev/null | grep -Eq 'libcurl\.so( |$)' \
    || die "libcurl.so still missing; install libcurl4-openssl-dev or set SOFUU_CURL_STATIC_DIR"

# ── Codec-less marker (build.rs supports exactly this basename) ────────
NOQTSQ="${TMPDIR:-/tmp}/no-qtsq"
mkdir -p "$NOQTSQ"
# Drop any cross-build overrides inherited from elsewhere; this is a NATIVE
# build, so the host cc/ar and system curl are what we want.
unset SOFUU_UV_DIR SOFUU_CURL_STATIC_DIR SOFUU_QTSQ_LIB SOFUU_ZLIB_DIR
export SOFUU_QTSQ_DIR="$NOQTSQ"
export SOFUU_BUILD="$(git rev-parse --short HEAD)"
say "building commit $SOFUU_BUILD with the codec-less QTSQ shape"
say "inference is unaffected; brain/session persistence is absent from this test build"

# ── Build ──────────────────────────────────────────────────────────────
# No CUDA toolkit and no libcudart: the PTX is already in src/cuda and the
# driver is loaded at runtime with dlopen (works because this is glibc).
cargo build --release -p sofuu-core

BIN="$REPO/target/release/sofuu"
[ -x "$BIN" ] || die "cargo finished but $BIN is missing"
[ "$(wc -c < "$BIN" | tr -d ' ')" -gt 1000000 ] || die "$BIN looks truncated"

# A static binary cannot load libcuda; catch the wrong build before a GPU run.
if ! grep -aq 'ld-linux-x86-64.so.2' "$BIN" && [ "$(uname -m)" = "x86_64" ]; then
    die "$BIN is static — it cannot dlopen libcuda. The native build should be dynamic; check CC/linker overrides."
fi
strings -a "$BIN" | grep -q 'GPU decoder armed' \
    || die "$BIN has no CUDA decoder (was target_env not gnu?)"

say "built: $BIN ($(wc -c < "$BIN" | tr -d ' ') B)"
"$BIN" --version 2>&1 | head -2 || die "$BIN cannot run; the codec-less build failed at startup"
printf '\n'
say "next:"
printf '  MODEL=$(find /kaggle/input -name "*.gguf" | head -1)\n'
printf '  SOFUU_BIN="%s" bash scripts/cuda/kaggle_infer_check.sh "$MODEL" "The capital of France is"\n' "$BIN"
printf '  # manual: SOFUU_CUDA=1 %s infer "$MODEL" -p "The capital of France is" -n 32 -t 4\n\n' "$BIN"
say "kernel-only gate (no build needed): bash scripts/cuda/kaggle_gpu_check.sh"
say "doctor will report QTSQ missing in this codec-less build — expected, not a GPU failure"
