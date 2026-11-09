#!/usr/bin/env bash
# scripts/cuda/make_gpu_cli_kit.sh — package the glibc CLI for a Kaggle T4.
#
# The release CLI is static musl, and musl's dlopen is a stub: a static musl
# binary cannot load libcuda.so.1, so the CUDA decoder is not even compiled
# into it. This kit carries the dynamically linked glibc build instead
# (dist/kaggle/sofuu-linux-x86_64-gnu), which uses the system loader for libcuda.
#
# Output: dist/kaggle/sofuu-gpu-kit.tar.gz
# Kaggle: upload as a Dataset, add it to a GPU notebook, then follow README.md.
#
# Usage: bash scripts/cuda/make_gpu_cli_kit.sh [BINARY] [OUTPUT.tar.gz]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${1:-$ROOT/dist/kaggle/sofuu-linux-x86_64-gnu}"
OUT="${2:-$ROOT/dist/kaggle/sofuu-gpu-kit.tar.gz}"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/sofuu-gpu-kit.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

if [ ! -f "$BIN" ]; then
    echo "missing $BIN" >&2
    echo "Build it with the glibc recipe:" >&2
    echo "  see scripts/cross/build_linux_capi_so.sh (same deps, -p sofuu-core)" >&2
    echo "  or: SOFUU_TARGET=gnu bash scripts/cuda/build_linux_gnu_cli.sh" >&2
    exit 1
fi

mkdir -p "$STAGE/scripts/cuda" "$STAGE/src"
cp "$BIN" "$STAGE/sofuu"
chmod +x "$STAGE/sofuu"
cp "$ROOT/scripts/cuda/kaggle_gpu_check.sh" "$STAGE/scripts/cuda/"
cp "$ROOT/scripts/cuda/kaggle_infer_check.sh" "$STAGE/scripts/cuda/"
cp "$ROOT/scripts/cuda/kaggle_longctx_check.sh" "$STAGE/scripts/cuda/"
# The kernel gate compiles cuda_loader.c + gpu_check.cpp against this tree.
# Ship the whole src/cuda directory so the kit is self-contained.
cp -R "$ROOT/src/cuda" "$STAGE/src/cuda"
find "$STAGE" -name '*.o' -delete
find "$STAGE" -name '.DS_Store' -delete

cat > "$STAGE/GPU-README.md" <<'EOF'
# Sofuu GPU kit — glibc x86_64 + CUDA (Tesla T4)

This is the **dynamically linked** Linux build. The normal release binary is
static musl; musl's `dlopen` is a stub, so it cannot load `libcuda.so.1` and its
CUDA decoder is not compiled in. This build uses the system loader and does.

## 1. Upload and unpack

Upload this tar.gz as a Kaggle Dataset, add it to the notebook, then:

```bash
mkdir -p /kaggle/working/sofuu
tar xzf /kaggle/input/<your-dataset>/sofuu-gpu-kit.tar.gz -C /kaggle/working/sofuu
chmod +x /kaggle/working/sofuu/sofuu
/kaggle/working/sofuu/sofuu --version
```

Accelerator must be **GPU T4 x1** (or any NVIDIA GPU) and Internet may stay off
once the model is attached.

## 2. Point at the 1B model you already have

```bash
MODEL=$(find /kaggle/input -name 'Llama-3.2-1B-Instruct-Q8_0.gguf' | head -1)
echo "$MODEL"
```

Any llama-architecture GGUF with F32 norms and Q8_0/Q4_K/Q6_K projections is
supported by the CUDA decoder. Other arches print a named refusal and keep the
CPU engine.

## 3. One-command gate (CPU vs GPU logits + rate)

```bash
cd /kaggle/working/sofuu
bash scripts/cuda/kaggle_infer_check.sh "$MODEL" "The capital of France is"
```

It runs the same prompt on the CPU (f32 KV) and the CUDA decoder, compares the
full last-token logit vectors (argmax must match), then times a 32-token GPU
decode. Exit 0 is the gate. On failure it prints where the receipts are.

## 4. Manual GPU run

```bash
SOFUU_CUDA=1 SOFUU_CUDA_DEBUG=1 \
  /kaggle/working/sofuu/sofuu infer "$MODEL" \
  "The capital of France is" -n 32 -t 4
```

What to look for on stderr:

```
[cuda] initialised ... on Tesla T4
[cuda] Tesla T4 cc7.5 · ... GiB VRAM · weights ... GiB · KV 4096 rows
[cuda] GPU decoder armed: 16 layers, 4096 KV rows
```

and a decode rate far above the 1.9 tok/s the CPU path measured. If you see
`GPU decoder NOT armed`, the printed reason is the whole story — paste it.

## 5. Prove the GPU logits against the CPU (manual)

The engine can dump the full last-token logit vector (`SFLOG01` container).
Run the same prompt twice:

```bash
SOFUU_CUDA=0 SOFUU_KV_F16=0 SOFUU_DUMP_LOGITS=/kaggle/working/cpu.sflog \
  /kaggle/working/sofuu/sofuu infer "$MODEL" "The capital of France is" -n 1 -t 4

SOFUU_CUDA=1 SOFUU_DUMP_LOGITS=/kaggle/working/gpu.sflog \
  /kaggle/working/sofuu/sofuu infer "$MODEL" "The capital of France is" -n 1 -t 4
```

Then:

```python
import struct, numpy as np
def load(p):
    b = open(p, 'rb').read()
    assert b[:8] == b'SFLOG01\0', b[:8]
    n = struct.unpack('<Q', b[8:16])[0]
    return np.frombuffer(b[16:16 + 4*n], dtype='<f4')
a, b = load('/kaggle/working/cpu.sflog'), load('/kaggle/working/gpu.sflog')
d = np.abs(a - b)
print('n', len(a), 'maxabs', float(d.max()), 'argmax', int(a.argmax()), int(b.argmax()))
print('top10 overlap', len(set(a.argsort()[-10:]) & set(b.argsort()[-10:])))
```

The argmax must match and the max absolute difference should be small. A large
difference means a kernel/layout bug, not "GPU noise" — send both numbers.

## 6. Kernel-only gate (already green on T4)

The `scripts/cuda/kaggle_gpu_check.sh` + `kernels.ptx` copies are included for a
fresh machine. They are independent of the CLI:

```bash
cd /kaggle/working/sofuu && bash scripts/cuda/kaggle_gpu_check.sh
```

## Notes

- `SOFUU_CUDA=1` is the opt-in; unset leaves the CPU engine in charge.
  `SOFUU_CUDA=0` or `--cpu` is the explicit kill switch.
- `SOFUU_CUDA_CTX=N` lowers the device KV window (default follows `--ctx`,
  which is 4096 when omitted). GPU KV compaction is not implemented in v1, so a
  conversation that reaches the window stops with a clear message.
- The decoder implements the standard Llama block graph. It refuses, by name,
  q/k biases, q/k norms, sliding-window attention, and non-llama arches; the CPU
  engine still runs those.
- First run includes ~1 s of PTX JIT and the weight upload; later tokens are the
  number that matters.
EOF

# LICENSE + NOTICE travel with every redistributed archive.
bash "$ROOT/scripts/pack/stage_legal.sh" "$STAGE" "$ROOT" >/dev/null 2>&1 || true
# stage_legal copies the repository README; the kit needs its own instructions.
mv -f "$STAGE/GPU-README.md" "$STAGE/README.md"
(
    cd "$STAGE"
    sha256sum sofuu > SHA256SUMS 2>/dev/null || shasum -a 256 sofuu > SHA256SUMS
)

mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" -C "$STAGE" .
bytes=$(wc -c < "$OUT" | tr -d ' ')
echo "[gpu-kit] wrote $OUT ($bytes B)"
echo "[gpu-kit] contents:"
tar -tzf "$OUT" | sed -n '1,30p'
