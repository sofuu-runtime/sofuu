#!/usr/bin/env python3
"""Generate a tiny, loadable llama-arch GGUF fixture for the determinism gate.

The fixture is a real 1-layer transformer the engine loads and runs — small
enough to commit (~55 KB), deterministic, and NaN-free by construction:

  * every q4_K block decodes to zero (scales bytes 0x00, d = dmin = f16(0.001)),
    so attention/ffn weights are exactly zero and logits stay small and finite;
  * token_embd / output.weight are Q8_0 with d = f16(0.01) and a small fixed
    per-index pattern, so distinct tokens produce distinct (tiny) logits;
  * the vocab is the 256 byte tokens "<0x00>".."<0xFF>", so any prompt encodes
    deterministically through the tokenizer's byte fallback — no merges needed.

Byte layout matches the engine exactly (crates/sofuu-core/src/infer/):
  * gguf.rs: v3 header, u64 string lengths, tensor infos, 32-byte data alignment
  * quant.rs type_row_size: F32 n*4, Q8_0 (n/32)*34, Q4_K (n/256)*144
  * model.rs synth_gguf test: the same KV set and tensor shapes

Usage:
  python3 scripts/contract/make_fixture_gguf.py [OUT]   # default:
      contract/fixtures/sofuu-fixture.gguf
"""
import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32

# gguf value types (ggml): 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32,
# 7 bool, 8 string, 9 array, 10 u64, 11 i64, 12 f64
T_U32, T_STR, T_ARR = 4, 8, 9
# ggml tensor types
F32, Q8_0, Q4_K = 0, 8, 12
QK8_0, QK_K = 32, 256

N_EMBD, N_LAYER, N_HEAD, N_HEAD_KV, N_FF, N_VOCAB, N_CTX = 64, 1, 2, 2, 96, 256, 2048


def put_u32(b, v):
    b += struct.pack("<I", v)


def put_u64(b, v):
    b += struct.pack("<Q", v)


def put_str(b, s):
    raw = s.encode()
    put_u64(b, len(raw))
    b += raw


KV_COUNT = 0  # incremented by the put_kv_* helpers — never hand-counted


def put_kv_u32(b, key, v):
    global KV_COUNT
    put_str(b, key)
    put_u32(b, T_U32)
    put_u32(b, v)
    KV_COUNT += 1


def put_kv_str(b, key, v):
    global KV_COUNT
    put_str(b, key)
    put_u32(b, T_STR)
    put_str(b, v)
    KV_COUNT += 1


def put_kv_str_array(b, key, items):
    global KV_COUNT
    put_str(b, key)
    put_u32(b, T_ARR)
    put_u32(b, T_STR)
    put_u64(b, len(items))
    for s in items:
        put_str(b, s)
    KV_COUNT += 1


def f16_bytes(v):
    return struct.pack("<e", v)


def row_size(ty, n):
    if ty == F32:
        return n * 4
    if ty == Q8_0:
        return (n // QK8_0) * 34
    if ty == Q4_K:
        return (n // QK_K) * 144
    raise ValueError(ty)


def tensor_data(ty, n, seed):
    """Valid, bounded, deterministic block bytes for n values."""
    out = bytearray()
    if ty == F32:
        for i in range(n):
            out += struct.pack("<f", 1.0)  # RMS norm weights
        return bytes(out)
    if ty == Q8_0:
        d = f16_bytes(0.01)
        nblocks = n // QK8_0
        for blk in range(nblocks):
            out += d
            for i in range(QK8_0):
                out += struct.pack("<b", ((blk * QK8_0 + i + seed) % 9) - 4)
        return bytes(out)
    if ty == Q4_K:
        d = f16_bytes(0.001)
        nblocks = n // QK_K
        for _ in range(nblocks):
            out += d          # d
            out += d          # dmin
            out += bytes(12)  # scales: all zero -> decoded weight exactly 0
            out += bytes(128)  # nibbles (unused while scales are zero)
        return bytes(out)
    raise ValueError(ty)


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("contract/fixtures/sofuu-fixture.gguf")

    # ── KVs ──────────────────────────────────────────────────────────────
    kv = bytearray()
    put_kv_str(kv, "general.architecture", "llama")
    put_kv_str(kv, "general.name", "sofuu-fixture")
    put_kv_u32(kv, "general.file_type", 7)  # mostly Q8_0
    put_kv_u32(kv, "general.alignment", ALIGN)
    put_kv_u32(kv, "llama.embedding_length", N_EMBD)
    put_kv_u32(kv, "llama.block_count", N_LAYER)
    put_kv_u32(kv, "llama.attention.head_count", N_HEAD)
    put_kv_u32(kv, "llama.attention.head_count_kv", N_HEAD_KV)
    put_kv_u32(kv, "llama.feed_forward_length", N_FF)
    put_kv_u32(kv, "llama.vocab_size", N_VOCAB)
    put_kv_u32(kv, "llama.context_length", N_CTX)
    put_kv_str(kv, "tokenizer.ggml.model", "llama")
    put_kv_str_array(kv, "tokenizer.ggml.tokens",
                     [f"<0x{i:02X}>" for i in range(N_VOCAB)])
    put_kv_u32(kv, "tokenizer.ggml.bos_token_id", 0)
    put_kv_u32(kv, "tokenizer.ggml.eos_token_id", 1)

    # ── tensor infos + data ──────────────────────────────────────────────
    # (name, dims, type, seed) — dims[0] is the contiguous row length.
    # All weights are Q8_0: the loader slices per row, and Q8_0 is valid at
    # any row length that is a multiple of 32, while q4_K needs multiples of
    # 256 (a 64-value row would read as a 0-byte slice and zero the output).
    tensors = [
        ("token_embd.weight", [N_EMBD, N_VOCAB], Q8_0, 1),
        ("blk.0.attn_norm.weight", [N_EMBD, 1], F32, 0),
        ("blk.0.attn_q.weight", [N_EMBD, N_EMBD], Q8_0, 2),
        ("blk.0.attn_k.weight", [N_EMBD, N_EMBD], Q8_0, 3),
        ("blk.0.attn_v.weight", [N_EMBD, N_EMBD], Q8_0, 4),
        ("blk.0.attn_output.weight", [N_EMBD, N_EMBD], Q8_0, 5),
        ("blk.0.ffn_norm.weight", [N_EMBD, 1], F32, 0),
        ("blk.0.ffn_gate.weight", [N_FF, N_EMBD], Q8_0, 6),
        ("blk.0.ffn_up.weight", [N_FF, N_EMBD], Q8_0, 7),
        ("blk.0.ffn_down.weight", [N_EMBD, N_FF], Q8_0, 8),
        ("output.weight", [N_EMBD, N_VOCAB], Q8_0, 9),
        ("output_norm.weight", [N_EMBD, 1], F32, 0),
    ]

    infos = bytearray()
    data = bytearray()
    off = 0
    for name, dims, ty, seed in tensors:
        n = dims[0] * dims[1]
        nbytes = row_size(ty, n)
        put_str(infos, name)
        put_u32(infos, len(dims))
        for d in dims:
            put_u64(infos, d)
        put_u32(infos, ty)
        put_u64(infos, off)
        data += tensor_data(ty, n, seed)
        off += nbytes

    # ── file ─────────────────────────────────────────────────────────────
    hdr = bytearray()
    put_u32(hdr, GGUF_MAGIC)
    put_u32(hdr, GGUF_VERSION)
    put_u64(hdr, len(tensors))
    put_u64(hdr, KV_COUNT)  # structural: incremented by the put_kv_* helpers

    blob = bytes(hdr) + bytes(kv) + bytes(infos)
    pad = (-len(blob)) % ALIGN
    blob += bytes(pad)
    blob += bytes(data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    print(f"wrote {out_path} ({len(blob)} bytes, {len(tensors)} tensors, "
          f"n_vocab={N_VOCAB}, n_embd={N_EMBD}, n_layer={N_LAYER})")


if __name__ == "__main__":
    main()
