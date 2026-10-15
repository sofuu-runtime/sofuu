#!/usr/bin/env python3
"""Benchmark one `sofuu serve --infer` instance: prefill + decode rates.

Usage: bench_serve.py <port> [prefill_n=200] [decode_n=16]
Prints one JSON line: {ttft_s, prefill_tokens, prefill_tps, decode_tps, text_head}.
The prompt is deterministic (~prefill_n tokens) — no external files, stable
across runs, safe to commit.
"""
import json
import sys
import time
import urllib.request

HOST = "http://127.0.0.1"
with open(__file__.replace("bench_serve.py", "bench_prompt.txt")) as f:
    PROMPT = f.read().strip()


def serve_token():
    with open("/Users/priyanshuboruah/.sofuu/serve_token") as f:
        return f.read().strip()


def stream_post(port, payload, timeout=3600):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{HOST}:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {serve_token()}"})
    t0 = time.perf_counter()
    ttft = None
    n_tok = 0
    text = []
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except Exception as e:
        detail = ""
        try:
            detail = e.read().decode()[:200]
        except Exception:
            pass
        return {"error": f"{e} {detail}"}
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            obj = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        piece = obj.get("choices", [{}])[0].get("text", "")
        if ttft is None:
            ttft = time.perf_counter() - t0
        n_tok += 1
        text.append(piece)
    total = time.perf_counter() - t0
    return {"ttft_s": ttft, "n_tokens": n_tok, "total_s": total,
            "text_head": "".join(text)[:80]}


def main():
    port = int(sys.argv[1])
    decode_n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    # Warmup: tiny request, 1 decode step (excluded from measurement).
    stream_post(port, {"prompt": "Hi", "max_tokens": 1, "temperature": 0})
    prefill_n = len(PROMPT.split())
    r = stream_post(port, {"prompt": PROMPT, "max_tokens": decode_n,
                           "temperature": 0, "stream": True})
    if "error" in r or not r.get("ttft_s"):
        print(json.dumps({"error": r.get("error", "no tokens")}))
        return
    # The prompt is committed prose (scripts/bench_prompt.txt); word count is
    # the stable, engine-independent unit. ~1.31 tokens/word for this mix —
    # token estimates scale from it, clearly labeled as such.
    est_tokens = round(prefill_n * 1.31)
    decode_tps = (r["n_tokens"] - 1) / (r["total_s"] - r["ttft_s"]) \
        if r["n_tokens"] > 1 and r["total_s"] > r["ttft_s"] else None
    out = {
        "ttft_s": round(r["ttft_s"], 3),
        "prompt_words": prefill_n,
        "prompt_tokens_est": est_tokens,
        "prefill_tps_est": round(est_tokens / r["ttft_s"], 2),
        "decode_tps": round(decode_tps, 3) if decode_tps else None,
        "text_head": r.get("text_head", ""),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
