#!/usr/bin/env python3
"""Full-text gate with logits-tolerance certification (llama.cpp-parity plan).

Runs one model through {CPU route, GPU route} with SOFUU_DECODE_PROBE, then
compares the step-aligned sampled-token sequences:

  IDENTICAL          — every generated token matches.
  NEAR-TIE-CERTIFIED — every diverging step's two routes differ on the top-1
                       logit by < BOUND (the measured composite f32-noise
                       class), i.e. the fork is a certification-class near
                       tie, not an arithmetic defect.
  FORK               — a diverging step exceeds BOUND: bug class, do not
                       flip any defaults; bisect with SOFUU_QKV_SNAP /
                       kv_route_diff / scripts/kv_route_diff.py.

Usage:
  python3 scripts/text_gate.py MODEL "PROMPT" [--max-tokens N] [--bound E]
      [--gpu-env SOFUU_X=1,SOFUU_Y=1] [--cpu-env SOFUU_NO_TILES=1]

The per-step receipts (pos, sampled tok, top-8) are left in /tmp under the
gate prefix; BOUND exists per model because the composite class grows with
depth (0.6B Paris/32 measured 3.1e-2 by pos 8 on the exact pair).
"""
import argparse, os, re, subprocess, sys

SOFT = os.path.join(os.path.dirname(__file__), "..", "target", "release", "sofuu")
TOK = re.compile(r"tok=(\d+)")
V1 = re.compile(r"\(([-\d.e]+),")
V2 = re.compile(r"\)\s*\(([-\d.e]+),")


def probe_rows(prefix):
    rows = []
    with open(prefix) as f:
        for ln in f:
            t = TOK.search(ln)
            v1 = V1.search(ln)
            v2 = V2.search(ln)
            if not (t and v1 and v2):
                continue
            g1 = float(re.search(r"\(([-\d.e]+),(\d+)\)", ln).group(1))
            g2 = float(v2.group(1))
            rows.append((int(t.group(1)), g1, g1 - g2))
    return rows


def run(env, prefix, model, prompt, max_tokens):
    e = dict(os.environ)
    e["SOFUU_DECODE_PROBE"] = prefix
    e.update(env)
    subprocess.run([SOFT, "infer", model, prompt, "--max-tokens", str(max_tokens), "--temp", "0"],
                   env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return open(prefix).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompt")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--bound", type=float, default=5e-2)
    ap.add_argument("--gpu-env", default="SOFUU_GPU_DECODE=1,SOFUU_GPU_PREFILL_GEMM=1")
    ap.add_argument("--cpu-env", default="")
    ap.add_argument("--label", default="gate")
    a = ap.parse_args()

    for f in (f"/tmp/tg_{a.label}_cpu", f"/tmp/tg_{a.label}_gpu"):
        if os.path.exists(f):
            os.remove(f)

    gpu_env = dict(kv.split("=", 1) for kv in a.gpu_env.split(",") if kv)
    cpu_env = dict(kv.split("=", 1) for kv in a.cpu_env.split(",") if kv)

    run(cpu_env, f"/tmp/tg_{a.label}_cpu", a.model, a.prompt, a.max_tokens)
    run(gpu_env, f"/tmp/tg_{a.label}_gpu", a.model, a.prompt, a.max_tokens)

    c, g = probe_rows(f"/tmp/tg_{a.label}_cpu"), probe_rows(f"/tmp/tg_{a.label}_gpu")
    n = min(len(c), len(g))
    if n == 0:
        print(f"[{a.label}] VERDICT=ERROR no probe rows")
        return 1
    diverge = []
    for i in range(n):
        if c[i][0] != g[i][0]:
            diverge.append(i)
    if not diverge:
        print(f"[{a.label}] VERDICT=IDENTICAL ({n} steps token-equal, both.sequence.len={len(c)}/{len(g)})")
        return 0
    # Only the FIRST diverging step is the fork CAUSE — after it the two
    # routes continue greedy in different contexts, so every later step is
    # context-divergent by construction and must not be judged.
    i = diverge[0]
    d = abs(c[i][1] - g[i][1])
    if d < a.bound:
        print(f"[{a.label}] VERDICT=NEAR-TIE-CERTIFIED first-fork-step={i} "
              f"cpu_tok={c[i][0]} gpu_tok={g[i][0]} |dtop1|={d:.4f} (bound {a.bound}) "
              f"steps_total={n} diverged_after={len(diverge)}")
        return 0
    print(f"[{a.label}] VERDICT=FORK first-fork-step={i} cpu_tok={c[i][0]} "
          f"gpu_tok={g[i][0]} |dtop1|={d:.4f} (bound {a.bound}) — bug class, "
          f"bisect with SOFUU_QKV_SNAP / scripts/kv_route_diff.py")
    return 2


if __name__ == "__main__":
    sys.exit(main())
