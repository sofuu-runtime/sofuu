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
      [--tier exact|tolerant|best-effort] [--backend cpu|gpu]
      [--prompt-id ID] [--receipt PATH|-] [--require-identical]
  python3 scripts/text_gate.py --self-check

The per-step receipts (pos, sampled tok, top-8) are left in /tmp under the
gate prefix; BOUND exists per model because the composite class grows with
depth (0.6B Paris/32 measured 3.1e-2 by pos 8 on the exact pair).

Receipts (SPEC-RUNTIME-CONTRACT §1.6, R3.4): every gate run can emit a
machine-readable receipt carrying tier, backend, model sha256, knob values,
prompt id, verdict and numbers. A receipt missing any field is invalid;
`--self-check` validates the schema without a model (CI runs it on every
push). `--require-identical` fails (exit 3) unless the verdict is IDENTICAL —
the exact-tier CI gate uses it.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

SOFT = os.path.join(os.path.dirname(__file__), "..", "target", "release", "sofuu")
TOK = re.compile(r"tok=(\d+)")
V1 = re.compile(r"\(([-\d.e]+),")
V2 = re.compile(r"\)\s*\(([-\d.e]+),")

RECEIPT_FIELDS = (
    "kind", "schema_version", "model", "backend", "tier", "knobs",
    "prompt_id", "verdict", "numbers", "ts",
)
VERDICTS = ("IDENTICAL", "NEAR-TIE-CERTIFIED", "FORK", "ERROR")
TIERS = ("exact", "tolerant", "best-effort")


def sha256_file(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_env(s):
    return dict(kv.split("=", 1) for kv in s.split(",") if kv)


def build_receipt(args, verdict, numbers, steps_total):
    knobs = {}
    knobs.update(parse_env(args.cpu_env))
    knobs.update(parse_env(args.gpu_env))
    prompt_id = args.prompt_id or (
        "prompt-" + hashlib.sha256(args.prompt.encode()).hexdigest()[:12]
    )
    return {
        "kind": "text_gate",
        "schema_version": 1,
        "model": {"path": args.model, "sha256": sha256_file(args.model)},
        "backend": args.backend,
        "tier": args.tier,
        "knobs": knobs,
        "prompt_id": prompt_id,
        "prompt": args.prompt,
        "verdict": verdict,
        "numbers": numbers,
        "max_tokens": args.max_tokens,
        "bound": args.bound,
        "steps_total": steps_total,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def check_receipt(r):
    missing = [f for f in RECEIPT_FIELDS if f not in r]
    if missing:
        raise ValueError(f"receipt missing fields: {missing}")
    if r["verdict"] not in VERDICTS:
        raise ValueError(f"bad verdict {r['verdict']!r}")
    if r["tier"] not in TIERS:
        raise ValueError(f"bad tier {r['tier']!r}")
    if not isinstance(r["knobs"], dict):
        raise ValueError("knobs must be an object")
    if not isinstance(r["numbers"], dict):
        raise ValueError("numbers must be an object")
    return True


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
    ap.add_argument("model", nargs="?", default="")
    ap.add_argument("prompt", nargs="?", default="")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--bound", type=str, default="5e-2")  # float, or "auto"
    ap.add_argument("--gpu-env", default="SOFUU_GPU_DECODE=1,SOFUU_GPU_PREFILL_GEMM=1")
    ap.add_argument("--cpu-env", default="")
    ap.add_argument("--label", default="gate")
    ap.add_argument("--tier", default="tolerant", choices=TIERS,
                    help="the declared determinism tier of the path under test")
    ap.add_argument("--backend", default="gpu", help="label for the tested backend")
    ap.add_argument("--prompt-id", default="", help="stable prompt id for the receipt")
    ap.add_argument("--receipt", default="",
                    help="write the §1.6 receipt JSON to PATH, or '-' for stdout")
    ap.add_argument("--require-identical", action="store_true",
                    help="exit 3 unless the verdict is IDENTICAL (exact-tier gate)")
    ap.add_argument("--self-check", action="store_true",
                    help="validate the receipt schema without running a model")
    a = ap.parse_args()

    if a.self_check:
        sample = build_receipt(a, "IDENTICAL", {"dtop1": 0.0, "bound": 5e-2}, 16)
        check_receipt(sample)
        print("receipt schema OK "
              f"({len(RECEIPT_FIELDS)} required fields; verdicts {VERDICTS}; tiers {TIERS})")
        return 0

    if not a.model or not a.prompt:
        print("error: MODEL and PROMPT are required (or use --self-check)", file=sys.stderr)
        return 2

    for f in (f"/tmp/tg_{a.label}_cpu", f"/tmp/tg_{a.label}_gpu"):
        if os.path.exists(f):
            os.remove(f)

    if a.bound.lower() == "auto":
        a.bound = 0.0          # sentinel: resolved from matched steps
    else:
        a.bound = float(a.bound)
    gpu_env = parse_env(a.gpu_env)
    cpu_env = parse_env(a.cpu_env)

    run(cpu_env, f"/tmp/tg_{a.label}_cpu", a.model, a.prompt, a.max_tokens)
    run(gpu_env, f"/tmp/tg_{a.label}_gpu", a.model, a.prompt, a.max_tokens)

    def emit(verdict, numbers, steps_total):
        receipt = build_receipt(a, verdict, numbers, steps_total)
        check_receipt(receipt)
        if a.receipt == "-":
            print(json.dumps(receipt, indent=2))
        elif a.receipt:
            with open(a.receipt, "w") as f:
                json.dump(receipt, f, indent=2)
        if a.require_identical and verdict != "IDENTICAL":
            print(f"[{a.label}] exact-tier gate FAILED: verdict={verdict} "
                  f"(required IDENTICAL, tier={a.tier}, backend={a.backend})")
            return 3
        return 0

    c, g = probe_rows(f"/tmp/tg_{a.label}_cpu"), probe_rows(f"/tmp/tg_{a.label}_gpu")
    n = min(len(c), len(g))
    if n == 0:
        print(f"[{a.label}] VERDICT=ERROR no probe rows")
        return emit("ERROR", {"reason": "no probe rows"}, 0) or 1
    diverge = []
    for i in range(n):
        if c[i][0] != g[i][0]:
            diverge.append(i)
    if not diverge:
        print(f"[{a.label}] VERDICT=IDENTICAL ({n} steps token-equal, both.sequence.len={len(c)}/{len(g)})")
        rc = emit("IDENTICAL", {"steps": n}, n)
        return 0 if rc == 0 else rc
    # Only the FIRST diverging step is the fork CAUSE — after it the two
    # routes continue greedy in different contexts, so every later step is
    # context-divergent by construction and must not be judged.
    i = diverge[0]
    d = abs(c[i][1] - g[i][1])
    if a.bound <= 0:
        # --bound auto: self-referential envelope — the fork-step |dtop1| is
        # certified when it does not exceed the WORST top-1 delta observed at
        # steps the two routes agreed on argmax (the same generation's own
        # measured noise band; no magic constant).
        matched_d = [abs(c[k][1] - g[k][1]) for k in range(n) if k not in set(diverge)]
        if matched_d:
            a.bound = max(matched_d)
            print(f"[{a.label}] auto-bound from matched steps = {a.bound:.4f}")
    if d < a.bound:
        print(f"[{a.label}] VERDICT=NEAR-TIE-CERTIFIED first-fork-step={i} "
              f"cpu_tok={c[i][0]} gpu_tok={g[i][0]} |dtop1|={d:.4f} (bound {a.bound}) "
              f"steps_total={n} diverged_after={len(diverge)}")
        rc = emit("NEAR-TIE-CERTIFIED",
                  {"dtop1": d, "bound": a.bound, "first_fork_step": i}, n)
        return 0 if rc == 0 else rc
    print(f"[{a.label}] VERDICT=FORK first-fork-step={i} cpu_tok={c[i][0]} "
          f"gpu_tok={g[i][0]} |dtop1|={d:.4f} (bound {a.bound}) — bug class, "
          f"bisect with SOFUU_QKV_SNAP / scripts/kv_route_diff.py")
    rc = emit("FORK", {"dtop1": d, "bound": a.bound, "first_fork_step": i}, n)
    return 2 if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
