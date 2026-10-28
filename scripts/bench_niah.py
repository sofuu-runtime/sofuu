#!/usr/bin/env python3
"""bench_niah.py — Long-Context Conformance Battery.

Implements the acceptance gates of docs/PLAN-LONG-CONTEXT-PERSISTENCE.md §5
as machine receipts, driving the SHIPPED `sofuu infer` CLI (no engine
edits). Every run appends one JSON line to docs/bench/long-context.jsonl.

Gates
  1  Needle in a Haystack — a random 6-digit vault code is planted at
     10/25/50/75/90% depth of an 8k/16k/32k context; greedy decode must
     reproduce the code exactly. PASS = every depth exact-matches.
  2  Perplexity drift (KIVI vs FP16 ≤ 1.04) — BLOCKED: the engine exposes
     no per-token logprobs, so PPL cannot be measured externally. Recorded
     as blocked with that reason until a logprob surface exists.
  3  RAM — peak RSS via /usr/bin/time -l; the official gate is
     8B Q4_K_M @ 32k < 7.5 GiB, but every run records its RSS.
  4  Decode monotonicity — aggregate proxy: decode tk/s at -n 16000 must
     stay ≥ 90% of decode tk/s at -n 200 (the CLI reports aggregate rates;
     per-position bucketing needs a logprob/timing surface).

Corpus: local repo text (docs + shipped JS + README), tiled — prose AND
code, deterministic, no network. The plan names Paul Graham essays; swap
with --corpus if you have them.

Usage
  python3 scripts/bench_niah.py --model ~/m.gguf \\
      --lengths 8192 --depths 10,25,50,75,90 --kv f16,kivi4 --quick
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time as _time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOFUU = os.path.join(ROOT, "sofuu")

# The stats line prints rates, which stay parseable whatever Duration's
# debug format does across the 1m/1s boundary ("1m 12.5s").
STATS_RE = re.compile(
    r"prefill: (\d+) tk in .*?\(([\d.]+) tk/s\).*?decode: (\d+) tk in .*?\(([\d.]+) tk/s\)"
)
# /usr/bin/time -l prints the value BEFORE the label.
RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")


def load_corpus(extra=None):
    """Deterministic prose+code filler from the repo itself."""
    paths = []
    for sub in ("docs", "src/js"):
        d = os.path.join(ROOT, sub)
        if os.path.isdir(d):
            paths += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith((".md", ".js"))]
    paths = [p for p in paths if os.path.isfile(p) and os.path.getsize(p) < 400_000]
    if extra:
        paths = [extra] + paths
    chunks = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                t = fh.read()
            if len(t) > 200:
                chunks.append(t)
        except OSError:
            pass
    if not chunks:
        raise SystemExit("no corpus available (pass --corpus <file>)")
    return chunks


def build_haystack(chunks, target_chars, rng):
    """Tile the corpus to EXACTLY target_chars (the last chunk is cut short —
    appending whole documents overshot by up to one full file, which once
    turned an 8k-target prompt into 23k tokens and triggered the engine's
    mid-prompt KV compaction, erasing the needle)."""
    out = []
    n = 0
    i = 0
    while n < target_chars:
        c = chunks[i % len(chunks)]
        out.append(c)
        n += len(c)
        i += 1
    return "".join(out)[:target_chars]




def run_infer(sofuu, model, prompt, args, extra, ctx=None):
    """One `sofuu infer` run under /usr/bin/time -l. Returns a result dict."""
    cmd = ["/usr/bin/time", "-l", sofuu, "infer", model, prompt,
           "--temp", "0", "--no-lookahead", "-n", str(args.max_tokens)]
    if args.cpu:
        cmd.append("--cpu")
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    # An explicit ctx keeps max_seq_len (and therefore the KV allocation and
    # the mid-prompt compaction behaviour) deterministic per target length.
    effective_ctx = ctx if ctx is not None else args.ctx
    if effective_ctx:
        snapped = max(4, (effective_ctx + 3) // 4 * 4)
        cmd += ["--ctx", str(snapped)]
    t0 = _time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = 124, "", "harness timeout"
    wall = _time.time() - t0

    m = STATS_RE.search(err)
    prefill_toks = prefill_ms = decode_toks = decode_ms = None
    if m:
        prefill_toks = int(m.group(1))
        prefill_ms = 1000.0 * prefill_toks / float(m.group(2)) if float(m.group(2)) > 0 else None
        decode_toks = int(m.group(3))
        decode_ms = 1000.0 * decode_toks / float(m.group(4)) if float(m.group(4)) > 0 else None
    rss = RSS_RE.search(err)
    return {
        "rc": rc,
        "stdout": out,
        "error_hint": "" if rc == 0 else next(
            (l for l in reversed(err.strip().splitlines())
             if ("error" in l.lower() or "warning" in l.lower())),
            err.strip().splitlines()[-1] if err.strip() else "",
        ),
        "wall_s": round(wall, 2),
        "prefill_tokens": prefill_toks,
        "prefill_ms": prefill_ms,
        "decode_tokens": decode_toks,
        "decode_ms": decode_ms,
        "peak_rss_mib": round(int(rss.group(1)) / 1048576.0, 1) if rss else None,
    }


def calibrate_cpt(sofuu, model, args, probe_text):
    """Measure the corpus's real chars-per-token so the haystack hits its
    target token length. The probe must mirror the haystack's MIX: tiling
    cycles the same chunk list, so the calibration text is one even sample
    across that cycle (code ~1 c/t, prose ~4 c/t; a head-of-list sample
    biased 3x and made 8k-target runs prefill 23k tokens, tripping the
    engine's mid-prompt KV compaction and erasing the needle)."""
    r = run_infer(sofuu, model, probe_text, args, [], ctx=20000)
    if r["prefill_tokens"] and r["prefill_tokens"] > 300:
        cpt = len(probe_text) / r["prefill_tokens"]
        print(f"  calibrate {os.path.basename(model)}: {cpt:.3f} chars/token "
              f"({r['prefill_tokens']} tk / {len(probe_text)} chars)")
        return cpt
    print("  calibrate FAILED — falling back to --chars-per-token")
    return args.chars_per_token


def corpus_probe_text(chunks, cap=40000):
    """One even sample across the whole tiling cycle, capped."""
    total = sum(len(c) for c in chunks)
    if total <= cap:
        return "".join(chunks)
    # Keep every chunk but trim each proportionally — mix preserved.
    frac = cap / total
    return "".join(c[: max(200, int(len(c) * frac))] for c in chunks)


def kv_flags(kv):
    if kv == "f16":
        return []
    if kv == "kivi4":
        return ["--kivi", "--cpu"]
    if kv == "kivi2":
        return ["--kivi-bits", "2"]
    raise SystemExit(f"unknown kv mode: {kv}")


def niah_run(sofuu, model, args, length, depth_pct, kv, chunks, rng, out_fh, cpt):
    code = f"{rng.randrange(100000, 999999)}"
    needle = f"The secret access code to the vault is {code}."
    question = ("\n\nQuestion: What is the secret access code to the vault? "
                "Reply with only the code, nothing else.")
    target_chars = int(length * cpt[model])
    body = build_haystack(chunks, target_chars, rng)
    cut = int(len(body) * depth_pct / 100.0)
    prompt = body[:cut] + "\n\n" + needle + "\n\n" + body[cut:] + question

    r = run_infer(sofuu, model, prompt, args, kv_flags(kv), ctx=length + 1024)
    # NIAH scoring on the DIGIT stream: small models answer "17 78 43." —
    # the six digits are right, the spacing is formatting. Exact-substring
    # would score that as a miss.
    hit = code in re.sub(r"[^0-9]", "", r["stdout"])
    rec = {
        "gate": 1,
        "kind": "niah",
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": os.path.basename(model),
        "ctx_arg": (args.ctx if args.ctx else f"{length + 1024} (auto)"),
        "target_tokens": length,
        "depth_pct": depth_pct,
        "kv": kv,
        "code": code,
        "retrieved": hit,
        "chars_per_token": round(len(prompt) / max(1, r["prefill_tokens"] or 1), 3),
        **{k: r[k] for k in ("rc", "prefill_tokens", "prefill_ms", "decode_tokens",
                             "decode_ms", "peak_rss_mib", "wall_s", "error_hint")},
        "ttft_s": round((r["prefill_ms"] or 0) / 1000.0, 3),
        "out_tail": (r["stdout"].strip()[-120:] if not hit else ""),
    }
    out_fh.write(json.dumps(rec) + "\n")
    out_fh.flush()
    status = "PASS" if hit else "FAIL"
    print(f"  [{status}] {os.path.basename(model)} {length}tk d{depth_pct}% {kv}: "
          f"code={code} prefill={r['prefill_tokens']}tk "
          f"rss={r['peak_rss_mib']}MiB {r['error_hint'][:60]}")
    return hit


def decode_probe(sofuu, model, args, kv, chunks, rng, out_fh, cpt):
    """Gate 4 proxy: aggregate decode rate at -n probe vs -n 200."""
    body = build_haystack(chunks, int(1000 * cpt[model]), rng)
    prompt = body[: int(len(body) * 0.98)] + "\n\nContinue this text."
    rates = {}
    for label, n in (("short", 200), ("long", args.decode_probe)):
        cmd_extra = kv_flags(kv)
        saved = args.max_tokens
        args.max_tokens = n
        # ctx must cover prompt + probe + slack: a too-small ctx silently
        # engages mid-decode KV compaction, and the probe then measures
        # compaction work, not long-position attention cost.
        r = run_infer(sofuu, model, prompt, args, cmd_extra, ctx=1000 + args.decode_probe + 1024)
        args.max_tokens = saved
        rate = (r["decode_tokens"] / (r["decode_ms"] / 1000.0)) if r["decode_ms"] else None
        rates[label] = rate
        rec = {
            "gate": 4,
            "kind": "decode_probe",
            "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": os.path.basename(model),
            "kv": kv,
            "probe_tokens": n,
            "decode_tk_s": round(rate, 2) if rate else None,
            **{k: r[k] for k in ("rc", "prefill_tokens", "decode_tokens", "decode_ms",
                                 "peak_rss_mib", "wall_s")},
        }
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()
        print(f"  decode {label} ({n} tk): {rate and round(rate, 2)} tk/s rss={r['peak_rss_mib']}MiB")
    if rates["short"] and rates["long"]:
        ratio = rates["long"] / rates["short"]
        ok = ratio >= 0.9
        print(f"  Gate4 {'PASS' if ok else 'FAIL'}: long/short = {ratio:.3f} (≥ 0.90)")
        out_fh.write(json.dumps({
            "gate": 4, "kind": "decode_monotonicity", "model": os.path.basename(model),
            "kv": kv, "ratio_long_short": round(ratio, 4), "pass": ok,
        }) + "\n")
        return ok
    print("  Gate4 UNKNOWN: missing decode rates")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sofuu", default=DEFAULT_SOFUU, help="sofuu binary")
    ap.add_argument("--model", action="append", required=True, help="GGUF path (repeatable)")
    ap.add_argument("--lengths", default="8192,16384,32768",
                    help="target context lengths in tokens (comma list)")
    ap.add_argument("--depths", default="10,25,50,75,90", help="needle depths %% (comma list)")
    ap.add_argument("--kv", default="f16,kivi4", help="f16,kivi4,kivi2 (comma list)")
    ap.add_argument("--max-tokens", type=int, default=32, help="generation budget for NIAH")
    ap.add_argument("--decode-probe", type=int, default=0,
                    help="Gate 4 probe length (e.g. 16000); 0 = skip")
    ap.add_argument("--ctx", type=int, default=0, help="--ctx passed to the engine (0 = model default)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force --cpu (default uses the engine default)")
    ap.add_argument("--chars-per-token", type=float, default=3.8,
                    help="fallback chars/token when --no-auto-cpt is set")
    ap.add_argument("--auto-cpt", dest="auto_cpt", action="store_true", default=True,
                    help="calibrate chars/token per model with a probe run (default)")
    ap.add_argument("--no-auto-cpt", dest="auto_cpt", action="store_false",
                    help="use --chars-per-token for every model")
    ap.add_argument("--timeout", type=int, default=3600, help="per-run timeout seconds")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/bench/long-context.jsonl"))
    ap.add_argument("--corpus", default=None, help="extra corpus file first in the tiling order")
    ap.add_argument("--quick", action="store_true", help="smoke: one depth, one length, both kv")
    args = ap.parse_args()

    if not os.path.isfile(args.sofuu) or not os.access(args.sofuu, os.X_OK):
        sys.exit(f"no executable sofuu at {args.sofuu}")

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    kvs = [x.strip() for x in args.kv.split(",") if x.strip()]
    if args.quick:
        lengths, depths = [lengths[0]], [depths[len(depths) // 2]]

    chunks = load_corpus(args.corpus)
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Per-model calibration (code-heavy corpora tokenize near 1 char/token).
    probe_text = corpus_probe_text(chunks)
    cpt = {}
    for model in args.model:
        cpt[model] = calibrate_cpt(args.sofuu, model, args, probe_text) if args.auto_cpt else args.chars_per_token

    failures = 0
    with open(args.out, "a", encoding="utf-8") as out_fh:
        out_fh.write(json.dumps({
            "gate": 0, "kind": "suite_start", "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sofuu": args.sofuu, "corpus_chunks": len(chunks),
            "lengths": lengths, "depths": depths, "kv": kvs,
            "chars_per_token": {os.path.basename(m): round(v, 3) for m, v in cpt.items()},
            "seed": args.seed,
            "note": "corpus = local repo text (docs + shipped JS), tiled",
        }) + "\n")
        for model in args.model:
            for kv in kvs:
                for length in lengths:
                    print(f"── NIAH {os.path.basename(model)} {length}tk {kv} ──")
                    for depth in depths:
                        if not niah_run(args.sofuu, model, args, length, depth, kv, chunks, rng, out_fh, cpt):
                            failures += 1
                if args.decode_probe:
                    print(f"── Gate4 decode probe {os.path.basename(model)} {kv} ──")
                    if decode_probe(args.sofuu, model, args, kv, chunks, rng, out_fh, cpt) is False:
                        failures += 1
        # Gate 2 is structurally unmeasurable today — record it every run.
        out_fh.write(json.dumps({
            "gate": 2, "kind": "perplexity_ratio", "status": "blocked",
            "reason": "engine exposes no per-token logprobs; needs a logprob surface in infer (peer lane)",
        }) + "\n")
    print(f"\nbench_niah: {failures} failing run(s); receipts appended to {args.out}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
