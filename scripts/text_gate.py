#!/usr/bin/env python3
"""Two-route determinism gate with logits-tolerance certification.

Runs one model through two engine routes — the primary route (--cpu-env) and
the comparison variant (--gpu-env) — with SOFUU_DECODE_PROBE, then compares
the step-aligned sampled-token sequences:

  IDENTICAL          — every generated token matches.
  NEAR-TIE-CERTIFIED — every diverging step's two routes differ on the top-1
                       logit by < BOUND (the measured composite f32-noise
                       class), i.e. the fork is a certification-class near
                       tie, not an arithmetic defect.
  FORK               — a diverging step exceeds BOUND: bug class, do not
                       flip any defaults; bisect with SOFUU_QKV_SNAP /
                       kv_route_diff / scripts/kv_route_diff.py.
  ERROR              — the two routes did not produce a comparable decode.

Model selection (RULES.md 1.9): the gate measures real inference, so it
requires an existing local GGUF that carries real weights — MODEL, or
$SOFUU_CONTRACT_MODEL when MODEL is empty. The synthetic contract fixture is
a format fixture and is never inference evidence: `--allow-synthetic-fixture`
runs it as a pure two-route parser check whose receipt carries
`inference_evidence: false`. A missing, unreadable or unloadable model
blocks (exit 2, "BLOCKED:") instead of reporting a verdict.

Usage:
  python3 scripts/text_gate.py MODEL "PROMPT" [--max-tokens N] [--bound E]
      [--gpu-env SOFUU_X=1,SOFUU_Y=1] [--cpu-env SOFUU_NO_TILES=1]
      [--tier exact|tolerant|best-effort] [--backend cpu|gpu]
      [--prompt-id ID] [--receipt PATH|-] [--require-identical]
      [--allow-synthetic-fixture]
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
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF_MAGIC = 0x46554747
TOK = re.compile(r"tok=(\d+)")
V1 = re.compile(r"\(([-0-9.eE+]+),")
V2 = re.compile(r"\)\s*\(([-0-9.eE+]+),")
TOP = re.compile(r"\(([-0-9.eE+]+),(\d+)\)")
DECODE = re.compile(r"decode:\s*(\d+)\s*tk")

RECEIPT_FIELDS = (
    "kind", "schema_version", "model", "backend", "tier", "knobs",
    "prompt_id", "verdict", "numbers", "ts",
)
VERDICTS = ("IDENTICAL", "NEAR-TIE-CERTIFIED", "FORK", "ERROR")
TIERS = ("exact", "tolerant", "best-effort")
MODEL_KINDS = ("local-weights", "synthetic-fixture", "vocab-only")
MODEL_KIND_LABEL = {
    "local-weights": "real local weights",
    "synthetic-fixture": "synthetic fixture",
    "vocab-only": "vocabulary-only fixture",
}


def blocked(message):
    """Report a missing prerequisite. stdout is flushed first so a merged
    stdout/stderr stream keeps the diagnostic in order."""
    sys.stdout.flush()
    print(f"BLOCKED: {message}", file=sys.stderr)
    sys.stderr.flush()


def sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_env(s):
    return dict(kv.split("=", 1) for kv in s.split(",") if kv)


def resolve_binary():
    """The engine under test: SOFUU_BIN, then the release build, then ./sofuu."""
    env = os.environ.get("SOFUU_BIN", "").strip()
    if env:
        return env
    for cand in (os.path.join(ROOT, "target", "release", "sofuu"),
                 os.path.join(ROOT, "sofuu")):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return os.path.join(ROOT, "target", "release", "sofuu")


class _GGUF:
    """Minimal GGUF reader: header metadata only, never tensor payloads."""

    def __init__(self, fh):
        self.fh = fh

    def take(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")


def _gguf_value(g, vtype, collect):
    if vtype == 0:
        return g.take(1)[0]
    if vtype == 1:
        return struct.unpack("<b", g.take(1))[0]
    if vtype == 2:
        return struct.unpack("<H", g.take(2))[0]
    if vtype == 3:
        return struct.unpack("<h", g.take(2))[0]
    if vtype == 4:
        return g.u32()
    if vtype == 5:
        return struct.unpack("<i", g.take(4))[0]
    if vtype == 6:
        return struct.unpack("<f", g.take(4))[0]
    if vtype == 7:
        return bool(g.take(1)[0])
    if vtype == 8:
        return g.string()
    if vtype == 9:
        etype = g.u32()
        count = g.u64()
        if collect:
            return [_gguf_value(g, etype, True) for _ in range(count)]
        for _ in range(count):
            _gguf_value(g, etype, False)
        return None
    if vtype == 10:
        return g.u64()
    if vtype == 11:
        return struct.unpack("<q", g.take(8))[0]
    if vtype == 12:
        return struct.unpack("<d", g.take(8))[0]
    raise ValueError("unknown GGUF value type %d" % vtype)


def inspect_gguf(path):
    """Return (version, tensor_count, metadata) for a GGUF file, else None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
            if len(head) < 24:
                return None
            magic, version, tensor_count, kv_count = struct.unpack("<IIQQ", head)
            if magic != GGUF_MAGIC:
                return None
            g = _GGUF(fh)
            meta = {}
            for _ in range(kv_count):
                key = g.string()
                vtype = g.u32()
                if key in ("general.name", "general.architecture"):
                    meta[key] = _gguf_value(g, vtype, True)
                else:
                    _gguf_value(g, vtype, False)
            return version, tensor_count, meta
    except (OSError, ValueError, struct.error):
        return None


def classify_model(path, meta, tensor_count):
    """local-weights = inference evidence; the rest are parser fixtures."""
    base = os.path.basename(path).lower()
    name = str(meta.get("general.name", "")).lower()
    if (base == "sofuu-fixture.gguf" or name == "sofuu-fixture"
            or os.path.join("contract", "fixtures") in path):
        return "synthetic-fixture"
    if tensor_count == 0:
        return "vocab-only"
    return "local-weights"


def model_block(path, kind, meta, tensor_count):
    return {
        "path": path,
        "sha256": sha256_file(path),
        "kind": kind,
        "general_name": meta.get("general.name"),
        "architecture": meta.get("general.architecture"),
        "tensor_count": tensor_count,
    }


def build_receipt(args, verdict, numbers, steps_total, ctx=None):
    ctx = ctx or {}
    knobs = {}
    knobs.update(parse_env(args.cpu_env))
    knobs.update(parse_env(args.gpu_env))
    prompt_id = args.prompt_id or (
        "prompt-" + hashlib.sha256(args.prompt.encode()).hexdigest()[:12]
    )
    return {
        "kind": "text_gate",
        "schema_version": 1,
        "model": ctx.get("model") or {
            "path": args.model, "sha256": None, "kind": "unknown",
        },
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
        "routes": ctx.get("routes", {}),
        "binary": ctx.get("binary", {}),
        "inference_evidence": ctx.get("inference_evidence", False),
        "note": ctx.get("note", ""),
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
    if not isinstance(r["inference_evidence"], bool):
        raise ValueError("inference_evidence must be a boolean")
    if r["model"].get("kind") not in MODEL_KINDS + ("unknown",):
        raise ValueError(f"bad model kind {r['model'].get('kind')!r}")
    return True


def probe_rows(prefix):
    rows = []
    if not os.path.exists(prefix):
        return rows
    with open(prefix) as f:
        for ln in f:
            t = TOK.search(ln)
            v1 = V1.search(ln)
            v2 = V2.search(ln)
            if not (t and v1 and v2):
                continue
            top = TOP.search(ln)
            if not top:
                continue
            try:
                g1 = float(top.group(1))
                g2 = float(v2.group(1))
            except ValueError:
                continue
            rows.append((int(t.group(1)), g1, g1 - g2))
    return rows


def run(binary, env, prefix, model, prompt, max_tokens):
    """Run one route; return (exit_code, stderr_tail, decoded_tokens).

    The engine prints the generated text on stdout and the decode summary on
    stderr; only the summary is read, so no completion is retained.
    """
    e = dict(os.environ)
    e["SOFUU_DECODE_PROBE"] = prefix
    e.update(env)
    try:
        proc = subprocess.run(
            [binary, "infer", model, prompt, "--max-tokens", str(max_tokens),
             "--temp", "0"],
            env=e, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError as exc:
        return None, "infer could not start: %s" % exc, None
    err = proc.stderr.decode("utf-8", "replace").strip()
    m = DECODE.search(err)
    return proc.returncode, err[-400:], int(m.group(1)) if m else None


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
    ap.add_argument("--allow-synthetic-fixture", action="store_true",
                    help="run the synthetic fixture as a pure two-route check "
                         "(never inference evidence)")
    a = ap.parse_args()

    if a.self_check:
        sample = build_receipt(a, "IDENTICAL", {"dtop1": 0.0, "bound": 5e-2}, 16)
        check_receipt(sample)
        for f in RECEIPT_FIELDS:
            broken = dict(sample)
            broken.pop(f)
            try:
                check_receipt(broken)
            except ValueError:
                continue
            raise SystemExit(f"receipt check accepted a receipt missing '{f}'")
        print("receipt schema OK "
              f"({len(RECEIPT_FIELDS)} required fields; verdicts {VERDICTS}; tiers {TIERS})")
        return 0

    if not a.prompt:
        print("error: PROMPT is required (or use --self-check)", file=sys.stderr)
        return 2

    model_path = a.model.strip() or os.environ.get("SOFUU_CONTRACT_MODEL", "").strip()
    if not model_path:
        blocked("no model selected — pass MODEL or set SOFUU_CONTRACT_MODEL "
                "to an existing local real-weight GGUF (the synthetic fixture is "
                "not inference evidence)")
        return 2
    if not os.path.isfile(model_path):
        blocked(f"model file not found: {model_path}")
        return 2
    inspected = inspect_gguf(model_path)
    if inspected is None:
        blocked(f"not a readable GGUF file: {model_path}")
        return 2
    version, tensor_count, meta = inspected
    kind = classify_model(model_path, meta, tensor_count)
    if kind != "local-weights" and not a.allow_synthetic_fixture:
        blocked(f"{model_path} is a {MODEL_KIND_LABEL[kind]}, not inference "
                f"evidence — select a real-weight GGUF (MODEL or "
                f"SOFUU_CONTRACT_MODEL), or pass --allow-synthetic-fixture for a "
                f"pure two-route check")
        return 2

    binary = resolve_binary()
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        blocked(f"sofuu binary not found or not executable: {binary}")
        return 2

    if a.bound.lower() == "auto":
        a.bound = 0.0          # sentinel: resolved from matched steps
    else:
        a.bound = float(a.bound)
    gpu_env = parse_env(a.gpu_env)
    cpu_env = parse_env(a.cpu_env)

    cpu_prefix = f"/tmp/tg_{a.label}_cpu"
    gpu_prefix = f"/tmp/tg_{a.label}_gpu"
    for f in (cpu_prefix, gpu_prefix):
        if os.path.exists(f):
            os.remove(f)

    if kind == "local-weights":
        print(f"[{a.label}] model {model_path} "
              f"(name={meta.get('general.name')}, arch={meta.get('general.architecture')}, "
              f"tensors={tensor_count}, gguf v{version}) — real inference")
    else:
        print(f"[{a.label}] model {model_path} is a {MODEL_KIND_LABEL[kind]} — pure "
              f"two-route check, NOT inference evidence")

    cpu_code, cpu_err, cpu_tk = run(binary, cpu_env, cpu_prefix, model_path,
                                     a.prompt, a.max_tokens)
    gpu_code, gpu_err, gpu_tk = run(binary, gpu_env, gpu_prefix, model_path,
                                    a.prompt, a.max_tokens)

    ctx = {
        "model": model_block(model_path, kind, meta, tensor_count),
        "binary": {"path": binary, "sha256": sha256_file(binary)},
        "routes": {
            "cpu": {"env": cpu_env, "exit": cpu_code, "decode_tk": cpu_tk},
            "gpu": {"env": gpu_env, "exit": gpu_code, "decode_tk": gpu_tk},
        },
        "inference_evidence": kind == "local-weights",
        "note": ("real local weights; the engine decoded both routes"
                 if kind == "local-weights" else
                 f"{kind}: pure two-route determinism check, not inference evidence"),
    }

    def emit(verdict, numbers, steps_total):
        receipt = build_receipt(a, verdict, numbers, steps_total, ctx)
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

    for name, code, err in (("cpu", cpu_code, cpu_err), ("gpu", gpu_code, gpu_err)):
        if code is None or code != 0:
            blocked(f"the {name} route did not complete inference on "
                    f"{model_path} (exit={code}): {err or 'no diagnostic output'}")
            return 2

    c, g = probe_rows(cpu_prefix), probe_rows(gpu_prefix)
    ctx["routes"]["cpu"]["steps"] = len(c)
    ctx["routes"]["gpu"]["steps"] = len(g)
    if not c or not g:
        blocked(f"no decode steps recorded (cpu={len(c)} steps, "
                f"gpu={len(g)} steps) — the engine reported "
                f"decode_tk={cpu_tk}/{gpu_tk}")
        return 2

    n = min(len(c), len(g))
    diverge = []
    for i in range(n):
        if c[i][0] != g[i][0]:
            diverge.append(i)
    if not diverge and len(c) != len(g):
        # Same tokens over the shared prefix, but the routes stopped at
        # different steps: the sequences are not comparable end to end.
        print(f"[{a.label}] VERDICT=ERROR route step-count mismatch "
              f"(cpu={len(c)} steps, gpu={len(g)} steps)")
        rc = emit("ERROR", {"reason": "route step-count mismatch",
                            "cpu_steps": len(c), "gpu_steps": len(g)}, n)
        return 1 if rc == 0 else rc
    if not diverge:
        print(f"[{a.label}] VERDICT=IDENTICAL ({n} steps token-equal, both.sequence.len={len(c)}/{len(g)})")
        rc = emit("IDENTICAL", {"steps": n, "cpu_steps": len(c),
                                 "gpu_steps": len(g),
                                 "decode_tk": cpu_tk}, n)
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
