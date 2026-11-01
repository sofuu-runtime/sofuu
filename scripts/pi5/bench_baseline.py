#!/usr/bin/env python3
"""scripts/pi5/bench_baseline.py — the M-R0 receipt for the Pi 5 program.

Records, on the device and in one JSON file:

  * cold start   — `sofuu eval "1+1"`, p50/p95, spawn included
  * idle RSS     — KiB while the loop is alive, from /proc
  * binary       — revision stamp, exact bytes, cap verdict
  * inference    — prefill/decode tok/s per pass, parsed from the runtime's
                   own `── prefill … decode … ──` receipt line
  * thermals     — SoC temperature and `vcgencmd get_throttled` while a
                   sustained inference soak runs; time to the soft limit and
                   the decode decay between the first and last pass

It never downloads anything. Point --model at a GGUF already on the device
(repeatable); with no flag, every known on-device path that exists is measured
and the inference section is marked skipped when none does.

  python3 scripts/pi5/bench_baseline.py \
      --model ~/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf \
      --soak-seconds 300 --out ~/pi5-baseline.json

Exit codes: 0 ok · 2 usage · 3 binary missing · 4 measurement failed ·
            5 an explicitly named model is missing.
"""

import argparse
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

CAP_BYTES = 5_767_168  # RULES.md §3.1 — 5.5 MiB
DEFAULT_MODELS = [
    "~/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf",
    "~/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf",
]
DEFAULT_BINARIES = [
    "~/Downloads/sofuu-linux-arm64",
    "~/Downloads/sofuu-linux-arm64-qtsq",
    "~/sofuu-build/sofuu-linux-arm64-qtsq",
]

# "── prefill: 234 tk in 1.23s (190.2 tk/s) · decode: 64 tk in 2.10s (30.5 tk/s) · total: 3.33s ──"
RECEIPT_RE = re.compile(
    r"prefill:\s*(\d+)\s*tk in ([\d.]+)(ns|µs|ms|s) \(([\d.]+) tk/s\)"
    r"\s*·\s*decode:\s*(\d+)\s*tk in ([\d.]+)(ns|µs|ms|s) \(([\d.]+) tk/s\)"
    r"\s*·\s*total:\s*([\d.]+)(ns|µs|ms|s)"
)
UNIT_S = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}

# vcgencmd get_throttled bit meanings (Raspberry Pi firmware).
THROTTLE_NOW = {
    0: "under_voltage",
    1: "freq_capped",
    2: "throttled",
    3: "soft_temp_limit",
}
THROTTLE_WAS = {
    16: "under_voltage_occurred",
    17: "freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def die(code, message):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def first_existing(paths):
    for p in paths:
        q = os.path.expanduser(p)
        if os.path.exists(q):
            return q
    return None


def run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ── host ───────────────────────────────────────────────────────────────


def cpu_model():
    try:
        for line in read_text("/proc/cpuinfo").splitlines():
            if line.lower().startswith("model name") or line.lower().startswith("hardware"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def governor():
    path = "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
    try:
        return read_text(path)
    except OSError:
        return None


def temp_c():
    """SoC temperature in °C, or None when this host has no sensor."""
    try:
        return int(read_text("/sys/class/thermal/thermal_zone0/temp")) / 1000.0
    except (OSError, ValueError):
        pass
    vc = shutil.which("vcgencmd")
    if vc:
        try:
            out = run([vc, "measure_temp"]).stdout
            m = re.search(r"([\d.]+)", out)
            if m:
                return float(m.group(1))
        except OSError:
            pass
    return None


def throttled():
    vc = shutil.which("vcgencmd")
    if not vc:
        return None
    try:
        out = run([vc, "get_throttled"]).stdout.strip()
    except OSError:
        return None
    m = re.search(r"0x([0-9a-fA-F]+)", out)
    if not m:
        return None
    val = int(m.group(1), 16)
    return {
        "raw": hex(val),
        "now": [name for bit, name in THROTTLE_NOW.items() if val & (1 << bit)],
        "occurred": [name for bit, name in THROTTLE_WAS.items() if val & (1 << bit)],
    }


# ── measurements ───────────────────────────────────────────────────────


def version_info(binary):
    out = {"revision": None, "version": None, "platform": None}
    try:
        res = run([binary, "version"], timeout=30)
        for line in res.stdout.splitlines():
            if line.startswith("sofuu "):
                out["version"] = line.split(" ", 1)[1].strip()
            elif line.startswith("revision "):
                out["revision"] = line.split(" ", 1)[1].strip()
            elif line.startswith("platform:"):
                out["platform"] = line.split(":", 1)[1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return out


def cold_start(binary, runs, warmup):
    for _ in range(warmup):
        try:
            run([binary, "eval", "1+1"], timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
    times, errors = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            res = run([binary, "eval", "1+1"], timeout=30)
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(str(e))
            continue
        dt = (time.perf_counter() - t0) * 1000.0
        if res.returncode == 0:
            times.append(dt)
        else:
            errors.append((res.stderr or "").strip()[:200])
    if not times:
        return {"runs": runs, "ok": False, "errors": errors}
    ordered = sorted(times)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "runs": runs,
        "ok": True,
        "median_ms": round(statistics.median(times), 2),
        "p95_ms": round(ordered[p95_index], 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "errors": errors,
    }


def read_rss_kb(pid):
    try:
        for line in read_text(f"/proc/{pid}/status").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    try:
        out = run(["ps", "-o", "rss=", "-p", str(pid)]).stdout.strip()
        return int(out) if out else None
    except (OSError, ValueError):
        return None


def idle_rss(binary, settle_s=1.5):
    """RSS while the event loop is alive (a 4 s timer keeps it up)."""
    try:
        proc = subprocess.Popen(
            [binary, "eval", "setTimeout(function(){}, 4000)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    try:
        time.sleep(settle_s)
        return read_rss_kb(proc.pid)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def parse_receipt(stderr):
    m = RECEIPT_RE.search(stderr)
    if not m:
        return None
    return {
        "prefill_tokens": int(m.group(1)),
        "prefill_s": round(float(m.group(2)) * UNIT_S[m.group(3)], 4),
        "prefill_tps": float(m.group(4)),
        "decode_tokens": int(m.group(5)),
        "decode_s": round(float(m.group(6)) * UNIT_S[m.group(7)], 4),
        "decode_tps": float(m.group(8)),
        "total_s": round(float(m.group(9)) * UNIT_S[m.group(10)], 4),
    }


def infer_once(binary, model, prompt, tokens, temp, trace_path):
    cmd = [binary, "infer", model, prompt,
           "-n", str(tokens), "--temp", str(temp)]
    if trace_path:
        cmd += ["--trace", trace_path]
    t0 = time.perf_counter()
    try:
        res = run(cmd, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "infer timed out (1800 s)"}
    wall = time.perf_counter() - t0
    receipt = parse_receipt(res.stderr or "")
    row = {
        "ok": res.returncode == 0 and receipt is not None,
        "wall_s": round(wall, 3),
        "returncode": res.returncode,
    }
    if receipt:
        row.update(receipt)
    else:
        row["stderr_tail"] = (res.stderr or "").strip()[-400:]
    return row


def soak(binary, model, prompt, tokens, temp, seconds, workdir):
    """Repeated short passes under continuous load.

    A single long generation would give better wall-clock continuity but the
    CLI writes its receipt and trace only when generation ends; killing it
    would lose both. Short passes keep every receipt and still keep the CPU
    hot: 64 tokens at ~24 t/s is a ~3 s pass plus a ~50 ms model load.
    """
    deadline = time.time() + seconds
    passes, samples = [], []
    first_throttle_s = None
    max_temp = None
    started = time.time()

    throttle_before = throttled()
    while time.time() < deadline:
        t_before = temp_c()
        tr = os.path.join(workdir, f"soak-{len(passes):03d}.jsonl")
        row = infer_once(binary, model, prompt, tokens, temp, tr)
        t_after = temp_c()
        elapsed = round(time.time() - started, 1)
        th_after = throttled()
        row["pass"] = len(passes)
        row["t_s"] = elapsed
        row["temp_before_c"] = t_before
        row["temp_after_c"] = t_after
        row["throttled_now"] = (th_after or {}).get("now", [])
        passes.append(row)

        for value in (t_before, t_after):
            if value is not None:
                samples.append({"t_s": elapsed, "temp_c": value})
                max_temp = value if max_temp is None else max(max_temp, value)
        if first_throttle_s is None and row["throttled_now"]:
            first_throttle_s = elapsed
        time.sleep(0.5)

    decode_values = [p["decode_tps"] for p in passes if p.get("ok") and p.get("decode_tps")]
    decay = None
    if len(decode_values) >= 4:
        half = len(decode_values) // 2
        first = statistics.median(decode_values[:half])
        last = statistics.median(decode_values[-half:])
        decay = {
            "first_half_decode_tps": round(first, 2),
            "second_half_decode_tps": round(last, 2),
            "ratio": round(last / first, 3) if first else None,
        }

    return {
        "requested_s": seconds,
        "actual_s": round(time.time() - started, 1),
        "passes": passes,
        "passes_ok": sum(1 for p in passes if p.get("ok")),
        "temp_samples": samples,
        "temp_max_c": max_temp,
        "throttled_before": throttle_before,
        "throttled_after": throttled(),
        "first_throttle_s": first_throttle_s,
        "decode_decay": decay,
    }


# ── main ───────────────────────────────────────────────────────────────


def main(argv):
    ap = argparse.ArgumentParser(
        description="Record the Sofuu Pi 5 M-R0 baseline receipt.")
    ap.add_argument("--binary", help="sofuu binary (default: probe known paths)")
    ap.add_argument("--runs", type=int, default=30, help="cold-start runs (default 30)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed warmups (default 3)")
    ap.add_argument("--model", action="append",
                    help="GGUF for the inference/thermal rows (repeatable; default: "
                         "every known on-device model that exists)")
    ap.add_argument("--prompt", default="The capital of France is",
                    help="prompt for the inference rows")
    ap.add_argument("--prompt-file", help="read the prompt from this file instead")
    ap.add_argument("--tokens", type=int, default=64, help="tokens per soak pass (default 64)")
    ap.add_argument("--temp", type=float, default=0.0, help="sampling temperature (default 0)")
    ap.add_argument("--soak-seconds", type=int, default=0,
                    help="sustained-inference length; 0 skips the thermal soak")
    ap.add_argument("--out", help="receipt path (default ./pi5-baseline-<utc>.json)")
    ap.add_argument("--json-only", action="store_true", help="suppress the human table")
    args = ap.parse_args(argv)

    if args.runs < 1:
        die(2, "--runs must be >= 1")
    if args.soak_seconds < 0:
        die(2, "--soak-seconds must be >= 0")

    binary = args.binary or first_existing(DEFAULT_BINARIES)
    if not binary:
        die(3, "no sofuu binary: pass --binary (probed: %s)"
            % ", ".join(DEFAULT_BINARIES))
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        die(3, f"{binary} is not an executable file")

    models = []
    if args.model:
        for given in args.model:
            path = os.path.expanduser(given)
            if not os.path.isfile(path):
                die(5, f"--model {path} does not exist (nothing is downloaded here)")
            models.append(path)
    else:
        models = [p for p in (os.path.expanduser(x) for x in DEFAULT_MODELS)
                  if os.path.isfile(p)]

    prompt = args.prompt
    if args.prompt_file:
        try:
            prompt = read_text(args.prompt_file)
        except OSError as e:
            die(2, f"--prompt-file: {e}")

    out_path = args.out or os.path.expanduser(
        f"~/pi5-baseline-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")

    print("Sofuu Pi 5 baseline — this takes a few minutes" + (
        "" if not args.soak_seconds else f" (+{args.soak_seconds}s soak)"))
    print(f"  binary: {binary}")
    if models:
        for m in models:
            print(f"  model:  {m}")
    else:
        print("  model:  (none found — inference rows will be skipped)")
    print()

    size = os.path.getsize(binary)
    receipt = {
        "schema": "sofuu.pi5.baseline.v1",
        "generated_at": now_iso(),
        "host": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "cpu": cpu_model(),
            "governor": governor(),
        },
        "binary": {
            "path": binary,
            "size_bytes": size,
            "cap_bytes": CAP_BYTES,
            "cap_ok": size <= CAP_BYTES,
            **version_info(binary),
        },
        "cold_start": cold_start(binary, args.runs, args.warmup),
        "idle_rss_kb": idle_rss(binary),
        "inference": [],
        "thermal": None,
    }

    workdir = tempfile.mkdtemp(prefix="sofuu-pi5-")
    receipt["traces_dir"] = workdir

    for index, model in enumerate(models):
        print(f"  inference pass {index + 1}/{len(models)}: {os.path.basename(model)} "
              f"({args.tokens} tokens) …")
        single = infer_once(binary, model, prompt, args.tokens, args.temp,
                            os.path.join(workdir, f"single-{index}.jsonl"))
        receipt["inference"].append({
            "model": model,
            "prompt_chars": len(prompt),
            "single": single,
        })

    if models and args.soak_seconds > 0:
        print(f"  thermal soak ({args.soak_seconds}s) on {os.path.basename(models[0])} …")
        receipt["thermal"] = soak(
            binary, models[0], prompt, args.tokens, args.temp,
            args.soak_seconds, workdir)
        receipt["thermal"]["model"] = models[0]

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
            f.write("\n")
    except OSError as e:
        die(4, f"cannot write {out_path}: {e}")

    if not args.json_only:
        b = receipt["binary"]
        print()
        print("── M-R0 receipt ──────────────────────────────────────────")
        print(f"  host      {receipt['host']['hostname']} · {receipt['host']['machine']} · "
              f"{receipt['host']['kernel']}")
        print(f"  governor  {receipt['host']['governor'] or 'unknown'}")
        print(f"  binary    {b['version'] or '?'} · revision {b['revision'] or '?'}")
        print(f"  size      {b['size_bytes']} B / cap {CAP_BYTES} B · "
              f"{'PASS' if b['cap_ok'] else 'OVER'} "
              f"({abs(CAP_BYTES - b['size_bytes'])} B)")
        cs = receipt["cold_start"]
        if cs.get("ok"):
            print(f"  cold start p50 {cs['median_ms']} ms · p95 {cs['p95_ms']} ms "
                  f"({cs['runs']} runs)")
        else:
            print(f"  cold start FAILED ({len(cs.get('errors', []))} errors)")
        print(f"  idle RSS  {receipt['idle_rss_kb']} KiB" if receipt["idle_rss_kb"]
              else "  idle RSS  unavailable")
        inf = receipt["inference"]
        if inf:
            for row in inf:
                s = row["single"]
                label = os.path.basename(row["model"])[:34]
                if s.get("ok"):
                    print(f"  infer     {label:<34} prefill {s['prefill_tps']:>6} tk/s · "
                          f"decode {s['decode_tps']:>6} tk/s ({s['decode_tokens']} tk)")
                else:
                    print(f"  infer     {label:<34} FAILED — see receipt")
        else:
            print("  infer     skipped (no model)")
        th = receipt["thermal"]
        if th:
            print(f"  thermal   max {th['temp_max_c']} °C over {th['actual_s']} s · "
                  f"first throttle {th['first_throttle_s']} s · "
                  f"{th['passes_ok']}/{len(th['passes'])} passes ok")
            if th["decode_decay"]:
                d = th["decode_decay"]
                print(f"  decay     {d['first_half_decode_tps']} → "
                      f"{d['second_half_decode_tps']} tk/s "
                      f"(ratio {d['ratio']})")
        print(f"  receipt   {out_path}")
        print("──────────────────────────────────────────────────────────")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
