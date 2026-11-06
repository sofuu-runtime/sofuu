#!/usr/bin/env python3
"""Extreme cold-start benchmark for the Sofuu runtime.

Measures execve -> runtime init -> minimal expression completion over many
runs, including:
  * process-spawn floor (/usr/bin/true)
  * CLI-only and engine-boot paths
  * current build vs a published darwin-arm64 artifact, when one is present
  * env variants that skip ML gates / tool drivers
  * concurrent cold-start bursts (1..200 simultaneous processes)
  * peak RSS per path

The page cache is WARM on macOS unless this script runs as root (purge needs
root); every number is therefore a warm-cache number, which is the steady
state for a repeatedly launched binary. The report says so explicitly.

Usage:
    python3 -I scripts/bench/extreme_cold_start.py

This is a standalone benchmark, not part of `make test`.
"""
import os
import platform
import statistics
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUR = os.path.join(ROOT, "sofuu")
OLD = os.path.join(ROOT, "dist", "sofuu-darwin-arm64")
TINY = "/tmp/sofuu-boot-tiny.js"
TRUE = "/usr/bin/true" if os.path.exists("/usr/bin/true") else "/bin/true"

with open(TINY, "w") as f:
    f.write("1+1;\n")

BASE_ENV = os.environ.copy()
BASE_ENV.pop("SOFUU_PROFILE", None)


def run_once(argv, env_extra=None, capture=False):
    env = BASE_ENV.copy()
    if env_extra:
        env.update(env_extra)
    out = subprocess.PIPE if capture else subprocess.DEVNULL
    t0 = time.perf_counter()
    r = subprocess.run(argv, stdout=out, stderr=subprocess.DEVNULL, env=env)
    dt = (time.perf_counter() - t0) * 1000.0
    if r.returncode != 0:
        raise RuntimeError("%s exited %d" % (argv, r.returncode))
    return (dt, r.stdout if capture else None)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(xs):
    xs = list(xs)
    return {
        "n": len(xs),
        "min": min(xs),
        "p50": pct(xs, .50),
        "p90": pct(xs, .90),
        "p95": pct(xs, .95),
        "p99": pct(xs, .99),
        "max": max(xs),
        "mean": statistics.fmean(xs),
        "sd": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
    }


def line(label, s):
    print("%-44s n=%-5d min=%6.2f  p50=%6.2f  p90=%6.2f  p95=%6.2f  p99=%6.2f  max=%7.2f  sd=%5.2f"
          % (label, s["n"], s["min"], s["p50"], s["p90"], s["p95"], s["p99"], s["max"], s["sd"]))


def bench(label, argv, runs, warmup=3, env_extra=None):
    for _ in range(warmup):
        run_once(argv, env_extra)
    xs = []
    for _ in range(runs):
        dt, _ = run_once(argv, env_extra)
        xs.append(dt)
    line(label, summarize(xs))
    return xs


def burst(n, rounds, binary):
    per = []
    totals = []
    for _ in range(rounds):
        barrier = threading.Barrier(n)
        results = [None] * n

        def worker(i):
            barrier.wait()
            try:
                dt, _ = run_once([binary, "eval", "1+1"])
                results[i] = dt
            except Exception:
                results[i] = float("nan")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        totals.append((time.perf_counter() - t0) * 1000.0)
        per.extend([x for x in results if x == x])
    return summarize(per), summarize(totals)


def rss_mb(argv):
    r = subprocess.run(["/usr/bin/time", "-l"] + argv,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for ln in r.stderr.decode("utf-8", "replace").splitlines():
        if "maximum resident set size" in ln:
            return int(ln.split()[0]) / (1024.0 * 1024.0)
    return None


def version_of(binary):
    return subprocess.run([binary, "version"], capture_output=True, text=True).stdout.replace("\n", " | ").strip()


def main():
    if not os.path.isfile(CUR):
        raise SystemExit("no sofuu binary at %s (run make)" % CUR)
    old_ok = os.path.isfile(OLD) and os.access(OLD, os.X_OK)

    print("Sofuu extreme cold-start benchmark")
    print("  host: %s %s, %d CPUs, load %.2f/%.2f/%.2f" % (
        platform.system(), platform.machine(), os.cpu_count() or 0,
        *os.getloadavg()))
    print("  current: %s" % version_of(CUR))
    if old_ok:
        print("  published: %s" % version_of(OLD))
    print("  page cache: WARM (macOS purge requires root; no cold-disk rows)")
    print()

    _, out = run_once([CUR, "eval", "1+1"], capture=True)
    assert out.strip() == b"2", "eval 1+1 did not print 2: %r" % out
    print("correctness: eval 1+1 ->", out.decode().strip())
    print()

    print("== sequential cold starts (wall clock, includes spawn) ==")
    bench("%s (spawn floor)" % TRUE, [TRUE], 1000)
    bench("current: version (CLI only)", [CUR, "version"], 500)
    if old_ok:
        bench("published: version (CLI only)", [OLD, "version"], 300)
    bench("current: eval 1+1 (engine boot)", [CUR, "eval", "1+1"], 500)
    if old_ok:
        bench("published: eval 1+1", [OLD, "eval", "1+1"], 300)
    bench("current: eval 1+1 (SOFUU_NO_ML=1)", [CUR, "eval", "1+1"], 300,
          env_extra={"SOFUU_NO_ML": "1"})
    bench("current: eval 1+1 (SOFUU_NO_TOOLS=1)", [CUR, "eval", "1+1"], 300,
          env_extra={"SOFUU_NO_TOOLS": "1"})
    bench("current: run tiny script", [CUR, "run", TINY], 300)
    bench("current: capabilities (Metal probe)", [CUR, "capabilities"], 150)
    print()

    print("== concurrent cold-start bursts (current binary) ==")
    print("%-8s %-4s %s" % ("N", "rnd", "per-process wall latency / burst wall"))
    for n, rounds in [(1, 100), (4, 50), (10, 30), (32, 15), (64, 10), (100, 5), (200, 3)]:
        per, total = burst(n, rounds, CUR)
        print("%-8d %-4d per: p50=%6.2f p95=%6.2f p99=%6.2f max=%7.2f | burst: p50=%7.2f p95=%7.2f max=%7.2f"
              % (n, rounds, per["p50"], per["p95"], per["p99"], per["max"],
                 total["p50"], total["p95"], total["max"]))
    print()

    if old_ok:
        print("== published artifact under the same burst (N=32) ==")
        per, total = burst(32, 10, OLD)
        print("published N=32      per: p50=%6.2f p95=%6.2f p99=%6.2f max=%7.2f | burst: p50=%7.2f p95=%7.2f max=%7.2f"
              % (per["p50"], per["p95"], per["p99"], per["max"],
                 total["p50"], total["p95"], total["max"]))
        print()

    print("== peak RSS (maximum resident set size) ==")
    rows = [
        ("current: version", [CUR, "version"]),
        ("current: eval 1+1", [CUR, "eval", "1+1"]),
        ("current: capabilities", [CUR, "capabilities"]),
    ]
    if old_ok:
        rows.append(("published: eval 1+1", [OLD, "eval", "1+1"]))
    for label, argv in rows:
        mb = rss_mb(argv)
        print("%-38s %8.2f MB" % (label, mb if mb is not None else -1))


if __name__ == "__main__":
    main()
