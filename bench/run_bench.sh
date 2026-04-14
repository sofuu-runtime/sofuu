#!/usr/bin/env bash
# bench/run_bench.sh — Sofuu benchmark suite
# Compares startup time, CPU (fibonacci), and async I/O vs Node/Bun/Deno
set -euo pipefail

SOFUU=./sofuu
ROUNDS=5   # warmup + measure

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Sofuu Benchmark Suite                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

run_bench() {
    local label="$1"
    local cmd="$2"
    local sum=0
    # warm-up
    eval "$cmd" > /dev/null 2>&1 || true
    for i in $(seq 1 $ROUNDS); do
        local t0=$(date +%s%3N)
        eval "$cmd" > /dev/null 2>&1 || true
        local t1=$(date +%s%3N)
        sum=$((sum + t1 - t0))
    done
    echo $((sum / ROUNDS))
}

# ── Startup time ──────────────────────────────────────────────
echo "📊 Startup time (sofuu run bench/startup.js)"
echo "   (lower is better, ms)"
echo ""

printf "%-12s %6s ms\n" "sofuu" "$(run_bench sofuu "$SOFUU run bench/startup.js")"

if command -v node &>/dev/null; then
    printf "%-12s %6s ms\n" "node" "$(run_bench node "node bench/startup.js")"
fi
if command -v bun &>/dev/null; then
    printf "%-12s %6s ms\n" "bun" "$(run_bench bun "bun bench/startup.js")"
fi
if command -v deno &>/dev/null; then
    printf "%-12s %6s ms\n" "deno" "$(run_bench deno "deno run bench/startup.js")"
fi

echo ""
echo "📊 CPU: fib(40)  (lower is better, ms)"
echo ""

printf "%-12s %6s ms\n" "sofuu" "$(run_bench sofuu "$SOFUU run bench/fib.js")"
if command -v node &>/dev/null; then printf "%-12s %6s ms\n" "node"  "$(run_bench node  "node bench/fib.js")"; fi
if command -v bun  &>/dev/null; then printf "%-12s %6s ms\n" "bun"   "$(run_bench bun   "bun bench/fib.js")";  fi
if command -v deno &>/dev/null; then printf "%-12s %6s ms\n" "deno"  "$(run_bench deno  "deno run bench/fib.js")"; fi

echo ""
echo "✅ Done"
echo ""
