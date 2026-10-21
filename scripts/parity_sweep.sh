#!/bin/bash
# P0.1 standing parity harness (llama.cpp-parity program): llama-bench bar vs
# Sofuu best, same window, per model, pp+tg, appended machine-checkably to
# docs/bench/parity.jsonl. Bar = llama.cpp DEFAULT config (Metal offload).
# Acceptance: sofuu_pp >= ll_pp AND sofuu_tg >= ll_tg.
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
OUT=/tmp/parity_sweep_run.txt; : > $OUT
MODELS=(~/Downloads/Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf
  ~/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf ~/Downloads/Spark-X2.5-4B-Q4_K_M.gguf
  ~/Downloads/Sharp-MiniCPM5-2B-Q6_K_XL.gguf
  ~/Downloads/Phi-4-mini-reasoning-UD-Q6_K_XL.gguf ~/Downloads/Qwen3-8B-Q4_K_M.gguf)
BIGP=$(python3 -c "print('Harbor wakes under a grey sky while fishermen mend nets and the first ferry pulls away from the quay trailing foam past warehouses that once stored spices silk and salt for a vanished trade. '*6)")
for M in "${MODELS[@]}"; do
  N=$(basename "$M" .gguf)
  L=$(llama-bench -m "$M" -t 10 -p 210 -n 82 -r 3 2>/dev/null | grep -oE "\| +[0-9.]+ . [0-9.]+ \|$" | tr -d ' |' | sed 's/-/ /')
  LLPP=$(python3 -c "import sys; xs=sys.stdin.read().split(); print(xs[0] if xs else 0)" <<< "$L")
  LLTG=$(python3 -c "import sys; xs=sys.stdin.read().split(); print(xs[1] if len(xs)>1 else 0)" <<< "$L")
  BS=$(env SOFUU_GPU_DECODE=1 ./target/release/sofuu infer "$M" "Write three sentences about Paris." --max-tokens 82 --temp 0 2>&1 | grep "prefill:" | sed 's/\x1b\[[0-9;]*m//g' | tail -1)
  BP=$(env SOFUU_GPU_DECODE=1 SOFUU_GPU_PREFILL_GEMM=1 ./target/release/sofuu infer "$M" "$BIGP" --max-tokens 8 --temp 0 2>&1 | grep "prefill:" | sed 's/\x1b\[[0-9;]*m//g' | tail -1)
  BP12=$(env SOFUU_GPU_DECODE=1 SOFUU_GPU_PREFILL_GEMM=1 SOFUU_PREFILL_CHUNK=128 ./target/release/sofuu infer "$M" "$BIGP" --max-tokens 8 --temp 0 2>&1 | grep "prefill:" | sed 's/\x1b\[[0-9;]*m//g' | tail -1)
  echo "$N|$LLPP|$LLTG|$BS|$BP|chunk=c32,load=$(uptime | sed 's/.*averages: //' | cut -d, -f1)" >> $OUT
  echo "$N|$LLPP|$LLTG|$BS|$BP12|chunk=c128,load=$(uptime | sed 's/.*averages: //' | cut -d, -f1)" >> $OUT
done
python3 - "$OUT" <<'PYEOF'
import sys, json, datetime, re
now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
def stat(seg, i):  # i=0 prefill t/s, 1 decode t/s from the engine stats line
    m = re.findall(r"\(([\d.]+) tk/s\)", seg or "")
    return float(m[i]) if len(m) > i else 0.0
with open(sys.argv[1]) as f, open("docs/bench/parity.jsonl", "a") as out:
    for line in f:
        p = line.strip().split("|")
        if len(p) < 6: continue
        name, llpp, lltg, bs, bp, ld = p[0], float(p[1]), float(p[2]), p[3], p[4], p[5]
        dec = stat(bs, 1); pfx = max(stat(bp, 0), 0.0)
        row = {"model": name, "ll_pp": llpp, "ll_tg": lltg, "sofuu_pp": pfx,
               "sofuu_tg": dec, "pass_pp": bool(pfx >= llpp), "pass_tg": bool(dec >= lltg),
               "note": ld, "ts": now}
        out.write(json.dumps(row) + "\n")
        print(f"{name}: ll {llpp}/{lltg} vs sofuu {pfx:.1f}/{dec:.1f} -> pp={'PASS' if row['pass_pp'] else 'fail'} tg={'PASS' if row['pass_tg'] else 'fail'} ({ld})")
PYEOF
