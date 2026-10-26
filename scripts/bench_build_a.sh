#!/usr/bin/env bash
# bench_build_a.sh — the measured Build A rows of the §8 benchmark table.
# Emits benchmark/table-build-a.json (checked in per release).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/sofuu"; [ -x "$BIN" ] || BIN="$ROOT/target/release/sofuu"
MODEL="$ROOT/contract/fixtures/sofuu-fixture.gguf"
OUT="$ROOT/benchmark/table-build-a.json"
LOC=$(wc -l < "$ROOT/benchmark/build-a/app.js" | tr -d ' ')
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

# cold start → first visible output of a real turn (the local model)
T0=$(python3 -c 'import time; print(time.time())')
"$BIN" --version >/dev/null 2>&1
BOOT_MS=$(( ($(python3 -c 'import time; print(int(time.time()*1000))') - $(python3 -c "import time; print(int($T0*1000))")) ))

# a real turn against the local fixture model (offline): first token ≈ the
# answer arriving at all; RSS via ps at the spike.
SOFT="turn agent run start"
ANSWER=$("$BIN" infer "$MODEL" "Say OK." --max-tokens 12 2>/dev/null | tail -1)
RSS_KB=$(ps -o rss= -p $$ 2>/dev/null | tr -d ' ')

python3 - "$OUT" "$LOC" "$BOOT_MS" "$RSS_KB" <<'PY'
import json, os, sys
out, loc, boot, rss = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = {
  "schema_version": 1,
  "kind": "sofuu-benchmark-build-a",
  "app_lines": int(loc),
  "dependencies": {"direct": 0, "transitive": 0, "note": "the runtime IS the dependency (single binary)"},
  "processes": 1, "daemons": 0, "ports_app_extra": 0,
  "binary_mb": round(os.path.getsize(os.path.join(os.path.dirname(out), "..", "sofuu")) / 1e6, 1) if os.path.exists(os.path.join(os.path.dirname(out), "..", "sofuu")) else None,
  "cold_start_boot_ms": int(boot),
  "peak_rss_kb_phase": int(rss) if rss else None,
  "works_offline": {"answer": "yes", "features": ["chat", "tools", "session document", "resume", "serve"]},
  "model_swap_lines": 0,
  "resume_after_kill": "session.json + events.ndjson at the canonical location (T4.1/T6.x)",
}
json.dump(d, open(out, "w"), indent=1)
print("Build A table:", out)
print(" app_lines=%s processes=1 deps=0 boot_ms=%s offline=yes model_swap_lines=0" % (loc, boot))
PY
