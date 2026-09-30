#!/usr/bin/env bash
# scripts/bench/fetch_datasets.sh — materialize the public eval sets the
# Q-phase scoreboard uses, as plain JSONL/TSV under $BENCH_DIR (default
# /tmp/sofuu_bench — NOT committed; the harness reads them from there).
#
# Sets:
#   sts-test.jsonl        STS benchmark test split (sentence pairs + gold score)
#   scifact-corpus.jsonl  BEIR SciFact corpus (~5.2k scientific abstracts)
#   scifact-queries.jsonl BEIR SciFact queries (300)
#   scifact-qrels.tsv     BEIR SciFact test qrels
#
# Sources are public and unauthenticated. SciFact ships as parquet, which the
# runtime deliberately does not depend on (a parquet reader would add ~10MB of
# Rust deps for a benchmark), so it is pulled through the HF datasets-server
# JSON API instead.
set -euo pipefail

BENCH_DIR="${SOFUU_BENCH_DIR:-/tmp/sofuu_bench}"
mkdir -p "$BENCH_DIR"

hf_get() { # url dest
  curl -fsSL --retry 3 --max-time 180 "$1" -o "$2"
}

echo "→ STS-B test split"
if [ ! -s "$BENCH_DIR/sts-test.jsonl" ]; then
  hf_get "https://huggingface.co/datasets/mteb/stsbenchmark-sts/resolve/main/test.jsonl.gz" \
    "$BENCH_DIR/sts-test.jsonl.gz"
  gunzip -f "$BENCH_DIR/sts-test.jsonl.gz"
fi

echo "→ SciFact qrels (test)"
if [ ! -s "$BENCH_DIR/scifact-qrels.tsv" ]; then
  hf_get "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv" \
    "$BENCH_DIR/scifact-qrels.tsv"
fi

# Paginated rows API: config/split come from the repo layout
# (corpus/queries/*.parquet each map to one config+split).
# The API rate-limits (HTTP 429) under fast paging, so requests are paced
# and retried with backoff, and progress resumes from the rows already on
# disk — a half-fetched corpus is never mistaken for a complete one.
fetch_split() { # repo config split dest
  local repo="$1" config="$2" split="$3" dest="$4"
  [ -s "$dest" ] && { echo "  (have $(basename "$dest"))"; return; }
  local offset=0 page rows attempt
  local tmp="$dest.tmp"
  # Resume a previous partial fetch instead of restarting it.
  if [ -s "$tmp" ]; then
    offset=$(wc -l < "$tmp" | tr -d ' ')
    echo "  resuming $(basename "$dest") at row ${offset}"
  else
    : > "$tmp"
  fi
  while :; do
    page=""
    attempt=0
    while [ -z "$page" ]; do
      attempt=$((attempt + 1))
      if page=$(curl -fsSL --max-time 180 \
        "https://datasets-server.huggingface.co/rows?dataset=${repo}&config=${config}&split=${split}&offset=${offset}&length=100" 2>/dev/null); then
        break
      fi
      [ "$attempt" -ge 8 ] && { echo "  giving up on offset ${offset}" >&2; return 1; }
      sleep $((attempt * 3))
    done
    rows=$(printf '%s' "$page" | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("rows",[])))')
    [ "$rows" = "0" ] && break
    printf '%s' "$page" | python3 -c '
import sys, json
d = json.load(sys.stdin)
for r in d.get("rows", []):
    print(json.dumps(r.get("row", {})))' >> "$tmp"
    offset=$((offset + rows))
    sleep 1
  done
  mv "$tmp" "$dest"
  echo "  $(basename "$dest"): $(wc -l < "$dest") rows"
}

echo "→ SciFact corpus (paginated)"
fetch_split "BeIR/scifact" "corpus" "corpus" "$BENCH_DIR/scifact-corpus.jsonl"
echo "→ SciFact queries (paginated)"
fetch_split "BeIR/scifact" "queries" "queries" "$BENCH_DIR/scifact-queries.jsonl"

echo
echo "Done. Datasets in $BENCH_DIR:"
ls -la "$BENCH_DIR"
echo
echo "Run the benchmark with:"
echo "  cargo run -p ml-train --release -- bench"
