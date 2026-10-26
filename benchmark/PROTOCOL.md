# The "substantially better" benchmark protocol (PLAN §8)

Build the **same** application twice, from the same spec:

> Streaming chat + tool calls (2 tools) + persistent memory + resume after
> restart + local fallback when offline + HTTP endpoint (OpenAI-compatible).

- **Build A — Sofuu In-Process Substrate:** [`build-a/app.js`](build-a/app.js) —
  single file, public runtime API only, zero npm dependencies, one process,
  no daemons. The binary IS the app (`sofuu` < 5 MB).
- **Build B — The Fragmented Frankenstein Stack:** [`build-b/`](build-b/README.md)
  — Node + LangChain + Ollama daemon + Chroma daemon + Express; two supply
  chains (npm + pip), three background processes/ports.

## The table (record per release into `table.md`)

| Metric | Build A (Sofuu) | Build B (fragmented) | Measured by |
|---|---|---|---|
| lines of app code | `wc -l build-a/app.js` | `wc -l build-b/app.js` | repo |
| dependencies (direct/transitive) | 0 / 0 | 6 npm + 4 pip / hundreds | manifests |
| processes / daemons / ports | 1 / 0 / 1 | 3 / 2 / 3 | compose |
| cold start → first token | `bash scripts/bench_build_a.sh` | compose up + warm the daemons | script |
| peak RSS of the app process | same | same | script |
| works offline (features) | full (local model) | no (daemons need images/models fetched) | doc |
| model swap: cloud ↔ local | 1 env var (`SOFUU_BASE_URL`) | rewrite the provider wiring | diff |
| move laptop → phone | same binary family; the session doc ports (R4) | re-host the daemons | doc |
| resume after kill | `session.json` + events.ndjson (R4/R6) — asserted | rebuild Chroma state | script |

Reproduce with: `bash scripts/bench_build_a.sh` (Build A rows), and
`docker compose up` in `build-b/` for Build B. A stranger should be able to
re-run both from this directory alone.
