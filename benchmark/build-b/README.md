# Build B — The Fragmented Frankenstein Stack (the §8 benchmark's contrast)

The SAME application spec as [Build A](../build-a/app.js) — one streaming
chat with 2 tools, persistent memory, resume after restart, local fallback,
an OpenAI-compatible HTTP endpoint — assembled from the fragmented stack:
Node/Python + LangChain + a local daemon (Ollama/vLLM) + a vector DB daemon
(Chroma/sqlite-vec) + Express.

## Reproduce

```bash
docker compose up          # ollama + chroma + app (3 background processes)
npm install                # transitive supply chain
node app.js                # the app
```

## Files

- `app.js` — the application (Express + LangChain + clients for every daemon)
- `docker-compose.yml` — 3 services/ports (ollama 11434, chroma 8000, app 8080)
- `requirements.txt` + `package.json` — the two-language supply chain
