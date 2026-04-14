# ⚡ Sofuu (素風)

> *A simple, fast JavaScript runtime built natively in C for the AI era.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0--alpha-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()

> ⚠️ **Experimental:** Sofuu is a research-grade runtime under active development. APIs may change.

---

## What is Sofuu?

Sofuu is a **server-side JavaScript runtime** — think Node.js, but built entirely in C using QuickJS instead of V8. It has no browser API, no DOM, and no renderer. It is purpose-built for:

- **AI backends** — stream LLM responses from any provider with zero npm installs
- **Edge & embedded scripting** — runs on hardware where Node.js can't
- **Lightweight microservices** — native HTTP server with a ~1MB binary

You build your frontend (React, Next.js, plain HTML) separately. Sofuu powers the backend logic.

---

## Install

```bash
curl -fsSL https://sofuu.xyz/install | sh
```

Or build from source:

```bash
git clone https://github.com/sofuu/sofuu
cd sofuu
make
./sofuu version
```

---

## 30-Second Demo

**Stream an LLM response — no npm, no SDK, just Sofuu:**

```js
// server.js
const server = sofuu.http.createServer(async (req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });

  const stream = sofuu.ai.stream("Explain SIMD in one sentence.", {
    provider: "ollama",
    model: "llama3",
  });

  for await (const chunk of stream) {
    res.write(chunk.text);
  }

  res.end();
});

server.listen(3000, "0.0.0.0");
console.log("Sofuu AI server running on port 3000");
```

```bash
sofuu run server.js
```

**TypeScript — no compiler, no config:**

```typescript
interface Message { role: string; content: string; }
const greet = (msg: Message): string => `[${msg.role}] ${msg.content}`;
console.log(greet({ role: 'user', content: 'Hello Sofuu!' }));
```

```bash
sofuu run hello.ts   # works out of the box
```

**Vector similarity — hardware SIMD, no npm packages:**

```js
const a = new Float32Array([0.1, 0.9, 0.4]);
const b = new Float32Array([0.2, 0.8, 0.5]);

// Executes on CPU NEON/AVX2 vector registers directly
const score = sofuu.ai.similarity(a, b);
console.log(score); // 0.998...
```

---

## Why Sofuu?

| Feature | Node.js | Deno | Bun | **Sofuu** |
|---|---|---|---|---|
| Binary size | ~100MB | ~80MB | ~60MB | **~1.1MB** |
| Startup time | ~50ms | ~30ms | ~7ms | **~3ms** |
| LLM streaming | npm install | npm install | npm install | **Native C** |
| Vector SIMD ops | npm install | npm install | npm install | **Native NEON/AVX2** |
| HTTP server | npm install | built-in | built-in | **Native C** |
| NPM Packages | ✅ | via import map | ✅ | **`sofuu add` + CJS shim** |
| TypeScript | tsc / swc | built-in | built-in | **Built-in C stripper** |
| Embeddable in C | ❌ | ❌ | ❌ | **✅** |

---

## CLI

```bash
sofuu                        # Start interactive REPL
sofuu run <file.js>          # Run a JavaScript file
sofuu run <file.ts>          # Run a TypeScript file (auto-transpiled)
sofuu eval "<code>"          # Evaluate JS inline
sofuu bundle <entry.js>      # Bundle to single distributable file
sofuu bundle <entry.js> -o <out.js>
sofuu install                # Install from package.json
sofuu add <pkg>              # Install npm package
sofuu version                # Print version and exit
sofuu help                   # Print this help
```

---

## API Reference

### `sofuu.ai`

```js
// Streaming — background C-thread over libuv, zero blocking
const stream = sofuu.ai.stream("Explain gravity.", {
  provider: "ollama",  // 'openai' | 'anthropic' | 'gemini' | 'ollama'
  model: "llama3",
});
for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}

// Completion
const result = await sofuu.ai.complete("What is 2+2?", { provider: "openai" });
console.log(result.text);    // "4"
console.log(result.tokens);  // { input: 12, output: 1 }
```

### SIMD Vector Math

```js
// All functions accept Float32Array (zero-copy) or plain Array
sofuu.ai.similarity(a, b)   // Cosine similarity  → [-1, 1]
sofuu.ai.dot(a, b)           // Dot product        → number
sofuu.ai.l2(a, b)            // Euclidean distance → number
```

### `sofuu.http`

```js
// Native HTTP server — no Express, no frameworks needed
const server = sofuu.http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ status: "online" }));
});
server.listen(8080, "0.0.0.0");
```

### `sofuu.fs`

```js
const data = await sofuu.fs.readFile('./data.json', 'utf8');
await sofuu.fs.writeFile('./out.txt', 'Hello Sofuu!');
const entries = await sofuu.fs.readdir('./src');
```

### `sofuu.spawn`

```js
const proc = sofuu.spawn("python3", ["script.py"]);
proc.stdout.on("data", (chunk) => console.log(chunk));
await proc.wait();
```

### Globals

```js
// Standard globals available everywhere
setTimeout(() => console.log("done"), 1000);
setInterval(() => console.log("tick"), 500);
await sleep(500);          // Native C sleep (no Promise overhead)
process.env.OPENAI_KEY     // environment variables
process.argv               // CLI arguments
process.cwd()              // current directory
console.log / warn / error // full console support
```

---

## Architecture

```
sofuu run agent.ts
       ↓
  [TS type stripper]   — pure C, zero deps, in-process
       ↓
  [QuickJS engine]     — ES2023, ESM modules
       ↓
  [libuv event loop]   — async I/O, timers, subprocess
       ↓
  Native C modules:
    ├── sofuu.http      (HTTP server + fetch client)
    ├── sofuu.ai        (LLM streaming + SIMD vector math)
    ├── sofuu.fs        (async file I/O)
    ├── sofuu.spawn     (OS process control)
    └── sofuu.os        (process, env, timers)
```

---

## Build from Source

```bash
# Prerequisites: clang, make, libcurl
git clone https://github.com/sofuu/sofuu
cd sofuu
make           # auto-detects arm64 (NEON) or x86_64 (AVX2)
make test      # run full test suite
make install   # copy to /usr/local/bin
```

---

## Roadmap

- [x] Phase 1 — QuickJS core (ESM, console, process)
- [x] Phase 2 — libuv event loop (timers, fs, subprocess)
- [x] Phase 3 — Networking (fetch, HTTP server, SSE)
- [x] Phase 4 — AI modules (ai.complete, ai.stream)
- [x] Phase 5 — SIMD vectors (NEON / AVX2)
- [x] Phase 6 — npm resolver (`sofuu add <pkg>`) + CJS shim
- [x] Phase 7 — TypeScript (built-in C type stripper)
- [x] Phase 8 — Interactive REPL
- [ ] Phase 9 — MCP client + server (`mcp.connect()`, `mcp.serve()`)
- [ ] Phase 10 — Agent runtime (`new Agent({ tools, model })`)
- [ ] Phase 11 — Cross-compilation (macOS → Linux → Windows)
- [ ] Phase 12 — Stable `sofuu.xyz` install script + docs

---

## License

MIT © Sofuu Contributors
