# ⚡ Sofuu (素風)

> *A fast, private JavaScript runtime built in C — designed from day one for AI-era workloads.*

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20NC%201.0.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0--alpha-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()

> ⚠️ **Alpha software.** Sofuu is under active development. APIs may change before a stable release.

---

## What is Sofuu?

Sofuu is a **server-side JavaScript runtime** — the kind of thing you'd use to run your AI backend, your API server, or your edge automation script. It has no browser API, no DOM, and no renderer.

What makes it different from Node.js, Deno, or Bun is that all the things you normally have to install separately — LLM streaming, vector math, MCP client/server, HTTP — are just built in at the C level. No npm packages needed for any of it.

You build your frontend (React, Next.js, plain HTML — whatever you like) completely separately. Sofuu runs the server side.

---

## Install

```bash
curl -fsSL https://sofuu.xyz/install | sh
```

Or build it yourself from source:

```bash
git clone https://github.com/sofuu-runtime/sofuu
cd sofuu
make
./sofuu version
```

---

## Quick Start

**Stream an AI response — no packages to install, no SDK to configure:**

```js
// server.js
const server = sofuu.http.createServer(async (req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });

  const stream = sofuu.ai.stream("Explain how SIMD works.", {
    provider: "ollama",
    model: "llama3",
  });

  for await (const chunk of stream) {
    res.write(chunk.text);
  }

  res.end();
});

server.listen(3000, "0.0.0.0");
console.log("Server running on port 3000");
```

```bash
sofuu run server.js
```

**TypeScript works out of the box — no tsconfig, no compiler step:**

```typescript
interface Message { role: string; content: string; }
const greet = (msg: Message): string => `[${msg.role}] ${msg.content}`;
console.log(greet({ role: "user", content: "Hello, Sofuu!" }));
```

```bash
sofuu run hello.ts
```

---

## Why Sofuu?

| | Node.js | Deno | Bun | **Sofuu** |
|---|---|---|---|---|
| Binary size | ~100 MB | ~80 MB | ~60 MB | **~1.1 MB** |
| Startup time | ~50 ms | ~30 ms | ~7 ms | **~3 ms** |
| LLM streaming | npm install | npm install | npm install | **Built in (C)** |
| MCP client + server | npm install | npm install | npm install | **Built in (C)** |
| SIMD vector math | npm install | npm install | npm install | **Built in (NEON/AVX2)** |
| TypeScript support | Separate compiler | Built in | Built in | **Built in (C type stripper)** |
| Embeddable in C apps | ❌ | ❌ | ❌ | **✅** |

---

## CLI Reference

```bash
sofuu                        # Start an interactive REPL
sofuu run <file.js>          # Run a JavaScript file
sofuu run <file.ts>          # Run a TypeScript file (auto-transpiled)
sofuu eval "<code>"          # Evaluate a JS expression inline
sofuu bundle <entry.js>      # Bundle everything into one distributable file
sofuu bundle <entry.js> -o <out.js>
sofuu install                # Install packages from package.json
sofuu add <package>          # Add and install an npm package
sofuu version                # Print version information
sofuu help                   # Print usage help
```

---

## API Reference

### `sofuu.ai` — LLM Streaming & Vector Math

```js
// Stream tokens as they arrive — non-blocking, C-backed
const stream = sofuu.ai.stream("What is the speed of light?", {
  provider: "ollama",   // "openai" | "anthropic" | "gemini" | "ollama"
  model: "llama3",
});
for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}

// One-shot completion
const result = await sofuu.ai.complete("What is 2 + 2?", { provider: "openai" });
console.log(result.text);    // "4"
console.log(result.tokens);  // { input: 12, output: 1 }

// SIMD-accelerated vector similarity (runs on CPU vector registers directly)
// Accepts Float32Array (zero-copy) or plain Array
sofuu.ai.similarity(a, b)   // Cosine similarity  → number in [-1, 1]
sofuu.ai.dot(a, b)           // Dot product        → number
sofuu.ai.l2(a, b)            // Euclidean distance → number
```

### `sofuu.mcp` — Model Context Protocol

```js
// Connect your script to any MCP-compatible tool server
const client = await sofuu.mcp.connect("npx @modelcontextprotocol/server-filesystem /tmp");
const tools   = await client.listTools();
const result  = await client.call("read_file", { path: "/tmp/notes.txt" });
client.disconnect();

// Build your own MCP tool server to expose tools to AI agents
const server = sofuu.mcp.serve();

server.tool("get_weather", {
  description: "Returns the weather for a given city.",
  schema: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
}, async ({ city }) => {
  const res  = await fetch(`https://wttr.in/${city}?format=3`);
  return await res.text();
});

server.start(); // Listens over stdio — works with Claude, any MCP client
```

### `sofuu.http` — HTTP Server & Client

```js
// Native HTTP server — no Express or other frameworks needed
const server = sofuu.http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ status: "ok" }));
});
server.listen(8080, "0.0.0.0");

// Standard fetch API
const res  = await fetch("https://api.example.com/data");
const json = await res.json();
```

### `sofuu.fs` — File System

```js
const text    = await sofuu.fs.readFile("./data.json", "utf8");
await sofuu.fs.writeFile("./output.txt", "Hello from Sofuu!");
const entries = await sofuu.fs.readdir("./src");
```

### `sofuu.spawn` — OS Processes

```js
const proc = sofuu.spawn("python3", ["process_data.py"]);
proc.stdout.on("data", (chunk) => console.log(chunk));
await proc.wait();
```

### Globals

```js
// Standard globals available everywhere — no imports needed
setTimeout(() => console.log("done"), 1000);
setInterval(() => console.log("tick"), 500);
await sleep(500);             // Native non-blocking sleep

process.env.API_KEY           // Read environment variables
process.argv                  // CLI argument array
process.cwd()                 // Current working directory
process.exit(0)               // Exit the process

console.log / warn / error    // Full console support
```

---

## Architecture

```
sofuu run agent.ts
       ↓
  [TypeScript stripper]   — pure C, in-process, no subprocess
       ↓
  [QuickJS engine]        — ES2023, native ESM module support
       ↓
  [libuv event loop]      — non-blocking I/O, timers, subprocess
       ↓
  Native C modules:
    ├── sofuu.ai          (LLM streaming, completions, SIMD vector math)
    ├── sofuu.mcp         (MCP client + server, JSON-RPC 2.0 over stdio)
    ├── sofuu.http        (HTTP/HTTPS server + fetch client)
    ├── sofuu.fs          (async file I/O)
    ├── sofuu.spawn       (OS process control)
    └── sofuu.os          (process, env, timers, signals)
```

---

## Build from Source

```bash
# Requirements: clang and make. libcurl is needed for HTTP client features.
git clone https://github.com/sofuu-runtime/sofuu
cd sofuu
make
make install   # copies binary to /usr/local/bin
```

---

## License

Sofuu is licensed under the **PolyForm Noncommercial License 1.0.0**.

This means you can use it freely for personal projects, learning, academic research, and open-source work. Commercial use requires a separate license.

See [LICENSE](LICENSE) for the full terms or contact [hello@sofuu.xyz](mailto:hello@sofuu.xyz) to discuss commercial licensing.

© 2026 Priyanshu Boruah
