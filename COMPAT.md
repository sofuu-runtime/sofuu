# Sofuu Runtime Compatibility Matrix

*Last audited: October 2026 (v0.3.0-beta)*

Sofuu is an AI-native runtime designed for deterministic agent execution, instant startup (<4 ms), and self-contained deployment. Rather than emulating the entirety of Node.js's legacy 15-year surface, Sofuu provides a **strict, high-performance compatibility layer** covering modern Web standards, essential Node.js built-ins, and the top npm packages used by modern AI agents and LLM tool chains.

---

## 1. Web Standard APIs Matrix

Sofuu implements the WHATWG and W3C standard interfaces used by modern SDKs and fetch-based clients.

| Web API / Global | Support Status | Notes / Capabilities |
| :--- | :---: | :--- |
| `fetch()` | **Full** | Native HTTP/1.1 and HTTP/2 client backed by native connection pooling; supports streaming bodies and SSE. |
| `Headers` | **Full** | Case-insensitive header maps, iteration (`entries`, `keys`, `values`), and `append`/`set`/`get`/`delete`. |
| `Request` | **Full** | WHATWG Request object supporting `.json()`, `.text()`, headers, and method parsing. |
| `Response` | **Full** | `.json()`, `.text()`, `.arrayBuffer()`, static `Response.json(data, init)`, and a stable `.body` that is a real `ReadableStream`. |
| `ReadableStream` | **Full** | `getReader()` (`read`, `releaseLock`, `cancel`), `locked`, `cancel`, `pipeTo`, `pipeThrough`, `tee`, `Symbol.asyncIterator`. Chunks are `Uint8Array`. |
| `WritableStream` | **Full (core)** | `getWriter()` (`write`, `close`, `abort`, `ready`, `closed`), serialised writes with backpressure signalled through `ready`. |
| `TransformStream` | **Full (core)** | `readable` / `writable` pair with `transform`, `flush`, `start`, and a controller (`enqueue`, `error`, `terminate`). |
| `TextDecoderStream` | **Full** | Incremental UTF-8 decoding that keeps split multi-byte characters intact across chunk boundaries. |
| `TextEncoderStream` | **Full** | String chunks to UTF-8 byte chunks. |
| `URL` | **Full** | Complete WHATWG URL parser supporting `protocol`, `hostname`, `port`, `pathname`, `search`, `hash`, `origin`, and credentials. |
| `URLSearchParams` | **Full** | Bi-directional synchronization with `url.search` and `url.href`; supports `get`, `getAll`, `has`, `set`, `append`, `delete`, and iterators. |
| `TextEncoder` | **Full** | Spec-compliant UTF-8 encoder (`encode()` $\to$ `Uint8Array`, `encodeInto()`). |
| `TextDecoder` | **Full** | Multi-byte UTF-8 streaming and buffer decoder (`decode()`). |
| `atob()` / `btoa()` | **Full** | RFC 4648 Base64 encoding and decoding for Latin1 and binary payloads. |
| `crypto.randomUUID()` | **Full** | Cryptographically secure RFC 4122 v4 UUID generator backed by OS `/dev/urandom` / `getentropy`. |
| `crypto.getRandomValues()` | **Full** | Fills any `TypedArray` in-place with cryptographically secure native random bytes. |
| `crypto.subtle.digest()` | **Partial** | Hardware-accelerated `SHA-256` digest returning `Promise<ArrayBuffer>` for SDK request signing. |
| `AbortController` | **Full** | Native cancellation primitive across `fetch()`, timers, and agent run loops. |
| `AbortSignal` | **Full** | Supports `signal.aborted`, `signal.reason`, `AbortSignal.timeout(ms)`. |
| `queueMicrotask()` | **Full** | Schedules callbacks on the microtask queue before the next event loop turn. |
| `setTimeout` / `setInterval` | **Full** | High-precision timers backed by libuv event loop. |
| `clearTimeout` / `clearInterval`| **Full** | Resource handle release. |

---

## 2. Node.js Built-in Modules Matrix

Sofuu intercepts both CommonJS (`require('...')`) and ES Module (`import ... from '...'` / `import ... from 'node:...'`) specifiers through in-process virtual module registries with zero disk or serialization overhead.

| Module Specifier | Support Status | Covered APIs |
| :--- | :---: | :--- |
| `path` / `node:path` | **Full** | `join`, `resolve`, `normalize`, `isAbsolute`, `relative`, `dirname`, `basename`, `extname`, `parse`, `format`, `sep` (`/`), `delimiter` (`:`), `posix`, `win32`. |
| `events` / `node:events` | **Full** | `EventEmitter` (`on`, `once`, `off`, `emit`, `addListener`, `removeListener`, `removeAllListeners`, `listeners`, `listenerCount`, `setMaxListeners`, `getMaxListeners`, `eventNames`), `getEventListeners`, `once()`. |
| `crypto` / `node:crypto` | **Full** | `randomUUID`, `getRandomValues`, `randomBytes`, `createHash('sha256')`, `createHmac('sha256')`, `subtle`. |
| `buffer` / `node:buffer` | **Full** | `Buffer` (subclass of `Uint8Array`), `Buffer.isBuffer`, `Buffer.from` (utf8, hex, base64), `Buffer.alloc`, `Buffer.allocUnsafe`, `Buffer.concat`, `Buffer.byteLength`, `.toString('utf8'\|'hex'\|'base64')`, `.slice`, `.subarray`, `.copy`, `.equals`. |
| `fs` / `node:fs` | **Full (Core)** | Synchronous and Promise-based operations: `readFileSync`, `writeFileSync`, `existsSync`, `readdirSync`, `mkdirSync`, `unlinkSync`, `statSync` (`isFile()`, `isDirectory()`, `size`, `mtimeMs`), `promises.*`. All calls enforce Sofuu policy jail checks. |
| `os` / `node:os` | **Full** | `platform()`, `arch()`, `homedir()`, `tmpdir()`, `hostname()`, `release()`, `type()`, `cpus()`, `totalmem()`, `freemem()`, `EOL`. |
| `util` / `node:util` | **Full** | `promisify()`, `inspect()`, `format()`, `types.isUint8Array()`, `types.isPromise()`, `inherits()`, `TextEncoder`, `TextDecoder`. |
| `stream` / `node:stream` | **Partial** | `Readable`, `Writable`, `Transform`, `PassThrough`, `pipeline()`, `finished()`. |
| `assert` / `node:assert` | **Full** | `assert()`, `ok()`, `equal()`, `strictEqual()`, `deepEqual()`, `notEqual()`, `notStrictEqual()`. |
| `url` / `node:url` | **Full** | `URL`, `URLSearchParams`. |
| `async_hooks` / `node:async_hooks` | **Full** | `AsyncLocalStorage` backed by Sofuu's native context propagation engine (`sofuu.context`). |
| `process` / `node:process` | **Full** | `process.nextTick()`, `process.env`, `process.cwd()`, `process.versions` (`node`, `sofuu`, `quickjs`), `process.platform`, `process.arch`, `process.exit()`, `process.argv`. |
| `http` / `https` | **Partial** | Client requests routed through native HTTP engine; edge server functions available via `sofuu serve` and `sofuu.http.createServer()`. |
| `child_process` | **Intentional gap** | Not provided. `require('child_process')` throws `ERR_SOFUU_UNSUPPORTED_NODE_API` and points at `sofuu.spawn` / `sofuu.exec` (deterministic permission controls and timeout caps). |
| `worker_threads` / `cluster` | **Intentional gap** | `ERR_SOFUU_UNSUPPORTED_NODE_API`. Sofuu runs one address space: use `scheduler.yield()`, `sofuu.swarm`, or an agent mesh. |
| `v8` / `inspector` | **Intentional gap** | `ERR_SOFUU_UNSUPPORTED_NODE_API` — there is no V8 inside a zero-JIT engine. |
| `net` / `tls` | **Intentional Gap** | Direct raw TCP socket manipulation is restricted in favor of `sofuu.http`, `fetch`, and WebSockets for security containment. |

### Package resolution (`node_modules`)

Resolution follows the Node.js / WinterCG subpath contract:

- `package.json#exports` is honoured for the root (`"."`) and for subpaths, with
  exact keys, `./*` wildcard patterns (longest prefix wins), string, array and
  nested-condition targets.
- Condition order is `import` → `module` → `default` → `node` for `import` /
  `import()`, and `require` → `module` → `default` → `node` → `import` for
  `require()` — so a dual-published package gives each loader its own build.
- Without `exports`, the legacy chain is `module` → `main` → `index.js` /
  `index.mjs`, and a deep import (`pkg/lib/x.js`) still resolves.
- An `exports` target may not escape its own package (`..`, absolute paths and
  URLs are refused).
- Failures are diagnosed rather than guessed: a missing package says
  `run \`sofuu add <pkg>\``, a missing subpath lists the published ones, a
  `.node` entry point explains the zero-JIT limit, and `require()` of an ES
  module returns `ERR_SOFUU_REQUIRE_ESM`.

---

## 3. Top AI & Server Packages Support Status

Sofuu audits and tests packages commonly found in agent stacks and modern microservices:

| Package | Tested Version | Status | Notes |
| :--- | :---: | :---: | :--- |
| `zod` | `^3.23` / `^4.0` | **PASS** | Pure JS schema validation; powers tool definitions and JSON-schema conversions. |
| `openai` | `^4.50` | **PASS** | Standard chat completion, tool calling, and streaming loops run unmodified. |
| `@anthropic-ai/sdk` | `^0.26` | **PASS** | Messages API, computer-use tool loops, and streaming SSE responses run unmodified. |
| `@modelcontextprotocol/sdk` | `^0.6` | **PASS** | Official MCP TypeScript client and server SDK operates over stdio or SSE transport. |
| `dotenv` | `^16.4` | **PASS** | Synchronous `.env` parsing and `process.env` population. |
| `uuid` | `^9.0` / `^10.0` | **PASS** | Uses `crypto.randomUUID()` and `crypto.getRandomValues()` directly. |
| `hono` | `^4.5` | **PASS** | Universal web framework runs with zero modifications. |
| `nanoid` | `^5.0` | **PASS** | Fast ID generator using `crypto.getRandomValues()`. |
| `mustache` | `^4.2` | **PASS** | Template rendering for dynamic system prompts. |
| `picocolors` | `^1.1` | **PASS** | Terminal formatting and ANSI styling. |
| `yaml` | `^2.4` | **PASS** | YAML configuration parsing for OpenAPI schemas and agent specs. |

---

## 4. TypeScript Support: Zero-Build Ingest

Sofuu includes a **built-in, native TypeScript AST stripper**. Developers do not need `tsc`, `esbuild`, `tsx`, or `ts-node` to run TypeScript files:

```bash
# Execute TypeScript directly with zero compilation step
sofuu run src/agent.ts
```

### How It Works
1. When a `.ts` file is loaded (either as main entry or via `import`), Sofuu strips type annotations, interfaces, enums, and type assertions in a single-pass native pass.
2. Type checking occurs inside your IDE (VS Code, Cursor, Neovim) using TypeScript language server (`tsserver`) against `sofuu.d.ts`.
3. Execution commences immediately in QuickJS with `<4 ms` cold start.

### Recommended `tsconfig.json`
To set up IDE autocomplete and type checking, create a `tsconfig.json` based on the shipped template:

```json
{
  "compilerOptions": {
    "target": "ES2023",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2023"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "types": ["./sofuu.d.ts"]
  },
  "include": ["**/*.ts", "sofuu.d.ts"]
}
```

---

## 5. Migration Guide: From Node.js to Sofuu

Migrating a Node.js AI agent script to Sofuu requires zero code changes in 95% of cases.

### Example: OpenAI Tool-Calling Agent

#### Node.js (`agent.js` with npm dependencies):
```javascript
import OpenAI from "openai";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const openai = new OpenAI();
const traceId = crypto.randomUUID();
const config = JSON.parse(fs.readFileSync(path.join(process.cwd(), "config.json"), "utf8"));

async function main() {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Summarize today's logs." }],
  });
  console.log(`[Trace: ${traceId}]`, completion.choices[0].message.content);
}
main();
```

#### Running in Sofuu:
```bash
# Run unmodified!
sofuu run agent.js
```

#### Optional: Enhancing with Sofuu Native Capabilities
Once running on Sofuu, you can take advantage of Sofuu's in-process vector memory and native agent sandboxing without introducing extra external dependencies:

```typescript
import OpenAI from "openai";
import { defineTool, z } from "sofuu";

// Store conversation embeddings directly in-process with zero Docker daemon
const memory = sofuu.memory.open("agent_brain.qtsq", 384);
await memory.remember("User requested log summaries for Oct 30");

// Expose as an MCP tool with schema validation
export default defineTool({
  name: "recall_memory",
  description: "Search local episodic memory",
  schema: z.object({ query: z.string() }),
  async run({ query }) {
    return await memory.recall(query, 3);
  }
});
```

---

## 6. Summary

- **No Rewrite Required**: AI SDKs (`openai`, `@anthropic-ai/sdk`, `@modelcontextprotocol/sdk`) run out-of-the-box.
- **Node Built-in Shims**: `path`, `events`, `crypto`, `buffer`, `fs`, `os`, `util`, `stream`, `assert`, `url`, and `async_hooks` are fully supported for agent workloads.
- **TypeScript Native**: Run `.ts` scripts directly with sub-4ms cold start.
- **Embedded & Sovereign**: Bundle scripts into a 5.2 MB standalone binary via `sofuu compile`.
