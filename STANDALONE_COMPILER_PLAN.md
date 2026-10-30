# ⚡ Sofuu Standalone Compiler Specification & Plan (`sofuu compile`)

**Feature Name:** First-Class Single-Binary Agent Compiler  
**CLI Command:** `sofuu compile <entry.ts|js> [-o <output>]`  
**Goal:** Allow any developer to bundle their entire AI agent (code, tools, QuickJS runtime, libuv event loop, and cognitive vector memory) into a **single, portable, self-executing 5.2 MB binary** with zero dependencies and <4ms boot time.

---

## 1. 🎯 Product Requirements & CLI Experience

### 1.1 Developer Experience (UX)
A developer creates an agent script (`assistant.ts`):
```typescript
import { z } from 'sofuu';

const brain = sofuu.memory.open("knowledge.qtsq");

const agent = sofuu.agent.define({
  model: "claude-3-5-sonnet",
  tools: [sofuu.tools.code],
  policy: { net: ["api.github.com"], maxTokens: 4000 }
});

const task = process.argv[2] || "Summarize repository health";
const result = await agent.run({
  prompt: task,
  context: brain.recall("architecture")
});

console.log(result.output);
```

They compile it with a single command:
```bash
sofuu compile assistant.ts -o dist/my-assistant
```

### 1.2 Execution Experience
The generated binary is completely standalone:
```bash
./dist/my-assistant "Check package vulnerabilities"
```
- **Zero dependencies required on host system** (no Node, no Python, no npm, no Docker).
- **Cold-start boot time:** $\le 4\text{ ms}$.
- **Output binary size:** $\approx 5.2\text{ MB} - 5.5\text{ MB}$ (uncompressed stripped native binary).
- **OS Support:** macOS (ARM64 / Apple Silicon, x86_64) and Linux (x86_64, aarch64 / Raspberry Pi 5).

---

## 2. 🏗️ Binary Architecture & Trailer Specification

### 2.1 The Append-Trailer Architecture
Instead of invoking external C compilers or cross-linkers, `sofuu compile` uses a **high-speed self-mounting executable payload**:

```
+───────────────────────────────────────────────────────────────+
|               Base Sofuu Native Executable                    |
|   (QuickJS + libuv + QTSQ Codec + C/SIMD Vector Math)        |
|   ~5.2 MB (Mach-O on macOS / ELF on Linux)                    |
+───────────────────────────────────────────────────────────────+
|               Embedded Bytecode & Asset Payload               |
|   1. Entrypoint QuickJS Bytecode (compiled AOT)               |
|   2. Bundled ESM Dependencies (resolved via bundler.rs)       |
|   3. Optional Embedded Cognitive Vector Brain (.qtsq)        |
|   4. Asset Manifest (JSON directory table)                    |
+───────────────────────────────────────────────────────────────+
|               Trailer Metadata Block (32 Bytes)               |
|   - Magic Bytes:      b"SOFUU_BIN_V1\0" (12 bytes)           |
|   - Format Version:   u16 (2 bytes)                           |
|   - Payload Flags:    u16 (bitflags: compressed, brain, etc.) |
|   - Payload Length:   u64 (8 bytes, little-endian)            |
|   - CRC32 Checksum:   u32 (4 bytes)                           |
|   - Padding / Reserved: 2 bytes                               |
+───────────────────────────────────────────────────────────────+
```

### 2.2 Boot Fast-Path Detection (< 0.5 ms)
When *any* Sofuu binary starts up:
1. `crates/sofuu-core/src/main.rs` opens its own executable file via `std::env::current_exe()`.
2. It seeks to `file_size - 32` and reads the trailer.
3. **If Magic != `SOFUU_BIN_V1\0`:**
   - Normal CLI execution proceeds (prints help, runs `sofuu run`, `sofuu chat`, etc.).
4. **If Magic == `SOFUU_BIN_V1\0`:**
   - **Standalone Mode Activated!**
   - Seek backward by `payload_length` and memory-map / read the payload block.
   - Verify CRC32 checksum.
   - Initialize QuickJS engine and deserialize bytecode via `JS_ReadObject`.
   - If an embedded `.qtsq` brain is present in payload, mount it automatically under `sofuu.memory.embedded`.
   - Forward `std::env::args()` into `process.argv` and `sofuu.args`.
   - Execute the agent main loop immediately.
   - **Total overhead: Under 0.5 ms.**

---

## 3. 🛠️ Implementation Plan (Phase by Phase)

### Phase 1: Payload Compiler & Trailer Generator (`crates/sofuu-core/src/compiler.rs`)
- **New Module:** `crates/sofuu-core/src/compiler.rs`
- **Responsibilities:**
  1. **Bundle Source:** Call `sofuu_core::bundler::bundle(&entry, &npm_resolve)` to resolve ESM graph into a unified JS string with TypeScript stripped.
  2. **Compile to Bytecode:** Feed bundled source into QuickJS `JS_WriteObject(..., JS_WRITE_OBJ_BYTECODE)` to generate serialized bytecode.
  3. **Package Assets:** Optional embedding of local files or initial `.qtsq` memory databases via `--embed-brain <file.qtsq>`.
  4. **Assemble Output:**
     - Copy the host `sofuu` binary executable to the destination path.
     - Append serialized payload.
     - Append 32-byte `Trailer` struct.
     - Set executable permissions (`0o755` on Unix).

### Phase 2: OS Binary Integrity & Signing
- **Linux (ELF):**
  - Linux handles trailing data after ELF section headers without issue.
  - Test on glibc and musl environments.
- **macOS (Mach-O):**
  - On macOS, appending arbitrary bytes can invalidate default code signatures.
  - Execute automatic ad-hoc re-signing after writing output:
    ```bash
    codesign -s - -f <output_binary>
    ```
  - This ensures Gatekeeper and macOS ARM64 kernel allow immediate zero-warning execution.

### Phase 3: Runtime Boot Deserializer (`crates/sofuu-core/src/main.rs`)
- In `main.rs`, before CLI dispatch:
  ```rust
  if let Some(payload) = compiler::detect_embedded_payload()? {
      return run_embedded_binary(payload);
  }
  ```
- Deserializes bytecode directly into QuickJS runtime, sets up standard globals (`sofuu`, `console`, `process`), and enters the libuv event loop.

### Phase 4: CLI Flags & Options
Add `sofuu compile` command to CLI:
```text
Usage: sofuu compile <entry.ts> [options]

Options:
  -o, --output <file>       Output binary path (default: basename without extension)
  --embed-brain <file.qtsq> Embed an encrypted vector memory container inside the binary
  --no-strip                Keep debug symbols in host binary
  -h, --help                Show help
```

---

## 4. 🧪 Testing & Verification Matrix

| Test Case | Method | Expected Result |
| :--- | :--- | :--- |
| **T1: Hello World Compilation** | `sofuu compile examples/hello.js -o dist/hello` | Standalone binary `./dist/hello` runs, prints output, exits 0. |
| **T2: TypeScript & Async Agent** | Compile an agent script with async tool calls and fetch. | Runs identically to `sofuu run agent.ts`. |
| **T3: Embedded QTSQ Brain** | `sofuu compile agent.ts --embed-brain test.qtsq -o dist/bot` | Embedded vector searches recall correct facts without external file dependencies. |
| **T4: Binary Size Audit** | Run `ls -lh dist/my-agent` across platforms. | Binary size must remain $\le 5.5\text{ MB}$. |
| **T5: Cold Start Benchmark** | Benchmark startup time via `hyperfine ./dist/my-agent`. | Cold start latency under $4.0\text{ ms}$. |

---

## 5. 🚀 Viral Marketing & Strategic Acquisition Playbook

Once `sofuu compile` is tested and verified, we launch it publicly:

### 1. Show HN Headline
> **Show HN: Sofuu – Compile any TypeScript AI agent into a 5MB self-contained executable**

### 2. The 15-Second Viral Video (X & LinkedIn)
- **Second 0–3:** Write a 15-line TypeScript agent with local vector memory.
- **Second 4–7:** Run `sofuu compile agent.ts -o agent`.
- **Second 8–11:** Inspect binary: `ls -lh agent` (5.2 MB) and `file agent` (standalone executable).
- **Second 12–15:** Run `./agent "task"` in 3ms.

### 3. The Big Tech Acquisition Thesis
- **Cloudflare & Vercel** cannot provide single-binary deployment for serverless / edge agents.
- **OpenAI & Anthropic** need a standardized, sandboxed, single-file distribution runtime for on-device agent tools.
- Sofuu becomes the **universal compiler and packaging layer** for all production AI agents.
