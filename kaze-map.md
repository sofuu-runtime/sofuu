# KAZE ARCHITECTURAL MAP & DIVISION REGISTRY

> Master blueprint for compartmentalization, ownership, self-hosting, and systems-level verification across the Kaze Language and Runtime (`v1.0`).
> This document is the single source of truth (SSOT) for division boundaries, owned files, testing harnesses, and implementation roadmaps.

---

## 1. Global Invariants & Non-Negotiable Rules

Every division of Kaze operates under uncompromising architectural boundaries to guarantee that changes in one domain never cascade, degrade safety, or introduce runtime panics:

1. **Zero Runtime Panics & Fail-Safe Model**:
   - The Kaze compiler and runtime are architected under a total-correctness paradigm. A program that compiles cleanly must never suffer uncontrolled crashes, panic aborts, null-pointer dereferences, or buffer overflows during execution.
   - All fallible operations (integer division, modulo, array indexing, dictionary lookups, external IO) require explicit fallback handling via `otherwise`, range contracts, or bounded types.
2. **Deterministic ANSI C99 Code Generation & Zero GC**:
   - The native code generator emits ultra-fast, zero-overhead ANSI C99 code competing directly with hand-written C and C++ in latency and memory footprint.
   - Memory management relies strictly on stack allocation and linear bump arenas (`dist/kaze_rt.h`, `src/arena.rs`). There is zero garbage collection, zero pause times, and zero reference counting overhead.
3. **Cross-Target Architecture Parity (32-Bit & 64-Bit)**:
   - Emitted C99 code maintains strict portability across both 32-bit embedded platforms (ARM Cortex-M0/M4/M7, RISC-V RV32IM, ESP32) and 64-bit platforms (x86_64, aarch64 Apple Silicon / Linux).
   - Word size and pointer widths are abstract (`kaze_int_t`, `kaze_uint_t`, `kaze_float_t`); integer overflows are checked at compile time or explicitly trapped via registered handlers.
4. **Pure Language Core Generalization & Zero Hardcoding**:
   - The compiler, lexer, parser, and runtime core remain 100% domain-agnostic (`RULES.md`, `AGENTS.md`).
   - Never hardcode domain parameter names (e.g., `entry_price`, `heart_rate`, `sentiment_score`, `altitude`) or external domain endpoints into the compiler or bootstrap runtime.
   - Standalone routines support a 12-factor configuration hierarchy: CLI Flags (`--arg`) > Environment Variables (`AGENT_*`/`SOFUU_*`) > Configuration File (`--config`) > Routine Signatures.
5. **Pure Kaze Source (Zero Auxiliary Wrappers)**:
   - Standalone agent and system routines are 100% pure Kaze (`*.kaze`). Auxiliary `.js` sidecars or `.sh` script wrappers are strictly prohibited.
6. **Zero-Ambient Authority & Fail-Closed Capabilities**:
   - Programs possess zero implicit ambient privileges. Every external interaction (filesystem, network, sensors, actuators, memory, AI model inference) requires explicit `needs: <capability>` declaration.
   - Any capability request lacking a grant or configured host backend must fail closed (`E_CAP_DENIED` / `E_CAP_MISSING`). A granted capability without a configured backend may never fabricate success (`RULES.md` §1.9).
7. **Dual-Syntax Equivalence & Parity**:
   - Human-oriented natural sentence syntax and machine-oriented tagged prefix-tree S-expressions `(kaze (format 1) ...)` share identical AST semantics, resource bounds, and authorization.
   - Bidirectional transpilation (`kaze to-machine` and `kaze to-human`) is lossless and verified by AST round-trip equivalence across the entire codebase.
8. **No Fabricated Execution & Local Weights Only**:
   - No simulated, mocked, stubbed, or fabricated AI, tool, memory, alert, or network responses in production code or test gates (`RULES.md` §1.9).
   - Test and benchmark suites run exclusively against pre-existing local GGUF models (`~/Downloads/*.gguf`). Zero remote model downloads are permitted.
9. **Zero Decorative Emojis**:
   - Documentation, compiler diagnostic messages, CLI outputs, and log streams must remain strictly professional without decorative emojis.

---

## 2. Division Index & Matrix

> Detailed specifications for every division reside in [`docs/kaze-map/`](docs/kaze-map/README.md).

| Division | Code | Name | Focus & Scope | Primary Locations | Maturity | Verification Command | Detailed Specification |
|---|---|---|---|---|---|---|---|
| **K01** | `ISA` | **Language Core, Bytecode ISA & Instructions** | 49 unified opcodes, format 5 bytecode, AST nodes, tagged values, serialization | `kaze/src/isa.rs`, `kaze/src/ast.rs`, `kaze/src/value.rs`, `kaze/compiler/token.kaze`, `kaze/compiler/ast.kaze` | Core | `cargo test --manifest-path kaze/Cargo.toml --lib isa` | [K01-language-core-isa.md](docs/kaze-map/K01-language-core-isa.md) |
| **K02** | `SYNTAX` | **Lexer & Dual Syntax (Human / Machine)** | Off-side indentation tokenizer, human sentences, machine S-exprs, bidirectional transpiler | `kaze/src/lexer.rs`, `kaze/src/parser.rs`, `kaze/src/machine.rs`, `kaze/compiler/lexer.kaze`, `kaze/compiler/parser.kaze` | Core | `cargo test --manifest-path kaze/Cargo.toml --lib machine` | [K02-lexer-dual-syntax.md](docs/kaze-map/K02-lexer-dual-syntax.md) |
| **K03** | `CHECK` | **Static Type Checker, Units & Contracts** | Zero-panic type system, dimensional physics algebra, pre/post range contracts, diagnostics | `kaze/src/checker.rs`, `kaze/src/strict.rs`, `kaze/src/diag.rs`, `kaze/compiler/checker.kaze` | Core | `cargo test --manifest-path kaze/Cargo.toml --test strict_semantics --test units --test contracts` | [K03-type-checker-units-contracts.md](docs/kaze-map/K03-type-checker-units-contracts.md) |
| **K04** | `COMPILER` | **Self-Hosted Compiler & Native C99 Code Gen** | Stage 1/2/3 multi-stage bootstrap, self-hosting (`codegen.kaze`), portable ANSI C99 generator, 32/64-bit | `kaze/src/compiler.rs`, `kaze/src/c_codegen.rs`, `kaze/compiler/codegen.kaze`, `kaze/compiler/kaze_self.kaze` | Core / Verified Bootstrap | `cargo test --manifest-path kaze/Cargo.toml --test emitted_c` | [K04-self-hosted-compiler-c99.md](docs/kaze-map/K04-self-hosted-compiler-c99.md) |
| **K05** | `VM` | **Virtual Machine, Linear Arena & Memory** | Bytecode interpreter, zero-GC bump arena, fuel quotas, declared and hard recursion ceilings ($\le 256$), deadlines | `kaze/src/vm.rs`, `kaze/src/arena.rs`, `kaze/src/deadline.rs`, `kaze/dist/kaze_rt.h` | Core | `cargo test --manifest-path kaze/Cargo.toml --lib vm --test rt_runtime` | [K05-vm-runtime-arena.md](docs/kaze-map/K05-vm-runtime-arena.md) |
| **K06** | `CAPS` | **Capability Gates, Security Policy & Sandbox** | Zero-ambient authority, monotonic `needs:`, fail-closed policy gates, real alert sinks, audit logs | `kaze/src/caps.rs`, `kaze/src/host.rs` | Core | `cargo test --manifest-path kaze/Cargo.toml --test capability_safety --test effector_safety` | [K06-capability-security-policy.md](docs/kaze-map/K06-capability-security-policy.md) |
| **K07** | `ROBOT` | **Robotics, Effector Drivers & Telemetry** | MAVLink 2.0 flight codec, GPIO pulse timers, PWM servos, geodesy math, 1500ms safety watchdog | `kaze/src/hardware.rs`, `kaze/src/mavlink.rs`, `kaze/src/gpio.rs`, `kaze/src/pwm.rs`, `kaze/src/geo.rs`, `kaze/src/link.rs` | Core / Gated Live Actuation | `cargo test --manifest-path kaze/Cargo.toml --test live` | [K07-robotics-hardware-telemetry.md](docs/kaze-map/K07-robotics-hardware-telemetry.md) |
| **K08** | `TOOLING` | **Developer Tooling, LSP & CLI Experience** | CLI commands (`run`, `check`, `fmt`, `emit-c`), native LSP server, opinionated code formatter | `kaze/src/main.rs`, `kaze/src/lsp.rs`, `kaze/src/fmt.rs`, `kaze/editors/` | Core | `cargo test --manifest-path kaze/Cargo.toml --lib fmt --test lsp` | [K08-tooling-lsp-cli.md](docs/kaze-map/K08-tooling-lsp-cli.md) |
| **K09** | `BRIDGE` | **Embedded Sofuu Bridge & C-FFI Integrations** | Embedded runtime mirror, async continuation bridge, standalone binary embedding (`sofuu compile`) | `crates/sofuu-core/src/domains/kaze/`, `crates/sofuu-core/src/script_cmd.rs`, `crates/sofuu-capi/` | Core | `cargo test -p sofuu-core domains::kaze` | [K09-embedded-sofuu-bridge.md](docs/kaze-map/K09-embedded-sofuu-bridge.md) |
| **K10** | `VERIFY` | **Verification, Conformance & Test Suite** | Strict cross-engine drift checker, Stage 2/3 fixed-point bootstrap verifier, conformance tests | `kaze/tests/`, `scripts/kaze-drift-check.sh`, `scripts/kaze-sync.sh`, `kaze/scripts/` | Core | `bash scripts/kaze-drift-check.sh --strict` | [K10-verification-conformance.md](docs/kaze-map/K10-verification-conformance.md) |

---

## 3. Detailed Division Specifications

### Division 01: Language Core, Bytecode ISA & Instructions (K01)
> Detailed Specification: [`docs/kaze-map/K01-language-core-isa.md`](docs/kaze-map/K01-language-core-isa.md)
* **Mission**: Provide the deterministic, low-level instruction set architecture (ISA), runtime value representations, and AST node specifications forming the bedrock for both the bytecode VM and native C99 emission.
* **Owned Files**:
  * `kaze/src/isa.rs` (Unified 49-opcode contract, opcode metadata, bytecode format 5 reader/writer)
  * `kaze/src/ast.rs` (Abstract syntax tree node data structures, source span mappings, record shapes)
  * `kaze/src/value.rs` (Tagged union value representations: ints, floats, strings, vectors, maps, points)
  * `kaze/compiler/token.kaze` (Self-hosted token type enumerations with byte offsets)
  * `kaze/compiler/ast.kaze` (Self-hosted AST representations in pure Kaze)
  * `crates/sofuu-core/src/domains/kaze/isa.rs` & `ast.rs` (Embedded runtime mirror)
* **Architectural Boundaries**:
  * Zero opcode drift: Opcode numbering and semantics match 100% across standalone and embedded engines (`scripts/kaze-drift-check.sh --strict`).
  * Bytecode format version 5: Relocatable binary encoding with per-item recursion ceilings, statements, source spans, and debug symbols.
  * Value immutability: Routine values are immutable by default; mutable references are strictly localized.
* **Active Tasks**:
  - [x] Unify 49 opcodes and 17 language verbs across standalone and embedded runtime.
  - [x] Standardize Bytecode Format 5 with per-item recursion ceilings and statement-level debug maps.
  - [x] Enforce zero-drift opcode verification via `scripts/kaze-drift-check.sh --strict`.
  - [x] Implement point value representation (`KZ_VAL_POINT`) with geodesy operations.
  - [ ] Implement constant-pool deduplication for binary footprint minimization.

---

### Division 02: Lexer & Dual Syntax (Human / Machine) (K02)
> Detailed Specification: [`docs/kaze-map/K02-lexer-dual-syntax.md`](docs/kaze-map/K02-lexer-dual-syntax.md)
* **Mission**: Deliver dual syntactic representations—natural human-oriented phrasing (`routine <name>:`) and unambiguous machine-oriented S-expressions `(kaze (format 1) ...)`—with 100% AST round-trip parity.
* **Owned Files**:
  * `kaze/src/lexer.rs` (Indentation-aware tokenizer, off-side rule handler, line/column span tracking)
  * `kaze/src/parser.rs` (Recursive-descent parser, capability clause validation)
  * `kaze/src/machine.rs` (Tagged prefix-tree S-expression parser, serializer, and round-trip verifier)
  * `kaze/compiler/lexer.kaze` (Self-hosted tokenizer in pure Kaze with 256-token chunking)
  * `kaze/compiler/parser.kaze` (Self-hosted recursive-descent parser in pure Kaze)
  * `kaze/compiler/machine.kaze` (Pure-Kaze dual-syntax transpiler)
* **Architectural Boundaries**:
  * Auto-detection: Parser auto-detects syntax style without requiring explicit flags.
  * Bidirectional transpilation: Semantic round-trip invariant $\text{AST}(H) \equiv \text{AST}(\text{to\_human}(\text{to\_machine}(H)))$ verified across 100+ sources.
  * Diagnostics: Exact line, column, and byte offset span preservation for caret error rendering.
* **Active Tasks**:
  - [x] Implement machine syntax tagged prefix-tree format `(kaze (format 1) ...)`.
  - [x] Implement auto-detection and bidirectional transpilation (`to-machine`, `to-human`).
  - [x] Replicate lexer, parser, and machine dual-syntax in self-hosted pure Kaze.
  - [x] Optimize lexer memory usage via 256-token buffer chunking (reducing peak RSS from 7.5 GB to 97 MB).
  - [ ] Add incremental re-lexing for language server keystroke latency minimization.

---

### Division 03: Static Type Checker, Physical Units & Range Contracts (K03)
> Detailed Specification: [`docs/kaze-map/K03-type-checker-units-contracts.md`](docs/kaze-map/K03-type-checker-units-contracts.md)
* **Mission**: Guarantee compile-time total correctness through static type inference, compile-time physical unit algebra, pre/post-condition routine contracts, and zero-panic boundary guarantees.
* **Owned Files**:
  * `kaze/src/checker.rs` (Semantic type checker, capability grant analyzer, unit dimension solver)
  * `kaze/src/strict.rs` (Zero-panic enforcer: mandatory `otherwise`, loop bounds, recursion cycles)
  * `kaze/src/diag.rs` (Structured visual caret error renderer)
  * `kaze/compiler/checker.kaze` (Self-hosted type checker and strictness verifier in pure Kaze)
* **Architectural Boundaries**:
  * Zero unhandled errors: Division, indexing, and IO must have verified bounds or mandatory `otherwise` fallbacks (`E_KAZE_MISSING_FALLBACK`).
  * Physical unit algebra: Dimensions (`meter`, `second`, `rad`, `deg`, `Celsius`, `mbar`, etc.) are checked at compile time; unit mismatches fail compilation (`E_KAZE_UNIT_MISMATCH`) with zero runtime instruction overhead.
  * Routine contracts: Preconditions (`requires`) and postconditions (`ensures`) checked statically or asserted at runtime, with elision when statically proven.
* **Active Tasks**:
  - [x] Static validation of mandatory capability declarations (`needs: ...`).
  - [x] Structured diagnostic errors with masked carets (`E_KAZE_*`).
  - [x] Enforce mandatory `otherwise` fallbacks across 72 test fixtures (`strict_semantics.rs`).
  - [x] Implement dimensional unit algebra across 22 unit spellings with zero runtime cost (`units.rs`).
  - [x] Implement routine contracts (`requires`/`ensures`) and untrusted input barriers (`contracts.rs`).
  - [x] Author pure-Kaze semantic checker (`compiler/checker.kaze`).

---

### Division 04: Self-Hosted Compiler & Native C99 Code Generator (K04)
> Detailed Specification: [`docs/kaze-map/K04-self-hosted-compiler-c99.md`](docs/kaze-map/K04-self-hosted-compiler-c99.md)
* **Mission**: Compile Kaze programs to portable, zero-overhead ANSI C99 native code rivaling C and C++ in execution speed, and establish complete self-hosting where Kaze compiles its own compiler.
* **Owned Files**:
  * `kaze/src/compiler.rs` (Bytecode compiler emitting format 5 binary modules)
  * `kaze/src/c_codegen.rs` (Rust Stage 0 native ANSI C99 code generator)
  * `kaze/compiler/codegen.kaze` (Self-hosted C99 and bytecode code generator in pure Kaze)
  * `kaze/compiler/kaze_self.kaze` (Unified self-hosted compiler monolith in pure Kaze)
  * `kaze/compiler/main.kaze` (Self-hosted compiler CLI driver in pure Kaze)
* **Architectural Boundaries**:
  * ANSI C99 portability: Generated C code compiles warning-free under `clang -Wall -Wextra -Werror -pedantic -std=c99`.
  * Fixed-point bootstrap: Stage 2 compiler binary compiles `kaze_self.kaze` to Stage 3 C source with zero byte differences ($\text{sha256}(s2.c) == \text{sha256}(s3.c)$).
  * 32-bit & 64-bit cross-target support: Pointer widths and integer representations scale cleanly from microcontrollers (Cortex-M) to 64-bit servers.
* **Active Tasks**:
  - [x] Implement Rust-based ANSI C99 native emitter (`kaze/src/c_codegen.rs`).
  - [x] Author pure-Kaze self-hosted compiler frontend and code generator (`kaze/compiler/`).
  - [x] Multi-stage bootstrap verified: Stage 1 compiles Stage 2, Stage 2 compiles Stage 3, bitwise identical output (`diff s2.c s3.c` is 0 bytes).
  - [x] Hook execution in native C backend (`when`/`every` compiled to `kz_run_hooks`).
  - [x] Four-tier 12-factor configuration resolution and strongly typed parameter coercion in compiled C binaries.
  - [ ] Phase 7: Retarget `compiler/codegen.kaze` to emit calls directly to `dist/kaze_rt.h` arena allocations.

---

### Division 05: Virtual Machine, Linear Arena Allocator & Fuel Budgeting (K05)
> Detailed Specification: [`docs/kaze-map/K05-vm-runtime-arena.md`](docs/kaze-map/K05-vm-runtime-arena.md)
* **Mission**: Execute Kaze bytecode with real-time determinism, microsecond memory allocation, zero garbage collection pauses, and strict resource quotas.
* **Owned Files**:
  * `kaze/src/vm.rs` (Bytecode interpreter, register stack, instruction execution loop)
  * `kaze/src/arena.rs` (Zero-GC linear bump allocator, bulk reset mechanics)
  * `kaze/dist/kaze_rt.h` & `kaze_rt.c` (Standalone portable C99 linear bump arena substrate)
  * `kaze/src/deadline.rs` (Wall-clock real-time execution deadline watchdogs)
* **Architectural Boundaries**:
  * Zero-GC: Memory allocations take place in linear bump arenas; entire routine working sets are reclaimed in bulk upon routine completion.
  * Recursion ceiling: Declared per-routine ceilings and hard recursion ceiling ($\le 256$ frames) strictly enforced.
  * Fuel budgeting: Every opcode consumes deterministic fuel units; execution cleanly halts with `E_KAZE_FUEL_EXHAUSTED` upon exhaustion.
* **Active Tasks**:
  - [x] Implement linear bump arena memory management with zero-leak bulk reset (`kaze_rt.h`, `kaze_rt.c`).
  - [x] Enforce declared and hard call recursion depth limit ($\le 256$ frames) in VM and native C runtime.
  - [x] Implement modulo and division-by-zero traps (`E_KAZE_DIV0`).
  - [x] Pass 1,000,000 arena allocate and reset cycles under ASan/UBSan without leaks (`rt-check.sh`).
  - [ ] Add configurable per-cycle fuel budgets and time-slice preemption.

---

### Division 06: Capability Gates, Security Policy & Sandbox (K06)
> Detailed Specification: [`docs/kaze-map/K06-capability-security-policy.md`](docs/kaze-map/K06-capability-security-policy.md)
* **Mission**: Enforce zero-ambient authority security across filesystem, network, memory, hardware effectors, and external AI models.
* **Owned Files**:
  * `kaze/src/caps.rs` (Capability declarations, policy arbiter, permission tokens)
  * `kaze/src/host.rs` (Capability host dispatch, fail-closed handlers, alert sinks)
  * `kaze/docs/CAPABILITY-SECURITY.md` (Formal capability security contract)
* **Architectural Boundaries**:
  * Fail-closed by default: Missing, empty, or unconfigured policies trigger immediate `E_CAP_DENIED` or `E_CAP_MISSING`. Zero silent success.
  * Monotonic capability delegation: Sub-routines can only inherit a subset of parent capabilities; escalation is impossible.
  * Real alert sinks: Output to `stdout`, `stderr`, or `file:<path>` enforces length bounds ($\le 1024$ bytes) and redacts credentials without fabricated mock success.
* **Active Tasks**:
  - [x] Formalize capability declaration syntax (`needs: ...`).
  - [x] Implement real alert sinks with RFC3339 timestamps and credential redaction.
  - [x] Implement fail-closed capability denial suite (`tests/capability_safety.rs`, `tests/effector_safety.rs`).
  - [x] Audited gate decisions wrapper with fail-closed denial on unwritable audit sinks.
  - [x] Eliminate mock success in compiled native C binaries (`kz_cap_call` fails closed with `E_CAP_MISSING` and exits 70).

---

### Division 07: Robotics, Effector Drivers & Telemetry Watchdogs (K07)
> Detailed Specification: [`docs/kaze-map/K07-robotics-hardware-telemetry.md`](docs/kaze-map/K07-robotics-hardware-telemetry.md)
* **Mission**: Provide fail-closed physical effector control, MAVLink 2.0 flight telemetry, Linux GPIO pulse timing, and geodesy math for autonomous physical edge systems.
* **Owned Files**:
  * `kaze/src/hardware.rs` (Hardware policy gates, fail-closed safety monitors)
  * `kaze/src/mavlink.rs` (MAVLink 2.0 protocol encoder and decoder)
  * `kaze/src/gpio.rs` & `kaze/src/gpio_chardev.rs` (Linux GPIO pin controller and pulse timers)
  * `kaze/src/pwm.rs` (PWM servo and throttle actuator interface)
  * `kaze/src/geo.rs` (Geodesy math, distance, bearing, and geofence polygon checks)
  * `kaze/src/link.rs` (Physical communication link monitors, 1500 ms heartbeat watchdog)
* **Architectural Boundaries**:
  * 1500 ms safety gate: Unified in `link.rs`; telemetry older than $1500\text{ ms}$ triggers uncatchable `E_SAFETY_VIOLATION`.
  * De-assert on termination: Actuator lines register cleanup handlers to de-assert on exit or abnormal termination (`SIGINT`/`SIGTERM`).
  * Simulator labelling: Simulated execution (`drone=sim`) is explicitly selected and labelled in all results.
* **Active Tasks**:
  - [x] Implement MAVLink 2.0 message codec.
  - [x] Unify 1500 ms telemetry freshness watchdog with per-field reception stamps.
  - [x] Implement GPIO chardev pulse timers with async-signal-safe de-assert cleanup handlers (`gpio.rs`).
  - [x] Parity in spatial math and point arithmetic between interpreter and native C backend.
  - [ ] Connect physical testbed hardware (flight controller / companion computer) for hardware-in-the-loop (HIL) certification.

---

### Division 08: Developer Tooling, LSP & CLI Experience (K08)
> Detailed Specification: [`docs/kaze-map/K08-tooling-lsp-cli.md`](docs/kaze-map/K08-tooling-lsp-cli.md)
* **Mission**: Deliver a lightning-fast developer experience through command-line utilities, an opinionated auto-formatter, and a native Language Server Protocol (LSP) daemon.
* **Owned Files**:
  * `kaze/src/main.rs` (CLI command dispatcher: `run`, `check`, `compile`, `fmt`, `emit-c`, `to-machine`, `to-human`)
  * `kaze/src/lsp.rs` & `kaze/src/lsp_main.rs` (Language Server Protocol daemon over JSON-RPC)
  * `kaze/src/fmt.rs` (Canonical AST auto-formatter)
  * `kaze/editors/` (VS Code and Neovim syntax definitions and grammar files)
* **Architectural Boundaries**:
  * Idempotent formatting: Running `kaze fmt` multiple times produces byte-identical outputs.
  * 12-factor CLI inputs: All inputs enter via `--arg key=value`, environment, config file, or stdin.
  * Sub-millisecond LSP responses: Diagnostics, symbol indexing, and type hovers computed in-memory directly from parsed AST.
* **Active Tasks**:
  - [x] Implement CLI command dispatcher supporting 12-factor argument hierarchy.
  - [x] Build native Language Server Protocol server daemon (`lsp.rs`).
  - [x] Implement canonical auto-formatter (`fmt.rs`).
  - [x] Complete 60-command CLI integration test matrix across all commands.
  - [ ] Add semantic token highlighting to LSP server for dimensional units.

---

### Division 09: Embedded Sofuu Bridge & C-FFI Integrations (K09)
> Detailed Specification: [`docs/kaze-map/K09-embedded-sofuu-bridge.md`](docs/kaze-map/K09-embedded-sofuu-bridge.md)
* **Mission**: Provide seamless, zero-drift integration between Kaze and the Sofuu runtime substrate, QuickJS host, and foreign language C-ABIs.
* **Owned Files**:
  * `crates/sofuu-core/src/domains/kaze/` (Embedded Kaze runtime mirror: lexer, parser, checker, compiler, VM, caps)
  * `crates/sofuu-core/src/script_cmd.rs` (`sofuu script check`, `compile`, `run`, `repl`)
  * `crates/sofuu-core/src/compiler.rs` (`sofuu compile` standalone binary compiler)
* **Architectural Boundaries**:
  * Zero drift: Embedded source maintains 100% parity with standalone `kaze/src/` (`scripts/kaze-drift-check.sh --strict`).
  * Asynchronous continuation bridge: `sofuu.kaze.runAsync` suspends at authorized `ai` or `tool` calls and resumes cleanly.
* **Active Tasks**:
  - [x] Synchronize embedded Kaze to unified 49-opcode and 17-verb contract (0 drift verified).
  - [x] Implement asynchronous continuation bridge for AI/tool dispatch (`544c680c8`).
  - [x] Maintain automated drift check script (`scripts/kaze-drift-check.sh --strict`).
  - [ ] Add format-5 reader to embedded Sofuu engine to ingest format-5 artifacts.

---

### Division 10: Verification, Conformance & Test Suite (K10)
> Detailed Specification: [`docs/kaze-map/K10-verification-conformance.md`](docs/kaze-map/K10-verification-conformance.md)
* **Mission**: Deliver rigorous mathematical verification, cross-engine drift prevention, bootstrap fixed-point validation, and automated compliance testing.
* **Owned Files**:
  * `kaze/tests/` (Unit, integration, strict semantics, units, contracts, diagnostics, emitted C, live, and capability safety suites)
  * `kaze/scripts/` (`rt-check.sh`, `diff-check.sh`, `diff-fuzz.sh`, `emitter-triangle.sh`, `self-sync.sh`)
  * `scripts/kaze-drift-check.sh` (Strict cross-engine drift detector)
* **Architectural Boundaries**:
  * Strict gating: Commits modifying Kaze must pass `scripts/kaze-drift-check.sh --strict` with zero diffs.
  * Local weights only: Inference tests must never attempt remote model downloads.
  * Reproducible builds: Multi-stage bootstrap verified bit-for-bit ($\text{sha256}(s2.c) == \text{sha256}(s3.c)$).
* **Active Tasks**:
  - [x] Establish automated cross-engine drift gate (`scripts/kaze-drift-check.sh --strict`).
  - [x] Capability safety and effector safety test suites (`capability_safety.rs`, `effector_safety.rs`).
  - [x] 12-flavor differential fuzzer (`diff-fuzz.sh`) asserting behavioral parity across engines.
  - [x] Multi-stage Stage 2 / Stage 3 bootstrap fixed-point verification.
  - [x] C-emitter triangle verification (`emitter-triangle.sh`).

---

## 4. Development Workflow & Change Discipline

When developing, refactoring, or extending Kaze:

1. **Identify the Division**: Determine which division (K01–K10) owns the change.
2. **Consult Architectural Boundaries**: Review the invariants in the corresponding division specification before writing code.
3. **Preserve Generalization**: Never introduce domain-specific parameters or endpoints into the core language.
4. **Enforce Zero-Panic Invariants**: Ensure all fallible paths require `otherwise` or contract assertions.
5. **Verify Locally**: Execute that division's independent verification command.
6. **Run Cross-Engine Drift Check**:
   ```bash
   bash scripts/kaze-drift-check.sh --strict
   ```
7. **Run Full Test Suites**:
   ```bash
   cargo test --manifest-path kaze/Cargo.toml
   cargo test -p sofuu-core domains::kaze
   ```
8. **Update Division Checklist**: Mark completed tasks in both this map and the division document.
