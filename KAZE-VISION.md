# Kaze Vision

Kaze is intended to be a general-purpose language within the domain of AI:
its scope spans AI software applications and AI-powered physical systems.
Examples include agents, AI apps and workflows, robotics, drones, smart devices,
and other AI workloads. These examples do not limit the intended scope to a
fixed list. Kaze is not intended to be general-purpose across all computing.

This is a product direction, not a claim that every AI application, platform,
language feature, integration, or physical device is supported today. Current
behavior and evidence must be reported per implementation, host, and artifact.
In particular, the embedded engine and standalone Kaze have different current
language and bytecode versions; a shared language contract remains planned.
See [KAZE-EMBEDDED.md](KAZE-EMBEDDED.md) and [sofuu-map.md](sofuu-map.md).

## One language, two syntaxes

Kaze supports two first-class source syntaxes for the single language:
- **Human syntax:** Natural, sentence-like, indentation-based phrasing designed for human reading, authoring, and review.
- **Machine syntax:** Tagged prefix-tree S-expression text `(kaze (format 1) ...)` designed for precise AI generation, parsing, and structured exchange.

Both syntaxes parse transparently into the unified Kaze AST program model, undergoing identical static checks, compiling to the same bytecode or native instructions, and enforcing identical fail-closed capability gates. Bidirectional conversion is supported via `to-machine` and `to-human` tooling. Specification details are in `docs/plans/PLAN-KAZE-DUAL-SYNTAX.md`.

## Design boundaries

- Kaze's compiler and runtime core remain domain-agnostic: do not hardcode
  application-specific parameters, defaults, APIs, or endpoints.
- Kaze agents remain pure Kaze, without required agent-specific JavaScript
  sidecars or shell wrappers. Supported host services and external backends may
  still be necessary; pure source does not mean all dependencies are embedded.
- External effects require explicit capabilities, grants, and applicable
  policy. Unavailable capabilities fail closed; maintain resource limits and
  reject invalid operations instead of silently changing them.
- JavaScript and TypeScript remain available Sofuu application surfaces; their
  existence does not restrict Kaze to agent routines or physical control.
  Which application facilities Kaze should provide is a design decision, not
  a claim of present implementation or a mandate to duplicate every JS API.
- Kaze's language core is independently designed and implemented: its lexer,
  parser, checker, compiler, VM, Kaze-specific execution and capability
  behavior, and Kaze-owned standard-library implementations are our own.
  Learn from other projects, but do not copy their code or exact implementation,
  even when a license would otherwise permit copying. This is a design rule,
  not an audit claim about the authorship of all existing code.
- External providers may be used at explicit, documented boundaries, such as
  OS services, hardware and GPU runtimes, TLS or cryptography providers, build
  tools, and model or MCP services. They remain external dependencies, not
  Kaze-owned code. Identify applicable dependencies and preserve their license
  and attribution requirements; do not claim an entire binary is Kaze-owned.
- Review existing core dependencies individually. A Kaze-owned replacement,
  such as a possible future JSON implementation, is a considered long-term
  direction where useful and properly verified, not a mandate to remove a
  working dependency prematurely or a claim that removal has happened.
- Describe real model, tool, memory, and physical behavior according to
  evidence. Explicitly labelled robotics/drone simulation is not evidence of
  physical-hardware execution.

Bounded execution and capability gating are safety design choices, not a
statement that Kaze is limited to a small set of scripts or use cases.

## Self-hosting & systems language direction

Kaze is evolving from an interpreted bytecode engine into a self-hosted systems language
written 100% in Kaze. The architecture reconciles high-level human syntax with low-level
systems execution:
- **High-level human syntax:** Natural, punctuation-free intent with zero lifetime annotations (`'a`) or raw pointer sigils.
- **C/C++ native performance:** Native AOT code generation, unboxed record layouts, linear bump arenas, SIMD vectorization, and zero GC pause.
- **Rust-level safety:** Guaranteed spatial memory safety, temporal safety through lexically scoped borrowing and regional arenas, non-null values, and fail-closed capability gates.
- **Full 32-bit & 64-bit cross-target support:** Native targeting of 64-bit workstations/servers (`arm64`, `x86_64`) and 32-bit robotics/drone microcontrollers (`armv7-m` Cortex-M4/M7, `riscv32`, ESP32).

The bootstrap path uses the current Rust engine (Stage 0) to compile the first Kaze-written
compiler (Stage 1), culminating in full self-compilation (Stage 2) with proven binary identity.
See `docs/plans/PLAN-KAZE-SELF-HOSTED-SYSTEMS-LANGUAGE.md`.
