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

The intended direction is one Kaze language with a human-oriented source
syntax and a machine-oriented source syntax for AI and other software. They
are not separate languages or permission tiers: equivalent programs must share
the applicable semantics, resource limits, capabilities, and authorization
rules. The machine syntax is not simply bytecode or an internal representation.
An independently implemented tagged prefix-tree text form is the leading
machine-syntax candidate for evaluation, not a finalized grammar. Compare it
with tagged JSON and generation of human syntax using representative programs
and measured results. Human syntax should evolve from existing readable Kaze,
with explicit compatibility and migration decisions. Grammar, conversion and
release scope remain open; machine syntax and parity are not implemented.
The local discussion draft is `docs/plans/PLAN-KAZE-DUAL-SYNTAX.md`.

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

## Self-hosting direction

A Kaze-written compiler is a long-term direction, not current functionality.
The current Rust implementation can serve as a bootstrap for an independently
written Kaze compiler after Kaze has the necessary facilities. Verify that
compiler against an agreed language contract and conformance cases before
claiming self-hosting. Whether the VM, standard-library components, or host
integrations should also be written in Kaze is a separate decision; a
self-hosted compiler does not imply that every runtime or provider is Kaze code.
