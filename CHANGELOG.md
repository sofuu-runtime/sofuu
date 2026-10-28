# Changelog

All notable changes to the `sofuu` runtime are documented here.
Versions follow the Cargo crate (`0.3.0-beta` line); npm and the CLI banner
carry the same version (checked by `scripts/verify-dist.sh`).

## 0.3.0-beta — 2026-10-28 (in development)

Mandatory QTSQ core integration and runtime conformance program completion:

- **Mandatory QTSQ Core** — Made QTSQ memory a permanent core component of the
  runtime across all targets; retired the legacy QTSQ-free build variant.
- **Contract Conformance Unblocking (51 PASS · 0 FAIL · 1 BLOCKED)** —
  Unblocked 9 previously blocked conformance test cases across capabilities
  manifest parity, single terminal event enforcement, tier and bound declarations,
  deterministic tolerant bounds, opt-in best effort decode, forward schema migration,
  MCP envelope indistinguishability, cross-surface error taxonomy parity, and token
  usage reconciliation.
- **Single Terminal Event Fix** — Resolved agent runtime double-stop bug under
  step budget exhaustion.
- **Cross-Surface Capabilities** — Added `sofuu capabilities` CLI subcommand and
  `/v1/capabilities` HTTP routes for 100% manifest parity with JS and C ABI.

## 0.2.0-beta — 2026-10-26

The runtime-contract program, phases P0–P4:

- **P0** — the normative contract (`docs/SPEC-RUNTIME-CONTRACT.md`, local-only
  by repo policy), the 48-cell surface matrix, the 52-case conformance suite,
  and the tracking artifacts under `contract/`.
- **P1** — the capability model: `sofuu.capabilities()`, `sofuu.model.describe`,
  `sofuu doctor` hard-fails on a missing declared feature, a typed
  `E_CAPABILITY` refusal on QTSQ-free builds, determinism receipts
  (`scripts/text_gate.py`) with the R3 CI gate, and the committed fixture
  model the gate runs (`contract/fixtures/sofuu-fixture.gguf`).
- **P2** — one async contract (`sofuu.async.*`: the §1.4 taxonomy, nested
  deadlines, queue bounds) and one event envelope (`sofuu.events.*`: the §7.2
  schema, validator, redaction, renderer, the per-session `events.ndjson`
  log). Live-path chaos harnesses: deadline kill, mid-SSE cancel, mid-tool
  cancel (subprocess killed, no orphan), server-side disconnect visibility,
  bounded-pump backpressure (`E_BACKPRESSURE`), usage reconciliation
  (`sofuu.events.canonicalUsage`; `ai.complete` reports provider usage).
- **P3** — the session document (`rt/session_doc.rs`, §5.2 v1): one
  export/import format behind `sofuu.session.*` and
  `sofuu session export|import`; the canonical
  `<project>/.sofuu/sessions/<id>/session.json` written by chat;
  restart-stable per-turn ids; `schema_version` on sessions, the event
  envelope, and the config; import-side format allowlisting (R4.5) and
  export redaction (R4.6); serve `/v1/*` OpenAI-wire conformance
  (models / SSE / JSON); the C-ABI symbol gate executable in the suite
  and CI (`CONTRACT_LIBSOFUU`).
- **P4** — one security policy everywhere (R7): the four-surface policy
  matrix (JS / CLI / HTTP / C ABI — same decision, same rule id, no bypass
  by entry point), approval deadline-deny and registration-order
  guarantees in the trace, the shared shell/fs refusal shape
  (`SOFUU_NO_SHELL` strips every surface), net-policy suffix anchoring with
  normalization, secrets-never-emitted checks, the MCP allowlist gate,
  project-policy auto-install at serve boot, and install integrity:
  checksummed `dist/` artifacts + the capabilities-vs-claim + version-parity
  gate (`scripts/verify-dist.sh`).

## 0.2.0-alpha — 2026-09

Engine substrate: in-process inference (GGUF/Metal), the agent loop,
encrypted QTSQ memory, MCP client/server, the C-ABI embed surface.
