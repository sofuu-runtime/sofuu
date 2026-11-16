# Security Policy

Sofuu executes agent-driven tools, spawns processes, reads and writes files, and
opens network connections on your behalf. Its security posture is therefore part
of the product, not a footnote. This page explains what is covered, how to report
a problem, and what we promise in return.

* **What R7 protects against, and what it cannot** → the Scope section below
  (guarantees and residual risks are stated there, inline)
* **Normative rules** → `docs/SPEC-RUNTIME-CONTRACT.md` §8 (R7.1–R7.9)
* **Executable evidence** → `contract/CONFORMANCE.md` (T7 cases) and
  `docs/SANDBOX.md`

---

## Supported Versions

| Version | Supported |
| :--- | :--- |
| 0.3.x | :white_check_mark: — security fixes land on `main` and in the next release |
| < 0.3 | :x: — please upgrade; no fixes are backported |

Security fixes are released as soon as they are verified. There is no
long-term-support branch; pre-1.0, `main` is the supported line.

---

## Reporting a Vulnerability

**Do not open a public GitHub issue, discussion, or pull request for a suspected
vulnerability.** Public disclosure before a fix exists puts every Sofuu user at
risk.

Send a private report to:

* **Email:** [security@sofuu.xyz](mailto:security@sofuu.xyz)
  (fallback: [hello@sofuu.xyz](mailto:hello@sofuu.xyz))
* **Subject:** `[SECURITY] <subsystem> — <one-line impact>`

### Encrypted reports

**No PGP key is published yet.** We will not pretend otherwise: until a key is
generated and its fingerprint published in this file, reports travel as ordinary
email. If your report is sensitive and you need encryption, say so first and we
will publish a key fingerprint for you to verify out-of-band before you send
details. Never send a working exploit to an unverified key.

### What to include

1. The affected surface — CLI/`sofuu run`, `sofuu chat`, TUI, `sofuu serve`,
   `libsofuu` C ABI, the sandbox, plugins, MCP, or the release/install path.
2. Sofuu version (`sofuu --version`), OS and architecture, and the exact policy
   in effect (`.sofuu/policy.json`, `--policy`, or `sofuu.policy.get()`).
3. A minimal reproduction: script, policy file, and the commands you ran. If the
   issue needs network access, describe the peer instead of pointing at a live
   host.
4. Impact: what an attacker gains (file read/write, code execution, credential
   disclosure, denial of service, policy bypass) and what preconditions they need.
5. Optional: a suggested fix or the rule id you believe is being bypassed.

Please give us a reasonable window to fix the issue before publishing. We will
credit you in the advisory unless you prefer to stay anonymous.

---

## Response commitments

| Stage | Target |
| --- | --- |
| Acknowledgement of the report | within 48 hours |
| Triage verdict (confirmed / not a vulnerability / needs info) with the affected rule id | within 5 business days |
| Fix for critical or high severity | within 7 days of confirmation |
| Fix for medium / low severity | next scheduled release |
| Public advisory | after the fix ships, coordinated with the reporter |

If a report is out of scope per the out-of-scope list below, we will say so
plainly and explain why, rather than silently closing it.

---

## Scope

**In scope** — anything that breaks a guarantee the R7 firewall makes:
policy bypass across any of the four surfaces (JS, CLI, HTTP, C ABI), shell-strip
evasion, filesystem-jail escape (`..`, symlinks, canonicalisation), network/SSRF
guard bypass, secret disclosure through events/traces/exports/errors, sandbox
escape or budget bypass, plugin shield bypass, memory-safety defects in the
TypedArray/SIMD or C-ABI boundary, and release-artifact integrity.

**Out of scope** — host/root compromise, hardware and side-channel attacks,
malicious plugins or MCP servers you chose to install, prompt-injection that
results in *allowed* actions, model behaviour, and denial of service that follows
from budgets you deliberately unset. These are documented residual risks, not
bugs — they follow from the trust model above (plugins, sandbox bridges, and
MCP servers are trusted code; redaction is key-based; symlink TOCTOU remains).

## Bug bounty

There is no paid bug bounty program. We do not want to imply one exists. We do
offer public credit, a detailed write-up of the fix, and our genuine thanks.

## Hardening you can apply today

* Run high-stakes agents with `approval.mode: "always"` and a narrow
  `tools.allow`.
* Set `fs.roots` to the project directory and `fs.write: "off"` when reads are
  enough.
* Keep `net.allow` narrow and enable the agent `netPolicy.ssrf` guard.
* Verify release artifacts against `SHA256SUMS` before installing; `install.sh`
  refuses to proceed when a checksum tool is unavailable.
* Run untrusted logic in `sofuu.sandbox` with explicit `fuel`, `timeoutMs`, and
  `maxMemoryMB` rather than in the host realm.
* Use the out-of-band **Scouter** engine (`/scouter` or `/health` in `sofuu chat`, or `sofuu.scouter` in scripts) to continuously monitor model context integrity and cognitive vector memory health.

---

## Scouter: Tamper-Proof Mathematical Context & Memory Integrity (R7.10)

Sofuu includes **Scouter** (`crates/sofuu-core/src/rt/scouter.rs`), a dedicated, real-time observability engine designed to detect prompt injection, context hacking, and cognitive memory poisoning before they can compromise downstream agent behavior.

### 1. Pure Mathematics & Zero Machine Learning
Scouter operates **strictly via deterministic mathematics and algorithms**—it never invokes neural networks, secondary LLMs, or ML classifiers:
* **Immunity to Jailbreaks:** Because Scouter uses no learned models or neural weights, an attacker cannot jailbreak, deceive, prompt-inject, or induce hallucinations in the monitor itself.
* **Bounded Latency:** Mathematical evaluations complete in under 0.5 ms per turn with zero dynamic memory allocation spikes.

### 2. Tamper-Proof Out-of-Band State
Scouter runs as an isolated subsystem in private Rust host memory:
* **Prototype Pollution Immunity:** The global health tracking state (`AtomicU32`, `Mutex<[ScouterThreat; 32]>`) resides outside the JavaScript heap. Compromised JS scripts cannot alter Scouter's internal history or suppress active threat alerts via `Object.prototype` poisoning.
* **Anti-Terminal Spoofing:** Tool outputs and incoming streams are filtered through an ANSI control-code sanitizer (`sanitize_terminal_stream`) before terminal display. Adversaries cannot clear lines (`\x1b[2K`), jump cursors (`\x1b[H`), or forge fake `[H:100% HEALTHY]` status chips in developer terminals.

### 3. Prompt Sovereignty
Scouter is **strictly observational**:
* It never silently alters, redacts, or rewrites prompts or tool results.
* It never silently truncates, corrupts, or deletes cognitive vector memories.
* The developer and operator retain 100% prompt sovereignty. When anomalies or poisoning attempts occur, Scouter emits real-time visual alerts (interactive HUD, footer health chip `[H:score%]`, or programmatic JSON reports) so human operators and systems supervisors can intervene intentionally.

### 4. Universal Execution Coverage (Not Just TUI)
Scouter is not merely an interactive terminal widget; it is an in-process, universal runtime engine capability integrated across all execution surfaces:
* **Headless Autonomous Agents (`sofuu agent run` & `src/js/agent.js`):** Continuously scores context health after each tool turn and completed step checkpoint. Threats emit structured `scouter` and `warning` events into `.sofuu/logs/agent-<run_id>.jsonl` traces, and the run summary reports live health in both human summary lines and `--json` payloads.
* **Background Supervisor Daemons (`sofuu daemon`):** Inspects the health of unattended background agent processes.
* **HTTP API Server (`sofuu serve`):** Exposes live health metrics on `GET /health` and `GET /v1/scouter`; evaluates inbound message payloads on `POST /v1/chat/completions`.
* **Built-in Tooling (`src/js/tools.js`):** The standard `system_info` tool includes live Scouter health metrics for agents inspecting their own runtime state.
* **Programmatic API (`sofuu.scouter`):** JavaScript and TypeScript scripts have full direct access via `sofuu.scouter.contextHealth()`, `sofuu.scouter.brainHealth()`, `sofuu.scouter.status()`, and `sofuu.scouter.sanitizeAnsi()`.
* **Interactive Terminal TUI (`sofuu chat`):** Provides the `/scouter` and `/health` diagnostic HUD along with the real-time footer status chip `[H:score%]`.
