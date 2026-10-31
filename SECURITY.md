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
