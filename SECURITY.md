# Security Policy

Sofuu takes security and safety seriously. Because Sofuu executes agent code, runs tools, and interfaces with local filesystem structures and network endpoints, maintaining airtight isolation is a core priority.

---

## Supported Versions

| Version | Supported |
| :--- | :--- |
| 0.3.x | :white_check_mark: |
| < 0.3 | :x: |

---

## Reporting a Vulnerability

If you discover a security vulnerability, sandbox escape, or credential leakage issue in Sofuu, please **do not report it via public GitHub issues**.

Instead, please send an encrypted or direct report to our security team:

* **Email:** [security@sofuu.xyz](mailto:security@sofuu.xyz) (or [hello@sofuu.xyz](mailto:hello@sofuu.xyz))
* **Subject:** `[SECURITY VULNERABILITY] <Component/Subsystem>`

Please include:
1. Detailed description of the vulnerability and affected surfaces (CLI, Sandbox v2, MCP bridge, or QTSQ container).
2. Minimal reproducible proof-of-concept (POC) script or configuration.
3. Impact assessment and potential remediation steps.

We will acknowledge receipt within 48 hours and work with you on a coordinated disclosure timeline.
