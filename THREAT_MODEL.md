# Sofuu Threat Model (R7)

**Scope:** the `sofuu` runtime, its CLI/TUI, the HTTP server (`sofuu serve`),
the embeddable `libsofuu` C ABI, the plugin loader, the in-process sandbox, and
the release/install path.
**Version:** applies to the `0.3.0-beta` line and later until superseded.
**Companion documents:** `SECURITY.md` (disclosure), `docs/SPEC-RUNTIME-CONTRACT.md`
§8 (R7 normative rules), `docs/SANDBOX.md` (sandbox limits),
`contract/CONFORMANCE.md` (the executable T7 cases).

This document is deliberately conservative: it lists what R7 **does**, what it
**cannot** do, and the residual risks an operator is accepting. If a claim here
and the code disagree, the code wins — please report the discrepancy.

---

## 1. Assets

| Asset | Why it matters |
| --- | --- |
| The developer's filesystem | An agent can read/write files through tools. |
| API keys, tokens, `process.env` | Exfiltration is the highest-value outcome of a prompt injection. |
| The host's network identity | SSRF into loopback, private ranges, or cloud metadata. |
| Host integrity | The runtime is a native process; memory-safety bugs are host bugs. |
| Spend and availability | Runaway loops, unbounded token spend, fork bombs. |
| Audit truth | Traces, session documents, and crash cards must not leak secrets. |

## 2. Trust boundaries

```
 UNTRUSTED                    │ TRUSTED (operator's choice)
 ─────────────────────────────┼──────────────────────────────────────────
 LLM output (tool calls)      │ The developer / operator
 Web pages, documents, MCP    │ Policy files, CLI flags, config
   tool results               │ Installed plugins (code you chose to run)
 Network payloads             │ Model weights you chose to load
 Sandboxed code (JS/TS)       │ The Rust core + vendored C (QuickJS, libuv, curl)
                              │ Host kernel (assumed sound)
```

Two boundaries deserve emphasis because they are easy to misread:

1. **Plugins are trusted code.** A plugin module is evaluated in the host
   realm and can call `fetch`, `sofuu.fs`, and any global directly. The five
   shields constrain the *tool callable* it registers (schema, policy,
   breaker, watchdog, panic guard), not the module body. Installing a plugin
   is equivalent to adding a dependency.
2. **Sandbox bridges are deliberate holes.** `sofuu.sandbox` runs a separate
   QuickJS realm with no host globals, but every function passed in `globals`
   is a callable host bridge — whatever that function can reach, sandbox code
   can reach through it.

## 3. In-scope threats and the controls that answer them

### 3.1 Indirect prompt injection → destructive or exfiltrating tool calls

A document or page convinces the model to run `rm -rf`, read `~/.ssh/id_rsa`,
or POST `process.env` somewhere. R7 cannot stop the model from *asking*; it
constrains what asking can achieve.

| Control | Rule | Behaviour |
| --- | --- | --- |
| Tool gate | `R7.6/tools.allow`, `R7.6/tools.deny` | Empty allow = all; deny wins; glob patterns accepted. |
| Shell strip | `R7.6/shell.enabled` | `false` blocks every shell interpreter (`sh`, `bash`, `zsh`, `busybox`, …) on **every** surface, including `sofuu.spawn` called from host JS. `SOFUU_NO_SHELL=1` strips the `bash` tool from the advertised set as well. |
| Shell allowlist | `R7.6/shell.allow` | Command patterns; the inventory question (`tool.run bash` with no command) reports `conditional` instead of guessing. |
| Approvals | `R7.4/approval.mode` | `always` turns every tool call into an ask; `never` runs without prompting (denies still win); `per-run` is the driver default. |
| Filesystem jail | `R7.6/fs.roots`, `R7.6/fs.write` | Paths are lexically folded and the deepest existing ancestor is canonicalised, so `..` and symlinks are judged by where a write would really land. `fs.write: "off"` makes the tree read-only. |
| Network egress | `R7.1/net.allow`, `R7.1/net.deny` | Hosts are normalised (scheme, `user@`, `:port`, `[v6]`, trailing dot) before suffix-anchored matching; deny wins. Loopback aliases are probed together, so denying `127.0.0.1` also denies `::1`/`localhost`. |
| Inbound routes | `R7.6/http.deny` | `sofuu serve` rejects denied path prefixes with `403 policy_denied` naming the rule. |

### 3.2 SSRF against cloud metadata and private networks

Layered, and **not uniform by default** — the honest statement:

| Surface | Default | Configurable |
| --- | --- | --- |
| Raw `fetch()` | `169.254.169.254` and non-`http(s)` schemes are always refused. Loopback/private targets are **allowed** (local dev servers are legitimate). | Agent `netPolicy.ssrf` refuses every internal range at connect time, after DNS, so rebinding cannot dodge it. |
| Agent tools (`web_open`, `web_search`) | Hostnames are classified before dialling: loopback names/IPs, RFC1918, link-local, CGNAT, unique-local v6, IPv4-mapped tails, decimal/octal/hex/`127.1` spellings. | Opt-out for deliberate internal use. |
| Policy `net.deny` | Empty. | Exact IPs are additionally enforced against the resolved address (`ip_denied`). |

Internal ranges covered by the IP guard: `127/8`, `10/8`, `172.16/12`,
`192.168/16`, `169.254/16`, `100.64/10`, `192.0.0/24`, `0.0.0.0`, broadcast,
`::1`, `fc00::/7`, `fe80::/10`, and IPv4-mapped forms of all of them.

**Residual risk:** a policy that allows a hostname does not pin DNS. The
resolved-IP guard closes the gap for the `ssrf` policy flag and for denied IP
literals, but an operator who allows `*.example.com` and disables `ssrf` is
trusting that domain's DNS.

### 3.3 Path traversal and symlink escape

`fs.roots` is enforced twice: lexically (shared by the JS and Rust layers, so
both agree on `..`) and against the canonicalised real path of the nearest
existing ancestor. A symlink that points outside the roots is caught by the
canonical prefix rather than its spelling; a write to a not-yet-existing file
is judged by the directory it will land in.

**Residual risk:** TOCTOU. Between the policy check and the operation, a local
attacker with write access to the jail could re-point a symlink. R7 does not
hold a file descriptor across the check; do not run untrusted local processes
with write access to your workspace and expect the jail to hold.

### 3.4 Secret leakage

| Channel | Control |
| --- | --- |
| Event envelope / traces / crash cards | `redact_value` replaces values under secret-looking keys (`api_key`, `apikey`, `secret`, `password`, `authorization`, `bearer`, `token`, `*_token`) before any write. |
| Session export | Export path strips secret-bearing fields (`R4.6`); an exported document carrying `api_key`/`Authorization` ships none. |
| Errors | Provider keys are never echoed into error text (T7.8 asserts this, including a 401 that echoes the key). |
| `sofuu serve` | Bearer token from `SOFUU_SERVE_TOKEN` compared in constant time; `--token` is accepted but documented as visible in `ps`. |

**Residual risk:** redaction is key-based, not content-based. A token stored
under an innocuous key, or pasted into free text the model repeats, is not
detected. Do not put secrets in prompts.

### 3.5 MCP trust boundary

`mcp.servers.allow` (when non-empty) refuses to connect to anything else;
`sofuu.tools.mcpResult` normalises MCP results so a server's tool is
indistinguishable from a built-in one (same envelope, same policy evaluation —
T5.6). MCP servers are otherwise trusted processes: they run with your
privileges and their output is model input.

### 3.6 Resource exhaustion

| Vector | Control |
| --- | --- |
| Runaway sandbox code | Instruction fuel + wall-clock deadline (default 2 s / 5M), memory and stack ceilings; every limit is a default, not a cap. |
| Catastrophic regex | The QuickJS bytecode meter now also polls inside native regexp backtracking. |
| Runaway agent loops | Step/depth/wall budgets, cancellation, `E_BACKPRESSURE` on bounded queues. |
| Memory bombs in the runtime | Rust code paths are bounded; session/event logs have size caps. |

### 3.7 Memory safety of the native substrate

The core is Rust; the C surfaces are QuickJS, libuv, libcurl, and the QTSQ
codec. The FFI boundary that matters most is the TypedArray bridge (a
JavaScript `Float32Array` handed to C SIMD code):

* only genuine `Float32Array` views are accepted (a `Float64Array` is never
  re-interpreted as f32);
* the view's byte offset and length are bounded by the underlying buffer, and
  detached buffers are refused with a typed error;
* element counts are checked against the expected dimension and against
  non-finite values (NaN/±Inf) at ingest and query;
* the same validator backs `sofuu.memory` and `sofuu.ai.similarity/dot/l2`.

Evidence: unit tests for each malformed shape, `crates/fuzz` targets, and an
AddressSanitizer CI leg (Linux nightly, `-Zbuild-std`). The vendored C is not
UBSan-instrumented, so this is ASAN coverage of the Rust side of the boundary.

## 4. Explicitly out of scope

* **A compromised host.** Root, a malicious kernel, or a debugger attached to
  the process defeats any in-process control. Sofuu is not a hypervisor
  (Firecracker/gVisor) and does not claim to contain a host compromise.
* **Hardware and side channels.** Spectre-class speculation, Rowhammer,
  power/EM analysis, cold-boot attacks.
* **Malicious code you install.** Plugins, npm packages pulled by the user,
  MCP servers, and model weights are trusted by decision. Sofuu does not
  audit them.
* **The model layer.** A model can be induced to *say* anything; R7 bounds
  actions, not speech. There is no semantic "prompt injection detector".
* **Native addons.** Sofuu deliberately never loads `.node` C++ addons; there
  is no JIT and no executable memory emission (iOS policy by construction).
* **Denial of service you inflict on yourself.** Unbounded budgets are a
  supported feature; choosing them is an operator decision.

## 5. Known residual risks

| # | Risk | Why it remains | Mitigation available today |
| --- | --- | --- | --- |
| R1 | Prompt injection can still cause *allowed* actions | R7 is a capability policy, not an intent classifier | Narrow `tools.allow`, `fs.roots`, `net.allow`; `approval.mode: "always"` for high-stakes runs |
| R2 | Plugin code has host access | Plugins are dependencies | Install only reviewed plugins; run untrusted logic in `sofuu.sandbox` instead |
| R3 | A bridged host function widens the sandbox | Bridges are the feature | Bridge narrow, validated functions only |
| R4 | DNS may change between allow-check and connect | No connection pinning for allowed hostnames | Keep `net.allow` narrow; enable `netPolicy.ssrf` |
| R5 | Symlink TOCTOU inside the jail | No fd-held jail | Do not co-locate untrusted local processes with the workspace |
| R6 | Key-based redaction misses unknown field names | Content scanning is unreliable and expensive | Do not put secrets in prompts or tool arguments |
| R7 | In-process bugs in C dependencies | FFI validation is not a proof | Keep ASAN/fuzz green; update vendored C promptly |

## 6. Attack-surface inventory

| Input | Entry point | Validated by |
| --- | --- | --- |
| CLI flags / env | `main.rs` | typed parsers; unknown values refused |
| Policy JSON | `--policy`, `.sofuu/policy.json`, `sofuu.policy.set` | schema-checked; invalid leaves the previous policy; fail closed |
| Script/TS source | `sofuu run`, plugins, sandbox | TS stripper; sandbox budgets |
| LLM tool calls | agent loop | tool gate → policy → approval → watchdog → panic guard |
| MCP results | `sofuu.tools.mcpResult` | envelope + policy parity |
| HTTP requests | `sofuu serve` | bearer auth (constant time), `http.deny`, route allowlist |
| Session/brain files | `sofuu.session`, `sofuu.memory` | schema allowlist on import; CRC/manifest checks in the codec |
| Release artifacts | `install.sh` | SHA-256 manifest; optional minisign signature; SBOM |

## 7. Verification

* `scripts/contract/run_contract_tests.sh` — T7.1–T7.9 execute the four-surface
  policy matrix, the shell strip, the fs jail, approval ordering, secret
  non-emission, `net` normalisation and the MCP allowlist.
* `make test` — JS suites including `policy_test.js`, `web_guard_test.js`,
  `shell_inventory_test.js`.
* `cargo test -p sofuu-core --lib rt::policy` — evaluator unit tests.
* `crates/fuzz` targets + `make asan` / the ASAN CI leg.
* `make sandbox-stress`, `make plugin-stress`, `make observability-stress`.

## 8. Reporting

See `SECURITY.md`. Please do not open a public issue for a suspected
vulnerability.
