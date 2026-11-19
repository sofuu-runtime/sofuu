# tools-mcp-policy — D04 tools, MCP and policy boundaries

Scope of this worker: `crates/sofuu-core/src/rt/mcp.rs`,
`crates/sofuu-core/src/rt/policy.rs`, `src/js/tools.js`, `src/js/plugins.js`,
`src/js/web.js`, `src/js/providers.js`. Division D04 (tools, MCP, policy
boundaries) of `docs/plans/PLAN-REAL-EXECUTION-REMEDIATION.md`.

## Checks performed

| Check | Command / source read | Result | Evidence |
|---|---|---|---|
| R1: a registered tool must execute its real implementation with real arguments, and every failure mode must stay an error | read `src/js/tools.js:297-352` (`executeWithPolicy` — the single enforcement point) and `src/js/tools.js:3985` (`installPolicyWrappers`); Node harness over the **real** `src/js/tools.js` source in a `vm` sandbox with a `sofuu` stub (`/tmp/sofuu-tools-check/check.js`) | pass (after the changes below) | 6 passed / 0 failed: `grep` with no `pattern` is rejected with `E_SCHEMA` and `path='$.pattern'` instead of returning "no matches for /undefined/"; `grep` with a pattern passes the gate; `read_file` returns the real file content through the same gate; an unknown tool is still `E_TOOL`; a tool registered after the wrappers were installed executes through `sofuu.tools.execute` and returns `'ran:7'`. |
| R1: a denied tool must not dispatch | read `src/js/tools.js:328-347` | pass | The policy verdict is resolved before `toolDispatchBegin()` and before the schema gate; a deny raises `E_POLICY_DENIED` (`tools.js:264-267`), an `ask` raises `E_POLICY_APPROVAL_REQUIRED` (`tools.js:226`). |
| R2: MCP stdio and MCP Streamable-HTTP must be real connections to a real server, and the harness must exercise the reviewed code path | `SOFUU_BIN="$PWD/target/release/sofuu" python3 scripts/test-real-mcp-file.py`; read `scripts/test-real-mcp-file.py`, `scripts/contract/harness/real_file_mcp_client.js`, `scripts/contract/harness/real_http_mcp_client.js` | pass — **against the pre-change artifact** | Both phases PASS: "PASS: real MCP file read, denial, missing-file error, deadlines, reconnect policy and no replay after lost responses" and "PASS: real HTTP MCP file read, policy denial, ping, deadlines and circuit breaker". The harness runs `sofuu run scripts/contract/harness/real_file_mcp_client.js` (stdio → `src/js/tools.js` `__paged` wrapper → native `crates/sofuu-core/src/rt/mcp.rs` stdio client: `uv_spawn` child + JSON-RPC over pipes) and `real_http_mcp_client.js` (HTTP against a Python `ThreadingHTTPServer` on 127.0.0.1, with `SOFUU_ALLOW_LOOPBACK=1`). The code path the harness drives is the code path under review. |
| R3: a DENIED operation must produce **zero dispatch** — the absence of the request, not the presence of a denial message | read `scripts/contract/harness/real_file_mcp_client.js:39-56` | pass | The harness asserts `!denied \|\| sofuu.fs.existsSync(receipt) \|\| sofuu.fs.existsSync(starts)` → "MCP connection denial did not prevent dispatch" (line 42) and the same for the tool-level denial (line 55). The denied server never ran: the receipt file and the start marker are absent. |
| R3: the allowlist and the SSRF guard sit on the real boundary | read `crates/sofuu-core/src/rt/policy.rs` (`evaluate`, `gate_fs`, `gate_process_spawn`, `gate_net_host`, `net_policy`, `ip_is_internal`, `loopback_allowed`, `ip_denied`) and `crates/sofuu-core/src/rt/http_client.rs` | pass | The native fetch resolves the host and enforces `gate_net_host` + `ip_is_internal` at the resolved address; `src/js/web.js` only adds a JS-level guard (`guardInternalUrl`) for the `opts.endpoint`/`SOFUU_WEB_ENDPOINT` escape hatch — the native layer still applies the policy. `src/js/web.js` has no fake-success branch: every non-200 is a typed `ERR_WEB_SEARCH_HTTP` error. |
| R4: a disconnect after dispatch must report unknown completion, with no unsafe automatic replay | read `scripts/contract/harness/real_file_mcp_client.js:78-121` | pass | A lost response → `E_MCP_OUTCOME_UNKNOWN` (line 80); a queued call with a lost response → `E_MCP_OUTCOME_UNKNOWN` (line 89); a call denied while queued → `E_POLICY_DENIED` (line 103); after the reconnect window elapses `writes === 'recorded\nrecorded\n'` — the side-effecting call ran exactly once (line 110); a timed-out call must not run more than once (line 120). |
| R4: the JS MCP wrapper classifies an uncertain transport failure instead of replaying it | read `src/js/tools.js` (`unknownCompletion`, `mayHaveEffects`, `isUncertainTransportFailure`, `wrapResilient`) | pass | The resilient wrapper returns the unknown-completion error and does not re-dispatch a side-effecting call; the reconnect policy denial surfaces as `E_POLICY_DENIED` (`tools.js:3292`). |
| R5: no provider may be registered under a real-service name and return canned output | Node harness over the **real** `src/js/providers.js` source (`/tmp/sofuu-provider-src-check.js`) against a real local `llama-server` serving `~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf` on `127.0.0.1:8091` | pass | 8 passed / 0 failed. The `llamacpp` adapter returned the **model's own answer** — `"Ping!"` — with real counters `{promptTokens:42, completionTokens:4, totalTokens:46}` and `model:'smollm2'`; the server log agrees (`prompt eval time = 31.02 ms / 42 tokens`, `eval time = 15.34 ms / 4 tokens`). Negative cases stay errors: an unreachable host → `E_TOOL llamacpp: request failed — fetch failed`; the key-required `openai` adapter without a key → `E_DENIED openai: no API key — pass ctx.apiKey, set ctx.env.OPENAI_API_KEY, or export OPENAI_API_KEY`; a 404 from the server → `E_TOOL llamacpp: HTTP 404 — {"error":{"message":"File Not Found",…}}`. No canned text anywhere. |
| R5: the JS tool/plugin/provider layer contains no swallowed errors | `python3 scripts/check-js-swallow.py` | pass | `js-swallow: OK - 0 un-commented swallows, baseline 0` (exit 0). |
| R5: the provider registry is complete and every adapter is a real HTTP client | read `src/js/providers.js:1099-1116`, `src/js/providers.js:842-899`, `src/js/providers.js:978-1039` | pass | All five adapters (`ollama`, `llamacpp`, `openai`, `lmstudio`, `vllm`) issue real HTTP to their own wire (`/api/chat` + `/api/tags` + `/api/version`; `/v1/chat/completions` + `/models`); `lmstudio`/`vllm` are documented aliases of the OpenAI wire. `sofuu.providers.health` returns an honest `{ok:false,…}` when a probe fails. |
| The endpoint-configuration contract is unambiguous | Node harness over the real `src/js/providers.js` (`/tmp/sofuu-provider-env-check.js`) | pass (after the change below) | 7 passed / 0 failed: `a.envVar` still reports the canonical `LLAMA_CPP_HOST`; the canonical name derives the base URL; the alias `LLAMACPP_HOST` derives it too; the canonical name wins when both are set; a clean environment keeps the shipped default `http://127.0.0.1:8080`; both names really reach the live server (`"Ping!"`, `"PING (Packet Incoming Program"`). |
| The edited JS still compiles with the snapshot compiler | `./sofuu compile-bc src/js/tools.js …`, `… src/js/providers.js …`, `… src/js/plugins.js …` | pass | The same compiler `make bc-check` uses: `tools.js (187204 bytes)`, `providers.js (45200 bytes)`, `plugins.js (193438 bytes)`. The snapshots themselves were **not** touched (see "Notes for the coordinator"). |
| The Rust change compiles | `cargo check -p sofuu-core --lib` | pass | `Finished dev profile … in 18.20s`; 51 pre-existing warnings, none from the changed files. (`cargo check -p sofuu-core --lib --tests` was also clean earlier.) |
| The Rust change does not break any existing MCP harness | read `scripts/contract/harness/t72_policy_matrix.js`, `t73_approval_harness.js`, `t56_mcp_parity.js`, `t75_shared_shape.js` | pass | No harness installs `shell.allow` / `shell.enabled:false` while connecting an MCP server: `t72` only *evaluates* the policy, `t73` installs no policy, `t56` sets only `tools.deny`. |

## Gaps found and fixed

| Gap | Fix | Path:line |
|---|---|---|
| The built-in tool table did not enforce its own declared `required` arguments at the single enforcement point. `grep` without a `pattern` answered a successful-looking "no matches for /undefined/" instead of an error, and the listing, the agent loop and the executor could disagree about what a call may look like | `executeWithPolicy` now validates the parameters against the tool's declared schema with the same `schemaWalk` validator the agent loop uses (`src/js/agent.js:4930`), and raises a typed `E_SCHEMA` carrying `.path` and `.tool` **before** `toolDispatchBegin()` — invalid arguments never dispatch | `src/js/tools.js:328-347` |
| `sofuu.tools.execute` could not run a plugin-registered tool: `defineTool` publishes into `sofuu.tools.TOOLS` *after* `installPolicyWrappers` had snapshotted the private `TOOL_IMPLEMENTATIONS` table, so a registered tool was reported as "unknown tool" | the executor falls back to the late-registered tool's own `execute()` (the plugin driver already applies its own policy check and Shield-1 schema validation, so the outer gate does not re-validate it) | `src/js/tools.js:301-311` |
| The in-process MCP server's HTTP and direct transports executed a tool without validating its arguments, unlike the native stdio server, which answers `-32602` for invalid params | `mcpToolSchema()` resolves the schema a tool was registered with (`server.tool(name, {description, schema}, handler)` stores the opts object, so the schema is `opts.schema`; a tool registered without one stays permissive) and `mcpArgsViolation()` runs the shared validator; the HTTP transport returns a JSON-RPC error `{code:-32602, message:'Invalid params: <path> <reason>', data:{field,reason}}` and the direct transport throws the same error — in both cases the handler is not reached | `src/js/tools.js:2223-2245` (helpers), `src/js/tools.js:2051-2059` (HTTP `dispatch`), `src/js/tools.js:2469-2476` (direct `tools/call`) |
| An MCP server is a process the runtime starts, but `js_mcp_connect` spawned it **before** any process-policy gate: `SOFUU_NO_SHELL`, `shell.enabled:false` and `shell.allow` did not bind an MCP server command | `js_mcp_connect` now evaluates `policy::gate_process_spawn` on the server command **before** the child exists (before `build_child_env`, the client allocation and `uv_spawn`), so a refused server reaches no process; the refusal is reported as `MCP_ERR_POLICY_DENIED = -32006` with `data:{code,rule,command}` and is translated to the JS-side `E_POLICY_DENIED` (with `.rule` and `.decision='deny'`) by the raw stdio client. The helper mirrors the private `process_spawn::policy_command_lines` (out of this scope) and also offers the `sh -c <payload>` payload to the gate | `crates/sofuu-core/src/rt/mcp.rs:154` (code), `crates/sofuu-core/src/rt/mcp.rs:1245-1263` (helper), `crates/sofuu-core/src/rt/mcp.rs:1426-1444` (gate), `src/js/tools.js:3145-3154` (JS translation) |
| The llama.cpp endpoint was configured under two different environment names: the runtime and the adapter use the canonical `LLAMA_CPP_HOST` (`crates/sofuu-core/src/rt/ai.rs:123`, asserted by `tests/providers_adapters_test.js:185`), while the sibling plugin file and the scaffold template used `LLAMACPP_HOST`. A host that set the documented name got the default `http://127.0.0.1:8080` instead — on this machine a key-protected `llama-server`, which answers **HTTP 401** | the adapter accepts the canonical name first and the legacy spelling as an alias (`envVars`), the scaffold template reads the canonical name first, and `a.envVar` still reports `LLAMA_CPP_HOST` | `src/js/providers.js:108-118` (`envFirst`), `src/js/providers.js:227-231` (`baseUrlOf`), `src/js/providers.js:860`, `src/js/providers.js:899`, `src/js/providers.js:1077-1082` (adapter), `src/js/plugins.js:4038` (scaffold template) |

Two new Rust unit tests were added with the last fix
(`crates/sofuu-core/src/rt/mcp.rs:3600-3660`): one asserts that a connect
refused by the process policy answers `REFUSED:-32006:…` and names the
`R7.6/shell.allow` rule; the other is a pure check that the `sh -c <payload>`
form is offered to the gate. Both are **type-checked only** (build policy).

## Register rows

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D04 tools, MCP, policy | Built-in tool execution (`sofuu.tools.execute` → the tool's real implementation) | `real` | The executor resolves the implementation and returns its real output; the policy verdict is resolved before dispatch (`src/js/tools.js:328-347`). Verified against the real source in a Node harness (6 passed / 0 failed). |
| D04 tools, MCP, policy | Tool argument contract (the declared schema is enforced) | `real` | `schemaWalk` at the single enforcement point raises `E_SCHEMA` with `.path`/`.tool` before dispatch (`src/js/tools.js:328-347`); the same validator already ran in the agent loop (`src/js/agent.js:4930`). Verified: `grep` without a pattern is rejected with `path='$.pattern'`. |
| D04 tools, MCP, policy | Plugin-registered tool reachable through `sofuu.tools.execute` | `real` | Late-registered tools fall back to their own `execute()` (`src/js/tools.js:301-311`); verified: a plugin tool registered after the wrappers were installed returns `'ran:7'` instead of "unknown tool". |
| D04 tools, MCP, policy | MCP stdio client (real child process, JSON-RPC over pipes) | `real` | `scripts/test-real-mcp-file.py` phase 1 PASS against the pre-change artifact; the native client is `uv_spawn` + pipes (`crates/sofuu-core/src/rt/mcp.rs`). |
| D04 tools, MCP, policy | MCP Streamable-HTTP client | `real` | `scripts/test-real-mcp-file.py` phase 2 PASS against the pre-change artifact (HTTP against a Python `ThreadingHTTPServer` on 127.0.0.1): real file read, policy denial, ping, deadlines, circuit breaker. |
| D04 tools, MCP, policy | MCP server transports (in-process HTTP and direct) validate arguments like the stdio server | `real` | `mcpArgsViolation` + `-32602` on both transports, handler not reached (`src/js/tools.js:2223-2245`, `2051-2059`, `2469-2476`). Verified with a Node harness over the real source (5 passed / 0 failed: missing required arg → `-32602`, mistyped arg → `-32602`, handler not run, valid call reaches the handler). **No contract harness drives these two transports**, so the end-to-end path is unverified. |
| D04 tools, MCP, policy | MCP server spawn bound by the process policy (`SOFUU_NO_SHELL`, `shell.enabled`, `shell.allow`) | `real` | `gate_process_spawn` runs before the child exists → zero dispatch; refusal code `-32006` → `E_POLICY_DENIED` (`crates/sofuu-core/src/rt/mcp.rs:1426-1444`, `src/js/tools.js:3145-3154`). **Behaviour unverified end-to-end** — the unit test is type-checked only and the shipped binary predates the change. |
| D04 tools, MCP, policy | A denied operation produces zero dispatch (connection denial, tool denial) | `real` | The harness asserts the **absence** of the side-effect receipt and the start marker, not the presence of a denial message (`scripts/contract/harness/real_file_mcp_client.js:42`, `:55`). PASS against the pre-change artifact. |
| D04 tools, MCP, policy | Unknown completion after a disconnect, and no automatic replay | `real` | `E_MCP_OUTCOME_UNKNOWN` on a lost response, on a queued call with a lost response and on a timeout (`real_file_mcp_client.js:80`, `:89`, `:115`); the side-effecting call ran exactly once (`writes === 'recorded\nrecorded\n'`, `:110`). PASS against the pre-change artifact. |
| D04 tools, MCP, policy | SSRF guard on the network boundary | `real` | The native fetch enforces `gate_net_host` + `ip_is_internal` at the resolved address (`crates/sofuu-core/src/rt/http_client.rs`); `src/js/web.js` adds a JS-level guard (`guardInternalUrl`) but no bypass — the `opts.endpoint`/`SOFUU_WEB_ENDPOINT` escape hatch skips the JS guard only, the native policy still applies. No fake-success branch in `web.js` (every non-200 is `ERR_WEB_SEARCH_HTTP`). |
| D04 tools, MCP, policy | Provider adapters return real answers, never canned output | `real` | Real inference against a local `llama-server` serving `~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf`: the adapter returned the model's own `"Ping!"` with real token counters, corroborated by the server-side log. Unreachable host → `E_TOOL`; missing API key → `E_DENIED`; HTTP 404 → `E_TOOL` with the server body. All five adapters are real HTTP clients (`src/js/providers.js:1099-1116`). |
| D04 tools, MCP, policy | The provider registry as shipped in the binary | `unverified` | The shipped `target/release/sofuu` (built 12:54) embeds an **older** provider driver: it ignores `LLAMA_CPP_HOST` and fell back to `http://127.0.0.1:8080`, where the developer's key-protected `llama-server` answers HTTP 401. The current source honours the canonical name (verified). A rebuilt binary is needed to close this row. |
| D04 tools, MCP, policy | One provider per real-service name | `unverified` | Two registrations exist under the name `llamacpp`: the first-party JS adapter (`src/js/providers.js`) and the discovered `plugins/llamacpp_provider.ts`, which overwrites the entry when `plugins/` is discovered (`src/js/plugins.js:3512` `P.discover` → `registerProvider` → `P.PROVIDERS[name] = spec`, `src/js/plugins.js:1619-1626`). Neither returns canned output, but the two disagree about the endpoint name. `plugins/llamacpp_provider.ts` is outside this worker's scope — an owner must reconcile it. |

## Blocked or unverified

- **The MCP harness was run against the pre-change artifact.** The build is the
  coordinator's, so `scripts/test-real-mcp-file.py` cannot be re-run against the
  edited Rust/JS until the binary is rebuilt. The MCP rows above are `real` for
  the code path and for the pre-change artifact; they are **not** yet receipts
  for the changed code.
- **The new Rust unit tests are type-checked only** (`cargo check -p sofuu-core
  --lib --tests`, clean). They were not executed (`cargo test` is out of policy
  for this worker).
- **The in-process MCP server's HTTP and direct transports have no contract
  harness.** Their argument validation is verified against the real source in a
  Node harness, but nothing drives them end-to-end through the runtime.
- **`additionalProperties:false` is enforced by the native stdio server but not
  by the JS transports.** `schemaWalk` implements the type/required/properties/
  items/enum subset; `additionalProperties` is outside it. Documented limitation,
  not silently ignored.
- **The provider registry was not exercised end-to-end through the shipped
  binary** — the shipped binary is stale (see the register rows). The real
  inference receipt is against the current `src/js/providers.js` source.
- **`crates/sofuu-core/src/rt/policy.rs` carries concurrent edits from another
  worker.** This worker did not modify it; the MCP spawn gate only *calls* its
  public `gate_process_spawn`.
- The two Node harnesses and the provider scripts live in `/tmp`
  (`/tmp/sofuu-tools-check/check.js`, `/tmp/sofuu-tools-check/mcp-check.js`,
  `/tmp/sofuu-provider-src-check.js`, `/tmp/sofuu-provider-env-check.js`,
  `/tmp/sofuu-provider-check.js`). `/tmp` is ephemeral — they must be recreated
  from this report before the next verification round.

## Notes for the coordinator

- **Edited JS drivers — the bytecode snapshots are now stale and must be
  regenerated.** Run `scripts/gen-driver-bc.sh` and rebuild, then `make bc-check`:
  - `src/js/tools.js` (schema gate, late-registered fallback, MCP transports, `-32006` translation)
  - `src/js/providers.js` (env-var alias)
  - `src/js/plugins.js` (scaffold template env var)
  `src/js/tools.bc`, `src/js/providers.bc`, `src/js/plugins.bc` (and their
  `.bc.gz` twins) were **not** hand-edited and still hold the previous
  snapshots.
- **The Rust change needs a rebuild** before `python3 scripts/test-real-mcp-file.py`
  and `cargo test -p sofuu-core` are re-run (the two new tests live in
  `crates/sofuu-core/src/rt/mcp.rs`).
- **`crates/sofuu-core/src/rt/mcp.rs` also carries concurrent edits from another
  worker** (hunks around the JSON-RPC read callback, `mcp_json_text`,
  `mcp_success_body`, `mcp_server_handle_message` and three test blocks). The
  hunks that are this worker's are the ones listed in "Gaps found and fixed";
  `cargo check -p sofuu-core --lib` is clean with both sets applied.
- **Out-of-scope fix that needs an owner:** `plugins/llamacpp_provider.ts:99`
  reads only `LLAMACPP_HOST`; it should read the canonical `LLAMA_CPP_HOST`
  (keeping the old spelling as an alias). Until then the discovered plugin
  shadows the first-party `llamacpp` adapter and ignores the documented
  configuration name — the exact trap that produced the HTTP 401 recorded
  above.
- **`contract/CAPABILITY-REGISTER.md` was not edited by this worker.** The D04
  rows there cite `scripts/test-real-mcp-file.py` as the MCP receipt; that
  receipt is currently the pre-change artifact and should be re-issued after
  the rebuild.
- `crates/sofuu-core/src/rt/ai.rs` was being edited concurrently earlier in
  this session (transient `cargo check` errors at `ai.rs:3463`/`3476`); those
  have cleared, and the last `cargo check -p sofuu-core --lib` is clean.
