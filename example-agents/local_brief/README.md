# Local Brief

> **Real, read-only example.** The routine reads one operator-selected local
> file through an authorized MCP tool and returns a real model's answer about
> that file. It writes nothing, places no order, touches no patient record and
> drives no actuator. Missing dependencies fail closed; nothing is simulated.

## What it does

`agent.kaze` is a pure Kaze routine (no JS sidecar, no shell wrapper). It
calls one MCP tool, `read_document`, and passes the returned text to
`ai.prompt`. The MCP service in `mcp/read_document_server.js` is a **separate
external MCP server**, not part of the agent: it is bound to the single file
its operator selected on the command line and only reads that file. Removing
it makes the run fail closed with `E_CAP_MISSING (E_MCP_TOOL_NOT_FOUND)`.

## Real dependencies

| Dependency | Requirement |
|---|---|
| Sofuu binary | A build that contains the Kaze asynchronous bridge (`sofuu script run`, `sofuu.kaze.runAsync`). The published binaries have not been verified for it. |
| Model | A real local GGUF file (for example `~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf`) with `SOFUU_AI_PROVIDER=local`, or a configured remote provider with its own credentials. |
| MCP service | The `local-brief-read` stdio server started from `mcp/read_document_server.js`, registered in `~/.sofuu/mcp.json` (or in a file named by `SOFUU_MCP_CONFIG`). |
| Policy | An operator policy that allows `kaze.log.*`, `kaze.ai.*`, `kaze.tool.*` and the `read_document` tool. `policy.json` in this directory is exactly that. |
| Permissions | The grants (`--cap ai --cap tool --cap log`) and the policy above. The example never asks for `memory`, `alert`, `sensor` or `device`. |

## Run it

Check the source without executing it:

```bash
sofuu script check example-agents/local_brief/agent.kaze
```

Run it against a real model and the real MCP service:

```bash
export SOFUU_AI_PROVIDER=local
export SOFUU_AI_MODEL="$HOME/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf"
export SOFUU_MCP_CONFIG="$PWD/example-agents/local_brief/mcp.local.json"

sofuu script run example-agents/local_brief/agent.kaze \
  --cap ai --cap tool --cap log \
  --policy "$PWD/example-agents/local_brief/policy.json" \
  --config "$PWD/example-agents/local_brief/agent.config.json"
```

`mcp.local.json` is a copy of `mcp.example.json` with two values replaced: the
path of the Sofuu binary to spawn and the path of the document to read. It
holds machine-specific paths and belongs in your own checkout only. The
spawned server inherits the working directory of the Sofuu process, so the
relative script path in `mcp.example.json` is resolved from where you run the
command.

The compiled standalone binary follows the same path:

```bash
sofuu compile example-agents/local_brief/agent.kaze -o local-brief
./local-brief --cap ai --cap tool --cap log \
  --policy "$PWD/example-agents/local_brief/policy.json"
```

## Configuration hierarchy

The routine takes one parameter, `instruction`, and resolves it in the
production order: CLI `--arg` or a positional, then `AGENT_INSTRUCTION` /
`SOFUU_INSTRUCTION`, then the configuration file (`--config <path>` or
`./agent.config.json`), then the routine signature. A missing parameter
produces a structured diagnostic naming all four sources, not a runtime
arithmetic error.

## What was actually observed

On 2026-11-16, on this machine, with locally built debug artifacts of
`target/debug/sofuu` (SHA-256 `b9669d547619329c35295cef7a230d83feb4a60c17e1be6e2008b720d3a83153`
for the morning runs and `51ca0b0d3db7462d6c5e17e6fed902b7e155935acae68ba332a2d57697189c5b`
for the final re-run; both built from a working tree at revision `853d01325`
with uncommitted changes) and the pre-existing local weights
`~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf` (386,404,960 bytes, loaded on
the Metal GPU; no weights were downloaded):

* the source path returned a real model answer that quoted the document read
  through the MCP tool, and the report carried `ok: true` with an empty
  `simulated_families` list;
* the compiled standalone binary returned a real model answer for the same
  document;
* with no model configured the run stopped with `E_CAP_MISSING` at
  `agent.kaze:14`; with no MCP server configured it stopped with
  `E_CAP_MISSING (E_MCP_TOOL_NOT_FOUND)` at `agent.kaze:13`; and with a policy
  that does not allow `kaze.tool.call` it stopped with `E_CAP_DENIED` at
  `agent.kaze:13`. None of those runs produced an answer.

The model is a 360M-parameter local model. Its wording is its own; the
verifiable facts are that the MCP service read the operator's file, that the
model was loaded from real local weights, and that the answer came back
through the model request. A model that answers badly is a bad answer, not a
missing one.
