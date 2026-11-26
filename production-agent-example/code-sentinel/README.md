# Code Sentinel (Multi-Agent Security & Code Intelligence)

Autonomous multi-agent codebase auditor built with pure JS on Sofuu (`code_sentinel` supervisor + `security_scanner` & `quality_expert` specialists + deterministic AST rules).

## Key Files
- [`index.js`](index.js): CLI & programmatic SDK entrypoint
- [`models.json`](models.json): Dedicated provider & model config for Code Sentinel
- [`agent.config.json`](agent.config.json): Budgets, audit thresholds, memory settings
- [`lib/sentinel.js`](lib/sentinel.js), [`lib/specialists/`](lib/specialists/), [`lib/tools/`](lib/tools/), [`lib/memory.js`](lib/memory.js)
- [`tests/test_sentinel.js`](tests/test_sentinel.js): Verification suite

## Usage
```bash
# Configure or inspect provider/model
./sofuu run production-agent-example/code-sentinel/index.js --setup-model
./sofuu run production-agent-example/code-sentinel/index.js --list-models

# Interactive chat
./sofuu run production-agent-example/code-sentinel/index.js --interactive

# Headless audit / CI JSON gate
./sofuu run production-agent-example/code-sentinel/index.js --target ./src --json

# Compile to standalone binary
./sofuu compile production-agent-example/code-sentinel/index.js -o ./code-sentinel

# Run tests
./sofuu run production-agent-example/code-sentinel/tests/test_sentinel.js --models-config production-agent-example/code-sentinel/models.json
```

## Messaging Channels

Code Sentinel uses the generalized `sofuu.channels` layer — there is no
Telegram/WhatsApp/HTTP code in this example.

```bash
# Configure one or more channels into channels.json
./sofuu run production-agent-example/code-sentinel/index.js --setup-channel

# Send the completed audit summary through the active channel
./sofuu run production-agent-example/code-sentinel/index.js --target ./src --notify

# Two-way: answer audit requests from Telegram (or a gateway)
./sofuu run production-agent-example/code-sentinel/index.js --listen
```

`--listen` uses outbound Telegram long polling (zero public ports). For
WhatsApp Cloud API or generic webhooks, run `sofuu gateway` (loopback by
default) in front of the same `channels.json`. Remote senders run the sentinel
with the `plan` permission profile — read-only analysis, no writes or shell.
