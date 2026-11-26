# Sofuu Production Agent Examples

This directory contains **two completely separate, standalone, production-grade AI agents** built with pure JavaScript on the Sofuu runtime. Each agent has its own isolated folder, `models.json`, configuration, tools/algorithms, and test suite:

| Agent Folder | Purpose | Entrypoint |
| :--- | :--- | :--- |
| **[`code-sentinel/`](code-sentinel/README.md)** | Autonomous Multi-Agent Code Review & Security Auditor (Supervisor + Security/Quality specialists + deterministic AST rules) | [`code-sentinel/index.js`](code-sentinel/index.js) |
| **[`market-sentinel/`](market-sentinel/README.md)** | 24/7 Real-Time Stock, Crypto & Currency Monitor (Live feeds + extensible quantitative algorithms + AI synthesis + multi-channel alerts) | [`market-sentinel/index.js`](market-sentinel/index.js) |

---

## Quick Start

### 1. Code Sentinel ([`code-sentinel/`](code-sentinel/README.md))
```bash
./sofuu run production-agent-example/code-sentinel/index.js --list-models
./sofuu run production-agent-example/code-sentinel/index.js --interactive
./sofuu run production-agent-example/code-sentinel/tests/test_sentinel.js --models-config production-agent-example/code-sentinel/models.json
```

### 2. 24/7 Market Sentinel ([`market-sentinel/`](market-sentinel/README.md))
```bash
./sofuu run production-agent-example/market-sentinel/index.js --list-models
./sofuu run production-agent-example/market-sentinel/index.js --once
./sofuu run production-agent-example/market-sentinel/index.js --watch --every 30s
./sofuu run production-agent-example/market-sentinel/index.js --interactive
./sofuu run production-agent-example/market-sentinel/tests/test_market_monitor.js --models-config production-agent-example/market-sentinel/models.json
```

Both agents support Sofuu's generalized two-way provider configuration (per-agent `models.json` + `--setup-model` / `--edit-model` / `--list-models`, including when compiled with `sofuu compile`).
