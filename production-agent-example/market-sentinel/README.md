# 24/7 Market Sentinel (Stocks, Crypto & Currency Monitor)

Real-time 24/7 monitoring agent built with pure JS on Sofuu. Fetches live market prices and OHLCV candles, evaluates built-in + custom user algorithms, combines mathematical signals with your AI model ([`models.json`](models.json)), and dispatches target-price & opportunity alerts.

## Key Files
- [`index.js`](index.js): 24/7 watch loop, single-scan, interactive chat, and CLI entrypoint
- [`models.json`](models.json): Dedicated provider & model config for Market Sentinel
- [`watchlist.json`](watchlist.json): Assets (`stock`, `crypto`, `currency`), target prices, cooldowns, notification settings
- [`algos/`](algos/): Drop-in user custom algorithms (see [`algos/example_vwap_dip.js`](algos/example_vwap_dip.js))
- [`lib/feeds.js`](lib/feeds.js): Live data feeds (Coinbase Exchange + Yahoo Finance + Open ER)
- [`lib/algorithms.js`](lib/algorithms.js): Built-in indicators (`rsi_reversion`, `ema_crossover`, `bollinger_squeeze`, `zscore_breakout`) + dynamic custom loader
- [`lib/notifier.js`](lib/notifier.js): Cooldown deduplication + Console, macOS Desktop Notification, Webhook, JSONL log, CMA memory
- [`lib/agent.js`](lib/agent.js): AI alert synthesis + `market_sentinel` interactive agent tools
- [`tests/test_market_monitor.js`](tests/test_market_monitor.js): Verification suite

## Usage
```bash
# Configure or inspect provider/model
./sofuu run production-agent-example/market-sentinel/index.js --setup-model
./sofuu run production-agent-example/market-sentinel/index.js --list-models

# Single live scan
./sofuu run production-agent-example/market-sentinel/index.js --once

# Continuous 24/7 monitor loop (polls every 30s)
./sofuu run production-agent-example/market-sentinel/index.js --watch --every 30s

# Interactive AI chat with live tools
./sofuu run production-agent-example/market-sentinel/index.js --interactive

# Add or remove target alerts
./sofuu run production-agent-example/market-sentinel/index.js --add SOL-USD --type crypto --above 200 --below 140
./sofuu run production-agent-example/market-sentinel/index.js --list-algos

# Compile to standalone binary
./sofuu compile production-agent-example/market-sentinel/index.js -o ./market-sentinel

# Run tests
./sofuu run production-agent-example/market-sentinel/tests/test_market_monitor.js --models-config production-agent-example/market-sentinel/models.json
```
