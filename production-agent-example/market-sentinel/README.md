# 24/7 Market Sentinel (Stocks, Crypto & Currency Monitor)

Real-time 24/7 monitoring agent built with pure JS on Sofuu. Fetches live market prices and OHLCV candles, evaluates built-in + custom user algorithms, combines mathematical signals with your AI model ([`models.json`](models.json)), and dispatches target-price & opportunity alerts.

## Runtime Primitives (PLAN-AUTONOMOUS-REACTIVE-RUNTIME)

Market Sentinel is also the reference consumer of the runtime's reactive
building blocks, so the example carries no hand-rolled loop, cooldown file or
plugin loader of its own:

- **`sofuu.triggers`** — the `--watch`/`--daemon` mode registers one
  `interval` runner. Its **`gate()`** does the deterministic fetch + algorithm
  scan and returns `null` on a quiet cycle, so **no model call is made and no
  tokens are spent**; only a non-null payload reaches **`onTrigger()`**, which
  synthesizes and dispatches alerts. The runner also supplies the
  non-overlap lock, jitter, exponential backoff and optional
  `budget.maxAiCallsPerDay` cap.
- **`sofuu.state`** — cooldowns are atomic check-and-stamp entries in
  `.sofuu/state/market-sentinel.json` (namespace `market-sentinel`), not a
  hand-written JSON file. Two overlapping cycles cannot both alert, and the
  stamps survive a restart.
- **`sofuu.evaluators`** — `algos/*.js` is loaded through
  `sofuu.evaluators.loadDir()`, which validates the `{ id, evaluate }` contract
  and reports a broken drop-in file without taking down the 24/7 loop.
- **Daemon parity** — `sofuu daemon run|install` keeps the trigger loop in one
  resident process and exports `SOFUU_DAEMON_INTERVAL_MS`; specifying
  `--every` on the CLI still wins.

## Key Files
- [`index.js`](index.js): reactive runner, single-scan, interactive chat, and CLI entrypoint
- [`models.json`](models.json): Dedicated provider & model config for Market Sentinel
- [`watchlist.json`](watchlist.json): Assets (`stock`, `crypto`, `currency`), target prices, cooldowns, notification settings
- [`algos/`](algos/): Drop-in user evaluators (see [`algos/example_vwap_dip.js`](algos/example_vwap_dip.js))
- [`lib/feeds.js`](lib/feeds.js): Live data feeds (Coinbase Exchange + Yahoo Finance + Open ER)
- [`lib/algorithms.js`](lib/algorithms.js): Built-in indicators (`rsi_reversion`, `ema_crossover`, `bollinger_squeeze`, `zscore_breakout`) + `sofuu.evaluators` loader
- [`lib/notifier.js`](lib/notifier.js): `sofuu.state` cooldown deduplication + Console, macOS Desktop Notification, Webhook, JSONL log, CMA memory
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

# Native 24/7 service (one resident process, launchd/systemd managed)
./sofuu daemon install production-agent-example/market-sentinel/index.js --every 60s --name market-sentinel

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

## Reactive Runtime Primitives

The example uses the generalized autonomous-runtime primitives directly; there
is no domain-specific loop, cooldown file, or plugin loader left in userland:

| Concern | Core primitive | What replaced it here |
| :--- | :--- | :--- |
| Continuous loop | `sofuu.triggers.register` | hand-rolled `while (!stopping)`, `sleep(intervalMs)`, two `SIGINT`/`SIGTERM` handlers and the retry `try/catch` block |
| Pre-LLM gate | `gate(ctx)` returns `null` for a quiet scan | a model call on every poll; a quiet scan now spends **0 tokens** |
| Cooldowns / dedupe | `sofuu.state.open('market-sentinel').allowAfterCooldown` | `loadCooldownState` / `saveCooldownState` JSON file reads, mkdir, parse/stringify and `markFired` bookkeeping |
| Custom algorithms | `sofuu.evaluators.loadDir` | a hand-written `readdirSync` + `import()` + contract-check loop |

`lib/notifier.js` now makes one atomic `allowAfterCooldown(key, ms)` call per
trigger instead of reading a JSON file, testing a timestamp, marking it and
writing the file back. `index.js` now registers one interval runner with
`nonOverlapping`, `jitterPct` and exponential `backoff` and calls
`await runner.start()` instead of managing a `stopping` flag and timer.
Together the custom cooldown + loop boilerplate dropped from ~42 lines to ~16
(about 62%), with jitter, overlap prevention and backoff added rather than
removed. `SOFUU_DAEMON_INTERVAL_MS` from `sofuu daemon install` is honoured
automatically when `--every` is absent.

Error codes are structured: bad trigger specs raise `E_TRIGGER_INVALID`, a
malformed drop-in rule is reported as `E_EVALUATOR_CONTRACT`, and storage
failures raise `E_STATE_IO`.

## Messaging Channels

Alerts and two-way commands use the generalized `sofuu.channels` layer — this
example contains no Telegram/WhatsApp/HTTP transport code.

```bash
# Configure Telegram, WhatsApp, a webhook or desktop notifications
./sofuu run production-agent-example/market-sentinel/index.js --setup-channel

# Watch + answer `status`, `scan`, or free-form market questions from chat
./sofuu run production-agent-example/market-sentinel/index.js --watch --listen

# WhatsApp/webhook inbound needs the loopback gateway + your tunnel/proxy
./sofuu gateway --channels-config production-agent-example/market-sentinel/channels.json
```

`watchlist.json`'s `notifications.channels` defaults to `true`: when a channel
profile exists, alerts go through `sofuu.channels.send` with Acknowledge/Mute
buttons. The legacy `desktop` and `webhookUrl` settings remain as fallbacks for
installations that have no `channels.json`.
