# 📊 Paper Trader — Simulated Crypto & Stock Market Analysis & Paper Execution AI Agent

> **An autonomous, 100% Pure Kaze AI agent designed for simulated paper trading, real-time technical & sentiment analysis, and risk-managed execution for Crypto and Stocks. Strictly simulated paper execution — NO REAL MONEY.**

---

## 1. Overview

**Paper Trader** is an autonomous market analysis and simulated paper trading agent programmed entirely in Kaze with zero JavaScript or TypeScript glue code. It automates the full trading desk lifecycle in a risk-managed, simulated environment:
* **Pre-Trade Risk Firewalls (`policy`)**: Enforces hard paper risk caps (max 25% allocation per position), positive price sanity checks, liquidity minimums ($50,000 24h volume), and credential shielding.
* **AI Technical & Sentiment Analysis (`ai`)**: Synthesizes support/resistance levels, sentiment ratings, and risk/reward metrics.
* **Simulated Paper Execution (`tool`)**: Dispatches paper orders to a simulated exchange tool (`paper_exchange`) without exposing real capital.
* **Persistent Trade Journal (`memory`)**: Retains every paper trade thesis and portfolio equity checkpoint in associative vector memory.
* **Audit & Risk Escalation (`log`, `alert`)**: Logs all simulated transactions and triggers alerts for large paper allocations ($\ge 20\%$).

---

## 2. Capabilities Manifest

```kaze
needs: ai, tool, policy, memory, log, alert
```

| Capability | Role in Paper Trader |
| :--- | :--- |
| **`ai`** | Technical analysis, sentiment scoring, and trade rationale synthesis (`ai.prompt`, `ai.complete`). |
| **`tool`** | Simulated paper exchange routing (`paper_exchange`) and portfolio tracking (`paper_portfolio_tracker`). |
| **`policy`** | R7 security firewall for API credential leak detection and risk limit compliance. |
| **`memory`** | Cross-session paper trade journal and strategy performance memory (`remember ... as ...`). |
| **`log`** | Institutional-grade paper trade execution audit logs (`log.info`). |
| **`alert`** | High-allocation alerts and risk limit breach notifications (`send alert`). |

---

## 3. Compilation to Standalone Native Binary

Compile the agent directly from source into a single self-contained executable (~4.5 MiB):

```bash
sofuu compile example-agents/paper_trader/agent.kaze -o example-agents/paper_trader/paper-trader
```

The compiled binary includes:
* Bounded linear arena VM (zero garbage collection stutters).
* Stream-aligned ANSI badges (`[AGENT]`, `[AI]`, `[TOOL]`, `[POLICY]`, `[MEMORY]`, `[VERDICT]`).
* Full CLI parameter forwarding (`--entry`, `--arg`, `--fuel`).

---

## 4. Execution Examples & Verification

### Test 1: Valid Simulated Trade Execution (Allowed)

Simulate analyzing and executing a paper trade on Bitcoin with 15% allocation:

```bash
./example-agents/paper_trader/paper-trader \
  --arg asset=BTC-USDT \
  --arg current_price=75893 \
  --arg volume_24h=367000000 \
  --arg sentiment_score=50 \
  --arg requested_action=BUY \
  --arg requested_allocation_pct=15
```

**Live Trace Output:**
```
[AGENT] agent :: analyze_and_trade
[STEP] log.info("Paper Trader analyzing BTC-USDT (BUY at $75893, allocation=15%)") -> nothing
[POLICY] policy.detects("secrets", "BTC-USDT") -> false
[AI] ai.prompt("Act as Paper Trader market analyst. Analyze BTC-USDT at price $75893 with 24h volume $367000000 and sentiment index 50 for simulated action: BUY. Calculate support/resistance levels, target price, stop-loss, and simulated trade rationale. Note: strictly paper trading simulation.") -> "Synthesized autonomous decision..."
[MEMORY] memory.save("BTC-USDT_paper_thesis", ...) -> nothing
[TOOL] tool.call("paper_exchange", "BTC-USDT:BUY@75893;allocation=15%") -> OK
[INFO] Paper Trader analyzing BTC-USDT (BUY at $75893, allocation=15%)
[INFO] Simulated paper order status: OK: tool 'paper_exchange' executed
[VERDICT] ALLOW "Synthesized autonomous decision..."
94 instructions · 0 fuel
```

**Expected Live Trace:**
```
[AGENT] agent :: analyze_and_trade
[POLICY] policy.detects("secrets", "BTC/USD") -> false
[AI] ai.prompt("Act as Paper Trader market analyst. Analyze BTC/USD at price $68450...")
[MEMORY] memory.save("BTC/USD_paper_thesis", ...) -> nothing
[TOOL] tool.call("paper_exchange", "BTC/USD:BUY@68450;allocation=15%") -> OK
[INFO] Paper Trader analyzing BTC/USD (BUY at $68450, allocation=15%)
[INFO] Simulated paper order status: OK: tool 'paper_exchange' executed
[VERDICT] ALLOW "Synthesized autonomous decision..."
```

---

### Test 2: Allocation Risk Breach Guardrail (Rejected)

Attempting to allocate 35% of the paper portfolio into a single position (exceeding the 25% safety cap):

```bash
./example-agents/paper_trader/paper-trader \
  --entry analyze_and_trade \
  --arg 'asset="SOL/USD"' \
  --arg 'current_price=185' \
  --arg 'volume_24h=850000000' \
  --arg 'sentiment_score=82' \
  --arg 'requested_action="BUY"' \
  --arg 'requested_allocation_pct=35'
```

**Expected Output (Fail-Closed Rejection in 44 instructions):**
```
[AGENT] agent :: analyze_and_trade
[POLICY] policy.detects("secrets", "SOL/USD") -> false
[ALERT] alert.send("RISK BREACH: Requested paper allocation (35%) exceeds 25% risk cap on SOL/USD")
[VERDICT] REJECT Risk limit violation: maximum single paper position allocation is 25%
```

---

### Test 3: Credential Leak Prevention Guardrail (Rejected)

If an automated signal feeds private API keys inside the asset identifier:

```bash
./example-agents/paper_trader/paper-trader \
  --entry analyze_and_trade \
  --arg 'asset="ETH-TOKEN-sk-ant-api03-secretkey123"' \
  --arg 'current_price=2600' \
  --arg 'volume_24h=500000000' \
  --arg 'sentiment_score=65' \
  --arg 'requested_action="BUY"' \
  --arg 'requested_allocation_pct=10'
```

**Expected Output:**
```
[AGENT] agent :: analyze_and_trade
[POLICY] policy.detects("secrets", ...) -> true
[VERDICT] REJECT Security refusal: asset identifier or payload contains private credentials
```

---

### Test 4: Illiquid Asset Rejection (Rejected)

If an asset has 24h volume below $50,000, preventing slippage simulation errors:

```bash
./example-agents/paper_trader/paper-trader \
  --entry analyze_and_trade \
  --arg 'asset="MICRO-CAP-COIN"' \
  --arg 'current_price=0.04' \
  --arg 'volume_24h=12000' \
  --arg 'sentiment_score=50' \
  --arg 'requested_action="BUY"' \
  --arg 'requested_allocation_pct=5'
```

**Expected Output:**
```
[AGENT] agent :: analyze_and_trade
[VERDICT] REJECT Illiquid asset refusal: 24h volume below simulated execution threshold ($50,000)
```

---

### Test 5: Paper Portfolio Performance Review (`action review_portfolio`)

Review simulated equity and rebalance recommendations:

```bash
./example-agents/paper_trader/paper-trader \
  --entry review_portfolio \
  --arg 'total_paper_equity=125400' \
  --arg 'active_positions="BTC: 40%, ETH: 25%, SOL: 15%, CASH: 20%"'
```

---

### Test 6: Strategy Backtesting (`action backtest_strategy`)

Run historical simulation over custom timeframes:

```bash
./example-agents/paper_trader/paper-trader \
  --entry backtest_strategy \
  --arg 'asset="BTC/USD"' \
  --arg 'timeframe_days=90' \
  --arg 'strategy_name="EMA_Cross_20_50"'
```

---

### Test 7: Autonomous Position Management & Take-Profit / Stop-Loss (`action manage_position`)

Monitor open paper positions and automatically trigger Take Profit (+2%) or Stop Loss (-2%):

#### A. Take Profit Triggered (+2% Profit Target Reached)
```bash
./example-agents/paper_trader/paper-trader \
  --entry manage_position \
  --arg asset=BTC-USDT \
  --arg current_price=76500 \
  --arg entry_price=75000 \
  --arg position_allocation_pct=20
```
**Expected Trace:**
```
[AGENT] agent :: manage_position
[INFO] Position Manager checking BTC-USDT at $76500 (Entry: $75000)
[INFO] TAKE PROFIT triggered on BTC-USDT (+2% profit target reached)
[TOOL] tool.call("paper_exchange", "BTC-USDT:SELL@76500;reason=TAKE_PROFIT")
[STEP] alert.send("PROFIT TAKEN: Closed BTC-USDT position at $76500")
57 instructions · 0 fuel
```

#### B. Stop Loss Triggered (-2% Risk Threshold Hit)
```bash
./example-agents/paper_trader/paper-trader \
  --entry manage_position \
  --arg asset=BTC-USDT \
  --arg current_price=73000 \
  --arg entry_price=75000 \
  --arg position_allocation_pct=20
```
**Expected Trace:**
```
[AGENT] agent :: manage_position
[INFO] Position Manager checking BTC-USDT at $73000 (Entry: $75000)
[INFO] STOP LOSS triggered on BTC-USDT (-2% risk threshold hit)
[TOOL] tool.call("paper_exchange", "BTC-USDT:SELL@73000;reason=STOP_LOSS")
[STEP] alert.send("STOP LOSS: Closed BTC-USDT position at $73000 to preserve capital")
65 instructions · 0 fuel
```

#### C. In Corridor (Hold Position)
```bash
./example-agents/paper_trader/paper-trader \
  --entry manage_position \
  --arg asset=BTC-USDT \
  --arg current_price=75200 \
  --arg entry_price=75000 \
  --arg position_allocation_pct=20
```
**Expected Trace:**
```
[AGENT] agent :: manage_position
[INFO] Position Manager checking BTC-USDT at $75200 (Entry: $75000)
[INFO] HOLD: BTC-USDT price within acceptable risk corridor
36 instructions · 0 fuel
```

---

## 24/7 Autonomous Paper Trading Execution

Run Paper Trader continuously around the clock with one command:

### 1. Foreground Supervisor Loop (`--daemon` / `--every`)
```bash
./example-agents/paper_trader/paper-trader \
  --daemon \
  --every 5s \
  --arg asset=BTC-USDT \
  --arg current_price=75893 \
  --arg volume_24h=367000000 \
  --arg sentiment_score=50 \
  --arg requested_action=HOLD \
  --arg requested_allocation_pct=15
```

### 2. Auto-Installed Persistent Background Service (`sofuu daemon install`)
```bash
# Install as launchd (macOS) or systemd (Linux) background service:
sofuu daemon install ./example-agents/paper_trader/paper-trader \
  --name btc-paper-trader \
  --every 10s \
  --arg asset=BTC-USDT \
  --arg current_price=75893 \
  --arg volume_24h=367000000 \
  --arg sentiment_score=50 \
  --arg requested_action=HOLD \
  --arg requested_allocation_pct=15

# View live daemon status:
sofuu daemon status btc-paper-trader

# Stream live real-time logs:
sofuu daemon logs btc-paper-trader -f

# Stop or remove service:
sofuu daemon stop btc-paper-trader
sofuu daemon uninstall btc-paper-trader
```
