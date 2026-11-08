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

## 4. Autonomous Execution Examples & Verification

### Test 1: Autonomous Market Analysis & BUY Order
The agent analyzes market data, synthesizes technical support/resistance via neural prompt, saves the thesis to vector memory, and **autonomously decides to BUY**:

```bash
./example-agents/paper_trader/paper-trader \
  --arg asset=BTC-USDT \
  --arg current_price=75000 \
  --arg sentiment_score=70
```

**Live Output:**
```
[AGENT] agent :: analyze_and_trade
[POLICY] policy.detects("secrets", "BTC-USDT") -> false
[AI] ai.prompt("Act as Paper Trader market analyst. Analyze BTC-USDT at price $75000...")
[MEMORY] memory.save("BTC-USDT_paper_thesis", ...) -> nothing
[TOOL] tool.call("paper_exchange", "BTC-USDT:BUY@75000;allocation=20%") -> OK
[INFO] Paper Trader analyzing BTC-USDT at $75000 (24h Vol: $100000000, Sentiment: 70, Entry: $0)
[INFO] AUTONOMOUS DECISION: BUY BTC-USDT at $75000 (Bullish sentiment 70, 20% allocation)
[VERDICT] ALLOW "BUY executed: OK: tool 'paper_exchange' executed..."
107 instructions · 0 fuel
```

---

### Test 2: Autonomous TAKE PROFIT Order (+2% Profit Target Hit)
Holding a position entered at \$75,000 when price moves to **\$76,600** (+2.1% profit). The agent autonomously detects the target, triggers a SELL order to lock in profit, and dispatches an alert:

```bash
./example-agents/paper_trader/paper-trader \
  --arg asset=BTC-USDT \
  --arg current_price=76600 \
  --arg entry_price=75000
```

**Live Output:**
```
[AGENT] agent :: analyze_and_trade
[INFO] AUTONOMOUS DECISION: TAKE PROFIT on BTC-USDT (+2% target reached from entry $75000)
[TOOL] tool.call("paper_exchange", "BTC-USDT:SELL@76600;reason=TAKE_PROFIT;allocation=100%")
[STEP] alert.send("PROFIT TAKEN: Closed BTC-USDT position at $76600")
[VERDICT] ALLOW "TAKE PROFIT executed..."
105 instructions · 0 fuel
```

---

### Test 3: Autonomous STOP LOSS Order (-2% Risk Boundary Hit)
Holding a position entered at \$75,000 when price drops to **\$73,200** (-2.4% loss). The agent autonomously detects the risk limit, triggers a defensive SELL order, and cuts the loss:

```bash
./example-agents/paper_trader/paper-trader \
  --arg asset=BTC-USDT \
  --arg current_price=73200 \
  --arg entry_price=75000
```

**Live Output:**
```
[AGENT] agent :: analyze_and_trade
[INFO] AUTONOMOUS DECISION: STOP LOSS on BTC-USDT (-2% risk threshold hit from entry $75000)
[TOOL] tool.call("paper_exchange", "BTC-USDT:SELL@73200;reason=STOP_LOSS;allocation=100%")
[STEP] alert.send("STOP LOSS: Closed BTC-USDT position at $73200 to cut loss")
[VERDICT] ALLOW "STOP LOSS executed..."
113 instructions · 0 fuel
```

---

### Test 4: Autonomous HOLD Decision (Monitoring within Corridor)
Holding a position entered at \$75,000 when price is at **\$75,200** (+0.2%, within corridor):

```bash
./example-agents/paper_trader/paper-trader \
  --arg asset=BTC-USDT \
  --arg current_price=75200 \
  --arg entry_price=75000
```

**Live Output:**
```
[AGENT] agent :: analyze_and_trade
[INFO] AUTONOMOUS DECISION: HOLD BTC-USDT (Price $75200 within corridor of entry $75000)
[VERDICT] ALLOW "HOLD: Position active within target corridor"
94 instructions · 0 fuel
```

---

### Test 5: Autonomous Bearish Market Entry Avoidance
When sentiment is bearish ($\le 40$) and no position is open, the agent refuses entry to prevent buying into a falling market:

```bash
./example-agents/paper_trader/paper-trader \
  --arg asset=BTC-USDT \
  --arg current_price=74000 \
  --arg sentiment_score=35
```

**Live Output:**
```
[AGENT] agent :: analyze_and_trade
[INFO] AUTONOMOUS DECISION: AVOID / BEARISH on BTC-USDT (Sentiment 35 indicates downward trend)
[VERDICT] REJECT Market entry refusal: bearish sentiment signals potential downward trend
86 instructions · 0 fuel
```

---

### Test 6: Allocation Risk Breach Guardrail (`execute_manual_order`)
Attempting to allocate 35% of the paper portfolio into a single position (exceeding the 25% safety cap):

```bash
./example-agents/paper_trader/paper-trader \
  --entry execute_manual_order \
  --arg asset=SOL-USDT \
  --arg current_price=185 \
  --arg volume_24h=850000000 \
  --arg requested_action=BUY \
  --arg requested_allocation_pct=35
```

**Expected Output:**
```
[AGENT] agent :: execute_manual_order
[ALERT] alert.send("RISK BREACH: Requested paper allocation (35%) exceeds 25% risk cap on SOL-USDT")
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
