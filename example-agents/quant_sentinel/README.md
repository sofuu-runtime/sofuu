# Quant Sentinel — Fintech AI Agent (100% Pure Kaze)

Quant Sentinel is an autonomous algorithmic trading and risk-monitoring AI agent programmed **entirely in Kaze (`agent.kaze`)**. No JavaScript, TypeScript, or node_modules required. Sofuu handles local LLM inference, tool execution, persistent memory, and R7 hardware safety guardrails under the hood.

---

## Capabilities Used

```kaze
needs: ai, tool, policy, memory, log, alert
```

- **`ai`**: LLM market analysis, volatility evaluation, and trade thesis synthesis (`ai.prompt`).
- **`tool`**: Interfacing with order routers and market data feeds (`tool.call`).
- **`policy`**: Deterministic R7 safety guardrails: credential leak detection and leverage hard caps.
- **`memory`**: Persisting generated theses for historical recall and pattern audits (`remember ... as ...`).
- **`log`**: Formal tamper-evident audit trail (`log.info`).
- **`alert`**: Real-time margin and risk escalation notifications (`send alert ...`).

---

## Agent Logic (`agent.kaze`)

```kaze
needs: ai, tool, policy, memory, log, alert

filter evaluate_trade(ticker, signal, leverage, price):
    log.info("Quant Sentinel evaluating trade: " + ticker + " (" + signal + " at $" + price + ", " + leverage + "x leverage)")
    
    # 1. Credential & Leak Detection Firewall
    if policy detects secrets in ticker:
        reject "Security violation: secret token leaked in symbol"
    
    # 2. Risk Guardrail: Reject leverage exceeding risk tolerance
    if leverage > 3:
        send alert "HIGH RISK: Excessive leverage (" + leverage + "x) requested on " + ticker
        reject "Risk policy violation: leverage " + leverage + " exceeds hard cap of 3.0x"
        
    if price <= 0:
        reject "Invalid market price: must be positive"
        
    # 3. AI Market Thesis & Risk Assessment
    let thesis = ai.prompt("Act as Quant Sentinel risk engine. Synthesize market thesis for " + ticker + " signal=" + signal + " price=" + price + " leverage=" + leverage + ". Provide target, stop-loss, and sizing.")
    
    # 4. Agent Memory: Persist thesis for future strategy reviews
    remember thesis as ticker + "_thesis"
    
    # 5. Order Router Tool Execution
    let route = tool.call("order_router", ticker + ":" + signal + "@" + price)
    log.info("Order router status: " + route)
    
    # 6. Safety Alert for elevated exposures
    if leverage >= 2:
        send alert "Elevated exposure active: " + ticker + " at " + leverage + "x leverage"
        
    allow thesis
```

---

## Running Directly with Sofuu CLI

```bash
# Evaluate a trade signal
sofuu script run example-agents/quant_sentinel/agent.kaze \
    --entry evaluate_trade \
    --arg ticker=NVDA \
    --arg signal=BUY \
    --arg leverage=2 \
    --arg price=145

# Test the risk guardrail (rejection on excessive leverage)
sofuu script run example-agents/quant_sentinel/agent.kaze \
    --entry evaluate_trade \
    --arg ticker=NVDA \
    --arg signal=BUY \
    --arg leverage=5 \
    --arg price=145
```

---

## Compiling to a Standalone Native Executable

Compile the `.kaze` file directly into a single self-contained binary:

```bash
sofuu compile example-agents/quant_sentinel/agent.kaze -o example-agents/quant_sentinel/quant-sentinel
```

### Production Execution Modes

#### 1. Config File Driven (`--config`)
Run cleanly using [`agent.config.json`](agent.config.json):
```bash
./example-agents/quant_sentinel/quant-sentinel --config example-agents/quant_sentinel/agent.config.json
```

#### 2. 24/7 Supervisor Loop (`--daemon --every`)
Run continuous market surveillance:
```bash
./example-agents/quant_sentinel/quant-sentinel --config example-agents/quant_sentinel/agent.config.json --daemon --every 2s
```

#### 3. Positional Arguments (Fast Execution)
```bash
./example-agents/quant_sentinel/quant-sentinel NVDA BUY 2.5 130
```

#### 4. Container / Cloud Environment Injection
```bash
export AGENT_TICKER=NVDA
export AGENT_SIGNAL=BUY
export AGENT_LEVERAGE=2.5
export AGENT_PRICE=130

./example-agents/quant_sentinel/quant-sentinel
```

#### 5. Interactive Agent TUI
```bash
./example-agents/quant_sentinel/quant-sentinel --interactive
```
