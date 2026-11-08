# 🌟 Sofuu Example AI Agents (100% Pure Kaze)

> **Complete autonomous AI agents programmed 100% in Kaze (`.kaze`) with zero JavaScript or TypeScript required, compiled directly into standalone single-binary native executables.**

---

## 1. Overview

These showcase agents demonstrate full-lifecycle autonomous agent engineering using only **Kaze** and the **Sofuu Runtime**. Each agent operates across a distinct sector, showing how Kaze combines:
1. **First-Class AI Reasoning (`ai`)**: High-level thesis generation, clinical protocols, and agronomy modeling.
2. **Deterministic Tool Execution (`tool`)**: Order routers, hospital EHR integrations, and irrigation actuators.
3. **Hard Fail-Closed Safety Guardrails (`policy`)**: R7 firewall leak detection, leverage hard caps, and HIPAA credential filtering.
4. **Stateful Associative Memory (`memory`)**: Cross-turn context retention and historical recall.
5. **Real-Time Audit Logging & Alerting (`log`, `alert`)**: Compliance-grade audit trails and escalation paging.

---

## 2. Directory Structure

```
example-agents/
├── build.sh                  # One-click script to compile all four agents
├── README.md                 # Complete guide & documentation (this file)
├── quant_sentinel/           # [Fintech] High-Frequency Trading Risk Sentinel
│   ├── agent.kaze            # 100% Pure Kaze source code
│   ├── quant-sentinel        # Standalone compiled native executable (~4.5 MiB)
│   └── README.md             # Architecture, parameters, and test queries
├── med_protocol/             # [Healthcare] Clinical Vital Triage & EHR Agent
│   ├── agent.kaze            # 100% Pure Kaze source code
│   ├── med-protocol          # Standalone compiled native executable (~4.5 MiB)
│   └── README.md             # Clinical safety gates, ESI scale, and EHR integration
├── agri_sentinel/            # [Agri-Tech] Autonomous Greenhouse Climate Agent
│   ├── agent.kaze            # 100% Pure Kaze source code
│   ├── agri-sentinel         # Standalone compiled native executable (~4.5 MiB)
│   └── README.md             # Sensor telemetry, relay actuation, and agronomy AI
└── paper_trader/             # [Simulated Trading] Crypto & Stock Paper Trader (NO REAL MONEY)
    ├── agent.kaze            # 100% Pure Kaze source code
    ├── paper-trader          # Standalone compiled native executable (~4.5 MiB)
    └── README.md             # Sentiment analysis, risk caps, and paper exchange simulation
```

---

## 3. Quickstart: Building All Agents

You can build all four agents with a single command:

```bash
./example-agents/build.sh
```

Or compile any individual agent using `sofuu compile`:

```bash
# Fintech
sofuu compile example-agents/quant_sentinel/agent.kaze -o example-agents/quant_sentinel/quant-sentinel

# Healthcare
sofuu compile example-agents/med_protocol/agent.kaze -o example-agents/med_protocol/med-protocol

# Agri-Tech
sofuu compile example-agents/agri_sentinel/agent.kaze -o example-agents/agri_sentinel/agri-sentinel

# Simulated Trading
sofuu compile example-agents/paper_trader/agent.kaze -o example-agents/paper_trader/paper-trader
```

Every compiled binary is a single self-contained native executable (**~4.5 MiB**) with zero external runtime dependencies.

---

## 4. The Showcase Agents

### 📈 1. Quant Sentinel (`quant_sentinel`)
* **Sector:** Fintech / Quantitative Trading
* **Capabilities:** `ai`, `tool`, `policy`, `memory`, `log`, `alert`
* **Purpose:** Evaluates incoming trade signals against strict pre-trade leverage constraints and API credential leak checks. If allowed, it invokes an AI reasoning model to generate an adaptive trading thesis, persists it to memory, and triggers the exchange order router.

#### Valid Trade Execution (Allowed):
```bash
./example-agents/quant_sentinel/quant-sentinel \
  --entry evaluate_trade \
  --arg 'ticker="NVDA"' \
  --arg 'signal="BUY"' \
  --arg 'leverage=2' \
  --arg 'price=145'
```
*Output:*
```
[AGENT] agent :: evaluate_trade
[POLICY] policy.detects("secrets", "NVDA") -> false
[AI] ai.prompt("Act as Quant Sentinel risk engine...") -> "Synthesized autonomous decision..."
[MEMORY] memory.save("NVDA_thesis", ...) -> nothing
[TOOL] tool.call("order_router", "NVDA:BUY@145") -> "OK: tool 'order_router' executed"
[VERDICT] ALLOW "Synthesized autonomous decision..."
```

#### Excessive Leverage Violation (Rejected):
```bash
./example-agents/quant_sentinel/quant-sentinel \
  --entry evaluate_trade \
  --arg 'ticker="NVDA"' \
  --arg 'signal="BUY"' \
  --arg 'leverage=5' \
  --arg 'price=145'
```
*Output:*
```
[AGENT] agent :: evaluate_trade
[ALERT] alert.send("HIGH RISK: Excessive leverage (5x) requested on NVDA")
[VERDICT] REJECT Risk policy violation: leverage 5 exceeds hard cap of 3.0x
```

---

### 🏥 2. Med Protocol (`med_protocol`)
* **Sector:** Healthcare / Emergency Clinical Triage
* **Capabilities:** `ai`, `tool`, `policy`, `memory`, `log`, `alert`
* **Purpose:** Triages emergency department vitals against severe arrhythmia and hypertensive crisis thresholds. Enforces HIPAA compliance by refusing clinical notes containing unredacted API credentials or tokens, recommends an ESI triage level, retains patient history, and charts findings to an EHR tool.

#### Valid Clinical Triage (Allowed):
```bash
./example-agents/med_protocol/med-protocol \
  --entry triage_patient \
  --arg 'patient_id="PT-901"' \
  --arg 'heart_rate=82' \
  --arg 'bp_systolic=120' \
  --arg 'bp_diastolic=80' \
  --arg 'symptoms="Mild tension headache"'
```
*Output:*
```
[AGENT] agent :: triage_patient
[POLICY] policy.detects("secrets", "Mild tension headache") -> false
[AI] ai.prompt("Clinical Triage AI: Patient PT-901 HR=82 BP=120/80...")
[MEMORY] memory.save("PT-901_triage", ...) -> nothing
[TOOL] tool.call("ehr_record_vitals", "PT-901:triage_complete")
[VERDICT] ALLOW "Synthesized autonomous decision..."
```

#### HIPAA Data Leak Violation (Rejected):
```bash
./example-agents/med_protocol/med-protocol \
  --entry triage_patient \
  --arg 'patient_id="PT-901"' \
  --arg 'heart_rate=82' \
  --arg 'bp_systolic=120' \
  --arg 'bp_diastolic=80' \
  --arg 'symptoms="Token: sk-ant-api03-abcdef123456789"'
```
*Output:*
```
[AGENT] agent :: triage_patient
[POLICY] policy.detects("secrets", ...) -> true
[VERDICT] REJECT HIPAA compliance refusal: unredacted credential or private key in clinical notes
```

---

### 🌾 3. Agri Sentinel (`agri_sentinel`)
* **Sector:** Agri-Tech / Precision Greenhouse & Vertical Farming
* **Capabilities:** `sensor`, `device`, `ai`, `tool`, `policy`, `memory`, `log`, `alert`
* **Purpose:** Continuously reads environmental microclimate telemetry (temperature, humidity), triggers physical relay actuators (ventilation fans, ultrasonic mister pumps), queries agronomy AI models for vapor pressure deficit (VPD) stress optimization, and confirms valve state via agricultural tools.

#### Automated Microclimate Regulation & Advisory (Allowed):
```bash
./example-agents/agri_sentinel/agri-sentinel \
  --entry monitor_greenhouse \
  --arg 'zone_id="ZONE-NORTH"' \
  --arg 'crop_type="Heirloom Tomatoes"' \
  --arg 'target_temp=24' \
  --arg 'target_humidity=65'
```
*Output:*
```
[AGENT] agent :: monitor_greenhouse
[STEP] sensor.temperature -> 22
[STEP] sensor.humidity -> 45
[STEP] alert.send("LOW MOISTURE ALERT in Zone ZONE-NORTH: 45%")
[STEP] device.pulse("mister", 15) -> nothing
[AI] ai.prompt("Precision Agri Agronomist: Zone ZONE-NORTH Crop=Heirloom Tomatoes...")
[MEMORY] memory.save("ZONE-NORTH_agronomy", ...)
[TOOL] tool.call("irrigation_controller", "ZONE-NORTH:cycle_applied")
[VERDICT] ALLOW "Synthesized autonomous decision..."
```

---

### 📊 4. Paper Trader (`paper_trader`)
* **Sector:** Simulated Trading / Crypto & Stock Market Analysis (NO REAL MONEY)
* **Capabilities:** `ai`, `tool`, `policy`, `memory`, `log`, `alert`
* **Purpose:** Performs technical analysis and market sentiment evaluation, enforces strict 25% single-position risk caps, prevents credential leaks, and dispatches simulated orders to a paper exchange tool.

#### Valid Simulated Paper Trade (Allowed):
```bash
./example-agents/paper_trader/paper-trader \
  --entry analyze_and_trade \
  --arg 'asset="BTC/USD"' \
  --arg 'current_price=68450' \
  --arg 'volume_24h=1420000000' \
  --arg 'sentiment_score=78' \
  --arg 'requested_action="BUY"' \
  --arg 'requested_allocation_pct=15'
```
*Output:*
```
[AGENT] agent :: analyze_and_trade
[POLICY] policy.detects("secrets", "BTC/USD") -> false
[AI] ai.prompt("Act as Paper Trader market analyst...")
[MEMORY] memory.save("BTC/USD_paper_thesis", ...)
[TOOL] tool.call("paper_exchange", "BTC/USD:BUY@68450;allocation=15%")
[VERDICT] ALLOW "Synthesized autonomous decision..."
```

#### Allocation Limit Violation (Rejected):
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
*Output:*
```
[AGENT] agent :: analyze_and_trade
[ALERT] alert.send("RISK BREACH: Requested paper allocation (35%) exceeds 25% risk cap on SOL/USD")
[VERDICT] REJECT Risk limit violation: maximum single paper position allocation is 25%
```

---

## 5. Architectural Benefits of Pure Kaze Agents

1. **Zero Garbage Collection Stutter**: Bounded arena allocations guarantee deterministic sub-millisecond execution loops without unpredictable GC pauses.
2. **Compile-Time Safety Verification**: Front-end checks (`sofuu.kaze.check`) validate capability declarations and type signatures at compile time before any binary is produced.
3. **Fail-Closed Security Matrix**: Capability family gating (`needs: ...`) prevents undeclared ambient I/O from executing inside the VM.
4. **Single Binary Distribution**: Standalone single-file native executables run on macOS (arm64, x86_64) and Linux (x86_64, aarch64 / Pi 5) without needing Node.js, Python, or npm.

---

## 6. Running Agents 24/7 with One Command

Run any standalone agent continuously around the clock with zero configuration files using either of the two unified modes:

### Mode A: Native Standalone Daemon (`--daemon` / `--every`)
Directly run the compiled standalone binary in a continuous autonomous supervisor loop:

```bash
./example-agents/paper_trader/paper-trader \
  --daemon \
  --every 5s \
  --entry analyze_and_trade \
  --arg 'asset=BTC-USDT' \
  --arg 'current_price=94200' \
  --arg 'volume_24h=18000000' \
  --arg 'sentiment_score=85' \
  --arg 'requested_action=BUY' \
  --arg 'requested_allocation_pct=2.0'
```

* **Crash Recovery:** Exceptions never crash the daemon; errors are contained and logged, and the next cycle runs on time.
* **Graceful Exit:** Press `Ctrl+C` at any time to see the shutdown summary badge:
  `[SUMMARY] Completed 42 cycles (42 succeeded, 0 failed) · Uptime: 3m 30s`

### Mode B: Zero-Config OS Background Service (`sofuu daemon install`)
Install the agent as a persistent system service that auto-restarts on boot and survives terminal closures:

```bash
# 1. Install & start service (launchd on macOS, systemd on Linux):
sofuu daemon install ./example-agents/paper_trader/paper-trader \
  --name btc-paper-trader \
  --every 10s \
  --entry analyze_and_trade \
  --arg 'asset=BTC-USDT' \
  --arg 'current_price=94200' \
  --arg 'volume_24h=18000000' \
  --arg 'sentiment_score=85' \
  --arg 'requested_action=BUY' \
  --arg 'requested_allocation_pct=2.0'

# 2. View running daemon status & PID:
sofuu daemon status btc-paper-trader

# 3. Stream real-time execution logs:
sofuu daemon logs btc-paper-trader -f

# 4. Stop or uninstall:
sofuu daemon stop btc-paper-trader
sofuu daemon uninstall btc-paper-trader
```

