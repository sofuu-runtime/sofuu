# Agri Sentinel — Agri-Tech Precision Greenhouse AI Agent (100% Pure Kaze)

Agri Sentinel is an autonomous precision greenhouse climate and irrigation agent programmed **entirely in Kaze (`agent.kaze`)**. It monitors microclimate sensors (temperature, humidity), actuates physical hardware relays (cooling fans, mister pumps), queries AI agronomy crop models for vapor pressure deficit (VPD) optimization, and integrates with farm irrigation controllers.

---

## Capabilities Used

```kaze
needs: sensor, device, ai, tool, policy, memory, log, alert
```

- **`sensor`**: Telemetry reads for ambient temperature and relative humidity (`sensor.temperature()`, `sensor.humidity()`).
- **`device`**: Direct actuation of physical relays and misting solenoids (`turn on fan`, `turn off fan`, `pulse mister for 15 seconds`).
- **`ai`**: Agronomy AI reasoning on crop stress, powdery mildew risk, and nutrient injection schedules (`ai.prompt`).
- **`tool`**: Interfacing with farm irrigation controllers (`tool.call`).
- **`policy`**: Environmental guardrails and actuator duty-cycle safety gates.
- **`memory`**: Persisting zone microclimate trends for harvest readiness audits (`remember ... as ...`).
- **`log`**: Farm operations audit logging (`log.info`).
- **`alert`**: Heatwave alerts and moisture warnings (`send alert ...`).

---

## Agent Logic (`agent.kaze`)

```kaze
needs: sensor, device, ai, tool, policy, memory, log, alert

filter monitor_greenhouse(zone_id, crop_type, target_temp, target_humidity):
    log.info("Agri Sentinel scanning Zone " + zone_id + " (" + crop_type + ")")
    
    let temp = sensor.temperature()
    let humidity = sensor.humidity()
    log.info("Telemetry in Zone " + zone_id + ": Temp=" + temp + "C, Humidity=" + humidity + "%")
    
    # 1. Environmental Threshold Guardrails: Extreme Temperature Protection
    if temp > target_temp + 3:
        send alert "HIGH TEMP WARNING in Zone " + zone_id + ": " + temp + "C"
        turn on fan
    otherwise:
        turn off fan
        
    # 2. Moisture & Irrigation Automation
    if humidity < target_humidity:
        send alert "LOW MOISTURE ALERT in Zone " + zone_id + ": " + humidity + "%"
        pulse mister for 15 seconds
        
    # 3. AI Agronomy Reasoning & Disease Prevention
    let agronomy_advice = ai.prompt("Precision Agri Agronomist: Zone " + zone_id + " Crop=" + crop_type + " Current Temp=" + temp + "C Current Humidity=" + humidity + "%. Target Temp=" + target_temp + "C Target Humidity=" + target_humidity + "%. Evaluate VPD stress risk, powdery mildew probability, and nutrient pump schedule.")
    
    # 4. Memory Retention: Track zone microclimate history
    remember agronomy_advice as zone_id + "_agronomy"
    
    # 5. Agricultural Controller Tool Invocation
    let valve_res = tool.call("irrigation_controller", zone_id + ":cycle_applied")
    log.info("Irrigation controller confirmation: " + valve_res)
    
    allow agronomy_advice
```

---

## Running Directly with Sofuu CLI

```bash
sofuu script run example-agents/agri_sentinel/agent.kaze \
    --entry monitor_greenhouse \
    --arg zone_id=Zone-3-North \
    --arg crop_type="Heirloom Tomato" \
    --arg target_temp=24 \
    --arg target_humidity=65 \
    --sim temperature=28.5,humidity=52
```

---

## Compiling to a Standalone Native Executable

```bash
# Compile pure Kaze file into a standalone executable:
sofuu compile example-agents/agri_sentinel/agent.kaze -o example-agents/agri_sentinel/agri-sentinel
```

### Production Execution Modes

#### 1. Config File Driven (`--config`)
Run cleanly using [`agent.config.json`](agent.config.json):
```bash
./example-agents/agri_sentinel/agri-sentinel --config example-agents/agri_sentinel/agent.config.json
```

#### 2. Positional Arguments (Rapid Field Assessment)
```bash
./example-agents/agri_sentinel/agri-sentinel North-Greenhouse Heirloom-Tomato 22 65
```

#### 3. 24/7 Precision Irrigation Daemon (`--daemon --every`)
Continuously monitor microclimate conditions and pulse misters/fans:
```bash
./example-agents/agri_sentinel/agri-sentinel --config example-agents/agri_sentinel/agent.config.json --daemon --every 5s
```

#### 4. Edge IoT / Gateway Environment Injection
```bash
export AGENT_ZONE_ID="North-Greenhouse"
export AGENT_CROP_TYPE="Heirloom-Tomato"
export AGENT_TARGET_TEMP=22
export AGENT_TARGET_HUMIDITY=65

./example-agents/agri_sentinel/agri-sentinel
```
