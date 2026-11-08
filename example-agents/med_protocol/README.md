# Med Protocol — Healthcare Clinical Triage AI Agent (100% Pure Kaze)

Med Protocol is an autonomous clinical triage and emergency response agent programmed **entirely in Kaze (`agent.kaze`)**. It evaluates real-time patient vital signs, enforces HIPAA privacy boundaries, fires life-saving alerts on critical bradycardia/tachycardia or hypertensive crises, records patient charts into Electronic Health Records (EHR), and consults AI clinical protocols.

---

## Capabilities Used

```kaze
needs: ai, tool, policy, memory, log, alert
```

- **`ai`**: Clinical assessment, emergency severity index (ESI Level 1–5) classification, and triage recommendation (`ai.prompt`).
- **`tool`**: Recording vitals into hospital Electronic Health Records (`tool.call`).
- **`policy`**: HIPAA compliance guardrail and unredacted PHI/credential protection (`policy detects secrets in ...`).
- **`memory`**: Persisting patient triage history for discharge summaries (`remember ... as ...`).
- **`log`**: Formal clinical audit logging (`log.info`).
- **`alert`**: Medical staff paging and crisis alerts (`send alert ...`).

---

## Agent Logic (`agent.kaze`)

```kaze
needs: ai, tool, policy, memory, log, alert

filter triage_patient(patient_id, heart_rate, bp_systolic, bp_diastolic, symptoms):
    log.info("Med Protocol triaging patient " + patient_id + ": HR=" + heart_rate + ", BP=" + bp_systolic + "/" + bp_diastolic)
    
    # 1. Privacy & PHI Protection Guardrail
    if policy detects secrets in symptoms:
        reject "HIPAA compliance refusal: unredacted credential or private key in clinical notes"
        
    # 2. Critical Physiological Safety Guardrails
    if heart_rate > 160:
        send alert "CRITICAL: Severe tachycardia (HR " + heart_rate + ") on patient " + patient_id
    if heart_rate < 40:
        send alert "CRITICAL: Severe bradycardia (HR " + heart_rate + ") on patient " + patient_id
        
    if bp_systolic > 180 or bp_diastolic > 120:
        send alert "EMERGENCY: Hypertensive crisis (BP " + bp_systolic + "/" + bp_diastolic + ") on patient " + patient_id
        
    # 3. AI Clinical Assessment & Triage Synthesis
    let triage_assessment = ai.prompt("Clinical Triage AI: Patient " + patient_id + " HR=" + heart_rate + " BP=" + bp_systolic + "/" + bp_diastolic + " Symptoms: " + symptoms + ". Assign emergency severity index (ESI Level 1-5), immediate interventions, and required physician consult.")
    
    # 4. Long-Term Patient Vitals & Triage Retention
    remember triage_assessment as patient_id + "_triage"
    
    # 5. EHR Chart Tool Integration
    let ehr_status = tool.call("ehr_record_vitals", patient_id + ":triage_complete")
    log.info("EHR vital chart status: " + ehr_status)
    
    allow triage_assessment
```

---

## Running Directly with Sofuu CLI

```bash
sofuu script run examples/agents/med_protocol/agent.kaze \
    --entry triage_patient \
    --arg patient_id='"PT-9042"' \
    --arg heart_rate=142 \
    --arg bp_systolic=165 \
    --arg bp_diastolic=98 \
    --arg symptoms='"Acute dyspnea and persistent diaphoresis"'
```

---

## Compiling to a Standalone Executable

```bash
# Compile pure Kaze file into a standalone executable:
sofuu compile examples/agents/med_protocol/agent.kaze -o med-protocol

# Run directly:
./med-protocol --entry triage_patient --arg patient_id='"PT-9042"' --arg heart_rate=80 --arg bp_systolic=120 --arg bp_diastolic=80 --arg symptoms='"Mild headache"'
```
