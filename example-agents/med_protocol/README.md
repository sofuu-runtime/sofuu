# Med Protocol

> **Language example only.** This is not validated medical software, it
> establishes no HIPAA or other compliance, and it must not be used for patient
> triage, EHR updates, clinical decision support or emergency alerting. The
> current embedded Kaze source fails closed on every capability it has no
> backend for; older published binaries may behave differently.

A clinical-themed routine that illustrates vital-sign guardrails, privacy
detection, memory and capability syntax. Nothing in this repository is
connected to a patient record, an EHR, a triage queue or a clinician.

## What actually happens when you run it

With the current source, the run stops at the first capability that has no
backend, with a typed error and the source line:

| Granted families | Observed result on 2026-11-16 |
|---|---|
| `ai`, `tool`, `log`, `alert`, `memory`, `policy` | `error[E_CAP_MISSING]: AI capability is unavailable…` at `agent.kaze:24` (the `ai.prompt` call). |
| No `--policy` | `rejection[E_CAP_DENIED]: policy needs tools.allow or tools.deny…` at `agent.kaze:8`. |
| `ai` with a real model configured, no `ehr_record_vitals` MCP tool | The AI step succeeds and `tool.call("ehr_record_vitals", …)` then fails closed with `E_CAP_MISSING (E_MCP_TOOL_NOT_FOUND)`. |

The vital-sign values in `agent.config.json` are static example data, not
readings. The `send alert` calls would reach only a local sink (`alert=stdout`,
`alert=stderr` or `alert=file:<path>`) — a local line of text, not a clinical
alert and not a notification to anyone. No compliance, safety or clinical
outcome is claimed or demonstrated.

## Inspect it

```bash
sofuu script check example-agents/med_protocol/agent.kaze
```

A real model-and-authorized-tool path exists for the read-only
[Local Brief](../local_brief/README.md) example; this clinical scenario is
deliberately not connected to one.
