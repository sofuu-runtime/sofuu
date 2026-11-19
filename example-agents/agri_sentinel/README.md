# Agri Sentinel

> **Language example only.** This routine is not connected to a greenhouse, a
> sensor, an irrigation controller, a fan or a mister, and it has never
> actuated anything. The current embedded Kaze source fails closed on every
> capability it has no backend for; older published binaries may behave
> differently. Do not connect it to real equipment.

A greenhouse-themed routine that illustrates sensor and device capability
syntax, threshold guardrails, memory and alerting. It establishes no real
sensor, model, irrigation or controller integration.

## What actually happens when you run it

With the current source, the run stops at the first capability that has no
backend, with a typed error and the source line:

| Granted families | Observed result on 2026-11-16 |
|---|---|
| `sensor`, `device`, `ai`, `tool`, `log`, `alert`, `memory`, `policy` | `error[E_CAP_MISSING]: embedded Kaze has no available backend for sensor.temperature` at `agent.kaze:10`. The embedded engine ships no sensor or device backend at all, so `turn on fan` and `pulse mister` are unreachable too. |
| No `--policy` | `rejection[E_CAP_DENIED]: policy needs tools.allow or tools.deny…` at `agent.kaze:8`. |

The temperature and humidity thresholds in `agent.config.json` are static
example values, not readings. The `send alert` calls would reach only a local
sink (`alert=stdout`, `alert=stderr` or `alert=file:<path>`); no irrigation
decision, no watering cycle and no alert to an operator has been produced.

## Inspect it

```bash
sofuu script check example-agents/agri_sentinel/agent.kaze
```

A real model-and-authorized-tool path exists for the read-only
[Local Brief](../local_brief/README.md) example; this greenhouse scenario is
deliberately not connected to one.
