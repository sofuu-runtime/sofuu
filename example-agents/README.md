# Kaze example routines

> **Status:** one example (Local Brief) is a real, read-only model-and-tool
> workflow; the other three are language examples whose capability calls fail
> closed because no backend is configured for them. No example here places an
> order, treats a patient, actuates equipment or sends a clinical alert.

- [Local Brief](local_brief/README.md): the real one. A pure Kaze routine
  reads one operator-selected local file through an authorized MCP tool and
  returns a real model's answer about it. Read-only.
- [Quant Sentinel](quant_sentinel/README.md): trade-risk language example. No
  broker, exchange or order router is connected.
- [Med Protocol](med_protocol/README.md): clinical-themed language example.
  Not medical software.
- [Agri Sentinel](agri_sentinel/README.md): greenhouse-themed language example.
  No sensor, controller or actuator is connected.

## Inspect without executing capability calls

```bash
sofuu script check example-agents/local_brief/agent.kaze
```

## Execute

`sofuu compile` packages a supported Kaze routine into one executable; the
model, the MCP service, the operator policy and the OS libraries stay
separate dependencies of the workload. A compiled executable does not by
itself establish real AI, tool, memory, alert or hardware integration — it
only establishes that the routine was packaged.

See [Local Brief](local_brief/README.md) for a run that was actually observed
against real local model weights and a real MCP service, and for the exact
dependencies it needs.
