# Kaze example routines

> **Status:** Kaze language example only. The embedded host can return synthetic AI, tool, memory and alert results. A real model-and-authorized-tool workflow is not established, and the current non-robotics simulation conflicts with [RULES.md §1.9](../RULES.md). Do not execute these examples as evidence of a real action.

These four pure Kaze source files illustrate routine syntax and capability declarations. They are not deployable integrations or evidence of reliable autonomous behavior. The names and domain scenarios are illustrative.

- [Quant Sentinel](quant_sentinel/README.md): trade-risk example.
- [Med Protocol](med_protocol/README.md): illustrative clinical-themed syntax, not medical software.
- [Agri Sentinel](agri_sentinel/README.md): greenhouse-themed example, not a tested controller.
- [Paper Trader](paper_trader/README.md): trading-themed example; no functioning exchange is established.

Inspect the sources or check syntax without executing capability calls:

```bash
sofuu script check example-agents/quant_sentinel/agent.kaze
```

`sofuu compile` can package a supported Kaze routine into one executable; OS libraries and the selected workflow's resources remain dependencies. A compiled executable does not establish real AI, tool, memory, alert, or hardware integration.
