# Quant Sentinel

> **Language example only.** This routine is not connected to a broker, an
> exchange, a market-data provider or an order router, and it has never placed
> an order. The current embedded Kaze source fails closed on every capability
> it has no backend for; older published binaries may behave differently. Do
> not use it to trade, and do not point it at a real trading account.

A trading-themed routine that illustrates parameter binding, policy
detection, risk guardrails, memory and capability syntax. It establishes no
live model, order router, exchange integration, durable memory or
tamper-evident log.

## What actually happens when you run it

With the current source, the run stops at the first capability that has no
backend, with a typed error and the source line:

| Granted families | Observed result on 2026-11-16 |
|---|---|
| `ai`, `tool`, `log`, `memory`, `alert`, `policy` | `error[E_CAP_MISSING]: AI capability is unavailable…` at `agent.kaze:23` (the `ai.prompt` call). |
| No `--policy` | `rejection[E_CAP_DENIED]: policy needs tools.allow or tools.deny…` at `agent.kaze:8`. |
| `ai` with a real model configured, no `order_router` MCP tool | The AI step succeeds and `tool.call("order_router", …)` then fails closed with `E_CAP_MISSING (E_MCP_TOOL_NOT_FOUND)`. |

No run of this example has produced an order, a fill, a thesis from a real
model, or an alert that left the machine. The `send alert` calls would reach
only a local sink (`alert=stdout`, `alert=stderr` or `alert=file:<path>`);
there is no broker, pager or broker-integration channel.

## Inspect it

```bash
sofuu script check example-agents/quant_sentinel/agent.kaze
```

A real model-and-authorized-tool path exists for the read-only
[Local Brief](../local_brief/README.md) example; this trading scenario is
deliberately not connected to one.
