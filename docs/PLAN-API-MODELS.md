# PLAN-API-MODELS — Hosted ML Models for Sofuu Cloud API

> **Status:** PRODUCT & ARCHITECTURAL SPECIFICATION  
> **Phase:** Commercial Growth (Phase 2 / Phase 3)  
> **Target:** Exclusive Proprietary Models Served via Hosted API  
> **Author:** Project Sofuu Taskforce (Haruhito)

---

## 1. Strategic Framing & Commercial Objective

### Why an API Layer?
While Sofuu’s core engine and 5 tiny ML gates remain **100% local, offline, and open-source (MIT)**, advanced AI workloads require models that exceed edge/consumer hardware capacity or benefit from centralized cognitive state:
1. **High-Density Cross-Encoder Embeddings:** For enterprise RAG and semantic code retrieval requiring 1024+ dimensional precision.
2. **Cognitive Memory & Entity-Graph Distillation:** Models trained specifically to read multi-hour agent conversation transcripts and distill them into compact, long-term cognitive memories.
3. **Speculative Decoding Draft Models:** Small, ultra-fast code draft models tailored to pair with frontier models (Claude, GPT, DeepSeek) to accelerate streaming by 2× and cut API latency.

### The Solo-Founder Capital Protection Rule
**Zero Fixed Idle GPU Costs.**
The founder must **not** rent dedicated 24/7 cloud GPU instances ($1,000+/mo). The API infrastructure must run strictly on **pay-per-millisecond serverless GPU pools** (e.g., Modal, RunPod Serverless, or Cloudflare Workers AI). When zero requests arrive, the cloud bill is **$0.00**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SOFUU API ARCHITECTURE                          │
├────────────────────────────────────────────────────────────────────────┤
│                           SOFUU CLIENT (CLI / SDK)                     │
│               `sofuu.ai.api.distillMemory()` / `embed()`               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTPS + Bearer Key
┌───────────────────────────────────▼────────────────────────────────────┐
│                       SOFUU CLOUD API GATEWAY                          │
│                   (Rust / Cloudflare Worker / Fly)                     │
│          • API Key Validation         • Usage Metering & Quotas        │
│          • Request Deduplication      • Cryptographic Anonymization    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Zero-Idle Spin-Up
┌───────────────────────────────────▼────────────────────────────────────┐
│                    SERVERLESS GPU EXECUTION POOL                       │
│                                                                        │
│  ┌───────────────────────┐  ┌───────────────────┐  ┌───────────────┐   │
│  │ Memory Distiller      │  │ Dense RAG Embed   │  │ Code Drafter  │   │
│  │ (3B Distilled QLoRA)  │  │ (300M Transformer)│  │ (Speculative) │   │
│  └───────────────────────┘  └───────────────────┘  └───────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Target Model Specifications

### Model A: `sofuu-brain-distill-v1` (Cognitive Memory Model)
- **Problem Solved:** Raw conversation logs grow to 100,000+ tokens. Storing them raw in vector DBs causes high recall noise and huge context retrieval costs.
- **Model Architecture:** 3B parameter instruction-tuned model (distilled from frontier agent traces) specialized in **episodic memory extraction**.
- **Input:** Raw agent trajectory JSON (user requests, model answers, tool outputs).
- **Output:** Structured atomic facts, updated user preferences, project constraints, and entity relationships formatted for Sofuu’s local QTSQ tensor store.
- **Serving SLA:** Asynchronous batch processing (0.5–2 seconds per transcript).

### Model B: `sofuu-embed-dense-v1` (Code & Trajectory Embedder)
- **Problem Solved:** General-purpose text embeddings (OpenAI text-embedding-3) perform poorly on raw code ASTs, diffs, and MCP tool call parameters.
- **Model Architecture:** 350M parameter dense bidirectional transformer fine-tuned on code repos, commit diffs, and tool schemas.
- **Embedding Space:** 1024 dimensions, calibrated for cosine similarity.
- **Serving SLA:** < 30 ms p95 latency.

### Model C: `sofuu-speculative-draft-v1` (Code Acceleration)
- **Problem Solved:** Frontier models stream code at 30–50 tokens/second. Developers waiting on 500 lines of code experience high latency.
- **Model Architecture:** 200M parameter specialized draft model trained on TypeScript/Python/Rust code syntax.
- **Mechanism:** Speculative decoding where the draft model generates 5–8 candidate tokens, and the primary model verifies them in a single forward pass.
- **Result:** **2× faster code generation** with bit-for-bit identical output from the primary model.

---

## 3. Training & Distillation Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA DISTILLATION PIPELINE                      │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Trajectory Curation │ Extract 100,000 multi-turn coding and agent   │
│                        │ transcripts (synthetic + opt-in dev sessions) │
├────────────────────────┼───────────────────────────────────────────────┤
│ 2. Teacher Labeling    │ High-reasoning frontier models generate       │
│                        │ ground-truth memory distillations and labels  │
├────────────────────────┼───────────────────────────────────────────────┤
│ 3. QLoRA Fine-Tuning   │ Train 3B base models using 4-bit LoRA on      │
│                        │ rented spot cloud instances (< $200 per run)  │
├────────────────────────┼───────────────────────────────────────────────┤
│ 4. Quantized Export    │ Export fine-tuned weights to GGUF / AWQ for   │
│                        │ sub-100ms serverless cold-start execution     │
└────────────────────────┴───────────────────────────────────────────────┘
```

---

## 4. Client Integration Experience

Developers configure their Sofuu API key once in `~/.sofuu/config.json` via CLI:

```bash
sofuu config set-key sofuu_cloud sfk_live_xxxxxxxx
```

Inside JavaScript/TypeScript:
```javascript
import { ai, memory } from "sofuu";

// Distill an entire 50-turn agent session into compact cognitive memories:
const summary = await ai.api.distillSession(sofuu.chat.history());

// Automatically sync the distilled memories into the local encrypted brain:
await memory.remember(summary.facts);
```

---

## 5. Pricing and Commercial Structure

| Tier | Monthly Price | Included API Quota | Overages |
|---|---|---|---|
| **Community** | **$0 / mo** | 1,000 memory distillations / mo | N/A (Hard cap) |
| **Developer Pro** | **$19 / mo** | 25,000 distillations + 5M embeddings | $0.20 per 1M tokens |
| **Team** | **$79 / mo** | Shared team pool + cross-device memory sync | $0.15 per 1M tokens |

---

## 6. Implementation Milestones

1. **Milestone 1 (Data & Distillation):** Synthesize 50,000 agent memory distillation pairs using synthetic trajectory generation scripts.
2. **Milestone 2 (Training Run):** Run QLoRA fine-tuning on a rented 8×A100 spot cluster (estimated compute cost: ~$150). Validate on held-out agent eval set.
3. **Milestone 3 (Serverless Deployment):** Deploy model on Modal / RunPod serverless endpoints with auto-scaling to zero.
4. **Milestone 4 (SDK & Billing):** Wire Stripe usage-based billing and release `@sofuu/api` client bindings.
