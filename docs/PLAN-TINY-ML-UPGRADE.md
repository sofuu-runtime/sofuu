# PLAN-TINY-ML-UPGRADE — Context-Economy Micro-Models Calibration

> **Status:** ENGINEERING SPECIFICATION & TRAINING PLAN  
> **Component:** `crates/sofuu-core/src/ml/` & `crates/ml-train/`  
> **Target:** Upgrading the 5 On-Device Tiny ML Models with Real-World Trajectory Data  
> **Author:** Project Sofuu Taskforce (Haruhito)

---

## 1. Baseline Assessment & The Problem Today

### What We Hold Today
Sofuu ships five tiny, on-device neural network models (~8.5k parameters each, ~150 KB total binary weight) that run on pure CPU in **< 20 microseconds** with zero API keys and zero network calls:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THE FIVE CONTEXT-ECONOMY GATES                       │
├────────────────────┬───────────────────────────────────────────────────┤
│ 1. Freshness Gate  │ Scores staleness of cached documents vs task.     │
│ 2. Compaction Gate │ Selects safe history turns to summarize/drop.     │
│ 3. Relevance Gate  │ Filters context candidates prior to model prompt. │
│ 4. Alloc Gate      │ Sets model output token budgets & detects limits. │
│ 5. Supervisor Gate │ Detects loops, redundant reads, & unsafe tools.   │
└────────────────────┴───────────────────────────────────────────────────┘
```

### The Current Evidence Debt
As documented in `TASKS.md` and `docs/PLAN-ML-GATES.md`:
- All five models were trained in `crates/ml-train` on **synthetic datasets with mechanical heuristic labels**.
- While all models pass held-out linear/logistic regression baseline tests, **none of them have been calibrated against real-world multi-turn developer agent traffic**.
- The critical end metric—**tokens saved per completed task** on real tasks—has not yet been empirically measured against a shadow control.

---

## 2. Upgrade Objectives & The Primary Metric

### The Primary End Metric
> **"Tokens Saved Per Completed Task Without Degrading Task Success Rate."**

The upgrade must prove that running an autonomous agent (e.g., a 20-step coding task) with the 5 tiny models active consumes **30% to 50% fewer total tokens** than an unmanaged baseline, while completing the exact same user task successfully.

### Architecture Invariants
- **Parameter Ceiling:** Each model must remain **≤ 15,000 parameters** (~60 KB per model).
- **Inference Latency:** Pure CPU forward-pass must execute in **< 50 microseconds**.
- **Delivery Model:** Weights are compiled directly into the binary via Rust `include_bytes!("weights_v2.f32")`. Zero cloud dependency, zero external files to download.

---

## 3. Real Data Sourcing & Ground-Truth Labeling

### 3.1 Real Trajectory Collection
Data will be sourced from three high-fidelity pools:
1. **Open-Source Agent Benchmarks:** SWE-bench, HumanEval, and AgentBench trajectory traces.
2. **Real Coding Sessions:** Git histories, multi-turn diffs, and tool invocation streams.
3. **Opt-in Local Tracing:** Anonymous local session traces from developer alpha testers (prompt lengths, tool counts, error codes).

### 3.2 Ground-Truth Labeling Methodology per Gate

```
┌────────────────────────────────────────────────────────────────────────┐
│                     GROUND-TRUTH LABELING PROTOCOL                     │
├──────────────┬────────────────────────┬────────────────────────────────┤
│ Gate Model   │ Real Feature Inputs    │ Ground-Truth Label (Target)    │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ Compaction   │ Turn age, token count, │ 1 = Turn was never referenced  │
│ Gate         │ user decisions, tool   │     again in subsequent steps. │
│              │ returns, lexical cues. │ 0 = Critical context needed.   │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ Supervisor   │ Tool repeat count, AST │ 1 = Tool failed, duplicated a  │
│ Gate         │ diff, command string,  │     prior call, or caused err. │
│              │ directory bounds.      │ 0 = Safe, productive call.     │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ Relevance    │ BM25 overlap, file age,│ 1 = File was actually edited   │
│ Gate         │ git recency, query sim.│     or cited in final patch.   │
│              │                        │ 0 = Unread noise candidate.    │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ Alloc Gate   │ Gateway wire, model ID,│ 1 = Request hit `finish_reason:│
│              │ prompt tokens, history.│     length` or gateway 400.    │
│              │                        │ 0 = Clean completion.          │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ Freshness    │ File modification time,│ 1 = Document content changed   │
│ Gate         │ git commit delta, task.│     before agent execution.    │
│              │                        │ 0 = Document content fresh.    │
└──────────────┴────────────────────────┴────────────────────────────────┘
```

---

## 4. Feature Engineering & Guardrails

### 4.1 Protected Lexical Floor
Even with ML advice, compaction must obey strict deterministic safeguards:
- **Protected Turns:** Any turn containing user instructions, passwords, architectural decisions, or active error logs is given an absolute protected floor and **can never be dropped by the compaction model**.
- **Per-Pass Budget Cap:** A single compaction pass cannot delete more than 25% of existing blocks, ensuring the model never hallucinates away the entire conversation.

### 4.2 Numerical Hardening
All inputs pass through `crates/sofuu-core/src/ml/sanitize_features`:
- Any `NaN`, `±Inf`, or malformed float is sanitized to `0.0`.
- Strings are head/tail capped to 2,000 characters before feature extraction to prevent denial-of-service from large files.

---

## 5. Training, Evaluation, and Baking Protocol

### Step 1: Retraining in `crates/ml-train`
Execute offline training using `cargo run -p ml-train --release`:
```bash
cargo run -p ml-train --release -- --dataset real_trajectories.jsonl --epochs 200 --gate all
```
The trainer evaluates each model against:
1. **Constant Baseline (Majority Class)**
2. **Standardized Logistic Regression**
3. **The Sofuu Micro-MLP**

*Pass Criteria:* The Micro-MLP must achieve statistically significant improvements in F1-score and Precision over both baselines on held-out test folds.

### Step 2: Weight Verification & Sanity Checks
Ensure the exported weights:
- Contain strictly finite 32-bit floats.
- Maintain zero bias drift.
- Are exported into:
  - `crates/sofuu-core/src/ml/compaction/weights_v2.f32`
  - `crates/sofuu-core/src/ml/freshness/weights_v2.f32`
  - `crates/sofuu-core/src/ml/relevance/weights_v2.f32`
  - `crates/sofuu-core/src/ml/alloc/weights_v2.f32`
  - `crates/sofuu-core/src/ml/supervisor/weights_v2.f32`

### Step 3: Shadow A/B Validation Harness
Add a non-intrusive logging harness in `crates/sofuu-core/src/ml/context.rs`:
- Runs both the heuristic decision and the Micro-MLP decision in parallel on live sessions.
- Logs divergence and measures actual token consumption over long-horizon tasks.

---

## 6. Execution Milestones

```
┌────────────────────────────────────────────────────────────────────────┐
│                     TINY ML UPGRADE MILESTONES                         │
├────────┬────────────────────────────────┬──────────────────────────────┤
│ Phase  │ Deliverable                    │ Completion Gate              │
├────────┼────────────────────────────────┼──────────────────────────────┤
│ M1     │ Collect & label 10,000 real-   │ Validated JSONL dataset      │
│        │ world agent trajectory steps   │ with zero synthetic labels   │
├────────┼────────────────────────────────┼──────────────────────────────┤
│ M2     │ Retrain 5 gates in ml-train    │ All 5 gates beat logistic    │
│        │ with cross-validation folds   │ regression by > 15% on F1    │
├────────┼────────────────────────────────┼──────────────────────────────┤
│ M3     │ Export `weights_v2.f32` and    │ `cargo check` and unit tests │
│        │ link into `sofuu-core`         │ pass with 0 size regressions │
├────────┼────────────────────────────────┼──────────────────────────────┤
│ M4     │ End-to-end token economy test  │ Verified ≥ 30% token savings │
│        │ on real 20-step coding tasks   │ on benchmark agent tasks     │
└────────┴────────────────────────────────┴──────────────────────────────┘
```
