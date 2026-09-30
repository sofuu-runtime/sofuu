# PLAN-POSITIONING-2026 — Sofuu's Path to a Billion-Dollar Product

> Status: RESEARCH DELIVERABLE / STRATEGY RECORD — delivered 2026-09-09.
> Researched 2026-09-08 via six parallel research tracks (~500 web fetches/searches:
> coding-agent market, on-device/embedded AI, AI-memory market, MCP/agent infra,
> runtime monetization comps, regulation/sovereign/funding).
> This is a positioning design record, not an implementation plan. Nothing here
> has been approved for execution; §12 lists the decisions that belong to the owner.
> Claim discipline: every market number in this doc carries a tag —
> **[V]** VERIFIED (multiple or primary source) · **[R]** REPORTED (single
> reputable source) · **[U]** RUMOR/UNVERIFIED. Appendix A holds the full fact base.

---

## §1 — Purpose and method

The question: how do we position Sofuu — main runtime, CLI, and headless
(embeddable) forms — to make it a billion-dollar product, using every lever
available as of 2026-09-08?

Method: six research agents fanned out over the live web on 2026-09-08, each
returning a tagged fact pack. Two had to be re-dispatched (one returned empty,
one hit a connection error); all six eventually completed. Where sources
conflict or a number rests on a single outlet, the tag says so. Positioning
rests on [V] claims wherever possible; every [R]/[U] dependency is flagged.

What this doc is NOT: a growth-hacking list, a "10 AI trends" post, or a
generic GTM template. Every recommendation below is anchored to a specific
verified market fact and to a specific Sofuu asset (§2).

---

## §2 — The asset inventory (positioning starts from what we actually hold)

What Sofuu uniquely is, as of 2026-09-09:

1. **A 2MB single-binary AI-native JS/TS runtime** — Rust shell, QuickJS C
   core, libuv, ~3ms startup, ~2.9MB darwin-arm64 CLI. Five engines at the
   C level with zero npm deps: LLM streaming (3 wire formats, BYO key),
   agent loop + sub-agents, MCP client AND server, HTTP server/client,
   web search, SIMD vector math, offline 768-dim embeddings.
2. **Pure interpreter — no JIT.** Apple's iOS executable-memory policy is
   satisfied by design; the C ABI (`libsofuu`) ships with Swift (iOS
   xcframework) and Kotlin/JNI (Android per-ABI .so) wrappers, proven via
   `make dist-ios` / `dist-android`. No other agent infrastructure ships
   this way (§4.2).
3. **QTSQ encrypted local persistence** — proprietary tensor-store format:
   brain (memory with decay/consolidation), sessions, vault, secure_text,
   deniable encryption, authorship anchoring. Portable encrypted "brain
   cards" (share/import). Cross-platform parity: macOS/Linux/Windows native
   builds, all four CLI binaries carry QTSQ (fingerprint-proven).
4. **Two-channel fused memory** — canonical hash-v1 + SEM2 semantic channel,
   RRF fusion, hash-only fallback always (deployed 2026-09-08; graded
   honestly: OVERALL 0.917 offline, G1/G3-paths/G6 failed on record).
5. **Five tiny on-device ML gates** (~8.5k params each, advise-only,
   zero-API) — freshness, supervisor, compaction, relevance, alloc — that
   cut context/token spend locally. The alloc model does per-model
   pre-flight fit checks against a per-model capability registry.
6. **Model-agnostic by construction** — endpoints are formats not companies
   (OpenAI/Anthropic/local wires), per-model caps registry, RLM routing,
   discovered-caps ladder. BYO key incl. local model servers. Zero telemetry.
7. **A session mesh** — real-time cross-session/cross-device context sharing
   over the encrypted local store; plus `serve --brain` HTTP mode with
   bearer-token auth.
8. **A Claude-Code-style CLI/TUI** — the free surface; ~46 e2e/pty test
   suites, audit-hardening in flight (both P0s + SSRF family closed 2026-09-08).

The thesis below exists because assets 2+3+5+6 form a combination nobody in
the verified competitive set (§4) ships.

---

## §3 — The market in six pictures (compressed; full facts in Appendix A)

### 3.1 Coding agents: huge spend, commoditized client layer

Enterprise AI spend hit $37B in 2025 (3.2× 2024), coding the largest app
category at $4.0B (up from $550M) **[V — Menlo Dec 2025]**. Claude Code holds
54% of coding-model share; Cursor reached ~$3B ARR **[R]** and exited at $60B
implied equity via SpaceX's option exercise **[V — SEC 8-K]**. But the
*client* layer is free everywhere: opencode ~206k stars MIT, Gemini CLI 1,000
free requests/day with 1M-token context **[V]**, Aider/Crush/Qwen Code all
free. Anthropic barred consumer Claude subscriptions in unauthorized
third-party harnesses in Feb 2026, forcing opencode to rip out support **[R]**
— thin-CLI-over-someone-else's-subscription is a policy-risk business.

The decisive signal: **Anthropic acquired Bun (Dec 2, 2025)** — a JS runtime
with $26M raised, **$0 revenue**, 7.2M monthly downloads — explicitly to make
Claude Code faster **[V]**. The pain is at the runtime layer, and distribution
has value without revenue.

### 3.2 On-device/embedded: free model, empty agent layer

Apple shipped the free offline ~3B Foundation Models framework with iOS 26
(Sept 2025) and ~16 third-party apps already ship features on it **[V — Apple
Newsroom 9/29/2025]**. WWDC 2026 went further: the Swift Language Model
protocol now accepts ANY provider, and Small Business Program apps (<2M
first-time downloads) get next-gen Apple FM on Private Cloud Compute at no
cloud API cost **[V — Apple WWDC26 guide]**. Apple commoditized the model —
but provides no agent loop, no MCP, no sub-agents, no cross-provider
portable memory, and is Swift-only.

Demand rails: Meta glasses ~7M sold 2025, ~13.4M-unit 2026 forecast, 10M+/yr
EssilorLuxottica target **[R]**; Figure $39B (Sept 2025), Physical
Intelligence $5.6B → ~$11B talks, Unitree ~$9B IPO **[all R]**; BlackBerry QNX
collects ~$950M/yr **[R]** in per-unit royalties across 275M vehicles **[V]**
— OEMs pay per-unit royalties for far dumber software than an agent runtime.
Edge-AI software market: $1.95B (2024) → ~$7.3–8.9B (2030/31), 24–29% CAGR
across three analysts **[R — they disagree]**.

The competitor scan found **no funded startup** shipping agent loop + MCP +
memory + model I/O as an embeddable local runtime — nearest adjacents are OSS
inference stacks (llama.cpp, ExecuTorch — which powers Instagram, WhatsApp,
Quest 3, Ray-Ban Meta **[V]**) and cloud memory layers. The slot is empty.

### 3.3 AI memory: monetizable layer, server-side incumbents, regulation tailwind

Memory is a funded category: Mem0 $24M raised (Oct 2025), 186M API calls in
Q3 2025, pricing Free → $19 → $249/mo **[V pricing / R traction]**; Letta
$20/mo + $0.10/agent/mo **[V]**; Zep sells SOC 2 + HIPAA BAA to Samsung-class
customers **[R]**; Supermemory to $399/mo **[V]**. Every funded incumbent holds
memory server-side and meters per token/request.

Context economics is an acknowledged industry problem: Anthropic's own Sept
2025 doctrine ("smallest set of high-signal tokens", compaction, sub-agents)
**[V]**, Chroma's context-rot study (18 LLMs degrade at every length increment)
**[V]**, cache reads priced 90–97.5% off **[V]**, Zep publishing 1.6K vs 115K
tokens per task (~90% cut) **[V — company-published]**.

Regulation converts memory from feature to budget line: EU AI Act Article 50
transparency live Aug 2, 2026 (watermarking grace Dec 2, 2026) with high-risk
Annex III deferred to Dec 2, 2027 **[R — Gibson Dunn]**; Colorado AI Act
effective June 30, 2026 **[V]**; Texas TRAIGA effective Jan 1, 2026 **[V]**.
Nobody in the verified set ships portable, encrypted, cross-provider memory.

### 3.4 MCP / agent infra: funded clouds, no runtime

LangChain $1.25B valuation (Oct 2025) **[V]**; OpenRouter $113M Series B at
$1.3B (May 2026) **[V]** with Stripe acquiring for $7B+ **[R]** — routing is
being absorbed into payments rails; CopilotKit $27M **[V]**; Composio $24M
**[V]**; Langfuse 50k+ companies, $29–$2,499/mo **[V]**. Vercel AI SDK: 21.6M
weekly npm downloads, free **[V]**. MCP itself: spec 2026-07-28, supported by
Claude/ChatGPT/VS Code/Cursor **[V]**, Registry still in preview **[V]**, no
foundation donation found — multi-steward under Anthropic/GitHub/Microsoft
**[V absence]**. Every funded incumbent is a cloud platform, hosted SDK, or
Python/TS library. **None is a single-binary local runtime.** Braintrust's
May 2026 breach **[V]** and Docker's Sept 2026 pivot to agent-harness
governance **[V]** both signal demand for controlled local agent execution.

### 3.5 Runtime monetization: runtimes don't bill, SDKs do

Bun: $26M raised, $0 revenue, monetized only via acquisition **[V]**. Deno:
$21M Series A, Deploy Pro $20/mo, subhosting 10B+ req/mo **[V pricing / R
volume]**. Node: 17 years, OpenJS, sells nothing **[V]**. The money is in
embedded-SDK economics: Mapbox $4/1K MAU **[V]**, Stream $399/mo @10K MAU +
$0.07–0.09/MAU **[V]**, Agora $0.05/MAU chat / $0.10/min conversational AI
**[V]**, RevenueCat 1% of tracked revenue **[V]**, QNX per-unit royalties
**[V/R]**. Local-first sync is a paid category post-Realm-deprecation (Sept
2024 **[V]**): PowerSync $49–599/mo **[V]**, Turso $4.99–417/mo **[V]**,
Obsidian Sync $4–8/mo **[V]**; 1Password at $400M+ ARR **[R]** proves
subscriptions for encrypted sync of owned data at scale. License-pivot
lessons: HashiCorp→IBM $6.4B and Redis→(forks)→AGPL-re-add both argue for
closed core + free surface.

### 3.6 Sovereign/regulated: where the multiples live

Palantir: FY2025 revenue $4.48B, market cap ~$411B (live 2026-09-08) — the
market pays ~65–90× revenue for sovereign-AI positioning **[V cap / R
financials]**. Microsoft sells "fully disconnected" sovereign tiers **[V]**;
Apple markets "Your data is never stored" **[V]** — big tech monetizes local-
first as premium. Gov AI sourcing is volatile (Anthropic $200M DoD July 2025
**[R]**; Feb–Mar 2026 stop-order → federal injunction **[R]**) — a neutral,
model-agnostic runtime is a hedge buyers understand. IBM: average breach
$4.99M (+12% YoY), AI-driven attacks +56% **[V]**. Agentic commerce is real
but early: x402 at 75.4M transactions but $24.2M volume in 30 days **[V]** —
attach to standards (MCP, x402, UCP), don't build rails. Funding climate
supports tiny technical founders: LiteLLM (2 founders, $1.6M seed → 50k
stars, Adobe) **[V]**; LiveKit $1B (Jan 2026) **[R]**.

---

## §4 — Strategic read

### 4.1 What's commoditized (do not compete here)

- **The coding-CLI client layer.** Free MIT competitors with 200k+ stars;
  Gemini CLI gives 1,000 req/day free. Cursor's exit at ~20× ARR was a
  distribution game with 100+ engineers. Anthropic's Feb 2026 harness
  crackdown shows the BYO-subscription thin-CLI model can be killed by a
  provider's ToS email. Sofuu's CLI stays free and serves a different job
  (§5).
- **Routing/gateway economics.** Stripe absorbing OpenRouter at $7B+ **[R]**
  means model I/O becomes payments-rail plumbing at commodity margins.
  Vercel AI SDK at 21.6M weekly downloads is free. Bundle routing, never
  bill for it.
- **The model itself.** Apple made a competent on-device model free
  (WWDC25) and opened the door to any-provider + free PCC (WWDC26). Any
  positioning that leads with "we have a model" is dead on arrival.

### 4.2 What's empty (the wedge)

The **embedded agent runtime** slot: a local, single-binary, embeddable SDK
that bundles agent loop + sub-agents + MCP + encrypted memory + multi-
provider I/O + context-economy ML. Across six research tracks, no funded
company ships this. The adjacents (LangChain, Mastra, CrewAI, CopilotKit —
TS/Python libraries and cloud SDKs; Mem0/Letta/Zep — cloud memory) all
require the developer's app to reach their cloud, their runtime debt, or
both. Sofuu is the only candidate that drops into an iOS binary at 2MB, no
JIT, no network dependency, keys never leaving the device.

Supporting vacuums with proven money attached:
- **Mobile local-first sync** — Realm's Sept 2024 deprecation left a paying
  gap (PowerSync/Turso fill it at $49+/mo on the server side; nobody fills
  it client-side with encrypted agent memory).
- **On-device orchestration for devices** — glasses at 13M units/yr,
  robotics at $39B valuations, 275M QNX cars all lack a standard agent
  layer; OEMs demonstrably pay per-unit royalties for less.
- **Portable encrypted agent memory** — every funded memory incumbent holds
  the brain server-side and meters it. None ships "your memory, encrypted,
  provider-switchable, exportable."

### 4.3 What's converging in our favor

1. Token/context economics became a named industry problem (Anthropic's
   doctrine, context-rot research, 90–97.5% cache discounts) — our five ML
   gates attack the customer's largest AI line item *locally*.
2. Privacy regulation became architecture-relevant (Art 50 live, Colorado,
   Texas) — encrypted local persistence + authorship anchoring + instant
   deletion is compliance posture, not a feature.
3. Cloud agent infra had a breach (Braintrust) and a governance panic
   (Docker's harness pivot) — local-first is now a trust differentiator,
   not a niche.
4. Apple's WWDC26 indie-PCC tier creates a beachhead cohort (small apps,
   free model access, no agent infrastructure to build on).
5. Runtime-layer value got top-tier validation (Anthropic×Bun).

---

## §5 — The positioning statement

> **Sofuu is the neutral, private agent runtime — the 2MB layer between any
> model (cloud, local, or on-device) and any app or device — that makes the
> agent's memory, tools, and token budget belong to the user instead of the
> provider.**

Three form factors, three roles:

| Form | Role in the strategy | Monetization |
|---|---|---|
| **CLI/TUI + desktop** | The funnel: credibility with developers, the live demo, free forever | Free (later: brain-sync Pro attach) |
| **Main runtime (`sofuu run`)** | The developer on-ramp: AI backends/servers written on Sofuu; adoption flywheel | Free binary; hosting is NOT a near-term line (Bun/Deno lesson) |
| **`libsofuu` headless** | **The product**: the embeddable agent-runtime SDK for apps (iOS/Android) and devices (glasses/robots/automotive) | Per-MAU, per-unit, site license |

The one-sentence version for each buyer:

- **App developer:** "Add a private AI agent with memory to your app in an
  afternoon — 2MB, no JIT, works offline, any model including Apple's free
  one, and the memory is yours (encrypted, exportable), not ours."
- **Device OEM:** "A standards-attached agent layer (MCP) that runs
  on-device next to your inference stack (ExecuTorch/llama.cpp), with
  encrypted local memory — per-unit royalty like QNX, 1/100th the size."
- **Regulated enterprise:** "AI agents that never phone home: BYO endpoint
  or air-gapped local models, encrypted persistence, cryptographically
  anchored authorship/records, instant deletion — procurement-ready."

Why this is defensible (each leg is structural, not a feature race):
- Cloud memory layers and gateways **earn on token volume**; a product whose
  core value is *cutting* tokens locally is structurally uncopyable by them
  without cutting their own revenue.
- Apple/Google structurally won't ship cross-provider neutrality or portable
  encrypted memory that makes the user (not the OS) own the brain.
- QTSQ closed core avoids the fork wars that maimed HashiCorp/Redis; the
  free CLI surface still earns distribution.

---

## §6 — Five revenue lines (comp-anchored pricing)

### Line 1 — App SDK, per-MAU (the lead line)

- **Buyer:** iOS/Android app developers — first target the WWDC26 indie-PCC
  cohort (free model access, nothing to build the agent on).
- **Offer:** libsofuu SPM/CocoaPods + Maven; private in-app agent: FM/local/
  OpenAI interchangeable, MCP tools, encrypted memory, sub-agents.
- **Pricing from comps:** Mapbox free ≤25K MAU then $4/1K MAU **[V]**; Stream
  free 1K MAU, $399/mo @10K **[V]**; Agora $0.05/MAU **[V]**. Sofuu shape:
  free ≤10K MAU → ~$0.01–0.05/MAU (priced under chat-SDK comps because we
  carry no per-message server cost).
- **Why it wins:** every comp bills because THEIR cloud carries the load;
  ours doesn't — our marginal cost per MAU is ~zero, so we can undercut
  every comp permanently and still hold margin.

### Line 2 — Device royalties, per-unit (the long game)

- **Order:** glasses → robotics → automotive (integration surface size,
  sales-cycle length).
- **Division of labor:** inference stays with ExecuTorch/llama.cpp (they
  power Quest 3 and Ray-Ban Meta today **[V]**); Sofuu owns the agent loop,
  tool/MCP layer, and encrypted memory around it.
- **Comp:** QNX ~$950M/yr across 275M vehicles **[R/V]** — per-unit royalties
  for less capable software are a 20-year-proven model; Cerence's −24% FY25
  **[R]** says the in-car assistant stack is being rebuilt right now.

### Line 3 — Brain sync + brain cards, subscription

- **Offer:** E2E-encrypted multi-device sync of the local brain (the session
  mesh + `/serve` are the core tech), plus portable brain cards.
- **Pricing:** consumer $4–15/mo (Obsidian Sync $4–8 **[V]**; 1Password
  $400M+ ARR **[R]** proves the category); business $8–20/seat (Bitwarden
  Enterprise $6, 1Password business tiers **[V]**).
- **Why it wins:** Mem0/Letta/Zep rent memory per token server-side
  ($19–249/mo **[V]**); we sell memory once, locally, and charge only for
  sync — a structural cost advantage no cloud layer can match. ChatGPT's
  900M WAU with cloud-locked memory **[R]** proves demand and exposes the
  weakness at once.

### Line 4 — Sovereign/regulated, site license

- **Offer:** the air-gapped agent suite — local models or BYO endpoint,
  encrypted persistence, authorship anchoring (maps to EU AI Act Art 50
  records/transparency duties), instant deletion (GDPR RTBF vs agent
  memory), audit logging. No cloud dependency anywhere in the binary.
- **Buyers:** defense (volatility hedge — the Feb–Mar 2026 stop-order saga
  **[R]** makes neutrality valuable), healthcare (Zep sells HIPAA BAA to
  Samsung-class buyers — the demand is proven **[R]**), legal/finance.
- **Comps:** Palantir ~65–90× revenue **[V/R]**; Microsoft's disconnected
  tiers **[V]**. This is where $100K–1M+/yr deals live; this line also sets
  the valuation multiple.
- **Note:** the current audit hardening (P0/P1 fixes) is literally this
  line's sales-readiness work.

### Line 5 — Token-economy guarantee, packaged as an SLA

- **Offer:** "same task, ≥X% fewer input tokens, signed export" — the five
  ML gates made measurable, built on the existing `/cost` machinery.
- **Grounding:** Anthropic's own doctrine **[V]**, Chroma context-rot
  **[V]**, Zep's 1.6K-vs-115K published delta **[V]**, cache economics at
  90–97.5% off **[V]** — the industry has already agreed the problem is
  real and is paying to mitigate it (caching) rather than solve it
  (context selection).
- **Structural protection:** every cloud layer earns on token volume; we
  are the only vendor paid *without* the meter running. An SLA here is
  also the strongest enterprise proof-point for Line 4.

---

## §7 — The honest math to "a billion dollars"

Three distinct meanings — keep them separate:

1. **$1B valuation** ≈ $30–60M ARR at 2025–26 agent-infra multiples
   (15–30× typical; LangChain $1.25B **[V]**, LiveKit $1B **[R]**,
   OpenRouter $1.3B→$7B exit **[V/R]**; sovereign stories reach 65×+
   **[V/R]**). A plausible year-3/4 mix:
   - ~2–3K SDK apps × avg 10–20K MAU × ~$0.02 → $5–12M
   - ~500K consumer sync subs × $8 → $48M (the swing factor)
   - ~1–2K business sync seats × $15 → modest
   - 30–80 regulated accounts × ~$150K → $5–12M
   - device pilots converting (glasses royalty at even $0.50/unit × 1M
     units = $500K/yr per design win)
   Even a conservative slice of this mix (say $35–50M ARR) clears $1B at
   20–30×.
2. **$1B revenue** = the device grind at QNX pace (20+ years to ~$950M
   **[R/V]**) riding the 24–29% CAGR edge market **[R]**. Honest horizon:
   not a 5-year plan. The plan's job is to make Sofuu *present* on every
   device when that market matures.
3. **Acquisition as a designed outcome, not a failure.** Bun→Anthropic
   ($0 revenue, runtime value) **[V]**; Browser Company→Atlassian $610M
   **[R]**; Windsurf→Google $2.4B acquihire **[R]**. Sofuu is buyable by
   every side *because* it is provider-neutral — OpenAI, Meta (glasses),
   Apple, Microsoft, and robotics companies all have a structural reason.
   The strategy deliberately keeps acquisition optional-value alive; it
   must NOT become the only plan.

---

## §8 — The negative space (what we deliberately do NOT do)

1. **No billing for the coding CLI** — commoditized client layer + harness-
   crackdown policy risk (§4.1). Free forever.
2. **No routing/gateway revenue** — Stripe×OpenRouter absorbs it into
   payments rails; margins commoditize to zero.
3. **No cloud memory API** — crowded (Mem0/Letta/Zep/Supermemory),
   token-metered, buyer-misaligned; it would forfeit the structural
   advantage (local memory, sync-only billing).
4. **No agentic-commerce rails** — x402 at $24M/30-day volume is too early
   to build on; ATTACH to MCP/x402/UCP standards instead (spec 2026-07-28,
   Shopify UCP already live **[V]**).
5. **No open-sourcing QTSQ** — closed core avoids the fork wars (HashiCorp
   → OpenTofu; Redis → Valkey → partial AGPL reversal — both **[V]**).
   Runtime shell could open later for adoption; the encrypted store and
   embed SDK stay closed.
6. **No leading with hosting** — Bun had $0 revenue at acquisition; Deno
   needed 10B req/mo subhosting scale to matter. Revisit only after the SDK
   line is proven.

---

## §9 — Sequencing

**Phase 0 (now, in flight)** — v0.2.0 hardening = sovereign-line sales
readiness. Finish the audit fix order (secrets/jail group next: P1-10 env
leak, P1-12 TOCTOU, js-5 grep/glob secrets; then memory-safety P1s), fix
P1-16 Linux TLS parity (device-OEM prerequisite), land P1-17 checksum
pinning. Ship v0.2.0 with the positioning story in the README.

**Phase 1 (0–6mo) — the iOS beachhead.** Package libsofuu as SPM +
CocoaPods with the Swift bridge polished; one flagship demo app (private
in-app agent: Apple FM / local / OpenAI interchangeable, encrypted memory,
MCP tool use); docs site section; free tier ≤10K MAU; CLI stays free.
Timing rides the WWDC26 indie-PCC cohort's search for agent infrastructure.

**Phase 2 (6–18mo)** — Android SDK (Maven + JNI); desktop SDK surface;
brain-sync GA (consumer $8/mo, business seats); first regulated design
partners (health/legal) on authorship-anchoring + local memory as
compliance architecture; publish the token-audit methodology as sales
collateral (Line 5 seed).

**Phase 3 (18mo+)** — device OEM pilots in order glasses → robotics →
automotive; sovereign/defense air-gapped packaging; reassess hosting and
open-core-shell decisions with real adoption data.

---

## §10 — Risks and hedges

| Risk | Severity | Hedge |
|---|---|---|
| **OS absorption** — Apple/Google ship agentic primitives (Swift LM protocol already points there) | The kill risk | Provider-neutrality vs Apple's locked stack; cross-OS portability; encrypted portable memory the OS structurally won't ship; MCP-standard attachment; win the indie cohort before the platforms care |
| Solo-founder scale (all of §9 is a lot) | High | The fundable story exists (§11); LiteLLM/LiveKit prove tiny-team infra scales; Phase 1 is deliberately narrow |
| QTSQ cold-start for developers | Medium | Free tier + JSON brain-card interop; closed core only where the value is (store + SDK) |
| Provider policy volatility (harness crackdowns, gov stop-orders) | Medium | Multi-provider + local-model support is ALREADY BUILT — it's the hedge, marketed as such |
| "Memory" unbudgeted at enterprises | Medium | Lead regulated sales with the compliance dates (Art 50, Colorado 6/30/26, Texas 1/1/26), not with "memory" |
| Wearables/robotics timing miss (device winters) | Medium | Lines 1/3/4 don't depend on devices; Line 2 is the long game by design |
| Report-card dependency: Cursor-exit-scale outcomes are rare | Low | §7 keeps valuation/revenue/acquisition separate; strategy doesn't require a $60B exit |

---

## §11 — The fundable story (for the day it's needed)

1. **Unoccupied category with rails laid:** every funded agent-infra company
   is a cloud platform or library (LangChain $1.25B, CopilotKit $27M,
   Composio $24M — all [V]); the embeddable agent-runtime slot is empty at
   the exact moment Apple made on-device models free and opened
   any-provider access (WWDC26, [V]).
2. **Adoption-first thesis:** Bun (7.2M downloads, $0 revenue) acquired by
   Anthropic; Vercel AI SDK 21.6M weekly downloads; our CLI + SDK free tier
   is built to generate the same curve.
3. **Sovereign demand is priced:** Palantir ~65–90× revenue; Microsoft sells
   disconnected tiers; regulated buyers need local encrypted agent memory
   the week Colorado/Texas/EU dates landed.
4. **Structural moat:** token-cutting economics that cloud incumbents
   cannot copy (their revenue is the meter) + closed encrypted store +
   no-JIT iOS eligibility.
5. **Tiny-team proof:** LiteLLM (2 founders, $1.6M seed → Adobe-class
   customers); LiveKit ($1B); BerriAI-style discipline already how this
   repo operates.

---

## §12 — Pending decisions (owner's call; nothing proceeds without these)

1. **Is acquisition-as-outcome acceptable to design for?** (§7.3 — it
   shapes how aggressively we court OEMs vs sell direct.)
2. **Which revenue line leads?** Recommendation: Line 1 (app SDK) with
   Line 4 (sovereign) as the second bet; Lines 3/5 ride along from Phase 2.
3. **Timing of the first fundraise** (if any) vs bootstrapping on design
   partners.
4. **When Phase 1 starts** — it gates on Phase 0 (v0.2.0 hardening) landing.
5. **README/site posture:** how loudly to lead with the positioning story
   before the SDK packaging exists (docs promise > shipped surface is a
   credibility risk).

---

## Appendix A — Fact base (by research track, tagged)

### A.1 Coding agents / CLI market

- Anthropic: $13B Series F @ $183B post (Sept 2, 2025) [V]; $30B @ $380B (Feb 12, 2026) [V]; $65B @ ~$965B (May 2026) [R — Reuters]; confidential IPO filing June 1, 2026 [R].
- Claude Code: GA May 2025 [V]; revenue "5.5×" by July 2025 [R — VentureBeat]; web version Oct 20, 2025 [V]; Anthropic holds 54% coding-model share, 40% enterprise LLM spend late 2025 [V — Menlo Dec 9, 2025].
- Anthropic acquired Bun Dec 2, 2025, "to improve Claude Code's speed and stability"; Claude Code ships as a Bun single-file executable [V/R].
- OpenAI: $12B annualized July 2025 [V]; $13.1B FY2025 [R — CNBC]; valuation $500B→$852B Oct 2025–Apr 2026 [R]; Codex CLI open-sourced Apr 16, 2025 [V]; Codex >2M WAU mid-Mar 2026, usage ×5 since start of 2026 [R — Reuters]; Codex at ~40% of Claude Code's level (from ~5% Sept 2025) [R — WIRED].
- Cursor: >$500M ARR @ $9.9B (Jun 5, 2025) [V]; >$1B annualized, $2.3B Series D @ $29.3B (Nov 13, 2025) [R]; ~$3B ARR (May 21, 2026) [R — Bloomberg]; SpaceX $60B all-stock deal closed Aug 14, 2026, implied equity value $60B [V — SEC 8-K; single-source detail R].
- Cursor pricing Free/$20 Pro/$40 Teams [V]; July 2025 usage-meter switch backlash [R].
- Windsurf→Cognition (July 14, 2025) after Google's $2.4B CEO acquihire [V/R]; rebranded Devin Desktop Jun 2, 2026 [V]; Cognition $10B→$26B (May 2026) [R].
- Gemini CLI: 106.9k stars; free 1,000 req/day, 60 req/min, 1M-token ctx [V].
- GitHub: ~80% of new devs use Copilot in first week; 180M+ devs; Copilot agent 1M+ PRs May–Sept 2025 [V]; AI-Credits usage billing June 1, 2026 + opt-out-only training default Apr 24, 2026 [V]; TypeScript GitHub's #1 language Aug 2025, 2.64M contributors +66.6% YoY [V].
- OSS CLIs: opencode ~206k stars/75+ providers [V] + Feb 2026 Anthropic harness bar → support removed [R]; Crush 28k stars [V]; Aider 48.8k stars, 6.8M installs, 15B tokens/wk [V]; OpenHands 86.9k stars, $5M Menlo seed [V]; Qwen Code 27.7k stars [V]; Amp $20/$200 with BYO-key tier [V].
- Agent browsers: Atlas launched Oct 21, 2025, merged into desktop app Mar 2026, shut down Aug 9, 2026 [V]; Comet free Oct 3, 2025 [V]; Atlassian×Browser Company $610M (Sept 2025) [R].
- Adoption: Stack Overflow 2025 (n≈49K): 84% use/plan AI, 51% of professionals daily, 46% distrust [V].
- Spend: enterprise AI $37B 2025 (3.2× 2024's $11.5B); coding largest app category $4.0B (from $550M); completion $2.3B; 76% bought-not-built; startups take 63% of app spend [V — Menlo Dec 2025]; AI-code-assist market $8.14B→$127.05B 2032, 48.1% CAGR [R — MarketsandMarkets].
- Quality counterweights: METR RCT — experienced OSS devs 19% slower with AI while believing +20% [R]; CodeRabbit — AI-co-authored PRs ~1.7× major issues, 2.74× security vulns (n=470) [R].
- Caching: Anthropic reads 0.1× (newest 0.025×), writes 1.25×/2× [V]; OpenAI GPT-5.6+ reads 0.1×, codex models 24-h retention [V].
- Thin-CLI monetization in the wild: free+OSS (opencode/Aider/OpenHands), bundled-inference subs (Amp/Cursor), usage-meter+BYO-key (Amp/Gemini CLI), corporate free (Qwen Code) [V].
- Lovable: $200M ARR ~8M users (Nov 2025); $6.6B round (Dec 2025); $13.3B (Aug 2026) [R].

### A.2 On-device / embedded

- Apple Foundation Models: free offline ~3B, iOS 26 Sept 2025 [V]; ~16 third-party apps shipping FM features by 9/29/2025 (OmniFocus 4, Agenda, SwingVision…) [V]; A17 Pro/M-floor, 9 languages [R].
- WWDC26: Swift LM protocol accepts any provider; Small Business Program apps <2M first-time downloads get next-gen FM on PCC free [V]; on-device model renamed "AFM 3 Core" (Jun 2026) [R].
- No-JIT shipping proven: ExecuTorch AOT .pte pipeline (XNNPACK + CoreML backends) powers Instagram, WhatsApp, Quest 3, Ray-Ban Meta [V]; llama.cpp ships official xcframeworks via SPM [V].
- Android: ML Kit GenAI = 6 on-device Gemini Nano capabilities [V]; Android 16 (6/10/2025), 17 (6/16/2026) [R].
- Edge-AI software market: $1.95B (2024)→$8.91B (2030) 29.2% CAGR [R — GVR]; $2.29B (2025)→$7.32B (2030) 26% [R — TBRC]; $8.89B (2031) 24.4% [R — MnM]. Analysts disagree; direction agrees.
- NPUs: AI-advanced PCs 39% of shipments 2025 → 59% 2026 [R — Counterpoint]; >100M AI-capable PCs 2025 [R — Canalys].
- Wearables: Ray-Ban Meta ~7M sold 2025 [R/U]; 10M/yr target late 2026, talks to double to 20M [R — Reuters/Bloomberg]; 13.4M 2026 forecast [R]; Oakley Meta 6/20/2025; Ray-Ban Display 9/18/2025; Meta Glasses $299 6/23/2026 [R]; Amazon Bee $49.99 CES 2026 relaunch [U]; OpenAI×io $6.5B all-equity (5/21/2025) [V].
- Robotics: Figure $1B+ @ $39B post (9/16/2025, Nvidia/Intel/Qualcomm/T-Mobile); Helix on-robot 2 GPUs; Helix 02 (1/27/2026); ~740 robots by 6/2026 [R]; Physical Intelligence $400M@2.4B (2024) → $600M@5.6B (Nov 2025) → ~$11B talks (Mar 2026) [R]; 1X NEO $20K/$499-mo, mostly teleoperated [R]; Unitree STAR IPO ~$9B debut Aug 2026 [R].
- Automotive: Cerence FY25 $251.8M (−24%); FY26 guide raised $305–320M [R]; BlackBerry QNX 275M vehicles [V], ~$950M figure (CNBC 8/25/2026) [R], ~$1B backlog, robotics fastest-growing segment [R].
- Competitor scan: NO funded "agent runtime for devices" startup found; adjacents = OSS inference stacks + cloud memory layers [R — absence across six tracks].

### A.3 Memory / context economics

- Mem0: $24M raised (Seed $3.9M + Series A, Oct 28, 2025; Basis Set/Peak XV/GitHub Fund/YC) [V]; 41K stars, 14M PyPI dl, 35M→186M API calls Q1→Q3 2025; AWS exclusive memory provider for its Agent SDK [R]; pricing Free (10K req) → $19 Starter → $249 Pro (graph) → Enterprise, all unlimited end users [V].
- Letta: stealth Sept 2024 (Jeff Dean/Felicis backers, size undisclosed) [R]; Free 3 agents → Pro $20 → API $20 + $0.10/agent + $0.00015/sec; Teams $20/seat [V].
- Zep: LoCoMo 94.7%/155ms/5,760 tokens; customers Samsung/Zscaler/Quorum/HoneyBook; SOC 2 II + HIPAA BAA + BYOC [R]; LongMemEval 71.2% vs 60.2% full-context, ~1.6K vs 115K tokens/task [V — company-published]; Graphiti 30.7K stars [V].
- Supermemory: Free → $19 → $100 → $399 (SOC 2/HIPAA/self-host); $0.005/1K tokens [V].
- Doctrine: Anthropic "Effective context engineering" (Sept 29, 2025) — attention budget, smallest high-signal token set, compaction, sub-agents [V]; Chroma Context Rot (July 2025) — 18 LLMs degrade at every length increment [V].
- Model-API spend $3.5B (Nov 2024) → $8.4B (mid-2025), >2× in ~7 months, within $13.8B genAI [V — Menlo 7/31/2025]; production shares: Anthropic 32%/OpenAI 25%/Google 20%/Llama 9%; Claude 42% of codegen; 74% of startups inference-dominant [V].
- ChatGPT: 900M WAU (Feb 2026); memory rollout Apr 2025 [R].
- Regulation: EU AI Act GPAI Aug 2, 2025 [V]; bulk Aug 2, 2026 with Annex III → Dec 2, 2027, Annex I → Aug 2, 2028 [V/R]; Art 50(2) labeling Dec 2, 2026 [V]; digital-omnibus final status unclear [U]; Colorado AI Act effective June 30, 2026 (moved by SB 4) [V]; Texas TRAIGA effective Jan 1, 2026 [V].
- Vault/sync comps: 1Password $250M→$400M+ ARR (Nov 2025), 100K+ business customers, $620M @ $6.8B (2022) [R]; Bitwarden $1.65/$4/$6 [V]; Obsidian Sync $4/$8 [V].
- MCP authorization spec: OAuth 2.1 + PKCE (June 18, 2025) [V].

### A.4 MCP / agent infra

- MCP: spec 2026-07-28, supported by Claude/ChatGPT/VS Code/Cursor [V]; Registry still in preview (2026-09-08) [V]; stewards Anthropic/GitHub/Microsoft/PulseMCP; no foundation donation found [V-absence]; Smithery 18,377+ servers, absorbed by Arcade.dev [V]; Docker MCP Catalog (Apr 2025) → MCP Enterprise Gateway (Sept 2026) [V]; Docker "Below the Harness" governance pivot (9/2/2026) [V]; Composio $24M Series A (Apr 2025) [V].
- Frameworks: LangChain $25M Series A (Jul 2025) [R] → $1.25B valuation on $125M (Oct 21, 2025) [V]; Mastra 27.8K stars, Replit/Salesforce/MongoDB usage claims [R]; CrewAI "billions of runs" Discovery marketing [R]; CopilotKit $27M (May 5, 2026) [V].
- Routing: OpenRouter $113M @ $1.3B (May 26, 2026; Sequoia/a16z/Menlo/CapitalG) [V]; 8M users, 400+ models [R]; Stripe acquiring for $7B+ [R — WSJ/Bloomberg]; earlier $40M @ $450M (Sept 2025) [U-not-reconfirmed].
- Vercel AI SDK `ai`: 21,639,648 weekly downloads (week ending 2026-09-06) [V — npm].
- Observability: Langfuse "50,000+ companies", $29/$199/$2,499 + $6–8/100k units [V]; Braintrust breach + forced key rotation (May 6, 2026) [V]; ClickHouse $15B valuation (1/16/2026), $250M ARR (5/27/2026) [V].
- Enterprise: ServiceNow AI Agent Orchestrator + Studio (Sept 2025) [V]; Meta "Muse" personal agent on dedicated cloud VM (9/8/2026) [V]; Salesforce Agentforce ARR claims — unverified [U].

### A.5 Runtime monetization comps

- Bun: $7M + $19M = $26M raised, **$0 revenue** at acquisition (founder's own blog, 12/2/2025) [V]; 7.2M monthly downloads +25% MoM (Oct 2025), npm 12.35M/month (9/8/2026) [V/R]; stays MIT under Anthropic [V]; "Bun Cloud/Deploy" as first-party product — not found [U].
- Deno: founded 3/29/2021; $21M Series A (Sequoia, 6/21/2022) [V]; Deploy Free/Pro $20/Builder $200 [V]; subhosting 10B+ req/mo (Slack/Netlify/GitHub/Supabase) [R]; 2.9.6 (8/27/2026), npm 577K/month [V].
- Node.js: no commercial entity since OpenJS 2019 [V] — 17 years, never billed.
- AI-app hosting (all fetched 9/8/2026): Cloudflare Workers $5/mo incl 10M req; Workers AI $0.011/1K neurons, Llama-3.3-70B $0.293/$2.253 per M in/out; Vercel Pro $20, Fluid $0.128/CPU-hr; Netlify $9/$20; Railway $5/$20; Modal $250 Team, CPU $0.0000131/core-sec; Replicate T4 $0.81/hr, H100 $5.49/hr; Baseten H100 ~$6.50/hr; LiveKit Cloud agent sessions $0.01/min (blended voice $0.0672/min) — all [V].
- Local-first sync: MongoDB deprecated Atlas Device Sync/Realm Sept 2024 [V; exact EOL U]; PowerSync Free 2GB → $49 → $599+, $1/GB, $30/1K concurrent [V]; Turso Free → $4.99 → $24.92 → $416.58 [V]; Electric joined Neon@Databricks (Aug 2026) [R].
- SDK economics: Mapbox free ≤25K MAU, $4/1K MAU (25–125K) [V]; Stream free 1K MAU, $399/mo @10K, $0.07–0.09/user [V]; Agora RTC $0.59/1K min, Conversational AI $0.10/min, Chat $0.05/MAU [V]; RevenueCat free ≤$2,500/mo tracked then 1% [V]; QNX per-unit royalty, 275M vehicles, free non-commercial since Jan 2025 [V/R].
- License lessons: HashiCorp BSL (Aug 2023) → OpenTofu fork → IBM $6.4B (closed 2/27/2025) [V]; Redis BSD→RSALv2/SSPLv1 (Mar 2024) → Valkey → AGPL re-added (5/1/2025) [V].

### A.6 Sovereign / regulated / funding

- Palantir: FY2025 revenue $4.48B, net $1.63B [R]; market cap ~$411B intraday 9/8/2026 [V]; Q2 FY26 revenue $1.94B, TTM $6.16B, P/E ~156 [V]; AIP since Apr 2023, US commercial +55% YoY (Q2 2024) [R].
- Anthropic gov: $200M DoD contract (July 2025) [R]; Feb 2026 classified-missions exclusivity → Hegseth supply-chain threat (2/27) → agency stop-order → Mar 26, 2026 federal injunction ("First Amendment retaliation") [R].
- Microsoft sovereign cloud: three tiers incl. "fully disconnected environments"; Azure Local on-prem AI; 100+ compliance offerings [V]; Apple marketing: "Your data is never stored" [V].
- IBM Cost of a Data Breach 2026: average $4.99M (+12% YoY); AI-driven attacks +56%; model-inversion avg $6M [V].
- Agentic commerce: x402 30-day (8/25/2026): 75.41M tx, $24.24M volume, 94.06K buyers, 22K sellers; Linux Foundation x402 Foundation [V]; Shopify Spring '26 (6/17/2026) Catalog API + UCP, merchants on ChatGPT/AI Mode/Copilot [V].
- Funding: LiveKit $100M Series C @ $1B (Index, Jan 2026), $183M total, powers ChatGPT voice [R]; LiteLLM 2 founders, $1.6M seed (YC W23), 50K+ stars, 100+ APIs, Adobe [V]; Cognition $10B→$26B [R]; Windsurf Google $2.4B acquihire [R].
- Sequoia AI 50 (4/10/2025): AI "graduates from answer engine to action engine" [R].

---

## Appendix B — Research gaps (what we could NOT verify; do not build on these)

- Claude Code absolute ARR; Anthropic overall revenue run-rate (never disclosed).
- Windsurf→Cognition price; Copilot ~15M users/$2B ARR (Microsoft sources blocked).
- "Bun Cloud/Deploy" as a first-party product (not found on bun.sh/bun.com).
- Turso/RevenueCat/Agora/Stream funding & ARR figures (search blocked).
- Exact Realm EOL date; Modal 2026 funding; Fly.io pricing.
- OpenRouter monthly request/token volume, revenue, margin; Vercel AI Gateway volume; Vercel valuation.
- LangChain/LangSmith ARR; CrewAI/LlamaIndex/Braintrust funding; Agentforce ARR; Copilot Studio revenue.
- YC official agent-company %; a16z/Sequoia agent-infra essays (sites timed out).
- MCP foundation donation (no evidence found — treated as NOT donated).
- Meta glasses 2024/2025 unit sales (weak sources only); io device shipping status; Limitless/Friend status.
- Digital-omnibus final legal status; Colorado/xAI lawsuit outcome; CA SB 243 specifics.
- Shadow-AI spend %; air-gapped coding-agent demand reports; median agent-infra round sizes/ARR multiples.

---

*End of record. Prepared 2026-09-09 from research conducted 2026-09-08.
Companion doc in-chat: the strategy walkthrough of 2026-09-09. Nothing herein is
approved for execution; see §12.*
