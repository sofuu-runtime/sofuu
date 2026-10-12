# PLAN-MULTIMODAL-EMBEDDINGS — Headless Embeddings + Image + Voice

> Status: planning record, approved for execution 2026-09-22 (owner).
> Owner decisions locked: **product first, SDK/headless first** (CLI is fast-follow);
> `ai.embed` supports both transports with **local default**; voice runs **both tracks
> together** (provider audio + OS speech bridges); image leads with **embeddings**;
> vision model stays **tiny distilled projector, 5MB binary cap intact**.
>
> Companion strategy: `PLAN-POSITIONING-2026.md` §4–§6 (structural edge: local,
> zero-marginal-cost memory vs token-metered cloud incumbents). Companion technical
> records: `PLAN-TINY-SEMANTIC-EMBEDDER.md` (SEM1/SEM2 recipe + budgets),
> `docs/EMBEDDING.md` (C ABI v2 contract), `PLAN-HEADLESS.md` (H0–H6 landed).
>
> Claim discipline (repo-wide): SHIPPED = implemented + tested here; PLANNED =
> specified below. Docs must never present PLANNED as SHIPPED.
>
> Status 2026-09-25: **Phases H-E1 + M1 + M2 + Q SHIPPED** (H-E1: native vector
> ABI, JS local-default + batch, Swift/Kotlin wrappers, c_embed_vec gate;
> M1: IMG1 projector R@5 1.000 on held-out tags, `sofuu_embed_image`,
> `ai.embedImage`, img1-64 space, multimodal e2e 9/9; M2: provider
> transcribe/speak + OS speech bridges + C ABI, voice e2e 12/12; Q: public
> scoreboard on BEIR SciFact + STS with a measured BM25 calibration row,
> published win-or-lose in README + landing `/docs#benchmarks`).
> Phase P remains PLANNED; agent-loop multimodal fusion and full-duplex
> voice_turn are fast-follow/sample-level.
>
> Status 2026-09-27 (SDK track): **the headless surface this plan is built on
> is now installable and truthful.** Two corrections worth recording, because
> this plan assumed both were fine:
> 1. **The brain was unreachable from C.** The plan's "SDK-first" premise
>    depended on `memory.*` existing in the funnel; 7 of 17 documented funnel
>    methods did not exist, and the brain was among them. Now fixed via a
>    handle facade and proven by an open→remember→recall→flush round-trip in
>    `examples/headless/c_embed.c`.
> 2. **The default space disagreed with the brain.** The C ABI defaulted to
>    `sem2-64` (64-dim) while the brain is 768-dim, so embed→remember was a
>    silent dimension mismatch. Default is now `hash-768`, matching the brain,
>    the docs, and the measured quality table.
> Install paths shipped: `Package.swift`, `Sofuu.podspec`, a Gradle AAR, an npm
> package for the CLI, and a typed Swift layer — all validated in CI. See
> `PLAN-HEADLESS.md` and `RELEASE-0.2.0-PREP.md`.

---

## 1. Objective

Make Sofuu's embeddings — text, then image, then voice I/O — a first-class
**headless SDK surface** (`libsofuu` C ABI + Swift/Kotlin wrappers) with local
offline as the default, so that:

- an app developer gets offline semantic search + visual recall + voice turns
  in ~2MB with no server, no key, no per-token meter;
- the JS surface (`ai.embed`, `ai.transcribe`/`ai.speak`) and the SDK surface share one
  space/version contract, so CLI prototypes and shipped apps interoperate;
- every new model artifact earns its place on a public eval or it does not ship
  (the SEM1/SEM2 rule).

## 2. What already exists (do not rebuild)

- **Text embeddings, bundled/offline**: hash-v1 768-dim (`sofuu.ai.embedLocal`),
  SEM1-64 (`embedLocalSemantic`), SEM2-64 (`embedLocalSemanticV2`), space
  manifests via `embedInfo`/`embedInfoV2`, golden vectors pinned. One-space-per
  index enforced per brain file (`brain.qtsq` vs `brain-v2.qtsq`).
- **Fused recall**: hash + SEM2 RRF at **0.917** on the internal 8-category suite
  (G1 0.750, G3-paths 0.667, G6 3/8 FAIL on record — fused ships second-channel
  only, never alone).
- **Vision input (API-mediated)**: user messages carry `images: [data URLs]`,
  mapped to OpenAI `image_url` parts and Anthropic base64 blocks, count/size
  capped (`rt/ai.rs` P2 multimodal — tested).
- **Headless plumbing**: `sofuu_rt_call`/`sofuu_rt_eval` JSON ABI, Swift + JNI
  bridges, platform packs (`scripts/dist/`), `abi_symbols.txt` baseline,
  `make headless-test`.
- **Voice**: nothing. Greenfield.

## 3. Hard constraints

1. **5MB CLI cap stands** (`make size-check`). No ONNX Runtime, PyTorch,
   Python, whisper.cpp, or network dependency in the shipped binary.
2. **No cloud embedding API, no per-token billing** (`PLAN-POSITIONING-2026`
   §8.3) — remote stays opt-in, local stays default.
3. **Space discipline** (extends the tiny-embedder rule): never mix embedding
   spaces in one index; every vector carries `{model_id, dim}`; cross-space
   writes refuse with a typed error, never silent corruption.
4. **No QTSQ-repo or desktop changes** unless a phase explicitly requires them.
5. **Embeddings are fingerprints, not content**: vectors are never presented as
   model input. Retrieval fetches context; pixels/text do the telling.

---

## Phase H-E1 — SDK-grade text embedding ABI (foundation)

**Goal:** an app embedding 10k docs never JSON-encodes a vector.

- `sofuu_embed_local(rt, text, space, float* out, size_t* dim)` — direct float
  output. Spaces: `sem2-64` (default), `hash-768`, `sem1-64`.
- `sofuu_embed_batch(rt, texts[], n, …)` — one call for ingestion; single lock
  acquisition, internally vectorized.
- `sofuu_embed_info()` — space manifest JSON (model ids, dims, artifact hashes).
- Swift (`SofuuBridge`) + Kotlin/JNI wrappers returning native float arrays.
- `scripts/dist/{macos,ios,android}.sh` + `docs/EMBEDDING-DIST.md` updated;
  `scripts/abi_symbols.txt` extended; new `examples/headless/c_embed_vec.c`
  in `make headless-test` (round-trip + batch + cross-space refusal).
- JS: `ai.embed` local by default; explicit provider/model selects network.
  Add `ai.embedBatch(texts[])` (local vectorized; network providers keep their
  own batching).

**Accept:** `c_embed_vec` green; ABI diff clean; batch 1k texts < 5s on
darwin-arm64; `make size-check` green.

## Phase M1 — Image embeddings (same ABI, new space)

**Goal:** "find that screenshot about QTSQ" — multimodal recall.

- Tiny distilled projector over frozen hand-built image features (color/edge/
  DCT/thumbnail stats), same ≤32KiB discipline as SEM1; trained offline against
  a CLIP-class teacher that never ships. New **image-only space** + manifest id.
- `sofuu_embed_image(rt, bytes, len, …)` C ABI + Swift/Kotlin from day one.
- Multimodal RRF fusion into brain/agent recall (text + image hit lists).
- Input ergonomics (fast-follow, same phase if cheap): file-path → data-URL
  helper for `images`; agent screenshot tool behind the OS permission model.
- Image **generation** stays API-mediated only — explicitly out of local scope.

**Accept:** text→image / image→text eval beats filename/tag baseline by an
owner-visible margin or it does not ship (the SEM1 rule); cross-modal golden
vectors pinned; size-check green.

## Phase M2 — Voice, both tracks together

**Goal:** speak to an agent, on-device where the OS allows it.

- CLI/API track: `ai.transcribe(audio)` + `ai.speak(text)` via provider audio
  endpoints (OpenAI-compatible wire; BYO key like completions).
- SDK track: bridge OS speech (Apple Speech framework / Android
  SpeechRecognizer + AVSpeech/TTS) through the Swift/Kotlin wrappers —
  on-device, free, private, zero binary cost. Sofuu orchestrates only.
- `sofuu_voice_turn` headless call (mic → STT partials → agent turn → TTS) +
  `ai.transcribe`/`ai.speak` JS for CLI prototyping. (Not `sofuu.voice.*` —
  no such namespace exists; voice lives under `sofuu.ai`.)
- No bundled whisper.cpp — recorded here as a deliberate no (10–50× binary).

**Accept:** CLI round-trip against a mock audio endpoint; Swift sample
transcribes on-device (manual QA on hardware); no size regression.

## Phase Q — Quality gate (before any marketing claim) — **SHIPPED 2026-09-25**

- ✅ **Public mini-benchmark shipped**: `ml-train bench` +
  `scripts/bench/fetch_datasets.sh` run **BEIR SciFact** (5,183 docs, 300
  judged queries, official qrels) and the **STS test split** (1,379 rated
  pairs) with a measured **BM25 reference** in the same harness. The
  BM25 row is the calibration check: 0.663 nDCG@10 against the ~0.665
  published for SciFact, so the harness agrees with the literature.
  Scoreboard (dims, params, KB, ms/embed, R@1/R@5, nDCG@10, Spearman ρ)
  published in README "Embedding benchmarks" and landing
  `/docs#benchmarks`.
- ❌ **Not run: the OpenAI `text-embedding-3-small` column.** It costs
  money per run and was not authorized. The docs say so and state the
  expectation plainly (a 1536-dim hosted embedder beats every row; the
  trade is a network round-trip, a key and cents per million documents).
- ✅ **Honest negative findings published rather than dropped**: on
  out-of-domain text the 64-dim learned spaces reach 12% (sem2) and 2%
  (sem1) of BM25's nDCG and lose to the 768-dim hash space they were
  meant to replace. The repo's own pre-registered `embed-eval` gate
  independently returns FAIL on 2 gates and says "do NOT replace the
  current embedder" — so the shipped default (hash-v1) was already right.
  The in-house suite (fused 0.917 vs hash 0.875) is labelled in-domain
  and is not comparable to the external table.
- ✅ **IMG1 labelled synthetic**: R@1 0.581 / R@5 1.000 on 93 held-out
  *procedural* scenes, published as a synthetic in-domain result. No
  real-image claim is made because no honest real-image number exists yet.
- ⬜ **G1/G3-paths/G6 teacher-pair closure is still open** — that work
  targets the semantic spaces, and the external benchmark says those
  spaces are not where the value is. Deprioritized deliberately rather
  than quietly dropped.

## Phase P — Proof surfaces + market

- Offline semantic-search sample (extend `SwiftSample`): airplane-mode demo.
- Offline iOS voice agent: OS STT/TTS + agent loop + encrypted memory.
- RAG-in-a-box recipe: embed folder → `.qtsq` store → answer offline.
- Landing SDK section ("offline semantic search in 2MB") + Swift/Kotlin
  snippets; content aimed at the WWDC26 indie-PCC cohort ("Apple gave you a
  free model, not memory"); cost contrast vs Mem0/Letta per-token metering.
  The benchmark table from Q is the strongest raw material here: the
  defensible claim is the *footprint* row (0 params, 0 bytes, 0.002 ms,
  no network), not a quality row.
- CLI fast-follow (does not gate revenue): semantic file search wired into
  `@file`/recall, `sofuu embed` one-liner.

---

## 6. Risks

| Risk | Hedge |
|---|---|
| Distilled image projector fails eval | Don't ship (SEM1 rule); image input + API path already cover "read this image" |
| OS speech API permission/latency variance | Provider-audio fallback in the same call; document per-platform behavior |
| Scope creep into generation/local STT | §5 explicit outs; size-check is the mechanical backstop |
| Docs promise > shipped surface | SHIPPED/PLANNED discipline (§0); scoreboard only publishes measured numbers |
