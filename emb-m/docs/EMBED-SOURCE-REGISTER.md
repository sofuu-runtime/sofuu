# Embedding Dataset Source Register (2026-10-10)

Project doctrine: commercial training data allowlist = public domain, CC0,
MIT, BSD (any clause count), Apache-2.0, PSF License, **plus CC-BY-4.0 only
when needed and only with a complete per-record attribution chain**
(author, title, source URL, license link). ODC-By 1.0 corpora are treated
like CC-BY (attribution-only dataset license) but the underlying-content
caveat always applies: dataset license ≠ content license.

Verdicts: APPROVED (may enter commercial training) · CONDITIONAL (usable
only under the stated restriction) · REJECTED (may not enter; reason given,
no exceptions without re-audit) · UNVERIFIED (insufficient evidence).

Every APPROVED/CONDITIONAL source lists: size, access method, storage
estimate, and what it feeds (corpus = raw material, pairs = training
examples, eval = held-out measurement only).

---

## 1. CODE — retrieval of implementations from behavior descriptions

| Source | Size | License evidence | Verdict | Use |
|---|---|---|---|---|
| bigcode/the-stack-v2-train-smol-ids (streamed slice) | 17 langs, ~525B tokens total; we take a filtered slice (see pilot) | Per-file SPDX `detected_licenses` + `license_type`; BigCode permissive list (Blue Oak + ScanCode); bulk access gated by SWH agreement + opt-out updates | CONDITIONAL: `license_type==permissive` AND SPDX ∈ {MIT, Apache-2.0, BSD-*, CC0-0} AND repo has explicit LICENSE file (drop `no_license` — unlicensed ≠ licensed) | corpus (functions + docstrings → pairs) |
| github/CodeSearchNet (+ `_licenses.pkl` per language) | ~2M (comment, code), 6 langs; repo-partitioned splits | Per-language `_licenses.pkl` ships with the download; project code MIT (data licenses vary per file) | CONDITIONAL: same SPDX filter via `_licenses.pkl`; repo-split discipline reused for OUR splits | corpus + eval protocol reference |
| google-research-datasets/mbpp | ~1k Python problems + tests | CC-BY-4.0 (HF card + repo README) | APPROVED (attribution tracked) | eval only (too small to train) |
| openai/human-eval | 164 Python tasks | MIT | APPROVED | eval only |
| microsoft/CodeXGLUE (code-search tasks) | CSN-AdvTest 251k train + WebQueryTest 1k | Code MIT; datasets under C-UDA (Computational Use of Data Agreement) | CONDITIONAL: verify C-UDA text permits commercial model training before any training use; eval use lower-risk but still verify | eval first; train pending C-UDA verdict |
| Self-built crawl (MIT/Apache/BSD repos, SPDX + license-file evidence per record) | pilot-sized, we control it | license file hash + SPDX id per record (our own gate) | APPROVED by construction (this is the fallback if Stack slicing stalls) | corpus |

REJECTED for training: StaQC (CC-BY-SA), CoNaLa (research-only), raw
GitHub scrapes without license evidence, any StackOverflow content
(CC-BY-SA). Note: Stack v2 bulk download needs an SWH/INRIA agreement —
for the pilot we STREAM parquet slices (no 67TB download, no agreement
needed for sampled streaming; re-check before any bulk pull).

Pilot (code): stream smol-ids Python+JavaScript, keep MIT/Apache-2.0 +
explicit-LICENSE repos only, extract functions with docstrings ≥ 1
sentence, target 25–100k pairs; eval = MBPP-style task set + held-out-repo
retrieval + CSN-AdvTest protocol (MRR, license-checked).

---

## 2. DOCS — question-to-section retrieval over technical documentation

| Source | Size | License evidence | Verdict | Use |
|---|---|---|---|---|
| kubernetes/website | full K8s docs (concepts, tasks, API ref) | CC-BY-4.0 (repo LICENSE) | APPROVED (attribution tracked) | corpus + pairs (task query ← headings; API ref ← symbol queries) |
| Rust: rust-lang/book + std docs | full book + API docs | MIT/Apache-2.0 dual (rust-lang.org/policies/licenses) | APPROVED | corpus + pairs |
| Python docs (docs.python.org) | full language + library ref | PSF License v2 (permissive) + doc code 0BSD | APPROVED | corpus + pairs |
| Go documentation | language + stdlib ref | Go is BSD-3-Clause; docs ship with the distribution | CONDITIONAL: confirm docs/ dir carries the same BSD grant (one-file check) before ingesting | corpus (pending check) |
| React / PyTorch / HF docs | large, versioned | MIT/Apache-family per project | CONDITIONAL: per-project license-file check at ingest (docs ≠ code license — verify, don't assume) | corpus (pending checks) |

REJECTED: MDN (CC-BY-SA), StackOverflow/StackExchange dumps (CC-BY-SA),
MS MARCO passages (non-commercial research only — see §4), any scraped
tutorial blogs without explicit licenses.

Pairs construction (all docs sources): section → (question templates +
heading-derived queries, version-pinned); hard negatives = same-page
sibling sections + same-API different-version passages. Dedupe mirrored
pages; never mix breaking-change versions in one split.

---

## 3. SEARCH — general-purpose text retrieval

| Source | Size | License evidence | Verdict | Use |
|---|---|---|---|---|
| HuggingFaceFW/fineweb-edu (streamed) | 1.3T tokens educational subset | ODC-By 1.0 (attribution-only) + Common Crawl ToU; underlying pages keep own terms | CONDITIONAL: documents only (never reproduce long spans in outputs); attribution recorded; pairs need separately-sourced queries | corpus (documents for pair mining) |
| allenai/c4 (EN slice, streamed) | ~750GB | ODC-BY (same structure as above) | CONDITIONAL: same restrictions as FineWeb | corpus (fallback/supplement) |
| S2ORC via Semantic Scholar bulk API | 81M papers metadata; 8.1M OA full texts | ODC-By 1.0 current (was non-commercial — do NOT use old guidance); underlying per-paper rights vary | CONDITIONAL: metadata/abstracts for mining; full text only where OA status confirmed per record | corpus + citation-link pair mining |

REJECTED for training: MS MARCO + ORCAS + MS-MARCO-Web-Search (all
explicitly non-commercial research only), Natural Questions / SQuAD
(CC-BY-SA annotations), BEIR aggregates (research-only terms; individual
subsets re-audited case-by-case if needed for eval).

Pairs construction: title↔body, question↔passage mined from
permissively-sourced Q&A (self-built), plus synthetic query paraphrases
(traceable, validated — never dominant). Eval: SciFact + STS harness
already in repo (reusable as-is).

---

## 4. MEMORY — stored-fact retrieval for agents

| Source | Size | License evidence | Verdict | Use |
|---|---|---|---|---|
| xiaowu0162/LongMemEval | 500 cleaned QA (ICLR 2025) | MIT (repo + bundle card) | APPROVED | eval only |
| Self-built synthetic scenarios (per project spec §9) | pilot-sized, we author it | ours (CC0-equivalent provenance: authored in-repo) | APPROVED by construction | train + eval (disjoint scenario sets) |

REJECTED for training: LoCoMo (CC-BY-NC upstream — non-commercial covers
all product R&D, including eval), BEAM (CC-BY-SA), Multi-Session Chat
personas pipelines with unclear terms.

Rationale: no public agent-memory corpus is both large and
commercially clean; the spec already mandates synthetic/curated records
with timestamps and validity intervals. External data serves only as
eval (LongMemEval) and as scenario templates (rewritten, not copied).

---

## 5. MATH — definition/theorem/example retrieval

| Source | Size | License evidence | Verdict | Use |
|---|---|---|---|---|
| openai/gsm8k | 8.5k word problems + solutions | MIT (HF card) | APPROVED | eval + seed pairs (small) |
| hendrycks/math (MATH + AMPS) | 12.5k competition problems; AMPS pretraining set | MIT (repo LICENSE) | APPROVED | corpus + pairs (problem↔solution, definition queries) |
| OpenStax textbooks | full textbook corpus (algebra → calculus) | CC-BY (per-book license pages) | APPROVED (attribution tracked) | corpus (definitions, theorems, worked examples) |
| keirp/OpenWebMath (streamed slice) | 6.3M docs / 14.7B tokens | ODC-By 1.0 + Common Crawl ToU (same structure as FineWeb) | CONDITIONAL: documents only; LaTeX/notation preserved, never stripped | corpus |
| arXiv CC0/CC-BY slice (math.*) | large, grows monthly | per-paper license field in arXiv metadata/OAI | CONDITIONAL: ingest only `license ∈ {CC0, CC-BY}` records; preserve source + license per record | corpus (theorem/definition mining) |

REJECTED: MathPile + MathPile_Commercial (CC-BY-NC-SA / CC-BY-SA —
share-alike is incompatible with commercial training regardless of
attribution), Khan Academy (CC-BY-NC-ND), MIT OCW (CC-BY-NC-SA),
Mathematics StackExchange (CC-BY-SA), ProofWiki (CC-BY-SA/GFDL).

Notation rule (§10): store original LaTeX AND normalized text; never train
on notation-stripped renders.

---

## 6. SCIENCE — paper/abstract/methods retrieval

| Source | Size | License evidence | Verdict | Use |
|---|---|---|---|---|
| PMC Open Access `oa_comm` bulk (commercial-use slice) | millions of full texts (CC0/CC-BY filter) | Per-article machine-readable license; `oa_comm_*` packages = commercial-use subset | APPROVED for the CC0/CC-BY slice (license recorded per article; SA/NC excluded record-wise) | corpus (abstract↔body, methods↔results pairs) |
| arXiv CC0/CC-BY slice (all domains) | large | per-record license field | CONDITIONAL: same filter as math; version/preprint duplicates deduped (preprint↔published same work must not cross splits) | corpus |
| S2ORC metadata (titles/abstracts/citation graph) | 81M records | ODC-By 1.0 | CONDITIONAL: citation-link pair mining (citing-context → cited-abstract); full text only where OA confirmed | pairs mining + corpus |

REJECTED: full PMC OA without license filtering (contains NC/SA/custom),
unfiltered S2ORC full text, SciDocs/NFCorpus/TREC-COVID/BioASQ as training
(research-only terms — eval-only use requires separate per-benchmark
license check first).

---

## 7. LEGAL — clause/provision retrieval (deferred model; register now)

| Source | Size | License evidence | Verdict | Use |
|---|---|---|---|---|
| theatticusproject/cuad (+ cuad-qa) | 510 contracts, 13k+ expert labels, 41 clause types | CC-BY-4.0, commercial use explicitly allowed (underlying EDGAR contracts carry no contrary representation) | APPROVED (attribution tracked) | eval + seed pairs (small; the gold standard for clause retrieval) |
| SEC EDGAR direct (contracts, 10-K exhibits) | millions of filings | U.S. federal public records (public domain; no copyright in federal works) | APPROVED (jurisdiction=US-federal recorded per record) | corpus (clause mining with CUAD labels as distant supervision) |

REJECTED: pile-of-law bundle as a whole (CC-BY-NC-SA 4.0 dataset license
— non-commercial + share-alike; individual PD sources re-sourced directly
instead), CaseHOLD/MAUD/LegalBench for training until per-dataset
commercial-use evidence exists (eval use: verify each benchmark's terms
first). CourtListener-derived data: respect the no-indexing request —
never re-host in indexable form.

Jurisdiction/time rule (§12): every legal record carries jurisdiction +
validity dates; cross-jurisdiction pairs are hard negatives, not positives.

---

## 8. Pilot sourcing plan (what we pull first)

| Step | Source | Filter | Target | Storage |
|---|---|---|---|---|
| P-code | Stack-v2-smol stream | Python+JS, MIT/Apache-2.0, explicit LICENSE, docstring ≥ 1 sentence | 25–100k pairs | < 1 GB |
| P-code-eval | MBPP (CC-BY) + HumanEval (MIT) + held-out repos | as licensed | ~1.2k tasks + repo-retrieval set | MBs |
| P-docs | K8s (CC-BY) + Rust book (MIT/Apache) + Python docs (PSF) | version-pinned sections | 10–30k pairs | < 500 MB |
| P-math-seed | GSM8K + MATH (both MIT) + OpenStax (CC-BY) | as licensed | 20k+ pairs | < 500 MB |

No bulk downloads before the license gate + streaming slicer exist
(Phase 2 tooling). Estimated total pilot footprint: < 2 GB on the Mac;
raw corpora never land on the Mac beyond the filtered slice.

## 9. Standing rules for this register

1. A verdict changes only with new evidence recorded here (URL + date).
2. CONDITIONAL restrictions are enforced in code by the license gate, not
   by convention.
3. CC-BY-4.0 / ODC-By sources accumulate attribution entries in the
   attribution registry from first ingest (no retroactive reconstruction).
4. `no_license`, "publicly visible", "research-only", NC, ND, SA are never
   training-approved. Eval use of research-only benchmarks needs its own
   per-benchmark sign-off.
5. This register is dated 2026-10-10; licenses rot — re-verify any source
   untouched for > 6 months before first ingest.
