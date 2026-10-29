# Sofuu Runtime — Architectural, Intellectual Property & Legal Rules

> **Status:** Mandatory Repository Policy  
> **Applicability:** All runtime code (`crates/`, `src/js/`), C-API/FFI bindings, CLI binaries, examples, landing pages, documentation, and agent automations.

---

## 1. Architectural Neutrality: An Unopinionated AI Runtime

Sofuu is a **general-purpose, embeddable, native AI runtime for JavaScript and TypeScript**. It is designed to power any AI workload — autonomous agents, RAG systems, multi-step workflows, streaming microservices, background pipelines, desktop apps, and mobile SDKs (`libsofuu` for iOS and Android).

### 1.1 No Domain-Specific Hardcoding in the Core
- **Never bake domain assumptions into core runtime primitives.** Sofuu is not exclusively a coding assistant, terminal chat bot, or IDE extension.
- The core runtime (`src/js/agent.js`, `crates/sofuu-core/`, etc.) must remain domain-agnostic.
- Domain-specific behaviors (such as software engineering instructions, terminal patch applicators, or IDE checklists) belong **strictly in optional presets** (e.g., `preset: 'coding'`) or domain toolpacks.

### 1.2 Prompt Sovereignty
- **The developer's system prompt is sacred.** When a developer specifies `def.system`, it must be preserved 100% as written.
- The runtime must **never** prepend, append, or secretly inject coding guidelines, IDE instructions, or hidden rules into developer prompts.
- When `def.system` is omitted, the default must be a neutral, domain-agnostic assistant persona (`"You are a helpful AI assistant."`). Coding-specific prompts (`CORE_PROMPT`) are restricted solely to `preset: 'coding'` (e.g., used by the interactive `sofuu chat` CLI).
- Repository context injection (such as reading `AGENTS.md` into prompts) must only occur when explicitly requested via `preset: 'coding'` or `opts.agentsMd: true`.

### 1.3 Tool-Aware & Opt-In Task Gates
- Runtime loops must never nag, stall, or mandate specific checklist steps (such as `todo_write`) unless:
  1. The required tool is present in the active toolset.
  2. The developer has explicitly opted in (e.g., `budget.taskGate: true`).
- General-purpose agents default to `taskGate: false`.

### 1.4 Decoupled Cognitive Memory
- Memory operations (`remember`, `recall`, `search`) operate strictly on the on-device vector memory (`.qtsq` containers / Continuous Memory Architecture).
- The built-in `remember` tool must never write to external workspace markdown files (like `AGENTS.md`) unless operating under an explicit coding preset.

### 1.5 Modular Domain Toolpacks
- All built-in tools are organized into domain toolpacks (`sofuu.tools.PACKS`):
  - `coding` (`code`): file manipulation, diffs, bash, sandboxed eval, task checklist.
  - `web`: web search and web open.
  - `memory`: vector memory search and remember.
  - `system`: platform, GPU, CPU, and runtime inspection.
- Agents must never be forced to inherit coding tools. Non-coding agents receive zero coding tools by default unless requested.

### 1.6 Generalized Session Documents
- Sessions are stateful records for any AI task: chat, agents, workflows, pipelines, and background tasks.
- Session structures support generic step classifications (`input`, `output`, `kind`, `data`) alongside legacy chat aliases (`prompt`, `answer`), and provide standardized top-level `state` and `metadata` containers.

---

## 2. Intellectual Property, Copyright & Legal Safety

Sofuu maintains a strict zero-liability policy regarding intellectual property, copyright, trademarks, and third-party terms of service.

### 2.1 Brand & Trademark Safety in Binaries, Code, and Landing Pages
- **No Trademark Infringement:** Do not use third-party registered trademarks, proprietary product names, or confusingly similar brand names as feature names, binary names, or product claims.
- **Nominative Fair Use Only:** When referencing third-party protocols or hardware features (e.g., Apple Metal, ARM NEON, OpenAI API, Anthropic, Hugging Face, QuickJS, libuv), they must be referenced strictly in a descriptive, nominative fair-use capacity (e.g., *"OpenAI-compatible HTTP endpoint"*, *"Accelerated via Apple Metal"*). Never imply endorsement, sponsorship, or official affiliation.
- **Marketing & Landing Pages:** All public-facing documentation, landing pages (`sofuu.xyz`), README files, and CLI marketing text must be original, legally vetted, and free from misleading or proprietary brand terms.
- **No Proprietary Brand Piggybacking:** Do not use proprietary commercial tool or framework names as internal identifiers or public marketing terminology.

### 2.2 Clean-Room Implementation & Licensing Hygiene
- **Strict Clean-Room Architecture:** All algorithms, drivers, and runtime components must be authored from first principles, public technical specifications, or permissively licensed reference implementations.
- **No Incompatible Code:** Never copy code from GPL, AGPL, SSPL, or proprietary source code into the Sofuu codebase. All internal code must be compatible with the project's permissive open-source license.
- **Permissive Dependencies Only:** Any third-party dependency vendored or linked must carry a permissive license (MIT, Apache-2.0, BSD-2/3-Clause, or ISC) and retain appropriate attribution notices.

### 2.3 Model Weights & Datasets: Zero-Download Rule
- **Never download model weights.** Not small models, not fixtures, and not background downloads.
- Benchmarks, tests, and local inference must strictly rely on weights already present locally on the machine (see `AGENTS.md`).
- Never distribute, redistribute, or package copyrighted model weights or proprietary dataset files within git repositories, release tarballs, npm packages, or container images.

### 2.4 Prohibited Terms & Confidential Projects
- Under **no circumstances** may confidential or prohibited legacy codenames be touched, referenced, committed, or mentioned anywhere in the codebase, documentation, commit messages, or release artefacts. Refer to any such project only as `[CONFIDENTIAL_PROJECT_NAME]` or "a legacy codename".

---

## 3. Engineering & Conformance Standards

### 3.1 Binary Size Limit
- The standalone release binary (`./sofuu`) has a hard budget: **$\le 5.5\text{ MB}$ ($5,767,168$ bytes)**.
- Every release must be stripped and verified against this ceiling.

### 3.2 Bytecode Snapshot Conformance
- Shipped QuickJS bytecode snapshots (`src/js/*.bc` and `*.bc.gz`) must stay in 100% sync with their JavaScript sources.
- Verified before every commit via `make bc-check`.

### 3.3 Conformance Verification
- All modifications must pass the contract suite (`scripts/contract/run_contract_tests.sh`) and distribution checks (`scripts/verify-dist.sh`).
- Regressions in test coverage or stability index tiers are unacceptable.

---

## 4. Enforcement Checklist for Contributors & Agents

Before committing code or documentation, verify:
- [ ] **Is the feature domain-neutral?** If it is specific to software development, is it isolated to `preset: 'coding'` or `sofuu.tools.PACKS.coding`?
- [ ] **Is prompt sovereignty preserved?** Does the runtime leave developer-provided prompts untouched?
- [ ] **Are all names legally safe?** Are there any proprietary trademarks, copyrighted terms, or misleading branding claims in code strings, binaries, CLI flags, or documentation?
- [ ] **Is licensing clean?** Is all new code original or permissively licensed with clean provenance?
- [ ] **Were any models downloaded?** (Must be zero).
- [ ] **Did `make bc-check` and `scripts/verify-dist.sh` pass?**
- [ ] **Is binary size $\le 5.5\text{ MB}$?**
