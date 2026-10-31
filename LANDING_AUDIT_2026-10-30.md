# Sofuu Landing Page & Documentation Audit

**Date:** 2026-10-30  
**Scope:** `sofuu-landing/` (17 page routes, layout, components, `public/`), the built export in `out/`, the live site at `https://sofuu.xyz`, the shipped 0.3.0-beta tarballs, the current `main` build, and the public machine-readable docs (`SKILL.md`, `llms.txt`, `llms-full.txt`).  
**Ground truth:** `RULES.md`, the runtime source, `sofuu.d.ts`, `docs/MCP.md`, `COMPAT.md`, `LICENSES/QTSQ-FORMAT.txt`, and direct execution of the shipped binary.  
**Status:** Read-only audit. No repository files were modified during the audit; this document is the only file added.  
**Verdict:** **Not shippable as-is.** Eight P0 defects, nine P1 defects, and a set of P2 polish/consistency issues. The technical build and SEO layer are strong; the problems are installer correctness, claim accuracy, legal/security wording, and docs-vs-artifact drift.

---

## 1. Executive summary

The landing site is a clean Next.js static export with good structural SEO: every real page has one `<h1>` and one canonical URL, the JSON-LD parses, the sitemap mirrors agree, redirects resolve, and the live content types are correct. `/runtime` and parts of `/compare` are unusually honest, publishing the measured 39.4 ms startup and the rows where Sofuu loses.

That quality is undermined by the following:

1. The **macOS `curl | sh` installer is broken** — it installs a directory named `sofuu` instead of the binary.
2. The **checksum files are malformed**, so the shell installer either aborts or proceeds without verifying; the site's "verifies SHA-256" claim is false for that path.
3. The **Terms page calls QTSQ "open source"** while the bundled license is a proprietary, all-rights-reserved license.
4. **Privacy and docs claim ML-KEM-768 encryption** for brain files; the actual brain path is AES-256-GCM with a machine-local key file.
5. The **`<4 ms` cold-start claim** is repeated across the homepage, `/docs`, `/compare`, and `SKILL.md`, directly contradicting the site's own measured 39.4 ms.
6. **Embed SDK install coordinates are unpublished** or wrong (0.3.0 vs published 0.2.0; missing Maven repository URL).
7. The **hero agent snippet does not run** — it throws on `agent.run` and passes undefined tools.
8. The **published binary is older than the docs** — it rejects `sofuu logs`, `sofuu resume`, and `sofuu plugin test`, which the docs describe.

In addition, `RULES.md` §3.1 is currently violated on `main`: the stripped build is **5.75 MiB**, over the 5.5 MiB hard cap, while the site claims 5.2 MB / ≤ 5.5 MiB.

---

## 2. Severity model

| Severity | Meaning |
| :--- | :--- |
| **P0** | Breaks the primary user journey, is factually/legally false, or would fail a reasonable release review. Fix before the next deploy. |
| **P1** | A documented claim or API is wrong in a way that wastes developer time, or a compliance claim does not hold. Fix in the next content pass. |
| **P2** | Consistency, accessibility, SEO polish, deploy hygiene, or unmeasured marketing copy. Batch these. |

---

## 3. Deployment state

- `sofuu-landing/` is a **separate git repository** (nested, ignored by the parent repo's `.gitignore`). A push to either repo does **not** deploy the site.
- Deployment is a **manual Cloudflare Pages upload**:

  ```bash
  cd sofuu-landing
  npm run build
  npx wrangler pages deploy out --project-name=sofuu-landing --branch=main --commit-dirty=true
  ```

- The last production deploy was **2026-10-29** (`199d52ff`). The working tree contains two uncommitted 2026-10-30 fixes: the homepage R7 overclaim and the `/docs` sandbox snippet. They are **not live**. The live site still serves:
  - "preventing agent hallucination escapes" (the R7 overclaim), and
  - the old `sofuu.sandbox.run` snippet with invented `fs: { mode: "jail" }` / `net: { allow: ... }` keys and "zero path escape".
- A build performed during the preceding investigation regenerated `out/` with those working-tree fixes, but the site was **not redeployed**.
- The two uncommitted fixes themselves are correct: I executed `sofuu.sandbox.run("1 + 1", { preset: "standard" })` against the shipped binary and confirmed the documented options (`fuel`, `timeoutMs`, `maxMemoryMB`, `maxStackKB`, `preset`, `console`, `ts`, `jsx`, `globals`) are real.

---

## 4. P0 findings

### P0-1 — macOS installer copies a directory instead of the binary

**What is wrong.** On macOS, `curl -fsSL https://sofuu.xyz/install | sh` ends with a directory named `sofuu` in the install path, not an executable. Running `sofuu version` then fails with "is a directory".

**Evidence.**

- The Darwin tarball extracts as `sofuu-darwin-arm64/` containing `sofuu`, `LICENSE`, `NOTICE`, `README.md`, `SHIP-MANIFEST.txt`, `LICENSES/`.
- The Linux tarballs put the binary at the archive root (`sofuu-linux-arm64`), which is why Linux works.
- `public/install:49` sets `ASSET_NAME="sofuu-${PLATFORM}-${ARCH}"`; at `public/install:119` the script does `cp -f "${TMP_DIR}/${ASSET_NAME}" "${INSTALL_DIR}/${BINARY}"`. On macOS `${TMP_DIR}/${ASSET_NAME}` is a directory.
- Reproduction with `SOFUU_INSTALL_DIR=/tmp/sofuu-install-test sh public/install`:
  - prints "Sofuu installed successfully!";
  - then `public/install: line 135: /tmp/sofuu-install-test/sofuu: is a directory`;
  - `ls` shows `/tmp/sofuu-install-test/sofuu` is a **directory**.

**Impact.** The primary install path is broken on the platform the project treats as its reference host.

**Fix.** Detect whether `${TMP_DIR}/${ASSET_NAME}` is a directory and use `${TMP_DIR}/${ASSET_NAME}/sofuu`; or standardise the macOS tarball to the Linux layout (binary at the archive root). Add a test that runs the installer for each archive layout.

---

### P0-2 — Checksum verification is ineffective

**What is wrong.** The published `.sha256` files contain only the hash and a newline, with no filename. `sha256sum -c` / `shasum -c` require `<hash>  <filename>`.

**Evidence.**

- `od -c public/downloads/sofuu-darwin-arm64.tar.gz.sha256` shows 64 hex characters followed by `\n` and nothing else.
- `shasum -a 256 -c sofuu-darwin-arm64.tar.gz.sha256` → `no properly formatted SHA checksum lines found`, exit **1**.
- `sha256sum -c sofuu-darwin-arm64.tar.gz.sha256` → `WARNING: 1 line is improperly formatted`, exit **0**.
- `public/install:72,75` uses exactly those commands.
- Consequence: on a Mac without GNU coreutils the installer aborts at the checksum step; on a system with GNU `sha256sum` it prints a warning and proceeds **without verifying**.
- The npm installer (`npm/install.js:198-205`) takes the first whitespace token and compares it to `sha256(tarball)`, so **npm verification works**; the shell path is the broken one.
- The docs' own instruction (`docs/page.tsx:304`) tells users to run `sha256sum -c sofuu-<platform>-<arch>.tar.gz.sha256`, which also fails.

**Impact.** A security claim ("The installer verifies the SHA-256 checksum") does not hold for the primary install path.

**Fix.** Regenerate every `.sha256` as `<hash>  <archive-name>`; add an install-integrity check to the release workflow that runs `sha256sum -c` against each published file; test both GNU and BSD paths.

---

### P0-3 — Terms page misstates QTSQ licensing

**What is wrong.** The Terms page calls QTSQ "open source" and says users may "freely use, inspect, and build on top of QTSQ memory containers". The bundled license is proprietary.

**Evidence.**

- `src/app/terms/page.tsx:38-42` — heading "Open Source QTSQ Memory Engine"; text says "is open source" and "You may freely use, inspect, and build on top".
- `LICENSES/QTSQ-FORMAT.txt` — **Quantum Tensor Sequence Proprietary License (QTSPL) v1.0**, "Copyright (c) 2026 Haruhito. All Rights Reserved." Free use is granted **only through the Sofuu Runtime**; competing implementations, format modification, and redistribution outside Sofuu are prohibited without a separate license.
- `public/llms.txt` states the correct position: "MIT core, with a proprietary QTSQ memory codec and format spec that are free to use through Sofuu."
- The CLI help also says "MIT License (QTSQ proprietary)".
- The `haruhito2666/qtsq` repository is Apache-2.0, but that is the codec repository, not the format license embedded in the Sofuu distribution. The Terms conflates the two.

**Impact.** A legal/IP misstatement on a public Terms page, directly contrary to `RULES.md` §2.2 and §2.1. It also implies rights the author has explicitly reserved.

**Fix.** Rewrite §3 to describe QTSPL: free use through Sofuu, proprietary format, no competing implementations, no redistribution outside Sofuu. Align §2's list ("local memory engine") with the same carve-out.

---

### P0-4 — ML-KEM-768 is claimed for brain encryption but is not the path used

**What is wrong.** Privacy and docs say brain data is encrypted with "AES-256-GCM and post-quantum ML-KEM-768 cryptography". The actual brain key path is a 32-byte machine-local key file plus AES-256-GCM.

**Evidence.**

- `src/app/privacy/page.tsx:37` — "stored locally in encrypted `.qtsq` containers using AES-256-GCM and post-quantum ML-KEM-768 cryptography."
- `src/app/docs/page.tsx:722` — "post-quantum encryption (ML-KEM-768 / AES-256-GCM)".
- `crates/sofuu-core/src/rt/brain_key.rs:1-20` — the brain key comes from `SOFUU_BRAIN_KEY` (64 hex chars) or `~/.sofuu/brain.key` (32 random bytes, mode 0600), and the vault is AES-256-GCM. The comment explicitly says the codec keystore's ML-KEM encrypt-to-self path is **a different mechanism** and was **not adopted** for brain files.

**Impact.** A security/privacy claim that overstates the cryptographic construction. This is the same class of overclaim the homepage correction was meant to eliminate.

**Fix.** State the real construction: AES-256-GCM with a machine-local 0600 key file or `SOFUU_BRAIN_KEY`. Mention ML-KEM only where it is actually used (the QTSQ keystore path), or remove it.

---

### P0-5 — `<4 ms` cold-start claim contradicts the site's own measurements

**What is wrong.** The homepage and several docs files claim a sub-4 ms cold start, including a receipt explicitly labelled "process to first response". The site's own `/runtime` page publishes **39.4 ms** for that exact measurement and says Sofuu is slower than Node and Bun.

**Evidence.**

- Homepage: `src/app/page.tsx:67` (FAQ JSON-LD), `:105` (code comment "Boots in <4ms"), `:172` ("<4 ms cold start"), `:315-316` ("Cold Start Latency | <4 ms"), `:511` ("Cold Start (process to first response) | <4 ms"), `:664` (visible FAQ).
- `/docs`: `src/app/docs/page.tsx:60` ("<4ms cold start" in the Quickstart code), `:378` ("instant cold starts").
- `/compare`: `src/app/compare/page.tsx:244` and `:331` (`<4 ms`), while `:362` says "Time to first HTTP response 39.4 ms" and marks Bun 25.8 ms as the winner.
- `/runtime`: `src/app/runtime/page.tsx:105` — "Wall clock from process spawn to the first HTTP response measured 39.4 ms in a same-harness comparison on this machine (Node 22: 32.4 ms, Bun 1.3: 25.8 ms), so Sofuu is not the fastest starter. The interpreter itself boots in single-digit milliseconds…"
- `/runtime:385` explains that "under 4 ms" refers to interpreter boot, not process start.
- `public/SKILL.md:36,121` repeats "~4 ms" / "under 4ms".

**Impact.** The live site contains both claims simultaneously. A visitor who checks `/runtime` sees the contradiction; a visitor who only reads the homepage is misled.

**Fix.** Pick one framing and apply it everywhere:
- "Interpreter boot <4 ms; process-to-first-response 39.4 ms (Node 32.4, Bun 25.8)" — the honest version; or
- keep `<4 ms` only in contexts explicitly labelled "in-process interpreter boot".

Purge the old number from homepage, `/compare`, `/docs`, and `SKILL.md` in the same pass.

---

### P0-6 — Embed SDK install coordinates do not resolve

**What is wrong.** The site tells users to install SwiftPM `from: "0.3.0"` and Gradle `com.sofuu:sofuu-android:0.3.0`. The published artifacts are 0.2.0, and the Gradle line omits the repository URL.

**Evidence.**

- `src/app/embed/page.tsx:23` — `.package(url: ".../sofuu.git", from: "0.3.0")`.
- `src/app/embed/page.tsx:30` — `implementation 'com.sofuu:sofuu-android:0.3.0'`.
- `src/app/docs/page.tsx:896-897` repeats both.
- `Package.swift:12` — the documented/committed version is `from: "0.2.0"`.
- `Sofuu.podspec:23` — `s.version = '0.2.0'`.
- Tags: `v0.3.0-beta`, `v0.1.0`, `android-sdk-v0.2.0`. There is no `0.3.0` semver tag, so SwiftPM `from: "0.3.0"` cannot resolve.
- `.github/workflows/android-maven-pages.yml` publishes **0.2.0** to a GitHub Pages Maven repository (`workflow_dispatch`, manual). The landing never tells users to add `maven { url ... }`, so Gradle cannot resolve the coordinate even at 0.2.0.
- `src/app/embed/page.tsx:90,99,101` — the `copyText` and the label say **0.2.0**, while the visible code says 0.3.0.

**Impact.** Every "Add it to your app" instruction fails for iOS/macOS and Android users.

**Fix.** Either publish 0.3.0 artifacts (AAR + SwiftPM tag + podspec + checksum) or change the site to 0.2.0. Add the Maven repository URL to the Android snippet. Make the visible code and `copyText` identical.

---

### P0-7 — The hero agent snippet does not run

**What is wrong.** The first code block on the homepage — the one a developer is most likely to copy — uses APIs that do not exist in the shipped binary.

**Evidence.**

- `src/app/page.tsx:97-101`:

  ```js
  const result = await sofuu.agent.run({
    prompt: "Summarise this project directory",
    tools:  [sofuu.tools.shell, sofuu.tools.fs],
    model:  { provider: "openai", model: "gpt-4o-mini" },
  });
  ```

- Executing it against the shipped 0.3.0-beta binary:
  - `agent.run(object)` → `THREW: agent.define: name is required`
  - `typeof sofuu.tools.shell` → `undefined`
  - `typeof sofuu.tools.fs` → `undefined`
- The real surface: `sofuu.tools.TOOLS` contains `read_file`, `write_file`, `edit_file`, `grep`, `glob`, `list_dir`, `bash`, `apply_patch`, `code_eval`, `todo_write`; agents are registered with `sofuu.agent.define(...)` and run with `sofuu.agent.run(name, task, opts?)`.
- `src/app/page.tsx:472` also documents `sofuu.agent.run(task, opts)`, missing the agent name.

**Impact.** The hero example is not reproducible; a developer's first run fails.

**Fix.** Replace with a runnable two-step example:

```js
sofuu.agent.define({
  name: "researcher",
  system: "You research precisely. Cite sources.",
  tools: ["web", "memory", "system"],
});
const result = await sofuu.agent.run("researcher", "Summarise this project directory");
```

and fix the API row to `sofuu.agent.run(name, task, opts?)`.

---

### P0-8 — The published binary is older than the docs

**What is wrong.** The docs and `SKILL.md` describe commands that the downloadable 0.3.0-beta binary rejects. The current repo build supports them, so this is a distribution lag, not just a docs error.

**Evidence.**

| Command | Shipped tarball (0.3.0-beta, Oct 29 build) | Current `main` build (Oct 30) |
| :--- | :--- | :--- |
| `sofuu logs` | `Unknown command 'logs'` | Lists runs (`RUN ID / STARTED / STEPS / STATUS`) |
| `sofuu resume` | `Unknown command 'resume'` | `requires a <run_id>` (command exists) |
| `sofuu plugin test` | `unknown plugin subcommand 'test' (list \| validate \| run \| init)` | `requires a tool name` (command exists) |

The site documents these at `src/app/docs/page.tsx:504-506`, `:557-575`, `:594`; `SKILL.md:44-51`, `:239`, `:253-263`.

**Impact.** A user who installs the published binary and follows the docs hits "Unknown command".

**Fix.** Republish the binaries from current `main` (after the installer/checksum fixes), or mark `logs`, `resume`, and `plugin test` as unreleased in the docs and `SKILL.md` until they ship.

---

## 5. P1 findings

### P1-1 — `RULES.md` §3.1 is violated on `main`, and the size copy is inconsistent

- Shipped macOS arm64 binary: **5,655,808 bytes = 5.39 MiB**, version `0.3.0-beta`.
- Current `main` stripped build: **6,032,976 bytes = 5.75 MiB**.
- Hard cap (`Makefile:49`, `RULES.md` §3.1): **5,767,168 bytes = 5.5 MiB**.
- Main is **265,808 bytes over the cap**.
- Site claims:
  - "5.2 MB stripped release" — `src/app/page.tsx:226`;
  - "5.2 MB (≤ 5.5 MB cap)" — `src/app/page.tsx:510`;
  - "5.2 MB" — `src/app/compare/page.tsx:237`;
  - "5.39 MiB" — `src/app/runtime/page.tsx:296`;
  - "5.4 MiB" — `src/app/docs/page.tsx:281` and the JSON-LD schema;
  - "5 MiB" — `src/app/ai/page.tsx:491`;
  - exact "5,655,792 bytes" — `src/app/runtime/page.tsx:101` (actual file is 5,655,808; 16 bytes off).
- "≤ 5.5 MiB" is true of the published tarball but false of the current source build.

**Fix.** Decide whether the claim is about the release artifact or `main`. If the release artifact: say 5.39 MiB and state the exact file. If `main`: fix the size first. Remove the 5.2 MB and 5 MiB figures. Also make `make size-check` actually run in CI (see P1-7).

### P1-2 — API reference drift in `/docs` and `/ai`

All of the following were tested against the shipped binary:

| Documented | Actual |
| :--- | :--- |
| `sofuu.ai.estimate(text)` (`docs:662`) | `undefined`; real name is `estimateTokens(text)` |
| `sofuu.ai.model()` / `sofuu.ai.resolve(name)` (`docs:663`) | both `undefined`; real names are `modelCaps(model, baseUrl?)` / `resolveCaps(...)` |
| `sofuu.http.serve(path)` (`docs:684`) | `undefined`; the runtime serving helper is `sofuu.serve.start` (the C ABI has `http.serve`) |
| `sofuu.kv.open(path)` + `kv.get/set` (`docs:733-734`) | `sofuu.kv.open` returns `save/search/flush`; the KV facade is `sofuu.storage.open({base_path}).kv.get/set` |
| `process.versions.node` (`'22.0.0'`) (`docs:638`) | `process.versions` is `undefined`; `process.version` is `"0.3.0-beta"`; `node:process` does not resolve |
| `/ai` `r.text; r.tokens` with usage `{input, output}` (`ai:189`) | result keys are `text, usage, raw`; `tokens` is `undefined`; usage is `{promptTokens, completionTokens}` |
| `similarity` "Float32Array only; anything else throws" (`docs:660`) | a plain `[1,2]` array returns `1`; the claim is false |
| `client.call("read_file", {path})` (`docs:847`) | real signature is `call(method, params, opts)`; the correct call is `call("tools/call", { name, arguments })` (the `/mcp` page does it correctly) |

Verified working: `ai.embed` defaults to the offline SEM2-64 model; `embedBatch` returns one `Float32Array(64)` per input; `embedLocal` is `Float32Array(768)`; `embedLocalSemantic` / `embedLocalSemanticV2` are 64-dim; `embedImage`, `transcribe`, `speak`, `confidence`, `tool`/`start` on `mcp.serve`, `memory.consolidate/count/stats`, `session.import/export`, `agent.cancel/renderTrace`, `rlm.query`, `ml.info`, `ml.freshness.score`, and `sandbox.run/eval/create` all exist.

**Fix.** Correct the wrong names and examples; prefer copying verified snippets from a test that executes against the shipped binary.

### P1-3 — `/mcp` says the legacy HTTP+SSE transport is refused; it is supported

- `src/app/mcp/page.tsx:106` — FAQ: "The deprecated legacy HTTP+SSE transport is refused with a typed E_TRANSPORT error rather than silently falling back."
- `src/app/mcp/page.tsx:343` — security table: "Deprecated HTTP+SSE is refused with a typed E_TRANSPORT; no silent downgrade."
- `src/app/mcp/page.tsx:379-380` — limits: "HTTP+SSE from before the Streamable HTTP revision is refused, not emulated. If you depend on it, use a Node or Python SDK."
- Reality: `docs/MCP.md:38` documents the `sse` transport (protocol `2024-11-05`), and `src/js/tools.js:1936` implements it (`M3 (2026-10-29): legacy HTTP+SSE transport`). The correct nuance is "never inferred from a bare URL; request `transport: 'sse'` explicitly".

**Fix.** Replace "refused" with "supported only when requested explicitly"; keep the "no silent downgrade" point.

### P1-4 — Security defaults are overclaimed in `/docs` and `/privacy`

Policy defaults are permissive unless configured:

- `net.allow` empty ⇒ all hosts allowed (`rt/policy.rs:295-310`); only `net.enabled: false` blocks everything.
- `net.ssrf` defaults false ⇒ private/loopback addresses are not refused by default (`rt/policy.rs:1206-1233`).
- `mcp.servers.allow` empty ⇒ all servers allowed (`rt/policy.rs:315`; `sofuu.d.ts:951-953`: "When non-empty, only these servers may connect").

Yet the site says:

- `src/app/docs/page.tsx:823` — "Prevents path traversal (`../`) and symlink breakout" (absolute).
- `src/app/docs/page.tsx:827` — "Guaranteed termination of unresponsive tasks."
- `src/app/docs/page.tsx:824` — "Scripts can only connect to explicitly permitted hosts" / "internal network scanning blocked".
- `src/app/privacy/page.tsx:50` — "Filesystem access is strictly jailed… network access is restricted to user-approved domain allowlists. Plugins cannot silently exfiltrate…"
- `src/app/mcp/page.tsx:338` — "A non-allowlisted server is refused."

**Impact.** This is the same overclaim class the homepage correction addressed; `/docs` and `/privacy` still carry it.

**Fix.** State the defaults: "network is open unless `net.allow` is configured; `--no-net` blocks everything; SSRF protection is opt-in via `net.ssrf`." Replace "prevents"/"guarantees"/"cannot" with "refuses when configured" and link the threat model.

### P1-5 — `/downloads` 404s

Referenced in three places but absent as a route:

- `public/install:7` — "https://sofuu.xyz/downloads (tarball + .sha256, any platform)".
- `public/llms.txt` — "Downloads: https://sofuu.xyz/downloads".
- `src/app/layout.tsx:169` — JSON-LD `downloadUrl: https://sofuu.xyz/downloads`.

Live check: `GET /downloads` → **404**. The actual files are `/downloads/*.tar.gz`.

**Fix.** Add a `/downloads` route (or a redirect to `/#download`), and update the install-script header/llms.txt to the real path.

### P1-6 — `/docs#mcp` is a dead anchor

- `src/app/page.tsx:480` links `/docs#mcp`.
- The docs page has no `id="mcp"`; MCP content lives under `id="advanced"` (`src/app/docs/page.tsx:836`).
- The built-export crawl found exactly one missing anchor: `/docs#mcp`.

**Fix.** Point the row at `/docs#advanced` or add `id="mcp"` to the MCP subsection.

### P1-7 — CI claims do not hold on current `main`

- `/runtime:101` — "Every release is stripped and checked by `make size-check` in CI."
- `/runtime:258` — "A bytecode consistency gate keeps snapshots and sources identical in CI."

Latest run (`41066226596`, 2026-10-30T13:47:52Z) is a **failure**:

- macOS job fails at **Build Sofuu (Rust-first)**; all later steps are **skipped**: Size cap, Bytecode determinism (bc-check), Policy DX, SBOM, verify-dist, headless embedding, sandbox adversarial, JS suite, determinism, contract, conformance, scoreboard gate, ABI, install integrity, Swift/JNI/manifest checks.
- Linux x86_64 and arm64 jobs fail at their build step.
- iOS typecheck passes; Windows is a known non-blocking fix-forward failure.

**Fix.** Either green the CI build (the QTSQ-free build blocker is known) or change the copy to "the gate exists in CI" without implying it passed. `RULES.md` §3.2/§3.3 require both gates to pass.

### P1-8 — `/embed` claims a doc-drift test that does not exist

- `src/app/embed/page.tsx:168` — "No doc drift: A test parses the method table out of the embedding doc and fails if a listed method doesn't resolve."
- Search across `tests/`, `scripts/`, `crates/`, `Makefile`, `.github/` found no such test. The only related hit is a C-API comment (`crates/sofuu-capi/src/lib.rs:5513`) about header/implementation drift, which is not a docs-method test.

**Fix.** Remove the claim or add the test (parse `docs/EMBEDDING.md`'s method table and assert every method resolves on the built library).

### P1-9 — Machine-readable docs propagate the errors

- `public/SKILL.md` (handed to AI coding assistants):
  - `<4 ms` / "~4 ms" at lines 36 and 121;
  - `sofuu logs`, `sofuu logs show`, `sofuu logs --last-error`, `sofuu logs --follow`, `sofuu resume` at lines 44-48;
  - `sofuu plugin test` at line 51;
  - `sofuu.kv.open` + `kv.set/get` at lines 132-134;
  - `sofuu.http.serve` at line 181;
  - `from: "0.3.0"` SwiftPM and `com.sofuu:sofuu-android:0.3.0` at lines 279 and 284.
- `public/llms.txt`:
  - `https://sofuu.xyz/downloads` → 404;
  - `https://github.com/sofuu-runtime/landing-page` → 404;
  - the "(Claude, Cursor, Copilot, Antigravity)" tag (see P2-6).
- `public/llms-full.txt` is the most accurate machine file (uses 24 ms / 39.4 ms and avoids `kv.get`), but still has the Android 0.3.0 coordinate at line 462.

**Impact.** AI assistants and search crawlers ingest these files; the errors will be repeated back to users.

**Fix.** Regenerate `SKILL.md` from the same verified snippets as the site; fix the two 404 links in `llms.txt`; update the Android coordinate.

---

## 6. P2 findings

### P2-1 — Consistency

- **RSS:** homepage "Resident Memory (idle boot) 12.4 MB measured" (`page.tsx:512`) vs `/runtime` and `/compare` "14.9 MB after serving". These may be two different metrics, but the homepage gives no method or source; label it or use the measured after-serving figure.
- **Version labels:** `embed/page.tsx` visible code says 0.3.0, `copyText` says 0.2.0 (lines 90, 99, 101); `llms-full.txt:462` says Android 0.3.0.
- **Size copy:** 5.2 MB, 5.39 MiB, 5.4 MiB, and 5 MiB all appear (see P1-1).
- **Tool names:** homepage shorthand `read`, `write`, `edit`, `bash`, `web_search` vs the registry's `read_file`, `write_file`, `edit_file`, `bash`, `web_search`; `/docs` has the correct names.
- **`memory.remember(text)` works** (the runtime auto-embeds text), so the homepage row at `page.tsx:477` is not a bug — do not "fix" it.

### P2-2 — Copy quality and unmeasured claims

- `/polling` snippet calls `sofuu.ai.stream("Explain gravity.")` with no provider/model, while `/ai:182` says "A model is required — there is no default"; the snippet also uses an undefined `res` (`src/app/polling/page.tsx`, code block).
- `/simd` says Node/V8 "requires V8's JIT compiler to step element-by-element, leading to horrible memory thrashing" — technically misleading; V8 optimizes typed-array loops.
- `/server` meta says "sub-millisecond response times" and "huge bursts of backend traffic effortlessly" — unmeasured.
- `/embedded`, `/simd`, `/polling`, `/server` are thin legacy pages with custom nav, no footer, no structured data, yet are in the sitemap at priority 0.6-0.7.

### P2-3 — Accessibility

- Contrast on `#faf0e6`:
  - `--text-faint #a29383` = **2.65:1**;
  - `--accent-orange #ff5c00` = **2.75:1**;
  - experimental label `#c9873a` = **2.66:1**.
  All fail WCAG AA for normal text (4.5:1) and are used at small sizes. Body text `#6e6459` (5.14:1) and moss (6.11:1) pass.
- Mobile menu: `.menuCheckbox { display: none }` (`page.module.css:45`) removes the checkbox from the tab order; the label has no text/aria name and there is no `aria-expanded`/`aria-controls`.
- `CodeBlock` copy button is icon-only (`title` only, no `aria-label`); the copied state is not announced (`src/components/CodeBlock.tsx:38-44`).
- `InstallModal` has no `role="dialog"`, `aria-modal`, focus trap, or Escape handler (`src/components/InstallModal.tsx`).
- No skip link; `html { scroll-behavior: smooth }` without a `prefers-reduced-motion` override (`globals.css`). The hero canvas does respect reduced motion, which is good.
- Body text is `font-weight: 600` globally; a design choice that can reduce legibility at small sizes.

### P2-4 — SEO / metadata

- **Canonical on non-content pages:** 404 and redirect pages inherit the homepage canonical from `layout.tsx:107-109`; a 404 should not canonicalize to `/`.
- **`/docs` structure:** the largest page has no `TechArticle` or `BreadcrumbList` JSON-LD.
- **Sitemap:** `public/sitemap.xml` lastmod is stale (Oct 28-29) and shadows the generated `sitemap.ts`; `sitemap.ts` uses build-time `new Date()`, so lastmod is not content-derived. The URL sets are in sync (13 each).
- **robots:** `Host: https://sofuu.xyz` should be the bare host (`sofuu.xyz`) per Yandex; Google ignores it.
- **Manifest:** one icon with `sizes: "any"` (no 192/512); `theme_color #0d0d0d` conflicts with the page viewport theme `#faf0e6`.
- **Open Graph/Twitter:** `/embed`, `/simd`, `/polling`, `/server`, `/embedded`, `/privacy`, `/terms` have no Twitter card block (some have OG).
- **External link:** the COMPAT.md link in `/docs` uses `rel="noreferrer"` without `noopener` and points at `haruhito2666/sofuu` (301s to canonical).

### P2-5 — Deploy hygiene

- `public/ascii-art.png` is **2.78 MB and unreferenced**; `file.svg`, `globe.svg`, `next.svg`, `vercel.svg`, `window.svg` are unreferenced Next boilerplate. All ship on every deploy.
- `public/wiscove/appcast.xml` points at `https://sofuu.xyz/wiscove/Wiscove-0.1.0.zip`, which **404s**.
- `package.json` repository `https://github.com/sofuu-runtime/landing-page` **404s**.
- `public/_headers` has no CSP, HSTS, X-Frame-Options, or Permissions-Policy. Cloudflare adds `nosniff` and `referrer-policy`, but not CSP.
- `public/install` and `public/install.sh` are identical duplicates; keep one canonical copy.
- `eslint` reports **235 `react/no-unescaped-entities` errors** (up from 51); `next build` does not lint, so they are dormant, but this is not a clean gate.

### P2-6 — Licensing / brand hygiene (RULES §2.1, §2.2)

- **Satoshi font:** `src/app/fonts/Satoshi-Variable*.woff2` ships with no license/attribution file. Fontshare's ITF license is not in `RULES.md` §2.2's permissive list (MIT/Apache-2.0/BSD/ISC); verify terms before redistributing the woff2 in the repo and the deploy artifact.
- **MicroSlats:** `src/components/MicroSlats.jsx` is described in code as a "React Bits variant" and has no license header or attribution. If it is third-party code, add the required notice or replace it.
- **Brand tag:** "Claude · Cursor · Copilot · Antigravity" on `docs/page.tsx:313` and in `llms.txt`/`SKILL.md` risks implying endorsement or affiliation. RULES §2.1 permits nominative references but not implied sponsorship. Prefer "works with assistants that read SKILL.md" or list them without a compatibility badge.
- `ogl` is MIT (fine).

---

## 7. `RULES.md` compliance matrix

| Rule | Verdict | Notes |
| :--- | :--- | :--- |
| §1.1 Domain neutrality | **Mostly pass** | AI positioning is marketing, not core hardcoding; "AI-specific general-purpose runtime" in the footer is semantically contradictory but not a core violation. |
| §1.2 Prompt sovereignty | **Pass** | `/ai` and `/docs` claim byte-for-byte preservation; the docs example uses `taskGate: false`. |
| §1.3 Opt-in task gates | **Pass** | Explicit `taskGate: false` shown. |
| §1.4 Decoupled memory | **Pass** | No claim that memory writes AGENTS.md. |
| §1.5 Modular toolpacks | **Pass, minor** | `/docs` table matches PACKS; homepage shorthand names are inconsistent. |
| §2.1 Brand/trademark safety | **Conditional** | Comparisons are nominative. The "Claude · Cursor · Copilot · Antigravity" tag risks implied affiliation. |
| §2.2 Clean-room / licensing | **FAIL** | Terms calls proprietary QTSQ "open source"; font and MicroSlats license texts/attribution are missing. |
| §2.3 Zero-download weights | **Pass** | No weights distributed or linked. |
| §2.4 Prohibited codenames | **No hits found** | Cannot prove exhaustively; nothing suspicious on the site. |
| §3.1 Binary ≤ 5.5 MB | **FAIL on `main`** | Current stripped build 6,032,976 B vs 5,767,168 B cap; site claims 5.2/5.39/≤5.5. |
| §3.2 Bytecode snapshot sync | **Unverified** | Site says the gate is in CI; the CI step is skipped. |
| §3.3 Conformance / verify-dist | **Unverified** | Site says CI checks; CI is red at build and every later gate is skipped. `/embed`'s claimed doc-drift test does not exist. |
| §4 Enforcement checklist | **Open** | Size, licensing, and claim accuracy all have open items. |

---

## 8. What is solid

- `npm run build` produces 20 static routes cleanly.
- Every real page has exactly one `<h1>` and one canonical; JSON-LD parses on every page; `FAQPage` appears only where the FAQ is visible (home, `/ai`, `/mcp`, `/runtime`, `/compare`).
- `sitemap.ts` and `public/sitemap.xml` list the same 13 URLs; all `_redirects` targets resolve; live content types are correct for `/llms.txt`, `/llms-full.txt`, `/SKILL.md`, `/robots.txt`, `/sitemap.xml`.
- All page routes resolve except `/downloads` and the `/docs#mcp` anchor.
- `/runtime` is the honesty benchmark: 39.4 ms, "not the fastest starter", 5.39 MiB, explicit "where it loses" rows. `/ai` and `/compare` contain genuine limitation sections. `llms-full.txt` is mostly consistent with those measurements.
- The install script has a real path-traversal defense and checksum *logic* (the file format is the bug); the npm installer verifies the tarball hash correctly.
- The hero canvas respects `prefers-reduced-motion`; there are no `<img>` elements (hence no alt-text defects); no third-party trackers or cookies were observed.
- Focus-visible styling is present; the colour system is coherent.

---

## 9. Recommended fix order

1. **Installer + checksums** (P0-1, P0-2): fix the Darwin archive handling; regenerate every `.sha256` as `<hash>  <filename>`; add a release-workflow check that runs `sha256sum -c` against each file; test `curl | sh` on macOS and Linux.
2. **Legal/security copy** (P0-3, P0-4): rewrite Terms §3 around QTSPL; replace the ML-KEM claim with AES-256-GCM + machine-local key; qualify the jail/allowlist/watchdog absolutes (P1-4).
3. **Embed coordinates** (P0-6): publish 0.3.0 artifacts or state 0.2.0; add the Maven repository URL; make visible code and `copyText` identical.
4. **Hero + API reference** (P0-7, P1-2): runnable agent example; fix `estimate`, `model`, `resolve`, `http.serve`, `kv`, `process.versions`, `r.tokens`, `similarity`, and the MCP `call` example.
5. **Republish binaries** (P0-8) so `logs`, `resume`, and `plugin test` exist in what visitors download.
6. **Cold-start + size** (P0-5, P1-1): pick one cold-start framing and purge the other; fix 5.2 MB; decide whether "≤ 5.5 MiB" refers to the release or `main`; green the size gate.
7. **MCP SSE + `/downloads` + `/docs#mcp`** (P1-3, P1-5, P1-6).
8. **CI and doc-drift claims** (P1-7, P1-8): green the gates or soften the copy; add or remove the claimed test.
9. **Machine-readable docs** (P1-9): regenerate `SKILL.md` and `llms.txt` from verified snippets.
10. **P2 polish**: contrast, mobile nav, modal/copy-button accessibility, 404 canonical, unused assets, headers, font/MicroSlats licensing, lint.

Deployment remains owner-gated: `wrangler pages deploy` is a production write.

---

## 10. Verification appendix

### 10.1 Commands used

```bash
# Shipped binary
tar xzf sofuu-landing/public/downloads/sofuu-darwin-arm64.tar.gz
stat -f "%z" sofuu-darwin-arm64/sofuu          # 5655808
./sofuu-darwin-arm64/sofuu --version           # sofuu 0.3.0-beta

# Current build size vs cap
cp target/release/sofuu /tmp/cur-stripped && strip -x /tmp/cur-stripped
stat -f "%z" /tmp/cur-stripped                 # 6032976; cap 5767168

# Installer (macOS path)
SOFUU_INSTALL_DIR=/tmp/sofuu-install-test sh sofuu-landing/public/install

# Checksum verification
shasum -a 256 -c sofuu-darwin-arm64.tar.gz.sha256   # exit 1
sha256sum -c sofuu-darwin-arm64.tar.gz.sha256       # warning, exit 0

# API existence/behaviour
./sofuu run existence.js      # 6 documented names undefined
./sofuu run behavior.js       # embedLocal Float32Array(768), embedBatch 2x(64)
./sofuu run provider.js       # complete -> {text, usage, raw}; no tokens

# CLI (note: zsh does not word-split unquoted variables)
bash -c './sofuu logs'
bash -c './sofuu plugin test'

# Live routes
curl -s -o /dev/null -w "%{http_code}" https://sofuu.xyz/downloads   # 404
curl -s -o /dev/null -w "%{http_code}" https://sofuu.xyz/docs         # 200

# CI
gh run view 41066226596 --json jobs --jq \
  '.jobs[] | {name, conclusion, steps: [.steps[] | {name, conclusion}]}'
```

### 10.2 API verification matrix (shipped 0.3.0-beta)

| Symbol | Result |
| :--- | :--- |
| `sofuu.ai.complete` | exists; returns `{ text, usage, raw }` |
| `sofuu.ai.stream` | exists; string + options form dispatches |
| `sofuu.ai.streamLocal` | exists |
| `sofuu.ai.embed` | exists; no-config call returns a local `Float32Array` |
| `sofuu.ai.embedBatch` | exists; array of `Float32Array(64)` |
| `sofuu.ai.embedImage` | exists |
| `sofuu.ai.transcribe` / `speak` | exist |
| `sofuu.ai.embedLocal` | `Float32Array(768)` |
| `sofuu.ai.embedLocalSemantic` / `V2` | 64-dim |
| `sofuu.ai.similarity` / `dot` / `l2` | exist; plain arrays accepted |
| `sofuu.ai.estimate` | **missing** (real: `estimateTokens`) |
| `sofuu.ai.model` / `resolve` | **missing** (real: `modelCaps` / `resolveCaps`) |
| `sofuu.http.createServer` / `fetch` | exist |
| `sofuu.http.serve` | **missing** (real serving: `sofuu.serve.start`) |
| `sofuu.fs.*` | exist, including `readFileBytes` |
| `sofuu.spawn` | exists |
| `sofuu.web.search` / `open` | exist |
| `sofuu.kv.open` | exists, but handle is `save/search/flush` (no `get/set`) |
| `sofuu.storage.open(...).kv` | real key-value facade |
| `sofuu.session.import` / `export` | exist |
| `sofuu.agent.define` / `run` / `runMany` / `cancel` / `renderTrace` | exist; `run(name, task, opts)` |
| `sofuu.mcp.connect` / `serve` | exist; `serve` has `tool/resource/prompt/start/stop` |
| `sofuu.rlm.query` | exists |
| `sofuu.ml.info` / `ml.freshness.score` | exist |
| `sofuu.sandbox.run` / `eval` / `create` | exist; presets `strict/standard/compute/worker` |
| `process.versions` | **undefined**; `process.version` = `0.3.0-beta` |
| `sofuu.tools.shell` / `fs` | **undefined**; registry is `sofuu.tools.TOOLS` |

### 10.3 Size / cold-start / version matrix

| Claim | Source | Measured | Verdict |
| :--- | :--- | :--- | :--- |
| 5.2 MB stripped | homepage:226,510; compare:237 | 5.39 MiB shipped | Wrong |
| 5.39 MiB | runtime:296 | 5,655,808 B = 5.39 MiB | Right |
| 5.4 MiB | docs:281; schema | 5.39 MiB | Rounding, acceptable |
| 5 MiB | ai:491 | 5.39 MiB | Understated |
| ≤ 5.5 MiB | multiple | shipped 5.39; main 5.75 | True only for the tarball |
| 5,655,792 B | runtime:101 | 5,655,808 B | 16 bytes off |
| `<4 ms` cold start | homepage ×5; docs; SKILL | 39.4 ms process-to-first-response | Wrong |
| 39.4 ms | runtime:105; compare:362 | 39.4 ms | Right |
| 12.4 MB RSS | homepage:512 | 14.9 MB after serving | Different metric; unsourced |
| 0.3.0 SwiftPM/AAR | embed:23,30; docs:896-897; SKILL:279,284 | published 0.2.0 | Wrong |
| `process.versions.node` 22.0.0 | docs:638; COMPAT.md:52 | `process.versions` undefined | Wrong |

### 10.4 Live route status (2026-10-30)

| Route | Status | Content type |
| :--- | :--- | :--- |
| `/`, `/docs`, `/ai`, `/runtime`, `/mcp`, `/compare`, `/embed`, `/embedded`, `/simd`, `/polling`, `/server`, `/privacy`, `/terms` | 200 | `text/html` |
| `/downloads` | **404** | `text/html` |
| `/llms.txt`, `/llms-full.txt` | 200 | `text/plain; charset=utf-8` |
| `/SKILL.md` | 200 | `text/markdown; charset=utf-8` |
| `/sitemap.xml` | 200 | `application/xml` |
| `/robots.txt` | 200 | `text/plain` |
| `/wiscove/appcast.xml` | 200 | `application/xml` |
| `/wiscove/Wiscove-0.1.0.zip` | **404** | `text/html` |

### 10.5 Limitations of this audit

- The entire repository `docs/` directory was not audited line by line; `docs/MCP.md` and `COMPAT.md` were used as ground truth for MCP and Node-compatibility claims.
- No Chrome/CrUX was available, so Lighthouse/Core Web Vitals were not measured. The contrast ratios are computed from the CSS tokens, not sampled from rendered pixels.
- The Linux binaries were not executed on a Linux host; their archive layout was inspected and checksums verified. The macOS binary was executed.
- iOS/Android SDKs were not built or run; their install claims were checked against `Package.swift`, `Sofuu.podspec`, release workflows, and Maven metadata.
- Prohibited codenames (RULES §2.4) cannot be exhaustively verified by search; no suspicious terms were found on the site.
- The Satoshi license could not be confirmed from Fontshare's public page during this audit; the finding is "license text is not committed; verify the terms".

---

## 11. File reference index

| Area | File |
| :--- | :--- |
| Homepage | `sofuu-landing/src/app/page.tsx` |
| Docs | `sofuu-landing/src/app/docs/page.tsx` |
| AI page | `sofuu-landing/src/app/ai/page.tsx` |
| Runtime page | `sofuu-landing/src/app/runtime/page.tsx` |
| MCP page | `sofuu-landing/src/app/mcp/page.tsx` |
| Compare page | `sofuu-landing/src/app/compare/page.tsx` |
| Embed page | `sofuu-landing/src/app/embed/page.tsx` |
| Privacy | `sofuu-landing/src/app/privacy/page.tsx` |
| Terms | `sofuu-landing/src/app/terms/page.tsx` |
| Layout / metadata | `sofuu-landing/src/app/layout.tsx` |
| Global CSS / tokens | `sofuu-landing/src/app/globals.css` |
| Components | `sofuu-landing/src/components/` |
| Install script | `sofuu-landing/public/install` |
| Downloads | `sofuu-landing/public/downloads/` |
| Machine docs | `sofuu-landing/public/SKILL.md`, `llms.txt`, `llms-full.txt` |
| Headers / redirects | `sofuu-landing/public/_headers`, `_redirects` |
| Rules | `RULES.md` |
| QTSQ license | `LICENSES/QTSQ-FORMAT.txt` |
| Brain key | `crates/sofuu-core/src/rt/brain_key.rs` |
| Policy defaults | `crates/sofuu-core/src/rt/policy.rs` |
| MCP docs | `docs/MCP.md` |
| Node compatibility | `COMPAT.md` |
| Type declarations | `sofuu.d.ts` |
| Release workflows | `.github/workflows/release.yml`, `android-maven-pages.yml`, `ci.yml` |
