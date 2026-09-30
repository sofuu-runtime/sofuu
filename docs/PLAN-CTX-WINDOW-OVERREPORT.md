# PLAN-CTX-WINDOW-OVERREPORT.md

**Context-window over-report — `z-ai/glm-5.3-free` @ tokenrouter**
Session date: 2026-09-14 (screenshot dated 2026-09-15 local)
Status: **FIXED 2026-09-19 — option (a) applied: cross-root plan-stem inheritance removed
(`lookup_any_root` is exact-id only); regression tests
`cross_root_is_exact_id_only_no_plan_stem_inheritance` (discovered store) +
`free_tier_variant_does_not_inherit_paid_stem_window` (policy ladder) pass.
Gates: cargo lib 379/0, `make test` 41/0/1. Re-probe: `-free` → window 32768
default (was 1310720 fiction); exact stem still 1310720 cross-root. Side issue:
doubled orca endpoint repaired in `~/.sofuu/config.json`. Original §5 options kept below for record.**

---

## 0. Original request

> "see this @image-viewer , the context window caculation has something wrong , its showing way too much then it support , debug this and check why its happen ."

Screenshot: `/Users/priyanshuboruah/Documents/Screenshot 2026-09-15 at 11.35.49 AM.png` — the ctx meter showing far more context than the model supports.

The image could NOT be viewed this session: image-viewer agent failed (`credit insufficient balance: balance=0 required=760`); direct Read returned "[Media omitted … does not support image input]". Diagnosis proceeded code-first from the known live config — no visual was needed once the model/provider were known.

## 1. TL;DR root cause

The ctx meter's denominator for `z-ai/glm-5.3-free` (provider **tokenrouter**) resolves to **1,310,720 (1.25M)** via the cross-root **plan-stem** fallback:

- No endpoint anywhere publishes the `-free` id, so the cross-root resolver strips `-free` (plan_stem), matches the **PAID** `z-ai/glm-5.3` on openrouter (1,310,720) and orcarouter (1,000,000), and deliberately takes the **LARGEST** (`max_by_key(ctx_window)`).
- The serving endpoint (tokenrouter) publishes **no caps at all** for these models (verified live: 137 models listed, glm-5.3 ids present, `context_length: None`).
- The static registry doesn't know the id (`known:false, ctxWindow:0`), the user's config has `ctx_window: 0` (no number), and there are no learned-400s for this (root, model).

So the weakest, most generous inference in the whole evidence ladder wins, and the meter claims ~1.25M for a free-tier variant that serves far less (its real window is unpublished — unknown to us).

**User-visible harm:** `ctxBudget = floor(window × 0.85) ≈ 1,114,112` — auto-compact, fitGuard, and attachment budgets all key off a window ~10× the real one. Requests will 400 (or the provider truncates) long before the meter looks anywhere near full.

## 2. Evidence chain (all verified this session)

1. **Live config** (`~/.sofuu/config.json`): active provider `tokenrouter`, endpoint `https://api.tokenrouter.com/v1/chat/completions`, model `z-ai/glm-5.3-free`, `ctx_window: 0`, `max_output: 384000`.
2. **Runtime probe** (`sofuu run /tmp/caps_probe.js` on the shipped ./sofuu):
   - `sofuu.ai.resolveCaps('z-ai/glm-5.3-free','',0,384000)` →
     `{"window":1310720,"maxOutput":384000,"known":true,"source":"cross-root","winSource":"cross-root","maxSource":"config","thinking":"unknown"}`
   - no-args resolve → `maxOutput: 943718` also cross-root (same flaw on the output side, currently masked by the config 384000).
   - `sofuu.ai.modelCaps('z-ai/glm-5.3-free')` → `known:false, ctxWindow:0` — static registry doesn't know it.
3. **Discovered store** (`~/.sofuu/ml/model_caps.json`, shape `{entries:{<root>:{<model>:[ctx,max,thinking,at]}}}`), roots present: `api.example.com/v1`, `api.orcarouter.ai/v1`, `caps-gw.example.com/v1`, `lightning.ai/api/v1`, `openrouter.ai/api/v1`:
   - `openrouter.ai/api/v1 :: z-ai/glm-5.3 => [1310720, 943718, 1, 1789025598]`
   - `api.orcarouter.ai/v1 :: z-ai/glm-5.3 => [1000000, 128000, -1, …]`
   - **No tokenrouter root at all; no `-free` entries anywhere.**
4. **Live tokenrouter listing probe** (GET `https://api.tokenrouter.com/v1/models`, key read from config into a shell var — never printed): **137 models**; `z-ai/glm-5.3-free`, `z-ai/glm-5.3`, `z-ai/glm-5.3-flash` all present but publish **no `context_length` / `max_completion_tokens`** (all `ctx: None`). So even a fresh harvest would record nothing usable for these ids — the store gap is not just staleness.
5. **Ladder mechanics confirmed by source** (§3): config (0) → registry (unknown) → same-root discovered (tokenrouter absent) → learned-400s (none) → cross-root Pass 1 exact (miss) → Pass 2 plan-stem (**1,310,720** wins as max).

## 3. Code map (PRE-FIX snapshot — `plan_stem()` and Pass 2 no longer exist as of the 2026-09-19 fix; `lookup_any_root` is exact-id only)

| Location | What |
|---|---|
| `crates/sofuu-core/src/rt/model_caps_discovered.rs:264` | `plan_stem()` — strips `:free, :batch, :nano, :preview, -free` suffixes (lowercased). |
| `crates/sofuu-core/src/rt/model_caps_discovered.rs:275` | `lookup_any_root()` — Pass 1 exact id across roots, `max_by_key(ctx_window)` (:282–290); Pass 2 plan-stem match, also `max_by_key` (:291–305). The comment claims "the larger is likelier the model's real window than a per-plan cap" — **this is the inverted assumption for free tiers**. |
| `crates/sofuu-core/src/rt/model_caps_discovered.rs:198` | `ingest_listing()` — rows without an id are skipped; per-row parse in `ingest_one` (not yet read — only needed if pursuing the harvest angle). |
| `crates/sofuu-core/src/rt/model_caps_discovered.rs:240` | `lookup()` — same-root, **exact id only** (no stem matching same-root). |
| `crates/sofuu-core/src/ml/alloc/policy.rs:207` | `resolve_with()` — per-side strongest evidence; config honored but clamped by the strongest real bound (cross-root stem evidence counts as a bound); sides resolve independently. `UNKNOWN_WINDOW = 32_768`, `UNKNOWN_MAX_OUTPUT = 4_096`. |
| `crates/sofuu-core/src/rt/ai.rs:3972` | `js_ai_resolve_caps` bridge — `sofuu.ai.resolveCaps(model, baseUrl, cfgWindow, cfgMaxOutput)` → JSON string. |
| `src/js/chat.js` ~190–249 | shipped-driver `resolveCaps`/`ctxWindow` — native bridge first, legacy JS fallback; feeds ring denominator, `ctxBudget()`, compaction trigger, fitGuard, attachment budgets. Both chat drivers (chat.rs DRIVER and shipped chat.js) hit the same native resolve. |
| `crates/sofuu-core/src/rt/ai.rs:4771` | desktop listing-harvest ingest path (listings fetched by the desktop refresh get ingested "for free" per api root). |

## 4. Why this is a design flaw, not just a data gap

- A plan suffix (`-free`) exists **precisely because** the variant is served under different limits. The stem model's published window says nothing about the variant; Pass 2's "larger is likelier" reasoning inverts for free tiers — the per-plan cap **is** the real constraint.
- `max_by_key` across roots means the most generous publisher's number (openrouter's paid 1.25M) defines the denominator for a free tier on a different provider.
- Cross-root stem evidence also acts as a **clamping bound** in `resolve_with` — e.g. a user setting `ctx_window: 2000000` would be "clamped" to the inherited 1.31M fiction rather than the endpoint's truth.
- The output side has the same flaw (`maxOutput 943718` cross-root when config is 0/unset).
- The ladder's designed answer for "unknown" already exists: `UNKNOWN_WINDOW = 32_768` + learn the real window from the first context 400 (learned-400s override everything). Pass 2 short-circuits that honest path.

## 5. Fix options (DECIDED 2026-09-19: (a) applied — kept below for record)

- **(a) RECOMMENDED — drop cross-root plan-stem inheritance.** Pass 2 only ever fires for plan-suffixed ids (it returns early when `stem == id`), so this means removing/gating Pass 2 in `lookup_any_root`. The variant then falls to `UNKNOWN_WINDOW = 32_768` (honest-conservative; small meter until the first real 400 teaches the true window). Exact-id cross-root (Pass 1) stays — same id, same weights, sound. Acceptable side effect: `:batch`-style variants that legitimately share the paid window also fall to default until traffic teaches it.
- (b) Take `min` instead of `max` in Pass 2 — **does not fix this case**: min across roots = 1,000,000, still ~8× inflated for a free tier.
- (c) Sanity clamp on stem inheritance (e.g. ≤ 262,144) — arbitrary magic number, rejected in spirit.
- (d) Let user `ctx_window` act as a hard ceiling when evidence is stem-only — doesn't help here (user has 0) and re-opens the P0 ring-bug surface.

**Whichever lands:** add a policy test (a `-free` id with two cross-root stem entries of differing windows must resolve to the 32,768 default, not the max), `export SOFUU_QTSQ_DIR=$HOME/projects/black-hole-disk`, `make`, re-run `make test` + cargo gates (lib 378 / capi 22 / ffi 11), re-probe resolveCaps, and re-copy `./sofuu` → `/usr/local/bin/sofuu` (manual install, `make` doesn't refresh it). Note: policy tests serialize on `ml::TEST_LOCK` and skip `persist()` so fixtures never clobber the user's ~590-model store.

## 6. What was done, step by step (session log)

1. User reported the over-reporting meter with a screenshot. image-viewer agent failed (credits); direct Read → media-omitted (session model can't process images). Fell back to code-path debugging with the known live config.
2. Read `src/js/chat.js` resolveCaps/ctxWindow region (~190–249): native ladder via `sofuu.ai.resolveCaps`.
3. Read `ai.rs:3972` bridge → `policy::resolve`.
4. Read `policy.rs` `resolve`/`resolve_with`: evidence ladder, per-side strongest evidence, config clamp semantics, UNKNOWN defaults.
5. Dumped live `~/.sofuu/ml/model_caps.json`: found orcarouter 1M and openrouter 1.31M entries for the stem id; no `-free`, no tokenrouter root.
6. Wrote `/tmp/caps_probe.js` (via Write tool — Mimosa blocks Bash `.js` writes) and ran `sofuu run /tmp/caps_probe.js`: window 1,310,720 `winSource:"cross-root"`; `modelCaps` known:false.
7. Read `lookup_any_root` Pass 1/Pass 2 and `plan_stem`: `-free` stripped; both passes `max_by_key(ctx_window)`.
8. Read live provider endpoints from `~/.sofuu/config.json` (python3): tokenrouter confirmed active; noted orca-ai's doubled endpoint (§7).
9. Live-probed tokenrouter `GET /models` with the key held in a shell var (never printed): 137 models, glm-5.3 ids present, `ctx: None` for all.
10. Read `ingest_listing` (:198) and `lookup` (:240, same-root exact only); located ingest call sites (ai.rs:4771).
11. Wrote this doc (owner request: "write everything you did till now … to continue later").

**Probe script** (`/tmp/caps_probe.js`, recreate via Write tool if /tmp was cleaned):

```js
var r = JSON.parse(sofuu.ai.resolveCaps('z-ai/glm-5.3-free', '', 0, 384000));
console.log('resolveCaps:', JSON.stringify(r));
console.log('modelCaps:', String(sofuu.ai.modelCaps && sofu​u.ai.modelCaps('z-ai/glm-5.3-free')));
```

(Check exact modelCaps return shape against the bridge before relying on the second line.)

**Key-safe listing probe:**

```bash
KEY=$(python3 -c "import json; print(next(p['api_key'] for p in json.load(open('$HOME/.sofuu/config.json'))['providers'] if p['name']=='tokenrouter'))")
curl -sS -m 12 -H "Authorization: Bearer ${KEY}" https://api.tokenrouter.com/v1/models | python3 -c "…filter id/ctx per model…"
```

## 7. Side observations (separate issues, NOT this bug)

- **orca-ai provider endpoint is doubled** in config: `https://api.orcarouter.ai/v1/chat/completions/chat/completions` — Pass-27-class doubled-suffix artifact; worth flagging/normalizing someday.
- **`max_output: 384000`** in user config is the old flat global (Pass 31); can 400 smaller models. Not part of this bug (window side is the complaint).
- The screenshot itself was never seen — diagnosis is from code + live probes; the 1.31M denominator is certain for this model/config regardless of what the meter rendered.
- Prior open items unrelated: bench S6 freshness (ML data problem, intentionally unfixed), round-10 embedder reopen path (needs explicit go), the ~130+-file uncommitted commit (user's call).

## 8. Environment / how to resume

- `export SOFUU_QTSQ_DIR=$HOME/projects/black-hole-disk` before every make/cargo run.
- `make` (NOT cargo build alone) refreshes `./sofuu` (src/js/*.js is include_str!-baked). After any rebuild: `cp ./sofuu /usr/local/bin/sofuu`.
- Alarm-wrap long runs: `perl -e 'alarm N; exec @ARGV' -- …`.
- Gates at last full run: `make test` 41/0/1; cargo lib 378/0, capi 22/0, ffi 11/0.
- Mimosa hook: blocks Bash source writes / cp copies / dynamic eval — use Write/Edit for ALL source changes; known false positives: `RegExp.exec(` ("command injection"), string concat inside call parens ("XSS"); Bash writes to files named like configs (.json/.js probes) are blocked even in /tmp.
- Keys are sensitive — never print them; read into shell vars as in §6.
- Tree stays UNCOMMITTED; commits to main are the owner's call.
- TUI ASCII-only.
