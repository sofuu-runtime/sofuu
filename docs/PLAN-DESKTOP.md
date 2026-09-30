# PLAN-DESKTOP — Sofuu Desktop (Tauri + TypeScript over sofuu-core)

> **Status (2026-08-24 night):** Phase 1 workstreams A, C1, D, E, F done & verified ·
> **UI design locked** on 4 reference screenshots (warm light cream, generous whitespace) ·
> `Sofuu.app` builds (8.3 MB, aarch64) · B and C2 deferred (see §9 deviations).
> Companion docs: `PLAN-HEADLESS.md` (embedding contract), `PLAN-AGENTS.md` (the agent
> loop this app renders), `AUDIT-2026-08-22.md` (engine constraints P0-7).

---

## 1. Goal

Replace the Electron `sofuu-desktop/` in place with a **Tauri v2 macOS app**: a Rust
backend that links `sofuu-core` **in-process**, and a **TypeScript (React) frontend
strictly for the UI**. The desktop app is a first-class Mac citizen with the full CLI
feature set — one runtime, one turn implementation, two renderers (TUI + desktop).

Destination: full parity with the CLI chat. Phased delivery; v1 scope in §5.

## 2. Decisions (settled with the user, 2026-08-24)

1. **Stack** — Tauri v2 shell. TypeScript strictly for the frontend UI; Rust for the
   backend. React 19 + Vite + `react-markdown`/`remark-gfm` (carried over from the old
   app's dependencies).
2. **Integration** — link `crates/sofuu-core` in-process via path dependency. **No**
   sidecar process, **no** daemon wire protocol. Never additionally link
   `sofuu-capi`/libsofuu (duplicate `no_mangle` symbols are fatal under fat LTO —
   depend on `sofuu-core` exactly once).
3. **Delivery** — phased (§5). v1 = streaming chat over the full agent loop (coding
   tools active), markdown rendering, tool-call timeline with approval prompts,
   provider/model/effort settings, session list + resume + new.
4. **Replacement** — the Electron `sofuu-desktop/` folder becomes the Tauri app in
   place. Its two-pane layout and settings design are reused; its
   subprocess-per-message architecture is discarded.
5. **Size** — the 5 MB cap is a **CLI-only** promise (`make size-check` measures
   `./sofuu`). No size gate on the desktop app; a ~30–60 MB `.app` bundle is expected
   and fine.
6. **Platform** — **macOS only** for now, built as a first-class Mac citizen (§4).
   Linux/Windows are dropped from Phases 1–2 entirely.
7. **UI design** — **locked 2026-08-24 late** on 4 reference screenshots
   (`~/Documents/Screenshot 2026-08-24 at 7.3*.png` — `7.39.52`, `7.39.55`, `7.39.58`,
   `7.40.24 PM.png`; contain `\u202f` NARROW NO-BREAK SPACE before AM/PM — copy to
   plain names first, e.g. `/tmp/sofuu_desktop_refs/ref{1..4}.png`). Target: replicate
   the reference design **exactly** (pixel-faithful two-pane + centered composer +
   modal); user will adjust later. Theme: **light only** (warm cream/ivory). Emphasis:
   beautiful, perfectly placed components, generous whitespace, clean card surfaces.
   The placeholder direction (dark-first, TUI magenta accent) is dead — see §4a for
   the settled spec.

## 3. Architecture

```
┌ sofuu-desktop ─────────────────────────────────────────────┐
│  React/TS UI (WKWebView)                                   │
│   ↕ invoke (commands)        ↕ emit (event stream)         │
│  Tauri Rust backend (macOS main thread)                    │
│   • commands.rs — send_turn, cancel, approve, sessions,    │
│     config, project dir, keychain                          │
│   • host.rs — event sink → AppHandle::emit, approval       │
│     registry, poke sender                                  │
│   • engine.rs — dedicated worker thread owning ONE         │
│     SofuuRuntime for the app lifetime; mpsc command loop   │
│        ↓ eval (blocking)          ↑ bridge natives         │
│  sofuu-core: shipped chat.js turn engine + agent.js loop   │
│  + lib chat/session modules + uv_async "poke" (Rust→JS)    │
└────────────────────────────────────────────────────────────┘
```

**Engine constraints that dictate this shape (verified in code):**

- `eval_string` blocks until all JS drains (`rt/engine.rs:725`, `rt/loop.rs:95`) →
  the engine lives on one dedicated worker thread with a command channel; a turn is
  one blocking eval; events stream out through bridge natives *during* the eval.
- Process-global libuv loop (`rt/loop.rs:29`) + thread-local registries
  (`rt/timer.rs:43`, `rt/http_client.rs:36`, …) → init and all engine use happen on
  one fixed thread; the UI stays on the macOS main thread.
- **P0-7** (`AUDIT-2026-08-22.md:110`): exactly **one** SofuuRuntime for the app's
  lifetime — never a second engine before dropping the first; dropped only at quit,
  on its own thread.
- `embed_config::configure(embedded, …)` already provides polite-guest mode
  (catchable `process.exit`, no signal handlers, console → log callback).
- `chat.rs`/`session.rs` are bin-only today (`main.rs:17`) → they must move into the
  lib for the desktop to reuse them.
- Build inputs already present locally: prebuilt `deps/libuv/build/libuv.a`, QTSQ at
  `~/projects/black-hole-disk`. Keep dynamic system curl (avoid colliding with
  wry/reqwest TLS stacks).

## 4a. UI spec — settled light theme (from 2026-08-24 refs)

> Refs: `ref1=7.40.24` Home empty, `ref3=7.39.52` Chat empty, `ref2=7.39.55` Settings/General, `ref4=7.39.58` Settings/Appearance. Copied to `/tmp/sofuu_desktop_refs/ref*.png` for tooling.

**Theme:** **light only** for v1. Warm cream/ivory, not pure white. No dark mode in v1 (Appearance tab's Dark/System options render but are disabled/hidden until v2). Same palette used by the Tauri window chrome (`hiddenInset` sidebar vibrancy tints to this palette, not magenta).

**Palette (hex, extracted):**

| Token | Value | Usage |
|---|---|---|
| `--bg-app` | `#FFFBF0` `#FEF9EC` | main canvas |
| `--bg-sidebar` | `#F2EEDD` `#EFE8D6` | left pane + active tab fill |
| `--bg-card` | `#F7F0DE` | composer card, active list item |
| `--bg-card-hover` | `#EFE8D6` | hover / selected nav |
| `--border` | `#E8E0C9` | card + input borders |
| `--border-strong` | `#E6DDC3` | pill button borders |
| `--text` | `#1A1A1A` | headings, body |
| `--text-muted` | `#6B6760` | descriptions, empty-state subtitle |
| `--text-faint` | `#A8A29A` `#8A857E` | placeholders, hints |
| `--accent` | `#7C2D8A` | toggles ON, focus rings, primary action tint |
| `--accent-hover` | `#6A2575` | accent hover |
| Outer frame | `#0A0A0A` | window screenshot border (not app bg) |

**Typography:** `Geist Variable` for UI (fallback `-apple-system, BlinkMacSystemFont, Inter, sans-serif`); `Geist Mono Variable` for code/terminal (fallback `ui-monospace, SFMono-Regular, Menlo, monospace`). Sizes: sidebar `13px` semibold headers / `14px` items, headline `20px/600` (`What should we work on?` / `Nothing here yet`), placeholder `14px`, description `13px`, tab `14px`, composer hint `12px`. Chat message body `17px` adjustable via Appearance → Chat font size (`−`/`+` stepper).

**Radii / shadows / spacing:**
- `--radius-card: 16px` (composer, modal), `--radius-pill: 999px` (tabs, split button), `--radius-field: 8px` (dropdowns).
- `--shadow-card: 0 1px 3px rgba(0,0,0,.06)`; `--shadow-modal: 0 20px 60px rgba(0,0,0,.15)` + backdrop `rgba(0,0,0,.18)` blur.
- Whitespace: sidebar `12px` row gap / `16px` pane padding; main outer `48px` horizontal; composer max-w `~720px` centered, inner `24px`; modal `~860×~520`, left nav `~180px`, right pane `24px`; section separators `1px #F0E8D0`.

**Layout anatomy (pixel-faithful):**

- **Window:** macOS traffic lights top-left, `hiddenInset` titlebar floats over sidebar; `#0A0A0A` outer screenshot frame not part of app; inner content `~12px` rounded window (`12–14px`).
- **Sidebar (220px):** header `Workspaces` `13px/600` + 3 icon buttons (filter `≡`, folder, `+`) `18px` square, muted `#8A857E` hover `#1A1A1A`; list items `8px` radius, `8px 12px` padding, active `#EEE6D1`; status icons (purple ✓ Done, amber ◐ In review, green ◑ In progress, dashed ○ Backlog, red ✕ Canceled, gray 𐃁 Archived); footer gear `⚙` + `[]` `16px` muted.
- **Main — Home (ref1):** centered headline `20px/600` with `32–40px` gap to composer; composer card `16px` radius `1px #E8E0C9` `24px` padding, stacked: top textarea (placeholder `Describe what you want to build` `#A8A29A`), bottom bar left `✦ Fable 5 1M ▼` `High ▼` + clipboard/layers icons `16px` muted, right split pill `↑ New Workspace ▼` (left `↑`+label `white` + `1px #E6DDC3`, right `▼` divider); below card `○ Just chat ▼` `13px #6B6760` left-aligned to card.
- **Main — Chat empty (ref3):** top tab bar `44px` `#FFFBF0` with `○ New chat` `13px` breadcrumb + active tab pill `✦ Untitled` `#F7F0DE` `8px` radius `8px 14px` + `+` `◷` `18px` right; center empty `Nothing here yet` `15px #6B6760` + link `New session in 2026-06-14/new-chat` `#6B6760` underline; bottom composer same card but placeholder `Ask to make changes, @mention files, run /commands` + `⌘L to focus` `12px #A8A29A` right, send `↑` `36px` square `8px` radius `white` `1px #E6DDC3`.
- **Modal (ref2/ref4):** centered `860×520` `16px` radius, header `16px/600` + `✕` `20px` top-right; left nav `General/Appearance/Models/Providers/Shortcuts/Accounts/Contexts/Experimental` `14px`, active `8px` radius `#EEE6D1`; right pane rows `16px 0` + `1px #F0E8D0` divider; toggle `28×18px` pill `#7C2D8A` on / `#DDD8C8` off; segmented theme control (Light/Dark/System with `☀ ☾ ☐` icons, active `#F7F0DE` pill); dropdown field `white` `1px #E6DDC3` `8px` `12px 36px 12px 12px`; stepper `−`/`+` `28px` square `8px` radius `white` border; mono fields `Geist Mono 12px` `#5A5752` on `#FFFBFA`.

**Mapping to Sofuu v1:**
- `Workspaces` → sessions (mesh registry) + project dir; `Chats 2` count pill → session count.
- `Fable 5 1M` → model picker (`sofuu.ai.modelCaps`); `High` → effort; `Just chat` → mode.
- `New Workspace` → `New chat` (`new_session`); Appearance font controls → chat body size + UI/code/terminal font families.
- Settings tabs map to `get_config`/`update_config` + macOS Keychain (Providers) + `listModels`/`listProviders`.

**Guardrails:** generous whitespace is mandatory (never compress composer/main padding); card borders stay `1px` hairline; toggles/accent stay `#7C2D8A` (not magenta); light theme only — do not ship dark styles in v1.

## 4. macOS-native feature set (v1 target)

- `hiddenInset` title bar — traffic lights float over the sidebar; optional
  `NSVisualEffectView` vibrancy behind the sidebar; proper fullscreen; retina-crisp.
- Real `NSMenu` menu bar (File/Edit/Window/Help + New Chat ⌘N, Stop ⌘., Settings ⌘,)
  wired to the same Tauri commands the UI buttons call; Mac keyboard conventions
  throughout the app.
- **API keys in the macOS Keychain** — the desktop reads provider keys from Keychain
  first (via `embed_config::set_api_key`); CLI env/config behavior unchanged.
- Native notifications (turn finished / approval needed) through
  UNUserNotificationCenter — long agent runs can finish while the user is elsewhere.
- Drag & drop of files/folders onto the chat → `@file` mentions.
- Dock badge/bounce for pending approvals and finished turns.
- Apple Silicon arm64-native build; code signing + notarization + DMG at release time
  (user's Apple Developer credentials required for that step).
- Reachable later in Tauri: tray icon with a quick-ask mini window (Phase 3
  candidate). **Not** reachable without a Swift companion layer: SwiftUI controls,
  widgets, Shortcuts/App Intents, Touch ID — explicitly out of scope unless asked.

## 5. Phases

### Phase 1 (implementation plan below)

Streaming chat with the full agent loop (web + code tools active), markdown rendering,
tool-call timeline with approval prompts, provider/model/effort settings, session
list + resume + new chat, all macOS-native integration from §4.

### Phase 2 (later)

Brain/memory UI (`/why` recall panel, `/remember`), cost dashboard (`/cost`), `@file`
picker UI, compact/clear controls, model/effort pickers in the UI, session
rename/delete.

### Phase 3 (later)

Agents UI (`@agent` mentions, delegate tree), session-mesh presence (peers, tasks,
notes, notify), second-model verify, ghost completion, file watching, hooks
indicators, tray quick-ask window, deep links (`sofuu://`), auto-updater, notarized
release builds, Linux/Windows.

## 6. Phase 1 workstreams

### A. Host "poke" primitive (sofuu-ffi + rt)

The missing piece for cancel + approvals: waking the busy engine from another thread.

- `sofuu-ffi/src/uv.rs`: add `uv_async_init/start/send` externs + handle-size shim
  (same pattern as the existing timer/check bindings).
- New `rt/host_poke.rs` in sofuu-core: one **unref'd** `uv_async_t` on the shared
  loop (unref'd ⇒ doesn't keep the loop alive; `uv_async_send` still wakes a blocked
  `uv_run`). Thread-safe API: `host_poke_send(json)` pushes to a `Mutex<VecDeque>` +
  sends; the async callback (loop thread) drains the queue and calls a registered JS
  global `__host_poke(json)`.
- JS state stays in JS: the driver handles `{type:"cancel"}` →
  `sofuu.agent.cancel('chat')`, `{type:"approval", id, allow}` → resolves the stored
  promise. No Rust-side JS-value registry.

### B. Lib-ify chat/session (sofuu-core)

Split `chat.rs` (4,586 lines) into a lib module tree + a thin bin TUI shell:

- **To lib** `crates/sofuu-core/src/chat/`:
  - `config.rs` — ChatConfig/ProviderEntry/McpServerConfig, load/save (shared
    `~/.sofuu/config.json`), defaults, clamps, effective_ctx_window.
  - `slash.rs` — ALL_COMMANDS/COMMAND_INFO; `handle_slash` refactored to return
    `{op, message}` instead of printing, so both hosts render it.
  - `cost.rs` — SessionCost/pricing/budget check.
  - `recall.rs`, `watch.rs` — recall-hit storage; fs watcher bridge.
  - `session.rs` becomes `pub mod session`; the `past_turns` 10-turn cap becomes a
    parameter (desktop resume wants more).
  - `host.rs` — registers the pure-logic `__chat_*` natives under their existing
    names (log, report_usage, cost_breakdown, budget_check, sessions, resume_turns,
    set/get_recall, watch_*, mcpservers, no_think, past_turns) against a pluggable
    event sink, so chat.js runs unchanged under either host.
- **Stays in bin** (renamed `chat_tui.rs`): welcome panel, TTY bridge functions
  (`__readline`/`__tui_*`/selectors/wizard prompts), `run_chat`, the DRIVER — now a
  renderer over workstream C.

### C. Extract the turn engine → shipped `src/js/chat.js`

The heart of the plan; prevents permanent drift between TUI and desktop (same "one
loop for everything" philosophy as agent.js). Move out of the DRIVER string, as-is
where possible, threading shared state through a module state object: hooks loader,
`expandMentions` (@file), `chatToolDefs` (web + code + MCP), MCP connect/call,
`ctxBudget`/`attachTokenBudget`/`historyTokens`/`summarizeHistory`/`trimHistory`
(auto-compaction), the `turn()` body (budget preflight, def build, onStep semantics
with ANSI stripped, usage/cost reporting, session logging, history + manifest
storage, GC, no-think retry), `compact()`.

- API: `sofuu.chat.{init, submit(text,{onEvent}), cancel, compact, clear, resume,
  resolveApproval, state}`.
- **Permission gate** (opt-in via init flag; the desktop sets it): chat.js wraps each
  tool's `execute` — read-only tools (read_file/grep/glob/list_dir) auto-pass;
  write_file/edit_file/bash emit `approval_request` and await a host-resolved
  promise; denial returns "denied by user" as the tool result so the model can react.
  The desktop def sets a long `toolTimeoutMs` so human waits survive agent.js's 30s
  default wrapper.
- The TUI DRIVER becomes a renderer over `sofuu.chat.*` — CLI behavior must stay
  byte-identical (gated by §7).

### D. Tauri backend (`sofuu-desktop/src-tauri/`)

- `engine.rs` — spawned in `tauri::setup`: `embed_config::configure(true, …)` + log
  callback, `SofuuRuntime::init()`, register host natives + `__desktop_event` sink,
  eval `sofuu.chat.init(…)`, then loop on an mpsc command channel. A turn = one
  blocking eval of `sofuu.chat.submit(...)`; config/session reads are served from
  the Rust mirror without engine round-trips.
- `host.rs` — sink → `AppHandle::emit("sofuu://event", …)`; approval registry
  (id → pending); cancel/approve → `host_poke_send`.
- `commands.rs` — `#[tauri::command]`: `send_turn`, `cancel_turn`,
  `resolve_approval`, `list_sessions`, `session_turns`, `new_session`, `get_config`,
  `update_config`, `open_project_dir`, keychain get/set.
- **Project directory** — a Finder-launched app has no cwd, so the backend chdirs to
  a user-picked project dir (persisted in config): it is the tool jail root and the
  session-mesh root.
- Added as a workspace member (path dep on sofuu-core). Inherits the workspace
  release profile — `panic="abort"` tradeoff accepted (a command panic crashes the
  app rather than erroring); dev profile used for daily work.

### E. Frontend (`sofuu-desktop/src/`, TypeScript strict)

Vite + React 19 + `@tauri-apps/api`. Components: `Sidebar` (§4a sidebar spec, 220px, active pill `#EEE6D1`), `TopBar` (breadcrumb + `✦ Untitled` pill tab + `+`/`◷` — ref3), `HomeEmpty` (`What should we work on?` + centered composer — ref1) / `ChatEmpty` (`Nothing here yet` — ref3) / `ChatPane` (message list, streaming cursor, markdown + collapsible thinking blocks), `ToolTimeline` (cards from `tool`/`tool_result`/`delegate`/`recall`/`plan` events), `ApprovalDialog` (tool, args, allow / deny / always-allow-this-session), `Composer` (ref1/ref3 card: `Describe…` / `Ask to make changes…`, `✦ model ▼` `effort ▼` + icons left, split `↑ New Workspace/New chat` pill right, `○ Just chat ▼` below, `⌘L to focus` hint + `↑` send — §4a), `SettingsModal` (two-pane modal `860×520`, left nav + right pane with toggles/dropdowns/segmented/mono fields — ref2/ref4, light only). Event protocol types in `lib/events.ts` mirror the Rust serde structs exactly.

**Visual design: LOCKED (light, §4a)** — pixel-faithful to the 4 refs: warm cream `#FFFBF0`/`#F2EEDD`/`#F7F0DE`, borders `#E8E0C9`/`#E6DDC3`, text `#1A1A1A`/`#6B6760`/`#A8A29A`, accent `#7C2D8A`, radii `16px` card/`999px` pill/`8px` field, `Geist Variable`/`Geist Mono`, generous whitespace (main `48px` outer, card `24px` inner, centered `~720px` composer). Dark/magenta placeholders removed.

### F. Replacement + build wiring

- Delete the Electron files (`main.cjs`, `preload.cjs`, `sofuu-bridge.js`,
  `src/*.jsx`, old configs) once the Tauri app runs; the folder becomes the Tauri app.
- `Makefile`: `desktop` (tauri build → .app/.dmg) and `desktop-dev` targets.
- README + TASKS.md updated; desktop CI job and packaging later (Phase 3).

## 7. Verification gates

1. **After B+C, before any desktop work** (the CLI must be behavior-identical):
   `cargo test --release` (194 green) · `./sofuu run examples/agent_test.js` (80/80)
   · `tests/chat_compact_e2e.sh` (pty e2e) · `make test` (20/20) · `make size-check`.
2. New Rust unit tests: refactored `handle_slash` results, poke queue drain, host
   event serialization.
3. Desktop: host-core integration test with a scripted mock provider
   (`turn_started → tool → approval gate blocks until resolved → tool_result →
   done`) + a manual run against a real provider.

## 8. Risks

- **chat.js extraction (C)** touches the most battle-tested code in the repo —
  mitigated by moving functions as-is and gating on the full existing test battery
  before desktop work starts.
- **panic="abort"** in the workspace profile: command panics crash the app; accepted
  for v1 (one workspace/lockfile, engine keeps the CLI's size-tuned profile).
- **QTSQ linkage** pulls 7 macOS frameworks into the app (expected; same as the CLI).
  If wry hits link-order issues, fallback: exclude `src-tauri` from the workspace
  (own lockfile).
- **UI pixel-fidelity** to the 4 settled refs (§4a) — screenshots already re-saved to `/tmp/sofuu_desktop_refs/ref*.png` (strip `\u202f`); remaining fidelity risk is font loading (`Geist Variable/Mono`) and Tauri `hiddenInset` tint alignment.

## 9. Implementation status & deviations (2026-08-24)

What shipped, and where reality diverged from §6:

- **A — done, as planned.** `uv_async` bindings + `rt/host_poke.rs` (unref'd async,
  `Mutex<VecDeque>` queue, drains into JS `__host_poke`).
- **B — DEFERRED.** Splitting chat.rs into a lib module tree is postponed until the
  ML-advisor session settles (it touches the same files). The desktop does not need
  it: chat.js (C1) runs on the existing natives, and the session mesh got a small
  sync seam instead (below). The TUI's DRIVER is untouched.
- **C — split into C1 (done) / C2 (deferred).**
  - **C1 done:** `src/js/chat.js` ships the turn engine with
    `sofuu.chat.{init, submit, cancel, compact, clear, resume, resolveApproval,
    state, sessionTurns, newSession, setProject}` (three more than §6 listed — the
    desktop commands need them). Permission gate as specified: read-only tools
    auto-pass, everything else emits `approval_request` with a `toolTimeoutMs`
    ceiling; denial returns "denied by user" as the tool result.
  - **Deviation — hooks.js middleware (F9) is NOT in chat.js.** The eval-based
    loader pattern was rejected by the security scanner for shipped JS; hooks stay
    TUI-only for now. The shared loader seam lands with workstream B.
  - **C2 deferred:** the TUI DRIVER does not yet render over `sofuu.chat.*` — the
    CLI keeps its embedded copy. Both engines are ports of the same code path; the
    byte-identical gate (§7.1) was run on the CLI as-is.
  - **Session mesh seam:** instead of lib-ifying `session.rs`, a small sync module
    `rt/session_js.rs` exposes `__qtsq_session_save/load`, `__session_write_file`
    (mkdirs parents, tmp+rename), `__session_read_file`, `__session_project_root`
    (derives the mesh root exactly like `session.rs::project_root`). chat.js
    implements the registry + SessionData format in JS, byte-compatible with the
    Rust mesh — verified by having the Rust CLI decrypt a JS-written `.qtsq`
    (`sofuu session show` on a fresh project dir). Ordering matters: the registry
    write (which mkdirs `.sofuu/sessions/`) must precede the first `.qtsq` persist,
    because the QTSQ C codec does not create parent dirs.
- **D — done, as planned.** `engine.rs` worker thread + command channel, `host.rs`
  event sink + approval registry, `commands.rs` Tauri commands, keychain-backed API
  keys via `embed_config` (never plaintext).
- **E — done, as planned.** React/TS frontend per §4a; typecheck + vite build green.
- **F — done.** Electron files replaced in place; `make desktop` / `desktop-dev`
  targets; full `npm run tauri build` verified: `Sofuu.app` 8.3 MB +
  `Sofuu_0.1.0_aarch64.dmg` 2.4 MB (no size gate on desktop — the 5 MB cap is
  CLI-only).
- **QTSQ checkout drift (fixed repo-wide):** the refreshed `libqtsq.a` references
  `qtc_decompress_*` from its own `compressor/libqtc.a`; `sofuu-ffi/build.rs` now
  links it whenever the checkout is present.
- **Verification gate results (2026-08-24):** `cargo test --release --workspace`
  216/216 (was 194; +session_js/chdir/shipped tests) · `examples/agent_test.js`
  ALL PASSED · `tests/chat_compact_e2e.sh` ALL PASSED · `make test` 20/20 ·
  `make size-check` 2.2 MB ≤ 5 MB · `tests/chat_js_smoke.js` ALL OK (mesh
  artifacts written, canonicalized roots).
