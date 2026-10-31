// sofuu-core — Rust chat UI (Phase 1).
//
// This is the Rust port of src/cli/chat.c (now retired from the build).
// It owns:
//   - the chat session config (~/.sofuu/config.json) load/save
//   - slash-command dispatch (/model /provider /effort /compact ...)
//   - the JS driver that does streaming (via sofuu.ai.* through QuickJS)
//
// The C runtime registers sofuu.ai.*, __readline, __ttyRaw/__ttyNormal,
// sofuu.memory, etc. We register our own __chat_* callbacks into the same
// QuickJS global and eval a small driver string — identical architecture to
// the old C chat.c, but the command logic is now safe Rust.

use sofuu_ffi::bridge::{
    js_new_bool, js_new_string, js_to_string, register_global_fn, JSCFunction, JSContext,
    JSValue, JSValueConst,
};
use sofuu_ffi::SofuuRuntime;
use std::os::raw::c_int;
use std::path::PathBuf;
use std::ptr;
use std::sync::{LazyLock, Mutex};

use crate::{output_archive, session, theme};

// ── Session config ──────────────────────────────────────────────

/// One entry from ~/.sofuu/mcp.json — a named MCP server command the chat
/// connects to at startup (zero-config tool wiring).
#[derive(Clone, Debug)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProviderEntry {
    pub name: String,
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub profile: String,
    /// Models remembered for this provider: every model the user set here
    /// (picked or typed) plus the last successful live listing. The /model
    /// picker replays them when the endpoint is unreachable. Absent in old
    /// configs (serde default) and omitted from JSON while empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ChatConfig {
    pub provider: String,
    pub model: String,
    pub effort: String,
    pub brain: bool,
    /// PLAN-ML-GATES: the context-economy gates (date fix, supervisor rule
    /// nudges, later the four tiny models). Default ON; /ml on|off.
    /// SOFUU_NO_ML=1 unregisters sofuu.ml entirely regardless of this flag.
    pub ml: bool,
    pub sync: bool,
    /// API key for the CURRENT (active) provider, persisted in ~/.sofuu/config.json.
    /// An env var (e.g. OPENAI_API_KEY) still takes precedence at request
    /// time (the C layer prefers opts.api_key, then the env).
    pub api_key: String,
    /// P3 (AUDIT-2026-09-07): the api_key came from `-k/--apikey` on argv —
    /// honored for this session only; save() never writes it to disk (argv
    /// is readable by any local process, so it must not silently become the
    /// durable key in config.json).
    pub api_key_from_cli: bool,
    /// Optional base URL override for the current provider (e.g. a
    /// self-hosted OpenAI-compatible endpoint). Empty = built-in URL.
    pub base_url: String,
    /// Wire format for provider "custom": openai|anthropic|local.
    /// Empty/"openai" = OpenAI-compatible (the common case).
    pub profile: String,
    /// Full registry of user-added providers. `provider`/`model`/etc above are
    /// a flat mirror of the `active` entry for backward compat.
    pub providers: Vec<ProviderEntry>,
    pub active: String,
    /// RLM routing for long-context turns: ""/"off" | "on" | "auto"
    /// ("auto" = the R3 heuristic decides per turn).
    pub rlm: String,
    /// Permission profile enforced by the ONE agent loop (agent.js):
    /// "full" | "edit" (read-only + jailed local writes, no shell/MCP) |
    /// "plan" (read-only). Same semantics as the desktop's profiles —
    /// the gate itself is shared in src/js/agent.js permissionBlocked().
    pub permissions: String,
    /// TUI theme name (see theme.rs — 13 muted dark-blend schemes).
    /// Unknown/empty resolves to the default at render time, never errors.
    pub theme: String,
    /// Per-session context window in tokens (used for history trimming and
    /// RLM routing). 0 = provider default (see default_ctx_window).
    pub ctx_window: i64,
    /// True when `ctx_window` was set deliberately for a specific model
    /// (`/ctx <n>` or `--ctx-window`) rather than inherited as a loose
    /// global. An explicit value is OBEYED even when it sits above the
    /// strongest known evidence (the user knows their endpoint's plan);
    /// an inherited one shrinks to the model's real bound, which is what
    /// keeps a stale global from shadowing a smaller model. Reset by
    /// `/ctx default`.
    pub ctx_window_explicit: bool,
    /// Per-session max OUTPUT tokens per response. 0 = provider default.
    pub max_output: i64,
    /// Optional REMOTE embeddings provider+model for the brain (recall /
    /// store). Both empty (default) = always use the bundled, offline
    /// `sofuu.ai.embedLocal` — no embeddings provider is required.
    pub embed_provider: String,
    pub embed_model: String,
    /// Memory embedding backend (PLAN-TINY-SEMANTIC-EMBEDDER §7):
    /// ""/"hash" = 768-dim trigram hash (default); "semantic" = opt-in
    /// 64-dim learned projector. SOFUU_MEMORY_BACKEND env overrides.
    /// Remote embed_provider+embed_model bypass local selection.
    pub memory_backend: String,
    /// Optional explicit brain file override (config.json "brain_path").
    /// Empty = the runtime default chain: <cwd>/.sofuu/brain/brain.qtsq
    /// (project-local — every workspace carries its own memories), then
    /// ~/.sofuu/brain/brain.qtsq. The agent runtime AND the chat driver's
    /// direct brain ops (/remember, /share, /import, ghost) honor the same
    /// value so they can never split stores; the driver still opens NO
    /// handle of its own — it shares the runtime's via sofuu.agent.brainFor.
    pub brain_path: String,
    /// F6: per-model pricing table, $ per 1M input/output tokens.
    /// Key = "provider/model" (e.g. "openai/gpt-4o"). Seeded with defaults.
    pub pricing: std::collections::BTreeMap<String, [f64; 2]>,
    /// F6: session spend cap in USD. 0 = off.
    pub budget_usd: f64,
    /// F6: all-time spend in USD (persisted across sessions).
    pub spend_total_usd: f64,
    /// F10: ghost prompt completion (default off — most "surprising" feature).
    pub ghost: bool,
    /// M6 (PLAN-MEMORY-TOKENS): soft RAM tripwire in MB — past this the
    /// footer RAM metric turns amber and a one-time notice prints. Inform,
    /// never kill. 0 = default (1024).
    pub rss_warn_mb: i64,
    /// P2 (PLAN-MEMORY-TOKENS): recall similarity floor for the chat brain.
    /// 0 = default (0.30, agent.js RECALL_MIN_SCORE).
    pub recall_min: f64,
    /// P2 (PLAN-MEMORY-TOKENS): recall block token budget for the chat brain.
    /// 0 = default (scales with the model's context window, agent.js).
    pub recall_budget: i64,
    /// Models this session learned cannot think: the provider rejected a
    /// thinking/reasoning parameter for them. The effort picker reports
    /// "does not support thinking" and turns run without effort for these
    /// (auto-detected once from the API error text; persisted).
    pub no_think_models: Vec<String>,
    /// Project-local output archive controls. The archive is additive to the
    /// session stream and defaults to local-only, redacted, and enabled.
    pub archive_enabled: bool,
    pub archive_retain_final: bool,
    pub archive_retain_tool_results: bool,
    pub archive_retain_partial: bool,
    pub archive_redact: bool,
    pub archive_max_bytes: u64,
    pub archive_max_count: usize,
    pub archive_max_age_secs: u64,
    pub archive_auto_recover: bool,
    pub archive_recovery_budget_chars: usize,
    pub threads: usize,
}

/// P3 (AUDIT-2026-09-07): f64 bits of the lifetime spend last flushed to
/// config.json. The per-turn cost hook used to rewrite the whole config
/// (every provider API key included) on EVERY turn just to persist the
/// running spend; it now flushes only when ≥1 more cent has accrued since
/// this mark. Updated by save() after a successful write.
static SPEND_SAVED_BITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Default for ChatConfig {
    /* One source of truth for "fresh config" — the derived Default would
     * silently disagree with defaults() (e.g. ml must be ON by default). */
    fn default() -> Self {
        Self::defaults()
    }
}

impl ChatConfig {
    pub fn active_entry(&self) -> Option<&ProviderEntry> {
        if self.active.is_empty() { return None; }
        self.providers.iter().find(|p| p.name == self.active)
    }
    fn sync_flat_from_active(&mut self) {
        if let Some(e) = self.active_entry().cloned() {
            self.provider = e.name;
            self.model = e.model;
            self.base_url = e.endpoint;
            self.api_key = e.api_key;
            self.profile = e.profile;
        }
    }
    fn upsert_provider(&mut self, entry: ProviderEntry) {
        let name = entry.name.clone();
        if let Some(existing) = self.providers.iter_mut().find(|p| p.name == name) {
            *existing = entry;
        } else {
            self.providers.push(entry);
        }
        self.active = name;
        self.sync_flat_from_active();
    }
}

impl ChatConfig {
    /// No default provider/model: the user picks both on first run (the
    /// provider wizard auto-launches when either is unset).
    pub fn defaults() -> Self {
        let mut pricing = std::collections::BTreeMap::new();
        // F6: seed defaults for popular models ($/1M in/out tokens).
        pricing.insert("openai/gpt-4o".into(), [2.50, 10.00]);
        pricing.insert("openai/gpt-4o-mini".into(), [0.15, 0.60]);
        pricing.insert("openai/gpt-4.1".into(), [2.00, 8.00]);
        pricing.insert("openai/gpt-4.1-mini".into(), [0.40, 1.60]);
        pricing.insert("openai/o1".into(), [15.00, 60.00]);
        pricing.insert("anthropic/claude-sonnet-4-6".into(), [3.00, 15.00]);
        pricing.insert("anthropic/claude-opus-4-1".into(), [15.00, 75.00]);
        pricing.insert("anthropic/claude-haiku-3-5".into(), [0.80, 4.00]);
        Self {
            provider: String::new(),
            model: String::new(),
            effort: "high".into(),
            brain: false,
            ml: true,
            sync: true,
            api_key: String::new(),
            api_key_from_cli: false,
            base_url: String::new(),
            profile: String::new(),
            providers: Vec::new(),
            active: String::new(),
            rlm: String::new(),
            permissions: "full".into(),
            theme: theme::DEFAULT_THEME.into(),
            embed_provider: String::new(),
            embed_model: String::new(),
            memory_backend: String::new(),
            brain_path: String::new(),
            ctx_window: 0,
            ctx_window_explicit: false,
            max_output: 0,
            pricing,
            budget_usd: 0.0,
            spend_total_usd: 0.0,
            ghost: false,
            rss_warn_mb: 1024,
            recall_min: 0.0,
            recall_budget: 0,
            no_think_models: Vec::new(),
            archive_enabled: true,
            archive_retain_final: true,
            archive_retain_tool_results: false,
            archive_retain_partial: true,
            archive_redact: true,
            archive_max_bytes: 100 * 1024 * 1024,
            archive_max_count: 10_000,
            archive_max_age_secs: 0,
            archive_auto_recover: true,
            archive_recovery_budget_chars: 12_000,
            threads: 0,
        }
    }

    pub(crate) fn archive_policy(&self) -> crate::output_archive::ArchivePolicy {
        crate::output_archive::ArchivePolicy {
            enabled: self.archive_enabled,
            retain_final: self.archive_retain_final,
            retain_tool_results: self.archive_retain_tool_results,
            retain_partial: self.archive_retain_partial,
            redact: self.archive_redact,
            max_bytes: self.archive_max_bytes,
            max_count: self.archive_max_count,
            max_age_secs: self.archive_max_age_secs,
            automatic_recovery: self.archive_auto_recover,
            recovery_budget_chars: self.archive_recovery_budget_chars,
        }
    }

    #[cfg(test)]
    fn config_dir_override() -> Option<PathBuf> {
        // Tests use a temp dir so they never touch the real ~/.sofuu.
        std::env::var("SOFUU_TEST_CONFIG_DIR").ok().map(PathBuf::from)
    }

    fn config_dir() -> PathBuf {
        #[cfg(test)]
        if let Some(d) = Self::config_dir_override() {
            return d;
        }
        /* Embedded hosts redirect all ~/.sofuu derivations to their
         * configured config_root (PLAN-HEADLESS H2.4). */
        if let Some(root) = sofuu_core::embed_config::get_config_root() {
            return PathBuf::from(root);
        }
        let home = sofuu_core::embed_config::home_dir().unwrap_or_else(|| ".".into());
        PathBuf::from(home).join(".sofuu")
    }

    pub fn config_path() -> PathBuf {
        Self::config_dir().join("config.json")
    }

    /// Optional MCP server list at ~/.sofuu/mcp.json:
    /// `[{"name":"fs","command":"npx @…/server-filesystem /tmp"}]`.
    /// Missing/corrupt → empty list (a broken file must never break chat).
    pub fn mcp_servers() -> Vec<McpServerConfig> {
        let path = Self::config_dir().join("mcp.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Vec::new();
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            /* Corrupt mcp.json must still not break chat (empty list), but
             * it must not be SILENT either — the user's servers silently
             * vanished. The file is left untouched for manual repair. */
            eprintln!("\x1b[33mWarning:\x1b[0m {} is corrupt — MCP servers disabled until it is fixed (left in place, not overwritten)",
                path.display());
            return Vec::new();
        };
        let Some(arr) = v.as_array() else {
            return Vec::new();
        };
        arr.iter()
            .filter_map(|item| {
                let name = item.get("name").and_then(|x| x.as_str())?.to_string();
                let command = item.get("command").and_then(|x| x.as_str())?.to_string();
                Some(McpServerConfig { name, command })
            })
            .collect()
    }

    /// Load config from ~/.sofuu/config.json. Missing/corrupt → defaults.
    pub fn load() -> Self {
        let mut cfg = Self::defaults();
        let Ok(text) = std::fs::read_to_string(Self::config_path()) else {
            return cfg;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            /* Corrupt config.json must still not break chat (defaults),
             * but silently losing every provider/key/model setting is a
             * data-loss trap: the next save would overwrite the file.
             * Warn loudly; the file itself is left for manual repair. */
            eprintln!("\x1b[33mWarning:\x1b[0m {} is corrupt — using defaults; providers/settings from this file are being ignored (left in place, not overwritten)",
                Self::config_path().display());
            return cfg;
        };
        // New registry: providers + active. Legacy flat "provider" synthesizes one entry.
        if let Some(arr) = v.get("providers").and_then(|x| x.as_array()) {
            for item in arr {
                let name = item.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
                if name.is_empty() { continue; }
                let endpoint = item.get("endpoint").and_then(|x| x.as_str()).or_else(|| item.get("base_url").and_then(|x| x.as_str())).unwrap_or("").to_string();
                let api_key = item.get("api_key").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let model = item.get("model").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let profile = item.get("profile").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let models = item.get("models").and_then(|x| x.as_array()).map(|arr| {
                    arr.iter().filter_map(|m| m.as_str()).map(|m| m.to_string()).collect()
                }).unwrap_or_default();
                cfg.providers.push(ProviderEntry { name, endpoint, api_key, model, profile, models });
            }
            if let Some(s) = v.get("active").and_then(|x| x.as_str()) {
                cfg.active = s.to_string();
            }
        }
        // Legacy flat mirror (always read — used to synthesize missing registry or as fallback).
        // When providers was already populated, these just seed the flat mirror; active determines truth.
        let mut legacy_provider = String::new();
        let mut legacy_model = String::new();
        let mut legacy_api_key = String::new();
        let mut legacy_base_url = String::new();
        let mut legacy_profile = String::new();
        if let Some(s) = v.get("provider").and_then(|x| x.as_str()) { legacy_provider = s.to_string(); }
        if let Some(s) = v.get("model").and_then(|x| x.as_str()) { legacy_model = s.to_string(); }
        if let Some(s) = v.get("api_key").and_then(|x| x.as_str()) { legacy_api_key = s.to_string(); }
        if let Some(s) = v.get("base_url").and_then(|x| x.as_str()) { legacy_base_url = s.to_string(); }
        if let Some(s) = v.get("profile").and_then(|x| x.as_str()) { legacy_profile = s.to_string(); }
        if cfg.providers.is_empty() && !legacy_provider.is_empty() {
            cfg.providers.push(ProviderEntry { name: legacy_provider.clone(), endpoint: legacy_base_url.clone(), api_key: legacy_api_key.clone(), model: legacy_model.clone(), profile: legacy_profile.clone(), models: Vec::new() });
            cfg.active = legacy_provider.clone();
        }
        // Flat mirror: prefer active entry when registry exists, else legacy flat.
        if let Some(active_entry) = cfg.providers.iter().find(|p| p.name == cfg.active).cloned() {
            cfg.provider = active_entry.name;
            cfg.model = active_entry.model;
            cfg.base_url = active_entry.endpoint;
            cfg.api_key = active_entry.api_key;
            cfg.profile = active_entry.profile;
        } else {
            cfg.provider = legacy_provider;
            cfg.model = legacy_model;
            cfg.base_url = legacy_base_url;
            cfg.api_key = legacy_api_key;
            cfg.profile = legacy_profile;
            if !cfg.provider.is_empty() && cfg.active.is_empty() { cfg.active = cfg.provider.clone(); }
        }
        if let Some(s) = v.get("effort").and_then(|x| x.as_str()) { cfg.effort = s.to_string(); }
        if let Some(b) = v.get("brain").and_then(|x| x.as_bool()) { cfg.brain = b; }
        if let Some(b) = v.get("ml").and_then(|x| x.as_bool()) { cfg.ml = b; }
        if let Some(b) = v.get("sync").and_then(|x| x.as_bool()) { cfg.sync = b; }
        if let Some(s) = v.get("rlm").and_then(|x| x.as_str()) { cfg.rlm = s.to_string(); }
        // TUI theme: only known names persist — a typo falls back to the
        // default at render time, but we don't enshrine it in the file.
        if let Some(s) = v.get("theme").and_then(|x| x.as_str()) {
            if theme::resolve(s).is_some() {
                cfg.theme = s.to_string();
            }
        }
        // Permission profile: only the three known modes are honored — a
        // typo in config.json must never mean "more access" (default full).
        if let Some(s) = v.get("permissions").and_then(|x| x.as_str()) {
            if s == "full" || s == "edit" || s == "plan" {
                cfg.permissions = s.to_string();
            }
        }
        if let Some(s) = v.get("embed_provider").and_then(|x| x.as_str()) { cfg.embed_provider = s.to_string(); }
        if let Some(s) = v.get("embed_model").and_then(|x| x.as_str()) { cfg.embed_model = s.to_string(); }
        if let Some(s) = v.get("memory_backend").and_then(|x| x.as_str()) { cfg.memory_backend = s.to_string(); }
        if let Some(s) = v.get("brain_path").and_then(|x| x.as_str()) { cfg.brain_path = s.to_string(); }
        if let Some(n) = v.get("ctx_window").and_then(|x| x.as_i64()) { cfg.ctx_window = clamp_ctx_window(n); }
        if let Some(b) = v.get("ctx_window_explicit").and_then(|x| x.as_bool()) { cfg.ctx_window_explicit = b; }
        if let Some(n) = v.get("max_output").and_then(|x| x.as_i64()) { cfg.max_output = clamp_max_output(n); }
        // F6/F7/F10 config fields.
        if let Some(p) = v.get("pricing").and_then(|x| x.as_object()) {
            for (k, val) in p {
                if let Some(arr) = val.as_array() {
                    let in_p = arr.first().and_then(|x| x.as_f64()).unwrap_or(0.0);
                    let out_p = arr.get(1).and_then(|x| x.as_f64()).unwrap_or(0.0);
                    cfg.pricing.insert(k.clone(), [in_p, out_p]);
                }
            }
        }
        if let Some(n) = v.get("budget_usd").and_then(|x| x.as_f64()) { cfg.budget_usd = n; }
        if let Some(n) = v.get("spend_total_usd").and_then(|x| x.as_f64()) { cfg.spend_total_usd = n; }
        if let Some(b) = v.get("ghost").and_then(|x| x.as_bool()) { cfg.ghost = b; }
        if let Some(n) = v.get("rss_warn_mb").and_then(|x| x.as_i64()) { cfg.rss_warn_mb = n; }
        if let Some(n) = v.get("recall_min").and_then(|x| x.as_f64()) { cfg.recall_min = n; }
        if let Some(n) = v.get("recall_budget").and_then(|x| x.as_i64()) { cfg.recall_budget = n; }
        if let Some(arr) = v.get("no_think_models").and_then(|x| x.as_array()) {
            cfg.no_think_models = arr
                .iter()
                .filter_map(|x| x.as_str())
                .map(|s| s.to_string())
                .collect();
        }
        if let Some(b) = v.get("archive_enabled").and_then(|x| x.as_bool()) { cfg.archive_enabled = b; }
        if let Some(b) = v.get("archive_retain_final").and_then(|x| x.as_bool()) { cfg.archive_retain_final = b; }
        if let Some(b) = v.get("archive_retain_tool_results").and_then(|x| x.as_bool()) { cfg.archive_retain_tool_results = b; }
        if let Some(b) = v.get("archive_retain_partial").and_then(|x| x.as_bool()) { cfg.archive_retain_partial = b; }
        if let Some(b) = v.get("archive_redact").and_then(|x| x.as_bool()) { cfg.archive_redact = b; }
        if let Some(n) = v.get("archive_max_bytes").and_then(|x| x.as_u64()) { cfg.archive_max_bytes = n.min(1u64 << 40); }
        if let Some(n) = v.get("archive_max_count").and_then(|x| x.as_u64()) { cfg.archive_max_count = n.min(100_000) as usize; }
        if let Some(n) = v.get("archive_max_age_secs").and_then(|x| x.as_u64()) { cfg.archive_max_age_secs = n; }
        if let Some(b) = v.get("archive_auto_recover").and_then(|x| x.as_bool()) { cfg.archive_auto_recover = b; }
        if let Some(n) = v.get("archive_recovery_budget_chars").and_then(|x| x.as_u64()) { cfg.archive_recovery_budget_chars = n.clamp(512, 131_072) as usize; }
        // Validate active still points at an entry; if not, fall back to first provider.
        if !cfg.active.is_empty() && !cfg.providers.iter().any(|p| p.name == cfg.active) {
            cfg.active = cfg.providers.first().map(|p| p.name.clone()).unwrap_or_default();
            if let Some(entry) = cfg.providers.iter().find(|p| p.name == cfg.active).cloned() {
                cfg.provider = entry.name; cfg.model = entry.model; cfg.base_url = entry.endpoint; cfg.api_key = entry.api_key; cfg.profile = entry.profile;
            }
        }
        if !cfg.ctx_window_explicit && !cfg.model.is_empty() {
            let is_gguf = cfg.model.ends_with(".gguf")
                || sofuu_core::infer::clean_model_path(&cfg.model).extension().map_or(false, |e| e == "gguf");
            if is_gguf {
                let caps = sofuu_core::rt::model_caps::lookup(Some(&cfg.model));
                if caps.ctx_window > 0 {
                    cfg.ctx_window = caps.ctx_window as i64;
                }
                if caps.max_output > 0 {
                    cfg.max_output = caps.max_output as i64;
                }
                if caps.thinking != sofuu_core::rt::model_caps::Thinking::None && (cfg.effort == "off" || cfg.effort.is_empty()) {
                    cfg.effort = "high".into();
                }
            }
        }
        cfg
    }

    pub fn save(&self) {
        #[cfg(not(test))]
        {
            // Sync flat mirror from active before writing, so legacy readers boot.
            let mut flat_provider = self.provider.clone();
            let mut flat_model = self.model.clone();
            let mut flat_base = self.base_url.clone();
            /* P3 (AUDIT-2026-09-07): a `-k/--apikey` session key never
             * lands on disk (see api_key_from_cli) — write the empty flat
             * key instead so save() can't turn an argv secret into the
             * durable config.json one. */
            let mut flat_key = if self.api_key_from_cli { String::new() } else { self.api_key.clone() };
            let mut flat_profile = self.profile.clone();
            if let Some(e) = self.providers.iter().find(|p| p.name == self.active) {
                flat_provider = e.name.clone(); flat_model = e.model.clone(); flat_base = e.endpoint.clone(); flat_key = e.api_key.clone(); flat_profile = e.profile.clone();
            }
            let dir = Self::config_dir();
            let _ = std::fs::create_dir_all(&dir);
            let json = serde_json::json!({
                /* P3 (R4.1): the config is a persisted artifact, so it carries
                 * the §5 schema_version; the reader treats absence as v0 and
                 * migrates by normalizing on the next save (R4.4: forward-only). */
                "schema_version": 1,
                "providers": self.providers,
                "active": self.active,
                "provider": flat_provider,
                "model": flat_model,
                "effort": self.effort,
                "brain": self.brain,
                "ml": self.ml,
                "sync": self.sync,
                "api_key": flat_key,
                "base_url": flat_base,
                "profile": flat_profile,
                "rlm": self.rlm,
                "permissions": self.permissions,
                "theme": self.theme,
                "embed_provider": self.embed_provider,
                "embed_model": self.embed_model,
                "memory_backend": self.memory_backend,
                "brain_path": self.brain_path,
                "ctx_window": self.ctx_window,
                "ctx_window_explicit": self.ctx_window_explicit,
                "max_output": self.max_output,
                "pricing": self.pricing,
                "budget_usd": self.budget_usd,
                "spend_total_usd": self.spend_total_usd,
                "ghost": self.ghost,
                "rss_warn_mb": self.rss_warn_mb,
                "recall_min": self.recall_min,
                "recall_budget": self.recall_budget,
                "no_think_models": self.no_think_models,
                "archive_enabled": self.archive_enabled,
                "archive_retain_final": self.archive_retain_final,
                "archive_retain_tool_results": self.archive_retain_tool_results,
                "archive_retain_partial": self.archive_retain_partial,
                "archive_redact": self.archive_redact,
                "archive_max_bytes": self.archive_max_bytes,
                "archive_max_count": self.archive_max_count,
                "archive_max_age_secs": self.archive_max_age_secs,
                "archive_auto_recover": self.archive_auto_recover,
                "archive_recovery_budget_chars": self.archive_recovery_budget_chars,
            })
            .to_string();
            let tmp = dir.join("config.json.tmp");
            /* P2-10: the config carries the API key — create the tmp with
             * 0600 from the FIRST write instead of write-then-chmod (which
             * left a world-readable window on every save). Non-unix has no
             * mode API; the write is the best available there. */
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                use std::io::Write;
                let _ = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .mode(0o600)
                    .open(&tmp)
                    .and_then(|mut f| f.write_all(json.as_bytes()));
            }
            #[cfg(not(unix))]
            {
                let _ = std::fs::write(&tmp, &json);
            }
            let _ = std::fs::rename(&tmp, Self::config_path());
            /* P3: record the flushed spend mark so the per-turn cost hook
             * (js_chat_report_usage) can throttle full-config rewrites to
             * once per accrued cent. */
            SPEND_SAVED_BITS.store(
                self.spend_total_usd.to_bits(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }
}

// ── Slash commands ──────────────────────────────────────────────

/// Wire profiles the TUI accepts by name. "ollama" and "llamacpp" are the
/// local inference engines (PLUGIN_PLAN Phase 2): ollama speaks its own
/// /api/chat wire (rt/ai.rs maps it to the local provider), llamacpp is an
/// OpenAI-compatible llama-server reached through --base-url.
const VALID_PROVIDERS: [&str; 7] = [
    "openai", "anthropic", "local", "ollama", "llamacpp", "lmstudio", "vllm",
];
const VALID_EFFORTS: [&str; 4] = ["low", "medium", "high", "max"];

/// Hard ceiling on a user-configured context window: 4Mi tokens.
/// Real 1M-window models (Gemini 2.5 Pro, GPT-4.1, GLM 5) are 1,048,576
/// tokens — the previous 1,000,000 ceiling made the /ctx picker's own "1M"
/// preset (1048576) REJECTED, and a literal `/ctx 1m` unusable. 2M-class
/// models now fit too. Smaller models still reject oversized requests at
/// the API and the error surfaces (and is learned).
const MAX_CTX_WINDOW: i64 = 4_194_304;

/// Parse a token count the way people type it: `1048576`, `128000`,
/// `128k`, `1m`, `1.5m`, `2M`. Suffixes are binary (1k = 1024, 1m =
/// 1048576) because every real model window is a power of two.
pub fn parse_token_count(s: &str) -> Option<i64> {
    let t = s.trim().to_ascii_lowercase();
    if t.is_empty() {
        return None;
    }
    let (num, mult) = if let Some(v) = t.strip_suffix('k') {
        (v, 1024.0)
    } else if let Some(v) = t.strip_suffix('m') {
        (v, 1024.0 * 1024.0)
    } else {
        (t.as_str(), 1.0)
    };
    let num = num.trim();
    let n: f64 = num.parse().ok()?;
    if !n.is_finite() || n <= 0.0 {
        return None;
    }
    Some((n * mult).round() as i64)
}

/// Hard ceiling on a user-configured max OUTPUT tokens: 384k. Models like
/// Gemini 2.5 Pro support up to 64k output natively; some reasoning models
/// and long-form agents accept far more. The API rejects requests above a
/// model's real ceiling and the error surfaces.
const MAX_OUTPUT_TOKENS: i64 = 384_000;

/// Legacy per-provider fallback context windows — used ONLY when neither
/// /ctx nor the model capability registry (rt/model_caps.rs) knows better.
/// The registry is the primary source; this map is the last resort.
fn default_ctx_window(provider: &str) -> i64 {
    match provider {
        "openai" | "anthropic" => 128_000,
        /* Local engines serve a modest window by default; the registry still
         * wins whenever it knows the real model. */
        "local" | "ollama" | "llamacpp" => 32_768,
        _ => 32_768,
    }
}

/// Effective window: the configured value, or the provider default.
pub fn effective_ctx_window(provider: &str, ctx_window: i64) -> i64 {
    if ctx_window > 0 {
        ctx_window
    } else {
        default_ctx_window(provider)
    }
}

/// The evidence-ladder resolve for the SELECTED model (the same one every
/// JS driver consumer uses via sofuu.ai.resolveCaps): learned-from-400s >
/// endpoint-discovered > registry > default, with the flat config honored
/// only as far as the strictest real evidence allows. This is what /ctx
/// reports back to the user — a global 1M override can no longer claim
/// the model has 1M when its real window is 262k.
fn resolved_ctx_window(cfg: &ChatConfig) -> i64 {
    resolved_ctx_caps(cfg).window
}

/// Full resolve for the active model, honoring the explicitness flag.
fn resolved_ctx_caps(cfg: &ChatConfig) -> sofuu_core::ml::alloc::policy::Resolved {
    sofuu_core::ml::alloc::policy::resolve_explicit(
        if cfg.model.is_empty() { None } else { Some(cfg.model.as_str()) },
        cfg.ctx_window,
        0,
        if cfg.base_url.is_empty() { None } else { Some(cfg.base_url.as_str()) },
        cfg.ctx_window_explicit,
    )
}

/// Clamp a configured window into [1024, 1M]; 0/negative = provider default.
fn clamp_ctx_window(n: i64) -> i64 {
    if n <= 0 {
        0
    } else {
        n.clamp(1024, MAX_CTX_WINDOW)
    }
}

/// Clamp a configured max-output into [1, 384k]; 0/negative = provider default.
fn clamp_max_output(n: i64) -> i64 {
    if n <= 0 {
        0
    } else {
        n.clamp(1, MAX_OUTPUT_TOKENS)
    }
}

/// Every slash command — the single source of truth for the C readline's
/// TAB completion (`__chat_complete`) and for "did you mean" suggestions.
const ALL_COMMANDS: [&str; 41] = [
    "/help",
    "/version",
    "/model",
    "/provider",
    "/effort",
    "/compact",
    "/clear",
    "/brain",
    "/ml",
    "/rlm",
    "/mode",
    "/theme",
    "/plan",
    "/edit",
    "/full",
    "/ctx",
    "/maxout",
    "/outputs",
    "/tools",
    "/agents",
    "/sessions",
    "/context",
    "/work",
    "/done",
    "/note",
    "/notify",
    "/sync",
    "/exit",
    // F1–F11 chat features (PLAN-CHAT-FEATURES.md)
    "/remember",
    "/why",
    "/memory",
    "/forget",
    "/resume",
    "/share",
    "/import",
    "/cost",
    "/watch",
    "/hooks",
    "/ghost",
    "/at",
    "/serve",
];

/// Commands matching a typed prefix (used by TAB completion).
fn complete_matches(prefix: &str) -> Vec<&'static str> {
    ALL_COMMANDS
        .iter()
        .copied()
        .filter(|c| c.starts_with(prefix))
        .collect()
}

/// Per-command help: description + usage/options. The TUI completion menu
/// renders these next to the highlighted command, so /<cmd> shows what it
/// does and what options it accepts.
const COMMAND_INFO: &[(&str, &str, &str)] = &[
    ("/help", "Show this command list", ""),
    ("/version", "Print the runtime version", ""),
    ("/model", "Switch the model", "<name>  (bare = interactive picker)"),
    ("/provider", "Manage providers / run setup wizard", "(bare = your providers + add)"),
    ("/effort", "Set the thinking/reasoning effort", "off | low | medium | high | max  (unsupported levels fall back to the model's highest; budgets scale with the model)"),
    ("/compact", "Summarize the conversation into context", ""),
    ("/clear", "Reset this session (keep settings)", ""),
    ("/brain", "Toggle the memory/brain integration", "on | off | info"),
    ("/ml", "ML context-economy gates + online learning", "on | off | learn | adopt | discard | reset | wrong | wasted | info"),
    ("/rlm", "Route long-context turns through RLM", "on | off | auto"),
    ("/mode", "View/set the permission mode", "full | edit | plan  (bare = show · TAB cycles)"),
    ("/theme", "Switch the TUI color theme", "[name]  (bare = picker over 13 dark-blend themes)"),
    ("/plan", "Plan mode: read-only (no writes, no shell)", ""),
    ("/edit", "Edit mode: reads + local file edits (no shell, no MCP)", ""),
    ("/full", "Full access: every tool allowed", ""),
    ("/ctx", "View/set the context window in tokens", "<tokens> | default  (default = model's real window, max 4M)"),
    ("/maxout", "View/set max output tokens per response", "<tokens> | default  (default = model's real max output, cap 384k)"),
    ("/outputs", "Inspect timestamped model/tool output history", "[list|show <id>|search <text>|context <task>|stats|rebuild-index|prune|on|off]"),
    ("/tools", "List connected MCP servers + tools", ""),
    ("/agents", "List agent definitions (+ ~/.sofuu/agents/*.js)", ""),
    ("/sessions", "Browse sessions on this project", "[id]  (bare = picker · ↑↓ + Enter resumes)"),
    ("/context", "Show a session's full context", "[id]  (default: this session)"),
    ("/work", "Announce what you are working on", "<description>"),
    ("/done", "Clear your current task", ""),
    ("/note", "Record a personal note (peers see it)", "<message>"),
    ("/notify", "Broadcast a critical notice to all sessions", "<message>"),
    ("/sync", "Toggle the session-mesh polling", "on | off"),
    ("/exit", "Save config and exit", ""),
    // F1–F11 chat features
    ("/remember", "Pin a fact to the brain + AGENTS.md", "<fact>"),
    ("/why", "Explain which memories shaped the last answer", ""),
    ("/memory", "Browse the brain's memories", "[tier|query] [--page N]  (tiers: working|episodic|semantic|entity|forgotten)"),
    ("/forget", "Delete one memory by id", "<id>  (shows the record first; add --yes to confirm)"),
    ("/resume", "Browse and resume a past session", "[id]  (bare = picker)"),
    ("/share", "Export brain as a plain JSON card (v1: metadata + pointers; not encrypted)", "[path.qtsq] [label]"),
    ("/import", "Import a brain card (merge)", "<path.qtsq>"),
    ("/cost", "Show token usage + spend breakdown", ""),
    ("/watch", "Watch a directory for changes in context", "<path> | off  (bare = list)"),
    ("/hooks", "Open ~/.sofuu/hooks.js (user middleware)", ""),
    ("/ghost", "Toggle ghost prompt completion", "on | off"),
    ("/at", "Show @file mention help and budget", ""),
    ("/serve", "Serve the brain over HTTP", "[--brain path] [--port n] [--host h] [--token t]"),
];

/// Look up the description/options for a command name.
fn command_info(cmd: &str) -> Option<(&'static str, &'static str)> {
    COMMAND_INFO
        .iter()
        .find(|(name, _, _)| *name == cmd)
        .map(|(_, desc, opts)| (*desc, *opts))
}

fn valid_provider(p: &str) -> bool {
    VALID_PROVIDERS.contains(&p)
}
fn valid_effort(e: &str) -> bool {
    VALID_EFFORTS.contains(&e)
}

/// One-line honest description of a permission mode (for /mode echoes —
/// semantics enforced once, in agent.js permissionBlocked).
fn mode_hint(mode: &str) -> &'static str {
    match mode {
        "plan" => "read-only: no writes, no shell",
        "edit" => "reads + local file edits; no shell, no MCP",
        _ => "every tool allowed",
    }
}

/// Set + persist the permission profile. The JS driver re-applies it to
/// the agent runtime (sofuu.agent.setPermissions) after every command.
fn set_mode(cfg: &mut ChatConfig, mode: &str) -> &'static str {
    cfg.permissions = mode.to_string();
    cfg.save();
    chat_out(&format!("  \u{2713} mode \u{2192} {} ({})\n", mode, mode_hint(mode)));
    "ok"
}

/// Handle a slash command. Returns a token the JS driver interprets:
///   "ok", "models", "providers", "clear", "compact", "exit", "unknown"
fn handle_slash(cfg: &mut ChatConfig, cmd: &str) -> &'static str {
    let (name, arg) = match cmd.split_once(' ') {
        Some((n, a)) => (n, a.trim()),
        None => (cmd, ""),
    };
    // One provider command only: /providers is a silent alias of /provider.
    let name = if name == "/providers" { "/provider" } else { name };

    match name {
        "/help" | "/?" => {
            print_help();
            "ok"
        }
        "/version" => {
            chat_out(&format!("  sofuu {} · QuickJS + libuv · AI-native\n", env!("CARGO_PKG_VERSION")));
            "ok"
        }
        "/exit" | "/quit" => {
            // Close the mesh session (end event + registry + .qtsq persist).
            if let Ok(mut guard) = SESS.lock() {
                if let Some(sess) = guard.as_mut() {
                    sess.finish();
                }
            }
            cfg.save();
            chat_out("\n  Bye! 👋\n");
            "exit"
        }
        // ── TUI theme ──────────────────────────────────────────
        // Bare /theme opens the picker in the driver ("pick_theme");
        // /theme <name> applies directly. Unknown names never unstyle
        // anything: the file keeps its old value and the user gets the
        // count + picker hint instead of a 13-line dump.
        "/theme" => {
            if arg.is_empty() {
                "pick_theme"
            } else if theme::resolve(arg).is_some() {
                let prev_name = cfg.theme.clone();
                cfg.theme = arg.to_string();
                cfg.save();
                let t = theme::lookup(arg);
                /* Whole-window flip: the theme owns the window background
                 * now. Set it, remap already-painted rows to the new
                 * foregrounds, and repaint everything (conversation +
                 * input chrome) so the switch lands at once instead of
                 * coloring only future rows. TTY-gated inside: piped
                 * /theme stays byte-clean. */
                if sofuu_ffi::tui_active() {
                    unsafe {
                        sofuu_core::rt::tui::tui_set_bg(t.bg as c_int);
                        sofuu_core::rt::tui::tui_set_sel(
                            theme::accent_index(t.accent).map(|v| v as c_int).unwrap_or(-1),
                        );
                    }
                    let from = theme::lookup(&prev_name);
                    sofuu_core::rt::tui::tui_remap_rows(&theme::fg_remap_pairs(from, t));
                    unsafe {
                        sofuu_core::rt::tui::tui_render_conversation();
                        sofuu_core::modules::process::tui_repaint_chrome();
                    }
                }
                chat_out(&format!(
                    "  ✓ Theme → {} ({} of 13)\n",
                    t.name,
                    theme::THEMES.iter().position(|x| x.name == t.name).map(|i| i + 1).unwrap_or(1)
                ));
                "ok"
            } else {
                chat_out(&format!(
                    "  Unknown theme '{arg}' — bare /theme lists all 13.\n"
                ));
                "ok"
            }
        }
        // ── Session mesh commands ──────────────────────────────
        "/sessions" => {
            /* Bare /sessions opens the interactive browser in the driver
             * (TTY picker with ↑↓ + Enter, plain table when piped);
             * /sessions <id> resumes directly like /resume <id>. The
             * outside-chat `sofuu sessions` CLI still prints the static
             * table via session::cmd_list. */
            "sessions"
        }
        "/context" => {
            /* Lock order is SESS → WATCH → PROJECT (documented above) —
             * take them in that order, never nested reversed (P2-17). */
            let own = SESS.lock().unwrap().as_ref().map(|s| s.id().to_string());
            let project = PROJECT.lock().unwrap().clone();
            match project {
                Some(p) => {
                    let id = if arg.is_empty() {
                        own.unwrap_or_default()
                    } else if let Some(o) = &own {
                        if arg == "self" || arg == o {
                            o.clone()
                        } else {
                            arg.to_string()
                        }
                    } else {
                        arg.to_string()
                    };
                    session::cmd_show(&p, &id);
                }
                None => chat_out(&format!("  [sync] session mesh is off (--sync on to enable)\n")),
            }
            "ok"
        }
        "/work" => {
            if arg.is_empty() {
                chat_out(&format!("  Usage: /work <what you are doing>\n"));
            } else if let Ok(mut guard) = SESS.lock() {
                if let Some(sess) = guard.as_mut() {
                    sess.set_task(Some(arg.to_string()));
                    chat_out(&format!("  ✓ Task set (visible to all sessions on this project): {}\n", arg));
                } else {
                    chat_out(&format!("  [sync] session mesh is off (--sync on to enable)\n"));
                }
            }
            "ok"
        }
        "/done" => {
            if let Ok(mut guard) = SESS.lock() {
                if let Some(sess) = guard.as_mut() {
                    sess.set_task(None);
                    chat_out(&format!("  ✓ Task cleared\n"));
                } else {
                    chat_out(&format!("  [sync] session mesh is off (--sync on to enable)\n"));
                }
            }
            "ok"
        }
        "/note" => {
            if arg.is_empty() {
                chat_out(&format!("  Usage: /note <message>\n"));
            } else if let Ok(mut guard) = SESS.lock() {
                if let Some(sess) = guard.as_mut() {
                    sess.add_note(arg);
                    chat_out(&format!("  ✓ Note recorded\n"));
                } else {
                    chat_out(&format!("  [sync] session mesh is off (--sync on to enable)\n"));
                }
            }
            "ok"
        }
        "/notify" => {
            if arg.is_empty() {
                chat_out(&format!("  Usage: /notify <critical message for all sessions>\n"));
            } else if let Ok(mut guard) = SESS.lock() {
                if let Some(sess) = guard.as_mut() {
                    sess.broadcast_critical(arg);
                    chat_out(&format!("  ✓ Critical notice broadcast to all sessions on this project: {}\n", arg));
                } else {
                    chat_out(&format!("  [sync] session mesh is off (--sync on to enable)\n"));
                }
            }
            "ok"
        }
        "/sync" => {
            // /sync → config panel (DRIVER handles "pick_sync")
            if arg.is_empty() {
                "pick_sync"
            } else if arg == "off" {
                cfg.sync = false;
                cfg.save();
                chat_out("  ✓ Sync disabled (this session stops polling peers)\n");
                "ok"
            } else if arg == "on" {
                cfg.sync = true;
                cfg.save();
                chat_out("  ✓ Sync enabled (restart chat to join the mesh)\n");
                "ok"
            } else {
                chat_out("  Usage: /sync [on|off]\n");
                "ok"
            }
        }
        "/model" => {
            if arg.is_empty() { return "pick_model"; }
            let picked = arg.trim().to_string();
            let is_gguf = picked.ends_with(".gguf")
                || sofuu_core::infer::clean_model_path(&picked).extension().map_or(false, |e| e == "gguf");
            if is_gguf {
                cfg.provider = "local".to_string();
                cfg.base_url.clear();
                cfg.api_key.clear();
                let caps = sofuu_core::rt::model_caps::lookup(Some(&picked));
                if caps.ctx_window > 0 {
                    cfg.ctx_window = caps.ctx_window as i64;
                    cfg.ctx_window_explicit = false;
                }
                if caps.max_output > 0 {
                    cfg.max_output = caps.max_output as i64;
                }
                if caps.thinking != sofuu_core::rt::model_caps::Thinking::None {
                    cfg.effort = "high".to_string();
                } else {
                    cfg.effort = "off".to_string();
                }
            }
            cfg.model = picked.clone();
            if let Some(entry) = cfg.providers.iter_mut().find(|p| p.name == cfg.active) {
                entry.model = cfg.model.clone();
                /* Remember every model set on this provider (picked from the
                 * live list or typed manually) — the /model picker replays
                 * them when the endpoint is unreachable. */
                if !picked.is_empty() && !entry.models.contains(&picked) {
                    entry.models.push(picked.clone());
                    while entry.models.len() > 50 { entry.models.remove(0); }
                }
            }
            cfg.save();
            chat_out(&format!("  ✓ Model → {}\n", cfg.model));
            "ok"
        }
        "/provider" => {
            if arg.is_empty() {
                return "pick_provider";
            }
            // Subcommands: select <name>, remove <name>, show
            if arg.starts_with("select ") {
                let name = arg["select ".len()..].trim();
                if name.is_empty() { chat_out("  Usage: /provider select <name>\n"); return "ok"; }
                if let Some(pos) = cfg.providers.iter().position(|p| p.name == name) {
                    cfg.active = name.to_string();
                    cfg.sync_flat_from_active();
                    cfg.save();
                    let _ = pos;
                    chat_out(&format!("  ✓ Active provider → {}\n", name));
                } else { chat_out(&format!("  No provider named '{}'\n", name)); }
                return "ok";
            }
            if arg.starts_with("remove ") {
                let name = arg["remove ".len()..].trim();
                if name.is_empty() { chat_out("  Usage: /provider remove <name>\n"); return "ok"; }
                let before = cfg.providers.len();
                cfg.providers.retain(|p| p.name != name);
                if cfg.providers.len() == before { chat_out(&format!("  No provider named '{}'\n", name)); }
                else {
                    if cfg.active == name { cfg.active = cfg.providers.first().map(|p| p.name.clone()).unwrap_or_default(); }
                    if cfg.active.is_empty() { cfg.provider.clear(); cfg.model.clear(); cfg.base_url.clear(); cfg.api_key.clear(); cfg.profile.clear(); } else { cfg.sync_flat_from_active(); }
                    cfg.save();
                    chat_out(&format!("  ✓ Removed provider '{}'\n", name));
                }
                return "ok";
            }
            if arg == "show" {
                if cfg.providers.is_empty() { chat_out("  No providers configured. Use /provider to add one.\n"); }
                else {
                    for p in &cfg.providers {
                        let star = if p.name == cfg.active { " ← active" } else { "" };
                        chat_out(&format!("  • {}{} · model {} · {}{}\n", p.name, star, if p.model.is_empty(){"(no model)"}else{&p.model}, p.endpoint, if p.profile.is_empty() {String::new()} else {format!(" · {}", p.profile)}));
                    }
                }
                return "ok";
            }
            // Legacy direct form: /provider openai [url] — upsert into registry.
            let parts: Vec<&str> = arg.split_whitespace().collect();
            if !valid_provider(parts[0]) {
                chat_out(&format!("  Unknown provider '{}' (openai|anthropic|local)\n  Hint: use /provider to add a named provider (e.g. orcarouter-ai)\n", parts[0]));
            } else {
                let name = parts[0].to_string();
                let base = if parts.len() >= 2 { parts[1..].join(" ") } else { String::new() };
                // Preserve existing entry's other fields when upserting via shorthand.
                let existing = cfg.providers.iter().find(|p| p.name == name).cloned();
                let entry = ProviderEntry { name: name.clone(), endpoint: if base.is_empty() { existing.as_ref().map(|e| e.endpoint.clone()).unwrap_or_default() } else { base }, api_key: existing.as_ref().map(|e| e.api_key.clone()).unwrap_or_default(), model: existing.as_ref().map(|e| e.model.clone()).unwrap_or_default(), profile: existing.as_ref().map(|e| e.profile.clone()).unwrap_or_default(), models: existing.as_ref().map(|e| e.models.clone()).unwrap_or_default() };
                cfg.upsert_provider(entry);
                cfg.save();
                if cfg.base_url.is_empty() { chat_out(&format!("  ✓ Provider → {}\n", cfg.provider)); } else { chat_out(&format!("  ✓ Provider → {} · URL → {}\n", cfg.provider, cfg.base_url)); }
            }
            "ok"
        }
        "/effort" => {
            if arg.is_empty() {
                return "pick_effort"; // interactive picker (DRIVER handles)
            } else if arg == "off" {
                cfg.effort.clear();
                cfg.save();
                chat_out(&format!("  ✓ Effort cleared (provider default)\n"));
            } else if !valid_effort(arg) {
                chat_out(&format!("  Effort must be one of: low, medium, high, max (or off)\n"));
            } else {
                cfg.effort = arg.to_string();
                cfg.save();
                /* Capability-aware confirmation: what the model will get. */
                let caps = sofuu_core::rt::model_caps::lookup(Some(cfg.model.as_str()));
                let note = match caps.thinking {
                    sofuu_core::rt::model_caps::Thinking::None => {
                        " — note: this model has no reasoning mode; the field is omitted".to_string()
                    }
                    _ => String::new(),
                };
                chat_out(&format!("  ✓ Effort → {}{}\n", cfg.effort, note));
            }
            "ok"
        }
        "/clear" => "clear",
        "/compact" => "compact",
        "/brain" => {
            // /brain → config panel (DRIVER handles "pick_brain");
            // /brain info → store panel (DRIVER handles "brain_info" — it
            // owns the shared brain handle, so the panel reads the SAME
            // store the turns recall from).
            if arg.is_empty() {
                "pick_brain"
            } else if arg == "info" {
                "brain_info"
            } else if arg == "off" {
                cfg.brain = false;
                cfg.save();
                chat_out("  ✓ Brain disabled\n");
                "ok"
            } else if arg == "on" {
                cfg.brain = true;
                cfg.save();
                chat_out("  ✓ Brain enabled\n");
                "ok"
            } else {
                chat_out("  Usage: /brain [on|off|info]\n");
                "ok"
            }
        }
        "/ml" => {
            // PLAN-ML-GATES: the context-economy gates. on|off toggles the
            // advise-only layer; learn|adopt|discard|reset|wrong|wasted|info
            // drive the online adaptation of the supervisor (§13, off by
            // default; two-step: learn trains a candidate, adopt applies it).
            if arg.is_empty() {
                let state = if cfg.ml { "on" } else { "off" };
                let (on_enabled, _obs, on_examples, on_adapted, on_adaptations, on_pending) =
                    sofuu_core::ml::online::status();
                chat_out(&format!(
                    "  ML gates: {}  --  /ml on|off | learn|adopt|discard|reset|wrong|wasted|info  (SOFUU_NO_ML=1 disables entirely)\n",
                    state
                ));
                chat_out(&format!(
                    "  online learning: {}, {} labeled example(s), adapted: {} ({} adaptation(s)){}\n",
                    if on_enabled { "on" } else { "off" },
                    on_examples,
                    if on_adapted { "yes" } else { "no" },
                    on_adaptations,
                    if on_pending { " -- candidate PENDING: /ml adopt or /ml discard" } else { "" }
                ));
                "ok"
            } else if arg == "off" {
                cfg.ml = false;
                cfg.save();
                chat_out("  ML gates disabled\n");
                "ok"
            } else if arg == "on" {
                cfg.ml = true;
                cfg.save();
                chat_out("  ML gates enabled\n");
                "ok"
            } else if arg == "learn" {
                let r = sofuu_core::ml::online::learn();
                chat_out(&format!("  /ml learn: {}\n", r.detail));
                "ok"
            } else if arg == "adopt" {
                let (adopted, persisted) = sofuu_core::ml::online::adopt();
                if adopted {
                    chat_out(&format!(
                        "  candidate adopted -- the adapted supervisor is now in charge ({})\n",
                        if persisted { "persisted" } else { "in-memory only, will not survive a restart" }
                    ));
                } else {
                    chat_out("  no pending candidate to adopt -- /ml learn first\n");
                }
                "ok"
            } else if arg == "discard" {
                if sofuu_core::ml::online::discard() {
                    chat_out("  pending candidate discarded -- the previous supervisor stays in charge\n");
                } else {
                    chat_out("  no pending candidate to discard\n");
                }
                "ok"
            } else if arg == "reset" {
                sofuu_core::ml::online::reset();
                chat_out("  online adaptation cleared -- the pretrained supervisor is back in charge\n");
                "ok"
            } else if arg == "wrong" {
                if sofuu_core::ml::online::label_wrong() {
                    chat_out("  marked the most recent supervisor flag as wrong -- /ml learn to apply\n");
                } else {
                    chat_out("  no recent supervisor flag to mark wrong\n");
                }
                "ok"
            } else if arg == "wasted" {
                if sofuu_core::ml::online::label_wasted() {
                    chat_out("  marked the most recent call as waste -- /ml learn to apply\n");
                } else {
                    chat_out("  no recent call to mark wasted\n");
                }
                "ok"
            } else if arg == "info" {
                let (on_enabled, obs, on_examples, on_adapted, on_adaptations, on_pending) =
                    sofuu_core::ml::online::status();
                let (runs, calls, segs, seg_tokens) = sofuu_core::ml::context::summary();
                chat_out(&format!(
                    "  supervisor: trained (baked weights), threshold {:.2}, online layer: {}\n",
                    sofuu_core::ml::supervisor::model::THRESHOLD,
                    if on_adapted { "adopted" } else { "pretrained" }
                ));
                chat_out(&format!(
                    "  online: {}, {} observation(s) awaiting labels, {} labeled example(s), {} adaptation(s){}\n",
                    if on_enabled { "on" } else { "off" },
                    obs,
                    on_examples,
                    on_adaptations,
                    if on_pending { ", candidate pending (/ml adopt|discard)" } else { "" }
                ));
                chat_out(&format!(
                    "  working set: {} run(s), {} call(s), {} segment(s), {} segment token(s)\n",
                    runs, calls, segs, seg_tokens
                ));
                "ok"
            } else {
                chat_out("  Usage: /ml [on|off|learn|adopt|discard|reset|wrong|wasted|info]\n");
                "ok"
            }
        }
        "/rlm" => {
            // /rlm            → config panel (DRIVER handles "pick_rlm")
            // /rlm on|off|auto → set + persist ("auto" = heuristic routing)
            match arg {
                "" => "pick_rlm",
                "on" | "off" | "auto" => {
                    cfg.rlm = if arg == "off" { String::new() } else { arg.to_string() };
                    cfg.save();
                    let hint = match arg {
                        "on" => "every turn goes through the RLM loop",
                        "auto" => "long-context turns route to RLM via the heuristic",
                        _ => "RLM off, plain model calls only",
                    };
                    chat_out(&format!("  ✓ RLM → {} ({})\n", arg, hint));
                    "ok"
                }
                _ => {
                    chat_out(&format!("  Usage: /rlm [on|off|auto]\n"));
                    "ok"
                }
            }
        }
        /* Permission modes. /mode is the query+set command; /plan /edit
         * /full are the direct shorthands. Enforcement lives in agent.js
         * (permissionBlocked); this persists the choice and the JS driver
         * re-applies it to the runtime after every command (applyMode). */
        "/mode" => match arg {
            "" => {
                chat_out(&format!("  mode: {} ({})\n", cfg.permissions, mode_hint(&cfg.permissions)));
                chat_out("  usage: /mode full|edit|plan  (shorthands: /full /edit /plan · TAB cycles)\n");
                "ok"
            }
            "full" | "edit" | "plan" => set_mode(cfg, arg),
            _ => {
                chat_out("  Usage: /mode [full|edit|plan]\n");
                "ok"
            }
        },
        "/plan" => set_mode(cfg, "plan"),
        "/edit" => set_mode(cfg, "edit"),
        "/full" => set_mode(cfg, "full"),
        "/outputs" => {
            match arg {
                "on" => {
                    cfg.archive_enabled = true;
                    cfg.save();
                    chat_out("  ✓ Output archive enabled\n");
                    "ok"
                }
                "off" => {
                    cfg.archive_enabled = false;
                    cfg.save();
                    chat_out("  ✓ Output archive disabled (existing records remain)\n");
                    "ok"
                }
                _ => "outputs",
            }
        }
        "/tools" => "tools",
        "/agents" => "agents",
        "/ctx" => {
            // /ctx          → config panel (DRIVER handles "pick_ctx")
            // /ctx <n>      → set for THIS model (obeyed as typed)
            // /ctx default  → reset to the provider default
            if arg.is_empty() {
                "pick_ctx"
            } else if arg == "default" || arg == "0" {
                cfg.ctx_window = 0;
                cfg.ctx_window_explicit = false;
                cfg.save();
                let eff = resolved_ctx_window(&cfg);
                chat_out(&format!("  ✓ Context window → provider default ({eff} tokens)\n"));
                "ok"
            } else if let Some(n) = parse_token_count(&arg) {
                if n > MAX_CTX_WINDOW {
                    chat_out(&format!(
                        "  Context window must be between 0 and {} tokens (0 = provider default)\n",
                        MAX_CTX_WINDOW
                    ));
                } else {
                    cfg.ctx_window = clamp_ctx_window(n);
                    cfg.ctx_window_explicit = true;
                    cfg.save();
                    let r = resolved_ctx_caps(&cfg);
                    chat_out(&format!("  ✓ Context window → {} tokens\n", r.window));
                    /* Say so when the value is above everything we know —
                     * obeyed as typed, but the user should see the gap
                     * rather than discover it as a provider 400 later. */
                    if let Some(bound) = r.config_exceeds_evidence {
                        if r.window > bound {
                            chat_out(&format!(
                                "  ⚠ {r_window} is above the strongest known evidence ({bound} tokens, {} source) — honored as set; if the provider rejects it, the real limit is learned automatically\n",
                                r.win_source.as_str(),
                                r_window = r.window
                            ));
                        } else {
                            chat_out(&format!(
                                "  · inherited value clamped to the model's known limit: {bound} tokens ({})\n",
                                r.win_source.as_str()
                            ));
                        }
                    }
                }
                "ok"
            } else {
                chat_out(&format!("  Usage: /ctx [<tokens>|default]  (0–{}; 0 = provider default)\n", MAX_CTX_WINDOW));
                "ok"
            }
        }
        "/maxout" => {
            // /maxout        → config panel (DRIVER handles "pick_maxout")
            // /maxout <n>    → set (clamped to [1, 384k]; 0 = provider default)
            // /maxout default→ reset to the provider default
            if arg.is_empty() {
                "pick_maxout"
            } else if arg == "default" || arg == "0" {
                cfg.max_output = 0;
                cfg.save();
                let eff = sofuu_core::ml::alloc::policy::resolve(
                    if cfg.model.is_empty() { None } else { Some(cfg.model.as_str()) },
                    0,
                    0,
                    if cfg.base_url.is_empty() { None } else { Some(cfg.base_url.as_str()) },
                )
                .max_output;
                chat_out(&format!("  ✓ Max output tokens → provider default ({eff})\n"));
                "ok"
            } else if let Some(n) = parse_token_count(&arg) {
                if n > MAX_OUTPUT_TOKENS {
                    chat_out(&format!(
                        "  Max output tokens must be between 0 and {} (0 = provider default)\n",
                        MAX_OUTPUT_TOKENS
                    ));
                } else {
                    cfg.max_output = clamp_max_output(n);
                    cfg.save();
                    /* Honest feedback: what the ladder will actually send
                     * for THIS model, not the raw config number. */
                    let eff = sofuu_core::ml::alloc::policy::resolve(
                        if cfg.model.is_empty() { None } else { Some(cfg.model.as_str()) },
                        0,
                        cfg.max_output,
                        if cfg.base_url.is_empty() { None } else { Some(cfg.base_url.as_str()) },
                    )
                    .max_output;
                    let note = if eff < cfg.max_output {
                        format!(" (clamped from {} — the model's real max is {})", cfg.max_output, eff)
                    } else {
                        String::new()
                    };
                    chat_out(&format!("  ✓ Max output tokens → {}{}\n", cfg.max_output, note));
                }
                "ok"
            } else {
                chat_out(&format!("  Usage: /maxout [<tokens>|default]  (0–{}; 0 = provider default)\n", MAX_OUTPUT_TOKENS));
                "ok"
            }
        }
        // ── F1: /remember + /why ───────────────────────────────────────
        "/remember" => {
            if arg.is_empty() {
                chat_out("  Usage: /remember <fact>\n");
                chat_out("  Pins a fact to the brain AND the project AGENTS.md (standing context).\n");
                return "ok";
            }
            "remember"
        }
        "/why" => "why",
        // ── B1: /memory + /forget (brain read-out) ─────────────────────
        "/memory" => "memory",
        "/forget" => {
            if arg.is_empty() {
                chat_out("  Usage: /forget <id>  (find ids with /memory)\n");
                chat_out("  Shows the record first; add --yes to delete it.\n");
                return "ok";
            }
            "forget"
        }
        // ── F2: /resume ────────────────────────────────────────────────
        "/resume" => {
            if arg.is_empty() {
                "resume"
            } else {
                // /resume <id> → direct resume (the driver reads the id from the raw command)
                "resume"
            }
        }
        // ── F3: @file mentions (the /at command shows help; the actual
        // expansion happens in the driver's expandMentions at turn() entry) ──
        "/at" => {
            chat_out("  \x1b[1m@file mentions\x1b[0m\n");
            chat_out("  Type @path or @path:start-end in a prompt to attach file contents.\n");
            chat_out("  Paths resolve against cwd; paths escaping the project root are rejected.\n");
            chat_out("  Attachments are token-budgeted (default 8192, config: attach_budget).\n");
            chat_out("  History stores the manifest only — follow-up turns don't re-send files.\n\n");
            chat_out("  \x1b[1m@agent mentions\x1b[0m\n");
            chat_out("  Type @name <task> or @agent:name <task> to run a loaded agent\n");
            chat_out("  (~/.sofuu/agents/*.js) directly with its own definition — a focused\n");
            chat_out("  run, no chat history injected. Agents win over same-named files;\n");
            chat_out("  unknown names fall through to the @file path. Esc stops the run.\n\n");
            chat_out("  \x1b[2mExamples:\x1b[0m\n");
            chat_out("    explain @src/main.rs\n");
            chat_out("    review @lib/utils.rs:1-40 @lib/api.rs:50-100\n");
            chat_out("    @researcher compare Rust and Zig error handling\n");
            "ok"
        }
        // ── F5: /share + /import (brain cards) ─────────────────────────
        "/share" => "share",
        "/import" => {
            if arg.is_empty() {
                chat_out("  Usage: /import <path.qtsq>\n");
                chat_out("  Imports a brain card and merges it into the current brain.\n");
                return "ok";
            }
            "import"
        }
        // ── F6: /cost ──────────────────────────────────────────────────
        "/cost" => "cost",
        // ── F8: /watch ─────────────────────────────────────────────────
        "/watch" => "watch",
        // ── F9: /hooks ─────────────────────────────────────────────────
        "/hooks" => {
            let hooks_path = ChatConfig::config_dir().join("hooks.js");
            if !hooks_path.exists() {
                chat_out("  No hooks.js found. Create one at:\n");
                chat_out(&format!("    {}\n\n", hooks_path.display()));
                chat_out("  \x1b[2m// ~/.sofuu/hooks.js — user middleware\x1b[0m\n");
                chat_out("  \x1b[2mexport async function pre({ text, cfg }) {\x1b[0m\n");
                chat_out("  \x1b[2m  return text; // or { text, skip }\x1b[0m\n");
                chat_out("  \x1b[2m}\x1b[0m\n");
                chat_out("  \x1b[2mexport async function post({ text, answer, usage }) {\x1b[0m\n");
                chat_out("  \x1b[2m  return; // void = keep answer as-is\x1b[0m\n");
                chat_out("  \x1b[2m}\x1b[0m\n");
            } else {
                chat_out(&format!("  hooks.js: {}\n", hooks_path.display()));
                chat_out("  Edit it and restart chat to reload.\n");
            }
            "ok"
        }
        // ── F10: /ghost ────────────────────────────────────────────────
        "/ghost" => {
            if arg.is_empty() {
                chat_out(&format!("  Ghost completion: {}\n", if cfg.ghost { "on" } else { "off" }));
                chat_out("  Usage: /ghost on|off\n");
                return "ok";
            } else if arg == "on" {
                cfg.ghost = true;
                cfg.save();
                chat_out("  ✓ Ghost completion enabled\n");
            } else if arg == "off" {
                cfg.ghost = false;
                cfg.save();
                chat_out("  ✓ Ghost completion disabled\n");
            } else {
                chat_out("  Usage: /ghost [on|off]\n");
            }
            "ok"
        }
        // ── F11: /serve (informational — the actual serve runs as `sofuu serve`) ──
        "/serve" => {
            chat_out("  Brain server: run `sofuu serve --brain <path> --port <n>`\n");
            chat_out("  Default: 127.0.0.1:7707, auto-generated token.\n");
            chat_out("  Endpoints: GET /health, POST /remember, GET /recall?q=…, POST /share\n");
            "ok"
        }
        _ => {
            // "Did you mean" — suggest commands that start with the type name.
            let (name, _) = cmd.split_once(' ').unwrap_or((cmd, ""));
            let sugg: Vec<&str> = complete_matches(name)
                .into_iter()
                .filter(|c| *c != name)
                .collect();
            if sugg.is_empty() {
                chat_out(&format!(
                    "  \x1b[31mUnknown command: {}\x1b[0m — try /help (or press TAB after '/')\n",
                    name
                ));
            } else {
                chat_out(&format!(
                    "  \x1b[31mUnknown command: {}\x1b[0m — did you mean: {}?\n",
                    name,
                    sugg.join(", ")
                ));
            }
            "unknown"
        }
    }
}

fn print_help() {
    // Rows are collected before rendering so the command column can be sized
    // to the longest label. It used to be a fixed 20 cells, which silently
    // broke the moment a command outgrew it: `/mode [full|edit|plan]` (21)
    // and `/ml [...]` (53) pushed their em-dashes out and left the block
    // ragged. Measuring first means the next long command cannot do that.
    //
    // Capped so a very long label cannot push the description off an 80-col
    // terminal: past the cap the label gets the full width and the
    // description wraps under it, which is still aligned.
    enum Row {
        Header(&'static str),
        Cmd(&'static str, &'static str),
        Note(&'static str),
        Blank,
    }
    let mut rows: Vec<Row> = Vec::new();
    let h = |rows: &mut Vec<Row>, s: &'static str| rows.push(Row::Header(s));
    let c = |rows: &mut Vec<Row>, cmd: &'static str, desc: &'static str| {
        rows.push(Row::Cmd(cmd, desc))
    };
    rows.push(Row::Blank);
    h(&mut rows, "  Sofuu Chat Commands");
    c(&mut rows, "/help", "this help");
    c(&mut rows, "/model", "interactive model picker (search, arrows, enter)");
    c(&mut rows, "/model <name>", "switch model directly (persisted)");
    c(&mut rows, "/provider", "your providers + add a new one");
    c(&mut rows, "/effort", "reasoning-effort picker (off/low…max)");
    c(&mut rows, "/compact", "summarize the conversation into context");
    c(&mut rows, "/clear", "reset this session (keep settings)");
    c(&mut rows, "/brain [on|off|info]", "toggle memory/brain integration · info = store panel");
    c(&mut rows, "/ml [on|off|learn|adopt|discard|reset|wrong|wasted|info]", "ML context-economy gates + supervisor online learning");
    c(&mut rows, "/rlm [on|off|auto]", "route long-context turns through the RLM loop");
    c(&mut rows, "/mode [full|edit|plan]", "permission mode: plan=read-only, edit=no shell, full=all tools");
    c(&mut rows, "/plan /edit /full", "shorthands for /mode plan|edit|full");
    c(&mut rows, "/theme [name]", "switch the TUI color theme (13 dark-blend)");
    c(&mut rows, "/ctx [<tokens>]", "view/set the context window (max 4M)");
    c(&mut rows, "/maxout [<tokens>]", "view/set max output tokens (max 384k)");
    c(&mut rows, "/tools", "list connected MCP servers + their tools");
    c(&mut rows, "/agents", "list agent definitions (~/.sofuu/agents/*.js)");
    c(&mut rows, "/version", "print the runtime version");
    h(&mut rows, "  Brain & memory");
    c(&mut rows, "/remember <fact>", "pin a fact directly to the brain");
    c(&mut rows, "/why", "which memories shaped the last answer");
    c(&mut rows, "/memory [tier|query]", "browse stored memories (paginated)");
    c(&mut rows, "/forget <id>", "delete one memory (shows it first; --yes confirms)");
    c(&mut rows, "/share [path]", "export a brain card (metadata + pointers)");
    c(&mut rows, "/import <path>", "import + merge a brain card");
    c(&mut rows, "/resume [id]", "browse + resume a past session");
    c(&mut rows, "/serve", "brain-server quick info (sofuu serve --brain)");
    h(&mut rows, "  Context & tools");
    c(&mut rows, "@file[:start-end]", "attach file contents to a prompt");
    c(&mut rows, "@name <task>", "run a loaded agent directly (@agent:name too)");
    c(&mut rows, "/watch <path>", "watch a directory for changes in context");
    c(&mut rows, "/cost", "token usage + spend breakdown + budget");
    c(&mut rows, "/ghost [on|off]", "toggle ghost prompt completion");
    c(&mut rows, "/hooks", "show ~/.sofuu/hooks.js user middleware info");
    h(&mut rows, "  Session mesh");
    c(&mut rows, "/sessions [id]", "browse sessions on this project (↑↓ + Enter resumes)");
    c(&mut rows, "/context [id]", "full context of a session (default: this one)");
    c(&mut rows, "/work <desc>", "announce what you are working on");
    c(&mut rows, "/done", "clear your current task");
    c(&mut rows, "/note <msg>", "record a personal note (seen by peers)");
    c(&mut rows, "/notify <msg>", "CRITICAL notice, shown to every session now");
    c(&mut rows, "/sync [on|off]", "toggle session-mesh polling (restart to join)");
    c(&mut rows, "/exit /quit", "save config and exit");
    rows.push(Row::Note("\n\x1b[2m  Sessions on the SAME project share context in real time (tasks,\x1b[0m"));
    rows.push(Row::Note("\x1b[2m  notes, critical notices). Data persists as .qtsq files in\x1b[0m"));
    rows.push(Row::Note("\x1b[2m  <project>/.sofuu/sessions/.\x1b[0m"));
    h(&mut rows, "\n  Shortcuts");
    c(&mut rows, "TAB", "complete after '/', cycle permission mode otherwise");
    c(&mut rows, "↑ / ↓", "input history");
    c(&mut rows, "PgUp / PgDn", "scroll the conversation");
    c(&mut rows, "Shift+Enter", "newline inside the input (Alt+Enter too)");
    c(&mut rows, "Esc", "stop the response (while streaming) / clear the input");
    c(&mut rows, "Ctrl-K", "copy the mouse-selected text (drag over rows, then Ctrl-K)");
    c(&mut rows, "Ctrl-C", "quit sofuu");
    c(&mut rows, "Ctrl-D", "exit");
    rows.push(Row::Blank);

    // ── Render ──────────────────────────────────────────────────
    let col = help_command_col(rows.iter().filter_map(|r| match r {
        Row::Cmd(cmd, _) => Some(char_cells_str(cmd)),
        _ => None,
    }));
    // Wrap to the real terminal when there is one. tui_width() floors at 40;
    // piped, there is no width, so assume the classic 80.
    let term_w = if sofuu_ffi::tui_active() {
        sofuu_ffi::tui_width().max(40) as usize
    } else {
        80
    };

    for row in &rows {
        match row {
            Row::Blank => chat_out(""),
            Row::Header(s) => chat_out(&format!("\n\x1b[1;36m{s}\x1b[0m")),
            Row::Note(s) => chat_out(&format!("\x1b[2m{s}\x1b[0m")),
            Row::Cmd(cmd, desc) => {
                // 4 indent + column + "— " puts the description at DESC_COL.
                let desc_col = 4 + col + 2;
                let w = char_cells_str(cmd);
                let label = if w <= col {
                    let pad = col - w;
                    format!("    \x1b[1m{cmd}{}\x1b[0m", " ".repeat(pad))
                } else {
                    // Label wider than the column: it keeps its own row and
                    // the description starts on the next one, still at the
                    // description column. `/ml` carries eight sub-commands
                    // and is unavoidably wide.
                    chat_out(&format!("    \x1b[1m{cmd}\x1b[0m"));
                    " ".repeat(desc_col)
                };
                // Wrap the description with a hanging indent so an 80-column
                // terminal gets a clean block instead of the terminal
                // soft-wrapping it mid-sentence.
                for (i, chunk) in wrap_help_desc(desc, term_w.saturating_sub(desc_col)).iter().enumerate() {
                    if i == 0 {
                        chat_out(&format!("{label}\x1b[2m— {chunk}\x1b[0m"));
                    } else {
                        chat_out(&format!("{}\x1b[2m{chunk}\x1b[0m", " ".repeat(desc_col)));
                    }
                }
            }
        }
    }
}

/// Display width of a UTF-8 string in terminal cells (char_cells() per
/// char). Used by the help renderer to size the command column so wide
/// glyphs — `↑ / ↓`, `…` — are measured, not assumed to be one cell.
fn char_cells_str(s: &str) -> usize {
    s.chars().map(char_cells).sum()
}

/// Width of the help screen's command column, in terminal cells.
///
/// Sized to the longest label so the em-dash column is identical on every
/// row, and clamped to keep an 80-column terminal usable: 4 indent + 24
/// label + 2 (em-dash + space) = 30, leaving 50 for the description. A
/// label wider than the cap keeps its own row and drops its description to
/// the next line at this column, so it cannot ragged the rest of the block.
///
/// This replaces a hardcoded 20, which broke alignment for `/mode
/// [full|edit|plan]` (21 cells) and `/ml [...]` (53).
/// Greedy word wrap of a help description to `width` terminal cells.
///
/// Returns the pieces to print one per line. A word longer than the budget
/// is hard-split rather than allowed to overflow — a URL or a long option
/// list must not push the row past the terminal edge.
fn wrap_help_desc(desc: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut out: Vec<String> = Vec::new();
    let mut line = String::new();
    for word in desc.split_whitespace() {
        let mut w = word;
        // Hard-split any single word that cannot fit on a line of its own.
        while char_cells_str(w) > width {
            if !line.is_empty() {
                out.push(std::mem::take(&mut line));
            }
            let mut head = String::new();
            for ch in w.chars() {
                if char_cells_str(&head) + char_cells(ch) > width {
                    break;
                }
                head.push(ch);
            }
            if head.is_empty() {
                break; // cannot make progress; avoid an infinite loop
            }
            out.push(head.clone());
            w = &w[head.len()..];
        }
        if !line.is_empty() && char_cells_str(&line) + 1 + char_cells_str(w) > width {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(w);
    }
    if !line.is_empty() || out.is_empty() {
        out.push(line);
    }
    out
}

fn help_command_col<'a>(labels: impl Iterator<Item = usize>) -> usize {
    const MIN: usize = 20;
    const MAX: usize = 24;
    labels.max().unwrap_or(MIN).clamp(MIN, MAX)
}

// ── Welcome panel ────────────────────────────────────────────────
// The bordered box shown at the top of the TUI: title, then
// column-aligned session facts. Built in Rust (testable), rendered by the
// C conversation area line by line.

/// Display width in terminal cells: skips ANSI CSI sequences; wide chars
/// (CJK/emoji) count 2 cells like a real terminal. Kept in sync with
/// char_cells/tui_disp_width() in rt/tui.rs — the panel pads with this,
/// the viewport truncates with that, and they must agree.
fn char_cells(c: char) -> usize {
    let o = c as u32;
    // Zero-width first: these occupy NO cell, and getting them wrong
    // mis-pads the welcome panel and truncates the viewport at the wrong
    // place. A combining accent ("e" + U+0301) is one cell on screen, not
    // two; neither is a variation selector or a ZWJ in an emoji sequence.
    if (0x0300..=0x036F).contains(&o)      // combining diacritical marks
        || (0x200B..=0x200F).contains(&o)   // ZWSP..ZWJ + bidi marks
        || o == 0xFEFF                     // zero-width no-break space
        || (0xFE00..=0xFE0F).contains(&o)   // variation selectors 1-16
        || (0xE0100..=0xE01EF).contains(&o) // variation selectors 17-256
        || (0x1F3FB..=0x1F3FF).contains(&o) // emoji skin-tone modifiers
    {
        return 0;
    }
    if (0x1100..=0x115F).contains(&o)
        || (0x2E80..=0xA4CF).contains(&o)
        || (0xAC00..=0xD7A3).contains(&o)
        || (0xF900..=0xFAFF).contains(&o)
        || (0xFE30..=0xFE6F).contains(&o)
        || (0xFF00..=0xFF60).contains(&o)
        || (0xFFE0..=0xFFE6).contains(&o)
        || (0x1F300..=0x1F64F).contains(&o)
        || (0x1F900..=0x1F9FF).contains(&o)
        || (0x20000..=0x3FFFD).contains(&o)
    {
        2
    } else {
        1
    }
}

fn cell_w(s: &str) -> usize {
    let mut w = 0;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            w += char_cells(c);
            continue;
        }
        // skip the CSI sequence: ESC [ … final(0x40..0x7E)
        if chars.peek() == Some(&'[') {
            chars.next();
            for c2 in chars.by_ref() {
                if (0x40..=0x7e).contains(&(c2 as u32)) {
                    break;
                }
            }
        }
    }
    w
}

/// Truncate plain (escape-free) text to `max` DISPLAY CELLS, marking the
/// cut.
///
/// Cell-based, not char-based. This used to compare and slice on
/// `chars().count()`, so a CJK/emoji value was allowed 2x the cells it
/// occupies: a 60-cell panel row holding wide glyphs came out 102 cells
/// wide and the right │ landed far off-screen, tearing the box.
fn trunc_cells(s: &str, max: usize) -> String {
    if cell_w(s) <= max {
        return s.to_string();
    }
    // Take whole chars while they FIT, reserving one cell for the ellipsis.
    let mut t = String::new();
    let mut w = 0;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        let cw = char_cells(c);
        if w + cw > max.saturating_sub(1) {
            break;
        }
        t.push(c);
        w += cw;
    }
    format!("{t}…")
}

/// UTF-8 sequence length implied by a lead byte (mirrors truncate_cells
/// in rt/tui.rs — a truncated tail is clamped with .min(len) at the call).
fn ln_of(c: u8) -> usize {
    if c >= 0xF0 {
        4
    } else if c >= 0xE0 {
        3
    } else if c >= 0xC0 {
        2
    } else {
        1
    }
}

/// Clip possibly-colored content to `max` visible cells. ANSI CSI
/// sequences are copied whole (zero cells); never splits a UTF-8 glyph;
/// appends a reset when clipped so the rest of the row renders sanely.
fn clip_cells(s: &str, max: usize) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(s.len());
    let mut cells = 0;
    let mut i = 0;
    let mut clipped = false;
    while i < b.len() {
        if b[i] == 0x1b {
            out.push(b[i]);
            i += 1;
            if i < b.len() && b[i] == b'[' {
                out.push(b[i]);
                while i + 1 < b.len() {
                    i += 1;
                    out.push(b[i]);
                    if (0x40..=0x7e).contains(&b[i]) {
                        i += 1;
                        break;
                    }
                }
            }
            continue;
        }
        if b[i] & 0xC0 != 0x80 {
            /* Count DISPLAY CELLS, not one per char. A wide glyph (CJK,
             * emoji) is 2 cells; counting it as 1 let clip_cells return
             * content twice as wide as the caller asked for, which
             * panel_row then padded against — the panel row overflowed its
             * own box. Use the same char_cells the measuring side
             * (cell_w) uses, so measure and clip can never disagree. */
            let cw = std::str::from_utf8(&b[i..(i + ln_of(b[i])).min(b.len())])
                .ok()
                .and_then(|ch| ch.chars().next())
                .map(char_cells)
                .unwrap_or(1);
            if cells + cw > max {
                clipped = true;
                break;
            }
            cells += cw;
        }
        out.push(b[i]);
        i += 1;
    }
    if clipped {
        out.extend_from_slice(b"\x1b[0m");
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// SGR open sequence for the active theme's panel border (dim + accent
/// hue), e.g. "\x1b[2;38;5;139m". The border follows the theme; unknown
/// names fall back to the default via theme::lookup.
fn theme_panel(cfg: &ChatConfig) -> String {
    format!("\x1b[{}m", theme::lookup(&cfg.theme).panel)
}

/// SGR open sequence for the active theme's accent (model names, titles).
fn theme_accent(cfg: &ChatConfig) -> String {
    format!("\x1b[{}m", theme::lookup(&cfg.theme).accent)
}

/// One box row: `│  <content><pad>│` at exactly `width` cells.
fn panel_row(content: &str, width: usize, border: &str) -> String {
    let inner = width.saturating_sub(4);
    let clipped;
    // ASCII "..." (not the ambiguous-width ellipsis) so a clipped row's
    // right border stays put on every terminal.
    let content = if cell_w(content) > inner {
        clipped = format!("{}...", clip_cells(content, inner.saturating_sub(3)));
        &clipped
    } else {
        content
    };
    let pad = inner.saturating_sub(cell_w(content));
    format!("{border}│\x1b[0m  {content}{pad}{border}│\x1b[0m", pad = " ".repeat(pad))
}

/// The 2 welcome-panel heading rows (title, then the subtitle aligned
/// under it). No mascot — the animated Shima-enaga mark was removed; the
/// title now stands alone at a fixed 2-cell indent.
fn welcome_bird_rows(cfg: &ChatConfig, _session: &str, _dir: &str, w: usize) -> Vec<String> {
    let border = theme_panel(cfg);
    let accent = theme_accent(cfg);
    vec![
        panel_row(&format!("  {accent}Welcome to Sofuu!\x1b[0m"), w, &border),
        panel_row(
            "  \x1b[2mAsk anything (/help for the command list)\x1b[0m",
            w,
            &border,
        ),
    ]
}

/// The full welcome panel, one String per display row, `width` cells wide.
/// `width` is the caller's VIEWPORT BUDGET (terminal width − renderer
/// gutter − one spare cell), not the raw terminal width: every returned
/// row must pass the TUI soft-wrap budget unchanged. `session` = mesh
/// short id ("" = no session).
fn welcome_panel_at(cfg: &ChatConfig, session: &str, dir: &str, width: usize) -> Vec<String> {
    // Floor 30, not 40: tui_size floors the terminal at 40, so the caller's
    // budget bottoms out at 37 — lifting that back up to 40 would push every
    // row over the wrap budget again and tear the box on minimal terminals.
    // Overflowing content is clipped by panel_row, not fixed by widening.
    let w = width.clamp(30, 400);
    let inner = w - 4;
    let mut rows = Vec::with_capacity(12);

    let dash = "─".repeat(w - 2);
    let border = theme_panel(cfg);
    rows.push(format!("{border}╭{dash}╮\x1b[0m"));
    rows.extend(welcome_bird_rows(cfg, session, dir, w));
    rows.push(panel_row("", w, &border));

    let effort = if cfg.effort.is_empty() {
        String::new()
    } else {
        format!(" - effort {}", cfg.effort)
    };
    let model = if cfg.provider.is_empty() || cfg.model.is_empty() {
        "not configured · run /provider to pick one".to_string()
    } else if cfg.provider == "local" {
        format!("local: {}{}", cfg.model, effort)
    } else {
        format!("{}/{}{}", cfg.provider, cfg.model, effort)
    };
    let labels: [(String, String); 6] = [
        ("Directory:".into(), trunc_cells(dir, inner.saturating_sub(14))),
        (
            "Session:".into(),
            if session.is_empty() { "none".into() } else { session.into() },
        ),
        ("Model:".into(), trunc_cells(&model, inner.saturating_sub(14))),
        (
            "Mode:".into(),
            match cfg.permissions.as_str() {
                "plan" => "plan - read-only (no writes, no shell)".into(),
                "edit" => "edit - local writes, no shell".into(),
                _ => "full access".into(),
            },
        ),
        (
            "Memory:".into(),
            if cfg.brain {
                /* Never claim persistence the binary cannot deliver: a
                 * QTSQ-free build no-ops every write while this line used
                 * to say "persists across sessions" — the exact lie that
                 * shipped in the 2026-09-22 macOS tarball. */
                if sofuu_ffi::qtsq_linked() {
                    "on (persists across sessions)".to_string()
                } else {
                    "on, but NOT persisting (QTSQ codec not linked in this build — run `sofuu doctor`)".to_string()
                }
            } else {
                "off (/brain on to enable)".to_string()
            },
        ),
        ("Version:".into(), env!("CARGO_PKG_VERSION").into()),
    ];
    for (label, value) in labels {
        // Dim label, plain value — the escape layout is identical on every
        // row so the value column stays byte-aligned (see the layout test).
        rows.push(panel_row(
            &format!("\x1b[2m{label:<10}\x1b[0m  {value}"),
            w,
            &border,
        ));
    }
    rows.push(format!("{border}╰{dash}╯\x1b[0m"));

    rows.push(String::new());
    if session.is_empty() {
        rows.push("  \x1b[90mNo session here yet — one starts with your first message.\x1b[0m".into());
    }
    rows
}

/// Compact two-line banner for non-TTY output (piped / logged runs).
fn welcome_plain(cfg: &ChatConfig, session: Option<&str>, dir: &str) -> Vec<String> {
    let effort = if cfg.effort.is_empty() {
        String::new()
    } else {
        format!(" · effort {}", cfg.effort)
    };
    let model_str = if cfg.provider.is_empty() || cfg.model.is_empty() {
        "not configured · run /provider to pick one".to_string()
    } else if cfg.provider == "local" {
        format!("local: {}{}", cfg.model, effort)
    } else {
        format!("{}/{}{}", cfg.provider, cfg.model, effort)
    };
    vec![
        format!(
            "\x1b[1;35m  ✻ sofuu\x1b[0m \x1b[90m— AI-native JS runtime · v{}\x1b[0m",
            env!("CARGO_PKG_VERSION")
        ),
        format!(
            "\x1b[90m  {} · {} · session {} — type /help, or just ask\x1b[0m",
            model_str,
            dir,
            session.unwrap_or("none"),
        ),
        String::new(),
    ]
}

// ── C ↔ Rust bridge (registered into QuickJS) ──────────────────
//
// The JS driver calls these. Because JSCFunction is a C fn pointer with no
// capture, we route through a Mutex<ChatConfig> global.

static CFG: Mutex<Option<ChatConfig>> = Mutex::new(None);
static EXIT_REQUESTED: Mutex<bool> = Mutex::new(false);
// Session mesh state (None when --sync off or no project root).
static SESS: Mutex<Option<session::Session>> = Mutex::new(None);
static WATCH: Mutex<Option<session::PeerWatch>> = Mutex::new(None);
static PROJECT: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Lock order across these three statics is always SESS → WATCH → PROJECT.

// F8: filesystem watcher state — watches directories for mtime/size changes
// and surfaces them through the __chat_poll tick (no new FFI needed).
static FS_WATCHER: Mutex<FsWatcher> = Mutex::new(FsWatcher::new());
// F1: last recall hits for /why — populated by the JS driver via __chat_set_recall.
static LAST_RECALL: Mutex<Vec<RecallHit>> = Mutex::new(Vec::new());
// F6: per-session cost tracking (the driver reports usage each turn).
static SESSION_COST: Mutex<SessionCost> = Mutex::new(SessionCost::new());

/// F1: one recalled memory hit (for /why). B6.3: `tier` says which tier
/// served the hit (0 working, 1 episodic, 2 semantic, 4 entity) — 0 is
/// also the serde default so older driver JSON (no tier) reads as working
/// only when the driver actually sent tier 0; absent tiers from old
/// drivers read as episodic (1) — the pre-B6 everything-is-persistent
/// world. The driver always sends the tier now.
#[derive(Clone, serde::Serialize)]
struct RecallHit {
    text: String,
    score: f64,
    role: String,
    #[serde(default = "default_recall_tier")]
    tier: u8,
}

fn default_recall_tier() -> u8 {
    1 /* episodic — the pre-B6 default for hits without a tier */
}

/// F6: per-session cost tracking.
#[derive(Default)]
struct SessionCost {
    /// Per-turn breakdowns: (model, input_tokens, output_tokens, cost_usd,
    /// cache_read_tokens, cache_write_tokens) — cache slots are 0 for
    /// providers that don't report them (P6.4). Ring-capped: only the most
    /// recent MAX_TURNS are kept (older turns fold into session_total).
    turns: Vec<(String, u64, u64, f64, u64, u64)>,
    /// Cumulative session spend in USD.
    session_total: f64,
    /// Cumulative cache-hit tokens this session (P6.4 surface).
    cache_read_total: u64,
}

impl SessionCost {
    /// Bounded history: enough for /cost detail, never unbounded.
    const MAX_TURNS: usize = 512;
    const fn new() -> Self {
        Self { turns: Vec::new(), session_total: 0.0, cache_read_total: 0 }
    }
    fn push_turn(&mut self, t: (String, u64, u64, f64, u64, u64)) {
        // Ring: drop oldest when at cap so thousands of turns stay O(1).
        if self.turns.len() >= Self::MAX_TURNS {
            self.turns.remove(0);
        }
        self.turns.push(t);
    }
}

/// F8: filesystem watcher — tracks mtimes/sizes of watched paths.
struct FsWatcher {
    /// (path, HashMap<file_path, (mtime, size)>)
    paths: Vec<(PathBuf, std::collections::HashMap<PathBuf, (u64, u64)>)>,
}

impl FsWatcher {
    /// Max watched dirs: each snapshots up to 2000 files, so cap dirs too.
    const MAX_WATCH_DIRS: usize = 32;
    const fn new() -> Self {
        Self { paths: Vec::new() }
    }

    fn add(&mut self, path: &str) {
        let p = PathBuf::from(path);
        if self.paths.iter().any(|(watched, _)| *watched == p) {
            return; // already watching
        }
        if self.paths.len() >= Self::MAX_WATCH_DIRS {
            // Evict oldest so the vec stays bounded over long sessions.
            self.paths.remove(0);
        }
        let snapshot = self.snapshot_dir(&p);
        self.paths.push((p, snapshot));
    }

    fn clear(&mut self) {
        self.paths.clear();
    }

    fn list(&self) -> Vec<String> {
        self.paths.iter().map(|(p, _)| p.display().to_string()).collect()
    }

    fn snapshot_dir(&self, dir: &std::path::Path) -> std::collections::HashMap<PathBuf, (u64, u64)> {
        let mut map = std::collections::HashMap::new();
        let mut count = 0u32;
        self.walk(dir, &mut map, &mut count);
        map
    }

    fn walk(&self, dir: &std::path::Path, map: &mut std::collections::HashMap<PathBuf, (u64, u64)>, count: &mut u32) {
        if *count > 2000 { return; } // F8: 2k file cap
        let Ok(entries) = std::fs::read_dir(dir) else { return; };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            // Skip-lists (F8 risk mitigation)
            if name == ".git" || name == "node_modules" || name == "target" || name == ".sofuu" {
                continue;
            }
            if path.is_dir() {
                self.walk(&path, map, count);
            } else {
                if *count > 2000 { return; }
                if let Ok(meta) = entry.metadata() {
                    let mtime = meta.modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    map.insert(path, (mtime, meta.len()));
                    *count += 1;
                }
            }
        }
    }

    /// Poll for changes, return Vec of (path, kind) and drain.
    fn poll(&mut self) -> Vec<(String, String)> {
        let mut changes = Vec::new();
        // Collect the watched dirs first to avoid borrowing self.paths
        // mutably while also calling self.snapshot_dir (immutable).
        let dirs: Vec<PathBuf> = self.paths.iter().map(|(d, _)| d.clone()).collect();
        for dir in &dirs {
            let new_snapshot = self.snapshot_dir(dir);
            // Find the index of this dir in self.paths
            if let Some(idx) = self.paths.iter().position(|(d, _)| d == dir) {
                let old_snapshot = &mut self.paths[idx].1;
                // Check for modified + created files
                for (path, (new_mtime, new_size)) in &new_snapshot {
                    match old_snapshot.get(path) {
                        Some((old_mtime, old_size)) => {
                            if new_mtime != old_mtime || new_size != old_size {
                                changes.push((path.display().to_string(), "modified".into()));
                            }
                        }
                        None => {
                            changes.push((path.display().to_string(), "created".into()));
                        }
                    }
                }
                // Check for deleted files
                let deleted: Vec<String> = old_snapshot.keys()
                    .filter(|path| !new_snapshot.contains_key(*path))
                    .map(|path| path.display().to_string())
                    .collect();
                for path in deleted {
                    changes.push((path, "deleted".into()));
                }
                *old_snapshot = new_snapshot;
            }
        }
        changes
    }
}

unsafe extern "C" fn js_chat_slash(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut result = "unknown";
    if argc >= 1 {
        let arg = js_to_string(ctx, *argv);
        if let Some(cmd) = arg {
            if let Ok(mut guard) = CFG.lock() {
                let cfg = guard.get_or_insert_with(ChatConfig::defaults);
                result = handle_slash(cfg, &cmd);
                if result == "exit" {
                    *EXIT_REQUESTED.lock().unwrap() = true;
                }
            }
        }
    }
    js_new_string(ctx, result)
}

unsafe extern "C" fn js_chat_getcfg(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let json = {
        let guard = CFG.lock().unwrap();
        /* P3 (AUDIT-2026-09-07): borrow instead of cloning the whole
         * ChatConfig (providers registry + pricing table included) on
         * every serialization; json! serializes through references. */
        let default_cfg;
        let cfg: &ChatConfig = match guard.as_ref() {
            Some(c) => c,
            None => {
                default_cfg = ChatConfig::defaults();
                &default_cfg
            }
        };
        serde_json::json!({
            "provider": cfg.provider,
            "model": cfg.model,
            "effort": cfg.effort,
            "brain": cfg.brain,
            "ml": cfg.ml,
            "sync": cfg.sync,
            "api_key": cfg.api_key,
            "base_url": cfg.base_url,
            "profile": cfg.profile,
            "providers": cfg.providers,
            "active": cfg.active,
            "rlm": cfg.rlm,
            "permissions": cfg.permissions,
            "theme": cfg.theme,
            "ctx_window": cfg.ctx_window,
            "ctx_window_explicit": cfg.ctx_window_explicit,
            "max_output": cfg.max_output,
            /* Hard ceilings, published so the driver's non-TTY usage lines
             * can state the valid range. Served from Rust rather than
             * written into the driver: MAX_CTX_WINDOW has already moved once
             * (1M -> 4 MiB) and the /help copy drifted with it, so a second
             * copy in JS would drift the same way. */
            "ctx_cap": MAX_CTX_WINDOW,
            "max_output_cap": MAX_OUTPUT_TOKENS,
            "embed_provider": cfg.embed_provider,
            "embed_model": cfg.embed_model,
            "memory_backend": cfg.memory_backend,
            "brain_path": cfg.brain_path,
            "pricing": cfg.pricing,
            "budget_usd": cfg.budget_usd,
            "spend_total_usd": cfg.spend_total_usd,
            "ghost": cfg.ghost,
            "rss_warn_mb": cfg.rss_warn_mb,
            "recall_min": cfg.recall_min,
            "recall_budget": cfg.recall_budget,
            "threads": cfg.threads,
            /* save() persists no_think_models but this serializer dropped it —
             * a JS-side cfg reload never saw the runtime thinking rejection,
             * so the THINK "no-effort retry" re-sent reasoning_effort and
             * hit the same 400 (every later turn repeated it). */
            "no_think_models": cfg.no_think_models,
            "archive_enabled": cfg.archive_enabled,
            "archive_retain_final": cfg.archive_retain_final,
            "archive_retain_tool_results": cfg.archive_retain_tool_results,
            "archive_retain_partial": cfg.archive_retain_partial,
            "archive_redact": cfg.archive_redact,
            "archive_max_bytes": cfg.archive_max_bytes,
            "archive_max_count": cfg.archive_max_count,
            "archive_max_age_secs": cfg.archive_max_age_secs,
            "archive_auto_recover": cfg.archive_auto_recover,
            "archive_recovery_budget_chars": cfg.archive_recovery_budget_chars,
        })
        .to_string()
    };
    js_new_string(ctx, &json)
}

unsafe extern "C" fn js_chat_exit_check(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let b = *EXIT_REQUESTED.lock().unwrap();
    js_new_bool(ctx, b)
}

// ── Provider wizard (called by the JS driver after /provider) ──────

/// Apply a provider setup decided by the interactive wizard:
/// __chat_apply_provider(name, endpoint, api_key, model, profile) → true.
/// Upserts into the providers registry (no longer overwrites a single slot).
unsafe extern "C" fn js_chat_apply_provider(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc >= 1 {
        let n = argc.min(5) as usize;
        let args: Vec<Option<String>> =
            (0..n).map(|i| js_to_string(ctx, *argv.add(i))).collect();
        if let Ok(mut guard) = CFG.lock() {
            let cfg = guard.get_or_insert_with(ChatConfig::defaults);
            let name = args.first().and_then(|x| x.as_ref()).cloned().unwrap_or_default();
            if name.is_empty() { return js_new_bool(ctx, false); }
            let endpoint = args.get(1).and_then(|x| x.as_ref()).cloned().unwrap_or_default();
            let api_key = args.get(2).and_then(|x| x.as_ref()).cloned().unwrap_or_default();
            let model = args.get(3).and_then(|x| x.as_ref()).cloned().unwrap_or_default();
            let raw_profile = args.get(4).and_then(|x| x.as_ref()).cloned().unwrap_or_default();
            let profile = if raw_profile == "anthropic" || raw_profile == "local" { raw_profile } else { String::new() };
            // Preserve existing model when wizard sent empty; otherwise use new.
            let existing = cfg.providers.iter().find(|p| p.name == name).cloned();
            let final_model = if model.is_empty() {
                existing.as_ref().map(|p| p.model.clone()).unwrap_or_default()
            } else { model };
            cfg.upsert_provider(ProviderEntry { name, endpoint, api_key, model: final_model, profile, models: existing.map(|p| p.models).unwrap_or_default() });
            cfg.save();
        }
    }
    js_new_bool(ctx, true)
}

/// `__chat_models_cache(name, json_array)` — remember a provider's models
/// (the last successful live listing) so the /model picker still offers them
/// when the endpoint is unreachable next time. Best-effort: unknown provider
/// or malformed JSON is a silent no-op.
unsafe extern "C" fn js_chat_models_cache(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc >= 2 {
        if let (Some(name), Some(list)) = (js_to_string(ctx, *argv), js_to_string(ctx, *argv.add(1))) {
            let name = name.trim().to_string();
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&list) {
                let mut models: Vec<String> = Vec::new();
                if let Some(arr) = parsed.as_array() {
                    for it in arr {
                        if let Some(s) = it.as_str() {
                            let s = s.trim();
                            if !s.is_empty() && !models.iter().any(|m| m == s) {
                                models.push(s.to_string());
                            }
                        }
                    }
                }
                if !name.is_empty() && !models.is_empty() {
                    if let Ok(mut guard) = CFG.lock() {
                        let cfg = guard.get_or_insert_with(ChatConfig::defaults);
                        if let Some(entry) = cfg.providers.iter_mut().find(|p| p.name == name) {
                            for m in models {
                                if !entry.models.contains(&m) { entry.models.push(m); }
                            }
                            while entry.models.len() > 50 { entry.models.remove(0); }
                            cfg.save();
                        }
                    }
                }
            }
        }
    }
    js_new_bool(ctx, true)
}

/// Runtime thinking-support detection: mark the model as "cannot think"
/// after the provider rejected a reasoning parameter for it. Persisted so
/// the picker reports it and turns skip effort for this model.
/// (The shipped chat.js driver persists the same field through
/// rt/session_js.rs's `__chat_note_no_think` — P2-4.)
unsafe extern "C" fn js_chat_no_think(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 { return js_new_bool(ctx, false); }
    let Some(model) = js_to_string(ctx, *argv) else { return js_new_bool(ctx, false); };
    if model.is_empty() { return js_new_bool(ctx, false); }
    if let Ok(mut guard) = CFG.lock() {
        let cfg = guard.get_or_insert_with(ChatConfig::defaults);
        if !cfg.no_think_models.iter().any(|m| m == &model) {
            cfg.no_think_models.push(model);
            cfg.save();
        }
        return js_new_bool(ctx, true);
    }
    js_new_bool(ctx, false)
}

unsafe extern "C" fn js_chat_select_provider(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 { return js_new_bool(ctx, false); }
    let Some(name) = js_to_string(ctx, *argv) else { return js_new_bool(ctx, false); };
    if name.is_empty() { return js_new_bool(ctx, false); }
    let ok = if let Ok(mut guard) = CFG.lock() {
        let cfg = guard.get_or_insert_with(ChatConfig::defaults);
        if cfg.providers.iter().any(|p| p.name == name) {
            cfg.active = name.clone();
            cfg.sync_flat_from_active();
            cfg.save();
            true
        } else { false }
    } else { false };
    js_new_bool(ctx, ok)
}

unsafe extern "C" fn js_chat_remove_provider(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 { return js_new_bool(ctx, false); }
    let Some(name) = js_to_string(ctx, *argv) else { return js_new_bool(ctx, false); };
    let ok = if let Ok(mut guard) = CFG.lock() {
        let cfg = guard.get_or_insert_with(ChatConfig::defaults);
        let before = cfg.providers.len();
        cfg.providers.retain(|p| p.name != name);
        let removed = cfg.providers.len() != before;
        if removed {
            if cfg.active == name {
                cfg.active = cfg.providers.first().map(|p| p.name.clone()).unwrap_or_default();
            }
            if cfg.active.is_empty() {
                cfg.provider.clear(); cfg.model.clear(); cfg.base_url.clear(); cfg.api_key.clear(); cfg.profile.clear();
            } else {
                cfg.sync_flat_from_active();
            }
            cfg.save();
        }
        removed
    } else { false };
    js_new_bool(ctx, ok)
}

// ── Slash-command completion (called by the C readline on TAB) ─────

unsafe extern "C" fn js_chat_complete(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    // Returns "\n"-separated matches; empty string = nothing matches.
    let mut out = String::new();
    if argc >= 1 {
        if let Some(prefix) = js_to_string(ctx, *argv) {
            for m in complete_matches(&prefix) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(m);
            }
        }
    }
    js_new_string(ctx, &out)
}

/// `__chat_command_info("/cmd")` → JSON `{"desc": "...", "opts": "..."}`
/// (empty strings when unknown). The TUI completion menu shows this next
/// to the highlighted command so users see what each option does.
unsafe extern "C" fn js_chat_command_info(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut desc = "";
    let mut opts = "";
    if argc >= 1 {
        if let Some(cmd) = js_to_string(ctx, *argv) {
            if let Some((d, o)) = command_info(cmd.trim()) {
                desc = d;
                opts = o;
            }
        }
    }
    let json = serde_json::json!({ "desc": desc, "opts": opts }).to_string();
    js_new_string(ctx, &json)
}

// ── Past turns (resume a transcript) ──────────────────────────────

/// Returns the last prompt→answer turns of THIS session as a JSON array:
/// `[{"prompt": "...", "answer": "..."}]` (oldest first).
unsafe extern "C" fn js_chat_past_turns(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let json = {
        let s = SESS.lock().unwrap();
        let p = PROJECT.lock().unwrap();
        match (s.as_ref(), p.as_ref()) {
            (Some(sess), Some(project)) => {
                let turns = session::past_turns(project, sess.id());
                serde_json::json!(
                    turns
                        .iter()
                        .map(|(p, a)| serde_json::json!({ "prompt": p, "answer": a }))
                        .collect::<Vec<_>>()
                )
                .to_string()
            }
            _ => "[]".to_string(),
        }
    };
    js_new_string(ctx, &json)
}


// ── F1: /remember + /why — recall hits storage ──────────────────────

/// `__chat_set_recall(json)` — the JS driver stores recall hits from the
/// brain each turn so /why can display them. The JSON is an array of
/// {text, score, role}.
unsafe extern "C" fn js_chat_set_recall(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc >= 1 {
        if let Some(json_str) = js_to_string(ctx, *argv) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json_str) {
                if let Some(arr) = v.as_array() {
                    let hits: Vec<RecallHit> = arr.iter().filter_map(|item| {
                        Some(RecallHit {
                            text: item.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string(),
                            score: item.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0),
                            role: item.get("role").and_then(|r| r.as_str()).unwrap_or("").to_string(),
                            tier: item.get("tier").and_then(|t| t.as_u64()).unwrap_or(1) as u8,
                        })
                    }).collect();
                    *LAST_RECALL.lock().unwrap() = hits;
                }
            }
        }
    }
    js_new_bool(ctx, true)
}

/// `__chat_get_recall()` → JSON array of the last recall hits.
unsafe extern "C" fn js_chat_get_recall(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let json = {
        let hits = LAST_RECALL.lock().unwrap();
        serde_json::json!(*hits).to_string()
    };
    js_new_string(ctx, &json)
}

// ── F2: /resume — sessions list for the picker ───────────────────────

/// `__chat_project()` → the project root the session mesh is scoped to
/// ($SOFUU_PROJECT, else git top-level, else cwd), or "" when the mesh is
/// off. The /sessions browser titles itself with this so the list is
/// visibly scoped to the directory it belongs to.
unsafe extern "C" fn js_chat_project(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let path = PROJECT
        .lock()
        .unwrap()
        .clone()
        .map(|p| p.display().to_string());
    js_new_string(ctx, path.as_deref().unwrap_or(""))
}

/// `__chat_theme()` → JSON palette for the active TUI theme:
/// {name, dark, bg, accent, tool, delegate, heading, code, warn, error,
/// add, del, hunk, panel}. Unknown names fall back to the default (see
/// theme::lookup), so a bad config value can never unstyle the TUI.
/// Resolving ALSO activates the theme's window background (idempotent):
/// the driver loads this palette before __tui_on at boot and re-reads it
/// on every command, so boot and any out-of-band config change follow
/// without a separate sync call. Emits nothing when the TUI is inactive.
unsafe extern "C" fn js_chat_theme(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let name = CFG
        .lock()
        .unwrap()
        .as_ref()
        .map(|c| c.theme.clone())
        .unwrap_or_default();
    let t = theme::lookup(&name);
    unsafe {
        sofuu_core::rt::tui::tui_set_bg(t.bg as c_int);
        sofuu_core::rt::tui::tui_set_sel(
            theme::accent_index(t.accent).map(|v| v as c_int).unwrap_or(-1),
        );
    }
    let json = serde_json::json!({
        "name": t.name, "dark": t.dark, "bg": t.bg,
        "accent": t.accent, "tool": t.tool, "delegate": t.delegate,
        "heading": t.heading, "code": t.code, "warn": t.warn,
        "error": t.error, "add": t.add, "del": t.del,
        "hunk": t.hunk, "panel": t.panel,
    })
    .to_string();
    js_new_string(ctx, &json)
}

/// `__chat_themes()` → JSON array of all themes: [{name, dark}].
/// Drives the /theme picker and the non-TTY usage list.
unsafe extern "C" fn js_chat_themes(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let arr: Vec<serde_json::Value> = theme::THEMES
        .iter()
        .map(|t| serde_json::json!({ "name": t.name, "dark": t.dark }))
        .collect();
    js_new_string(ctx, &serde_json::json!(arr).to_string())
}

/// `__chat_sessions()` → JSON array of sessions from the registry, excluding
/// the current session. Each: {id, short, started_at, model, task, ended, turns}.
unsafe extern "C" fn js_chat_sessions(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let json = {
        let own = SESS.lock().unwrap().as_ref().map(|s| s.id().to_string());
        let project = PROJECT.lock().unwrap().clone();
        match project {
            Some(p) => {
                let reg = session::Registry::read(&p);
                let arr: Vec<serde_json::Value> = reg.sessions.iter()
                    .filter(|s| own.as_deref() != Some(&s.id))
                    .map(|s| {
                        let turns = session::past_turns(&p, &s.id).len();
                        serde_json::json!({
                            "id": s.id,
                            "short": session::short_id(&s.id),
                            "started_at": s.started_at,
                            "model": s.model,
                            "task": s.task,
                            "ended": s.ended,
                            "turns": turns,
                        })
                    }).collect();
                serde_json::json!(arr).to_string()
            }
            None => "[]".to_string(),
        }
    };
    js_new_string(ctx, &json)
}

/// `__chat_resume_turns(id)` → JSON array of turns from a past session.
unsafe extern "C" fn js_chat_resume_turns(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return js_new_string(ctx, "[]");
    }
    let Some(id) = js_to_string(ctx, *argv) else {
        return js_new_string(ctx, "[]");
    };
    let json = {
        let project = PROJECT.lock().unwrap().clone();
        match project {
            Some(p) => {
                /* /resume advertises the SHORT id in its picker (P1-10) —
                 * resolve short/ambiguous ids against the registry before
                 * loading, same as the session inspect/repair commands. */
                let id = session::resolve_session_id(&p, &id).unwrap_or_else(|e| {
                    eprintln!("\x1b[33m  {e}\x1b[0m");
                    id.clone()
                });
                let turns = session::past_turns(&p, &id);
                serde_json::json!(
                    turns.iter()
                        .map(|(p, a)| serde_json::json!({ "prompt": p, "answer": a }))
                        .collect::<Vec<_>>()
                ).to_string()
            }
            None => "[]".to_string(),
        }
    };
    js_new_string(ctx, &json)
}

// ── F6: /cost — session cost tracking ───────────────────────────────

/// `__chat_report_usage(model, promptTokens, completionTokens[, cacheRead, cacheWrite])`
/// → JSON {cost_usd, session_total, lifetime_total}.
/// Computes the cost from the pricing table and accumulates session + lifetime.
/// Cache token slots (P6.4) are optional and default to 0.
unsafe extern "C" fn js_chat_report_usage(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 3 {
        return js_new_string(ctx, r#"{"cost_usd":0,"session_total":0,"lifetime_total":0}"#);
    }
    let model = js_to_string(ctx, *argv).unwrap_or_default();
    /* Clamp at parse: a provider reporting negative/overflowing usage must
     * not walk lifetime spend backwards (P1-7) — the cache slots below were
     * already clamped; pt/ct were the gap. `pt as u64` on a negative i64
     * would also wrap into ~1.8e19 garbage in /cost. */
    let pt = js_to_int(ctx, *argv.add(1)).max(0);
    let ct = js_to_int(ctx, *argv.add(2)).max(0);
    let cache_read = if argc > 3 { js_to_int(ctx, *argv.add(3)).max(0) } else { 0 };
    let cache_write = if argc > 4 { js_to_int(ctx, *argv.add(4)).max(0) } else { 0 };
    let result = if let Ok(mut guard) = CFG.lock() {
        let cfg = guard.get_or_insert_with(ChatConfig::defaults);
        let key = format!("{}/{}", cfg.provider, model);
        let [in_p, out_p] = cfg.pricing.get(&key).cloned().unwrap_or([0.0, 0.0]);
        let cost = (pt as f64 / 1_000_000.0) * in_p + (ct as f64 / 1_000_000.0) * out_p;
        SESSION_COST.lock().unwrap().push_turn((model, pt as u64, ct as u64, cost, cache_read as u64, cache_write as u64));
        SESSION_COST.lock().unwrap().session_total += cost;
        SESSION_COST.lock().unwrap().cache_read_total += cache_read as u64;
        cfg.spend_total_usd += cost;
        /* P3 (AUDIT-2026-09-07): this hook runs EVERY turn and save()
         * rewrites the whole config.json (all provider API keys) each
         * time. Flush to disk only once per accrued cent — the in-memory
         * total is always exact and /cost reads it live. */
        let saved = f64::from_bits(SPEND_SAVED_BITS.load(std::sync::atomic::Ordering::Relaxed));
        if cfg.spend_total_usd - saved >= 0.01 {
            cfg.save();
        }
        let session_total = SESSION_COST.lock().unwrap().session_total;
        serde_json::json!({
            "cost_usd": (cost * 1e6).round() / 1e6,
            "session_total": (session_total * 1e6).round() / 1e6,
            "lifetime_total": (cfg.spend_total_usd * 1e6).round() / 1e6,
        }).to_string()
    } else {
        r#"{"cost_usd":0,"session_total":0,"lifetime_total":0}"#.to_string()
    };
    js_new_string(ctx, &result)
}

/// `__chat_cost_breakdown()` → JSON array of per-turn costs + totals.
unsafe extern "C" fn js_chat_cost_breakdown(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let json = {
        let sc = SESSION_COST.lock().unwrap();
        let turns: Vec<serde_json::Value> = sc.turns.iter().enumerate().map(|(i, (model, pt, ct, cost, cr, cw))| {
            serde_json::json!({
                "turn": i + 1,
                "model": model,
                "input_tokens": pt,
                "output_tokens": ct,
                "cost_usd": (cost * 1e6).round() / 1e6,
                "cache_read_tokens": cr,
                "cache_write_tokens": cw,
            })
        }).collect();
        let lifetime = CFG.lock().unwrap().as_ref().map(|c| c.spend_total_usd).unwrap_or(0.0);
        serde_json::json!({
            "turns": turns,
            "session_total": (sc.session_total * 1e6).round() / 1e6,
            "lifetime_total": (lifetime * 1e6).round() / 1e6,
            "cache_read_total": sc.cache_read_total,
        }).to_string()
    };
    js_new_string(ctx, &json)
}

/// `__chat_budget_check()` → JSON {ok, spent, budget, pct}.
unsafe extern "C" fn js_chat_budget_check(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let json = {
        let sc = SESSION_COST.lock().unwrap();
        let budget = CFG.lock().unwrap().as_ref().map(|c| c.budget_usd).unwrap_or(0.0);
        let spent = sc.session_total;
        let pct = if budget > 0.0 { (spent / budget * 100.0).round() as u64 } else { 0 };
        serde_json::json!({
            "ok": budget <= 0.0 || spent < budget,
            "spent": (spent * 1e6).round() / 1e6,
            "budget": budget,
            "pct": pct,
        }).to_string()
    };
    js_new_string(ctx, &json)
}

// ── F8: /watch — filesystem watcher bridge ──────────────────────────

/// `__chat_watch_add(path)` — add a path to the filesystem watcher.
unsafe extern "C" fn js_chat_watch_add(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc >= 1 {
        if let Some(path) = js_to_string(ctx, *argv) {
            FS_WATCHER.lock().unwrap().add(&path);
        }
    }
    js_new_bool(ctx, true)
}

/// `__chat_watch_clear()` — clear all watched paths.
unsafe extern "C" fn js_chat_watch_clear(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    FS_WATCHER.lock().unwrap().clear();
    js_new_bool(ctx, true)
}

/// `__chat_watch_list()` → JSON array of watched paths.
unsafe extern "C" fn js_chat_watch_list(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let json = serde_json::json!(FS_WATCHER.lock().unwrap().list()).to_string();
    js_new_string(ctx, &json)
}

/// `__chat_watch_poll()` → JSON array of {path, kind} changes since last poll.
unsafe extern "C" fn js_chat_watch_poll(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let changes: Vec<serde_json::Value> = FS_WATCHER.lock().unwrap().poll().into_iter()
        .map(|(path, kind)| serde_json::json!({ "path": path, "kind": kind }))
        .collect();
    js_new_string(ctx, &serde_json::json!(changes).to_string())
}

// ── F10: ghost completion ────────────────────────────────────────────

/// `__chat_ghost(prefix)` → a ghost completion suggestion string, or "".
/// Uses the bundled sync embedLocal (no network) + brain.recall filtered to
/// user-prompt memories.
unsafe extern "C" fn js_chat_ghost(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    // Ghost completion is implemented in the JS driver (it needs the brain
    // object + embedLocal which live in JS); this bridge just returns "" —
    // the driver overrides globalThis.__chat_ghost when brain is on.
    let _ = (ctx, argc, argv);
    js_new_string(ctx, "")
}

// ── Helper: read a JS number arg as i64 ──────────────────────────────

/// Read a JSValue as an i64 (falls back to 0). We convert via string since
/// the FFI only exposes sofuu_js_to_uint32 (no int64 helper).
unsafe fn js_to_int(ctx: *mut JSContext, val: JSValueConst) -> i64 {
    if let Some(s) = js_to_string(ctx, val) {
        let t = s.trim();
        if let Ok(n) = t.parse::<i64>() {
            return n;
        }
        /* P3 (AUDIT-2026-09-07): usage numbers can arrive as JS floats
         * ("1234.5"); the i64-only parse turned them into 0 — a silent
         * bill of zero. Fall back to an f64 parse truncated toward zero
         * (Rust's saturating float→int cast). */
        t.parse::<f64>().map(|f| f as i64).unwrap_or(0)
    } else {
        0
    }
}


/// Print through the TUI conversation area when the full-screen chat UI
/// is active; otherwise plain stdout (so piped/REPL output stays clean).
pub fn chat_out(s: &str) {
    if sofuu_ffi::tui_active() {
        sofuu_ffi::tui_log(s);
    } else {
        println!("{s}");
    }
}

/// `__chat_welcome()` — emit the welcome panel (TUI) or compact banner.
unsafe extern "C" fn js_chat_welcome(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    // Read config/session without nesting the locks (order rules apply to
    // SESS/WATCH/PROJECT; keep CFG out of any nesting as well).
    let cfg = CFG
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .unwrap_or_default();
    let sess = SESS
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|s| s.short_id().to_string()));
    let dir = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".into());

    if sofuu_ffi::tui_active() {
        let w = sofuu_ffi::tui_width();
        // The renderer indents every conversation row by GUTTER cells itself
        // (render_rows) and tui_push_split soft-wraps any stored row to
        // w − GUTTER, so the panel must be logged at w − GUTTER − 1 cells.
        // Pre-indenting here used to double the gutter: stored rows hit w
        // cells, every row exceeded the wrap budget by GUTTER, and wrap_row
        // tore each one apart — the right │ was pushed onto its own line
        // while the dash borders survived as clean fragments. The spare
        // cell keeps the border off the last column so the paint-time "…"
        // clip can never touch it either.
        let rows = welcome_panel_at(
            &cfg,
            sess.as_deref().unwrap_or(""),
            &dir,
            w.saturating_sub(sofuu_core::rt::tui::GUTTER + 1),
        );
        for row in &rows {
            sofuu_ffi::tui_log(row);
        }
    } else {
        for line in welcome_plain(&cfg, sess.as_deref(), &dir) {
            chat_out(&line);
        }
    }
    js_new_bool(ctx, true)
}

/// `__chat_refresh()` — clear the conversation area and re-log the welcome
/// panel from the CURRENT config. Called after the wizard or any slash
/// command that changes provider/model/effort/ctx/maxout so the header
/// reflects the new settings instantly (not only the footer).
unsafe extern "C" fn js_chat_refresh(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    if sofuu_ffi::tui_active() {
        sofuu_ffi::tui_reset();
        sofuu_ffi::tui_clear_gap();
    }
    js_chat_welcome(ctx, _this, 0, ptr::null());
    js_new_bool(ctx, true)
}

/// `__chat_phase(line)` — paint the agent-activity indicator ("Thinking",
/// "Coding", …) on the gap row directly above the input box. The JS driver
/// owns the label/animation math (see `phaseForStep`/the `PHASE_*` timer in
/// the embedded driver below) and passes the already-rendered ANSI line
/// here each tick; an empty string clears the row (turn ended/idle — same
/// row `tui_clear_gap()` blanks, so this never leaves a stale line behind).
unsafe extern "C" fn js_chat_phase(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if sofuu_ffi::tui_active() {
        let line = if argc >= 1 { js_to_string(ctx, *argv) } else { None };
        match line {
            Some(s) if !s.is_empty() => {
                sofuu_ffi::tui_overlay_row(sofuu_ffi::tui_phase_row(), &s);
            }
            _ => sofuu_ffi::tui_clear_gap(),
        }
    }
    js_new_bool(ctx, true)
}

/// Current resident set size (bytes) of this process — used for the
/// real-time RAM metric in the TUI footer. macOS: task_info; Linux: /proc.
fn rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: task_info fills the struct; returns KERN_SUCCESS (0).
        unsafe {
            let mut info: libc::mach_task_basic_info = std::mem::zeroed();
            let mut count = libc::MACH_TASK_BASIC_INFO_COUNT as libc::mach_msg_type_number_t;
            let kr = libc::task_info(
                libc::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                &mut info as *mut libc::mach_task_basic_info as *mut libc::integer_t,
                &mut count,
            );
            if kr == 0 {
                return info.resident_size as u64;
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(kb) = s.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()) {
                return kb * 4096; /* pages → bytes (page size 4k) */
            }
        }
    }
    0
}

/// `__chat_rss()` → resident memory of this process in bytes (JS number).
unsafe extern "C" fn js_chat_rss(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    sofuu_ffi::qjs::sofuu_js_new_int64(ctx, rss_bytes() as i64)
}

/// `__chat_gc()` — full QuickJS GC between turns (M2, PLAN-MEMORY-TOKENS).
/// A chat turn allocates streams/traces/history churn that refcounting
/// alone may leave as unreclaimed cycles; one full GC per completed turn
/// (~ms at chat heap sizes) bounds garbage to a single turn's worth
/// instead of letting it accumulate against the 512MB heap cap.
unsafe extern "C" fn js_chat_gc(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let rt = sofuu_ffi::qjs::JS_GetRuntime(ctx);
    if !rt.is_null() {
        sofuu_ffi::qjs::JS_RunGC(rt);
    }
    sofuu_ffi::qjs::sofuu_js_undefined()
}

/// Poll peers + refresh shared context. Returns a JSON string:
/// `{"notices":[{id,kind,text}...], "shared":"...", "me":"s-xxxx"}`.
unsafe extern "C" fn js_chat_poll(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let json = poll_json();
    js_new_string(ctx, &json)
}

fn poll_json() -> String {
    // Lock order: SESS → WATCH → PROJECT (never the reverse).
    let mut s = match SESS.lock() {
        Ok(g) => g,
        Err(_) => return r#"{"notices":[],"shared":""}"#.to_string(),
    };
    let mut w = match WATCH.lock() {
        Ok(g) => g,
        Err(_) => return r#"{"notices":[],"shared":""}"#.to_string(),
    };
    let p = match PROJECT.lock() {
        Ok(g) => g,
        Err(_) => return r#"{"notices":[],"shared":""}"#.to_string(),
    };
    let (Some(sess), Some(watch), Some(project)) = (s.as_mut(), w.as_mut(), p.as_ref()) else {
        return r#"{"notices":[],"shared":""}"#.to_string();
    };

    // Heartbeat: peers see "last seen" refresh on every user interaction.
    sess.heartbeat();

    let notices: Vec<serde_json::Value> = watch
        .poll(project, sess.id())
        .into_iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "kind": n.kind,
                "text": n.text,
            })
        })
        .collect();
    let shared = session::shared_context(project, sess.id());
    let me = sess.short_id();
    serde_json::json!({ "notices": notices, "shared": shared, "me": me }).to_string()
}

/// Log an event into THIS session's .qtsq: `__chat_log('prompt', text)`.
unsafe extern "C" fn js_chat_log(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc >= 2 {
        let kind = js_to_string(ctx, *argv);
        let text = js_to_string(ctx, *argv.add(1));
        if let (Some(k), Some(t)) = (kind, text) {
            if let Ok(mut guard) = SESS.lock() {
                if let Some(sess) = guard.as_mut() {
                    match k.as_str() {
                        "prompt" => {
                            sess.log_prompt(&t);
                            sess.heartbeat();
                        }
                        "answer" => sess.log_answer(&t),
                        "end" => sess.finish(),
                        _ => {}
                    }
                }
            }
        }
    }
    js_new_bool(ctx, true)
}

// ── Project-local output archive bridge ───────────────────────────────

fn current_archive_config() -> ChatConfig {
    CFG.lock()
        .ok()
        .and_then(|guard| guard.clone())
        .unwrap_or_else(ChatConfig::load)
}

fn current_archive_policy() -> output_archive::ArchivePolicy {
    current_archive_config().archive_policy()
}

fn current_archive_session_id() -> String {
    SESS.lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|session| session.id().to_string()))
        .unwrap_or_default()
}

fn current_archive_provider_model() -> (String, String) {
    let cfg = current_archive_config();
    (cfg.provider, cfg.model)
}

fn archive_write_request_json(raw: &str) -> Result<String, String> {
    let mut value: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| format!("invalid output write request: {e}"))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| "output write request must be a JSON object".to_string())?;
    let session_id = current_archive_session_id();
    let (provider, model) = current_archive_provider_model();
    if object.get("session_id").and_then(|value| value.as_str()).unwrap_or_default().is_empty() {
        object.insert("session_id".into(), serde_json::Value::String(session_id));
    }
    if object.get("provider").and_then(|value| value.as_str()).unwrap_or_default().is_empty() {
        object.insert("provider".into(), serde_json::Value::String(provider));
    }
    if object.get("model").and_then(|value| value.as_str()).unwrap_or_default().is_empty() {
        object.insert("model".into(), serde_json::Value::String(model));
    }
    Ok(value.to_string())
}

fn archive_project_or_empty() -> Option<PathBuf> {
    session::project_root()
}

unsafe extern "C" fn js_chat_archive_write(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let result = if argc < 1 {
        r#"{"ok":false,"error":"missing output write request"}"#.to_string()
    } else if let Some(raw) = js_to_string(ctx, *argv) {
        match archive_project_or_empty() {
            Some(project) => match archive_write_request_json(&raw) {
                Ok(request) => output_archive::write_json(&project, &request, &current_archive_policy()),
                Err(error) => serde_json::json!({ "ok": false, "error": error }).to_string(),
            },
            None => serde_json::json!({ "ok": true, "stored": false, "reason": "no project root" }).to_string(),
        }
    } else {
        r#"{"ok":false,"error":"output write request is not a string"}"#.to_string()
    };
    js_new_string(ctx, &result)
}

unsafe extern "C" fn js_chat_archive_list(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let result = match archive_project_or_empty() {
        Some(project) => {
            let raw = if argc >= 1 { js_to_string(ctx, *argv).unwrap_or_else(|| "{}".into()) } else { "{}".into() };
            match serde_json::from_str::<output_archive::OutputQuery>(&raw) {
                Ok(query) => output_archive::list_json(&project, query),
                Err(error) => serde_json::json!({ "ok": false, "error": format!("invalid output query: {error}") }).to_string(),
            }
        }
        None => serde_json::json!({ "ok": true, "items": [] }).to_string(),
    };
    js_new_string(ctx, &result)
}

unsafe extern "C" fn js_chat_archive_get(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let result = if argc < 1 {
        serde_json::json!({ "ok": false, "error": "missing output id" }).to_string()
    } else if let Some(id) = js_to_string(ctx, *argv) {
        match archive_project_or_empty() {
            Some(project) => output_archive::get_json(&project, id.trim()),
            None => serde_json::json!({ "ok": false, "error": "no project root" }).to_string(),
        }
    } else {
        serde_json::json!({ "ok": false, "error": "output id is not a string" }).to_string()
    };
    js_new_string(ctx, &result)
}

unsafe extern "C" fn js_chat_archive_search(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let result = match archive_project_or_empty() {
        Some(project) => {
            let raw = if argc >= 1 { js_to_string(ctx, *argv).unwrap_or_else(|| "{}".into()) } else { "{}".into() };
            match serde_json::from_str::<output_archive::OutputQuery>(&raw) {
                Ok(query) => output_archive::search_json(&project, query),
                Err(error) => serde_json::json!({ "ok": false, "error": format!("invalid output query: {error}") }).to_string(),
            }
        }
        None => serde_json::json!({ "ok": true, "items": [] }).to_string(),
    };
    js_new_string(ctx, &result)
}

unsafe extern "C" fn js_chat_archive_context(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let result = match archive_project_or_empty() {
        Some(project) => {
            let raw = if argc >= 1 { js_to_string(ctx, *argv).unwrap_or_else(|| "{}".into()) } else { "{}".into() };
            match serde_json::from_str::<output_archive::RecoveryQuery>(&raw) {
                Ok(query) => output_archive::context_json(&project, query),
                Err(error) => serde_json::json!({ "ok": false, "error": format!("invalid recovery query: {error}") }).to_string(),
            }
        }
        None => serde_json::json!({ "ok": true, "bundle": { "context": "", "items": [], "total_chars": 0, "total_bytes": 0, "omitted_count": 0, "truncated": false } }).to_string(),
    };
    js_new_string(ctx, &result)
}

unsafe extern "C" fn js_chat_archive_stats(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let result = match archive_project_or_empty() {
        Some(project) => output_archive::stats_json(&project, &current_archive_policy()),
        None => serde_json::json!({ "ok": true, "stats": { "enabled_root": false, "index_entries": 0 } }).to_string(),
    };
    js_new_string(ctx, &result)
}

unsafe extern "C" fn js_chat_archive_rebuild(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let result = match archive_project_or_empty() {
        Some(project) => output_archive::rebuild_index_json(&project),
        None => serde_json::json!({ "ok": true, "report": { "rebuilt": false, "indexed": 0, "corrupt": 0, "orphaned": 0, "temporary": 0 } }).to_string(),
    };
    js_new_string(ctx, &result)
}

unsafe extern "C" fn js_chat_archive_prune(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let result = match archive_project_or_empty() {
        Some(project) => {
            let raw = if argc >= 1 { js_to_string(ctx, *argv).unwrap_or_else(|| "{}".into()) } else { "{}".into() };
            let mut value: serde_json::Value = match serde_json::from_str(&raw) {
                Ok(value) => value,
                Err(error) => return js_new_string(ctx, &serde_json::json!({ "ok": false, "error": format!("invalid prune policy: {error}") }).to_string()),
            };
            if let Some(object) = value.as_object_mut() {
                object.entry("dry_run").or_insert(serde_json::Value::Bool(true));
                let defaults = current_archive_policy();
                if defaults.max_age_secs > 0 {
                    object.entry("older_than_secs").or_insert(serde_json::Value::from(defaults.max_age_secs));
                }
                if defaults.max_bytes > 0 {
                    object.entry("max_bytes").or_insert(serde_json::Value::from(defaults.max_bytes));
                }
                if defaults.max_count > 0 {
                    object.entry("max_count").or_insert(serde_json::Value::from(defaults.max_count));
                }
            }
            match serde_json::from_value::<output_archive::PrunePolicy>(value) {
                Ok(policy) => output_archive::prune_json(&project, policy),
                Err(error) => serde_json::json!({ "ok": false, "error": format!("invalid prune policy: {error}") }).to_string(),
            }
        }
        None => serde_json::json!({ "ok": true, "report": { "dry_run": true, "candidate_count": 0, "candidate_ids": [], "candidate_content_bytes": 0, "candidate_file_bytes": 0, "removed": 0 } }).to_string(),
    };
    js_new_string(ctx, &result)
}

/// `__chat_mcpservers()` → JSON string of the configured MCP servers from
/// ~/.sofuu/mcp.json: `[{"name":"fs","command":"..."}]` (empty → `[]`).
unsafe extern "C" fn js_chat_mcpservers(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let servers = ChatConfig::mcp_servers();
    let json = serde_json::json!(
        servers
            .iter()
            .map(|s| serde_json::json!({ "name": s.name, "command": s.command }))
            .collect::<Vec<_>>()
    )
    .to_string();
    js_new_string(ctx, &json)
}

pub(crate) fn register_bridge(rt: &SofuuRuntime) {
    let ctx = rt.engine_ctx() as *mut JSContext;
    if ctx.is_null() {
        return;
    }
    // SAFETY: ctx is the valid QuickJS context from the C engine.
    unsafe {
        register_global_fn(ctx, "__chat_slash", js_chat_slash as JSCFunction);
        register_global_fn(ctx, "__chat_getcfg", js_chat_getcfg as JSCFunction);
        register_global_fn(ctx, "__chat_exit_check", js_chat_exit_check as JSCFunction);
        register_global_fn(ctx, "__chat_poll", js_chat_poll as JSCFunction);
        register_global_fn(ctx, "__chat_log", js_chat_log as JSCFunction);
        register_global_fn(ctx, "__chat_archive_write", js_chat_archive_write as JSCFunction);
        register_global_fn(ctx, "__chat_archive_list", js_chat_archive_list as JSCFunction);
        register_global_fn(ctx, "__chat_archive_get", js_chat_archive_get as JSCFunction);
        register_global_fn(ctx, "__chat_archive_search", js_chat_archive_search as JSCFunction);
        register_global_fn(ctx, "__chat_archive_context", js_chat_archive_context as JSCFunction);
        register_global_fn(ctx, "__chat_archive_stats", js_chat_archive_stats as JSCFunction);
        register_global_fn(ctx, "__chat_archive_rebuild", js_chat_archive_rebuild as JSCFunction);
        register_global_fn(ctx, "__chat_archive_prune", js_chat_archive_prune as JSCFunction);
        register_global_fn(ctx, "__chat_complete", js_chat_complete as JSCFunction);
        register_global_fn(ctx, "__chat_command_info", js_chat_command_info as JSCFunction);
        register_global_fn(ctx, "__chat_apply_provider", js_chat_apply_provider as JSCFunction);
        register_global_fn(ctx, "__chat_models_cache", js_chat_models_cache as JSCFunction);
        register_global_fn(ctx, "__chat_select_provider", js_chat_select_provider as JSCFunction);
        register_global_fn(ctx, "__chat_no_think", js_chat_no_think as JSCFunction);
        register_global_fn(ctx, "__chat_remove_provider", js_chat_remove_provider as JSCFunction);
        register_global_fn(ctx, "__chat_past_turns", js_chat_past_turns as JSCFunction);
        register_global_fn(ctx, "__chat_welcome", js_chat_welcome as JSCFunction);
        register_global_fn(ctx, "__chat_phase", js_chat_phase as JSCFunction);
        register_global_fn(ctx, "__chat_rss", js_chat_rss as JSCFunction);
        register_global_fn(ctx, "__chat_gc", js_chat_gc as JSCFunction);
        register_global_fn(ctx, "__chat_refresh", js_chat_refresh as JSCFunction);
        register_global_fn(ctx, "__chat_mcpservers", js_chat_mcpservers as JSCFunction);
        // F1–F11 bridge functions
        register_global_fn(ctx, "__chat_set_recall", js_chat_set_recall as JSCFunction);
        register_global_fn(ctx, "__chat_get_recall", js_chat_get_recall as JSCFunction);
        register_global_fn(ctx, "__chat_sessions", js_chat_sessions as JSCFunction);
        register_global_fn(ctx, "__chat_project", js_chat_project as JSCFunction);
        register_global_fn(ctx, "__chat_theme", js_chat_theme as JSCFunction);
        register_global_fn(ctx, "__chat_themes", js_chat_themes as JSCFunction);
        register_global_fn(ctx, "__chat_resume_turns", js_chat_resume_turns as JSCFunction);
        register_global_fn(ctx, "__chat_report_usage", js_chat_report_usage as JSCFunction);
        register_global_fn(ctx, "__chat_cost_breakdown", js_chat_cost_breakdown as JSCFunction);
        register_global_fn(ctx, "__chat_budget_check", js_chat_budget_check as JSCFunction);
        register_global_fn(ctx, "__chat_watch_add", js_chat_watch_add as JSCFunction);
        register_global_fn(ctx, "__chat_watch_clear", js_chat_watch_clear as JSCFunction);
        register_global_fn(ctx, "__chat_watch_list", js_chat_watch_list as JSCFunction);
        register_global_fn(ctx, "__chat_watch_poll", js_chat_watch_poll as JSCFunction);
        register_global_fn(ctx, "__chat_ghost", js_chat_ghost as JSCFunction);
    }
}

// ── JS driver (streaming + input; calls back into Rust for commands) ──

// AUDIT-2026-10-31 P1-8: the TUI driver text lives in src/js/chat_tui.js and
// is embedded gzipped (198,338 B -> ~59 KB) by build.rs. It was the second
// largest .rodata item in the binary, for text inflated once, here, on the
// chat-only path. The .js file stays the single source of truth.
const DRIVER_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/chat_tui.js.gz"));
static DRIVER: LazyLock<String> = LazyLock::new(|| {
    sofuu_core::shipped::inflate_gz(DRIVER_GZ, "src/js/chat_tui.js").unwrap_or_else(|| {
        // Loud, not silent: the chat UI cannot run without its driver.
        String::from(
            "throw new Error(\"sofuu chat UI unavailable: embedded chat_tui.js failed to inflate\");\n",
        )
    })
});

/// Run the chat UI. `initial` is the loaded config (with CLI overrides applied).
pub fn run_chat(rt: &SofuuRuntime, initial: ChatConfig) -> i32 {
    *CFG.lock().unwrap() = Some(initial);
    *EXIT_REQUESTED.lock().unwrap() = false;

    // ── Session mesh: join the project's session registry (same project =
    // same mesh). Sessions sync tasks/notes/criticals in near-real time and
    // persist their data as .qtsq files under <project>/.sofuu/sessions/.
    let sync = CFG
        .lock()
        .unwrap()
        .as_ref()
        .map(|c| c.sync)
        .unwrap_or(false);
    if sync {
        match session::project_root() {
            Some(project) => {
                let (model, provider) = {
                    let guard = CFG.lock().unwrap();
                    let c = guard.as_ref().unwrap();
                    (c.model.clone(), c.provider.clone())
                };
                let sess = session::Session::join(&project, &model, &provider);
                chat_out(&format!(
                    "\x1b[90m  [sync] session {} registered for {}\x1b[0m",
                    sess.short_id(),
                    project.display()
                ));
                *SESS.lock().unwrap() = Some(sess);
                *WATCH.lock().unwrap() = Some(session::PeerWatch::new());
                *PROJECT.lock().unwrap() = Some(project);
            }
            None => {
                *SESS.lock().unwrap() = None;
                *WATCH.lock().unwrap() = None;
                *PROJECT.lock().unwrap() = None;
                chat_out(&format!("\x1b[90m  [sync] no project root found — mesh disabled\x1b[0m"));
            }
        }
    } else {
        *SESS.lock().unwrap() = None;
        *WATCH.lock().unwrap() = None;
        *PROJECT.lock().unwrap() = None;
    }

    register_bridge(rt);
    rt.eval_string(&DRIVER, "<chat>")
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The help screen's command column must line up on every row.
    ///
    /// It used to be a fixed 20 cells, so the first command that outgrew it
    /// pushed its em-dash out and left the whole block ragged — `/mode
    /// [full|edit|plan]` is 21 cells and `/ml [...]` is 53. Both shipped
    /// that way because nothing checked the rendered output.
    #[test]
    fn help_column_is_sized_to_the_longest_label() {
        // The two real offenders, plus a short one.
        let labels = ["/help", "/mode [full|edit|plan]", "/ml [on|off|learn|adopt|discard|reset|wrong|wasted|info]"];
        let widths: Vec<usize> = labels.iter().map(|s| char_cells_str(s)).collect();
        assert_eq!(widths, vec![5, 22, 56], "these lengths are what broke alignment");
        // The regression in one line: both of these outgrew the old fixed 20.
        assert!(widths[1] > 20 && widths[2] > 20);

        // Clamped to the cap, so an 80-col terminal keeps a description column.
        assert_eq!(help_command_col(widths.iter().copied()), 24);
        // A block of short commands still gets the readable minimum.
        assert_eq!(help_command_col([5usize, 12, 18].into_iter()), 20);
        // Empty input must not panic and must fall back to the minimum.
        assert_eq!(help_command_col(std::iter::empty()), 20);
    }

    /// Wide glyphs are measured, not assumed to be one cell — the help has
    /// both ("↑ / ↓" in Shortcuts, "…" in a few descriptions), and byte
    /// length would be badly wrong for either.
    #[test]
    fn help_column_measures_wide_glyphs() {
        // CJK is unambiguously 2 cells per char in any terminal.
        assert_eq!(char_cells_str("日本語"), 6);
        // Multi-byte but single-cell: measured by cells, not bytes.
        assert_eq!(char_cells_str("…"), 1);
        assert!(char_cells_str("…") < "…".len(), "must not count UTF-8 bytes");
        // ASCII baseline.
        assert_eq!(char_cells_str("abc"), 3);
        assert_eq!(char_cells_str(""), 0);
        // Combining marks do not advance the cursor.
        assert_eq!(char_cells_str("e\u{0301}"), 1);
    }

    /// Help descriptions must wrap to the terminal with a hanging indent.
    ///
    /// Unwrapped, the longer rows ran to 94 cells and the terminal
    /// soft-wrapped them mid-sentence, which is what made the block look
    /// broken on a standard 80-column window.
    #[test]
    fn help_descriptions_wrap_within_budget() {
        let desc = "copy the mouse-selected text (drag over rows, then Ctrl-K)";
        let lines = wrap_help_desc(desc, 50);
        assert!(lines.len() > 1, "should have wrapped: {lines:?}");
        for l in &lines {
            assert!(cell_w(l) <= 50, "line over budget ({}): {l:?}", cell_w(l));
        }
        // No text is lost or duplicated across the wrap.
        assert_eq!(lines.join(" ").split_whitespace().collect::<Vec<_>>(),
                   desc.split_whitespace().collect::<Vec<_>>());

        // A single word longer than the budget is hard-split, never allowed
        // to overflow (a long path or URL must not break the column).
        let long = "a".repeat(120);
        for l in wrap_help_desc(&long, 30) {
            assert!(cell_w(&l) <= 30, "hard-split failed: {} cells", cell_w(&l));
        }
        assert_eq!(wrap_help_desc(&long, 30).concat(), long, "hard-split lost bytes");

        // Short text is a single line; empty input must not panic or emit
        // an empty trailing row.
        assert_eq!(wrap_help_desc("short", 50), vec!["short".to_string()]);
        assert_eq!(wrap_help_desc("", 50).len(), 1);

        // Widths are measured in cells, not bytes.
        let cjk = "日本語のテキストが折り返されるべきです";
        for l in wrap_help_desc(cjk, 20) {
            assert!(cell_w(&l) <= 20, "CJK line over budget ({}): {l:?}", cell_w(&l));
        }
    }

    /// The hard ceilings must reach the driver, so its non-TTY usage lines
    /// can state the real range instead of leaving the user to guess.
    ///
    /// They are served from Rust on purpose: MAX_CTX_WINDOW has already
    /// moved once (1M -> 4 MiB) and the copy drifted with it. A literal in
    /// the driver JS would be a third place to forget.
    #[test]
    fn getcfg_publishes_the_ceiling_constants() {
        let guard = CFG.lock().unwrap();
        let default_cfg;
        let cfg: &ChatConfig = match guard.as_ref() {
            Some(c) => c,
            None => {
                default_cfg = ChatConfig::defaults();
                &default_cfg
            }
        };
        let caps = serde_json::json!({
            "ctx_cap": MAX_CTX_WINDOW,
            "max_output_cap": MAX_OUTPUT_TOKENS,
            "ctx_window": cfg.ctx_window,
        });
        assert_eq!(caps["ctx_cap"], 4_194_304);
        assert_eq!(caps["max_output_cap"], 384_000);
        // The driver reads these exact key names; renaming one silently
        // turns "(0-undefined)" into the usage line.
        assert_eq!(caps["ctx_window"], cfg.ctx_window);
    }

    /// The help text must not advertise a ceiling the code does not
    /// enforce. MAX_CTX_WINDOW is 4 MiB, but the copy still advertised the
    /// old one-megabyte ceiling in three places — the same stale number the
    /// /ctx clamp test used to assert — so /help told users half the real
    /// limit. (The old figure is spelled via format!() below rather than
    /// written out, so this guard cannot match its own source.)
    ///
    /// Checks the literals the UI actually prints, not a blanket grep: a
    /// grep also matches this test's own doc comment, which is exactly the
    /// kind of thing that makes a guard quietly useless.
    #[test]
    fn documented_ceilings_match_the_enforced_ones() {
        assert_eq!(MAX_CTX_WINDOW, 4_194_304);
        assert_eq!(MAX_OUTPUT_TOKENS, 384_000);
        // Every /ctx and /maxout help + usage row, spelled out.
        let rows: [&str; 4] = [
            "view/set the context window (max 4M)",
            "view/set max output tokens (max 384k)",
            "<tokens> | default  (default = model's real window, max 4M)",
            "Sofuu chat --ctx-window <n> Context window in tokens (0 = provider default, max 4M)",
        ];
        let chat = include_str!("chat.rs");
        let main = include_str!("main.rs");
        for row in rows {
            assert!(
                chat.contains(row) || main.contains(row),
                "expected ceiling text missing: {row}"
            );
        }
        // And no row still claims the old ceiling. The needle is built at
        // runtime from fragments: written literally it would appear in this
        // file and match itself, which is how the first version of this test
        // failed against its own assertion message.
        let stale = format!("max {}M", 1);
        assert!(
            !chat.contains(&stale) && !main.contains(&stale),
            "a /ctx surface still advertises the stale 1M ceiling"
        );
    }

    /// The embedded DRIVER is JavaScript, not Rust. Pasting a Rust
    /// statement into it (`let mut x = …`) compiles fine here — the whole
    /// driver is one `r#"…"#` string literal — and then fails at RUNTIME
    /// with "SyntaxError: expecting ';'" in the middle of a chat turn,
    /// which is a miserable way to find out. This keeps the obvious
    /// Rust-only constructs out of the JS body.
    #[test]
    fn driver_js_contains_no_rust_isms() {
        // The driver text comes from src/js/chat_tui.js (AUDIT-2026-10-31
        // P1-8 embeds it gzipped), so the value starts with the IIFE
        // directly — it carries no Rust raw-string markers.
        let js = &*DRIVER;
        let start = js.find("(function").unwrap_or(0);
        for (i, line) in js[start..].lines().enumerate() {
            let t = line.trim_start();
            assert!(
                !t.starts_with("let mut "),
                "driver line {}: Rust `let mut` inside the JS driver: {}",
                i + 1,
                t
            );
            assert!(
                !t.starts_with("impl ") && !t.contains(" -> bool {") && !t.contains(" -> f32 {"),
                "driver line {}: Rust signature inside the JS driver: {}",
                i + 1,
                t
            );
        }
    }

    /// The compaction guard must exist in BOTH drivers. They are separate
    /// copies of the same logic, and a fix applied to only one ships as a
    /// bug in whichever binary the user happens to run.
    #[test]
    fn compaction_guard_present_in_both_drivers() {
        assert!(
            DRIVER.contains("function isProtectedBlock"),
            "chat.rs driver: protected-turn guard missing"
        );
        assert!(DRIVER.contains("maxDrops"), "chat.rs driver: per-pass cap missing");
        let shipped = include_str!("../../../src/js/chat.js");
        assert!(
            shipped.contains("function isProtectedBlock"),
            "chat.js: protected-turn guard missing"
        );
        assert!(shipped.contains("maxDrops"), "chat.js: per-pass cap missing");
    }

    /// The tool-result painter must keep three branches that each fix a
    /// real invisibility: the unified-diff painter for edit_file /
    /// write_file (WHAT changed, not just its byte count), the red error
    /// line (tool errors used to print nothing at all — the error emit
    /// carries no `result`, so the old branch skipped it and a failed
    /// edit died silently on screen), and the compact one-row summary for
    /// everything else. Presence-pinned like the compaction guard above:
    /// the per-kind color scheme degrades silently if any branch rots.
    #[test]
    fn tool_result_painter_keeps_all_three_branches() {
        assert!(
            DRIVER.contains("function paintDiff"),
            "chat.rs driver: diff painter missing — edits render byte counts only"
        );
        assert!(
            DRIVER.contains("p.diff"),
            "chat.rs driver: tool_result ignores the event diff payload"
        );
        assert!(
            DRIVER.contains("T('error')"),
            "chat.rs driver: tool-error line must use the theme error role — failures go silent"
        );
        assert!(
            DRIVER.contains("T('add')") && DRIVER.contains("T('del')"),
            "chat.rs driver: diff +/- rows must use theme roles (tints live in theme.rs, not the driver)"
        );
    }

    /// Transcript hierarchy, rhythm, checklist: the driver must keep the
    /// markdown formatter (headings/bold/code on the final paint), the
    /// inline-span tokenizer, the checklist painter, the blank-row
    /// separator before tool groups, and the markdown call on both final
    /// answer paints. Presence-pinned: each degrades silently (raw
    /// markers on screen / tools glued to prose / checklist invisible).
    #[test]
    fn transcript_hierarchy_rhythm_checklist_present() {
        assert!(
            DRIVER.contains("function formatMarkdown"),
            "driver: markdown hierarchy formatter missing — answers render as a wall"
        );
        assert!(
            DRIVER.contains("function fmtInline"),
            "driver: inline-span tokenizer missing — bold/code markers leak or mis-nest"
        );
        assert!(
            DRIVER.contains("function paintChecklist"),
            "driver: todo checklist painter missing — the list stays invisible"
        );
        assert!(
            DRIVER.contains("lastRowWasTool"),
            "driver: tool-group spacing flag missing — tools run into prose"
        );
        assert!(
            DRIVER.contains("outLast(formatMarkdown(answer))"),
            "driver: final answer paint must format markdown, not print raw"
        );
        assert!(
            DRIVER.contains("paintChecklist(p.todos)"),
            "driver: todo_write results must paint the list, not the one-row summary"
        );
    }

    /// Compile gate for the embedded JS: cargo builds the DRIVER string
    /// blind — a JS syntax slip ships a binary whose chat dies on boot
    /// with `SyntaxError` (this exact failure shipped: two stray braces
    /// after paintDiff, caught only by a manual pty probe). Parse DRIVER
    /// and the shipped agent/tools JS with COMPILE_ONLY so a typo fails
    /// `cargo test` in seconds instead of a session at runtime.
    #[test]
    fn embedded_js_drivers_parse() {
        use sofuu_ffi::qjs;
        use std::ffi::CString;
        fn check_parse(name: &str, src: &str, expect_ok: bool) {
            unsafe {
                let rt = qjs::JS_NewRuntime();
                assert!(!rt.is_null(), "{name}: no QuickJS runtime");
                let ctx = qjs::JS_NewContext(rt);
                assert!(!ctx.is_null(), "{name}: no QuickJS context");
                let c_src = CString::new(src).expect("driver has interior NUL");
                let c_name = CString::new(name).unwrap();
                let v = qjs::JS_Eval(
                    ctx,
                    c_src.as_ptr(),
                    c_src.as_bytes().len(),
                    c_name.as_ptr(),
                    qjs::JS_EVAL_TYPE_GLOBAL | qjs::JS_EVAL_FLAG_COMPILE_ONLY,
                );
                let failed = qjs::is_exception(v);
                qjs::sofuu_js_free_value(ctx, v);
                qjs::JS_FreeContext(ctx);
                qjs::JS_FreeRuntime(rt);
                if expect_ok {
                    assert!(!failed, "{name}: JS syntax error — chat would die on boot");
                } else {
                    /* Self-proof: the gate must actually reject broken JS,
                     * or a future refactor could neuter it (wrong flags,
                     * swallowed exception) and every check above would pass
                     * vacuously. */
                    assert!(failed, "{name}: gate accepted broken JS — it proves nothing");
                }
            }
        }
        check_parse("<negative-control>", "function broken( {", false);
        check_parse("<chat-driver>", &DRIVER, true);
        check_parse("agent.js", include_str!("../../../src/js/agent.js"), true);
        check_parse("tools.js", include_str!("../../../src/js/tools.js"), true);
        check_parse("extractor.js", include_str!("../../../src/js/extractor.js"), true);
        check_parse("storage.js", include_str!("../../../src/js/storage.js"), true);
        check_parse("chat.js", include_str!("../../../src/js/chat.js"), true);
        check_parse("memory_api.js", include_str!("../../../src/js/memory_api.js"), true);
    }

    /// Settings changes must never wipe a lived-in transcript: every
    /// panel refresh goes through maybeRefreshPanel, which only fires
    /// pre-first-turn (TTY-only — piped, a refresh appends a duplicate).
    /// A mode/theme/model switch mid-conversation used to tui_reset() the
    /// scrollback and look like a fresh session. Presence-pinned plus a
    /// call-count: exactly one direct __chat_refresh() may exist (inside
    /// the helper), so no settings path can wipe around it.
    #[test]
    fn settings_changes_keep_a_lived_in_transcript() {
        assert!(
            DRIVER.contains("function maybeRefreshPanel"),
            "driver: gated panel refresh missing"
        );
        assert!(
            DRIVER.contains("history.length === 0"),
            "driver: refresh gate must key on lived-in history"
        );
        assert_eq!(
            DRIVER.matches("__chat_refresh();").count(),
            1,
            "exactly one direct refresh call may exist (inside the gate)"
        );
    }

    /// Theme plumbing: the driver must resolve colors through T(role)
    /// (backed by __chat_theme, cached, reset on every command so a
    /// switch recolors the next paint), offer pickTheme over
    /// __chat_themes, and fall back to the classic look when the bridge
    /// is missing — never unstyled.
    #[test]
    fn theme_plumbing_present() {
        assert!(
            DRIVER.contains("function T(role)"),
            "driver: T(role) palette resolver missing"
        );
        assert!(
            DRIVER.contains("__chat_theme()") && DRIVER.contains("__chat_themes()"),
            "driver: theme bridges unwired"
        );
        assert!(
            DRIVER.contains("async function pickTheme()"),
            "driver: theme picker missing"
        );
        assert!(
            DRIVER.contains("THEME_FALLBACK"),
            "driver: classic-color fallback missing"
        );
        for role in ["T('accent')", "T('tool')", "T('heading')", "T('error')"] {
            assert!(
                DRIVER.contains(role),
                "driver: transcript must actually use {role}"
            );
        }
    }

    /// Resume distills instead of dumping: handleResume must route old
    /// turns through summarizeHistory with the resume framing (not the
    /// stock compaction prompt), label the archive record 'resume' (not
    /// 'compaction'), and point at /context for the full transcript.
    /// Presence-pinned: without this branch a resume silently loads the
    /// whole past into context and prints all of it.
    #[test]
    fn resume_distills_through_the_active_model() {
        assert!(
            DRIVER.contains("summarizeHistory(0, {"),
            "handleResume must distill via summarizeHistory instead of dumping turns"
        );
        assert!(
            DRIVER.contains("source: 'resume'"),
            "resume archive record must be sourced 'resume', not 'compaction'"
        );
        assert!(
            DRIVER.contains("full transcript: /context"),
            "resume must point at /context for the full transcript"
        );
        assert!(
            DRIVER.contains("summary unavailable"),
            "resume must say why it falls back to the full transcript"
        );
    }

    /// Transcript typography: body text renders at full weight, thinking
    /// in italic, signal lines bold. The transcript used to be almost
    /// entirely dim, which read as a whisper next to the white answer
    /// text; terminals have no percentages, so the scheme is: markers dim,
    /// content normal, signal (diff +/-, errors) bold. Italic falls back
    /// to normal on terminals without italic support — still readable.
    #[test]
    fn transcript_typography_weights() {
        assert!(
            DRIVER.contains("▸ thinking \\x1b[3m"),
            "thinking must render italic, not dim"
        );
        assert!(
            DRIVER.contains("↳\\x1b[0m "),
            "tool-result ↳ marker must close before the content so the content renders at full weight"
        );
        assert!(
            DRIVER.contains("T('add')") && DRIVER.contains("T('del')") && DRIVER.contains("T('error')"),
            "signal lines must use theme roles, not hardcoded escapes"
        );
    }

    /// The lexical floor is the last line of defence for the one
    /// irreversible action in the chat, so the pattern itself is pinned:
    /// loosening it would make a question or an approval deletable on the
    /// model's word alone.
    #[test]
    fn protect_pattern_covers_the_documented_cases() {
        let src = include_str!("../../../src/js/chat.js");
        let at = src.find("PROTECT_RE = ").expect("protect pattern");
        // `find` on the `src[at..]` sub-slice returns an offset RELATIVE to
        // `at`, but the old code used it as an absolute end index — so it
        // sliced src[16693..169] and panicked on every run with
        // "begin <= end". Convert the offset before slicing.
        let line_end = src[at..].find('\n').map(|off| at + off).unwrap_or(src.len());
        let line = &src[at..line_end];
        for needed in ["do\\s+not", "never", "always", "must", "approved", "decided", "\\?"] {
            assert!(
                line.contains(needed),
                "protect pattern lost {:?}: {}",
                needed,
                line
            );
        }
    }

    #[test]
    fn defaults_are_unconfigured() {
        // "No default model" design rule: a fresh config names NO provider
        // and NO model (first-run wizard fills them in), and embeddings
        // default to the bundled offline embedder (empty embed_* fields).
        let d = ChatConfig::defaults();
        assert!(d.provider.is_empty());
        assert!(d.model.is_empty());
        assert!(d.embed_provider.is_empty());
        assert!(d.embed_model.is_empty());
        // Default thinking effort is high (a fresh config thinks hard).
        assert_eq!(d.effort, "high");
        // PLAN-ML-GATES: the ML context-economy gates default to ON.
        assert!(d.ml);
    }

    #[test]
    fn config_roundtrip() {
        // Use a unique temp dir (thread-unique) so parallel tests don't race
        // on the same file, and we never touch the real ~/.sofuu/config.json.
        let tmp = std::env::temp_dir().join(format!(
            "sofuu-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::env::set_var("SOFUU_TEST_CONFIG_DIR", &tmp);
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        // Write the file directly (save() is a no-op in tests).
        std::fs::write(
            tmp.join("config.json"),
            "{\n  \"provider\": \"openai\",\n  \"model\": \"gpt-4o\",\n  \"effort\": \"high\",\n  \"brain\": true,\n  \"ml\": false,\n  \"ctx_window\": 1000000,\n  \"max_output\": 384000,\n  \"permissions\": \"plan\"\n}\n",
        )
        .unwrap();

        let l = ChatConfig::load();
        assert_eq!(l.provider, "openai");
        assert_eq!(l.model, "gpt-4o");
        assert_eq!(l.effort, "high");
        assert!(l.brain);
        assert!(!l.ml, "ml flag round-trips from config.json");
        assert_eq!(l.ctx_window, 1_000_000);
        assert_eq!(l.max_output, 384_000);
        assert_eq!(l.permissions, "plan", "persisted permission mode loads");

        // A typo in config.json must never mean "more access": unknown
        // profiles fall back to the default, they never load.
        std::fs::write(
            tmp.join("config.json"),
            "{\n  \"provider\": \"openai\",\n  \"permissions\": \"sudo\"\n}\n",
        )
        .unwrap();
        assert_eq!(
            ChatConfig::load().permissions,
            "full",
            "unknown persisted profile must not raise access"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn slash_dispatch() {
        let mut c = ChatConfig::defaults();
        assert_eq!(handle_slash(&mut c, "/clear"), "clear");
        assert_eq!(handle_slash(&mut c, "/compact"), "compact");
        // Removed commands are unknown.
        assert_eq!(handle_slash(&mut c, "/models"), "unknown");
        assert_eq!(handle_slash(&mut c, "/apikey"), "unknown");
        assert_eq!(handle_slash(&mut c, "/baseurl"), "unknown");
        // Bare choice-commands open interactive pickers (the DRIVER renders
        // them); arg forms set directly.
        assert_eq!(handle_slash(&mut c, "/model"), "pick_model");
        assert_eq!(handle_slash(&mut c, "/provider"), "pick_provider");
        assert_eq!(handle_slash(&mut c, "/effort"), "pick_effort");
        // /providers is a hidden alias of /provider.
        assert_eq!(handle_slash(&mut c, "/providers"), "pick_provider");
        assert_eq!(handle_slash(&mut c, "/providers openai"), "ok");
        assert_eq!(c.provider, "openai");
        assert_eq!(handle_slash(&mut c, "/model gpt-4o"), "ok");
        assert_eq!(c.model, "gpt-4o");
        assert_eq!(handle_slash(&mut c, "/effort high"), "ok");
        assert_eq!(c.effort, "high");
        // /effort off clears to the provider default.
        assert_eq!(handle_slash(&mut c, "/effort off"), "ok");
        assert!(c.effort.is_empty());
        // Bare config commands open their picker panel (DRIVER handles it).
        assert_eq!(handle_slash(&mut c, "/brain"), "pick_brain");
        assert_eq!(handle_slash(&mut c, "/brain info"), "brain_info");
        assert_eq!(handle_slash(&mut c, "/brain on"), "ok");
        assert!(c.brain);
        // /ml: bare prints status (no picker); on/off persist.
        assert!(c.ml, "ml defaults on");
        assert_eq!(handle_slash(&mut c, "/ml"), "ok");
        assert_eq!(handle_slash(&mut c, "/ml off"), "ok");
        assert!(!c.ml);
        assert_eq!(handle_slash(&mut c, "/ml bogus"), "ok");
        assert!(!c.ml, "bad /ml arg must not change state");
        assert_eq!(handle_slash(&mut c, "/ml on"), "ok");
        assert!(c.ml);
        // /rlm: bare opens panel; on/auto/off persist (off stores "").
        assert_eq!(handle_slash(&mut c, "/rlm"), "pick_rlm");
        assert_eq!(handle_slash(&mut c, "/rlm auto"), "ok");
        assert_eq!(c.rlm, "auto");
        assert_eq!(handle_slash(&mut c, "/rlm bogus"), "ok");
        assert_eq!(c.rlm, "auto", "bad /rlm arg must not change state");
        assert_eq!(handle_slash(&mut c, "/rlm off"), "ok");
        assert!(c.rlm.is_empty());
        // Permission modes: /plan /edit /full shorthands; /mode <name> sets,
        // bare /mode shows; a bad arg never changes access (fail closed).
        assert_eq!(c.permissions, "full", "defaults to full access");
        assert_eq!(handle_slash(&mut c, "/plan"), "ok");
        assert_eq!(c.permissions, "plan");
        assert_eq!(handle_slash(&mut c, "/mode edit"), "ok");
        assert_eq!(c.permissions, "edit");
        assert_eq!(handle_slash(&mut c, "/mode"), "ok", "bare /mode just shows");
        assert_eq!(c.permissions, "edit", "bare /mode must not change the mode");
        assert_eq!(handle_slash(&mut c, "/mode sudo"), "ok");
        assert_eq!(c.permissions, "edit", "unknown mode must not change access");
        assert_eq!(handle_slash(&mut c, "/full"), "ok");
        assert_eq!(c.permissions, "full");
        // /ctx: bare opens panel; set, clamp, reset to default.
        assert_eq!(handle_slash(&mut c, "/ctx"), "pick_ctx");
        assert_eq!(handle_slash(&mut c, "/ctx 1000000"), "ok");
        assert_eq!(c.ctx_window, 1_000_000);
        assert_eq!(handle_slash(&mut c, "/ctx 9999999"), "ok", "over the 1M cap is rejected");
        assert_eq!(c.ctx_window, 1_000_000, "rejected /ctx must not change state");
        assert_eq!(handle_slash(&mut c, "/ctx default"), "ok");
        assert_eq!(c.ctx_window, 0);
        // /maxout: bare opens panel; set, clamp, reset.
        assert_eq!(handle_slash(&mut c, "/maxout"), "pick_maxout");
        assert_eq!(handle_slash(&mut c, "/maxout 384000"), "ok");
        assert_eq!(c.max_output, 384_000);
        assert_eq!(handle_slash(&mut c, "/maxout 9999999"), "ok", "over the 384k cap is rejected");
        assert_eq!(c.max_output, 384_000, "rejected /maxout must not change state");
        assert_eq!(handle_slash(&mut c, "/maxout default"), "ok");
        assert_eq!(c.max_output, 0);
        assert_eq!(handle_slash(&mut c, "/bogus"), "unknown");
        assert_eq!(handle_slash(&mut c, "/exit"), "exit");
    }

    #[test]
    fn slash_dispatch_f1_f11() {
        // F1–F11 new commands dispatch the right op tokens.
        let mut c = ChatConfig::defaults();
        // F1: /remember + /why
        assert_eq!(handle_slash(&mut c, "/remember"), "ok"); // empty → usage
        assert_eq!(handle_slash(&mut c, "/remember my fact"), "remember");
        assert_eq!(handle_slash(&mut c, "/why"), "why");
        // F2: /resume
        assert_eq!(handle_slash(&mut c, "/resume"), "resume");
        assert_eq!(handle_slash(&mut c, "/resume s-abc"), "resume");
        assert_eq!(handle_slash(&mut c, "/sessions"), "sessions");
        assert_eq!(handle_slash(&mut c, "/sessions s-abc"), "sessions");
        assert_eq!(handle_slash(&mut c, "/theme"), "pick_theme");
        assert_eq!(handle_slash(&mut c, "/theme slate"), "ok");
        assert_eq!(c.theme, "slate", "applying a known theme persists it on the config");
        assert_eq!(handle_slash(&mut c, "/theme SLATE"), "ok", "theme names match case-insensitively");
        assert_eq!(theme::lookup(&c.theme).name, "slate");
        assert_eq!(handle_slash(&mut c, "/theme no-such-theme"), "ok");
        assert_eq!(theme::lookup(&c.theme).name, "slate", "an unknown name must not clobber the active theme");
        // F3: /at (help)
        assert_eq!(handle_slash(&mut c, "/at"), "ok");
        // F5: /share + /import
        assert_eq!(handle_slash(&mut c, "/share"), "share");
        assert_eq!(handle_slash(&mut c, "/share card.qtsq my-label"), "share");
        assert_eq!(handle_slash(&mut c, "/import"), "ok"); // empty → usage
        assert_eq!(handle_slash(&mut c, "/import card.qtsq"), "import");
        // F6: /cost
        assert_eq!(handle_slash(&mut c, "/cost"), "cost");
        // F8: /watch
        assert_eq!(handle_slash(&mut c, "/watch"), "watch");
        assert_eq!(handle_slash(&mut c, "/watch /tmp"), "watch");
        // F9: /hooks
        assert_eq!(handle_slash(&mut c, "/hooks"), "ok");
        // F10: /ghost
        assert_eq!(handle_slash(&mut c, "/ghost"), "ok"); // bare → status
        assert_eq!(handle_slash(&mut c, "/ghost on"), "ok");
        assert!(c.ghost);
        assert_eq!(handle_slash(&mut c, "/ghost off"), "ok");
        assert!(!c.ghost);
        // F11: /serve (informational)
        assert_eq!(handle_slash(&mut c, "/serve"), "ok");
    }

    #[test]
    fn fs_watcher_basic() {
        let mut w = FsWatcher::new();
        assert!(w.list().is_empty());
        // Add a path (doesn't need to exist for add to work).
        w.add("/tmp/sofuu-watch-test-nonexistent");
        assert_eq!(w.list().len(), 1);
        // Adding the same path twice is a no-op.
        w.add("/tmp/sofuu-watch-test-nonexistent");
        assert_eq!(w.list().len(), 1);
        // Clear.
        w.clear();
        assert!(w.list().is_empty());
    }

    #[test]
    fn pricing_defaults_seeded() {
        let c = ChatConfig::defaults();
        // Seeded defaults exist for popular models.
        assert!(c.pricing.contains_key("openai/gpt-4o"));
        assert!(c.pricing.contains_key("anthropic/claude-sonnet-4-6"));
        // gpt-4o is $2.50/$10.00 per 1M in/out.
        let prices = c.pricing.get("openai/gpt-4o").unwrap();
        assert_eq!(*prices, [2.50, 10.00]);
    }

    #[test]
    fn ctx_window_effective_and_clamped() {
        // 0 = provider default.
        assert_eq!(effective_ctx_window("openai", 0), 128_000);
        assert_eq!(effective_ctx_window("anthropic", 0), 128_000);
        assert_eq!(effective_ctx_window("local", 0), 32_768);
        assert_eq!(effective_ctx_window("custom-xyz", 0), 32_768);
        // An explicit window wins over the default — up to the 1M cap.
        assert_eq!(effective_ctx_window("openai", 1_000_000), 1_000_000);
        assert_eq!(effective_ctx_window("local", 1_000_000), 1_000_000);
        // load() clamps corrupt/over-the-cap persisted values. The cap is
        // MAX_CTX_WINDOW (4 MiB) — it was raised from 1,000,000 because the
        // /ctx picker's own "1M" preset (1048576) was being REJECTED by the
        // old 1,000,000 ceiling. This assertion still expected the old 1M
        // and so failed on every run; it now pins the real cap by name.
        assert_eq!(
            clamp_ctx_window(999_999_999),
            MAX_CTX_WINDOW,
            "an absurd persisted window clamps to the real cap, not a stale one"
        );
        assert_eq!(clamp_ctx_window(-5), 0);
        assert_eq!(clamp_ctx_window(512), 1024, "below 1024 clamps up to the floor");
    }

    #[test]
    fn max_output_clamped() {
        // 0 = provider default.
        assert_eq!(clamp_max_output(0), 0);
        assert_eq!(clamp_max_output(-1), 0);
        // Valid values pass through, capped at 384k.
        assert_eq!(clamp_max_output(1), 1);
        assert_eq!(clamp_max_output(4096), 4096);
        assert_eq!(clamp_max_output(384_000), 384_000);
        // Over the cap clamps down (not rejected silently into 0).
        assert_eq!(clamp_max_output(999_999_999), 384_000);
    }

    /// The Memory row must never claim a persistence the binary cannot
    /// deliver: a QTSQ-free build no-ops every brain write (and that is
    /// exactly what shipped in the 2026-09-22 macOS tarball while the
    /// banner still read "persists across sessions").
    #[test]
    fn memory_row_never_lies_about_persistence() {
        let mut cfg = ChatConfig { brain: true, ..Default::default() };
        let rows = welcome_panel_at(&cfg, "s-ab12", "/tmp/work", 120);
        let line = rows
            .iter()
            .find(|r| r.contains("Memory:"))
            .unwrap_or_else(|| panic!("no Memory row in {:?}", rows));
        if sofuu_ffi::qtsq_linked() {
            assert!(line.contains("persists across sessions"), "{line}");
        } else {
            assert!(!line.contains("persists across sessions"), "{line}");
            assert!(line.contains("NOT persisting"), "{line}");
        }
        cfg.brain = false;
        let rows = welcome_panel_at(&cfg, "s-ab12", "/tmp/work", 120);
        let line = rows.iter().find(|r| r.contains("Memory:")).unwrap();
        assert!(line.contains("off"), "{line}");
    }

    #[test]
    fn welcome_panel_layout() {
        let cfg = ChatConfig {
            provider: "test-prov".into(),
            model: "tm-1".into(),
            effort: "high".into(),
            ..Default::default()
        };
        let rows = welcome_panel_at(&cfg, "s-ab12", "/tmp/work", 80);

        // Box frame: first row opens, the border row closes.
        assert!(rows[0].contains('╭'), "top border: {:?}", rows[0]);
        assert!(rows.iter().any(|r| r.contains('╯')), "bottom border missing");

        // Box rows fill the exact terminal width; the free-flowing tail
        // (after the bottom border) must stay inside it.
        let mut in_box = true;
        for r in &rows {
            if r.contains('╯') {
                in_box = false;
            }
            let cells = cell_w(r);
            if in_box {
                assert_eq!(cells, 80, "box row width off: {:?}", r);
            } else {
                assert!(cells <= 80, "tail row too wide: {:?}", r);
            }
        }

        // Facts present, values column-aligned (same byte offset per row).
        let find_row = |l: &str| rows.iter().find(|r| r.contains(l)).unwrap();
        let dir_row = find_row("Directory:");
        let sess_row = find_row("Session:");
        let model_row = find_row("Model:");
        let ver_row = find_row("Version:");
        assert!(dir_row.contains("/tmp/work"));
        assert!(sess_row.contains("s-ab12"));
        assert_eq!(sess_row.matches("s-ab12").count(), 1);
        assert!(model_row.contains("test-prov/tm-1 - effort high"));
        assert!(ver_row.contains(env!("CARGO_PKG_VERSION")));
        let col = dir_row.find("/tmp/work").unwrap();
        assert_eq!(sess_row.find("s-ab12").unwrap(), col, "session col");
        assert_eq!(model_row.find("test-prov/tm-1").unwrap(), col, "model col");
        assert_eq!(ver_row.find(env!("CARGO_PKG_VERSION")).unwrap(), col, "version col");

        // Unconfigured config renders honestly (no provider-shaped placeholder).
        let unconf = welcome_panel_at(&ChatConfig::defaults(), "", "/tmp/work", 80);
        let unconf_model = unconf.iter().find(|r| r.contains("Model:")).unwrap();
        assert!(unconf_model.contains("not configured"), "{unconf_model:?}");
        // No provider-shaped placeholder: nothing resembles a provider/model pair.
        assert!(!unconf_model.contains("//"), "{unconf_model:?}");
        // The panel must survive a narrow terminal. Its width is clamped to a
        // floor of 30, so anything below that must still produce closed,
        // equal-width box rows rather than a torn border -- the failure the
        // panel_probe.py pty probe exists to catch, but that probe has not
        // been run since the box layout changed and is not in any runner.
        for w in [30usize, 34, 40, 56, 80, 120] {
            let narrow = welcome_panel_at(&ChatConfig::defaults(), "s-ab12", "/tmp/work", w);
            assert!(!narrow.is_empty(), "w={w}: panel rendered nothing");
            assert!(narrow[0].contains('╭'), "w={w}: no top border {:?}", narrow[0]);
            assert!(
                narrow.iter().any(|r| r.contains('╯')),
                "w={w}: no bottom border in {narrow:?}"
            );
            // Every row is exactly the requested width: a row that overruns
            // is what soft-wraps and tears the right-hand │ onto its own
            // line.
            let want = w.max(30);
            for r in &narrow {
                if r.is_empty() {
                    continue; // trailing spacer, legitimately zero-width
                }
                assert_eq!(
                    cell_w(r),
                    want,
                    "w={w}: row width {} != {want}: {r:?}",
                    cell_w(r)
                );
            }
        }
        // A width below the floor is lifted to it, not honoured literally.
        let tiny = welcome_panel_at(&ChatConfig::defaults(), "s", "/w", 10);
        for r in &tiny {
            if r.is_empty() {
                continue;
            }
            assert_eq!(cell_w(r), 30, "under-floor width not lifted: {r:?}");
        }

        let plain = welcome_plain(&ChatConfig::defaults(), None, "/tmp/work");
        assert!(plain[1].contains("not configured"), "{plain:?}");

        // Title and session-less copy.
        assert!(rows.iter().any(|r| r.contains("Welcome to Sofuu!")));
        let no_sess = welcome_panel_at(&cfg, "", "/tmp/work", 80);
        assert!(no_sess
            .iter()
            .any(|r| r.contains("No session here yet")));
        assert!(no_sess.iter().any(|r| r.contains("Session:") && r.contains("none")));
    }

    #[test]
    fn welcome_panel_narrow_terminals() {
        let cfg = ChatConfig::defaults();
        for w in [40usize, 60, 120, 500] {
            let rows = welcome_panel_at(&cfg, "", "/very/long/dir/that/keeps/going/past/the/box", w);
            let expect = w.clamp(40, 400);
            for r in rows.iter().filter(|r| r.contains('│') || r.contains('╭')) {
                assert_eq!(cell_w(r), expect, "width {} row: {:?}", w, r);
            }
        }
    }

    #[test]
    fn welcome_panel_rows_survive_tui_wrap() {
        // js_chat_welcome logs each panel row at term − GUTTER − 1 and
        // tui_push_split soft-wraps stored rows to term − GUTTER — a stored
        // row wider than that budget is torn apart (the right │ lands on
        // its own line; the broken welcome rectangle of 2026-09-09). Every
        // box row must therefore pass through wrap_row byte-identical at
        // every reachable terminal width (tui_size floors at 40); free
        // tail text may soft-wrap and is exempt.
        use sofuu_core::rt::tui::{wrap_row, GUTTER};
        let cfg = ChatConfig {
            provider: "test-prov".into(),
            model: "tm-1".into(),
            ..Default::default()
        };
        for term in [40usize, 44, 57, 80, 120, 157, 241] {
            let budget = term - GUTTER;
            let rows = welcome_panel_at(&cfg, "s-ab12", "/tmp/work", budget - 1);
            let is_box = |r: &String| r.contains('│') || r.contains('╭') || r.contains('╰');
            let widths: Vec<usize> = rows.iter().filter(|r| is_box(r)).map(|r| cell_w(r)).collect();
            assert!(!widths.is_empty(), "term {term}: no box rows");
            assert!(
                widths.iter().all(|&x| x == widths[0]),
                "term {term}: box rows disagree on width: {widths:?}"
            );
            for r in rows.iter().filter(|r| is_box(r)) {
                let wrapped = wrap_row(r, budget);
                assert_eq!(wrapped.len(), 1, "term {term}: box row torn: {r:?}");
                assert_eq!(wrapped[0], *r, "term {term}: box row mutated");
            }
        }

        let long_cfg = ChatConfig {
            provider: "local".into(),
            model: "/Users/priyanshuboruah/Downloads/Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf".into(),
            effort: "high".into(),
            ..Default::default()
        };
        for term in [40usize, 60, 80, 100, 120] {
            let budget = term - GUTTER;
            let rows = welcome_panel_at(&long_cfg, "s-5654", "/Users/priyanshuboruah/ai-native-js-runtime", budget - 1);
            let is_box = |r: &String| r.contains('│') || r.contains('╭') || r.contains('╰');
            for r in rows.iter().filter(|r| is_box(r)) {
                let wrapped = wrap_row(r, budget);
                assert_eq!(wrapped.len(), 1, "term {term}: box row torn by long model path: {r:?}");
                assert_eq!(cell_w(r), budget - 1, "term {term}: row width mismatch");
            }
        }
    }
}

    /// Wide glyphs are 2 cells, so the panel's measure (cell_w) and its
    /// clip (clip_cells/trunc_cells) MUST both count cells. They did not:
    /// the clip side counted chars, letting a 60-cell row come out 102
    /// cells wide — the right border landed far off-screen and the welcome
    /// box tore open. Any overflowing CJK/emoji value hit this.
    #[test]
    fn panel_row_wide_glyphs_clip_to_the_box_width() {
        let cases = [
            "Directory: ".to_string() + &"素".repeat(60),
            "Model: ".to_string() + &"🎉".repeat(60),
            "Model: ".to_string() + &"a".repeat(60),
            // mixed: wide and narrow interleaved, so the bug needs both
            "Model: ".to_string() + &"素🎉ab".repeat(20),
            // exactly at the boundary (no clip) and one cell over
            "Model: ".to_string() + &"素".repeat(26),
            "Model: ".to_string() + &"素".repeat(27),
        ];
        for width in [40usize, 60, 80] {
            for content in &cases {
                let row = panel_row(content, width, "\x1b[2;35m");
                assert_eq!(
                    cell_w(&row),
                    width,
                    "width {width}, content cells {} chars {}: {:?}",
                    cell_w(content),
                    content.chars().count(),
                    &content.chars().take(16).collect::<String>()
                );
            }
        }
    }

    /// trunc_cells is the standalone (escape-free) truncation used for the
    /// Directory value. Same cell/char confusion as panel_row: it compared
    /// `chars().count()` to a cell budget, so a wide-glyph path was kept
    /// whole at ~2x the width it occupies.
    #[test]
    fn trunc_cells_counts_display_cells_not_chars() {
        for max in [4usize, 10, 20] {
            // 2x over budget in cells: must be cut.
            let wide = "素".repeat(max);
            let out = trunc_cells(&wide, max);
            assert!(
                cell_w(&out) <= max,
                "wide {max}: got {} cells from {:?}",
                cell_w(&out),
                out
            );
            assert!(out.ends_with('…'), "wide {max}: expected a cut marker");
            // Narrow text at the same budget must be untouched.
            let narrow = "a".repeat(max);
            assert_eq!(trunc_cells(&narrow, max), narrow, "narrow {max}");
            // Under budget: returned verbatim, no ellipsis.
            let short = "素".repeat(max / 2);
            assert_eq!(trunc_cells(&short, max), short, "short {max}");
        }
    }
