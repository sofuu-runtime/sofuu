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
use std::sync::Mutex;

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

const VALID_PROVIDERS: [&str; 3] = ["openai", "anthropic", "local"];
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
        "local" => 32_768,
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
const ALL_COMMANDS: [&str; 39] = [
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
    ("/brain", "Toggle the memory/brain integration", "on | off"),
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
            // /brain → config panel (DRIVER handles "pick_brain")
            if arg.is_empty() {
                "pick_brain"
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
                chat_out("  Usage: /brain [on|off]\n");
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
    c(&mut rows, "/brain [on|off]", "toggle memory/brain integration");
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
    } else {
        format!("{}/{}{}", cfg.provider, cfg.model, effort)
    };
    let labels: [(String, String); 6] = [
        ("Directory:".into(), trunc_cells(dir, inner.saturating_sub(14))),
        (
            "Session:".into(),
            if session.is_empty() { "none".into() } else { session.into() },
        ),
        ("Model:".into(), model),
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

/// F1: one recalled memory hit (for /why).
#[derive(Clone, serde::Serialize)]
struct RecallHit {
    text: String,
    score: f64,
    role: String,
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

const DRIVER: &str = r#"
(function() {
  let cfg = JSON.parse(__chat_getcfg());
  /* Permission mode: cfg.permissions is the persisted source of truth
   * (/mode, /plan, /edit, /full write it via Rust). The runtime gate is
   * ONE — sofuu.agent.setPermissions drives agent.js's permissionBlocked.
   * Idempotent + guarded so a not-yet-loaded agent.js can't break boot. */
  function applyMode() {
    try {
      if (sofuu.agent && typeof sofuu.agent.setPermissions === 'function')
        sofuu.agent.setPermissions(cfg.permissions || 'full');
    } catch (e) {}
  }
  applyMode();
  let history = [];
  let lastShared = '';
  /* Output archive state is deliberately driver-local. Storage is additive
   * and best-effort: a QTSQ/index failure must never turn a usable model
   * response into a failed chat turn. */
  let archiveTurn = 0;
  let archiveAttempt = 1;
  let archivePendingRecovery = '';
  let archiveLastFailure = null;
  let archiveRecoveryUsed = false;
  /* session-2 (AUDIT-2026-09-07): the prompt event logs ONCE per logical
   * turn. turn() runs again on THINK/CAPACITY retries with the same
   * archiveTurn, so gate on it — duplicate prompt events would replay as
   * extra user turns on resume. */
  let promptLoggedTurn = -1;
  function archiveWrite(kind, status, source, content, metadata, links) {
    if (cfg.archive_enabled === false || content === undefined || content === null) return null;
    const text = String(content);
    if (!text.trim()) return null;
    const request = {
      turn: archiveTurn,
      attempt: archiveAttempt,
      kind: kind,
      status: status,
      source: source,
      provider: (applyFailover(cfg).provider || cfg.provider || ''),
      model: (applyFailover(cfg).model || cfg.model || ''),
      content: text,
      metadata: metadata || {},
      related_output_ids: (links && links.related_output_ids) || [],
      retry_of: (links && links.retry_of) || null,
      redact: true,
    };
    try {
      const result = JSON.parse(__chat_archive_write(JSON.stringify(request)) || '{}');
      return (result && result.stored && result.ref) ? result.ref : null;
    } catch (e) {
      return null;
    }
  }
  function archiveErrorCode(message) {
    const text = String(message || '').toLowerCase();
    if (/context|prompt.{0,12}too long|maximum context|token limit/.test(text)) return 'context_limit';
    if (/max.{0,12}output|length.{0,12}limit|too many output|finish.?reason.{0,8}length/.test(text)) return 'output_limit';
    if (/auth|unauthori[sz]ed|forbidden|api.?key|invalid token|credential/.test(text)) return 'auth';
    if (/timeout|timed out|connection|transport|http\/?2|empty reply|reset by peer|rate.?limit|overloaded|temporarily/.test(text)) return 'transport';
    if (/provider|model|endpoint|http \d+/.test(text)) return 'provider';
    return 'runtime';
  }
  function archiveContext(query) {
    const automatic = !!(query && query.automatic === true);
    if (cfg.archive_enabled === false || (automatic && cfg.archive_auto_recover === false)) return '';
    try {
      const requestedBudget = Number(query && query.budget_chars || cfg.archive_recovery_budget_chars || 12000);
      const q = Object.assign({}, query || {}, {
        automatic: automatic,
        budget_chars: Math.max(512, Math.min(131072, isFinite(requestedBudget) ? requestedBudget : 12000)),
        ml_relevance: cfg.ml !== false && !!(sofuu.ml && sofuu.ml.relevance &&
          typeof sofuu.ml.relevance.plan === 'function'),
      });
      const result = JSON.parse(__chat_archive_context(JSON.stringify(q)) || '{}');
      if (automatic && result && result.bundle && Array.isArray(result.bundle.items) && result.bundle.items.length) {
        const items = result.bundle.items;
        const latest = items.reduce(function (best, item) {
          return !best || String(item.created_at || '') > String(best.created_at || '') ? item : best;
        }, null);
        const kinds = items.slice(0, 3).map(function (item) { return String(item.kind || 'output'); }).join(' + ');
        try { out('\x1b[90m  ⏺ recovered ' + items.length + ' historical output' + (items.length === 1 ? '' : 's') +
          (latest && latest.created_at ? ' · latest ' + latest.created_at : '') +
          (kinds ? ' · ' + kinds : '') + '\x1b[0m\n'); } catch (eNotice) {}
      }
      return result && result.bundle && result.bundle.context ? String(result.bundle.context) : '';
    } catch (e) {
      return '';
    }
  }
  /* Public native-backed surface for scripts running inside the main chat
   * runtime. It returns parsed JSON envelopes so callers do not need to know
   * about the QuickJS string bridge. Historical context remains explicitly
   * bounded and labeled by the native implementation. */
  if (!sofuu.outputs) sofuu.outputs = {};
  function archiveApiResult(raw) {
    try { return JSON.parse(raw || '{}'); }
    catch (e) { return { ok: false, error: 'invalid output archive response' }; }
  }
  sofuu.outputs.info = function () {
    return archiveApiResult(__chat_archive_stats());
  };
  sofuu.outputs.list = function (query) {
    return archiveApiResult(__chat_archive_list(JSON.stringify(query || {})));
  };
  sofuu.outputs.get = function (id) {
    return archiveApiResult(__chat_archive_get(String(id || '')));
  };
  sofuu.outputs.search = function (query) {
    const q = typeof query === 'string' ? { text: query } : (query || {});
    return archiveApiResult(__chat_archive_search(JSON.stringify(q)));
  };
  sofuu.outputs.context = function (query) {
    const q = typeof query === 'string' ? { current_task: query } : Object.assign({}, query || {});
    q.automatic = false;
    return archiveApiResult(__chat_archive_context(JSON.stringify(q)));
  };
  sofuu.outputs.rebuildIndex = function () {
    return archiveApiResult(__chat_archive_rebuild());
  };
  sofuu.outputs.prune = function (policy) {
    const p = Object.assign({ dry_run: true }, policy || {});
    return archiveApiResult(__chat_archive_prune(JSON.stringify(p)));
  };
  /* Provider failover (P1): when a turn dies with a capacity/auth-class
   * error on the active provider, the main loop picks the next
   * configured provider entry and sets this override for the retry. All
   * per-turn consumers (streamOpts, resolveCaps, the agent def) read cfg
   * through applyFailover() so the whole turn runs against the target.
   * Cleared at turn end — the SAVED config never changes; failover is a
   * per-turn recovery, not a provider switch. */
  let failoverTo = null;
  function applyFailover(c) {
    if (!failoverTo) return c;
    return Object.assign({}, c, {
      provider: failoverTo.name,
      base_url: failoverTo.endpoint,
      api_key: failoverTo.api_key || '',
      profile: failoverTo.profile || '',
      model: failoverTo.model,
    });
  }
  function failoverChain() {
    const chain = [null];
    if (Array.isArray(cfg.providers)) {
      for (const pe of cfg.providers) {
        if (pe && pe.name && pe.model && pe.endpoint && pe.name !== cfg.provider) {
          chain.push({ name: pe.name, endpoint: pe.endpoint,
                       api_key: pe.api_key || '', profile: pe.profile || '',
                       model: pe.model });
        }
      }
    }
    return chain;
  }
  /* Footer meter = CURRENT context usage (what the next request's prompt
   * will be), not a lifetime total: it grows as history grows and drops
   * when /compact (or auto-compaction) folds history, exactly like the
   * context meters in other CLIs. Calibrated against the FIRST LLM request
   * of a turn — its prompt is system + history + tools + ephemeral, i.e.
   * the context size, before any tool results re-count it. Falls back to
   * the budget logic's own estimator when a provider reports no usage. */
  let usedCtx = 0;
  let ctxOverhead = -1;   /* −1 = not calibrated (history estimate only) */
  /* Live mid-turn growth (estimated tokens accumulated since the latest
   * in-flight request): the footer adds this WHILE a turn is running so
   * the meter climbs as the answer streams and tool results land — reset
   * to 0 when the turn ends (the end-of-turn calibration takes over). */
  let liveCtxExtra = 0;
  function ctxMeter() {
    const h = historyTokens();
    const base = ctxOverhead >= 0 ? h + ctxOverhead : h;
    return base + liveCtxExtra;
  }
  let lastChip = '';  /* last per-turn usage chip, so /compact can refresh the footer without clearing it */
  let rssWarned = false; /* M6: one-time RAM tripwire notice */
  /* ── MCP tool wiring (zero-config via ~/.sofuu/mcp.json) ────────
   * OPTIONAL and LAZY: servers are NOT connected at chat startup — a slow
   * or dead server must never block the prompt. Connections are made on
   * demand: the first /tools, or when the agent loop needs tools. Failed
   * servers are logged and skipped, never fatal. */
  let mcpClients = [];     /* [{name, client}] — live connections */
  let mcpTools = [];       /* [{server, name, description, schema}] */
  let mcpConnected = 0;
  let mcpFailed = 0;
  let mcpConnecting = null; /* in-flight connect promise (dedupe) */
  function mcpServersList() {
    try { return JSON.parse(__chat_mcpservers() || '[]'); } catch (e) { return []; }
  }
  async function connectMcpServers() {
    if (mcpConnecting) return mcpConnecting; /* one connect at a time */
    const list = mcpServersList();
    if (!list.length) return;
    if (!sofuu.mcp || !sofuu.mcp.connect) { out('\x1b[33m  ⚠ sofuu.mcp unavailable — tools off\x1b[0m'); return; }
    mcpConnecting = (async () => {
      const results = await Promise.allSettled(list.map(async (srv) => {
        const client = await sofuu.mcp.connect(srv.command);
        const res = await client.listTools();
        const arr = (res && res.tools) || [];
        return { name: srv.name, client, tools: arr };
      }));
      for (const r of results) {
        if (r.status === 'fulfilled' && r.value) {
          mcpClients.push({ name: r.value.name, client: r.value.client });
          for (const t of (r.value.tools || [])) {
            mcpTools.push({ server: r.value.name, name: t.name,
                            description: t.description || '', schema: t.inputSchema });
          }
          mcpConnected++;
        } else {
          mcpFailed++;
        }
      }
      if (mcpConnected > 0) {
        out('\x1b[90m  ⏺ mcp: ' + mcpConnected + ' server' + (mcpConnected === 1 ? '' : 's') + ' · ' + mcpTools.length + ' tools\x1b[0m');
        if (mcpFailed > 0) out('\x1b[90m  ⚠ ' + mcpFailed + ' MCP server' + (mcpFailed === 1 ? '' : 's') + ' unreachable (see ~/.sofuu/mcp.json)\x1b[0m');
      } else if (mcpFailed > 0) {
        out('\x1b[33m  ⚠ all ' + mcpFailed + ' MCP server' + (mcpFailed === 1 ? '' : 's') + ' unreachable — tools off\x1b[0m');
      }
    })();
    try {
      await mcpConnecting;
    } finally {
      mcpConnecting = null;
    }
  }
  function showTools() {
    if (mcpTools.length === 0) {
      out('\n\x1b[90m  No MCP servers connected. Add one to ~/.sofuu/mcp.json:\x1b[0m');
      out('  \x1b[90m  [{"name":"fs","command":"npx @modelcontextprotocol/server-filesystem /tmp"}]\x1b[0m\n');
      return;
    }
    out('\n\x1b[1m  MCP tools\x1b[0m');
    let lastServer = '';
    for (const t of mcpTools) {
      if (t.server !== lastServer) {
        out('  \x1b[36m' + t.server + '\x1b[0m');
        lastServer = t.server;
      }
      out('    • ' + t.name + (t.description ? ' — ' + String(t.description).slice(0, 80) : ''));
    }
    out('');
  }
  /* /agents — list the agent registry (PLAN-AGENTS A8.3). Definitions
   * load lazily from ~/.sofuu/agents/*.js on first use; a broken file is
   * listed with its error, never fatal to chat. */
  let agentsDirLoaded = false;
  async function handleAgentsCmd() {
    if (!sofuu.agent || typeof sofuu.agent.list !== 'function') {
      out('\x1b[90m  agent runtime unavailable\x1b[0m\n');
      return;
    }
    if (!agentsDirLoaded) {
      agentsDirLoaded = true;
      const ld = await sofuu.agent.loadDir();
      for (const b of (ld && ld.broken) || []) {
        out('\x1b[33m  ⚠ ' + b.file + ': ' + b.error + '\x1b[0m');
      }
    }
    const list = sofuu.agent.list();
    if (!list.length) {
      out('\n\x1b[90m  No agents defined. Add one at ~/.sofuu/agents/<name>.js:\x1b[0m');
      out('  \x1b[90msofuu.agent.define({ name: "researcher", system: "You research precisely.", tools: ["web"] });\x1b[0m');
      out('  \x1b[90mThen: sofuu agent run researcher "find out X"\x1b[0m\n');
      return;
    }
    out('\n\x1b[1m  Agents\x1b[0m');
    for (const a of list) {
      out('  \x1b[36m' + a.name + '\x1b[0m' +
          (a.model ? ' · ' + a.model : '') +
          ' · memory ' + a.memory +
          ' · ' + a.tools + ' tool' + (a.tools === 1 ? '' : 's') +
          (a.mcpServers ? ' · ' + a.mcpServers + ' mcp server' + (a.mcpServers === 1 ? '' : 's') : '') +
          ((a.agents && a.agents.length) ? ' · delegates: ' + a.agents.join(', ') : '') +
          (a.runs24h ? ' · ' + a.runs24h + ' run' + (a.runs24h === 1 ? '' : 's') + ' today' : ''));
      if (a.system) out('    \x1b[90m' + a.system + '\x1b[0m');
    }
    out('');
  }
  async function callMcpTool(name, args) {
    /* F4a / PLAN-AGENTS A6: route by name→OWNER server. The old
     * first-server-wins loop was dangerously wrong under multiple servers
     * (two servers exposing the same tool name = silent cross-calls). */
    let owner = null;
    for (const t of mcpTools) { if (t.name === name) { owner = t.server; break; } }
    if (!owner) throw new Error('tool ' + name + ' not found on any MCP server');
    for (const c of mcpClients) {
      if (c.name !== owner) continue;
      const res = await c.client.call('tools/call', { name: name, arguments: args || {} });
      const text = res && res.content && res.content[0] && res.content[0].text;
      return (typeof text === 'string') ? text : JSON.stringify(text);
    }
    throw new Error('tool ' + name + ': owning server "' + owner + '" is not connected');
  }
  /* Tool set handed to sofuu.agent.run each turn (A1.5): built-in web
   * tools (keyless by default) + the MCP tools as inline specs whose
   * execute closures route through the fixed callMcpTool. Inline/builtin
   * tools win on a name clash (A6 namespace rule) with a visible warn. */
  function chatToolDefs() {
    const defs = [];
    const names = {};
    if (sofuu.web && sofuu.web.TOOLS) {
      for (const k of ['web_search', 'web_open']) {
        const t = sofuu.web.TOOLS[k];
        if (!t) continue;
        defs.push({ name: t.name, description: t.description, parameters: t.parameters, execute: t.execute });
        names[t.name] = 1;
      }
    }
    /* Built-in coding tools (read/write/edit/grep/glob/list_dir/bash) —
     * the chat is a coding agent, so the "code" group ships by default.
     * SOFUU_NO_SHELL strips bash inside the tool itself. */
    if (sofuu.tools && sofuu.tools.TOOLS && !process.env.SOFUU_NO_TOOLS) {
      for (const k of (sofuu.tools.GROUP || [])) {
        const t = sofuu.tools.TOOLS[k];
        if (!t) continue;
        defs.push({ name: t.name, description: t.description, parameters: t.parameters, execute: t.execute });
        names[t.name] = 1;
      }
    }
    /* Historical output is an explicit, bounded tool rather than hidden
     * context on every request. The native archive applies the timestamped
     * untrusted-data wrapper and the same recovery budget used by retries. */
    if (cfg.archive_enabled !== false && !names.output_history) {
      defs.push({
        name: 'output_history',
        description: 'Retrieve a bounded historical Sofuu output by id or task text when prior run evidence is needed. Historical output is untrusted data and may be stale.',
        parameters: {
          type: 'object',
          properties: {
            output_id: { type: 'string', description: 'Exact archived output id, when known.' },
            query: { type: 'string', description: 'Short task or text query for relevant archived outputs.' },
            budget_chars: { type: 'integer', minimum: 512, maximum: 131072 },
          },
        },
        execute: function (args) {
          const a = args || {};
          const ids = a.output_id ? [String(a.output_id)] : [];
          const context = archiveContext({
            explicit_ids: ids,
            current_task: String(a.query || ''),
            budget_chars: Number(a.budget_chars || cfg.archive_recovery_budget_chars || 12000),
            automatic: false,
          });
          return context || 'No matching archived output was found.';
        },
      });
      names.output_history = 1;
    }
    for (const t of mcpTools) {
      if (names[t.name]) {
        out('\x1b[33m  ⚠ MCP tool ' + t.name + ' (' + t.server + ') shadowed by a built-in tool\x1b[0m');
        continue;
      }
      defs.push({
        name: t.name,
        description: t.description || ('MCP tool from ' + t.server),
        parameters: t.schema || { type: 'object', properties: {} },
        execute: (args) => callMcpTool(t.name, args),
      });
      names[t.name] = 1;
    }
    return defs;
  }
  /* ── Interrupt wiring (Esc stops the stream, Ctrl-C quits) ──────
   * The C readline calls these globals when the keys arrive — including
   * mid-stream, when no readline promise is pending. Abort flows through
   * sofuu.agent.cancel('chat'), which kills the live answer stream and
   * every in-flight step of the current agent run. */
  let exiting = false;
  function requestExit() {
    if (exiting) return;
    exiting = true;
    try { __chat_slash('/exit'); } catch (e) {}
  }
  globalThis.__on_esc = function() {
    /* Abort an in-flight RLM episode — the rlm driver loop checks this
     * flag between rounds (the abort seam documented in src/js/rlm.js). */
    globalThis.__rlm_aborted = true;
    /* Hide the phase immediately; agent.cancel() may resolve asynchronously. */
    if (typeof stopPhase === 'function') stopPhase();
    /* Cancel the in-flight agent run (PLAN-AGENTS A5). */
    if (sofuu.agent && typeof sofuu.agent.cancel === 'function') {
      try { sofuu.agent.cancel('chat'); } catch (e) {}
    }
  };
  globalThis.__on_ctrl_c = function() {
    globalThis.__on_esc();
    requestExit();
  };
  /* TAB on a non-slash line (process.rs tty_read_cb) cycles the permission
   * mode. Reuses the /mode command so the echo + persistence come from the
   * same Rust path as the slash commands, then mirrors the submit loop's
   * refresh set: engine (sofuu.agent), footer chip, welcome panel. */
  /* Refresh the welcome panel ONLY when it cannot destroy anything: the
   * full refresh wipes the scrollback, so running it mid-conversation
   * erases the visible transcript (a mode/theme switch looked like a
   * fresh session). Before the first turn the panel is all there is, so
   * refresh freely; after that the command's own confirmation line plus
   * the live footer already report the new settings. TTY-only: piped, a
   * refresh appends a duplicate panel instead of replacing in place. */
  function maybeRefreshPanel() {
    if (TTY && history.length === 0) { try { __chat_refresh(); } catch (e) {} }
  }
  globalThis.__on_tab = function() {
    const order = ['full', 'edit', 'plan'];
    const cur = cfg.permissions || 'full';
    let next = order.indexOf(cur);
    next = order[(next + 1) % order.length];
    __chat_slash('/mode ' + next);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    applyMode();
    refreshStatus(lastChip);
    maybeRefreshPanel();
  };
  /* ── Memory: OWNED BY THE AGENT RUNTIME (PLAN-AGENTS A1.4) ──────
   * Recall augmentation, scoped storage, and markPositive live in
   * src/js/agent.js (sofuu.agent.run with memory:'shared'); this driver
   * only passes cfg.brain through and renders recall events. */
  // Bounds: history is re-sent to the model every turn, so it must not grow
  // without limit (memory + token cost). The entry cap is a safety net; the
  // real limit is the token budget derived from the configured context
  // window (see trimHistory).
  const MAX_HISTORY_ENTRIES = 2000;
  const MAX_ANSWER_CHARS = 200000;

  /* ═════════════════════════════════════════════════════════════════
   * F1–F11: Chat & Brain Features (PLAN-CHAT-FEATURES.md)
   * ═════════════════════════════════════════════════════════════════ */

  /* ── F1: /remember + /why ────────────────────────────────────────── */
  /* ONE brain handle, ONE store: the agent runtime's BRAINS cache owns the
   * brain file for this process — a QTSQ flush rewrites the whole file from
   * the handle's in-memory view, so a second driver-owned handle would
   * clobber turns stored after its hydrate (last flush wins). Every direct
   * brain op (/remember, /share, /import, ghost) goes through
   * sofuu.agent.brainFor(), which returns that SAME cached handle the turn
   * loop recalls and stores with. Path default is project-local
   * (<cwd>/.sofuu/brain/brain.qtsq), shared by every session in this
   * folder; config "brain_path" overrides for both sides. (The old
   * ~/.sofuu_brain.qtsq default forked /remember onto a file the runtime
   * never opened — fixed 2026-09-09.) */
  function brainDef() {
    return {
      name: 'chat',
      memory: cfg.brain ? 'shared' : 'off',
      brainPath: cfg.brain_path || undefined,
      embed_provider: cfg.embed_provider || undefined,
      embed_model: cfg.embed_model || undefined,
      embedding: cfg.memory_backend || undefined,
    };
  }
  function brainPath() {
    /* Display/metadata mirror of agent.js brainFor's path chain: explicit
     * config override, then project-local, then HOME. */
    if (cfg.brain_path) return cfg.brain_path;
    try { return process.cwd() + '/.sofuu/brain/brain.qtsq'; } catch (e) {}
    return (env('HOME') || env('USERPROFILE') || '.') + '/.sofuu/brain/brain.qtsq';
  }
  function env(n) { try { return process.env[n] || ''; } catch (e) { return ''; } }
  function ensureDriverBrain() {
    try {
      if (!sofuu.agent || typeof sofuu.agent.brainFor !== 'function') return null;
      const entry = sofuu.agent.brainFor(brainDef());
      if (!entry || !entry.cma) return null;
      if (entry.backend) driverBrainBackend = entry.backend;
      return entry.cma;
    } catch (e) { return null; }
  }
  /* Driver-side embedding backend (PLAN-TINY-SEMANTIC-EMBEDDER §7) —
   * mirrors agent.js selection; now only a FALLBACK identity source for
   * embedText(): the real backend of the shared handle arrives via
   * sofuu.agent.brainFor (ensureDriverBrain caches entry.backend).
   * Dims come from sofuu.ai.embedInfo(), never literals. Null = the
   * selected backend is unavailable (memory declines, never substitutes). */
  let driverBrainBackend = null;
  function driverBackend() {
    try {
      if (cfg.embed_provider && cfg.embed_model) {
        return { kind: 'remote', id: 'remote:' + cfg.embed_provider + ':' + cfg.embed_model, dim: 0 };
      }
      let name = String(cfg.memory_backend || env('SOFUU_MEMORY_BACKEND') || '').toLowerCase();
      if (name === 'semantic' || name === 'semantic-projector-v1') {
        if (sofuu.ai && typeof sofuu.ai.embedInfo === 'function' && typeof sofuu.ai.embedLocalSemantic === 'function') {
          const info = JSON.parse(sofuu.ai.embedInfo());
          if (info && info.id === 'semantic-projector-v1' && info.available && (info.dimension | 0) > 0) {
            return { kind: 'semantic', id: info.id, dim: info.dimension | 0 };
          }
        }
        return null;
      }
      let dim = 768;
      if (sofuu.ai && typeof sofuu.ai.embedLocal === 'function') {
        try { const v = sofuu.ai.embedLocal(''); if (v && v.length) dim = v.length; } catch (e) {}
      }
      return { kind: 'hash', id: 'hash-v1', dim: dim };
    } catch (e) { return null; }
  }
  async function embedText(text) {
    const be = driverBrainBackend || driverBackend();
    if (!be) return null;
    /* Local-first: use the selected backend (hash default, semantic opt-in). */
    if (be.kind === 'semantic' && sofuu.ai && typeof sofuu.ai.embedLocalSemantic === 'function') {
      try {
        const v = await sofuu.ai.embedLocalSemantic(text);
        if (v && v.length === be.dim) return v;
      } catch (e) {}
      return null;
    }
    if (sofuu.ai && typeof sofuu.ai.embedLocal === 'function') {
      try {
        const v = await sofuu.ai.embedLocal(text);
        if (v && v.length === be.dim) return v;
      } catch (e) {}
    }
    /* Fall back to a remote embedding provider if configured. */
    if (cfg.embed_provider && cfg.embed_model && sofuu.ai && typeof sofuu.ai.embed === 'function') {
      try {
        const r = await sofuu.ai.embed(text, { provider: cfg.embed_provider, model: cfg.embed_model });
        if (r && r.vector) return r.vector;
        if (r && r.embedding) return r.embedding;
        if (Array.isArray(r)) return r;
      } catch (e) {}
    }
    return null;
  }
  async function handleRemember(fact) {
    if (!cfg.brain) { out('\x1b[90m  Brain is off — /brain on to enable\x1b[0m\n'); return; }
    fact = String(fact || '').trim();
    if (!fact) { out('\x1b[90m  Usage: /remember <fact>\x1b[0m\n'); return; }
    const brain = ensureDriverBrain();
    if (!brain) { out('\x1b[90m  Brain unavailable (QTSQ not linked?)\x1b[0m\n'); return; }
    const vec = await embedText(fact);
    if (!vec) { out('\x1b[90m  Embeddings unavailable — cannot store fact\x1b[0m\n'); return; }
    try {
      brain.remember(new Float32Array(vec), fact, 'user_pin', 0);
      brain.flush();
      /* AGENTS.md (2026-09-12): the fact also lands in the project context
       * file so it rides EVERY request as standing context, not just via
       * semantic recall. Never fatal — the brain pin above already held. */
      let fileNote = '';
      if (sofuu.agent && typeof sofuu.agent.pinProjectFact === 'function') {
        try {
          const r = await sofuu.agent.pinProjectFact(fact);
          if (r === 'added') fileNote = ' · pinned to AGENTS.md';
          else if (r === 'exists') fileNote = ' · already in AGENTS.md';
        } catch (eP) {}
      }
      out('\x1b[90m  ⏺ remembered · ' + brain.count() + ' memories · ' + brainPath() + fileNote + '\x1b[0m\n');
    } catch (e) { out('\x1b[31m  ✗ ' + String(e.message || e) + '\x1b[0m\n'); }
  }
  function handleWhy() {
    let hits = [];
    try { hits = JSON.parse(__chat_get_recall()); } catch (e) { hits = []; }
    if (!hits || hits.length === 0) {
      out('\x1b[90m  No memories were recalled for the last answer.\x1b[0m');
      out('\x1b[90m  (Brain off, no recall this turn, or no matching memories.)\x1b[0m\n');
      return;
    }
    out('\n\x1b[1m  Memories that shaped the last answer:\x1b[0m');
    for (const h of hits) {
      const score = (h.score || 0).toFixed(3);
      const role = h.role ? '\x1b[2m(' + h.role + ')\x1b[0m ' : '';
      const text = clip1(h.text, 120);
      out('  \x1b[90m' + score + '\x1b[0m  ' + role + text);
    }
    out('');
  }

  /* ── Timestamped output archive ──────────────────────────────────── */
  function printArchiveItems(items, title) {
    if (title) out('\n\x1b[1m  ' + title + '\x1b[0m');
    if (!items.length) {
      out('\x1b[90m  (no archived outputs)\x1b[0m\n');
      return;
    }
    for (const item of items) {
      out('  \x1b[36m' + clip1(item.id, 56) + '\x1b[0m · ' +
          String(item.created_at || '?') + ' · ' + String(item.kind || '?') +
          ' · ' + String(item.status || '?') +
          ' · ' + clip1(item.summary || '', 120) +
          (item.missing ? ' \x1b[33m[missing]\x1b[0m' : ''));
    }
    out('');
  }
  function archiveItems(query, title) {
    let result;
    try { result = JSON.parse(__chat_archive_list(JSON.stringify(query || {})) || '{}'); }
    catch (e) { result = null; }
    if (!result || result.ok === false) {
      out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n');
      return [];
    }
    const items = Array.isArray(result.items) ? result.items : [];
    printArchiveItems(items, title);
    return items;
  }
  function handleOutputs(arg) {
    const raw = String(arg || '').trim();
    const parts = raw ? raw.split(/\s+/) : [];
    const op = parts[0] || 'list';
    if (op === 'list' || op === 'ls') {
      const limit = parts.indexOf('--limit') >= 0 ? parseInt(parts[parts.indexOf('--limit') + 1], 10) : 20;
      archiveItems({ limit: isFinite(limit) ? limit : 20 }, 'Recent archived outputs');
      return;
    }
    if (op === 'show' || op === 'get') {
      const id = parts[1] || '';
      if (!id) { out('\x1b[90m  Usage: /outputs show <output-id>\x1b[0m\n'); return; }
      try {
        const result = JSON.parse(__chat_archive_get(id) || '{}');
        if (!result || result.ok === false || !result.record) {
          out('\x1b[33m  ⚠ output not found\x1b[0m\n');
          return;
        }
        const record = result.record;
        out('\n\x1b[1m  ' + record.id + '\x1b[0m · ' + record.created_at +
            ' · ' + record.kind + ' · ' + record.status);
        out(String(record.content || (record.artifact ? JSON.stringify(record.artifact, null, 2) : '')) + '\n');
      } catch (e) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); }
      return;
    }
    if (op === 'search') {
      const text = parts.slice(1).filter(function (p) { return p !== '--limit'; }).join(' ');
      let result;
      try { result = JSON.parse(__chat_archive_search(JSON.stringify({ text: text, limit: 50 })) || '{}'); }
      catch (e) { result = null; }
      if (!result || result.ok === false) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); return; }
      printArchiveItems(Array.isArray(result.items) ? result.items : [], 'Search results for "' + clip1(text, 80) + '"');
      return;
    }
    if (op === 'context') {
      const task = parts.slice(1).join(' ');
      if (!task) { out('\x1b[90m  Usage: /outputs context <task>\x1b[0m\n'); return; }
      const context = archiveContext({ current_task: task, automatic: false });
      if (context) out('\n' + context);
      else out('\x1b[90m  No relevant archived output found.\x1b[0m\n');
      return;
    }
    if (op === 'stats' || op === 'info') {
      try {
        const result = JSON.parse(__chat_archive_stats() || '{}');
        const s = result && result.stats;
        if (!s) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); return; }
        out('\n\x1b[1m  Output archive\x1b[0m');
        out('  ' + (s.index_entries || 0) + ' records · ' + (s.indexed_bytes || 0) + ' logical bytes · ' +
            (s.payload_file_bytes || 0) + ' archive bytes');
        out('  raw ' + (s.raw_records || 0) + ' · compressed ' + (s.compressed_records || 0) +
            ' · encrypted ' + (s.encrypted_records || 0) + ' · corrupt ' + (s.corrupt_records || 0) +
            ' · missing ' + (s.missing_records || 0));
        out('');
      } catch (e) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); }
      return;
    }
    if (op === 'rebuild-index' || op === 'repair') {
      try { out('\n  ' + JSON.stringify(JSON.parse(__chat_archive_rebuild() || '{}'), null, 2) + '\n'); }
      catch (e) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); }
      return;
    }
    if (op === 'prune') {
      const apply = parts.indexOf('--apply') >= 0 || parts.indexOf('apply') >= 0;
      try {
        const result = JSON.parse(__chat_archive_prune(JSON.stringify({ dry_run: !apply })) || '{}');
        out('\n  ' + JSON.stringify(result, null, 2) + '\n');
      } catch (e) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); }
      return;
    }
    out('\x1b[90m  Usage: /outputs [list|show <id>|search <text>|context <task>|stats|rebuild-index|prune [--apply]|on|off]\x1b[0m\n');
  }

  /* ── F2: /resume ──────────────────────────────────────────────────── */
  async function handleResume(arg) {
    let sessionId = '';
    if (arg && arg.length > 0) {
      /* /resume <id> — direct resume */
      sessionId = arg;
    } else {
      /* Bare /resume — show picker */
      let sessions = [];
      try { sessions = JSON.parse(__chat_sessions()); } catch (e) {}
      if (!sessions.length) {
        out('\x1b[90m  No other sessions found on this project.\x1b[0m\n');
        return;
      }
      if (!TTY) {
        out('\x1b[90m  Sessions on this project:\x1b[0m');
        for (const s of sessions) {
          out('  \x1b[36m' + s.short + '\x1b[0m · ' + s.turns + ' turns · ' +
              (s.task || '(no task)') + (s.ended ? '' : ' \x1b[33m(active)\x1b[0m'));
        }
        out('\n\x1b[90m  Use /resume <id> to resume one.\x1b[0m\n');
        return;
      }
      const items = sessions.map(s => ({
        id: s.id,
        label: s.short + ' · ' + s.turns + ' turns',
        note: (s.task || '(no task)') + (s.ended ? '' : ' (active)'),
      }));
      const picked = await selectMenu({
        title: 'Resume a session',
        hint: '↑↓ to navigate, enter to resume, esc to cancel',
        items: items,
      });
      if (!picked) { out('\x1b[90m  Cancelled\x1b[0m\n'); return; }
      sessionId = picked.id;
    }
    let turns = [];
    try { turns = JSON.parse(__chat_resume_turns(sessionId)); } catch (e) {}
    if (!turns.length) {
      out('\x1b[90m  Session has no turns to resume.\x1b[0m\n');
      return;
    }
    /* Safety: cap at 60 entries (30 turns). */
    let capped = false;
    if (turns.length > 30) {
      turns = turns.slice(-30);
      capped = true;
    }
    history = [];
    for (const t of turns) {
      history.push({ role: 'user', content: t.prompt });
      history.push({ role: 'assistant', content: t.answer });
    }
    /* Resume distills instead of dumping: the old turns go through the
     * ACTIVE model for a brief continuity summary, which becomes the
     * context — the full transcript is neither printed nor (past the kept
     * last turn) loaded. A single turn has nothing worth distilling, and
     * without a usable provider there is nothing to distill WITH: both
     * fall through to the direct load below. */
    const canDistill = turns.length > 1 && usableConfig();
    let resumedSummary = null;
    if (canDistill) {
      try {
        resumedSummary = await summarizeHistory(0, {
          source: 'resume',
          prompt: 'You are resuming a previous coding session. Summarize the conversation below BRIEFLY for continuity: what the user was working on, key decisions, files touched (exact paths), current task state, and the next step. Output only the summary, no preamble.',
        });
      } catch (e) { resumedSummary = null; }
    }
    /* Resume is an explicit recovery boundary. Attach only the bounded,
     * same-session archive context selected by the native index; ordinary
     * turns never perform this historical scan. Attached AFTER the distill
     * so archive context is never summarized as if it were conversation. */
    const resumedArchive = archiveContext({
      session_id: sessionId,
      automatic: true,
    });
    if (resumedArchive) history.unshift({ role: 'system', content: resumedArchive });
    usedCtx = ctxMeter(); /* meter reflects the resumed history immediately */
    const short = sessionId.length > 8 ? sessionId.slice(0, 8) : sessionId;
    const capNote = capped ? ' (partial — kept most recent 30)' : '';
    if (resumedSummary) {
      out('\x1b[90m  ⏺ resumed ' + short + ' · ' + turns.length + ' turns → summary' + capNote + '\x1b[0m\n');
      out(resumedSummary);
      /* The most recent exchange stays verbatim (summarizeHistory keeps
       * the last block) so immediate context is exact, not paraphrased. */
      const last = turns[turns.length - 1];
      out('\x1b[2m  ›\x1b[0m ' + last.prompt + '\x1b[0m');
      if (last.answer) out(last.answer);
      out('\x1b[90m  full transcript: /context ' + short + '\x1b[0m\n');
      return;
    }
    out('\x1b[90m  ⏺ resumed ' + short + ' · ' + turns.length + ' turns' + capNote + '\x1b[0m\n');
    if (canDistill) {
      out('\x1b[90m  (summary unavailable — showing full transcript)\x1b[0m');
    }
    /* Direct load: the transcript enters context AND the conversation area
     * (same array both places, so display and context agree). */
    for (const t of turns) {
      out('\x1b[2m  ›\x1b[0m ' + t.prompt + '\x1b[0m');
      if (t.answer) out(t.answer);
    }
    out('\x1b[90m  ── end of resumed transcript ──\x1b[0m\n');
  }

  /* ── /sessions: interactive session browser ───────────────────────
   * /sessions used to print a static table (session::cmd_list) that looked
   * selectable but took no keys — arrows and Enter did nothing there. Bare
   * /sessions now opens the same keyboard-driven picker as /resume, scoped
   * to this project (the title names it); Enter resumes the highlighted
   * session. /sessions <id> resumes directly. Piped output keeps the plain
   * table. */
  async function handleSessions(arg) {
    if (arg && arg.length > 0) { await handleResume(arg); return; }
    let project = '';
    try { project = __chat_project() || ''; } catch (e) {}
    if (!project) { try { project = process.cwd(); } catch (e) {} }
    let sessions = [];
    try { sessions = JSON.parse(__chat_sessions()); } catch (e) {}
    if (!sessions.length) {
      out('\x1b[90m  No other sessions found on this project' +
          (project ? ' (' + project + ')' : '') + '.\x1b[0m\n');
      return;
    }
    if (!TTY) {
      out('\x1b[90m  Sessions for ' + project + ':\x1b[0m');
      for (const s of sessions) {
        out('  \x1b[36m' + s.short + '\x1b[0m · ' + s.turns + ' turns · ' +
            (s.task || '(no task)') + (s.ended ? '' : ' \x1b[33m(active)\x1b[0m'));
      }
      out('\n\x1b[90m  Use /sessions <id> (or /resume <id>) to resume one.\x1b[0m\n');
      return;
    }
    const items = sessions.map(s => ({
      id: s.id,
      label: s.short + ' · ' + s.turns + ' turns',
      note: (s.task || '(no task)') + (s.ended ? '' : ' (active)'),
    }));
    const picked = await selectMenu({
      title: 'Sessions for ' + project,
      hint: '↑↓ to navigate, enter to resume, esc to cancel',
      items: items,
    });
    if (!picked) { out('\x1b[90m  Cancelled\x1b[0m\n'); return; }
    await handleResume(picked.id);
  }

  /* ── F3: @file mentions ──────────────────────────────────────────── */
  /* Total attachment budget scales with the model's context window
   * (attachTokenBudget below) — not a flat constant. */
  const ATTACH_TOKEN_BUDGET_FALLBACK = 8192;
  async function expandMentions(text) {
    const mentions = [];
    /* Tokenize @path or @path:start-end (allow quoted paths with spaces). */
    const re = /@(?:("[^"]+")|([^\s]+(?:\:\d+-\d+)?))/g;
    let m;
    while ((m = re.exec(text)) !== null) {
      const raw = m[1] ? m[1].slice(1, -1) : m[2];
      mentions.push(raw);
    }
    if (mentions.length === 0) return { text: text, manifest: '' };
    const cwd = process.cwd();
    let totalTokens = 0;
    const blocks = [];
    const manifestParts = [];
    for (const mention of mentions) {
      let pathPart = mention;
      let startLine, endLine;
      const rangeMatch = mention.match(/^(.+)\:(\d+)-(\d+)$/);
      if (rangeMatch) {
        pathPart = rangeMatch[1];
        startLine = parseInt(rangeMatch[2], 10);
        endLine = parseInt(rangeMatch[3], 10);
        /* P3-7: "0-5" parses to startLine=0 — slice(-1,5) would wrap and
         * return the LAST line. Clamp to 1 (line numbers are 1-based). */
        if (startLine < 1) startLine = 1;
        if (endLine < startLine) endLine = startLine;
      }
      /* Use the path as-is (sofuu.fs.readFile resolves relative to cwd). */
      let content = '';
      try {
        /* Reject paths escaping the project root: any '..' component, or an
         * absolute path that is not under cwd (matches the documented
         * "paths escaping the project root are rejected" contract). */
        const normalized = pathPart.startsWith('/') ? pathPart : (cwd + '/' + pathPart);
        const underRoot = normalized === cwd || normalized.startsWith(cwd + '/');
        if (pathPart.includes('..') || !underRoot) {
          blocks.push('### ' + pathPart + '\n```\n(rejected: path escapes project root)\n```');
          manifestParts.push('@' + pathPart + ' (rejected)');
          continue;
        }
        if (sofuu.fs && typeof sofuu.fs.readFile === 'function') {
          /* readFile is async — await it so missing files reject into the
           * catch below (→ "(error: …)" block) instead of leaking an
           * unhandled rejection and attaching "[object Promise]". */
          content = await sofuu.fs.readFile(pathPart, 'utf8');
        } else {
          /* Fallback: try process-level fs if available */
          content = '';
        }
        if (!content) {
          blocks.push('### ' + pathPart + '\n```\n(file not found or empty)\n```');
          manifestParts.push('@' + pathPart + ' (missing)');
          continue;
        }
        /* Apply line range if specified. */
        if (startLine !== undefined) {
          const lines = content.split('\n');
          const sliced = lines.slice(startLine - 1, endLine);
          content = sliced.join('\n');
        }
        const tok = estTok(content);
        const attachBudget = attachTokenBudget();
        if (totalTokens + tok > attachBudget) {
          const remaining = attachBudget - totalTokens;
          if (remaining > 0) {
            const lines = content.split('\n');
            let kept = 0;
            let keptTok = 0;
            for (const ln of lines) {
              const lt = estTok(ln);
              if (keptTok + lt > remaining) break;
              kept++; keptTok += lt;
            }
            content = lines.slice(0, kept).join('\n') +
              '\n…[trimmed ' + (lines.length - kept) + ' lines, ' + (tok - keptTok) + ' tk]';
            totalTokens += keptTok;
          } else {
            content = '…[budget exceeded]';
          }
        } else {
          totalTokens += tok;
        }
        const rangeLabel = startLine !== undefined ? (':' + startLine + '-' + endLine) : '';
        blocks.push('### ' + pathPart + rangeLabel + '\n```\n' + content + '\n```');
        manifestParts.push('@' + pathPart + rangeLabel + ' (' + (estTok(content)) + ' tk)');
      } catch (e) {
        blocks.push('### ' + pathPart + '\n```\n(error: ' + String(e.message || e) + ')\n```');
        manifestParts.push('@' + pathPart + ' (error)');
      }
    }
    const attached = blocks.length > 0
      ? '\n\nAttached files:\n' + blocks.join('\n\n')
      : '';
    return {
      text: text + attached,
      manifest: manifestParts.join(', '),
    };
  }

  /* ── F5: /share + /import (brain cards) ──────────────────────────── */
  async function handleShare(arg) {
    if (!cfg.brain) { out('\x1b[90m  Brain is off — /brain on to enable\x1b[0m\n'); return; }
    const brain = ensureDriverBrain();
    if (!brain) { out('\x1b[90m  Brain unavailable\x1b[0m\n'); return; }
    const parts = arg ? arg.split(/\s+/).filter(Boolean) : [];
    const path = parts[0] || (brainPath() + '.card.qtsq');
    const label = parts.slice(1).join(' ') || '';
    /* P2-18: only append .json when the user's path doesn't already carry
     * it — "/share card.qtsq" used to write "card.qtsq.json" (double
     * suffix). */
    const cardPath = /\.json$/i.test(path) ? path : path + '.json';
    try {
      const count = brain.count();
      if (count === 0) { out('\x1b[90m  Brain is empty — nothing to export\x1b[0m\n'); return; }
      /* Write a simple JSON card (the QTSQ session machinery is the long-term
       * format, but a JSON card works for v1 without extra FFI). */
      const card = {
        schema: 'sofuu-brain-card@1',
        exported_at: Math.floor(Date.now() / 1000),
        label: label,
        brain_path: brainPath(),
        record_count: count,
      };
      const cardJson = JSON.stringify(card, null, 2);
      /* Write the card metadata JSON. */
      if (sofuu.fs && typeof sofuu.fs.writeFile === 'function') {
        try { sofuu.fs.writeFile(cardPath, cardJson); }
        catch (e) { out('\x1b[33m  ⚠ Could not write card file: ' + String(e.message || e) + '\x1b[0m\n'); return; }
      }
      out('\x1b[90m  ⏺ exported ' + count + ' memories → ' + cardPath + '\x1b[0m');
      out('\x1b[90m  Card metadata written. Brain file: ' + brainPath() + '\x1b[0m');
      out('\x1b[90m  Share both files; import with /import ' + cardPath + '\x1b[0m\n');
    } catch (e) { out('\x1b[31m  ✗ ' + String(e.message || e) + '\x1b[0m\n'); }
  }
  async function handleImport(arg) {
    if (!arg) { out('  Usage: /import <path.qtsq>\n'); return; }
    if (!cfg.brain) { out('\x1b[90m  Brain is off — /brain on to enable\x1b[0m\n'); return; }
    const brain = ensureDriverBrain();
    if (!brain) { out('\x1b[90m  Brain unavailable\x1b[0m\n'); return; }
    try {
      if (!sofuu.fs || typeof sofuu.fs.readFile !== 'function') {
        out('\x1b[31m  ✗ fs.readFile unavailable\x1b[0m\n'); return;
      }
      let cardJson;
      try { cardJson = sofuu.fs.readFile(arg, 'utf8'); }
      catch (e) { out('\x1b[31m  ✗ Cannot read file: ' + arg + '\x1b[0m\n'); return; }
      if (!cardJson || typeof cardJson !== 'string') {
        out('\x1b[31m  ✗ Cannot read file (not a text file): ' + arg + '\x1b[0m\n'); return;
      }
      const card = JSON.parse(cardJson);
      if (!card || card.schema !== 'sofuu-brain-card@1') {
        out('\x1b[31m  ✗ Not a valid sofuu brain card\x1b[0m\n'); return;
      }
      /* For v1, we report what would be merged — actual vector merge needs
       * the records_json + vectors exports wired to JS. */
      out('\x1b[90m  ⏺ Card: ' + (card.label || '(unlabeled)') + ' · ' + card.record_count + ' records\x1b[0m');
      out('\x1b[90m  Source brain: ' + card.brain_path + '\x1b[0m');
      out('\x1b[90m  Full vector-level merge requires the source brain file.\x1b[0m\n');
    } catch (e) { out('\x1b[31m  ✗ ' + String(e.message || e) + '\x1b[0m\n'); }
  }

  /* ── F6: /cost ───────────────────────────────────────────────────── */
  function handleCost() {
    let breakdown;
    try { breakdown = JSON.parse(__chat_cost_breakdown()); } catch (e) { breakdown = null; }
    if (!breakdown || !breakdown.turns || breakdown.turns.length === 0) {
      out('\x1b[90m  No usage recorded yet this session.\x1b[0m\n');
      return;
    }
    out('\n\x1b[1m  Cost breakdown\x1b[0m');
    for (const t of breakdown.turns) {
      /* P6.4: show prefix-cache hits when the provider reported them. */
      const cache = (t.cache_read_tokens > 0 || t.cache_write_tokens > 0)
        ? '  \x1b[2m· cache ' + fmtTk(t.cache_read_tokens) + ' hit' +
          (t.cache_write_tokens > 0 ? ' / ' + fmtTk(t.cache_write_tokens) + ' write' : '') + '\x1b[0m'
        : '';
      out('  \x1b[90m#' + t.turn + '\x1b[0m  ' + t.model +
          '  \x1b[2min ' + fmtTk(t.input_tokens) + ' · out ' + fmtTk(t.output_tokens) + '\x1b[0m' + cache +
          '  \x1b[1m$' + t.cost_usd.toFixed(6) + '\x1b[0m');
    }
    out('\n  \x1b[1mSession total:\x1b[0m $' + breakdown.session_total.toFixed(6));
    if (breakdown.cache_read_total > 0) {
      out('  \x1b[2mCache hits this session: ' + fmtTk(breakdown.cache_read_total) + ' tk\x1b[0m');
    }
    out('  \x1b[1mLifetime total:\x1b[0m $' + breakdown.lifetime_total.toFixed(6));
    /* Budget status */
    let budget;
    try { budget = JSON.parse(__chat_budget_check()); } catch (e) { budget = null; }
    if (budget && budget.budget > 0) {
      out('  \x1b[1mBudget:\x1b[0m $' + budget.spent.toFixed(6) + ' / $' + budget.budget.toFixed(2) +
          ' (' + budget.pct + '%)');
      if (budget.pct >= 80) out('\x1b[33m  ⚠ Approaching budget cap\x1b[0m');
    }
    out('');
  }

  /* ── F8: /watch ──────────────────────────────────────────────────── */
  let watchChangesPending = [];
  function handleWatch(arg) {
    if (!arg) {
      let list;
      try { list = JSON.parse(__chat_watch_list()); } catch (e) { list = []; }
      if (!list.length) {
        out('\x1b[90m  No paths being watched. Usage: /watch <path>\x1b[0m\n');
      } else {
        out('\x1b[1m  Watching:\x1b[0m');
        for (const p of list) out('  \x1b[36m' + p + '\x1b[0m');
        out('\n\x1b[90m  /watch <path> to add · /watch off to clear\x1b[0m\n');
      }
      return;
    }
    if (arg === 'off' || arg === 'clear') {
      __chat_watch_clear();
      out('\x1b[90m  ✓ Watcher cleared\x1b[0m\n');
      return;
    }
    __chat_watch_add(arg);
    out('\x1b[90m  ✓ Watching ' + arg + ' (changes surface in chat context)\x1b[0m\n');
  }

  /* ── F9: ~/.sofuu/hooks.js — user middleware ─────────────────────── */
  let hooks = null;
  let hooksLoadAttempted = false;
  let hooksFailCount = 0;
  async function loadHooks() {
    if (hooksLoadAttempted) return;
    hooksLoadAttempted = true;
    /* ~/.sofuu/hooks.js — the user's own middleware file. */
    const home = env('HOME') || '.';
    const hooksPath = home + '/.sofuu/hooks.js';
    /* Check if the file exists — we can't use dynamic import directly in
     * QuickJS without ESM; instead, read + eval it. */
    try {
      if (!sofuu.fs || typeof sofuu.fs.readFile !== 'function') return;
      let code;
      try { code = await sofuu.fs.readFile(hooksPath, 'utf8'); }
      catch (e) { return; } /* file doesn't exist — no hooks, zero overhead */
      if (!code || typeof code !== 'string' || code.trim().length === 0) return;
      /* Wrap in a function that returns the pre/post exports. */
      const wrapped = '(function() { var module = { exports: {} }; var exports = module.exports; ' +
        code + '\n; return module.exports; })()';
      const result = eval(wrapped);
      if (result && (typeof result.pre === 'function' || typeof result.post === 'function')) {
        hooks = result;
        out('\x1b[90m  ⏺ hooks.js loaded\x1b[0m');
      }
    } catch (e) {
      out('\x1b[33m  ⚠ hooks.js load failed: ' + String(e.message || e) + '\x1b[0m');
    }
  }
  async function runHookPre(text) {
    if (!hooks || typeof hooks.pre !== 'function') return text;
    try {
      // Redacted cfg: hooks get everything EXCEPT secrets (api_key,
      // providers[].api_key). A malicious hooks.js could otherwise exfil
      // keys with one fetch — local file-write = key theft.
      var safeCfg = {};
      try {
        safeCfg = JSON.parse(JSON.stringify(cfg || {}));
        delete safeCfg.api_key;
        if (Array.isArray(safeCfg.providers)) {
          for (var hi = 0; hi < safeCfg.providers.length; hi++) {
            if (safeCfg.providers[hi]) delete safeCfg.providers[hi].api_key;
          }
        }
      } catch (eRedact) { safeCfg = {}; }
      const result = await Promise.race([
        hooks.pre({ text: text, cfg: safeCfg }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 3000)),
      ]);
      if (typeof result === 'string') return result;
      if (result && typeof result === 'object') {
        if (result.skip) return null;
        if (typeof result.text === 'string') return result.text;
      }
      return text;
    } catch (e) {
      hooksFailCount++;
      out('\x1b[33m  ⚠ hook pre() error: ' + String(e.message || e) + '\x1b[0m');
      if (hooksFailCount >= 3) {
        out('\x1b[33m  ⚠ hooks disabled after 3 failures\x1b[0m');
        hooks = null;
      }
      return text;
    }
  }
  async function runHookPost(text, answer, usage) {
    if (!hooks || typeof hooks.post !== 'function') return answer;
    try {
      const result = await Promise.race([
        hooks.post({ text: text, answer: answer, usage: usage }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 3000)),
      ]);
      if (typeof result === 'string') return result;
      return answer;
    } catch (e) {
      hooksFailCount++;
      out('\x1b[33m  ⚠ hook post() error: ' + String(e.message || e) + '\x1b[0m');
      if (hooksFailCount >= 3) {
        out('\x1b[33m  ⚠ hooks disabled after 3 failures\x1b[0m');
        hooks = null;
      }
      return answer;
    }
  }

  /* ── F10: ghost completion ────────────────────────────────────────── */
  /* Ghost completion is driven by the readline key handler. The driver
   * installs a global __chat_ghost_check(prefix) that returns a suffix
   * or ''. It uses sofuu.ai.embedLocal (sync) + brain.recall. */
  let ghostLastPrefix = '';
  let ghostLastResult = '';
  let ghostBusyUntil = 0;
  globalThis.__chat_ghost_check = function(prefix) {
    if (!cfg.ghost || !cfg.brain) return '';
    if (prefix.length < 8) return '';
    /* Debounce: return cached result if prefix hasn't changed enough. */
    if (prefix === ghostLastPrefix) return ghostLastResult;
    /* Only re-check if the last check was > 250ms ago. (P3-3: this used to
     * test a ghostTimer that was never assigned — the debounce was dead —
     * so every keystroke > 8 chars re-embedded. The prefix-equality cache
     * above still holds; this bounds the rest.) */
    const now = Date.now();
    if (now < ghostBusyUntil) return ghostLastResult;
    ghostBusyUntil = now + 250;
    ghostLastPrefix = prefix;
    try {
      if (!sofuu.ai || typeof sofuu.ai.embedLocal !== 'function') return '';
      const vec = sofuu.ai.embedLocal(prefix);
      if (!vec) return '';
      const brain = ensureDriverBrain();
      if (!brain) return '';
      const results = brain.recall(new Float32Array(vec), 1);
      if (!results || results.length === 0) { ghostLastResult = ''; return ''; }
      const hit = results[0];
      if (!hit || !hit.text || hit.role !== 'user') { ghostLastResult = ''; return ''; }
      /* The hit text should be a past user prompt. Check cosine ≥ 0.9. */
      if ((hit.score || 0) < 0.9) { ghostLastResult = ''; return ''; }
      /* Return the suffix (the part after the prefix). */
      const fullText = hit.text;
      if (fullText.startsWith(prefix)) {
        const suffix = fullText.slice(prefix.length);
        ghostLastResult = suffix;
        return suffix;
      }
      ghostLastResult = '';
      return '';
    } catch (e) { ghostLastResult = ''; return ''; }
  };

  /* ── F9: hooks init (called lazily on first turn) ────────────────── */
  async function ensureHooks() {
    if (!hooksLoadAttempted) await loadHooks();
  }

  // Interactive terminal? (drives spinner animation / inline redraws).
  const TTY = (typeof __chat_is_tty === 'function') ? !!__chat_is_tty() : false;
  const SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  /* ── Agent-phase indicator (one line above the input box) ────────
   * CURRENT_PHASE is the label shown ("Thinking"/"Coding"/…), driven by
   * the real onStep events from the agent loop (see phaseForStep below —
   * never guesswork). null = no turn in progress = row hidden. */
  let CURRENT_PHASE = null;
  let PHASE_FRAME = 0; /* index of the dim "cursor" letter in CURRENT_PHASE */
  /* ── Token-aware history trimming + auto-compaction (P5) ─────────
   * The entry cap (MAX_HISTORY_ENTRIES) is only a safety net. The real
   * limit is the context window: past COMPACT_AT of the budget, old turns
   * are SUMMARIZED into one system entry instead of dropped (info kept);
   * if that fails (or usage still exceeds), the old drop-oldest path
   * guards the window. estimateTokens ≈ chars/4; fallback is same math. */
  const COMPACT_AT = 0.70;        /* trigger auto-compaction at 70% of budget */
  const COMPACT_KEEP_TURNS = 4;   /* verbatim turns kept by auto-compaction */
  let autoCompactArmed = true;    /* one shot per threshold crossing */

  /* Capability lookup for the active model (native registry, rt/model_caps,
   * plus the discovered store). Second arg: the ENDPOINT this request will
   * hit — when present, caps the endpoint itself published for this exact
   * model override the name-keyed registry. Returns null for models nobody
   * knows — callers apply their own fallback. */
  function modelCaps() {
    try {
      if (sofuu.ai && typeof sofuu.ai.modelCaps === 'function') {
        const v = applyFailover(cfg);
        const c = JSON.parse(sofuu.ai.modelCaps(v.model || '', v.base_url || undefined));
        return (c && c.known) ? c : null;
      }
    } catch (e) {}
    return null;
  }
  /* Evidence-ladder resolve (same native the shipped chat.js uses):
   * learned > endpoint-discovered > registry > conservative default. An
   * INHERITED config number shrinks to the strictest real evidence (a
   * stale global can never shadow the selected model's real window); an
   * EXPLICIT one (`/ctx n` typed this session) is obeyed as typed, with
   * the evidence reported as an advisory. Every budget consumer in this
   * driver (footer meter, compaction, attachments, output cap, the agent
   * def) routes through here. */
  function resolveCaps() {
    const c = applyFailover(cfg);
    try {
      if (sofuu.ai && typeof sofuu.ai.resolveCaps === 'function') {
        const r = JSON.parse(sofuu.ai.resolveCaps(
          c.model || '', c.base_url || '',
          c.ctx_window > 0 ? c.ctx_window : 0,
          c.max_output > 0 ? c.max_output : 0,
          c.ctx_window_explicit === true));
        if (r && r.window > 0) return r;
      }
    } catch (e) {}
    /* Legacy ladder when the native is absent: config, capped by any
     * caps evidence (registry/discovered), then the per-provider floor.
     * An explicit value keeps the caller's number. */
    const capsL = modelCaps() || {};
    const explicit = c.ctx_window_explicit === true;
    let win = c.ctx_window > 0 ? c.ctx_window
      : (capsL.ctxWindow > 0 ? capsL.ctxWindow
      : ({ openai: 128000, anthropic: 128000, local: 32768 }[c.provider] || 32768));
    if (!explicit && capsL.ctxWindow > 0 && win > capsL.ctxWindow) win = capsL.ctxWindow;
    let mx = c.max_output > 0 ? c.max_output
      : (capsL.maxOutput > 0 ? capsL.maxOutput : 0);
    if (capsL.maxOutput > 0 && mx > capsL.maxOutput) mx = capsL.maxOutput;
    return { window: win, maxOutput: mx, known: !!(capsL.known), source: 'legacy',
             clampedConfig: false, thinking: (capsL.thinking || 'unknown') };
  }

  /* ── Caps discovery (provider-agnostic) ─────────────────────────
   * Same contract as shipped chat.js: the active endpoint's model listing
   * carries its models' real limits; harvest it into the alloc gate's
   * discovered store (keyed by API root + model in Rust — any provider,
   * any field spelling). TTL-gated, best-effort, silent on failure. */
  const DISCOVER_TTL_MS = 7 * 24 * 3600 * 1000;
  /* P3 (AUDIT-2026-09-07): failed discovery harvests retry on this short
   * clock instead of being cached for the full TTL. */
  const DISCOVER_RETRY_MS = 10 * 60 * 1000;
  let capsDiscoveredAt = 0;
  function isLocalHost(u) {
    return /^https?:\/\/(localhost|127\.|0\.0\.0\.0|\[::1\]|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)/i.test(String(u || ''));
  }
  function modelsRootUrl(endpoint) {
    let u = String(endpoint || '');
    u = u.split('#')[0].split('?')[0];
    while (/\/(chat\/completions|completions|messages)\/?$/.test(u)) {
      u = u.replace(/\/(chat\/completions|completions|messages)\/?$/, '');
    }
    if (u.charAt(u.length - 1) === '/') u = u.slice(0, -1);
    return u + '/models';
  }
  function discoverCaps() {
    try {
      if (!cfg.model || !cfg.base_url) return;
      if (isLocalHost(cfg.base_url)) return;
      if (Date.now() - capsDiscoveredAt < DISCOVER_TTL_MS) return;
      /* P3 (AUDIT-2026-09-07): the full TTL used to be stamped BEFORE the
       * harvest, so a fully failed run (network down, 401s) was cached for
       * 7 days like a success. A short retry clock goes up front (still
       * throttles re-entry while in flight — a dead endpoint is not
       * hammered every turn); the full TTL lands only when ≥1 listing
       * actually succeeded. */
      capsDiscoveredAt = Date.now() - DISCOVER_TTL_MS + DISCOVER_RETRY_MS;
      /* Harvest the ACTIVE endpoint AND every other configured provider
       * with a distinct endpoint — cross-root fallback needs their
       * listings to know the same model's real caps when the active
       * endpoint publishes none. Provider-agnostic: whatever is
       * configured, nothing hardcoded. */
      const targets = [cfg.base_url];
      if (Array.isArray(cfg.providers)) {
        for (const pe of cfg.providers) {
          if (pe && pe.endpoint && pe.endpoint !== cfg.base_url &&
              targets.indexOf(pe.endpoint) < 0) {
            targets.push(pe.endpoint);
          }
        }
      }
      const jobs = [];
      for (let ti = 0; ti < targets.length; ti++) {
        const endpoint = targets[ti];
        let apiKey = ti === 0 ? cfg.api_key : '';
        if (ti > 0 && Array.isArray(cfg.providers)) {
          const pe = cfg.providers.filter(p => p && p.endpoint === endpoint)[0];
          if (pe) apiKey = pe.api_key || '';
        }
        const headers = {};
        if (apiKey) headers['Authorization'] = 'Bearer ' + apiKey;
        jobs.push(
          fetch(modelsRootUrl(endpoint), { headers: headers })
            .then(res => (res && res.ok) ? res.json() : null)
            .then(j => ({ endpoint: endpoint, listing: j }))
            .catch(() => ({ endpoint: endpoint, listing: null }))
        );
      }
      Promise.all(jobs).then(function (results) {
        let anyOk = false;
        for (const r of results) {
          if (!r || !r.listing) continue;
          anyOk = true;
          if (sofuu.ml && sofuu.ml.alloc && typeof sofuu.ml.alloc.ingestListing === 'function') {
            try { sofuu.ml.alloc.ingestListing(r.endpoint, JSON.stringify(r.listing)); } catch (eI) {}
          }
        }
        if (anyOk) capsDiscoveredAt = Date.now();
      }).catch(function () {});
    } catch (e) {}
  }

  /* ── F2: detect-on-select ──────────────────────────────────────────
   * Force a harvest for the ACTIVE endpoint right after a model/provider
   * switch and report the detected window in one line. Unlike
   * discoverCaps() this ignores the TTL (the user just asked about this
   * model) and never stays silent: either the endpoint published caps, or
   * it did not, and both outcomes are worth saying. */
  async function detectCapsNow() {
    try {
      const c = applyFailover(cfg);
      if (!c.model || !c.base_url) return;
      const local = isLocalHost(c.base_url);
      const endpoint = c.base_url;
      /* Show the API root, not the full chat/completions URL. */
      let shown = String(endpoint).replace(/\/(chat\/completions|completions|messages)\/?$/, '');
      if (shown.length > 72) shown = shown.slice(0, 69) + '…';
      const headers = {};
      if (c.api_key) headers['Authorization'] = 'Bearer ' + c.api_key;
      let listing = null;
      try {
        const res = await fetch(modelsRootUrl(endpoint), { headers: headers });
        if (res && res.ok) listing = await res.json();
      } catch (eF) { listing = null; }
      let ingested = 0;
      if (listing && sofuu.ml && sofuu.ml.alloc && typeof sofuu.ml.alloc.ingestListing === 'function') {
        try { ingested = sofuu.ml.alloc.ingestListing(endpoint, JSON.stringify(listing)) || 0; } catch (eI) { ingested = 0; }
      }
      /* Stamp the TTL so the per-turn harvest does not re-fetch what we
       * just pulled (a failed pull keeps the short retry clock). */
      capsDiscoveredAt = Date.now() - DISCOVER_TTL_MS + (ingested > 0 ? DISCOVER_TTL_MS : DISCOVER_RETRY_MS);
      const r = resolveCaps();
      const known = r && r.known;
      if (!listing) {
        if (local) {
          out('\x1b[90m  local endpoint — no model list to read; context is detected from its first response (default ' + fmtTk(r.window) + ')\x1b[0m\n');
        } else {
          out('\x1b[33m  ⚠ could not read the model list from ' + shown + ' (auth, network, or no /models) — using ' + fmtTk(r.window) + ' context\x1b[0m\n');
        }
      } else if (ingested > 0 && known) {
        out('\x1b[32m  ✓ detected ' + fmtTk(r.window) + ' context for ' + c.model + ' (' + (r.source || 'evidence') + ', ' + ingested + ' model' + (ingested === 1 ? '' : 's') + ')\x1b[0m\n');
      } else if (ingested > 0) {
        out('\x1b[33m  ⚠ ' + shown + ' lists ' + ingested + ' models but publishes no limits for ' + c.model + ' — using ' + fmtTk(r.window) + ' (set /ctx explicitly if you know better)\x1b[0m\n');
      } else if (local) {
        out('\x1b[90m  local endpoint — ' + shown + ' publishes no limits for ' + c.model + '; using ' + fmtTk(r.window) + ' until its first response teaches us\x1b[0m\n');
      } else {
        out('\x1b[33m  ⚠ ' + shown + ' publishes no limits for ' + c.model + ' — using ' + fmtTk(r.window) + ' (set /ctx explicitly if you know better)\x1b[0m\n');
      }
      refreshStatus('');
    } catch (e) {}
  }

  function ctxBudget() {
    /* The ladder resolves the SELECTED model's real window (config can
     * only shrink it); the budget is 85% of that. Never a flat constant. */
    return Math.floor(resolveCaps().window * 0.85);
  }
  /* @file mention attachments scale with the same window (~25% of it). */
  function attachTokenBudget() {
    /* alloc gate (§14): the plan's attachment budget comes first — tight
     * sessions attach less, slack sessions attach more. Legacy ratio when
     * ML is off. */
    const planA = attachPlanCache();
    if (planA && planA.attachBudgetTok > 0) return planA.attachBudgetTok;
    return Math.min(65536, Math.max(2048, Math.floor(resolveCaps().window * 0.25)));
  }
  /* ── alloc gate (PLAN-ML-GATES §14): model-aware config allocation ──
   * Layer 2: a tiny net measures context pressure from this session's live
   * state — the CALIBRATED overhead meter (ctxOverhead) is its best
   * feature — and a deterministic clamped policy maps it to per-turn
   * config: compactAt (replaces the fixed cliff), attach budget, output
   * reserve. The model's capabilities are resolved INSIDE Rust (registry →
   * learned limits → conservative defaults); JS never supplies caps.
   * Returns null when ML is off (/ml off, SOFUU_NO_ML=1) or unavailable —
   * every consumer keeps the legacy ratios exactly as before. */
  function allocPlanChat(taskTk, attachTk) {
    if (!cfg.ml) return null;
    try {
      if (!sofuu.ml || !sofuu.ml.alloc || typeof sofuu.ml.alloc.plan !== 'function') return null;
      const sizes = [];
      let histTk = 0;
      for (const m of history) {
        const t = estTok(String(m.content || ''));
        sizes.push(t); histTk += t;
      }
      let growthTk = 0, accel = 0;
      if (sizes.length >= 2) {
        const last3 = sizes.slice(-3);
        growthTk = last3.reduce((a, b) => a + b, 0) / last3.length;
        if (sizes.length >= 4) {
          const half = Math.floor(sizes.length / 2);
          const m1 = sizes.slice(0, half).reduce((a, b) => a + b, 0) / half;
          const m2 = sizes.slice(half).reduce((a, b) => a + b, 0) / (sizes.length - half);
          if (m1 > 0) accel = m2 / m1;
        }
      }
      const p = JSON.parse(sofuu.ml.alloc.plan(JSON.stringify({
        model: cfg.model || '',
        /* The endpoint the request will hit — caps it published for THIS
         * exact model (discovered store) take precedence over the
         * name-keyed registry inside Rust. */
        baseUrl: cfg.base_url || '',
        cfgWindow: cfg.ctx_window > 0 ? cfg.ctx_window : 0,
        cfgMaxOutput: cfg.max_output > 0 ? cfg.max_output : 0,
        overheadTk: ctxOverhead >= 0 ? ctxOverhead : 0,
        calibrated: ctxOverhead >= 0,
        historyTk: histTk,
        turns: Math.floor(history.length / 2),
        /* Chat history carries no tool transcripts between turns — 0. */
        toolFrac: 0,
        summary: history.length > 0 && history[0].role === 'system' &&
          String(history[0].content || '').indexOf('Prior conversation summary') === 0,
        growthTk: growthTk, growthAccel: accel,
        taskTk: taskTk || 0, attachTk: attachTk || 0,
        /* Chat turns always offer the full tool set. */
        toolHeavy: true,
        writing: false,
      })));
      return (p && p.window > 0) ? p : null;
    } catch (e) { return null; }
  }
  /* expandMentions calls attachTokenBudget per mention parse — cache ONE
   * plan per turn (invalidated at turn entry) instead of re-running the
   * net per @file. */
  let allocTurnPlan = null, allocPlanValid = false;
  function attachPlanCache() {
    if (!allocPlanValid) { allocTurnPlan = allocPlanChat(0, 0); allocPlanValid = true; }
    return allocTurnPlan;
  }
  /* Allocation notes already shown this session (deduped — a clamped
   * /maxout note prints once, not every turn). */
  const allocNotedNotes = {};
  function estTok(s) {
    if (sofuu.ai && typeof sofuu.ai.estimateTokens === 'function') {
      try { return sofuu.ai.estimateTokens(s) | 0; } catch (e) {}
    }
    return Math.ceil(String(s).length / 4);
  }
  /* ── Turn blocks (Claude-style retention, 2026-09-03 e) ────────────
   * History is a list of TURN BLOCKS: one user message, then zero or more
   * retained tool_calls/tool messages (the turn's transcript), then the
   * assistant answer. Trimming/compaction moves whole blocks — an
   * orphaned tool message (its assistant tool_calls pair dropped) is a
   * hard 400 at OpenAI-compatible providers. */
  function turnStartAt(i) {
    let j = Math.max(0, Math.min(i, history.length - 1));
    while (j > 0 && history[j].role !== 'user') j--;
    return j;
  }
  function removeTurnBlock(i) {
    const s = turnStartAt(i);
    let e = s + 1;
    while (e < history.length && history[e].role !== 'user') e++;
    history.splice(s, e - s);
  }
  /* A turn the compaction gate may never delete on its own authority.
   * LEXICAL and model-free on purpose: the user asked a question, gave an
   * instruction, or recorded a decision/approval. Mirrors the shipped
   * chat.js guard exactly — both drivers must enforce the same floor. */
  const PROTECT_RE = /(\?|^\s*(please\s+)?(do\s+not|don't|never|always|must|make\s+sure|keep)\b)|(\b(approved|approve|accepted|rejected|decided|decision|agreed|sign\s*off)\b)/i;
  function isProtectedBlock(hist, idx) {
    const s = turnStartAt(idx);
    if (s < 0 || s >= hist.length) return false;
    let e = s + 1;
    while (e < hist.length && hist[e].role !== 'user') e++;
    for (let i = s; i < e; i++) {
      const m = hist[i];
      /* Only the USER's own words protect the turn. */
      if (m && m.role === 'user' && PROTECT_RE.test(String(m.content || ''))) return true;
    }
    return false;
  }
  function historyTokens() {
    let n = 0;
    for (const m of history) {
      n += estTok(String(m.content || ''));
      /* Retained transcripts bill their tool_calls JSON too — the provider
       * counts the whole assistant message, not just its (null) content. */
      if (m.tool_calls) { try { n += estTok(JSON.stringify(m.tool_calls)); } catch (eTC) {} }
    }
    return n;
  }
  /* Summarize all but the last keepTurns turns into one system entry
   * (same shape manual /compact has always produced). Returns false when
   * there is nothing to fold or the summarizer failed/returned junk —
   * callers fall back to dropping, never block the turn. */
  async function summarizeHistory(keepTurns, opts) {
    /* Block-aware split point: index where the (keepTurns+1)-th-from-last
     * turn block starts. Everything before it is folded — whole blocks,
     * so a block's retained tool transcript never straddles the fold. */
    let foldFrom = history.length;
    let seen = 0;
    for (let fi = history.length - 1; fi >= 0; fi--) {
      if (history[fi].role === 'user') {
        seen++;
        if (seen > keepTurns) { foldFrom = fi; break; }
      }
    }
    /* Nothing to fold only when EVERY block is kept (the scan never moved
     * foldFrom) — foldFrom === 0 is the normal two-block case and must fold. */
    if (foldFrom >= history.length) return false;
    const oldCount = foldFrom;
    /* opts.prompt overrides the stock compaction instruction (resume uses
     * its own framing); opts.source labels the archive record honestly
     * instead of stamping everything 'compaction'. Returns the summary
     * text on success (truthy — existing `if (await …)` callers keep
     * working), false when there is nothing usable. */
    const sysPrompt = (opts && opts.prompt) ||
      'You are a conversation summarizer. Compress the following conversation into a compact summary that preserves key facts, decisions, and the user\'s intent. Output only the summary.';
    /* Strict gateways reject a request whose last message is not
     * role=user (2026-10-09: every auto + manual compaction on tokenrouter
     * died with "The last message must have role=user" — the folded prefix
     * ends with an assistant message). The instruction rides LAST as a user
     * message: system-first stays for lenient providers, and
     * instruction-last is the better summary shape anyway. */
    const tailAsk = { role: 'user', content: (opts && opts.tailPrompt) ||
      'Summarize the conversation above into a compact summary that preserves key facts, decisions, and the user\'s intent. Output only the summary.' };
    const summary = await complete([
      { role: 'system', content: sysPrompt },
      ...history.slice(0, oldCount),
      tailAsk,
    ]);
    if (!summary || typeof summary !== 'string' || !summary.trim() || summary.trim() === '(no response)') return false;
    archiveWrite('summary', 'complete', (opts && opts.source) || 'compaction', summary, {
      compacted_messages: oldCount,
      kept_turns: keepTurns,
    });
    history = [{ role: 'system', content: 'Prior conversation summary: ' + summary },
               ...history.slice(oldCount)];
    return summary.trim();
  }
  async function trimHistory() {
    /* Hard safety net: the absolute entry cap — shed whole turn blocks from
     * the front so retained tool transcripts are never orphaned. */
    if (history.length > MAX_HISTORY_ENTRIES) {
      let over = history.length - MAX_HISTORY_ENTRIES;
      while (over > 0 && history.length > 0) {
        const before0 = history.length;
        removeTurnBlock(0);
        over -= (before0 - history.length);
      }
    }
    const budget = ctxBudget();
    /* alloc gate (§14): the compaction cliff moves with context pressure —
     * 0.70 of budget when the session is slack, down to 0.50 when the net
     * sees overflow coming within ~3 turns. Fixed cliff when ML is off. */
    const planC = allocPlanChat(0, 0);
    const compactAt = (planC && planC.compactAt > 0) ? planC.compactAt : COMPACT_AT;
    /* Re-arm one-shot compaction once usage drains back below half. */
    if (!autoCompactArmed && historyTokens() < budget * 0.5) autoCompactArmed = true;
    /* PLAN-ML-GATES §12: ML compaction gate — opportunistic passes that
     * free MECHANICAL junk (duplicates, boilerplate, re-fetchable reads)
     * before the one-shot cliff ever fires. Runs while usage is past
     * half the budget. The net SELECTS — only free-tier segments are
     * dropped here, no LLM calls. History stays strictly alternating,
     * so a turn goes only when BOTH its user prompt and assistant
     * answer are flagged; one-sided junk waits for the cliff. Turns go
     * oldest-first until usage drains below half the budget (the pass
     * budget is the whole flagged set — the drain target is what keeps
     * passes small). System summaries are never candidates. The cliff
     * below + the drop-oldest guard remain the safety nets; with the
     * gate off (/ml off or SOFUU_NO_ML=1) behaviour is exactly as
     * before. */
    if (cfg.ml && history.length > 4 && historyTokens() > budget * 0.5) {
      try {
        if (sofuu.ml && sofuu.ml.compaction && typeof sofuu.ml.compaction.plan === 'function') {
          const segs = [], segHist = [];
          for (let i = 0; i < history.length; i++) {
            const m = history[i];
            if (m.role !== 'user' && m.role !== 'assistant') continue;
            const text = String(m.content || '');
            segs.push({ text: text, tokens: estTok(text), age: 0,
                        kind: m.role === 'user' ? 0 : 1,
                        retrievable: false, compacted: false });
            segHist.push(i);
          }
          for (let j = 0; j < segs.length; j++) segs[j].age = segs.length - 1 - j;
          if (segs.length > 4) {
            let task = '';
            for (let i = history.length - 1; i >= 0; i--) {
              if (history[i].role === 'user') { task = String(history[i].content || ''); break; }
            }
            const recentTxt = history.slice(-4).map(m => String(m.content || '')).join('\n');
            const plan = JSON.parse(sofuu.ml.compaction.plan(JSON.stringify({
              task: task, summary: '', recent: recentTxt, segments: segs,
            }), JSON.stringify({ budget: historyTokens() })), '{}');
            const compact = plan.compact || [], tiers = plan.tiers || [];
            const flaggedFree = new Set();
            for (let c = 0; c < compact.length; c++) {
              const t = tiers[c];
              if (t !== 'dup' && t !== 'boilerplate' && t !== 'retrievable') continue;
              const hi = segHist[compact[c]];
              if (hi !== undefined) flaggedFree.add(hi);
            }
            /* Complete turn BLOCKS only (user+assistant both flagged),
             * never the newest block; oldest first, until usage drains
             * below half. Removal is block-atomic — a flagged block's
             * retained tool transcript leaves with it. */
            const blockStarts = [];
            for (let bi = 0; bi < history.length - 1; bi++) {
              if (history[bi].role !== 'user') continue;
              let bAns = -1;
              for (let bj = bi + 1; bj < history.length; bj++) {
                if (history[bj].role === 'user') break;
                if (history[bj].role === 'assistant' && !history[bj].tool_calls) { bAns = bj; break; }
              }
              if (bAns >= 0 && flaggedFree.has(bi) && flaggedFree.has(bAns)) blockStarts.push(bi);
            }
            let freed = 0, droppedTurns = 0;
            /* blockObjs hold the block user-message OBJECTS — after each
             * removal the array shifts, so re-locate by identity. */
            const blockObjs = blockStarts.map(bx => history[bx]);
            /* SAFETY CAP (2026-09-25): one pass may never gut the session.
             * Without it the only stop condition is "usage below half the
             * budget", which on a long history authorises deleting dozens
             * of turns whenever the gate is wrong. A quarter of the
             * complete blocks, never the two most recent. */
            /* count complete turns (a user message + its answer) */
            let completeBlocks = 0;
            for (let cb = 0; cb < history.length - 1; cb++) {
              if (history[cb].role === 'user') completeBlocks++;
            }
            const cap0 = Math.max(1, Math.floor(completeBlocks * 0.25));
            const maxDrops = (completeBlocks - cap0 < 2)
              ? Math.max(0, completeBlocks - 2) : cap0;
            let skippedProtected = 0;
            for (let b = 0; b < blockObjs.length && droppedTurns < maxDrops &&
                   historyTokens() > budget * 0.5; b++) {
              const blockIdx = history.indexOf(blockObjs[b]);
              if (blockIdx < 0) continue; /* already removed */
              /* Model-independent guard — the gate is the component under
               * suspicion, so the last line of defence cannot be the gate. */
              if (isProtectedBlock(history, blockIdx)) { skippedProtected++; continue; }
              const beforeBlk = historyTokens();
              removeTurnBlock(blockIdx);
              freed += beforeBlk - historyTokens();
              droppedTurns++;
            }
            if (droppedTurns > 0) {
              out('\x1b[90m  ml-compaction freed ' + fmtTk(freed) + ' tk (' +
                  droppedTurns + ' of ' + completeBlocks + ' turn' +
                  (completeBlocks === 1 ? '' : 's') +
                  ': dup/boilerplate/re-fetchable' +
                  (skippedProtected > 0 ? '; ' + skippedProtected + ' protected kept' : '') +
                  ')\x1b[90m\x1b[0m');
            }
          }
        }
      } catch (e) { /* the gate advises — any failure falls through to the cliff */ }
    }
    /* P5: auto-compaction — when the 1M (or model) window is nearly full,
     * summarize the whole context and reset usage to near-zero so the
     * session can continue indefinitely. Per-answer 32k thinking / 64k
     * output are *not* session caps — total thinking+output across many
     * turns can exceed them, only the per-answer limit is enforced. */
    if (autoCompactArmed && historyTokens() > budget * compactAt && history.length > 2) {
      autoCompactArmed = false;
      const savedTk = historyTokens();
      try {
        // Summarize *all* turns and keep only the summary → window → ~0
        if (await summarizeHistory(0)) {
          const m0 = Math.max(0, savedTk + (ctxOverhead >= 0 ? ctxOverhead : 0));
          usedCtx = ctxMeter();
          out('\x1b[90m  ⚙ auto-compacted → summary (' +
              fmtTk(Math.max(0, savedTk - historyTokens())) + ' tk saved, meter ' +
              fmtTk(m0) + '→' + fmtTk(usedCtx) + ')\x1b[0m');
        }
      } catch (e) {
        /* A failed auto-compact must not be silent: the drop-oldest guard
         * below will eat turns with no summary kept, which reads as
         * context loss. One dim line names the cause. */
        out('\x1b[90m  [auto-compact failed: ' + String(e.message || e) + ' — history will trim instead]\x1b[0m\n');
      }
    }
    /* Drop-oldest guard: still over budget after compaction (or compaction
     * failed)? Shed oldest whole turn blocks — never the newest block, and
     * never an orphaned tool message (hard 400 at the provider). */
    let droppedTurns = 0;
    while (history.length > 0 && historyTokens() > budget) {
      const lastStart = turnStartAt(history.length - 1);
      if (lastStart === 0) break; /* only the newest block left */
      const beforeDrop = historyTokens();
      removeTurnBlock(0);
      if (historyTokens() >= beforeDrop) break; /* safety: no progress */
      droppedTurns++;
    }
    if (droppedTurns > 0) {
      out('\x1b[33m  ⚠ history trimmed by ' + droppedTurns + ' turn' + (droppedTurns === 1 ? '' : 's') +
          ' to fit the ' + Math.floor(ctxBudget() / 100) / 10 + 'k-token context budget\x1b[0m');
    }
  }
  function streamOpts() {
    const c = applyFailover(cfg);
    const o = { messages: [], provider: c.provider, model: c.model };
    if (c.effort) o.effort = c.effort;
    if (c.api_key) o.api_key = c.api_key;
    if (c.base_url) o.base_url = c.base_url;
    if (c.profile) o.profile = c.profile;
    /* Output cap from the ladder: config honored only as far as the
     * strictest real evidence for THIS model allows. A number the ladder
     * DEFAULTED (model unknown everywhere) is never sent — the endpoint
     * applies its own default (Pass-31 rule). Layer 0 re-clamps on the
     * wire — this keeps the driver consistent with it. */
    const capR = resolveCaps();
    if (capR.maxOutput > 0 && capR.maxSource !== 'default') o.max_tokens = capR.maxOutput;
    return o;
  }
  async function complete(messages, opts) {
    const o = streamOpts(); o.messages = messages;
    if (opts && opts.tools) o.tools = opts.tools;
    const r = await sofuu.ai.complete(o);
    if (r && r.toolCalls) return r;
    return r && r.text ? r.text.trim() : '(no response)';
  }
  /* Known built-in providers. An unknown provider name is treated as
   * "custom" by the AI layer and needs a base URL. */
  const KNOWN = ['openai', 'anthropic', 'local'];
  function usableConfig(view) {
    const v = view || cfg;
    if (!v.provider || !v.model) return false;
    if (KNOWN.indexOf(v.provider) < 0 && !v.base_url) return false;
    if (v.provider === 'local') return false; /* not shipped yet */
    return true;
  }
  /* Pretty tool-call rendering: show the interesting argument as a quoted
   * string (query/text/path/…) instead of a raw JSON blob; fall back to
   * compact JSON. Clipped so the line stays one readable row. */
  function prettyToolArgs(argsStr) {
    let o = null;
    try { o = JSON.parse(argsStr); } catch (e) { o = null; }
    if (o && typeof o === 'object') {
      const keys = ['query', 'text', 'message', 'msg', 'path', 'file', 'url', 'city', 'pattern', 'task', 'code'];
      for (const k of keys) {
        if (typeof o[k] === 'string' && o[k]) {
          let s = o[k];
          if (s.length > 60) s = s.slice(0, 60) + '…';
          return JSON.stringify(s);
        }
      }
      const s = JSON.stringify(o);
      return s.length > 80 ? s.slice(0, 80) + '…' : s;
    }
    const s = String(argsStr || '');
    return s.length > 60 ? s.slice(0, 60) + '…' : s;
  }
  function clip1(s, n) {
    s = String(s == null ? '' : s);
    return s.length > n ? s.slice(0, n) + '…' : s;
  }

  /* ── TUI theme palette ────────────────────────────────────────────
   * All transcript colors resolve through T(role) into the active
   * theme's 256-palette params (see theme.rs — fixed colors, never the
   * terminal-themed 30-37 range, so a theme looks the same under any
   * terminal theme). THEME_FALLBACK reproduces the pre-theme look so a
   * missing bridge degrades to exactly today's colors, never unstyled.
   * PAL resets after every slash command (theme switches apply to the
   * very next paint) and is otherwise cached for the session. */
  const THEME_FALLBACK = { accent: '1;35', tool: '36', delegate: '35',
    heading: '1;36', code: '33', warn: '33', error: '1;31',
    add: '1;32', del: '1;31', hunk: '36' };
  let PAL = null;
  function pal() {
    if (!PAL) {
      try { PAL = JSON.parse(__chat_theme()); } catch (e) { PAL = null; }
      if (!PAL || typeof PAL !== 'object') PAL = {};
    }
    return PAL;
  }
  function T(role) {
    const p = pal();
    return (p && typeof p[role] === 'string' && p[role]) ? p[role] : THEME_FALLBACK[role];
  }

  /* Inline spans: code yellow, bold otherwise. Single pass split on
   * backticks — even segments are prose (bold applies), odd closed ones
   * are code (literal, so ** inside code never bolds). An unterminated
   * backtick keeps its mark and stays prose; unpaired ** stays literal.
   * Bold never spans segments, so markers can't merge across a span. */
  function fmtInline(t) {
    const segs = String(t).split('`');
    let r = '';
    for (let i = 0; i < segs.length; i++) {
      if (i % 2 === 1 && i !== segs.length - 1) {
        r += '\x1b[' + T('code') + 'm' + segs[i] + '\x1b[0m';
      } else {
        if (i % 2 === 1) r += '`';
        r += segs[i].replace(/\*\*([^*]+)\*\*/g, '\x1b[1m$1\x1b[0m');
      }
    }
    return r;
  }

  /* Markdown hierarchy for the final answer paint (TTY only). Model
   * answers arrive as undifferentiated white text — headings, emphasis
   * and code all read at the same level, so long answers are a wall.
   * Applied ONCE to the complete answer at turn end (never to the
   * streaming deltas, where markers can straddle chunk boundaries, and
   * never piped output, which stays raw). Newlines are never added or
   * removed, so soft-wrap row counts stay close to what streamed. */
  function formatMarkdown(s) {
    const lines = String(s == null ? '' : s).split('\n');
    const out = [];
    let inFence = false;
    for (const line of lines) {
      /* Fence lines dim whole; content inside fences is left alone
       * (code blocks are already visually distinct by indentation). */
      if (/^\s*```/.test(line)) { inFence = !inFence; out.push('\x1b[2m' + line + '\x1b[0m'); continue; }
      if (inFence) { out.push(line); continue; }
      /* Setext/HR rules and quotes recede. */
      if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) { out.push('\x1b[2m' + line + '\x1b[0m'); continue; }
      if (/^\s>/.test(line)) { out.push('\x1b[2m' + line + '\x1b[0m'); continue; }
      let t = line;
      /* ATX headings: bold cyan, markers stripped. The body stays literal
       * (no inline spans inside) so the whole title reads as one level —
       * a reset mid-title would split it into two visual weights. */
      const hm = t.match(/^(#{1,6})\s+(.*)$/);
      if (hm) { out.push('\x1b[' + T('heading') + 'm' + hm[2] + '\x1b[0m'); continue; }
      out.push(fmtInline(t));
    }
    return out.join('\n');
  }

  /* Live todo checklist (2026-10-03): the agent maintains its plan via
   * todo_write on long tasks, and the checklist rides the tool_result
   * event — but the TUI never rendered it, so users never SAW a todo
   * list. Painted here, every update: done dim ✓ (finished work
   * recedes), doing cyan ▸ at full weight (the current step pops),
   * queued dim ○. Bounded at 15 rows with a "+N more" tail so a 50-item
   * list cannot flood the transcript. Replaces the one-row "checklist
   * updated (N steps)" summary — the list IS the result. */
  function paintChecklist(todos) {
    const list = Array.isArray(todos) ? todos : [];
    let done = 0;
    for (const t of list) if (t && t.status === 'done') done++;
    out('\x1b[1m  ☑ ' + done + '/' + list.length + '\x1b[0m');
    const shown = Math.min(list.length, 15);
    for (let i = 0; i < shown; i++) {
      const t = list[i] || {};
      const c = String(t.content || '').replace(/\s+/g, ' ').trim();
      if (t.status === 'done') out('\x1b[2m    ✓ ' + c + '\x1b[0m');
      else if (t.status === 'doing') out('\x1b[' + T('tool') + 'm    ▸ ' + c + '\x1b[0m');
      else out('\x1b[2m    ○ ' + c + '\x1b[0m');
    }
    if (list.length > shown) out('\x1b[2m    +' + (list.length - shown) + ' more\x1b[0m');
  }

  /* Paint a unified diff from an edit_file/write_file tool_result event,
   * one row per diff line so the TUI shows WHAT changed, not just that
   * something did. Rows arrive as `NNNN mark content` (4-wide gutter,
   * mark in {'-','+',' '}, ANSI-free — the payload also travels to the
   * model). The gutter paints dim; +/- rows get a full-row background
   * tint (dark green / dark red) under bold bright text, difftool-style;
   * hunk headers cyan; context, file markers and notes dim. Text-span
   * tints only (no padding to terminal width — a padded row that wraps
   * would tear the gutter on the continuation line). */
  function paintDiff(diff) {
    const rows = String(diff == null ? '' : diff).split('\n');
    for (const row of rows) {
      const t = row.replace(/^\s+/, '');
      if (t.indexOf('+++') === 0 || t.indexOf('---') === 0 || t.indexOf('...') === 0) {
        out('\x1b[2m    ' + t + '\x1b[0m');
      } else if (t.indexOf('@@') === 0) {
        out('\x1b[' + T('hunk') + 'm    ' + t + '\x1b[0m');
      } else {
        const m = row.match(/^\s*(\d+)\s+([-+ ])\s?([\s\S]*)$/);
        if (!m) { out('\x1b[2m    ' + row + '\x1b[0m'); continue; }
        let g = m[1];
        while (g.length < 4) g = ' ' + g;
        if (m[2] === '+') out('    \x1b[2m' + g + '\x1b[0m \x1b[' + T('add') + 'm+ ' + m[3] + '\x1b[0m');
        else if (m[2] === '-') out('    \x1b[2m' + g + '\x1b[0m \x1b[' + T('del') + 'm- ' + m[3] + '\x1b[0m');
        else out('\x1b[2m    ' + g + '   ' + m[3] + '\x1b[0m');
      }
    }
  }

  /* ── Agent-phase label mapping ────────────────────────────────────
   * Pure function: onStep's {kind, payload} → the phase label, or null to
   * leave CURRENT_PHASE unchanged (tool_result: the tool/delegate phase
   * that started the work keeps showing until the next step arrives). */
  function phaseForStep(kind, payload) {
    const p = (payload === undefined || payload === null) ? {} : payload;
    if (kind === 'plan' || kind === 'think' || kind === 'recall') return 'Thinking';
    if (kind === 'tool' || kind === 'delegate') {
      const name = String((p.name || p.agent || p.tool || p.action || '') || '');
      if (/web_search|web_open|search/i.test(name)) return 'Web searching';
      if (/debug|trace|error|fix|diagnos/i.test(name)) return 'Debugging';
      if (/write|create|edit|implement|refactor|generate/i.test(name) &&
          !/debug|test/i.test(name)) return 'Coding';
      if (/read|list|scan|grep|inspect|review|audit|test|lint|check|verify/i.test(name)) return 'Auditing';
      return 'Working';
    }
    if (kind === 'tool_result') return null; /* keep the preceding phase */
    if (kind === 'answer_delta') return 'Working';
    if (typeof kind === 'string' && kind.indexOf('rlm:') === 0) return 'Working';
    return null;
  }
  /* Renders `label` with one dim "cursor" letter at PHASE_FRAME, advances
   * the cursor, and overlays the result above the input box. A single dim
   * position sweeps left→right and loops — not a fading trail. */
  function tickPhase() {
    if (!CURRENT_PHASE) { try { __chat_phase(''); } catch (e) {} return; }
    const label = CURRENT_PHASE;
    let line = '';
    for (let i = 0; i < label.length; i++) {
      line += (i === PHASE_FRAME ? '\x1b[2m' + label[i] + '\x1b[0m' : label[i]);
    }
    PHASE_FRAME = (PHASE_FRAME + 1) % label.length;
    try { __chat_phase('  ' + line); } catch (e) {}
  }
  let phaseTimer = null;
  const startPhase = () => {
    if (!TTY || phaseTimer) return;
    PHASE_FRAME = 0;
    tickPhase();
    phaseTimer = setInterval(tickPhase, 120);
  };
  function stopPhase() {
    if (phaseTimer) { clearInterval(phaseTimer); phaseTimer = null; }
    CURRENT_PHASE = null;
    if (TTY) { try { __chat_phase(''); } catch (e) {} }
  }

  async function turn(text, ov, turnNo, attemptNo) {
    failoverTo = ov || null;
    archiveTurn = turnNo || archiveTurn;
    archiveAttempt = attemptNo || archiveAttempt;
    const recoveryForTurn = archivePendingRecovery;
    archivePendingRecovery = '';
    const c = applyFailover(cfg);
    /* Unusable config = never call the API: one clean line, nothing else.
     * Judged on the MERGED view so a failover target is usable even when
     * the active config is not. */
    if (!usableConfig(c)) {
      failoverTo = null;
      out('\x1b[90m  No usable model configured — run /provider to set one up\x1b[0m\n');
      return;
    }
    /* Caps discovery (TTL-gated, best-effort): the endpoint's model
     * listing carries what its models really accept — harvest it into
     * the alloc gate's discovered store. Silent on any failure; local
     * endpoints skipped. */
    discoverCaps();
    /* F9: load hooks.js lazily on first turn. */
    await ensureHooks();
    /* F9: pre-hook — may modify or skip the prompt. */
    const hookedText = await runHookPre(text);
    if (hookedText === null) { out('\x1b[90m  (skipped by pre-hook)\x1b[0m\n'); return; }
    text = hookedText;
    /* F6: budget preflight — block if session spend ≥ budget. */
    let budget;
    try { budget = JSON.parse(__chat_budget_check()); } catch (e) { budget = null; }
    if (budget && !budget.ok) {
      out('\x1b[33m  ⚠ Budget cap reached ($' + budget.spent.toFixed(4) + ' / $' + budget.budget.toFixed(2) + ')\x1b[0m\n');
      return;
    }
    /* ── The agent runtime owns the loop (PLAN-AGENTS A1.5) ──────────
     * Recall augmentation, RLM routing, tool planning/execution, brain
     * storage, budgets and cancellation all live in src/js/agent.js —
     * ONE implementation for chat and headless. This driver renders
     * onStep events and owns input/history/session logging. */
    if (!sofuu.agent || typeof sofuu.agent.run !== 'function') {
      out('\x1b[31m  ✗ agent runtime unavailable (src/js/agent.js failed to load)\x1b[0m\n');
      stopPhase();
      return;
    }
    /* MCP is lazy: connect on demand only when a tool-using turn starts. */
    if (mcpTools.length === 0 && mcpServersList().length > 0) {
      await connectMcpServers();
    }
    /* Sub-agents (PLAN-AGENTS A8.3): definitions from ~/.sofuu/agents/*.js
     * are offered to the model via the `delegate` tool — loaded once per
     * session (same lazy load /agents uses; maxDepth:1 means children
     * can't delegate further). Loaded BEFORE mention parsing so `@name`
     * can resolve on the first turn too. */
    if (!agentsDirLoaded) {
      agentsDirLoaded = true;
      try { if (sofuu.agent && typeof sofuu.agent.loadDir === 'function') await sofuu.agent.loadDir(); } catch (e) {}
    }
    var subAgentNames = [];
    try {
      if (sofuu.agent && typeof sofuu.agent.list === 'function')
        subAgentNames = sofuu.agent.list().map(function (a) { return a.name; });
    } catch (e) {}
    /* ── @agent mentions (both forms) ────────────────────────────────
     * "@name task" or "@agent:name task" runs the named agent DIRECTLY
     * with its own definition (system/tools/memory/model) as a focused
     * run — no chat history injected. Agents win over same-named files;
     * unknown names fall through to the @file path below. Esc cancels
     * via signal:'chat' like any turn. */
    const typed = text;
    let agentMention = null;
    {
      const t = text.trim();
      let m = /^@agent:([A-Za-z0-9_.-]+)(?::\s|\s+)([\s\S]+)$/.exec(t);
      if (!m) m = /^@([A-Za-z0-9_.-]+)(?::\s|\s+)([\s\S]+)$/.exec(t);
      if (m) {
        let name = null;
        if (subAgentNames.indexOf(m[1]) >= 0) name = m[1];
        else {
          const low = m[1].toLowerCase();
          const ci = subAgentNames.filter(function (n) { return n.toLowerCase() === low; });
          if (ci.length === 1) name = ci[0];
        }
        if (name) agentMention = { name: name, task: m[2].trim() };
      }
    }
    if (agentMention) text = agentMention.task;
    /* F3: expand @file mentions (on the remainder — the task itself may
     * attach files). Fresh alloc plan for this turn (attachment budget). */
    allocPlanValid = false;
    const expanded = await expandMentions(text);
    const turnText = expanded.text;
    const manifest = expanded.manifest;
    if (promptLoggedTurn !== archiveTurn) {
      promptLoggedTurn = archiveTurn;
      try { __chat_log('prompt', String(typed)); } catch (e) {}
    }
    /* The turn starts in Thinking, including while lazy MCP/agent setup runs. */
    CURRENT_PHASE = 'Thinking';
    startPhase();
    if (TTY) out('\x1b[' + T('accent') + 'm  ⏺ you\x1b[0m \x1b[2m·\x1b[0m ' + typed);
    /* M1: /watch changes surface in chat context ONCE — drained into this
     * turn's ephemeral context note and cleared (hard-capped at the poll
     * site too, so an idle session can't accumulate). */
    const watchedNote = watchChangesPending.length
      ? watchChangesPending.map(c => '- ' + c.path + ' (' + c.kind + ')').join('\n')
      : '';
    watchChangesPending.length = 0;
    /* Strict effort: the CLI follows EXACTLY what the user picked.
     * `off`/empty → no thinking on ANY endpoint (never auto-injects).
     * The thinking block is only sent when c.effort is low/medium/high/max.
     * Models detected as thinking-incapable at runtime (no_think_models)
     * also never get an effort parameter. */
    let noThink = (c.no_think_models || []).indexOf(c.model) >= 0;
    const def = {
      name: 'chat',
      /* P1: the byte-stable shared core prompt. lastShared mesh notes no
       * longer concatenate into the system string — they ride the ephemeral
       * context message via opts.shared so prefix caching works (P6). */
      system: sofuu.agent.CORE_PROMPT,
      tools: chatToolDefs(),
      agents: subAgentNames.length ? subAgentNames : undefined,
      provider: c.provider, model: c.model,
      /* strict: only what the user selected, never more; off → undefined */
      effort: noThink ? undefined : (c.effort && c.effort !== 'off' ? c.effort : undefined),
      api_key: c.api_key || undefined,
      base_url: c.base_url || undefined,
      profile: c.profile || undefined,
      max_tokens: (function () { const r = resolveCaps(); return (r.maxOutput > 0 && r.maxSource !== 'default') ? r.maxOutput : undefined; })(),
      embed_provider: c.embed_provider || undefined,
      embed_model: c.embed_model || undefined,
      /* Memory embedding backend (PLAN-TINY-SEMANTIC-EMBEDDER §7):
       * ''/hash default; 'semantic' opts the brain into the 64-dim
       * projector (agent.js resolves + enforces, SOFUU_MEMORY_BACKEND
       * env also honored there). */
      embedding: c.memory_backend || undefined,
      memory: c.brain ? 'shared' : 'off',
      /* Config "brain_path" override — an empty value falls through to
       * agent.js's project-local <cwd>/.sofuu/brain/brain.qtsq default,
       * so this only bites when a host deliberately points the brain
       * elsewhere. The driver's /remember etc. share whatever this
       * resolves to (sofuu.agent.brainFor). */
      brainPath: c.brain_path || undefined,
      /* PLAN-ML-GATES: context-economy gates for this turn ('off' skips
       * every gate in agent.js; SOFUU_NO_ML=1 unregisters sofuu.ml too). */
      ml: c.ml ? 'on' : 'off',
      rlm: c.rlm === 'on' ? 'on' : (c.rlm === 'auto' ? 'auto' : 'off'),
      ctx_window: (function () { const r = resolveCaps(); return r.window > 0 ? r.window : undefined; })(),
      /* P2: config-level recall gating knobs (0 = agent.js defaults). */
      recallMin: c.recall_min > 0 ? c.recall_min : undefined,
      recallBudget: c.recall_budget > 0 ? c.recall_budget : undefined,
      /* Chat historically has no budgets beyond the step cap and the 200k
       * answer cap (both enforced inside agent.js). The step cap is
       * deliberately generous: real coding turns routinely need 10-30
       * rounds (read → read sections → grep → edit → verify), and the old
       * flat 8 made long tool-using turns die in a bare "(no response)"
       * (AUDIT-NO-RESPONSE-2026-08-24 cause b). On the rare breach the
       * agent spends one salvage round summarizing progress (agent.js) and
       * the driver prints why it stopped. SOFUU_CHAT_MAX_STEPS is a test
       * seam, not a user knob. */
      /* maxContinuations: how many times the step allowance renews before
       * the run finally stops (2026-10-01). 0 = hard stop at maxSteps,
       * which is what a test that only wants to exercise the breach
       * path sets — otherwise every breach test silently runs 3x the
       * windows it thinks it is running. */
      budget: { maxSteps: (parseInt(env('SOFUU_CHAT_MAX_STEPS'), 10) || 200), maxDepth: 1, maxTokens: 1e9, maxWallMs: 1e9,
                maxContinuations: env('SOFUU_CHAT_MAX_CONTINUATIONS') === ''
                  ? 2 : Math.max(0, parseInt(env('SOFUU_CHAT_MAX_CONTINUATIONS'), 10) || 0) },
    };
    /* alloc gate (§14): surface allocation notes BEFORE the request goes
     * out — config clamped to the model's caps, a learned limit applied,
     * unknown-model conservative defaults. One dim ASCII line each,
     * deduped per session; silent when ML is off. */
    try {
      const planN = allocPlanChat(estTok(turnText), manifest ? estTok(String(manifest)) : 0);
      if (planN && Array.isArray(planN.notes)) {
        for (const n of planN.notes) {
          if (allocNotedNotes[n]) continue;
          allocNotedNotes[n] = true;
          out('\x1b[90m  ~ alloc · ' + n + '\x1b[0m');
        }
      }
    } catch (e) {}
    /* Streaming render state — same UX as the pre-migration driver:
     * spinner + growing line in the TUI, plain writes when piped. */
    let acc = '', anim = null, si2 = 0, sawThink = false, capped = false;
    /* Whitespace rhythm: a blank row opens each tool-call group so tools
     * don't run directly into the prose above. Set on tool/delegate,
     * cleared by answer text — consecutive calls in one burst share one
     * blank row instead of each taking one. TTY only (pipes stay clean). */
    let lastRowWasTool = false;
    const thinks = [];
    const stopAnim = () => { if (anim) { clearTimeout(anim); anim = null; } };
    const tick2 = () => { outLast('\x1b[2m' + SPINNER[si2++ % SPINNER.length] + '\x1b[0m ' + acc); };
    const startAnim = () => { if (TTY && !anim) anim = setTimeout(function loop() { tick2(); anim = setTimeout(loop, 120); }, 120); };
    let rlmSummary = null;
    let answer = '(no response)';
    let res = null;
    /* The phase indicator is started at turn entry and always torn down in
     * the `finally` below, mirroring stopAnim(). One onStep renderer for
     * both paths (chat loop + direct @agent mention run). */
    let firstPromptTk = 0;  /* first LLM request's prompt = current context size */
    /* ── Live context meter (real-time during a running turn) ─────────
     * The footer's usedCtx used to freeze until the turn ENDED: the
     * numbers only calibrated from the final result. Mid-turn the real
     * signal exists though — EVERY 'llm' event carries that request's
     * promptTokens (each tool round is bigger than the last), and the
     * streaming deltas/tool results estimate the growth BETWEEN requests.
     * liveReqTk = last real request size; liveGrownTk = estimated growth
     * since that request (this turn's deltas + fresh tool results). The
     * 2s metricsTimer paints refreshStatus, which folds them in. */
    let liveReqTk = 0;       /* latest request's real promptTokens */
    let liveGrownTk = 0;     /* estimated growth since that request */
    let liveTurnTk = 0;      /* est. tokens of THIS turn's prompt text */
    let lastLivePaint = 0;   /* throttle: painted footer at most 1/s */
    liveCtxExtra = 0;        /* fresh turn: no mid-turn growth yet */
    const paintLive = function () {
      liveCtxExtra = liveGrownTk;
      const now = Date.now();
      if (now - lastLivePaint > 1000) { lastLivePaint = now; refreshStatus(''); }
    };
    const onStep = function (e) {
          const p = (e.payload === undefined || e.payload === null) ? {} : e.payload;
          /* Meter capture: the first LLM request's prompt size. */
          if (!firstPromptTk && e.kind === 'llm' && p.usage &&
              p.usage.promptTokens > 0) {
            firstPromptTk = p.usage.promptTokens;
          }
          /* Live meter: each round's real request size replaces the
           * estimate; growth since it starts from zero again. The first
           * request also calibrates ctxOverhead EARLY (same formula as the
           * turn-end calibration) so the +overhead part is real mid-turn. */
          if (e.kind === 'llm' && p.usage && p.usage.promptTokens > 0) {
            liveReqTk = e.payload.usage.promptTokens;
            liveGrownTk = 0;
            if (!liveTurnTk) liveTurnTk = estTok(text);
            if (ctxOverhead < 0) {
              const hNow = historyTokens();
              ctxOverhead = Math.max(0, liveReqTk - hNow - liveTurnTk);
            }
            usedCtx = liveReqTk;
            paintLive();
          }
          /* Between requests: the streamed answer + fresh tool results
           * grow the context the NEXT request will carry. Estimate only —
           * replaced by the real number on the next 'llm' event. */
          if (e.kind === 'answer_delta') {
            liveGrownTk += estTok(String(p));
            paintLive();
          } else if (e.kind === 'tool_result' && p && p.result !== undefined) {
            liveGrownTk += estTok(String(p.result));
            paintLive();
          }
          const ph = phaseForStep(e.kind, p);
          if (ph !== null && ph !== CURRENT_PHASE) {
            CURRENT_PHASE = ph;
            PHASE_FRAME = 0;
          }
          if (e.kind === 'think') {
            stopAnim();
            thinks.push(String(p.text || ''));
            /* Effort OFF: thinking runs under the hood — collected (it
             * still feeds the Thinking phase) but not painted into the
             * transcript, and sawThink stays false so no seal line is
             * emitted for a line that was never drawn. */
            if (c.effort) {
              sawThink = true;
              outLast('  ▸ thinking \x1b[3m' + thinks.join('') + '\x1b[0m');
            }
          } else if (e.kind === 'answer_delta') {
            if (sawThink && TTY) { out(''); sawThink = false; thinks.length = 0; }
            acc += String(p);
            if (acc.length > MAX_ANSWER_CHARS && !capped) {
              capped = true;
              out('  \x1b[' + T('warn') + 'm⚠ answer capped at ' + MAX_ANSWER_CHARS + ' chars\x1b[0m');
            }
            if (TTY) { startAnim(); tick2(); }
            else { process.stdout.write(String(p)); }
            lastRowWasTool = false;
          } else if (e.kind === 'plan') {
            /* A new planning round discards any preliminary streamed text. */
            acc = '';
          } else if (e.kind === 'tool') {
            stopAnim();
            if (TTY && !lastRowWasTool) out(' '); /* air before a tool group */
            lastRowWasTool = true;
            const a = prettyToolArgs(p.args);
            out('\x1b[90m  ⏺\x1b[0m \x1b[' + T('tool') + 'm' + p.name + '\x1b[0m\x1b[2m(' + a + ')\x1b[0m');
          } else if (e.kind === 'delegate') {
            stopAnim();
            if (TTY && !lastRowWasTool) out(' '); /* air before a tool group */
            lastRowWasTool = true;
            out('\x1b[90m  ⏺\x1b[0m \x1b[' + T('delegate') + 'm' + p.agent + '\x1b[0m\x1b[2m ← ' + clip1(p.task, 60) + '\x1b[0m');
          } else if (e.kind === 'recall') {
            /* F1/P2: /why reports exactly what gating let through this turn
             * (hits already thresholded, deduped, budget-cut by agent.js). */
            try { __chat_set_recall(JSON.stringify(Array.isArray(p.hits) ? p.hits : [])); } catch (e2) {}
            /* Brain feedback: one dim line so a recalled context is visible
             * (and the brain being ON is observable), never the contents. */
            if (p && p.count > 0) {
              out('\x1b[90m  ⏺ brain · ' + p.count + ' memor' + (p.count === 1 ? 'y' : 'ies') +
                  ' recalled (/why to inspect)\x1b[0m');
            }
          } else if (e.kind === 'consolidated') {
            /* Brain housekeeping (rare, rate-limited): weak-old episodic
             * memories were merged into semantic clusters. */
            out('\x1b[90m  ⏳ brain · consolidated ' + (p.clusters || 1) +
                ' cluster' + ((p.clusters || 1) === 1 ? '' : 's') +
                ' — old memories merged\x1b[0m');
          } else if (e.kind === 'mlgate') {
            /* PLAN-ML-GATES: a supervisor rule gate flagged waste (repeat
             * call / re-read of an unchanged file). One dim ASCII-only line;
             * the advisory nudge itself rides the tool result in-band. */
            stopAnim();
            out('\x1b[90m  ~ ml · ' + p.rule + ' · ' + p.tool + ' (step ' + p.step + ')\x1b[0m');
          } else if (e.kind === 'allocgate') {
            /* alloc gate (§14): the pre-flight fit check corrected the
             * request (re-capped tool results / clamped output / dropped
             * old messages), or error learning picked up a provider limit.
             * One dim ASCII line — allocation stays observable. */
            stopAnim();
            out('\x1b[90m  ~ alloc · ' +
                (p.action ? p.action + (p.model ? ' · ' + p.model : '') +
                            (p.error ? ' · ' + clip1(p.error, 80) : '')
                          : 'fit ' + String(p.fit || '') + ' · ' + String(p.actions || '')) +
                '\x1b[0m');
          } else if (e.kind === 'tool_result') {
            /* Keep tool output sparse: normal results stay in the live
             * transcript, while failures (or an explicitly marked result)
             * become independently recoverable archive records. */
            if (p && (p.error || p.archive === true)) {
              const material = p.error || p.result;
              if (material !== undefined && material !== null) {
                archiveWrite('tool_result', p.error ? 'failed' : 'complete', 'tool', material, {
                  tool_name: String(p.name || ''),
                  error_code: p.error ? archiveErrorCode(p.error) : undefined,
                  chars: p.chars || 0,
                });
              }
            }
            if (TTY && p.result) {
              /* todo_write carries its checklist on the event: paint the
               * list itself instead of the "checklist updated" summary. */
              if (p.name === 'todo_write' && Array.isArray(p.todos) && p.todos.length) {
                paintChecklist(p.todos);
              } else {
                /* Compact one-row result: embedded newlines become " · "
                 * so multi-line tool output (search results, file dumps)
                 * no longer splats unindented rows into the transcript.
                 * The ↳ marker stays dim but the CONTENT renders at full
                 * weight — the whole row used to be dim and read as a
                 * whisper next to the white answer text. */
                const r = String(p.result).replace(/\s*\n+\s*/g, ' · ').replace(/\s+/g, ' ').trim();
                out('\x1b[2m  ↳\x1b[0m ' + clip1(r, 140) + '\x1b[0m');
                /* edit_file/write_file carry their unified diff on p.diff
                 * (the 200-char event result could never hold it): paint it
                 * multi-line below the summary row so the change itself is
                 * visible, not just its byte count. */
                if ((p.name === 'edit_file' || p.name === 'write_file') && p.diff) {
                  paintDiff(p.diff);
                }
              }
            }
            /* Tool ERRORS used to print nothing at all: the error emit
             * carries {name, error} with no `result`, so the branch above
             * skipped it and a failed edit_file died silently on screen
             * while only the model saw the message in-band. Red ✗, always
             * visible, so a failed write is unmistakable. */
            else if (TTY && p.error) {
              /* The error text itself usually starts with "<tool>:" (the
               * tools name their own failures) — don't print the name
               * twice ("✗ edit_file: edit_file: …"). */
              const em = clip1(String(p.error), 200);
              const nm = String(p.name || 'tool');
              out('\x1b[' + T('error') + 'm  ✗ ' + (em.indexOf(nm + ':') === 0 ? em : nm + ': ' + em) + '\x1b[0m');
            }
          } else if (e.kind === 'rlm:route') {
            out('\x1b[90m  ⏺ rlm · working…\x1b[0m');
          } else if (e.kind === 'rlm:done') {
            rlmSummary = p;
          }
    };
    let runError = null;
    /* Re-apply every turn: covers the mode being set (or persisted from a
     * previous session) after agent.js finished loading past boot. */
    applyMode();
    try {
      res = agentMention
        ? await sofuu.agent.run(agentMention.name, turnText, {
            signal: 'chat', onStep, context: recoveryForTurn || undefined,
          })
        : await sofuu.agent.run(def, turnText, {
            history: history, signal: 'chat', onStep,
            /* P1/P6: volatile context rides the ephemeral context message —
             * never inside the byte-stable system prompt. */
            shared: lastShared || '',
            watched: watchedNote || '',
            context: recoveryForTurn || undefined,
          });
      answer = String((res && res.answer) || '(no response)');
    } catch (e) {
      runError = e;
    } finally {
      stopAnim();
      stopPhase(); /* turn end (G): hide the indicator, never leave it stale */
      /* Turn over — the mid-turn growth estimate is superseded by the
       * end-of-turn calibration below; zero it so ctxMeter()/refreshStatus
       * paint the settled value, not estimate + old growth. */
      liveCtxExtra = 0;
    }
    if (runError) {
      const message = String((runError && runError.message) || runError);
      const errorCode = archiveErrorCode(message);
      const partialRef = acc.trim()
        ? archiveWrite('partial', 'partial', 'assistant', acc, {
            error_code: errorCode,
            turn_complete: false,
          }, { retry_of: archiveLastFailure && archiveLastFailure.errorId })
        : null;
      const related = partialRef ? [partialRef.id] : [];
      const errorRef = archiveWrite('error', 'failed',
        errorCode === 'runtime' ? 'runtime' : 'provider', message, {
          error_code: errorCode,
          turn_complete: false,
        }, { related_output_ids: related, retry_of: archiveLastFailure && archiveLastFailure.errorId });
      archiveLastFailure = {
        errorId: errorRef && errorRef.id,
        partialId: partialRef && partialRef.id,
        errorCode: errorCode,
      };
      throw runError;
    }
    if (rlmSummary) {
      /* RLM turn: the answer arrived complete — print it plus the
       * one-line trace summary (same format as the pre-migration path). */
      out(TTY ? formatMarkdown(answer) : answer);
      out('\n\x1b[90m  ⏺ rlm · ' + (rlmSummary.calls || 0) + ' calls · ' + (rlmSummary.rounds || 0) + ' rounds · '
          + ((rlmSummary.ms || 0) / 1000).toFixed(1) + 's' + (rlmSummary.stopped ? ' · ' + rlmSummary.stopped : '') + '\x1b[0m\n');
    } else {
      if (TTY) {
        if (sawThink) out('');          /* seal a think-only answer */
        /* Full-text repaint with markdown hierarchy (headings/bold/code
         * get their levels — the streamed rows above showed raw markers).
         * Same newlines in and out, so the replaceable unit just swaps. */
        outLast(formatMarkdown(answer)); /* final line, no spinner */
      } else {
        /* Piped mode renders only answer_delta chunks — a turn with zero
         * deltas would print nothing at all, so emit the final answer
         * (e.g. "(no response)") when nothing streamed. */
        if (!acc && answer) out(answer);
        out('\n');
      }
    }
    if (res && res.stopped === 'cancelled') out('\x1b[90m  ⏹ stopped (esc)\x1b[0m');
    else if (res && res.stopped && String(res.stopped).indexOf('budget_') === 0) {
      /* Budget breach (steps/wall/tokens): the agent attempted a salvage
       * summary before stopping; say WHY it stopped so a breach is never
       * mistaken for a dead provider (AUDIT-NO-RESPONSE-2026-08-24 b). */
      /* With renewals, naming only maxSteps reads as a lie: the run
       * genuinely executed several windows' worth of rounds. Report the
       * renewals in the headline and let the detail line carry the
       * totals. */
      const renewed = !!(res.breach && res.breach.continuations);
      const why = res.stopped === 'budget_steps'
        ? 'step budget' + (renewed
            ? ' after ' + res.breach.continuations + ' renewal' +
              (res.breach.continuations === 1 ? '' : 's') +
              ' (' + (res.breach.maxSteps || '?') + ' rounds/window)'
            : (agentMention ? '' : ' (' + def.budget.maxSteps + ' rounds)'))
        : (res.stopped === 'budget_wall' ? 'wall-clock budget' : 'token budget');
      out('\x1b[90m  ⏹ stopped: ' + why + ' reached\x1b[0m');
      /* Say WHY, not just that it stopped. A bare "step budget reached"
       * cannot be acted on: 200 rounds on a genuinely large task needs a
       * bigger budget or a /compact, while 200 rounds re-reading one file
       * is a different bug entirely. agent.js collects this at the moment
       * of the breach, which is the only time the numbers exist. */
      try {
        const b = res.breach;
        if (b && b.kind === 'budget_steps') {
          const tk = (b.promptTokens || 0) + (b.completionTokens || 0);
          const win = b.ctxWindow || 0;
          const pct = win > 0 ? Math.round((tk / win) * 100) : null;
          out('\x1b[90m    ↳ ' + (b.steps || 0) + '/' + (b.maxSteps || 0) + ' rounds · ' +
              (b.llmCalls || 0) + ' llm calls · ' + (b.toolCalls || 0) + ' tool calls' +
              (b.loopMsgs ? ' · ' + b.loopMsgs + ' msgs in turn' : '') +
              (tk ? ' · ctx ' + fmtTk(tk) + (pct !== null ? ' (' + pct + '% of ' + fmtTk(win) + ')' : '') : '') +
              (b.mostCalledTool ? ' · most: ' + b.mostCalledTool : '') +
              '\x1b[0m');
          /* Renewals already spent — the difference between "never got a
           * second window" and "renewed and still not done". */
          if (b.continuations) {
            out('\x1b[90m    ↳ ' + b.continuations + '/' + (b.maxContinuations || '?') +
                ' renewals spent before it stopped\x1b[0m');
          }
          /* If the window was the real constraint, say so — /compact is
           * the immediate action and the user cannot infer it. */
          if (pct !== null && pct >= 85) {
            out('\x1b[90m    ↳ context was at ' + pct + '% — /compact now, or lower /ctx\x1b[0m');
          }
        }
      } catch (eB) {}
    } else if (res && res.stopped) {
      /* Unknown stop cause (future agent.js values): still say SOMETHING —
       * a silent stop reads as a dead provider (P3-2). */
      out('\x1b[90m  ⏹ stopped: ' + String(res.stopped) + '\x1b[0m');
    }
    if (agentMention && res) {
      /* @agent mention: compact run summary under the answer. */
      const u = res.usage || {};
      out('\x1b[90m  ⏺ via ' + agentMention.name + ' · ' + (u.llmCalls || 0) + ' llm calls · '
          + (u.toolCalls || 0) + ' tool calls' + (res.stopped ? ' · ' + res.stopped : '') + '\x1b[0m');
    }
    const ctxTk = (res && res.usage && res.usage.promptTokens) || 0;
    const outTk = (res && res.usage && res.usage.completionTokens) || 0;
    /* Meter calibration on a REAL request size: prefer the first LLM
     * request of this turn; a single-LLM-call turn's aggregate is that
     * same number. Multi-round tool turns are never used as-is — their
     * aggregate re-counts tool results and would balloon the meter.
     * Overhead = system prompt + tool schemas + ephemeral context: the
     * request's prompt minus the history it carries and minus this
     * turn's own text, so historyTokens() + overhead stays equal to the
     * request size and never double-counts the turn. */
    const histAtReq = historyTokens();
    const turnTk = estTok(text);
    if (firstPromptTk > 0) {
      usedCtx = firstPromptTk;
      ctxOverhead = Math.max(0, firstPromptTk - histAtReq - turnTk);
    } else if (ctxTk > 0 && ((res && res.usage && res.usage.llmCalls) || 0) <= 1) {
      usedCtx = ctxTk;
      ctxOverhead = Math.max(0, ctxTk - histAtReq - turnTk);
    }
    /* Chip names the two numbers (user request 2026-09-03 f): "in 30k ·
     * out 800" — the bare "30k→800" made the sides ambiguous once the
     * footer also shows ctx. Multi-round turns: the usage aggregate SUMS
     * every round's prompt — it is the turn's total INPUT bill, not one
     * request's context size (each tool round re-sends the whole context);
     * the call count says so. */
    const nCalls = ((res && res.usage && res.usage.llmCalls) || 0);
    let tk = (ctxTk > 0 || outTk > 0)
      ? ('in ' + fmtTk(ctxTk) + ' · out ' + fmtTk(outTk)) : '';
    if (nCalls > 1) tk = tk + ' · ' + nCalls + ' calls';
    /* F6: report usage → compute cost + persist. Cache token slots (P6.4)
     * ride along so /cost can show prefix-cache hits when non-zero. */
    if (ctxTk > 0 || outTk > 0) {
      const cacheR = (res && res.usage && res.usage.cacheReadTokens) || 0;
      const cacheW = (res && res.usage && res.usage.cacheWriteTokens) || 0;
      try { __chat_report_usage(c.model || '', ctxTk, outTk, cacheR, cacheW); } catch (e) {}
      /* Show running spend in the footer next to tk. */
      let cost;
      try { cost = JSON.parse(__chat_budget_check()); } catch (e) { cost = null; }
      if (cost && cost.spent > 0) {
        tk = tk + ' · $' + cost.spent.toFixed(4);
      }
    }
    out('\n');
    refreshStatus(tk);
    /* Persist the answer to this session's .qtsq (the prompt was already
     * logged at the start of the turn) so a resumed session replays the
     * full transcript. A failed-salvage placeholder is NEVER persisted nor
     * pushed into history (P2-14): a resumed session would replay
     * "(no response)" as if it were the answer. */
    const noResponse = answer === '(no response)' && res && res.stopped;
    if (!noResponse) {
      try { __chat_log('answer', String(answer)); } catch (e) {}
      /* F3: history stores the manifest only (not full file text). */
      const histText = manifest ? text + ' [' + manifest + ']' : text;
      history.push({ role: 'user', content: histText });
      /* Claude-style retention (2026-09-03 e): the turn's tool transcript
       * (assistant tool_calls + tool results) persists into history between
       * the prompt and the answer — the model keeps its tool context across
       * turns; compaction sheds it, not a per-turn release. */
      const turnTranscript = (res && Array.isArray(res.transcript)) ? res.transcript : null;
      if (turnTranscript) {
        for (const tm of turnTranscript) history.push(tm);
      }
      history.push({ role: 'assistant', content: answer });
      await trimHistory();
    }
    /* Meter: re-estimate for the NEXT request. With transcripts retained
     * (2026-09-03 e) the settled value stays close to the live one — it
     * moves only when compaction/drop-oldest just shrank history. A big
     * drop is compaction (its own ⚙ line explains it), so the settle note
     * stays generic; fire only on a real unexplained shift. */
    const liveShown = usedCtx + liveCtxExtra;
    usedCtx = ctxMeter();
    if (TTY && liveShown > usedCtx * 1.2 && liveShown > 1000) {
      out('\x1b[90m  · ctx meter settled ' + fmtTk(Math.round(liveShown)) + '→' +
          fmtTk(usedCtx) + '\x1b[0m');
    }
    /* M2: one full GC per completed turn — bounds JS garbage to a single
     * turn's worth instead of accumulating against the heap cap. */
    try { __chat_gc(); } catch (e) {}
    /* F9: post-hook — may modify the answer. */
    const finalAnswer = await runHookPost(text, answer, res && res.usage);
    if (finalAnswer !== answer) {
      /* Update the last history entry if the post-hook changed the answer. */
      if (history.length >= 1) history[history.length - 1].content = finalAnswer;
    }
    /* Materialize the user-visible result after post-processing. A cancelled
     * or budget-stopped run is partial evidence, never a successful final. */
    const usage = (res && res.usage) || {};
    const archiveMeta = {
      prompt_tokens: usage.promptTokens || 0,
      completion_tokens: usage.completionTokens || 0,
      llm_calls: usage.llmCalls || 0,
      tool_calls: usage.toolCalls || 0,
      stopped: res && res.stopped ? String(res.stopped) : undefined,
    };
    const partialText = (acc && acc.trim()) ? acc :
      ((finalAnswer && finalAnswer !== '(no response)' && finalAnswer !== '(cancelled)') ? finalAnswer : '');
    if (res && res.stopped === 'cancelled') {
      const partialRef = partialText
        ? archiveWrite('partial', 'cancelled', 'assistant', partialText, archiveMeta,
            { retry_of: archiveLastFailure && archiveLastFailure.errorId })
        : null;
      archiveWrite('error', 'cancelled', 'runtime', 'turn cancelled by user', {
        error_code: 'cancelled',
        stopped: 'cancelled',
      }, { related_output_ids: partialRef ? [partialRef.id] : [] });
    } else if (res && res.stopped) {
      const errorCode = String(res.stopped) === 'budget_tokens' ? 'output_limit' : 'runtime';
      const partialRef = partialText
        ? archiveWrite('partial', 'partial', 'assistant', partialText,
            Object.assign({}, archiveMeta, { error_code: errorCode }),
            { retry_of: archiveLastFailure && archiveLastFailure.errorId })
        : null;
      archiveWrite('error', 'failed', 'runtime', 'turn stopped: ' + String(res.stopped), {
        error_code: errorCode,
        stopped: String(res.stopped),
      }, { related_output_ids: partialRef ? [partialRef.id] : [] });
    } else if (finalAnswer && finalAnswer !== '(no response)') {
      archiveWrite('final', 'complete', 'assistant', finalAnswer, archiveMeta, {
        retry_of: archiveLastFailure && archiveLastFailure.errorId,
      });
    } else {
      archiveWrite('error', 'failed', 'runtime', 'model returned no response', {
        error_code: 'empty_response',
      }, { retry_of: archiveLastFailure && archiveLastFailure.errorId });
    }
  }
  // ── Provider setup wizard: /provider (no args) ──────────────────
  // Three wire formats only: openai-compatible, anthropic, local (ollama).
  // The endpoint is AUTO-DETECTED from the URL; the profile question is
  // only asked when detection can't decide.
  const WIZARD_PROVIDERS = ['openai', 'anthropic', 'local'];
  const WIZARD_BUILTIN_BASE = {
    openai: 'https://api.openai.com/v1',
    anthropic: 'https://api.anthropic.com/v1',
    local: 'http://127.0.0.1:11434',
  };
  function detectProfile(base) {
    const u = String(base || '').toLowerCase();
    /* Host-based detection — the reliable, unambiguous signals. */
    if (u.includes('anthropic') || u.includes('claude')) return 'anthropic';
    if (u.includes('127.0.0.1') || u.includes('localhost') || u.includes('0.0.0.0')) return 'local';
    /* Path-based: /v1 with /chat/completions or /models → openai-compat;
     * /v1/messages → anthropic. */
    if (u.includes('/messages')) return 'anthropic';
    if (u.includes('ollama') || u.includes(':11434')) return 'local';
    /* Default: anything with /v1 (or nothing recognizable) is treated as
     * OpenAI-compatible — the most common server format. */
    return 'openai';
  }
  function sleepMs(ms) { return new Promise(r => setTimeout(r, ms)); }
  async function fetchWithTimeout(url, opts, ms) {
    let timer = null;
    const to = new Promise((_, rej) => { timer = setTimeout(() => rej(new Error('timeout')), ms || 8000); });
    try {
      return await Promise.race([fetch(url, opts), to]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
  /* Stored endpoint → API root for the models listing. The base may be a
   * bare host, a /v1 root, or the FULL completion URL — including corrupted
   * forms like /v1/chat/completions/chat/completions (saved by old desktop
   * builds). Collapse doubles, loop-strip every completion suffix, then
   * drop a leftover /chat: the listing lives at the version root
   * (…/v1/models — /chat/models is not a wire endpoint). stripV1 is for the
   * local branch (ollama's /api/tags sits at the server ROOT). */
  function wizardModelsRoot(base, stripV1) {
    let u = String(base || '').trim();
    u = u.split('#')[0].split('?')[0].replace(/\/+$/, '');
    u = u.replace(/\/chat\/completions\/chat\/completions$/i, '/chat/completions');
    let again = true;
    while (again) {
      again = false;
      const sufs = ['/chat/completions', '/completions', '/messages', '/api/chat'];
      for (const s of sufs) {
        if (u.toLowerCase().endsWith(s)) { u = u.slice(0, u.length - s.length); again = true; }
      }
    }
    while (u.charAt(u.length - 1) === '/') u = u.slice(0, -1);
    if (stripV1) u = u.replace(/\/v1$/i, '');
    else if (/\/chat$/i.test(u)) u = u.slice(0, -'/chat'.length);
    return u;
  }
  async function wizardFetchModels(prov, base, key) {
    try {
      if (prov === 'local') {
        /* /api/tags lives at the server ROOT (P3-5: appending it to a
         * completion endpoint 404'd every local fetch). */
        const root = wizardModelsRoot(base || 'http://127.0.0.1:11434', true);
        const r = await fetchWithTimeout(root + '/api/tags', {}, 4000);
        const j = JSON.parse(await r.text());
        return (j.models || []).map(m => m.name);
      }
      if (prov === 'anthropic') return []; // no public model list
      /* openai-compatible: listing at the API root + /models. */
      const root = wizardModelsRoot(base || WIZARD_BUILTIN_BASE.openai, false);
      const h = key ? { Authorization: 'Bearer ' + key } : {};
      const r = await fetchWithTimeout(root + '/models', { headers: h }, 6000);
      /* text() + JSON.parse — Response.json() truncated some chunked
       * bodies ("unexpected data at the end", e.g. lightning.ai's 53-model
       * listing), so parse the full body ourselves. */
      const j = JSON.parse(await r.text());
      return (j.data || []).map(m => m.id).filter(Boolean);
    } catch (e) {
      return []; // unreachable / no key → fall back to manual entry
    }
  }
  async function providerWizard(prefill) {
    out('\n\x1b[1m  ⚙ Provider setup\x1b[0m  \x1b[90m(Ctrl-D cancels)\x1b[0m');
    const pre = prefill || {};
    // 1. Provider NAME — the real name the user knows (e.g. fireworks-ai,
    // groq, my-local-llm). "openai/anthropic/local" are endpoint FORMATS,
    // not names — never stored as the provider name.
    let name = '';
    while (!name) {
      const hint = pre.name ? ' [' + pre.name + ']' : '';
      const inp = await __readline('  Provider name (e.g. fireworks-ai, groq, deepseek)' + hint + ': ');
      if (inp === null || inp === undefined) { out(''); return false; }
      name = String(inp).trim() || (pre.name || '');
      if (!name) out('\x1b[31m  A provider name is required.\x1b[0m');
    }
    // 2. Base URL — the REAL endpoint (required; the format is detected
    // from it, so e.g. https://api.fireworks.ai/inference/v1 works).
    let base = '';
    while (!base) {
      const hint = pre.base_url ? ' [' + pre.base_url + ']' : '';
      const inp = await __readline('  Base URL (e.g. https://api.fireworks.ai/inference/v1)' + hint + ': ');
      base = (inp === null || inp === undefined ? '' : String(inp).trim()) || (pre.base_url || '');
      if (!base) out('\x1b[31m  A base URL is required.\x1b[0m');
    }
    // 2b. AUTO-DETECT the wire format from the URL. Ask only when the
    // detection is ambiguous (a custom host that looks like neither).
    let profile = detectProfile(base);
    if (profile === 'local') {
      profile = 'local';
    }
    if (profile === 'openai' && !/api\.openai|openai|anthropic|claude|ollama|11434|localhost|127\.0\.0\.1/.test(base.toLowerCase())) {
      // Custom host that didn't match any known pattern — confirm the format.
      const q = await __readline('  Can\'t auto-detect the format — 1) OpenAI-compatible 2) Anthropic: ');
      const qt = String(q || '').trim();
      if (qt === '2') profile = 'anthropic';
      else profile = 'openai';
    }
    // 3. API key
    const keyInp = await __readline('  API key (Enter = skip): ');
    const key = (keyInp === null || keyInp === undefined ? '' : String(keyInp).trim()) || '';
    // 4. fetch models (the detected format decides the list endpoint)
    outLast('  \x1b[90m⏳ Fetching models…\x1b[0m');
    const models = await wizardFetchModels(profile, base, key);
    if (models.length > 0) {
      outLast('  \x1b[90m✓ ' + models.length + ' models found\x1b[0m');
    } else {
      outLast('  \x1b[90m⚠ could not fetch models — type one manually\x1b[0m');
    }
    let model = '';
    if (models.length > 0) {
      out('\n  \x1b[1mAvailable models\x1b[0m');
      const page = models.slice(0, 30);
      page.forEach((m, i) => out('    ' + (i + 1) + ') ' + m));
      if (models.length > page.length) out('    \x1b[90m… ' + (models.length - page.length) + ' more\x1b[0m');
      while (!model) {
        const inp = await __readline('  Pick 1-' + page.length + ', 0 = type manually: ');
        if (inp === null || inp === undefined) { out(''); return false; }
        const t = String(inp).trim();
        const n = parseInt(t, 10);
        if (n >= 1 && n <= page.length) model = page[n - 1];
        else if (t === '0' || t === '') {
          const c = await __readline('  Model name: ');
          model = (c === null || c === undefined ? '' : String(c).trim());
        }
      }
    } else {
      const inp = await __readline('  (Could not fetch models) Type a model name: ');
      model = (inp === null || inp === undefined ? '' : String(inp).trim());
    }
    // 5. persist — provider NAME is the real name the user gave; the
    // profile (wire format) is stored separately for the AI layer.
    // Deduplicate any re-appended /chat/completions (wizard guard: editing an
    // existing endpoint that already contains the suffix would 404).
    let bare = base.replace(/\/+$/, '').replace(/\/chat\/completions\/chat\/completions/g, '/chat/completions');
    let hasChatSuffix = bare.endsWith('/chat/completions') || bare.endsWith('/completions') || bare.endsWith('/api/chat');
    let endpoint = hasChatSuffix ? bare : (profile === 'local' ? bare + '/api/chat' : (profile === 'openai' ? bare + '/chat/completions' : bare));
    try { __chat_apply_provider(name, endpoint, key, model, profile); } catch (e) {}
    cfg = JSON.parse(__chat_getcfg());
    out('\n  \x1b[32m✓ Provider configured:\x1b[0m ' +
      name +
      (model ? ' · model ' + model : '') +
      ' · ' + endpoint +
      (key ? ' · key set' : '') + '\n');
    refreshStatus('');
    maybeRefreshPanel();
    return true;
  }

  async function compact() {
    if (history.length === 0) { out('\x1b[90m  [Nothing to compact yet.]\x1b[0m\n'); return; }
    out('\x1b[90m  [Compacting ' + Math.floor(history.length / 2) + ' turns…]\x1b[0m');
    const before = historyTokens();
    try {
      /* Manual /compact folds everything but the most recent turn. */
      if (!await summarizeHistory(1)) {
        out('\x1b[90m  [Nothing older than one turn to fold — history unchanged]\x1b[0m\n');
        return;
      }
      const m0 = Math.max(0, before + (ctxOverhead >= 0 ? ctxOverhead : 0));
      usedCtx = ctxMeter();
      out('\x1b[90m  [Compacted → context (' + fmtTk(Math.max(0, before - historyTokens())) +
          ' tk saved, meter ' + fmtTk(m0) + '→' + fmtTk(usedCtx) + ')]\x1b[0m\n');
      refreshStatus(lastChip); /* meter drops visibly right away */
    } catch (e) { out('\x1b[31m  [Compact failed: ' + String(e.message || e) + ']\x1b[0m\n'); }
  }
  // Output routing: in the full-screen TUI everything lands in the
  // conversation area; otherwise plain stdout (piped/REPL stays clean).
  function out(s) {
    if (TTY) { try { __tui_log(s); } catch (e) {} }
    else { console.log(s); }
  }
  function outLast(s) {
    if (TTY) { try { __tui_log_last(s); } catch (e) {} }
    else { console.log(s); }
  }
  function fmtTk(n) {
    return n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k' : String(n);
  }
  function refreshStatus(tk) {
    lastChip = tk;
    if (typeof __chat_status !== 'function') return;
    const dim = (s) => '\x1b[2m' + s + '\x1b[0m';
    let s = 'sofuu ' + dim('·') + ' \x1b[' + T('accent') + 'm' + (cfg.model || '(no model)') + '\x1b[0m'
      /* Mode chip rides the status row only when RESTRICTIVE (plan/edit):
       * the default 'full' stays clean, but a gated turn must never be
       * confused for an unguarded one — amber so it reads at a glance.
       * Placed RIGHT AFTER the model: the footer truncates from the tail
       * (≈47 cells on an 80-col pty once hints are subtracted), and the
       * active restriction must survive where provider/effort may not. */
      + (cfg.permissions && cfg.permissions !== 'full'
        ? ' ' + dim('·') + ' \x1b[33m' + 'mode ' + cfg.permissions + '\x1b[0m'
        : '');
    if (cfg.provider) {
      const prov = cfg.provider === 'custom'
        ? ((cfg.profile || 'openai') + '-compat')
        : cfg.provider;
      s += ' ' + dim('·') + ' ' + dim(prov);
    }
    if (cfg.effort) s += ' ' + dim('·') + ' ' + dim('effort ' + cfg.effort);
    /* The per-turn usage chip ("in 96 · out 1 · $") goes to the METRIC row,
     * not here: the status row shares its width with the hints column
     * (W-gutter-1-hints ≈ 32 cells on an 80-col terminal) and the chip
     * was silently truncated away there — "no token display" was really
     * "no room". Row 2 has the space. */
    const chip = tk || '';
    /* Footer row 1 right: dim hints. Row 2: left = context used,
     * right = real-time RAM of the sofuu process. The window is the
     * ladder-resolved one for the SELECTED model — the footer can't
     * show a stale global number the model doesn't have. */
    const win = resolveCaps().window;
    /* Mid-turn: the meter shows the live estimate (last real request +
     * growth since) so it climbs while the answer streams/tools run. */
    const used = usedCtx > 0 ? usedCtx + liveCtxExtra : liveCtxExtra;
    const pct = used > 0 ? Math.round(used / win * 100) : 0;
    /* Percent only once it is meaningful (≥1%) — "ctx 0.2k/128k (0%)"
     * reads like a bug. Turns amber past 80% of the window. */
    let ctxMet;
    if (used > 0) {
      ctxMet = 'ctx ' + fmtTk(used) + '/' + fmtTk(win) + (pct >= 1 ? ' · ' + pct + '%' : '');
      if (used / win > 0.8) ctxMet = '\x1b[33m' + ctxMet + '\x1b[0m';
    } else {
      ctxMet = 'ctx 0/' + fmtTk(win);
    }
    /* Brain state rides the ctx metric — the footer stays clean when the
     * brain is off. The per-turn usage chip rides the SAME row (left of
     * RAM): "ctx 2.4k/128k · brain · in 30k · out 800 · $0.0121". */
    if (cfg.brain) { ctxMet = ctxMet + dim(' · brain'); }
    if (chip) { ctxMet = ctxMet + dim(' · ' + chip); }
    let ramMet = '';
    try {
      if (typeof __chat_rss === 'function') {
        const b = Number(__chat_rss()) || 0;
        if (b > 0) {
          const mb = b / 1048576;
          ramMet = 'ram: ' + mb.toFixed(1) + ' MB';
          /* M6: soft tripwire — amber past the warn threshold + a one-time
           * notice. Inform, never kill; if this fires persistently, some
           * structure is leaking. */
          const warnMb = cfg.rss_warn_mb > 0 ? cfg.rss_warn_mb : 1024;
          if (mb >= warnMb) {
            ramMet = '\x1b[33m' + ramMet + '\x1b[0m';
            if (!rssWarned) {
              rssWarned = true;
              out('\x1b[33m  ⚠ RAM usage crossed ' + warnMb + ' MB — watch the footer; persistent growth means something is leaking\x1b[0m');
            }
          }
        }
      }
    } catch (e) {}
    try { __chat_status(s, '/help · ⏎ · esc · ctrl-k copy', ramMet, ctxMet); } catch (e) {}
  }
  // ── Interactive selector overlay (pickers for /model /provider /effort)
  // C routes keys + draws the overlay rows; ALL state lives here.
  let selectorCtl = null;
  globalThis.__selector_key = function(name, ch) {
    if (selectorCtl) { try { selectorCtl(name, ch); } catch (e) {} }
  };
  async function selectMenu(opts) {
    if (typeof __selector_open !== 'function') return null;
    const title = opts.title || 'Select';
    const hint  = opts.hint  || '';
    const tabs  = opts.tabs  || [];
    const items   = opts.items || [];
    const currentId = opts.currentId || '';
    let filter = '', tabIdx = 0, sel = 0, off = 0, result = null;
    const H = (typeof __tui_height === 'function') ? __tui_height() : 24;
    const maxV = Math.max(3, Math.min(10, H - 12));
    if (opts.initialTab) {
      const at = tabs.indexOf(opts.initialTab);
      if (at >= 0) tabIdx = at;
    }

    function filteredItems() {
      const t = tabs[tabIdx];
      const f = filter.toLowerCase();
      return items.filter(it =>
        (!t || t === 'all' || it.tab === t) &&
        (!f || it.label.toLowerCase().includes(f) || (it.note || '').toLowerCase().includes(f)));
    }
    function clampSel(list) {
      if (list.length === 0) { sel = 0; off = 0; return; }
      if (sel >= list.length) sel = list.length - 1;
      if (sel < 0) sel = 0;
      if (sel < off) off = sel;
      if (sel >= off + maxV) off = sel - maxV + 1;
    }
    function draw() {
      const list = filteredItems();
      clampSel(list);
      const rows = [];
      /* Spacing rule: section gaps are multiples of 2 (2 blank rows). */
      rows.push('');
      rows.push('');
      rows.push('  \x1b[1;35m' + title + '\x1b[0m  \x1b[90m(type to search · esc cancels)\x1b[0m');
      if (hint) rows.push('  \x1b[90m' + hint + '\x1b[0m');
      if (filter)
        rows.push('  \x1b[90mfilter:\x1b[0m ' + filter + ' \x1b[90m· ' + list.length + ' match' + (list.length === 1 ? '' : 'es') + '\x1b[0m');
      if (tabs.length > 1) {
        rows.push('');
        rows.push('');
        rows.push('  ' + tabs.map((t, i) =>
          i === tabIdx ? '\x1b[7;35m ' + t + ' \x1b[0m' : '\x1b[90m ' + t + ' \x1b[0m').join('  '));
      }
      rows.push('');
      rows.push('');
      const vis = list.slice(off, off + maxV);
      for (let i = 0; i < vis.length; i++) {
        const gi = off + i, it = vis[i], curRow = gi === sel;
        rows.push((curRow ? '\x1b[35m❯\x1b[0m \x1b[1m' : '  ') + it.label + (curRow ? '\x1b[0m' : '') +
                  (it.note ? '  \x1b[90m' + it.note + '\x1b[0m' : '') +
                  (it.id === currentId ? ' \x1b[32m← current\x1b[0m' : ''));
      }
      if (list.length === 0) rows.push('  \x1b[90m(no matches)\x1b[0m');
      if (off + maxV < list.length) rows.push('  \x1b[90m▼ ' + (list.length - off - maxV) + ' more\x1b[0m');
      if (off > 0) rows.push('  \x1b[90m▲ ' + off + ' above\x1b[0m');
      __selector_draw(rows.join('\n'));
    }

    const opened = __selector_open();
    draw();
    selectorCtl = function(name, ch) {
      const list = filteredItems();
      if (name === 'char') { filter += ch; sel = 0; off = 0; draw(); return; }
      if (name === 'backspace') { filter = filter.slice(0, -1); sel = 0; off = 0; draw(); return; }
      if (name === 'redraw') { draw(); return; }
      if (name === 'tab' || name === 'shift-tab' ||
          ((name === 'left' || name === 'right') && tabs.length > 1)) {
        const d = (name === 'shift-tab' || name === 'left') ? tabs.length - 1 : 1;
        tabIdx = (tabIdx + d) % tabs.length;
        sel = 0; off = 0; draw(); return;
      }
      if (name === 'up'   || (name === 'left'  && tabs.length <= 1)) {
        if (list.length) { sel = (sel - 1 + list.length) % list.length; draw(); } return;
      }
      if (name === 'down' || (name === 'right' && tabs.length <= 1)) {
        if (list.length) { sel = (sel + 1) % list.length; draw(); } return;
      }
      if (name === 'pageup')   { if (list.length) { sel = Math.max(0, sel - maxV); draw(); } return; }
      if (name === 'pagedown') { if (list.length) { sel = Math.min(list.length - 1, sel + maxV); draw(); } return; }
      if (name === 'enter') {
        result = list[sel] || null;
        selectorCtl = null;
        __selector_close(result ? 'select' : 'cancel');
        return;
      }
      if (name === 'esc') {
        selectorCtl = null;
        __selector_close('cancel');
        return;
      }
    };
    await opened; /* resolves when __selector_close runs */
    return result;
  }
  async function pickModel() {
    /* One tab per USER-ADDED provider (Tab / Shift-Tab / ←→ switch). Each
     * tab lists that provider's REAL live models fetched from its own
     * endpoint; an unreachable endpoint falls back to manual entry instead
     * of fake suggestions. Picking from another provider's tab switches
     * the active provider first, so the model applies to the right
     * endpoint/key. */
    if (!cfg.provider || !cfg.model) {
      out('\x1b[90m  No provider configured yet — run /provider first\x1b[0m\n');
      await providerWizard();
      try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
      return;
    }
    let provs = (cfg.providers && cfg.providers.length) ? cfg.providers.slice() : [];
    if (!provs.length) {
      /* Legacy flat config: synthesize the single active provider. */
      provs = [{ name: cfg.provider, endpoint: cfg.base_url, api_key: cfg.api_key || '', model: cfg.model, profile: '' }];
    }
    const tabs = provs.map(p => p.name);

    outLast('\x1b[90m  ⏳ Fetching models from ' + tabs.length + ' provider' + (tabs.length === 1 ? '' : 's') + '…\x1b[0m');
    /* All providers in PARALLEL — each fetch races its own timeout, so one
     * slow/unreachable endpoint can't stall the others. */
    const lists = await Promise.all(provs.map(async p => {
      try {
        const profile = p.profile || detectProfile(p.endpoint);
        const got = await Promise.race([
          wizardFetchModels(profile, p.endpoint, p.api_key || ''),
          new Promise(r => setTimeout(() => r(null), 6000)),
        ]);
        if (Array.isArray(got) && got.length) {
          /* Remember the live listing on the provider entry — an endpoint
           * that is unreachable NEXT time still offers these models. */
          try { __chat_models_cache(p.name, JSON.stringify(got)); } catch (e) {}
        }
        return Array.isArray(got) ? got : [];
      } catch (e) { return []; }
    }));
    const okCount = lists.filter(l => l.length > 0).length;
    if (okCount > 0) {
      outLast('\x1b[90m  ✓ live models from ' + okCount + '/' + tabs.length + ' providers\x1b[0m');
    } else {
      outLast('\x1b[90m  ⚠ no endpoint reachable — type a model name manually\x1b[0m');
    }

    const items = [], seen = {};
    const activeName = cfg.active || cfg.provider;
    for (let ti = 0; ti < provs.length; ti++) {
      const p = provs[ti], live = lists[ti] || [], isCur = p.name === activeName;
      if (isCur) {
        /* Current model first on its own tab (sel starts at 0). */
        const id = p.name + '|' + cfg.model;
        if (!seen[id]) { seen[id] = 1; items.push({ id: id, label: cfg.model, tab: p.name, note: 'current' }); }
      }
      for (const m of live) {
        const id = p.name + '|' + m;
        if (seen[id]) continue; seen[id] = 1;
        items.push({ id: id, label: m, tab: p.name, note: '' });
      }
      /* Models remembered for this provider (every model the user set here
       * + the last successful listing) — shown even when the live fetch
       * came back empty, so an unreachable endpoint no longer hides them. */
      const stored = Array.isArray(p.models) ? p.models : [];
      for (const m of stored) {
        const id = p.name + '|' + m;
        if (seen[id]) continue; seen[id] = 1;
        items.push({ id: id, label: m, tab: p.name, note: 'saved' });
      }
      /* The provider's stored default model, when not in the live list. */
      if (p.model && !seen[p.name + '|' + p.model]) {
        seen[p.name + '|' + p.model] = 1;
        items.push({ id: p.name + '|' + p.model, label: p.model, tab: p.name, note: 'saved' });
      }
      if (isCur && live.length === 0) {
        items.push({ id: '\x00manual', label: '⌨ Type a model name…', tab: p.name,
                     note: 'endpoint unreachable — enter it manually' });
      }
    }
    const it = await selectMenu({
      title: 'Select a model',
      hint: '↑↓ move · tab switch provider · enter apply · esc cancel',
      tabs: tabs, items: items, initialTab: activeName,
      currentId: activeName + '|' + cfg.model,
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    if (it.id === '\x00manual') {
      const m = await __readline('  Model name: ');
      if (m && String(m).trim()) __chat_slash('/model ' + String(m).trim());
    } else {
      const bar = it.id.indexOf('|');
      const provName = it.id.slice(0, bar);
      const m = it.id.slice(bar + 1);
      /* Model picked on ANOTHER provider's tab → switch provider first. */
      if (provName && provName !== activeName && tabs.indexOf(provName) >= 0) {
        try { __chat_select_provider(provName); } catch (e) {}
      }
      __chat_slash('/model ' + m);
    }
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickProvider() {
    const provs = (cfg.providers && cfg.providers.length) ? cfg.providers : [];
    const items = [];
    for (const p of provs) {
      const active = p.name === cfg.active;
      const note = (p.model || '(no model)') + ' · ' + (p.endpoint || '(no endpoint)') + (p.api_key ? ' · key set' : '') + (active ? ' ← active' : '');
      items.push({ id: '\x00select:' + p.name, label: p.name + (active ? ' ✓ active' : ''), tab: '', note: note });
      items.push({ id: '\x00edit:' + p.name, label: '  ✎ Edit ' + p.name, tab: '', note: 'fix name / endpoint / key / model' });
      items.push({ id: '\x00remove:' + p.name, label: '  🗑 Remove ' + p.name, tab: '', note: 'remove from list' });
    }
    if (!provs.length) {
      items.push({ id: '\x00none', label: '(no providers yet)', tab: '', note: 'add one below' });
    }
    items.push({ id: '\x00add', label: '＋ Add a provider', tab: '', note: 'OpenAI-compatible · Anthropic · local (guided setup)' });
    const currentId = cfg.active ? ('\x00select:' + cfg.active) : '\x00add';
    const it = await selectMenu({ title: 'Providers', hint: '↑↓ move · enter apply · esc cancel', tabs: [], items: items, currentId: currentId });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    if (it.id === '\x00add' || it.id === '\x00none') {
      await providerWizard();
    } else if (it.id.indexOf('\x00select:') === 0) {
      const name = it.id.slice('\x00select:'.length);
      try { __chat_select_provider(name); } catch (e) {}
      try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
      out('\x1b[32m  ✓ Active provider → ' + name + '\x1b[0m');
    } else if (it.id.indexOf('\x00edit:') === 0) {
      const name = it.id.slice('\x00edit:'.length);
      const entry = provs.find(function(p){ return p.name === name; });
      if (entry) await providerWizard({ name: entry.name, base_url: entry.endpoint });
    } else if (it.id.indexOf('\x00remove:') === 0) {
      const name = it.id.slice('\x00remove:'.length);
      const ans = await __readline('  Remove provider "' + name + '"? y/N: ');
      if (ans && String(ans).trim().toLowerCase() === 'y') {
        try { __chat_remove_provider(name); } catch (e) {}
        out('\x1b[90m  Removed ' + name + '\x1b[0m');
      } else { out('\x1b[90m  (unchanged)\x1b[0m'); try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {} refreshStatus(''); maybeRefreshPanel(); return; }
    }
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickEffort() {
    /* Capability-aware picker (rt/model_caps via sofuu.ai.modelCaps):
     * notes show the REAL budgets this model will get; models that
     * cannot think say so instead of offering a fake ladder. */
    const caps = modelCaps();
    const fmt = n => n >= 1000 ? Math.round(n / 1000) + 'k' : String(n);
    const runtimeNoThink = (cfg.no_think_models || []).indexOf(cfg.model) >= 0;
    const isAnthropicWire = (cfg.profile === 'anthropic' || cfg.provider === 'anthropic');
    if (runtimeNoThink) {
      out('\x1b[33m  ⓘ ' + cfg.model + ' does not support thinking — effort is not applicable.\x1b[0m\n');
      return;
    }
    // Anthropic endpoint: 64k thinking is endpoint property, any model on it can use it
    // OpenAI endpoint: per-model ladder, generic for unknown models, none only for known non-reasoning
    if (!isAnthropicWire && caps && caps.thinking === 'none') {
      out('\x1b[33m  ⓘ ' + cfg.model + ' does not support thinking — effort is not applicable.\x1b[0m\n');
      return;
    }
    let order, notes;
    if (!isAnthropicWire && caps && caps.thinking === 'effort') {
      /* Discrete ladder model (OpenAI style): only its real levels.
       * Selecting an unsupported level falls back to the highest one.
       * OpenAI endpoint is generic — any provider's models can appear here. */
      order = ['off'].concat(caps.efforts || []);
      notes = { off: 'strict off — no thinking' };
      for (const l of (caps.efforts || [])) {
        notes[l] = l === caps.efforts[caps.efforts.length - 1]
          ? 'highest this model supports'
          : l + ' reasoning';
      }
      if (cfg.effort === 'max') notes.currentNote = "max → falls back to '" + caps.efforts[caps.efforts.length - 1] + "'";
    } else if (isAnthropicWire) {
      /* Anthropic wire: 64k is thinking budget (not context window), endpoint-driven, any model */
      order = ['off', 'low', 'medium', 'high', 'max'];
      const cap = 64000;
      notes = { off: 'strict off — no thinking',
                low: '≈' + fmt(Math.ceil(cap * 0.0625)) + ' tk',
                medium: '≈' + fmt(Math.ceil(cap * 0.25)) + ' tk',
                high: '≈' + fmt(Math.ceil(cap * 0.5)) + ' tk',
                max: '≈' + fmt(Math.ceil(cap * 0.9)) + ' tk (of ' + fmt(cap) + ' thinking)' };
    } else {
      /* Budget-style unknown on OpenAI generic endpoint or fallback */
      order = ['off', 'low', 'medium', 'high', 'max'];
      notes = { off: 'strict off — no thinking',
                low: '~6% thinking budget', medium: '~25%', high: '~50%', max: '~90% (uses the full ceiling)' };
      if (caps && caps.maxThinkingBudget > 0) {
        const cap = caps.maxThinkingBudget;
        notes.low = '≈' + fmt(Math.ceil(cap * 0.0625)) + ' tk';
        notes.medium = '≈' + fmt(Math.ceil(cap * 0.25)) + ' tk';
        notes.high = '≈' + fmt(Math.ceil(cap * 0.5)) + ' tk';
        notes.max = '≈' + fmt(Math.ceil(cap * 0.9)) + ' tk (of ' + fmt(cap) + ')';
      }
    }
    const it = await selectMenu({
      title: 'Thinking effort' + (cfg.model ? ' — ' + cfg.model : ''),
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: order.map(l => ({ id: l, label: l, tab: '', note: notes[l] || '' })),
      currentId: cfg.effort || 'off',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/effort ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickTheme() {
    /* 13 dark-blend themes, independent of the terminal's own theme.
     * Applies through /theme <name> (the Rust side validates + persists)
     * so the picker and the typed command share one code path. */
    let themes = [];
    try { themes = JSON.parse(__chat_themes()); } catch (e) {}
    if (!themes.length) { out('\x1b[90m  No themes available.\x1b[0m\n'); return; }
    const it = await selectMenu({
      title: 'Theme',
      hint: '↑↓ move · enter apply · esc cancel',
      tabs: [],
      items: themes.map(t => ({ id: t.name, label: t.name })),
      currentId: cfg.theme || '',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/theme ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    PAL = null;
    maybeRefreshPanel();
  }
  async function pickRlm() {
    const notes = { off: 'provider default routing', on: 'every long turn goes through the RLM loop', auto: 'heuristic routing for long-context turns' };
    const order = ['off', 'on', 'auto'];
    const it = await selectMenu({
      title: 'RLM routing',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: order.map(l => ({ id: l, label: l, tab: '', note: notes[l] })),
      currentId: cfg.rlm || 'off',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/rlm ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickCtx() {
    const mc = modelCaps();
    const eff = resolveCaps().window;
    const presets = [
      { id: '0', label: 'default', note: (mc ? 'model default (' + fmtTk(mc.ctxWindow) + ')' : 'provider default (' + fmtTk(eff) + ')') },
      { id: '32768', label: '32k', note: '32,768 tokens' },
      { id: '131072', label: '128k', note: '131,072 tokens' },
      { id: '1048576', label: '1M', note: '1,048,576 tokens' },
    ];
    const it = await selectMenu({
      title: 'Context window',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: presets.map(p => ({ id: p.id, label: p.label, tab: '', note: p.note })),
      currentId: String(cfg.ctx_window || 0),
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/ctx ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickMaxout() {
    const mc = modelCaps();
    const presets = [
      { id: '0', label: 'default', note: mc && mc.maxOutput > 0
          ? "model's max output (" + fmtTk(mc.maxOutput) + ')'
          : 'provider default' },
      { id: '4096', label: '4k', note: '4,096 tokens' },
      { id: '32768', label: '32k', note: '32,768 tokens' },
      { id: '131072', label: '128k', note: '131,072 tokens' },
      { id: '384000', label: '384k', note: '384,000 tokens (max)' },
    ];
    const it = await selectMenu({
      title: 'Max output tokens',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: presets.map(p => ({ id: p.id, label: p.label, tab: '', note: p.note })),
      currentId: String(cfg.max_output || 0),
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/maxout ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickBrain() {
    const notes = { on: 'memory recall + agent loop enabled', off: 'memory integration disabled' };
    const it = await selectMenu({
      title: 'Brain (memory)',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: ['on', 'off'].map(l => ({ id: l, label: l, tab: '', note: notes[l] })),
      currentId: cfg.brain ? 'on' : 'off',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/brain ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickSync() {
    const notes = { on: 'session-mesh polling enabled', off: 'this session stops polling peers' };
    const it = await selectMenu({
      title: 'Session sync',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: ['on', 'off'].map(l => ({ id: l, label: l, tab: '', note: notes[l] })),
      currentId: cfg.sync ? 'on' : 'off',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/sync ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function main() {
    let me = '';
    try {
      const p0 = JSON.parse(__chat_poll());
      me = p0.me || '';
      if (p0.shared) lastShared = p0.shared;
      for (const n of (p0.notices || [])) out('\x1b[90m  ⇄ ' + n.id + ' · ' + n.kind + ': ' + n.text + '\x1b[0m');
    } catch (e) {}
    /* MCP is OPTIONAL and LAZY: no connect at startup — a slow/dead server
     * must never delay the prompt. The first /tools (or a tool-using agent
     * turn) connects on demand. */
    refreshStatus('');
    /* Real-time metrics: refresh the footer (ctx used + RAM) every 2s so
     * the numbers track the live process without waiting for input. The
     * repaint carries lastChip — the per-turn usage chip ("in→out tk · $")
     * must SURVIVE the timer, not vanish 2s after the turn ends. */
    const metricsTimer = setInterval(function() {
      try { refreshStatus(lastChip); } catch (e) {}
    }, 2000);
    const stopMetrics = () => { try { clearInterval(metricsTimer); } catch (e) {} };
    globalThis.__metrics_stop = stopMetrics;
    if (TTY) {
      /* Full-screen interface: take over the terminal. */
      try { __tui_on(); } catch (e) {}
      /* Bordered welcome panel (title + session facts) — built in Rust. */
      try { __chat_welcome(); } catch (e) {}
      /* Re-open the previous transcript (persisted in the session .qtsq). */
      try {
        const past = JSON.parse(__chat_past_turns());
        for (const t of past) {
          out('\x1b[' + T('accent') + 'm  ⏺ you\x1b[0m · ' + t.prompt);
          if (t.answer) out(t.answer);
        }
        if (past.length) out('');
      } catch (e) {}
    } else {
      /* Compact two-line banner (built in Rust, same content as the panel). */
      try { __chat_welcome(); } catch (e) {}
    }
    /* ── First-run onboarding: no usable model → force the guided setup
     * wizard (provider → URL → key → model list → pick). The wizard
     * persists the choice to ~/.sofuu/config.json, so it survives restarts
     * until the user changes it with /provider or /model. */
    if (!usableConfig()) {
      out('\x1b[90m  No usable model configured yet — set one up to start chatting.\x1b[0m');
      await providerWizard();
      try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
      refreshStatus('');
      if (usableConfig()) {
        out('\x1b[32m  ✓ Ready — ' + cfg.provider + ' · ' + cfg.model + '\x1b[0m\n');
      } else {
        out('\x1b[90m  Setup skipped — run /provider any time to configure a model.\x1b[0m\n');
      }
      maybeRefreshPanel();
    }
    while (!__chat_exit_check()) {
      let notices = [];
      try {
        const poll = JSON.parse(__chat_poll());
        notices = poll.notices || [];
        if (poll.shared) lastShared = poll.shared;
      } catch (e) {}
      for (const n of notices) out('\x1b[90m  ⇄ ' + n.id + ' · ' + n.kind + ': ' + n.text + '\x1b[0m');
      /* F8: filesystem watcher — poll for changes each tick. */
      try {
        const watchList = JSON.parse(__chat_watch_list());
        if (watchList.length > 0) {
          const changes = JSON.parse(__chat_watch_poll ? __chat_watch_poll() : '[]');
          for (const c of changes) {
            out('\x1b[90m  📂 ' + c.path + ' (' + c.kind + ')\x1b[0m');
            watchChangesPending.push(c);
          }
          /* M1: hard cap — the pending queue must never grow unbounded in
           * a long /watch session (turn() drains it each turn anyway). */
          if (watchChangesPending.length > 100) {
            watchChangesPending.splice(0, watchChangesPending.length - 100);
          }
        }
      } catch (e) {}
      const inp = await __readline('\x1b[' + T('accent') + 'm❯\x1b[0m ');
      if (inp === null || inp === undefined) { requestExit(); break; }
      const t = inp.trim();
      if (t === '') continue;
      if (t[0] === '/') {
        /* Snapshot the settings this command could touch, so the panel
         * refresh below can be gated on an ACTUAL change rather than on the
         * command's name. */
        const before = { p: cfg.provider, m: cfg.model, e: cfg.effort,
                         x: cfg.ctx_window, o: cfg.max_output, b: cfg.brain,
                         ml: cfg.ml, rl: cfg.rlm, sy: cfg.sync, th: cfg.theme,
                         pm: cfg.permissions, api: cfg.api_key, bu: cfg.base_url };
        const r = __chat_slash(t);
        // Slash commands may have changed provider/model/api_key — re-read
        // the Rust-side config so the live session uses the new values.
        try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
        /* A theme switch must recolor the very next paint, not next
         * session: drop the cached palette on every command (re-read is
         * one JSON parse per command — noise). */
        PAL = null;
        /* Any command that can change settings re-renders the welcome panel
         * so the header (model/provider/effort) updates instantly. */
        const cfgCmds = ['/model', '/provider', '/effort', '/ctx', '/maxout', '/brain', '/ml', '/rlm', '/sync', '/outputs', '/mode', '/plan', '/edit', '/full', '/theme'];
        const isCfgCmd = cfgCmds.some(c => t === c || t.indexOf(c + ' ') === 0);
        /* ...but only repaint when something really moved. Gating on the
         * name alone re-printed the whole banner for a bare `/ctx`, which
         * changes nothing and can never open its picker without a TTY — so
         * piping a session showed the header twice. Under a TTY the repaint
         * is in place (tui_reset) and harmless; in a pipe it APPENDS, and
         * that duplicate is the bug. */
        const changed = isCfgCmd && (
             before.p !== cfg.provider  || before.m  !== cfg.model
          || before.e !== cfg.effort   || before.x  !== cfg.ctx_window
          || before.o !== cfg.max_output || before.b !== cfg.brain
          || before.ml !== cfg.ml      || before.rl !== cfg.rlm
          || before.sy !== cfg.sync    || before.pm !== cfg.permissions
          || before.th !== cfg.theme
          || before.api !== cfg.api_key || before.bu !== cfg.base_url);
        if (isCfgCmd) {
          /* A mode change must reach the engine runtime, not just the panel:
           * applyMode() re-pushes cfg.permissions into sofuu.agent so the
           * next turn is gated immediately, and refreshStatus re-renders
           * the amber mode chip. */
          applyMode();
          refreshStatus(lastChip);
        }
        if (changed) maybeRefreshPanel();
        /* F2 detect-on-select: a model/provider switch is exactly when the
         * user needs the NEW model's real window, and the 7-day discovery
         * TTL usually has nothing cached for a root just selected. Force
         * the harvest for the active endpoint and report what it found —
         * a silent 32k for a 1M model (or vice versa) is the bug this
         * removes. */
        const reselect = t === '/model' || t.indexOf('/model ') === 0 ||
                         t === '/provider' || t.indexOf('/provider ') === 0;
        if (reselect && (r === 'ok' || r === 'pick_model' || r === 'pick_provider')) {
          await detectCapsNow();
        }
        if (r === 'pick_model') {
          if (TTY) await pickModel();
          else out('\x1b[90m  Usage: /model <name>  (the picker needs a TTY)\x1b[0m\n');
        } else if (r === 'pick_provider') {
          if (TTY) await pickProvider();
          else out('\x1b[90m  Usage: /provider <name> [url]  (the picker needs a TTY)\x1b[0m\n');
        } else if (r === 'pick_effort') {
          if (TTY) await pickEffort();
          else out('\x1b[90m  Usage: /effort <off|low|medium|high|max>  (current: ' + (cfg.effort || 'off') + ')\x1b[0m\n');
        } else if (r === 'pick_rlm') {
          if (TTY) await pickRlm();
          else out('\x1b[90m  Usage: /rlm <on|off|auto>  (current: ' + (cfg.rlm || 'off') + ')\x1b[0m\n');
        } else if (r === 'pick_ctx') {
          if (TTY) await pickCtx();
          else out('\x1b[90m  Usage: /ctx <tokens|default>  (0-' + (cfg.ctx_cap || 0) + '; current: ' + (cfg.ctx_window || 'provider default') + ')\x1b[0m\n');
        } else if (r === 'pick_maxout') {
          if (TTY) await pickMaxout();
          else out('\x1b[90m  Usage: /maxout <tokens|default>  (0-' + (cfg.max_output_cap || 0) + '; current: ' + (cfg.max_output || 'provider default') + ')\x1b[0m\n');
        } else if (r === 'pick_brain') {
          if (TTY) await pickBrain();
          else out('\x1b[90m  Usage: /brain <on|off>  (current: ' + (cfg.brain ? 'on' : 'off') + ')\x1b[0m\n');
        } else if (r === 'pick_sync') {
          if (TTY) await pickSync();
          else out('\x1b[90m  Usage: /sync <on|off>  (current: ' + (cfg.sync ? 'on' : 'off') + ')\x1b[0m\n');
        } else if (r === 'pick_theme') {
          if (TTY) { await pickTheme(); }
          else {
            let themes = [];
            try { themes = JSON.parse(__chat_themes()); } catch (e) {}
            out('\x1b[90m  Usage: /theme <name>  (current: ' + (cfg.theme || 'sofuu') + ')\x1b[0m');
            for (const th of themes) {
              const cur = th.name === cfg.theme ? ' \x1b[32m← current\x1b[0m' : '';
              out('  \x1b[36m' + th.name + '\x1b[0m' + cur);
            }
            out('');
          }
        } else if (r === 'clear') {
          history = [];
          /* /compact resets the meter (below) — /clear must too, or the
           * footer shows stale tokens until the next turn re-syncs (P2-9).
           * The calibration stays: the per-request overhead is a property
           * of the provider, not of the (now empty) history. */
          usedCtx = ctxMeter();
          out('\x1b[90m  ✓ session cleared\x1b[0m\n');
        }
        else if (r === 'compact') { await compact(); }
        else if (r === 'tools') { await connectMcpServers(); showTools(); }
        else if (r === 'agents') { await handleAgentsCmd(); }
        // F1–F11 feature dispatch
        else if (r === 'remember') { await handleRemember(t.replace(/^\/remember\s*/, '').trim()); }
        else if (r === 'why') { handleWhy(); }
        else if (r === 'resume') { await handleResume(t.replace(/^\/resume\s*/, '').trim()); }
        else if (r === 'sessions') { await handleSessions(t.replace(/^\/sessions\s*/, '').trim()); }
        else if (r === 'share') { await handleShare(t.replace(/^\/share\s*/, '').trim()); }
        else if (r === 'import') { await handleImport(t.replace(/^\/import\s*/, '').trim()); }
        else if (r === 'cost') { handleCost(); }
        else if (r === 'outputs') { handleOutputs(t.replace(/^\/outputs\s*/, '').trim()); }
        else if (r === 'watch') {
          var watchArg = t.replace(/^\/watch\s*/, '').trim();
          handleWatch(watchArg);
        }
        // 'unknown' prints its message (with did-you-mean) in Rust
        // handle_slash — nothing more to show here.
        continue;
      }
      /* One logical user turn may have several provider attempts. Keep one
       * stable turn number, but give every retry its own archive attempt and
       * allow at most one narrowly selected recovery lookup. */
      archiveTurn++;
      archiveAttempt = 1;
      archivePendingRecovery = '';
      archiveLastFailure = null;
      archiveRecoveryUsed = false;
      /* A failed turn must never kill the chat: print the error and loop
       * back to the prompt. (Without this, an API/config exception rejects
       * the driver's main() promise and the CLI exits immediately.)
       *
       * Thinking-support runtime detection: when the provider rejects a
       * reasoning parameter for this model, remember it (persisted), tell
       * the user, and retry the turn once WITHOUT effort. */
      const THINK_ERR_RE = /thinking|reasoning_effort|extended.?thinking|reasoning/i;
      /* Provider failover (P1): capacity/auth-class failures (429 / 5xx /
       * rate-limit / 401/403 — a bad key is specific to that provider
       * entry) fall through to the next configured provider; validation
       * errors (400) stay fatal. One no-effort retry per model — the
       * reasoning-parameter rejection is per-model, persisted via
       * __chat_no_think. */
      const CAPACITY_ERR_RE = /HTTP (40[13]|429|5\d\d)|rate.?limit|overloaded|temporarily|quota|invalid token|unauthorized|forbidden|api key/i;
      const chain = failoverChain();
      const noThinkTried = {};
      let chainIdx = 0;
      let turnErr = null;
      function prepareArchiveRecovery(message) {
        if (!archiveLastFailure || archiveRecoveryUsed) return;
        archiveRecoveryUsed = true;
        const ids = [];
        if (archiveLastFailure.errorId) ids.push(archiveLastFailure.errorId);
        if (archiveLastFailure.partialId) ids.push(archiveLastFailure.partialId);
        const recovered = archiveContext({
          turn: archiveTurn,
          attempt: archiveAttempt,
          error_code: archiveLastFailure.errorCode || archiveErrorCode(message),
          related_output_ids: ids,
          current_task: t,
          automatic: true,
        });
        if (recovered) archivePendingRecovery = recovered;
      }
      for (;;) {
        const cur = chain[chainIdx];
        const curModel = cur ? cur.model : cfg.model;
        try {
          await turn(t, cur, archiveTurn, archiveAttempt);
          turnErr = null;
          break;
        } catch (e) {
          failoverTo = null;
          turnErr = e;
          const msg = String((e && e.message) || e);
          /* Setup/MCP/hook failures can happen before the agent's run
           * boundary. Preserve those too, while keeping the archive path
           * non-fatal to the normal retry loop. */
          if (!archiveLastFailure) {
            const errorCode = archiveErrorCode(msg);
            const errorRef = archiveWrite('error', 'failed',
              errorCode === 'runtime' ? 'runtime' : 'provider', msg, {
                error_code: errorCode,
                turn_complete: false,
              });
            archiveLastFailure = { errorId: errorRef && errorRef.id, partialId: null, errorCode: errorCode };
          }
          if (THINK_ERR_RE.test(msg) && cfg.effort && curModel &&
              !noThinkTried[curModel] &&
              !(cfg.no_think_models || []).includes(curModel)) {
            noThinkTried[curModel] = true;
            try { __chat_no_think(curModel); } catch (eNT) {}
            /* session-2 e2e catch: __chat_no_think only updates the Rust
             * config — this JS mirror still showed an empty
             * no_think_models, so the "no-effort retry" re-sent
             * reasoning_effort and hit the same rejection (the retry was a
             * no-op on the wire; every later turn repeated it too).
             * Reload the mirror, exactly like the provider wizard does. */
            try { cfg = JSON.parse(__chat_getcfg()); } catch (eCFG) {}
            out('\x1b[33m  ⓘ ' + curModel + ' rejected thinking (' + msg.slice(0, 160) +
                ') — disabling effort for this model and retrying.\x1b[0m\n');
            prepareArchiveRecovery(msg);
            archiveAttempt++;
            continue;
          }
          if (CAPACITY_ERR_RE.test(msg) && chainIdx + 1 < chain.length) {
            chainIdx++;
            const next = chain[chainIdx];
            out('\x1b[33m  ⓘ ' + (chain[chainIdx - 1] ? chain[chainIdx - 1].name : cfg.provider) +
                ' unavailable (' + msg.slice(0, 120) + ') — failing over to ' +
                next.name + ' / ' + next.model + '\x1b[0m\n');
            prepareArchiveRecovery(msg);
            archiveAttempt++;
            continue;
          }
          break;
        }
      }
      failoverTo = null;
      archiveLastFailure = null;
      if (turnErr) out('\x1b[31m  ✗ ' + String((turnErr && turnErr.message) || turnErr) + '\x1b[0m\n');
    }
    /* Brain flush happens per-store inside sofuu.agent.run (agent.js). */
    /* Disconnect every MCP server: their child processes hold ref'd libuv
     * handles, which would pin the event loop and hang the exit. */
    for (const c of mcpClients) { try { c.client.disconnect(); } catch (e) {} }
    /* Stop the metrics timer (it would pin the loop). */
    try { if (globalThis.__metrics_stop) globalThis.__metrics_stop(); } catch (e) {}
    /* Leave the full-screen interface: restore the previous terminal. */
    if (TTY) {
      try { __tui_off(); } catch (e) {}
      process.stdout.write('\n');
    }
  }
  return main();
})()
"#;

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
    rt.eval_string(DRIVER, "<chat>")
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
        // DRIVER is the string VALUE of the r#"…"# const — the literal
        // markers live in the declaration, not in the value, so scanning
        // DRIVER for `r#"` always returned None and this test failed on
        // every run. It was pre-existing and hidden because `make test`
        // only runs the library target's --lib tests, not --bin sofuu.
        // The value starts with the IIFE directly.
        let js = DRIVER;
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
        check_parse("<chat-driver>", DRIVER, true);
        check_parse("agent.js", include_str!("../../../src/js/agent.js"), true);
        check_parse("tools.js", include_str!("../../../src/js/tools.js"), true);
        check_parse("chat.js", include_str!("../../../src/js/chat.js"), true);
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
