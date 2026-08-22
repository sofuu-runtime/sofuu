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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::session;

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
}

#[derive(Clone, Debug, Default)]
pub struct ChatConfig {
    pub provider: String,
    pub model: String,
    pub effort: String,
    pub brain: bool,
    pub sync: bool,
    /// API key for the CURRENT (active) provider, persisted in ~/.sofuu/config.json.
    /// An env var (e.g. OPENAI_API_KEY) still takes precedence at request
    /// time (the C layer prefers opts.api_key, then the env).
    pub api_key: String,
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
    /// Per-session context window in tokens (used for history trimming and
    /// RLM routing). 0 = provider default (see default_ctx_window).
    pub ctx_window: i64,
    /// Per-session max OUTPUT tokens per response. 0 = provider default.
    pub max_output: i64,
    /// Optional REMOTE embeddings provider+model for the brain (recall /
    /// store). Both empty (default) = always use the bundled, offline
    /// `sofuu.ai.embedLocal` — no embeddings provider is required.
    pub embed_provider: String,
    pub embed_model: String,
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
    /// 0 = default (1024, agent.js RECALL_BUDGET_TOK).
    pub recall_budget: i64,
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
        pricing.insert("gemini/gemini-1.5-pro".into(), [1.25, 5.00]);
        pricing.insert("gemini/gemini-2.0-flash".into(), [0.10, 0.40]);
        Self {
            provider: String::new(),
            model: String::new(),
            effort: "high".into(),
            brain: false,
            sync: true,
            api_key: String::new(),
            base_url: String::new(),
            profile: String::new(),
            providers: Vec::new(),
            active: String::new(),
            rlm: String::new(),
            embed_provider: String::new(),
            embed_model: String::new(),
            ctx_window: 0,
            max_output: 0,
            pricing,
            budget_usd: 0.0,
            spend_total_usd: 0.0,
            ghost: false,
            rss_warn_mb: 1024,
            recall_min: 0.0,
            recall_budget: 0,
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
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
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
                cfg.providers.push(ProviderEntry { name, endpoint, api_key, model, profile });
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
            cfg.providers.push(ProviderEntry { name: legacy_provider.clone(), endpoint: legacy_base_url.clone(), api_key: legacy_api_key.clone(), model: legacy_model.clone(), profile: legacy_profile.clone() });
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
        if let Some(b) = v.get("sync").and_then(|x| x.as_bool()) { cfg.sync = b; }
        if let Some(s) = v.get("rlm").and_then(|x| x.as_str()) { cfg.rlm = s.to_string(); }
        if let Some(s) = v.get("embed_provider").and_then(|x| x.as_str()) { cfg.embed_provider = s.to_string(); }
        if let Some(s) = v.get("embed_model").and_then(|x| x.as_str()) { cfg.embed_model = s.to_string(); }
        if let Some(n) = v.get("ctx_window").and_then(|x| x.as_i64()) { cfg.ctx_window = clamp_ctx_window(n); }
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
            let mut flat_key = self.api_key.clone();
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
                "sync": self.sync,
                "api_key": flat_key,
                "base_url": flat_base,
                "profile": flat_profile,
                "rlm": self.rlm,
                "embed_provider": self.embed_provider,
                "embed_model": self.embed_model,
                "ctx_window": self.ctx_window,
                "max_output": self.max_output,
                "pricing": self.pricing,
                "budget_usd": self.budget_usd,
                "spend_total_usd": self.spend_total_usd,
                "ghost": self.ghost,
                "rss_warn_mb": self.rss_warn_mb,
                "recall_min": self.recall_min,
                "recall_budget": self.recall_budget,
            })
            .to_string();
            let tmp = dir.join("config.json.tmp");
            let _ = std::fs::write(&tmp, &json);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
            }
            let _ = std::fs::rename(&tmp, Self::config_path());
        }
    }
}

// ── Slash commands ──────────────────────────────────────────────

const VALID_PROVIDERS: [&str; 3] = ["openai", "anthropic", "local"];
const VALID_EFFORTS: [&str; 4] = ["low", "medium", "high", "max"];

/// Hard ceiling on a user-configured context window: 1M tokens.
/// Real 1M-window models (e.g. Gemini 2.5 Pro) accept it; smaller models
/// still reject oversized requests at the API and the error surfaces.
const MAX_CTX_WINDOW: i64 = 1_000_000;

/// Hard ceiling on a user-configured max OUTPUT tokens: 384k. Models like
/// Gemini 2.5 Pro support up to 64k output natively; some reasoning models
/// and long-form agents accept far more. The API rejects requests above a
/// model's real ceiling and the error surfaces.
const MAX_OUTPUT_TOKENS: i64 = 384_000;

/// Per-provider default context windows (tokens), used when ctx_window is 0.
/// Kept deliberately tiny (current mainstream limits); users on bigger-window
/// models raise it with /ctx or --ctx-window.
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
const ALL_COMMANDS: [&str; 32] = [
    "/help",
    "/version",
    "/model",
    "/provider",
    "/effort",
    "/compact",
    "/clear",
    "/brain",
    "/rlm",
    "/ctx",
    "/maxout",
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
    ("/effort", "Set the thinking/reasoning effort", "off | low | medium | high | max"),
    ("/compact", "Summarize the conversation into context", ""),
    ("/clear", "Reset this session (keep settings)", ""),
    ("/brain", "Toggle the memory/brain integration", "on | off"),
    ("/rlm", "Route long-context turns through RLM", "on | off | auto"),
    ("/ctx", "View/set the context window in tokens", "<tokens> | default  (max 1M)"),
    ("/maxout", "View/set max output tokens per response", "<tokens> | default  (max 384k)"),
    ("/tools", "List connected MCP servers + tools", ""),
    ("/agents", "List agent definitions (+ ~/.sofuu/agents/*.js)", ""),
    ("/sessions", "List sessions on this project", ""),
    ("/context", "Show a session's full context", "[id]  (default: this session)"),
    ("/work", "Announce what you are working on", "<description>"),
    ("/done", "Clear your current task", ""),
    ("/note", "Record a personal note (peers see it)", "<message>"),
    ("/notify", "Broadcast a critical notice to all sessions", "<message>"),
    ("/sync", "Toggle the session-mesh polling", "on | off"),
    ("/exit", "Save config and exit", ""),
    // F1–F11 chat features
    ("/remember", "Pin a fact directly to the brain", "<fact>"),
    ("/why", "Explain which memories shaped the last answer", ""),
    ("/resume", "Browse and resume a past session", "[id]  (bare = picker)"),
    ("/share", "Export brain as an encrypted card", "[path.qtsq] [label]"),
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
        // ── Session mesh commands ──────────────────────────────
        "/sessions" => {
            let project = PROJECT.lock().unwrap().clone();
            match project {
                Some(p) => {
                    session::cmd_list(&p);
                }
                None => chat_out(&format!("  [sync] session mesh is off (--sync on to enable)\n")),
            }
            "ok"
        }
        "/context" => {
            let (project, own) = {
                let p = PROJECT.lock().unwrap().clone();
                let own = SESS.lock().unwrap().as_ref().map(|s| s.id().to_string());
                (p, own)
            };
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
            cfg.model = arg.to_string();
            if let Some(entry) = cfg.providers.iter_mut().find(|p| p.name == cfg.active) { entry.model = cfg.model.clone(); }
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
                let entry = ProviderEntry { name: name.clone(), endpoint: if base.is_empty() { existing.as_ref().map(|e| e.endpoint.clone()).unwrap_or_default() } else { base }, api_key: existing.as_ref().map(|e| e.api_key.clone()).unwrap_or_default(), model: existing.as_ref().map(|e| e.model.clone()).unwrap_or_default(), profile: existing.as_ref().map(|e| e.profile.clone()).unwrap_or_default() };
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
                chat_out(&format!("  ✓ Effort → {}\n", cfg.effort));
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
        "/tools" => "tools",
        "/agents" => "agents",
        "/ctx" => {
            // /ctx          → config panel (DRIVER handles "pick_ctx")
            // /ctx <n>      → set (tokens; clamped to [1024, 1M]; 0 = default)
            // /ctx default  → reset to the provider default
            if arg.is_empty() {
                "pick_ctx"
            } else if arg == "default" || arg == "0" {
                cfg.ctx_window = 0;
                cfg.save();
                let eff = effective_ctx_window(&cfg.provider, cfg.ctx_window);
                chat_out(&format!("  ✓ Context window → provider default ({eff} tokens)\n"));
                "ok"
            } else if let Ok(n) = arg.parse::<i64>() {
                if n < 0 || n > MAX_CTX_WINDOW {
                    chat_out(&format!(
                        "  Context window must be between 0 and {} tokens (0 = provider default)\n",
                        MAX_CTX_WINDOW
                    ));
                } else {
                    cfg.ctx_window = clamp_ctx_window(n);
                    cfg.save();
                    let eff = effective_ctx_window(&cfg.provider, cfg.ctx_window);
                    chat_out(&format!("  ✓ Context window → {eff} tokens\n"));
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
                chat_out(&format!("  ✓ Max output tokens → provider default\n"));
                "ok"
            } else if let Ok(n) = arg.parse::<i64>() {
                if n < 0 || n > MAX_OUTPUT_TOKENS {
                    chat_out(&format!(
                        "  Max output tokens must be between 0 and {} (0 = provider default)\n",
                        MAX_OUTPUT_TOKENS
                    ));
                } else {
                    cfg.max_output = clamp_max_output(n);
                    cfg.save();
                    chat_out(&format!("  ✓ Max output tokens → {}\n", cfg.max_output));
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
                chat_out("  Pins a fact directly to the brain (survives decay).\n");
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
    let h = |s: &str| chat_out(&format!("\n\x1b[1;36m{s}\x1b[0m")); // section header
    let c = |cmd: &str, desc: &str| {
        // aligned command column + dim description
        chat_out(&format!("    \x1b[1m{cmd:<20}\x1b[0m\x1b[2m— {desc}\x1b[0m"));
    };
    chat_out("");
    h("  Sofuu Chat Commands");
    c("/help", "this help");
    c("/model", "interactive model picker (search, arrows, enter)");
    c("/model <name>", "switch model directly (persisted)");
    c("/provider", "your providers + add a new one");
    c("/effort", "reasoning-effort picker (off/low…max)");
    c("/compact", "summarize the conversation into context");
    c("/clear", "reset this session (keep settings)");
    c("/brain [on|off]", "toggle memory/brain integration");
    c("/rlm [on|off|auto]", "route long-context turns through the RLM loop");
    c("/ctx [<tokens>]", "view/set the context window (max 1M)");
    c("/maxout [<tokens>]", "view/set max output tokens (max 384k)");
    c("/tools", "list connected MCP servers + their tools");
    c("/agents", "list agent definitions (~/.sofuu/agents/*.js)");
    c("/version", "print the runtime version");
    h("  Brain & memory");
    c("/remember <fact>", "pin a fact directly to the brain");
    c("/why", "which memories shaped the last answer");
    c("/share [path]", "export a brain card (metadata + pointers)");
    c("/import <path>", "import + merge a brain card");
    c("/resume [id]", "browse + resume a past session");
    c("/serve", "brain-server quick info (sofuu serve --brain)");
    h("  Context & tools");
    c("@file[:start-end]", "attach file contents to a prompt");
    c("@name <task>", "run a loaded agent directly (@agent:name too)");
    c("/watch <path>", "watch a directory for changes in context");
    c("/cost", "token usage + spend breakdown + budget");
    c("/ghost [on|off]", "toggle ghost prompt completion");
    c("/hooks", "show ~/.sofuu/hooks.js user middleware info");
    h("  Session mesh");
    c("/sessions", "list all sessions on this project");
    c("/context [id]", "full context of a session (default: this one)");
    c("/work <desc>", "announce what you are working on");
    c("/done", "clear your current task");
    c("/note <msg>", "record a personal note (seen by peers)");
    c("/notify <msg>", "CRITICAL notice, shown to every session now");
    c("/sync [on|off]", "toggle session-mesh polling (restart to join)");
    c("/exit /quit", "save config and exit");
    chat_out(&format!("\n\x1b[2m  Sessions on the SAME project share context in real time (tasks,\x1b[0m"));
    chat_out(&format!("\x1b[2m  notes, critical notices). Data persists as .qtsq files in\x1b[0m"));
    chat_out(&format!("\x1b[2m  <project>/.sofuu/sessions/.\x1b[0m"));
    h("\n  Shortcuts");
    c("TAB", "accept slash-command completion");
    c("↑ / ↓", "input history");
    c("PgUp / PgDn", "scroll the conversation");
    c("Shift+Enter", "newline inside the input (Alt+Enter too)");
    c("Esc", "stop the response (while streaming) / clear the input");
    c("Ctrl-C", "quit sofuu");
    c("Ctrl-D", "exit");
    chat_out("");
}

// ── Welcome panel ────────────────────────────────────────────────
// The bordered box shown at the top of the TUI: logo tile + title, then
// column-aligned session facts. Built in Rust (testable), rendered by the
// C conversation area line by line.

/// Shima-enaga (long-tailed tit) mascot — a round fluffy white bird with
/// black bead eyes, a tiny orange beak and a fixed long pink tail. The
/// mascot NEVER moves: every frame is exactly 7 cells wide (body 6 + tail)
/// with the tail pinned at the same spot; only the EYES change in place
/// (open → blink → happy → blink). Row 1 is also 7 cells so the title
/// text column aligns across all three rows.
// Mascot mark — ASCII-only on purpose. The previous art used U+25xx
// geometric glyphs (░▒█●▸◕) and U+203E, which are East-Asian
// ambiguous-width: terminals that render them 2 cells wide broke the
// panel's right-border alignment on exactly the mascot rows. The mark is
// a compact one-line bird whose eyes animate: open (o) → blink (-) →
// happy (^) → blink (-) — every character is 1 cell in every terminal.
const SHIMA_MARKS: [&str; 4] = [
    // open eyes
    "\x1b[38;5;231m(\x1b[0m\x1b[38;5;231mo\x1b[0m\x1b[38;5;214m>\x1b[0m\x1b[38;5;231m)\x1b[0m",
    // blink
    "\x1b[38;5;231m(\x1b[0m\x1b[38;5;231m-\x1b[0m\x1b[38;5;214m>\x1b[0m\x1b[38;5;231m)\x1b[0m",
    // happy
    "\x1b[38;5;231m(\x1b[0m\x1b[38;5;214m^\x1b[0m\x1b[38;5;214m>\x1b[0m\x1b[38;5;231m)\x1b[0m",
    // blink
    "\x1b[38;5;231m(\x1b[0m\x1b[38;5;231m-\x1b[0m\x1b[38;5;214m>\x1b[0m\x1b[38;5;231m)\x1b[0m",
];

/// The current logo frame (0-3) — rotated by the JS animation loop.
static LOGO_FRAME: AtomicUsize = AtomicUsize::new(0);

/// Display width in terminal cells: skips ANSI CSI sequences; wide chars
/// (CJK/emoji) count 2 cells like a real terminal. Kept in sync with
/// char_cells/tui_disp_width() in rt/tui.rs — the panel pads with this,
/// the viewport truncates with that, and they must agree.
fn char_cells(c: char) -> usize {
    let o = c as u32;
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

/// Truncate plain (escape-free) text to `max` cells, marking the cut.
fn trunc_cells(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let t: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{t}…")
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
            if cells >= max {
                clipped = true;
                break;
            }
            cells += 1;
        }
        out.push(b[i]);
        i += 1;
    }
    if clipped {
        out.extend_from_slice(b"\x1b[0m");
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// One box row: `│  <content><pad>│` at exactly `width` cells.
fn panel_row(content: &str, width: usize) -> String {
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
    format!(
        "\x1b[2;35m│\x1b[0m  {}{}\x1b[2;35m│\x1b[0m",
        content,
        " ".repeat(pad)
    )
}

/// The 2 welcome-panel rows that contain the mascot (mark + title, then
/// the subtitle aligned under the title) for a GIVEN frame. Used both by
/// the initial panel build and by the animation overlay, so the eyes swap
/// frames WITHOUT moving or erasing the surrounding text/borders.
fn welcome_bird_rows(
    _cfg: &ChatConfig,
    _session: &str,
    _dir: &str,
    w: usize,
    frame: usize,
) -> Vec<String> {
    let mark = SHIMA_MARKS[frame % SHIMA_MARKS.len()];
    vec![
        panel_row(
            &format!("{}   \x1b[1;35mWelcome to Sofuu!\x1b[0m", mark),
            w,
        ),
        panel_row(
            "       \x1b[2mAsk anything (/help for the command list)\x1b[0m",
            w,
        ),
    ]
}

/// The full welcome panel, one String per display row, for a terminal
/// `width` cells wide. `session` = mesh short id ("" = no session).
fn welcome_panel_at(cfg: &ChatConfig, session: &str, dir: &str, width: usize) -> Vec<String> {
    let w = width.clamp(40, 400);
    let inner = w - 4;
    let frame = LOGO_FRAME.load(Ordering::Relaxed) % SHIMA_MARKS.len();
    let mut rows = Vec::with_capacity(12);

    let dash = "─".repeat(w - 2);
    rows.push(format!("\x1b[2;35m╭{dash}╮\x1b[0m"));
    rows.extend(welcome_bird_rows(cfg, session, dir, w, frame));
    rows.push(panel_row("", w));

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
    let labels: [(String, String); 5] = [
        ("Directory:".into(), trunc_cells(dir, inner.saturating_sub(14))),
        (
            "Session:".into(),
            if session.is_empty() { "none".into() } else { session.into() },
        ),
        ("Model:".into(), model),
        (
            "Memory:".into(),
            if cfg.brain {
                "on (persists across sessions)".to_string()
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
        ));
    }
    rows.push(format!("\x1b[2;35m╰{dash}╯\x1b[0m"));

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
    /// providers that don't report them (P6.4).
    turns: Vec<(String, u64, u64, f64, u64, u64)>,
    /// Cumulative session spend in USD.
    session_total: f64,
    /// Cumulative cache-hit tokens this session (P6.4 surface).
    cache_read_total: u64,
}

impl SessionCost {
    const fn new() -> Self {
        Self { turns: Vec::new(), session_total: 0.0, cache_read_total: 0 }
    }
}

/// F8: filesystem watcher — tracks mtimes/sizes of watched paths.
struct FsWatcher {
    /// (path, HashMap<file_path, (mtime, size)>)
    paths: Vec<(PathBuf, std::collections::HashMap<PathBuf, (u64, u64)>)>,
}

impl FsWatcher {
    const fn new() -> Self {
        Self { paths: Vec::new() }
    }

    fn add(&mut self, path: &str) {
        let p = PathBuf::from(path);
        if self.paths.iter().any(|(watched, _)| *watched == p) {
            return; // already watching
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
        let cfg = guard.as_ref().cloned().unwrap_or_default();
        serde_json::json!({
            "provider": cfg.provider,
            "model": cfg.model,
            "effort": cfg.effort,
            "brain": cfg.brain,
            "sync": cfg.sync,
            "api_key": cfg.api_key,
            "base_url": cfg.base_url,
            "profile": cfg.profile,
            "providers": cfg.providers,
            "active": cfg.active,
            "rlm": cfg.rlm,
            "ctx_window": cfg.ctx_window,
            "max_output": cfg.max_output,
            "embed_provider": cfg.embed_provider,
            "embed_model": cfg.embed_model,
            "pricing": cfg.pricing,
            "budget_usd": cfg.budget_usd,
            "spend_total_usd": cfg.spend_total_usd,
            "ghost": cfg.ghost,
            "rss_warn_mb": cfg.rss_warn_mb,
            "recall_min": cfg.recall_min,
            "recall_budget": cfg.recall_budget,
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
            let existing_model = cfg.providers.iter().find(|p| p.name == name).map(|p| p.model.clone()).unwrap_or_default();
            let final_model = if model.is_empty() { existing_model } else { model };
            cfg.upsert_provider(ProviderEntry { name, endpoint, api_key, model: final_model, profile });
            cfg.save();
        }
    }
    js_new_bool(ctx, true)
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
    let pt = js_to_int(ctx, *argv.add(1));
    let ct = js_to_int(ctx, *argv.add(2));
    let cache_read = if argc > 3 { js_to_int(ctx, *argv.add(3)).max(0) } else { 0 };
    let cache_write = if argc > 4 { js_to_int(ctx, *argv.add(4)).max(0) } else { 0 };
    let result = if let Ok(mut guard) = CFG.lock() {
        let cfg = guard.get_or_insert_with(ChatConfig::defaults);
        let key = format!("{}/{}", cfg.provider, model);
        let [in_p, out_p] = cfg.pricing.get(&key).cloned().unwrap_or([0.0, 0.0]);
        let cost = (pt as f64 / 1_000_000.0) * in_p + (ct as f64 / 1_000_000.0) * out_p;
        SESSION_COST.lock().unwrap().turns.push((model, pt as u64, ct as u64, cost, cache_read as u64, cache_write as u64));
        SESSION_COST.lock().unwrap().session_total += cost;
        SESSION_COST.lock().unwrap().cache_read_total += cache_read as u64;
        cfg.spend_total_usd += cost;
        cfg.save();
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
        s.trim().parse::<i64>().unwrap_or(0)
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
        let gutter = sofuu_core::rt::tui::GUTTER;
        // The renderer indents every conversation row by GUTTER cells, so the
        // panel is built GUTTER cells narrower and pre-indented here — the
        // box borders land inside the viewport instead of under the "…" clip.
        let pad = " ".repeat(gutter);
        let rows = welcome_panel_at(&cfg, sess.as_deref().unwrap_or(""), &dir, w.saturating_sub(gutter));
        for row in &rows {
            sofuu_ffi::tui_log(&format!("{pad}{row}"));
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

/// `__chat_logo()` — advance to the next animation frame and overlay the
/// 2 mascot rows (mark + title, subtitle) over the welcome panel at fixed
/// screen rows 2-3. The eyes swap frames in place: nothing moves and no
/// text is erased.
unsafe extern "C" fn js_chat_logo(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let next = LOGO_FRAME.load(Ordering::Relaxed).wrapping_add(1);
    LOGO_FRAME.store(next % SHIMA_MARKS.len(), Ordering::Relaxed);
    if sofuu_ffi::tui_active() {
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
        let w = sofuu_ffi::tui_width();
        let frame = LOGO_FRAME.load(Ordering::Relaxed) % SHIMA_MARKS.len();
        // Match js_chat_welcome: rows are logged GUTTER-indented at a
        // GUTTER-narrower width, so the overlay must use the same shape.
        let gutter = sofuu_core::rt::tui::GUTTER;
        let pad = " ".repeat(gutter);
        for (i, row) in welcome_bird_rows(&cfg, sess.as_deref().unwrap_or(""), &dir, w.saturating_sub(gutter), frame)
            .iter()
            .enumerate()
        {
            sofuu_ffi::tui_overlay_row(2 + i as i32, &format!("{pad}{row}"));
        }
    }
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

fn register_bridge(rt: &SofuuRuntime) {
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
        register_global_fn(ctx, "__chat_complete", js_chat_complete as JSCFunction);
        register_global_fn(ctx, "__chat_command_info", js_chat_command_info as JSCFunction);
        register_global_fn(ctx, "__chat_apply_provider", js_chat_apply_provider as JSCFunction);
        register_global_fn(ctx, "__chat_select_provider", js_chat_select_provider as JSCFunction);
        register_global_fn(ctx, "__chat_remove_provider", js_chat_remove_provider as JSCFunction);
        register_global_fn(ctx, "__chat_past_turns", js_chat_past_turns as JSCFunction);
        register_global_fn(ctx, "__chat_welcome", js_chat_welcome as JSCFunction);
        register_global_fn(ctx, "__chat_logo", js_chat_logo as JSCFunction);
        register_global_fn(ctx, "__chat_phase", js_chat_phase as JSCFunction);
        register_global_fn(ctx, "__chat_rss", js_chat_rss as JSCFunction);
        register_global_fn(ctx, "__chat_gc", js_chat_gc as JSCFunction);
        register_global_fn(ctx, "__chat_refresh", js_chat_refresh as JSCFunction);
        register_global_fn(ctx, "__chat_mcpservers", js_chat_mcpservers as JSCFunction);
        // F1–F11 bridge functions
        register_global_fn(ctx, "__chat_set_recall", js_chat_set_recall as JSCFunction);
        register_global_fn(ctx, "__chat_get_recall", js_chat_get_recall as JSCFunction);
        register_global_fn(ctx, "__chat_sessions", js_chat_sessions as JSCFunction);
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
  let history = [];
  let lastShared = '';
  let totalTk = 0;    /* cumulative session tokens (footer metric) */
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
  /* brain handle for /remember + /why (cached; the agent runtime has its
   * own — this is a lightweight driver-level handle for direct brain ops). */
  let driverBrain = null;
  function brainPath() {
    try { return (cfg.brainPath) || ((env('HOME') || '.') + '/.sofuu_brain.qtsq'); }
    catch (e) { return (env('HOME') || '.') + '/.sofuu_brain.qtsq'; }
  }
  function env(n) { try { return process.env[n] || ''; } catch (e) { return ''; } }
  function ensureDriverBrain() {
    if (driverBrain) return driverBrain;
    if (!sofuu.memory || typeof sofuu.memory.open !== 'function') return null;
    try {
      driverBrain = sofuu.memory.open(brainPath(), 768);
      return driverBrain;
    } catch (e) { return null; }
  }
  async function embedText(text) {
    /* Local-first: use the bundled offline embedder (no network needed). */
    if (sofuu.ai && typeof sofuu.ai.embedLocal === 'function') {
      try { return await sofuu.ai.embedLocal(text); } catch (e) {}
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
    const brain = ensureDriverBrain();
    if (!brain) { out('\x1b[90m  Brain unavailable (QTSQ not linked?)\x1b[0m\n'); return; }
    const vec = await embedText(fact);
    if (!vec) { out('\x1b[90m  Embeddings unavailable — cannot store fact\x1b[0m\n'); return; }
    try {
      brain.remember(new Float32Array(vec), fact, 'user_pin', 0);
      brain.flush();
      out('\x1b[90m  ⏺ remembered · ' + brain.count() + ' memories\x1b[0m\n');
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
    const short = sessionId.length > 8 ? sessionId.slice(0, 8) : sessionId;
    out('\x1b[90m  ⏺ resumed ' + short + ' · ' + turns.length + ' turns' +
        (capped ? ' (partial — kept most recent 30)' : '') + '\x1b[0m\n');
  }

  /* ── F3: @file mentions ──────────────────────────────────────────── */
  const ATTACH_TOKEN_BUDGET = 8192;
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
      }
      /* Resolve against cwd. */
      let resolved;
      try { resolved = sofuu.fs ? null : null; } catch (e) {}
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
        if (totalTokens + tok > ATTACH_TOKEN_BUDGET) {
          const remaining = ATTACH_TOKEN_BUDGET - totalTokens;
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
        try { sofuu.fs.writeFile(path + '.json', cardJson); }
        catch (e) { out('\x1b[33m  ⚠ Could not write card file: ' + String(e.message || e) + '\x1b[0m\n'); return; }
      }
      out('\x1b[90m  ⏺ exported ' + count + ' memories → ' + path + '.json\x1b[0m');
      out('\x1b[90m  Card metadata written. Brain file: ' + brainPath() + '\x1b[0m');
      out('\x1b[90m  Share both files; import with /import ' + path + '.json\x1b[0m\n');
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
          '  \x1b[2m' + t.input_tokens + '→' + t.output_tokens + ' tk\x1b[0m' + cache +
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
      const result = await Promise.race([
        hooks.pre({ text: text, cfg: cfg }),
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
  let ghostTimer = null;
  globalThis.__chat_ghost_check = function(prefix) {
    if (!cfg.ghost || !cfg.brain) return '';
    if (prefix.length < 8) return '';
    /* Debounce: return cached result if prefix hasn't changed enough. */
    if (prefix === ghostLastPrefix) return ghostLastResult;
    /* Only re-check if the last check was > 250ms ago. */
    if (ghostTimer) return ghostLastResult;
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

  function ctxBudget() {
    const win = cfg.ctx_window > 0
      ? cfg.ctx_window
      : ({ openai: 128000, anthropic: 128000, local: 32768 }[cfg.provider] || 32768);
    return Math.floor(win * 0.85);
  }
  function estTok(s) {
    if (sofuu.ai && typeof sofuu.ai.estimateTokens === 'function') {
      try { return sofuu.ai.estimateTokens(s) | 0; } catch (e) {}
    }
    return Math.ceil(String(s).length / 4);
  }
  function historyTokens() {
    let n = 0;
    for (const m of history) n += estTok(String(m.content || ''));
    return n;
  }
  /* Summarize all but the last keepTurns turns into one system entry
   * (same shape manual /compact has always produced). Returns false when
   * there is nothing to fold or the summarizer failed/returned junk —
   * callers fall back to dropping, never block the turn. */
  async function summarizeHistory(keepTurns) {
    const keepMsgs = Math.min(keepTurns * 2, history.length);
    const oldCount = history.length - keepMsgs;
    if (oldCount <= 0) return false;
    const summary = await complete([
      { role: 'system', content: 'You are a conversation summarizer. Compress the following conversation into a compact summary that preserves key facts, decisions, and the user\'s intent. Output only the summary.' },
      ...history.slice(0, oldCount)
    ]);
    if (!summary || typeof summary !== 'string' || !summary.trim() || summary.trim() === '(no response)') return false;
    history = [{ role: 'system', content: 'Prior conversation summary: ' + summary },
               ...history.slice(oldCount)];
    return true;
  }
  async function trimHistory() {
    /* Hard safety net: the absolute entry cap. */
    if (history.length > MAX_HISTORY_ENTRIES) {
      history.splice(0, history.length - MAX_HISTORY_ENTRIES);
    }
    const budget = ctxBudget();
    /* Re-arm one-shot compaction once usage drains back below half. */
    if (!autoCompactArmed && historyTokens() < budget * 0.5) autoCompactArmed = true;
    /* P5: auto-compaction — summarize instead of silently dropping. */
    if (autoCompactArmed && historyTokens() > budget * COMPACT_AT && history.length > COMPACT_KEEP_TURNS * 2) {
      autoCompactArmed = false;
      const savedTk = historyTokens();
      try {
        if (await summarizeHistory(COMPACT_KEEP_TURNS)) {
          out('\x1b[90m  ⚙ auto-compacted → summary (' +
              fmtTk(Math.max(0, savedTk - historyTokens())) + ' tk saved)\x1b[0m');
        }
      } catch (e) { /* summarizer failed — the drop loop below still guards */ }
    }
    /* Drop-oldest guard: still over budget after compaction (or compaction
     * failed)? Shed oldest pairs — always keep at least the last turn. */
    let droppedTurns = 0;
    while (history.length > 2 && historyTokens() > budget) {
      history.splice(0, 2);
      droppedTurns++;
    }
    if (droppedTurns > 0) {
      out('\x1b[33m  ⚠ history trimmed by ' + droppedTurns + ' turn' + (droppedTurns === 1 ? '' : 's') +
          ' to fit the ' + Math.floor(ctxBudget() / 100) / 10 + 'k-token context budget\x1b[0m');
    }
  }
  function streamOpts() {
    const o = { messages: [], provider: cfg.provider, model: cfg.model };
    if (cfg.effort) o.effort = cfg.effort;
    if (cfg.api_key) o.api_key = cfg.api_key;
    if (cfg.base_url) o.base_url = cfg.base_url;
    if (cfg.profile) o.profile = cfg.profile;
    if (cfg.max_output > 0) o.max_tokens = cfg.max_output;
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
  function usableConfig() {
    if (!cfg.provider || !cfg.model) return false;
    if (KNOWN.indexOf(cfg.provider) < 0 && !cfg.base_url) return false;
    if (cfg.provider === 'local') return false; /* not shipped yet */
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

  async function turn(text) {
    /* Unusable config = never call the API: one clean line, nothing else. */
    if (!usableConfig()) {
      out('\x1b[90m  No usable model configured — run /provider to set one up\x1b[0m\n');
      return;
    }
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
     * attach files). */
    const expanded = await expandMentions(text);
    const turnText = expanded.text;
    const manifest = expanded.manifest;
    try { __chat_log('prompt', String(typed)); } catch (e) {}
    /* The turn starts in Thinking, including while lazy MCP/agent setup runs. */
    CURRENT_PHASE = 'Thinking';
    startPhase();
    if (TTY) out('\x1b[1;35m  ⏺ you\x1b[0m \x1b[2m·\x1b[0m ' + typed);
    /* M1: /watch changes surface in chat context ONCE — drained into this
     * turn's ephemeral context note and cleared (hard-capped at the poll
     * site too, so an idle session can't accumulate). */
    const watchedNote = watchChangesPending.length
      ? watchChangesPending.map(c => '- ' + c.path + ' (' + c.kind + ')').join('\n')
      : '';
    watchChangesPending.length = 0;
    const def = {
      name: 'chat',
      /* P1: the byte-stable shared core prompt. lastShared mesh notes no
       * longer concatenate into the system string — they ride the ephemeral
       * context message via opts.shared so prefix caching works (P6). */
      system: sofuu.agent.CORE_PROMPT,
      tools: chatToolDefs(),
      agents: subAgentNames.length ? subAgentNames : undefined,
      provider: cfg.provider, model: cfg.model,
      /* Effort OFF still thinks UNDER THE HOOD: Anthropic omits its
       * thinking block entirely when no effort is sent, so map off →
       * 'low' (minimal budget) for the anthropic wire format. OpenAI-
       * compatible endpoints keep the omission (their default already
       * reasons, and forcing the field can error on non-reasoning
       * models). */
      effort: cfg.effort ||
        ((cfg.profile === 'anthropic' || cfg.provider === 'anthropic') ? 'low' : undefined),
      api_key: cfg.api_key || undefined,
      base_url: cfg.base_url || undefined,
      profile: cfg.profile || undefined,
      max_tokens: cfg.max_output > 0 ? cfg.max_output : undefined,
      embed_provider: cfg.embed_provider || undefined,
      embed_model: cfg.embed_model || undefined,
      memory: cfg.brain ? 'shared' : 'off',
      rlm: cfg.rlm === 'on' ? 'on' : (cfg.rlm === 'auto' ? 'auto' : 'off'),
      ctx_window: cfg.ctx_window > 0 ? cfg.ctx_window : undefined,
      /* P2: config-level recall gating knobs (0 = agent.js defaults). */
      recallMin: cfg.recall_min > 0 ? cfg.recall_min : undefined,
      recallBudget: cfg.recall_budget > 0 ? cfg.recall_budget : undefined,
      /* Chat historically has no budgets beyond the 8-step tool cap and
       * the 200k answer cap (both enforced inside agent.js). */
      budget: { maxSteps: 8, maxDepth: 1, maxTokens: 1e9, maxWallMs: 1e9 },
    };
    /* Streaming render state — same UX as the pre-migration driver:
     * spinner + growing line in the TUI, plain writes when piped. */
    let acc = '', anim = null, si2 = 0, sawThink = false, capped = false;
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
    const onStep = function (e) {
          const p = (e.payload === undefined || e.payload === null) ? {} : e.payload;
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
            if (cfg.effort) {
              sawThink = true;
              outLast('  ▸ thinking \x1b[2m' + thinks.join('') + '\x1b[0m');
            }
          } else if (e.kind === 'answer_delta') {
            if (sawThink && TTY) { out(''); sawThink = false; thinks.length = 0; }
            acc += String(p);
            if (acc.length > MAX_ANSWER_CHARS && !capped) {
              capped = true;
              out('  \x1b[33m⚠ answer capped at ' + MAX_ANSWER_CHARS + ' chars\x1b[0m');
            }
            if (TTY) { startAnim(); tick2(); }
            else { process.stdout.write(String(p)); }
          } else if (e.kind === 'plan') {
            /* A new planning round discards any preliminary streamed text. */
            acc = '';
          } else if (e.kind === 'tool') {
            stopAnim();
            const a = prettyToolArgs(p.args);
            out('\x1b[90m  ⏺\x1b[0m \x1b[36m' + p.name + '\x1b[0m\x1b[2m(' + a + ')\x1b[0m');
          } else if (e.kind === 'delegate') {
            stopAnim();
            out('\x1b[90m  ⏺\x1b[0m \x1b[35m' + p.agent + '\x1b[0m\x1b[2m ← ' + clip1(p.task, 60) + '\x1b[0m');
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
          } else if (e.kind === 'tool_result') {
            if (TTY && p.result) {
              /* Compact one-row result: embedded newlines become " · " so
               * multi-line tool output (search results, file dumps) no
               * longer splats unindented rows into the transcript. */
              const r = String(p.result).replace(/\s*\n+\s*/g, ' · ').replace(/\s+/g, ' ').trim();
              out('\x1b[2m  ↳ ' + clip1(r, 140) + '\x1b[0m');
            }
          } else if (e.kind === 'rlm:route') {
            out('\x1b[90m  ⏺ rlm · working…\x1b[0m');
          } else if (e.kind === 'rlm:done') {
            rlmSummary = p;
          }
    };
    try {
      res = agentMention
        ? await sofuu.agent.run(agentMention.name, turnText, { signal: 'chat', onStep })
        : await sofuu.agent.run(def, turnText, {
            history: history, signal: 'chat', onStep,
            /* P1/P6: volatile context rides the ephemeral context message —
             * never inside the byte-stable system prompt. */
            shared: lastShared || '',
            watched: watchedNote || '',
          });
      answer = String((res && res.answer) || '(no response)');
    } finally {
      stopAnim();
      stopPhase(); /* turn end (G): hide the indicator, never leave it stale */
    }
    if (rlmSummary) {
      /* RLM turn: the answer arrived complete — print it plus the
       * one-line trace summary (same format as the pre-migration path). */
      out(answer);
      out('\n\x1b[90m  ⏺ rlm · ' + (rlmSummary.calls || 0) + ' calls · ' + (rlmSummary.rounds || 0) + ' rounds · '
          + ((rlmSummary.ms || 0) / 1000).toFixed(1) + 's' + (rlmSummary.stopped ? ' · ' + rlmSummary.stopped : '') + '\x1b[0m\n');
    } else {
      if (TTY) {
        if (sawThink) out('');          /* seal a think-only answer */
        outLast(answer);                /* final line, no spinner */
      } else {
        out('\n');
      }
    }
    if (res && res.stopped === 'cancelled') out('\x1b[90m  ⏹ stopped (esc)\x1b[0m');
    if (agentMention && res) {
      /* @agent mention: compact run summary under the answer. */
      const u = res.usage || {};
      out('\x1b[90m  ⏺ via ' + agentMention.name + ' · ' + (u.llmCalls || 0) + ' llm calls · '
          + (u.toolCalls || 0) + ' tool calls' + (res.stopped ? ' · ' + res.stopped : '') + '\x1b[0m');
    }
    const ctxTk = (res && res.usage && res.usage.promptTokens) || 0;
    const outTk = (res && res.usage && res.usage.completionTokens) || 0;
    totalTk += ctxTk + outTk;
    let tk = (ctxTk > 0 || outTk > 0) ? (ctxTk + '→' + outTk + ' tk') : '';
    /* F6: report usage → compute cost + persist. Cache token slots (P6.4)
     * ride along so /cost can show prefix-cache hits when non-zero. */
    if (ctxTk > 0 || outTk > 0) {
      const cacheR = (res && res.usage && res.usage.cacheReadTokens) || 0;
      const cacheW = (res && res.usage && res.usage.cacheWriteTokens) || 0;
      try { __chat_report_usage(cfg.model || '', ctxTk, outTk, cacheR, cacheW); } catch (e) {}
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
     * full transcript. */
    try { __chat_log('answer', String(answer)); } catch (e) {}
    /* F3: history stores the manifest only (not full file text). */
    const histText = manifest ? text + ' [' + manifest + ']' : text;
    history.push({ role: 'user', content: histText }, { role: 'assistant', content: answer });
    await trimHistory();
    /* M2: one full GC per completed turn — bounds JS garbage to a single
     * turn's worth instead of accumulating against the heap cap. */
    try { __chat_gc(); } catch (e) {}
    /* F9: post-hook — may modify the answer. */
    const finalAnswer = await runHookPost(text, answer, res && res.usage);
    if (finalAnswer !== answer) {
      /* Update the last history entry if the post-hook changed the answer. */
      if (history.length >= 1) history[history.length - 1].content = finalAnswer;
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
  async function wizardFetchModels(prov, base, key) {
    try {
      if (prov === 'local') {
        const r = await fetchWithTimeout((base || 'http://127.0.0.1:11434') + '/api/tags', {}, 4000);
        const j = await r.json();
        return (j.models || []).map(m => m.name);
      }
      if (prov === 'anthropic') return []; // no public model list
      /* openai-compatible endpoint. The stored base_url is the FULL
       * completion URL (e.g. https://host/v1/chat/completions) — the
       * models list lives at the API ROOT + /models, so strip any
       * trailing /chat/completions (or /completions) first. */
      let root = (base || WIZARD_BUILTIN_BASE.openai).replace(/\/+$/, '');
      root = root.replace(/\/chat\/completions$/, '').replace(/\/completions$/, '');
      const h = key ? { Authorization: 'Bearer ' + key } : {};
      const r = await fetchWithTimeout(root + '/models', { headers: h }, 6000);
      const j = await r.json();
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
    try { __chat_refresh(); } catch (e) {}
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
      out('\x1b[90m  [Compacted → context (' + fmtTk(Math.max(0, before - historyTokens())) + ' tk saved)]\x1b[0m\n');
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
    if (typeof __chat_status !== 'function') return;
    const dim = (s) => '\x1b[2m' + s + '\x1b[0m';
    let s = 'sofuu ' + dim('·') + ' \x1b[35m' + (cfg.model || '(no model)') + '\x1b[0m';
    if (cfg.provider) {
      const prov = cfg.provider === 'custom'
        ? ((cfg.profile || 'openai') + '-compat')
        : cfg.provider;
      s += ' ' + dim('·') + ' ' + dim(prov);
    }
    if (cfg.effort) s += ' ' + dim('·') + ' ' + dim('effort ' + cfg.effort);
    if (tk) s += ' ' + dim('·') + ' ' + dim(tk);
    /* Footer row 1 right: dim hints. Row 2: left = context used,
     * right = real-time RAM of the sofuu process. */
    const win = cfg.ctx_window > 0
      ? cfg.ctx_window
      : ({ openai: 128000, anthropic: 128000, local: 32768 }[cfg.provider] || 32768);
    const used = totalTk;
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
     * brain is off. */
    if (cfg.brain) { ctxMet = ctxMet + dim(' · brain'); }
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
    try { __chat_status(s, '/help · enter ⏎ · esc stop', ramMet, ctxMet); } catch (e) {}
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
    try { __chat_refresh(); } catch (e) {}
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
      } else { out('\x1b[90m  (unchanged)\x1b[0m'); try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {} refreshStatus(''); try { __chat_refresh(); } catch (e) {} return; }
    }
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    try { __chat_refresh(); } catch (e) {}
  }
  async function pickEffort() {
    /* Budgets mirror the C mapping (Anthropic thinking.budget_tokens). */
    const notes = { off: 'provider default', low: '1k thinking budget',
                    medium: '4k thinking budget', high: '16k thinking budget',
                    max: '32k thinking budget' };
    const order = ['off', 'low', 'medium', 'high', 'max'];
    const it = await selectMenu({
      title: 'Thinking effort',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: order.map(l => ({ id: l, label: l, tab: '', note: notes[l] })),
      currentId: cfg.effort || 'off',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/effort ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    try { __chat_refresh(); } catch (e) {}
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
    try { __chat_refresh(); } catch (e) {}
  }
  async function pickCtx() {
    const eff = (cfg.ctx_window > 0 ? cfg.ctx_window : ({ openai: 128000, anthropic: 128000, local: 32768 }[cfg.provider] || 32768));
    const presets = [
      { id: '0', label: 'default', note: 'provider default (' + fmtTk(eff) + ')' },
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
    try { __chat_refresh(); } catch (e) {}
  }
  async function pickMaxout() {
    const presets = [
      { id: '0', label: 'default', note: 'provider default' },
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
    try { __chat_refresh(); } catch (e) {}
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
    try { __chat_refresh(); } catch (e) {}
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
    try { __chat_refresh(); } catch (e) {}
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
     * the numbers track the live process without waiting for input. */
    const metricsTimer = setInterval(function() {
      try { refreshStatus(''); } catch (e) {}
    }, 2000);
    const stopMetrics = () => { try { clearInterval(metricsTimer); } catch (e) {} };
    globalThis.__metrics_stop = stopMetrics;
    if (TTY) {
      /* Full-screen interface: take over the terminal. */
      try { __tui_on(); } catch (e) {}
      /* Bordered welcome panel (logo + session facts) — built in Rust. */
      try { __chat_welcome(); } catch (e) {}
      /* Shima-enaga logo animation: cycle the bird frames in place (rows
       * 3-5 overlay). Stops on the first user input. */
      let logoAnim = setInterval(function() {
        try { __chat_logo(); } catch (e) {}
      }, 500);
      const stopLogo = () => { try { clearInterval(logoAnim); } catch (e) {} };
      /* Re-open the previous transcript (persisted in the session .qtsq). */
      try {
        const past = JSON.parse(__chat_past_turns());
        for (const t of past) {
          out('\x1b[1;35m  ⏺ you\x1b[0m · ' + t.prompt);
          if (t.answer) out(t.answer);
        }
        if (past.length) out('');
      } catch (e) {}
      /* Stop the animation on the first input (readline arming). */
      const stopLogoOnce = function() { stopLogo(); };
      globalThis.__logo_stop = stopLogoOnce;
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
      try { __chat_refresh(); } catch (e) {}
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
      const inp = await __readline('\x1b[1;35m❯\x1b[0m ');
      /* First user input: freeze the logo animation. */
      try { if (globalThis.__logo_stop) { globalThis.__logo_stop(); globalThis.__logo_stop = null; } } catch (e) {}
      if (inp === null || inp === undefined) { requestExit(); break; }
      const t = inp.trim();
      if (t === '') continue;
      if (t[0] === '/') {
        const r = __chat_slash(t);
        // Slash commands may have changed provider/model/api_key — re-read
        // the Rust-side config so the live session uses the new values.
        try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
        /* Any command that can change settings re-renders the welcome panel
         * so the header (model/provider/effort) updates instantly. */
        const cfgCmds = ['/model', '/provider', '/effort', '/ctx', '/maxout', '/brain', '/rlm', '/sync'];
        const changed = cfgCmds.some(c => t === c || t.indexOf(c + ' ') === 0);
        if (changed) { try { __chat_refresh(); } catch (e) {} }
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
          else out('\x1b[90m  Usage: /ctx <tokens|default>  (current: ' + (cfg.ctx_window || 'provider default') + ')\x1b[0m\n');
        } else if (r === 'pick_maxout') {
          if (TTY) await pickMaxout();
          else out('\x1b[90m  Usage: /maxout <tokens|default>  (current: ' + (cfg.max_output || 'provider default') + ')\x1b[0m\n');
        } else if (r === 'pick_brain') {
          if (TTY) await pickBrain();
          else out('\x1b[90m  Usage: /brain <on|off>  (current: ' + (cfg.brain ? 'on' : 'off') + ')\x1b[0m\n');
        } else if (r === 'pick_sync') {
          if (TTY) await pickSync();
          else out('\x1b[90m  Usage: /sync <on|off>  (current: ' + (cfg.sync ? 'on' : 'off') + ')\x1b[0m\n');
        } else if (r === 'clear') { history = []; out('\x1b[90m  ✓ session cleared\x1b[0m\n'); }
        else if (r === 'compact') { await compact(); }
        else if (r === 'tools') { await connectMcpServers(); showTools(); }
        else if (r === 'agents') { await handleAgentsCmd(); }
        // F1–F11 feature dispatch
        else if (r === 'remember') { await handleRemember(t.replace(/^\/remember\s*/, '').trim()); }
        else if (r === 'why') { handleWhy(); }
        else if (r === 'resume') { await handleResume(t.replace(/^\/resume\s*/, '').trim()); }
        else if (r === 'share') { await handleShare(t.replace(/^\/share\s*/, '').trim()); }
        else if (r === 'import') { await handleImport(t.replace(/^\/import\s*/, '').trim()); }
        else if (r === 'cost') { handleCost(); }
        else if (r === 'watch') {
          var watchArg = t.replace(/^\/watch\s*/, '').trim();
          handleWatch(watchArg);
        }
        // 'unknown' prints its message (with did-you-mean) in Rust
        // handle_slash — nothing more to show here.
        continue;
      }
      /* A failed turn must never kill the chat: print the error and loop
       * back to the prompt. (Without this, an API/config exception rejects
       * the driver's main() promise and the CLI exits immediately.) */
      try {
        await turn(t);
      } catch (e) {
        out('\x1b[31m  ✗ ' + String((e && e.message) || e) + '\x1b[0m\n');
      }
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
            "{\n  \"provider\": \"openai\",\n  \"model\": \"gpt-4o\",\n  \"effort\": \"high\",\n  \"brain\": true,\n  \"ctx_window\": 1000000,\n  \"max_output\": 384000\n}\n",
        )
        .unwrap();

        let l = ChatConfig::load();
        assert_eq!(l.provider, "openai");
        assert_eq!(l.model, "gpt-4o");
        assert_eq!(l.effort, "high");
        assert!(l.brain);
        assert_eq!(l.ctx_window, 1_000_000);
        assert_eq!(l.max_output, 384_000);

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
        // /rlm: bare opens panel; on/auto/off persist (off stores "").
        assert_eq!(handle_slash(&mut c, "/rlm"), "pick_rlm");
        assert_eq!(handle_slash(&mut c, "/rlm auto"), "ok");
        assert_eq!(c.rlm, "auto");
        assert_eq!(handle_slash(&mut c, "/rlm bogus"), "ok");
        assert_eq!(c.rlm, "auto", "bad /rlm arg must not change state");
        assert_eq!(handle_slash(&mut c, "/rlm off"), "ok");
        assert!(c.rlm.is_empty());
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
        // load() clamps corrupt/over-the-cap persisted values.
        assert_eq!(clamp_ctx_window(999_999_999), 1_000_000);
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

    #[test]
    fn welcome_panel_layout() {
        // Explicit fixture values (layout test data — no provider is the
        // default anywhere; these are just strings).
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
}
