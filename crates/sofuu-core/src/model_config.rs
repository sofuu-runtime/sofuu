//! Per-agent provider and model configuration (`models.json`) and generalized
//! CLI commands (`--setup-model`, `--edit-model`, `--list-models`).
//!
//! Designed so every Sofuu agent (whether run via `sofuu run <script>` or
//! compiled to a standalone binary `./my-agent`) supports two clean,
//! domain-agnostic ways to configure any provider and model:
//!   1. A dedicated per-agent JSON file (`models.json` or `<bin>.models.json`)
//!      containing ONLY providers and models.
//!   2. Built-in CLI commands on the binary/script (`--setup-model`,
//!      `--edit-model`, `--list-models`).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ProviderProfile {
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AgentModelsFile {
    #[serde(default)]
    pub active: String,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderProfile>,
}

#[derive(Clone, Debug)]
pub struct AgentTarget {
    pub path: PathBuf,
    pub is_binary: bool,
}

impl AgentTarget {
    pub fn binary(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            is_binary: true,
        }
    }

    pub fn script(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            is_binary: false,
        }
    }

    fn stem(&self) -> String {
        self.path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("agent")
            .to_string()
    }

    fn parent_dir(&self) -> PathBuf {
        self.path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

#[derive(Clone, Debug, Default)]
pub struct CliOverrides {
    pub models_config: Option<PathBuf>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedAgentModel {
    pub config_path: Option<PathBuf>,
    pub active_profile: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub profiles: BTreeMap<String, ProviderProfile>,
}

impl ResolvedAgentModel {
    /// Serializes a non-secret summary object for `globalThis.__sofuu_agent_models`.
    pub fn to_js_json(&self) -> String {
        let mut profiles_json = serde_json::Map::new();
        for (name, p) in &self.profiles {
            profiles_json.insert(
                name.clone(),
                serde_json::json!({
                    "provider": p.provider,
                    "model": p.model,
                    "baseUrl": p.base_url,
                    "hasApiKey": p.api_key.as_deref().is_some_and(|k| !k.trim().is_empty()),
                    "apiKeySource": mask_key(p.api_key.as_deref()),
                }),
            );
        }
        serde_json::json!({
            "configPath": self.config_path.as_ref().map(|p| p.display().to_string()),
            "active": self.active_profile,
            "provider": self.provider,
            "model": self.model,
            "baseUrl": self.base_url,
            "hasApiKey": self.api_key.as_deref().is_some_and(|k| !k.trim().is_empty()),
            "providers": profiles_json,
        })
        .to_string()
    }
}

/// Resolves an `"env:VAR_NAME"` reference or returns a literal non-empty key.
pub fn resolve_api_key_value(raw: Option<&str>) -> Option<String> {
    let s = raw?.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(var_name) = s.strip_prefix("env:") {
        let name = var_name.trim();
        if name.is_empty() {
            return None;
        }
        return env::var(name).ok().filter(|v| !v.trim().is_empty());
    }
    Some(s.to_string())
}

/// Masks raw API keys for terminal display while preserving `env:VAR` labels.
pub fn mask_key(raw: Option<&str>) -> String {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return "(none)".to_string();
    };
    if s.starts_with("env:") {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= 8 {
        return "****".to_string();
    }
    let head: String = chars[..4].iter().collect();
    let tail: String = chars[chars.len() - 4..].iter().collect();
    format!("{head}...{tail}")
}

fn dir_is_writable(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let probe = dir.join(format!(".sofuu-write-probe-{}", std::process::id()));
    match fs::OpenOptions::new().write(true).create_new(true).open(&probe) {
        Ok(_) => {
            let _ = fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Candidate paths searched when reading an agent's dedicated model config.
pub fn candidate_paths(target: &AgentTarget, explicit: Option<&Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = explicit {
        out.push(p.to_path_buf());
        return out;
    }
    if let Ok(env_p) = env::var("SOFUU_MODELS_CONFIG") {
        let trimmed = env_p.trim();
        if !trimmed.is_empty() {
            out.push(PathBuf::from(trimmed));
            return out;
        }
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let target_dir = target.parent_dir();
    let stem = target.stem();

    if target.is_binary {
        out.push(cwd.join(format!("{stem}.models.json")));
        out.push(target_dir.join(format!("{stem}.models.json")));
        out.push(target_dir.join("models.json"));
        out.push(cwd.join("models.json"));
    } else {
        if stem != "index" {
            out.push(target_dir.join(format!("{stem}.models.json")));
        }
        out.push(target_dir.join("models.json"));
        out.push(cwd.join("models.json"));
    }
    out
}

pub fn find_config_path(target: &AgentTarget, explicit: Option<&Path>) -> Option<PathBuf> {
    candidate_paths(target, explicit)
        .into_iter()
        .find(|p| p.is_file())
}

/// Chooses the target file path for `--setup-model` / `--edit-model`.
pub fn resolve_write_path(target: &AgentTarget, explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Ok(env_p) = env::var("SOFUU_MODELS_CONFIG") {
        let trimmed = env_p.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Some(existing) = find_config_path(target, None) {
        return existing;
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let target_dir = target.parent_dir();
    let stem = target.stem();

    if target.is_binary {
        let filename = format!("{stem}.models.json");
        if dir_is_writable(&target_dir) {
            target_dir.join(filename)
        } else {
            cwd.join(filename)
        }
    } else if dir_is_writable(&target_dir) {
        target_dir.join("models.json")
    } else {
        cwd.join("models.json")
    }
}

pub fn load_models_file(path: &Path) -> Result<AgentModelsFile, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str::<AgentModelsFile>(&raw)
        .map_err(|e| format!("invalid JSON in {}: {e}", path.display()))
}

pub fn save_models_file(path: &Path, cfg: &AgentModelsFile) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(cfg)
        .map_err(|e| format!("failed to serialize models config: {e}"))?;
    fs::write(path, format!("{json}\n"))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Parses model-related CLI flags from argv (stopping at `--`).
pub fn extract_cli_overrides(args: &[String]) -> CliOverrides {
    let mut ov = CliOverrides::default();
    let mut i = 1usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--" {
            break;
        }
        let next = args.get(i + 1).cloned();
        if (a == "--models-config") && next.is_some() {
            ov.models_config = next.map(PathBuf::from);
            i += 2;
            continue;
        }
        if let Some(rest) = a.strip_prefix("--models-config=") {
            if !rest.is_empty() {
                ov.models_config = Some(PathBuf::from(rest));
            }
            i += 1;
            continue;
        }
        if (a == "--provider" || a == "-p") && next.is_some() {
            ov.provider = next;
            i += 2;
            continue;
        }
        if let Some(rest) = a.strip_prefix("--provider=") {
            if !rest.is_empty() {
                ov.provider = Some(rest.to_string());
            }
            i += 1;
            continue;
        }
        if (a == "--model" || a == "-m") && next.is_some() {
            ov.model = next;
            i += 2;
            continue;
        }
        if let Some(rest) = a.strip_prefix("--model=") {
            if !rest.is_empty() {
                ov.model = Some(rest.to_string());
            }
            i += 1;
            continue;
        }
        if a == "--api-key" && next.is_some() {
            ov.api_key = next;
            i += 2;
            continue;
        }
        if let Some(rest) = a.strip_prefix("--api-key=") {
            if !rest.is_empty() {
                ov.api_key = Some(rest.to_string());
            }
            i += 1;
            continue;
        }
        if a == "--base-url" && next.is_some() {
            ov.base_url = next;
            i += 2;
            continue;
        }
        if let Some(rest) = a.strip_prefix("--base-url=") {
            if !rest.is_empty() {
                ov.base_url = Some(rest.to_string());
            }
            i += 1;
            continue;
        }
        i += 1;
    }
    ov
}

/// Resolves effective model config and installs defaults into `embed_config`.
pub fn apply_runtime_config(args: &[String], target: &AgentTarget) -> ResolvedAgentModel {
    let ov = extract_cli_overrides(args);
    let config_path = find_config_path(target, ov.models_config.as_deref());
    let file = config_path
        .as_deref()
        .and_then(|p| load_models_file(p).ok())
        .unwrap_or_default();

    // If `--provider <name>` matches a profile name in models.json, load that profile.
    let selected_profile_name = ov
        .provider
        .as_deref()
        .filter(|p| file.providers.contains_key(*p))
        .map(|s| s.to_string())
        .or_else(|| {
            if !file.active.is_empty() && file.providers.contains_key(&file.active) {
                Some(file.active.clone())
            } else {
                file.providers.keys().next().cloned()
            }
        });

    let profile = selected_profile_name
        .as_deref()
        .and_then(|name| file.providers.get(name));

    // CLI `--provider` overrides only if it wasn't simply naming a profile in models.json,
    // or if no profile matched.
    let provider = match (&ov.provider, profile) {
        (Some(p), Some(prof)) if file.providers.contains_key(p) => Some(prof.provider.clone()),
        (Some(p), _) => Some(p.clone()),
        (None, Some(prof)) => Some(prof.provider.clone()),
        (None, None) => None,
    };
    let model = ov
        .model
        .clone()
        .or_else(|| profile.map(|p| p.model.clone()))
        .filter(|s| !s.trim().is_empty());
    let base_url = ov
        .base_url
        .clone()
        .or_else(|| profile.and_then(|p| p.base_url.clone()))
        .filter(|s| !s.trim().is_empty());
    let api_key = ov
        .api_key
        .clone()
        .or_else(|| resolve_api_key_value(profile.and_then(|p| p.api_key.as_deref())))
        .filter(|s| !s.trim().is_empty());

    // Push resolved defaults into `embed_config` so `sofuu.ai.*` and
    // `sofuu.agent.*` honor the agent's model config automatically.
    if provider.is_some() || model.is_some() || base_url.is_some() {
        crate::embed_config::set_defaults(provider.clone(), model.clone(), base_url.clone());
    }
    if let Some(ref key) = api_key {
        let key_slot = match provider.as_deref() {
            Some("anthropic") => "anthropic",
            _ => "openai",
        };
        crate::embed_config::set_api_key(key_slot, key.clone());
    }

    ResolvedAgentModel {
        config_path,
        active_profile: selected_profile_name,
        provider,
        model,
        base_url,
        api_key,
        profiles: file.providers,
    }
}

/// Injects `globalThis.__sofuu_agent_models` into the JS context.
pub fn js_prelude_for(resolved: &ResolvedAgentModel) -> String {
    let json = resolved.to_js_json();
    format!("try{{globalThis.__sofuu_agent_models={json};}}catch(e){{}}")
}

// ── Interactive CLI Commands (--setup-model / --edit-model / --list-models) ──

fn prompt_line(prompt: &str) -> io::Result<String> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut line = String::new();
    let stdin = io::stdin();
    let n = stdin.lock().read_line(&mut line)?;
    if n == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
    }
    Ok(line.trim().to_string())
}

fn prompt_with_default(label: &str, default_val: &str) -> io::Result<String> {
    let answer = if default_val.is_empty() {
        prompt_line(&format!("{label}: "))?
    } else {
        prompt_line(&format!("{label} [{default_val}]: "))?
    };
    if answer.is_empty() {
        Ok(default_val.to_string())
    } else {
        Ok(answer)
    }
}

fn print_profiles_table(path: &Path, cfg: &AgentModelsFile) {
    println!("\nAgent Model Config: {}", path.display());
    if cfg.providers.is_empty() {
        println!("  (no providers configured yet)");
        println!("  Run with --setup-model to add one.\n");
        return;
    }
    println!();
    for (idx, (name, p)) in cfg.providers.iter().enumerate() {
        let marker = if name == &cfg.active { "*" } else { " " };
        let url_part = p
            .base_url
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(|u| format!(" · {u}"))
            .unwrap_or_default();
        let key_part = mask_key(p.api_key.as_deref());
        println!(
            "  {marker} [{}] {name:<14} provider={:<10} model={}{url_part}  key={key_part}",
            idx + 1,
            p.provider,
            p.model
        );
    }
    println!();
}

fn prompt_new_profile(default_name: Option<&str>) -> io::Result<(String, ProviderProfile)> {
    println!("\nChoose provider type:");
    println!("  [1] OpenAI (api.openai.com)");
    println!("  [2] Anthropic (api.anthropic.com)");
    println!("  [3] Custom / OpenAI-compatible URL (OpenRouter, DeepSeek, Groq, vLLM, ...)");
    println!("  [4] Local Ollama (http://localhost:11434)");
    println!("  [5] Local llama.cpp server (http://localhost:8080/v1)");
    let choice = prompt_with_default("Choice [1-5]", "1")?;

    let (provider, default_profile, default_model_hint, needs_url, needs_key, default_key_hint) =
        match choice.as_str() {
            "2" | "anthropic" => (
                "anthropic",
                "anthropic",
                "claude-3-5-sonnet-20241022",
                false,
                true,
                "env:ANTHROPIC_API_KEY",
            ),
            "3" | "custom" => (
                "custom",
                "custom",
                "",
                true,
                true,
                "env:OPENAI_API_KEY",
            ),
            "4" | "ollama" | "local" => (
                "ollama",
                "local",
                "",
                true,
                false,
                "",
            ),
            "5" | "llamacpp" => (
                "llamacpp",
                "llamacpp",
                "default",
                true,
                false,
                "",
            ),
            _ => (
                "openai",
                "openai",
                "gpt-4o",
                false,
                true,
                "env:OPENAI_API_KEY",
            ),
        };

    let name_def = default_name.unwrap_or(default_profile);
    let profile_name = prompt_with_default("Profile name", name_def)?;
    let model = loop {
        let m = prompt_with_default("Model name", default_model_hint)?;
        if !m.trim().is_empty() {
            break m;
        }
        println!("  Model name is required.");
    };

    let base_url = if needs_url {
        let def_u = match provider {
            "ollama" => "http://localhost:11434",
            "llamacpp" => "http://localhost:8080/v1",
            _ => "",
        };
        let u = prompt_with_default("Base URL", def_u)?;
        if u.trim().is_empty() { None } else { Some(u) }
    } else {
        None
    };

    let api_key = if needs_key {
        let k = prompt_with_default(
            "API key (paste key, 'env:VAR_NAME', or '-' for none)",
            default_key_hint,
        )?;
        if k.trim().is_empty() || k == "-" {
            None
        } else {
            Some(k)
        }
    } else {
        None
    };

    Ok((
        profile_name,
        ProviderProfile {
            provider: provider.to_string(),
            model,
            base_url,
            api_key,
        },
    ))
}

fn edit_single_profile(name: &str, prof: &mut ProviderProfile) -> io::Result<()> {
    println!("\nEditing profile '{name}' (press Enter to keep current value, '-' to clear):");
    prof.provider = prompt_with_default("  Provider (openai/anthropic/custom/ollama/llamacpp)", &prof.provider)?;
    prof.model = prompt_with_default("  Model", &prof.model)?;

    let cur_url = prof.base_url.as_deref().unwrap_or("");
    let new_url = prompt_with_default("  Base URL", cur_url)?;
    prof.base_url = if new_url.trim().is_empty() || new_url == "-" {
        None
    } else {
        Some(new_url)
    };

    let cur_key = prof.api_key.as_deref().unwrap_or("");
    let key_prompt = if cur_key.is_empty() {
        String::new()
    } else {
        mask_key(Some(cur_key))
    };
    let ans = if key_prompt.is_empty() {
        prompt_line("  API key (key or env:VAR_NAME): ")?
    } else {
        prompt_line(&format!("  API key [{key_prompt}]: "))?
    };
    if ans == "-" {
        prof.api_key = None;
    } else if !ans.is_empty() {
        prof.api_key = Some(ans);
    }
    Ok(())
}

fn resolve_profile_pick(cfg: &AgentModelsFile, raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if cfg.providers.contains_key(trimmed) {
        return Some(trimmed.to_string());
    }
    if let Ok(idx) = trimmed.parse::<usize>() {
        if idx >= 1 {
            return cfg.providers.keys().nth(idx - 1).cloned();
        }
    }
    None
}

fn run_setup_model(target: &AgentTarget, args: &[String]) -> ExitCode {
    let ov = extract_cli_overrides(args);
    let path = resolve_write_path(target, ov.models_config.as_deref());
    let mut cfg = if path.is_file() {
        load_models_file(&path).unwrap_or_default()
    } else {
        AgentModelsFile::default()
    };

    // Non-interactive path when both --provider and --model are passed with --setup-model.
    if let (Some(prov), Some(model)) = (ov.provider.clone(), ov.model.clone()) {
        let profile_name = args
            .windows(2)
            .find(|w| w[0] == "--profile")
            .map(|w| w[1].clone())
            .unwrap_or_else(|| prov.clone());
        cfg.providers.insert(
            profile_name.clone(),
            ProviderProfile {
                provider: prov,
                model,
                base_url: ov.base_url,
                api_key: ov.api_key,
            },
        );
        cfg.active = profile_name.clone();
        return match save_models_file(&path, &cfg) {
            Ok(()) => {
                println!("✓ Saved profile '{profile_name}' (active) to {}", path.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("Error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    println!("\n┌──────────────────────────────────────────────────────────┐");
    println!("│ Agent Provider & Model Setup                             │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!("Config file: {}", path.display());

    let (name, profile) = match prompt_new_profile(None) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Setup aborted: {e}");
            return ExitCode::from(2);
        }
    };

    let make_active = prompt_with_default(&format!("Set '{name}' as active? (Y/n)"), "y")
        .map(|s| !s.eq_ignore_ascii_case("n") && !s.eq_ignore_ascii_case("no"))
        .unwrap_or(true);

    cfg.providers.insert(name.clone(), profile);
    if make_active || cfg.active.is_empty() {
        cfg.active = name.clone();
    }

    match save_models_file(&path, &cfg) {
        Ok(()) => {
            println!(
                "\n✓ Saved profile '{name}'{} to {}",
                if cfg.active == name { " (active)" } else { "" },
                path.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_edit_model(target: &AgentTarget, args: &[String], direct_profile: Option<&str>) -> ExitCode {
    let ov = extract_cli_overrides(args);
    let path = resolve_write_path(target, ov.models_config.as_deref());
    let mut cfg = if path.is_file() {
        match load_models_file(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        AgentModelsFile::default()
    };

    if let Some(target_name) = direct_profile {
        if let Some(key) = resolve_profile_pick(&cfg, target_name) {
            if let Some(prof) = cfg.providers.get_mut(&key) {
                if let Err(e) = edit_single_profile(&key, prof) {
                    eprintln!("Edit aborted: {e}");
                    return ExitCode::from(2);
                }
                return match save_models_file(&path, &cfg) {
                    Ok(()) => {
                        println!("\n✓ Updated '{key}' in {}", path.display());
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("Error: {e}");
                        ExitCode::FAILURE
                    }
                };
            }
        } else {
            eprintln!("Profile '{target_name}' not found in {}", path.display());
            print_profiles_table(&path, &cfg);
            return ExitCode::from(2);
        }
    }

    loop {
        print_profiles_table(&path, &cfg);
        println!("Actions:");
        println!("  [s] Switch active provider/model");
        println!("  [e] Edit a provider/model");
        println!("  [a] Add a new provider/model");
        println!("  [d] Delete a provider/model");
        println!("  [q] Save & quit");
        let Ok(action) = prompt_line("Action [s/e/a/d/q]: ") else {
            break;
        };
        match action.to_lowercase().as_str() {
            "s" => {
                let Ok(pick) = prompt_line("Enter profile name or number to activate: ") else {
                    break;
                };
                if let Some(key) = resolve_profile_pick(&cfg, &pick) {
                    cfg.active = key.clone();
                    if let Err(e) = save_models_file(&path, &cfg) {
                        eprintln!("Error: {e}");
                        return ExitCode::FAILURE;
                    }
                    println!("✓ Active profile set to '{key}'.");
                } else {
                    println!("  Not found: '{pick}'");
                }
            }
            "e" => {
                let Ok(pick) = prompt_line("Enter profile name or number to edit: ") else {
                    break;
                };
                if let Some(key) = resolve_profile_pick(&cfg, &pick) {
                    if let Some(prof) = cfg.providers.get_mut(&key) {
                        if let Err(e) = edit_single_profile(&key, prof) {
                            eprintln!("Edit aborted: {e}");
                            return ExitCode::from(2);
                        }
                        if let Err(e) = save_models_file(&path, &cfg) {
                            eprintln!("Error: {e}");
                            return ExitCode::FAILURE;
                        }
                        println!("✓ Updated '{key}'.");
                    }
                } else {
                    println!("  Not found: '{pick}'");
                }
            }
            "a" => {
                let Ok((name, prof)) = prompt_new_profile(None) else {
                    break;
                };
                cfg.providers.insert(name.clone(), prof);
                if cfg.active.is_empty() {
                    cfg.active = name.clone();
                }
                if let Err(e) = save_models_file(&path, &cfg) {
                    eprintln!("Error: {e}");
                    return ExitCode::FAILURE;
                }
                println!("✓ Added '{name}'.");
            }
            "d" => {
                let Ok(pick) = prompt_line("Enter profile name or number to delete: ") else {
                    break;
                };
                if let Some(key) = resolve_profile_pick(&cfg, &pick) {
                    cfg.providers.remove(&key);
                    if cfg.active == key {
                        cfg.active = cfg.providers.keys().next().cloned().unwrap_or_default();
                    }
                    if let Err(e) = save_models_file(&path, &cfg) {
                        eprintln!("Error: {e}");
                        return ExitCode::FAILURE;
                    }
                    println!("✓ Deleted '{key}'.");
                } else {
                    println!("  Not found: '{pick}'");
                }
            }
            "q" | "" => {
                if let Err(e) = save_models_file(&path, &cfg) {
                    eprintln!("Error: {e}");
                    return ExitCode::FAILURE;
                }
                println!("Saved: {}", path.display());
                return ExitCode::SUCCESS;
            }
            _ => println!("  Unknown action '{action}'."),
        }
    }
    ExitCode::SUCCESS
}

fn run_list_models(target: &AgentTarget, args: &[String]) -> ExitCode {
    let ov = extract_cli_overrides(args);
    let path = find_config_path(target, ov.models_config.as_deref())
        .unwrap_or_else(|| resolve_write_path(target, ov.models_config.as_deref()));
    let cfg = if path.is_file() {
        match load_models_file(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        AgentModelsFile::default()
    };
    print_profiles_table(&path, &cfg);
    ExitCode::SUCCESS
}

/// Intercepts `--setup-model`, `--edit-model`, and `--list-models` before `--`.
pub fn handle_cli_command(args: &[String], target: &AgentTarget) -> Option<ExitCode> {
    let mut i = 1usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--" {
            return None;
        }
        if a == "--setup-model" {
            return Some(run_setup_model(target, args));
        }
        if a == "--list-models" {
            return Some(run_list_models(target, args));
        }
        if a == "--edit-model" {
            let direct = args
                .get(i + 1)
                .map(|s| s.as_str())
                .filter(|s| !s.starts_with('-'));
            return Some(run_edit_model(target, args, direct));
        }
        if let Some(rest) = a.strip_prefix("--edit-model=") {
            let direct = if rest.is_empty() { None } else { Some(rest) };
            return Some(run_edit_model(target, args, direct));
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_env_key_and_masks_literals() {
        env::set_var("SOFUU_TEST_MODEL_KEY", "secret-val-99");
        assert_eq!(
            resolve_api_key_value(Some("env:SOFUU_TEST_MODEL_KEY")).as_deref(),
            Some("secret-val-99")
        );
        env::remove_var("SOFUU_TEST_MODEL_KEY");
        assert_eq!(
            resolve_api_key_value(Some("literal-key")).as_deref(),
            Some("literal-key")
        );
        assert_eq!(mask_key(Some("env:OPENAI_API_KEY")), "env:OPENAI_API_KEY");
        assert_eq!(mask_key(Some("sk-1234567890abcd")), "sk-1...abcd");
        assert_eq!(mask_key(None), "(none)");
    }

    #[test]
    fn setup_model_noninteractive_creates_per_binary_file() {
        let tmp = env::temp_dir().join(format!("sofuu-mc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let bin = tmp.join("my-agent");
        let target = AgentTarget::binary(&bin);
        let args: Vec<String> = vec![
            bin.display().to_string(),
            "--setup-model".into(),
            "--provider".into(),
            "anthropic".into(),
            "--model".into(),
            "claude-3-5-sonnet-20241022".into(),
            "--api-key".into(),
            "env:ANTHROPIC_API_KEY".into(),
        ];

        let code = handle_cli_command(&args, &target);
        assert_eq!(code, Some(ExitCode::SUCCESS));

        let expected = tmp.join("my-agent.models.json");
        assert!(expected.is_file(), "should write per-binary .models.json");
        let loaded = load_models_file(&expected).unwrap();
        assert_eq!(loaded.active, "anthropic");
        assert_eq!(loaded.providers["anthropic"].model, "claude-3-5-sonnet-20241022");

        let resolved = apply_runtime_config(&[bin.display().to_string()], &target);
        assert_eq!(resolved.provider.as_deref(), Some("anthropic"));
        assert_eq!(resolved.model.as_deref(), Some("claude-3-5-sonnet-20241022"));

        let _ = fs::remove_dir_all(&tmp);
    }
}
