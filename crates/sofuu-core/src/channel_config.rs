//! Per-agent messaging channel configuration (`channels.json`) and the
//! generalized CLI commands `--setup-channel`, `--edit-channel` and
//! `--list-channels`.
//!
//! This mirrors [`crate::model_config`] deliberately: an agent configures its
//! providers and its messaging channels with one mental model.
//!   1. A dedicated per-agent JSON file (`channels.json` or
//!      `<bin>.channels.json`) containing ONLY channel profiles.
//!   2. Built-in commands on the binary/script (`--setup-channel`,
//!      `--edit-channel`, `--list-channels`).
//!
//! Secrets support `"env:VAR_NAME"` resolution and are written with `0600`
//! permissions on Unix. Nothing in this module knows about a business domain:
//! the four transports (telegram, whatsapp, webhook, desktop) are platform
//! adapters, and every message body is supplied by the caller.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ChannelProfile {
    /// `telegram`, `whatsapp`, `webhook` or `desktop`.
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phone_number_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_secret: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_recipient: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_senders: Vec<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub allow_anonymous: bool,
    /// `outbound_polling`, `outbound_only`, `gateway_webhook` or `local`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_mode: Option<String>,
    /// Test/self-hosted endpoint override for the platform API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
    /// Generic-webhook body shape: `json`, `slack`, `discord`, `ntfy` or `text`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// Forward-compatible extension fields; a channel adapter may read them.
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

fn is_false(v: &bool) -> bool {
    !*v
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AgentChannelsFile {
    #[serde(default)]
    pub active: String,
    #[serde(default)]
    pub channels: BTreeMap<String, ChannelProfile>,
}

#[derive(Clone, Debug, Default)]
pub struct CliChannelOverrides {
    pub channels_config: Option<PathBuf>,
    pub channel: Option<String>,
    pub kind: Option<String>,
    pub channel_name: Option<String>,
    pub token: Option<String>,
    pub access_token: Option<String>,
    pub phone_number_id: Option<String>,
    pub app_secret: Option<String>,
    pub verify_token: Option<String>,
    pub url: Option<String>,
    pub default_recipient: Option<String>,
    pub allowed_senders: Vec<String>,
    pub allow_anonymous: bool,
    pub network_mode: Option<String>,
    pub api_base: Option<String>,
    pub test_send: bool,
}

#[derive(Clone, Debug)]
pub struct ResolvedChannels {
    pub config_path: Option<PathBuf>,
    pub active_profile: Option<String>,
    pub profiles: BTreeMap<String, ChannelProfile>,
}

impl ResolvedChannels {
    /// Serializes the runtime view. Secret values ARE included here — this is
    /// the process-local handoff to `src/js/channels.js`, never terminal output.
    pub fn to_js_json(&self) -> String {
        let mut channels = serde_json::Map::new();
        for (name, p) in &self.profiles {
            channels.insert(name.clone(), profile_js(p));
        }
        serde_json::json!({
            "configPath": self.config_path.as_ref().map(|p| p.display().to_string()),
            "active": self.active_profile,
            "channels": channels,
        })
        .to_string()
    }
}

fn profile_js(p: &ChannelProfile) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), serde_json::json!(p.kind));
    for (key, value) in [
        ("token", &p.token),
        ("accessToken", &p.access_token),
        ("phoneNumberId", &p.phone_number_id),
        ("appSecret", &p.app_secret),
        ("verifyToken", &p.verify_token),
        ("url", &p.url),
        ("defaultRecipient", &p.default_recipient),
        ("networkMode", &p.network_mode),
        ("apiBase", &p.api_base),
        ("format", &p.format),
    ] {
        if let Some(raw) = value {
            if let Some(resolved) = resolve_env_value(Some(raw)) {
                obj.insert(key.into(), serde_json::json!(resolved));
            }
        }
    }
    obj.insert(
        "allowedSenders".into(),
        serde_json::json!(p
            .allowed_senders
            .iter()
            .filter_map(|s| resolve_env_value(Some(s)))
            .collect::<Vec<String>>()),
    );
    if p.allow_anonymous {
        obj.insert("allowAnonymous".into(), serde_json::json!(true));
    }
    if !p.headers.is_empty() {
        let mut headers = serde_json::Map::new();
        for (k, v) in &p.headers {
            if let Some(resolved) = resolve_env_value(Some(v)) {
                headers.insert(k.clone(), serde_json::json!(resolved));
            }
        }
        obj.insert("headers".into(), serde_json::Value::Object(headers));
    }
    if !p.extra.is_empty() {
        obj.insert("extra".into(), serde_json::json!(p.extra));
    }
    serde_json::Value::Object(obj)
}

/// Resolves an `"env:VAR_NAME"` reference or returns a literal non-empty value.
pub fn resolve_env_value(raw: Option<&str>) -> Option<String> {
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

/// Masks a secret for terminal display while preserving `env:VAR` labels.
pub fn mask_secret(raw: Option<&str>) -> String {
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

/// Candidate paths searched when reading an agent's dedicated channel config.
pub fn candidate_paths(target: &crate::model_config::AgentTarget, explicit: Option<&Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = explicit {
        out.push(p.to_path_buf());
        return out;
    }
    if let Ok(env_p) = env::var("SOFUU_CHANNELS_CONFIG") {
        let trimmed = env_p.trim();
        if !trimmed.is_empty() {
            out.push(PathBuf::from(trimmed));
            return out;
        }
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let target_dir = target
        .path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = target
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("agent")
        .to_string();

    if target.is_binary {
        out.push(cwd.join(format!("{stem}.channels.json")));
        out.push(target_dir.join(format!("{stem}.channels.json")));
        out.push(target_dir.join("channels.json"));
        out.push(cwd.join("channels.json"));
    } else {
        if stem != "index" {
            out.push(target_dir.join(format!("{stem}.channels.json")));
        }
        out.push(target_dir.join("channels.json"));
        out.push(cwd.join("channels.json"));
    }
    out
}

pub fn find_config_path(
    target: &crate::model_config::AgentTarget,
    explicit: Option<&Path>,
) -> Option<PathBuf> {
    candidate_paths(target, explicit).into_iter().find(|p| p.is_file())
}

/// Chooses the target file path for `--setup-channel` / `--edit-channel`.
pub fn resolve_write_path(
    target: &crate::model_config::AgentTarget,
    explicit: Option<&Path>,
) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Ok(env_p) = env::var("SOFUU_CHANNELS_CONFIG") {
        let trimmed = env_p.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Some(existing) = find_config_path(target, None) {
        return existing;
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let target_dir = target
        .path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = target
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("agent")
        .to_string();

    if target.is_binary {
        let filename = format!("{stem}.channels.json");
        if dir_is_writable(&target_dir) {
            target_dir.join(filename)
        } else {
            cwd.join(filename)
        }
    } else if dir_is_writable(&target_dir) {
        target_dir.join("channels.json")
    } else {
        cwd.join("channels.json")
    }
}

pub fn load_channels_file(path: &Path) -> Result<AgentChannelsFile, String> {
    let raw =
        fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str::<AgentChannelsFile>(&raw)
        .map_err(|e| format!("invalid JSON in {}: {e}", path.display()))
}

pub fn save_channels_file(path: &Path, cfg: &AgentChannelsFile) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(cfg)
        .map_err(|e| format!("failed to serialize channels config: {e}"))?;
    fs::write(path, format!("{json}\n"))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

fn split_list(raw: &str) -> Vec<String> {
    raw.split(|c| c == ',' || c == ';' || c == ' ')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

/// Parses channel-related CLI flags from argv (stopping at `--`).
pub fn extract_cli_overrides(args: &[String]) -> CliChannelOverrides {
    let mut ov = CliChannelOverrides::default();
    let mut i = 1usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--" {
            break;
        }
        let next = args.get(i + 1).cloned();
        if (a == "--channels-config" || a == "--channels") && next.is_some() {
            ov.channels_config = next.map(PathBuf::from);
            i += 2;
            continue;
        }
        if let Some(rest) = a
            .strip_prefix("--channels-config=")
            .or_else(|| a.strip_prefix("--channels="))
        {
            if !rest.is_empty() {
                ov.channels_config = Some(PathBuf::from(rest));
            }
            i += 1;
            continue;
        }
        if a == "--channel" && next.is_some() {
            ov.channel = next;
            i += 2;
            continue;
        }
        if let Some(rest) = a.strip_prefix("--channel=") {
            if !rest.is_empty() {
                ov.channel = Some(rest.to_string());
            }
            i += 1;
            continue;
        }
        if (a == "--type" || a == "--channel-type") && next.is_some() {
            ov.kind = next;
            i += 2;
            continue;
        }
        if let Some(rest) = a
            .strip_prefix("--type=")
            .or_else(|| a.strip_prefix("--channel-type="))
        {
            if !rest.is_empty() {
                ov.kind = Some(rest.to_string());
            }
            i += 1;
            continue;
        }
        if (a == "--channel-name" || a == "--profile") && next.is_some() {
            ov.channel_name = next;
            i += 2;
            continue;
        }
        if let Some(rest) = a
            .strip_prefix("--channel-name=")
            .or_else(|| a.strip_prefix("--profile="))
        {
            if !rest.is_empty() {
                ov.channel_name = Some(rest.to_string());
            }
            i += 1;
            continue;
        }
        let value_flag = match a {
            "--token" => Some(0),
            "--access-token" => Some(1),
            "--phone-number-id" => Some(2),
            "--app-secret" => Some(3),
            "--verify-token" => Some(4),
            "--url" => Some(5),
            "--default-recipient" => Some(6),
            "--network-mode" => Some(7),
            "--api-base" => Some(8),
            _ => None,
        };
        if let Some(slot) = value_flag {
            if let Some(v) = next {
                match slot {
                    0 => ov.token = Some(v),
                    1 => ov.access_token = Some(v),
                    2 => ov.phone_number_id = Some(v),
                    3 => ov.app_secret = Some(v),
                    4 => ov.verify_token = Some(v),
                    5 => ov.url = Some(v),
                    6 => ov.default_recipient = Some(v),
                    7 => ov.network_mode = Some(v),
                    8 => ov.api_base = Some(v),
                    _ => {}
                }
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        let eq_flag = [
            ("--token=", 0),
            ("--access-token=", 1),
            ("--phone-number-id=", 2),
            ("--app-secret=", 3),
            ("--verify-token=", 4),
            ("--url=", 5),
            ("--default-recipient=", 6),
            ("--network-mode=", 7),
            ("--api-base=", 8),
        ]
        .iter()
        .find_map(|(prefix, slot)| a.strip_prefix(prefix).map(|rest| (*slot, rest)));
        if let Some((slot, v)) = eq_flag {
            if !v.is_empty() {
                let v = v.to_string();
                match slot {
                    0 => ov.token = Some(v),
                    1 => ov.access_token = Some(v),
                    2 => ov.phone_number_id = Some(v),
                    3 => ov.app_secret = Some(v),
                    4 => ov.verify_token = Some(v),
                    5 => ov.url = Some(v),
                    6 => ov.default_recipient = Some(v),
                    7 => ov.network_mode = Some(v),
                    8 => ov.api_base = Some(v),
                    _ => {}
                }
            }
            i += 1;
            continue;
        }
        if (a == "--allowed-senders" || a == "--allowed-sender") && next.is_some() {
            if let Some(v) = next {
                ov.allowed_senders.extend(split_list(&v));
            }
            i += 2;
            continue;
        }
        if let Some(rest) = a
            .strip_prefix("--allowed-senders=")
            .or_else(|| a.strip_prefix("--allowed-sender="))
        {
            ov.allowed_senders.extend(split_list(rest));
            i += 1;
            continue;
        }
        if a == "--allow-anonymous" {
            ov.allow_anonymous = true;
            i += 1;
            continue;
        }
        if a == "--test-send" {
            ov.test_send = true;
            i += 1;
            continue;
        }
        i += 1;
    }
    ov
}

/// Resolves effective channel config. `--channel` selects a profile for this
/// run without rewriting the file; an unknown name falls back to the file's
/// active profile.
pub fn apply_runtime_config(
    args: &[String],
    target: &crate::model_config::AgentTarget,
) -> ResolvedChannels {
    let ov = extract_cli_overrides(args);
    let config_path = find_config_path(target, ov.channels_config.as_deref());
    let file = config_path
        .as_deref()
        .and_then(|p| load_channels_file(p).ok())
        .unwrap_or_default();

    let active_profile = ov
        .channel
        .as_deref()
        .filter(|name| file.channels.contains_key(*name))
        .map(|s| s.to_string())
        .or_else(|| {
            if !file.active.is_empty() && file.channels.contains_key(&file.active) {
                Some(file.active.clone())
            } else {
                file.channels.keys().next().cloned()
            }
        });

    ResolvedChannels {
        config_path,
        active_profile,
        profiles: file.channels,
    }
}

/// Injects `globalThis.__sofuu_agent_channels` into the JS context.
pub fn js_prelude_for(resolved: &ResolvedChannels) -> String {
    let json = resolved.to_js_json();
    format!("try{{globalThis.__sofuu_agent_channels={json};}}catch(e){{}}")
}

// ── Interactive CLI Commands ──────────────────────────────────────────

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

fn default_network_mode(kind: &str) -> &'static str {
    match kind {
        "telegram" => "outbound_polling",
        "whatsapp" => "gateway_webhook",
        "webhook" => "outbound_only",
        "desktop" => "local",
        _ => "outbound_only",
    }
}

/// Renders the listing used by `--list-channels` and the edit manager. Kept as
/// a string builder so the masking contract is testable without capturing
/// stdout.
pub fn render_channels_table(path: &Path, cfg: &AgentChannelsFile) -> String {
    let mut out = String::new();
    out.push_str(&format!("\nAgent Channel Config: {}\n", path.display()));
    if cfg.channels.is_empty() {
        out.push_str("  (no channels configured yet)\n");
        out.push_str("  Run with --setup-channel to add one.\n\n");
        return out;
    }
    out.push('\n');
    for (idx, (name, p)) in cfg.channels.iter().enumerate() {
        let marker = if name == &cfg.active { "*" } else { " " };
        let network = p
            .network_mode
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default_network_mode(&p.kind));
        let recipient = p
            .default_recipient
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("(none)");
        let allowed = if p.allow_anonymous {
            "any".to_string()
        } else if p.allowed_senders.is_empty() {
            "0 (DROPS)".to_string()
        } else {
            p.allowed_senders.len().to_string()
        };
        let secret = mask_secret(
            p.token
                .as_deref()
                .or(p.access_token.as_deref())
                .or(p.app_secret.as_deref())
                .or(p.verify_token.as_deref())
                .or(p.url.as_deref()),
        );
        out.push_str(&format!(
            "  {marker} [{}] {name:<16} type={:<9} network={:<17} to={:<18} allowed={:<8} secret={secret}\n",
            idx + 1,
            p.kind,
            network,
            recipient,
            allowed
        ));
    }
    out.push('\n');
    out
}

fn print_channels_table(path: &Path, cfg: &AgentChannelsFile) {
    print!("{}", render_channels_table(path, cfg));
}

fn prompt_new_channel(default_name: Option<&str>) -> io::Result<(String, ChannelProfile)> {
    println!("\nChoose channel type:");
    println!("  [1] Telegram Bot API (outbound + long polling, zero open ports)");
    println!("  [2] WhatsApp Cloud API (outbound + signed webhook ingress)");
    println!("  [3] Generic Webhook (Slack, Discord, ntfy, custom HTTP)");
    println!("  [4] Desktop notification (macOS / Linux)");
    let choice = prompt_with_default("Choice [1-4]", "1")?;

    let (kind, default_profile, default_network) = match choice.as_str() {
        "2" | "whatsapp" => ("whatsapp", "my-whatsapp", "gateway_webhook"),
        "3" | "webhook" => ("webhook", "team-webhook", "outbound_only"),
        "4" | "desktop" => ("desktop", "desktop", "local"),
        _ => ("telegram", "my-telegram", "outbound_polling"),
    };

    let name = prompt_with_default("Profile name", default_name.unwrap_or(default_profile))?;
    let mut p = ChannelProfile {
        kind: kind.to_string(),
        network_mode: Some(
            prompt_with_default("Network mode", default_network)?.trim().to_string(),
        ),
        ..Default::default()
    };

    match kind {
        "telegram" => {
            p.token = Some(prompt_with_default(
                "Bot token (paste or env:TELEGRAM_BOT_TOKEN)",
                "env:TELEGRAM_BOT_TOKEN",
            )?);
            let to = prompt_with_default("Default chat id", "")?;
            p.default_recipient = if to.is_empty() { None } else { Some(to) };
        }
        "whatsapp" => {
            p.phone_number_id = Some(prompt_with_default("Phone number id", "")?);
            p.access_token = Some(prompt_with_default(
                "Access token (paste or env:WHATSAPP_ACCESS_TOKEN)",
                "env:WHATSAPP_ACCESS_TOKEN",
            )?);
            p.app_secret = Some(prompt_with_default(
                "App secret for X-Hub-Signature-256 (or env:WHATSAPP_APP_SECRET)",
                "env:WHATSAPP_APP_SECRET",
            )?);
            p.verify_token = Some(prompt_with_default(
                "Webhook verify token (or env:WHATSAPP_VERIFY_TOKEN)",
                "env:WHATSAPP_VERIFY_TOKEN",
            )?);
            let to = prompt_with_default("Default recipient phone (E.164)", "")?;
            p.default_recipient = if to.is_empty() { None } else { Some(to) };
        }
        "webhook" => {
            p.url = Some(prompt_with_default(
                "Webhook URL (paste or env:ALERT_WEBHOOK_URL)",
                "env:ALERT_WEBHOOK_URL",
            )?);
            let fmt = prompt_with_default("Body format [json/slack/discord/ntfy/text]", "json")?;
            p.format = if fmt.trim().is_empty() { None } else { Some(fmt) };
        }
        _ => {}
    }

    let allowed_default = p.default_recipient.clone().unwrap_or_default();
    let allowed =
        prompt_with_default("Allowed sender ids (comma-separated; empty = drop all)", &allowed_default)?;
    p.allowed_senders = split_list(&allowed);
    if p.allowed_senders.is_empty() {
        let anon = prompt_with_default("Allow any sender (fail-closed default is No)? (y/N)", "n")?;
        p.allow_anonymous = anon.eq_ignore_ascii_case("y") || anon.eq_ignore_ascii_case("yes");
    }

    let extra = prompt_with_default("Extra API base URL for self-hosted/test (optional)", "")?;
    p.api_base = if extra.is_empty() { None } else { Some(extra) };

    Ok((name, p))
}

fn edit_single_channel(name: &str, p: &mut ChannelProfile) -> io::Result<()> {
    println!("\nEditing channel '{name}' (press Enter to keep current, '-' to clear):");
    p.kind = prompt_with_default("  Type (telegram/whatsapp/webhook/desktop)", &p.kind)?;
    let network = p.network_mode.clone().unwrap_or_else(|| default_network_mode(&p.kind).to_string());
    p.network_mode = Some(prompt_with_default("  Network mode", &network)?);

    let edit_secret = |label: &str, slot: &mut Option<String>| -> io::Result<()> {
        let current = slot.clone().unwrap_or_default();
        let shown = if current.is_empty() { String::new() } else { mask_secret(Some(&current)) };
        let ans = if shown.is_empty() {
            prompt_line(&format!("  {label} (value or env:VAR): "))?
        } else {
            prompt_line(&format!("  {label} [{shown}]: "))?
        };
        if ans == "-" {
            *slot = None;
        } else if !ans.is_empty() {
            *slot = Some(ans);
        }
        Ok(())
    };

    edit_secret("Token", &mut p.token)?;
    edit_secret("Access token", &mut p.access_token)?;
    edit_secret("Phone number id", &mut p.phone_number_id)?;
    edit_secret("App secret", &mut p.app_secret)?;
    edit_secret("Verify token", &mut p.verify_token)?;
    edit_secret("Webhook URL", &mut p.url)?;

    let recipient = p.default_recipient.clone().unwrap_or_default();
    let ans = prompt_with_default("  Default recipient", &recipient)?;
    p.default_recipient = if ans == "-" || ans.is_empty() { None } else { Some(ans) };

    let cur_allowed = p.allowed_senders.join(",");
    let ans = prompt_with_default("  Allowed senders (comma-separated)", &cur_allowed)?;
    p.allowed_senders = split_list(&ans);
    let anon = prompt_with_default("  Allow anonymous senders? (y/N)", if p.allow_anonymous { "y" } else { "n" })?;
    p.allow_anonymous = anon.eq_ignore_ascii_case("y") || anon.eq_ignore_ascii_case("yes");

    let fmt = p.format.clone().unwrap_or_default();
    let ans = prompt_with_default("  Webhook body format", &fmt)?;
    p.format = if ans.is_empty() || ans == "-" { None } else { Some(ans) };
    Ok(())
}

fn channel_from_overrides(ov: &CliChannelOverrides) -> Result<ChannelProfile, String> {
    let kind = ov
        .kind
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "--type is required for a non-interactive --setup-channel".to_string())?;
    if !matches!(kind.as_str(), "telegram" | "whatsapp" | "webhook" | "desktop") {
        return Err(format!(
            "unknown channel type '{kind}' (expected telegram|whatsapp|webhook|desktop)"
        ));
    }
    let p = ChannelProfile {
        kind: kind.clone(),
        token: ov.token.clone(),
        access_token: ov.access_token.clone(),
        phone_number_id: ov.phone_number_id.clone(),
        app_secret: ov.app_secret.clone(),
        verify_token: ov.verify_token.clone(),
        url: ov.url.clone(),
        default_recipient: ov.default_recipient.clone(),
        allowed_senders: ov
            .allowed_senders
            .iter()
            .filter_map(|s| resolve_env_value(Some(s)))
            .collect(),
        allow_anonymous: ov.allow_anonymous,
        network_mode: Some(
            ov.network_mode
                .clone()
                .unwrap_or_else(|| default_network_mode(&kind).to_string()),
        ),
        api_base: ov.api_base.clone(),
        ..Default::default()
    };
    match kind.as_str() {
        "telegram" if p.token.as_deref().unwrap_or("").trim().is_empty() => {
            return Err("--token is required for a telegram channel".into())
        }
        "whatsapp" => {
            if p.access_token.as_deref().unwrap_or("").trim().is_empty() {
                return Err("--access-token is required for a whatsapp channel".into());
            }
            if p.phone_number_id.as_deref().unwrap_or("").trim().is_empty() {
                return Err("--phone-number-id is required for a whatsapp channel".into());
            }
        }
        "webhook" if p.url.as_deref().unwrap_or("").trim().is_empty() => {
            return Err("--url is required for a webhook channel".into())
        }
        _ => {}
    }
    if p.allowed_senders.is_empty() && !p.allow_anonymous {
        eprintln!(
            "  ⚠ allowed_senders is empty: inbound messages will be dropped (fail-closed).\n\
             \x20   Pass --allowed-senders <id,...> or --allow-anonymous."
        );
    }
    Ok(p)
}

fn test_send_profile(name: &str, cfg: &AgentChannelsFile) {
    let resolved = ResolvedChannels {
        config_path: None,
        active_profile: Some(name.to_string()),
        profiles: cfg.channels.clone(),
    };
    let prelude = js_prelude_for(&resolved);
    let name_js = serde_json::to_string(name).unwrap_or_else(|_| "\"\"".into());
    let code = format!(
        "{prelude}\
         globalThis.__sofuu_channel_test={{done:false,ok:false,err:''}};\
         sofuu.channels.send({{channel:{name_js},text:'Sofuu channel connected!'}})\
           .then(function(){{globalThis.__sofuu_channel_test.ok=true;}})\
           .catch(function(e){{globalThis.__sofuu_channel_test.err=String((e&&e.message)||e);}})\
           .finally(function(){{globalThis.__sofuu_channel_test.done=true;}});"
    );
    let rt = match sofuu_ffi::SofuuRuntime::init() {
        Some(rt) => rt,
        None => {
            eprintln!("  ⚠ test send skipped: runtime init failed");
            return;
        }
    };
    let _ = rt.eval_string(&code, "<channel-test-send>");
    let mut done = false;
    for _ in 0..600 {
        rt.run_jobs();
        if let Some(out) = rt.eval_repl("globalThis.__sofuu_channel_test.done") {
            if out.trim() == "true" {
                done = true;
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    if !done {
        eprintln!("  ⚠ test send timed out");
        return;
    }
    let ok = rt
        .eval_repl("globalThis.__sofuu_channel_test.ok")
        .map(|s| s.trim() == "true")
        .unwrap_or(false);
    let err = rt
        .eval_repl("globalThis.__sofuu_channel_test.err")
        .unwrap_or_default();
    if ok {
        println!("  ✓ test message sent through '{name}'");
    } else {
        eprintln!("  ⚠ test send failed: {}", err.trim().trim_matches('"'));
    }
}

fn run_setup_channel(target: &crate::model_config::AgentTarget, args: &[String]) -> ExitCode {
    let ov = extract_cli_overrides(args);
    let path = resolve_write_path(target, ov.channels_config.as_deref());
    let mut cfg = if path.is_file() {
        load_channels_file(&path).unwrap_or_default()
    } else {
        AgentChannelsFile::default()
    };

    // Non-interactive path: --setup-channel --type <kind> plus its fields.
    if ov.kind.is_some() {
        let profile = match channel_from_overrides(&ov) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Error: {e}");
                return ExitCode::from(2);
            }
        };
        let name = ov
            .channel_name
            .clone()
            .unwrap_or_else(|| profile.kind.clone());
        cfg.channels.insert(name.clone(), profile);
        cfg.active = name.clone();
        return match save_channels_file(&path, &cfg) {
            Ok(()) => {
                println!("✓ Saved channel '{name}' (active) to {}", path.display());
                if ov.test_send {
                    test_send_profile(&name, &cfg);
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("Error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    println!("\n┌──────────────────────────────────────────────────────────┐");
    println!("│ Agent Messaging Channel Setup                            │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!("Config file: {}", path.display());

    let (name, profile) = match prompt_new_channel(ov.channel_name.as_deref()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Setup aborted: {e}");
            return ExitCode::from(2);
        }
    };

    let make_active = prompt_with_default(&format!("Set '{name}' as active? (Y/n)"), "y")
        .map(|s| !s.eq_ignore_ascii_case("n") && !s.eq_ignore_ascii_case("no"))
        .unwrap_or(true);
    cfg.channels.insert(name.clone(), profile);
    if make_active || cfg.active.is_empty() {
        cfg.active = name.clone();
    }

    match save_channels_file(&path, &cfg) {
        Ok(()) => {
            println!(
                "\n✓ Saved channel '{name}'{} to {}",
                if cfg.active == name { " (active)" } else { "" },
                path.display()
            );
            let test = prompt_with_default("Send a test message now? (y/N)", "n")
                .map(|s| s.eq_ignore_ascii_case("y") || s.eq_ignore_ascii_case("yes"))
                .unwrap_or(false);
            if test {
                test_send_profile(&name, &cfg);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn resolve_channel_pick(cfg: &AgentChannelsFile, raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if cfg.channels.contains_key(trimmed) {
        return Some(trimmed.to_string());
    }
    if let Ok(idx) = trimmed.parse::<usize>() {
        if idx >= 1 {
            return cfg.channels.keys().nth(idx - 1).cloned();
        }
    }
    None
}

fn run_edit_channel(
    target: &crate::model_config::AgentTarget,
    args: &[String],
    direct_profile: Option<&str>,
) -> ExitCode {
    let ov = extract_cli_overrides(args);
    let path = resolve_write_path(target, ov.channels_config.as_deref());
    let mut cfg = if path.is_file() {
        match load_channels_file(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        AgentChannelsFile::default()
    };

    if let Some(target_name) = direct_profile {
        if let Some(key) = resolve_channel_pick(&cfg, target_name) {
            if let Some(prof) = cfg.channels.get_mut(&key) {
                if let Err(e) = edit_single_channel(&key, prof) {
                    eprintln!("Edit aborted: {e}");
                    return ExitCode::from(2);
                }
                return match save_channels_file(&path, &cfg) {
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
            eprintln!("Channel '{target_name}' not found in {}", path.display());
            print_channels_table(&path, &cfg);
            return ExitCode::from(2);
        }
    }

    loop {
        print_channels_table(&path, &cfg);
        println!("Actions:");
        println!("  [s] Switch active channel");
        println!("  [e] Edit a channel");
        println!("  [a] Add a new channel");
        println!("  [d] Delete a channel");
        println!("  [q] Save & quit");
        let Ok(action) = prompt_line("Action [s/e/a/d/q]: ") else {
            break;
        };
        match action.to_lowercase().as_str() {
            "s" => {
                let Ok(pick) = prompt_line("Enter channel name or number to activate: ") else {
                    break;
                };
                if let Some(key) = resolve_channel_pick(&cfg, &pick) {
                    cfg.active = key.clone();
                    if let Err(e) = save_channels_file(&path, &cfg) {
                        eprintln!("Error: {e}");
                        return ExitCode::FAILURE;
                    }
                    println!("✓ Active channel set to '{key}'.");
                } else {
                    println!("  Not found: '{pick}'");
                }
            }
            "e" => {
                let Ok(pick) = prompt_line("Enter channel name or number to edit: ") else {
                    break;
                };
                if let Some(key) = resolve_channel_pick(&cfg, &pick) {
                    if let Some(prof) = cfg.channels.get_mut(&key) {
                        if let Err(e) = edit_single_channel(&key, prof) {
                            eprintln!("Edit aborted: {e}");
                            return ExitCode::from(2);
                        }
                        if let Err(e) = save_channels_file(&path, &cfg) {
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
                let Ok((name, prof)) = prompt_new_channel(None) else {
                    break;
                };
                cfg.channels.insert(name.clone(), prof);
                if cfg.active.is_empty() {
                    cfg.active = name.clone();
                }
                if let Err(e) = save_channels_file(&path, &cfg) {
                    eprintln!("Error: {e}");
                    return ExitCode::FAILURE;
                }
                println!("✓ Added '{name}'.");
            }
            "d" => {
                let Ok(pick) = prompt_line("Enter channel name or number to delete: ") else {
                    break;
                };
                if let Some(key) = resolve_channel_pick(&cfg, &pick) {
                    cfg.channels.remove(&key);
                    if cfg.active == key {
                        cfg.active = cfg.channels.keys().next().cloned().unwrap_or_default();
                    }
                    if let Err(e) = save_channels_file(&path, &cfg) {
                        eprintln!("Error: {e}");
                        return ExitCode::FAILURE;
                    }
                    println!("✓ Deleted '{key}'.");
                } else {
                    println!("  Not found: '{pick}'");
                }
            }
            "q" | "" => {
                if let Err(e) = save_channels_file(&path, &cfg) {
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

fn run_list_channels(target: &crate::model_config::AgentTarget, args: &[String]) -> ExitCode {
    let ov = extract_cli_overrides(args);
    let path = find_config_path(target, ov.channels_config.as_deref())
        .unwrap_or_else(|| resolve_write_path(target, ov.channels_config.as_deref()));
    let cfg = if path.is_file() {
        match load_channels_file(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        AgentChannelsFile::default()
    };
    print_channels_table(&path, &cfg);
    ExitCode::SUCCESS
}

/// Intercepts channel management flags before `--`.
pub fn handle_cli_command(
    args: &[String],
    target: &crate::model_config::AgentTarget,
) -> Option<ExitCode> {
    let mut i = 1usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--" {
            return None;
        }
        if a == "--setup-channel" {
            return Some(run_setup_channel(target, args));
        }
        if a == "--list-channels" {
            return Some(run_list_channels(target, args));
        }
        if a == "--edit-channel" {
            let direct = args
                .get(i + 1)
                .map(|s| s.as_str())
                .filter(|s| !s.starts_with('-'));
            return Some(run_edit_channel(target, args, direct));
        }
        if let Some(rest) = a.strip_prefix("--edit-channel=") {
            let direct = if rest.is_empty() { None } else { Some(rest) };
            return Some(run_edit_channel(target, args, direct));
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::AgentTarget;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!(
            "sofuu-channels-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn resolves_env_values_and_masks_literals() {
        env::set_var("SOFUU_TEST_CHANNEL_TOKEN", "telegram-secret-99");
        assert_eq!(
            resolve_env_value(Some("env:SOFUU_TEST_CHANNEL_TOKEN")).as_deref(),
            Some("telegram-secret-99")
        );
        env::remove_var("SOFUU_TEST_CHANNEL_TOKEN");
        assert_eq!(
            resolve_env_value(Some("literal-token")).as_deref(),
            Some("literal-token")
        );
        assert_eq!(mask_secret(Some("env:TELEGRAM_BOT_TOKEN")), "env:TELEGRAM_BOT_TOKEN");
        assert_eq!(mask_secret(Some("123456:ABCDEFGHIJK")), "1234...HIJK");
        assert_eq!(mask_secret(None), "(none)");
    }

    #[test]
    fn setup_channel_noninteractive_creates_per_binary_file_with_0600() {
        let tmp = tmp_dir("setup");
        let bin = tmp.join("my-agent");
        let target = AgentTarget::binary(&bin);
        let args: Vec<String> = vec![
            bin.display().to_string(),
            "--setup-channel".into(),
            "--type".into(),
            "telegram".into(),
            "--channel-name".into(),
            "alerts".into(),
            "--token".into(),
            "env:TELEGRAM_BOT_TOKEN".into(),
            "--default-recipient".into(),
            "123456789".into(),
            "--allowed-senders".into(),
            "123456789, 987654321".into(),
        ];
        let code = handle_cli_command(&args, &target);
        assert_eq!(code, Some(ExitCode::SUCCESS));

        let expected = tmp.join("my-agent.channels.json");
        assert!(expected.is_file(), "should write per-binary .channels.json");
        let loaded = load_channels_file(&expected).unwrap();
        assert_eq!(loaded.active, "alerts");
        assert_eq!(loaded.channels["alerts"].kind, "telegram");
        assert_eq!(loaded.channels["alerts"].allowed_senders.len(), 2);
        assert_eq!(
            loaded.channels["alerts"].network_mode.as_deref(),
            Some("outbound_polling")
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&expected).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "channel secrets must be 0600");
        }

        let resolved = apply_runtime_config(&[bin.display().to_string()], &target);
        assert_eq!(resolved.active_profile.as_deref(), Some("alerts"));
        let prelude = js_prelude_for(&resolved);
        assert!(prelude.contains("__sofuu_agent_channels"));
        assert!(prelude.contains("allowedSenders"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn discovery_prefers_explicit_then_per_binary_then_project() {
        let tmp = tmp_dir("discovery");
        let bin = tmp.join("sentinel");
        let target = AgentTarget::binary(&bin);

        let explicit = tmp.join("custom.json");
        let paths = candidate_paths(&target, Some(&explicit));
        assert_eq!(paths, vec![explicit.clone()]);

        let paths = candidate_paths(&target, None);
        assert_eq!(paths[0], env::current_dir().unwrap().join("sentinel.channels.json"));
        assert_eq!(paths[1], tmp.join("sentinel.channels.json"));
        assert_eq!(paths[2], tmp.join("channels.json"));
        assert_eq!(paths[3], env::current_dir().unwrap().join("channels.json"));

        let cfg = AgentChannelsFile {
            active: "a".into(),
            channels: BTreeMap::from([(
                "a".to_string(),
                ChannelProfile {
                    kind: "telegram".into(),
                    token: Some("env:TELEGRAM_BOT_TOKEN".into()),
                    allowed_senders: vec!["1".into()],
                    ..Default::default()
                },
            )]),
        };
        save_channels_file(&tmp.join("channels.json"), &cfg).unwrap();
        assert_eq!(
            find_config_path(&target, None).unwrap(),
            tmp.join("channels.json")
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn listing_masks_secrets_and_flags_empty_allowlists() {
        let cfg = AgentChannelsFile {
            active: "tg".into(),
            channels: BTreeMap::from([
                (
                    "tg".to_string(),
                    ChannelProfile {
                        kind: "telegram".into(),
                        token: Some("123456789:ABCDEFGHIJKLMNOP".into()),
                        default_recipient: Some("42".into()),
                        allowed_senders: vec!["42".into()],
                        ..Default::default()
                    },
                ),
                (
                    "hook".to_string(),
                    ChannelProfile {
                        kind: "webhook".into(),
                        url: Some("https://example.invalid/hook".into()),
                        ..Default::default()
                    },
                ),
            ]),
        };
        let text = render_channels_table(Path::new("/tmp/channels.json"), &cfg);
        assert!(text.contains("type=telegram"));
        assert!(text.contains("1234...MNOP"), "{text}");
        assert!(!text.contains("123456789:ABCDEFGHIJKLMNOP"), "{text}");
        assert!(text.contains("0 (DROPS)"), "{text}");
        assert!(text.lines().any(|l| l.contains("tg") && l.trim_start().starts_with('*')), "{text}");
    }
}
