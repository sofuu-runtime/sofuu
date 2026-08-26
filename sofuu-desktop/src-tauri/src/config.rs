// config.rs — desktop config access (PLAN-DESKTOP D).
//
// Two files:
//   ~/.sofuu/config.json  — the SHARED chat config (same identity as the
//                           CLI). Read/write here mirrors ChatConfig's
//                           flat-from-active sync so both hosts boot.
//                           API keys are REDACTED in every response — the
//                           desktop keeps provider keys in the Keychain.
//   ~/.sofuu/desktop.json — desktop-only state (project dir). Kept
//                           separate so the CLI's save() (which writes only
//                           keys it knows) can never drop it.
//
// Workstream B will lib-ify ChatConfig; then this module can delegate to
// the real type instead of mirroring the JSON shape.

use serde_json::{json, Value};
use std::path::PathBuf;

fn sofuu_dir() -> PathBuf {
    if let Some(root) = sofuu_core::embed_config::get_config_root() {
        return PathBuf::from(root);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".sofuu")
}

fn config_path() -> PathBuf {
    sofuu_dir().join("config.json")
}

fn desktop_path() -> PathBuf {
    sofuu_dir().join("desktop.json")
}

/// Read the shared config as JSON with api_key fields REDACTED (the WebView
/// never sees raw keys — only whether one is set).
pub fn get_config_redacted() -> Value {
    let mut v = std::fs::read_to_string(config_path())
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .unwrap_or_else(|| json!({}));
    redact(&mut v);
    v
}

fn redact(v: &mut Value) {
    if let Some(obj) = v.as_object_mut() {
        if let Some(key) = obj.get("api_key").and_then(|k| k.as_str()).map(|s| s.to_string()) {
            obj.insert("api_key".into(), json!(""));
            obj.insert("has_api_key".into(), json!(!key.is_empty()));
        }
        if let Some(providers) = obj.get_mut("providers").and_then(|p| p.as_array_mut()) {
            for p in providers.iter_mut() {
                if let Some(po) = p.as_object_mut() {
                    let has = po
                        .get("api_key")
                        .and_then(|k| k.as_str())
                        .map(|s| !s.is_empty())
                        .unwrap_or(false);
                    po.insert("api_key".into(), json!(""));
                    po.insert("has_api_key".into(), json!(has));
                }
            }
        }
    }
}

/// Whitelisted keys the settings UI may change. Anything else is ignored —
/// the desktop must not clobber fields it does not understand.
const PATCHABLE: &[&str] = &[
    "provider", "model", "effort", "brain", "ml", "sync", "rlm", "base_url", "profile",
    "embed_provider", "embed_model", "ctx_window", "max_output", "budget_usd", "ghost",
    "recall_min", "recall_budget", "providers", "active",
];

/// Apply a settings patch to the shared config. Keeps the legacy flat
/// mirror in sync with the active provider entry (ChatConfig's contract),
/// and never touches api_key fields (Keychain owns keys on desktop).
pub fn update_config(patch: Value) -> Result<Value, String> {
    let mut v = std::fs::read_to_string(config_path())
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .unwrap_or_else(|| json!({}));
    let Some(obj) = v.as_object_mut() else {
        return Err("config.json is not an object".into());
    };
    let Some(patch_obj) = patch.as_object() else {
        return Err("patch must be an object".into());
    };
    for (k, val) in patch_obj {
        if PATCHABLE.contains(&k.as_str()) {
            obj.insert(k.clone(), val.clone());
        }
    }
    // Sync the flat mirror from the active entry (same rule as ChatConfig).
    let active = obj.get("active").and_then(|a| a.as_str()).unwrap_or("").to_string();
    if !active.is_empty() {
        let entry = obj
            .get("providers")
            .and_then(|p| p.as_array())
            .and_then(|arr| arr.iter().find(|p| p.get("name").and_then(|n| n.as_str()) == Some(active.as_str())))
            .cloned();
        if let Some(e) = entry {
            // Preserve the existing api_key — Keychain/CLI own it.
            let existing_key = obj.get("api_key").cloned().unwrap_or_else(|| json!(""));
            obj.insert("provider".into(), e.get("name").cloned().unwrap_or(json!("")));
            obj.insert("model".into(), e.get("model").cloned().unwrap_or(json!("")));
            obj.insert("base_url".into(), e.get("endpoint").cloned().unwrap_or(json!("")));
            obj.insert("profile".into(), e.get("profile").cloned().unwrap_or(json!("")));
            obj.insert("api_key".into(), existing_key);
        }
    }
    // Atomic write, 0600 (keys may still live here from CLI use).
    let dir = sofuu_dir();
    let _ = std::fs::create_dir_all(&dir);
    let tmp = dir.join("config.json.tmp");
    let out = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
    std::fs::write(&tmp, out).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    std::fs::rename(&tmp, config_path()).map_err(|e| e.to_string())?;
    let mut redacted = v.clone();
    redact(&mut redacted);
    Ok(redacted)
}

// ── desktop.json (desktop-only state) ───────────────────────────────

pub fn get_desktop_state() -> Value {
    std::fs::read_to_string(desktop_path())
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .unwrap_or_else(|| json!({}))
}

pub fn set_project_dir(path: &str) -> Result<(), String> {
    let mut v = get_desktop_state();
    if let Some(obj) = v.as_object_mut() {
        obj.insert("project_dir".into(), json!(path));
    }
    let dir = sofuu_dir();
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(desktop_path(), serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

pub fn project_dir() -> Option<String> {
    get_desktop_state()
        .get("project_dir")
        .and_then(|p| p.as_str())
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
}
