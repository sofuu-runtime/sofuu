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

use serde_json::{json, Map, Value};
use std::path::PathBuf;

fn sofuu_dir() -> PathBuf {
    if let Some(root) = sofuu_core::embed_config::get_config_root() {
        return PathBuf::from(root);
    }
    let home = sofuu_core::embed_config::home_dir().unwrap_or_else(|| ".".into());
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
/// the desktop must not clobber fields it does not understand. "sync" is
/// deliberately absent: the session mesh is built-in (chat.js persists
/// sessions unconditionally), so no UI surface may disable it.
const PATCHABLE: &[&str] = &[
    "provider", "model", "effort", "brain", "ml", "rlm", "base_url", "profile",
    "embed_provider", "embed_model", "ctx_window", "max_output", "budget_usd", "ghost",
    "recall_min", "recall_budget", "providers", "active",
];

/// Merge a WebView "providers" patch into the stored array by provider
/// name. The frontend only sees REDACTED entries (api_key: "", plus a
/// derived has_api_key), so a wholesale replace would wipe CLI-written
/// keys and any field the patch does not carry. The patch updates the
/// fields it names and NEVER touches api_key/has_api_key; new names are
/// appended, unknown existing entries are preserved.
fn merge_providers(obj: &mut Map<String, Value>, val: &Value) {
    let Some(patch) = val.as_array() else { return };
    let mut existing: Vec<Value> = obj
        .get("providers")
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();
    for pv in patch {
        let Some(po) = pv.as_object() else { continue };
        let Some(name) = po.get("name").and_then(|n| n.as_str()) else { continue };
        if name.is_empty() {
            continue;
        }
        let pos = existing
            .iter()
            .position(|e| e.get("name").and_then(|n| n.as_str()) == Some(name));
        let target = if let Some(pos) = pos {
            &mut existing[pos]
        } else {
            existing.push(json!({}));
            existing.last_mut().unwrap()
        };
        let Some(to) = target.as_object_mut() else { continue };
        for (k, v) in po {
            if k == "api_key" || k == "has_api_key" {
                continue;
            }
            to.insert(k.clone(), v.clone());
        }
    }
    obj.insert("providers".into(), Value::Array(existing));
}

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
        if k == "providers" {
            merge_providers(obj, val);
            continue;
        }
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
    // Atomic write, 0600 from the first byte (no write-then-chmod window).
    let dir = sofuu_dir();
    let _ = std::fs::create_dir_all(&dir);
    let tmp = dir.join("config.json.tmp");
    let out = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        use std::io::Write;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)
            .and_then(|mut f| f.write_all(out.as_bytes()))
            .map_err(|e| e.to_string())?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(&tmp, out).map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, config_path()).map_err(|e| e.to_string())?;
    let mut redacted = v.clone();
    redact(&mut redacted);
    Ok(redacted)
}

/// Delete a provider by name: drop its entry from the providers array.
/// Refuses the active provider (switch first — the engine has no fallback
/// and a missing active provider would fail every turn). Returns the
/// redacted config for the caller to hand back to the WebView.
pub fn remove_provider(name: &str) -> Result<Value, String> {
    let mut v = std::fs::read_to_string(config_path())
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .unwrap_or_else(|| json!({}));
    // Read before the mutable borrow below (same object).
    let active = v
        .get("active")
        .and_then(|a| a.as_str())
        .unwrap_or("")
        .to_string();
    if name == active {
        return Err(format!("{name} is the active provider — switch first"));
    }
    let Some(obj) = v.as_object_mut() else {
        return Err("config.json is not an object".into());
    };
    let Some(arr) = obj.get_mut("providers").and_then(|p| p.as_array_mut()) else {
        return Err("no providers configured".into());
    };
    let before = arr.len();
    arr.retain(|p| p.get("name").and_then(|n| n.as_str()) != Some(name));
    if arr.len() == before {
        return Err(format!("no provider named {name}"));
    }
    // Atomic write, 0600 from the first byte — same contract as update_config.
    let dir = sofuu_dir();
    let _ = std::fs::create_dir_all(&dir);
    let tmp = dir.join("config.json.tmp");
    let out = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        use std::io::Write;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)
            .and_then(|mut f| f.write_all(out.as_bytes()))
            .map_err(|e| e.to_string())?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(&tmp, out).map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, config_path()).map_err(|e| e.to_string())?;
    let mut redacted = v.clone();
    redact(&mut redacted);
    Ok(redacted)
}

/// Raw provider lookup for the model picker: (endpoint, profile, api_key).
/// KEY IS FROM THE CONFIG FILE ONLY (CLI-written) — a desktop-kept key lives
/// in the Keychain and is resolved by the caller. Never returns this to the
/// WebView: the picker command bakes it into an engine-side fetch script.
pub fn provider_entry(name: &str) -> Option<(String, String, String)> {
    let v = std::fs::read_to_string(config_path())
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())?;
    let arr = v.get("providers")?.as_array()?;
    for p in arr {
        let Some(o) = p.as_object() else { continue };
        if o.get("name").and_then(|n| n.as_str()) != Some(name) {
            continue;
        }
        return Some((
            o.get("endpoint").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            o.get("profile").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            o.get("api_key").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        ));
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (PathBuf, std::sync::MutexGuard<'static, ()>) {
        // The config root is a process-global; tests run in parallel, so a
        // shared dir would cross-seed AND a second configure() would
        // re-point the root mid-test. Unique dir + a global lock keeps
        // every config test serialized on its own root.
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "sofuu-desktop-config-test-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        sofuu_core::embed_config::configure(false, Some(dir.to_string_lossy().to_string()), false);
        (dir, guard)
    }

    fn seed(dir: &PathBuf, v: Value) {
        std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&v).unwrap()).unwrap();
    }

    #[test]
    fn update_config_merges_providers_and_preserves_keys() {
        let (dir, _lock) = setup();

        // 1. Patching the ACTIVE provider's model (redacted round-trip) must
        //    keep the CLI-written api_key and update the flat mirror.
        seed(&dir, json!({
            "providers": [{
                "name": "openai", "endpoint": "https://api.openai.com/v1/chat/completions",
                "profile": "openai", "model": "gpt-4o", "api_key": "sk-secret"
            }],
            "active": "openai", "provider": "openai", "model": "gpt-4o"
        }));
        update_config(json!({
            "providers": [{
                "name": "openai", "endpoint": "https://api.openai.com/v1/chat/completions",
                "profile": "openai", "model": "gpt-5", "api_key": "", "has_api_key": true
            }],
            "active": "openai", "model": "gpt-5"
        }))
        .unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let p = &v["providers"][0];
        assert_eq!(p["api_key"], "sk-secret", "api_key must survive a redacted patch");
        assert_eq!(p["model"], "gpt-5");
        assert!(p.get("has_api_key").is_none(), "has_api_key is webview-only, never persisted");
        assert_eq!(v["model"], "gpt-5", "flat mirror follows the entry");
        assert_eq!(v["provider"], "openai");

        // 2. Adding a NEW provider appends it and switches active + mirror.
        update_config(json!({
            "providers": [
                { "name": "openai", "endpoint": "https://api.openai.com/v1/chat/completions",
                  "profile": "openai", "model": "gpt-5", "api_key": "" },
                { "name": "anthropic", "endpoint": "https://api.anthropic.com",
                  "profile": "anthropic", "model": "claude-3-7", "api_key": "" }
            ],
            "active": "anthropic", "model": "claude-3-7"
        }))
        .unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let arr = v["providers"].as_array().unwrap();
        assert_eq!(arr.len(), 2, "new provider is appended, old entries preserved");
        assert_eq!(arr[0]["api_key"], "sk-secret");
        assert_eq!(arr[1]["model"], "claude-3-7");
        assert_eq!(v["active"], "anthropic");
        assert_eq!(v["model"], "claude-3-7");
        assert_eq!(v["provider"], "anthropic");

        // 3. Switching to a provider with an EMPTY model clears the flat
        //    mirror — "no model selected" until the user picks one.
        update_config(json!({
            "providers": [
                { "name": "openai", "endpoint": "https://api.openai.com/v1/chat/completions",
                  "profile": "openai", "model": "gpt-5", "api_key": "" },
                { "name": "anthropic", "endpoint": "https://api.anthropic.com",
                  "profile": "anthropic", "model": "claude-3-7", "api_key": "" },
                { "name": "local", "endpoint": "http://127.0.0.1:11434",
                  "profile": "local", "model": "", "api_key": "" }
            ],
            "active": "local"
        }))
        .unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        assert_eq!(v["model"], "", "no model until picked");
        assert_eq!(v["base_url"], "http://127.0.0.1:11434");
        assert_eq!(v["provider"], "local");
    }

    #[test]
    fn remove_provider_refuses_active_and_unknown() {
        let (dir, _lock) = setup();
        seed(&dir, json!({
            "providers": [
                { "name": "openai", "endpoint": "https://api.openai.com/v1", "profile": "openai", "model": "gpt-5", "api_key": "sk-secret" },
                { "name": "local", "endpoint": "http://127.0.0.1:11434", "profile": "local", "model": "", "api_key": "" }
            ],
            "active": "openai"
        }));
        // The active provider is refused — the engine has no fallback.
        let err = remove_provider("openai").unwrap_err();
        assert!(err.contains("active"), "got: {err}");
        // Unknown names are refused too.
        assert!(remove_provider("ghost").is_err());
        // A non-active provider drops cleanly and keys survive elsewhere.
        let cfg = remove_provider("local").unwrap();
        assert_eq!(cfg["providers"].as_array().unwrap().len(), 1);
        assert_eq!(cfg["providers"][0]["api_key"], "", "returned config stays redacted");
        let raw: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        assert_eq!(raw["providers"][0]["api_key"], "sk-secret", "kept provider's key untouched on disk");
    }
}
