// commands.rs — the Tauri command surface (PLAN-DESKTOP D).
//
// Thin adapters: turns/cancel/approvals go to the engine worker (or the
// poke for mid-turn signals), reads are served from the Rust mirrors.

use std::time::Duration;

use serde_json::{json, Value};
use tauri::State;

use crate::config;
use crate::engine::EngineHandle;
use crate::keychain;
use crate::sessions;

const QUERY_TIMEOUT: Duration = Duration::from_secs(30);

/// Current project dir (tool jail + session root). Falls back to $HOME so a
/// first-launch without a picked dir still has a sane root.
fn project_dir() -> String {
    config::project_dir().unwrap_or_else(|| std::env::var("HOME").unwrap_or_else(|_| ".".into()))
}

// ── Turns ─────────────────────────────────────────────────────────

#[tauri::command]
pub fn send_turn(engine: State<'_, EngineHandle>, text: String, opts: Option<Value>) -> Result<Value, String> {
    engine.send_turn(text, opts.unwrap_or_else(|| json!({})))?;
    Ok(json!({ "ok": true }))
}

/// Cancel the running turn. Goes through the poke — safe while the engine
/// is blocked inside the turn's eval.
#[tauri::command]
pub fn cancel_turn() -> Result<Value, String> {
    sofuu_core::rt::host_poke::host_poke_send(r#"{"type":"cancel"}"#);
    Ok(json!({ "ok": true }))
}

/// Resolve a pending tool approval (allow/deny, optionally always-allow
/// for the session). Also poke-delivered — the promise lives in chat.js.
#[tauri::command]
pub fn resolve_approval(id: String, allow: bool, always: Option<bool>) -> Result<Value, String> {
    let msg = json!({
        "type": "approval",
        "id": id,
        "allow": allow,
        "always": always.unwrap_or(false),
    });
    sofuu_core::rt::host_poke::host_poke_send(&msg.to_string());
    Ok(json!({ "ok": true }))
}

// ── Sessions ──────────────────────────────────────────────────────

#[tauri::command]
pub fn list_sessions() -> Result<Value, String> {
    let sessions = sessions::list_sessions(std::path::Path::new(&project_dir()));
    serde_json::to_value(sessions).map_err(|e| e.to_string())
}

/// Turn history for a session — served by chat.js through the engine (the
/// .qtsq files are encrypted; only the runtime can read them).
#[tauri::command]
pub fn session_turns(engine: State<'_, EngineHandle>, id: String) -> Result<Value, String> {
    let id_js = serde_json::to_string(&id).unwrap_or_else(|_| "\"\"".into());
    let script = format!(
        "__desktop_reply(JSON.stringify( \
           (sofuu.chat && sofuu.chat.sessionTurns) ? sofuu.chat.sessionTurns({id_js}) : {{error:'chat engine unavailable'}} \
         ))"
    );
    let reply = engine.query(script, QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn new_session(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script =
        "__desktop_reply(JSON.stringify((sofuu.chat && sofuu.chat.newSession) ? sofuu.chat.newSession() : {error:'chat engine unavailable'}))";
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

/// Switch the engine's active session to `id` (loads its transcript into
/// the engine so the next turn continues that session). Fresh sessions
/// without any turns fail here — the frontend still keeps the cached
/// transcript on its side.
#[tauri::command]
pub fn resume_session(engine: State<'_, EngineHandle>, id: String) -> Result<Value, String> {
    let id_js = serde_json::to_string(&id).unwrap_or_else(|_| "\"\"".into());
    let script = format!(
        "__desktop_reply(JSON.stringify( \
           (sofuu.chat && sofuu.chat.resume) ? sofuu.chat.resume({id_js}) : {{error:'chat engine unavailable'}} \
         ))"
    );
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

/// Set the permission profile: 'full' | 'edit' | 'plan' | 'prompt'.
#[tauri::command]
pub fn set_permissions(engine: State<'_, EngineHandle>, profile: String) -> Result<Value, String> {
    let profile_js = serde_json::to_string(&profile).unwrap_or_else(|_| "\"\"".into());
    let script = format!(
        "__desktop_reply(JSON.stringify( \
           (sofuu.chat && sofuu.chat.setPermissions) ? sofuu.chat.setPermissions({profile_js}) : {{error:'chat engine unavailable'}} \
         ))"
    );
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn chat_state(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script =
        "__desktop_reply(JSON.stringify((sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {error:'chat engine unavailable'}))";
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn compact(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script =
        "(function(){ if (sofuu.chat && sofuu.chat.compact) { Promise.resolve(sofuu.chat.compact()).then(function(r){ __desktop_reply(JSON.stringify(r||{ok:true})); }, function(e){ __desktop_reply(JSON.stringify({error:String(e&&e.message||e)})); }); } else { __desktop_reply(JSON.stringify({error:'chat engine unavailable'})); } })()";
    let reply = engine.query(script.to_string(), Duration::from_secs(120))?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

// ── Config ────────────────────────────────────────────────────────

#[tauri::command]
pub fn get_config() -> Result<Value, String> {
    Ok(config::get_config_redacted())
}

#[tauri::command]
pub fn update_config(patch: Value) -> Result<Value, String> {
    config::update_config(patch)
}

#[tauri::command]
pub fn get_desktop_state() -> Result<Value, String> {
    Ok(config::get_desktop_state())
}

/// Pick (or accept a dropped) project dir: persist + chdir + re-point the
/// chat engine. The engine stays on its thread — chat.js does the chdir
/// through the runtime so the tool jail follows.
#[tauri::command]
pub fn set_project_dir(engine: State<'_, EngineHandle>, path: String) -> Result<Value, String> {
    let meta = std::fs::metadata(&path).map_err(|e| format!("cannot open {path}: {e}"))?;
    if !meta.is_dir() {
        return Err(format!("{path} is not a directory"));
    }
    config::set_project_dir(&path)?;
    let path_js = serde_json::to_string(&path).unwrap_or_else(|_| "\"\"".into());
    let script = format!(
        "__desktop_reply(JSON.stringify((sofuu.chat && sofuu.chat.setProject) ? sofuu.chat.setProject({path_js}) : {{ok:false,error:'chat engine unavailable'}}))"
    );
    let reply = engine.query(script, QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

/// Native folder picker (NSOpenPanel via the dialog plugin). Returns the
/// chosen path or null when cancelled.
#[tauri::command]
pub async fn pick_project_dir(window: tauri::Window) -> Result<Value, String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = std::sync::mpsc::channel::<Option<String>>();
    window
        .dialog()
        .file()
        .set_title("Choose a project folder")
        .pick_folder(move |selection| {
            let path = selection.map(|s| s.to_string());
            let _ = tx.send(path);
        });
    match rx.recv_timeout(Duration::from_secs(300)) {
        Ok(Some(path)) => Ok(json!(path)),
        Ok(None) => Ok(Value::Null),
        Err(_) => Err("folder picker timed out".into()),
    }
}

/// Native multi-file picker for the composer's attach button. The chosen
/// paths come back so the UI can splice them in as @file mentions (chat.js
/// expands those into the prompt, jailed to the project root).
#[tauri::command]
pub async fn pick_files(window: tauri::Window) -> Result<Value, String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = std::sync::mpsc::channel::<Option<Vec<String>>>();
    window
        .dialog()
        .file()
        .set_title("Attach files")
        .pick_files(move |selection| {
            let paths = selection
                .map(|items| items.into_iter().map(|s| s.to_string()).collect::<Vec<_>>());
            let _ = tx.send(paths);
        });
    match rx.recv_timeout(Duration::from_secs(300)) {
        Ok(Some(paths)) => Ok(json!(paths)),
        Ok(None) => Ok(Value::Null),
        Err(_) => Err("file picker timed out".into()),
    }
}

// ── Keychain ──────────────────────────────────────────────────────

#[tauri::command]
pub fn keychain_has(provider: String) -> Result<Value, String> {
    Ok(json!({ "has_key": keychain::get_key(&provider).is_some() }))
}

#[tauri::command]
pub fn keychain_set(provider: String, key: String) -> Result<Value, String> {
    keychain::set_key(&provider, &key)?;
    // Live-update the running engine's provider key as well.
    sofuu_core::embed_config::set_api_key(&provider, key);
    Ok(json!({ "ok": true }))
}

#[tauri::command]
pub fn keychain_delete(provider: String) -> Result<Value, String> {
    keychain::delete_key(&provider)?;
    Ok(json!({ "ok": true }))
}
