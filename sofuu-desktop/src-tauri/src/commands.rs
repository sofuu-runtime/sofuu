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
    config::project_dir()
        .unwrap_or_else(|| sofuu_core::embed_config::home_dir().unwrap_or_else(|| ".".into()))
}

/// Frontend crash/error forwarding: the webview's console is invisible in
/// release builds — window.onerror (installed by an initialization script)
/// routes here so JS failures reach stderr. Never blocks the UI.
#[tauri::command]
pub fn frontend_log(msg: String) -> Result<Value, String> {
    eprintln!("[frontend] {msg}");
    Ok(json!({ "ok": true }))
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
pub async fn list_sessions(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    // The registry is qtsq-encrypted now (project store, no plaintext) —
    // the ENGINE reads it (sofuu.chat.sessions). The Rust plaintext-json
    // mirror stays as a fallback for pre-layout projects (and the old
    // TUI-written registry) so the sidebar never empties spuriously.
    let script = "(__desktop_reply(JSON.stringify( \
         (sofuu.chat && sofuu.chat.sessions) ? sofuu.chat.sessions() : {sessions: []} \
       )))";
    // CONTRACT: the frontend expects a BARE ARRAY of SessionInfo (the old
    // plaintext-mirror shape). The engine reply is {sessions:[…]} — strip
    // the envelope, never leak it (a leaked object makes TopBar's
    // sessions.map crash → React unmounts → blank window).
    if let Ok(reply) = engine.query(script.to_string(), QUERY_TIMEOUT) {
        if let Ok(v) = serde_json::from_str::<Value>(&reply) {
            if let Some(arr) = v.get("sessions").and_then(|s| s.as_array()) {
                return Ok(json!(arr));
            }
        }
    }
    let sessions = sessions::list_sessions(std::path::Path::new(&project_dir()));
    serde_json::to_value(sessions).map_err(|e| e.to_string())
}

/// Turn history for a session — served by chat.js through the engine (the
/// .qtsq files are encrypted; only the runtime can read them).
#[tauri::command]
pub async fn session_turns(engine: State<'_, EngineHandle>, id: String) -> Result<Value, String> {
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
pub async fn new_session(engine: State<'_, EngineHandle>) -> Result<Value, String> {
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
pub async fn resume_session(engine: State<'_, EngineHandle>, id: String) -> Result<Value, String> {
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
pub async fn set_permissions(engine: State<'_, EngineHandle>, profile: String) -> Result<Value, String> {
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
pub async fn chat_state(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script =
        "__desktop_reply(JSON.stringify((sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {error:'chat engine unavailable'}))";
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

/// Usage report for Settings › Usage — served by chat.js through the
/// engine (the per-turn usage records live inside the encrypted .sofuu
/// session store; only the runtime can read them). Returns {ok, totals,
/// models, days, sessions} or {error}.
#[tauri::command]
pub async fn usage_report(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script =
        "__desktop_reply(JSON.stringify((sofuu.chat && sofuu.chat.usage) ? sofuu.chat.usage() : {error:'chat engine unavailable'}))";
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn compact(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script =
        "(function(){ if (sofuu.chat && sofuu.chat.compact) { Promise.resolve(sofuu.chat.compact()).then(function(r){ __desktop_reply(JSON.stringify(r||{ok:true})); }, function(e){ __desktop_reply(JSON.stringify({error:String(e&&e.message||e)})); }); } else { __desktop_reply(JSON.stringify({error:'chat engine unavailable'})); } })()";
    let reply = engine.query(script.to_string(), Duration::from_secs(120))?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

// ── Parity surface: tools / agents / brain / ml / context ──────────
// These mirror CLI slash commands (/tools /agents /remember /why /ml
// /context) as whitelisted engine queries — the WebView never evals
// arbitrary JS, only these fixed scripts.

#[tauri::command]
pub async fn list_tools(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script = "__desktop_reply(JSON.stringify((function(){ try { \
        var defs = []; \
        if (sofuu.tools && sofuu.tools.TOOLS) { for (var k in sofuu.tools.TOOLS) defs.push(k); } \
        var mcp = []; \
        try { if (sofuu.chat && sofuu.chat.state) { var st = sofuu.chat.state(); if (st && st.mcpTools) mcp = st.mcpTools; } } catch(e){} \
        return {ok:true, builtin:defs, mcp:mcp}; \
      } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })()))";
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_agents(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script = "__desktop_reply(JSON.stringify((function(){ try { \
        var list = (sofuu.agent && sofuu.agent.list) ? sofuu.agent.list() : []; \
        return {ok:true, agents:list}; \
      } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })()))";
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn brain_remember(engine: State<'_, EngineHandle>, fact: String) -> Result<Value, String> {
    let fact_js = serde_json::to_string(&fact).unwrap_or_else(|_| "\"\"".into());
    let script = format!(
        "__desktop_reply(JSON.stringify((function(){{ try {{ \
          if (sofuu.memory && sofuu.memory.remember) {{ sofuu.memory.remember({fact_js}); return {{ok:true}}; }} \
          return {{ok:false, error:'brain unavailable (enable /brain)'}}; \
        }} catch(e){{ return {{ok:false, error:String(e&&e.message||e)}}; }} }})()))"
    );
    let reply = engine.query(script, QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn brain_why(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script = "__desktop_reply(JSON.stringify((function(){ try { \
        var st = (sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {}; \
        return {ok:true, lastRecall:(st && st.lastRecall) || null, recall:(st && st.lastRecallHits) || null}; \
      } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })()))";
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn ml_info(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script = "__desktop_reply(JSON.stringify((function(){ try { \
        var info = (sofuu.ml && sofuu.ml.info) ? sofuu.ml.info() : null; \
        return {ok:true, info:info}; \
      } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })()))";
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn context_dump(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script = "__desktop_reply(JSON.stringify((function(){ try { \
        var st = (sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {}; \
        return {ok:true, state:st}; \
      } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })()))";
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    serde_json::from_str(&reply).map_err(|e| e.to_string())
}

// ── Config ────────────────────────────────────────────────────────

/// Live model list for one provider — mirrors the CLI picker's
/// wizardFetchModels: openai-compatible endpoints serve GET {root}/models,
/// local serves {root}/api/tags, Anthropic has no public list. The fetch
/// runs INSIDE the engine (its JS fetch is the runtime's, and the API key
/// never enters the WebView): the key is the config file's if CLI-written,
/// else the Keychain. Returns {ok, models: [..]} or {ok:false, error, models:[]}
/// — the UI then falls back to the provider's saved model + manual entry.
/// Strip trailing path suffixes repeatedly — stored endpoints carry them
/// doubled (e.g. ".../v1/chat/completions/chat/completions"), so a single
/// strip leaves the API root unreachable and /models 404s.
fn strip_until_core(mut url: String, suffixes: &[&str]) -> String {
    loop {
        let before = url.len();
        for s in suffixes {
            if url.ends_with(s) {
                url.truncate(url.len() - s.len());
                break;
            }
        }
        if url.len() == before {
            break;
        }
    }
    url
}

/// True when the stored entry is on the Anthropic wire (no public model list).
fn is_anthropic(endpoint: &str, profile: &str) -> bool {
    let prof = profile.trim().to_lowercase();
    let low = endpoint.to_lowercase();
    prof == "anthropic" || (prof.is_empty() && (low.contains("anthropic") || low.contains("claude")))
}

/// True when the stored entry is a local Ollama endpoint.
fn is_local(endpoint: &str, profile: &str) -> bool {
    let prof = profile.trim().to_lowercase();
    let low = endpoint.to_lowercase();
    prof == "local"
        || (prof.is_empty()
            && (low.contains("11434") || low.contains("localhost") || low.contains("127.0.0.1")))
}

/// The /models (or /api/tags for local) URL for a provider entry. OpenAI-
/// compatible entries store the FULL completion URL, so the suffix is
/// stripped down to the API root first.
fn models_url(endpoint: &str, profile: &str) -> Result<String, String> {
    if is_anthropic(endpoint, profile) {
        // Caller handles Anthropic — there is no list API to build a URL for.
        return Err("anthropic".to_string());
    }
    if is_local(endpoint, profile) {
        let root = if endpoint.trim().is_empty() {
            "http://127.0.0.1:11434".to_string()
        } else {
            endpoint.trim().trim_end_matches('/').to_string()
        };
        let root = strip_until_core(root, &["/api/chat", "/api/tags"]);
        return Ok(format!("{root}/api/tags"));
    }
    let endpoint = endpoint.trim().trim_end_matches('/').to_string();
    if endpoint.is_empty() {
        return Err("provider has no endpoint".to_string());
    }
    let root = strip_until_core(endpoint, &["/chat/completions", "/completions"]);
    Ok(format!("{root}/models"))
}

/// Build the engine-side script that fetches a provider's model list. The
/// URL and key are baked in as JSON-escaped literals; a Bearer header is
/// sent only when a key exists. The script answers through __desktop_reply
/// within 12s (the runtime's fetch is the native one — real keys never
/// enter the WebView; the app never waits on this at click time — the
/// cache is refreshed in the background).
fn models_fetch_script(url: &str, key: &str) -> String {
    let url_js = serde_json::to_string(url).unwrap_or_else(|_| "\"\"".into());
    let key_js = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".into());
    // The script returns the RAW listing JSON alongside the parsed model
    // names: the engine-side ingest (sofuu.ml.alloc.ingestListing) harvests
    // each model's real caps (context window, max output) from it — the
    // same field vocabulary check lives in Rust, so any provider's
    // spelling works. Provider-agnostic by construction.
    format!(
        "(function () {{ \
           var url = {url_js}; var key = {key_js}; \
           var headers = key ? {{ 'Authorization': 'Bearer ' + key }} : {{}}; \
           var done = function (r) {{ try {{ __desktop_reply(JSON.stringify(r)); }} catch (e) {{}} }}; \
           var timer = setTimeout(function () {{ done({{ok:false, error:'timeout', models:[]}}); }}, 12000); \
           try {{ \
             fetch(url, {{ headers: headers }}).then(function (res) {{ \
               if (!res.ok) {{ clearTimeout(timer); done({{ok:false, error:'HTTP ' + res.status, models:[]}}); return; }} \
               return res.json(); \
             }}).then(function (j) {{ \
               clearTimeout(timer); \
               var raw = (j && j.data && Array.isArray(j.data)) ? j.data \
                        : (j && j.models && Array.isArray(j.models)) ? j.models : []; \
               var arr = raw.map(function (m) {{ return (m && (m.id || m.name)) || m; }}) \
                            .filter(function (x) {{ return x && String(x).trim(); }}); \
               done({{ok:true, models: arr, raw: JSON.stringify(j)}}); \
             }}).catch(function (e) {{ \
               clearTimeout(timer); \
               done({{ok:false, error:String(e && e.message || e), models:[]}}); \
             }}); \
           }} catch (e) {{ \
             clearTimeout(timer); \
             done({{ok:false, error:String(e && e.message || e), models:[]}}); \
           }} \
         }})()"
    )
}

// ── Model-list cache (prefetched in the background; clicks never fetch) ──

#[derive(Clone)]
struct ModelCacheEntry {
    models: Vec<String>,
    ok: bool,
    error: Option<String>,
    note: Option<String>,
    at: u64,
}

static MODEL_CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, ModelCacheEntry>>> =
    std::sync::OnceLock::new();
static MODELS_REFRESHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn model_cache() -> &'static std::sync::Mutex<std::collections::HashMap<String, ModelCacheEntry>> {
    MODEL_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// SSRF guard for the engine-side model-list fetch: http/https only, and
/// the endpoint host must not be a loopback/private/link-local/unspecified
/// address (no localhost, no intranet probes — the shared config's
/// provider endpoints are an attack surface). Hostnames are resolved and
/// every address is checked before the request goes out.
fn validate_models_url(url: &str) -> Result<(), String> {
    let Some(rest) = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
    else {
        return Err("unsupported URL scheme (http/https only)".into());
    };
    let auth = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    let host: &str = if auth.starts_with('[') {
        auth[1..].split(']').next().unwrap_or(auth)
    } else {
        auth.split(':').next().unwrap_or(auth)
    };
    let host = host.to_lowercase();
    if host.is_empty() {
        return Err("empty host".into());
    }

    fn blocked(ip: std::net::IpAddr) -> bool {
        use std::net::IpAddr::*;
        match ip {
            V4(v4) => {
                let o = v4.octets();
                v4.is_loopback()
                    || v4.is_unspecified()
                    || v4.is_multicast()
                    || o[0] == 0
                    || o[0] == 10
                    || (o[0] == 127)
                    || (o[0] == 172 && (16..=31).contains(&o[1]))
                    || (o[0] == 192 && o[1] == 168)
                    || (o[0] == 169 && o[1] == 254)
            }
            V6(v6) => {
                let s = v6.segments();
                v6.is_loopback()
                    || v6.is_unspecified()
                    || v6.is_multicast()
                    || (s[0] & 0xfe00) == 0xfc00 /* fc00::/7 ULA */
                    || (s[0] & 0xffc0) == 0xfe80 /* fe80::/10 link-local */
            }
        }
    }

    if host == "localhost" || host.ends_with(".local") || host.ends_with(".lan") {
        return Err("local hosts are blocked (ssrf guard)".into());
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return if blocked(ip) {
            Err("private/local addresses are blocked (ssrf guard)".into())
        } else {
            Ok(())
        };
    }
    // Hostname: reject if ANY resolved address is loopback/private etc.
    match std::net::ToSocketAddrs::to_socket_addrs(&(host.as_str(), 443)) {
        Ok(addrs) => {
            for addr in addrs {
                if blocked(addr.ip()) {
                    return Err("private/local addresses are blocked (ssrf guard)".into());
                }
            }
            Ok(())
        }
        Err(_) => Err("host did not resolve".into()),
    }
}

/// Fetch one provider's model list through the engine and cache it.
fn fetch_provider_models(engine: &EngineHandle, name: &str) -> ModelCacheEntry {
    let entry = |models: Vec<String>, ok: bool, error: Option<&str>, note: Option<&str>| ModelCacheEntry {
        models,
        ok,
        error: error.map(|s| s.to_string()),
        note: note.map(|s| s.to_string()),
        at: now_secs(),
    };
    let Some((endpoint, profile, file_key)) = config::provider_entry(name) else {
        return entry(vec![], false, Some("unknown provider"), None);
    };
    let key = if file_key.is_empty() {
        keychain::get_key(name).unwrap_or_default()
    } else {
        file_key
    };

    if is_anthropic(&endpoint, &profile) {
        // No public model list API — the picker shows the saved model + manual entry.
        return entry(vec![], true, None, Some("anthropic"));
    }
    let url = match models_url(&endpoint, &profile) {
        Ok(url) => url,
        Err(e) => return entry(vec![], false, Some(&e), None),
    };
    if let Err(e) = validate_models_url(&url) {
        return entry(vec![], false, Some(&e), None);
    }

    let script = models_fetch_script(&url, &key);
    match engine.query(script, Duration::from_secs(20)) {
        Ok(reply) => serde_json::from_str::<Value>(&reply)
            .map(|v| {
                // Harvest caps: the listing the endpoint just returned is
                // ground truth for what THIS endpoint's models accept
                // (context window, max output). Ingested in the engine's
                // discovered store, keyed by this API root — the wire
                // builders and the alloc ladder clamp every later request
                // against it. Ingest runs on a thread-local registry of
                // the desktop's own; the ENGINE's copy is fed by its own
                // discovery (chat.js init) — both key identically, and
                // this side also persists to the shared ~/.sofuu/ml file
                // the engine loads at boot.
                if v["ok"] == true {
                    if let Some(raw) = v["raw"].as_str() {
                        let _ = sofuu_core::rt::model_caps_discovered::ingest_listing_json(&url, raw);
                    }
                }
                entry(
                    v["models"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|m| m.as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_default(),
                    v["ok"] == true,
                    v["error"].as_str(),
                    v["note"].as_str(),
                )
            })
            .unwrap_or_else(|_| entry(vec![], false, Some("unparseable reply"), None)),
        Err(e) => entry(vec![], false, Some(&e), None),
    }
}

/// Cache-only read: the model list is prefetched in the background, so a
/// click never waits on the network. Returns `cached: false` when the
/// prefetch has not populated this provider yet.
#[tauri::command]
pub fn list_models(name: String) -> Value {
    let cache = model_cache().lock().unwrap();
    match cache.get(&name) {
        Some(e) => json!({
            "models": e.models,
            "ok": e.ok,
            "error": e.error,
            "note": e.note,
            "cached": true,
            "at": e.at,
        }),
        None => json!({ "models": [], "ok": false, "error": "no cache yet", "cached": false }),
    }
}

/// Fetch + cache the model lists for ALL configured providers. Runs in the
/// background (async command → tokio worker; the UI never blocks). Called
/// at app start and on a timer; overlapping runs are dropped.
#[tauri::command]
pub async fn refresh_model_cache(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    if MODELS_REFRESHING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return Ok(json!({ "ok": false, "error": "refresh already running" }));
    }
    let names: Vec<String> = {
        let cfg = config::get_config_redacted();
        cfg["providers"]
            .as_array()
            .map(|a| a.iter().filter_map(|p| p["name"].as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default()
    };
    for name in &names {
        let entry = fetch_provider_models(&engine, name);
        model_cache().lock().unwrap().insert(name.clone(), entry);
    }
    MODELS_REFRESHING.store(false, std::sync::atomic::Ordering::SeqCst);
    Ok(json!({ "ok": true, "providers": names.len() }))
}

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
/// Delete a chat: the engine detaches it (if active) and drops the
/// registry entry; the Rust side removes the session folder (project
/// store: sessions/<sid>/) and any legacy flat file. Registry write
/// happens BEFORE the rm so a failed detach never orphans the entry.
#[tauri::command]
pub async fn delete_session(engine: State<'_, EngineHandle>, id: String) -> Result<Value, String> {
    let id_js = serde_json::to_string(&id).unwrap_or_else(|_| "\"\"".into());
    let script = format!(
        "__desktop_reply(JSON.stringify( \
           (sofuu.chat && sofuu.chat.deleteSession) ? sofuu.chat.deleteSession({id_js}) : {{ok:false, error:'chat engine unavailable'}} \
         ))"
    );
    let reply = engine.query(script, QUERY_TIMEOUT)?;
    let parsed: Value = serde_json::from_str(&reply).unwrap_or(json!({ "ok": false }));
    if parsed.get("ok") != Some(&json!(true)) {
        return Ok(parsed);
    }
    // Remove the files: project-store folder + legacy flat layout.
    let base = sessions::sessions_dir(&sessions::mesh_root_for(std::path::Path::new(&project_dir())));
    let _ = std::fs::remove_dir_all(base.join(&id));
    let _ = std::fs::remove_file(base.join(format!("{id}.qtsq")));
    Ok(json!({ "ok": true, "wasActive": parsed.get("wasActive") == Some(&json!(true)) }))
}

/// Remove every chat in the CURRENT workspace: the engine empties the
/// registry (qtsq + legacy plaintext) and returns the ids; the Rust side
/// removes each session's folder (project store: sessions/<sid>/) and any
/// legacy flat file. Registry write happens BEFORE the rm so a failed
/// detach never orphans an entry.
#[tauri::command]
pub async fn clear_all_sessions(engine: State<'_, EngineHandle>) -> Result<Value, String> {
    let script =
        "__desktop_reply(JSON.stringify((sofuu.chat && sofuu.chat.clearSessions) ? sofuu.chat.clearSessions() : {ok:false, error:'chat engine unavailable'}))";
    let reply = engine.query(script.to_string(), QUERY_TIMEOUT)?;
    let parsed: Value = serde_json::from_str(&reply).unwrap_or(json!({ "ok": false }));
    if parsed.get("ok") != Some(&json!(true)) {
        return Ok(parsed);
    }
    let base = sessions::sessions_dir(&sessions::mesh_root_for(std::path::Path::new(&project_dir())));
    if let Some(ids) = parsed.get("ids").and_then(|v| v.as_array()) {
        for id in ids {
            if let Some(sid) = id.as_str() {
                // Path-safety: session ids are engine-generated, but never
                // let a registry entry turn into a traversal.
                if sid.is_empty() || sid.contains('/') || sid.contains('\\') || sid.starts_with('.') {
                    continue;
                }
                let _ = std::fs::remove_dir_all(base.join(sid));
                let _ = std::fs::remove_file(base.join(format!("{sid}.qtsq")));
            }
        }
    }
    Ok(json!({ "ok": true, "cleared": parsed.get("cleared").and_then(|v| v.as_u64()).unwrap_or(0) }))
}

#[tauri::command]
pub async fn set_project_dir(engine: State<'_, EngineHandle>, path: String) -> Result<Value, String> {
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
    // The running engine holds a copy of the key in memory — clear it too,
    // or the provider keeps working until relaunch after a delete.
    sofuu_core::embed_config::clear_api_key(&provider);
    Ok(json!({ "ok": true }))
}

/// Delete a provider: drop the config entry (the active one is refused —
/// switch first), its Keychain key, and any cached model list for it.
#[tauri::command]
pub fn remove_provider(provider: String) -> Result<Value, String> {
    let cfg = config::remove_provider(&provider)?;
    let _ = keychain::delete_key(&provider);
    sofuu_core::embed_config::clear_api_key(&provider);
    model_cache().lock().unwrap().remove(&provider);
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn strips_doubled_suffixes_to_the_api_root() {
        // The user's own config had a doubled path — the exact case that
        // 404'd model listing for the active provider.
        assert_eq!(
            strip_until_core(
                "https://api.orcarouter.ai/v1/chat/completions/chat/completions".into(),
                &["/chat/completions", "/completions"]
            ),
            "https://api.orcarouter.ai/v1"
        );
        assert_eq!(
            strip_until_core("https://api.openai.com/v1/chat/completions".into(), &["/chat/completions", "/completions"]),
            "https://api.openai.com/v1"
        );
        // Plain base URL: untouched (no completion suffix to strip).
        assert_eq!(
            strip_until_core("https://api.tokenrouter.com/v1".into(), &["/chat/completions", "/completions"]),
            "https://api.tokenrouter.com/v1"
        );
        // Doubled local chat suffix.
        assert_eq!(
            strip_until_core("http://127.0.0.1:11434/api/chat/api/chat".into(), &["/api/chat", "/api/tags"]),
            "http://127.0.0.1:11434"
        );
    }

    #[test]
    fn validates_models_url_ssrf() {
        // Loopback / private / link-local / unspecified / multicast hostnames
        // and IPs must be rejected before the engine fetch; non-hosts too.
        for bad in [
            "http://localhost:11434/api/tags",
            "http://127.0.0.1:11434/api/tags",
            "http://10.0.0.1/v1/models",
            "http://192.168.1.10/v1/models",
            "http://172.16.0.1/v1/models",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/api/tags",
            "http://[fd00::1]/v1/models",
            "http://0.0.0.0/v1/models",
            "file:///etc/passwd",
            "http://host.local/v1/models",
            "",
        ] {
            assert!(validate_models_url(bad).is_err(), "{bad} must be blocked");
        }
        // Public IP literals pass without DNS.
        assert!(validate_models_url("https://8.8.8.8/v1/models").is_ok());
        // Public hostnames: check passes when DNS resolves only to public
        // addresses (openrouter.ai resolves to a CDN).
        assert!(validate_models_url("https://openrouter.ai/api/v1/models").is_ok());
    }

    /// Headless repro of the model picker's engine fetch: boot the same
    /// runtime + host natives the desktop engine uses, run the exact query
    /// script against the user's real orca-ai endpoint, and require a
    /// non-empty live model list. Read the provider data straight from the
    /// config file so the test never races the config-root global.
    #[test]
    fn engine_fetch_lists_real_models() {
        let home = std::env::var("HOME").expect("HOME");
        let raw = std::fs::read_to_string(format!("{home}/.sofuu/config.json")).expect("config.json");
        let cfg: Value = serde_json::from_str(&raw).expect("config json");
        let orca = cfg["providers"]
            .as_array()
            .and_then(|a| a.iter().find(|p| p["name"] == "orca-ai"))
            .expect("orca-ai provider in config")
            .clone();
        let endpoint = orca["endpoint"].as_str().unwrap_or("").to_string();
        let profile = orca["profile"].as_str().unwrap_or("").to_string();
        let key = orca["api_key"].as_str().unwrap_or("").to_string();

        assert!(!is_anthropic(&endpoint, &profile), "orca-ai is an openai-compat provider");
        let url = models_url(&endpoint, &profile).expect("models url");
        assert_eq!(url, "https://api.orcarouter.ai/v1/models");

        let Some(rt) = sofuu_ffi::SofuuRuntime::init() else {
            panic!("runtime init failed");
        };
        let ctx = rt.engine_ctx() as *mut sofuu_ffi::bridge::JSContext;
        // SAFETY: ctx is the live engine context on this thread.
        unsafe { crate::host::register_host_natives(ctx) };

        let rx = crate::host::arm_reply();
        let script = models_fetch_script(&url, &key);
        let rc = rt.eval_string(&script, "<query-test>");
        assert_eq!(rc, 0, "engine eval failed");
        let reply = rx
            .recv_timeout(Duration::from_secs(30))
            .expect("no reply from engine fetch");
        let v: Value = serde_json::from_str(&reply).expect("reply is JSON");
        assert_eq!(v["ok"], true, "fetch failed: {v}");
        let models = v["models"].as_array().expect("models array");
        assert!(models.len() > 0, "models came back empty: {v}");
    }
}
