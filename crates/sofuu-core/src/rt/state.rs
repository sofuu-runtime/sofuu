// rt/state.rs — `sofuu.state`: the transactional operational-state store
// (PLAN-AUTONOMOUS-REACTIVE-RUNTIME Phase 1).
//
// Operational state is NOT semantic memory. Cooldown stamps, cursors,
// watermarks and rate windows are exact values that must survive a restart,
// never vectors that may decay; QTSQ/CMA stays reserved for recall. This
// module is the small, boring half: one JSON document per namespace under
// `<project>/.sofuu/state/` (or `SOFUU_STATE_ROOT`), committed with the same
// tmp -> fsync -> rename discipline the rest of the runtime uses.
//
// Design constraints (the plan's, verbatim):
//   * 100% domain-agnostic: keys are opaque strings, values opaque JSON.
//   * Isolated strictly inside the configured state root — a namespace that
//     contains a path separator or `..` is refused, never sanitised into a
//     different file.
//   * Single-writer atomic flush. In-process handles for one namespace share
//     one store, so two drivers cannot clobber each other's counters.
//   * `allowAfterCooldown` is check-and-stamp in ONE operation: the caller
//     cannot race a read against a later write.
//   * `0600` on Unix; the store may hold operational secrets (webhook
//     cursors are not public).
//
// JS surface (src/js/state.js):
//   sofuu.state.open(namespace?) -> StateStore
//   get / set({ttlMs}) / delete / keys(prefix)
//   allowAfterCooldown(key, cooldownMs, {dryRun})
//   resetCooldown(key)
//   incrementWindow(key, windowMs, maxAllowed)
//
// Errors are structured (`RULES.md §3.5`): every host function that fails
// throws a JS error whose message begins with `E_STATE_IO:`.

#![allow(clippy::missing_safety_doc)]

use std::collections::{BTreeMap, HashMap};
use std::ffi::CString;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use sofuu_ffi::bridge::{
    js_new_bool, js_new_string, js_to_string, register_global_fn, JSContext, JSValue,
    JSValueConst,
};
use sofuu_ffi::qjs;

const STATE_SCHEMA: &str = "sofuu-state@1";
const MAX_NAMESPACE_LEN: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateEntry {
    value: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct WindowEntry {
    start: i64,
    count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateDoc {
    schema: String,
    namespace: String,
    updated_at: i64,
    #[serde(default)]
    entries: BTreeMap<String, StateEntry>,
    /// key -> epoch-ms of the last allowed stamp
    #[serde(default)]
    cooldowns: BTreeMap<String, i64>,
    /// key -> rolling window
    #[serde(default)]
    windows: BTreeMap<String, WindowEntry>,
}

impl StateDoc {
    fn empty(namespace: &str) -> Self {
        Self {
            schema: STATE_SCHEMA.to_string(),
            namespace: namespace.to_string(),
            updated_at: now_ms(),
            entries: BTreeMap::new(),
            cooldowns: BTreeMap::new(),
            windows: BTreeMap::new(),
        }
    }
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Validate a namespace: it is a FILE NAME, never a path. Rejecting instead
/// of sanitising is deliberate — `open('../secrets')` must fail loudly, not
/// silently read/write some other file.
pub fn validate_namespace(ns: &str) -> Result<(), String> {
    if ns.is_empty() || ns.len() > MAX_NAMESPACE_LEN {
        return Err(format!(
            "E_STATE_IO: namespace must be 1..={MAX_NAMESPACE_LEN} characters"
        ));
    }
    let mut chars = ns.chars();
    let first = chars.next().ok_or_else(|| "E_STATE_IO: empty namespace".to_string())?;
    if !first.is_ascii_alphanumeric() {
        return Err(
            "E_STATE_IO: namespace must start with an ASCII letter or digit".to_string(),
        );
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-') {
        return Err(
            "E_STATE_IO: namespace may contain only ASCII letters, digits, '.', '_' and '-'"
                .to_string(),
        );
    }
    if ns == "." || ns == ".." || ns.contains("..") {
        return Err("E_STATE_IO: namespace must not contain '..'".to_string());
    }
    Ok(())
}

/// The default root: `$SOFUU_STATE_ROOT` when set, else `<cwd>/.sofuu/state`.
pub fn default_root() -> PathBuf {
    if let Some(raw) = std::env::var_os("SOFUU_STATE_ROOT") {
        let p = PathBuf::from(raw);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".sofuu")
        .join("state")
}

fn create_dir(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(path)
            .map_err(|e| format!("E_STATE_IO: cannot create {}: {e}", path.display()))
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(path)
            .map_err(|e| format!("E_STATE_IO: cannot create {}: {e}", path.display()))
    }
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts
        .open(path)
        .map_err(|e| format!("E_STATE_IO: cannot stage {}: {e}", path.display()))?;
    f.write_all(bytes)
        .map_err(|e| format!("E_STATE_IO: cannot write {}: {e}", path.display()))?;
    Ok(())
}

/// One namespace's document.
#[derive(Debug)]
pub struct StateStore {
    pub path: PathBuf,
    pub namespace: String,
    doc: StateDoc,
}

impl StateStore {
    /// Open (or create) the namespace under `root`. A corrupt document is a
    /// hard error: silently starting empty would resurrect every cooldown
    /// and re-fire alerts the operator already received.
    pub fn open(root: &Path, namespace: &str) -> Result<Self, String> {
        validate_namespace(namespace)?;
        create_dir(root)?;
        let path = root.join(format!("{namespace}.json"));
        let doc = if path.exists() {
            let raw = std::fs::read(&path)
                .map_err(|e| format!("E_STATE_IO: cannot read {}: {e}", path.display()))?;
            let mut parsed: StateDoc = serde_json::from_slice(&raw).map_err(|e| {
                format!(
                    "E_STATE_IO: corrupt state file {}: {e}",
                    path.display()
                )
            })?;
            if parsed.schema != STATE_SCHEMA {
                return Err(format!(
                    "E_STATE_IO: {} has schema '{}', expected '{STATE_SCHEMA}'",
                    path.display(),
                    parsed.schema
                ));
            }
            parsed.namespace = namespace.to_string();
            parsed
        } else {
            StateDoc::empty(namespace)
        };
        let mut store = Self {
            path,
            namespace: namespace.to_string(),
            doc,
        };
        if store.sweep_expired() {
            store.persist()?;
        }
        Ok(store)
    }

    fn persist(&mut self) -> Result<(), String> {
        self.doc.updated_at = now_ms();
        let json = serde_json::to_vec(&self.doc)
            .map_err(|e| format!("E_STATE_IO: cannot serialize state: {e}"))?;
        let tmp = self.path.with_extension("json.tmp");
        write_private(&tmp, &json)?;
        if !crate::rt::wal::publish_paths(&tmp, &self.path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!(
                "E_STATE_IO: cannot commit state file {}",
                self.path.display()
            ));
        }
        Ok(())
    }

    /// Drop expired TTL entries. Returns true when the document changed.
    fn sweep_expired(&mut self) -> bool {
        let now = now_ms();
        let before = self.doc.entries.len();
        self.doc
            .entries
            .retain(|_, e| e.expires_at.map(|t| t > now).unwrap_or(true));
        before != self.doc.entries.len()
    }

    pub fn get(&mut self, key: &str) -> Result<Option<serde_json::Value>, String> {
        let expired = match self.doc.entries.get(key) {
            None => return Ok(None),
            Some(e) => e.expires_at.map(|t| t <= now_ms()).unwrap_or(false),
        };
        if expired {
            self.doc.entries.remove(key);
            self.persist()?;
            return Ok(None);
        }
        Ok(self.doc.entries.get(key).map(|e| e.value.clone()))
    }

    pub fn set(
        &mut self,
        key: &str,
        value: serde_json::Value,
        ttl_ms: Option<i64>,
    ) -> Result<bool, String> {
        if key.is_empty() {
            return Err("E_STATE_IO: key must be a non-empty string".to_string());
        }
        if let Some(ttl) = ttl_ms {
            if ttl < 0 {
                return Err("E_STATE_IO: ttlMs must be >= 0".to_string());
            }
        }
        let expires_at = ttl_ms.filter(|t| *t > 0).map(|t| now_ms().saturating_add(t));
        self.doc
            .entries
            .insert(key.to_string(), StateEntry { value, expires_at });
        self.persist()?;
        Ok(true)
    }

    pub fn delete(&mut self, key: &str) -> Result<bool, String> {
        let existed = self.doc.entries.remove(key).is_some()
            | self.doc.cooldowns.remove(key).is_some()
            | self.doc.windows.remove(key).is_some();
        if existed {
            self.persist()?;
        }
        Ok(existed)
    }

    pub fn keys(&mut self, prefix: Option<&str>) -> Result<Vec<String>, String> {
        if self.sweep_expired() {
            self.persist()?;
        }
        let pre = prefix.unwrap_or("");
        let mut out: Vec<String> = self
            .doc
            .entries
            .keys()
            .filter(|k| k.starts_with(pre))
            .cloned()
            .collect();
        out.sort();
        Ok(out)
    }

    /// Atomic check-and-stamp. `dry_run` answers the question without taking
    /// the slot, so a caller can probe ("would this fire?") without spending
    /// the cooldown.
    pub fn allow_after_cooldown(
        &mut self,
        key: &str,
        cooldown_ms: i64,
        dry_run: bool,
    ) -> Result<bool, String> {
        if key.is_empty() {
            return Err("E_STATE_IO: key must be a non-empty string".to_string());
        }
        let cooldown = cooldown_ms.max(0);
        let now = now_ms();
        let allowed = match self.doc.cooldowns.get(key) {
            None => true,
            Some(last) => now.saturating_sub(*last) >= cooldown,
        };
        if allowed && !dry_run {
            self.doc.cooldowns.insert(key.to_string(), now);
            self.persist()?;
        }
        Ok(allowed)
    }

    pub fn reset_cooldown(&mut self, key: &str) -> Result<bool, String> {
        if self.doc.cooldowns.remove(key).is_some() {
            self.persist()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// A rolling window: `max_allowed` successful increments per `window_ms`.
    /// The count saturates at the cap (the sixth attempt when the cap is
    /// five reports `{count:5, allowed:false}`), so a hot loop cannot grow
    /// the document without bound.
    pub fn increment_window(
        &mut self,
        key: &str,
        window_ms: i64,
        max_allowed: Option<u64>,
    ) -> Result<(u64, bool), String> {
        if key.is_empty() {
            return Err("E_STATE_IO: key must be a non-empty string".to_string());
        }
        if window_ms <= 0 {
            return Err("E_STATE_IO: windowMs must be > 0".to_string());
        }
        let now = now_ms();
        let entry = self
            .doc
            .windows
            .entry(key.to_string())
            .or_insert(WindowEntry { start: now, count: 0 });
        if now.saturating_sub(entry.start) >= window_ms {
            entry.start = now;
            entry.count = 0;
        }
        let allowed = match max_allowed {
            None => true,
            Some(max) => entry.count < max,
        };
        if allowed {
            entry.count = entry.count.saturating_add(1);
        }
        let count = entry.count;
        self.persist()?;
        Ok((count, allowed))
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Handle registry
 * ══════════════════════════════════════════════════════════════════════ */

struct SharedStore {
    root: PathBuf,
    store: StateStore,
}

static STORES: OnceLock<Mutex<HashMap<PathBuf, SharedStore>>> = OnceLock::new();
static HANDLES: OnceLock<Mutex<HashMap<u64, PathBuf>>> = OnceLock::new();
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn stores() -> &'static Mutex<HashMap<PathBuf, SharedStore>> {
    STORES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn handles() -> &'static Mutex<HashMap<u64, PathBuf>> {
    HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn state_err(ctx: *mut JSContext, msg: String) -> JSValue {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    // SAFETY: ctx is the live context; c is valid for the call.
    unsafe { qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), c.as_ptr()) }
}

unsafe fn arg_string(ctx: *mut JSContext, argv: *const JSValueConst, i: c_int) -> Option<String> {
    // SAFETY: argv has at least i+1 values when the caller checked argc.
    unsafe { js_to_string(ctx, *argv.add(i as usize)) }
}

unsafe fn arg_i64(ctx: *mut JSContext, argv: *const JSValueConst, i: c_int) -> i64 {
    let mut out: i64 = 0;
    // SAFETY: argv has at least i+1 values when the caller checked argc.
    unsafe { qjs::JS_ToInt64(ctx, &mut out, *argv.add(i as usize)) };
    out
}

/// Resolve a handle to its root path, rejecting unknown/stale handles.
fn handle_root(handle: u64) -> Option<PathBuf> {
    handles().lock().ok()?.get(&handle).cloned()
}

/// Open/create the shared store for a (root, namespace) pair and return a
/// fresh handle. Handles for the same pair share one document.
fn open_shared(root: PathBuf, namespace: &str) -> Result<u64, String> {
    validate_namespace(namespace)?;
    let key = root.join(format!("{namespace}.json"));
    {
        let mut map = stores()
            .lock()
            .map_err(|_| "E_STATE_IO: state registry is unavailable".to_string())?;
        if !map.contains_key(&key) {
            let store = StateStore::open(&root, namespace)?;
            map.insert(
                key.clone(),
                SharedStore {
                    root: root.clone(),
                    store,
                },
            );
        }
    }
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    handles()
        .lock()
        .map_err(|_| "E_STATE_IO: state registry is unavailable".to_string())?
        .insert(handle, key);
    Ok(handle)
}

fn with_store<F, T>(handle: u64, f: F) -> Result<T, String>
where
    F: FnOnce(&mut StateStore) -> Result<T, String>,
{
    let key = handle_root(handle).ok_or_else(|| "E_STATE_IO: invalid state handle".to_string())?;
    let mut map = stores()
        .lock()
        .map_err(|_| "E_STATE_IO: state registry is unavailable".to_string())?;
    let shared = map
        .get_mut(&key)
        .ok_or_else(|| "E_STATE_IO: state store is not open".to_string())?;
    let _ = &shared.root;
    f(&mut shared.store)
}

/* ══════════════════════════════════════════════════════════════════════
 * QuickJS host functions
 * ══════════════════════════════════════════════════════════════════════ */

unsafe extern "C" fn js_state_open(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let namespace = if argc >= 1 {
        unsafe { arg_string(ctx, argv, 0) }.unwrap_or_else(|| "default".to_string())
    } else {
        "default".to_string()
    };
    let root = if argc >= 2 {
        unsafe { arg_string(ctx, argv, 1) }
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(default_root)
    } else {
        default_root()
    };
    match open_shared(root, &namespace) {
        Ok(handle) => unsafe { qjs::sofuu_js_new_int64(ctx, handle as i64) },
        Err(e) => state_err(ctx, e),
    }
}

unsafe extern "C" fn js_state_get(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return state_err(ctx, "E_STATE_IO: get requires (handle, key)".to_string());
    }
    let handle = unsafe { arg_i64(ctx, argv, 0) } as u64;
    let Some(key) = (unsafe { arg_string(ctx, argv, 1) }) else {
        return state_err(ctx, "E_STATE_IO: key must be a string".to_string());
    };
    match with_store(handle, |s| s.get(&key)) {
        Ok(Some(v)) => match serde_json::to_string(&v) {
            Ok(json) => {
                let c = CString::new(json).unwrap_or_default();
                unsafe { js_new_string(ctx, c.to_str().unwrap_or("null")) }
            }
            Err(e) => state_err(ctx, format!("E_STATE_IO: cannot serialize value: {e}")),
        },
        Ok(None) => unsafe { qjs::sofuu_js_null() },
        Err(e) => state_err(ctx, e),
    }
}

unsafe extern "C" fn js_state_set(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 3 {
        return state_err(ctx, "E_STATE_IO: set requires (handle, key, json)".to_string());
    }
    let handle = unsafe { arg_i64(ctx, argv, 0) } as u64;
    let Some(key) = (unsafe { arg_string(ctx, argv, 1) }) else {
        return state_err(ctx, "E_STATE_IO: key must be a string".to_string());
    };
    let Some(json) = (unsafe { arg_string(ctx, argv, 2) }) else {
        return state_err(ctx, "E_STATE_IO: value must be JSON-serializable".to_string());
    };
    let value: serde_json::Value = match serde_json::from_str(&json) {
        Ok(v) => v,
        Err(e) => return state_err(ctx, format!("E_STATE_IO: value is not valid JSON: {e}")),
    };
    let mut ttl: Option<i64> = None;
    if argc >= 4 {
        let raw = unsafe { qjs::JS_ToBool(ctx, *argv.add(3)) };
        let ms = unsafe { arg_i64(ctx, argv, 3) };
        // -1 / undefined / false-style "no TTL": the driver passes 0 for none.
        if raw != 0 && ms > 0 {
            ttl = Some(ms);
        }
    }
    match with_store(handle, |s| s.set(&key, value, ttl)) {
        Ok(ok) => unsafe { js_new_bool(ctx, ok) },
        Err(e) => state_err(ctx, e),
    }
}

unsafe extern "C" fn js_state_delete(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return state_err(ctx, "E_STATE_IO: delete requires (handle, key)".to_string());
    }
    let handle = unsafe { arg_i64(ctx, argv, 0) } as u64;
    let Some(key) = (unsafe { arg_string(ctx, argv, 1) }) else {
        return state_err(ctx, "E_STATE_IO: key must be a string".to_string());
    };
    match with_store(handle, |s| s.delete(&key)) {
        Ok(ok) => unsafe { js_new_bool(ctx, ok) },
        Err(e) => state_err(ctx, e),
    }
}

unsafe extern "C" fn js_state_keys(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return state_err(ctx, "E_STATE_IO: keys requires (handle, prefix?)".to_string());
    }
    let handle = unsafe { arg_i64(ctx, argv, 0) } as u64;
    let prefix = if argc >= 2 {
        unsafe { arg_string(ctx, argv, 1) }
    } else {
        None
    };
    match with_store(handle, |s| s.keys(prefix.as_deref())) {
        Ok(keys) => match serde_json::to_string(&keys) {
            Ok(json) => {
                let c = CString::new(json).unwrap_or_default();
                unsafe { js_new_string(ctx, c.to_str().unwrap_or("[]")) }
            }
            Err(e) => state_err(ctx, format!("E_STATE_IO: cannot serialize keys: {e}")),
        },
        Err(e) => state_err(ctx, e),
    }
}

unsafe extern "C" fn js_state_cooldown(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 3 {
        return state_err(
            ctx,
            "E_STATE_IO: allowAfterCooldown requires (handle, key, cooldownMs)".to_string(),
        );
    }
    let handle = unsafe { arg_i64(ctx, argv, 0) } as u64;
    let Some(key) = (unsafe { arg_string(ctx, argv, 1) }) else {
        return state_err(ctx, "E_STATE_IO: key must be a string".to_string());
    };
    let cooldown = unsafe { arg_i64(ctx, argv, 2) };
    let dry = argc >= 4 && unsafe { qjs::JS_ToBool(ctx, *argv.add(3)) } != 0;
    match with_store(handle, |s| s.allow_after_cooldown(&key, cooldown, dry)) {
        Ok(ok) => unsafe { js_new_bool(ctx, ok) },
        Err(e) => state_err(ctx, e),
    }
}

unsafe extern "C" fn js_state_reset_cooldown(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return state_err(
            ctx,
            "E_STATE_IO: resetCooldown requires (handle, key)".to_string(),
        );
    }
    let handle = unsafe { arg_i64(ctx, argv, 0) } as u64;
    let Some(key) = (unsafe { arg_string(ctx, argv, 1) }) else {
        return state_err(ctx, "E_STATE_IO: key must be a string".to_string());
    };
    match with_store(handle, |s| s.reset_cooldown(&key)) {
        Ok(ok) => unsafe { js_new_bool(ctx, ok) },
        Err(e) => state_err(ctx, e),
    }
}

unsafe extern "C" fn js_state_window(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 3 {
        return state_err(
            ctx,
            "E_STATE_IO: incrementWindow requires (handle, key, windowMs, maxAllowed?)".to_string(),
        );
    }
    let handle = unsafe { arg_i64(ctx, argv, 0) } as u64;
    let Some(key) = (unsafe { arg_string(ctx, argv, 1) }) else {
        return state_err(ctx, "E_STATE_IO: key must be a string".to_string());
    };
    let window = unsafe { arg_i64(ctx, argv, 2) };
    // The driver passes -1 (or omits it) for "no cap": JS has one numeric
    // spelling, so the host does not need a second type test.
    let max = if argc >= 4 {
        let raw = unsafe { arg_i64(ctx, argv, 3) };
        if raw >= 0 { Some(raw as u64) } else { None }
    } else {
        None
    };
    match with_store(handle, |s| s.increment_window(&key, window, max)) {
        Ok((count, allowed)) => {
            let json = format!(
                "{{\"count\":{},\"allowed\":{}}}",
                count,
                if allowed { "true" } else { "false" }
            );
            let c = CString::new(json).unwrap_or_default();
            unsafe { js_new_string(ctx, c.to_str().unwrap_or("{}")) }
        }
        Err(e) => state_err(ctx, e),
    }
}

unsafe extern "C" fn js_state_close(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return unsafe { js_new_bool(ctx, false) };
    }
    let handle = unsafe { arg_i64(ctx, argv, 0) } as u64;
    let removed = handles()
        .lock()
        .ok()
        .and_then(|mut h| h.remove(&handle))
        .is_some();
    unsafe { js_new_bool(ctx, removed) }
}

/// Register `__sofuu_state_*`. Called from `engine_register_builtins`
/// immediately after `sofuu.storage`, whose commit discipline it shares.
pub unsafe fn mod_state_register(ctx: *mut JSContext) {
    register_global_fn(ctx, "__sofuu_state_open", js_state_open);
    register_global_fn(ctx, "__sofuu_state_get", js_state_get);
    register_global_fn(ctx, "__sofuu_state_set", js_state_set);
    register_global_fn(ctx, "__sofuu_state_delete", js_state_delete);
    register_global_fn(ctx, "__sofuu_state_keys", js_state_keys);
    register_global_fn(ctx, "__sofuu_state_cooldown", js_state_cooldown);
    register_global_fn(ctx, "__sofuu_state_reset_cooldown", js_state_reset_cooldown);
    register_global_fn(ctx, "__sofuu_state_window", js_state_window);
    register_global_fn(ctx, "__sofuu_state_close", js_state_close);
}

/* ══════════════════════════════════════════════════════════════════════
 * Unit tests
 * ══════════════════════════════════════════════════════════════════════ */

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_root(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("sofuu-state-{tag}-{}-{n}", std::process::id()));
        p
    }

    #[test]
    fn ttl_evicts_on_read_and_open() {
        let root = tmp_root("ttl");
        {
            let mut s = StateStore::open(&root, "app").unwrap();
            s.set("short", serde_json::json!({"a": 1}), Some(400)).unwrap();
            s.set("forever", serde_json::json!("kept"), None).unwrap();
            assert_eq!(s.get("short").unwrap(), Some(serde_json::json!({"a": 1})));
        }
        std::thread::sleep(std::time::Duration::from_millis(450));
        {
            let mut s = StateStore::open(&root, "app").unwrap();
            // open() swept the expired entry before the read.
            assert_eq!(s.get("short").unwrap(), None);
            assert_eq!(s.get("forever").unwrap(), Some(serde_json::json!("kept")));
            assert_eq!(s.keys(None).unwrap(), vec!["forever".to_string()]);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cooldown_is_atomic_check_and_stamp() {
        let root = tmp_root("cooldown");
        let mut s = StateStore::open(&root, "alerts").unwrap();
        assert!(s.allow_after_cooldown("btc", 800, false).unwrap());
        assert!(!s.allow_after_cooldown("btc", 800, false).unwrap());
        // dry-run must answer without taking the slot
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!s.allow_after_cooldown("btc", 800, true).unwrap());
        assert!(!s.allow_after_cooldown("btc", 800, false).unwrap());
        std::thread::sleep(std::time::Duration::from_millis(800));
        assert!(s.allow_after_cooldown("btc", 800, false).unwrap());
        assert!(s.reset_cooldown("btc").unwrap());
        assert!(s.allow_after_cooldown("btc", 50_000, false).unwrap());
        drop(s);
        let mut again = StateStore::open(&root, "alerts").unwrap();
        assert!(!again.allow_after_cooldown("btc", 50_000, false).unwrap());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn window_caps_and_rolls_over() {
        let root = tmp_root("window");
        let mut s = StateStore::open(&root, "budget").unwrap();
        assert_eq!(s.increment_window("hour", 900, Some(2)).unwrap(), (1, true));
        assert_eq!(s.increment_window("hour", 900, Some(2)).unwrap(), (2, true));
        // the third attempt is refused and the count saturates
        assert_eq!(s.increment_window("hour", 900, Some(2)).unwrap(), (2, false));
        assert_eq!(s.increment_window("hour", 900, Some(2)).unwrap(), (2, false));
        std::thread::sleep(std::time::Duration::from_millis(950));
        assert_eq!(s.increment_window("hour", 900, Some(2)).unwrap(), (1, true));
        // no cap counts everything
        assert_eq!(s.increment_window("all", 60_000, None).unwrap(), (1, true));
        assert_eq!(s.increment_window("all", 60_000, None).unwrap(), (2, true));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn crash_safe_reload_roundtrips_values() {
        let root = tmp_root("reload");
        {
            let mut s = StateStore::open(&root, "app").unwrap();
            s.set("cursor", serde_json::json!({"offset": 42, "name": "x"}), None)
                .unwrap();
            s.set("flag", serde_json::json!(true), None).unwrap();
        }
        let mut s = StateStore::open(&root, "app").unwrap();
        assert_eq!(
            s.get("cursor").unwrap(),
            Some(serde_json::json!({"offset": 42, "name": "x"}))
        );
        assert_eq!(s.get("flag").unwrap(), Some(serde_json::json!(true)));
        assert_eq!(s.get("missing").unwrap(), None);
        assert!(s.delete("flag").unwrap());
        assert!(!s.delete("flag").unwrap());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn namespace_is_a_filename_not_a_path() {
        for bad in ["", ".", "..", "../escape", "a/b", "a\\b", ".hidden", "x".repeat(129).as_str()] {
            assert!(validate_namespace(bad).is_err(), "{bad:?} must be refused");
        }
        for good in ["default", "app.v1", "agent_1", "a-b", "0"] {
            assert!(validate_namespace(good).is_ok(), "{good:?} must be accepted");
        }
    }

    #[test]
    fn corrupt_document_fails_loudly() {
        let root = tmp_root("corrupt");
        create_dir(&root).unwrap();
        std::fs::write(root.join("app.json"), b"{not json").unwrap();
        let err = StateStore::open(&root, "app").unwrap_err();
        assert!(err.starts_with("E_STATE_IO:"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn state_file_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let root = tmp_root("perm");
        let mut s = StateStore::open(&root, "app").unwrap();
        s.set("k", serde_json::json!(1), None).unwrap();
        let mode = std::fs::metadata(root.join("app.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "state file must be 0600");
        let _ = std::fs::remove_dir_all(&root);
    }
}
