// model_caps_discovered.rs — caps the ENDPOINT itself publishes.
//
// WHY THIS EXISTS
// ───────────────
// The static registry (model_caps.rs) keys capabilities by model-family
// NAME and can never cover every gateway's catalogue: "unknown model"
// there is the norm, not the exception. But the endpoint the request is
// actually going to already knows the truth — the same /models listing
// that fills the desktop's model picker carries each model's real
// context window and max output. Throwing that away is what made
// "config does not respect the selected model" a whole failure class.
//
// DESIGN CONTRACTS
// ────────────────
// * PROVIDER-AGNOSTIC: nothing keys on a provider NAME. Entries are
//   keyed by (api_root, model_id) — the normalized base URL of the
//   endpoint plus the model id the request carries. Any OpenAI-wire,
//   Anthropic-wire or local endpoint that publishes a model list works,
//   whatever it calls itself.
// * FIELD VOCABULARY, NOT PROVIDERS: ingestion accepts a union of field
//   spellings seen in the wild (context_length / context_window / ctx,
//   max_completion_tokens / max_tokens / max_output_tokens,
//   top_provider.{context_length,max_completion_tokens}). An endpoint
//   that publishes nothing contributes nothing — error learning
//   (ml/alloc/policy.rs) remains the backstop.
// * EVIDENCE, NOT CONFIG: this store is read-only ground truth for the
//   alloc resolve ladder (learned-from-400s > discovered > registry >
//   default). User config is clamped to it; it never overwrites user
//   numbers silently — the ladder reports the clamp.
// * DISK TRUTH: persisted to ~/.sofuu/ml/model_caps.json with a per-entry
//   timestamp and TTL, so cold boots start warm and stale entries age
//   out. Schema-checked on load; a corrupt file is ignored, never fatal.
//
// THREADING: lives behind a mutex, like every runtime-global store; the
// wire builders run on the engine thread, the desktop's refresh runs
// wherever the engine query runs. Both go through the same lock.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// How long a discovered entry stays authoritative before a refresh is
/// expected (the desktop refreshes every 5 minutes; the CLI refreshes
//  lazily at TTL). Entries are KEPT past TTL — a stale truth still beats
//  a registry guess — but resolve() reports them as stale so callers can
/// re-discover.
pub const DISCOVERED_TTL_SECS: u64 = 7 * 24 * 3600;

/// One (api_root, model) discovery. `ctx_window` / `max_output` are 0
/// when the listing did not carry that number — partial evidence is
/// fine, each side resolves independently.
#[derive(Clone, Copy, Debug)]
pub struct DiscoveredCaps {
    pub ctx_window: i32,
    pub max_output: i32,
    /// Whether the listing also told us the model supports tool calls
    /// (input/output token shapes are the fields alloc needs; the
    /// supported-parameters field is kept verbatim for later gates).
    pub supports_tools: Option<bool>,
    pub at: u64,
}

#[derive(Default)]
struct Store {
    /// (api_root, model) → caps. Lowercase model ids: listings are
    /// case-stable but the request may not be.
    map: HashMap<(String, String), DiscoveredCaps>,
}

static STORE: LazyLock<Mutex<Store>> = LazyLock::new(|| Mutex::new(Store::default()));

/// Normalize an endpoint URL to its API root for keying: lowercase
/// scheme/host, strip the path down to the version segment when present
/// (/v1, /v2 …), drop query/fragment. Two spellings of the same endpoint
/// (trailing /chat/completions, doubled suffixes, trailing slashes) MUST
/// key together — the desktop already strips suffixes before listing,
/// but requests may carry the full endpoint URL.
pub fn normalize_api_root(endpoint: &str) -> String {
    let e = endpoint.trim();
    if e.is_empty() {
        return String::new();
    }
    let (scheme, rest) = if let Some(r) = e.strip_prefix("https://") {
        ("https://", r)
    } else if let Some(r) = e.strip_prefix("http://") {
        ("http://", r)
    } else {
        ("", e)
    };
    // Cut at the first query/fragment, then at any known API-path suffix —
    // completion paths (what requests carry) AND the listing path (what
    // harvesters pass) must all key to the same root.
    let mut path = rest
        .split(['?', '#'])
        .next()
        .unwrap_or("")
        .to_string();
    for suffix in ["/chat/completions", "/completions", "/messages", "/models"] {
        let s = suffix.to_string();
        while path.ends_with(&s) {
            let cut = path.len() - s.len();
            path.truncate(cut);
        }
    }
    while path.ends_with('/') {
        path.truncate(path.len() - 1);
    }
    // Lowercase the host portion only (paths are case-sensitive).
    let (host, tail) = match path.split_once('/') {
        Some((h, t)) => (h, format!("/{t}")),
        None => (path.as_str(), String::new()),
    };
    format!("{scheme}{}{}", host.to_lowercase(), tail)
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Pull one integer out of a serde value, tolerating the shapes listings
/// actually use (i64/u64, JSON float like 8192.0, numeric string).
fn int_field(v: &serde_json::Value) -> Option<i64> {
    match v {
        serde_json::Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        serde_json::Value::String(s) => s.trim().replace(',', "").parse::<i64>().ok(),
        _ => None,
    }
}

/// Ingest ONE model entry from a listing, for the API root the listing
/// was fetched from. Accepts the union vocabulary; unknown shapes are
/// silently skipped (a listing that carries no caps must not fail the
/// ingest of the ones that do). Returns true when this entry actually
/// carried at least one usable number.
pub fn ingest_one(api_root: &str, model_id: &str, entry: &serde_json::Value) -> bool {
    let root = normalize_api_root(api_root);
    let id = model_id.trim().to_lowercase();
    if root.is_empty() || id.is_empty() || !entry.is_object() {
        return false;
    }
    let obj = entry.as_object().unwrap();

    // Context window: context_length | context_window | ctx_window | ctx
    let ctx = ["context_length", "context_window", "ctx_window", "ctx"]
        .iter()
        .find_map(|k| obj.get(*k).and_then(int_field))
        .filter(|v| *v > 0);

    // Max output — direct fields first, then nested top_provider/
    // per_request_limits objects (OpenRouter-style listings nest them).
    let max_out = ["max_completion_tokens", "max_tokens", "max_output_tokens"]
        .iter()
        .find_map(|k| obj.get(*k).and_then(int_field))
        .filter(|v| *v > 0)
        .or_else(|| {
            ["top_provider", "per_request_limits"]
                .iter()
                .find_map(|k| obj.get(*k).and_then(|sub| {
                    sub.as_object().map(|s| {
                        ["max_completion_tokens", "max_tokens", "max_output_tokens"]
                            .iter()
                            .find_map(|k2| s.get(*k2).and_then(int_field))
                    })
                }))
                .flatten()
                .filter(|v| *v > 0)
        });

    // Tool support (optional): supported_parameters containing "tools",
    // or architecture.input_modalities-ish boolean — kept for later
    // gates, never required for the entry to count.
    let supports_tools = obj.get("supported_parameters").and_then(|v| {
        v.as_array().map(|arr| {
            arr.iter()
                .any(|p| p.as_str().is_some_and(|s| s.eq_ignore_ascii_case("tools")))
        })
    });

    if ctx.is_none() && max_out.is_none() {
        return false; // nothing usable — never store an empty entry
    }
    let caps = DiscoveredCaps {
        ctx_window: ctx.unwrap_or(0) as i32,
        max_output: max_out.unwrap_or(0) as i32,
        supports_tools,
        at: now_secs(),
    };
    if let Ok(mut st) = STORE.lock() {
        st.map.insert((root, id), caps);
    }
    true
}

/// Ingest a whole listing ({"data": [...]} | {"models": [...]} | bare
/// array). Returns the count of entries that carried caps. Never fails:
/// a malformed listing ingests nothing.
pub fn ingest_listing(api_root: &str, listing: &serde_json::Value) -> usize {
    let rows = listing
        .as_object()
        .and_then(|o| {
            o.get("data")
                .or_else(|| o.get("models"))
                .and_then(|d| d.as_array())
                .cloned()
                .or_else(|| o.get("models").and_then(|d| d.as_array()).cloned())
        })
        .or_else(|| listing.as_array().cloned())
        .unwrap_or_default();
    let mut n = 0;
    for row in &rows {
        let Some(id) = row
            .as_object()
            .and_then(|o| o.get("id").or_else(|| o.get("name")))
            .and_then(|v| v.as_str())
        else {
            continue;
        };
        if ingest_one(api_root, id, row) {
            n += 1;
        }
    }
    if n > 0 {
        persist();
    }
    n
}

/// Parse a listing JSON string and ingest it (the desktop's fetch script
/// hand this over verbatim). Err on malformed JSON — the caller surfaces
/// the fetch error, not us.
pub fn ingest_listing_json(api_root: &str, listing_json: &str) -> Result<usize, String> {
    let v: serde_json::Value =
        serde_json::from_str(listing_json).map_err(|e| format!("bad listing JSON: {e}"))?;
    Ok(ingest_listing(api_root, &v))
}

/// Look up what the endpoint told us. None when this (root, model) was
/// never discovered. Values may be partial (0 on one side).
pub fn lookup(api_root: &str, model: &str) -> Option<DiscoveredCaps> {
    let root = normalize_api_root(api_root);
    let id = model.trim().to_lowercase();
    if root.is_empty() || id.is_empty() {
        return None;
    }
    STORE.lock().ok().and_then(|st| st.map.get(&(root, id)).copied())
}

/// Cross-root fallback: the caps the SAME model id published on ANY other
/// endpoint. Different gateways host the same underlying models under the
/// same ids (openrouter and a mirror both list "z-ai/glm-5.3"); when the
/// endpoint serving the request publishes no numbers of its own, the
/// model's numbers from another root are far better truth than a config
/// guess — same model, same weights, same window. Weaker than a
/// same-root discovery (a mirror may re-serve a model with different
/// caps), so the ladder only consults it when the same-root side and the
/// registry both know nothing.
///
/// EXACT ID ONLY — no plan-stem inheritance (2026-09-15 fix for the
/// ctx-window over-report: "z-ai/glm-5.3-free" used to inherit the PAID
/// "z-ai/glm-5.3" window 1,310,720 from another root via plan_stem +
/// max_by_key, inflating the meter ~10x for a free tier that serves far
/// less. A plan suffix exists precisely because the variant is served
/// under different limits, so the stem's published window says nothing
/// about the variant. Unknown plan variants now fall to UNKNOWN_WINDOW =
/// 32,768 until the first real 400 teaches the true window; learned-400s
/// still override everything).
pub fn lookup_any_root(model: &str) -> Option<DiscoveredCaps> {
    let id = model.trim().to_lowercase();
    if id.is_empty() {
        return None;
    }
    let st = STORE.lock().ok()?;
    // Exact id across roots (the only sound match: same id, same weights).
    st.map
        .iter()
        .filter(|((_, m), _)| *m == id)
        .map(|(_, caps)| *caps)
        .max_by_key(|c| c.ctx_window)
}

/// Whether a discovered entry is past TTL (still true, but a refresh is
/// due). Unknown entries are conservatively "stale" so callers fetch.
pub fn is_stale(api_root: &str, model: &str) -> bool {
    match lookup(api_root, model) {
        Some(c) => now_secs().saturating_sub(c.at) > DISCOVERED_TTL_SECS,
        None => true,
    }
}

/// Drop everything (tests).
pub fn clear() {
    if let Ok(mut st) = STORE.lock() {
        st.map.clear();
    }
}

/// Test seam: how many entries are held.
pub fn len() -> usize {
    STORE.lock().map(|st| st.map.len()).unwrap_or(0)
}

/* ── Disk truth (~/.sofuu/ml/model_caps.json) ────────────────────────
 * Shape: {"v":1,"entries":{"<root>":{"<model>":[ctx,max,tools?,at]}}}.
 * Loaded lazily at first use; persisted after each ingest. Corrupt or
 * schema-mismatched files are ignored — the store just starts empty. */

fn disk_path() -> Option<PathBuf> {
    let home = crate::embed_config::home_dir()?;
    let mut p = PathBuf::from(home);
    p.push(".sofuu");
    p.push("ml");
    p.push("model_caps.json");
    Some(p)
}

fn persist() {
    /* Tests ingest fixtures against the live process-global store; persisting
     * them would clobber the USER's harvested caps (~590 real models) with
     * example.com fixtures. Tests serialize on ml::TEST_LOCK and clear() at
     * the end, so the in-memory store is their only surface — skipping
     * persist under cfg(test) is complete, nothing real is lost. */
    if cfg!(test) {
        return;
    }
    let Some(path) = disk_path() else { return };
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let Ok(st) = STORE.lock() else { return };
    // Compact row form: [ctx, max, tools(0/1/-1), at].
    let mut roots: HashMap<String, serde_json::Value> = HashMap::new();
    for ((root, model), c) in &st.map {
        let row = serde_json::json!([
            c.ctx_window,
            c.max_output,
            match c.supports_tools { Some(true) => 1, Some(false) => 0, None => -1 },
            c.at
        ]);
        let entry = roots.entry(root.clone()).or_insert_with(|| serde_json::json!({}));
        if let Some(o) = entry.as_object_mut() {
            o.insert(model.clone(), row);
        }
    }
    let doc = serde_json::json!({ "v": 1, "entries": roots });
    // Write-to-temp + rename: a half-written file must never replace a
    // good one (rename is atomic on the same filesystem).
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, serde_json::to_vec(&doc).unwrap_or_default()).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

fn load() {
    /* Same isolation as persist(): a test binary never wants the user's
     * real caps in the store — fixtures would race the first lookup and
     * a real entry could leak between tests. Tests that need disk-truth
     * behavior test persist/load shapes against their own maps. */
    if cfg!(test) {
        return;
    }
    let Some(path) = disk_path() else { return };
    let Ok(bytes) = std::fs::read(&path) else { return };
    // Explicit size guard before parsing: a runaway file is a bug, not
    // a listing.
    if bytes.len() > 4 * 1024 * 1024 {
        return;
    }
    let Ok(doc) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return;
    };
    if doc.get("v").and_then(|v| v.as_i64()) != Some(1) {
        return;
    }
    let Some(entries) = doc.get("entries").and_then(|e| e.as_object()) else {
        return;
    };
    let mut map: HashMap<(String, String), DiscoveredCaps> = HashMap::new();
    for (root, models) in entries {
        let Some(models) = models.as_object() else { continue };
        for (model, row) in models {
            let Some(arr) = row.as_array() else { continue };
            if arr.len() != 4 {
                continue;
            }
            let (Some(ctx), Some(max), tools, Some(at)) = (
                arr[0].as_i64(),
                arr[1].as_i64(),
                arr[2].as_i64(),
                arr[3].as_u64(),
            ) else {
                continue;
            };
            if ctx < 0 || max < 0 {
                continue;
            }
            map.insert(
                (root.clone(), model.to_lowercase()),
                DiscoveredCaps {
                    ctx_window: ctx as i32,
                    max_output: max as i32,
                    supports_tools: match tools {
                        Some(1) => Some(true),
                        Some(0) => Some(false),
                        _ => None,
                    },
                    at,
                },
            );
        }
    }
    if let Ok(mut st) = STORE.lock() {
        if st.map.is_empty() {
            st.map = map;
        }
    }
}

/// Ensure the disk truth is loaded (idempotent; called by lookup paths
/// on first use).
pub fn ensure_loaded() {
    static LOADED: std::sync::Once = std::sync::Once::new();
    LOADED.call_once(load);
}

#[cfg(test)]
mod tests {
    use super::*;

    /* Every test that touches the process-global STORE holds the shared
     * ML test lock for its whole body (ml::TEST_LOCK — see its doc there):
     * cargo runs tests on parallel threads and the alloc-policy and rt/ai
     * wire tests clear/ingest this same store. */
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::ml::TEST_LOCK.lock().unwrap()
    }

    #[test]
    fn normalizes_endpoint_spellings_to_one_root() {
        let a = normalize_api_root("https://api.example.com/v1/chat/completions");
        let b = normalize_api_root("https://api.example.com/v1/");
        let c = normalize_api_root("https://API.example.com/v1/chat/completions");
        let d = normalize_api_root("https://api.example.com/v1/chat/completions/chat/completions");
        assert_eq!(a, "https://api.example.com/v1");
        assert_eq!(a, b, "trailing slash + suffix must key together");
        assert_eq!(a, c, "host case must not split the key");
        assert_eq!(a, d, "doubled suffix must key together");
        assert_eq!(normalize_api_root("https://x.io"), "https://x.io");
        // The LISTING url (what harvesters pass) keys to the same root as
        // the request endpoint.
        assert_eq!(
            normalize_api_root("https://api.example.com/v1/models"),
            "https://api.example.com/v1"
        );
        // Anthropic-style endpoint lands on its version root too.
        assert_eq!(
            normalize_api_root("https://api.example.com/v1/messages"),
            "https://api.example.com/v1"
        );
    }

    #[test]
    fn ingests_union_field_shapes() {
        let _store = lock();
        clear();
        // OpenRouter shape (nested top_provider).
        let or = serde_json::json!({
            "id": "some/model:free",
            "context_length": 65536,
            "top_provider": { "max_completion_tokens": 8192 }
        });
        assert!(ingest_one("https://api.example.com/v1", "some/model:free", &or));
        let c = lookup("https://api.example.com/v1", "Some/Model:FREE").unwrap();
        assert_eq!(c.ctx_window, 65536);
        assert_eq!(c.max_output, 8192);

        // Flat shape with different spellings + numeric string.
        let flat = serde_json::json!({
            "id": "m2",
            "context_window": "131,072",
            "max_tokens": 4096
        });
        assert!(ingest_one("https://api.example.com/v1", "m2", &flat));
        let c2 = lookup("https://api.example.com/v1", "m2").unwrap();
        assert_eq!(c2.ctx_window, 131072, "comma-numeric strings parse");
        assert_eq!(c2.max_output, 4096);

        // A listing with no caps contributes nothing.
        assert!(!ingest_one(
            "https://api.example.com/v1",
            "m3",
            &serde_json::json!({ "id": "m3", "name": "Model Three" })
        ));
        assert!(lookup("https://api.example.com/v1", "m3").is_none());
        clear();
    }

    #[test]
    fn listing_shapes_and_partial_rows() {
        let _store = lock();
        clear();
        // {"models":[...]} shape, one entry with ctx only, one with
        // nothing, one with max only.
        let listing = serde_json::json!({
            "models": [
                { "id": "ctx-only", "context_length": 32000 },
                { "id": "empty" },
                { "name": "max-only", "top_provider": { "max_completion_tokens": 2048 } }
            ]
        });
        assert_eq!(ingest_listing("https://api.example.com/v1", &listing), 2);
        assert_eq!(lookup("https://api.example.com/v1", "ctx-only").unwrap().ctx_window, 32000);
        assert_eq!(lookup("https://api.example.com/v1", "ctx-only").unwrap().max_output, 0);
        assert!(lookup("https://api.example.com/v1", "empty").is_none());
        // "name" works as the id field for listings that use it.
        assert_eq!(lookup("https://api.example.com/v1", "max-only").unwrap().max_output, 2048);
        clear();
    }

    #[test]
    fn different_roots_never_collide() {
        let _store = lock();
        clear();
        let a = serde_json::json!({ "id": "shared-model", "context_length": 8000, "max_tokens": 1000 });
        let b = serde_json::json!({ "id": "shared-model", "context_length": 1000000, "max_tokens": 64000 });
        ingest_one("https://one.example.com/v1", "shared-model", &a);
        ingest_one("https://two.example.com/v1", "shared-model", &b);
        // The same model id behind two gateways keeps two truths.
        assert_eq!(lookup("https://one.example.com/v1", "shared-model").unwrap().ctx_window, 8000);
        assert_eq!(lookup("https://two.example.com/v1", "shared-model").unwrap().ctx_window, 1000000);
        // An endpoint suffix spelling of root one keys together with it.
        assert_eq!(
            lookup("https://one.example.com/v1/chat/completions", "shared-model").unwrap().ctx_window,
            8000
        );
        clear();
    }

    #[test]
    fn cross_root_is_exact_id_only_no_plan_stem_inheritance() {
        let _store = lock();
        clear();
        // The live over-report case (2026-09-15): tokenrouter serves
        // "z-ai/glm-5.3-free" and publishes nothing; openrouter lists the
        // PAID stem "z-ai/glm-5.3" at 1,310,720 and orcarouter at
        // 1,000,000. The free-tier variant must NOT inherit either —
        // a plan suffix exists precisely because the variant is served
        // under different limits.
        let or = serde_json::json!({
            "data": [ { "id": "z-ai/glm-5.3", "context_length": 1310720,
                        "top_provider": { "max_completion_tokens": 131072 } } ]
        });
        ingest_listing("https://openrouter.ai/api/v1", &or);
        let orca = serde_json::json!({
            "data": [ { "id": "z-ai/glm-5.3", "context_length": 1000000,
                        "max_tokens": 128000 } ]
        });
        ingest_listing("https://api.orcarouter.ai/v1", &orca);
        // Exact id match still works (same id, same weights — sound).
        assert_eq!(lookup_any_root("z-ai/glm-5.3").unwrap().ctx_window, 1310720);
        // Plan-suffixed variants resolve to NOTHING (caller falls to
        // UNKNOWN_WINDOW = 32,768 until a real 400 teaches the truth) —
        // not the max (1,310,720), not the min (1,000,000).
        assert!(
            lookup_any_root("z-ai/glm-5.3-free").is_none(),
            "free-tier variant must not inherit the paid stem window"
        );
        assert!(lookup_any_root("z-ai/glm-5.3:batch").is_none());
        assert!(lookup_any_root("z-ai/glm-5.3:free").is_none());
        // A DIFFERENT model is NOT aliased either.
        assert!(lookup_any_root("z-ai/glm-4.5-air").is_none());
        clear();
    }
}
