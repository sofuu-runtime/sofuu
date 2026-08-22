// rt/ai.rs — the AI module (PLAN-RUST-MIGRATION M8).
//
// Port of the deleted `src/modules/mod_ai.c` (2,543 lines), semantics
// verbatim: sofuu.ai.{complete,stream,embed,embedLocal,similarity,dot,l2,
// estimateTokens,listModels,listProviders} over one CURLM multi handle
// bridged to the libuv loop (curl socket/timer callbacks + uv_poll + a
// uv_timer), plus the global `__ai_abort(sid)` stream-cancellation hook.
//
// Also ports the deleted `src/memory/tfidf_embed.c` (73 lines) INLINE as
// sofuu_tfidf_embed() — the embedLocal local embedder (char trigrams →
// MurmurHash3 → L2-normalize) is now pure Rust. The SIMD kernels
// (sofuu_dot_f32 / sofuu_l2_f32 / sofuu_cosine_f32 in src/simd/{neon,avx}.c)
// STAY C and are called over FFI.
//
// C symbols replaced: `mod_ai_register` (engine.c calls it unchanged) +
// the sofuu.ai.* surface.

use std::cell::Cell;
use std::ffi::{CStr, CString, c_int, c_long, c_void};
use std::ptr;

use sofuu_ffi::curl::{self, Curl, CurlM, CurlSlist};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use sofuu_ffi::uv::{self, UvCheck, UvHandle, UvPoll, UvTimer};

use crate::rt::event_loop::sofuu_loop_get;
use crate::rt::promise::{
    sofuu_flush_jobs, sofuu_promise_new, sofuu_promise_reject, sofuu_promise_resolve,
    PromiseHandle,
};

// SIMD kernels stay C (src/simd/simd.h + {neon,avx}.c) — called over FFI.
extern "C" {
    fn sofuu_dot_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    fn sofuu_l2_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    fn sofuu_cosine_f32(a: *const f32, b: *const f32, n: usize) -> f32;
}

/* ------------------------------------------------------------------ */
/* Providers                                                            */
/* ------------------------------------------------------------------ */

#[repr(i32)]
#[derive(Clone, Copy, PartialEq)]
enum Provider {
    OpenAi = 0,
    Anthropic = 1,
    Local = 2,  /* roadmap Track B — real local inference is not built yet */
    Custom = 3, /* user-defined endpoint; wire format comes from `profile` */
}

fn parse_provider(name: Option<&str>) -> Provider {
    match name {
        None => Provider::OpenAi,
        Some("anthropic") => Provider::Anthropic,
        /* Legacy configs may still say "ollama" — treat it as local. */
        Some("ollama") => Provider::Local,
        Some("local") => Provider::Local,
        Some(_) => Provider::Custom, /* any other name = custom provider */
    }
}

/* No provider_default_model(): the caller must always name a model — an
 * absent/empty model is an actionable error at the complete/stream/embed
 * entry points, never a silent substitution. The picker's static
 * suggestion table (listModels below) is a list of SUGGESTIONS, not
 * defaults. */

fn provider_env_key(p: Provider) -> Option<&'static str> {
    match p {
        Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
        Provider::Local => None,
        _ => Some("OPENAI_API_KEY"),
    }
}

/// Name under which embedded hosts register API keys via the capi
/// config (PLAN-HEADLESS H2.4). Consulted after an explicit per-call
/// api_key, before env vars.
fn provider_embed_key_name(p: Provider) -> Option<&'static str> {
    match p {
        Provider::Anthropic => Some("anthropic"),
        Provider::Local => None,
        _ => Some("openai"),
    }
}

/// Key resolution order: explicit per-call key → host-provided embed
/// key → environment variable.
fn resolve_api_key(p: Provider, explicit: Option<&str>) -> Option<String> {
    explicit
        .map(|s| s.to_string())
        .or_else(|| provider_embed_key_name(p).and_then(|n| crate::embed_config::api_key(n)))
        .or_else(|| provider_env_key(p).and_then(|k| std::env::var(k).ok()))
}

fn provider_api_url(p: Provider, _model: Option<&str>) -> String {
    match p {
        Provider::Anthropic => "https://api.anthropic.com/v1/messages".to_string(),
        Provider::Local => "http://localhost:11434/api/chat".to_string(),
        _ => "https://api.openai.com/v1/chat/completions".to_string(),
    }
}

/// For PROVIDER_CUSTOM the wire format (request body + response parsing +
/// auth header) comes from `profile`; everything else uses its own format.
/// Profile NULL/""/unknown → OpenAI-compatible (the common case).
fn effective_provider(cfg: &AiRequestConfig) -> Provider {
    if cfg.provider != Provider::Custom {
        return cfg.provider;
    }
    match cfg.profile.as_deref() {
        Some("anthropic") => Provider::Anthropic,
        Some("local") | Some("ollama") => Provider::Local,
        _ => Provider::OpenAi,
    }
}

/* ------------------------------------------------------------------ */
/* Config Structures                                                    */
/* ------------------------------------------------------------------ */

#[derive(Clone, Default)]
struct AiMessage {
    role: Option<String>,
    content: Option<String>,
    /* Tool-call round-trip: assistant messages carry the raw tool_calls
     * JSON array, tool messages carry the tool_call_id they answer. */
    tool_calls: Option<String>,
    tool_call_id: Option<String>,
}

#[derive(Clone, Default)]
struct AiToolDef {
    name: Option<String>,
    description: Option<String>,
    parameters_json: Option<String>,
}

#[derive(Clone)]
struct AiRequestConfig {
    provider: Provider,
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,        /* full endpoint override; None → built-in */
    profile: Option<String>,         /* PROVIDER_CUSTOM wire format: openai|anthropic|local */
    system_prompt: Option<String>,
    response_format: Option<String>, /* "json" → trigger JSON mode; None → default */
    effort: Option<String>,          /* "low" | "medium" | "high" | "max" | None → omit */
    max_tokens: i32,                 // default: 4096 (for anthropic)
    temperature: f64,                // default: -1.0 (omit)
    top_p: f64,                      // default: -1.0 (omit)
    stream: bool,
    timeout_ms: c_long,              /* 0 = unlimited (default); >0 = ms timeout */
    messages: Vec<AiMessage>,
    tools: Vec<AiToolDef>,
}

impl Default for AiRequestConfig {
    fn default() -> Self {
        Self {
            provider: Provider::OpenAi,
            model: None,
            api_key: None,
            base_url: None,
            profile: None,
            system_prompt: None,
            response_format: None,
            effort: None,
            max_tokens: 0,
            temperature: -1.0,
            top_p: -1.0,
            stream: false,
            timeout_ms: 0,
            messages: Vec::new(),
            tools: Vec::new(),
        }
    }
}

struct AiEmbedConfig {
    provider: Provider,
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>, /* full endpoint override; None → built-in */
    inputs: Vec<String>,
}

/* ------------------------------------------------------------------ */
/* Request body builders (owned Strings — the C returned heap strings)  */
/* ------------------------------------------------------------------ */

/* Simple JSON string escaping — NULL-safe, escapes ALL control characters
 * (< 0x20) plus " \ \n \r \t \b \f and the JS line separators U+2028/U+2029,
 * so provider payloads are always valid JSON and cannot smuggle JSON control
 * or escape sequences. Iterates chars(), never bytes, so non-ASCII text
 * (é, 中文, emoji) round-trips intact. */
fn json_escape(s: Option<&str>) -> String {
    let mut out = String::with_capacity(s.map_or(0, |s| s.len() * 2) + 16);
    for c in s.unwrap_or("").chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Byte index just past the closing brace of the object that starts at
/// `s[0]` (assumed `{`), honoring nested braces/strings. Falls back to
/// the full length when unbalanced.
fn find_obj_end(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.first() != Some(&b'{') {
        return None;
    }
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &c) in b.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract the top-level string value of `"key"` from a JSON object slice
/// ("" when missing or not a string). Used for the tool-call round-trip.
fn json_field(obj: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let mut rest = obj;
    while let Some(at) = rest.find(&needle) {
        let after = &rest[at + needle.len()..];
        let after = after.trim_start();
        let Some(colon_rest) = after.strip_prefix(':') else {
            rest = after;
            continue;
        };
        let colon_rest = colon_rest.trim_start();
        if colon_rest.starts_with('"') {
            let inner = &colon_rest[1..];
            let mut out = String::new();
            let mut esc = false;
            for ch in inner.chars() {
                if esc {
                    out.push(ch);
                    esc = false;
                } else if ch == '\\' {
                    esc = true;
                } else if ch == '"' {
                    return Some(out);
                } else {
                    out.push(ch);
                }
            }
            return Some(out); /* unterminated — return what we have */
        }
        rest = colon_rest;
    }
    None
}

/* Appends "tools":[...] in the provider's specific wire format. */
fn append_tools_openai(body: &mut String, cfg: &AiRequestConfig) {
    if cfg.tools.is_empty() {
        return;
    }
    body.push_str(",\"tools\":[");
    for (i, t) in cfg.tools.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        let d = json_escape(t.description.as_deref());
        let n = t.name.as_deref().unwrap_or("");
        let p = t.parameters_json.as_deref().unwrap_or("{}");
        body.push_str(&format!(
            "{{\"type\":\"function\",\"function\":{{\"name\":\"{}\",\"description\":\"{}\",\"parameters\":{}}}}}",
            n, d, p
        ));
    }
    body.push(']');
}

fn append_tools_anthropic(body: &mut String, cfg: &AiRequestConfig) {
    if cfg.tools.is_empty() {
        return;
    }
    body.push_str(",\"tools\":[");
    let last = cfg.tools.len() - 1;
    for (i, t) in cfg.tools.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        let d = json_escape(t.description.as_deref());
        let n = t.name.as_deref().unwrap_or("");
        let p = t.parameters_json.as_deref().unwrap_or("{}");
        body.push_str(&format!(
            "{{\"name\":\"{}\",\"description\":\"{}\",\"input_schema\":{}",
            n, d, p
        ));
        /* P6 (PLAN-MEMORY-TOKENS): cache marker on the LAST tool — the
         * stable tools+system prefix gets cached across turns. This is a
         * wire-format detail, not a design focus: OpenAI-compatible and
         * local endpoints need nothing (automatic prefix caching). */
        if i == last {
            body.push_str(",\"cache_control\":{\"type\":\"ephemeral\"}");
        }
        body.push('}');
    }
    body.push(']');
}

fn build_openai_body_v2(cfg: &AiRequestConfig) -> String {
    /* model may be NULL when the user passed a non-string — never emit
     * "(null)" or crash; an empty model is rejected by the provider. */
    let mut body = format!("{{\"model\":\"{}\"", cfg.model.as_deref().unwrap_or(""));

    if cfg.temperature >= 0.0 {
        body.push_str(&format!(",\"temperature\":{:.2}", cfg.temperature));
    }
    if cfg.top_p >= 0.0 {
        body.push_str(&format!(",\"top_p\":{:.2}", cfg.top_p));
    }
    if cfg.max_tokens > 0 {
        body.push_str(&format!(",\"max_tokens\":{}", cfg.max_tokens));
    }

    /* JSON structured output — OpenAI/Ollama: response_format field */
    let json_mode = cfg.response_format.as_deref() == Some("json");
    if json_mode {
        body.push_str(",\"response_format\":{\"type\":\"json_object\"}");
    }

    /* Reasoning effort — OpenAI o-series / OpenRouter reasoning models */
    if let Some(effort) = cfg.effort.as_deref() {
        if !effort.is_empty() {
            body.push_str(&format!(",\"reasoning_effort\":\"{}\"", effort));
        }
    }

    body.push_str(&format!(
        ",\"stream\":{},\"messages\":[",
        if cfg.stream { "true" } else { "false" }
    ));

    let mut first = true;
    if let Some(sp) = cfg.system_prompt.as_deref() {
        let es = json_escape(Some(sp));
        body.push_str(&format!("{{\"role\":\"system\",\"content\":\"{}\"}}", es));
        first = false;
    }

    for m in &cfg.messages {
        let es = json_escape(m.content.as_deref());
        if !first {
            body.push(',');
        }
        body.push_str(&format!(
            "{{\"role\":\"{}\",\"content\":\"{}\"",
            m.role.as_deref().unwrap_or(""),
            es
        ));
        /* Assistant tool-call message: splice the tool_calls array. */
        if let Some(tc) = &m.tool_calls {
            if !tc.is_empty() {
                body.push_str(",\"tool_calls\":");
                body.push_str(tc);
            }
        }
        /* Tool result message: the id it answers. */
        if let Some(id) = &m.tool_call_id {
            if !id.is_empty() {
                let eid = json_escape(Some(id));
                body.push_str(&format!(",\"tool_call_id\":\"{}\"", eid));
            }
        }
        body.push('}');
        first = false;
    }

    body.push(']');
    append_tools_openai(&mut body, cfg);
    body.push('}');
    body
}

fn build_anthropic_body_v2(cfg: &AiRequestConfig) -> String {
    let max_tokens = if cfg.max_tokens > 0 { cfg.max_tokens } else { 4096 };
    let mut body = format!(
        "{{\"model\":\"{}\",\"max_tokens\":{},\"stream\":{}",
        cfg.model.as_deref().unwrap_or(""),
        max_tokens,
        if cfg.stream { "true" } else { "false" }
    );

    if cfg.temperature >= 0.0 {
        body.push_str(&format!(",\"temperature\":{:.2}", cfg.temperature));
    }
    if cfg.top_p >= 0.0 {
        body.push_str(&format!(",\"top_p\":{:.2}", cfg.top_p));
    }

    /* Anthropic doesn't have a native JSON mode flag, so we must instruct
     * it via the system prompt. P6: the system rides as a content block
     * with a cache_control marker — the stable-prefix anchor for Anthropic's
     * prompt cache (provider-neutral ordering is done at the agent layer).
     * The agent/chat drivers send the system as messages[0] (the OpenAI
     * shape), so a LEADING system message is hoisted here; explicit
     * opts.system still wins. Mid-array system entries (compaction
     * summaries) keep their historical user-role mapping. */
    let mut hoisted: Option<String> = None;
    let start_idx = if cfg.system_prompt.is_none() {
        match cfg.messages.first() {
            Some(m)
                if m.role.as_deref() == Some("system")
                    && m.tool_calls.is_none()
                    && m.tool_call_id.is_none() =>
            {
                hoisted = m.content.clone();
                1
            }
            _ => 0,
        }
    } else {
        0
    };

    let json_mode = cfg.response_format.as_deref() == Some("json");
    let eff_sys = cfg.system_prompt.clone().or(hoisted);
    if eff_sys.is_some() || json_mode {
        let mut sys_text = json_escape(eff_sys.as_deref());
        if json_mode {
            sys_text.push_str(" Please format your response as valid JSON.");
        }
        body.push_str(&format!(
            ",\"system\":[{{\"type\":\"text\",\"text\":\"{}\",\"cache_control\":{{\"type\":\"ephemeral\"}}}}]",
            sys_text
        ));
    }

    /* Reasoning effort → Anthropic extended thinking.
     * budget_tokens must be < max_tokens, so clamp below it. Also clamp to
     * Anthropic's per-request thinking ceiling (16384 for sonnet/opus) —
     * "max" (32k) would otherwise be rejected by the API. */
    if let Some(effort) = cfg.effort.as_deref() {
        if !effort.is_empty() {
            let mut budget = match effort {
                "low" => 1024,
                "medium" => 4096,
                "high" => 16384,
                "max" => 32768,
                _ => 4096, /* unknown effort → sensible default */
            };
            if budget > 16384 {
                budget = 16384; /* anthropic thinking ceiling */
            }
            if budget >= max_tokens {
                budget = if max_tokens > 1024 { max_tokens / 2 } else { 1024 };
            }
            body.push_str(&format!(
                ",\"thinking\":{{\"type\":\"enabled\",\"budget_tokens\":{}}}",
                budget
            ));
        }
    }

    /* Anthropic requires strictly alternating user/assistant turns. The agent
     * layer can hand us adjacent same-role plain messages (ephemeral context
     * + task on an empty history, or a compaction summary mapped to user
     * followed by the user turn) — merge those into one turn so the wire is
     * valid. Structured messages (tool_calls / tool_call_id) are never
     * merged; they keep their own content blocks. */
    fn eff_role(m: &AiMessage) -> &'static str {
        match m.role.as_deref() {
            Some("user") => "user",
            Some("assistant") => "assistant",
            _ => "user",
        }
    }
    fn is_plain(m: &AiMessage) -> bool {
        m.tool_calls.as_deref().map_or(true, |t| t.is_empty())
            && m.tool_call_id.as_deref().map_or(true, |t| t.is_empty())
    }
    let mut merged: Vec<AiMessage> = Vec::with_capacity(cfg.messages.len());
    for m in &cfg.messages[start_idx..] {
        if is_plain(m) {
            if let Some(prev) = merged.last_mut() {
                if is_plain(prev) && eff_role(prev) == eff_role(m) {
                    let mut joined = prev.content.clone().unwrap_or_default();
                    let add = m.content.clone().unwrap_or_default();
                    if !joined.is_empty() && !add.is_empty() {
                        joined.push_str("\n\n");
                    }
                    joined.push_str(&add);
                    prev.content = Some(joined);
                    continue;
                }
            }
        }
        merged.push(m.clone());
    }

    body.push_str(",\"messages\":[");
    let mut first = true;
    for m in &merged {
        /* Anthropic only supports 'user' or 'assistant'. Default to user if
         * role is missing OR not stringifiable (ToCString returned NULL). */
        let role = eff_role(m);
        let es = json_escape(m.content.as_deref());
        if !first {
            body.push(',');
        }
        /* Tool-call round-trip:
         *  - assistant message with tool_calls → content:[{type:"tool_use",…}]
         *  - tool message (role "tool") → user content:[{type:"tool_result",…}]
         * The JS driver hands us tool_calls as the raw JSON array (each
         * element {id,type,function:{name,arguments}}). */
        if let Some(tc) = &m.tool_calls {
            if !tc.is_empty() {
                body.push_str(&format!("{{\"role\":\"{}\",\"content\":[", role));
                let mut tc_first = true;
                /* Parse the array minimally: {"id":..,"function":{"name":..,"arguments":..}} */
                let mut seg = tc.as_str();
                while let Some(st) = seg.find("\"id\"") {
                    let before = &seg[..st];
                    let obj_start = before.rfind('{');
                    let Some(obj_start) = obj_start else { break };
                    let obj = &seg[obj_start..];
                    let Some(close) = find_obj_end(obj) else { break };
                    let entry = &obj[..close];
                    let id = json_field(entry, "id").unwrap_or_default();
                    let fname = json_field(entry, "name").unwrap_or_default();
                    let fargs = json_field(entry, "arguments").unwrap_or_default();
                    if !tc_first {
                        body.push(',');
                    }
                    body.push_str(&format!(
                        "{{\"type\":\"tool_use\",\"id\":\"{}\",\"name\":\"{}\",\"input\":{}}}",
                        json_escape(Some(&id)),
                        json_escape(Some(&fname)),
                        if fargs.is_empty() { "{}" } else { fargs.as_str() }
                    ));
                    tc_first = false;
                    seg = &seg[st + close..];
                }
                body.push(']');
                body.push('}');
                first = false;
                continue;
            }
        }
        if let Some(tid) = &m.tool_call_id {
            if !tid.is_empty() {
                /* tool result → user message with tool_result block */
                body.push_str(&format!(
                    "{{\"role\":\"user\",\"content\":[{{\"type\":\"tool_result\",\"tool_use_id\":\"{}\",\"content\":\"{}\"}}]}}",
                    json_escape(Some(tid)),
                    es
                ));
                first = false;
                continue;
            }
        }
        body.push_str(&format!("{{\"role\":\"{}\",\"content\":\"{}\"}}", role, es));
        first = false;
    }

    body.push(']');
    append_tools_anthropic(&mut body, cfg);
    body.push('}');
    body
}

fn build_ollama_body_v2(cfg: &AiRequestConfig) -> String {
    /* Ollama OpenAI-compat path — always inject options block so
       num_predict (max output) and num_ctx (context window) are explicit. */
    let json_mode = cfg.response_format.as_deref() == Some("json");

    /* Build with the OpenAI builder first (clears response_format if json) */
    let mut tmp = cfg.clone();
    if json_mode {
        tmp.response_format = None; /* we'll add format:json ourselves */
    }
    let base = build_openai_body_v2(&tmp);

    /* Build options + optional json format patch */
    let patch: String = if cfg.max_tokens > 0 {
        if json_mode {
            format!(",\"options\":{{\"num_predict\":{}}},\"format\":\"json\"}}", cfg.max_tokens)
        } else {
            format!(",\"options\":{{\"num_predict\":{}}}}}", cfg.max_tokens)
        }
    } else if json_mode {
        ",\"format\":\"json\"}".to_string()
    } else {
        /* no options, no json format, just put the closing brace back */
        "}".to_string()
    };

    /* strip trailing } of base, append patch */
    format!("{}{}", &base[..base.len() - 1], patch)
}

fn build_request_body(cfg: &AiRequestConfig) -> String {
    match effective_provider(cfg) {
        Provider::Anthropic => build_anthropic_body_v2(cfg),
        Provider::Local => build_ollama_body_v2(cfg),
        Provider::OpenAi => build_openai_body_v2(cfg),
        _ => build_openai_body_v2(cfg),
    }
}

fn build_openai_embed_body(cfg: &AiEmbedConfig) -> String {
    let mut body = format!(
        "{{\"model\":\"{}\",\"input\":[",
        cfg.model.as_deref().unwrap_or("")
    );
    for (i, input) in cfg.inputs.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        let es = json_escape(Some(input));
        body.push_str(&format!("\"{}\"", es));
    }
    body.push_str("]}");
    body
}

fn build_embed_body(cfg: &AiEmbedConfig) -> String {
    match cfg.provider {
        Provider::Local => build_openai_embed_body(cfg),
        _ => build_openai_embed_body(cfg),
    }
}

unsafe fn build_headers(p: Provider, api_key: Option<&str>) -> *mut CurlSlist {
    let mut h: *mut CurlSlist = ptr::null_mut();
    h = curl::curl_slist_append(h, c"Content-Type: application/json".as_ptr());

    if let Some(key) = api_key {
        /* A CR/LF inside the key would inject extra headers (curl < 7.83).
         * Skip the auth header entirely rather than emit a hostile one. */
        if key.contains('\r') || key.contains('\n') {
            return h;
        }
        if p == Provider::Anthropic {
            let buf = format!("x-api-key: {}", key);
            let c = CString::new(buf).unwrap_or_default();
            h = curl::curl_slist_append(h, c.as_ptr());
            h = curl::curl_slist_append(h, c"anthropic-version: 2023-06-01".as_ptr());
        } else {
            let buf = format!("Authorization: Bearer {}", key);
            let c = CString::new(buf).unwrap_or_default();
            h = curl::curl_slist_append(h, c.as_ptr());
        }
    }
    h
}
/* ------------------------------------------------------------------ */
/* Local TF-IDF embedder (port of src/memory/tfidf_embed.c — deleted)   */
/* ------------------------------------------------------------------ */

/*
 * MurmurHash3 — 32-bit finalizer. Public domain.
 * Maps a trigram (3 chars packed into uint32) → bucket in [0, dim).
 */
fn murmur3_fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/*
 * sofuu_tfidf_embed (port of the C symbol of the same name)
 *
 * Algorithm:
 *   For each character trigram (c[i], c[i+1], c[i+2]):
 *     pack = c[i] | (c[i+1] << 8) | (c[i+2] << 16)
 *     bucket = murmur3(pack) % dim
 *     out_vec[bucket] += 1.0
 *   Then L2-normalize the vector.
 *
 * This is a hashing-trick TF-IDF with char trigrams.
 * It captures subword morphology and is robust for most languages.
 */
fn sofuu_tfidf_embed(text: &[u8], out_vec: &mut [f32], dim: usize) {
    if dim == 0 {
        return;
    }

    /* Zero the output */
    out_vec.fill(0.0);

    let len = text.len();
    if len == 0 {
        return;
    }

    /* Accumulate trigram frequencies */
    for i in 0..len.saturating_sub(2) {
        let pack = text[i] as u32
            | ((text[i + 1] as u32) << 8)
            | ((text[i + 2] as u32) << 16);
        let h = murmur3_fmix32(pack);
        let bucket = (h % dim as u32) as usize;
        out_vec[bucket] += 1.0;
    }

    /* Handle short texts that produce no trigrams */
    if len == 1 {
        let h = murmur3_fmix32(text[0] as u32);
        out_vec[(h % dim as u32) as usize] += 1.0;
    } else if len == 2 {
        let pack = text[0] as u32 | ((text[1] as u32) << 8);
        let h = murmur3_fmix32(pack);
        out_vec[(h % dim as u32) as usize] += 1.0;
    }

    /* L2-normalize to unit vector */
    let mut norm = 0.0f32;
    for &v in out_vec.iter() {
        norm += v * v;
    }
    if norm > 0.0 {
        let inv = 1.0f32 / norm.sqrt();
        for v in out_vec.iter_mut() {
            *v *= inv;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Response helpers (string scans — verbatim port of the C)            */
/* ------------------------------------------------------------------ */

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/* Extracts the value of "\"key\":" as a raw (unescaped) string. */
fn extract_json_field(body: &[u8], key: &str) -> Vec<u8> {
    let needle1 = format!("\"{}\":", key);
    let needle2 = format!("\"{}\" :", key); /* some providers use formatting spaces */
    let Some(start) = find_bytes(body, needle1.as_bytes()).or_else(|| find_bytes(body, needle2.as_bytes())) else {
        return Vec::new(); /* strdup("") */
    };

    let body = &body[start..];
    let Some(colon) = body.iter().position(|&b| b == b':') else {
        return Vec::new();
    };
    let mut p = colon + 1;
    while p < body.len() && (body[p] == b' ' || body[p] == b'\n') {
        p += 1;
    }
    if p >= body.len() || body[p] != b'"' {
        return Vec::new();
    }
    p += 1;

    let mut out: Vec<u8> = Vec::with_capacity(256);
    while p < body.len() && body[p] != b'"' {
        if body[p] == b'\\' {
            p += 1;
            if p >= body.len() {
                break;
            }
            match body[p] {
                b'n' => out.push(b'\n'),
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                other => out.push(other),
            }
            p += 1;
            continue;
        }
        out.push(body[p]);
        p += 1;
    }
    out
}

/* Strips reasoning/thinking blocks from `src` and returns just the answer.
 * Matches the REAL tags used by reasoning models:
 *   <thinking>…</thinking>, <think>…</think>,
 *   <｜thinking｜>…<｜/thinking｜> (deepseek), <reasoning>…</reasoning>.
 * Plain text containing the words "thinking"/"response" is NEVER touched.
 * Thinking text is returned separately (think_out), truncated at
 * think_cap-1 bytes — the C kept a 64KB per-request buffer. */
fn strip_think_tags(src: &[u8], think_cap: usize) -> (Vec<u8>, Vec<u8>) {
    const OPENS: [&[u8]; 4] = [
        b"<thinking>",
        b"<think>",
        b"<|thinking|>",
        b"<reasoning>",
    ];
    const CLOSES: [&[u8]; 4] = [
        b"</thinking>",
        b"</think>",
        b"<|/thinking|>",
        b"</reasoning>",
    ];
    let mut think: Vec<u8> = Vec::new();
    let mut out: Vec<u8> = Vec::with_capacity(src.len());
    let mut p = 0usize;

    while p < src.len() {
        let mut matched = false;
        for (oi, open) in OPENS.iter().enumerate() {
            if src[p..].starts_with(open) {
                let inner = p + open.len();
                let close = CLOSES[oi];
                match find_bytes(&src[inner..], close) {
                    None => {
                        /* Unclosed tag — treat the rest as thinking */
                        let rem = src.len() - inner;
                        if think.len() + rem < think_cap - 1 {
                            think.extend_from_slice(&src[inner..]);
                        }
                        p = src.len();
                        matched = true;
                        break;
                    }
                    Some(rel_end) => {
                        let end = inner + rel_end;
                        let tlen = end - inner;
                        if think.len() + tlen < think_cap - 1 {
                            think.extend_from_slice(&src[inner..end]);
                        }
                        p = end + close.len();
                        /* Skip a leading newline right after the close */
                        while p < src.len() && (src[p] == b'\n' || src[p] == b'\r') {
                            p += 1;
                        }
                        matched = true;
                        break;
                    }
                }
            }
        }
        if matched {
            continue;
        }
        out.push(src[p]);
        p += 1;
    }

    /* Trim leading whitespace from answer */
    let trimmed_start = out
        .iter()
        .position(|&b| b != b'\n' && b != b'\r' && b != b' ')
        .unwrap_or(out.len());
    out.drain(..trimmed_start);
    (out, think)
}

fn extract_text(p: Provider, body: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let out = if p == Provider::Anthropic {
        extract_json_field(body, "text")
    } else {
        extract_json_field(body, "content")
    };
    strip_think_tags(&out, 65536)
}

/* ------------------------------------------------------------------ */
/* Tool-call extractor — returns JS_UNDEFINED when no tool calls found  */
/* Handles OpenAI/Ollama, Anthropic, and Gemini wire formats           */
/* ------------------------------------------------------------------ */

/// # Safety
/// `ctx` live; `body` is the raw JSON response bytes.
unsafe fn extract_tool_calls(ctx: *mut JSContext, p: Provider, body: &[u8]) -> JSValue {
    if p == Provider::Anthropic {
        /* Anthropic: "type":"tool_use" blocks in the content array */
        if find_bytes(body, b"\"tool_use\"").is_none() {
            return qjs::sofuu_js_undefined();
        }
        /* Rewind to find the outer "content" array */
        if find_bytes(body, b"\"content\":").is_none() {
            return qjs::sofuu_js_undefined();
        }
    } else {
        /* OpenAI / local: choices[0].message.tool_calls */
        if find_bytes(body, b"\"tool_calls\":").is_none() {
            return qjs::sofuu_js_undefined();
        }
    }

    /* Use QuickJS's built-in JSON parser so we don't have to hand-roll
       a recursive JSON parser. We parse the entire response body and
       then navigate the resulting JS object tree. */
    let body_c = CString::new(body).unwrap_or_default();
    let parsed = qjs::JS_ParseJSON(ctx, body_c.as_ptr(), body.len(), c"<ai_response>".as_ptr());
    if qjs::is_exception(parsed) {
        qjs::sofuu_js_get_exception(ctx); /* clear the exception */
        return qjs::sofuu_js_undefined();
    }

    let result = qjs::JS_NewArray(ctx);
    let mut ri: u32 = 0;

    if p == Provider::Anthropic {
        let content = qjs::sofuu_js_get_property_str(ctx, parsed, c"content".as_ptr());
        if qjs::JS_IsArray(ctx, content) != 0 {
            let mut clen: u32 = 0;
            let cv = qjs::sofuu_js_get_property_str(ctx, content, c"length".as_ptr());
            qjs::sofuu_js_to_uint32(ctx, &mut clen, cv);
            qjs::sofuu_js_free_value(ctx, cv);
            for i in 0..clen {
                let item = qjs::JS_GetPropertyUint32(ctx, content, i);
                let type_v = qjs::sofuu_js_get_property_str(ctx, item, c"type".as_ptr());
                let ts = qjs::sofuu_js_to_cstring(ctx, type_v);
                if !ts.is_null() && CStr::from_ptr(ts).to_bytes() == b"tool_use" {
                    let tc = qjs::sofuu_js_new_object(ctx);
                    let name = qjs::sofuu_js_get_property_str(ctx, item, c"name".as_ptr());
                    let inp = qjs::sofuu_js_get_property_str(ctx, item, c"input".as_ptr());
                    qjs::sofuu_js_set_property_str(ctx, tc, c"name".as_ptr(), name);
                    qjs::sofuu_js_set_property_str(ctx, tc, c"arguments".as_ptr(), inp);
                    qjs::JS_SetPropertyUint32(ctx, result, ri, tc);
                    ri += 1;
                }
                if !ts.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, ts);
                }
                qjs::sofuu_js_free_value(ctx, type_v);
                qjs::sofuu_js_free_value(ctx, item);
            }
        }
        qjs::sofuu_js_free_value(ctx, content);
    } else {
        /* OpenAI / local (ollama-compatible) */
        let choices = qjs::sofuu_js_get_property_str(ctx, parsed, c"choices".as_ptr());
        if qjs::JS_IsArray(ctx, choices) != 0 {
            let ch0 = qjs::JS_GetPropertyUint32(ctx, choices, 0);
            let message = qjs::sofuu_js_get_property_str(ctx, ch0, c"message".as_ptr());
            let tcs = qjs::sofuu_js_get_property_str(ctx, message, c"tool_calls".as_ptr());
            if qjs::JS_IsArray(ctx, tcs) != 0 {
                let mut tlen: u32 = 0;
                let tv = qjs::sofuu_js_get_property_str(ctx, tcs, c"length".as_ptr());
                qjs::sofuu_js_to_uint32(ctx, &mut tlen, tv);
                qjs::sofuu_js_free_value(ctx, tv);
                for i in 0..tlen {
                    let tc = qjs::JS_GetPropertyUint32(ctx, tcs, i);
                    let fn_v = qjs::sofuu_js_get_property_str(ctx, tc, c"function".as_ptr());
                    let name = qjs::sofuu_js_get_property_str(ctx, fn_v, c"name".as_ptr());
                    let args_str = qjs::sofuu_js_get_property_str(ctx, fn_v, c"arguments".as_ptr());
                    /* arguments is a JSON string — parse it for the caller */
                    let as_ptr = qjs::sofuu_js_to_cstring(ctx, args_str);
                    let args = if as_ptr.is_null() {
                        qjs::sofuu_js_new_object(ctx)
                    } else {
                        let a = qjs::JS_ParseJSON(ctx, as_ptr, CStr::from_ptr(as_ptr).to_bytes().len(), c"<tool_args>".as_ptr());
                        if qjs::is_exception(a) {
                            qjs::sofuu_js_get_exception(ctx);
                            qjs::sofuu_js_new_object(ctx)
                        } else {
                            a
                        }
                    };
                    if !as_ptr.is_null() {
                        qjs::sofuu_js_free_cstring(ctx, as_ptr);
                    }
                    let out_tc = qjs::sofuu_js_new_object(ctx);
                    qjs::sofuu_js_set_property_str(ctx, out_tc, c"name".as_ptr(), name);
                    qjs::sofuu_js_set_property_str(ctx, out_tc, c"arguments".as_ptr(), args);
                    qjs::JS_SetPropertyUint32(ctx, result, ri, out_tc);
                    ri += 1;
                    qjs::sofuu_js_free_value(ctx, args_str);
                    qjs::sofuu_js_free_value(ctx, fn_v);
                    qjs::sofuu_js_free_value(ctx, tc);
                }
            }
            qjs::sofuu_js_free_value(ctx, tcs);
            qjs::sofuu_js_free_value(ctx, message);
            qjs::sofuu_js_free_value(ctx, ch0);
        }
        qjs::sofuu_js_free_value(ctx, choices);
    }

    qjs::sofuu_js_free_value(ctx, parsed);

    if ri == 0 {
        qjs::sofuu_js_free_value(ctx, result);
        return qjs::sofuu_js_undefined();
    }
    result
}

/* ------------------------------------------------------------------ */
/* Embeddings extractor — parses into JS Float32Array(s)               */
/* ------------------------------------------------------------------ */

/// # Safety
/// `ctx` live; `json_arr` a live JS array value of `ctx`.
unsafe fn create_float32_array(ctx: *mut JSContext, json_arr: JSValueConst) -> JSValue {
    if qjs::JS_IsArray(ctx, json_arr) == 0 {
        return qjs::sofuu_js_undefined();
    }
    let mut len: u32 = 0;
    let len_val = qjs::sofuu_js_get_property_str(ctx, json_arr, c"length".as_ptr());
    qjs::sofuu_js_to_uint32(ctx, &mut len, len_val);
    qjs::sofuu_js_free_value(ctx, len_val);

    let global = qjs::sofuu_js_get_global_object(ctx);
    let f32_ctor = qjs::sofuu_js_get_property_str(ctx, global, c"Float32Array".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);

    let len_arg = qjs::sofuu_js_new_int32(ctx, len as i32);
    let f32_arr = qjs::JS_CallConstructor(ctx, f32_ctor, 1, &len_arg);
    qjs::sofuu_js_free_value(ctx, len_arg);
    qjs::sofuu_js_free_value(ctx, f32_ctor);
    if qjs::is_exception(f32_arr) {
        return qjs::sofuu_js_undefined();
    }

    for i in 0..len {
        let el = qjs::JS_GetPropertyUint32(ctx, json_arr, i);
        let mut val: f64 = 0.0;
        qjs::JS_ToFloat64(ctx, &mut val, el);
        qjs::JS_SetPropertyUint32(ctx, f32_arr, i, qjs::sofuu_js_new_float64(ctx, val));
        qjs::sofuu_js_free_value(ctx, el);
    }
    f32_arr
}

/// # Safety
/// `ctx` live; `body` the raw JSON response bytes.
unsafe fn extract_embeddings(
    ctx: *mut JSContext,
    p: Provider,
    body: &[u8],
    num_inputs: usize,
) -> JSValue {
    let body_c = CString::new(body).unwrap_or_default();
    let parsed = qjs::JS_ParseJSON(ctx, body_c.as_ptr(), body.len(), c"<embed_response>".as_ptr());
    if qjs::is_exception(parsed) {
        qjs::sofuu_js_get_exception(ctx);
        return qjs::sofuu_js_undefined();
    }

    let results = qjs::JS_NewArray(ctx);
    let mut ri: u32 = 0;

    if p == Provider::Local {
        /* /api/embed -> { "embeddings": [ [...], [...] ] } */
        let embs = qjs::sofuu_js_get_property_str(ctx, parsed, c"embeddings".as_ptr());
        if qjs::JS_IsArray(ctx, embs) != 0 {
            let mut elen: u32 = 0;
            let cv = qjs::sofuu_js_get_property_str(ctx, embs, c"length".as_ptr());
            qjs::sofuu_js_to_uint32(ctx, &mut elen, cv);
            qjs::sofuu_js_free_value(ctx, cv);
            for i in 0..elen {
                let arr = qjs::JS_GetPropertyUint32(ctx, embs, i);
                let f64 = create_float32_array(ctx, arr);
                if !qjs::is_undefined(f64) {
                    qjs::JS_SetPropertyUint32(ctx, results, ri, f64);
                    ri += 1;
                }
                qjs::sofuu_js_free_value(ctx, arr);
            }
        }
        qjs::sofuu_js_free_value(ctx, embs);
    } else {
        /* OpenAI -> { "data": [ { "embedding": [...] }, ... ] } */
        let data = qjs::sofuu_js_get_property_str(ctx, parsed, c"data".as_ptr());
        if qjs::JS_IsArray(ctx, data) != 0 {
            let mut dlen: u32 = 0;
            let cv = qjs::sofuu_js_get_property_str(ctx, data, c"length".as_ptr());
            qjs::sofuu_js_to_uint32(ctx, &mut dlen, cv);
            qjs::sofuu_js_free_value(ctx, cv);
            for i in 0..dlen {
                let item = qjs::JS_GetPropertyUint32(ctx, data, i);
                let emb = qjs::sofuu_js_get_property_str(ctx, item, c"embedding".as_ptr());
                let f64 = create_float32_array(ctx, emb);
                if !qjs::is_undefined(f64) {
                    qjs::JS_SetPropertyUint32(ctx, results, ri, f64);
                    ri += 1;
                }
                qjs::sofuu_js_free_value(ctx, emb);
                qjs::sofuu_js_free_value(ctx, item);
            }
        }
        qjs::sofuu_js_free_value(ctx, data);
    }

    qjs::sofuu_js_free_value(ctx, parsed);

    if ri == 0 {
        qjs::sofuu_js_free_value(ctx, results);
        return qjs::sofuu_js_undefined();
    }

    /* Return flat array if only 1 input requested, else Array<Float32Array> */
    if num_inputs <= 1 && ri == 1 {
        let flat = qjs::JS_GetPropertyUint32(ctx, results, 0);
        qjs::sofuu_js_free_value(ctx, results);
        return flat;
    }

    results
}

/* ------------------------------------------------------------------ */
/* Stream delta extractors — parse a single SSE data line via QuickJS  */
/* ------------------------------------------------------------------ */

/// # Safety
/// `ctx` live; `data` a single SSE `data:` line (or NULL → no delta).
unsafe fn extract_stream_think_qjs(ctx: *mut JSContext, p: Provider, data: &[u8]) -> Option<CString> {
    if data == b"[DONE]" {
        return None;
    }

    let data_c = CString::new(data).unwrap_or_default();
    let ev = qjs::JS_ParseJSON(ctx, data_c.as_ptr(), data_c.as_bytes().len(), c"<sse_think>".as_ptr());
    if qjs::is_exception(ev) {
        qjs::sofuu_js_get_exception(ctx);
        return None;
    }

    let mut th = qjs::sofuu_js_undefined();
    if p == Provider::Anthropic {
        let delta = qjs::sofuu_js_get_property_str(ctx, ev, c"delta".as_ptr());
        if qjs::is_object(delta) {
            let type_v = qjs::sofuu_js_get_property_str(ctx, delta, c"type".as_ptr());
            let ts = if qjs::sofuu_js_is_string(type_v) != 0 {
                qjs::sofuu_js_to_cstring(ctx, type_v)
            } else {
                ptr::null()
            };
            if !ts.is_null() && CStr::from_ptr(ts).to_bytes() == b"thinking_delta" {
                th = qjs::sofuu_js_get_property_str(ctx, delta, c"thinking".as_ptr());
            }
            if !ts.is_null() {
                qjs::sofuu_js_free_cstring(ctx, ts);
            }
            qjs::sofuu_js_free_value(ctx, type_v);
        }
        qjs::sofuu_js_free_value(ctx, delta);
    } else if p == Provider::Local {
        let msg = qjs::sofuu_js_get_property_str(ctx, ev, c"message".as_ptr());
        if qjs::is_object(msg) {
            th = qjs::sofuu_js_get_property_str(ctx, msg, c"thinking".as_ptr());
        }
        qjs::sofuu_js_free_value(ctx, msg);
    } else {
        /* OpenAI-compatible reasoning fields */
        let choices = qjs::sofuu_js_get_property_str(ctx, ev, c"choices".as_ptr());
        if qjs::JS_IsArray(ctx, choices) != 0 {
            let ch0 = qjs::JS_GetPropertyUint32(ctx, choices, 0);
            let delta = qjs::sofuu_js_get_property_str(ctx, ch0, c"delta".as_ptr());
            if qjs::is_object(delta) {
                th = qjs::sofuu_js_get_property_str(ctx, delta, c"reasoning_content".as_ptr());
                if qjs::sofuu_js_is_string(th) == 0 {
                    qjs::sofuu_js_free_value(ctx, th);
                    th = qjs::sofuu_js_get_property_str(ctx, delta, c"thinking".as_ptr());
                }
                if qjs::sofuu_js_is_string(th) == 0 {
                    qjs::sofuu_js_free_value(ctx, th);
                    th = qjs::sofuu_js_get_property_str(ctx, delta, c"reasoning".as_ptr());
                }
            }
            qjs::sofuu_js_free_value(ctx, delta);
            qjs::sofuu_js_free_value(ctx, ch0);
        }
        qjs::sofuu_js_free_value(ctx, choices);
    }

    let mut result: Option<CString> = None;
    if qjs::sofuu_js_is_string(th) != 0 {
        let s = qjs::sofuu_js_to_cstring(ctx, th);
        if !s.is_null() && CStr::from_ptr(s).to_bytes().len() > 0 {
            result = Some(CString::new(CStr::from_ptr(s).to_bytes()).unwrap_or_default());
        }
        if !s.is_null() {
            qjs::sofuu_js_free_cstring(ctx, s);
        }
    }
    qjs::sofuu_js_free_value(ctx, th);
    qjs::sofuu_js_free_value(ctx, ev);
    result
}

/// # Safety
/// `ctx` live; `data` a single SSE `data:` line.
unsafe fn extract_stream_delta_qjs(ctx: *mut JSContext, p: Provider, data: &[u8]) -> Option<CString> {
    if data == b"[DONE]" {
        return None;
    }

    let data_c = CString::new(data).unwrap_or_default();
    let ev = qjs::JS_ParseJSON(ctx, data_c.as_ptr(), data_c.as_bytes().len(), c"<sse>".as_ptr());
    if qjs::is_exception(ev) {
        qjs::sofuu_js_get_exception(ctx);
        return None;
    }

    let mut text_val = qjs::sofuu_js_undefined();

    if p == Provider::Anthropic {
        /*
         * Anthropic events:
         *   content_block_delta → delta.type="text_delta" → delta.text
         *   thinking_delta      → delta.type="thinking_delta" → delta.thinking
         * We only forward text_delta tokens to the push callback.
         */
        let delta = qjs::sofuu_js_get_property_str(ctx, ev, c"delta".as_ptr());
        if !qjs::is_undefined(delta) && !qjs::is_null(delta) {
            let dtype = qjs::sofuu_js_get_property_str(ctx, delta, c"type".as_ptr());
            let type_str = qjs::sofuu_js_to_cstring(ctx, dtype);
            if !type_str.is_null() && CStr::from_ptr(type_str).to_bytes() == b"text_delta" {
                text_val = qjs::sofuu_js_get_property_str(ctx, delta, c"text".as_ptr());
            }
            if !type_str.is_null() {
                qjs::sofuu_js_free_cstring(ctx, type_str);
            }
            qjs::sofuu_js_free_value(ctx, dtype);
        }
        qjs::sofuu_js_free_value(ctx, delta);
    } else if p == Provider::Local {
        /* local /api/chat streaming lines:
         * { "message": { "role": "assistant", "content": "..." } } */
        let msg = qjs::sofuu_js_get_property_str(ctx, ev, c"message".as_ptr());
        if qjs::is_object(msg) {
            text_val = qjs::sofuu_js_get_property_str(ctx, msg, c"content".as_ptr());
        }
        qjs::sofuu_js_free_value(ctx, msg);
    } else {
        /*
         * OpenAI:
         * choices[0].delta.content
         */
        let choices = qjs::sofuu_js_get_property_str(ctx, ev, c"choices".as_ptr());
        if qjs::JS_IsArray(ctx, choices) != 0 {
            let ch0 = qjs::JS_GetPropertyUint32(ctx, choices, 0);
            let delta = qjs::sofuu_js_get_property_str(ctx, ch0, c"delta".as_ptr());
            text_val = qjs::sofuu_js_get_property_str(ctx, delta, c"content".as_ptr());
            qjs::sofuu_js_free_value(ctx, delta);
            qjs::sofuu_js_free_value(ctx, ch0);
        }
        qjs::sofuu_js_free_value(ctx, choices);
    }

    let mut result: Option<CString> = None;
    if !qjs::is_undefined(text_val) && !qjs::is_null(text_val) {
        let s = qjs::sofuu_js_to_cstring(ctx, text_val);
        if !s.is_null() && CStr::from_ptr(s).to_bytes().len() > 0 {
            result = Some(CString::new(CStr::from_ptr(s).to_bytes()).unwrap_or_default());
        }
        if !s.is_null() {
            qjs::sofuu_js_free_cstring(ctx, s);
        }
    }
    qjs::sofuu_js_free_value(ctx, text_val);
    qjs::sofuu_js_free_value(ctx, ev);
    result
}

/// F4b: Anthropic streamed `tool_use` accumulation. `content_block_start`
/// (block type "tool_use") opens a tool call (id + name); each
/// `content_block_delta` of type `input_json_delta` appends `partial_json`
/// to its arguments. Fragments are reshaped to the OpenAI delta shape
/// ({index, id, function:{name|arguments}}) so the one JS-side merger
/// (agent.js mergeStreamToolCalls) is provider-neutral. Returns a one-
/// element JS array owned by the caller, or None when the event carries no
/// tool fragment.
///
/// # Safety
/// `ctx` live; `ev` a parsed JSON event value (not consumed).
unsafe fn anthropic_tool_fragments(ctx: *mut JSContext, ev: JSValue) -> Option<JSValue> {
    let etype = qjs::sofuu_js_get_property_str(ctx, ev, c"type".as_ptr());
    let etype_s = qjs::sofuu_js_to_cstring(ctx, etype);
    let etype_b = if etype_s.is_null() {
        Vec::new()
    } else {
        CStr::from_ptr(etype_s).to_bytes().to_vec()
    };
    if !etype_s.is_null() {
        qjs::sofuu_js_free_cstring(ctx, etype_s);
    }
    qjs::sofuu_js_free_value(ctx, etype);
    if etype_b != b"content_block_start" && etype_b != b"content_block_delta" {
        return None;
    }

    let index_v = qjs::sofuu_js_get_property_str(ctx, ev, c"index".as_ptr());
    let mut idx: i32 = 0;
    qjs::JS_ToInt32(ctx, &mut idx, index_v);
    qjs::sofuu_js_free_value(ctx, index_v);

    let func = qjs::sofuu_js_new_object(ctx);
    let frag = qjs::sofuu_js_new_object(ctx);
    qjs::sofuu_js_set_property_str(ctx, frag, c"index".as_ptr(), qjs::sofuu_js_new_int32(ctx, idx));
    qjs::sofuu_js_set_property_str(ctx, frag, c"type".as_ptr(), qjs::sofuu_js_new_string(ctx, c"function".as_ptr()));

    let is_tool;
    if etype_b == b"content_block_start" {
        let block = qjs::sofuu_js_get_property_str(ctx, ev, c"content_block".as_ptr());
        let bt = qjs::sofuu_js_get_property_str(ctx, block, c"type".as_ptr());
        let bt_s = qjs::sofuu_js_to_cstring(ctx, bt);
        is_tool = !bt_s.is_null() && CStr::from_ptr(bt_s).to_bytes() == b"tool_use";
        if !bt_s.is_null() {
            qjs::sofuu_js_free_cstring(ctx, bt_s);
        }
        qjs::sofuu_js_free_value(ctx, bt);
        if is_tool {
            let id = qjs::sofuu_js_get_property_str(ctx, block, c"id".as_ptr());
            let name = qjs::sofuu_js_get_property_str(ctx, block, c"name".as_ptr());
            if !qjs::is_undefined(id) {
                qjs::sofuu_js_set_property_str(ctx, frag, c"id".as_ptr(), qjs::sofuu_js_dup_value(ctx, id));
            }
            if !qjs::is_undefined(name) {
                qjs::sofuu_js_set_property_str(ctx, func, c"name".as_ptr(), qjs::sofuu_js_dup_value(ctx, name));
            }
            qjs::sofuu_js_free_value(ctx, id);
            qjs::sofuu_js_free_value(ctx, name);
            qjs::sofuu_js_set_property_str(ctx, func, c"arguments".as_ptr(), qjs::sofuu_js_new_string(ctx, c"".as_ptr()));
        }
        qjs::sofuu_js_free_value(ctx, block);
    } else {
        let delta = qjs::sofuu_js_get_property_str(ctx, ev, c"delta".as_ptr());
        let dt = qjs::sofuu_js_get_property_str(ctx, delta, c"type".as_ptr());
        let dt_s = qjs::sofuu_js_to_cstring(ctx, dt);
        is_tool = !dt_s.is_null() && CStr::from_ptr(dt_s).to_bytes() == b"input_json_delta";
        if !dt_s.is_null() {
            qjs::sofuu_js_free_cstring(ctx, dt_s);
        }
        qjs::sofuu_js_free_value(ctx, dt);
        if is_tool {
            let pj = qjs::sofuu_js_get_property_str(ctx, delta, c"partial_json".as_ptr());
            if !qjs::is_undefined(pj) {
                qjs::sofuu_js_set_property_str(ctx, func, c"arguments".as_ptr(), qjs::sofuu_js_dup_value(ctx, pj));
            }
            qjs::sofuu_js_free_value(ctx, pj);
        }
        qjs::sofuu_js_free_value(ctx, delta);
    }

    if !is_tool {
        qjs::sofuu_js_free_value(ctx, func);
        qjs::sofuu_js_free_value(ctx, frag);
        return None;
    }
    qjs::sofuu_js_set_property_str(ctx, frag, c"function".as_ptr(), func);
    let arr = qjs::JS_NewArray(ctx);
    // JS_SetPropertyUint32 steals the fragment reference (same contract as
    // JS_SetPropertyStr); array index 0 — each SSE event yields one fragment.
    qjs::JS_SetPropertyUint32(ctx, arr, 0, frag);
    Some(arr)
}
/* ------------------------------------------------------------------ */
/* Shared curl-multi / libuv bridge                                     */
/* ------------------------------------------------------------------ */

/* Tag to distinguish complete vs stream requests stored in CURLINFO_PRIVATE */
const REQ_TAG_COMPLETE: c_int = 1;
const REQ_TAG_STREAM: c_int = 2;
const REQ_TAG_EMBED: c_int = 3;

/* Hard cap on any single provider response (complete/stream/embed) so a
 * misbehaving endpoint cannot grow the heap without bound. */
const AI_MAX_RESPONSE: usize = 64 * 1024 * 1024;

thread_local! {
    static G_MULTI: Cell<*mut CurlM> = const { Cell::new(ptr::null_mut()) };
    static G_TIMER: Cell<*mut UvTimer> = const { Cell::new(ptr::null_mut()) };
    static G_INIT_DONE: Cell<c_int> = const { Cell::new(0) };
    /* Deferred stream-flush: one-shot check handle that runs JS microtasks
     * OUTSIDE the curl write callback (see ai_flush_defer). */
    static G_FLUSH_CHECK: Cell<*mut UvCheck> = const { Cell::new(ptr::null_mut()) };
    static G_FLUSH_CHECK_INIT: Cell<c_int> = const { Cell::new(0) };
    static G_FLUSH_CTX: Cell<*mut JSContext> = const { Cell::new(ptr::null_mut()) };
    /* active streams (LIFO list) + per-stream id source (C statics) */
    static G_STREAMS: Cell<*mut AiStreamReq> = const { Cell::new(ptr::null_mut()) };
    static G_STREAM_SID: Cell<i64> = const { Cell::new(0) };
}

/* ai.complete() / ai.embed() request — tag is the FIRST field so the
 * CURLINFO_PRIVATE pointer can be tag-dispatched like C's req_base_t. */
#[repr(C)]
struct AiCompleteReq {
    tag: c_int,
    ctx: *mut JSContext,
    promise: *mut PromiseHandle,
    provider: Provider,
    response_body: Vec<u8>,
    headers: *mut CurlSlist,
    post_body: Option<CString>,
    easy: *mut Curl,
}

#[repr(C)]
struct AiEmbedReq {
    tag: c_int,
    ctx: *mut JSContext,
    promise: *mut PromiseHandle,
    provider: Provider,
    response_body: Vec<u8>,
    headers: *mut CurlSlist,
    post_body: Option<CString>,
    easy: *mut Curl,
    num_inputs: usize,
}

struct AiPollCtx {
    sockfd: c_int,
    poll: *mut UvPoll,
}

unsafe extern "C" fn ai_poll_close_cb(h: *mut UvHandle) {
    // SAFETY: data set at creation (poll handle's own storage is separate,
    // same split as the M4 fetch port's CurlContext).
    let c = *(h as *mut *mut AiPollCtx);
    libc::free(h as *mut c_void);
    drop(Box::from_raw(c));
}

unsafe extern "C" fn ai_poll_cb(req: *mut UvPoll, _status: c_int, events: c_int) {
    let mut flags: c_int = 0;
    if events & uv::UV_READABLE != 0 {
        flags |= curl::CURL_CSELECT_IN;
    }
    if events & uv::UV_WRITABLE != 0 {
        flags |= curl::CURL_CSELECT_OUT;
    }
    // SAFETY: the poll handle's data points at its AiPollCtx (set at init).
    let c = *(req as *mut *mut AiPollCtx);
    let mut running: c_int = 0;
    curl::curl_multi_socket_action(g_multi(), (*c).sockfd, flags, &mut running);
    /* NOTE: do NOT run JS microtasks here — we are inside libcurl's
     * socket_action call stack. Running JS would call curl_multi_add_handle
     * from within a curl callback (CURLM_RECURSIVE_API_CALL).
     * sofuu_loop_run pumps jobs after every uv_run(UV_RUN_ONCE). */
    ai_check_multi();
}

unsafe extern "C" fn ai_timer_cb(_t: *mut UvTimer) {
    let mut running: c_int = 0;
    curl::curl_multi_socket_action(g_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running);
    /* NOTE: same as ai_poll_cb — no JS flush here. */
    ai_check_multi();
}

unsafe extern "C" fn ai_socket_fn(
    _e: *mut Curl,
    s: c_int,
    action: c_int,
    _userp: *mut c_void,
    socketp: *mut c_void,
) -> c_int {
    if action == curl::CURL_POLL_IN || action == curl::CURL_POLL_OUT || action == curl::CURL_POLL_INOUT {
        let c: *mut AiPollCtx = if !socketp.is_null() {
            socketp as *mut AiPollCtx
        } else {
            let context = Box::into_raw(Box::new(AiPollCtx { sockfd: s, poll: ptr::null_mut() }));
            let poll = libc::malloc(uv::sofuu_uv_poll_size()) as *mut UvPoll;
            (*context).poll = poll;
            // SAFETY: fresh malloc'd storage; data is the first field.
            uv::uv_poll_init_socket(sofuu_loop_get(), poll, s);
            *(poll as *mut *mut AiPollCtx) = context;
            curl::curl_multi_assign(g_multi(), s, context as *mut c_void);
            context
        };
        let mut ev: c_int = 0;
        if action != curl::CURL_POLL_IN {
            ev |= uv::UV_WRITABLE;
        }
        if action != curl::CURL_POLL_OUT {
            ev |= uv::UV_READABLE;
        }
        uv::uv_poll_start((*c).poll, ev, Some(ai_poll_cb));
    } else {
        if !socketp.is_null() {
            let c = socketp as *mut AiPollCtx;
            uv::uv_poll_stop((*c).poll);
            uv::uv_close((*c).poll as *mut UvHandle, Some(ai_poll_close_cb));
            curl::curl_multi_assign(g_multi(), s, ptr::null_mut());
        }
    }
    0
}

unsafe extern "C" fn ai_timer_fn(_m: *mut CurlM, ms: c_long, _userp: *mut c_void) -> c_int {
    let timer = G_TIMER.with(|t| t.get());
    if ms < 0 {
        uv::uv_timer_stop(timer);
    } else {
        let t = if ms == 0 { 1 } else { ms as u64 };
        uv::uv_timer_start(timer, Some(ai_timer_cb), t, 0);
    }
    0
}

unsafe fn ai_ensure_init() {
    let done = G_INIT_DONE.with(|d| d.get());
    if done != 0 {
        return;
    }
    curl::curl_global_init(curl::CURL_GLOBAL_ALL);
    let timer = libc::malloc(uv::sofuu_uv_timer_size()) as *mut UvTimer;
    G_TIMER.with(|t| t.set(timer));
    uv::uv_timer_init(sofuu_loop_get(), timer);
    let multi = curl::curl_multi_init();
    G_MULTI.with(|m| m.set(multi));
    curl::curl_multi_setopt(multi, CURLMOPT_SOCKETFUNCTION, ai_socket_fn as *const c_void);
    curl::curl_multi_setopt(multi, CURLMOPT_TIMERFUNCTION, ai_timer_fn as *const c_void);
    G_INIT_DONE.with(|d| d.set(1));
}

fn g_multi() -> *mut CurlM {
    G_MULTI.with(|m| m.get())
}

/// Deferred JS flush — NEVER run sofuu_flush_jobs inside a curl write
/// callback (we are inside curl_multi_socket_action; running JS there can
/// re-enter curl and fail with CURLM_RECURSIVE_API_CALL). Instead, arm a
/// one-shot uv_check_t: it fires on the loop's next iteration, after the
/// curl call stack has unwound, and flushes in a safe JS context. This is
/// the same pattern the fetch stream uses (rt/http_client.rs).
unsafe fn ai_flush_defer() {
    if G_FLUSH_CHECK_INIT.with(|i| i.get()) == 0 {
        let check = libc::malloc(uv::sofuu_uv_check_size()) as *mut UvCheck;
        G_FLUSH_CHECK.with(|c| c.set(check));
        uv::uv_check_init(sofuu_loop_get(), check);
        G_FLUSH_CHECK_INIT.with(|i| i.set(1));
    }
    let check = G_FLUSH_CHECK.with(|c| c.get());
    uv::uv_check_start(check, Some(ai_flush_check_cb));
}

/// One-shot check callback: stop ourselves, then flush the JS job queue in
/// a safe context (the curl stack has fully unwound by now).
unsafe extern "C" fn ai_flush_check_cb(_h: *mut UvCheck) {
    let check = G_FLUSH_CHECK.with(|c| c.get());
    uv::uv_check_stop(check);
    let ctx = G_FLUSH_CTX.with(|c| c.get());
    if !ctx.is_null() {
        sofuu_flush_jobs(ctx);
    }
}

/* ------------------------------------------------------------------ */
/* Write callbacks                                                      */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn complete_write_cb(
    ptr: *mut c_void,
    size: usize,
    nmemb: usize,
    ud: *mut c_void,
) -> usize {
    let n = size * nmemb;
    let req = ud as *mut AiCompleteReq;
    if (*req).response_body.len() + n + 1 > AI_MAX_RESPONSE {
        return 0; /* → curl write error */
    }
    (*req)
        .response_body
        .extend_from_slice(std::slice::from_raw_parts(ptr as *const u8, n));
    n
}

/* ------------------------------------------------------------------ */
/* Stream request lifecycle                                             */
/* ------------------------------------------------------------------ */

#[repr(C)]
struct AiStreamReq {
    tag: c_int, /* must be first — REQ_TAG_STREAM */
    ctx: *mut JSContext,
    provider: Provider,
    push_fn: JSValue,
    done_fn: JSValue,
    error_fn: JSValue,
    think_fn: JSValue,    /* factory setThink — reasoning text */
    tool_calls_fn: JSValue, /* factory setToolCalls — streamed delta.tool_calls */
    sse_buf: Vec<u8>,
    headers: *mut CurlSlist,
    post_body: Option<CString>,
    easy: *mut Curl,
    /* Token usage tracking — populated from the final SSE chunk */
    prompt_tokens: i32,
    completion_tokens: i32,
    /* P6: provider cache accounting when reported (Anthropic
     * cache_read/cache_creation input tokens; OpenAI cached_tokens). */
    cache_read_tokens: i32,
    cache_write_tokens: i32,
    done_called: c_int, /* done_fn fired exactly once */
    /// First bytes of the response body (used to surface provider JSON
    /// `error.message` when status >=400 — avoids bare "HTTP 429").
    error_body: Vec<u8>,
    /* Abort support: streams register here until they finish, so a
     * global __ai_abort(sid) (Esc in the chat) can stop them mid-flight. */
    sid: i64,
    next: *mut AiStreamReq,
}

unsafe fn stream_unlink(req: *mut AiStreamReq) {
    G_STREAMS.with(|s| {
        let head = s.get();
        if head == req {
            s.set((*req).next);
            return;
        }
        let mut p = head;
        while !p.is_null() {
            if (*p).next == req {
                (*p).next = (*req).next;
                return;
            }
            p = (*p).next;
        }
    });
}

/* Free everything a stream owns and drop it from the active list. The
 * caller handles the curl side (DONE path removes+cleans; abort too). */
unsafe fn stream_req_destroy(ctx: *mut JSContext, req: *mut AiStreamReq) {
    stream_unlink(req);
    qjs::sofuu_js_free_value(ctx, (*req).push_fn);
    qjs::sofuu_js_free_value(ctx, (*req).done_fn);
    qjs::sofuu_js_free_value(ctx, (*req).error_fn);
    qjs::sofuu_js_free_value(ctx, (*req).think_fn);
    qjs::sofuu_js_free_value(ctx, (*req).tool_calls_fn);
    if !(*req).headers.is_null() {
        curl::curl_slist_free_all((*req).headers);
    }
    drop(Box::from_raw(req));
}

/// Fill the provider-neutral usage stats object handed to stream.done()
/// (P6: cacheRead/cacheWrite included; 0 when the provider doesn't report).
unsafe fn set_stream_stats(ctx: *mut JSContext, stats: JSValue, prompt: i32, completion: i32,
                           cache_read: i32, cache_write: i32) {
    qjs::sofuu_js_set_property_str(
        ctx,
        stats,
        c"promptTokens".as_ptr(),
        qjs::sofuu_js_new_int32(ctx, prompt),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stats,
        c"completionTokens".as_ptr(),
        qjs::sofuu_js_new_int32(ctx, completion),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stats,
        c"cacheReadTokens".as_ptr(),
        qjs::sofuu_js_new_int32(ctx, cache_read),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stats,
        c"cacheWriteTokens".as_ptr(),
        qjs::sofuu_js_new_int32(ctx, cache_write),
    );
}

/* Stop one in-flight stream gracefully: end the async iterator with the
 * usage collected so far, then tear down. Called from __ai_abort (Esc/Ctrl-C
 * in the chat) — we are in a JS call, NOT a curl callback, so the multi
 * operations here can't recurse. */
unsafe fn stream_abort_one(req: *mut AiStreamReq) {
    let ctx = (*req).ctx;
    if (*req).done_called == 0 {
        (*req).done_called = 1;
        if qjs::JS_IsFunction(ctx, (*req).done_fn) != 0 {
            let stats = qjs::sofuu_js_new_object(ctx);
            set_stream_stats(ctx, stats, (*req).prompt_tokens, (*req).completion_tokens,
                             (*req).cache_read_tokens, (*req).cache_write_tokens);
            let ret = qjs::JS_Call(ctx, (*req).done_fn, qjs::sofuu_js_undefined(), 1, &stats);
            if qjs::is_exception(ret) {
                qjs::js_std_dump_error(ctx);
            }
            qjs::sofuu_js_free_value(ctx, ret);
            qjs::sofuu_js_free_value(ctx, stats);
        }
    }
    if !(*req).easy.is_null() {
        /* libcurl ≥ 8: remove_handle also purges queued messages for the
         * handle, so its (about to be recycled) address can never resurface
         * in ai_check_multi(). */
        curl::curl_multi_remove_handle(g_multi(), (*req).easy);
        curl::curl_easy_cleanup((*req).easy);
        (*req).easy = ptr::null_mut();
    }
    stream_req_destroy(ctx, req);
}

unsafe extern "C" fn stream_write_cb(
    ptr: *mut c_void,
    size: usize,
    nmemb: usize,
    ud: *mut c_void,
) -> usize {
    let n = size * nmemb;
    let req = ud as *mut AiStreamReq;
    let ctx = (*req).ctx;

    /* Cap the SSE buffer: a never-terminated "data:" line must not grow
     * the heap forever. Returning 0 aborts the transfer with a curl error. */
    if (*req).sse_buf.len() + n + 1 > AI_MAX_RESPONSE {
        return 0;
    }
    let chunk = std::slice::from_raw_parts(ptr as *const u8, n);
    (*req).sse_buf.extend_from_slice(chunk);
    // Also capture first bytes for error surfacing (when status >=400 the SSE
    // parser may never see a data: line — the provider returns JSON directly).
    if (*req).error_body.len() < 8192 {
        let room = 8192 - (*req).error_body.len();
        (*req).error_body.extend_from_slice(&chunk[..chunk.len().min(room)]);
    }

    let mut head = 0usize;
    let end = (*req).sse_buf.len();

    while head < end {
        /* Bind the buffer once — Rust indexing through `(*p).field[i]`
         * trips the implicit-autoref lint (same rule as mcp.rs). */
        let sse = &mut (*req).sse_buf;
        let Some(nl_rel) = sse[head..end].iter().position(|&b| b == b'\n') else {
            break;
        };
        let nl = head + nl_rel;

        let mut line_end = nl;
        if line_end > head && sse[line_end - 1] == b'\r' {
            line_end -= 1;
        }
        let line = &sse[head..line_end];

        if let Some(rest) = line.strip_prefix(b"data:") {
            /* SSE spec allows both "data: x" and "data:x"; some gateways
             * add trailing whitespace. Trim before comparing to [DONE]. */
            let mut data = rest.strip_prefix(b" ").unwrap_or(rest);
            while data.last().is_some_and(|&b| b == b' ' || b == b'\t' || b == b'\r') {
                data = &data[..data.len() - 1];
            }

            if data == b"[DONE]" {
                /* Stream finished — signal done with token stats (only
                 * once; the CURLMSG_DONE path also "ensures" done). */
                if (*req).done_called != 0 {
                    head = nl + 1;
                    continue;
                }
                (*req).done_called = 1;
                if qjs::JS_IsFunction(ctx, (*req).done_fn) != 0 {
                    let stats = qjs::sofuu_js_new_object(ctx);
                    set_stream_stats(ctx, stats, (*req).prompt_tokens, (*req).completion_tokens,
                                     (*req).cache_read_tokens, (*req).cache_write_tokens);
                    let ret = qjs::JS_Call(ctx, (*req).done_fn, qjs::sofuu_js_undefined(), 1, &stats);
                    if qjs::is_exception(ret) {
                        qjs::js_std_dump_error(ctx);
                    }
                    qjs::sofuu_js_free_value(ctx, ret);
                    qjs::sofuu_js_free_value(ctx, stats);
                }
                head = nl + 1;
                continue;
            }

            /* Try to extract token usage from this chunk (may appear on any chunk) */
            {
                let data_c = CString::new(data).unwrap_or_default();
                let ev_use = qjs::JS_ParseJSON(ctx, data_c.as_ptr(), data_c.as_bytes().len(), c"<sse_use>".as_ptr());
                if !qjs::is_exception(ev_use) {
                    let mut use_v = qjs::sofuu_js_get_property_str(ctx, ev_use, c"usage".as_ptr());
                    if qjs::is_undefined(use_v) || qjs::is_null(use_v) {
                        // Anthropic message_start nests it: message.usage.
                        qjs::sofuu_js_free_value(ctx, use_v);
                        let msg = qjs::sofuu_js_get_property_str(ctx, ev_use, c"message".as_ptr());
                        use_v = qjs::sofuu_js_get_property_str(ctx, msg, c"usage".as_ptr());
                        qjs::sofuu_js_free_value(ctx, msg);
                    }
                    if !qjs::is_undefined(use_v) && !qjs::is_null(use_v) {
                        let mut pt = qjs::sofuu_js_get_property_str(ctx, use_v, c"prompt_tokens".as_ptr());
                        let mut ct = qjs::sofuu_js_get_property_str(ctx, use_v, c"completion_tokens".as_ptr());
                        /* Anthropic streams report {input_tokens, output_tokens}
                         * (message_start / message_delta) — map them onto the
                         * same slots so stream usage is provider-neutral. */
                        if qjs::is_undefined(pt) {
                            qjs::sofuu_js_free_value(ctx, pt);
                            pt = qjs::sofuu_js_get_property_str(ctx, use_v, c"input_tokens".as_ptr());
                        }
                        if qjs::is_undefined(ct) {
                            qjs::sofuu_js_free_value(ctx, ct);
                            ct = qjs::sofuu_js_get_property_str(ctx, use_v, c"output_tokens".as_ptr());
                        }
                        let mut pv: i32 = 0;
                        let mut cv: i32 = 0;
                        if !qjs::is_undefined(pt) {
                            qjs::JS_ToInt32(ctx, &mut pv, pt);
                        }
                        if !qjs::is_undefined(ct) {
                            qjs::JS_ToInt32(ctx, &mut cv, ct);
                        }
                        if pv > 0 {
                            (*req).prompt_tokens = pv;
                        }
                        if cv > 0 {
                            (*req).completion_tokens = cv;
                        }
                        qjs::sofuu_js_free_value(ctx, pt);
                        qjs::sofuu_js_free_value(ctx, ct);
                        /* P6: cache accounting when the provider reports it —
                         * Anthropic: cache_read_input_tokens /
                         * cache_creation_input_tokens; OpenAI:
                         * prompt_tokens_details.cached_tokens. Absent → 0. */
                        let mut cr = qjs::sofuu_js_get_property_str(
                            ctx,
                            use_v,
                            c"cache_read_input_tokens".as_ptr(),
                        );
                        if qjs::is_undefined(cr) || qjs::is_null(cr) {
                            qjs::sofuu_js_free_value(ctx, cr);
                            let ptd = qjs::sofuu_js_get_property_str(
                                ctx,
                                use_v,
                                c"prompt_tokens_details".as_ptr(),
                            );
                            cr = qjs::sofuu_js_get_property_str(ctx, ptd, c"cached_tokens".as_ptr());
                            qjs::sofuu_js_free_value(ctx, ptd);
                        }
                        let cw = qjs::sofuu_js_get_property_str(
                            ctx,
                            use_v,
                            c"cache_creation_input_tokens".as_ptr(),
                        );
                        let mut crv: i32 = 0;
                        let mut cwv: i32 = 0;
                        if !qjs::is_undefined(cr) && !qjs::is_null(cr) {
                            qjs::JS_ToInt32(ctx, &mut crv, cr);
                        }
                        if !qjs::is_undefined(cw) && !qjs::is_null(cw) {
                            qjs::JS_ToInt32(ctx, &mut cwv, cw);
                        }
                        if crv > 0 {
                            (*req).cache_read_tokens = crv;
                        }
                        if cwv > 0 {
                            (*req).cache_write_tokens = cwv;
                        }
                        qjs::sofuu_js_free_value(ctx, cr);
                        qjs::sofuu_js_free_value(ctx, cw);
                    }
                    qjs::sofuu_js_free_value(ctx, use_v);
                    qjs::sofuu_js_free_value(ctx, ev_use);
                }
            }

            /* Reasoning/thinking text travels with the chunk as `think`
             * (the driver renders it dimmed). Extract + set BEFORE push so
             * the pushed chunk carries it. */
            let think = extract_stream_think_qjs(ctx, (*req).provider, data);
            let delta = extract_stream_delta_qjs(ctx, (*req).provider, data);
            /* Streamed tool-call deltas: surface them via setToolCalls so
             * the driver can run its tool loop even when the model streams
             * the tool call instead of returning it in a single complete()
             * response. OpenAI(-compat): choices[0].delta.tool_calls;
             * Anthropic (F4b): content_block_start/tool_use +
             * input_json_delta fragments, reshaped to the OpenAI delta
             * shape so the JS merger is provider-neutral. */
            if qjs::JS_IsFunction(ctx, (*req).tool_calls_fn) != 0 {
                if (*req).provider == Provider::Anthropic {
                    let data_c2 = CString::new(data).unwrap_or_default();
                    let ev2 = qjs::JS_ParseJSON(ctx, data_c2.as_ptr(), data_c2.as_bytes().len(), c"<sse_tc>".as_ptr());
                    if !qjs::is_exception(ev2) {
                        if let Some(tc) = anthropic_tool_fragments(ctx, ev2) {
                            let ret = qjs::JS_Call(ctx, (*req).tool_calls_fn, qjs::sofuu_js_undefined(), 1, &tc);
                            if qjs::is_exception(ret) {
                                qjs::js_std_dump_error(ctx);
                            }
                            qjs::sofuu_js_free_value(ctx, ret);
                            qjs::sofuu_js_free_value(ctx, tc);
                        }
                        qjs::sofuu_js_free_value(ctx, ev2);
                    }
                } else {
                    let data_c2 = CString::new(data).unwrap_or_default();
                    let ev2 = qjs::JS_ParseJSON(ctx, data_c2.as_ptr(), data_c2.as_bytes().len(), c"<sse_tc>".as_ptr());
                    if !qjs::is_exception(ev2) {
                        let choices2 = qjs::sofuu_js_get_property_str(ctx, ev2, c"choices".as_ptr());
                        if qjs::JS_IsArray(ctx, choices2) != 0 {
                            let ch0 = qjs::JS_GetPropertyUint32(ctx, choices2, 0);
                            let d2 = qjs::sofuu_js_get_property_str(ctx, ch0, c"delta".as_ptr());
                            let tc = qjs::sofuu_js_get_property_str(ctx, d2, c"tool_calls".as_ptr());
                            if qjs::JS_IsArray(ctx, tc) != 0 && !qjs::is_undefined(tc) {
                                let ret = qjs::JS_Call(ctx, (*req).tool_calls_fn, qjs::sofuu_js_undefined(), 1, &tc);
                                if qjs::is_exception(ret) {
                                    qjs::js_std_dump_error(ctx);
                                }
                                qjs::sofuu_js_free_value(ctx, ret);
                            }
                            qjs::sofuu_js_free_value(ctx, tc);
                            qjs::sofuu_js_free_value(ctx, d2);
                            qjs::sofuu_js_free_value(ctx, ch0);
                        }
                        qjs::sofuu_js_free_value(ctx, choices2);
                        qjs::sofuu_js_free_value(ctx, ev2);
                    }
                }
            }
            if let Some(t) = &think {
                if !t.is_empty() && qjs::JS_IsFunction(ctx, (*req).think_fn) != 0 {
                    let jt = qjs::sofuu_js_new_string(ctx, t.as_ptr());
                    let ret = qjs::JS_Call(ctx, (*req).think_fn, qjs::sofuu_js_undefined(), 1, &jt);
                    if qjs::is_exception(ret) {
                        qjs::js_std_dump_error(ctx);
                    }
                    qjs::sofuu_js_free_value(ctx, ret);
                    qjs::sofuu_js_free_value(ctx, jt);
                }
            }
            let has_think = think.as_ref().is_some_and(|t| !t.is_empty());
            if delta.is_some() || has_think {
                if qjs::JS_IsFunction(ctx, (*req).push_fn) != 0 {
                    let jd = match &delta {
                        Some(d) => qjs::sofuu_js_new_string(ctx, d.as_ptr()),
                        None => qjs::sofuu_js_new_string(ctx, c"".as_ptr()),
                    };
                    let ret = qjs::JS_Call(ctx, (*req).push_fn, qjs::sofuu_js_undefined(), 1, &jd);
                    if qjs::is_exception(ret) {
                        qjs::js_std_dump_error(ctx);
                    }
                    qjs::sofuu_js_free_value(ctx, ret);
                    qjs::sofuu_js_free_value(ctx, jd);
                    /* Defer the microtask pump: we are inside the curl write
                     * callback (curl_multi_socket_action). Running JS here
                     * would let the for-await continuation start a new
                     * ai.complete → curl_multi_add_handle from inside curl
                     * (CURLM_RECURSIVE_API_CALL). The uv_check_t flushes
                     * after the curl stack unwinds; the loop also pumps
                     * after every uv_run. */
                    G_FLUSH_CTX.with(|c| c.set(ctx));
                    ai_flush_defer();
                }
            }
        }

        head = nl + 1;
    }

    /* Slide remaining bytes to front */
    let consumed = head;
    if consumed > 0 {
        let buf = &mut (*req).sse_buf;
        buf.copy_within(consumed.., 0);
        buf.truncate(buf.len() - consumed);
    }

    n
}

/* ------------------------------------------------------------------ */
/* Shared multi-handle completion checker                               */
/* ------------------------------------------------------------------ */

unsafe fn ai_check_multi() {
    let mut pending: c_int = 0;
    loop {
        let msg_ptr = curl::curl_multi_info_read(g_multi(), &mut pending);
        if msg_ptr.is_null() {
            break;
        }
        let msg = *msg_ptr;
        if msg.msg != curl::CURLMSG_DONE {
            continue;
        }

        let easy = msg.easy_handle;
        let code: c_int = msg.result;

        let mut base: *mut c_void = ptr::null_mut();
        curl::curl_easy_getinfo(easy, curl::CURLINFO_PRIVATE, &mut base);

        if base.is_null() {
            curl::curl_multi_remove_handle(g_multi(), easy);
            curl::curl_easy_cleanup(easy);
            continue;
        }

        let tag = (base as *const c_int).read();

        if tag == REQ_TAG_COMPLETE {
            let req = base as *mut AiCompleteReq;
            let ctx = (*req).ctx;

            if code == curl::CURLE_OK {
                let mut status: c_long = 0;
                curl::curl_easy_getinfo(easy, curl::CURLINFO_RESPONSE_CODE, &mut status as *mut c_long);
                let raw: &[u8] = &(*req).response_body;

                if status >= 400 {
                    let raw_c = CString::new(raw).unwrap_or_default();
                    let err = qjs::sofuu_js_new_string(ctx, raw_c.as_ptr());
                    sofuu_promise_reject((*req).promise, err);
                } else {
                    /* extract_text returns the stripped answer + the think
                       text — the C wrote thinking into the per-request
                       think_buf (no shared global → no data race). */
                    let (text, think) = extract_text((*req).provider, raw);
                    let result = qjs::sofuu_js_new_object(ctx);
                    let text_c = CString::new(text.as_slice()).unwrap_or_default();
                    qjs::sofuu_js_set_property_str(
                        ctx,
                        result,
                        c"text".as_ptr(),
                        qjs::sofuu_js_new_string(ctx, text_c.as_ptr()),
                    );

                    /* native_think: Anthropic extended-thinking field */
                    let native_think = extract_json_field(raw, "thinking");
                    if !native_think.is_empty() {
                        let nt_c = CString::new(native_think.as_slice()).unwrap_or_default();
                        qjs::sofuu_js_set_property_str(
                            ctx,
                            result,
                            c"thinking".as_ptr(),
                            qjs::sofuu_js_new_string(ctx, nt_c.as_ptr()),
                        );
                    } else if !think.is_empty() {
                        let th_c = CString::new(think.as_slice()).unwrap_or_default();
                        qjs::sofuu_js_set_property_str(
                            ctx,
                            result,
                            c"thinking".as_ptr(),
                            qjs::sofuu_js_new_string(ctx, th_c.as_ptr()),
                        );
                    }
                    /* Attach tool_calls array if the model requested a function call */
                    let tool_calls = extract_tool_calls(ctx, (*req).provider, raw);
                    if !qjs::is_undefined(tool_calls) {
                        qjs::sofuu_js_set_property_str(ctx, result, c"toolCalls".as_ptr(), tool_calls);
                        /* Convenience: mark the response so callers can branch cleanly */
                        qjs::sofuu_js_set_property_str(
                            ctx,
                            result,
                            c"stop_reason".as_ptr(),
                            qjs::sofuu_js_new_string(ctx, c"tool_use".as_ptr()),
                        );
                    }
                    /* Expose the raw JSON response payload for advanced manual stat extraction */
                    let raw_c = CString::new(raw).unwrap_or_default();
                    qjs::sofuu_js_set_property_str(
                        ctx,
                        result,
                        c"raw".as_ptr(),
                        qjs::sofuu_js_new_string(ctx, raw_c.as_ptr()),
                    );
                    sofuu_promise_resolve((*req).promise, result);
                    qjs::sofuu_js_free_value(ctx, result);
                }
            } else {
                let err = qjs::sofuu_js_new_string(ctx, curl::curl_easy_strerror(code));
                sofuu_promise_reject((*req).promise, err);
            }

            if !(*req).headers.is_null() {
                curl::curl_slist_free_all((*req).headers);
            }
            drop(Box::from_raw(req));
        } else if tag == REQ_TAG_EMBED {
            let req = base as *mut AiEmbedReq;
            let ctx = (*req).ctx;

            if code == curl::CURLE_OK {
                let mut status: c_long = 0;
                curl::curl_easy_getinfo(easy, curl::CURLINFO_RESPONSE_CODE, &mut status as *mut c_long);
                let raw: &[u8] = &(*req).response_body;

                if status >= 400 {
                    let raw_c = CString::new(raw).unwrap_or_default();
                    sofuu_promise_reject((*req).promise, qjs::sofuu_js_new_string(ctx, raw_c.as_ptr()));
                } else {
                    let result = extract_embeddings(ctx, (*req).provider, raw, (*req).num_inputs);
                    if qjs::is_undefined(result) {
                        sofuu_promise_reject(
                            (*req).promise,
                            qjs::sofuu_js_new_string(ctx, c"Failed to parse embeddings.".as_ptr()),
                        );
                    } else {
                        sofuu_promise_resolve((*req).promise, result);
                        qjs::sofuu_js_free_value(ctx, result);
                    }
                }
            } else {
                let err = qjs::sofuu_js_new_string(ctx, curl::curl_easy_strerror(code));
                sofuu_promise_reject((*req).promise, err);
            }

            if !(*req).headers.is_null() {
                curl::curl_slist_free_all((*req).headers);
            }
            drop(Box::from_raw(req));
        } else if tag == REQ_TAG_STREAM {
            let req = base as *mut AiStreamReq;
            let ctx = (*req).ctx;

            if code != curl::CURLE_OK {
                if qjs::JS_IsFunction(ctx, (*req).error_fn) != 0 {
                    let e = qjs::sofuu_js_new_string(ctx, curl::curl_easy_strerror(code));
                    let r = qjs::JS_Call(ctx, (*req).error_fn, qjs::sofuu_js_undefined(), 1, &e);
                    qjs::sofuu_js_free_value(ctx, e); /* JS_Call does not consume args */
                    qjs::sofuu_js_free_value(ctx, r);
                }
            } else {
                /* A 4xx/5xx must surface as an error, not as a silent empty
                 * success (the COMPLETE/EMBED branches already do this). */
                let mut http_code: c_long = 0;
                curl::curl_easy_getinfo(easy, curl::CURLINFO_RESPONSE_CODE, &mut http_code as *mut c_long);
                if http_code >= 400 {
                    if qjs::JS_IsFunction(ctx, (*req).error_fn) != 0 {
                        let raw = (*req).error_body.clone();
                        let mut detail = String::new();
                        // Try JSON {error:{message,code}} or {message}
                        if !raw.is_empty() {
                            let body_str = String::from_utf8_lossy(&raw);
                            // Lightweight: look for "message" field without a full JSON dep.
                            {
                                let v = extract_json_field(&raw, "message");
                                if !v.is_empty() { /* exercised — real extraction below uses JS parse */ }
                            }
                            // Try to parse as JSON via QuickJS for richer error.message extraction (when JS context is alive).
                            // Fallback: truncated raw body.
                            let parsed_msg = {
                                let c = CString::new(raw.clone()).unwrap_or_default();
                                let v = qjs::JS_ParseJSON(ctx, c.as_ptr(), c.as_bytes().len(), c"<err>".as_ptr());
                                let mut out: Option<String> = None;
                                if !qjs::is_exception(v) {
                                    let err = qjs::sofuu_js_get_property_str(ctx, v, c"error".as_ptr());
                                    if qjs::is_object(err) {
                                        let m = qjs::sofuu_js_get_property_str(ctx, err, c"message".as_ptr());
                                        if let Some(s) = cstr_opt(ctx, m) { if !s.is_empty() { out = Some(s); } }
                                        qjs::sofuu_js_free_value(ctx, m);
                                    }
                                    if out.is_none() {
                                        let m = qjs::sofuu_js_get_property_str(ctx, v, c"message".as_ptr());
                                        if let Some(s) = cstr_opt(ctx, m) { if !s.is_empty() { out = Some(s); } }
                                        qjs::sofuu_js_free_value(ctx, m);
                                    }
                                    qjs::sofuu_js_free_value(ctx, err);
                                } else { qjs::sofuu_js_get_exception(ctx); }
                                qjs::sofuu_js_free_value(ctx, v);
                                out.or_else(|| { let t = body_str.trim(); if t.is_empty() { None } else { Some(t.chars().take(400).collect()) } })
                            };
                            if let Some(m) = parsed_msg { detail = m; }
                        }
                        let mut ebuf = format!("HTTP {}", http_code);
                        if !detail.is_empty() { ebuf.push_str(" — "); ebuf.push_str(&detail); }
                        if http_code == 429 { ebuf.push_str(" (rate limited — retry after a moment or check your quota)"); }
                        else if http_code == 502 || detail.contains("Too many open files") || detail.contains("gateway error") {
                            ebuf.push_str(" (provider gateway overloaded — retry in a moment or try a different model; not a prompt bug)");
                        }
                        let ec = CString::new(ebuf).unwrap_or_default();
                        let e = qjs::sofuu_js_new_string(ctx, ec.as_ptr());
                        let r = qjs::JS_Call(ctx, (*req).error_fn, qjs::sofuu_js_undefined(), 1, &e);
                        qjs::sofuu_js_free_value(ctx, e);
                        qjs::sofuu_js_free_value(ctx, r);
                    }
                } else if (*req).done_called == 0 {
                    /* Ensure done is called even if server omitted [DONE]
                     * (Anthropic ends with message_stop, not [DONE]) — and
                     * carry the usage collected so far, same as the [DONE]
                     * path (calling done_fn with no argument would freeze
                     * the factory's initial {0,0} usage object). */
                    (*req).done_called = 1;
                    if qjs::JS_IsFunction(ctx, (*req).done_fn) != 0 {
                        let stats = qjs::sofuu_js_new_object(ctx);
                        set_stream_stats(ctx, stats, (*req).prompt_tokens, (*req).completion_tokens,
                                         (*req).cache_read_tokens, (*req).cache_write_tokens);
                        let r = qjs::JS_Call(ctx, (*req).done_fn, qjs::sofuu_js_undefined(), 1, &stats);
                        if qjs::is_exception(r) {
                            qjs::js_std_dump_error(ctx);
                        }
                        qjs::sofuu_js_free_value(ctx, r);
                        qjs::sofuu_js_free_value(ctx, stats);
                    }
                }
            }

            stream_req_destroy(ctx, req);
        }

        curl::curl_multi_remove_handle(g_multi(), easy);
        curl::curl_easy_cleanup(easy);
    }
}

/* ------------------------------------------------------------------ */
/* Argument parsing (JS → config structs)                              */
/* ------------------------------------------------------------------ */

/// # Safety
/// `ctx` live; `v` a live value; returns None when ToCString yields NULL.
unsafe fn cstr_opt(ctx: *mut JSContext, v: JSValueConst) -> Option<String> {
    let s = qjs::sofuu_js_to_cstring(ctx, v);
    if s.is_null() {
        return None;
    }
    let out = CStr::from_ptr(s).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, s);
    Some(out)
}

unsafe fn parse_ai_args(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
    is_stream: bool,
) -> Result<AiRequestConfig, ()> {
    let mut cfg = AiRequestConfig {
        stream: is_stream,
        ..AiRequestConfig::default()
    };

    if argc < 1 {
        qjs::JS_ThrowTypeError(ctx, c"ai.complete/stream: prompt or config object required".as_ptr());
        return Err(());
    }

    let mut opts = qjs::sofuu_js_undefined();
    let mut provider_str: Option<String> = None;

    if qjs::sofuu_js_is_string(*argv) != 0 {
        // Legacy mode: complete("prompt", opts)
        cfg.messages = vec![AiMessage {
            role: Some("user".to_string()),
            content: cstr_opt(ctx, *argv),
            tool_calls: None,
            tool_call_id: None,
        }];

        if argc > 1 && qjs::is_object(*argv.add(1)) {
            opts = *argv.add(1);
        }
    } else if qjs::is_object(*argv) {
        // V2 mode: complete({ messages: [...], ... })
        opts = *argv;
        let msgs = qjs::sofuu_js_get_property_str(ctx, opts, c"messages".as_ptr());
        if qjs::JS_IsArray(ctx, msgs) != 0 {
            let mut len: u32 = 0;
            let len_val = qjs::sofuu_js_get_property_str(ctx, msgs, c"length".as_ptr());
            qjs::sofuu_js_to_uint32(ctx, &mut len, len_val);
            qjs::sofuu_js_free_value(ctx, len_val);

            let mut messages = Vec::with_capacity(len as usize);
            for i in 0..len {
                let msg = qjs::JS_GetPropertyUint32(ctx, msgs, i);
                if qjs::is_object(msg) {
                    let r = qjs::sofuu_js_get_property_str(ctx, msg, c"role".as_ptr());
                    let c = qjs::sofuu_js_get_property_str(ctx, msg, c"content".as_ptr());
                    let role = if qjs::sofuu_js_is_string(r) != 0 { cstr_opt(ctx, r) } else { None };
                    /* content may be legitimately null (assistant tool-call
                     * messages) — coercing JS undefined/null through
                     * JS_ToCString yields the STRING "undefined"/"null",
                     * which corrupts the body (regression fixed 2026-08-16:
                     * plain messages emitted "tool_calls":undefined —
                     * invalid JSON every API rejects). */
                    let content = if qjs::sofuu_js_is_string(c) != 0 { cstr_opt(ctx, c) } else { None };
                    /* tool_calls arrives as an ARRAY from JS — JSON-stringify
                     * it so the body builders can splice it verbatim (a raw
                     * JS_ToCString on an array yields "[object Object]").
                     * Anything else (usually undefined) is NOT a tool-call
                     * message. */
                    let tc = qjs::sofuu_js_get_property_str(ctx, msg, c"tool_calls".as_ptr());
                    let tool_calls = if qjs::JS_IsArray(ctx, tc) != 0 {
                        let json = qjs::JS_JSONStringify(
                            ctx,
                            tc,
                            qjs::sofuu_js_undefined(),
                            qjs::sofuu_js_undefined(),
                        );
                        let s = if qjs::is_undefined(json) {
                            None
                        } else {
                            let s = cstr_opt(ctx, json);
                            qjs::sofuu_js_free_value(ctx, json);
                            s
                        };
                        s
                    } else if qjs::sofuu_js_is_string(tc) != 0 {
                        /* pre-stringified array (defensive) */
                        cstr_opt(ctx, tc)
                    } else {
                        None
                    };
                    let tid = qjs::sofuu_js_get_property_str(ctx, msg, c"tool_call_id".as_ptr());
                    let tool_call_id = if qjs::sofuu_js_is_string(tid) != 0 { cstr_opt(ctx, tid) } else { None };
                    qjs::sofuu_js_free_value(ctx, r);
                    qjs::sofuu_js_free_value(ctx, c);
                    qjs::sofuu_js_free_value(ctx, tc);
                    qjs::sofuu_js_free_value(ctx, tid);
                    messages.push(AiMessage { role, content, tool_calls, tool_call_id });
                } else {
                    messages.push(AiMessage { role: None, content: None, tool_calls: None, tool_call_id: None });
                }
                qjs::sofuu_js_free_value(ctx, msg);
            }
            cfg.messages = messages;
        } else {
            // Handle { prompt: "..." } fallback?
            qjs::JS_ThrowTypeError(
                ctx,
                c"ai.complete/stream: config object must contain a 'messages' array".as_ptr(),
            );
            qjs::sofuu_js_free_value(ctx, msgs);
            return Err(());
        }
        qjs::sofuu_js_free_value(ctx, msgs);
    } else {
        qjs::JS_ThrowTypeError(ctx, c"Expected string prompt or config object".as_ptr());
        return Err(());
    }

    if qjs::is_object(opts) {
        let mut v;
        v = qjs::sofuu_js_get_property_str(ctx, opts, c"provider".as_ptr());
        if !qjs::is_undefined(v) {
            provider_str = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"model".as_ptr());
        if !qjs::is_undefined(v) {
            cfg.model = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"system".as_ptr());
        if !qjs::is_undefined(v) {
            cfg.system_prompt = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"api_key".as_ptr());
        if !qjs::is_undefined(v) {
            cfg.api_key = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"base_url".as_ptr());
        if !qjs::is_undefined(v) {
            cfg.base_url = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"profile".as_ptr());
        if !qjs::is_undefined(v) {
            cfg.profile = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"temperature".as_ptr());
        if qjs::is_number(v) {
            qjs::JS_ToFloat64(ctx, &mut cfg.temperature, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"topP".as_ptr());
        if qjs::is_undefined(v) {
            qjs::sofuu_js_free_value(ctx, v);
            v = qjs::sofuu_js_get_property_str(ctx, opts, c"top_p".as_ptr());
        }
        if qjs::is_number(v) {
            qjs::JS_ToFloat64(ctx, &mut cfg.top_p, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"maxTokens".as_ptr());
        if qjs::is_undefined(v) {
            qjs::sofuu_js_free_value(ctx, v);
            v = qjs::sofuu_js_get_property_str(ctx, opts, c"max_tokens".as_ptr());
        }
        if qjs::is_number(v) {
            qjs::JS_ToInt32(ctx, &mut cfg.max_tokens, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        /* JSON structured output: accept "responseFormat", "response_format", or shorthand "json" */
        v = qjs::sofuu_js_get_property_str(ctx, opts, c"responseFormat".as_ptr());
        if qjs::is_undefined(v) {
            qjs::sofuu_js_free_value(ctx, v);
            v = qjs::sofuu_js_get_property_str(ctx, opts, c"response_format".as_ptr());
        }
        if qjs::sofuu_js_is_string(v) != 0 {
            cfg.response_format = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        /* timeout: milliseconds, 0 = unlimited */
        v = qjs::sofuu_js_get_property_str(ctx, opts, c"timeout".as_ptr());
        if qjs::is_number(v) {
            let mut tms: f64 = 0.0;
            qjs::JS_ToFloat64(ctx, &mut tms, v);
            cfg.timeout_ms = if tms > 0.0 { tms as c_long } else { 0 };
        }
        qjs::sofuu_js_free_value(ctx, v);

        /* reasoning effort: "effort" or "reasoning_effort" */
        v = qjs::sofuu_js_get_property_str(ctx, opts, c"effort".as_ptr());
        if qjs::is_undefined(v) {
            qjs::sofuu_js_free_value(ctx, v);
            v = qjs::sofuu_js_get_property_str(ctx, opts, c"reasoning_effort".as_ptr());
        }
        if qjs::sofuu_js_is_string(v) != 0 {
            cfg.effort = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"tools".as_ptr());
        if qjs::JS_IsArray(ctx, v) != 0 {
            let mut len: u32 = 0;
            let len_val = qjs::sofuu_js_get_property_str(ctx, v, c"length".as_ptr());
            qjs::sofuu_js_to_uint32(ctx, &mut len, len_val);
            qjs::sofuu_js_free_value(ctx, len_val);

            let mut tools = Vec::with_capacity(len as usize);
            for i in 0..len {
                let t = qjs::JS_GetPropertyUint32(ctx, v, i);
                if qjs::is_object(t) {
                    let name = qjs::sofuu_js_get_property_str(ctx, t, c"name".as_ptr());
                    let desc = qjs::sofuu_js_get_property_str(ctx, t, c"description".as_ptr());
                    let params = qjs::sofuu_js_get_property_str(ctx, t, c"parameters".as_ptr());

                    let name_s = cstr_opt(ctx, name);
                    let desc_s = cstr_opt(ctx, desc);
                    let mut parameters_json: Option<String> = None;

                    if qjs::is_object(params) {
                        let json = qjs::JS_JSONStringify(
                            ctx,
                            params,
                            qjs::sofuu_js_undefined(),
                            qjs::sofuu_js_undefined(),
                        );
                        if !qjs::is_undefined(json) {
                            parameters_json = cstr_opt(ctx, json);
                            qjs::sofuu_js_free_value(ctx, json);
                        } else {
                            parameters_json = Some("{}".to_string());
                            qjs::sofuu_js_free_value(ctx, json);
                        }
                    }

                    qjs::sofuu_js_free_value(ctx, name);
                    qjs::sofuu_js_free_value(ctx, desc);
                    qjs::sofuu_js_free_value(ctx, params);
                    tools.push(AiToolDef {
                        name: name_s,
                        description: desc_s,
                        parameters_json,
                    });
                } else {
                    tools.push(AiToolDef::default());
                }
                qjs::sofuu_js_free_value(ctx, t);
            }
            cfg.tools = tools;
        }
        qjs::sofuu_js_free_value(ctx, v);
    }

    cfg.provider = parse_provider(provider_str.as_deref());
    /* No model substitution: missing model surfaces as an actionable error
     * at the call sites (see require_model). */

    Ok(cfg)
}

unsafe fn parse_embed_args(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
) -> Result<AiEmbedConfig, ()> {
    let mut cfg = AiEmbedConfig {
        provider: Provider::OpenAi,
        model: None,
        api_key: None,
        base_url: None,
        inputs: Vec::new(),
    };

    if argc < 1 {
        qjs::JS_ThrowTypeError(ctx, c"ai.embed: input string or array required".as_ptr());
        return Err(());
    }

    if qjs::sofuu_js_is_string(*argv) != 0 {
        cfg.inputs.push(cstr_opt(ctx, *argv).unwrap_or_default());
    } else if qjs::JS_IsArray(ctx, *argv) != 0 {
        let mut len: u32 = 0;
        let len_val = qjs::sofuu_js_get_property_str(ctx, *argv, c"length".as_ptr());
        qjs::sofuu_js_to_uint32(ctx, &mut len, len_val);
        qjs::sofuu_js_free_value(ctx, len_val);
        for i in 0..len {
            let item = qjs::JS_GetPropertyUint32(ctx, *argv, i);
            cfg.inputs.push(cstr_opt(ctx, item).unwrap_or_default());
            qjs::sofuu_js_free_value(ctx, item);
        }
    } else {
        qjs::JS_ThrowTypeError(
            ctx,
            c"ai.embed: first argument must be string or string array".as_ptr(),
        );
        return Err(());
    }

    let mut provider_str: Option<String> = None;
    if argc > 1 && qjs::is_object(*argv.add(1)) {
        let opts = *argv.add(1);
        let mut v = qjs::sofuu_js_get_property_str(ctx, opts, c"provider".as_ptr());
        if !qjs::is_undefined(v) {
            provider_str = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"model".as_ptr());
        if !qjs::is_undefined(v) {
            cfg.model = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"api_key".as_ptr());
        if !qjs::is_undefined(v) {
            cfg.api_key = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);

        v = qjs::sofuu_js_get_property_str(ctx, opts, c"base_url".as_ptr());
        if !qjs::is_undefined(v) {
            cfg.base_url = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);
    }

    cfg.provider = parse_provider(provider_str.as_deref());
    /* No model substitution: missing model is an actionable error at
     * js_ai_embed (see require_model). */

    Ok(cfg)
}

/* Empty/absent model in a request is an ERROR, never a silent substitution
 * ("no default model" design rule — the user picks provider+model once, in
 * chat via /provider or headless via an explicit opts.model). */
const NO_MODEL_MSG: &std::ffi::CStr =
    c"No model configured — set one via /model or config (~/.sofuu/config.json), e.g. model: \"<name>\"";

/// True when `m` is a usable model name (present and non-empty).
fn require_model(m: Option<&str>) -> bool {
    matches!(m, Some(m) if !m.trim().is_empty())
}

/* ------------------------------------------------------------------ */
/* JS: sofuu.ai.complete(prompt, opts?)  → Promise<{text}>             */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ai_complete(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let cfg = match parse_ai_args(ctx, argc, argv, false) {
        Ok(c) => c,
        Err(()) => return qjs::sofuu_js_exception(),
    };

    let eff = effective_provider(&cfg);
    if !require_model(cfg.model.as_deref()) {
        return qjs::JS_ThrowTypeError(ctx, NO_MODEL_MSG.as_ptr());
    }
    if cfg.provider == Provider::Local || eff == Provider::Local {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"provider \"local\" is not available yet: local inference is roadmap \
              Track B (llama.cpp backend / custom engine). Use a remote provider \
              (openai, anthropic, local) or a custom endpoint."
                .as_ptr(),
        );
    }
    if cfg.provider == Provider::Custom
        && !matches!(cfg.base_url.as_deref(), Some(u) if !u.is_empty())
    {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"custom provider needs a base URL (use /provider wizard or /baseurl)".as_ptr(),
        );
    }
    let api_key: Option<String> = resolve_api_key(eff, cfg.api_key.as_deref());
    /* A base_url override (self-hosted endpoints) wins over the provider's
     * built-in endpoint. Normalize duplicated /chat/completions suffix — the
     * config can re-corrupt if an old binary re-saves while editing. */
    let url: String = {
        let raw = match cfg.base_url.as_deref() {
            Some(u) if !u.is_empty() => u.to_string(),
            _ => provider_api_url(eff, cfg.model.as_deref()),
        };
        if raw.matches("/chat/completions").count() > 1 {
            let first = raw.find("/chat/completions").unwrap();
            raw[..first + "/chat/completions".len()].to_string()
        } else { raw }
    };

    let mut req = Box::new(AiCompleteReq {
        tag: REQ_TAG_COMPLETE,
        ctx,
        promise: ptr::null_mut(),
        provider: eff,
        response_body: Vec::new(),
        headers: build_headers(eff, api_key.as_deref()),
        post_body: Some(CString::new(build_request_body(&cfg)).unwrap_or_default()),
        easy: ptr::null_mut(),
    });

    let easy = curl::curl_easy_init();
    req.easy = easy;
    let url_c = CString::new(url).unwrap_or_default();
    let body = req.post_body.as_ref().unwrap();
    curl::curl_easy_setopt(easy, curl::CURLOPT_URL, url_c.as_ptr());
    curl::curl_easy_setopt(easy, curl::CURLOPT_POSTFIELDS, body.as_ptr());
    curl::curl_easy_setopt(easy, curl::CURLOPT_POSTFIELDSIZE, body.as_bytes().len() as c_long);
    curl::curl_easy_setopt(easy, curl::CURLOPT_HTTPHEADER, req.headers);
    curl::curl_easy_setopt(easy, curl::CURLOPT_WRITEFUNCTION, complete_write_cb as *const c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_WRITEDATA, &mut *req as *mut AiCompleteReq as *mut c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_PRIVATE, &mut *req as *mut AiCompleteReq as *mut c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_SSL_VERIFYPEER, 1 as c_long);
    /* A remote endpoint must never hang the CLI forever: default to a
     * generous 120s request timeout (matching the C original) so long
     * reasoning/thinking streams aren't killed mid-thought. Callers may
     * set opts.timeout_ms to raise/lower it. */
    curl::curl_easy_setopt(
        easy,
        curl::CURLOPT_TIMEOUT_MS,
        if cfg.timeout_ms > 0 { cfg.timeout_ms } else { 120000 },
    );
    curl::curl_easy_setopt(easy, curl::CURLOPT_MAXFILESIZE, AI_MAX_RESPONSE as c_long);

    // SAFETY: promise handle written into our still-owned Box.
    let promise = sofuu_promise_new(ctx, &mut req.promise);
    let req_ptr = Box::into_raw(req);

    ai_ensure_init();
    let mcode = curl::curl_multi_add_handle(g_multi(), (*req_ptr).easy);
    if mcode != curl::CURLM_OK {
        /* e.g. CURLM_RECURSIVE_API_CALL when ai.complete runs from inside a
         * streaming write-callback flush — fail loudly instead of hanging
         * forever and leaking the request. (Mirrors the embed guard.) */
        if !(*req_ptr).headers.is_null() {
            curl::curl_slist_free_all((*req_ptr).headers);
        }
        curl::curl_easy_cleanup((*req_ptr).easy);
        let err = qjs::sofuu_js_new_string(ctx, curl::curl_multi_strerror(mcode));
        sofuu_promise_reject((*req_ptr).promise, err);
        drop(Box::from_raw(req_ptr));
        return promise;
    }

    let mut running: c_int = 0;
    curl::curl_multi_socket_action(g_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running);
    ai_check_multi();

    promise
}

/* ------------------------------------------------------------------ */
/* JS: sofuu.ai.stream(prompt, opts?)  → AsyncIterator<{text}>         */
/* ------------------------------------------------------------------ */

const STREAM_FACTORY: &str = r#"(function() {
  var queue = [];
  var _resolve = null, _done = false, _error = null, _aborted = false;
  var _usage = { promptTokens: 0, completionTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 };
  var iterator = {
    [Symbol.asyncIterator]: function() { return this; },
    get usage() { return _usage; },
    get aborted() { return _aborted; },
    /* Stop the in-flight request (Esc in the chat). _sid is attached by C
     * before the iterator is handed out; __ai_abort tears the stream down
     * and ends this iterator with done:true. */
    abort: function() {
      _aborted = true;
      if (typeof __ai_abort === 'function') __ai_abort(iterator._sid);
    },
    next: function() {
      if (queue.length > 0) {
        return Promise.resolve({ value: queue.shift(), done: false });
      }
      if (_done)  return Promise.resolve({ value: undefined, done: true });
      if (_error) return Promise.reject(_error);
      return new Promise(function(res, rej) {
        _resolve = { res: res, rej: rej };
      });
    }
  };
  var _think = '';
  var _toolCalls = [];
  function setThink(t) { _think = t; }
  function setToolCalls(tc) { if (Array.isArray(tc)) _toolCalls = _toolCalls.concat(tc); }
  function push(text) {
    const value = { text: text, think: _think, done: false };
    _think = '';
    if (_resolve) { var r = _resolve; _resolve = null; r.res({ value: value }); }
    else { queue.push(value); }
  }
  function done(stats) {
    if (stats) { _usage = stats; }
    _done = true;
    if (_resolve) { var r = _resolve; _resolve = null; r.res({ value: undefined, done: true }); }
  }
  function error(msg) {
    _error = new Error(msg);
    if (_resolve) { var r = _resolve; _resolve = null; r.rej(_error); }
  }
  Object.defineProperty(iterator, 'toolCalls', { get: function() { return _toolCalls; } });
  return { push: push, done: done, error: error, setThink: setThink, setToolCalls: setToolCalls, iterator: iterator };
})()"#;

/// JS: __ai_abort(sid?) — stop the given stream, or ALL active streams.
/// Wired to iterator.abort() per stream; the chat's Esc key calls it too.
unsafe extern "C" fn js_ai_abort(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut sid: i64 = -1;
    if argc >= 1 {
        qjs::JS_ToInt64(ctx, &mut sid, *argv);
    }
    let mut r = G_STREAMS.with(|s| s.get());
    while !r.is_null() {
        let next = (*r).next;
        if sid < 0 || (*r).sid == sid {
            stream_abort_one(r);
        }
        r = next;
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_ai_stream(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let cfg = match parse_ai_args(ctx, argc, argv, true) {
        Ok(c) => c,
        Err(()) => return qjs::sofuu_js_exception(),
    };

    let eff = effective_provider(&cfg);
    if !require_model(cfg.model.as_deref()) {
        return qjs::JS_ThrowTypeError(ctx, NO_MODEL_MSG.as_ptr());
    }
    if cfg.provider == Provider::Local || eff == Provider::Local {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"provider \"local\" is not available yet: local inference is roadmap \
              Track B (llama.cpp backend / custom engine). Use a remote provider \
              (openai, anthropic, local) or a custom endpoint."
                .as_ptr(),
        );
    }
    if cfg.provider == Provider::Custom
        && !matches!(cfg.base_url.as_deref(), Some(u) if !u.is_empty())
    {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"custom provider needs a base URL (use /provider wizard or /baseurl)".as_ptr(),
        );
    }
    let api_key: Option<String> = resolve_api_key(eff, cfg.api_key.as_deref());
    let url: String = {
        let raw = match cfg.base_url.as_deref() {
            Some(u) if !u.is_empty() => u.to_string(),
            _ => provider_api_url(eff, cfg.model.as_deref()),
        };
        if raw.matches("/chat/completions").count() > 1 {
            let first = raw.find("/chat/completions").unwrap();
            raw[..first + "/chat/completions".len()].to_string()
        } else { raw }
    };

    let factory_c = CString::new(STREAM_FACTORY).unwrap();
    let factory = qjs::JS_Eval(
        ctx,
        factory_c.as_ptr(),
        factory_c.as_bytes().len(),
        c"<ai.stream>".as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL,
    );
    if qjs::is_exception(factory) {
        return qjs::sofuu_js_exception();
    }

    let push_fn = qjs::sofuu_js_get_property_str(ctx, factory, c"push".as_ptr());
    let done_fn = qjs::sofuu_js_get_property_str(ctx, factory, c"done".as_ptr());
    let error_fn = qjs::sofuu_js_get_property_str(ctx, factory, c"error".as_ptr());
    let think_fn = qjs::sofuu_js_get_property_str(ctx, factory, c"setThink".as_ptr());
    let tool_calls_fn = qjs::sofuu_js_get_property_str(ctx, factory, c"setToolCalls".as_ptr());
    let iterator = qjs::sofuu_js_get_property_str(ctx, factory, c"iterator".as_ptr());
    qjs::sofuu_js_free_value(ctx, factory);

    let sid = G_STREAM_SID.with(|s| {
        let v = s.get() + 1;
        s.set(v);
        v
    });
    let mut req = Box::new(AiStreamReq {
        tag: REQ_TAG_STREAM,
        ctx,
        provider: eff,
        push_fn: qjs::sofuu_js_dup_value(ctx, push_fn),
        done_fn: qjs::sofuu_js_dup_value(ctx, done_fn),
        error_fn: qjs::sofuu_js_dup_value(ctx, error_fn),
        think_fn: qjs::sofuu_js_dup_value(ctx, think_fn),
        tool_calls_fn: qjs::sofuu_js_dup_value(ctx, tool_calls_fn),
        sse_buf: Vec::new(),
        headers: build_headers(eff, api_key.as_deref()),
        post_body: Some(CString::new(build_request_body(&cfg)).unwrap_or_default()),
        easy: ptr::null_mut(),
        prompt_tokens: 0,
        completion_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        done_called: 0,
        error_body: Vec::new(),
        sid,
        next: ptr::null_mut(),
    });

    qjs::sofuu_js_free_value(ctx, push_fn);
    qjs::sofuu_js_free_value(ctx, done_fn);
    qjs::sofuu_js_free_value(ctx, error_fn);
    qjs::sofuu_js_free_value(ctx, think_fn);

    let easy = curl::curl_easy_init();
    req.easy = easy;
    let url_c = CString::new(url).unwrap_or_default();
    let body = req.post_body.as_ref().unwrap();
    curl::curl_easy_setopt(easy, curl::CURLOPT_URL, url_c.as_ptr());
    curl::curl_easy_setopt(easy, curl::CURLOPT_POSTFIELDS, body.as_ptr());
    curl::curl_easy_setopt(easy, curl::CURLOPT_POSTFIELDSIZE, body.as_bytes().len() as c_long);
    curl::curl_easy_setopt(easy, curl::CURLOPT_HTTPHEADER, req.headers);
    curl::curl_easy_setopt(easy, curl::CURLOPT_WRITEFUNCTION, stream_write_cb as *const c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_WRITEDATA, &mut *req as *mut AiStreamReq as *mut c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_PRIVATE, &mut *req as *mut AiStreamReq as *mut c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_SSL_VERIFYPEER, 1 as c_long);
    /* A remote endpoint must never hang the CLI forever: default to a
     * generous 120s request timeout (matching the C original) so long
     * reasoning/thinking streams aren't killed mid-thought. Callers may
     * set opts.timeout_ms to raise/lower it. */
    curl::curl_easy_setopt(
        easy,
        curl::CURLOPT_TIMEOUT_MS,
        if cfg.timeout_ms > 0 { cfg.timeout_ms } else { 120000 },
    );
    curl::curl_easy_setopt(easy, curl::CURLOPT_MAXFILESIZE, AI_MAX_RESPONSE as c_long);

    ai_ensure_init();
    let mcode = curl::curl_multi_add_handle(g_multi(), req.easy);
    if mcode != curl::CURLM_OK {
        /* Same re-entrancy guard as ai.complete — reject the request
         * instead of silently never delivering anything. */
        curl::curl_easy_cleanup(req.easy);
        if !req.headers.is_null() {
            curl::curl_slist_free_all(req.headers);
        }
        qjs::sofuu_js_free_value(ctx, req.push_fn);
        qjs::sofuu_js_free_value(ctx, req.done_fn);
        qjs::sofuu_js_free_value(ctx, req.error_fn);
        qjs::sofuu_js_free_value(ctx, req.think_fn);
        drop(req);
        return qjs::sofuu_js_exception();
    }

    /* Abortable: Esc/Ctrl-C in the chat can now find and stop this stream. */
    let req_ptr = Box::into_raw(req);
    (*req_ptr).next = G_STREAMS.with(|s| s.get());
    G_STREAMS.with(|s| s.set(req_ptr));
    qjs::sofuu_js_set_property_str(ctx, iterator, c"_sid".as_ptr(), qjs::sofuu_js_new_int64(ctx, sid));

    let mut running: c_int = 0;
    curl::curl_multi_socket_action(g_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running);
    ai_check_multi();

    iterator
}

/* ------------------------------------------------------------------ */
/* JS: sofuu.ai.embed(input, opts?)  → Promise<Float32Array|Array>     */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ai_embed(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let cfg = match parse_embed_args(ctx, argc, argv) {
        Ok(c) => c,
        Err(()) => return qjs::sofuu_js_exception(),
    };

    if !require_model(cfg.model.as_deref()) {
        return qjs::JS_ThrowTypeError(ctx, NO_MODEL_MSG.as_ptr());
    }
    if cfg.provider == Provider::Local {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"provider \"local\" is not available yet: local inference is roadmap \
              Track B. Use embedLocal() for offline embeddings, or a remote provider."
                .as_ptr(),
        );
    }
    if cfg.provider == Provider::Custom
        && !matches!(cfg.base_url.as_deref(), Some(u) if !u.is_empty())
    {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"custom provider needs a base URL (use /provider wizard or /baseurl)".as_ptr(),
        );
    }
    if cfg.provider == Provider::Anthropic {
        qjs::JS_ThrowTypeError(
            ctx,
            c"ai.embed: Anthropic does not natively provide standard generic embeddings APIs directly via the platform.".as_ptr(),
        );
        return qjs::sofuu_js_exception();
    }

    let mut req = Box::new(AiEmbedReq {
        tag: REQ_TAG_EMBED,
        ctx,
        promise: ptr::null_mut(),
        provider: cfg.provider,
        response_body: Vec::new(),
        headers: build_headers(
            cfg.provider,
            resolve_api_key(cfg.provider, cfg.api_key.as_deref()).as_deref(),
        ),
        post_body: Some(CString::new(build_embed_body(&cfg)).unwrap_or_default()),
        easy: ptr::null_mut(),
        num_inputs: cfg.inputs.len(),
    });

    let mut url: String = if cfg.provider == Provider::OpenAi {
        "https://api.openai.com/v1/embeddings".to_string()
    } else if cfg.provider == Provider::Local {
        "http://127.0.0.1:11434/api/embed".to_string()
    } else {
        provider_api_url(cfg.provider, cfg.model.as_deref())
    };

    /* A base_url override beats the provider's built-in endpoint. For a
     * custom OpenAI-compatible endpoint the stored base_url is the CHAT
     * URL (…/chat/completions) — derive the embeddings URL from it so the
     * request goes to …/embeddings instead of posting an embed body to the
     * chat endpoint. */
    if let Some(bu) = cfg.base_url.as_deref() {
        if !bu.is_empty() {
            url = if bu.ends_with("/chat/completions") {
                format!("{}/embeddings", bu.trim_end_matches("/chat/completions"))
            } else {
                bu.to_string()
            };
        }
    }

    let easy = curl::curl_easy_init();
    req.easy = easy;
    let url_c = CString::new(url).unwrap_or_default();
    let body = req.post_body.as_ref().unwrap();
    curl::curl_easy_setopt(easy, curl::CURLOPT_URL, url_c.as_ptr());
    curl::curl_easy_setopt(easy, curl::CURLOPT_POSTFIELDS, body.as_ptr());
    curl::curl_easy_setopt(easy, curl::CURLOPT_POSTFIELDSIZE, body.as_bytes().len() as c_long);
    curl::curl_easy_setopt(easy, curl::CURLOPT_HTTPHEADER, req.headers);
    curl::curl_easy_setopt(easy, curl::CURLOPT_WRITEFUNCTION, complete_write_cb as *const c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_WRITEDATA, &mut *req as *mut AiEmbedReq as *mut c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_PRIVATE, &mut *req as *mut AiEmbedReq as *mut c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_SSL_VERIFYPEER, 1 as c_long);
    curl::curl_easy_setopt(easy, curl::CURLOPT_TIMEOUT_MS, 0 as c_long); /* embeddings may be slow */
    curl::curl_easy_setopt(easy, curl::CURLOPT_MAXFILESIZE, AI_MAX_RESPONSE as c_long);

    // SAFETY: promise handle written into our still-owned Box.
    let promise = sofuu_promise_new(ctx, &mut req.promise);
    let req_ptr = Box::into_raw(req);

    ai_ensure_init();
    let mcode = curl::curl_multi_add_handle(g_multi(), (*req_ptr).easy);
    if mcode != curl::CURLM_OK {
        /* Should not happen after the M4 recursion fix, but guard anyway.
         * Reject immediately so the awaiting JS coroutine resumes. */
        curl::curl_easy_cleanup((*req_ptr).easy);
        if !(*req_ptr).headers.is_null() {
            curl::curl_slist_free_all((*req_ptr).headers);
        }
        let err = qjs::sofuu_js_new_string(ctx, curl::curl_multi_strerror(mcode));
        sofuu_promise_reject((*req_ptr).promise, err);
        drop(Box::from_raw(req_ptr));
        return promise;
    }

    let mut running: c_int = 0;
    curl::curl_multi_socket_action(g_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running);
    ai_check_multi();

    promise
}

/* ------------------------------------------------------------------ */
/* SIMD Vector JS Bindings                                              */
/* ------------------------------------------------------------------ */

/// # Safety
/// `val` a live JS value of `ctx`; returns a copied float vector or None.
unsafe fn extract_f32(ctx: *mut JSContext, val: JSValueConst) -> Option<Vec<f32>> {
    let mut byte_offset: usize = 0;
    let mut byte_length: usize = 0;
    let buf = qjs::JS_GetTypedArrayBuffer(ctx, val, &mut byte_offset, &mut byte_length, ptr::null_mut());
    if !qjs::is_exception(buf) {
        let mut buf_size: usize = 0;
        let ptr = qjs::JS_GetArrayBuffer(ctx, &mut buf_size, buf);
        qjs::sofuu_js_free_value(ctx, buf);
        if ptr.is_null() {
            return None;
        }
        let n = byte_length / std::mem::size_of::<f32>();
        let mut out = vec![0.0f32; n];
        std::ptr::copy_nonoverlapping(ptr.add(byte_offset) as *const f32, out.as_mut_ptr(), n);
        return Some(out);
    }
    qjs::sofuu_js_free_value(ctx, buf);
    let len_v = qjs::sofuu_js_get_property_str(ctx, val, c"length".as_ptr());
    let mut len32: u32 = 0;
    qjs::sofuu_js_to_uint32(ctx, &mut len32, len_v);
    qjs::sofuu_js_free_value(ctx, len_v);
    let mut out = vec![0.0f32; len32 as usize];
    for i in 0..len32 {
        let el = qjs::JS_GetPropertyUint32(ctx, val, i);
        let mut d: f64 = 0.0;
        qjs::JS_ToFloat64(ctx, &mut d, el);
        qjs::sofuu_js_free_value(ctx, el);
        out[i as usize] = d as f32;
    }
    Some(out)
}

unsafe extern "C" fn js_ai_similarity(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::JS_ThrowTypeError(ctx, c"similarity(a,b) requires 2 vectors".as_ptr());
    }
    let a = extract_f32(ctx, *argv);
    let b = extract_f32(ctx, *argv.add(1));
    let (Some(a), Some(b)) = (a, b) else {
        return qjs::JS_ThrowTypeError(ctx, c"vectors must be same length".as_ptr());
    };
    if a.len() != b.len() {
        return qjs::JS_ThrowTypeError(ctx, c"vectors must be same length".as_ptr());
    }
    // SAFETY: a/b live for the call; n matches their length.
    let r = sofuu_cosine_f32(a.as_ptr(), b.as_ptr(), a.len());
    qjs::sofuu_js_new_float64(ctx, r as f64)
}

unsafe extern "C" fn js_ai_dot(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::JS_ThrowTypeError(ctx, c"dot(a,b) requires 2 vectors".as_ptr());
    }
    let a = extract_f32(ctx, *argv);
    let b = extract_f32(ctx, *argv.add(1));
    let (Some(a), Some(b)) = (a, b) else {
        return qjs::JS_ThrowTypeError(ctx, c"same length required".as_ptr());
    };
    if a.len() != b.len() {
        return qjs::JS_ThrowTypeError(ctx, c"same length required".as_ptr());
    }
    // SAFETY: a/b live for the call; n matches their length.
    let r = sofuu_dot_f32(a.as_ptr(), b.as_ptr(), a.len());
    qjs::sofuu_js_new_float64(ctx, r as f64)
}

unsafe extern "C" fn js_ai_l2(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::JS_ThrowTypeError(ctx, c"l2(a,b) requires 2 vectors".as_ptr());
    }
    let a = extract_f32(ctx, *argv);
    let b = extract_f32(ctx, *argv.add(1));
    let (Some(a), Some(b)) = (a, b) else {
        return qjs::JS_ThrowTypeError(ctx, c"same length required".as_ptr());
    };
    if a.len() != b.len() {
        return qjs::JS_ThrowTypeError(ctx, c"same length required".as_ptr());
    }
    // SAFETY: a/b live for the call; n matches their length.
    let r = sofuu_l2_f32(a.as_ptr(), b.as_ptr(), a.len());
    qjs::sofuu_js_new_float64(ctx, r as f64)
}

unsafe extern "C" fn js_ai_estimate_tokens(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"ai.estimateTokens expected string or array".as_ptr());
    }
    let mut total: i32 = 0;

    if qjs::sofuu_js_is_string(*argv) != 0 {
        if let Some(s) = cstr_opt(ctx, *argv) {
            total = (s.len() / 4) as i32;
        }
    } else if qjs::JS_IsArray(ctx, *argv) != 0 {
        let mut len: u32 = 0;
        let len_val = qjs::sofuu_js_get_property_str(ctx, *argv, c"length".as_ptr());
        qjs::sofuu_js_to_uint32(ctx, &mut len, len_val);
        qjs::sofuu_js_free_value(ctx, len_val);
        for i in 0..len {
            let msg = qjs::JS_GetPropertyUint32(ctx, *argv, i);
            if qjs::is_object(msg) {
                let c = qjs::sofuu_js_get_property_str(ctx, msg, c"content".as_ptr());
                if qjs::sofuu_js_is_string(c) != 0 {
                    if let Some(s) = cstr_opt(ctx, c) {
                        total += (s.len() / 4) as i32 + 4; /* ~4 tokens of overhead per message role */
                    }
                }
                qjs::sofuu_js_free_value(ctx, c);
            }
            qjs::sofuu_js_free_value(ctx, msg);
        }
    } else {
        return qjs::JS_ThrowTypeError(ctx, c"ai.estimateTokens expected string or array".as_ptr());
    }
    qjs::sofuu_js_new_int32(ctx, total)
}

/* ------------------------------------------------------------------ */
/* sofuu.ai.embedLocal(text, dim?) → Float32Array                       */
/* Pure-Rust offline TF-IDF fallback (trigrams → MurmurHash3 → L2-norm).*/
/* Returns a unit vector of `dim` floats (default 768).                */
/* ------------------------------------------------------------------ */

const EMBED_LOCAL_DEFAULT_DIM: i32 = 768;

unsafe extern "C" fn js_ai_embed_local(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || qjs::sofuu_js_is_string(*argv) == 0 {
        return qjs::JS_ThrowTypeError(ctx, c"ai.embedLocal(text[, dim]) expected a string".as_ptr());
    }

    let mut dim: i32 = EMBED_LOCAL_DEFAULT_DIM;
    if argc > 1 && qjs::is_number(*argv.add(1)) {
        let mut d: i32 = 0;
        qjs::JS_ToInt32(ctx, &mut d, *argv.add(1));
        if d < 16 {
            return qjs::JS_ThrowRangeError(ctx, c"ai.embedLocal: dim must be >= 16".as_ptr());
        }
        if d > 1 << 20 {
            return qjs::JS_ThrowRangeError(ctx, c"ai.embedLocal: dim too large".as_ptr());
        }
        dim = d;
    }

    let text = match cstr_opt(ctx, *argv) {
        Some(t) => t,
        None => return qjs::sofuu_js_exception(),
    };

    /* Build the Float32Array result */
    let global = qjs::sofuu_js_get_global_object(ctx);
    let f32_ctor = qjs::sofuu_js_get_property_str(ctx, global, c"Float32Array".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    let len_arg = qjs::sofuu_js_new_int32(ctx, dim);
    let vec = qjs::JS_CallConstructor(ctx, f32_ctor, 1, &len_arg);
    qjs::sofuu_js_free_value(ctx, len_arg);
    qjs::sofuu_js_free_value(ctx, f32_ctor);
    if qjs::is_exception(vec) {
        return qjs::sofuu_js_exception();
    }

    /* Fill it from the Rust embedder via the typed-array buffer */
    let mut byte_offset: usize = 0;
    let mut byte_length: usize = 0;
    let buf = qjs::JS_GetTypedArrayBuffer(ctx, vec, &mut byte_offset, &mut byte_length, ptr::null_mut());
    if qjs::is_exception(buf) {
        qjs::sofuu_js_free_value(ctx, vec);
        return qjs::sofuu_js_exception();
    }
    let mut buf_size: usize = 0;
    let ptr = qjs::JS_GetArrayBuffer(ctx, &mut buf_size, buf);
    let ptr = if ptr.is_null() {
        qjs::sofuu_js_free_value(ctx, buf);
        qjs::sofuu_js_free_value(ctx, vec);
        return qjs::JS_ThrowTypeError(ctx, c"ai.embedLocal: cannot access typed-array buffer".as_ptr());
    } else {
        ptr
    };

    /* Fast path — the vector lives in the first (only) buffer segment. */
    if byte_offset + dim as usize * std::mem::size_of::<f32>() <= buf_size {
        let out = std::slice::from_raw_parts_mut(ptr.add(byte_offset) as *mut f32, dim as usize);
        sofuu_tfidf_embed(text.as_bytes(), out, dim as usize);
        qjs::sofuu_js_free_value(ctx, buf);
        return vec;
    }
    qjs::sofuu_js_free_value(ctx, buf);

    /* Slow path — the elements are packed in JS-visible properties. */
    let mut tmp = vec![0.0f32; dim as usize];
    sofuu_tfidf_embed(text.as_bytes(), &mut tmp, dim as usize);
    for (i, &v) in tmp.iter().enumerate() {
        qjs::JS_SetPropertyUint32(ctx, vec, i as u32, qjs::sofuu_js_new_float64(ctx, v as f64));
    }
    vec
}

/* ------------------------------------------------------------------ */
/* Module registration                                                   */
/* ------------------------------------------------------------------ */

thread_local! {
    // JS_CFUNC_DEF(name, length, func): magic=0, u.func = { length, generic, func }.
      static AI_FUNCS: [qjs::JSCFunctionListEntry; 8] = [
        qjs::JSCFunctionListEntry {
            name: c"complete".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_complete },
        },
        qjs::JSCFunctionListEntry {
            name: c"stream".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_stream },
        },
        qjs::JSCFunctionListEntry {
            name: c"embed".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_embed },
        },
        qjs::JSCFunctionListEntry {
            name: c"embedLocal".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_embed_local },
        },
        qjs::JSCFunctionListEntry {
            name: c"similarity".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_similarity },
        },
        qjs::JSCFunctionListEntry {
            name: c"dot".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_dot },
        },
        qjs::JSCFunctionListEntry {
            name: c"l2".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_l2 },
        },
        qjs::JSCFunctionListEntry {
            name: c"estimateTokens".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_ai_estimate_tokens },
        },
    ];
}

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_ai_register(ctx: *mut JSContext) {
    ai_ensure_init();

    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());

    if qjs::is_undefined(sofuu) {
        sofuu = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), qjs::sofuu_js_dup_value(ctx, sofuu));
    }

    let ai_obj = qjs::sofuu_js_new_object(ctx);
    let ai_funcs = AI_FUNCS.with(|f| f.as_ptr());
    qjs::JS_SetPropertyFunctionList(ctx, ai_obj, ai_funcs, AI_FUNCS.with(|f| f.len()) as c_int);
    qjs::sofuu_js_set_property_str(ctx, sofuu, c"ai".as_ptr(), ai_obj);

    /* Stream abort (Esc / Ctrl-C in the chat; iterator.abort() calls this). */
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__ai_abort".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_ai_abort, c"__ai_abort".as_ptr(), 1),
    );

    qjs::sofuu_js_free_value(ctx, sofuu);
    qjs::sofuu_js_free_value(ctx, global);
}

// CURLMOPT_* — FUNCTIONPOINT(20000) + enum index (multi.h: SOCKETFUNCTION=1,
// TIMERFUNCTION=4) — same values verified for the M4 fetch port against the
// system SDK headers.
const CURLMOPT_SOCKETFUNCTION: c_int = 20001;
const CURLMOPT_TIMERFUNCTION: c_int = 20004;

/* ── Tests ───────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ffi::bridge::{register_global_fn, JSCFunction};

    /// Eval `src` on a bare context and return Some(exception-text) when it
    /// throws. JSON/JS strings are NUL-terminated buffers (the QuickJS
    /// reader peeks past input_len — see rlm/js_api.rs).
    fn eval_exc(ctx: *mut JSContext, src: &str) -> Option<String> {
        let c_src = CString::new(src).unwrap();
        unsafe {
            let r = qjs::JS_Eval(
                ctx,
                c_src.as_ptr(),
                src.len(),
                c"<ai-test>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            let exc = if qjs::is_exception(r) {
                let e = qjs::JS_GetException(ctx);
                let p = qjs::sofuu_js_to_cstring(ctx, e);
                let msg = if p.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                };
                if !p.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, p);
                }
                qjs::sofuu_js_free_value(ctx, e);
                Some(msg)
            } else {
                None
            };
            qjs::sofuu_js_free_value(ctx, r);
            exc
        }
    }

    /// "No default model": complete/stream/embed with a missing or empty
    /// model must throw the actionable error (before any network work), and
    /// never substitute llama3/gpt-4o/nomic-embed-text behind the caller's
    /// back.
    #[test]
    fn missing_model_is_an_actionable_error() {
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            assert!(!ctx.is_null());
            // Register only the three entry points; the throw path runs
            // before any curl/loop state is touched.
            register_global_fn(ctx, "__ai_complete", js_ai_complete as JSCFunction);
            register_global_fn(ctx, "__ai_stream", js_ai_stream as JSCFunction);
            register_global_fn(ctx, "__ai_embed", js_ai_embed as JSCFunction);

            for (name, call) in [
                ("complete legacy+no model", r#"__ai_complete("hi")"#),
                ("complete empty model", r#"__ai_complete({ messages: [{ role: "user", content: "hi" }], provider: "openai", model: "" })"#),
                ("complete whitespace model", r#"__ai_complete({ messages: [{ role: "user", content: "hi" }], provider: "ollama", model: "   " })"#),
                ("stream no model", r#"__ai_stream({ messages: [{ role: "user", content: "hi" }], provider: "openai" })"#),
                ("embed no model", r#"__ai_embed("hello", { provider: "ollama" })"#),
            ] {
                let exc = eval_exc(ctx, call)
                    .unwrap_or_else(|| panic!("{name}: must throw, got success"));
                assert!(
                    exc.contains("No model configured"),
                    "{name}: expected the actionable no-model error, got: {exc}"
                );
            }

            // Sanity: WITH a model the no-model error is gone (the request
            // then proceeds to curl — not exercised here).
            let exc = eval_exc(
                ctx,
                r#"__ai_embed("hello", { provider: "anthropic", model: "m" })"#,
            )
            .expect("still throws (anthropic has no embeddings)");
            assert!(exc.contains("does not natively provide"), "{exc}");
            assert!(!exc.contains("No model configured"), "{exc}");

            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }

    /// F4b: Anthropic streamed tool_use events must reshape into OpenAI
    /// delta fragments (index/id/function.name on content_block_start,
    /// concatenated function.arguments on input_json_delta), and non-tool
    /// events (text blocks, message bookkeeping) must yield nothing.
    #[test]
    fn anthropic_tool_fragments_shapes() {
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            assert!(!ctx.is_null());

            let frag_json = |event: &str| -> Option<String> {
                let c = CString::new(event).unwrap_or_default();
                let ev = qjs::JS_ParseJSON(ctx, c.as_ptr(), event.len(), c"<f4b>".as_ptr());
                assert!(!qjs::is_exception(ev), "parse: {event}");
                let out = anthropic_tool_fragments(ctx, ev).map(|arr| {
                    let j = qjs::JS_JSONStringify(
                        ctx,
                        arr,
                        qjs::sofuu_js_undefined(),
                        qjs::sofuu_js_undefined(),
                    );
                    let p = qjs::sofuu_js_to_cstring(ctx, j);
                    let s = if p.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(p).to_string_lossy().into_owned()
                    };
                    if !p.is_null() {
                        qjs::sofuu_js_free_cstring(ctx, p);
                    }
                    qjs::sofuu_js_free_value(ctx, j);
                    qjs::sofuu_js_free_value(ctx, arr);
                    s
                });
                qjs::sofuu_js_free_value(ctx, ev);
                out
            };

            // Tool block opens: id + name, empty arguments.
            assert_eq!(
                frag_json(r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01A","name":"get_weather","input":{}}}"#)
                    .as_deref(),
                Some(r#"[{"index":1,"type":"function","id":"toolu_01A","function":{"name":"get_weather","arguments":""}}]"#)
            );
            // Argument fragments carry only index + arguments (merged by
            // the JS side; key is the shared block index).
            assert_eq!(
                frag_json(r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}"#)
                    .as_deref(),
                Some(r#"[{"index":1,"type":"function","function":{"arguments":"{\"city\":\"Paris\"}"}}]"#)
            );
            // Non-tool events yield nothing.
            assert_eq!(frag_json(r#"{"type":"message_start","message":{}}"#), None);
            assert_eq!(
                frag_json(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#),
                None
            );
            assert_eq!(
                frag_json(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#),
                None
            );
            assert_eq!(frag_json(r#"{"type":"content_block_stop","index":1}"#), None);

            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }
    #[test]
    fn tool_call_roundtrip_serialized_on_both_formats() {
        let tc_json = r#"[{"id":"call_1","type":"function","function":{"name":"mock_tool","arguments":"{\"x\":1}"}}]"#;
        let msgs = vec![
            AiMessage {
                role: Some("assistant".into()),
                content: None,
                tool_calls: Some(tc_json.to_string()),
                tool_call_id: None,
            },
            AiMessage {
                role: Some("tool".into()),
                content: Some("tool result".into()),
                tool_calls: None,
                tool_call_id: Some("call_1".into()),
            },
        ];
        let mk = |provider: Provider| AiRequestConfig {
            provider,
            model: Some("m".into()),
            api_key: None,
            base_url: None,
            profile: None,
            system_prompt: None,
            response_format: None,
            effort: None,
            max_tokens: 0,
            temperature: -1.0,
            top_p: -1.0,
            stream: false,
            timeout_ms: 0,
            messages: msgs.clone(),
            tools: vec![],
        };

        // OpenAI-compatible
        let oa = build_openai_body_v2(&mk(Provider::OpenAi));
        assert!(oa.contains("\"tool_calls\":["), "openai body must carry tool_calls: {oa}");
        assert!(oa.contains("\"tool_call_id\":\"call_1\""), "openai body must carry tool_call_id: {oa}");
        assert!(oa.contains("mock_tool"), "openai body must carry the tool name: {oa}");

        // Anthropic
        let an = build_anthropic_body_v2(&mk(Provider::Anthropic));
        assert!(an.contains("\"type\":\"tool_use\""), "anthropic body must carry tool_use: {an}");
        assert!(an.contains("\"type\":\"tool_result\""), "anthropic body must carry tool_result: {an}");
        assert!(an.contains("call_1"), "anthropic body must carry the tool id: {an}");
        assert!(an.contains("mock_tool"), "anthropic body must carry the tool name: {an}");
    }

    /// P6 (PLAN-MEMORY-TOKENS): the anthropic wire carries the stable
    /// prefix as a cacheable system content block + a marker on the LAST
    /// tool; a LEADING system message is hoisted into that block (the
    /// agent/chat drivers send OpenAI-shaped messages); the OpenAI wire
    /// stays byte-shape unchanged (no cache_control — automatic prefix
    /// caching needs nothing).
    #[test]
    fn anthropic_prefix_cache_markers_and_system_hoist() {
        let mk = |provider: Provider| AiRequestConfig {
            provider,
            model: Some("m".into()),
            api_key: None,
            base_url: None,
            profile: None,
            system_prompt: None,
            response_format: None,
            effort: None,
            max_tokens: 0,
            temperature: -1.0,
            top_p: -1.0,
            stream: false,
            timeout_ms: 0,
            messages: vec![
                AiMessage {
                    role: Some("system".into()),
                    content: Some("Sofuu coding agent.".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                AiMessage {
                    role: Some("user".into()),
                    content: Some("hello".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            tools: vec![
                AiToolDef {
                    name: Some("first_tool".into()),
                    description: Some("a".into()),
                    parameters_json: Some("{}".into()),
                },
                AiToolDef {
                    name: Some("last_tool".into()),
                    description: Some("b".into()),
                    parameters_json: Some("{}".into()),
                },
            ],
        };

        let an = build_anthropic_body_v2(&mk(Provider::Anthropic));
        // System hoisted into a single cacheable block…
        assert!(an.contains(
            "\"system\":[{\"type\":\"text\",\"text\":\"Sofuu coding agent.\",\"cache_control\":{\"type\":\"ephemeral\"}}]"
        ), "system block with marker: {an}");
        // …exactly once, and NOT duplicated as a user-role turn.
        assert_eq!(an.matches("\"cache_control\"").count(), 2, "one system marker + one last-tool marker: {an}");
        assert!(!an.contains("\"role\":\"system\""), "leading system hoisted out of messages: {an}");
        assert!(an.contains("\"content\":\"hello\""), "user message intact: {an}");
        // Marker only on the LAST tool.
        let last_pos = an.find("last_tool").unwrap();
        let first_pos = an.find("first_tool").unwrap();
        let marker_after_last = an[last_pos..].contains("\"cache_control\"");
        let marker_between = an[first_pos..last_pos].contains("\"cache_control\"");
        assert!(marker_after_last && !marker_between, "marker on last tool only: {an}");

        // Explicit opts.system wins over the leading messages[0] hoist.
        let mut explicit = mk(Provider::Anthropic);
        explicit.system_prompt = Some("explicit wins".into());
        let an2 = build_anthropic_body_v2(&explicit);
        assert!(an2.contains("\"text\":\"explicit wins\""), "{an2}");

        // OpenAI wire: untouched by P6.
        let oa = build_openai_body_v2(&mk(Provider::OpenAi));
        assert!(oa.contains("\"role\":\"system\",\"content\":\"Sofuu coding agent.\""), "{oa}");
        assert!(!oa.contains("cache_control"), "openai wire stays cache-free: {oa}");
    }

    /// Anthropic requires alternating user/assistant turns. Adjacent plain
    /// same-role messages (ephemeral context + task on an empty history,
    /// compaction summary + user turn) must merge into ONE turn; structured
    /// tool messages are never merged.
    #[test]
    fn anthropic_merges_consecutive_same_role_turns() {
        let mk = |msgs: Vec<AiMessage>| AiRequestConfig {
            provider: Provider::Anthropic,
            model: Some("m".into()),
            api_key: None,
            base_url: None,
            profile: None,
            system_prompt: None,
            response_format: None,
            effort: None,
            max_tokens: 0,
            temperature: -1.0,
            top_p: -1.0,
            stream: false,
            timeout_ms: 0,
            messages: msgs,
            tools: vec![],
        };
        let plain = |role: &str, content: &str| AiMessage {
            role: Some(role.into()),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        };

        // ctx + task on an empty history → one merged user turn.
        let b = build_anthropic_body_v2(&mk(vec![
            plain("system", "core"),
            plain("user", "Context from the shared brain:\n- fact"),
            plain("user", "the task"),
        ]));
        assert!(b.contains("Context from the shared brain:\\n- fact\\n\\nthe task"), "{b}");
        assert_eq!(b.matches("\"role\":\"user\"").count(), 1, "one merged user turn: {b}");

        // Compaction summary (system→user) + user turn merge too.
        let b = build_anthropic_body_v2(&mk(vec![
            plain("system", "core"),
            plain("system", "Prior conversation summary: X"),
            plain("user", "next question"),
        ]));
        assert_eq!(b.matches("\"role\":\"user\"").count(), 1, "{b}");
        assert!(b.contains("Prior conversation summary: X\\n\\nnext question"), "{b}");

        // Alternating roles stay separate.
        let b = build_anthropic_body_v2(&mk(vec![
            plain("user", "q1"),
            plain("assistant", "a1"),
            plain("user", "q2"),
        ]));
        assert_eq!(b.matches("\"role\":\"user\"").count(), 2, "{b}");
        assert_eq!(b.matches("\"role\":\"assistant\"").count(), 1, "{b}");

        // Structured tool messages are never merged.
        let b = build_anthropic_body_v2(&mk(vec![
            plain("user", "q"),
            AiMessage {
                role: Some("assistant".into()),
                content: None,
                tool_calls: Some(r#"[{"id":"c1","type":"function","function":{"name":"t","arguments":"{}"}}]"#.into()),
                tool_call_id: None,
            },
            AiMessage {
                role: Some("tool".into()),
                content: Some("result".into()),
                tool_calls: None,
                tool_call_id: Some("c1".into()),
            },
        ]));
        assert!(b.contains("\"type\":\"tool_use\""), "{b}");
        assert!(b.contains("\"type\":\"tool_result\""), "{b}");
    }

    /// P6 integration: two turns that differ ONLY in the last user message
    /// must produce byte-identical request bodies up to that message — the
    /// stable prefix every backend's cache (OpenAI auto, Anthropic markers,
    /// local KV reuse) anchors on. Nothing dynamic may leak ahead of it.
    #[test]
    fn prefix_is_byte_identical_across_turns() {
        let mk = |provider: Provider, task: &str| AiRequestConfig {
            provider,
            model: Some("m".into()),
            api_key: None,
            base_url: None,
            profile: None,
            system_prompt: None,
            response_format: None,
            effort: None,
            max_tokens: 0,
            temperature: -1.0,
            top_p: -1.0,
            stream: false,
            timeout_ms: 0,
            messages: vec![
                AiMessage {
                    role: Some("system".into()),
                    content: Some("Sofuu coding agent. Be direct and concise.".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                AiMessage {
                    role: Some("user".into()),
                    content: Some("turn one question".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                AiMessage {
                    role: Some("assistant".into()),
                    content: Some("turn one answer".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                AiMessage {
                    role: Some("user".into()),
                    content: Some(task.into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            tools: vec![AiToolDef {
                name: Some("read_file".into()),
                description: Some("read a file".into()),
                parameters_json: Some("{}".into()),
            }],
        };

        for (name, provider, build) in [
            ("openai", Provider::OpenAi, build_openai_body_v2 as fn(&AiRequestConfig) -> String),
            ("anthropic", Provider::Anthropic, build_anthropic_body_v2 as fn(&AiRequestConfig) -> String),
        ] {
            let a = build(&mk(provider, "turn two question A"));
            let b = build(&mk(provider, "turn two question B"));
            let common = a
                .bytes()
                .zip(b.bytes())
                .take_while(|(x, y)| x == y)
                .count();
            /* The divergence must be exactly at the last user message —
             * everything before it (system, tools, history) is stable.
             * common >= diverge_at means the first differing byte is at or
             * after the last user message starts; common < len means the
             * bodies do actually differ (at the A/B char). */
            let diverge_at = a.rfind("turn two question").unwrap();
            assert!(
                common >= diverge_at && common < a.len(),
                "{name}: prefix diverged before the last user message (common={common}, last-user starts at {diverge_at})\nA: {a}\nB: {b}"
            );
            assert!(common > 100, "{name}: suspiciously short stable prefix ({common})");
        }
    }

    /// Anthropic thinking budget: explicit per-effort budgets, clamped to
    /// the 16k ceiling and always below max_tokens.
    #[test]
    fn anthropic_thinking_budget_mapping() {
        let base = |effort: &str, max_tokens: i32| AiRequestConfig {
            provider: Provider::Anthropic,
            model: Some("claude".into()),
            api_key: None,
            base_url: None,
            profile: None,
            system_prompt: None,
            response_format: None,
            effort: Some(effort.into()),
            max_tokens,
            temperature: -1.0,
            top_p: -1.0,
            stream: false,
            timeout_ms: 0,
            messages: vec![],
            tools: vec![],
        };
        let b = build_anthropic_body_v2(&base("low", 4096));
        assert!(b.contains("\"budget_tokens\":1024"), "low: {b}");
        let b = build_anthropic_body_v2(&base("medium", 4096));
        assert!(b.contains("\"budget_tokens\":2048"), "medium must clamp below max_tokens: {b}");
        let b = build_anthropic_body_v2(&base("high", 4096));
        assert!(b.contains("\"budget_tokens\":2048"), "high must clamp below max_tokens: {b}");
        let b = build_anthropic_body_v2(&base("max", 4096));
        assert!(b.contains("\"budget_tokens\":2048"), "max must clamp below max_tokens: {b}");
        /* With a large max_tokens, max clamps to the 16k anthropic ceiling. */
        let b = build_anthropic_body_v2(&base("max", 32000));
        assert!(b.contains("\"budget_tokens\":16384"), "max must clamp to 16k ceiling: {b}");
        let b = build_anthropic_body_v2(&base("medium", 32000));
        assert!(b.contains("\"budget_tokens\":4096"), "medium explicit 4k: {b}");
        let b = build_anthropic_body_v2(&base("high", 32000));
        assert!(b.contains("\"budget_tokens\":16384"), "high explicit 16k: {b}");
    }

    /// json_escape: escapes controls/JS line separators and NEVER corrupts
    /// non-ASCII text (the old byte-wise impl emitted Latin-1 mojibake).
    #[test]
    fn json_escape_preserves_utf8_and_escapes_controls() {
        assert_eq!(json_escape(None), "");
        assert_eq!(json_escape(Some("plain")), "plain");
        assert_eq!(json_escape(Some("a\"b\\c\nd\re\tf\x08g\x0ch")), "a\\\"b\\\\c\\nd\\re\\tf\\bg\\fh");
        assert_eq!(json_escape(Some("caf\u{e9} \u{4e2d}\u{6587} \u{1f600}")), "caf\u{e9} \u{4e2d}\u{6587} \u{1f600}");
        assert_eq!(json_escape(Some("\u{2028}sep\u{2029}")), "\\u2028sep\\u2029");
        assert_eq!(json_escape(Some("\u{01}ctrl")), "\\u0001ctrl");
    }

    /// strip_think_tags: strips REAL reasoning tags and never corrupts
    /// plain text that merely contains the words "thinking"/"response".
    #[test]
    fn strip_think_tags_handles_real_tags_only() {
        // Real <thinking> block stripped, answer kept.
        let (out, think) = strip_think_tags(b"<thinking>let me reason</thinking>The answer is 42.", 65536);
        assert_eq!(String::from_utf8(out).unwrap(), "The answer is 42.");
        assert_eq!(String::from_utf8(think).unwrap(), "let me reason");

        // Plain text with "thinking" / "response" as words must be preserved.
        let (out, _) = strip_think_tags(b"The model is thinking about the response now.", 65536);
        assert_eq!(String::from_utf8(out).unwrap(), "The model is thinking about the response now.");

        // Deepseek variant.
        let (out, think) = strip_think_tags(b"<|thinking|>secret<|/thinking|>public", 65536);
        assert_eq!(String::from_utf8(out).unwrap(), "public");
        assert_eq!(String::from_utf8(think).unwrap(), "secret");

        // <think> short form.
        let (out, think) = strip_think_tags(b"<think>hmm</think>done", 65536);
        assert_eq!(String::from_utf8(out).unwrap(), "done");
        assert_eq!(String::from_utf8(think).unwrap(), "hmm");

        // Unclosed tag: rest is thinking.
        let (out, think) = strip_think_tags(b"answer<thinking>never closed", 65536);
        assert_eq!(String::from_utf8(out).unwrap(), "answer");
        assert!(think.len() > 0);
    }
}
