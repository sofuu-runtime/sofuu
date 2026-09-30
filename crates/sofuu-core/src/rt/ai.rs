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
use std::ffi::{CStr, CString, c_char, c_int, c_long, c_void};
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
    /* Multimodal (P2): user-message images as data URLs
     * ("data:image/png;base64,…"). Empty = text-only message. */
    images: Vec<String>,
}

/// Parse the JSON-stringified images array (["data:image/png;base64,…", …])
/// into typed entries; caps count (8) and per-image size (6 MB) so a giant
/// paste cannot balloon the wire. Non-`data:` entries are dropped — the
/// wire builders only inline data URLs.
fn parse_image_urls(json: Option<&str>) -> Vec<String> {
    let Some(json) = json else { return Vec::new() };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else { return Vec::new() };
    let Some(arr) = v.as_array() else { return Vec::new() };
    const MAX_IMAGES: usize = 8;
    const MAX_URL_CHARS: usize = 8 * 1024 * 1024; // ~6 MB binary per image
    arr.iter()
        .filter_map(|x| x.as_str())
        .filter(|u| u.starts_with("data:image/") && u.len() <= MAX_URL_CHARS)
        .take(MAX_IMAGES)
        .map(str::to_string)
        .collect()
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
    max_tokens: i32,                 // explicit override; 0 = model's max output (model_caps)
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
    provider_explicit: bool, /* true when opts.provider was present */
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>, /* full endpoint override; None → built-in */
    space: Option<String>,   /* local embedding space; None → default */
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
pub(crate) fn json_escape(s: Option<&str>) -> String {
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
        let n = json_escape(t.name.as_deref());
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
        let n = json_escape(t.name.as_deref());
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

/// The effective per-response output cap: an explicit config value wins;
/// otherwise the MODEL's published maximum. Capabilities resolve through
/// the endpoint-aware ladder — registry → caps the endpoint itself
/// published for this exact model (discovered store, keyed by cfg.base_url)
/// — so a gateway hosting a known family at a smaller cap is honoured.
/// For the OpenAI wire unknown models omit the field (provider default);
/// Anthropic wire's fallback is endpoint-driven (see build_anthropic).
///
/// Layer 0 of the alloc gate: an explicit value is clamped DOWN to the
/// model's hard output limit when ANY real source knows it. Sending a
/// max_tokens the model cannot honour is a provider 400 and a dead
/// turn — the wire must never carry it, whatever the config says.
/// User-facing notes for the clamp are emitted by the layers above
/// (chat driver / agent plan), which run the same resolve ladder.
fn effective_max_output(cfg: &AiRequestConfig) -> i32 {
    let caps = crate::rt::model_caps::lookup_for(cfg.model.as_deref(), cfg.base_url.as_deref());
    if cfg.max_tokens > 0 {
        if caps.max_output > 0 && cfg.max_tokens > caps.max_output {
            return caps.max_output;
        }
        /* Zero-evidence guard: a flat config cap (the 384k-global class)
         * riding a model the registry, the endpoint's listing AND any
         * learned-400 all know nothing about is a gamble that empties
         * some gateways (200 + zero text). Omit and let the endpoint
         * apply its own default. Once ANY evidence lands (a harvest, a
         * learned limit), config rides again — clamped by it. */
        if caps.max_output == 0
            && caps.ctx_window == 0
            && !crate::ml::alloc::policy::has_any_evidence(
                cfg.model.as_deref(),
                cfg.base_url.as_deref(),
            )
        {
            return 0;
        }
        cfg.max_tokens
    } else {
        caps.max_output
    }
}

/// Fallback for unknown models on the Anthropic wire where max_tokens is
/// REQUIRED. Layer 0 of the alloc gate: conservative and WINDOW-DERIVED —
/// the allocator's unknown-model window (32k) with an eighth reserved for
/// output. The old endpoint-shaped constant (71,680) assumed every
/// unknown model was Claude-class and 400'd on small models hosted
/// behind Anthropic-shaped gateways — exactly the "config does not
/// respect the selected model" failure the allocator exists to kill.
/// Registry-known models never reach this path.
fn anthropic_fallback_max_tokens() -> i32 {
    crate::ml::alloc::policy::UNKNOWN_MAX_OUTPUT as i32
}

fn build_openai_body_v2(cfg: &AiRequestConfig) -> String {
    build_openai_body_inner(cfg, true)
}

/// `emit_max=false` is used by the local-server wrapper, which carries the
/// output cap in `options.num_predict` instead of a top-level field.
fn build_openai_body_inner(cfg: &AiRequestConfig, emit_max: bool) -> String {
    /* model may be NULL when the user passed a non-string — never emit
     * "(null)" or crash; an empty model is rejected by the provider. */
    let mut body = format!("{{\"model\":\"{}\"", cfg.model.as_deref().unwrap_or(""));

    let caps = crate::rt::model_caps::lookup_for(cfg.model.as_deref(), cfg.base_url.as_deref());

    /* Dynamic output cap: explicit config wins, else the MODEL's published
     * max output (registry → endpoint-discovered). Never a flat constant.
     * Unknown-to-everything models omit the field entirely so the
     * endpoint applies its own default. */
    if emit_max {
        let max_out = effective_max_output(cfg);
        if max_out > 0 {
            /* Param name keyed by MODEL family, not endpoint host: the
             * reasoning ladder families (o-series/gpt-5) REJECT
             * `max_tokens` on modern gateways, and every gateway that
             * hosts those names accepts `max_completion_tokens`. All
             * other models — including Claude/deepseek/qwen served over
             * THIS endpoint by third-party providers — keep the widely
             * compatible `max_tokens`. */
            if matches!(caps.thinking, crate::rt::model_caps::Thinking::Effort(_)) {
                body.push_str(&format!(",\"max_completion_tokens\":{}", max_out));
            } else {
                body.push_str(&format!(",\"max_tokens\":{}", max_out));
            }
        }
    }

    if cfg.temperature >= 0.0 {
        body.push_str(&format!(",\"temperature\":{:.2}", cfg.temperature));
    }
    if cfg.top_p >= 0.0 {
        body.push_str(&format!(",\"top_p\":{:.2}", cfg.top_p));
    }

    /* JSON structured output — OpenAI/Ollama: response_format field */
    let json_mode = cfg.response_format.as_deref() == Some("json");
    if json_mode {
        body.push_str(",\"response_format\":{\"type\":\"json_object\"}");
    }

    /* Reasoning effort — OpenAI o-series / GPT-5 style ladders.
     * Capability-aware: unsupported levels fall back to the HIGHEST level
     * the model supports ("max" on a low/medium/high ladder → "high");
     * models without any reasoning support omit the field entirely
     * instead of triggering a provider 400. */
    if let Some(effort) = crate::rt::model_caps::resolve_openai_effort(
        &caps,
        cfg.effort.as_deref(),
    ) {
        body.push_str(&format!(",\"reasoning_effort\":\"{}\"", effort));
    }

    /* OpenAI chat-completions omits the final usage chunk unless this is
     * sent (spec behavior since 2024-06; OpenRouter/vLLM also honor it,
     * and strict gateways tolerate the extra field). Without it every
     * OpenAI-family stream reports usage=0: budget_tokens can never fire,
     * budget_usd never trips, and the ctx meter silently falls back to
     * its estimator. Anthropic is unaffected — usage rides
     * message_start/message_delta unconditionally. `emit_max=false` marks
     * the local-server wrapper (Ollama native /api/chat), which ignores
     * this option and returns usage unprompted. */
    if cfg.stream && emit_max {
        body.push_str(",\"stream_options\":{\"include_usage\":true}");
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
        /* Multimodal (P2): user images inline as content parts —
         * [{type:text},{type:image_url}] per the OpenAI chat shape. */
        if !m.images.is_empty() {
            body.push_str(&format!("{{\"role\":\"{}\",\"content\":[", m.role.as_deref().unwrap_or("")));
            body.push_str(&format!("{{\"type\":\"text\",\"text\":\"{}\"}}", es));
            for img in &m.images {
                let eimg = json_escape(Some(img));
                body.push_str(&format!(
                    ",{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"{}\"}}}}",
                    eimg
                ));
            }
            body.push(']');
        } else {
            body.push_str(&format!(
                "{{\"role\":\"{}\",\"content\":\"{}\"",
                m.role.as_deref().unwrap_or(""),
                es
            ));
        }
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
    /* Dynamic output cap (registry → endpoint-discovered): explicit config →
     * model's published max output → conservative floor for models nobody
     * knows. The old flat `4096` fallback truncated every answer
     * regardless of model. */
    let caps = crate::rt::model_caps::lookup_for(cfg.model.as_deref(), cfg.base_url.as_deref());
    let max_tokens = effective_max_output(cfg);
    let max_tokens = if max_tokens > 0 {
        max_tokens
    } else {
        anthropic_fallback_max_tokens()
    };
    let mut body = format!(
        "{{\"model\":\"{}\",\"max_tokens\":{},\"stream\":{}",
        cfg.model.as_deref().unwrap_or(""),
        max_tokens,
        if cfg.stream { "true" } else { "false" }
    );

    /* Reasoning effort → Anthropic extended thinking.
     * max_tokens is OUTPUT-ONLY (caps.maxOutput, strictly output limit).
     * Thinking lives inside the total context window (caps.ctxWindow) per
     * effort fraction, not inside max_output — so thinking does not define
     * output limit. API requires budget < max_tokens, so budget is capped
     * to max_tokens-10% only to satisfy the wire, otherwise thinking is
     * context-derived and output stays independent. */
    let thinking_budget: Option<i32> = match cfg.effort.as_deref() {
        Some(effort) if !effort.is_empty() && effort != "off" => {
            let mut b = crate::rt::model_caps::resolve_thinking_budget(&caps, effort);
            // Wire requires budget < max_tokens; cap only to satisfy it, keep thinking context-derived
            if let Some(bv) = b {
                let room = ((max_tokens as f64 * 0.10).ceil() as i32).max(1);
                if bv >= max_tokens {
                    // P3 (AUDIT-2026-09-07): with max_tokens=1 the old clamp
                    // produced budget==max_tokens, which the wire REJECTS
                    // (strict budget < max_tokens). Drop the budget instead
                    // of emitting an invalid request.
                    b = if max_tokens > 1 {
                        Some((max_tokens - room).max(1))
                    } else {
                        None
                    };
                }
            }
            b
        }
        _ => None,
    };

    if cfg.temperature >= 0.0 && thinking_budget.is_none() {
        /* Temperature is skipped when extended thinking rides along —
         * the API rejects temperature ≠ 1 on thinking requests. */
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

    if let Some(budget) = thinking_budget {
        body.push_str(&format!(
            ",\"thinking\":{{\"type\":\"enabled\",\"budget_tokens\":{}}}",
            budget
        ));
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
                /* P3 (AUDIT-2026-09-07): parse with serde_json, not the old
                 * find("\"id\"")/rfind('{') string-scan — a tool whose
                 * arguments JSON contained its own "id" key made the scan
                 * grab the INNER object and emit a corrupted tool_use block
                 * on the Anthropic wire. The scan also only decoded
                 * string-valued arguments; object-valued ones are now
                 * re-serialized verbatim. */
                let calls: Vec<serde_json::Value> =
                    serde_json::from_str(tc).unwrap_or_default();
                for call in calls {
                    let id = call
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let fname = call
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let fargs = match call.get("function").and_then(|f| f.get("arguments")) {
                        Some(serde_json::Value::String(s)) => s.clone(),
                        Some(o @ serde_json::Value::Object(_)) => o.to_string(),
                        _ => String::new(),
                    };
                    if !tc_first {
                        body.push(',');
                    }
                    body.push_str(&format!(
                        "{{\"type\":\"tool_use\",\"id\":\"{}\",\"name\":\"{}\",\"input\":{}}}",
                        json_escape(Some(id)),
                        json_escape(Some(fname)),
                        if fargs.is_empty() { "{}" } else { fargs.as_str() }
                    ));
                    tc_first = false;
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
        /* Multimodal (P2): images → Anthropic base64 image blocks
         * [{type:image, source:{type:base64, media_type, data}}, …]
         * (data URL "data:<mime>;base64,<b64>" split apart). */
        if !m.images.is_empty() {
            body.push_str(&format!("{{\"role\":\"{}\",\"content\":[", role));
            body.push_str(&format!("{{\"type\":\"text\",\"text\":\"{}\"}}", es));
            for img in &m.images {
                // data:image/png;base64,AAAA → media_type + b64 payload
                let (mime, b64) = match img.strip_prefix("data:") {
                    Some(rest) => match rest.split_once(";base64,") {
                        Some((m_, b_)) => (m_, b_),
                        None => continue,
                    },
                    None => continue,
                };
                let emime = json_escape(Some(mime));
                let eb64 = json_escape(Some(b64));
                body.push_str(&format!(
                    ",{{\"type\":\"image\",\"source\":{{\"type\":\"base64\",\"media_type\":\"{}\",\"data\":\"{}\"}}}}",
                    emime, eb64
                ));
            }
            body.push_str("]}");
            first = false;
            continue;
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

    /* Build with the OpenAI builder first (clears response_format if json);
     * suppress its top-level max field — the local wire carries the cap
     * in options.num_predict below. */
    let mut tmp = cfg.clone();
    if json_mode {
        tmp.response_format = None; /* we'll add format:json ourselves */
    }
    let base = build_openai_body_inner(&tmp, false);

    /* Build options + optional json format patch — num_predict uses the
     * same dynamic cap (explicit config → model's published max output). */
    let num_predict = effective_max_output(cfg);
    let patch: String = if num_predict > 0 {
        if json_mode {
            format!(",\"options\":{{\"num_predict\":{}}},\"format\":\"json\"}}", num_predict)
        } else {
            format!(",\"options\":{{\"num_predict\":{}}}}}", num_predict)
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
pub(crate) fn sofuu_tfidf_embed(text: &[u8], out_vec: &mut [f32], dim: usize) {
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

/// Capture WHY the provider ended the stream, per wire:
///   OpenAI:  choices[0].finish_reason ("stop" | "length" | "tool_calls" | …)
///   Anthropic: delta.stop_reason (message_delta event)
///   Local: (none)
/// Written into `req.finish_reason` — surfaced in the done() usage stats so
/// an EMPTY stream (no text, no tool calls) is diagnosable instead of
/// silently becoming "(no response)".
unsafe fn capture_finish_reason(ctx: *mut JSContext, p: Provider, req: *mut AiStreamReq, data: &[u8]) {
    if data == b"[DONE]" || (*req).finish_len > 0 {
        return; /* already captured, or end-of-stream marker */
    }
    let data_c = CString::new(data).unwrap_or_default();
    let ev = qjs::JS_ParseJSON(ctx, data_c.as_ptr(), data_c.as_bytes().len(), c"<sse_fr>".as_ptr());
    if qjs::is_exception(ev) {
        qjs::sofuu_js_get_exception(ctx);
        return;
    }
    let fr = if p == Provider::Anthropic {
        let delta = qjs::sofuu_js_get_property_str(ctx, ev, c"delta".as_ptr());
        let v = if qjs::is_object(delta) {
            qjs::sofuu_js_get_property_str(ctx, delta, c"stop_reason".as_ptr())
        } else {
            qjs::sofuu_js_undefined()
        };
        if !qjs::is_undefined(delta) && !qjs::is_null(delta) {
            qjs::sofuu_js_free_value(ctx, delta);
        }
        v
    } else {
        let choices = qjs::sofuu_js_get_property_str(ctx, ev, c"choices".as_ptr());
        let mut v = qjs::sofuu_js_undefined();
        if qjs::JS_IsArray(ctx, choices) != 0 {
            let ch0 = qjs::JS_GetPropertyUint32(ctx, choices, 0);
            if qjs::is_object(ch0) {
                v = qjs::sofuu_js_get_property_str(ctx, ch0, c"finish_reason".as_ptr());
            }
            qjs::sofuu_js_free_value(ctx, ch0);
        }
        qjs::sofuu_js_free_value(ctx, choices);
        v
    };
    if qjs::sofuu_js_is_string(fr) != 0 {
        let s = qjs::sofuu_js_to_cstring(ctx, fr);
        if !s.is_null() {
            let bytes = CStr::from_ptr(s).to_bytes();
            if !bytes.is_empty() {
                let n = bytes.len().min(24);
                (&mut (*req).finish_reason)[..n].copy_from_slice(&bytes[..n]);
                (*req).finish_len = n;
            }
        }
        if !s.is_null() {
            qjs::sofuu_js_free_cstring(ctx, s);
        }
    }
    qjs::sofuu_js_free_value(ctx, fr);
    qjs::sofuu_js_free_value(ctx, ev);
}

/// Append `finishReason` to the done() usage stats (empty string when the
/// provider never reported one — itself the diagnostic signature).
unsafe fn set_stream_finish(ctx: *mut JSContext, stats: JSValue, req: *mut AiStreamReq) {
    let fr = (&(*req).finish_reason)[..(*req).finish_len].to_vec();
    let c = CString::new(fr).unwrap_or_default();
    qjs::sofuu_js_set_property_str(
        ctx,
        stats,
        c"finishReason".as_ptr(),
        qjs::sofuu_js_new_string(ctx, c.as_ptr()),
    );
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
const REQ_TAG_AUDIO: c_int = 4;

/* Hard cap on any single provider response (complete/stream/embed) so a
 * misbehaving endpoint cannot grow the heap without bound. */
const AI_MAX_RESPONSE: usize = 64 * 1024 * 1024;

/* How long a model may stay completely silent — no byte at all, whether
 * waiting for the first one (TTFB) or between chunks — before we give up
 * with a clear error. Shared-pool thinking models can legitimately sit for
 * minutes before the first token, so the window is generous (5 min, any
 * provider/model). This replaces the old flat 120s TOTAL timeout, which
 * killed long thinking generations mid-stream while still hanging long
 * enough on dead ones. An ACTIVE stream is never capped. Env override
 * (seconds) exists so tests can shrink it. */
const AI_STALL_TIMEOUT_SECS: c_long = 300;
fn ai_stall_timeout_secs() -> c_long {
    std::env::var("SOFUU_STALL_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<c_long>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(AI_STALL_TIMEOUT_SECS)
}

/// Friendly message for a libcurl-side timeout (code 28). With the stall
/// watchdog in place this path is reached for connect-phase timeouts (the
/// 30s CONNECTTIMEOUT — cheap to auto-retry) or a caller-set total cap.
fn ai_timeout_message(http_code: c_long, _stall_secs: c_long) -> String {
    if http_code > 0 {
        /* Headers arrived: a caller-capped total timeout fired mid-transfer.
         * Worded to stay OUT of the agent's transient-retry regex. */
        String::from("request exceeded its time budget — the model is unresponsive (retry, or switch models)")
    } else {
        /* Never reached: connect-phase timeout. Cheap to retry. */
        String::from("connection timed out — provider unreachable (check the endpoint URL; retrying may help)")
    }
}

/// HTTP status seen so far on an easy handle (0 = headers never arrived).
unsafe fn easy_http_code(easy: *mut Curl) -> c_long {
    let mut http_code: c_long = 0;
    curl::curl_easy_getinfo(easy, curl::CURLINFO_RESPONSE_CODE, &mut http_code as *mut c_long);
    http_code
}

/// Apply the provider-patience options shared by complete/stream/embed:
/// a bounded connect phase, and the caller's explicit total cap ONLY when
/// asked (opts.timeout_ms). Silence after that is policed by the stall
/// watchdog (libcurl's LOW_SPEED check never self-wakes in the socket-API
/// multi setup — verified empirically), not by curl.
unsafe fn ai_set_patience(easy: *mut Curl, timeout_ms: c_long) {
    curl::curl_easy_setopt(easy, curl::CURLOPT_CONNECTTIMEOUT, 30 as c_long);
    curl::curl_easy_setopt(easy, curl::CURLOPT_TIMEOUT_MS, if timeout_ms > 0 { timeout_ms } else { 0 });
}

/* ------------------------------------------------------------------ */
/* Stall watchdog                                                      */
/* ------------------------------------------------------------------ */

/// Common prefix of AiCompleteReq / AiStreamReq / AiEmbedReq — all three
/// keep these fields at identical offsets, so any in-flight request can
/// be walked as this header (intrusive list + last-byte timestamp).
#[repr(C)]
struct AiReqHdr {
    tag: c_int,
    wd_next: *mut c_void,
    last_rx: std::time::Instant,
}

/// Message when the watchdog aborts a request the model never answered.
/// Deliberately NOT matched by the agent's transient-retry regex: another
/// 5-minute silent wait is not what "wait up to 5 minutes" means.
fn ai_stall_message(stall_secs: c_long) -> String {
    format!(
        "provider sent no data for {}s — the model is unresponsive (retry, or switch models)",
        stall_secs
    )
}

unsafe fn active_link(req: *mut c_void) {
    let hdr = req as *mut AiReqHdr;
    let head = G_ACTIVE.with(|a| a.get());
    (*hdr).wd_next = head;
    G_ACTIVE.with(|a| a.set(req));
    ai_watchdog_arm();
    /* REF the watchdog for every in-flight request (unref'd in
     * active_unlink): libuv computes the poll timeout from REF'D timers
     * only, so an unref'd watchdog would never tick while the loop sits
     * in epoll_wait on the silent provider socket — the exact hang this
     * guard exists to break. */
    let wd = G_WATCHDOG.with(|w| w.get());
    uv::uv_ref(wd as *mut UvHandle);
}

unsafe fn active_unlink(req: *mut c_void) {
    let after = (*(req as *mut AiReqHdr)).wd_next;
    G_ACTIVE.with(|a| {
        let head = a.get();
        if head == req {
            a.set(after);
            return;
        }
        let mut p = head;
        while !p.is_null() {
            let hdr = p as *mut AiReqHdr;
            if (*hdr).wd_next == req {
                (*hdr).wd_next = after;
                return;
            }
            p = (*hdr).wd_next;
        }
    });
    /* Balanced with the uv_ref in active_link: when the last request
     * leaves, the watchdog stops pinning the event loop. */
    let wd = G_WATCHDOG.with(|w| w.get());
    if !wd.is_null() {
        uv::uv_unref(wd as *mut UvHandle);
    }
}

/// Create the watchdog once (neutral ref count — active_link/active_unlink
/// hold one ref per in-flight request) and keep it repeating every second.
unsafe fn ai_watchdog_arm() {
    let wd = G_WATCHDOG.with(|w| w.get());
    if !wd.is_null() {
        return;
    }
    let timer = libc::malloc(uv::sofuu_uv_timer_size()) as *mut UvTimer;
    G_WATCHDOG.with(|w| w.set(timer));
    uv::uv_timer_init(sofuu_loop_get(), timer);
    uv::uv_unref(timer as *mut UvHandle); /* init starts ref'd; neutralize */
    uv::uv_timer_start(timer, Some(ai_watchdog_cb), 1000, 1000);
}

/// Abort one stalled request. Runs from the uv timer callback — outside
/// the curl stack, so remove_handle + JS calls are safe (same pattern as
/// stream_abort_one / ai_flush_check_cb).
unsafe fn stall_abort(req: *mut c_void, tag: c_int) {
    let stall = ai_stall_timeout_secs();
    match tag {
        REQ_TAG_STREAM => {
            let r = req as *mut AiStreamReq;
            let ctx = (*r).ctx;
            if (*r).done_called == 0 {
                (*r).done_called = 1;
                if qjs::JS_IsFunction(ctx, (*r).error_fn) != 0 {
                    let m = CString::new(ai_stall_message(stall)).unwrap_or_default();
                    let e = qjs::sofuu_js_new_string(ctx, m.as_ptr());
                    let ret = qjs::JS_Call(ctx, (*r).error_fn, qjs::sofuu_js_undefined(), 1, &e);
                    if qjs::is_exception(ret) {
                        qjs::js_std_dump_error(ctx);
                    }
                    qjs::sofuu_js_free_value(ctx, ret);
                    qjs::sofuu_js_free_value(ctx, e);
                }
            }
            /* done_called == 1: [DONE] already delivered the answer but the
             * provider never closed the connection — end it quietly. */
            if !(*r).easy.is_null() {
                curl::curl_multi_remove_handle(g_multi(), (*r).easy);
                curl::curl_easy_cleanup((*r).easy);
                (*r).easy = ptr::null_mut();
            }
            stream_req_destroy(ctx, r);
        }
        REQ_TAG_COMPLETE => {
            let r = req as *mut AiCompleteReq;
            let ctx = (*r).ctx;
            let m = CString::new(ai_stall_message(stall)).unwrap_or_default();
            let err = qjs::sofuu_js_new_string(ctx, m.as_ptr());
            sofuu_promise_reject((*r).promise, err);
            if !(*r).easy.is_null() {
                curl::curl_multi_remove_handle(g_multi(), (*r).easy);
                curl::curl_easy_cleanup((*r).easy);
                (*r).easy = ptr::null_mut();
            }
            active_unlink(req);
            if !(*r).headers.is_null() {
                curl::curl_slist_free_all((*r).headers);
            }
            drop(Box::from_raw(r));
        }
        REQ_TAG_EMBED => {
            let r = req as *mut AiEmbedReq;
            let ctx = (*r).ctx;
            let m = CString::new(ai_stall_message(stall)).unwrap_or_default();
            let err = qjs::sofuu_js_new_string(ctx, m.as_ptr());
            sofuu_promise_reject((*r).promise, err);
            if !(*r).easy.is_null() {
                curl::curl_multi_remove_handle(g_multi(), (*r).easy);
                curl::curl_easy_cleanup((*r).easy);
                (*r).easy = ptr::null_mut();
            }
            active_unlink(req);
            if !(*r).headers.is_null() {
                curl::curl_slist_free_all((*r).headers);
            }
            drop(Box::from_raw(r));
        }
        REQ_TAG_AUDIO => {
            let r = req as *mut AiAudioReq;
            let ctx = (*r).ctx;
            let m = CString::new(ai_stall_message(stall)).unwrap_or_default();
            let err = qjs::sofuu_js_new_string(ctx, m.as_ptr());
            sofuu_promise_reject((*r).promise, err);
            if !(*r).easy.is_null() {
                curl::curl_multi_remove_handle(g_multi(), (*r).easy);
                curl::curl_easy_cleanup((*r).easy);
                (*r).easy = ptr::null_mut();
            }
            active_unlink(req);
            audio_req_free_mime(r);
            if !(*r).headers.is_null() {
                curl::curl_slist_free_all((*r).headers);
            }
            drop(Box::from_raw(r));
        }
        _ => {}
    }
}

unsafe extern "C" fn ai_watchdog_cb(_t: *mut UvTimer) {
    let stall = ai_stall_timeout_secs();
    let now = std::time::Instant::now();
    let mut p = G_ACTIVE.with(|a| a.get());
    while !p.is_null() {
        let hdr = p as *mut AiReqHdr;
        /* Capture BEFORE stall_abort: it destroys the node (and the JS it
         * runs may even enqueue fresh requests at the list head). */
        let next = (*hdr).wd_next;
        let tag = (*hdr).tag;
        if now.duration_since((*hdr).last_rx).as_secs() as c_long >= stall {
            stall_abort(p, tag);
        }
        p = next;
    }
}

/// F-2 teardown abort for ONE request: mirror stall_abort's cleanup WITHOUT
/// running any user JS (engine_destroy must not let a stream error_fn or a
/// promise .then enqueue fresh requests/handles after the shutdown pass).
unsafe fn ai_teardown_abort(req: *mut c_void, tag: c_int) {
    match tag {
        REQ_TAG_STREAM => {
            let r = req as *mut AiStreamReq;
            if !(*r).easy.is_null() {
                curl::curl_multi_remove_handle(g_multi(), (*r).easy);
                curl::curl_easy_cleanup((*r).easy);
                (*r).easy = ptr::null_mut();
            }
            /* Frees the JS refs (push/done/error/think/tool_calls fns) and
             * the box — no user JS, safe with ctx still live. */
            stream_req_destroy((*r).ctx, r);
        }
        REQ_TAG_COMPLETE => {
            let r = req as *mut AiCompleteReq;
            let m = c"engine shutting down".as_ptr();
            let err = qjs::sofuu_js_new_string((*r).ctx, m);
            sofuu_promise_reject((*r).promise, err);
            if !(*r).easy.is_null() {
                curl::curl_multi_remove_handle(g_multi(), (*r).easy);
                curl::curl_easy_cleanup((*r).easy);
                (*r).easy = ptr::null_mut();
            }
            active_unlink(req);
            if !(*r).headers.is_null() {
                curl::curl_slist_free_all((*r).headers);
            }
            drop(Box::from_raw(r));
        }
        REQ_TAG_EMBED => {
            let r = req as *mut AiEmbedReq;
            let m = c"engine shutting down".as_ptr();
            let err = qjs::sofuu_js_new_string((*r).ctx, m);
            sofuu_promise_reject((*r).promise, err);
            if !(*r).easy.is_null() {
                curl::curl_multi_remove_handle(g_multi(), (*r).easy);
                curl::curl_easy_cleanup((*r).easy);
                (*r).easy = ptr::null_mut();
            }
            active_unlink(req);
            if !(*r).headers.is_null() {
                curl::curl_slist_free_all((*r).headers);
            }
            drop(Box::from_raw(r));
        }
        REQ_TAG_AUDIO => {
            let r = req as *mut AiAudioReq;
            let m = c"engine shutting down".as_ptr();
            let err = qjs::sofuu_js_new_string((*r).ctx, m);
            sofuu_promise_reject((*r).promise, err);
            if !(*r).easy.is_null() {
                curl::curl_multi_remove_handle(g_multi(), (*r).easy);
                curl::curl_easy_cleanup((*r).easy);
                (*r).easy = ptr::null_mut();
            }
            active_unlink(req);
            audio_req_free_mime(r);
            if !(*r).headers.is_null() {
                curl::curl_slist_free_all((*r).headers);
            }
            drop(Box::from_raw(r));
        }
        _ => {}
    }
}

/// F-2 (AUDIT-2026-09-01-CLI): abort every in-flight AI request owned by
/// `ctx`. engine_destroy calls this before the armed-handle shutdown pass —
/// with several engines sharing the process-global loop, a dying engine's
/// still-transferring request would otherwise fire its curl callbacks into
/// freed Rust boxes/JS during a surviving engine's uv_run.
///
/// Promise rejects only enqueue jobs (they die with the context) and free
/// the promise handle; streams get stream_req_destroy. The trailing
/// CURL_SOCKET_TIMEOUT kick delivers curl's POLL_REMOVE, closing each
/// request's per-socket poll handle (ai_poll_close_cb untracks it).
///
/// # Safety
/// Loop thread, teardown path only — `ctx` must still be live and no JS may
/// run after this.
pub unsafe fn ai_abort_requests_for_ctx(ctx: *mut JSContext) {
    let mut p = G_ACTIVE.with(|a| a.get());
    while !p.is_null() {
        let hdr = p as *mut AiReqHdr;
        /* Capture BEFORE the abort: it destroys the node. */
        let next = (*hdr).wd_next;
        let tag = (*hdr).tag;
        let mine = match tag {
            REQ_TAG_STREAM => (*(p as *mut AiStreamReq)).ctx == ctx,
            REQ_TAG_COMPLETE => (*(p as *mut AiCompleteReq)).ctx == ctx,
            REQ_TAG_EMBED => (*(p as *mut AiEmbedReq)).ctx == ctx,
            REQ_TAG_AUDIO => (*(p as *mut AiAudioReq)).ctx == ctx,
            _ => false,
        };
        if mine {
            ai_teardown_abort(p, tag);
        }
        p = next;
    }
    let mut running: c_int = 0;
    curl::curl_multi_socket_action(g_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running);
}

thread_local! {
    static G_MULTI: Cell<*mut CurlM> = const { Cell::new(ptr::null_mut()) };
    static G_TIMER: Cell<*mut UvTimer> = const { Cell::new(ptr::null_mut()) };
    static G_INIT_DONE: Cell<c_int> = const { Cell::new(0) };
    /* Stall watchdog: repeating unref'd timer + the registry of in-flight
     * AI requests (chained through the AiReqHdr prefix). libcurl's own
     * LOW_SPEED check never self-wakes under the socket-API multi setup,
     * so silence is enforced here instead. */
    static G_WATCHDOG: Cell<*mut UvTimer> = const { Cell::new(ptr::null_mut()) };
    static G_ACTIVE: Cell<*mut c_void> = const { Cell::new(ptr::null_mut()) };
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
 * CURLINFO_PRIVATE pointer can be tag-dispatched like C's req_base_t.
 * wd_next + last_rx (the AiReqHdr prefix) are shared by the stall
 * watchdog; complete_write_cb is shared with AiEmbedReq, so both structs
 * MUST keep the prefix field order identical. */
#[repr(C)]
/// repr(C): complete_write_cb + the watchdog read the header prefix
/// through AiCompleteReq/AiReqHdr casts — field order is load-bearing.
#[repr(C)]
struct AiCompleteReq {
    tag: c_int,
    wd_next: *mut c_void,
    last_rx: std::time::Instant,
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
    wd_next: *mut c_void,
    last_rx: std::time::Instant,
    ctx: *mut JSContext,
    promise: *mut PromiseHandle,
    provider: Provider,
    response_body: Vec<u8>,
    headers: *mut CurlSlist,
    post_body: Option<CString>,
    easy: *mut Curl,
    num_inputs: usize,
}

/// M2: provider audio request (transcribe/speak). Same header prefix as
/// AiEmbedReq so complete_write_cb + the watchdog work unchanged.
/// `mime` owns the multipart body (transcribe); `post_body` the JSON body
/// (speak). `audio` keeps upload bytes alive for the transfer.
#[repr(C)]
struct AiAudioReq {
    tag: c_int, /* must be first — REQ_TAG_AUDIO */
    wd_next: *mut c_void,
    last_rx: std::time::Instant,
    ctx: *mut JSContext,
    promise: *mut PromiseHandle,
    provider: Provider,
    response_body: Vec<u8>,
    headers: *mut CurlSlist,
    post_body: Option<CString>,
    mime: *mut curl::CurlMime,
    audio: Vec<u8>,
    kind: u8, /* 0 = transcribe (JSON {text}), 1 = speak (raw bytes) */
    format: String, /* speak response label ("mp3"); empty for transcribe */
    easy: *mut Curl,
}

/// Free the mime body (curl_easy_cleanup does NOT free it).
unsafe fn audio_req_free_mime(r: *mut AiAudioReq) {
    if !(*r).mime.is_null() {
        curl::curl_mime_free((*r).mime);
        (*r).mime = ptr::null_mut();
    }
}

/// Parse a transcription body (`{"text": "..."}`) into `{text}`.
/// Undefined on any shape failure (caller rejects).
unsafe fn extract_transcript(ctx: *mut JSContext, raw: &[u8]) -> JSValue {
    let c = match CString::new(raw) {
        Ok(c) => c,
        Err(_) => return qjs::sofuu_js_undefined(),
    };
    let v = qjs::JS_ParseJSON(ctx, c.as_ptr(), c.as_bytes().len(), c"<transcript>".as_ptr());
    if qjs::is_exception(v) {
        qjs::sofuu_js_get_exception(ctx);
        return qjs::sofuu_js_undefined();
    }
    let t = qjs::sofuu_js_get_property_str(ctx, v, c"text".as_ptr());
    let s = cstr_opt(ctx, t);
    qjs::sofuu_js_free_value(ctx, t);
    qjs::sofuu_js_free_value(ctx, v);
    let text = match s {
        Some(t) if !t.is_empty() => t,
        _ => return qjs::sofuu_js_undefined(),
    };
    let obj = qjs::JS_NewObject(ctx);
    if qjs::is_exception(obj) {
        return obj;
    }
    let cs = CString::new(text).unwrap_or_default();
    let sv = qjs::sofuu_js_new_string(ctx, cs.as_ptr());
    qjs::sofuu_js_set_property_str(ctx, obj, c"text".as_ptr(), sv);
    obj
}

/// Package raw audio bytes as `{audio: Uint8Array, format}`.
/// Undefined when the buffer cannot be constructed.
unsafe fn audio_bytes_result(ctx: *mut JSContext, raw: &[u8], format: &str) -> JSValue {
    if raw.is_empty() || raw.len() > AI_MAX_RESPONSE {
        return qjs::sofuu_js_undefined();
    }
    let ab = qjs::JS_NewArrayBufferCopy(ctx, raw.as_ptr(), raw.len());
    if qjs::is_exception(ab) {
        return ab;
    }
    let global = qjs::sofuu_js_get_global_object(ctx);
    let ctor = qjs::sofuu_js_get_property_str(ctx, global, c"Uint8Array".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    let u8 = qjs::JS_CallConstructor(ctx, ctor, 1, &ab);
    qjs::sofuu_js_free_value(ctx, ab);
    qjs::sofuu_js_free_value(ctx, ctor);
    if qjs::is_exception(u8) {
        return u8;
    }
    let obj = qjs::JS_NewObject(ctx);
    if qjs::is_exception(obj) {
        qjs::sofuu_js_free_value(ctx, u8);
        return obj;
    }
    qjs::sofuu_js_set_property_str(ctx, obj, c"audio".as_ptr(), u8);
    let fc = CString::new(format).unwrap_or_default();
    let fv = qjs::sofuu_js_new_string(ctx, fc.as_ptr());
    qjs::sofuu_js_set_property_str(ctx, obj, c"format".as_ptr(), fv);
    obj
}

/// Strict base64 decode (standard alphabet + `=` pad only; rejects
/// whitespace/URL-safe variants). The headless bridge encoding — hosts
/// base64 audio themselves (1 line in Swift/Kotlin), so the C ABI never
/// touches raw bytes and the funnel stays JSON.
fn b64_val(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        b'=' => Some(0),
        _ => None,
    }
}

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let b = s.as_bytes();
    if b.is_empty() || b.len() % 4 != 0 {
        return None;
    }
    // Padding (max 2) only in the final quantum.
    for (i, &c) in b.iter().enumerate() {
        let last_q = i / 4 == b.len() / 4 - 1;
        if c == b'=' && !last_q {
            return None;
        }
        b64_val(c)?;
    }
    // In the final quantum, pad may only trail (positions 2..3, and
    // position 2 requires position 3).
    let last = &b[b.len() - 4..];
    if last[0] == b'=' || last[1] == b'=' || (last[2] == b'=' && last[3] != b'=') {
        return None;
    }
    let mut out = Vec::with_capacity(b.len() / 4 * 3);
    for q in b.chunks_exact(4) {
        let n = ((b64_val(q[0])? as u32) << 18)
            | ((b64_val(q[1])? as u32) << 12)
            | ((b64_val(q[2])? as u32) << 6)
            | (b64_val(q[3])? as u32);
        out.push((n >> 16) as u8);
        if q[2] != b'=' {
            out.push((n >> 8) as u8);
        }
        if q[3] != b'=' {
            out.push(n as u8);
        }
    }
    Some(out)
}

/// Read a Uint8Array/ArrayBuffer argument into owned bytes.
unsafe fn read_audio_bytes(ctx: *mut JSContext, val: JSValueConst) -> Result<Vec<u8>, JSValue> {
    let err = || qjs::JS_ThrowTypeError(ctx, c"expected audio bytes (Uint8Array)".as_ptr());
    let mut byte_offset: usize = 0;
    let mut byte_length: usize = 0;
    let buf = qjs::JS_GetTypedArrayBuffer(ctx, val, &mut byte_offset, &mut byte_length, ptr::null_mut());
    if qjs::is_exception(buf) {
        qjs::sofuu_js_free_value(ctx, buf);
        return Err(err());
    }
    let mut buf_size: usize = 0;
    let p = qjs::JS_GetArrayBuffer(ctx, &mut buf_size, buf);
    if p.is_null() || byte_length == 0 || byte_offset + byte_length > buf_size {
        qjs::sofuu_js_free_value(ctx, buf);
        return Err(err());
    }
    let out = std::slice::from_raw_parts(p.add(byte_offset), byte_length).to_vec();
    qjs::sofuu_js_free_value(ctx, buf);
    Ok(out)
}

struct AiPollCtx {
    sockfd: c_int,
    poll: *mut UvPoll,
}

unsafe extern "C" fn ai_poll_close_cb(h: *mut UvHandle) {
    crate::rt::event_loop::untrack_handle(h);
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
    /* Any byte resets the stall watchdog (shared cb: AiEmbedReq keeps the
     * AiReqHdr prefix at identical offsets). */
    (*(req as *mut AiReqHdr)).last_rx = std::time::Instant::now();
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
    wd_next: *mut c_void,
    last_rx: std::time::Instant,
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
    /// Why the provider ended the stream (OpenAI `choices[0].finish_reason`,
    /// Anthropic `delta.stop_reason`): "stop" | "length" | "tool_calls" |
    /// "content_filter" | … Empty when the provider never said (a stream
    /// that ends with NO finish_reason and NO output is the "(no response)"
    /// signature — surfaced in usage so the agent loop can react).
    finish_reason: [u8; 24],
    finish_len: usize,
    /// First bytes of the response body (used to surface provider JSON
    /// `error.message` when status >=400 — avoids bare "HTTP 429").
    error_body: Vec<u8>,
    /// In-stream SSE error frame (`data: {"error": ...}` inside an HTTP 200
    /// stream — OpenRouter's shape when the upstream fails AFTER headers
    /// were sent). Stashed by the write callback, which aborts the transfer;
    /// the DONE path surfaces this instead of the generic curl write-error.
    stream_err: Option<String>,
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
    active_unlink(req as *mut c_void);
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
            set_stream_finish(ctx, stats, req);
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
    /* Any byte (even an SSE comment/heartbeat) resets the stall watchdog. */
    (*(req as *mut AiReqHdr)).last_rx = std::time::Instant::now();

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
                    set_stream_finish(ctx, stats, req);
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

            /* In-stream error frame: gateways (OpenRouter et al.) can return
             * HTTP 200 and then report the upstream failure as a
             * `data: {"error": {...}}` SSE frame. Without this check the
             * stream completes with zero chunks and the caller silently sees
             * "(no response)". Stash the message and abort the transfer —
             * the DONE path surfaces it via error_fn, same as a >=400 body.
             * Skip once done was signalled: content already delivered. */
            if (*req).done_called == 0 {
                let data_c = CString::new(data).unwrap_or_default();
                let ev_err = qjs::JS_ParseJSON(ctx, data_c.as_ptr(), data_c.as_bytes().len(), c"<sse_err>".as_ptr());
                if qjs::is_exception(ev_err) {
                    /* Non-JSON data line — clear the pending exception and
                     * let the delta/usage extractors ignore it as before. */
                    qjs::sofuu_js_get_exception(ctx);
                } else {
                    let err = qjs::sofuu_js_get_property_str(ctx, ev_err, c"error".as_ptr());
                    let mut emsg: Option<String> = None;
                    let mut ecode: i32 = 0;
                    let had_err = qjs::is_object(err);
                    if had_err {
                        let m = qjs::sofuu_js_get_property_str(ctx, err, c"message".as_ptr());
                        if let Some(s) = cstr_opt(ctx, m) {
                            if !s.is_empty() { emsg = Some(s); }
                        }
                        qjs::sofuu_js_free_value(ctx, m);
                        let cp = qjs::sofuu_js_get_property_str(ctx, err, c"code".as_ptr());
                        if !qjs::is_undefined(cp) && !qjs::is_null(cp) {
                            qjs::JS_ToInt32(ctx, &mut ecode, cp);
                        }
                        qjs::sofuu_js_free_value(ctx, cp);
                    } else if !qjs::is_undefined(err) && !qjs::is_null(err) {
                        /* Some gateways send "error": "<string>" */
                        if let Some(s) = cstr_opt(ctx, err) {
                            if !s.is_empty() && s != "undefined" { emsg = Some(s); }
                        }
                    }
                    qjs::sofuu_js_free_value(ctx, err);
                    qjs::sofuu_js_free_value(ctx, ev_err);
                    if emsg.is_none() && had_err {
                        /* Error object without a message — keep the raw frame
                         * so SOMETHING reaches the user. */
                        emsg = Some(String::from_utf8_lossy(data).chars().take(200).collect());
                    }
                    if let Some(m) = emsg {
                        let mut ebuf = if ecode >= 400 {
                            format!("HTTP {}", ecode)
                        } else {
                            String::from("stream error")
                        };
                        ebuf.push_str(" — ");
                        ebuf.push_str(&m);
                        if ecode == 429 {
                            ebuf.push_str(" (rate limited — retry after a moment or check your quota)");
                        }
                        (*req).stream_err = Some(ebuf);
                        return 0; /* abort → CURLMSG_DONE with the stashed message */
                    }
                }
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
            capture_finish_reason(ctx, (*req).provider, req, data);
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
                    /* P3 (AUDIT-2026-09-07): JS_NewStringLen — CString::new
                     * fails on an interior NUL and unwrap_or_default() then
                     * rejected the promise with an EMPTY error payload. */
                    let err = qjs::JS_NewStringLen(
                        ctx,
                        raw.as_ptr() as *const c_char,
                        raw.len(),
                    );
                    sofuu_promise_reject((*req).promise, err);
                } else {
                    /* extract_text returns the stripped answer + the think
                       text — the C wrote thinking into the per-request
                       think_buf (no shared global → no data race). */
                    let (text, think) = extract_text((*req).provider, raw);
                    let result = qjs::sofuu_js_new_object(ctx);
                    /* P3 (AUDIT-2026-09-07): JS_NewStringLen — CString::new
                     * fails on an interior NUL and unwrap_or_default() then
                     * surfaced an EMPTY text (same class as proc-10). */
                    let text_js = qjs::JS_NewStringLen(
                        ctx,
                        text.as_ptr() as *const c_char,
                        text.len(),
                    );
                    qjs::sofuu_js_set_property_str(ctx, result, c"text".as_ptr(), text_js);

                    /* native_think: Anthropic extended-thinking field */
                    let native_think = extract_json_field(raw, "thinking");
                    if !native_think.is_empty() {
                        let nt_js = qjs::JS_NewStringLen(
                            ctx,
                            native_think.as_ptr() as *const c_char,
                            native_think.len(),
                        );
                        qjs::sofuu_js_set_property_str(ctx, result, c"thinking".as_ptr(), nt_js);
                    } else if !think.is_empty() {
                        let th_js = qjs::JS_NewStringLen(
                            ctx,
                            think.as_ptr() as *const c_char,
                            think.len(),
                        );
                        qjs::sofuu_js_set_property_str(ctx, result, c"thinking".as_ptr(), th_js);
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
                    /* P3 (AUDIT-2026-09-07): JS_NewStringLen — CString::new
                     * fails on an interior NUL and unwrap_or_default()
                     * surfaced an EMPTY raw payload (same class as proc-10). */
                    let raw_js = qjs::JS_NewStringLen(
                        ctx,
                        raw.as_ptr() as *const c_char,
                        raw.len(),
                    );
                    qjs::sofuu_js_set_property_str(ctx, result, c"raw".as_ptr(), raw_js);
                    sofuu_promise_resolve((*req).promise, result);
                    qjs::sofuu_js_free_value(ctx, result);
                }
            } else {
                let err = if code == curl::CURLE_OPERATION_TIMEDOUT {
                    let m = CString::new(
                        ai_timeout_message(easy_http_code(easy), ai_stall_timeout_secs()),
                    )
                    .unwrap_or_default();
                    qjs::sofuu_js_new_string(ctx, m.as_ptr())
                } else {
                    qjs::sofuu_js_new_string(ctx, curl::curl_easy_strerror(code))
                };
                sofuu_promise_reject((*req).promise, err);
            }

            active_unlink(req as *mut c_void);
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
                    /* P3 (AUDIT-2026-09-07): JS_NewStringLen — CString::new
                     * fails on an interior NUL and unwrap_or_default() then
                     * rejected with an EMPTY error payload. */
                    sofuu_promise_reject(
                        (*req).promise,
                        qjs::JS_NewStringLen(
                            ctx,
                            raw.as_ptr() as *const c_char,
                            raw.len(),
                        ),
                    );
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
                let err = if code == curl::CURLE_OPERATION_TIMEDOUT {
                    let m = CString::new(
                        ai_timeout_message(easy_http_code(easy), ai_stall_timeout_secs()),
                    )
                    .unwrap_or_default();
                    qjs::sofuu_js_new_string(ctx, m.as_ptr())
                } else {
                    qjs::sofuu_js_new_string(ctx, curl::curl_easy_strerror(code))
                };
                sofuu_promise_reject((*req).promise, err);
            }

            active_unlink(req as *mut c_void);
            if !(*req).headers.is_null() {
                curl::curl_slist_free_all((*req).headers);
            }
            drop(Box::from_raw(req));
        } else if tag == REQ_TAG_AUDIO {
            let req = base as *mut AiAudioReq;
            let ctx = (*req).ctx;

            if code == curl::CURLE_OK {
                let mut status: c_long = 0;
                curl::curl_easy_getinfo(easy, curl::CURLINFO_RESPONSE_CODE, &mut status as *mut c_long);
                let raw: &[u8] = &(*req).response_body;

                if status >= 400 {
                    sofuu_promise_reject(
                        (*req).promise,
                        qjs::JS_NewStringLen(
                            ctx,
                            raw.as_ptr() as *const c_char,
                            raw.len(),
                        ),
                    );
                } else if (*req).kind == 0 {
                    // Transcribe: JSON {"text": "..."}.
                    let result = extract_transcript(ctx, raw);
                    if qjs::is_undefined(result) {
                        sofuu_promise_reject(
                            (*req).promise,
                            qjs::sofuu_js_new_string(ctx, c"Failed to parse transcription.".as_ptr()),
                        );
                    } else {
                        sofuu_promise_resolve((*req).promise, result);
                        qjs::sofuu_js_free_value(ctx, result);
                    }
                } else {
                    // Speak: raw audio bytes → {audio: Uint8Array, format}.
                    let result = audio_bytes_result(ctx, raw, (*req).format.as_str());
                    if qjs::is_undefined(result) {
                        sofuu_promise_reject(
                            (*req).promise,
                            qjs::sofuu_js_new_string(ctx, c"Failed to package spoken audio.".as_ptr()),
                        );
                    } else {
                        sofuu_promise_resolve((*req).promise, result);
                        qjs::sofuu_js_free_value(ctx, result);
                    }
                }
            } else {
                let err = if code == curl::CURLE_OPERATION_TIMEDOUT {
                    let m = CString::new(
                        ai_timeout_message(easy_http_code(easy), ai_stall_timeout_secs()),
                    )
                    .unwrap_or_default();
                    qjs::sofuu_js_new_string(ctx, m.as_ptr())
                } else {
                    qjs::sofuu_js_new_string(ctx, curl::curl_easy_strerror(code))
                };
                sofuu_promise_reject((*req).promise, err);
            }

            active_unlink(req as *mut c_void);
            audio_req_free_mime(req);
            if !(*req).headers.is_null() {
                curl::curl_slist_free_all((*req).headers);
            }
            drop(Box::from_raw(req));
        } else if tag == REQ_TAG_STREAM {
            let req = base as *mut AiStreamReq;
            let ctx = (*req).ctx;

            if code != curl::CURLE_OK {
                /* net-11: once [DONE] signalled, the turn is committed —
                 * a late transfer error (e.g. CURLE_PARTIAL_FILE after the
                 * answer) must not overlay it. Same rule as stall_abort. */
                if (*req).done_called == 0 && qjs::JS_IsFunction(ctx, (*req).error_fn) != 0 {
                    /* An aborted-by-us transfer (in-stream error frame)
                     * carries the real provider message; a timeout gets the
                     * stall/connect explanation; anything else the plain
                     * curl error string. */
                    let timeout_msg = if code == curl::CURLE_OPERATION_TIMEDOUT {
                        Some(ai_timeout_message(easy_http_code(easy), ai_stall_timeout_secs()))
                    } else {
                        None
                    };
                    let e = match (*req).stream_err.as_deref().map(|s| s.to_string()).or(timeout_msg) {
                        Some(m) => {
                            let cm = CString::new(m).unwrap_or_default();
                            qjs::sofuu_js_new_string(ctx, cm.as_ptr())
                        }
                        None => qjs::sofuu_js_new_string(ctx, curl::curl_easy_strerror(code)),
                    };
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
                    /* net-11: an exact-length 4xx/5xx body can itself carry
                     * a `data: [DONE]` frame (the write callback parses
                     * frames regardless of status) — done wins over the
                     * status error. */
                    if (*req).done_called == 0 && qjs::JS_IsFunction(ctx, (*req).error_fn) != 0 {
                        let raw = (*req).error_body.clone();
                        let mut detail = String::new();
                        // Try JSON {error:{message,code}} or {message}
                        if !raw.is_empty() {
                            let body_str = String::from_utf8_lossy(&raw);
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
                    /* OpenAI-wire providers (openai | custom): a clean EOF
                     * with NO [DONE] and NO captured finish_reason is a
                     * truncated transfer, not a complete answer — calling
                     * done here made the factory return the partial text
                     * as "success" and the retry/CONTINUE ladder never
                     * fired (bench S3 rcut: partial text then abrupt
                     * res.end()). Route it to error_fn with a
                     * transient-classified message ("transfer closed") so
                     * streamWithRetry retries or continues from the
                     * partial. Anthropic ends with message_stop and Local
                     * providers end without either marker — for them (and
                     * when no error_fn exists, or a finish_reason WAS
                     * captured) the original ensure-done path stands. */
                    if matches!((*req).provider, Provider::OpenAi | Provider::Custom)
                        && (*req).finish_len == 0
                        && qjs::JS_IsFunction(ctx, (*req).error_fn) != 0
                    {
                        (*req).done_called = 1; /* committed: nothing re-fires */
                        let emsg = CString::new(
                            "stream ended without [DONE] (transfer closed) — possible truncation",
                        )
                        .unwrap_or_default();
                        let e = qjs::sofuu_js_new_string(ctx, emsg.as_ptr());
                        let r = qjs::JS_Call(ctx, (*req).error_fn, qjs::sofuu_js_undefined(), 1, &e);
                        if qjs::is_exception(r) {
                            qjs::js_std_dump_error(ctx);
                        }
                        qjs::sofuu_js_free_value(ctx, e);
                        qjs::sofuu_js_free_value(ctx, r);
                    } else {
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
                            set_stream_finish(ctx, stats, req);
                            let r = qjs::JS_Call(ctx, (*req).done_fn, qjs::sofuu_js_undefined(), 1, &stats);
                            if qjs::is_exception(r) {
                                qjs::js_std_dump_error(ctx);
                            }
                            qjs::sofuu_js_free_value(ctx, r);
                            qjs::sofuu_js_free_value(ctx, stats);
                        }
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
            images: Vec::new(),
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
                    /* images: optional array of data-URL strings on user
                     * messages — JSON-stringified like tool_calls so the
                     * body builders can split it without re-parsing. */
                    let imgs = qjs::sofuu_js_get_property_str(ctx, msg, c"images".as_ptr());
                    let images = if qjs::JS_IsArray(ctx, imgs) != 0 {
                        let json = qjs::JS_JSONStringify(
                            ctx,
                            imgs,
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
                    } else {
                        None
                    };
                    qjs::sofuu_js_free_value(ctx, r);
                    qjs::sofuu_js_free_value(ctx, c);
                    qjs::sofuu_js_free_value(ctx, tc);
                    qjs::sofuu_js_free_value(ctx, tid);
                    qjs::sofuu_js_free_value(ctx, imgs);
                    messages.push(AiMessage { role, content, tool_calls, tool_call_id, images: parse_image_urls(images.as_deref()) });
                } else {
                    messages.push(AiMessage { role: None, content: None, tool_calls: None, tool_call_id: None, images: Vec::new() });
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
    /* E2: headless defaults. A host that set `provider`/`model`/`base_url` in
     * sofuu_rt_new config (there is no interactive /model picker inside an
     * app) gets them here — but ONLY when the call did not specify them, so
     * an explicit per-call value always wins. Chat/CLI is unaffected: it
     * never installs runtime settings, so these are always None there and
     * the "no model configured" error still fires. */
    if provider_str.is_none() {
        if let Some(p) = crate::embed_config::default_provider() {
            cfg.provider = parse_provider(Some(&p));
        }
    }
    if cfg.model.is_none() {
        cfg.model = crate::embed_config::default_model();
    }
    if cfg.base_url.is_none() {
        cfg.base_url = crate::embed_config::default_base_url();
    }
    /* Still no silent model substitution: with no default configured either,
     * a missing model surfaces as an actionable error (see require_model). */

    Ok(cfg)
}

unsafe fn parse_embed_args(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
) -> Result<AiEmbedConfig, ()> {
    let mut cfg = AiEmbedConfig {
        provider: Provider::OpenAi,
        provider_explicit: false,
        model: None,
        api_key: None,
        base_url: None,
        space: None,
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

        /* H-E1: local embedding space ("sem2-64" default, "sem1-64",
         * "hash-768"). Only meaningful on the local path. */
        v = qjs::sofuu_js_get_property_str(ctx, opts, c"space".as_ptr());
        if !qjs::is_undefined(v) {
            cfg.space = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);
    }

    cfg.provider_explicit = provider_str.is_some();
    cfg.provider = parse_provider(provider_str.as_deref());
    /* E2: same headless defaults as the completion path, applied only when
     * the call omitted them. Lets a host pin one provider/model/endpoint for
     * the whole embedded runtime (e.g. a local OpenAI-compatible server). */
    if !cfg.provider_explicit {
        if let Some(p) = crate::embed_config::default_provider() {
            cfg.provider = parse_provider(Some(&p));
            cfg.provider_explicit = true;
        }
    }
    if cfg.model.is_none() {
        cfg.model = crate::embed_config::default_model();
    }
    if cfg.base_url.is_none() {
        cfg.base_url = crate::embed_config::default_base_url();
    }
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

    /* P3 (AUDIT-2026-09-07): curl_easy_init() can return NULL (global init
     * failure / OOM) — an unchecked handle made every setopt a silent no-op
     * and the request die with an opaque curl error. Init here, before
     * anything is allocated, so the failure path is a clean throw. */
    let easy = curl::curl_easy_init();
    if easy.is_null() {
        qjs::JS_ThrowInternalError(ctx, c"ai.complete: curl init failed".as_ptr());
        return qjs::sofuu_js_exception();
    }

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
        wd_next: ptr::null_mut(),
        last_rx: std::time::Instant::now(),
        ctx,
        promise: ptr::null_mut(),
        provider: eff,
        response_body: Vec::new(),
        headers: build_headers(eff, api_key.as_deref()),
        post_body: Some(CString::new(build_request_body(&cfg)).unwrap_or_default()),
        easy: ptr::null_mut(),
    });

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
    /* A remote endpoint must never hang the CLI forever — but an active
     * generation must never be killed mid-thought either: stall guard
     * (5 min of silence → abort with a clear error), bounded connect
     * phase, no total cap unless the caller opts in (opts.timeout_ms). */
    ai_set_patience(easy, cfg.timeout_ms);
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
    active_link(req_ptr as *mut c_void); /* stall watchdog registry */

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

    /* P3 (AUDIT-2026-09-07): curl_easy_init() can return NULL (global init
     * failure / OOM) — an unchecked handle made every setopt a silent no-op
     * and the request die with an opaque curl error. Init here, before
     * anything is allocated, so the failure path is a clean throw. */
    let easy = curl::curl_easy_init();
    if easy.is_null() {
        qjs::JS_ThrowInternalError(ctx, c"ai.stream: curl init failed".as_ptr());
        return qjs::sofuu_js_exception();
    }

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
        wd_next: ptr::null_mut(),
        last_rx: std::time::Instant::now(),
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
        finish_reason: [0u8; 24],
        finish_len: 0usize,
        stream_err: None,
        sid,
        next: ptr::null_mut(),
    });

    qjs::sofuu_js_free_value(ctx, push_fn);
    qjs::sofuu_js_free_value(ctx, done_fn);
    qjs::sofuu_js_free_value(ctx, error_fn);
    qjs::sofuu_js_free_value(ctx, think_fn);
    qjs::sofuu_js_free_value(ctx, tool_calls_fn);

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
    /* A remote endpoint must never hang the CLI forever — but an active
     * generation must never be killed mid-thought either: stall guard
     * (5 min of silence → abort with a clear error), bounded connect
     * phase, no total cap unless the caller opts in (opts.timeout_ms). */
    ai_set_patience(easy, cfg.timeout_ms);
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
        qjs::sofuu_js_free_value(ctx, req.tool_calls_fn);
        drop(req);
        qjs::sofuu_js_free_value(ctx, iterator);
        return qjs::sofuu_js_exception();
    }

    /* Abortable: Esc/Ctrl-C in the chat can now find and stop this stream. */
    let req_ptr = Box::into_raw(req);
    (*req_ptr).next = G_STREAMS.with(|s| s.get());
    G_STREAMS.with(|s| s.set(req_ptr));
    /* Stall watchdog registry: a silent provider must surface an error at
     * the patience cap, not hang the turn forever. */
    active_link(req_ptr as *mut c_void);
    qjs::sofuu_js_set_property_str(ctx, iterator, c"_sid".as_ptr(), qjs::sofuu_js_new_int64(ctx, sid));

    let mut running: c_int = 0;
    curl::curl_multi_socket_action(g_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running);
    ai_check_multi();

    iterator
}

/* ------------------------------------------------------------------ */
/* H-E1: local embedding path for ai.embed / ai.embedBatch             */
/* ------------------------------------------------------------------ */

/// Space tag: 0 = SEM2-64 (default), 1 = SEM1-64, 2 = hash-768.
unsafe fn check_embed_space(ctx: *mut JSContext, space: Option<&str>) -> Result<u8, JSValue> {
    match space {
        None | Some("") | Some("sem2-64") | Some("sem2") => Ok(0),
        Some("sem1-64") | Some("sem1") => Ok(1),
        Some("hash-768") | Some("hash") => Ok(2),
        _ => Err(qjs::JS_ThrowTypeError(
            ctx,
            c"ai.embed: unknown space (\"sem2-64\" default, \"sem1-64\", \"hash-768\")".as_ptr(),
        )),
    }
}

unsafe fn embed_one_local(tag: u8, text: &str) -> Option<Vec<f32>> {
    match tag {
        0 => crate::embedding::semantic_v2::semantic_v2(text),
        1 => crate::embedding::semantic_v1(text),
        _ => {
            let mut h = vec![0f32; crate::embedding::HASH_DIM];
            crate::embedding::hash_v1_features_into(text, &mut h);
            Some(h)
        }
    }
}

/// Build a Float32Array from Rust floats (fast typed-buffer path with a
/// property fallback, mirroring js_ai_embed_local). Returns an exception
/// value on failure.
unsafe fn new_f32array(ctx: *mut JSContext, vals: &[f32]) -> JSValue {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let f32_ctor = qjs::sofuu_js_get_property_str(ctx, global, c"Float32Array".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    let len_arg = qjs::sofuu_js_new_int32(ctx, vals.len() as i32);
    let arr = qjs::JS_CallConstructor(ctx, f32_ctor, 1, &len_arg);
    qjs::sofuu_js_free_value(ctx, len_arg);
    qjs::sofuu_js_free_value(ctx, f32_ctor);
    if qjs::is_exception(arr) || vals.is_empty() {
        return arr;
    }
    let mut byte_offset: usize = 0;
    let mut byte_length: usize = 0;
    let buf = qjs::JS_GetTypedArrayBuffer(ctx, arr, &mut byte_offset, &mut byte_length, ptr::null_mut());
    if qjs::is_exception(buf) {
        qjs::sofuu_js_free_value(ctx, arr);
        return qjs::sofuu_js_exception();
    }
    let mut buf_size: usize = 0;
    let p = qjs::JS_GetArrayBuffer(ctx, &mut buf_size, buf);
    if p.is_null() || byte_offset + vals.len() * std::mem::size_of::<f32>() > buf_size {
        qjs::sofuu_js_free_value(ctx, buf);
        qjs::sofuu_js_free_value(ctx, arr);
        return qjs::JS_ThrowTypeError(ctx, c"ai.embed: cannot access typed-array buffer".as_ptr());
    }
    let out = std::slice::from_raw_parts_mut(p.add(byte_offset) as *mut f32, vals.len());
    out.copy_from_slice(vals);
    qjs::sofuu_js_free_value(ctx, buf);
    arr
}

/// Resolve the local path: embed `inputs` in `space` and return an
/// already-resolved Promise (single Float32Array, or Array of them when
/// `always_array` or more than one input). Unknown space throws (arg
/// validation); a missing baked model rejects (runtime failure).
unsafe fn resolve_local_embed(
    ctx: *mut JSContext,
    inputs: &[String],
    space: Option<&str>,
    always_array: bool,
) -> JSValue {
    let tag = match check_embed_space(ctx, space) {
        Ok(t) => t,
        Err(exc) => return exc,
    };
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
    for t in inputs {
        match embed_one_local(tag, t) {
            Some(v) => vecs.push(v),
            None => {
                let mut h: *mut PromiseHandle = ptr::null_mut();
                let p = sofuu_promise_new(ctx, &mut h);
                if qjs::is_exception(p) {
                    return p;
                }
                let e = qjs::sofuu_js_new_string(ctx, c"ai.embed: bundled embedding model unavailable".as_ptr());
                sofuu_promise_reject(h, e);
                return p;
            }
        }
    }
    let value = if !always_array && vecs.len() == 1 {
        new_f32array(ctx, &vecs[0])
    } else {
        let arr = qjs::JS_NewArray(ctx);
        if qjs::is_exception(arr) {
            return arr;
        }
        let mut ok = true;
        for (i, v) in vecs.iter().enumerate() {
            let el = new_f32array(ctx, v);
            if qjs::is_exception(el) {
                qjs::sofuu_js_free_value(ctx, el);
                ok = false;
                break;
            }
            qjs::JS_SetPropertyUint32(ctx, arr, i as u32, el);
        }
        if !ok {
            qjs::sofuu_js_free_value(ctx, arr);
            return qjs::sofuu_js_exception();
        }
        arr
    };
    if qjs::is_exception(value) {
        return value;
    }
    let mut h: *mut PromiseHandle = ptr::null_mut();
    let p = sofuu_promise_new(ctx, &mut h);
    if qjs::is_exception(p) {
        qjs::sofuu_js_free_value(ctx, value);
        return p;
    }
    // resolve frees the handle (promise.rs); the value ref transfers in.
    sofuu_promise_resolve(h, value);
    qjs::sofuu_js_free_value(ctx, value);
    p
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

    /* P3 (AUDIT-2026-09-07): curl_easy_init() can return NULL (global init
     * failure / OOM) — an unchecked handle made every setopt a silent no-op
     * and the request die with an opaque curl error. Init here, before
     * anything is allocated, so the failure path is a clean throw. */
    let easy = curl::curl_easy_init();
    if easy.is_null() {
        qjs::JS_ThrowInternalError(ctx, c"ai.embed: curl init failed".as_ptr());
        return qjs::sofuu_js_exception();
    }

    /* H-E1: bundled local embeddings. Explicit provider:"local" (incl.
     * legacy "ollama") embeds offline, and so does a call naming no
     * provider, model, key, or URL at all (local default — zero-key
     * offline). Anything else takes the network path below. */
    let want_local = cfg.provider == Provider::Local
        || (!cfg.provider_explicit
            && cfg.model.is_none()
            && cfg.base_url.is_none()
            && cfg.api_key.is_none());
    if want_local {
        return resolve_local_embed(ctx, &cfg.inputs, cfg.space.as_deref(), false);
    }
    if !require_model(cfg.model.as_deref()) {
        return qjs::JS_ThrowTypeError(ctx, NO_MODEL_MSG.as_ptr());
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
        wd_next: ptr::null_mut(),
        last_rx: std::time::Instant::now(),
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

    /* H-E1: Provider::Local never reaches here — the local path returns
     * above. (P3 AUDIT-2026-09-07: the old Local arm was dead code behind
     * the Track-B reject; the reject is gone, the early return replaces it.) */
    let mut url: String = if cfg.provider == Provider::OpenAi {
        "https://api.openai.com/v1/embeddings".to_string()
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
    /* Same patience as complete/stream: embeddings may be slow, but a
     * silent one must not hang the brain forever. */
    ai_set_patience(easy, 0);
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
    active_link(req_ptr as *mut c_void); /* stall watchdog registry */

    let mut running: c_int = 0;
    curl::curl_multi_socket_action(g_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running);
    ai_check_multi();

    promise
}

/* ------------------------------------------------------------------ */
/* H-E1 JS: sofuu.ai.embedBatch(texts[], opts?) → Promise<Array>        */
/* Local-only vectorized batch (single lock, no JSON). Remote callers   */
/* keep ai.embed([...]) with an explicit provider.                      */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ai_embed_batch(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || qjs::JS_IsArray(ctx, *argv) == 0 {
        return qjs::JS_ThrowTypeError(ctx, c"ai.embedBatch(string[]) expected an array of strings".as_ptr());
    }
    let cfg = match parse_embed_args(ctx, argc, argv) {
        Ok(c) => c,
        Err(()) => return qjs::sofuu_js_exception(),
    };
    /* Local-only: any remote signal (explicit non-local provider, model,
     * URL, key) is a caller error — remote batching lives on ai.embed([...]).
     * No provider named = local, mirroring ai.embed's default. */
    let remote_signal = (cfg.provider_explicit && cfg.provider != Provider::Local)
        || cfg.model.is_some()
        || cfg.base_url.is_some()
        || cfg.api_key.is_some();
    if remote_signal {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"ai.embedBatch is local-only; use ai.embed([...], { provider, model }) for remote providers".as_ptr(),
        );
    }
    resolve_local_embed(ctx, &cfg.inputs, cfg.space.as_deref(), true)
}

/* ------------------------------------------------------------------ */
/* M1 JS: sofuu.ai.embedImage(bytes) → Float32Array (sync)              */
/* Image bytes (Uint8Array, e.g. from sofuu.fs.readFileBytes) → 64-dim  */
/* unit vector in img1-64 space (joint with sem2-64 text geometry).     */
/* Throws on undecodable bytes or a missing baked model.               */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ai_embed_image(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"ai.embedImage(bytes) expected a Uint8Array".as_ptr());
    }
    let mut byte_offset: usize = 0;
    let mut byte_length: usize = 0;
    let buf = qjs::JS_GetTypedArrayBuffer(ctx, *argv, &mut byte_offset, &mut byte_length, ptr::null_mut());
    if qjs::is_exception(buf) {
        qjs::sofuu_js_free_value(ctx, buf);
        return qjs::JS_ThrowTypeError(ctx, c"ai.embedImage(bytes) expected a Uint8Array".as_ptr());
    }
    let mut buf_size: usize = 0;
    let p = qjs::JS_GetArrayBuffer(ctx, &mut buf_size, buf);
    if p.is_null() || byte_offset + byte_length > buf_size {
        qjs::sofuu_js_free_value(ctx, buf);
        return qjs::JS_ThrowTypeError(ctx, c"ai.embedImage: cannot access byte buffer".as_ptr());
    }
    let bytes = std::slice::from_raw_parts(p.add(byte_offset), byte_length).to_vec();
    qjs::sofuu_js_free_value(ctx, buf);
    match crate::embedding::image::semantic_img(&bytes) {
        Some(v) => new_f32array(ctx, &v),
        None => qjs::JS_ThrowTypeError(
            ctx,
            c"ai.embedImage: undecodable image (PNG/JPEG, 2x2..2048px) or IMG1 model unavailable".as_ptr(),
        ),
    }
}

/* ------------------------------------------------------------------ */
/* M2: provider audio — ai.transcribe / ai.speak (OpenAI-compatible)    */
/* ------------------------------------------------------------------ */

/// M2: 32MB input cap (readFileBytes allows 256MB; audio endpoints don't
/// want novels — fail fast with a TypeError, not a 413 mid-transfer).
const AUDIO_MAX_INPUT: usize = 32 * 1024 * 1024;

/// Auth headers WITHOUT Content-Type (multipart sets its own boundary).
unsafe fn build_auth_headers(p: Provider, api_key: Option<&str>) -> *mut CurlSlist {
    let mut h: *mut CurlSlist = ptr::null_mut();
    if let Some(key) = api_key {
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

/// Derive an audio endpoint from a chat-style base URL:
/// …/chat/completions → …/audio/<leaf>; otherwise append /audio/<leaf>.
fn audio_url_from_base(base: &str, leaf: &str) -> String {
    let b = base.trim_end_matches('/');
    if let Some(root) = b.strip_suffix("/chat/completions") {
        format!("{root}/audio/{leaf}")
    } else {
        format!("{b}/audio/{leaf}")
    }
}

fn audio_default_url(provider: Provider, leaf: &str) -> String {
    // Only OpenAi has a built-in audio endpoint; every other provider must
    // supply base_url (enforced by check_audio_provider). The parameter
    // stays so callers read uniformly.
    let _ = provider;
    format!("https://api.openai.com/v1/audio/{leaf}")
}

/// Guess a filename extension → (filename, mime). Providers sniff format
/// off the extension, so never send extensionless parts.
fn audio_part_name(filename: Option<&str>) -> (CString, CString) {
    let name = filename.filter(|s| !s.trim().is_empty()).unwrap_or("audio.wav");
    let lower = name.to_lowercase();
    let mime = if lower.ends_with(".mp3") {
        "audio/mpeg"
    } else if lower.ends_with(".m4a") || lower.ends_with(".mp4") {
        "audio/mp4"
    } else if lower.ends_with(".ogg") || lower.ends_with(".oga") {
        "audio/ogg"
    } else if lower.ends_with(".flac") {
        "audio/flac"
    } else if lower.ends_with(".webm") {
        "audio/webm"
    } else {
        "audio/wav"
    };
    (
        CString::new(name).unwrap_or_default(),
        CString::new(mime).unwrap_or_default(),
    )
}

/// Shared audio opts: provider/model/api_key/base_url (+ voice/format/
/// language/filename per call). Returns Err(()) after throwing.
struct AudioOpts {
    provider: Provider,
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    extra: Option<String>, // language (transcribe) or voice (speak)
    format: Option<String>,
    filename: Option<String>,
}

unsafe fn parse_audio_opts(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
    what: &str,
) -> Result<AudioOpts, ()> {
    let mut o = AudioOpts {
        provider: Provider::OpenAi,
        model: None,
        api_key: None,
        base_url: None,
        extra: None,
        format: None,
        filename: None,
    };
    if argc > 1 && qjs::is_object(*argv.add(1)) {
        let opts = *argv.add(1);
        let mut v = qjs::sofuu_js_get_property_str(ctx, opts, c"provider".as_ptr());
        let provider_str = if !qjs::is_undefined(v) { cstr_opt(ctx, v) } else { None };
        qjs::sofuu_js_free_value(ctx, v);
        v = qjs::sofuu_js_get_property_str(ctx, opts, c"model".as_ptr());
        if !qjs::is_undefined(v) {
            o.model = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);
        v = qjs::sofuu_js_get_property_str(ctx, opts, c"api_key".as_ptr());
        if !qjs::is_undefined(v) {
            o.api_key = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);
        v = qjs::sofuu_js_get_property_str(ctx, opts, c"base_url".as_ptr());
        if !qjs::is_undefined(v) {
            o.base_url = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);
        // Per-call knobs share two slots; the caller names them.
        let (extra_key, fmt_key, file_key) = if what == "transcribe" {
            ("language", "", "filename")
        } else {
            ("voice", "format", "")
        };
        v = qjs::sofuu_js_get_property_str(
            ctx,
            opts,
            CString::new(extra_key).unwrap_or_default().as_ptr(),
        );
        if !qjs::is_undefined(v) {
            o.extra = cstr_opt(ctx, v);
        }
        qjs::sofuu_js_free_value(ctx, v);
        if !fmt_key.is_empty() {
            v = qjs::sofuu_js_get_property_str(
                ctx,
                opts,
                CString::new(fmt_key).unwrap_or_default().as_ptr(),
            );
            if !qjs::is_undefined(v) {
                o.format = cstr_opt(ctx, v);
            }
            qjs::sofuu_js_free_value(ctx, v);
        }
        if !file_key.is_empty() {
            v = qjs::sofuu_js_get_property_str(
                ctx,
                opts,
                CString::new(file_key).unwrap_or_default().as_ptr(),
            );
            if !qjs::is_undefined(v) {
                o.filename = cstr_opt(ctx, v);
            }
            qjs::sofuu_js_free_value(ctx, v);
        }
        o.provider = parse_provider(provider_str.as_deref());
    }
    Ok(o)
}

/// Validate provider/model for audio (no local STT in-binary; Anthropic
/// has no audio API; custom needs a base URL). Throws + Err on failure.
unsafe fn check_audio_provider(
    ctx: *mut JSContext,
    o: &AudioOpts,
    what: &str,
) -> Result<(), ()> {
    if o.provider == Provider::Local {
        let msg = format!("ai.{what}: no bundled speech model — point provider at an OpenAI-compatible audio endpoint");
        let c = CString::new(msg).unwrap_or_default();
        qjs::JS_ThrowTypeError(ctx, c.as_ptr());
        return Err(());
    }
    if o.provider == Provider::Anthropic {
        let msg = format!("ai.{what}: Anthropic provides no audio API");
        let c = CString::new(msg).unwrap_or_default();
        qjs::JS_ThrowTypeError(ctx, c.as_ptr());
        return Err(());
    }
    if !require_model(o.model.as_deref()) {
        qjs::JS_ThrowTypeError(ctx, NO_MODEL_MSG.as_ptr());
        return Err(());
    }
    if o.provider == Provider::Custom
        && !matches!(o.base_url.as_deref(), Some(u) if !u.is_empty())
    {
        qjs::JS_ThrowTypeError(
            ctx,
            c"custom provider needs a base URL (use /provider wizard or /baseurl)".as_ptr(),
        );
        return Err(());
    }
    Ok(())
}

/// Launch an audio request (mime or post_body pre-attached to `req`;
/// caller sets the URL + body kind first). Returns the JS Promise.
unsafe fn audio_launch(
    ctx: *mut JSContext,
    easy: *mut Curl,
    mut req: Box<AiAudioReq>,
    url: String,
    is_mime: bool,
) -> JSValue {
    req.easy = easy;
    let url_c = CString::new(url).unwrap_or_default();
    curl::curl_easy_setopt(easy, curl::CURLOPT_URL, url_c.as_ptr());
    curl::curl_easy_setopt(easy, curl::CURLOPT_HTTPHEADER, req.headers);
    if is_mime {
        curl::curl_easy_setopt(easy, curl::CURLOPT_MIMEPOST, req.mime);
    } else if let Some(body) = req.post_body.as_ref() {
        curl::curl_easy_setopt(easy, curl::CURLOPT_POSTFIELDS, body.as_ptr());
        curl::curl_easy_setopt(
            easy,
            curl::CURLOPT_POSTFIELDSIZE,
            body.as_bytes().len() as c_long,
        );
    }
    curl::curl_easy_setopt(easy, curl::CURLOPT_WRITEFUNCTION, complete_write_cb as *const c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_WRITEDATA, &mut *req as *mut AiAudioReq as *mut c_void);
    curl::curl_easy_setopt(easy, curl::CURLOPT_SSL_VERIFYPEER, 1 as c_long);
    curl::curl_easy_setopt(easy, curl::CURLOPT_MAXFILESIZE, AI_MAX_RESPONSE as c_long);
    curl::curl_easy_setopt(easy, curl::CURLOPT_PRIVATE, &mut *req as *mut AiAudioReq as *mut c_void);
    ai_set_patience(easy, 0);
    let promise = sofuu_promise_new(ctx, &mut req.promise);
    let req_ptr = Box::into_raw(req);

    ai_ensure_init();
    let mcode = curl::curl_multi_add_handle(g_multi(), (*req_ptr).easy);
    if mcode != curl::CURLM_OK {
        curl::curl_easy_cleanup((*req_ptr).easy);
        audio_req_free_mime(req_ptr);
        if !(*req_ptr).headers.is_null() {
            curl::curl_slist_free_all((*req_ptr).headers);
        }
        let err = qjs::sofuu_js_new_string(ctx, curl::curl_multi_strerror(mcode));
        sofuu_promise_reject((*req_ptr).promise, err);
        drop(Box::from_raw(req_ptr));
        return promise;
    }
    active_link(req_ptr as *mut c_void);

    let mut running: c_int = 0;
    curl::curl_multi_socket_action(g_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running);
    ai_check_multi();

    promise
}

/* JS: sofuu.ai.transcribe(audioBytes, opts?) → Promise<{text}> */
unsafe extern "C" fn js_ai_transcribe(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"ai.transcribe(audioBytes, opts?) expected audio bytes".as_ptr());
    }
    // Headless bridge form: {audio_b64: "..."} (hosts base64 themselves;
    // the funnel stays JSON). Otherwise a Uint8Array.
    let audio = if qjs::is_object(*argv) {
        let b = qjs::sofuu_js_get_property_str(ctx, *argv, c"audio_b64".as_ptr());
        let is_b64 = !qjs::is_undefined(b) && qjs::sofuu_js_is_string(b) != 0;
        let s = if is_b64 { cstr_opt(ctx, b) } else { None };
        qjs::sofuu_js_free_value(ctx, b);
        if !is_b64 {
            match read_audio_bytes(ctx, *argv) {
                Ok(bytes) => bytes,
                Err(e) => return e,
            }
        } else {
            match s.as_deref().and_then(b64_decode) {
                Some(bytes) if !bytes.is_empty() => bytes,
                _ => {
                    return qjs::JS_ThrowTypeError(
                        ctx,
                        c"ai.transcribe: audio_b64 is not valid base64".as_ptr(),
                    )
                }
            }
        }
    } else {
        match read_audio_bytes(ctx, *argv) {
            Ok(b) => b,
            Err(e) => return e,
        }
    };
    if audio.len() > AUDIO_MAX_INPUT {
        return qjs::JS_ThrowTypeError(ctx, c"ai.transcribe: audio over the 32MB input cap".as_ptr());
    }
    let o = match parse_audio_opts(ctx, argc, argv, "transcribe") {
        Ok(o) => o,
        Err(()) => return qjs::sofuu_js_exception(),
    };
    if check_audio_provider(ctx, &o, "transcribe").is_err() {
        return qjs::sofuu_js_exception();
    }
    let easy = curl::curl_easy_init();
    if easy.is_null() {
        qjs::JS_ThrowInternalError(ctx, c"ai.transcribe: curl init failed".as_ptr());
        return qjs::sofuu_js_exception();
    }

    // Multipart: file + model + response_format=json (+ language).
    let mime = curl::curl_mime_init(easy);
    if mime.is_null() {
        curl::curl_easy_cleanup(easy);
        qjs::JS_ThrowInternalError(ctx, c"ai.transcribe: mime init failed".as_ptr());
        return qjs::sofuu_js_exception();
    }
    // From here every failure path must free the mime handle.
    let fail = |ctx: *mut JSContext,
                easy: *mut Curl,
                mime: *mut curl::CurlMime,
                msg: &std::ffi::CStr|
     -> JSValue {
        curl::curl_mime_free(mime);
        curl::curl_easy_cleanup(easy);
        qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
        qjs::sofuu_js_exception()
    };
    let model_c = CString::new(o.model.clone().unwrap_or_default()).unwrap_or_default();
    let (fname_c, ftype_c) = audio_part_name(o.filename.as_deref());
    // Already inside unsafe fn js_ai_transcribe — no nested block needed.
    let mut ok: bool;
    let p = curl::curl_mime_addpart(mime);
    ok = !p.is_null()
        && curl::curl_mime_name(p, c"file".as_ptr()) == 0
        && curl::curl_mime_filename(p, fname_c.as_ptr()) == 0
        && curl::curl_mime_type(p, ftype_c.as_ptr()) == 0
        && curl::curl_mime_data(p, audio.as_ptr() as *const c_char, audio.len()) == 0;
    if ok {
        let p = curl::curl_mime_addpart(mime);
        ok = !p.is_null()
            && curl::curl_mime_name(p, c"model".as_ptr()) == 0
            && curl::curl_mime_data(p, model_c.as_ptr(), model_c.as_bytes().len()) == 0;
    }
    if ok {
        let p = curl::curl_mime_addpart(mime);
        ok = !p.is_null()
            && curl::curl_mime_name(p, c"response_format".as_ptr()) == 0
            && curl::curl_mime_data(p, c"json".as_ptr(), 4) == 0;
    }
    if ok {
        if let Some(lang) = o.extra.as_deref().filter(|s| !s.is_empty()) {
            let lang_c = CString::new(lang).unwrap_or_default();
            let p = curl::curl_mime_addpart(mime);
            ok = !p.is_null()
                && curl::curl_mime_name(p, c"language".as_ptr()) == 0
                && curl::curl_mime_data(p, lang_c.as_ptr(), lang_c.as_bytes().len()) == 0;
        }
    }
    if !ok {
        return fail(ctx, easy, mime, c"ai.transcribe: failed to build multipart body");
    }

    let url = match o.base_url.as_deref().filter(|s| !s.is_empty()) {
        Some(bu) => audio_url_from_base(bu, "transcriptions"),
        None => audio_default_url(o.provider, "transcriptions"),
    };
    let req = Box::new(AiAudioReq {
        tag: REQ_TAG_AUDIO,
        wd_next: ptr::null_mut(),
        last_rx: std::time::Instant::now(),
        ctx,
        promise: ptr::null_mut(),
        provider: o.provider,
        response_body: Vec::new(),
        headers: build_auth_headers(
            o.provider,
            resolve_api_key(o.provider, o.api_key.as_deref()).as_deref(),
        ),
        post_body: None,
        mime,
        audio,
        kind: 0,
        format: String::new(),
        easy: ptr::null_mut(),
    });
    // curl_mime_data copies the payload, but the Box owns the bytes anyway.
    audio_launch(ctx, easy, req, url, true)
}

/* JS: sofuu.ai.speak(text, opts?) → Promise<{audio: Uint8Array, format}> */
unsafe extern "C" fn js_ai_speak(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || qjs::sofuu_js_is_string(*argv) == 0 {
        return qjs::JS_ThrowTypeError(ctx, c"ai.speak(text, opts?) expected a string".as_ptr());
    }
    let text = cstr_opt(ctx, *argv).unwrap_or_default();
    if text.trim().is_empty() {
        return qjs::JS_ThrowTypeError(ctx, c"ai.speak: text must not be empty".as_ptr());
    }
    let o = match parse_audio_opts(ctx, argc, argv, "speak") {
        Ok(o) => o,
        Err(()) => return qjs::sofuu_js_exception(),
    };
    if check_audio_provider(ctx, &o, "speak").is_err() {
        return qjs::sofuu_js_exception();
    }
    let voice = o.extra.as_deref().filter(|s| !s.trim().is_empty()).unwrap_or("alloy");
    let format = o.format.as_deref().filter(|s| !s.trim().is_empty()).unwrap_or("mp3");
    if !matches!(format, "mp3" | "opus" | "aac" | "flac" | "wav" | "pcm") {
        return qjs::JS_ThrowTypeError(ctx, c"ai.speak: format must be mp3|opus|aac|flac|wav|pcm".as_ptr());
    }
    let easy = curl::curl_easy_init();
    if easy.is_null() {
        qjs::JS_ThrowInternalError(ctx, c"ai.speak: curl init failed".as_ptr());
        return qjs::sofuu_js_exception();
    }
    let body = serde_json::json!({
        "model": o.model.as_deref().unwrap_or(""),
        "input": text,
        "voice": voice,
        "response_format": format,
    })
    .to_string();
    let url = match o.base_url.as_deref().filter(|s| !s.is_empty()) {
        Some(bu) => audio_url_from_base(bu, "speech"),
        None => audio_default_url(o.provider, "speech"),
    };
    let req = Box::new(AiAudioReq {
        tag: REQ_TAG_AUDIO,
        wd_next: ptr::null_mut(),
        last_rx: std::time::Instant::now(),
        ctx,
        promise: ptr::null_mut(),
        provider: o.provider,
        response_body: Vec::new(),
        headers: build_headers(
            o.provider,
            resolve_api_key(o.provider, o.api_key.as_deref()).as_deref(),
        ),
        post_body: Some(CString::new(body).unwrap_or_default()),
        mime: ptr::null_mut(),
        audio: Vec::new(),
        kind: 1,
        format: format.to_string(),
        easy: ptr::null_mut(),
    });
    audio_launch(ctx, easy, req, url, false)
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
/* sofuu.ai.modelCaps(model, baseUrl?) → JSON string                    */
/* Per-model capability resolution (rt/model_caps.rs + the discovered   */
/* store): context window, max output tokens, thinking kind + ceiling  */
/* + effort levels. `baseUrl` (optional) is the endpoint the request    */
/* will hit — when present, caps the endpoint itself published for this */
/* exact model (harvested from its model listing) override the static   */
/* family table. Returns a JSON STRING (parse on the JS side).          */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ai_model_caps(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let model: Option<String> = if argc >= 1 && qjs::sofuu_js_is_string(*argv) != 0 {
        cstr_opt(ctx, *argv)
    } else {
        None
    };
    let base_url: Option<String> = if argc >= 2 && qjs::sofuu_js_is_string(*argv.add(1)) != 0 {
        cstr_opt(ctx, *argv.add(1))
    } else {
        None
    };
    let caps =
        crate::rt::model_caps::lookup_for(model.as_deref(), base_url.as_deref());
    let json = caps.to_json(model.as_deref().unwrap_or(""));
    match CString::new(json) {
        Ok(c) => qjs::sofuu_js_new_string(ctx, c.as_ptr()),
        Err(_) => qjs::sofuu_js_new_string(ctx, c"{}".as_ptr()),
    }
}

/* ------------------------------------------------------------------ */
/* sofuu.ai.resolveCaps(model, baseUrl?, cfgWindow?, cfgMaxOutput?,   */
/*                          explicit?) → JSON string                   */
/*                                                                     */
/* The SINGLE evidence-ladder resolve every budget consumer uses        */
/* (ring denominator, compaction trigger, attachment budgets, output   */
/* caps, fitGuard): learned-from-400s > endpoint-discovered > registry */
/* > conservative default.                                              */
/*                                                                     */
/* An INHERITED config number (config.json, carried across models)     */
/* shrinks to the strictest REAL evidence — a flat ctx_window=1M can   */
/* never shadow a 262k model's real window again (the P0 ring bug).    */
/* An EXPLICIT number (the 5th arg true — `/ctx 1000000` typed now) is */
/* obeyed as given and merely ADVISED against, so an explicit raise     */
/* never silently turns into a no-op. Returns a JSON STRING:            */
/*   {"window":N,"maxOutput":N,"known":bool,"source":"...",            */
/*    "clampedConfig":bool,"configExceedsEvidence":N|null,              */
/*    "winSource":"...","maxSource":"...","thinking":"..."}             */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ai_resolve_caps(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let model: Option<String> = if argc >= 1 && qjs::sofuu_js_is_string(*argv) != 0 {
        cstr_opt(ctx, *argv)
    } else {
        None
    };
    let base_url: Option<String> = if argc >= 2 && qjs::sofuu_js_is_string(*argv.add(1)) != 0 {
        cstr_opt(ctx, *argv.add(1))
    } else {
        None
    };
    let cfg_window: i64 = if argc >= 3 && qjs::is_number(*argv.add(2)) {
        let mut n: i64 = 0;
        qjs::JS_ToInt64(ctx, &mut n, *argv.add(2));
        n
    } else {
        0
    };
    let cfg_max_output: i64 = if argc >= 4 && qjs::is_number(*argv.add(3)) {
        let mut n: i64 = 0;
        qjs::JS_ToInt64(ctx, &mut n, *argv.add(3));
        n
    } else {
        0
    };

    let explicit: bool = argc >= 5 && qjs::is_bool(*argv.add(4)) && qjs::JS_ToBool(ctx, *argv.add(4)) != 0;

    let r = crate::ml::alloc::policy::resolve_explicit(
        model.as_deref(),
        cfg_window,
        cfg_max_output,
        base_url.as_deref(),
        explicit,
    );
    let thinking = crate::rt::model_caps::lookup(model.as_deref()).thinking;
    let t_kind = match thinking {
        crate::rt::model_caps::Thinking::None => "none",
        crate::rt::model_caps::Thinking::Effort(_) => "effort",
        crate::rt::model_caps::Thinking::Budget(_) => "budget",
        crate::rt::model_caps::Thinking::Unknown => "unknown",
    };
    let json = format!(
        "{{\"window\":{},\"maxOutput\":{},\"known\":{},\"source\":\"{}\",\"clampedConfig\":{},\"configExceedsEvidence\":{},\"winSource\":\"{}\",\"maxSource\":\"{}\",\"thinking\":\"{}\"}}",
        r.window,
        r.max_output,
        r.known,
        r.source.as_str(),
        r.clamped_config,
        r.config_exceeds_evidence
            .map(|n| n.to_string())
            .unwrap_or_else(|| "null".to_string()),
        r.win_source.as_str(),
        r.max_source.as_str(),
        t_kind
    );
    match CString::new(json) {
        Ok(c) => qjs::sofuu_js_new_string(ctx, c.as_ptr()),
        Err(_) => qjs::sofuu_js_new_string(ctx, c"{}".as_ptr()),
    }
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
/* sofuu.ai.embedLocalSemantic(text) → Float32Array                    */
/* The brain-facing compact learned projector.  The legacy embedLocal      */
/* above remains unchanged so callers that request arbitrary dimensions   */
/* keep their existing ABI.                                             */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ai_embed_local_semantic(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || qjs::sofuu_js_is_string(*argv) == 0 {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"ai.embedLocalSemantic(text) expected a string".as_ptr(),
        );
    }
    let text = match cstr_opt(ctx, *argv) {
        Some(t) => t,
        None => return qjs::sofuu_js_exception(),
    };
    let Some(values) = crate::embedding::semantic_v1(&text) else {
        return qjs::JS_ThrowInternalError(
            ctx,
            c"ai.embedLocalSemantic: baked model unavailable or invalid".as_ptr(),
        );
    };

    let global = qjs::sofuu_js_get_global_object(ctx);
    let f32_ctor = qjs::sofuu_js_get_property_str(ctx, global, c"Float32Array".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    let len_arg = qjs::sofuu_js_new_int32(ctx, crate::embedding::SEMANTIC_DIM as i32);
    let vec = qjs::JS_CallConstructor(ctx, f32_ctor, 1, &len_arg);
    qjs::sofuu_js_free_value(ctx, len_arg);
    qjs::sofuu_js_free_value(ctx, f32_ctor);
    if qjs::is_exception(vec) {
        return qjs::sofuu_js_exception();
    }

    let mut byte_offset: usize = 0;
    let mut byte_length: usize = 0;
    let buf = qjs::JS_GetTypedArrayBuffer(
        ctx,
        vec,
        &mut byte_offset,
        &mut byte_length,
        ptr::null_mut(),
    );
    if qjs::is_exception(buf) {
        qjs::sofuu_js_free_value(ctx, vec);
        return qjs::sofuu_js_exception();
    }
    let mut buf_size: usize = 0;
    let ptr = qjs::JS_GetArrayBuffer(ctx, &mut buf_size, buf);
    if !ptr.is_null()
        && byte_offset + values.len() * std::mem::size_of::<f32>() <= buf_size
        && byte_length >= values.len() * std::mem::size_of::<f32>()
    {
        let out = std::slice::from_raw_parts_mut(
            ptr.add(byte_offset) as *mut f32,
            values.len(),
        );
        out.copy_from_slice(&values);
        qjs::sofuu_js_free_value(ctx, buf);
        return vec;
    }
    qjs::sofuu_js_free_value(ctx, buf);
    for (i, &v) in values.iter().enumerate() {
        qjs::JS_SetPropertyUint32(ctx, vec, i as u32, qjs::sofuu_js_new_float64(ctx, v as f64));
    }
    vec
}

/// `sofuu.ai.embedInfo()` — stable identity for memory backends.  The JS
/// memory layer uses this before opening a brain so its dimension and model
/// id are always paired.
unsafe extern "C" fn js_ai_embed_info(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let json = crate::embedding::model_info_json();
    match CString::new(json) {
        Ok(c) => qjs::sofuu_js_new_string(ctx, c.as_ptr()),
        Err(_) => qjs::sofuu_js_new_string(ctx, c"{}".as_ptr()),
    }
}

/* ------------------------------------------------------------------ */
/* sofuu.ai.embedLocalSemanticV2(text) → Float32Array                  */
/* sofuu.ai.embedInfoV2() → JSON                                       */
/* The round-9 SEM2 table embedder — the semantic channel of the       */
/* fused brain.  Shipped alongside (never in place of) embedLocal       */
/* (hash-v1) and embedLocalSemantic (SEM1 v1); the old two keep their   */
/* exact semantics.  A missing/corrupt baked artifact throws — JS       */
/* treats that as "v2 unavailable" and runs hash-only.                  */
/* ------------------------------------------------------------------ */

/// Pack an f32 slice into a fresh JS Float32Array (the typed-array fast
/// path with the property-set fallback, mirroring embedLocalSemantic).
unsafe fn js_new_f32_array(ctx: *mut JSContext, values: &[f32]) -> JSValue {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let f32_ctor = qjs::sofuu_js_get_property_str(ctx, global, c"Float32Array".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    let len_arg = qjs::sofuu_js_new_int32(ctx, values.len() as i32);
    let vec = qjs::JS_CallConstructor(ctx, f32_ctor, 1, &len_arg);
    qjs::sofuu_js_free_value(ctx, len_arg);
    qjs::sofuu_js_free_value(ctx, f32_ctor);
    if qjs::is_exception(vec) {
        return qjs::sofuu_js_exception();
    }

    let mut byte_offset: usize = 0;
    let mut byte_length: usize = 0;
    let buf = qjs::JS_GetTypedArrayBuffer(
        ctx,
        vec,
        &mut byte_offset,
        &mut byte_length,
        ptr::null_mut(),
    );
    if qjs::is_exception(buf) {
        qjs::sofuu_js_free_value(ctx, vec);
        return qjs::sofuu_js_exception();
    }
    let mut buf_size: usize = 0;
    let ptr = qjs::JS_GetArrayBuffer(ctx, &mut buf_size, buf);
    if !ptr.is_null()
        && byte_offset + values.len() * std::mem::size_of::<f32>() <= buf_size
        && byte_length >= values.len() * std::mem::size_of::<f32>()
    {
        let out = std::slice::from_raw_parts_mut(ptr.add(byte_offset) as *mut f32, values.len());
        out.copy_from_slice(values);
        qjs::sofuu_js_free_value(ctx, buf);
        return vec;
    }
    qjs::sofuu_js_free_value(ctx, buf);
    for (i, &v) in values.iter().enumerate() {
        qjs::JS_SetPropertyUint32(ctx, vec, i as u32, qjs::sofuu_js_new_float64(ctx, v as f64));
    }
    vec
}

unsafe extern "C" fn js_ai_embed_local_semantic_v2(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || qjs::sofuu_js_is_string(*argv) == 0 {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"ai.embedLocalSemanticV2(text) expected a string".as_ptr(),
        );
    }
    let text = match cstr_opt(ctx, *argv) {
        Some(t) => t,
        None => return qjs::sofuu_js_exception(),
    };
    let Some(values) = crate::embedding::semantic_v2::semantic_v2(&text) else {
        return qjs::JS_ThrowInternalError(
            ctx,
            c"ai.embedLocalSemanticV2: baked SEM2 model unavailable or invalid".as_ptr(),
        );
    };
    js_new_f32_array(ctx, &values)
}

/// `sofuu.ai.embedInfoV2()` — SEM2 space identity (id `semantic-table-v2`,
/// dim 64, input `hash-v1`).  Separate from embedInfo() by design: one
/// probe per space, never one ambiguous payload.
unsafe extern "C" fn js_ai_embed_info_v2(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let json = crate::embedding::semantic_v2::model_info_json_v2();
    match CString::new(json) {
        Ok(c) => qjs::sofuu_js_new_string(ctx, c.as_ptr()),
        Err(_) => qjs::sofuu_js_new_string(ctx, c"{}".as_ptr()),
    }
}

/* ------------------------------------------------------------------ */
/* Module registration                                                   */
/* ------------------------------------------------------------------ */

thread_local! {
    // JS_CFUNC_DEF(name, length, func): magic=0, u.func = { length, generic, func }.
        static AI_FUNCS: [qjs::JSCFunctionListEntry; 18] = [
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
            name: c"embedBatch".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_embed_batch },
        },
        qjs::JSCFunctionListEntry {
            name: c"embedImage".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_ai_embed_image },
        },
        qjs::JSCFunctionListEntry {
            name: c"transcribe".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_transcribe },
        },
        qjs::JSCFunctionListEntry {
            name: c"speak".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_speak },
        },
        qjs::JSCFunctionListEntry {
            name: c"embedLocal".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_embed_local },
        },
        qjs::JSCFunctionListEntry {
            name: c"embedLocalSemantic".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_ai_embed_local_semantic },
        },
        qjs::JSCFunctionListEntry {
            name: c"embedInfo".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 0, cproto: 0, _pad: [0; 6], cfunc: js_ai_embed_info },
        },
        qjs::JSCFunctionListEntry {
            name: c"embedLocalSemanticV2".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_ai_embed_local_semantic_v2 },
        },
        qjs::JSCFunctionListEntry {
            name: c"embedInfoV2".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 0, cproto: 0, _pad: [0; 6], cfunc: js_ai_embed_info_v2 },
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
        qjs::JSCFunctionListEntry {
            name: c"modelCaps".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            /* P3 (AUDIT-2026-09-07): length is the JS `.length` metadata —
             * modelCaps takes (model, baseUrl?), resolveCaps takes 4 args
             * (model, baseUrl?, cfgWindow?, cfgMaxOutput?); the old values
             * under-declared both. */
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_ai_model_caps },
        },
        qjs::JSCFunctionListEntry {
            name: c"resolveCaps".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 4, cproto: 0, _pad: [0; 6], cfunc: js_ai_resolve_caps },
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
    use sofuu_ffi::qjs::CtxPtr;

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
                /* H-E1: provider:"ollama" is local now (embeds offline), so
                 * the no-model throw moved to an explicit remote provider. */
                ("embed no model", r#"__ai_embed("hello", { provider: "openai" })"#),
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

    /// H-E1: ai.embed defaults to bundled local (SEM2-64); explicit
    /// provider:"local" too; ai.embedBatch batches locally; unknown space
    /// throws; remote-without-model still throws the actionable error.
    #[test]
    fn embed_local_default_and_batch() {
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            assert!(!ctx.is_null());
            register_global_fn(ctx, "__ai_embed", js_ai_embed as JSCFunction);
            register_global_fn(ctx, "__ai_embedBatch", js_ai_embed_batch as JSCFunction);

            let run = |src: &str| -> JSValue {
                let c = CString::new(src).unwrap();
                qjs::JS_Eval(
                    ctx,
                    c.as_ptr(),
                    src.len(),
                    c"<embed-h1>".as_ptr(),
                    qjs::JS_EVAL_TYPE_GLOBAL,
                )
            };
            let pump = || {
                loop {
                    let mut ctx1: *mut JSContext = ptr::null_mut();
                    if qjs::JS_ExecutePendingJob(rt, &mut ctx1) <= 0 {
                        break;
                    }
                }
            };
            let read_int = |name: &str| -> i32 {
                let src = format!("globalThis.{name}");
                let c = CString::new(src).unwrap();
                let v = qjs::JS_Eval(
                    ctx,
                    c.as_ptr(),
                    c.as_bytes().len(),
                    c"<embed-h1>".as_ptr(),
                    qjs::JS_EVAL_TYPE_GLOBAL,
                );
                let mut d: i32 = -999;
                qjs::JS_ToInt32(ctx, &mut d, v);
                qjs::sofuu_js_free_value(ctx, v);
                d
            };
            let read_f64 = |name: &str| -> f64 {
                let src = format!("globalThis.{name}");
                let c = CString::new(src).unwrap();
                let v = qjs::JS_Eval(
                    ctx,
                    c.as_ptr(),
                    c.as_bytes().len(),
                    c"<embed-h1>".as_ptr(),
                    qjs::JS_EVAL_TYPE_GLOBAL,
                );
                let mut d: f64 = f64::NAN;
                qjs::JS_ToFloat64(ctx, &mut d, v);
                qjs::sofuu_js_free_value(ctx, v);
                d
            };

            // 1. Bare call → local SEM2, 64 dims.
            let r = run("globalThis.__r=null; __ai_embed('hello world').then(v=>{globalThis.__r=v.length});");
            assert!(!qjs::is_exception(r), "bare ai.embed must not throw");
            qjs::sofuu_js_free_value(ctx, r);
            pump();
            assert_eq!(read_int("__r"), 64, "default space is SEM2-64");

            // 2. Explicit provider:"local" → same.
            let r = run("globalThis.__r2=null; __ai_embed('hello world',{provider:'local'}).then(v=>{globalThis.__r2=v.length});");
            assert!(!qjs::is_exception(r));
            qjs::sofuu_js_free_value(ctx, r);
            pump();
            assert_eq!(read_int("__r2"), 64);

            // 3. Deterministic: same text → identical first element.
            let r = run("globalThis.__f1=null; globalThis.__f2=null; __ai_embed('same').then(v=>{globalThis.__f1=v[0];}); __ai_embed('same').then(v=>{globalThis.__f2=v[0];});");
            qjs::sofuu_js_free_value(ctx, r);
            pump();
            assert_eq!(read_f64("__f1").to_bits(), read_f64("__f2").to_bits(), "local embeds are deterministic");

            // 4. Space opt: hash-768 → 768 dims.
            let r = run("globalThis.__h=null; __ai_embed('x',{space:'hash-768'}).then(v=>{globalThis.__h=v.length});");
            assert!(!qjs::is_exception(r));
            qjs::sofuu_js_free_value(ctx, r);
            pump();
            assert_eq!(read_int("__h"), 768);

            // 5. Batch: 3 inputs → Array of 3 × 64-dim.
            let r = run("globalThis.__n=null; globalThis.__d=null; __ai_embedBatch(['a','b','c']).then(v=>{globalThis.__n=v.length; globalThis.__d=v[0].length;});");
            assert!(!qjs::is_exception(r));
            qjs::sofuu_js_free_value(ctx, r);
            pump();
            assert_eq!(read_int("__n"), 3);
            assert_eq!(read_int("__d"), 64);

            // 6. Unknown space throws (arg validation, sync).
            let exc = eval_exc(ctx, "__ai_embed('x',{space:'nope'})")
                .expect("unknown space must throw");
            assert!(exc.contains("unknown space"), "{exc}");

            // 7. Remote without model still throws the actionable error.
            let exc = eval_exc(ctx, "__ai_embed('x',{provider:'openai'})")
                .expect("remote embed without model must throw");
            assert!(exc.contains("No model configured"), "{exc}");

            // 8. Batch rejects remote + non-array input.
            let exc = eval_exc(ctx, "__ai_embedBatch(['a'],{provider:'openai',model:'m'})")
                .expect("remote batch must throw");
            assert!(exc.contains("local-only"), "{exc}");
            let exc = eval_exc(ctx, "__ai_embedBatch('nope')")
                .expect("non-array batch must throw");
            assert!(exc.contains("array of strings"), "{exc}");

            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }

    /// M2: bridge base64 decoding (strict alphabet, pad rules).
    #[test]
    fn audio_b64_decode_vectors() {
        assert_eq!(b64_decode("QUJD").as_deref(), Some(&[65u8, 66, 67][..]));
        assert_eq!(b64_decode("QUJDRA==").as_deref(), Some(&[65u8, 66, 67, 68][..]));
        assert_eq!(b64_decode("YWI=").as_deref(), Some(&[97u8, 98][..]));
        assert_eq!(b64_decode("+/+/").as_deref(), Some(&[251u8, 255, 191][..]));
        for bad in ["", "ABC", "AB=C", "A===", "AB C", "AB-C", "QUJD RQ==", "===="] {
            assert!(b64_decode(bad).is_none(), "{bad:?} must refuse");
        }
    }

    /// M2: transcribe/speak arg validation throws before any network work.
    /// (Transfer paths are covered by tests/voice_test.js against a mock.)
    #[test]
    fn audio_arg_validation() {
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            assert!(!ctx.is_null());
            register_global_fn(ctx, "__ai_transcribe", js_ai_transcribe as JSCFunction);
            register_global_fn(ctx, "__ai_speak", js_ai_speak as JSCFunction);

            for (name, call, want) in [
                ("transcribe no args", "__ai_transcribe()", "expected audio bytes"),
                ("transcribe non-bytes", "__ai_transcribe('nope')", "expected audio bytes"),
                (
                    "transcribe local",
                    "__ai_transcribe(new Uint8Array([1,2,3]), {provider:'local', model:'m'})",
                    "no bundled speech model",
                ),
                (
                    "transcribe anthropic",
                    "__ai_transcribe(new Uint8Array([1,2,3]), {provider:'anthropic', model:'m'})",
                    "no audio API",
                ),
                (
                    "transcribe remote no model",
                    "__ai_transcribe(new Uint8Array([1,2,3]), {provider:'openai'})",
                    "No model configured",
                ),
                ("speak no args", "__ai_speak()", "expected a string"),
                ("speak empty", "__ai_speak('   ')", "must not be empty"),
                ("speak bad format", "__ai_speak('hi', {provider:'openai', model:'m', base_url:'http://x/', format:'exe'})", "mp3|opus"),
                (
                    "speak remote no model",
                    "__ai_speak('hi', {provider:'openai'})",
                    "No model configured",
                ),
            ] {
                let exc = eval_exc(ctx, call)
                    .unwrap_or_else(|| panic!("{name}: must throw, got success"));
                assert!(exc.contains(want), "{name}: want {want:?}, got: {exc}");
            }

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
            images: Vec::new()
    },
            AiMessage {
                role: Some("tool".into()),
                content: Some("tool result".into()),
                tool_calls: None,
                tool_call_id: Some("call_1".into()),
            images: Vec::new()
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

    /// P2 multimodal: images on a user message become wire content parts —
    /// OpenAI image_url parts, Anthropic base64 image blocks. Text-only
    /// messages keep the plain-string shape byte-for-byte.
    #[test]
    fn image_parts_on_both_wires() {
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
            tools: vec![],
            messages: vec![AiMessage {
                role: Some("user".into()),
                content: Some("what is this?".into()),
                tool_calls: None,
                tool_call_id: None,
                images: vec![
                    "data:image/png;base64,AAAA".into(),
                    "data:image/jpeg;base64,BBBB".into(),
                ],
            }],
        };
        let oa = build_openai_body_v2(&mk(Provider::OpenAi));
        assert!(oa.contains("\"content\":[{\"type\":\"text\",\"text\":\"what is this?\"}"), "openai text part: {oa}");
        assert!(oa.contains("\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}"), "openai image part: {oa}");
        assert!(oa.contains("data:image/jpeg;base64,BBBB"), "both images carried: {oa}");

        let an = build_anthropic_body_v2(&mk(Provider::Anthropic));
        assert!(an.contains("\"type\":\"text\",\"text\":\"what is this?\""), "anthropic text block: {an}");
        assert!(an.contains("\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"AAAA\"}"), "anthropic image block: {an}");
        assert!(an.contains("\"media_type\":\"image/jpeg\""), "jpeg media type: {an}");

        // Text-only messages stay plain-string content.
        let mk_plain = |provider: Provider| {
            let mut c = mk(provider);
            c.messages[0].images = Vec::new();
            c
        };
        let oa2 = build_openai_body_v2(&mk_plain(Provider::OpenAi));
        assert!(oa2.contains("\"content\":\"what is this?\""), "plain shape preserved: {oa2}");
        assert!(!oa2.contains("image_url"), "no image parts on text-only: {oa2}");
    }

    /// Zero-evidence guard: a flat config max_tokens on a model the
    /// registry, the listing AND learned limits all know nothing about is
    /// OMITTED (endpoint default) — the empty-gateway class. With evidence
    /// (a listing entry), config rides clamped as usual.
    #[test]
    fn zero_evidence_model_omits_flat_config_max_tokens() {
        /* Mutates the process-global learned/discovered stores: hold the
         * shared ML test lock (the alloc-policy and discovered-caps tests
         * clear/ingest the same stores) for the whole body. */
        let _ml = crate::ml::TEST_LOCK.lock().unwrap();
        crate::ml::alloc::policy::clear_learned();
        crate::rt::model_caps_discovered::clear();
        let mk = |provider: Provider| AiRequestConfig {
            provider,
            model: Some("totally/unknown-empty-model".into()),
            api_key: None,
            base_url: None,
            profile: None,
            system_prompt: None,
            response_format: None,
            effort: None,
            max_tokens: 384_000,
            temperature: -1.0,
            top_p: -1.0,
            stream: false,
            timeout_ms: 0,
            tools: vec![],
            messages: vec![AiMessage {
                role: Some("user".into()),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
                images: Vec::new(),
            }],
        };
        let body = build_openai_body_v2(&mk(Provider::OpenAi));
        assert!(!body.contains("\"max_tokens\""), "flat cap must be omitted for a zero-evidence model: {body}");
        // With evidence (discovered listing), config rides — clamped.
        let listing = serde_json::json!({
            "data": [ { "id": "totally/unknown-empty-model",
                        "context_length": 131072,
                        "top_provider": { "max_completion_tokens": 8192 } } ]
        });
        crate::rt::model_caps_discovered::ingest_listing("https://gw-evidence.example/v1", &listing);
        let mk_ev = |provider: Provider| {
            let mut c = mk(provider);
            c.base_url = Some("https://gw-evidence.example/v1".into());
            c
        };
        let body2 = build_openai_body_v2(&mk_ev(Provider::OpenAi));
        assert!(body2.contains("\"max_tokens\":8192"), "evidence-backed clamp applies: {body2}");
        crate::rt::model_caps_discovered::clear();
    }

    /// parse_image_urls: data URLs only, capped count + size.
    #[test]
    fn image_parser_caps_and_filters() {
        assert_eq!(parse_image_urls(None).len(), 0);
        assert_eq!(parse_image_urls(Some("[]")).len(), 0);
        let ok = "[\"data:image/png;base64,AA\",\"https://x/y.png\"]";
        let v = parse_image_urls(Some(ok));
        assert_eq!(v.len(), 1, "non-data URLs dropped: {v:?}");
        let big = format!("[\"{}\"]", format!("data:image/png;base64,{}", "A".repeat(9 * 1024 * 1024)));
        assert_eq!(parse_image_urls(Some(&big)).len(), 0, "oversized dropped");
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
                images: Vec::new()
    },
                AiMessage {
                    role: Some("user".into()),
                    content: Some("hello".into()),
                    tool_calls: None,
                    tool_call_id: None,
                images: Vec::new()
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
        images: Vec::new()
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
            images: Vec::new()
    },
            AiMessage {
                role: Some("tool".into()),
                content: Some("result".into()),
                tool_calls: None,
                tool_call_id: Some("c1".into()),
            images: Vec::new()
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
                images: Vec::new()
    },
                AiMessage {
                    role: Some("user".into()),
                    content: Some("turn one question".into()),
                    tool_calls: None,
                    tool_call_id: None,
                images: Vec::new()
    },
                AiMessage {
                    role: Some("assistant".into()),
                    content: Some("turn one answer".into()),
                    tool_calls: None,
                    tool_call_id: None,
                images: Vec::new()
    },
                AiMessage {
                    role: Some("user".into()),
                    content: Some(task.into()),
                    tool_calls: None,
                    tool_call_id: None,
                images: Vec::new()
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

    /// Anthropic thinking budgets are CAPABILITY-SCALED (model_caps): a
    /// 64k sonnet-class model gets real proportional budgets, unknown
    /// models derive capacity from max_tokens, and non-thinking models
    /// never emit a thinking block.
    #[test]
    fn anthropic_thinking_budget_mapping() {
        let base = |effort: &str, model: &str, max_tokens: i32| AiRequestConfig {
            provider: Provider::Anthropic,
            model: Some(model.into()),
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
        // sonnet: thinking from ctx 200k*0.30=60k, max 90% => 54000, output stays 64k (output-only)
        let b = build_anthropic_body_v2(&base("max", "claude-sonnet-4-6", 0));
        assert!(b.contains("\"max_tokens\":64000"), "dynamic max_tokens from caps: {b}");
        assert!(b.contains("\"budget_tokens\":54000"), "ctx-derived max budget (60000*0.9): {b}");
        let b = build_anthropic_body_v2(&base("low", "claude-sonnet-4-6", 0));
        assert!(b.contains("\"budget_tokens\":3750"), "ctx-derived low budget (60000/16): {b}");

        // Explicit /maxout wins but thinking still ctx-derived, capped to max-10% to satisfy wire
        let b = build_anthropic_body_v2(&base("max", "claude-sonnet-4-6", 8_192));
        assert!(b.contains("\"max_tokens\":8192"), "explicit override: {b}");
        assert!(b.contains("\"budget_tokens\":7372"), "ctx-derived 54000 capped to max-10% (7372): {b}");

        // Unknown model: required max_tokens is the alloc gate's
        // conservative window-derived bound (4096), never an
        // endpoint-shaped assumption; thinking is capped to max-10% to
        // satisfy the wire.
        let b = build_anthropic_body_v2(&base("high", "weird-model-9000", 0));
        assert!(b.contains("\"max_tokens\":4096"), "unknown-model conservative max: {b}");
        assert!(b.contains("\"budget_tokens\":3686"), "thinking capped to max-10% (3686): {b}");

        // Layer 0: an explicit /maxout above the model's hard output cap
        // is clamped DOWN to the cap — the wire never carries an
        // oversized max_tokens.
        let b = build_anthropic_body_v2(&base("max", "claude-sonnet-4-6", 999_999));
        assert!(b.contains("\"max_tokens\":64000"), "override clamped to caps: {b}");

        // Any model on Anthropic can use thinking now (endpoint-driven 64k, not per-family)
        for effort in ["low", "medium", "high", "max"] {
            let b = build_anthropic_body_v2(&base(effort, "claude-3-5-haiku", 0));
            assert!(b.contains("\"thinking\":"), "{effort} on Anthropic now allowed for any model: {b}");
        }

        // effort off → no thinking block even on thinking models.
        let b = build_anthropic_body_v2(&base("off", "claude-sonnet-4-6", 0));
        assert!(!b.contains("\"thinking\":"), "off omits thinking: {b}");
    }

    /// OpenAI wire: reasoning_effort is capability-aware — unsupported
    /// levels fall back to the highest supported level ("max" → "high"),
    /// non-reasoning models omit it, and reasoning families carry
    /// `max_completion_tokens` instead of `max_tokens`.
    #[test]
    fn openai_effort_and_output_param_wiring() {
        let cfg = |model: &str, effort: Option<&str>| AiRequestConfig {
            provider: Provider::OpenAi,
            model: Some(model.into()),
            api_key: None,
            base_url: None,
            profile: None,
            system_prompt: None,
            response_format: None,
            effort: effort.map(Into::into),
            max_tokens: 0,
            temperature: -1.0,
            top_p: -1.0,
            stream: false,
            timeout_ms: 0,
            messages: vec![],
            tools: vec![],
        };
        let b = build_openai_body_v2(&cfg("gpt-5", Some("max")));
        assert!(b.contains("\"reasoning_effort\":\"high\""), "max falls back to high: {b}");
        assert!(b.contains("\"max_completion_tokens\":128000"), "reasoning family param name + dynamic cap: {b}");
        assert!(!b.contains("\"max_tokens\":"), "no legacy field on reasoning families: {b}");

        let b = build_openai_body_v2(&cfg("gpt-4o", Some("max")));
        assert!(!b.contains("reasoning_effort"), "non-reasoning model omits effort: {b}");
        assert!(b.contains("\"max_tokens\":16384"), "dynamic cap via legacy name: {b}");

        let b = build_openai_body_v2(&cfg("o3", Some("medium")));
        assert!(b.contains("\"reasoning_effort\":\"medium\""), "supported level passes through: {b}");
        assert!(b.contains("\"max_completion_tokens\":100000"), "{b}");

        // Unknown model + no explicit config → omit the output cap and effort entirely
        // (the endpoint applies its own default; nothing is hardcoded, no assumption).
        let b = build_openai_body_v2(&cfg("mystery-model", Some("high")));
        assert!(!b.contains("\"max_tokens\":"), "{b}");
        assert!(!b.contains("max_completion_tokens"), "{b}");
        assert!(!b.contains("reasoning_effort"), "unknown on OpenAI omits effort: {b}");
    }

    /// Layer 0 through the ENDPOINT-AWARE ladder: caps the endpoint itself
    /// published for this exact model (discovered store, keyed by
    /// base_url) clamp an oversized explicit config and supply unknown
    /// models' numbers. The registry stays the fallback when nothing
    /// was discovered. This is the exact "config does not respect the
    /// selected model" failure class — an explicit max_tokens above the
    /// endpoint's real cap must never reach the wire.
    #[test]
    fn openai_body_clamps_via_discovered_endpoint_caps() {
        /* Same shared ML test lock — see the zero-evidence test above. */
        let _ml = crate::ml::TEST_LOCK.lock().unwrap();
        crate::rt::model_caps_discovered::clear();
        let listing = serde_json::json!({
            "data": [
                { "id": "any/unknown-model:free",
                  "context_length": 65536,
                  "top_provider": { "max_completion_tokens": 8192 } },
                { "id": "gpt-4o", "context_length": 32000, "max_tokens": 4096 }
            ]
        });
        assert_eq!(
            crate::rt::model_caps_discovered::ingest_listing("https://gw.example.com/v1", &listing),
            2
        );

        let cfg = |model: &str, base_url: Option<&str>, max_tokens: i32| AiRequestConfig {
            provider: Provider::Custom,
            model: Some(model.into()),
            api_key: None,
            base_url: base_url.map(Into::into),
            profile: None,
            system_prompt: None,
            response_format: None,
            effort: None,
            max_tokens,
            temperature: -1.0,
            top_p: -1.0,
            stream: false,
            timeout_ms: 0,
            messages: vec![],
            tools: vec![],
        };

        // The liquid-400 class: unknown model + config maxout 384000 vs an
        // endpoint that publishes 8192. The wire must carry 8192.
        let b = build_openai_body_v2(&cfg(
            "any/unknown-model:free",
            Some("https://gw.example.com/v1/chat/completions"),
            384_000,
        ));
        assert!(b.contains("\"max_tokens\":8192"), "oversized config clamped to the endpoint's real cap: {b}");

        // Same model, NO explicit config → the discovered cap supplies it.
        let b = build_openai_body_v2(&cfg(
            "any/unknown-model:free",
            Some("https://gw.example.com/v1"),
            0,
        ));
        assert!(b.contains("\"max_tokens\":8192"), "discovered cap for an otherwise-unknown model: {b}");

        // A KNOWN registry family the endpoint hosts smaller: the
        // endpoint's numbers win over the name-keyed table.
        let b = build_openai_body_v2(&cfg("gpt-4o", Some("https://gw.example.com/v1"), 0));
        assert!(b.contains("\"max_tokens\":4096"), "gateway's real cap overrides the registry guess: {b}");

        // Without discovery at that root the registry number stands.
        let b = build_openai_body_v2(&cfg("gpt-4o", Some("https://other.example.com/v1"), 0));
        assert!(b.contains("\"max_tokens\":16384"), "registry cap when nothing discovered: {b}");

        // No base_url at all → historical registry behavior.
        let b = build_openai_body_v2(&cfg("gpt-4o", None, 0));
        assert!(b.contains("\"max_tokens\":16384"), "{b}");

        crate::rt::model_caps_discovered::clear();
    }

    /// Wires and model families are ORTHOGONAL: the openai endpoint
    /// carries most providers' non-OpenAI models, and an anthropic-format
    /// endpoint may serve non-Claude models. The builders must produce a
    /// sane body for ANY (wire × family) combination — limits from the
    /// registry, syntax from the wire.
    #[test]
    fn wires_and_model_families_are_orthogonal() {
        let cfg = |model: &str, profile: Provider, effort: Option<&str>| AiRequestConfig {
            provider: profile,
            model: Some(model.into()),
            api_key: None,
            base_url: None,
            profile: None,
            system_prompt: None,
            response_format: None,
            effort: effort.map(Into::into),
            max_tokens: 0,
            temperature: -1.0,
            top_p: -1.0,
            stream: false,
            timeout_ms: 0,
            messages: vec![],
            tools: vec![],
        };

        // A NON-OpenAI model over the OpenAI endpoint (the common case):
        // dynamic cap via the legacy param name; effort folded to high.
        let b = build_openai_body_v2(&cfg("deepseek-chat", Provider::OpenAi, Some("max")));
        assert!(b.contains("\"max_tokens\":8192"), "{b}");
        assert!(!b.contains("max_completion_tokens"), "{b}");

        // A Claude model over the OpenAI endpoint (gateway-served): no
        // anthropic-style thinking block leaks through; effort passes as
        // the ladder-style parameter with max→high folding.
        let b = build_openai_body_v2(&cfg("claude-sonnet-4-6", Provider::Custom, Some("max")));
        assert!(b.contains("\"reasoning_effort\":\"high\""), "{b}");
        assert!(!b.contains("\"thinking\":"), "{b}");
        assert!(b.contains("\"max_tokens\":64000"), "claude caps still apply: {b}");

        // A non-Claude model over the Anthropic-format endpoint: syntax is
        // the wire's (budget block + required max_tokens), conservative
        // alloc bound for unknown models
        let b = build_anthropic_body_v2(&cfg("my-gateway-model", Provider::Custom, Some("high")));
        assert!(b.contains("\"max_tokens\":4096"), "anthropic conservative max for unknown: {b}");
        assert!(b.contains("\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":3686}"), "{b}");

        // A reasoning-family name on the Anthropic-format endpoint: budget
        // syntax wins (it's the wire's), capacity derived from max_tokens.
        let b = build_anthropic_body_v2(&cfg("gpt-5", Provider::Custom, Some("medium")));
        assert!(b.contains("\"thinking\":{\"type\":\"enabled\""), "{b}");
        assert!(!b.contains("reasoning_effort"), "{b}");
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

    /// net-10 (AUDIT-2026-09-07): a quote/backslash in a tool NAME (MCP tool
    /// names are external input) must not break the request JSON — both wire
    /// formats escape the name exactly like the description.
    #[test]
    fn tool_names_escaped_in_wire_json() {
        let evil = "evil\"name\\x";
        let mut cfg = AiRequestConfig::default();
        cfg.tools = vec![AiToolDef {
            name: Some(evil.into()),
            description: Some("d\"esc".into()),
            parameters_json: Some("{\"type\":\"object\"}".into()),
        }];

        for (label, build) in [
            ("openai", build_openai_body_v2 as fn(&AiRequestConfig) -> String),
            ("anthropic", build_anthropic_body_v2 as fn(&AiRequestConfig) -> String),
        ] {
            let b = build(&cfg);
            /* The full request body must still be well-formed JSON... */
            let v: serde_json::Value = serde_json::from_str(&b)
                .unwrap_or_else(|e| panic!("{label}: request body is not valid JSON ({e}): {b}"));
            /* ...and the name must round-trip through a REAL parser
             * (OpenAI nests it at tools[i].function.name, Anthropic at
             * tools[i].name directly). */
            let tools = v.get("tools").and_then(|t| t.as_array()).expect("tools array");
            assert_eq!(tools.len(), 1, "{label}");
            let node = if label == "openai" {
                tools[0].get("function").unwrap_or_else(|| panic!("{label}: tools[0].function missing: {b}"))
            } else {
                &tools[0]
            };
            let parsed_name = node
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or_else(|| panic!("{label}: tool name missing: {b}"));
            assert_eq!(parsed_name, evil, "{label}: tool name must round-trip: {b}");
            /* ...and the exact escaped rendering must be on the wire. */
            let escaped = format!("\"name\":\"{}\"", json_escape(Some(evil)));
            assert!(
                b.contains(&escaped),
                "{label}: escaped name form missing from wire: {b}"
            );
        }
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

    // ── P1-1: js_ai_stream must release every local ref it takes off the
    // STREAM_FACTORY result on all paths. Before the fix, the setToolCalls
    // local leaked per stream — and because QuickJS treats refcount>0
    // objects as GC roots, that one ref pinned the whole factory closure
    // (queue, _resolve, iterator, all method closures) for the life of the
    // runtime. The request box's dups are freed by stream_req_destroy on
    // the error path, so the ONLY per-stream residue is the leaked local.

    #[repr(C)]
    struct JSMemoryUsage {
        malloc_size: i64, malloc_limit: i64, memory_used_size: i64,
        malloc_count: i64,
        memory_used_count: i64,
        atom_count: i64, atom_size: i64,
        str_count: i64, str_size: i64,
        obj_count: i64, obj_size: i64,
        prop_count: i64, prop_size: i64,
        shape_count: i64, shape_size: i64,
        js_func_count: i64, js_func_size: i64, js_func_code_size: i64,
        js_func_pc2line_count: i64, js_func_pc2line_size: i64,
        c_func_count: i64, array_count: i64,
        fast_array_count: i64, fast_array_elements: i64,
        binary_object_count: i64, binary_object_size: i64,
    }

    extern "C" {
        fn JS_ComputeMemoryUsage(rt: *mut qjs::JSRuntime, s: *mut JSMemoryUsage);
    }

    unsafe fn gc_obj_count(rt: *mut qjs::JSRuntime) -> i64 {
        qjs::JS_RunGC(rt);
        let mut m: JSMemoryUsage = std::mem::zeroed();
        JS_ComputeMemoryUsage(rt, &mut m);
        m.obj_count
    }

    unsafe fn read_global_i64(ctx: *mut JSContext, name: &CStr) -> i64 {
        let global = qjs::sofuu_js_get_global_object(ctx);
        let v = qjs::sofuu_js_get_property_str(ctx, global, name.as_ptr());
        qjs::sofuu_js_free_value(ctx, global);
        let mut out: i64 = -1;
        qjs::JS_ToInt64(ctx, &mut out, v);
        qjs::sofuu_js_free_value(ctx, v);
        out
    }

    /// Read a string global ("" when missing or not a string).
    unsafe fn read_global_str(ctx: *mut JSContext, name: &CStr) -> String {
        let global = qjs::sofuu_js_get_global_object(ctx);
        let v = qjs::sofuu_js_get_property_str(ctx, global, name.as_ptr());
        qjs::sofuu_js_free_value(ctx, global);
        let mut out = String::new();
        let p = qjs::sofuu_js_to_cstring(ctx, v);
        if !p.is_null() {
            out = CStr::from_ptr(p).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, p);
        }
        qjs::sofuu_js_free_value(ctx, v);
        out
    }

    /// N streams against 127.0.0.1:1 (connect refused instantly) each settle
    /// through error_fn → for-await throws → catch. Asserts every one settled
    /// and returns after the loop has fully drained.
    const STREAM_DRIVER: &str = r#"
        globalThis.settled = 0;
        async function drive(n) {
          for (let i = 0; i < n; i++) {
            try {
              for await (const c of __ai_stream({
                messages: [{ role: "user", content: "hi" }],
                provider: "probe",
                base_url: "http://127.0.0.1:1/v1/chat/completions",
                model: "m",
              })) { /* connect fails before any chunk */ }
            } catch (e) { /* expected: connect error surfaces here */ }
            globalThis.settled++;
          }
        }
        drive(N_STREAMS);
    "#;

    #[test]
    fn stream_error_path_does_not_retain_factory_objects() {
        unsafe {
            let _loop_guard = crate::rt::test_loop_lock();
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);
            crate::rt::event_loop::sofuu_loop_init();
            register_global_fn(ctx, "__ai_stream", js_ai_stream as JSCFunction);

            // Warmup: settle 2 streams first so one-shot init (curl global,
            // multi handle, factory eval, function objects) and the first
            // poll/socket machinery are out of the measured window.
            let src = CString::new(STREAM_DRIVER.replacen("N_STREAMS", "2", 1)).unwrap();
            let r = qjs::JS_Eval(ctx, src.as_ptr(), src.as_bytes().len(),
                                 c"<p1-1-warmup>".as_ptr(), qjs::JS_EVAL_TYPE_GLOBAL);
            assert!(!qjs::is_exception(r), "warmup eval threw");
            qjs::sofuu_js_free_value(ctx, r);
            crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60)));
            assert_eq!(read_global_i64(ctx, c"settled"), 2, "warmup streams must all settle");

            let before = gc_obj_count(rt);

            let src = CString::new(STREAM_DRIVER.replacen("N_STREAMS", "8", 1)).unwrap();
            let r = qjs::JS_Eval(ctx, src.as_ptr(), src.as_bytes().len(),
                                 c"<p1-1-measure>".as_ptr(), qjs::JS_EVAL_TYPE_GLOBAL);
            assert!(!qjs::is_exception(r), "measure eval threw");
            qjs::sofuu_js_free_value(ctx, r);
            crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60)));
            assert_eq!(read_global_i64(ctx, c"settled"), 8, "measured streams must all settle");

            let after = gc_obj_count(rt);
            // A settled stream retains nothing: the box dups die in
            // stream_req_destroy and (with the fix) every local ref is
            // freed. Pre-fix each stream leaked the setToolCalls ref, pinning
            // the factory closure ≈ 10+ objects — 8 streams grew obj_count
            // by ~80+. Allow small fixed noise, not per-stream growth.
            assert!(
                after - before < 10,
                "stream objects retained across settled streams: {} → {} (delta {})",
                before, after, after - before
            );

            crate::rt::event_loop::sofuu_loop_close();
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }

    // ── net-11: error_fn must not fire after [DONE] ─────────────────────
    // The turn is already committed; pre-fix the CURLMSG_DONE error and the
    // HTTP ≥400 branches called error_fn even when done_fn had already
    // delivered the answer. The factory's post-done error is invisible to
    // its consumer (next() checks _done before _error), so the only
    // observable is the `new Error(msg)` inside factory error() — counted
    // via a globalThis.Error stub.
    //
    // Scenarios, each driven through js_ai_stream against a local server:
    //   trunc  = 200 + overlong Content-Length, SSE body, then FIN
    //            → curl CURLE_PARTIAL_FILE arrives after the body bytes.
    //   0. trunc WITH a trailing `data: [DONE]` — done fires during the
    //      write callback; the later PARTIAL_FILE error must be suppressed.
    //   1. trunc WITHOUT [DONE] — done never fired, the error must STILL
    //      surface through the iterator throw (the gate must not over-block).
    //   2. HTTP 400 with an exact-length body containing `data: [DONE]` —
    //      the write callback parses [DONE] regardless of status, so done
    //      fires first; the ≥400 error_fn must be suppressed.

    const NET11_DRIVER: &str = r#"
        globalThis.__errs = 0;
        globalThis.__errCaught = 0;
        globalThis.__errMsg = '';
        globalThis.__chunkText = '';
        globalThis.__settled = 0;
        async function drive() {
          try {
            for await (const c of __ai_stream({
              messages: [{ role: "user", content: "hi" }],
              provider: "probe",
              base_url: "__BASE__",
              model: "m",
            })) { globalThis.__chunkText += c.text; }
          } catch (e) {
            globalThis.__errCaught = 1;
            globalThis.__errMsg = String((e && e.message) || e);
          }
          globalThis.__settled = 1;
        }
        drive();
    "#;

    /// Serve one connection: `head` verbatim, then `body`, then FIN. The
    /// head carries whatever Content-Length the caller declared — an
    /// overlong one makes curl finish with CURLE_PARTIAL_FILE after the
    /// body bytes were delivered. Replies are constant bytes (never echo
    /// the request).
    fn net11_spawn_server(head: String, body: Vec<u8>) -> (u16, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let t = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(2)));
            let mut buf = [0u8; 8192];
            let mut req = Vec::new();
            // Drain the request head (and small JSON body) until the
            // double CRLF terminator — then answer with constants.
            loop {
                match std::io::Read::read(&mut s, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        req.extend_from_slice(&buf[..n]);
                        if req.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            let _ = std::io::Write::write_all(&mut s, head.as_bytes());
            let _ = std::io::Write::write_all(&mut s, &body);
            let _ = s.shutdown(std::net::Shutdown::Write);
            // Give curl a moment to read before the full teardown.
            let _ = std::io::Read::read(&mut s, &mut buf);
            drop(s);
        });
        (port, t)
    }

    fn net11_head(status: &str, declared_len: usize) -> String {
        format!(
            "{status}\r\nContent-Type: text/event-stream\r\nContent-Length: {declared_len}\r\nConnection: close\r\n\r\n"
        )
    }

    /// Runtime with `__ai_stream` registered and the counting Error stub
    /// installed: factory error() does `new Error(msg)` and unqualified
    /// `Error` resolves from the global at call time, so every error_fn
    /// dispatch lands in the stub. Internal QuickJS errors (JS_Throw*)
    /// never construct through the user-visible global, so the count is
    /// exactly the factory error() calls.
    unsafe fn net11_new_ctx() -> (*mut qjs::JSRuntime, *mut JSContext, CtxPtr<'static>) {
        let rt = qjs::JS_NewRuntime();
        let ctx = qjs::JS_NewContext(rt);
        let guard = CtxPtr::new(ctx);
        crate::rt::event_loop::sofuu_loop_init();
        register_global_fn(ctx, "__ai_stream", js_ai_stream as JSCFunction);
        let stub = CString::new(
            r#"
            globalThis.__RealError = globalThis.Error;
            globalThis.Error = function (msg) {
              globalThis.__errs++;
              return new globalThis.__RealError(msg);
            };
            globalThis.Error.prototype = globalThis.__RealError.prototype;
        "#,
        )
        .unwrap();
        let r = qjs::JS_Eval(ctx, stub.as_ptr(), stub.as_bytes().len(),
                             c"<net11-stub>".as_ptr(), qjs::JS_EVAL_TYPE_GLOBAL);
        assert!(!qjs::is_exception(r), "stub eval threw");
        qjs::sofuu_js_free_value(ctx, r);
        (rt, ctx, guard)
    }

    /// Eval the driver against `port` and pump the loop until it settles.
    unsafe fn net11_drive(ctx: *mut JSContext, port: u16) {
        let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
        let src = CString::new(NET11_DRIVER.replacen("__BASE__", &url, 1)).unwrap();
        let r = qjs::JS_Eval(ctx, src.as_ptr(), src.as_bytes().len(),
                             c"<net11-drive>".as_ptr(), qjs::JS_EVAL_TYPE_GLOBAL);
        assert!(!qjs::is_exception(r), "driver eval threw");
        qjs::sofuu_js_free_value(ctx, r);
        crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60)));
    }

    unsafe fn net11_close(rt: *mut qjs::JSRuntime, ctx: *mut JSContext) {
        crate::rt::event_loop::sofuu_loop_close();
        qjs::JS_FreeContext(ctx);
        qjs::JS_FreeRuntime(rt);
    }

    #[test]
    fn stream_post_done_error_suppressed_after_truncation() {
        let delta = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        let mut body = delta.to_vec();
        body.extend_from_slice(b"data: [DONE]\n\n");
        // Declare far more than we send; the FIN then lands as
        // CURLE_PARTIAL_FILE — after done was already signalled.
        let (port, srv) =
            net11_spawn_server(net11_head("HTTP/1.1 200 OK", body.len() + 400), body);
        unsafe {
            // Poison-tolerant: an earlier net-11 test can panic while
            // holding the lock (deliberate negative-control failures) —
            // the loop state itself is per-test.
            let _loop_guard = crate::rt::TEST_LOOP_LOCK
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let (rt, ctx, _g) = net11_new_ctx();
            net11_drive(ctx, port);
            let _ = srv.join();
            assert_eq!(read_global_i64(ctx, c"__settled"), 1, "stream must settle");
            assert_eq!(
                read_global_i64(ctx, c"__errs"), 0,
                "error_fn fired after [DONE] — the turn was already delivered"
            );
            assert_eq!(
                read_global_i64(ctx, c"__errCaught"), 0,
                "a clean [DONE] stream must not throw in the consumer"
            );
            assert_eq!(
                read_global_str(ctx, c"__chunkText"), "hi",
                "the delta before [DONE] must be delivered"
            );
            net11_close(rt, ctx);
        }
    }

    #[test]
    fn stream_truncation_without_done_still_errors() {
        // Same truncation, but the body ends without [DONE]: done never
        // fired, so the error must STILL surface (the gate must not
        // over-block a genuine failure).
        let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n".to_vec();
        let (port, srv) =
            net11_spawn_server(net11_head("HTTP/1.1 200 OK", body.len() + 400), body);
        unsafe {
            let _loop_guard = crate::rt::TEST_LOOP_LOCK
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let (rt, ctx, _g) = net11_new_ctx();
            net11_drive(ctx, port);
            let _ = srv.join();
            assert_eq!(read_global_i64(ctx, c"__settled"), 1, "stream must settle");
            assert_eq!(
                read_global_i64(ctx, c"__errCaught"), 1,
                "a truncated stream without [DONE] must still surface an error"
            );
            assert!(
                read_global_i64(ctx, c"__errs") >= 1,
                "the gate must not suppress a genuine pre-done failure"
            );
            assert_eq!(
                read_global_str(ctx, c"__chunkText"), "hi",
                "the delta delivered before the truncation must not be lost"
            );
            net11_close(rt, ctx);
        }
    }

    #[test]
    fn stream_post_done_error_suppressed_http_400() {
        // HTTP 400 with an exact-length body containing `data: [DONE]` —
        // the write callback parses [DONE] regardless of status, so done
        // fires first and the ≥400 error_fn must be suppressed.
        let body = b"data: [DONE]\n\n".to_vec();
        let (port, srv) =
            net11_spawn_server(net11_head("HTTP/1.1 400 Bad Request", body.len()), body);
        unsafe {
            let _loop_guard = crate::rt::TEST_LOOP_LOCK
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let (rt, ctx, _g) = net11_new_ctx();
            net11_drive(ctx, port);
            let _ = srv.join();
            assert_eq!(read_global_i64(ctx, c"__settled"), 1, "stream must settle");
            assert_eq!(
                read_global_i64(ctx, c"__errs"), 0,
                "the ≥400 error_fn fired after [DONE] — done already committed the turn"
            );
            assert_eq!(
                read_global_i64(ctx, c"__errCaught"), 0,
                "a stream that ended via [DONE] must not throw in the consumer"
            );
            net11_close(rt, ctx);
        }
    }
}
