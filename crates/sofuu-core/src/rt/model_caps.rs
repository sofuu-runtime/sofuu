// model_caps.rs — per-model capability registry (single source of truth).
//
// WHY THIS EXISTS
// ───────────────
// Sofuu talks to ENDPOINTS, not providers: the OpenAI wire
// (/chat/completions), the Anthropic wire (/v1/messages) and the local
// server wire (/api/chat). A "provider" is just a saved endpoint + key +
// profile. What a request may ask for — context window, max output
// tokens, thinking/reasoning support and its ceiling — belongs to the
// MODEL, never to the endpoint. Before this module those numbers were
// hardcoded per-wire (the notorious `max_tokens = 4096` fallback, the
// flat 32768 context default, the 16384 thinking clamp), which starved
// big models and confused small ones.
//
// SHAPE
// ─────
// A static prefix table (first match wins) + a JSON accessor exposed to
// JS as `sofuu.ai.modelCaps(model)` (returns a JSON string; parse on the
// JS side). Everything downstream — request builders in rt/ai.rs, the
// agent loop (src/js/agent.js) and the chat driver's budgets — resolves
// limits through this registry instead of constants.
//
// ACCURACY CONTRACT
// ─────────────────
// Entries are best-effort published limits for each model family and are
// deliberately generous-but-real. Users can always override per session:
//   /ctx <n>      raises/lowers the assumed input window
//   /maxout <n>   raises/lowers the assumed output cap
// Unknown models report zeros; consumers then fall back gracefully
// (OpenAI wire omits max_tokens entirely → provider default; Anthropic
// wire uses the configured value or the documented floor).

/// How a model "thinks".
#[derive(Clone, Copy, PartialEq)]
pub enum Thinking {
    /// The model has no reasoning mode (and no reasoning parameter).
    /// Sending one is at best ignored, at worst a 400.
    None,
    /// OpenAI-style discrete levels (`reasoning_effort`). The slice is the
    /// FULL supported list, lowest→highest; selection clamps to it.
    Effort(&'static [&'static str]),
    /// Anthropic-style extended thinking (`thinking.budget_tokens`).
    /// The value is the largest budget_tokens this family accepts.
    Budget(i32),
    /// Unknown family — callers decide their own safe behavior.
    Unknown,
}

#[derive(Clone, Copy)]
pub struct ModelCaps {
    /// Input context window (tokens). 0 when unknown.
    pub ctx_window: i32,
    /// Max output tokens per response. 0 when unknown.
    pub max_output: i32,
    pub thinking: Thinking,
}

const NO_CAPS: ModelCaps =
    ModelCaps { ctx_window: 0, max_output: 0, thinking: Thinking::Unknown };

const GPT5_EFFORTS: &[&str] = &["minimal", "low", "medium", "high"];
const O_EFFORTS: &[&str] = &["low", "medium", "high"];

/// (prefix, ctx_window, max_output, thinking) — first match wins; order
/// longer prefixes before shorter ones.
static TABLE: &[(&str, i32, i32, Thinking)] = &[
    /* ── Model families (WIRE-AGNOSTIC) ──────────────────────────────
     * Entries below are keyed by MODEL NAME because capabilities belong
     * to the model, never to the endpoint that serves it. Either wire
     * can carry any family: most providers use the OpenAI endpoint for
     * their own non-OpenAI models (deepseek/qwen/llama/grok…), and an
     * Anthropic-format endpoint may proxy non-Claude models too. The
     * REQUEST BUILDERS decide syntax per wire; these numbers only
     * supply limits. "Thinking" kind reflects how the FAMILY reasons:
     * `Effort` = a discrete reasoning_effort ladder it understands,
     * `Budget` = an extended-thinking budget ceiling — builders on the
     * other wire translate defensively (e.g. claude over the OpenAI
     * endpoint passes effort levels through; gateways map them). */
    // GPT-5 generation: huge windows, discrete effort ladder (no "max").
    ("gpt-5", 400_000, 128_000, Thinking::Effort(GPT5_EFFORTS)),
    ("gpt-4.1", 1_000_000, 32_768, Thinking::None),
    ("gpt-4o", 128_000, 16_384, Thinking::None),
    ("chatgpt-4o", 128_000, 16_384, Thinking::None),
    ("gpt-4-turbo", 128_000, 4_096, Thinking::None),
    ("gpt-4", 8_192, 8_192, Thinking::None),
    // o-series reasoners: effort ladder without "minimal"/"max".
    ("o3", 200_000, 100_000, Thinking::Effort(O_EFFORTS)),
    ("o4-mini", 200_000, 100_000, Thinking::Effort(O_EFFORTS)),
    ("o1", 200_000, 100_000, Thinking::Effort(O_EFFORTS)),
    /* ── Other openai-compatible families ──────────────────────────── */
    // DeepSeek thinks via <think> tags in the text (stripped client-side),
    // not via an API parameter — modeled as Thinking::None here.
    // V4 generation (V4-Flash / V4.1-Flash): 1M context, 128k output tier
    // (providers tier higher — up to ~384k; the per-endpoint ladder
    // (discovered/learned) refines this guess). MUST precede the V3-era
    // row: first match wins, and the bare prefix would swallow V4 names
    // into 64k (2026-10-09: footer read 65.5k on a 1M model).
    ("deepseek-v4", 1_048_576, 131_072, Thinking::None),
    ("deepseek", 65_536, 8_192, Thinking::None),
    // Kimi K2.5/K2.6: 256K per Moonshot (HF card + vendor docs) — NOT 1M
    // (that figure is Kimi K3). Max outputs are provider-tiered; these
    // conservative tier values defer to discovered/learned evidence.
    ("kimi-k2.6", 262_144, 32_768, Thinking::None),
    ("kimi-k2.5", 262_144, 32_768, Thinking::None),
    ("kimi-k3", 1_048_576, 131_072, Thinking::None),
    ("qwen3", 131_072, 32_768, Thinking::None),
    ("qwen", 32_768, 8_192, Thinking::None),
    ("llama4", 1_000_000, 8_192, Thinking::None),
    ("llama3.1", 131_072, 4_096, Thinking::None),
    ("llama3", 8_192, 4_096, Thinking::None),
    ("llama", 4_096, 2_048, Thinking::None),
    ("mistral", 32_768, 8_192, Thinking::None),
    ("mixtral", 32_768, 8_192, Thinking::None),
    // Grok-4 reasons internally; there is no client-side effort knob.
    ("grok-4", 256_000, 32_768, Thinking::None),
    ("grok", 131_072, 8_192, Thinking::None),
    /* ── Claude family (named for the /v1/messages origin; any endpoint
     * may serve them — caps stay name-keyed) ───────────────────────── */
    // 4th-gen+ extended thinking: budget_tokens only needs to stay below
    // max_tokens, so the practical ceiling IS the output cap. Sofuu sends
    // max_output as max_tokens, which unlocks real 64k thinking budgets.
    ("claude-opus-4", 200_000, 32_000, Thinking::Budget(32_000)),
    ("claude-sonnet-4", 200_000, 64_000, Thinking::Budget(64_000)),
    ("claude-haiku-4", 200_000, 64_000, Thinking::Budget(64_000)),
    // 3-7 introduced extended thinking.
    ("claude-3-7", 200_000, 64_000, Thinking::Budget(64_000)),
    ("claude-3-5", 200_000, 8_192, Thinking::None),
    ("claude-3", 200_000, 4_096, Thinking::None),
    ("claude-2", 100_000, 4_096, Thinking::None),
    ("claude", 200_000, 8_192, Thinking::None),
];

/// Resolve capabilities by model name (case-insensitive prefix match;
/// `org/model` forms match too).
pub fn lookup(model: Option<&str>) -> ModelCaps {
    let m = model.unwrap_or("").trim().to_lowercase();
    if m.is_empty() {
        return NO_CAPS;
    }
    for (prefix, ctx, out, thinking) in TABLE {
        // P3 (AUDIT-2026-09-07): allocation-free `contains("/{prefix}")` —
        // the old format! built a fresh String per table row per lookup,
        // and lookup runs on every request.
        let hit = m.starts_with(prefix)
            || m.match_indices('/').any(|(i, _)| m[i + 1..].starts_with(prefix));
        if hit {
            return ModelCaps { ctx_window: *ctx, max_output: *out, thinking: *thinking };
        }
    }
    NO_CAPS
}

/// Resolve capabilities for a request that carries an endpoint: the
/// numbers the ENDPOINT itself published for this exact model (the
/// discovered store, harvested from its model listing) override the
/// static family table when present — the gateway serving the request is
/// better evidence than a name-keyed guess (a proxy may host gpt-4o at
/// 32k, or an unknown model the table has never heard of). The THINKING
/// kind still comes from the registry: listings carry no reasoning
/// syntax, and the wire builders need it to decide parameter shape.
/// `base_url` may be None or a full endpoint URL — normalization is
/// provider-agnostic and strips completion suffixes.
pub fn lookup_for(model: Option<&str>, base_url: Option<&str>) -> ModelCaps {
    let reg = lookup(model);
    let Some(url) = base_url else { return reg };
    super::model_caps_discovered::ensure_loaded();
    let Some(d) = super::model_caps_discovered::lookup(url, model.unwrap_or("")) else {
        return reg;
    };
    // Each side independently: a discovered 0 leaves the registry number.
    ModelCaps {
        ctx_window: if d.ctx_window > 0 { d.ctx_window } else { reg.ctx_window },
        max_output: if d.max_output > 0 { d.max_output } else { reg.max_output },
        thinking: reg.thinking,
    }
}

impl ModelCaps {
    pub fn known(&self) -> bool {
        self.ctx_window > 0 || self.max_output > 0
    }

    /// Highest level of this family's reasoning ladder. None when the
    /// model has no reasoning mode. (The `reasoning_effort` PARAMETER
    /// itself only exists on the OpenAI endpoint; builders on other
    /// wires translate.)
    pub fn highest_effort(&self) -> Option<&'static str> {
        match self.thinking {
            Thinking::Effort(levels) => levels.last().copied(),
            _ => None,
        }
    }

    /// Serialize to the JSON shape handed across the JS seam
    /// (`sofuu.ai.modelCaps(model)` returns this string).
    pub fn to_json(&self, model: &str) -> String {
        let (kind, efforts, budget): (&str, String, i32) = match self.thinking {
            Thinking::None => ("none", "[]".to_string(), 0),
            Thinking::Effort(levels) => (
                "effort",
                format!(
                    "[{}]",
                    levels
                        .iter()
                        .map(|l| format!("\"{l}\""))
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                0,
            ),
            Thinking::Budget(b) => ("budget", "[]".to_string(), b),
            Thinking::Unknown => ("unknown", "[]".to_string(), 0),
        };
        format!(
            "{{\"model\":\"{}\",\"known\":{},\"ctxWindow\":{},\"maxOutput\":{},\"thinking\":\"{}\",\"efforts\":{},\"maxThinkingBudget\":{}}}",
            crate::rt::ai::json_escape(Some(model)),
            self.known(),
            self.ctx_window,
            self.max_output,
            kind,
            efforts,
            budget
        )
    }
}

/* ── Shared resolution helpers used by the request builders ────────── */

/// Endpoint's max thinking budget for the Anthropic wire (provider limit,
/// not per-model context window). Applies to ANY model on that endpoint —
/// not only Claude — up to the model's own max_output.
pub const ANTHROPIC_MAX_THINKING: i32 = 64_000;

/// Effort → fraction of the endpoint's thinking capacity. Strictly follows
/// what the user picked: low < medium < high < max, never more than the
/// selected level. `off` is handled before calling (returns None).
pub fn effort_fraction(effort: &str) -> f64 {
    match effort {
        "low" => 1.0 / 16.0,
        "medium" => 0.25,
        "high" => 0.5,
        "max" => 0.9,
        _ => 0.25, /* unknown effort → sensible middle, still fraction */
    }
}

/// Resolve Anthropic thinking budget: thinking lives *inside* the total
/// context window (not inside max_output). max_output stays output-only.
/// Budget is a fraction of the model's ctx_window (dynamic, no flat 1024),
/// capped at the endpoint's 64k ceiling and at per-model Budget ceiling.
/// Strict: exactly the fraction for the selected effort, never more; `off` → None.
pub fn resolve_thinking_budget(caps: &ModelCaps, effort: &str) -> Option<i32> {
    if effort.is_empty() || effort == "off" {
        return None;
    }
    // Dynamic capacity from context window, not max_output. Unknown models get endpoint max.
    let ctx_cap = if caps.ctx_window > 0 {
        // ~30% of context window is available for thinking, capped at endpoint max
        ((caps.ctx_window as f64 * 0.30).ceil() as i32).min(ANTHROPIC_MAX_THINKING)
    } else {
        ANTHROPIC_MAX_THINKING
    };
    let mut capacity = ctx_cap;
    if let Thinking::Budget(ceiling) = caps.thinking {
        capacity = capacity.min(ceiling);
    }
    // Strict fraction of the context-derived capacity, does not define output limit
    let budget = (capacity as f64 * effort_fraction(effort)).ceil() as i32;
    Some(budget.max(1))
}

/// Pick the effort string actually sent on the OpenAI wire: unsupported
/// levels fall back to the HIGHEST level the model supports (user-facing
/// rule: "max" on a low/medium/high ladder → "high"); models without any
/// effort support get None (omit the field entirely).
pub fn resolve_openai_effort(caps: &ModelCaps, effort: Option<&str>) -> Option<&'static str> {
    let e = effort?;
    if e.is_empty() || e == "off" {
        return None;
    }
    match caps.thinking {
        Thinking::None => None,
        Thinking::Effort(levels) => {
            if levels.contains(&e) {
                Some(levels.iter().find(|l| **l == e).unwrap())
            } else {
                levels.last().copied()
            }
        }
        Thinking::Budget(_) => {
            /* Claude via OpenAI gateway — translate budget levels to ladder. */
            match e {
                "low" => Some("low"),
                "medium" => Some("medium"),
                "high" => Some("high"),
                "minimal" => Some("minimal"),
                _ => Some("high"),
            }
        }
        Thinking::Unknown => None, // unknown model on OpenAI: don't assume thinking, provider default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_known_families() {
        let c = lookup(Some("claude-sonnet-4-6"));
        assert_eq!(c.ctx_window, 200_000);
        assert_eq!(c.max_output, 64_000);
        assert!(matches!(c.thinking, Thinking::Budget(64_000)));

        let c = lookup(Some("claude-opus-4-1"));
        assert_eq!(c.max_output, 32_000);

        let c = lookup(Some("gpt-5-mini"));
        assert_eq!(c.ctx_window, 400_000);
        assert_eq!(c.max_output, 128_000);
        assert_eq!(c.highest_effort(), Some("high"));

        let c = lookup(Some("o3"));
        assert_eq!(c.highest_effort(), Some("high"));
        assert!(matches!(c.thinking, Thinking::Effort(O_EFFORTS)));

        let c = lookup(Some("gpt-4o"));
        assert!(matches!(c.thinking, Thinking::None));

        // org-prefixed forms match too.
        assert_eq!(lookup(Some("openai/gpt-4.1")).ctx_window, 1_000_000);
    }

    #[test]
    fn lookup_unknown_is_zeroed() {
        let c = lookup(Some("totally-made-up-9000"));
        assert!(!c.known());
        assert_eq!(c.ctx_window, 0);
        assert_eq!(c.max_output, 0);
        assert!(matches!(c.thinking, Thinking::Unknown));
        assert_eq!(lookup(None).ctx_window, 0);
        assert_eq!(lookup(Some("")).max_output, 0);
    }

    #[test]
    fn longest_prefix_wins() {
        // gpt-4-turbo must beat gpt-4; llama3.1 must beat llama3/llama.
        assert_eq!(lookup(Some("gpt-4-turbo")).max_output, 4_096);
        assert_eq!(lookup(Some("gpt-4.1-mini")).ctx_window, 1_000_000);
        assert_eq!(lookup(Some("llama3.1-70b")).ctx_window, 131_072);
        assert_eq!(lookup(Some("llama3-8b")).ctx_window, 8_192);
    }

    #[test]
    fn versioned_families_resolve_to_model_truth() {
        // 2026-10-09: deepseek-v4.1-flash:free resolved to the V3-era 64k
        // row — the chat footer read 65.5k on a 1M-context model, so the
        // 85%-of-window budget bled turns at ~55k and auto-compact could
        // never reach a sane threshold.
        let c = lookup(Some("deepseek-v4.1-flash:free"));
        assert_eq!(c.ctx_window, 1_048_576);
        assert_eq!(c.max_output, 131_072);
        // The V3 generation keeps its real 64k — the versioned row must
        // not swallow it.
        let c = lookup(Some("deepseek-chat"));
        assert_eq!(c.ctx_window, 65_536);
        assert_eq!(c.max_output, 8_192);
        // Kimi K2.5 is 256K per Moonshot (HF card + vendor docs), not 1M —
        // the 1M figure belongs to Kimi K3.
        assert_eq!(lookup(Some("kimi-k2.5")).ctx_window, 262_144);
        assert_eq!(lookup(Some("kimi-k2.6")).ctx_window, 262_144);
        assert_eq!(lookup(Some("kimi-k3")).ctx_window, 1_048_576);
        // org-prefixed forms match too.
        assert_eq!(lookup(Some("moonshotai/kimi-k2.5")).ctx_window, 262_144);
    }

    #[test]
    fn json_shape_round_trip() {
        let j = lookup(Some("claude-sonnet-4-6")).to_json("claude-sonnet-4-6");
        assert!(j.contains("\"known\":true"), "{j}");
        assert!(j.contains("\"ctxWindow\":200000"), "{j}");
        assert!(j.contains("\"maxOutput\":64000"), "{j}");
        assert!(j.contains("\"thinking\":\"budget\""), "{j}");
        assert!(j.contains("\"maxThinkingBudget\":64000"), "{j}");

        let j = lookup(Some("o3")).to_json("o3");
        assert!(j.contains("\"thinking\":\"effort\""), "{j}");
        assert!(j.contains("\"efforts\":[\"low\",\"medium\",\"high\"]"), "{j}");

        let j = lookup(Some("mystery")).to_json("mystery");
        assert!(j.contains("\"known\":false"), "{j}");
        assert!(j.contains("\"thinking\":\"unknown\""), "{j}");

        let j = lookup(Some("gpt-4o")).to_json("gpt-4o");
        assert!(j.contains("\"thinking\":\"none\""), "{j}");
    }

    #[test]
    fn anthropic_budget_scales_with_model() {
        let sonnet = lookup(Some("claude-sonnet-4-6")); // ctx 200k → 60k thinking cap
        // Thinking is inside context window (200k*0.30=60k), not output
        assert_eq!(resolve_thinking_budget(&sonnet, "low"), Some(3_750));
        assert_eq!(resolve_thinking_budget(&sonnet, "medium"), Some(15_000));
        assert_eq!(resolve_thinking_budget(&sonnet, "high"), Some(30_000));
        assert_eq!(resolve_thinking_budget(&sonnet, "max"), Some(54_000));

        // Any model on Anthropic can use thinking now (endpoint-driven, ctx-derived)
        let haiku35 = lookup(Some("claude-3-5-haiku")); // same 200k ctx → same budgets
        assert_eq!(resolve_thinking_budget(&haiku35, "max"), Some(54_000));
        let qwen = lookup(Some("qwen3-32b")); // ctx 131072*0.30=39322 → max 35390
        assert_eq!(resolve_thinking_budget(&qwen, "max"), Some(35_390));

        // Unknown model (ctx 0) gets endpoint max 64k → high 32000
        let unk = lookup(Some("weird-model"));
        assert_eq!(resolve_thinking_budget(&unk, "high"), Some(32_000));
        assert_eq!(resolve_thinking_budget(&unk, "max"), Some(57_600));
    }

    #[test]
    fn openai_effort_falls_back_to_highest_supported() {
        let gpt5 = lookup(Some("gpt-5")); // minimal..high
        assert_eq!(resolve_openai_effort(&gpt5, Some("max")), Some("high"));
        assert_eq!(resolve_openai_effort(&gpt5, Some("minimal")), Some("minimal"));
        assert_eq!(resolve_openai_effort(&gpt5, Some("off")), None);

        let o3 = lookup(Some("o3")); // low..high
        assert_eq!(resolve_openai_effort(&o3, Some("max")), Some("high"));
        assert_eq!(resolve_openai_effort(&o3, Some("medium")), Some("medium"));

        // Non-reasoning model: omit entirely.
        let gpt4o = lookup(Some("gpt-4o"));
        assert_eq!(resolve_openai_effort(&gpt4o, Some("max")), None);

        // Unknown model on OpenAI: omit (don't assume), provider default
        let unk = lookup(Some("weird-model"));
        assert_eq!(resolve_openai_effort(&unk, Some("max")), None);
        assert_eq!(resolve_openai_effort(&unk, Some("medium")), None);
    }
}
