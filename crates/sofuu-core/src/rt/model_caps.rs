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

    /* ── OpenAI family ── */
    // GPT-6 generation (Astra, Sol, Terra, Luna): ~1.05M context, 128k output, effort reasoning
    ("gpt-6", 1_050_000, 128_000, Thinking::Effort(GPT5_EFFORTS)),
    // GPT-5.4+ generation (GPT-5.4, GPT-5.5, GPT-5.6 Sol/Terra/Luna): ~1.05M context, 128k output
    ("gpt-5.6", 1_050_000, 128_000, Thinking::Effort(GPT5_EFFORTS)),
    ("gpt-5.5", 1_050_000, 128_000, Thinking::Effort(GPT5_EFFORTS)),
    ("gpt-5.4", 1_050_000, 128_000, Thinking::Effort(GPT5_EFFORTS)),
    // GPT-5 core generation (GPT-5, 5.1, 5.2, 5.3, mini, nano, pro, codex): 400k context, 128k output
    ("gpt-5", 400_000, 128_000, Thinking::Effort(GPT5_EFFORTS)),
    ("gpt-4.1", 1_000_000, 32_768, Thinking::None),
    ("gpt-4o", 128_000, 16_384, Thinking::None),
    ("chatgpt-4o", 128_000, 16_384, Thinking::None),
    ("gpt-4-turbo", 128_000, 4_096, Thinking::None),
    ("gpt-4", 8_192, 8_192, Thinking::None),
    ("gpt-3.5-turbo-16k", 16_384, 4_096, Thinking::None),
    ("gpt-3.5", 4_096, 4_096, Thinking::None),
    ("gpt-audio", 128_000, 16_384, Thinking::None),
    ("gpt-oss", 131_072, 117_964, Thinking::Effort(O_EFFORTS)),
    ("gpt", 1_000_000, 128_000, Thinking::Effort(GPT5_EFFORTS)),
    // o-series reasoners: effort ladder without "minimal"/"max".
    ("o3-pro", 200_000, 100_000, Thinking::Effort(O_EFFORTS)),
    ("o3", 200_000, 100_000, Thinking::Effort(O_EFFORTS)),
    ("o4-mini", 200_000, 100_000, Thinking::Effort(O_EFFORTS)),
    ("o1", 200_000, 100_000, Thinking::Effort(O_EFFORTS)),

    /* ── Claude family ── */
    // 5th-gen+ (Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.x): 1M context, adaptive thinking up to 128k output
    ("claude-haiku-5", 1_000_000, 128_000, Thinking::Budget(128_000)),
    ("claude-sonnet-5", 1_000_000, 128_000, Thinking::Budget(128_000)),
    ("claude-opus-5", 1_000_000, 128_000, Thinking::Budget(128_000)),
    ("claude-fable-5", 1_000_000, 128_000, Thinking::Budget(128_000)),
    ("claude-fable", 1_000_000, 128_000, Thinking::Budget(128_000)),
    ("claude-haiku-latest", 1_000_000, 128_000, Thinking::Budget(128_000)),
    ("claude-sonnet-latest", 1_000_000, 128_000, Thinking::Budget(128_000)),
    ("claude-opus-latest", 1_000_000, 128_000, Thinking::Budget(128_000)),
    ("claude-5", 1_000_000, 128_000, Thinking::Budget(128_000)),
    // 4th-gen+ extended thinking: budget_tokens only needs to stay below
    // max_tokens, so the practical ceiling IS the output cap. Sofuu sends
    // max_output as max_tokens, which unlocks real 64k thinking budgets.
    ("claude-opus-4", 200_000, 32_000, Thinking::Budget(32_000)),
    ("claude-sonnet-4", 200_000, 64_000, Thinking::Budget(64_000)),
    ("claude-haiku-4", 200_000, 64_000, Thinking::Budget(64_000)),
    ("claude-4", 200_000, 64_000, Thinking::Budget(64_000)),
    // 3-7 introduced extended thinking.
    ("claude-3-7", 200_000, 64_000, Thinking::Budget(64_000)),
    ("claude-3-5", 200_000, 8_192, Thinking::None),
    ("claude-3", 200_000, 4_096, Thinking::None),
    ("claude-2", 100_000, 4_096, Thinking::None),
    ("claude", 200_000, 8_192, Thinking::None),

    /* ── Google Gemini, Gemma & Lyria ── */
    // Gemini 4 generation (Argon, Carbon, Hydrogen, Neon): 1M context, huge output ceiling
    ("gemini-4", 1_048_576, 943_718, Thinking::Budget(128_000)),
    // Gemini 3.x generation (3.0, 3.1, 3.5, 3.6, 3.7, 3.8 Flash, Flash-Lite, Pro): 1M context, 64k output
    ("gemini-3", 1_048_576, 65_536, Thinking::Budget(65_536)),
    // Gemini 2.5 / 2.0 / 1.5
    ("gemini-2.5", 1_048_576, 65_536, Thinking::None),
    ("gemini-2.0", 1_048_576, 8_192, Thinking::None),
    ("gemini-1.5", 1_048_576, 8_192, Thinking::None),
    ("gemini-nano", 65_536, 58_982, Thinking::None),
    ("gemini", 1_048_576, 65_536, Thinking::None),
    // Gemma
    ("gemma-4", 262_144, 32_768, Thinking::None),
    ("gemma-3", 131_072, 16_384, Thinking::None),
    ("gemma-2", 8_192, 2_048, Thinking::None),
    ("gemma", 8_192, 2_048, Thinking::None),
    // Lyria
    ("lyria-3", 1_048_576, 65_536, Thinking::None),
    ("lyria", 1_048_576, 65_536, Thinking::None),

    /* ── DeepSeek ── */
    // V4 generation (V4-Flash / V4.1-Flash / V4-Pro): 1M context, up to ~393k output
    ("deepseek-v4", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),
    ("deepseek-v3", 163_840, 147_456, Thinking::None),
    ("deepseek-r1", 163_840, 32_000, Thinking::Effort(O_EFFORTS)),
    ("deepseek-flash", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),
    ("deepseek-pro", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),
    ("deepseek-chat", 65_536, 8_192, Thinking::None),
    ("deepseek", 65_536, 8_192, Thinking::None),

    /* ── Qwen (Alibaba) ── */
    ("qwen4", 1_000_000, 131_072, Thinking::None),
    ("qwen3.8", 1_000_000, 131_072, Thinking::None),
    ("qwen3.7", 1_000_000, 131_072, Thinking::None),
    ("qwen3.6", 1_000_000, 65_536, Thinking::None),
    ("qwen3.5", 1_000_000, 65_536, Thinking::None),
    ("qwen3-coder-flash", 1_000_000, 65_536, Thinking::None),
    ("qwen3-coder", 262_144, 65_536, Thinking::None),
    ("qwen3-vl", 262_144, 32_768, Thinking::None),
    ("qwen3", 131_072, 32_768, Thinking::None),
    ("qwq", 131_072, 32_768, Thinking::Effort(O_EFFORTS)),
    ("qwen-2.5-coder", 128_000, 29_491, Thinking::None),
    ("qwen-2.5", 128_000, 16_384, Thinking::None),
    ("qwen", 32_768, 8_192, Thinking::None),

    /* ── Zhipu GLM ── */
    ("glm-5.3", 1_048_576, 131_072, Thinking::None),
    ("glm-5.2", 1_048_576, 943_718, Thinking::None),
    ("glm-5", 204_800, 131_072, Thinking::None),
    ("glm-4.7", 200_000, 117_964, Thinking::None),
    ("glm-4.6", 204_800, 131_072, Thinking::None),
    ("glm-4.5", 131_072, 98_304, Thinking::None),
    ("glm-4", 131_072, 32_768, Thinking::None),
    ("glm", 131_072, 32_768, Thinking::None),

    /* ── Moonshot Kimi ── */
    ("kimi-k3", 1_048_576, 943_718, Thinking::None),
    ("kimi-k2.7", 262_144, 235_929, Thinking::None),
    ("kimi-k2.6", 262_144, 235_929, Thinking::None),
    ("kimi-k2.5", 262_144, 235_929, Thinking::None),
    ("kimi-k2", 131_072, 98_304, Thinking::None),
    ("kimi", 262_144, 131_072, Thinking::None),

    /* ── MiniMax ── */
    ("minimax-m3", 1_048_576, 512_000, Thinking::None),
    ("minimax-m2.7", 196_608, 131_072, Thinking::None),
    ("minimax-m2.5", 198_000, 32_768, Thinking::None),
    ("minimax", 196_608, 131_072, Thinking::None),

    /* ── xAI Grok ── */
    ("grok-4.7", 500_000, 450_000, Thinking::None),
    ("grok-4.6", 500_000, 450_000, Thinking::None),
    ("grok-4.5", 500_000, 450_000, Thinking::None),
    ("grok-4.3", 1_000_000, 900_000, Thinking::None),
    ("grok-4.2", 2_000_000, 1_800_000, Thinking::None),
    ("grok-4", 500_000, 65_536, Thinking::None),
    ("grok", 131_072, 8_192, Thinking::None),

    /* ── Mistral ── */
    ("mistral-large-4", 1_048_576, 262_144, Thinking::None),
    ("mistral-large-25", 262_144, 209_715, Thinking::None),
    ("mistral-large", 131_072, 104_857, Thinking::None),
    ("mistral-medium-3", 262_144, 209_715, Thinking::None),
    ("mistral-small-26", 262_144, 209_715, Thinking::None),
    ("mistral-small-3", 256_000, 16_384, Thinking::None),
    ("devstral", 262_144, 209_715, Thinking::None),
    ("codestral", 256_000, 204_800, Thinking::None),
    ("pixtral-large", 128_000, 102_400, Thinking::None),
    ("pixtral-12b", 128_000, 8_192, Thinking::None),
    ("pixtral", 128_000, 32_768, Thinking::None),
    ("voxtral-small", 32_768, 26_214, Thinking::Effort(O_EFFORTS)),
    ("voxtral", 32_768, 26_214, Thinking::Effort(O_EFFORTS)),
    ("mistral-nemo", 131_072, 16_384, Thinking::None),
    ("ministral", 262_144, 209_715, Thinking::None),
    ("mixtral-8x22b", 65_536, 52_428, Thinking::None),
    ("mixtral", 32_768, 8_192, Thinking::None),
    ("mistral", 32_768, 8_192, Thinking::None),

    /* ── Meta Llama & Muse ── */
    ("llama-4", 1_000_000, 32_768, Thinking::None),
    ("llama4", 1_000_000, 32_768, Thinking::None),
    ("llama-3.3", 131_072, 115_200, Thinking::None),
    ("llama3.3", 131_072, 115_200, Thinking::None),
    ("llama-3.2", 131_072, 117_964, Thinking::None),
    ("llama3.2", 131_072, 117_964, Thinking::None),
    ("llama-3.1", 131_072, 16_384, Thinking::None),
    ("llama3.1", 131_072, 16_384, Thinking::None),
    ("llama-3", 8_192, 4_096, Thinking::None),
    ("llama3", 8_192, 4_096, Thinking::None),
    ("llama", 4_096, 2_048, Thinking::None),
    ("muse", 1_048_576, 943_718, Thinking::Budget(128_000)),
    ("ember", 1_048_576, 943_718, Thinking::Effort(O_EFFORTS)),

    /* ── Inclusion Ling & Recursive Flint ── */
    ("ling-3.2", 1_048_576, 262_144, Thinking::None),
    ("ling-3.1", 1_000_000, 262_144, Thinking::None),
    ("ling-3", 262_144, 32_768, Thinking::None),
    ("flint-v1", 262_144, 235_929, Thinking::None),

    /* ── Xiaomi Mimo ── */
    ("mimo-v2.6", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),
    ("mimo-v2.5", 1_000_000, 65_536, Thinking::Effort(O_EFFORTS)),
    ("mimo", 1_000_000, 65_536, Thinking::Effort(O_EFFORTS)),

    /* ── Aion Labs ── */
    ("aion-3.5", 262_144, 32_768, Thinking::Effort(O_EFFORTS)),
    ("aion-3", 131_072, 32_768, Thinking::Effort(O_EFFORTS)),
    ("aion-2", 131_072, 32_768, Thinking::Effort(O_EFFORTS)),
    ("aion", 131_072, 32_768, Thinking::Effort(O_EFFORTS)),

    /* ── Pareto ── */
    ("pareto-code", 2_000_000, 128_000, Thinking::Effort(O_EFFORTS)),
    ("pareto-26.10", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),
    ("pareto-26.9", 262_144, 131_072, Thinking::Effort(O_EFFORTS)),
    ("pareto", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),

    /* ── Perplexity Sonar ── */
    ("sonar-deep-research", 128_000, 115_200, Thinking::None),
    ("sonar-reasoning", 128_000, 115_200, Thinking::Effort(O_EFFORTS)),
    ("sonar-pro", 200_000, 8_000, Thinking::None),
    ("sonar", 128_000, 115_200, Thinking::None),

    /* ── Tencent Hunyuan & HY ── */
    ("hy4", 1_048_576, 65_536, Thinking::Effort(O_EFFORTS)),
    ("hy3", 262_144, 235_929, Thinking::Effort(O_EFFORTS)),
    ("hy-mt", 8_192, 4_096, Thinking::None),
    ("hunyuan", 131_072, 117_964, Thinking::None),

    /* ── Sakana AI ── */
    ("fugu-ultra", 1_000_000, 128_000, Thinking::Effort(O_EFFORTS)),
    ("fugu-max", 1_000_000, 128_000, Thinking::Effort(O_EFFORTS)),
    ("fugu", 1_000_000, 128_000, Thinking::Effort(O_EFFORTS)),
    ("sakana-namazu", 262_144, 65_536, Thinking::None),
    ("sakana", 262_144, 65_536, Thinking::None),

    /* ── Upstage Solar ── */
    ("solar-pro", 524_288, 131_072, Thinking::Effort(O_EFFORTS)),
    ("solar-mini", 524_288, 131_072, Thinking::Effort(O_EFFORTS)),
    ("solar", 524_288, 131_072, Thinking::Effort(O_EFFORTS)),

    /* ── ThinkingMachines Inkling ── */
    ("inkling-2", 1_048_576, 943_718, Thinking::Effort(O_EFFORTS)),
    ("inkling-small", 524_288, 262_144, Thinking::Effort(O_EFFORTS)),
    ("inkling", 524_288, 471_859, Thinking::Effort(O_EFFORTS)),

    /* ── StepFun Step ── */
    ("step-5", 1_000_000, 64_000, Thinking::Effort(O_EFFORTS)),
    ("step-3.7", 262_144, 230_400, Thinking::Effort(O_EFFORTS)),
    ("step-3.5", 262_144, 65_536, Thinking::Effort(O_EFFORTS)),
    ("step-3", 262_144, 65_536, Thinking::Effort(O_EFFORTS)),
    ("step", 262_144, 65_536, Thinking::Effort(O_EFFORTS)),

    /* ── Liquid AI LFM ── */
    ("lfm-3", 262_144, 32_768, Thinking::None),
    ("lfm-2.5", 65_536, 8_192, Thinking::None),
    ("lfm", 65_536, 8_192, Thinking::None),

    /* ── Microsoft Phi ── */
    ("phi-4", 16_384, 16_384, Thinking::None),
    ("phi-3.5", 128_000, 4_096, Thinking::None),
    ("phi-3", 128_000, 4_096, Thinking::None),
    ("phi", 16_384, 4_096, Thinking::None),

    /* ── Nex-AGI Nex ── */
    ("nex-n2.6", 262_144, 235_929, Thinking::None),
    ("nex-n2.5", 262_144, 235_929, Thinking::None),
    ("nex", 262_144, 235_929, Thinking::None),

    /* ── Morph Studio ── */
    ("morph-v3-large", 262_144, 131_072, Thinking::None),
    ("morph-v3-fast", 81_920, 38_000, Thinking::None),
    ("morph-v3", 131_072, 65_536, Thinking::None),
    ("morph", 131_072, 65_536, Thinking::None),

    /* ── Relace ── */
    ("relace-apply-3", 256_000, 128_000, Thinking::None),
    ("relace-search", 256_000, 128_000, Thinking::None),
    ("relace", 256_000, 128_000, Thinking::None),

    /* ── Typesafe JEV ── */
    ("jev-router", 1_000_000, 128_000, Thinking::Effort(O_EFFORTS)),
    ("jev", 1_000_000, 128_000, Thinking::Effort(O_EFFORTS)),

    /* ── Particle AI ── */
    ("particle-1", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),
    ("particle", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),

    /* ── Prism Ternary Bonsai ── */
    ("ternary-bonsai-2", 262_144, 65_536, Thinking::Effort(O_EFFORTS)),
    ("ternary-bonsai", 262_144, 65_536, Thinking::Effort(O_EFFORTS)),
    ("ternary", 262_144, 65_536, Thinking::Effort(O_EFFORTS)),

    /* ── Dots Studio ── */
    ("dots-3", 512_000, 460_800, Thinking::Effort(O_EFFORTS)),
    ("dots", 512_000, 460_800, Thinking::Effort(O_EFFORTS)),

    /* ── Elorian Glimpse ── */
    ("glimpse", 262_144, 256_000, Thinking::Effort(O_EFFORTS)),

    /* ── IBM Granite ── */
    ("granite-4.2", 131_072, 117_964, Thinking::Effort(O_EFFORTS)),
    ("granite-3.2", 131_072, 8_192, Thinking::None),
    ("granite-3", 131_072, 8_192, Thinking::None),
    ("granite", 131_072, 8_192, Thinking::None),

    /* ── Nous Hermes ── */
    ("hermes-4", 131_072, 117_964, Thinking::None),
    ("hermes-3", 131_072, 8_192, Thinking::None),
    ("hermes", 131_072, 8_192, Thinking::None),

    /* ── Meituan Longcat ── */
    ("longcat-2.0", 1_048_576, 262_144, Thinking::None),
    ("longcat", 1_048_576, 262_144, Thinking::None),

    /* ── Moondream ── */
    ("moondream3.1", 32_000, 4_096, Thinking::None),
    ("moondream3", 32_000, 4_096, Thinking::None),
    ("moondream", 32_000, 4_096, Thinking::None),

    /* ── Motif ── */
    ("motif-3", 262_144, 235_929, Thinking::None),
    ("motif", 262_144, 235_929, Thinking::None),

    /* ── Writer Palmyra ── */
    ("palmyra-x5", 1_040_000, 8_192, Thinking::None),
    ("palmyra", 1_040_000, 8_192, Thinking::None),

    /* ── Recursive Ripple ── */
    ("ripple-65b", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),
    ("ripple", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),

    /* ── Reka ── */
    ("reka-flash-3", 65_536, 58_982, Thinking::None),
    ("reka-edge", 16_384, 14_745, Thinking::None),
    ("reka", 65_536, 16_384, Thinking::None),

    /* ── Schematron ── */
    ("schematron-v2", 128_000, 8_192, Thinking::None),
    ("schematron", 128_000, 8_192, Thinking::None),

    /* ── Beam ── */
    ("beam-1.0", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),
    ("beam", 1_048_576, 131_072, Thinking::Effort(O_EFFORTS)),

    /* ── TheDrummer models ── */
    ("cydonia-24b", 131_072, 117_964, Thinking::None),
    ("cydonia", 131_072, 117_964, Thinking::None),
    ("daisyrose-27b", 262_144, 235_929, Thinking::Effort(O_EFFORTS)),
    ("daisyrose", 262_144, 235_929, Thinking::Effort(O_EFFORTS)),
    ("skyfall-36b", 32_768, 29_491, Thinking::None),
    ("skyfall", 32_768, 29_491, Thinking::None),
    ("unslopnemo-12b", 1_024_000, 819_200, Thinking::None),
    ("unslopnemo", 1_024_000, 819_200, Thinking::None),

    /* ── Amazon Nova ── */
    ("nova-2-lite", 1_000_000, 65_535, Thinking::None),
    ("nova-premier", 1_000_000, 32_000, Thinking::None),
    ("nova-pro", 300_000, 5_120, Thinking::None),
    ("nova-lite", 300_000, 5_120, Thinking::None),
    ("nova-micro", 128_000, 5_120, Thinking::None),

    /* ── Baidu Ernie ── */
    ("ernie-5.5", 1_024_000, 921_600, Thinking::None),
    ("ernie", 1_024_000, 131_072, Thinking::None),

    /* ── Bytedance Doubao, Seedance & UI-TARS ── */
    ("doubao-1.5", 262_144, 65_536, Thinking::None),
    ("doubao", 131_072, 32_768, Thinking::None),
    ("seedance-2.5", 262_144, 131_072, Thinking::Effort(O_EFFORTS)),
    ("seedance", 262_144, 131_072, Thinking::Effort(O_EFFORTS)),
    ("seedream-5", 262_144, 131_072, Thinking::None),
    ("seedream-4", 262_144, 131_072, Thinking::None),
    ("seed-2", 262_144, 131_072, Thinking::None),
    ("ui-tars", 128_000, 2_048, Thinking::None),

    /* ── NVIDIA Nemotron ── */
    ("nemotron-3.5", 1_048_576, 235_929, Thinking::None),
    ("nemotron-3", 262_144, 65_536, Thinking::None),

    /* ── Cohere Command & North ── */
    ("north-mini-code", 256_000, 64_000, Thinking::Effort(O_EFFORTS)),
    ("north", 256_000, 64_000, Thinking::Effort(O_EFFORTS)),
    ("command-a", 256_000, 64_000, Thinking::None),
    ("command-r+", 128_000, 4_000, Thinking::None),
    ("command-r", 128_000, 4_000, Thinking::None),

    /* ── Meta routers & Utilities ── */
    ("openrouter/auto", 2_000_000, 128_000, Thinking::Effort(O_EFFORTS)),
    ("openrouter/free", 200_000, 16_384, Thinking::Effort(O_EFFORTS)),
    ("auto-code", 2_000_000, 128_000, Thinking::Effort(O_EFFORTS)),
    ("auto", 2_000_000, 128_000, Thinking::Effort(O_EFFORTS)),
    ("switchyard", 1_000_000, 128_000, Thinking::None),
    ("fusion", 1_000_000, 128_000, Thinking::None),
    ("extract", 1_000_000, 900_000, Thinking::None),
    ("summarize", 1_000_000, 4_000, Thinking::None),

    /* ── Specialized & Community models ── */
    ("dolphin-mistral", 128_000, 8_192, Thinking::None),
    ("dolphin", 128_000, 8_192, Thinking::None),
    ("mythomax", 8_192, 7_372, Thinking::None),
    ("perceptron-mk1", 32_768, 8_192, Thinking::None),
    ("perceptron", 32_768, 8_192, Thinking::None),
    ("midm-k-3", 261_795, 235_615, Thinking::None),
    ("midm", 261_795, 235_615, Thinking::None),
    ("l3.1", 131_072, 16_384, Thinking::None),
    ("l3", 8_192, 7_372, Thinking::None),
    ("magnum", 32_768, 4_096, Thinking::None),
    ("remm-slerp", 6_144, 5_529, Thinking::None),
    ("weaver", 8_000, 6_000, Thinking::None),
    ("0gm", 262_144, 32_768, Thinking::Effort(O_EFFORTS)),
    ("free", 200_000, 16_384, Thinking::Effort(O_EFFORTS)),
    ("bodybuilder", 128_000, 4_096, Thinking::None),
];

fn inspect_gguf_caps(path: &str) -> Option<ModelCaps> {
    let clean = crate::infer::clean_model_path(path);
    let clean_str = clean.to_string_lossy().into_owned();
    use std::sync::Mutex;
    static CACHE: Mutex<Option<(String, ModelCaps)>> = Mutex::new(None);
    if let Ok(guard) = CACHE.lock() {
        if let Some((cached_path, caps)) = &*guard {
            if cached_path == &clean_str {
                return Some(*caps);
            }
        }
    }

    let gguf = crate::infer::gguf::GgufFile::open(&clean).ok()?;
    let arch = gguf.get_str("general.architecture").unwrap_or("llama");
    let mut ctx_u32 = gguf
        .get_u32(&format!("{arch}.context_length"))
        .or_else(|| gguf.get_u32("context_length"))
        .or_else(|| gguf.get_u32("context_window"));
    if ctx_u32.is_none() {
        for k in gguf.metadata.keys() {
            if k.ends_with(".context_length") || k.ends_with(".context_window") {
                if let Some(val) = gguf.get_u32(k) {
                    ctx_u32 = Some(val);
                    break;
                }
            }
        }
    }
    let ctx = ctx_u32.unwrap_or(4096) as i32;
    let max_output = (ctx / 4).clamp(512, 4096);

    let chat_tmpl = gguf.get_str("tokenizer.chat_template").unwrap_or("");
    let name = gguf.get_str("general.name").unwrap_or("");
    let finetune = gguf.get_str("general.finetune").unwrap_or("");
    let path_lower = path.to_lowercase();
    let is_reasoning = chat_tmpl.contains("<think>")
        || chat_tmpl.contains("reasoning_content")
        || chat_tmpl.contains("enable_thinking")
        || name.to_lowercase().contains("reasoning")
        || name.to_lowercase().contains("opus-distilled")
        || name.to_lowercase().contains("qwq")
        || name.to_lowercase().contains("r1")
        || finetune.to_lowercase().contains("reasoning")
        || finetune.to_lowercase().contains("distill")
        || path_lower.contains("deepseek-r1")
        || path_lower.contains("qwq")
        || path_lower.contains("distilled")
        || path_lower.contains("reasoning");

    let thinking = if is_reasoning {
        Thinking::Effort(O_EFFORTS)
    } else {
        Thinking::None
    };

    let caps = ModelCaps {
        ctx_window: ctx,
        max_output,
        thinking,
    };

    if let Ok(mut guard) = CACHE.lock() {
        *guard = Some((clean_str, caps));
    }
    Some(caps)
}

/// Resolve capabilities by model name (case-insensitive prefix match;
/// `org/model` forms match too).
pub fn lookup(model: Option<&str>) -> ModelCaps {
    let raw = model.unwrap_or("").trim();
    if raw.is_empty() {
        return NO_CAPS;
    }
    let cleaned = crate::infer::clean_model_path(raw);
    let is_gguf = cleaned.extension().is_some_and(|e| e == "gguf")
        || raw.ends_with(".gguf");
    if is_gguf {
        if let Some(caps) = inspect_gguf_caps(raw) {
            return caps;
        }
        return ModelCaps {
            ctx_window: 4096,
            max_output: 2048,
            thinking: Thinking::None,
        };
    }
    let m = raw.to_lowercase();
    for (prefix, ctx, out, thinking) in TABLE {
        // P3 (AUDIT-2026-09-07): allocation-free `contains("/{prefix}")` —
        // the old format! built a fresh String per table row per lookup,
        // and lookup runs on every request.
        let hit = m.starts_with(prefix)
            || m.match_indices(&['/', ':', '.', '@', '~'][..]).any(|(i, _)| m[i + 1..].starts_with(prefix));
        if hit {
            return ModelCaps { ctx_window: *ctx, max_output: *out, thinking: *thinking };
        }
    }
    if m.contains("thinking") || m.contains("reasoning") {
        return ModelCaps {
            ctx_window: 131_072,
            max_output: 32_768,
            thinking: Thinking::Effort(O_EFFORTS),
        };
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
    if model.is_some_and(|m| m.contains(".gguf")) {
        return reg;
    }
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

        // 5th-gen Claude (Haiku 5.5): 1M context, 128k output, adaptive thinking budget
        let c5 = lookup(Some("claude-haiku-5.5"));
        assert_eq!(c5.ctx_window, 1_000_000);
        assert_eq!(c5.max_output, 128_000);
        assert!(matches!(c5.thinking, Thinking::Budget(128_000)));

        // Generic model with thinking/reasoning in name
        let cr = lookup(Some("custom-provider/my-model-reasoning"));
        assert_eq!(cr.ctx_window, 131_072);
        assert!(matches!(cr.thinking, Thinking::Effort(_)));

        // GPT-6 flagship
        let c6 = lookup(Some("gpt-6.2-sol"));
        assert_eq!(c6.ctx_window, 1_050_000);
        assert_eq!(c6.max_output, 128_000);
        assert!(matches!(c6.thinking, Thinking::Effort(_)));

        // Gemini 4 / 3
        let cg4 = lookup(Some("google/gemini-4-carbon"));
        assert_eq!(cg4.ctx_window, 1_048_576);
        let cg3 = lookup(Some("gemini-3.7-flash"));
        assert_eq!(cg3.ctx_window, 1_048_576);

        // GLM 5.3 & MiniMax M3
        assert_eq!(lookup(Some("z-ai/glm-5.3-flash")).ctx_window, 1_048_576);
        assert_eq!(lookup(Some("minimax/minimax-m3.1-flash")).ctx_window, 1_048_576);

        // Inclusion Ling & Recursive Flint
        assert_eq!(lookup(Some("inclusionai/ling-3.2-flash:free")).ctx_window, 1_048_576);
        assert_eq!(lookup(Some("recursive/flint-v1.0:free")).ctx_window, 262_144);

        // Separator matching (dot, colon, slash, tilde)
        assert_eq!(lookup(Some("bedrock.claude-5-sonnet")).ctx_window, 1_000_000);
        assert_eq!(lookup(Some("0g/claude-haiku-5.5")).ctx_window, 1_000_000);
        assert_eq!(lookup(Some("~anthropic/claude-fable-latest")).ctx_window, 1_000_000);
        assert_eq!(lookup(Some("~openai/gpt-astra-latest")).ctx_window, 1_000_000);

        // New families
        assert_eq!(lookup(Some("meta/muse-ember")).ctx_window, 1_048_576);
        assert_eq!(lookup(Some("xiaomi/mimo-v2.6-flash")).ctx_window, 1_048_576);
        assert_eq!(lookup(Some("perplexity/sonar-pro")).ctx_window, 200_000);
        assert_eq!(lookup(Some("sakana/fugu-ultra")).ctx_window, 1_000_000);
        assert_eq!(lookup(Some("tencent/hy4")).ctx_window, 1_048_576);
        assert_eq!(lookup(Some("microsoft/phi-4")).ctx_window, 16_384);
        assert_eq!(lookup(Some("upstage/solar-pro4")).ctx_window, 524_288);
        assert_eq!(lookup(Some("liquid/lfm-3")).ctx_window, 262_144);
        assert_eq!(lookup(Some("openrouter/auto")).ctx_window, 2_000_000);

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

    #[test]
    fn test_qwen_gguf_caps() {
        let p = "/Users/priyanshuboruah/Downloads/Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf";
        if std::path::Path::new(p).exists() {
            let caps = lookup(Some(p));
            println!("QWEN GGUF CAPS: ctx_window={}, max_output={}", caps.ctx_window, caps.max_output);
            assert!(caps.ctx_window > 0);
        }
    }
}
