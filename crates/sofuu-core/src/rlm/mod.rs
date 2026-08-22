// rlm/mod.rs — Recursive-Language-Model scaffolding (PLAN-RLM Part 1).
//
// The context lives as data inside a hermetic QuickJS sandbox
// (`sandbox`); the model writes small JS snippets against a whitelisted
// API (`chunk`/`peek`/`grep`/`llm`/`final`) and this crate drives the
// loop (`episode`) — no provider/network code in here, the model is an
// external callback. `router` holds the R3 heuristic + decision log.

pub mod episode;
pub mod js_api;
pub mod router;
pub mod sandbox;

use serde::{Deserialize, Serialize};

/// One RLM query: a long context plus a question, never windowed whole.
#[derive(Debug, Clone)]
pub struct RlmRequest {
    pub context: String,
    pub question: String,
    pub opts: RlmOpts,
}

/// A host-tool call a sandbox snippet made via `tool()`/`toolBatch()`
/// (PLAN-AGENTS A4 full form). The episode suspends, the JS driver
/// executes each call through the agent's normal tool path, and the
/// results re-enter the sandbox through `__TOOL_CACHE`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// Arbitrary JSON arguments (default `{}`).
    #[serde(default = "serde_json::Value::default")]
    pub args: serde_json::Value,
}

/// Declarative tool description (schema only — never an executor; the
/// executors stay JS-side so credentials never cross into Rust, G3).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// JSON-schema-ish parameters object, shown to the model.
    #[serde(default)]
    pub parameters: serde_json::Value,
}

/// Budgets and recursion target. Defaults come from PLAN-RLM R1.
#[derive(Debug, Clone)]
pub struct RlmOpts {
    /// Recursion target identifier (used by the provider layer, not here).
    pub provider: String,
    pub model: String,
    /// Max recursion depth for `llm()` sub-calls (default 3).
    pub max_depth: u32,
    /// Max total `llm()`/`llmBatch()` sub-prompts per episode (default 24;
    /// a batch of 3 counts 3).
    pub max_llm_calls: u32,
    /// Wall-clock budget (default 120s).
    pub max_wall_ms: u64,
    /// Per-snippet-eval time slice (default 5s). Each eval is aborted at
    /// min(remaining episode wall, this slice), so one runaway snippet
    /// can't burn the whole episode budget in a single JS_Eval.
    pub eval_slice_ms: u64,
    /// Max model replies per episode (default 16).
    pub max_rounds: u32,
    /// Chunk size in characters (default 16_000 ≈ 4k tokens).
    pub chunk_chars: usize,
    /// Keep the full trace in the result (routing training data, R3).
    pub trace: bool,
    /// Agent tool whitelist (A4): documented in the system prompt and
    /// callable from snippets via `tool()`/`toolBatch()`. Empty = no tool
    /// surface (the pre-A4 sandbox behavior).
    pub tools: Vec<ToolSpec>,
    /// Max total `tool()` calls per episode (default 16; a batch of 3
    /// counts 3).
    pub max_tool_calls: u32,
}

impl Default for RlmOpts {
    fn default() -> Self {
        Self {
            provider: String::new(),
            model: String::new(),
            max_depth: 3,
            max_llm_calls: 24,
            max_wall_ms: 120_000,
            eval_slice_ms: 5_000,
            max_rounds: 16,
            chunk_chars: 16_000,
            trace: false,
            tools: Vec::new(),
            max_tool_calls: 16,
        }
    }
}

/// The episode outcome: final (possibly partial) answer + the trace.
#[derive(Debug, Clone, Serialize)]
pub struct RlmResult {
    pub answer: String,
    /// Number of fulfilled `llm()` sub-prompts.
    pub calls: u32,
    /// Number of fulfilled `tool()` calls (A4).
    #[serde(default)]
    pub tool_calls: u32,
    /// Depth this episode ran at (recursion is driven by the JS layer).
    pub depth_reached: u32,
    /// Number of model replies consumed.
    pub rounds: u32,
    /// Wall-clock milliseconds.
    pub ms: u64,
    /// Every chunk read / peek / grep / emit / llm batch / final / snippet —
    /// capped at 512 events total; overflow is counted, not stored.
    pub trace: Vec<RlmEvent>,
    /// Events refused past the trace cap (G2: the trace must never grow
    /// unboundedly under a snippet flood).
    #[serde(default)]
    pub dropped_events: u32,
    /// Why the episode stopped early, if it did ("budget=max_llm_calls",
    /// "budget=max_rounds", "budget=max_wall_ms", "budget=max_tool_calls",
    /// "no_snippet", "loop_detected").
    pub stopped: Option<String>,
}

/// One trace event: kind + offset-from-start + a short human repr.
#[derive(Debug, Clone, Serialize)]
pub struct RlmEvent {
    /// "chunk" | "peek" | "grep" | "emit" | "llm" | "final" | "snippet" | "note"
    pub kind: String,
    /// Milliseconds since the episode started.
    pub t: u64,
    /// Short, truncated summary — never the whole context.
    pub repr: String,
}

/// A chat message in the model conversation (`Episode` builds these).
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: &str) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: &str) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_opts_match_plan() {
        let o = RlmOpts::default();
        assert_eq!(o.max_depth, 3);
        assert_eq!(o.max_llm_calls, 24);
        assert_eq!(o.max_wall_ms, 120_000);
        assert_eq!(o.max_rounds, 16);
        assert_eq!(o.chunk_chars, 16_000);
    }

    #[test]
    fn result_serializes() {
        let r = RlmResult {
            answer: "a".into(),
            calls: 2,
            tool_calls: 1,
            depth_reached: 0,
            rounds: 3,
            ms: 41,
            trace: vec![RlmEvent { kind: "snippet".into(), t: 5, repr: "len()".into() }],
            dropped_events: 0,
            stopped: None,
        };
        let j = serde_json::to_string(&r).expect("serialize");
        assert!(j.contains("\"answer\":\"a\""));
        assert!(j.contains("\"kind\":\"snippet\""));
    }
}
