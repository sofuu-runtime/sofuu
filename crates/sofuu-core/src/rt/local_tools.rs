// rt/local_tools.rs — local-model tool calling (PLAN-SOFUU-CORE milestone C1).
//
// Local models (Qwen-3, LFM-2) have no provider-side tool-call API: the
// model expresses a call in TEXT, using the Hermes/Qwen protocol — a JSON
// object with `name` + `arguments` inside <tool_call></tool_call> tags.
// Remote providers instead stream OpenAI-shaped `tool_calls` deltas. The
// agent loop consumes exactly ONE shape (mergeStreamToolCalls in
// src/js/agent.js), so this module is the bridge for the local path:
//
//   * render_tools_block          — tool specs for the local system prompt
//   * render_assistant_tool_calls — prior calls back into ChatML history
//   * ToolCallParser              — incremental tag scanner over tokens
//
// Parsing is deliberately conservative: a block becomes a tool call ONLY
// when its payload parses as an object with a non-empty string `name`.
// Anything else — malformed JSON, an unknown shape — falls through as
// visible text, so no model output is ever silently swallowed.

use std::sync::atomic::{AtomicU64, Ordering};

/// One complete tool call, OpenAI-shaped for the JS-side merger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCall {
    pub id: String,
    pub name: String,
    /// Arguments as a JSON string (the OpenAI `function.arguments` shape).
    pub arguments: String,
}

/// A tool spec to render into the local system prompt.
pub struct ToolSpec<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub parameters_json: Option<&'a str>,
}

/// One event out of the incremental parser.
#[derive(Debug, PartialEq, Eq)]
pub enum ToolEvent {
    Text(String),
    Call(ParsedCall),
}

/// Accepted opening tags: the canonical Hermes/Qwen `<tool_call>` plus the
/// `<tools>` near-miss small models emit when they confuse the spec wrapper
/// with the call wrapper. Either way the payload must still parse as a
/// `{name, arguments}` object, so prose about tools can never become a call.
pub const OPEN_TAGS: [&str; 2] = ["<tool_call>", "<tools>"];
pub const CLOSE_TAGS: [&str; 2] = ["</tool_call>", "</tools>"];

static CALL_SEQ: AtomicU64 = AtomicU64::new(1);

fn next_call_id() -> String {
    let n = CALL_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("call_local_{n}")
}

/* ------------------------------------------------------------------ */
/* Prompt rendering                                                    */
/* ------------------------------------------------------------------ */

/// The Qwen-3 / Hermes tool-call system block. This is the exact protocol
/// the Qwen-3 family was trained on: specs inside <tools></tools>, calls
/// inside <tool_call></tool_call>.
pub fn render_tools_block(specs: &[ToolSpec]) -> String {
    if specs.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "# Tools\n\n\
         You may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n\
         <tools>\n",
    );
    for s in specs {
        let params: serde_json::Value = s
            .parameters_json
            .and_then(|p| serde_json::from_str(p).ok())
            .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
        let obj = serde_json::json!({
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": params,
            }
        });
        out.push_str(&obj.to_string());
        out.push('\n');
    }
    out.push_str(
        "</tools>\n\n\
         For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
         <tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
         </tool_call>",
    );
    out
}

/// Render an assistant message's OpenAI-shaped `tool_calls` JSON array back
/// into Hermes `<tool_call>` blocks for the ChatML transcript, so the model
/// sees its own prior calls on the next turn. Returns "" when the JSON is
/// unusable (the caller then renders content only).
pub fn render_assistant_tool_calls(tool_calls_json: &str) -> String {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(tool_calls_json) else {
        return String::new();
    };
    let Some(arr) = v.as_array() else {
        return String::new();
    };
    let mut out = String::new();
    for entry in arr {
        let f = entry.get("function").unwrap_or(entry);
        let Some(name) = f.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        let args_val: serde_json::Value = match f.get("arguments").or_else(|| f.get("parameters")) {
            Some(serde_json::Value::String(s)) => serde_json::from_str::<serde_json::Value>(s)
                .unwrap_or_else(|_| serde_json::Value::String(s.clone())),
            Some(other) => other.clone(),
            None => serde_json::json!({}),
        };
        let call = serde_json::json!({"name": name, "arguments": args_val});
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(OPEN_TAGS[0]);
        out.push('\n');
        out.push_str(&call.to_string());
        out.push('\n');
        out.push_str(CLOSE_TAGS[0]);
    }
    out
}

/* ------------------------------------------------------------------ */
/* Payload parsing                                                     */
/* ------------------------------------------------------------------ */

/// Strip a ```json … ``` fence around the payload, if present.
fn strip_code_fence(s: &str) -> &str {
    let t = s.trim();
    let Some(rest) = t.strip_prefix("```") else {
        return t;
    };
    // Drop the fence's first line (```json / ```), then the trailing fence.
    let body = match rest.find('\n') {
        Some(nl) => &rest[nl + 1..],
        None => rest,
    };
    body.trim_end().strip_suffix("```").map(str::trim).unwrap_or(body.trim())
}

/// Drop commas that are immediately followed (past whitespace) by `}` or `]`.
/// The single most common small-model JSON slip; everything else falls
/// through to the visible-text path rather than being guessed at.
fn repair_trailing_commas(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut iter = s.chars().peekable();
    while let Some(c) = iter.next() {
        if c == ',' {
            let mut look = Vec::new();
            while let Some(&n) = iter.peek() {
                if n.is_whitespace() {
                    look.push(iter.next().unwrap());
                } else {
                    break;
                }
            }
            if matches!(iter.peek(), Some('}') | Some(']')) {
                out.extend(look); // drop the comma, keep the whitespace
                continue;
            }
            out.push(c);
            out.extend(look);
        } else {
            out.push(c);
        }
    }
    out
}

/// Parse a `<tool_call>` payload into (name, arguments-json-string).
/// `None` means "not a tool call" — the caller emits it as text.
fn parse_call_payload(payload: &str) -> Option<(String, String)> {
    let cleaned = strip_code_fence(payload);
    let v: serde_json::Value = serde_json::from_str(cleaned)
        .or_else(|_| serde_json::from_str(&repair_trailing_commas(cleaned)))
        .ok()?;
    let obj = v.as_object()?;
    let name = obj.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let args_json = match obj.get("arguments").or_else(|| obj.get("parameters")) {
        Some(serde_json::Value::String(s)) => {
            // Some models emit arguments as a JSON *string*; normalize to
            // the object text when it parses, else keep the raw string.
            match serde_json::from_str::<serde_json::Value>(s) {
                Ok(inner) => inner.to_string(),
                Err(_) => serde_json::Value::String(s.clone()).to_string(),
            }
        }
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    Some((name, args_json))
}

/* ------------------------------------------------------------------ */
/* Incremental parser                                                  */
/* ------------------------------------------------------------------ */

/// Longest suffix of `buf` that is a proper prefix of any accepted opening
/// tag (so it can still grow into one on the next feed) — held back, never
/// emitted.
fn partial_tag_suffix(buf: &str) -> usize {
    let mut best = 0usize;
    for tag in OPEN_TAGS {
        let max = tag.len().saturating_sub(1).min(buf.len());
        for k in (1..=max).rev() {
            if buf.ends_with(&tag[..k]) {
                if k > best {
                    best = k;
                }
                break;
            }
        }
    }
    best
}

/// Incremental tool-call scanner over streamed text. Feed it token
/// fragments in order; it emits visible text and completed calls. Call
/// `finish()` once at end-of-stream to flush held-back text and to accept
/// a final unterminated block (models routinely drop the closing tag).
pub struct ToolCallParser {
    buf: String,
    in_call: bool,
    call_buf: String,
    open_tag: &'static str,
    close_tag: &'static str,
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            in_call: false,
            call_buf: String::new(),
            open_tag: OPEN_TAGS[0],
            close_tag: CLOSE_TAGS[0],
        }
    }

    pub fn feed(&mut self, chunk: &str) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        if self.in_call {
            self.call_buf.push_str(chunk);
        } else {
            self.buf.push_str(chunk);
        }
        self.drain(&mut out);
        out
    }

    pub fn finish(&mut self) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        if self.in_call {
            let payload = std::mem::take(&mut self.call_buf);
            let open = self.open_tag;
            self.in_call = false;
            match parse_call_payload(&payload) {
                Some((name, arguments)) => out.push(ToolEvent::Call(ParsedCall {
                    id: next_call_id(),
                    name,
                    arguments,
                })),
                None => out.push(ToolEvent::Text(format!("{open}{payload}"))),
            }
        }
        if !self.buf.is_empty() {
            out.push(ToolEvent::Text(std::mem::take(&mut self.buf)));
        }
        out
    }

    fn drain(&mut self, out: &mut Vec<ToolEvent>) {
        loop {
            if self.in_call {
                let close = self.close_tag;
                let Some(pos) = self.call_buf.find(close) else {
                    return;
                };
                let payload = self.call_buf[..pos].to_string();
                let rest = self.call_buf[pos + close.len()..].to_string();
                let open = self.open_tag;
                self.call_buf.clear();
                self.in_call = false;
                match parse_call_payload(&payload) {
                    Some((name, arguments)) => out.push(ToolEvent::Call(ParsedCall {
                        id: next_call_id(),
                        name,
                        arguments,
                    })),
                    None => out.push(ToolEvent::Text(format!("{open}{payload}{close}"))),
                }
                self.buf.push_str(&rest);
                continue;
            }

            /* Earliest accepted opening tag wins. */
            let mut found: Option<(usize, &'static str, &'static str)> = None;
            for (open, close) in OPEN_TAGS.iter().zip(CLOSE_TAGS.iter()) {
                let (open, close): (&'static str, &'static str) = (*open, *close);
                if let Some(pos) = self.buf.find(open) {
                    if found.map_or(true, |(best, _, _)| pos < best) {
                        found = Some((pos, open, close));
                    }
                }
            }
            if let Some((pos, open, close)) = found {
                if pos > 0 {
                    out.push(ToolEvent::Text(self.buf[..pos].to_string()));
                }
                let rest = self.buf[pos + open.len()..].to_string();
                self.buf.clear();
                self.call_buf = rest;
                self.in_call = true;
                self.open_tag = open;
                self.close_tag = close;
                continue;
            }

            // No complete opening tag: hold back a suffix that could still
            // grow into one; everything before it is safe to emit.
            let hold = partial_tag_suffix(&self.buf);
            let emit_len = self.buf.len() - hold;
            if emit_len > 0 {
                out.push(ToolEvent::Text(self.buf[..emit_len].to_string()));
                self.buf.drain(..emit_len);
            }
            return;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Tests                                                               */
/* ------------------------------------------------------------------ */

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(events: &[ToolEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| match e {
                ToolEvent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    fn calls(events: &[ToolEvent]) -> Vec<ParsedCall> {
        events
            .iter()
            .filter_map(|e| match e {
                ToolEvent::Call(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn plain_text_passes_through() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed("hello world");
        ev.extend(p.finish());
        assert_eq!(texts(&ev), vec!["hello world"]);
        assert!(calls(&ev).is_empty());
    }

    #[test]
    fn single_call_one_feed() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed(
            "Let me check.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>\nDone.",
        );
        ev.extend(p.finish());
        let cs = calls(&ev);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].name, "read_file");
        assert_eq!(cs[0].arguments, "{\"path\":\"a.rs\"}");
        assert!(cs[0].id.starts_with("call_local_"));
        let ts = texts(&ev).join("");
        assert_eq!(ts, "Let me check.\n\nDone.");
    }

    #[test]
    fn tag_split_across_feeds() {
        let mut p = ToolCallParser::new();
        let mut ev = Vec::new();
        for frag in ["<tool", "_call>", "{\"name\": \"x\", \"argu", "ments\": {}}", "</tool_", "call>"] {
            ev.extend(p.feed(frag));
        }
        ev.extend(p.finish());
        let cs = calls(&ev);
        assert_eq!(cs.len(), 1, "events: {ev:?}");
        assert_eq!(cs[0].name, "x");
        assert_eq!(cs[0].arguments, "{}");
        assert!(texts(&ev).is_empty());
    }

    #[test]
    fn partial_tag_is_held_back() {
        let mut p = ToolCallParser::new();
        let ev1 = p.feed("hi <tool");
        // "<tool" could still become a tag — only "hi " may be emitted.
        assert_eq!(texts(&ev1), vec!["hi "]);
        let mut ev2 = p.feed("_call>\n{\"name\": \"t\", \"arguments\": {}}\n</tool_call>");
        ev2.extend(p.finish());
        assert_eq!(calls(&ev2).len(), 1);
        assert_eq!(calls(&ev2)[0].name, "t");
    }

    #[test]
    fn two_calls_one_feed() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed(
            "<tool_call>{\"name\": \"a\", \"arguments\": {\"n\": 1}}</tool_call> mid <tool_call>{\"name\": \"b\", \"arguments\": {\"n\": 2}}</tool_call>",
        );
        ev.extend(p.finish());
        let cs = calls(&ev);
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0].name, "a");
        assert_eq!(cs[1].name, "b");
        assert_eq!(texts(&ev).join(""), " mid ");
    }

    #[test]
    fn arguments_as_json_string_normalized() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed(
            "<tool_call>{\"name\": \"r\", \"arguments\": \"{\\\"path\\\": \\\"x\\\"}\"}</tool_call>",
        );
        ev.extend(p.finish());
        let cs = calls(&ev);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].arguments, "{\"path\":\"x\"}");
    }

    #[test]
    fn trailing_comma_repaired() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed("<tool_call>{\"name\": \"t\", \"arguments\": {\"a\": 1,},}</tool_call>");
        ev.extend(p.finish());
        let cs = calls(&ev);
        assert_eq!(cs.len(), 1, "events: {ev:?}");
        assert_eq!(cs[0].arguments, "{\"a\":1}");
    }

    #[test]
    fn code_fenced_payload() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed(
            "<tool_call>\n```json\n{\"name\": \"t\", \"arguments\": {}}\n```\n</tool_call>",
        );
        ev.extend(p.finish());
        assert_eq!(calls(&ev).len(), 1);
    }

    #[test]
    fn malformed_payload_passes_through_as_text() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed("<tool_call>not json at all</tool_call>");
        ev.extend(p.finish());
        assert!(calls(&ev).is_empty());
        assert_eq!(texts(&ev).join(""), "<tool_call>not json at all</tool_call>");
    }

    #[test]
    fn unknown_shape_is_text() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed("<tool_call>{\"foo\": 1}</tool_call>");
        ev.extend(p.finish());
        assert!(calls(&ev).is_empty());
        assert_eq!(texts(&ev).join(""), "<tool_call>{\"foo\": 1}</tool_call>");
    }

    #[test]
    fn unterminated_call_parsed_at_finish() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed("<tool_call>{\"name\": \"t\", \"arguments\": {\"a\": 1}}");
        ev.extend(p.finish());
        let cs = calls(&ev);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].name, "t");
    }

    #[test]
    fn unterminated_malformed_at_finish_is_text() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed("<tool_call>{\"name\": ");
        ev.extend(p.finish());
        assert!(calls(&ev).is_empty());
        assert_eq!(texts(&ev).join(""), "<tool_call>{\"name\": ");
    }

    #[test]
    fn held_back_partial_flushed_at_finish() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed("tail <tool");
        ev.extend(p.finish());
        assert_eq!(texts(&ev).join(""), "tail <tool");
    }

    #[test]
    fn render_tools_block_carries_specs_and_protocol() {
        let block = render_tools_block(&[
            ToolSpec {
                name: "read_file",
                description: "Read a file",
                parameters_json: Some("{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}"),
            },
            ToolSpec {
                name: "no_params",
                description: "",
                parameters_json: None,
            },
        ]);
        assert!(block.contains("<tools>"));
        assert!(block.contains("</tools>"));
        assert!(block.contains("\"name\":\"read_file\""));
        assert!(block.contains("\"description\":\"Read a file\""));
        assert!(block.contains("\"path\""));
        assert!(block.contains("\"name\":\"no_params\""));
        assert!(block.contains("\"parameters\":{\"properties\":{},\"type\":\"object\"}"));
        assert!(block.contains("<tool_call>"));
    }

    #[test]
    fn render_tools_block_empty_for_no_specs() {
        assert!(render_tools_block(&[]).is_empty());
    }

    #[test]
    fn render_assistant_tool_calls_openai_shape() {
        let json = r#"[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}]"#;
        let out = render_assistant_tool_calls(json);
        assert_eq!(
            out,
            "<tool_call>\n{\"arguments\":{\"path\":\"a.rs\"},\"name\":\"read_file\"}\n</tool_call>"
        );
    }

    #[test]
    fn render_assistant_tool_calls_object_args_and_bare_shape() {
        let json = r#"[{"name":"t","arguments":{"a":1}}]"#;
        let out = render_assistant_tool_calls(json);
        assert!(out.contains("\"name\":\"t\""));
        assert!(out.contains("\"arguments\":{\"a\":1}"));
    }

    #[test]
    fn render_assistant_tool_calls_bad_json_is_empty() {
        assert!(render_assistant_tool_calls("not json").is_empty());
        assert!(render_assistant_tool_calls("{}").is_empty());
    }

    #[test]
    fn tools_alias_tag_accepted() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed("<tools>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"secret.txt\"}}\n</tools>");
        ev.extend(p.finish());
        let cs = calls(&ev);
        assert_eq!(cs.len(), 1, "events: {ev:?}");
        assert_eq!(cs[0].name, "read_file");
        assert_eq!(cs[0].arguments, "{\"path\":\"secret.txt\"}");
        assert!(texts(&ev).is_empty());
    }

    #[test]
    fn tools_alias_spec_payload_stays_text() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"x\"}}\n</tools>");
        ev.extend(p.finish());
        assert!(calls(&ev).is_empty());
        assert!(texts(&ev).join("").contains("<tools>"));
    }

    #[test]
    fn mixed_alias_and_canonical_tags() {
        let mut p = ToolCallParser::new();
        let mut ev = p.feed(
            "<tools>{\"name\": \"a\", \"arguments\": {}}</tools> mid <tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>",
        );
        ev.extend(p.finish());
        let cs = calls(&ev);
        assert_eq!(cs.len(), 2, "events: {ev:?}");
        assert_eq!(cs[0].name, "a");
        assert_eq!(cs[1].name, "b");
        assert_eq!(texts(&ev).join(""), " mid ");
    }
}
