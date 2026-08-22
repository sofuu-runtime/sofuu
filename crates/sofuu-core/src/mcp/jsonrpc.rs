// sofuu-core — MCP JSON-RPC 2.0 layer (security-sensitive parsing).
//
// Rust port of the protocol logic in src/mcp/mcp.c: request/notification/
// response builders + safe field extraction + response routing. The C
// version hand-rolled string scanning (buffer overflow surface); here we
// use serde_json for parsing (safe, no manual buffer management) and a
// minimal hand-rolled builder for the wire format to keep the binary small.

use std::sync::atomic::{AtomicU32, Ordering};

/// Next request id (per-process, like the C `g_next_id`).
pub fn next_id() -> u32 {
    static ID: AtomicU32 = AtomicU32::new(1);
    ID.fetch_add(1, Ordering::Relaxed)
}

/// Build a JSON-RPC request line: `{"jsonrpc":"2.0","id":N,"method":"...","params":{...}}\n`
pub fn request(id: u32, method: &str, params_json: Option<&str>) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":\"{}\",\"params\":{}}}\n",
        escape(method),
        params_json.unwrap_or("{}")
    )
}

/// Build a JSON-RPC notification (no id).
pub fn notify(method: &str, params_json: Option<&str>) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"method\":\"{}\",\"params\":{}}}\n",
        escape(method),
        params_json.unwrap_or("{}")
    )
}

/// Build a JSON-RPC success response.
pub fn response(id: u32, result_json: Option<&str>) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{}}}\n",
        result_json.unwrap_or("null")
    )
}

/// Build a JSON-RPC error response.
pub fn error(id: u32, code: i32, message: &str) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"error\":{{\"code\":{code},\"message\":\"{}\"}}}}\n",
        escape(message)
    )
}

/// Minimal string escape for embedding into a JSON string literal.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// What a parsed inbound message is.
#[derive(Debug, Clone, PartialEq)]
pub enum Inbound {
    /// A request we must answer (has id + method).
    Request {
        id: u64,
        method: String,
        params: Option<serde_json::Value>,
    },
    /// A notification (no id — no reply).
    Notification {
        method: String,
        params: Option<serde_json::Value>,
    },
    /// A response to one of our requests.
    Response {
        id: u64,
        result: Option<serde_json::Value>,
        error: Option<RpcError>,
    },
    /// Malformed — not valid JSON-RPC.
    Invalid(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

/// Parse a JSON-RPC message. Tolerant: trailing newlines allowed.
pub fn parse(line: &str) -> Inbound {
    let v: serde_json::Value = match serde_json::from_str(line.trim()) {
        Ok(v) => v,
        Err(e) => return Inbound::Invalid(format!("parse error: {e}")),
    };
    let obj = match v.as_object() {
        Some(o) => o,
        None => return Inbound::Invalid("not an object".into()),
    };
    // jsonrpc must be "2.0" (tolerate missing — MCP always sends it).
    let id = obj.get("id").and_then(|i| {
        if i.is_number() {
            i.as_u64()
        } else if i.is_string() {
            i.as_str().and_then(|s| s.parse().ok())
        } else {
            None
        }
    });
    let method = obj.get("method").and_then(|m| m.as_str());
    let params = obj.get("params").cloned();

    if let Some(method) = method {
        let method = method.to_string();
        match id {
            Some(id) => Inbound::Request {
                id,
                method,
                params,
            },
            None => Inbound::Notification { method, params },
        }
    } else if let Some(id) = id {
        // A response: result XOR error.
        let result = obj.get("result").cloned();
        let err = obj.get("error").and_then(|e| {
            e.as_object().map(|o| RpcError {
                code: o.get("code").and_then(|c| c.as_i64()).unwrap_or(-1),
                message: o
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error")
                    .to_string(),
            })
        });
        Inbound::Response { id, result, error: err }
    } else {
        Inbound::Invalid("no id, method, or result".into())
    }
}

/// Extract a string field from a JSON object value (safe).
pub fn field_string(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(|s| s.to_string())
}

/// Extract a u64 field from a JSON object value (safe).
pub fn field_u64(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_u64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_request() {
        let s = request(1, "initialize", Some("{\"protocolVersion\":\"1.0\"}"));
        assert!(s.contains("\"jsonrpc\":\"2.0\""));
        assert!(s.contains("\"id\":1"));
        assert!(s.contains("\"method\":\"initialize\""));
        assert!(s.ends_with('\n'));
    }

    #[test]
    fn builds_notification() {
        let s = notify("notifications/cancelled", None);
        assert!(!s.contains("\"id\""));
        assert!(s.contains("\"method\":\"notifications/cancelled\""));
    }

    #[test]
    fn builds_response_and_error() {
        let r = response(3, Some("{\"ok\":true}"));
        assert!(r.contains("\"result\":{\"ok\":true}"));
        let e = error(3, -32601, "method not found");
        assert!(e.contains("\"code\":-32601"));
        assert!(e.contains("\"message\":\"method not found\""));
    }

    #[test]
    fn escapes_quotes_in_message() {
        let e = error(1, -32000, "bad \"input\"\nnewline");
        assert!(e.contains("bad \\\"input\\\"\\nnewline"));
    }

    #[test]
    fn parses_request() {
        let line = "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"x\"}}\n";
        match parse(line) {
            Inbound::Request { id, method, params } => {
                assert_eq!(id, 5);
                assert_eq!(method, "tools/call");
                assert!(params.is_some());
            }
            other => panic!("expected request, got {other:?}"),
        }
    }

    #[test]
    fn parses_notification() {
        let line = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}";
        match parse(line) {
            Inbound::Notification { method, .. } => {
                assert_eq!(method, "notifications/initialized");
            }
            other => panic!("expected notification, got {other:?}"),
        }
    }

    #[test]
    fn parses_response() {
        let line = "{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"tools\":[]}}";
        match parse(line) {
            Inbound::Response { id, result, error } => {
                assert_eq!(id, 7);
                assert!(result.is_some());
                assert!(error.is_none());
            }
            other => panic!("expected response, got {other:?}"),
        }
    }

    #[test]
    fn parses_error_response() {
        let line = "{\"jsonrpc\":\"2.0\",\"id\":8,\"error\":{\"code\":-32602,\"message\":\"invalid params\"}}";
        match parse(line) {
            Inbound::Response { error: Some(e), .. } => {
                assert_eq!(e.code, -32602);
                assert_eq!(e.message, "invalid params");
            }
            other => panic!("expected error response, got {other:?}"),
        }
    }

    #[test]
    fn rejects_garbage() {
        assert!(matches!(parse("not json"), Inbound::Invalid(_)));
        assert!(matches!(parse("42"), Inbound::Invalid(_)));
        assert!(matches!(parse("{\"foo\":1}"), Inbound::Invalid(_)));
    }

    #[test]
    fn parses_with_trailing_whitespace() {
        let line = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"x\"}\r\n";
        assert!(matches!(parse(line), Inbound::Request { .. }));
    }
}
