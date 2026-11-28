//! Backend-free emulator mocks (`sofuu ui snapshot --mock`, `sofuu.emu`).
//!
//! The emulator never calls a network from a mocked session. `--mock` installs
//! a `fetch` interceptor plus a deterministic `sofuu.ai.complete/stream` stub
//! before the app entry runs, so a screen that calls "Submit" or "Ask Agent"
//! renders its local scenario instead of failing on a missing API key.
//!
//! Config is deliberately service-shaped, not domain-shaped: callers list
//! routes and payloads; the emulator matches method + URL and returns the
//! configured JSON. Unmatched requests get a benign empty payload (or, with
//! `--offline`, a loud `E_EMU_OFFLINE` rejection) so "zero network calls" is a
//! property of the runtime, not a convention the app has to follow.

use serde_json::{json, Value};

/// One mocked HTTP endpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct MockRoute {
    pub url: String,
    pub method: String,
    pub status: u16,
    pub json: Option<Value>,
    pub text: Option<String>,
    pub headers: Value,
}

/// Parsed `--mock` / `--offline` / `--mock-config` state.
#[derive(Clone, Debug, PartialEq)]
pub struct MockOptions {
    pub enabled: bool,
    /// Reject unmatched requests instead of returning the benign default.
    pub offline: bool,
    pub routes: Vec<MockRoute>,
    /// Text returned by the `sofuu.ai.complete/stream` stub.
    pub ai_text: String,
}

impl Default for MockOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            offline: false,
            routes: Vec::new(),
            ai_text: "[emu mock] deterministic local response".to_string(),
        }
    }
}

impl MockOptions {
    /// Parse a `--mock-config` payload. Accepted shapes:
    ///
    /// ```json
    /// {"offline": true, "ai_text": "…", "routes": [
    ///   {"url": "https://api.example.com/state", "method": "GET",
    ///    "status": 200, "json": {"status": "ready"}}
    /// ]}
    /// ```
    ///
    /// A bare array is treated as `routes`.
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let mut out = MockOptions {
            enabled: true,
            ..MockOptions::default()
        };
        let routes = match value {
            Value::Array(items) => items.clone(),
            Value::Object(map) => {
                if let Some(offline) = map.get("offline").and_then(Value::as_bool) {
                    out.offline = offline;
                }
                if let Some(text) = map.get("ai_text").and_then(Value::as_str) {
                    out.ai_text = text.to_string();
                }
                match map.get("routes") {
                    Some(Value::Array(items)) => items.clone(),
                    Some(Value::Null) | None => Vec::new(),
                    Some(_) => return Err("mock config `routes` must be an array".to_string()),
                }
            }
            _ => return Err("mock config must be an object or an array of routes".to_string()),
        };
        for item in routes {
            out.routes.push(parse_route(&item)?);
        }
        Ok(out)
    }

    /// JSON form accepted by [`MockOptions::from_json`]. Used to re-create the
    /// same mocks in the toolbar's snapshot child process.
    pub fn to_json(&self) -> Value {
        json!({
            "offline": self.offline,
            "ai_text": self.ai_text,
            "routes": self.routes.iter().map(|r| json!({
                "url": r.url,
                "method": r.method,
                "status": r.status,
                "json": r.json,
                "text": r.text,
                "headers": r.headers,
            })).collect::<Vec<_>>(),
        })
    }

    /// JS prelude that installs the interceptor. Injected before the app file.
    pub fn prelude(&self) -> String {
        let cfg = json!({
            "offline": self.offline,
            "routes": self.routes.iter().map(|r| json!({
                "url": r.url,
                "method": r.method,
                "status": r.status,
                "json": r.json,
                "text": r.text,
                "headers": r.headers,
            })).collect::<Vec<_>>(),
            "ai_text": self.ai_text,
        });
        let cfg = serde_json::to_string(&cfg).unwrap_or_else(|_| "{}".to_string());
        /* U+2028/U+2029 are legal in JSON strings but terminate a JS line. */
        let cfg = cfg
            .replace('\u{2028}', "\\u2028")
            .replace('\u{2029}', "\\u2029");
        format!(
            r#"(function () {{
  var cfg = {cfg};
  var g = globalThis;
  var baseFetch = (typeof g.fetch === 'function') ? g.fetch : null;
  g.__sofuu_emu_mock = {{
    enabled: true,
    offline: !!cfg.offline,
    routes: cfg.routes.length,
    scenario: g.__sofuu_emu_scenario || null
  }};
  function stripQuery(u) {{ var i = u.indexOf('?'); return i < 0 ? u : u.slice(0, i); }}
  function routeMatch(route, method, url) {{
    if (route.method && route.method !== method) return false;
    var pattern = String(route.url || '');
    if (!pattern) return false;
    if (pattern.charAt(pattern.length - 1) === '*') {{
      return url.indexOf(pattern.slice(0, -1)) === 0;
    }}
    return url === pattern || stripQuery(url) === stripQuery(pattern);
  }}
  function bodyText(route) {{
    if (route.text !== null && route.text !== undefined) return String(route.text);
    if (route.json !== null && route.json !== undefined) return JSON.stringify(route.json);
    return '';
  }}
  function makeResponse(route) {{
    var status = (route.status | 0) || 200;
    var body = bodyText(route);
    var headers = route.headers || {{}};
    function parse() {{
      if (route.json !== null && route.json !== undefined) return route.json;
      try {{ return JSON.parse(body); }} catch (e) {{ return {{}}; }}
    }}
    return {{
      ok: status >= 200 && status < 300,
      status: status,
      statusText: '',
      url: route.url,
      headers: {{
        get: function (k) {{ var v = headers[String(k).toLowerCase()]; return v === undefined ? null : v; }},
        has: function (k) {{ return headers[String(k).toLowerCase()] !== undefined; }}
      }},
      json: function () {{ return Promise.resolve(parse()); }},
      text: function () {{ return Promise.resolve(body); }},
      clone: function () {{ return makeResponse(route); }}
    }};
  }}
  g.fetch = function (input, init) {{
    var url = typeof input === 'string' ? input : ((input && input.url) || String(input));
    var method = String((init && init.method) || (input && input.method) || 'GET').toUpperCase();
    for (var i = 0; i < cfg.routes.length; i++) {{
      if (routeMatch(cfg.routes[i], method, url)) {{
        return Promise.resolve(makeResponse(cfg.routes[i]));
      }}
    }}
    if (cfg.offline) {{
      return Promise.reject(new Error('E_EMU_OFFLINE: ' + method + ' ' + url + ' is not mocked'));
    }}
    /* Mocked sessions never touch the network: unmatched reads resolve to the
     * current scenario shape instead of crashing a render. */
    return Promise.resolve(makeResponse({{
      url: url,
      status: 200,
      json: {{ status: g.__sofuu_emu_scenario || 'ready', items: [], mock: true }}
    }}));
  }};
  function installAi() {{
    var ai = g.sofuu && g.sofuu.ai;
    if (!ai) return;
    try {{
      ai.complete = function () {{
        return Promise.resolve({{ text: cfg.ai_text, usage: {{ input_tokens: 0, output_tokens: 0 }} }});
      }};
      ai.stream = function () {{
        var used = false;
        var it = {{
          abort: function () {{ used = true; }},
          [Symbol.asyncIterator]: function () {{
            return {{
              next: function () {{
                if (used) return Promise.resolve({{ done: true, value: undefined }});
                used = true;
                return Promise.resolve({{ done: false, value: {{ text: cfg.ai_text }} }});
              }}
            }};
          }}
        }};
        return it;
      }};
    }} catch (e) {{ /* a frozen host namespace keeps the real methods */ }}
  }}
  installAi();
}})();"#
        )
    }
}

fn parse_route(item: &Value) -> Result<MockRoute, String> {
    let map = item
        .as_object()
        .ok_or_else(|| "each mock route must be an object".to_string())?;
    let url = map
        .get("url")
        .or_else(|| map.get("match"))
        .and_then(Value::as_str)
        .ok_or_else(|| "mock route needs a `url` (exact, or a prefix ending in `*`)".to_string())?
        .to_string();
    let method = map
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("GET")
        .to_uppercase();
    let status = map.get("status").and_then(Value::as_u64).unwrap_or(200);
    if !(100..=599).contains(&status) {
        return Err(format!("mock route `{url}` has invalid status {status}"));
    }
    let json = map.get("json").cloned().filter(|v| !v.is_null());
    let text = map
        .get("text")
        .or_else(|| map.get("body"))
        .and_then(Value::as_str)
        .map(str::to_string);
    if json.is_none() && text.is_none() {
        return Err(format!("mock route `{url}` needs `json` or `text`"));
    }
    let headers = map.get("headers").cloned().unwrap_or_else(|| json!({}));
    if !headers.is_object() {
        return Err(format!("mock route `{url}` headers must be an object"));
    }
    Ok(MockRoute {
        url,
        method,
        status: status as u16,
        json,
        text,
        headers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_disabled_and_benign() {
        let m = MockOptions::default();
        assert!(!m.enabled);
        assert!(!m.offline);
        assert!(m.routes.is_empty());
        assert!(m.ai_text.contains("emu mock"));
    }

    #[test]
    fn parses_routes_and_offline() {
        let v = json!({
            "offline": true,
            "ai_text": "hello",
            "routes": [
                {"url": "https://api.test/state", "method": "get", "json": {"ok": true}},
                {"url": "https://api.test/items*", "status": 404, "text": "missing",
                 "headers": {"content-type": "text/plain"}}
            ]
        });
        let m = MockOptions::from_json(&v).expect("config");
        assert!(m.enabled);
        assert!(m.offline);
        assert_eq!(m.ai_text, "hello");
        assert_eq!(m.routes.len(), 2);
        assert_eq!(m.routes[0].method, "GET");
        assert_eq!(m.routes[0].status, 200);
        assert_eq!(m.routes[1].status, 404);
        assert_eq!(m.routes[1].text.as_deref(), Some("missing"));
    }

    #[test]
    fn accepts_a_bare_route_array() {
        let v = json!([{"url": "https://api.test/x", "json": []}]);
        let m = MockOptions::from_json(&v).expect("config");
        assert_eq!(m.routes.len(), 1);
        assert_eq!(m.routes[0].json, Some(json!([])));
    }

    #[test]
    fn rejects_malformed_routes() {
        assert!(MockOptions::from_json(&json!(42)).is_err());
        assert!(MockOptions::from_json(&json!({"routes": {}})).is_err());
        assert!(MockOptions::from_json(&json!([{"json": {}}])).is_err());
        assert!(MockOptions::from_json(&json!([{"url": "x"}])).is_err());
        assert!(MockOptions::from_json(&json!([{"url": "x", "json": {}, "status": 999}])).is_err());
    }

    #[test]
    fn to_json_roundtrips_routes() {
        let v = json!({"offline": true, "routes": [
            {"url": "https://api.test/x", "method": "POST", "status": 201,
             "json": {"ok": true}, "headers": {"x-mock": "1"}}
        ]});
        let m = MockOptions::from_json(&v).expect("config");
        let back = MockOptions::from_json(&m.to_json()).expect("roundtrip");
        assert_eq!(m, back);
    }

    #[test]
    fn prelude_is_valid_js_scaffolding() {
        let mut m = MockOptions::default();
        m.enabled = true;
        m.routes.push(MockRoute {
            url: "https://api.test/x".to_string(),
            method: "GET".to_string(),
            status: 200,
            json: Some(json!({"line": "a\u{2028}b"})),
            text: None,
            headers: json!({}),
        });
        let js = m.prelude();
        assert!(js.contains("g.fetch = function"));
        assert!(js.contains("ai.complete = function"));
        assert!(js.contains("E_EMU_OFFLINE"));
        assert!(
            js.contains("\\u2028"),
            "line separators must be escaped for JS"
        );
        assert!(!js.contains('\u{2028}'));
    }
}
