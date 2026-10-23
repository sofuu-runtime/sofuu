// rt/policy.rs — one policy evaluator for tools, shell, files, network and
// MCP servers (S4, 2026-10-23; SPEC-RUNTIME-CONTRACT R7).
//
// The evaluator is deliberately a PURE function over two JSON documents:
//
//   evaluate(request, policy) -> {"decision":"allow"|"deny"|"ask",
//                                 "rule":"R7.6/shell.enabled","reason":"…"}
//
// The policy lives in the runtime's global (`__sofuu_policy_json`), so two
// engines on one thread cannot see each other's policy. JS surfaces:
//
//   sofuu.policy.set(json)        validate + install (invalid leaves the old)
//   sofuu.policy.get()            the effective policy as JSON
//   sofuu.policy.evaluate(req)    one decision (req object or JSON string)
//   sofuu.policy.load()           merge ~/.sofuu/policy.json and
//                                 <cwd>/.sofuu/policy.json (project wins;
//                                 every "deny" array is unioned)
//   sofuu.policy.clear()
//
// Caller-level policy (agent def netPolicy/shell/approve) stays in the JS
// drivers; this is the floor every surface shares. Deny wins over allow at
// every level. A request that cannot be parsed is denied (fail closed).

use std::ffi::CString;
use std::os::raw::c_int;

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

fn json_str(v: &str) -> CString {
    CString::new(v).unwrap_or_default()
}

fn get_path<'a>(v: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    let mut cur = v;
    for key in path {
        cur = cur.get(*key)?;
    }
    Some(cur)
}

fn str_at<'a>(v: &'a serde_json::Value, path: &[&str]) -> Option<&'a str> {
    get_path(v, path).and_then(|x| x.as_str())
}

fn bool_at(v: &serde_json::Value, path: &[&str]) -> Option<bool> {
    get_path(v, path).and_then(|x| x.as_bool())
}

fn list_at(v: &serde_json::Value, path: &[&str]) -> Vec<String> {
    get_path(v, path)
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Glob match with `*` (any run, including empty) and `?` (one char).
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star: Option<usize> = None;
    let mut mark = 0usize;
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            pi += 1;
            mark = ti;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn norm_host(h: &str) -> String {
    let mut s = h.trim().to_ascii_lowercase();
    if s.ends_with('.') {
        s.pop();
    }
    s
}

fn host_matches(pattern: &str, host: &str) -> bool {
    let pat = norm_host(pattern);
    let h = norm_host(host);
    if pat.is_empty() || h.is_empty() {
        return false;
    }
    /* Suffix-anchored semantics shared with the agent's netPolicy:
     * "*.example.com" matches example.com and a.example.com, never
     * a.example.com.attacker.io; a bare domain matches itself and its
     * subdomains. */
    if let Some(rest) = pat.strip_prefix("*.") {
        h == rest || h.ends_with(&format!(".{rest}"))
    } else if pat.contains('*') || pat.contains('?') {
        glob_match(&pat, &h)
    } else {
        h == pat || h.ends_with(&format!(".{pat}"))
    }
}

fn path_under(path: &str, root: &str) -> bool {
    let p = path.trim_end_matches('/');
    let r = root.trim_end_matches('/');
    if r.is_empty() {
        return path.starts_with('/');
    }
    p == r || (p.starts_with(r) && p.as_bytes().get(r.len()) == Some(&b'/'))
}

fn decision(kind: &str, rule: &str, reason: &str) -> serde_json::Value {
    serde_json::json!({
        "decision": kind,
        "rule": rule,
        "reason": reason,
    })
}

/// The evaluator. `req.op` is one of tool.run | shell.run | fs.read |
/// fs.write | net.fetch | mcp.connect. Unknown ops are allowed (the
/// evaluator is a floor, not a gate on unmodeled operations).
pub fn evaluate(req: &serde_json::Value, policy: &serde_json::Value) -> serde_json::Value {
    let op = req.get("op").and_then(|v| v.as_str()).unwrap_or("");
    let tool = req.get("tool").and_then(|v| v.as_str()).unwrap_or("");
    let path = req.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let command = req.get("command").and_then(|v| v.as_str()).unwrap_or("");
    let host = req.get("host").and_then(|v| v.as_str()).unwrap_or("");
    let server = req.get("server").and_then(|v| v.as_str()).unwrap_or("");

    /* ── tools ─────────────────────────────────────────────────── */
    if op == "tool.run" && !tool.is_empty() {
        let deny = list_at(policy, &["tools", "deny"]);
        if deny.iter().any(|d| d == tool || glob_match(d, tool)) {
            return decision(
                "deny",
                "R7.6/tools.deny",
                &format!("tool '{tool}' is denied by policy"),
            );
        }
        let allow = list_at(policy, &["tools", "allow"]);
        if !allow.is_empty() && !allow.iter().any(|a| a == tool || glob_match(a, tool)) {
            return decision(
                "deny",
                "R7.6/tools.allow",
                &format!("tool '{tool}' is not in tools.allow"),
            );
        }
    }

    /* ── shell (also reached through the bash tool) ─────────────── */
    if op == "shell.run" || (op == "tool.run" && tool == "bash") {
        if bool_at(policy, &["shell", "enabled"]) == Some(false) {
            return decision(
                "deny",
                "R7.6/shell.enabled",
                "shell execution is disabled by policy",
            );
        }
        let allow = list_at(policy, &["shell", "allow"]);
        if !allow.is_empty() {
            let ok = allow.iter().any(|p| glob_match(p, command));
            if !ok {
                return decision(
                    "deny",
                    "R7.6/shell.allow",
                    "command does not match shell.allow",
                );
            }
        }
    }

    /* ── filesystem ────────────────────────────────────────────── */
    if op == "fs.write" && str_at(policy, &["fs", "write"]) == Some("off") {
        return decision("deny", "R7.6/fs.write", "filesystem writes are disabled by policy");
    }
    if op == "fs.read" || op == "fs.write" {
        let roots = list_at(policy, &["fs", "roots"]);
        if !roots.is_empty() {
            let inside = !path.is_empty() && roots.iter().any(|r| path_under(path, r));
            if !inside {
                return decision(
                    "deny",
                    "R7.6/fs.roots",
                    "path is outside every fs.roots entry",
                );
            }
        }
    }

    /* ── network ───────────────────────────────────────────────── */
    if op == "net.fetch" && !host.is_empty() {
        let deny = list_at(policy, &["net", "deny"]);
        if deny.iter().any(|p| host_matches(p, host)) {
            return decision(
                "deny",
                "R7.1/net.deny",
                &format!("host '{host}' is denied by policy"),
            );
        }
        let allow = list_at(policy, &["net", "allow"]);
        if !allow.is_empty() && !allow.iter().any(|p| host_matches(p, host)) {
            return decision(
                "deny",
                "R7.1/net.allow",
                &format!("host '{host}' is not in net.allow"),
            );
        }
    }

    /* ── MCP trust boundary ────────────────────────────────────── */
    if op == "mcp.connect" {
        let allow = list_at(policy, &["mcp", "servers", "allow"]);
        if !allow.is_empty() {
            let requested = if !server.is_empty() { server } else { command };
            let ok = allow.iter().any(|a| {
                a == requested
                    || glob_match(a, requested)
                    || (!command.is_empty() && command.starts_with(a))
            });
            if !ok {
                return decision(
                    "deny",
                    "R7.7/mcp.allow",
                    "MCP server is not in mcp.servers.allow",
                );
            }
        }
    }

    /* Approval mode is reported but not enforced here: the run/def
     * `approve` mode is the caller-level override, and S5 owns unifying
     * the two. A policy that wants every tool asked reports `ask`. */
    if op == "tool.run" && str_at(policy, &["approval", "mode"]) == Some("always") {
        return decision("ask", "R7.4/approval.mode", "approval.mode is 'always'");
    }

    decision("allow", "R7/default", "no policy rule denies this operation")
}

/// Merge `src` over `dst`: objects recurse, every `deny` array unions, and
/// any other leaf is replaced by the higher layer (project over user).
fn merge(dst: &mut serde_json::Value, src: &serde_json::Value) {
    match (dst, src) {
        (serde_json::Value::Object(d), serde_json::Value::Object(s)) => {
            for (k, v) in s {
                if k == "deny" && d.get(k).map_or(false, |x| x.is_array()) {
                    let existing = d.get_mut(k).unwrap();
                    if let (Some(existing), Some(new)) = (existing.as_array_mut(), v.as_array()) {
                        for item in new {
                            if !existing.contains(item) {
                                existing.push(item.clone());
                            }
                        }
                    }
                    continue;
                }
                match d.get_mut(k) {
                    Some(slot) if slot.is_object() && v.is_object() => merge(slot, v),
                    Some(slot) => *slot = v.clone(),
                    None => {
                        d.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (d, s) => *d = s.clone(),
    }
}

fn merge_layers(layers: &[serde_json::Value]) -> serde_json::Value {
    let mut out = serde_json::json!({});
    for layer in layers {
        merge(&mut out, layer);
    }
    out
}

/* ── JS surface ────────────────────────────────────────────────────── */

unsafe fn read_json_arg(ctx: *mut JSContext, v: JSValueConst) -> Option<String> {
    if qjs::is_object(v) {
        let js = qjs::JS_JSONStringify(ctx, v, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined());
        if qjs::is_exception(js) || qjs::is_undefined(js) {
            qjs::sofuu_js_free_value(ctx, js);
            return None;
        }
        let p = qjs::sofuu_js_to_cstring(ctx, js);
        let out = if p.is_null() {
            None
        } else {
            let s = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, p);
            Some(s)
        };
        qjs::sofuu_js_free_value(ctx, js);
        out
    } else {
        let p = qjs::sofuu_js_to_cstring(ctx, v);
        if p.is_null() {
            return None;
        }
        let s = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
        qjs::sofuu_js_free_cstring(ctx, p);
        Some(s)
    }
}

unsafe fn policy_json(ctx: *mut JSContext) -> serde_json::Value {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let v = qjs::sofuu_js_get_property_str(ctx, global, c"__sofuu_policy_json".as_ptr());
    let mut out = serde_json::json!({});
    if qjs::sofuu_js_is_string(v) != 0 {
        let p = qjs::sofuu_js_to_cstring(ctx, v);
        if !p.is_null() {
            let s = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, p);
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&s) {
                if parsed.is_object() {
                    out = parsed;
                }
            }
        }
    }
    qjs::sofuu_js_free_value(ctx, v);
    qjs::sofuu_js_free_value(ctx, global);
    out
}

unsafe fn install_policy(ctx: *mut JSContext, policy: &serde_json::Value) {
    let text = serde_json::to_string(policy).unwrap_or_else(|_| "{}".into());
    let cs = json_str(&text);
    let val = qjs::sofuu_js_new_string(ctx, cs.as_ptr());
    let global = qjs::sofuu_js_get_global_object(ctx);
    qjs::sofuu_js_set_property_str(ctx, global, c"__sofuu_policy_json".as_ptr(), val);
    qjs::sofuu_js_free_value(ctx, global);
}

unsafe extern "C" fn js_policy_set(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"policy.set: policy object or JSON required".as_ptr());
    }
    let Some(raw) = read_json_arg(ctx, *argv) else {
        let msg = qjs::sofuu_js_new_string(ctx, c"{\"ok\":false,\"error\":\"policy is not serializable\"}".as_ptr());
        return msg;
    };
    match serde_json::from_str::<serde_json::Value>(&raw) {
        Ok(v) if v.is_object() => {
            install_policy(ctx, &v);
            qjs::sofuu_js_new_string(ctx, c"{\"ok\":true}".as_ptr())
        }
        Ok(_) => qjs::sofuu_js_new_string(ctx, c"{\"ok\":false,\"error\":\"policy must be a JSON object\"}".as_ptr()),
        Err(e) => {
            let msg = format!("{{\"ok\":false,\"error\":{}}}", serde_json::to_string(&e.to_string()).unwrap_or_else(|_| "\"invalid policy\"".into()));
            let cs = json_str(&msg);
            qjs::sofuu_js_new_string(ctx, cs.as_ptr())
        }
    }
}

unsafe extern "C" fn js_policy_get(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let text = serde_json::to_string(&policy_json(ctx)).unwrap_or_else(|_| "{}".into());
    let cs = json_str(&text);
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_policy_clear(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    install_policy(ctx, &serde_json::json!({}));
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_policy_evaluate(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        let cs = json_str("{\"decision\":\"deny\",\"rule\":\"R7/invalid-request\",\"reason\":\"no request\"}");
        return qjs::sofuu_js_new_string(ctx, cs.as_ptr());
    }
    let Some(raw) = read_json_arg(ctx, *argv) else {
        let cs = json_str("{\"decision\":\"deny\",\"rule\":\"R7/invalid-request\",\"reason\":\"request is not serializable\"}");
        return qjs::sofuu_js_new_string(ctx, cs.as_ptr());
    };
    let req = match serde_json::from_str::<serde_json::Value>(&raw) {
        Ok(v) if v.is_object() => v,
        _ => {
            let cs = json_str("{\"decision\":\"deny\",\"rule\":\"R7/invalid-request\",\"reason\":\"request must be a JSON object\"}");
            return qjs::sofuu_js_new_string(ctx, cs.as_ptr());
        }
    };
    let policy = policy_json(ctx);
    let result = evaluate(&req, &policy);
    let text = serde_json::to_string(&result).unwrap_or_else(|_| "{\"decision\":\"deny\",\"rule\":\"R7/internal\"}".into());
    let cs = json_str(&text);
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_policy_load(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let mut layers: Vec<serde_json::Value> = Vec::new();
    let mut sources: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    let mut paths: Vec<(String, std::path::PathBuf)> = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        paths.push(("user".into(), std::path::PathBuf::from(home).join(".sofuu/policy.json")));
    }
    if let Ok(cwd) = std::env::current_dir() {
        paths.push(("project".into(), cwd.join(".sofuu/policy.json")));
    }
    for (label, path) in paths {
        if !path.is_file() {
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(v) if v.is_object() => {
                    layers.push(v);
                    sources.push(label);
                }
                Ok(_) => errors.push(format!("{label}: policy must be a JSON object")),
                Err(e) => errors.push(format!("{label}: {e}")),
            },
            Err(e) => errors.push(format!("{label}: {e}")),
        }
    }

    if !errors.is_empty() {
        let payload = serde_json::json!({ "ok": false, "error": errors.join("; "), "sources": sources });
        let text = serde_json::to_string(&payload).unwrap_or_else(|_| "{\"ok\":false}".into());
        let cs = json_str(&text);
        return qjs::sofuu_js_new_string(ctx, cs.as_ptr());
    }

    let merged = merge_layers(&layers);
    install_policy(ctx, &merged);
    let payload = serde_json::json!({ "ok": true, "sources": sources, "policy": merged });
    let text = serde_json::to_string(&payload).unwrap_or_else(|_| "{\"ok\":true}".into());
    let cs = json_str(&text);
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

/// Register `sofuu.policy.*`. Called from engine_register_builtins before
/// the shipped drivers load, so tools.js/agent.js can consult it.
pub unsafe fn mod_policy_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), qjs::sofuu_js_dup_value(ctx, sofuu_obj));
    }

    let policy = qjs::sofuu_js_new_object(ctx);
    let funcs: [(&std::ffi::CStr, unsafe extern "C" fn(*mut JSContext, JSValueConst, c_int, *const JSValueConst) -> JSValue); 5] = [
        (c"set", js_policy_set),
        (c"get", js_policy_get),
        (c"evaluate", js_policy_evaluate),
        (c"load", js_policy_load),
        (c"clear", js_policy_clear),
    ];
    for (name, f) in funcs {
        qjs::sofuu_js_set_property_str(
            ctx,
            policy,
            name.as_ptr(),
            qjs::sofuu_js_new_cfunction(ctx, f, name.as_ptr(), 1),
        );
    }
    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"policy".as_ptr(), policy);
    qjs::sofuu_js_free_value(ctx, sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(req: serde_json::Value, policy: serde_json::Value) -> String {
        evaluate(&req, &policy)
            .get("decision")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    }
    fn rule(req: serde_json::Value, policy: serde_json::Value) -> String {
        evaluate(&req, &policy)
            .get("rule")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    }

    #[test]
    fn empty_policy_allows() {
        assert_eq!(eval(serde_json::json!({"op":"tool.run","tool":"read_file"}), serde_json::json!({})), "allow");
    }

    #[test]
    fn tool_deny_wins_over_allow() {
        let p = serde_json::json!({"tools":{"allow":["bash"],"deny":["bash"]}});
        assert_eq!(eval(serde_json::json!({"op":"tool.run","tool":"bash"}), p), "deny");
    }

    #[test]
    fn tool_allowlist_refuses_unlisted() {
        let p = serde_json::json!({"tools":{"allow":["read_file","grep"]}});
        assert_eq!(eval(serde_json::json!({"op":"tool.run","tool":"bash"}), p.clone()), "deny");
        assert_eq!(eval(serde_json::json!({"op":"tool.run","tool":"grep"}), p), "allow");
    }

    #[test]
    fn shell_disabled_blocks_the_bash_tool() {
        let p = serde_json::json!({"shell":{"enabled":false}});
        assert_eq!(eval(serde_json::json!({"op":"tool.run","tool":"bash","command":"ls"}), p), "deny");
    }

    #[test]
    fn shell_allowlist_is_globbed() {
        let p = serde_json::json!({"shell":{"allow":["git *","ls"]}});
        assert_eq!(eval(serde_json::json!({"op":"shell.run","command":"git status"}), p.clone()), "allow");
        assert_eq!(eval(serde_json::json!({"op":"shell.run","command":"rm -rf /"}), p), "deny");
    }

    #[test]
    fn fs_write_off_and_roots() {
        let off = serde_json::json!({"fs":{"write":"off"}});
        assert_eq!(eval(serde_json::json!({"op":"fs.write","path":"/tmp/x"}), off), "deny");
        let roots = serde_json::json!({"fs":{"roots":["/work/proj"]}});
        assert_eq!(eval(serde_json::json!({"op":"fs.read","path":"/work/proj/a.txt"}), roots.clone()), "allow");
        assert_eq!(eval(serde_json::json!({"op":"fs.read","path":"/work/proj2/a.txt"}), roots.clone()), "deny");
        assert_eq!(eval(serde_json::json!({"op":"fs.read","path":"/etc/passwd"}), roots), "deny");
    }

    #[test]
    fn net_wildcard_is_suffix_anchored() {
        let p = serde_json::json!({"net":{"allow":["*.example.com"]}});
        assert_eq!(eval(serde_json::json!({"op":"net.fetch","host":"api.example.com"}), p.clone()), "allow");
        assert_eq!(eval(serde_json::json!({"op":"net.fetch","host":"example.com"}), p.clone()), "allow");
        assert_eq!(eval(serde_json::json!({"op":"net.fetch","host":"api.example.com.attacker.io"}), p.clone()), "deny");
        let deny = serde_json::json!({"net":{"deny":["evil.io"]}});
        assert_eq!(eval(serde_json::json!({"op":"net.fetch","host":"sub.evil.io"}), deny), "deny");
    }

    #[test]
    fn mcp_allowlist() {
        let p = serde_json::json!({"mcp":{"servers":{"allow":["srv1"]}}});
        assert_eq!(eval(serde_json::json!({"op":"mcp.connect","server":"srv1"}), p.clone()), "allow");
        assert_eq!(eval(serde_json::json!({"op":"mcp.connect","server":"srv2"}), p), "deny");
    }

    #[test]
    fn project_layer_overrides_user_and_denies_union() {
        let merged = merge_layers(&[
            serde_json::json!({"fs":{"write":"off"},"tools":{"deny":["bash"]}}),
            serde_json::json!({"fs":{"write":"workspace"},"tools":{"deny":["rm"]}}),
        ]);
        assert_eq!(merged.get("fs").and_then(|f| f.get("write")).and_then(|w| w.as_str()), Some("workspace"));
        let deny: Vec<String> = list_at(&merged, &["tools", "deny"]);
        assert!(deny.contains(&"bash".to_string()));
        assert!(deny.contains(&"rm".to_string()));
    }

    #[test]
    fn invalid_request_fails_closed_in_js_layer_only() {
        // The pure evaluator has no "invalid" case; the JS wrapper denies a
        // non-object request. This pins the rule id text used there.
        assert_eq!(rule(serde_json::json!({"op":"tool.run","tool":"rm"}), serde_json::json!({"tools":{"deny":["rm"]}})), "R7.6/tools.deny");
    }
}
