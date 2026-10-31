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
use std::net::{IpAddr, Ipv4Addr};
use std::os::raw::c_int;
use std::path::{Path, PathBuf};

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

/* ── schema check (AUDIT-2026-10-31 P1-15) ─────────────────────────────
 * The R7 contract (SECURITY.md / docs/SPEC-RUNTIME-CONTRACT.md §8) for policy
 * JSON: "Policy JSON
 * | --policy, .sofuu/policy.json, sofuu.policy.set | schema-checked; invalid
 * leaves the previous policy; fail closed". Only the "is it a JSON object"
 * half existed, so a typo installed cleanly and then allowed everything it
 * was supposed to refuse: {"net":{"denny":["x"]}} (misspelt key),
 * {"net":{"deny":"x"}} (a string where a list belongs) and
 * {"shell":{"enabled":"false"}} (a string that is not Some(false)) all read
 * as "no rule". The checker below refuses those, and the caller keeps the
 * policy it already had. */

#[derive(Clone, Copy)]
enum Ty {
    Bool,
    Num,
    Str,
    /// A list of strings. `single` also accepts one bare string, which the
    /// filesystem resolver reads as a one-entry list (fs.roots).
    StrList { single: bool },
    /// fs.read / fs.write: a mode string, a boolean, or an inline allowlist.
    /// Every spelling below reaches the ONE resolver in `fs_mode_and_roots`,
    /// so nothing here can widen a box.
    FsMode,
    /// One of a fixed set; compared case-insensitively.
    OneOf(&'static [&'static str]),
    /// A nested section of the shape { <key>: Ty }.
    Section(&'static [(&'static str, Ty)]),
}

const FS_MODES: &[&str] = &[
    "", "any", "all", "on", "off", "none", "deny", "workspace", "allowlist", "allow-list",
    "explicit", "roots",
];

const POLICY_SCHEMA: &[(&str, Ty)] = &[
    (
        "tools",
        Ty::Section(&[
            ("allow", Ty::StrList { single: false }),
            ("deny", Ty::StrList { single: false }),
        ]),
    ),
    (
        "shell",
        Ty::Section(&[
            ("enabled", Ty::Bool),
            ("allow", Ty::StrList { single: false }),
            ("deny", Ty::StrList { single: false }),
            ("spawn_strict", Ty::Bool),
            ("spawnStrict", Ty::Bool),
        ]),
    ),
    (
        "fs",
        Ty::Section(&[
            ("write", Ty::FsMode),
            ("read", Ty::FsMode),
            ("roots", Ty::StrList { single: true }),
        ]),
    ),
    (
        "net",
        Ty::Section(&[
            ("enabled", Ty::Bool),
            ("allow", Ty::StrList { single: false }),
            ("deny", Ty::StrList { single: false }),
            ("ssrf", Ty::Bool),
            ("allow_private", Ty::Bool),
            ("allowPrivate", Ty::Bool),
        ]),
    ),
    (
        "http",
        Ty::Section(&[
            ("deny", Ty::StrList { single: false }),
            ("auth", Ty::OneOf(&["none", "bearer"])),
        ]),
    ),
    (
        "mcp",
        Ty::Section(&[
            ("servers", Ty::Section(&[("allow", Ty::StrList { single: false })])),
            ("callTimeoutMs", Ty::Num),
        ]),
    ),
    (
        "approval",
        Ty::Section(&[
            ("mode", Ty::OneOf(&["never", "per-run", "always"])),
            ("cache", Ty::Bool),
        ]),
    ),
    (
        "secrets",
        Ty::Section(&[
            ("source", Ty::OneOf(&["env", "file", "keychain", "host"])),
            ("file", Ty::Str),
        ]),
    ),
    ("memory", Ty::Section(&[("write", Ty::OneOf(&["on", "off"]))])),
];

fn describe(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

fn want_text(ty: Ty) -> String {
    match ty {
        Ty::Bool => "a boolean".into(),
        Ty::Num => "a number".into(),
        Ty::Str => "a string".into(),
        Ty::StrList { single: true } => "an array of strings (or one string)".into(),
        Ty::StrList { single: false } => "an array of strings".into(),
        Ty::FsMode => "off|workspace|allowlist|any, a boolean, or an array of roots".into(),
        Ty::OneOf(vals) => format!("one of {}", vals.join("|")),
        Ty::Section(_) => "an object".into(),
    }
}

fn check_ty(path: &str, ty: Ty, v: &serde_json::Value, errs: &mut Vec<String>) {
    let ok = match ty {
        Ty::Bool => v.is_boolean(),
        Ty::Num => v.is_number(),
        Ty::Str => v.is_string(),
        Ty::StrList { single } => {
            if v.is_array() {
                v.as_array().map(|a| a.iter().all(|x| x.is_string())).unwrap_or(false)
            } else {
                single && v.is_string()
            }
        }
        Ty::FsMode => {
            if v.is_array() {
                v.as_array().map(|a| a.iter().all(|x| x.is_string())).unwrap_or(false)
            } else if let Some(s) = v.as_str() {
                FS_MODES.contains(&s.trim().to_ascii_lowercase().as_str())
            } else {
                v.is_boolean()
            }
        }
        Ty::OneOf(vals) => v
            .as_str()
            .map(|s| vals.contains(&s.trim().to_ascii_lowercase().as_str()))
            .unwrap_or(false),
        Ty::Section(schema) => {
            if let Some(obj) = v.as_object() {
                for (k, inner) in schema {
                    if let Some(child) = obj.get(*k) {
                        check_ty(&format!("{path}.{k}"), *inner, child, errs);
                    }
                }
                for k in obj.keys() {
                    if !schema.iter().any(|(known, _)| known == k) {
                        errs.push(format!(
                            "{path}.{k}: unknown key (known: {})",
                            schema.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
                        ));
                    }
                }
                true
            } else {
                false
            }
        }
    };
    if !ok {
        errs.push(format!(
            "{path}: expected {}, got {}",
            want_text(ty),
            describe(v)
        ));
    }
}

/// Refuse a policy that is not the documented R7 shape. An unknown section or
/// key is an error, not a no-op — that is the whole point of the check.
fn validate_policy(v: &serde_json::Value) -> Result<(), String> {
    let Some(obj) = v.as_object() else {
        return Err("policy must be a JSON object".into());
    };
    let mut errs: Vec<String> = Vec::new();
    for (k, ty) in POLICY_SCHEMA {
        if let Some(child) = obj.get(*k) {
            check_ty(k, *ty, child, &mut errs);
        }
    }
    for k in obj.keys() {
        /* A leading underscore is the file's own comment channel: the
         * `sofuu policy init` profiles ship `_notes` so a human reading
         * .sofuu/policy.json sees what each line is for. Nothing consumes it,
         * so it is allowed and not checked. */
        if k.starts_with('_') {
            continue;
        }
        if !POLICY_SCHEMA.iter().any(|(known, _)| known == k) {
            errs.push(format!(
                "{k}: unknown policy section (known: {})",
                POLICY_SCHEMA.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
            ));
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs.join("; "))
    }
}

/// The fs.<key> mode and the roots it is bounded by — ONE resolver, shared by
/// the pure evaluator and `gate_fs`, so an introspection answer and an actual
/// read/write can never disagree (AUDIT-2026-10-31 P1-6: `evaluate` honoured
/// only the literal string "off", while the gate also read `false`, `true`,
/// inline arrays and unknown modes).
///
/// `workspace` adds the process cwd to the roots (the gate confines to it), so
/// the evaluator has to see it too.
fn fs_mode_and_roots(policy: &serde_json::Value, key: &str) -> (String, Vec<String>) {
    let mut roots = list_at(policy, &["fs", "roots"]);
    /* A scalar root ("fs":{"roots":"/x"}) is a plausible spelling of the same
     * intent; reading it as "no roots" would silently disable the box. */
    if roots.is_empty() {
        if let Some(single) = str_at(policy, &["fs", "roots"]) {
            if !single.trim().is_empty() {
                roots.push(single.trim().to_string());
            }
        }
    }
    let (mode, inline) = match get_path(policy, &["fs", key]) {
        /* `fs.write: false` is the shortest way to write "off", and a boolean
         * must never be read as "unset" (that would widen the policy). */
        Some(serde_json::Value::Bool(false)) => ("off".to_string(), Vec::new()),
        Some(serde_json::Value::Bool(true)) => ("any".to_string(), Vec::new()),
        Some(serde_json::Value::String(s)) => (s.trim().to_ascii_lowercase(), Vec::new()),
        Some(serde_json::Value::Array(a)) => (
            "allowlist".to_string(),
            a.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect::<Vec<String>>(),
        ),
        _ => (
            if roots.is_empty() { "any".to_string() } else { "allowlist".to_string() },
            Vec::new(),
        ),
    };
    for r in inline {
        if !roots.contains(&r) {
            roots.push(r);
        }
    }
    (mode, roots)
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

/// Normalize a host/authority before any rule sees it — the SAME
/// normalization the evaluator applies, exported so the fetch path asks the
/// policy about the host it will actually dial (`http_client`).
pub(crate) fn norm_host(h: &str) -> String {
    // R7.1: `:port`, `user@`, a scheme:// prefix, a path suffix, and an
    // RFC-3986 [v6] literal all normalize BEFORE the suffix match — the
    // agent's JS netPolicy already did this (hostOf), so the evaluator must
    // not answer differently on a URL-shaped host.
    let mut s = h.trim().to_ascii_lowercase();
    if let Some(i) = s.find("://") {
        s = s[i + 3..].to_string();
    }
    if let Some(i) = s.find('/') {
        s.truncate(i);
    }
    if let Some(i) = s.rfind('@') {
        s = s[i + 1..].to_string();
    }
    if s.starts_with('[') {
        // [v6] literal: everything up to ']' is the host (a trailing :port
        // after the bracket is cut with it).
        if let Some(end) = s.find(']') {
            s = s[1..end].to_string();
        }
    } else if let Some(colon) = s.rfind(':') {
        // Bare-host name/port: strip the port only when the tail is digits
        // (a name may itself contain no ':', so this is safe either way).
        let tail = &s[colon + 1..];
        if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) {
            s.truncate(colon);
        }
    }
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
    /* Set when a rule that can only be judged per call was skipped because
     * the caller asked a tool-level question (an inventory row, H-01). */
    let mut conditional = false;

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
        /* R7.6/shell.deny (R7.P3): the default profile's destructive-command
         * list. Full-command globs; allow is consulted only after deny, so a
         * deny cannot be re-enabled by a broader allow. Best-effort by
         * design — a pattern matches the command STRING, not a parsed shell,
         * which is why shell.enabled is the real switch. */
        let deny = list_at(policy, &["shell", "deny"]);
        if !deny.is_empty() {
            let inventory = op == "tool.run" && command.trim().is_empty();
            if !inventory && deny.iter().any(|p| glob_match(p, command)) {
                return decision(
                    "deny",
                    "R7.6/shell.deny",
                    &format!("command matches a shell.deny pattern: {command}"),
                );
            }
            if inventory {
                conditional = true;
            }
        }
        let allow = list_at(policy, &["shell", "allow"]);
        if !allow.is_empty() {
            /* `tool.run` for bash with NO command is the inventory question
             * sofuu.tools.list() asks (H-01: the listing and the executor
             * must consult one evaluator and never disagree). A command-
             * scoped rule cannot be answered for a command the caller never
             * supplied, so the row is reported conditional instead of denying
             * a command that may never run. shell.run — and every real
             * execution — always carries the command, so the rule is never
             * skipped on a call. */
            let inventory = op == "tool.run" && command.trim().is_empty();
            let ok = inventory || allow.iter().any(|p| glob_match(p, command));
            if !ok {
                return decision(
                    "deny",
                    "R7.6/shell.allow",
                    "command does not match shell.allow",
                );
            }
            if inventory {
                conditional = true;
            }
        }
    }

    /* ── filesystem ────────────────────────────────────────────── */
    /* AUDIT-2026-10-31 P1-6: the mode comes from the SAME resolver the
     * filesystem gate uses, so `evaluate` cannot answer "allow" for a
     * `fs.write: false` (or an inline allowlist, or a misspelt mode) that
     * `gate_fs` refuses. */
    if op == "fs.read" || op == "fs.write" {
        let key = if op == "fs.write" { "write" } else { "read" };
        let rule = if op == "fs.write" { "R7.6/fs.write" } else { "R7.6/fs.read" };
        let (mode, mut roots) = fs_mode_and_roots(policy, key);
        match mode.as_str() {
            "" | "any" | "all" | "on" => {}
            "off" | "none" | "deny" => {
                return decision(
                    "deny",
                    rule,
                    &format!("filesystem {key}s are disabled by policy"),
                )
            }
            "workspace" => {
                /* The gate confines to fs.roots OR the process cwd. */
                if let Ok(cwd) = std::env::current_dir() {
                    let c = std::fs::canonicalize(&cwd).unwrap_or(cwd);
                    let c = c.to_string_lossy().to_string();
                    if !roots.contains(&c) {
                        roots.push(c);
                    }
                }
            }
            "allowlist" | "allow-list" | "explicit" | "roots" => {}
            /* Fail closed: a typo in a security setting must never widen it
             * (the same reading `gate_fs` takes). */
            other => {
                return decision(
                    "deny",
                    rule,
                    &format!(
                        "unknown fs.{key} mode '{other}' (expected off|workspace|allowlist|any)"
                    ),
                )
            }
        }
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

    /* ── inbound HTTP (serve routes; H1) ───────────────────────── */
    if op == "http.request" && !path.is_empty() {
        let deny = list_at(policy, &["http", "deny"]);
        if deny.iter().any(|p| path.starts_with(p.as_str())) {
            return decision(
                "deny",
                "R7.6/http.deny",
                &format!("route '{path}' is denied by policy"),
            );
        }
    }

    /* ── network ───────────────────────────────────────────────── */
    /* R7.1/net.enabled (R7.P3): the `--no-net` switch and the offline
     * profile. Checked before allow/deny so `net.allow` cannot widen it —
     * a policy that says "no network" means no network. */
    if op == "net.fetch" && bool_at(policy, &["net", "enabled"]) == Some(false) {
        return decision(
            "deny",
            "R7.1/net.enabled",
            "outbound network is disabled by policy",
        );
    }
    /* AUDIT-2026-10-31 P1-6: the two rules below decide by host, so a
     * net.fetch request that carries NO host used to fall through to
     * R7/default and answer "allow". The audit's
     * `{op:'net.fetch', target:'example.com'}` (a field the evaluator never
     * reads) therefore reported allow while the real fetch was denied. When
     * the policy actually has network rules, an unanswerable request is
     * refused instead of waved through. */
    if op == "net.fetch" && host.is_empty() && has_net_rules(policy) {
        return decision(
            "deny",
            "R7.1/net.host-required",
            "net.fetch needs a 'host' to be judged against net.allow/net.deny; \
             the request supplied none, so it cannot be allowed",
        );
    }
    /* net.ssrf / net.allow_private refuse internal targets by RESOLVED
     * address at fetch time. The evaluator is pure, but an IP literal (or a
     * loopback alias) needs no resolver, and that is exactly the question
     * `{host:'169.254.169.254'}` asks. */
    if op == "net.fetch" && !host.is_empty() && net_ssrf(policy) {
        if let Some(ip) = parse_ip_literal(&host) {
            if ip_is_internal(ip) {
                return decision(
                    "deny",
                    "R7.1/net.ssrf",
                    &format!("host '{host}' is an internal address and net.ssrf is on"),
                );
            }
        }
    }
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

    /* Approval mode (S5, 2026-10-23): 'always' asks for every tool call;
     * 'never' runs without prompting (denies above still win);
     * 'per-run' is the default gate the drivers already implement. */
    if op == "tool.run" {
        match str_at(policy, &["approval", "mode"]) {
            Some("always") => {
                return decision("ask", "R7.4/approval.mode", "approval.mode is 'always'")
            }
            Some("never") => {
                return decision(
                    "allow",
                    "R7.4/approval.mode",
                    "approval.mode is 'never'",
                )
            }
            _ => {}
        }
    }

    let mut out = decision("allow", "R7/default", "no policy rule denies this operation");
    if conditional {
        if let Some(obj) = out.as_object_mut() {
            obj.insert("conditional".into(), serde_json::Value::Bool(true));
        }
    }
    out
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

/// The layered policy from disk — `~/.sofuu/policy.json`, then
/// `<cwd>/.sofuu/policy.json` (project wins; deny arrays union) — WITHOUT
/// installing it. The CLI audit needs the effective document without a live
/// JS context; `sofuu.policy.load()` is the runtime counterpart and uses the
/// same merge.
///
/// An invalid file is an ERROR, not a skipped layer: a policy the operator
/// thinks they wrote but that silently does not apply is worse than a hard
/// stop.
pub fn layered_policy_from_disk() -> Result<serde_json::Value, String> {
    let mut layers: Vec<serde_json::Value> = Vec::new();
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
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("{label} policy {}: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("{label} policy {}: not valid JSON: {e}", path.display()))?;
        if !v.is_object() {
            return Err(format!("{label} policy {}: must be a JSON object", path.display()));
        }
        /* AUDIT-2026-10-31 P1-15: the CLI audit reads the same files, so it
         * has to refuse the same typos the runtime refuses. */
        validate_policy(&v)
            .map_err(|e| format!("{label} policy {}: {e}", path.display()))?;
        layers.push(v);
    }
    Ok(merge_layers(&layers))
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

/// The effective policy document (per runtime, stored on the global).
pub(crate) unsafe fn policy_json(ctx: *mut JSContext) -> serde_json::Value {
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
        Ok(v) if v.is_object() => match validate_policy(&v) {
            Ok(()) => {
                install_policy(ctx, &v);
                qjs::sofuu_js_new_string(ctx, c"{\"ok\":true}".as_ptr())
            }
            /* AUDIT-2026-10-31 P1-15: a schema error is a refusal, not a
             * no-op — the previous policy stays installed and the caller is
             * told why (the R7 contract: "invalid leaves the previous
             * policy; fail closed"). */
            Err(e) => policy_error(ctx, &e),
        },
        Ok(_) => qjs::sofuu_js_new_string(ctx, c"{\"ok\":false,\"error\":\"policy must be a JSON object\"}".as_ptr()),
        Err(e) => policy_error(ctx, &e.to_string()),
    }
}

unsafe fn policy_error(ctx: *mut JSContext, msg: &str) -> JSValue {
    let payload = serde_json::json!({ "ok": false, "error": msg });
    let text = serde_json::to_string(&payload).unwrap_or_else(|_| "{\"ok\":false}".into());
    let cs = json_str(&text);
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
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

/// `policy.explain(request)` — the same decision as `evaluate`, plus a
/// one-line human explanation. For debugging denials without guessing which
/// rule fired.
unsafe extern "C" fn js_policy_explain(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let result = js_policy_evaluate(ctx, _this, argc, argv);
    if qjs::sofuu_js_is_string(result) == 0 {
        return result;
    }
    let p = qjs::sofuu_js_to_cstring(ctx, result);
    if p.is_null() {
        return result;
    }
    let raw = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, p);
    qjs::sofuu_js_free_value(ctx, result);
    let mut v: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|_| serde_json::json!({"decision":"deny","rule":"R7/invalid-request"}));
    let decision = v.get("decision").and_then(|x| x.as_str()).unwrap_or("deny");
    let rule = v.get("rule").and_then(|x| x.as_str()).unwrap_or("R7/unknown");
    let reason = v.get("reason").and_then(|x| x.as_str()).unwrap_or("");
    let explanation = format!("{decision} [{rule}] {reason}");
    if let Some(obj) = v.as_object_mut() {
        obj.insert(
            "explanation".into(),
            serde_json::Value::String(explanation),
        );
    }
    let text = serde_json::to_string(&v).unwrap_or_else(|_| "{}".into());
    let cs = json_str(&text);
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

/// `policy.secret(name)` — resolve a secret at CALL time (R7.5).
///
/// Sources (`policy.secrets.source`):
///   * `env`  (default) — the process environment
///   * `file` — a JSON object at `policy.secrets.file`, re-read per call so a
///              rotated file is picked up without restarting
///
/// The return value is a plain JS string the caller passes to the consumer
/// (e.g. an MCP child's env). Nothing here logs it, and configs carry only
/// the NAME, never the value.
unsafe extern "C" fn js_policy_secret(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"policy.secret: name required".as_ptr());
    }
    let p = qjs::sofuu_js_to_cstring(ctx, *argv);
    if p.is_null() {
        return qjs::sofuu_js_exception();
    }
    let name = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, p);
    if name.trim().is_empty() {
        return qjs::JS_ThrowTypeError(ctx, c"policy.secret: name is empty".as_ptr());
    }
    let policy = policy_json(ctx);
    let source = str_at(&policy, &["secrets", "source"]).unwrap_or("env");
    let value: Option<String> = match source {
        "env" => std::env::var(&name).ok(),
        "file" => {
            let path = str_at(&policy, &["secrets", "file"]).unwrap_or("");
            if path.is_empty() {
                None
            } else {
                std::fs::read_to_string(path)
                    .ok()
                    .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
                    .and_then(|v| {
                        v.get(&name)
                            .and_then(|x| x.as_str())
                            .map(|s| s.to_string())
                    })
            }
        }
        _ => None,
    };
    match value {
        Some(v) => {
            let cs = CString::new(v).unwrap_or_default();
            qjs::sofuu_js_new_string(ctx, cs.as_ptr())
        }
        None => {
            let msg = format!("policy.secret: '{name}' is not set (secrets.source={source})");
            let cm = json_str(&msg);
            qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), cm.as_ptr())
        }
    }
}

/// `fs.roots` from the active policy (used for MCP `roots/list`).
pub(crate) unsafe fn fs_roots(ctx: *mut JSContext) -> Vec<String> {
    list_at(&policy_json(ctx), &["fs", "roots"])
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
                Ok(v) if v.is_object() => match validate_policy(&v) {
                    Ok(()) => {
                        layers.push(v);
                        sources.push(label);
                    }
                    /* AUDIT-2026-10-31 P1-15: a malformed file is reported
                     * and NOTHING is installed, so a reload can never turn a
                     * working policy into a wider one. */
                    Err(e) => errors.push(format!("{label}: {e}")),
                },
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
    let funcs: [(&std::ffi::CStr, unsafe extern "C" fn(*mut JSContext, JSValueConst, c_int, *const JSValueConst) -> JSValue); 7] = [
        (c"set", js_policy_set),
        (c"get", js_policy_get),
        (c"evaluate", js_policy_evaluate),
        (c"explain", js_policy_explain),
        (c"secret", js_policy_secret),
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

/* ── Host-bridge gates (H-02, R-24; 2026-10-29) ─────────────────────
 *
 * The evaluator above answers a pure question about two JSON documents. The
 * host bridges — process_spawn, fs, http_client — additionally have to touch
 * the real world: the operator's env kill switch, the filesystem, and the
 * address libcurl is about to dial. This is that half of the contract, and it
 * lives here so every bridge asks the SAME code for the SAME policy.
 *
 * Every gate reads the policy through policy_json(ctx) — the runtime's own
 * global — and never from process-wide state. That is what keeps a harness
 * that installs its own policy, a second engine on the same thread, and a
 * child process that loaded its own --policy / .sofuu/policy.json from
 * leaking decisions into each other.
 */

/// Shell interpreters whose execution `shell.enabled: false` must stop.
///
/// The audit's H-02(a) bypass was `sofuu.spawn({command:'/bin/sh', args:['-c',
/// ...]})` under `shell.enabled: false`: the gate only bound tool-originated
/// spawns, so a host script could still run arbitrary shell. "Shell execution
/// is disabled" has to mean shell *interpreters*, whoever starts them, while
/// plain host plumbing (`curl`, `mkdir`, `sofuu serve`) stays available for
/// harnesses such as t72 that install the policy and then drive the binary.
pub(crate) fn program_is_shell(cmdline: &str) -> bool {
    let prog = cmdline.split_whitespace().next().unwrap_or("");
    let base = prog.rsplit(['/', '\\']).next().unwrap_or(prog);
    matches!(
        base,
        "sh" | "bash" | "zsh" | "dash" | "ksh" | "mksh" | "ash" | "csh" | "tcsh" | "fish" | "busybox"
    ) || base.starts_with("sh") && base.len() > 2 && base.ends_with("sh")
}

/// A typed refusal from a host bridge.
#[derive(Debug, Clone)]
pub(crate) struct PolicyDenied {
    /// R2-style typed prefix (`E_PERMISSION_DENIED`, `E_PATH_OUTSIDE`).
    pub code: &'static str,
    /// The R7 rule id that fired (R7.3: every refusal names its rule).
    pub rule: String,
    /// What was refused — always names the offending path/host/command.
    pub message: String,
}

impl PolicyDenied {
    fn new(code: &'static str, rule: &str, message: String) -> Self {
        Self { code, rule: rule.to_string(), message }
    }

    /// `CODE: message` — the typed shape the audit names. Used by the spawn
    /// bridge, whose refusal text is part of the contract
    /// (`E_PERMISSION_DENIED: Process execution disabled by policy`).
    pub(crate) fn text(&self) -> String {
        format!("{}: {}", self.code, self.message)
    }

    /// `CODE: message (rule)` — for bridges whose only channel to JS is a
    /// rejection string, so the rule id is never lost (R7.3).
    pub(crate) fn text_rule(&self) -> String {
        if self.rule.is_empty() {
            self.text()
        } else {
            format!("{}: {} ({})", self.code, self.message, self.rule)
        }
    }
}

/// Throw a policy refusal as a real `Error` carrying `.code` and `.rule`.
///
/// The bridges used `JS_ThrowTypeError(ctx, "%s", text)`, so a caller could
/// only string-match `e.message`: the audit's "typed refusal" requirement
/// (R2/R7.3) was satisfied in the text and not in the value. Every bridge now
/// throws through here, so `catch (e) { if (e.code === 'E_PATH_OUTSIDE') ... }`
/// works and `e.rule` still names the R7 rule that fired.
///
/// # Safety
/// `ctx` must be live and on the runtime thread.
pub(crate) unsafe fn throw_denied(
    ctx: *mut JSContext,
    denied: &PolicyDenied,
) -> qjs::JSValue {
    use std::ffi::CString;
    let m = CString::new(denied.text_rule()).unwrap_or_default();
    /* Build the exception first (TypeError gives a stack), then stamp the
     * machine-readable fields on it and re-throw the same object. */
    let _ = qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), m.as_ptr());
    let exc = qjs::JS_GetException(ctx);
    if !qjs::is_exception(exc) {
        if let Ok(c) = CString::new(denied.code) {
            let v = qjs::JS_NewString(ctx, c.as_ptr());
            qjs::JS_SetPropertyStr(ctx, exc, c"code".as_ptr(), v);
        }
        if let Ok(r) = CString::new(denied.rule.as_str()) {
            let v = qjs::JS_NewString(ctx, r.as_ptr());
            qjs::JS_SetPropertyStr(ctx, exc, c"rule".as_ptr(), v);
        }
    }
    qjs::JS_Throw(ctx, exc)
}

/// True when this runtime carries no policy at all. Every gate checks it
/// first: an unconfigured runtime pays one property read, not a JSON parse —
/// and the fetch path consequently never resolves a name for a policy that
/// does not exist.
pub(crate) unsafe fn policy_is_empty(ctx: *mut JSContext) -> bool {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let v = qjs::sofuu_js_get_property_str(ctx, global, c"__sofuu_policy_json".as_ptr());
    let mut empty = true;
    if qjs::sofuu_js_is_string(v) != 0 {
        let p = qjs::sofuu_js_to_cstring(ctx, v);
        if !p.is_null() {
            let s = std::ffi::CStr::from_ptr(p).to_string_lossy();
            let t = s.trim();
            empty = t.is_empty() || t == "{}" || t == "null";
            qjs::sofuu_js_free_cstring(ctx, p);
        }
    }
    qjs::sofuu_js_free_value(ctx, v);
    qjs::sofuu_js_free_value(ctx, global);
    empty
}

/// Fold `.`/`..` lexically without touching the filesystem (shared with
/// fs.rs's native-root resolver, so the two jail layers agree on what `..`
/// means).
pub(crate) fn lexical_clean(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Canonicalise `p` WITHOUT creating anything: `..` is folded lexically, the
/// deepest EXISTING ancestor is canonicalised (which resolves symlinks), and
/// the missing tail is re-attached. A write to a brand-new file is therefore
/// still judged by the real directory it will land in, and a symlink pointing
/// out of the roots is caught by the canonical prefix, not by its spelling.
fn canonical_nearest(p: &Path) -> Option<PathBuf> {
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(p)
    };
    let clean = lexical_clean(&abs);
    if let Ok(c) = std::fs::canonicalize(&clean) {
        return Some(c);
    }
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut probe = clean;
    /* Bounded: a path can only be this deep by accident. */
    for _ in 0..64 {
        let parent = probe.parent()?;
        let name = probe.file_name()?.to_os_string();
        tail.push(name);
        match std::fs::canonicalize(parent) {
            Ok(mut c) => {
                for part in tail.iter().rev() {
                    c.push(part);
                }
                return Some(c);
            }
            Err(_) => probe = parent.to_path_buf(),
        }
    }
    None
}

/// Which filesystem operation is asking. Reads and writes have separate
/// policy keys (`fs.read` / `fs.write`) and separate rule ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FsKind {
    Read,
    Write,
}

impl FsKind {
    fn key(self) -> &'static str {
        match self {
            FsKind::Read => "read",
            FsKind::Write => "write",
        }
    }
    fn rule(self) -> &'static str {
        match self {
            FsKind::Read => "R7.6/fs.read",
            FsKind::Write => "R7.6/fs.write",
        }
    }
    fn verb(self) -> &'static str {
        match self {
            FsKind::Read => "read",
            FsKind::Write => "write",
        }
    }
}

/// H-02(c) / R-24 — the one filesystem-confinement resolver.
///
/// Modes, per kind (`fs.read` / `fs.write`):
///   `off`        deny every operation of that kind
///   `workspace`  the target must canonicalise INSIDE `fs.roots` or the
///                process cwd (`..` that climbs out and symlinks that point
///                out are rejected by the canonical comparison)
///   `allowlist`  the target must canonicalise inside an explicit `fs.roots`
///                entry; cwd is NOT implied. An inline array
///                (`fs.write: ["/a", "/b"]`) is this mode with its own roots.
///   `any`        no confinement (the host's explicit choice)
///
/// An absent key means `allowlist` when `fs.roots` is configured — the policy
/// author drew a box, and the fs API must not ignore it — and `any`
/// otherwise. That last case is deliberate: the runtime's own drivers write
/// run logs, tool-result spills and plugin scaffolds outside the workspace,
/// so a *host with no fs policy* keeps its previous reach. A policy that says
/// `workspace` gets a workspace.
///
/// Returns the canonical path the caller should use, so "check then open"
/// cannot be aimed at a different file than the one that was vetted.
pub(crate) unsafe fn gate_fs(
    ctx: *mut JSContext,
    kind: FsKind,
    path: &str,
) -> Result<PathBuf, PolicyDenied> {
    let given = PathBuf::from(path);
    if policy_is_empty(ctx) {
        return Ok(given);
    }
    let policy = policy_json(ctx);
    /* AUDIT-2026-10-31 P1-6: the mode/roots resolution lives in ONE helper so
     * `sofuu.policy.evaluate` answers what this gate enforces. */
    let key = kind.key();
    let (mode, roots) = fs_mode_and_roots(&policy, key);

    let confine_to_cwd = match mode.as_str() {
        "" | "any" | "all" | "on" => return Ok(given),
        "off" | "none" | "deny" => {
            return Err(PolicyDenied::new(
                "E_PERMISSION_DENIED",
                kind.rule(),
                format!("filesystem {}s are disabled by policy; refusing '{path}'", kind.verb()),
            ))
        }
        "workspace" => true,
        "allowlist" | "allow-list" | "explicit" | "roots" => false,
        other => {
            /* Fail closed: a typo in a security setting must never widen it. */
            return Err(PolicyDenied::new(
                "E_PERMISSION_DENIED",
                kind.rule(),
                format!(
                    "unknown fs.{key} mode '{other}' (expected off|workspace|allowlist|any); refusing '{path}'"
                ),
            ));
        }
    };

    let mut allowed: Vec<PathBuf> = Vec::new();
    for r in &roots {
        if let Some(c) = canonical_nearest(Path::new(r)) {
            if !allowed.contains(&c) {
                allowed.push(c);
            }
        }
    }
    if confine_to_cwd {
        if let Ok(cwd) = std::env::current_dir() {
            let c = std::fs::canonicalize(&cwd).unwrap_or(cwd);
            if !allowed.contains(&c) {
                allowed.push(c);
            }
        }
    }
    if allowed.is_empty() {
        return Err(PolicyDenied::new(
            "E_PATH_OUTSIDE",
            "R7.6/fs.roots",
            format!("no fs.roots entry resolved to a real directory; refusing '{path}'"),
        ));
    }
    let Some(target) = canonical_nearest(&given) else {
        return Err(PolicyDenied::new(
            "E_PATH_OUTSIDE",
            kind.rule(),
            format!("cannot resolve '{path}' to a real path; refusing it"),
        ));
    };
    if allowed.iter().any(|r| target.starts_with(r)) {
        return Ok(target);
    }
    let roots_txt = allowed
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<String>>()
        .join(", ");
    Err(PolicyDenied::new(
        "E_PATH_OUTSIDE",
        if confine_to_cwd { "R7.6/fs.workspace" } else { "R7.6/fs.roots" },
        format!(
            "'{path}' resolves to '{}', outside the allowed roots [{roots_txt}]",
            target.display()
        ),
    ))
}

/* ── process spawning (H-02a) ─────────────────────────────────────── */

/// tools.js raises this integer depth while a policy-governed tool dispatch
/// is in flight — H-01's single enforcement point. It is how the spawn bridge
/// tells "the runtime is acting for a model or a served HTTP caller" (the
/// only way a model can reach a process: built-in tools are policied and
/// code_eval is hermetic) from "a host script is running its own plumbing".
const TOOL_ACTIVE_GLOBAL: &std::ffi::CStr = c"__sofuu_tool_active";

unsafe fn tool_dispatch_active(ctx: *mut JSContext) -> bool {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let v = qjs::sofuu_js_get_property_str(ctx, global, TOOL_ACTIVE_GLOBAL.as_ptr());
    let mut depth: i32 = 0;
    if qjs::sofuu_js_is_number(v) != 0 {
        let _ = qjs::JS_ToInt32(ctx, &mut depth, v);
    }
    qjs::sofuu_js_free_value(ctx, v);
    qjs::sofuu_js_free_value(ctx, global);
    depth > 0
}

/// H-02(a): may this runtime start a process at all?
///
/// * `SOFUU_NO_SHELL` is the operator's kill switch and binds EVERY caller,
///   host scripts included (R7.6: strip execution everywhere).
/// * `shell.enabled == false` binds every spawn the runtime makes on behalf
///   of a policy-governed dispatch, and every spawn when the policy asks for
///   `shell.spawn_strict: true`. A top-level host script is the policy's
///   OWNER — it can call sofuu.policy.set('{}') at any point — so sandboxing
///   its own plumbing would be theatre, and the contract harnesses depend on
///   that (t72/t75 install shell.enabled:false and then spawn curl and the
///   CLI to exercise the other surfaces). The strict mode is there for
///   embedders that want the R7.2 reading unconditionally.
/// * `shell.allow` binds the command line in EVERY context: it is an
///   explicit, command-scoped allow-list, so a non-matching spawn is a
///   denial even for host plumbing.
pub(crate) unsafe fn gate_process_spawn(
    ctx: *mut JSContext,
    command_lines: &[String],
) -> Result<(), PolicyDenied> {
    let disabled = || {
        PolicyDenied::new(
            "E_PERMISSION_DENIED",
            "R7.6/shell.enabled",
            "Process execution disabled by policy".to_string(),
        )
    };
    if std::env::var_os("SOFUU_NO_SHELL").is_some() {
        return Err(disabled());
    }
    if policy_is_empty(ctx) {
        return Ok(());
    }
    let policy = policy_json(ctx);
    if bool_at(&policy, &["shell", "enabled"]) == Some(false) {
        /* H-02(a) (AUDIT-2026-10-29): a shell INTERPRETER is refused
         * unconditionally — that is the very thing `shell.enabled: false`
         * says is off, and the audit bypassed it with a plain
         * `sofuu.spawn({command: '/bin/sh', args: ['-c', ...]})`. */
        if command_lines.iter().any(|l| program_is_shell(l)) {
            return Err(disabled());
        }
        /* Everything else: only tool-originated dispatch (or an explicit
         * `shell.spawn_strict`) is bound, so a harness that installs this
         * policy and then drives the binary keeps working. */
        let strict = bool_at(&policy, &["shell", "spawn_strict"]) == Some(true)
            || bool_at(&policy, &["shell", "spawnStrict"]) == Some(true)
            || tool_dispatch_active(ctx);
        if strict {
            return Err(disabled());
        }
    }
    let allow = list_at(&policy, &["shell", "allow"]);
    if !allow.is_empty() {
        /* A spawn offers several spellings of the same command (see
         * process_spawn::policy_command_lines): the argv join, the same with
         * argv[0] reduced to its basename, and — for `sh -c <payload>` — the
         * payload. Any match allows. That is what makes `shell.allow:
         * ["git *"]` mean the same thing whether the work arrives through the
         * bash tool (a `tool.run` request carrying the command string) or
         * through sofuu.spawn (an argv of `/bin/sh -c git status`). Judging
         * the bare argv alone denied a command the policy had just allowed,
         * and `shell.allow` must not be narrower than the tool it governs. */
        let ok = command_lines.iter().any(|line| {
            let first = line.split(' ').next().unwrap_or("");
            let rest = line.get(first.len()..).unwrap_or("");
            let short = format!("{}{}", first.rsplit('/').next().unwrap_or(first), rest);
            allow
                .iter()
                .any(|p| glob_match(p, line) || glob_match(p, &short))
        });
        if !ok {
            let shown = command_lines
                .first()
                .cloned()
                .unwrap_or_else(|| "(empty command line)".to_string());
            return Err(PolicyDenied::new(
                "E_PERMISSION_DENIED",
                "R7.6/shell.allow",
                format!("command '{shown}' does not match shell.allow"),
            ));
        }
    }
    Ok(())
}

/* ── network egress (H-02b) ───────────────────────────────────────── */

/// The network half of the effective policy, resolved once per fetch.
pub(crate) struct NetPolicy {
    /// `net.ssrf: true` (or `net.allow_private: false`): refuse
    /// loopback / link-local / private / ULA targets by RESOLVED address.
    pub ssrf: bool,
    /// `net.deny` entries that are IP literals, matched against the RESOLVED
    /// address as well, so `net.deny: ["127.0.0.1"]` also stops `localhost`
    /// and any DNS name that resolves there.
    pub deny_ips: Vec<IpAddr>,
}

fn parse_ip_literal(s: &str) -> Option<IpAddr> {
    let h = norm_host(s);
    let h = h.trim_start_matches('[').trim_end_matches(']');
    h.parse::<IpAddr>().ok()
}

pub(crate) unsafe fn net_policy(ctx: *mut JSContext) -> NetPolicy {
    if policy_is_empty(ctx) {
        return NetPolicy { ssrf: false, deny_ips: Vec::new() };
    }
    let policy = policy_json(ctx);
    let ssrf = net_ssrf(&policy);
    let deny_ips = list_at(&policy, &["net", "deny"])
        .iter()
        .filter_map(|p| parse_ip_literal(p))
        .collect();
    NetPolicy { ssrf, deny_ips }
}

/// Is `ip` an address SSRF protection refuses (loopback, private, CGNAT,
/// link-local, unique-local, unspecified/broadcast)?
pub(crate) fn ip_is_internal(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => ipv4_is_internal(v4),
        IpAddr::V6(v6) => {
            /* IPv4-mapped ::ffff:a.b.c.d — judged as the IPv4 it reaches.
             * Written out rather than via to_ipv4_mapped() so the check does
             * not depend on a newer std than the toolchain pins. */
            let o = v6.octets();
            if o[..10].iter().all(|&b| b == 0) && o[10] == 0xff && o[11] == 0xff {
                return ipv4_is_internal(Ipv4Addr::new(o[12], o[13], o[14], o[15]));
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || (u16::from_be_bytes([o[0], o[1]]) & 0xfe00) == 0xfc00 /* fc00::/7 */
                || (u16::from_be_bytes([o[0], o[1]]) & 0xffc0) == 0xfe80 /* fe80::/10 */
        }
    }
}

fn ipv4_is_internal(v4: Ipv4Addr) -> bool {
    let o = v4.octets();
    v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_unspecified()
        || v4.is_broadcast()
        || (o[0] == 100 && (o[1] & 0xc0) == 64) /* 100.64/10 CGNAT */
        || (o[0] == 192 && o[1] == 0 && o[2] == 0) /* 192.0.0/24 IETF */
}

/// net.ssrf / net.allow_private (`allowPrivate`), shared by `net_policy`
/// and the pure evaluator so the two read the same switch.
fn net_ssrf(policy: &serde_json::Value) -> bool {
    bool_at(policy, &["net", "ssrf"]) == Some(true)
        || bool_at(policy, &["net", "allow_private"]) == Some(false)
        || bool_at(policy, &["net", "allowPrivate"]) == Some(false)
}

/// Does this policy contain any rule that only a host can satisfy?
fn has_net_rules(policy: &serde_json::Value) -> bool {
    !list_at(policy, &["net", "deny"]).is_empty()
        || !list_at(policy, &["net", "allow"]).is_empty()
        || net_ssrf(policy)
}

/// Does a `net.deny` IP entry cover this resolved address? Loopback is one
/// trust zone: denying either loopback family denies both, so
/// `net.deny: ["127.0.0.1"]` cannot be side-stepped with `localhost` → `::1`.
pub(crate) fn ip_denied(deny: &[IpAddr], ip: IpAddr) -> bool {
    deny.iter()
        .any(|d| *d == ip || (d.is_loopback() && ip.is_loopback()))
}

/// H-02(b): the fetch-level host gate. Consults the SAME evaluator every
/// other surface uses (`op: "net.fetch"`), so `net.deny` / `net.allow` decide
/// before curl is handed the URL. Loopback aliases are probed too: a deny of
/// `127.0.0.1` must also catch `localhost`, `[::1]` and friends.
pub(crate) unsafe fn gate_net_host(ctx: *mut JSContext, host: &str) -> Result<(), PolicyDenied> {
    if policy_is_empty(ctx) {
        return Ok(());
    }
    let policy = policy_json(ctx);
    let h = norm_host(host);
    let mut candidates: Vec<String> = vec![host.to_string()];
    if h == "localhost"
        || h.ends_with(".localhost")
        || h == "127.0.0.1"
        || h == "::1"
        || h == "0.0.0.0"
        || h == "::"
    {
        candidates.push("127.0.0.1".to_string());
        candidates.push("[::1]".to_string());
    } else if parse_ip_literal(&h).map(|ip| ip.is_loopback()).unwrap_or(false) {
        candidates.push("127.0.0.1".to_string());
        candidates.push("[::1]".to_string());
    }
    for cand in candidates {
        let req = serde_json::json!({ "op": "net.fetch", "host": cand });
        let d = evaluate(&req, &policy);
        if d.get("decision").and_then(|v| v.as_str()) == Some("deny") {
            let rule = d.get("rule").and_then(|v| v.as_str()).unwrap_or("R7.1/net.deny");
            let reason = d
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("host is denied by policy");
            return Err(PolicyDenied::new(
                "E_PERMISSION_DENIED",
                rule,
                format!("{reason}; refusing '{host}'"),
            ));
        }
    }
    Ok(())
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

    /* R7.P3: shell.deny — the default profile's destructive-command list. */
    #[test]
    fn shell_deny_blocks_listed_commands_and_wins_over_allow() {
        let p = serde_json::json!({
            "shell": {
                "deny": ["rm -rf *", "mkfs*"],
                "allow": ["rm *", "echo *"]
            }
        });
        let deny = serde_json::json!({"op":"shell.run","command":"rm -rf /"});
        assert_eq!(eval(deny.clone(), p.clone()), "deny");
        assert_eq!(rule(deny, p.clone()), "R7.6/shell.deny");
        /* The same command through the bash tool is the same gate. */
        let via_tool = serde_json::json!({"op":"tool.run","tool":"bash","command":"mkfs.ext4 /dev/sda"});
        assert_eq!(eval(via_tool, p.clone()), "deny");
        /* A command that only the broad allow matches still runs. */
        let ok = serde_json::json!({"op":"shell.run","command":"echo hi"});
        assert_eq!(eval(ok, p.clone()), "allow");
        /* An unlisted command hits the allowlist, not the deny list. */
        let unlisted = serde_json::json!({"op":"shell.run","command":"ls -la"});
        assert_eq!(eval(unlisted.clone(), p.clone()), "deny");
        assert_eq!(rule(unlisted, p), "R7.6/shell.allow");
    }

    #[test]
    fn shell_deny_inventory_is_conditional_not_denied() {
        let p = serde_json::json!({"shell":{"deny":["rm -rf *"]}});
        let d = evaluate(
            &serde_json::json!({"op":"tool.run","tool":"bash"}),
            &p,
        );
        assert_eq!(d.get("decision").and_then(|v| v.as_str()), Some("allow"));
        assert_eq!(d.get("conditional").and_then(|v| v.as_bool()), Some(true));
    }

    /* R7.P3: net.enabled — `--no-net` and the offline profile. */
    #[test]
    fn net_enabled_false_denies_every_host_and_cannot_be_widened() {
        let p = serde_json::json!({"net":{"enabled":false,"allow":["api.example.com"]}});
        for host in ["api.example.com", "localhost", "127.0.0.1", "evil.test"] {
            let req = serde_json::json!({"op":"net.fetch","host":host});
            assert_eq!(eval(req.clone(), p.clone()), "deny", "host {host}");
            assert_eq!(rule(req, p.clone()), "R7.1/net.enabled");
        }
        /* No host on the request is still a network question. */
        assert_eq!(
            eval(serde_json::json!({"op":"net.fetch"}), p.clone()),
            "deny"
        );
        /* Unset / true keeps the normal allow/deny semantics. */
        let plain = serde_json::json!({"net":{"allow":["api.example.com"]}});
        assert_eq!(
            eval(serde_json::json!({"op":"net.fetch","host":"api.example.com"}), plain.clone()),
            "allow"
        );
        assert_eq!(
            eval(serde_json::json!({"op":"net.fetch","host":"other.test"}), plain),
            "deny"
        );
    }

    #[test]
    fn approval_mode_always_and_never() {
        let always = serde_json::json!({"approval":{"mode":"always"}});
        assert_eq!(
            eval(serde_json::json!({"op":"tool.run","tool":"read_file"}), always.clone()),
            "ask"
        );
        assert_eq!(
            rule(serde_json::json!({"op":"tool.run","tool":"read_file"}), always),
            "R7.4/approval.mode"
        );
        let never = serde_json::json!({"approval":{"mode":"never"}});
        assert_eq!(
            eval(serde_json::json!({"op":"tool.run","tool":"bash","command":"ls"}), never.clone()),
            "allow"
        );
        assert_eq!(
            rule(serde_json::json!({"op":"tool.run","tool":"bash"}), never),
            "R7.4/approval.mode"
        );
        /* A deny still wins over 'never'. */
        let both = serde_json::json!({"approval":{"mode":"never"},"tools":{"deny":["bash"]}});
        assert_eq!(
            eval(serde_json::json!({"op":"tool.run","tool":"bash"}), both),
            "deny"
        );
    }

    #[test]
    fn http_route_deny_prefix() {
        let p = serde_json::json!({"http":{"deny":["/v1/tools"]}});
        assert_eq!(
            eval(serde_json::json!({"op":"http.request","path":"/v1/tools/read_file/call"}), p.clone()),
            "deny"
        );
        assert_eq!(
            rule(serde_json::json!({"op":"http.request","path":"/v1/tools"}), p.clone()),
            "R7.6/http.deny"
        );
        assert_eq!(
            eval(serde_json::json!({"op":"http.request","path":"/v1/chat/completions"}), p),
            "allow"
        );
    }

    #[test]
    fn empty_policy_allows() {        assert_eq!(eval(serde_json::json!({"op":"tool.run","tool":"read_file"}), serde_json::json!({})), "allow");
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

    /* ── H-02 / R-24 gates ─────────────────────────────────────────── */

    #[test]
    fn tool_inventory_query_is_conditional_not_denied_under_shell_allow() {
        // H-01: sofuu.tools.list() asks tool.run for bash with no command. A
        // command-scoped rule cannot deny a command that was never supplied,
        // so the row reports itself conditional — and a real execution with a
        // non-matching command still denies.
        let p = serde_json::json!({"shell":{"allow":["git *"]}});
        let inv = evaluate(&serde_json::json!({"op":"tool.run","tool":"bash"}), &p);
        assert_eq!(inv.get("decision").and_then(|v| v.as_str()), Some("allow"));
        assert_eq!(inv.get("conditional").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(eval(serde_json::json!({"op":"tool.run","tool":"bash","command":"git status"}), p.clone()), "allow");
        assert_eq!(eval(serde_json::json!({"op":"tool.run","tool":"bash","command":"rm -rf /"}), p), "deny");
    }

    #[test]
    fn ssrf_classes_cover_loopback_private_and_linklocal() {
        for s in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.9",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fd00::1",
            "::ffff:127.0.0.1",
        ] {
            let ip: IpAddr = s.parse().unwrap();
            assert!(ip_is_internal(ip), "{s} must count as internal");
        }
        for s in ["8.8.8.8", "1.1.1.1", "2606:4700::1111"] {
            let ip: IpAddr = s.parse().unwrap();
            assert!(!ip_is_internal(ip), "{s} must not count as internal");
        }
    }

    #[test]
    fn net_deny_ip_entries_cover_loopback_aliases() {
        let deny: Vec<IpAddr> = vec!["127.0.0.1".parse().unwrap()];
        assert!(ip_denied(&deny, "127.0.0.1".parse().unwrap()));
        assert!(ip_denied(&deny, "::1".parse().unwrap()));
        assert!(!ip_denied(&deny, "10.0.0.1".parse().unwrap()));
        assert!(parse_ip_literal("[::1]:8080").is_some(), "normalised before parsing");
    }

    /* ── AUDIT-2026-10-31 P1-15: the schema check ────────────────── */

    #[test]
    fn schema_accepts_every_shipped_profile_shape() {
        assert!(validate_policy(&serde_json::json!({})).is_ok());
        assert!(validate_policy(&serde_json::json!({
            "_notes": ["generated by sofuu policy init"],
            "fs": { "roots": ["/work/proj"], "write": "off" },
            "shell": { "enabled": false, "deny": ["rm -rf *"], "allow": ["git *"] },
            "net": { "enabled": false, "allow": ["api.example.com"], "deny": ["evil.io"] },
            "http": { "deny": ["/v1/tools"], "auth": "none" },
            "mcp": { "servers": { "allow": ["srv1"] }, "callTimeoutMs": 80 },
            "tools": { "allow": ["read_file"], "deny": ["bash"] },
            "approval": { "mode": "per-run", "cache": true },
            "secrets": { "source": "file", "file": "/tmp/secrets.json" },
            "memory": { "write": "off" },
        }))
        .is_ok());
        /* The forms the fs resolver really understands. */
        assert!(validate_policy(&serde_json::json!({"fs":{"write":false}})).is_ok());
        assert!(validate_policy(&serde_json::json!({"fs":{"write":["/a","/b"]}})).is_ok());
        assert!(validate_policy(&serde_json::json!({"fs":{"roots":"/a"}})).is_ok());
    }

    #[test]
    fn schema_refuses_typos_that_used_to_fail_open() {
        /* A misspelt section: nothing reads it, everything stays allowed. */
        let e = validate_policy(&serde_json::json!({"nt":{"deny":["evil.io"]}}))
            .expect_err("unknown section must be refused");
        assert!(e.contains("unknown policy section"), "{e}");
        /* A misspelt key inside a real section. */
        let e = validate_policy(&serde_json::json!({"net":{"denny":["evil.io"]}}))
            .expect_err("unknown key must be refused");
        assert!(e.contains("net.denny"), "{e}");
        /* A string where a list belongs: list_at reads it as an empty list. */
        let e = validate_policy(&serde_json::json!({"net":{"deny":"evil.io"}}))
            .expect_err("deny must be a list");
        assert!(e.contains("net.deny"), "{e}");
        /* A string boolean: bool_at returns None, so the switch never fires. */
        let e = validate_policy(&serde_json::json!({"shell":{"enabled":"false"}}))
            .expect_err("enabled must be a boolean");
        assert!(e.contains("shell.enabled"), "{e}");
        /* An unknown fs mode would deny every call at the gate — refuse it at
         * set time instead, while the operator is looking. */
        let e = validate_policy(&serde_json::json!({"fs":{"write":"readonly"}}))
            .expect_err("unknown fs.write mode must be refused");
        assert!(e.contains("fs.write"), "{e}");
        /* An unknown approval mode silently defaults to per-run. */
        assert!(validate_policy(&serde_json::json!({"approval":{"mode":"sometimes"}})).is_err());
        /* A non-object policy. */
        assert!(validate_policy(&serde_json::json!(["off"])).is_err());
    }

    /* ── AUDIT-2026-10-31 P1-6: evaluate() vs the gate ───────────── */

    #[test]
    fn evaluate_reads_fs_write_the_way_the_gate_does() {
        /* The audit's "write off" payload was the boolean form; the evaluator
         * used to answer "allow" while gate_fs refused the write. */
        let p = serde_json::json!({"fs":{"write":false}});
        assert_eq!(eval(serde_json::json!({"op":"fs.write","path":"/etc/hosts"}), p.clone()), "deny");
        assert_eq!(rule(serde_json::json!({"op":"fs.write","path":"/etc/hosts"}), p), "R7.6/fs.write");
        /* An unknown mode fails closed on both sides. */
        let typo = serde_json::json!({"fs":{"write":"readonly"}});
        assert_eq!(eval(serde_json::json!({"op":"fs.write","path":"/tmp/x"}), typo.clone()), "deny");
        assert_eq!(rule(serde_json::json!({"op":"fs.write","path":"/tmp/x"}), typo), "R7.6/fs.write");
        /* An inline allowlist is "allowlist", not "no rule". */
        let inline = serde_json::json!({"fs":{"write":["/work/proj"]}});
        assert_eq!(eval(serde_json::json!({"op":"fs.write","path":"/work/proj/a"}), inline.clone()), "allow");
        assert_eq!(eval(serde_json::json!({"op":"fs.write","path":"/etc/hosts"}), inline), "deny");
        /* workspace confines to fs.roots and the cwd — a path under neither
         * is denied, which is what gate_fs enforces. */
        let ws = serde_json::json!({"fs":{"roots":["/work/proj"],"write":"workspace"}});
        assert_eq!(eval(serde_json::json!({"op":"fs.write","path":"/etc/hosts"}), ws.clone()), "deny");
        assert_eq!(eval(serde_json::json!({"op":"fs.write","path":"/work/proj/a"}), ws), "allow");
    }

    #[test]
    fn evaluate_will_not_allow_a_net_fetch_it_cannot_judge() {
        /* The audit asked {op:'net.fetch', target:'example.com'} — a field the
         * evaluator never reads — while the policy had net.deny, and got
         * "allow" from an API whose own fetch was denied. */
        let p = serde_json::json!({"net":{"deny":["example.com"]}});
        let d = evaluate(&serde_json::json!({"op":"net.fetch","target":"example.com"}), &p);
        assert_eq!(d.get("decision").and_then(|v| v.as_str()), Some("deny"));
        assert_eq!(d.get("rule").and_then(|v| v.as_str()), Some("R7.1/net.host-required"));
        /* With no network rules at all there is nothing to judge, so the
         * answer stays allow — the empty policy must keep working. */
        assert_eq!(eval(serde_json::json!({"op":"net.fetch"}), serde_json::json!({})), "allow");
        /* net.ssrf refuses an internal IP literal without a resolver. */
        let ssrf = serde_json::json!({"net":{"ssrf":true}});
        let d = evaluate(&serde_json::json!({"op":"net.fetch","host":"169.254.169.254"}), &ssrf);
        assert_eq!(d.get("decision").and_then(|v| v.as_str()), Some("deny"));
        assert_eq!(d.get("rule").and_then(|v| v.as_str()), Some("R7.1/net.ssrf"));
        assert_eq!(eval(serde_json::json!({"op":"net.fetch","host":"8.8.8.8"}), ssrf), "allow");
        /* allow_private:false is the same switch spelled the other way. */
        let ap = serde_json::json!({"net":{"allow_private":false}});
        assert_eq!(eval(serde_json::json!({"op":"net.fetch","host":"10.0.0.1"}), ap), "deny");
    }

    #[test]
    fn lexical_clean_folds_dot_segments() {
        assert_eq!(lexical_clean(Path::new("/a/b/../c/./d")), PathBuf::from("/a/c/d"));
        assert_eq!(lexical_clean(Path::new("a/./b")), PathBuf::from("a/b"));
    }
}
