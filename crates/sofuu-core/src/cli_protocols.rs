// cli_protocols.rs — `sofuu tools`, `sofuu mcp`, `sofuu policy` (D1).
//
// The three commands that make the protocol surfaces inspectable from a
// shell: the tool inventory (T3), MCP connectivity/config (M1/M2/S6), and
// the policy evaluator (S4/S5). Everything here is read-only except
// `mcp add|remove`, which rewrites ~/.sofuu/mcp.json atomically (validate →
// tmp file 0600 → rename) so a crash cannot truncate a working config.

use std::io::Write;
use std::path::PathBuf;

use sofuu_ffi::SofuuRuntime;

fn mcp_config_path() -> PathBuf {
    if let Ok(p) = std::env::var("SOFUU_MCP_CONFIG") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let home = sofuu_core::embed_config::home_dir().unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".sofuu").join("mcp.json")
}

/// Load + validate the config array. A corrupt file is an error (never
/// silently overwritten — the operator must fix or delete it).
fn load_mcp_config() -> Result<Vec<serde_json::Value>, String> {
    let path = mcp_config_path();
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    let v: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;
    match v {
        serde_json::Value::Array(list) => Ok(list),
        _ => Err(format!("{} must contain a JSON array of server entries", path.display())),
    }
}

fn save_mcp_config(list: &[serde_json::Value]) -> Result<(), String> {
    let path = mcp_config_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let body = serde_json::to_string_pretty(list).map_err(|e| format!("serialize: {e}"))?;
    let tmp = path.with_extension(format!("json.tmp{}", std::process::id()));
    #[cfg(unix)]
    let write = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)
            .and_then(|mut f| f.write_all(body.as_bytes()))
    };
    #[cfg(not(unix))]
    let write = std::fs::write(&tmp, body.as_bytes());
    write.map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot replace {}: {e}", path.display())
    })
}

/// Strip ANSI SGR sequences (eval_repl's display colors every value).
fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\u{1b}' {
            if it.peek() == Some(&'[') {
                it.next();
                while let Some(n) = it.next() {
                    if n.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Run a script that prints its own output and leaves the exit code in
/// `globalThis.__sofuu_cli_rc`. The value is read back through eval_repl's
/// display, so ANSI + quotes are stripped before the integer parse.
fn eval_rc(rt: &SofuuRuntime, script: &str) -> i32 {
    if rt.eval_string(script, "<cli>") != 0 {
        return 1;
    }
    rt.run_jobs();
    let raw = rt.eval_repl("String(globalThis.__sofuu_cli_rc | 0)").unwrap_or_default();
    let clean = strip_ansi(&raw);
    let digits: String = clean.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.parse::<i32>().unwrap_or(0) == 0 { 0 } else { 1 }
}

/* ── tools ─────────────────────────────────────────────────────── */

/// `sofuu tools list [--json] [--policy <file>]`
pub fn tools(rt: &SofuuRuntime, args: &[String]) -> i32 {
    let sub = args.get(2).map(|s| s.as_str()).unwrap_or("list");
    if sub != "list" {
        eprintln!("Usage: sofuu tools list [--json]");
        return 1;
    }
    if let Some(p) = policy_arg(args) {
        if let Err(e) = crate::apply_policy_file(rt, &p) {
            eprintln!("Error: {e}");
            return 1;
        }
    }
    let want_json = args.iter().any(|a| a == "--json");
    let no_shell = std::env::var_os("SOFUU_NO_SHELL").is_some();
    let script = format!(
        r#"globalThis.__sofuu_cli_rc = 0;
(function () {{
  var list = sofuu.tools.list();
  if ({json}) {{ console.log(JSON.stringify(list)); return; }}
  console.log('\x1b[1msofuu tools\x1b[0m — ' + list.length + ' in the inventory');
  for (var i = 0; i < list.length; i++) {{
    var it = list[i];
    var pol = it.policy || {{}};
    var verdict = (pol.allowed === false) ? '\x1b[31mdenied\x1b[0m'
      : (pol.requiresApproval ? '\x1b[33mask\x1b[0m' : 'allow');
    var name = String(it.name);
    while (name.length < 18) name += ' ';
    var src = String(it.source);
    while (src.length < 8) src += ' ';
    console.log('  ' + name + src + verdict);
  }}
  if ({no_shell}) console.log('\x1b[90m  (SOFUU_NO_SHELL=1 — bash is not advertised)\x1b[0m');
}})();"#,
        json = if want_json { "true" } else { "false" },
        no_shell = if no_shell { "true" } else { "false" }
    );
    eval_rc(rt, &script)
}

/* ── mcp ───────────────────────────────────────────────────────── */

/// `sofuu mcp list | test <cmd|url> | add <name> <command> [args…] | remove <name>`
pub fn mcp(rt: &SofuuRuntime, args: &[String]) -> i32 {
    match args.get(2).map(|s| s.as_str()).unwrap_or("") {
        "list" => {
            let list = match load_mcp_config() {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error: {e}");
                    return 1;
                }
            };
            println!("\x1b[1msofuu mcp\x1b[0m — {}", mcp_config_path().display());
            if list.is_empty() {
                println!("  (no servers configured — sofuu mcp add <name> <command>)");
                return 0;
            }
            for s in &list {
                let name = s.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let cmd = s.get("command").and_then(|v| v.as_str()).unwrap_or("?");
                let disabled = s.get("enabled").and_then(|v| v.as_bool()) == Some(false);
                println!("  {name:<16} {cmd}{}", if disabled { "  (disabled)" } else { "" });
            }
            0
        }
        "test" => {
            let target = match args.get(3) {
                Some(t) => t.clone(),
                None => {
                    eprintln!("Usage: sofuu mcp test <command line | http(s)://url>");
                    return 1;
                }
            };
            if let Some(p) = policy_arg(args) {
                if let Err(e) = crate::apply_policy_file(rt, &p) {
                    eprintln!("Error: {e}");
                    return 1;
                }
            }
            let target_json = serde_json::to_string(&target).unwrap_or_else(|_| "\"\"".into());
            let script = format!(
                r#"globalThis.__sofuu_cli_rc = 1;
(async function () {{
  var target = {target};
  try {{
    var client = /^https?:\/\//i.test(target)
      ? await sofuu.mcp.connect({{ url: target, transport: 'auto' }})
      : await sofuu.mcp.connect(target);
    var protocol = (typeof client.protocolVersion === 'function') ? String(client.protocolVersion() || '') : '';
    var tools = await client.listTools();
    var toolsN = (tools && tools.tools) ? tools.tools.length : 0;
    var resN = -1;
    try {{
      var res = await client.listResources();
      resN = (res && res.resources) ? res.resources.length : 0;
    }} catch (eR) {{}}
    var prN = -1;
    try {{
      var pr = await client.call('prompts/list', {{}});
      prN = (pr && pr.prompts) ? pr.prompts.length : 0;
    }} catch (eP) {{}}
    console.log('\x1b[32m✓ ' + target + '\x1b[0m');
    console.log('  protocol ' + (protocol || '(none)') + ' · ' + toolsN + ' tool(s) · ' +
      (resN < 0 ? 'resources (unsupported)' : resN + ' resource(s)') + ' · ' +
      (prN < 0 ? 'prompts (unsupported)' : prN + ' prompt(s)'));
    globalThis.__sofuu_cli_rc = 0;
    try {{ client.disconnect(); }} catch (eD) {{}}
  }} catch (e) {{
    console.log('\x1b[31m✗ ' + target + '\x1b[0m');
    console.log('  ' + String((e && e.message) || e));
    globalThis.__sofuu_cli_rc = 1;
  }}
}})();"#,
                target = target_json
            );
            eval_rc(rt, &script)
        }
        "add" => {
            let (name, command, rest) = match (args.get(3), args.get(4)) {
                (Some(n), Some(c)) => {
                    let rest: &[String] = if args.len() > 5 { &args[5..] } else { &[] };
                    (n.clone(), c.clone(), rest)
                }
                _ => {
                    eprintln!("Usage: sofuu mcp add <name> <command> [args…]");
                    return 1;
                }
            };
            if name.trim().is_empty() {
                eprintln!("Error: server name is required");
                return 1;
            }
            let mut list = match load_mcp_config() {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error: {e}");
                    return 1;
                }
            };
            if list.iter().any(|s| s.get("name").and_then(|v| v.as_str()) == Some(name.as_str())) {
                eprintln!("Error: a server named '{name}' already exists (remove it first)");
                return 1;
            }
            let mut entry = serde_json::json!({ "name": name, "command": command });
            if !rest.is_empty() {
                entry["args"] = serde_json::json!(rest);
            }
            list.push(entry);
            if let Err(e) = save_mcp_config(&list) {
                eprintln!("Error: {e}");
                return 1;
            }
            println!("\x1b[32m✓\x1b[0m added '{name}' to {}", mcp_config_path().display());
            0
        }
        "remove" => {
            let name = match args.get(3) {
                Some(n) => n.clone(),
                None => {
                    eprintln!("Usage: sofuu mcp remove <name>");
                    return 1;
                }
            };
            let list = match load_mcp_config() {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error: {e}");
                    return 1;
                }
            };
            let before = list.len();
            let next: Vec<serde_json::Value> = list
                .into_iter()
                .filter(|s| s.get("name").and_then(|v| v.as_str()) != Some(name.as_str()))
                .collect();
            if next.len() == before {
                eprintln!("Error: no server named '{name}'");
                return 1;
            }
            if let Err(e) = save_mcp_config(&next) {
                eprintln!("Error: {e}");
                return 1;
            }
            println!("\x1b[32m✓\x1b[0m removed '{name}' from {}", mcp_config_path().display());
            0
        }
        _ => {
            eprintln!("Usage: sofuu mcp <list|test <cmd|url>|add <name> <command>|remove <name>>");
            1
        }
    }
}

/* ── policy ────────────────────────────────────────────────────── */

/// `--policy <file>` from a subcommand's own args.
fn policy_arg(args: &[String]) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == "--policy" && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
    }
    None
}

fn policy_from_args(args: &[String]) -> Result<serde_json::Value, String> {
    if let Some(path) = policy_arg(args) {
        let raw = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
        return serde_json::from_str(&raw).map_err(|e| format!("{path} is not valid JSON: {e}"));
    }
    if let Ok(path) = std::env::var("SOFUU_POLICY") {
        if !path.is_empty() {
            let raw = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
            return serde_json::from_str(&raw).map_err(|e| format!("{path} is not valid JSON: {e}"));
        }
    }
    Ok(serde_json::json!({}))
}

/// `sofuu policy check <file>` — evaluate the probe set, print each rule id.
pub fn policy(_rt: &SofuuRuntime, args: &[String]) -> i32 {
    match args.get(2).map(|s| s.as_str()).unwrap_or("") {
        "check" => {
            let path = match args.get(3) {
                Some(p) if !p.starts_with("--") => p.clone(),
                _ => {
                    eprintln!("Usage: sofuu policy check <policy.json>");
                    return 1;
                }
            };
            let raw = match std::fs::read_to_string(&path) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Error: cannot read {path}: {e}");
                    return 1;
                }
            };
            let policy: serde_json::Value = match serde_json::from_str::<serde_json::Value>(&raw) {
                Ok(v) if v.is_object() => v,
                Ok(_) => {
                    eprintln!("Error: {path} must contain a JSON object");
                    return 1;
                }
                Err(e) => {
                    eprintln!("Error: {path} is not valid JSON: {e}");
                    return 1;
                }
            };
            let probes = [
                ("tool.run read_file", serde_json::json!({"op":"tool.run","tool":"read_file"})),
                ("tool.run bash", serde_json::json!({"op":"tool.run","tool":"bash","command":"git status"})),
                ("shell.run rm -rf /", serde_json::json!({"op":"shell.run","command":"rm -rf /"})),
                ("fs.read /etc/passwd", serde_json::json!({"op":"fs.read","path":"/etc/passwd"})),
                ("net.fetch api.example.com", serde_json::json!({"op":"net.fetch","host":"api.example.com"})),
                ("mcp.connect probe", serde_json::json!({"op":"mcp.connect","server":"probe","command":"probe"})),
                ("http.request /v1/chat/completions", serde_json::json!({"op":"http.request","path":"/v1/chat/completions"})),
                ("http.request /v1/tools", serde_json::json!({"op":"http.request","path":"/v1/tools"})),
            ];
            println!("\x1b[1msofuu policy check\x1b[0m — {path}");
            let total = probes.len();
            let mut denied = 0usize;
            for (label, req) in probes {
                let d = sofuu_core::rt::policy::evaluate(&req, &policy);
                let dec = d.get("decision").and_then(|v| v.as_str()).unwrap_or("?");
                let rule = d.get("rule").and_then(|v| v.as_str()).unwrap_or("");
                let colored = match dec {
                    "deny" => {
                        denied += 1;
                        "\x1b[31mdeny\x1b[0m"
                    }
                    "ask" => "\x1b[33mask\x1b[0m",
                    _ => "allow",
                };
                println!("  {:<38} {:<16} {}", label, colored, rule);
            }
            println!("\x1b[90m  {denied} of {total} probes denied\x1b[0m");
            0
        }
        "why" => {
            let op = match args.get(3) {
                Some(o) if !o.starts_with("--") => o.clone(),
                _ => {
                    eprintln!("Usage: sofuu policy why <op> [key=value …] [--policy <file>]");
                    eprintln!("  e.g. sofuu policy why tool.run tool=bash command='rm -rf /'");
                    return 1;
                }
            };
            let mut req = serde_json::json!({ "op": op });
            for a in &args[4..] {
                if a == "--policy" {
                    continue;
                }
                let Some((k, v)) = a.split_once('=') else { continue };
                if k.is_empty() {
                    continue;
                }
                req[k] = serde_json::Value::String(v.to_string());
            }
            let policy = match policy_from_args(args) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Error: {e}");
                    return 1;
                }
            };
            let d = sofuu_core::rt::policy::evaluate(&req, &policy);
            let dec = d.get("decision").and_then(|v| v.as_str()).unwrap_or("allow");
            let rule = d.get("rule").and_then(|v| v.as_str()).unwrap_or("");
            let reason = d.get("reason").and_then(|v| v.as_str()).unwrap_or("");
            let colored = match dec {
                "deny" => "\x1b[31mdeny\x1b[0m",
                "ask" => "\x1b[33mask\x1b[0m",
                _ => "\x1b[32mallow\x1b[0m",
            };
            println!("{colored}  {}  {reason}", if rule.is_empty() { "" } else { rule });
            println!("{}", serde_json::to_string(&d).unwrap_or_default());
            0
        }
        _ => {
            eprintln!("Usage: sofuu policy <check <file>|why <op> [key=value …]>");
            1
        }
    }
}
