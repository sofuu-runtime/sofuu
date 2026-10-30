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
    /* No explicit file: answer under the policy that is actually in force at
     * boot (user then project layers), so `why` and the runtime cannot
     * disagree. An invalid on-disk policy is an error, not "no policy". */
    sofuu_core::rt::policy::layered_policy_from_disk()
}

/// `sofuu policy check <file>` — evaluate the probe set, print each rule id.
pub fn policy(_rt: &SofuuRuntime, args: &[String]) -> i32 {
    match args.get(2).map(|s| s.as_str()).unwrap_or("") {
        "init" => policy_init(args),
        "check" => {
            let path = match args.get(3) {
                Some(p) if !p.starts_with("--") => p.clone(),
                _ => {
                    eprintln!("Usage: sofuu policy check <policy.json | script.ts> [--policy <file>]");
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
            /* Two modes, one command: a policy DOCUMENT is validated with the
             * probe set (the historical behaviour, exit 0); a SCRIPT is
             * audited against the active policy and exits non-zero when a
             * requirement would be denied, so CI can gate on it. */
            let is_document = path.ends_with(".json") || raw.trim_start().starts_with('{');
            if !is_document {
                return policy_audit(args, &path, &raw, false);
            }
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
        "dry-run" => {
            let path = match args.get(3) {
                Some(p) if !p.starts_with("--") => p.clone(),
                _ => {
                    eprintln!("Usage: sofuu policy dry-run <script.ts> [--policy <file>]");
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
            policy_audit(args, &path, &raw, true);
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
            eprintln!("Usage: sofuu policy <init|check|dry-run|why>");
            eprintln!("  sofuu policy init [--profile default|strict|offline] [--force] [--global] [--print]");
            eprintln!("  sofuu policy check <policy.json | script.ts> [--policy <file>] [--json]");
            eprintln!("  sofuu policy dry-run <script.ts> [--policy <file>]");
            eprintln!("  sofuu policy why <op> [key=value …] [--policy <file>]");
            1
        }
    }
}

/* ── R7.P3: policy init / static audit / dry-run ─────────────────── */

/// The destructive-command list the `default` and `offline` profiles ship.
/// Full-command globs (rt::policy::glob_match), so they are patterns over the
/// command STRING — a best-effort floor, not a shell parser. `shell.enabled:
/// false` is the real switch.
const SHELL_DENY_DEFAULT: &[&str] = &[
    "rm -rf *",
    "rm -fr *",
    "rm -r /*",
    "mkfs*",
    "dd if=*of=/dev/*",
    "shutdown*",
    "reboot*",
    "chmod -R 777 /*",
    "git push --force*",
    "git reset --hard*",
];

fn policy_file_path(global: bool) -> PathBuf {
    if global {
        let home = sofuu_core::embed_config::home_dir().unwrap_or_else(|| ".".into());
        PathBuf::from(home).join(".sofuu").join("policy.json")
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".sofuu")
            .join("policy.json")
    }
}

/// The three shipped profiles. `roots` is the ABSOLUTE project directory as
/// it is at generation time: the evaluator compares lexically, so a relative
/// root would only match paths spelled with that same prefix. Regenerate the
/// file if the project moves.
fn policy_profile(name: &str) -> Option<serde_json::Value> {
    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| std::fs::canonicalize(&p).ok().or(Some(p)))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".into());
    let deny = serde_json::json!(SHELL_DENY_DEFAULT);
    Some(match name {
        "default" => serde_json::json!({
            "_notes": [
                "Sofuu R7 policy — project scope. Generated by `sofuu policy init`.",
                "fs.roots confines filesystem access to this project (regenerate if it moves).",
                "shell.deny blocks destructive command patterns; it is best-effort.",
                "approval.mode: per-run — each driver's own gate."
            ],
            "fs": { "roots": [cwd] },
            "shell": { "enabled": true, "deny": deny },
            "approval": { "mode": "per-run" }
        }),
        "strict" => serde_json::json!({
            "_notes": [
                "Sofuu R7 policy — strict. Generated by `sofuu policy init --profile strict`.",
                "Filesystem is read-only and confined to this project.",
                "Shell is disabled for every surface (equivalent to SOFUU_NO_SHELL=1).",
                "Outbound network is OFF. To permit specific hosts, replace",
                "`net: {enabled: false}` with `net: {allow: [\"host.example\"]}`.",
                "Every tool call asks for approval."
            ],
            "fs": { "roots": [cwd], "write": "off" },
            "shell": { "enabled": false },
            "net": { "enabled": false },
            "approval": { "mode": "always" }
        }),
        "offline" => serde_json::json!({
            "_notes": [
                "Sofuu R7 policy — offline. Generated by `sofuu policy init --profile offline`.",
                "Network is completely disabled (net.enabled: false).",
                "Filesystem is confined to this project; shell keeps its destructive-command floor."
            ],
            "fs": { "roots": [cwd] },
            "net": { "enabled": false },
            "shell": { "enabled": true, "deny": deny }
        }),
        _ => return None,
    })
}

/// `sofuu policy init [--profile <p>] [--force] [--global] [--print]`
fn policy_init(args: &[String]) -> i32 {
    let mut profile = "default";
    let mut force = false;
    let mut global = false;
    let mut print_only = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" | "-p" => match args.get(i + 1) {
                Some(p) => {
                    profile = p.as_str();
                    i += 2;
                    continue;
                }
                None => {
                    eprintln!("Error: --profile needs a value (default|strict|offline)");
                    return 1;
                }
            },
            other if other.starts_with("--profile=") => {
                profile = &other["--profile=".len()..];
            }
            "--force" | "-f" => force = true,
            "--global" | "-g" => global = true,
            "--print" => print_only = true,
            other => {
                eprintln!("Error: unknown argument '{other}'");
                return 1;
            }
        }
        i += 1;
    }
    let Some(doc) = policy_profile(profile) else {
        eprintln!("Error: unknown profile '{profile}' (want default, strict or offline)");
        return 1;
    };
    let body = match serde_json::to_string_pretty(&doc) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {e}");
            return 1;
        }
    };
    if print_only {
        println!("{body}");
        return 0;
    }
    let path = policy_file_path(global);
    if path.exists() && !force {
        eprintln!(
            "Error: {} already exists — pass --force to overwrite it",
            path.display()
        );
        return 1;
    }
    if let Some(dir) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("Error: cannot create {}: {e}", dir.display());
            return 1;
        }
    }
    if let Err(e) = std::fs::write(&path, body.as_bytes()) {
        eprintln!("Error: cannot write {}: {e}", path.display());
        return 1;
    }
    println!("\x1b[32m✓\x1b[0m wrote {} ({profile} profile)", path.display());
    let roots = doc
        .get("fs")
        .and_then(|v| v.get("roots"))
        .and_then(|v| v.get(0))
        .and_then(|v| v.as_str())
        .unwrap_or(".");
    match profile {
        "default" => {
            println!("  fs    confined to {roots}");
            println!("  shell destructive commands blocked (best-effort) · approvals per-run");
        }
        "strict" => {
            println!("  fs    read-only, confined to {roots}");
            println!("  shell disabled · net OFF · every tool call asks");
        }
        _ => {
            println!("  fs    confined to {roots}");
            println!("  net   OFF · shell keeps its destructive-command floor");
        }
    }
    println!("\x1b[90m  inspect with `sofuu policy check {path}`\x1b[0m", path = path.display());
    0
}

/// One capability the script asks for, already turned into an evaluator
/// request.
struct Finding {
    op: &'static str,
    target: String,
    req: serde_json::Value,
}

/// Strip `//` and `/* */` comments while preserving strings (a URL in a
/// comment is not a requirement; a URL in a string is).
fn strip_comments(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut quote: Option<char> = None;
    while i < b.len() {
        let c = b[i];
        if let Some(q) = quote {
            out.push(c);
            if c == '\\' && i + 1 < b.len() {
                out.push(b[i + 1]);
                i += 2;
                continue;
            }
            if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        match c {
            '"' | '\'' | '`' => {
                quote = Some(c);
                out.push(c);
                i += 1;
            }
            '/' if i + 1 < b.len() && b[i + 1] == '/' => {
                while i < b.len() && b[i] != '\n' {
                    i += 1;
                }
            }
            '/' if i + 1 < b.len() && b[i + 1] == '*' => {
                i += 2;
                while i + 1 < b.len() && !(b[i] == '*' && b[i + 1] == '/') {
                    if b[i] == '\n' {
                        out.push('\n');
                    }
                    i += 1;
                }
                i = (i + 2).min(b.len());
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// The first quoted literal after `from`, if any (single, double or backtick).
fn literal_after(src: &str, from: usize) -> Option<String> {
    let rest = &src[from.min(src.len())..];
    let bytes = rest.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i] != b'"' && bytes[i] != b'\'' && bytes[i] != b'`' {
        /* Stop at the end of the call/statement: a literal belonging to the
         * NEXT call is not this call's argument. */
        if matches!(bytes[i], b')' | b';' | b'\n' | b'[' | b']' | b'{' | b'}' | b'=') {
            return None;
        }
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    let q = bytes[i];
    let start = i + 1;
    let mut j = start;
    while j < bytes.len() && bytes[j] != q && bytes[j] != b'\n' {
        j += 1;
    }
    (j > start).then(|| rest[start..j].to_string())
}

fn push_unique(out: &mut Vec<Finding>, seen: &mut std::collections::HashSet<String>, f: Finding) {
    let key = format!("{}\u{0}{}", f.op, f.target);
    if seen.insert(key) {
        out.push(f);
    }
}

/// The static scanner. Deliberately lexical: it reports what the SOURCE asks
/// for (calls, literals, `process.env` reads) and lets the one evaluator
/// answer. Dynamic targets are reported as `<dynamic>` and judged as such —
/// the runtime still enforces them per call.
fn scan_source(src: &str) -> Vec<Finding> {
    let s = strip_comments(src);
    let mut out: Vec<Finding> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    /* URLs → net.fetch. Covers fetch(), sofuu.http.request and bare literals. */
    for scheme in ["https://", "http://"] {
        let mut at = 0;
        while let Some(rel) = s[at..].find(scheme) {
            let start = at + rel;
            let tail = &s[start + scheme.len()..];
            let end = tail
                .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '`' | ')' | ']' | '}' | ',' | ';'))
                .unwrap_or(tail.len());
            let authority = &tail[..end];
            let host = authority.split('/').next().unwrap_or(authority);
            if !host.is_empty() {
                push_unique(
                    &mut out,
                    &mut seen,
                    Finding {
                        op: "net.fetch",
                        target: host.to_string(),
                        req: serde_json::json!({"op":"net.fetch","host":host}),
                    },
                );
            }
            at = start + scheme.len();
        }
    }
    for needle in ["fetch(", "sofuu.http.request(", "http.request("] {
        if let Some(pos) = s.find(needle) {
            let has_host = out.iter().any(|f| f.op == "net.fetch");
            if !has_host {
                let host = literal_after(&s, pos + needle.len()).unwrap_or_else(|| "<dynamic>".into());
                let host = host
                    .trim_start_matches("https://")
                    .trim_start_matches("http://")
                    .split('/')
                    .next()
                    .unwrap_or(&host)
                    .to_string();
                push_unique(
                    &mut out,
                    &mut seen,
                    Finding {
                        op: "net.fetch",
                        target: host.clone(),
                        req: serde_json::json!({"op":"net.fetch","host":host}),
                    },
                );
            }
        }
    }

    /* Filesystem APIs → fs.read / fs.write. */
    let read_apis = [
        "readFile(", "readFileSync(", "readFileBytes(", "readdir(", "readdirSync(",
        "listDir(", "exists(", "existsSync(", "stat(", "statSync(",
    ];
    let write_apis = [
        "writeFile(", "writeFileSync(", "appendFile(", "appendFileSync(", "mkdir(",
        "mkdirSync(", "rm(", "rmSync(", "unlink(", "unlinkSync(", "rename(", "renameSync(",
        "copyFile(",
    ];
    for (apis, op) in [(read_apis.as_slice(), "fs.read"), (write_apis.as_slice(), "fs.write")] {
        for api in apis {
            let mut at = 0;
            while let Some(rel) = s[at..].find(api) {
                let pos = at + rel;
                at = pos + api.len();
                /* Only a call that names a filesystem object; JSON.parse and
                 * friends share substrings but not the receiver. */
                let before = &s[..pos];
                let receiver = before.trim_end().rsplit(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')).next().unwrap_or("");
                if !(receiver.ends_with("fs.") || receiver.ends_with("fs") || receiver.contains("sofuu.fs")) {
                    continue;
                }
                let path = literal_after(&s, at).unwrap_or_else(|| "<dynamic>".into());
                let mut req = serde_json::json!({"op": op, "path": path});
                if let Ok(cwd) = std::env::current_dir() {
                    if !path.starts_with('/') && path != "<dynamic>" {
                        req = serde_json::json!({
                            "op": op,
                            "path": cwd.join(&path).to_string_lossy(),
                        });
                    }
                }
                push_unique(&mut out, &mut seen, Finding { op, target: path, req });
            }
        }
    }

    /* Shell execution → shell.run. */
    for needle in ["sofuu.spawn(", "sofuu.exec(", "shell.run("] {
        let mut at = 0;
        while let Some(rel) = s[at..].find(needle) {
            let pos = at + rel;
            at = pos + needle.len();
            let cmd = literal_after(&s, at).unwrap_or_else(|| "<dynamic>".into());
            push_unique(
                &mut out,
                &mut seen,
                Finding {
                    op: "shell.run",
                    target: cmd.clone(),
                    req: serde_json::json!({"op":"shell.run","command":cmd}),
                },
            );
        }
    }

    /* MCP → mcp.connect. */
    if let Some(pos) = s.find("sofuu.mcp.connect(") {
        let server =
            literal_after(&s, pos + "sofuu.mcp.connect(".len()).unwrap_or_else(|| "<dynamic>".into());
        push_unique(
            &mut out,
            &mut seen,
            Finding {
                op: "mcp.connect",
                target: server.clone(),
                req: serde_json::json!({"op":"mcp.connect","server":server}),
            },
        );
    }

    /* Tools → tool.run: TOOLS.<name> and declareTool/defineTool names. */
    let mut at = 0;
    while let Some(rel) = s[at..].find("TOOLS.") {
        let pos = at + rel + "TOOLS.".len();
        let name: String = s[pos..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        at = pos;
        if !name.is_empty() {
            push_unique(
                &mut out,
                &mut seen,
                Finding {
                    op: "tool.run",
                    target: name.clone(),
                    req: serde_json::json!({"op":"tool.run","tool":name}),
                },
            );
        }
    }
    if let Some(pos) = s.find("defineTool(") {
        if let Some(name) = s[pos..].find("name:").and_then(|r| literal_after(&s, pos + r + 5)) {
            push_unique(
                &mut out,
                &mut seen,
                Finding {
                    op: "tool.run",
                    target: name.clone(),
                    req: serde_json::json!({"op":"tool.run","tool":name}),
                },
            );
        }
    }

    /* Environment reads are informational: no policy op gates them, but an
     * env read next to a network call is the exfiltration shape an operator
     * wants to see. */
    let mut at = 0;
    while let Some(rel) = s[at..].find("process.env.") {
        let pos = at + rel + "process.env.".len();
        let name: String = s[pos..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        at = pos;
        if !name.is_empty() && !name.starts_with("NA") {
            push_unique(
                &mut out,
                &mut seen,
                Finding {
                    op: "env.read",
                    target: name.clone(),
                    req: serde_json::json!({"op":"env.read","name":name}),
                },
            );
        }
    }

    out
}

/// The audit shared by `policy check <script>` and `policy dry-run <script>`.
fn policy_audit(args: &[String], path: &str, raw: &str, dry_run: bool) -> i32 {
    let policy = match policy_arg(args) {
        Some(p) => {
            let text = match std::fs::read_to_string(&p) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: cannot read {p}: {e}");
                    return 1;
                }
            };
            match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(v) if v.is_object() => v,
                Ok(_) => {
                    eprintln!("Error: {p} must contain a JSON object");
                    return 1;
                }
                Err(e) => {
                    eprintln!("Error: {p} is not valid JSON: {e}");
                    return 1;
                }
            }
        }
        None => match sofuu_core::rt::policy::layered_policy_from_disk() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Error: {e}");
                return 1;
            }
        },
    };
    let findings = scan_source(raw);
    let want_json = args.iter().any(|a| a == "--json");

    let mut rows: Vec<(String, String, String, String, String)> = Vec::new();
    let mut denied = 0usize;
    for f in &findings {
        let d = sofuu_core::rt::policy::evaluate(&f.req, &policy);
        let dec = d.get("decision").and_then(|v| v.as_str()).unwrap_or("allow").to_string();
        let rule = d.get("rule").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let reason = d.get("reason").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if dec == "deny" {
            denied += 1;
        }
        rows.push((f.op.to_string(), f.target.clone(), dec, rule, reason));
    }

    if want_json {
        let arr: Vec<serde_json::Value> = rows
            .iter()
            .map(|(op, target, dec, rule, reason)| {
                serde_json::json!({"op":op,"target":target,"decision":dec,"rule":rule,"reason":reason})
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr).unwrap_or_else(|_| "[]".into()));
        return if !dry_run && denied > 0 { 1 } else { 0 };
    }

    if dry_run {
        for (op, target, dec, rule, reason) in &rows {
            let tag = match dec.as_str() {
                "deny" => "\x1b[31mDENY \x1b[0m",
                "ask" => "\x1b[33mASK  \x1b[0m",
                _ => "\x1b[32mALLOW\x1b[0m",
            };
            let suffix = if dec == "deny" {
                format!(" — would fail closed ({reason})")
            } else {
                String::new()
            };
            println!("[POLICY DRY-RUN] {tag} {op:<12} -> {target} ({rule}){suffix}");
        }
        println!(
            "\x1b[90m  {} requirement(s), {denied} would be denied. Static audit: the script was NOT executed.\x1b[0m",
            rows.len()
        );
        return 0;
    }

    println!("\x1b[1msofuu policy check\x1b[0m — {path} (static audit)");
    for (op, target, dec, rule, _reason) in &rows {
        let colored = match dec.as_str() {
            "deny" => "\x1b[31mdeny\x1b[0m",
            "ask" => "\x1b[33mask\x1b[0m",
            _ => "allow",
        };
        println!("  {:<12} {:<32} {:<16} {}", op, target, colored, rule);
    }
    let source = if policy_arg(args).is_some() {
        policy_arg(args).unwrap_or_default()
    } else {
        "~/.sofuu/policy.json + .sofuu/policy.json".to_string()
    };
    println!(
        "\x1b[90m  {} requirement(s) · {denied} denied · policy: {source}\x1b[0m",
        rows.len()
    );
    if denied > 0 {
        1
    } else {
        0
    }
}
