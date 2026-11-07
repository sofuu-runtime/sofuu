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
        "init" => {
            let name = match args.get(3) {
                Some(n) => n.trim().to_string(),
                None => {
                    eprintln!("Usage: sofuu mcp init <name>");
                    return 1;
                }
            };
            if name.is_empty() {
                eprintln!("Error: server name is required");
                return 1;
            }
            let clean_name = name
                .chars()
                .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                .collect::<String>();
            let file_name = if clean_name.ends_with(".ts") || clean_name.ends_with(".js") {
                clean_name.clone()
            } else {
                format!("{clean_name}.ts")
            };
            let file_path = std::path::Path::new(&file_name);
            if file_path.exists() {
                eprintln!("Error: file '{}' already exists", file_path.display());
                return 1;
            }
            let template = format!(
r#"import {{ mcp }} from 'sofuu';

// Initialize Model Context Protocol (MCP) server
const server = mcp.serve({{
  name: "{clean_name}",
  version: "1.0.0"
}});

// Sample tool definition
server.tool("echo", {{
  text: "string"
}}, async ({{ text }}) => {{
  return "Echo: " + (text || "");
}});

// Start stdio transport listener
server.start();
"#);
            if let Err(e) = std::fs::write(file_path, template) {
                eprintln!("Error: cannot write file '{}': {e}", file_path.display());
                return 1;
            }
            println!("\x1b[32m✓\x1b[0m created MCP server scaffold in \x1b[1m{}\x1b[0m", file_path.display());
            println!("  Run directly:  sofuu run {}", file_path.display());
            println!("  Add to Sofuu:  sofuu mcp add {} sofuu run {}", clean_name, file_path.display());
            println!("  Test with MCP: sofuu mcp test \"sofuu run {}\"", file_path.display());
            0
        }
        "link" => {
            let target = match args.get(3).map(|s| s.as_str()) {
                Some("claude-desktop" | "claude") => "claude",
                Some("cursor") => "cursor",
                _ => {
                    eprintln!("Usage: sofuu mcp link <claude-desktop|cursor> [--local]");
                    return 1;
                }
            };
            let is_local = args.iter().any(|a| a == "--local");
            let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_else(|_| ".".into());
            let target_path = if target == "claude" {
                #[cfg(target_os = "macos")]
                {
                    std::path::PathBuf::from(&home)
                        .join("Library/Application Support/Claude/claude_desktop_config.json")
                }
                #[cfg(not(target_os = "macos"))]
                {
                    std::path::PathBuf::from(&home)
                        .join(".config/Claude/claude_desktop_config.json")
                }
            } else {
                if is_local || std::path::Path::new(".cursor").is_dir() {
                    std::path::PathBuf::from(".cursor/mcp.json")
                } else {
                    std::path::PathBuf::from(&home).join(".cursor/mcp.json")
                }
            };

            let sofuu_servers = match load_mcp_config() {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error reading Sofuu MCP config: {e}");
                    return 1;
                }
            };
            if sofuu_servers.is_empty() {
                eprintln!("Warning: No MCP servers currently configured in ~/.sofuu/mcp.json");
                eprintln!("Add one first with: sofuu mcp add <name> <command> [args...]");
            }

            let mut root_val: serde_json::Value = if target_path.exists() {
                match std::fs::read_to_string(&target_path) {
                    Ok(content) => {
                        let ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        let backup_path = target_path.with_extension(format!("json.{ts}.backup"));
                        let _ = std::fs::write(&backup_path, &content);
                        match serde_json::from_str(&content) {
                            Ok(v) => v,
                            Err(e) => {
                                eprintln!("Error: Existing config at {} is malformed: {e}", target_path.display());
                                return 1;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error reading {}: {e}", target_path.display());
                        return 1;
                    }
                }
            } else {
                serde_json::json!({})
            };

            if !root_val.is_object() {
                root_val = serde_json::json!({});
            }

            if !root_val.get("mcpServers").map(|v| v.is_object()).unwrap_or(false) {
                root_val["mcpServers"] = serde_json::json!({});
            }

            let mcp_map = root_val.get_mut("mcpServers").and_then(|v| v.as_object_mut()).expect("mcpServers is object");
            let mut linked_count = 0;
            let mut server_names = Vec::new();

            for srv in &sofuu_servers {
                let name = match srv.get("name").and_then(|v| v.as_str()) {
                    Some(n) if !n.trim().is_empty() => n.trim(),
                    _ => continue,
                };
                let cmd = match srv.get("command").and_then(|v| v.as_str()) {
                    Some(c) if !c.trim().is_empty() => c.trim(),
                    _ => continue,
                };
                let mut entry = serde_json::json!({
                    "command": cmd
                });
                if let Some(args_val) = srv.get("args") {
                    if args_val.is_array() {
                        entry["args"] = args_val.clone();
                    }
                }
                if let Some(env_val) = srv.get("env") {
                    if env_val.is_object() {
                        entry["env"] = env_val.clone();
                    }
                }
                mcp_map.insert(name.to_string(), entry);
                linked_count += 1;
                server_names.push(name.to_string());
            }

            if let Some(parent) = target_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            let pretty_json = match serde_json::to_string_pretty(&root_val) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error serializing updated config: {e}");
                    return 1;
                }
            };

            if let Err(e) = std::fs::write(&target_path, pretty_json) {
                eprintln!("Error writing {}: {e}", target_path.display());
                return 1;
            }

            let ide_display = if target == "claude" { "Claude Desktop" } else { "Cursor" };
            println!("\x1b[32m✓\x1b[0m Linked {linked_count} Sofuu MCP server(s) to \x1b[1m{ide_display}\x1b[0m");
            println!("  Config: {}", target_path.display());
            if !server_names.is_empty() {
                println!("  Servers: {}", server_names.join(", "));
            }
            0
        }
        "inspect" => {
            let target = match args.get(3) {
                Some(t) => t.trim().to_string(),
                None => {
                    eprintln!("Usage: sofuu mcp inspect <name | command line | http(s)://url> [--call <tool> [--args <json>]]");
                    return 1;
                }
            };
            if target.is_empty() {
                eprintln!("Error: target is required");
                return 1;
            }
            if let Some(p) = policy_arg(args) {
                if let Err(e) = crate::apply_policy_file(rt, &p) {
                    eprintln!("Error: {e}");
                    return 1;
                }
            }

            let mut call_tool: Option<String> = None;
            let mut call_args: String = "{}".to_string();
            let mut i = 4;
            while i < args.len() {
                if args[i] == "--call" {
                    if let Some(t) = args.get(i + 1) {
                        call_tool = Some(t.clone());
                        i += 2;
                        continue;
                    }
                } else if args[i] == "--args" {
                    if let Some(a) = args.get(i + 1) {
                        call_args = a.clone();
                        i += 2;
                        continue;
                    }
                }
                i += 1;
            }

            let mut resolved_command = target.clone();
            let mut resolved_args: Option<Vec<String>> = None;
            let mut resolved_url: Option<String> = None;
            if let Ok(config_list) = load_mcp_config() {
                for s in config_list {
                    if s.get("name").and_then(|v| v.as_str()) == Some(&target) {
                        if let Some(u) = s.get("url").and_then(|v| v.as_str()) {
                            resolved_url = Some(u.to_string());
                        } else if let Some(c) = s.get("command").and_then(|v| v.as_str()) {
                            resolved_command = c.to_string();
                            if let Some(av) = s.get("args").and_then(|v| v.as_array()) {
                                resolved_args = Some(av.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect());
                            }
                        }
                        break;
                    }
                }
            }

            let connect_spec = if let Some(u) = resolved_url {
                serde_json::json!({
                    "url": u,
                    "transport": "auto"
                })
            } else if let Some(a) = resolved_args {
                serde_json::json!({
                    "command": resolved_command,
                    "args": a
                })
            } else if resolved_command.starts_with("http://") || resolved_command.starts_with("https://") {
                serde_json::json!({
                    "url": resolved_command,
                    "transport": "auto"
                })
            } else {
                serde_json::json!({
                    "command": resolved_command
                })
            };

            let connect_spec_json = serde_json::to_string(&connect_spec).unwrap_or_else(|_| "{}".into());
            let target_display = serde_json::to_string(&target).unwrap_or_else(|_| "\"\"".into());
            let call_tool_json = serde_json::to_string(&call_tool).unwrap_or_else(|_| "null".into());
            let call_args_json = serde_json::to_string(&call_args).unwrap_or_else(|_| "\"{}\"".into());

            let script = format!(
r#"globalThis.__sofuu_cli_rc = 1;
(async function () {{
  var spec = {spec};
  var targetDisp = {target_display};
  var callTool = {call_tool};
  var callArgsStr = {call_args};
  try {{
    var client = await sofuu.mcp.connect(spec);
    var pVer = (typeof client.protocolVersion === 'function') ? String(client.protocolVersion() || 'unknown') : 'unknown';
    var transport = client.transport || (/^https?:\/\//i.test(spec.url || '') ? 'http' : 'stdio');
    var toolsRes = await client.listTools().catch(function () {{ return {{ tools: [] }}; }});
    var tools = (toolsRes && toolsRes.tools) || [];
    var resRes = await client.listResources().catch(function () {{ return null; }});
    var resources = (resRes && resRes.resources) || null;
    var promptsRes = await client.call('prompts/list', {{}}).catch(function () {{ return null; }});
    var prompts = (promptsRes && promptsRes.prompts) || null;

    console.log('\x1b[1;36m┌────────────────────────────────────────────────────────────────────────┐\x1b[0m');
    console.log('\x1b[1;36m│\x1b[0m \x1b[1mModel Context Protocol (MCP) Server Inspector\x1b[0m');
    console.log('\x1b[1;36m├────────────────────────────────────────────────────────────────────────┤\x1b[0m');
    console.log('\x1b[1;36m│\x1b[0m \x1b[33mTarget:\x1b[0m    ' + targetDisp);
    console.log('\x1b[1;36m│\x1b[0m \x1b[33mTransport:\x1b[0m ' + transport + (client.pid && client.pid() ? ' (PID ' + client.pid() + ')' : ''));
    console.log('\x1b[1;36m│\x1b[0m \x1b[33mProtocol:\x1b[0m  ' + pVer);
    console.log('\x1b[1;36m└────────────────────────────────────────────────────────────────────────┘\x1b[0m\n');

    console.log('\x1b[1;32m● Tools (' + tools.length + ')\x1b[0m');
    if (tools.length === 0) {{
      console.log('  (no tools exposed)');
    }} else {{
      for (var i = 0; i < tools.length; i++) {{
        var t = tools[i];
        console.log('  \x1b[1m' + (i + 1) + '. ' + t.name + '\x1b[0m' + (t.description ? ' — ' + t.description : ''));
        var schema = t.inputSchema || {{}};
        var props = schema.properties || {{}};
        var req = schema.required || [];
        var pKeys = Object.keys(props);
        if (pKeys.length > 0) {{
          console.log('     \x1b[2mParameters:\x1b[0m');
          for (var p = 0; p < pKeys.length; p++) {{
            var pk = pKeys[p];
            var pdef = props[pk] || {{}};
            var isReq = req.indexOf(pk) >= 0;
            var typeStr = pdef.type || (pdef.enum ? ('enum[' + pdef.enum.join(',') + ']') : 'any');
            var descStr = pdef.description ? ' (' + pdef.description + ')' : '';
            var reqMark = isReq ? ' \x1b[31m[required]\x1b[0m' : ' \x1b[2m[optional]\x1b[0m';
            console.log('       - \x1b[33m' + pk + '\x1b[0m \x1b[35m<' + typeStr + '>\x1b[0m' + reqMark + descStr);
          }}
        }} else {{
          console.log('     \x1b[2mParameters: (none)\x1b[0m');
        }}
      }}
    }}
    console.log('');

    if (resources !== null) {{
      console.log('\x1b[1;34m● Resources (' + resources.length + ')\x1b[0m');
      for (var r = 0; r < resources.length; r++) {{
        var resObj = resources[r];
        console.log('  - ' + (resObj.name || resObj.uri) + ' \x1b[2m<' + (resObj.mimeType || 'unknown') + '>\x1b[0m (' + resObj.uri + ')');
      }}
      console.log('');
    }}

    if (prompts !== null) {{
      console.log('\x1b[1;35m● Prompt Templates (' + prompts.length + ')\x1b[0m');
      for (var pr = 0; pr < prompts.length; pr++) {{
        var prObj = prompts[pr];
        console.log('  - ' + prObj.name + (prObj.description ? ' — ' + prObj.description : ''));
      }}
      console.log('');
    }}

    if (callTool) {{
      console.log('\x1b[1;33m● Executing Tool Call: ' + callTool + '\x1b[0m');
      var parsedArgs = {{}};
      try {{ parsedArgs = JSON.parse(callArgsStr); }} catch (eA) {{
        console.log('\x1b[31mError: could not parse --args as JSON: ' + (eA.message || eA) + '\x1b[0m');
        globalThis.__sofuu_cli_rc = 1;
        try {{ client.disconnect(); }} catch (eD) {{}}
        return;
      }}
      console.log('  Arguments: ' + JSON.stringify(parsedArgs));
      var startT = Date.now();
      try {{
        var callRes = await client.call('tools/call', {{ name: callTool, arguments: parsedArgs }});
        var dur = Date.now() - startT;
        console.log('\x1b[32m✓ Completed in ' + dur + 'ms:\x1b[0m');
        var resMapped = sofuu.tools.mcpResult(callRes);
        console.log(resMapped);
      }} catch (eCall) {{
        var durFail = Date.now() - startT;
        console.log('\x1b[31m✗ Tool execution failed after ' + durFail + 'ms:\x1b[0m');
        console.log('  ' + (eCall.message || eCall));
      }}
    }}

    globalThis.__sofuu_cli_rc = 0;
    try {{ client.disconnect(); }} catch (eD2) {{}}
  }} catch (e) {{
    console.log('\x1b[31m✗ Failed to inspect MCP server: ' + targetDisp + '\x1b[0m');
    console.log('  ' + String((e && e.message) || e));
    globalThis.__sofuu_cli_rc = 1;
  }}
}})();"#,
                spec = connect_spec_json,
                target_display = target_display,
                call_tool = call_tool_json,
                call_args = call_args_json
            );
            eval_rc(rt, &script)
        }
        "build" => {
            let entry = match args.get(3) {
                Some(e) => e.clone(),
                None => {
                    eprintln!("Usage: sofuu mcp build <entry.ts|js> [-o <output>]");
                    return 1;
                }
            };
            let mut opts = sofuu_core::compiler::CompileOptions::default();
            opts.profile = sofuu_core::compiler::CompileProfile::Micro;
            let mut i = 4;
            while i < args.len() {
                if (args[i] == "-o" || args[i] == "--output") && i + 1 < args.len() {
                    opts.output = Some(args[i + 1].clone());
                    i += 2;
                } else if args[i].starts_with("--output=") {
                    opts.output = Some(args[i]["--output=".len()..].to_string());
                    i += 1;
                } else if args[i] == "--no-strip" {
                    opts.no_strip = true;
                    i += 1;
                } else {
                    i += 1;
                }
            }
            match sofuu_core::compiler::compile(&entry, &opts) {
                Ok(report) => {
                    println!("\x1b[32m✓\x1b[0m Compiled standalone MCP micro-server: \x1b[1m{}\x1b[0m", report.output);
                    println!("  Size:    {} B ({:.2} MiB)", report.total, report.total as f64 / (1024.0 * 1024.0));
                    println!("  Base:    {} (profile: {})", report.base_source, report.profile);
                    println!("  Payload: {} B (bytecode {} B)", report.payload, report.bytecode_stored);
                    println!("Run with:  {}", report.output);
                    0
                }
                Err(e) => {
                    eprintln!("\x1b[31mError:\x1b[0m {e}");
                    1
                }
            }
        }
        _ => {
            eprintln!("Usage: sofuu mcp <list|test|inspect|add|remove|init|build|link>");
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
