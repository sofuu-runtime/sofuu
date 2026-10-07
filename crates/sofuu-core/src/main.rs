// sofuu-core — Rust entrypoint for the Sofuu runtime.
//
// Phase 1: this binary is a drop-in for the C main(). It dispatches the same
// commands (run/eval/repl/chat/bundle/install/add/version/help/...) and
// delegates to the C core through sofuu-ffi. The chat UI is now Rust (chat.rs);
// remaining C subsystems migrate in later phases (see ROADMAP Track D).
//
// NOTE: the C binary still builds via `make c-only`. The Rust build is `make`
// (cargo build --release).

use std::env;
use std::ffi::{CString, c_char, c_int};
use std::process::ExitCode;

use sofuu_ffi::SofuuRuntime;
/// Skip a test that needs the QTSQ codec when this build has none.
///
/// Some binary-side features are wired THROUGH the codec rather than gated on
/// it: `output_archive` hashes records with `qtsq_sha256_hex`, and the session
/// store and chat persistence both call QTSQ-backed helpers. Without the codec
/// those return `None`/`Err`, so a test that `unwrap()`s them fails. They are
/// not testing a degraded path — they test real behaviour that only exists when
/// the codec is linked.
///
/// CI builds QTSQ-free on purpose (the checkout is proprietary), so without
/// this these 23 tests could only ever pass on one machine — a gate that runs
/// in one place is not a gate.
#[cfg(test)]
macro_rules! require_qtsq {
    () => {
        if !sofuu_core::HAS_QTSQ {
            eprintln!(
                "skipping: this build has no QTSQ codec (see sofuu_core::HAS_QTSQ), \
                 and the code under test is wired through it"
            );
            return;
        }
    };
}


mod chat;
mod doctor;
mod output_archive;
mod session;
mod session_store;
mod theme;

const VERSION: &str = "0.2.0-beta";

fn print_banner() {
    println!("\x1b[1;36m  ┌─────────────────────────────────┐");
    println!("  │   ⚡ Sofuu (素風) Runtime        │");
    println!("  │   v{VERSION:<31}│");
    println!("  │   Simple/Pure Wind — AI-Native  │");
    println!("  └─────────────────────────────────┘\x1b[0m");
}

fn print_help() {
    print_banner();
    println!("\n\x1b[1mUsage:\x1b[0m");
    println!("  sofuu                    Start interactive AI chat");
    println!("  sofuu chat               Same interactive AI chat");
    println!("  sofuu chat -m <model>    Chat with a specific model");
    println!("  sofuu chat -p <provider> Chat with a specific provider");
    println!("  sofuu chat -k <key>      Set the API key for this session");
    println!("  sofuu chat --base-url <u> Override the provider endpoint URL");
    println!("  sofuu chat --ctx-window <n> Context window in tokens (0 = provider default, max 4M)");
    println!("  sofuu chat --max-output <n> Max output tokens per response (0 = provider default, max 384k)");
    println!("  sofuu repl               JavaScript evaluation REPL");
    println!("  sofuu run <file.js>      Run a JavaScript file");
    println!("  sofuu run <file.ts>      Run a TypeScript file");
    println!("  sofuu eval \"<code>\"      Evaluate JS inline");
    println!("  sofuu bundle <entry.js>  Bundle to single file");
    println!("  sofuu bundle <entry.js> -o <out.js>");
    println!("  sofuu install            Install from package.json");
    println!("  sofuu add <pkg>          Install npm package");
    println!("  sofuu agent list         List agent definitions (~/.sofuu/agents/*.js)");
    println!("  sofuu agent run <name> \"task\"  Run one agent headless");
    println!("  sofuu agent run <name> \"task\" --json   …with a JSON result");
    println!("  sofuu serve --brain <path>   Serve the brain over HTTP (--port, --host, --token)");
    println!("  sofuu doctor             Health check: QTSQ, brain round-trip, ctx resolution");
    println!("  sofuu version            Print version and exit");
    println!("  sofuu licenses           Print open-source licenses");
    println!("  sofuu help               Print this help");
    println!("  sofuu session list       List active sessions on this project");
    println!("  sofuu session show <id>  Show one session's full context");
    println!("  sofuu session inspect <id>  Inspect storage without payload text");
    println!("  sofuu session repair <id>   Rebuild a damaged segmented manifest");
    println!("  sofuu session migrate <id>  Migrate one legacy flat session");
    println!("  sofuu session storage-stats <id>  Show segment/storage counters");
    println!("  sofuu session prune      Remove ended sessions older than 7 days");
    println!("  sofuu outputs list       List timestamped model/tool outputs");
    println!("  sofuu outputs show <id>  Decode one archived output");
    println!("  sofuu outputs search <text>  Search archived output summaries/content");
    println!("  sofuu outputs context <task>  Build bounded historical context");
    println!("  sofuu outputs stats      Show archive counters and diagnostics");
    println!("  sofuu outputs rebuild-index  Rebuild output metadata index");
    println!("  sofuu outputs prune      Dry-run retention candidates (use --apply to delete)");
    println!("\n\x1b[1mChat commands (inside the chat UI):\x1b[0m");
    /* P3 (AUDIT-2026-09-07): /models was deliberately removed (unknown
     * command — the model list lives in /provider) — stop advertising it. */
    println!("  /help  /providers  /model <n>  /provider <n>");
    println!("  /effort <lvl>  /compact  /clear  /brain  /exit");
    println!("  /sessions | /context [id] | /work <desc> | /done | /note <msg> | /notify <msg>");
    println!("\nSession mesh: `sofuu chat` syncs with other sessions on the SAME project");
    println!("in real time (tasks, notes, critical notices, shared context).");
    println!("Disable with `--sync off`. Main session data uses bounded .store QTSQ segments; legacy flat .qtsq files remain readable.");
    println!("\n\x1b[90mBuilt with Rust + QuickJS + C + QTSQ | MIT License (QTSQ proprietary)\x1b[0m\n");
}

fn print_version() {
    println!("sofuu {VERSION}");
    println!("quickjs 2024-01-13");
    let os = if cfg!(target_os = "macos") {
        "macOS"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "windows") {
        "Windows"
    } else {
        "unknown"
    };
    let arch = if cfg!(target_arch = "aarch64") {
        " arm64"
    } else if cfg!(target_arch = "x86_64") {
        " x86_64"
    } else {
        ""
    };
    println!("platform: {os}{arch}");
}

/// `sofuu eval "<code>"` — evaluate a JS string.
fn cmd_eval(rt: &SofuuRuntime, code: &str) -> i32 {
    rt.eval_string(code, "<eval>")
}

/// `sofuu agent list` / `sofuu agent run <name> <task…> [--json]` —
/// the headless agent surface (PLAN-AGENTS A8.2). Everything runs in the
/// engine's shipped agent runtime (src/js/agent.js); this is just CLI
/// sugar: load ~/.sofuu/agents/*.js, then list or one-shot run.
fn cmd_agent(rt: &SofuuRuntime, args: &[String]) -> i32 {
    if args.len() < 3 {
        eprintln!("\x1b[1mUsage:\x1b[0m");
        eprintln!("  sofuu agent list");
        eprintln!("  sofuu agent run <name> <task…> [--json]");
        return 1;
    }
    match args[2].as_str() {
        "list" | "ls" => {
            let code = r#"(async function () {
  const ld = await sofuu.agent.loadDir();
  for (const b of (ld && ld.broken) || []) console.log('⚠ ' + b.file + ': ' + b.error);
  const list = sofuu.agent.list();
  if (!list.length) {
    console.log('(no agents defined — add one at ~/.sofuu/agents/<name>.js)');
    return;
  }
  for (const a of list) {
    console.log(a.name +
      (a.model ? ' · ' + a.model : '') +
      ' · memory ' + a.memory +
      ' · ' + a.tools + ' tool' + (a.tools === 1 ? '' : 's') +
      (a.mcpServers ? ' · ' + a.mcpServers + ' mcp' : '') +
      ((a.agents && a.agents.length) ? ' · delegates: ' + a.agents.join(', ') : ''));
    if (a.system) console.log('    ' + a.system);
  }
})().catch(function (e) { console.error(String((e && e.message) || e)); process.exit(1); });"#;
            rt.eval_string(code, "<agent-list>")
        }
        "run" => {
            // The headless agent uses the same native archive bridge as chat,
            // so one-shot runs remain inspectable without starting the chat UI.
            chat::register_bridge(rt);
            if args.len() < 5 {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu agent run' requires <name> and a task");
                eprintln!("Usage: sofuu agent run <name> <task…> [--json]");
                return 1;
            }
            let name = args[3].clone();
            let mut json = false;
            let mut parts: Vec<String> = Vec::new();
            for a in &args[4..] {
                if a == "--json" {
                    json = true;
                } else {
                    parts.push(a.clone());
                }
            }
            let task = parts.join(" ");
            if task.is_empty() {
                eprintln!("\x1b[31mError:\x1b[0m task must not be empty");
                return 1;
            }
            let name_js = serde_json::to_string(&name).unwrap_or_else(|_| "\"\"".into());
            let task_js = serde_json::to_string(&task).unwrap_or_else(|_| "\"\"".into());
            let on_step = if json {
                // Keep stdout pure JSON; progress lines go to stderr.
                r#"onStep: function (e) {
                    if (e.kind === 'tool' || e.kind === 'delegate') {
                      console.error('  ⏺ ' + e.kind + ' ' + (e.payload && (e.payload.name || e.payload.agent)) + '()');
                    }
                  },"#
            } else {
                /* P3 (AUDIT-2026-09-07): raw tool args / delegate task text
                 * can carry secrets into stderr scrollback — print the name
                 * and arg KEYS only (matching --json's name-only lines). */
                r#"onStep: function (e) {
                    if (e.kind === 'tool') {
                      var keys = '';
                      try { keys = Object.keys(JSON.parse((e.payload && e.payload.args) || '{}')).join(', '); } catch (e2) {}
                      console.error('  ⏺ tool ' + (e.payload && e.payload.name) + '(' + keys + ')');
                    } else if (e.kind === 'delegate') {
                      console.error('  ⏺ agent ' + (e.payload && e.payload.agent) + '()');
                    }
                  },"#
            };
            let print = if json {
                r#"console.log(JSON.stringify({
                  answer: r.answer, steps: r.steps, stopped: r.stopped,
                  usage: r.usage, subRuns: (r.subRuns || []).length }, null, 2));"#
            } else {
                r#"console.log('');
                  console.log(r.answer);
                  console.log('\n · ' + r.steps + ' steps · ' + r.usage.llmCalls + ' llm calls · ' +
                              r.usage.toolCalls + ' tool calls · ' + r.usage.promptTokens + '→' +
                              r.usage.completionTokens + ' tk' +
                              (r.stopped ? ' · stopped: ' + r.stopped : ''));"#
            };
            let code = format!(
                r#"(async function () {{
  const ld = await sofuu.agent.loadDir();
  for (const b of (ld && ld.broken) || []) console.error('⚠ ' + b.file + ': ' + b.error);
  function archiveAgent(kind, status, source, content, metadata) {{
    if (content === undefined || content === null || !String(content).trim()) return;
    try {{
      __chat_archive_write(JSON.stringify({{
        turn: 1, attempt: 1, kind: kind, status: status, source: source,
        content: String(content), metadata: metadata || {{}}, redact: true
      }}));
    }} catch (_) {{}}
  }}
  try {{
    const r = await sofuu.agent.run({name_js}, {task_js}, {{
      {on_step}
    }});
    if (r && r.answer && String(r.answer).trim()) {{
      archiveAgent(r.stopped ? 'partial' : 'final',
                   r.stopped === 'cancelled' ? 'cancelled' : (r.stopped ? 'partial' : 'complete'),
                   'assistant', r.answer, {{ headless: true, steps: r.steps, stopped: r.stopped || null }});
    }} else {{
      archiveAgent('error', 'failed', 'runtime', 'agent returned no response',
                   {{ headless: true, error_code: 'empty_response' }});
    }}
    {print}
  }} catch (e) {{
    archiveAgent('error', 'failed', 'runtime', String((e && e.message) || e),
                 {{ headless: true, error_code: 'runtime' }});
    throw e;
  }}
}})().catch(function (e) {{ console.error(String((e && e.message) || e)); process.exit(1); }});"#
            );
            rt.eval_string(&code, "<agent-run>")
        }
        other => {
            eprintln!("\x1b[31mError:\x1b[0m unknown agent subcommand '{other}' (list | run)");
            1
        }
    }
}

/// `sofuu run <file>` — run a JS/TS file.
fn cmd_run(rt: &SofuuRuntime, path: &str) -> i32 {
    rt.eval_file(path)
}

/// `sofuu repl` — JavaScript evaluation REPL (delegates to C for now).
fn cmd_repl(rt: &SofuuRuntime) -> i32 {
    // Phase 0: delegate to the C REPL via the FFI. We eval a script that
    // invokes the C sofuu_repl through the engine — but sofuu_repl is a C
    // function not exposed via the runtime API, so for Phase 0 we run the
    // REPL's logic by calling the C entry through a small shim below.
    repl_shim(rt)
}

// ── REPL shim ────────────────────────────────────────────────────
// Phase 0: the C REPL (src/repl/repl.c) is not exposed via sofuu.h, so we
// re-implement a minimal REPL loop here in Rust, calling into the C engine
// for each line via `eval_repl`. This keeps `sofuu repl` working while the
// full REPL UI is ported. (The C `sofuu repl` binary still exists via
// `make c-only`; this shim will be replaced by the real port.)

use std::io::{self, BufRead, Write};

fn repl_shim(rt: &SofuuRuntime) -> i32 {
    println!("\n\x1b[1;36m  ⚡ Sofuu (素風) REPL\x1b[0m  \x1b[2mv{VERSION}\x1b[0m");
    println!("\x1b[2m  Type .help for commands, Ctrl-D to exit\x1b[0m\n");

    let stdin = io::stdin();
    let mut accum = String::new();

    loop {
        // Multi-line continuation vs fresh prompt
        if accum.is_empty() {
            print!("\x1b[1;36m> \x1b[0m");
        } else {
            print!("\x1b[2m… \x1b[0m");
        }
        io::stdout().flush().ok();

        let mut line = String::new();
        let n = stdin.lock().read_line(&mut line).unwrap_or(0);
        if n == 0 {
            println!();
            break; // Ctrl-D / EOF
        }
        let line = line.trim_end();
        if line.is_empty() && accum.is_empty() {
            continue;
        }

        // Dot-commands
        if accum.is_empty() && line.starts_with('.') {
            match line {
                ".exit" | ".quit" => {
                    println!("\n  Bye! 👋\n");
                    break;
                }
                ".help" => {
                    println!("\n\x1b[1m  Commands:\x1b[0m");
                    println!("    .help    — this help");
                    println!("    .clear   — clear screen");
                    println!("    .exit    — quit REPL");
                    println!("    .version — Sofuu version\n");
                    continue;
                }
                ".clear" => {
                    print!("\x1b[2J\x1b[H");
                    io::stdout().flush().ok();
                    println!("\x1b[1;36m  ⚡ Sofuu (素風) REPL\x1b[0m  \x1b[2mv{VERSION}\x1b[0m\n");
                    continue;
                }
                ".version" => {
                    println!("  sofuu {VERSION}\n");
                    continue;
                }
                _ => {
                    println!("\x1b[31m  ? Unknown: {line}\x1b[0m\n");
                    continue;
                }
            }
        }

        // Accumulate multi-line input (simple brace balance)
        if !accum.is_empty() {
            accum.push('\n');
        }
        accum.push_str(line);

        // Bound the accumulator (the C REPL it replaced capped at 64 KiB too)
        // so a giant paste cannot grow memory without limit.
        if accum.len() > 65536 {
            eprintln!("\x1b[31m  ? Input exceeds 64 KiB — discarded.\x1b[0m");
            accum.clear();
            continue;
        }

        if !needs_more(&accum) {
            let out = rt.eval_repl(&accum);
            accum.clear();
            if let Some(s) = out {
                if !(s.contains("undefined") && s.len() < 30) {
                    println!("{s}");
                }
            }
            println!();
        }
    }
    0
}

/// Simple multi-line detection — balanced braces/parens/brackets/templates.
fn needs_more(s: &str) -> bool {
    let mut br = 0i32;
    let mut par = 0i32;
    let mut sqb = 0i32;
    let mut sq = false;
    let mut dq = false;
    let mut tmpl = false;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if (sq || dq) && c == '\\' {
            i += 1;
        } else if c == '\'' && !dq && !tmpl {
            sq = !sq;
        } else if c == '"' && !sq && !tmpl {
            dq = !dq;
        } else if c == '`' && !sq && !dq {
            tmpl = !tmpl;
        } else if !(sq || dq || tmpl) {
            match c {
                '{' => br += 1,
                '}' => br -= 1,
                '(' => par += 1,
                ')' => par -= 1,
                '[' => sqb += 1,
                ']' => sqb -= 1,
                '/' => {
                    if i + 1 < chars.len() && chars[i + 1] == '/' {
                        break;
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    br > 0 || par > 0 || sqb > 0 || tmpl
}

/// The value of a value-taking CLI flag — unless the next token is itself
/// flag-shaped. P3 (AUDIT-2026-09-07): `sofuu chat -m --brain` used to set
/// model="--brain" and silently swallow the --brain flag.
fn value_arg(args: &[String], i: usize) -> Option<String> {
    let v = args.get(i + 1)?;
    if v.starts_with('-') && v.len() > 1 {
        None
    } else {
        Some(v.clone())
    }
}

/// `sofuu chat` — interactive chat UI (Rust, Phase 1).
fn cmd_chat(rt: &SofuuRuntime, args: &[String]) -> i32 {
    let mut cfg = chat::ChatConfig::load();
    // CLI overrides: -m/--model, -p/--provider, --effort/--think, --brain,
    // --sync on|off, -k/--apikey, --base-url, --ctx-window, --max-output.
    // P3 (AUDIT-2026-09-07): value flags parse through value_arg so a
    // following flag is never eaten as the value; missing values error out.
    let mut i = 2;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-m" | "--model" => match value_arg(args, i) {
                Some(v) => {
                    cfg.model = v;
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {a} needs a value");
                    return 1;
                }
            },
            "-p" | "--provider" => match value_arg(args, i) {
                /* P2-15 (AUDIT-2026-09-01): a typo'd provider used to be
                 * accepted silently and surface as a confusing turn-time
                 * request error. Validate against the wire profiles and
                 * the configured provider names; custom base_url implies
                 * openai-compat, which the profile "openai" covers. */
                Some(p) => match p.as_str() {
                    "openai" | "anthropic" | "local" => {
                        cfg.provider = p;
                        i += 2;
                    }
                    _ => {
                        let known = chat::ChatConfig::load()
                            .providers
                            .iter()
                            .any(|e| e.name == p);
                        if known {
                            cfg.provider = p;
                            i += 2;
                        } else {
                            eprintln!(
                                "\x1b[31mError:\x1b[0m unknown provider '{p}' (known: openai, anthropic, local)."
                            );
                            eprintln!("  Add it in the TUI (/provider) or use --base-url for a custom endpoint.");
                            return 1;
                        }
                    }
                },
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {a} needs a value");
                    return 1;
                }
            },
            "--effort" | "--think" => match value_arg(args, i) {
                Some(v) => {
                    cfg.effort = v;
                    i += 2;
                }
                None => {
                    if a == "--think" {
                        // bare --think → effort high (a following flag is
                        // NOT its value — P3)
                        cfg.effort = "high".into();
                        i += 1;
                    } else {
                        eprintln!("\x1b[31mError:\x1b[0m {a} needs a value");
                        return 1;
                    }
                }
            },
            "--brain" => {
                cfg.brain = true;
                i += 1;
            }
            "--sync" => {
                /* P3 (AUDIT-2026-09-07): only "on"/"off" are values — the
                 * old guard consumed ANY following token (`--sync --brain`
                 * set sync=false and ate the flag). Bare `--sync` = on
                 * (matches "Disable with --sync off"). */
                match args.get(i + 1).map(|s| s.as_str()) {
                    Some("on") => {
                        cfg.sync = true;
                        i += 2;
                    }
                    Some("off") => {
                        cfg.sync = false;
                        i += 2;
                    }
                    Some(v) if !v.starts_with('-') => {
                        eprintln!("\x1b[33mWarning:\x1b[0m --sync wants on|off (got '{v}'); treating as on");
                        cfg.sync = true;
                        i += 2;
                    }
                    _ => {
                        cfg.sync = true;
                        i += 1;
                    }
                }
            }
            "--no-sync" => {
                cfg.sync = false;
                i += 1;
            }
            "-k" | "--apikey" => match value_arg(args, i) {
                Some(v) => {
                    /* P3 (AUDIT-2026-09-07): the key sits in argv (readable
                     * by any local process via `ps`) and the old path let it
                     * persist into config.json. Warn, honor it for this
                     * session, and mark it CLI-only — save() never writes
                     * it (see ChatConfig::api_key_from_cli). */
                    eprintln!(
                        "\x1b[33mWarning:\x1b[0m -k/--apikey exposes the key in the process list; prefer the keychain, config.json, or SOFUU_API_KEY."
                    );
                    cfg.api_key = v;
                    cfg.api_key_from_cli = true;
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {a} needs a value");
                    return 1;
                }
            },
            "--base-url" => match value_arg(args, i) {
                Some(v) => {
                    cfg.base_url = v;
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {a} needs a value");
                    return 1;
                }
            },
            "--ctx-window" | "--ctx" => match value_arg(args, i) {
                Some(v) => {
                    if let Some(n) = chat::parse_token_count(&v) {
                        if n > 4_194_304 {
                            eprintln!(
                                "\x1b[33mWarning:\x1b[0m --ctx-window must be 0–4194304 (accepts 1m, 128k); ignoring '{v}'"
                            );
                        } else {
                            cfg.ctx_window = n;
                            cfg.ctx_window_explicit = n > 0;
                        }
                    } else {
                        eprintln!("\x1b[33mWarning:\x1b[0m --ctx-window wants a number (1048576, 1m, 128k); ignoring '{v}'");
                    }
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {a} needs a value");
                    return 1;
                }
            },
            "--max-output" | "--maxout" | "--max-tokens" => match value_arg(args, i) {
                Some(v) => {
                    if let Some(n) = chat::parse_token_count(&v) {
                        if n > 384_000 {
                            eprintln!(
                                "\x1b[33mWarning:\x1b[0m --max-output must be 0–384000 (0 = provider default); ignoring '{v}'"
                            );
                        } else {
                            cfg.max_output = n;
                        }
                    } else {
                        eprintln!("\x1b[33mWarning:\x1b[0m --max-output wants a number (65536, 64k); ignoring '{v}'");
                    }
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {a} needs a value");
                    return 1;
                }
            },
            _ => i += 1,
        }
    }
    chat::run_chat(rt, cfg)
}

/// `sofuu session list|show|inspect|repair|migrate|storage-stats|prune` —
/// inspect and maintain the project's session mesh
/// without starting a chat (no runtime needed).
fn cmd_session(args: &[String]) -> i32 {
    let Some(project) = session::project_root() else {
        eprintln!("\x1b[31mError:\x1b[0m cannot determine project root (cwd?)");
        return 1;
    };
    println!("\x1b[2mproject: {}\x1b[0m\n", project.display());

    let sub = args.get(2).map(|s| s.as_str()).unwrap_or("list");
    match sub {
        "list" | "ls" | "--list" => session::cmd_list(&project),
        "show" | "info" => {
            let Some(id) = args.get(3) else {
                println!("  Usage: sofuu session show <id>\n");
                return 1;
            };
            session::cmd_show(&project, id)
        }
        "inspect" => {
            let Some(id) = args.get(3) else {
                println!("  Usage: sofuu session inspect <id>\n");
                return 1;
            };
            session::cmd_inspect(&project, id)
        }
        "repair" => {
            let Some(id) = args.get(3) else {
                println!("  Usage: sofuu session repair <id>\n");
                return 1;
            };
            session::cmd_repair(&project, id)
        }
        "migrate" => {
            let Some(id) = args.get(3) else {
                println!("  Usage: sofuu session migrate <id>\n");
                return 1;
            };
            session::cmd_migrate(&project, id)
        }
        "storage-stats" | "stats" => {
            let Some(id) = args.get(3) else {
                println!("  Usage: sofuu session storage-stats <id>\n");
                return 1;
            };
            session::cmd_storage_stats(&project, id)
        }
        "prune" | "clean" => session::cmd_prune(&project),
        other => {
            eprintln!("\x1b[31mError:\x1b[0m unknown session command '{other}' (list|show|inspect|repair|migrate|storage-stats|prune)");
            1
        }
    }
}

/// `sofuu outputs ...` — inspect and maintain the main runtime's project-local
/// timestamped output archive without starting a chat or desktop host.
fn cmd_outputs(args: &[String]) -> i32 {
    let Some(project) = session::project_root() else {
        eprintln!("\x1b[31mError:\x1b[0m cannot determine project root (cwd?)");
        return 1;
    };
    println!("\x1b[2mproject: {}\x1b[0m\n", project.display());
    let policy = chat::ChatConfig::load().archive_policy();
    output_archive::cli(&project, args, &policy)
}

/// `sofuu serve --brain <path> --port <n> --host <h> --token <t>` —
/// F11: serve the brain over HTTP. The server runs as a shipped JS file
/// through the normal runtime — endpoints: GET /health, POST /remember,
/// GET /recall, POST /share. Auth: Bearer token (generated or supplied).
fn cmd_serve(rt: &SofuuRuntime, args: &[String]) -> i32 {
    let mut brain_path = String::new();
    let mut port: u16 = 7707;
    let mut host = "127.0.0.1".to_string();
    // Prefer env over argv: --token is visible via `ps`. SOFUU_SERVE_TOKEN
    // wins when set; --token stays for scripts but prints a warning.
    let mut token = std::env::var("SOFUU_SERVE_TOKEN").unwrap_or_default();
    let mut token_from_argv = false;

    let mut i = 2;
    while i < args.len() {
        /* P3 (AUDIT-2026-09-07): value flags refuse a flag-shaped token as
         * their value (the old loop swallowed whatever followed —
         * `--brain --port 9` silently made the brain path "--port"), a bad
         * --port warns instead of silently keeping the default, and --token
         * no longer overwrites SOFUU_SERVE_TOKEN (the comment above — and
         * the safer precedence — say env wins). */
        match args[i].as_str() {
            "--brain" | "-b" => match value_arg(args, i) {
                Some(v) => {
                    brain_path = v;
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {} needs a value", args[i]);
                    return 1;
                }
            },
            "--port" | "-p" => match value_arg(args, i) {
                Some(v) => {
                    if let Ok(p) = v.parse::<u16>() {
                        port = p;
                    } else {
                        eprintln!("\x1b[33mWarning:\x1b[0m --port wants 0-65535; keeping default {port}");
                    }
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {} needs a value", args[i]);
                    return 1;
                }
            },
            "--host" => match value_arg(args, i) {
                Some(v) => {
                    host = v;
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {} needs a value", args[i]);
                    return 1;
                }
            },
            "--token" | "-t" => match value_arg(args, i) {
                Some(v) => {
                    if token.is_empty() {
                        token = v;
                        token_from_argv = true;
                    } else {
                        eprintln!("\x1b[33mWarning:\x1b[0m --token ignored: SOFUU_SERVE_TOKEN is set (env wins).");
                    }
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {} needs a value", args[i]);
                    return 1;
                }
            },
            _ => i += 1,
        }
    }

    /// HOME-derived paths must stay inside the home directory — reject any
    /// `..` component so a hostile HOME cannot redirect writes elsewhere.
    /// Both separators: Windows USERPROFILE paths are backslash-joined.
    fn home_checked() -> Option<String> {
        let home = sofuu_core::embed_config::home_dir()?;
        if home.is_empty() || home.split(['/', '\\']).any(|seg| seg == "..") {
            return None;
        }
        Some(home)
    }

    // Default brain path.
    if brain_path.is_empty() {
        let home = home_checked().unwrap_or_else(|| ".".into());
        brain_path = format!("{home}/.sofuu_brain.qtsq");
    }

    // Generate a token if none supplied. 16 bytes from /dev/urandom — a
    // timestamp-derived token (the old scheme) had ~10^6/second of entropy
    // and was brute-forceable by any local process that could bound the
    // server start time.
    let mut auto_token = false;
    if token.is_empty() {
        auto_token = true;
        let mut entropy = [0u8; 16];
        let read = std::fs::File::open("/dev/urandom")
            .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut entropy));
        if let Err(e) = read {
            eprintln!("\x1b[31mError:\x1b[0m cannot read /dev/urandom to generate a token: {e}");
            return 1;
        }
        token = format!("sofuu-{}", entropy.iter().map(|b| format!("{b:02x}")).collect::<String>());
        // Persist the token to ~/.sofuu/serve_token. P2-10: create with
        // 0600 from the first write (write-then-chmod left a world-
        // readable window).
        if let Some(home) = home_checked() {
            let dir = format!("{home}/.sofuu");
            let _ = std::fs::create_dir_all(&dir);
            let token_path = format!("{dir}/serve_token");
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                use std::io::Write;
                let _ = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .mode(0o600)
                    .open(&token_path)
                    .and_then(|mut f| f.write_all(token.as_bytes()));
            }
            #[cfg(not(unix))]
            {
                let _ = std::fs::write(&token_path, &token);
            }
        }
    }

    // Remote binding requires a user-chosen token (security). The old
    // prefix check ("sofuu-") let any --token sofuu-anything through.
    if (host != "127.0.0.1" && host != "localhost") && auto_token {
        eprintln!("\x1b[31mError:\x1b[0m remote binding requires an explicit --token");
        eprintln!("  Use: sofuu serve --host {} --token <your-secret>\n", host);
        eprintln!("  Prefer: SOFUU_SERVE_TOKEN=<secret> sofuu serve --host {}", host);
        return 1;
    }
    if token_from_argv {
        eprintln!("\x1b[33m  ⚠ --token is visible via `ps`; prefer SOFUU_SERVE_TOKEN env.\x1b[0m");
    }

    eprintln!("\x1b[1;36m  ┌─────────────────────────────────┐");
    eprintln!("  │   ⚡ Sofuu Brain Server          │");
    eprintln!("  │   v{:<31}│", VERSION);
    eprintln!("  └─────────────────────────────────┘\x1b[0m");
    eprintln!("\x1b[2m  brain: {}\x1b[0m", brain_path);
    eprintln!("\x1b[2m  listen: http://{}:{}\x1b[0m", host, port);
    // Token is a bearer secret: never print it to stderr (scrollback/CI
    // logs). Point at the persisted file or env instead.
    if auto_token {
        eprintln!("\x1b[2m  token: (auto-generated, stored in ~/.sofuu/serve_token with 0600)\x1b[0m");
    } else {
        eprintln!("\x1b[2m  token: (from --token / SOFUU_SERVE_TOKEN)\x1b[0m");
    }
    eprintln!("\x1b[2m  endpoints:\x1b[0m");
    eprintln!("\x1b[2m    GET  /health\x1b[0m");
    eprintln!("\x1b[2m    POST /remember  {{text, role?, tier?}}\x1b[0m");
    eprintln!("\x1b[2m    GET  /recall?q=…&k=…\x1b[0m");
    eprintln!("\x1b[2m    POST /share     (export card)\x1b[0m");
    eprintln!();

    let brain_path_js = serde_json::to_string(&brain_path).unwrap_or_else(|_| "\"\"".into());
    let token_js = serde_json::to_string(&token).unwrap_or_else(|_| "\"\"".into());
    let host_js = serde_json::to_string(&host).unwrap_or_else(|_| "\"\"".into());

    let serve_js = format!(r#"
(async function () {{
  var g = globalThis;
  if (!g.sofuu) g.sofuu = {{}};
  var brainPath = {brain_path_js};
  var token = {token_js};
  var host = {host_js};
  var port = {port};
  var brain = null;
  if (sofuu.memory && typeof sofuu.memory.open === 'function') {{
    try {{ brain = sofuu.memory.open(brainPath, 768); }} catch (e) {{ console.error('brain open: ' + e); }}
  }}
  /* Pin the brain + server to globals: their JS objects' reachability
   * controls native lifetime (the finalizer closes the handle) — locals
   * inside this async IIFE die at resolution and would kill the server. */
  g.__sofuu_serve_brain = brain;
  g.__sofuu_serve_server = null;
  var server = sofuu.http.createServer(async function (req, res) {{
    /* Constant-time comparison — a plain !== compare leaks the matching
     * prefix length to a local timing attacker. */
    function ctEq(a, b) {{
      if (typeof a !== 'string' || a.length !== b.length) return false;
      var d = 0;
      for (var i = 0; i < b.length; i++) d |= (a.charCodeAt(i) ^ b.charCodeAt(i));
      return d === 0;
    }}
    var auth = (req.headers && req.headers.authorization) || '';
    if (!ctEq(auth, 'Bearer ' + token)) {{
      res.writeHead(401, {{ 'Content-Type': 'application/json' }});
      res.end(JSON.stringify({{ ok: false, error: 'unauthorized' }}));
      return;
    }}
    var url = req.url || '/';
    var path = url.split('?')[0];
    if (req.method === 'GET' && path === '/health') {{
      var count = brain ? brain.count() : 0;
      res.writeHead(200, {{ 'Content-Type': 'application/json' }});
      res.end(JSON.stringify({{ ok: true, memories: count }}));
      return;
    }}
    if (req.method === 'POST' && path === '/remember') {{
      /* req.body is a fully-buffered string — iterating it as chunks would
       * concatenate one character per round (O(n^2)). */
      var body = typeof req.body === 'string' ? req.body : '';
      try {{
        var data = JSON.parse(body || '{{}}');
        if (!data.text) {{
          res.writeHead(400, {{ 'Content-Type': 'application/json' }});
          res.end(JSON.stringify({{ ok: false, error: 'text required' }}));
          return;
        }}
        if (!brain) {{
          res.writeHead(503, {{ 'Content-Type': 'application/json' }});
          res.end(JSON.stringify({{ ok: false, error: 'brain not available' }}));
          return;
        }}
        var vec = sofuu.ai.embedLocal(data.text);
        var role = data.role || 'user';
        brain.remember(new Float32Array(vec), data.text, role, 0);
        brain.flush();
        res.writeHead(200, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: true, memories: brain.count() }}));
      }} catch (e) {{
        res.writeHead(500, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: String(e.message || e) }}));
      }}
      return;
    }}
    if (req.method === 'GET' && path === '/recall') {{
      var qs = {{}};
      var q = (url.split('?')[1] || '').split('&');
      for (var i = 0; i < q.length; i++) {{
        var kv = q[i].split('=');
        if (kv.length === 2) qs[decodeURIComponent(kv[0])] = decodeURIComponent(kv[1]);
      }}
      if (!qs.q) {{
        res.writeHead(400, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: 'q parameter required' }}));
        return;
      }}
      if (!brain) {{
        res.writeHead(503, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: 'brain not available' }}));
        return;
      }}
      var vec = sofuu.ai.embedLocal(qs.q);
      var k = parseInt(qs.k || '5', 10);
      var results = brain.recall(new Float32Array(vec), k);
      res.writeHead(200, {{ 'Content-Type': 'application/json' }});
      res.end(JSON.stringify({{ ok: true, results: results }}));
      return;
    }}
    if (req.method === 'POST' && path === '/share') {{
      res.writeHead(200, {{ 'Content-Type': 'application/json' }});
      res.end(JSON.stringify({{ ok: true, message: 'share endpoint not yet implemented' }}));
      return;
    }}
    res.writeHead(404, {{ 'Content-Type': 'application/json' }});
    res.end(JSON.stringify({{ ok: false, error: 'not found' }}));
  }});
  g.__sofuu_serve_server = server;
  server.listen(port, host);
  console.log('brain server listening on http://' + host + ':' + port);
}})().catch(function (e) {{ console.error(String((e && e.message) || e)); process.exit(1); }});
"#);

    rt.eval_string(&serve_js, "<serve>")
}

/// `sofuu bundle <entry> [-o <out>]` — bundle to a single file.
/// Pure Rust pipeline (sofuu_core::bundler); npm resolution via the Rust
/// walk-up resolver. The C bundler remains only for `make c-only`.
fn cmd_bundle(_rt: &SofuuRuntime, args: &[String]) -> i32 {
    if args.len() < 3 {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu bundle' requires an entry file");
        eprintln!("Usage: sofuu bundle <entry.js> [-o <out.js>]");
        return 1;
    }
    let entry = std::path::PathBuf::from(&args[2]);
    let mut output = "bundle.js".to_string();
    let mut i = 3;
    /* P3-6 (AUDIT-2026-09-01): a TRAILING `-o` with no value used to be
     * skipped by `i + 1 < len` and silently wrote bundle.js. Error loudly. */
    if args.len() > 3 && (args[args.len() - 1] == "-o" || args[args.len() - 1] == "--out") {
        eprintln!("\x1b[31mError:\x1b[0m -o requires an output path");
        eprintln!("Usage: sofuu bundle <entry.js> [-o <out.js>]");
        return 1;
    }
    while i + 1 < args.len() {
        if args[i] == "-o" || args[i] == "--out" {
            output = args[i + 1].clone();
        }
        i += 1;
    }

    match sofuu_core::bundler::bundle(&entry, &sofuu_core::npm::npm_resolve) {
        Ok(src) => {
            if let Err(e) = std::fs::write(&output, &src) {
                eprintln!("\x1b[31mError:\x1b[0m cannot write {}: {}", output, e);
                return 1;
            }
            println!("\x1b[1m⚡ sofuu bundle\x1b[0m");
            println!("  Entry: {}", entry.display());
            println!(
                "\x1b[32m✓\x1b[0m Bundle written: \x1b[1m{}\x1b[0m ({} B)",
                output,
                src.len()
            );
            println!("\x1b[90mRun with: sofuu run {}\x1b[0m", output);
            0
        }
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m bundle failed: {}", e);
            1
        }
    }
}

/// `sofuu install` — install deps from package.json in cwd.
fn cmd_install(rt: &SofuuRuntime) -> i32 {
    let cwd = std::env::current_dir().unwrap_or_default();
    rt.npm_install_local(&cwd.to_string_lossy())
}

/// `sofuu add <pkg>...` — install npm packages.
fn cmd_add(rt: &SofuuRuntime, args: &[String]) -> i32 {
    if args.len() < 3 {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu add' requires a package name");
        eprintln!("Usage: sofuu add <package>[@version]");
        return 1;
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let cwd = cwd.to_string_lossy().to_string();
    let mut failed = false;
    for pkg in &args[2..] {
        // Specs validated by the Rust safety core before any C network work.
        if !sofuu_core::npm::spec_is_safe(pkg) {
            eprintln!("\x1b[31mError:\x1b[0m invalid package spec: '{}'", pkg);
            failed = true;
            continue;
        }
        println!("\n\x1b[1msofuu add {}\x1b[0m", pkg);
        if rt.npm_install(pkg, &cwd) != 0 {
            failed = true;
        }
    }
    if failed {
        1
    } else {
        0
    }
}

fn print_licenses() {
    println!("\n\x1b[1m⚡ Sofuu (素風) Open Source Notices\x1b[0m\n");
    println!("Sofuu is MIT Licensed.");
    println!("Copyright (c) 2024 Priyanshu Boruah\n");
    println!("\x1b[1mThird-party components bundled in this binary:\x1b[0m\n");
    println!("  QuickJS (JS Engine)");
    println!("    Copyright (c) 2017-2021 Fabrice Bellard, Charlie Gordon");
    println!("    License: MIT  |  https://bellard.org/quickjs/\n");
    println!("  libuv (Async I/O)");
    println!("    Copyright (c) 2015-present libuv project contributors");
    println!("    Copyright (c) Joyent, Inc. and other Node contributors");
    println!("    License: MIT  |  https://github.com/libuv/libuv\n");
    println!("  http-parser (HTTP/1.1 Parser)");
    println!("    Copyright (c) Joyent, Inc. and other Node contributors");
    println!("    License: MIT  |  https://github.com/nodejs/http-parser\n");
    println!("  libcurl (HTTP Client)");
    println!("    Copyright (c) 1996-2024 Daniel Stenberg and curl contributors");
    println!("    License: curl (MIT-style)  |  https://curl.se/docs/copyright.html\n");
    println!("\x1b[90mFull notices: see NOTICE file or https://sofuu.dev/licenses\x1b[0m\n");
}

fn main() -> ExitCode {
    // conhost (cmd.exe) prints ANSI escapes as literal text until the
    // process opts in — enable VT processing before anything prints
    // (no-op off-Windows).
    sofuu_core::rt::tui::windows_enable_vt();
    let args: Vec<String> = env::args().collect();

    // M3: give the JS side process.argv (the ported mod_process_set_args;
    // the retired C main.c did the same).
    // SAFETY: argv is built here and lives for the call; the callee copies
    // the strings into CString storage.
    unsafe {
        let argv: Vec<CString> = args
            .iter()
            .map(|a| CString::new(a.as_str()).unwrap_or_default())
            .collect();
        let mut ptrs: Vec<*mut c_char> = argv.iter().map(|c| c.as_ptr() as *mut c_char).collect();
        sofuu_core::modules::process::mod_process_set_args(args.len() as c_int, ptrs.as_mut_ptr());
    }

    // Bare `sofuu` → chat.
    if args.len() < 2 {
        let rt = match SofuuRuntime::init() {
            Some(rt) => rt,
            None => {
                eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                return ExitCode::FAILURE;
            }
        };
        let rc = cmd_chat(&rt, &args);
        return if rc == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }

    let cmd = args[1].as_str();

    match cmd {
        "version" | "--version" | "-v" => {
            print_version();
            ExitCode::SUCCESS
        }
        "help" | "--help" | "-h" => {
            print_help();
            ExitCode::SUCCESS
        }
        "licenses" | "--licenses" => {
            print_licenses();
            ExitCode::SUCCESS
        }
        "repl" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_repl(&rt);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "eval" => {
            if args.len() < 3 {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu eval' requires a code string argument");
                return ExitCode::FAILURE;
            }
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_eval(&rt, &args[2]);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "doctor" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            if doctor::run(&rt, &args) == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "run" => {
            if args.len() < 3 {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu run' requires a file argument");
                return ExitCode::FAILURE;
            }
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_run(&rt, &args[2]);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "bundle" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_bundle(&rt, &args);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "install" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_install(&rt);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "add" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_add(&rt, &args);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "chat" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_chat(&rt, &args);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "session" | "sessions" => {
            let rc = cmd_session(&args);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "outputs" | "output" => {
            let rc = cmd_outputs(&args);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "agent" | "agents" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_agent(&rt, &args);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "serve" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_serve(&rt, &args);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        _ => {
            eprintln!("\x1b[31mError:\x1b[0m Unknown command '{}'\nRun 'sofuu help' for usage.\x1b[0m", cmd);
            ExitCode::FAILURE
        }
    }
}
