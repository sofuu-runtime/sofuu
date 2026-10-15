#![recursion_limit = "256"]

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
    println!("  sofuu infer <model.gguf> \"<prompt>\" Fast on-device CPU inference on GGUF models");
    println!("  sofuu swarm worker|leader   Distributed edge swarm cluster over LAN (layer pipelining)");
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
                    let cleaned = sofuu_core::infer::clean_model_path(&v);
                    let is_gguf = v.ends_with(".gguf")
                        || cleaned.extension().map_or(false, |e| e == "gguf");
                    if is_gguf {
                        cfg.provider = "local".into();
                        cfg.base_url.clear();
                        cfg.api_key.clear();
                        let caps = sofuu_core::rt::model_caps::lookup(Some(&v));
                        if caps.ctx_window > 0 {
                            cfg.ctx_window = caps.ctx_window as i64;
                            cfg.ctx_window_explicit = false;
                        }
                        if caps.max_output > 0 {
                            cfg.max_output = caps.max_output as i64;
                        }
                        if caps.thinking != sofuu_core::rt::model_caps::Thinking::None {
                            cfg.effort = "high".into();
                        } else {
                            cfg.effort = "off".into();
                        }
                    }
                    cfg.model = v;
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {a} needs a value");
                    return 1;
                }
            },
            "-t" | "--threads" => match value_arg(args, i) {
                Some(v) => {
                    if let Ok(n) = v.parse::<usize>() {
                        cfg.threads = n;
                        sofuu_core::infer::quant::set_thread_count(n);
                    }
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
    // Local-inference prefill on large models can legitimately produce no
    // response bytes for many minutes (e.g. a 27B ternary prompt decode) —
    // SOFUU_HTTP_IDLE_MS stays 0 (no deadline at all once a request is in).
    // P2 (audit 2026-10-14): the pre-AUTH window is what must stay bounded
    // — an accepted connection that never sends its headers ("silent" or
    // dribbling) used to be able to hold an fd forever before any auth
    // check ran. Default HEADERS deadline: 30s (a legit client sends in
    // milliseconds; the operator can still pin either value explicitly).
    if std::env::var_os("SOFUU_HTTP_HEADERS_MS").is_none() {
        std::env::set_var("SOFUU_HTTP_HEADERS_MS", "30000");
    }
    if std::env::var_os("SOFUU_HTTP_IDLE_MS").is_none() {
        std::env::set_var("SOFUU_HTTP_IDLE_MS", "0");
    }
    let mut brain_path = String::new();
    let mut model_path = String::new();
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
            "--model" | "-m" => match value_arg(args, i) {
                Some(v) => {
                    model_path = v;
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m {} needs a value", args[i]);
                    return 1;
                }
            },
            "--infer" => {
                i += 1;
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
    eprintln!("  │   ⚡ Sofuu Edge Server          │");
    eprintln!("  │   v{:<31}│", VERSION);
    eprintln!("  └─────────────────────────────────┘\x1b[0m");
    if !model_path.is_empty() {
        eprintln!("\x1b[2m  model: {}\x1b[0m", model_path);
    }
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
    eprintln!("\x1b[2m    GET  /v1/models\x1b[0m");
    eprintln!("\x1b[2m    POST /v1/chat/completions  (OpenAI SSE stream & json)\x1b[0m");
    eprintln!("\x1b[2m    POST /v1/completions       (text completions)\x1b[0m");
    eprintln!("\x1b[2m    POST /infer                (direct local infer)\x1b[0m");
    eprintln!("\x1b[2m    POST /remember             {{text, role?, source?}}\x1b[0m");
    eprintln!("\x1b[2m    GET  /recall?q=…&k=…\x1b[0m");
    eprintln!("\x1b[2m    GET  /memories?tier=…&role=…&text=…&offset=…&limit=…\x1b[0m");
    eprintln!("\x1b[2m    POST /forget               {{id}}\x1b[0m");
    eprintln!("\x1b[2m    POST /share                (export card: records + vectors + manifest)\x1b[0m");
    eprintln!();

    let brain_path_js = serde_json::to_string(&brain_path).unwrap_or_else(|_| "\"\"".into());
    let model_path_js = serde_json::to_string(&model_path).unwrap_or_else(|_| "\"\"".into());
    let token_js = serde_json::to_string(&token).unwrap_or_else(|_| "\"\"".into());
    let host_js = serde_json::to_string(&host).unwrap_or_else(|_| "\"\"".into());

    let serve_js = format!(r#"
(async function () {{
  var g = globalThis;
  if (!g.sofuu) g.sofuu = {{}};
  var brainPath = {brain_path_js};
  var modelPath = {model_path_js};
  var token = {token_js};
  var host = {host_js};
  var port = {port};
  var brain = null;
  /* B1.5: open the file by ITS OWN manifest, not a hardcoded 768 — this
   * server used to be able to serve hash brains only. `inspect` reads the
   * manifest without opening; `embed` then follows the space so /remember
   * and /recall produce vectors the store actually holds. */
  var embed = function (t) {{ return sofuu.ai.embedLocal(String(t)); }};
  if (sofuu.memory && typeof sofuu.memory.open === 'function') {{
    var dim = 768;
    var space = undefined;
    try {{
      if (typeof sofuu.memory.inspect === 'function') {{
        var info = JSON.parse(sofuu.memory.inspect(brainPath));
        if (info && info.ok) {{
          if (info.dimension) dim = info.dimension;
          if (info.id) space = info.id;
        }}
      }}
    }} catch (e0) {{}}
    try {{
      brain = space === undefined
        ? sofuu.memory.open(brainPath, dim)
        : sofuu.memory.open(brainPath, dim, space);
    }} catch (e) {{ console.error('brain open: ' + e); }}
    if (space === 'semantic-projector-v1' && typeof sofuu.ai.embedLocalSemantic === 'function') {{
      embed = function (t) {{ return sofuu.ai.embedLocalSemantic(String(t)); }};
    }} else if (space === 'semantic-table-v2' && typeof sofuu.ai.embedLocalSemanticV2 === 'function') {{
      embed = function (t) {{ return sofuu.ai.embedLocalSemanticV2(String(t)); }};
    }}
    /* P2 (audit 2026-10-14): a manifest space this server cannot embed
     * LOCALLY (a remote provider space, a custom agent space) used to fall
     * through to the default hash embedder — hash-768 vectors stored under
     * a foreign-space manifest, mislabeled in /share cards and useless at
     * recall. Refuse instead: the store simply stays unavailable (503s
     * below) with the reason on stderr. hash-v1 needs no switch. */
    if (brain && space && space !== 'hash-v1' &&
        space !== 'semantic-projector-v1' && space !== 'semantic-table-v2') {{
      console.error("brain space '" + space + "' has no local embedder in this server — refusing to serve (hash vectors would be mislabeled in a '" + space + "' store)");
      brain = null;
    }}
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
      res.end(JSON.stringify({{ ok: true, memories: count, model: modelPath || null }}));
      return;
    }}
    if (req.method === 'GET' && path === '/v1/models') {{
      var mId = modelPath ? modelPath.split('/').pop() : 'sofuu-default';
      res.writeHead(200, {{ 'Content-Type': 'application/json' }});
      res.end(JSON.stringify({{
        object: 'list',
        data: [{{
          id: mId,
          object: 'model',
          created: Math.floor(Date.now() / 1000),
          owned_by: 'sofuu'
        }}]
      }}));
      return;
    }}
    if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/chat/completions')) {{
      var body = typeof req.body === 'string' ? req.body : '';
      try {{
        var data = JSON.parse(body || '{{}}');
        var targetModel = data.model || modelPath;
        if (!targetModel) {{
          res.writeHead(400, {{ 'Content-Type': 'application/json' }});
          res.end(JSON.stringify({{ ok: false, error: 'no model loaded; specify --model <path> or pass model in body' }}));
          return;
        }}
        var prompt = '';
        if (Array.isArray(data.messages)) {{
          for (var mi = 0; mi < data.messages.length; mi++) {{
            var m = data.messages[mi];
            prompt += (m.role || 'user') + ': ' + (m.content || '') + '\n';
          }}
          prompt += 'assistant:';
        }} else {{
          prompt = String(data.prompt || '');
        }}
        var maxTokens = data.max_tokens || 512;
        var temp = typeof data.temperature === 'number' ? data.temperature : 0.7;
        var reqId = 'chatcmpl-' + Math.random().toString(36).substring(2, 12);
        var created = Math.floor(Date.now() / 1000);

        if (data.stream) {{
          res.writeHead(200, {{
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
          }});
          var it = sofuu.ai.streamLocal(targetModel, prompt, {{ maxTokens: maxTokens, temperature: temp }});
          while (true) {{
            var n = await it.next();
            if (n.done) break;
            var chunk = n.value;
            if (chunk && chunk.text) {{
              var payload = {{
                id: reqId,
                object: 'chat.completion.chunk',
                created: created,
                model: targetModel,
                choices: [{{ index: 0, delta: {{ content: chunk.text }}, finish_reason: null }}]
              }};
              res.write('data: ' + JSON.stringify(payload) + '\n\n');
            }}
          }}
          var stopPayload = {{
            id: reqId,
            object: 'chat.completion.chunk',
            created: created,
            model: targetModel,
            choices: [{{ index: 0, delta: {{}}, finish_reason: 'stop' }}]
          }};
          res.write('data: ' + JSON.stringify(stopPayload) + '\n\n');
          res.write('data: [DONE]\n\n');
          res.end();
        }} else {{
          var it = sofuu.ai.streamLocal(targetModel, prompt, {{ maxTokens: maxTokens, temperature: temp }});
          var fullText = '';
          while (true) {{
            var n = await it.next();
            if (n.done) break;
            if (n.value && n.value.text) fullText += n.value.text;
          }}
          res.writeHead(200, {{ 'Content-Type': 'application/json' }});
          res.end(JSON.stringify({{
            id: reqId,
            object: 'chat.completion',
            created: created,
            model: targetModel,
            choices: [{{
              index: 0,
              message: {{ role: 'assistant', content: fullText }},
              finish_reason: 'stop'
            }}]
          }}));
        }}
      }} catch (e) {{
        res.writeHead(500, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: String(e.message || e) }}));
      }}
      return;
    }}
    if (req.method === 'POST' && (path === '/v1/completions' || path === '/infer')) {{
      var body = typeof req.body === 'string' ? req.body : '';
      try {{
        var data = JSON.parse(body || '{{}}');
        var targetModel = data.model || modelPath;
        if (!targetModel) {{
          res.writeHead(400, {{ 'Content-Type': 'application/json' }});
          res.end(JSON.stringify({{ ok: false, error: 'no model loaded; specify --model <path> or pass model in body' }}));
          return;
        }}
        var prompt = String(data.prompt || '');
        var maxTokens = data.max_tokens || 512;
        var temp = typeof data.temperature === 'number' ? data.temperature : 0.7;
        var reqId = 'cmpl-' + Math.random().toString(36).substring(2, 12);
        var created = Math.floor(Date.now() / 1000);

        if (data.stream) {{
          res.writeHead(200, {{
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
          }});
          var it = sofuu.ai.streamLocal(targetModel, prompt, {{ maxTokens: maxTokens, temperature: temp }});
          while (true) {{
            var n = await it.next();
            if (n.done) break;
            if (n.value && n.value.text) {{
              res.write('data: ' + JSON.stringify({{
                id: reqId,
                object: 'text_completion',
                created: created,
                choices: [{{ text: n.value.text, index: 0, finish_reason: null }}]
              }}) + '\n\n');
            }}
          }}
          res.write('data: ' + JSON.stringify({{
            id: reqId,
            object: 'text_completion',
            created: created,
            choices: [{{ text: '', index: 0, finish_reason: 'stop' }}]
          }}) + '\n\n');
          res.write('data: [DONE]\n\n');
          res.end();
        }} else {{
          var it = sofuu.ai.streamLocal(targetModel, prompt, {{ maxTokens: maxTokens, temperature: temp }});
          var fullText = '';
          while (true) {{
            var n = await it.next();
            if (n.done) break;
            if (n.value && n.value.text) fullText += n.value.text;
          }}
          res.writeHead(200, {{ 'Content-Type': 'application/json' }});
          res.end(JSON.stringify({{
            id: reqId,
            object: 'text_completion',
            created: created,
            model: targetModel,
            choices: [{{ text: fullText, index: 0, finish_reason: 'stop' }}]
          }}));
        }}
      }} catch (e) {{
        res.writeHead(500, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: String(e.message || e) }}));
      }}
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
        var vec = embed(data.text);
        var role = data.role || 'user';
        brain.remember(new Float32Array(vec), data.text, role, 0, data.source || 'server');
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
      try {{
        var q = (url.split('?')[1] || '').split('&');
        for (var i = 0; i < q.length; i++) {{
          if (!q[i]) continue;
          /* P2 (audit): split on the FIRST '=' only — a value containing
           * '=' (base64, predicates) used to be silently dropped — and
           * '+' means space before percent-decoding. */
          var eq = q[i].indexOf('=');
          if (eq < 0) continue;
          qs[decodeURIComponent(q[i].slice(0, eq))] =
            decodeURIComponent(q[i].slice(eq + 1).replace(/\+/g, ' '));
        }}
      }} catch (eQ) {{
        /* P1 (audit 2026-10-14): a malformed %sequence used to throw
         * outside any handler → the promise rejected → no bytes written
         * and serve runs without deadlines → the socket hung forever. */
        res.writeHead(400, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: 'bad query encoding' }}));
        return;
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
      try {{
        var vec = embed(qs.q);
        var k = parseInt(qs.k || '5', 10);
        var results = brain.recall(new Float32Array(vec), k);
        res.writeHead(200, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: true, results: results }}));
      }} catch (eR) {{
        res.writeHead(500, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: String(eR.message || eR) }}));
      }}
      return;
    }}
    if (req.method === 'GET' && path === '/memories') {{
      if (!brain) {{
        res.writeHead(503, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: 'brain not available' }}));
        return;
      }}
      var mqs = {{}};
      try {{
        var mq = (url.split('?')[1] || '').split('&');
        for (var mi = 0; mi < mq.length; mi++) {{
          if (!mq[mi]) continue;
          var meq = mq[mi].indexOf('=');
          if (meq < 0) continue;
          mqs[decodeURIComponent(mq[mi].slice(0, meq))] =
            decodeURIComponent(mq[mi].slice(meq + 1).replace(/\+/g, ' '));
        }}
      }} catch (eM) {{
        res.writeHead(400, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: 'bad query encoding' }}));
        return;
      }}
      var lopts = {{}};
      if (mqs.tier !== undefined) lopts.tier = parseInt(mqs.tier, 10) || 0;
      if (mqs.role) lopts.role = mqs.role;
      if (mqs.entity) lopts.entity = mqs.entity;
      if (mqs.text) lopts.text = mqs.text;
      if (mqs.offset) lopts.offset = parseInt(mqs.offset, 10) || 0;
      lopts.limit = mqs.limit ? (parseInt(mqs.limit, 10) || 20) : 20;
      try {{
        res.writeHead(200, {{ 'Content-Type': 'application/json' }});
        res.end(brain.list(lopts));
      }} catch (e) {{
        res.writeHead(500, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: String(e.message || e) }}));
      }}
      return;
    }}
    if (req.method === 'POST' && path === '/forget') {{
      var fbody = typeof req.body === 'string' ? req.body : '';
      try {{
        var fdata = JSON.parse(fbody || '{{}}');
        if (!brain) {{
          res.writeHead(503, {{ 'Content-Type': 'application/json' }});
          res.end(JSON.stringify({{ ok: false, error: 'brain not available' }}));
          return;
        }}
        var fid = parseInt(fdata.id, 10);
        if (!(fid >= 0)) {{
          res.writeHead(400, {{ 'Content-Type': 'application/json' }});
          res.end(JSON.stringify({{ ok: false, error: 'id required' }}));
          return;
        }}
        var forgot = brain.forget(fid);
        brain.flush();
        res.writeHead(200, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: true, forgot: forgot, id: fid }}));
      }} catch (e) {{
        res.writeHead(500, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: String(e.message || e) }}));
      }}
      return;
    }}
    if (req.method === 'POST' && path === '/share') {{
      /* B2.1: a real card — records + vectors + manifest. */
      if (!brain) {{
        res.writeHead(503, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: 'brain not available' }}));
        return;
      }}
      try {{
        res.writeHead(200, {{ 'Content-Type': 'application/json' }});
        res.end(brain.export());
      }} catch (e) {{
        res.writeHead(500, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ ok: false, error: String(e.message || e) }}));
      }}
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
    println!("Copyright (c) 2026 Haruhito\n");
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
        "infer" => {
            if let Ok(spec) = std::env::var("SOFUU_BENCH_GEMM") {
                let parts: Vec<usize> = spec
                    .split(',')
                    .map(|p| p.trim().parse::<usize>().unwrap_or(0))
                    .collect();
                if parts.len() == 4 && parts.iter().all(|&v| v > 0) {
                    eprintln!("[bench] SOFUU_BENCH_GEMM rows={} cols={} batch={} iters={}", parts[0], parts[1], parts[2], parts[3]);
                    sofuu_core::infer::quant::gemm_bench(12, parts[0], parts[1], parts[2], parts[3]);
                    sofuu_core::infer::quant::gemm_bench(8, parts[0], parts[1], parts[2], parts[3]);
                    return ExitCode::SUCCESS;
                }
                // 5th element = iters>0: additionally A/B the q4_K nibble-plane
                // kernel vs the packed one on the same shape.
                if parts.len() == 5 && parts.iter().all(|&v| v > 0) {
                    eprintln!("[bench] SOFUU_BENCH_GEMM rows={} cols={} batch={} iters={} (+planes)", parts[0], parts[1], parts[2], parts[3]);
                    sofuu_core::infer::quant::gemm_bench(12, parts[0], parts[1], parts[2], parts[3]);
                    sofuu_core::infer::quant::gemm_bench_plane(parts[0], parts[1], parts[2], parts[4]);
                    sofuu_core::infer::quant::gemm_bench(8, parts[0], parts[1], parts[2], parts[3]);
                    return ExitCode::SUCCESS;
                }
            }
            if args.len() < 4 {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu infer' requires a model file and a prompt");
                eprintln!("Usage: sofuu infer <model.gguf> \"<prompt>\" [--max-tokens <n>] [--temp <f>]");
                return ExitCode::FAILURE;
            }
            let model_path = &args[2];
            let prompt = &args[3];

            let mut max_tokens = 256;
            let mut temperature = 0.7f32;
            let mut threads = 0usize;
            let mut sliding_window = 0usize;
            let mut custom_ctx = 0usize;
            let mut gpu_layers = 999usize;
            let mut lookahead = false;
            let mut snapkv = false;
            let mut dynamic_ffn = false;
            let mut kivi = false;
            let mut kivi_bits = 4usize;
            let mut streaming_weights = false;

            let mut i = 4;
            while i < args.len() {
                if args[i] == "--max-tokens" && i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        max_tokens = v;
                    }
                    i += 2;
                } else if args[i] == "--temp" && i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f32>() {
                        temperature = v;
                    }
                    i += 2;
                } else if (args[i] == "--threads" || args[i] == "-t") && i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        threads = v;
                    }
                    i += 2;
                } else if (args[i] == "--sliding-window" || args[i] == "--swa") && i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        sliding_window = v;
                    }
                    i += 2;
                } else if (args[i] == "--ctx" || args[i] == "-c") && i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        custom_ctx = v;
                    }
                    i += 2;
                } else if (args[i] == "--gpu-layers" || args[i] == "--ngl") && i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        gpu_layers = v;
                    }
                    i += 2;
                } else if args[i] == "--gpu" {
                    gpu_layers = 999;
                    i += 1;
                } else if args[i] == "--cpu" {
                    gpu_layers = 0;
                    i += 1;
                } else if args[i] == "--lookahead" {
                    lookahead = true;
                    i += 1;
                } else if args[i] == "--snapkv" {
                    snapkv = true;
                    i += 1;
                } else if args[i] == "--dynamic-ffn" {
                    dynamic_ffn = true;
                    i += 1;
                } else if args[i] == "--kivi" {
                    kivi = true;
                    i += 1;
                } else if args[i] == "--kivi-bits" && i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        kivi_bits = v;
                    }
                    kivi = true;
                    i += 2;
                } else if args[i] == "--streaming-weights" || args[i] == "--cache-bypass" {
                    streaming_weights = true;
                    i += 1;
                } else {
                    i += 1;
                }
            }

            let engine_type = if gpu_layers > 0 { "⚡ Sofuu Neural Engine (Heterogeneous CPU+GPU)" } else { "⚡ Sofuu CPU Inference Engine" };
            eprintln!("\x1b[36m{}\x1b[0m — loading {}...", engine_type, model_path);
            let t0 = std::time::Instant::now();
            let load_result = if gpu_layers == 0 {
                sofuu_core::infer::InferEngine::load_with_seq_len_cpu(model_path, custom_ctx)
            } else {
                sofuu_core::infer::InferEngine::load_with_seq_len(model_path, custom_ctx)
            };
            let mut engine = match load_result {
                Ok(eng) => eng,
                Err(e) => {
                    eprintln!("\x1b[31mError loading model:\x1b[0m {e}");
                    return ExitCode::FAILURE;
                }
            };
            #[cfg(target_os = "macos")]
            engine.model.set_gpu_layers(gpu_layers);

            let load_dur = t0.elapsed();
            let active_threads = if threads > 0 {
                threads
            } else {
                sofuu_core::infer::quant::get_thread_count()
            };
            eprintln!(
                "\x1b[32m✓ Model loaded in {:.2?}\x1b[0m (dim={}, layers={}, heads={}, vocab={}, threads={})\n",
                load_dur,
                engine.model.config.dim,
                engine.model.config.n_layers,
                engine.model.config.n_heads,
                engine.model.config.vocab_size,
                active_threads
            );

            let t_gen = std::time::Instant::now();
            let mut token_count = 0usize;
            use std::io::Write;
            let res = engine.stream(
                prompt,
                sofuu_core::infer::GenerateOptions {
                    max_tokens,
                    temperature,
                    threads,
                    sliding_window,
                    lookahead,
                    snapkv,
                    dynamic_ffn,
                    kivi,
                    kivi_bits,
                    streaming_weights,
                    ..Default::default()
                },
                |tok| {
                    token_count += 1;
                    print!("{tok}");
                    let _ = std::io::stdout().flush();
                    true
                },
            );

            println!();
            let gen_dur = t_gen.elapsed();
            if let Err(e) = res {
                eprintln!("\n\x1b[31mInference error:\x1b[0m {e}");
                return ExitCode::FAILURE;
            }
            let prefill_toks = engine.last_prefill_tokens;
            let prefill_dur = engine.last_prefill_duration;
            let prefill_rate = if prefill_dur.as_secs_f32() > 0.0 {
                prefill_toks as f32 / prefill_dur.as_secs_f32()
            } else {
                0.0
            };
            let decode_dur = gen_dur.saturating_sub(prefill_dur);
            let decode_rate = if decode_dur.as_secs_f32() > 0.0 {
                token_count as f32 / decode_dur.as_secs_f32()
            } else {
                0.0
            };
            eprintln!(
                "\n\x1b[90m── prefill: {} tk in {:.2?} ({:.1} tk/s) · decode: {} tk in {:.2?} ({:.1} tk/s) · total: {:.2?} ──\x1b[0m",
                prefill_toks, prefill_dur, prefill_rate, token_count, decode_dur, decode_rate, gen_dur
            );
            ExitCode::SUCCESS
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
        "swarm" => cmd_swarm(&args),
        _ => {
            eprintln!("\x1b[31mError:\x1b[0m Unknown command '{}'\nRun 'sofuu help' for usage.\x1b[0m", cmd);
            ExitCode::FAILURE
        }
    }
}

fn print_swarm_help() {
    println!("\x1b[1;36m  ┌────────────────────────────────────────────────────────┐");
    println!("  │   ⚡ Sofuu Swarm — Distributed Edge AI Swarm Cluster    │");
    println!("  │   Sequential Layer Pipelining across LAN Edge Nodes    │");
    println!("  └────────────────────────────────────────────────────────┘\x1b[0m\n");
    println!("\x1b[1mUsage:\x1b[0m");
    println!("  sofuu swarm worker --model <path> [--listen <ip:port>] [--layers <range>] [--node <idx/total>]");
    println!("  sofuu swarm leader --model <path> --workers <ip:port,...> \"<prompt>\" [--layers <range>]");
    println!("  sofuu swarm run    --model <path> --workers <ip:port,...> \"<prompt>\" [--max-tokens <n>]");
    println!("  sofuu swarm discover [--timeout <secs>]");
    println!("\n\x1b[1mExamples:\x1b[0m");
    println!("  # Start worker on second node (e.g. Raspberry Pi 5):");
    println!("  sofuu swarm worker --model ./model.gguf --listen 0.0.0.0:9001 --layers 16..32\n");
    println!("  # Run distributed inference from leader node (Mac / PC):");
    println!("  sofuu swarm leader --model ./model.gguf --workers 192.168.1.50:9001 --layers 0..16 \"Explain gravity.\"\n");
    println!("  # Auto-partitioning across 2 nodes:");
    println!("  sofuu swarm worker --model ./model.gguf --listen 0.0.0.0:9001 --node 2/2");
    println!("  sofuu swarm leader --model ./model.gguf --workers 192.168.1.50:9001 --node 1/2 \"Hello world\"");
}

fn cmd_swarm(args: &[String]) -> ExitCode {
    if args.len() < 3 {
        print_swarm_help();
        return ExitCode::SUCCESS;
    }

    let subcmd = args[2].as_str();
    if subcmd == "--help" || subcmd == "-h" || subcmd == "help" {
        print_swarm_help();
        return ExitCode::SUCCESS;
    }

    match subcmd {
        "discover" => {
            println!("\x1b[36m⚡ Searching for Sofuu Swarm nodes on local network (UDP 9770)...\x1b[0m");
            let peers = sofuu_core::infer::swarm::SwarmDiscovery::listen(std::time::Duration::from_secs(3));
            if peers.is_empty() {
                println!("\x1b[33mNo swarm nodes discovered on local subnet.\x1b[0m");
            } else {
                println!("\x1b[32m✓ Found {} swarm peer(s):\x1b[0m", peers.len());
                for p in &peers {
                    println!("  • \x1b[1m{}\x1b[0m ({}) — layers: {} | model: '{}'", p.addr, p.role, p.layers, p.model);
                }
            }
            ExitCode::SUCCESS
        }
        "worker" | "join" => {
            let mut model_path = String::new();
            let mut listen_addr = format!("0.0.0.0:{}", sofuu_core::infer::swarm::DEFAULT_SWARM_PORT);
            let mut layers_str = String::new();
            let mut node_str = String::new();
            let mut threads = 0usize;
            let mut gpu_layers = 999usize;
            let mut custom_ctx = 0usize;
            let mut broadcast = true;

            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--model" | "-m" => {
                        if i + 1 < args.len() { model_path = args[i + 1].clone(); i += 2; } else { i += 1; }
                    }
                    "--listen" | "-l" | "--port" | "-p" => {
                        if i + 1 < args.len() {
                            let val = &args[i + 1];
                            if val.contains(':') {
                                listen_addr = val.clone();
                            } else {
                                listen_addr = format!("0.0.0.0:{}", val);
                            }
                            i += 2;
                        } else { i += 1; }
                    }
                    "--layers" => {
                        if i + 1 < args.len() { layers_str = args[i + 1].clone(); i += 2; } else { i += 1; }
                    }
                    "--node" => {
                        if i + 1 < args.len() { node_str = args[i + 1].clone(); i += 2; } else { i += 1; }
                    }
                    "--threads" | "-t" => {
                        if i + 1 < args.len() { threads = args[i + 1].parse().unwrap_or(0); i += 2; } else { i += 1; }
                    }
                    "--gpu-layers" | "--ngl" => {
                        if i + 1 < args.len() { gpu_layers = args[i + 1].parse().unwrap_or(999); i += 2; } else { i += 1; }
                    }
                    "--gpu" => { gpu_layers = 999; i += 1; }
                    "--cpu" => { gpu_layers = 0; i += 1; }
                    "--ctx" | "-c" => {
                        if i + 1 < args.len() { custom_ctx = args[i + 1].parse().unwrap_or(0); i += 2; } else { i += 1; }
                    }
                    "--no-broadcast" => { broadcast = false; i += 1; }
                    _ => { i += 1; }
                }
            }

            if model_path.is_empty() {
                eprintln!("\x1b[31mError:\x1b[0m --model <path> is required for swarm worker");
                return ExitCode::FAILURE;
            }

            if threads > 0 {
                sofuu_core::infer::quant::set_thread_count(threads);
            }

            eprintln!("\x1b[36mLoading model for swarm worker:\x1b[0m {}", model_path);
            let load_res = if gpu_layers == 0 {
                sofuu_core::infer::InferEngine::load_with_seq_len_cpu(&model_path, custom_ctx)
            } else {
                sofuu_core::infer::InferEngine::load_with_seq_len(&model_path, custom_ctx)
            };
            let mut engine = match load_res {
                Ok(eng) => eng,
                Err(e) => {
                    eprintln!("\x1b[31mFailed to load model:\x1b[0m {e}");
                    return ExitCode::FAILURE;
                }
            };
            #[cfg(target_os = "macos")]
            engine.model.set_gpu_layers(gpu_layers);

            let total_layers = engine.model.config.n_layers;

            let (start_layer, end_layer, node_idx, total_nodes) = if !layers_str.is_empty() {
                match sofuu_core::infer::swarm::LayerPartition::parse_range(&layers_str, total_layers) {
                    Ok((s, e)) => (s, e, 1, 2),
                    Err(e) => {
                        eprintln!("\x1b[31mInvalid layer range:\x1b[0m {e}");
                        return ExitCode::FAILURE;
                    }
                }
            } else if !node_str.is_empty() {
                if let Some((idx_str, tot_str)) = node_str.split_once('/') {
                    let idx: usize = idx_str.parse().unwrap_or(2);
                    let tot: usize = tot_str.parse().unwrap_or(2);
                    let partitions = sofuu_core::infer::swarm::LayerPartition::auto_partition(total_layers, tot);
                    if idx > 0 && idx <= partitions.len() {
                        let (s, e) = partitions[idx - 1];
                        (s, e, idx - 1, tot)
                    } else {
                        eprintln!("\x1b[31mInvalid node index {}/{} (valid: 1..={})\x1b[0m", idx, tot, tot);
                        return ExitCode::FAILURE;
                    }
                } else {
                    eprintln!("\x1b[31mInvalid --node format (expected '2/2')\x1b[0m");
                    return ExitCode::FAILURE;
                }
            } else {
                // Default: second half of model
                let mid = total_layers / 2;
                (mid, total_layers, 1, 2)
            };

            let partition = sofuu_core::infer::swarm::LayerPartition {
                node_id: node_idx,
                total_nodes,
                start_layer,
                end_layer,
                total_layers,
            };

            if broadcast {
                let service_port = listen_addr.split(':').last().and_then(|p| p.parse().ok()).unwrap_or(sofuu_core::infer::swarm::DEFAULT_SWARM_PORT);
                sofuu_core::infer::swarm::SwarmDiscovery::broadcast_beacon(
                    "sofuu-default",
                    "worker",
                    service_port,
                    &format!("{}..{}", start_layer, end_layer),
                    &model_path,
                    std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
                );
            }

            let mut worker = sofuu_core::infer::swarm::SwarmWorker::new(engine, partition);
            if let Err(e) = worker.serve(&listen_addr, None) {
                eprintln!("\x1b[31mWorker error:\x1b[0m {e}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        "leader" | "run" | "host" => {
            let mut model_path = String::new();
            let mut workers_raw = String::new();
            let mut prompt = String::new();
            let mut layers_str = String::new();
            let mut node_str = String::new();
            let mut max_tokens = 256;
            let mut temperature = 0.7f32;
            let mut threads = 0usize;
            let mut gpu_layers = 999usize;
            let mut custom_ctx = 0usize;

            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--model" | "-m" => {
                        if i + 1 < args.len() { model_path = args[i + 1].clone(); i += 2; } else { i += 1; }
                    }
                    "--workers" | "--connect" | "-w" => {
                        if i + 1 < args.len() { workers_raw = args[i + 1].clone(); i += 2; } else { i += 1; }
                    }
                    "--layers" => {
                        if i + 1 < args.len() { layers_str = args[i + 1].clone(); i += 2; } else { i += 1; }
                    }
                    "--node" => {
                        if i + 1 < args.len() { node_str = args[i + 1].clone(); i += 2; } else { i += 1; }
                    }
                    "--max-tokens" => {
                        if i + 1 < args.len() { max_tokens = args[i + 1].parse().unwrap_or(256); i += 2; } else { i += 1; }
                    }
                    "--temp" => {
                        if i + 1 < args.len() { temperature = args[i + 1].parse().unwrap_or(0.7); i += 2; } else { i += 1; }
                    }
                    "--threads" | "-t" => {
                        if i + 1 < args.len() { threads = args[i + 1].parse().unwrap_or(0); i += 2; } else { i += 1; }
                    }
                    "--gpu-layers" | "--ngl" => {
                        if i + 1 < args.len() { gpu_layers = args[i + 1].parse().unwrap_or(999); i += 2; } else { i += 1; }
                    }
                    "--gpu" => { gpu_layers = 999; i += 1; }
                    "--cpu" => { gpu_layers = 0; i += 1; }
                    "--ctx" | "-c" => {
                        if i + 1 < args.len() { custom_ctx = args[i + 1].parse().unwrap_or(0); i += 2; } else { i += 1; }
                    }
                    other => {
                        if !other.starts_with('-') && prompt.is_empty() {
                            prompt = other.to_string();
                        }
                        i += 1;
                    }
                }
            }

            if model_path.is_empty() {
                eprintln!("\x1b[31mError:\x1b[0m --model <path> is required for swarm leader");
                return ExitCode::FAILURE;
            }
            if workers_raw.is_empty() {
                // Try auto-discovery
                eprintln!("\x1b[33mNo --workers specified. Probing subnet for discovered peers...\x1b[0m");
                let discovered = sofuu_core::infer::swarm::SwarmDiscovery::listen(std::time::Duration::from_millis(1500));
                if !discovered.is_empty() {
                    workers_raw = discovered.into_iter().map(|p| p.addr).collect::<Vec<_>>().join(",");
                    eprintln!("\x1b[32m✓ Auto-discovered workers: {}\x1b[0m", workers_raw);
                } else {
                    eprintln!("\x1b[31mError:\x1b[0m --workers <ip:port> is required (or run 'sofuu swarm worker' on worker first)");
                    return ExitCode::FAILURE;
                }
            }

            if prompt.is_empty() {
                prompt = "Why is the sky blue? Answer in 1 concise sentence.".to_string();
            }

            let worker_addrs: Vec<String> = workers_raw
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(|s| if s.contains(':') { s } else { format!("{}:{}", s, sofuu_core::infer::swarm::DEFAULT_SWARM_PORT) })
                .collect();

            if threads > 0 {
                sofuu_core::infer::quant::set_thread_count(threads);
            }

            eprintln!("\x1b[36m⚡ Sofuu Swarm Leader\x1b[0m — loading {}...", model_path);
            let t0 = std::time::Instant::now();
            let load_res = if gpu_layers == 0 {
                sofuu_core::infer::InferEngine::load_with_seq_len_cpu(&model_path, custom_ctx)
            } else {
                sofuu_core::infer::InferEngine::load_with_seq_len(&model_path, custom_ctx)
            };
            let mut engine = match load_res {
                Ok(eng) => eng,
                Err(e) => {
                    eprintln!("\x1b[31mFailed to load model:\x1b[0m {e}");
                    return ExitCode::FAILURE;
                }
            };
            #[cfg(target_os = "macos")]
            engine.model.set_gpu_layers(gpu_layers);

            let total_layers = engine.model.config.n_layers;
            let (start_layer, end_layer, node_idx, total_nodes) = if !layers_str.is_empty() {
                match sofuu_core::infer::swarm::LayerPartition::parse_range(&layers_str, total_layers) {
                    Ok((s, e)) => (s, e, 0, worker_addrs.len() + 1),
                    Err(e) => {
                        eprintln!("\x1b[31mInvalid layer range:\x1b[0m {e}");
                        return ExitCode::FAILURE;
                    }
                }
            } else if !node_str.is_empty() {
                if let Some((idx_str, tot_str)) = node_str.split_once('/') {
                    let idx: usize = idx_str.parse().unwrap_or(1);
                    let tot: usize = tot_str.parse().unwrap_or(worker_addrs.len() + 1);
                    let partitions = sofuu_core::infer::swarm::LayerPartition::auto_partition(total_layers, tot);
                    if idx > 0 && idx <= partitions.len() {
                        let (s, e) = partitions[idx - 1];
                        (s, e, idx - 1, tot)
                    } else {
                        eprintln!("\x1b[31mInvalid node index {}/{} (valid: 1..={})\x1b[0m", idx, tot, tot);
                        return ExitCode::FAILURE;
                    }
                } else {
                    eprintln!("\x1b[31mInvalid --node format (expected '1/2')\x1b[0m");
                    return ExitCode::FAILURE;
                }
            } else {
                // Default: leader takes first half, workers take rest
                let partitions = sofuu_core::infer::swarm::LayerPartition::auto_partition(total_layers, worker_addrs.len() + 1);
                let (s, e) = partitions[0];
                (s, e, 0, worker_addrs.len() + 1)
            };

            let partition = sofuu_core::infer::swarm::LayerPartition {
                node_id: node_idx,
                total_nodes,
                start_layer,
                end_layer,
                total_layers,
            };

            eprintln!(
                "\x1b[32m✓ Model loaded in {:.2?}\x1b[0m (dim={}, layers={}, leader layers: {}..{}/{})",
                t0.elapsed(),
                engine.model.config.dim,
                total_layers,
                start_layer,
                end_layer,
                total_layers
            );

            let mut leader = sofuu_core::infer::swarm::SwarmLeader::new(engine, partition);
            if let Err(e) = leader.connect_workers(&worker_addrs) {
                eprintln!("\x1b[31mFailed to connect swarm cluster:\x1b[0m {e}");
                return ExitCode::FAILURE;
            }

            println!("\n\x1b[1;36mPrompt:\x1b[0m {}\n", prompt);
            use std::io::Write;
            let t_gen = std::time::Instant::now();
            let mut token_count = 0usize;
            let res = leader.stream(
                &prompt,
                sofuu_core::infer::GenerateOptions {
                    max_tokens,
                    temperature,
                    threads,
                    ..Default::default()
                },
                |tok| {
                    token_count += 1;
                    print!("{}", tok);
                    std::io::stdout().flush().ok();
                    true
                },
            );
            println!();
            let dur = t_gen.elapsed();
            if let Err(e) = res {
                eprintln!("\n\x1b[31mSwarm generation error:\x1b[0m {e}");
                return ExitCode::FAILURE;
            }
            let rate = if dur.as_secs_f32() > 0.0 { token_count as f32 / dur.as_secs_f32() } else { 0.0 };
            eprintln!(
                "\n\x1b[90m── swarm generated: {} tk in {:.2?} ({:.1} tk/s) across {} node(s) ──\x1b[0m",
                token_count, dur, rate, worker_addrs.len() + 1
            );
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("\x1b[31mUnknown swarm subcommand '{}'\x1b[0m. Run 'sofuu swarm' for usage.", subcmd);
            ExitCode::FAILURE
        }
    }
}
