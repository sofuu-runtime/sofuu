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
mod cli_protocols;
mod conformance;
mod doctor;
mod output_archive;
mod session;
mod session_store;
mod theme;

const VERSION: &str = env!("CARGO_PKG_VERSION");

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
    println!("  sofuu eval \"<code>\"      Evaluate JS inline (prints the value; -q to silence)");
    println!("  sofuu bundle <entry.js>  Bundle to single file");
    println!("  sofuu bundle <entry.js> -o <out.js>");
    println!("  sofuu compile <entry.ts> Compile to a standalone executable (-o, --embed-brain)");
    println!("  sofuu install            Install from package.json");
    println!("  sofuu add <pkg>          Install npm package");
    println!("  sofuu agent list         List agent definitions (~/.sofuu/agents/*.js)");
    println!("  sofuu agent run <name> \"task\"  Run one agent headless");
    println!("  sofuu agent run <name> \"task\" --json   …with a JSON result");
    /* OBSERVABILITY-SPEC §4: the observation surface sits next to the agent
     * commands because that is what it observes. Descriptions start at the
     * same column as every other line in this block. */
    println!("  sofuu logs               List recent runs: id, start, steps, status");
    println!("  sofuu logs show <id>     Render one run's event trace");
    println!("  sofuu logs --follow      Stream a run's log live (Ctrl-C exits)");
    println!("  sofuu logs --last-error  Post-mortem box for the newest crash");
    println!("  sofuu resume <id>        Resume an interrupted agent run from its checkpoint");
    println!("  sofuu plugin list        List plugins in ./plugins/ and ~/.sofuu/plugins/");
    println!("  sofuu plugin validate <f> Validate plugin syntax, export, and schema");
    println!("  sofuu plugin run <tool>  Execute a plugin tool with 5-shield safety");
    println!("  sofuu plugin test <tool> Alias of plugin run; executes one tool in isolation");
    println!("  sofuu plugin init <name> Scaffold a plugin (--category=tool|service|...)");
    println!("  sofuu dev [dir]          Start AI-native full-stack dev server (experimental) (--port, --host, --model)");
    println!("  sofuu serve --brain <path>   Serve the brain over HTTP (--port, --host, --token)");
    /* W7 (QTSQ_UPGRADE_PLAN): the storage engine shipped with no CLI. These
     * sit beside `serve` because they operate on the same file, and
     * `sofuu brain --help` documents every one of them. */
    println!("  sofuu brain verify <path>    Full scrub: manifest, segments, digests (--json)");
    println!("  sofuu brain repair <path>    Quarantine what is torn, keep what verifies");
    println!("  sofuu brain stats <path>     Records, segments, generations, recovery counters");
    println!("  sofuu brain backup <src> <dst>  Hardlink snapshot of the sealed segments");
    println!("  sofuu brain restore <b> <dst>   Rebuild a brain from a backup");
    println!("  sofuu brain migrate <path>   Convert a v1 flat brain to the segmented store");
    println!("  sofuu infer <model.gguf> \"<prompt>\" Fast on-device CPU inference on GGUF models");
    println!("  sofuu swarm worker|leader   Distributed edge swarm cluster over LAN (layer pipelining)");
    println!("  sofuu doctor             Health check: QTSQ, brain round-trip, ctx resolution");
    println!("  sofuu version            Print version and exit");
    println!("  sofuu licenses           Print open-source licenses");
    println!("  sofuu help               Print this help");
    println!("  sofuu <command> --help   Show every flag a subcommand accepts");
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
    /* PC-08 (audit 2026-10-29): the capability/tooling commands are real
     * subcommands and were missing from this page. They get their own block
     * with the COMMAND NAME first, because these are the commands a script
     * probes (`sofuu --help | grep -E '^\s+capabilities\b'`) and because a
     * name column reads like the `Chat commands` list below; descriptions
     * still open at the same column as every `sofuu …` line above. */
    println!("\n\x1b[1mProtocol commands (run as `sofuu <command>`):\x1b[0m");
    println!("  capabilities             Print the runtime capability manifest as JSON");
    println!("  tools list               List the tools the active policy allows (--json)");
    println!("  policy init|check|dry-run|why  Write a profile, audit a script, explain a rule");
    println!("  mcp list|test|add|remove Manage MCP servers and inspect their tools");
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

// ── R-29: per-subcommand help ─────────────────────────────────────
//
// `sofuu serve --help` used to START a real HTTP server on
// 127.0.0.1:7707 and create ~/.sofuu/serve_token, because no command
// parsed a help flag at all (only `swarm` did). A help request must be
// pure documentation: no runtime init, no listener, no token file, no state
// on disk. main() therefore checks wants_help() BEFORE any command arm
// runs — in particular before SofuuRuntime::init() and cmd_serve().
//
// Every flag below is one the parser actually accepts; the docs are derived
// from the parsers, not invented.

/// True when argv asks the SUBCOMMAND for help.
///
/// Only the first argument after the command (or a leading `-` flag token)
/// is considered, so `sofuu eval 'x --help'` still evaluates the string and
/// `sofuu run file.js --help` passes `--help` to the script — the code/args
/// of a command are never stolen as a help request.
fn wants_help(args: &[String]) -> bool {
    match args.get(2).map(String::as_str) {
        Some("-h") | Some("--help") | Some("help") => true,
        Some(first) if first.starts_with('-') => {
            args[2..].iter().any(|a| a == "-h" || a == "--help")
        }
        _ => false,
    }
}

/// Print help for `cmd`; returns false when the command has no help page
/// (the caller then falls through to the normal dispatch).
fn print_subcommand_help(cmd: &str) -> bool {
    match cmd {
        "serve" => print_serve_help(),
        "infer" => print_infer_help(),
        "dev" | "web" => print_dev_help(),
        "chat" => print_chat_help(),
        "run" | "eval" => print_run_help(cmd),
        "repl" => println!(
            "Usage: sofuu repl\n\n\
             JavaScript REPL. Dot-commands: .help .clear .exit .version\n"
        ),
        "bundle" => println!(
            "Usage: sofuu bundle <entry.js> [-o <out.js>]\n\n\
             Bundle an ES module entry (npm resolution included) into one file.\n\
               -o, --out <out.js>   output path (default: printed to stdout)\n"
        ),
        "install" => println!("Usage: sofuu install\n\nInstall the dependencies from ./package.json.\n"),
        "add" => println!(
            "Usage: sofuu add <package>[@version] [<package>…]\n\n\
             Install npm packages into ./node_modules (specs are validated before\n\
             any network or disk work).\n"
        ),
        "compile" => println!(
            "Usage: sofuu compile <entry.ts|js> [-o <out>] [--embed-brain <file.qtsq>] [--no-strip]\n\n\
             Compile an agent into a standalone executable: this `sofuu` binary plus\n\
             the entry's bundled bytecode and, optionally, a `.qtsq` brain appended\n\
             behind a 32-byte trailer. The result runs on a machine with no Sofuu.\n\
               -o, --output <file>      output path (default: the entry's stem)\n\
               --embed-brain <file>     carry a brain; `sofuu.memory.embedded` opens it\n\
               --no-strip               keep the base binary's symbols\n\n\
             The artifact is tied to the QuickJS build that produced it, so recompile\n\
             when upgrading `sofuu`. SOFUU_PLAIN=1 runs a compiled binary as a normal CLI.\n"
        ),
        "compile-bc" => println!(
            "Usage: sofuu compile-bc <input.js> <output.bc>\n\n\
             Compile a shipped JS driver to a QuickJS bytecode snapshot.\n"
        ),
        "agent" | "agents" => print_agent_help(),
        "logs" => print_logs_help(),
        "resume" => print_resume_help(),
        "plugin" | "plugins" => print_plugin_help(),
        "session" | "sessions" => print_session_help(),
        "outputs" | "output" => print_outputs_help(),
        "doctor" => println!(
            "Usage: sofuu doctor [-m <model.gguf>] [--json]\n\n\
             Health check: QTSQ codec, brain round-trip, context resolution.\n\
               -m, --model <path>   model to probe (default: configured)\n\
               --json               machine-readable report\n"
        ),
        "capabilities" => println!(
            "Usage: sofuu capabilities\n\n\
             Print the runtime capability manifest as JSON (features, GPU\n\
             availability, inference backends, limits).\n"
        ),
        "tools" => println!(
            "Usage: sofuu tools list [--json]\n\n\
             List the tools the active policy allows, with schemas.\n"
        ),
        "policy" => println!(
            "Usage: sofuu policy init [--profile default|strict|offline] [--force] [--global] [--print]\n\
                    sofuu policy check <policy.json | script.ts> [--policy <file>] [--json]\n\
                    sofuu policy dry-run <script.ts> [--policy <file>]\n\
                    sofuu policy why <op> [key=value …] [--policy <file>]\n\n\
             init    write .sofuu/policy.json (refuses to overwrite without --force)\n\
             check   validate a policy document, or statically audit a script's\n\
                     fetch/fs/shell/env/MCP requirements against the active policy\n\
             dry-run print each requirement the script implies and the would-be\n\
                     decision (static: the script is not executed)\n\
             why     explain one operation\n\n\
             Guard flags accepted by `sofuu run` and `sofuu chat`:\n\
               --read-only    fs.write off + shell disabled\n\
               --no-net       outbound network denied\n\
               --confirm-all  approval.mode always\n"
        ),
        "mcp" => println!(
            "Usage: sofuu mcp list\n\
                    sofuu mcp test <command line | http(s)://url>\n\
                    sofuu mcp add <name> <command> [args…]\n\
                    sofuu mcp remove <name>\n\n\
             Manage MCP servers and inspect the tools they expose.\n"
        ),
        "conformance" => println!(
            "Usage: sofuu conformance [--suite <name>]\n\n\
             Run the built-in conformance checks against this binary.\n"
        ),
        "swarm" => print_swarm_help(),
        /* W7: `sofuu brain --help` — the module owns the page so the
         * subcommand list can never drift from the parser. */
        "brain" => sofuu_core::brain_cmd::print_help(),
        "version" | "licenses" | "help" => print_help(),
        _ => return false,
    }
    true
}

fn print_serve_help() {
    println!(
        "\x1b[1mUsage:\x1b[0m sofuu serve [options]\n\n\
         Serve the brain over HTTP (GET /health, POST /remember, GET /recall,\n\
         POST /share) with Bearer-token auth. The server runs as a shipped JS\n\
         driver on the normal runtime.\n\n\
         \x1b[1mOptions:\x1b[0m\n\
           -b, --brain <path>     brain store path (default: ~/.sofuu/brain.qtsq)\n\
           -m, --model <path>     local GGUF model to serve\n\
               --model-id <id>    model id advertised to clients\n\
               --infer            enable on-server inference for /share\n\
           -p, --port <n>         listen port 0-65535 (default: 7707)\n\
               --host <addr>      listen address (default: 127.0.0.1)\n\
           -t, --token <secret>   Bearer token; SOFUU_SERVE_TOKEN (env) wins and is\n\
                                  preferred, because --token is visible in `ps`\n\
               --policy <file>    policy file applied before the server starts\n\n\
         \x1b[1mEnvironment:\x1b[0m\n\
           SOFUU_SERVE_TOKEN       auth token (takes precedence over --token)\n\
           SOFUU_HTTP_HEADERS_MS   pre-auth header deadline (default: 30000)\n\
           SOFUU_HTTP_IDLE_MS      post-request idle deadline (default: 0 = none)\n\n\
         Without a token, one is generated and written to ~/.sofuu/serve_token.\n\
         \x1b[90mThis page is documentation only: `sofuu serve --help` starts no\n\
         listener and creates no token.\x1b[0m\n"
    );
}

fn print_dev_help() {
    println!(
        "\x1b[1mUsage:\x1b[0m sofuu dev [dir] [options]\n\n\
         Start the AI-native full-stack dev server (frontend engine + SSR) (experimental).\n\n\
         \x1b[1mOptions:\x1b[0m\n\
           -p, --port <n>       listen port (default: 3000)\n\
           -h, --host <addr>    listen address (default: 127.0.0.1)\n\
           -m, --model <path>   local GGUF model for the dev assistant\n"
    );
}

fn print_chat_help() {
    println!(
        "\x1b[1mUsage:\x1b[0m sofuu chat [options]      (bare `sofuu` does the same)\n\n\
         \x1b[1mOptions:\x1b[0m\n\
           -m, --model <name>        model to chat with\n\
           -p, --provider <name>     openai | anthropic | local | ollama | llamacpp |
                                     lmstudio | vllm\n\
           -k, --apikey <key>        API key for this session\n\
               --base-url <url>      override the provider endpoint\n\
                                     (llamacpp defaults to http://127.0.0.1:8080/v1)\n\
           -t, --threads <n>         inference threads (local provider)\n\
               --effort <level>      reasoning effort / thinking budget\n\
               --think <level>       alias of --effort\n\
               --brain               use the on-device brain\n\
               --sync on|off         session mesh sync (--no-sync is --sync off)\n\
               --ctx-window, --ctx <n>     context window in tokens (0 = provider default)\n\
               --max-output, --maxout, --max-tokens <n>   max output tokens per response\n\
               --on-error=<mode>     auto-retry | fail-stop | interactive;\n\
                                     sets SOFUU_ON_ERROR for this session\n\n\
         \x1b[1mEnvironment:\x1b[0m\n\
           SOFUU_ON_ERROR         terminal-failure policy, same three values;\n\
                                  --on-error exports it for the session it starts\n\n\
         Chat commands are listed in `sofuu help`.\n"
    );
}

fn print_run_help(cmd: &str) {
    if cmd == "run" {
        println!(
            "\x1b[1mUsage:\x1b[0m sofuu run <file.js|file.ts>\n\n\
             Evaluate a file as an ES module on a fresh runtime, then drain the\n\
             event loop (timers, I/O, promise chains).\n\n\
             \x1b[1mExit status:\x1b[0m\n\
               0    the script completed\n\
               1    an uncaught exception, an unhandled promise rejection, or a\n\
                    microtask-queue overflow\n\
               N    whatever the script passed to process.exit(N), or set via\n\
                    process.exitCode = N\n"
        );
    } else {
        println!(
            "\x1b[1mUsage:\x1b[0m sofuu eval \"<code>\" [options]\n\n\
             Evaluate a JavaScript string on a fresh runtime, then print the value the\n\
             string completed with — formatted the way `sofuu repl` formats its results.\n\
             Nothing is printed when that value is `undefined` (a program of statements,\n\
             or `console.log(...)`, which returns undefined), when it is a Promise (log\n\
             from its `.then` instead), or when the code throws: a failed eval reports on\n\
             stderr and exits 1, exactly like `sofuu run`.\n\n\
             \x1b[1mOptions:\x1b[0m\n\
               -q, --no-print    print no completion value — the old silent form, for\n\
                                 scripts that only want the code's side effects\n\n\
             Same exit-status rules as `sofuu run`.\n"
        );
    }
}

fn print_agent_help() {
    println!(
        "\x1b[1mUsage:\x1b[0m sofuu agent <subcommand>\n\n\
           list                       list agent definitions (~/.sofuu/agents/*.js)\n\
           run <name> <task…> [--json] run one agent headless\n\
           init <name>                scaffold ~/.sofuu/agents/<name>.js\n\
           validate <file.js>         dry-run a definition with precise errors\n\n\
         \x1b[1mOptions:\x1b[0m\n\
               --json             emit the result as JSON (run)\n\
               --on-error=<mode>  auto-retry | fail-stop | interactive (run)\n\
               --policy <file>    policy file applied before the agent starts\n"
    );
}

/// OBSERVABILITY-SPEC §4. Every flag below is one cmd_logs accepts; a bare
/// run id is shorthand for show.
fn print_logs_help() {
    println!(
        "\x1b[1mUsage:\x1b[0m sofuu logs [subcommand]\n\n\
           list | ls                 recent runs (default): id, start, steps, status\n\
           show <run_id>             rendered event trace for one run\n\
           follow [run_id]           stream a run's log (Ctrl-C exits)\n\
           last-error [run_id]       post-mortem box for the newest crash\n\n\
         \x1b[1mOptions:\x1b[0m\n\
           -f, --follow              same as follow\n\
               --last-error          same as last-error\n\n\
         Agent runs log to .sofuu/logs/agent-<run_id>.jsonl, chat sessions to\n\
         .sofuu/sessions/<sid>/events.ndjson, crash cards to .sofuu/crashes/.\n\
         Rendering lives in the shipped driver src/js/observability.js.\n"
    );
}

/// OBSERVABILITY-SPEC §4. --force and --step=N are the two knobs a resume
/// needs; --agent exists because a log that lost its run_start event cannot
/// name the definition to re-enter.
fn print_resume_help() {
    println!(
        "\x1b[1mUsage:\x1b[0m sofuu resume <run_id> [options]\n\n\
         Continue an interrupted agent run from its last checkpoint: the loop\n\
         re-enters at the checkpoint's step with the same messages, task, model\n\
         and provider, so a crash costs one turn instead of the whole run.\n\n\
         \x1b[1mOptions:\x1b[0m\n\
               --step=<n>         resume from the checkpoint recorded at step n\n\
                                  (default: the newest checkpoint)\n\
               --force            resume a run whose log already ended in\n\
                                  run_complete; without it a finished run is\n\
                                  refused, because replaying it may spend tokens\n\
                                  on an answer that already exists\n\
               --agent=<name>     which definition to re-enter (default: the\n\
                                  run_start event's name)\n\
               --on-error=<mode>  auto-retry | fail-stop | interactive\n\
               --json             emit the result as JSON\n\
               --policy <file>    policy file applied before the run resumes\n"
    );
}

fn print_plugin_help() {
    println!(
        "\x1b[1mUsage:\x1b[0m sofuu plugin <subcommand>\n\n\
           list [--json]                        list installed plugins\n\
           validate <file.js|file.ts>           validate syntax, exports, schema\n\
           run <tool_name> [args_json]          execute one plugin tool\n\
           test <tool_name> [args_json]         alias of run, isolated execution\n\
           init <name> [--category=<kind>]      scaffold a plugin\n\n\
         \x1b[1mOptions:\x1b[0m\n\
               --policy <file>    policy file applied before the plugin runs\n"
    );
}

fn print_session_help() {
    println!(
        "\x1b[1mUsage:\x1b[0m sofuu session <subcommand>\n\n\
           list | ls                 list sessions on this project (default)\n\
           show | info <id>          full session context\n\
           inspect <id>              storage view without payload text\n\
           repair <id>               rebuild a damaged segmented manifest\n\
           migrate <id>              migrate one legacy flat session\n\
           storage-stats | stats <id>  segment/storage counters\n\
           prune | clean             remove ended sessions older than 7 days\n\
           export <id> [--out <f>]   portable session document (secrets stripped)\n\
           import <file|->           validate + persist a session document\n"
    );
}

fn print_outputs_help() {
    println!(
        "\x1b[1mUsage:\x1b[0m sofuu outputs <subcommand>\n\n\
           list | ls [--session <id>] [--kind <k>] [--status <s>] [--limit <n>]\n\
           show | get <output-id>\n\
           search <text>\n\
           context <task>\n\
           stats | info\n\
           rebuild-index | repair\n\
           prune [--dry-run|--apply] [--older-than <s>] [--max-bytes <n>] [--max-count <n>]\n\n\
         `prune` is a dry run unless --apply is given.\n"
    );
}

fn print_infer_help() {
    println!(
        "\x1b[1mUsage:\x1b[0m sofuu infer <model.gguf> \"<prompt>\" [options]\n\
                 sofuu infer <model.gguf> --score <file> [options]\n\n\
         On-device inference over a local GGUF file. All eleven-plus flags the\n\
         parser accepts are documented here; anything else is left to the prompt.\n\n\
         \x1b[1mGeneration:\x1b[0m\n\
           -n, --max-tokens <n>        tokens to generate (default: 256)\n\
               --temp, --temperature <f>  sampling temperature (default: 0.7)\n\
           -t, --threads <n>           worker threads (0 = auto)\n\
               --chat                  wrap the prompt in the model's own chat template\n\
               --trace <file>          write a per-token JSON trace\n\n\
         \x1b[1mContext / attention:\x1b[0m\n\
           -c, --ctx <n>               custom context length (0 = model default)\n\
               --swa, --sliding-window <n>  sliding-window attention size (0 = model default)\n\
               --snapkv                SnapKV attention eviction\n\
               --dynamic-ffn           dynamic FFN width\n\
               --lookahead             speculative lookahead decoding (default on)\n\
               --no-lookahead          serial decoding (same as SOFUU_LOOKAHEAD=0)\n\n\
         \x1b[1mKV cache / memory:\x1b[0m\n\
               --kv-dtype <d>          KV cache dtype: f32 | 4 | 2 | 8 | f16k (aliases:\n\
                                       q4/4bit, q2/2bit, q8/8bit, k16)\n\
               --kivi                  quantized KIVI KV stream on the CPU layer loop\n\
               --kivi-bits <n>         KIVI bit width (default: 4)\n\
               --streaming-weights, --cache-bypass  stream weights instead of caching them\n\n\
         \x1b[1mPlacement:\x1b[0m\n\
               --ngl, --gpu-layers <n>  layers to place on the GPU (default: 999 = all)\n\
               --gpu                   prefer the GPU (same as --gpu-layers 999)\n\
               --cpu                   CPU only (same as --gpu-layers 0)\n\n\
         \x1b[1mScoring:\x1b[0m\n\
               --score <file>          score a file instead of generating (prompt unused;\n\
                                       may directly follow the model path)\n\
               --score-window <n>      scoring window size (0 = whole file)\n\n\
         \x1b[1mEnvironment:\x1b[0m\n\
           SOFUU_LOOKAHEAD=0           same as --no-lookahead\n\
           SOFUU_KV_RING=<n>           KIVI ring-KV size (default: 4096, min 256)\n"
    );
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
///
/// PC-09 (audit 2026-10-29): this is the one-shot REPL, so it echoes the
/// value its program completed with — `sofuu eval "1+1"` says `2`. A silent
/// `eval` is indistinguishable from a broken one and useless for exactly the
/// quick check the command exists for. `-q` / `--no-print` restores the old
/// silent form for scripts that only want the code's side effects.
fn cmd_eval(rt: &SofuuRuntime, code: &str, print_value: bool) -> i32 {
    if !print_value {
        // Byte-identical to the pre-PC-09 command: evaluate the string as
        // given and add nothing of our own to stdout.
        return rt.eval_string(code, "<eval>");
    }
    // The value is rendered by the engine, through the very `repl_inspect`
    // `sofuu repl` prints — not by a JS wrapper here, which would add frames
    // to every throw's stack trace. `display` is None when there is nothing
    // to echo (undefined completion, a Promise, or a failed program).
    // SAFETY: engine_ctx() is this runtime's live QuickJS context.
    let (rc, display) = unsafe {
        sofuu_core::rt::engine::eval_completion_display(rt.engine_ctx(), code, "<eval>")
    };
    if let Some(display) = display {
        print_repl_value(&display);
    }
    rc
}

/// Echo one value the way `sofuu repl` echoes it.
///
/// `display` is `repl_inspect`'s output, SGR colours included. The REPL always
/// has a human in front of it; `eval`'s stdout is routinely piped
/// (`sofuu eval '1+1' | …`), so the paint is dropped when stdout is not a
/// terminal. The FORMAT is the REPL's in both cases — only the colour goes.
fn print_repl_value(display: &str) {
    // SAFETY: isatty(1) has no preconditions and no side effects.
    let text = if unsafe { libc::isatty(1) } == 1 {
        display.to_string()
    } else {
        strip_sgr(display)
    };
    // Write like console.error/log do (lock + ignore the error): `sofuu eval`
    // is fed into `grep -q` / `head` pipelines, and a closed pipe must end the
    // command, not abort it with `println!`'s broken-pipe panic.
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{text}");
    let _ = out.flush();
}

/// Drop ANSI SGR sequences (ESC '[' … final letter).
///
/// cli_protocols.rs, doctor.rs and modules/process.rs each carry this same
/// private helper for the same reason: a value read back through `eval_repl`
/// arrives painted, and whatever consumes it wants the text.
fn strip_sgr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
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

/// S5 (2026-10-23): pull `--policy <file>` / `--policy=<file>` out of a CLI
/// argument list before the command's own parser sees it.
fn take_policy_arg(args: &[String]) -> (Option<String>, Vec<String>) {
    let mut out: Vec<String> = Vec::with_capacity(args.len());
    let mut path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--policy" {
            if i + 1 < args.len() {
                path = Some(args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        } else if let Some(rest) = args[i].strip_prefix("--policy=") {
            path = Some(rest.to_string());
            i += 1;
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    (path, out)
}

/// `--on-error=<mode>` / `--on-error <mode>` - pull a run's terminal-failure
/// policy out of an argument list before the command's own parser sees it
/// (OBSERVABILITY-SPEC §3). Shaped exactly like take_policy_arg, so the
/// pre-parsers in front of the runtime can be read as one family.
fn take_on_error_arg(args: &[String]) -> (Option<String>, Vec<String>) {
    let mut out: Vec<String> = Vec::with_capacity(args.len());
    let mut mode: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--on-error" {
            if i + 1 < args.len() {
                mode = Some(args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        } else if let Some(rest) = args[i].strip_prefix("--on-error=") {
            mode = Some(rest.to_string());
            i += 1;
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    (mode, out)
}

/// The three modes the agent driver accepts (SPEC §3). A typo fails the
/// command here instead of silently running with the default policy - the
/// one thing worse than no policy is thinking you set one.
fn parse_on_error_mode(mode: &str) -> Result<&'static str, String> {
    match mode {
        "auto-retry" => Ok("auto-retry"),
        "fail-stop" => Ok("fail-stop"),
        "interactive" => Ok("interactive"),
        other => Err(format!(
            "--on-error '{other}': want auto-retry, fail-stop, or interactive"
        )),
    }
}

/// `sofuu chat --on-error=<mode>`: the chat turn engine is the Rust UI in
/// chat.rs, whose ChatConfig has no error-policy field, so the flag cannot
/// ride through as an option the way `sofuu agent run`'s does. Its contract is the
/// ENVIRONMENT instead: SOFUU_ON_ERROR is exported for this process before
/// the chat config is loaded, which is the one channel a chat-side reader can
/// consume without a signature change (and the same variable an operator or a
/// wrapper script can set without the flag at all).
fn apply_on_error_env(mode: &str) -> Result<(), String> {
    let canonical = parse_on_error_mode(mode)?;
    std::env::set_var("SOFUU_ON_ERROR", canonical);
    Ok(())
}

/// Load a policy JSON file and install it. Fail closed: an unreadable,
/// non-JSON or rejected policy aborts the command instead of running with
/// defaults.
pub(crate) fn apply_policy_file(rt: &SofuuRuntime, path: &str) -> Result<(), String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("--policy {path}: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("--policy {path}: not valid JSON: {e}"))?;
    if !v.is_object() {
        return Err(format!("--policy {path}: must be a JSON object"));
    }
    apply_policy_value(rt, &v, &format!("--policy {path}"))
}

/// Install an already-parsed policy document through the runtime's own
/// `sofuu.policy.set` (one validator for file, flags and JS callers).
pub(crate) fn apply_policy_value(
    rt: &SofuuRuntime,
    doc: &serde_json::Value,
    label: &str,
) -> Result<(), String> {
    let json = serde_json::to_string(doc).map_err(|e| format!("{label}: {e}"))?;
    let code = format!(
        "var __p = sofuu.policy.set({json}); var __o = JSON.parse(__p); if (!__o.ok) throw new Error(__o.error || 'invalid policy');"
    );
    if rt.eval_string(&code, "<policy>") != 0 {
        return Err(format!("{label}: rejected by the policy parser"));
    }
    Ok(())
}

/// R7.P3 shortcut flags: `--read-only`, `--no-net`, `--confirm-all`.
#[derive(Default, Clone, Copy)]
struct GuardFlags {
    read_only: bool,
    no_net: bool,
    confirm_all: bool,
}

impl GuardFlags {
    fn any(self) -> bool {
        self.read_only || self.no_net || self.confirm_all
    }
    /// One line for stderr, so the operator sees the guard took effect.
    fn describe(self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if self.read_only {
            parts.push("read-only (no writes, no shell)");
        }
        if self.no_net {
            parts.push("no-net (outbound denied)");
        }
        if self.confirm_all {
            parts.push("confirm-all (every tool call asks)");
        }
        parts.join(" · ")
    }
    /// Narrow a policy document. Only ever TIGHTENS: each flag sets a key the
    /// evaluator treats as a floor, and nothing is removed.
    fn apply(self, doc: &mut serde_json::Value) {
        if !doc.is_object() {
            *doc = serde_json::json!({});
        }
        if self.read_only {
            doc["fs"]["write"] = serde_json::json!("off");
            doc["shell"]["enabled"] = serde_json::json!(false);
        }
        if self.no_net {
            doc["net"]["enabled"] = serde_json::json!(false);
        }
        if self.confirm_all {
            doc["approval"]["mode"] = serde_json::json!("always");
        }
    }
}

/// Pull the guard flags out of an argument list (long form only, no values).
/// Deliberately exact — no `--yes`/`--offline` aliases: `--yes` already means
/// "confirm" inside the chat `/forget` command line, and an alias that is not
/// in `--help` is a trap for whoever scripts it.
fn take_guard_flags(args: &[String]) -> (GuardFlags, Vec<String>) {
    let mut g = GuardFlags::default();
    let mut out: Vec<String> = Vec::with_capacity(args.len());
    for a in args {
        match a.as_str() {
            "--read-only" => g.read_only = true,
            "--no-net" => g.no_net = true,
            "--confirm-all" => g.confirm_all = true,
            _ => out.push(a.clone()),
        }
    }
    (g, out)
}

/// Install the effective policy for a command that accepted `--policy` and
/// the guard flags: an explicit file (fail closed if unreadable), otherwise
/// the layered on-disk policy — narrowed by the flags. No file and no flags
/// leaves the runtime's own boot-time load untouched.
pub(crate) fn install_command_policy(
    rt: &SofuuRuntime,
    policy_path: Option<&str>,
    guard: GuardFlags,
) -> Result<(), String> {
    if policy_path.is_none() && !guard.any() {
        return Ok(());
    }
    let mut doc = match policy_path {
        Some(p) => {
            let raw = std::fs::read_to_string(p).map_err(|e| format!("--policy {p}: {e}"))?;
            let v: serde_json::Value = serde_json::from_str(&raw)
                .map_err(|e| format!("--policy {p}: not valid JSON: {e}"))?;
            if !v.is_object() {
                return Err(format!("--policy {p}: must be a JSON object"));
            }
            v
        }
        /* Base the flags on what is already in force, so `--no-net` does not
         * silently discard a layered fs jail. An invalid on-disk policy is an
         * error rather than a reset to permissive defaults. */
        None => sofuu_core::rt::policy::layered_policy_from_disk()?,
    };
    if guard.any() {
        eprintln!("\x1b[90m[sofuu] policy guard: {}\x1b[0m", guard.describe());
        guard.apply(&mut doc);
    }
    apply_policy_value(rt, &doc, "command policy")
}

/// `sofuu agent list` / `sofuu agent run <name> <task…> [--json]` —
/// the headless agent surface (PLAN-AGENTS A8.2). Everything runs in the
/// engine's shipped agent runtime (src/js/agent.js); this is just CLI
/// sugar: load ~/.sofuu/agents/*.js, then list or one-shot run.
/// `on_error` is the run's terminal-failure policy (OBSERVABILITY-SPEC
/// §3), pre-parsed by the dispatcher so a typo fails before the engine
/// starts; None leaves the driver's own TTY-aware default in place.
fn cmd_agent(rt: &SofuuRuntime, args: &[String], on_error: Option<&str>) -> i32 {
    if args.len() < 3 {
        eprintln!("\x1b[1mUsage:\x1b[0m");
        eprintln!("  sofuu agent list");
        eprintln!("  sofuu agent run <name> <task…> [--json]");
        eprintln!("  sofuu agent init <name>          (scaffold ~/.sofuu/agents/<name>.js)");
        eprintln!("  sofuu agent validate <file.js>   (define dry-run, precise errors)");
        return 1;
    }
    match args[2].as_str() {
        "init" => {
            // P4.3 (agentic-core Phase 4): scaffold a def file into the same
            // dir `sofuu agent list`/`run` already load from. Writing happens
            // through the runtime's own fs bridge (HOME resolution + jail
            // semantics live there); the name is sanitised here so it can
            // never traverse out of the agents dir.
            let name = args[3].as_str();
            if name.is_empty()
                || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
                || name.starts_with('.')
            {
                eprintln!("\x1b[31mError:\x1b[0m agent name must match [A-Za-z0-9_-] (dots only in a trailing .js-less id)");
                return 1;
            }
            let name_js = serde_json::to_string(name).unwrap_or_else(|_| "\"\"".into());
            // The JS env() bridge does not carry HOME in every launch context
            // (a smoke test scaffolded ./.sofuu into the CWD) — resolve the
            // home dir HERE and hand the JS the absolute agents dir.
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_default();
            if home.is_empty() {
                eprintln!("\x1b[31mError:\x1b[0m cannot resolve the home directory (HOME/USERPROFILE unset)");
                return 1;
            }
            let dir_js = serde_json::to_string(&format!("{}/.sofuu/agents", home))
                .unwrap_or_else(|_| "\"\"".into());
            let code = format!(
                r#"(async function () {{
  const dir = {d};
  const file = dir + '/{n}.js';
  if (await sofuu.fs.exists(file)) {{
    console.error('refusing to overwrite existing ' + file);
    process.exit(1);
  }}
  const tpl = [
    "// sofuu agent '{n}' — scaffolded by \`sofuu agent init\`; edit freely.",
    "sofuu.agent.define({{",
    "  name: '{n}',",
    "  system: 'You are {n}. Be direct and precise; verify claims against tool results before stating them.',",
    "  memory: 'agent',            // 'agent' | 'shared' | 'off'",
    "  rlm: 'auto',",
    "  // approve: 'none',         // 'sensitive' | 'all' — prompts the host via opts.onApproval",
    "  // shell: 'readonly',       // or allow-list: ['git *', 'npm test*']",
    "  // netPolicy: {{ allow: ['example.com'] }},",
    "  // budget: {{ verify: true, maxSteps: 16 }},",
    "  // responseSchema: {{ type: 'object', required: ['summary'], properties: {{ summary: {{ type: 'string' }} }} }},",
    "  // learn: true,             // harvest a one-line playbook after successful runs",
    "  provider: 'openai',         // match your \`sofuu config\` provider",
    "  model: '',                  // '' = provider default",
    "  tools: ['web'],             // 'web', {{ builtin: 'bash' }}, inline tools, or {{ mcp: [...] }}",
    "}});",
  ].join('\n');
  await sofuu.fs.mkdir(dir, {{ recursive: true }});
  await sofuu.fs.writeFile(file, tpl + '\n');
  console.log('created ' + file);
  console.log("next: edit the file, then `sofuu agent validate {n}`, then `sofuu agent run {n} \"task\"`");
}})().catch(function (e) {{ console.error(String((e && e.message) || e)); process.exit(1); }});"#,
                d = dir_js, n = name_js.trim_matches('"')
            );
            rt.eval_string(&code, "<agent-init>")
        }
        "validate" => {
            // P4.3: define the file exactly like loadDir does (indirect eval
            // in global scope) but report EVERY define error verbatim and a
            // normalized summary; warnings (unknown delegate targets) do not
            // fail the run, define errors do.
            if args.len() < 4 {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu agent validate' requires a <file.js> path");
                return 1;
            }
            let fjs = serde_json::to_string(&args[3]).unwrap_or_else(|_| "\"\"".into());
            let code = format!(
                r#"(async function () {{
  const f = {f};
  let src;
  try {{ src = await sofuu.fs.readFile(f); }} catch (e) {{
    console.error('read failed: ' + String((e && e.message) || e));
    process.exit(1);
  }}
  if (typeof src !== 'string') src = String(src);
  const before = sofuu.agent.list().map(function (a) {{ return a.name; }});
  try {{ (0, eval)(src); }} catch (e) {{
    console.error('DEFINE ERROR in ' + f + ': ' + String((e && e.message) || e));
    process.exit(1);
  }}
  const defs = sofuu.agent.list().filter(function (a) {{ return before.indexOf(a.name) < 0; }});
  if (!defs.length) {{
    console.error('no sofuu.agent.define() call executed' + (sofuu.agent.list().length ? ' (agents were already loaded)' : ''));
    process.exit(1);
  }}
  const known = sofuu.agent.list().map(function (a) {{ return a.name; }});
  let warn = 0;
  for (const a of defs) {{
    console.log('✓ ' + a.name +
      (a.model ? ' · ' + a.model : '') +
      ' · memory ' + a.memory + ' · ' + a.tools + ' tool' + (a.tools === 1 ? '' : 's') +
      ((a.agents && a.agents.length) ? ' · delegates: ' + a.agents.join(', ') : ''));
    for (const dg of (a.agents || [])) {{
      if (known.indexOf(dg) < 0) {{
        console.log('  ⚠ delegates to unknown agent "' + dg + '" (define it first — runs fail only when it is actually called)');
        warn++;
      }}
    }}
  }}
  console.log(defs.length + ' agent(s) valid' + (warn ? ', ' + warn + ' warning(s)' : ''));
  if (warn > 0) process.exit(0);
}})().catch(function (e) {{ console.error(String((e && e.message) || e)); process.exit(1); }});"#,
                f = fjs
            );
            rt.eval_string(&code, "<agent-validate>")
        }
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
            /* §3: the policy rides through opts.onError. The key is only
             * sent when the operator named a mode, because the driver owns
             * the default (interactive on a TTY, auto-retry otherwise). */
            let on_error_js = match on_error {
                Some(mode) => format!(
                    "onError: {},",
                    serde_json::to_string(mode).unwrap_or_else(|_| "\"auto-retry\"".into())
                ),
                None => String::new(),
            };
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
      {on_error_js}
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
/// `sofuu logs [list | show <run_id> | --follow [run_id] | --last-error]`
/// - the observability surface (OBSERVABILITY-SPEC §4). A thin eval shim
/// exactly like cmd_agent: this fn turns the CLI's flags into the argv array
/// the shipped renderer (src/js/observability.js) takes, the renderer
/// prints, and its return value is the exit code (0 ok, 1 the named run is
/// missing or unreadable, 2 usage). All of the formatting lives in JS so a
/// column change never touches the size-capped binary (§0).
fn cmd_logs(rt: &SofuuRuntime, args: &[String]) -> i32 {
    let mut sub: Option<String> = None;
    let mut rest: Vec<String> = Vec::new();
    let mut follow = false;
    let mut last_error = false;
    for a in &args[2..] {
        match a.as_str() {
            "--follow" | "-f" => follow = true,
            "--last-error" | "--last" => last_error = true,
            "-h" | "--help" => {
                print_logs_help();
                return 0;
            }
            s if sub.is_none() && matches!(s, "list" | "ls" | "show" | "follow" | "last-error") => {
                sub = Some(s.to_string());
            }
            s if s.starts_with('-') => {
                eprintln!("\x1b[31mError:\x1b[0m logs: unknown flag '{s}'");
                print_logs_help();
                return 2;
            }
            s => rest.push(s.to_string()),
        }
    }
    /* --follow / --last-error are FLAGS on the CLI (§4) and subcommands in
     * argv[0] for the renderer; a bare run id is shorthand for show. */
    let mut argv: Vec<String> = Vec::new();
    if last_error {
        argv.push("last-error".to_string());
    } else if follow {
        argv.push("follow".to_string());
    } else if let Some(s) = sub.as_deref() {
        argv.push(if s == "ls" { "list".to_string() } else { s.to_string() });
    } else if !rest.is_empty() {
        argv.push("show".to_string());
    } else {
        argv.push("list".to_string());
    }
    argv.extend(rest);
    let argv_js = serde_json::to_string(&argv).unwrap_or_else(|_| "[]".into());
    let code = format!(
        r#"(async function () {{
  if (!sofuu.observability || typeof sofuu.observability.cli !== 'function') {{
    console.error('logs: the observability driver is not loaded in this build');
    process.exit(1);
  }}
  /* argv[0] is the subcommand (list | show | follow | last-error); the shim
   * above only turned flags into it. The renderer owns the printing and its
   * return value is this command's exit code. */
  const rc = await sofuu.observability.cli({argv_js});
  if (typeof rc === 'number' && rc !== 0) process.exit(rc);
}})().catch(function (e) {{ console.error(String((e && e.message) || e)); process.exit(1); }});"#
    );
    rt.eval_string(&code, "<logs>")
}

/// `sofuu resume <run_id> [--step=N] [--force] [--agent=<name>]`
/// - continue an interrupted agent run from its checkpoint
/// (OBSERVABILITY-SPEC §3/§4). The checkpoint IS the rehydration unit: the
/// loop re-enters at its step with the same messages, task, model and
/// provider, so a crash costs one turn instead of a whole run.
///
/// --force resumes a run whose log already ended in run_complete (without it
/// a finished run is refused, because replaying it may spend tokens on an
/// answer that already exists). --step=N picks the checkpoint recorded at
/// step N instead of the newest one: the newest is what a crash needs, an
/// older one is what an investigation needs.
fn cmd_resume(rt: &SofuuRuntime, args: &[String]) -> i32 {
    let mut run_id: Option<String> = None;
    let mut step: Option<u64> = None;
    let mut force = false;
    let mut agent: Option<String> = None;
    let mut on_error: Option<String> = None;
    let mut json = false;
    let mut i = 2;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--force" {
            force = true;
            i += 1;
        } else if a == "--json" {
            json = true;
            i += 1;
        } else if a == "--step" {
            match args.get(i + 1).and_then(|v| v.parse::<u64>().ok()) {
                Some(n) => {
                    step = Some(n);
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m --step wants a step number");
                    return 2;
                }
            }
        } else if let Some(v) = a.strip_prefix("--step=") {
            match v.parse::<u64>() {
                Ok(n) => {
                    step = Some(n);
                    i += 1;
                }
                Err(_) => {
                    eprintln!("\x1b[31mError:\x1b[0m --step wants a step number (got '{v}')");
                    return 2;
                }
            }
        } else if let Some(v) = a.strip_prefix("--agent=") {
            agent = Some(v.to_string());
            i += 1;
        } else if a == "--on-error" {
            match args.get(i + 1) {
                Some(v) => {
                    on_error = Some(v.clone());
                    i += 2;
                }
                None => {
                    eprintln!("\x1b[31mError:\x1b[0m --on-error wants a mode");
                    return 2;
                }
            }
        } else if let Some(v) = a.strip_prefix("--on-error=") {
            on_error = Some(v.to_string());
            i += 1;
        } else if !a.starts_with('-') && run_id.is_none() {
            run_id = Some(a.to_string());
            i += 1;
        } else {
            eprintln!("\x1b[31mError:\x1b[0m resume: unexpected argument '{a}'");
            print_resume_help();
            return 2;
        }
    }
    let Some(run_id) = run_id else {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu resume' requires a <run_id> (sofuu logs lists them)");
        return 2;
    };
    if let Some(mode) = on_error.as_deref() {
        if let Err(e) = parse_on_error_mode(mode) {
            eprintln!("\x1b[31mError:\x1b[0m {e}");
            return 2;
        }
    }
    let run_id_js = serde_json::to_string(&run_id).unwrap_or_else(|_| "\"\"".into());
    let step_js = step.map(|n| n.to_string()).unwrap_or_else(|| "null".to_string());
    let force_js = if force { "true" } else { "false" };
    let agent_js = match agent {
        Some(name) => serde_json::to_string(&name).unwrap_or_else(|_| "null".into()),
        None => "null".to_string(),
    };
    let on_error_js = match on_error {
        Some(mode) => serde_json::to_string(&mode).unwrap_or_else(|_| "null".into()),
        None => "null".to_string(),
    };
    let json_js = if json { "true" } else { "false" };
    let code = format!(
        r#"(async function () {{
  const runId = {run_id_js};
  const wantStep = {step_js};
  const force = {force_js};
  const agentFlag = {agent_js};
  const onError = {on_error_js};
  const asJson = {json_js};

  /* Host functions take ONE JSON string and answer with one JSON string (the
   * sofuu.events.append convention); a missing one collapses to null so every
   * step below can fall back to reading the log itself. */
  function host(fn, args) {{
    try {{
      if (!sofuu.observability || typeof sofuu.observability[fn] !== 'function') return null;
      const raw = sofuu.observability[fn](JSON.stringify(args || {{}}));
      if (typeof raw === 'string') return raw.trim() ? JSON.parse(raw) : null;
      return raw && typeof raw === 'object' ? raw : null;
    }} catch (e) {{ return null; }}
  }}
  function payload(ev) {{
    return (ev && ev.payload && typeof ev.payload === 'object') ? ev.payload : (ev || {{}});
  }}
  async function exists(p) {{
    try {{ return !!(await sofuu.fs.exists(p)); }} catch (e) {{ return false; }}
  }}
  /* The lister knows about the driver's logDir override; the two SPEC §1 paths
   * are the fallback, so resume works even before the Rust surface lands. */
  async function locate() {{
    const res = host('runs', {{ project: '.' }});
    const rows = (res && res.runs) || [];
    for (const r of rows) {{
      if (String(r.run_id) === runId && r.path) {{
        return {{ path: String(r.path), status: String(r.status || '') }};
      }}
    }}
    const cands = ['.sofuu/logs/agent-' + runId + '.jsonl',
                   '.sofuu/sessions/' + runId + '/events.ndjson'];
    for (const c of cands) if (await exists(c)) return {{ path: c, status: '' }};
    return null;
  }}
  async function readEvents(path) {{
    const text = String(await sofuu.fs.readFile(path, 'utf8'));
    const out = [];
    for (const line of text.split('\n')) {{
      const s = line.trim();
      if (!s) continue;
      try {{ const v = JSON.parse(s); if (v && typeof v === 'object') out.push(v); }} catch (e) {{}}
    }}
    return out;
  }}

  const loc = await locate();
  if (!loc) {{
    console.error('resume: no log for run ' + runId + ' (looked in .sofuu/logs and .sofuu/sessions)');
    process.exit(1);
  }}
  let events;
  try {{ events = await readEvents(loc.path); }}
  catch (e) {{
    console.error('resume: cannot read ' + loc.path + ': ' + String((e && e.message) || e));
    process.exit(1);
  }}

  /* A run that already reached run_complete is refused without --force:
   * replaying a completed turn can spend tokens on an answer that exists. A
   * halted run is the case this command exists for, so it is never gated. */
  let terminal = '';
  for (let i = events.length - 1; i >= 0; i--) {{
    const k = String((events[i] && events[i].kind) || '');
    if (k === 'run_complete' || k === 'halt' || k === 'fatal_error') {{ terminal = k; break; }}
  }}
  const status = String(loc.status || (terminal === 'run_complete' ? 'OK' : (terminal ? 'HALTED' : ''))).toUpperCase();
  if (status === 'OK' && !force) {{
    console.error('resume: run ' + runId + ' already completed (status OK); pass --force to resume it anyway');
    process.exit(1);
  }}

  /* The newest checkpoint is what a crash needs (lastCheckpoint scans the log
   * in reverse, SPEC §2); --step=N asks for a specific one instead. */
  let cp = null;
  if (wantStep === null) {{
    const lc = host('lastCheckpoint', {{ project: '.', run_id: runId }});
    if (lc && lc.ok && lc.found && lc.checkpoint && typeof lc.checkpoint === 'object') {{
      cp = lc.checkpoint;
    }} else {{
      console.error('resume: sofuu.observability.lastCheckpoint is unavailable in this build; scanning the log directly');
    }}
  }}
  if (!cp) {{
    const want = wantStep === null ? null : Number(wantStep);
    for (let i = events.length - 1; i >= 0; i--) {{
      if (String((events[i] && events[i].kind) || '') !== 'checkpoint') continue;
      const p = payload(events[i]);
      if (want === null || Number(p.step) === want) {{ cp = p; break; }}
    }}
  }}
  if (!cp) {{
    console.error(wantStep === null
      ? 'resume: run ' + runId + ' has no checkpoint (' + loc.path + '); a run writes one per completed turn'
      : 'resume: run ' + runId + ' has no checkpoint at step ' + String(wantStep) + ' (' + loc.path + ')');
    process.exit(1);
  }}

  /* Which agent: the checkpoint payload carries no name by contract, so the
   * run_start event names it and --agent= covers a log that lost one. */
  let name = agentFlag ? String(agentFlag) : '';
  if (!name) name = String(cp.agent || cp.name || '');
  if (!name) {{
    for (let i = 0; i < events.length; i++) {{
      if (String((events[i] && events[i].kind) || '') !== 'run_start') continue;
      const p = payload(events[i]);
      name = String(p.agent || p.name || p.agentName || '');
      if (name) break;
    }}
  }}
  if (!name) {{
    console.error('resume: cannot tell which agent run ' + runId + ' belongs to (no run_start event); pass --agent=<name>');
    process.exit(1);
  }}
  /* Definitions live in ~/.sofuu/agents — the dir 'sofuu agent run' loads.
   * Resume is a NEW process, so it starts with none defined and every name
   * would read as "unknown agent"; load the dir first, then fail with the
   * directory named instead of the bare lookup error. */
  try {{ await sofuu.agent.loadDir(); }} catch (e) {{}}
  try {{
    const known = sofuu.agent.list().map(function (a) {{ return a.name; }});
    if (known.length && known.indexOf(name) < 0) {{
      console.error('resume: agent "' + name + '" is not defined (loaded: ' + known.join(', ') +
                    '); put it in ~/.sofuu/agents/<name>.js');
      process.exit(1);
    }}
  }} catch (e) {{}}
  const task = (cp.task === undefined || cp.task === null) ? '' : String(cp.task);
  if (!task) {{
    console.error('resume: the checkpoint carries no task, so there is nothing to resume (step ' + String(cp.step) + ')');
    process.exit(1);
  }}

  const cpStep = (cp.step === undefined || cp.step === null) ? null : cp.step;
  const opts = {{ resumeFrom: cp }};
  if (onError) opts.onError = onError;
  if (!asJson) {{
    /* Name-only progress on stderr, exactly like 'sofuu agent run': tool
     * arguments can carry secrets and stderr is scrollback that gets pasted
     * into issues. */
    opts.onStep = function (e) {{
      if (e.kind === 'tool') console.error('  > tool ' + (e.payload && e.payload.name) + '()');
      else if (e.kind === 'delegate') console.error('  > agent ' + (e.payload && e.payload.agent) + '()');
    }};
  }}
  console.error('resume: ' + name + ' from step ' + String(cpStep === null ? '?' : cpStep) +
                ' (' + loc.path + ')' + (onError ? ', on-error ' + onError : ''));

  let r;
  try {{ r = await sofuu.agent.run(name, task, opts); }}
  catch (e) {{
    console.error('resume: ' + String((e && e.message) || e));
    process.exit(1);
  }}
  if (asJson) {{
    console.log(JSON.stringify({{
      answer: r.answer, steps: r.steps, stopped: r.stopped, usage: r.usage,
      resumedFrom: cpStep, runId: r.runId || runId
    }}, null, 2));
  }} else {{
    const u = r.usage || {{}};
    console.log('');
    console.log(r.answer);
    console.log(' resumed from step ' + String(cpStep === null ? '?' : cpStep) + ' · ' + r.steps + ' steps · ' +
                (u.llmCalls || 0) + ' llm calls · ' + (u.toolCalls || 0) + ' tool calls' +
                (r.stopped ? ' · stopped: ' + r.stopped : ''));
  }}
}})().catch(function (e) {{ console.error(String((e && e.message) || e)); process.exit(1); }});"#
    );
    rt.eval_string(&code, "<resume>")
}
/// `sofuu plugin list` / `validate` / `run` / `test` / `init` —
/// the extensibility standard CLI surface (PLAN-PLUGIN-AND-EXTENSIBILITY-STANDARD).
fn cmd_plugin(rt: &SofuuRuntime, args: &[String]) -> i32 {
    if args.len() < 3 {
        eprintln!("\x1b[1mUsage:\x1b[0m");
        eprintln!("  sofuu plugin list [--json]");
        eprintln!("  sofuu plugin validate <file.js|file.ts>");
        eprintln!("  sofuu plugin run <tool_name> [args_json]");
        eprintln!("  sofuu plugin test <tool_name> [args_json]   (alias of run)");
        eprintln!("  sofuu plugin init <name> [--category=tool|provider|store|vectorstore|service|openapi]");
        return 1;
    }
    match args[2].as_str() {
        "list" | "ls" => {
            let json = if args.iter().any(|a| a == "--json") { "true" } else { "false" };
            let code = format!("sofuu.plugins.cliList({json})");
            rt.eval_string(&code, "<plugin-list>")
        }
        "validate" => {
            if args.len() < 4 {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu plugin validate' requires a file argument");
                return 1;
            }
            let file = &args[3];
            if !std::path::Path::new(file).exists() {
                eprintln!("\x1b[31mError:\x1b[0m file '{}' does not exist", file);
                return 1;
            }
            let fjs = serde_json::to_string(file).unwrap_or_else(|_| "\"\"".into());
            let code = format!("sofuu.plugins.cliValidate({fjs})");
            rt.eval_string(&code, "<plugin-validate>")
        }
        // "test" is an alias of "run" (PLUGIN_PLAN DX table). It prefers the
        // isolated driver entry point so the two can diverge later, and falls
        // back to cliRun when the loaded bytecode predates cliTest — an alias
        // must never become a hard dependency on a bytecode refresh.
        "run" | "test" => {
            if args.len() < 4 {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu plugin {}' requires a tool name", args[2]);
                eprintln!("Usage: sofuu plugin <run|test> <tool_name> [args_json]");
                return 1;
            }
            let tool_name = &args[3];
            let raw_args = if args.len() >= 5 { args[4..].join(" ") } else { "{}".to_string() };
            let tool_js = serde_json::to_string(tool_name).unwrap_or_else(|_| "\"\"".into());
            let args_js = serde_json::to_string(&raw_args).unwrap_or_else(|_| "\"{}\"".into());
            let code = format!(
                "(typeof sofuu!=='undefined'&&sofuu.plugins&&typeof sofuu.plugins.cliTest==='function')?sofuu.plugins.cliTest({tool_js},{args_js}):sofuu.plugins.cliRun({tool_js},{args_js})"
            );
            rt.eval_string(&code, "<plugin-run>")
        }
        "init" => {
            if args.len() < 4 {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu plugin init' requires a plugin name");
                eprintln!("Usage: sofuu plugin init <name> [--category=tool|provider|store|vectorstore|service|openapi]");
                return 1;
            }
            let name = &args[3];
            let mut category = "tool";
            for a in &args[4..] {
                if let Some(c) = a.strip_prefix("--category=") {
                    category = c;
                }
            }
            let name_js = serde_json::to_string(name).unwrap_or_else(|_| "\"\"".into());
            let cat_js = serde_json::to_string(category).unwrap_or_else(|_| "\"tool\"".into());
            let code = format!("sofuu.plugins.cliInit({name_js}, {cat_js})");
            rt.eval_string(&code, "<plugin-init>")
        }
        other => {
            eprintln!(
                "\x1b[31mError:\x1b[0m unknown plugin subcommand '{other}' (list | validate | run | test | init)"
            );
            1
        }
    }
}

/// `sofuu run <file>` — run a JS/TS file.
fn cmd_run(rt: &SofuuRuntime, path: &str) -> i32 {
    rt.eval_file(path)
}

/// True when THIS build can place layers on a GPU for `sofuu infer`.
///
/// The whole-layer GPU client is compiled for macOS only (Metal); every
/// other target links the CPU engine alone, and a Mac without a usable Metal
/// device falls back too. Used to keep `--gpu-layers` honest and to stop the
/// banner advertising a CPU+GPU engine on a CPU-only build (audit A).
fn infer_gpu_backend_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            /// Metal device probe (src/metal/*.m) — the same symbol
            /// capabilities.rs reports as gpu.available.
            fn sofuu_metal_available() -> bool;
        }
        // SAFETY: argument-free C getter; returns whether a usable Metal
        // device exists.
        unsafe { sofuu_metal_available() }
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// `--gpu-layers N` on a build that cannot place a single layer on a GPU.
///
/// PC-05 (audit 2026-10-29): on every non-Metal build `--gpu-layers` was
/// accepted, silently dropped, and the run continued on the CPU — while the
/// engine banner talked about a heterogeneous CPU+GPU engine and the flag's
/// documented default (999) meant an operator could not tell whether the
/// request had been honoured. A dropped request must say so, once, on stderr
/// (stdout stays the model's), and the run must continue: a model load is not
/// a place to fail over a placement preference.
///
/// Once per process, not per call and not per layer: the same process can
/// reach this from more than one command path, and only the first warning is
/// information. `--cpu` / `--gpu-layers 0` never reach here, and a host whose
/// GPU probe succeeds never warns.
fn warn_gpu_layers_unavailable(requested: usize) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "[sofuu] --gpu-layers {requested}: this build has no GPU backend for {} \
             (the whole-layer GPU client is macOS/Metal only), so the request cannot be \
             honoured — using the CPU instead. Pass --cpu (or --gpu-layers 0) to make \
             that explicit and silence this warning.",
            std::env::consts::OS
        );
    });
}

/// The layer count this build can actually honour, given what the operator
/// asked for.
///
/// `gpu_layers > 0` on a build without a GPU backend normalises to 0 — the
/// CPU engine is what runs either way, and normalising here keeps the banner
/// and the load branch honest — after warning once (PC-05). `0` means
/// `--cpu` / `--gpu-layers 0` and stays silent by contract.
fn resolve_gpu_layers(requested: usize) -> usize {
    if requested > 0 && !infer_gpu_backend_available() {
        warn_gpu_layers_unavailable(requested);
        return 0;
    }
    requested
}

/// C-01: the process status for a command that evaluated JavaScript.
///
/// Precedence (Node-compatible):
///   1. `process.exitCode` — an explicit choice by the script always wins
///      (including 0 over an error, which is how a script says "expected").
///   2. a non-zero engine rc — compile errors, missing files, bad usage.
///   3. `has_uncaught_error` — an uncaught synchronous throw, an unhandled
///      promise rejection, or a microtask-queue overflow → 1.
///   4. 0.
fn js_exit_code(rt: &SofuuRuntime, rc: i32) -> ExitCode {
    let st = sofuu_core::rt::engine::run_status_raw(rt.engine_ctx());
    let code = match (st.exit_code, rc, st.has_uncaught_error) {
        (Some(chosen), _, _) => chosen,
        (None, rc, _) if rc != 0 => 1,
        (None, _, true) => 1,
        _ => 0,
    };
    // Same truncation the OS applies; never panic on a hostile value.
    ExitCode::from(code.clamp(0, 255) as u8)
}

/// Run a `sofuu compile` artifact's embedded program.
///
/// Order matters: the prelude (argv + the embedded brain handle) is installed
/// BEFORE the program runs, because the program's first line may already read
/// `sofuu.args` or `sofuu.memory.embedded`.
fn run_compiled_payload(payload: &sofuu_core::compiler::EmbeddedPayload) -> ExitCode {
    let rt = match SofuuRuntime::init() {
        Some(rt) => rt,
        None => {
            eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
            return ExitCode::FAILURE;
        }
    };
    let brain = match sofuu_core::compiler::materialize_brain(payload) {
        Ok(brain) => brain,
        Err(e) => {
            /* The artifact still runs: a program that never touches memory is
             * unaffected, and one that does gets the open error with the path
             * it tried. Losing the payload's seed silently would be worse. */
            eprintln!("[sofuu] embedded brain not mounted: {e}");
            None
        }
    };
    install_compiled_prelude(&rt, brain.as_ref());

    // SAFETY: engine_ctx() is this runtime's live QuickJS context.
    let rc = unsafe {
        sofuu_core::rt::engine::eval_compiled_and_run(
            rt.engine_ctx() as *mut sofuu_ffi::qjs::JSContext,
            &payload.bytecode,
            &payload.entry,
        )
    };
    js_exit_code(&rt, rc)
}

/// Install the globals a compiled artifact's program expects.
///
/// `sofuu.args` is the argv after the binary itself — the natural reading for
/// `./agent "task"`. `sofuu.memory.embedded` is `{path, dim, open()}` over the
/// brain the payload carried; `open()` needs the dimension recorded at compile
/// time, which is exactly why the payload stores it.
fn install_compiled_prelude(
    rt: &SofuuRuntime,
    brain: Option<&sofuu_core::compiler::MountedBrain>,
) {
    let argv: Vec<String> = env::args().skip(1).collect();
    let argv_json = serde_json::to_string(&argv).unwrap_or_else(|_| "[]".to_string());
    let embedded = match brain {
        Some(b) => {
            let path = serde_json::to_string(&b.path).unwrap_or_else(|_| "\"\"".to_string());
            format!(
                "try{{S.memory.embedded={{path:{path},dim:{},open:function(d){{return S.memory.open(this.path,d||this.dim);}}}};}}catch(e){{}}",
                b.dim
            )
        }
        None => String::new(),
    };
    let js = format!(
        "(function(){{var S=globalThis.sofuu;if(!S)return;try{{S.args={argv_json};}}catch(e){{}}if(!S.memory)return;{embedded}}})();"
    );
    /* A prelude failure must not be attributed to the program: the try/catch
     * arms keep the run status clean, and eval_string only dirties it on an
     * escaping throw. */
    let _ = rt.eval_string(&js, "<compiled-prelude>");
}

struct CompileArgs {
    entry: String,
    options: sofuu_core::compiler::CompileOptions,
}

/// `sofuu compile <entry.ts|js> [-o <out>] [--embed-brain <file.qtsq>] [--no-strip]`
///
/// A trailing flag that expects a value is an ERROR, not a silently ignored
/// word: `-o` with nothing after it used to mean "write bundle.js" in
/// `sofuu bundle`, and that class of typo must not ship a mislabelled binary.
fn parse_compile_args(rest: &[String]) -> Result<CompileArgs, String> {
    let mut options = sofuu_core::compiler::CompileOptions::default();
    let mut entry: Option<String> = None;
    let mut i = 0;
    while i < rest.len() {
        let a = rest[i].as_str();
        match a {
            "-o" | "--output" => {
                let v = rest
                    .get(i + 1)
                    .ok_or_else(|| "-o requires an output path".to_string())?;
                options.output = Some(v.clone());
                i += 2;
            }
            "--embed-brain" => {
                let v = rest
                    .get(i + 1)
                    .ok_or_else(|| "--embed-brain requires a .qtsq path".to_string())?;
                options.embed_brain = Some(v.clone());
                i += 2;
            }
            "--no-strip" => {
                options.no_strip = true;
                i += 1;
            }
            _ if a.starts_with("--output=") => {
                options.output = Some(a["--output=".len()..].to_string());
                i += 1;
            }
            _ if a.starts_with("--embed-brain=") => {
                options.embed_brain = Some(a["--embed-brain=".len()..].to_string());
                i += 1;
            }
            _ if a.starts_with('-') => {
                return Err(format!("unknown option '{a}'"));
            }
            _ => {
                if entry.is_some() {
                    return Err(format!(
                        "unexpected argument '{a}' — sofuu compile takes one entry file"
                    ));
                }
                entry = Some(a.to_string());
                i += 1;
            }
        }
    }
    let Some(entry) = entry else {
        return Err("'sofuu compile' requires an entry file\nUsage: sofuu compile <entry.ts|js> [-o <out>]".to_string());
    };
    Ok(CompileArgs { entry, options })
}

fn print_compile_report(r: &sofuu_core::compiler::CompileReport) {
    println!("\x1b[1m⚡ sofuu compile\x1b[0m");
    println!("  Output: \x1b[1m{}\x1b[0m", r.output);
    match r.brain {
        Some(brain) => println!(
            "  Size:   {} B total ({} B base + {} B payload; bytecode {} B in, {} B stored{}; brain {} B)",
            r.total,
            r.base,
            r.payload,
            r.bytecode_raw,
            r.bytecode_stored,
            if r.gzipped { " gzip" } else { "" },
            brain
        ),
        None => println!(
            "  Size:   {} B total ({} B base + {} B payload; bytecode {} B in, {} B stored{})",
            r.total,
            r.base,
            r.payload,
            r.bytecode_raw,
            r.bytecode_stored,
            if r.gzipped { " gzip" } else { "" }
        ),
    }
    println!(
        "  Flags:  {}{}",
        if r.stripped { "stripped " } else { "unstripped " },
        if r.signed { "ad-hoc signed" } else { "unsigned" }
    );
    println!("\x1b[90mRun with: {}\x1b[0m", artifact_run_hint(&r.output));
}

/// How to tell the user to launch the artifact they just compiled.
///
/// `./name` only makes sense for a bare file name: an absolute path or one
/// that already carries a directory component is printed as itself, so the
/// old unconditional `./` prefix no longer produces `.//abs/path`.
fn artifact_run_hint(path: &str) -> String {
    match std::path::Path::new(path).parent() {
        Some(parent) if !parent.as_os_str().is_empty() => path.to_string(),
        _ => format!("./{path}"),
    }
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
/// Shared body of `sofuu chat ...` and of a bare `sofuu --provider ...`.
fn run_chat(args: &[String]) -> ExitCode {
    let (policy, clean) = take_policy_arg(args);
    let (guard, clean) = take_guard_flags(&clean);
    let rt = match SofuuRuntime::init() {
        Some(rt) => rt,
        None => {
            eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = install_command_policy(&rt, policy.as_deref(), guard) {
        eprintln!("\x1b[31mError:\x1b[0m {e}");
        return ExitCode::FAILURE;
    }
    let rc = cmd_chat(&rt, &clean);
    if rc == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn cmd_chat(rt: &SofuuRuntime, args: &[String]) -> i32 {
    /* OBSERVABILITY-SPEC §4: --on-error is accepted here as an ENVIRONMENT
     * switch (apply_on_error_env), so it is exported BEFORE ChatConfig::load
     * reads config: every reader in the chat path then sees one value, and
     * a pre-set SOFUU_ON_ERROR stays the operator's choice when the flag is
     * absent. A bad mode fails here, not mid-turn. */
    if let Some(rest) = args.get(2..) {
        let (mode, _) = take_on_error_arg(rest);
        if let Some(mode) = mode {
            if let Err(e) = apply_on_error_env(&mode) {
                eprintln!("\x1b[31mError:\x1b[0m {e}");
                return 1;
            }
        }
    }
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
                    /* Local inference engines are first-class names (PLUGIN_PLAN
                     * Phase 2). "ollama" needs no endpoint: rt/ai.rs already
                     * maps it to the local provider and its
                     * http://localhost:11434/api/chat default. "llamacpp" is an
                     * OpenAI-compatible llama-server, so it needs one. */
                    /* lmstudio and vllm ship as adapters (src/js/providers.js)
                     * and chat.js already advertises them, so the flag must
                     * accept them too. Both are OpenAI-compatible with their
                     * own local default ports, supplied by the adapter — no
                     * base_url fill-in needed here. */
                    "openai" | "anthropic" | "local" | "ollama" | "lmstudio" | "vllm" => {
                        cfg.provider = p;
                        i += 2;
                    }
                    "llamacpp" => {
                        cfg.provider = p;
                        /* Without a default, a bare -p llamacpp would fall
                         * through to the cloud OpenAI endpoint. Only fill it in
                         * when no --base-url appears anywhere on the command
                         * line, since the flag may come after -p. */
                        if cfg.base_url.is_empty() && !args.iter().any(|a| a == "--base-url") {
                            cfg.base_url = "http://127.0.0.1:8080/v1".to_string();
                        }
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
                                "\x1b[31mError:\x1b[0m unknown provider '{p}' (known: openai, anthropic, local, ollama, llamacpp, lmstudio, vllm)."
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
            /* Already exported by the prescan at the top of this fn; consume
             * the flag and its value so neither is read as a positional or
             * swallowed as another flag's value. */
            "--on-error" => i += 2,
            _ if a.starts_with("--on-error=") => i += 1,
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
        "export" => {
            // P3 (R4.2): the portable session document (SPEC §5.2). Secrets
            // are stripped before anything leaves the process (R4.6).
            let Some(id) = args.get(3) else {
                println!("  Usage: sofuu session export <id> [--out <file>]\n");
                return 1;
            };
            let out = (4..args.len())
                .find(|&i| args[i] == "--out")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
            sofuu_core::rt::session_doc::cli_export(&project.to_string_lossy(), id, out)
        }
        "import" => {
            // P3 (R4.2/R4.4): validate + persist a session document; `-` is
            // stdin. Import is idempotent and reports anything it refuses.
            let Some(src) = args.get(3) else {
                println!("  Usage: sofuu session import <file|->\n");
                return 1;
            };
            sofuu_core::rt::session_doc::cli_import(&project.to_string_lossy(), src)
        }
        other => {
            eprintln!("\x1b[31mError:\x1b[0m unknown session command '{other}' (list|show|inspect|repair|migrate|storage-stats|prune|export|import)");
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
    let mut model_id = String::new();
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
            "--model-id" => match value_arg(args, i) {
                Some(v) => {
                    model_id = v;
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
    eprintln!("\x1b[2m    GET  /health               (liveness, no auth)\x1b[0m");
    eprintln!("\x1b[2m    GET  /v1/models\x1b[0m");
    eprintln!("\x1b[2m    POST /v1/chat/completions  (OpenAI SSE stream & json)\x1b[0m");
    eprintln!("\x1b[2m    POST /v1/completions       (text completions)\x1b[0m");
    eprintln!("\x1b[2m    POST /v1/embeddings        (OpenAI embeddings)\x1b[0m");
    eprintln!("\x1b[2m    POST /v1/sessions          (process-local session history)\x1b[0m");
    eprintln!("\x1b[2m    POST /v1/agent/runs        (agent loop, SSE events)\x1b[0m");
    eprintln!("\x1b[2m    GET  /v1/tools             (inventory + policy verdict)\x1b[0m");
    eprintln!("\x1b[2m    POST /v1/tools/:name/call  (execute under policy)\x1b[0m");
    eprintln!("\x1b[2m    POST /infer                (direct local infer)\x1b[0m");
    eprintln!("\x1b[2m    POST /remember             {{text, role?, source?}}\x1b[0m");
    eprintln!("\x1b[2m    GET  /recall?q=…&k=…\x1b[0m");
    eprintln!("\x1b[2m    GET  /memories?tier=…&role=…&text=…&offset=…&limit=…\x1b[0m");
    eprintln!("\x1b[2m    POST /forget               {{id}}\x1b[0m");
    eprintln!("\x1b[2m    POST /share                (export card: records + vectors + manifest)\x1b[0m");
    eprintln!();

    /* Workstream H1: the handler lives in src/js/serve.js (a shipped
     * driver). This boot call passes the resolved config; everything else —
     * routes, templates, usage, stop sequences, policy — is testable JS. */
    let cfg = serde_json::json!({
        "brainPath": brain_path,
        "modelPath": model_path,
        "modelId": model_id,
        "token": token,
        "host": host,
        "port": port,
        "version": VERSION,
    });
    let serve_js = format!("sofuu.serve.start({});", cfg);
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

/// `print!` panics when its stream is gone, and that panic report would be
/// written to the very pipe that closed. Turn exactly that panic into the
/// silent SIGPIPE-status exit the old default disposition gave (status 141,
/// no output); every other panic keeps the normal report.
fn quiet_broken_pipe_exit() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("");
        /* std's print helpers panic with exactly these prefixes. */
        if msg.starts_with("failed printing to stdout")
            || msg.starts_with("failed printing to stderr")
        {
            sofuu_core::rt::broken_pipe_exit();
        }
        prev(info);
    }));
}

fn main() -> ExitCode {
    // conhost (cmd.exe) prints ANSI escapes as literal text until the
    // process opts in — enable VT processing before anything prints
    // (no-op off-Windows).
    sofuu_core::rt::tui::windows_enable_vt();
    /* Unix convention: a reader that closes the pipe early (`| head`, `| grep
     * -q`) is a normal end of output, not a crash — observed on the Pi for
     * `sofuu capabilities | head -c 400`. But SIGPIPE must NOT be given its
     * DEFAULT disposition here, because this binary is not only a printer:
     * `run`/`serve`/`chat` hold sockets open, and libuv's writev() raises
     * SIGPIPE whenever the peer has already hung up. With the default the
     * FIRST client that abandons a response kills the whole process (status
     * 141) and takes the listening socket with it — measured on Linux: an SSE
     * client that aborts on an in-stream error frame made a `sofuu run` mock
     * server die mid-response, and every later turn then truthfully reported
     * "Couldn't connect to server" (tests/chat_no_response_e2e.sh, T2.6).
     * Rust's SIG_IGN default therefore stays, and the quiet `| head` exit is
     * reproduced exactly where the EPIPE surfaces — on OUR OWN output
     * streams, never on a socket: quiet_broken_pipe_exit here for `print!`
     * (which panics), rt::broken_pipe_exit from the console writer (which
     * swallows its error). */
    quiet_broken_pipe_exit();
    /* SIGINT must stop run/chat/agent run/serve the way SIGTERM does — exit
     * 130, no hang — including when a tool holds the JS thread and the uv
     * loop therefore never reaches the signal dispatch. SIGTERM keeps its
     * default disposition (rc 143). */
    sofuu_core::modules::process::cli_install_sigint_stop();
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

    /* Standalone payload: a `sofuu compile` artifact carries its program (and
     * optionally its brain) behind a 32-byte append trailer. Payload mode wins
     * over every subcommand — the compiled agent owns its argv, so `./agent
     * --help` is the agent's flag, not Sofuu's. The probe is a seek to EOF and
     * a 32-byte read on a plain binary; `SOFUU_PLAIN=1` opts back into CLI
     * mode when a compiled artifact has to be inspected or recompiled. */
    match sofuu_core::compiler::detect_embedded_payload() {
        Ok(Some(payload)) => return run_compiled_payload(&payload),
        Ok(None) => {}
        /* A trailer that proves a payload exists but whose bytes do not check
         * out is fatal: falling through would run the wrong program. */
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {e}");
            return ExitCode::FAILURE;
        }
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

    // R-29: a per-subcommand help request is answered HERE, before dispatch —
    // so before SofuuRuntime::init(), before any listener binds, and before
    // cmd_serve() can create ~/.sofuu/serve_token. `sofuu serve --help` must
    // never start a server.
    if wants_help(&args) && print_subcommand_help(cmd) {
        return ExitCode::SUCCESS;
    }

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
            // PC-09: `-q` / `--no-print` opts out of the completion echo. The
            // flag may sit before or after the code (`sofuu eval -q "1+1"`,
            // `sofuu eval "1+1" -q`); the first non-flag argument is the code,
            // as it always was, and any further non-flag argument stays
            // ignored (`eval '<code>' extra` never meant anything).
            let mut print_value = true;
            let mut code: Option<&str> = None;
            for a in &args[2..] {
                if a == "-q" || a == "--no-print" {
                    print_value = false;
                } else if code.is_none() {
                    code = Some(a.as_str());
                }
            }
            let Some(code) = code else {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu eval' requires a code string argument");
                return ExitCode::FAILURE;
            };
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_eval(&rt, code, print_value);
            js_exit_code(&rt, rc)
        }
        "dev" | "web" => {
            let mut port = 3000u16;
            let mut host = "127.0.0.1".to_string();
            let mut root_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            let mut model_path = None;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--port" | "-p" => {
                        if i + 1 < args.len() {
                            port = args[i + 1].parse().unwrap_or(3000);
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    "--host" | "-h" => {
                        if i + 1 < args.len() {
                            host = args[i + 1].clone();
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    "--model" | "-m" => {
                        if i + 1 < args.len() {
                            model_path = Some(std::path::PathBuf::from(&args[i + 1]));
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    dir if !dir.starts_with('-') => {
                        root_dir = std::path::PathBuf::from(dir);
                        i += 1;
                    }
                    _ => {
                        i += 1;
                    }
                }
            }

            let cfg = sofuu_core::frontend::DevServerConfig {
                root_dir,
                host,
                port,
                model_path,
            };

            if let Err(e) = sofuu_core::frontend::start_dev_server(cfg) {
                eprintln!("\x1b[31mServer error:\x1b[0m {e}");
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        "compile" => {
            let opts = match parse_compile_args(&args[2..]) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("\x1b[31mError:\x1b[0m {e}");
                    return ExitCode::FAILURE;
                }
            };
            match sofuu_core::compiler::compile(&opts.entry, &opts.options) {
                Ok(r) => {
                    print_compile_report(&r);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("\x1b[31mError:\x1b[0m {} — {}", e.code, e.message);
                    ExitCode::FAILURE
                }
            }
        }
        "compile-bc" => {
            if args.len() < 4 {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu compile-bc <input.js> <output.bc>'");
                return ExitCode::FAILURE;
            }
            let in_path = &args[2];
            let out_path = &args[3];
            let src = match std::fs::read_to_string(in_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error reading {in_path}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let rt = unsafe { sofuu_ffi::qjs::JS_NewRuntime() };
            let ctx = unsafe { sofuu_ffi::qjs::JS_NewContext(rt) };
            let bc = unsafe { sofuu_core::shipped::compile_to_bytecode(ctx, &src, in_path) };
            unsafe {
                sofuu_ffi::qjs::JS_FreeContext(ctx);
                sofuu_ffi::qjs::JS_FreeRuntime(rt);
            }
            match bc {
                Some(bytes) => {
                    if let Err(e) = std::fs::write(out_path, &bytes) {
                        eprintln!("Error writing {out_path}: {e}");
                        return ExitCode::FAILURE;
                    }
                    println!("Compiled {in_path} ({} bytes) -> {out_path} ({} bytes)", src.len(), bytes.len());
                    ExitCode::SUCCESS
                }
                None => {
                    eprintln!("Compilation failed for {in_path}");
                    ExitCode::FAILURE
                }
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
        "capabilities" => {
            println!("{}", sofuu_core::capabilities::manifest_json());
            ExitCode::SUCCESS
        }
        "tools" | "policy" | "mcp" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = if cmd == "tools" {
                cli_protocols::tools(&rt, &args)
            } else if cmd == "policy" {
                cli_protocols::policy(&rt, &args)
            } else {
                cli_protocols::mcp(&rt, &args)
            };
            if rc == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        "infer" => {
            // Pool resilience smoke test (no model needed): drives the three
            // paths that used to kill the engine — nested dispatch, a throwing
            // task, and a normal job after the throw — on the live pool.
            // Prints the contained-failure counters as a device receipt.
            if std::env::var_os("SOFUU_POOL_SELFTEST").is_some() {
                let rc = sofuu_core::infer::quant::pool_selftest();
                eprintln!(
                    "[pool] selftest rc={:#x} contained_failures={} (1 injected by the selftest) \
                     short_thread_fallback={}",
                    rc,
                    sofuu_core::infer::quant::pool_failures(),
                    sofuu_core::infer::quant::pool_short_fallback()
                );
                return if rc == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE };
            }
            if let Ok(spec) = std::env::var("SOFUU_BENCH_GEMV") {
                // type,rows,cols,iters — serial-decode weight-stream rate.
                let parts: Vec<usize> = spec
                    .split(',')
                    .map(|p| p.trim().parse::<usize>().unwrap_or(0))
                    .collect();
                if parts.len() == 4 && parts.iter().all(|&v| v > 0) {
                    eprintln!(
                        "[bench] SOFUU_BENCH_GEMV type={} rows={} cols={} iters={}",
                        parts[0], parts[1], parts[2], parts[3]
                    );
                    sofuu_core::infer::quant::gemv_bench(parts[0] as u32, parts[1], parts[2], parts[3]);
                    return ExitCode::SUCCESS;
                }
            }
            if let Ok(spec) = std::env::var("SOFUU_BENCH_ATTN") {
                // b,heads,kv_heads,head_dim,seq_len,iters — decode attention cost.
                let parts: Vec<usize> = spec
                    .split(',')
                    .map(|p| p.trim().parse::<usize>().unwrap_or(0))
                    .collect();
                if parts.len() == 6 && parts.iter().all(|&v| v > 0) {
                    eprintln!("[bench] SOFUU_BENCH_ATTN {}", spec);
                    unsafe {
                        sofuu_core::infer::quant::bench_attn(
                            parts[0] as i32, parts[1] as i32, parts[2] as i32,
                            parts[3] as i32, parts[4] as i32, parts[5] as i32,
                        );
                    }
                    return ExitCode::SUCCESS;
                }
            }
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
            let prompt = args[3].clone();

            let mut max_tokens = 256;
            let mut temperature = 0.7f32;
            let mut threads = 0usize;
            let mut sliding_window = 0usize;
            let mut custom_ctx = 0usize;
            let mut gpu_layers = 999usize;
            // Lookahead speculative decoding is the OUTPUT default on the GGUF
            // batched path (Pi 5 Llama-1B Q8_0: 9.0 -> 19.7 tk/s, temp-0 outputs
            // token-identical; the engine only engages it on the GGUF backend).
            // SOFUU_LOOKAHEAD=0 or --no-lookahead restores serial decoding.
            let mut lookahead = std::env::var_os("SOFUU_LOOKAHEAD")
                .map(|v| v != "0")
                .unwrap_or(true);
            let mut snapkv = false;
            let mut dynamic_ffn = false;
            let mut kivi = false;
            let mut kv_dtype_explicit = false;
            let mut kivi_bits = 4usize;
            let mut streaming_weights = false;
            let mut chat = false;
            let mut score_path: Option<String> = None;
            let mut score_window = 0usize;
            let mut trace_path: Option<String> = None;

            let mut i = 4;
            while i < args.len() {
                /* (the --score-after-model form is handled before the loop) */
                if (args[i] == "--max-tokens" || args[i] == "-n") && i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        max_tokens = v;
                    }
                    i += 2;
                } else if (args[i] == "--temp" || args[i] == "--temperature") && i + 1 < args.len() {
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
                } else if args[i] == "--no-lookahead" {
                    lookahead = false;
                    i += 1;
                } else if args[i] == "--snapkv" {
                    snapkv = true;
                    i += 1;
                } else if args[i] == "--dynamic-ffn" {
                    dynamic_ffn = true;
                    i += 1;
                } else if args[i] == "--kv-dtype" && i + 1 < args.len() {
                    kv_dtype_explicit = true;
                    // GPU decode KV cache precision: 4 (default) or 2 quantize
                    // the history into KIVI-layout packed blocks; f32 keeps the
                    // byte-identical f32 cache. Communicated through the env so
                    // every route reads one knob (SOFUU_KV_DTYPE overrides).
                    let v = args[i + 1].to_ascii_lowercase();
                    if !matches!(
                        v.as_str(),
                        "f32" | "4" | "4bit" | "q4" | "2" | "2bit" | "q2" | "f16k" | "k16" | "8" | "q8" | "8bit"
                    ) {
                        eprintln!("[sofuu] --kv-dtype accepts f32, 4 or 2 (got {})", args[i + 1]);
                        std::process::exit(2);
                    }
                    if kivi && v != "f32" {
                        eprintln!(
                            "[sofuu] --kivi cannot run on a quantized GPU KV cache: pass --kv-dtype f32 (or drop one of the two flags)"
                        );
                        std::process::exit(2);
                    }
                    let canonical = match v.as_str() {
                        "f32" => "f32",
                        "2" | "2bit" | "q2" => "2",
                        "f16k" | "k16" => "f16k",
                        "8" | "q8" | "8bit" => "8",
                        _ => "4",
                    };
                    std::env::set_var("SOFUU_KV_DTYPE", canonical);
                    i += 2;
                } else if args[i] == "--kivi" {
                    kivi = true;
                    // --kivi quantizes the f32 history itself on the CPU layer
                    // loop; the quantized GPU cache keeps only a ring of f32
                    // rows, so the two cannot share a cache. f32 unless the
                    // operator asked for a specific dtype.
                    if kv_dtype_explicit {
                        if std::env::var("SOFUU_KV_DTYPE").ok().as_deref() != Some("f32") {
                            eprintln!(
                                "[sofuu] --kivi cannot run on a quantized GPU KV cache: pass --kv-dtype f32 (or drop one of the two flags)"
                            );
                            std::process::exit(2);
                        }
                    } else {
                        std::env::set_var("SOFUU_KV_DTYPE", "f32");
                        eprintln!(
                            "[sofuu infer] --kivi: keeping the f32 KV cache (the CPU KIVI stream does its own quantization)"
                        );
                    }
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
                } else if args[i] == "--chat" {
                    chat = true;
                    i += 1;
                } else if args[i] == "--score" && i + 1 < args.len() {
                    score_path = Some(args[i + 1].clone());
                    i += 2;
                } else if args[i] == "--score-window" && i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        score_window = v;
                    }
                    i += 2;
                } else if args[i] == "--trace" && i + 1 < args.len() {
                    // §8 P3 surface: per-token forward timing + top logprob
                    // as JSONL (one line per generated token).
                    trace_path = Some(args[i + 1].clone());
                    i += 2;
                } else if args[i].starts_with('-') && args[i] != "-" {
                    // A silently-dropped flag is a correctness hazard: e.g.
                    // "--temperature 0" would leave the sampler at the 0.7
                    // default with a wall-clock RNG seed.
                    eprintln!("[sofuu infer] warning: unrecognized flag `{}` ignored", args[i]);
                    i += 1;
                } else {
                    i += 1;
                }
            }

            // KIVI × speculation: two independent hazards, so a --kivi run
            // decodes serially.
            //   1. The speculative tree commits accepted draft tokens through
            //      commit_speculative_candidates(), which writes the KV cache
            //      but never appends to the KIVI stream — the quantized cache
            //      would silently attend a truncated history (its recent
            //      window lags the accepted positions).
            //   2. The frozen-logits class that produced F-LC4 ("Berlin Berlin
            //      Berlin…" from the step after prefill) lived in
            //      forward_impl's GPU-head early-return; fixed by gating that
            //      return on execute_layers_checked(). Until the tree path
            //      learns to feed the KIVI stream, speculation stays off here.
            if kivi && lookahead {
                lookahead = false;
                eprintln!(
                    "[sofuu infer] --kivi: speculative lookahead does not feed the quantized KV cache (its tree commit bypasses the KIVI stream) — running serial decode"
                );
            }

            // --kivi honesty gate (2026-10-28). The flag sets gpu_layers = 0 and
            // decodes through the CPU layer loop, so on a Metal build it silently
            // disarms the GPU client. Measured on this M2 Pro (Qwen3-0.6B Q4_K_M,
            // ~890-token prompt, 6 runs/arm, quiet machine): --kivi 5.1 tk/s vs
            // 99.3 tk/s for the f32 cache on the GPU decode path - 19x slower. The
            // retired README row ("+168.7%", 26.6 tk/s) compared it against a
            // 9.9 tk/s CPU baseline, which read as a speedup while being a
            // regression against the default. Refuse instead of landing there
            // silently; SOFUU_KIVI_CPU=1 is the explicit opt-in for the CPU stream.
            // CPU-only targets (no Metal client, e.g. the Raspberry Pi 5 profile)
            // keep the flag: there the CPU layer loop is the only path, and
            // kivi.rs stays compiled and reached.
            if kivi {
                // Two explicit acknowledgements are accepted, so a script does
                // not have to learn a new knob: the operator-level env escape
                // hatch, or an explicit --cpu / --gpu-layers 0 on the command.
                let kivi_cpu_opt_in = std::env::var("SOFUU_KIVI_CPU").ok().as_deref() == Some("1")
                    || gpu_layers == 0;
                if cfg!(any(target_os = "macos", target_os = "ios")) {
                    if !kivi_cpu_opt_in {
                        eprintln!("[sofuu infer] refusing --kivi: on a Metal build it sets gpu_layers=0 and decodes through the CPU layer loop.");
                        eprintln!("[sofuu infer] Measured on this M2 Pro (Qwen3-0.6B Q4_K_M, ~890-token prompt, 6 runs/arm, quiet machine):");
                        eprintln!("[sofuu infer]   5.1 tk/s with --kivi vs 99.3 tk/s with the f32 KV cache on the GPU decode path - 19x slower.");
                        eprintln!("[sofuu infer] Replacement: drop --kivi and keep the default f32 cache (it also slides past --ctx). --kv-dtype 8|4 is");
                        eprintln!("[sofuu infer] the GPU quantized cache and measured slower than f32 at every context too (see contract/KV4-OPERATOR-NOTE.md).");
                        eprintln!("[sofuu infer] To run the CPU KIVI stream anyway, say so explicitly: pass --cpu (or set SOFUU_KIVI_CPU=1).");
                        std::process::exit(2);
                    }
                    eprintln!("[sofuu infer] --kivi on the CPU path (explicit): gpu_layers=0, the GPU decode client is disarmed and decode runs the CPU layer loop with the quantized KIVI stream.");
                    eprintln!("[sofuu infer] Measured 5.1 tk/s vs 99.3 tk/s for the f32 cache on the GPU (M2 Pro, ~890-token prompt, 6 runs/arm) - the slow path, chosen explicitly.");
                } else {
                    eprintln!("[sofuu infer] --kivi: quantized KIVI stream on the CPU layer loop (this build has no Metal client, so no GPU path is being disabled).");
                    eprintln!("[sofuu infer] On an M2 Pro the same stream measured 5.1 tk/s vs 99.3 tk/s for f32 on the GPU; on a CPU-only target it is the only path KIVI has.");
                }
                // KIVI is a CPU-path contract, so the load below is CPU-only.
                // Zero it here rather than inside the load branch: the banner
                // printed just after this used to advertise the CPU+GPU engine
                // for a run that could never use the GPU client.
                gpu_layers = 0;
            }

            /* PC-05 (audit 2026-10-29): the whole-layer GPU client exists only
             * when this build links + can use it (Metal on macOS). Off that
             * build `--gpu-layers` used to be silently inert while the banner
             * still advertised "Heterogeneous CPU+GPU" — including for the 999
             * default, which is the value every operator on a CPU-only host
             * runs with. Normalise the request to the hardware that is
             * actually here and say so once (warn_gpu_layers_unavailable);
             * `--cpu` / `--gpu-layers 0` stay silent by contract. */
            gpu_layers = resolve_gpu_layers(gpu_layers);
            let engine_type = if gpu_layers > 0 {
                "⚡ Sofuu Neural Engine (Heterogeneous CPU+GPU)"
            } else {
                "⚡ Sofuu CPU Inference Engine"
            };
            eprintln!("\x1b[36m{}\x1b[0m — loading {}...", engine_type, model_path);
            let t0 = std::time::Instant::now();
            // P7 ring-KV (long-context §8): --kivi loads the CPU engine with
            // a ringed f32 KV cache — the quantized streams carry positions
            // past the ring via the streaming prefill. KIVI is a CPU-path
            // contract (P6), so the GPU whole-layer client is disarmed.
            let load_result = if kivi {
                let ring = std::env::var("SOFUU_KV_RING")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|v| *v >= 256)
                    .unwrap_or(4096);
                gpu_layers = 0;
                sofuu_core::infer::InferEngine::load_with_seq_len_kv_ring_cpu(model_path, custom_ctx, ring)
            } else if gpu_layers == 0 {
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

            // F-LC2 (PLAN-LONG-CONTEXT-PERSISTENCE §8 P9): a raw prompt on an
            // instruct model ignores the format it was trained on — the
            // distilled Qwen emits immediate EOS on some raw-QA prompts and
            // decodes zero tokens. `--chat` wraps the prompt with the model's
            // OWN template family, detected from the GGUF
            // `tokenizer.chat_template` (ChatML im_start / Llama-3
            // start_header_id); unknown templates pass the prompt raw with a
            // note instead of guessing.
            // `--score` may directly follow the model — the prompt is unused
            // when scoring: `sofuu infer <model> --score <file>`.
            let mut prompt = prompt.clone();
            let mut i = i; // shadowed below; keep parse-loop position mutable
            if args.len() > 4 && args[3] == "--score" {
                score_path = Some(args[4].clone());
                prompt = String::new();
                i = 5;
            }
            if chat {
                let tmpl = sofuu_core::infer::gguf::GgufFile::open(model_path)
                    .ok()
                    .and_then(|g| g.get_str("tokenizer.chat_template").map(|s| s.to_string()))
                    .unwrap_or_default();
                prompt = if tmpl.contains("im_start") {
                    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
                } else if tmpl.contains("start_header_id") {
                    format!(
                        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                } else {
                    eprintln!(
                        "[sofuu infer] --chat: no known chat template in this GGUF; passing the prompt raw"
                    );
                    prompt
                };
            }
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

            // --score <file> (PLAN-LONG-CONTEXT-PERSISTENCE §8 P3): teacher-
            // forced NLL over the file — the Gate 2 surface (PPL ratio f16 vs
            // KIVI) without touching the prefill kernels. Prints one JSON
            // line on stdout.
            if let Some(path) = &score_path {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("cannot read {}: {e}", path))
                    .unwrap_or_else(|e| {
                        eprintln!("\x1b[31mError:\x1b[0m {e}");
                        std::process::exit(1);
                    });
                let stats = engine
                    .score_text(
                        &text,
                        sofuu_core::infer::GenerateOptions {
                            kivi,
                            kivi_bits,
                            threads,
                            ..Default::default()
                        },
                        score_window,
                    )
                    .unwrap_or_else(|e| {
                        eprintln!("\x1b[31mScoring error:\x1b[0m {e}");
                        std::process::exit(1);
                    });
                // The ppl is only comparable to another tool when the
                // convention matches, and it is only a perplexity when enough
                // tokens were averaged: print both, and say so out loud when
                // the sample is small (a 1-token "ppl" once read as a 32x
                // quality gap against llama-perplexity's 1.42).
                let score_kv4 = engine.model.kv_bits();
                let scored_frac = if stats.tokens_total > 1 {
                    stats.scored as f64 / (stats.tokens_total - 1) as f64
                } else {
                    0.0
                };
                let sample_warn = stats.scored < sofuu_core::infer::MIN_SCORE_SAMPLE;
                if sample_warn {
                    eprintln!(
                        "\x1b[33m[sofuu infer]\x1b[0m --score: ppl averaged over only {} of {} tokens — a few-sample estimate, not a perplexity. Score a longer text; llama-perplexity's nearest setting is -c {}.",
                        stats.scored, stats.tokens_total, score_window
                    );
                }
                println!(
                    "{{\"kind\":\"score\",\"model\":{},\"kv\":\"{}\",\"window\":{},\"tokens_total\":{},\"scored\":{},\"windows\":{},\"nll_total\":{:.4},\"ppl\":{:.4},\"scored_frac\":{:.4},\"sample_warn\":{},\"convention\":\"ppl = exp(mean NLL) over the SCORED tokens; natural log, full-vocab log-softmax; window = max left context a scored token sees (llama-perplexity -c equivalent); the first tokens of the text are context-only, every later token is scored exactly once, and a window holds at most window+context positions\"}}",
                    serde_json::to_string(&model_path).unwrap_or_else(|_| "\"\"".into()),
                    if kivi {
                        format!("kivi{}", kivi_bits)
                    } else if score_kv4 > 0 {
                        // the receipt must name the cache it measured: the GPU
                        // decode path now defaults to the quantized KV cache.
                        format!("kv{}", score_kv4)
                    } else {
                        "f16".to_string()
                    },
                    stats.window,
                    stats.tokens_total,
                    stats.scored,
                    stats.windows,
                    stats.nll_total,
                    stats.ppl,
                    scored_frac,
                    sample_warn
                );
                return ExitCode::SUCCESS;
            }

            // SOFUU_PREFIX_TEST=1: multi-turn prefix-cache receipt. The stock
            // CLI is one-shot, so nothing else ever exercises the reuse path.
            // Turn 1 prefills the prompt; turn 2 re-sends it with the rendered
            // reply and a new user line (what a chat client does); turn 3
            // edits one character inside that reply. Prints the tokens each
            // call actually prefilled plus the engine's own prefix comparison.
            if std::env::var_os("SOFUU_PREFIX_TEST").is_some() {
                let opts = |max_tokens: usize| sofuu_core::infer::GenerateOptions {
                    max_tokens,
                    temperature: 0.0,
                    ..Default::default()
                };
                let mut reply = String::new();
                let _ = engine.stream(&prompt, opts(8), |t| {
                    reply.push_str(&t);
                    true
                });
                let p1 = engine.last_prefill_tokens;
                let cached = engine.cached_tokens.len();

                let turn2 = format!("{prompt}{reply}\n\nUser: Summarize that again.\nAssistant:");
                let t2 = engine.tokenizer.encode(&turn2);
                let mut c2 = 0usize;
                while c2 < engine.cached_tokens.len()
                    && c2 < t2.len()
                    && engine.cached_tokens[c2] == t2[c2]
                {
                    c2 += 1;
                }
                let _ = engine.stream(&turn2, opts(0), |_| true);
                let p2 = engine.last_prefill_tokens;

                let head: String = reply.chars().take(20).collect();
                let edited = reply.replacen(&head, &format!("{head}X"), 1);
                let turn3 = format!("{prompt}{edited}\n\nUser: Summarize that again.\nAssistant:");
                let t3 = engine.tokenizer.encode(&turn3);
                let mut c3 = 0usize;
                while c3 < engine.cached_tokens.len()
                    && c3 < t3.len()
                    && engine.cached_tokens[c3] == t3[c3]
                {
                    c3 += 1;
                }
                let _ = engine.stream(&turn3, opts(0), |_| true);
                let p3 = engine.last_prefill_tokens;

                eprintln!(
                    "[prefix] turn1 prefill={p1} cached={cached} | turn2 tokens={} common={c2} prefill={p2} | turn3 tokens={} common={c3} prefill={p3} | reply_chars={}",
                    t2.len(),
                    t3.len(),
                    reply.chars().count()
                );
                return ExitCode::SUCCESS;
            }

            let t_gen = std::time::Instant::now();
            let mut token_count = 0usize;
            use std::io::Write;
            let res = engine.stream(
                &prompt,
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
                    trace_tokens: trace_path.is_some(),
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
            if let Some(tp) = trace_path {
                match std::fs::File::create(&tp) {
                    Ok(mut f) => {
                        use std::io::Write as _;
                        for t in &engine.last_token_trace {
                            let piece = engine.tokenizer.decode_piece(t.token);
                            let piece_json: String = piece
                                .chars()
                                .map(|c| {
                                    if c == '"' || c == '\\' {
                                        format!("\\{}", c)
                                    } else if c.is_control() {
                                        format!("\\u{:04x}", c as u32)
                                    } else {
                                        c.to_string()
                                    }
                                })
                                .collect();
                            let _ = writeln!(
                                f,
                                "{{\"i\":{},\"pos\":{},\"tok\":{},\"piece\":\"{}\",\"forward_ns\":{},\"logprob\":{:.6}}}",
                                t.step, t.pos, t.token, piece_json, t.forward_ns, t.logprob
                            );
                        }
                        eprintln!(
                            "[sofuu] token trace: {} rows -> {}",
                            engine.last_token_trace.len(),
                            tp
                        );
                    }
                    Err(e) => {
                        eprintln!("[sofuu] --trace: cannot write {}: {}", tp, e);
                    }
                }
            }
            ExitCode::SUCCESS
        }
        "run" => {
            /* R7.P2/P3: --policy and the shortcut guards are honoured on the
             * plain script runner too (they used to exist only on chat/agent),
             * and flags are stripped BEFORE the script path is read so
             * `sofuu run --read-only x.js` works as well as the trailing
             * form. */
            let (policy, clean) = take_policy_arg(&args);
            let (guard, clean) = take_guard_flags(&clean);
            if clean.len() < 3 {
                eprintln!("\x1b[31mError:\x1b[0m 'sofuu run' requires a file argument");
                eprintln!(
                    "\x1b[90m  flags: --policy <file> --read-only --no-net --confirm-all\x1b[0m"
                );
                return ExitCode::FAILURE;
            }
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            if let Err(e) = install_command_policy(&rt, policy.as_deref(), guard) {
                eprintln!("\x1b[31mError:\x1b[0m {e}");
                return ExitCode::FAILURE;
            }
            let rc = cmd_run(&rt, &clean[2]);
            js_exit_code(&rt, rc)
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
        "chat" => run_chat(&args),
        /* PLAN §4 Phase 2 spells provider selection as
         * `sofuu --provider ollama --model llama3.2` — a bare CLI whose first
         * token is a flag is a chat invocation, not an unknown command.
         * --help/-h/--version still fall through to their own handlers. */
        flag if flag.starts_with('-')
            && flag != "-h"
            && flag != "--help"
            && flag != "-V"
            && flag != "--version" =>
        {
            let mut chat_args: Vec<String> = args.clone();
            chat_args.insert(1, "chat".to_string());
            run_chat(&chat_args)
        }
        "conformance" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = conformance::cmd_conformance(&rt, VERSION, &args);
            if rc == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
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
            let (policy, clean) = take_policy_arg(&args);
            /* §4: --on-error belongs to `agent run`. Strip and validate it
             * here, i.e. before the runtime starts, so a typo costs nothing
             * and never reaches the driver as an unknown mode. */
            let (on_error, clean) = take_on_error_arg(&clean);
            let on_error = match on_error.as_deref() {
                Some(mode) => match parse_on_error_mode(mode) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        eprintln!("\x1b[31mError:\x1b[0m {e}");
                        return ExitCode::FAILURE;
                    }
                },
                None => None,
            };
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            if let Some(p) = policy {
                if let Err(e) = apply_policy_file(&rt, &p) {
                    eprintln!("\x1b[31mError:\x1b[0m {e}");
                    return ExitCode::FAILURE;
                }
            }
            let rc = cmd_agent(&rt, &clean, on_error);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        /* OBSERVABILITY-SPEC §4: the observation surface. Rendering (and the
         * exit code) come from the shipped driver; the runtime is still
         * initialised because sofuu.observability lives on the engine. */
        "logs" => {
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            let rc = cmd_logs(&rt, &args);
            js_exit_code(&rt, rc)
        }
        /* §4: resume takes --policy like the agent commands next to it,
         * because it re-enters the same loop under the same policy. */
        "resume" => {
            let (policy, clean) = take_policy_arg(&args);
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            if let Some(p) = policy {
                if let Err(e) = apply_policy_file(&rt, &p) {
                    eprintln!("\x1b[31mError:\x1b[0m {e}");
                    return ExitCode::FAILURE;
                }
            }
            let rc = cmd_resume(&rt, &clean);
            js_exit_code(&rt, rc)
        }
        "plugin" | "plugins" => {
            let (policy, clean) = take_policy_arg(&args);
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            if let Some(p) = policy {
                if let Err(e) = apply_policy_file(&rt, &p) {
                    eprintln!("\x1b[31mError:\x1b[0m {e}");
                    return ExitCode::FAILURE;
                }
            }
            let rc = cmd_plugin(&rt, &clean);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "serve" => {
            let (policy, clean) = take_policy_arg(&args);
            let rt = match SofuuRuntime::init() {
                Some(rt) => rt,
                None => {
                    eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                    return ExitCode::FAILURE;
                }
            };
            if let Some(p) = policy {
                if let Err(e) = apply_policy_file(&rt, &p) {
                    eprintln!("\x1b[31mError:\x1b[0m {e}");
                    return ExitCode::FAILURE;
                }
            }
            let rc = cmd_serve(&rt, &clean);
            if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        "swarm" => cmd_swarm(&args),
        "brain" => {
            /* W7: `sofuu brain …`. Only `migrate` needs the JS engine (the
             * runtime's own open→flush conversion); verify/repair/stats/
             * backup/restore are pure storage tooling, so the runtime is
             * built on first use and a `--help` or a usage error never pays
             * for it. */
            let mut engine: Option<SofuuRuntime> = None;
            let mut eval = |script: &str| -> Result<String, String> {
                if engine.is_none() {
                    engine = SofuuRuntime::init();
                }
                let rt = engine
                    .as_ref()
                    .ok_or_else(|| "runtime init failed".to_string())?;
                let ctx = rt.engine_ctx() as *mut sofuu_ffi::qjs::JSContext;
                if ctx.is_null() {
                    return Err("the runtime has no engine context".to_string());
                }
                let rc = rt.eval_string(script, "<brain-migrate>");
                unsafe {
                    let global = sofuu_ffi::qjs::sofuu_js_get_global_object(ctx);
                    let v = sofuu_ffi::qjs::sofuu_js_get_property_str(
                        ctx,
                        global,
                        c"__brainReport".as_ptr(),
                    );
                    let cstr = sofuu_ffi::qjs::sofuu_js_to_cstring(ctx, v);
                    let out = if cstr.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(cstr).to_string_lossy().into_owned()
                    };
                    if !cstr.is_null() {
                        sofuu_ffi::qjs::sofuu_js_free_cstring(ctx, cstr);
                    }
                    sofuu_ffi::qjs::sofuu_js_free_value(ctx, v);
                    sofuu_ffi::qjs::sofuu_js_free_value(ctx, global);
                    if rc != 0 {
                        return Err("the migration script threw".to_string());
                    }
                    if out.is_empty() {
                        return Err("the migration script left no report".to_string());
                    }
                    Ok(out)
                }
            };
            ExitCode::from(sofuu_core::brain_cmd::run(&args, &mut eval))
        }
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

            // PC-05: a swarm node without a GPU backend gets the same single
            // warning `sofuu infer` gives — the flag used to be inert here too.
            gpu_layers = resolve_gpu_layers(gpu_layers);
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
                    "--temp" | "--temperature" => {
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
                        } else if other.starts_with('-') && other != "-" {
                            // A silently-dropped flag is a correctness hazard:
                            // e.g. "--temperature 0" would leave the sampler at
                            // the 0.7 default with a wall-clock RNG seed.
                            eprintln!("[sofuu infer] warning: unrecognized flag `{}` ignored", other);
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

            // PC-05: same one-shot warning as `sofuu infer` / `swarm worker`.
            gpu_layers = resolve_gpu_layers(gpu_layers);
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

#[cfg(test)]
mod artifact_run_hint_tests {
    use super::artifact_run_hint;

    #[test]
    fn paths_with_a_directory_print_as_themselves() {
        assert_eq!(artifact_run_hint("/tmp/agent"), "/tmp/agent");
        assert_eq!(artifact_run_hint("dist/agent"), "dist/agent");
        assert_eq!(artifact_run_hint("./agent"), "./agent");
    }

    #[test]
    fn a_bare_file_name_gets_the_dot_slash() {
        assert_eq!(artifact_run_hint("agent"), "./agent");
    }
}
