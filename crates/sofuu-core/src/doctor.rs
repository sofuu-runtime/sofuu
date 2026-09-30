// `sofuu doctor` — one command that answers "is this install healthy?".
//
// The two things users cannot otherwise verify are (a) whether the QTSQ
// brain was linked into THIS binary and really persists, and (b) what the
// context-window ladder currently believes about the active model. Both
// used to fail silently: no QTSQ means memory calls quietly no-op, and an
// unknown model means a quiet 32,768 window. This command probes both for
// real and prints what it finds, with a non-zero exit when a hard
// dependency (QTSQ) is missing.

use std::path::{Path, PathBuf};

use crate::chat::ChatConfig;
use sofuu_core::ml::alloc::policy;

fn ok(msg: &str) {
    println!("\x1b[32m  ✓\x1b[0m {msg}");
}

fn warn(msg: &str) {
    println!("\x1b[33m  ⚠\x1b[0m {msg}");
}

fn bad(msg: &str) {
    println!("\x1b[31m  ✗\x1b[0m {msg}");
}

fn info(msg: &str) {
    println!("\x1b[90m  ·\x1b[0m {msg}");
}

fn section(title: &str) {
    println!("\n\x1b[1m{title}\x1b[0m");
}

/// Byte size of a file, or None when it does not exist.
fn file_len(p: &Path) -> Option<u64> {
    std::fs::metadata(p).ok().map(|m| m.len())
}

fn human(n: u64) -> String {
    if n >= 1 << 20 {
        format!("{:.1} MB", n as f64 / (1u64 << 20) as f64)
    } else if n >= 1 << 10 {
        format!("{:.1} KB", n as f64 / (1u64 << 10) as f64)
    } else {
        format!("{n} B")
    }
}

/// Pull the `error` field out of the probe's JSON without a parser.
fn js_error(probe_out: &str) -> Option<String> {
    let key = "\"error\":\"";
    let at = probe_out.find(key)? + key.len();
    let rest = &probe_out[at..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// The brain path the runtime actually uses: config override, else the
/// project-local default (<cwd>/.sofuu/brain/brain.qtsq).
fn brain_path(cfg: &ChatConfig) -> PathBuf {
    if !cfg.brain_path.is_empty() {
        return PathBuf::from(&cfg.brain_path);
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cwd.join(".sofuu").join("brain").join("brain.qtsq")
}

/// Run the health check. Returns the process exit code.
pub fn run(rt: &sofuu_ffi::SofuuRuntime, args: &[String]) -> i32 {
    let json = args.iter().any(|a| a == "--json");
    let cfg = ChatConfig::load();
    let mut failures = 0usize;

    println!("\x1b[1msofuu doctor\x1b[0m — v{} · {}", crate::VERSION, std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| "?".into()));

    // ── 1. QTSQ + brain (the silent-failure pair) ──────────────────
    section("QTSQ / brain");
    /* The probe path is INLINED, not passed through process.env: the
     * process.env object is a snapshot taken at engine init, so a var set
     * afterwards is invisible to JS (it would open the literal path
     * "undefined" and quietly write a brain file with that name). */
    let tmp_brain = std::env::temp_dir().join("sofuu-doctor-probe.qtsq");
    let _ = std::fs::remove_file(&tmp_brain);
    let tmp_brain_js = tmp_brain.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
    let probe = format!(
        r#"(function () {{
  var PATH = "{path}";
  var out = {{
    memory: typeof sofuu.memory,
    kv: typeof sofuu.kv,
    sessionSave: typeof __qtsq_session_save,
    agentCreate: typeof (sofuu.agent && sofuu.agent.create),
  }};
  try {{
    /* Phase 1: write into a fresh temp brain and flush. */
    var v = new Float32Array(64);
    for (var i = 0; i < 64; i++) v[i] = 0.125; /* unit: 64 * (1/8)^2 = 1 */
    var m = sofuu.memory.open(PATH, 64, 'doctor-probe');
    var idx = m.remember(v, 'sofuu-doctor-canary', 'doctor', 0);
    m.flush();
    out.remembered = idx >= 0;
    /* Phase 2: a SECOND open re-hydrates from the FILE (cma_open builds a
     * fresh index every time — nothing is cached), so a hit here can only
     * come from what actually reached disk. */
    var m2 = sofuu.memory.open(PATH, 64, 'doctor-probe');
    var hit = m2.recall(v, 3) || [];
    out.roundTrip = hit.length > 0;
    out.roundTripHits = hit.length;
  }} catch (e) {{
    out.error = String((e && e.message) || e);
  }}
  return JSON.stringify(out);
}})()"#,
        path = tmp_brain_js
    );

    let probe_out = rt.eval_repl(&probe).unwrap_or_default();
    rt.run_jobs();

    let qtsq_linked = sofuu_ffi::qtsq_linked();
    let round_trip = probe_out.contains("\"roundTrip\":true");
    let mem_type = if probe_out.contains("\"memory\":\"object\"") { "object" } else { "missing" };

    if qtsq_linked {
        ok("QTSQ linked into this binary (brain persistence available)");
    } else {
        bad("QTSQ NOT linked — brain/memory/sessions silently no-op in this build");
        info("rebuild with SOFUU_QTSQ_DIR=<qtsq checkout> make (see cli/README.md)");
        failures += 1;
    }
    info(&format!("sofuu.memory={mem_type} · sofuu.kv present={}", probe_out.contains("\"kv\":\"object\"")));

    if round_trip {
        let size = file_len(&tmp_brain).unwrap_or(0);
        ok(&format!("brain write→read round-trip passed (temp store {human_bytes})", human_bytes = human(size)));
    } else if qtsq_linked {
        warn("brain round-trip FAILED — QTSQ is linked but a write did not read back");
        if let Some(msg) = js_error(&probe_out) {
            info(&format!("probe error: {msg}"));
        }
        failures += 1;
    } else {
        warn("brain round-trip skipped (QTSQ not linked)");
    }
    let _ = std::fs::remove_file(&tmp_brain);

    // ── 2. The real brain file ──────────────────────────────────────
    let bp = brain_path(&cfg);
    match file_len(&bp) {
        Some(n) => ok(&format!("brain file present: {} ({})", bp.display(), human(n))),
        None => info(&format!("brain file not created yet: {} (first /remember writes it)", bp.display())),
    }
    if !cfg.brain_path.is_empty() {
        info(&format!("brain_path override from config: {}", cfg.brain_path));
    } else {
        info("no brain_path override — using the project-local default");
    }

    // ── 3. Context window resolution (F1/F2 surface) ────────────────
    section("context window");
    if cfg.model.is_empty() {
        warn("no model configured — set one with /model (or --model) to check its window");
    } else {
        let model = cfg.model.clone();
        let base = if cfg.base_url.is_empty() { None } else { Some(cfg.base_url.clone()) };
        let r = policy::resolve_explicit(
            Some(model.as_str()),
            cfg.ctx_window,
            cfg.max_output,
            base.as_deref(),
            cfg.ctx_window_explicit,
        );
        info(&format!(
            "model {} @ {}",
            model,
            base.clone().unwrap_or_else(|| "(no endpoint)".into())
        ));
        info(&format!(
            "resolved window {} (source: {}) · max output {} ({})",
            r.window,
            r.win_source.as_str(),
            r.max_output,
            r.max_source.as_str()
        ));
        if let Some(bound) = r.config_exceeds_evidence {
            if r.window > bound {
                warn(&format!(
                    "your /ctx value exceeds the strongest known evidence ({bound} tokens, {}) — honored; a provider 400 will teach the real limit",
                    r.win_source.as_str()
                ));
            } else {
                info(&format!("inherited ctx value clamped to the model's real bound: {bound} tokens"));
            }
        }
        if !r.known {
            warn(&format!(
                "nothing is known about this model — using the conservative {} default; /model or /provider now detects it from the endpoint listing",
                r.window
            ));
        }
        // Endpoint-discovered store: how much truth we hold.
        sofuu_core::rt::model_caps_discovered::ensure_loaded();
        let n = sofuu_core::rt::model_caps_discovered::len();
        info(&format!(
            "discovered model caps: {n} entries cached{}",
            if n == 0 { " (run /model in chat to harvest the endpoint listing)" } else { "" }
        ));
    }

    // ── 4. Config sanity ────────────────────────────────────────────
    section("config");
    let cfg_path = ChatConfig::config_path();
    match file_len(&cfg_path) {
        Some(n) => ok(&format!("{} ({})", cfg_path.display(), human(n))),
        None => warn(&format!("{} not found — provider keys must be set before chatting", cfg_path.display())),
    }
    if cfg.api_key.is_empty() {
        warn("no API key set for the active provider (chat still runs, providers will 401)");
    } else {
        let masked = if cfg.api_key.len() > 8 {
            format!("{}…{}", &cfg.api_key[..4], &cfg.api_key[cfg.api_key.len() - 2..])
        } else {
            "set".to_string()
        };
        ok(&format!("API key present for {} ({masked})", cfg.active));
    }

    // ── 5. Runtime itself ───────────────────────────────────────────
    section("runtime");
    ok(&format!("engine initialized (QJS + libuv + {} embed spaces)", 4));

    // ── verdict ─────────────────────────────────────────────────────
    section("verdict");
    if failures == 0 {
        println!("\x1b[32m  healthy\x1b[0m — brain persistence and caps resolution both operational");
    } else {
        println!("\x1b[31m  {failures} problem(s) found\x1b[0m — see the ⚠/✗ lines above");
    }
    let _ = json;
    if failures == 0 { 0 } else { 1 }
}
