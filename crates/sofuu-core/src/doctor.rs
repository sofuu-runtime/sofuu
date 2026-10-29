// `sofuu doctor` — one command that answers "is this install healthy?".
//
// The two things users cannot otherwise verify are (a) whether the QTSQ
// brain was linked into THIS binary and really persists, and (b) what the
// context-window ladder currently believes about the active model. Both
// used to fail silently: no QTSQ means memory calls quietly no-op, and an
// unknown model means a quiet 32,768 window. This command probes both for
// real and prints what it finds, with a non-zero exit when a hard
// dependency (QTSQ) is missing.
//
// PC-06 (audit 2026-10-29, Pi 5): every probe here is READ-ONLY. The brain
// key and the brain store are created by the first real write, never by a
// diagnostic — `HOME=$(mktemp -d) sofuu doctor` used to leave a 65 B
// brain.key in that home and then blame the project's existing brain for
// not opening with the key it had just invented.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};

use crate::chat::ChatConfig;
use sofuu_core::ml::alloc::policy;

/* D2 (workstream protocols): `--json` emits the WHOLE report as structured
 * entries instead of dropping the flag. The text helpers are the single
 * renderer, so the JSON and the human output can never diverge. */
thread_local! {
    static REPORT: RefCell<Vec<serde_json::Value>> = const { RefCell::new(Vec::new()) };
    static CURRENT_SECTION: RefCell<String> = const { RefCell::new(String::new()) };
    static JSON_MODE: Cell<bool> = const { Cell::new(false) };
}

fn emit(level: &str, msg: &str) {
    if !JSON_MODE.with(|j| j.get()) {
        let mark = match level {
            "ok" => "\x1b[32m  ✓\x1b[0m",
            "warn" => "\x1b[33m  ⚠\x1b[0m",
            "bad" => "\x1b[31m  ✗\x1b[0m",
            _ => "\x1b[90m  ·\x1b[0m",
        };
        println!("{mark} {msg}");
    }
    let section = CURRENT_SECTION.with(|s| s.borrow().clone());
    REPORT.with(|r| {
        r.borrow_mut().push(serde_json::json!({
            "section": section,
            "level": level,
            "message": msg,
        }))
    });
}

fn ok(msg: &str) {
    emit("ok", msg);
}

fn warn(msg: &str) {
    emit("warn", msg);
}

fn bad(msg: &str) {
    emit("bad", msg);
}

fn info(msg: &str) {
    emit("info", msg);
}

fn section(title: &str) {
    if !JSON_MODE.with(|j| j.get()) {
        println!("\n\x1b[1m{title}\x1b[0m");
    }
    CURRENT_SECTION.with(|s| *s.borrow_mut() = title.to_string());
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
    JSON_MODE.with(|j| j.set(json));
    let mut cfg = ChatConfig::load();
    let mut model_override = None;
    for i in 0..args.len() {
        if (args[i] == "-m" || args[i] == "--model") && i + 1 < args.len() {
            model_override = Some(args[i + 1].clone());
            break;
        }
    }
    if let Some(m) = model_override {
        let is_gguf = m.ends_with(".gguf")
            || sofuu_core::infer::clean_model_path(&m).extension().map_or(false, |e| e == "gguf");
        if is_gguf {
            cfg.provider = "local".into();
            cfg.base_url.clear();
            cfg.api_key.clear();
            let caps = sofuu_core::rt::model_caps::lookup(Some(&m));
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
        cfg.model = m;
    }
    let mut failures = 0usize;

    let header = format!(
        "sofuu doctor — v{} · {}",
        crate::VERSION,
        std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| "?".into())
    );
    if json {
        /* The JSON document must be the only thing on stdout. */
        REPORT.with(|r| {
            r.borrow_mut().push(serde_json::json!({
                "section": "",
                "level": "header",
                "message": header,
            }))
        });
    } else {
        println!("\x1b[1msofuu doctor\x1b[0m — v{} · {}", crate::VERSION, std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| "?".into()));
    }

    // ── 1. QTSQ + brain (the silent-failure pair) ──────────────────
    section("QTSQ / brain");
    /* PC-06 (audit 2026-10-29, Pi 5): a diagnostic must not tidy up the
     * install it is describing. Every brain open below resolves the vault
     * password, and the runtime's resolver CREATES `~/.sofuu/` plus a 0600
     * brain.key on first use — so `HOME=$(mktemp -d) sofuu doctor` left a
     * 65 B key behind and then reported the project's pre-existing brain
     * as "cannot be opened with any known key": the fresh key it had just
     * written was not the one that wrote the file. Observe the key without
     * creating it, and either pin that exact password (a real HOME: every
     * read below behaves as the runtime would) or a throwaway per-process
     * key that never reaches disk (no key yet: throwaway stores still
     * round-trip, an existing store refuses). */
    let key_state = sofuu_core::rt::brain_key::key_state();
    sofuu_core::rt::brain_key::pin_vault_password(key_state.password());
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

    /* ── 2b. The brain store itself (B7.2, ROADMAP-BRAIN-CMA): manifest,
     * space, tier histogram, capacity headroom, journal state, last
     * compaction — the same numbers /brain info and cma.stats() report,
     * read from the real file. (brain_store is QTSQ-gated; so is this.) */
    #[cfg(has_qtsq)]
    if qtsq_linked && bp.exists() {
        section("brain store");
        use sofuu_core::rt::brain_key::KeyState;
        /* PC-06 (audit 2026-10-29): every line below is rendered from the
         * key state OBSERVED in section 1 — the same resolution order the
         * runtime uses, minus the first-use creation. A present key means
         * the store can be opened exactly as the runtime would; no key
         * means it cannot be opened by anything this process may write,
         * so the store is described but never asked to open. */
        let key_usable = key_state.present();
        // P2 (audit 2026-10-14): WHICH key would the next flush use — the
        // env override, the key file, or a process-local one that never
        // reopens? Observation only: this never creates the key file.
        match &key_state {
            KeyState::Env(v) => {
                let hex64 = v.len() == 64 && v.chars().all(|c| c.is_ascii_hexdigit());
                if hex64 {
                    info("SOFUU_BRAIN_KEY override in use (64 hex chars)");
                } else {
                    warn(&format!(
                        "SOFUU_BRAIN_KEY used as given ({} chars, not the documented 64-hex shape) — keep it byte-exact; brains written under it reopen only with the same value",
                        v.len()
                    ));
                }
            }
            KeyState::File { path, .. } => {
                let mode = std::fs::metadata(path).ok().map(|md| {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        md.permissions().mode() & 0o777
                    }
                    #[cfg(not(unix))]
                    {
                        let _ = md;
                        0o600
                    }
                });
                match mode {
                    Some(0o600) => ok(&format!("machine key present: {path} (0600)")),
                    Some(m) => warn(&format!(
                        "machine key present but mode {m:o} (expected 600): {path}"
                    )),
                    /* Vanished between the observation and this line: say so
                     * rather than claim a mode. */
                    None => info(&format!("machine key reported present, now unreadable: {path}")),
                }
            }
            KeyState::NoKeyYet { path } => info(&format!(
                "no brain key yet ({} missing) — it will be created on first use; this check creates nothing",
                path.clone()
                    .unwrap_or_else(|| "~/.sofuu/brain.key".into())
            )),
            KeyState::NoHome => info(
                "HOME is unset — the runtime uses a per-process key; brains written now will not reopen (this check creates nothing)",
            ),
        }
        if !key_usable {
            /* PC-06: without a key the store cannot be opened here, and
             * creating one to find out would both write to the user's home
             * and still fail — the file needs the key of the HOME that
             * wrote it. The brain is optional, so this is an observation,
             * not a problem: no probe runs, nothing is counted. */
            info(
                "brain store not opened by this check: a store written under another HOME's key needs that key (restore ~/.sofuu/brain.key or set SOFUU_BRAIN_KEY) — the first brain write here creates a new one, which will not open the existing file",
            );
        }
        // Journal state first (the same recovery an open runs): a crashed
        // commit that is rolled forward here is a fix, not a problem —
        // but the user should see it happened.
        if let Some(dir) = bp.parent() {
            let report = sofuu_core::rt::wal::recover(&dir.to_string_lossy());
            if report.journal_found {
                ok("write-ahead journal found — a crashed commit was rolled forward (recovered)");
            } else {
                info("journal state clean (no interrupted commits)");
            }
        }
        // Manifest: space + segments + compaction generation. A legacy
        // (pre-B3) brain has no manifest — still healthy, just unsegmented.
        // `_ex` distinguishes "not segmented" from "a container no known
        // key opens" — a wrong-key brain used to read as a healthy ✓.
        // PC-06: with no key this read cannot succeed, so it is not made —
        // an -4 from a key this process would have had to create is not a
        // fact about the install.
        let manifest_ex = if key_usable {
            unsafe { sofuu_core::rt::brain_store::read_manifest_ex(&bp.to_string_lossy()) }
        } else {
            Ok(None)
        };
        match &manifest_ex {
            Ok(Some((_, true))) => info(
                "legacy-password encryption (pre-B4) — re-encrypts under the machine key on its next write (backup kept)",
            ),
            Ok(Some((_, false))) => info("encrypted under this machine's key"),
            _ => {}
        }
        let manifest = match &manifest_ex {
            Ok(Some((m, _))) => Some(m.clone()),
            Ok(None) => None,
            Err(rc) => {
                warn(&format!(
                    "brain file cannot be opened with any known key (qtsq error {rc}) — restore ~/.sofuu/brain.key or set SOFUU_BRAIN_KEY to its original value"
                ));
                failures += 1;
                None
            }
        };
        match &manifest {
            Some(m) => {
                let space = m
                    .embedding
                    .get("id")
                    .and_then(|v: &serde_json::Value| v.as_str())
                    .unwrap_or("unknown");
                let dim = m.embedding.get("dimension").and_then(|v: &serde_json::Value| v.as_i64()).unwrap_or(0);
                let bytes: u64 = m
                    .segments
                    .iter()
                    .filter_map(|s| {
                        std::fs::metadata(format!(
                            "{}/{}",
                            sofuu_core::rt::brain_store::segments_dir(&bp.to_string_lossy()),
                            s.file
                        ))
                        .ok()
                        .map(|md| md.len())
                    })
                    .sum();
                ok(&format!(
                    "segmented store: {} segment(s), {} record version(s), epoch {} ({})",
                    m.segments.len(),
                    m.total_versions(),
                    m.epoch,
                    human(bytes)
                ));
                info(&format!("vector space: {space} · {dim} dims"));
                if m.last_decay_at > 0 {
                    let age = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs().saturating_sub(m.last_decay_at))
                        .unwrap_or(0);
                    info(&format!(
                        "last decay pass: {}s ago (strengths current as of then)",
                        age
                    ));
                }
                if m.evicted_total > 0 {
                    info(&format!(
                        "lifetime capacity evictions: {} (reported, never silent)",
                        m.evicted_total
                    ));
                }
            }
            /* PC-06: guarded — with no machine key this open is a
             * guaranteed -4, and the "not opened" line above already said
             * why; probing anyway would only add a scary warning plus a
             * stderr refusal for a state that is normal on a new machine. */
            None if key_usable => {
                // Not segmented: a legacy single-file brain (or a plain
                // fresh file). Probe its encryption directly — B4 migration
                // state is invisible to the manifest reader. Observation
                // only: the container is closed without any write.
                unsafe {
                    match sofuu_core::rt::brain_store::container_open_ex(&bp.to_string_lossy()) {
                        Ok((ctx, legacy)) => {
                            sofuu_core::rt::brain_store::container_free(ctx);
                            if legacy {
                                info("legacy-password encrypted single-file brain (pre-B4) — the next flush re-encrypts under the machine key (backups kept)");
                            } else {
                                info("single-file brain, encrypted under this machine's key");
                            }
                        }
                        Err(rc) => {
                            warn(&format!(
                                "brain file cannot be opened with any known key (qtsq error {rc}) — restore ~/.sofuu/brain.key or set SOFUU_BRAIN_KEY to its original value"
                            ));
                            failures += 1;
                        }
                    }
                }
            }
            None => {}
        }
        // Live stats + tier histogram through the REAL store (the same
        // surface /brain info reads). Open runs the recovery + decay
        // replay; nothing here flushes, so doctor never writes.
        let bp_js = bp.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
        /* P1 (audit 2026-10-14): open the real brain with the shape the
         * manifest declares — the probe used to call sofuu.memory.open(path)
         * with ONE argument, which the native binding rejects (it needs a
         * vector dimension), so this section failed on every healthy
         * install. Candidates are tried in order: the manifest's space,
         * then the two built-in shapes for legacy/unmanifested brains. */
        let mut cands: Vec<serde_json::Value> = Vec::new();
        if let Some(m) = &manifest {
            cands.push(serde_json::json!({
                "dim": m.embedding.get("dimension").and_then(|v| v.as_u64()).unwrap_or(0),
                "space": m.embedding.get("id").and_then(|v| v.as_str()).unwrap_or(""),
            }));
        }
        cands.push(serde_json::json!({"dim": 768, "space": ""}));
        cands.push(serde_json::json!({"dim": 64, "space": ""}));
        let cands_json = serde_json::to_string(&cands).unwrap_or_else(|_| "[]".into());
        let probe = format!(
            r#"(function () {{
  var out = {{}};
  try {{
    var cands = {cands};
    var b = null, used = null, lastErr = "";
    for (var ci = 0; ci < cands.length && !b; ci++) {{
      try {{
        b = cands[ci].space
          ? sofuu.memory.open("{p}", cands[ci].dim, cands[ci].space)
          : sofuu.memory.open("{p}", cands[ci].dim);
        used = cands[ci];
      }} catch (eC) {{ lastErr = String((eC && eC.message) || eC); }}
    }}
    if (!b) throw new Error(lastErr || "sofuu.memory.open refused every known shape");
    out.shape = (used && used.space ? used.space : "inferred") + "@" + (used && used.dim);
    var s = JSON.parse(b.stats());
    out.records = s.records;
    out.working = s.working;
    out.epoch = s.epoch;
    out.needs_compaction = s.needs_compaction;
    out.dirty = s.dirty;
    out.ef_search = s.ef_search;
    out.diversity_floor = s.diversity_floor;
    out.tiers = [];
    for (var t = 0; t <= 4; t++) {{
      var r = JSON.parse(b.list({{ tier: t, limit: 1 }}));
      out.tiers.push(r.total | 0);
    }}
  }} catch (e) {{
    out.error = String((e && e.message) || e);
  }}
  return JSON.stringify(out);
}})()"#,
            cands = cands_json,
            p = bp_js
        );
        /* PC-06: with no machine key there is nothing to open and nothing
         * to report — the "brain store not opened" line above already said
         * why, so the probe does not run and no `Failed to initialize CMA`
         * can be blamed on a HOME that simply has not written a brain yet. */
        let out = if key_usable {
            let o = rt.eval_repl(&probe).unwrap_or_default();
            rt.run_jobs();
            Some(o)
        } else {
            None
        };
        let parse = |key: &str| -> Option<serde_json::Value> {
            let out = out.as_deref()?;
            let marker = format!("\"{key}\":");
            let at = out.find(&marker)? + marker.len();
            let rest = out[at..].trim_start();
            /* An ARRAY value must read to its closing bracket — stopping at
             * the first ',' inside "[0,16,0,0,0]" left a fragment that
             * never parsed (the histogram rendered empty). */
            let end = if rest.starts_with('[') {
                rest.find(']')? + 1
            } else {
                rest.find(|c: char| c == ',' || c == '}')
                    .unwrap_or(rest.len())
            };
            serde_json::from_str(rest[..end].trim_end_matches(',')).ok()
        };
        if let Some(err) = out.as_deref().and_then(js_error) {
            warn(&format!("brain store probe failed: {err}"));
            failures += 1;
        } else if out.is_some() {
            let records = parse("records").and_then(|v| v.as_i64()).unwrap_or(0);
            let working = parse("working").and_then(|v| v.as_i64()).unwrap_or(0);
            let dirty = parse("dirty").and_then(|v| v.as_i64()).unwrap_or(0);
            let needs_compaction = parse("needs_compaction")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let tiers: Vec<i64> = parse("tiers")
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default()
                .iter()
                .filter_map(|v| v.as_i64())
                .collect();
            let names = ["working", "episodic", "semantic", "forgotten", "entity"];
            let hist: Vec<String> = names
                .iter()
                .zip(tiers.iter())
                .map(|(n, c)| format!("{n} {c}"))
                .collect();
            if let Some(shape) = parse("shape").and_then(|v| v.as_str().map(str::to_string)) {
                info(&format!("opened as {shape}"));
            }
            ok(&format!("records: {records} · tiers: {}", hist.join(" · ")));
            if working > 0 {
                info(&format!("session working records: {working} (in-RAM only — gone on reload unless promoted)"));
            }
            // Capacity headroom: the same policy the store enforces.
            let cap = std::env::var("SOFUU_BRAIN_MAX_RECORDS")
                .ok()
                .and_then(|v| v.trim().parse::<i64>().ok())
                .unwrap_or(100_000);
            if cap == 0 {
                info("capacity cap disabled (SOFUU_BRAIN_MAX_RECORDS=0)");
            } else {
                let headroom = (cap - records).max(0);
                let pct = if cap > 0 {
                    (records as f64 / cap as f64 * 100.0).min(100.0)
                } else {
                    0.0
                };
                info(&format!(
                    "capacity: {records}/{cap} records ({pct:.1}% full · {headroom} headroom)"
                ));
                if pct >= 90.0 {
                    warn("brain is near its capacity cap — weakest records will be evicted (never pins or entities)");
                }
            }
            if needs_compaction {
                info(&format!("compaction pending ({dirty} dirty record(s)) — the next turn-boundary flush rewrites the store"));
            } else {
                info("no compaction pending");
            }
            info(&format!(
                "recall knobs: ef_search {} (0 = default) · diversity floor {}",
                parse("ef_search").and_then(|v| v.as_i64()).unwrap_or(0),
                parse("diversity_floor")
                    .and_then(|v| v.as_f64())
                    .map(|f| format!("{f:.2}"))
                    .unwrap_or_else(|| "off".into())
            ));
        }
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

    // ── 6. Protocols, sandbox, policy (D2) ─────────────────────────
    section("protocols / sandbox");
    let jail_ok = protocol_probes(rt, &mut failures);

    // ── 7. Capability manifest (R1, T1.4) ──────────────────────────
    // Doctor asserts the build's own claims: a declared feature that fails
    // its probe, or a build compiled without a core feature, is a failure —
    // "healthy" must never describe a degraded build.
    section("capabilities");
    // NOTE: doctor.rs is compiled into both the lib and the bin crate, so the
    // manifest must be reached through the lib crate name, not `crate::`.
    let caps = sofuu_core::capabilities::manifest();
    let missing: Vec<String> = caps
        .pointer("/build/missing")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    // A feature the target platform cannot provide (Metal on Linux/Windows) is
    // not a degraded build. Counting it as a failure made `doctor` exit 1 on
    // every healthy Linux install — and because `scripts/verify-dist.sh` gates
    // on `doctor` exiting 0, the repo's own RULES §3.3 dist check could never
    // pass on the Linux artefact.
    let platform_absent: Vec<String> = caps
        .pointer("/build/platform_absent")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let real_missing: Vec<&String> = missing
        .iter()
        .filter(|m| !platform_absent.iter().any(|p| p == *m))
        .collect();
    for m in &platform_absent {
        info(&format!(
            "capability absent on this platform: {m} — not provided by the target OS, so its APIs are unavailable by design"
        ));
    }
    if real_missing.is_empty() {
        ok("all declared build features present");
    } else {
        for m in real_missing {
            bad(&format!(
                "capability missing: {m} — this binary was built without it and refuses the matching APIs loudly (rebuild with the feature enabled; see the Makefile guard)"
            ));
        }
        failures += 1;
    }
    let metal_declared = caps
        .pointer("/build/features")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().any(|x| x.as_str() == Some("metal")))
        .unwrap_or(false);
    let gpu_available = caps
        .pointer("/runtime/gpu/available")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if metal_declared && !gpu_available {
        bad("metal is compiled in but no Metal device is available at runtime");
        failures += 1;
    }
    info(&format!(
        "determinism paths: {}",
        caps.pointer("/engine/paths")
            .map(|p| p.to_string())
            .unwrap_or_else(|| "{}".into())
    ));
    info(&format!(
        "runtime: {} {} · {} threads · gpu available: {}",
        caps.pointer("/runtime/os").and_then(|v| v.as_str()).unwrap_or("?"),
        caps.pointer("/runtime/arch").and_then(|v| v.as_str()).unwrap_or("?"),
        caps.pointer("/runtime/threads").and_then(|v| v.as_u64()).unwrap_or(0),
        gpu_available
    ));

    // ── verdict ─────────────────────────────────────────────────────
    section("verdict");
    if failures == 0 {
        ok("healthy — brain persistence, caps resolution and the protocol floors are operational");
    } else {
        bad(&format!("{failures} problem(s) found — see the ⚠/✗ lines above"));
    }
    if json {
        let report = serde_json::json!({
            "version": crate::VERSION,
            "cwd": std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| "?".into()),
            "ok": failures == 0,
            "failures": failures,
            "jail": jail_ok,
            "capabilities": caps,
            "entries": REPORT.with(|r| r.borrow().clone()),
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into()));
    }
    if failures == 0 { 0 } else { 1 }
}

/// D2: the probes the protocol tests use, run in doctor so "doctor green"
/// and "tests green" cannot diverge. Returns whether the reader jail held.
fn protocol_probes(rt: &sofuu_ffi::SofuuRuntime, failures: &mut usize) -> bool {
    let dir = std::env::temp_dir().join(format!("sofuu-doctor-jail-{}", std::process::id()));
    let outside = std::env::temp_dir().join(format!("sofuu-doctor-outside-{}.txt", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join("inside.txt"), "INSIDE-OK\n");
    let _ = std::fs::write(&outside, "OUTSIDE-SECRET\n");
    #[cfg(unix)]
    let link = std::os::unix::fs::symlink(&outside, dir.join("escape"));
    #[cfg(not(unix))]
    let link: std::io::Result<()> =
        Err(std::io::Error::new(std::io::ErrorKind::Other, "no symlink support"));

    let esc = |p: &Path| p.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
    let script = format!(
        r#"(async function () {{
  var out = {{}};
  try {{ sofuu.fs.setRoot("{root}"); }} catch (e) {{ out.setRootError = String((e && e.message) || e); }}
  try {{ var t = await sofuu.fs.readFile('inside.txt', 'utf8'); out.inside = String(t).indexOf('INSIDE-OK') >= 0; }}
  catch (e) {{ out.insideError = String((e && e.message) || e); }}
  try {{ await sofuu.fs.readFile('escape', 'utf8'); out.escapeRead = true; }} catch (e) {{ out.escapeBlocked = true; }}
  try {{ await sofuu.fs.readFile("{outside}", 'utf8'); out.absRead = true; }} catch (e) {{ out.absBlocked = true; }}
  try {{ sofuu.fs.clearRoot(); }} catch (e) {{}}
  try {{
    var sb = sofuu.sandbox.run('while(true){{}}', {{ fuel: 20000, timeoutMs: 2000 }});
    out.sandboxBounded = !!(sb && sb.success === false);
    out.sandboxErr = String((sb && (sb.errorType || sb.error)) || '');
  }} catch (e) {{ out.sandboxError = String((e && e.message) || e); }}
  try {{
    sofuu.policy.set({{ tools: {{ deny: ['bash'] }} }});
    var d = JSON.parse(sofuu.policy.evaluate({{ op: 'tool.run', tool: 'bash' }}));
    var a = JSON.parse(sofuu.policy.evaluate({{ op: 'tool.run', tool: 'read_file' }}));
    sofuu.policy.clear();
    out.policyDeny = d.decision === 'deny' && d.rule === 'R7.6/tools.deny';
    out.policyAllow = a.decision !== 'deny';
  }} catch (e) {{ out.policyError = String((e && e.message) || e); }}
  try {{
    var cfg = await sofuu.mcp.loadConfig();
    out.mcpOk = cfg.ok !== false;
    out.mcpPath = String(cfg.path || '');
    out.mcpServers = (cfg.servers || []).length;
    out.mcpError = String(cfg.error || '');
  }} catch (e) {{ out.mcpError = String((e && e.message) || e); }}
  out.toolsList = typeof (sofuu.tools && sofuu.tools.list) === 'function';
  out.serveRoutes = !!(sofuu.serve && typeof sofuu.serve.start === 'function' &&
    typeof sofuu.serve.createHandler === 'function');
  globalThis.__sofuu_doctor_proto = JSON.stringify(out);
}})();"#,
        root = esc(&dir),
        outside = esc(&outside)
    );

    if rt.eval_string(&script, "<doctor-proto>") != 0 {
        bad("protocol probe script failed to evaluate");
        *failures += 1;
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&outside);
        return false;
    }
    rt.run_jobs();
    let raw = rt.eval_repl("globalThis.__sofuu_doctor_proto || ''").unwrap_or_default();
    let clean = strip_ansi(&raw);
    let clean = clean.trim().trim_matches('\'');
    let v: serde_json::Value = serde_json::from_str(clean).unwrap_or_else(|_| serde_json::json!({}));

    let inside = v.get("inside").and_then(|b| b.as_bool()) == Some(true);
    let escape_blocked = v.get("escapeBlocked").and_then(|b| b.as_bool()) == Some(true);
    let abs_blocked = v.get("absBlocked").and_then(|b| b.as_bool()) == Some(true);
    let jail_ok = inside && escape_blocked && abs_blocked;
    if jail_ok {
        ok("reader jail: inside read OK · symlink escape blocked · absolute escape blocked");
    } else {
        let mut why = Vec::new();
        if !inside {
            why.push("inside.txt unreadable".to_string());
        }
        if !escape_blocked {
            why.push("symlink read escaped".to_string());
        }
        if !abs_blocked {
            why.push("absolute read escaped".to_string());
        }
        bad(&format!("reader jail FAILED: {}", why.join(", ")));
        *failures += 1;
    }
    if link.is_err() {
        info("symlink probe skipped: this platform cannot create symlinks here");
    }

    if v.get("sandboxBounded").and_then(|b| b.as_bool()) == Some(true) {
        ok(&format!(
            "sandbox bounded: an infinite loop stopped at its fuel (err {})",
            v.get("sandboxErr").and_then(|s| s.as_str()).unwrap_or("?")
        ));
    } else {
        bad("sandbox bounds FAILED: an infinite loop was not stopped by fuel/timeout");
        *failures += 1;
    }

    let pol_deny = v.get("policyDeny").and_then(|b| b.as_bool()) == Some(true);
    let pol_allow = v.get("policyAllow").and_then(|b| b.as_bool()) == Some(true);
    if pol_deny && pol_allow {
        ok("policy round-trip: tools.deny blocks bash and leaves read_file allowed (R7.6/tools.deny)");
    } else {
        bad("policy round-trip FAILED: the evaluator did not apply tools.deny");
        *failures += 1;
    }

    if v.get("mcpOk").and_then(|b| b.as_bool()) == Some(true) {
        ok(&format!(
            "MCP config parsed: {} ({} server(s))",
            v.get("mcpPath").and_then(|s| s.as_str()).unwrap_or("(no path)"),
            v.get("mcpServers").and_then(|n| n.as_i64()).unwrap_or(0)
        ));
    } else {
        bad(&format!(
            "MCP config FAILED: {}",
            v.get("mcpError").and_then(|s| s.as_str()).unwrap_or("unreadable")
        ));
        *failures += 1;
    }

    if v.get("toolsList").and_then(|b| b.as_bool()) == Some(true) {
        ok("tool inventory reachable (sofuu.tools.list)");
    } else {
        bad("tool inventory MISSING (sofuu.tools.list is not a function)");
        *failures += 1;
    }
    if v.get("serveRoutes").and_then(|b| b.as_bool()) == Some(true) {
        ok("serve routes registered (sofuu.serve.start/createHandler)");
    } else {
        bad("serve routes MISSING (sofuu.serve.start/createHandler not registered)");
        *failures += 1;
    }

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&outside);
    jail_ok
}
