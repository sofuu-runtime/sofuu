//! brain_gates — measure the three QTSQ_UPGRADE_PLAN.md Phase exit gates that
//! nobody had measured, and fail the build when one of them is not met.
//!
//!   Gate 1 (Phase 1): a 100k-record brain OPENS in < 1.000 s.
//!   Gate 2 (Phase 1): flush P99 < 50 ms over >= 200 incremental flushes.
//!   Gate 3 (Phase 2): 10k genuinely distinct texts yield 10k records
//!                     (the dual-key dedup gate does not collapse them).
//!
//! ## What is actually being measured
//!
//! The persistent brain has no Rust-level public open/flush API: `cma_open`,
//! `cma_flush`, `load_segmented` and `append_segments` are private to
//! `rt/memory.rs`. The public persistence API is the one a user's script hits —
//! the `sofuu.memory` JS shell (`{open, remember, recall, flush, count,
//! stats}`). This harness therefore embeds the runtime through the crate's
//! public `rt::engine` entry points (`sofuu_init` / `sofuu_get_engine` /
//! `engine_eval_repl` / `sofuu_destroy`), the same calls the CLI makes, and
//! drives that shell. Nothing here re-implements the store: the segment
//! format, the dirty-set flush, the dual-key dedup gate and the HNSW rebuild
//! are all the shipped code paths.
//!
//! The open gate is measured the only honest way: build the brain once, DROP
//! the runtime (a fresh `sofuu_init` is a cold start — no shell, no graph), and
//! time the reopen of data that is already on disk. The harness reports both
//! the bare `open()` call and the first `count()`/`recall()` after it, because
//! a lazy open would move the load out of `open()` into the first operation
//! that needs the records. The GATE is applied to `usable = open + count +
//! first_query` — the time until the brain has actually answered — which is
//! the same number under an eager or a lazy implementation. The decomposition
//! is printed either way, and the mode (eager/lazy) is inferred from which
//! operation paid the load.
//!
//! ## How to run
//!
//! ```text
//! # full gate run (the real thing; ~100k x 64-dim fixture, a few hundred MB
//! # of RAM and well under 200 MB of disk in a /tmp dir it removes itself)
//! cargo run --release -p sofuu-core --example brain_gates
//!
//! # CI-sized run (still 200 flushes, so the P99 gate stays meaningful)
//! cargo run --release -p sofuu-core --example brain_gates -- \
//!     --n 20000 --texts 2000 --flushes 200
//!
//! # leave the fixture on disk to inspect it
//! cargo run --release -p sofuu-core --example brain_gates -- --n 2000 --texts 500 --keep
//! ```
//!
//! Flags: `--n --dim --texts --flushes --dirty --keep`.
//! Defaults: n=100000, dim=64, texts=10000, flushes=200, dirty=256.
//!
//! Deterministic: every vector comes from a seeded xorshift32 keyed only by the
//! record index, every text is a pure function of its index, and no assertion
//! depends on the wall clock except the two latency gates themselves.
//!
//! Exit status: 0 only when all three gates pass; 1 when any gate fails; 2 when
//! the build has no QTSQ codec (nothing here can be measured without it).

#[cfg(has_qtsq)]
mod gates {
    use std::ffi::{CStr, CString};
    use std::time::Instant;

    /// Gate 1 (QTSQ_UPGRADE_PLAN.md §5, Phase 1): 100k open < 1.0 s.
    const OPEN_LIMIT_MS: f64 = 1_000.0;
    /// Gate 2 (§5, Phase 1): flush P99 < 50 ms.
    const FLUSH_P99_LIMIT_MS: f64 = 50.0;
    /// Gate 3 (§5, Phase 2): 10k distinct texts -> 10k records.
    const DISTINCT_TEXTS_GATE: usize = 10_000;
    /// The plan's flush gate is defined over at least this many flushes.
    const MIN_FLUSHES: usize = 200;

    struct Args {
        n: usize,
        dim: usize,
        texts: usize,
        flushes: usize,
        dirty: usize,
        keep: bool,
    }

    impl Default for Args {
        fn default() -> Self {
            Args {
                n: 100_000,
                dim: 64,
                texts: DISTINCT_TEXTS_GATE,
                flushes: MIN_FLUSHES,
                dirty: 256, // one full segment's worth of records per flush
                keep: false,
            }
        }
    }

    fn parse_args() -> Args {
        let mut a = Args::default();
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let key = argv[i].as_str();
            let val = argv.get(i + 1).cloned().unwrap_or_default();
            match key {
                "--n" => a.n = val.parse().expect("--n"),
                "--dim" => a.dim = val.parse().expect("--dim"),
                "--texts" => a.texts = val.parse().expect("--texts"),
                "--flushes" => a.flushes = val.parse().expect("--flushes"),
                "--dirty" => a.dirty = val.parse().expect("--dirty"),
                "--keep" => {
                    a.keep = true;
                    i += 1;
                    continue;
                }
                "--help" | "-h" => {
                    println!(
                        "usage: brain_gates [--n N] [--dim D] [--texts T] [--flushes F]\n\
                         \x20                  [--dirty D] [--keep]\n\
                         \x20      defaults: n=100000 dim=64 texts=10000 flushes=200 dirty=256\n\
                         \x20      gates: open<{OPEN_LIMIT_MS}ms, flush_p99<{FLUSH_P99_LIMIT_MS}ms, \
                         texts->records"
                    );
                    std::process::exit(0);
                }
                other => panic!("unknown flag {other} (try --help)"),
            }
            i += 2;
        }
        a
    }

    /* ══════════════════════════════════════════════════════════════
     * The embedded runtime
     * ══════════════════════════════════════════════════════════════ */

    /// One embedded Sofuu runtime. Dropping it destroys the QuickJS engine AND
    /// every brain shell it holds, so a second value in this process is a real
    /// cold start for the store under test.
    struct Rt(*mut sofuu_core::rt::engine::SofuuRuntime);

    impl Rt {
        fn new() -> Rt {
            let p = unsafe { sofuu_core::rt::engine::sofuu_init() };
            assert!(!p.is_null(), "sofuu_init returned NULL");
            Rt(p)
        }

        /// Evaluate one JS expression and return its REPL rendering with the
        /// terminal colour codes stripped. `Err` carries the JS exception.
        fn eval(&self, src: &str) -> Result<String, String> {
            let eng = unsafe { sofuu_core::rt::engine::sofuu_get_engine(self.0) };
            let c = CString::new(src).map_err(|e| format!("NUL in JS source: {e}"))?;
            let mut is_err: i32 = 0;
            let out =
                unsafe { sofuu_core::rt::engine::engine_eval_repl(eng, c.as_ptr(), &mut is_err) };
            if out.is_null() {
                return Err("engine_eval_repl returned NULL".to_string());
            }
            let raw = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
            unsafe { libc::free(out as *mut core::ffi::c_void) };
            let clean = strip_ansi(&raw);
            if is_err != 0 {
                Err(clean.trim().to_string())
            } else {
                Ok(clean)
            }
        }

        /// Evaluate an expression whose value is an object/number and parse it.
        fn eval_json(&self, src: &str) -> Result<serde_json::Value, String> {
            let repl = self.eval(src)?;
            json_of(&repl)
        }
    }

    impl Drop for Rt {
        fn drop(&mut self) {
            unsafe { sofuu_core::rt::engine::sofuu_destroy(self.0) };
        }
    }

    /// The REPL renders values with ANSI colour escapes; the JSON underneath
    /// is what we want.
    fn strip_ansi(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut it = s.chars();
        while let Some(c) = it.next() {
            if c == '\u{1b}' {
                for c2 in it.by_ref() {
                    if c2 == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    /// `engine_eval_repl` prints a JS string as `'…'` and an object as JSON.
    fn json_of(repl: &str) -> Result<serde_json::Value, String> {
        let t = repl.trim();
        if t.is_empty() {
            return Err("empty REPL result".to_string());
        }
        let body = if t.len() >= 2 && t.starts_with('\'') && t.ends_with('\'') {
            &t[1..t.len() - 1]
        } else {
            t
        };
        serde_json::from_str(body).map_err(|e| format!("non-JSON REPL result {body:?}: {e}"))
    }

    /// Fill `$NAME` placeholders. JS is kept in raw strings so its braces never
    /// collide with `format!`.
    fn tpl(src: &str, subs: &[(&str, String)]) -> String {
        let mut out = src.to_string();
        for (k, v) in subs {
            out = out.replace(k, v);
        }
        out
    }

    /* ══════════════════════════════════════════════════════════════
     * JS fixtures (deterministic, index-keyed)
     * ══════════════════════════════════════════════════════════════ */

    /// A seeded xorshift32 vector per record index. The same function feeds
    /// the initial ingest and every later "dirty this record again" write, so
    /// the two agree without keeping 100k Float32Arrays alive in JS.
    const MKVEC: &str = r#"
      globalThis.mkvec = function(i){
        let s = (i ^ 0x9e3779b9) >>> 0;
        const v = new Float32Array($DIM);
        for (let d = 0; d < $DIM; d++) {
          s ^= (s << 13); s >>>= 0;
          s ^= (s >>> 17);
          s ^= (s << 5);  s >>>= 0;
          v[d] = (s >>> 8) / 16777216.0;
        }
        return v;
      };
      globalThis.rowtext = function(i){
        return "gate record " + i + " about topic " + (i % 97);
      };
    "#;

    fn with_fixture(dim: usize, body: &str) -> String {
        tpl(
            &format!("(function(){{ {} {} }})()", MKVEC, body),
            &[("$DIM", dim.to_string())],
        )
    }

    /* ══════════════════════════════════════════════════════════════
     * The three gates
     * ══════════════════════════════════════════════════════════════ */

    struct OpenReport {
        open_ms: f64,
        count_ms: f64,
        first_ms: f64,
        steady_ms: f64,
        lazy: bool,
        records: usize,
        stats: String,
    }

    impl OpenReport {
        /// The number the gate is applied to: the runtime has opened the brain
        /// AND answered a query. Under an eager open this is `open + ~2 ms`;
        /// under a lazy open it carries the materialisation, wherever it moved.
        fn usable_ms(&self) -> f64 {
            self.open_ms + self.count_ms + self.first_ms
        }
        fn pass(&self) -> bool {
            self.usable_ms() <= OPEN_LIMIT_MS
        }
        fn mode(&self) -> &'static str {
            if self.lazy {
                "lazy"
            } else {
                "eager"
            }
        }
    }

    /// Phase 1 — build the fixture and put it on disk. Not a gate; its cost is
    /// reported so the CI price of the harness is visible.
    fn build_fixture(a: &Args, path: &str) -> Result<(usize, String, f64, f64), String> {
        let rt = Rt::new();
        let body = r#"
          globalThis.mind = sofuu.memory.open("$PATH", $DIM);
          let last = -1;
          for (let i = 0; i < $N; i++) {
            last = mind.remember(mkvec(i), rowtext(i), "user", 0);
          }
          return {count: mind.count(), last: last};
        "#;
        let ingest = tpl(body, &[("$PATH", path.to_string()), ("$N", a.n.to_string())]);
        let src = with_fixture(a.dim, &ingest);
        let t0 = Instant::now();
        let v = rt.eval_json(&src).map_err(|e| format!("ingest failed: {e}"))?;
        let ingest_ms = t0.elapsed().as_secs_f64() * 1e3;
        let count = v.get("count").and_then(|c| c.as_u64()).unwrap_or(0) as usize;

        let t1 = Instant::now();
        rt.eval_json(r#"(function(){ globalThis.mind.flush(); return {ok:true}; })()"#)
            .map_err(|e| format!("build flush failed: {e}"))?;
        let flush_ms = t1.elapsed().as_secs_f64() * 1e3;

        Ok((count, stats_of(&rt), ingest_ms, flush_ms))
    }

    /// `cma.stats()` as compact JSON (it is the store's own accounting:
    /// records / segments / versions / bytes).
    fn stats_of(rt: &Rt) -> String {
        rt.eval("(function(){ return mind.stats(); })()")
            .map(|s| json_of(&s).map(|v| v.to_string()).unwrap_or(s))
            .unwrap_or_else(|e| format!("stats unavailable: {e}"))
    }

    /// Phase 2 — time a cold reopen of data already on disk.
    fn measure_open(a: &Args, path: &str) -> Result<OpenReport, String> {
        let rt = Rt::new(); // fresh engine: no shell, no graph, no cache
        /* NOTE: every body handed to `with_fixture` is the INSIDE of an IIFE
         * — it must `return`, and must not be wrapped in its own `(function…)()`
         * or the wrapper's value is discarded (undefined). */
        let open_src = tpl(
            r#"globalThis.mind = sofuu.memory.open("$PATH", $DIM); return {ok:true};"#,
            &[("$PATH", path.to_string())],
        );
        let t0 = Instant::now();
        rt.eval_json(&with_fixture(a.dim, &open_src))
            .map_err(|e| format!("open failed: {e}"))?;
        let open_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let v = rt
            .eval_json(r#"(function(){ return {count: mind.count()}; })()"#)
            .map_err(|e| format!("count after open failed: {e}"))?;
        let count_ms = t1.elapsed().as_secs_f64() * 1e3;
        let records = v.get("count").and_then(|c| c.as_u64()).unwrap_or(0) as usize;

        // A query goes through the HNSW; on a lazy open this is the call that
        // has to materialise the graph.
        let q = r#"(function(){ const h = mind.recall("gate record 7 about topic 7", 10); return {hits: h.length}; })()"#;
        let t2 = Instant::now();
        rt.eval_json(q).map_err(|e| format!("first query failed: {e}"))?;
        let first_ms = t2.elapsed().as_secs_f64() * 1e3;

        let mut steady: Vec<f64> = Vec::with_capacity(3);
        for _ in 0..3 {
            let t = Instant::now();
            rt.eval_json(q).map_err(|e| format!("steady query failed: {e}"))?;
            steady.push(t.elapsed().as_secs_f64() * 1e3);
        }
        steady.sort_by(|x, y| x.total_cmp(y));
        let steady_ms = steady[steady.len() / 2];

        let stats = stats_of(&rt);
        // Lazy = the load was paid by count() or by the first query, not by
        // open(). The 25 ms floor keeps a slow query on a small brain from
        // being mistaken for a materialisation.
        let lazy = (count_ms > 25.0 && count_ms > 4.0 * steady_ms)
            || (first_ms > 25.0 && first_ms > 4.0 * steady_ms);
        Ok(OpenReport {
            open_ms,
            count_ms,
            first_ms,
            steady_ms,
            lazy,
            records,
            stats,
        })
    }

    struct FlushReport {
        p50: f64,
        p99: f64,
        max: f64,
        n: usize,
        dirty: usize,
        written_total: u64,
        compactions: usize,
        records_after: usize,
    }

    impl FlushReport {
        fn pass(&self) -> bool {
            self.p99 <= FLUSH_P99_LIMIT_MS
        }
    }

    /// Phase 3 — >= 200 incremental flushes, each dirtying `--dirty` records
    /// (re-remembering a stored fact reinforces it, which marks exactly that
    /// record dirty — the realistic write: a session's recalls/reinforcements,
    /// not a full rewrite).
    fn measure_flushes(a: &Args, path: &str) -> Result<FlushReport, String> {
        let rt = Rt::new();
        let open_src = tpl(
            r#"globalThis.mind = sofuu.memory.open("$PATH", $DIM); return {count: mind.count()};"#,
            &[("$PATH", path.to_string())],
        );
        rt.eval_json(&with_fixture(a.dim, &open_src))
            .map_err(|e| format!("flush-phase open failed: {e}"))?;

        let mut dur: Vec<f64> = Vec::with_capacity(a.flushes);
        let mut written_total: u64 = 0;
        let mut compactions = 0usize;
        for f in 0..a.flushes {
            let base = (f * a.dirty) % a.n;
            let dirty_src = tpl(
                r#"for (let j = 0; j < $DIRTY; j++) {
                      const i = ($BASE + j) % $N;
                      mind.remember(mkvec(i), rowtext(i), "user", 0);
                    }
                    return {count: mind.count()};"#,
                &[
                    ("$DIRTY", a.dirty.to_string()),
                    ("$BASE", base.to_string()),
                    ("$N", a.n.to_string()),
                ],
            );
            rt.eval_json(&with_fixture(a.dim, &dirty_src))
                .map_err(|e| format!("dirty batch {f} failed: {e}"))?;

            let t = Instant::now();
            let v = rt
                .eval_json(r#"(function(){ return mind.flush(); })()"#)
                .map_err(|e| format!("flush {f} failed: {e}"))?;
            dur.push(t.elapsed().as_secs_f64() * 1e3);
            let wrote = v.get("recordsWritten").and_then(|x| x.as_u64()).unwrap_or(0);
            written_total += wrote;
            /* A flush that writes far more than the dirty batch is a
             * compaction (or a capacity eviction), not an O(dirty) append.
             * Counted so a P99 dominated by one rewrite is legible. */
            if wrote > (a.dirty as u64) * 2 {
                compactions += 1;
            }
        }
        let records_after = rt
            .eval_json(r#"(function(){ return {count: mind.count()}; })()"#)
            .map_err(|e| format!("count after flushes failed: {e}"))?
            .get("count")
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;

        let mut sorted = dur.clone();
        sorted.sort_by(|x, y| x.total_cmp(y));
        Ok(FlushReport {
            p50: pct(&sorted, 0.50),
            p99: pct(&sorted, 0.99),
            max: sorted.last().copied().unwrap_or(0.0),
            n: sorted.len(),
            dirty: a.dirty,
            written_total,
            compactions,
            records_after,
        })
    }

    /// Nearest-rank percentile.
    fn pct(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1);
        sorted[idx]
    }

    struct DistinctReport {
        attempted: usize,
        after_ingest: usize,
        after_flush: usize,
        after_reopen: usize,
    }

    impl DistinctReport {
        fn collapsed(&self) -> usize {
            self.attempted.saturating_sub(self.after_reopen)
        }
        fn pass(&self) -> bool {
            self.after_reopen == self.attempted
        }
    }

    /// Phase 4 — 10k genuinely distinct texts. Ingested through the store's
    /// OWN embedder (`remember(text)`, no hand-built vectors), because that is
    /// the path whose dedup gate swallowed 10,000 texts into 340: near
    /// vectors, distinct text. The texts differ in a unique number, so token
    /// Jaccard stays far below the 0.85 text key and every one must survive.
    fn measure_distinct(a: &Args, path: &str) -> Result<DistinctReport, String> {
        let rt = Rt::new();
        let ingest = tpl(
            r#"globalThis.tmind = sofuu.memory.open("$PATH", $DIM);
                for (let i = 0; i < $TEXTS; i++) {
                  tmind.remember("record " + i + " about topic " + (i % 13), "user");
                }
                return {count: tmind.count()};"#,
            &[("$PATH", path.to_string()), ("$TEXTS", a.texts.to_string())],
        );
        let after_ingest = rt
            .eval_json(&with_fixture(a.dim, &ingest))
            .map_err(|e| format!("distinct ingest failed: {e}"))?
            .get("count")
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;

        let after_flush = rt
            .eval_json(
                r#"(function(){ tmind.flush(); return {count: tmind.count()}; })()"#,
            )
            .map_err(|e| format!("distinct flush failed: {e}"))?
            .get("count")
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;
        drop(rt); // close the store before reopening it

        let rt2 = Rt::new();
        let reopen = tpl(
            r#"globalThis.tmind = sofuu.memory.open("$PATH", $DIM); return {count: tmind.count()};"#,
            &[("$PATH", path.to_string())],
        );
        let after_reopen = rt2
            .eval_json(&with_fixture(a.dim, &reopen))
            .map_err(|e| format!("distinct reopen failed: {e}"))?
            .get("count")
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;

        Ok(DistinctReport {
            attempted: a.texts,
            after_ingest,
            after_flush,
            after_reopen,
        })
    }

    /* ══════════════════════════════════════════════════════════════
     * Temp dir + accounting
     * ══════════════════════════════════════════════════════════════ */

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> std::io::Result<TempDir> {
            let p = std::env::temp_dir().join(format!("sofuu-brain-gates-{}", std::process::id()));
            if p.exists() {
                std::fs::remove_dir_all(&p)?;
            }
            std::fs::create_dir_all(&p)?;
            Ok(TempDir(p))
        }
        fn path(&self, name: &str) -> String {
            self.0.join(name).to_string_lossy().to_string()
        }
        fn bytes(&self) -> u64 {
            fn walk(dir: &std::path::Path) -> u64 {
                let mut total = 0;
                let Ok(rd) = std::fs::read_dir(dir) else {
                    return 0;
                };
                for e in rd.flatten() {
                    let p = e.path();
                    match e.file_type() {
                        Ok(t) if t.is_dir() => total += walk(&p),
                        Ok(_) => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                        _ => {}
                    }
                }
                total
            }
            walk(&self.0)
        }
        fn cleanup(self) -> std::io::Result<()> {
            if self.0.exists() {
                std::fs::remove_dir_all(&self.0)
            } else {
                Ok(())
            }
        }
    }

    /* ══════════════════════════════════════════════════════════════
     * main
     * ══════════════════════════════════════════════════════════════ */

    pub fn run() {
        let a = parse_args();
        let wall = Instant::now();
        if a.flushes < MIN_FLUSHES {
            println!(
                "!! --flushes {} < {}: the plan's flush P99 gate is defined over at least {} \
                 flushes; this run's P99 is NOT the gate",
                a.flushes, MIN_FLUSHES, MIN_FLUSHES
            );
        }
        /* The store caps live records at 100k by default; the gate is about
         * open latency, not the capacity policy, so the cap is disabled for
         * the run and the fixture is exactly `--n` records. */
        std::env::set_var("SOFUU_BRAIN_MAX_RECORDS", "0");

        println!("── brain_gates: QTSQ_UPGRADE_PLAN.md Phase exit gates, measured ──");
        println!(
            "n={} dim={} texts={} flushes={} dirty_per_flush={} cap=disabled",
            a.n, a.dim, a.texts, a.flushes, a.dirty
        );

        let tmp = match TempDir::new() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("brain_gates: cannot create a temp dir: {e}");
                std::process::exit(2);
            }
        };
        let big = tmp.path("big.qtsq");
        let texts_db = tmp.path("texts.qtsq");

        // ── Phase 1: fixture built once, on disk ──────────────────────
        let built = match build_fixture(&a, &big) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("brain_gates: {e}");
                let _ = tmp.cleanup();
                std::process::exit(2);
            }
        };
        let (built_count, build_stats, ingest_ms, build_flush_ms) = built;
        println!(
            "BUILD n={} stored={} ingest={:.1}ms initial_flush={:.1}ms",
            a.n, built_count, ingest_ms, build_flush_ms
        );
        println!("      store: {}", build_stats.replace('\n', " "));

        // ── Phase 2: cold reopen ──────────────────────────────────────
        let open = match measure_open(&a, &big) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("brain_gates: {e}");
                let _ = tmp.cleanup();
                std::process::exit(2);
            }
        };
        let integrity_ok = open.records == built_count && built_count == a.n;
        println!(
            "OPEN  open_call={:.1}ms count()={:.1}ms first_query={:.1}ms steady_query={:.1}ms \
             mode={} records={}/{}",
            open.open_ms,
            open.count_ms,
            open.first_ms,
            open.steady_ms,
            open.mode(),
            open.records,
            a.n
        );
        println!("      store: {}", open.stats.replace('\n', " "));
        println!(
            "      usable(open+count+first_query)={:.1}ms  limit={:.0}ms  {}   [gate applies to \
             usable, the number that is true for an eager AND a lazy open]",
            open.usable_ms(),
            OPEN_LIMIT_MS,
            if open.pass() { "PASS" } else { "FAIL" }
        );
        if !integrity_ok {
            println!(
                "      !! record count changed across the reopen (stored {built_count}, opened {}) \
                 — the open gate number is measured on the records that DID load",
                open.records
            );
        }

        // ── Phase 3: flush P99 ────────────────────────────────────────
        let flushes = match measure_flushes(&a, &big) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("brain_gates: {e}");
                let _ = tmp.cleanup();
                std::process::exit(2);
            }
        };
        println!(
            "FLUSH flushes={} dirty_each={} written_total={} compactions={} p50={:.1}ms \
             p99={:.1}ms max={:.1}ms limit={:.0}ms records_after={}  {}",
            flushes.n,
            flushes.dirty,
            flushes.written_total,
            flushes.compactions,
            flushes.p50,
            flushes.p99,
            flushes.max,
            FLUSH_P99_LIMIT_MS,
            flushes.records_after,
            if flushes.pass() { "PASS" } else { "FAIL" }
        );
        if flushes.compactions > 0 {
            println!(
                "      note: {} of {} flushes were full compactions (wrote > 2x the dirty \
                 batch), not O(dirty) appends — they are included in P99",
                flushes.compactions, flushes.n
            );
        }

        // ── Phase 4: 10k distinct texts ───────────────────────────────
        let distinct = match measure_distinct(&a, &texts_db) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("brain_gates: {e}");
                let _ = tmp.cleanup();
                std::process::exit(2);
            }
        };
        println!(
            "DISTINCT attempted={} after_ingest={} after_flush={} after_reopen={} collapsed={}  {}",
            distinct.attempted,
            distinct.after_ingest,
            distinct.after_flush,
            distinct.after_reopen,
            distinct.collapsed(),
            if distinct.pass() { "PASS" } else { "FAIL" }
        );
        if !distinct.pass() {
            println!(
                "      !! the dual-key dedup gate COLLAPSED {} of {} genuinely distinct texts \
                 (unique number + shared topic words) — that is the failure the plan exists to fix",
                distinct.collapsed(),
                distinct.attempted
            );
        }

        // ── Cleanup + verdict ─────────────────────────────────────────
        let bytes = tmp.bytes();
        let keep = a.keep;
        let tmp_path = tmp.0.clone();
        if !keep {
            if let Err(e) = tmp.cleanup() {
                eprintln!("brain_gates: could not remove {}: {e}", tmp_path.display());
            }
        }
        println!(
            "DISK  wrote {:.1} MB under {} ({})",
            bytes as f64 / (1024.0 * 1024.0),
            tmp_path.display(),
            if keep { "kept" } else { "removed" }
        );

        let open_ok = open.pass() && integrity_ok;
        let flush_ok = flushes.pass();
        let distinct_ok = distinct.pass();
        let fails = [!open_ok, !flush_ok, !distinct_ok].iter().filter(|&&f| f).count();
        /* One line, both the measured number and the threshold, in the shape
         * the plan's gate table reads: the comparison sign flips when a gate
         * fails, so a red run can never be misread as green. */
        let open_cmp = format!(
            "{}{:.3}s",
            if open_ok { "<=" } else { ">" },
            OPEN_LIMIT_MS / 1000.0
        );
        let flush_cmp = format!(
            "{}{:.0}ms",
            if flush_ok { "<=" } else { ">" },
            FLUSH_P99_LIMIT_MS
        );
        println!();
        println!(
            "GATES: open={:.3}s {} ({}) | flush_p99={:.1}ms {} ({}) | distinct={}/{} {}",
            open.usable_ms() / 1000.0,
            if open_ok { "PASS" } else { "FAIL" },
            open_cmp,
            flushes.p99,
            if flush_ok { "PASS" } else { "FAIL" },
            flush_cmp,
            distinct.after_reopen,
            distinct.attempted,
            if distinct_ok { "PASS" } else { "FAIL" }
        );
        println!(
            "RESULT {}/3 gates pass | wall={:.1}s",
            3 - fails,
            wall.elapsed().as_secs_f64()
        );
        std::process::exit(if fails == 0 { 0 } else { 1 });
    }
}

#[cfg(has_qtsq)]
fn main() {
    gates::run();
}

#[cfg(not(has_qtsq))]
fn main() {
    eprintln!(
        "brain_gates: this build has no QTSQ codec (§rt::brain_store is compiled out), so \
         there is no persistent store to measure. Build against a QTSQ checkout: \
         SOFUU_QTSQ_DIR=~/projects/<qtsq-checkout> cargo run --release -p sofuu-core \
         --example brain_gates"
    );
    std::process::exit(2);
}
