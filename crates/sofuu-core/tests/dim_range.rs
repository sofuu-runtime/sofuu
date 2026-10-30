// crates/sofuu-core/tests/dim_range.rs — regression gate for the vector-width
// range `sofuu.memory` accepts.
//
// The brain used to be hard-wired enough around the 768d chat geometry that
// an unusual embedding width was a coin flip. This test pins BOTH ends of the
// documented range through the REAL runtime, exactly the way the shipped
// binary runs JS:
//
//   • dims 1, 2, 7, 16, 384, 1024  — a FULL round trip each:
//         open → remember → count → flush → close → reopen → count → recall
//     with the top hit required to be the exact record written (distance ~0).
//     (384 is the semantic-model width, 1024 the hash-embedder width; the
//     others are widths no baked model covers — the legacy-dim-N space.)
//   • dim 8192 (CAP_DIM)           — the cap is INCLUSIVE: opened, one record
//     remembered, flushed, reopened with the row still there. Set
//     SOFUU_DIM_RANGE_FULL=1 to run the same full round trip (including
//     recall) at the cap.
//   • dim 8193 (CAP_DIM + 1)       — one past the cap must be REFUSED with a
//     thrown, coded error (E_CORRUPT), must create no brain on disk, and must
//     leave the process healthy — never a panic/abort. The engine is probed
//     again after the refusal as the liveness proof.
//
// Harness: the same one ml_shim_hardening.rs uses — boot sofuu_ffi's runtime
// (sofuu.memory is registered by engine_register_builtins, rt/memory.rs:1258
// is the `vec_dim > 8192` check under test) and drive it with eval_string;
// any contract break THROWS in JS and the eval returns non-zero. This is a
// separate test binary = its own process and its own libuv loop, so it needs
// no cross-binary serialization.
//
// Run:  CARGO_TARGET_DIR=/tmp/cargo-main cargo test -p sofuu-core --test dim_range -- --nocapture

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Widths that must survive a full round trip (fast — a few records each).
const FULL_ROUND_TRIP_DIMS: &[usize] = &[1, 2, 7, 16, 384, 1024];
/// The documented cap: `vec_dim > 8192` is refused (rt/memory.rs cma_open_ex).
const CAP_DIM: usize = 8192;
const OVER_CAP_DIM: usize = CAP_DIM + 1;

/// A unique temp directory per run, removed on the way out (also on a failed
/// assertion — Drop runs during unwinding), so nothing leaks between runs.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "sofuu_dim_range_{}_{}_{}",
            tag,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&dir).expect("create the per-run temp dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn brain_path(dir: &Path, dim: usize) -> PathBuf {
    dir.join(format!("brain_{dim}.qtsq"))
}

/// A JS path literal for `path` (the temp paths are plain ASCII, but quote
/// and backslash escaping keeps this honest on any host).
fn js_str(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"")
}

/// One full round trip at `dim`, as a JS program that THROWS (with the dim and
/// the failing step in the message) on any contract break. `recall` adds the
/// read-back assertions (off for the cap's cheap gate).
fn js_round_trip(dim: usize, path: &Path, records: usize, recall: bool) -> String {
    let src = r#"
(function () {
  var DIM = __DIM__, PATH = "__PATH__", RECORDS = __RECORDS__, DO_RECALL = __RECALL__;
  var M = sofuu.memory;
  function fail(step, extra) {
    throw new Error('dim ' + DIM + ': ' + step + (extra === undefined ? '' : ' — ' + extra));
  }
  if (!M || typeof M.open !== 'function') fail('sofuu.memory is not registered in this build');
  /* Deterministic, NON-degenerate vector of any width: values live in
   * [0.125, 0.875], so even dim 1 never sends the zero vector. */
  function vecFor(seed) {
    var v = new Float32Array(DIM);
    for (var i = 0; i < DIM; i++) v[i] = (((i + seed) % 7) + 1) / 8;
    return v;
  }
  var text = 'dim ' + DIM + ' fact';
  var v = vecFor(DIM + 1);

  var b = M.open(PATH, DIM);
  if (!b || typeof b.remember !== 'function') fail('open returned no handle');
  if (b.count() !== 0) fail('a fresh store is not empty', 'count ' + b.count());

  for (var r = 0; r < RECORDS; r++) {
    var id = b.remember(r === 0 ? v : vecFor(DIM + 1 + r),
                        r === 0 ? text : text + ' #' + r, 'user', 0);
    if (typeof id !== 'number' || id < 0) fail('remember returned ' + id);
  }
  if (b.count() !== RECORDS) fail('count after remember', b.count() + '/' + RECORDS);
  var f = b.flush();
  if (!f || f.ok !== true || f.recordsWritten !== RECORDS) fail('flush reported ' + JSON.stringify(f));
  if (M.close(b) !== true) fail('close did not report success');
  if (M.handles() !== 0) fail('a handle leaked through close', 'handles ' + M.handles());

  var b2 = M.open(PATH, DIM);
  if (!b2) fail('the brain did not reopen at its own width');
  if (b2.count() !== RECORDS) fail('count after reopen', b2.count() + '/' + RECORDS);
  if (DO_RECALL) {
    var hits = b2.recall(v, 3);
    if (!hits || hits.length < 1) fail('recall returned no hit');
    if (hits[0].text !== text) fail('top hit text', String(hits[0].text));
    if (!(hits[0].distance < 1e-4)) fail('top hit is not an exact match', 'distance ' + hits[0].distance);
    if (hits[0].role !== 'user') fail('the hit lost its role', String(hits[0].role));
  }
  M.close(b2);
  if (M.handles() !== 0) fail('a handle leaked through the second close', 'handles ' + M.handles());
})();
"#;
    src.replace("__DIM__", &dim.to_string())
        .replace("__PATH__", &js_str(path))
        .replace("__RECORDS__", &records.to_string())
        .replace("__RECALL__", if recall { "true" } else { "false" })
}

/// One past the cap: `open` must throw a coded refusal, no handle may leak,
/// and a normal brain must still work afterwards (liveness).
fn js_over_cap(over_dim: usize, over_path: &Path, live_path: &Path) -> String {
    let src = r#"
(function () {
  var OVER = __OVER__, OVER_PATH = "__OVER_PATH__", LIVE_PATH = "__LIVE_PATH__";
  var M = sofuu.memory;
  var e = null;
  try {
    var bad = M.open(OVER_PATH, OVER);
    /* Reaching here means the cap was not enforced at all. */
    throw new Error('dim ' + OVER + ': open was ACCEPTED (returned a handle) — the >8192 cap is not enforced');
  } catch (thrown) {
    if (String(thrown && thrown.message).indexOf('open was ACCEPTED') >= 0) throw thrown;
    e = thrown;
  }
  if (!e) throw new Error('dim ' + OVER + ': open neither opened nor threw');
  if (e.code !== 'E_CORRUPT') throw new Error('dim ' + OVER + ': refusal lost its code — got ' + String(e.code));
  if (!/could not be opened/.test(String(e.message))) {
    throw new Error('dim ' + OVER + ': refusal message is not the documented one — got ' + String(e.message));
  }
  if (M.handles() !== 0) throw new Error('dim ' + OVER + ': the refusal leaked a handle (' + M.handles() + ')');

  /* The engine must be HEALTHY after the refusal: a legitimate brain still
   * opens and stores a row. A panic/abort would never reach this line. */
  var live = M.open(LIVE_PATH, 16);
  var id = live.remember(new Float32Array(16).map(function (_, i) { return (i + 1) / 17; }),
                         'after the refusal', 'user', 0);
  if (typeof id !== 'number' || id < 0) throw new Error('the engine is broken after the refusal: remember -> ' + id);
  if (live.count() !== 1) throw new Error('the engine is broken after the refusal: count -> ' + live.count());
  M.close(live);
})();
"#;
    src.replace("__OVER__", &over_dim.to_string())
        .replace("__OVER_PATH__", &js_str(over_path))
        .replace("__LIVE_PATH__", &js_str(live_path))
}

#[test]
fn memory_accepts_arbitrary_dims_up_to_the_cap_and_refuses_above_it() {
    if !sofuu_core::HAS_QTSQ {
        // The same honest skip the JS suites use (exit 77): a QTSQ-free build
        // has no sofuu.memory at all, so there is nothing here to regress.
        eprintln!("SKIP dim_range — this build has no QTSQ codec, so sofuu.memory does not exist");
        return;
    }

    let dir = TempDir::new("main");
    let full_cap = std::env::var("SOFUU_DIM_RANGE_FULL").is_ok();
    println!("dim_range: temp dir {}", dir.path().display());

    let rt = sofuu_ffi::SofuuRuntime::init().expect("runtime init");
    assert!(
        rt.eval_string(
            "if (typeof sofuu.memory === 'undefined' || typeof sofuu.memory.open !== 'function') \
             throw new Error('sofuu.memory is not registered although the build links QTSQ');",
            "<dim-range-boot>"
        ) == 0,
        "sofuu.memory must be registered in a QTSQ build"
    );

    /* ── the unusual widths: a FULL round trip each ──────────────────── */
    for &dim in FULL_ROUND_TRIP_DIMS {
        let path = brain_path(dir.path(), dim);
        let prog = js_round_trip(dim, &path, 2, true);
        let rc = rt.eval_string(&prog, &format!("<dim-range-{dim}d>"));
        assert_eq!(
            rc, 0,
            "{dim}d round trip FAILED — open/remember/flush/close/reopen/count/recall is not \
             width-agnostic (see the thrown Error above, which names the step)"
        );
        let segment_dir = PathBuf::from(format!("{}.d", path.display()));
        assert!(
            path.exists() && segment_dir.exists(),
            "{dim}d: flush must leave a segmented brain on disk (manifest + .d segment dir)"
        );
        println!("ok: {dim}d open → remember → flush → close → reopen → count → recall");
    }

    /* ── the cap itself (8192): inclusive, cheap gate by default ─────── */
    {
        let path = brain_path(dir.path(), CAP_DIM);
        let (records, recall) = if full_cap { (2, true) } else { (1, false) };
        let prog = js_round_trip(CAP_DIM, &path, records, recall);
        let rc = rt.eval_string(&prog, &format!("<dim-range-{CAP_DIM}d>"));
        assert_eq!(
            rc, 0,
            "{CAP_DIM}d (the documented cap) must behave like every other width — the cap is inclusive"
        );
        println!(
            "ok: {CAP_DIM}d (cap) opened, {records} record(s) remembered, flushed, reopened, count preserved{}",
            if recall { ", recall verified" } else { " (SOFUU_DIM_RANGE_FULL=1 adds the recall leg)" }
        );
    }

    /* ── one past the cap: refused cleanly, no crash ─────────────────── */
    {
        let over = brain_path(dir.path(), OVER_CAP_DIM);
        /* A FRESH path: the liveness control must prove the engine still
         * accepts writes, not re-read a brain an earlier dim already filled. */
        let live = dir.path().join("post_refusal_control.qtsq");
        let prog = js_over_cap(OVER_CAP_DIM, &over, &live);
        let rc = rt.eval_string(&prog, &format!("<dim-range-{OVER_CAP_DIM}d>"));
        assert_eq!(
            rc, 0,
            "{OVER_CAP_DIM}d must be REFUSED with a thrown E_CORRUPT and a still-healthy engine — \
             an accepted open, a silent handle, or a panic/abort all land here"
        );
        assert!(
            !over.exists(),
            "{OVER_CAP_DIM}d: a refused open must not leave a brain file behind"
        );
        assert!(
            live.exists(),
            "the post-refusal control brain (16d) must have been written — proof the engine ran on"
        );

        // Belt and braces from Rust: the process that survived a refusal can
        // still evaluate JS (a JS throw would give rc 1, an abort would kill
        // the test binary before this line).
        assert_eq!(
            rt.eval_string("if (1 + 1 !== 2) throw new Error('engine dead');", "<dim-range-liveness>"),
            0,
            "the engine must still evaluate JS after an over-cap refusal"
        );
        println!("ok: {OVER_CAP_DIM}d refused with E_CORRUPT (no file, no leaked handle, engine still healthy)");
    }

    println!("ok: dim_range passed — 1 … {CAP_DIM} accepted, {OVER_CAP_DIM} refused cleanly");
}
