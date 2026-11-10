// build.rs — mirrors crates/sofuu-ffi/build.rs's QTSQ probe for sofuu-core.
//
// `cargo:rustc-cfg` emitted by a build script applies only to that script's
// own crate, so the two crates probe independently. rt/memory.rs uses
// `#[cfg(has_qtsq)]` to keep the QTSQ-dependent memory shells (brain-file +
// KV page persistence) compiled only when the vendored libqtsq.a checkout is
// available — the same condition the C build uses for SOFUU_QTSQ_PRESENT.
// Without it the `mod_*_register` exports exist as no-ops and engine.c (which
// only calls them under its own QTSQ guard) sees the same JS-visible absence.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

// ── PC-07: source revision stamp ──────────────────────────────────────────
//
// AUDIT-2026-10-29: published binaries carried no way back to their source.
// The stamp is the short git revision, plus `-dirty` when the tree differs
// from it (the same signal scripts/pack/stage_legal.sh records in
// SHIP-MANIFEST.txt). capabilities.rs reports it as `build.revision`, so it is
// queryable with `sofuu capabilities` and present in the binary's strings.
//
// git is optional at build time: no checkout or no git binary yields
// "unknown" instead of a build failure (release tarball, vendored source
// drops). SOFUU_BUILD overrides the whole thing so those builds can stamp the
// revision they came from.
/// Read a source file that a person or another agent may be rewriting right now.
///
/// `fs::read` on a file whose writer just truncated it returns a short buffer and
/// no error, and a build script has no way to notice. That is how a torn read of
/// `src/js/agent.js` once embedded a fragment of the file: the resulting binary
/// failed `shipped::tests::shipped_sources_are_nonempty_and_parseable_names` with
/// "agent.js: failed to compile to bytecode", and cargo did not rebuild it - the
/// file on disk was unchanged from what cargo had already recorded, so nothing
/// invalidated the stale embed.
///
/// Read, then check the file did not change underneath the read (same length
/// before and after, and the bytes match that length). Retry a few times, then
/// give up and return whatever the last read produced.
fn read_stable(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    for _ in 0..5 {
        let before = std::fs::metadata(path)?.len();
        let data = std::fs::read(path)?;
        let after = std::fs::metadata(path)?.len();
        if before == after && data.len() as u64 == after {
            return Ok(data);
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    eprintln!("cargo:warning={} kept changing while being read", path.display());
    std::fs::read(path)
}

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn build_stamp(repo: &Path) -> String {
    if let Ok(v) = env::var("SOFUU_BUILD") {
        if !v.trim().is_empty() {
            return v.trim().to_string();
        }
    }
    let Some(sha) = git(repo, &["rev-parse", "--short", "HEAD"]).filter(|s| !s.is_empty()) else {
        return "unknown".to_string();
    };
    /* A modified TRACKED file means the binary does not match the commit.
     * Untracked files do not: this repository always carries local planning
     * documents (AUDIT_PLAN.md, REMEDIATION_PLAN.md, …) that are deliberately
     * not committed, and a release built beside them is still exactly the
     * commit it claims to be. */
    let dirty = git(repo, &["status", "--porcelain", "--untracked-files=no"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    if dirty {
        format!("{sha}-dirty")
    } else {
        sha
    }
}


// ── f16 compaction of the TinyMlp weight blobs (AUDIT-2026-11-08 §1) ─────
// The container is a 7 x u32 little-endian header (magic "SML1", version,
// in_dim, h1, h2, params, crc32) followed by params f32 weights. Version 2
// stores the same weights as IEEE half precision: half the bytes, and the five
// advisory gates are unaffected at a 10-bit mantissa. Conversion happens here
// so the committed .f32 files stay the source of truth for the tests.
const ML_BLOB_HEADER: usize = 7 * 4;
const ML_BLOB_MAGIC: u32 = u32::from_le_bytes(*b"SML1");
const ML_BLOB_VERSION: u32 = 1;
const ML_BLOB_VERSION_F16: u32 = 2;

fn blob_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Round-to-nearest-even f32 -> f16. The trainer emits no NaNs; from_blob
/// rejects one if it ever appeared.
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let up = rem > halfway || (rem == halfway && (half & 1) == 1);
        return sign | half | up as u16;
    }
    let mut half = ((e as u16) << 10) | (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1;
    }
    sign | half
}

/// Rewrite one TinyMlp blob with f16 weights. None when the input is not a
/// version-1 TinyMlp blob (the caller then refuses to stage it).
fn f16_blob(raw: &[u8]) -> Option<Vec<u8>> {
    if raw.len() < ML_BLOB_HEADER {
        return None;
    }
    let mut hdr = [0u32; 7];
    for (i, h) in hdr.iter_mut().enumerate() {
        *h = u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().ok()?);
    }
    let [magic, version, in_dim, h1, h2, params, crc] = hdr;
    if magic != ML_BLOB_MAGIC || version != ML_BLOB_VERSION {
        return None;
    }
    let body = &raw[ML_BLOB_HEADER..];
    if body.len() != params as usize * 4 || blob_crc32(body) != crc {
        return None;
    }
    let mut weights = Vec::with_capacity(body.len() / 2);
    for c in body.chunks_exact(4) {
        let v = f32::from_le_bytes(c.try_into().ok()?);
        weights.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    let mut out = Vec::with_capacity(ML_BLOB_HEADER + weights.len());
    for v in [
        ML_BLOB_MAGIC,
        ML_BLOB_VERSION_F16,
        in_dim,
        h1,
        h2,
        params,
        blob_crc32(&weights),
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&weights);
    Some(out)
}

fn main() {
    // Declare the cfg so the `unexpected_cfgs` lint stays quiet in both
    // configurations (see Cargo's check-cfg docs).
    println!("cargo:rustc-check-cfg=cfg(has_qtsq)");
    // The whitelisted experiment widths are emitted as cfgs below; declare
    // them so setting SOFUU_EMB_H doesn't trip the unexpected_cfgs lint.
    println!("cargo:rustc-check-cfg=cfg(emb_h_32,emb_h_36,emb_h_48,emb_h_64,emb_h_96,emb_h_128)");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_windows = target_os == "windows";
    // R-22 (AUDIT-2026-10-29): the probe used to end in a hardcoded absolute
    // developer checkout path, which made the source non-portable and put an
    // individual engineer's home directory in the repository. Resolution is
    // now, in order:
    //   1. SOFUU_QTSQ_DIR (the documented, CI-safe mechanism);
    //   2. $HOME/projects/black-hole-disk — the same checkout expressed
    //      relatively, so a developer box keeps working with no setup while
    //      no absolute username path is compiled into the tree;
    //   3. a vendored deps/qtsq inside this repository, if one is added later.
    // A missing library compiles the memory module out exactly as before.
    //
    // The `no-qtsq` marker (repo-wide convention — scripts/dist/ios.sh): a
    // SOFUU_QTSQ_DIR whose basename is `no-qtsq` is an EXPLICIT codec-less
    // build. Honor it before anything else — the marker directory is
    // deliberately empty, and the fallbacks below must never drag a host
    // QTSQ into an archive that is documented to ship without the codec.
    let qtsq_explicit = env::var("SOFUU_QTSQ_DIR").ok().map(PathBuf::from);
    let no_qtsq_marker = qtsq_explicit
        .as_ref()
        .and_then(|p| p.file_name().map(|n| n == "no-qtsq"))
        .unwrap_or(false);
    let qtsq_dir = if no_qtsq_marker {
        qtsq_explicit.clone()
    } else {
        qtsq_explicit
            .filter(|p| p.join("libqtsq.a").exists())
            .or_else(|| {
                let home = env::var("HOME").ok()?;
                let candidate = PathBuf::from(home).join("projects").join("black-hole-disk");
                if candidate.join("libqtsq.a").exists() {
                    Some(candidate)
                } else {
                    None
                }
            })
            .or_else(|| {
                let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
                let candidate = manifest.join("../../deps/qtsq");
                if candidate.join("libqtsq.a").exists() {
                    Some(candidate)
                } else {
                    None
                }
            })
    };
    // P3 (AUDIT-2026-09-07): the probe used to look for libqtsq.a on every
    // host while sofuu-ffi/build.rs links qtsq.lib (or SOFUU_QTSQ_LIB) on
    // Windows — the two crates' has_qtsq cfgs could disagree on a Windows
    // MSVC layout, compiling the memory module out while the FFI crate
    // still links QTSQ. Mirror the FFI resolution exactly.
    let qtsq_lib = match &qtsq_dir {
        Some(dir) if is_windows => env::var("SOFUU_QTSQ_LIB")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dir.join("windows/build-win64/Release/qtsq.lib")),
        Some(dir) => dir.join("libqtsq.a"),
        None => PathBuf::new(),
    };
    let has_qtsq = !no_qtsq_marker && qtsq_dir.is_some() && qtsq_lib.exists();
    // B7 (ROADMAP-BRAIN-CMA): watch the probed paths on EVERY outcome —
    // emitting them only on success let a failed probe cache forever (the
    // codec owner rebuilds libqtsq.a, cargo never re-probes, and the build
    // stays silently memory-less until a manual `touch build.rs`).
    if let Some(dir) = &qtsq_dir {
        println!("cargo:rerun-if-changed={}", qtsq_lib.display());
        println!("cargo:rerun-if-changed={}", dir.join("include/qtsq_format.h").display());
        println!("cargo:rerun-if-changed={}", dir.join("include/qtsq.h").display());
    }
    if has_qtsq {
        println!("cargo:rustc-cfg=has_qtsq");
    } else if no_qtsq_marker {
        // Explicit codec-less build (dist/no-qtsq marker): compile the memory
        // surface out and proceed. iOS SDK packs ship this shape by design.
        println!("cargo:warning=sofuu-core: codec-less build (SOFUU_QTSQ_DIR is the no-qtsq marker)");
    } else if let Some(dir) = &qtsq_dir {
        panic!(
            "{} not found at {} — QTSQ is a mandatory core component of Sofuu. \
             Set SOFUU_QTSQ_DIR to a built QTSQ checkout (SOFUU_QTSQ_LIB overrides the archive path on Windows); \
             CONTRIBUTING.md \"Building & Testing Locally\" lists what building that checkout needs.",
            qtsq_lib.file_name().and_then(|n| n.to_str()).unwrap_or("QTSQ archive"),
            dir.display()
        );
    } else {
        panic!(
            "SOFUU_QTSQ_DIR not set — QTSQ is a mandatory core component of Sofuu. \
             Set it to a built QTSQ checkout; CONTRIBUTING.md \"Building & Testing Locally\" \
             lists the rest of the prerequisites (zlib headers for QTSQ, libcurl/libssl for the link)."
        );
    }
    println!("cargo:rerun-if-env-changed=SOFUU_QTSQ_DIR");
    println!("cargo:rerun-if-env-changed=SOFUU_QTSQ_LIB");
    // hidden-width experiments: whitelist + emit emb_h_* cfgs
    match std::env::var("SOFUU_EMB_H").as_deref() {
        Ok("32") => println!("cargo:rustc-cfg=emb_h_32"),
        Ok("36") => println!("cargo:rustc-cfg=emb_h_36"),
        Ok("48") => println!("cargo:rustc-cfg=emb_h_48"),
        Ok("64") => println!("cargo:rustc-cfg=emb_h_64"),
        Ok("96") => println!("cargo:rustc-cfg=emb_h_96"),
        Ok("128") => println!("cargo:rustc-cfg=emb_h_128"),
        // P3 (AUDIT-2026-09-07): anything else used to be swallowed — a typo
        // like SOFUU_EMB_H=16 silently built the default width and the
        // experiment never happened. Make the rejection visible.
        Ok(other) => println!(
            "cargo:warning=SOFUU_EMB_H={other} is not a whitelisted width \
             (32/36/48/64/96/128) — building with the default H. \
             See the ml embedding model for the supported set."
        ),
        Err(_) => {}
    }
    println!("cargo:rerun-if-env-changed=SOFUU_EMB_H");
    // Prefill Phase 3: AMX via Accelerate's cblas_sgemm (macOS only; the NEON
    // tile path stays as fallback and Windows/Linux never link the framework).
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    println!("cargo:rerun-if-changed=build.rs");
    // M1: stage the trained image-projector artifact (weights_img.sem) into
    // OUT_DIR for include_bytes!. Absent until first training → stage an
    // empty blob; the runtime treats it as model-unavailable (tested), never
    // as silent garbage. Rebuilds when the trainer refreshes the artifact.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let staged = out_dir.join("weights_img.sem");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let baked = manifest.join("src/embedding/weights_img.sem");
    println!("cargo:rerun-if-changed={}", baked.display());
    let bytes = std::fs::read(&baked).unwrap_or_default();
    if bytes.is_empty() && baked.exists() {
        println!("cargo:warning=weights_img.sem is empty — image embeddings report model-unavailable until trained");
    }
    let _ = std::fs::write(&staged, bytes);

    // ── Watch all JS source files and driver snapshots in src/js ─────────
    let js_dir = manifest.join("../../src/js");
    println!("cargo:rerun-if-changed={}", js_dir.display());
    if let Ok(entries) = std::fs::read_dir(&js_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if ext == "js" || ext == "bc" || ext == "gz" {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }

    // ── AUDIT-2026-10-31 P1-8: shipped JS assets, embedded gzipped ───────
    // These files are read as text at runtime but only on one code path each
    // (engine compat seam, RLM driver, chat TUI). Embedded raw they were
    // 271 KB of the binary's .rodata against a 5.5 MB cap; gzip-9 is ~4x
    // smaller, and the inflate happens where the text is first needed
    // (shipped::inflate_gz, cached for the life of the process). The .js
    // files stay the single source of truth: the compression runs here, so an
    // edit can never ship stale bytes and there is no committed .gz to forget
    // to regenerate.
    // memory_api.js (24,449 B -> 8,373) and memory_adapters.js (34,361 ->
    // 9,525) are the memory surface's two largest .rodata items; both are
    // inflated once, on the path that needs them (memory_register / the first
    // non-native backend open), so they ride the same mechanism.
    for asset in [
        "compat.js",
        "rlm.js",
        "chat_tui.js",
        "memory_api.js",
        "memory_adapters.js",
        /* PLAN-DRONES §4.6: the drone surface and skill layer are embedded but
         * read on exactly one path (a policy that enables `hardware`), so they
         * ride the same gzip mechanism as the other single-path drivers. */
        "mavlink.js",
        "drone.js",
    ] {
        use std::io::Write as _;
        let path = manifest.join("../../src/js").join(asset);
        println!("cargo:rerun-if-changed={}", path.display());
        let src = read_stable(&path).unwrap_or_else(|e| {
            panic!("src/js/{asset} must exist (embedded by the binary): {e}")
        });
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        enc.write_all(&src).expect("gzip shipped JS asset");
        let gz = enc.finish().expect("finish gzip shipped JS asset");
        std::fs::write(out_dir.join(format!("{asset}.gz")), &gz)
            .expect("stage gzipped shipped JS asset");
    }

    // ── Learned-model weight blobs, embedded gzipped ─────────────────────
    // ~320 KB of committed weights sat raw in .rodata. They compress poorly
    // (5% for the f32 nets, 21% for the .sem projectors), so each blob is
    // gzipped here and inflated once, lazily, behind the OnceLock that already
    // guards its model. The five TinyMlp nets are additionally stored as f16
    // (see f16_blob below): halving 250 KB of weights bought back more than
    // the 5.5 MiB cap could absorb after the 2026-11 audit remediation, and a
    // 10-bit mantissa is invisible to advisory gates. Five modules ship a file
    // called weights_v2.f32, so the staged names are made unique.
    for (src, dst) in [
        ("src/ml/relevance/weights_v2.f32", "ml_relevance_weights_v2.f32.gz"),
        ("src/ml/supervisor/weights_v2.f32", "ml_supervisor_weights_v2.f32.gz"),
        ("src/ml/compaction/weights_v2.f32", "ml_compaction_weights_v2.f32.gz"),
        ("src/ml/freshness/weights_v2.f32", "ml_freshness_weights_v2.f32.gz"),
        ("src/ml/alloc/weights_v2.f32", "ml_alloc_weights_v2.f32.gz"),
        ("src/embedding/weights_v1.sem", "embedding_weights_v1.sem.gz"),
        ("src/embedding/weights_v2.sem", "embedding_weights_v2.sem.gz"),
    ] {
        use std::io::Write as _;
        let path = manifest.join(src); // src/ lives inside the crate
        println!("cargo:rerun-if-changed={}", path.display());
        let raw = read_stable(&path).unwrap_or_else(|e| {
            panic!("{src} must exist (embedded by the binary): {e}")
        });
        /* The five TinyMlp nets ship as v2 (f16) blobs; the .sem projectors
         * keep their own format and are staged as-is. */
        let raw = if src.ends_with("weights_v2.f32") {
            f16_blob(&raw).unwrap_or_else(|| {
                panic!("{src} must be a version-1 TinyMlp blob with a valid CRC")
            })
        } else {
            raw
        };
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        enc.write_all(&raw).expect("gzip embedded weights");
        let gz = enc.finish().expect("finish gzip embedded weights");
        std::fs::write(out_dir.join(dst), &gz).expect("stage gzipped weights");
    }

    // Gzip matrix.json (contract matrix: 18 KB -> 2.5 KB)
    {
        use std::io::Write as _;
        let path = manifest.join("../../contract/matrix.json");
        println!("cargo:rerun-if-changed={}", path.display());
        let src = read_stable(&path).unwrap_or_else(|e| {
            panic!("contract/matrix.json must exist (embedded by capabilities.rs): {e}")
        });
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        enc.write_all(&src).expect("gzip contract/matrix.json");
        let gz = enc.finish().expect("finish gzip contract/matrix.json");
        std::fs::write(out_dir.join("matrix.json.gz"), &gz)
            .expect("stage gzipped contract/matrix.json");
    }

    // Gzip rlm_bootstrap.js (7.5 KB -> 2.1 KB)
    {
        use std::io::Write as _;
        let path = manifest.join("js/rlm_bootstrap.js");
        println!("cargo:rerun-if-changed={}", path.display());
        let src = read_stable(&path).unwrap_or_else(|e| {
            panic!("js/rlm_bootstrap.js must exist: {e}")
        });
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        enc.write_all(&src).expect("gzip rlm_bootstrap.js");
        let gz = enc.finish().expect("finish gzip rlm_bootstrap.js");
        std::fs::write(out_dir.join("rlm_bootstrap.js.gz"), &gz)
            .expect("stage gzipped rlm_bootstrap.js");
    }

    // ── PC-07: stamp the source revision ─────────────────────────────────
    let root = manifest.canonicalize().unwrap_or_else(|_| manifest.clone());
    // CARGO_MANIFEST_DIR is <repo>/crates/sofuu-core.
    let repo = root
        .parent()
        .and_then(Path::parent)
        .unwrap_or(&root)
        .to_path_buf();
    println!("cargo:rustc-env=SOFUU_BUILD={}", build_stamp(&repo));
    println!("cargo:rerun-if-env-changed=SOFUU_BUILD");
    // The stamp is only as fresh as the last build-script run, so watch what
    // git moves: HEAD on checkout, the branch ref on commit, the index on add.
    for f in ["HEAD", "index"] {
        let p = repo.join(".git").join(f);
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
    if let Some(reference) = git(&repo, &["symbolic-ref", "-q", "HEAD"]) {
        let p = repo.join(".git").join(reference);
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
}
