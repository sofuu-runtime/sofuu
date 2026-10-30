// brain_cmd.rs — `sofuu brain <cmd>`: operational tooling for a brain store
// (QTSQ_UPGRADE_PLAN W7.1–W7.5).
//
// The dispatcher in main.rs had no `brain` arm: the only brain-adjacent
// surface was `sofuu serve --brain <path>`. The storage engine, however, was
// already written — `rt/brain_store.rs` owns the on-disk format and
// `rt/memory.rs:repair` already says "this report is what a CLI
// (`sofuu brain repair <path>`) should print". This module is that CLI. It
// contains NO storage logic of its own: every read, digest, quarantine and
// manifest publish goes through the engine's own public primitives, so a
// CLI verdict and an in-process open can never disagree.
//
// The layout it operates on (rt/brain_store.rs is the source of truth):
//
//   <path>                            manifest container ("manifest" JSON stream)
//   <path>.d/seg-<epoch>-<seq>.qtsq   immutable sealed segments
//
// A brain written before B3 (the "v1 format") is ONE flat container holding a
// "memories" tensor plus a "metadata" JSON stream; `migrate` converts that
// through the runtime's own open→flush path (never by rewriting bytes here).
//
// Exit codes are part of the contract:
//   0  the operation succeeded and the store is (or now is) intact
//   1  damage or record loss was detected — including when the repair
//      succeeded, because the operator still needs to know what was lost
//   2  usage error: bad arguments, missing path, non-empty destination
//
// `--json` prints the SAME report the human renderer prints, never a
// different one: the JSON is built from the report struct and the text lines
// are derived from that struct too.

use std::path::Path;

/* ------------------------------------------------------------------ */
/* Exit codes + typed errors                                           */
/* ------------------------------------------------------------------ */

pub const EXIT_OK: u8 = 0;
pub const EXIT_DAMAGE: u8 = 1;
pub const EXIT_USAGE: u8 = 2;

/// A CLI failure with a STABLE code.
///
/// The runtime's error taxonomy lives in `rt::async_contract` for JS-side
/// contracts; the CLI has its own small set, and every one of them is
/// prefixed `sofuu.brain.` so a script can branch on `error.code` exactly
/// like it branches on the exit code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainError {
    pub code: &'static str,
    pub message: String,
    pub exit: u8,
}

impl BrainError {
    pub fn new(code: &'static str, message: impl Into<String>, exit: u8) -> Self {
        Self {
            code,
            message: message.into(),
            exit,
        }
    }

    fn usage(message: impl Into<String>) -> Self {
        Self::new("sofuu.brain.usage", message, EXIT_USAGE)
    }

    fn missing(path: &str) -> Self {
        Self::new(
            "sofuu.brain.missing",
            format!("{path}: no such brain file"),
            EXIT_USAGE,
        )
    }

    fn foreign(path: &str, why: impl std::fmt::Display) -> Self {
        Self::new(
            "sofuu.brain.foreign",
            format!("{path}: not a brain this runtime can read ({why})"),
            EXIT_DAMAGE,
        )
    }

    /* Used only by the codec-less wrappers below; a codec-present build
     * still needs the method to exist for those `cfg` arms to compile. */
    #[allow(dead_code)]
    fn codec_missing() -> Self {
        Self::new(
            "sofuu.brain.codec_missing",
            "this build has no QTSQ codec (sofuu_core::HAS_QTSQ == false) — \
             brain stores cannot be read or written by it",
            EXIT_DAMAGE,
        )
    }

    fn io(path: &str, what: &str, e: impl std::fmt::Display) -> Self {
        Self::new(
            "sofuu.brain.io",
            format!("{path}: {what}: {e}"),
            EXIT_DAMAGE,
        )
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "code": self.code, "message": self.message })
    }
}

impl std::fmt::Display for BrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for BrainError {}

fn emit_error(e: &BrainError) {
    eprintln!("\x1b[31mError:\x1b[0m {e}");
}

/* ------------------------------------------------------------------ */
/* Reports                                                             */
/* ------------------------------------------------------------------ */

/// Result of `sofuu brain verify`. `ok == false` always means exit 1.
#[derive(Clone, Debug, Default)]
pub struct VerifyReport {
    pub path: String,
    /// "segmented" (current), "flat-v1" (legacy single file), "foreign".
    pub format: String,
    pub schema: String,
    pub ok: bool,
    /// Distinct record ids found across every segment (the row count).
    pub records: usize,
    /// Records that are not tombstones.
    pub live: usize,
    /// Records whose tier is `TIER_FORGOTTEN` (a real tombstone, not a loss).
    pub tombstones: usize,
    /// Segment entries the manifest references.
    pub segments: usize,
    pub verified_segments: usize,
    pub damaged_segments: usize,
    pub missing_files: Vec<String>,
    pub errors: Vec<String>,
    /// Informational lines that are not failures.
    pub notes: Vec<String>,
    pub bytes: u64,
    /// Manifest generation (bumped by every compaction).
    pub generation: u64,
    pub embedding_id: String,
    pub embedding_dim: u64,
}

impl VerifyReport {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "command": "verify",
            "path": self.path,
            "ok": self.ok,
            "format": self.format,
            "schema": self.schema,
            "records": self.records,
            "live": self.live,
            "tombstones": self.tombstones,
            "segments": self.segments,
            "verified_segments": self.verified_segments,
            "damaged_segments": self.damaged_segments,
            "missing_files": self.missing_files,
            "errors": self.errors,
            "notes": self.notes,
            "bytes": self.bytes,
            "generation": self.generation,
            "embedding": {
                "id": self.embedding_id,
                "dimension": self.embedding_dim,
            },
        })
    }
}

/// Result of `sofuu brain repair` — the engine's own `RepairReport` plus the
/// CLI's verdict on it.
#[derive(Clone, Debug, Default)]
pub struct RepairOutcome {
    pub path: String,
    pub ok: bool,
    /// The store is a segmented one this engine recognised.
    pub segmented: bool,
    pub records: usize,
    pub segments: usize,
    pub quarantined: Vec<String>,
    pub adopted: Vec<String>,
    pub missing: Vec<String>,
    pub records_dropped: usize,
    pub rewritten: bool,
    /// Anything was lost: a quarantine, a missing file, or dropped versions.
    pub lost: bool,
    pub errors: Vec<String>,
    pub message: String,
}

impl RepairOutcome {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "command": "repair",
            "path": self.path,
            "ok": self.ok,
            "segmented": self.segmented,
            /* The engine's report, verbatim — the same fields `rt::memory`
             * hands an in-process caller. */
            "found": self.segmented,
            "records": self.records,
            "segments": self.segments,
            "quarantined": self.quarantined,
            "adopted": self.adopted,
            "missing": self.missing,
            "records_dropped": self.records_dropped,
            "rewritten": self.rewritten,
            "lost": self.lost,
            "errors": self.errors,
            "message": self.message,
        })
    }
}

/// Result of `sofuu brain stats`.
#[derive(Clone, Debug, Default)]
pub struct StatsReport {
    pub path: String,
    pub format: String,
    pub schema: String,
    pub ok: bool,
    pub records: usize,
    pub live: usize,
    pub tombstones: usize,
    /// Record VERSIONS on disk (append segments can hold superseded ones).
    pub versions: usize,
    pub segments: usize,
    /// One entry per generation present in the manifest, oldest first.
    pub generations: Vec<(u64, usize, usize)>, // (epoch, segments, records)
    pub generation: u64,
    /// Manifest + referenced segment bytes.
    pub bytes: u64,
    /// Everything under `<path>.d` plus the manifest, orphans included.
    pub disk_bytes: u64,
    pub manifest_bytes: u64,
    pub evicted_total: u64,
    pub last_decay_at: u64,
    /// Recovery counters the store really keeps: quarantined (`*.torn`)
    /// files, unreferenced segments left by a crashed compaction, missing
    /// referenced segments, damaged referenced segments.
    pub quarantined_files: usize,
    pub orphan_segments: usize,
    pub missing_segments: usize,
    pub damaged_segments: usize,
    pub embedding_id: String,
    pub embedding_dim: u64,
    pub errors: Vec<String>,
}

impl StatsReport {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "command": "stats",
            "path": self.path,
            "ok": self.ok,
            "format": self.format,
            "schema": self.schema,
            "records": self.records,
            "live": self.live,
            "tombstones": self.tombstones,
            "versions": self.versions,
            "segments": self.segments,
            "generations": self.generations.iter().map(|(e, s, r)| {
                serde_json::json!({ "generation": e, "segments": s, "records": r })
            }).collect::<Vec<_>>(),
            "generation": self.generation,
            "bytes": self.bytes,
            "disk_bytes": self.disk_bytes,
            "manifest_bytes": self.manifest_bytes,
            "evicted_total": self.evicted_total,
            "last_decay_at": self.last_decay_at,
            "recovery": {
                "quarantined_files": self.quarantined_files,
                "orphan_segments": self.orphan_segments,
                "missing_segments": self.missing_segments,
                "damaged_segments": self.damaged_segments,
            },
            "embedding": {
                "id": self.embedding_id,
                "dimension": self.embedding_dim,
            },
            "errors": self.errors,
        })
    }
}

/// Result of `sofuu brain backup` / `sofuu brain restore`.
#[derive(Clone, Debug, Default)]
pub struct SnapshotReport {
    pub command: String,
    pub ok: bool,
    pub src: String,
    pub dst: String,
    pub format: String,
    /// Files written into the destination's store directory.
    pub files: usize,
    /// Segment files hardlinked (inode shared with the source) — backup only.
    pub linked: usize,
    /// Files byte-copied (the manifest is always one of them).
    pub copied: usize,
    /// Linked files whose inode was read back and matched the source's.
    pub inode_verified: usize,
    pub bytes: u64,
    pub manifest_published: bool,
    pub source_ok: bool,
    pub destination_ok: bool,
    pub destination_records: usize,
    pub destination_segments: usize,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl SnapshotReport {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "command": self.command,
            "ok": self.ok,
            "src": self.src,
            "dst": self.dst,
            "format": self.format,
            "files": self.files,
            "linked": self.linked,
            "copied": self.copied,
            "inode_verified": self.inode_verified,
            "bytes": self.bytes,
            "manifest_published": self.manifest_published,
            "source_ok": self.source_ok,
            "destination_ok": self.destination_ok,
            "destination_records": self.destination_records,
            "destination_segments": self.destination_segments,
            "errors": self.errors,
            "warnings": self.warnings,
        })
    }
}

/// Result of `sofuu brain migrate`.
#[derive(Clone, Debug, Default)]
pub struct MigrateReport {
    pub ok: bool,
    pub migrated: bool,
    pub src: String,
    pub from_format: String,
    pub to_format: String,
    pub from_dimension: usize,
    pub from_records: usize,
    pub space: String,
    pub records_imported: usize,
    pub segments: usize,
    pub v1backup: String,
    pub message: String,
    pub errors: Vec<String>,
}

impl MigrateReport {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "command": "migrate",
            "ok": self.ok,
            "migrated": self.migrated,
            "src": self.src,
            "from": {
                "format": self.from_format,
                "dimension": self.from_dimension,
                "records": self.from_records,
                "space": self.space,
            },
            "to": { "format": self.to_format },
            "records_imported": self.records_imported,
            "segments": self.segments,
            "v1backup": self.v1backup,
            "message": self.message,
            "errors": self.errors,
        })
    }
}

/* ------------------------------------------------------------------ */
/* Entry point + argument parsing                                      */
/* ------------------------------------------------------------------ */

const SUBCOMMANDS: &[&str] = &[
    "verify", "repair", "stats", "backup", "restore", "migrate",
];

/// Run `sofuu brain …`. `args` is the WHOLE argv (argv[0] is the binary).
///
/// `eval` runs a JavaScript expression in an initialised runtime and returns
/// the string the script parked on `globalThis.__brainReport`; only
/// `migrate` needs it (the runtime's own open→flush migration path lives in
/// the JS surface). Tests pass a closure that fails, which is exactly the
/// "no engine" behaviour.
pub fn run(args: &[String], eval: &mut dyn FnMut(&str) -> Result<String, String>) -> u8 {
    let Some(sub) = args.get(2).map(String::as_str) else {
        print_help();
        return EXIT_USAGE;
    };
    if sub == "help" || sub == "-h" || sub == "--help" {
        print_help();
        return EXIT_OK;
    }
    let rest: Vec<String> = args[3..].to_vec();
    let wants_help = rest.iter().any(|a| a == "-h" || a == "--help");
    if !SUBCOMMANDS.contains(&sub) {
        eprintln!(
            "\x1b[31mError:\x1b[0m [sofuu.brain.usage] unknown subcommand 'brain {sub}'"
        );
        print_help();
        return EXIT_USAGE;
    }
    if wants_help {
        print_sub_help(sub);
        return EXIT_OK;
    }

    let mut json = false;
    let mut allow_damaged = false;
    let mut positional: Vec<String> = Vec::new();
    for a in &rest {
        match a.as_str() {
            "--json" => json = true,
            "--allow-damaged" => allow_damaged = true,
            other if other.starts_with('-') => {
                eprintln!(
                    "\x1b[31mError:\x1b[0m [sofuu.brain.usage] unknown flag '{other}' for 'brain {sub}'"
                );
                print_sub_help(sub);
                return EXIT_USAGE;
            }
            other => positional.push(other.to_string()),
        }
    }

    let want_args = if sub == "backup" || sub == "restore" {
        2
    } else {
        1
    };
    if positional.len() != want_args {
        eprintln!(
            "\x1b[31mError:\x1b[0m [sofuu.brain.usage] 'brain {sub}' takes {want_args} path argument(s), got {}",
            positional.len()
        );
        print_sub_help(sub);
        return EXIT_USAGE;
    }

    match sub {
        "verify" => cmd_verify(&positional[0], json),
        "repair" => cmd_repair(&positional[0], json),
        "stats" => cmd_stats(&positional[0], json),
        "backup" => cmd_backup(&positional[0], &positional[1], json, allow_damaged),
        "restore" => cmd_restore(&positional[0], &positional[1], json, allow_damaged),
        "migrate" => cmd_migrate(&positional[0], json, eval),
        _ => EXIT_USAGE, // unreachable: SUBCOMMANDS is the match above
    }
}

/* ------------------------------------------------------------------ */
/* verify                                                              */
/* ------------------------------------------------------------------ */

fn cmd_verify(path: &str, json: bool) -> u8 {
    match verify_report(path) {
        Ok(r) => {
            if json {
                print_json(&r.to_json());
            } else {
                render_verify(&r);
            }
            if r.ok {
                EXIT_OK
            } else {
                EXIT_DAMAGE
            }
        }
        Err(e) => {
            if json {
                print_json(&serde_json::json!({
                    "command": "verify",
                    "path": path,
                    "ok": false,
                    "errors": [e.message],
                    "error": e.to_json(),
                }));
            }
            emit_error(&e);
            e.exit
        }
    }
}

fn render_verify(r: &VerifyReport) {
    println!("\x1b[1mbrain verify\x1b[0m {}", r.path);
    println!("  {:<11}{}", "format", describe_format(&r.format, &r.schema, r.generation));
    println!(
        "  {:<11}{} record(s) — {} live, {} tombstone(s)",
        "records", r.records, r.live, r.tombstones
    );
    println!(
        "  {:<11}{} ({}) — {} byte(s) in segments",
        "segments",
        r.segments,
        if r.damaged_segments == 0 {
            format!("{} verified", r.verified_segments)
        } else {
            format!("{} verified, {} DAMAGED", r.verified_segments, r.damaged_segments)
        },
        r.bytes
    );
    if !r.embedding_id.is_empty() || r.embedding_dim > 0 {
        println!(
            "  {:<11}{} ({} dim)",
            "space",
            if r.embedding_id.is_empty() { "?" } else { &r.embedding_id },
            r.embedding_dim
        );
    }
    for n in &r.notes {
        println!("  \x1b[90m· {n}\x1b[0m");
    }
    for m in &r.missing_files {
        println!("  \x1b[31m✗ missing\x1b[0m {m}");
    }
    for e in &r.errors {
        println!("  \x1b[31m✗ {e}\x1b[0m");
    }
    if r.ok {
        println!("  \x1b[32m✓ ok\x1b[0m");
    } else {
        println!(
            "  \x1b[31m✗ NOT OK\x1b[0m — {} error(s), {} missing file(s), {} damaged segment(s)",
            r.errors.len(),
            r.missing_files.len(),
            r.damaged_segments
        );
    }
}

fn describe_format(format: &str, schema: &str, generation: u64) -> String {
    match format {
        "segmented" => format!("segmented ({schema}), generation {generation}"),
        "flat-v1" => "flat v1 (single-file container, pre-segmentation)".to_string(),
        other => other.to_string(),
    }
}

/* ------------------------------------------------------------------ */
/* repair                                                              */
/* ------------------------------------------------------------------ */

fn cmd_repair(path: &str, json: bool) -> u8 {
    match repair_outcome(path) {
        Ok(r) => {
            if json {
                print_json(&r.to_json());
            } else {
                println!("\x1b[1mbrain repair\x1b[0m {}", r.path);
                println!("  {:<11}{}", "message", r.message);
                if r.segmented {
                    println!(
                        "  {:<11}{} record(s) in {} segment(s)",
                        "store", r.records, r.segments
                    );
                    println!(
                        "  {:<11}{} quarantined, {} adopted, {} missing, {} version(s) dropped",
                        "recovery",
                        r.quarantined.len(),
                        r.adopted.len(),
                        r.missing.len(),
                        r.records_dropped
                    );
                    for q in &r.quarantined {
                        println!("  \x1b[33m⌁ quarantined\x1b[0m {q}");
                    }
                    for a in &r.adopted {
                        println!("  \x1b[33m⌁ adopted\x1b[0m {a} (manifest digest corrected)");
                    }
                    for m in &r.missing {
                        println!("  \x1b[31m✗ missing\x1b[0m {m}");
                    }
                    println!(
                        "  {:<11}{}",
                        "rewritten",
                        if r.rewritten { "yes" } else { "no" }
                    );
                }
                for e in &r.errors {
                    println!("  \x1b[31m✗ {e}\x1b[0m");
                }
                if !r.ok {
                    println!("  \x1b[31m✗ NOT OK\x1b[0m");
                } else if r.lost {
                    println!(
                        "  \x1b[33m⚠ repaired with loss\x1b[0m — {} record version(s) dropped; \
                         the damaged bytes are beside the store as *.torn",
                        r.records_dropped
                    );
                } else {
                    println!("  \x1b[32m✓ ok\x1b[0m — nothing was lost");
                }
            }
            if r.ok && !r.lost {
                EXIT_OK
            } else {
                EXIT_DAMAGE
            }
        }
        Err(e) => {
            if json {
                print_json(&serde_json::json!({
                    "command": "repair",
                    "path": path,
                    "ok": false,
                    "errors": [e.message],
                    "error": e.to_json(),
                }));
            }
            emit_error(&e);
            e.exit
        }
    }
}

/* ------------------------------------------------------------------ */
/* stats                                                               */
/* ------------------------------------------------------------------ */

fn cmd_stats(path: &str, json: bool) -> u8 {
    match stats_report(path) {
        Ok(r) => {
            if json {
                print_json(&r.to_json());
            } else {
                println!("\x1b[1mbrain stats\x1b[0m {}", r.path);
                println!("  {:<14}{}", "format", describe_format(&r.format, &r.schema, r.generation));
                println!(
                    "  {:<14}{} record(s) — {} live, {} tombstone(s), {} version(s) on disk",
                    "records", r.records, r.live, r.tombstones, r.versions
                );
                println!(
                    "  {:<14}{} segment(s) across {} generation(s)",
                    "segments",
                    r.segments,
                    r.generations.len()
                );
                for (epoch, segs, recs) in &r.generations {
                    println!("  {:<14}generation {epoch}: {segs} segment(s), {recs} version(s)", "");
                }
                println!("  {:<14}{} byte(s) (store), {} byte(s) on disk", "bytes", r.bytes, r.disk_bytes);
                println!("  {:<14}{} byte(s)", "manifest", r.manifest_bytes);
                println!(
                    "  {:<14}{} capacity eviction(s), last decay at {}",
                    "recovery", r.evicted_total, r.last_decay_at
                );
                println!(
                    "  {:<14}{} quarantined (*.torn), {} orphan, {} missing, {} damaged",
                    "damage",
                    r.quarantined_files,
                    r.orphan_segments,
                    r.missing_segments,
                    r.damaged_segments
                );
                if !r.embedding_id.is_empty() || r.embedding_dim > 0 {
                    println!(
                        "  {:<14}{} ({} dim)",
                        "space",
                        if r.embedding_id.is_empty() { "?" } else { &r.embedding_id },
                        r.embedding_dim
                    );
                }
                for e in &r.errors {
                    println!("  \x1b[31m✗ {e}\x1b[0m");
                }
                if r.ok {
                    println!("  \x1b[32m✓ ok\x1b[0m");
                } else {
                    println!("  \x1b[31m✗ DAMAGED\x1b[0m — the counters above exclude what is missing");
                }
            }
            if r.ok {
                EXIT_OK
            } else {
                EXIT_DAMAGE
            }
        }
        Err(e) => {
            if json {
                print_json(&serde_json::json!({
                    "command": "stats",
                    "path": path,
                    "ok": false,
                    "errors": [e.message],
                    "error": e.to_json(),
                }));
            }
            emit_error(&e);
            e.exit
        }
    }
}

/* ------------------------------------------------------------------ */
/* backup / restore                                                    */
/* ------------------------------------------------------------------ */

fn cmd_backup(src: &str, dst: &str, json: bool, allow_damaged: bool) -> u8 {
    match snapshot(src, dst, allow_damaged, true) {
        Ok(r) => {
            if json {
                print_json(&r.to_json());
            } else {
                render_snapshot(&r);
            }
            if r.ok {
                EXIT_OK
            } else {
                EXIT_DAMAGE
            }
        }
        Err(e) => {
            if json {
                print_json(&serde_json::json!({
                    "command": "backup",
                    "src": src,
                    "dst": dst,
                    "ok": false,
                    "errors": [e.message],
                    "error": e.to_json(),
                }));
            }
            emit_error(&e);
            e.exit
        }
    }
}

fn cmd_restore(src: &str, dst: &str, json: bool, allow_damaged: bool) -> u8 {
    match snapshot(src, dst, allow_damaged, false) {
        Ok(r) => {
            if json {
                print_json(&r.to_json());
            } else {
                render_snapshot(&r);
            }
            if r.ok {
                EXIT_OK
            } else {
                EXIT_DAMAGE
            }
        }
        Err(e) => {
            if json {
                print_json(&serde_json::json!({
                    "command": "restore",
                    "src": src,
                    "dst": dst,
                    "ok": false,
                    "errors": [e.message],
                    "error": e.to_json(),
                }));
            }
            emit_error(&e);
            e.exit
        }
    }
}

fn render_snapshot(r: &SnapshotReport) {
    println!("\x1b[1mbrain {}\x1b[0m {} → {}", r.command, r.src, r.dst);
    println!("  {:<13}{}", "format", r.format);
    println!(
        "  {:<13}{} file(s): {} hardlinked, {} copied, {} byte(s)",
        "files", r.files, r.linked, r.copied, r.bytes
    );
    if r.linked > 0 {
        println!(
            "  {:<13}{} of {} link(s) verified inode-equal to the source",
            "inodes", r.inode_verified, r.linked
        );
    }
    println!(
        "  {:<13}{} (published last: {})",
        "manifest",
        r.dst,
        if r.manifest_published { "yes" } else { "NO" }
    );
    println!(
        "  {:<13}source {}, destination {} — {} record(s), {} segment(s)",
        "verify",
        if r.source_ok { "ok" } else { "DAMAGED" },
        if r.destination_ok { "ok" } else { "DAMAGED" },
        r.destination_records,
        r.destination_segments
    );
    for w in &r.warnings {
        println!("  \x1b[33m⚠ {w}\x1b[0m");
    }
    for e in &r.errors {
        println!("  \x1b[31m✗ {e}\x1b[0m");
    }
    if r.ok {
        println!("  \x1b[32m✓ ok\x1b[0m");
    } else {
        println!("  \x1b[31m✗ NOT OK\x1b[0m");
    }
}

/* ------------------------------------------------------------------ */
/* migrate                                                             */
/* ------------------------------------------------------------------ */

fn cmd_migrate(path: &str, json: bool, eval: &mut dyn FnMut(&str) -> Result<String, String>) -> u8 {
    match migrate_store(path, eval) {
        Ok(r) => {
            if json {
                print_json(&r.to_json());
            } else {
                println!("\x1b[1mbrain migrate\x1b[0m {}", r.src);
                println!(
                    "  {:<13}{} → {}",
                    "format", r.from_format, r.to_format
                );
                if r.migrated {
                    println!(
                        "  {:<13}{} record(s) imported into {} segment(s)",
                        "records", r.records_imported, r.segments
                    );
                    println!("  {:<13}{}", "v1backup", r.v1backup);
                }
                if !r.message.is_empty() {
                    println!("  {:<13}{}", "message", r.message);
                }
                for e in &r.errors {
                    println!("  \x1b[31m✗ {e}\x1b[0m");
                }
                if r.ok {
                    println!("  \x1b[32m✓ ok\x1b[0m");
                } else {
                    println!("  \x1b[31m✗ NOT OK\x1b[0m");
                }
            }
            if r.ok {
                EXIT_OK
            } else {
                EXIT_DAMAGE
            }
        }
        Err(e) => {
            if json {
                print_json(&serde_json::json!({
                    "command": "migrate",
                    "src": path,
                    "ok": false,
                    "migrated": false,
                    "errors": [e.message],
                    "error": e.to_json(),
                }));
            }
            emit_error(&e);
            e.exit
        }
    }
}

fn print_json(v: &serde_json::Value) {
    match serde_json::to_string_pretty(v) {
        Ok(s) => println!("{s}"),
        Err(_) => println!("{v}"),
    }
}

/* ------------------------------------------------------------------ */
/* Engine-backed operations (codec-gated)                              */
/* ------------------------------------------------------------------ */

fn verify_report(path: &str) -> Result<VerifyReport, BrainError> {
    #[cfg(has_qtsq)]
    {
        store::verify(path)
    }
    #[cfg(not(has_qtsq))]
    {
        let _ = path;
        Err(BrainError::codec_missing())
    }
}

fn repair_outcome(path: &str) -> Result<RepairOutcome, BrainError> {
    #[cfg(has_qtsq)]
    {
        store::repair(path)
    }
    #[cfg(not(has_qtsq))]
    {
        let _ = path;
        Err(BrainError::codec_missing())
    }
}

fn stats_report(path: &str) -> Result<StatsReport, BrainError> {
    #[cfg(has_qtsq)]
    {
        store::stats(path)
    }
    #[cfg(not(has_qtsq))]
    {
        let _ = path;
        Err(BrainError::codec_missing())
    }
}

fn snapshot(
    src: &str,
    dst: &str,
    allow_damaged: bool,
    link: bool,
) -> Result<SnapshotReport, BrainError> {
    #[cfg(has_qtsq)]
    {
        store::snapshot(src, dst, allow_damaged, link)
    }
    #[cfg(not(has_qtsq))]
    {
        let _ = (src, dst, allow_damaged, link);
        Err(BrainError::codec_missing())
    }
}

fn migrate_store(
    path: &str,
    eval: &mut dyn FnMut(&str) -> Result<String, String>,
) -> Result<MigrateReport, BrainError> {
    #[cfg(has_qtsq)]
    {
        store::migrate(path, eval)
    }
    #[cfg(not(has_qtsq))]
    {
        let _ = (path, eval);
        Err(BrainError::codec_missing())
    }
}

/* ------------------------------------------------------------------ */
/* The store layer — everything below talks to the engine's own        */
/* primitives (rt::brain_store, rt::memory, rt::wal); nothing here      */
/* re-implements the format.                                           */
/* ------------------------------------------------------------------ */

#[cfg(has_qtsq)]
mod store {
    use super::*;
    use crate::rt::brain_store as bs;
    use crate::rt::wal;
    use std::collections::{BTreeMap, BTreeSet};

    /// What a path holds, decided by the engine's own readers.
    enum Fmt {
        Segmented(bs::Manifest),
        Flat(FlatInfo),
        Foreign(i32),
    }

    struct FlatInfo {
        dim: usize,
        records: usize,
        /// Embedding-space id from the container's metadata, when it has one.
        space: String,
        /// Dimension the metadata claims, when it has one.
        declared_dim: u64,
    }

    fn inspect(path: &str) -> Result<Fmt, BrainError> {
        let p = Path::new(path);
        if !p.exists() {
            return Err(BrainError::missing(path));
        }
        if !p.is_file() {
            return Err(BrainError::usage(format!("{path}: is not a file")));
        }
        unsafe {
            match bs::read_manifest_ex(path) {
                /* A container this key cannot open: fail loudly. Treating it
                 * as "absent" would report a wrong-key brain as an empty one. */
                Err(rc) => {
                    return Err(BrainError::new(
                        "sofuu.brain.unreadable",
                        format!(
                            "{path}: the container exists but this machine's brain key \
                             cannot open it (codec rc {rc})"
                        ),
                        EXIT_DAMAGE,
                    ))
                }
                Ok(Some((m, _legacy))) => return Ok(Fmt::Segmented(m)),
                Ok(None) => { /* legacy single file, or not a brain */ }
            }
            match bs::container_open_ex(path) {
                Ok((c, _legacy)) => {
                    // Legacy flat brain: "memories" tensor + "metadata" JSON.
                    let tensor = bs::read_tensor_stream(c, bs::SEGMENT_TENSOR_STREAM);
                    let meta = bs::read_json_stream(c, "metadata");
                    bs::container_free(c);
                    let Some((vals, dims)) = tensor else {
                        return Ok(Fmt::Foreign(bs::ERR_NOT_CONTAINER));
                    };
                    let (records, dim) = if dims.len() >= 2 {
                        (dims[0] as usize, dims[1] as usize)
                    } else {
                        (0, 0)
                    };
                    let _ = vals;
                    let mut space = String::new();
                    let mut declared_dim = 0u64;
                    if let Some(meta) = meta.as_deref() {
                        if let Ok(root) = serde_json::from_str::<serde_json::Value>(meta) {
                            if let Some(emb) = root.get("embedding") {
                                space = emb
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                declared_dim =
                                    emb.get("dimension").and_then(|v| v.as_u64()).unwrap_or(0);
                            }
                        }
                    }
                    if space.is_empty() && dim == crate::embedding::HASH_DIM {
                        /* The retired shell wrote no manifest; 768 dims is the
                         * one legacy space that is safe to infer. */
                        space = crate::embedding::INPUT_EMBEDDER_ID.to_string();
                    }
                    Ok(Fmt::Flat(FlatInfo {
                        dim,
                        records,
                        space,
                        declared_dim,
                    }))
                }
                Err(rc) => Ok(Fmt::Foreign(rc)),
            }
        }
    }

    /* ---------------- verify ---------------- */

    pub(super) fn verify(path: &str) -> Result<VerifyReport, BrainError> {
        let f = inspect(path)?;
        match f {
            Fmt::Segmented(m) => Ok(scrub_segmented(path, &m)),
            Fmt::Flat(flat) => Ok(scrub_flat(path, &flat)),
            Fmt::Foreign(rc) => Err(BrainError::foreign(
                path,
                format!("codec rc {rc}; not a QTSQ container this build can read"),
            )),
        }
    }

    /// Group the manifest's segment entries by generation, in manifest order.
    fn generations(m: &bs::Manifest) -> Vec<(u64, usize, usize)> {
        let mut out: Vec<(u64, usize, usize)> = Vec::new();
        for s in &m.segments {
            let epoch = seg_epoch(&s.file).unwrap_or(m.epoch);
            match out.last_mut() {
                Some(last) if last.0 == epoch => {
                    last.1 += 1;
                    last.2 += s.records;
                }
                _ => out.push((epoch, 1, s.records)),
            }
        }
        out.sort_by_key(|g| g.0);
        out
    }

    fn seg_epoch(name: &str) -> Option<u64> {
        let body = name.strip_prefix("seg-")?;
        let (epoch, _seq) = body.split_once('-')?;
        epoch.parse::<u64>().ok()
    }

    /// The full scrub: manifest shape, every referenced segment present,
    /// digest equal to the one the manifest published, every segment decodes,
    /// record counts agree, ids dense, sizes match the declared vector dim.
    fn scrub_segmented(path: &str, m: &bs::Manifest) -> VerifyReport {
        let dir = bs::segments_dir(path);
        let emb_dim = m
            .embedding
            .get("dimension")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let emb_id = m
            .embedding
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let mut r = VerifyReport {
            path: path.to_string(),
            format: "segmented".into(),
            schema: m.schema.clone(),
            segments: m.segments.len(),
            generation: m.epoch,
            embedding_id: emb_id,
            embedding_dim: emb_dim,
            ..Default::default()
        };

        /* id → tombstone?  A later version of the same id wins, exactly like
         * the loader's by_id insert. */
        let mut rows: BTreeMap<u32, bool> = BTreeMap::new();

        for seg in &m.segments {
            let file = format!("{dir}/{}", seg.file);
            let fp = Path::new(&file);
            if !fp.exists() {
                r.missing_files.push(seg.file.clone());
                r.damaged_segments += 1;
                r.errors.push(format!(
                    "missing segment file: {} (the manifest lists {} record(s) it carried)",
                    seg.file, seg.records
                ));
                continue;
            }
            let mut bad = false;
            if let Ok(md) = std::fs::metadata(fp) {
                r.bytes += md.len();
            }
            let digest = bs::file_digest(&file);
            match (digest.as_deref(), seg.sha256.as_str()) {
                (None, _) => {
                    bad = true;
                    r.errors.push(format!("unreadable segment file: {}", seg.file));
                }
                (Some(got), want) if !want.is_empty() && got != want => {
                    bad = true;
                    r.errors.push(format!(
                        "digest mismatch: {} (manifest {}, file {})",
                        seg.file, want, got
                    ));
                }
                _ => {}
            }
            match unsafe { bs::read_segment(&file) } {
                Ok(recs) => {
                    if recs.len() != seg.records {
                        bad = true;
                        r.errors.push(format!(
                            "record count mismatch: {} holds {} record(s), the manifest says {}",
                            seg.file,
                            recs.len(),
                            seg.records
                        ));
                    }
                    if emb_dim > 0 {
                        if let Some(bad_vec) = recs.iter().find(|x| x.vec.len() as u64 != emb_dim) {
                            bad = true;
                            r.errors.push(format!(
                                "vector width mismatch in {}: record id {} has {} value(s), the space is {}-dimensional",
                                seg.file,
                                bad_vec.id,
                                bad_vec.vec.len(),
                                emb_dim
                            ));
                        }
                    }
                    for rec in &recs {
                        let tomb = rec
                            .meta
                            .get("tier")
                            .and_then(|t| t.as_u64())
                            .map(|t| t as u8 == crate::memory::cma::TIER_FORGOTTEN)
                            .unwrap_or(false);
                        rows.insert(rec.id, tomb);
                    }
                }
                Err(rc) => {
                    bad = true;
                    r.errors.push(format!(
                        "segment does not decode: {} (codec rc {rc})",
                        seg.file
                    ));
                }
            }
            if bad {
                r.damaged_segments += 1;
            } else {
                r.verified_segments += 1;
            }
        }

        r.records = rows.len();
        r.tombstones = rows.values().filter(|t| **t).count();
        r.live = r.records - r.tombstones;

        /* The manifest stamps the live row count at every flush. A mismatch
         * means the manifest no longer describes the bytes — including a
         * segment cut short by a crash that still decodes. */
        if m.records > 0 && m.records as usize != rows.len() {
            r.errors.push(format!(
                "manifest integrity: it records {} row(s) but the segments hold {} distinct id(s)",
                m.records,
                rows.len()
            ));
        }
        if m.schema != bs::SEGMENT_SCHEMA {
            r.errors.push(format!(
                "manifest schema '{}' is not the one this runtime writes ('{}')",
                m.schema,
                bs::SEGMENT_SCHEMA
            ));
        }
        /* Ids must be dense — the HNSW index addresses vectors by position. */
        if let Some((i, id)) = rows.keys().enumerate().find(|(i, id)| **id as usize != *i) {
            r.errors.push(format!(
                "id density: position {i} holds id {id} — a hole means an id was lost"
            ));
        }
        r.ok = r.errors.is_empty() && r.missing_files.is_empty() && r.damaged_segments == 0;
        r
    }

    fn scrub_flat(path: &str, flat: &FlatInfo) -> VerifyReport {
        let mut r = VerifyReport {
            path: path.to_string(),
            format: "flat-v1".into(),
            schema: "sofuu-cma-flat@0".into(),
            records: flat.records,
            live: flat.records,
            tombstones: 0,
            segments: 0,
            embedding_id: flat.space.clone(),
            embedding_dim: if flat.declared_dim > 0 {
                flat.declared_dim
            } else {
                flat.dim as u64
            },
            ..Default::default()
        };
        if let Ok(md) = std::fs::metadata(path) {
            r.bytes = md.len();
        }
        if flat.records == 0 {
            r.notes.push("the container holds no records".to_string());
        }
        if flat.declared_dim > 0 && flat.declared_dim != flat.dim as u64 {
            r.errors.push(format!(
                "metadata declares a {}-dimensional space but the tensor is {}-dimensional",
                flat.declared_dim, flat.dim
            ));
        }
        if flat.space.is_empty() && flat.dim != crate::embedding::HASH_DIM {
            r.errors.push(format!(
                "unmanifested {}-dimensional vectors: this runtime refuses to guess a vector space",
                flat.dim
            ));
        }
        r.notes.push(
            "flat v1 brain: there is no segment store to scrub — `sofuu brain migrate <path>` \
             converts it to the segmented format"
                .to_string(),
        );
        r.ok = r.errors.is_empty();
        r
    }

    /* ---------------- repair ---------------- */

    pub(super) fn repair(path: &str) -> Result<RepairOutcome, BrainError> {
        let f = inspect(path)?;
        match f {
            Fmt::Foreign(rc) => Err(BrainError::foreign(
                path,
                format!("codec rc {rc}; not a QTSQ container this build can read"),
            )),
            Fmt::Flat(_) => {
                /* Nothing to repair: the engine's repair pass only knows the
                 * segmented store, and a flat brain has no manifest/segment
                 * pair to disagree. Say so instead of faking a repair. */
                Ok(RepairOutcome {
                    path: path.to_string(),
                    ok: true,
                    segmented: false,
                    message: format!(
                        "{path}: not a segmented brain — a flat v1 brain has no segment store to repair"
                    ),
                    ..Default::default()
                })
            }
            Fmt::Segmented(_) => {
                let rep = crate::rt::memory::repair(path);
                let lost = !rep.quarantined.is_empty()
                    || !rep.missing.is_empty()
                    || rep.records_dropped > 0;
                let mut errors = Vec::new();
                if !rep.found {
                    errors.push(rep.message.clone());
                } else if !rep.ok {
                    errors.push(format!(
                        "{path}: damaged and NOT repaired on disk — the bytes are untouched"
                    ));
                }
                Ok(RepairOutcome {
                    path: path.to_string(),
                    /* A store the engine could not even recognise is not
                     * "ok"; one it healed (with or without loss) is. */
                    ok: rep.found && rep.ok,
                    segmented: rep.found,
                    records: rep.records,
                    segments: rep.segments,
                    quarantined: rep.quarantined,
                    adopted: rep.adopted,
                    missing: rep.missing,
                    records_dropped: rep.records_dropped,
                    rewritten: rep.rewritten,
                    lost,
                    errors,
                    message: rep.message,
                })
            }
        }
    }

    /* ---------------- stats ---------------- */

    pub(super) fn stats(path: &str) -> Result<StatsReport, BrainError> {
        let f = inspect(path)?;
        match f {
            Fmt::Foreign(rc) => Err(BrainError::foreign(
                path,
                format!("codec rc {rc}; not a QTSQ container this build can read"),
            )),
            Fmt::Flat(flat) => {
                let manifest_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                Ok(StatsReport {
                    path: path.to_string(),
                    format: "flat-v1".into(),
                    schema: "sofuu-cma-flat@0".into(),
                    ok: true,
                    records: flat.records,
                    live: flat.records,
                    versions: flat.records,
                    generation: 0,
                    bytes: manifest_bytes,
                    disk_bytes: manifest_bytes,
                    manifest_bytes,
                    embedding_id: flat.space.clone(),
                    embedding_dim: if flat.declared_dim > 0 {
                        flat.declared_dim
                    } else {
                        flat.dim as u64
                    },
                    ..Default::default()
                })
            }
            Fmt::Segmented(m) => {
                let v = scrub_segmented(path, &m);
                let dir = bs::segments_dir(path);
                let referenced: BTreeSet<&str> =
                    m.segments.iter().map(|s| s.file.as_str()).collect();
                let on_disk = bs::list_segment_files(&dir);
                let orphan_segments = on_disk
                    .iter()
                    .filter(|n| !referenced.contains(n.as_str()))
                    .count();
                let quarantined_files = std::fs::read_dir(&dir)
                    .map(|rd| {
                        rd.flatten()
                            .filter(|e| {
                                e.file_name()
                                    .to_str()
                                    .map(|n| n.contains(".qtsq.torn"))
                                    .unwrap_or(false)
                            })
                            .count()
                    })
                    .unwrap_or(0);
                /* Everything the store occupies: every file under <path>.d
                 * (referenced, orphaned and quarantined alike) plus the
                 * manifest. `bytes` above is the referenced part only, so
                 * the difference is exactly the reclaimable slack. */
                let mut disk_bytes = 0u64;
                if let Ok(rd) = std::fs::read_dir(&dir) {
                    for e in rd.flatten() {
                        if let Ok(md) = e.metadata() {
                            if md.is_file() {
                                disk_bytes += md.len();
                            }
                        }
                    }
                }
                let manifest_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                disk_bytes += manifest_bytes;
                Ok(StatsReport {
                    path: path.to_string(),
                    format: "segmented".into(),
                    schema: m.schema.clone(),
                    ok: v.ok,
                    records: v.records,
                    live: v.live,
                    tombstones: v.tombstones,
                    versions: m.total_versions(),
                    segments: m.segments.len(),
                    generations: generations(&m),
                    generation: m.epoch,
                    bytes: v.bytes,
                    disk_bytes,
                    manifest_bytes,
                    evicted_total: m.evicted_total,
                    last_decay_at: m.last_decay_at,
                    quarantined_files,
                    orphan_segments,
                    missing_segments: v.missing_files.len(),
                    damaged_segments: v.damaged_segments,
                    embedding_id: v.embedding_id.clone(),
                    embedding_dim: v.embedding_dim,
                    errors: v.errors.clone(),
                })
            }
        }
    }

    /* ---------------- backup / restore ---------------- */

    /// Refuse to write into a destination that already holds bytes.
    fn ensure_empty_destination(dst: &str) -> Result<(), BrainError> {
        let target = Path::new(dst);
        if target.exists() {
            if target.is_dir() {
                let nonempty = std::fs::read_dir(target)
                    .map(|mut rd| rd.next().is_some())
                    .unwrap_or(false);
                if nonempty {
                    return Err(BrainError::new(
                        "sofuu.brain.dest_not_empty",
                        format!("{dst}: destination directory is not empty — refusing to clobber it"),
                        EXIT_USAGE,
                    ));
                }
            } else if std::fs::metadata(target).map(|m| m.len()).unwrap_or(0) > 0 {
                return Err(BrainError::new(
                    "sofuu.brain.dest_not_empty",
                    format!("{dst}: destination already exists and is not empty — refusing to clobber it"),
                    EXIT_USAGE,
                ));
            }
        }
        let dir = bs::segments_dir(dst);
        if let Ok(rd) = std::fs::read_dir(&dir) {
            let mut it = rd.flatten();
            if it.next().is_some() {
                return Err(BrainError::new(
                    "sofuu.brain.dest_not_empty",
                    format!(
                        "{dir}: the destination's segment directory is not empty — refusing to clobber it"
                    ),
                    EXIT_USAGE,
                ));
            }
        }
        Ok(())
    }

    #[cfg(unix)]
    fn same_inode(a: &str, b: &str) -> bool {
        use std::os::unix::fs::MetadataExt;
        match (std::fs::metadata(a), std::fs::metadata(b)) {
            (Ok(ma), Ok(mb)) => ma.dev() == mb.dev() && ma.ino() == mb.ino(),
            _ => false,
        }
    }

    #[cfg(not(unix))]
    fn same_inode(a: &str, b: &str) -> bool {
        /* No portable inode identity: fall back to size equality (the
         * digest check in the destination verify is the real gate). */
        match (std::fs::metadata(a), std::fs::metadata(b)) {
            (Ok(ma), Ok(mb)) => ma.len() == mb.len(),
            _ => false,
        }
    }

    fn copy_durably(src: &str, dst: &str) -> Result<u64, BrainError> {
        let tmp = format!("{dst}.tmp");
        let n = std::fs::copy(src, &tmp).map_err(|e| BrainError::io(dst, "copy failed", e))?;
        if !wal::publish(&tmp, dst) {
            let _ = std::fs::remove_file(&tmp);
            return Err(BrainError::new(
                "sofuu.brain.io",
                format!("{dst}: staged copy could not be published (fsync/rename failed)"),
                EXIT_DAMAGE,
            ));
        }
        Ok(n)
    }

    pub(super) fn snapshot(
        src: &str,
        dst: &str,
        allow_damaged: bool,
        link: bool,
    ) -> Result<SnapshotReport, BrainError> {
        if Path::new(src) == Path::new(dst) {
            return Err(BrainError::usage(format!(
                "{src}: source and destination are the same path"
            )));
        }
        let f = inspect(src)?;
        let (format, manifest) = match &f {
            Fmt::Segmented(m) => ("segmented".to_string(), Some(m.clone())),
            Fmt::Flat(_) => ("flat-v1".to_string(), None),
            Fmt::Foreign(rc) => {
                return Err(BrainError::foreign(
                    src,
                    format!("codec rc {rc}; not a QTSQ container this build can read"),
                ))
            }
        };

        /* The source must verify before we hand out something that says
         * "backup": a snapshot of a store we cannot read would be a promise
         * we cannot keep. `--allow-damaged` snapshots anyway, and the report
         * still says ok:false + exit 1. */
        let src_verify = verify(src)?;
        if !src_verify.ok && !allow_damaged {
            return Err(BrainError::new(
                "sofuu.brain.damaged",
                format!(
                    "{src}: the store does not verify ({} error(s)) — refusing to write a snapshot \
                     that would look like a good backup (use --allow-damaged to snapshot it anyway)",
                    src_verify.errors.len() + src_verify.missing_files.len()
                ),
                EXIT_DAMAGE,
            ));
        }
        ensure_empty_destination(dst)?;

        let mut rep = SnapshotReport {
            command: if link { "backup" } else { "restore" }.to_string(),
            src: src.to_string(),
            dst: dst.to_string(),
            format: format.clone(),
            source_ok: src_verify.ok,
            ..Default::default()
        };
        if !src_verify.ok {
            rep.warnings.push(format!(
                "the source does not verify ({} error(s)) — this snapshot is NOT a clean backup",
                src_verify.errors.len() + src_verify.missing_files.len()
            ));
        }

        // Manifests/segments land in the destination's store directory.
        let dst_dir = bs::segments_dir(dst);
        if let Some(parent) = Path::new(dst).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if manifest.is_some() {
            std::fs::create_dir_all(&dst_dir)
                .map_err(|e| BrainError::io(&dst_dir, "cannot create the store directory", e))?;
        }

        if let Some(m) = &manifest {
            for seg in &m.segments {
                let from = format!("{}/{}", bs::segments_dir(src), seg.file);
                if !Path::new(&from).exists() {
                    /* Already reported by the source verify; with
                     * --allow-damaged we simply do not carry it over. */
                    continue;
                }
                let to = format!("{dst_dir}/{}", seg.file);
                let mut linked = false;
                if link {
                    if std::fs::hard_link(&from, &to).is_ok() && same_inode(&from, &to) {
                        linked = true;
                    } else {
                        /* Filesystem without hardlinks, or a link that did not
                         * produce the same inode — fall back to a real copy. */
                        let _ = std::fs::remove_file(&to);
                    }
                }
                if linked {
                    rep.linked += 1;
                    rep.inode_verified += 1;
                } else {
                    rep.copied += 1;
                    copy_durably(&from, &to)?;
                }
                rep.files += 1;
                rep.bytes += std::fs::metadata(&from).map(|md| md.len()).unwrap_or(0);
            }
            // Seal the directory entries before the manifest makes them
            // authoritative.
            wal::fsync_dir(&dst_dir);
        }

        /* THE CONSISTENCY POINT: the manifest lands last, through the same
         * staged fsync→rename→fsync-dir publish the store itself uses. A
         * reader that sees it sees every segment it references. */
        let manifest_bytes = std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
        copy_durably(src, dst)?;
        rep.manifest_published = true;
        rep.files += 1;
        rep.copied += 1;
        rep.bytes += manifest_bytes;

        // Read the snapshot back through the same verifier an operator would.
        match verify(dst) {
            Ok(v) => {
                rep.destination_ok = v.ok;
                rep.destination_records = v.records;
                rep.destination_segments = v.segments;
                if !v.ok {
                    rep.errors.extend(v.errors.iter().cloned());
                }
            }
            Err(e) => {
                rep.destination_ok = false;
                rep.errors.push(e.message.clone());
            }
        }
        rep.ok = rep.destination_ok && rep.source_ok;
        Ok(rep)
    }

    /* ---------------- migrate ---------------- */

    /// Take the pre-migration snapshot: `<path>.v1backup` (`.1`, `.2`, … when
    /// taken). Byte-identical copy, digest reported on stderr by the caller's
    /// human renderer.
    fn v1_snapshot(path: &str) -> Result<String, BrainError> {
        for n in 0..100u32 {
            let candidate = if n == 0 {
                format!("{path}.v1backup")
            } else {
                format!("{path}.v1backup.{n}")
            };
            if Path::new(&candidate).exists() {
                continue;
            }
            let n_bytes = copy_durably(path, &candidate)?;
            if n_bytes == 0 {
                return Err(BrainError::new(
                    "sofuu.brain.io",
                    format!("{candidate}: pre-migration snapshot is empty"),
                    EXIT_DAMAGE,
                ));
            }
            return Ok(candidate);
        }
        Err(BrainError::new(
            "sofuu.brain.io",
            format!("{path}: could not find a free .v1backup name"),
            EXIT_DAMAGE,
        ))
    }

    pub(super) fn migrate(
        path: &str,
        eval: &mut dyn FnMut(&str) -> Result<String, String>,
    ) -> Result<MigrateReport, BrainError> {
        let f = inspect(path)?;
        let (from_format, flat) = match f {
            Fmt::Segmented(_) => {
                return Ok(MigrateReport {
                    ok: true,
                    migrated: false,
                    src: path.to_string(),
                    from_format: "segmented".to_string(),
                    to_format: "segmented".to_string(),
                    message: format!(
                        "{path}: already on the current segmented format — nothing to migrate"
                    ),
                    ..Default::default()
                })
            }
            Fmt::Flat(flat) => ("flat-v1".to_string(), flat),
            Fmt::Foreign(rc) => {
                return Err(BrainError::foreign(
                    path,
                    format!("codec rc {rc}; not a QTSQ container this build can read"),
                ))
            }
        };

        /* Snapshot BEFORE anything writes: the migration path replaces the
         * flat container in place. */
        let backup = v1_snapshot(path)?;

        let spaces = if flat.space.is_empty() {
            String::new()
        } else {
            format!(", {}", serde_json::to_string(&flat.space).unwrap_or_default())
        };
        /* The runtime's OWN migration path: open the flat brain (which
         * classifies the vector space and refuses an incompatible one), then
         * flush — the first segmented flush is what converts the flat
         * container into a manifest + segments. No byte rewriting here. */
        let script = format!(
            "(function () {{
                var out = {{ ok: false, records: 0, flushed: null, message: \"\" }};
                try {{
                    var b = sofuu.memory.open({path_lit}, {dim}{spaces});
                    out.records = b.count();
                    /* W6.1: flush() returns {{ok, recordsWritten, segmentsSealed}}
                     * (it used to return 0/1), so accept either shape. */
                    var f = b.flush();
                    out.flushed = f;
                    out.ok = (f === true) || (f === 0) || !!(f && f.ok === true);
                }} catch (e) {{
                    out.message = String((e && e.message) ? e.message : e);
                }}
                globalThis.__brainReport = JSON.stringify(out);
            }})()",
            path_lit = serde_json::to_string(path).unwrap_or_else(|_| "\"\"".to_string()),
            dim = flat.dim,
            spaces = spaces,
        );
        let raw = eval(&script).map_err(|e| {
            BrainError::new(
                "sofuu.brain.engine",
                format!("{path}: the runtime engine could not run the migration: {e}"),
                EXIT_DAMAGE,
            )
        })?;
        let engine: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
            BrainError::new(
                "sofuu.brain.engine",
                format!("{path}: the migration returned an unparseable report: {e}"),
                EXIT_DAMAGE,
            )
        })?;
        let engine_ok = engine.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
        let engine_message = engine
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let engine_records = engine.get("records").and_then(|v| v.as_u64());

        /* The verdict comes from the BYTES, never from the engine's own flag:
         * the engine may fail after it has already written (a flush that
         * publishes the store and then errors), and a report that says
         * "unchanged" about a converted store would be a lie. */
        let now_segmented = matches!(inspect(path), Ok(Fmt::Segmented(_)));
        if !now_segmented {
            return Ok(MigrateReport {
                ok: false,
                migrated: false,
                src: path.to_string(),
                from_format,
                to_format: "segmented".to_string(),
                from_dimension: flat.dim,
                from_records: flat.records,
                space: flat.space,
                v1backup: backup,
                message: if engine_message.is_empty() {
                    "the runtime refused to open this brain".to_string()
                } else {
                    engine_message
                },
                errors: vec![format!(
                    "{path}: the runtime refused the migration — the store is still the flat v1 \
                     container (pre-migration copy at the .v1backup path)"
                )],
                ..Default::default()
            });
        }

        let mut rep = MigrateReport {
            ok: false,
            migrated: true,
            src: path.to_string(),
            from_format,
            to_format: "segmented".to_string(),
            from_dimension: flat.dim,
            from_records: flat.records,
            space: flat.space,
            records_imported: engine_records.unwrap_or(flat.records as u64) as usize,
            v1backup: backup,
            ..Default::default()
        };
        /* Read the converted store back through the same verifier an operator
         * would, so "migrated" is a claim about bytes, not about intent. */
        match verify(path) {
            Ok(v) => {
                rep.segments = v.segments;
                rep.records_imported = v.records;
                rep.message = format!(
                    "imported {} record(s) from the flat v1 container into {} segment(s)",
                    v.records, v.segments
                );
                if v.ok
                    && (flat.records == 0
                        || v.records == flat.records
                        || engine_records == Some(v.records as u64))
                {
                    rep.ok = true;
                } else {
                    rep.errors.extend(v.errors.iter().cloned());
                    if flat.records > 0 && v.records != flat.records {
                        rep.errors.push(format!(
                            "record count changed during migration: {} in the v1 container, {} imported",
                            flat.records, v.records
                        ));
                    }
                }
            }
            Err(e) => rep.errors.push(e.message.clone()),
        }
        if !engine_ok && rep.ok {
            /* The store converted even though the engine reported a failure
             * (e.g. its flush returned an error after publishing): say so. */
            rep.message.push_str(&format!(
                " (the engine still reported: {})",
                if engine_message.is_empty() {
                    "not ok"
                } else {
                    &engine_message
                }
            ));
        }
        Ok(rep)
    }
}

/* ------------------------------------------------------------------ */
/* Help                                                                */
/* ------------------------------------------------------------------ */

/// The `sofuu brain --help` page. A const so the tests can assert on the
/// promise it makes (`v1`, `.v1backup`, `SFB2`) instead of on stdout.
const HELP_TEXT: &str = concat!(
    "\x1b[1mUsage:\x1b[0m\n",
    "  sofuu brain verify <path> [--json]\n",
    "  sofuu brain repair <path> [--json]\n",
    "  sofuu brain stats  <path> [--json]\n",
    "  sofuu brain backup <src> <dst> [--json] [--allow-damaged]\n",
    "  sofuu brain restore <backup> <dst> [--json] [--allow-damaged]\n",
    "  sofuu brain migrate <path> [--json]\n",
    "  sofuu brain <command> --help\n",
    "\n",
    "\x1b[1mWhat it operates on\x1b[0m — this runtime's brain store
  <path>                          manifest container (committed generation)
  <path>.d/seg-<gen>-<seq>.qtsq   immutable sealed segments, digest-pinned
  A brain written before the segmented store (the \x1b[1mv1\x1b[0m format) is ONE
  flat QTSQ container; `migrate` converts it. The sfb tool's SFB2/SFS2 files
  are a DIFFERENT format: they are refused with sofuu.brain.foreign, never
  guessed at.

\x1b[1mSubcommands\x1b[0m
  \x1b[1mverify\x1b[0m <path>   Full scrub: manifest integrity, every referenced segment
                  present, its SHA-256 equal to the published one, every
                  segment decodes, record counts agree, ids are dense.
                  --json reports ok, records, segments, damaged_segments,
                  missing_files, errors. Exit 1 when not ok.
  \x1b[1mrepair\x1b[0m <path>   Runs the storage engine's own repair pass (the one a
                  resilient open runs): what cannot be trusted is quarantined
                  beside the store as <name>.torn (bytes kept), a clean segment
                  behind a stale manifest digest is adopted, the manifest is
                  re-published for the survivors, and every dropped record
                  version is reported. Exit 1 when anything was lost.
  \x1b[1mstats\x1b[0m <path>    Records (live / tombstones), record versions, segments,
                  generations, bytes, and the recovery counters the store
                  really keeps (capacity evictions, quarantined *.torn files,
                  orphan segments, missing and damaged segments).
  \x1b[1mbackup\x1b[0m <src> <dst>  Snapshot: sealed segments are HARDLINKED (the inode is
                  read back and must match), anything else is copied, the
                  manifest is written LAST as the consistency point, and the
                  result is verified through the same scrub as `verify`.
                  Refuses a non-empty destination.
  \x1b[1mrestore\x1b[0m <backup> <dst>  Rebuild a brain from a backup: segments are
                  byte-copied (a restored store must not share inodes with
                  the backup it came from), the manifest last, then verified.
                  Refuses a non-empty destination.
  \x1b[1mmigrate\x1b[0m <path>   Converts a \x1b[1mv1\x1b[0m brain (the flat single-file container
                  written before the segmented store) into the current
                  format through the runtime's own open→flush path — never by
                  rewriting bytes here. A byte-identical pre-migration
                  snapshot is taken first at <path>.v1backup, and the imported
                  record count is reported. A brain already in the segmented
                  format reports migrated:false and exits 0.

\x1b[1mOptions\x1b[0m
  --json             machine-readable report (the same fields as the text)
  --allow-damaged    backup/restore: snapshot even when the source does not
                     verify (the report still says ok:false and exits 1)

\x1b[1mExit codes\x1b[0m
  0  ok                1  damage or record loss detected
  2  usage error       errors carry a sofuu.brain.* code on stderr

\x1b[1mEnvironment\x1b[0m
  SOFUU_BRAIN_KEY    vault key override (else the machine key file)\n"
);

pub fn print_help() {
    println!("{HELP_TEXT}");
}

/// The help page as text (tests assert the promise it makes).
pub fn print_help_text() -> &'static str {
    HELP_TEXT
}

/* W7: one page per subcommand, in the same shape as HELP_TEXT — a plain
 * multi-line constant, so the indentation survives (a `\`-continued format
 * string strips it) and nothing needs brace escaping. */

const VERIFY_HELP: &str = "\x1b[1mUsage:\x1b[0m sofuu brain verify <path> [--json]

Full scrub of a brain store: the manifest must parse and carry this
runtime's schema, every segment it references must exist, its SHA-256 must
equal the digest the manifest published, every segment must decode with the
record count the manifest claims, and the record ids must stay dense.

\x1b[1mArguments\x1b[0m
  <path>        brain path (the manifest; segments live in <path>.d/)

\x1b[1mOptions\x1b[0m
  --json        {\"ok\",\"records\",\"segments\",\"damaged_segments\",
                 \"missing_files\",\"errors\",\"bytes\",\"format\",…}

\x1b[1mExit\x1b[0m  0 ok · 1 damage detected · 2 usage
";

const REPAIR_HELP: &str = "\x1b[1mUsage:\x1b[0m sofuu brain repair <path> [--json]

Runs the storage engine's own repair pass over a segmented brain: torn or
digest-mismatched segments are quarantined as <name>.torn (the bytes are kept
for salvage), a segment that reads cleanly but does not match a stale manifest
is adopted, the manifest is re-published for what survived, and the record
versions that were dropped are counted.

\x1b[1mArguments\x1b[0m
  <path>        brain path (single-writer: do not repair a brain another
                process has open)

\x1b[1mOptions\x1b[0m
  --json        {\"ok\",\"segmented\",\"records\",\"segments\",\"quarantined\",
                 \"adopted\",\"missing\",\"records_dropped\",\"lost\",…}

\x1b[1mExit\x1b[0m  0 ok, nothing lost · 1 anything lost or not repaired · 2 usage
";

const STATS_HELP: &str = "\x1b[1mUsage:\x1b[0m sofuu brain stats <path> [--json]

Counters read from the store itself: records (live / tombstoned), record
versions, segments, generations, bytes, plus the recovery counters the engine
tracks (capacity evictions, quarantined *.torn files, orphan segments a
crashed compaction left, missing and damaged segments).

\x1b[1mArguments\x1b[0m
  <path>        brain path

\x1b[1mOptions\x1b[0m
  --json        the same counters as JSON

\x1b[1mExit\x1b[0m  0 ok · 1 the store is damaged · 2 usage
";

const BACKUP_HELP: &str = "\x1b[1mUsage:\x1b[0m sofuu brain backup <src> <dst> [--json] [--allow-damaged]

Snapshots <src> at <dst>. Sealed segments are hardlinked where the filesystem
allows it (link() then the inode is read back and must match), anything else
is copied; the manifest is written LAST through the store's own staged
fsync→rename→fsync-dir publish, so a reader that sees the destination manifest
sees every segment it names. The result is verified with the same scrub as
`brain verify`, and the linked/copied counts are reported.

\x1b[1mArguments\x1b[0m
  <src>         brain to snapshot
  <dst>         destination manifest path; segments go to <dst>.d/
                (must not exist, or be empty)

\x1b[1mOptions\x1b[0m
  --json            machine-readable report
  --allow-damaged   snapshot a source that does not verify

\x1b[1mExit\x1b[0m  0 ok · 1 damage/loss · 2 usage (incl. a non-empty destination)
";

const RESTORE_HELP: &str = "\x1b[1mUsage:\x1b[0m sofuu brain restore <backup> <dst> [--json] [--allow-damaged]

Rebuilds a brain at <dst> from a backup made by `brain backup` or by hand.
Segments are byte-copied — not hardlinked, because a restored store must not
share inodes with the backup it came from — the manifest is published last as
the consistency point, and the result is verified.

\x1b[1mArguments\x1b[0m
  <backup>      backup manifest path (segments in <backup>.d/)
  <dst>         destination manifest path; segments go to <dst>.d/
                (must not exist, or be empty)

\x1b[1mOptions\x1b[0m
  --json            machine-readable report
  --allow-damaged   restore from a backup that does not verify

\x1b[1mExit\x1b[0m  0 ok · 1 damage/loss · 2 usage (incl. a non-empty destination)
";

const MIGRATE_HELP: &str = "\x1b[1mUsage:\x1b[0m sofuu brain migrate <path> [--json]

Migrates a \x1b[1mv1\x1b[0m brain — the flat single-file container written before the
segmented store — to the current format. The conversion runs through the
runtime's own open→flush path (the same code a normal open uses), never by
rewriting bytes here, and a byte-identical pre-migration snapshot is taken
first at <path>.v1backup.

A brain that is ALREADY segmented reports migrated:false and exits 0.
A container this runtime cannot read (for example the sfb tool's SFB2/SFS2)
fails with sofuu.brain.foreign: there is no such reader on this side, and this
command will not pretend to convert one.

\x1b[1mArguments\x1b[0m
  <path>        brain path (must not be open by another process)

\x1b[1mOptions\x1b[0m
  --json        {\"ok\",\"migrated\",\"from\",\"to\",\"records_imported\",
                 \"segments\",\"v1backup\",…}

\x1b[1mExit\x1b[0m  0 migrated or already current · 1 refused/failed · 2 usage
";

fn sub_help_text(sub: &str) -> &'static str {
    match sub {
        "verify" => VERIFY_HELP,
        "repair" => REPAIR_HELP,
        "stats" => STATS_HELP,
        "backup" => BACKUP_HELP,
        "restore" => RESTORE_HELP,
        "migrate" => MIGRATE_HELP,
        _ => HELP_TEXT,
    }
}

fn print_sub_help(sub: &str) {
    println!("{}", sub_help_text(sub));
}

/* ------------------------------------------------------------------ */
/* Tests                                                              */
/* ------------------------------------------------------------------ */

#[cfg(all(test, has_qtsq))]
mod tests {
    /* `store::snapshot` and the crate-private wrapper in the parent module
     * share a name; alias the engine one so the glob imports cannot clash. */
    use super::store::snapshot as store_snapshot;
    use super::*;
    use crate::rt::brain_store as bs;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!(
            "sofuu-braincmd-{tag}-{}-{nanos}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn rec(id: u32, dim: usize, text: &str) -> bs::SegRecord {
        let mut v = vec![0.0f32; dim];
        v[(id as usize) % dim] = 1.0;
        v[0] = 0.5;
        bs::SegRecord {
            id,
            vec: v,
            meta: serde_json::json!({
                "tier": 1,
                "strength": 1.0,
                "text": text,
                "role": "memory",
                "created_at": 7,
            }),
        }
    }

    /// A clean segmented store: `chunks` segments of `per` records each, ids
    /// dense from 0, manifest digests taken from the bytes exactly like a
    /// flush does.
    fn build_store(
        path: &std::path::Path,
        chunks: &[Vec<bs::SegRecord>],
        dim: usize,
    ) -> bs::Manifest {
        let p = path.to_string_lossy().to_string();
        let dir = bs::segments_dir(&p);
        std::fs::create_dir_all(&dir).unwrap();
        let mut m = bs::Manifest::new(serde_json::json!({
            "id": "hash-v1",
            "input": "hash-v1",
            "dimension": dim,
            "artifact": "",
        }));
        m.epoch = 1;
        for (i, chunk) in chunks.iter().enumerate() {
            let name = bs::segment_file(1, i as u64 + 1);
            let file = format!("{dir}/{name}");
            unsafe {
                bs::write_segment(&file, dim, chunk).expect("segment write");
            }
            let sha = bs::file_digest(&file).unwrap();
            m.segments.push(bs::SegmentMeta {
                file: name,
                records: chunk.len(),
                sha256: sha,
            });
        }
        m.records = chunks.iter().map(|c| c.len()).sum::<usize>() as u64;
        unsafe {
            bs::write_manifest(&p, &m).expect("manifest write");
        }
        m
    }

    fn two_segments(path: &std::path::Path) -> bs::Manifest {
        build_store(
            path,
            &[
                vec![rec(0, 8, "alpha"), rec(1, 8, "beta")],
                vec![rec(2, 8, "gamma"), rec(3, 8, "delta")],
            ],
            8,
        )
    }

    fn no_engine(_script: &str) -> Result<String, String> {
        Err("no runtime in this test".to_string())
    }

    /* ── verify ──────────────────────────────────────────────── */

    #[test]
    fn verify_on_a_clean_store_passes() {
        let dir = tmpdir("verify-clean");
        let path = dir.join("brain.qtsq");
        two_segments(&path);
        let p = path.to_string_lossy().to_string();

        let r = verify_report(&p).expect("verify runs");
        assert!(r.ok, "a clean store must verify: {:?}", r.errors);
        assert_eq!(r.format, "segmented");
        assert_eq!(r.records, 4, "four distinct ids");
        assert_eq!(r.live, 4);
        assert_eq!(r.tombstones, 0);
        assert_eq!(r.segments, 2);
        assert_eq!(r.verified_segments, 2);
        assert_eq!(r.damaged_segments, 0);
        assert!(r.missing_files.is_empty());
        assert_eq!(r.embedding_dim, 8);
        assert_eq!(r.generation, 1);
        assert!(r.bytes > 0);

        /* The JSON report carries the fields the contract names. */
        let j = r.to_json();
        for key in [
            "ok",
            "records",
            "segments",
            "damaged_segments",
            "missing_files",
            "errors",
        ] {
            assert!(j.get(key).is_some(), "JSON report is missing {key}");
        }
        assert_eq!(j["ok"], serde_json::json!(true));
        assert_eq!(j["records"], serde_json::json!(4));

        /* And the CLI exit code agrees with the report. */
        assert_eq!(cmd_verify(&p, false), EXIT_OK);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_after_a_byte_flip_fails_with_a_damaged_segment() {
        let dir = tmpdir("verify-flip");
        let path = dir.join("brain.qtsq");
        let m = two_segments(&path);
        let p = path.to_string_lossy().to_string();
        let victim = format!("{}/{}", bs::segments_dir(&p), m.segments[1].file);

        /* Flip a byte in the payload region of the second segment. */
        let mut bytes = std::fs::read(&victim).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&victim, &bytes).unwrap();

        let r = verify_report(&p).expect("verify still runs");
        assert!(!r.ok, "a flipped byte must fail the scrub");
        assert_eq!(r.damaged_segments, 1, "exactly one damaged segment");
        assert_eq!(r.verified_segments, 1);
        assert!(
            r.errors.iter().any(|e| e.contains(&m.segments[1].file)),
            "the error must name the damaged file: {:?}",
            r.errors
        );
        assert_eq!(cmd_verify(&p, false), EXIT_DAMAGE, "exit 1 on damage");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_reports_a_missing_segment_file() {
        let dir = tmpdir("verify-missing");
        let path = dir.join("brain.qtsq");
        let m = two_segments(&path);
        let p = path.to_string_lossy().to_string();
        std::fs::remove_file(format!("{}/{}", bs::segments_dir(&p), m.segments[0].file)).unwrap();

        let r = verify_report(&p).expect("verify still runs");
        assert!(!r.ok);
        assert_eq!(r.missing_files, vec![m.segments[0].file.clone()]);
        assert_eq!(r.damaged_segments, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /* ── repair ──────────────────────────────────────────────── */

    #[test]
    fn repair_quarantines_the_torn_segment_and_keeps_the_good_records() {
        let dir = tmpdir("repair");
        let path = dir.join("brain.qtsq");
        let m = two_segments(&path);
        let p = path.to_string_lossy().to_string();
        let victim_name = m.segments[1].file.clone();
        let victim = format!("{}/{}", bs::segments_dir(&p), victim_name);
        tear(&victim);

        let r = repair_outcome(&p).expect("repair runs");
        assert!(r.segmented, "the engine recognised a segmented store");
        assert!(
            r.quarantined.contains(&victim_name),
            "an undecodable segment must be quarantined: quarantined={:?} adopted={:?}",
            r.quarantined,
            r.adopted
        );
        assert!(r.ok, "the store must verify again after repair: {:?}", r.errors);
        assert_eq!(r.records, 2, "the two good records survive");
        assert!(
            r.lost,
            "records were dropped by the quarantine, so the loss must be reported"
        );
        assert!(r.records_dropped >= 2);
        /* The bytes are KEPT beside the store, never deleted. */
        assert!(
            std::fs::read_dir(bs::segments_dir(&p))
                .unwrap()
                .flatten()
                .any(|e| e.file_name().to_string_lossy().contains(".torn")),
            "the quarantined segment must exist as *.torn"
        );
        /* The repaired store verifies, and re-running repair now reports no
         * loss (there is nothing left to lose). */
        assert!(verify_report(&p).unwrap().ok);
        assert_eq!(cmd_verify(&p, false), EXIT_OK);
        assert_eq!(cmd_repair(&p, false), EXIT_OK);

        /* The EXIT CODE of a repair that drops records is 1 — measured on a
         * fresh torn store, because the one above is already healed. */
        let path2 = dir.join("brain2.qtsq");
        let m2 = two_segments(&path2);
        let p2 = path2.to_string_lossy().to_string();
        tear(&format!("{}/{}", bs::segments_dir(&p2), m2.segments[1].file));
        assert_eq!(cmd_repair(&p2, false), EXIT_DAMAGE);
        assert_eq!(cmd_verify(&p2, false), EXIT_OK);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A torn write: destroy the container header so the codec refuses the
    /// file outright (a single flipped payload byte still DECODES, and the
    /// engine then *adopts* it instead of quarantining).
    fn tear(path: &str) {
        let mut bytes = std::fs::read(path).unwrap();
        assert!(bytes.len() > 64);
        for b in bytes.iter_mut().take(64) {
            *b ^= 0xFF;
        }
        std::fs::write(path, &bytes).unwrap();
    }

    #[test]
    fn repair_on_a_healthy_store_reports_no_loss() {
        let dir = tmpdir("repair-clean");
        let path = dir.join("brain.qtsq");
        two_segments(&path);
        let p = path.to_string_lossy().to_string();
        let r = repair_outcome(&p).expect("repair runs");
        assert!(r.ok);
        assert!(!r.lost);
        assert_eq!(r.records, 4);
        assert_eq!(r.segments, 2);
        assert_eq!(cmd_repair(&p, false), EXIT_OK, "nothing lost → exit 0");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /* ── stats ───────────────────────────────────────────────── */

    #[test]
    fn stats_parses_and_counts_what_the_store_tracks() {
        let dir = tmpdir("stats");
        let path = dir.join("brain.qtsq");
        two_segments(&path);
        let p = path.to_string_lossy().to_string();

        let s = stats_report(&p).expect("stats runs");
        assert!(s.ok);
        assert_eq!(s.format, "segmented");
        assert_eq!(s.records, 4);
        assert_eq!(s.live, 4);
        assert_eq!(s.tombstones, 0);
        assert_eq!(s.versions, 4, "no append segments → versions == rows");
        assert_eq!(s.segments, 2);
        assert_eq!(s.generations, vec![(1, 2, 4)]);
        assert_eq!(s.generation, 1);
        assert!(s.bytes > 0);
        assert!(s.disk_bytes >= s.bytes);
        assert_eq!(s.evicted_total, 0);
        assert_eq!(s.quarantined_files, 0);
        assert_eq!(s.orphan_segments, 0);

        let j = s.to_json();
        let back: serde_json::Value = serde_json::from_str(&j.to_string()).unwrap();
        assert_eq!(back["records"], serde_json::json!(4));
        assert_eq!(back["segments"], serde_json::json!(2));
        assert_eq!(back["generations"][0]["generation"], serde_json::json!(1));
        assert!(back["recovery"].is_object());
        assert_eq!(cmd_stats(&p, true), EXIT_OK);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stats_counts_tombstones_and_orphans() {
        let dir = tmpdir("stats-tomb");
        let path = dir.join("brain.qtsq");
        let mut dead = rec(1, 8, "buried");
        dead.meta["tier"] = serde_json::json!(crate::memory::cma::TIER_FORGOTTEN);
        build_store(&path, &[vec![rec(0, 8, "alpha"), dead]], 8);
        let p = path.to_string_lossy().to_string();
        /* An unreferenced segment left behind by a crashed compaction. */
        std::fs::write(
            format!("{}/{}", bs::segments_dir(&p), bs::segment_file(1, 9)),
            b"leftover",
        )
        .unwrap();

        let s = stats_report(&p).expect("stats runs");
        assert_eq!(s.records, 2);
        assert_eq!(s.live, 1);
        assert_eq!(s.tombstones, 1);
        assert_eq!(s.orphan_segments, 1, "the stray seg-000001-000009 must be counted");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /* ── backup / restore ────────────────────────────────────── */

    #[cfg(unix)]
    #[test]
    fn backup_hardlinks_segments_and_restore_round_trips() {
        use std::os::unix::fs::MetadataExt;
        let dir = tmpdir("backup");
        let src = dir.join("src.qtsq");
        two_segments(&src);
        let s = src.to_string_lossy().to_string();
        let dst = dir.join("backup.qtsq");
        let d = dst.to_string_lossy().to_string();

        let r = store_snapshot(&s, &d, false, true).expect("backup runs");
        assert!(r.ok, "backup must verify: {:?}", r.errors);
        assert_eq!(r.linked, 2, "both sealed segments are hardlinked");
        assert_eq!(r.inode_verified, 2, "every link was verified inode-equal");
        assert_eq!(r.copied, 1, "the manifest is copied, not linked");
        assert_eq!(r.files, 3);
        assert!(r.manifest_published);
        assert!(r.destination_ok);
        assert_eq!(r.destination_records, 4);

        for seg in bs::list_segment_files(&bs::segments_dir(&s)) {
            let a = format!("{}/{}", bs::segments_dir(&s), seg);
            let b = format!("{}/{}", bs::segments_dir(&d), seg);
            let (ma, mb) = (std::fs::metadata(&a).unwrap(), std::fs::metadata(&b).unwrap());
            assert_eq!(ma.ino(), mb.ino(), "{seg} must share an inode (hardlink)");
            assert_eq!(ma.dev(), mb.dev());
        }

        /* restore into a fresh path, then verify the round trip */
        let back = dir.join("restored.qtsq");
        let b = back.to_string_lossy().to_string();
        let rr = store_snapshot(&d, &b, false, false).expect("restore runs");
        assert!(rr.ok, "restore must verify: {:?}", rr.errors);
        assert_eq!(rr.linked, 0, "restore copies: no shared inodes");
        assert_eq!(rr.copied, 3);
        assert_eq!(rr.destination_records, 4);
        assert_eq!(rr.destination_segments, 2);
        let v = verify_report(&b).unwrap();
        assert!(v.ok);
        assert_eq!(v.records, 4);
        /* The restored bytes are identical to the source's. */
        for seg in bs::list_segment_files(&bs::segments_dir(&s)) {
            let a = std::fs::read(format!("{}/{}", bs::segments_dir(&s), seg)).unwrap();
            let c = std::fs::read(format!("{}/{}", bs::segments_dir(&b), seg)).unwrap();
            assert_eq!(a, c, "{seg} must round-trip byte-identically");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn backup_refuses_a_non_empty_destination() {
        let dir = tmpdir("backup-refuse");
        let src = dir.join("src.qtsq");
        two_segments(&src);
        let s = src.to_string_lossy().to_string();
        let dst = dir.join("taken.qtsq");
        let d = dst.to_string_lossy().to_string();
        std::fs::write(&dst, b"do not clobber me").unwrap();

        let err = store_snapshot(&s, &d, false, true).expect_err("backup must refuse");
        assert_eq!(err.code, "sofuu.brain.dest_not_empty");
        assert_eq!(err.exit, EXIT_USAGE);
        assert_eq!(cmd_backup(&s, &d, true, false), EXIT_USAGE);
        assert_eq!(std::fs::read(&dst).unwrap(), b"do not clobber me");
        /* Nothing was written beside it either. */
        assert!(!Path::new(&bs::segments_dir(&d)).exists());

        /* A non-empty <dst>.d alone is enough to refuse. */
        let dst2 = dir.join("taken2.qtsq");
        let d2 = dst2.to_string_lossy().to_string();
        std::fs::create_dir_all(bs::segments_dir(&d2)).unwrap();
        std::fs::write(format!("{}/stale.qtsq", bs::segments_dir(&d2)), b"x").unwrap();
        assert!(store_snapshot(&s, &d2, false, true).is_err());

        /* Same for restore. */
        assert_eq!(cmd_restore(&s, &d, true, false), EXIT_USAGE);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn backup_refuses_a_damaged_source_unless_allowed() {
        let dir = tmpdir("backup-damaged");
        let src = dir.join("src.qtsq");
        let m = two_segments(&src);
        let s = src.to_string_lossy().to_string();
        let victim = format!("{}/{}", bs::segments_dir(&s), m.segments[0].file);
        tear(&victim);

        let dst = dir.join("b.qtsq");
        let d = dst.to_string_lossy().to_string();
        let err = store_snapshot(&s, &d, false, true).expect_err("must refuse");
        assert_eq!(err.code, "sofuu.brain.damaged");
        assert!(!dst.exists(), "no destination is created on refusal");

        /* --allow-damaged snapshots anyway, and still reports ok:false. */
        let r = store_snapshot(&s, &d, true, true).expect("allowed");
        assert!(!r.ok, "--allow-damaged must not turn a damaged source into ok");
        assert!(!r.source_ok);
        assert!(!r.warnings.is_empty());
        /* …and the CLI exit code agrees, on a destination of its own. */
        let d2 = dir.join("b2.qtsq").to_string_lossy().to_string();
        assert_eq!(cmd_backup(&s, &d2, true, true), EXIT_DAMAGE);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /* ── migrate ─────────────────────────────────────────────── */

    #[test]
    fn migrate_reports_already_current_and_never_pretends() {
        let dir = tmpdir("migrate-current");
        let path = dir.join("brain.qtsq");
        two_segments(&path);
        let p = path.to_string_lossy().to_string();
        let mut eval = no_engine;
        let r = migrate_store(&p, &mut eval).expect("migrate runs");
        assert!(r.ok);
        assert!(!r.migrated, "nothing to migrate on a segmented store");
        assert_eq!(r.from_format, "segmented");
        assert!(r.v1backup.is_empty(), "no snapshot is taken when nothing changes");
        assert_eq!(cmd_migrate(&p, true, &mut no_engine), EXIT_OK);
        assert!(migrate_report_help_mentions_v1());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_refuses_a_foreign_container_with_a_typed_code() {
        let dir = tmpdir("migrate-foreign");
        let path = dir.join("nope.qtsq");
        std::fs::write(&path, b"SFB2 not a qtsq file").unwrap();
        let p = path.to_string_lossy().to_string();
        let err = verify_report(&p).expect_err("foreign bytes are refused");
        assert_eq!(err.code, "sofuu.brain.foreign");
        assert_eq!(err.exit, EXIT_DAMAGE);
        let err = migrate_store(&p, &mut no_engine).expect_err("foreign bytes are refused");
        assert_eq!(err.code, "sofuu.brain.foreign");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn migrate_report_help_mentions_v1() -> bool {
        /* The help text is the contract for "say which it is". */
        let help = print_help_text();
        help.contains("v1") && help.contains(".v1backup") && help.contains("SFB2")
    }

    /* ── usage ───────────────────────────────────────────────── */

    #[test]
    fn usage_errors_exit_2_and_help_exits_0() {
        let argv = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut eval = no_engine;
        assert_eq!(run(&argv(&["sofuu", "brain"]), &mut eval), EXIT_USAGE);
        assert_eq!(
            run(&argv(&["sofuu", "brain", "verify"]), &mut eval),
            EXIT_USAGE,
            "verify without a path"
        );
        assert_eq!(
            run(&argv(&["sofuu", "brain", "backup", "only-one"]), &mut eval),
            EXIT_USAGE
        );
        assert_eq!(
            run(&argv(&["sofuu", "brain", "verify", "/x", "--nope"]), &mut eval),
            EXIT_USAGE,
            "unknown flag"
        );
        assert_eq!(
            run(&argv(&["sofuu", "brain", "frobnicate", "/x"]), &mut eval),
            EXIT_USAGE
        );
        assert_eq!(
            run(&argv(&["sofuu", "brain", "--help"]), &mut eval),
            EXIT_OK
        );
        assert_eq!(
            run(&argv(&["sofuu", "brain", "verify", "--help"]), &mut eval),
            EXIT_OK
        );
        assert_eq!(
            run(&argv(&["sofuu", "brain", "migrate", "--help"]), &mut eval),
            EXIT_OK
        );
    }

    #[test]
    fn verify_of_a_missing_path_is_a_usage_error() {
        let err = verify_report("/nonexistent/definitely-not-a-brain").expect_err("missing");
        assert_eq!(err.code, "sofuu.brain.missing");
        assert_eq!(err.exit, EXIT_USAGE);
    }
}
