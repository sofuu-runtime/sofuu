// rt/wal.rs — durable atomic publish + write-ahead journal (PLAN-SOFUU-CORE C2).
//
// The persistence paths stage through a `{file}.tmp` sibling and rename over
// the target. Rename is atomic with respect to the DIRECTORY ENTRY, but on a
// power loss the tmp file's DATA blocks may not have reached the platter
// while the rename did — the classic failure that leaves a brain file that
// exists and is garbage. The write path therefore needs three things this
// module provides:
//
//   1. `publish` — fsync the staged file, rename, fsync the directory, in
//      that order. Without the first fsync the data can be lost; without the
//      second the rename itself can be lost. (session_store.rs already uses
//      this discipline for sessions; the brain/KV paths did not.)
//
//   2. `Journal` — a write-ahead intent record. Before any rename the
//      intended content is described (target, tmp, sha256) and fsynced, so a
//      crash mid-commit is recoverable: `recover` rolls the commit FORWARD
//      when the staged bytes are intact and match, and BACK when they are
//      not, never leaving a half-committed pair (page file + index.json).
//
//   3. `sha256_hex` — the digest the journal records. Pure Rust so the
//      module (and its tests) build in QTSQ-free CI, where the codec's own
//      hash is not linked.
//
// Scope note: per-STREAM content integrity (sha256 of the original bytes,
// written into the container header at compress time and verified after
// decode) is already provided by the QTSQ codec itself (qtsq_singularity.c
// writes header.checksum; qtsq_radiation.c verifies it post-decode). This
// module covers what the codec cannot: the durability of the rename, the
// multi-file commit, and the crash window between them.

use std::path::Path;

pub const JOURNAL_NAME: &str = ".sofuu-wal.json";
const JOURNAL_VERSION: u32 = 1;
/// A journal younger than this may belong to a LIVE writer (desktop + CLI on
/// one project), so it is never recovered out from under it — the same
/// reasoning as the 10-minute tmp-file sweep in rt/memory.rs, with a much
/// shorter window because a journal only lives for the duration of a rename.
const JOURNAL_STALE_SECS: u64 = 30;

/* ------------------------------------------------------------------ */
/* SHA-256 (FIPS 180-4, pure Rust — no crate, no codec dependency)     */
/* ------------------------------------------------------------------ */

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Streaming SHA-256: hash discontiguous parts without concatenating them.
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.buf_len != 56 {
            self.update(&[0x00]);
        }
        // update() above already counted padding bytes in total_len, but the
        // length field must be the MESSAGE length — write it directly.
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        self.compress(&block);
        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Digest one file without loading it whole.
pub fn sha256_file(path: &str) -> Option<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Some(hex(&hasher.finalize()))
}

/* ------------------------------------------------------------------ */
/* Durability primitives                                               */
/* ------------------------------------------------------------------ */

/// fsync a file so its bytes are on stable storage.
pub fn fsync_file(path: &str) -> bool {
    std::fs::File::open(path)
        .and_then(|f| f.sync_all())
        .is_ok()
}

/// fsync a directory so a rename/creation inside it is durable.
pub fn fsync_dir(dir: &str) {
    if let Ok(f) = std::fs::File::open(dir) {
        let _ = f.sync_all();
    }
}

/// Durable atomic publish: fsync the staged file, rename it over the target,
/// then fsync the directory. Returns false (leaving the target untouched)
/// when the staged file cannot be made durable or the rename fails.
pub fn publish(tmp: &str, target: &str) -> bool {
    if !fsync_file(tmp) {
        return false;
    }
    if std::fs::rename(tmp, target).is_err() {
        return false;
    }
    if let Some(parent) = Path::new(target).parent() {
        fsync_dir(&parent.to_string_lossy());
    }
    true
}

/* ------------------------------------------------------------------ */
/* Write-ahead journal                                                 */
/* ------------------------------------------------------------------ */

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalEntry {
    pub target: String,
    pub tmp: String,
    pub sha256: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    pub rolled_forward: usize,
    pub rolled_back: usize,
    pub journal_found: bool,
}

/// A write-ahead intent record for one directory. `begin` claims the journal
/// file exclusively (recovering a crashed predecessor first), `record`
/// appends one staged file with its digest, and `commit` clears the journal
/// once every rename has landed.
pub struct Journal {
    dir: String,
    path: String,
    entries: Vec<JournalEntry>,
}

impl Journal {
    /// Claim the journal for `dir`. A STALE journal (a crashed writer's)
    /// is recovered first, then the claim is retried; a FRESH one may belong
    /// to a live writer, so `None` is returned and the caller publishes
    /// without an intent record rather than stealing it.
    pub fn begin(dir: &str) -> Option<Journal> {
        let path = format!("{dir}/{JOURNAL_NAME}");
        if std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .is_ok()
        {
            let mut j = Journal {
                dir: dir.to_string(),
                path,
                entries: Vec::new(),
            };
            j.flush()?;
            return Some(j);
        }
        if !journal_is_stale(&path) {
            return None;
        }
        recover(dir);
        if std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .is_ok()
        {
            let mut j = Journal {
                dir: dir.to_string(),
                path,
                entries: Vec::new(),
            };
            j.flush()?;
            return Some(j);
        }
        None
    }

    /// Record one staged file (hashed now, before any rename).
    pub fn record(&mut self, tmp: &str, target: &str) -> bool {
        let Some(sha) = sha256_file(tmp) else {
            return false;
        };
        self.entries.push(JournalEntry {
            target: target.to_string(),
            tmp: tmp.to_string(),
            sha256: sha,
        });
        self.flush().is_some()
    }

    /// Clear the journal — every recorded rename has landed durably.
    pub fn commit(self) -> bool {
        if std::fs::remove_file(&self.path).is_err() {
            return false;
        }
        fsync_dir(&self.dir);
        true
    }

    fn flush(&mut self) -> Option<()> {
        let json = serialize(&self.entries);
        if std::fs::write(&self.path, json.as_bytes()).is_err() {
            return None;
        }
        if !fsync_file(&self.path) {
            return None;
        }
        Some(())
    }
}

/// True when the journal file is old enough that its writer cannot still be
/// mid-commit. An unreadable mtime can never prove liveness, so it counts as
/// NOT stale (the conservative direction: never steal a journal we cannot age).
fn journal_is_stale(path: &str) -> bool {
    let Ok(md) = std::fs::metadata(path) else {
        return false;
    };
    let Ok(modified) = md.modified() else {
        return false;
    };
    std::time::SystemTime::now()
        .duration_since(modified)
        .map(|age| age.as_secs() >= JOURNAL_STALE_SECS)
        .unwrap_or(false)
}

fn serialize(entries: &[JournalEntry]) -> String {
    let mut out = format!("{{\"version\":{JOURNAL_VERSION},\"entries\":[");
    for (i, e) in entries.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"target\":{},\"tmp\":{},\"sha256\":{}}}",
            json_string(&e.target),
            json_string(&e.tmp),
            json_string(&e.sha256)
        ));
    }
    out.push_str("]}");
    out
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Minimal reader for the journal shape this module writes.
fn parse_entries(text: &str) -> Vec<JournalEntry> {
    let mut entries = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("{\"target\":") {
        let body = &rest[start..];
        let Some(end) = body.find('}') else { break };
        let obj = &body[..end + 1];
        let target = extract_field(obj, "\"target\":");
        let tmp = extract_field(obj, "\"tmp\":");
        let sha = extract_field(obj, "\"sha256\":");
        if let (Some(target), Some(tmp), Some(sha)) = (target, tmp, sha) {
            entries.push(JournalEntry { target, tmp, sha256: sha });
        }
        rest = &body[end + 1..];
    }
    entries
}

fn extract_field(obj: &str, key: &str) -> Option<String> {
    let start = obj.find(key)? + key.len();
    let rest = obj[start..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => {
                    let hex: String = chars.by_ref().take(4).collect();
                    let code = u32::from_str_radix(&hex, 16).ok()?;
                    out.push(char::from_u32(code)?);
                }
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    None
}

/// Replay a crashed writer's journal: roll each entry forward when the staged
/// bytes are intact and match the recorded digest, back when they are not.
/// A target that already carries the recorded digest counts as committed.
/// Returns the report; the journal is removed either way.
pub fn recover(dir: &str) -> RecoveryReport {
    let path = format!("{dir}/{JOURNAL_NAME}");
    let mut report = RecoveryReport::default();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return report;
    };
    report.journal_found = true;
    for entry in parse_entries(&text) {
        let tmp_ok = sha256_file(&entry.tmp).map(|s| s == entry.sha256).unwrap_or(false);
        if tmp_ok {
            if publish(&entry.tmp, &entry.target) {
                report.rolled_forward += 1;
                continue;
            }
        }
        let target_ok = sha256_file(&entry.target)
            .map(|s| s == entry.sha256)
            .unwrap_or(false);
        if target_ok {
            /* The rename landed before the crash; drop the stale staging. */
            let _ = std::fs::remove_file(&entry.tmp);
            continue;
        }
        /* Neither side matches the intent: abort this entry. The old target
         * (if any) is left exactly as it was. */
        let _ = std::fs::remove_file(&entry.tmp);
        report.rolled_back += 1;
    }
    let _ = std::fs::remove_file(&path);
    fsync_dir(dir);
    report
}

/* ------------------------------------------------------------------ */
/* Tests                                                               */
/* ------------------------------------------------------------------ */

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("sofuu-wal-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // 56-byte message: exercises the padding boundary.
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Multi-block (112 bytes): the second NIST short-message vector.
        assert_eq!(
            sha256_hex(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            ),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
    }

    #[test]
    fn sha256_streaming_equals_oneshot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let one = sha256_hex(&data);
        let mut h = Sha256::new();
        for chunk in data.chunks(7) {
            h.update(chunk);
        }
        assert_eq!(hex(&h.finalize()), one);
    }

    #[test]
    fn publish_moves_and_removes_tmp() {
        let dir = tmpdir("publish");
        let tmp = dir.join("x.tmp");
        let target = dir.join("x");
        std::fs::write(&tmp, b"payload").unwrap();
        assert!(publish(&tmp.to_string_lossy(), &target.to_string_lossy()));
        assert!(!tmp.exists());
        assert_eq!(std::fs::read(&target).unwrap(), b"payload");
    }

    #[test]
    fn publish_fails_without_tmp() {
        let dir = tmpdir("publish-missing");
        let target = dir.join("x");
        assert!(!publish(
            &dir.join("nope.tmp").to_string_lossy(),
            &target.to_string_lossy()
        ));
        assert!(!target.exists());
    }

    #[test]
    fn journal_records_and_commits() {
        let dir = tmpdir("journal");
        let staged = dir.join("a.tmp");
        std::fs::write(&staged, b"one").unwrap();
        let j = Journal::begin(&dir.to_string_lossy()).expect("begin");
        let jpath = dir.join(JOURNAL_NAME);
        assert!(jpath.exists(), "journal file must exist while open");
        let mut j = j;
        assert!(j.record(&staged.to_string_lossy(), &dir.join("a").to_string_lossy()));
        let text = std::fs::read_to_string(&jpath).unwrap();
        assert!(text.contains("a.tmp"), "{text}");
        assert!(j.commit());
        assert!(!jpath.exists(), "commit must clear the journal");
    }

    #[test]
    fn recover_rolls_forward_intact_staging() {
        let dir = tmpdir("recover-fwd");
        let staged = dir.join("page.tmp");
        std::fs::write(&staged, b"new-bytes").unwrap();
        let target = dir.join("page");
        std::fs::write(&target, b"old-bytes").unwrap();

        let d = dir.to_string_lossy().to_string();
        let mut j = Journal::begin(&d).unwrap();
        assert!(j.record(&staged.to_string_lossy(), &target.to_string_lossy()));
        // Simulate the crash: journal left behind, rename never happened.
        drop(j);

        let r = recover(&d);
        assert!(r.journal_found);
        assert_eq!(r.rolled_forward, 1, "{r:?}");
        assert_eq!(std::fs::read(&target).unwrap(), b"new-bytes");
        assert!(!staged.exists());
        assert!(!dir.join(JOURNAL_NAME).exists());
    }

    #[test]
    fn recover_rolls_back_corrupt_staging() {
        let dir = tmpdir("recover-back");
        let staged = dir.join("page.tmp");
        std::fs::write(&staged, b"new-bytes").unwrap();
        let target = dir.join("page");
        std::fs::write(&target, b"old-bytes").unwrap();

        let d = dir.to_string_lossy().to_string();
        let mut j = Journal::begin(&d).unwrap();
        assert!(j.record(&staged.to_string_lossy(), &target.to_string_lossy()));
        drop(j);

        // Torn write: the staged file no longer matches the recorded digest.
        std::fs::write(&staged, b"torn").unwrap();

        let r = recover(&d);
        assert_eq!(r.rolled_back, 1, "{r:?}");
        assert_eq!(r.rolled_forward, 0);
        assert_eq!(std::fs::read(&target).unwrap(), b"old-bytes");
        assert!(!staged.exists());
    }

    #[test]
    fn recover_accepts_already_committed_target() {
        let dir = tmpdir("recover-done");
        let staged = dir.join("page.tmp");
        std::fs::write(&staged, b"new-bytes").unwrap();
        let target = dir.join("page");

        let d = dir.to_string_lossy().to_string();
        let mut j = Journal::begin(&d).unwrap();
        assert!(j.record(&staged.to_string_lossy(), &target.to_string_lossy()));
        drop(j);

        // The rename landed; only the journal removal was lost.
        std::fs::rename(&staged, &target).unwrap();

        let r = recover(&d);
        assert_eq!(r.rolled_forward, 0, "{r:?}");
        assert_eq!(r.rolled_back, 0, "{r:?}");
        assert_eq!(std::fs::read(&target).unwrap(), b"new-bytes");
        assert!(!dir.join(JOURNAL_NAME).exists());
    }

    #[test]
    fn recover_is_a_noop_without_journal() {
        let dir = tmpdir("recover-clean");
        let r = recover(&dir.to_string_lossy());
        assert!(!r.journal_found);
        assert_eq!(r.rolled_forward, 0);
        assert_eq!(r.rolled_back, 0);
    }

    #[test]
    fn begin_refuses_a_fresh_journal() {
        let dir = tmpdir("begin-fresh");
        let d = dir.to_string_lossy().to_string();
        let j = Journal::begin(&d).expect("first begin");
        // A second writer must NOT steal a live journal.
        assert!(Journal::begin(&d).is_none());
        assert!(j.commit());
    }

    /// Backdate a file's mtime so it reads as a crashed writer's leftover.
    fn age_file(path: &std::path::Path) {
        let old = std::time::SystemTime::now()
            - std::time::Duration::from_secs(JOURNAL_STALE_SECS + 10);
        let f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        f.set_modified(old).unwrap();
    }

    #[test]
    fn begin_recovers_a_crashed_predecessor() {
        let dir = tmpdir("begin-recovers");
        let staged = dir.join("page.tmp");
        std::fs::write(&staged, b"new-bytes").unwrap();
        let target = dir.join("page");

        let d = dir.to_string_lossy().to_string();
        let mut j = Journal::begin(&d).unwrap();
        assert!(j.record(&staged.to_string_lossy(), &target.to_string_lossy()));
        drop(j);

        // A crashed writer's journal is old; a new writer must recover it.
        age_file(&dir.join(JOURNAL_NAME));
        let j2 = Journal::begin(&d).expect("second begin");
        assert_eq!(std::fs::read(&target).unwrap(), b"new-bytes");
        assert!(j2.commit());
    }

    #[test]
    fn journal_roundtrips_awkward_paths() {
        let entries = vec![
            JournalEntry {
                target: "/tmp/a b/c\".qtsq".to_string(),
                tmp: "/tmp/a b/c\".qtsq.tmp".to_string(),
                sha256: "ab".repeat(32),
            },
            JournalEntry {
                target: "/tmp/ünïcode/файл.qtsq".to_string(),
                tmp: "/tmp/ünïcode/файл.qtsq.tmp".to_string(),
                sha256: "cd".repeat(32),
            },
        ];
        let text = serialize(&entries);
        assert_eq!(parse_entries(&text), entries);
    }

    #[test]
    fn sha256_file_matches_bytes() {
        let dir = tmpdir("sha-file");
        let p = dir.join("blob");
        let data: Vec<u8> = (0..300_000u32).map(|i| (i % 253) as u8).collect();
        std::fs::write(&p, &data).unwrap();
        assert_eq!(
            sha256_file(&p.to_string_lossy()).unwrap(),
            sha256_hex(&data)
        );
    }
}
