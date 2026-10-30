// rt/brain_store.rs — segmented brain storage (ROADMAP-BRAIN-CMA B3).
//
// A brain used to be ONE .qtsq container holding every record, and every
// flush rewrote the whole file — O(total) bytes for O(new) records
// (measured: 2000 records → 6.2 MB rewritten, 195 ms, per flush). The
// canonical path now holds a small MANIFEST and the records live in
// bounded SEGMENTS beside it:
//
//   <path>                          manifest container (one "manifest" JSON stream)
//   <path>.d/seg-<epoch>-<seq>.qtsq segments (tensor "memories" + JSON "meta")
//
// A segment entry is a full record SNAPSHOT keyed by id, and loading
// applies segments in order — so a later entry is a record UPDATE (recall
// reinforcement, a tier change, a tombstone), and a flush appends only the
// records that changed. Closed segments are never rewritten; the active
// one is bounded by SEGMENT_CAP records, which bounds a flush.
//
// This mirrors the proven session_store.rs contract: one committed
// manifest, immutable sealed segments, at most one mutable active segment,
// every publish behind the C2 journal (fsync → rename → fsync dir).
//
// W3.5 (QTSQ_UPGRADE_PLAN): a segment used to be an all-or-nothing unit —
// one damaged record cost every record the segment carried. New segments
// set a header flag (`SEG_FLAG_PER_RECORD_CRC`) meaning "each record
// carries its own CRC32 covering id + metadata + THE VECTOR", and end in a
// 24-byte trailer that pins the exact file length. A reader that finds the
// flag verifies every record and drops exactly the damaged ones; it still
// refuses the WHOLE segment when the fault is structural (a truncated
// file, a short tensor, a broken CRC table), because a half-written
// segment is not a record-level fault. Segments written before the flag
// existed carry no flag and keep the old whole-segment behaviour.
//
// W3.4: a compaction bumps `epoch`, which names BOTH the segments and the
// graph snapshot of that generation. `reclaim_unreferenced` deletes the
// generations nobody references any more, so a store that keeps compacting
// does not grow without bound.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;

use sofuu_ffi::qtsq;

/// B4 (ROADMAP-BRAIN-CMA): the vault password is the MACHINE-LOCAL key
/// (rt/brain_key.rs) — a 0600 key file under the user's home, or the
/// SOFUU_BRAIN_KEY override. Every brain written before B4 carries the
/// legacy fixed password instead; reads fall back to it and the first write
/// re-encrypts (see `decrypt_with_fallback`).
pub fn vault_password() -> String {
    crate::rt::brain_key::local_vault_password()
}

/// The password every pre-B4 brain file carries. Read-only: it exists to
/// migrate those files, never to write new ones.
pub const LEGACY_PASSWORD: &str = crate::rt::brain_key::LEGACY_PASSWORD;

/// Not a QTSQ container at all (garbage bytes, a foreign file). Distinct
/// from a decrypt failure: a caller may fall back to another reader for
/// this, but must REFUSE a container it cannot open.
pub const ERR_NOT_CONTAINER: c_int = -1001;

/// Codec tensor-compression level: 0 = full contest (see rt/memory.rs).
///
/// COST, measured (SOFUU_PROFILE=1, 256x64 segment): compress ≈ 2.5 ms,
/// pack 0.0 ms, write 0.5 ms — and **encrypt ≈ 26 ms**. `encrypt` is
/// `qtsq_vault_encrypt_password`, which runs Argon2id at m_cost = 64 MB on
/// EVERY call, and the store has no derived-key API to share that work.
/// A flush touches three containers (read manifest, write segment, write
/// manifest) = ~78 ms of key derivation before anything else happens, and a
/// 100k-record open reads 391 of them = ~11 s. That, not the codec, is the
/// floor under the plan's "flush P99 < 50 ms" and "open < 1 s" gates; the fix
/// is a key-derivation cache or a derived-key entry point in the vault.
pub const QTSQ_LEVEL_FULL: c_int = 0;

pub const MANIFEST_STREAM: &str = "manifest";
pub const SEGMENT_META_STREAM: &str = "meta";
pub const SEGMENT_TENSOR_STREAM: &str = "memories";
pub const SEGMENT_SCHEMA: &str = "sofuu-cma-segments@1";

/// Records per segment before rotation. Bounds both the active rewrite
/// (a flush touches at most one segment) and the read amplification of a
/// single segment. 256 records × 768 dims ≈ 786 KB of vectors.
pub const SEGMENT_CAP: usize = 256;

/* ------------------------------------------------------------------ */
/* W3.5: per-record integrity                                          */
/* ------------------------------------------------------------------ */

/// A segment-header flag bit meaning "every record carries its own CRC32
/// over id + metadata + vector" (see `record_crc`). Readers that do not
/// know the bit ignore it — it lives in the segment's `meta` JSON, whose
/// only keys the pre-W3.5 reader touched were `dim` and `records`.
pub const SEG_FLAG_PER_RECORD_CRC: u32 = 1 << 0;

/// The `meta` schema a CRC-carrying segment advertises. Absent (as in
/// every segment written before W3.5) means "no flags, whole-segment
/// integrity only".
pub const SEGMENT_RECORD_SCHEMA: &str = "sofuu-cma-segment@2";

/// Trailer bytes appended after the container: a fixed-length sentinel
/// that makes TRUNCATION (and only truncation) a whole-segment fault.
///
/// Why it exists: the codec happily reads a container with its tail cut
/// off (measured 2026-10-30 on a 71 KB segment: cutting 4 KB off the end
/// still decoded all 256 records), so a half-written segment could
/// otherwise be mistaken for a short-but-healthy one. The trailer records
/// the payload length, so ANY truncation of a flagged segment is detected
/// without a whole-file digest — which would defeat the point of
/// per-record integrity by rejecting the segment for one bad record.
///
/// `payload_len: u64 | flags: u32 | crc32(payload_len|flags): u32 |
///  b"SFUSEGT1": 8` = 24 bytes.
pub const SEG_TRAILER_LEN: usize = 24;
const SEG_TRAILER_MAGIC: &[u8; 8] = b"SFUSEGT1";

/// What one segment read produced.
///
/// `records` is everything that could be trusted; `dropped` is the count
/// of records that carried a CRC and failed it — the number the store
/// reports as `records_dropped`. For a segment without
/// `SEG_FLAG_PER_RECORD_CRC` a damaged file is still an `Err` (or, for a
/// legacy file the codec shortens silently, the records it yields), and
/// `dropped` is always 0: the old whole-segment contract is unchanged.
#[derive(Clone, Debug, Default)]
pub struct SegmentRead {
    pub records: Vec<SegRecord>,
    pub dropped: usize,
}

/// The trailer of `path`, if it has one that verifies.
///
/// * `Ok(true)`  — a valid W3.5 trailer.
/// * `Ok(false)` — no trailer at all: a segment written before W3.5.
/// * `Err`       — a trailer IS there but does not verify (a torn or
///   extended file). The caller must refuse the whole segment.
fn segment_trailer(path: &str) -> Result<bool, c_int> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).map_err(|_| qtsq::QTSQ_ERR_FORMAT)?;
    let len = f.metadata().map_err(|_| qtsq::QTSQ_ERR_FORMAT)?.len();
    if len < SEG_TRAILER_LEN as u64 {
        return Ok(false);
    }
    f.seek(SeekFrom::End(-(SEG_TRAILER_LEN as i64)))
        .map_err(|_| qtsq::QTSQ_ERR_FORMAT)?;
    let mut tail = [0u8; SEG_TRAILER_LEN];
    f.read_exact(&mut tail).map_err(|_| qtsq::QTSQ_ERR_FORMAT)?;
    if &tail[16..24] != SEG_TRAILER_MAGIC {
        /* Either a pre-W3.5 segment, or a flagged one that lost its tail.
         * The caller decides: a flagged segment without a trailer is
         * truncated, an unflagged one is simply old. */
        return Ok(false);
    }
    let payload_len = u64::from_le_bytes(tail[0..8].try_into().unwrap_or([0; 8]));
    let crc = u32::from_le_bytes(tail[12..16].try_into().unwrap_or([0; 4]));
    if crc != crc32(&tail[0..12]) {
        return Err(qtsq::QTSQ_ERR_FORMAT);
    }
    if payload_len.saturating_add(SEG_TRAILER_LEN as u64) != len {
        /* Bytes lost or gained after the container was sealed. */
        return Err(qtsq::QTSQ_ERR_FORMAT);
    }
    Ok(true)
}

/// Append the W3.5 trailer to a freshly written container.
fn append_trailer(path: &str, flags: u32) -> Result<(), c_int> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|_| qtsq::QTSQ_ERR_FORMAT)?;
    let payload_len = f.metadata().map_err(|_| qtsq::QTSQ_ERR_FORMAT)?.len();
    let mut buf = Vec::with_capacity(SEG_TRAILER_LEN);
    buf.extend_from_slice(&payload_len.to_le_bytes());
    buf.extend_from_slice(&flags.to_le_bytes());
    let crc = crc32(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf.extend_from_slice(SEG_TRAILER_MAGIC);
    f.write_all(&buf).map_err(|_| qtsq::QTSQ_ERR_FORMAT)?;
    f.sync_all().map_err(|_| qtsq::QTSQ_ERR_FORMAT)
}

/// CRC-32 (IEEE 802.3, the zlib polynomial) — the same cheap check the C
/// engine's `BRAIN_SEG_PER_RECORD_CRC` records carry. Hand-rolled to keep
/// the no-new-dependencies rule; the table is built at compile time.
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// Rolling CRC-32 over a byte stream.
struct Crc32(u32);
impl Crc32 {
    fn new() -> Self {
        Crc32(0xFFFF_FFFF)
    }
    fn update(&mut self, data: &[u8]) {
        let mut c = self.0;
        for &b in data {
            c = CRC32_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
        self.0 = c;
    }
    fn finish(&self) -> u32 {
        !self.0
    }
}

/// CRC-32 of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finish()
}

/// W3.5: the CRC a record carries.
///
/// Covers exactly what a reader consumes and nothing it invents: the
/// explicit `id`, the metadata WITHOUT the transport `id` key (tier,
/// strength, text, role, corecall ids, …) and the VECTOR, in a fixed
/// little-endian layout. Length prefixes make the digest unambiguous (a
/// shifted boundary cannot collide), and the metadata is hashed as its
/// canonical JSON — `serde_json` maps are key-sorted here, so the same
/// record always hashes the same bytes.
pub fn record_crc(id: u32, meta: &serde_json::Value, vec: &[f32]) -> u32 {
    let mut c = Crc32::new();
    c.update(b"sofuu-rec@1");
    c.update(&id.to_le_bytes());
    let text = serde_json::to_vec(meta).unwrap_or_default();
    c.update(&(text.len() as u64).to_le_bytes());
    c.update(&text);
    c.update(&(vec.len() as u64).to_le_bytes());
    for v in vec {
        c.update(&v.to_bits().to_le_bytes());
    }
    c.finish()
}

/// W3.5: per-record verification is ALWAYS on in production. The mutation
/// test flips this off to prove the acceptance tests actually depend on
/// the verification (and not on some accident of re-encoding) — see
/// `mutation_proof_without_record_verification_the_flip_survives`.
#[cfg(test)]
static PER_RECORD_VERIFY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

fn per_record_verify_enabled() -> bool {
    #[cfg(test)]
    {
        PER_RECORD_VERIFY.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        true
    }
}

/// One segment file the manifest references.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SegmentMeta {
    pub file: String,
    pub records: usize,
    /// SHA-256 of the segment file as published. The codec's container read
    /// accepts a TRUNCATED file as a valid, silently-shortened container
    /// (measured 2026-10-13: a segment cut to 1/3 loaded as 6 of 10 records
    /// with no error), so the store carries its own digest and refuses a
    /// segment that does not match — a torn write must never read as a
    /// brain that quietly forgot things. Empty = written before this field
    /// existed (verified by shape only).
    #[serde(default)]
    pub sha256: String,
}

/// The committed manifest — the source of truth for a segmented brain.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub schema: String,
    /// The embedding block, verbatim (id/input/dimension/artifact) — the
    /// same shape a legacy brain's metadata carries, so the loader can
    /// hand downstream code an identical manifest.
    #[serde(default)]
    pub embedding: serde_json::Value,
    pub segments: Vec<SegmentMeta>,
    /// Bumped whenever a compaction rewrites the store; segment names carry
    /// it so a new generation never collides with the old one.
    #[serde(default)]
    pub epoch: u64,
    /// Unix seconds the persisted strengths were current as of. A load
    /// replays the decay that happened while the process was down (capped —
    /// see the loader), so decay never has to be written to disk.
    #[serde(default)]
    pub last_decay_at: u64,
    /// Lifetime count of capacity evictions (reported, never silent).
    #[serde(default)]
    pub evicted_total: u64,
    /// W5.3: the decay policy this brain was flushed under — `Some(tag,
    /// seconds)` once an owner has configured one. Absent on manifests
    /// written before W5.3, which is why it is optional and defaulted: an
    /// old brain opens unchanged, and an open that has nothing to say about
    /// decay inherits whatever was persisted instead of silently reverting
    /// to the default curve (the bug W5.3 exists to fix).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decay_tag: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decay_seconds: Option<f64>,
    /// Live record count as of the last flush — the record table's row
    /// count, tombstones included (the number `count()` reports). Append
    /// segments can carry several VERSIONS of one record, so
    /// `total_versions()` overstates what a reader actually sees.
    /// Manifests written before this field existed carry 0; callers fall
    /// back to `total_versions()`.
    #[serde(default)]
    pub records: u64,
}

impl Manifest {
    pub fn new(embedding: serde_json::Value) -> Self {
        Self {
            schema: SEGMENT_SCHEMA.to_string(),
            embedding,
            segments: Vec::new(),
            epoch: 1,
            last_decay_at: 0,
            evicted_total: 0,
            decay_tag: None,
            decay_seconds: None,
            records: 0,
        }
    }

    pub fn total_versions(&self) -> usize {
        self.segments.iter().map(|s| s.records).sum()
    }
}

/// One record snapshot inside a segment.
#[derive(Clone, Debug)]
pub struct SegRecord {
    pub id: u32,
    pub vec: Vec<f32>,
    /// The record's metadata (tier, strength, text, role, …) — the same
    /// fields the flat brain's RecordJson carries, plus the explicit id.
    pub meta: serde_json::Value,
}

/* ------------------------------------------------------------------ */
/* Paths                                                               */
/* ------------------------------------------------------------------ */

pub fn segments_dir(path: &str) -> String {
    format!("{path}.d")
}

pub fn segment_file(epoch: u64, seq: u64) -> String {
    format!("seg-{epoch:06}-{seq:06}.qtsq")
}

/// W3.1: the persisted HNSW graph snapshot of a generation, written
/// beside the manifest (in the store directory, next to the segments it
/// was built from). Both files of a generation share its epoch, so a
/// compaction leaves the old generation's graph reclaimable by
/// `reclaim_unreferenced`.
pub fn graph_file(generation: u64) -> String {
    format!("graph-{generation:06}.bin")
}

/// Absolute path of `generation`'s graph snapshot for the brain at `path`.
pub fn graph_path(path: &str, generation: u64) -> String {
    format!("{}/{}", segments_dir(path), graph_file(generation))
}

/// Graph snapshots present in `dir`, sorted.
pub fn list_graph_files(dir: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|n| n.starts_with("graph-") && n.ends_with(".bin"))
        .collect();
    out.sort();
    out
}

/// The generation a store file belongs to: `seg-<epoch>-<seq>.qtsq` →
/// `epoch`, `graph-<generation>.bin` → `generation`. `None` for anything
/// else (including a partially-written name), so a sweep never deletes a
/// file whose generation it cannot prove.
pub fn generation_of(name: &str) -> Option<u64> {
    if let Some(body) = name.strip_prefix("seg-").and_then(|b| b.strip_suffix(".qtsq")) {
        let (epoch, seq) = body.split_once('-')?;
        let epoch = epoch.parse::<u64>().ok()?;
        seq.parse::<u64>().ok()?;
        return Some(epoch);
    }
    if let Some(body) = name.strip_prefix("graph-").and_then(|b| b.strip_suffix(".bin")) {
        return body.parse::<u64>().ok();
    }
    None
}

/// Segment files present in `dir`, sorted. Only the naming scheme this
/// module writes is recognised — anything else is left alone.
pub fn list_segment_files(dir: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|n| n.starts_with("seg-") && n.ends_with(".qtsq"))
        .collect();
    out.sort();
    out
}

/* ------------------------------------------------------------------ */
/* Container primitives                                                */
/* ------------------------------------------------------------------ */

/// One stream to write into a container.
pub enum Stream<'a> {
    Tensor {
        name: &'a str,
        data: &'a [f32],
        dims: &'a [u32],
    },
    Json {
        name: &'a str,
        body: &'a str,
    },
}

unsafe fn encrypt(ctx: *mut qtsq::QtsqContext) -> c_int {
    let pw = CString::new(vault_password()).unwrap_or_default();
    qtsq::qtsq_vault_encrypt_password(ctx, pw.as_ptr())
}

/// Decrypt with the machine key, falling back to the legacy fixed password.
/// Returns `Ok(true)` when the LEGACY password opened it — the caller then
/// re-encrypts on the next write. Both failing is `Err`: fail closed, never
/// a half-read brain.
///
/// Retrying on the same context is safe: the codec verifies the HMAC before
/// touching `singularity` (qtsq_vault.c `vault_sym_decrypt`), so a wrong
/// password returns without modifying the ciphertext.
unsafe fn decrypt_with_fallback(ctx: *mut qtsq::QtsqContext) -> Result<bool, c_int> {
    if !(*ctx).is_encrypted() {
        return Ok(false);
    }
    let pw = CString::new(vault_password()).unwrap_or_default();
    if qtsq::qtsq_vault_decrypt_password(ctx, pw.as_ptr()) == qtsq::QTSQ_OK {
        return Ok(false);
    }
    let legacy = CString::new(LEGACY_PASSWORD).unwrap_or_default();
    if qtsq::qtsq_vault_decrypt_password(ctx, legacy.as_ptr()) == qtsq::QTSQ_OK {
        return Ok(true);
    }
    Err(qtsq::QTSQ_ERR_FORMAT)
}

/// Build a container from `streams`, pack → encrypt → write to `path`.
/// Empty streams are skipped (the codec rejects a zero-length payload).
pub unsafe fn container_write(path: &str, streams: &[Stream]) -> Result<(), c_int> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let container = qtsq::qtsq_ctx_alloc();
    if container.is_null() {
        return Err(qtsq::QTSQ_ERR_ALLOC);
    }
    let mut r = qtsq::qtsq_container_create(container);
    if r == qtsq::QTSQ_OK {
        (*container).set_codec(qtsq::QTSQ_CODEC_QTC);
        for s in streams {
            let sub = qtsq::qtsq_ctx_alloc();
            if sub.is_null() {
                r = qtsq::QTSQ_ERR_ALLOC;
                break;
            }
            let init = qtsq::qtsq_init(sub);
            if init != qtsq::QTSQ_OK {
                qtsq::qtsq_ctx_free(sub);
                r = init;
                break;
            }
            (*sub).set_codec(qtsq::QTSQ_CODEC_QTC);
            let (rc, name) = match s {
                Stream::Tensor { name, data, dims } => {
                    if data.is_empty() {
                        qtsq::qtsq_free(sub);
                        qtsq::qtsq_ctx_free(sub);
                        continue;
                    }
                    (
                        qtsq::qtsq_compress_tensor(
                            sub,
                            data.as_ptr(),
                            data.len(),
                            dims.as_ptr(),
                            dims.len() as u8,
                            QTSQ_LEVEL_FULL,
                        ),
                        *name,
                    )
                }
                Stream::Json { name, body } => {
                    if body.is_empty() {
                        qtsq::qtsq_free(sub);
                        qtsq::qtsq_ctx_free(sub);
                        continue;
                    }
                    let cb = CString::new(*body).unwrap_or_default();
                    (
                        qtsq::qtsq_compress_json(sub, cb.as_ptr(), cb.as_bytes().len()),
                        *name,
                    )
                }
            };
            if rc == qtsq::QTSQ_OK {
                let cname = CString::new(name).unwrap_or_default();
                r = qtsq::qtsq_container_add_stream(container, sub, cname.as_ptr());
            } else {
                r = rc;
            }
            qtsq::qtsq_free(sub);
            qtsq::qtsq_ctx_free(sub);
            if r != qtsq::QTSQ_OK {
                break;
            }
        }
    }
    let prof = std::env::var_os("SOFUU_PROFILE").is_some();
    let mut t = std::time::Instant::now();
    if r == qtsq::QTSQ_OK {
        r = qtsq::qtsq_container_pack(container);
    }
    if prof {
        eprintln!("[prof] pack {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        t = std::time::Instant::now();
    }
    if r == qtsq::QTSQ_OK {
        r = encrypt(container);
    }
    if prof {
        eprintln!("[prof] encrypt {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        t = std::time::Instant::now();
    }
    if r == qtsq::QTSQ_OK {
        let cpath = CString::new(path).unwrap_or_default();
        r = qtsq::qtsq_write(container, cpath.as_ptr());
    }
    if prof {
        eprintln!("[prof] write {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    }
    qtsq::qtsq_free(container);
    qtsq::qtsq_ctx_free(container);
    if r == qtsq::QTSQ_OK {
        Ok(())
    } else {
        Err(r)
    }
}

/// Open + decrypt a container. The caller frees it with `container_free`.
pub unsafe fn container_open(path: &str) -> Result<*mut qtsq::QtsqContext, c_int> {
    container_open_ex(path).map(|(ctx, _)| ctx)
}

/// `container_open`, also reporting whether the LEGACY password opened it.
pub unsafe fn container_open_ex(
    path: &str,
) -> Result<(*mut qtsq::QtsqContext, bool), c_int> {
    let cpath = CString::new(path).unwrap_or_default();
    let container = qtsq::qtsq_ctx_alloc();
    if container.is_null() {
        return Err(qtsq::QTSQ_ERR_ALLOC);
    }
    let mut r = qtsq::qtsq_init(container);
    if r == qtsq::QTSQ_OK {
        r = qtsq::qtsq_read(container, cpath.as_ptr());
    }
    if r != qtsq::QTSQ_OK {
        /* The bytes are not a QTSQ file — a foreign file the caller may
         * still want to back up and replace (the corrupt-brain path). */
        qtsq::qtsq_free(container);
        qtsq::qtsq_ctx_free(container);
        return Err(ERR_NOT_CONTAINER);
    }
    let mut legacy = false;
    match decrypt_with_fallback(container) {
        Ok(l) => legacy = l,
        Err(e) => r = e,
    }
    if r == qtsq::QTSQ_OK && (*container).data_type() != qtsq::QTSQ_TYPE_CONTAINER {
        r = ERR_NOT_CONTAINER;
    }
    if r != qtsq::QTSQ_OK {
        qtsq::qtsq_free(container);
        qtsq::qtsq_ctx_free(container);
        return Err(r);
    }
    Ok((container, legacy))
}

pub unsafe fn container_free(container: *mut qtsq::QtsqContext) {
    qtsq::qtsq_free(container);
    qtsq::qtsq_ctx_free(container);
}

/// Read one named stream as JSON text.
pub unsafe fn read_json_stream(
    container: *mut qtsq::QtsqContext,
    want: &str,
) -> Option<String> {
    let sub = find_stream(container, want)?;
    let mut data: *mut u8 = ptr::null_mut();
    let mut size: usize = 0;
    let r = qtsq::qtsq_decompress_horizon(sub, &mut data, &mut size);
    let out = if r == qtsq::QTSQ_OK && !data.is_null() {
        let bytes = std::slice::from_raw_parts(data, size).to_vec();
        String::from_utf8(bytes).ok()
    } else {
        None
    };
    if !data.is_null() {
        libc::free(data as *mut c_void);
    }
    container_free(sub);
    out
}

/// Read one named stream as an f32 tensor plus its schema dims.
pub unsafe fn read_tensor_stream(
    container: *mut qtsq::QtsqContext,
    want: &str,
) -> Option<(Vec<f32>, Vec<u32>)> {
    let sub = find_stream(container, want)?;
    let mut data: *mut f32 = ptr::null_mut();
    let mut count: usize = 0;
    let r = qtsq::qtsq_decompress_tensor(sub, &mut data, &mut count);
    let mut out = None;
    if r == qtsq::QTSQ_OK && !data.is_null() {
        let vals = std::slice::from_raw_parts(data, count).to_vec();
        let nd = (*sub).schema_num_dims();
        let dims: Vec<u32> = (0..nd as usize).map(|i| (*sub).schema_dim(i)).collect();
        out = Some((vals, dims));
    }
    if !data.is_null() {
        libc::free(data as *mut c_void);
    }
    container_free(sub);
    out
}

/// Extract the stream named `want` into a fresh context (caller frees).
unsafe fn find_stream(
    container: *mut qtsq::QtsqContext,
    want: &str,
) -> Option<*mut qtsq::QtsqContext> {
    let mut num: c_uint = 0;
    if qtsq::qtsq_container_get_count(container, &mut num) != qtsq::QTSQ_OK {
        return None;
    }
    let want_c = CString::new(want).unwrap_or_default();
    for i in 0..num {
        let sub = qtsq::qtsq_ctx_alloc();
        if sub.is_null() {
            return None;
        }
        let mut name_buf = [0u8; 256];
        let rc = qtsq::qtsq_container_get_stream(
            container,
            i,
            sub,
            name_buf.as_mut_ptr() as *mut c_char,
            name_buf.len(),
        );
        if rc == qtsq::QTSQ_OK {
            let name = CStr::from_ptr(name_buf.as_ptr() as *const c_char).to_bytes();
            if name == want_c.as_bytes() {
                return Some(sub);
            }
        }
        qtsq::qtsq_ctx_free(sub);
    }
    None
}

/// Digest of a published file (None when unreadable).
pub fn file_digest(path: &str) -> Option<String> {
    crate::rt::wal::sha256_file(path)
}

/// The stream names a container carries (diagnostics + shape checks).
pub unsafe fn stream_names(container: *mut qtsq::QtsqContext) -> Vec<String> {
    let mut out = Vec::new();
    let mut num: c_uint = 0;
    if qtsq::qtsq_container_get_count(container, &mut num) != qtsq::QTSQ_OK {
        return out;
    }
    for i in 0..num {
        let sub = qtsq::qtsq_ctx_alloc();
        if sub.is_null() {
            break;
        }
        let mut name_buf = [0u8; 256];
        if qtsq::qtsq_container_get_stream(
            container,
            i,
            sub,
            name_buf.as_mut_ptr() as *mut c_char,
            name_buf.len(),
        ) == qtsq::QTSQ_OK
        {
            if let Ok(s) = CStr::from_ptr(name_buf.as_ptr() as *const c_char).to_str() {
                out.push(s.to_string());
            }
        }
        qtsq::qtsq_ctx_free(sub);
    }
    out
}

/* ------------------------------------------------------------------ */
/* Segments + manifest                                                 */
/* ------------------------------------------------------------------ */

/// Pack ONE segment container: tensor of the records' vectors + their
/// metadata, plus — when `flags` carries `SEG_FLAG_PER_RECORD_CRC` — the
/// header flag, the exact tensor length and the per-record CRC table, and
/// the truncation trailer.
///
/// `crcs` must be parallel to `recs` when the flag is set. With
/// `flags == 0` the bytes produced are EXACTLY what the pre-W3.5 writer
/// produced (no schema key, no CRC table, no trailer), which is what the
/// backward-compatibility test builds and what old builders expect.
unsafe fn pack_segment(
    path: &str,
    dim: usize,
    recs: &[SegRecord],
    crcs: Option<&[u32]>,
    flags: u32,
) -> Result<(), c_int> {
    let mut flat: Vec<f32> = Vec::with_capacity(recs.len() * dim);
    let mut metas: Vec<serde_json::Value> = Vec::with_capacity(recs.len());
    for r in recs {
        if r.vec.len() != dim {
            return Err(qtsq::QTSQ_ERR_FORMAT);
        }
        flat.extend_from_slice(&r.vec);
        let mut m = r.meta.clone();
        if let Some(o) = m.as_object_mut() {
            o.insert("id".into(), serde_json::json!(r.id));
        }
        metas.push(m);
    }
    let dims: [u32; 2] = [recs.len() as u32, dim as u32];
    let mut root = serde_json::Map::new();
    root.insert("dim".into(), serde_json::json!(dim));
    root.insert("records".into(), serde_json::Value::Array(metas));
    if flags & SEG_FLAG_PER_RECORD_CRC != 0 {
        let Some(crcs) = crcs else {
            return Err(qtsq::QTSQ_ERR_FORMAT);
        };
        if crcs.len() != recs.len() {
            return Err(qtsq::QTSQ_ERR_FORMAT);
        }
        root.insert("schema".into(), serde_json::json!(SEGMENT_RECORD_SCHEMA));
        root.insert("flags".into(), serde_json::json!(flags));
        /* The exact number of tensor components the writer put in the
         * file: a segment whose tensor comes back shorter lost payload,
         * which is a whole-segment fault, never a record-level one. */
        root.insert("tlen".into(), serde_json::json!(flat.len()));
        root.insert(
            "crcs".into(),
            serde_json::Value::Array(crcs.iter().map(|c| serde_json::json!(c)).collect()),
        );
    } else if crcs.is_some() {
        /* CRCs without the flag would be dead weight no reader checks. */
        return Err(qtsq::QTSQ_ERR_FORMAT);
    }
    let meta_json = serde_json::Value::Object(root).to_string();
    /* A re-pack in place (the two-phase write below) must not inherit a
     * longer previous image: the trailer records the exact length. */
    let _ = std::fs::remove_file(path);
    container_write(
        path,
        &[
            Stream::Tensor {
                name: SEGMENT_TENSOR_STREAM,
                data: &flat,
                dims: &dims,
            },
            Stream::Json {
                name: SEGMENT_META_STREAM,
                body: &meta_json,
            },
        ],
    )?;
    if flags & SEG_FLAG_PER_RECORD_CRC != 0 {
        append_trailer(path, flags)?;
    }
    Ok(())
}

/// Write one segment: tensor of the records' vectors + their metadata.
///
/// W3.5: new segments carry a per-record CRC over id + metadata + vector,
/// so a reader can drop one damaged record instead of the whole segment.
///
/// The CRC has to cover the bytes a READER will hash, and this container's
/// tensor transform is not the identity on every input (measured
/// 2026-10-30: a `0.0` comes back as 2^-62, and a vector with a wide
/// dynamic range biases its small components), so the write is
/// deliberately two-phase:
///
///   1. pack with CRCs taken from the caller's vectors, decode the file
///      back, and stop when the codec reproduced the input exactly (the
///      common case for real embeddings);
///   2. otherwise pack AGAIN with the SAME vectors and CRCs taken from the
///      decoded ones. The tensor input is byte-identical to phase 1's, so
///      the file a reader decodes is phase 1's decode — the CRCs describe
///      exactly what it will hash. (Re-packing the DECODED vectors instead
///      would apply the transform a second time and drift: measured on a
///      256-record segment, the codec is not idempotent at that size.)
pub unsafe fn write_segment(
    path: &str,
    dim: usize,
    recs: &[SegRecord],
) -> Result<(), c_int> {
    let crcs: Vec<u32> = recs
        .iter()
        .map(|r| record_crc(r.id, &r.meta, &r.vec))
        .collect();
    pack_segment(path, dim, recs, Some(&crcs), SEG_FLAG_PER_RECORD_CRC)?;
    /* Learn what a reader will actually decode. */
    let decoded = open_segment_raw(path)?.recs;
    if records_bit_identical(recs, &decoded) {
        /* The codec was exact: phase 1's CRCs already describe it. */
        return Ok(());
    }
    let crcs: Vec<u32> = decoded
        .iter()
        .map(|r| record_crc(r.id, &r.meta, &r.vec))
        .collect();
    pack_segment(path, dim, recs, Some(&crcs), SEG_FLAG_PER_RECORD_CRC)
}

/// True when two record slices are the same records bit for bit: same
/// order, ids, metadata and VECTOR BITS.
fn records_bit_identical(a: &[SegRecord], b: &[SegRecord]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| {
            x.id == y.id
                && x.meta == y.meta
                && x.vec.len() == y.vec.len()
                && x.vec.iter().zip(y.vec.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

/// A segment as it sits on disk: records in file order (metadata WITHOUT
/// the transport `id`), plus the header flag and the CRC table when the
/// segment carries one. NO per-record verification happens here — this is
/// both the writer's read-back and the verified reader's first half.
struct RawSegment {
    recs: Vec<SegRecord>,
    crcs: Option<Vec<u32>>,
    flags: u32,
    legacy: bool,
}

/// Open and decode one segment, enforcing every STRUCTURAL rule: the
/// container must open, the metadata must parse, the tensor shape must
/// agree with the record count, a flagged segment must carry a valid
/// trailer, an unshortened tensor (`tlen`) and a CRC table of exactly one
/// entry per record.
unsafe fn open_segment_raw(path: &str) -> Result<RawSegment, c_int> {
    let (container, legacy) = container_open_ex(path)?;
    let meta_json = read_json_stream(container, SEGMENT_META_STREAM);
    let tensor = read_tensor_stream(container, SEGMENT_TENSOR_STREAM);
    container_free(container);

    let Some(meta_json) = meta_json else {
        return Err(qtsq::QTSQ_ERR_FORMAT);
    };
    let Ok(root) = serde_json::from_str::<serde_json::Value>(&meta_json) else {
        return Err(qtsq::QTSQ_ERR_FORMAT);
    };
    let dim = root.get("dim").and_then(|d| d.as_u64()).unwrap_or(0) as usize;
    let flags = root.get("flags").and_then(|f| f.as_u64()).unwrap_or(0) as u32;
    let Some(arr) = root.get("records").and_then(|r| r.as_array()) else {
        return Err(qtsq::QTSQ_ERR_FORMAT);
    };
    if dim == 0 {
        return Err(qtsq::QTSQ_ERR_FORMAT);
    }
    let (vals, dims) = match tensor {
        Some(t) => t,
        None if arr.is_empty() => (Vec::new(), vec![0, dim as u32]),
        None => return Err(qtsq::QTSQ_ERR_FORMAT),
    };
    /* The tensor's declared shape must agree with the metadata count —
     * a hostile/corrupt segment must not hand misaligned vectors out. */
    if dims.len() >= 2 {
        let (n, d) = (dims[0] as usize, dims[1] as usize);
        if d != dim || n != arr.len() || vals.len() != n * d {
            return Err(qtsq::QTSQ_ERR_FORMAT);
        }
    } else if vals.len() != arr.len() * dim {
        return Err(qtsq::QTSQ_ERR_FORMAT);
    }
    /* W3.5: a flagged segment also declares its exact tensor length, so a
     * container that decoded "successfully" but short is refused. */
    if flags & SEG_FLAG_PER_RECORD_CRC != 0 {
        let tlen = root.get("tlen").and_then(|t| t.as_u64()).unwrap_or(u64::MAX);
        if tlen != vals.len() as u64 {
            return Err(qtsq::QTSQ_ERR_FORMAT);
        }
        /* A half-written segment is not a record-level fault: refuse the
         * whole file when the trailer that pins its length is missing. */
        if !segment_trailer(path)? {
            return Err(qtsq::QTSQ_ERR_FORMAT);
        }
    }

    let mut out = Vec::with_capacity(arr.len());
    for (i, m) in arr.iter().enumerate() {
        let Some(id) = m.get("id").and_then(|v| v.as_u64()) else {
            return Err(qtsq::QTSQ_ERR_FORMAT);
        };
        let mut meta = m.clone();
        if let Some(o) = meta.as_object_mut() {
            o.remove("id");
        }
        out.push(SegRecord {
            id: id as u32,
            vec: vals[i * dim..(i + 1) * dim].to_vec(),
            meta,
        });
    }
    let crcs = if flags & SEG_FLAG_PER_RECORD_CRC != 0 {
        let Some(list) = root.get("crcs").and_then(|c| c.as_array()) else {
            return Err(qtsq::QTSQ_ERR_FORMAT);
        };
        if list.len() != out.len() {
            return Err(qtsq::QTSQ_ERR_FORMAT);
        }
        let mut crcs = Vec::with_capacity(list.len());
        for v in list {
            let Some(c) = v.as_u64() else {
                return Err(qtsq::QTSQ_ERR_FORMAT);
            };
            crcs.push(c as u32);
        }
        Some(crcs)
    } else {
        None
    };
    Ok(RawSegment {
        recs: out,
        crcs,
        flags,
        legacy,
    })
}

/// Read one segment back. Records come out in file order.
pub unsafe fn read_segment(path: &str) -> Result<Vec<SegRecord>, c_int> {
    read_segment_ex(path).map(|(recs, _)| recs)
}

/// `read_segment`, also reporting whether the LEGACY password opened it.
pub unsafe fn read_segment_ex(
    path: &str,
) -> Result<(Vec<SegRecord>, bool), c_int> {
    read_segment_report(path).map(|(read, legacy)| (read.records, legacy))
}

/// W3.5: the full read report — the records that verified, and how many
/// were dropped because their per-record CRC did not.
///
/// This is the function the store's recovery accounting should call: its
/// `dropped` count is what belongs in `records_dropped`/`RepairReport`
/// alongside the whole-segment losses. `read_segment`/`read_segment_ex`
/// stay as they were for callers that only need the records.
pub unsafe fn read_segment_report(
    path: &str,
) -> Result<(SegmentRead, bool), c_int> {
    let raw = open_segment_raw(path)?;
    let legacy = raw.legacy;
    let Some(crcs) = raw.crcs else {
        /* Pre-W3.5 segment: the whole-segment contract, untouched. */
        return Ok((
            SegmentRead {
                records: raw.recs,
                dropped: 0,
            },
            legacy,
        ));
    };
    if !per_record_verify_enabled() {
        /* Mutation hook (see PER_RECORD_VERIFY): behave exactly as the
         * pre-W3.5 reader did and hand every record out. */
        return Ok((
            SegmentRead {
                records: raw.recs,
                dropped: 0,
            },
            legacy,
        ));
    }
    let total = raw.recs.len();
    let mut records = Vec::with_capacity(total);
    let mut dropped = 0usize;
    for (r, want) in raw.recs.into_iter().zip(crcs) {
        let got = record_crc(r.id, &r.meta, &r.vec);
        if got == want {
            records.push(r);
        } else {
            dropped += 1;
            eprintln!(
                "[sofuu/cma] RECORD DAMAGE: {} record id {} failed its CRC (stored {want:08x}, found {got:08x}) — dropping that record only, keeping the other {}",
                path,
                r.id,
                records.len()
            );
        }
    }
    debug_assert_eq!(raw.flags & SEG_FLAG_PER_RECORD_CRC, SEG_FLAG_PER_RECORD_CRC);
    Ok((SegmentRead { records, dropped }, legacy))
}

/// Write the manifest container (caller publishes it atomically).
pub unsafe fn write_manifest(path: &str, m: &Manifest) -> Result<(), c_int> {
    let body = serde_json::to_string(m).map_err(|_| qtsq::QTSQ_ERR_FORMAT)?;
    container_write(
        path,
        &[Stream::Json {
            name: MANIFEST_STREAM,
            body: &body,
        }],
    )
}

/// Read a manifest, or None when `path` is missing, unreadable, or a LEGACY
/// brain (which carries memories/metadata streams instead).
pub unsafe fn read_manifest(path: &str) -> Option<Manifest> {
    read_manifest_ex(path).ok().flatten().map(|(m, _)| m)
}

/// `read_manifest`, also reporting whether the LEGACY password opened it.
///
/// `Ok(None)` means "this is not a segmented brain" (no manifest stream, or
/// a foreign schema). `Err` means "this IS a container we could not open" —
/// a wrong key, a torn file — which the caller MUST NOT treat as an absent
/// manifest: falling through to the single-file reader would open a
/// wrong-key brain as an EMPTY one, silently forgetting everything.
pub unsafe fn read_manifest_ex(
    path: &str,
) -> Result<Option<(Manifest, bool)>, c_int> {
    let (container, legacy) = match container_open_ex(path) {
        Ok(v) => v,
        /* Not a container: the caller's single-file reader owns this file
         * (backup + fresh start). A decrypt failure stays an Err. */
        Err(e) if e == ERR_NOT_CONTAINER => return Ok(None),
        Err(e) => return Err(e),
    };
    let body = read_json_stream(container, MANIFEST_STREAM);
    container_free(container);
    let Some(body) = body else {
        return Ok(None);
    };
    let Ok(m) = serde_json::from_str::<Manifest>(&body) else {
        return Ok(None);
    };
    if m.schema != SEGMENT_SCHEMA {
        return Ok(None);
    }
    Ok(Some((m, legacy)))
}

/// Delete segment files the manifest does not reference — the leftovers a
/// crashed compaction leaves behind. Age-guarded: a file younger than
/// `min_age_secs` may belong to a concurrent writer mid-commit.
pub fn remove_orphan_segments(dir: &str, keep: &[String], min_age_secs: u64) -> usize {
    let mut removed = 0;
    for name in list_segment_files(dir) {
        if keep.iter().any(|k| k == &name) {
            continue;
        }
        let path = format!("{dir}/{name}");
        let Ok(md) = std::fs::metadata(&path) else {
            continue;
        };
        let Ok(modified) = md.modified() else {
            continue;
        };
        let Ok(age) = std::time::SystemTime::now().duration_since(modified) else {
            continue;
        };
        if age.as_secs() >= min_age_secs && std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

/* ------------------------------------------------------------------ */
/* W3.4: generation reclamation                                        */
/* ------------------------------------------------------------------ */

/// Default age guard for `reclaim_unreferenced`: a store file younger than
/// this may belong to a concurrent writer that has published its bytes but
/// not yet its manifest (`remove_orphan_segments` uses the same guard).
pub const RECLAIM_MIN_AGE_SECS: u64 = 600;

/// Total bytes of the store files in `dir` — the numerator of the disk
/// amplification the reclaim primitive exists to bound.
pub fn dir_bytes(dir: &str) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// W3.4: delete the segments AND graph snapshots of every generation the
/// caller no longer references — the storage-level primitive behind the
/// steady state "disk amplification stays bounded while the store keeps
/// compacting".
///
/// `keep` is the set of generations the committed manifest still points
/// at: the current `epoch` plus the epoch of any segment it still lists
/// (`generation_of(name)` recovers it from a file name). Anything in `dir`
/// that belongs to another generation is superseded by construction — a
/// compaction bumps the epoch, and no reader ever resolves an old file
/// through a new manifest.
///
/// Age-guarded (`RECLAIM_MIN_AGE_SECS`): a concurrent writer's fresh files
/// have a newer epoch than the manifest it is about to publish, so they
/// must survive this sweep. Files whose generation cannot be parsed are
/// left strictly alone. Returns the number of files removed.
pub fn reclaim_unreferenced(dir: &str, keep: &[u64]) -> usize {
    reclaim_unreferenced_now(dir, keep, RECLAIM_MIN_AGE_SECS)
}

/// `reclaim_unreferenced` with an explicit age guard (0 = "sweep now",
/// which is what the tests and an offline repair want).
pub fn reclaim_unreferenced_now(dir: &str, keep: &[u64], min_age_secs: u64) -> usize {
    let mut names: Vec<String> = list_segment_files(dir);
    names.extend(list_graph_files(dir));
    let mut removed = 0;
    for name in names {
        let Some(generation) = generation_of(&name) else {
            continue;
        };
        if keep.contains(&generation) {
            continue;
        }
        let path = format!("{dir}/{name}");
        let Ok(md) = std::fs::metadata(&path) else {
            continue;
        };
        let Ok(modified) = md.modified() else {
            continue;
        };
        let Ok(age) = std::time::SystemTime::now().duration_since(modified) else {
            continue;
        };
        if age.as_secs() >= min_age_secs && std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("sofuu-brainstore-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn rec(id: u32, dim: usize, text: &str) -> SegRecord {
        let mut v = vec![0.0f32; dim];
        v[(id as usize) % dim] = 1.0;
        SegRecord {
            id,
            vec: v,
            meta: serde_json::json!({
                "tier": 1, "strength": 1.0, "text": text, "role": "user",
            }),
        }
    }

    #[test]
    fn segment_round_trips_records_and_vectors() {
        let dir = tmpdir("seg");
        let path = dir.join("seg-000001-000001.qtsq");
        let recs = vec![rec(0, 8, "alpha"), rec(7, 8, "beta")];
        unsafe {
            write_segment(&path.to_string_lossy(), 8, &recs).expect("write");
            let back = read_segment(&path.to_string_lossy()).expect("read");
            assert_eq!(back.len(), 2);
            assert_eq!(back[0].id, 0);
            assert_eq!(back[1].id, 7, "ids are explicit, not positional");
            assert_eq!(back[0].meta["text"], "alpha");
            /* The codec's tensor transform is exact for ordinary values but
             * leaves ~2e-19 noise where the input was 0.0 — irrelevant for
             * embedding geometry (cosine shifts by ~0), so compare with a
             * tolerance rather than bit equality. */
            for (a, b) in back[0].vec.iter().zip(recs[0].vec.iter()) {
                assert!((a - b).abs() < 1e-6, "{a} vs {b}");
            }
            for (a, b) in back[1].vec.iter().zip(recs[1].vec.iter()) {
                assert!((a - b).abs() < 1e-6, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn manifest_round_trips() {
        let dir = tmpdir("manifest");
        let path = dir.join("brain.qtsq");
        let mut m = Manifest::new(serde_json::json!({"id": "hash-v1", "dimension": 768}));
        m.segments.push(SegmentMeta {
            file: "seg-000001-000001.qtsq".into(),
            records: 3,
            sha256: "ab".repeat(32),
        });
        m.last_decay_at = 1234;
        m.evicted_total = 9;
        m.records = 7;
        unsafe {
            write_manifest(&path.to_string_lossy(), &m).expect("write");
            let back = read_manifest(&path.to_string_lossy()).expect("read");
            assert_eq!(back.schema, SEGMENT_SCHEMA);
            assert_eq!(back.segments.len(), 1);
            assert_eq!(back.segments[0].records, 3);
            assert_eq!(back.segments[0].sha256, "ab".repeat(32));
            assert_eq!(back.last_decay_at, 1234);
            assert_eq!(back.evicted_total, 9);
            assert_eq!(back.records, 7);
            assert_eq!(back.embedding["id"], "hash-v1");
        }
    }

    /// A manifest written before `records` existed (no such key on disk)
    /// must still load — the field defaults to 0 and callers fall back to
    /// `total_versions()`.
    #[test]
    fn manifest_without_records_field_loads() {
        let dir = tmpdir("manifest-legacy-field");
        let path = dir.join("brain.qtsq");
        let raw = r#"{"schema":"sofuu-cma-segments@1","embedding":{"id":"hash-v1","dimension":768},"segments":[{"file":"seg-000001-000001.qtsq","records":4}],"epoch":1}"#;
        unsafe {
            container_write(
                &path.to_string_lossy(),
                &[Stream::Json {
                    name: MANIFEST_STREAM,
                    body: raw,
                }],
            )
            .expect("write");
            let mut back = read_manifest(&path.to_string_lossy()).expect("read");
            assert_eq!(back.segments[0].records, 4);
            assert_eq!(back.records, 0, "the key was absent on disk → 0");
            back.records = back.total_versions() as u64;
            assert_eq!(back.records, 4, "fallback is total_versions");
        }
    }

    #[test]
    fn read_manifest_is_none_for_a_legacy_brain() {
        /* A legacy brain carries memories+metadata streams, no manifest. */
        let dir = tmpdir("legacy");
        let path = dir.join("brain.qtsq");
        unsafe {
            let vals = [1.0f32, 2.0, 3.0, 4.0];
            let dims = [1u32, 4];
            container_write(
                &path.to_string_lossy(),
                &[
                    Stream::Tensor { name: "memories", data: &vals, dims: &dims },
                    Stream::Json { name: "metadata", body: "{\"records\":[]}" },
                ],
            )
            .expect("legacy write");
            assert!(read_manifest(&path.to_string_lossy()).is_none());
        }
    }

    #[test]
    fn read_manifest_is_none_for_a_missing_file() {
        let dir = tmpdir("missing");
        let path = dir.join("nope.qtsq");
        unsafe {
            assert!(read_manifest(&path.to_string_lossy()).is_none());
        }
    }

    #[test]
    fn orphan_sweep_keeps_referenced_and_fresh_files() {
        let dir = tmpdir("orphans");
        let d = dir.to_string_lossy().to_string();
        let keep = "seg-000001-000001.qtsq".to_string();
        let stale = "seg-000001-000002.qtsq".to_string();
        std::fs::write(format!("{d}/{keep}"), b"x").unwrap();
        std::fs::write(format!("{d}/{stale}"), b"x").unwrap();
        /* Fresh orphan: must survive (a concurrent writer may hold it). */
        assert_eq!(remove_orphan_segments(&d, &[keep.clone()], 600), 0);
        assert!(std::path::Path::new(&format!("{d}/{stale}")).exists());
        /* Age it past the guard → swept. */
        let old = std::time::SystemTime::now()
            - std::time::Duration::from_secs(700);
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(format!("{d}/{stale}"))
            .unwrap();
        f.set_modified(old).unwrap();
        assert_eq!(remove_orphan_segments(&d, &[keep.clone()], 600), 1);
        assert!(!std::path::Path::new(&format!("{d}/{stale}")).exists());
        assert!(std::path::Path::new(&format!("{d}/{keep}")).exists());
    }

    #[test]
    fn segment_write_rejects_a_short_vector() {
        let dir = tmpdir("short");
        let path = dir.join("seg.qtsq");
        let bad = SegRecord {
            id: 0,
            vec: vec![1.0, 2.0],
            meta: serde_json::json!({"text": "x"}),
        };
        unsafe {
            assert!(write_segment(&path.to_string_lossy(), 8, &[bad]).is_err());
        }
    }

    /* ══════════════════════════════════════════════════════════════
     * W3.5: per-record integrity
     * ══════════════════════════════════════════════════════════════ */

    /// Deterministic "embedding-like" vector (values in [-0.5, 0.5)).
    fn embedding(id: u32, dim: usize) -> Vec<f32> {
        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        (0..dim)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32) / 16_777_216.0 - 0.5
            })
            .collect()
    }

    /// A record with a realistic (codec-exact) vector.
    fn rec_emb(id: u32, dim: usize) -> SegRecord {
        SegRecord {
            id,
            vec: embedding(id, dim),
            meta: serde_json::json!({
                "tier": 1, "strength": 1.0 - (id % 7) as f64 / 10.0,
                "text": format!("record {id}"), "role": "user",
            }),
        }
    }

    /// A record whose vector the codec does NOT reproduce bit-exactly (zeros
    /// come back as 2^-62), so the two-phase write has to re-derive the CRCs
    /// from the decoded values.
    fn rec_sparse(id: u32, dim: usize) -> SegRecord {
        let mut v = vec![0.0f32; dim];
        v[(id as usize * 3) % dim] = 1.0;
        SegRecord {
            id,
            vec: v,
            meta: serde_json::json!({
                "tier": 0, "strength": 0.5, "text": format!("sparse {id}"), "role": "assistant",
            }),
        }
    }

    /// 512 records in 2 segments of SEGMENT_CAP — DENSE embedding-like
    /// vectors, the production shape (see `w3_5_zero_heavy_vectors_...`
    /// for what a zero-heavy segment does differently).
    ///
    /// The records are returned so the tamper helper can re-pack EXACTLY
    /// the vectors that were written (see `tamper`).
    fn two_segments(
        dir: &std::path::Path,
        dim: usize,
    ) -> (String, String, Vec<SegRecord>, Vec<SegRecord>) {
        let recs: Vec<SegRecord> = (0..512u32).map(|i| rec_emb(i, dim)).collect();
        let a = dir.join(segment_file(1, 1)).to_string_lossy().to_string();
        let b = dir.join(segment_file(1, 2)).to_string_lossy().to_string();
        assert_eq!(recs.len(), 2 * SEGMENT_CAP);
        unsafe {
            write_segment(&a, dim, &recs[..SEGMENT_CAP]).expect("segment 1");
            write_segment(&b, dim, &recs[SEGMENT_CAP..]).expect("segment 2");
        }
        let (first, second) = recs.split_at(SEGMENT_CAP);
        (a, b, first.to_vec(), second.to_vec())
    }

    /// Simulate ONE DAMAGED RECORD ON DISK: re-encode the segment from the
    /// very vectors that were written, with ONE record's payload changed
    /// and that record's ORIGINAL CRC kept — which is what a reader sees
    /// when the bytes it decodes no longer match the CRC published with
    /// them.
    ///
    /// Re-packing the ORIGINALS (not the decoded values) matters: this
    /// container's tensor transform is not idempotent at segment size, so
    /// re-encoding decoded vectors would perturb every record and the test
    /// would be measuring codec drift, not one damaged record. The helper
    /// asserts that exactly one record comes back changed.
    ///
    /// A raw file-byte flip is not a usable seam here: the container is
    /// authenticated AND entropy-coded, so flipping a file byte breaks the
    /// container before any record exists (that is the whole-segment fault)
    /// and the record layer never sees it. The fault the per-record CRC
    /// exists for is the one that reaches the record layer — bit rot that
    /// still decodes, a codec/adapter bug, a writer without the key, a
    /// segment adopted against a stale manifest digest — so the payload is
    /// what the test damages.
    fn tamper(path: &str, dim: usize, originals: &[SegRecord], at: usize, what: Tamper) {
        unsafe {
            let raw = open_segment_raw(path).expect("decode for tamper");
            let crcs = raw.crcs.clone().expect("segment carries per-record CRCs");
            let mut recs = originals.to_vec();
            match what {
                Tamper::VectorBit(bit) => {
                    let v = &mut recs[at].vec;
                    v[3] = f32::from_bits(v[3].to_bits() ^ (1 << bit));
                }
                Tamper::TextByte => {
                    let o = recs[at]
                        .meta
                        .as_object_mut()
                        .expect("record metadata is an object");
                    o.insert("text".into(), serde_json::json!("rEcord with one flipped byte"));
                }
            }
            pack_segment(path, dim, &recs, Some(&crcs), SEG_FLAG_PER_RECORD_CRC)
                .expect("repack the damaged segment");
            /* The damage must be confined to that one record after the codec
             * round trip, or the test would be measuring nothing (or
             * everything). The baseline is the ORIGINAL FILE'S DECODE — the
             * codec is lossy, so the pre-encode inputs are not what a reader
             * ever saw. */
            let after = open_segment_raw(path).expect("re-read");
            let changed: Vec<usize> = after
                .recs
                .iter()
                .zip(raw.recs.iter())
                .enumerate()
                .filter(|(_, (got, want))| {
                    got.id != want.id
                        || got.meta != want.meta
                        || got.vec.iter().zip(want.vec.iter()).any(|(a, b)| a.to_bits() != b.to_bits())
                })
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                changed,
                vec![at],
                "harness: exactly the damaged record must differ on disk"
            );
        }
    }

    #[derive(Clone, Copy)]
    enum Tamper {
        VectorBit(u32),
        TextByte,
    }

    /// (a) A flipped bit inside one record's VECTOR payload, and (b) a
    /// flipped TEXT byte, each cost exactly that record: 511 of 512 records
    /// survive, one is dropped, and the damaged segment still LOADS.
    ///
    /// WHAT "A BIT INSIDE THE PAYLOAD" CAN MEAN HERE, exactly. The segment
    /// container authenticates and entropy-codes its payload, so a raw flip
    /// of a file byte is caught by the container's own checksum and the
    /// segment is refused whole (pinned separately in
    /// `w3_5_a_raw_bit_flip_in_the_sealed_payload_is_refused_whole`). The
    /// damage this CRC layer exists for is the kind that REACHES the record
    /// layer — a record whose stored bytes no longer match the CRC
    /// published with it — which `tamper` reproduces by re-encoding the
    /// very same vectors with one record's value changed.
    ///
    /// Granularity note: the tensor codec derives a tensor-wide step, so a
    /// perturbed component can move OTHER records' reconstructed bytes too
    /// (measured: 1 of 256 for this dense fixture, 156 of 256 for a
    /// zero-heavy one — see `w3_5_zero_heavy_vectors_...`). Where it is not
    /// confined, the CRC drops every record whose bytes actually changed and
    /// the accounting stays complete; here it is confined, and `tamper`
    /// asserts that before the assertions below run.
    #[test]
    fn w3_5_one_damaged_vector_or_text_record_costs_only_that_record() {
        let dim = 64usize;
        for (label, what) in [
            ("vector", Tamper::VectorBit(9)),
            ("text", Tamper::TextByte),
        ] {
            let dir = tmpdir(&format!("w35-{label}"));
            let (a, b, first, _second) = two_segments(&dir, dim);
            /* Every record verifies before the damage. */
            unsafe {
                let (ok_a, _) = read_segment_report(&a).expect("segment 1 reads");
                let (ok_b, _) = read_segment_report(&b).expect("segment 2 reads");
                assert_eq!((ok_a.records.len(), ok_a.dropped), (SEGMENT_CAP, 0));
                assert_eq!((ok_b.records.len(), ok_b.dropped), (SEGMENT_CAP, 0));
            }

            /* Damage record 100 of the FIRST segment. `tamper` also proves
             * the damage is confined to that record on disk. */
            tamper(&a, dim, &first, 100, what);

            unsafe {
                let (two, _) = read_segment_report(&b).expect("segment 2 still reads");
                assert_eq!(two.records.len(), SEGMENT_CAP, "segment 2 is untouched");
                assert_eq!(two.dropped, 0);

                /* The damaged segment must LOAD — not be refused, not be
                 * quarantined, and not be silently shortened. */
                let (one, _) = read_segment_report(&a)
                    .expect("a damaged record must NOT take the whole segment with it");
                assert_eq!(one.records.len(), SEGMENT_CAP - 1, "{label}: all but one survive");
                assert_eq!(one.dropped, 1, "{label}: exactly one record is dropped");

                let live = one.records.len() + two.records.len();
                assert_eq!(live, 511, "{label}: 511 live records");
                assert_eq!(one.dropped + two.dropped, 1, "{label}: exactly 1 dropped");

                /* The surviving records are the RIGHT ones: the damaged id
                 * is gone, every other id of both segments is present. */
                let victim = 100u32;
                let mut ids: Vec<u32> = one
                    .records
                    .iter()
                    .chain(two.records.iter())
                    .map(|r| r.id)
                    .collect();
                ids.sort_unstable();
                assert!(!ids.contains(&victim), "{label}: the damaged record is dropped");
                assert_eq!(ids.len(), 511);
                assert!(
                    ids.iter().enumerate().all(|(i, id)| *id as usize
                        == if i < victim as usize { i } else { i + 1 }),
                    "{label}: every other record kept its id"
                );
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// The container is the FIRST line of integrity: a raw bit flip anywhere
    /// in the sealed payload is refused as a whole segment, never repaired
    /// record by record. (This is why the record-level seam above re-encodes
    /// instead of flipping a file byte.)
    #[test]
    fn w3_5_a_raw_bit_flip_in_the_sealed_payload_is_refused_whole() {
        let dim = 64usize;
        let dir = tmpdir("w35-rawflip");
        let (a, _b, _, _) = two_segments(&dir, dim);
        let bytes = std::fs::read(&a).expect("read");
        for frac in [10usize, 50, 90] {
            let mut b = bytes.clone();
            let at = b.len() * frac / 100;
            b[at] ^= 0b0001_0000;
            let p = dir.join(format!("flip{frac}.qtsq"));
            std::fs::write(&p, &b).expect("write");
            unsafe {
                assert!(
                    read_segment_report(&p.to_string_lossy()).is_err(),
                    "a raw flip at {frac}% must be refused whole"
                );
            }
        }
    }

    /// The many-zero case, pinned honestly: this container's tensor
    /// transform reconstructs near-zero components from tensor-global state,
    /// so re-encoding a segment of zero-heavy vectors moves OTHER records'
    /// decoded bytes too. The per-record CRC then reports exactly that —
    /// the records whose stored bytes really did change — and the segment
    /// still LOADS with a complete account (`records + dropped == cap`)
    /// instead of being refused. That is the codec's granularity, not the
    /// CRC's: with dense embedding vectors (production) the same test shows
    /// 1 of 256. A raw on-disk bit flip never reaches this layer at all —
    /// the container authenticates its payload first and is refused whole.
    #[test]
    fn w3_5_zero_heavy_vectors_drop_every_record_the_codec_actually_changed() {
        let dim = 64usize;
        let dir = tmpdir("w35-zero-heavy");
        let path = dir.join(segment_file(1, 1)).to_string_lossy().to_string();
        let recs: Vec<SegRecord> = (0..SEGMENT_CAP as u32)
            .map(|i| if i % 2 == 0 { rec_emb(i, dim) } else { rec_sparse(i, dim) })
            .collect();
        unsafe {
            write_segment(&path, dim, &recs).expect("write");
            let (before, _) = read_segment_report(&path).expect("reads");
            assert_eq!(
                (before.records.len(), before.dropped),
                (SEGMENT_CAP, 0),
                "a freshly written segment verifies completely, zeros and all"
            );
            let raw = open_segment_raw(&path).expect("decode");
            let crcs = raw.crcs.clone().expect("per-record CRCs");
            let mut damaged = recs.clone();
            damaged[100].vec[3] = f32::from_bits(damaged[100].vec[3].to_bits() ^ (1 << 9));
            pack_segment(&path, dim, &damaged, Some(&crcs), SEG_FLAG_PER_RECORD_CRC)
                .expect("repack");
            let after = open_segment_raw(&path).expect("decode");
            let changed = after
                .recs
                .iter()
                .zip(raw.recs.iter())
                .filter(|(a, b)| {
                    a.vec.iter().zip(b.vec.iter()).any(|(x, y)| x.to_bits() != y.to_bits())
                })
                .count();
            let (read, _) = read_segment_report(&path).expect("the segment still loads");
            assert_eq!(
                read.dropped, changed,
                "every record whose stored bytes changed is dropped, and no others"
            );
            assert_eq!(
                read.records.len() + read.dropped,
                SEGMENT_CAP,
                "a complete account: nothing vanishes silently"
            );
            assert!(read.records.iter().all(|r| r.id != 100), "the damaged record is gone");
            println!(
                "W3.5 zero-heavy: the codec moved {changed}/{SEGMENT_CAP} records (global near-zero reconstruction); the CRC dropped exactly those {changed} and kept {}",
                read.records.len()
            );
        }
    }

    /// (c) A 64-byte TRUNCATION is not a record-level fault: the segment is
    /// refused as a unit, never "repaired" into a short store.
    #[test]
    fn w3_5_a_truncated_segment_is_refused_as_a_unit() {
        let dim = 64usize;
        let dir = tmpdir("w35-trunc");
        let (a, _b, _, _) = two_segments(&dir, dim);
        let bytes = std::fs::read(&a).expect("read segment");
        assert!(bytes.len() > 64);
        let cut = dir.join("truncated.qtsq");
        std::fs::write(&cut, &bytes[..bytes.len() - 64]).expect("truncate");
        unsafe {
            let r = read_segment_report(&cut.to_string_lossy());
            assert!(
                r.is_err(),
                "a 64-byte truncation must reject the whole segment, got {r:?}"
            );
            /* Not even the records that are still present may leak out. */
            assert!(read_segment(&cut.to_string_lossy()).is_err());
            assert!(read_segment_ex(&cut.to_string_lossy()).is_err());
        }
        /* The intact file is still fine (the check is about the file).
         * And a segment truncated into the middle of its payload — losing
         * records, not just the trailer — is refused the same way. */
        let before_payload = dir.join("before-payload.qtsq");
        std::fs::write(&before_payload, &bytes[..bytes.len() / 3]).expect("truncate hard");
        unsafe {
            assert!(read_segment(&before_payload.to_string_lossy()).is_err());
        }
    }

    /// (d) A segment written by the OLD format — no flag, no CRC table, no
    /// trailer — still loads COMPLETELY, and still behaves exactly as it did
    /// (a damaged record is invisible to it, because it carries no digest to
    /// check: that is what "unset → today's whole-segment behaviour" means).
    #[test]
    fn w3_5_an_old_format_segment_still_loads_completely() {
        let dim = 32usize;
        let dir = tmpdir("w35-legacy");
        let path = dir.join("seg-000001-000001.qtsq").to_string_lossy().to_string();
        let recs: Vec<SegRecord> = (0..SEGMENT_CAP as u32).map(|i| rec_emb(i, dim)).collect();
        unsafe {
            /* flags == 0 is byte-for-byte the pre-W3.5 layout. */
            pack_segment(&path, dim, &recs, None, 0).expect("write old layout");
            let (read, _) = read_segment_report(&path).expect("an old segment must load");
            assert_eq!(read.records.len(), SEGMENT_CAP, "every old record loads");
            assert_eq!(read.dropped, 0, "nothing to verify, nothing dropped");
            for (got, want) in read.records.iter().zip(recs.iter()) {
                assert_eq!(got.id, want.id);
                assert_eq!(got.meta, want.meta);
            }
            /* No trailer, no flag: the file is what the old builder wrote. */
            assert_eq!(segment_trailer(&path), Ok(false));
            let container = container_open(&path).expect("reopen");
            let meta = read_json_stream(container, SEGMENT_META_STREAM).expect("meta");
            container_free(container);
            let root: serde_json::Value = serde_json::from_str(&meta).unwrap();
            assert!(root.get("flags").is_none(), "no flag key in the old layout");
            assert!(root.get("crcs").is_none(), "no CRC table in the old layout");
        }
    }

    /// (e) MUTATION PROOF: with per-record verification switched off, the
    /// vector-flip acceptance test must FAIL. This proves the assertions in
    /// `w3_5_one_damaged_vector_or_text_record_costs_only_that_record`
    /// depend on the verification itself and not on some accident of
    /// re-encoding.
    #[test]
    fn w3_5_mutation_proof_without_record_verification_the_flip_survives() {
        let dim = 64usize;
        let dir = tmpdir("w35-mutation");
        let (a, b, first, _) = two_segments(&dir, dim);
        tamper(&a, dim, &first, 100, Tamper::VectorBit(9));

        let victim = 100u32;
        unsafe {
            /* Baseline: verification ON → the record is caught. */
            assert!(
                per_record_verify_enabled(),
                "precondition: verification is on in production"
            );
            let (on, _) = read_segment_report(&a).expect("reads");
            assert_eq!(on.dropped, 1);
            assert!(!on.records.iter().any(|r| r.id == victim));

            /* Mutant: verification OFF → the damaged record is handed out,
             * so the acceptance assertions (511 live / 1 dropped / victim
             * gone) all fail. */
            PER_RECORD_VERIFY.store(false, std::sync::atomic::Ordering::Relaxed);
            let (off, _) = read_segment_report(&a).expect("reads");
            PER_RECORD_VERIFY.store(true, std::sync::atomic::Ordering::Relaxed);

            assert_eq!(off.dropped, 0, "MUTANT: nothing is detected");
            assert_eq!(off.records.len(), SEGMENT_CAP, "MUTANT: the segment is whole");
            let mutant_victim = off
                .records
                .iter()
                .find(|r| r.id == victim)
                .expect("MUTANT: the damaged record is still served");
            let (other, _) = read_segment_report(&b).expect("reads");
            let mutant_live = other.records.len() + off.records.len();
            println!(
                "W3.5 mutation: verification OFF → {mutant_live} live / {} dropped (acceptance expects 511 / 1) — the damaged record is served with a changed vector component {:.9}",
                off.dropped, mutant_victim.vec[3]
            );
            assert_ne!(
                mutant_live, 511,
                "MUTANT must fail the acceptance count (511 live)"
            );
        }
        /* Back on for every later test in this process. */
        assert!(per_record_verify_enabled());
    }

    /// The CRC covers what the reader consumes, record by record: a change
    /// to the id, to any metadata field, or to any vector component moves it.
    #[test]
    fn w3_5_the_record_crc_covers_id_metadata_and_vector() {
        let dim = 8usize;
        let v = embedding(3, dim);
        let meta = serde_json::json!({"tier": 1, "strength": 0.75, "text": "hello", "role": "user"});
        let base = record_crc(3, &meta, &v);
        assert_eq!(base, record_crc(3, &meta, &v), "stable for the same record");
        assert_ne!(base, record_crc(4, &meta, &v), "id is covered");
        let mut m2 = meta.clone();
        m2["text"] = serde_json::json!("hellp");
        assert_ne!(base, record_crc(3, &m2, &v), "text is covered");
        let mut m3 = meta.clone();
        m3["strength"] = serde_json::json!(0.751);
        assert_ne!(base, record_crc(3, &m3, &v), "strength is covered");
        let mut m4 = meta.clone();
        m4["role"] = serde_json::json!("assistant");
        assert_ne!(base, record_crc(3, &m4, &v), "role is covered");
        let mut v2 = v.clone();
        v2[5] = f32::from_bits(v2[5].to_bits() ^ 1);
        assert_ne!(base, record_crc(3, &meta, &v2), "the vector is covered");
        assert_ne!(crc32(b"a"), crc32(b"b"));
    }

    /* ══════════════════════════════════════════════════════════════
     * W3.4: generation reclamation
     * ══════════════════════════════════════════════════════════════ */

    /// Write one generation's segments (chunks of SEGMENT_CAP), published
    /// the way the store does it: stage → rename.
    fn write_generation(dir: &str, epoch: u64, dim: usize, live: &[SegRecord]) -> Vec<SegmentMeta> {
        let mut out = Vec::new();
        for (i, chunk) in live.chunks(SEGMENT_CAP).enumerate() {
            let name = segment_file(epoch, 1 + i as u64);
            let final_path = format!("{dir}/{name}");
            let tmp = format!("{final_path}.tmp");
            unsafe {
                write_segment(&tmp, dim, chunk).expect("write segment");
            }
            let sha256 = file_digest(&tmp).unwrap_or_default();
            assert!(
                crate::rt::wal::publish(&tmp, &final_path),
                "publish {final_path}"
            );
            out.push(SegmentMeta {
                file: name,
                records: chunk.len(),
                sha256,
            });
        }
        out
    }

    /// Save the graph snapshot of a generation (topology only).
    fn save_graph(dir: &str, generation: u64, dim: usize, live: &[SegRecord]) {
        let mut h = crate::memory::hnsw::Hnsw::new(dim);
        for r in live {
            h.add_vector(&r.vec);
        }
        h.save_to(&format!("{dir}/{}", graph_file(generation)))
            .expect("save graph snapshot");
    }

    fn publish_manifest(path: &str, epoch: u64, segments: &[SegmentMeta], records: usize) {
        let m = Manifest {
            schema: SEGMENT_SCHEMA.to_string(),
            embedding: serde_json::json!({"id": "test", "dimension": 768}),
            segments: segments.to_vec(),
            epoch,
            last_decay_at: 0,
            evicted_total: 0,
            records: records as u64,
            decay_tag: None,
            decay_seconds: None,
        };
        let tmp = format!("{path}.tmp");
        unsafe { write_manifest(&tmp, &m).expect("write manifest") };
        assert!(crate::rt::wal::publish(&tmp, path));
    }

    fn segment_bytes(dir: &str, segments: &[SegmentMeta]) -> u64 {
        segments
            .iter()
            .filter_map(|s| std::fs::metadata(format!("{dir}/{}", s.file)).ok())
            .map(|m| m.len())
            .sum()
    }

    /// W3.4 ACCEPTANCE: five write → supersede → reclaim cycles.
    ///
    /// Amplification is `(bytes on disk) / (bytes of live data)`, where live
    /// data is the current generation's segments (the bytes a reader must
    /// read to reconstruct the store) and the disk total includes the graph
    /// snapshot. Every cycle asserts it lands under 1.5x, and that no file
    /// the committed manifest references was ever deleted.
    #[test]
    fn w3_4_five_cycles_stay_under_1_5x_and_never_delete_a_referenced_file() {
        let dim = 128usize;
        let n = 300usize;
        let root = tmpdir("w34");
        let manifest = root.join("brain.qtsq").to_string_lossy().to_string();
        let dir = segments_dir(&manifest);
        std::fs::create_dir_all(&dir).unwrap();

        let mut live: Vec<SegRecord> = (0..n as u32).map(|i| rec_emb(i, dim)).collect();
        let mut epoch = 1u64;
        let mut segments = write_generation(&dir, epoch, dim, &live);
        save_graph(&dir, epoch, dim, &live);
        publish_manifest(&manifest, epoch, &segments, live.len());

        let mut before_reclaim: Vec<f64> = Vec::new();
        let mut after_reclaim: Vec<f64> = Vec::new();
        for cycle in 0..5u32 {
            /* 1. A FLUSH: the changed records append as a new segment of the
             *    CURRENT generation (memory.rs's append_segments). */
            live[(cycle as usize * 17) % n].meta["strength"] = serde_json::json!(0.25);
            let changed: Vec<SegRecord> = vec![live[(cycle as usize * 17) % n].clone()];
            let seq = segments.len() as u64 + 1;
            let name = segment_file(epoch, seq);
            let final_path = format!("{dir}/{name}");
            let tmp = format!("{final_path}.tmp");
            unsafe { write_segment(&tmp, dim, &changed).expect("append") };
            let sha = file_digest(&tmp).unwrap_or_default();
            assert!(crate::rt::wal::publish(&tmp, &final_path));
            segments.push(SegmentMeta {
                file: name,
                records: changed.len(),
                sha256: sha,
            });
            publish_manifest(&manifest, epoch, &segments, live.len());
            /* The append itself must stay bounded: disk ≈ live + one append. */
            let live_b = segment_bytes(&dir, &segments);
            let amp = dir_bytes(&dir) as f64 / live_b as f64;
            assert!(
                amp < 1.5,
                "cycle {cycle}: append amplification {amp:.3}x exceeds 1.5x"
            );

            /* 2. A COMPACTION: the epoch bumps and every live record is
             *    rewritten into the new generation (memory.rs's
             *    compact_segments), which SUPERSEDES the old one. */
            epoch += 1;
            segments = write_generation(&dir, epoch, dim, &live);
            save_graph(&dir, epoch, dim, &live);
            publish_manifest(&manifest, epoch, &segments, live.len());

            let live_b = segment_bytes(&dir, &segments);
            assert!(live_b > 0);
            before_reclaim.push(dir_bytes(&dir) as f64 / live_b as f64);

            /* 3. RECLAIM: the manifest references generation `epoch` only. */
            let removed = reclaim_unreferenced_now(&dir, &[epoch], 0);
            let amp = dir_bytes(&dir) as f64 / live_b as f64;
            after_reclaim.push(amp);
            assert!(
                amp < 1.5,
                "cycle {cycle}: amplification {amp:.3}x after reclaim exceeds 1.5x"
            );

            /* 4. NEVER DELETE A REFERENCED FILE — the manifest opens, every
             *    segment it lists is present with every record, and the
             *    generation's graph snapshot survives the sweep. */
            let m = unsafe { read_manifest(&manifest) }.expect("manifest reads");
            assert_eq!(m.epoch, epoch);
            let mut total = 0usize;
            for s in &m.segments {
                let p = format!("{dir}/{}", s.file);
                assert!(
                    std::path::Path::new(&p).exists(),
                    "cycle {cycle}: referenced segment {} was deleted",
                    s.file
                );
                let (read, _) = unsafe { read_segment_report(&p) }.expect("segment reads");
                assert_eq!(read.dropped, 0, "cycle {cycle}: {} lost records", s.file);
                total += read.records.len();
            }
            assert_eq!(total, live.len(), "cycle {cycle}: every live record survives");
            let graph = format!("{dir}/{}", graph_file(epoch));
            assert!(
                std::path::Path::new(&graph).exists(),
                "cycle {cycle}: the current generation's graph snapshot was deleted"
            );
            let loaded = crate::memory::hnsw::Hnsw::load_with_vectors(
                &graph,
                live.iter().map(|r| r.vec.clone()).collect(),
            )
            .expect("the surviving snapshot still loads");
            assert_eq!(loaded.len(), live.len());
            assert!(removed > 0, "cycle {cycle}: the superseded generation was swept");
        }

        /* The steady state: the sweep is what keeps the store bounded — the
         * ratio would drift past the budget without it (asserted above via
         * `before_reclaim`). */
        let worst_after = after_reclaim.iter().cloned().fold(0.0f64, f64::max);
        let mean_before = before_reclaim.iter().sum::<f64>() / before_reclaim.len() as f64;
        println!(
            "W3.4 reclaim: amplification before reclaim {mean_before:.2}x (mean), after reclaim {worst_after:.2}x (worst of {} cycles); files remaining: {}",
            after_reclaim.len(),
            std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0)
        );
        assert!(
            mean_before > 1.5,
            "the superseded generation DID cost amplification ({mean_before:.2}x) — otherwise this test proves nothing"
        );
    }

    /// The sweep's naming rules: only store files are considered, the
    /// generations in `keep` are untouched, and an unparseable name is left
    /// strictly alone.
    #[test]
    fn w3_4_reclaim_only_touches_unreferenced_store_files() {
        let dir = tmpdir("w34-names");
        let d = dir.to_string_lossy().to_string();
        for name in [
            "seg-000001-000001.qtsq",
            "seg-000002-000001.qtsq",
            "seg-000002-000002.qtsq",
            "graph-000001.bin",
            "graph-000002.bin",
            "graph-000003.bin",
            "notes.txt",
            "seg-broken.qtsq",
        ] {
            std::fs::write(format!("{d}/{name}"), b"x").unwrap();
        }
        assert_eq!(generation_of("seg-000002-000002.qtsq"), Some(2));
        assert_eq!(generation_of("graph-000003.bin"), Some(3));
        assert_eq!(generation_of("notes.txt"), None);
        assert_eq!(generation_of("seg-broken.qtsq"), None);

        assert_eq!(reclaim_unreferenced_now(&d, &[2], 0), 3);
        let left: Vec<String> = std::fs::read_dir(&d)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        for kept in [
            "seg-000002-000001.qtsq",
            "seg-000002-000002.qtsq",
            "graph-000002.bin",
        ] {
            assert!(left.iter().any(|n| n == kept), "{kept} must survive");
        }
        for gone in ["seg-000001-000001.qtsq", "graph-000001.bin", "graph-000003.bin"] {
            assert!(!left.iter().any(|n| n == gone), "{gone} must be reclaimed");
        }
        assert!(left.iter().any(|n| n == "notes.txt"), "foreign files are never touched");
        assert!(left.iter().any(|n| n == "seg-broken.qtsq"), "unparseable names are never touched");

        /* The production call keeps the age guard: fresh files survive. */
        std::fs::write(format!("{d}/graph-000009.bin"), b"x").unwrap();
        assert_eq!(reclaim_unreferenced(&d, &[2]), 0, "fresh files are age-guarded");
        assert!(std::path::Path::new(&format!("{d}/graph-000009.bin")).exists());
    }

}

