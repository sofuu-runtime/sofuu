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

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;

use sofuu_ffi::qtsq;

/// The deterministic local vault password. Same value the legacy single-file
/// adapter uses (rt/memory.rs aliases this constant) — brain files are
/// machine-local at-rest obfuscation, and a fixed password avoids macOS
/// Keychain prompts on every launch.
pub const LOCAL_PASSWORD: &str = "sofuu-kv-local-v1";

/// Codec tensor-compression level: 0 = full contest (see rt/memory.rs).
pub const QTSQ_LEVEL_FULL: c_int = 0;

pub const MANIFEST_STREAM: &str = "manifest";
pub const SEGMENT_META_STREAM: &str = "meta";
pub const SEGMENT_TENSOR_STREAM: &str = "memories";
pub const SEGMENT_SCHEMA: &str = "sofuu-cma-segments@1";

/// Records per segment before rotation. Bounds both the active rewrite
/// (a flush touches at most one segment) and the read amplification of a
/// single segment. 256 records × 768 dims ≈ 786 KB of vectors.
pub const SEGMENT_CAP: usize = 256;

/// One segment file the manifest references.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SegmentMeta {
    pub file: String,
    pub records: usize,
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
    let pw = CString::new(LOCAL_PASSWORD).unwrap_or_default();
    qtsq::qtsq_vault_encrypt_password(ctx, pw.as_ptr())
}

unsafe fn decrypt(ctx: *mut qtsq::QtsqContext) -> c_int {
    if !(*ctx).is_encrypted() {
        return qtsq::QTSQ_OK;
    }
    let pw = CString::new(LOCAL_PASSWORD).unwrap_or_default();
    qtsq::qtsq_vault_decrypt_password(ctx, pw.as_ptr())
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
    if r == qtsq::QTSQ_OK {
        r = qtsq::qtsq_container_pack(container);
    }
    if r == qtsq::QTSQ_OK {
        r = encrypt(container);
    }
    if r == qtsq::QTSQ_OK {
        let cpath = CString::new(path).unwrap_or_default();
        r = qtsq::qtsq_write(container, cpath.as_ptr());
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
    let cpath = CString::new(path).unwrap_or_default();
    let container = qtsq::qtsq_ctx_alloc();
    if container.is_null() {
        return Err(qtsq::QTSQ_ERR_ALLOC);
    }
    let mut r = qtsq::qtsq_init(container);
    if r == qtsq::QTSQ_OK {
        r = qtsq::qtsq_read(container, cpath.as_ptr());
    }
    if r == qtsq::QTSQ_OK {
        r = decrypt(container);
    }
    if r == qtsq::QTSQ_OK && (*container).data_type() != qtsq::QTSQ_TYPE_CONTAINER {
        r = qtsq::QTSQ_ERR_TYPE;
    }
    if r != qtsq::QTSQ_OK {
        qtsq::qtsq_free(container);
        qtsq::qtsq_ctx_free(container);
        return Err(r);
    }
    Ok(container)
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
        let dims: Vec<u32> = (0..nd).map(|i| (*sub).schema_dim(i)).collect();
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

/// Write one segment: tensor of the records' vectors + their metadata.
pub unsafe fn write_segment(
    path: &str,
    dim: usize,
    recs: &[SegRecord],
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
    let meta_json = serde_json::json!({
        "dim": dim,
        "records": metas,
    })
    .to_string();
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
    )
}

/// Read one segment back. Records come out in file order.
pub unsafe fn read_segment(path: &str) -> Result<Vec<SegRecord>, c_int> {
    let container = container_open(path)?;
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
    Ok(out)
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
    let container = container_open(path).ok()?;
    let body = read_json_stream(container, MANIFEST_STREAM);
    container_free(container);
    let body = body?;
    let m: Manifest = serde_json::from_str(&body).ok()?;
    if m.schema != SEGMENT_SCHEMA {
        return None;
    }
    Some(m)
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
            assert_eq!(back[0].vec, recs[0].vec);
            assert_eq!(back[1].vec, recs[1].vec);
        }
    }

    #[test]
    fn manifest_round_trips() {
        let dir = tmpdir("manifest");
        let path = dir.join("brain.qtsq");
        let mut m = Manifest::new(serde_json::json!({"id": "hash-v1", "dimension": 768}));
        m.segments.push(SegmentMeta { file: "seg-000001-000001.qtsq".into(), records: 3 });
        m.last_decay_at = 1234;
        m.evicted_total = 9;
        unsafe {
            write_manifest(&path.to_string_lossy(), &m).expect("write");
            let back = read_manifest(&path.to_string_lossy()).expect("read");
            assert_eq!(back.schema, SEGMENT_SCHEMA);
            assert_eq!(back.segments.len(), 1);
            assert_eq!(back.segments[0].records, 3);
            assert_eq!(back.last_decay_at, 1234);
            assert_eq!(back.evicted_total, 9);
            assert_eq!(back.embedding["id"], "hash-v1");
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
}
