//! Bounded, crash-safe QTSQ storage for main Sofuu sessions.
//!
//! The desktop JavaScript session layout has a different schema and lives in
//! a different namespace. This module is only for the Rust Sofuu binary.

use crate::session::{SessionData, SessionEvent};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const SOFT_MAX_EVENTS: usize = 64;
pub(crate) const SOFT_MAX_PAYLOAD: usize = 256 * 1024;
pub(crate) const HARD_MAX_EVENTS: usize = 128;
pub(crate) const HARD_MAX_PAYLOAD: usize = 1024 * 1024;

const STORE_SUFFIX: &str = ".store";
const MANIFEST_FILE: &str = "manifest.qtsq";
const SEGMENTS_DIR: &str = "segments";
const LOCK_FILE: &str = ".lock";
const MANIFEST_FORMAT: &str = "sofuu.session.manifest";
const SEGMENT_FORMAT: &str = "sofuu.session.segment";
const STORAGE_SCHEMA: u32 = 1;
const MAX_SEGMENTS: usize = 4096;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Manifest {
    pub format: String,
    pub storage_schema: u32,
    pub session_id: String,
    pub created: u64,
    pub host: String,
    pub cwd: String,
    pub model: String,
    pub provider: String,
    pub task: Option<String>,
    pub notes: Vec<String>,
    pub next_seq: u64,
    pub retained_from_seq: u64,
    pub event_count: u64,
    pub last_event_t: u64,
    #[serde(default)]
    pub source_sha256: Option<String>,
    #[serde(default)]
    pub migrated_at: Option<u64>,
    pub segments: Vec<SegmentMeta>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SegmentMeta {
    pub id: String,
    pub file: String,
    pub state: String,
    pub start_seq: u64,
    pub end_seq: u64,
    pub event_count: u64,
    pub payload_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SegmentPayload {
    format: String,
    storage_schema: u32,
    session_id: String,
    start_seq: u64,
    end_seq: u64,
    events: Vec<StoredEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredEvent {
    seq: u64,
    t: u64,
    kind: String,
    text: String,
}

impl StoredEvent {
    fn from_event(event: &SessionEvent, fallback_seq: u64) -> Self {
        Self {
            seq: if event.seq == 0 { fallback_seq } else { event.seq },
            t: event.t,
            kind: event.kind.clone(),
            text: event.text.clone(),
        }
    }

    fn into_event(self) -> SessionEvent {
        SessionEvent {
            seq: self.seq,
            t: self.t,
            kind: self.kind,
            text: self.text,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StorageStats {
    pub segmented: bool,
    pub manifest_schema: u32,
    pub segments: Vec<SegmentMeta>,
    pub record_count: usize,
    pub raw_record_count: usize,
    pub compressed_record_count: usize,
    pub encrypted_record_count: usize,
    pub segment_count: usize,
    pub sealed_count: usize,
    pub active_events: u64,
    pub active_bytes: u64,
    pub retained_from_seq: u64,
    pub next_seq: u64,
    pub event_count: u64,
    pub payload_bytes: u64,
    pub source_sha256: Option<String>,
    pub migrated_at: Option<u64>,
    pub missing_count: usize,
    pub corrupt_count: usize,
    pub orphaned_count: usize,
    pub quarantined_count: usize,
    pub incomplete_count: usize,
}

struct FileLock(File);

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn with_store_lock<T>(
    project: &Path,
    id: &str,
    f: impl FnOnce() -> T,
) -> Option<T> {
    let dir = store_dir(project, id)?;
    if ensure_store_dirs(&dir).is_err() {
        return None;
    }
    let lock_path = dir.join(LOCK_FILE);
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)
        .ok()?;
    // Blocks until we hold the exclusive lock (flock / LockFileEx).
    if file.lock().is_err() {
        return None;
    }
    let _guard = FileLock(file);
    Some(f())
}

pub(crate) fn store_dir(project: &Path, id: &str) -> Option<PathBuf> {
    if !crate::session::valid_session_id(id) {
        return None;
    }
    Some(crate::session::sessions_dir(project).join(format!("{id}{STORE_SUFFIX}")))
}

fn manifest_path(project: &Path, id: &str) -> Option<PathBuf> {
    Some(store_dir(project, id)?.join(MANIFEST_FILE))
}

fn manifest_path_at(store: &Path) -> PathBuf {
    store.join(MANIFEST_FILE)
}

pub(crate) fn has_manifest(project: &Path, id: &str) -> bool {
    manifest_path(project, id)
        .map(|path| path.is_file() && !is_symlink(&path))
        .unwrap_or(false)
}

fn is_symlink(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_symlink())
        .unwrap_or(false)
}

fn ensure_store_dirs(dir: &Path) -> std::io::Result<()> {
    if dir.exists() && is_symlink(dir) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "session store is a symlink",
        ));
    }
    fs::create_dir_all(dir.join(SEGMENTS_DIR))?;
    if is_symlink(&dir.join(SEGMENTS_DIR)) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "session segments directory is a symlink",
        ));
    }
    set_private_permissions(dir, true);
    set_private_permissions(&dir.join(SEGMENTS_DIR), true);
    Ok(())
}

#[cfg(unix)]
fn set_private_permissions(path: &Path, _directory: bool) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = fs::metadata(path) {
        let mut permissions = meta.permissions();
        /* Narrow toward private (strip group/other access) but NEVER widen:
         * re-applying a full 0o700 on every persist would silently undo a
         * deliberately frozen store (e.g. chmod 0o500) and mask the
         * unwritable-store failure that persist must report. */
        let cur = permissions.mode() & 0o777;
        let narrowed = cur & !0o077;
        if narrowed != cur {
            permissions.set_mode(narrowed);
            let _ = fs::set_permissions(path, permissions);
        }
    }
}

#[cfg(not(unix))]
fn set_private_permissions(_path: &Path, _directory: bool) {}

fn safe_segment_filename(name: &str) -> bool {
    if name.is_empty()
        || !name.ends_with(".qtsq")
        || name.contains(".tmp")
        || !(name.starts_with("segment-") || name.starts_with("active-"))
    {
        return false;
    }
    name.bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
}

fn safe_segment_path(store: &Path, relative: &str) -> Option<PathBuf> {
    let rel = Path::new(relative);
    let mut components = rel.components();
    match components.next() {
        Some(Component::Normal(value)) if value == SEGMENTS_DIR => {}
        _ => return None,
    }
    let Some(Component::Normal(name)) = components.next() else {
        return None;
    };
    if components.next().is_some() {
        return None;
    }
    let name = name.to_str()?;
    if !safe_segment_filename(name) {
        return None;
    }
    let segments = store.join(SEGMENTS_DIR);
    let path = segments.join(name);
    if is_symlink(store) || is_symlink(&segments) || is_symlink(&path) {
        return None;
    }
    Some(path)
}

fn unique_temp_path(target: &Path, tag: &str) -> PathBuf {
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = target
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "qtsq".into());
    target.with_file_name(format!(
        ".{name}.{tag}.{}.{}.tmp",
        std::process::id(),
        counter
    ))
}

fn unique_active_filename(first_seq: u64) -> String {
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "active-{first_seq:08}-{:08x}-{counter:016x}.qtsq",
        std::process::id()
    )
}

fn sync_file(path: &Path) -> bool {
    File::open(path)
        .and_then(|file| file.sync_all())
        .is_ok()
}

fn sync_dir(path: &Path) {
    if let Ok(file) = File::open(path) {
        let _ = file.sync_all();
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(crate) fn save_qtsq(path: &Path, bytes: &[u8]) -> bool {
    let Some(path_str) = path.to_str() else {
        return false;
    };
    sofuu_ffi::qtsq_session_save(path_str, bytes) == 0
}

fn publish_qtsq(path: &Path, bytes: &[u8]) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    if fs::create_dir_all(parent).is_err() {
        return false;
    }
    let temp = unique_temp_path(path, "publish");
    if !save_qtsq(&temp, bytes) || !sync_file(&temp) {
        let _ = fs::remove_file(&temp);
        return false;
    }
    if fs::rename(&temp, path).is_err() {
        let _ = fs::remove_file(&temp);
        return false;
    }
    sync_dir(parent);
    set_private_permissions(path, false);
    true
}

fn read_qtsq(project: &Path, path: &Path) -> Option<Vec<u8>> {
    if is_symlink(path) {
        return None;
    }
    let path_str = path.to_str()?;
    sofuu_ffi::qtsq_session_load(path_str, &crate::session::session_password(project))
}

/// session-1 (AUDIT-2026-09-07): strict read of the desktop JS driver's
/// qtsq registry for destructive paths. The plaintext registry reader
/// fails open (missing == empty), and the desktop driver never writes
/// registry.json — so prune's active-set protection MUST consult this
/// store too or every desktop session looks inactive and its archive
/// becomes prunable. Missing file -> Ok(None) (fresh project, empty is
/// truthful); symlink, undecryptable or corrupt -> Err (never "empty").
pub(crate) fn registry_qtsq_strict(project: &Path) -> Result<Option<crate::session::Registry>, String> {
    let path = crate::session::sessions_dir(project).join("registry.qtsq");
    if !path.exists() {
        return Ok(None);
    }
    if is_symlink(&path) {
        return Err(format!("registry.qtsq is a symlink ({}): refusing destructive path", path.display()));
    }
    let bytes = read_qtsq(project, &path).ok_or_else(|| {
        format!("registry.qtsq unreadable ({}): decrypt/parse failed", path.display())
    })?;
    serde_json::from_slice::<crate::session::Registry>(&bytes)
        .map(Some)
        .map_err(|e| format!("registry.qtsq corrupt ({}): {e}", path.display()))
}

#[derive(Default)]
struct RecordProfile {
    raw: bool,
    compressed: bool,
    encrypted: bool,
}

fn record_profile(path: &Path) -> Option<RecordProfile> {
    let bytes = fs::read(path).ok()?;
    let strategy = *bytes.get(5)?;
    let encrypted = path
        .to_str()
        .and_then(sofuu_ffi::qtsq_session_is_encrypted)
        .unwrap_or(false);
    Some(RecordProfile {
        raw: strategy == 0,
        compressed: strategy != 0,
        encrypted,
    })
}

impl Manifest {
    fn validate(&self) -> Result<(), String> {
        if self.format != MANIFEST_FORMAT {
            return Err("wrong session manifest format".into());
        }
        if self.storage_schema != STORAGE_SCHEMA {
            return Err("unsupported session manifest schema".into());
        }
        if !crate::session::valid_session_id(&self.session_id) {
            return Err("invalid session id in manifest".into());
        }
        if self.next_seq == 0 || self.retained_from_seq == 0 {
            return Err("invalid session sequence boundary".into());
        }
        if self.segments.len() > MAX_SEGMENTS {
            return Err("too many session segments".into());
        }
        let mut previous_end: Option<u64> = None;
        let mut active_count = 0usize;
        for (index, segment) in self.segments.iter().enumerate() {
            if segment.start_seq == 0 || segment.start_seq > segment.end_seq {
                return Err("invalid segment range".into());
            }
            if segment.event_count != segment.end_seq - segment.start_seq + 1 {
                return Err("segment event count does not match range".into());
            }
            if segment.state != "sealed" && segment.state != "active" {
                return Err("invalid segment state".into());
            }
            if segment.state == "active" {
                active_count += 1;
                if index + 1 != self.segments.len() {
                    return Err("active segment is not last".into());
                }
            }
            if let Some(end) = previous_end {
                if segment.start_seq != end.saturating_add(1) {
                    return Err("segment ranges are not contiguous".into());
                }
            } else if self.event_count > 0
                && (segment.start_seq > self.retained_from_seq
                    || segment.end_seq < self.retained_from_seq)
            {
                return Err("first segment does not cover retention boundary".into());
            }
            previous_end = Some(segment.end_seq);
        }
        if active_count > 1 {
            return Err("multiple active segments".into());
        }
        match previous_end {
            None => {
                if self.event_count != 0 || self.next_seq != 1 || self.retained_from_seq != 1 {
                    return Err("empty manifest has invalid counters".into());
                }
            }
            Some(end) => {
                if self.next_seq != end.saturating_add(1) {
                    return Err("manifest next sequence does not match segments".into());
                }
                let retained_end = self.next_seq - 1;
                let retained_count = self
                    .segments
                    .iter()
                    .map(|segment| {
                        let start = segment.start_seq.max(self.retained_from_seq);
                        let end = segment.end_seq.min(retained_end);
                        if start <= end { end - start + 1 } else { 0 }
                    })
                    .sum::<u64>();
                if retained_count != self.event_count {
                    return Err("manifest event count does not match retention range".into());
                }
            }
        }
        Ok(())
    }
}

fn load_manifest_at(project: &Path, id: &str, path: &Path) -> Option<Manifest> {
    if !path.is_file() || is_symlink(&path) {
        return None;
    }
    let bytes = read_qtsq(project, &path)?;
    let manifest: Manifest = serde_json::from_slice(&bytes).ok()?;
    manifest.validate().ok()?;
    if manifest.session_id != id {
        return None;
    }
    Some(manifest)
}

fn load_manifest(project: &Path, id: &str) -> Option<Manifest> {
    let path = manifest_path(project, id)?;
    load_manifest_at(project, id, &path)
}

fn stored_events(data: &SessionData) -> Option<Vec<StoredEvent>> {
    let mut expected = None;
    let mut out = Vec::with_capacity(data.events.len());
    for event in &data.events {
        let fallback_seq = expected.unwrap_or(1);
        let stored = StoredEvent::from_event(event, fallback_seq);
        if let Some(expected_seq) = expected {
            if stored.seq != expected_seq {
                return None;
            }
        }
        expected = Some(stored.seq.checked_add(1)?);
        out.push(stored);
    }
    Some(out)
}

fn segment_payload(session_id: &str, events: Vec<StoredEvent>) -> SegmentPayload {
    let start_seq = events.first().map(|event| event.seq).unwrap_or(1);
    let end_seq = events.last().map(|event| event.seq).unwrap_or(0);
    SegmentPayload {
        format: SEGMENT_FORMAT.into(),
        storage_schema: STORAGE_SCHEMA,
        session_id: session_id.into(),
        start_seq,
        end_seq,
        events,
    }
}

fn payload_bytes(payload: &SegmentPayload) -> Option<Vec<u8>> {
    serde_json::to_vec(payload).ok()
}

fn segment_meta(file: String, state: &str, payload: &SegmentPayload, bytes: usize) -> SegmentMeta {
    SegmentMeta {
        id: file
            .strip_prefix("segments/")
            .unwrap_or(&file)
            .strip_suffix(".qtsq")
            .unwrap_or(&file)
            .into(),
        file,
        state: state.into(),
        start_seq: payload.start_seq,
        end_seq: payload.end_seq,
        event_count: payload.events.len() as u64,
        payload_bytes: bytes as u64,
    }
}

fn write_segment(
    _project: &Path,
    store: &Path,
    session_id: &str,
    events: Vec<StoredEvent>,
    state: &str,
    filename: String,
) -> Option<SegmentMeta> {
    if events.is_empty() {
        return None;
    }
    let payload = segment_payload(session_id, events);
    let bytes = payload_bytes(&payload)?;
    if payload.events.len() > HARD_MAX_EVENTS
        || (payload.events.len() > 1 && bytes.len() > HARD_MAX_PAYLOAD)
    {
        return None;
    }
    let relative = format!("{SEGMENTS_DIR}/{filename}");
    let path = safe_segment_path(store, &relative)?;
    if !publish_qtsq(&path, &bytes) {
        return None;
    }
    Some(segment_meta(relative, state, &payload, bytes.len()))
}

fn load_segment(
    project: &Path,
    store: &Path,
    session_id: &str,
    meta: &SegmentMeta,
) -> Option<Vec<StoredEvent>> {
    let path = safe_segment_path(store, &meta.file)?;
    let bytes = read_qtsq(project, &path)?;
    let payload: SegmentPayload = serde_json::from_slice(&bytes).ok()?;
    if payload.format != SEGMENT_FORMAT
        || payload.storage_schema != STORAGE_SCHEMA
        || payload.session_id != session_id
        || payload.start_seq != meta.start_seq
        || payload.end_seq != meta.end_seq
        || payload.events.len() as u64 != meta.event_count
        || bytes.len() as u64 != meta.payload_bytes
    {
        return None;
    }
    let mut expected = payload.start_seq;
    for event in &payload.events {
        if event.seq != expected {
            return None;
        }
        expected = expected.checked_add(1)?;
    }
    if payload.events.last().map(|event| event.seq) != Some(payload.end_seq) {
        return None;
    }
    Some(payload.events)
}

fn chunk_events(session_id: &str, events: &[StoredEvent]) -> Vec<Vec<StoredEvent>> {
    let mut chunks: Vec<Vec<StoredEvent>> = Vec::new();
    let mut current = Vec::new();
    for event in events {
        let would_rotate = if current.is_empty() {
            false
        } else if current.len() >= SOFT_MAX_EVENTS {
            true
        } else {
            let mut candidate = current.clone();
            candidate.push(event.clone());
            let payload = segment_payload(session_id, candidate);
            payload_bytes(&payload)
                .map(|bytes| bytes.len() > SOFT_MAX_PAYLOAD)
                .unwrap_or(true)
        };
        if would_rotate {
            chunks.push(std::mem::take(&mut current));
        }
        current.push(event.clone());
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn should_rotate(session_id: &str, current: &[StoredEvent], next: &StoredEvent) -> bool {
    if current.is_empty() {
        return false;
    }
    if current.len() >= SOFT_MAX_EVENTS {
        return true;
    }
    let mut candidate = current.to_vec();
    candidate.push(next.clone());
    let payload = segment_payload(session_id, candidate);
    payload_bytes(&payload)
        .map(|bytes| bytes.len() > SOFT_MAX_PAYLOAD)
        .unwrap_or(true)
}

fn manifest_for_data(
    data: &SessionData,
    segments: Vec<SegmentMeta>,
    next_seq: u64,
) -> Option<Manifest> {
    let retained_from_seq = data.events.first().map(|event| {
        if event.seq == 0 { 1 } else { event.seq }
    }).unwrap_or(next_seq);
    let manifest = Manifest {
        format: MANIFEST_FORMAT.into(),
        storage_schema: STORAGE_SCHEMA,
        session_id: data.id.clone(),
        created: data.created,
        host: data.host.clone(),
        cwd: data.cwd.clone(),
        model: data.model.clone(),
        provider: data.provider.clone(),
        task: data.task.clone(),
        notes: data.notes.clone(),
        next_seq,
        retained_from_seq,
        event_count: data.events.len() as u64,
        last_event_t: data.events.last().map(|event| event.t).unwrap_or(0),
        source_sha256: None,
        migrated_at: None,
        segments,
    };
    manifest.validate().ok()?;
    Some(manifest)
}

fn publish_manifest_at(path: &Path, manifest: &Manifest) -> bool {
    let Ok(bytes) = serde_json::to_vec(manifest) else {
        return false;
    };
    publish_qtsq(path, &bytes)
}

fn publish_manifest(project: &Path, id: &str, manifest: &Manifest) -> bool {
    let Some(path) = manifest_path(project, id) else {
        return false;
    };
    publish_manifest_at(&path, manifest)
}

fn has_non_temp_segment_files(dir: &Path) -> bool {
    fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .any(|entry| {
            let path = entry.path();
            path.is_file()
                && !path
                    .file_name()
                    .map(|name| name.to_string_lossy().contains(".tmp"))
                    .unwrap_or(true)
        })
}

fn create_store_at(project: &Path, data: &SessionData, store: &Path) -> bool {
    if ensure_store_dirs(store).is_err() {
        return false;
    }
    let manifest = manifest_path_at(store);
    if manifest.exists() || has_non_temp_segment_files(&store.join(SEGMENTS_DIR)) {
        return false;
    }
    let Some(events) = stored_events(data) else {
        return false;
    };
    let next_seq = events
        .last()
        .and_then(|event| event.seq.checked_add(1))
        .unwrap_or(1);
    let chunks = chunk_events(&data.id, &events);
    let chunk_count = chunks.len();
    let mut metas = Vec::with_capacity(chunks.len());
    for (index, chunk) in chunks.into_iter().enumerate() {
        let state = if index + 1 == chunk_count { "active" } else { "sealed" };
        let first = chunk.first().map(|event| event.seq).unwrap_or(1);
        let last = chunk.last().map(|event| event.seq).unwrap_or(first);
        let filename = if state == "active" {
            format!("active-{first:08}.qtsq")
        } else {
            format!("segment-{first:08}-{last:08}.qtsq")
        };
        let Some(meta) = write_segment(project, &store, &data.id, chunk, state, filename) else {
            return false;
        };
        metas.push(meta);
    }
    let Some(manifest_data) = manifest_for_data(data, metas, next_seq) else {
        return false;
    };
    publish_manifest_at(&manifest, &manifest_data)
}

fn create_store_locked(project: &Path, data: &SessionData) -> bool {
    let Some(store) = store_dir(project, &data.id) else {
        return false;
    };
    create_store_at(project, data, &store)
}

fn current_active(
    project: &Path,
    id: &str,
    manifest: &Manifest,
    store: &Path,
) -> Option<(SegmentMeta, Vec<StoredEvent>)> {
    let meta = manifest.segments.last()?.clone();
    if meta.state != "active" {
        return None;
    }
    let events = load_segment(project, store, id, &meta)?;
    Some((meta, events))
}

fn refresh_manifest(manifest: &mut Manifest, data: &SessionData, next_seq: u64) -> bool {
    manifest.task = data.task.clone();
    manifest.notes = data.notes.clone();
    manifest.next_seq = next_seq;
    manifest.retained_from_seq = data
        .events
        .first()
        .map(|event| if event.seq == 0 { 1 } else { event.seq })
        .unwrap_or(next_seq);
    manifest.event_count = data.events.len() as u64;
    manifest.last_event_t = data.events.last().map(|event| event.t).unwrap_or(0);
    manifest.validate().is_ok()
}

fn persist_existing_locked(project: &Path, data: &SessionData) -> bool {
    let Some(store) = store_dir(project, &data.id) else {
        return false;
    };
    let Some(mut manifest) = load_manifest(project, &data.id) else {
        return false;
    };
    if manifest.session_id != data.id {
        return false;
    }
    let Some(events) = stored_events(data) else {
        return false;
    };
    let pending: Vec<StoredEvent> = events
        .iter()
        .filter(|event| event.seq >= manifest.next_seq)
        .cloned()
        .collect();
    if let Some(first) = pending.first() {
        if first.seq != manifest.next_seq {
            return false;
        }
        for pair in pending.windows(2) {
            if pair[1].seq != pair[0].seq.saturating_add(1) {
                return false;
            }
        }
    }

    let previous_segments = manifest.segments.clone();
    let previous_active = current_active(project, &data.id, &manifest, &store);
    if !manifest.segments.is_empty() && previous_active.is_none() {
        return false;
    }
    let (old_active_meta, mut active_events) = previous_active
        .map(|(meta, events)| (Some(meta), events))
        .unwrap_or((None, Vec::new()));
    let mut sealed_batches: Vec<Vec<StoredEvent>> = Vec::new();
    for event in pending {
        if should_rotate(&data.id, &active_events, &event) {
            if active_events.is_empty() {
                return false;
            }
            sealed_batches.push(std::mem::take(&mut active_events));
        }
        active_events.push(event);
    }

    let desired_next_seq = events
        .last()
        .and_then(|event| event.seq.checked_add(1))
        .unwrap_or(manifest.next_seq);
    if desired_next_seq < manifest.next_seq {
        return false;
    }

    let mut new_segments: Vec<SegmentMeta> = previous_segments
        .iter()
        .filter(|meta| meta.state != "active")
        .cloned()
        .collect();
    let mut published_files = Vec::new();
    for batch in sealed_batches {
        let Some(first) = batch.first().map(|event| event.seq) else {
            return false;
        };
        let Some(last) = batch.last().map(|event| event.seq) else {
            return false;
        };
        let filename = format!("segment-{first:08}-{last:08}.qtsq");
        let Some(meta) = write_segment(
            project,
            &store,
            &data.id,
            batch,
            "sealed",
            filename,
        ) else {
            return false;
        };
        published_files.push(meta.file.clone());
        new_segments.push(meta);
    }

    if !active_events.is_empty() {
        let first = active_events.first().map(|event| event.seq).unwrap_or(1);
        // Never replace the file referenced by the current manifest in place.
        // A versioned final name leaves the old active segment usable until
        // the replacement manifest has been atomically published.
        let active_filename = unique_active_filename(first);
        let Some(meta) = write_segment(
            project,
            &store,
            &data.id,
            active_events,
            "active",
            active_filename,
        ) else {
            return false;
        };
        published_files.push(meta.file.clone());
        new_segments.push(meta);
    }

    manifest.segments = new_segments;
    if !refresh_manifest(&mut manifest, data, desired_next_seq) {
        return false;
    }
    let retained_from = manifest.retained_from_seq;
    manifest
        .segments
        .retain(|meta| meta.end_seq >= retained_from);
    if !manifest.validate().is_ok() {
        return false;
    }
    if !publish_manifest(project, &data.id, &manifest) {
        return false;
    }

    let referenced: std::collections::HashSet<String> =
        manifest.segments.iter().map(|meta| meta.file.clone()).collect();
    for old in previous_segments {
        if !referenced.contains(&old.file) && !published_files.contains(&old.file) {
            if let Some(path) = safe_segment_path(&store, &old.file) {
                let _ = fs::remove_file(path);
            }
        }
    }
    if let Some(old) = old_active_meta {
        if !referenced.contains(&old.file) {
            if let Some(path) = safe_segment_path(&store, &old.file) {
                let _ = fs::remove_file(path);
            }
        }
    }
    sync_dir(&store.join(SEGMENTS_DIR));
    true
}

pub(crate) fn persist_snapshot(project: &Path, data: &SessionData) -> bool {
    let Some(store) = store_dir(project, &data.id) else {
        return false;
    };
    if ensure_store_dirs(&store).is_err() {
        return false;
    }
    with_store_lock(project, &data.id, || {
        if has_manifest(project, &data.id) {
            persist_existing_locked(project, data)
        } else {
            create_store_locked(project, data)
        }
    })
    .unwrap_or(false)
}

fn session_data_from_manifest(
    manifest: &Manifest,
    events: Vec<SessionEvent>,
) -> SessionData {
    SessionData {
        schema: 1,
        id: manifest.session_id.clone(),
        created: manifest.created,
        host: manifest.host.clone(),
        cwd: manifest.cwd.clone(),
        model: manifest.model.clone(),
        provider: manifest.provider.clone(),
        task: manifest.task.clone(),
        notes: manifest.notes.clone(),
        events,
    }
}

fn all_events_at(
    project: &Path,
    id: &str,
    store: &Path,
    manifest: &Manifest,
) -> Option<Vec<SessionEvent>> {
    let mut expected = manifest.retained_from_seq;
    let mut events = Vec::with_capacity(manifest.event_count as usize);
    for meta in &manifest.segments {
        let stored = load_segment(project, &store, id, meta)?;
        for event in stored {
            if event.seq < manifest.retained_from_seq {
                continue;
            }
            if event.seq != expected {
                return None;
            }
            expected = expected.checked_add(1)?;
            events.push(event.into_event());
        }
    }
    if events.len() as u64 != manifest.event_count
        || expected != manifest.next_seq
    {
        return None;
    }
    Some(events)
}

fn all_events(
    project: &Path,
    id: &str,
    manifest: &Manifest,
) -> Option<Vec<SessionEvent>> {
    let store = store_dir(project, id)?;
    all_events_at(project, id, &store, manifest)
}

pub(crate) fn load_full(project: &Path, id: &str) -> Option<SessionData> {
    // P3 (AUDIT-2026-09-07): readers ran lockless — during a concurrent
    // persist's rotation window the manifest and the segments it points
    // to could be observed mid-rename and the load failed transiently
    // (the session looked broken while its data was fine). Readers now
    // take the store lock; it is exclusive, but reads are short, and no
    // writer path nests inside one (deadlock-checked).
    with_store_lock(project, id, || {
        let manifest = load_manifest(project, id)?;
        let events = all_events(project, id, &manifest)?;
        Some(session_data_from_manifest(&manifest, events))
    })
    .flatten()
}

pub(crate) fn load_tail(
    project: &Path,
    id: &str,
    limit: usize,
) -> Option<SessionData> {
    // P3 (AUDIT-2026-09-07): same lockless-reader race as load_full —
    // take the store lock for the whole multi-segment read.
    with_store_lock(project, id, || load_tail_locked(project, id, limit)).flatten()
}

fn load_tail_locked(
    project: &Path,
    id: &str,
    limit: usize,
) -> Option<SessionData> {
    let manifest = load_manifest(project, id)?;
    if limit == 0 {
        return Some(session_data_from_manifest(&manifest, Vec::new()));
    }
    let store = store_dir(project, id)?;
    let mut reverse = Vec::new();
    for meta in manifest.segments.iter().rev() {
        let stored = load_segment(project, &store, id, meta)?;
        for event in stored.into_iter().rev() {
            if event.seq >= manifest.retained_from_seq {
                reverse.push(event.into_event());
                if reverse.len() >= limit {
                    break;
                }
            }
        }
        if reverse.len() >= limit {
            break;
        }
    }
    reverse.reverse();
    Some(session_data_from_manifest(&manifest, reverse))
}

pub(crate) fn read_since(
    project: &Path,
    id: &str,
    after_seq: u64,
) -> Option<(u64, Vec<SessionEvent>)> {
    // P3 (AUDIT-2026-09-07): same lockless-reader race as load_full —
    // the peer watcher polled this without the store lock.
    with_store_lock(project, id, || read_since_locked(project, id, after_seq)).flatten()
}

fn read_since_locked(
    project: &Path,
    id: &str,
    after_seq: u64,
) -> Option<(u64, Vec<SessionEvent>)> {
    let manifest = load_manifest(project, id)?;
    let latest = manifest.next_seq.saturating_sub(1);
    if latest <= after_seq {
        return Some((latest, Vec::new()));
    }
    let store = store_dir(project, id)?;
    let mut out = Vec::new();
    for meta in &manifest.segments {
        if meta.end_seq <= after_seq || meta.end_seq < manifest.retained_from_seq {
            continue;
        }
        let stored = load_segment(project, &store, id, meta)?;
        for event in stored {
            if event.seq > after_seq && event.seq >= manifest.retained_from_seq {
                out.push(event.into_event());
            }
        }
    }
    out.sort_by_key(|event| event.seq);
    if out.windows(2).any(|pair| pair[1].seq == pair[0].seq) {
        return None;
    }
    Some((latest, out))
}

pub(crate) fn stats(project: &Path, id: &str) -> Option<StorageStats> {
    let manifest = load_manifest(project, id)?;
    let store = store_dir(project, id)?;
    let mut payload_bytes = 0u64;
    let mut sealed_count = 0usize;
    let mut active_events = 0u64;
    let mut active_bytes = 0u64;
    let mut raw_record_count = 0usize;
    let mut compressed_record_count = 0usize;
    let mut encrypted_record_count = 0usize;
    let mut missing_count = 0usize;
    let mut corrupt_count = 0usize;
    let mut referenced = std::collections::HashSet::new();

    if let Some(path) = manifest_path(project, id) {
        if let Some(profile) = record_profile(&path) {
            raw_record_count += usize::from(profile.raw);
            compressed_record_count += usize::from(profile.compressed);
            encrypted_record_count += usize::from(profile.encrypted);
        } else {
            corrupt_count += 1;
        }
    }

    for meta in &manifest.segments {
        payload_bytes = payload_bytes.saturating_add(meta.payload_bytes);
        if meta.state == "sealed" {
            sealed_count += 1;
        } else {
            active_events = meta.event_count;
            active_bytes = meta.payload_bytes;
        }
        referenced.insert(meta.file.clone());
        let Some(path) = safe_segment_path(&store, &meta.file) else {
            missing_count += 1;
            continue;
        };
        if !path.is_file() {
            missing_count += 1;
            continue;
        }
        let payload_valid = load_segment(project, &store, id, meta).is_some();
        if !payload_valid {
            corrupt_count += 1;
        }
        if let Some(profile) = record_profile(&path) {
            raw_record_count += usize::from(profile.raw);
            compressed_record_count += usize::from(profile.compressed);
            encrypted_record_count += usize::from(profile.encrypted);
        } else if payload_valid {
            corrupt_count += 1;
        }
    }

    let mut orphaned_count = 0usize;
    let mut quarantined_count = 0usize;
    let mut incomplete_count = 0usize;
    if let Ok(entries) = fs::read_dir(store.join(SEGMENTS_DIR)) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name.contains(".tmp") {
                incomplete_count += 1;
                continue;
            }
            if name.contains(".quarantine.") {
                quarantined_count += 1;
                continue;
            }
            if !path.is_file() || is_symlink(&path) || !safe_segment_filename(name) {
                continue;
            }
            let relative = format!("{SEGMENTS_DIR}/{name}");
            if !referenced.contains(&relative) {
                orphaned_count += 1;
            }
        }
    }

    Some(StorageStats {
        segmented: true,
        manifest_schema: manifest.storage_schema,
        segments: manifest.segments.clone(),
        record_count: manifest.segments.len() + 1,
        raw_record_count,
        compressed_record_count,
        encrypted_record_count,
        segment_count: manifest.segments.len(),
        sealed_count,
        active_events,
        active_bytes,
        retained_from_seq: manifest.retained_from_seq,
        next_seq: manifest.next_seq,
        event_count: manifest.event_count,
        payload_bytes,
        source_sha256: manifest.source_sha256.clone(),
        migrated_at: manifest.migrated_at,
        missing_count,
        corrupt_count,
        orphaned_count,
        quarantined_count,
        incomplete_count,
    })
}

#[derive(Clone)]
struct RepairCandidate {
    meta: SegmentMeta,
    events: Vec<StoredEvent>,
}

fn parse_segment_filename(name: &str) -> Option<(u64, Option<u64>, bool)> {
    let stem = name.strip_suffix(".qtsq")?;
    if let Some(value) = stem.strip_prefix("active-") {
        let parts: Vec<&str> = value.split('-').collect();
        if parts.len() != 1 && parts.len() != 3 {
            return None;
        }
        let start = parts[0];
        if start.len() != 8 || !start.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        if parts.len() == 3
            && (parts[1].is_empty()
                || parts[2].is_empty()
                || !parts[1].bytes().all(|b| b.is_ascii_hexdigit())
                || !parts[2].bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return None;
        }
        return Some((start.parse().ok()?, None, true));
    }
    let value = stem.strip_prefix("segment-")?;
    let (start, end) = value.split_once('-')?;
    if start.len() != 8
        || end.len() != 8
        || !start.bytes().all(|b| b.is_ascii_digit())
        || !end.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    Some((start.parse().ok()?, Some(end.parse().ok()?), false))
}

fn quarantine_segment(path: &Path, reason: &str) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    let base = format!("{name}.quarantine.{reason}");
    let mut target = parent.join(&base);
    let mut suffix = 0u32;
    while target.exists() {
        suffix = suffix.saturating_add(1);
        target = parent.join(format!("{base}.{suffix}"));
    }
    fs::rename(path, &target).is_ok()
}

fn valid_segment_payload(
    payload: &SegmentPayload,
    session_id: &str,
    bytes_len: usize,
) -> bool {
    if payload.format != SEGMENT_FORMAT
        || payload.storage_schema != STORAGE_SCHEMA
        || payload.session_id != session_id
        || payload.start_seq == 0
        || payload.start_seq > payload.end_seq
        || payload.events.len() as u64 != payload.end_seq - payload.start_seq + 1
    {
        return false;
    }
    let mut expected = payload.start_seq;
    for event in &payload.events {
        if event.seq != expected {
            return false;
        }
        let Some(next) = expected.checked_add(1) else {
            return false;
        };
        expected = next;
    }
    payload.events.last().map(|event| event.seq) == Some(payload.end_seq)
        && payload_bytes(payload).map(|bytes| bytes.len()) == Some(bytes_len)
}

fn repair_locked(project: &Path, id: &str) -> bool {
    if let Some(manifest) = load_manifest(project, id) {
        if all_events(project, id, &manifest).is_some() {
            return true;
        }
    }
    let Some(store) = store_dir(project, id) else {
        return false;
    };
    let segments = store.join(SEGMENTS_DIR);
    let mut candidates = Vec::new();
    let Ok(entries) = fs::read_dir(&segments) else {
        return false;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() || is_symlink(&path) {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !safe_segment_filename(name) {
            continue;
        }
        let Some((filename_start, filename_end, is_active)) =
            parse_segment_filename(name)
        else {
            continue;
        };
        let Some(bytes) = read_qtsq(project, &path) else {
            let _ = quarantine_segment(&path, "unreadable");
            continue;
        };
        let Ok(payload) = serde_json::from_slice::<SegmentPayload>(&bytes) else {
            let _ = quarantine_segment(&path, "json");
            continue;
        };
        if !valid_segment_payload(&payload, id, bytes.len())
            || payload.start_seq != filename_start
            || filename_end.is_some_and(|end| end != payload.end_seq)
        {
            let _ = quarantine_segment(&path, "invalid");
            continue;
        }
        let relative = format!("{SEGMENTS_DIR}/{name}");
        let state = if is_active { "active" } else { "sealed" };
        let meta = segment_meta(relative, state, &payload, bytes.len());
        candidates.push(RepairCandidate { meta, events: payload.events });
    }
    if candidates.is_empty() {
        return false;
    }
    candidates.sort_by_key(|candidate| {
        (
            candidate.meta.start_seq,
            candidate.meta.end_seq,
            candidate.meta.state != "sealed",
        )
    });

    let mut expected = candidates[0].meta.start_seq;
    let retained_from_seq = expected;
    let mut chosen = Vec::new();
    loop {
        let Some(candidate) = candidates
            .iter()
            .filter(|candidate| candidate.meta.start_seq == expected)
            .max_by_key(|candidate| {
                (
                    candidate.meta.end_seq,
                    candidate.meta.state == "sealed",
                )
            })
            .cloned()
        else {
            break;
        };
        expected = candidate.meta.end_seq.saturating_add(1);
        chosen.push(candidate);
        if expected == u64::MAX {
            break;
        }
    }
    if chosen.is_empty() {
        return false;
    }

    let prior = manifest_path(project, id)
        .and_then(|path| read_qtsq(project, &path))
        .and_then(|bytes| serde_json::from_slice::<Manifest>(&bytes).ok());
    let registry = crate::session::Registry::read(project);
    let info = registry.sessions.iter().find(|entry| entry.id == id);
    let last_event_t = chosen
        .last()
        .and_then(|candidate| candidate.events.last())
        .map(|event| event.t)
        .unwrap_or(0);
    let chosen_len = chosen.len();
    let mut metas = Vec::with_capacity(chosen_len);
    for (index, candidate) in chosen.into_iter().enumerate() {
        let mut meta = candidate.meta;
        meta.state = if index + 1 == chosen_len && meta.state == "active" {
            "active".into()
        } else {
            "sealed".into()
        };
        metas.push(meta);
    }
    let manifest = Manifest {
        format: MANIFEST_FORMAT.into(),
        storage_schema: STORAGE_SCHEMA,
        session_id: id.into(),
        created: prior
            .as_ref()
            .map(|value| value.created)
            .or_else(|| info.map(|value| value.started_at))
            .unwrap_or(last_event_t),
        host: prior
            .as_ref()
            .map(|value| value.host.clone())
            .or_else(|| info.map(|value| value.host.clone()))
            .unwrap_or_default(),
        cwd: prior
            .as_ref()
            .map(|value| value.cwd.clone())
            .or_else(|| info.map(|value| value.cwd.clone()))
            .unwrap_or_default(),
        model: prior
            .as_ref()
            .map(|value| value.model.clone())
            .or_else(|| info.map(|value| value.model.clone()))
            .unwrap_or_default(),
        provider: prior
            .as_ref()
            .map(|value| value.provider.clone())
            .or_else(|| info.map(|value| value.provider.clone()))
            .unwrap_or_default(),
        task: prior
            .as_ref()
            .and_then(|value| value.task.clone())
            .or_else(|| info.and_then(|value| value.task.clone())),
        notes: prior
            .as_ref()
            .map(|value| value.notes.clone())
            .unwrap_or_default(),
        next_seq: expected,
        retained_from_seq,
        event_count: expected.saturating_sub(retained_from_seq),
        last_event_t,
        source_sha256: prior
            .as_ref()
            .and_then(|value| value.source_sha256.clone()),
        migrated_at: prior.as_ref().and_then(|value| value.migrated_at),
        segments: metas,
    };
    if manifest.validate().is_err() {
        return false;
    }
    publish_manifest(project, id, &manifest)
}

pub(crate) fn repair(project: &Path, id: &str) -> bool {
    let Some(store) = store_dir(project, id) else {
        return false;
    };
    if !store.exists() || is_symlink(&store) {
        return false;
    }
    with_store_lock(project, id, || repair_locked(project, id)).unwrap_or(false)
}

pub(crate) fn migrate_legacy(project: &Path, id: &str) -> Result<(), String> {
    if !crate::session::valid_session_id(id) {
        return Err("invalid session id".into());
    }
    if has_manifest(project, id) {
        return Ok(());
    }
    let final_store = store_dir(project, id).ok_or_else(|| "invalid store path".to_string())?;
    if final_store.exists() || is_symlink(&final_store) {
        return Err("segmented store exists but has no valid manifest; run repair first".into());
    }
    let legacy = crate::session::sessions_dir(project).join(format!("{id}.qtsq"));
    let bytes = read_qtsq(project, &legacy).ok_or_else(|| "legacy session is unreadable".to_string())?;
    let data: SessionData =
        serde_json::from_slice(&bytes).map_err(|_| "legacy session JSON is invalid".to_string())?;
    if data.id != id {
        return Err("legacy session id does not match requested id".into());
    }

    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let stage = final_store.with_file_name(format!(
        ".{id}.store.migrate.{}.{}",
        std::process::id(),
        counter
    ));
    if stage.exists() || is_symlink(&stage) {
        return Err("migration staging path already exists".into());
    }
    if !create_store_at(project, &data, &stage) {
        let _ = fs::remove_dir_all(&stage);
        return Err("could not write staged segmented store".into());
    }
    let Some(mut staged_manifest) = load_manifest_at(project, id, &manifest_path_at(&stage)) else {
        let _ = fs::remove_dir_all(&stage);
        return Err("staged manifest failed validation".into());
    };
    staged_manifest.source_sha256 = sofuu_ffi::qtsq_sha256_hex(&bytes);
    staged_manifest.migrated_at = Some(unix_now());
    if !staged_manifest.validate().is_ok()
        || !publish_manifest_at(&manifest_path_at(&stage), &staged_manifest)
    {
        let _ = fs::remove_dir_all(&stage);
        return Err("staged migration metadata failed validation".into());
    }
    if all_events_at(project, id, &stage, &staged_manifest).is_none() {
        let _ = fs::remove_dir_all(&stage);
        return Err("staged segments failed validation".into());
    }
    if final_store.exists()
        || is_symlink(&final_store)
        || fs::rename(&stage, &final_store).is_err()
    {
        let _ = fs::remove_dir_all(&stage);
        return Err("could not publish segmented store".into());
    }
    sync_dir(&crate::session::sessions_dir(project));
    Ok(())
}

pub(crate) fn remove_store(project: &Path, id: &str) -> bool {
    let Some(store) = store_dir(project, id) else {
        return false;
    };
    if !store.exists() || is_symlink(&store) {
        return false;
    }
    fs::remove_dir_all(store).is_ok()
}
