//! Real Kaze memory backends over Sofuu's actual stores.
//!
//! Two backends exist and BOTH are real stores — neither fabricates a
//! recall and neither is a simulated result:
//!
//! * `EphemeralMemoryHost` (`memory=ephemeral`) — Sofuu's in-process CMA
//!   (`crate::memory::cma::Cma`). A real vector store that lives for exactly
//!   one run: NOT durable, NOT shared with another process, and NOT the
//!   durable store. Its records are gone when the process exits, by design.
//! * `DurableMemoryHost` (`memory=durable`, `memory=file:<path>`) — a real
//!   file-backed record store. Every `save` is committed to disk (staged,
//!   fsynced, renamed over the target) before the call returns, so the
//!   record survives the process and the next process that opens the same
//!   path reads it back. Each path is its own store: two paths never see one
//!   another's records, and neither ever sees the ephemeral CMA.
//!
//! `memory.save(label, value)` stores JSON-compatible data; `memory.recall`
//! returns the values whose label matches exactly. The VM has already
//! enforced declarations, grants and policy before calling this host.
//!
//! The contract for both backends: an operation either performs the
//! documented read or write against the SELECTED store, or fails with a
//! typed `E_KAZE_MEMORY` error. A store that exists but cannot be read is an
//! ERROR, never an empty answer — "this label has no record" and "this store
//! cannot be read" are different outcomes and are never conflated.
//!
//! ## The stored contract
//!
//! * **Key / query semantics.** The label is the key. `memory.recall`
//!   matches the label EXACTLY (byte for byte) and returns every value
//!   stored under it, in the order the records were written. There is no
//!   fuzzy, prefix, substring or embedding-similarity match: the CMA's
//!   approximate vector retrieval is deliberately not used for recall.
//! * **Value encoding.** A value is any JSON-compatible Kaze value: null
//!   (unit), a boolean, a finite number, text, a list, or a record with text
//!   keys. A non-finite number, a non-text record key, a duplicate record
//!   key, a value nested deeper than [`MAX_VALUE_DEPTH`] or one with more
//!   than [`MAX_VALUE_NODES`] nodes is refused before anything is stored.
//!   The durable store writes the records as one JSON array; the ephemeral
//!   store keeps the same JSON object per record inside the CMA.
//! * **Isolation.** One store per path, per process. A durable path never
//!   sees another path's records, and neither ever sees the ephemeral CMA.
//!   Two writers on one durable path are arbitrated by an exclusive lock on
//!   `<path>.lock`; a writer that cannot take that lock is refused rather
//!   than allowed to interleave a read-modify-write.
//! * **Durability.** `memory=ephemeral` is NOT durable and says so in the
//!   trace. `memory=durable` and `memory=file:<path>` commit every record to
//!   disk before the call returns; the claim is only proved by reopening the
//!   same path in a SEPARATE process and reading the record back
//!   (`scripts/test-kaze-memory-durable.py`).
//! * **Size bounds.** A label is 1..=[`MAX_LABEL_BYTES`] bytes; a store
//!   holds at most [`MAX_RECORDS`] records and at most [`MAX_BYTES`] bytes;
//!   a value nested deeper than [`MAX_VALUE_DEPTH`] or with more than
//!   [`MAX_VALUE_NODES`] nodes is refused. Exceeding a bound is an error,
//!   never a silent truncation.

use std::path::PathBuf;

use serde_json::{Map, Number, Value as Json};

use crate::memory::cma::Cma;
use crate::sfs::arena::Arena;
use crate::sfs::caps::{Action, ActionMode, CapHost, RuntimeHost};
use crate::sfs::value::{Heap, Value};

const DIM: usize = 64;
const MAX_RECORDS: usize = 256;
const MAX_BYTES: usize = 256 * 1024;
const MAX_LABEL_BYTES: usize = 1024;
const MAX_VALUE_DEPTH: usize = 16;
const MAX_VALUE_NODES: usize = 4096;

/// The backend names the trace records, so a reader can tell a per-run
/// store from a committed one without reading the prose beside the line.
const EPHEMERAL_MEMORY_BACKEND: &str = "memory-ephemeral";
const DURABLE_MEMORY_BACKEND: &str = "memory-durable";

fn validate_label(value: Option<&Value>, arena: &Arena) -> Result<String, String> {
    let Some(Value::Str(text)) = value else {
        return Err("E_KAZE_MEMORY: memory label must be nonempty text".into());
    };
    let text = arena.get(*text);
    if text.is_empty() || text.len() > MAX_LABEL_BYTES {
        return Err(format!(
            "E_KAZE_MEMORY: memory label must contain 1 to {} bytes",
            MAX_LABEL_BYTES
        ));
    }
    Ok(text.to_string())
}

pub(crate) struct EphemeralMemoryHost {
    base: RuntimeHost,
    store: Cma,
    accepted_bytes: usize,
}

impl EphemeralMemoryHost {
    pub(crate) fn new(base: RuntimeHost) -> Self {
        let mut store = Cma::new(DIM);
        // Do not merge different values merely because their embeddings are
        // similar. Identical label/value records are still deduplicated by CMA.
        store.set_dup_gate(false);
        Self {
            base,
            store,
            accepted_bytes: 0,
        }
    }

    fn label(value: Option<&Value>, arena: &Arena) -> Result<String, String> {
        validate_label(value, arena)
    }

    fn save(&mut self, args: &[Value], arena: &Arena, heap: &Heap) -> Result<Value, String> {
        if args.len() != 2 {
            return Err("E_KAZE_MEMORY: memory.save requires a label and a value".into());
        }
        let label = Self::label(args.first(), arena)?;
        let mut nodes = 0;
        let value = to_json(args[1], arena, heap, 0, &mut nodes)?;
        let record = serde_json::json!({"label": label, "value": value}).to_string();
        if self.store.len() >= MAX_RECORDS
            || self.accepted_bytes.saturating_add(record.len()) > MAX_BYTES
        {
            return Err("E_KAZE_MEMORY: per-run ephemeral memory limit reached".into());
        }
        let vector = crate::embedding::hash_features_at(&label, DIM);
        self.store.remember_outcome(&vector, &record, "kaze", 0);
        self.accepted_bytes += record.len();
        self.base.note_with_backend(
            "memory",
            "save",
            &["<redacted>".into()],
            "stored in the per-run ephemeral CMA store (in-process, not durable)".into(),
            ActionMode::Real,
            EPHEMERAL_MEMORY_BACKEND,
        );
        Ok(Value::Unit)
    }

    fn recall(
        &mut self,
        args: &[Value],
        arena: &mut Arena,
        heap: &mut Heap,
    ) -> Result<Value, String> {
        if args.len() != 1 {
            return Err("E_KAZE_MEMORY: memory.recall requires one label".into());
        }
        let label = Self::label(args.first(), arena)?;
        let mut matches = Vec::new();
        // Exact matching is deliberately distinct from CMA's approximate
        // vector retrieval. This run owns all records in this ephemeral CMA.
        for stored in &self.store.records {
            let record: Json = serde_json::from_str(&stored.text)
                .map_err(|_| "E_KAZE_MEMORY: unreadable in-memory record".to_string())?;
            if record.get("label").and_then(Json::as_str) == Some(label.as_str()) {
                let value = record
                    .get("value")
                    .ok_or_else(|| "E_KAZE_MEMORY: invalid in-memory record".to_string())?;
                matches.push(
                    crate::sfs::json_to_value(value, arena, heap)
                        .map_err(|e| format!("E_KAZE_MEMORY: {}", e))?,
                );
            }
        }
        let count = matches.len();
        let value = heap
            .push_array(matches)
            .map_err(|e| format!("E_KAZE_MEMORY: {}", e))?;
        self.base.note_with_backend(
            "memory",
            "recall",
            &["<redacted>".into()],
            format!("{count} exact match(es) in the per-run ephemeral CMA store (not durable)"),
            ActionMode::Real,
            EPHEMERAL_MEMORY_BACKEND,
        );
        Ok(value)
    }
}

impl CapHost for EphemeralMemoryHost {
    fn call(
        &mut self,
        family: &str,
        op: &str,
        args: &[Value],
        arena: &mut Arena,
        heap: &mut Heap,
    ) -> Result<Value, String> {
        match (family, op) {
            ("memory", "save") => {
                self.base.check_trace_capacity()?;
                self.save(args, arena, heap)
            }
            ("memory", "recall") => {
                self.base.check_trace_capacity()?;
                self.recall(args, arena, heap)
            }
            _ => self.base.call(family, op, args, arena, heap),
        }
    }

    fn actions(&self) -> &[Action] {
        self.base.actions()
    }
    fn logs(&self) -> &[(String, String)] {
        self.base.logs()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredRecord {
    label: String,
    value: Json,
}

pub(crate) struct DurableMemoryHost {
    base: RuntimeHost,
    path: PathBuf,
    /// Exclusive cross-process lock for the WRITE path. It is taken lazily on
    /// the first `save` — a read-only run leaves no side files — and held for
    /// the rest of the run, so two writers cannot interleave a
    /// read-modify-write and silently lose one of the two writes.
    lock: Option<crate::rt::memory::brain_lock::BrainLock>,
}

impl DurableMemoryHost {
    pub(crate) fn new(base: RuntimeHost, path: PathBuf) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("E_KAZE_MEMORY: cannot create storage directory {}: {}", parent.display(), e))?;
            }
        }
        Ok(Self {
            base,
            path,
            lock: None,
        })
    }

    /// Read the store's records.
    ///
    /// A store that does not exist yet is an empty store — a genuine empty
    /// result. A store that EXISTS but cannot be read is a FAILURE and is
    /// reported as one: an unreadable store and an empty store are different
    /// facts, and a read error must never be presented as "no records".
    fn read_records(&self) -> Result<Vec<StoredRecord>, String> {
        /* The store's size is bounded by the contract, so a file that grew
         * past the bound is refused BEFORE it is loaded: the limit is a
         * guarantee about what the store holds, not a number checked after
         * the whole file is already in memory. */
        let meta = match std::fs::metadata(&self.path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(format!(
                    "E_KAZE_MEMORY: cannot read durable memory store {}: {}",
                    self.path.display(),
                    e
                ))
            }
        };
        if meta.len() > MAX_BYTES as u64 {
            return Err(format!(
                "E_KAZE_MEMORY: durable memory store {} is {} bytes — larger than the {} byte limit",
                self.path.display(),
                meta.len(),
                MAX_BYTES
            ));
        }
        let bytes = match std::fs::read(&self.path) {
            Ok(b) => b,
            Err(e) => {
                return Err(format!(
                    "E_KAZE_MEMORY: cannot read durable memory store {}: {}",
                    self.path.display(),
                    e
                ))
            }
        };
        if bytes.is_empty() {
            return Err(format!(
                "E_KAZE_MEMORY: durable memory store {} is empty — it is not a readable store",
                self.path.display()
            ));
        }
        serde_json::from_slice(&bytes)
            .map_err(|e| format!("E_KAZE_MEMORY: corrupt storage data in {}: {}", self.path.display(), e))
    }

    /// Commit the records to the store: stage beside the target, fsync the
    /// staged bytes, rename over the target, fsync the directory. The rename
    /// is atomic, so a reader sees either the previous file or the new one —
    /// never a half-written store — and the bytes are on stable storage
    /// before the call reports success.
    fn write_records(&mut self, records: &[StoredRecord]) -> Result<(), String> {
        let serialized = serde_json::to_vec(records)
            .map_err(|e| format!("E_KAZE_MEMORY: failed to serialize records: {}", e))?;
        if serialized.len() > MAX_BYTES {
            return Err("E_KAZE_MEMORY: durable memory store size limit reached".into());
        }
        let mut tmp_name = self.path.clone().into_os_string();
        tmp_name.push(format!(".tmp.{}", std::process::id()));
        let tmp_path = PathBuf::from(tmp_name);
        std::fs::write(&tmp_path, &serialized)
            .map_err(|e| format!("E_KAZE_MEMORY: cannot write temporary storage file {}: {}", tmp_path.display(), e))?;
        if !crate::rt::wal::publish_paths(&tmp_path, &self.path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(format!(
                "E_KAZE_MEMORY: cannot commit durable memory store {}",
                self.path.display()
            ));
        }
        Ok(())
    }

    /// Take the exclusive writer lock for this store. A second writer is
    /// refused with a typed error instead of being allowed to interleave a
    /// read-modify-write with this one.
    ///
    /// A lock that could not be taken at all is just as bad as a conflicting
    /// one: the write below is a read-modify-write, and without arbitration
    /// two writers can each read the old records and each publish their own
    /// set — one record silently lost, and both writes reported as
    /// successful. An unarbitrated writer is refused.
    fn ensure_writer(&mut self) -> Result<(), String> {
        if self.lock.is_some() {
            return Ok(());
        }
        let path = self.path.to_string_lossy().to_string();
        match crate::rt::memory::brain_lock::acquire(
            &path,
            crate::rt::memory::brain_lock::Mode::Exclusive,
        ) {
            Ok(lock) => {
                if let Some(reason) = lock.degraded() {
                    return Err(format!(
                        "E_KAZE_MEMORY: durable memory store {} cannot arbitrate concurrent writers: {}",
                        self.path.display(),
                        reason
                    ));
                }
                self.lock = Some(lock);
                Ok(())
            }
            Err(e) => Err(format!(
                "E_KAZE_MEMORY: durable memory store {} is held by another writer: {}",
                self.path.display(),
                e
            )),
        }
    }

    fn save(&mut self, args: &[Value], arena: &Arena, heap: &Heap) -> Result<Value, String> {
        if args.len() != 2 {
            return Err("E_KAZE_MEMORY: memory.save requires a label and a value".into());
        }
        let label = validate_label(args.first(), arena)?;
        let mut nodes = 0;
        let value = to_json(args[1], arena, heap, 0, &mut nodes)?;
        self.ensure_writer()?;
        let mut records = self.read_records()?;
        if records.len() >= MAX_RECORDS {
            return Err("E_KAZE_MEMORY: durable memory store record limit reached".into());
        }
        records.push(StoredRecord { label, value });
        self.write_records(&records)?;
        self.base.note_with_backend(
            "memory",
            "save",
            &["<redacted>".into()],
            "stored in the durable file store (committed to disk)".into(),
            ActionMode::Real,
            DURABLE_MEMORY_BACKEND,
        );
        Ok(Value::Unit)
    }

    fn recall(
        &mut self,
        args: &[Value],
        arena: &mut Arena,
        heap: &mut Heap,
    ) -> Result<Value, String> {
        if args.len() != 1 {
            return Err("E_KAZE_MEMORY: memory.recall requires one label".into());
        }
        let label = validate_label(args.first(), arena)?;
        let records = self.read_records()?;
        let mut matches = Vec::new();
        for record in records {
            if record.label == label {
                matches.push(
                    crate::sfs::json_to_value(&record.value, arena, heap)
                        .map_err(|e| format!("E_KAZE_MEMORY: {}", e))?,
                );
            }
        }
        let count = matches.len();
        let value = heap
            .push_array(matches)
            .map_err(|e| format!("E_KAZE_MEMORY: {}", e))?;
        self.base.note_with_backend(
            "memory",
            "recall",
            &["<redacted>".into()],
            format!("{count} exact match(es) in the durable file store"),
            ActionMode::Real,
            DURABLE_MEMORY_BACKEND,
        );
        Ok(value)
    }
}

impl CapHost for DurableMemoryHost {
    fn call(
        &mut self,
        family: &str,
        op: &str,
        args: &[Value],
        arena: &mut Arena,
        heap: &mut Heap,
    ) -> Result<Value, String> {
        match (family, op) {
            ("memory", "save") => {
                self.base.check_trace_capacity()?;
                self.save(args, arena, heap)
            }
            ("memory", "recall") => {
                self.base.check_trace_capacity()?;
                self.recall(args, arena, heap)
            }
            _ => self.base.call(family, op, args, arena, heap),
        }
    }

    fn actions(&self) -> &[Action] {
        self.base.actions()
    }
    fn logs(&self) -> &[(String, String)] {
        self.base.logs()
    }
}

fn to_json(
    value: Value,
    arena: &Arena,
    heap: &Heap,
    depth: usize,
    nodes: &mut usize,
) -> Result<Json, String> {
    *nodes += 1;
    if depth > MAX_VALUE_DEPTH || *nodes > MAX_VALUE_NODES {
        return Err("E_KAZE_MEMORY: memory value is too deeply nested or complex".into());
    }
    match value {
        Value::Unit => Ok(Json::Null),
        Value::Bool(b) => Ok(Json::Bool(b)),
        Value::Num(n) => Number::from_f64(n)
            .map(Json::Number)
            .ok_or_else(|| "E_KAZE_MEMORY: memory numbers must be finite".into()),
        Value::Str(s) => Ok(Json::String(arena.get(s).to_string())),
        Value::Arr(i) => {
            let entries = heap
                .array(i)
                .ok_or_else(|| "E_KAZE_MEMORY: invalid list".to_string())?;
            let mut result = Vec::with_capacity(entries.len());
            for entry in entries {
                result.push(to_json(*entry, arena, heap, depth + 1, nodes)?);
            }
            Ok(Json::Array(result))
        }
        Value::Map(i) => {
            let entries = heap
                .map(i)
                .ok_or_else(|| "E_KAZE_MEMORY: invalid record".to_string())?;
            let mut result = Map::new();
            for (key, entry) in entries {
                let Value::Str(key) = key else {
                    return Err("E_KAZE_MEMORY: record keys must be text".into());
                };
                let key = arena.get(*key).to_string();
                if result.contains_key(&key) {
                    return Err("E_KAZE_MEMORY: duplicate record key".into());
                }
                result.insert(key, to_json(*entry, arena, heap, depth + 1, nodes)?);
            }
            Ok(Json::Object(result))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfs::caps::SimState;

    fn base_host() -> RuntimeHost {
        RuntimeHost::new(SimState::default(), Vec::new(), Vec::new())
    }

    fn tmp_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("sofuu-kaze-memory-{tag}-{}-{n}", std::process::id()));
        p
    }

    fn durable(path: PathBuf) -> DurableMemoryHost {
        DurableMemoryHost::new(base_host(), path).expect("the store directory must be creatable")
    }

    fn save_text(host: &mut dyn CapHost, label: &str, value: &str) -> Result<(), String> {
        let mut arena = Arena::new(1024);
        let mut heap = Heap::default();
        let l = Value::Str(arena.alloc(label)?);
        let v = Value::Str(arena.alloc(value)?);
        host.call("memory", "save", &[l, v], &mut arena, &mut heap)
            .map(|_| ())
    }

    fn recall_texts(host: &mut dyn CapHost, label: &str) -> Result<Vec<String>, String> {
        let mut arena = Arena::new(1024);
        let mut heap = Heap::default();
        let l = Value::Str(arena.alloc(label)?);
        let out = host.call("memory", "recall", &[l], &mut arena, &mut heap)?;
        let Value::Arr(idx) = out else {
            return Err("recall must return a list".into());
        };
        let entries = heap.array(idx).ok_or("recall must return a list")?;
        let mut texts = Vec::new();
        for entry in entries {
            let Value::Str(s) = entry else {
                return Err("recall must return text values".into());
            };
            texts.push(arena.get(*s).to_string());
        }
        Ok(texts)
    }

    #[test]
    fn a_full_trace_refuses_the_memory_write_before_it_occurs() {
        let base = RuntimeHost::new(SimState::default(), Vec::new(), Vec::new());
        let mut host = EphemeralMemoryHost::new(base);
        let mut arena = Arena::new(1024);
        let mut heap = Heap::default();
        let detector = Value::Str(arena.alloc("secrets").unwrap());
        let data = Value::Str(arena.alloc("value").unwrap());
        let label = Value::Str(arena.alloc("label").unwrap());
        let mut full = false;
        for _ in 0..10_000 {
            if host.call("policy", "detects", &[detector, data], &mut arena, &mut heap).is_err() {
                full = true;
                break;
            }
        }
        assert!(full, "the trace must have a finite limit");
        let error = host.call("memory", "save", &[label, data], &mut arena, &mut heap).unwrap_err();
        assert!(error.starts_with("E_KAZE_MEMORY:"));
        assert_eq!(host.store.len(), 0);
    }

    /// The ephemeral store is a REAL store: what `save` accepted is what
    /// `recall` returns, and it is a per-run store — a second run's store
    /// never sees it.
    #[test]
    fn the_ephemeral_store_returns_what_it_stored_and_is_per_run() {
        let mut host = EphemeralMemoryHost::new(base_host());
        save_text(&mut host, "label-a", "value-a").expect("the ephemeral store must accept a record");
        assert_eq!(recall_texts(&mut host, "label-a").unwrap(), vec!["value-a".to_string()]);
        assert!(recall_texts(&mut host, "label-b").unwrap().is_empty());

        let mut other_run = EphemeralMemoryHost::new(base_host());
        assert!(
            recall_texts(&mut other_run, "label-a").unwrap().is_empty(),
            "a per-run store must not leak into the next run"
        );
    }

    /// The label is the key and the match is exact: every value stored under
    /// one label comes back, in the order it was written, and a label that
    /// merely resembles another never does.
    #[test]
    fn recall_matches_the_label_exactly_and_returns_every_value() {
        let mut host = EphemeralMemoryHost::new(base_host());
        save_text(&mut host, "label-a", "value-1").expect("the ephemeral store must accept a record");
        save_text(&mut host, "label-a", "value-2").expect("the ephemeral store must accept a record");
        save_text(&mut host, "label-b", "value-3").expect("the ephemeral store must accept a record");
        assert_eq!(
            recall_texts(&mut host, "label-a").unwrap(),
            vec!["value-1".to_string(), "value-2".to_string()]
        );
        assert_eq!(
            recall_texts(&mut host, "label-b").unwrap(),
            vec!["value-3".to_string()]
        );
        assert!(
            recall_texts(&mut host, "label-a-extra").unwrap().is_empty(),
            "an exact match must not widen to a prefix"
        );
    }

    /// A value that cannot be encoded is refused before anything is stored.
    #[test]
    fn a_value_that_cannot_be_encoded_is_refused() {
        let mut host = EphemeralMemoryHost::new(base_host());
        let mut arena = Arena::new(1024);
        let mut heap = Heap::default();
        let label = Value::Str(arena.alloc("label-a").unwrap());
        let value = Value::Num(f64::NAN);
        let error = host.call("memory", "save", &[label, value], &mut arena, &mut heap).unwrap_err();
        assert!(error.starts_with("E_KAZE_MEMORY:"), "{error}");
        assert!(error.contains("finite"), "{error}");
        assert_eq!(host.store.len(), 0, "a refused value must not be stored");
    }

    /// The trace must name the store that answered, so a reader can tell a
    /// per-run store from a committed one without inferring it from prose.
    #[test]
    fn the_trace_names_the_store_that_answered() {
        let mut ephemeral_host = EphemeralMemoryHost::new(base_host());
        save_text(&mut ephemeral_host, "label-a", "value-a")
            .expect("the ephemeral store must accept a record");
        assert_eq!(ephemeral_host.actions()[0].backend, EPHEMERAL_MEMORY_BACKEND);

        let path = tmp_path("trace").join("store.json");
        let mut durable_host = durable(path.clone());
        save_text(&mut durable_host, "label-a", "value-a")
            .expect("the durable store must accept a record");
        assert_eq!(durable_host.actions()[0].backend, DURABLE_MEMORY_BACKEND);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A store larger than the documented bound is refused instead of being
    /// loaded whole: the bound is a guarantee about the store, not a number
    /// checked after the file is already in memory.
    #[test]
    fn a_store_larger_than_the_limit_is_refused() {
        let path = tmp_path("oversized");
        std::fs::create_dir_all(&path).expect("the temporary directory must be creatable");
        let store = path.join("store.json");
        std::fs::write(&store, vec![b'x'; MAX_BYTES + 1]).expect("the fixture must be writable");
        let mut host = durable(store.clone());
        let error = recall_texts(&mut host, "label-a").unwrap_err();
        assert!(error.starts_with("E_KAZE_MEMORY:"), "{error}");
        assert!(error.contains("larger than the"), "{error}");
        let _ = std::fs::remove_dir_all(&path);
    }

    /// A writer that cannot take the store's writer lock is refused. The
    /// durable write is a read-modify-write: two unarbitrated writers would
    /// each publish their own set and one record would vanish, with both
    /// writes reported as successful.
    #[test]
    fn an_unarbitrated_writer_is_refused() {
        let path = tmp_path("unarbitrated");
        std::fs::create_dir_all(&path).expect("the temporary directory must be creatable");
        let store = path.join("store.json");
        // A directory where the lock file belongs: the lock cannot be taken.
        let lock = crate::rt::memory::brain_lock::lock_path(&store.to_string_lossy().to_string());
        std::fs::create_dir_all(&lock).expect("the decoy must be creatable");
        let mut host = durable(store.clone());
        let error = save_text(&mut host, "label-a", "value-a").unwrap_err();
        assert!(error.starts_with("E_KAZE_MEMORY:"), "{error}");
        assert!(error.contains("cannot arbitrate concurrent writers"), "{error}");
        assert!(
            !store.exists(),
            "a refused write must not have created the store"
        );
        let _ = std::fs::remove_dir_all(&path);
    }

    /// A durable `save` must be readable by the NEXT process that opens the
    /// same path — the claim the durable backend makes.
    #[test]
    fn the_durable_store_is_readable_by_a_later_open_of_the_same_path() {
        let path = tmp_path("reopen").join("store.json");
        {
            let mut host = durable(path.clone());
            save_text(&mut host, "label-a", "value-a").expect("the durable store must accept a record");
        }
        let mut reopened = durable(path.clone());
        assert_eq!(
            recall_texts(&mut reopened, "label-a").unwrap(),
            vec!["value-a".to_string()],
            "a reopened durable store must return the record that was written"
        );
        assert!(recall_texts(&mut reopened, "label-b").unwrap().is_empty());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A durable store that does not exist yet is EMPTY — a genuine empty
    /// result, not a failure.
    #[test]
    fn an_absent_durable_store_recalls_nothing() {
        let path = tmp_path("absent").join("store.json");
        let mut host = durable(path.clone());
        assert!(recall_texts(&mut host, "label-a").unwrap().is_empty());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A store that EXISTS but cannot be read is a FAILURE. Reporting it as
    /// "no records" would be a fabricated recall result.
    #[test]
    fn an_unreadable_durable_store_fails_instead_of_recalling_nothing() {
        let path = tmp_path("unreadable");
        std::fs::create_dir_all(&path).expect("the temporary directory must be creatable");
        // A directory where the store file is expected: reading it fails.
        let store = path.join("store.json");
        std::fs::create_dir_all(&store).expect("the decoy must be creatable");
        let mut host = durable(store.clone());
        let error = recall_texts(&mut host, "label-a").unwrap_err();
        assert!(error.starts_with("E_KAZE_MEMORY:"), "{error}");
        assert!(
            error.contains("cannot read durable memory store"),
            "a read failure must say so: {error}"
        );
        let _ = std::fs::remove_dir_all(&path);
    }

    /// A store whose bytes cannot be parsed is corrupt — never an empty
    /// result.
    #[test]
    fn a_corrupt_durable_store_fails_closed() {
        let path = tmp_path("corrupt");
        std::fs::create_dir_all(&path).expect("the temporary directory must be creatable");
        let store = path.join("store.json");
        std::fs::write(&store, b"{not json").expect("the fixture must be writable");
        let mut host = durable(store.clone());
        let error = recall_texts(&mut host, "label-a").unwrap_err();
        assert!(error.starts_with("E_KAZE_MEMORY:"), "{error}");
        assert!(error.contains("corrupt storage data"), "{error}");
        let _ = std::fs::remove_dir_all(&path);
    }

    /// A write that cannot reach the disk is an ERROR — never a success.
    #[test]
    fn a_durable_write_that_cannot_reach_the_disk_fails_closed() {
        let path = tmp_path("unwritable");
        std::fs::create_dir_all(&path).expect("the temporary directory must be creatable");
        let store = path.join("store.json");
        std::fs::create_dir_all(&store).expect("the decoy must be creatable");
        let mut host = durable(store.clone());
        let error = save_text(&mut host, "label-a", "value-a").unwrap_err();
        assert!(error.starts_with("E_KAZE_MEMORY:"), "{error}");
        let _ = std::fs::remove_dir_all(&path);
    }

    /// Two durable stores are two stores: neither sees the other's records,
    /// and neither sees the ephemeral CMA.
    #[test]
    fn durable_stores_are_isolated_from_each_other_and_from_the_ephemeral_store() {
        let first = tmp_path("isolated-a").join("store.json");
        let second = tmp_path("isolated-b").join("store.json");
        let mut a = durable(first.clone());
        let mut b = durable(second.clone());
        save_text(&mut a, "label-a", "value-a").expect("the first store must accept a record");
        assert!(recall_texts(&mut b, "label-a").unwrap().is_empty());

        let mut ephemeral = EphemeralMemoryHost::new(base_host());
        save_text(&mut ephemeral, "label-a", "ephemeral-value").expect("the ephemeral store must accept a record");
        assert_eq!(recall_texts(&mut ephemeral, "label-a").unwrap(), vec!["ephemeral-value".to_string()]);
        assert!(
            recall_texts(&mut a, "label-a").unwrap() != vec!["ephemeral-value".to_string()],
            "the ephemeral store must not leak into a durable store"
        );
        let _ = std::fs::remove_dir_all(first.parent().unwrap());
        let _ = std::fs::remove_dir_all(second.parent().unwrap());
    }

    /// The durable claim, checked the only way that proves it: a DIFFERENT
    /// PROCESS writes the record and exits, and this process then opens the
    /// same path and reads the record back.
    #[test]
    fn a_durable_record_survives_the_process_that_wrote_it() {
        // The child process: write one record into the store the parent
        // named, then exit.
        if let Ok(child_store) = std::env::var("SOFUU_SLICE9_DURABLE_STORE") {
            let mut host = durable(PathBuf::from(child_store));
            save_text(&mut host, "cross-process-label", "cross-process-value")
                .expect("the child must be able to write");
            return;
        }

        let path = tmp_path("cross-process");
        std::fs::create_dir_all(&path).expect("the temporary directory must be creatable");
        let store = path.join("store.json");

        let exe = std::env::current_exe().expect("the test binary path must be known");
        let status = std::process::Command::new(exe)
            .args([
                "--exact",
                "rt::kaze_memory::tests::a_durable_record_survives_the_process_that_wrote_it",
                "--nocapture",
            ])
            .env("SOFUU_SLICE9_DURABLE_STORE", &store)
            .status()
            .expect("the child process must start");
        assert!(status.success(), "the writing process must exit cleanly");
        assert!(store.exists(), "the durable store must exist on disk after the writer exited");

        let mut reopened = durable(store.clone());
        assert_eq!(
            recall_texts(&mut reopened, "cross-process-label").unwrap(),
            vec!["cross-process-value".to_string()],
            "a separate process must read back the record the writer actually wrote"
        );
        let _ = std::fs::remove_dir_all(&path);
    }
}
