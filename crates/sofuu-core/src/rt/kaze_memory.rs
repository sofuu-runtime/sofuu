//! Explicit per-run Kaze adapter over Sofuu's actual in-memory CMA.
//!
//! `memory.save(label, value)` stores JSON-compatible data; `memory.recall`
//! returns the values whose label matches exactly. This deliberately does not
//! promise persistence, semantic search, or cross-run sharing. The VM has
//! already enforced declarations, grants and policy before calling this host.

use serde_json::{Map, Number, Value as Json};

use crate::memory::cma::Cma;
use crate::sfs::arena::Arena;
use crate::sfs::caps::{Action, CapHost, RuntimeHost};
use crate::sfs::value::{Heap, Value};

const DIM: usize = 64;
const MAX_RECORDS: usize = 256;
const MAX_BYTES: usize = 256 * 1024;
const MAX_LABEL_BYTES: usize = 1024;
const MAX_VALUE_DEPTH: usize = 16;
const MAX_VALUE_NODES: usize = 4096;

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
        self.base.note(
            "memory",
            "save",
            &["<redacted>".into()],
            "stored in per-run CMA".into(),
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
        self.base.note(
            "memory",
            "recall",
            &["<redacted>".into()],
            format!("{} exact match(es) in per-run CMA", count),
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
}
