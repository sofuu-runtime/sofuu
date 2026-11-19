// rt/storage.rs — Native QTSQ Structured Storage Driver (CMA v2 / Phase 3).
//
// Implements strict columnar table storage, weekly database container rotation
// (storage_YYYY_Www.qtsq), manifest index tracking, hardware AES-256-GCM vault
// encryption at rest, in-memory working cache, and debounced auto-flushing.
//
// Author: Haruhito

#![allow(clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use sofuu_ffi::bridge::{
    js_new_bool, js_new_string, js_to_string, register_global_fn, JSContext,
    JSValue, JSValueConst,
};
use sofuu_ffi::qjs;
use sofuu_ffi::qtsq_table::{
    SofuQuery, SofuSchema, SofuTable, SofuValue, SofuuColType, SOFUU_COL_AUTO_INC,
    SOFUU_COL_INDEXED, SOFUU_COL_PRIMARY_KEY, SOFUU_COL_REQUIRED,
};

/// Format Unix epoch milliseconds to ISO-8601 calendar (Year, ISO Week Number).
/// Pure integer arithmetic — 0 extra dependencies.
pub fn epoch_to_year_week(timestamp_ms: i64) -> (i32, u32) {
    let secs = timestamp_ms / 1000;
    // Days since Unix epoch (1970-01-01 was Thursday)
    let days = (secs.div_euclid(86400)) as i32;
    // Day of week: 1 = Monday ... 7 = Sunday
    let dow = (days + 3).rem_euclid(7) + 1;
    // Thursday in the current week determines the ISO year
    let thu_days = days + 4 - dow;

    // Estimate Gregorian year of the Thursday
    let mut year = 1970 + (thu_days as f64 / 365.2425).floor() as i32;

    fn days_before_year(y: i32) -> i32 {
        let y_prev = y - 1;
        y_prev * 365 + y_prev / 4 - y_prev / 100 + y_prev / 400 - 719162
    }

    while days_before_year(year + 1) <= thu_days {
        year += 1;
    }
    while days_before_year(year) > thu_days {
        year -= 1;
    }

    let doy = (thu_days - days_before_year(year)) as u32;
    let week = 1 + doy / 7;
    (year, week)
}

pub fn weekly_segment_name(year: i32, week: u32) -> String {
    format!("storage_{year}_w{week:02}.qtsq")
}

pub fn weekly_table_segment_name(year: i32, week: u32, table_name: &str) -> String {
    format!("storage_{year}_w{week:02}_{table_name}.qtsq")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSegmentMeta {
    pub name: String,
    pub year: i32,
    pub week: u32,
    pub start_time: i64,
    pub end_time: i64,
    pub tables: Vec<String>,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageManifest {
    pub schema: String,
    pub active_segment: String,
    pub segments: Vec<StorageSegmentMeta>,
    pub updated_at: i64,
}

impl Default for StorageManifest {
    fn default() -> Self {
        Self {
            schema: "sofuu-storage-manifest@1".into(),
            active_segment: String::new(),
            segments: Vec::new(),
            updated_at: 0,
        }
    }
}

pub struct StorageManager {
    pub base_dir: PathBuf,
    pub password: Option<String>,
    pub active_year: i32,
    pub active_week: u32,
    pub manifest: StorageManifest,
    pub tables: HashMap<String, SofuTable>,
    pub schemas: HashMap<String, SofuSchema>,
    pub dirty: bool,
    /// The reason the last flush failed, kept so a close that could not
    /// persist the pending rows can report the failure instead of claiming
    /// the close succeeded.
    pub last_flush_error: Option<String>,
}

impl StorageManager {
    pub fn open(base_path: &Path, password: Option<&str>) -> Result<Self, String> {
        let base_dir = if base_path.extension().is_some() {
            base_path.parent().unwrap_or(Path::new(".")).to_path_buf()
        } else {
            base_path.to_path_buf()
        };

        fs::create_dir_all(&base_dir)
            .map_err(|e| format!("E_STORAGE: cannot create the store directory {}: {e}", base_dir.display()))?;

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let (year, week) = epoch_to_year_week(now_ms);
        let active_seg = weekly_segment_name(year, week);

        // Hardware-keyed vault password default if none passed
        let vault_pass = match password {
            Some(p) if !p.is_empty() => Some(p.to_string()),
            _ => {
                #[cfg(has_qtsq)]
                {
                    Some(crate::rt::brain_key::local_vault_password())
                }
                #[cfg(not(has_qtsq))]
                {
                    None
                }
            }
        };

        /* The manifest is the store's index. A manifest that EXISTS but
         * cannot be read or parsed is a damaged store, not a fresh one:
         * replacing it with a default would silently drop the recorded
         * segment list and make the store look empty. */
        let manifest_path = base_dir.join("manifest.json");
        let mut manifest = match fs::read_to_string(&manifest_path) {
            Ok(data) => serde_json::from_str::<StorageManifest>(&data).map_err(|e| {
                format!(
                    "E_STORAGE: corrupt store manifest {}: {e}",
                    manifest_path.display()
                )
            })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => StorageManifest::default(),
            Err(e) => {
                return Err(format!(
                    "E_STORAGE: cannot read the store manifest {}: {e}",
                    manifest_path.display()
                ))
            }
        };

        manifest.active_segment = active_seg.clone();
        manifest.updated_at = now_ms;

        let mut mgr = Self {
            base_dir,
            password: vault_pass,
            active_year: year,
            active_week: week,
            manifest,
            tables: HashMap::new(),
            schemas: HashMap::new(),
            dirty: false,
            last_flush_error: None,
        };

        // Initialize standard schemas
        mgr.init_default_schemas()?;

        // Scan base_dir for all table segments of the active week:
        // storage_{year}_w{week:02}_<table_name>.qtsq
        let prefix = format!("storage_{year}_w{week:02}_");
        let read_dir = fs::read_dir(&mgr.base_dir).map_err(|e| {
            format!(
                "E_STORAGE: cannot list the store directory {}: {e}",
                mgr.base_dir.display()
            )
        })?;
        let mut segments: Vec<(String, PathBuf)> = Vec::new();
        for entry in read_dir {
            let entry = entry.map_err(|e| {
                format!(
                    "E_STORAGE: cannot list the store directory {}: {e}",
                    mgr.base_dir.display()
                )
            })?;
            let fname = entry.file_name().to_string_lossy().into_owned();
            if fname.starts_with(&prefix) && fname.ends_with(".qtsq") {
                let table_name = fname[prefix.len()..fname.len() - 5].to_string();
                segments.push((table_name, entry.path()));
            }
        }
        /* A segment that cannot be opened is a damaged store, not an absent
         * table: skipping it would let the store answer "no rows" for a
         * table whose rows are on disk and unreadable. */
        for (table_name, seg_file) in segments {
            let tbl = SofuTable::read_qtsq(&seg_file, &table_name, mgr.password.as_deref())
                .map_err(|e| {
                    format!(
                        "E_STORAGE: cannot open the store segment {}: {e}",
                        seg_file.display()
                    )
                })?;
            mgr.schemas.insert(table_name.clone(), tbl.schema().clone());
            mgr.tables.insert(table_name, tbl);
        }

        // Initialize any default schemas that were not on disk
        for (name, schema) in &mgr.schemas {
            if !mgr.tables.contains_key(name) {
                let tbl = SofuTable::create(schema)
                    .map_err(|e| format!("E_STORAGE: cannot create the table '{name}': {e}"))?;
                mgr.tables.insert(name.clone(), tbl);
            }
        }

        Ok(mgr)
    }

    fn init_default_schemas(&mut self) -> Result<(), String> {
        // 1. Messages Schema
        let msg_schema = SofuSchema::new("messages")?
            .add_col("id", SofuuColType::Int64, SOFUU_COL_PRIMARY_KEY | SOFUU_COL_AUTO_INC)?
            .add_col("session_id", SofuuColType::String, SOFUU_COL_REQUIRED | SOFUU_COL_INDEXED)?
            .add_col("role", SofuuColType::String, SOFUU_COL_REQUIRED)?
            .add_col("content", SofuuColType::String, SOFUU_COL_REQUIRED)?
            .add_col("metadata", SofuuColType::Json, 0)?
            .add_col("timestamp", SofuuColType::Timestamp, SOFUU_COL_REQUIRED)?;
        self.schemas.insert("messages".into(), msg_schema);

        // 2. Agent Traces Schema
        let trace_schema = SofuSchema::new("traces")?
            .add_col("id", SofuuColType::Int64, SOFUU_COL_PRIMARY_KEY | SOFUU_COL_AUTO_INC)?
            .add_col("session_id", SofuuColType::String, SOFUU_COL_REQUIRED | SOFUU_COL_INDEXED)?
            .add_col("step_index", SofuuColType::Int32, 0)?
            .add_col("tool_name", SofuuColType::String, 0)?
            .add_col("input", SofuuColType::Json, 0)?
            .add_col("output", SofuuColType::String, 0)?
            .add_col("duration_ms", SofuuColType::Float64, 0)?
            .add_col("timestamp", SofuuColType::Timestamp, SOFUU_COL_REQUIRED)?;
        self.schemas.insert("traces".into(), trace_schema);

        // 3. Key-Value Store Schema
        let kv_schema = SofuSchema::new("_kv_store")?
            .add_col("id", SofuuColType::Int64, SOFUU_COL_PRIMARY_KEY | SOFUU_COL_AUTO_INC)?
            .add_col("key", SofuuColType::String, SOFUU_COL_REQUIRED | SOFUU_COL_INDEXED)?
            .add_col("val", SofuuColType::Json, SOFUU_COL_REQUIRED)?
            .add_col("timestamp", SofuuColType::Timestamp, SOFUU_COL_REQUIRED)?;
        self.schemas.insert("_kv_store".into(), kv_schema);

        Ok(())
    }

    pub fn define_table(&mut self, name: &str, col_defs: &[(String, SofuuColType, u8)]) -> Result<(), String> {
        let mut schema = SofuSchema::new(name)?;
        for (col_name, col_type, flags) in col_defs {
            schema = schema.add_col(col_name, *col_type, *flags)?;
        }
        let tbl = SofuTable::create(&schema)?;
        self.schemas.insert(name.to_string(), schema);
        self.tables.insert(name.to_string(), tbl);
        self.dirty = true;
        Ok(())
    }

    pub fn check_weekly_rotation(&mut self, now_ms: i64) -> Result<(), String> {
        let (cur_year, cur_week) = epoch_to_year_week(now_ms);
        if cur_year != self.active_year || cur_week != self.active_week {
            // Rotate weekly segment: seal existing segment
            self.flush()?;

            // Update segment metadata in manifest
            let old_seg_name = weekly_segment_name(self.active_year, self.active_week);
            if !self.manifest.segments.iter().any(|s| s.name == old_seg_name) {
                self.manifest.segments.push(StorageSegmentMeta {
                    name: old_seg_name,
                    year: self.active_year,
                    week: self.active_week,
                    start_time: (self.active_year as i64) * 1000,
                    end_time: now_ms,
                    tables: self.tables.keys().cloned().collect(),
                    updated_at: now_ms,
                });
            }

            self.active_year = cur_year;
            self.active_week = cur_week;
            let new_seg_name = weekly_segment_name(cur_year, cur_week);
            self.manifest.active_segment = new_seg_name.clone();
            self.manifest.updated_at = now_ms;

            // Re-create active working tables for new week
            self.tables.clear();
            for (name, schema) in &self.schemas {
                let tbl = SofuTable::create(schema)?;
                self.tables.insert(name.clone(), tbl);
            }
            self.dirty = true;
        }
        Ok(())
    }

    pub fn insert(&mut self, table_name: &str, obj: &serde_json::Value) -> Result<i64, String> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        self.check_weekly_rotation(now_ms)?;

        // Auto-define custom table if not yet registered
        if !self.schemas.contains_key(table_name) {
            let mut col_defs = Vec::new();
            col_defs.push(("id".to_string(), SofuuColType::Int64, SOFUU_COL_PRIMARY_KEY | SOFUU_COL_AUTO_INC));
            if let Some(map) = obj.as_object() {
                for (k, v) in map {
                    if k == "id" || k == "timestamp" { continue; }
                    let ct = match v {
                        serde_json::Value::Bool(_) => SofuuColType::Bool,
                        serde_json::Value::Number(n) if n.is_i64() => SofuuColType::Int64,
                        serde_json::Value::Number(_) => SofuuColType::Float64,
                        serde_json::Value::String(_) => SofuuColType::String,
                        _ => SofuuColType::Json,
                    };
                    col_defs.push((k.clone(), ct, 0));
                }
            }
            col_defs.push(("timestamp".to_string(), SofuuColType::Timestamp, 0));
            self.define_table(table_name, &col_defs)?;
        }

        let schema = self.schemas.get(table_name)
            .ok_or_else(|| format!("E_STORAGE: unknown table '{table_name}'"))?;
        let col_names = schema.col_names();

        let mut row_vals = Vec::with_capacity(col_names.len());

        for (col_idx, col_name) in col_names.iter().enumerate() {
            let col_type = schema.col_type(col_idx);
            let json_val = obj.get(col_name);
            let val = match json_val {
                None | Some(serde_json::Value::Null) => {
                    // Check if timestamp column needs auto-fill
                    if schema.timestamp_col_idx() == schema.find_col(col_name) {
                        SofuValue::Timestamp(now_ms)
                    } else {
                        SofuValue::Null
                    }
                }
                Some(v) if col_type == Some(SofuuColType::Json) => {
                    SofuValue::Json(v.to_string())
                }
                Some(serde_json::Value::Bool(b)) => SofuValue::Bool(*b),
                Some(serde_json::Value::Number(n)) => {
                    if let Some(i) = n.as_i64() {
                        SofuValue::Int64(i)
                    } else if let Some(f) = n.as_f64() {
                        SofuValue::Float64(f)
                    } else {
                        SofuValue::Null
                    }
                }
                Some(serde_json::Value::String(s)) => {
                    if schema.timestamp_col_idx() == schema.find_col(col_name) {
                        if let Ok(ts) = s.parse::<i64>() {
                            SofuValue::Timestamp(ts)
                        } else {
                            SofuValue::String(s.clone())
                        }
                    } else {
                        SofuValue::String(s.clone())
                    }
                }
                Some(v) => SofuValue::Json(v.to_string()),
            };
            row_vals.push(val);
        }

        let tbl = self.tables.get_mut(table_name)
            .ok_or_else(|| format!("E_STORAGE: table '{table_name}' is not initialized"))?;

        tbl.insert(&row_vals)
            .map_err(|e| format!("E_STORAGE: cannot insert into '{table_name}': {e}"))?;
        self.dirty = true;

        let row_id = tbl.row_count() as i64;
        Ok(row_id)
    }

    pub fn query(
        &self,
        table_name: &str,
        query: &SofuQuery,
    ) -> Result<Vec<serde_json::Value>, String> {
        let tbl = self.tables.get(table_name)
            .ok_or_else(|| format!("E_STORAGE: table '{table_name}' not found"))?;

        let schema = self.schemas.get(table_name)
            .ok_or_else(|| format!("E_STORAGE: schema for '{table_name}' not found"))?;
        let col_names = schema.col_names();

        let res_table = tbl
            .query(query)
            .map_err(|e| format!("E_STORAGE: cannot query '{table_name}': {e}"))?;
        let count = res_table.row_count();
        let mut results = Vec::with_capacity(count);

        for r in 0..count {
            let row = res_table
                .get_row(r)
                .map_err(|e| format!("E_STORAGE: cannot read a row of '{table_name}': {e}"))?;
            let mut map = serde_json::Map::new();

            for (ci, val) in row.into_iter().enumerate() {
                if ci < col_names.len() {
                    let key = &col_names[ci];
                    map.insert(key.clone(), row_value_to_json(val, key)?);
                }
            }
            results.push(serde_json::Value::Object(map));
        }

        // If results are fewer than query.limit (or no limit) and historical segments exist on disk:
        if query.limit == 0 || results.len() < query.limit as usize {
            let active_seg_file = weekly_table_segment_name(self.active_year, self.active_week, table_name);
            let read_dir = fs::read_dir(&self.base_dir).map_err(|e| {
                format!(
                    "E_STORAGE: cannot list the store directory {}: {e}",
                    self.base_dir.display()
                )
            })?;
            let suffix = format!("_{table_name}.qtsq");
            let mut historical_files = Vec::new();
            for entry in read_dir {
                let entry = entry.map_err(|e| {
                    format!(
                        "E_STORAGE: cannot list the store directory {}: {e}",
                        self.base_dir.display()
                    )
                })?;
                let fname = entry.file_name().to_string_lossy().into_owned();
                if fname.starts_with("storage_") && fname.ends_with(&suffix) && fname != active_seg_file {
                    historical_files.push(entry.path());
                }
            }
            historical_files.sort();
            if query.order_desc {
                historical_files.reverse();
            }

            for hfile in historical_files {
                if query.limit > 0 && results.len() >= query.limit as usize {
                    break;
                }
                let mut rem_query = query.clone();
                if query.limit > 0 {
                    rem_query.limit = (query.limit as usize - results.len()) as u32;
                }
                /* A historical segment that cannot be read is a store
                 * failure. Skipping it would answer the query with the rows
                 * that happen to be readable and present that as the whole
                 * result. */
                let h_tbl = SofuTable::query_qtsq(&hfile, table_name, self.password.as_deref(), &rem_query)
                    .map_err(|e| {
                        format!(
                            "E_STORAGE: cannot read the store segment {}: {e}",
                            hfile.display()
                        )
                    })?;
                let h_count = h_tbl.row_count();
                for r in 0..h_count {
                    let row = h_tbl.get_row(r).map_err(|e| {
                        format!(
                            "E_STORAGE: cannot read a row of the store segment {}: {e}",
                            hfile.display()
                        )
                    })?;
                    let mut map = serde_json::Map::new();
                    for (ci, val) in row.into_iter().enumerate() {
                        if ci < col_names.len() {
                            let key = &col_names[ci];
                            map.insert(key.clone(), row_value_to_json(val, key)?);
                        }
                    }
                    results.push(serde_json::Value::Object(map));
                    if query.limit > 0 && results.len() >= query.limit as usize {
                        break;
                    }
                }
            }
        }

        Ok(results)
    }

    /// The number of rows the ACTIVE segment holds for `table_name`, or
    /// `None` when the table is not registered. A registered table with no
    /// rows is `Some(0)`; an unknown table is never reported as zero.
    pub fn count(&self, table_name: &str) -> Option<usize> {
        self.tables.get(table_name).map(|t| t.row_count())
    }

    pub fn flush(&mut self) -> Result<(), String> {
        if !self.dirty {
            return Ok(());
        }
        match self.write_pending() {
            Ok(()) => {
                self.dirty = false;
                self.last_flush_error = None;
                Ok(())
            }
            Err(e) => {
                self.last_flush_error = Some(e.clone());
                Err(e)
            }
        }
    }

    /// Write every pending table to its segment and commit the manifest.
    /// The rows stay pending when this fails, so a later attempt can still
    /// commit them.
    fn write_pending(&self) -> Result<(), String> {
        /* Each segment is staged beside its target and published with the
         * runtime's durable atomic publish (fsync the staged bytes, rename
         * over the target, fsync the directory). A reader therefore sees
         * either the previous segment or the new one — never a half-written
         * container. */
        for (name, tbl) in &self.tables {
            if tbl.row_count() > 0 {
                let seg_name = weekly_table_segment_name(self.active_year, self.active_week, name);
                let seg_file = self.base_dir.join(&seg_name);
                let seg_tmp = self.base_dir.join(format!(".{seg_name}.tmp"));
                tbl.write_qtsq(&seg_tmp, self.password.as_deref())
                    .map_err(|e| format!("E_STORAGE: failed to flush table '{name}': {e}"))?;
                if !crate::rt::wal::publish_paths(&seg_tmp, &seg_file) {
                    let _ = fs::remove_file(&seg_tmp);
                    return Err(format!(
                        "E_STORAGE: cannot commit the store segment {}",
                        seg_file.display()
                    ));
                }
            }
        }

        // Atomically save manifest
        let manifest_path = self.base_dir.join("manifest.json");
        let manifest_tmp = self.base_dir.join(".manifest.tmp");

        let manifest_bytes = serde_json::to_vec_pretty(&self.manifest)
            .map_err(|e| format!("E_STORAGE: cannot serialize the store manifest: {e}"))?;
        fs::write(&manifest_tmp, &manifest_bytes)
            .map_err(|e| format!("E_STORAGE: cannot write the store manifest {}: {e}", manifest_tmp.display()))?;
        if !crate::rt::wal::publish_paths(&manifest_tmp, &manifest_path) {
            let _ = fs::remove_file(&manifest_tmp);
            return Err(format!(
                "E_STORAGE: cannot commit the store manifest {}",
                manifest_path.display()
            ));
        }
        Ok(())
    }
}

/// One stored column value as JSON. A column whose bytes are not the JSON
/// they claim to be is a corrupt store, not a value that may be reshaped
/// into something else.
fn row_value_to_json(val: SofuValue, key: &str) -> Result<serde_json::Value, String> {
    Ok(match val {
        SofuValue::Null => serde_json::Value::Null,
        SofuValue::Int64(i) => serde_json::json!(i),
        SofuValue::Int32(i) => serde_json::json!(i),
        SofuValue::Float64(f) => serde_json::json!(f),
        SofuValue::Float32(f) => serde_json::json!(f),
        SofuValue::Bool(b) => serde_json::json!(b),
        SofuValue::Timestamp(t) => serde_json::json!(t),
        SofuValue::String(s) => serde_json::json!(s),
        SofuValue::Bytes(b) => serde_json::json!(b),
        SofuValue::Json(j) => serde_json::from_str(&j)
            .map_err(|e| format!("E_STORAGE: corrupt JSON column '{key}': {e}"))?,
    })
}

impl Drop for StorageManager {
    fn drop(&mut self) {
        /* A final flush that fails must not vanish: the rows would be lost
         * and the caller has already gone. */
        if let Err(e) = self.flush() {
            eprintln!("[sofuu/storage] the final flush failed: {e}");
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Registry & QuickJS Native Bridge Functions
// ══════════════════════════════════════════════════════════════════════

static STORAGE_REGISTRY: OnceLock<Mutex<HashMap<u64, StorageManager>>> = OnceLock::new();
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn get_registry() -> &'static Mutex<HashMap<u64, StorageManager>> {
    STORAGE_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

unsafe extern "C" fn js_storage_open(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"storage.open requires base_path".as_ptr());
    }

    let path_str = match js_to_string(ctx, *argv) {
        Some(s) => s,
        None => return qjs::JS_ThrowTypeError(ctx, c"Invalid path string".as_ptr()),
    };

    let password = if argc > 1 {
        js_to_string(ctx, *argv.add(1))
    } else {
        None
    };

    let path = Path::new(&path_str);
    match StorageManager::open(path, password.as_deref()) {
        Ok(mgr) => {
            let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
            let mut reg = match get_registry().lock() {
                Ok(r) => r,
                Err(_) => return qjs::JS_ThrowTypeError(ctx, c"storage registry is unavailable".as_ptr()),
            };
            reg.insert(handle, mgr);
            qjs::sofuu_js_new_int64(ctx, handle as i64)
        }
        Err(e) => {
            let err_msg = CString::new(format!("storage open error: {e}")).unwrap();
            qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), err_msg.as_ptr())
        }
    }
}

unsafe extern "C" fn js_storage_insert(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 3 {
        return qjs::JS_ThrowTypeError(ctx, c"storage.insert requires (handle, table, row)".as_ptr());
    }

    let mut handle: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut handle, *argv);

    let table_name = match js_to_string(ctx, *argv.add(1)) {
        Some(s) => s,
        None => return qjs::JS_ThrowTypeError(ctx, c"Invalid table name".as_ptr()),
    };

    let row_json = match js_to_string(ctx, *argv.add(2)) {
        Some(s) => s,
        None => return qjs::JS_ThrowTypeError(ctx, c"Invalid row json".as_ptr()),
    };

    let obj: serde_json::Value = match serde_json::from_str(&row_json) {
        Ok(v) => v,
        Err(e) => {
            let err_c = CString::new(format!("Invalid JSON row: {e}")).unwrap();
            return qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), err_c.as_ptr());
        }
    };

    let mut reg = match get_registry().lock() {
        Ok(r) => r,
        Err(_) => return qjs::JS_ThrowTypeError(ctx, c"Lock poison".as_ptr()),
    };

    let mgr = match reg.get_mut(&(handle as u64)) {
        Some(m) => m,
        None => return qjs::JS_ThrowTypeError(ctx, c"Invalid storage handle".as_ptr()),
    };

    match mgr.insert(&table_name, &obj) {
        Ok(id) => qjs::sofuu_js_new_int64(ctx, id),
        Err(e) => {
            let err_c = CString::new(format!("Insert failed: {e}")).unwrap();
            qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), err_c.as_ptr())
        }
    }
}

unsafe extern "C" fn js_storage_query(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 3 {
        return qjs::JS_ThrowTypeError(ctx, c"storage.query requires (handle, table, queryJson)".as_ptr());
    }

    let mut handle: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut handle, *argv);

    let table_name = match js_to_string(ctx, *argv.add(1)) {
        Some(s) => s,
        None => return qjs::JS_ThrowTypeError(ctx, c"Invalid table name".as_ptr()),
    };

    let query_json = match js_to_string(ctx, *argv.add(2)) {
        Some(s) => s,
        None => return qjs::JS_ThrowTypeError(ctx, c"Invalid query json".as_ptr()),
    };

    let q = match parse_query(&query_json) {
        Ok(q) => q,
        Err(e) => {
            let err_c = CString::new(e).unwrap();
            return qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), err_c.as_ptr());
        }
    };

    let reg = match get_registry().lock() {
        Ok(r) => r,
        Err(_) => return qjs::JS_ThrowTypeError(ctx, c"Lock poison".as_ptr()),
    };

    let mgr = match reg.get(&(handle as u64)) {
        Some(m) => m,
        None => return qjs::JS_ThrowTypeError(ctx, c"Invalid storage handle".as_ptr()),
    };

    match mgr.query(&table_name, &q) {
        Ok(rows) => {
            /* A row set that cannot be serialized is a FAILED read, not an
             * empty one: answering `[]` would present the serialization
             * failure as "the table holds no matching rows". */
            let json_str = match serde_json::to_string(&rows) {
                Ok(s) => s,
                Err(e) => {
                    let err_c = CString::new(format!("E_STORAGE: cannot serialize the query result: {e}"))
                        .unwrap();
                    return qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), err_c.as_ptr());
                }
            };
            let c_json = CString::new(json_str).unwrap();
            qjs::JS_ParseJSON(ctx, c_json.as_ptr(), c_json.as_bytes().len(), c"<query>".as_ptr())
        }
        Err(e) => {
            let err_c = CString::new(format!("Query failed: {e}")).unwrap();
            qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), err_c.as_ptr())
        }
    }
}

/// The query the JS surface sends, decoded.
///
/// A query that is not a JSON OBJECT is refused. It used to decode to a null
/// query, which carries no filter at all: a typo in the query string
/// silently widened the read to the whole table, and the caller received
/// every row as though it had asked for them.
fn parse_query(query_json: &str) -> Result<SofuQuery, String> {
    let q_obj: serde_json::Value = serde_json::from_str(query_json)
        .map_err(|e| format!("E_STORAGE: the query is not valid JSON: {e}"))?;
    if !q_obj.is_object() {
        return Err("E_STORAGE: the query must be a JSON object".to_string());
    }

    let mut q = SofuQuery::new();
    if let Some(min_id) = q_obj.get("minId").or_else(|| q_obj.get("min_id")).and_then(|v| v.as_i64()) {
        q = q.min_id(min_id);
    }
    if let Some(max_id) = q_obj.get("maxId").or_else(|| q_obj.get("max_id")).and_then(|v| v.as_i64()) {
        q = q.max_id(max_id);
    }
    if let Some(min_t) = q_obj.get("minTimestamp")
        .or_else(|| q_obj.get("min_timestamp"))
        .or_else(|| q_obj.get("time_start"))
        .or_else(|| q_obj.get("minTime"))
        .and_then(|v| v.as_i64()) {
        q = q.min_timestamp(min_t);
    }
    if let Some(max_t) = q_obj.get("maxTimestamp")
        .or_else(|| q_obj.get("max_timestamp"))
        .or_else(|| q_obj.get("time_end"))
        .or_else(|| q_obj.get("maxTime"))
        .and_then(|v| v.as_i64()) {
        q = q.max_timestamp(max_t);
    }
    if let Some(lim) = q_obj.get("limit").and_then(|v| v.as_u64()) {
        q = q.limit(lim as u32);
    }
    if let Some(off) = q_obj.get("offset").and_then(|v| v.as_u64()) {
        q = q.offset(off as u32);
    }
    if let Some(desc) = q_obj.get("orderDesc")
        .or_else(|| q_obj.get("order_desc"))
        .and_then(|v| v.as_bool())
        .or_else(|| q_obj.get("order").and_then(|v| v.as_str()).map(|s| s.eq_ignore_ascii_case("desc"))) {
        q = q.order_desc(desc);
    }
    if let (Some(col), Some(val)) = (
        q_obj.get("filterCol").or_else(|| q_obj.get("filter_col")).and_then(|v| v.as_str()),
        q_obj.get("filterVal").or_else(|| q_obj.get("filter_val")).and_then(|v| v.as_str()),
    ) {
        q = q.filter_str(col, val);
    }
    Ok(q)
}

unsafe extern "C" fn js_storage_count(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::JS_ThrowTypeError(ctx, c"storage.count requires (handle, table)".as_ptr());
    }

    let mut handle: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut handle, *argv);

    let table_name = match js_to_string(ctx, *argv.add(1)) {
        Some(s) => s,
        None => return qjs::JS_ThrowTypeError(ctx, c"Invalid table name".as_ptr()),
    };

    let reg = match get_registry().lock() {
        Ok(r) => r,
        Err(_) => return qjs::JS_ThrowTypeError(ctx, c"Lock poison".as_ptr()),
    };

    /* An unknown handle or an unknown table is a FAILURE. Reporting zero
     * rows for a store that cannot be reached would present a storage
     * failure as an empty table. */
    let mgr = match reg.get(&(handle as u64)) {
        Some(m) => m,
        None => return qjs::JS_ThrowTypeError(ctx, c"Invalid storage handle".as_ptr()),
    };
    match mgr.count(&table_name) {
        Some(count) => qjs::sofuu_js_new_int64(ctx, count as i64),
        None => qjs::JS_ThrowTypeError(ctx, c"Unknown table".as_ptr()),
    }
}

unsafe extern "C" fn js_storage_flush(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return js_new_bool(ctx, false);
    }

    let mut handle: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut handle, *argv);

    let mut reg = match get_registry().lock() {
        Ok(r) => r,
        Err(_) => return js_new_bool(ctx, false),
    };

    if let Some(mgr) = reg.get_mut(&(handle as u64)) {
        match mgr.flush() {
            Ok(()) => js_new_bool(ctx, true),
            Err(e) => {
                /* The caller's only signal is the boolean, so the reason is
                 * written to the log as well: a flush that failed must not
                 * vanish. */
                eprintln!("[sofuu/storage] flush failed: {e}");
                js_new_bool(ctx, false)
            }
        }
    } else {
        js_new_bool(ctx, false)
    }
}

unsafe extern "C" fn js_storage_close(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return js_new_bool(ctx, false);
    }

    let mut handle: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut handle, *argv);

    if let Ok(mut reg) = get_registry().lock() {
        if let Some(mut mgr) = reg.remove(&(handle as u64)) {
            /* A close that could not persist the pending rows is NOT a
             * successful close: those rows would be lost, so the close
             * reports the failure instead of claiming success. The boolean
             * is the caller's only signal, so the reason goes to the log
             * as well — a close that failed must not vanish. */
            return match mgr.flush() {
                Ok(()) => js_new_bool(ctx, true),
                Err(e) => {
                    eprintln!("[sofuu/storage] the close could not persist the pending rows: {e}");
                    js_new_bool(ctx, false)
                }
            };
        }
    }
    js_new_bool(ctx, false)
}

unsafe extern "C" fn js_storage_active_segment(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_undefined();
    }

    let mut handle: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut handle, *argv);

    let reg = match get_registry().lock() {
        Ok(r) => r,
        Err(_) => return qjs::sofuu_js_undefined(),
    };

    if let Some(mgr) = reg.get(&(handle as u64)) {
        js_new_string(ctx, &mgr.manifest.active_segment)
    } else {
        qjs::sofuu_js_undefined()
    }
}

/// Register native host functions for `sofuu.storage`
pub unsafe fn mod_storage_register(ctx: *mut JSContext) {
    register_global_fn(ctx, "__sofuu_storage_open", js_storage_open);
    register_global_fn(ctx, "__sofuu_storage_insert", js_storage_insert);
    register_global_fn(ctx, "__sofuu_storage_query", js_storage_query);
    register_global_fn(ctx, "__sofuu_storage_count", js_storage_count);
    register_global_fn(ctx, "__sofuu_storage_flush", js_storage_flush);
    register_global_fn(ctx, "__sofuu_storage_close", js_storage_close);
    register_global_fn(ctx, "__sofuu_storage_active_segment", js_storage_active_segment);
}

// ══════════════════════════════════════════════════════════════════════
// Unit Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("sofuu-storage-{tag}-{}-{n}", std::process::id()));
        p
    }

    #[test]
    fn test_epoch_to_year_week() {
        // 2026-10-21 (Wednesday of Week 43)
        let ts_20261021 = 1792576000000i64;
        let (y, w) = epoch_to_year_week(ts_20261021);
        assert_eq!(y, 2026);
        assert_eq!(w, 43);

        // 1970-01-01 (Thursday of Week 1)
        let ts_1970 = 0i64;
        let (y0, w0) = epoch_to_year_week(ts_1970);
        assert_eq!(y0, 1970);
        assert_eq!(w0, 1);
    }

    #[test]
    fn test_storage_manager_lifecycle_and_tables() {
        let temp_dir = std::env::temp_dir().join(format!("sofuu_storage_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);

        let mut mgr = StorageManager::open(&temp_dir, Some("test-vault-password")).unwrap();

        // 1. Insert into default messages table
        let msg = serde_json::json!({
            "session_id": "session_alpha",
            "role": "user",
            "content": "Hello QTSQ Structured Storage!",
            "metadata": { "client": "terminal" }
        });

        let id = mgr.insert("messages", &msg).unwrap();
        assert_eq!(id, 1);
        assert_eq!(mgr.count("messages"), Some(1));

        // 2. Query messages
        let q = SofuQuery::new().filter_str("session_id", "session_alpha");
        let results = mgr.query("messages", &q).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["content"], "Hello QTSQ Structured Storage!");

        // 3. Insert and retrieve from KV table
        let kv_doc = serde_json::json!({
            "key": "user_theme",
            "val": { "mode": "dark", "accent": "violet" }
        });
        mgr.insert("_kv_store", &kv_doc).unwrap();
        assert_eq!(mgr.count("_kv_store"), Some(1));

        let kv_query = SofuQuery::new().filter_str("key", "user_theme");
        let kv_res = mgr.query("_kv_store", &kv_query).unwrap();
        assert_eq!(kv_res.len(), 1);
        assert_eq!(kv_res[0]["val"]["mode"], "dark");

        // 4. Flush and reload from disk
        mgr.flush().unwrap();
        drop(mgr);

        // Re-open and verify persistence
        let mgr2 = StorageManager::open(&temp_dir, Some("test-vault-password")).unwrap();
        assert_eq!(mgr2.count("messages"), Some(1));
        let q2 = SofuQuery::new().filter_str("session_id", "session_alpha");
        let results2 = mgr2.query("messages", &q2).unwrap();
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0]["content"], "Hello QTSQ Structured Storage!");

        let _ = fs::remove_dir_all(temp_dir);
    }

    /// An unknown table is not an empty table: `count` must not answer zero
    /// for a table the store does not have.
    #[test]
    fn an_unknown_table_is_not_reported_as_zero_rows() {
        let dir = tmp_dir("unknown-table");
        let mgr = StorageManager::open(&dir, Some("test-vault-password")).unwrap();
        assert_eq!(mgr.count("messages"), Some(0));
        assert_eq!(mgr.count("no_such_table"), None);
        assert!(mgr.query("no_such_table", &SofuQuery::new()).is_err());
        let _ = fs::remove_dir_all(dir);
    }

    /// A manifest that exists but cannot be parsed is a damaged store. It
    /// must not be replaced with a default manifest, which would present the
    /// store as a fresh, empty one.
    #[test]
    fn a_corrupt_manifest_fails_the_open_instead_of_starting_an_empty_store() {
        let dir = tmp_dir("corrupt-manifest");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("manifest.json"), b"{not json").unwrap();
        let error = match StorageManager::open(&dir, Some("test-vault-password")) {
            Ok(_) => panic!("a corrupt manifest must not open as a fresh store"),
            Err(e) => e,
        };
        assert!(error.starts_with("E_STORAGE:"), "{error}");
        assert!(error.contains("corrupt store manifest"), "{error}");
        let _ = fs::remove_dir_all(dir);
    }

    /// A segment that exists but cannot be opened is a damaged store, not an
    /// absent table: skipping it would make the store answer "no rows" for
    /// rows that are on disk.
    #[test]
    fn an_unreadable_segment_fails_the_open_instead_of_hiding_its_table() {
        let dir = tmp_dir("unreadable-segment");
        fs::create_dir_all(&dir).unwrap();
        let (year, week) = epoch_to_year_week(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0),
        );
        let seg = dir.join(weekly_table_segment_name(year, week, "messages"));
        fs::write(&seg, b"not a qtsq container").unwrap();
        let error = match StorageManager::open(&dir, Some("test-vault-password")) {
            Ok(_) => panic!("an unreadable segment must not open as an empty store"),
            Err(e) => e,
        };
        assert!(error.starts_with("E_STORAGE:"), "{error}");
        assert!(error.contains("cannot open the store segment"), "{error}");
        let _ = fs::remove_dir_all(dir);
    }

    /// A flush that cannot reach the disk is an error, and the manager stays
    /// dirty so a later attempt can still commit it.
    #[test]
    fn a_flush_that_cannot_reach_the_disk_is_reported() {
        let dir = tmp_dir("unflushable");
        fs::create_dir_all(&dir).unwrap();
        let mut mgr = StorageManager::open(&dir, Some("test-vault-password")).unwrap();
        mgr.insert(
            "messages",
            &serde_json::json!({
                "session_id": "s",
                "role": "user",
                "content": "c"
            }),
        )
        .unwrap();
        assert!(mgr.dirty);
        // Replace the store directory with a file so the segment commit
        // cannot create its staged file.
        fs::remove_dir_all(&dir).unwrap();
        fs::write(&dir, b"not a directory").unwrap();
        let error = mgr.flush().unwrap_err();
        assert!(error.starts_with("E_STORAGE:"), "{error}");
        assert!(mgr.dirty, "a failed flush must leave the store pending");
        let _ = fs::remove_file(&dir);
    }

    /// The durable claim, checked across a process boundary: one process
    /// writes and exits, the next opens the same store and reads the rows
    /// that were actually written.
    #[test]
    fn a_flushed_row_survives_the_process_that_wrote_it() {
        // The child process: write one row, flush, exit.
        if let Ok(child_dir) = std::env::var("SOFUU_SLICE9_STORAGE_CHILD") {
            let mut mgr = StorageManager::open(Path::new(&child_dir), Some("test-vault-password")).unwrap();
            mgr.insert(
                "messages",
                &serde_json::json!({
                    "session_id": "cross-process",
                    "role": "user",
                    "content": "cross-process-content"
                }),
            )
            .unwrap();
            mgr.flush().unwrap();
            return;
        }

        let dir = tmp_dir("cross-process");
        let exe = std::env::current_exe().expect("the test binary path must be known");
        let status = std::process::Command::new(exe)
            .args([
                "--exact",
                "rt::storage::tests::a_flushed_row_survives_the_process_that_wrote_it",
                "--nocapture",
            ])
            .env("SOFUU_SLICE9_STORAGE_CHILD", &dir)
            .status()
            .expect("the child process must start");
        assert!(status.success(), "the writing process must exit cleanly");

        let mgr = StorageManager::open(&dir, Some("test-vault-password")).unwrap();
        let q = SofuQuery::new().filter_str("session_id", "cross-process");
        let rows = mgr.query("messages", &q).unwrap();
        assert_eq!(rows.len(), 1, "the reopened store must return the written row");
        assert_eq!(rows[0]["content"], "cross-process-content");
        let _ = fs::remove_dir_all(dir);
    }

    /// A malformed query is REFUSED. It used to decode to a null query,
    /// which carries no filter at all: a typo in the query string widened
    /// the read to the whole table, and the caller received every row as
    /// though it had asked for them.
    #[test]
    fn a_malformed_query_is_refused_instead_of_widening_the_read() {
        for bad in ["{not json", "[]", "null", "\"text\"", "42", "true"] {
            match parse_query(bad) {
                Ok(_) => panic!("a malformed query must be refused: {bad}"),
                Err(e) => assert!(e.starts_with("E_STORAGE:"), "{bad}: {e}"),
            }
        }
        assert!(
            parse_query("{}").is_ok(),
            "an empty query object is a valid query"
        );
        assert!(
            parse_query(r#"{"filterCol":"session_id","filterVal":"s","limit":5}"#).is_ok(),
            "a well-formed query must still decode"
        );
    }
}
