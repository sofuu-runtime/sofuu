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
}

impl StorageManager {
    pub fn open(base_path: &Path, password: Option<&str>) -> Result<Self, String> {
        let base_dir = if base_path.extension().is_some() {
            base_path.parent().unwrap_or(Path::new(".")).to_path_buf()
        } else {
            base_path.to_path_buf()
        };

        fs::create_dir_all(&base_dir).map_err(|e| e.to_string())?;

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

        let manifest_path = base_dir.join("manifest.json");
        let mut manifest = if manifest_path.exists() {
            fs::read_to_string(&manifest_path)
                .ok()
                .and_then(|data| serde_json::from_str::<StorageManifest>(&data).ok())
                .unwrap_or_default()
        } else {
            StorageManifest::default()
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
        };

        // Initialize standard schemas
        mgr.init_default_schemas()?;

        // Scan base_dir for all table segments of the active week:
        // storage_{year}_w{week:02}_<table_name>.qtsq
        let prefix = format!("storage_{year}_w{week:02}_");
        if let Ok(entries) = fs::read_dir(&mgr.base_dir) {
            for entry in entries.flatten() {
                let fname = entry.file_name().to_string_lossy().into_owned();
                if fname.starts_with(&prefix) && fname.ends_with(".qtsq") {
                    let table_name = fname[prefix.len()..fname.len() - 5].to_string();
                    let seg_file = entry.path();
                    if let Ok(tbl) = SofuTable::read_qtsq(&seg_file, &table_name, mgr.password.as_deref()) {
                        mgr.schemas.insert(table_name.clone(), tbl.schema().clone());
                        mgr.tables.insert(table_name, tbl);
                    }
                }
            }
        }

        // Initialize any default schemas that were not on disk
        for (name, schema) in &mgr.schemas {
            if !mgr.tables.contains_key(name) {
                if let Ok(tbl) = SofuTable::create(schema) {
                    mgr.tables.insert(name.clone(), tbl);
                }
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
            .ok_or_else(|| format!("Unknown table '{table_name}'"))?;
        let col_names = schema.col_names();

        let mut row_vals = Vec::with_capacity(col_names.len());

        for col_name in &col_names {
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
            .ok_or_else(|| format!("Table '{table_name}' not initialized"))?;

        tbl.insert(&row_vals)?;
        self.dirty = true;

        let row_id = tbl.row_count() as i64;

        // Auto-flush when bounded block reaches capacity
        if row_id > 0 && row_id % 128 == 0 {
            let _ = self.flush();
        }

        Ok(row_id)
    }

    pub fn query(
        &self,
        table_name: &str,
        query: &SofuQuery,
    ) -> Result<Vec<serde_json::Value>, String> {
        let tbl = self.tables.get(table_name)
            .ok_or_else(|| format!("Table '{table_name}' not found"))?;

        let schema = self.schemas.get(table_name)
            .ok_or_else(|| format!("Schema for '{table_name}' not found"))?;
        let col_names = schema.col_names();

        let res_table = tbl.query(query)?;
        let count = res_table.row_count();
        let mut results = Vec::with_capacity(count);

        for r in 0..count {
            let row = res_table.get_row(r)?;
            let mut map = serde_json::Map::new();

            for (ci, val) in row.into_iter().enumerate() {
                if ci < col_names.len() {
                    let key = &col_names[ci];
                    let jv = match val {
                        SofuValue::Null => serde_json::Value::Null,
                        SofuValue::Int64(i) => serde_json::json!(i),
                        SofuValue::Int32(i) => serde_json::json!(i),
                        SofuValue::Float64(f) => serde_json::json!(f),
                        SofuValue::Float32(f) => serde_json::json!(f),
                        SofuValue::Bool(b) => serde_json::json!(b),
                        SofuValue::Timestamp(t) => serde_json::json!(t),
                        SofuValue::String(s) => serde_json::json!(s),
                        SofuValue::Bytes(b) => serde_json::json!(b),
                        SofuValue::Json(j) => {
                            serde_json::from_str(&j).unwrap_or(serde_json::Value::String(j))
                        }
                    };
                    map.insert(key.clone(), jv);
                }
            }
            results.push(serde_json::Value::Object(map));
        }

        // If results are fewer than query.limit (or no limit) and historical segments exist on disk:
        if query.limit == 0 || results.len() < query.limit as usize {
            let active_seg_file = weekly_table_segment_name(self.active_year, self.active_week, table_name);
            if let Ok(entries) = fs::read_dir(&self.base_dir) {
                let suffix = format!("_{table_name}.qtsq");
                let mut historical_files = Vec::new();
                for entry in entries.flatten() {
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
                    if let Ok(h_tbl) = SofuTable::query_qtsq(&hfile, table_name, self.password.as_deref(), &rem_query) {
                        let h_count = h_tbl.row_count();
                        for r in 0..h_count {
                            if let Ok(row) = h_tbl.get_row(r) {
                                let mut map = serde_json::Map::new();
                                for (ci, val) in row.into_iter().enumerate() {
                                    if ci < col_names.len() {
                                        let key = &col_names[ci];
                                        let jv = match val {
                                            SofuValue::Null => serde_json::Value::Null,
                                            SofuValue::Int64(i) => serde_json::json!(i),
                                            SofuValue::Int32(i) => serde_json::json!(i),
                                            SofuValue::Float64(f) => serde_json::json!(f),
                                            SofuValue::Float32(f) => serde_json::json!(f),
                                            SofuValue::Bool(b) => serde_json::json!(b),
                                            SofuValue::Timestamp(t) => serde_json::json!(t),
                                            SofuValue::String(s) => serde_json::json!(s),
                                            SofuValue::Bytes(b) => serde_json::json!(b),
                                            SofuValue::Json(j) => {
                                                serde_json::from_str(&j).unwrap_or(serde_json::Value::String(j))
                                            }
                                        };
                                        map.insert(key.clone(), jv);
                                    }
                                }
                                results.push(serde_json::Value::Object(map));
                                if query.limit > 0 && results.len() >= query.limit as usize {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    pub fn count(&self, table_name: &str) -> usize {
        self.tables.get(table_name).map(|t| t.row_count()).unwrap_or(0)
    }

    pub fn flush(&mut self) -> Result<(), String> {
        if !self.dirty {
            return Ok(());
        }

        for (name, tbl) in &self.tables {
            if tbl.row_count() > 0 {
                let seg_name = weekly_table_segment_name(self.active_year, self.active_week, name);
                let seg_file = self.base_dir.join(&seg_name);
                tbl.write_qtsq(&seg_file, self.password.as_deref())
                    .map_err(|e| format!("Failed to flush table '{name}': {e}"))?;
            }
        }

        // Atomically save manifest
        let manifest_path = self.base_dir.join("manifest.json");
        let manifest_tmp = self.base_dir.join(".manifest.tmp");

        let manifest_bytes = serde_json::to_vec_pretty(&self.manifest).map_err(|e| e.to_string())?;
        fs::write(&manifest_tmp, &manifest_bytes).map_err(|e| e.to_string())?;
        fs::rename(&manifest_tmp, &manifest_path).map_err(|e| e.to_string())?;

        self.dirty = false;
        Ok(())
    }
}

impl Drop for StorageManager {
    fn drop(&mut self) {
        let _ = self.flush();
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
            if let Ok(mut reg) = get_registry().lock() {
                reg.insert(handle, mgr);
            }
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

    let q_obj: serde_json::Value = serde_json::from_str(&query_json).unwrap_or(serde_json::Value::Null);

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
            let json_str = serde_json::to_string(&rows).unwrap_or_else(|_| "[]".into());
            let c_json = CString::new(json_str).unwrap();
            qjs::JS_ParseJSON(ctx, c_json.as_ptr(), c_json.as_bytes().len(), c"<query>".as_ptr())
        }
        Err(e) => {
            let err_c = CString::new(format!("Query failed: {e}")).unwrap();
            qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), err_c.as_ptr())
        }
    }
}

unsafe extern "C" fn js_storage_count(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::sofuu_js_new_int32(ctx, 0);
    }

    let mut handle: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut handle, *argv);

    let table_name = match js_to_string(ctx, *argv.add(1)) {
        Some(s) => s,
        None => return qjs::sofuu_js_new_int32(ctx, 0),
    };

    let reg = match get_registry().lock() {
        Ok(r) => r,
        Err(_) => return qjs::sofuu_js_new_int32(ctx, 0),
    };

    let count = reg.get(&(handle as u64)).map(|m| m.count(&table_name)).unwrap_or(0);
    qjs::sofuu_js_new_int64(ctx, count as i64)
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
        let res = mgr.flush().is_ok();
        js_new_bool(ctx, res)
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
            let _ = mgr.flush();
            return js_new_bool(ctx, true);
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
        assert_eq!(mgr.count("messages"), 1);

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
        assert_eq!(mgr.count("_kv_store"), 1);

        let kv_query = SofuQuery::new().filter_str("key", "user_theme");
        let kv_res = mgr.query("_kv_store", &kv_query).unwrap();
        assert_eq!(kv_res.len(), 1);
        assert_eq!(kv_res[0]["val"]["mode"], "dark");

        // 4. Flush and reload from disk
        mgr.flush().unwrap();
        drop(mgr);

        // Re-open and verify persistence
        let mgr2 = StorageManager::open(&temp_dir, Some("test-vault-password")).unwrap();
        assert_eq!(mgr2.count("messages"), 1);
        let q2 = SofuQuery::new().filter_str("session_id", "session_alpha");
        let results2 = mgr2.query("messages", &q2).unwrap();
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0]["content"], "Hello QTSQ Structured Storage!");

        let _ = fs::remove_dir_all(temp_dir);
    }
}
