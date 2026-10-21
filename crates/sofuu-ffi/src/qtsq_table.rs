// sofuu-ffi — High-speed columnar structured table engine powered by QTSQ.
// Part of Phase 3: Native QTSQ Structured Storage Driver.
//
// Implements strict schema columnar tables, bounded 128-row block chunking,
// and checkpoint indexing with AES-256-GCM vault encryption or plaintext writes.
//
// Author: Haruhito

#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;

pub const SOFUU_TABLE_MAGIC: u32 = 0x53465554; // 'SFUT'
pub const SOFUU_TABLE_VERSION: u32 = 1;
pub const SOFUU_MAX_COLS: usize = 32;
pub const SOFUU_MAX_NAME_LEN: usize = 64;
pub const SOFUU_ROWS_PER_BLOCK: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SofuuColType {
    Null = 0,
    Int64 = 1,
    Int32 = 2,
    Float64 = 3,
    Float32 = 4,
    Bool = 5,
    Timestamp = 6,
    String = 7,
    Bytes = 8,
    Json = 9,
}

impl SofuuColType {
    pub fn from_u8(val: u8) -> Self {
        match val {
            1 => SofuuColType::Int64,
            2 => SofuuColType::Int32,
            3 => SofuuColType::Float64,
            4 => SofuuColType::Float32,
            5 => SofuuColType::Bool,
            6 => SofuuColType::Timestamp,
            7 => SofuuColType::String,
            8 => SofuuColType::Bytes,
            9 => SofuuColType::Json,
            _ => SofuuColType::Null,
        }
    }
}

pub const SOFUU_COL_PRIMARY_KEY: u8 = 1 << 0;
pub const SOFUU_COL_REQUIRED: u8 = 1 << 1;
pub const SOFUU_COL_INDEXED: u8 = 1 << 2;
pub const SOFUU_COL_AUTO_INC: u8 = 1 << 3;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SofuuColDef {
    pub name: [c_char; SOFUU_MAX_NAME_LEN],
    pub col_type: u8,
    pub flags: u8,
    pub reserved: u16,
}

impl Default for SofuuColDef {
    fn default() -> Self {
        Self {
            name: [0; SOFUU_MAX_NAME_LEN],
            col_type: 0,
            flags: 0,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SofuuSchemaC {
    pub table_name: [c_char; SOFUU_MAX_NAME_LEN],
    pub num_cols: u32,
    pub cols: [SofuuColDef; SOFUU_MAX_COLS],
    pub pk_col_idx: i32,
    pub timestamp_col_idx: i32,
}

impl Default for SofuuSchemaC {
    fn default() -> Self {
        Self {
            table_name: [0; SOFUU_MAX_NAME_LEN],
            num_cols: 0,
            cols: [SofuuColDef::default(); SOFUU_MAX_COLS],
            pk_col_idx: -1,
            timestamp_col_idx: -1,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SofuuStrRef {
    pub ptr: *const c_char,
    pub len: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SofuuBytesRef {
    pub ptr: *const u8,
    pub len: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union SofuuValAs {
    pub i64_val: i64,
    pub i32_val: i32,
    pub f64_val: f64,
    pub f32_val: f32,
    pub b_val: bool,
    pub str_val: SofuuStrRef,
    pub bytes_val: SofuuBytesRef,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SofuuValC {
    pub val_type: u8,
    pub as_val: SofuuValAs,
}

impl Default for SofuuValC {
    fn default() -> Self {
        Self {
            val_type: 0,
            as_val: SofuuValAs { i64_val: 0 },
        }
    }
}

#[repr(C)]
pub struct SofuuQueryC {
    pub min_id: i64,
    pub max_id: i64,
    pub min_timestamp: i64,
    pub max_timestamp: i64,
    pub filter_col_name: *const c_char,
    pub filter_val: SofuuValC,
    pub limit: u32,
    pub offset: u32,
    pub order_desc: bool,
}

impl Default for SofuuQueryC {
    fn default() -> Self {
        Self {
            min_id: 0,
            max_id: 0,
            min_timestamp: 0,
            max_timestamp: 0,
            filter_col_name: std::ptr::null(),
            filter_val: SofuuValC::default(),
            limit: 0,
            offset: 0,
            order_desc: false,
        }
    }
}

pub enum SofuuTableOpaque {}

extern "C" {
    pub fn sofuu_schema_init(schema: *mut SofuuSchemaC, table_name: *const c_char);
    pub fn sofuu_schema_add_col(
        schema: *mut SofuuSchemaC,
        name: *const c_char,
        col_type: u8,
        flags: u8,
    ) -> c_int;
    pub fn sofuu_schema_find_col(schema: *const SofuuSchemaC, name: *const c_char) -> c_int;

    pub fn sofuu_table_create(schema: *const SofuuSchemaC) -> *mut SofuuTableOpaque;
    pub fn sofuu_table_free(tbl: *mut SofuuTableOpaque);
    pub fn sofuu_table_insert(
        tbl: *mut SofuuTableOpaque,
        vals: *const SofuuValC,
        num_vals: u32,
    ) -> c_int;
    pub fn sofuu_table_row_count(tbl: *const SofuuTableOpaque) -> i64;
    pub fn sofuu_table_get_row(
        tbl: *const SofuuTableOpaque,
        row_idx: u32,
        out_vals: *mut SofuuValC,
        max_vals: u32,
    ) -> c_int;

    pub fn sofuu_query_init(q: *mut SofuuQueryC);
    pub fn sofuu_table_query(
        tbl: *const SofuuTableOpaque,
        q: *const SofuuQueryC,
    ) -> *mut SofuuTableOpaque;

    pub fn sofuu_table_write_qtsq(
        tbl: *const SofuuTableOpaque,
        filepath: *const c_char,
        password: *const c_char,
    ) -> c_int;
    pub fn sofuu_table_read_qtsq(
        filepath: *const c_char,
        table_name: *const c_char,
        password: *const c_char,
    ) -> *mut SofuuTableOpaque;
    pub fn sofuu_table_query_qtsq(
        filepath: *const c_char,
        table_name: *const c_char,
        password: *const c_char,
        q: *const SofuuQueryC,
    ) -> *mut SofuuTableOpaque;
}

// ══════════════════════════════════════════════════════════════════════
// High-Level Safe Rust Abstractions
// ══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
pub enum SofuValue {
    Null,
    Int64(i64),
    Int32(i32),
    Float64(f64),
    Float32(f32),
    Bool(bool),
    Timestamp(i64),
    String(String),
    Bytes(Vec<u8>),
    Json(String),
}

impl From<i64> for SofuValue {
    fn from(v: i64) -> Self {
        SofuValue::Int64(v)
    }
}

impl From<i32> for SofuValue {
    fn from(v: i32) -> Self {
        SofuValue::Int32(v)
    }
}

impl From<f64> for SofuValue {
    fn from(v: f64) -> Self {
        SofuValue::Float64(v)
    }
}

impl From<f32> for SofuValue {
    fn from(v: f32) -> Self {
        SofuValue::Float32(v)
    }
}

impl From<bool> for SofuValue {
    fn from(v: bool) -> Self {
        SofuValue::Bool(v)
    }
}

impl From<&str> for SofuValue {
    fn from(v: &str) -> Self {
        SofuValue::String(v.to_string())
    }
}

impl From<String> for SofuValue {
    fn from(v: String) -> Self {
        SofuValue::String(v)
    }
}

impl From<Vec<u8>> for SofuValue {
    fn from(v: Vec<u8>) -> Self {
        SofuValue::Bytes(v)
    }
}

#[derive(Clone)]
pub struct SofuSchema {
    raw: SofuuSchemaC,
}

impl SofuSchema {
    pub fn new(table_name: &str) -> Result<Self, String> {
        let c_name = CString::new(table_name).map_err(|e| e.to_string())?;
        let mut raw = SofuuSchemaC::default();
        unsafe {
            sofuu_schema_init(&mut raw, c_name.as_ptr());
        }
        Ok(Self { raw })
    }

    pub fn add_col(mut self, name: &str, col_type: SofuuColType, flags: u8) -> Result<Self, String> {
        let c_name = CString::new(name).map_err(|e| e.to_string())?;
        let rc = unsafe {
            sofuu_schema_add_col(&mut self.raw, c_name.as_ptr(), col_type as u8, flags)
        };
        if rc < 0 {
            return Err(format!("Failed to add column '{name}' (rc: {rc})"));
        }
        Ok(self)
    }

    pub fn find_col(&self, name: &str) -> Option<usize> {
        let c_name = CString::new(name).ok()?;
        let idx = unsafe { sofuu_schema_find_col(&self.raw, c_name.as_ptr()) };
        if idx >= 0 {
            Some(idx as usize)
        } else {
            None
        }
    }

    pub fn num_cols(&self) -> usize {
        self.raw.num_cols as usize
    }

    pub fn table_name(&self) -> String {
        unsafe {
            CStr::from_ptr(self.raw.table_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        }
    }

    pub fn col_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for i in 0..self.raw.num_cols as usize {
            let s = unsafe {
                CStr::from_ptr(self.raw.cols[i].name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            names.push(s);
        }
        names
    }

    pub fn pk_col_idx(&self) -> Option<usize> {
        if self.raw.pk_col_idx >= 0 {
            Some(self.raw.pk_col_idx as usize)
        } else {
            None
        }
    }

    pub fn timestamp_col_idx(&self) -> Option<usize> {
        if self.raw.timestamp_col_idx >= 0 {
            Some(self.raw.timestamp_col_idx as usize)
        } else {
            None
        }
    }

    pub fn raw(&self) -> &SofuuSchemaC {
        &self.raw
    }
}

#[derive(Default, Clone)]
pub struct SofuQuery {
    pub min_id: i64,
    pub max_id: i64,
    pub min_timestamp: i64,
    pub max_timestamp: i64,
    pub filter_col_name: Option<String>,
    pub filter_val: Option<SofuValue>,
    pub limit: u32,
    pub offset: u32,
    pub order_desc: bool,
}

impl SofuQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn min_id(mut self, val: i64) -> Self {
        self.min_id = val;
        self
    }

    pub fn max_id(mut self, val: i64) -> Self {
        self.max_id = val;
        self
    }

    pub fn min_timestamp(mut self, val: i64) -> Self {
        self.min_timestamp = val;
        self
    }

    pub fn max_timestamp(mut self, val: i64) -> Self {
        self.max_timestamp = val;
        self
    }

    pub fn filter_str(mut self, col_name: &str, val: &str) -> Self {
        self.filter_col_name = Some(col_name.to_string());
        self.filter_val = Some(SofuValue::String(val.to_string()));
        self
    }

    pub fn limit(mut self, limit: u32) -> Self {
        self.limit = limit;
        self
    }

    pub fn offset(mut self, offset: u32) -> Self {
        self.offset = offset;
        self
    }

    pub fn order_desc(mut self, desc: bool) -> Self {
        self.order_desc = desc;
        self
    }
}

pub struct SofuTable {
    raw: *mut SofuuTableOpaque,
    schema: SofuSchema,
}

// Send + Sync safety for SofuTable: it is self-contained heap memory
unsafe impl Send for SofuTable {}
unsafe impl Sync for SofuTable {}

impl SofuTable {
    pub fn create(schema: &SofuSchema) -> Result<Self, String> {
        let ptr = unsafe { sofuu_table_create(&schema.raw) };
        if ptr.is_null() {
            return Err("Failed to create Sofuu table (allocation failed)".into());
        }
        Ok(Self {
            raw: ptr,
            schema: schema.clone(),
        })
    }

    pub fn row_count(&self) -> usize {
        if self.raw.is_null() {
            return 0;
        }
        let rc = unsafe { sofuu_table_row_count(self.raw) };
        if rc < 0 { 0 } else { rc as usize }
    }

    pub fn schema(&self) -> &SofuSchema {
        &self.schema
    }

    pub fn insert(&mut self, row: &[SofuValue]) -> Result<(), String> {
        if self.raw.is_null() {
            return Err("Table is closed".into());
        }
        if row.len() > SOFUU_MAX_COLS {
            return Err(format!("Row exceeds max columns ({SOFUU_MAX_COLS})"));
        }

        // Prepare temporary C value array
        let mut c_vals = [SofuuValC::default(); SOFUU_MAX_COLS];

        for (i, val) in row.iter().enumerate() {
            match val {
                SofuValue::Null => {
                    c_vals[i].val_type = SofuuColType::Null as u8;
                }
                SofuValue::Int64(v) => {
                    c_vals[i].val_type = SofuuColType::Int64 as u8;
                    c_vals[i].as_val.i64_val = *v;
                }
                SofuValue::Int32(v) => {
                    c_vals[i].val_type = SofuuColType::Int32 as u8;
                    c_vals[i].as_val.i32_val = *v;
                }
                SofuValue::Float64(v) => {
                    c_vals[i].val_type = SofuuColType::Float64 as u8;
                    c_vals[i].as_val.f64_val = *v;
                }
                SofuValue::Float32(v) => {
                    c_vals[i].val_type = SofuuColType::Float32 as u8;
                    c_vals[i].as_val.f32_val = *v;
                }
                SofuValue::Bool(v) => {
                    c_vals[i].val_type = SofuuColType::Bool as u8;
                    c_vals[i].as_val.b_val = *v;
                }
                SofuValue::Timestamp(v) => {
                    c_vals[i].val_type = SofuuColType::Timestamp as u8;
                    c_vals[i].as_val.i64_val = *v;
                }
                SofuValue::String(s) => {
                    c_vals[i].val_type = SofuuColType::String as u8;
                    c_vals[i].as_val.str_val = SofuuStrRef {
                        ptr: s.as_ptr() as *const c_char,
                        len: s.len() as u32,
                    };
                }
                SofuValue::Bytes(b) => {
                    c_vals[i].val_type = SofuuColType::Bytes as u8;
                    c_vals[i].as_val.bytes_val = SofuuBytesRef {
                        ptr: b.as_ptr(),
                        len: b.len() as u32,
                    };
                }
                SofuValue::Json(j) => {
                    c_vals[i].val_type = SofuuColType::Json as u8;
                    c_vals[i].as_val.str_val = SofuuStrRef {
                        ptr: j.as_ptr() as *const c_char,
                        len: j.len() as u32,
                    };
                }
            }
        }

        let rc = unsafe {
            sofuu_table_insert(self.raw, c_vals.as_ptr(), row.len() as u32)
        };
        if rc != 0 {
            Err(format!("Table insert failed with error code: {rc}"))
        } else {
            Ok(())
        }
    }

    pub fn get_row(&self, row_idx: usize) -> Result<Vec<SofuValue>, String> {
        if self.raw.is_null() {
            return Err("Table is closed".into());
        }
        let total = self.row_count();
        if row_idx >= total {
            return Err(format!("Row index out of bounds: {row_idx} >= {total}"));
        }

        let mut out_vals = [SofuuValC::default(); SOFUU_MAX_COLS];

        let copied = unsafe {
            sofuu_table_get_row(self.raw, row_idx as u32, out_vals.as_mut_ptr(), SOFUU_MAX_COLS as u32)
        };
        if copied < 0 {
            return Err(format!("Failed to get row {row_idx} (rc: {copied})"));
        }

        let mut row = Vec::with_capacity(copied as usize);
        for i in 0..copied as usize {
            let v = &out_vals[i];
            let val = match SofuuColType::from_u8(v.val_type) {
                SofuuColType::Null => SofuValue::Null,
                SofuuColType::Int64 => unsafe { SofuValue::Int64(v.as_val.i64_val) },
                SofuuColType::Int32 => unsafe { SofuValue::Int32(v.as_val.i32_val) },
                SofuuColType::Float64 => unsafe { SofuValue::Float64(v.as_val.f64_val) },
                SofuuColType::Float32 => unsafe { SofuValue::Float32(v.as_val.f32_val) },
                SofuuColType::Bool => unsafe { SofuValue::Bool(v.as_val.b_val) },
                SofuuColType::Timestamp => unsafe { SofuValue::Timestamp(v.as_val.i64_val) },
                SofuuColType::String => unsafe {
                    let s_ref = v.as_val.str_val;
                    if s_ref.ptr.is_null() || s_ref.len == 0 {
                        SofuValue::String(String::new())
                    } else {
                        let slice = std::slice::from_raw_parts(s_ref.ptr as *const u8, s_ref.len as usize);
                        SofuValue::String(String::from_utf8_lossy(slice).into_owned())
                    }
                },
                SofuuColType::Bytes => unsafe {
                    let b_ref = v.as_val.bytes_val;
                    if b_ref.ptr.is_null() || b_ref.len == 0 {
                        SofuValue::Bytes(Vec::new())
                    } else {
                        let slice = std::slice::from_raw_parts(b_ref.ptr, b_ref.len as usize);
                        SofuValue::Bytes(slice.to_vec())
                    }
                },
                SofuuColType::Json => unsafe {
                    let s_ref = v.as_val.str_val;
                    if s_ref.ptr.is_null() || s_ref.len == 0 {
                        SofuValue::Json(String::new())
                    } else {
                        let slice = std::slice::from_raw_parts(s_ref.ptr as *const u8, s_ref.len as usize);
                        SofuValue::Json(String::from_utf8_lossy(slice).into_owned())
                    }
                },
            };
            row.push(val);
        }

        Ok(row)
    }

    pub fn query(&self, q: &SofuQuery) -> Result<SofuTable, String> {
        if self.raw.is_null() {
            return Err("Table is closed".into());
        }

        let mut c_query = SofuuQueryC::default();
        c_query.min_id = q.min_id;
        c_query.max_id = q.max_id;
        c_query.min_timestamp = q.min_timestamp;
        c_query.max_timestamp = q.max_timestamp;
        c_query.limit = q.limit;
        c_query.offset = q.offset;
        c_query.order_desc = q.order_desc;

        let _pinned_name = q.filter_col_name.as_ref().map(|s| CString::new(s.as_str()).unwrap());
        if let Some(ref c_str) = _pinned_name {
            c_query.filter_col_name = c_str.as_ptr();
        }

        if let Some(SofuValue::String(ref s)) = q.filter_val {
            c_query.filter_val.val_type = SofuuColType::String as u8;
            c_query.filter_val.as_val.str_val = SofuuStrRef {
                ptr: s.as_ptr() as *const c_char,
                len: s.len() as u32,
            };
        }

        let res_ptr = unsafe { sofuu_table_query(self.raw, &c_query) };
        if res_ptr.is_null() {
            return Err("Query execution failed".into());
        }

        Ok(SofuTable {
            raw: res_ptr,
            schema: self.schema.clone(),
        })
    }

    pub fn write_qtsq<P: AsRef<Path>>(&self, path: P, password: Option<&str>) -> Result<(), String> {
        if self.raw.is_null() {
            return Err("Table is closed".into());
        }
        let path_str = path.as_ref().to_str().ok_or("Invalid UTF-8 path")?;
        let c_path = CString::new(path_str).map_err(|e| e.to_string())?;

        let _pass_cs = password.map(|p| CString::new(p).unwrap());
        let pass_ptr = _pass_cs.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null());

        let rc = unsafe { sofuu_table_write_qtsq(self.raw, c_path.as_ptr(), pass_ptr) };
        if rc != 0 {
            Err(format!("sofuu_table_write_qtsq failed with rc: {rc}"))
        } else {
            Ok(())
        }
    }

    pub fn read_qtsq<P: AsRef<Path>>(
        path: P,
        table_name: &str,
        password: Option<&str>,
    ) -> Result<Self, String> {
        let path_str = path.as_ref().to_str().ok_or("Invalid UTF-8 path")?;
        let c_path = CString::new(path_str).map_err(|e| e.to_string())?;
        let c_name = CString::new(table_name).map_err(|e| e.to_string())?;

        let _pass_cs = password.map(|p| CString::new(p).unwrap());
        let pass_ptr = _pass_cs.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null());

        let res_ptr = unsafe {
            sofuu_table_read_qtsq(c_path.as_ptr(), c_name.as_ptr(), pass_ptr)
        };
        if res_ptr.is_null() {
            return Err(format!("Failed to read table '{table_name}' from {}", path_str));
        }

        // Read deserialized schema directly from the table (offset 0)
        let schema_c = unsafe { *(res_ptr as *const SofuuSchemaC) };
        let schema = SofuSchema { raw: schema_c };

        Ok(Self {
            raw: res_ptr,
            schema,
        })
    }

    /// Fast checkpoint-indexed query directly against an on-disk QTSQ file.
    /// Only matching columnar blocks are decompressed, enabling sub-1.5ms pagination.
    pub fn query_qtsq<P: AsRef<Path>>(
        path: P,
        table_name: &str,
        password: Option<&str>,
        query: &SofuQuery,
    ) -> Result<Self, String> {
        let path_str = path.as_ref().to_str().ok_or("Invalid UTF-8 path")?;
        let c_path = CString::new(path_str).map_err(|e| e.to_string())?;
        let c_name = CString::new(table_name).map_err(|e| e.to_string())?;

        let _pass_cs = password.map(|p| CString::new(p).unwrap());
        let pass_ptr = _pass_cs.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null());

        let mut c_query = SofuuQueryC::default();
        c_query.min_id = query.min_id;
        c_query.max_id = query.max_id;
        c_query.min_timestamp = query.min_timestamp;
        c_query.max_timestamp = query.max_timestamp;
        c_query.limit = query.limit;
        c_query.offset = query.offset;
        c_query.order_desc = query.order_desc;

        let _pinned_name = query.filter_col_name.as_ref().map(|s| CString::new(s.as_str()).unwrap());
        if let Some(ref c_str) = _pinned_name {
            c_query.filter_col_name = c_str.as_ptr();
        }

        if let Some(SofuValue::String(ref s)) = query.filter_val {
            c_query.filter_val.val_type = SofuuColType::String as u8;
            c_query.filter_val.as_val.str_val = SofuuStrRef {
                ptr: s.as_ptr() as *const c_char,
                len: s.len() as u32,
            };
        }

        let res_ptr = unsafe {
            sofuu_table_query_qtsq(c_path.as_ptr(), c_name.as_ptr(), pass_ptr, &c_query)
        };
        if res_ptr.is_null() {
            return Err(format!("Failed to query table '{table_name}' from {}", path_str));
        }

        let schema_c = unsafe { *(res_ptr as *const SofuuSchemaC) };
        let schema = SofuSchema { raw: schema_c };

        Ok(Self {
            raw: res_ptr,
            schema,
        })
    }
}

impl Drop for SofuTable {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { sofuu_table_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Unit Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation_and_columns() {
        let schema = SofuSchema::new("test_messages")
            .unwrap()
            .add_col("id", SofuuColType::Int64, SOFUU_COL_PRIMARY_KEY | SOFUU_COL_AUTO_INC)
            .unwrap()
            .add_col("session_id", SofuuColType::String, SOFUU_COL_REQUIRED | SOFUU_COL_INDEXED)
            .unwrap()
            .add_col("role", SofuuColType::String, SOFUU_COL_REQUIRED)
            .unwrap()
            .add_col("content", SofuuColType::String, SOFUU_COL_REQUIRED)
            .unwrap()
            .add_col("timestamp", SofuuColType::Timestamp, SOFUU_COL_REQUIRED)
            .unwrap();

        assert_eq!(schema.num_cols(), 5);
        assert_eq!(schema.table_name(), "test_messages");
        assert_eq!(schema.pk_col_idx(), Some(0));
        assert_eq!(schema.timestamp_col_idx(), Some(4));
        assert_eq!(schema.find_col("session_id"), Some(1));
        assert_eq!(schema.find_col("nonexistent"), None);
    }

    #[test]
    fn test_table_insert_and_get_row() {
        let schema = SofuSchema::new("users")
            .unwrap()
            .add_col("id", SofuuColType::Int64, SOFUU_COL_PRIMARY_KEY | SOFUU_COL_AUTO_INC)
            .unwrap()
            .add_col("name", SofuuColType::String, SOFUU_COL_REQUIRED)
            .unwrap()
            .add_col("active", SofuuColType::Bool, 0)
            .unwrap()
            .add_col("created_at", SofuuColType::Timestamp, 0)
            .unwrap();

        let mut tbl = SofuTable::create(&schema).unwrap();
        assert_eq!(tbl.row_count(), 0);

        let row1 = vec![
            SofuValue::Int64(0), // Auto-inc fills this
            SofuValue::String("Haruhito".into()),
            SofuValue::Bool(true),
            SofuValue::Timestamp(1729500000000),
        ];
        tbl.insert(&row1).unwrap();

        let row2 = vec![
            SofuValue::Int64(0),
            SofuValue::String("Alice".into()),
            SofuValue::Bool(false),
            SofuValue::Timestamp(1729500001000),
        ];
        tbl.insert(&row2).unwrap();

        assert_eq!(tbl.row_count(), 2);

        let r0 = tbl.get_row(0).unwrap();
        assert_eq!(r0[0], SofuValue::Int64(1)); // Auto-inc started at 1
        assert_eq!(r0[1], SofuValue::String("Haruhito".into()));
        assert_eq!(r0[2], SofuValue::Bool(true));
        assert_eq!(r0[3], SofuValue::Timestamp(1729500000000));

        let r1 = tbl.get_row(1).unwrap();
        assert_eq!(r1[0], SofuValue::Int64(2));
        assert_eq!(r1[1], SofuValue::String("Alice".into()));
        assert_eq!(r1[2], SofuValue::Bool(false));
    }

    #[test]
    fn test_table_roundtrip_plaintext_and_encrypted() {
        let temp_dir = std::env::temp_dir();
        let plain_file = temp_dir.join("test_sofuu_plain.qtsq");
        let enc_file = temp_dir.join("test_sofuu_enc.qtsq");

        let schema = SofuSchema::new("events")
            .unwrap()
            .add_col("id", SofuuColType::Int64, SOFUU_COL_PRIMARY_KEY | SOFUU_COL_AUTO_INC)
            .unwrap()
            .add_col("event_type", SofuuColType::String, SOFUU_COL_REQUIRED)
            .unwrap()
            .add_col("timestamp", SofuuColType::Timestamp, 0)
            .unwrap();

        let mut tbl = SofuTable::create(&schema).unwrap();
        for i in 0..150 {
            tbl.insert(&[
                SofuValue::Int64(0),
                SofuValue::String(format!("evt_{i}")),
                SofuValue::Timestamp(1729500000000 + i * 1000),
            ])
            .unwrap();
        }
        assert_eq!(tbl.row_count(), 150);

        // 1. Plaintext Write & Read
        tbl.write_qtsq(&plain_file, None).unwrap();
        let loaded_plain = SofuTable::read_qtsq(&plain_file, "events", None).unwrap();
        assert_eq!(loaded_plain.row_count(), 150);
        let first = loaded_plain.get_row(0).unwrap();
        assert_eq!(first[1], SofuValue::String("evt_0".into()));
        let last = loaded_plain.get_row(149).unwrap();
        assert_eq!(last[1], SofuValue::String("evt_149".into()));

        // 2. Encrypted Write & Read (AES-256-GCM vault)
        let pass = "sofuu-vault-key-2026";
        tbl.write_qtsq(&enc_file, Some(pass)).unwrap();
        let loaded_enc = SofuTable::read_qtsq(&enc_file, "events", Some(pass)).unwrap();
        assert_eq!(loaded_enc.row_count(), 150);
        let enc_row = loaded_enc.get_row(50).unwrap();
        assert_eq!(enc_row[1], SofuValue::String("evt_50".into()));

        // Cleanup
        let _ = std::fs::remove_file(plain_file);
        let _ = std::fs::remove_file(enc_file);
    }

    #[test]
    fn test_checkpoint_query_filtering() {
        let schema = SofuSchema::new("logs")
            .unwrap()
            .add_col("id", SofuuColType::Int64, SOFUU_COL_PRIMARY_KEY | SOFUU_COL_AUTO_INC)
            .unwrap()
            .add_col("level", SofuuColType::String, SOFUU_COL_REQUIRED)
            .unwrap()
            .add_col("timestamp", SofuuColType::Timestamp, 0)
            .unwrap();

        let mut tbl = SofuTable::create(&schema).unwrap();
        for i in 0..200 {
            let level = if i % 2 == 0 { "INFO" } else { "ERROR" };
            tbl.insert(&[
                SofuValue::Int64(0),
                SofuValue::String(level.into()),
                SofuValue::Timestamp(1729500000000 + i * 1000),
            ])
            .unwrap();
        }

        // Query with time-range filtering
        let q = SofuQuery::new()
            .min_timestamp(1729500010000)
            .max_timestamp(1729500020000);
        let res = tbl.query(&q).unwrap();
        assert_eq!(res.row_count(), 11);

        // Query with limit
        let q_lim = SofuQuery::new().limit(5);
        let res_lim = tbl.query(&q_lim).unwrap();
        assert_eq!(res_lim.row_count(), 5);
    }

    #[test]
    fn test_selective_disk_query_qtsq() {
        let schema = SofuSchema::new("events")
            .unwrap()
            .add_col("id", SofuuColType::Int64, SOFUU_COL_PRIMARY_KEY | SOFUU_COL_AUTO_INC)
            .unwrap()
            .add_col("data", SofuuColType::String, SOFUU_COL_REQUIRED)
            .unwrap()
            .add_col("timestamp", SofuuColType::Timestamp, 0)
            .unwrap();

        let mut tbl = SofuTable::create(&schema).unwrap();
        for i in 0..250 {
            tbl.insert(&[
                SofuValue::Int64(0),
                SofuValue::String(format!("Event payload #{i}")),
                SofuValue::Timestamp(1729500000000 + i * 1000),
            ])
            .unwrap();
        }

        let tmp_file = std::env::temp_dir().join(format!("test_sofuu_query_{}.qtsq", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        tbl.write_qtsq(&tmp_file, None).unwrap();

        // Selective query for 20 latest events descending
        let q = SofuQuery::new()
            .min_timestamp(1729500100000)
            .limit(20)
            .order_desc(true);

        let q_res = SofuTable::query_qtsq(&tmp_file, "events", None, &q).unwrap();
        assert_eq!(q_res.row_count(), 20);

        let top_row = q_res.get_row(0).unwrap();
        assert_eq!(top_row[0], SofuValue::Int64(250));

        let _ = std::fs::remove_file(&tmp_file);
    }
}
