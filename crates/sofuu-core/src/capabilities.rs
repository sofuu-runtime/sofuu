// capabilities.rs — the R1 capability manifest (SPEC-RUNTIME-CONTRACT §2).
//
// One source of truth for "what can this build do", assembled from three
// places and exposed as a single JSON document:
//
//   * compile time  — version, os/arch, feature cfgs (qtsq, metal, simd)
//   * runtime       — thread count, RAM, GPU availability
//   * contract/     — the audited surface matrix (contract/matrix.json),
//                     embedded at build time so the manifest cannot drift
//                     from the P0 audit by hand.
//
// JS surface: `sofuu.capabilities()` returns the JSON string. A JSON string
// (rather than an object) matches the rest of the Rust-backed surface
// (`ai.modelCaps`, `memory.stats`) and keeps the QuickJS boundary one call.
//
// Phase P1. This module only reads; it changes no engine behavior.

use std::ffi::CString;
use std::os::raw::c_int;

use serde_json::{json, Map, Value};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

pub const SCHEMA: u32 = 1;

/// The P0 audit is the source of truth for per-surface availability.
/// `contract/` is tracked; a build without it fails here by design rather
/// than silently reporting a stale matrix.
const MATRIX: &str = include_str!("../../../contract/matrix.json");

#[cfg(any(target_os = "macos", target_os = "ios"))]
unsafe extern "C" {
    fn sofuu_metal_available() -> bool;
}

fn metal_available() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        unsafe { sofuu_metal_available() }
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        false
    }
}

/// Total physical memory in MiB. `None` means "not probed on this platform",
/// which the manifest reports as `null` rather than guessing.
fn ram_mb() -> Option<u64> {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut value: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        let name = b"hw.memsize\0";
        let rc = libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            &mut value as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        );
        if rc == 0 {
            return Some(value / (1024 * 1024));
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(raw) = std::fs::read_to_string("/proc/meminfo") {
            for line in raw.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    if let Some(kb) = rest
                        .split_whitespace()
                        .next()
                        .and_then(|x| x.parse::<u64>().ok())
                    {
                        return Some(kb / 1024);
                    }
                }
            }
        }
    }
    None
}

/// Transform the audited matrix cells into the manifest's surface map:
/// surface -> primitive -> {status, missing?, reason?, applicable?}.
fn surfaces() -> Value {
    let parsed: Value = serde_json::from_str(MATRIX).unwrap_or(Value::Null);
    let mut out = Map::new();
    let Some(cells) = parsed.get("cells").and_then(Value::as_array) else {
        return Value::Object(out);
    };
    for cell in cells {
        let (Some(primitive), Some(surface)) = (
            cell.get("primitive").and_then(Value::as_str),
            cell.get("surface").and_then(Value::as_str),
        ) else {
            continue;
        };
        let raw = cell
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unavailable");
        let mut entry = Map::new();
        if raw == "n/a" {
            entry.insert("status".into(), json!("unavailable"));
            entry.insert("applicable".into(), json!(false));
            entry.insert("reason".into(), json!("not applicable on this surface"));
        } else {
            entry.insert("status".into(), json!(raw));
            if let Some(missing) = cell.get("missing").and_then(Value::as_array) {
                if !missing.is_empty() {
                    entry.insert("missing".into(), Value::Array(missing.clone()));
                }
            }
            if let Some(reason) = cell.get("reason") {
                entry.insert("reason".into(), reason.clone());
            }
        }
        let slot = out
            .entry(surface.to_string())
            .or_insert_with(|| json!({}));
        if let Some(map) = slot.as_object_mut() {
            map.insert(primitive.to_string(), Value::Object(entry));
        }
    }
    Value::Object(out)
}

/// Logical cores available to this process.
///
/// `std::thread::available_parallelism()` returns **1** in the static-musl
/// Linux build (its `sched_getaffinity` path fails and Rust falls back to 1),
/// which made `capabilities()` — and therefore `doctor`, which prints this
/// field — report a single core on a 4-core Raspberry Pi 5 while the
/// inference engine fanned out to 4 (see `infer/qwen35.rs:58`, which documents
/// the same defect and works around it by borrowing the C++ pool's count).
/// A host that sized its own thread pool from `capabilities` was handed one
/// worker on a four-core machine.
///
/// The pool is authoritative once initialised; otherwise use the OS view and,
/// on Linux, fall back to counting `/sys/devices/system/cpu/cpu[0-9]+`.
fn logical_cpus() -> usize {
    let from_pool = crate::infer::quant::get_thread_count();
    if from_pool > 1 {
        return from_pool;
    }
    if let Ok(n) = std::thread::available_parallelism() {
        if n.get() > 1 {
            return n.get();
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") {
            let n = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.strip_prefix("cpu")
                        .map(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
                        .unwrap_or(false)
                })
                .count();
            if n > 0 {
                return n;
            }
        }
    }
    from_pool.max(1)
}

/// Features the *target platform* cannot provide at all, as opposed to
/// features this build dropped. Metal exists only on Apple platforms, so a
/// Linux or Windows build listing it under `missing` is not a degraded build —
/// `doctor` used to count it as a failure and exited 1 on every healthy
/// Linux install (R-25b, AUDIT-2026-10-29).
fn platform_absent() -> Vec<&'static str> {
    if cfg!(any(target_os = "macos", target_os = "ios")) {
        Vec::new()
    } else {
        vec!["metal"]
    }
}

/// The manifest as a JSON value.
pub fn manifest() -> Value {
    let mut features = vec!["simd", "qtsq"];
    let mut missing = Vec::new();
    if cfg!(any(target_os = "macos", target_os = "ios")) {
        features.push("metal");
    } else {
        missing.push("metal");
    }

    json!({
        "schema": SCHEMA,
        "runtime": {
            "version": env!("CARGO_PKG_VERSION"),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "threads": logical_cpus(),
            "ram_mb": ram_mb(),
            "gpu": { "backend": "metal", "available": metal_available() }
        },
        "build": {
            "features": features,
            "missing": missing,
            "platform_absent": platform_absent()
        },
        "engine": {
            "formats": ["gguf", "safetensors", "pt", "npy"],
            "quant_types": ["f32", "f16", "q4_K", "q6_K", "q8_0", "q3_K_S"],
            "max_ctx": Value::Null,
            "kv_modes": ["f16", "kivi4", "kivi2"],
            "prefill_backends": ["cpu-tiles", "metal-mma"],
            "paths": {
                "cpu-tiles": { "tier": "exact", "default": true },
                "metal-mma": { "tier": "tolerant", "bound": "2e-2 abs/scale", "default": false },
                "mmq": { "tier": "best-effort", "default": false }
            }
        },
        "model": { "loaded": false },
        "surfaces": surfaces(),
        "notes": [
            "max_ctx is model-dependent; null until a model is loaded",
            "ram_mb is null when the platform probe is unavailable",
            "quant_types lists the types with shipped kernels; the loader gate is authoritative"
        ]
    })
}

/// Serialized manifest, for the JS boundary and tests.
pub fn manifest_json() -> String {
    serde_json::to_string(&manifest()).unwrap_or_else(|_| "{}".to_string())
}

unsafe extern "C" fn js_capabilities(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let cs = CString::new(manifest_json()).unwrap_or_default();
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

/// Register `sofuu.capabilities()`. Called from `engine_register_builtins`
/// after the native modules, so the `sofuu` object already exists.
pub unsafe fn mod_capabilities_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(
            ctx,
            global,
            c"sofuu".as_ptr(),
            qjs::sofuu_js_dup_value(ctx, sofuu_obj),
        );
    }
    qjs::sofuu_js_set_property_str(
        ctx,
        sofuu_obj,
        c"capabilities".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_capabilities, c"capabilities".as_ptr(), 0),
    );
    // R1 (T1.5): sofuu.model.describe(path) — the per-model capability block.
    let model_obj = qjs::sofuu_js_new_object(ctx);
    qjs::sofuu_js_set_property_str(
        ctx,
        model_obj,
        c"describe".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_model_describe, c"describe".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"model".as_ptr(), model_obj);
    qjs::sofuu_js_free_value(ctx, sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

// ── model.describe (T1.5) ────────────────────────────────────────────────────
//
// Reads a GGUF header directly (magic/version/kv/tensor-infos) so the R1
// model block never depends on a loader that may refuse the file for other
// reasons. The loader gate stays authoritative for "can this build run it";
// describe answers "what is this file".

fn rd_exact(r: &mut impl std::io::Read, n: usize) -> Result<Vec<u8>, String> {
    let mut v = vec![0u8; n];
    r.read_exact(&mut v).map_err(|e| format!("read: {e}"))?;
    Ok(v)
}

fn rd_u32(r: &mut impl std::io::Read) -> Result<u32, String> {
    let b = rd_exact(r, 4)?;
    Ok(u32::from_le_bytes(b.try_into().unwrap()))
}

fn rd_u64(r: &mut impl std::io::Read) -> Result<u64, String> {
    let b = rd_exact(r, 8)?;
    Ok(u64::from_le_bytes(b.try_into().unwrap()))
}

fn rd_str(r: &mut impl std::io::Read) -> Result<String, String> {
    let n = rd_u64(r)? as usize;
    if n > (1 << 22) {
        return Err("metadata string exceeds 4 MiB".into());
    }
    Ok(String::from_utf8_lossy(&rd_exact(r, n)?).into_owned())
}

fn scalar_size(ty: u32) -> Option<usize> {
    Some(match ty {
        0 | 1 | 7 => 1,   // u8, i8, bool
        2 | 3 => 2,       // u16, i16
        4 | 5 | 6 => 4,   // u32, i32, f32
        10 | 11 | 12 => 8, // u64, i64, f64
        _ => return None, // 8 string, 9 array
    })
}

fn skip_value(r: &mut impl std::io::Read, ty: u32) -> Result<(), String> {
    if let Some(n) = scalar_size(ty) {
        rd_exact(r, n)?;
        return Ok(());
    }
    match ty {
        8 => {
            rd_str(r)?;
        }
        9 => {
            let et = rd_u32(r)?;
            if scalar_size(et).is_none() && et != 8 {
                return Err(format!("unknown gguf array element type {et}"));
            }
            let n = rd_u64(r)?;
            if n > (1 << 28) {
                return Err("metadata array exceeds 256M elements".into());
            }
            for _ in 0..n {
                skip_value(r, et)?;
            }
        }
        other => return Err(format!("unknown gguf value type {other}")),
    }
    Ok(())
}

fn ggml_type_name(t: u32) -> String {
    match t {
        0 => "f32".into(),
        1 => "f16".into(),
        2 => "q4_0".into(),
        3 => "q4_1".into(),
        6 => "q5_0".into(),
        7 => "q5_1".into(),
        8 => "q8_0".into(),
        9 => "q8_1".into(),
        10 => "q2_K".into(),
        11 => "q3_K".into(),
        12 => "q4_K".into(),
        13 => "q5_K".into(),
        14 => "q6_K".into(),
        25 => "iq4_nl".into(),
        28 => "iq4_xs".into(),
        other => format!("t{other}"),
    }
}

fn gguf_file_type_name(t: u32) -> String {
    match t {
        0 => "f32".into(),
        1 => "f16".into(),
        2 => "q4_0".into(),
        3 => "q4_1".into(),
        7 => "q8_0".into(),
        8 => "q5_0".into(),
        9 => "q5_1".into(),
        10 => "q2_K".into(),
        11 => "q3_K_S".into(),
        12 => "q3_K_M".into(),
        13 => "q3_K_L".into(),
        14 => "q4_K_S".into(),
        15 => "q4_K_M".into(),
        16 => "q5_K_S".into(),
        17 => "q5_K_M".into(),
        18 => "q6_K".into(),
        19 => "iq2_xxs".into(),
        20 => "iq2_xs".into(),
        26 => "iq4_xs".into(),
        other => format!("ft{other}"),
    }
}

/// Parse the GGUF header far enough to describe the model. Reads KVs and
/// tensor infos only — never tensor data.
fn read_gguf_summary(path: &str) -> Result<Value, String> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut r = std::io::BufReader::new(f);

    let magic = rd_u32(&mut r)?;
    if magic != 0x46554747 {
        return Err("not a GGUF file (bad magic)".into());
    }
    let version = rd_u32(&mut r)?;
    if version != 2 && version != 3 {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let tensor_count = rd_u64(&mut r)?;
    let kv_count = rd_u64(&mut r)?;
    if kv_count > (1 << 16) {
        return Err("unreasonable kv count".into());
    }

    let mut meta: Map<String, Value> = Map::new();
    for _ in 0..kv_count {
        let key = rd_str(&mut r)?;
        let ty = rd_u32(&mut r)?;
        // Keep scalar metadata and top-level strings that describe the model
        // (general.architecture is a string); skip tokenizer.* and arrays.
        let keep = !key.starts_with("tokenizer.")
            && (scalar_size(ty).is_some() || ty == 8)
            && ty != 9;
        if keep {
            let v: Value = match ty {
                4 => json!(rd_u32(&mut r)?),
                10 | 11 => json!(rd_u64(&mut r)? as i64),
                8 => json!(rd_str(&mut r)?),
                6 => {
                    let b = rd_exact(&mut r, 4)?;
                    json!(f32::from_le_bytes(b.try_into().unwrap()))
                }
                12 => {
                    let b = rd_exact(&mut r, 8)?;
                    json!(f64::from_le_bytes(b.try_into().unwrap()))
                }
                5 => {
                    let b = rd_exact(&mut r, 4)?;
                    json!(i32::from_le_bytes(b.try_into().unwrap()))
                }
                3 => {
                    let b = rd_exact(&mut r, 2)?;
                    json!(i16::from_le_bytes(b.try_into().unwrap()))
                }
                2 => {
                    let b = rd_exact(&mut r, 2)?;
                    json!(u16::from_le_bytes(b.try_into().unwrap()))
                }
                _ => {
                    let b = rd_exact(&mut r, 1)?;
                    json!(b[0])
                }
            };
            meta.insert(key, v);
        } else {
            skip_value(&mut r, ty)?;
        }
    }

    // Tensor infos: collect the distinct weight encodings actually present.
    let mut quant_types: Vec<String> = Vec::new();
    for _ in 0..tensor_count.min(1 << 16) {
        let _name = rd_str(&mut r)?;
        let n_dims = rd_u32(&mut r)?;
        if n_dims > 4 {
            return Err(format!("tensor with {n_dims} dims"));
        }
        for _ in 0..n_dims {
            rd_u64(&mut r)?;
        }
        let ty = rd_u32(&mut r)?;
        rd_u64(&mut r)?; // data offset
        let name = ggml_type_name(ty);
        if !quant_types.contains(&name) && quant_types.len() < 16 {
            quant_types.push(name);
        }
    }

    let arch = meta
        .get("general.architecture")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let file_type = meta.get("general.file_type").and_then(Value::as_u64);
    Ok(json!({
        "arch": arch,
        "gguf_version": version,
        "ctx": meta.get(format!("{arch}.context_length").as_str()).cloned().unwrap_or(Value::Null),
        "blocks": meta.get(format!("{arch}.block_count").as_str()).cloned().unwrap_or(Value::Null),
        "heads": meta.get(format!("{arch}.attention.head_count").as_str()).cloned().unwrap_or(Value::Null),
        "embedding": meta.get(format!("{arch}.embedding_length").as_str()).cloned().unwrap_or(Value::Null),
        "quant": file_type
            .map(|t| Value::String(gguf_file_type_name(t as u32)))
            .unwrap_or(Value::Null),
        "quant_types": quant_types,
    }))
}

/// The R1 `model` block for one local GGUF file (SPEC §2.2). Never throws:
/// an unreadable model returns a typed `E_CAPABILITY` envelope so hosts can
/// render the reason instead of guessing.
pub fn describe_model(path: &str) -> Value {
    let size_mb = std::fs::metadata(path).ok().map(|m| m.len() / (1024 * 1024));
    match read_gguf_summary(path) {
        Ok(mut v) => {
            v["id"] = json!(path);
            v["ok"] = json!(true);
            v["source"] = json!("local");
            v["privacy_class"] = json!("local");
            v["cost_class"] = json!("free");
            v["tool_calling"] = json!("emulated");
            v["structured_out"] = json!("emulated");
            v["vision"] = json!(false);
            v["streaming"] = json!(true);
            v["size_mb"] = json!(size_mb);
            v["tier"] = json!("exact");
            v["latency_class"] = json!(match size_mb {
                Some(mb) if mb < 1024 => "fast",
                Some(mb) if mb < 4096 => "mid",
                _ => "slow",
            });
            v
        }
        Err(e) => json!({
            "id": path, "ok": false, "code": "E_CAPABILITY", "op": "model.describe",
            "surface": "js", "retryable": false,
            "message": format!("cannot read model metadata: {e}"),
        }),
    }
}

unsafe extern "C" fn js_model_describe(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let path = if argc < 1 {
        String::new()
    } else {
        let mut len = 0usize;
        let c = qjs::JS_ToCStringLen2(ctx, &mut len, *argv, 0);
        if c.is_null() {
            String::new()
        } else {
            let s = std::ffi::CStr::from_ptr(c).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, c);
            s
        }
    };
    if path.is_empty() {
        let cs = CString::new(
            "{\"ok\":false,\"code\":\"E_CAPABILITY\",\"op\":\"model.describe\",\
             \"message\":\"model.describe(path) requires a model path\"}",
        )
        .unwrap_or_default();
        return qjs::sofuu_js_new_string(ctx, cs.as_ptr());
    }
    let cs = CString::new(
        serde_json::to_string(&describe_model(&path)).unwrap_or_else(|_| "{}".into()),
    )
    .unwrap_or_default();
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

// ── R1.2: loud refusal on QTSQ-free builds (T1.3) ───────────────────────────
//
// A build without the QTSQ codec registers every memory/kv/storage entry
// point as a throwing stub. The 2026-09-22 brain-less tarball shipped
// because these silently no-opped; the typed error names the missing
// capability and the remediation.

#[cfg(not(has_qtsq))]
const MEMORY_METHODS: [&str; 24] = [
    "open", "remember", "learn", "recall", "search", "count", "flush", "forget", "close", "list",
    "export", "importRecords", "stats", "retain", "consolidate", "compact", "markPositive",
    "rememberWorking", "promoteWorking", "clearWorking", "rememberEntity", "migrate",
    "embeddingInfo", "inspect",
];

#[cfg(not(has_qtsq))]
unsafe extern "C" fn js_capability_missing(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let msg = "E_CAPABILITY: this binary was built without the QTSQ codec, so \
               memory/storage are unavailable and nothing was stored. Rebuild \
               with SOFUU_QTSQ_DIR set (see the `make` guard) or install a \
               QTSQ-bearing build.";
    let cm = CString::new(msg).unwrap_or_default();
    qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), cm.as_ptr())
}

#[cfg(not(has_qtsq))]
unsafe fn register_throwing_namespace(ctx: *mut JSContext, parent: JSValue, name: &str, methods: &[&str]) {
    let obj = qjs::sofuu_js_new_object(ctx);
    for m in methods {
        let cm = CString::new(*m).unwrap_or_default();
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            cm.as_ptr(),
            qjs::sofuu_js_new_cfunction(ctx, js_capability_missing, cm.as_ptr(), 0),
        );
    }
    let cn = CString::new(name).unwrap_or_default();
    qjs::sofuu_js_set_property_str(ctx, parent, cn.as_ptr(), obj);
}

/// Register `sofuu.memory` / `sofuu.brain` / `sofuu.kv` / `sofuu.storage` as
/// loudly-refusing stubs. Called from `engine_register_builtins` on builds
/// compiled without the QTSQ codec.
#[cfg(not(has_qtsq))]
pub unsafe fn mod_memory_unavailable_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(
            ctx,
            global,
            c"sofuu".as_ptr(),
            qjs::sofuu_js_dup_value(ctx, sofuu_obj),
        );
    }
    register_throwing_namespace(ctx, sofuu_obj, "memory", &MEMORY_METHODS);
    register_throwing_namespace(ctx, sofuu_obj, "brain", &MEMORY_METHODS);
    register_throwing_namespace(
        ctx,
        sofuu_obj,
        "kv",
        &["open", "save", "search", "flush", "close"],
    );
    register_throwing_namespace(ctx, sofuu_obj, "storage", &["open", "close"]);
    qjs::sofuu_js_free_value(ctx, sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_has_required_blocks() {
        let m = manifest();
        assert_eq!(m["schema"], SCHEMA);
        for key in ["runtime", "build", "engine", "model", "surfaces"] {
            assert!(m.get(key).is_some(), "missing block {key}");
        }
        assert!(m["runtime"]["version"].as_str().is_some());
        assert!(m["runtime"]["threads"].as_u64().unwrap_or(0) > 0);
        assert!(m["build"]["features"].as_array().is_some());
    }

    #[test]
    fn surfaces_cover_every_matrix_cell() {
        let m = manifest();
        let s = m["surfaces"].as_object().expect("surfaces object");
        assert_eq!(s.len(), 6, "six surfaces");
        for (name, primitives) in s {
            assert_eq!(
                primitives.as_object().map(|o| o.len()).unwrap_or(0),
                8,
                "surface {name} covers eight primitives"
            );
        }
        assert_eq!(m["surfaces"]["mcp"]["session"]["applicable"], false);
        // Finding #13 (the module-vs-instance shape split) is resolved: the
        // engine installs one flat memory/brain facade on QTSQ builds, so the
        // audited cell is `full` and carries no missing methods. The
        // QTSQ-free lane still refuses via build.missing + E_CAPABILITY.
        assert_eq!(m["surfaces"]["js"]["memory"]["status"], "full");
        assert!(m["surfaces"]["js"]["memory"].get("missing").is_none());
    }

    #[test]
    fn manifest_json_roundtrips() {
        let raw = manifest_json();
        let parsed: Value = serde_json::from_str(&raw).expect("valid JSON");
        assert_eq!(parsed["schema"], SCHEMA);
    }

    #[test]
    fn describe_model_reads_a_synthetic_gguf() {
        // Minimal v3 GGUF: magic, version, 1 tensor, 2 KVs
        // (general.architecture, llama.context_length), one f32 tensor info.
        fn put_u32(b: &mut Vec<u8>, v: u32) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        fn put_u64(b: &mut Vec<u8>, v: u64) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        fn put_str(b: &mut Vec<u8>, s: &str) {
            put_u64(b, s.len() as u64);
            b.extend_from_slice(s.as_bytes());
        }
        let mut b: Vec<u8> = Vec::new();
        put_u32(&mut b, 0x46554747);
        put_u32(&mut b, 3);
        put_u64(&mut b, 1); // tensor count
        put_u64(&mut b, 2); // kv count
        put_str(&mut b, "general.architecture");
        put_u32(&mut b, 8);
        put_str(&mut b, "llama");
        put_str(&mut b, "llama.context_length");
        put_u32(&mut b, 4);
        put_u32(&mut b, 4096);
        put_str(&mut b, "tok_embeddings.weight");
        put_u32(&mut b, 2); // n_dims
        put_u64(&mut b, 64);
        put_u64(&mut b, 16);
        put_u32(&mut b, 0); // f32
        put_u64(&mut b, 0); // offset

        let p = std::env::temp_dir().join(format!("sofuu-cap-gguf-{}.gguf", std::process::id()));
        std::fs::write(&p, &b).unwrap();
        let d = describe_model(p.to_str().unwrap());
        let _ = std::fs::remove_file(&p);
        assert_eq!(d["ok"], true, "{}", d["message"]);
        assert_eq!(d["arch"], "llama");
        assert_eq!(d["ctx"], 4096);
        assert_eq!(d["quant_types"][0], "f32");
        assert_eq!(d["tool_calling"], "emulated");
        assert_eq!(d["cost_class"], "free");
        assert_eq!(d["tier"], "exact");
    }

    #[test]
    fn describe_model_rejects_garbage_with_a_typed_error() {
        let p = std::env::temp_dir().join(format!("sofuu-cap-bad-{}.bin", std::process::id()));
        std::fs::write(&p, b"not a gguf at all").unwrap();
        let d = describe_model(p.to_str().unwrap());
        let _ = std::fs::remove_file(&p);
        assert_eq!(d["ok"], false);
        assert_eq!(d["code"], "E_CAPABILITY");
        assert_eq!(d["op"], "model.describe");
    }
}
