// ffi_exports.rs — C-ABI exports that let the remaining C core call into
// safe-Rust implementations mid-migration (ROADMAP Track D3: "wire Rust in").
//
// Memory contract: every returned buffer is allocated with libc::malloc so
// the C side can free() it plainly ( Rust's default allocator IS the system
// malloc here, but we do not rely on that).
//
// All functions tolerate NULL/zero inputs by returning NULL/""-style
// failures so the C caller can fall back to its previous behavior.

#![allow(clippy::missing_safety_doc)]

use std::os::raw::c_char;

#[inline]
unsafe fn cstr_or_null<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        None
    } else {
        std::ffi::CStr::from_ptr(p).to_str().ok()
    }
}

#[inline]
unsafe fn malloc_cstr(s: &str) -> *mut c_char {
    let b = s.as_bytes();
    let buf = libc::malloc(b.len() + 1) as *mut c_char;
    if buf.is_null() {
        return std::ptr::null_mut();
    }
    std::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, b.len());
    *buf.add(b.len()) = 0;
    buf
}

/* ── MCP JSON-RPC builders ────────────────────────────────────────
 * String-in/string-out wire formatting (newline-terminated), with real
 * JSON escaping — the retired C builders didn't escape their inputs. */

/// `sofuu_jsonrpc_parse_rs(line)` → malloc'd JSON describing an inbound
/// message, or NULL on malformed input. Shapes:
///   {"kind":"request","id":5,"method":"tools/call","params":{...}}
///   {"kind":"notification","method":"...","params":{...}}
///   {"kind":"response","id":7,"result":{...},"error":null}
///   {"kind":"response","id":8,"result":null,"error":{"code":-32601,"message":"..."}}
#[no_mangle]
pub unsafe extern "C" fn sofuu_jsonrpc_parse_rs(line: *const c_char) -> *mut c_char {
    let Some(l) = cstr_or_null(line) else {
        return std::ptr::null_mut();
    };
    use crate::mcp::jsonrpc::Inbound;
    let out = match crate::mcp::jsonrpc::parse(l) {
        Inbound::Request { id, method, params } => serde_json::json!({
            "kind": "request",
            "id": id,
            "method": method,
            "params": params,
        }),
        Inbound::Notification { method, params } => serde_json::json!({
            "kind": "notification",
            "method": method,
            "params": params,
        }),
        Inbound::Response { id, result, error } => serde_json::json!({
            "kind": "response",
            "id": id,
            "result": result,
            "error": error.map(|e| serde_json::json!({ "code": e.code, "message": e.message })),
        }),
        Inbound::Invalid(_) => return std::ptr::null_mut(),
    };
    malloc_cstr(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()))
}

/// `sofuu_jsonrpc_field_str_rs(json, key)` → malloc'd string value of a
/// top-level field, or NULL when absent/not a string. Safe extraction for
/// the remaining hand-rolled C call sites.
#[no_mangle]
pub unsafe extern "C" fn sofuu_jsonrpc_field_str_rs(
    json: *const c_char,
    key: *const c_char,
) -> *mut c_char {
    let (Some(j), Some(k)) = (cstr_or_null(json), cstr_or_null(key)) else {
        return std::ptr::null_mut();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(j) else {
        return std::ptr::null_mut();
    };
    match crate::mcp::jsonrpc::field_string(&v, k) {
        Some(s) => malloc_cstr(&s),
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_jsonrpc_request_rs(
    id: std::os::raw::c_longlong,
    method: *const c_char,
    params_json: *const c_char,
) -> *mut c_char {
    let Some(m) = cstr_or_null(method) else {
        return std::ptr::null_mut();
    };
    malloc_cstr(&crate::mcp::jsonrpc::request(id as u32, m, cstr_or_null(params_json)))
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_jsonrpc_notify_rs(
    method: *const c_char,
    params_json: *const c_char,
) -> *mut c_char {
    let Some(m) = cstr_or_null(method) else {
        return std::ptr::null_mut();
    };
    malloc_cstr(&crate::mcp::jsonrpc::notify(m, cstr_or_null(params_json)))
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_jsonrpc_response_rs(
    id: std::os::raw::c_longlong,
    result_json: *const c_char,
) -> *mut c_char {
    malloc_cstr(&crate::mcp::jsonrpc::response(id as u32, cstr_or_null(result_json)))
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_jsonrpc_error_rs(
    id: std::os::raw::c_longlong,
    code: std::os::raw::c_int,
    message: *const c_char,
) -> *mut c_char {
    let Some(m) = cstr_or_null(message) else {
        return std::ptr::null_mut();
    };
    malloc_cstr(&crate::mcp::jsonrpc::error(id as u32, code as i32, m))
}

/* ── SSE parser (opaque state object) ─────────────────────────────
 * The JS `sofuu.SSEParser` class in C holds one of these behind its
 * QuickJS opaque pointer; every feed() runs the safe Rust state machine
 * (\n\n AND \r\n\r\n blocks, multi-line data: joins, named events). */

/// `sofuu_sse_new()` → heap parser (caller must `sofuu_sse_free`).
#[no_mangle]
pub unsafe extern "C" fn sofuu_sse_new() -> *mut std::os::raw::c_void {
    let p = Box::new(crate::http::sse::SseParser::new());
    Box::into_raw(p) as *mut std::os::raw::c_void
}

/// `sofuu_sse_free(p)` — drops a parser from sofuu_sse_new. NULL-tolerant.
#[no_mangle]
pub unsafe extern "C" fn sofuu_sse_free(p: *mut std::os::raw::c_void) {
    if !p.is_null() {
        drop(Box::from_raw(p as *mut crate::http::sse::SseParser));
    }
}

/// `sofuu_sse_feed(p, chunk, len)` → malloc'd NUL-terminated JSON array of
/// complete events: `[{"event":"message","data":"..."}]`, even when empty.
/// NULL only on OOM-equivalent failure. Chunks are lossy-decoded (SSE text
/// may arrive split mid-codepoint across network reads).
#[no_mangle]
pub unsafe extern "C" fn sofuu_sse_feed(
    p: *mut std::os::raw::c_void,
    chunk: *const c_char,
    len: usize,
) -> *mut c_char {
    if p.is_null() || (chunk.is_null() && len > 0) {
        return std::ptr::null_mut();
    }
    let parser = &mut *(p as *mut crate::http::sse::SseParser);
    let bytes = std::slice::from_raw_parts(chunk as *const u8, len);
    let text = String::from_utf8_lossy(bytes);
    let events = parser.feed(&text);

    // Serialize compactly with serde_json (correct escaping for any bytes).
    let arr: Vec<serde_json::Value> = events
        .iter()
        .map(|e| serde_json::json!({ "event": e.event, "data": e.data }))
        .collect();
    let out = serde_json::to_string(&arr).unwrap_or_else(|_| "[]".into());
    let b = out.as_bytes();
    let buf = libc::malloc(b.len() + 1) as *mut c_char;
    if buf.is_null() {
        return std::ptr::null_mut();
    }
    std::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, b.len());
    *buf.add(b.len()) = 0;
    buf
}

/// `sofuu_npm_sha1_file_rs(path, out_hex)` — out_hex is a 41-byte buffer the
/// C caller provides. Returns 0 on success (same convention as the retired
/// C `sha1_file_hex`). Rust core: `npm::Sha1`.
#[no_mangle]
pub unsafe extern "C" fn sofuu_npm_sha1_file_rs(
    path: *const c_char,
    out_hex: *mut c_char,
) -> std::os::raw::c_int {
    if path.is_null() || out_hex.is_null() {
        return -1;
    }
    let cpath = match std::ffi::CStr::from_ptr(path).to_str() {
        Ok(p) => p,
        Err(_) => return -1,
    };
    let hex = match crate::npm::Sha1::file_hex(std::path::Path::new(cpath)) {
        Some(h) => h,
        None => return -1,
    };
    std::ptr::copy_nonoverlapping(hex.as_ptr(), out_hex as *mut u8, 40);
    *out_hex.add(40) = 0;
    0
}

/// `sofuu_npm_extract_safe_rs(tgz_path, dest_dir, strip_components, err, cap)`
/// → 0 ok / -1 with a NUL-terminated reason in `err_out`. Pure-Rust
/// safe extractor (rejects traversal/absolute paths/symlinks, no `tar` exec).
#[no_mangle]
pub unsafe extern "C" fn sofuu_npm_extract_safe_rs(
    tgz_path: *const c_char,
    dest_dir: *const c_char,
    strip_components: usize,
    err_out: *mut c_char,
    err_cap: usize,
) -> std::os::raw::c_int {
    let fail = |msg: &str| -> std::os::raw::c_int {
        if !err_out.is_null() && err_cap > 0 {
            let b = msg.as_bytes();
            let n = b.len().min(err_cap - 1);
            std::ptr::copy_nonoverlapping(b.as_ptr(), err_out as *mut u8, n);
            *err_out.add(n) = 0;
        }
        -1
    };
    if tgz_path.is_null() || dest_dir.is_null() {
        return fail("null path");
    }
    let tgz = match std::ffi::CStr::from_ptr(tgz_path).to_str() {
        Ok(p) => p,
        Err(_) => return fail("bad tgz path"),
    };
    let dest = match std::ffi::CStr::from_ptr(dest_dir).to_str() {
        Ok(p) => p,
        Err(_) => return fail("bad dest path"),
    };
    let data = match std::fs::read(tgz) {
        Ok(d) => d,
        Err(e) => return fail(&format!("read tgz: {e}")),
    };
    match crate::npm::extract_tarball_safe(&data, std::path::Path::new(dest), strip_components) {
        Ok(_) => 0,
        Err(e) => fail(&e),
    }
}

/// `sofuu_npm_resolve_rs(start_dir, module_name)` → malloc'd absolute path to
/// the resolved module file, or NULL. Rust twin of the C npm_resolve walk-up
/// (node_modules climb + package.json "main" + index.js/mjs).
#[no_mangle]
pub unsafe extern "C" fn sofuu_npm_resolve_rs(
    start_dir: *const c_char,
    module_name: *const c_char,
) -> *mut c_char {
    let (Some(dir), Some(name)) = (cstr_or_null(start_dir), cstr_or_null(module_name)) else {
        return std::ptr::null_mut();
    };
    match crate::npm::npm_resolve(std::path::Path::new(dir), name) {
        Some(p) => malloc_cstr(&p.to_string_lossy()),
        None => std::ptr::null_mut(),
    }
}

/// `sofuu_ts_strip_rs(src, len, &out_len)` → malloc'd NUL-terminated stripped
/// source, or NULL (non-UTF-8 input / OOM — C keeps the original text).
/// Contract-identical to the retired C `ts_strip()`.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ts_strip_rs(
    src: *const c_char,
    len: usize,
    out_len: *mut usize,
) -> *mut c_char {
    if src.is_null() {
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(src as *const u8, len);
    let text = match std::str::from_utf8(bytes) {
        Ok(t) => t,
        Err(_) => return std::ptr::null_mut(),
    };
    let stripped = crate::ts::strip(text);
    let b = stripped.as_bytes();
    let buf = libc::malloc(b.len() + 1) as *mut c_char;
    if buf.is_null() {
        return std::ptr::null_mut();
    }
    std::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, b.len());
    *buf.add(b.len()) = 0;
    if !out_len.is_null() {
        *out_len = b.len();
    }
    buf
}

/* ── CMA (cognitive memory) ────────────────────────────────────────
 * The C shell (mod_memory.c) holds an opaque `Cma*` per JS handle. All
 * state lives in the Rust struct; the QTSQ brain file (vectors + metadata
 * JSON) is read/written by the C adapter, hydrated/dumped through here. */

#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_new(vec_dim: usize) -> *mut std::os::raw::c_void {
    let cma = Box::new(crate::memory::cma::Cma::new(vec_dim));
    Box::into_raw(cma) as *mut std::os::raw::c_void
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_free(p: *mut std::os::raw::c_void) {
    if !p.is_null() {
        drop(Box::from_raw(p as *mut crate::memory::cma::Cma));
    }
}

/// `sofuu_cma_hydrate(p, vecs, n, dim, meta_json)` → 1 on success, 0 on a
/// malformed payload (the C caller then starts fresh). Copies the vectors.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_hydrate(
    p: *mut std::os::raw::c_void,
    vecs: *const f32,
    n: usize,
    dim: usize,
    meta_json: *const c_char,
) -> std::os::raw::c_int {
    if p.is_null() || vecs.is_null() || meta_json.is_null() {
        return 0;
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    let meta = match std::ffi::CStr::from_ptr(meta_json).to_str() {
        Ok(m) => m,
        Err(_) => return 0,
    };
    let slice = std::slice::from_raw_parts(vecs, n * dim);
    if cma.hydrate(slice, n, dim, meta) {
        1
    } else {
        0
    }
}

/// `sofuu_cma_remember(p, vec, text, role, kv_page_id)` → record id (≥0).
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_remember(
    p: *mut std::os::raw::c_void,
    vec: *const f32,
    text: *const c_char,
    role: *const c_char,
    kv_page_id: u32,
) -> std::os::raw::c_int {
    if p.is_null() || vec.is_null() || text.is_null() {
        return -1;
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    let text = match std::ffi::CStr::from_ptr(text).to_str() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    let role = if role.is_null() {
        "unknown"
    } else {
        match std::ffi::CStr::from_ptr(role).to_str() {
            Ok(r) => r,
            Err(_) => return -1,
        }
    };
    let v = std::slice::from_raw_parts(vec, cma.vec_dim);
    cma.remember(v, text, role, kv_page_id)
}

/// `sofuu_cma_recall(p, query, top_k, out_buf, out_cap)` — writes the recall
/// JSON array into out_buf (malloc'd by the C caller, cap-bounded) and returns
/// the JSON string length (or -1 on error/truncation). Recall JSON:
/// `[{"id":..,"distance":..,"score":..,"role":"..","text":"..","tier":..,"strength":..,"entity":null|".."}]`.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_recall(
    p: *mut std::os::raw::c_void,
    query: *const f32,
    top_k: usize,
    out_buf: *mut c_char,
    out_cap: usize,
) -> std::os::raw::c_int {
    if p.is_null() || query.is_null() || out_buf.is_null() || out_cap == 0 {
        return -1;
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    let q = std::slice::from_raw_parts(query, cma.vec_dim);
    let hits = cma.recall(q, top_k);
    let arr: Vec<serde_json::Value> = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "id": h.id,
                "distance": h.distance,
                "score": h.score,
                "role": h.role,
                "text": h.text,
                "tier": h.tier,
                "strength": h.strength,
                "entity": h.entity,
            })
        })
        .collect();
    let json = serde_json::to_string(&arr).unwrap_or_else(|_| "[]".into());
    let b = json.as_bytes();
    if b.len() + 1 > out_cap {
        return -1; // caller buffer too small — truncation is a hard error
    }
    std::ptr::copy_nonoverlapping(b.as_ptr(), out_buf as *mut u8, b.len());
    *out_buf.add(b.len()) = 0;
    b.len() as std::os::raw::c_int
}

/// `sofuu_cma_kv_hints(p, query, n, out_buf, out_cap)` — JSON array of the
/// distinct kv_page_ids of the top-n nearest memories. Returns length or -1.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_kv_hints(
    p: *mut std::os::raw::c_void,
    query: *const f32,
    n: usize,
    out_buf: *mut c_char,
    out_cap: usize,
) -> std::os::raw::c_int {
    if p.is_null() || query.is_null() || out_buf.is_null() || out_cap == 0 {
        return -1;
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    let q = std::slice::from_raw_parts(query, cma.vec_dim);
    let ids = cma.kv_hints(q, n);
    let json = serde_json::to_string(&ids).unwrap_or_else(|_| "[]".into());
    let b = json.as_bytes();
    if b.len() + 1 > out_cap {
        return -1;
    }
    std::ptr::copy_nonoverlapping(b.as_ptr(), out_buf as *mut u8, b.len());
    *out_buf.add(b.len()) = 0;
    b.len() as std::os::raw::c_int
}

/// `sofuu_cma_entities(p, out_buf, out_cap)` — JSON object of entity-name →
/// {type, text, id} for all TIER_ENTITY records. Returns length or -1.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_entities(
    p: *mut std::os::raw::c_void,
    out_buf: *mut c_char,
    out_cap: usize,
) -> std::os::raw::c_int {
    if p.is_null() || out_buf.is_null() || out_cap == 0 {
        return -1;
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    let mut map = serde_json::Map::new();
    for r in &cma.records {
        if r.tier == crate::memory::cma::TIER_ENTITY {
            if let Some(name) = &r.entity_name {
                map.insert(
                    name.clone(),
                    serde_json::json!({
                        "type": r.entity_type.clone().unwrap_or_default(),
                        "text": r.text,
                        "id": r.id,
                    }),
                );
            }
        }
    }
    let json = serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_else(|_| "{}".into());
    let b = json.as_bytes();
    if b.len() + 1 > out_cap {
        return -1;
    }
    std::ptr::copy_nonoverlapping(b.as_ptr(), out_buf as *mut u8, b.len());
    *out_buf.add(b.len()) = 0;
    b.len() as std::os::raw::c_int
}

/// `sofuu_cma_remember_entity(p, vec, text, name, etype)` → record id (≥0).
/// Upserts by name — an existing entity with the same name is updated
/// in place and the same id returned (matches the old C semantics).
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_remember_entity(
    p: *mut std::os::raw::c_void,
    vec: *const f32,
    text: *const c_char,
    name: *const c_char,
    etype: *const c_char,
) -> std::os::raw::c_int {
    if p.is_null() || vec.is_null() || text.is_null() || name.is_null() {
        return -1;
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    let text = match std::ffi::CStr::from_ptr(text).to_str() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    let name = match std::ffi::CStr::from_ptr(name).to_str() {
        Ok(n) => n,
        Err(_) => return -1,
    };
    let etype = if etype.is_null() {
        "unknown"
    } else {
        match std::ffi::CStr::from_ptr(etype).to_str() {
            Ok(e) => e,
            Err(_) => return -1,
        }
    };
    let v = std::slice::from_raw_parts(vec, cma.vec_dim);
    cma.remember_entity(v, text, name, etype)
}

/// `sofuu_cma_mark_positive(p, ids, n)` → count marked.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_mark_positive(
    p: *mut std::os::raw::c_void,
    ids: *const u32,
    n: usize,
) -> u32 {
    if p.is_null() || (ids.is_null() && n > 0) {
        return 0;
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    let slice = if n > 0 {
        std::slice::from_raw_parts(ids, n)
    } else {
        &[]
    };
    cma.mark_positive(slice)
}

/// `sofuu_cma_forget(p, id)` → 1 on success.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_forget(
    p: *mut std::os::raw::c_void,
    id: u32,
) -> std::os::raw::c_int {
    if p.is_null() {
        return 0;
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    if cma.forget(id) {
        1
    } else {
        0
    }
}

/// `sofuu_cma_decay(p, dt_seconds)`.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_decay(p: *mut std::os::raw::c_void, dt: u32) {
    if !p.is_null() {
        let cma = &mut *(p as *mut crate::memory::cma::Cma);
        cma.decay_tick(dt);
    }
}

/// `sofuu_cma_consolidate(p)` → number of clusters consolidated.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_consolidate(
    p: *mut std::os::raw::c_void,
) -> std::os::raw::c_int {
    if p.is_null() {
        return 0;
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    cma.consolidate() as std::os::raw::c_int
}

/// `sofuu_cma_count(p)` → number of memories.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_count(p: *mut std::os::raw::c_void) -> u32 {
    if p.is_null() {
        return 0;
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    cma.len() as u32
}

/// `sofuu_cma_records_json(p)` → malloc'd metadata JSON (for the brain-file
/// flush path). NULL on null input.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_records_json(p: *mut std::os::raw::c_void) -> *mut c_char {
    if p.is_null() {
        return std::ptr::null_mut();
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    malloc_cstr(&cma.records_json())
}

/// `sofuu_cma_vectors(p, out_n, out_dim)` → malloc'd flat f32 vector table
/// (caller frees), or NULL. For the brain-file flush path.
#[no_mangle]
pub unsafe extern "C" fn sofuu_cma_vectors(
    p: *mut std::os::raw::c_void,
    out_n: *mut usize,
    out_dim: *mut usize,
) -> *mut f32 {
    if p.is_null() || out_n.is_null() || out_dim.is_null() {
        return std::ptr::null_mut();
    }
    let cma = &mut *(p as *mut crate::memory::cma::Cma);
    let n = cma.vectors.len();
    let dim = cma.vec_dim;
    if n == 0 {
        *out_n = 0;
        *out_dim = dim;
        return std::ptr::null_mut();
    }
    let total = n * dim;
    let buf = libc::malloc(total * std::mem::size_of::<f32>()) as *mut f32;
    if buf.is_null() {
        return std::ptr::null_mut();
    }
    for (i, v) in cma.vectors.iter().enumerate() {
        std::ptr::copy_nonoverlapping(v.as_ptr(), buf.add(i * dim), dim);
    }
    *out_n = n;
    *out_dim = dim;
    buf
}
