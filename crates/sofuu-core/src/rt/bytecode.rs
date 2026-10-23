// rt/bytecode.rs - AOT bytecode snapshots for user-supplied agent scripts.
//
// PLAN-AUTONOMOUS-AGENT-LOOP-AND-DRIVERS Phase 6. sofuu.agent.loadDir used
// to re-parse and re-compile every script in ~/.sofuu/agents on every start.
// A script that has not changed should be compiled once:
//
//   sofuu.aotEval(jsPath, bcPath)
//     -> { status: 'hit' | 'compiled' | 'threw' | 'unavailable',
//          bytes, hash, error }
//
//   hit          bcPath held bytecode for exactly this source; it ran.
//   compiled     no usable snapshot; the source was compiled, bcPath was
//                written (header + QuickJS bytecode) and the result ran.
//   threw        the bytecode ran and threw. The source is NOT re-run here:
//                a script with side effects must never execute twice.
//   unavailable  the source could not be read or compiled (a syntax error,
//                typically). The caller must eval the source itself so the
//                real error surfaces - this layer never invents a message.
//
// Staleness. The snapshot is keyed by a 32-bit FNV-1a hash of the source
// bytes, stored in a fixed 17-byte header:
//
//   "SOFUBC1 " + 8 hex chars + 0x0a + <QuickJS bytecode>
//
// The hash is checked on every load, so a stale snapshot can never win. The
// shipped drivers in shipped.rs have no such runtime guard; they are checked
// at build time instead (scripts/check-driver-bc.sh), where a mismatch is a
// build failure rather than a silent wrong-code load.
//
// Why a host binding rather than shelling out to "sofuu compile-bc": a child
// process needs the binary's own path, a temp file and an IPC hop, and it
// still cannot hand the bytecode back for evaluation in this context.

use std::ffi::{c_char, c_int, CString};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

const MAGIC: &[u8] = b"SOFUBC1 ";
const HASH_LEN: usize = 8;
const HEADER_LEN: usize = MAGIC.len() + HASH_LEN + 1;

/// FNV-1a over the source bytes (32-bit; the header carries 8 hex chars).
fn hash32(data: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in data {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn hex8(h: u32) -> String {
    format!("{:08x}", h)
}

/// Read a JS value as a Rust string (empty when it is not convertible).
unsafe fn js_string(ctx: *mut JSContext, v: JSValueConst) -> String {
    let p = unsafe { qjs::sofuu_js_to_cstring(ctx, v) };
    if p.is_null() {
        return String::new();
    }
    let s = unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned();
    unsafe { qjs::sofuu_js_free_cstring(ctx, p) };
    s
}

/// Set a string property, skipping values a C string cannot carry.
unsafe fn set_str(ctx: *mut JSContext, obj: JSValueConst, key: *const c_char, val: &str) {
    if let Ok(c_val) = CString::new(val) {
        let v = unsafe { qjs::sofuu_js_new_string(ctx, c_val.as_ptr()) };
        unsafe { qjs::sofuu_js_set_property_str(ctx, obj, key, v) };
    }
}

unsafe fn set_i32(ctx: *mut JSContext, obj: JSValueConst, key: *const c_char, val: i32) {
    let v = unsafe { qjs::sofuu_js_new_int32(ctx, val) };
    unsafe { qjs::sofuu_js_set_property_str(ctx, obj, key, v) };
}

enum Ran {
    Ok,
    ReadFailed,
    Threw(String),
}

/// Run one bytecode blob in global scope. Without JS_READ_OBJ_ROM_DATA the
/// reader copies, so a blob that came from a file (or a JS buffer) needs no
/// lifetime contract with the resulting function object.
unsafe fn run_bytecode(ctx: *mut JSContext, bc: &[u8]) -> Ran {
    if bc.is_empty() {
        return Ran::ReadFailed;
    }
    let fun = unsafe { qjs::JS_ReadObject(ctx, bc.as_ptr(), bc.len(), qjs::JS_READ_OBJ_BYTECODE) };
    if unsafe { qjs::is_exception(fun) } {
        let exc = unsafe { qjs::JS_GetException(ctx) };
        unsafe { qjs::sofuu_js_free_value(ctx, exc) };
        unsafe { qjs::sofuu_js_free_value(ctx, fun) };
        return Ran::ReadFailed;
    }
    // JS_EvalFunction takes ownership of fun.
    let res = unsafe { qjs::JS_EvalFunction(ctx, fun) };
    if unsafe { qjs::is_exception(res) } {
        let exc = unsafe { qjs::JS_GetException(ctx) };
        let msg = unsafe { js_string(ctx, exc) };
        unsafe { qjs::sofuu_js_free_value(ctx, exc) };
        unsafe { qjs::sofuu_js_free_value(ctx, res) };
        return Ran::Threw(msg);
    }
    unsafe { qjs::sofuu_js_free_value(ctx, res) };
    Ran::Ok
}

unsafe fn reply(
    ctx: *mut JSContext,
    status: &str,
    bytes: usize,
    hash: &str,
    error: &str,
) -> JSValue {
    let out = unsafe { qjs::sofuu_js_new_object(ctx) };
    unsafe {
        set_str(ctx, out, c"status".as_ptr(), status);
        set_str(ctx, out, c"hash".as_ptr(), hash);
        set_i32(ctx, out, c"bytes".as_ptr(), bytes as i32);
    }
    if !error.is_empty() {
        unsafe { set_str(ctx, out, c"error".as_ptr(), error) };
    }
    out
}

/// sofuu.aotEval(jsPath, bcPath) - see the module header.
unsafe extern "C" fn js_aot_eval(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 || argv.is_null() {
        return unsafe {
            reply(
                ctx,
                "unavailable",
                0,
                "",
                "sofuu.aotEval(jsPath, bcPath) needs two paths",
            )
        };
    }
    let js_path = unsafe { js_string(ctx, *argv) };
    let bc_path = unsafe { js_string(ctx, *argv.add(1)) };

    let src = match std::fs::read(&js_path) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("cannot read {js_path}: {e}");
            return unsafe { reply(ctx, "unavailable", 0, "", &msg) };
        }
    };
    let hash = hex8(hash32(&src));

    // 1. A snapshot whose hash matches this exact source wins.
    if let Ok(blob) = std::fs::read(&bc_path) {
        let magic_ok = blob.len() > HEADER_LEN && &blob[..MAGIC.len()] == MAGIC;
        let hash_ok = magic_ok
            && &blob[MAGIC.len()..MAGIC.len() + HASH_LEN] == hash.as_bytes();
        if hash_ok {
            let body = &blob[HEADER_LEN..];
            match unsafe { run_bytecode(ctx, body) } {
                Ran::Ok => return unsafe { reply(ctx, "hit", body.len(), &hash, "") },
                Ran::Threw(m) => {
                    return unsafe { reply(ctx, "threw", body.len(), &hash, &m) }
                }
                // Corrupt snapshot: fall through and recompile.
                Ran::ReadFailed => {}
            }
        }
    }

    // 2. Compile from source and write the snapshot.
    let text = match std::str::from_utf8(&src) {
        Ok(t) => t,
        Err(_) => {
            return unsafe {
                reply(ctx, "unavailable", 0, &hash, "source is not valid UTF-8")
            }
        }
    };
    let compiled = unsafe { crate::shipped::compile_to_bytecode(ctx, text, &js_path) };
    let Some(bc) = compiled else {
        return unsafe {
            reply(ctx, "unavailable", 0, &hash, "compile failed (eval the source for the error)")
        };
    };
    let mut blob = Vec::with_capacity(HEADER_LEN + bc.len());
    blob.extend_from_slice(MAGIC);
    blob.extend_from_slice(hash.as_bytes());
    blob.push(10u8);
    blob.extend_from_slice(&bc);
    let _ = std::fs::write(&bc_path, &blob);

    match unsafe { run_bytecode(ctx, &bc) } {
        Ran::Ok => unsafe { reply(ctx, "compiled", bc.len(), &hash, "") },
        Ran::Threw(m) => unsafe { reply(ctx, "threw", bc.len(), &hash, &m) },
        Ran::ReadFailed => unsafe {
            reply(ctx, "unavailable", bc.len(), &hash, "bytecode read failed")
        },
    }
}

// -- Module Registration ------------------------------------------

/// Install sofuu.aotEval on the engine's sofuu namespace.
///
/// # Safety
/// ctx must be a live QuickJS context on the engine thread.
pub unsafe fn mod_bytecode_register(ctx: *mut JSContext) {
    let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
    let mut sofuu_obj = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr()) };
    if unsafe { qjs::is_undefined(sofuu_obj) } {
        sofuu_obj = unsafe { qjs::sofuu_js_new_object(ctx) };
        unsafe {
            qjs::sofuu_js_set_property_str(
                ctx,
                global,
                c"sofuu".as_ptr(),
                qjs::sofuu_js_dup_value(ctx, sofuu_obj),
            )
        };
    }

    let f = unsafe { qjs::sofuu_js_new_cfunction(ctx, js_aot_eval, c"aotEval".as_ptr(), 2) };
    unsafe { qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"aotEval".as_ptr(), f) };

    unsafe { qjs::sofuu_js_free_value(ctx, sofuu_obj) };
    unsafe { qjs::sofuu_js_free_value(ctx, global) };
}
