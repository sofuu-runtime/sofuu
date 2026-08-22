// rt/http_sse.rs — the JS `sofuu.SSEParser` shell (PLAN-RUST-MIGRATION M4).
//
// Port of the deleted `src/http/sse.c` (211 lines). The parser itself lives
// in Rust (crates/sofuu-core/src/http/sse.rs, exported via ffi_exports.rs
// as sofuu_sse_new/free/feed); this module owns the JS plumbing: the
// SSEParser class, the feed() bridge and the onMessage event dispatch.
//
// C symbols replaced: `mod_http_sse_register` (engine.c calls it unchanged).

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

extern "C" {
    // Rust SSE parser (ffi_exports.rs) — malloc'd opaque state.
    fn sofuu_sse_new() -> *mut c_void;
    fn sofuu_sse_free(p: *mut c_void);
    fn sofuu_sse_feed(p: *mut c_void, chunk: *const c_char, len: usize) -> *mut c_char;
}

static SSE_CLASS_ID: AtomicU32 = AtomicU32::new(0);

struct SseState {
    ctx: *mut JSContext,
    parser: *mut c_void,
    this_val: JSValue, /* non-dup'd GC-visible self ref — as before */
}

unsafe extern "C" fn sofuu_sse_finalizer(_rt: *mut qjs::JSRuntime, val: JSValue) {
    let sse = qjs::JS_GetOpaque(val, SSE_CLASS_ID.load(Ordering::Relaxed));
    if sse.is_null() {
        return;
    }
    let sse = sse as *mut SseState;
    if !(*sse).parser.is_null() {
        sofuu_sse_free((*sse).parser);
    }
    drop(Box::from_raw(sse));
}

unsafe extern "C" fn js_sse_constructor(
    ctx: *mut JSContext,
    _new_target: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let obj = qjs::JS_NewObjectClass(ctx, SSE_CLASS_ID.load(Ordering::Relaxed) as c_int);
    if qjs::is_exception(obj) {
        return obj;
    }

    let sse = Box::new(SseState {
        ctx,
        parser: sofuu_sse_new(),
        this_val: obj, /* non-dup'd GC-visible self ref — as before */
    });
    let sse_ptr = Box::into_raw(sse);

    qjs::JS_SetOpaque(obj, sse_ptr as *mut c_void);
    obj
}

/// Emits one parsed event to JS. NULL-safe: a NULL event name or NULL data
/// must never reach JS_NewString/JS_Call (SIGSEGV via strlen(NULL)).
unsafe fn emit_event(sse: *mut SseState, event_name: *const c_char, data: JSValueConst) {
    let on_message = qjs::sofuu_js_get_property_str((*sse).ctx, (*sse).this_val, c"onMessage".as_ptr());
    if qjs::JS_IsFunction((*sse).ctx, on_message) != 0 {
        /* The event name may legitimately be NULL when a feed produced an
         * event without a name — default to "message" instead of crashing. */
        let name_ptr = if event_name.is_null() { c"message".as_ptr() } else { event_name };
        let data_val = if qjs::is_undefined(data) || qjs::is_null(data) {
            qjs::sofuu_js_new_string((*sse).ctx, c"".as_ptr())
        } else {
            qjs::sofuu_js_dup_value((*sse).ctx, data)
        };
        let mut args: [JSValueConst; 2] = [
            qjs::sofuu_js_new_string((*sse).ctx, name_ptr),
            data_val,
        ];
        let ret = qjs::JS_Call((*sse).ctx, on_message, (*sse).this_val, 2, args.as_mut_ptr());
        if qjs::is_exception(ret) {
            qjs::js_std_dump_error((*sse).ctx);
        }
        qjs::sofuu_js_free_value((*sse).ctx, ret);
        qjs::sofuu_js_free_value((*sse).ctx, args[0]);
        qjs::sofuu_js_free_value((*sse).ctx, args[1]);
    }
    qjs::sofuu_js_free_value((*sse).ctx, on_message);
}

unsafe extern "C" fn js_sse_feed(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let sse = qjs::JS_GetOpaque(this_val, SSE_CLASS_ID.load(Ordering::Relaxed));
    if sse.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid SSEParser".as_ptr());
    }
    let sse = sse as *mut SseState;
    if (*sse).parser.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid SSEParser".as_ptr());
    }
    if argc < 1 {
        return qjs::sofuu_js_undefined();
    }

    let mut chunk_len: usize = 0;
    let chunk = qjs::JS_ToCStringLen2(ctx, &mut chunk_len, *argv, 0);
    if chunk.is_null() {
        return qjs::sofuu_js_exception();
    }

    let json = sofuu_sse_feed((*sse).parser, chunk, chunk_len);
    qjs::sofuu_js_free_cstring(ctx, chunk);
    if json.is_null() {
        return qjs::JS_ThrowOutOfMemory(ctx);
    }
    let json_c = CStr::from_ptr(json);
    let json_bytes = json_c.to_bytes();

    let arr = qjs::JS_ParseJSON(ctx, json, json_bytes.len(), c"<sse_rs>".as_ptr());
    libc::free(json as *mut c_void);
    if qjs::is_exception(arr) {
        qjs::sofuu_js_get_exception(ctx);
        return qjs::sofuu_js_undefined();
    }

    let mut n: u32 = 0;
    let lenv = qjs::sofuu_js_get_property_str(ctx, arr, c"length".as_ptr());
    qjs::sofuu_js_to_uint32(ctx, &mut n, lenv);
    qjs::sofuu_js_free_value(ctx, lenv);

    for i in 0..n {
        let ev = qjs::JS_GetPropertyUint32(ctx, arr, i);
        let evname = qjs::sofuu_js_get_property_str(ctx, ev, c"event".as_ptr());
        let data = qjs::sofuu_js_get_property_str(ctx, ev, c"data".as_ptr());
        let en_ptr = qjs::sofuu_js_to_cstring(ctx, evname);
        /* NULL-safe: a missing `event` field (or one that fails to
         * stringify) must NOT crash the feed — default to "message".
         * This was a real SIGSEGV: JS_ToCString on undefined returns NULL
         * and JS_NewString(NULL) → strlen(NULL). */
        emit_event(sse, if en_ptr.is_null() { c"message".as_ptr() } else { en_ptr }, data);
        if !en_ptr.is_null() {
            qjs::sofuu_js_free_cstring(ctx, en_ptr);
        }
        qjs::sofuu_js_free_value(ctx, data);
        qjs::sofuu_js_free_value(ctx, evname);
        qjs::sofuu_js_free_value(ctx, ev);
    }
    qjs::sofuu_js_free_value(ctx, arr);
    qjs::sofuu_js_undefined()
}

thread_local! {
    // JS_CFUNC_DEF("feed", 1, js_sse_feed).
    static SSE_PROTO_FUNCS: [qjs::JSCFunctionListEntry; 1] = [qjs::JSCFunctionListEntry {
        name: c"feed".as_ptr(),
        prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
        def_type: qjs::JS_DEF_CFUNC,
        magic: 0,
        u: qjs::JSCFunctionListEntryFunc {
            length: 1,
            cproto: 0, /* JS_CFUNC_generic */
            _pad: [0; 6],
            cfunc: js_sse_feed,
        },
    }];
}

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_http_sse_register(ctx: *mut JSContext) {
    let mut class_id: qjs::JSClassID = 0;
    qjs::JS_NewClassID(&mut class_id);
    SSE_CLASS_ID.store(class_id, Ordering::Relaxed);
    let class_def = qjs::JSClassDef {
        class_name: c"SSEParser".as_ptr(),
        finalizer: Some(sofuu_sse_finalizer),
        gc_mark: ptr::null_mut(),
        call: ptr::null_mut(),
        exotic: ptr::null_mut(),
    };
    qjs::JS_NewClass(qjs::JS_GetRuntime(ctx), class_id, &class_def);

    let proto = qjs::sofuu_js_new_object(ctx);
    SSE_PROTO_FUNCS.with(|f| qjs::JS_SetPropertyFunctionList(ctx, proto, f.as_ptr(), 1));
    qjs::JS_SetClassProto(ctx, class_id, proto);

    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());

    if qjs::is_undefined(sofuu) {
        sofuu = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), qjs::sofuu_js_dup_value(ctx, sofuu));
    }

    let constructor = qjs::JS_NewCFunction2(
        ctx,
        js_sse_constructor,
        c"SSEParser".as_ptr(),
        0,
        qjs::JS_CFUNC_constructor,
        0,
    );

    qjs::sofuu_js_set_property_str(ctx, sofuu, c"SSEParser".as_ptr(), constructor);

    qjs::sofuu_js_free_value(ctx, sofuu);
    qjs::sofuu_js_free_value(ctx, global);
}
