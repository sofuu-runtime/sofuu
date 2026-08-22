// sofuu-ffi — QuickJS bindings (M0 keystone, PLAN-RUST-MIGRATION.md).
//
// The migration end-state inverts the arrows: Rust calls QuickJS directly
// over FFI; C glue dissolves. This module is the hand-written binding layer
// for the ~95 QuickJS entry points the runtime uses.
//
// ABI notes (deps/quickjs/quickjs.h):
//   - JS_PTR64 is defined, so JS_NAN_BOXING is NOT: JSValue is a 16-byte
//     struct { JSValueUnion u; int64_t tag; }, passed by value.
//   - Many QuickJS functions are `static inline` and have no linkable
//     symbol; those go through src/ffi_shim.c (sofuu_js_* wrappers).
//   - Everything else is a real C export and declared directly here.
//
// GC discipline (the M6-crash fix, applied from day one):
//   - `Rooted<T>` dups a JSValue on creation and frees it on drop, so a
//     value held across an FFI boundary / uv callback stays alive.
//   - `CtxPtr` is `!Send` (PhantomData<*const ()>): the compiler forbids
//     touching a QuickJS context from another thread.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

pub type JSContext = c_void;
pub type JSRuntime = c_void;
pub type JSClassID = u32;

/// The C `JSValueUnion` — 8 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub union JSValueUnion {
    pub int32: i32,
    pub float64: f64,
    pub ptr: *mut c_void,
}

/// The C `JSValue` — 16 bytes, passed by value (non-NaN-boxing build).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JSValue {
    pub u: JSValueUnion,
    pub tag: i64,
}

/// `JSValueConst` is `#define JSValueConst JSValue` — same type.
pub type JSValueConst = JSValue;

/// The C `JSClassDef` (we only set `.finalizer` — same shape as C).
#[repr(C)]
pub struct JSClassDef {
    pub class_name: *const c_char,
    pub finalizer: Option<unsafe extern "C" fn(*mut JSRuntime, JSValue)>,
    pub gc_mark: *mut c_void,
    pub call: *mut c_void,
    pub exotic: *mut c_void,
}

/// JSCFunction: JSValue (*)(JSContext*, JSValueConst, int, JSValueConst*)
pub type JSCFunction = unsafe extern "C" fn(
    *mut JSContext,
    JSValueConst,
    c_int,
    *const JSValueConst,
) -> JSValue;

/// The C `JSCFunctionData` FUNCTION POINTER type — the callback shape that
/// `JS_NewCFunctionData` installs: (ctx, this_val, argc, argv, magic,
/// func_data) -> JSValue. (The struct above carries the C name for the
/// property-list machinery; this alias avoids the collision.)
pub type JSCFunctionDataFn = unsafe extern "C" fn(
    *mut JSContext,
    JSValueConst,
    c_int,
    *const JSValueConst,
    c_int,
    *mut JSValue,
) -> JSValue;

/// The C `JSInterruptHandler`: return != 0 to abort evaluation.
pub type JSInterruptHandler = unsafe extern "C" fn(*mut JSRuntime, *mut c_void) -> c_int;

/// The C `JSHostPromiseRejectionTracker` callback (JS_SetHostPromiseRejectionTracker):
/// fired for every rejected promise — `is_handled` is 0 at rejection time,
/// 1 when a handler is attached later.
pub type JSHostPromiseRejectionTracker = unsafe extern "C" fn(
    *mut JSContext,
    JSValueConst,
    JSValueConst,
    c_int,
    *mut c_void,
);

/// The C `JSModuleNormalizeFunc` (JS_SetModuleLoaderFunc): returns a
/// malloc'd normalized module name or NULL. The engine passes NULL.
pub type JsModuleNormalizeFunc = unsafe extern "C" fn(
    *mut JSContext,
    *const c_char,
    *mut c_void,
) -> *mut c_char;

/// The C `JSModuleLoaderFunc` (JS_SetModuleLoaderFunc): resolves + compiles
/// an ESM module; returns the JSModuleDef* (opaque) or NULL on error.
pub type JsModuleLoaderFunc = unsafe extern "C" fn(
    *mut JSContext,
    *const c_char,
    *mut c_void,
) -> *mut c_void;

// ── M2: console.table + fs/subprocess helpers ────────────────────────

/// `typedef uint32_t JSAtom` (quickjs.h:54).
pub type JSAtom = u32;

/// `JSPropertyEnum { JS_BOOL is_enumerable; JSAtom atom; }`
/// (quickjs.h:437-440) — 8 bytes on all supported targets.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JSPropertyEnum {
    pub is_enumerable: c_int,
    pub atom: JSAtom,
}

pub const JS_GPN_STRING_MASK: c_int = 1 << 0;
pub const JS_GPN_ENUM_ONLY: c_int = 1 << 4;
pub const JS_PROP_WRITABLE: u8 = 0x02;

/// JSCFunctionData (JS_NewCFunctionData) — the C struct is { length, length2,
/// magic, data[0..] } at the pointer we pass; QuickJS reads magic + data.
#[repr(C)]
pub struct JSCFunctionData {
    pub length: i16,
    pub length2: i16,
    pub magic: i16,
    pub data: [*const c_void; 1],
}

// ── M10: the ffi_shim.c bridges are now plain Rust ────────────────
// src/ffi_shim.c is DELETED. The static-inline QuickJS helpers are
// reimplemented here against the public quickjs.h semantics:
//   - tag checks: compare the `int64_t tag` field (JS_PTR64 build —
//     struct JSValue, no NaN boxing). Tags from quickjs.h:69-87.
//   - immediates: JS_MKVAL(tag, val) = { u.int32 = val, tag }.
//   - refcounted payloads: HAS_REF_COUNT iff tag < JS_TAG_FIRST;
//     the refcount lives at JSRefCountHeader { int ref_count; } at the
//     start of the payload (JS_VALUE_GET_PTR); freeing defers to the
//     real exports __JS_FreeValue(ctx, v) / __JS_FreeValueRT(rt, v).
//     JS_NewBool/NewInt32/NewInt64/NewFloat64 fall back to constructing
//     the 16-byte value directly (JS_MKVAL / __JS_NewFloat64 body).
// All of these keep the old names + signatures so every caller (qjs
// wrappers, sofuu-core modules, tests) links unchanged.

pub const JS_TAG_FIRST: i32 = -11;
pub const JS_TAG_STRING: i32 = -7;
pub const JS_TAG_OBJECT: i32 = -1;
pub const JS_TAG_INT: i32 = 0;
pub const JS_TAG_BOOL: i32 = 1;
pub const JS_TAG_NULL: i32 = 2;
pub const JS_TAG_UNDEFINED: i32 = 3;
pub const JS_TAG_EXCEPTION: i32 = 6;
pub const JS_TAG_FLOAT64: i32 = 7;

/// `JS_MKVAL(tag, val)` — an immediate (non-refcounted) JSValue.
#[inline]
pub fn js_mkval(tag: c_int, val: i32) -> JSValue {
    JSValue { u: JSValueUnion { int32: val }, tag: tag as i64 }
}

/// True when `v` carries a reference count (the payload starts with a
/// JSRefCountHeader). Equivalent to `(unsigned)tag >= (unsigned)JS_TAG_FIRST`.
#[inline]
fn has_ref_count(v: JSValueConst) -> bool {
    (v.tag as i32) < 0
}

/// `__JS_NewFloat64(ctx, d)` — the inline body (quickjs.h:225): a plain
/// float64-tagged JSValue; the context is only used for allocation in the
/// NAN-boxing builds, never here.
#[inline]
pub unsafe fn new_float64_value(d: f64) -> JSValue {
    JSValue { u: JSValueUnion { float64: d }, tag: JS_TAG_FLOAT64 as i64 }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_is_string(v: JSValueConst) -> c_int {
    ((v.tag as i32) == JS_TAG_STRING) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_is_exception(v: JSValueConst) -> c_int {
    ((v.tag as i32) == JS_TAG_EXCEPTION) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_is_null(v: JSValueConst) -> c_int {
    ((v.tag as i32) == JS_TAG_NULL) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_is_undefined(v: JSValueConst) -> c_int {
    ((v.tag as i32) == JS_TAG_UNDEFINED) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_is_object(v: JSValueConst) -> c_int {
    ((v.tag as i32) == JS_TAG_OBJECT) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_is_number(v: JSValueConst) -> c_int {
    let tag = v.tag as i32;
    (tag == JS_TAG_INT || (tag as u32) == JS_TAG_FLOAT64 as u32) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_free_value(ctx: *mut JSContext, v: JSValue) {
    if has_ref_count(v) {
        // SAFETY: v is refcounted — its payload starts with JSRefCountHeader.
        let p = v.u.ptr as *mut c_int;
        unsafe {
            *p -= 1;
            if *p <= 0 {
                __JS_FreeValue(ctx, v);
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_free_value_rt(rt: *mut JSRuntime, v: JSValue) {
    if has_ref_count(v) {
        // SAFETY: v is refcounted — its payload starts with JSRefCountHeader.
        let p = v.u.ptr as *mut c_int;
        unsafe {
            *p -= 1;
            if *p <= 0 {
                __JS_FreeValueRT(rt, v);
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_dup_value(_ctx: *mut JSContext, v: JSValueConst) -> JSValue {
    if has_ref_count(v) {
        // SAFETY: v is refcounted — its payload starts with JSRefCountHeader.
        let p = v.u.ptr as *mut c_int;
        unsafe { *p += 1 };
    }
    v
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_to_cstring(ctx: *mut JSContext, v: JSValueConst) -> *const c_char {
    // SAFETY: JS_ToCStringLen2 with a NULL length pointer == JS_ToCString.
    unsafe { JS_ToCStringLen2(ctx, ptr::null_mut(), v, 0) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_to_uint32(
    ctx: *mut JSContext,
    pres: *mut u32,
    val: JSValueConst,
) -> c_int {
    // JS_ToUint32 is the inline cast to JS_ToInt32 (quickjs.h:704).
    // SAFETY: pres aliases an int32_t* (same size/enumeration).
    unsafe { JS_ToInt32(ctx, pres as *mut i32, val) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_free_cstring(ctx: *mut JSContext, ptr: *const c_char) {
    // SAFETY: ptr must come from JS_ToCString* on this context.
    unsafe { JS_FreeCString(ctx, ptr) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_new_cfunction(
    ctx: *mut JSContext,
    func: JSCFunction,
    name: *const c_char,
    length: c_int,
) -> JSValue {
    // JS_NewCFunction == JS_NewCFunction2(.., JS_CFUNC_generic, 0).
    // SAFETY: func must be 'static; name valid.
    unsafe { JS_NewCFunction2(ctx, func, name, length, 0, 0) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_new_bool(_ctx: *mut JSContext, val: c_int) -> JSValue {
    js_mkval(JS_TAG_BOOL, (val != 0) as i32)
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_new_int32(_ctx: *mut JSContext, val: i32) -> JSValue {
    js_mkval(JS_TAG_INT, val)
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_new_int64(_ctx: *mut JSContext, val: i64) -> JSValue {
    let i = val as i32;
    if val == i as i64 {
        js_mkval(JS_TAG_INT, i)
    } else {
        new_float64_value(val as f64)
    }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_new_float64(_ctx: *mut JSContext, d: f64) -> JSValue {
    // JS_NewFloat64: an integral d exactly representable as int32 becomes
    // an int-tagged value (bit-exact union compare, quickjs.h:548).
    let i = d as i32;
    if d.to_bits() == (i as f64).to_bits() {
        js_mkval(JS_TAG_INT, i)
    } else {
        new_float64_value(d)
    }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_new_string(ctx: *mut JSContext, str_: *const c_char) -> JSValue {
    // SAFETY: str_ must be a NUL-terminated C string.
    unsafe { JS_NewString(ctx, str_) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_new_object(ctx: *mut JSContext) -> JSValue {
    // SAFETY: trivial.
    unsafe { JS_NewObject(ctx) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_get_global_object(ctx: *mut JSContext) -> JSValue {
    // SAFETY: trivial.
    unsafe { JS_GetGlobalObject(ctx) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_get_property_str(
    ctx: *mut JSContext,
    this_obj: JSValueConst,
    prop: *const c_char,
) -> JSValue {
    // SAFETY: prop must be a NUL-terminated C string.
    unsafe { JS_GetPropertyStr(ctx, this_obj, prop) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_set_property_str(
    ctx: *mut JSContext,
    this_obj: JSValueConst,
    prop: *const c_char,
    val: JSValue,
) -> c_int {
    // SAFETY: prop must be a NUL-terminated C string; val is owned by the
    // property on success (JS_SetPropertyStr steals the reference).
    unsafe { JS_SetPropertyStr(ctx, this_obj, prop, val) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_get_exception(ctx: *mut JSContext) -> JSValue {
    // SAFETY: trivial.
    unsafe { JS_GetException(ctx) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_new_uint32(_ctx: *mut JSContext, val: u32) -> JSValue {
    // JS_NewUint32: values fitting in int32 are int-tagged (quickjs.h:534).
    if val <= i32::MAX as u32 {
        js_mkval(JS_TAG_INT, val as i32)
    } else {
        new_float64_value(val as f64)
    }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_undefined() -> JSValue {
    js_mkval(JS_TAG_UNDEFINED, 0)
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_exception() -> JSValue {
    js_mkval(JS_TAG_EXCEPTION, 0)
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_null() -> JSValue {
    js_mkval(JS_TAG_NULL, 0)
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_value_get_ptr(v: JSValueConst) -> *mut c_void {
    v.u.ptr
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_js_get_property(
    ctx: *mut JSContext,
    this_obj: JSValueConst,
    prop: JSAtom,
) -> JSValue {
    // JS_GetProperty is the inline JS_GetPropertyInternal(.., receiver, 0).
    // SAFETY: ctx live; prop a valid atom of ctx.
    unsafe { JS_GetPropertyInternal(ctx, this_obj, prop, this_obj, 0) }
}

// ── Real exports used by the shims / engine (all linkable) ────────

extern "C" {
    pub fn JS_ToString(ctx: *mut JSContext, val: JSValueConst) -> JSValue;
    // quickjs.h:851 — real export (M1 rejection tracker install).
    pub fn JS_SetHostPromiseRejectionTracker(
        rt: *mut JSRuntime,
        cb: Option<JSHostPromiseRejectionTracker>,
        opaque: *mut c_void,
    );
    pub fn JS_CallConstructor(
        ctx: *mut JSContext,
        func_obj: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue;
    pub fn JS_GetOwnPropertyNames(
        ctx: *mut JSContext,
        ptab: *mut *mut JSPropertyEnum,
        plen: *mut u32,
        obj: JSValueConst,
        flags: c_int,
    ) -> c_int;
    pub fn JS_AtomToCString(ctx: *mut JSContext, atom: JSAtom) -> *const c_char;
    pub fn JS_FreeAtom(ctx: *mut JSContext, atom: JSAtom);
    pub fn js_free(ctx: *mut JSContext, ptr: *mut c_void);
    // ── M4: HTTP client (Response.body accessor + SSE) ──
    pub fn JS_NewAtom(ctx: *mut JSContext, str_: *const c_char) -> JSAtom;
    pub fn JS_DefineProperty(
        ctx: *mut JSContext,
        this_obj: JSValueConst,
        prop: JSAtom,
        val: JSValueConst,
        getter: JSValueConst,
        setter: JSValueConst,
        flags: c_int,
    ) -> c_int;
    pub fn JS_ThrowOutOfMemory(ctx: *mut JSContext) -> JSValue;
    pub fn JS_NewCFunction2(
        ctx: *mut JSContext,
        func: JSCFunction,
        name: *const c_char,
        length: c_int,
        cproto: c_int,
        magic: c_int,
    ) -> JSValue;
    pub fn js_std_dump_error(ctx: *mut JSContext);
    pub fn js_malloc(rt: *mut JSRuntime, size: usize) -> *mut c_void;
    pub fn JS_NewArrayBuffer(
        ctx: *mut JSContext,
        buf: *mut c_void,
        len: usize,
        free_func: Option<unsafe extern "C" fn(*mut JSRuntime, *mut c_void, *mut c_void)>,
        opaque: *mut c_void,
        flags: c_int,
    ) -> JSValue;
    // ── M10: the real exports behind the reimplemented shims ──
    pub fn __JS_FreeValue(ctx: *mut JSContext, v: JSValue);
    pub fn __JS_FreeValueRT(rt: *mut JSRuntime, v: JSValue);
    pub fn JS_NewString(ctx: *mut JSContext, str_: *const c_char) -> JSValue;
    pub fn JS_NewObject(ctx: *mut JSContext) -> JSValue;
    pub fn JS_GetGlobalObject(ctx: *mut JSContext) -> JSValue;
    pub fn JS_GetPropertyStr(
        ctx: *mut JSContext,
        this_obj: JSValueConst,
        prop: *const c_char,
    ) -> JSValue;
    pub fn JS_SetPropertyStr(
        ctx: *mut JSContext,
        this_obj: JSValueConst,
        prop: *const c_char,
        val: JSValue,
    ) -> c_int;
    pub fn JS_GetPropertyInternal(
        ctx: *mut JSContext,
        obj: JSValueConst,
        prop: JSAtom,
        receiver: JSValueConst,
        flags: c_int,
    ) -> JSValue;
    pub fn JS_FreeCString(ctx: *mut JSContext, ptr: *const c_char);
}

pub const JS_PROP_HAS_CONFIGURABLE: c_int = 1 << 8;
pub const JS_PROP_HAS_WRITABLE: c_int = 1 << 9; // verified (quickjs.h:277)
pub const JS_PROP_HAS_ENUMERABLE: c_int = 1 << 10; // verified (quickjs.h:278)
pub const JS_PROP_HAS_GET: c_int = 1 << 11;
pub const JS_PROP_HAS_SET: c_int = 1 << 12;

/// `JS_CFUNC_constructor` (quickjs.h JSCFunctionEnum) for JS_NewCFunction2.
pub const JS_CFUNC_constructor: c_int = 2;

// ── Direct C exports (linkable from Rust) ─────────────────────────

extern "C" {
    pub fn JS_NewRuntime() -> *mut JSRuntime;
    pub fn JS_FreeRuntime(rt: *mut JSRuntime);
    pub fn JS_NewContext(rt: *mut JSRuntime) -> *mut JSContext;
    pub fn JS_FreeContext(ctx: *mut JSContext);
    pub fn JS_GetRuntime(ctx: *mut JSContext) -> *mut JSRuntime;
    pub fn JS_RunGC(rt: *mut JSRuntime);
    pub fn JS_SetClassProto(ctx: *mut JSContext, class_id: JSClassID, obj: JSValue);
    pub fn JS_NewClassID(pclass_id: *mut JSClassID) -> JSClassID;
    pub fn JS_NewClass(rt: *mut JSRuntime, class_id: JSClassID, class_def: *const JSClassDef) -> c_int;
    pub fn JS_NewObjectClass(ctx: *mut JSContext, class_id: c_int) -> JSValue;
    pub fn JS_NewArray(ctx: *mut JSContext) -> JSValue;
    pub fn JS_IsArray(ctx: *mut JSContext, val: JSValueConst) -> c_int;
    pub fn JS_IsFunction(ctx: *mut JSContext, val: JSValueConst) -> c_int;
    pub fn JS_GetPropertyUint32(
        ctx: *mut JSContext,
        this_obj: JSValueConst,
        idx: u32,
    ) -> JSValue;
    pub fn JS_SetPropertyUint32(
        ctx: *mut JSContext,
        this_obj: JSValueConst,
        idx: u32,
        val: JSValue,
    ) -> c_int;
    pub fn JS_ToInt32(ctx: *mut JSContext, pres: *mut i32, val: JSValueConst) -> c_int;
    pub fn JS_ToInt64(ctx: *mut JSContext, pres: *mut i64, val: JSValueConst) -> c_int;
    pub fn JS_ToFloat64(ctx: *mut JSContext, pres: *mut f64, val: JSValueConst) -> c_int;
    pub fn JS_ToBool(ctx: *mut JSContext, val: JSValueConst) -> c_int;
    pub fn JS_NewStringLen(ctx: *mut JSContext, str1: *const c_char, len1: usize) -> JSValue;
    pub fn JS_NewError(ctx: *mut JSContext) -> JSValue;
    pub fn JS_ThrowTypeError(ctx: *mut JSContext, fmt: *const c_char, ...) -> JSValue;
    pub fn JS_ThrowInternalError(ctx: *mut JSContext, fmt: *const c_char, ...) -> JSValue; // quickjs.c:642 — real export
    pub fn JS_ThrowRangeError(ctx: *mut JSContext, fmt: *const c_char, ...) -> JSValue;
    pub fn JS_ThrowReferenceError(ctx: *mut JSContext, fmt: *const c_char, ...) -> JSValue;
    pub fn JS_Throw(ctx: *mut JSContext, obj: JSValue) -> JSValue;
    pub fn JS_GetException(ctx: *mut JSContext) -> JSValue;
    pub fn JS_Call(
        ctx: *mut JSContext,
        func_obj: JSValueConst,
        this_obj: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue;
    pub fn JS_Eval(
        ctx: *mut JSContext,
        input: *const c_char,
        input_len: usize,
        filename: *const c_char,
        eval_flags: c_int,
    ) -> JSValue;
    pub fn JS_ParseJSON(
        ctx: *mut JSContext,
        buf: *const c_char,
        buf_len: usize,
        filename: *const c_char,
    ) -> JSValue;
    pub fn JS_JSONStringify(
        ctx: *mut JSContext,
        obj: JSValueConst,
        replacer: JSValueConst,
        space: JSValueConst,
    ) -> JSValue;
    pub fn JS_SetOpaque(obj: JSValue, opaque: *mut c_void);
    pub fn JS_GetOpaque(obj: JSValueConst, class_id: JSClassID) -> *mut c_void;
    pub fn JS_GetOpaque2(
        ctx: *mut JSContext,
        obj: JSValueConst,
        class_id: JSClassID,
    ) -> *mut c_void;
    pub fn JS_GetTypedArrayBuffer(
        ctx: *mut JSContext,
        obj: JSValueConst,
        pbyte_offset: *mut usize,
        pbyte_length: *mut usize,
        pbytes_per_element: *mut usize,
    ) -> JSValue;
    pub fn JS_GetArrayBuffer(
        ctx: *mut JSContext,
        psize: *mut usize,
        obj: JSValueConst,
    ) -> *mut u8;
    pub fn JS_NewPromiseCapability(
        ctx: *mut JSContext,
        resolving_funcs: *mut JSValue,
    ) -> JSValue;
    pub fn JS_ExecutePendingJob(rt: *mut JSRuntime, pctx: *mut *mut JSContext) -> c_int;
    pub fn JS_IsJobPending(rt: *mut JSRuntime) -> c_int;
    pub fn JS_SetPropertyFunctionList(
        ctx: *mut JSContext,
        obj: JSValueConst,
        tab: *const JSCFunctionListEntry,
        len: c_int,
    );
    // RLM sandbox additions (all real exports in quickjs.c — no shim needed):
    pub fn JS_SetMemoryLimit(rt: *mut JSRuntime, limit: usize);
    pub fn JS_SetMaxStackSize(rt: *mut JSRuntime, stack_size: usize);
    pub fn JS_SetInterruptHandler(
        rt: *mut JSRuntime,
        cb: Option<JSInterruptHandler>,
        opaque: *mut c_void,
    );
    pub fn JS_NewCFunctionData(
        ctx: *mut JSContext,
        func: JSCFunctionDataFn,
        length: c_int,
        magic: c_int,
        data_len: c_int,
        data: *const JSValueConst,
    ) -> JSValue;
    // ── M10: engine boot + ESM loader (real exports) ──
    pub fn JS_SetGCThreshold(rt: *mut JSRuntime, gc_threshold: usize);
    pub fn JS_SetModuleLoaderFunc(
        rt: *mut JSRuntime,
        normalize: Option<JsModuleNormalizeFunc>,
        loader: Option<JsModuleLoaderFunc>,
        opaque: *mut c_void,
    );
    // quickjs-libc.c — the "std"/"os" helper modules (js_init_module_std/os).
    pub fn js_init_module_std(ctx: *mut JSContext, module_name: *const c_char) -> *mut c_void;
    pub fn js_init_module_os(ctx: *mut JSContext, module_name: *const c_char) -> *mut c_void;
    pub fn JS_ToCStringLen2(
        ctx: *mut JSContext,
        plen: *mut usize,
        val: JSValueConst,
        cesu8: c_int,
    ) -> *const c_char;
}

// ── JSCFunctionListEntry (JS_CFUNC_DEF tables) — quickjs.h:976-1009 ──
// { const char *name; uint8_t prop_flags; uint8_t def_type; int16_t magic;
//   union { struct { uint8_t length; uint8_t cproto; JSCFunctionType cfunc; }
//           func; ... } u; } — 32 bytes total. The pointer lives at union
// offset 8, NOT offset 0 (a 24-byte mirror here corrupted the atom table in
// M2 — the pointers shifted into the next entry).
// JS_CFUNC_DEF(name, length, func) = { name, WRITABLE|CONFIGURABLE,
//   JS_DEF_CFUNC, 0, { length, JS_CFUNC_generic, { .generic = func } } }.

/// `u.func` — layout-equal to the C struct (JSCFunctionType's `.generic`
/// slot is a plain JSCFunction pointer).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JSCFunctionListEntryFunc {
    pub length: u8,
    pub cproto: u8, /* JSCFunctionEnum: JS_CFUNC_generic == 0 */
    pub _pad: [u8; 6],
    pub cfunc: JSCFunction,
}

#[repr(C)]
pub struct JSCFunctionListEntry {
    pub name: *const c_char,
    pub prop_flags: u8,
    pub def_type: u8,
    pub magic: i16,
    pub u: JSCFunctionListEntryFunc,
}

pub const JS_DEF_CFUNC: u8 = 0;
pub const JS_PROP_CONFIGURABLE: u8 = 0x10;

// ── Eval flags (quickjs.h:296-309 — verified against the vendored
// header after M10's loader debug: JS_EVAL_FLAG_COMPILE_ONLY is (1<<5),
// NOT 0x400; the wrong value made the ESM loader fully link+run every
// imported module inside JS_Eval, then the parent linked them a second
// time → js_inner_module_linking asserts + corrupt module defs) ──
pub const JS_EVAL_TYPE_GLOBAL: c_int = 0;
pub const JS_EVAL_TYPE_MODULE: c_int = 1;
pub const JS_EVAL_FLAG_STRICT: c_int = 1 << 3;
pub const JS_EVAL_FLAG_STRIP: c_int = 1 << 4;
pub const JS_EVAL_FLAG_COMPILE_ONLY: c_int = 1 << 5;
pub const JS_EVAL_FLAG_BACKTRACE_BARRIER: c_int = 1 << 6;

// ── Rooted<T> — the GC-safety guard ───────────────────────────────
//
// Takes OWNERSHIP of a JSValue (the caller's reference transfers in) and
// frees it on drop. This is the exact discipline that prevents the M6
// crash class: a value held across an FFI boundary (a uv callback, a job
// pump) stays alive for the Rooted's whole scope, and is freed exactly
// once when the Rooted drops. Callers must NOT free the value separately.

pub struct Rooted<'a> {
    pub ctx: CtxPtr<'a>,
    pub value: JSValue,
}

impl<'a> Rooted<'a> {
    /// Takes ownership of `v`. `ctx` must be the value's context.
    ///
    /// # Safety
    /// `v` must be a live value of `ctx`, and the caller must transfer its
    /// reference (no separate free afterward).
    pub unsafe fn new(ctx: CtxPtr<'a>, v: JSValue) -> Self {
        Self { ctx, value: v }
    }

    pub fn as_value(&self) -> JSValueConst {
        self.value
    }
}

impl Drop for Rooted<'_> {
    fn drop(&mut self) {
        // SAFETY: self.value is owned by this Rooted; self.ctx is the same
        // context. Freed exactly once here.
        unsafe { sofuu_js_free_value(self.ctx.as_ptr(), self.value) };
    }
}

// ── CtxPtr — loop-thread affinity ─────────────────────────────────
//
// Wraps a *mut JSContext and is `!Send`: QuickJS contexts may only be used
// from the thread that created them (single-threaded runtime). The compiler
// enforces this — no runtime check needed.

#[derive(Clone, Copy)]
pub struct CtxPtr<'a> {
    ctx: *mut JSContext,
    _not_send: std::marker::PhantomData<*const ()>,
    _lifetime: std::marker::PhantomData<&'a ()>,
}

impl<'a> CtxPtr<'a> {
    /// # Safety
    /// `ctx` must be a valid, live QuickJS context owned by the current
    /// thread (created here or via the runtime init).
    pub unsafe fn new(ctx: *mut JSContext) -> Self {
        Self {
            ctx,
            _not_send: std::marker::PhantomData,
            _lifetime: std::marker::PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *mut JSContext {
        self.ctx
    }
}

// `CtxPtr` is deliberately !Send: PhantomData<*const ()> makes it so.

// ── Safe-ish value helpers (all unsafe — caller holds a live ctx) ─

/// # Safety
/// `ctx` valid; returned value must be freed or rooted.
pub unsafe fn new_string(ctx: CtxPtr, s: &str) -> JSValue {
    let c = CString::new(s).unwrap_or_default();
    // SAFETY: c valid for the call; QuickJS copies.
    sofuu_js_new_string(ctx.as_ptr(), c.as_ptr())
}

/// # Safety
/// `ctx` valid; returned value must be freed or rooted.
pub unsafe fn new_float64(ctx: CtxPtr, d: f64) -> JSValue {
    // SAFETY: trivial.
    sofuu_js_new_float64(ctx.as_ptr(), d)
}

/// # Safety
/// `ctx` valid; returned value must be freed or rooted.
pub unsafe fn new_int32(ctx: CtxPtr, i: i32) -> JSValue {
    // SAFETY: trivial.
    sofuu_js_new_int32(ctx.as_ptr(), i)
}

/// # Safety
/// `ctx` valid; returned value must be freed or rooted.
pub unsafe fn new_bool(ctx: CtxPtr, b: bool) -> JSValue {
    // SAFETY: trivial.
    sofuu_js_new_bool(ctx.as_ptr(), if b { 1 } else { 0 })
}

/// # Safety
/// `ctx` valid; returned value must be freed or rooted.
pub unsafe fn new_object(ctx: CtxPtr) -> JSValue {
    // SAFETY: trivial.
    sofuu_js_new_object(ctx.as_ptr())
}

/// # Safety
/// `ctx` valid; returned value must be freed or rooted.
pub unsafe fn new_array(ctx: CtxPtr) -> JSValue {
    // SAFETY: trivial.
    JS_NewArray(ctx.as_ptr())
}

/// # Safety
/// `ctx` valid; returned value must be freed or rooted.
pub unsafe fn global_object(ctx: CtxPtr) -> JSValue {
    // SAFETY: trivial.
    sofuu_js_get_global_object(ctx.as_ptr())
}

/// # Safety
/// `ctx` valid; `v` a live value of `ctx`.
pub unsafe fn get_property_str(ctx: CtxPtr, obj: JSValueConst, prop: &str) -> JSValue {
    let c = CString::new(prop).unwrap_or_default();
    // SAFETY: c valid; obj live.
    sofuu_js_get_property_str(ctx.as_ptr(), obj, c.as_ptr())
}

/// # Safety
/// `ctx` valid; `val` is a live value; the returned string must be freed
/// with `free_cstring`.
pub unsafe fn to_cstring(ctx: CtxPtr, val: JSValueConst) -> *const c_char {
    // SAFETY: delegated to the C shim.
    sofuu_js_to_cstring(ctx.as_ptr(), val)
}

/// # Safety
/// `ptr` must come from `to_cstring` on this context.
pub unsafe fn free_cstring(ctx: CtxPtr, ptr: *const c_char) {
    // SAFETY: delegated.
    sofuu_js_free_cstring(ctx.as_ptr(), ptr)
}

/// Convert a JS string to a Rust String. None if not a string.
///
/// # Safety
/// `ctx` valid; `val` a live value of `ctx`.
pub unsafe fn value_to_string(ctx: CtxPtr, val: JSValueConst) -> Option<String> {
    // SAFETY: shim inspects the tag.
    if sofuu_js_is_string(val) == 0 {
        return None;
    }
    // SAFETY: to_cstring returns a ptr we free.
    let p = to_cstring(ctx, val);
    if p.is_null() {
        return None;
    }
    let s = CStr::from_ptr(p).to_string_lossy().into_owned();
    // SAFETY: p from to_cstring on this ctx.
    free_cstring(ctx, p);
    Some(s)
}

/// # Safety
/// `ctx` valid; `func` must be `'static`; returned value must be freed.
pub unsafe fn new_cfunction(ctx: CtxPtr, name: &str, func: JSCFunction, length: c_int) -> JSValue {
    let c = CString::new(name).unwrap_or_default();
    // SAFETY: c valid; func 'static.
    sofuu_js_new_cfunction(ctx.as_ptr(), func, c.as_ptr(), length)
}

/// Pump the job queue until empty. Returns 0 when calm, -1 on an
/// unhandled job exception (exception left in `pctx`).
///
/// # Safety
/// `rt` must be the runtime owning the current context.
pub unsafe fn flush_jobs(rt: *mut JSRuntime) -> c_int {
    // SAFETY: rt valid.
    loop {
        let mut ctx2: *mut JSContext = ptr::null_mut();
        let err = JS_ExecutePendingJob(rt, &mut ctx2);
        if err <= 0 {
            return err;
        }
    }
}

// ── Tag helpers (via shims — never peek the layout) ──────────────

/// # Safety
/// `v` must be a valid JSValue (from a live context).
pub unsafe fn is_exception(v: JSValueConst) -> bool {
    // SAFETY: shim.
    sofuu_js_is_exception(v) != 0
}

/// # Safety
/// `v` must be a valid JSValue.
pub unsafe fn is_undefined(v: JSValueConst) -> bool {
    // SAFETY: shim.
    sofuu_js_is_undefined(v) != 0
}

/// # Safety
/// `v` must be a valid JSValue.
pub unsafe fn is_null(v: JSValueConst) -> bool {
    // SAFETY: shim.
    sofuu_js_is_null(v) != 0
}

/// # Safety
/// `v` must be a valid JSValue.
pub unsafe fn is_object(v: JSValueConst) -> bool {
    // SAFETY: shim.
    sofuu_js_is_object(v) != 0
}

/// # Safety
/// `v` must be a valid JSValue.
pub unsafe fn is_number(v: JSValueConst) -> bool {
    // SAFETY: shim.
    sofuu_js_is_number(v) != 0
}

/// # Safety
/// `ctx` valid; `v` a live value; `out` is written on success.
pub unsafe fn to_int32(ctx: CtxPtr, v: JSValueConst, out: *mut i32) -> bool {
    // SAFETY: delegated.
    JS_ToInt32(ctx.as_ptr(), out, v) == 0
}

/// # Safety
/// `ctx` valid; `v` a live value; `out` is written on success.
pub unsafe fn to_int64(ctx: CtxPtr, v: JSValueConst, out: *mut i64) -> bool {
    // SAFETY: delegated.
    JS_ToInt64(ctx.as_ptr(), out, v) == 0
}

/// # Safety
/// `ctx` valid; `v` a live value; `out` is written on success.
pub unsafe fn to_float64(ctx: CtxPtr, v: JSValueConst, out: *mut f64) -> bool {
    // SAFETY: delegated.
    JS_ToFloat64(ctx.as_ptr(), out, v) == 0
}
