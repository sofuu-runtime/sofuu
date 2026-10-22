// zerocopy.rs — Zero-Copy TypedArray & Native Memory Bridge (Phase 3).
//
// Bridges native Rust memory (aligned SIMD allocations, Vec, Arc) directly
// into JavaScript ArrayBuffer and TypedArray views without copying.
// External deallocators registered with QuickJS ensure that when JavaScript
// garbage-collects the ArrayBuffer, the Rust backing buffer is safely reclaimed.

use std::alloc::{alloc, dealloc, Layout};
use std::ffi::c_void;
use std::os::raw::c_int;
use std::ptr;
use std::sync::Arc;

use sofuu_ffi::bridge::{JSContext, JSValue, JSValueConst};
use sofuu_ffi::qjs;

/// Represents a 64-byte aligned native heap buffer for high-throughput SIMD vector operations.
pub struct AlignedBuffer {
    pub ptr: *mut u8,
    pub layout: Layout,
    pub size: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate `size` bytes aligned to `align` (default 64 bytes for AVX-512 / NEON).
    pub fn new(size: usize, align: usize) -> Result<Self, String> {
        if size == 0 {
            return Err("cannot allocate 0 bytes".to_string());
        }
        let align = if align == 0 || !align.is_power_of_two() {
            64
        } else {
            align
        };
        let layout = Layout::from_size_align(size, align)
            .map_err(|e| format!("invalid memory layout: {e}"))?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(format!("failed to allocate {size} bytes aligned to {align}"));
        }
        // Zero-initialize memory for safety
        unsafe { ptr::write_bytes(ptr, 0, size) };
        Ok(Self { ptr, layout, size })
    }
}

// ── QuickJS C Deallocator Trampolines ─────────────────────────────

unsafe extern "C" fn free_aligned_buffer_cb(
    _rt: *mut qjs::JSRuntime,
    opaque: *mut c_void,
    _ptr: *mut c_void,
) {
    if !opaque.is_null() {
        let buf = Box::from_raw(opaque as *mut AlignedBuffer);
        dealloc(buf.ptr, buf.layout);
    }
}

unsafe extern "C" fn free_vec_u8_cb(
    _rt: *mut qjs::JSRuntime,
    opaque: *mut c_void,
    _ptr: *mut c_void,
) {
    if !opaque.is_null() {
        let _ = Box::from_raw(opaque as *mut Vec<u8>);
    }
}

unsafe extern "C" fn free_vec_f32_cb(
    _rt: *mut qjs::JSRuntime,
    opaque: *mut c_void,
    _ptr: *mut c_void,
) {
    if !opaque.is_null() {
        let _ = Box::from_raw(opaque as *mut Vec<f32>);
    }
}


// ── Rust-side Zero-Copy Constructors ──────────────────────────────

/// Wrap an AlignedBuffer into a JavaScript ArrayBuffer with zero copying.
///
/// # Safety
/// `ctx` must be a live QuickJS context on the engine thread.
pub unsafe fn new_zerocopy_aligned_buffer(
    ctx: *mut JSContext,
    buf: AlignedBuffer,
) -> Result<JSValue, String> {
    let size = buf.size;
    let ptr = buf.ptr;
    let boxed = Box::into_raw(Box::new(buf));

    let ab = qjs::JS_NewArrayBuffer(
        ctx,
        ptr as *mut c_void,
        size,
        Some(free_aligned_buffer_cb),
        boxed as *mut c_void,
        0, // is_shared = false
    );

    if qjs::is_exception(ab) {
        // SAFETY: ArrayBuffer creation failed; deallocate buffer immediately.
        let buf = Box::from_raw(boxed);
        dealloc(buf.ptr, buf.layout);
        return Err("failed to create JS ArrayBuffer".to_string());
    }

    Ok(ab)
}

/// Wrap a Rust Vec<u8> into a JavaScript ArrayBuffer with zero copying.
///
/// # Safety
/// `ctx` must be a live QuickJS context.
pub unsafe fn new_zerocopy_u8_buffer(ctx: *mut JSContext, mut vec: Vec<u8>) -> JSValue {
    let len = vec.len();
    if len == 0 {
        return qjs::JS_NewArrayBufferCopy(ctx, ptr::null(), 0);
    }
    let ptr = vec.as_mut_ptr();
    let boxed = Box::into_raw(Box::new(vec));

    let ab = qjs::JS_NewArrayBuffer(
        ctx,
        ptr as *mut c_void,
        len,
        Some(free_vec_u8_cb),
        boxed as *mut c_void,
        0,
    );

    if qjs::is_exception(ab) {
        let _ = Box::from_raw(boxed);
    }
    ab
}

/// Wrap a Rust Vec<f32> into a JavaScript Float32Array with zero copying.
///
/// # Safety
/// `ctx` must be a live QuickJS context.
pub unsafe fn new_zerocopy_f32_array(ctx: *mut JSContext, mut vec: Vec<f32>) -> JSValue {
    let len = vec.len();
    let byte_len = len * std::mem::size_of::<f32>();
    if len == 0 {
        let empty_ab = qjs::JS_NewArrayBufferCopy(ctx, ptr::null(), 0);
        return create_typed_view(ctx, empty_ab, "Float32Array", None, None);
    }
    let ptr = vec.as_mut_ptr() as *mut u8;
    let boxed = Box::into_raw(Box::new(vec));

    let ab = qjs::JS_NewArrayBuffer(
        ctx,
        ptr as *mut c_void,
        byte_len,
        Some(free_vec_f32_cb),
        boxed as *mut c_void,
        0,
    );

    if qjs::is_exception(ab) {
        let _ = Box::from_raw(boxed);
        return ab;
    }

    create_typed_view(ctx, ab, "Float32Array", None, None)
}

/// Wrap a Rust Arc<Vec<f32>> into a JavaScript Float32Array safely.
/// If the Arc is uniquely owned, ownership is taken with zero copies;
/// if shared across multiple owners, it clones the buffer so JavaScript
/// writes cannot violate Rust's shared reference immutability invariants.
///
/// # Safety
/// `ctx` must be a live QuickJS context.
pub unsafe fn new_shared_f32_array(ctx: *mut JSContext, arc: Arc<Vec<f32>>) -> JSValue {
    let vec = match Arc::try_unwrap(arc) {
        Ok(v) => v,
        Err(shared) => (*shared).clone(),
    };
    new_zerocopy_f32_array(ctx, vec)
}

/// Create a TypedArray view (e.g. Float32Array, Uint8Array) over an ArrayBuffer.
///
/// # Safety
/// `ctx` must be a live QuickJS context.
pub unsafe fn create_typed_view(
    ctx: *mut JSContext,
    ab: JSValue,
    constructor_name: &str,
    byte_offset: Option<usize>,
    length: Option<usize>,
) -> JSValue {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let c_name = std::ffi::CString::new(constructor_name).unwrap_or_default();
    let ctor = qjs::sofuu_js_get_property_str(ctx, global, c_name.as_ptr());
    qjs::sofuu_js_free_value(ctx, global);

    if qjs::is_exception(ctor) {
        qjs::sofuu_js_free_value(ctx, ab);
        return ctor;
    }

    let result = if byte_offset.is_some() || length.is_some() {
        let off = byte_offset.unwrap_or(0);
        let off_val = qjs::sofuu_js_new_int32(ctx, off as i32);
        let mut args = vec![ab, off_val];
        if let Some(l) = length {
            args.push(qjs::sofuu_js_new_int32(ctx, l as i32));
        }
        let res = qjs::JS_CallConstructor(ctx, ctor, args.len() as c_int, args.as_ptr());
        qjs::sofuu_js_free_value(ctx, off_val);
        if args.len() == 3 {
            qjs::sofuu_js_free_value(ctx, args[2]);
        }
        res
    } else {
        qjs::JS_CallConstructor(ctx, ctor, 1, &ab)
    };

    qjs::sofuu_js_free_value(ctx, ab);
    qjs::sofuu_js_free_value(ctx, ctor);
    result
}

/// Safely extract raw pointer, byte offset, byte length, and element count from any JS TypedArray or ArrayBuffer.
///
/// # Safety
/// `ctx` must be a live QuickJS context.
pub unsafe fn extract_buffer_info(
    ctx: *mut JSContext,
    val: JSValueConst,
) -> Option<(*mut u8, usize, usize)> {
    let mut byte_offset: usize = 0;
    let mut byte_length: usize = 0;
    let ab = qjs::JS_GetTypedArrayBuffer(
        ctx,
        val,
        &mut byte_offset,
        &mut byte_length,
        ptr::null_mut(),
    );

    if !qjs::is_exception(ab) {
        let mut buf_size: usize = 0;
        let p = qjs::JS_GetArrayBuffer(ctx, &mut buf_size, ab);
        qjs::sofuu_js_free_value(ctx, ab);
        if p.is_null() || byte_offset + byte_length > buf_size {
            return None;
        }
        return Some((p.add(byte_offset), byte_length, buf_size));
    }
    qjs::sofuu_js_free_value(ctx, ab);

    // Maybe val is directly an ArrayBuffer
    let mut buf_size: usize = 0;
    let p = qjs::JS_GetArrayBuffer(ctx, &mut buf_size, val);
    if !p.is_null() {
        return Some((p, buf_size, buf_size));
    }

    None
}

// ── JavaScript Host API Implementations ───────────────────────────

/// JS: `sofuu.memory.allocAligned(bytes, alignment = 64)` -> ArrayBuffer
unsafe extern "C" fn js_memory_alloc_aligned(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"allocAligned requires size in bytes".as_ptr());
    }
    let mut bytes: i32 = 0;
    if qjs::JS_ToInt32(ctx, &mut bytes, *argv) < 0 || bytes <= 0 {
        return qjs::JS_ThrowRangeError(ctx, c"allocAligned size must be positive".as_ptr());
    }
    let mut align: i32 = 64;
    if argc >= 2 {
        let _ = qjs::JS_ToInt32(ctx, &mut align, *argv.add(1));
    }
    if align <= 0 || (align & (align - 1)) != 0 {
        align = 64;
    }

    let buf = match AlignedBuffer::new(bytes as usize, align as usize) {
        Ok(b) => b,
        Err(e) => {
            let err_c = std::ffi::CString::new(e).unwrap_or_default();
            return qjs::JS_ThrowInternalError(ctx, err_c.as_ptr());
        }
    };

    match new_zerocopy_aligned_buffer(ctx, buf) {
        Ok(ab) => ab,
        Err(e) => {
            let err_c = std::ffi::CString::new(e).unwrap_or_default();
            qjs::JS_ThrowInternalError(ctx, err_c.as_ptr())
        }
    }
}

/// JS: `sofuu.memory.allocFloat32(length)` -> Float32Array (64-byte aligned)
unsafe extern "C" fn js_memory_alloc_float32(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"allocFloat32 requires length".as_ptr());
    }
    let mut len: i32 = 0;
    if qjs::JS_ToInt32(ctx, &mut len, *argv) < 0 || len < 0 {
        return qjs::JS_ThrowRangeError(ctx, c"allocFloat32 length must be non-negative".as_ptr());
    }
    let byte_len = (len as usize) * std::mem::size_of::<f32>();
    let byte_len = if byte_len == 0 { 64 } else { byte_len };

    let buf = match AlignedBuffer::new(byte_len, 64) {
        Ok(b) => b,
        Err(e) => {
            let err_c = std::ffi::CString::new(e).unwrap_or_default();
            return qjs::JS_ThrowInternalError(ctx, err_c.as_ptr());
        }
    };

    let ab = match new_zerocopy_aligned_buffer(ctx, buf) {
        Ok(ab) => ab,
        Err(e) => {
            let err_c = std::ffi::CString::new(e).unwrap_or_default();
            return qjs::JS_ThrowInternalError(ctx, err_c.as_ptr());
        }
    };

    create_typed_view(ctx, ab, "Float32Array", Some(0), Some(len as usize))
}

/// JS: `sofuu.memory.allocUint8(length)` -> Uint8Array (64-byte aligned)
unsafe extern "C" fn js_memory_alloc_uint8(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"allocUint8 requires length".as_ptr());
    }
    let mut len: i32 = 0;
    if qjs::JS_ToInt32(ctx, &mut len, *argv) < 0 || len < 0 {
        return qjs::JS_ThrowRangeError(ctx, c"allocUint8 length must be non-negative".as_ptr());
    }
    let byte_len = if len == 0 { 64 } else { len as usize };

    let buf = match AlignedBuffer::new(byte_len, 64) {
        Ok(b) => b,
        Err(e) => {
            let err_c = std::ffi::CString::new(e).unwrap_or_default();
            return qjs::JS_ThrowInternalError(ctx, err_c.as_ptr());
        }
    };

    let ab = match new_zerocopy_aligned_buffer(ctx, buf) {
        Ok(ab) => ab,
        Err(e) => {
            let err_c = std::ffi::CString::new(e).unwrap_or_default();
            return qjs::JS_ThrowInternalError(ctx, err_c.as_ptr());
        }
    };

    create_typed_view(ctx, ab, "Uint8Array", Some(0), Some(len as usize))
}

/// JS: `sofuu.memory.viewFloat32(arrayBuffer, byteOffset?, length?)` -> Float32Array
unsafe extern "C" fn js_memory_view_float32(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"viewFloat32 requires an ArrayBuffer".as_ptr());
    }
    let ab = *argv;
    let mut byte_off = None;
    if argc >= 2 {
        let mut off: i32 = 0;
        if qjs::JS_ToInt32(ctx, &mut off, *argv.add(1)) == 0 && off >= 0 {
            byte_off = Some(off as usize);
        }
    }
    let mut len = None;
    if argc >= 3 {
        let mut l: i32 = 0;
        if qjs::JS_ToInt32(ctx, &mut l, *argv.add(2)) == 0 && l >= 0 {
            len = Some(l as usize);
        }
    }
    create_typed_view(ctx, qjs::sofuu_js_dup_value(ctx, ab), "Float32Array", byte_off, len)
}

/// JS: `sofuu.memory.viewUint8(arrayBuffer, byteOffset?, length?)` -> Uint8Array
unsafe extern "C" fn js_memory_view_uint8(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"viewUint8 requires an ArrayBuffer".as_ptr());
    }
    let ab = *argv;
    let mut byte_off = None;
    if argc >= 2 {
        let mut off: i32 = 0;
        if qjs::JS_ToInt32(ctx, &mut off, *argv.add(1)) == 0 && off >= 0 {
            byte_off = Some(off as usize);
        }
    }
    let mut len = None;
    if argc >= 3 {
        let mut l: i32 = 0;
        if qjs::JS_ToInt32(ctx, &mut l, *argv.add(2)) == 0 && l >= 0 {
            len = Some(l as usize);
        }
    }
    create_typed_view(ctx, qjs::sofuu_js_dup_value(ctx, ab), "Uint8Array", byte_off, len)
}

/// JS: `sofuu.memory.detach(arrayBuffer)` -> boolean
unsafe extern "C" fn js_memory_detach(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"detach requires an ArrayBuffer".as_ptr());
    }
    qjs::JS_DetachArrayBuffer(ctx, *argv);
    qjs::sofuu_js_new_bool(ctx, 1)
}

/// JS: `sofuu.memory.isDetached(arrayBuffer)` -> boolean
unsafe extern "C" fn js_memory_is_detached(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_new_bool(ctx, 1);
    }
    let mut size: usize = 0;
    let ptr = qjs::JS_GetArrayBuffer(ctx, &mut size, *argv);
    let detached = ptr.is_null() && size == 0;
    qjs::sofuu_js_new_bool(ctx, if detached { 1 } else { 0 })
}

/// JS: `sofuu.memory.addressOf(typedArrayOrBuffer)` -> string
unsafe extern "C" fn js_memory_address_of(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_new_string(ctx, c"0x0".as_ptr());
    }
    let addr_str = if let Some((ptr, _, _)) = extract_buffer_info(ctx, *argv) {
        format!("0x{:x}", ptr as usize)
    } else {
        "0x0".to_string()
    };
    let c_str = std::ffi::CString::new(addr_str).unwrap_or_default();
    qjs::sofuu_js_new_string(ctx, c_str.as_ptr())
}

/// JS: `sofuu.memory.isAligned(typedArrayOrBuffer, alignment = 64)` -> boolean
unsafe extern "C" fn js_memory_is_aligned(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_new_bool(ctx, 0);
    }
    let mut align: i32 = 64;
    if argc >= 2 {
        let _ = qjs::JS_ToInt32(ctx, &mut align, *argv.add(1));
    }
    if align <= 0 {
        align = 64;
    }
    if let Some((ptr, _, _)) = extract_buffer_info(ctx, *argv) {
        let is_al = ((ptr as usize) % (align as usize)) == 0;
        qjs::sofuu_js_new_bool(ctx, if is_al { 1 } else { 0 })
    } else {
        qjs::sofuu_js_new_bool(ctx, 0)
    }
}

// ── Module Registration ───────────────────────────────────────────

/// Register zero-copy memory bridge primitives onto `sofuu.memory` and `sofuu.buffer`.
///
/// # Safety
/// `ctx` must be the live QuickJS context during engine initialization.
pub unsafe fn mod_zerocopy_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let sofuu = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());

    // Ensure `sofuu.memory` exists (may already exist via QTSQ CMA)
    let memory_obj = qjs::sofuu_js_get_property_str(ctx, sofuu, c"memory".as_ptr());
    let memory = if qjs::is_object(memory_obj) {
        memory_obj
    } else {
        qjs::sofuu_js_free_value(ctx, memory_obj);
        let m = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, sofuu, c"memory".as_ptr(), qjs::sofuu_js_dup_value(ctx, m));
        m
    };

    // Helper macro to register a C function on a target JS object
    macro_rules! reg_fn {
        ($obj:expr, $name:literal, $func:expr, $len:expr) => {{
            let c_name = std::ffi::CString::new($name).unwrap_or_default();
            let f = qjs::sofuu_js_new_cfunction(
                ctx,
                $func,
                c_name.as_ptr(),
                $len,
            );
            qjs::sofuu_js_set_property_str(ctx, $obj, c_name.as_ptr(), f);
        }};
    }

    // Register on `sofuu.memory`
    reg_fn!(memory, "allocAligned", js_memory_alloc_aligned, 2);
    reg_fn!(memory, "allocFloat32", js_memory_alloc_float32, 1);
    reg_fn!(memory, "allocUint8", js_memory_alloc_uint8, 1);
    reg_fn!(memory, "viewFloat32", js_memory_view_float32, 3);
    reg_fn!(memory, "viewUint8", js_memory_view_uint8, 3);
    reg_fn!(memory, "detach", js_memory_detach, 1);
    reg_fn!(memory, "isDetached", js_memory_is_detached, 1);
    reg_fn!(memory, "addressOf", js_memory_address_of, 1);
    reg_fn!(memory, "isAligned", js_memory_is_aligned, 2);

    // Also register on `sofuu.buffer` for modern Node/Bun-style ergonomic access
    let buffer_obj = qjs::sofuu_js_new_object(ctx);
    reg_fn!(buffer_obj, "allocAligned", js_memory_alloc_aligned, 2);
    reg_fn!(buffer_obj, "allocFloat32", js_memory_alloc_float32, 1);
    reg_fn!(buffer_obj, "allocUint8", js_memory_alloc_uint8, 1);
    reg_fn!(buffer_obj, "viewFloat32", js_memory_view_float32, 3);
    reg_fn!(buffer_obj, "viewUint8", js_memory_view_uint8, 3);
    reg_fn!(buffer_obj, "detach", js_memory_detach, 1);
    reg_fn!(buffer_obj, "isDetached", js_memory_is_detached, 1);
    reg_fn!(buffer_obj, "addressOf", js_memory_address_of, 1);
    reg_fn!(buffer_obj, "isAligned", js_memory_is_aligned, 2);
    qjs::sofuu_js_set_property_str(ctx, sofuu, c"buffer".as_ptr(), buffer_obj);

    qjs::sofuu_js_free_value(ctx, memory);
    qjs::sofuu_js_free_value(ctx, sofuu);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_allocation_and_alignment() {
        let buf = AlignedBuffer::new(1024, 64).expect("aligned buffer");
        assert_eq!(buf.size, 1024);
        assert!(!buf.ptr.is_null());
        assert_eq!((buf.ptr as usize) % 64, 0, "must be 64-byte aligned");
    }

    #[test]
    fn test_zerocopy_f32_array_creation() {
        let rt = unsafe { qjs::JS_NewRuntime() };
        assert!(!rt.is_null());
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        assert!(!ctx.is_null());

        let floats = vec![1.0f32, 2.5f32, -4.0f32, 100.25f32];
        let js_arr = unsafe { new_zerocopy_f32_array(ctx, floats) };
        assert!(unsafe { qjs::is_object(js_arr) });

        let info = unsafe { extract_buffer_info(ctx, js_arr) };
        assert!(info.is_some(), "must extract buffer info");
        let (ptr, byte_len, _) = info.unwrap();
        assert_eq!(byte_len, 4 * 4);

        let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, 4) };
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1], 2.5);
        assert_eq!(slice[2], -4.0);
        assert_eq!(slice[3], 100.25);

        unsafe {
            qjs::sofuu_js_free_value(ctx, js_arr);
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }
}
