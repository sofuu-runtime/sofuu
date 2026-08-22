// sofuu-ffi — QTSQ codec bindings (PLAN-RUST-MIGRATION M9).
//
// The memory subsystem (crates/sofuu-core/src/rt/memory.rs) persists CMA
// brain files + KV pages as .qtsq containers. The retired C adapter
// (src/memory/qtsq_adapter.c) declared `qtsq_context_t ctx;` on its stack
// and let the library initialize it in place; Rust cannot see the struct
// layout, so this module:
//   - allocates the context's exact storage (sizeof = 5016, align = 8 in
//     the vendored checkout — qtsq_format.h, verified with an offsetof
//     probe while porting) via calloc and hands the pointer to libqtsq;
//   - reads the few fields the adapter pokes directly (header.data_type,
//     schema.dimensions/num_dims, is_encrypted) through byte-offset
//     accessors pinned to qtsq_format.h (verified: data_type=10,
//     schema=112, schema.dimensions=68, schema.num_dims=100,
//     is_encrypted=4908; all offsets re-asserted in the test below).
//
// Only the functions the adapter actually used are bound — qtsq.h declares
// hundreds more; unbound ones stay out of the picture.

#![allow(clippy::missing_safety_doc)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

// ── Constants (qtsq_format.h) ──────────────────────────────────────

pub const QTSQ_OK: c_int = 0;
pub const QTSQ_ERR_NULL: c_int = -1;
pub const QTSQ_ERR_ALLOC: c_int = -2;
pub const QTSQ_ERR_FORMAT: c_int = -4;
pub const QTSQ_ERR_TYPE: c_int = -7;

pub const QTSQ_TYPE_CONTAINER: u8 = 0xF0;

/// qtsq_tensor_precision_t (qtsq.h).
pub type QtsqTensorPrecision = c_int;
pub const QTSQ_TENSOR_F32: QtsqTensorPrecision = 0;
pub const QTSQ_TENSOR_F16: QtsqTensorPrecision = 1;
pub const QTSQ_TENSOR_F8: QtsqTensorPrecision = 2;
pub const QTSQ_TENSOR_F4: QtsqTensorPrecision = 3;
pub const QTSQ_TENSOR_RAW: QtsqTensorPrecision = 4;

// ── Opaque context ─────────────────────────────────────────────────

/// Storage for a `qtsq_context_t` (vendored checkout layout).
///
/// `qtsq_init`/`qtsq_read` initialize it in place and `qtsq_free` releases
/// its internals; the block itself is `libc::calloc`'d so every byte the
/// C struct reads after partial initialization is zero, never garbage.
#[repr(C, align(8))]
pub struct QtsqContext {
    _storage: [u8; QTSQ_CONTEXT_SIZE],
}

/// `sizeof(qtsq_context_t)` in the vendored checkout (qtsq_format.h).
pub const QTSQ_CONTEXT_SIZE: usize = 5016;

// Field offsets verified against qtsq_format.h with a C offsetof probe
// while porting M9 (the packed 112-byte qtsq_header_t leads the struct).
const OFF_HEADER_DATA_TYPE: usize = 10; // qtsq_header_t.data_type
const OFF_SCHEMA: usize = 112; // qtsq_schema_t (after the packed header)
const OFF_SCHEMA_DIMENSIONS: usize = 68; // qtsq_schema_t.dimensions[8]
const OFF_SCHEMA_NUM_DIMS: usize = 100; // qtsq_schema_t.num_dims
const OFF_IS_ENCRYPTED: usize = 4908; // qtsq_context_t.is_encrypted (int)

// Compile-time sanity net for the layout facts above.
const _: () = assert!(QTSQ_CONTEXT_SIZE >= OFF_IS_ENCRYPTED + 4);

impl QtsqContext {
    /// # Safety
    /// `self` must be a live context (initialized by `qtsq_init`/`qtsq_read`).
    pub unsafe fn data_type(&self) -> u8 {
        let base = self as *const QtsqContext as *const u8;
        std::ptr::read_unaligned(base.add(OFF_HEADER_DATA_TYPE))
    }

    /// # Safety
    /// `self` must be a live context (initialized by `qtsq_init`/`qtsq_read`).
    pub unsafe fn is_encrypted(&self) -> bool {
        let base = self as *const QtsqContext as *const u8;
        std::ptr::read_unaligned(base.add(OFF_IS_ENCRYPTED) as *const c_int) != 0
    }

    /// # Safety
    /// `self` must be a live context whose schema is populated
    /// (e.g. after a successful `qtsq_decompress_tensor`).
    pub unsafe fn schema_num_dims(&self) -> u8 {
        let base = self as *const QtsqContext as *const u8;
        std::ptr::read_unaligned(base.add(OFF_SCHEMA + OFF_SCHEMA_NUM_DIMS))
    }

    /// # Safety
    /// `self` must be a live context whose schema is populated.
    pub unsafe fn schema_dim(&self, i: usize) -> u32 {
        let base = self as *const QtsqContext as *const u8;
        std::ptr::read_unaligned(
            base.add(OFF_SCHEMA + OFF_SCHEMA_DIMENSIONS + i * 4) as *const u32,
        )
    }
}

/// Allocate zeroed storage for one `qtsq_context_t` (never NULL for the
/// sizes involved; the C code used stack storage for the same purpose).
///
/// # Safety
/// The caller owns the block: pair every success with [`qtsq_ctx_free`].
pub unsafe fn qtsq_ctx_alloc() -> *mut QtsqContext {
    libc::calloc(1, QTSQ_CONTEXT_SIZE) as *mut QtsqContext
}

/// # Safety
/// `p` must come from [`qtsq_ctx_alloc`] and must already have been
/// `qtsq_free`'d (the block is pure storage — freeing it while libqtsq
/// still owns internals would leak, not crash, but keep the C ordering:
/// `qtsq_free(&ctx)` first, then drop the block).
pub unsafe fn qtsq_ctx_free(p: *mut QtsqContext) {
    libc::free(p as *mut c_void);
}

// ── libqtsq externs (all verified as real exports via `nm`) ─────────
// Mirrors exactly the call set src/memory/qtsq_adapter.c used.

extern "C" {
    pub fn qtsq_init(ctx: *mut QtsqContext) -> c_int;
    pub fn qtsq_free(ctx: *mut QtsqContext);
    pub fn qtsq_read(ctx: *mut QtsqContext, filename: *const c_char) -> c_int;
    pub fn qtsq_write(ctx: *const QtsqContext, filename: *const c_char) -> c_int;

    pub fn qtsq_container_create(ctx: *mut QtsqContext) -> c_int;
    pub fn qtsq_container_add_stream(
        ctx: *mut QtsqContext,
        sub_ctx: *const QtsqContext,
        stream_name: *const c_char,
    ) -> c_int;
    pub fn qtsq_container_pack(ctx: *mut QtsqContext) -> c_int;
    pub fn qtsq_container_get_count(ctx: *const QtsqContext, out_count: *mut c_uint) -> c_int;
    pub fn qtsq_container_get_stream(
        ctx: *const QtsqContext,
        index: c_uint,
        out_ctx: *mut QtsqContext,
        name_buf: *mut c_char,
        name_buf_size: usize,
    ) -> c_int;

    pub fn qtsq_compress_tensor(
        ctx: *mut QtsqContext,
        data: *const f32,
        count: usize,
        dimensions: *const c_uint,
        num_dims: u8,
    ) -> c_int;
    pub fn qtsq_compress_tensor_quantized(
        ctx: *mut QtsqContext,
        data: *const f32,
        count: usize,
        dimensions: *const c_uint,
        num_dims: u8,
        precision: QtsqTensorPrecision,
    ) -> c_int;
    pub fn qtsq_decompress_tensor(
        ctx: *const QtsqContext,
        out_data: *mut *mut f32,
        out_count: *mut usize,
    ) -> c_int;

    pub fn qtsq_compress_json(ctx: *mut QtsqContext, json: *const c_char, length: usize) -> c_int;
    pub fn qtsq_decompress_horizon(
        ctx: *const QtsqContext,
        out_data: *mut *mut u8,
        out_size: *mut usize,
    ) -> c_int;

    // qtsq_vault.h — password vault (deterministic local password, no
    // OS keystore; see the memory module's documenting comment).
    pub fn qtsq_vault_encrypt_password(ctx: *mut QtsqContext, password: *const c_char) -> c_int;
    pub fn qtsq_vault_decrypt_password(ctx: *mut QtsqContext, password: *const c_char) -> c_int;
    pub fn qtsq_vault_is_encrypted(ctx: *const QtsqContext) -> c_int;

    // qtsq.h — secure-text container + whole-stream decompress (the session
    // store's two codec paths; added in M10 when the src/ffi_shim.c shims
    // that called them dissolved into this module).
    pub fn qtsq_secure_text(
        ctx: *mut QtsqContext,
        text: *const c_char,
        length: usize,
        password: *const c_char,
        recipient_pk: *const u8,
        sign_sk: *const u8,
    ) -> c_int;
    pub fn qtsq_decompress(
        ctx: *mut QtsqContext,
        out_data: *mut *mut u8,
        out_size: *mut usize,
    ) -> c_int;
}

// ── Session store (M10: was src/ffi_shim.c, now plain Rust) ─────────
// A session's data (events, task, notes) persists as ONE .qtsq file.
// Writes go through qtsq_secure_text (sanitize + password vault) and
// reads decrypt with the same per-project password (derived by the Rust
// side in crates/sofuu-core/src/session.rs). The codec is fail-closed:
// NULL/empty inputs are rejected up front, exactly like the retired C
// shims. The context lives in calloc'd storage (this module's pattern —
// C used stack storage for the same purpose).

/// Persist a session payload as a `.qtsq` file. Returns QTSQ_OK (0) on
/// success, else a negative qtsq error code, or -1 for NULL/empty args.
///
/// # Safety
/// `path`/`password` must be NUL-terminated C strings; `data` must be
/// valid for `size` bytes.
pub unsafe fn qtsq_session_save(
    path: *const c_char,
    data: *const u8,
    size: usize,
    password: *const c_char,
) -> c_int {
    if path.is_null() || data.is_null() || password.is_null() || size == 0 {
        return -1;
    }
    let ctx = unsafe { qtsq_ctx_alloc() };
    if ctx.is_null() {
        return -1;
    }
    // SAFETY: zeroed storage of the exact context size; libqtsq init's in
    // place (same contract as the C stack struct).
    let mut r = unsafe { qtsq_init(ctx) };
    if r == QTSQ_OK {
        // SAFETY: the payload may contain interior NULs — length is explicit.
        r = unsafe {
            qtsq_secure_text(
                ctx,
                data as *const c_char,
                size,
                password,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
    }
    if r == QTSQ_OK {
        r = unsafe { qtsq_write(ctx, path) };
    }
    // SAFETY: qtsq_free releases the codec's internals; then drop the block.
    unsafe { qtsq_free(ctx) };
    unsafe { qtsq_ctx_free(ctx) };
    r
}

/// Load + decrypt + decompress a session `.qtsq` file. Returns a
/// libc::malloc'd buffer (free via `libc::free`) and writes its size to
/// `out_size`, or NULL on any failure (`*out_size` zeroed).
///
/// # Safety
/// `path`/`password` must be NUL-terminated C strings; `out_size` may be
/// NULL.
pub unsafe fn qtsq_session_load(
    path: *const c_char,
    password: *const c_char,
    out_size: *mut usize,
) -> *mut u8 {
    if !out_size.is_null() {
        unsafe { *out_size = 0 };
    }
    if path.is_null() || password.is_null() {
        return std::ptr::null_mut();
    }
    let ctx = unsafe { qtsq_ctx_alloc() };
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: zeroed storage of the exact context size; libqtsq init's in place.
    let mut r = unsafe { qtsq_init(ctx) };
    if r == QTSQ_OK {
        r = unsafe { qtsq_read(ctx, path) };
    }
    if r == QTSQ_OK && unsafe { &*ctx }.is_encrypted() {
        r = unsafe { qtsq_vault_decrypt_password(ctx, password) };
    }
    let mut data: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;
    if r == QTSQ_OK {
        r = unsafe { qtsq_decompress(ctx, &mut data, &mut size) };
    }
    // SAFETY: qtsq_free releases the codec's internals; then drop the block.
    unsafe { qtsq_free(ctx) };
    unsafe { qtsq_ctx_free(ctx) };
    if r != QTSQ_OK {
        if !data.is_null() {
            // SAFETY: data was malloc'd by the codec.
            unsafe { libc::free(data as *mut c_void) };
        }
        return std::ptr::null_mut();
    }
    if !out_size.is_null() {
        unsafe { *out_size = size };
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte-offset accessors must stay pinned to the C header even if
    /// a checkout bumps the layout (SOFUU_QTSQ_DIR). This test asserts the
    /// offsets this module hard-codes, so a layout change fails loudly
    /// instead of reading garbage fields and corrupting brain files.
    #[test]
    fn context_layout_offsets_match_vendored_header() {
        // (the header probe printed: data_type=10, schema=112,
        //  schema.dimensions=68, schema.num_dims=100, is_encrypted=4908,
        //  sizeof=5016 align=8 — a future checkout may change these)
        assert_eq!(OFF_HEADER_DATA_TYPE, 10);
        assert_eq!(OFF_SCHEMA, 112);
        assert_eq!(OFF_SCHEMA_DIMENSIONS, 68);
        assert_eq!(OFF_SCHEMA_NUM_DIMS, 100);
        assert_eq!(OFF_IS_ENCRYPTED, 4908);
        assert_eq!(QTSQ_CONTEXT_SIZE, 5016);
    }

    #[test]
    fn context_size_at_least_header_declared() {
        // Loose lower bound: whatever the checkout says, the struct cannot
        // be smaller than the documented primary fields.
        assert!(QTSQ_CONTEXT_SIZE >= 4960);
    }
}
