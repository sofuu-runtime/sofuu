// sofuu-ffi — QTSQ codec bindings (PLAN-RUST-MIGRATION M9).
//
// The memory subsystem (crates/sofuu-core/src/rt/memory.rs) persists CMA
// brain files + KV pages as .qtsq containers. The retired C adapter
// (src/memory/qtsq_adapter.c) declared `qtsq_context_t ctx;` on its stack
// and let the library initialize it in place; Rust cannot see the struct
// layout, so this module:
//   - allocates the context's exact storage (sizeof = 5024, align = 8 in
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

use std::ffi::{CStr, CString};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// ── Constants (qtsq_format.h) ──────────────────────────────────────

pub const QTSQ_OK: c_int = 0;
pub const QTSQ_ERR_NULL: c_int = -1;
pub const QTSQ_ERR_ALLOC: c_int = -2;
pub const QTSQ_ERR_FORMAT: c_int = -4;
pub const QTSQ_ERR_TYPE: c_int = -7;

/// Compress aggregate Sofuu records at 500 KiB (512,000 bytes). This is a
/// binary-size threshold, matching the existing 1024-based size convention.
const QTSQ_COMPRESS_THRESHOLD: usize = 500 * 1024;
const QTSQ_V1_HEADER_SIZE: usize = 80;
const QTSQ_FORMAT_VERSION_V1: u8 = 1;
const QTSQ_STRATEGY_NONE: u8 = 0;
const QTSQ_TYPE_JSON: u8 = 0x52;
const QTSQ_FLAG_CHECKSUMMED: u16 = 0x0004;
pub const QTSQ_TYPE_CONTAINER: u8 = 0xF0;

/// qtsq_format.h codec ids (qtsq_context_t.codec, Phase 6a/7).
pub const QTSQ_CODEC_ZLIB: u8 = 0; // legacy — read path only
pub const QTSQ_CODEC_QTC: u8 = 1; // the compressor/ rANS+LZ77 core

static QTSQ_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

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
pub const QTSQ_CONTEXT_SIZE: usize = 5024;

// Field offsets verified against qtsq_format.h with a C offsetof probe
// while porting M9 (the packed 112-byte qtsq_header_t leads the struct).
const OFF_HEADER_DATA_TYPE: usize = 10; // qtsq_header_t.data_type
const OFF_HEADER_STRATEGY: usize = 5; // qtsq_header_t.strategy
const OFF_SCHEMA: usize = 112; // qtsq_schema_t (after the packed header)
const OFF_SCHEMA_DIMENSIONS: usize = 68; // qtsq_schema_t.dimensions[8]
const OFF_SCHEMA_NUM_DIMS: usize = 100; // qtsq_schema_t.num_dims
const OFF_IS_ENCRYPTED: usize = 4908; // qtsq_context_t.is_encrypted (int)
const OFF_CODEC: usize = 5016; // qtsq_context_t.codec (uint8, Phase 6a)
const OFF_QUALITY: usize = 5020; // qtsq_context_t.quality (float, after codec)

// Compile-time sanity net for the layout facts above.
const _: () = assert!(QTSQ_CONTEXT_SIZE >= OFF_CODEC + 4);

impl QtsqContext {
    /// # Safety
    /// `self` must be a live context (initialized by `qtsq_init`/`qtsq_read`).
    pub unsafe fn data_type(&self) -> u8 {
        let base = self as *const QtsqContext as *const u8;
        std::ptr::read_unaligned(base.add(OFF_HEADER_DATA_TYPE))
    }

    /// # Safety
    /// `self` must be a live context initialized by `qtsq_read`.
    pub unsafe fn strategy(&self) -> u8 {
        let base = self as *const QtsqContext as *const u8;
        std::ptr::read_unaligned(base.add(OFF_HEADER_STRATEGY))
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

    /// Force the container codec on a not-yet-written context. The calloc'd
    /// storage zeroes `codec` to QTSQ_CODEC_ZLIB (0) — the Phase-6a default —
    /// but the Phase-7 checkout's zlib WRITE path produces containers the
    /// read path cannot parse back (verified empirically 2026-08-30: small
    /// payloads happen to ride qtc and round-trip; anything the library
    /// routes to zlib writes an unloadable file). Every sofuu write pins
    /// QTSQ_CODEC_QTC (1) — the new compressor; zlib stays read-only for
    /// legacy files, which qtsq_decompress still auto-detects by magic.
    ///
    /// # Safety
    /// `self` must point to a context-sized block (calloc'd or initialized).
    pub unsafe fn set_codec(&mut self, codec: u8) {
        let base = self as *mut QtsqContext as *mut u8;
        std::ptr::write_unaligned(base.add(OFF_CODEC), codec);
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
    pub fn qtsq_write_plaintext(ctx: *const QtsqContext, filename: *const c_char) -> c_int;
    pub fn qtsq_sha256(data: *const u8, size: usize, hash: *mut u8);

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
// New Sofuu session/conversation writes use qtsq_write_plaintext: they are
// local application records and do not need QTSQ's media-vault encryption.
// Reads still accept the legacy encrypted files and decrypt them with the
// same per-project password (derived by the Rust side in
// crates/sofuu-core/src/session.rs).
//
// The session store intentionally does not use qtsq_secure_text or the vault:
// those APIs add the media-format security envelope, which is unnecessary for
// Sofuu's local cache records and dominates the size of small events. Below
// 500 KiB the session path stores bytes directly with a compact raw QTSQ
// header; at or above 500 KiB it uses compress_json plus the explicit
// plaintext writer.
//
// NULL/empty inputs are rejected up front. If the codec cannot represent a
// valid large payload (for example, an individual token exceeds its
// dictionary limits), the session path falls back to the raw QTSQ writer
// after cleaning up the failed compression attempt. This keeps persistence
// durable without pretending that a partial compressed stream is valid.
// The context lives in calloc'd storage (this module's pattern — C used
// stack storage for the same purpose).

/// Build the smallest standard QTSQ header that can represent a local raw
/// session record. The v1 header is sufficient below the compression
/// threshold because all sizes fit in u32; qtsq_read upgrades it in memory.
fn raw_session_header(data: &[u8]) -> Option<[u8; QTSQ_V1_HEADER_SIZE]> {
    let size = u32::try_from(data.len()).ok()?;
    let mut header = [0u8; QTSQ_V1_HEADER_SIZE];
    header[0..4].copy_from_slice(b"QTS\x01");
    header[4] = QTSQ_FORMAT_VERSION_V1;
    header[5] = QTSQ_STRATEGY_NONE;
    header[6..8].copy_from_slice(&QTSQ_FLAG_CHECKSUMMED.to_le_bytes());
    header[8] = QTSQ_TYPE_JSON;
    header[16..24].copy_from_slice(&(data.len() as u64).to_le_bytes());
    // The v1 header stores the checksum at [32..64].
    unsafe { qtsq_sha256(data.as_ptr(), data.len(), header[32..64].as_mut_ptr()) };
    header[72..76].copy_from_slice(&size.to_le_bytes());
    Some(header)
}

fn unique_temp_path(target: &Path, tag: &str) -> PathBuf {
    let counter = QTSQ_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "qtsq".into());
    target.with_file_name(format!(
        ".{name}.{tag}.{}.{}.tmp",
        std::process::id(),
        counter
    ))
}

fn sync_file(path: &Path) -> bool {
    fs::OpenOptions::new()
        .read(true)
        .open(path)
        .and_then(|file| file.sync_all())
        .is_ok()
}

fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(file) = fs::OpenOptions::new().read(true).open(parent) {
            let _ = file.sync_all();
        }
    }
}

/// Atomically write an uncompressed, unencrypted QTSQ session record.
///
/// This is deliberately in Sofuu: QTSQ already defines strategy NONE and
/// qtsq_read/qtsq_decompress support it, but the public QTSQ API has no setter
/// for attaching an existing byte slice without invoking a compressor.
fn write_raw_session_qtsq(path: &CStr, data: &[u8]) -> c_int {
    let Ok(path_str) = path.to_str() else { return -1 };
    let target = Path::new(path_str);
    let mut temp = unique_temp_path(target, "raw");
    let Some(header) = raw_session_header(data) else { return -1 };

    let result = (|| -> std::io::Result<()> {
        // P3 (AUDIT-2026-09-07): O_EXCL semantics — create_new refuses to
        // follow a pre-planted symlink or truncate a pre-existing file at
        // the (vanishingly rare) pid+counter collision; a collision instead
        // regenerates the name and retries, bounded.
        let mut attempts = 0;
        let file = loop {
            attempts += 1;
            match OpenOptions::new().write(true).create_new(true).open(&temp) {
                Ok(f) => break f,
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::AlreadyExists && attempts < 4 =>
                {
                    temp = unique_temp_path(target, "raw");
                }
                Err(e) => return Err(e),
            }
        };
        let mut file = file;
        file.write_all(&header)?;
        file.write_all(data)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, target)?;
        sync_parent(target);
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temp);
        return -1;
    }
    QTSQ_OK
}

/// Persist an unencrypted session payload as a `.qtsq` file. Payloads below
/// 500 KiB are stored as-is; larger payloads use QTSQ compression. Returns
/// QTSQ_OK (0) on success, else a negative qtsq error code, or -1 for
/// NULL/empty args.
///
/// # Safety
/// `path` must be a NUL-terminated C string; `data` must be valid for `size`
/// bytes.
pub unsafe fn qtsq_session_save(
    path: *const c_char,
    data: *const u8,
    size: usize,
) -> c_int {
    if path.is_null() || data.is_null() || size == 0 {
        return -1;
    }
    let path_cstr = unsafe { CStr::from_ptr(path) };
    let bytes = unsafe { std::slice::from_raw_parts(data, size) };
    // Tiny session records are not worth sending through a compressor: the
    // compressor's framing/dictionary overhead can exceed the payload. Keep
    // the exact bytes in a compact, valid raw QTSQ record instead.
    if size < QTSQ_COMPRESS_THRESHOLD {
        return write_raw_session_qtsq(path_cstr, bytes);
    }
    let ctx = unsafe { qtsq_ctx_alloc() };
    if ctx.is_null() {
        return write_raw_session_qtsq(path_cstr, bytes);
    }
    // SAFETY: zeroed storage of the exact context size; libqtsq init's in
    // place (same contract as the C stack struct).
    let mut r = unsafe { qtsq_init(ctx) };
    let mut generated: Option<(PathBuf, PathBuf)> = None;
    if r == QTSQ_OK {
        // Phase 7: qtc is the write codec (see the module comment). Set
        // AFTER qtsq_init (init may reset fields) and BEFORE compressing.
        unsafe { (*ctx).set_codec(QTSQ_CODEC_QTC) };
        // SAFETY: qtsq_compress_json receives an explicit length and does not
        // require a trailing NUL; avoid copying a multi-megabyte payload.
        r = unsafe { qtsq_compress_json(ctx, data as *const c_char, size) };
    }
    if r == QTSQ_OK {
        // Explicit local-cache path: no security block, no password-vault
        // expansion. qtsq_write() remains encrypted/fail-closed for all
        // ordinary QTSQ callers.
        let Ok(path_str) = path_cstr.to_str() else {
            r = -1;
            unsafe { qtsq_free(ctx) };
            unsafe { qtsq_ctx_free(ctx) };
            return r;
        };
        let target = Path::new(path_str).to_path_buf();
        let mut temp = unique_temp_path(&target, "compressed");
        // P3 (AUDIT-2026-09-07): the C writer opens its temp with
        // O_CREAT|O_TRUNC — no O_EXCL is available on this path — so at
        // least never reuse a pre-existing name: regenerate until fresh
        // (bounded). A planted/symlinked temp is thus not followed unless
        // it is created inside the tiny post-check window.
        let mut attempts = 0;
        while temp.exists() && attempts < 4 {
            attempts += 1;
            temp = unique_temp_path(&target, "compressed");
        }
        let Ok(temp_cstr) = CString::new(temp.to_string_lossy().as_bytes()) else {
            r = -1;
            unsafe { qtsq_free(ctx) };
            unsafe { qtsq_ctx_free(ctx) };
            return r;
        };
        generated = Some((temp.clone(), target));
        r = unsafe { qtsq_write_plaintext(ctx, temp_cstr.as_ptr()) };
    }
    // SAFETY: qtsq_free releases the codec's internals; then drop the block.
    unsafe { qtsq_free(ctx) };
    unsafe { qtsq_ctx_free(ctx) };
    if r == QTSQ_OK {
        if let Some((temp, target)) = generated {
            if !sync_file(&temp) || fs::rename(&temp, &target).is_err() {
                let _ = fs::remove_file(&temp);
                r = -1;
            } else {
                sync_parent(&target);
            }
        }
    } else if let Some((temp, _)) = generated {
        let _ = fs::remove_file(temp);
    }
    // A large payload is eligible for compression, but QTSQ's dictionary
    // codec may reject otherwise valid JSON with an unrepresentable token.
    // Store the original bytes as a raw, unencrypted QTSQ record in that
    // case. The fallback is deliberately after qtsq_free/qtsq_ctx_free and
    // temporary-file cleanup, so callers never observe a half-written file.
    if r != QTSQ_OK && size <= u32::MAX as usize {
        let fallback = write_raw_session_qtsq(path_cstr, bytes);
        if fallback == QTSQ_OK {
            return fallback;
        }
    }
    r
}

/// Load, optionally decrypt a legacy encrypted file, and decompress a session
/// `.qtsq` file. Returns a
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
        // Raw records use strategy NONE and must bypass the dictionary
        // decoder. Compressed/current and legacy records retain the previous
        // horizon-first compatibility path.
        if unsafe { (&*ctx).strategy() } == QTSQ_STRATEGY_NONE {
            r = unsafe { qtsq_decompress(ctx, &mut data, &mut size) };
        } else {
            // Two generations of session containers:
            //   • current: compress_json stream → decompress_horizon
            //   • legacy (pre-2026-08, secure_text-written): whole-stream
            //     qtsq_decompress. Try the current one first, fall back.
            r = unsafe { qtsq_decompress_horizon(ctx, &mut data, &mut size) };
            if r != QTSQ_OK {
                r = unsafe { qtsq_decompress(ctx, &mut data, &mut size) };
            }
        }
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

/// Read only the encryption bit from a QTSQ session record. This is used by
/// Sofuu diagnostics to distinguish new plaintext records from legacy
/// encrypted records without exposing payload text or attempting a write.
///
/// # Safety
/// `path` must be a NUL-terminated C string.
pub unsafe fn qtsq_session_is_encrypted(path: *const c_char) -> Option<bool> {
    if path.is_null() {
        return None;
    }
    let ctx = unsafe { qtsq_ctx_alloc() };
    if ctx.is_null() {
        return None;
    }
    let mut r = unsafe { qtsq_init(ctx) };
    if r == QTSQ_OK {
        r = unsafe { qtsq_read(ctx, path) };
    }
    let encrypted = if r == QTSQ_OK {
        Some(unsafe { (&*ctx).is_encrypted() })
    } else {
        None
    };
    unsafe { qtsq_free(ctx) };
    unsafe { qtsq_ctx_free(ctx) };
    encrypted
}

/// Return a lowercase SHA-256 digest for a byte slice using the QTSQ runtime's
/// already-linked hash implementation.
pub fn sha256_hex(data: &[u8]) -> Option<String> {
    if data.is_empty() {
        return None;
    }
    let mut hash = [0u8; 32];
    unsafe { qtsq_sha256(data.as_ptr(), data.len(), hash.as_mut_ptr()) };
    Some(hash.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte-offset accessors must stay pinned to the C header even if
    /// a checkout bumps the layout (SOFUU_QTSQ_DIR). This test guards the
    /// Rust-side constants against accidental edits; the REAL check against
    /// the live checkout is the build-time `_Static_assert` guard that
    /// sofuu-ffi/build.rs compiles against qtsq_format.h — a checkout that
    /// changes the layout fails the build, not just this module's tests.
    #[test]
    fn context_layout_offsets_match_vendored_header() {
        // (the header probe printed: data_type=10, schema=112,
        //  schema.dimensions=68, schema.num_dims=100, is_encrypted=4908,
        //  codec=5016, quality=5020, sizeof=5024 align=8 — a future
        //  checkout may change these)
        assert_eq!(OFF_HEADER_DATA_TYPE, 10);
        assert_eq!(OFF_HEADER_STRATEGY, 5);
        assert_eq!(OFF_SCHEMA, 112);
        assert_eq!(OFF_SCHEMA_DIMENSIONS, 68);
        assert_eq!(OFF_SCHEMA_NUM_DIMS, 100);
        assert_eq!(OFF_IS_ENCRYPTED, 4908);
        assert_eq!(OFF_CODEC, 5016);
        assert_eq!(OFF_QUALITY, 5020);
        assert_eq!(QTSQ_CONTEXT_SIZE, 5024);
    }

    #[test]
    fn context_size_at_least_header_declared() {
        // Loose lower bound: whatever the checkout says, the struct cannot
        // be smaller than the documented primary fields.
        assert!(QTSQ_CONTEXT_SIZE >= 4960);
    }

    #[test]
    fn raw_session_header_is_compact_and_unencrypted() {
        let data = b"{\"schema\":1}";
        let header = raw_session_header(data).expect("small raw header");
        assert_eq!(&header[0..4], b"QTS\x01");
        assert_eq!(header[4], QTSQ_FORMAT_VERSION_V1);
        assert_eq!(header[5], QTSQ_STRATEGY_NONE);
        assert_eq!(u16::from_le_bytes([header[6], header[7]]), QTSQ_FLAG_CHECKSUMMED);
        assert_eq!(header[8], QTSQ_TYPE_JSON);
        assert_eq!(u64::from_le_bytes(header[16..24].try_into().unwrap()), data.len() as u64);
        assert_eq!(u32::from_le_bytes(header[72..76].try_into().unwrap()), data.len() as u32);
    }

    #[test]
    fn compression_starts_at_500_kib() {
        let path = std::env::temp_dir().join(format!(
            "sofuu-qtsq-threshold-{}-{}.qtsq",
            std::process::id(),
            QTSQ_COMPRESS_THRESHOLD
        ));
        let _ = std::fs::remove_file(&path);
        // Use many short JSON tokens instead of one giant string token: the
        // dictionary codec intentionally rejects individual words > u16::MAX.
        let mut data = Vec::with_capacity(QTSQ_COMPRESS_THRESHOLD);
        data.push(b'[');
        let items = (QTSQ_COMPRESS_THRESHOLD - 4) / 2;
        for i in 0..items {
            if i != 0 {
                data.push(b',');
            }
            data.push(b'0');
        }
        data.extend_from_slice(b",[]]");
        assert_eq!(data.len(), QTSQ_COMPRESS_THRESHOLD);
        let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();

        let rc = unsafe { qtsq_session_save(c_path.as_ptr(), data.as_ptr(), data.len()) };
        assert_eq!(rc, QTSQ_OK);
        let stored = std::fs::read(&path).expect("read threshold record");
        assert_eq!(stored.get(4), Some(&3), "500 KiB record should use the current QTSQ writer");
        assert_ne!(stored.get(5), Some(&QTSQ_STRATEGY_NONE), "500 KiB record should use compression");
        assert!(stored.len() < data.len(), "repeated 500 KiB record should compress");
        assert_eq!(unsafe { qtsq_session_is_encrypted(c_path.as_ptr()) }, Some(false));

        let password = std::ffi::CString::new("").unwrap();
        let mut out_size = 0usize;
        let ptr = unsafe { qtsq_session_load(c_path.as_ptr(), password.as_ptr(), &mut out_size) };
        assert!(!ptr.is_null(), "threshold record should round-trip");
        let loaded = unsafe { std::slice::from_raw_parts(ptr, out_size) }.to_vec();
        unsafe { libc::free(ptr as *mut c_void) };
        assert_eq!(loaded, data);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn one_byte_below_threshold_stays_raw() {
        let path = std::env::temp_dir().join(format!(
            "sofuu-qtsq-raw-boundary-{}-{}.qtsq",
            std::process::id(),
            QTSQ_COMPRESS_THRESHOLD - 1
        ));
        let _ = std::fs::remove_file(&path);
        let data = vec![b'x'; QTSQ_COMPRESS_THRESHOLD - 1];
        let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();

        let rc = unsafe { qtsq_session_save(c_path.as_ptr(), data.as_ptr(), data.len()) };
        assert_eq!(rc, QTSQ_OK);
        let stored = std::fs::read(&path).expect("read raw boundary record");
        assert_eq!(stored.get(4), Some(&QTSQ_FORMAT_VERSION_V1));
        assert_eq!(stored.get(5), Some(&QTSQ_STRATEGY_NONE));
        assert_eq!(unsafe { qtsq_session_is_encrypted(c_path.as_ptr()) }, Some(false));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unsupported_large_payload_falls_back_to_raw() {
        let path = std::env::temp_dir().join(format!(
            "sofuu-qtsq-large-fallback-{}-{}.qtsq",
            std::process::id(),
            QTSQ_COMPRESS_THRESHOLD
        ));
        let _ = std::fs::remove_file(&path);
        // A single dictionary token larger than the codec's u16 token limit
        // is valid JSON but not representable by the current compressor.
        let data = format!("{{\"payload\":\"{}\"}}", "x".repeat(QTSQ_COMPRESS_THRESHOLD));
        let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();

        let rc = unsafe { qtsq_session_save(c_path.as_ptr(), data.as_ptr(), data.len()) };
        assert_eq!(rc, QTSQ_OK, "large unsupported payload must remain durable");
        let stored = std::fs::read(&path).expect("read fallback record");
        assert_eq!(stored.get(4), Some(&QTSQ_FORMAT_VERSION_V1));
        assert_eq!(stored.get(5), Some(&QTSQ_STRATEGY_NONE));
        assert_eq!(unsafe { qtsq_session_is_encrypted(c_path.as_ptr()) }, Some(false));

        let password = std::ffi::CString::new("").unwrap();
        let mut out_size = 0usize;
        let ptr = unsafe { qtsq_session_load(c_path.as_ptr(), password.as_ptr(), &mut out_size) };
        assert!(!ptr.is_null(), "fallback record should round-trip");
        let loaded = unsafe { std::slice::from_raw_parts(ptr, out_size) }.to_vec();
        unsafe { libc::free(ptr as *mut c_void) };
        assert_eq!(loaded, data.as_bytes());
        let _ = std::fs::remove_file(&path);
    }
}
