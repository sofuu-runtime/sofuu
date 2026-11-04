// crates/sofuu-core/src/infer/cuda.rs — safe Rust surface over the runtime
// CUDA driver loader (PLAN-GPU-CUDA C0/C1).
//
// Design constraints from the plan:
//   * no CUDA toolkit at build time, no libcudart at runtime;
//   * the driver is dlopen()ed through src/cuda/cuda_loader.c;
//   * the kernel module is checked-in PTX, inflated here with flate2 (Rust)
//     rather than zlib (musl cross builds have no system libz);
//   * every failure degrades to None / Err — callers keep the CPU engine.
//
// Platform policy: the real backend is Linux. On macOS this module still
// compiles, but the release build references no loader symbols, so the C
// object is not linked and the tight macOS binary budget is unaffected.
// Tests on either OS exercise the loader's no-driver path directly.

#[allow(unused_imports)]
use std::ffi::{CStr, CString};
#[allow(unused_imports)]
use std::os::raw::{c_char, c_int, c_void};
#[allow(unused_imports)]
use std::ptr;
/// Every kernel name the checked-in PTX must export. Kept in one place so the
/// generator script, the Rust init check and the tests cannot drift.
pub const PTX_ENTRIES: [&str; 18] = [
    "sofuu_gemv_q8_0",
    "sofuu_gemv_q4_k",
    "sofuu_gemv_q6_k",
    "sofuu_gemv_f32",
    "sofuu_rmsnorm",
    "sofuu_swiglu",
    "sofuu_rope_f32",
    "sofuu_kv_write_f32",
    "sofuu_attn_decode_f32",
    "sofuu_add_f32",
    "sofuu_gemm_q4_k_b8",
    "sofuu_gemm_q6_k_b8",
    "sofuu_gemm_q8_0_b8",
    "sofuu_gemm_f32_b8",
    "sofuu_rmsnorm_batch",
    "sofuu_rope_batch",
    "sofuu_kv_write_batch",
    "sofuu_attn_prefill_f32",
];

/// Weight layouts the C1 kernels cover. `block_bytes` is the GGML on-disk
/// block size; `qk` is the number of values per block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    F32,
    Q8_0,
    Q4K,
    Q6K,
}

impl Quant {
    pub fn entry(self) -> &'static str {
        match self {
            Quant::F32 => "sofuu_gemv_f32",
            Quant::Q8_0 => "sofuu_gemv_q8_0",
            Quant::Q4K => "sofuu_gemv_q4_k",
            Quant::Q6K => "sofuu_gemv_q6_k",
        }
    }

    /// Batched-prefill kernel: one warp per token, eight tokens per weight
    /// read. Keeping the name beside `entry` means a new quant cannot be added
    /// to one path and forgotten in the other.
    pub fn batch_entry(self) -> &'static str {
        match self {
            Quant::F32 => "sofuu_gemm_f32_b8",
            Quant::Q8_0 => "sofuu_gemm_q8_0_b8",
            Quant::Q4K => "sofuu_gemm_q4_k_b8",
            Quant::Q6K => "sofuu_gemm_q6_k_b8",
        }
    }

    /// Values per quant block (1 for f32).
    pub fn qk(self) -> usize {
        match self {
            Quant::F32 => 1,
            Quant::Q8_0 => 32,
            Quant::Q4K | Quant::Q6K => 256,
        }
    }

    /// Bytes per quant block (4 for f32).
    pub fn block_bytes(self) -> usize {
        match self {
            Quant::F32 => 4,
            Quant::Q8_0 => 34,
            Quant::Q4K => 144,
            Quant::Q6K => 210,
        }
    }

    /// True when `cols` is a legal row width for this layout.
    pub fn accepts_cols(self, cols: usize) -> bool {
        cols != 0 && cols % self.qk() == 0
    }

    pub fn row_bytes(self, cols: usize) -> Option<usize> {
        if !self.accepts_cols(cols) {
            return None;
        }
        Some((cols / self.qk()) * self.block_bytes())
    }
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub total_memory: u64,
    pub integrated: bool,
    pub unified_addressing: bool,
    pub driver_version: i32,
    pub library: String,
}

#[derive(Clone, Debug)]
pub struct CudaError(pub String);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CudaError {}

/// A device allocation. `ptr` is a CUdeviceptr value, not a host pointer.
#[derive(Clone, Copy, Debug)]
pub struct DeviceBuf {
    pub ptr: u64,
    pub bytes: usize,
}

/// Validate a KV-cache write before it reaches the driver.
///
/// The kernel stores at `(head * cap + pos) * head_dim` with no bounds check of
/// its own, so a position past the end of the cache silently writes over
/// whatever CUDA allocated next. That is unrecoverable corruption of a model
/// mid-generation, and it is cheap to refuse here: we know the device buffer
/// sizes because we allocated them.
#[allow(clippy::too_many_arguments)]
pub fn validate_kv_write(
    pos: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cap: usize,
    k_cache_bytes: usize,
    v_cache_bytes: usize,
    k_bytes: usize,
    v_bytes: usize,
) -> Result<(), CudaError> {
    if n_kv_heads == 0 || head_dim == 0 || cap == 0 {
        return Err(CudaError(
            "kv_write needs non-zero n_kv_heads, head_dim and cap".into(),
        ));
    }
    if pos >= cap {
        return Err(CudaError(format!(
            "kv_write position {pos} is outside a cache of {cap} rows"
        )));
    }
    let bytes = std::mem::size_of::<f32>();
    let need = n_kv_heads
        .checked_mul(cap)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(bytes))
        .ok_or_else(|| CudaError("kv_write cache dimensions overflow".into()))?;
    if k_cache_bytes < need || v_cache_bytes < need {
        return Err(CudaError(format!(
            "kv_write needs {need} bytes per cache ({n_kv_heads} heads x {cap} rows x              {head_dim} dims), got {k_cache_bytes} and {v_cache_bytes}"
        )));
    }
    let src = n_kv_heads
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(bytes))
        .ok_or_else(|| CudaError("kv_write source dimensions overflow".into()))?;
    if k_bytes < src || v_bytes < src {
        return Err(CudaError(format!(
            "kv_write needs {src} bytes per source row set, got {k_bytes} and {v_bytes}"
        )));
    }
    Ok(())
}

/// Validate a decode-attention launch. The kernel returns early on bad shapes
/// (leaving `out` untouched), so a caller that skips this check gets stale
/// activations rather than an error — check here, where the shapes are known.
#[allow(clippy::too_many_arguments)]
pub fn validate_attn_decode(
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    cap: usize,
    q_bytes: usize,
    k_bytes: usize,
    v_bytes: usize,
    out_bytes: usize,
) -> Result<(), CudaError> {
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || cap == 0 {
        return Err(CudaError(
            "attn_decode needs non-zero n_heads, n_kv_heads, head_dim and cap".into(),
        ));
    }
    if head_dim > 256 {
        return Err(CudaError(format!(
            "attn_decode head_dim {head_dim} exceeds the 256-thread block"
        )));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(CudaError(format!(
            "attn_decode n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"
        )));
    }
    if seq_len == 0 || seq_len > cap {
        return Err(CudaError(format!(
            "attn_decode seq_len {seq_len} is outside a cache of {cap} rows"
        )));
    }
    let f32b = std::mem::size_of::<f32>();
    let q_need = n_heads
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(f32b))
        .ok_or_else(|| CudaError("attn_decode q shape overflow".into()))?;
    let cache_need = n_kv_heads
        .checked_mul(cap)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(f32b))
        .ok_or_else(|| CudaError("attn_decode cache shape overflow".into()))?;
    if q_bytes < q_need || out_bytes < q_need {
        return Err(CudaError(format!(
            "attn_decode needs {q_need} bytes for q/out, got {q_bytes} and {out_bytes}"
        )));
    }
    if k_bytes < cache_need || v_bytes < cache_need {
        return Err(CudaError(format!(
            "attn_decode needs {cache_need} bytes per KV cache, got {k_bytes} and {v_bytes}"
        )));
    }
    Ok(())
}

/// Validate a batched-prefill GEMV. `x` holds `n_tokens` rows of `cols`
/// activations and `y` holds `n_tokens` rows of `rows` outputs, both
/// token-major; the kernel writes `y[t * rows + row]`.
#[allow(clippy::too_many_arguments)]
pub fn validate_gemm_b8(
    quant: Quant,
    rows: usize,
    cols: usize,
    n_tokens: usize,
    row_tile: usize,
    w_bytes: usize,
    x_bytes: usize,
    y_bytes: usize,
) -> Result<(), CudaError> {
    if rows == 0 || cols == 0 {
        return Err(CudaError("gemm_b8 needs non-zero rows and cols".into()));
    }
    // Activations are loaded as aligned float4s; a row width that is not a
    // multiple of four would misalign every token row after the first.
    if !cols.is_multiple_of(4) {
        return Err(CudaError(format!(
            "gemm_b8 cols {cols} is not a multiple of 4 (aligned float4 rows)"
        )));
    }
    if n_tokens == 0 || n_tokens > 8 {
        return Err(CudaError(format!(
            "gemm_b8 n_tokens {n_tokens} is outside 1..=8 (one warp per token)"
        )));
    }
    if row_tile == 0 || row_tile > 32 || !row_tile.is_multiple_of(8) {
        return Err(CudaError(format!(
            "gemm_b8 row_tile {row_tile} must be a multiple of 8 in 8..=32"
        )));
    }
    let Some(row_bytes) = quant.row_bytes(cols) else {
        return Err(CudaError(format!(
            "{} does not accept cols={cols}",
            quant.batch_entry()
        )));
    };
    let need_w = row_bytes
        .checked_mul(rows)
        .ok_or_else(|| CudaError("gemm_b8 weight size overflow".into()))?;
    let need_x = n_tokens
        .checked_mul(cols)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| CudaError("gemm_b8 x size overflow".into()))?;
    let need_y = n_tokens
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| CudaError("gemm_b8 y size overflow".into()))?;
    if w_bytes < need_w || x_bytes < need_x || y_bytes < need_y {
        return Err(CudaError(format!(
            "gemm_b8 buffers too small: w {w_bytes}<{need_w}, x {x_bytes}<{need_x}, y {y_bytes}<{need_y}"
        )));
    }
    Ok(())
}

/// Validate a batched KV write for `n_tokens` consecutive positions.
#[allow(clippy::too_many_arguments)]
pub fn validate_kv_write_batch(
    start_pos: usize,
    n_tokens: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cap: usize,
    k_cache_bytes: usize,
    v_cache_bytes: usize,
    k_bytes: usize,
    v_bytes: usize,
) -> Result<(), CudaError> {
    if n_tokens == 0 || n_kv_heads == 0 || head_dim == 0 || cap == 0 {
        return Err(CudaError(
            "kv_write_batch needs non-zero tokens, heads, head_dim and cap".into(),
        ));
    }
    if start_pos
        .checked_add(n_tokens)
        .is_none_or(|end| end > cap)
    {
        return Err(CudaError(format!(
            "kv_write_batch positions {start_pos}..{} leave a cache of {cap} rows",
            start_pos + n_tokens
        )));
    }
    let bytes = std::mem::size_of::<f32>();
    let cache_need = n_kv_heads
        .checked_mul(cap)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(bytes))
        .ok_or_else(|| CudaError("kv_write_batch cache shape overflow".into()))?;
    if k_cache_bytes < cache_need || v_cache_bytes < cache_need {
        return Err(CudaError(format!(
            "kv_write_batch needs {cache_need} bytes per cache, got {k_cache_bytes} and {v_cache_bytes}"
        )));
    }
    let src = n_tokens
        .checked_mul(n_kv_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(bytes))
        .ok_or_else(|| CudaError("kv_write_batch source shape overflow".into()))?;
    if k_bytes < src || v_bytes < src {
        return Err(CudaError(format!(
            "kv_write_batch needs {src} bytes per source batch, got {k_bytes} and {v_bytes}"
        )));
    }
    Ok(())
}

/// Validate the batched causal-attention launch. Every token in the chunk
/// attends over at most `start_pos + n_tokens` rows, so that is the bound the
/// cache must hold.
#[allow(clippy::too_many_arguments)]
pub fn validate_attn_prefill(
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    n_tokens: usize,
    cap: usize,
    q_bytes: usize,
    k_bytes: usize,
    v_bytes: usize,
    out_bytes: usize,
) -> Result<(), CudaError> {
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || cap == 0 || n_tokens == 0 {
        return Err(CudaError(
            "attn_prefill needs non-zero heads, head_dim, cap and tokens".into(),
        ));
    }
    if head_dim > 256 {
        return Err(CudaError(format!(
            "attn_prefill head_dim {head_dim} exceeds the 256-thread block"
        )));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(CudaError(format!(
            "attn_prefill n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"
        )));
    }
    if start_pos
        .checked_add(n_tokens)
        .is_none_or(|end| end > cap)
    {
        return Err(CudaError(format!(
            "attn_prefill chunk {start_pos}..{} leaves a cache of {cap} rows",
            start_pos + n_tokens
        )));
    }
    let f32b = std::mem::size_of::<f32>();
    let q_need = n_tokens
        .checked_mul(n_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(f32b))
        .ok_or_else(|| CudaError("attn_prefill q shape overflow".into()))?;
    let cache_need = n_kv_heads
        .checked_mul(cap)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(f32b))
        .ok_or_else(|| CudaError("attn_prefill cache shape overflow".into()))?;
    if q_bytes < q_need || out_bytes < q_need {
        return Err(CudaError(format!(
            "attn_prefill needs {q_need} bytes for q/out, got {q_bytes} and {out_bytes}"
        )));
    }
    if k_bytes < cache_need || v_bytes < cache_need {
        return Err(CudaError(format!(
            "attn_prefill needs {cache_need} bytes per KV cache, got {k_bytes} and {v_bytes}"
        )));
    }
    Ok(())
}

/// One reader for SOFUU_CUDA_DEBUG: the loader owns it, this asks. Two parsers
/// of one flag is two behaviours waiting to disagree.
#[cfg(any(target_os = "linux", all(test, target_os = "macos")))]
fn debug_on() -> bool {
    // SAFETY: takes no arguments and returns an int; the loader only reads the
    // environment, so this is safe before any driver is opened.
    unsafe { sys::sofuu_cuda_debug_enabled() != 0 }
}

/// Platforms that never link the loader have no CUDA to trace.
#[cfg(not(any(target_os = "linux", all(test, target_os = "macos"))))]
fn debug_on() -> bool {
    false
}

// Dead on macOS by construction: every caller below is inside a block that
// only runs when a CUDA driver was opened, which cannot happen off Linux.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn kill_switch() -> bool {
    matches!(
        std::env::var("SOFUU_CUDA").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn debug(msg: impl std::fmt::Display) {
    if debug_on() {
        eprintln!("[cuda] {msg}");
    }
}

// The loader is linked on Linux in every build, and on macOS only in test
// binaries (a macOS release has no CUDA consumer, so the object stays out of
// the binary and the 5.5 MiB budget is unchanged).
#[cfg(any(target_os = "linux", all(test, target_os = "macos")))]
mod sys {
    use super::*;

    extern "C" {
        pub fn sofuu_cuda_driver_open(path: *const c_char) -> *mut c_void;
        pub fn sofuu_cuda_driver_close(driver: *mut c_void);
        pub fn sofuu_cuda_driver_path(driver: *const c_void) -> *const c_char;
        pub fn sofuu_cuda_driver_version(driver: *const c_void) -> c_int;
        pub fn sofuu_cuda_driver_device_count(driver: *const c_void) -> c_int;
        pub fn sofuu_cuda_driver_device_info(
            driver: *const c_void,
            ordinal: c_int,
            name: *mut c_char,
            name_cap: usize,
            cc_major: *mut c_int,
            cc_minor: *mut c_int,
            total_mem: *mut u64,
            integrated: *mut c_int,
            unified: *mut c_int,
        ) -> c_int;
        pub fn sofuu_cuda_ctx_create(driver: *mut c_void, ordinal: c_int) -> *mut c_void;
        pub fn sofuu_cuda_ctx_destroy(ctx: *mut c_void);
        pub fn sofuu_cuda_ctx_load_ptx(ctx: *mut c_void, ptx: *const c_char) -> c_int;
        pub fn sofuu_cuda_ctx_function(
            ctx: *mut c_void,
            name: *const c_char,
            out: *mut *mut c_void,
        ) -> c_int;
        pub fn sofuu_cuda_ctx_alloc(ctx: *mut c_void, bytes: u64, out: *mut u64) -> c_int;
        pub fn sofuu_cuda_ctx_free(ctx: *mut c_void, ptr: u64) -> c_int;
        pub fn sofuu_cuda_ctx_upload(
            ctx: *mut c_void,
            dst: u64,
            src: *const c_void,
            bytes: usize,
        ) -> c_int;
        pub fn sofuu_cuda_ctx_download(
            ctx: *mut c_void,
            dst: *mut c_void,
            src: u64,
            bytes: usize,
        ) -> c_int;
        pub fn sofuu_cuda_ctx_launch(
            ctx: *mut c_void,
            function: *mut c_void,
            gx: u32,
            gy: u32,
            gz: u32,
            bx: u32,
            by: u32,
            bz: u32,
            shared: u32,
            params: *mut *mut c_void,
        ) -> c_int;
        pub fn sofuu_cuda_ctx_sync(ctx: *mut c_void) -> c_int;
        pub fn sofuu_cuda_ctx_host_register(
            ctx: *mut c_void,
            host: *mut c_void,
            bytes: usize,
            out_dev: *mut u64,
        ) -> c_int;
        pub fn sofuu_cuda_ctx_host_unregister(ctx: *mut c_void, host: *mut c_void) -> c_int;
        pub fn sofuu_cuda_last_error() -> *const c_char;
        pub fn sofuu_cuda_driver_present() -> c_int;
        pub fn sofuu_cuda_debug_enabled() -> c_int;
    }

    pub fn last_error() -> String {
        unsafe {
            let p = sofuu_cuda_last_error();
            if p.is_null() {
                "unknown CUDA error".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }
}

pub use imp::CudaEngine;

#[cfg(any(target_os = "linux", all(test, target_os = "macos")))]
mod imp {
    use super::*;
    use std::collections::HashMap;

    // Embedded on Linux for the real backend, and in test builds so the PTX
    // inflation path is exercised on any host (the bytes are 5.7 KB gzipped).
    const PTX_GZ: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../src/cuda/kernels.ptx.gz"
    ));

    fn inflate_ptx() -> Result<String, CudaError> {
        use std::io::Read;
        let mut gz = flate2::read::GzDecoder::new(PTX_GZ);
        let mut out = String::new();
        gz.read_to_string(&mut out)
            .map_err(|e| CudaError(format!("inflate kernels.ptx.gz: {e}")))?;
        if out.is_empty() {
            return Err(CudaError("embedded kernels.ptx.gz is empty".into()));
        }
        Ok(out)
    }

    fn take_error(what: &str) -> CudaError {
        let msg = sys::last_error();
        debug(format!("{what}: {msg}"));
        CudaError(format!("{what}: {msg}"))
    }

    fn info_from(driver: *const c_void, ordinal: i32) -> Result<DeviceInfo, CudaError> {
        let mut name = [0 as c_char; 256];
        let (mut major, mut minor) = (0 as c_int, 0 as c_int);
        let mut total = 0u64;
        let (mut integrated, mut unified) = (0 as c_int, 0 as c_int);
        let rc = unsafe {
            sys::sofuu_cuda_driver_device_info(
                driver,
                ordinal,
                name.as_mut_ptr(),
                name.len(),
                &mut major,
                &mut minor,
                &mut total,
                &mut integrated,
                &mut unified,
            )
        };
        if rc != 0 {
            return Err(take_error("device_info"));
        }
        let cname = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let lib = unsafe {
            let p = sys::sofuu_cuda_driver_path(driver);
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Ok(DeviceInfo {
            name: cname,
            cc_major: major,
            cc_minor: minor,
            total_memory: total,
            integrated: integrated != 0,
            unified_addressing: unified != 0,
            driver_version: unsafe { sys::sofuu_cuda_driver_version(driver) },
            library: lib,
        })
    }

    /// Cheap probe: opens the driver and reads device 0, without creating a
    /// context or JIT-compiling the module.
    pub fn probe() -> Option<DeviceInfo> {
        if kill_switch() {
            debug("SOFUU_CUDA=0 — CUDA disabled");
            return None;
        }
        let driver = unsafe { sys::sofuu_cuda_driver_open(ptr::null()) };
        if driver.is_null() {
            debug(format!("no CUDA driver: {}", sys::last_error()));
            return None;
        }
        let ordinal = device_ordinal();
        let info = info_from(driver, ordinal);
        unsafe { sys::sofuu_cuda_driver_close(driver) };
        match info {
            Ok(i) => Some(i),
            Err(e) => {
                debug(e.0);
                None
            }
        }
    }

    pub fn driver_present() -> bool {
        unsafe { sys::sofuu_cuda_driver_present() != 0 }
    }

    fn device_ordinal() -> i32 {
        std::env::var("SOFUU_CUDA_DEVICE")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0)
    }

    pub struct CudaEngine {
        driver: *mut c_void,
        ctx: *mut c_void,
        funcs: HashMap<&'static str, *mut c_void>,
        info: DeviceInfo,
    }

    // SAFETY: the handle is a driver context plus a function table, both
    // immutable after try_init. CUDA contexts are current per-thread, so the
    // only safe way to use this from another thread is to make the context
    // current there first — the engine has no cuCtxSetCurrent call. CudaModel
    // records its owner thread and refuses a forward elsewhere; model load and
    // generation run on one thread in every CLI path. The impl exists because
    // InferEngine is moved as a value, not because cross-thread use is allowed.
    unsafe impl Send for CudaEngine {}

    impl CudaEngine {
        /// Initialise driver → device → context → PTX module → entry points.
        /// Any failure returns None; the caller keeps the CPU engine.
        pub fn try_init() -> Option<Self> {
            if kill_switch() {
                debug("SOFUU_CUDA=0 — CUDA disabled");
                return None;
            }
            unsafe {
                let driver = sys::sofuu_cuda_driver_open(ptr::null());
                if driver.is_null() {
                    debug(format!("no CUDA driver: {}", sys::last_error()));
                    return None;
                }
                let ordinal = device_ordinal();
                let info = match info_from(driver, ordinal) {
                    Ok(i) => i,
                    Err(e) => {
                        debug(e.0);
                        sys::sofuu_cuda_driver_close(driver);
                        return None;
                    }
                };
                let ctx = sys::sofuu_cuda_ctx_create(driver, ordinal);
                if ctx.is_null() {
                    debug(format!("context: {}", sys::last_error()));
                    sys::sofuu_cuda_driver_close(driver);
                    return None;
                }

                let ptx = match inflate_ptx() {
                    Ok(p) => p,
                    Err(e) => {
                        debug(e.0);
                        sys::sofuu_cuda_ctx_destroy(ctx);
                        sys::sofuu_cuda_driver_close(driver);
                        return None;
                    }
                };
                let Ok(ptx_c) = CString::new(ptx) else {
                    debug("PTX contains an interior NUL");
                    sys::sofuu_cuda_ctx_destroy(ctx);
                    sys::sofuu_cuda_driver_close(driver);
                    return None;
                };
                if sys::sofuu_cuda_ctx_load_ptx(ctx, ptx_c.as_ptr()) != 0 {
                    debug(format!("cuModuleLoadData: {}", sys::last_error()));
                    sys::sofuu_cuda_ctx_destroy(ctx);
                    sys::sofuu_cuda_driver_close(driver);
                    return None;
                }

                let mut funcs = HashMap::with_capacity(PTX_ENTRIES.len());
                for name in PTX_ENTRIES {
                    let Ok(cname) = CString::new(name) else {
                        continue;
                    };
                    let mut f: *mut c_void = ptr::null_mut();
                    if sys::sofuu_cuda_ctx_function(ctx, cname.as_ptr(), &mut f) != 0 {
                        debug(format!("missing kernel {name}: {}", sys::last_error()));
                        sys::sofuu_cuda_ctx_destroy(ctx);
                        sys::sofuu_cuda_driver_close(driver);
                        return None;
                    }
                    funcs.insert(name, f);
                }

                debug(format!(
                    "initialised {} cc{}.{} on {}",
                    info.name, info.cc_major, info.cc_minor, info.library
                ));
                Some(CudaEngine {
                    driver,
                    ctx,
                    funcs,
                    info,
                })
            }
        }

        pub fn device_info(&self) -> &DeviceInfo {
            &self.info
        }

        pub fn is_unified_memory(&self) -> bool {
            self.info.integrated || self.info.unified_addressing
        }

        pub fn alloc(&self, bytes: usize) -> Result<DeviceBuf, CudaError> {
            if bytes == 0 {
                return Err(CudaError("cannot allocate 0 bytes".into()));
            }
            let mut ptr = 0u64;
            if unsafe { sys::sofuu_cuda_ctx_alloc(self.ctx, bytes as u64, &mut ptr) } != 0 {
                return Err(take_error("cuMemAlloc"));
            }
            Ok(DeviceBuf { ptr, bytes })
        }

        pub fn free_buf(&self, buf: DeviceBuf) -> Result<(), CudaError> {
            if unsafe { sys::sofuu_cuda_ctx_free(self.ctx, buf.ptr) } != 0 {
                return Err(take_error("cuMemFree"));
            }
            Ok(())
        }

        pub fn upload(&self, dst: DeviceBuf, host: &[u8]) -> Result<(), CudaError> {
            if host.len() > dst.bytes {
                return Err(CudaError(format!(
                    "upload {} B into a {} B device buffer",
                    host.len(),
                    dst.bytes
                )));
            }
            if unsafe {
                sys::sofuu_cuda_ctx_upload(
                    self.ctx,
                    dst.ptr,
                    host.as_ptr() as *const c_void,
                    host.len(),
                )
            } != 0
            {
                return Err(take_error("cuMemcpyHtoD"));
            }
            Ok(())
        }

        pub fn download(&self, dst: &mut [u8], src: DeviceBuf) -> Result<(), CudaError> {
            if dst.len() > src.bytes {
                return Err(CudaError(format!(
                    "download {} B from a {} B device buffer",
                    dst.len(),
                    src.bytes
                )));
            }
            if unsafe {
                sys::sofuu_cuda_ctx_download(
                    self.ctx,
                    dst.as_mut_ptr() as *mut c_void,
                    src.ptr,
                    dst.len(),
                )
            } != 0
            {
                return Err(take_error("cuMemcpyDtoH"));
            }
            Ok(())
        }

        pub fn sync(&self) -> Result<(), CudaError> {
            if unsafe { sys::sofuu_cuda_ctx_sync(self.ctx) } != 0 {
                return Err(take_error("cuStreamSynchronize"));
            }
            Ok(())
        }

        fn function(&self, name: &str) -> Result<*mut c_void, CudaError> {
            self.funcs
                .get(name)
                .copied()
                .ok_or_else(|| CudaError(format!("kernel {name} is not in the PTX module")))
        }

        fn launch(
            &self,
            name: &str,
            grid: (u32, u32, u32),
            block: (u32, u32, u32),
            params: &mut [*mut c_void],
        ) -> Result<(), CudaError> {
            let f = self.function(name)?;
            let rc = unsafe {
                sys::sofuu_cuda_ctx_launch(
                    self.ctx,
                    f,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    0,
                    params.as_mut_ptr(),
                )
            };
            if rc != 0 {
                return Err(take_error(name));
            }
            // No cuStreamSynchronize here. Every launch is on the same stream,
            // so order is guaranteed; a sync after each of ~240 launches per
            // token only added driver latency. The one place the host needs
            // the result is the logits download, which syncs explicitly.
            Ok(())
        }

        /// Row-major GEMV: y[r] = dot(weights[r], x). `weights` holds
        /// `rows * row_bytes(cols)` bytes in GGML layout for `quant`.
        pub fn gemv(
            &self,
            quant: Quant,
            weights: DeviceBuf,
            x: DeviceBuf,
            y: DeviceBuf,
            rows: usize,
            cols: usize,
        ) -> Result<(), CudaError> {
            let Some(row_bytes) = quant.row_bytes(cols) else {
                return Err(CudaError(format!(
                    "{} does not accept cols={cols}",
                    quant.entry()
                )));
            };
            let need_w = row_bytes
                .checked_mul(rows)
                .ok_or_else(|| CudaError("weight size overflow".into()))?;
            let need_x = cols * 4; // every kernel takes f32 activations
            let need_y = rows * 4;
            if weights.bytes < need_w || x.bytes < need_x || y.bytes < need_y {
                return Err(CudaError(format!(
                    "gemv buffers too small: w {}<{}, x {}<{}, y {}<{}",
                    weights.bytes, need_w, x.bytes, need_x, y.bytes, need_y
                )));
            }
            let mut w = weights.ptr;
            let mut xp = x.ptr;
            let mut yp = y.ptr;
            let mut r = rows as i32;
            let mut c = cols as i32;
            let mut params: [*mut c_void; 5] = [
                &mut w as *mut u64 as *mut c_void,
                &mut xp as *mut u64 as *mut c_void,
                &mut yp as *mut u64 as *mut c_void,
                &mut r as *mut i32 as *mut c_void,
                &mut c as *mut i32 as *mut c_void,
            ];
            self.launch(quant.entry(), (rows as u32, 1, 1), (256, 1, 1), &mut params)
        }

        pub fn rmsnorm(
            &self,
            x: DeviceBuf,
            w: DeviceBuf,
            y: DeviceBuf,
            n: usize,
            eps: f32,
        ) -> Result<(), CudaError> {
            let mut xp = x.ptr;
            let mut wp = w.ptr;
            let mut yp = y.ptr;
            let mut nn = n as i32;
            let mut ee = eps;
            let mut params: [*mut c_void; 5] = [
                &mut xp as *mut u64 as *mut c_void,
                &mut wp as *mut u64 as *mut c_void,
                &mut yp as *mut u64 as *mut c_void,
                &mut nn as *mut i32 as *mut c_void,
                &mut ee as *mut f32 as *mut c_void,
            ];
            self.launch("sofuu_rmsnorm", (1, 1, 1), (256, 1, 1), &mut params)
        }

        pub fn swiglu(
            &self,
            g: DeviceBuf,
            u: DeviceBuf,
            y: DeviceBuf,
            n: usize,
        ) -> Result<(), CudaError> {
            let mut gp = g.ptr;
            let mut up = u.ptr;
            let mut yp = y.ptr;
            let mut nn = n as i32;
            let mut params: [*mut c_void; 4] = [
                &mut gp as *mut u64 as *mut c_void,
                &mut up as *mut u64 as *mut c_void,
                &mut yp as *mut u64 as *mut c_void,
                &mut nn as *mut i32 as *mut c_void,
            ];
            let grid = ((n + 255) / 256) as u32;
            self.launch("sofuu_swiglu", (grid, 1, 1), (256, 1, 1), &mut params)
        }

        pub fn rope(
            &self,
            x: DeviceBuf,
            cos_tab: DeviceBuf,
            sin_tab: DeviceBuf,
            n_heads: usize,
            head_dim: usize,
            n_rot: usize,
            interleaved: bool,
        ) -> Result<(), CudaError> {
            let mut xp = x.ptr;
            let mut cp = cos_tab.ptr;
            let mut sp = sin_tab.ptr;
            let mut h = n_heads as i32;
            let mut hd = head_dim as i32;
            let mut nr = n_rot as i32;
            let mut il = if interleaved { 1i32 } else { 0i32 };
            let mut params: [*mut c_void; 7] = [
                &mut xp as *mut u64 as *mut c_void,
                &mut cp as *mut u64 as *mut c_void,
                &mut sp as *mut u64 as *mut c_void,
                &mut h as *mut i32 as *mut c_void,
                &mut hd as *mut i32 as *mut c_void,
                &mut nr as *mut i32 as *mut c_void,
                &mut il as *mut i32 as *mut c_void,
            ];
            self.launch(
                "sofuu_rope_f32",
                (n_heads as u32, 1, 1),
                (128, 1, 1),
                &mut params,
            )
        }

        pub fn kv_write(
            &self,
            k_cache: DeviceBuf,
            v_cache: DeviceBuf,
            k: DeviceBuf,
            v: DeviceBuf,
            pos: usize,
            n_kv_heads: usize,
            head_dim: usize,
            cap: usize,
        ) -> Result<(), CudaError> {
            validate_kv_write(
                pos, n_kv_heads, head_dim, cap, k_cache.bytes, v_cache.bytes, k.bytes, v.bytes,
            )?;
            let mut kc = k_cache.ptr;
            let mut vc = v_cache.ptr;
            let mut kp = k.ptr;
            let mut vp = v.ptr;
            let mut p = pos as i32;
            let mut h = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut c = cap as i32;
            let mut params: [*mut c_void; 8] = [
                &mut kc as *mut u64 as *mut c_void,
                &mut vc as *mut u64 as *mut c_void,
                &mut kp as *mut u64 as *mut c_void,
                &mut vp as *mut u64 as *mut c_void,
                &mut p as *mut i32 as *mut c_void,
                &mut h as *mut i32 as *mut c_void,
                &mut hd as *mut i32 as *mut c_void,
                &mut c as *mut i32 as *mut c_void,
            ];
            self.launch(
                "sofuu_kv_write_f32",
                (n_kv_heads as u32, 1, 1),
                (128, 1, 1),
                &mut params,
            )
        }

        /// Decode attention for one token over `seq_len` cached rows.
        /// `q` is [n_heads, head_dim], `k_cache`/`v_cache` are
        /// [n_kv_heads, cap, head_dim], `out` is [n_heads, head_dim].
        #[allow(clippy::too_many_arguments)]
        pub fn attn_decode(
            &self,
            q: DeviceBuf,
            k_cache: DeviceBuf,
            v_cache: DeviceBuf,
            out: DeviceBuf,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            seq_len: usize,
            cap: usize,
            scale: f32,
        ) -> Result<(), CudaError> {
            validate_attn_decode(
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                cap,
                q.bytes,
                k_cache.bytes,
                v_cache.bytes,
                out.bytes,
            )?;
            let mut qp = q.ptr;
            let mut kc = k_cache.ptr;
            let mut vc = v_cache.ptr;
            let mut op = out.ptr;
            let mut nh = n_heads as i32;
            let mut nk = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut sl = seq_len as i32;
            let mut c = cap as i32;
            let mut s = scale;
            let mut params: [*mut c_void; 10] = [
                &mut qp as *mut u64 as *mut c_void,
                &mut kc as *mut u64 as *mut c_void,
                &mut vc as *mut u64 as *mut c_void,
                &mut op as *mut u64 as *mut c_void,
                &mut nh as *mut i32 as *mut c_void,
                &mut nk as *mut i32 as *mut c_void,
                &mut hd as *mut i32 as *mut c_void,
                &mut sl as *mut i32 as *mut c_void,
                &mut c as *mut i32 as *mut c_void,
                &mut s as *mut f32 as *mut c_void,
            ];
            self.launch(
                "sofuu_attn_decode_f32",
                (n_heads as u32, 1, 1),
                (256, 1, 1),
                &mut params,
            )
        }

        /// y = a + b, elementwise (in-place when `y` aliases `a`).
        pub fn add(&self, a: DeviceBuf, b: DeviceBuf, y: DeviceBuf, n: usize) -> Result<(), CudaError> {
            let need = n
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| CudaError("add length overflow".into()))?;
            if a.bytes < need || b.bytes < need || y.bytes < need {
                return Err(CudaError(format!(
                    "add buffers too small for {n} floats: a {} b {} y {}",
                    a.bytes, b.bytes, y.bytes
                )));
            }
            let mut ap = a.ptr;
            let mut bp = b.ptr;
            let mut yp = y.ptr;
            let mut nn = n as i32;
            let mut params: [*mut c_void; 4] = [
                &mut ap as *mut u64 as *mut c_void,
                &mut bp as *mut u64 as *mut c_void,
                &mut yp as *mut u64 as *mut c_void,
                &mut nn as *mut i32 as *mut c_void,
            ];
            self.launch(
                "sofuu_add_f32",
                (((n + 255) / 256) as u32, 1, 1),
                (256, 1, 1),
                &mut params,
            )
        }

        /// Batched prefill GEMV: one warp per token, `row_tile` output rows per
        /// block. `x` is [n_tokens, cols] and `y` is [n_tokens, rows], both
        /// token-major; the weight is read once and reused across the chunk.
        #[allow(clippy::too_many_arguments)]
        pub fn gemm_b8(
            &self,
            quant: Quant,
            weights: DeviceBuf,
            x: DeviceBuf,
            y: DeviceBuf,
            rows: usize,
            cols: usize,
            n_tokens: usize,
            row_tile: usize,
        ) -> Result<(), CudaError> {
            validate_gemm_b8(
                quant, rows, cols, n_tokens, row_tile, weights.bytes, x.bytes, y.bytes,
            )?;
            let mut wp = weights.ptr;
            let mut xp = x.ptr;
            let mut yp = y.ptr;
            let mut r = rows as i32;
            let mut c = cols as i32;
            let mut n = n_tokens as i32;
            let mut rt = row_tile as i32;
            let mut params: [*mut c_void; 7] = [
                &mut wp as *mut u64 as *mut c_void,
                &mut xp as *mut u64 as *mut c_void,
                &mut yp as *mut u64 as *mut c_void,
                &mut r as *mut i32 as *mut c_void,
                &mut c as *mut i32 as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
                &mut rt as *mut i32 as *mut c_void,
            ];
            let grid = ((rows + row_tile - 1) / row_tile) as u32;
            self.launch(quant.batch_entry(), (grid, 1, 1), (256, 1, 1), &mut params)
        }

        /// RMSNorm over `rows` consecutive rows of `n` values.
        pub fn rmsnorm_batch(
            &self,
            x: DeviceBuf,
            w: DeviceBuf,
            y: DeviceBuf,
            n: usize,
            eps: f32,
            rows: usize,
        ) -> Result<(), CudaError> {
            let need = rows
                .checked_mul(n)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| CudaError("rmsnorm_batch shape overflow".into()))?;
            if n == 0 || rows == 0 || x.bytes < need || y.bytes < need || w.bytes < n * 4 {
                return Err(CudaError(format!(
                    "rmsnorm_batch buffers too small: x {} y {} w {} need {need}/{need}/{}",
                    x.bytes, y.bytes, w.bytes, n * 4
                )));
            }
            let mut xp = x.ptr;
            let mut wp = w.ptr;
            let mut yp = y.ptr;
            let mut nn = n as i32;
            let mut ee = eps;
            let mut rr = rows as i32;
            let mut params: [*mut c_void; 6] = [
                &mut xp as *mut u64 as *mut c_void,
                &mut wp as *mut u64 as *mut c_void,
                &mut yp as *mut u64 as *mut c_void,
                &mut nn as *mut i32 as *mut c_void,
                &mut ee as *mut f32 as *mut c_void,
                &mut rr as *mut i32 as *mut c_void,
            ];
            self.launch(
                "sofuu_rmsnorm_batch",
                (rows as u32, 1, 1),
                (256, 1, 1),
                &mut params,
            )
        }

        /// RoPE over `n_tokens` tokens. `table` holds one [head_dim] row per
        /// token (cosine half then sine half), built by `quant::rope_table_rows`.
        #[allow(clippy::too_many_arguments)]
        pub fn rope_batch(
            &self,
            x: DeviceBuf,
            table: DeviceBuf,
            n_heads: usize,
            head_dim: usize,
            n_rot: usize,
            interleaved: bool,
            n_tokens: usize,
        ) -> Result<(), CudaError> {
            let need = n_tokens
                .checked_mul(head_dim)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| CudaError("rope_batch table shape overflow".into()))?;
            if table.bytes < need {
                return Err(CudaError(format!(
                    "rope_batch table {}<{need} bytes",
                    table.bytes
                )));
            }
            let mut xp = x.ptr;
            let mut tp = table.ptr;
            let mut h = n_heads as i32;
            let mut hd = head_dim as i32;
            let mut nr = n_rot as i32;
            let mut il = if interleaved { 1i32 } else { 0i32 };
            let mut n = n_tokens as i32;
            let mut params: [*mut c_void; 7] = [
                &mut xp as *mut u64 as *mut c_void,
                &mut tp as *mut u64 as *mut c_void,
                &mut h as *mut i32 as *mut c_void,
                &mut hd as *mut i32 as *mut c_void,
                &mut nr as *mut i32 as *mut c_void,
                &mut il as *mut i32 as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
            ];
            self.launch(
                "sofuu_rope_batch",
                ((n_tokens * n_heads) as u32, 1, 1),
                (128, 1, 1),
                &mut params,
            )
        }

        /// Write `n_tokens` consecutive K/V rows into the cache.
        #[allow(clippy::too_many_arguments)]
        pub fn kv_write_batch(
            &self,
            k_cache: DeviceBuf,
            v_cache: DeviceBuf,
            k: DeviceBuf,
            v: DeviceBuf,
            start_pos: usize,
            n_tokens: usize,
            n_kv_heads: usize,
            head_dim: usize,
            cap: usize,
        ) -> Result<(), CudaError> {
            validate_kv_write_batch(
                start_pos,
                n_tokens,
                n_kv_heads,
                head_dim,
                cap,
                k_cache.bytes,
                v_cache.bytes,
                k.bytes,
                v.bytes,
            )?;
            let mut kc = k_cache.ptr;
            let mut vc = v_cache.ptr;
            let mut kp = k.ptr;
            let mut vp = v.ptr;
            let mut p = start_pos as i32;
            let mut nt = n_tokens as i32;
            let mut h = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut c = cap as i32;
            let mut params: [*mut c_void; 9] = [
                &mut kc as *mut u64 as *mut c_void,
                &mut vc as *mut u64 as *mut c_void,
                &mut kp as *mut u64 as *mut c_void,
                &mut vp as *mut u64 as *mut c_void,
                &mut p as *mut i32 as *mut c_void,
                &mut nt as *mut i32 as *mut c_void,
                &mut h as *mut i32 as *mut c_void,
                &mut hd as *mut i32 as *mut c_void,
                &mut c as *mut i32 as *mut c_void,
            ];
            self.launch(
                "sofuu_kv_write_batch",
                ((n_tokens * n_kv_heads) as u32, 1, 1),
                (128, 1, 1),
                &mut params,
            )
        }

        /// Causal attention for a whole chunk: one block per (token, head),
        /// token t attending over [0, start_pos+t].
        #[allow(clippy::too_many_arguments)]
        pub fn attn_prefill(
            &self,
            q: DeviceBuf,
            k_cache: DeviceBuf,
            v_cache: DeviceBuf,
            out: DeviceBuf,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            start_pos: usize,
            n_tokens: usize,
            cap: usize,
            scale: f32,
        ) -> Result<(), CudaError> {
            validate_attn_prefill(
                n_heads,
                n_kv_heads,
                head_dim,
                start_pos,
                n_tokens,
                cap,
                q.bytes,
                k_cache.bytes,
                v_cache.bytes,
                out.bytes,
            )?;
            let mut qp = q.ptr;
            let mut kc = k_cache.ptr;
            let mut vc = v_cache.ptr;
            let mut op = out.ptr;
            let mut nh = n_heads as i32;
            let mut nk = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut sp = start_pos as i32;
            let mut nt = n_tokens as i32;
            let mut c = cap as i32;
            let mut s = scale;
            let mut params: [*mut c_void; 11] = [
                &mut qp as *mut u64 as *mut c_void,
                &mut kc as *mut u64 as *mut c_void,
                &mut vc as *mut u64 as *mut c_void,
                &mut op as *mut u64 as *mut c_void,
                &mut nh as *mut i32 as *mut c_void,
                &mut nk as *mut i32 as *mut c_void,
                &mut hd as *mut i32 as *mut c_void,
                &mut sp as *mut i32 as *mut c_void,
                &mut nt as *mut i32 as *mut c_void,
                &mut c as *mut i32 as *mut c_void,
                &mut s as *mut f32 as *mut c_void,
            ];
            self.launch(
                "sofuu_attn_prefill_f32",
                ((n_tokens * n_heads) as u32, 1, 1),
                (256, 1, 1),
                &mut params,
            )
        }

        /// Host-side convenience used by integration tests: upload, run, read
        /// back. Not on the model's hot path.
        pub fn gemv_host(
            &self,
            quant: Quant,
            weights: &[u8],
            x: &[f32],
            rows: usize,
            cols: usize,
        ) -> Result<Vec<f32>, CudaError> {
            let row_bytes = quant
                .row_bytes(cols)
                .ok_or_else(|| CudaError("bad cols".into()))?;
            if weights.len() < row_bytes * rows {
                return Err(CudaError("weights slice too short".into()));
            }
            if x.len() < cols {
                return Err(CudaError("x slice too short".into()));
            }
            let dw = self.alloc(row_bytes * rows)?;
            let dx = self.alloc(cols * 4)?;
            let dy = self.alloc(rows * 4)?;
            let result = (|| {
                self.upload(dw, &weights[..row_bytes * rows])?;
                self.upload(dx, bytemuck_view(x))?;
                self.gemv(quant, dw, dx, dy, rows, cols)?;
                let mut out = vec![0f32; rows];
                let mut bytes = vec![0u8; rows * 4];
                self.download(&mut bytes, dy)?;
                for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Ok(out)
            })();
            let _ = self.free_buf(dw);
            let _ = self.free_buf(dx);
            let _ = self.free_buf(dy);
            result
        }
    }

    /// View a float slice as bytes without unsafe pointer casts at call sites.
    fn bytemuck_view(v: &[f32]) -> &[u8] {
        // Safe: `[f32]` has no padding and every bit pattern is valid for u8.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }

    impl Drop for CudaEngine {
        fn drop(&mut self) {
            unsafe {
                if !self.ctx.is_null() {
                    sys::sofuu_cuda_ctx_destroy(self.ctx);
                    self.ctx = ptr::null_mut();
                }
                if !self.driver.is_null() {
                    sys::sofuu_cuda_driver_close(self.driver);
                    self.driver = ptr::null_mut();
                }
            }
        }
    }
}

#[cfg(not(any(target_os = "linux", all(test, target_os = "macos"))))]
mod imp {
    use super::*;

    /// On non-Linux release targets the native backend is not built. The type
    /// still exists so model code can compile a single portable call site.
    pub struct CudaEngine {
        _private: (),
    }

    impl CudaEngine {
        pub fn try_init() -> Option<Self> {
            None
        }

        pub fn device_info(&self) -> &DeviceInfo {
            unreachable!("CudaEngine can never be constructed on this target")
        }

        pub fn is_unified_memory(&self) -> bool {
            false
        }

        pub fn alloc(&self, _bytes: usize) -> Result<DeviceBuf, CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn free_buf(&self, _buf: DeviceBuf) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn upload(&self, _dst: DeviceBuf, _host: &[u8]) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn download(&self, _dst: &mut [u8], _src: DeviceBuf) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn sync(&self) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn gemv(
            &self,
            _quant: Quant,
            _weights: DeviceBuf,
            _x: DeviceBuf,
            _y: DeviceBuf,
            _rows: usize,
            _cols: usize,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn rmsnorm(
            &self,
            _x: DeviceBuf,
            _w: DeviceBuf,
            _y: DeviceBuf,
            _n: usize,
            _eps: f32,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn swiglu(
            &self,
            _g: DeviceBuf,
            _u: DeviceBuf,
            _y: DeviceBuf,
            _n: usize,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn rope(
            &self,
            _x: DeviceBuf,
            _cos: DeviceBuf,
            _sin: DeviceBuf,
            _n_heads: usize,
            _head_dim: usize,
            _n_rot: usize,
            _interleaved: bool,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn kv_write(
            &self,
            _kc: DeviceBuf,
            _vc: DeviceBuf,
            _k: DeviceBuf,
            _v: DeviceBuf,
            _pos: usize,
            _n_kv_heads: usize,
            _head_dim: usize,
            _cap: usize,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn attn_decode(
            &self,
            _q: DeviceBuf,
            _kc: DeviceBuf,
            _vc: DeviceBuf,
            _out: DeviceBuf,
            _n_heads: usize,
            _n_kv_heads: usize,
            _head_dim: usize,
            _seq_len: usize,
            _cap: usize,
            _scale: f32,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn add(
            &self,
            _a: DeviceBuf,
            _b: DeviceBuf,
            _y: DeviceBuf,
            _n: usize,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn gemm_b8(
            &self,
            _quant: Quant,
            _weights: DeviceBuf,
            _x: DeviceBuf,
            _y: DeviceBuf,
            _rows: usize,
            _cols: usize,
            _n_tokens: usize,
            _row_tile: usize,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn rmsnorm_batch(
            &self,
            _x: DeviceBuf,
            _w: DeviceBuf,
            _y: DeviceBuf,
            _n: usize,
            _eps: f32,
            _rows: usize,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn rope_batch(
            &self,
            _x: DeviceBuf,
            _table: DeviceBuf,
            _n_heads: usize,
            _head_dim: usize,
            _n_rot: usize,
            _interleaved: bool,
            _n_tokens: usize,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn kv_write_batch(
            &self,
            _kc: DeviceBuf,
            _vc: DeviceBuf,
            _k: DeviceBuf,
            _v: DeviceBuf,
            _start_pos: usize,
            _n_tokens: usize,
            _n_kv_heads: usize,
            _head_dim: usize,
            _cap: usize,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn attn_prefill(
            &self,
            _q: DeviceBuf,
            _kc: DeviceBuf,
            _vc: DeviceBuf,
            _out: DeviceBuf,
            _n_heads: usize,
            _n_kv_heads: usize,
            _head_dim: usize,
            _start_pos: usize,
            _n_tokens: usize,
            _cap: usize,
            _scale: f32,
        ) -> Result<(), CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }

        pub fn gemv_host(
            &self,
            _quant: Quant,
            _weights: &[u8],
            _x: &[f32],
            _rows: usize,
            _cols: usize,
        ) -> Result<Vec<f32>, CudaError> {
            Err(CudaError("CUDA backend is Linux-only".into()))
        }
    }
}

/// True when the operator asked for CUDA (`SOFUU_CUDA=1|true|on|yes`).
///
/// This is the opt-in, not the availability probe: unset means "leave the
/// model on the CPU", which keeps a new backend from silently changing what
/// every existing Linux+NVIDIA run does. `SOFUU_CUDA=0` stays the explicit
/// kill switch. `driver_present()` answers the other half (is there a device).
pub fn requested() -> bool {
    match std::env::var("SOFUU_CUDA") {
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            !matches!(t.as_str(), "0" | "false" | "off" | "no" | "")
        }
        Err(_) => false,
    }
}

/// True when a CUDA driver library + device are present. Cheap enough for
/// diagnostics; does not create a context or JIT the module.
pub fn driver_present() -> bool {
    #[cfg(target_os = "linux")]
    {
        imp::driver_present()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Device identity without initialising the module. None when CUDA is
/// unavailable or disabled with SOFUU_CUDA=0.
pub fn probe() -> Option<DeviceInfo> {
    #[cfg(target_os = "linux")]
    {
        imp::probe()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_is_absent_without_a_driver() {
        // macOS/CI: graceful None, never a panic or a link error. On a real
        // NVIDIA host the GPU gate (Kaggle) covers initialisation instead.
        if driver_present() {
            return;
        }
        assert!(CudaEngine::try_init().is_none());
    }

    #[cfg(any(target_os = "linux", all(test, target_os = "macos")))]
    #[test]
    fn loader_reports_missing_driver_cleanly() {
        let p = unsafe { sys::sofuu_cuda_driver_open(ptr::null()) };
        if !p.is_null() {
            unsafe { sys::sofuu_cuda_driver_close(p) };
            return; // a real GPU host: the GPU gate covers this path
        }
        let msg = sys::last_error();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("no CUDA driver") || msg.contains("libcuda"),
            "unexpected error text: {msg}"
        );
    }

    #[cfg(any(target_os = "linux", all(test, target_os = "macos")))]
    #[test]
    fn loader_rejects_a_bogus_library_path() {
        let bogus = CString::new("/nonexistent/sofuu-libcuda.so.1").unwrap();
        let p = unsafe { sys::sofuu_cuda_driver_open(bogus.as_ptr()) };
        assert!(p.is_null());
        let msg = sys::last_error();
        assert!(!msg.is_empty());
    }

    #[test]
    fn ptx_manifest_contains_every_kernel() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/cuda/kernels.ptx");
        let ptx = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for name in PTX_ENTRIES {
            assert!(
                ptx.contains(&format!(".entry {name}(")),
                "kernels.ptx is missing {name}"
            );
        }
    }

    /// Every kernel must carry a `.maxntid`, so a future edit cannot silently
    /// blow the register budget and fail the launch with error 701 on hardware
    /// nobody tested here. The block size is declared twice — in kernels.cu via
    /// SOFUU_LAUNCH_BOUNDS and here in the checked-in PTX — and this is what
    /// keeps the two honest.

    #[test]
    fn kv_write_bounds_are_refused_before_the_driver_sees_them() {
        // 2 heads x 8 rows x 4 dims x 4 bytes = 256 bytes per cache.
        let ok = (256, 64);
        assert!(validate_kv_write(7, 2, 4, 8, ok.0, ok.0, ok.1, ok.1).is_ok());

        // One past the last row is corruption, not a clamp.
        let e = validate_kv_write(8, 2, 4, 8, 256, 256, 64, 64).unwrap_err();
        assert!(e.0.contains("outside a cache"), "{}", e.0);

        // A cache smaller than heads*rows*dims is refused with the arithmetic.
        let e = validate_kv_write(0, 2, 4, 8, 128, 256, 64, 64).unwrap_err();
        assert!(e.0.contains("needs 256 bytes"), "{}", e.0);

        // A short source would have the kernel read past the caller's buffer.
        let e = validate_kv_write(0, 2, 4, 8, 256, 256, 16, 64).unwrap_err();
        assert!(e.0.contains("source"), "{}", e.0);

        // Degenerate shapes are refused up front rather than divided by zero.
        assert!(validate_kv_write(0, 0, 4, 8, 256, 256, 64, 64).is_err());
        assert!(validate_kv_write(0, 2, 0, 8, 256, 256, 64, 64).is_err());
        assert!(validate_kv_write(0, 2, 4, 0, 256, 256, 64, 64).is_err());
    }

    #[test]
    fn gemm_b8_bounds_are_refused_before_the_driver() {
        // q4_K, 8 rows x 256 cols, eight tokens, eight-row tiles.
        let w = 8 * 144;
        let x = 8 * 256 * 4;
        let y = 8 * 8 * 4;
        assert!(validate_gemm_b8(Quant::Q4K, 8, 256, 8, 8, w, x, y).is_ok());

        // The kernels are one warp per token and cannot take a ninth.
        let e = validate_gemm_b8(Quant::Q4K, 8, 256, 9, 8, w, x, y).unwrap_err();
        assert!(e.0.contains("outside 1..=8"), "{}", e.0);
        assert!(validate_gemm_b8(Quant::Q4K, 8, 256, 0, 8, w, x, y).is_err());

        // Row tiles are the 8-row accumulator groups.
        assert!(validate_gemm_b8(Quant::Q4K, 8, 256, 8, 7, w, x, y).is_err());
        assert!(validate_gemm_b8(Quant::Q4K, 8, 256, 8, 40, w, x, y).is_err());

        // A short activation or output buffer would read/write out of bounds.
        let e = validate_gemm_b8(Quant::Q4K, 8, 256, 8, 8, w, x - 4, y).unwrap_err();
        assert!(e.0.contains("buffers too small"), "{}", e.0);
        let e = validate_gemm_b8(Quant::Q4K, 8, 256, 8, 8, w, x, y - 4).unwrap_err();
        assert!(e.0.contains("buffers too small"), "{}", e.0);

        // f32 rows still need the aligned float4 width.
        let e = validate_gemm_b8(Quant::F32, 8, 6, 1, 8, 8 * 6 * 4, 6 * 4, 8 * 4)
            .unwrap_err();
        assert!(e.0.contains("multiple of 4"), "{}", e.0);
    }

    #[test]
    fn attn_prefill_bounds_are_refused_before_the_driver() {
        let q = 8 * 8 * 64 * 4;
        let cache = 2 * 32 * 64 * 4;
        assert!(validate_attn_prefill(8, 2, 64, 5, 8, 32, q, cache, cache, q).is_ok());

        // The chunk's last token attends over start_pos + n_tokens rows.
        let e = validate_attn_prefill(8, 2, 64, 30, 8, 32, q, cache, cache, q)
            .unwrap_err();
        assert!(e.0.contains("leaves a cache"), "{}", e.0);
        assert!(validate_attn_prefill(8, 2, 64, 5, 0, 32, q, cache, cache, q).is_err());
        assert!(validate_attn_prefill(8, 3, 64, 5, 8, 32, q, cache, cache, q).is_err());
        assert!(validate_attn_prefill(8, 2, 64, 5, 8, 32, q - 4, cache, cache, q).is_err());
    }

    #[test]
    fn attn_decode_bounds_are_refused_before_the_driver() {
        // 8 query heads, 2 KV heads, head_dim 64, seq 32 of a 32-row cache.
        let q = 8 * 64 * 4;
        let cache = 2 * 32 * 64 * 4;
        assert!(validate_attn_decode(8, 2, 64, 32, 32, q, cache, cache, q).is_ok());

        // A sequence longer than the cache would read rows that were never written.
        let e = validate_attn_decode(8, 2, 64, 33, 32, q, cache, cache, q).unwrap_err();
        assert!(e.0.contains("outside a cache"), "{}", e.0);

        // GQA must divide evenly or head -> kv-head mapping is undefined.
        let e = validate_attn_decode(8, 3, 64, 32, 32, q, cache, cache, q).unwrap_err();
        assert!(e.0.contains("not a multiple"), "{}", e.0);

        // The block is 256 threads; a wider head would index past its shared array.
        let e = validate_attn_decode(8, 2, 512, 32, 32, q, cache, cache, q).unwrap_err();
        assert!(e.0.contains("head_dim"), "{}", e.0);

        // Short KV buffers are caught here, not by the kernel's early return.
        let e = validate_attn_decode(8, 2, 64, 32, 32, q, cache - 4, cache, q).unwrap_err();
        assert!(e.0.contains("KV cache"), "{}", e.0);
    }

    #[test]
    fn every_ptx_entry_declares_its_block_size() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/cuda/kernels.ptx");
        let ptx = std::fs::read_to_string(&path).expect("read kernels.ptx");
        for name in PTX_ENTRIES {
            let start = ptx
                .find(&format!(".entry {name}("))
                .unwrap_or_else(|| panic!("kernels.ptx is missing {name}"));
            // .maxntid sits between the parameter list and the opening brace, so
            // the window runs to the body of this entry and no further.
            let body = ptx[start..].find('{').map(|i| start + i).unwrap_or(start);
            let window = &ptx[start..(body + 200).min(ptx.len())];
            assert!(
                window.contains(".maxntid"),
                "{name} has no .maxntid: regenerate with scripts/cuda/gen_ptx.sh \
                 after giving the kernel a SOFUU_LAUNCH_BOUNDS in kernels.cu"
            );
        }
    }

    #[test]
    fn checked_in_gzip_matches_ptx_source() {
        use std::io::Read;
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/cuda");
        let ptx = std::fs::read_to_string(base.join("kernels.ptx")).unwrap();
        let gz = std::fs::read(base.join("kernels.ptx.gz")).unwrap();
        let mut decoded = String::new();
        flate2::read::GzDecoder::new(&gz[..])
            .read_to_string(&mut decoded)
            .expect("inflate kernels.ptx.gz");
        assert_eq!(decoded, ptx, "kernels.ptx.gz is stale");
    }

    #[test]
    fn quant_layout_guards_are_honest() {
        assert_eq!(Quant::Q8_0.block_bytes(), 34);
        assert_eq!(Quant::Q4K.block_bytes(), 144);
        assert_eq!(Quant::Q6K.block_bytes(), 210);
        assert_eq!(Quant::Q4K.row_bytes(4096), Some(16 * 144));
        assert_eq!(Quant::Q4K.row_bytes(1000), None);
        assert_eq!(Quant::Q8_0.row_bytes(1024), Some(32 * 34));
        assert_eq!(Quant::F32.row_bytes(7), Some(28));
    }
}
