// crates/sofuu-core/src/infer/metal.rs — Safe Rust wrapper for Apple Silicon Metal GPU Engine.
//
// Target note (PLAN-EMBED-SDK E3): the engine behind this wrapper
// (src/metal/metal_engine.mm) compiles for macOS AND iOS, so
// `any(target_os = "macos", target_os = "ios")` appears throughout the
// inference engine wherever a Metal item means "this target has the GPU
// path". Before E3 the gates said macOS only and the iOS slices contained
// no Metal code at all — an iPhone ran CPU-only no matter what the config
// asked for. Widening them is what makes the same GPU path reachable on
// both. On an iOS older than 17 the bf16 kernels cannot build (MSL 3.1
// introduced `bfloat`); init then returns None and the caller runs CPU-only,
// which is the designed degradation, not a crash.

use std::ffi::c_void;
use super::quant::*;

pub const OP_KINDS: usize = 17;
pub const K_GGUF: usize = 0;
pub const K_GGUF_B4: usize = 1;
pub const K_GGUF_GEMM: usize = 2;
pub const K_RMSNORM: usize = 3;
pub const K_RMSNORM_B4: usize = 4;
pub const K_KVWRITE: usize = 5;
pub const K_KVWRITE_B4: usize = 6;
pub const K_HEAD_RMSNORM: usize = 7;
pub const K_ATTN: usize = 8;
pub const K_ATTN_PART: usize = 9;
pub const K_ATTN_FIN: usize = 10;
pub const K_ATTN_1C: usize = 11;
pub const K_ADD: usize = 12;
pub const K_SWIGLU: usize = 13;
pub const K_HEAD: usize = 14;
/// The sectioned mma8 form (one dispatch for q/k/v or gate/up): counted apart
/// from K_GGUF_B4 so a SOFUU_GPU_TIMING=2 census shows exactly what merged.
pub const K_GGUF_B4_SECS: usize = 15;
/// Qwen2/2.5 q/k/v bias epilogue (one dispatch per layer, activation slab
/// only). Booked apart so a SOFUU_GPU_TIMING=2 census shows what bias support
/// actually costs instead of hiding it inside "add".
pub const K_BIAS: usize = 16;
pub const OP_NAMES: [&str; OP_KINDS] = [
    "gguf", "gguf_b4", "gemm", "rmsnorm", "rmsnorm_b4", "kv_write", "kv_write_b4",
    "head_rmsnorm", "attn", "attn_part", "attn_fin", "attn_1c", "add", "swiglu",
    "head", "gguf_b4s", "chan_bias",
];

/// Column granularity a BATCHED GEMM actually needs, per weight type, read off
/// the kernels rather than assumed (src/metal/kernels.metal):
///
///   * q8_0 (ty 8) - every q8_0 GEMM body walks `nb = cols / 32` 34-byte
///     blocks, 32 values per block: gemm_q8_0, gemm_q8_0_multi, gemm_q8_0_acc,
///     gemm_q8_0_t32, gemm_q8_0_mma64 (`nb = cols / 32`, `nbk = cols / MM_BK`
///     with MM_BK = 32) and gemm_q8_0_tiled_fused (`nbk = cols / TILED_BK`
///     with TILED_BK = 32; its ROW stride is (cols/32)*34 bytes). => 32.
///   * q4_K (ty 12) - a 256-value super-block in a 144-byte block
///     (gemm_q4_k `nb = cols / 256`, gemm_q4_k_tiled*/mma*/mmq likewise,
///     f16_mma64_tile's q4_K branch `nb = cols / 256`). => 256.
///   * q6_K (ty 14) - 256 values in a 210-byte block (gemm_q6_k,
///     gemm_q6_k_multi, gemm_q6_k_acc, gemm_q6_k_t32, gemm_q6_k_tiled_fused
///     `nb = cols / 256`). => 256.
///
/// A type not in this list has no batched GEMM at all: refuse rather than
/// dispatch something that computes a partial dot product. The floor is a
/// HARD requirement - every kernel drops a `cols % block` tail, so a shape
/// below the floor is a silent wrong answer, not a slow one.
pub fn gemm_cols_floor(ty: u32) -> Option<usize> {
    match ty {
        GGML_TYPE_Q4_0 => Some(32),
        GGML_TYPE_Q8_0 => Some(32),
        GGML_TYPE_Q4_K => Some(256),
        GGML_TYPE_Q6_K => Some(256),
        _ => None,
    }
}

/// True when one batched GEMM dispatch can cover `cols` columns exactly for
/// this weight type. `false` is always safe (the caller keeps the per-row
/// GEMV walk, which covers arbitrary columns one row at a time).
pub fn gemm_cols_ok(ty: u32, cols: usize) -> bool {
    match gemm_cols_floor(ty) {
        Some(f) => cols != 0 && cols % f == 0,
        None => false,
    }
}

/// Loud, once-per-process note when a batched GEMM refuses a shape. The
/// fallback (one dispatch per token) is CORRECT but 3x slower than the CPU
/// tile path on a 960-wide q8_0 model, so a shape change must never silently
/// land there.
pub fn note_gemm_shape_refusal(ty: u32, cols: usize, entry: &str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static ONCE: AtomicBool = AtomicBool::new(false);
    if let Some(want) = gemm_cols_floor(ty) {
        if !ONCE.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[gemm-floor] {entry} refused: ty={ty} cols={cols} (needs cols % {want} == 0         for this quant block); that projection now walks one dispatch per token"
            );
        }
    }
}


pub struct MetalEngine {
    /// P0.2 per-op HOST enqueue timing (SOFUU_GPU_TIMING=2): nanos + count
    /// per op kind, reported by take_op_timing/print_op_timing.
    op_ns: [std::sync::atomic::AtomicU64; OP_KINDS],
    op_ct: [std::sync::atomic::AtomicU64; OP_KINDS],
    tgate: std::sync::atomic::AtomicU8,

    /// fp16 materialisation of the chain GEMM weights (SOFUU_GPU_F16W=1):
    /// set once at load, read-only afterwards. A OnceLock keeps the dispatch
    /// path a single atomic load and keeps MetalEngine Sync (it is a process
    /// global on some routes).
    pub f16w: std::sync::OnceLock<F16Weights>,

    ctx: *mut c_void,
}

/// The dequantised-weights arena. `map` keys a tensor by the byte offset the
/// chain passes (data_offset + tensor offset) and holds the slot inside `buf`
/// plus the shape it was built for, so a shape mismatch falls back instead of
/// reading the wrong rows.
pub struct F16Weights {
    pub buf: MetalBuffer,
    map: std::collections::HashMap<usize, (usize, usize, usize)>,
    pub bytes: usize,
}

impl F16Weights {
    /// Arena slot (byte offset) for one tensor, or None when it was never
    /// materialised or the shape differs.
    #[inline]
    fn slot(&self, w_offset: usize, rows: usize, cols: usize) -> Option<usize> {
        match self.map.get(&w_offset) {
            Some(&(off, r, c)) if r == rows && c == cols => Some(off),
            _ => None,
        }
    }
}

unsafe impl Send for MetalEngine {}
unsafe impl Sync for MetalEngine {}

/// LEDGER #192: clears the decode-scoped flat-GEMV force on drop, so every return path
/// (including the rejected-command-buffer ones) restores prefill behaviour.
pub struct FlatGuard;

impl Drop for FlatGuard {
    fn drop(&mut self) {
        MetalEngine::set_gemv_flat_forced(false);
    }
}

impl MetalEngine {
    /// One dispatch for a group of independent q4_K projections that share x and
    /// cols: secs is {w_off, y_off, rows} per section.
    pub fn batch_gguf_gemv_q4k_secs(
        &self, w_buf: &MetalBuffer, w_base: usize, x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, cols: usize, secs: &[[u32; 3]],
    ) -> bool {
        if secs.is_empty() { return false; }
        unsafe {
            sofuu_metal_batch_gguf_gemv_q4k_secs(
                self.ctx, w_buf.handle, w_base, x_buf.handle, x_offset, y_buf.handle,
                cols, secs.as_ptr() as *const u32, secs.len(),
            ) == 1
        }
    }

    /// Temporary instrumentation: achievable read bandwidth over the weight buffer
    /// at a given thread count, measured inside one commit (SOFUU_BW_PROBE).
    pub fn bw_probe(&self, w_buf: *mut c_void, nbytes: usize, threads: usize, iters: usize, lanes: usize, tgsz: usize, mode: usize, slice_blocks: usize) -> f32 {
        unsafe { sofuu_metal_bw_probe(self.ctx, w_buf, nbytes, threads, iters, lanes, tgsz, mode, slice_blocks) }
    }
    pub fn new() -> Option<Self> {
        unsafe {
            if !sofuu_metal_available() {
                return None;
            }
            let ctx = sofuu_metal_init();
            if ctx.is_null() {
                return None;
            }
            Some(Self {
                ctx,
                op_ns: Default::default(),
                op_ct: Default::default(),
                tgate: std::sync::atomic::AtomicU8::new(0),
                f16w: std::sync::OnceLock::new(),
            })
        }
    }

    /// P0.2: host-side enqueue tick — a scoped Instant, ns+count into the
    /// kind bucket, only when SOFUU_GPU_TIMING=2 (gate cached in an atomic).
    fn op_tick(&self, kind: usize) -> Option<std::time::Instant> {
        use std::sync::atomic::Ordering::Relaxed;
        let g = self.tgate.load(Relaxed);
        if g == 0 {
            let v = std::env::var("SOFUU_GPU_TIMING")
                .map(|x| x.trim() == "2")
                .unwrap_or(false) as u8;
            self.tgate.store(if v != 0 { 2 } else { 1 }, Relaxed);
            if v != 0 { return Some(std::time::Instant::now()); }
            return None;
        }
        if g == 2 { Some(std::time::Instant::now()) } else { None }
    }
    fn op_tock(&self, kind: usize, t0: Option<std::time::Instant>) {
        if let Some(t0) = t0 {
            use std::sync::atomic::Ordering::Relaxed;
            self.op_ns[kind].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
            self.op_ct[kind].fetch_add(1, Relaxed);
        }
    }
    /// P0.2 report: per op-kind host-enqueue totals; returns the text (the
    /// caller prints it so logs stay in one stream). Gate stays enabled.
    pub fn print_op_timing(&self, label: &str) {
        use std::sync::atomic::Ordering::Relaxed;
        let mut any = false;
        let mut out = format!("[op-timing {label}] kind count total_us avg_us");
        for k in 0..OP_KINDS {
            let ct = self.op_ct[k].swap(0, Relaxed);
            if ct == 0 { continue; }
            let ns = self.op_ns[k].swap(0, Relaxed);
            any = true;
            out.push_str(&format!(
                "\n[op-timing {label}] {} {} {:.0} {:.2}",
                OP_NAMES[k], ct, ns as f64 / 1e3, ns as f64 / 1e3 / ct as f64
            ));
        }
        if any { eprintln!("{out}"); }
    }

    pub fn register_zerocopy_buffer(&self, slice: &[u8]) -> Option<MetalBuffer> {
        unsafe {
            let buf = sofuu_metal_buffer_create_zerocopy(
                self.ctx,
                slice.as_ptr() as *const c_void,
                slice.len(),
            );
            if !buf.is_null() {
                return Some(MetalBuffer { handle: buf });
            }
            // No-copy is a shortcut, not a requirement: fall back to a
            // copying buffer so a platform that cannot wrap this memory still
            // gets the GPU path instead of losing it entirely. (The iOS
            // simulator's MTLSimDriver exports buffer storage over XPC shared
            // memory, and libxpc aborts the process rather than failing when
            // allocator memory cannot be exported as one mapping — see the
            // TARGET_OS_SIMULATOR notes in src/metal/metal_engine.mm.)
            self.create_buffer(slice)
        }
    }

    /// Zero-copy ONLY: None when the memory cannot be wrapped in place.
    /// The GPU decode clients require this — they write the same host pages
    /// the GPU reads (activation slab, KV handoff), so a copied buffer would
    /// desynchronize the two silently.
    pub fn register_zerocopy_only(&self, slice: &[u8]) -> Option<MetalBuffer> {
        unsafe {
            let buf = sofuu_metal_buffer_create_zerocopy(
                self.ctx,
                slice.as_ptr() as *const c_void,
                slice.len(),
            );
            if buf.is_null() {
                None
            } else {
                Some(MetalBuffer { handle: buf })
            }
        }
    }

    /// Always a driver-managed copy (never wraps the caller's pages). The
    /// whole-step decode client uses this for the weight region: whether
    /// device-owned placement streams faster than the wrapped mmap is a
    /// measured question, and byte offsets stay identical either way.
    pub fn create_buffer_copy(&self, slice: &[u8]) -> Option<MetalBuffer> {
        unsafe {
            let buf = sofuu_metal_buffer_create_copy(
                self.ctx,
                slice.as_ptr() as *const c_void,
                slice.len(),
            );
            if buf.is_null() {
                None
            } else {
                Some(MetalBuffer { handle: buf })
            }
        }
    }

    pub fn create_buffer(&self, slice: &[u8]) -> Option<MetalBuffer> {
        unsafe {
            let buf = sofuu_metal_buffer_create(
                self.ctx,
                slice.as_ptr() as *const c_void,
                slice.len(),
            );
            if buf.is_null() {
                None
            } else {
                Some(MetalBuffer { handle: buf })
            }
        }
    }

    pub fn gemv_bf16(
        &self,
        w_buf: &MetalBuffer,
        w_offset: usize,
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        unsafe {
            sofuu_metal_gemv_bf16(
                self.ctx,
                w_buf.handle,
                w_offset,
                x.as_ptr(),
                y.as_mut_ptr(),
                rows,
                cols,
            );
        }
    }

    pub fn gemv_qkv_bf16(
        &self,
        w_buf: &MetalBuffer,
        q_offset: usize,
        k_offset: usize,
        v_offset: usize,
        x: &[f32],
        q_out: &mut [f32],
        k_out: &mut [f32],
        v_out: &mut [f32],
        q_rows: usize,
        k_rows: usize,
        v_rows: usize,
        cols: usize,
    ) {
        unsafe {
            sofuu_metal_gemv_qkv_bf16(
                self.ctx,
                w_buf.handle,
                q_offset,
                k_offset,
                v_offset,
                x.as_ptr(),
                q_out.as_mut_ptr(),
                k_out.as_mut_ptr(),
                v_out.as_mut_ptr(),
                q_rows,
                k_rows,
                v_rows,
                cols,
            );
        }
    }

    pub fn gemv_gate_up_bf16(
        &self,
        w_buf: &MetalBuffer,
        gate_offset: usize,
        up_offset: usize,
        x: &[f32],
        gate_out: &mut [f32],
        up_out: &mut [f32],
        inter_rows: usize,
        cols: usize,
    ) {
        unsafe {
            sofuu_metal_gemv_gate_up_bf16(
                self.ctx,
                w_buf.handle,
                gate_offset,
                up_offset,
                x.as_ptr(),
                gate_out.as_mut_ptr(),
                up_out.as_mut_ptr(),
                inter_rows,
                cols,
            );
        }
    }

    pub fn gemv_f16(
        &self,
        w_buf: &MetalBuffer,
        w_offset: usize,
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        unsafe {
            sofuu_metal_gemv_f16(
                self.ctx,
                w_buf.handle,
                w_offset,
                x.as_ptr(),
                y.as_mut_ptr(),
                rows,
                cols,
            );
        }
    }

    pub fn gemv_f32(
        &self,
        w_buf: &MetalBuffer,
        w_offset: usize,
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        unsafe {
            sofuu_metal_gemv_f32(
                self.ctx,
                w_buf.handle,
                w_offset,
                x.as_ptr(),
                y.as_mut_ptr(),
                rows,
                cols,
            );
        }
    }

    pub fn rmsnorm(
        &self,
        x: &[f32],
        w_buf: &MetalBuffer,
        w_offset: usize,
        y: &mut [f32],
        dim: usize,
        eps: f32,
    ) {
        unsafe {
            sofuu_metal_rmsnorm(
                self.ctx,
                x.as_ptr(),
                w_buf.handle,
                w_offset,
                y.as_mut_ptr(),
                dim,
                eps,
            );
        }
    }

    pub fn swiglu(&self, g: &mut [f32], u: &[f32]) {
        unsafe {
            sofuu_metal_swiglu(
                self.ctx,
                g.as_mut_ptr(),
                u.as_ptr(),
                g.len().min(u.len()),
            );
        }
    }

    pub fn conv1d_3(
        &self,
        conv_w_buf: &MetalBuffer,
        conv_w_offset: usize,
        state: &[f32],
        out: &mut [f32],
        channels: usize,
    ) {
        unsafe {
            sofuu_metal_conv1d_3(
                self.ctx,
                conv_w_buf.handle,
                conv_w_offset,
                state.as_ptr(),
                out.as_mut_ptr(),
                channels,
            );
        }
    }

    pub fn gemv_mlx4(
        &self,
        w_buf: &MetalBuffer,
        w_offset: usize,
        scales_buf: &MetalBuffer,
        scales_offset: usize,
        biases_buf: &MetalBuffer,
        biases_offset: usize,
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        cols: usize,
        group_size: usize,
        is_bf16: bool,
    ) {
        unsafe {
            sofuu_metal_gemv_mlx4(
                self.ctx,
                w_buf.handle,
                w_offset,
                scales_buf.handle,
                scales_offset,
                biases_buf.handle,
                biases_offset,
                x.as_ptr(),
                y.as_mut_ptr(),
                rows,
                cols,
                group_size,
                is_bf16 as i32,
            );
        }
    }

    pub fn gemv_qkv_mlx4(
        &self,
        q_buf: &MetalBuffer, q_offset: usize, q_rows: usize,
        q_scales_buf: &MetalBuffer, q_scales_offset: usize,
        q_biases_buf: &MetalBuffer, q_biases_offset: usize,
        k_buf: &MetalBuffer, k_offset: usize, k_rows: usize,
        k_scales_buf: &MetalBuffer, k_scales_offset: usize,
        k_biases_buf: &MetalBuffer, k_biases_offset: usize,
        v_buf: &MetalBuffer, v_offset: usize, v_rows: usize,
        v_scales_buf: &MetalBuffer, v_scales_offset: usize,
        v_biases_buf: &MetalBuffer, v_biases_offset: usize,
        x: &[f32],
        q_out: &mut [f32], k_out: &mut [f32], v_out: &mut [f32],
        cols: usize, group_size: usize, is_bf16: bool,
    ) {
        unsafe {
            sofuu_metal_gemv_qkv_mlx4(
                self.ctx,
                q_buf.handle, q_offset, q_rows,
                q_scales_buf.handle, q_scales_offset,
                q_biases_buf.handle, q_biases_offset,
                k_buf.handle, k_offset, k_rows,
                k_scales_buf.handle, k_scales_offset,
                k_biases_buf.handle, k_biases_offset,
                v_buf.handle, v_offset, v_rows,
                v_scales_buf.handle, v_scales_offset,
                v_biases_buf.handle, v_biases_offset,
                x.as_ptr(),
                q_out.as_mut_ptr(), k_out.as_mut_ptr(), v_out.as_mut_ptr(),
                cols, group_size, is_bf16 as i32,
            );
        }
    }

    pub fn gemv_gate_up_mlx4(
        &self,
        gate_buf: &MetalBuffer, gate_offset: usize, gate_rows: usize,
        gate_scales_buf: &MetalBuffer, gate_scales_offset: usize,
        gate_biases_buf: &MetalBuffer, gate_biases_offset: usize,
        up_buf: &MetalBuffer, up_offset: usize, up_rows: usize,
        up_scales_buf: &MetalBuffer, up_scales_offset: usize,
        up_biases_buf: &MetalBuffer, up_biases_offset: usize,
        x: &[f32],
        gate_out: &mut [f32], up_out: &mut [f32],
        cols: usize, group_size: usize, is_bf16: bool,
    ) {
        unsafe {
            sofuu_metal_gemv_gate_up_mlx4(
                self.ctx,
                gate_buf.handle, gate_offset, gate_rows,
                gate_scales_buf.handle, gate_scales_offset,
                gate_biases_buf.handle, gate_biases_offset,
                up_buf.handle, up_offset, up_rows,
                up_scales_buf.handle, up_scales_offset,
                up_biases_buf.handle, up_biases_offset,
                x.as_ptr(),
                gate_out.as_mut_ptr(), up_out.as_mut_ptr(),
                cols, group_size, is_bf16 as i32,
            );
        }
    }

    pub fn gemv_qkv_multi(
        &self,
        q_buf: &MetalBuffer, q_offset: usize, q_rows: usize,
        k_buf: &MetalBuffer, k_offset: usize, k_rows: usize,
        v_buf: &MetalBuffer, v_offset: usize, v_rows: usize,
        x: &[f32],
        q_out: &mut [f32], k_out: &mut [f32], v_out: &mut [f32],
        cols: usize,
        dtype: i32,
    ) {
        unsafe {
            sofuu_metal_gemv_qkv_multi(
                self.ctx,
                q_buf.handle, q_offset, q_rows,
                k_buf.handle, k_offset, k_rows,
                v_buf.handle, v_offset, v_rows,
                x.as_ptr(),
                q_out.as_mut_ptr(), k_out.as_mut_ptr(), v_out.as_mut_ptr(),
                cols,
                dtype,
            );
        }
    }

    pub fn gemv_gate_up_multi(
        &self,
        gate_buf: &MetalBuffer, gate_offset: usize,
        up_buf: &MetalBuffer, up_offset: usize,
        x: &[f32],
        gate_out: &mut [f32], up_out: &mut [f32],
        inter_rows: usize,
        cols: usize,
        dtype: i32,
    ) {
        unsafe {
            sofuu_metal_gemv_gate_up_multi(
                self.ctx,
                gate_buf.handle, gate_offset,
                up_buf.handle, up_offset,
                x.as_ptr(),
                gate_out.as_mut_ptr(), up_out.as_mut_ptr(),
                inter_rows,
                cols,
                dtype,
            );
        }
    }

    pub fn gemv_async(
        &self,
        w_buf: &MetalBuffer,
        w_offset: usize,
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        cols: usize,
        dtype: i32,
    ) {
        unsafe {
            sofuu_metal_gemv_async(
                self.ctx,
                w_buf.handle,
                w_offset,
                x.as_ptr(),
                y.as_mut_ptr(),
                rows,
                cols,
                dtype,
            );
        }
    }

    pub fn sync(&self) {
        unsafe {
            sofuu_metal_sync(self.ctx);
        }
    }
}

static GLOBAL_METAL: std::sync::OnceLock<Option<MetalEngine>> = std::sync::OnceLock::new();

pub fn get_or_init_global_metal() -> Option<&'static MetalEngine> {
    GLOBAL_METAL.get_or_init(MetalEngine::new).as_ref()
}

impl Drop for MetalEngine {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                sofuu_metal_free(self.ctx);
                self.ctx = std::ptr::null_mut();
            }
        }
    }
}

pub struct MetalBuffer {
    pub handle: *mut c_void,
}

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                sofuu_metal_buffer_free(self.handle);
                self.handle = std::ptr::null_mut();
            }
        }
    }
}

// ── G0: real multi-op command-buffer batching ──────────────────────────────
// Phase G0 of PLAN-GPU-METAL: one GPU submission per decode step regardless
// of op count. Ops chain y→x with zero CPU staging (caller-owned MTLBuffers).

extern "C" {

    fn sofuu_metal_batch_barrier_mark(ctx: *mut c_void);
    fn sofuu_metal_batch_open(ctx: *mut c_void) -> i32;
    fn sofuu_metal_batch_gemv(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, dtype: i32,
    );
    fn sofuu_metal_batch_end(ctx: *mut c_void) -> i32;
    // W1 CPU+GPU row split (the only consumers are the split driver in
    // infer/model.rs): commit the open batch without waiting, open the next
    // batch on a fresh command buffer, drain everything committed so far.
    fn sofuu_metal_batch_commit_async(ctx: *mut c_void) -> i32;
    fn sofuu_metal_batch_open_async(ctx: *mut c_void) -> i32;
    fn sofuu_metal_batch_wait_all(ctx: *mut c_void) -> i32;
    fn sofuu_metal_batch_mlx4(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        s_buf: *mut c_void, s_offset: usize,
        b_buf: *mut c_void, b_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, group_size: usize, is_bf16: i32,
    );
    fn sofuu_metal_batch_gemv_staged(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x: *const f32,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, dtype: i32,
    );
    // GPU1: in-batch glue ops (whole-decode-step batches).
    fn sofuu_metal_batch_add(
        ctx: *mut c_void,
        acc_buf: *mut c_void, acc_offset: usize,
        add_buf: *mut c_void, add_offset: usize,
        n: usize,
    ) -> i32;
    fn sofuu_metal_batch_swiglu(
        ctx: *mut c_void,
        g_buf: *mut c_void, g_offset: usize,
        u_buf: *mut c_void, u_offset: usize,
        n: usize,
    ) -> i32;
    /// Qwen2/2.5 bias epilogue: y[t][j] += bias[j] over b token-major rows,
    /// up to four sections in one dispatch (f32 ELEMENT offsets).
    fn sofuu_metal_batch_add_chan_bias(
        ctx: *mut c_void,
        y_buf: *mut c_void,
        bias_buf: *mut c_void,
        secs: *const u32,
        nsec: usize,
        b: usize,
    ) -> i32;
    // W2 feature split: compact [rows][n] block -> column slice of the
    // canonical [rows][dst_stride] layout.
    fn sofuu_metal_batch_strided_copy_f32(
        ctx: *mut c_void,
        dst_buf: *mut c_void, dst_offset: usize,
        dst_stride: usize,
        src_buf: *mut c_void, src_offset: usize,
        src_stride: usize,
        n: usize, rows: usize,
    ) -> i32;
    // W2 feature split: y[t][j] += (j < n_a ? pa[t][j] : pb[t][j - n_a]).
    fn sofuu_metal_batch_resid_split_add_f32(
        ctx: *mut c_void,
        y_buf: *mut c_void, y_offset: usize,
        pa_buf: *mut c_void, pa_offset: usize,
        n_a: usize,
        pb_buf: *mut c_void, pb_offset: usize,
        n_b: usize,
        cols: usize, rows: usize,
    ) -> i32;
    fn sofuu_metal_batch_rmsnorm(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        w_buf: *mut c_void, w_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        dim: usize, eps: f32,
    );
    fn sofuu_metal_batch_conv_gate(
        ctx: *mut c_void,
        bcx_buf: *mut c_void, bcx_offset: usize,
        conv_w_buf: *mut c_void, conv_w_offset: usize,
        state_buf: *mut c_void, state_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        dim: usize,
    );
    fn sofuu_metal_batch_kv_write(
        ctx: *mut c_void,
        k_buf: *mut c_void, k_offset: usize,
        v_buf: *mut c_void, v_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        kln_buf: *mut c_void, kln_offset: usize,
        pos: u32, head_dim: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32,
    );
    fn sofuu_metal_batch_gguf_b4(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
    );
    fn sofuu_metal_batch_gguf_mma8(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_mma8_acc(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        acc_buf: *mut c_void, acc_offset: usize,
        rows: usize, cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_mma8_secs(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_base: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void,
        cols: usize, B: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    // fp16-input forms of the same tile (SOFUU_GPU_F16W): buffer 0 is the
    // materialised arena and the weight address is a plain half offset.
    fn sofuu_metal_batch_gguf_f16_mma8(
        ctx: *mut c_void,
        w16_buf: *mut c_void, w16_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_f16_mma8_acc(
        ctx: *mut c_void,
        w16_buf: *mut c_void, w16_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        acc_buf: *mut c_void, acc_offset: usize,
        rows: usize, cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_f16_mma8_secs(
        ctx: *mut c_void,
        w16_buf: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void,
        cols: usize, B: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    fn sofuu_metal_dequant_q4k_f16(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        rows: usize, cols: usize,
    ) -> i32;
    fn sofuu_metal_buffer_alloc(ctx: *mut c_void, size_bytes: usize) -> *mut c_void;
    fn sofuu_metal_batch_gguf_gemm(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_tiled(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_tiled_multi(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_base_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows_total: usize, cols: usize, B: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_tiled_acc(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        res_buf: *mut c_void, res_offset: usize, res_stride: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, B: usize,
    ) -> i32;
    // 8x4-tile MMQ mirror (SOFUU_GPU_TILE_MMQ=1)
    fn sofuu_metal_batch_quantize_q8_k(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_quantize_q8_k_pos(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gemm_q4_k_mmq(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        xq_buf: *mut c_void, xq_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gemm_q6_k_mmq(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        xq_buf: *mut c_void, xq_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_quantize_q8_ext(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        nb_total: usize,
    ) -> i32;
    fn sofuu_metal_batch_gemv_q4_k_mmq32(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        xq_buf: *mut c_void, xq_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
    ) -> i32;
    fn sofuu_metal_batch_gemv_q6_k_mmq32(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        xq_buf: *mut c_void, xq_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_multi(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_base_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows_total: usize, cols: usize, B: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_acc(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        res_buf: *mut c_void, res_offset: usize, res_stride: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_b4multi(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_base_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void,
        rows_total: usize, cols: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf6_b4multi(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_base_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void,
        rows_total: usize, cols: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf8_b4multi(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_base_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void,
        rows_total: usize, cols: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf8_b4r4(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_base_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void,
        rows_total: usize, cols: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    fn sofuu_metal_batch_llama_kv_write_b4(
        ctx: *mut c_void,
        k_buf: *mut c_void, k_offset: usize,
        v_buf: *mut c_void, v_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        kln_buf: *mut c_void, kln_offset: usize,
        has_norm: u32,
        pos_buf: *mut c_void, pos_offset: usize,
        rows: u32,
        head_dim: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
        rope_tab: *mut c_void, rope_tab_offset: usize, has_table: u32,
    );
    fn sofuu_metal_batch_head_rmsnorm_b4(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        w_buf: *mut c_void, w_offset: usize,
        pos_buf: *mut c_void, pos_offset: usize,
        rows: u32,
        head_dim: u32, n_heads: u32,
        has_norm: u32, eps: f32, rope_base: f32, rope_mode: u32,
        rope_tab: *mut c_void, rope_tab_offset: usize, has_table: u32,
    );
    fn sofuu_metal_batch_llama_attn_part_b4(
        ctx: *mut c_void,
        q_buf: *mut c_void, q_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        meta_buf: *mut c_void, meta_offset: usize,
        pos_buf: *mut c_void, pos_offset: usize,
        rows: u32,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        n_chunks_max: u32, meta_stride: u32,
    );
    fn sofuu_metal_batch_llama_attn_fin_b4(
        ctx: *mut c_void,
        meta_buf: *mut c_void, meta_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        pos_buf: *mut c_void, pos_offset: usize,
        rows: u32,
        n_heads: u32, head_dim: u32,
        n_chunks_max: u32, meta_stride: u32,
    );
    fn sofuu_metal_batch_llama_attn_1c_b4(
        ctx: *mut c_void,
        q_buf: *mut c_void, q_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        pos_buf: *mut c_void, pos_offset: usize,
        rows: u32,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        rope_tab: *mut c_void, rope_tab_offset: usize, has_table: u32,
    ) -> i32;
    fn sofuu_metal_peak_fma16(ctx: *mut c_void, out_buf: *mut c_void, n_blocks: usize) -> i32;
    fn sofuu_metal_peak_fma(ctx: *mut c_void, out_buf: *mut c_void, n_blocks: usize) -> i32;
    fn sofuu_metal_peak_mma(ctx: *mut c_void, out_buf: *mut c_void, n_blocks: usize) -> i32;
    fn sofuu_metal_batch_gguf_gemv_q4k_v4(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemv_q4k_v3(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
    ) -> i32;
    fn sofuu_metal_batch_add_rmsnorm_f32w(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_off: usize,
        r_buf: *mut c_void, r_off: usize,
        xs_buf: *mut c_void, xs_off: usize,
        y_buf: *mut c_void, y_off: usize,
        w_buf: *mut c_void, w_off: usize,
        dim: usize, eps: f32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn sofuu_metal_batch_llama_attn_w16(
        ctx: *mut c_void, q: *mut c_void, q_off: usize,
        kc: *mut c_void, kc_off: usize, vc: *mut c_void, vc_off: usize,
        qln: *mut c_void, qln_off: usize, out: *mut c_void, out_off: usize,
        has_norm: u32, pos: u32,
        head_dim: u32, n_heads: u32, n_kv: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> i32;
    fn sofuu_metal_batch_llama_attn_w8(
        ctx: *mut c_void, q: *mut c_void, q_off: usize,
        kc: *mut c_void, kc_off: usize, vc: *mut c_void, vc_off: usize,
        qln: *mut c_void, qln_off: usize, out: *mut c_void, out_off: usize,
        has_norm: u32, pos: u32,
        head_dim: u32, n_heads: u32, n_kv: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> i32;
    fn sofuu_metal_batch_llama_attn_w4(
        ctx: *mut c_void, q: *mut c_void, q_off: usize,
        kc: *mut c_void, kc_off: usize, vc: *mut c_void, vc_off: usize,
        qln: *mut c_void, qln_off: usize, out: *mut c_void, out_off: usize,
        has_norm: u32, pos: u32,
        head_dim: u32, n_heads: u32, n_kv: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> i32;
    fn sofuu_metal_batch_llama_attn_fast(
        ctx: *mut c_void, q: *mut c_void, q_off: usize,
        kc: *mut c_void, kc_off: usize, vc: *mut c_void, vc_off: usize,
        qln: *mut c_void, qln_off: usize, out: *mut c_void, out_off: usize,
        has_norm: u32, pos: u32,
        head_dim: u32, n_heads: u32, n_kv: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemv_q4k_flat(
        ctx: *mut c_void, w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize, y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemv_q4k_split(
        ctx: *mut c_void, w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize, y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemv_q4k_r8(
        ctx: *mut c_void, w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize, y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemv_q4k_secs(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_base: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void,
        cols: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    fn sofuu_metal_bw_probe(
        ctx: *mut c_void,
        w_buf: *mut c_void, nbytes: usize,
        threads: usize, iters: usize, lanes: usize, tgsz: usize, mode: usize, slice_blocks: usize,
    ) -> f32;
    fn sofuu_metal_batch_gguf_gemv_q4k_v2(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_mma64_acc(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        res_buf: *mut c_void, res_offset: usize, res_stride: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_mma64_multi(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_base_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows_total: usize, cols: usize, B: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_mma64(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
        B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_tiled_r88(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
        B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_tiled_wide_multi(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
        B: usize,
        secs: *const u32, n_sec: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_tiled_wide(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
        B: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf_gemm_mma(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
        B: usize,
    ) -> i32;
    fn sofuu_metal_batch_llama_attn_flash_b4(
        ctx: *mut c_void,
        q_buf: *mut c_void, q_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        pos_buf: *mut c_void, pos_offset: usize,
        rows: u32,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        has_table: u32,
    ) -> i32;
    fn sofuu_metal_batch_llama_attn_row(
        ctx: *mut c_void,
        q_buf: *mut c_void, q_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        pos_buf: *mut c_void, pos_offset: usize,
        rows: u32,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        out_stride: u32,
    ) -> i32;
    fn sofuu_metal_batch_llama_attn_w32_out(
        ctx: *mut c_void,
        q_buf: *mut c_void, q_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        pos_buf: *mut c_void, pos_offset: usize,
        rows: u32,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        out_stride: u32,
    ) -> i32;
    // GQA shape table (see metal_engine.h): the shipped selection rule and the
    // table itself, exposed so the generalized shapes are testable without a
    // GPU and without duplicating the table in Rust.
    fn sofuu_metal_attn_gqa_pick(ratio: u32, avail_mask: u32, allow_opt_in: i32) -> i32;
    // Same rule, but tells "no form for this ratio" (-1) from "the table has
    // forms and none of them is usable" (-2), which the dispatcher reports.
    fn sofuu_metal_attn_gqa_pick_ex(ratio: u32, avail_mask: u32, allow_opt_in: i32) -> i32;
    fn sofuu_metal_attn_gqa_count() -> u32;
    fn sofuu_metal_attn_gqa_threads(idx: i32) -> u32;
    fn sofuu_metal_attn_gqa_rg(idx: i32) -> u32;
    fn sofuu_metal_attn_gqa_ratio(idx: i32) -> u32;
    fn sofuu_metal_attn_gqa_last() -> i32;
    fn sofuu_metal_batch_gguf_b4r(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        acc_buf: *mut c_void, acc_offset: usize,
        rows: usize, cols: usize, acc_dim: usize,
    ) -> i32;
    fn sofuu_metal_batch_gguf(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize,
    ) -> i32;
    fn sofuu_metal_batch_attn(
        ctx: *mut c_void,
        q_buf: *mut c_void, q_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        qln_buf: *mut c_void, qln_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        pos: u32, head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32,
    );
    fn sofuu_metal_batch_rmsnorm_f32w(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        w_buf: *mut c_void, w_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        dim: usize, eps: f32,
    ) -> i32;
    fn sofuu_metal_batch_rmsnorm_f32w_b4(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        w_buf: *mut c_void, w_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        dim: usize, eps: f32, rows: usize,
    ) -> i32;
    fn sofuu_metal_batch_llama_kv_write(
        ctx: *mut c_void,
        k_buf: *mut c_void, k_offset: usize,
        v_buf: *mut c_void, v_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        kln_buf: *mut c_void, kln_offset: usize,
        has_norm: u32, pos: u32,
        head_dim: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> i32;
    fn sofuu_metal_batch_head_rmsnorm(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        w_buf: *mut c_void, w_offset: usize,
        head_dim: u32, n_heads: u32,
        has_norm: u32, pos: u32, eps: f32, rope_base: f32, rope_mode: u32,
    ) -> i32;
    fn sofuu_metal_batch_llama_attn_part(
        ctx: *mut c_void,
        q_buf: *mut c_void, q_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        qln_buf: *mut c_void, qln_offset: usize,
        meta_buf: *mut c_void, meta_offset: usize,
        has_norm: u32, pos: u32,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> i32;
    fn sofuu_metal_batch_llama_attn_fin(
        ctx: *mut c_void,
        meta_buf: *mut c_void, meta_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        pos_alt: u32, n_heads: u32, head_dim: u32,
    ) -> i32;
    fn sofuu_metal_batch_llama_attn(
        ctx: *mut c_void,
        q_buf: *mut c_void, q_offset: usize,
        kc_buf: *mut c_void, kc_layer_offset: usize,
        vc_buf: *mut c_void, vc_layer_offset: usize,
        qln_buf: *mut c_void, qln_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        has_norm: u32, pos: u32,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    );
}

impl MetalEngine {
    /// Open a batch: subsequent enqueue calls append to ONE command buffer.

    /// LEDGER #190: emit the per-layer buffer barrier (one per layer, not per op).
    pub fn batch_barrier_mark(&self) {
        unsafe {
            sofuu_metal_batch_barrier_mark(self.ctx);
        }
    }

    pub fn batch_open(&self) -> bool {
        unsafe { sofuu_metal_batch_open(self.ctx) == 0 }
    }

    /// Append a GEMV op: y = W·x, both staged in caller-owned MTLBuffers.
    /// x may be a previous op's y (chained without CPU round trips).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_gemv(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, dtype: i32,
    ) {
        unsafe {
            sofuu_metal_batch_gemv(
                self.ctx,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows, cols, dtype,
            );
        }
    }

    /// End the batch: end encoding, commit once, wait. Returns false when
    /// the GPU rejected the command buffer — outputs are then untouched and
    /// the caller MUST treat the whole pass as not-happened (fall back to
    /// the CPU route; reading the slab after a failed buffer is how garbage
    /// KV/logits reached the CPU repack under load).
    pub fn batch_end(&self) -> bool {
        unsafe { sofuu_metal_batch_end(self.ctx) != 0 }
    }

    /// W1 row split: commit the open batch and RETURN without waiting. The
    /// host keeps running (the CPU's rows) while this command buffer executes;
    /// the queue is serial, so the next batch_open_async batch runs after it.
    pub fn batch_commit_async(&self) -> bool {
        unsafe { sofuu_metal_batch_commit_async(self.ctx) != 0 }
    }

    /// W1 row split: open the next batch on a NEW command buffer without
    /// draining the ones already committed.
    pub fn batch_open_async(&self) -> bool {
        unsafe { sofuu_metal_batch_open_async(self.ctx) == 0 }
    }

    /// W1 row split: drain every committed-but-unwaited command buffer.
    /// Returns false when any of them failed to complete (outputs untouched ->
    /// the caller must treat the pass as not-happened).
    pub fn batch_wait_all(&self) -> bool {
        unsafe { sofuu_metal_batch_wait_all(self.ctx) == 0 }
    }

    /// Batched mlx4 GEMV op (binds mirror the per-op kernel exactly).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_mlx4(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        s_buf: &MetalBuffer, s_offset: usize,
        b_buf: &MetalBuffer, b_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, group_size: usize, is_bf16: bool,
    ) {
        unsafe {
            sofuu_metal_batch_mlx4(
                self.ctx,
                w_buf.handle, w_offset,
                s_buf.handle, s_offset,
                b_buf.handle, b_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows, cols, group_size, is_bf16 as i32,
            );
        }
    }

    /// First-op GEMV with CPU-staged x (memcpy into the private internal
    /// buffer) — GPU-written buffers chain the rest of the batch.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_gemv_staged(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        x: &[f32],
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, dtype: i32,
    ) {
        unsafe {
            sofuu_metal_batch_gemv_staged(
                self.ctx,
                w_buf.handle, w_offset,
                x.as_ptr(),
                y_buf.handle, y_offset,
                rows, cols, dtype,
            );
        }
    }

    // ── GPU1: in-batch glue ops (whole-decode-step batches) ────────────────
    // All offsets are BYTE offsets into caller MTLBuffers. Each op inserts a
    // memory barrier before dispatch, so chained reads see prior writes.
    // The CPU touches the slab only between batch_end and the next open.

    pub fn batch_add(
        &self,
        acc_buf: &MetalBuffer, acc_offset: usize,
        add_buf: &MetalBuffer, add_offset: usize,
        n: usize,
    ) -> bool {        let __t = self.op_tick(K_ADD);
        let __r = unsafe { sofuu_metal_batch_add(
                self.ctx,
                acc_buf.handle, acc_offset,
                add_buf.handle, add_offset,
                n,
            ) == 1 };
        self.op_tock(K_ADD, __t);
        __r
    }

    /// Qwen2/2.5 attention bias: after the sectioned q/k/v GEMM has written
    /// [b][rows] into each slot, add the per-output-channel bias. ONE dispatch
    /// covers q, k and v (the section list); it reads only the activation slab
    /// and the f32 bias vectors and never touches a weight byte. It is the
    /// cheap shape of the change as well as the safe one: folding it into the
    /// GEMM epilogue would have meant teaching every chain arm
    /// (mma8 / sectioned / tiled / b4 / per-row) a new parameter, while the
    /// additive pass performs the CPU's own y[i] += b[i] in the same f32
    /// order, so the result is bit-identical to the CPU glue.
    /// secs = (y_off, bias_off, rows, 0) per section, f32 ELEMENT units.
    pub fn batch_add_chan_bias(
        &self,
        y_buf: &MetalBuffer,
        bias_buf: &MetalBuffer,
        secs: &[[u32; 4]],
        b: usize,
    ) -> bool {
        let __t = self.op_tick(K_BIAS);
        let mut flat: [u32; 16] = [0; 16];
        for (i, s) in secs.iter().take(4).enumerate() {
            flat[i * 4..i * 4 + 4].copy_from_slice(s);
        }
        let __r = unsafe {
            sofuu_metal_batch_add_chan_bias(
                self.ctx,
                y_buf.handle,
                bias_buf.handle,
                flat.as_ptr(),
                secs.len(),
                b,
            ) == 1
        };
        self.op_tock(K_BIAS, __t);
        __r
    }

    pub fn batch_swiglu(
        &self,
        g_buf: &MetalBuffer, g_offset: usize,
        u_buf: &MetalBuffer, u_offset: usize,
        n: usize,
    ) -> bool {        let __t = self.op_tick(K_SWIGLU);
        let __r = unsafe { sofuu_metal_batch_swiglu(
                self.ctx,
                g_buf.handle, g_offset,
                u_buf.handle, u_offset,
                n,
            ) == 1 };
        self.op_tock(K_SWIGLU, __t);
        __r
    }

    /// W2 feature split: copy a compact [rows][n] block into a column slice of
    /// a canonical [rows][dst_stride] block (offsets/strides in ELEMENTS).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_strided_copy_f32(
        &self,
        dst_buf: &MetalBuffer, dst_offset: usize, dst_stride: usize,
        src_buf: &MetalBuffer, src_offset: usize, src_stride: usize,
        n: usize, rows: usize,
    ) -> bool {
        let __t = self.op_tick(K_ADD);
        let __r = unsafe {
            sofuu_metal_batch_strided_copy_f32(
                self.ctx,
                dst_buf.handle, dst_offset, dst_stride,
                src_buf.handle, src_offset, src_stride,
                n, rows,
            ) == 1
        };
        self.op_tock(K_ADD, __t);
        __r
    }

    /// W2 feature split: fold a two-owner residual partial pair into y.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_resid_split_add_f32(
        &self,
        y_buf: &MetalBuffer, y_offset: usize,
        pa_buf: &MetalBuffer, pa_offset: usize, n_a: usize,
        pb_buf: &MetalBuffer, pb_offset: usize, n_b: usize,
        cols: usize, rows: usize,
    ) -> bool {
        let __t = self.op_tick(K_ADD);
        let __r = unsafe {
            sofuu_metal_batch_resid_split_add_f32(
                self.ctx,
                y_buf.handle, y_offset,
                pa_buf.handle, pa_offset, n_a,
                pb_buf.handle, pb_offset, n_b,
                cols, rows,
            ) == 1
        };
        self.op_tock(K_ADD, __t);
        __r
    }

    /// GGUF dequant-dot GEMV op in the live batch (ty: 2=q4_0 8=q8_0
    /// 12=q4_K 14=q6_K).

    pub fn batch_gguf(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize,
    ) -> bool {
        if std::env::var_os("SOFUU_GGUF_DBG").is_some() {
            Self::gguf_histogram().lock().unwrap().entry((ty, rows, cols)).and_modify(|c| *c += 1).or_insert(1);
        }
        if ty == 12 && (Self::gemv_flat_enabled() || Self::gemv_flat_forced()) && cols % 256 == 0
            && unsafe {
                sofuu_metal_batch_gguf_gemv_q4k_flat(
                    self.ctx, w_buf.handle, w_offset, x_buf.handle, x_offset,
                    y_buf.handle, y_offset, rows, cols,
                ) == 1
            }
        {
            return true;
        }
        if ty == 12 && Self::gemv_split_enabled() && cols % 256 == 0
            && unsafe {
                sofuu_metal_batch_gguf_gemv_q4k_split(
                    self.ctx, w_buf.handle, w_offset, x_buf.handle, x_offset,
                    y_buf.handle, y_offset, rows, cols,
                ) == 1
            }
        {
            return true;
        }
        if ty == 12 && Self::gemv_r8_enabled() && cols % 256 == 0
            && unsafe {
                sofuu_metal_batch_gguf_gemv_q4k_r8(
                    self.ctx, w_buf.handle, w_offset, x_buf.handle, x_offset,
                    y_buf.handle, y_offset, rows, cols,
                ) == 1
            }
        {
            return true;
        }
        if ty == 12 && Self::gemv_v4_enabled() && cols % 256 == 0 && cols <= 4096
            && unsafe {
                sofuu_metal_batch_gguf_gemv_q4k_v4(
                    self.ctx, w_buf.handle, w_offset, x_buf.handle, x_offset,
                    y_buf.handle, y_offset, rows, cols,
                ) == 1
            }
        {
            return true;
        }
        if ty == 12 && Self::gemv_v3_enabled() && cols % 256 == 0
            && unsafe {
                sofuu_metal_batch_gguf_gemv_q4k_v3(
                    self.ctx, w_buf.handle, w_offset, x_buf.handle, x_offset,
                    y_buf.handle, y_offset, rows, cols,
                ) == 1
            }
        {
            return true;
        }
        if ty == 12 && Self::gemv_v2_enabled() && cols % 256 == 0
            && unsafe {
                sofuu_metal_batch_gguf_gemv_q4k_v2(
                    self.ctx, w_buf.handle, w_offset, x_buf.handle, x_offset,
                    y_buf.handle, y_offset, rows, cols,
                ) == 1
            }
        {
            return true;
        }
        let __t = self.op_tick(K_GGUF);
        let __r = unsafe { sofuu_metal_batch_gguf(
                self.ctx,
                ty,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows, cols,
            ) == 1 };
        self.op_tock(K_GGUF, __t);
        __r
    }

    /// Residual add + RMSNorm in one dispatch.
    pub fn batch_add_rmsnorm_f32w(
        &self,
        x_buf: &MetalBuffer, x_off: usize,
        r_buf: &MetalBuffer, r_off: usize,
        xs_buf: &MetalBuffer, xs_off: usize,
        y_buf: &MetalBuffer, y_off: usize,
        w_buf: &MetalBuffer, w_off: usize,
        dim: usize, eps: f32,
    ) -> bool {
        unsafe {
            sofuu_metal_batch_add_rmsnorm_f32w(
                self.ctx, x_buf.handle, x_off, r_buf.handle, r_off,
                xs_buf.handle, xs_off, y_buf.handle, y_off,
                w_buf.handle, w_off, dim, eps,
            ) == 1
        }
    }

    pub fn batch_rmsnorm(
        &self,
        x_buf: &MetalBuffer, x_offset: usize,
        w_buf: &MetalBuffer, w_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        dim: usize, eps: f32,
    ) {
        unsafe {
            sofuu_metal_batch_rmsnorm(
                self.ctx,
                x_buf.handle, x_offset,
                w_buf.handle, w_offset,
                y_buf.handle, y_offset,
                dim, eps,
            );
        }
    }

    pub fn batch_conv_gate(
        &self,
        bcx_buf: &MetalBuffer, bcx_offset: usize,
        conv_w_buf: &MetalBuffer, conv_w_offset: usize,
        state_buf: &MetalBuffer, state_offset: usize,
        out_buf: &MetalBuffer, out_offset: usize,
        dim: usize,
    ) {
        unsafe {
            sofuu_metal_batch_conv_gate(
                self.ctx,
                bcx_buf.handle, bcx_offset,
                conv_w_buf.handle, conv_w_offset,
                state_buf.handle, state_offset,
                out_buf.handle, out_offset,
                dim,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_kv_write(
        &self,
        k_buf: &MetalBuffer, k_offset: usize,
        v_buf: &MetalBuffer, v_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        kln_buf: &MetalBuffer, kln_offset: usize,
        pos: u32, head_dim: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32,
    ) {
        unsafe {
            sofuu_metal_batch_kv_write(
                self.ctx,
                k_buf.handle, k_offset,
                v_buf.handle, v_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                kln_buf.handle, kln_offset,
                pos, head_dim, n_kv_heads, max_seq,
                eps, rope_base,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_attn(
        &self,
        q_buf: &MetalBuffer, q_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        qln_buf: &MetalBuffer, qln_offset: usize,
        out_buf: &MetalBuffer, out_offset: usize,
        pos: u32, head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32,
    ) {
        unsafe {
            sofuu_metal_batch_attn(
                self.ctx,
                q_buf.handle, q_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                qln_buf.handle, qln_offset,
                out_buf.handle, out_offset,
                pos, head_dim, n_heads, n_kv_heads, max_seq,
                eps, rope_base,
            );
        }
    }

    /// GPU3: batched (B=4) q4_K GEMV. Token-major layouts match the
    /// engine's x_batch/y_batch slices: x[4][cols] -> y[4][rows].
    pub fn mma8_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("SOFUU_GEMM_MMA8").map(|v| v == "1").unwrap_or(false))
    }

    pub fn batch_gguf_mma8(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {
        let __t = self.op_tick(K_GGUF_B4);
        let mut __r = false;
        if let Some((wh, off)) = self.f16_slot(w_offset, rows, cols, b) {
            __r = unsafe {
                sofuu_metal_batch_gguf_f16_mma8(
                    self.ctx,
                    wh, off,
                    x_buf.handle, x_offset,
                    y_buf.handle, y_offset,
                    rows, cols, b,
                ) == 1
            };
        }
        if !__r {
            __r = unsafe {
                sofuu_metal_batch_gguf_mma8(
                    self.ctx,
                    w_buf.handle, w_offset,
                    x_buf.handle, x_offset,
                    y_buf.handle, y_offset,
                    rows, cols, b,
                ) == 1
            };
        }
        self.op_tock(K_GGUF_B4, __t);
        // LEDGER #179 test: serialise every MMA dispatch to test for a WAR hazard
        // between the host's slab writes and the GPU's reads inside one deferred batch.
        if std::env::var_os("SOFUU_SYNC_EACH").is_some() {
            self.sync();
        }
        __r
    }

    /// mma8 with the residual add folded into the store: same arithmetic as
    /// batch_gguf_mma8 followed by batch_add, one dispatch shorter and without
    /// the y-slot round trip. Only for rows == the residual width (o, down).
    pub fn batch_gguf_mma8_acc(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        acc_buf: &MetalBuffer, acc_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {
        let __t = self.op_tick(K_GGUF_B4);
        let mut __r = false;
        if let Some((wh, off)) = self.f16_slot(w_offset, rows, cols, b) {
            __r = unsafe {
                sofuu_metal_batch_gguf_f16_mma8_acc(
                    self.ctx,
                    wh, off,
                    x_buf.handle, x_offset,
                    acc_buf.handle, acc_offset,
                    rows, cols, b,
                ) == 1
            };
        }
        if !__r {
            __r = unsafe {
                sofuu_metal_batch_gguf_mma8_acc(
                    self.ctx,
                    w_buf.handle, w_offset,
                    x_buf.handle, x_offset,
                    acc_buf.handle, acc_offset,
                    rows, cols, b,
                ) == 1
            };
        }
        if std::env::var_os("SOFUU_SYNC_EACH").is_some() {
            self.sync();
        }
        __r
    }

    /// ONE dispatch for a group of q4_K projections that share x and cols and
    /// write disjoint y slots: q/k/v off the normed input, gate/up off the
    /// ffn-normed one. Same kernel body, tile and accumulation order as
    /// batch_gguf_mma8, so each section's result is bit-identical to the
    /// three-dispatches form; it only removes dispatch ramps and barriers.
    /// secs: {w_off - w_base (256-aligned), w_rem, rows, y_off bytes} per section.
    pub fn batch_gguf_mma8_secs(
        &self,
        w_buf: &MetalBuffer, w_base: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer,
        cols: usize, b: usize,
        secs: &[[u32; 4]],
    ) -> bool {
        if secs.is_empty() { return false; }
        let __t = self.op_tick(K_GGUF_B4_SECS);
        let mut __r = false;
        // fp16 arm: every section must be materialised (the tile reads one
        // arena), so the translation is all-or-nothing. secs[0] becomes the
        // arena byte offset, secs[1] (the q4_K 256-byte remainder) is unused.
        if let Some((wh, _)) = self.f16_slot(
            w_base + secs[0][0] as usize + secs[0][1] as usize,
            secs[0][2] as usize, cols, b,
        ) {
            if let Some(w) = self.f16w.get() {
                let mut fsecs: Vec<[u32; 4]> = Vec::with_capacity(secs.len());
                let mut all = true;
                for s in secs {
                    let raw = w_base + s[0] as usize + s[1] as usize;
                    match w.slot(raw, s[2] as usize, cols) {
                        Some(off) if off <= u32::MAX as usize => {
                            fsecs.push([off as u32, 0u32, s[2], s[3]]);
                        }
                        _ => { all = false; break; }
                    }
                }
                if all {
                    __r = unsafe {
                        sofuu_metal_batch_gguf_f16_mma8_secs(
                            self.ctx,
                            wh,
                            x_buf.handle, x_offset,
                            y_buf.handle,
                            cols, b,
                            fsecs.as_ptr() as *const u32, fsecs.len(),
                        ) == 1
                    };
                }
            }
        }
        if !__r {
            __r = unsafe {
                sofuu_metal_batch_gguf_mma8_secs(
                    self.ctx,
                    w_buf.handle, w_base,
                    x_buf.handle, x_offset,
                    y_buf.handle,
                    cols, b,
                    secs.as_ptr() as *const u32, secs.len(),
                ) == 1
            };
        }
        if std::env::var_os("SOFUU_SYNC_EACH").is_some() {
            self.sync();
        }
        __r
    }

    /// fp16 materialisation is ON by default: measured bit-identical (whole-vocab
    /// FNV f0a88a5bd964de90 in both arms, 40/40 runs) and +4.5..+5.5% prefill on
    /// the chain route (in-situ GEMM -10.7%, chunk -5.9%), order-balanced.
    /// SOFUU_GPU_F16W=0 turns it off. Cost is one arena, ~728 MB for a 0.6B Q4_K_M
    /// model (2 B/param for the q4_K chain matrices only), capped by
    /// SOFUU_GPU_F16W_MAX_MB; over cap, an allocation failure or a rejected
    /// dispatch each print "[f16w] refused: ..." and leave the q4_K path in charge.
    pub fn f16_weights_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("SOFUU_GPU_F16W").map(|v| v != "0").unwrap_or(true)
        })
    }

    /// Arena slot for one tensor in the chain GEMM form, or None to stay on the
    /// q4_K path. `ntok` gates the diversion: below a 16-token chunk the chain
    /// walks the b4 GEMV class, and an fp16 buffer would only double the bytes
    /// that class streams.

    #[inline]
    fn f16_slot(
        &self, w_offset: usize, rows: usize, cols: usize, ntok: usize,
    ) -> Option<(*mut c_void, usize)> {
        let Some(w) = self.f16w.get() else { return None; };
        w.slot(w_offset, rows, cols).map(|off| (w.buf.handle, off))
    }

    /// Materialise every (weight byte offset, rows, cols) q4_K matrix into ONE
    /// fp16 arena. One dispatch per tensor, one commit for all of them.
    /// Returns the arena bytes on success. Any refusal (flag off, over the cap,
    /// allocation failure, rejected dispatch) returns None and leaves the q4_K
    /// path in charge — never a half-materialised table.
    pub fn materialize_f16(
        &self,
        w_buf: &MetalBuffer,
        tensors: &[(usize, usize, usize)],
    ) -> Option<usize> {
        if !Self::f16_weights_enabled() { return None; }
        if tensors.is_empty() { return None; }
        // 2 KiB per-tensor alignment: 16 B is all the tile needs, the slack
        // keeps every slot on a page-ish boundary for the bind offsets.
        const SLOT_ALIGN: usize = 2048;
        let mut total = 0usize;
        for &(_, rows, cols) in tensors {
            if rows == 0 || cols == 0 || cols % 256 != 0 { return None; }
            total += ((rows * cols * 2 + SLOT_ALIGN - 1) / SLOT_ALIGN) * SLOT_ALIGN;
        }
        let cap_mb: usize = std::env::var("SOFUU_GPU_F16W_MAX_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        // The section table carries arena offsets as u32, so 4 GiB is a hard
        // ceiling on top of the memory cap.
        let cap = (cap_mb * 1024 * 1024).min(u32::MAX as usize);
        if total > cap {
            eprintln!(
                "[f16w] refused: {} MB of fp16 weights exceeds the {cap_mb} MB cap (SOFUU_GPU_F16W_MAX_MB); staying on the q4_K path",
                total / (1024 * 1024)
            );
            return None;
        }
        let handle = unsafe { sofuu_metal_buffer_alloc(self.ctx, total) };
        if handle.is_null() {
            eprintln!("[f16w] refused: {} MB arena allocation failed; staying on the q4_K path",
                total / (1024 * 1024));
            return None;
        }
        let buf = MetalBuffer { handle };
        let t0 = std::time::Instant::now();
        let mut map = std::collections::HashMap::with_capacity(tensors.len());
        let mut off = 0usize;
        let mut ok = self.batch_open();
        if ok {
            for &(w_off, rows, cols) in tensors {
                if !unsafe {
                    sofuu_metal_dequant_q4k_f16(
                        self.ctx, w_buf.handle, w_off, buf.handle, off, rows, cols,
                    ) == 1
                } {
                    eprintln!("[f16w] refused: materialisation dispatch rejected at {}x{}; staying on the q4_K path", rows, cols);
                    ok = false;
                    break;
                }
                map.insert(w_off, (off, rows, cols));
                off += ((rows * cols * 2 + SLOT_ALIGN - 1) / SLOT_ALIGN) * SLOT_ALIGN;
            }
        }
        let committed = ok && self.batch_end();
        if !ok || !committed {
            eprintln!("[f16w] refused: materialisation batch failed; staying on the q4_K path");
            return None;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let gbs = if ms > 0.0 { (total + total / 2) as f64 / 1e9 / (ms / 1e3) } else { 0.0 };
        // SOFUU_GPU_F16W_DRY=1 keeps the arena and its 728 MB resident but does
        // not install the table, so every dispatch stays on q4_K. That isolates
        // "the memory cost of the arena" from "the fp16 kernels" in one binary.
        if std::env::var_os("SOFUU_GPU_F16W_DRY").is_some() {
            eprintln!("[f16w] DRY run: arena built, table not installed (q4_K path in charge)");
            return Some(total);
        }
        let _ = self.f16w.set(F16Weights { buf, map, bytes: total });
        eprintln!(
            "[f16w] materialised {} q4_K matrices -> {} MB fp16 arena in {ms:.1} ms ({} GB/s effective, cap {cap_mb} MB)",
            tensors.len(),
            total / (1024 * 1024),
            gbs as u64
        );
        Some(total)
    }

    pub fn batch_gguf_b4(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize,
    ) {        let __t = self.op_tick(K_GGUF_B4);
        unsafe {
            sofuu_metal_batch_gguf_b4(
                self.ctx,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows, cols,
            );
        }
        self.op_tock(K_GGUF_B4, __t);
    }

    /// Prefill GEMM (P1): y[t][row] = dot(W[row], X[t]) for B tokens —
    /// x is [B x cols], y is [B x rows], row-major, one op in the live
    /// batch. ty is the GGML type id (q4_K = 12). Returns false when the
    /// kernel class is unavailable (metallib predates it) or args are bad —
    /// callers must fall back, never assume the output was written.
    pub fn batch_gguf_gemm(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {
        // Per-type floor, not a blanket 256: q8_0 tiles 32 values, q4_K/q6_K a
        // 256-value super-block (see gemm_cols_floor). The .mm dispatch itself
        // has no column guard, so without this a cols % 32 != 0 q8_0 shape
        // would dispatch a kernel that silently drops its tail.
        if !gemm_cols_ok(ty, cols) {
            note_gemm_shape_refusal(ty, cols, "batch_gguf_gemm");
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm(
                self.ctx,
                ty,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// 8x4-tile MMQ mirror: act-quantize [B x cols] f32 rows into per-256
    /// TileQ8K blocks (d + qs[256], stride 260B). y region must be
    /// B*(cols/256)*260 bytes.
    pub fn batch_quantize_q8_k(
        &self,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        cols: usize, b: usize,
    ) -> bool {        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_quantize_q8_k(
                self.ctx,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// q4_K gemm over TileQ8K activations mirroring the CPU tiles' integer/
    /// f32 order bit-for-bit (y = [B x rows]).
    pub fn batch_gemm_q4_k_mmq(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        xq_buf: &MetalBuffer, xq_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gemm_q4_k_mmq(
                self.ctx,
                w_buf.handle, w_offset,
                xq_buf.handle, xq_offset,
                y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// q6_K MMQ mirror over TileQ8K activations (bit-twin of the CPU q6_K tile
    /// pipeline; y = [B x rows]).
    pub fn batch_gemm_q6_k_mmq(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        xq_buf: &MetalBuffer, xq_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gemm_q6_k_mmq(
                self.ctx,
                w_buf.handle, w_offset,
                xq_buf.handle, xq_offset,
                y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// Per-32 act-quant (a76 class): B*cols/32 Q8Ext blocks (stride 40B).
    pub fn batch_quantize_q8_ext(
        &self,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        nb_total: usize,
    ) -> bool {        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_quantize_q8_ext(
                self.ctx,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                nb_total,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// Positive-d row-quantizer variant (sofuu_quantize_row_q8_K semantics):
    /// d = amax/127, rint(x*127/amax) — the CPU tile GEMV's convention.
    pub fn batch_quantize_q8_k_pos(
        &self,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        cols: usize, b: usize,
    ) -> bool {        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_quantize_q8_k_pos(
                self.ctx,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// q4_K decode gemv mirror: consumes Q8Ext activations, mirrors the CPU
    /// dotprod-a76 summation order bit-for-bit (y = [rows]).
    pub fn batch_gemv_q4_k_mmq32(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        xq_buf: &MetalBuffer, xq_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize,
    ) -> bool {        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gemv_q4_k_mmq32(
                self.ctx,
                w_buf.handle, w_offset,
                xq_buf.handle, xq_offset,
                y_buf.handle, y_offset,
                rows, cols,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// q6_K decode gemv mirror (same class contract as q4).
    pub fn batch_gemv_q6_k_mmq32(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        xq_buf: &MetalBuffer, xq_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize,
    ) -> bool {        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gemv_q6_k_mmq32(
                self.ctx,
                w_buf.handle, w_offset,
                xq_buf.handle, xq_offset,
                y_buf.handle, y_offset,
                rows, cols,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// P1 op-diet: fused multi-section GEMM (qkv / gate_up). Sections carry
    /// their own 256-aligned weight byte offset + remainder + cumulative row
    /// end + ABSOLUTE y-slab byte offset + y row stride, so one dispatch
    /// fills the per-region slots the glue kernels already read. Returns
    /// false when unavailable (falls back to per-tensor gemm calls).
    /// P4 arm-d switch (SOFUU_GPU_GEMM_TILED): the tiled GEMM class
    /// (32-row x 64-token threadgroup tile). Parsed once.
    fn gemm_tiled_enabled() -> bool {
        static CACHE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
        match CACHE.load(std::sync::atomic::Ordering::Relaxed) {
            0 => false,
            1 => true,
            _ => {
                let on = std::env::var("SOFUU_GPU_GEMM_TILED")
                    .map(|v| v != "0")
                    .unwrap_or(false);
                CACHE.store(on as u8, std::sync::atomic::Ordering::Relaxed);
                on
            }
        }
    }

    /// P4 arm-d: tiled multi-section GEMM (qkv / gate_up fusion) — same
    /// section-table ABI as batch_gguf_gemm_multi, 8x less weight traffic.
    pub fn batch_gguf_gemm_tiled_multi(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_base_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows_total: usize, cols: usize, b: usize,
        secs: &[[u32; 5]],
    ) -> bool {
        // b < 16: the 64-token tile would run at <25% utilization while the
        // shipped 8-token class still reads every weight row once — the
        // prefill's 3-token tail chunk was paying a full tile pass per GEMM.
        if !matches!(ty, 8 | 12 | 14) || secs.is_empty() || secs.len() > 4 || !Self::gemm_tiled_enabled()
            || b < 16
        {
            return false;
        }
        if !gemm_cols_ok(ty, cols) {
            note_gemm_shape_refusal(ty, cols, "batch_gguf_gemm_tiled_multi");
            return false;
        }
        if std::env::var_os("SOFUU_TILED_DEBUG").is_some() {
            eprintln!("[tiled-dbg] multi ty={ty} rows={rows_total} cols={cols} B={b}");
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_tiled_multi(
                self.ctx,
                ty,
                w_buf.handle, w_base_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows_total, cols, b,
                secs.as_ptr() as *const u32, secs.len(),
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// P4 arm-d: tiled resid-epilogue GEMM (o / down).
    pub fn batch_gguf_gemm_tiled_acc(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        res_buf: &MetalBuffer, res_offset: usize, res_stride: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {
        if !matches!(ty, 8 | 12 | 14) || !Self::gemm_tiled_enabled() || b < 16 {
            return false;
        }
        if !gemm_cols_ok(ty, cols) {
            note_gemm_shape_refusal(ty, cols, "batch_gguf_gemm_tiled_acc");
            return false;
        }
        if std::env::var_os("SOFUU_TILED_DEBUG").is_some() {
            eprintln!("[tiled-dbg] acc ty={ty} rows={rows} cols={cols} B={b}");
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_tiled_acc(
                self.ctx,
                ty,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                res_buf.handle, res_offset, res_stride,
                y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }


    /// Chain-route batching for the NON-q4_K projections (q6_K on the i1
    /// Q4_K_M class, q8_0 on the 8B): the tiled fused GEMM class with the
    /// typed single-tensor section table. Same weight layout as the per-row
    /// gemv_q6_k it replaces and the same f32 dequant, so batching a
    /// projection does not change its arithmetic class - only the summation
    /// order across k-blocks. Deliberately NOT behind SOFUU_GPU_GEMM_TILED:
    /// that switch belongs to the PREFILL_GEMM arm's kernel choice, and the
    /// chain's q6_K projections have no other batched route.
    pub fn batch_gguf_gemm_tiled_multi_chain(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_base_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows_total: usize, cols: usize, b: usize,
        secs: &[[u32; 5]],
    ) -> bool {
        if !matches!(ty, 2 | 8 | 12 | 14) || secs.is_empty() || secs.len() > 4 || b < 2 {
            return false;
        }
        if !gemm_cols_ok(ty, cols) {
            note_gemm_shape_refusal(ty, cols, "batch_gguf_gemm_tiled_multi_chain");
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_tiled_multi(
                self.ctx,
                ty,
                w_buf.handle, w_base_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows_total, cols, b,
                secs.as_ptr() as *const u32, secs.len(),
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// Chain-route resid-epilogue form of the above (ffn_down, and any o
    /// projection that is not q4_K): one dispatch computes y = res + W x.
    /// Every output element is read and written by the same thread, so
    /// passing the residual slot itself as y is not a race.
    pub fn batch_gguf_gemm_tiled_acc_chain(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        res_buf: &MetalBuffer, res_offset: usize, res_stride: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {
        if !matches!(ty, 2 | 8 | 12 | 14) || b < 2 || res_stride != rows {
            return false;
        }
        if !gemm_cols_ok(ty, cols) {
            note_gemm_shape_refusal(ty, cols, "batch_gguf_gemm_tiled_acc_chain");
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_tiled_acc(
                self.ctx,
                ty,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                res_buf.handle, res_offset, res_stride,
                y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        __r
    }

    /// P4 arm-d: threadgroup-tiled q4_K GEMM (32-row x 64-token tile with
    /// threadgroup-memory staging). Returns false when the pipeline is
    /// unavailable or the type is not q4_K, so callers keep the shipped class.
    pub fn batch_gguf_gemm_tiled(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {
        if ty != 12 {
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_tiled(
                self.ctx,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    pub fn batch_gguf_gemm_multi(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_base_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows_total: usize, cols: usize, b: usize,
        secs: &[[u32; 5]], // host-padded to 4 entries
    ) -> bool {
        debug_assert!(secs.len() <= 4);
        if !gemm_cols_ok(ty, cols) {
            note_gemm_shape_refusal(ty, cols, "batch_gguf_gemm_multi");
            return false;
        }
        // The retired 256-value hold on q8_0 lived here. It was a SYMPTOM
        // refusal with the wrong root cause: gemm_q8_0_multi stored its
        // accumulators at yr[k * y_stride] instead of yr[(t0 + k) * y_stride],
        // so the GEMM_TS=8 subtile grouping (not the column count) was wrong:
        // every subtile >= 1 clobbered subtile 0's slots and tokens 8..B-1
        // were NEVER written (stale scratch read back as logits). It is wrong
        // for every B > 8 at EVERY width, 256-divisible or not - which is why
        // the hold "fixed" 960 and never showed at 1024 (cols 1024 was measured
        // on the mma64 twin, which always had the term).
        // Fixed in kernels.metal (q8_0 + the identical q6_K copy) and pinned
        // both ways: gemm_q8_0_multi_matches_dequant_oracle (numeric, 960/992/
        // 1024/2080/3200 x b_tok 9/17 vs a host f64 dequant reference) and
        // gemm_multi_kernels_store_the_subtile_token_offset (source). The only
        // column requirement left is the REAL one from gemm_cols_ok above:
        // whole q8_0 blocks (cols % 32 == 0), already checked at entry.
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_multi(
                self.ctx,
                ty,
                w_buf.handle, w_base_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows_total, cols, b,
                secs.as_ptr() as *const u32, secs.len(),
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// P1 wave 2: resid-add epilogue GEMM (o/down). y[t*rows + row] =
    /// dot(W[row], X[t]) + res[t*res_stride + row] — one writer per element.
    /// Returns false when unavailable (kernel PSO missing or non-q4_K).
    pub fn batch_gguf_gemm_acc(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        res_buf: &MetalBuffer, res_offset: usize, res_stride: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {
        if !gemm_cols_ok(ty, cols) {
            note_gemm_shape_refusal(ty, cols, "batch_gguf_gemm_acc");
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_acc(
                self.ctx,
                ty,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                res_buf.handle, res_offset, res_stride,
                y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// P2 decode diet: multi-section q4_K b4 GEMV — the byte-lane b4c shape
    /// with a section table (no GEMM subtiles; all 4 token rows per
    /// threadgroup). y region offsets are ABSOLUTE slab bytes, strides are
    /// region rows. Returns false when unavailable.
    pub fn batch_gguf_b4multi(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_base_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer,
        rows_total: usize, cols: usize,
        secs: &[[u32; 5]],
    ) -> bool {
        debug_assert!(secs.len() <= 4);
        let __t = self.op_tick(K_GGUF);
        let __r = unsafe {
            sofuu_metal_batch_gguf_b4multi(
                self.ctx,
                ty,
                w_buf.handle, w_base_offset,
                x_buf.handle, x_offset,
                y_buf.handle,
                rows_total, cols,
                secs.as_ptr() as *const u32, secs.len(),
            ) == 1
        };
        self.op_tock(K_GGUF, __t);
        __r
    }

    /// P1 completion: multi-section q6_K b4 GEMV (attn_v / MiniCPM FFN).
    pub fn batch_gguf6_b4multi(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_base_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer,
        rows_total: usize, cols: usize,
        secs: &[[u32; 5]],
    ) -> bool {
        debug_assert!(secs.len() <= 4);
        let __t = self.op_tick(K_GGUF);
        let __r = unsafe {
            sofuu_metal_batch_gguf6_b4multi(
                self.ctx,
                ty,
                w_buf.handle, w_base_offset,
                x_buf.handle, x_offset,
                y_buf.handle,
                rows_total, cols,
                secs.as_ptr() as *const u32, secs.len(),
            ) == 1
        };
        self.op_tock(K_GGUF, __t);
        __r
    }

    /// P2: multi-section q8_0 b4 GEMV (1B fleet; MiniCPM q8_0 tensors).
    pub fn batch_gguf8_b4multi(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_base_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer,
        rows_total: usize, cols: usize,
        secs: &[[u32; 5]],
    ) -> bool {
        debug_assert!(secs.len() <= 4);
        let __t = self.op_tick(K_GGUF);
        let __r = unsafe {
            sofuu_metal_batch_gguf8_b4multi(
                self.ctx,
                ty,
                w_buf.handle, w_base_offset,
                x_buf.handle, x_offset,
                y_buf.handle,
                rows_total, cols,
                secs.as_ptr() as *const u32, secs.len(),
            ) == 1
        };
        self.op_tock(K_GGUF, __t);
        __r
    }

    /// P2 rate wave: q8_0 with 4 weight-rows per threadgroup (x traffic /4).
    pub fn batch_gguf8_b4r4(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_base_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer,
        rows_total: usize, cols: usize,
        secs: &[[u32; 5]],
    ) -> bool {
        debug_assert!(secs.len() <= 4);
        let __t = self.op_tick(K_GGUF);
        let __r = unsafe {
            sofuu_metal_batch_gguf8_b4r4(
                self.ctx,
                ty,
                w_buf.handle, w_base_offset,
                x_buf.handle, x_offset,
                y_buf.handle,
                rows_total, cols,
                secs.as_ptr() as *const u32, secs.len(),
            ) == 1
        };
        self.op_tock(K_GGUF, __t);
        __r
    }

    /// RMSNorm with F32 weights (GGUF norm tensors). The weight buffer is
    /// bound at a 256-byte-aligned offset with the element remainder passed
    /// through the kernel; the C side does that alignment.
    pub fn batch_rmsnorm_f32w(
        &self,
        x_buf: &MetalBuffer, x_offset: usize,
        w_buf: &MetalBuffer, w_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        dim: usize, eps: f32,
    ) -> bool {        let __t = self.op_tick(K_RMSNORM);
        let __r = unsafe { sofuu_metal_batch_rmsnorm_f32w(
                self.ctx,
                x_buf.handle, x_offset,
                w_buf.handle, w_offset,
                y_buf.handle, y_offset,
                dim, eps,
            ) == 1 };
        self.op_tock(K_RMSNORM, __t);
        __r
    }

    /// B-row f32-weight RMSNorm for the spec chain: rows are contiguous
    /// [B][dim]; one dispatch instead of B. Returns false when the pipeline is
    /// missing (a bundled metallib predating it) so the caller can fall back to
    /// the per-row dispatch instead of silently skipping the norm.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_rmsnorm_f32w_b4(
        &self,
        x_buf: &MetalBuffer, x_offset: usize,
        w_buf: &MetalBuffer, w_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        dim: usize, eps: f32, rows: usize,
    ) -> bool {        let __t = self.op_tick(K_RMSNORM_B4);
        let __r = unsafe {
            sofuu_metal_batch_rmsnorm_f32w_b4(
                self.ctx,
                x_buf.handle, x_offset,
                w_buf.handle, w_offset,
                y_buf.handle, y_offset,
                dim, eps, rows,
            ) == 1
        };
        self.op_tock(K_RMSNORM_B4, __t);
        __r
    }

    /// Llama-shape KV append: optional f32 head-norm + RoPE (either mode),
    /// then k(roped)/v(raw) at `pos` of this layer's cache slice.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_kv_write(
        &self,
        k_buf: &MetalBuffer, k_offset: usize,
        v_buf: &MetalBuffer, v_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        kln_buf: Option<&MetalBuffer>, kln_offset: usize,
        has_norm: u32, pos: u32,
        head_dim: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> bool {
        let (kln_handle, kln_off) = match kln_buf {
            Some(b) => (b.handle, kln_offset),
            None => (k_buf.handle, 0),
        };        let __t = self.op_tick(K_KVWRITE);
        let __r = unsafe { sofuu_metal_batch_llama_kv_write(
                self.ctx,
                k_buf.handle, k_offset,
                v_buf.handle, v_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                kln_handle, kln_off,
                has_norm, pos,
                head_dim, n_kv_heads, max_seq,
                eps, rope_base, rope_mode,
            ) == 1 };
        self.op_tock(K_KVWRITE, __t);
        __r
    }

    /// Per-head f32 RMSNorm (q-norm) on a [heads x head_dim] slab slot.
    pub fn batch_head_rmsnorm(
        &self,
        x_buf: &MetalBuffer, x_offset: usize,
        w_buf: Option<&MetalBuffer>, w_offset: usize,
        head_dim: u32, n_heads: u32,
        has_norm: u32, pos: u32, eps: f32, rope_base: f32, rope_mode: u32,
    ) -> bool {
        let (w_handle, w_off) = match w_buf {
            Some(b) => (b.handle, w_offset),
            None => (x_buf.handle, 0),
        };
        let __r = unsafe { sofuu_metal_batch_head_rmsnorm(
                self.ctx,
                x_buf.handle, x_offset,
                w_handle, w_off,
                head_dim, n_heads,
                has_norm, pos, eps, rope_base, rope_mode,
            ) == 1 };
        __r
    }

    /// GPU3 attention stage 1: chunked softmax+V partials into meta.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_attn_part(
        &self,
        q_buf: &MetalBuffer, q_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        qln_buf: Option<&MetalBuffer>, qln_offset: usize,
        meta_buf: &MetalBuffer, meta_offset: usize,
        has_norm: u32, pos: u32,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> bool {
        let (qln_handle, qln_off) = match qln_buf {
            Some(b) => (b.handle, qln_offset),
            None => (q_buf.handle, 0),
        };
        let __r = unsafe { sofuu_metal_batch_llama_attn_part(
                self.ctx,
                q_buf.handle, q_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                qln_handle, qln_off,
                meta_buf.handle, meta_offset,
                has_norm, pos,
                head_dim, n_heads, n_kv_heads, max_seq,
                eps, rope_base, rope_mode,
            ) == 1 };
        __r
    }

    /// GPU3 attention stage 2: rescale chunk partials into head_out.
    pub fn batch_llama_attn_fin(
        &self,
        meta_buf: &MetalBuffer, meta_offset: usize,
        out_buf: &MetalBuffer, out_offset: usize,
        pos_alt: u32, n_heads: u32, head_dim: u32,
    ) -> bool {
        let __r = unsafe { sofuu_metal_batch_llama_attn_fin(
                self.ctx,
                meta_buf.handle, meta_offset,
                out_buf.handle, out_offset,
                pos_alt, n_heads, head_dim,
            ) == 1 };
        __r
    }

    /// Multi-row (B≤4 chain) KV write: positions from a u32 buffer, grid =
    /// rows × kv_heads; rows are contiguous [B][kv_heads*hd].
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_kv_write_b4(
        &self,
        k_buf: &MetalBuffer, k_offset: usize,
        v_buf: &MetalBuffer, v_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        kln_buf: Option<&MetalBuffer>, kln_offset: usize,
        has_norm: u32,
        pos_buf: &MetalBuffer, pos_offset: usize,
        rows: usize,
        head_dim: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
        rope_tab: Option<(&MetalBuffer, usize)>,
    ) {
        let (kln_handle, kln_off) = match kln_buf {
            Some(b) => (b.handle, kln_offset),
            None => (k_buf.handle, 0),
        };
        let (tab_handle, tab_off, has_table) = match rope_tab {
            Some((b, o)) => (b.handle, o, 1u32),
            None => (pos_buf.handle, 0, 0u32),
        };
        let __t = self.op_tick(K_KVWRITE_B4);
        unsafe {
            sofuu_metal_batch_llama_kv_write_b4(
                self.ctx,
                k_buf.handle, k_offset,
                v_buf.handle, v_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                kln_handle, kln_off,
                has_norm,
                pos_buf.handle, pos_offset,
                rows as u32,
                head_dim, n_kv_heads, max_seq,
                eps, rope_base, rope_mode,
                tab_handle, tab_off, has_table,
            );
        }
        self.op_tock(K_KVWRITE_B4, __t);
    }

    /// Multi-row per-head q-norm + RoPE.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_head_rmsnorm_b4(
        &self,
        x_buf: &MetalBuffer, x_offset: usize,
        w_buf: Option<&MetalBuffer>, w_offset: usize,
        pos_buf: &MetalBuffer, pos_offset: usize,
        rows: usize,
        head_dim: u32, n_heads: u32,
        has_norm: u32, eps: f32, rope_base: f32, rope_mode: u32,
        rope_tab: Option<(&MetalBuffer, usize)>,
    ) {
        let (w_handle, w_off) = match w_buf {
            Some(b) => (b.handle, w_offset),
            None => (x_buf.handle, 0),
        };
        let (tab_handle, tab_off, has_table) = match rope_tab {
            Some((b, o)) => (b.handle, o, 1u32),
            None => (x_buf.handle, 0, 0u32),
        };
        let __t = self.op_tick(K_HEAD_RMSNORM);
        unsafe {
            sofuu_metal_batch_head_rmsnorm_b4(
                self.ctx,
                x_buf.handle, x_offset,
                w_handle, w_off,
                pos_buf.handle, pos_offset,
                rows as u32,
                head_dim, n_heads,
                has_norm, eps, rope_base, rope_mode,
                tab_handle, tab_off, has_table,
            );
        }
        self.op_tock(K_HEAD_RMSNORM, __t);
    }

    /// Multi-row chunked attention partials: row r's softmax bound is
    /// positions[r] (chain causality within the batch); meta regions are
    /// meta_stride floats apart per row.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_attn_part_b4(
        &self,
        q_buf: &MetalBuffer, q_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        meta_buf: &MetalBuffer, meta_offset: usize,
        pos_buf: &MetalBuffer, pos_offset: usize,
        rows: usize,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        n_chunks_max: u32, meta_stride: u32,
    ) {        let __t = self.op_tick(K_ATTN_PART);
        unsafe {
            sofuu_metal_batch_llama_attn_part_b4(
                self.ctx,
                q_buf.handle, q_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                meta_buf.handle, meta_offset,
                pos_buf.handle, pos_offset,
                rows as u32,
                head_dim, n_heads, n_kv_heads, max_seq,
                n_chunks_max, meta_stride,
            );
        }
        self.op_tock(K_ATTN_PART, __t);
    }

    /// Multi-row finish rescale (pairs with batch_llama_attn_part_b4).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_attn_fin_b4(
        &self,
        meta_buf: &MetalBuffer, meta_offset: usize,
        out_buf: &MetalBuffer, out_offset: usize,
        pos_buf: &MetalBuffer, pos_offset: usize,
        rows: usize,
        n_heads: u32, head_dim: u32,
        n_chunks_max: u32, meta_stride: u32,
    ) {        let __t = self.op_tick(K_ATTN_FIN);
        unsafe {
            sofuu_metal_batch_llama_attn_fin_b4(
                self.ctx,
                meta_buf.handle, meta_offset,
                out_buf.handle, out_offset,
                pos_buf.handle, pos_offset,
                rows as u32,
                n_heads, head_dim,
                n_chunks_max, meta_stride,
            );
        }
        self.op_tock(K_ATTN_FIN, __t);
    }

    /// FUSED 1-chunk attention (pos < 128): part+fin in ONE dispatch, no
    /// meta buffer, no barrier between the two phases. Returns false when
    /// the PSO is unavailable (caller falls back to part+fin).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_attn_1c_b4(
        &self,
        q_buf: &MetalBuffer, q_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        out_buf: &MetalBuffer, out_offset: usize,
        pos_buf: &MetalBuffer, pos_offset: usize,
        rows: usize,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        rope_tab_inert: Option<(&MetalBuffer, usize)>,
    ) -> bool {
        let __t = self.op_tick(K_ATTN_1C);
        let __r = unsafe {
            sofuu_metal_batch_llama_attn_1c_b4(
                self.ctx,
                q_buf.handle, q_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                out_buf.handle, out_offset,
                pos_buf.handle, pos_offset,
                rows as u32,
                head_dim, n_heads, n_kv_heads, max_seq,
                q_buf.handle, 0, 0u32,
            ) == 1
        };
        self.op_tock(K_ATTN_1C, __t);
        __r
    }

    /// Peak probes: fp32 FMA pipe vs simdgroup_matrix pipe, no memory traffic.
    pub fn peak_fma(&self, out_buf: &MetalBuffer, n_blocks: usize) -> bool {
        unsafe { sofuu_metal_peak_fma(self.ctx, out_buf.handle, n_blocks) == 1 }
    }
    pub fn peak_fma16(&self, out_buf: &MetalBuffer, n_blocks: usize) -> bool {
        unsafe { sofuu_metal_peak_fma16(self.ctx, out_buf.handle, n_blocks) == 1 }
    }
    pub fn peak_mma(&self, out_buf: &MetalBuffer, n_blocks: usize) -> bool {
        unsafe { sofuu_metal_peak_mma(self.ctx, out_buf.handle, n_blocks) == 1 }
    }

    /// LEDGER #7: section-fused matrix-pipe GEMM (one dispatch per section,
    /// all inside the caller's commit). q4_K only for now.
    /// Matrix-pipe GEMM that adds into the residual (o and down projections).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_gguf_gemm_mma64_acc(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        res_buf: &MetalBuffer, res_offset: usize, res_stride: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {
        if !matches!(ty, 8 | 12 | 14) || !Self::gemm_mma_enabled() || b < 16 {
            return false;
        }
        if !gemm_cols_ok(ty, cols) {
            note_gemm_shape_refusal(ty, cols, "batch_gguf_gemm_mma64_acc");
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_mma64_acc(
                self.ctx, ty,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                res_buf.handle, res_offset, res_stride,
                y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_gguf_gemm_mma64_multi(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_base_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows_total: usize, cols: usize, b: usize,
        secs: &[[u32; 5]; 4],
    ) -> bool {
        let secs: &[[u32; 5]] = &secs[..];
        if !matches!(ty, 8 | 12 | 14) || !Self::gemm_mma_enabled() || b < 16 || secs.is_empty() {
            return false;
        }
        if !gemm_cols_ok(ty, cols) {
            note_gemm_shape_refusal(ty, cols, "batch_gguf_gemm_mma64_multi");
            return false;
        }
        let n_sec = secs.len();
        if n_sec == 0 || n_sec > 4 {
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_mma64_multi(
                self.ctx, ty,
                w_buf.handle, w_base_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows_total, cols, b,
                secs.as_ptr() as *const u32, n_sec,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// LEDGER #7: the matrix-pipe q4_K GEMM (fp16 tiles, fp32 accumulate).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_gguf_gemm_mma64(
        &self,
        w_buf: &MetalBuffer, x_buf: &MetalBuffer, y_buf: &MetalBuffer,
        y_offset: usize, rows: usize, cols: usize, b: usize,
    ) -> bool {
        if !Self::gemm_tiled_enabled() || b < 16 || cols % 256 != 0 {
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_mma64(
                self.ctx, w_buf.handle, 0, x_buf.handle, 0, y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// item 2b: 8x8 register tile (threadgroup-bandwidth fix).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_gguf_gemm_tiled_r88(
        &self,
        w_buf: &MetalBuffer, x_buf: &MetalBuffer, y_buf: &MetalBuffer,
        y_offset: usize, rows: usize, cols: usize, b: usize,
    ) -> bool {
        if !Self::gemm_tiled_enabled() || b < 16 || cols % 256 != 0 {
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_tiled_r88(
                self.ctx, w_buf.handle, 0, x_buf.handle, 0, y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }



    /// item 0: double-buffered k-slice staging, same ABI/arithmetic as
    /// `batch_gguf_gemm_tiled`. Prefetches slice kb+1 while computing kb, so

    /// arm-E: 64-row tiled q4_K GEMM — same ABI and output layout as
    /// `batch_gguf_gemm_tiled`, but x is staged once per 64-row block
    /// (the ffn gate/up shape is x-traffic bound at 32 rows).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_gguf_gemm_tiled_wide(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {
        if ty != 12 || !Self::gemm_tiled_enabled() || b < 16 {
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_tiled_wide(
                self.ctx,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// arm-E: 64-row fused tiled q4_K GEMM (section table). Enabled only for
    /// the shapes that measured a win: rows >= 4096 (the ffn gate/up side,
    /// where the activation tile is re-staged once per row block).
    #[allow(clippy::too_many_arguments)]
    fn gemm_wide_enabled() -> bool {
        use std::sync::atomic::{AtomicU8, Ordering};
        static WIDE: AtomicU8 = AtomicU8::new(0);
        match WIDE.load(Ordering::Relaxed) {
            0 => {
                // OFF by default: the 64-row tile wins 1.32x on the single
                // dominant GEMM shape but that slice is too small to resolve
                // end to end (0.6B: 560 vs 564 ms under load 5.0). Flip only
                // after an idle-machine A/B shows a win on both 0.6B and 8B.
                let on = matches!(
                    std::env::var("SOFUU_GPU_GEMM_WIDE").as_deref(),
                    Ok("1") | Ok("on") | Ok("true")
                );
                WIDE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
                on
            }
            1 => true,
            _ => false,
        }
    }

    pub fn batch_gguf_gemm_tiled_wide_multi(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows_total: usize, cols: usize, b: usize,
        secs: &[[u32; 5]],
    ) -> bool {
        if ty != 12
            || !Self::gemm_tiled_enabled()
            || !Self::gemm_wide_enabled()
            || b < 16
            || rows_total < 4096
            || secs.is_empty()
            || secs.len() > 4
        {
            return false;
        }
        let mut flat = [0u32; 20];
        for (i, s) in secs.iter().enumerate() {
            flat[i * 5..i * 5 + 5].copy_from_slice(s);
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_tiled_wide_multi(
                self.ctx,
                ty,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows_total, cols, b,
                flat.as_ptr(), secs.len(),
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// P5 arm-c: simdgroup-matrix q4_K GEMM (fp16 MMA units, 64x64 tile).
    /// Same ABI and output layout as `batch_gguf_gemm_tiled`. Opt-in behind
    /// SOFUU_GPU_GEMM_MMA; false keeps the SIMT tiled class.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_gguf_gemm_mma(
        &self,
        ty: u32,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        rows: usize, cols: usize, b: usize,
    ) -> bool {
        if ty != 12 || !Self::gemm_mma_enabled() || b < 2 {
            return false;
        }
        let __t = self.op_tick(K_GGUF_GEMM);
        let __r = unsafe {
            sofuu_metal_batch_gguf_gemm_mma(
                self.ctx,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows, cols, b,
            ) == 1
        };
        self.op_tock(K_GGUF_GEMM, __t);
        __r
    }

    /// Signature histogram for the single-token GEMV entry, so a per-token
    /// dispatch inside a batched pass can be identified by shape.
    fn gguf_histogram() -> &'static std::sync::Mutex<std::collections::HashMap<(u32, usize, usize), u64>> {
        static H: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(u32, usize, usize), u64>>> =
            std::sync::OnceLock::new();
        H.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
    }

    /// Dumps (and clears) the batch_gguf signature histogram.
    pub fn print_gguf_histogram(&self, label: &str) {
        if std::env::var_os("SOFUU_GGUF_DBG").is_none() {
            return;
        }
        let mut h = match Self::gguf_histogram().lock() {
            Ok(h) => h,
            Err(_) => return,
        };
        let mut rows: Vec<_> = h.drain().collect();
        rows.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        for ((ty, rows, cols), c) in rows {
            eprintln!("[gguf-hist {label}] ty={ty} rows={rows} cols={cols} calls={c}");
        }
    }

    /// Fused residual-add + RMSNorm. SOFUU_FUSE_NORM=1 arms it. Measured neutral
    /// end to end (90.1 vs 90.3 tg/s), so it stays off.
    fn fuse_norm_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("SOFUU_FUSE_NORM").map(|v| v == "1").unwrap_or(false))
    }

    /// Flat one-dispatch q4_K GEMV. SOFUU_GEMV_FLAT=1 arms it. It is 3x faster than the
    /// shipped kernel in isolation (149 vs 49 GB/s on the same bytes) and 2% slower
    /// per token, so it stays off until something upstream changes.
    /// LEDGER #192: the flat single-thread-per-(row,block) sweep measured 6->30+ GB/s
    /// effective for decode (3/3 interleaved wins, +35% median) but 7-13% WORSE for
    /// prefill, so it is armed only around a decode token, never globally.
    fn flat_flag() -> &'static std::sync::atomic::AtomicBool {
        static F: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        &F
    }

    fn gemv_flat_forced() -> bool {
        Self::flat_flag().load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_gemv_flat_forced(on: bool) {
        Self::flat_flag().store(on, std::sync::atomic::Ordering::Relaxed);
    }

    fn gemv_flat_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("SOFUU_GEMV_FLAT").map(|v| v == "1").unwrap_or(false))
    }

    /// Warp-cooperative decode attention. SOFUU_ATTN_FAST=0 turns it off.
    /// Warps per head for decode attention (4/8/16, default 16). Measured in one window:
    /// serial 10177 us/token, w4 7988, w8 7659, w16 7238 - it keeps paying until the
    /// per-warp position count gets too small to fill a warp.
    pub fn attn_warps() -> u32 {
        static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        *N.get_or_init(|| std::env::var("SOFUU_ATTN_WARPS").ok().and_then(|v| v.parse().ok()).unwrap_or(16))
    }

    /// Position at which the chunked attention path takes over from the warp kernel.
    /// SOFUU_ATTN_CHUNK_MIN overrides it. The warp kernel covers everything up to the
    /// 4096 it can stage, and at a 320-token context it is 106.8 vs 67.5 tg/s against
    /// the chunked path, so chunked is now only the >4096 fallback.
    pub fn attn_chunk_min() -> u32 {
        static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        // Clamped to the warp kernel's staging limit: above it that kernel declines
        // and the batch would fail instead of falling back to the chunked path.
        *N.get_or_init(|| {
            std::env::var("SOFUU_ATTN_CHUNK_MIN")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(4096)
                .min(4096)
        })
    }

    pub fn attn_w4_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("SOFUU_ATTN_W4").map(|v| v != "0").unwrap_or(true))
    }

    pub fn attn_fast_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("SOFUU_ATTN_FAST").map(|v| v != "0").unwrap_or(true))
    }

    /// Split-K q4_K GEMV (flat pass 1 + reduce). SOFUU_GEMV_SPLIT=1 arms it.
    fn gemv_split_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("SOFUU_GEMV_SPLIT").map(|v| v == "1").unwrap_or(false))
    }

    /// 8 rows per threadgroup q4_K GEMV. SOFUU_GEMV_R8=1 arms it.
    fn gemv_r8_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("SOFUU_GEMV_R8").map(|v| v == "1").unwrap_or(false))
    }

    /// Decode GEMV v4 (x staged in threadgroup memory). SOFUU_GEMV_V4=1 arms it.
    fn gemv_v4_enabled() -> bool {
        use std::sync::atomic::{AtomicU8, Ordering};
        static STATE: AtomicU8 = AtomicU8::new(0);
        match STATE.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => {
                let on = std::env::var("SOFUU_GEMV_V4")
                    .map(|v| {
                        let v = v.trim().to_ascii_lowercase();
                        v == "1" || v == "true" || v == "on"
                    })
                    .unwrap_or(false);
                STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
                on
            }
        }
    }

    /// Decode GEMV v3 (split-K: 4 warps per row). SOFUU_GEMV_V3=1 arms it.
    fn gemv_v3_enabled() -> bool {
        use std::sync::atomic::{AtomicU8, Ordering};
        static STATE: AtomicU8 = AtomicU8::new(0);
        match STATE.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => {
                let on = std::env::var("SOFUU_GEMV_V3")
                    .map(|v| {
                        let v = v.trim().to_ascii_lowercase();
                        v == "1" || v == "true" || v == "on"
                    })
                    .unwrap_or(false);
                STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
                on
            }
        }
    }

    /// Decode GEMV v2 (8 rows per threadgroup, vectorized qs/x loads).
    /// SOFUU_GEMV_V2=0 opts out; default off until its e2e receipt lands.
    fn gemv_v2_enabled() -> bool {
        use std::sync::atomic::{AtomicU8, Ordering};
        static STATE: AtomicU8 = AtomicU8::new(0);
        match STATE.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => {
                let on = std::env::var("SOFUU_GEMV_V2")
                    .map(|v| {
                        let v = v.trim().to_ascii_lowercase();
                        v == "1" || v == "true" || v == "on"
                    })
                    .unwrap_or(false);
                STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
                on
            }
        }
    }

    /// Cached env gate for the MMA GEMM class (default off).
    pub fn gemm_mma_enabled() -> bool {
        use std::sync::atomic::{AtomicU8, Ordering};
        static STATE: AtomicU8 = AtomicU8::new(0);
        match STATE.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => {
                // Default ON: the matrix pipe is 2.1x the fp32 tile on the
                // kernel and 1.14x on the 0.6B pass, and its text gate is
                // NEAR-TIE-CERTIFIED at |dtop1| 0.3150 against bound 1.112 -
                // the same drift the fp32 path shows, so the arithmetic class
                // change buys speed without moving quality. SOFUU_GPU_GEMM_MMA=0
                // opts out.
                let on = std::env::var("SOFUU_GPU_GEMM_MMA")
                    .map(|v| {
                        let v = v.trim().to_ascii_lowercase();
                        !(v == "0" || v == "false" || v == "off")
                    })
                    .unwrap_or(true);
                STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
                on
            }
        }
    }

    /// Whole-row chunk attention (chain 128-row pass). ONE dispatch computes
    /// every row's full softmax and writes [row][head][head_dim] straight into
    /// out_buf — no meta scratch, no copy finish, no per-4-rows grid.
    ///
    /// SOFUU_CHAIN_ATTN picks the arm: "row" (default, kernels.metal
    /// llama_attn_pre_row — 32 rows per 1024-thread threadgroup, one warp per
    /// row, so the group shares one K/V stream), "w32" (the shipped
    /// llama_attn_pre_w32, one threadgroup per (row, head)) and "b4" (returns
    /// false so the caller keeps the flash/1c/part-fin family). Returns false
    /// with no dispatch at all when the arm cannot run, so the caller falls back
    /// instead of leaving attention undone.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_attn_row(
        &self,
        q_buf: &MetalBuffer, q_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        out_buf: &MetalBuffer, out_offset: usize,
        pos_buf: &MetalBuffer, pos_offset: usize,
        rows: usize,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
    ) -> bool {
        use std::sync::atomic::{AtomicU8, Ordering};
        static ARM: AtomicU8 = AtomicU8::new(0);
        let mut arm = ARM.load(Ordering::Relaxed);
        if arm == 0 {
            arm = match std::env::var("SOFUU_CHAIN_ATTN") {
                Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                    "b4" | "0" | "off" => 3u8,
                    "w32" | "prew" => 2u8,
                    _ => 1u8,
                },
                Err(_) => 1u8,
            };
            ARM.store(arm, Ordering::Relaxed);
        }
        if rows == 0 || arm == 3 {
            return false;
        }
        let out_stride = (n_heads as usize * head_dim as usize) as u32;
        let __t = self.op_tick(K_ATTN_1C);
        let __r = unsafe {
            let r = if arm == 2 {
                sofuu_metal_batch_llama_attn_w32_out(
                    self.ctx,
                    q_buf.handle, q_offset,
                    kc_buf.handle, kc_layer_offset,
                    vc_buf.handle, vc_layer_offset,
                    out_buf.handle, out_offset,
                    pos_buf.handle, pos_offset,
                    rows as u32,
                    head_dim, n_heads, n_kv_heads, max_seq,
                    out_stride,
                )
            } else {
                sofuu_metal_batch_llama_attn_row(
                    self.ctx,
                    q_buf.handle, q_offset,
                    kc_buf.handle, kc_layer_offset,
                    vc_buf.handle, vc_layer_offset,
                    out_buf.handle, out_offset,
                    pos_buf.handle, pos_offset,
                    rows as u32,
                    head_dim, n_heads, n_kv_heads, max_seq,
                    out_stride,
                )
            };
            r == 1
        };
        self.op_tock(K_ATTN_1C, __t);
        // One-shot coverage trace (SOFUU_CHAIN_TRACE=1): the GQA form the
        // picker chose for THIS model's ratio. -1 means the table has no form
        // for the ratio and the per-head arm owns it — qwen2/Qwen2.5's 14:2
        // (7:1) is exactly such a ratio, so a Qwen2.5 run prints -1 and pays
        // the per-head k/v re-read; -2 means forms exist but none is live on
        // this GPU (a real configuration loss, not a table gap).
        if std::env::var_os("SOFUU_CHAIN_TRACE").is_some() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static TRACED: AtomicBool = AtomicBool::new(false);
            if !TRACED.swap(true, Ordering::Relaxed) {
                let ratio = if n_kv_heads > 0 { n_heads / n_kv_heads } else { 0 };
                let idx = unsafe { sofuu_metal_attn_gqa_last() };
                let why = match idx {
                    -1 => " (no form for this ratio: per-head arm)",
                    -2 => " (forms exist, none live here: per-head arm)",
                    _ => " (paired k/v arm)",
                };
                eprintln!(
                    "[chain] attn: {n_heads}q/{n_kv_heads}kv heads = ratio {ratio}, hd {head_dim} -> table form {idx}{why}",
                );
            }
        }
        __r
    }

    /// W2 feature split: `batch_llama_attn_row` with an EXPLICIT output row
    /// stride. The attention head split gives each engine a compact q block
    /// ([rows][n_heads*hd]) but both must land their rows in the SAME
    /// canonical [rows][q_rows] head_out, at their own head offset — the
    /// kernel already takes the stride, only the shipped wrapper derives it
    /// from n_heads.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_attn_row_os(
        &self,
        q_buf: &MetalBuffer, q_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        out_buf: &MetalBuffer, out_offset: usize,
        pos_buf: &MetalBuffer, pos_offset: usize,
        rows: usize,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32, out_stride: u32,
    ) -> bool {
        if rows == 0 {
            return false;
        }
        let __t = self.op_tick(K_ATTN_1C);
        let __r = unsafe {
            sofuu_metal_batch_llama_attn_row(
                self.ctx,
                q_buf.handle, q_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                out_buf.handle, out_offset,
                pos_buf.handle, pos_offset,
                rows as u32,
                head_dim, n_heads, n_kv_heads, max_seq,
                out_stride,
            ) == 1
        };
        self.op_tock(K_ATTN_1C, __t);
        __r
    }

    /// P5 arm-b: flash-style batched prefill attention. Same math and ABI as
    /// `batch_llama_attn_1c_b4` (plus the row count), but each score is
    /// computed once, lanes stride the head_dim (coalesced K/V), a threadgroup
    /// covers 4 rows (shared K/V), and the softmax runs online in 64-position
    /// blocks so ONE dispatch covers any context length.
    /// Env-gated by SOFUU_GPU_ATTN_FLASH; false keeps the existing route.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_attn_flash_b4(
        &self,
        q_buf: &MetalBuffer, q_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        out_buf: &MetalBuffer, out_offset: usize,
        pos_buf: &MetalBuffer, pos_offset: usize,
        rows: usize,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
    ) -> bool {
        if !Self::attn_flash_enabled() || rows < 2 {
            return false;
        }
        let __t = self.op_tick(K_ATTN_1C);
        let __r = unsafe {
            sofuu_metal_batch_llama_attn_flash_b4(
                self.ctx,
                q_buf.handle, q_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                out_buf.handle, out_offset,
                pos_buf.handle, pos_offset,
                rows as u32,
                head_dim, n_heads, n_kv_heads, max_seq,
                0u32,
            ) == 1
        };
        self.op_tock(K_ATTN_1C, __t);
        __r
    }

    /// Cached env gate for the flash prefill attention (default off).
    pub fn attn_flash_enabled() -> bool {
        use std::sync::atomic::{AtomicU8, Ordering};
        static STATE: AtomicU8 = AtomicU8::new(0);
        match STATE.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => {
                let on = std::env::var("SOFUU_GPU_ATTN_FLASH")
                    .map(|v| {
                        let v = v.trim().to_ascii_lowercase();
                        !(v.is_empty() || v == "0" || v == "false" || v == "off")
                    })
                    .unwrap_or(false);
                STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
                on
            }
        }
    }

    /// B4 q4_K GEMV with the resid-add epilogue (o/down: out width ==
    /// residual width). Returns false when unavailable; the caller falls
    /// back to batch_gguf_b4 + batch_add.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_gguf_b4r(
        &self,
        w_buf: &MetalBuffer, w_offset: usize,
        x_buf: &MetalBuffer, x_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        acc_buf: &MetalBuffer, acc_offset: usize,
        rows: usize, cols: usize, acc_dim: usize,
    ) -> bool {
        unsafe {
            sofuu_metal_batch_gguf_b4r(
                self.ctx,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                acc_buf.handle, acc_offset,
                rows, cols, acc_dim,
            ) == 1
        }
    }

    /// Llama-shape GQA decode attention: optional f32 q-norm + RoPE, softmax
    /// over the full cache prefix, weighted V. kln/qln buffers are only read
    /// when has_norm is set (a None binding is a placeholder there).
    #[allow(clippy::too_many_arguments)]
    /// Decode attention, 4 warps per head. SOFUU_ATTN_W4=0 opts out. 17% faster per
    /// token than the one-warp serial kernel (8940 -> 7456 us median, 4/4 reps).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_attn_w16(
        &self,
        q: &MetalBuffer, q_off: usize,
        kc: &MetalBuffer, kc_off: usize,
        vc: &MetalBuffer, vc_off: usize,
        qln: Option<&MetalBuffer>, qln_off: usize,
        out: &MetalBuffer, out_off: usize,
        has_norm: bool, pos: u32,
        head_dim: u32, n_heads: u32, n_kv: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> bool {
        let qln_handle = qln.map(|b| b.handle).unwrap_or(std::ptr::null_mut());
        unsafe {
            sofuu_metal_batch_llama_attn_w16(
                self.ctx, q.handle, q_off, kc.handle, kc_off, vc.handle, vc_off,
                qln_handle, qln_off, out.handle, out_off,
                has_norm as u32, pos, head_dim, n_heads, n_kv, max_seq,
                eps, rope_base, rope_mode,
            ) == 1
        }
    }

    pub fn batch_llama_attn_w8(
        &self,
        q: &MetalBuffer, q_off: usize,
        kc: &MetalBuffer, kc_off: usize,
        vc: &MetalBuffer, vc_off: usize,
        qln: Option<&MetalBuffer>, qln_off: usize,
        out: &MetalBuffer, out_off: usize,
        has_norm: bool, pos: u32,
        head_dim: u32, n_heads: u32, n_kv: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> bool {
        let qln_handle = qln.map(|b| b.handle).unwrap_or(std::ptr::null_mut());
        unsafe {
            sofuu_metal_batch_llama_attn_w8(
                self.ctx, q.handle, q_off, kc.handle, kc_off, vc.handle, vc_off,
                qln_handle, qln_off, out.handle, out_off,
                has_norm as u32, pos, head_dim, n_heads, n_kv, max_seq,
                eps, rope_base, rope_mode,
            ) == 1
        }
    }

    pub fn batch_llama_attn_w4(
        &self,
        q: &MetalBuffer, q_off: usize,
        kc: &MetalBuffer, kc_off: usize,
        vc: &MetalBuffer, vc_off: usize,
        qln: Option<&MetalBuffer>, qln_off: usize,
        out: &MetalBuffer, out_off: usize,
        has_norm: bool, pos: u32,
        head_dim: u32, n_heads: u32, n_kv: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> bool {
        let qln_handle = qln.map(|b| b.handle).unwrap_or(std::ptr::null_mut());
        unsafe {
            sofuu_metal_batch_llama_attn_w4(
                self.ctx, q.handle, q_off, kc.handle, kc_off, vc.handle, vc_off,
                qln_handle, qln_off, out.handle, out_off,
                has_norm as u32, pos, head_dim, n_heads, n_kv, max_seq,
                eps, rope_base, rope_mode,
            ) == 1
        }
    }

    /// Decode attention via the warp-cooperative kernel. SOFUU_ATTN_FAST=0 opts out.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_llama_attn_fast(
        &self,
        q: &MetalBuffer, q_off: usize,
        kc: &MetalBuffer, kc_off: usize,
        vc: &MetalBuffer, vc_off: usize,
        qln: Option<&MetalBuffer>, qln_off: usize,
        out: &MetalBuffer, out_off: usize,
        has_norm: bool, pos: u32,
        head_dim: u32, n_heads: u32, n_kv: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) -> bool {
        let qln_handle = qln.map(|b| b.handle).unwrap_or(std::ptr::null_mut());
        unsafe {
            sofuu_metal_batch_llama_attn_fast(
                self.ctx, q.handle, q_off, kc.handle, kc_off, vc.handle, vc_off,
                qln_handle, qln_off, out.handle, out_off,
                has_norm as u32, pos, head_dim, n_heads, n_kv, max_seq,
                eps, rope_base, rope_mode,
            ) == 1
        }
    }

    pub fn batch_llama_attn(
        &self,
        q_buf: &MetalBuffer, q_offset: usize,
        kc_buf: &MetalBuffer, kc_layer_offset: usize,
        vc_buf: &MetalBuffer, vc_layer_offset: usize,
        qln_buf: Option<&MetalBuffer>, qln_offset: usize,
        out_buf: &MetalBuffer, out_offset: usize,
        has_norm: u32, pos: u32,
        head_dim: u32, n_heads: u32, n_kv_heads: u32, max_seq: u32,
        eps: f32, rope_base: f32, rope_mode: u32,
    ) {
        let (qln_handle, qln_off) = match qln_buf {
            Some(b) => (b.handle, qln_offset),
            None => (q_buf.handle, 0),
        };        let __t = self.op_tick(K_ATTN);
        unsafe {
            sofuu_metal_batch_llama_attn(
                self.ctx,
                q_buf.handle, q_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                qln_handle, qln_off,
                out_buf.handle, out_offset,
                has_norm, pos,
                head_dim, n_heads, n_kv_heads, max_seq,
                eps, rope_base, rope_mode,
            );
        }
        self.op_tock(K_ATTN, __t);
    }
}


#[cfg(all(test, target_os = "macos"))]
mod batch_gguf_parity_tests {
    use super::*;

    struct Arena {
        ptr: *mut u8,
        len: usize,
    }

    impl Arena {
        fn new(bytes: usize) -> Self {
            let page = 16384usize;
            let len = ((bytes + page - 1) / page) * page;
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED, "arena mmap");
            Self { ptr: p as *mut u8, len }
        }
        fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
        fn f32_view(&self) -> &[f32] {
            unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.len / 4) }
        }
    }

    /// G2 acceptance: the GGUF dequant-dot kernels' VALUES are integer-exact
    /// vs the CPU dequant; only the lane-parallel summation order differs
    /// (same tolerance class as the landed mlx4 kernels).
    #[test]
    fn batch_gguf_gemv_matches_cpu_vecdot() {
        let engine = MetalEngine::new().expect("metal engine");
        let rows = 4usize;
        let cols = 1024usize;

        let mut state = 0x5DEECE66Du64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            // f32 -> f16 with round-to-nearest on the common range; exact
            // values used here (small powers/multiples) mostly round-trip.
            let v = v.clamp(-65504.0, 65504.0);
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let mut exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
            let mut man = (b >> 13) & 0x3FF;
            if exp <= 0 {
                exp = 1;
            }
            if exp > 30 {
                exp = 30;
                man = 0x3FF;
            }
            let raw = sign | ((exp as u16) << 10) | man as u16;
            raw.to_le_bytes()
        };

        let x: Vec<f32> = (0..cols).map(|i| ((i % 17) as f32) * 0.25f32 - 2.0f32).collect();

        type BuildW = Box<dyn Fn(&mut [u8], usize)>;
        let mut y_cpu = vec![0f32; rows];

        let cases: Vec<(u32, &str, usize)> = vec![
            (8, "q8_0", 4),
            (2, "q4_0", 4),
            (12, "q4_K", 4),
            (14, "q6_K", 4),
        ];
        for (ty, name, row_bytes_div) in cases {
            let (row_bytes, nb) = match ty {
                8 => (cols / 32 * 34, cols / 32),
                2 => (cols / 32 * 18, cols / 32),
                12 => (cols / 256 * 144, cols / 256),
                _ => (cols / 256 * 210, cols / 256),
            };
            let _ = row_bytes_div;
            let mut wbytes = vec![0u8; rows * row_bytes];
            let w16 = f16_bytes(0.03125f32 * ((rand() % 64) as f32 + 1.0).max(1.0));
            for r in 0..rows {
                let base = r * row_bytes;
                match ty {
                    8 => {
                        for b in 0..nb {
                            let off = base + b * 34;
                            wbytes[off..off + 2].copy_from_slice(&f16_bytes(0.03125f32 * ((rand() % 32) as f32 + 1.0)));
                            for j in 0..32 {
                                wbytes[off + 2 + j] = (rand() % 255) as u8;
                            }
                        }
                    }
                    2 => {
                        for b in 0..nb {
                            let off = base + b * 18;
                            wbytes[off..off + 2].copy_from_slice(&f16_bytes(0.03125f32 * ((rand() % 32) as f32 + 1.0)));
                            for j in 0..16 {
                                wbytes[off + 2 + j] = (rand() % 255) as u8;
                            }
                        }
                    }
                    12 => {
                        for b in 0..nb {
                            let off = base + b * 144;
                            wbytes[off..off + 2].copy_from_slice(&f16_bytes(0.0625f32));
                            wbytes[off + 2..off + 4].copy_from_slice(&f16_bytes(0.0005f32));
                            for j in 0..12 {
                                wbytes[off + 4 + j] = (rand() & 0x3F) as u8 | ((rand() & 3) << 6) as u8;
                            }
                            for j in 0..128 {
                                wbytes[off + 16 + j] = (rand() % 255) as u8;
                            }
                        }
                    }
                    _ => {
                        for b in 0..nb {
                            let off = base + b * 210;
                            for j in 0..128 {
                                wbytes[off + j] = (rand() % 255) as u8;
                            }
                            for j in 0..64 {
                                wbytes[off + 128 + j] = (rand() % 255) as u8;
                            }
                            for j in 0..16 {
                                wbytes[off + 192 + j] = (rand() % 255) as u8;
                            }
                            wbytes[off + 208..off + 210]
                                .copy_from_slice(&f16_bytes(0.0625f32));
                        }
                    }
                }
            }
            let _ = w16;

            // CPU reference: per-row dot over the same bytes.
            for r in 0..rows {
                let wb = &wbytes[r * row_bytes..(r + 1) * row_bytes];
                y_cpu[r] = match ty {
                    8 => crate::infer::quant::vec_dot_q8_0(wb, &x),
                    2 => crate::infer::quant::vec_dot_q4_0(wb, &x),
                    12 => unsafe {
                        crate::infer::quant::vec_dot_raw_q4_k(wb.as_ptr(), x.as_ptr(), nb)
                    },
                    _ => unsafe {
                        crate::infer::quant::vec_dot_raw_q6_k(wb.as_ptr(), x.as_ptr(), nb)
                    },
                };
            }

            // GPU: one batch op over all rows; x and y in one page-aligned
            // slab (x at 0, y after a 16-byte-aligned gap).
            let arena = Arena::new(cols * 4 + rows * 4 + 256);
            unsafe {
                std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, arena.ptr, cols * 4);
            }
            let w_buf = engine
                .register_zerocopy_buffer(wbytes.as_slice())
                .expect("w buf");
            let slab_buf = engine
                .register_zerocopy_buffer(arena.as_slice())
                .expect("slab buf");
            let y_off = (cols * 4 + 63) & !63;
            assert!(engine.batch_open(), "batch open");
            engine.batch_gguf(ty, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols);
            engine.batch_end();

            let y_gpu = &arena.f32_view()[y_off / 4..y_off / 4 + rows];
            let mut max_rel = 0f64;
            for r in 0..rows {
                let denom = y_cpu[r].abs().max(1e-3);
                let rel = ((y_gpu[r] - y_cpu[r]) as f64 / denom as f64).abs();
                max_rel = max_rel.max(rel);
            }
            eprintln!(
                "[batch_gguf_parity] {name}: max_rel={max_rel:.3e} cpu[0]={:.4} gpu[0]={:.4}",
                y_cpu[0], y_gpu[0]
            );
            assert!(
                max_rel < 1e-2,
                "{name}: GPU vs CPU divergence {max_rel} (cpu[0]={} gpu[0]={})",
                y_cpu[0],
                y_gpu[0]
            );
        }
    }

    /// JOB A oracle: gemm_q8_0_multi against a plain f64 dequant reference
    /// computed ON THE HOST from the same quantised bytes.
    ///
    /// It runs the multi dispatch DIRECTLY (below the model route's shape
    /// guards) at the widths the fleet uses plus the remainders that break
    /// 256, at TWO batch shapes, and it is the permanent regression gate for
    /// this kernel. It earned its keep immediately: the first defect it found
    /// was not the weight cursor everybody suspected but the STORE index -
    /// `yr[k * sec_ystr]` with the GEMM_TS subtile offset `t0` missing, so
    /// every subtile > 0 clobbered subtile 0 and tokens 8..B-1 were never
    /// written at all. A B <= 8 oracle cannot see that, hence b_tok 9 AND 17.
    ///
    /// Data design: the quant bytes are f16 scales of `k/32` and the x values
    /// are multiples of 1/4, so every dot product is an exact dyadic multiple
    /// of 1/128 below 2^24/128 - the GPU's f32 accumulate and the host f64
    /// reference agree BITWISE. That is why a healthy run reports 0.000e0 and
    /// why the bound below is a formality: any non-zero error is structural.
    /// (The 1e-4 relative bound is kept so the test stays honest if the data
    /// pattern is ever changed to a non-dyadic one.)
    ///
    /// Failure modes are separated instead of lumped: an element still
    /// holding the poison sentinel means the kernel never wrote it (dropped
    /// subtile / dropped dispatch), a wrong value with the right magnitude
    /// means a cursor problem. SOFUU_ORACLE_DIAG=1 inverts a bad cell across
    /// ALL (section, token, row) candidates - not just the same token - which
    /// is what names a store-index defect, and prints a per-token census of
    /// never-written cells. SOFUU_ORACLE_DUMP=<dir> writes gpu/ref/x/w bytes
    /// so a failure can be re-analysed offline without re-running the GPU.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn gemm_q8_0_multi_matches_dequant_oracle() {
        let engine = match MetalEngine::new() {
            Some(e) => e,
            None => {
                eprintln!("[oracle] no Metal device - skipped");
                return;
            }
        };
        let diag = std::env::var_os("SOFUU_ORACLE_DIAG").is_some();

        // 9 = one full subtile + a 1-token tail; 17 = two full subtiles + a
        // 1-token tail, i.e. the shape that exposed the missing subtile token
        // offset (a single-subtile B <= 8 dispatch is correct either way, so a
        // b_tok <= 8 oracle would have blessed the broken kernel forever).
        let b_toks = [9usize, 17usize];
        let sec_rows = [640usize, 320usize, 320usize]; // q/k/v-like spans
        let total: usize = sec_rows.iter().sum();
        // fleet widths + remainders: 960/2080/3200 are NOT multiples of 256
        // (192, 32, 128 left over); 992 and 1024 bracket 1000.
        let widths = [960usize, 992, 1024, 2080, 3200];

        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
            let man = (b >> 13) & 0x3FF;
            (sign | (exp << 10) | (man as u16)).to_le_bytes()
        };
        let f16_read = |lo: u8, hi: u8| -> f64 {
            let h = u16::from_le_bytes([lo, hi]);
            let sign = if h & 0x8000 != 0 { -1.0f64 } else { 1.0f64 };
            let exp = ((h >> 10) & 0x1F) as i32;
            let man = (h & 0x3FF) as f64;
            if exp == 0 {
                sign * man * 2f64.powi(-24)
            } else {
                sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15)
            }
        };

        let mut rows: Vec<(usize, usize, f64, f64)> = Vec::new(); // cols, b_tok, max|d|, rel

        for &b_tok in b_toks.iter() {
        for &cols in widths.iter() {
            let nb = cols / 32; // q8_0 block = 32 values / 34 bytes
            let row_bytes = nb * 34;

            // -- weights: three tensors at RAW offsets with a non-zero byte
            // remainder, so the section cursor's off+rem split is exercised
            let mut sec_raw = [0usize; 3];
            let mut cur = 96usize;
            let mut w_len = 0usize;
            for s in 0..3 {
                sec_raw[s] = cur;
                cur += row_bytes * sec_rows[s];
                w_len = cur;
                cur += 64 + 128 * s; // inter-section pad (not 256-aligned)
            }
            let w_arena = Arena::new(w_len.max(4096));
            {
                let w = unsafe { std::slice::from_raw_parts_mut(w_arena.ptr, w_arena.len) };
                for s in 0..3 {
                    for r in 0..sec_rows[s] {
                        let base = sec_raw[s] + r * row_bytes;
                        for b in 0..nb {
                            let off = base + b * 34;
                            w[off..off + 2]
                                .copy_from_slice(&f16_bytes(0.03125f32 * ((rand() % 32) + 1) as f32));
                            for j in 0..32 {
                                // full int8 range: as_type<int8_t> must sign-extend
                                w[off + 2 + j] = (rand() % 256) as u8;
                            }
                        }
                    }
                }
            }
            let w_buf = engine.register_zerocopy_only(w_arena.as_slice()).expect("w buf");

            // -- slab: x[b_tok][cols] then the three output regions
            let x_f32 = b_tok * cols;
            let region = b_tok * total;
            let out_off = x_f32 * 4;
            let slab_arena = Arena::new(out_off + region * 4);
            let slab = unsafe { std::slice::from_raw_parts_mut(slab_arena.ptr, slab_arena.len) };
            for t in 0..b_tok {
                for i in 0..cols {
                    let v = ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5;
                    slab[(t * cols + i) * 4..(t * cols + i) * 4 + 4]
                        .copy_from_slice(&v.to_le_bytes());
                }
            }
            // poison the output region: a silently dropped dispatch would
            // otherwise read back as zeros and look like a clean tensor
            for byte in slab[out_off..].iter_mut() {
                *byte = 0x7F;
            }
            let slab_buf = engine.register_zerocopy_only(slab_arena.as_slice()).expect("slab");

            // -- y layout: the sections are CONTIGUOUS regions in the slab
            // (prefix sum of b_tok*rows), which is exactly what the table's
            // y_off encodes. Every index below MUST use this prefix base: the
            // first cut of this test used s*b_tok*sec_rows[s], which makes
            // section 1 start at 2880 instead of 5760 and silently compares
            // (and references) across sections - an unsound oracle that
            // "failed" a correct kernel and could have hidden a real one.
            let mut sec_ybase = [0usize; 3];
            {
                let mut acc = 0usize;
                for s in 0..3 {
                    sec_ybase[s] = acc;
                    acc += b_tok * sec_rows[s];
                }
            }

            // -- host reference: dequant from the SAME bytes, f64 dot
            let reference = {
                let mut v = vec![0f64; region];
                let w = w_arena.as_slice();
                for s in 0..3 {
                    for t in 0..b_tok {
                        for r in 0..sec_rows[s] {
                            let base = sec_raw[s] + r * row_bytes;
                            let mut acc = 0f64;
                            for b in 0..nb {
                                let off = base + b * 34;
                                let d = f16_read(w[off], w[off + 1]);
                                for j in 0..32 {
                                    let q = w[off + 2 + j] as i8 as f64;
                                    let xb = (t * cols + b * 32 + j) * 4;
                                    let xv = f32::from_le_bytes(
                                        slab[xb..xb + 4].try_into().unwrap(),
                                    ) as f64;
                                    acc += d * q * xv;
                                }
                            }
                            v[sec_ybase[s] + t * sec_rows[s] + r] = acc;
                        }
                    }
                }
                v
            };

            // -- section table exactly as the model route builds it
            let mut blob = [[0u32; 5]; 4];
            let base = (0..3).map(|s| sec_raw[s] & !255usize).min().unwrap();
            let mut end = 0u32;
            let mut y_acc = out_off;
            for s in 0..3 {
                blob[s][0] = ((sec_raw[s] & !255usize) - base) as u32;
                blob[s][1] = (sec_raw[s] & 255usize) as u32;
                end += sec_rows[s] as u32;
                blob[s][2] = end;
                blob[s][3] = y_acc as u32;
                blob[s][4] = sec_rows[s] as u32;
                y_acc += b_tok * sec_rows[s] * 4;
            }

            assert!(engine.batch_open());
            let ok = unsafe {
                sofuu_metal_batch_gguf_gemm_multi(
                    engine.ctx,
                    8u32, // GGML_TYPE_Q8_0
                    w_buf.handle,
                    base,
                    slab_buf.handle,
                    0,
                    slab_buf.handle,
                    0,
                    total,
                    cols,
                    b_tok,
                    blob.as_ptr() as *const u32,
                    3,
                ) == 1
            };
            assert!(engine.batch_end(), "[oracle] batch_end failed cols={cols}");
            assert!(ok, "[oracle] gemm_q8_0_multi unavailable at cols={cols}");

            // -- compare every element
            let read = |s: usize, t: usize, r: usize| -> f32 {
                let byte = out_off + (sec_ybase[s] + t * sec_rows[s] + r) * 4;
                f32::from_le_bytes(slab[byte..byte + 4].try_into().unwrap())
            };
            let mut max_d = 0f64;
            let mut max_ref = 0f64;
            let mut worst = (0usize, 0usize, 0usize);
            let mut nbad = 0usize;
            for s in 0..3 {
                for t in 0..b_tok {
                    for r in 0..sec_rows[s] {
                        let g = read(s, t, r) as f64;
                        let e = reference[sec_ybase[s] + t * sec_rows[s] + r];
                        max_ref = max_ref.max(e.abs());
                        let d = (g - e).abs();
                        if d > max_d {
                            max_d = d;
                            worst = (s, t, r);
                        }
                        if d > 4e-3 * e.abs().max(1.0) {
                            nbad += 1;
                        }
                    }
                }
            }
            // A cell that still holds the poison sentinel was never written:
            // report it apart from "wrote the wrong number" (a dropped
            // subtile/threadgroup vs a cursor defect), and fail on it even if
            // some reference value happened to be huge enough to absorb it.
            let poison = f32::from_bits(0x7F7F_7F7F);
            let mut n_untouched = 0usize;
            for s in 0..3 {
                for t in 0..b_tok {
                    for r in 0..sec_rows[s] {
                        if read(s, t, r) == poison {
                            n_untouched += 1;
                        }
                    }
                }
            }
            assert!(
                n_untouched == 0,
                "[oracle] cols={cols} b_tok={b_tok}: {n_untouched}/{} output cells were NEVER \
                 written (still the 0x7F poison sentinel) - a subtile/threadgroup was dropped \
                 or its store index is wrong",
                region
            );
            let max_rel = if max_ref > 0.0 { max_d / max_ref } else { max_d };
            println!(
                "[oracle] cols={cols:4} b_tok={b_tok:2} nb={nb} row_bytes={row_bytes} \
                 max_d={max_d:.3e} rel={max_rel:.3e} max_ref={max_ref:.3e} bad={nbad}/{} \
                 worst=(s={} t={} r={})",
                region, worst.0, worst.1, worst.2
            );
            if diag && nbad > 0 {
                let (s, t, r) = worst;
                let g = read(s, t, r) as f64;
                // Full inversion, not same-token only: a store that drops the
                // subtile token offset (t0) puts token T's value in token
                // T-t0's slot, and a same-token search cannot name that. The
                // reference values here are exact dyadics (< 2^24/128), so an
                // equality hit is a genuine provenance match, not rounding.
                let mut best = (f64::INFINITY, 0usize, 0usize, 0usize);
                for s2 in 0..3 {
                    for t2 in 0..b_tok {
                        for r2 in 0..sec_rows[s2] {
                            let e = reference[sec_ybase[s2] + t2 * sec_rows[s2] + r2];
                            let d = (g - e).abs();
                            if d < best.0 {
                                best = (d, s2, t2, r2);
                            }
                        }
                    }
                }
                // poison marker (0x7F7F7F7F) => the element was NEVER written
                let unwritten = read(s, t, r) == f32::from_bits(0x7F7F_7F7F);
                println!(
                    "[oracle]   diag: gpu={g:.6e} expected=(s={s} t={t} r={r})                      unwritten={unwritten} looked_like=(s={} t={} r={} d={:.3e})",
                    best.1, best.2, best.3, best.0
                );
                // per-token write census: a missing subtile leaves whole
                // token planes untouched -> that is the signature to report.
                let poison = f32::from_bits(0x7F7F_7F7F);
                let mut census = String::new();
                for t2 in 0..b_tok {
                    let mut n_untouched = 0usize;
                    for s2 in 0..3 {
                        for r2 in 0..sec_rows[s2] {
                            if read(s2, t2, r2) == poison {
                                n_untouched += 1;
                            }
                        }
                    }
                    census.push_str(&format!(" {t2}:{n_untouched}"));
                }
                println!("[oracle]   diag: untouched-cells per token (of {total}) ->{census}");
            }
            // SOFUU_ORACLE_DUMP=<dir>: write gpu/ref/reference metadata so the
            // failure pattern can be analysed without re-running the binary.
            if let Some(dir) = std::env::var_os("SOFUU_ORACLE_DUMP") {
                let dir = std::path::PathBuf::from(dir);
                let mut meta = String::new();
                meta.push_str(&format!("cols {cols}\nb_tok {b_tok}\ntotal {total}\n"));
                for s in 0..3 {
                    meta.push_str(&format!("sec {s} rows {} raw {}\n", sec_rows[s], sec_raw[s]));
                }
                let tag = format!("{cols}x{b_tok}");
                std::fs::write(dir.join(format!("meta_{tag}.txt")), meta).unwrap();
                let gpu: Vec<u8> = slab[out_off..out_off + region * 4].to_vec();
                std::fs::write(dir.join(format!("gpu_{tag}.bin")), &gpu).unwrap();
                let rf: Vec<u8> = reference.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect();
                std::fs::write(dir.join(format!("ref_{tag}.bin")), &rf).unwrap();
                let xr: Vec<u8> = slab[..out_off].to_vec();
                std::fs::write(dir.join(format!("x_{tag}.bin")), &xr).unwrap();
                // the weight arena too: a wrong (section,row) inversion is only
                // provable against the bytes the kernel actually read
                std::fs::write(dir.join(format!("w_{tag}.bin")), w_arena.as_slice()).unwrap();
            }
            rows.push((cols, b_tok, max_d, max_rel));
        }
        }

        let bad: Vec<_> = rows.iter().filter(|(_, _, _, rel)| *rel > 1e-4).collect();
        assert!(
            bad.is_empty(),
            "[oracle] gemm_q8_0_multi is NOT the dequant reference at these shapes: {:?} \
             (width, b_tok, max_abs_delta, rel)",
            bad
        );
    }



    /// The q4_0 twin of the q8_0 oracle above: same 32x64 tiled fused GEMM
    /// (gemm_q4_0_tiled_fused), same section table, DIFFERENT block layout —
    /// 18 bytes per 32 values, f16 d then 16 packed bytes where byte j carries
    /// value j in its low nibble and value j+16 in its high nibble, dequant
    /// d*(nib-8). q4_0 was the one quant class with NO wide-chunk GEMM at all
    /// (every projection walked one GEMV per token), so this kernel is new and
    /// gets its own reference rather than a tolerance on the per-row path.
    ///
    /// Fixtures: three sections at RAW, non-256-aligned offsets with
    /// inter-section padding (the section cursor's off+rem split); widths
    /// 960/992/1024/2080/3200 = 30/31/32/65/100 blocks, i.e. many k-tiles, and
    /// 960 & 2080 & 3200 are NOT multiples of 256 so the q4_K-family floor
    /// could never cover them; b_tok 9/17 (a 1-token tail past the 8-value
    /// subtile) and 70 (past the 64-token tile, where the store cursor has to
    /// carry t0 forward — the defect that once left tokens 8..B-1 unwritten).
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn gemm_q4_0_multi_matches_dequant_oracle() {
        let engine = match MetalEngine::new() {
            Some(e) => e,
            None => {
                eprintln!("[oracle-q4_0] no Metal device - skipped");
                return;
            }
        };

        let b_toks = [9usize, 17usize, 70usize];
        let sec_rows = [640usize, 320usize, 320usize];
        let total: usize = sec_rows.iter().sum();
        let widths = [960usize, 992, 1024, 2080, 3200];

        let mut state = 0x51ED270B9F4A7C15u64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
            let man = (b >> 13) & 0x3FF;
            (sign | (exp << 10) | (man as u16)).to_le_bytes()
        };
        let f16_read = |lo: u8, hi: u8| -> f64 {
            let h = u16::from_le_bytes([lo, hi]);
            let sign = if h & 0x8000 != 0 { -1.0f64 } else { 1.0f64 };
            let exp = ((h >> 10) & 0x1F) as i32;
            let man = (h & 0x3FF) as f64;
            if exp == 0 {
                sign * man * 2f64.powi(-24)
            } else {
                sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15)
            }
        };

        let mut rows: Vec<(usize, usize, f64, f64)> = Vec::new();

        for &b_tok in b_toks.iter() {
            for &cols in widths.iter() {
                let nb = cols / 32; // q4_0 block = 32 values / 18 bytes
                let row_bytes = nb * 18;

                let mut sec_raw = [0usize; 3];
                let mut cur = 96usize;
                let mut w_len = 0usize;
                for s in 0..3 {
                    sec_raw[s] = cur;
                    cur += row_bytes * sec_rows[s];
                    w_len = cur;
                    cur += 64 + 128 * s;
                }
                let w_arena = Arena::new(w_len.max(4096));
                {
                    let w = unsafe { std::slice::from_raw_parts_mut(w_arena.ptr, w_arena.len) };
                    for s in 0..3 {
                        for r in 0..sec_rows[s] {
                            let base = sec_raw[s] + r * row_bytes;
                            for b in 0..nb {
                                let off = base + b * 18;
                                w[off..off + 2]
                                    .copy_from_slice(&f16_bytes(0.03125f32 * ((rand() % 32) + 1) as f32));
                                // every nibble value 0..15 appears: the -8
                                // offset and both halves of each byte matter.
                                for j in 0..16 {
                                    w[off + 2 + j] = (rand() % 256) as u8;
                                }
                            }
                        }
                    }
                }
                let w_buf = engine.register_zerocopy_only(w_arena.as_slice()).expect("w buf");

                let x_f32 = b_tok * cols;
                let region = b_tok * total;
                let out_off = x_f32 * 4;
                let slab_arena = Arena::new(out_off + region * 4);
                let slab = unsafe { std::slice::from_raw_parts_mut(slab_arena.ptr, slab_arena.len) };
                for t in 0..b_tok {
                    for i in 0..cols {
                        let v = ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5;
                        slab[(t * cols + i) * 4..(t * cols + i) * 4 + 4]
                            .copy_from_slice(&v.to_le_bytes());
                    }
                }
                for byte in slab[out_off..].iter_mut() {
                    *byte = 0x7F;
                }
                let slab_buf = engine.register_zerocopy_only(slab_arena.as_slice()).expect("slab");

                let mut sec_ybase = [0usize; 3];
                {
                    let mut acc = 0usize;
                    for s in 0..3 {
                        sec_ybase[s] = acc;
                        acc += b_tok * sec_rows[s];
                    }
                }

                // -- host reference: dequant from the SAME bytes in GGML order
                // (byte j -> value j and value j+16), f64 accumulate.
                let reference = {
                    let mut v = vec![0f64; region];
                    let w = w_arena.as_slice();
                    for s in 0..3 {
                        for t in 0..b_tok {
                            for r in 0..sec_rows[s] {
                                let base = sec_raw[s] + r * row_bytes;
                                let mut acc = 0f64;
                                for b in 0..nb {
                                    let off = base + b * 18;
                                    let d = f16_read(w[off], w[off + 1]);
                                    let xb0 = (t * cols + b * 32) * 4;
                                    for j in 0..16 {
                                        let byte = w[off + 2 + j];
                                        let lo = ((byte & 0x0F) as f64) - 8.0;
                                        let hi = ((byte >> 4) as f64) - 8.0;
                                        let x_lo = f32::from_le_bytes(
                                            slab[xb0 + j * 4..xb0 + j * 4 + 4].try_into().unwrap(),
                                        ) as f64;
                                        let x_hi = f32::from_le_bytes(
                                            slab[xb0 + (j + 16) * 4..xb0 + (j + 16) * 4 + 4]
                                                .try_into()
                                                .unwrap(),
                                        ) as f64;
                                        acc += d * (lo * x_lo + hi * x_hi);
                                    }
                                }
                                v[sec_ybase[s] + t * sec_rows[s] + r] = acc;
                            }
                        }
                    }
                    v
                };

                let mut blob = [[0u32; 5]; 4];
                let base = (0..3).map(|s| sec_raw[s] & !255usize).min().unwrap();
                let mut end = 0u32;
                let mut y_acc = out_off;
                for s in 0..3 {
                    blob[s][0] = ((sec_raw[s] & !255usize) - base) as u32;
                    blob[s][1] = (sec_raw[s] & 255usize) as u32;
                    end += sec_rows[s] as u32;
                    blob[s][2] = end;
                    blob[s][3] = y_acc as u32;
                    blob[s][4] = sec_rows[s] as u32;
                    y_acc += b_tok * sec_rows[s] * 4;
                }

                assert!(engine.batch_open());
                let ok = unsafe {
                    sofuu_metal_batch_gguf_gemm_tiled_multi(
                        engine.ctx,
                        2u32, // GGML_TYPE_Q4_0
                        w_buf.handle,
                        base,
                        slab_buf.handle,
                        0,
                        slab_buf.handle,
                        0,
                        total,
                        cols,
                        b_tok,
                        blob.as_ptr() as *const u32,
                        3,
                    ) == 1
                };
                assert!(engine.batch_end(), "[oracle-q4_0] batch_end failed cols={cols}");
                assert!(ok, "[oracle-q4_0] gemm_q4_0_tiled_fused unavailable at cols={cols}");

                let read = |s: usize, t: usize, r: usize| -> f32 {
                    let byte = out_off + (sec_ybase[s] + t * sec_rows[s] + r) * 4;
                    f32::from_le_bytes(slab[byte..byte + 4].try_into().unwrap())
                };
                let mut max_d = 0f64;
                let mut max_ref = 0f64;
                let mut worst = (0usize, 0usize, 0usize);
                let mut nbad = 0usize;
                for s in 0..3 {
                    for t in 0..b_tok {
                        for r in 0..sec_rows[s] {
                            let g = read(s, t, r) as f64;
                            let e = reference[sec_ybase[s] + t * sec_rows[s] + r];
                            max_ref = max_ref.max(e.abs());
                            let d = (g - e).abs();
                            if d > max_d {
                                max_d = d;
                                worst = (s, t, r);
                            }
                            if d > 4e-3 * e.abs().max(1.0) {
                                nbad += 1;
                            }
                        }
                    }
                }
                let poison = f32::from_bits(0x7F7F_7F7F);
                let mut n_untouched = 0usize;
                for s in 0..3 {
                    for t in 0..b_tok {
                        for r in 0..sec_rows[s] {
                            if read(s, t, r) == poison {
                                n_untouched += 1;
                            }
                        }
                    }
                }
                assert!(
                    n_untouched == 0,
                    "[oracle-q4_0] cols={cols} b_tok={b_tok}: {n_untouched}/{region} output cells \
                     were NEVER written (still the 0x7F poison sentinel) - a subtile/threadgroup \
                     was dropped or its store index is wrong"
                );
                let max_rel = if max_ref > 0.0 { max_d / max_ref } else { max_d };
                println!(
                    "[oracle-q4_0] cols={cols:4} b_tok={b_tok:2} nb={nb} row_bytes={row_bytes} \
                     max_d={max_d:.3e} rel={max_rel:.3e} max_ref={max_ref:.3e} bad={nbad}/{region} \
                     worst=(s={} t={} r={})",
                    worst.0, worst.1, worst.2
                );
                rows.push((cols, b_tok, max_d, max_rel));
            }
        }

        let bad: Vec<_> = rows.iter().filter(|(_, _, _, rel)| *rel > 1e-4).collect();
        assert!(
            bad.is_empty(),
            "[oracle-q4_0] gemm_q4_0_tiled_fused is NOT the dequant reference at these shapes: \
             {:?} (width, b_tok, max_abs_delta, rel)",
            bad
        );
    }

    /// Same oracle for the q6_K twin (\`gemm_q6_k_multi\`), which carried the
    /// identical store-index defect and had no numeric test of its own.
    ///
    /// q6_K moves 256 values per 210-byte block (ql[128], qh[64], sc[16]i8,
    /// d f16 at +208), so this arm is limited to the widths where the host
    /// floor actually admits q6_K (cols % 256 == 0): 1024 and 3072. The
    /// reference is the GGML dequant order (dequantize_row_q6_K) in f64 -
    /// NOT a transcription of the kernel's lane mapping, so the two agree only
    /// if the kernel's mapping is right.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn gemm_q6_k_multi_matches_dequant_oracle() {
        let engine = match MetalEngine::new() {
            Some(e) => e,
            None => {
                eprintln!("[oracle-q6] no Metal device - skipped");
                return;
            }
        };
        let b_toks = [9usize, 17usize];
        let sec_rows = [512usize, 256usize, 256usize];
        let total: usize = sec_rows.iter().sum();
        let widths = [1024usize, 3072];

        let mut state = 0xC2B2AE3D27D4EB4Fu64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
            let man = (b >> 13) & 0x3FF;
            (sign | (exp << 10) | (man as u16)).to_le_bytes()
        };
        let f16_read = |lo: u8, hi: u8| -> f64 {
            let h = u16::from_le_bytes([lo, hi]);
            let sign = if h & 0x8000 != 0 { -1.0f64 } else { 1.0f64 };
            let exp = ((h >> 10) & 0x1F) as i32;
            let man = (h & 0x3FF) as f64;
            if exp == 0 {
                sign * man * 2f64.powi(-24)
            } else {
                sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15)
            }
        };

        let mut rows: Vec<(usize, usize, f64, f64)> = Vec::new();

        for &b_tok in b_toks.iter() {
            for &cols in widths.iter() {
                let nb = cols / 256; // q6_K super-block = 256 values / 210 bytes
                let row_bytes = nb * 210;

                let mut sec_raw = [0usize; 3];
                let mut cur = 192usize; // non-zero byte remainder on section 0
                let mut w_len = 0usize;
                for s in 0..3 {
                    sec_raw[s] = cur;
                    cur += row_bytes * sec_rows[s];
                    w_len = cur;
                    cur += 96 + 64 * s; // pad, deliberately not 256-aligned
                }
                let w_arena = Arena::new(w_len.max(4096));
                {
                    let w = unsafe { std::slice::from_raw_parts_mut(w_arena.ptr, w_arena.len) };
                    for s in 0..3 {
                        for r in 0..sec_rows[s] {
                            let base = sec_raw[s] + r * row_bytes;
                            for b in 0..nb {
                                let off = base + b * 210;
                                for j in 0..128 {
                                    w[off + j] = (rand() % 256) as u8; // ql
                                }
                                for j in 0..64 {
                                    w[off + 128 + j] = (rand() % 256) as u8; // qh
                                }
                                for j in 0..16 {
                                    w[off + 192 + j] = (rand() % 256) as u8; // i8 scales
                                }
                                w[off + 208..off + 210]
                                    .copy_from_slice(&f16_bytes(0.03125 * ((rand() % 8) + 1) as f32));
                            }
                        }
                    }
                }
                let w_buf = engine.register_zerocopy_only(w_arena.as_slice()).expect("q6 w buf");

                let x_f32 = b_tok * cols;
                let region = b_tok * total;
                let out_off = x_f32 * 4;
                let slab_arena = Arena::new(out_off + region * 4);
                let slab = unsafe { std::slice::from_raw_parts_mut(slab_arena.ptr, slab_arena.len) };
                for t in 0..b_tok {
                    for i in 0..cols {
                        let v = ((i * 5 + t * 7) % 19) as f32 * 0.25 - 2.25;
                        slab[(t * cols + i) * 4..(t * cols + i) * 4 + 4]
                            .copy_from_slice(&v.to_le_bytes());
                    }
                }
                for byte in slab[out_off..].iter_mut() {
                    *byte = 0x7F;
                }
                let slab_buf = engine.register_zerocopy_only(slab_arena.as_slice()).expect("slab");

                let mut sec_ybase = [0usize; 3];
                {
                    let mut acc = 0usize;
                    for s in 0..3 {
                        sec_ybase[s] = acc;
                        acc += b_tok * sec_rows[s];
                    }
                }

                // -- host reference: GGML q6_K dequant order, f64 dots
                let reference = {
                    let mut v = vec![0f64; region];
                    let w = w_arena.as_slice();
                    for s in 0..3 {
                        for t in 0..b_tok {
                            for r in 0..sec_rows[s] {
                                let base = sec_raw[s] + r * row_bytes;
                                let mut acc = 0f64;
                                for b in 0..nb {
                                    let off = base + b * 210;
                                    let d = f16_read(w[off + 208], w[off + 209]);
                                    for n in 0..2 {
                                        for l in 0..32 {
                                            let qh = w[off + 128 + n * 32 + l];
                                            let isn = 8 * n + l / 16;
                                            let q1 = (w[off + n * 64 + l] & 0x0F)
                                                | (((qh >> 0) & 3) << 4);
                                            let q2 = (w[off + n * 64 + l + 32] & 0x0F)
                                                | (((qh >> 2) & 3) << 4);
                                            let q3 = (w[off + n * 64 + l] >> 4)
                                                | (((qh >> 4) & 3) << 4);
                                            let q4 = (w[off + n * 64 + l + 32] >> 4)
                                                | (((qh >> 6) & 3) << 4);
                                            let sc = |o: usize| w[off + 192 + isn + o] as i8 as f64;
                                            let k0 = b * 256 + n * 128 + l;
                                            let xs = |k: usize| -> f64 {
                                                let xb = (t * cols + k) * 4;
                                                f32::from_le_bytes(
                                                    slab[xb..xb + 4].try_into().unwrap(),
                                                ) as f64
                                            };
                                            acc += d * sc(0) * (q1 as f64 - 32.0) * xs(k0);
                                            acc += d * sc(2) * (q2 as f64 - 32.0) * xs(k0 + 32);
                                            acc += d * sc(4) * (q3 as f64 - 32.0) * xs(k0 + 64);
                                            acc += d * sc(6) * (q4 as f64 - 32.0) * xs(k0 + 96);
                                        }
                                    }
                                }
                                v[sec_ybase[s] + t * sec_rows[s] + r] = acc;
                            }
                        }
                    }
                    v
                };

                let mut blob = [[0u32; 5]; 4];
                let base = (0..3).map(|s| sec_raw[s] & !255usize).min().unwrap();
                let mut end = 0u32;
                let mut y_acc = out_off;
                for s in 0..3 {
                    blob[s][0] = ((sec_raw[s] & !255usize) - base) as u32;
                    blob[s][1] = (sec_raw[s] & 255usize) as u32;
                    end += sec_rows[s] as u32;
                    blob[s][2] = end;
                    blob[s][3] = y_acc as u32;
                    blob[s][4] = sec_rows[s] as u32;
                    y_acc += b_tok * sec_rows[s] * 4;
                }

                assert!(engine.batch_open());
                let ok = unsafe {
                    sofuu_metal_batch_gguf_gemm_multi(
                        engine.ctx,
                        14u32, // GGML_TYPE_Q6_K
                        w_buf.handle,
                        base,
                        slab_buf.handle,
                        0,
                        slab_buf.handle,
                        0,
                        total,
                        cols,
                        b_tok,
                        blob.as_ptr() as *const u32,
                        3,
                    ) == 1
                };
                assert!(engine.batch_end(), "[oracle-q6] batch_end failed cols={cols}");
                assert!(ok, "[oracle-q6] gemm_q6_k_multi unavailable at cols={cols}");

                let read = |s: usize, t: usize, r: usize| -> f32 {
                    let byte = out_off + (sec_ybase[s] + t * sec_rows[s] + r) * 4;
                    f32::from_le_bytes(slab[byte..byte + 4].try_into().unwrap())
                };
                let poison = f32::from_bits(0x7F7F_7F7F);
                let mut max_d = 0f64;
                let mut max_ref = 0f64;
                let mut n_untouched = 0usize;
                let mut worst = (0usize, 0usize, 0usize);
                for s in 0..3 {
                    for t in 0..b_tok {
                        for r in 0..sec_rows[s] {
                            let g = read(s, t, r);
                            if g == poison {
                                n_untouched += 1;
                            }
                            let g = g as f64;
                            let e = reference[sec_ybase[s] + t * sec_rows[s] + r];
                            max_ref = max_ref.max(e.abs());
                            let dd = (g - e).abs();
                            if dd > max_d {
                                max_d = dd;
                                worst = (s, t, r);
                            }
                        }
                    }
                }
                let max_rel = if max_ref > 0.0 { max_d / max_ref } else { max_d };
                println!(
                    "[oracle-q6] cols={cols:4} b_tok={b_tok:2} nb={nb} row_bytes={row_bytes} \
                     max_d={max_d:.3e} rel={max_rel:.3e} max_ref={max_ref:.3e} \
                     untouched={n_untouched}/{} worst=(s={} t={} r={})",
                    region, worst.0, worst.1, worst.2
                );
                assert!(
                    n_untouched == 0,
                    "[oracle-q6] cols={cols} b_tok={b_tok}: {n_untouched}/{} cells never written",
                    region
                );
                rows.push((cols, b_tok, max_d, max_rel));
            }
        }

        let bad: Vec<_> = rows.iter().filter(|(_, _, _, rel)| *rel > 1e-4).collect();
        assert!(
            bad.is_empty(),
            "[oracle-q6] gemm_q6_k_multi is NOT the dequant reference: {:?} \
             (width, b_tok, max_abs_delta, rel)",
            bad
        );
    }
}


#[cfg(all(test, target_os = "macos"))]
mod q4k_crafted_probe {
    use super::*;

    /// GPU2.1 acceptance: the one-commit glue kernels (f32-weight RMSNorm,
    /// QK-norm + RoPE + KV append, GQA attention) match a CPU reference built
    /// from the SAME free functions the CPU path uses (quant::batched_
    /// rmsnorm_in_place + quant::rope_style + plain softmax), for BOTH rope
    /// interleave modes. Same tolerance class as the GEMV gates above:
    /// different summation order, no different values.
    #[test]
    fn llama_one_commit_kernels_match_cpu_reference() {
        use crate::infer::quant::{batched_rmsnorm_in_place, rope_style};

        let engine = MetalEngine::new().expect("metal engine");
        let dim = 128usize;
        let n_heads = 4usize;
        let n_kv = 2usize;
        let hd = 32usize;
        let max_seq = 64usize;
        let pos = 7usize;
        let eps = 1e-5f32;
        let rope_base = 10000.0f32;

        for rope_mode in [0u32, 1u32] {
            // ── deterministic inputs ──
            let x: Vec<f32> = (0..dim)
                .map(|i| ((i % 13) as f32) * 0.5f32 - 3.0f32)
                .collect();
            let q0: Vec<f32> = (0..n_heads * hd)
                .map(|i| ((i % 9) as f32) * 0.25f32 - 1.0f32)
                .collect();
            let k0: Vec<f32> = (0..n_kv * hd)
                .map(|i| ((i % 7) as f32) * 0.5f32 - 1.5f32)
                .collect();
            let v0: Vec<f32> = (0..n_kv * hd)
                .map(|i| ((i % 11) as f32) * 0.125f32 - 0.5f32)
                .collect();
            let q_ln: Vec<f32> = (0..hd).map(|i| 1.0f32 + (i % 5) as f32 * 0.05).collect();
            let k_ln: Vec<f32> = (0..hd).map(|i| 1.0f32 + (i % 3) as f32 * 0.07).collect();
            let norm_w: Vec<f32> = (0..dim)
                .map(|i| 0.9f32 + (i % 6) as f32 * 0.03)
                .collect();

            // ── CPU reference ──
            let mut xn = vec![0f32; dim];
            crate::infer::quant::rmsnorm(&mut xn, &x, &norm_w, eps);
            let mut q = q0.clone();
            let mut k = k0.clone();
            batched_rmsnorm_in_place(&mut q, &q_ln, n_heads, hd, eps);
            batched_rmsnorm_in_place(&mut k, &k_ln, n_kv, hd, eps);
            rope_style(&mut q, n_heads, hd, pos, rope_base, rope_mode as i32);
            rope_style(&mut k, n_kv, hd, pos, rope_base, rope_mode as i32);
            let mut k_cache = vec![0f32; n_kv * max_seq * hd];
            let mut v_cache = vec![0f32; n_kv * max_seq * hd];
            for kv in 0..n_kv {
                let off = (kv * max_seq + pos) * hd;
                k_cache[off..off + hd].copy_from_slice(&k[kv * hd..(kv + 1) * hd]);
                v_cache[off..off + hd].copy_from_slice(&v0[kv * hd..(kv + 1) * hd]);
            }
            let mut head_out = vec![0f32; n_heads * hd];
            let scale = 1.0f32 / (hd as f32).sqrt();
            for h in 0..n_heads {
                let kv = h / (n_heads / n_kv);
                let qh = &q[h * hd..(h + 1) * hd];
                let seq = pos + 1;
                let mut scores = vec![0f32; seq];
                let mut mx = f32::NEG_INFINITY;
                for t in 0..seq {
                    let base = (kv * max_seq + t) * hd;
                    let mut dot = 0f32;
                    for i in 0..hd {
                        dot += qh[i] * k_cache[base + i];
                    }
                    scores[t] = dot * scale;
                    mx = mx.max(scores[t]);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                for d in 0..hd {
                    let mut acc = 0f32;
                    for t in 0..seq {
                        acc += scores[t] * v_cache[(kv * max_seq + t) * hd + d];
                    }
                    head_out[h * hd + d] = acc;
                }
            }

            // ── GPU one-commit chain ──
            let page = 16384usize;
            // weights buffer: q_ln @0, k_ln@128, attn_norm@4096 bytes
            let mut wb = vec![0u8; 8192];
            wb[0..hd * 4].copy_from_slice(unsafe { std::slice::from_raw_parts(q_ln.as_ptr() as *const u8, hd * 4) });
            wb[128..128 + hd * 4].copy_from_slice(unsafe { std::slice::from_raw_parts(k_ln.as_ptr() as *const u8, hd * 4) });
            wb[4096..4096 + dim * 4].copy_from_slice(unsafe { std::slice::from_raw_parts(norm_w.as_ptr() as *const u8, dim * 4) });
            let wb = engine.register_zerocopy_buffer(&wb).expect("wbuf");

            // act slab (f32 offsets, byte offsets = *4): x@0 xn@dim qa@2dim ka@3dim ... —
            // simple fixed map:
            let f_x = 0usize;
            let f_xn = dim;
            let f_qa = 2 * dim;
            let f_ka = 3 * dim;
            let f_va = f_ka + n_kv * hd;
            let f_ao = f_va + n_kv * hd;
            let slab_f32 = f_ao + n_heads * hd + 256;
            let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
            let slab_ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    slab_alloc,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(slab_ptr != libc::MAP_FAILED);
            let slab: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(slab_ptr as *mut u8, slab_alloc) };
            let kv_alloc = page; // n_kv*max_seq*hd*4 = 16KB = one page
            let k_ptr = unsafe {
                libc::mmap(std::ptr::null_mut(), kv_alloc, libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
            };
            let v_ptr = unsafe {
                libc::mmap(std::ptr::null_mut(), kv_alloc, libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
            };
            assert!(k_ptr != libc::MAP_FAILED && v_ptr != libc::MAP_FAILED);
            let k_cache_b: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(k_ptr as *mut u8, kv_alloc) };
            let v_cache_b: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(v_ptr as *mut u8, kv_alloc) };

            let put = |slab: &mut [u8], f_off: usize, vals: &[f32]| {
                unsafe {
                    std::slice::from_raw_parts_mut(slab.as_mut_ptr() as *mut f32, slab.len() / 4)
                        [f_off..f_off + vals.len()]
                        .copy_from_slice(vals)
                }
            };
            put(slab, f_x, &x);
            put(slab, f_qa, &q0);
            put(slab, f_ka, &k0);
            put(slab, f_va, &v0);

            let slab_buf = engine.register_zerocopy_buffer(slab).expect("slab");
            let k_buf = engine.register_zerocopy_buffer(k_cache_b).expect("kc");
            let v_buf = engine.register_zerocopy_buffer(v_cache_b).expect("vc");

            assert!(engine.batch_open());
            engine.batch_rmsnorm_f32w(&slab_buf, f_x * 4, &wb, 4096, &slab_buf, f_xn * 4, dim, eps);
            engine.batch_llama_kv_write(
                &slab_buf, f_ka * 4,
                &slab_buf, f_va * 4,
                &k_buf, 0,
                &v_buf, 0,
                Some(&wb), 128,
                1, pos as u32,
                hd as u32, n_kv as u32, max_seq as u32,
                eps, rope_base, rope_mode,
            );
            engine.batch_llama_attn(
                &slab_buf, f_qa * 4,
                &k_buf, 0,
                &v_buf, 0,
                Some(&wb), 0,
                &slab_buf, f_ao * 4,
                1, pos as u32,
                hd as u32, n_heads as u32, n_kv as u32, max_seq as u32,
                eps, rope_base, rope_mode,
            );
            engine.batch_end();

            let view = unsafe { std::slice::from_raw_parts(slab.as_ptr() as *const f32, slab_alloc / 4) };
            // xn parity: only summation order differs
            for i in 0..dim {
                let a = view[f_xn + i];
                let b = xn[i];
                assert!(
                    (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
                    "rope_mode {rope_mode}: xn[{i}] {a} vs {b}"
                );
            }
            // head_out parity
            for i in 0..n_heads * hd {
                let a = view[f_ao + i];
                let b = head_out[i];
                assert!(
                    (a - b).abs() <= 2e-3 * (1.0 + b.abs()),
                    "rope_mode {rope_mode}: head_out[{i}] {a} vs {b}"
                );
            }
            // KV append parity: roped k lands in the cache at pos
            let k_view = unsafe { std::slice::from_raw_parts(k_ptr as *const f32, kv_alloc / 4) };
            for kv in 0..n_kv {
                let off = (kv * max_seq + pos) * hd;
                for i in 0..hd {
                    let a = k_view[off + i];
                    let b = k_cache[off + i];
                    assert!(
                        (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                        "rope_mode {rope_mode}: k_cache[{off}+{i}] {a} vs {b}"
                    );
                }
            }
        }
    }

    /// Working-set experiment: q4_K GEMV (12288x4096) whose weights sit
    /// (a) at offset 0 of a compact 28MB buffer vs (b) 3GB deep inside a
    /// 5GB registered buffer (real-client geometry). Same window, same op.
    #[test]
    #[ignore]
    fn bench_weights_geometry() {
        use std::os::fd::AsRawFd;
        use std::time::Instant;
        let engine = MetalEngine::new().expect("metal engine");
        let rows = 12288usize;
        let cols = 4096usize;
        let blocks = cols / 256;
        let wlen = rows * blocks * 144;
        // slab with x + y slots
        let page = 16384usize;
        let slab_alloc = page;
        let slab_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), slab_alloc, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(slab_ptr != libc::MAP_FAILED);
        unsafe {
            let xf = slab_ptr as *mut f32;
            for i in 0..cols { *xf.add(i) = ((i % 7) as f32) * 0.25 - 0.75; }
        }
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr as *mut u8, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        let y_off = 8192; // after 4096 f32 x

        let gb5 = 5usize * (1 << 30);
        let big_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), gb5, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(big_ptr != libc::MAP_FAILED);
        let deep = 3usize * (1 << 30);
        unsafe {
            // fill the deep window with a pattern (touches the pages)
            let p = (big_ptr as *mut u8).add(deep);
            for i in (0..wlen).step_by(4096) {
                *p.add(i) = 0x5A;
            }
        }
        let big: &[u8] = unsafe { std::slice::from_raw_parts(big_ptr as *const u8, gb5) };
        let big_buf = engine.register_zerocopy_only(big).expect("big");

        let compact = vec![0x5Au8; wlen];
        let compact_buf = engine.register_zerocopy_only(&compact).expect("compact");

        let run = |buf: &MetalBuffer, off: usize, tag: &str| {
            for rep in 0..2 {
                let iters = if rep == 0 { 2 } else { 5 };
                let t0 = Instant::now();
                for _ in 0..iters {
                    assert!(engine.batch_open());
                    engine.batch_gguf(12, buf, off, &slab_buf, 0, &slab_buf, y_off, rows, cols);
                    engine.batch_end();
                }
                let per = t0.elapsed().as_secs_f64() / iters as f64;
                println!(
                    "[weights geometry] {:<12} per_op={:.2}ms bw={:.1} GB/s",
                    tag,
                    per * 1e3,
                    wlen as f64 / per / 1e9
                );
            }
        };
        run(&compact_buf, 0, "compact-28M");
        run(&big_buf, deep, "3G-in-5G");
        run(&big_buf, 1 << 30, "1G-in-5G");
        run(&big_buf, gb5 - wlen - 16 * 1024, "4.8G-in-5G");
        run(&compact_buf, 0, "compact-28M-again");
        run(&big_buf, deep, "3G-in-5G-again");

        // File-backed vs anon: same 28MB GEMV, weights in a real FILE mmap.
        let fpath = std::env::temp_dir().join("sofuu_gpugeo_5g.bin");
        {
            let f = std::fs::File::create(&fpath).expect("create file");
            f.set_len(gb5 as u64).expect("set_len");
            use std::io::{Write, Seek, SeekFrom};
            let mut f = std::fs::OpenOptions::new().write(true).open(&fpath).expect("open");
            f.seek(SeekFrom::Start(deep as u64)).expect("seek");
            let chunk = vec![0x5Au8; 1 << 20];
            for _ in 0..(wlen >> 20) {
                f.write_all(&chunk).expect("write");
            }
            f.write_all(&vec![0x5Au8; wlen % (1 << 20)]).expect("write tail");
            f.sync_all().expect("sync");
        }
        let f = std::fs::File::open(&fpath).expect("open");
        let fmap = unsafe {
            libc::mmap(std::ptr::null_mut(), gb5, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0)
        };
        assert!(fmap != libc::MAP_FAILED);
        let fbuf: &[u8] = unsafe { std::slice::from_raw_parts(fmap as *const u8, gb5) };
        let file_buf = engine.register_zerocopy_only(fbuf).expect("filebuf");
        run(&file_buf, deep, "file-deep-5G");
        run(&file_buf, deep, "file-deep-5G-warm");

        // Full-token sweep: 40 chained GEMVs, each 28MB but at offsets spread
        // across the whole 5GB buffer (like the 36-layer sequence touching
        // every part of the file once per token). ~1.1GB touched per sweep.
        for rep in 0..3 {
            let spread: Vec<usize> = (0..40)
                .map(|i| (i + 1) * (gb5 / 41) & !16383)
                .collect();
            let t0 = Instant::now();
            let iters = 3;
            for _ in 0..iters {
                assert!(engine.batch_open());
                for (j, &off) in spread.iter().enumerate() {
                    let yoff = 8192 + (j % 2) * 4096 * 4;
                    let xoff = if j % 2 == 0 { 0 } else { 8192 };
                    engine.batch_gguf(12, &big_buf, off, &slab_buf, xoff, &slab_buf, yoff, rows, cols);
                }
                engine.batch_end();
            }
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            let bytes = (spread.len() * wlen) as f64;
            println!(
                "[token-sweep] 40x28MB spread over 5GB: per_sweep={:.1}ms aggregate_bw={:.1} GB/s",
                dt * 1e3,
                bytes / dt / 1e9
            );
        }
    }

    /// Op-overhead attribution: how much does each in-batch op/barrier cost
    /// at REAL client shapes, chained (dependent) like the one-commit client?
    /// Bench cases per iteration of K ops in one batch:
    ///   glue : K x [rmsnorm(4096) -> add(4096)] chained in/out same slots
    ///   gguf : K x batch_gguf q4_K 12288x4096 chained y->x slot ping-pong
    /// Run: cargo test -p sofuu-core --lib bench_op_overhead -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_op_overhead() {
        use std::time::Instant;
        let engine = MetalEngine::new().expect("metal engine");
        let dim = 4096usize;
        let page = 16384usize;
        // slab: x@0 xn@dim xo@2dim (f32 slots), plus ping-pong y slots
        let slots_f32 = 8 * dim;
        let alloc = ((slots_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                alloc,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(slab_ptr != libc::MAP_FAILED);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr as *mut u8, alloc) };
        let slab_buf = engine.register_zerocopy_buffer(slab).expect("slab");
        let norm_f32 = vec![1.0f32; dim];
        let mut wb = vec![0u8; 16384];
        unsafe {
            std::ptr::copy_nonoverlapping(
                norm_f32.as_ptr() as *const u8,
                wb.as_mut_ptr(),
                dim * 4,
            )
        };
        let w_buf = engine.register_zerocopy_buffer(&wb).expect("wbuf");

        // gguf chain W: reuse the q4_K FFN weight blob (zeros fine for timing)
        let rows = 12288usize;
        let cols = 4096usize;
        let blocks = cols / 256;
        let wbuf2 = vec![0x80u8; rows * blocks * 144];
        let w_buf2 = engine.register_zerocopy_buffer(&wbuf2).expect("w2");

        for case in ["glue", "gguf"] {
            let iters = if case == "glue" { 4 } else { 2 };
            for rep in 0..iters {
                let k = if case == "glue" { 256 } else { 36 };
                let t0 = Instant::now();
                assert!(engine.batch_open());
                for j in 0..k {
                    if case == "glue" {
                        // alternating rmsnorm/add chain, x->xn->x slots
                        if j % 2 == 0 {
                            engine.batch_rmsnorm_f32w(
                                &slab_buf, 0, &w_buf, 0, &slab_buf, (1 * dim) * 4, dim, 1e-5,
                            );
                        } else {
                            engine.batch_add(&slab_buf, 0, &slab_buf, (1 * dim) * 4, dim);
                        }
                    } else {
                        // ping-pong y into slots 2..3 dim-offset region (timing only)
                        let yoff = ((2 + (j % 2)) * dim) * 4;
                        let xoff = if j % 2 == 0 { 0 } else { (2 + ((j + 1) % 2)) * dim * 4 };
                        engine.batch_gguf(12, &w_buf2, 0, &slab_buf, xoff, &slab_buf, yoff, rows, cols);
                    }
                }
                engine.batch_end();
                let per = t0.elapsed().as_secs_f64() / k as f64;
                println!(
                    "[bench op-overhead] case={} k={:<4} per_op={:.0} us  ops/s/slot-sweep={:.1}",
                    case,
                    k,
                    per * 1e6,
                    1.0 / per
                );
            }
        }
    }

    /// Reproduce ONE REAL Qwen3-8B layer, operator-for-operator, in one batch
    /// and ablate pieces to attribute the per-layer time. Shapes: dim 4096,
    /// 32/8 heads x 128, inter 12288, pos 8, all GEMVs q4_K.
    /// Ablate via: cargo ... bench_layer_mimic -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_layer_mimic() {
        use std::time::Instant;
        let engine = MetalEngine::new().expect("metal engine");
        let dim = 4096usize;
        let q_rows = 4096usize;   // 32 x 128
        let kv_rows = 1024usize;  // 8 x 128
        let hd = 128usize;
        let inter = 12288usize;
        let vocab_rows = 151936usize;
        let page = 16384usize;

        // ── weight blobs (q4_K), nominal offsets ──
        let qk4 = |rows: usize, cols: usize| vec![0x80u8; rows * (cols / 256) * 144];
        let wq = qk4(q_rows, dim);
        let wk = qk4(kv_rows, dim);
        let wv = qk4(kv_rows, dim);
        let wo = qk4(dim, q_rows);
        let wg = qk4(inter, dim);
        let wu = qk4(inter, dim);
        let wd = qk4(dim, inter);
        let q6_row_b = (dim / 256) * 210;
        let wh = vec![0x80u8; vocab_rows * q6_row_b];
        let wj = wq.len() + wk.len() + wv.len();

        let w_alloc = wq.len() + wk.len() + wv.len() + wo.len() + wg.len() + wu.len() + wd.len();
        let w_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                ((w_alloc + page - 1) / page) * page,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(w_ptr != libc::MAP_FAILED);
        let wbase = w_ptr as *mut u8;
        let mut put = |off: usize, src: &[u8]| unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), wbase.add(off), src.len())
        };
        put(0, &wq);
        put(wq.len(), &wk);
        put(wq.len() + wk.len(), &wv);
        put(wj, &wo);
        put(wj + wo.len(), &wg);
        put(wj + wo.len() + wg.len(), &wu);
        put(wj + wo.len() + wg.len() + wu.len(), &wd);
        let w_slice: &[u8] =
            unsafe { std::slice::from_raw_parts(wbase, ((w_alloc + page - 1) / page) * page) };
        let w_buf = engine.register_zerocopy_only(w_slice).expect("weights");
        let head_alloc = ((wh.len() + page - 1) / page) * page;
        let wh_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                head_alloc,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(wh_ptr != libc::MAP_FAILED);
        unsafe {
            std::ptr::copy_nonoverlapping(wh.as_ptr(), wh_ptr as *mut u8, wh.len())
        };
        let wh_slice: &[u8] = unsafe { std::slice::from_raw_parts(wh_ptr as *const u8, head_alloc) };
        let wh_buf = engine.register_zerocopy_only(wh_slice).expect("head weights");

        // ── activation slab: x, xn, qa, ka, va, ao, xo, gate, up, logits (f32) ──
        let logits_f32 = vocab_rows;
        let slab_f32 = 6 * dim + q_rows + 2 * kv_rows + 2 * inter + logits_f32;
        let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                slab_alloc,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(slab_ptr != libc::MAP_FAILED);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr as *mut u8, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        let fx = 0usize;
        let fxn = dim;
        let fqa = 2 * dim;
        let fka = fqa + q_rows;
        let fva = fka + kv_rows;
        let fao = fva + kv_rows;
        let fxo = fao + q_rows;
        let fgate = fxo + dim;
        let fup = fgate + inter;
        let flog = fup + inter;
        let _ = flog;

        // norms (f32 dim) @0/8164, q/k norms (hd) @ head-dim offsets
        let mk = |bytes: usize| unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        let nw3_ptr = mk(8192);
        assert!(nw3_ptr != libc::MAP_FAILED);
        let wb3: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(nw3_ptr as *mut u8, 8192) };
        unsafe {
            std::ptr::copy_nonoverlapping(
                [1.0f32; 4096].as_ptr() as *const u8,
                wb3.as_mut_ptr(),
                4096 * 4,
            )
        };
        let wnorm = engine.register_zerocopy_only(wb3).expect("norm");

        // KV caches
        let kv_f32 = kv_rows * 64 * hd; // max_seq 64
        let kv_alloc = (((kv_f32 * 4) + page - 1) / page) * page;
        let k_ptr = mk(kv_alloc);
        let v_ptr = mk(kv_alloc);
        assert!(k_ptr != libc::MAP_FAILED && v_ptr != libc::MAP_FAILED);
        let kb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(k_ptr as *mut u8, kv_alloc) };
        let vb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(v_ptr as *mut u8, kv_alloc) };
        let k_buf = engine.register_zerocopy_only(kb).expect("k");
        let v_buf = engine.register_zerocopy_only(vb).expect("v");

        // ablations: full | no-attn (skip kv_write+attn) | no-glue (skip norms+adds) | gemv-only | with-head
        for mode in ["gemv-only", "no-glue", "no-attn", "full", "full+head"] {
            for rep in 0..2 {
                let k_layers = if mode == "full+head" { 4 } else { 8 };
                let t0 = Instant::now();
                assert!(engine.batch_open());
                for li in 0..k_layers {
                    if mode != "no-glue" && mode != "gemv-only" {
                        engine.batch_rmsnorm_f32w(&slab_buf, fx * 4, &wnorm, 0, &slab_buf, fxn * 4, dim, 1e-5);
                    }
                    engine.batch_gguf(12, &w_buf, 0, &slab_buf, fxn * 4, &slab_buf, fqa * 4, q_rows, dim);
                    engine.batch_gguf(12, &w_buf, wq.len(), &slab_buf, fxn * 4, &slab_buf, fka * 4, kv_rows, dim);
                    engine.batch_gguf(12, &w_buf, wq.len() + wk.len(), &slab_buf, fxn * 4, &slab_buf, fva * 4, kv_rows, dim);
                    if mode == "full" || mode == "full+head" {
                        engine.batch_llama_kv_write(
                            &slab_buf, fka * 4,
                            &slab_buf, fva * 4,
                            &k_buf, 0,
                            &v_buf, 0,
                            Some(&wnorm), 4096,
                            1, 8,
                            hd as u32, 8, 64,
                            1e-5, 10000.0, 0,
                        );
                        engine.batch_llama_attn(
                            &slab_buf, fqa * 4,
                            &k_buf, 0,
                            &v_buf, 0,
                            Some(&wnorm), 4096,
                            &slab_buf, fao * 4,
                            1, 8,
                            hd as u32, 32, 8, 64,
                            1e-5, 10000.0, 0,
                        );
                    }
                    engine.batch_gguf(12, &w_buf, wj, &slab_buf, fao * 4, &slab_buf, fxo * 4, dim, q_rows);
                    if mode != "no-glue" && mode != "gemv-only" {
                        engine.batch_add(&slab_buf, fx * 4, &slab_buf, fxo * 4, dim);
                        engine.batch_rmsnorm_f32w(&slab_buf, fx * 4, &wnorm, 0, &slab_buf, fxn * 4, dim, 1e-5);
                    }
                    engine.batch_gguf(12, &w_buf, wj + wo.len(), &slab_buf, fxn * 4, &slab_buf, fgate * 4, inter, dim);
                    engine.batch_gguf(12, &w_buf, wj + wo.len() + wg.len(), &slab_buf, fxn * 4, &slab_buf, fup * 4, inter, dim);
                    if mode != "gemv-only" {
                        engine.batch_swiglu(&slab_buf, fgate * 4, &slab_buf, fup * 4, inter);
                    }
                    engine.batch_gguf(12, &w_buf, wj + wo.len() + wg.len() + wu.len(), &slab_buf, fgate * 4, &slab_buf, fxo * 4, dim, inter);
                    if mode != "no-glue" && mode != "gemv-only" {
                        engine.batch_add(&slab_buf, fx * 4, &slab_buf, fxo * 4, dim);
                    }
                }
                if mode == "full+head" {
                    engine.batch_gguf(14, &wh_buf, 0, &slab_buf, fxn * 4, &slab_buf, flog * 4, vocab_rows, dim);
                }
                engine.batch_end();
                let per = t0.elapsed().as_secs_f64() / k_layers as f64;
                println!(
                    "[layer mimic] mode={:<10} per_layer={:.3} ms (x36={:.1} ms/token)",
                    mode,
                    per * 1e3,
                    per * 1e3 * 36.0
                );
            }
        }
    }

    /// P1 prefill GEMM parity (llama.cpp-parity program): y[t][r] for 5
    /// tokens (odd B exercises the subtile mask) must equal the CPU oracle
    /// vec_dot_raw_q4_k at every (t, r). Tolerance-gated like the GEMV class;
    /// also a compact-blob rate measurement for the kernel class.
    #[test]
    #[ignore] // GPU bench class: run with `cargo test -- --ignored` at the engine
    fn gemm_q4_k_matches_cpu() {
        use std::time::Instant;

        let rows = 1024usize;
        let blocks = 16usize; // cols 4096
        let cols = blocks * 256;
        let row_bytes = blocks * 144;
        let B = 5usize;

        let mut state = 0x5DEECE66Du64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
            let man = (b >> 13) & 0x3FF;
            (sign | (exp << 10) | (man as u16)).to_le_bytes()
        };
        // sanitized fixture (random bytes in f16 headers would decode NaN —
        // the bitwise-fixture trap): fixed d/dmin, random packed scales+qs
        let mut w = vec![0u8; rows * row_bytes];
        for r in 0..rows {
            for b in 0..blocks {
                let off = r * row_bytes + b * 144;
                w[off..off + 2].copy_from_slice(&f16_bytes(0.0625f32));
                w[off + 2..off + 4].copy_from_slice(&f16_bytes(0.0005f32));
                for j in 0..12 {
                    w[off + 4 + j] = (rand() & 0x3F) as u8 | ((rand() & 3) << 6) as u8;
                }
                for j in 0..128 {
                    w[off + 16 + j] = (rand() % 255) as u8;
                }
            }
        }
        let toks: Vec<Vec<f32>> = (0..B)
            .map(|t| (0..cols).map(|i| ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5).collect())
            .collect();
        let mut y_cpu = vec![0f32; B * rows];
        for t in 0..B {
            for r in 0..rows {
                let off = r * row_bytes;
                y_cpu[t * rows + r] = unsafe {
                    crate::infer::quant::vec_dot_raw_q4_k(w[off..].as_ptr(), toks[t].as_ptr(), blocks)
                };
            }
        }

        let page = 16384usize;
        let w_alloc = ((w.len() + page - 1) / page) * page;
        let mk = |bytes: usize| -> *mut u8 {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED);
            p as *mut u8
        };
        let w_ptr = mk(w_alloc);
        let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w.len()) };
        wb.copy_from_slice(&w);
        let engine = MetalEngine::new().expect("engine");
        let w_buf = engine.register_zerocopy_only(wb).expect("w");

        // x[B][cols] + y[B][rows] in one slab
        let slab_f32 = B * cols + B * rows;
        let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = mk(slab_alloc);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        for t in 0..B {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    toks[t].as_ptr() as *const u8,
                    slab_ptr.add(t * cols * 4),
                    cols * 4,
                )
            };
        }
        let y_off = B * cols * 4;

        // probe + dispatch + wait
        let dispatched = {
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
            engine.batch_end();
            ok
        };
        assert!(dispatched, "gemm_q4_k PSO unavailable — build the metallib");

        let y_union: &[u8] = unsafe { std::slice::from_raw_parts(slab_ptr.add(y_off), B * rows * 4) };
        let y_gpu: &[f32] = unsafe { std::slice::from_raw_parts(y_union.as_ptr() as *const f32, B * rows) };
        // dots here reach ~1.7e4, so the gate is RELATIVE (abs 3.9e-3 was
        // 2.3e-7 relative = plain f32 summation reassociation, same class
        // as the GEMV gates); 1e-6 rel + 1e-4 abs floor for near-zero rows
        let mut worst = 0f32;
        let (mut wt, mut wr, mut wa, mut wb) = (0usize, 0usize, 0f32, 0f32);
        for t in 0..B {
            for r in 0..rows {
                let a = y_gpu[t * rows + r];
                let b = y_cpu[t * rows + r];
                let tol = 1e-6 * b.abs().max(1.0);
                let e = (a - b).abs();
                if e > worst { worst = e; wt = t; wr = r; wa = a; wb = b; }
                assert!(
                    e <= tol.max(1e-4),
                    "gemm_q4_k parity broken at t={t} r={r}: gpu={a:.6e} cpu={b:.6e} (rel {:.2e})",
                    e / b.abs().max(1.0)
                );
            }
        }
        println!("[gemm_q4_k] rows={rows} cols={cols} B={B} worst_abs={worst:e} worst t={wt} r={wr} gpu={wa:.6e} cpu={wb:.6e} — PARITY PASS");

        // rate: repeat the dispatch with fresh x, time the commit
        let t0 = Instant::now();
        let reps = 8usize;
        for _ in 0..reps {
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
            engine.batch_end();
            assert!(ok);
        }
        let per = t0.elapsed() / reps as u32;
        let bytes = rows as f64 * row_bytes as f64;
        println!("[gemm_q4_k] per-dispatch {:.3?} ms -> {:.1} GB/s weight stream", per.as_secs_f64() * 1e3, bytes / per.as_secs_f64() / 1e9);
    }

    /// P1 op-diet parity: gemm_q4_k_multi must equal per-tensor gemm_q4_k
    /// for EVERY section (3 odd row counts exercise the span resolver, the
    /// per-section y offsets and per-region strides).
    #[test]
    #[ignore]
    fn gemm_q4_k_multi_matches_split() {
        let blocks = 8usize; // cols 2048
        let cols = blocks * 256;
        let row_bytes = blocks * 144;
        let B = 9usize; // odd: exercises the subtile mask
        let sec_rows = [512usize, 384usize, 640usize];
        let total: usize = sec_rows.iter().sum();

        let mut state = 0x2545F4914F6CDD1Du64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
            let man = (b >> 13) & 0x3FF;
            (sign | (exp << 10) | (man as u16)).to_le_bytes()
        };
        // three DIFFERENT sanitized weight tensors (different per-block d?
        // no — fixed d/dmin like the gemm gate; scales+qs random per tensor)
        let mut w = vec![0u8; 3 * total * row_bytes + 4096];
        let mut sec_w = [0usize; 3];
        let mut cur = 0usize; // byte cursor in w
        for s in 0..3 {
            sec_w[s] = cur;
            for r in 0..sec_rows[s] {
                let row_start = cur;
                for b in 0..blocks {
                    let off = row_start + b * 144;
                    w[off..off + 2].copy_from_slice(&f16_bytes(0.0625f32));
                    w[off + 2..off + 4].copy_from_slice(&f16_bytes(0.0005f32));
                    for j in 0..12 {
                        w[off + 4 + j] = (rand() & 0x3F) as u8 | ((rand() & 3) << 6) as u8;
                    }
                    for j in 0..128 {
                        w[off + 16 + j] = (rand() % 255) as u8;
                    }
                }
                cur = row_start + row_bytes;
            }
            cur += 1024; // inter-section pad (also breaks same-offset luck)
        }
        let w_len = cur;

        let page = 16384usize;
        let mk = |bytes: usize| -> *mut u8 {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED);
            p as *mut u8
        };
        let w_alloc = ((w_len + page - 1) / page) * page;
        let w_ptr = mk(w_alloc);
        let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w_len) };
        wb.copy_from_slice(&w[..w_len]);
        let engine = MetalEngine::new().expect("engine");
        let w_buf = engine.register_zerocopy_only(wb).expect("w");

        // slab: x[B][cols] then ref[B][total] then out[B][total]
        let x_f32 = B * cols;
        let region = B * total;
        let alloc_f32 = x_f32 + 2 * region;
        let slab_alloc = ((alloc_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = mk(slab_alloc);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        for t in 0..B {
            for i in 0..cols {
                let v = ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5;
                unsafe { *(slab_ptr.add((t * cols + i) * 4) as *mut f32) = v };
            }
        }
        let ref_off = x_f32 * 4;
        let out_off = (x_f32 + region) * 4;

        // per-tensor singles into ref regions — PACKED by cumulative prefix,
        // not s*B*rows[s] (that packing overlaps s1/s2 and spilled past the
        // region end — the first test run's phantom "kernel bug")
        let mut sec_ref: [usize; 3] = [0; 3];
        let mut sec_out: [usize; 3] = [0; 3];
        {
            let mut accr = ref_off;
            let mut acco = out_off;
            for s in 0..3 {
                sec_ref[s] = accr;
                sec_out[s] = acco;
                accr += B * sec_rows[s] * 4;
                acco += B * sec_rows[s] * 4;
            }
        }
        let mut probe = true;
        for s in 0..3 {
            let rows = sec_rows[s];
            let y_s = sec_ref[s];
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(
                12, &w_buf, sec_w[s] & !255, &slab_buf, 0, &slab_buf, y_s, rows, cols, B,
            );
            engine.batch_end();
            assert!(ok, "single gemm s={s} unavailable");
            probe &= ok;
        }

        // fused multi into out region (sections at their own strided slices)
        let mut blob: [[u32; 5]; 4] = [[0u32; 5]; 4];
        let mut end = 0u32;
        let base = sec_w[0] & !255usize;
        for (s, rows) in sec_rows.iter().enumerate() {
            let raw = sec_w[s];
            blob[s][0] = (((raw & !255usize) - base).min(u32::MAX as usize)) as u32;
            blob[s][1] = (raw & 255usize) as u32;
            end += *rows as u32;
            blob[s][2] = end;
            blob[s][3] = sec_out[s] as u32;
            blob[s][4] = *rows as u32; // region row stride == region rows
        }
        assert!(engine.batch_open());
        let ok = engine.batch_gguf_gemm_multi(
            12, &w_buf, base, &slab_buf, 0, &slab_buf, 0, total, cols, B, &blob,
        );
        engine.batch_end();
        assert!(ok, "multi gemm unavailable");
        let _ = probe;

        // compare section-by-section
        let f32_at = |byte_off: usize, idx: usize| -> f32 {
            unsafe { *(slab_ptr.add(byte_off + idx * 4) as *const f32) }
        };
        let mut worst = 0f32;
        let mut fails = 0usize;
        let (mut f_s, mut f_r, mut f_t) = (0usize, 0usize, 0usize);
        let mut per_t = [0usize; 64];
        for s in 0..3 {
            let rows = sec_rows[s];
            for t in 0..B {
                for r in 0..rows {
                    let a = f32_at(sec_out[s], t * rows + r);
                    let b = f32_at(sec_ref[s], t * rows + r);
                    let e = (a - b).abs();
                    let tol = 1e-6 * b.abs().max(1.0);
                    if e > tol.max(1e-4) {
                        fails += 1;
                        if fails <= 6 {
                            println!("[gemm_multi] FAIL s={s} t={t} r={r}: {a:.4e} vs {b:.4e}");
                        }
                        per_t[t] += 1;
                        worst = worst.max(e);
                        f_s = s; f_r = r; f_t = t;
                        continue;
                    }
                    worst = worst.max(e);
                }
            }
        }
        println!(
            "[gemm_multi] fails={fails} worst={worst:e} per_t={:?} last_s={f_s} last_r={f_r} last_t={f_t}",
            &per_t[..B]
        );
        assert!(fails == 0, "gemm_q4_k_multi parity broken: {fails} mismatches");
    }

    /// P2 decode-diet parity: gemv_q4_k_b4multi must equal per-section
    /// batch_gguf_b4 for every section (all 4 token rows per threadgroup).
    #[test]
    #[ignore]
    fn gemv_q4_k_b4multi_matches_split() {
        let blocks = 8usize; // cols 2048
        let cols = blocks * 256;
        let row_bytes = blocks * 144;
        let sec_rows = [512usize, 384usize, 640usize];
        let total: usize = sec_rows.iter().sum();

        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
            let man = (b >> 13) & 0x3FF;
            (sign | (exp << 10) | (man as u16)).to_le_bytes()
        };
        let mut w = vec![0u8; 3 * total * row_bytes + 4096];
        let mut sec_w = [0usize; 3];
        let mut cur = 0usize;
        for s in 0..3 {
            sec_w[s] = cur;
            for r in 0..sec_rows[s] {
                let row_start = cur;
                for b in 0..blocks {
                    let off = row_start + b * 144;
                    w[off..off + 2].copy_from_slice(&f16_bytes(0.0625f32));
                    w[off + 2..off + 4].copy_from_slice(&f16_bytes(0.0005f32));
                    for j in 0..12 {
                        w[off + 4 + j] = (rand() & 0x3F) as u8 | ((rand() & 3) << 6) as u8;
                    }
                    for j in 0..128 {
                        w[off + 16 + j] = (rand() % 255) as u8;
                    }
                }
                cur = row_start + row_bytes;
            }
            // NON-ALIGNED inter-section pads (+64): the model's real tensors
            // sit at raw offsets with rem up to 255 (e.g. ffn_gate rem=64)
            cur += 64;
        }
        let w_len = cur;

        let page = 16384usize;
        let mk = |bytes: usize| -> *mut u8 {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED);
            p as *mut u8
        };
        let w_alloc = ((w_len + page - 1) / page) * page;
        let w_ptr = mk(w_alloc);
        let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w_len) };
        wb.copy_from_slice(&w[..w_len]);
        let engine = MetalEngine::new().expect("engine");
        let w_buf = engine.register_zerocopy_only(wb).expect("w");

        // slab: x[4][cols] + ref[4][total] + out[4][total]
        let x_f32 = 4 * cols;
        let region = 4 * total;
        let alloc_f32 = x_f32 + 2 * region;
        let slab_alloc = ((alloc_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = mk(slab_alloc);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        for t in 0..4 {
            for i in 0..cols {
                let v = ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5;
                unsafe { *(slab_ptr.add((t * cols + i) * 4) as *mut f32) = v };
            }
        }
        let ref_off = x_f32 * 4;
        let out_off = (x_f32 + region) * 4;

        // per-tensor singles: batch_gguf_b4 into [4][rows] regions (packed)
        let mut sec_ref: [usize; 3] = [0; 3];
        let mut sec_out: [usize; 3] = [0; 3];
        {
            let mut accr = ref_off;
            let mut acco = out_off;
            for s in 0..3 {
                sec_ref[s] = accr;
                sec_out[s] = acco;
                accr += 4 * sec_rows[s] * 4;
                acco += 4 * sec_rows[s] * 4;
            }
        }
        for s in 0..3 {
            assert!(engine.batch_open());
            engine.batch_gguf_b4(
                &w_buf, sec_w[s], &slab_buf, 0, &slab_buf, sec_ref[s], sec_rows[s], cols,
            );
            engine.batch_end();
        }

        // fused multi
        let mut blob: [[u32; 5]; 4] = [[0u32; 5]; 4];
        let mut end = 0u32;
        let base = sec_w[0] & !255usize;
        for (s, rows) in sec_rows.iter().enumerate() {
            let raw = sec_w[s];
            blob[s][0] = (((raw & !255usize) - base).min(u32::MAX as usize)) as u32;
            blob[s][1] = (raw & 255usize) as u32;
            end += *rows as u32;
            blob[s][2] = end;
            blob[s][3] = sec_out[s] as u32;
            blob[s][4] = *rows as u32;
        }
        assert!(engine.batch_open());
        let ok = engine.batch_gguf_b4multi(
            12, &w_buf, base, &slab_buf, 0, &slab_buf, total, cols, &blob,
        );
        engine.batch_end();
        assert!(ok, "b4multi unavailable");

        let f32_at = |byte_off: usize, idx: usize| -> f32 {
            unsafe { *(slab_ptr.add(byte_off + idx * 4) as *const f32) }
        };
        let mut fails = 0usize;
        let mut worst = 0f32;
        let mut per_sec = [0usize; 3];
        let mut per_tok = [0usize; 4];
        let mut first = None;
        for s in 0..3 {
            let rows = sec_rows[s];
            for t in 0..4 {
                for r in 0..rows {
                    let a = f32_at(sec_out[s], t * rows + r);
                    let b = f32_at(sec_ref[s], t * rows + r);
                    let e = (a - b).abs();
                    let tol = 1e-6 * b.abs().max(1.0);
                    if e > tol.max(1e-4) {
                        fails += 1;
                        per_sec[s] += 1;
                        per_tok[t] += 1;
                        if first.is_none() { first = Some((s, t, r, a, b, sec_w[s])); }
                    }
                    worst = worst.max(e);
                }
            }
        }
        println!(
            "[b4multi] fails={fails} worst={worst:e} per_sec={per_sec:?} per_tok={per_tok:?} first={:?}",
            first.map(|(s, t, r, a, b, raw)| (s, t, r, a as u64, b as u64, raw, raw & 255))
        );
        assert!(fails == 0, "b4multi parity broken: {fails} mismatches");
    }

    /// P1 completion: gemv_q6_k_b4multi must equal per-section q6_K dots
    /// (vec_dot_raw_q6_k oracle) across 3 sections with unaligned offsets.
    #[test]
    #[ignore]
    fn gemv_q6_k_b4multi_oracle() {
        let blocks = 8usize; // cols 2048
        let cols = blocks * 256;
        let row_bytes = blocks * 210; // q6_K: 210 bytes per 256 values
        let sec_rows = [512usize, 384usize, 640usize];
        let total: usize = sec_rows.iter().sum();

        let mut state = 0x853C49E6748FEA9Bu64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
            let man = (b >> 13) & 0x3FF;
            (sign | (exp << 10) | (man as u16)).to_le_bytes()
        };
        // sanitized: d fixed, random qh (6-bit packing!) + random scales
        // (signed int8) + random ql
        let mut w = vec![0u8; 3 * total * row_bytes + 4096];
        let mut sec_w = [0usize; 3];
        let mut cur = 0usize;
        for s in 0..3 {
            sec_w[s] = cur;
            for r in 0..sec_rows[s] {
                let row_start = cur;
                for b in 0..blocks {
                    let off = row_start + b * 210;
                    w[off..off + 2].copy_from_slice(&f16_bytes(0.03125f32));
                    for j in 0..128 {
                        w[off + j] = (rand() % 256) as u8;
                    }
                    for j in 0..64 {
                        w[off + 128 + j] = (rand() % 256) as u8;
                    }
                    for j in 0..16 {
                        w[off + 192 + j] = ((rand() % 63) as i32 - 31) as u8; // int8 scale
                    }
                }
                cur = row_start + row_bytes;
            }
            cur += 64; // unaligned pads (rem != 0)
        }
        let w_len = cur;

        let page = 16384usize;
        let mk = |bytes: usize| -> *mut u8 {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED);
            p as *mut u8
        };
        let w_alloc = ((w_len + page - 1) / page) * page;
        let w_ptr = mk(w_alloc);
        let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w_len) };
        wb.copy_from_slice(&w[..w_len]);
        let engine = MetalEngine::new().expect("engine");
        let w_buf = engine.register_zerocopy_only(wb).expect("w");

        let x_f32 = 4 * cols;
        let region = 4 * total;
        let alloc_f32 = x_f32 + region;
        let slab_alloc = ((alloc_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = mk(slab_alloc);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        for t in 0..4 {
            for i in 0..cols {
                let v = ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5;
                unsafe { *(slab_ptr.add((t * cols + i) * 4) as *mut f32) = v };
            }
        }
        let out_off = x_f32 * 4;

        // CPU oracle per token per row
        let mut sec_cpu = [[0f32; 4]; 3];
        let mut outs: Vec<(usize, usize)> = Vec::new(); // (sec, y_off)
        {
            let mut off = out_off;
            for s in 0..3 {
                outs.push((s, off));
                off += 4 * sec_rows[s] * 4;
            }
        }
        // y regions: REUSE one slab (out areas after x)
        let mut blob: [[u32; 5]; 4] = [[0u32; 5]; 4];
        let mut end = 0u32;
        let base = sec_w[0] & !255usize;
        for (s, rows) in sec_rows.iter().enumerate() {
            let raw = sec_w[s];
            blob[s][0] = (((raw & !255usize) - base).min(u32::MAX as usize)) as u32;
            blob[s][1] = (raw & 255usize) as u32;
            end += *rows as u32;
            blob[s][2] = end;
            blob[s][3] = outs[s].1 as u32;
            blob[s][4] = *rows as u32;
        }
        assert!(engine.batch_open());
        let ok = engine.batch_gguf6_b4multi(14, &w_buf, base, &slab_buf, 0, &slab_buf, total, cols, &blob);
        engine.batch_end();
        assert!(ok, "q6 b4multi unavailable");

        for s in 0..3 {
            for t in 0..4 {
                for r in 0..sec_rows[s] {
                    let row = sec_w[s] + r * row_bytes;
                    let x = unsafe { std::slice::from_raw_parts(slab_ptr as *const f32, cols) };
                    let _ = x; // oracle reads token t
                    let xt = unsafe {
                        std::slice::from_raw_parts(
                            (slab_ptr as *const f32).add(t * cols),
                            cols,
                        )
                    };
                    let cpu = unsafe {
                        crate::infer::quant::vec_dot_raw_q6_k(
                            w.as_ptr().add(row), xt.as_ptr(), blocks,
                        )
                    };
                    sec_cpu[s][t] = cpu; // store one sample per (s,t)
                    break; // only row 0 per (s,t) for the assert loop below
                }
            }
        }
        let f32_at = |byte_off: usize, idx: usize| -> f32 {
            unsafe { *(slab_ptr.add(byte_off + idx * 4) as *const f32) }
        };
        let mut worst = 0f32;
        let mut fails = 0usize;
        for (s, off) in &outs {
            let rows = sec_rows[*s];
            for t in 0..4 {
                let a = f32_at(*off, t * rows);
                let b = sec_cpu[*s][t];
                let e = (a - b).abs();
                let tol = 1e-5 * b.abs().max(1.0);
                if e > tol.max(1e-3) {
                    fails += 1;
                    println!("[q6b4] FAIL s={s} t={t} r=0: {a:.4e} vs {b:.4e}");
                }
                worst = worst.max(e);
            }
        }
        println!("[q6b4] worst={worst:e} fails={fails}");
        assert!(fails == 0, "q6 b4multi parity broken");
        println!("[q6b4] PARITY PASS (3 sections, unaligned, q6_K oracle)");
    }

    /// P2: gemv_q8_0_b4multi oracle parity (3 sections, unaligned offsets).
    #[test]
    #[ignore]
    fn gemv_q8_0_b4multi_oracle() {
        let blocks = 8usize; // cols 256
        let cols = blocks * 32;
        let row_bytes = blocks * 34; // q8_0: 32 qs + 2 d
        let sec_rows = [64usize, 48usize, 80usize];
        let total: usize = sec_rows.iter().sum();

        let mut state = 0xDEADBEEFCAFEBABEu64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
            let man = (b >> 13) & 0x3FF;
            (sign | (exp << 10) | (man as u16)).to_le_bytes()
        };
        let mut w = vec![0u8; 3 * total * row_bytes + 4096];
        let mut sec_w = [0usize; 3];
        let mut cur = 0usize;
        for s in 0..3 {
            sec_w[s] = cur;
            for r in 0..sec_rows[s] {
                let row_start = cur;
                for b in 0..blocks {
                    let off = row_start + b * 34;
                    w[off..off + 2].copy_from_slice(&f16_bytes(0.03125f32));
                    for j in 0..32 {
                        w[off + 2 + j] = (rand() % 256) as u8;
                    }
                }
                cur = row_start + row_bytes;
            }
            cur += 64; // unaligned pads
        }
        let w_len = cur;

        let page = 16384usize;
        let mk = |bytes: usize| -> *mut u8 {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED);
            p as *mut u8
        };
        let w_alloc = ((w_len + page - 1) / page) * page;
        let w_ptr = mk(w_alloc);
        let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w_len) };
        wb.copy_from_slice(&w[..w_len]);
        let engine = MetalEngine::new().expect("engine");
        let w_buf = engine.register_zerocopy_only(wb).expect("w");

        let x_f32 = 4 * cols;
        let alloc_f32 = x_f32 + 4 * total;
        let slab_alloc = ((alloc_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = mk(slab_alloc);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        for t in 0..4 {
            for i in 0..cols {
                let v = ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5;
                unsafe { *(slab_ptr.add((t * cols + i) * 4) as *mut f32) = v };
            }
        }
        let out_off = x_f32 * 4;

        let mut blob: [[u32; 5]; 4] = [[0u32; 5]; 4];
        let mut end = 0u32;
        let base = sec_w[0] & !255usize;
        let mut offs = [0usize; 3];
        let mut o = out_off;
        for (s, rows) in sec_rows.iter().enumerate() {
            offs[s] = o;
            o += 4 * rows * 4; // placeholder spacing (unused directly)
        }
        // recompute exact per-section y offsets (divisible regions in order)
        let mut oo = out_off;
        for (s, rows) in sec_rows.iter().enumerate() {
            offs[s] = oo;
            oo += 4 * rows * 4;
        }
        for (s, rows) in sec_rows.iter().enumerate() {
            let raw = sec_w[s];
            blob[s][0] = (((raw & !255usize) - base).min(u32::MAX as usize)) as u32;
            blob[s][1] = (raw & 255usize) as u32;
            end += *rows as u32;
            blob[s][2] = end;
            blob[s][3] = offs[s] as u32;
            blob[s][4] = *rows as u32;
        }
        assert!(engine.batch_open());
        let ok = engine.batch_gguf8_b4multi(8, &w_buf, base, &slab_buf, 0, &slab_buf, total, cols, &blob);
        engine.batch_end();
        assert!(ok, "q8 b4multi unavailable");

        let f32_at = |byte_off: usize, idx: usize| -> f32 {
            unsafe { *(slab_ptr.add(byte_off + idx * 4) as *const f32) }
        };
        let mut fails = 0usize;
        let mut worst = 0f32;
        for s in 0..3 {
            for t in 0..4 {
                for r in 0..sec_rows[s] {
                    let row = sec_w[s] + r * row_bytes;
                    let xt = unsafe {
                        std::slice::from_raw_parts((slab_ptr as *const f32).add(t * cols), cols)
                    };
                    let cpu = unsafe {
                        crate::infer::quant::vec_dot_raw_q8_0(w.as_ptr().add(row), xt.as_ptr(), blocks)
                    };
                    let a = f32_at(offs[s], t * sec_rows[s] + r);
                    let e = (a - cpu).abs();
                    let tol = 1e-5 * cpu.abs().max(1.0);
                    if e > tol.max(2e-3) {
                        fails += 1;
                        if fails <= 4 {
                            println!("[q8b4] FAIL s={s} t={t} r={r}: {a:.4e} vs {cpu:.4e}");
                        }
                    }
                    worst = worst.max(e);
                }
            }
        }
        println!("[q8b4] worst={worst:e} fails={fails}");
        assert!(fails == 0, "q8 b4multi parity broken: {fails}");

        // c2 oracle gate (the serial-decode class): batch_gguf ty=8 routes
        // to gemv_q8_0_c2 when SOFUU_Q4K_COALESCED>0 at engine init. Single
        // token (x row 0), compare rows 0..min(sec rows,4) against the C
        // q8_0 oracle.
        {
            let mut cblob_ok = false;
            // region for y: reuse the per-token layout — y[sec_rows] at offs
            let c2_y = out_off; // same tail region (refresh from x row 0)
            assert!(engine.batch_open());
            engine.batch_gguf(8, &w_buf, sec_w[0] & !255, &slab_buf, 0, &slab_buf, c2_y, 1, cols);
            engine.batch_end();
            cblob_ok = true;
            assert!(cblob_ok, "c2 dispatch failed");
            let row = sec_w[0];
            let cpu = unsafe {
                crate::infer::quant::vec_dot_raw_q8_0(
                    w.as_ptr().add(row),
                    slab_ptr as *const f32,
                    blocks,
                )
            };
            let a = f32_at(c2_y, 0);
            let e2 = (a - cpu).abs();
            assert!(e2 <= (1e-5 * cpu.abs().max(1.0)).max(2e-3), "c2 parity: {a:.4e} vs {cpu:.4e}");
            println!("[q8c2] c2-class oracle gate PASS (section 0, COALESCED env decides the psO)");
        }

        // rate: the REAL 1B FFN shape (4096 rows x 2048 cols = 13.5MB q8_0),
        // 4-token x; llama.cpp's effective decode rate on the 1B ~150 GB/s
        use std::time::Instant;
        {
            let fb_rows = 4096usize;
            let fb_cols = 2048usize;
            let fb_blocks = fb_cols / 32; // 64
            let fb_rb = fb_blocks * 34;
            let fb_bytes = fb_rows * fb_rb;
            let fb_alloc = ((fb_bytes + page - 1) / page) * page;
            let fb_ptr = mk(fb_alloc);
            let mut state2 = 0x1234567890ABCDEFu64;
            let mut rand2 = move || {
                state2 = state2
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state2 >> 33) as u32
            };
            let fbb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(fb_ptr, fb_bytes) };
            // sanitized: d first (f16 0.03125), random int8 qs
            for r in 0..fb_rows {
                for b in 0..fb_blocks {
                    let off = r * fb_rb + b * 34;
                    fbb[off..off + 2].copy_from_slice(&f16_bytes(0.03125f32));
                    for j in 0..32 {
                        fbb[off + 2 + j] = (rand2() % 256) as u8;
                    }
                }
            }
            let fb_buf = engine.register_zerocopy_only(fbb).expect("fb");
            // x slab already [4][cols-test] — cols differ (2048 vs test cols
            // 256): reuse slab_base x region? test x = [4][256] — too narrow.
            // allocate a dedicated x for the big dispatch (4 x 2048 floats).
            let bx_bytes = 4 * fb_cols * 4;
            let bx_ptr = mk(bx_bytes.max(page));
            for t in 0..4 {
                for i in 0..fb_cols {
                    let v = ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5;
                    unsafe { *(bx_ptr.add((t * fb_cols + i) * 4) as *mut f32) = v };
                }
            }
            let bx_buf = engine
                .register_zerocopy_only(unsafe {
                    std::slice::from_raw_parts_mut(bx_ptr, bx_bytes.max(page))
                })
                .expect("bx");
            // y region: reuse the tail of the big x slab? place y after x in a
            // padded slab (4 x 4096 floats)
            let y_needed = 4 * fb_rows * 4;
            let byy = bx_bytes.max(page);
            // y lives inside the SAME registered buffer boundaries — bx is
            // exactly bx_bytes; register one more buffer for y
            let by_ptr = mk(y_needed.max(page));
            let by_buf = engine
                .register_zerocopy_only(unsafe {
                    std::slice::from_raw_parts_mut(by_ptr, y_needed.max(page))
                })
                .expect("by");
            // one q8_0 section covering the whole tensor: blob base = aligned 0
            let mut bblob: [[u32; 5]; 4] = [[0u32; 5]; 4];
            bblob[0][0] = 0;
            bblob[0][1] = 0;
            bblob[0][2] = fb_rows as u32;
            bblob[0][3] = 0; // y bound at 0 (absolute)
            // y buffer bound at offset 0 -> y_off absolute 0
            bblob[0][4] = fb_rows as u32;
            let reps = 8usize;
            let t0 = Instant::now();
            for _ in 0..reps {
                assert!(engine.batch_open());
                let ok = engine.batch_gguf8_b4r4(8, &fb_buf, 0, &bx_buf, 0, &by_buf, fb_rows, fb_cols, &bblob);
                engine.batch_end();
                assert!(ok, "r4 fb dispatch unavailable");
            }
            let per = t0.elapsed().as_secs_f64();
            println!(
                "[q8b4] R4 rate: {} dispatches {:.2?} -> {:.1} GB/s weight stream ({} rows x {} cols)",
                reps, t0.elapsed(), fb_bytes as f64 * reps as f64 / per / 1e9, fb_rows, fb_cols
            );
            // V1 per-ROW class (batch_gguf ty=8 = the serial-decode kernel):
            // rows_total single-token x width cols — run 4 times for the 4 tokens? the serial decode uses ONE token: x = [cols] single row. Use x rows = 1 token: bind bx row 0 (bx holds 4 rows) and 4 sequential single dispatches? batch_gguf handles one (x,y) pair per op: rows=4096, x=bx row0 (2048 floats at the bx buffer), y=by row 0.
            {
                let mut vblob: [[u32; 5]; 4] = [[0u32; 5]; 4];
                let _ = &vblob;
                let t1 = Instant::now();
                for _ in 0..reps {
                    assert!(engine.batch_open());
                    engine.batch_gguf(8, &fb_buf, 0, &bx_buf, 0, &by_buf, 0, fb_rows, fb_cols);
                    engine.batch_end();
                }
                let per1 = t1.elapsed().as_secs_f64();
                println!(
                    "[q8b4] V1 per-row GEMV rate (single token): -> {:.1} GB/s",
                    fb_bytes as f64 * reps as f64 / per1 / 1e9
                );
            }
            // q4_K class reference on the SAME shape (4096 x 2048 padded q4 blob)
            let q4_rb = 144usize * (fb_cols / 256); // 144*8 = 1152
            let qb_bytes = fb_rows * q4_rb;
            let qb_ptr = mk(((qb_bytes + page - 1) / page) * page);
            let qbb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(qb_ptr, qb_bytes) };
            // fixed d/dmin headers + random bytes (matches the gemm fixture rules)
            for r in 0..fb_rows {
                for b in 0..(fb_cols / 256) {
                    let off = r * q4_rb + b * 144;
                    qbb[off..off + 2].copy_from_slice(&f16_bytes(0.0625f32));
                    qbb[off + 2..off + 4].copy_from_slice(&f16_bytes(0.0005f32));
                    for j in 0..140 {
                        qbb[off + 4 + j] = (rand2() % 256) as u8;
                    }
                }
            }
            let qb_buf = engine.register_zerocopy_only(qbb).expect("qb");
            let qt = 2048usize; // 4-token x width for q4 = cols 2048 (q4 gemm shape)
            let mut qblob: [[u32; 5]; 4] = [[0u32; 5]; 4];
            qblob[0][0] = 0; qblob[0][1] = 0; qblob[0][2] = fb_rows as u32; qblob[0][3] = 0; qblob[0][4] = fb_rows as u32;
            let t2 = Instant::now();
            for _ in 0..reps {
                assert!(engine.batch_open());
                let ok = engine.batch_gguf_gemm(12, &qb_buf, 0, &bx_buf, 0, &by_buf, 0, fb_rows, qt, 4);
                engine.batch_end();
                assert!(ok, "q4 gemm fallback");
            }
            let per2 = t2.elapsed().as_secs_f64();
            println!(
                "[q8b4] Q4K-GEMM reference -> {:.1} GB/s (same shape, 4 tokens)",
                qb_bytes as f64 * reps as f64 / per2 / 1e9
            );
        }
    }

    /// GPU3 acceptance at the REAL Qwen3-8B shape: hd 128, 32/8 heads,
    /// pos 1300, max_seq 4096 (bytes copied to a compact buffer — geometry
    /// equivalence already receipted).
    #[test]
    fn llama_attn_chunked_matches_cpu_real_shape() {
        use crate::infer::quant::{batched_rmsnorm_in_place, rope_style};
        let engine = MetalEngine::new().expect("metal engine");
        let n_heads = 32usize;
        let n_kv = 8usize;
        let hd = 128usize;
        let max_seq = 4096usize;
        let pos = 1300usize;
        let eps = 1e-5f32;
        let rope_base = 500000.0f32;
        let rope_mode = 0u32;

        let q0: Vec<f32> = (0..n_heads * hd).map(|i| ((i % 9) as f32) * 0.25 - 1.0).collect();
        let k0: Vec<f32> = (0..n_kv * hd).map(|i| ((i % 7) as f32) * 0.5 - 1.5).collect();
        let v0: Vec<f32> = (0..n_kv * hd).map(|i| ((i % 11) as f32) * 0.125 - 0.5).collect();
        let q_ln: Vec<f32> = (0..hd).map(|i| 1.0 + (i % 5) as f32 * 0.05).collect();
        let k_ln: Vec<f32> = (0..hd).map(|i| 1.0 + (i % 3) as f32 * 0.07).collect();

        let mut q = q0.clone();
        batched_rmsnorm_in_place(&mut q, &q_ln, n_heads, hd, eps);
        rope_style(&mut q, n_heads, hd, pos, rope_base, rope_mode as i32);
        let mut k = k0.clone();
        batched_rmsnorm_in_place(&mut k, &k_ln, n_kv, hd, eps);
        rope_style(&mut k, n_kv, hd, pos, rope_base, rope_mode as i32);
        let mut k_cache = vec![0f32; n_kv * max_seq * hd];
        let mut v_cache = vec![0f32; n_kv * max_seq * hd];
        for t in 0..=pos {
            for kv in 0..n_kv {
                let off = (kv * max_seq + t) * hd;
                for i in 0..hd {
                    let s = (t * 31 + i * 7 + kv * 13) % 211;
                    if t == pos {
                        // current token rows from the ROPED k / raw v
                        if i < 1 {
                            k_cache[off + i] = k[kv * hd + i];
                        }
                        v_cache[off + i] = v0[kv * hd + i];
                    } else {
                        k_cache[off + i] = (s as f32) * 0.01 - 1.0;
                        v_cache[off + i] = ((s * 3 % 97) as f32) * 0.02 - 0.5;
                    }
                }
            }
        }
        // write full roped-k current row
        for kv in 0..n_kv {
            let off = (kv * max_seq + pos) * hd;
            k_cache[off..off + hd].copy_from_slice(&k[kv * hd..(kv + 1) * hd]);
}
        let mut head_out = vec![0f32; n_heads * hd];
        let scale = 1.0f32 / (hd as f32).sqrt();
        for h in 0..n_heads {
            let kv = h / (n_heads / n_kv);
            let qh = &q[h * hd..(h + 1) * hd];
            let seq = pos + 1;
            let mut scores = vec![0f32; seq];
            let mut mx = f32::NEG_INFINITY;
            for t in 0..seq {
                let base = (kv * max_seq + t) * hd;
                let mut dot = 0f32;
                for i in 0..hd {
                    dot += qh[i] * k_cache[base + i];
                }
                scores[t] = dot * scale;
                mx = mx.max(scores[t]);
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for d in 0..hd {
                let mut acc = 0f32;
                for t in 0..seq {
                    acc += scores[t] * v_cache[(kv * max_seq + t) * hd + d];
                }
                head_out[h * hd + d] = acc;
            }
        }

        let page = 16384usize;
        let mut wb = vec![0u8; 8192];
        unsafe {
            std::ptr::copy_nonoverlapping(q_ln.as_ptr() as *const u8, wb.as_mut_ptr(), hd * 4);
            std::ptr::copy_nonoverlapping(k_ln.as_ptr() as *const u8, wb.as_mut_ptr().add(4096), hd * 4);
        }
        let wb = engine.register_zerocopy_buffer(&wb).expect("wbuf");
        let kc = engine.register_zerocopy_buffer(unsafe {
            std::slice::from_raw_parts(k_cache.as_ptr() as *const u8, k_cache.len() * 4)
        }).expect("kc");
        let vc = engine.register_zerocopy_buffer(unsafe {
            std::slice::from_raw_parts(v_cache.as_ptr() as *const u8, v_cache.len() * 4)
        }).expect("vc");
        let slab_f32 = 12 * 4096;
        let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), slab_alloc, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(slab_ptr != libc::MAP_FAILED);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr as *mut u8, slab_alloc) };
        unsafe {
            std::slice::from_raw_parts_mut(slab.as_mut_ptr() as *mut f32, slab.len() / 4)
                [..q0.len()].copy_from_slice(&q0);
        }
        let slab_buf = engine.register_zerocopy_buffer(slab).expect("slab");
        let n_chunks = (max_seq + 127) / 128;
        let meta_f32 = n_heads * n_chunks * 2 + n_heads * n_chunks * hd;
        let meta_alloc = ((meta_f32 * 4 + page - 1) / page) * page;
        let meta_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), meta_alloc, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(meta_ptr != libc::MAP_FAILED);
        let meta: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(meta_ptr as *mut u8, meta_alloc) };
        let meta_buf = engine.register_zerocopy_buffer(meta).expect("meta");

        let f_ao = 8 * 4096usize;
        assert!(engine.batch_open());
        engine.batch_head_rmsnorm(
            &slab_buf, 0,
            Some(&wb), 0,
            hd as u32, n_heads as u32,
            1, pos as u32, eps, rope_base, rope_mode,
        );
        engine.batch_llama_attn_part(
            &slab_buf, 0,
            &kc, 0,
            &vc, 0,
            None, 0,
            &meta_buf, 0,
            0, pos as u32,
            hd as u32, n_heads as u32, n_kv as u32, max_seq as u32,
            eps, rope_base, rope_mode,
        );
        engine.batch_llama_attn_fin(
            &meta_buf, 0,
            &slab_buf, f_ao * 4,
            pos as u32, n_heads as u32, hd as u32,
        );
        engine.batch_end();
        let view = unsafe { std::slice::from_raw_parts(slab.as_ptr() as *const f32, slab_alloc / 4) };
        let mut max_err = 0f32;
        for i in 0..n_heads * hd {
            let a = view[f_ao + i];
            let b = head_out[i];
            max_err = max_err.max((a - b).abs());
            assert!((a - b).abs() <= 3e-2 * (1.0 + b.abs()), "real-shape attn: head_out[{i}] {a} vs {b}");
        }
        println!("[chunked attn parity real-shape] ok: max_err={max_err}");
    }

    /// GPU3 acceptance: the B=4 batched q4_K GEMV matches the RECEIPTED CPU
    /// oracle (quant::vec_dot_raw_q4_k) for FOUR tokens at once, on crafted
    /// superblocks whose packed scale bytes zero out half the subgroups —
    /// so both scale decode and nibble→value mapping are exercised.
    #[test]
    fn gguf_b4_matches_cpu() {
        let engine = MetalEngine::new().expect("metal engine");
        let rows = 8usize;
        let blocks = 4usize; // 4 superblocks (1024 cols)
        let cols = blocks * 256;
        // superblock with dmin=0 and per-group scales = 1, qs bytes patterned
        let mut w = vec![0u8; rows * blocks * 144];
        for r in 0..rows {
            for b in 0..blocks {
                let off = (r * blocks + b) * 144;
                w[off..off + 2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1
                // scales: each group pair (sc_lo=1, sc_hi=1), mins 0
                // q4_K packed scale encoding (mirror q4k_scale_min):
                //   d(j<4)  = scales[j]&63        -> scales[0],[1] = 1 give
                //      d-scale 1 for the lo/hi subgroups of superblock pairs
                //   d(j>=4) = scales[j+4]&15 | (scales[j-4]>>6)<<4
                //                                       -> scales[8],[9] = 1
                //   m(*) via scales[4..8]>>4 and scales[j]>>6 — stay 0.
                w[off + 4 + 0] = 1;
                w[off + 4 + 1] = 1;
                w[off + 4 + 8] = 1;
                w[off + 4 + 9] = 1;
                for i in 0..128 {
                    // deterministic nibble bytes
                    w[off + 16 + i] = ((i * 37 + r * 5 + b) % 256) as u8;
                }
            }
        }
        let toks: Vec<Vec<f32>> = (0..4)
            .map(|t| (0..cols).map(|i| ((i * 5 + t * 3) % 17) as f32 * 0.25 - 2.0).collect())
            .collect();

        // CPU reference: the RECEIPTED q4_K dot (same fn the vecdot parity
        // gate uses) per token per row — no hand-rolled decode here: the
        // crafted scale bytes make subgroups {2,3,6,7} decode to d=0, and a
        // naive all-groups-with-sc=1 formula silently disagrees with that.
        let mut y_cpu = vec![0f32; 4 * rows];
        for t in 0..4 {
            for r in 0..rows {
                let off = r * blocks * 144;
                y_cpu[t * rows + r] = unsafe {
                    crate::infer::quant::vec_dot_raw_q4_k(
                        w[off..].as_ptr(),
                        toks[t].as_ptr(),
                        blocks,
                    )
                };
            }
        }

        // GPU
        let page = 16384usize;
        assert_ne!(w[16], w[16 + 576], "w blob rows must differ");
        assert_ne!(w[16 + 144], w[16 + 576 + 144], "block-1 bytes must differ too");
        let wb = engine.register_zerocopy_buffer(&w).expect("w");
        let xb_len = 4 * cols * 4 + 4 * rows * 4;
        let xb_len = ((xb_len + page - 1) / page) * page;
        let xb_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                xb_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(xb_ptr != libc::MAP_FAILED);
        let xb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(xb_ptr as *mut u8, xb_len) };
        for t in 0..4 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    toks[t].as_ptr() as *const u8,
                    xb.as_mut_ptr().add(t * cols * 4),
                    cols * 4,
                );
            }
        }
        let y_off = 4 * cols * 4;
        let yb = engine.register_zerocopy_only(xb).expect("xy");
        assert!(engine.batch_open());
        engine.batch_gguf_b4(&wb, 0, &yb, 0, &yb, y_off, rows, cols);
        engine.batch_end();
        // capture b4's answer BEFORE anything else writes these slots
        let view = unsafe { std::slice::from_raw_parts(xb.as_ptr() as *const f32, xb.len() / 4) };
        let y_b4: Vec<f32> = view[y_off / 4..y_off / 4 + 4 * rows].to_vec();

        // cross-check against the SHIPPED v1 kernel on token 0 (overwrites
        // slots y0..yrows, hence the capture above)
        assert!(engine.batch_open());
        engine.batch_gguf(12, &wb, 0, &yb, 0, &yb, y_off, rows, cols);
        engine.batch_end();
        let view = unsafe { std::slice::from_raw_parts(xb.as_ptr() as *const f32, xb.len() / 4) };
        println!(
            "[b4-vs-v1] row0: b4={} v1={} cpu={}",
            y_b4[0], view[y_off / 4], y_cpu[0]
        );

        // v1 oracle sanity: receipted kernel must match vec_dot_raw_q4_k
        let mut v1_rel = 0f64;
        for r in 0..rows {
            let a = view[y_off / 4 + r] as f64;
            let b = y_cpu[r] as f64;
            v1_rel = v1_rel.max((a - b).abs() / b.abs().max(1e-3));
        }
        println!("[b4 parity] v1-vs-cpu max_rel={v1_rel:.3e}");

        // the b4 gate: all 4 tokens x all rows against the receipted dot
        let mut max_rel = 0f64;
        for i in 0..4 * rows {
            let a = y_b4[i] as f64;
            let b = y_cpu[i] as f64;
            max_rel = max_rel.max((a - b).abs() / b.abs().max(1e-3));
        }
        println!("[b4 parity] b4-vs-cpu max_rel={max_rel:.3e}");
        assert!(v1_rel < 1e-2, "v1 oracle drifted vs vec_dot_raw_q4_k");
        assert!(
            max_rel < 1e-2,
            "b4 GPU vs CPU divergence {max_rel} (b4[0]={} cpu[0]={})",
            y_b4[0],
            y_cpu[0]
        );
        println!("[gguf b4 parity] ok: 4 tokens x {rows} rows, cols={cols}");
    }

    /// The ~150GB/s kernel-class attack: coalesced-lane q4_K GEMVs
    /// (gemv_q4_k_c single + gemv_q4_k_b4c batched, SOFUU_Q4K_COALESCED=1
    /// flips both PSOs at engine init). Different f32 summation class (per-
    /// lane partials + one simd_sum), so parity is the vec_dot_raw_q4_k
    /// 1e-2-rel class — NOT the bit-exact b4 gate. One serialized test so
    /// the process env can't race the other engines' PSO classes: coalesced
    /// engine parity first, env cleared, then a v1 engine for the same-
    /// window bench.
    ///
    /// Run: cargo test -p sofuu-core --lib q4k_coalesced -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q4k_coalesced_matches_cpu_and_bench() {
        use std::time::Instant;

        let rows = 12288usize; // FFN shape: same bench_b4_vs_v1 tensor
        let blocks = 16usize; // cols = 4096 (the 8B attn/FFN input width)
        let cols = blocks * 256;
        let row_bytes = blocks * 144;

        // deterministic q4_K blob: random packed scales + random qs (both
        // d/m and d may be zero — the oracle handles the real encoding)
        let mut state = 0x5DEECE66Du64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
            let man = (b >> 13) & 0x3FF;
            (sign | (exp << 10) | (man as u16)).to_le_bytes()
        };
        let mut w = vec![0u8; rows * row_bytes];
        for r in 0..rows {
            for b in 0..blocks {
                let off = r * row_bytes + b * 144;
                w[off..off + 2].copy_from_slice(&f16_bytes(0.0625f32));
                w[off + 2..off + 4].copy_from_slice(&f16_bytes(0.0005f32));
                for j in 0..12 {
                    w[off + 4 + j] = (rand() & 0x3F) as u8 | ((rand() & 3) << 6) as u8;
                }
                for j in 0..128 {
                    w[off + 16 + j] = (rand() % 255) as u8;
                }
            }
        }
        let toks: Vec<Vec<f32>> = (0..4)
            .map(|t| (0..cols).map(|i| ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5).collect())
            .collect();
        // CPU oracle per token per row
        let mut y_cpu = vec![0f32; 4 * rows];
        for t in 0..4 {
            for r in 0..rows {
                let off = r * row_bytes;
                y_cpu[t * rows + r] = unsafe {
                    crate::infer::quant::vec_dot_raw_q4_k(w[off..].as_ptr(), toks[t].as_ptr(), blocks)
                };
            }
        }

        let page = 16384usize;
        // one page-aligned slab: x[4][cols] then y[4][rows]
        let slab_len = 4 * cols * 4 + 4 * rows * 4 + page;
        let alloc = ((slab_len + page - 1) / page) * page;
        let slab_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                alloc,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(slab_ptr != libc::MAP_FAILED);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr as *mut u8, alloc) };
        for t in 0..4 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    toks[t].as_ptr() as *const u8,
                    slab.as_mut_ptr().add(t * cols * 4),
                    cols * 4,
                );
            }
        }
        let y_off = 4 * cols * 4;
        let y_off_f = y_off / 4;

        std::env::set_var("SOFUU_Q4K_COALESCED", "1");
        let engine_c = MetalEngine::new().expect("metal engine (coalesced)");
        std::env::remove_var("SOFUU_Q4K_COALESCED");

        let wb = engine_c.register_zerocopy_buffer(&w).expect("w");
        let slab_buf = engine_c.register_zerocopy_buffer(slab).expect("slab");

        // coalesced single-row parity
        assert!(engine_c.batch_open());
        engine_c.batch_gguf(12, &wb, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols);
        engine_c.batch_end();
        let view = unsafe { std::slice::from_raw_parts(slab.as_ptr() as *const f32, alloc / 4) };
        let mut rel = 0f64;
        for r in 0..rows {
            rel = rel.max(
                (view[y_off_f + r] as f64 - y_cpu[r] as f64).abs() / y_cpu[r].abs().max(1e-3) as f64,
            );
        }
        println!("[q4k coalesced] single max_rel={rel:.3e}");
        assert!(rel < 1e-2, "coalesced single vs oracle {rel}");

        // coalesced b4 parity (all 4 x all rows)
        assert!(engine_c.batch_open());
        engine_c.batch_gguf_b4(&wb, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols);
        engine_c.batch_end();
        let view = unsafe { std::slice::from_raw_parts(slab.as_ptr() as *const f32, alloc / 4) };
        let mut rel = 0f64;
        for i in 0..4 * rows {
            rel = rel.max(
                (view[y_off_f + i] as f64 - y_cpu[i] as f64).abs() / y_cpu[i].abs().max(1e-3) as f64,
            );
        }
        println!("[q4k coalesced] b4 max_rel={rel:.3e}");
        assert!(rel < 1e-2, "coalesced b4 vs oracle {rel}");

        // same-window bench: coalesced engine then a fresh v1 engine
        let wlen = rows * row_bytes as usize;
        for iters in [2usize, 8] {
            let t0 = Instant::now();
            for _ in 0..iters {
                assert!(engine_c.batch_open());
                engine_c.batch_gguf_b4(&wb, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols);
                engine_c.batch_end();
            }
            let per = t0.elapsed().as_secs_f64() / iters as f64;
            if iters == 8 {
                println!(
                    "[q4k coalesced] b4c per_group={:.3}ms eff_bw={:.1} GB/s",
                    per * 1e3,
                    4.0 * wlen as f64 / per / 1e9
                );
            }
        }
        std::env::remove_var("SOFUU_Q4K_COALESCED");
        // single-row coalesced GEMV vs the v1 server (the serial client's
        // actual op shape — the b4 kernels above can't serve it)
        {
            let x1s: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32) * 0.25 - 1.5).collect();
            unsafe {
                std::ptr::copy_nonoverlapping(x1s.as_ptr() as *const u8, slab.as_mut_ptr(), cols * 4);
            }
            for iters in [2usize, 8] {
                let t0 = Instant::now();
                for _ in 0..iters {
                    assert!(engine_c.batch_open());
                    engine_c.batch_gguf(12, &wb, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols);
                    engine_c.batch_end();
                }
                let per = t0.elapsed().as_secs_f64() / iters as f64;
                if iters == 8 {
                    println!(
                        "[q4k coalesced] c-single per_op={:.3}ms bw={:.1} GB/s",
                        per * 1e3,
                        wlen as f64 / per / 1e9
                    );
                }
            }
        }
        // ── q6_K coalesced (the lm_head class): parity on random blobs then
        // the REAL head shape as the bench (151936 x 4096 = ~514MB) ──
        for (q6rows, label) in [(8usize, "parity-8"), (151936usize, "head-bench")] {
            let q6cols = 4096usize;
            let q6blocks = q6cols / 256;
            let q6row = q6blocks * 210;
            let mut w6 = vec![0u8; q6rows * q6row];
            for r in 0..q6rows {
                for b in 0..q6blocks {
                    let off = r * q6row + b * 210;
                    for j in 0..128 {
                        w6[off + j] = (rand() % 255) as u8; // ql
                    }
                    for j in 0..64 {
                        w6[off + 128 + j] = (rand() % 255) as u8; // qh
                    }
                    for j in 0..16 {
                        w6[off + 192 + j] = (rand() % 255) as i8 as u8; // int8 scales
                    }
                    w6[off + 208..off + 210].copy_from_slice(&f16_bytes(0.0625f32));
                }
            }
            let tok6: Vec<f32> = (0..q6cols).map(|i| ((i * 5 + 7) % 19) as f32 * 0.25 - 2.25).collect();
            if label == "parity-8" {
                let mut y6 = vec![0f32; q6rows];
                for r in 0..q6rows {
                    y6[r] = unsafe {
                        crate::infer::quant::vec_dot_raw_q6_k(w6[r * q6row..].as_ptr(), tok6.as_ptr(), q6blocks)
                    };
                }
                let wb6 = engine_c.register_zerocopy_buffer(&w6).expect("w6");
                let page6 = 16384usize;
                let sl6n = q6cols * 4 + q6rows * 4 + page6;
                let al6 = ((sl6n + page6 - 1) / page6) * page6;
                let p6 = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        al6,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                assert!(p6 != libc::MAP_FAILED);
                let sl6: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(p6 as *mut u8, al6) };
                unsafe {
                    std::ptr::copy_nonoverlapping(tok6.as_ptr() as *const u8, sl6.as_mut_ptr(), q6cols * 4)
                };
                let yoff6 = (q6cols * 4 + 63) & !63;
                let sb6 = engine_c.register_zerocopy_buffer(sl6).expect("slab6");
                assert!(engine_c.batch_open());
                engine_c.batch_gguf(14, &wb6, 0, &sb6, 0, &sb6, yoff6, q6rows, q6cols);
                engine_c.batch_end();
                let v6 = unsafe { std::slice::from_raw_parts(sl6.as_ptr() as *const f32, al6 / 4) };
                let yf6 = yoff6 / 4;
                let mut rel = 0f64;
                for r in 0..q6rows {
                    rel = rel.max((v6[yf6 + r] as f64 - y6[r] as f64).abs() / y6[r].abs().max(1e-3) as f64);
                }
                println!("[q6k coalesced] single max_rel={rel:.3e}");
                assert!(rel < 1e-2, "q6_K coalesced vs oracle {rel}");
                let _ = (wb6, sb6);
            } else {
                // head-shape bench: coalesced engine vs fresh v1 engine
                std::env::set_var("SOFUU_Q4K_COALESCED", "1");
                let e6b = MetalEngine::new().expect("engine q6b");
                std::env::remove_var("SOFUU_Q4K_COALESCED");
                let page6 = 16384usize;
                let sl6n = q6cols * 4 + q6rows * 4 + page6;
                let al6 = ((sl6n + page6 - 1) / page6) * page6;
                let p6 = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        al6,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                assert!(p6 != libc::MAP_FAILED);
                let sl6: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(p6 as *mut u8, al6) };
                unsafe {
                    std::ptr::copy_nonoverlapping(tok6.as_ptr() as *const u8, sl6.as_mut_ptr(), q6cols * 4)
                };
                let yoff6 = (q6cols * 4 + 63) & !63;
                let _ = engine_c.register_zerocopy_buffer(&w6);
                let wb6 = e6b.register_zerocopy_buffer(&w6).expect("w6b");
                let sb6 = e6b.register_zerocopy_buffer(sl6).expect("sl6b");
                for iters in [2usize, 4] {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        assert!(e6b.batch_open());
                        e6b.batch_gguf(14, &wb6, 0, &sb6, 0, &sb6, yoff6, q6rows, q6cols);
                        e6b.batch_end();
                    }
                    let per = t0.elapsed().as_secs_f64() / iters as f64;
                    if iters == 4 {
                        println!(
                            "[q6k coalesced] head {} rows per_op={:.3}ms bw={:.1} GB/s",
                            q6rows,
                            per * 1e3,
                            (q6rows * q6row) as f64 / per / 1e9
                        );
                    }
                }
            }
        }
        // ── 2x2 matrix: kernel level (1 byte-lane, 2 vec4) x buffer
        // (compact vs deep-in-5GB) — the pass uses level 2 on a 5GB GGUF
        // mmap; the original receipts were level 1 on compact buffers.
        for (lvl, bufcase) in [(1usize, "compact"), (1, "5G"), (2, "compact"), (2, "5G")] {
            std::env::set_var("SOFUU_Q4K_COALESCED", lvl.to_string());
            let e = MetalEngine::new().expect("matrix engine");
            std::env::remove_var("SOFUU_Q4K_COALESCED");
            let wbb = e.register_zerocopy_buffer(&w).expect("w");
            let sbb = e.register_zerocopy_buffer(slab).expect("slab");
            let mut woff = 0usize;
            let mut big_cleanup: Option<(usize, usize)> = None;
            if bufcase == "5G" {
                let gb5 = 5usize << 30;
                let deep = gb5 - w.len() - 64 * 1024 * 1024;
                let big_ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        gb5,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                        -1,
                        0,
                    )
                };
                assert!(big_ptr != libc::MAP_FAILED);
                unsafe {
                    std::ptr::copy_nonoverlapping(w.as_ptr(), (big_ptr as *mut u8).add(deep), w.len());
                }
                let big_slice: &[u8] =
                    unsafe { std::slice::from_raw_parts(big_ptr as *const u8, gb5) };
                woff = deep;
                let _ = e.register_zerocopy_buffer(big_slice);
                big_cleanup = Some((deep, gb5));
            }
            for iters in [2usize, 6] {
                let t0 = Instant::now();
                for _ in 0..iters {
                    assert!(e.batch_open());
                    e.batch_gguf_b4(&wbb, woff, &sbb, 0, &sbb, y_off, rows, cols);
                    e.batch_end();
                }
                let per = t0.elapsed().as_secs_f64() / iters as f64;
                if iters == 6 {
                    println!(
                        "[q4k matrix] level={} {:8} per_group={:.3}ms eff_bw={:.1} GB/s",
                        lvl,
                        bufcase,
                        per * 1e3,
                        4.0 * wlen as f64 / per / 1e9
                    );
                }
            }
            drop(big_cleanup);
        }
        // 5GB-mmap variant: the CLIENT's weights live in one huge GGUF mmap —
        // does the 16B-span vec4 kernel keep its rate at deep offsets?
        {
            let gb5 = 5usize << 30;
            let deep = gb5 - w.len() - 64 * 1024 * 1024;
            let big_ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    gb5,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            assert!(big_ptr != libc::MAP_FAILED);
            unsafe {
                std::ptr::copy_nonoverlapping(w.as_ptr(), (big_ptr as *mut u8).add(deep), w.len());
            }
            let big_slice: &[u8] = unsafe { std::slice::from_raw_parts(big_ptr as *const u8, gb5) };
            let big_buf = engine_c.register_zerocopy_buffer(big_slice).expect("big");
            for iters in [2usize, 8] {
                let t0 = Instant::now();
                for _ in 0..iters {
                    assert!(engine_c.batch_open());
                    engine_c
                        .batch_gguf_b4(&big_buf, deep, &slab_buf, 0, &slab_buf, y_off, rows, cols);
                    engine_c.batch_end();
                }
                let per = t0.elapsed().as_secs_f64() / iters as f64;
                if iters == 8 {
                    println!(
                        "[q4k coalesced] b4c2-in-5G per_group={:.3}ms eff_bw={:.1} GB/s",
                        per * 1e3,
                        4.0 * wlen as f64 / per / 1e9
                    );
                }
            }
        }
        let engine_v1 = MetalEngine::new().expect("metal engine (v1)");
        let wb1 = engine_v1.register_zerocopy_buffer(&w).expect("w");
        let sb1 = engine_v1.register_zerocopy_buffer(slab).expect("slab");
        for iters in [2usize, 8] {
            let t0 = Instant::now();
            for _ in 0..iters {
                assert!(engine_v1.batch_open());
                for _t in 0..4 {
                    engine_v1.batch_gguf(12, &wb1, 0, &sb1, 0, &sb1, y_off, rows, cols);
                }
                engine_v1.batch_end();
            }
            let per = t0.elapsed().as_secs_f64() / iters as f64;
            if iters == 8 {
                println!(
                    "[q4k coalesced] v1x4 per_group={:.3}ms eff_bw={:.1} GB/s",
                    per * 1e3,
                    4.0 * wlen as f64 / per / 1e9
                );
            }
        }
        // v1-class q6_K head bench (same shape as the coalesced head above)
        {
            let q6cols = 4096usize;
            let q6blocks = q6cols / 256;
            let q6row = q6blocks * 210;
            let mut state = 0xC0FFEEu64;
            let mut rand = move || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as u32
            };
            let mut w6 = vec![0u8; 151936 * q6row];
            for chunk in w6.chunks_mut(4096) {
                for b in chunk.iter_mut() {
                    *b = (rand() % 255) as u8;
                }
            }
            let tok6: Vec<f32> = (0..q6cols).map(|i| ((i * 5 + 7) % 19) as f32 * 0.25 - 2.25).collect();
            let page6 = 16384usize;
            let sl6n = q6cols * 4 + 151936 * 4 + page6;
            let al6 = ((sl6n + page6 - 1) / page6) * page6;
            let p6 = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    al6,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(p6 != libc::MAP_FAILED);
            let sl6: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(p6 as *mut u8, al6) };
            unsafe {
                std::ptr::copy_nonoverlapping(tok6.as_ptr() as *const u8, sl6.as_mut_ptr(), q6cols * 4)
            };
            let yoff6 = (q6cols * 4 + 63) & !63;
            let wb6 = engine_v1.register_zerocopy_buffer(&w6).expect("w6v1");
            let sb6 = engine_v1.register_zerocopy_buffer(sl6).expect("sl6v1");
            for iters in [1usize, 3] {
                let t0 = Instant::now();
                for _ in 0..iters {
                    assert!(engine_v1.batch_open());
                    engine_v1.batch_gguf(14, &wb6, 0, &sb6, 0, &sb6, yoff6, 151936, q6cols);
                    engine_v1.batch_end();
                }
                let per = t0.elapsed().as_secs_f64() / iters as f64;
                if iters == 3 {
                    println!(
                        "[q6k coalesced] v1 head per_op={:.3}ms bw={:.1} GB/s",
                        per * 1e3,
                        (151936 * q6row) as f64 / per / 1e9
                    );
                }
            }
        }
    }

    /// Standalone-vs-interleaved b4c2 A/B: the spec pass's FFN GEMVs run
    /// 16-24 GB/s while the same kernel standalone runs 93-104. Escalating
    /// contexts, same engine/buffers:
    ///   alone -> x36 back-to-back -> +glue interleave -> +b4 attention
    ///   interleave. Each step names the culprit (or absolves it).
    ///
    /// Run: cargo test -p sofuu-core --lib bench_interleaved_gemv -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_interleaved_gemv() {
        use std::time::Instant;
        std::env::set_var("SOFUU_Q4K_COALESCED", "2");
        let engine = MetalEngine::new().expect("metal engine");
        std::env::remove_var("SOFUU_Q4K_COALESCED");

        let rows = 12288usize; // FFN out, cols = dim
        let dim = 4096usize;
        let q_rows = 4096usize;
        let kv_rows = 1024usize;
        let hd = 128usize;
        let n_heads = 32usize;
        let n_kv = 8usize;
        let b = 4usize;
        let layers = 36usize;
        let blocks = dim / 256;
        let wlen = rows * blocks * 144;

        let mut state = 0x1234ABCDu64;
        let w: Vec<u8> = {
            let mut v = vec![0u8; wlen];
            for x in v.iter_mut() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *x = (state >> 33) as u8;
            }
            v
        };

        // one slab: x | xn | gate+up | qa | ka | va | ao (all [B][..]) + pos
        let f32s = b * (dim * 2 + 12288 * 2 + q_rows + kv_rows * 2 + q_rows) + 256;
        let page = 16384usize;
        let alloc = ((f32s * 4 + page - 1) / page) * page;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                alloc,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(ptr != libc::MAP_FAILED);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, alloc) };
        let w_buf = engine.register_zerocopy_buffer(&w).expect("w");
        let slab_buf = engine.register_zerocopy_buffer(slab).expect("slab");

        let kc_bytes = layers * n_kv * 4096 * hd * 4;
        let kc_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                kc_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert!(kc_ptr != libc::MAP_FAILED);
        let vc_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                kc_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert!(vc_ptr != libc::MAP_FAILED);
        let kc: &[u8] = unsafe { std::slice::from_raw_parts(kc_ptr as *const u8, kc_bytes) };
        let vc: &[u8] = unsafe { std::slice::from_raw_parts(vc_ptr as *const u8, kc_bytes) };
        let kc_buf = engine.register_zerocopy_buffer(kc).expect("kc");
        let vc_buf = engine.register_zerocopy_buffer(vc).expect("vc");

        let mut nw = vec![0u8; 16384];
        unsafe {
            std::ptr::copy_nonoverlapping(
                [1.0f32; 4096].as_ptr() as *const u8,
                nw.as_mut_ptr(),
                4096 * 4,
            );
        }
        let nw_buf = engine.register_zerocopy_buffer(&nw).expect("nw");
        let pos: Vec<u32> = (0..16).map(|t| 10 + t as u32).collect();
        let pos_bytes: Vec<u8> = pos.iter().flat_map(|v| v.to_le_bytes()).collect();
        let pos_buf = engine.register_zerocopy_buffer(&pos_bytes).expect("pos");

        let n_chunks_max = 1usize; // positions ~10 -> 1 chunk of 128
        let meta_stride = n_heads * n_chunks_max * 2 + n_heads * n_chunks_max * hd;
        let meta_alloc = ((meta_stride * b * 4) + page - 1) / page * page;
        let m_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                meta_alloc,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(m_ptr != libc::MAP_FAILED);
        let mslice: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(m_ptr as *mut u8, meta_alloc) };
        let meta_buf = engine.register_zerocopy_buffer(mslice).expect("meta");

        // slot offsets (floats)
        let ox = 0usize;
        let oxn = ox + b * dim;
        let og = oxn + b * dim;
        let ou = og + b * 12288;
        let oqa = ou + b * 12288;
        let oka = oqa + b * q_rows;
        let ova = oka + b * kv_rows;
        let oao = ova + b * kv_rows;

        let yoff = oxn * 4; // GEMV y target = xn region (any writable region)
        let gemv_alone = |iters: usize| -> f64 {
            let t0 = Instant::now();
            for _ in 0..iters {
                assert!(engine.batch_open());
                engine.batch_gguf_b4(&w_buf, 0, &slab_buf, 0, &slab_buf, yoff, rows, dim);
                engine.batch_end();
            }
            t0.elapsed().as_secs_f64() / iters as f64
        };
        let warm = gemv_alone(2);
        let alone = gemv_alone(8);
        println!(
            "[interleave] alone    per_op={:.3}ms bw={:.1} GB/s (warm {:.3}ms)",
            alone * 1e3,
            wlen as f64 / alone / 1e9,
            warm * 1e3
        );

        {
            let t0 = Instant::now();
            assert!(engine.batch_open());
            for _ in 0..36 {
                engine.batch_gguf_b4(&w_buf, 0, &slab_buf, 0, &slab_buf, yoff, rows, dim);
            }
            engine.batch_end();
            let per = t0.elapsed().as_secs_f64() / 36.0;
            println!("[interleave] x36      per_op={:.3}ms bw={:.1} GB/s", per * 1e3, wlen as f64 / per / 1e9);
        }

        {
            let t0 = Instant::now();
            assert!(engine.batch_open());
            for _ in 0..36 {
                engine.batch_rmsnorm_f32w_b4(
                    &slab_buf, ox * 4, &nw_buf, 0, &slab_buf, oxn * 4, dim, 1e-5, b,
                );
                engine.batch_gguf_b4(&w_buf, 0, &slab_buf, oxn * 4, &slab_buf, yoff, rows, dim);
                engine.batch_add(&slab_buf, ox * 4, &slab_buf, yoff, b * dim);
            }
            engine.batch_end();
            let per = t0.elapsed().as_secs_f64() / 36.0;
            println!("[interleave] +glue    per_layer={:.3}ms (gemv alone {:.3}ms)", per * 1e3, alone * 1e3);
        }

        {
            let t0 = Instant::now();
            assert!(engine.batch_open());
            for li in 0..8 {
                let layer_off = li * n_kv * 4096 * hd * 4;
                engine.batch_llama_kv_write_b4(
                    &slab_buf, oka * 4, &slab_buf, ova * 4,
                    &kc_buf, layer_off, &vc_buf, layer_off,
                    Some(&nw_buf), 0, 1,
                    &pos_buf, 0, b,
                    hd as u32, n_kv as u32, 4096,
                    1e-5, 10000.0, 0,
                None,
                );
                engine.batch_head_rmsnorm_b4(
                    &slab_buf, oqa * 4, Some(&nw_buf), 0,
                    &pos_buf, 0, b,
                    hd as u32, n_heads as u32, 1, 1e-5, 10000.0, 0,
                None,
                );
                engine.batch_llama_attn_part_b4(
                    &slab_buf, oqa * 4, &kc_buf, layer_off, &vc_buf, layer_off,
                    &meta_buf, 0, &pos_buf, 0, b,
                    hd as u32, n_heads as u32, n_kv as u32, 4096,
                    n_chunks_max as u32, meta_stride as u32,
                );
                engine.batch_llama_attn_fin_b4(
                    &meta_buf, 0, &slab_buf, oao * 4, &pos_buf, 0, b,
                    n_heads as u32, hd as u32, n_chunks_max as u32, meta_stride as u32,
                );
                engine.batch_gguf_b4(&w_buf, 0, &slab_buf, oxn * 4, &slab_buf, yoff, rows, dim);
                engine.batch_add(&slab_buf, ox * 4, &slab_buf, yoff, b * dim);
            }
            engine.batch_end();
            let per = t0.elapsed().as_secs_f64() / 8.0;
            println!("[interleave] +b4attn  per_layer={:.3}ms (gemv alone {:.3}ms)", per * 1e3, alone * 1e3);
        }
        // which attention op carries the cost? one op added per sub-case
        for which in ["kv4", "part4", "kvp"] {
            let t0 = Instant::now();
            let layer_off = 0usize;
            assert!(engine.batch_open());
            for _ in 0..8 {
                engine.batch_gguf_b4(&w_buf, 0, &slab_buf, oxn * 4, &slab_buf, yoff, rows, dim);
                match which {
                    "kv4" => {
                        for _ in 0..4 {
                            engine.batch_llama_kv_write_b4(
                                &slab_buf, oka * 4, &slab_buf, ova * 4,
                                &kc_buf, layer_off, &vc_buf, layer_off,
                                Some(&nw_buf), 0, 1,
                                &pos_buf, 0, b,
                                hd as u32, n_kv as u32, 4096,
                                1e-5, 10000.0, 0,
                            None,
                            );
                        }
                    }
                    "part4" => {
                        for _ in 0..4 {
                            engine.batch_llama_attn_part_b4(
                                &slab_buf, oqa * 4, &kc_buf, layer_off, &vc_buf, layer_off,
                                &meta_buf, 0, &pos_buf, 0, b,
                                hd as u32, n_heads as u32, n_kv as u32, 4096,
                                n_chunks_max as u32, meta_stride as u32,
                            );
                        }
                    }
                    "kvp" => {
                        engine.batch_llama_kv_write_b4(
                            &slab_buf, oka * 4, &slab_buf, ova * 4,
                            &kc_buf, layer_off, &vc_buf, layer_off,
                            Some(&nw_buf), 0, 1,
                            &pos_buf, 0, b,
                            hd as u32, n_kv as u32, 4096,
                            1e-5, 10000.0, 0,
                        None,
                        );
                        engine.batch_llama_attn_part_b4(
                            &slab_buf, oqa * 4, &kc_buf, layer_off, &vc_buf, layer_off,
                            &meta_buf, 0, &pos_buf, 0, b,
                            hd as u32, n_heads as u32, n_kv as u32, 4096,
                            n_chunks_max as u32, meta_stride as u32,
                        );
                    }
                    "pf" => {
                        engine.batch_llama_attn_part_b4(
                            &slab_buf, oqa * 4, &kc_buf, layer_off, &vc_buf, layer_off,
                            &meta_buf, 0, &pos_buf, 0, b,
                            hd as u32, n_heads as u32, n_kv as u32, 4096,
                            n_chunks_max as u32, meta_stride as u32,
                        );
                        engine.batch_llama_attn_fin_b4(
                            &meta_buf, 0, &slab_buf, oao * 4, &pos_buf, 0, b,
                            n_heads as u32, hd as u32, n_chunks_max as u32, meta_stride as u32,
                        );
                    }
                    "kvqn" => {
                        engine.batch_llama_kv_write_b4(
                            &slab_buf, oka * 4, &slab_buf, ova * 4,
                            &kc_buf, layer_off, &vc_buf, layer_off,
                            Some(&nw_buf), 0, 1,
                            &pos_buf, 0, b,
                            hd as u32, n_kv as u32, 4096,
                            1e-5, 10000.0, 0,
                        None,
                        );
                        engine.batch_head_rmsnorm_b4(
                            &slab_buf, oqa * 4, Some(&nw_buf), 0,
                            &pos_buf, 0, b,
                            hd as u32, n_heads as u32, 1, 1e-5, 10000.0, 0,
                        None,
                        );
                    }
                    "kv" => {
                        engine.batch_llama_kv_write_b4(
                            &slab_buf, oka * 4, &slab_buf, ova * 4,
                            &kc_buf, layer_off, &vc_buf, layer_off,
                            Some(&nw_buf), 0, 1,
                            &pos_buf, 0, b,
                            hd as u32, n_kv as u32, 4096,
                            1e-5, 10000.0, 0,
                        None,
                        );
                    }
                    "qn" => {
                        engine.batch_head_rmsnorm_b4(
                            &slab_buf, oqa * 4, Some(&nw_buf), 0,
                            &pos_buf, 0, b,
                            hd as u32, n_heads as u32, 1, 1e-5, 10000.0, 0,
                        None,
                        );
                    }
                    "part" => {
                        engine.batch_llama_attn_part_b4(
                            &slab_buf, oqa * 4, &kc_buf, layer_off, &vc_buf, layer_off,
                            &meta_buf, 0, &pos_buf, 0, b,
                            hd as u32, n_heads as u32, n_kv as u32, 4096,
                            n_chunks_max as u32, meta_stride as u32,
                        );
                    }
                    _ => {
                        engine.batch_llama_attn_fin_b4(
                            &meta_buf, 0, &slab_buf, oao * 4, &pos_buf, 0, b,
                            n_heads as u32, hd as u32, n_chunks_max as u32, meta_stride as u32,
                        );
                    }
                }
                engine.batch_add(&slab_buf, ox * 4, &slab_buf, yoff, b * dim);
            }
            engine.batch_end();
            let per = t0.elapsed().as_secs_f64() / 8.0;
            println!("[interleave] op+{}    per_layer={:.3}ms (gemv alone {:.3}ms)", which, per * 1e3, alone * 1e3);
        }
        println!("[interleave] done");
    }

    /// MULTI-ROW acceptance: the b4 attention family (kv_write_b4 ->
    /// head_rmsnorm_b4 -> attn_part_b4 -> attn_fin_b4, per-row positions from
    /// the buffer) must equal a per-row CPU reference — each row's softmax
    /// bounded by ITS position (chain causality) even though all rows' KV
    /// landed in the cache before any attention ran. The rows' cache slots
    /// are pre-filled with garbage so a missing/late kv_write fails loudly.
    #[test]
    fn llama_attn_b4_matches_cpu_reference() {
        use crate::infer::quant::{batched_rmsnorm_in_place, rope_style};

        let engine = MetalEngine::new().expect("metal engine");
        let n_heads = 4usize;
        let n_kv = 2usize;
        let hd = 32usize;
        let max_seq = 64usize;
        let b = 4usize;
        let p0 = 5usize; // rows at positions 5..8
        let eps = 1e-5f32;
        let rope_base = 10000.0f32;
        let rope_mode = 0u32;

        // raw per-row q/k/v inputs (pre-norm, pre-rope)
        let q_raw: Vec<f32> = (0..b * n_heads * hd).map(|i| ((i % 9) as f32) * 0.25 - 1.0).collect();
        let k_raw: Vec<f32> = (0..b * n_kv * hd).map(|i| ((i % 7) as f32) * 0.5 - 1.5).collect();
        let v_raw: Vec<f32> = (0..b * n_kv * hd).map(|i| ((i % 11) as f32) * 0.125 - 0.5).collect();
        let q_ln: Vec<f32> = (0..hd).map(|i| 1.0 + (i % 5) as f32 * 0.05).collect();
        let k_ln: Vec<f32> = (0..hd).map(|i| 1.0 + (i % 3) as f32 * 0.07).collect();

        // prefix history (positions 0..p0) + garbage in the rows' own slots
        let mut k_cache = vec![0f32; n_kv * max_seq * hd];
        let mut v_cache = vec![0f32; n_kv * max_seq * hd];
        for t in 0..p0 {
            for kv in 0..n_kv {
                let off = (kv * max_seq + t) * hd;
                for i in 0..hd {
                    let s = (t * 31 + i * 7 + kv) % 100;
                    k_cache[off + i] = (s as f32) * 0.02 - 1.0;
                    v_cache[off + i] = ((s * 3 % 50) as f32) * 0.02 - 0.5;
                }
            }
        }
        for t in p0..p0 + b {
            for kv in 0..n_kv {
                let off = (kv * max_seq + t) * hd;
                for i in 0..hd {
                    k_cache[off + i] = 7777.0f32; // garbage: must be overwritten
                    v_cache[off + i] = -8888.0f32;
                }
            }
        }

        // CPU reference: per row — norm+rope k/v, write cache, norm+rope q,
        // softmax over 0..=pos, weighted V
        let mut q_ref = q_raw.clone();
        for t in 0..b {
            let q_row = &mut q_ref[t * n_heads * hd..(t + 1) * n_heads * hd];
            batched_rmsnorm_in_place(q_row, &q_ln, n_heads, hd, eps);
            rope_style(q_row, n_heads, hd, p0 + t, rope_base, rope_mode as i32);
            let k_row = &mut k_raw[t * n_kv * hd..(t + 1) * n_kv * hd].to_vec();
            batched_rmsnorm_in_place(k_row, &k_ln, n_kv, hd, eps);
            rope_style(k_row, n_kv, hd, p0 + t, rope_base, rope_mode as i32);
            for kv in 0..n_kv {
                let off = (kv * max_seq + p0 + t) * hd;
                k_cache[off..off + hd].copy_from_slice(&k_row[kv * hd..(kv + 1) * hd]);
                v_cache[off..off + hd].copy_from_slice(&v_raw[t * n_kv * hd + kv * hd..t * n_kv * hd + (kv + 1) * hd]);
            }
        }
        let mut head_ref = vec![0f32; b * n_heads * hd];
        for t in 0..b {
            let pos = p0 + t;
            let scale = 1.0f32 / (hd as f32).sqrt();
            for h in 0..n_heads {
                let kv = h / (n_heads / n_kv);
                let qh = &q_ref[t * n_heads * hd + h * hd..t * n_heads * hd + (h + 1) * hd];
                let seq = pos + 1;
                let mut scores = vec![0f32; seq];
                let mut mx = f32::NEG_INFINITY;
                for tt in 0..seq {
                    let base = (kv * max_seq + tt) * hd;
                    let mut dot = 0f32;
                    for i in 0..hd {
                        dot += qh[i] * k_cache[base + i];
                    }
                    scores[tt] = dot * scale;
                    mx = mx.max(scores[tt]);
                }
                let mut sum = 0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - mx).exp();
                    sum += *sc;
                }
                for sc in scores.iter_mut() {
                    *sc /= sum;
                }
                for d in 0..hd {
                    let mut acc = 0f32;
                    for tt in 0..seq {
                        acc += scores[tt] * v_cache[(kv * max_seq + tt) * hd + d];
                    }
                    head_ref[t * n_heads * hd + h * hd + d] = acc;
                }
            }
        }

        // GPU: b4 attention family
        let page = 16384usize;
        let q_off = 0usize;
        let k_off = q_off + (((b * n_heads * hd * 4) + page - 1) / page) * page;
        let v_off = k_off + (((b * n_kv * hd * 4) + page - 1) / page) * page;
        let ao_off = v_off + (((b * n_kv * hd * 4) + page - 1) / page) * page;
        let pos_off = ao_off + (((b * n_heads * hd * 4) + page - 1) / page) * page;
        let slab_len = pos_off + page;
        let alloc = ((slab_len + page - 1) / page) * page;
        let p_slab = unsafe {
            libc::mmap(std::ptr::null_mut(), alloc, libc::PROT_READ | libc::PROT_WRITE,
                       libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(p_slab != libc::MAP_FAILED);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(p_slab as *mut u8, alloc) };
        unsafe { std::ptr::copy_nonoverlapping(q_raw.as_ptr() as *const u8, slab.as_mut_ptr().add(q_off), q_raw.len() * 4); }
        unsafe { std::ptr::copy_nonoverlapping(k_raw.as_ptr() as *const u8, slab.as_mut_ptr().add(k_off), k_raw.len() * 4); }
        unsafe { std::ptr::copy_nonoverlapping(v_raw.as_ptr() as *const u8, slab.as_mut_ptr().add(v_off), v_raw.len() * 4); }
        let pos_u32: Vec<u32> = (0..16).map(|t| (p0 + t) as u32).collect();
        unsafe { std::ptr::copy_nonoverlapping(pos_u32.as_ptr() as *const u8, slab.as_mut_ptr().add(pos_off), 16 * 4); }

        let mut cache_bytes = vec![0u8; k_cache.len() * 4];
        unsafe {
            std::ptr::copy_nonoverlapping(k_cache.as_ptr() as *const u8, cache_bytes.as_mut_ptr(), cache_bytes.len());
        }
        let mut vcache_bytes = vec![0u8; v_cache.len() * 4];
        unsafe {
            std::ptr::copy_nonoverlapping(v_cache.as_ptr() as *const u8, vcache_bytes.as_mut_ptr(), vcache_bytes.len());
        }
        // k/v caches must be page-aligned anon (engine registers them)
        let kc_ptr = unsafe { libc::mmap(std::ptr::null_mut(), ((k_cache.len() * 4 + page - 1) / page) * page, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
        assert!(kc_ptr != libc::MAP_FAILED);
        unsafe { std::ptr::copy_nonoverlapping(k_cache.as_ptr() as *const u8, kc_ptr as *mut u8, k_cache.len() * 4); }
        let kc_slice: &[u8] = unsafe { std::slice::from_raw_parts(kc_ptr as *const u8, ((k_cache.len() * 4 + page - 1) / page) * page) };
        let vc_ptr = unsafe { libc::mmap(std::ptr::null_mut(), ((v_cache.len() * 4 + page - 1) / page) * page, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
        assert!(vc_ptr != libc::MAP_FAILED);
        unsafe { std::ptr::copy_nonoverlapping(v_cache.as_ptr() as *const u8, vc_ptr as *mut u8, v_cache.len() * 4); }
        let vc_slice: &[u8] = unsafe { std::slice::from_raw_parts(vc_ptr as *const u8, ((v_cache.len() * 4 + page - 1) / page) * page) };

        // norm w buffer: q_ln@0, k_ln@4096
        let mut wb = vec![0u8; 8192];
        unsafe {
            std::ptr::copy_nonoverlapping(q_ln.as_ptr() as *const u8, wb.as_mut_ptr(), hd * 4);
            std::ptr::copy_nonoverlapping(k_ln.as_ptr() as *const u8, wb.as_mut_ptr().add(4096), hd * 4);
        }

        let slab_buf = engine.register_zerocopy_buffer(slab).expect("slab");
        let wb_buf = engine.register_zerocopy_buffer(&wb).expect("wb");
        let kc_buf = engine.register_zerocopy_buffer(kc_slice).expect("kc");
        let vc_buf = engine.register_zerocopy_buffer(vc_slice).expect("vc");

        let n_chunks_max = (p0 + b + 127) / 128; // 1 at these positions
        let meta_stride = n_heads * n_chunks_max * 2 + n_heads * n_chunks_max * hd;
        let meta_alloc = (((meta_stride * b * 4) + page - 1) / page) * page;
        let m_ptr = unsafe { libc::mmap(std::ptr::null_mut(), meta_alloc, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
        assert!(m_ptr != libc::MAP_FAILED);
        let m_slice: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(m_ptr as *mut u8, meta_alloc) };
        let meta_buf = engine.register_zerocopy_buffer(m_slice).expect("meta");

        assert!(engine.batch_open());
        engine.batch_llama_kv_write_b4(
            &slab_buf, k_off, &slab_buf, v_off,
            &kc_buf, 0, &vc_buf, 0,
            Some(&wb_buf), 4096, 1,
            &slab_buf, pos_off, b,
            hd as u32, n_kv as u32, max_seq as u32,
            eps, rope_base, rope_mode,
        None,
        );
        engine.batch_head_rmsnorm_b4(
            &slab_buf, q_off,
            Some(&wb_buf), 0,
            &slab_buf, pos_off,
            b,
            hd as u32, n_heads as u32, 1, eps, rope_base, rope_mode,
        None,
        );
        engine.batch_llama_attn_part_b4(
            &slab_buf, q_off, &kc_buf, 0, &vc_buf, 0,
            &meta_buf, 0, &slab_buf, pos_off, b,
            hd as u32, n_heads as u32, n_kv as u32, max_seq as u32,
            n_chunks_max as u32, meta_stride as u32,
        );
        engine.batch_llama_attn_fin_b4(
            &meta_buf, 0, &slab_buf, ao_off, &slab_buf, pos_off, b,
            n_heads as u32, hd as u32, n_chunks_max as u32, meta_stride as u32,
        );
        engine.batch_end();

        let view = unsafe { std::slice::from_raw_parts(slab.as_ptr() as *const f32, alloc / 4) };
        let ao_f = ao_off / 4;
        let mut max_err = 0f64;
        for i in 0..b * n_heads * hd {
            max_err = max_err.max((view[ao_f + i] as f64 - head_ref[i] as f64).abs());
        }
        println!("[attn b4] max_err={max_err:.3e}");
        assert!(max_err < 5e-2, "b4 attention vs per-row CPU ref {max_err}");
        // the GPU pass also OVERWROTE the row cache slots through the
        // registered cache buffers — the CPU consumed them identically.
        let _ = (cache_bytes, vcache_bytes);
    }

    /// GPU3 acceptance: chunked attention (head_rmsnorm+rope -> part -> fin)
    /// matches the CPU reference (quant:: fns + plain full-prefix softmax).
    #[test]
    fn llama_attn_chunked_matches_cpu_reference() {
        use crate::infer::quant::{batched_rmsnorm_in_place, rope_style};

        let engine = MetalEngine::new().expect("metal engine");
        let n_heads = 4usize;
        let n_kv = 2usize;
        let hd = 32usize;
        let max_seq = 64usize;
        let pos = 48usize;
        let eps = 1e-5f32;
        let rope_base = 10000.0f32;
        let rope_mode = 0u32;

        let q0: Vec<f32> = (0..n_heads * hd).map(|i| ((i % 9) as f32) * 0.25 - 1.0).collect();
        let k0: Vec<f32> = (0..n_kv * hd).map(|i| ((i % 7) as f32) * 0.5 - 1.5).collect();
        let v0: Vec<f32> = (0..n_kv * hd).map(|i| ((i % 11) as f32) * 0.125 - 0.5).collect();
        let q_ln: Vec<f32> = (0..hd).map(|i| 1.0 + (i % 5) as f32 * 0.05).collect();
        let k_ln: Vec<f32> = (0..hd).map(|i| 1.0 + (i % 3) as f32 * 0.07).collect();

        // CPU reference
        let mut q = q0.clone();
        batched_rmsnorm_in_place(&mut q, &q_ln, n_heads, hd, eps);
        rope_style(&mut q, n_heads, hd, pos, rope_base, rope_mode as i32);
        let mut k = k0.clone();
        batched_rmsnorm_in_place(&mut k, &k_ln, n_kv, hd, eps);
        rope_style(&mut k, n_kv, hd, pos, rope_base, rope_mode as i32);
        let mut k_cache = vec![0f32; n_kv * max_seq * hd];
        let mut v_cache = vec![0f32; n_kv * max_seq * hd];
        for kv in 0..n_kv {
            let off = (kv * max_seq + pos) * hd;
            k_cache[off..off + hd].copy_from_slice(&k[kv * hd..(kv + 1) * hd]);
            v_cache[off..off + hd].copy_from_slice(&v0[kv * hd..(kv + 1) * hd]);
        }
        // history rows: pseudo values at all earlier positions
        for t in 0..pos {
            for kv in 0..n_kv {
                let off = (kv * max_seq + t) * hd;
                for i in 0..hd {
                    let s = (t * 31 + i * 7 + kv) % 100;
                    k_cache[off + i] = (s as f32) * 0.02 - 1.0;
                    v_cache[off + i] = ((s * 3 % 50) as f32) * 0.02 - 0.5;
                }
            }
        }
        let mut head_out = vec![0f32; n_heads * hd];
        let scale = 1.0f32 / (hd as f32).sqrt();
        for h in 0..n_heads {
            let kv = h / (n_heads / n_kv);
            let qh = &q[h * hd..(h + 1) * hd];
            let seq = pos + 1;
            let mut scores = vec![0f32; seq];
            let mut mx = f32::NEG_INFINITY;
            for t in 0..seq {
                let base = (kv * max_seq + t) * hd;
                let mut dot = 0f32;
                for i in 0..hd {
                    dot += qh[i] * k_cache[base + i];
                }
                scores[t] = dot * scale;
                mx = mx.max(scores[t]);
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for d in 0..hd {
                let mut acc = 0f32;
                for t in 0..seq {
                    acc += scores[t] * v_cache[(kv * max_seq + t) * hd + d];
                }
                head_out[h * hd + d] = acc;
            }
        }

        // GPU: head_rmsnorm+rope on qa, part, fin
        let page = 16384usize;
        // wbuf: q_ln@0, k_ln@4096
        let mut wb = vec![0u8; 8192];
        unsafe {
            std::ptr::copy_nonoverlapping(q_ln.as_ptr() as *const u8, wb.as_mut_ptr(), hd * 4);
            std::ptr::copy_nonoverlapping(k_ln.as_ptr() as *const u8, wb.as_mut_ptr().add(4096), hd * 4);
        }
        let wb = engine.register_zerocopy_buffer(&wb).expect("wbuf");

        let mut kc_b = vec![0u8; n_kv * max_seq * hd * 4];
        let mut vc_b = vec![0u8; n_kv * max_seq * hd * 4];
        unsafe {
            std::ptr::copy_nonoverlapping(k_cache.as_ptr() as *const u8, kc_b.as_mut_ptr(), kc_b.len());
            std::ptr::copy_nonoverlapping(v_cache.as_ptr() as *const u8, vc_b.as_mut_ptr(), vc_b.len());
        }
        let kc = engine.register_zerocopy_buffer(&kc_b).expect("kc");
        let vc = engine.register_zerocopy_buffer(&vc_b).expect("vc");

        // act slab: qa@0 ao@q_rows ... slots aligned
        let slab_f32 = 12 * 4096;
        let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), slab_alloc, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(slab_ptr != libc::MAP_FAILED);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr as *mut u8, slab_alloc) };
        unsafe {
            std::slice::from_raw_parts_mut(slab.as_mut_ptr() as *mut f32, slab.len() / 4)
                [..q0.len()].copy_from_slice(&q0);
        }
        let slab_buf = engine.register_zerocopy_buffer(slab).expect("slab");
        let meta_f32 = n_heads * ((max_seq + 127) / 128) * 2 + n_heads * ((max_seq + 127) / 128) * hd;
        let meta_alloc = ((meta_f32 * 4 + page - 1) / page) * page;
        let meta_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), meta_alloc, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(meta_ptr != libc::MAP_FAILED);
        let meta: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(meta_ptr as *mut u8, meta_alloc) };
        let meta_buf = engine.register_zerocopy_buffer(meta).expect("meta");

        let f_ao = 8 * 4096usize; // fresh region
        assert!(engine.batch_open());
        engine.batch_head_rmsnorm(
            &slab_buf, 0,
            Some(&wb), 0,
            hd as u32, n_heads as u32,
            1, pos as u32, eps, rope_base, rope_mode,
        );
        engine.batch_llama_attn_part(
            &slab_buf, 0,
            &kc, 0,
            &vc, 0,
            None, 0,
            &meta_buf, 0,
            0, pos as u32,
            hd as u32, n_heads as u32, n_kv as u32, max_seq as u32,
            eps, rope_base, rope_mode,
        );
        engine.batch_llama_attn_fin(
            &meta_buf, 0,
            &slab_buf, f_ao * 4,
            pos as u32, n_heads as u32, hd as u32,
        );
        engine.batch_end();

        let view = unsafe { std::slice::from_raw_parts(slab.as_ptr() as *const f32, slab_alloc / 4) };
        for i in 0..n_heads * hd {
            let a = view[f_ao + i];
            let b = head_out[i];
            assert!(
                (a - b).abs() <= 2e-3 * (1.0 + b.abs()),
                "chunked attn: head_out[{i}] {a} vs {b}"
            );
        }
        println!("[chunked attn parity] ok: pos={pos} heads={n_heads} kv={n_kv} hd={hd}");
    }

    /// Timing-only micro-bench: one GEMV at the Qwen3-8B FFN shape
    /// (12288x4096 = 28.3MB q4_K weights) — measures the kernel's effective
    /// weight-stream bandwidth, largely insensitive to CPU load storms.
    /// Run: cargo test -p sofuu-core --lib bench_gguf -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_gguf_gemv_bandwidth() {
        use std::time::Instant;
        let engine = MetalEngine::new().expect("metal engine");
        let rows = 12288usize;
        let cols = 4096usize;
        let blocks = cols / 256;
        let wlen = rows * blocks * 144; // q4_K bytes
        let wbuf = vec![0x80u8; wlen];
        let mut x = vec![0f32; cols];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 7) as f32) * 0.25 - 0.75;
        }
        let page = 16384usize;
        let ylen = page;
        let xylen = x.len() * 4 + ylen;
        let alloc = ((xylen + page - 1) / page) * page;
        let slab_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                alloc,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(slab_ptr != libc::MAP_FAILED);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr as *mut u8, alloc) };
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, slab.as_mut_ptr(), x.len() * 4)
        };
        let y_off = (x.len() * 4 + 63) & !63;
        let w_buf = engine.register_zerocopy_buffer(&wbuf).expect("w");
        let slab_buf = engine.register_zerocopy_buffer(slab).expect("slab");

        for iter in 0..5 {
            let iters = if iter == 0 { 2 } else { 6 };
            let t0 = Instant::now();
            for _ in 0..iters {
                assert!(engine.batch_open());
                engine.batch_gguf(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols);
                engine.batch_end();
            }
            let dt = t0.elapsed();
            let per = dt.as_secs_f64() / iters as f64;
            println!(
                "[bench gguf q4_K] {:55} per_op={:.3}ms effective_bw={:.1} GB/s",
                if iter == 0 { "warmup" } else { "measured" },
                per * 1e3,
                wlen as f64 / per / 1e9
            );
        }
    }

    /// G3 economics A/B (#[ignore], run on a calm Mac): does ONE batch_gguf_b4
    /// pass over the FFN shape cost one weight stream or more? Compares the
    /// shipped scalar GEMV (the serial client's cost) against the B=4 kernel
    /// on the identical tensor, same process, same window.
    ///
    /// Run: cargo test -p sofuu-core --lib bench_b4_vs_v1 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_b4_vs_v1() {
        use std::time::Instant;
        let engine = MetalEngine::new().expect("metal engine");
        let rows = 12288usize;
        let cols = 4096usize;
        let blocks = cols / 256;
        let wlen = rows * blocks * 144; // 28.3MB q4_K
        // random bytes: a constant blob is page-compressible on the GPU —
        // constant-byte receipts were inflated ~4-5x (b4c2 "100 GB/s")
        let mut state = 0xBADC0DEu64;
        let wbuf = {
            let mut v = vec![0u8; wlen];
            for b in v.iter_mut() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *b = (state >> 33) as u8;
            }
            v
        };
        // x: [4][cols] token-major
        let x: Vec<f32> = (0..4 * cols)
            .map(|i| ((i % 11) as f32) * 0.25 - 1.25)
            .collect();
        let page = 16384usize;
        let ylen = page;
        let xylen = x.len() * 4 + ylen;
        let alloc = ((xylen + page - 1) / page) * page;
        let slab_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                alloc,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(slab_ptr != libc::MAP_FAILED);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr as *mut u8, alloc) };
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, slab.as_mut_ptr(), x.len() * 4)
        };
        let y_off = (x.len() * 4 + 63) & !63;
        let w_buf = engine.register_zerocopy_buffer(&wbuf).expect("w");
        let slab_buf = engine.register_zerocopy_buffer(slab).expect("slab");

        // v1: four separate tokens = four batch_gguf calls (serial client cost/
        // 4-token group; the x of tokens 1..3 all points at row 0's x — the
        // timing here is about the weight stream, not values).
        for iter in 0..3 {
            let iters = if iter == 0 { 2 } else { 6 };
            let t0 = Instant::now();
            for _ in 0..iters {
                assert!(engine.batch_open());
                for _t in 0..4 {
                    engine.batch_gguf(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols);
                }
                engine.batch_end();
            }
            let per = t0.elapsed().as_secs_f64() / iters as f64;
            println!(
                "[bench b4-vs-v1] v1 x4   per_group={:.3}ms stream_bw={:.1} GB/s",
                per * 1e3,
                4.0 * wlen as f64 / per / 1e9
            );
        }
        // b4: one call, one [4][cols] batch.
        for iter in 0..3 {
            let iters = if iter == 0 { 2 } else { 6 };
            let t0 = Instant::now();
            for _ in 0..iters {
                assert!(engine.batch_open());
                engine.batch_gguf_b4(&w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols);
                engine.batch_end();
            }
            let per = t0.elapsed().as_secs_f64() / iters as f64;
            println!(
                "[bench b4-vs-v1] b4      per_group={:.3}ms eff_bw={:.1} GB/s vs-v1-x4",
                per * 1e3,
                4.0 * wlen as f64 / per / 1e9
            );
        }
    }

    fn run_gpu_ty(ty: u32, wbytes: &[u8], x: &[f32], engine: &MetalEngine) -> Vec<f32> {
        let rows = wbytes.len() / ((x.len() / 256) * 144);
        let slab_len = x.len() * 4 + rows * 4 + 256;
        let page = 16384usize;
        let alloc = ((slab_len + page - 1) / page) * page;
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                alloc,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(p != libc::MAP_FAILED);
        let slab: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(p as *mut u8, alloc) };
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, slab.as_mut_ptr(), x.len() * 4) };
        let y_off = (x.len() * 4 + 63) & !63;
        let w_buf = engine.register_zerocopy_buffer(wbytes).expect("w");
        let slab_buf = engine.register_zerocopy_buffer(slab).expect("slab");
        assert!(engine.batch_open());
        engine.batch_gguf(ty, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, x.len());
        engine.batch_end();
        let f32_view: &[f32] = unsafe { std::slice::from_raw_parts(slab.as_ptr() as *const f32, alloc / 4) };
        f32_view[y_off / 4..y_off / 4 + rows].to_vec()
    }

    #[test]
    fn q4k_crafted() {
        let engine = MetalEngine::new().expect("metal engine");
        let cols = 256usize; // one block
        let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32) * 0.5f32 - 3.0f32).collect();

        // d = 1.0 (f16 0x3C00), everything else zero unless a case sets it.
        let w = |scales: [u8; 12], qs: u8, d: f32, dmin: f32| -> Vec<u8> {
            let mut wbytes = vec![0u8; 144];
            let d16: u16 = if d == 1.0 { 0x3C00 } else { 0x3800 }; // 1.0 / 0.5
            wbytes[0..2].copy_from_slice(&d16.to_le_bytes());
            let m16: u16 = if dmin == 0.0 { 0x0000 } else { 0x3800 };
            wbytes[2..4].copy_from_slice(&m16.to_le_bytes());
            wbytes[4..16].copy_from_slice(&scales);
            for b in 0..128 {
                wbytes[16 + b] = qs;
            }
            wbytes
        };

        let cases: Vec<(&str, Vec<u8>)> = vec![
            (
                "A lo-nibble: sc0=1 all-else-0 qs=0x01 -> values [0..32)=1",
                w([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0x01, 1.0, 0.0),
            ),
            (
                "B hi-nibble: sc1=1 qs=0x10 -> values [32..64)=1",
                w([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0x10, 1.0, 0.0),
            ),
            (
                "C min: sc0=4 m0=2 dmin=0.5 qs=0x01 -> values [0..32)=3",
                w([4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0], 0x01, 1.0, 0.5),
            ),
        ];

        // D: four blocks on lanes 0..3 — cross-block/lane contamination check.
        {
            let cols4 = 1024usize;
            let x4: Vec<f32> = (0..cols4).map(|i| ((i % 13) as f32) * 0.5f32 - 3.0f32).collect();
            let mut wbytes = vec![0u8; 4 * 144];
            for b in 0..4 {
                let off = b * 144;
                wbytes[off..off + 2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1
                wbytes[off] = 0x00;
                wbytes[off] = 0x00;
                wbytes[off + 0] = 0x00;
                wbytes[off + 1] = 0x3C;
                wbytes[off + 4] = 1; // sc(j=0) = 1, everything else 0
                for q in 0..128 {
                    wbytes[off + 16 + q] = 0x01;
                }
            }
            let gpu = run_gpu_ty(12, &wbytes, &x4, &engine);
            let mut deq = vec![0f32; cols4];
            crate::infer::quant::dequantize_q4_k_row(&wbytes, &mut deq);
            let cpu_dot: f32 = deq.iter().zip(x4.iter()).map(|(a, b)| a * b).sum();
            eprintln!("[q4k-crafted] D 4-blocks gpu[0]={:.4} cpu_dot={:.4}", gpu[0], cpu_dot);
            assert!(
                (cpu_dot - gpu[0]).abs() < 1e-2,
                "D: cpu {cpu_dot} vs gpu {}",
                gpu[0]
            );
        }

        // E: the main test's scale-byte generator (random 6-bit + random top
        // 2 bits), one block — exercises the j>=4 composed-scale decode.
        {
            let mut state = 0xABCDEFu64;
            let mut rand = move || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as u32)
            };
            for _ in 0..8 {
                let x1: Vec<f32> = (0..256).map(|_| ((rand() % 100) as f32) * 0.25f32 - 12.0f32).collect();
                let mut wbytes = vec![0u8; 144];
                let d16 = 0x3C00u16; // 1.0
                wbytes[0..2].copy_from_slice(&d16.to_le_bytes());
                for j in 0..12 {
                    wbytes[4 + j] = (rand() & 0x3F) as u8 | ((rand() & 3) << 6) as u8;
                }
                for q in 0..128 {
                    wbytes[16 + q] = (rand() % 255) as u8;
                }
                let gpu = run_gpu_ty(12, &wbytes, &x1, &engine);
                let mut deq = vec![0f32; 256];
                crate::infer::quant::dequantize_q4_k_row(&wbytes, &mut deq);
                let cpu_dot: f32 = deq.iter().zip(x1.iter()).map(|(a, b)| a * b).sum();
                eprintln!("[q4k-crafted] E random gpu[0]={:.4} cpu_dot={:.4}", gpu[0], cpu_dot);
                assert!(
                    (cpu_dot - gpu[0]).abs() / cpu_dot.abs().max(1.0) < 1e-3,
                    "E: cpu {cpu_dot} vs gpu {}",
                    gpu[0]
                );
            }
        }
        {
        } // end of scoped E
        let _ = (cols, cases);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod q6k_crafted_probe {
    use super::*;

    #[test]
    fn q6k_crafted() {
        let engine = MetalEngine::new().expect("metal engine");
        let cols = 256usize;
        let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32) * 0.5f32 - 3.0f32).collect();

        // 210-byte q6_K block: ql[128], qh[64], sc[16], d f16.
        let mut wbytes = vec![0u8; 210];
        // Case A: ql=0x11 (low=1 -> 6-bit 1), qh=0, sc[i]=1, d=1.0
        // CPU value = (1-32)*d*sc = -31 everywhere.
        for q in 0..128 {
            wbytes[q] = 0x11;
        }
        for s in 0..16 {
            wbytes[192 + s] = 1;
        }
        wbytes[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0
        run_one(&engine, &wbytes, &x, "A ql=0x11 sc=1 d=1 -> all -31");

        // Case B: hi-2-bits: ql=0x00, qh=0x55 (val1 bits 1) -> 6-bit = 4 -> 4-32=-28.
        let mut wbytes_b = vec![0u8; 210];
        for q in 0..64 {
            wbytes_b[128 + q] = 0x55;
        }
        for s in 0..16 {
            wbytes_b[192 + s] = 1;
        }
        wbytes_b[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
        run_one(&engine, &wbytes_b, &x, "B qh=0x55 -> all -28");

        // Case C: signed scale: sc[0]=-1 (0xFF) for the first quarter only.
        let mut wbytes_c = vec![0u8; 210];
        for q in 0..128 {
            wbytes_c[q] = 0x11;
        }
        for s in 0..16 {
            wbytes_c[192 + s] = 1;
        }
        wbytes_c[192] = 0xFF;
        wbytes_c[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
        run_one(&engine, &wbytes_c, &x, "C sc[0]=-1 d=1");

        fn run_one(engine: &MetalEngine, wbytes: &[u8], x: &[f32], name: &str) {
            let page = 16384usize;
            let alloc = ((x.len() * 4 + 256 + page - 1) / page) * page;
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    alloc,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED);
            let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, alloc) };
            unsafe {
                std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, slab.as_mut_ptr(), x.len() * 4)
            };
            let y_off = (x.len() * 4 + 63) & !63;
            let w_buf = engine.register_zerocopy_buffer(wbytes).expect("w");
            let slab_buf = engine.register_zerocopy_buffer(slab).expect("slab");
            assert!(engine.batch_open());
            engine.batch_gguf(14, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, 1, x.len());
            engine.batch_end();
            let f32_view: &[f32] =
                unsafe { std::slice::from_raw_parts(slab.as_ptr() as *const f32, alloc / 4) };
            let gpu = f32_view[y_off / 4];
            // CPU: mirror of the intended mapping (trivial since uniform bytes)
            // A: all -31; B: all -28; C: first 16 values scale -1, rest +1.
            let expected = if name.starts_with('A') {
                -31.0f32 * x.iter().sum::<f32>()
            } else if name.starts_with('B') {
                // qh=0x55 -> every 2-bit field = 1 -> 6-bit = 1<<4 = 16 -> q = -16
                -16.0f32 * x.iter().sum::<f32>()
            } else {
                x.iter().enumerate()
                    .map(|(i, xi)| {
                        let s = if i < 16 { -1.0f32 } else { 1.0f32 };
                        (1.0 - 32.0) * s * xi
                    })
                    .sum::<f32>()
            };
            eprintln!("[q6k-crafted] {name} gpu={gpu:.4} expected={expected:.4}");
            assert!(
                (gpu - expected).abs() / 31.0f32.max(1.0) < 0.05,
                "{name}: gpu {gpu} vs expected {expected}"
            );
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod batch_g0_tests {
    use super::*;

    fn bytes_of(v: &mut [f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }

    /// G0 acceptance: N chained GEMV ops in ONE command buffer.
    /// Parity is checked on a 4-op run (pure chains at n=32 overflow into
    /// 1e35 - that is the math of chaining a spectral-radius>1 matrix, not a
    /// bug); the 32-op run is the submission-cost receipt.
    #[test]
    #[ignore = "Metal + timed; run explicitly on a calm Mac"]
    fn batch_gemv_chain_matches_per_op_and_cpu() {
        let engine = MetalEngine::new().expect("metal engine");
        let rows = 2048usize;
        let cols = 512usize;

        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 1000) as u16
        };
        let f32_to_bf16 = |v: f32| -> u16 { ((v.to_bits() + 0x8000) >> 16) as u16 };
        // Small weights keep chained grows in f32 range for 4 ops.
        let wbytes: Vec<u8> = (0..rows * cols)
            .flat_map(|_| f32_to_bf16(((rand() % 1000) as f32 - 500.0) * 0.001).to_le_bytes())
            .collect();
        let x0: Vec<f32> = (0..cols).map(|_| ((rand() % 1000) as f32 - 500.0) * 0.001).collect();
        let w_buf = engine.register_zerocopy_buffer(&wbytes).expect("w buf");
        let n_max = 32usize;

        // --- per-op reference (one staging + wait + y-copy per op) ---
        let run_plain = |n: usize| -> Vec<Vec<f32>> {
            let mut chain: Vec<Vec<f32>> = (0..n + 1).map(|_| vec![0f32; rows]).collect();
            chain[0][..cols].copy_from_slice(&x0);
            for i in 0..n {
                let cur = chain[i][..cols].to_vec();
                engine.gemv_bf16(&w_buf, 0, &cur, &mut chain[i + 1][..rows], rows, cols);
            }
            engine.sync();
            chain
        };

        // --- batched (one submission per group) ---
        // zcopy requires PAGE-ALIGNED host memory: allocate one anonymous
        // mmap slab, register it once, address each slot by byte offset.
        let slot_f32 = rows; // read/write both fit in `rows` f32
        let slab_len = ((n_max + 1) * slot_f32 * 4 + 65535) & !65535;
        let slab = unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                slab_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if p == libc::MAP_FAILED {
                panic!("slab mmap failed");
            }
            std::slice::from_raw_parts_mut(p as *mut u8, slab_len)
        };
        let slab_buf = engine
            .register_zerocopy_buffer(unsafe { std::slice::from_raw_parts(slab.as_ptr(), slab_len) })
            .expect("slab buf");
        let run_batch = |n: usize| -> Vec<Vec<f32>> {
            // SAFETY: slab is MAP_PRIVATE writable memory owned by this test.
            let f32_view: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(slab.as_ptr() as *mut f32, slab_len / 4)
            };
            for v in f32_view.iter_mut() {
                *v = 0.0;
            }
            assert!(engine.batch_open(), "batch open");
            // op0: CPU-staged x; ops 1..n chain on GPU buffers — ALL in ONE
            // commit (dedicated slot per op: op i reads slot i, writes i+1).
            engine.batch_gemv_staged(
                &w_buf, 0, &x0, &slab_buf, slot_f32 * 4, rows, cols, 0,
            );
            for i in 1..n {
                engine.batch_gemv(
                    &w_buf,
                    0,
                    &slab_buf, i * slot_f32 * 4,
                    &slab_buf, (i + 1) * slot_f32 * 4,
                    rows, cols, 0,
                );
            }
            engine.batch_end();
            (0..=n).map(|k| f32_view[k * slot_f32..(k + 1) * slot_f32].to_vec()).collect()
        };

        // triangulate: SAME op sequence but one commit per op (guaranteed
        // cross-op visibility). If this matches per-op, the bug is in-buffer
        // visibility only; if not, bindings.
        {
            // NOTE: Vec f32 not page-aligned -> copying buffers -> CPU-side
            // edits invisible. Use the slab instead with 3 slots.
            let f32_view: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(slab.as_ptr() as *mut f32, slab_len / 4)
            };
            for v in f32_view.iter_mut() { *v = 0.0; }
            f32_view[..cols].copy_from_slice(&x0);
            for i in 0..2 {
                let (xin, yin) = if i % 2 == 0 { (0usize, 1usize) } else { (1usize, 0usize) };
                assert!(engine.batch_open());
                engine.batch_gemv(&w_buf, 0, &slab_buf, xin * slot_f32 * 4, &slab_buf, yin * slot_f32 * 4, rows, cols, 0);
                engine.batch_end();
                eprintln!("[g0-tri] op{} slots01={:?} {:?}", i, &f32_view[..3], &f32_view[slot_f32..slot_f32 + 3]);
            }
        }

        // parity at 4 ops
        let b1early = run_batch(1);
        eprintln!("[g0-order] first-ever batch op0={:?}", &b1early[1][..3]);
        let plain4 = run_plain(4);
        let batch4 = run_batch(4);
        let p1 = run_plain(1); let b1 = run_batch(1);
        eprintln!("[g0-op0] plain={:?} batch={:?} x0={:?}", &p1[1][..3], &b1[1][..3], &x0[..3]);
        let mut max_rel: f64 = 0.0;
        // ping-pong: after k ops the result lives in slot (k odd ? 1 : 0)
        for k in 1..5 {
            let mut rk: f64 = 0.0;
            for (a, b) in batch4[k][..rows].iter().zip(plain4[k][..rows].iter()) {
                let d = (*a - *b).abs() / b.abs().max(1e-2);
                rk = rk.max(d as f64);
            }
            eprintln!("[g0-k] k={k} rel={rk:.2e} plain0={:?} batch0={:?}", plain4[k][0], batch4[k][0]);
            max_rel = max_rel.max(rk);
        }
        eprintln!(
            "[g0] parity@4 max_rel={:.2e} plain1={:?} batch1={:?}",
            max_rel, &plain4[1][..3], &batch4[1][..3]
        );
        assert!(max_rel < 1e-2, "batched vs per-op divergence {max_rel}");

        // submission-cost receipt at 32 ops
        let n = 32usize;
        let t0 = std::time::Instant::now();
        run_plain(n);
        let per_op = t0.elapsed().as_micros() as f64 / n as f64;
        let t1 = std::time::Instant::now();
        run_batch(n);
        let batched = t1.elapsed().as_micros() as f64 / n as f64;
        eprintln!(
            "G0 receipt: per-op {:.0}us/op vs batched {:.0}us/op  ({:.2}x) at N={}",
            per_op, batched, per_op / batched, n
        );
        let _ = (f32_to_bf16, rows, cols);
    }
}

// ── Phase 3: q8_0 / q6_K GEMM oracles (same contract as gemm_q4_k) ─────────
// The 1B is ALL-q8_0 and the 0.6B carries q6_K attn_v; before Phase 3 these
// dtypes rode the per-token GEMV loop in the prefill route (B dispatches per
// tensor per layer). Shapes mirror the real fleet: q8_0 attn_q 2048×2048
// (1B), q6_K attn_v 1024×1024 (0.6B), B=5 parity + B=32/B=128 rates (128
// maxed-tile = the chunk-128 operating point).

#[test]
#[ignore] // GPU bench class: run with `cargo test -- --ignored`
fn gemm_q8_0_matches_cpu() {
    use std::time::Instant;

    let rows = 2048usize; // 1B attn_q
    let blocks = 64usize; // cols 2048
    let cols = blocks * 32;
    let row_bytes = blocks * 34; // [f16 d][32 int8]

    let mut state = 0x2545F4914F6CDD1Du64;
    let mut rand = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let f16_bytes = |v: f32| -> [u8; 2] {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
        let man = (b >> 13) & 0x3FF;
        (sign | (exp << 10) | (man as u16)).to_le_bytes()
    };
    let mut w = vec![0u8; rows * row_bytes];
    for r in 0..rows {
        for b in 0..blocks {
            let off = r * row_bytes + b * 34;
            w[off..off + 2].copy_from_slice(&f16_bytes(0.03125f32));
            for j in 0..32 {
                w[off + 2 + j] = (rand() % 255) as u8;
            }
        }
    }
    for b_case in [5usize, 32usize, 128usize] {
        let B = b_case;
        let toks: Vec<Vec<f32>> = (0..B)
            .map(|t| (0..cols).map(|i| ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5).collect())
            .collect();
        let mut y_cpu = vec![0f32; B * rows];
        for t in 0..B {
            for r in 0..rows {
                y_cpu[t * rows + r] =
                    crate::infer::quant::vec_dot_raw_q8_0(w[r * row_bytes..].as_ptr(), toks[t].as_ptr(), blocks);
            }
        }
        let page = 16384usize;
        let mk = |bytes: usize| -> *mut u8 {
            let p = unsafe {
                libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
            };
            assert!(p != libc::MAP_FAILED);
            p as *mut u8
        };
        let w_alloc = ((w.len() + page - 1) / page) * page;
        let w_ptr = mk(w_alloc);
        let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w.len()) };
        wb.copy_from_slice(&w);
        let engine = MetalEngine::new().expect("engine");
        let w_buf = engine.register_zerocopy_only(wb).expect("w");
        let slab_f32 = B * cols + B * rows;
        let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = mk(slab_alloc);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        for t in 0..B {
            unsafe {
                std::ptr::copy_nonoverlapping(toks[t].as_ptr() as *const u8, slab_ptr.add(t * cols * 4), cols * 4)
            };
        }
        let y_off = B * cols * 4;
        let dispatched = {
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(8, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
            engine.batch_end();
            ok
        };
        assert!(dispatched, "gemm_q8_0 PSO unavailable at B={B} — build the metallib");
        let y_gpu: &[f32] =
            unsafe { std::slice::from_raw_parts(slab_ptr.add(y_off) as *const f32, B * rows) };
        let mut worst = 0f64;
        let (mut wt, mut wr, mut wa, mut wcr) = (0usize, 0usize, 0f32, 0f32);
        for t in 0..B {
            for r in 0..rows {
                let a = y_gpu[t * rows + r];
                let b = y_cpu[t * rows + r];
                let e = (a as f64 - b as f64).abs() / b.abs().max(1.0) as f64;
                if e > worst { worst = e; wt = t; wr = r; wa = a; wcr = b; }
            }
        }
        assert!(worst <= 1e-6, "gemm_q8_0 parity broken at B={B} t={wt} r={wr}: gpu={wa:.6e} cpu={wcr:.6e} (rel {worst:.2e})");
        println!("[gemm_q8_0] B={B} rows={rows} cols={cols} worst_rel={worst:.2e} — PARITY PASS");
        let t0 = Instant::now();
        let reps = 4usize;
        for _ in 0..reps {
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(8, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
            engine.batch_end();
            assert!(ok);
        }
        let per = t0.elapsed() / reps as u32;
        let bytes = rows as f64 * row_bytes as f64;
        println!("[gemm_q8_0] B={B} per-dispatch {:.3?} ms -> {:.1} GB/s weight stream", per.as_secs_f64() * 1e3, bytes / per.as_secs_f64() / 1e9);
        unsafe { libc::munmap(w_ptr as *mut libc::c_void, w_alloc) };
        unsafe { libc::munmap(slab_ptr as *mut libc::c_void, slab_alloc) };
    }
}

#[test]
#[ignore] // GPU bench class: run with `cargo test -- --ignored`
fn gemm_q6_k_matches_cpu() {
    use std::time::Instant;

    let rows = 1024usize; // 0.6B attn_v (kv_rows)
    let blocks = 4usize; // cols 1024
    let cols = blocks * 256;
    let row_bytes = blocks * 210; // ql[128] qh[64] scales[16] f16

    let mut state = 0x9E3779B97F4A7C15u64;
    let mut rand = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let f16_bytes = |v: f32| -> [u8; 2] {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
        let man = (b >> 13) & 0x3FF;
        (sign | (exp << 10) | (man as u16)).to_le_bytes()
    };
    let mut w = vec![0u8; rows * row_bytes];
    for r in 0..rows {
        for b in 0..blocks {
            let off = r * row_bytes + b * 210;
            for j in 0..128 {
                w[off + j] = (rand() % 255) as u8;
            }
            for j in 0..64 {
                w[off + 128 + j] = (rand() % 255) as u8;
            }
            for j in 0..16 {
                w[off + 192 + j] = (rand() % 63) as u8; // signed scales, magnitude-kept
            }
            w[off + 208..off + 210].copy_from_slice(&f16_bytes(0.03125f32));
        }
    }
    for b_case in [5usize, 32usize, 128usize] {
        let B = b_case;
        let toks: Vec<Vec<f32>> = (0..B)
            .map(|t| (0..cols).map(|i| ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5).collect())
            .collect();
        let mut y_cpu = vec![0f32; B * rows];
        for t in 0..B {
            for r in 0..rows {
                y_cpu[t * rows + r] =
                    crate::infer::quant::vec_dot_raw_q6_k(w[r * row_bytes..].as_ptr(), toks[t].as_ptr(), blocks);
            }
        }
        let page = 16384usize;
        let mk = |bytes: usize| -> *mut u8 {
            let p = unsafe {
                libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
            };
            assert!(p != libc::MAP_FAILED);
            p as *mut u8
        };
        let w_alloc = ((w.len() + page - 1) / page) * page;
        let w_ptr = mk(w_alloc);
        let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w.len()) };
        wb.copy_from_slice(&w);
        let engine = MetalEngine::new().expect("engine");
        let w_buf = engine.register_zerocopy_only(wb).expect("w");
        let slab_f32 = B * cols + B * rows;
        let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = mk(slab_alloc);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        for t in 0..B {
            unsafe {
                std::ptr::copy_nonoverlapping(toks[t].as_ptr() as *const u8, slab_ptr.add(t * cols * 4), cols * 4)
            };
        }
        let y_off = B * cols * 4;
        let dispatched = {
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(14, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
            engine.batch_end();
            ok
        };
        assert!(dispatched, "gemm_q6_k PSO unavailable at B={B} — build the metallib");
        let y_gpu: &[f32] =
            unsafe { std::slice::from_raw_parts(slab_ptr.add(y_off) as *const f32, B * rows) };
        let mut worst = 0f64;
        let (mut wt, mut wr, mut wa, mut wcr) = (0usize, 0usize, 0f32, 0f32);
        for t in 0..B {
            for r in 0..rows {
                let a = y_gpu[t * rows + r];
                let b = y_cpu[t * rows + r];
                let e = (a as f64 - b as f64).abs() / b.abs().max(1.0) as f64;
                if e > worst { worst = e; wt = t; wr = r; wa = a; wcr = b; }
            }
        }
        assert!(worst <= 1e-5, "gemm_q6_k parity broken at B={B} t={wt} r={wr}: gpu={wa:.6e} cpu={wcr:.6e} (rel {worst:.2e})");
        println!("[gemm_q6_k] B={B} rows={rows} cols={cols} worst_rel={worst:.2e} — PARITY PASS");
        let t0 = Instant::now();
        let reps = 4usize;
        for _ in 0..reps {
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(14, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
            engine.batch_end();
            assert!(ok);
        }
        let per = t0.elapsed() / reps as u32;
        let bytes = rows as f64 * row_bytes as f64;
        println!("[gemm_q6_k] B={B} per-dispatch {:.3?} ms -> {:.1} GB/s weight stream", per.as_secs_f64() * 1e3, bytes / per.as_secs_f64() / 1e9);
        unsafe { libc::munmap(w_ptr as *mut libc::c_void, w_alloc) };
        unsafe { libc::munmap(slab_ptr as *mut libc::c_void, slab_alloc) };
    }
}

/// P4 arm-d parity + rate: the threadgroup-tiled q4_K GEMM must match the
/// CPU vec_dot reference and beat the shipped 8-token class on the same
/// shape (0.6B gate/up 3072x1024 at B=128 — the prefill route's biggest
/// single tensor). Run:
///   cargo test --release -p sofuu-core -- --ignored --nocapture gemm_q4_k_tiled
#[test]
#[ignore] // GPU bench class
fn gemm_q4_k_tiled_matches_cpu() {
    use std::time::Instant;

    let rows = 3072usize; // 0.6B ffn_gate rows
    let blocks = 4usize; // cols 1024
    let cols = blocks * 256;
    let row_bytes = blocks * 144;
    let B = 128usize;

    let mut state = 0xA24BAED4963EE407u64;
    let mut rand = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let f16_bytes = |v: f32| -> [u8; 2] {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
        let man = (b >> 13) & 0x3FF;
        (sign | (exp << 10) | (man as u16)).to_le_bytes()
    };
    let mut w = vec![0u8; rows * row_bytes];
    for r in 0..rows {
        for b in 0..blocks {
            let off = r * row_bytes + b * 144;
            w[off..off + 2].copy_from_slice(&f16_bytes(0.0625f32));
            w[off + 2..off + 4].copy_from_slice(&f16_bytes(0.0005f32));
            for j in 0..12 {
                w[off + 4 + j] = ((rand() & 0x3F) as u8) | (((rand() & 3) as u8) << 6);
            }
            for j in 0..128 {
                w[off + 16 + j] = (rand() % 255) as u8;
            }
        }
    }
    let toks: Vec<Vec<f32>> = (0..B)
        .map(|t| {
            (0..cols)
                .map(|i| ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5)
                .collect()
        })
        .collect();
    let mut y_cpu = vec![0f32; B * rows];
    for t in 0..B {
        for r in 0..rows {
            let off = r * row_bytes;
            y_cpu[t * rows + r] = unsafe {
                crate::infer::quant::vec_dot_raw_q4_k(w[off..].as_ptr(), toks[t].as_ptr(), blocks)
            };
        }
    }

    let page = 16384usize;
    let mk = |bytes: usize| -> *mut u8 {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(p != libc::MAP_FAILED);
        p as *mut u8
    };
    let w_alloc = ((w.len() + page - 1) / page) * page;
    let w_ptr = mk(w_alloc);
    let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w.len()) };
    wb.copy_from_slice(&w);
    let engine = MetalEngine::new().expect("engine");
    let w_buf = engine.register_zerocopy_only(wb).expect("w");

    let slab_f32 = B * cols + B * rows;
    let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
    let slab_ptr = mk(slab_alloc);
    let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
    let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
    for t in 0..B {
        unsafe {
            std::ptr::copy_nonoverlapping(
                toks[t].as_ptr() as *const u8,
                slab_ptr.add(t * cols * 4),
                cols * 4,
            )
        };
    }
    let y_off = B * cols * 4;

    let dispatched = {
        assert!(engine.batch_open());
        let ok = engine.batch_gguf_gemm_tiled(
            12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B,
        );
        engine.batch_end();
        ok
    };
    assert!(dispatched, "gemm_q4_k_tiled PSO unavailable");

    let y_gpu: &[f32] =
        unsafe { std::slice::from_raw_parts(slab_ptr.add(y_off) as *const f32, B * rows) };
    let mut worst = 0f32;
    let mut worst_rel = 0f32;
    for i in 0..B * rows {
        let a = y_gpu[i];
        let b = y_cpu[i];
        assert!(a.is_finite(), "non-finite tiled output at {i}");
        let e = (a - b).abs();
        let rel = e / b.abs().max(1.0);
        if e > worst {
            worst = e;
        }
        if rel > worst_rel {
            worst_rel = rel;
        }
        assert!(
            rel <= 1e-5,
            "tiled parity broken at {i}: gpu={a:.6e} cpu={b:.6e} (rel {rel:.2e})"
        );
    }
    println!(
        "[tiled] rows={rows} cols={cols} B={B} worst_abs={worst:e} worst_rel={worst_rel:e} — PARITY PASS"
    );

    // rate: tiled vs shipped 8-token class, same buffers
    let mut bench = |label: &str, tiled: bool| {
        let reps = 5usize;
        assert!(engine.batch_open());
        if tiled {
            engine.batch_gguf_gemm_tiled(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
        } else {
            engine.batch_gguf_gemm(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
        }
        engine.batch_end();
        let t0 = Instant::now();
        for _ in 0..reps {
            assert!(engine.batch_open());
            let ok = if tiled {
                engine.batch_gguf_gemm_tiled(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B)
            } else {
                engine.batch_gguf_gemm(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B)
            };
            engine.batch_end();
            assert!(ok);
        }
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        let wt = rows as f64 * row_bytes as f64;
        let flop = 2.0 * rows as f64 * cols as f64 * B as f64;
        println!(
            "[tiled] {label:<10} per={:>7.3} ms  wt-GB/s={:>6.1}  TFLOP/s={:>6.2}  tok/s-eq={:>7.0}",
            per * 1e3,
            wt / per / 1e9,
            flop / per / 1e12,
            B as f64 / per
        );
        per
    };
    let p_old = bench("shipped-t8", false);
    let p_new = bench("tiled", true);
    println!("[tiled] speedup = {:.2}x", p_old / p_new);

    unsafe {
        libc::munmap(slab_ptr as *mut libc::c_void, slab_alloc);
        libc::munmap(w_ptr as *mut libc::c_void, w_alloc);
    }
}

    /// P4 arm-d parity: tiled fused q6_K / q8_0 staging must match the CPU
    /// vec_dot references. Isolated because the tiled class is one kernel per
    /// dtype and a single bad staging expression shows up as NaN logits only
    /// two layers away.
    #[test]
    #[ignore]
    fn gemm_tiled_q6_k_matches_cpu() {
        for &(ty, block_bytes) in &[(14u32, 210usize), (8u32, 34usize)] {
            std::env::set_var("SOFUU_GPU_GEMM_TILED", "1");
            // q4_K/q6_K hold 256 values per block, q8_0 holds 32: the block
            // count per ROW differs per dtype even at the same cols.
            let blocks = 2usize;
            let cols = blocks * 256;
            let nblk = if ty == 8 { cols / 32 } else { blocks };
            let row_bytes = nblk * block_bytes;
            let rows = 96usize;
            let B = 5usize;
            let mut state = 0x9E3779B97F4A7C15u64;
            let mut rand = move || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as u32
            };
            let f16_bytes = |v: f32| -> [u8; 2] {
                let b = v.to_bits();
                let sign = ((b >> 16) & 0x8000) as u16;
                let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
                let man = (b >> 13) & 0x3FF;
                (sign | (exp << 10) | (man as u16)).to_le_bytes()
            };
            let mut w = vec![0u8; rows * row_bytes];
            for r in 0..rows {
                for b in 0..nblk {
                    let off = r * row_bytes + b * block_bytes;
                    match ty {
                        14 => {
                            for j in 0..128 { w[off + j] = (rand() % 256) as u8; }
                            for j in 0..64 { w[off + 128 + j] = (rand() % 256) as u8; }
                            for j in 0..16 { w[off + 192 + j] = (rand() & 0x7F) as u8; }
                            w[off + 208..off + 210].copy_from_slice(&f16_bytes(0.0213f32));
                        }
                        _ => {
                            w[off..off + 2].copy_from_slice(&f16_bytes(0.0156f32));
                            for j in 0..32 { w[off + 2 + j] = (rand() % 255) as u8; }
                        }
                    }
                }
            }
            let toks: Vec<Vec<f32>> = (0..B)
                .map(|t| (0..cols).map(|i| ((i * 5 + t * 11) % 19) as f32 * 0.2 - 1.8).collect())
                .collect();
            let mut y_cpu = vec![0f32; B * rows];
            for t in 0..B {
                for r in 0..rows {
                    let off = r * row_bytes;
                    y_cpu[t * rows + r] = unsafe {
                        if ty == 14 {
                            crate::infer::quant::vec_dot_raw_q6_k(w[off..].as_ptr(), toks[t].as_ptr(), blocks)
                        } else {
                            // q8_0 blocks hold 32 values (q4_K/q6_K hold 256)
                            crate::infer::quant::vec_dot_raw_q8_0(
                                w[off..].as_ptr(), toks[t].as_ptr(), cols / 32,
                            )
                        }
                    };
                }
            }
            let page = 16384usize;
            let mk = |bytes: usize| -> *mut u8 {
                let p = unsafe {
                    libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
                };
                assert!(p != libc::MAP_FAILED);
                p as *mut u8
            };
            let w_alloc = ((w.len() + page - 1) / page) * page;
            let w_ptr = mk(w_alloc);
            let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w.len()) };
            wb.copy_from_slice(&w);
            let engine = MetalEngine::new().expect("engine");
            let w_buf = engine.register_zerocopy_only(wb).expect("w");
            let alloc_f32 = B * cols + B * rows;
            let slab_alloc = ((alloc_f32 * 4 + page - 1) / page) * page;
            let slab_ptr = mk(slab_alloc);
            let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
            let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
            for t in 0..B {
                for i in 0..cols {
                    let v = toks[t][i];
                    unsafe { *(slab_ptr.add((t * cols + i) * 4) as *mut f32) = v };
                }
            }
            let y_off = B * cols * 4;
            // single-section table: whole tensor, y region at y_off
            let blob: [[u32; 5]; 1] = [[0u32, 0u32, rows as u32, y_off as u32, rows as u32]];
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm_tiled_multi(
                ty, &w_buf, 0, &slab_buf, 0, &slab_buf, 0, rows, cols, B, &blob,
            );
            engine.batch_end();
            if B < 16 {
                // the route hands b<16 to the shipped 8-token class on purpose
                // (a 3-token tail must not pay for a 64-token tile): the tiled
                // entry refuses it, so there is nothing to compare here.
                assert!(!ok, "tiled multi should refuse B={B}");
                continue;
            }
            assert!(ok, "tiled multi ty={ty} unavailable");
            let y_gpu: &[f32] = unsafe { std::slice::from_raw_parts(slab_ptr.add(y_off) as *const f32, B * rows) };
            let mut worst = 0f32;
            for i in 0..B * rows {
                let (a, b) = (y_gpu[i], y_cpu[i]);
                assert!(a.is_finite(), "ty={ty} non-finite at {i} (t={} r={})", i / rows, i % rows);
                // 32-value k-slices accumulate in a different order than the
                // CPU's per-256-value block dot, and q6_K's ±32 quants cancel:
                // 1e-4 relative is the noise class, an indexing defect is >=1e-1.
                let e = (a - b).abs() / b.abs().max(1.0);
                worst = worst.max(e);
                assert!(e <= 1e-3, "ty={ty} parity broken at i={i}: gpu={a:.6e} cpu={b:.6e}");
            }
            println!("[tiled-{ty}] rows={rows} cols={cols} B={B} worst_rel={worst:e} — PARITY PASS");
            unsafe {
                libc::munmap(slab_ptr as *mut libc::c_void, slab_alloc);
                libc::munmap(w_ptr as *mut libc::c_void, w_alloc);
            }
        }
    }

/// P4 rate harness (llama.cpp-parity arm d): the GEMM class's honest per-shape
/// number at fleet dimensions, reported as weight-stream GB/s, TFLOP/s and a
/// token-rate equivalent — so "we are Nx from the bar" stops being a ratio of
/// two e2e runs with different loads. llama.cpp's bar implies ~3-4 TFLOP/s of
/// effective GEMM throughput on this box; this prints what the shipped class
/// actually delivers. Run:
///   cargo test --release -p sofuu-core -- --ignored --nocapture bench_gemm_rate_bands
#[test]
#[ignore] // GPU bench class
fn bench_gemm_rate_bands() {
    use std::time::Instant;

    let page = 16384usize;
    let mk = |bytes: usize| -> *mut u8 {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(p != libc::MAP_FAILED);
        p as *mut u8
    };
    let f16_bytes = |v: f32| -> [u8; 2] {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
        let man = (b >> 13) & 0x3FF;
        (sign | (exp << 10) | (man as u16)).to_le_bytes()
    };
    let mut state = 0x5DEECE66Du64;
    let mut rand = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };

    // (dtype, rows, cols, label) — rows = output rows, cols = reduction dim.
    let cases: &[(u32, usize, usize, &str)] = &[
        (12, 1024, 1024, "q4K o-proj"),
        (12, 1024, 2048, "q4K q-proj"),
        (12, 3072, 1024, "q4K gate"),
        (12, 1024, 3072, "q4K down"),
        (14, 3072, 1024, "q6K gate"),
        (14, 1024, 3072, "q6K down"),
        (8, 2048, 2048, "q8_0 q"),
        (8, 8192, 2048, "q8_0 gate"),
    ];
    let bs = [8usize, 32, 64, 128];
    let bmax = *bs.iter().max().unwrap();
    let engine = MetalEngine::new().expect("engine");

    println!("[rate] dtype    shape            B    per-ms   wt-GB/s  TFLOP/s  tok/s-eq");
    for &(ty, rows, cols, label) in cases {
        let (bvals, bbytes) = match ty {
            8 => (32usize, 34usize),
            12 => (256, 144),
            14 => (256, 210),
            _ => unreachable!(),
        };
        assert!(cols % bvals == 0);
        let blocks = cols / bvals;
        let row_bytes = blocks * bbytes;
        let mut w = vec![0u8; rows * row_bytes];
        for r in 0..rows {
            for b in 0..blocks {
                let off = r * row_bytes + b * bbytes;
                match ty {
                    8 => {
                        w[off..off + 2].copy_from_slice(&f16_bytes(0.03125f32));
                        for j in 0..32 {
                            w[off + 2 + j] = (rand() % 255) as u8;
                        }
                    }
                    12 => {
                        w[off..off + 2].copy_from_slice(&f16_bytes(0.0625f32));
                        w[off + 2..off + 4].copy_from_slice(&f16_bytes(0.0005f32));
                        for j in 0..12 {
                            w[off + 4 + j] = ((rand() & 0x3F) as u8) | (((rand() & 3) as u8) << 6);
                        }
                        for j in 0..128 {
                            w[off + 16 + j] = (rand() % 255) as u8;
                        }
                    }
                    14 => {
                        for j in 0..128 {
                            w[off + j] = (rand() % 255) as u8;
                        }
                        for j in 0..64 {
                            w[off + 128 + j] = (rand() % 255) as u8;
                        }
                        for j in 0..16 {
                            w[off + 192 + j] = (rand() % 63) as u8;
                        }
                        w[off + 208..off + 210].copy_from_slice(&f16_bytes(0.03125f32));
                    }
                    _ => unreachable!(),
                }
            }
        }
        let w_alloc = ((w.len() + page - 1) / page) * page;
        let w_ptr = mk(w_alloc);
        let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w.len()) };
        wb.copy_from_slice(&w);
        let w_buf = engine.register_zerocopy_only(wb).expect("w");

        let slab_f32 = bmax * cols + bmax * rows;
        let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = mk(slab_alloc);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");

        for &b in &bs {
            let y_off = b * cols * 4;
            // warmup (PSO select + any first-touch), then timed reps
            assert!(engine.batch_open());
            engine.batch_gguf_gemm(ty, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, b);
            engine.batch_end();
            let reps = 5usize;
            let t0 = Instant::now();
            for _ in 0..reps {
                assert!(engine.batch_open());
                let ok = engine
                    .batch_gguf_gemm(ty, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, b);
                engine.batch_end();
                assert!(ok, "gemm PSO unavailable for ty={ty}");
            }
            let per = t0.elapsed().as_secs_f64() / reps as f64;
            let wt = rows as f64 * row_bytes as f64;
            let flop = 2.0 * rows as f64 * cols as f64 * b as f64;
            println!(
                "[rate] {:<8} {:>5}x{:<5} {:>4}  {:>8.3}  {:>8.1}  {:>7.2}  {:>9.0}",
                match ty { 8 => "q8_0", 12 => "q4K", 14 => "q6K", _ => "?" },
                rows,
                cols,
                b,
                per * 1e3,
                wt / per / 1e9,
                flop / per / 1e12,
                b as f64 / per
            );
        }
        unsafe {
            libc::munmap(slab_ptr as *mut libc::c_void, slab_alloc);
            libc::munmap(w_ptr as *mut libc::c_void, w_alloc);
        }
    }
}

/// Phase 3b: the tile-32 (lane = token) q8_0 GEMM — weights read once per
/// threadgroup instead of once per 8-token tile (the tile-8 class re-reads
/// weights 16x at B=128 and its stream RATE DROPS with B). Same fixture,
/// reference and tolerances as gemm_q8_0_matches_cpu; run ignored tests
/// with --test-threads=1 (the SOFUU_GEMM_TILE env selection is process-
/// global and read at engine init).
#[test]
#[ignore] // GPU bench class: `cargo test -- --ignored --test-threads=1`
fn gemm_q8_0_t32_matches_cpu() {
    use std::time::Instant;

    std::env::set_var("SOFUU_GEMM_TILE", "32");
    let rows = 2048usize;
    let blocks = 64usize;
    let cols = blocks * 32;
    let row_bytes = blocks * 34;

    let mut state = 0x2545F4914F6CDD1Du64;
    let mut rand = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let f16_bytes = |v: f32| -> [u8; 2] {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
        let man = (b >> 13) & 0x3FF;
        (sign | (exp << 10) | (man as u16)).to_le_bytes()
    };
    let mut w = vec![0u8; rows * row_bytes];
    for r in 0..rows {
        for b in 0..blocks {
            let off = r * row_bytes + b * 34;
            w[off..off + 2].copy_from_slice(&f16_bytes(0.03125f32));
            for j in 0..32 {
                w[off + 2 + j] = (rand() % 255) as u8;
            }
        }
    }
    for b_case in [5usize, 32usize, 128usize] {
        let B = b_case;
        let toks: Vec<Vec<f32>> = (0..B)
            .map(|t| (0..cols).map(|i| ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5).collect())
            .collect();
        let mut y_cpu = vec![0f32; B * rows];
        for t in 0..B {
            for r in 0..rows {
                y_cpu[t * rows + r] =
                    crate::infer::quant::vec_dot_raw_q8_0(w[r * row_bytes..].as_ptr(), toks[t].as_ptr(), blocks);
            }
        }
        let page = 16384usize;
        let mk = |bytes: usize| -> *mut u8 {
            let p = unsafe {
                libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
            };
            assert!(p != libc::MAP_FAILED);
            p as *mut u8
        };
        let w_alloc = ((w.len() + page - 1) / page) * page;
        let w_ptr = mk(w_alloc);
        let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w.len()) };
        wb.copy_from_slice(&w);
        let engine = MetalEngine::new().expect("engine");
        let w_buf = engine.register_zerocopy_only(wb).expect("w");
        let slab_f32 = B * cols + B * rows;
        let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = mk(slab_alloc);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        for t in 0..B {
            unsafe {
                std::ptr::copy_nonoverlapping(toks[t].as_ptr() as *const u8, slab_ptr.add(t * cols * 4), cols * 4)
            };
        }
        let y_off = B * cols * 4;
        let dispatched = {
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(8, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
            engine.batch_end();
            ok
        };
        assert!(dispatched, "gemm_q8_0_t32 PSO unavailable at B={B}");
        let y_gpu: &[f32] =
            unsafe { std::slice::from_raw_parts(slab_ptr.add(y_off) as *const f32, B * rows) };
        let mut worst = 0f64;
        let (mut wt, mut wr, mut wa, mut wcr) = (0usize, 0usize, 0f32, 0f32);
        for t in 0..B {
            for r in 0..rows {
                let a = y_gpu[t * rows + r];
                let b = y_cpu[t * rows + r];
                let e = (a as f64 - b as f64).abs() / b.abs().max(1.0) as f64;
                if e > worst { worst = e; wt = t; wr = r; wa = a; wcr = b; }
            }
        }
        assert!(worst <= 1e-6, "gemm_q8_0_t32 parity broken at B={B} t={wt} r={wr}: gpu={wa:.6e} cpu={wcr:.6e} (rel {worst:.2e})");
        println!("[gemm_q8_0_t32] B={B} rows={rows} cols={cols} worst_rel={worst:.2e} — PARITY PASS");
        let t0 = Instant::now();
        let reps = 4usize;
        for _ in 0..reps {
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(8, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
            engine.batch_end();
            assert!(ok);
        }
        let per = t0.elapsed() / reps as u32;
        let bytes = rows as f64 * row_bytes as f64;
        println!("[gemm_q8_0_t32] B={B} per-dispatch {:.3?} ms -> {:.1} GB/s weight stream", per.as_secs_f64() * 1e3, bytes / per.as_secs_f64() / 1e9);
        unsafe { libc::munmap(w_ptr as *mut libc::c_void, w_alloc) };
        unsafe { libc::munmap(slab_ptr as *mut libc::c_void, slab_alloc) };
    }
}

/// Phase 3b: tile-32 q6_K GEMM oracle (0.6B attn_v shape), same fixture and
/// tolerances as gemm_q6_k_matches_cpu; run with --test-threads=1 (the
/// SOFUU_GEMM_TILE env selection is process-global at engine init).
#[test]
#[ignore] // GPU bench class: `cargo test -- --ignored --test-threads=1`
fn gemm_q6_k_t32_matches_cpu() {
    use std::time::Instant;

    std::env::set_var("SOFUU_GEMM_TILE", "32");
    let rows = 1024usize; // 0.6B attn_v (kv_rows)
    let blocks = 4usize; // cols 1024
    let cols = blocks * 256;
    let row_bytes = blocks * 210;

    let mut state = 0x9E3779B97F4A7C15u64;
    let mut rand = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let f16_bytes = |v: f32| -> [u8; 2] {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
        let man = (b >> 13) & 0x3FF;
        (sign | (exp << 10) | (man as u16)).to_le_bytes()
    };
    let mut w = vec![0u8; rows * row_bytes];
    for r in 0..rows {
        for b in 0..blocks {
            let off = r * row_bytes + b * 210;
            for j in 0..128 {
                w[off + j] = (rand() % 255) as u8;
            }
            for j in 0..64 {
                w[off + 128 + j] = (rand() % 255) as u8;
            }
            for j in 0..16 {
                w[off + 192 + j] = (rand() % 63) as u8;
            }
            w[off + 208..off + 210].copy_from_slice(&f16_bytes(0.03125f32));
        }
    }
    for b_case in [5usize, 32usize, 128usize] {
        let B = b_case;
        let toks: Vec<Vec<f32>> = (0..B)
            .map(|t| (0..cols).map(|i| ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5).collect())
            .collect();
        let mut y_cpu = vec![0f32; B * rows];
        for t in 0..B {
            for r in 0..rows {
                y_cpu[t * rows + r] =
                    crate::infer::quant::vec_dot_raw_q6_k(w[r * row_bytes..].as_ptr(), toks[t].as_ptr(), blocks);
            }
        }
        let page = 16384usize;
        let mk = |bytes: usize| -> *mut u8 {
            let p = unsafe {
                libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
            };
            assert!(p != libc::MAP_FAILED);
            p as *mut u8
        };
        let w_alloc = ((w.len() + page - 1) / page) * page;
        let w_ptr = mk(w_alloc);
        let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w.len()) };
        wb.copy_from_slice(&w);
        let engine = MetalEngine::new().expect("engine");
        let w_buf = engine.register_zerocopy_only(wb).expect("w");
        let slab_f32 = B * cols + B * rows;
        let slab_alloc = ((slab_f32 * 4 + page - 1) / page) * page;
        let slab_ptr = mk(slab_alloc);
        let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
        let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
        for t in 0..B {
            unsafe {
                std::ptr::copy_nonoverlapping(toks[t].as_ptr() as *const u8, slab_ptr.add(t * cols * 4), cols * 4)
            };
        }
        let y_off = B * cols * 4;
        let dispatched = {
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(14, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
            engine.batch_end();
            ok
        };
        assert!(dispatched, "gemm_q6_k_t32 PSO unavailable at B={B}");
        let y_gpu: &[f32] =
            unsafe { std::slice::from_raw_parts(slab_ptr.add(y_off) as *const f32, B * rows) };
        let mut worst = 0f64;
        let (mut wt, mut wr, mut wa, mut wcr) = (0usize, 0usize, 0f32, 0f32);
        for t in 0..B {
            for r in 0..rows {
                let a = y_gpu[t * rows + r];
                let b = y_cpu[t * rows + r];
                let e = (a as f64 - b as f64).abs() / b.abs().max(1.0) as f64;
                if e > worst { worst = e; wt = t; wr = r; wa = a; wcr = b; }
            }
        }
        assert!(worst <= 1e-5, "gemm_q6_k_t32 parity broken at B={B} t={wt} r={wr}: gpu={wa:.6e} cpu={wcr:.6e} (rel {worst:.2e})");
        println!("[gemm_q6_k_t32] B={B} rows={rows} cols={cols} worst_rel={worst:.2e} — PARITY PASS");
        let t0 = Instant::now();
        let reps = 4usize;
        for _ in 0..reps {
            assert!(engine.batch_open());
            let ok = engine.batch_gguf_gemm(14, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
            engine.batch_end();
            assert!(ok);
        }
        let per = t0.elapsed() / reps as u32;
        let bytes = rows as f64 * row_bytes as f64;
        println!("[gemm_q6_k_t32] B={B} per-dispatch {:.3?} ms -> {:.1} GB/s weight stream", per.as_secs_f64() * 1e3, bytes / per.as_secs_f64() / 1e9);
        unsafe { libc::munmap(w_ptr as *mut libc::c_void, w_alloc) };
        unsafe { libc::munmap(slab_ptr as *mut libc::c_void, slab_alloc) };
    }
}

/// Phase-3 defect pin: the 1B (32 heads, hd 64, nkv 8) faults its GPU queue
/// on the start_pos=128, b=128 multi-chunk attention pass — every in-model
/// run, every load. This test dispatches each attention-glue op
/// (kv_write_b4 / head_rmsnorm_b4 / attn_part_b4 / attn_fin_b4) STANDALONE
/// at exactly those shapes, each in its own command buffer so batch_end's
/// completion bool names the faulting op, plus an all-in-one chain.
/// MTL_SHADER_VALIDATION=1 makes the GPU reject OOB shader accesses with a
/// named kernel instead of a queue restart. Run:
/// cargo test -- --ignored --test-threads=1
#[test]
#[ignore]
fn attn_glue_b4_fault_isolation_1b() {
    std::env::set_var("MTL_SHADER_VALIDATION", "1");
    let nh = 32u32;
    let nkv = 8u32;
    let hd = 64u32;
    let q_rows = (nh * hd) as usize;
    let kv_rows = (nkv * hd) as usize;
    let ncm = 32u32; // meta_chunks at max_seq 4096, as the engine passes it
    let max_seq = 4096u32;
    let meta_per_layer =
        (nh as usize) * (ncm as usize) * 2 + (nh as usize) * (ncm as usize) * (hd as usize);

    let page = 16384usize;
    let mk = |bytes: usize| -> *mut u8 {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(p != libc::MAP_FAILED);
        p as *mut u8
    };
    let alloc = |bytes: usize| bytes.div_ceil(page) * page;

    let bmax = 128usize;
    let sizes: Vec<(usize, bool)> = vec![
        (q_rows * bmax * 4, true),   // q: random f32
        (kv_rows * bmax * 4, true),  // k
        (kv_rows * bmax * 4, true),  // v
        (q_rows * bmax * 4, true),   // out
        ((nkv as usize) * (max_seq as usize) * (hd as usize) * 4, false), // kv cache
        (meta_per_layer * 256 * 4, false), // meta (must be zeroed first)
        (bmax * 4, false),           // positions
        ((hd as usize) * 4 * 2, true), // q/k norm weights (small)
    ];
    let mut ptrs: Vec<(*mut u8, usize)> = Vec::new();
    for &(bytes, _) in &sizes {
        let a = alloc(bytes);
        ptrs.push((mk(a), a));
    }
    for (i, &(bytes, random)) in sizes.iter().enumerate() {
        let (p, a) = ptrs[i];
        unsafe {
            let sl = std::slice::from_raw_parts_mut(p, bytes);
            if random {
                let f = std::slice::from_raw_parts_mut(sl.as_ptr() as *mut f32, bytes / 4);
                for (j, v) in f.iter_mut().enumerate() {
                    *v = (((j as u64 * 2654435761 + 0x9e37) >> 33) % 2003) as f32 / 1000.0 - 1.0;
                }
            } else {
                std::ptr::write_bytes(sl.as_mut_ptr(), 0, bytes);
            }
        }
    }
    let engine = MetalEngine::new().expect("engine");
    let registered: Vec<MetalBuffer> = {
        let mut regs = Vec::new();
        for i in 0..sizes.len() {
            let sl = unsafe { std::slice::from_raw_parts_mut(ptrs[i].0, sizes[i].0) };
            regs.push(engine.register_zerocopy_only(sl).expect("reg"));
        }
        regs
    };
    let (q_b, k_buf, v_buf, out_b, kvc_b, meta_b, pos_b, w_b) = (
        &registered[0], &registered[1], &registered[2], &registered[3],
        &registered[4], &registered[5], &registered[6], &registered[7],
    );
    let eps = 1e-5f32;
    let rope = 500000.0f32;
    let mut faults: Vec<String> = Vec::new();

    for &b in &[32usize, 60, 64, 100, 128] {
        // positions 128..128+b-1 — the in-model failing pass shapes
        unsafe {
            let ps = std::slice::from_raw_parts_mut(ptrs[6].0, b * 4);
            for t in 0..b {
                ps[t * 4..t * 4 + 4].copy_from_slice(&((128 + t) as u32).to_le_bytes());
            }
        }
        let layer8_off = 8 * meta_per_layer * 4;
        macro_rules! run_isolated {
            ($name:literal, $ops:expr) => {{
                engine.batch_open();
                $ops;
                let ok = engine.batch_end();
                println!("[glue-pin] b={b} {} -> {}", $name, if ok { "OK" } else { "FAULT" });
                if !ok {
                    faults.push(format!("b={b} {}", $name));
                }
            }};
        }
        run_isolated!("kv_write_b4", engine.batch_llama_kv_write_b4(
            k_buf, 0, v_buf, 0, kvc_b, 0, kvc_b, 0, Some(w_b), (hd as usize) * 4, 1u32,
            pos_b, 0, b, hd, nkv, max_seq, eps, rope, 0u32, None));
        run_isolated!("head_rmsnorm_b4", engine.batch_head_rmsnorm_b4(
            q_b, 0, Some(w_b), 0, pos_b, 0, b, hd, nh, 1u32, eps, rope, 0u32, None));
        run_isolated!("attn_part_b4", engine.batch_llama_attn_part_b4(
            q_b, 0, kvc_b, 0, kvc_b, 0, meta_b, layer8_off, pos_b, 0, b,
            hd, nh, nkv, max_seq, ncm, meta_per_layer as u32));
        run_isolated!("attn_fin_b4", engine.batch_llama_attn_fin_b4(
            meta_b, layer8_off, out_b, 0, pos_b, 0, b, nh, hd, ncm, meta_per_layer as u32));
        run_isolated!("ALL-IN-ONE", {
            engine.batch_llama_kv_write_b4(
                k_buf, 0, v_buf, 0, kvc_b, 0, kvc_b, 0, Some(w_b), (hd as usize) * 4, 1u32,
                pos_b, 0, b, hd, nkv, max_seq, eps, rope, 0u32, None);
            engine.batch_head_rmsnorm_b4(
                q_b, 0, Some(w_b), 0, pos_b, 0, b, hd, nh, 1u32, eps, rope, 0u32, None);
            engine.batch_llama_attn_part_b4(
                q_b, 0, kvc_b, 0, kvc_b, 0, meta_b, layer8_off, pos_b, 0, b,
                hd, nh, nkv, max_seq, ncm, meta_per_layer as u32);
            engine.batch_llama_attn_fin_b4(
                meta_b, layer8_off, out_b, 0, pos_b, 0, b, nh, hd, ncm, meta_per_layer as u32);
        });
    }
    for f in &faults {
        eprintln!("[glue-pin] FAULT: {f}");
    }
    assert!(faults.is_empty(), "attention-glue faults at 1B shapes: {faults:?}");
    for (p, a) in &ptrs {
        unsafe { libc::munmap(*p as *mut libc::c_void, *a) };
    }
}

/// Dual-kernel MMQ probe (SOFUU_TILE_MMQ_PROBE): the shipped CPU tile gemm
/// (sofuu_gemm_q4_k, with tile_warm — production parity) vs the GPU mirror
/// (batch_quantize_q8_k + batch_gemm_q4_k_mmq) on the SAME synthetic
/// weights/activations at rows=8/cols=256/batch=4. Structured scale bytes
/// make any scale/min mapping difference show up as a per-(row, subblock)
/// delta pattern; the quantizer diff shows up as a d-column delta. This is
/// the byte-identity receipt for the 8x4-tile mirror. Run: cargo test --
/// --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn mmq_tile_value_probe() {
    use std::os::raw::c_void;
    extern "C" {
        fn sofuu_gemm_q4_k(
            w: *const u8, x: *const f32, out: *mut f32,
            rows: usize, cols: usize, batch: usize,
        );
        fn sofuu_tile_warm(w: *const u8, rows: usize, cols: usize, ty: u32) -> i32;
    }
    let rows = 8usize;
    let cols = 256usize;
    let batch = 4usize;
    // weights: 1 block per row; d=1.0, dmin=0.25; scales bytes = structured
    let mut w: Vec<u8> = Vec::with_capacity(rows * 144);
    for r in 0..rows {
        w.extend_from_slice(&1.0f32.to_le_bytes()[..2]); // f16 1.0
        w.extend_from_slice(&(0x3CA0u16).to_le_bytes()); // f16 0.03125
        let mut sb = Vec::with_capacity(12);
        for i in 0..12 {
            sb.push(((r * 31 + i * 17 + 5) & 0xFF) as u8);
        }
        w.extend_from_slice(&sb);
        // qs: deterministic nibble pattern
        for i in 0..128 {
            w.push(((i * 37 + r * 91 + 11) & 0xFF) as u8);
        }
    }
    // activations: 4 tokens
    let mut x: Vec<f32> = Vec::with_capacity(batch * cols);
    for t in 0..batch {
        for j in 0..cols {
            x.push(((j * 7 + t * 61) % 253) as f32 / 25.0 - 5.0);
        }
    }

    // ---- CPU shipped kernel (production parity: warm first) ----
    let warm = unsafe { sofuu_tile_warm(w.as_ptr(), rows, cols, 12) };
    eprintln!("[probe] tile_warm = {warm}");
    let mut cpu_out = vec![0f32; batch * rows];
    unsafe {
        sofuu_gemm_q4_k(w.as_ptr(), x.as_ptr(), cpu_out.as_mut_ptr(), rows, cols, batch);
    }

    // ---- host model of the pipeline (my p-algorithm reduction) ----
    fn gsmk(j: usize, q: &[u8]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }
    let host_out = {
        #[inline]
        fn magic_round(v: f32) -> i32 {
            let q = v + 12582912.0f32;
            (q - 12582912.0f32) as i32
        }
        let mut out = vec![0f32; batch * rows];
        for t in 0..batch {
            let seg = &x[t * cols..(t + 1) * cols];
            let amax = seg.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mut mx = 0.0f32;
            for v in seg.iter() {
                if v.abs() == amax { mx = *v; break; }
            }
            let iscale = if amax != 0.0 { -127.0f32 / mx } else { 0.0 };
            let d8 = if amax != 0.0 { 1.0 / iscale } else { 0.0 };
            let q8: Vec<i32> = seg.iter().map(|v| magic_round(v * iscale)).collect();
            for r in 0..rows {
                let blk = &w[r * 144..(r + 1) * 144];
                let d = half_to_f32((blk[1] as u16) << 8 | blk[0] as u16);
                let dmin = half_to_f32((blk[3] as u16) << 8 | blk[2] as u16);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let sbd_scale = d * d8;
                let sbd_min = dmin * d8;
                let mut acc = 0.0f32;
                let mut bias: i32 = 0;
                for s in 0..4 {
                    let (sc_lo, m_lo) = gsmk(s * 2, scales);
                    let (sc_hi, m_hi) = gsmk(s * 2 + 1, scales);
                    let mut acc_lo = 0i32;
                    let mut acc_hi = 0i32;
                    let mut bsum_lo = 0i32;
                    let mut bsum_hi = 0i32;
                    for k in 0..32usize {
                        let wq = qs[s * 32 + k];
                        let ql = q8[s * 64 + k];
                        let qh = q8[s * 64 + 32 + k];
                        acc_lo += ((wq & 0x0F) as i32) * ql;
                        acc_hi += ((wq >> 4) as i32) * qh;
                        bsum_lo += ql;
                        bsum_hi += qh;
                    }
                    let sum_int = (sc_lo as i32) * acc_lo + (sc_hi as i32) * acc_hi;
                    acc = acc + sbd_scale * (sum_int as f32); // host f32 approx
                    bias += (m_lo as i32) * bsum_lo + (m_hi as i32) * bsum_hi;
                }
                acc -= sbd_min * (bias as f32);
                out[t * rows + r] = acc;
            }
        }
        out
    };

    // ---- GPU mirror ----
    let page = 16384usize;
    let engine = MetalEngine::new().expect("engine");
    let mk = |bytes: usize| -> (*mut u8, usize) {
        let sz = bytes.div_ceil(page) * page;
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), sz, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(p != libc::MAP_FAILED);
        (p as *mut u8, sz)
    };
    let (wp, wsz) = mk(w.len());
    unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), wp, w.len()) };
    let w_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(wp, wsz) }).expect("w");
    let (xp, xsz) = mk(x.len() * 4);
    unsafe { std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, xp, x.len() * 4) };
    let x_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(xp, xsz) }).expect("x");
    let xq8_bytes = batch * (cols / 256) * 260;
    let (qp, qsz) = mk(xq8_bytes);
    let q_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(qp, qsz) }).expect("q");
    let (op, osz) = mk(batch * rows * 4);
    let o_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(op, osz) }).expect("o");
    let dispatched = {
        assert!(engine.batch_open());
        let q_ok = engine.batch_quantize_q8_k(&x_buf, 0, &q_buf, 0, cols, batch);
        let g_ok = q_ok && engine.batch_gemm_q4_k_mmq(&w_buf, 0, &q_buf, 0, &o_buf, 0, rows, cols, batch);
        engine.batch_end();
        q_ok && g_ok
    };
    assert!(dispatched, "probe: PSO unavailable");
    let gpu_out: Vec<f32> = (0..batch * rows)
        .map(|i| unsafe { (op as *const f32).add(i).read() })
        .collect();

    // dump the GPU quantized block for token 0
    let d0 = unsafe { (qp as *const f32).read() };
    eprintln!("[probe] GPU xq8 d[token0] = {d8_repr}", d8_repr = d0);
    // reconstruct the host-side quantized ints for token 0 and diff against
    // the GPU's qs bytes
    let seg0 = &x[0..cols];
    let amax0 = seg0.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let mut mx0 = 0.0f32;
    for v in seg0.iter() {
        if v.abs() == amax0 { mx0 = *v; break; }
    }
    let iscale0 = if amax0 != 0.0 { -127.0f32 / mx0 } else { 0.0 };
    let mut q_mismatch = 0usize;
    for j in 0..cols {
        let qh = ((seg0[j] * iscale0 + 12582912.0f32) - 12582912.0f32) as i32 as i8;
        let qg = unsafe { *((qp as *const i8).add(4 + j)) }; // d at 0..4
        if qh != qg { q_mismatch += 1; }
    }
    eprintln!(
        "[probe] token0 qs: mismatches={q_mismatch}/256 host_q[0..8]={:?} gpu_q[0..8]={:?}",
        (0..8).map(|j| ((seg0[j] * iscale0 + 12582912.0f32) - 12582912.0f32) as i32 as i8).collect::<Vec<_>>(),
        (0..8).map(|j| unsafe { *((qp as *const i8).add(4 + j)) }).collect::<Vec<_>>(),
    );

    let mut worst = 0f32;
    let mut wr = 0usize;
    for t in 0..batch {
        for r in 0..rows {
            let i = t * rows + r;
            let d = (cpu_out[i] - gpu_out[i]).abs();
            if d > worst { worst = d; wr = i; }
        }
    }
    let mut worst_host = 0f32;
    for t in 0..batch {
        for r in 0..rows {
            let i = t * rows + r;
            let d = (cpu_out[i] - host_out[i]).abs();
            if d > worst_host { worst_host = d; }
        }
    }
    eprintln!(
        "[probe] CPU-vs-GPU worst={worst:.4e} at idx {wr} (tok{} row{}) | CPU-vs-HOSTMODEL worst={worst_host:.4e}",
        wr / rows, wr % rows
    );
    for t in 0..batch {
        for r in 0..rows {
            eprintln!(
                "r{r} t{t}: cpu={:+.6} gpu={:+.6} host={:+.6}",
                cpu_out[t * rows + r], gpu_out[t * rows + r], host_out[t * rows + r]
            );
        }
    }
    unsafe { libc::munmap(wp as *mut libc::c_void, wsz) };
    unsafe { libc::munmap(xp as *mut libc::c_void, xsz) };
    unsafe { libc::munmap(qp as *mut libc::c_void, qsz) };
    unsafe { libc::munmap(op as *mut libc::c_void, osz) };

    // ── q6_K section ────────────────────────────────────────────────
    #[inline]
    fn rint_i32(v: f32) -> i32 {
        let r = v + 12582912.0f32;
        ((r - 12582912.0f32) as i32)
    }
    extern "C" {
        fn sofuu_gemm_q6_k(
            w: *const u8, x: *const f32, out: *mut f32,
            rows: usize, cols: usize, batch: usize,
        );
    }
    let mut w6: Vec<u8> = Vec::with_capacity(rows * 210);
    for r in 0..rows {
        for i in 0..128u32 {
            w6.push(((i * 29 + (r as u32) * 13 + 7) & 0xFF) as u8);
        }
        for i in 0..64u32 {
            w6.push(((i * 53 + (r as u32) * 41 + 3) & 0x3F) as u8);
        }
        for i in 0..16u8 {
            w6.push(((i as i8) - 8) as u8);
        }
        w6.extend_from_slice(&0x3C00u16.to_le_bytes()); // f16 1.0 — repo block_q6_K: d LAST
    }
    let warm6 = unsafe { sofuu_tile_warm(w6.as_ptr(), rows, cols, 14) };
    eprintln!("[probe] q6 tile_warm = {warm6}");
    let mut cpu6 = vec![0f32; batch * rows];
    unsafe {
        sofuu_gemm_q6_k(w6.as_ptr(), x.as_ptr(), cpu6.as_mut_ptr(), rows, cols, batch);
    }
    let host6 = {
        let mut out = vec![0f32; batch * rows];
        for t in 0..batch {
            let seg = &x[t * cols..(t + 1) * cols];
            let amax = seg.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mut mx = 0.0f32;
            for v in seg.iter() {
                if v.abs() == amax { mx = *v; break; }
            }
            let iscale = if amax != 0.0 { -127.0f32 / mx } else { 0.0 };
            let d8 = if amax != 0.0 { 1.0 / iscale } else { 0.0 };
            let q8: Vec<i32> = seg.iter().map(|v| rint_i32(v * iscale)).collect();
            for r in 0..rows {
                let blk = &w6[r * 210..(r + 1) * 210];
                let d = half_to_f32((blk[209] as u16) << 8 | blk[208] as u16);
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc = &blk[192..208];
                let sbd = d * d8;
                let mut acc_int: i64 = 0;
                for n in 0..2usize {
                    for sb in 0..4usize {
                        let mut acc_l: i64 = 0;
                        let mut acc_h: i64 = 0;
                        for j in 0..32usize {
                            let m = sb * 32 + j;
                            let qn = sb;
                            let qlb = if qn % 2 == 0 { ql[n * 64 + j] } else { ql[n * 64 + j + 32] };
                            let nib = if qn < 2 { qlb & 0x0F } else { qlb >> 4 };
                            let v = (((nib | (((qh[n * 32 + j] >> (qn * 2)) & 3) << 4)) as i32) - 32) as i64;
                            let qv = q8[n * 128 + m] as i64;
                            if j < 16 { acc_l += v * qv; } else { acc_h += v * qv; }
                        }
                        let sc_l = sc[n * 8 + sb * 2] as i8 as i64;
                        let sc_h = sc[n * 8 + 1 + sb * 2] as i8 as i64;
                        acc_int += sc_l * acc_l + sc_h * acc_h;
                    }
                }
                out[t * rows + r] = sbd * (acc_int as f32);
            }
        }
        out
    };
    let (wp6, wsz6) = mk(w6.len());
    unsafe { std::ptr::copy_nonoverlapping(w6.as_ptr(), wp6, w6.len()) };
    let w6_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(wp6, wsz6) }).expect("w6");
    let (op6, osz6) = mk(batch * rows * 4);
    let o6_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(op6, osz6) }).expect("o6");
    let dispatched6 = {
        assert!(engine.batch_open());
        let q_ok = engine.batch_quantize_q8_k(&x_buf, 0, &q_buf, 0, cols, batch);
        let g_ok = q_ok && engine.batch_gemm_q6_k_mmq(&w6_buf, 0, &q_buf, 0, &o6_buf, 0, rows, cols, batch);
        engine.batch_end();
        q_ok && g_ok
    };
    assert!(dispatched6, "q6 PSO unavailable");
    let mut worst6 = 0f32;
    let mut hworst6 = 0f32;
    for i in 0..batch * rows {
        let g = unsafe { (op6 as *const f32).add(i).read() };
        let d = (cpu6[i] - g).abs();
        if d > worst6 { worst6 = d; }
        let d2 = (cpu6[i] - host6[i]).abs();
        if d2 > hworst6 { hworst6 = d2; }
    }
    eprintln!("[probe] q6 CPU-vs-GPU worst={worst6:.4e} | CPU-vs-HOST worst={hworst6:.4e}");
    for t in 0..batch.min(2) {
        for r in 0..rows {
            eprintln!(
                "q6 r{r} t{t}: cpu={:+.4} gpu={:+.4} host={:+.4}",
                cpu6[t * rows + r],
                unsafe { (op6 as *const f32).add(t * rows + r).read() },
                host6[t * rows + r]
            );
        }
    }
    unsafe { libc::munmap(wp6 as *mut libc::c_void, wsz6) };
    unsafe { libc::munmap(op6 as *mut libc::c_void, osz6) };

    // ── per-32 decode gemv section (batch = 1 token) ───────────────
    extern "C" {
        fn sofuu_gemv_q4_k(
            w: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize,
        );
        fn sofuu_gemv_q6_k(
            w: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize,
        );
    }
    let x1: Vec<f32> = x[..cols].to_vec();
    // CPU references (production parity: warm first so the tile gemv is live;
    // the mmq32 mirror targets the DOTPROD path — force it via warm miss on
    // shape? rows=8/cols=256 IS tile-eligible → the shipped gemv takes the
    // TILE path. The mmq32 target is the dotprod class; compare honestly):
    let mut cq4 = vec![0f32; rows];
    let mut cq6 = vec![0f32; rows];
    unsafe {
        sofuu_gemv_q4_k(w.as_ptr(), x1.as_ptr(), cq4.as_mut_ptr(), rows, cols);
        sofuu_gemv_q6_k(w6.as_ptr(), x1.as_ptr(), cq6.as_mut_ptr(), rows, cols);
    }
    let nb_total = cols / 32;
    let (qp2, qsz2) = mk(nb_total * 40);
    let q2_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(qp2, qsz2) }).expect("q2");
    let (yp4, ysz4) = mk(rows * 4);
    let y4_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(yp4, ysz4) }).expect("y4");
    let (yp6, ysz6) = mk(rows * 4);
    let y6_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(yp6, ysz6) }).expect("y6");
    // x1 into a page buffer
    let (xp1, xsz1) = mk(x1.len() * 4);
    unsafe { std::ptr::copy_nonoverlapping(x1.as_ptr() as *const u8, xp1, x1.len() * 4) };
    let x1_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(xp1, xsz1) }).expect("x1");
    let d32 = {
        assert!(engine.batch_open());
        let okq = engine.batch_quantize_q8_ext(&x1_buf, 0, &q2_buf, 0, nb_total);
        let g4 = okq && engine.batch_gemv_q4_k_mmq32(&w_buf, 0, &q2_buf, 0, &y4_buf, 0, rows, cols);
        let g6 = okq && engine.batch_gemv_q6_k_mmq32(&w6_buf, 0, &q2_buf, 0, &y6_buf, 0, rows, cols);
        engine.batch_end();
        okq && g4 && g6
    };
    assert!(d32, "q32 PSO unavailable");
    let mut w432 = 0f32;
    let mut w632 = 0f32;
    for r in 0..rows {
        let g4 = unsafe { (yp4 as *const f32).add(r).read() };
        let g6 = unsafe { (yp6 as *const f32).add(r).read() };
        let d4 = (cq4[r] - g4).abs();
        let d6 = (cq6[r] - g6).abs();
        if d4 > w432 { w432 = d4; }
        if d6 > w632 { w632 = d6; }
    }
    eprintln!("[probe] q32 gemv4 CPU-vs-GPU worst={w432:.4e} | gemv6 CPU-vs-GPU worst={w632:.4e}");
    // decide the decode mirror target: CPU TILE GEMV (b=1) vs the per-256
    // batch-mmq kernels at B=1 (the q4 probe's host model, token 0 of the
    // same activations). NB: the per-256 runs also verify at batch=1.
    let mut g4_256 = vec![0f32; rows];
    let mut g6_256 = vec![0f32; rows];
    let xq8_256_bytes = (cols / 256) * 260;
    let (qpx, qsx) = mk(xq8_256_bytes);
    let qx_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(qpx, qsx) }).expect("qx");
    let (ypx, ysx) = mk(rows * 4);
    let yx_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(ypx, ysx) }).expect("yx");
    let (yp6x, ys6x) = mk(rows * 4);
    let y6x_buf = engine.register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(yp6x, ys6x) }).expect("y6x");
    let ok256 = {
        assert!(engine.batch_open());
        let _ = engine.batch_quantize_q8_k(&x1_buf, 0, &qx_buf, 0, cols, 1);
        let q_okp = engine.batch_quantize_q8_k_pos(&x1_buf, 0, &qx_buf, 0, cols, 1);
        // NOTE: separate outputs — a shared y would let the q6 pass clobber
        // the q4 result (the earlier 3839 ghost).
        let g4 = q_okp && engine.batch_gemm_q4_k_mmq(&w_buf, 0, &qx_buf, 0, &yx_buf, 0, rows, cols, 1);
        let g6 = engine.batch_gemm_q6_k_mmq(&w6_buf, 0, &qx_buf, 0, &y6x_buf, 0, rows, cols, 1);
        engine.batch_end();
        q_okp && g4 && g6
    };
    assert!(ok256, "per-256 B=1 dispatch failed");
    let mut d4_256 = 0f32;
    let mut d6_256 = 0f32;
    for r in 0..rows {
        let g4 = unsafe { (ypx as *const f32).add(r).read() };
        let g6 = unsafe { (yp6x as *const f32).add(r).read() };
        d4_256 = d4_256.max((cq4[r] - g4).abs());
        d6_256 = d6_256.max((cq6[r] - g6).abs());
    }
    eprintln!("[probe] DECODE-TARGET check: gemv4-tile vs per-256-mmq worst={d4_256:.4e} | gemv6-tile vs per-256-mmq worst={d6_256:.4e}");
    // quantizer receipt for the pos convention: GPU qs bytes vs host ints
    let iscalep = 127.0f32 / (x1.iter().fold(0.0f32, |a, v| a.max(v.abs())));
    let mut qpos_mis = 0usize;
    for j in 0..cols {
        let qh: i8 = rint_i32(x1[j] * iscalep) as i8;
        let qg = unsafe { *((qpx as *const i8).add(4 + j)) }; // TileQ8K.d at 0..4
        if qh != qg { if qpos_mis < 4 { eprintln!("[probe] qpos j={j} host={qh} gpu={qg} x={:+.4}", x1[j]); } qpos_mis += 1; }
    }
    eprintln!("[probe] pos-quant mismatches={qpos_mis}/256 (GPU d={})", unsafe { (qpx as *const f32).read() });
    for r in 0..rows.min(4) {
        eprintln!(
            "[probe] dtarget r{r}: gemv_cpu={:+.4} gpu256={:+.4}",
            cq4[r], unsafe { (ypx as *const f32).add(r).read() }
        );
    }
    unsafe { libc::munmap(qpx as *mut libc::c_void, qsx) };
    unsafe { libc::munmap(ypx as *mut libc::c_void, ysx) };
    unsafe { libc::munmap(qp2 as *mut libc::c_void, qsz2) };
    unsafe { libc::munmap(yp4 as *mut libc::c_void, ysz4) };
    unsafe { libc::munmap(yp6 as *mut libc::c_void, ysz6) };
    unsafe { libc::munmap(xp1 as *mut libc::c_void, xsz1) };
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            (sign << 31) | (113 << 23) | (mant << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// P5 arm-b parity: flash prefill attention vs the CPU textbook serial
/// attention. Two shapes: (2 rows, 2 priors) covers the row-guard tile and a
/// single softmax block; (5 rows, 200 priors) covers multi-block online
/// softmax with mixed per-row causality inside one 4-row tile.
#[test]
#[ignore]
fn attn_flash_matches_cpu_reference() {
    std::env::set_var("SOFUU_GPU_ATTN_FLASH", "1");
    for (b, priors, first_pos) in [(2usize, 2usize, 0usize), (5usize, 200usize, 197usize)] {
        let nh = 32usize;
        let nkv = 8usize;
        let hd = 64usize;
        let mut state = 0xfeedbeefcafeu64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) % 2003) as f32 / 1000.0f32 - 1.0f32
        };
        // the KV cache must hold every position the rows attend to
        let kv_len = first_pos + b;
        let q: Vec<f32> = (0..b * nh * hd).map(|_| rand()).collect();
        let kc: Vec<f32> = (0..nkv * kv_len * hd).map(|_| rand()).collect();
        let vc: Vec<f32> = (0..nkv * kv_len * hd).map(|_| rand()).collect();
        let posb: Vec<u32> = (0..b).map(|t| (first_pos + t) as u32).collect();
        let mut expected = vec![0f32; b * nh * hd];
        for row in 0..b {
            let pos = posb[row] as usize;
            for h in 0..nh {
                let kvh = h * nkv / nh;
                let qo = (row * nh + h) * hd;
                let mut mx = f32::NEG_INFINITY;
                let mut scores = vec![0f32; pos + 1];
                for t in 0..=pos {
                    let ko = (kvh * kv_len + t) * hd;
                    let s: f32 = (0..hd).map(|i| q[qo + i] * kc[ko + i]).sum();
                    scores[t] = s / (hd as f32).sqrt();
                    if scores[t] > mx { mx = scores[t]; }
                }
                let esum: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
                for i in 0..hd {
                    let mut acc = 0f32;
                    for t in 0..=pos {
                        let vo = (kvh * kv_len + t) * hd;
                        acc += vc[vo + i] * ((scores[t] - mx).exp() / esum);
                    }
                    expected[(row * nh + h) * hd + i] = acc;
                }
            }
        }
        let page = 16384usize;
        let mk = |bytes: usize| -> *mut u8 {
            let p = unsafe {
                libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
            };
            assert!(p != libc::MAP_FAILED);
            p as *mut u8
        };
        let alloc = |bytes: usize| bytes.div_ceil(page) * page;
        let engine = MetalEngine::new().expect("engine");
        let mut ptrs: Vec<(*mut u8, usize)> = Vec::new();
        let mut sub: Vec<(*mut u8, MetalBuffer)> = Vec::new();
        for f in [&q, &kc, &vc, &expected] {
            let sz = alloc(f.len() * 4);
            let p = mk(sz);
            ptrs.push((p, sz));
            unsafe { std::ptr::copy_nonoverlapping(f.as_ptr() as *const u8, p, f.len() * 4) };
            let buf = engine.register_zerocopy_only(unsafe {
                std::slice::from_raw_parts_mut(p, sz)
            }).expect("reg");
            sub.push((p, buf));
        }
        let pos_sz = alloc(b * 4);
        let posptr = mk(pos_sz);
        ptrs.push((posptr, pos_sz));
        for (t, v) in posb.iter().enumerate() {
            unsafe { (posptr as *mut u32).add(t).write_unaligned(*v) };
        }
        let posbuf = engine.register_zerocopy_only(unsafe {
            std::slice::from_raw_parts_mut(posptr, pos_sz)
        }).expect("reg pos");
        let out_buf = engine.register_zerocopy_only(unsafe {
            std::slice::from_raw_parts_mut(sub[3].0, alloc(expected.len() * 4))
        }).expect("reg out");
        let dispatched = {
            assert!(engine.batch_open());
            let ok = engine.batch_llama_attn_flash_b4(
                &sub[0].1, 0,
                &sub[1].1, 0,
                &sub[2].1, 0,
                &out_buf, 0,
                &posbuf, 0,
                b,
                hd as u32, nh as u32, nkv as u32, kv_len as u32,
            );
            engine.batch_end();
            ok
        };
        assert!(dispatched, "flash attn dispatch unavailable");
        let mut worst = 0f32;
        let mut wi = 0usize;
        for (i, e) in expected.iter().enumerate() {
            let g = unsafe { (sub[3].0 as *const f32).add(i).read() };
            let d = (g - e).abs();
            if d > worst { worst = d; wi = i; }
            assert!(d <= 2e-3, "flash attn mismatch b={b} priors={priors} idx {i}: gpu={g:.6} cpu={e:.6}");
        }
        println!("[attn_flash] b={b} priors={priors} first_pos={first_pos} worst|d|={worst:.3e} at idx {wi} — VALUES PASS");
        for (p, sz) in &ptrs {
            unsafe { libc::munmap(*p as *mut libc::c_void, *sz) };
        }
    }
}

/// P5 arm-c go/no-go: the simdgroup-matrix q4_K tile vs the CPU dot and vs
/// the SIMT tiled class, on the 0.6B ffn_gate shape (3072 x 1024, B=128).
/// fp16 staging is a different arithmetic class than the f32 SIMT path, so
/// the reference tolerance is the fp16 rounding class (2e-4 relative), not
/// the SIMT 1e-5 — the text gate is what judges the class change.
#[test]
#[ignore]
fn gemm_q4_k_mma_matches_cpu_and_rates() {
    use std::time::Instant;
    let rows = 3072usize;
    let blocks = 4usize;
    let cols = blocks * 256;
    let row_bytes = blocks * 144;
    let B = 128usize;

    let mut state = 0xA24BAED4963EE407u64;
    let mut rand = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let f16_bytes = |v: f32| -> [u8; 2] {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = (((b >> 23) & 0xFF) as i32 - 127 + 15).clamp(1, 30) as u16;
        let man = (b >> 13) & 0x3FF;
        (sign | (exp << 10) | (man as u16)).to_le_bytes()
    };
    let mut w = vec![0u8; rows * row_bytes];
    for r in 0..rows {
        for b in 0..blocks {
            let off = r * row_bytes + b * 144;
            w[off..off + 2].copy_from_slice(&f16_bytes(0.0625f32));
            w[off + 2..off + 4].copy_from_slice(&f16_bytes(0.0005f32));
            for j in 0..12 {
                w[off + 4 + j] = ((rand() & 0x3F) as u8) | (((rand() & 3) as u8) << 6);
            }
            for j in 0..128 {
                w[off + 16 + j] = (rand() % 255) as u8;
            }
        }
    }
    let toks: Vec<Vec<f32>> = (0..B)
        .map(|t| {
            (0..cols)
                .map(|i| ((i * 7 + t * 3) % 23) as f32 * 0.25 - 2.5)
                .collect()
        })
        .collect();
    let mut y_cpu = vec![0f32; B * rows];
    for t in 0..B {
        for r in 0..rows {
            let off = r * row_bytes;
            y_cpu[t * rows + r] = unsafe {
                crate::infer::quant::vec_dot_raw_q4_k(w[off..].as_ptr(), toks[t].as_ptr(), blocks)
            };
        }
    }
    let page = 16384usize;
    let mk = |bytes: usize| -> *mut u8 {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(p != libc::MAP_FAILED);
        p as *mut u8
    };
    let w_alloc = w.len().div_ceil(page) * page;
    let w_ptr = mk(w_alloc);
    let wb: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(w_ptr, w.len()) };
    wb.copy_from_slice(&w);
    let engine = MetalEngine::new().expect("engine");
    let w_buf = engine.register_zerocopy_only(wb).expect("w");
    let slab_f32 = B * cols + B * rows;
    let slab_alloc = (slab_f32 * 4).div_ceil(page) * page;
    let slab_ptr = mk(slab_alloc);
    let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(slab_ptr, slab_alloc) };
    let slab_buf = engine.register_zerocopy_only(slab).expect("slab");
    for t in 0..B {
        unsafe {
            std::ptr::copy_nonoverlapping(
                toks[t].as_ptr() as *const u8,
                slab_ptr.add(t * cols * 4),
                cols * 4,
            )
        };
    }
    let y_off = B * cols * 4;

    std::env::set_var("SOFUU_GPU_GEMM_MMA", "1");
    let dispatched = {
        assert!(engine.batch_open());
        let ok = engine.batch_gguf_gemm_mma(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B);
        engine.batch_end();
        ok
    };
    assert!(dispatched, "gemm_q4_k_mma PSO unavailable (shader compile?)");
    let y_gpu: &[f32] =
        unsafe { std::slice::from_raw_parts(slab_ptr.add(y_off) as *const f32, B * rows) };
    let mut worst_rel = 0f32;
    let mut wi = 0usize;
    for i in 0..B * rows {
        let a = y_gpu[i];
        let b = y_cpu[i];
        assert!(a.is_finite(), "non-finite MMA output at {i}");
        let rel = (a - b).abs() / b.abs().max(1.0);
        if rel > worst_rel {
            worst_rel = rel;
            wi = i;
        }
        // fp16 staging: weights up to ~59 in this fixture, so a half ulp per
        // term lands at the 1e-3-relative class. An indexing defect is >=1e-1.
        assert!(rel <= 2e-3, "MMA parity broken at {i}: gpu={a:.6e} cpu={b:.6e} rel={rel:.2e}");
    }
    println!("[mma] rows={rows} cols={cols} B={B} worst_rel={worst_rel:e} at {wi} — PARITY PASS (fp16 class)");

    let mut bench = |label: &str, which: u8| {
        let reps = 5usize;
        assert!(engine.batch_open());
        match which {
            1 => { engine.batch_gguf_gemm_mma(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B); }
            2 => { engine.batch_gguf_gemm_tiled(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B); }
            _ => { engine.batch_gguf_gemm(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B); }
        }
        engine.batch_end();
        let t0 = Instant::now();
        for _ in 0..reps {
            assert!(engine.batch_open());
            let ok = match which {
                1 => engine.batch_gguf_gemm_mma(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B),
                2 => engine.batch_gguf_gemm_tiled(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B),
                _ => engine.batch_gguf_gemm(12, &w_buf, 0, &slab_buf, 0, &slab_buf, y_off, rows, cols, B),
            };
            engine.batch_end();
            assert!(ok, "{label} dispatch failed");
        }
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        let wt = (rows * row_bytes) as f64;
        let flop = 2.0 * rows as f64 * cols as f64 * B as f64;
        println!(
            "[mma-rate] {label:8} {:.3} ms  {:.2} TFLOP/s  {:.1} GB/s",
            per * 1e3,
            flop / per / 1e12,
            wt / per / 1e9
        );
    };
    bench("shipped", 0);
    bench("tiled", 2);
    bench("mma", 1);
}

/// Arm-E probe: the tiled class at the REAL 0.6B prefill shapes, to see how
/// much of the 489 ms pass the GEMMs actually explain. Each shape is timed
/// alone (5 reps) after a warmup, buffers reused, no CPU reference.
#[test]
#[ignore]
fn bench_tiled_real_shapes() {
    use std::time::Instant;
    std::env::set_var("SOFUU_GPU_GEMM_TILED", "1");
    std::env::set_var("SOFUU_GPU_GEMM_WIDE", "1");
    let engine = MetalEngine::new().expect("engine");
    let page = 16384usize;
    let mk = |bytes: usize| -> *mut u8 {
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(p != libc::MAP_FAILED);
        p as *mut u8
    };
    let shapes: [(usize, usize); 5] = [
        (6144, 1024), (3072, 1024), (1024, 2048), (1024, 3072), (1024, 4096),
    ];
    let max_rows = 6144usize;
    let max_cols = 4096usize;
    const B: usize = 128;
    // x lives at 0 and needs B*cols; y starts at a page boundary after it and
    // needs rows*B. Sizing this from offsets alone (the first version used 32 KB
    // for a 3 MB output) let the kernel write past the mapping.
    let y_off = ((B * max_cols * 4).div_ceil(page)) * page;
    let slab_bytes = (((y_off + max_rows * B * 4).div_ceil(page)) * page).max(page);
    let sp = mk(slab_bytes);
    let slab: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(sp, slab_bytes) };
    for i in 0..slab.len() { slab[i] = 0; }
    let s_buf = engine.register_zerocopy_only(slab).expect("slab");
    let w_page = (((max_rows * (max_cols / 256) * 144).div_ceil(page)) * page).max(page);
    let wp = mk(w_page);
    let w: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(wp, w_page) };
    {
        let mut s = 1234567u64;
        for i in 0..w.len() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            w[i] = (s >> 33) as u8;
        }
    }
    let w_buf = engine.register_zerocopy_only(w).expect("w");
    // Interleaved, min-of-N. Under this machine's load a sequential A/B drifts
    // 2x between variants (the 32-row tile itself measured 0.41, 0.72 and 0.48
    // TFLOP/s on the same shape in three runs), so every variant is sampled in
    // every round and the reported number is its MINIMUM.
    const ROUNDS: usize = 7;
    for (rows, cols) in shapes {
        let flop = 2.0 * rows as f64 * cols as f64 * B as f64;
        let mut best = [f64::INFINITY; 4];
        let names = ["32row", "64row", "r8x8", "mma64"];
        for round in 0..ROUNDS {
            for (vi, name) in names.iter().enumerate() {
                if round == 0 { continue; } // skip warm-up round
                const REP: usize = 8;
                let t0 = Instant::now();
                assert!(engine.batch_open());
                let mut ok = true;
                for _ in 0..REP {
                ok = match *name {
                    "32row" => engine.batch_gguf_gemm_tiled(12, &w_buf, 0, &s_buf, 0, &s_buf, y_off, rows, cols, B),
                    "64row" => engine.batch_gguf_gemm_tiled_wide(12, &w_buf, 0, &s_buf, 0, &s_buf, y_off, rows, cols, B),
                    "r8x8" => engine.batch_gguf_gemm_tiled_r88(&w_buf, &s_buf, &s_buf, y_off, rows, cols, B),
                    _ => engine.batch_gguf_gemm_mma64(&w_buf, &s_buf, &s_buf, y_off, rows, cols, B),
                };
                if !ok { break; }
                }
                engine.batch_end();
                assert!(ok, "{} refused {rows}x{cols}", name);
                let s = t0.elapsed().as_secs_f64() / REP as f64;
                if s < best[vi] { best[vi] = s; }
            }
        }
        println!(
            "[real-shape] ty=12 {rows}x{cols} B={B}: {}",
            (0..4).map(|vi| format!("{} {:.2} TF/s", names[vi], flop / best[vi] / 1e12)).collect::<Vec<_>>().join(" | ")
        );
    }
    println!("--- machine peaks (no memory traffic) ---");
    {
        // 16 cores x 128 lanes, 512 iters x 8 FMA = 4096 FMA/lane = 8192 FLOP/lane
        let o: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(mk(page), page) };
        for i in 0..o.len() { o[i] = 0; }
        let o_buf = engine.register_zerocopy_only(o).expect("out");
        let blocks = 16usize * 8;          // saturate 16 cores
        let lanes = blocks * 128;
        let flop_fma = lanes as f64 * 8192.0 * 8.0 * 2.0;
        // mma: 512 iters x 4 acc x (8*8*8 MAC) per simdgroup = 1M MAC per simdgroup
        let flop_mma = (blocks * 4) as f64 * 8192.0 * 4.0 * (8.0 * 8.0 * 8.0) * 2.0;
        let flop_fma16 = lanes as f64 * 4096.0 * 16.0 * 2.0;
        for (name, flop, f) in [
            ("peak_fma", flop_fma, 0usize),
            ("peak_fma16", flop_fma16, 2usize),
            ("peak_mma", flop_mma, 1usize),
        ] {
            let mut best = f64::INFINITY;
            for _ in 0..6 {
                let t0 = Instant::now();
                assert!(engine.batch_open());
                for _ in 0..4 {
                    match f {
                        0 => assert!(engine.peak_fma(&o_buf, blocks)),
                        2 => assert!(engine.peak_fma16(&o_buf, blocks)),
                        _ => assert!(engine.peak_mma(&o_buf, blocks)),
                    }
                }
                engine.batch_end();
                let s = t0.elapsed().as_secs_f64() / 4.0;
                if s < best { best = s; }
            }
            println!("[peak] {name}: {:.3} ms -> {:.2} TFLOP/s", best * 1e3, flop / best / 1e12);
        }
    }
    unsafe { libc::munmap(sp as *mut libc::c_void, slab_bytes); }
    unsafe { libc::munmap(wp as *mut libc::c_void, w_page); }
}




/// THE FINAL PIN: 1c attention VALUES at the P2 tail-chunk micro-shape
/// (b=2 rows, 2 priors, 32 heads / 8 kv / hd 64). The CPU mirror is the
/// textbook serial attention: causal softmax(q·k/√hd) × v over priors.
/// Deviation beyond f32-noise = the divergence the KV diff attributed to
/// the attention output. Run: cargo test -- --ignored --test-threads=1
#[test]
#[ignore]
fn attn_1c_values_micro_shape() {
    let nh = 32usize;
    let nkv = 8usize;
    let hd = 64usize;
    let b = 2usize;
    let priors = 2usize;
    let mut state = 0xfeedbeefcafeu64;
    let mut rand = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) % 2003) as f32 / 1000.0f32 - 1.0f32
    };
    let q: Vec<f32> = (0..b * nh * hd).map(|_| rand()).collect();
    let kc: Vec<f32> = (0..nkv * priors * hd).map(|_| rand()).collect();
    let vc: Vec<f32> = (0..nkv * priors * hd).map(|_| rand()).collect();
    let posb: Vec<u32> = (0..b).map(|t| (priors - b + t) as u32).collect();
    let mut expected = vec![0f32; b * nh * hd];
    for row in 0..b {
        let pos = posb[row] as usize;
        for h in 0..nh {
            let kvh = h * nkv / nh;
            let qo = (row * nh + h) * hd;
            let mut mx = f32::NEG_INFINITY;
            let mut scores = vec![0f32; pos + 1];
            for t in 0..=pos {
                let ko = (kvh * priors + t) * hd;
                let s: f32 = (0..hd).map(|i| q[qo + i] * kc[ko + i]).sum();
                scores[t] = s / (hd as f32).sqrt();
                if scores[t] > mx { mx = scores[t]; }
            }
            let esum: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
            for i in 0..hd {
                let mut acc = 0f32;
                for t in 0..=pos {
                    let vo = (kvh * priors + t) * hd;
                    acc += vc[vo + i] * ((scores[t] - mx).exp() / esum);
                }
                expected[(row * nh + h) * hd + i] = acc;
            }
        }
    }
    let page = 16384usize;
    let mk = |bytes: usize| -> *mut u8 {
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(p != libc::MAP_FAILED);
        p as *mut u8
    };
    let alloc = |bytes: usize| bytes.div_ceil(page) * page;
    let engine = MetalEngine::new().expect("engine");
    let mut ptrs: Vec<(*mut u8, usize)> = Vec::new();
    let mut sub: Vec<(*mut u8, MetalBuffer, usize)> = Vec::new();
    for f in [&q, &kc, &vc, &expected] {
        let sz = alloc(f.len() * 4);
        let p = mk(sz);
        ptrs.push((p, sz));
        unsafe { std::ptr::copy_nonoverlapping(f.as_ptr() as *const u8, p, f.len() * 4) };
        let buf = engine.register_zerocopy_only(unsafe {
            std::slice::from_raw_parts_mut(p, sz)
        }).expect("reg");
        sub.push((p, buf, f.len()));
    }
    let pos_sz = alloc(b * 4);
    let posptr = mk(pos_sz);
    ptrs.push((posptr, pos_sz));
    for (t, v) in posb.iter().enumerate() {
        unsafe { (posptr as *mut u32).add(t).write_unaligned(*v) };
    }
    let posbuf = engine.register_zerocopy_only(unsafe {
        std::slice::from_raw_parts_mut(posptr, pos_sz)
    }).expect("reg pos");
    let q_buf = (&sub[0].1, &sub[1].1, &sub[2].1, &sub[3].1);
    let (kc_buf, vc_buf, out_buf) = (q_buf.1, q_buf.2, q_buf.3);
    let dispatched = {
        assert!(engine.batch_open());
        let ok = engine.batch_llama_attn_1c_b4(
            q_buf.0, 0,
            kc_buf, 0,
            vc_buf, 0,
            out_buf, 0,
            &posbuf, 0,
            b,
            hd as u32, nh as u32, nkv as u32, priors as u32,
            None,
        );
        engine.batch_end();
        ok
    };
    assert!(dispatched, "attn_1c PSO unavailable");
    // read back the out region and compare
    let mut worst = 0f32;
    let mut wi = 0usize;
    for (i, e) in expected.iter().enumerate() {
        let g = unsafe { (sub[3].0 as *const f32).add(i).read() };
        let d = (g - e).abs();
        if d > worst { worst = d; wi = i; }
        assert!(d <= 1e-3, "attn_1c value mismatch at idx {i}: gpu={g:.6} cpu={e:.6}");
    }
    println!("[attn_1c] micro-shape worst|d|={worst:.3e} at idx {wi} — VALUES PASS");
    for (p, sz) in &ptrs {
        unsafe { libc::munmap(*p as *mut libc::c_void, *sz) };
    }
}

extern "C" {
    fn sofuu_metal_buffer_contents(buf: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
}

pub unsafe fn buffer_contents_raw(buf: &MetalBuffer) -> *mut u8 {
    sofuu_metal_buffer_contents(buf.handle) as *mut u8
}




// ── generalised GQA shapes: table + refusal regression ─────────────────────
// The shape a chunk's attention runs on is chosen by sofuu_attn_gqa_pick from
// the table in metal_engine.mm. Both are read here through the C ABI, so these
// tests exercise the SHIPPED rule rather than a Rust copy of it.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn gqa_shape_table_covers_every_supported_ratio() {
    let n = unsafe { sofuu_metal_attn_gqa_count() } as i32;
    assert!(n >= 12, "shape table shrank to {n} entries");
    let all: u32 = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
    for i in 0..n {
        let th = unsafe { sofuu_metal_attn_gqa_threads(i) };
        let rg = unsafe { sofuu_metal_attn_gqa_rg(i) };
        let ra = unsafe { sofuu_metal_attn_gqa_ratio(i) };
        assert!((32..=1024).contains(&th) && th % 32 == 0, "entry {i}: {th} threads");
        assert!((1..=64).contains(&rg), "entry {i}: {rg} rows/threadgroup");
        assert!(ra >= 1, "entry {i}: ratio {ra}");
        // rows per threadgroup must be at least the warps available for them
        assert!(rg * 32 >= th, "entry {i}: rg {rg} cannot host {th} threads");
    }
    // the shipped 2:1 default must stay on the shape it was measured on
    assert_eq!(unsafe { sofuu_metal_attn_gqa_pick(2, all, 1) }, 1);
    // every ratio the fleet ships has a form, and the form's compile-time R
    // equals the ratio (a mismatch would read another head's K/V)
    for ratio in [1u32, 2, 3, 4, 8] {
        let idx = unsafe { sofuu_metal_attn_gqa_pick(ratio, all, 1) };
        assert!(idx >= 0, "ratio {ratio} has no GQA shape");
        assert_eq!(unsafe { sofuu_metal_attn_gqa_ratio(idx) }, ratio);
    }
    // a ratio with no form must fall back to the per-head kernel, never to a
    // near-miss shape
    for ratio in [0u32, 5, 6, 7, 9, 16, 64, 65] {
        assert_eq!(
            unsafe { sofuu_metal_attn_gqa_pick(ratio, all, 1) },
            -1,
            "ratio {ratio} must not dispatch a GQA shape"
        );
    }
    // 1:1 shares no K/V between heads, so it is opt-in until a 1:1 model has
    // been measured: the default rule must not select it.
    let one = (0..n)
        .find(|i| unsafe { sofuu_metal_attn_gqa_ratio(*i) } == 1)
        .expect("1:1 shape missing from the table");
    assert_eq!(unsafe { sofuu_metal_attn_gqa_pick(1, all, 0) }, -1);
    assert_eq!(unsafe { sofuu_metal_attn_gqa_pick(1, all, 1) }, one);
    // a 384-thread ceiling (the reason the 256-thread 4:1 form exists): the
    // rule may only pick shapes that fit, and 4:1 must still get one.
    let mut small: u32 = 0;
    for i in 0..n {
        if unsafe { sofuu_metal_attn_gqa_threads(i) } <= 384 {
            small |= 1u32 << i;
        }
    }
    for ratio in [1u32, 2, 3, 4, 8] {
        let idx = unsafe { sofuu_metal_attn_gqa_pick(ratio, small, 1) };
        if idx >= 0 {
            assert!(
                unsafe { sofuu_metal_attn_gqa_threads(idx) } <= 384,
                "ratio {ratio} picked entry {idx} above the ceiling"
            );
        }
    }
    let r4 = unsafe { sofuu_metal_attn_gqa_pick(4, small, 1) };
    assert!(r4 >= 0, "4:1 lost its <=384-thread form");
    assert!(unsafe { sofuu_metal_attn_gqa_threads(r4) } <= 384);
}

// ── the two failure modes this table exists to stop ─────────────────────────
// 1. A threadgroup wider than the pipeline's own reported ceiling is SILENTLY
//    DROPPED by Metal in release (stale scratch, no error). The rule must only
//    ever return a form the dispatch can actually host: on this box the
//    pipelines report 896 for the 1024-thread _pf form, so it is not in the
//    live mask and must never come back - not even as a fallback.
// 2. A ratio whose forms are all unusable must be REPORTED, not folded into the
//    "-1" that means "this ratio has no form at all" (per-head is the design
//    only in that second case). That conflation is exactly what let a dead
//    shape table hide behind the per-head kernel.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn gqa_picker_never_leaves_the_live_mask() {
    let n = unsafe { sofuu_metal_attn_gqa_count() } as i32;
    assert!(n > 0 && n < 32, "shape table grew past the 32-bit mask: {n}");
    let all: u32 = (1u32 << n) - 1;
    // The rule is mask-driven: a one-shape mask selects that shape and nothing
    // else, so a pinned-but-dead shape can never be dispatched.
    for i in 0..n {
        let ratio = unsafe { sofuu_metal_attn_gqa_ratio(i) };
        assert!(ratio >= 1, "entry {i} has no ratio");
        assert_eq!(
            unsafe { sofuu_metal_attn_gqa_pick(ratio, 1u32 << i, 1) },
            i,
            "a mask holding only entry {i} (ratio {ratio}) must select entry {i}"
        );
    }
    // A GPU that reports 896 threads/group (this one) puts only the sub-1024
    // shapes in the mask. Every ratio must still get a form, it must be in the
    // mask, and it must not be a 1024-thread one.
    let mut m896: u32 = 0;
    for i in 0..n {
        if unsafe { sofuu_metal_attn_gqa_threads(i) } <= 896 {
            m896 |= 1u32 << i;
        }
    }
    for ratio in [1u32, 2, 3, 4, 8] {
        let idx = unsafe { sofuu_metal_attn_gqa_pick(ratio, m896, 1) };
        assert!(idx >= 0, "ratio {ratio} lost its <=896-thread form");
        assert!(
            m896 & (1u32 << idx) != 0,
            "ratio {ratio} picked entry {idx}, which is not in the 896-thread mask"
        );
        let th = unsafe { sofuu_metal_attn_gqa_threads(idx) };
        assert!(
            th <= 896,
            "ratio {ratio} picked a {th}-thread form under an 896-thread mask"
        );
    }
    // Every (ratio, mask) pair returns an in-mask entry or -1; never an
    // out-of-mask one (the mode-pin path this rule replaced could do that by
    // mapping the env number straight to a table index).
    let mut state = 0x2545f4914f6cdd1du64;
    for _ in 0..256 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mask = (state >> 33) as u32 & all;
        for ratio in [1u32, 2, 3, 4, 8] {
            let idx = unsafe { sofuu_metal_attn_gqa_pick(ratio, mask, 1) };
            if idx >= 0 {
                assert!(idx < n, "mask {mask:#x} ratio {ratio}: out-of-table {idx}");
                assert!(
                    mask & (1u32 << idx) != 0,
                    "mask {mask:#x} ratio {ratio}: picked dead entry {idx}"
                );
                assert_eq!(unsafe { sofuu_metal_attn_gqa_ratio(idx) }, ratio);
            }
        }
    }
    // A dead preferred form falls THROUGH to the next live form of the ratio -
    // it must not become a per-head dispatch while a paired form is live.
    let nr1_rg32_and_nr4 = (1u32 << 0) | (1u32 << 3);
    let fell = unsafe { sofuu_metal_attn_gqa_pick_ex(2, nr1_rg32_and_nr4, 1) };
    assert_eq!(fell, 0, "2:1 must fall through to its live form, got {fell}");
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn gqa_picker_reports_a_ratio_whose_shapes_are_all_dead() {
    let n = unsafe { sofuu_metal_attn_gqa_count() } as i32;
    let all: u32 = (1u32 << n) - 1;
    // Every ratio the fleet ships has forms in the table. An empty live mask
    // therefore means "none usable" (-2): a configuration failure the
    // dispatcher prints, not the silent per-head of the "-1" path.
    for ratio in [1u32, 2, 3, 4, 8] {
        assert_eq!(
            unsafe { sofuu_metal_attn_gqa_pick_ex(ratio, 0, 1) },
            -2,
            "ratio {ratio}: forms exist but none is live - must be reported, not silent"
        );
    }
    // The 1:1 form is opt-in: with SOFUU_ATTN_GQA_ALL off it is unreachable, so
    // the ratio has forms and none usable -> reported, not "no such ratio".
    assert_eq!(unsafe { sofuu_metal_attn_gqa_pick_ex(1, all, 0) }, -2);
    assert!(unsafe { sofuu_metal_attn_gqa_pick_ex(1, all, 1) } >= 0);
    // A ratio the table genuinely has no form for is NOT a failure: the
    // per-head kernel is the correct owner and nothing is printed.
    for ratio in [5u32, 6, 7, 9, 16, 32] {
        assert_eq!(
            unsafe { sofuu_metal_attn_gqa_pick_ex(ratio, all, 1) },
            -1,
            "ratio {ratio} has no table form - per-head is the design"
        );
    }
    // pick() keeps the historical index-or--1 contract for both cases.
    assert_eq!(unsafe { sofuu_metal_attn_gqa_pick(2, 0, 1) }, -1);
    assert_eq!(unsafe { sofuu_metal_attn_gqa_pick(5, all, 1) }, -1);
}


// ── every generalised shape against the per-head kernel, on real hardware ──
// The paired-K/V kernel exists in one instantiation per (R, NR, RG) triple and
// nothing in the model fleet exercises most of them, so each one is dispatched
// here at several head_dims and compared against the shape the whole engine was
// built on. The GQA form computes the same per-(row, head) softmax chain as
// llama_attn_pre_row, so the outputs must match to f32 noise; any difference
// means the pairing changed the math for a (ratio, head_dim) that no model in
// the fleet would have caught.
//   SOFUU_ATTN_GQA_ONLY pins the shape under test, because the auto rule picks
//   the widest live form and a 256-thread entry never wins on a GPU that allows
//   1024. Run: cargo test --release -p sofuu-core --lib -- --ignored --test-threads=1
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
#[ignore]
fn gqa_each_shape_matches_per_head_kernel() {
    // (table index, n_kv_heads, head_dim). Covers every new entry at a head_dim
    // from each divisibility class: 64 and 96 are not multiples of 128, the only
    // value the shipped model was ever measured at.
    let cases: [(i32, usize, usize); 14] = [
        (1, 4, 128),  // shipped 2:1 default, the reference shape
        (2, 4, 96),   // 2:1, 256 threads, hd 96
        (5, 4, 128),  // 4:1 512
        (6, 4, 64),   // 4:1 256  <- the <=384-thread form, hd 64
        (6, 4, 96),   // 4:1 256, hd 96
        (7, 4, 128),  // 4:1, 4 rows per warp
        (8, 4, 64),   // 3:1 512
        (9, 4, 96),   // 3:1 256
        (10, 4, 128), // 8:1
        (10, 4, 64),  // 8:1, hd 64
        (0, 4, 128),  // 2:1 pairing-only shape
        (3, 4, 96),   // 2:1, 4 rows per warp
        (11, 4, 128), // 1:1 opt-in form: it shares no K/V, so it is correct to
        (11, 4, 64),  //   prove here but never auto-selected without a 1:1 model
    ];
    let rows = 32usize;
    let max_seq = 64usize;
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut rand = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) % 2003) as f32 / 1000.0f32 - 1.0f32
    };
    let page = 16384usize;
    let alloc = |bytes: usize| bytes.div_ceil(page) * page;
    let mk = |bytes: usize| -> *mut u8 {
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE,
                       libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert!(p != libc::MAP_FAILED);
        p as *mut u8
    };
    let mut checked = 0usize;
    let mut skipped: Vec<i32> = Vec::new();
    for &(idx, nkv, hd) in &cases {
        let ratio = unsafe { sofuu_metal_attn_gqa_ratio(idx) } as usize;
        if ratio == 0 {
            skipped.push(idx);
            continue;
        }
        let nh = nkv * ratio;
        let q: Vec<f32> = (0..rows * nh * hd).map(|_| rand()).collect();
        let kc: Vec<f32> = (0..nkv * max_seq * hd).map(|_| rand()).collect();
        let vc: Vec<f32> = (0..nkv * max_seq * hd).map(|_| rand()).collect();
        let posb: Vec<u32> = (0..rows).map(|t| (max_seq - rows + t) as u32).collect();
        // one engine per arm: the shape pin is read once, at init
        let run = |pin: Option<i32>| -> (Vec<f32>, i32) {
            match pin {
                Some(i) => {
                    std::env::set_var("SOFUU_ATTN_GQA_ONLY", i.to_string());
                    std::env::set_var("SOFUU_ATTN_GQA_ALL", "1");
                    std::env::remove_var("SOFUU_ATTN_GQA");
                }
                None => {
                    std::env::set_var("SOFUU_ATTN_GQA", "0");
                    std::env::remove_var("SOFUU_ATTN_GQA_ONLY");
                }
            }
            let engine = MetalEngine::new().expect("engine");
            let mut ptrs: Vec<(*mut u8, usize)> = Vec::new();
            let mut bufs: Vec<MetalBuffer> = Vec::new();
            for f in [&q, &kc, &vc] {
                let sz = alloc(f.len() * 4);
                let p = mk(sz);
                ptrs.push((p, sz));
                unsafe { std::ptr::copy_nonoverlapping(f.as_ptr() as *const u8, p, f.len() * 4) };
                bufs.push(engine
                    .register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(p, sz) })
                    .expect("reg"));
            }
            let out_sz = alloc(rows * nh * hd * 4);
            let outp = mk(out_sz);
            ptrs.push((outp, out_sz));
            let out_buf = engine
                .register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(outp, out_sz) })
                .expect("reg out");
            let pos_sz = alloc(rows * 4);
            let posptr = mk(pos_sz);
            ptrs.push((posptr, pos_sz));
            for (t, v) in posb.iter().enumerate() {
                unsafe { (posptr as *mut u32).add(t).write_unaligned(*v) };
            }
            let pos_buf = engine
                .register_zerocopy_only(unsafe { std::slice::from_raw_parts_mut(posptr, pos_sz) })
                .expect("reg pos");
            assert!(engine.batch_open());
            let ok = engine.batch_llama_attn_row(
                &bufs[0], 0, &bufs[1], 0, &bufs[2], 0, &out_buf, 0, &pos_buf, 0,
                rows, hd as u32, nh as u32, nkv as u32, max_seq as u32,
            );
            engine.batch_end();
            assert!(ok, "shape {idx} (ratio {ratio}, hd {hd}): dispatch refused");
            let last = unsafe { sofuu_metal_attn_gqa_last() };
            let mut got = vec![0f32; rows * nh * hd];
            unsafe {
                std::ptr::copy_nonoverlapping(outp as *const f32, got.as_mut_ptr(), got.len())
            };
            for (p, sz) in &ptrs {
                unsafe { libc::munmap(*p as *mut libc::c_void, *sz) };
            }
            (got, last)
        };
        let (ref_out, ref_last) = run(None);
        assert_eq!(ref_last, -1, "per-head arm dispatched GQA shape {ref_last}");
        let (got, last) = run(Some(idx));
        std::env::remove_var("SOFUU_ATTN_GQA_ONLY");
        std::env::remove_var("SOFUU_ATTN_GQA_ALL");
        std::env::remove_var("SOFUU_ATTN_GQA");
        if last != idx {
            // the threadgroup guard refused this pipeline on this GPU: a real
            // outcome, but it leaves the shape unverified here.
            println!("[gqa-parity] shape {idx} (ratio {ratio}, hd {hd}): pipeline unavailable on this GPU - skipped");
            skipped.push(idx);
            continue;
        }
        let mut worst = 0f32;
        let mut wi = 0usize;
        let mut nbad = 0usize;
        for i in 0..got.len() {
            let d = (got[i] - ref_out[i]).abs();
            if got[i].to_bits() != ref_out[i].to_bits() {
                nbad += 1;
            }
            if d > worst {
                worst = d;
                wi = i;
            }
        }
        println!(
            "[gqa-parity] shape {idx} ratio {ratio} hd {hd} nh {nh}: worst|d|={worst:.3e} at {wi} ({nbad}/{} lanes differ)",
            got.len()
        );
        assert!(
            worst <= 1.2e-7,
            "shape {idx} (ratio {ratio}, hd {hd}) diverges from the per-head kernel: worst|d|={worst:.3e} at {wi}"
        );
        checked += 1;
    }
    assert!(checked >= 8, "only {checked} shapes verified (skipped {skipped:?})");
    println!("[gqa-parity] {checked} shapes match the per-head kernel; skipped {skipped:?}");
}

// ── batched-GEMM column floor - regression gate ─────────────────────────────
// The floor is the contract between the host dispatch guard and the kernels'
// block arithmetic. Too coarse and a whole model class falls back to one
// dispatch per token (SmolLM2 q8_0 at cols=960 ran 3x slower than the CPU
// tile path); too fine and a kernel silently drops the tail of every row.
// Both directions are wrong answers, so both are pinned here.
#[test]
fn gemm_cols_floor_matches_the_kernel_block_size() {
    // q4_0 is the 18-byte / 32-value block family: same floor as q8_0, and
    // the qwen2.5-0.5b q4_0 class depends on it (before the q4_0 kernel that
    // model walked one GEMV per token per projection).
    assert_eq!(gemm_cols_floor(GGML_TYPE_Q4_0), Some(32));
    assert_eq!(gemm_cols_floor(GGML_TYPE_Q8_0), Some(32));
    assert_eq!(gemm_cols_floor(GGML_TYPE_Q4_K), Some(256));
    assert_eq!(gemm_cols_floor(GGML_TYPE_Q6_K), Some(256));
    // No batched GEMM exists for anything else: refuse, never fall through
    // to a kernel that would compute a partial dot product.
    assert_eq!(gemm_cols_floor(GGML_TYPE_F32), None);
    assert_eq!(gemm_cols_floor(GGML_TYPE_F16), None);
    assert_eq!(gemm_cols_floor(99), None);
}

#[test]
fn gemm_cols_ok_accepts_the_fleet_widths_and_refuses_every_tail() {
    // every projection width in the model fleet (960 is SmolLM2's hidden
    // size: below the retired 256 floor but exactly 30 q8_0 blocks)
    for &cols in &[960usize, 1024, 2048, 2560, 3072, 6144, 8192, 10240] {
        assert!(gemm_cols_ok(GGML_TYPE_Q8_0, cols), "q8_0 must cover cols={cols}");
    }
    assert!(gemm_cols_ok(GGML_TYPE_Q8_0, 960));
    // ...and the same width is STILL refused for the 256-value super-block
    // types: relaxing q8_0 must not relax them.
    assert!(!gemm_cols_ok(GGML_TYPE_Q6_K, 960));
    assert!(!gemm_cols_ok(GGML_TYPE_Q4_K, 960));
    // non-multiple-of-32 stays refused (the kernel would drop the tail)
    for &cols in &[0usize, 1, 31, 1000, 2047, 4095] {
        assert!(!gemm_cols_ok(GGML_TYPE_Q8_0, cols), "q8_0 must refuse cols={cols}");
    }
    assert!(gemm_cols_ok(GGML_TYPE_Q4_K, 3072));
    assert!(!gemm_cols_ok(GGML_TYPE_Q4_K, 3200));
    assert!(!gemm_cols_ok(GGML_TYPE_Q6_K, 2080));
    assert!(!gemm_cols_ok(GGML_TYPE_F16, 2048));
    // q4_0 covers exactly the whole-32-value widths (the qwen2.5-0.5b's own
    // 896 and 4864 are both multiples of 32 but NOT of 256, so the q4_K floor
    // could never have carried them).
    for &cols in &[896usize, 960, 1024, 2080, 4864] {
        assert!(gemm_cols_ok(GGML_TYPE_Q4_0, cols), "q4_0 must cover cols={cols}");
    }
    for &cols in &[0usize, 1, 31, 1000, 2047, 4095] {
        assert!(!gemm_cols_ok(GGML_TYPE_Q4_0, cols), "q4_0 must refuse cols={cols}");
    }
}

/// The multi-section kernels must put the SUBTILE token offset into the store
/// index: `yr[(t0 + k) * y_stride]`, not `yr[k * y_stride]`.
///
/// This is pinned at the source because the defect it guards is invisible to
/// every per-call test that dispatches fewer than GEMM_TS+1 tokens, and it is
/// invisible on paper too - the weight/section cursor is correct either way.
/// What it does: with GEMM_TS = 8 every subtile > 0 wrote its 8 accumulators
/// over subtile 0's slots, so slots 0..7 held whichever subtile won the race
/// and slots 8..B-1 were never written at all (stale slab read back as
/// logits). That is a 10-17 max|dlogit| error on a 39-token prefill at ANY
/// column count, which is what the retired "q8_0 is wrong off the 256 grid"
/// hold was really seeing (cols 1024 was only ever measured on the mma64
/// twin, which always had the term). Numeric receipt:
/// gemm_q8_0_multi_matches_dequant_oracle below.
#[test]
fn gemm_multi_kernels_store_the_subtile_token_offset() {
    fn kernel_body<'a>(all: &'a str, name: &str) -> &'a str {
        let start = match all.find(name) {
            Some(s) => s,
            None => panic!("{name} missing from kernels.metal"),
        };
        let rest = &all[start..];
        let end = match rest[1..].find("\nkernel void ") {
            Some(i) => i + 1,
            None => rest.len(),
        };
        &rest[..end]
    }
    const KERNELS: &str = include_str!("../../../../src/metal/kernels.metal");
    for name in ["kernel void gemm_q8_0_multi(", "kernel void gemm_q6_k_multi("] {
        let body = kernel_body(KERNELS, name);
        assert!(
            body.contains("uint t0 = tg.y * GEMM_TS;"),
            "{name} must keep the GEMM_TS subtile grouping"
        );
        for k in 0..8 {
            let want = format!("yr[(size_t)(t0 + {k}) * sec_ystr]");
            assert!(
                body.contains(&want),
                "{name} must store acc{k} at {want} - a store without t0 \
                 clobbers subtile 0 and never writes tokens 8..B-1"
            );
        }
        assert!(
            !body.contains("yr[0] = acc0"),
            "{name} stores acc0 without the subtile token offset again"
        );
    }
    // the entry must no longer refuse q8_0 off the 256-value grid: the real
    // column requirement is whole 32-value q8_0 blocks (gemm_cols_ok).
    const SRC: &str = include_str!("metal.rs");
    let start = SRC
        .find("pub fn batch_gguf_gemm_multi(")
        .expect("batch_gguf_gemm_multi missing");
    let rest = &SRC[start..];
    let end = rest[1..]
        .find("\n    pub fn ")
        .map(|i| i + 1)
        .unwrap_or(rest.len());
    let body = &rest[..end];
    assert!(
        !body.contains("cols % 256 != 0"),
        "batch_gguf_gemm_multi must not re-impose a 256-value hold on q8_0: \
         gemm_q8_0_multi is exact at 960/992/2080/3200 (oracle) and at any B"
    );
}

/// The kernel that carries a non-q4_K chain projection must stride a weight
/// ROW by whole 32-value q8_0 blocks, which is exactly what the 32 floor
/// promises. The retired `(cols/256)*8*34` form agrees only on 256-divisible
/// widths, so pin the source text to the 32-block form (and pin the q6_K twin
/// to its 256-value blocks): an edit that changes the tile granularity without
/// changing the host floor fails here instead of on a 960-wide model.
#[test]
fn q8_0_tiled_fused_strides_rows_by_32_value_blocks() {
    fn kernel_body<'a>(all: &'a str, name: &str) -> &'a str {
        let start = match all.find(name) {
            Some(s) => s,
            None => panic!("{name} missing from kernels.metal"),
        };
        let rest = &all[start..];
        let end = match rest[1..].find("\nkernel void ") {
            Some(i) => i + 1,
            None => rest.len(),
        };
        &rest[..end]
    }
    const KERNELS: &str = include_str!("../../../../src/metal/kernels.metal");
    let q8 = kernel_body(KERNELS, "kernel void gemm_q8_0_tiled_fused(");
    assert!(
        q8.contains("cols / 32u") && q8.contains("nb32 * 34u"),
        "gemm_q8_0_tiled_fused must stride rows by (cols/32)*34 bytes"
    );
    assert!(
        !q8.contains("cols / 256"),
        "gemm_q8_0_tiled_fused still assumes 256-divisible cols"
    );
    let q6 = kernel_body(KERNELS, "kernel void gemm_q6_k_tiled_fused(");
    assert!(
        q6.contains("cols / 256"),
        "gemm_q6_k_tiled_fused must keep 256-value super-blocks"
    );
    // q4_0's twin: 18 bytes per 32 values, so a row is (cols/32)*18 bytes and
    // the dequant must be d*(nib-8) with byte j feeding value j (low nibble)
    // and value j+16 (high nibble) - the GGML order, not consecutive pairs.
    let q40 = kernel_body(KERNELS, "kernel void gemm_q4_0_tiled_fused(");
    assert!(
        q40.contains("* nb * 18u") && q40.contains("cols / 32u"),
        "gemm_q4_0_tiled_fused must stride rows by (cols/32)*18 bytes"
    );
    assert!(
        q40.contains("(v & 15u)") && q40.contains("(v < 16u)") && q40.contains("- 8.0f"),
        "gemm_q4_0_tiled_fused must dequant d*((nib)-8) with byte j = values j and j+16"
    );
}



// ─── Qwen2/2.5 bias epilogue: value + organisation gates ───────────────────
#[cfg(all(test, target_os = "macos"))]
mod chan_bias_tests {
    use super::*;

    struct Buf {
        ptr: *mut u8,
        len: usize,
    }

    impl Buf {
        fn new(bytes: usize) -> Self {
            let page = 16384usize;
            let len = ((bytes + page - 1) / page) * page;
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED, "arena mmap");
            Self { ptr: p as *mut u8, len }
        }
        fn f32s(&self) -> &[f32] {
            unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.len / 4) }
        }
        fn f32s_mut(&mut self) -> &mut [f32] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, self.len / 4) }
        }
        fn bytes(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
        }
    }

    /// T1 (bias kernel, 0 tolerance): the one-dispatch three-section bias
    /// epilogue is BIT-IDENTICAL to the CPU glue's f32 y[i] += bias[i], and a
    /// single-section dispatch produces exactly the same bytes for its
    /// section — the merge of q/k/v into one dispatch is an organisation
    /// change, not an arithmetic one.
    #[test]
    fn chan_bias_kernel_matches_cpu_loop() {
        let engine = MetalEngine::new().expect("metal engine");
        let b = 8usize;
        // q/k/v slot geometry of a small qwen2 layer (q 64 rows, k/v 16).
        let qa = 0usize; // bytes
        let ka = b * 64 * 4;
        let va = ka + b * 16 * 4;
        let total = va + b * 16 * 4;
        let mut act = Buf::new(total);
        for (i, v) in act.f32s_mut()[..total / 4].iter_mut().enumerate() {
            *v = (i % 37) as f32 * 0.125 - 2.0;
        }
        let sect: [(usize, usize, usize); 3] = [
            (0, qa, 64),
            (64, ka, 16),
            (80, va, 16),
        ];
        let mut wbuf = Buf::new(256);
        let bias_vals: Vec<f32> = (0..96).map(|i| (i % 11) as f32 * 0.25 - 1.0).collect();
        wbuf.f32s_mut()[..96].copy_from_slice(&bias_vals);
        let act_buf = engine.register_zerocopy_only(act.bytes()).expect("act reg");
        let w_buf = engine.register_zerocopy_only(wbuf.bytes()).expect("w reg");

        // CPU reference on a copy.
        let mut want = act.f32s()[..total / 4].to_vec();
        for (boff, yoff, rows) in sect {
            for t in 0..b {
                for j in 0..rows {
                    want[yoff / 4 + t * rows + j] += bias_vals[boff + j];
                }
            }
        }

        let secs: Vec<[u32; 4]> = sect
            .iter()
            .map(|(boff, yoff, rows)| [(*yoff / 4) as u32, *boff as u32, *rows as u32, 0])
            .collect();
        assert!(engine.batch_open());
        assert!(
            engine.batch_add_chan_bias(&act_buf, &w_buf, &secs, b),
            "three-section bias dispatch refused"
        );
        assert!(engine.batch_end(), "bias batch rejected");
        for (i, w) in want.iter().enumerate() {
            assert_eq!(
                act.f32s()[i].to_bits(),
                w.to_bits(),
                "chan_bias idx {i}: gpu {} cpu {}",
                act.f32s()[i],
                w
            );
        }

        // Organisation equivalence: one dispatch per section, same bytes.
        for (s, (_boff, _yoff, _rows)) in sect.iter().enumerate() {
            let mut solo = Buf::new(total);
            solo.f32s_mut()[..total / 4].copy_from_slice(&want); // start from the SUM
            let reg = engine.register_zerocopy_only(solo.bytes()).expect("solo reg");
            // subtract the section's bias first so one dispatch re-adds it.
            for t in 0..b {
                for j in 0..sect[s].2 {
                    let idx = sect[s].1 / 4 + t * sect[s].2 + j;
                    solo.f32s_mut()[idx] -= bias_vals[sect[s].0 + j];
                }
            }
            assert!(engine.batch_open());
            assert!(engine.batch_add_chan_bias(&reg, &w_buf, &secs[s..s + 1], b));
            assert!(engine.batch_end());
            for i in 0..total / 4 {
                assert_eq!(
                    solo.f32s()[i].to_bits(),
                    want[i].to_bits(),
                    "section {s} solo dispatch differs at {i}"
                );
            }
        }
        println!("[chan_bias] b={b} sections=3 — BIT-EXACT vs CPU glue (0 delta)");
    }

    /// The regression the arm was asked for: a BIASED sectioned GEMM (q, k, v in
    /// ONE dispatch) equals per-tensor GEMM + bias, bit for bit, and equals the
    /// CPU dequant dot + bias to the fp16-staging tolerance. Folding the bias
    /// into a sectioned epilogue is only correct if the section merge itself
    /// changes nothing — this pins both halves.
    #[test]
    fn biased_sectioned_gemm_equals_per_tensor_plus_bias() {
        let engine = MetalEngine::new().expect("metal engine");
        let cols = 256usize;
        let b = 8usize;
        let row_bytes = 144usize; // one q4_K super-block per row
        let (q_rows, k_rows, v_rows) = (64usize, 16usize, 32usize);

        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let f16_bytes = |v: f32| -> [u8; 2] {
            let bits = v.to_bits();
            let sign = ((bits >> 16) & 0x8000) as u16;
            let exp = ((((bits >> 23) & 0xFF) as i32) - 127 + 15).clamp(1, 30) as u16;
            let man = ((bits >> 13) & 0x3FF) as u16;
            (sign | (exp << 10) | man).to_le_bytes()
        };
        let sec_rows = [q_rows, k_rows, v_rows];
        let rows_total: usize = sec_rows.iter().sum();
        let mut w = Buf::new(rows_total * row_bytes);
        {
            let wb = w.f32s_mut();
            let _ = wb;
        }
        // write the q4_K blocks through the raw byte view
        let wbytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(w.ptr, rows_total * row_bytes)
        };
        for r in 0..rows_total {
            let off = r * row_bytes;
            wbytes[off..off + 2].copy_from_slice(&f16_bytes(0.0625f32));
            wbytes[off + 2..off + 4].copy_from_slice(&f16_bytes(0.0005f32));
            for j in 0..12 {
                wbytes[off + 4 + j] = ((rand() & 0x3F) as u8) | (((rand() & 3) as u8) << 6);
            }
            for j in 0..128 {
                wbytes[off + 16 + j] = (rand() % 255) as u8;
            }
        }

        let x: Vec<f32> = (0..b * cols)
            .map(|i| ((i % 23) as f32) * 0.125 - 1.5)
            .collect();
        let mut xb = Buf::new(b * cols * 4);
        xb.f32s_mut()[..b * cols].copy_from_slice(&x);

        // y slab: one compact [b][rows] region per section, laid out like the
        // qa/ka/va slots (byte offsets are what the mma8 sections take).
        let y_off: Vec<usize> = {
            let mut v = Vec::new();
            let mut cur = 0usize;
            for r in sec_rows {
                v.push(cur);
                cur += b * r * 4;
            }
            v
        };
        let y_bytes = y_off[2] + b * v_rows * 4;
        let mut y = Buf::new(y_bytes);

        // bias vectors for the three sections, contiguous f32 slots.
        let bias: Vec<f32> = (0..rows_total).map(|i| (i % 13) as f32 * 0.125 - 0.75).collect();
        let mut bbuf = Buf::new(rows_total * 4);
        bbuf.f32s_mut()[..rows_total].copy_from_slice(&bias);

        let w_buf = engine.register_zerocopy_only(w.bytes()).expect("w reg");
        let x_buf = engine.register_zerocopy_only(xb.bytes()).expect("x reg");
        let y_buf = engine.register_zerocopy_only(y.bytes()).expect("y reg");
        let bias_buf = engine.register_zerocopy_only(bbuf.bytes()).expect("bias reg");

        // base = min section offset (all 256-aligned here), secs[0] = delta.
        let mut wsecs: Vec<[u32; 4]> = Vec::new();
        let mut bsecs: Vec<[u32; 4]> = Vec::new();
        let mut cur = 0usize;
        let mut bias_off = 0usize;
        for (i, r) in sec_rows.iter().enumerate() {
            wsecs.push([cur as u32, 0, *r as u32, y_off[i] as u32]);
            bsecs.push([(y_off[i] / 4) as u32, bias_off as u32, *r as u32, 0]);
            cur += r * row_bytes;
            bias_off += r;
        }

        // (a) sectioned GEMM + one bias dispatch
        assert!(engine.batch_open());
        assert!(
            engine.batch_gguf_mma8_secs(&w_buf, 0, &x_buf, 0, &y_buf, cols, b, &wsecs),
            "sectioned mma8 refused (pipeline missing?)"
        );
        assert!(engine.batch_add_chan_bias(&y_buf, &bias_buf, &bsecs, b));
        assert!(engine.batch_end());
        let gemm_sectioned_bias: Vec<f32> = y.f32s()[..y_bytes / 4].to_vec();

        // (b) per-tensor GEMM (three dispatches) + bias added on the HOST
        let mut y2 = Buf::new(y_bytes);
        let y2_buf = engine.register_zerocopy_only(y2.bytes()).expect("y2 reg");
        assert!(engine.batch_open());
        for (i, r) in sec_rows.iter().enumerate() {
            assert!(engine.batch_gguf_mma8(
                &w_buf,
                cur_base_hack(i, &sec_rows, row_bytes),
                &x_buf,
                0,
                &y2_buf,
                y_off[i],
                *r,
                cols,
                b,
            ));
        }
        assert!(engine.batch_end());
        let mut want = y2.f32s()[..y_bytes / 4].to_vec();
        let mut bias_off = 0usize;
        for (i, r) in sec_rows.iter().enumerate() {
            for t in 0..b {
                for j in 0..*r {
                    want[y_off[i] / 4 + t * r + j] += bias[bias_off + j];
                }
            }
            bias_off += r;
        }
        for (i, w) in want.iter().enumerate() {
            assert_eq!(
                gemm_sectioned_bias[i].to_bits(),
                w.to_bits(),
                "bit mismatch at {i}: sectioned+bias {} per-tensor+bias {}",
                gemm_sectioned_bias[i],
                w
            );
        }

        // (c) CPU dequant dot + bias. The mma8 tile stages BOTH sides in fp16,
        // so the bound is the fp16 epsilon class measured against the sum of
        // |w*x| terms (this synthetic matrix cancels hard: |dot| ~ 0.2 over
        // terms ~ 3, so a bound relative to the RESULT would be meaningless and
        // a bound relative to the terms is the one that actually tests the
        // kernel).
        let mut bias_off = 0usize;
        let mut worst_rel = 0f32;
        let mut deq = vec![0f32; cols];
        for (i, r) in sec_rows.iter().enumerate() {
            for t in 0..b {
                for row in 0..*r {
                    let woff = cur_base_hack(i, &sec_rows, row_bytes) + row * row_bytes;
                    crate::infer::quant::dequantize_q4_k_row(
                        &wbytes[woff..woff + row_bytes],
                        &mut deq,
                    );
                    let mut term_abs = 0f32;
                    for j in 0..cols {
                        term_abs += (deq[j] * x[t * cols + j]).abs();
                    }
                    let cpu = unsafe {
                        crate::infer::quant::vec_dot_raw_q4_k(
                            wbytes[woff..].as_ptr(),
                            x[t * cols..].as_ptr(),
                            cols / 256,
                        )
                    } + bias[bias_off + row];
                    let gpu = gemm_sectioned_bias[y_off[i] / 4 + t * r + row];
                    let d = (gpu - cpu).abs();
                    let rel = d / term_abs.max(1e-6);
                    if rel > worst_rel {
                        worst_rel = rel;
                    }
                    assert!(
                        rel <= 2e-3,
                        "vs CPU: sec {i} t {t} row {row}: gpu {gpu:.7} cpu {cpu:.7} d {d:.3e} terms {term_abs:.4}"
                    );
                }
            }
            bias_off += r;
        }
        println!(
            "[bias_gemm] b={b} cols={cols} sections=q{}k{}v{} — sectioned+bias == per-tensor+bias BIT-EXACT, worst|d|/sum|w*x| vs CPU = {worst_rel:.3e} (fp16 class)",
            q_rows, k_rows, v_rows
        );
    }

    /// Byte offset of section i's weights in the synthetic arena (sections are
    /// packed in q,k,v order, each rows*144 bytes).
    fn cur_base_hack(i: usize, sec_rows: &[usize], row_bytes: usize) -> usize {
        sec_rows[..i].iter().map(|r| r * row_bytes).sum()
    }
}

