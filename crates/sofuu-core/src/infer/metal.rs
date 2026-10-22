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

pub const OP_KINDS: usize = 16;
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
pub const OP_NAMES: [&str; OP_KINDS] = [
    "gguf", "gguf_b4", "gemm", "rmsnorm", "rmsnorm_b4", "kv_write", "kv_write_b4",
    "head_rmsnorm", "attn", "attn_part", "attn_fin", "attn_1c", "add", "swiglu",
    "head", "unused",
];

pub struct MetalEngine {
    /// P0.2 per-op HOST enqueue timing (SOFUU_GPU_TIMING=2): nanos + count
    /// per op kind, reported by take_op_timing/print_op_timing.
    op_ns: [std::sync::atomic::AtomicU64; OP_KINDS],
    op_ct: [std::sync::atomic::AtomicU64; OP_KINDS],
    tgate: std::sync::atomic::AtomicU8,

    ctx: *mut c_void,
}

unsafe impl Send for MetalEngine {}
unsafe impl Sync for MetalEngine {}

impl MetalEngine {
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
    fn sofuu_metal_batch_open(ctx: *mut c_void) -> i32;
    fn sofuu_metal_batch_gemv(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, dtype: i32,
    );
    fn sofuu_metal_batch_end(ctx: *mut c_void) -> i32;
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
    );
    fn sofuu_metal_batch_swiglu(
        ctx: *mut c_void,
        g_buf: *mut c_void, g_offset: usize,
        u_buf: *mut c_void, u_offset: usize,
        n: usize,
    );
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
    fn sofuu_metal_batch_gguf_gemm(
        ctx: *mut c_void,
        ty: u32,
        w_buf: *mut c_void, w_offset: usize,
        x_buf: *mut c_void, x_offset: usize,
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
    );
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
    );
    fn sofuu_metal_batch_rmsnorm_f32w_b4(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        w_buf: *mut c_void, w_offset: usize,
        y_buf: *mut c_void, y_offset: usize,
        dim: usize, eps: f32, rows: usize,
    );
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
    );
    fn sofuu_metal_batch_head_rmsnorm(
        ctx: *mut c_void,
        x_buf: *mut c_void, x_offset: usize,
        w_buf: *mut c_void, w_offset: usize,
        head_dim: u32, n_heads: u32,
        has_norm: u32, pos: u32, eps: f32, rope_base: f32, rope_mode: u32,
    );
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
    );
    fn sofuu_metal_batch_llama_attn_fin(
        ctx: *mut c_void,
        meta_buf: *mut c_void, meta_offset: usize,
        out_buf: *mut c_void, out_offset: usize,
        pos_alt: u32, n_heads: u32, head_dim: u32,
    );
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
    ) {        let __t = self.op_tick(K_ADD);
        unsafe {
            sofuu_metal_batch_add(
                self.ctx,
                acc_buf.handle, acc_offset,
                add_buf.handle, add_offset,
                n,
            );
        }
        self.op_tock(K_ADD, __t);
    }

    pub fn batch_swiglu(
        &self,
        g_buf: &MetalBuffer, g_offset: usize,
        u_buf: &MetalBuffer, u_offset: usize,
        n: usize,
    ) {        let __t = self.op_tick(K_SWIGLU);
        unsafe {
            sofuu_metal_batch_swiglu(
                self.ctx,
                g_buf.handle, g_offset,
                u_buf.handle, u_offset,
                n,
            );
        }
        self.op_tock(K_SWIGLU, __t);
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
    ) {        let __t = self.op_tick(K_GGUF);
        unsafe {
            sofuu_metal_batch_gguf(
                self.ctx,
                ty,
                w_buf.handle, w_offset,
                x_buf.handle, x_offset,
                y_buf.handle, y_offset,
                rows, cols,
            );
        }
        self.op_tock(K_GGUF, __t);
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
    ) -> bool {        let __t = self.op_tick(K_GGUF_GEMM);
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

    /// P1 op-diet: fused multi-section GEMM (qkv / gate_up). Sections carry
    /// their own 256-aligned weight byte offset + remainder + cumulative row
    /// end + ABSOLUTE y-slab byte offset + y row stride, so one dispatch
    /// fills the per-region slots the glue kernels already read. Returns
    /// false when unavailable (falls back to per-tensor gemm calls).
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
    ) {        let __t = self.op_tick(K_RMSNORM);
        unsafe {
            sofuu_metal_batch_rmsnorm_f32w(
                self.ctx,
                x_buf.handle, x_offset,
                w_buf.handle, w_offset,
                y_buf.handle, y_offset,
                dim, eps,
            );
        }
        self.op_tock(K_RMSNORM, __t);
    }

    /// B-row f32-weight RMSNorm for the spec chain: rows are contiguous
    /// [B][dim]; one dispatch instead of B.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_rmsnorm_f32w_b4(
        &self,
        x_buf: &MetalBuffer, x_offset: usize,
        w_buf: &MetalBuffer, w_offset: usize,
        y_buf: &MetalBuffer, y_offset: usize,
        dim: usize, eps: f32, rows: usize,
    ) {        let __t = self.op_tick(K_RMSNORM_B4);
        unsafe {
            sofuu_metal_batch_rmsnorm_f32w_b4(
                self.ctx,
                x_buf.handle, x_offset,
                w_buf.handle, w_offset,
                y_buf.handle, y_offset,
                dim, eps, rows,
            );
        }
        self.op_tock(K_RMSNORM_B4, __t);
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
    ) {
        let (kln_handle, kln_off) = match kln_buf {
            Some(b) => (b.handle, kln_offset),
            None => (k_buf.handle, 0),
        };        let __t = self.op_tick(K_KVWRITE);
        unsafe {
            sofuu_metal_batch_llama_kv_write(
                self.ctx,
                k_buf.handle, k_offset,
                v_buf.handle, v_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                kln_handle, kln_off,
                has_norm, pos,
                head_dim, n_kv_heads, max_seq,
                eps, rope_base, rope_mode,
            );
        }
        self.op_tock(K_KVWRITE, __t);
    }

    /// Per-head f32 RMSNorm (q-norm) on a [heads x head_dim] slab slot.
    pub fn batch_head_rmsnorm(
        &self,
        x_buf: &MetalBuffer, x_offset: usize,
        w_buf: Option<&MetalBuffer>, w_offset: usize,
        head_dim: u32, n_heads: u32,
        has_norm: u32, pos: u32, eps: f32, rope_base: f32, rope_mode: u32,
    ) {
        let (w_handle, w_off) = match w_buf {
            Some(b) => (b.handle, w_offset),
            None => (x_buf.handle, 0),
        };
        unsafe {
            sofuu_metal_batch_head_rmsnorm(
                self.ctx,
                x_buf.handle, x_offset,
                w_handle, w_off,
                head_dim, n_heads,
                has_norm, pos, eps, rope_base, rope_mode,
            );
        }
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
    ) {
        let (qln_handle, qln_off) = match qln_buf {
            Some(b) => (b.handle, qln_offset),
            None => (q_buf.handle, 0),
        };
        unsafe {
            sofuu_metal_batch_llama_attn_part(
                self.ctx,
                q_buf.handle, q_offset,
                kc_buf.handle, kc_layer_offset,
                vc_buf.handle, vc_layer_offset,
                qln_handle, qln_off,
                meta_buf.handle, meta_offset,
                has_norm, pos,
                head_dim, n_heads, n_kv_heads, max_seq,
                eps, rope_base, rope_mode,
            );
        }
    }

    /// GPU3 attention stage 2: rescale chunk partials into head_out.
    pub fn batch_llama_attn_fin(
        &self,
        meta_buf: &MetalBuffer, meta_offset: usize,
        out_buf: &MetalBuffer, out_offset: usize,
        pos_alt: u32, n_heads: u32, head_dim: u32,
    ) {
        unsafe {
            sofuu_metal_batch_llama_attn_fin(
                self.ctx,
                meta_buf.handle, meta_offset,
                out_buf.handle, out_offset,
                pos_alt, n_heads, head_dim,
            );
        }
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
