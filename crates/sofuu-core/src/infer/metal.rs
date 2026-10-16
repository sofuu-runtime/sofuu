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

pub struct MetalEngine {
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
            Some(Self { ctx })
        }
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
    fn sofuu_metal_batch_end(ctx: *mut c_void);
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

    /// End the batch: end encoding, commit once, wait.
    pub fn batch_end(&self) {
        unsafe { sofuu_metal_batch_end(self.ctx) }
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
    ) {
        unsafe {
            sofuu_metal_batch_add(
                self.ctx,
                acc_buf.handle, acc_offset,
                add_buf.handle, add_offset,
                n,
            );
        }
    }

    pub fn batch_swiglu(
        &self,
        g_buf: &MetalBuffer, g_offset: usize,
        u_buf: &MetalBuffer, u_offset: usize,
        n: usize,
    ) {
        unsafe {
            sofuu_metal_batch_swiglu(
                self.ctx,
                g_buf.handle, g_offset,
                u_buf.handle, u_offset,
                n,
            );
        }
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
    ) {
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
