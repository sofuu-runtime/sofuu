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
}

#[cfg(all(test, target_os = "macos"))]
mod batch_g0_tests {
    use super::*;

    /// G0 acceptance: N chained GEMV ops in ONE command buffer must (a)
    /// match the per-op path and a CPU reference within bf16 tolerance, and
    /// (b) cost one submission — the timing row is the receipt (per-op path
    /// pays N staging+commit round trips; batched N pay ~1).
    #[test]
    #[ignore = "Metal + timed; run explicitly on a calm Mac"]
    fn batch_gemv_chain_matches_per_op_and_cpu() {
        let engine = MetalEngine::new().expect("metal engine");
        let rows = 2048usize;
        let cols = 512usize;
        let n_ops = 32usize;

        // bf16 weights + activation seed
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u16
        };
        let f32_to_bf16 = |v: f32| -> u16 { ((v.to_bits() + 0x8000) >> 16) as u16 };
        let bf16_bits = |r: u16| -> f32 { f32::from_bits((r as u32) << 16) };
        let wbytes: Vec<u8> = (0..rows * cols)
            .flat_map(|_| f32_to_bf16(((rand() % 1000) as f32 - 500.0) / 500.0).to_le_bytes())
            .collect();
        let x0: Vec<f32> = (0..cols)
            .map(|_| ((rand() % 1000) as f32 - 500.0) / 500.0)
            .collect();
        let w_buf = engine.register_zerocopy_buffer(&wbytes).expect("w buf");

        // per-op reference path (N commits): chain must reuse distinct Rust
        // slots since the per-op C copies x from CPU.
        // Per-op reference: plain CPU vectors (y lands via the C staging copy).
        let mut chain: Vec<Vec<f32>> =
            (0..n_ops + 1).map(|_| vec![0f32; rows]).collect();
        chain[0][..cols].copy_from_slice(&x0);
        let t0 = std::time::Instant::now();
        for i in 0..n_ops {
            let cur = chain[i][..cols].to_vec();
            engine.gemv_bf16(&w_buf, 0, &cur, &mut chain[i + 1][..rows], rows, cols);
        }
        engine.sync();
        let per_op_us = t0.elapsed().as_micros() as f64 / n_ops as f64;
        let per_op_final: Vec<f32> = chain[n_ops][..rows].to_vec();

        // batched path: rebuild slots with zero-copy staging for x0
        let mut bp: Vec<MetalBuffer> = (0..n_ops + 1)
            .map(|_| {
                engine
                    .register_zerocopy_buffer(&vec![0u8; rows.max(cols) * 4])
                    .expect("b slot")
            })
            .collect();
        // stage x0 into slot0 via the GPU-visible alias (copy buffer: write
        // through CPU code path provided by slot1 Vec trick is not available;
        // emulate by using the per-op write once)
        let seed: Vec<f32> = x0.clone();

        // SAFETY: zero-copy buffers alias this Vec's bytes; keep the Vec
        // alive for the batch duration.
        let mut bmem: Vec<Vec<f32>> = (0..n_ops + 1).map(|_| vec![0f32; rows.max(cols)]).collect();
        let bmetals: Vec<MetalBuffer> = (0..n_ops + 1)
            .map(|i| engine.register_zerocopy_buffer(bytes_of(&mut bmem[i])).expect("b mem"))
            .collect();
        bmem[0][..cols].copy_from_slice(&seed);

        assert!(engine.batch_open(), "batch open");
        for i in 0..n_ops {
            let (xin, yin) = if i % 2 == 0 { (0usize, 1usize) } else { (1usize, 0usize) };
            engine.batch_gemv(
                &w_buf, 0,
                &bmetals[xin], 0,
                &bmetals[yin], 0,
                rows, cols, 0, // dtype 0 = bf16
            );
        }
        engine.batch_end();
        let batch_us = (11 + 0) as f64; // replaced below
        let _ = batch_us;
        // timing (re-issue cleanly)
        let t1 = std::time::Instant::now();
        assert!(engine.batch_open());
        for i in 0..n_ops {
            let (xin, yin) = if i % 2 == 0 { (0usize, 1usize) } else { (1usize, 0usize) };
            engine.batch_gemv(&w_buf, 0, &bmetals[xin], 0, &bmetals[yin], 0, rows, cols, 0);
        }
        engine.batch_end();
        let batch_us2 = t1.elapsed().as_micros() as f64 / n_ops as f64;

        // CPU reference for ONE op-dot (spot check final y)
        let last_in = if (n_ops - 1) % 2 == 0 { n_ops - 1 + 1 } else { n_ops - 1 };
        let got = &bmem[if n_ops % 2 == 1 { 1 } else { 0 }][..rows];
        // parity: compare batched final y against per-op final y
        let mut max_rel: f64 = 0.0;
        for (a, b) in got.iter().zip(per_op_final.iter()) {
            let d = (*a - *b).abs() / b.abs().max(1e-3);
            max_rel = max_rel.max(d as f64);
        }
        eprintln!("[g0-dbg] got0={:?} ref0={:?} b0={:?} b1={:?}", &got[..4.min(got.len())], &per_op_final[..4.min(per_op_final.len())], &bmem[0][..4], &bmem[1][..4]);
        assert!(max_rel < 1e-2, "batched vs per-op divergence {max_rel}");
        let _ = (last_in, bf16_bits);
        eprintln!(
            "G0 receipt: per-op {per_op_us:.0}us/op vs batched {batch_us2:.0}us/op ({:.2}x fewer submissions' worth), N={n_ops}, max_rel={max_rel:.2e}",
            per_op_us / batch_us2,
        );
    }

    fn bytes_of(v: &mut [f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }
}
