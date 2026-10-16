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
    fn sofuu_metal_batch_gemv_staged(
        ctx: *mut c_void,
        w_buf: *mut c_void, w_offset: usize,
        x: *const f32,
        y_buf: *mut c_void, y_offset: usize,
        rows: usize, cols: usize, dtype: i32,
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
