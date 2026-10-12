// crates/sofuu-core/src/infer/metal.rs — Safe Rust wrapper for Apple Silicon Metal GPU Engine.

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
