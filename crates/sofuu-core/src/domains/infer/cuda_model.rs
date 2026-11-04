// crates/sofuu-core/src/domains/infer/cuda_model.rs — the CUDA-resident
// transformer (PLAN-GPU-CUDA C2).
//
// One decode step runs end-to-end on the T4: every projection GEMV, RMSNorm,
// RoPE, KV write and attention reads/writes device memory. The host does the
// embedding lookup (the GGUF stays mmap'd and the CPU path needs it anyway)
// and, when asked, downloads the logits.
//
// Scope is deliberately the standard Llama block list, with separate-or-fused
// q/k/v and gate/up views (the loader already splits fused tensors into
// row views). Anything that changes the block graph — q/k norms, q/k/v biases,
// sliding-window layers, Gemma2's post-norms — is refused at load with a named
// reason, and the caller keeps the CPU engine. A refusal is visible; a model
// that silently computes the wrong graph is not.
//
// Threading: a CUDA context is current on the thread that created it. The
// loader has no cuCtxSetCurrent call, so this type records its owner thread and
// refuses a forward from any other one instead of issuing driver calls against
// a context that is not current there.

use super::cuda::{CudaEngine, DeviceBuf, Quant};
use super::gguf::GgufFile;
use super::model::{LayerWeights, ModelConfig, TensorView};
use super::quant::{self, GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0};

/// The device KV window follows the engine's own context (`--ctx`), which the
/// GGUF loader already caps at 4096 when the caller asks for 0. For the 1B's
/// 16x8x64 geometry that is ~268 MiB per tensor. `SOFUU_CUDA_CTX` can only
/// LOWER it (or set it when a future loader default changes); the preflight
/// refuses a request that would crowd the weights out of VRAM.

struct CudaLayer {
    attn_norm: DeviceBuf,
    attn_q: DeviceBuf,
    attn_q_quant: Quant,
    attn_k: DeviceBuf,
    attn_k_quant: Quant,
    attn_v: DeviceBuf,
    attn_v_quant: Quant,
    attn_output: DeviceBuf,
    attn_output_quant: Quant,
    ffn_norm: DeviceBuf,
    ffn_gate: DeviceBuf,
    ffn_gate_quant: Quant,
    ffn_up: DeviceBuf,
    ffn_up_quant: Quant,
    ffn_down: DeviceBuf,
    ffn_down_quant: Quant,
}

pub struct CudaModel {
    engine: CudaEngine,
    owner: std::thread::ThreadId,

    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    vocab: usize,
    q_dim: usize,
    kv_dim: usize,
    per_layer_kv: usize,
    eps: f32,
    rope_base: f32,
    rope_n_rot: usize,
    rope_mscale: f32,
    rope_interleaved: bool,
    rope_adjust: Vec<f32>,

    layers: Vec<CudaLayer>,
    output: DeviceBuf,
    output_quant: Quant,
    output_norm: DeviceBuf,

    cap: usize,
    pos: usize,

    k_cache: DeviceBuf,
    v_cache: DeviceBuf,

    x: DeviceBuf,
    x_norm: DeviceBuf,
    q: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    attn_out: DeviceBuf,
    proj: DeviceBuf,
    gate: DeviceBuf,
    up: DeviceBuf,
    act: DeviceBuf,
    down: DeviceBuf,
    logits_dev: DeviceBuf,
    rope_dev: DeviceBuf,

    logits_host: Vec<f32>,
    logit_bytes: Vec<u8>,
    rope_host: Vec<f32>,
}

impl CudaModel {
    /// Build the VRAM-resident model, or return the reason it cannot be built.
    /// The caller treats every `Err` as "keep the CPU engine" and prints it.
    pub fn from_gguf(
        gguf: &GgufFile,
        cfg: &ModelConfig,
        token_embd: &TensorView,
        output_norm: &TensorView,
        output: &TensorView,
        layers: &[LayerWeights],
        rope_adjust: &[f32],
    ) -> Result<Self, String> {
        let arch = gguf.get_str("general.architecture").unwrap_or("llama");
        if arch != "llama" {
            return Err(format!(
                "the CUDA block graph implements llama only (checkpoint arch '{arch}'); \
                 the CPU engine handles this model"
            ));
        }
        if cfg.sliding_window != 0 {
            return Err(format!(
                "sliding-window attention ({}) is not implemented by the CUDA attention kernel",
                cfg.sliding_window
            ));
        }
        if cfg.kv_ring != 0 {
            return Err("ring KV / KIVI streaming is a CPU-path contract".into());
        }
        if cfg.n_kv_heads == 0 || cfg.n_heads % cfg.n_kv_heads != 0 {
            return Err(format!(
                "GQA geometry n_heads={} n_kv_heads={} is not an integer group",
                cfg.n_heads, cfg.n_kv_heads
            ));
        }
        if cfg.head_dim == 0 || cfg.head_dim % 2 != 0 || cfg.head_dim > 256 {
            return Err(format!(
                "CUDA attention needs an even head_dim in 2..=256 (got {})",
                cfg.head_dim
            ));
        }
        // The KV cache is sized from key_length. A checkpoint whose value
        // width differs would need a second geometry; refusing is cheaper than
        // silently reading V rows at the wrong stride.
        if let Some(vlen) = gguf.get_u32(&format!("{arch}.attention.value_length")) {
            if vlen as usize != cfg.head_dim {
                return Err(format!(
                    "value_length {vlen} differs from key_length/head_dim {}; \
                     the CUDA KV cache assumes they are equal",
                    cfg.head_dim
                ));
            }
        }
        let n_rot = effective_n_rot(cfg);
        if n_rot % 2 != 0 {
            return Err(format!("RoPE width {n_rot} is not an even number of dims"));
        }
        if layers.len() != cfg.n_layers {
            return Err(format!(
                "{} layer records for {} blocks",
                layers.len(),
                cfg.n_layers
            ));
        }
        for (i, l) in layers.iter().enumerate() {
            if l.attn_q_bias.is_some()
                || l.attn_k_bias.is_some()
                || l.attn_v_bias.is_some()
            {
                return Err(format!(
                    "block {i} carries q/k/v biases; the CUDA GEMV path is bias-free (llama block graph)"
                ));
            }
            if l.attn_q_norm.is_some() || l.attn_k_norm.is_some() {
                return Err(format!(
                    "block {i} carries q/k RMSNorm; that graph is not wired on CUDA"
                ));
            }
        }

        let engine = CudaEngine::try_init().ok_or_else(|| {
            "no CUDA driver/device (is libcuda.so.1 present, and SOFUU_CUDA not 0?). \
             A static-musl binary cannot dlopen libcuda — use a glibc build for CUDA"
                .to_string()
        })?;
        let info = engine.device_info().clone();

        let dim = cfg.dim;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let cap = context_cap(cfg.max_seq_len);
        let per_layer_kv = cfg
            .n_kv_heads
            .checked_mul(cap)
            .and_then(|v| v.checked_mul(cfg.head_dim))
            .ok_or_else(|| "KV dimensions overflow".to_string())?;

        macro_rules! alloc_f32 {
            ($n:expr, $what:literal) => {
                engine
                    .alloc($n * std::mem::size_of::<f32>())
                    .map_err(|e| format!("{}: allocate {} floats: {e}", $what, $n))?
            };
        }

        let output_quant = tensor_quant(output, cfg.vocab_size, dim)?;
        let output_buf = upload_weight(
            &engine,
            gguf,
            output,
            "output/lm_head",
            cfg.vocab_size,
            dim,
        )?;
        let output_norm_buf = upload_norm(&engine, gguf, output_norm, dim, "output_norm")?;

        let mut cuda_layers = Vec::with_capacity(cfg.n_layers);
        let mut weight_bytes = output_buf.bytes + output_norm_buf.bytes;
        for l in layers.iter() {
            let attn_norm = upload_norm(&engine, gguf, &l.attn_norm, dim, "attn_norm")?;
            let attn_q = upload_weight(
                &engine,
                gguf,
                &l.attn_q,
                "attn_q",
                q_dim,
                dim,
            )?;
            let attn_k = upload_weight(
                &engine,
                gguf,
                &l.attn_k,
                "attn_k",
                kv_dim,
                dim,
            )?;
            let attn_v = upload_weight(
                &engine,
                gguf,
                &l.attn_v,
                "attn_v",
                kv_dim,
                dim,
            )?;
            let attn_output = upload_weight(
                &engine,
                gguf,
                &l.attn_output,
                "attn_output",
                dim,
                q_dim,
            )?;
            let ffn_norm = upload_norm(&engine, gguf, &l.ffn_norm, dim, "ffn_norm")?;
            let ffn_gate = upload_weight(
                &engine,
                gguf,
                &l.ffn_gate,
                "ffn_gate",
                cfg.hidden_dim,
                dim,
            )?;
            let ffn_up = upload_weight(
                &engine,
                gguf,
                &l.ffn_up,
                "ffn_up",
                cfg.hidden_dim,
                dim,
            )?;
            let ffn_down = upload_weight(
                &engine,
                gguf,
                &l.ffn_down,
                "ffn_down",
                dim,
                cfg.hidden_dim,
            )?;
            weight_bytes += attn_norm.bytes
                + attn_q.bytes
                + attn_k.bytes
                + attn_v.bytes
                + attn_output.bytes
                + ffn_norm.bytes
                + ffn_gate.bytes
                + ffn_up.bytes
                + ffn_down.bytes;
            cuda_layers.push(CudaLayer {
                attn_norm,
                attn_q,
                attn_q_quant: tensor_quant(&l.attn_q, q_dim, dim)?,
                attn_k,
                attn_k_quant: tensor_quant(&l.attn_k, kv_dim, dim)?,
                attn_v,
                attn_v_quant: tensor_quant(&l.attn_v, kv_dim, dim)?,
                attn_output,
                attn_output_quant: tensor_quant(&l.attn_output, dim, q_dim)?,
                ffn_norm,
                ffn_gate,
                ffn_gate_quant: tensor_quant(&l.ffn_gate, cfg.hidden_dim, dim)?,
                ffn_up,
                ffn_up_quant: tensor_quant(&l.ffn_up, cfg.hidden_dim, dim)?,
                ffn_down,
                ffn_down_quant: tensor_quant(&l.ffn_down, dim, cfg.hidden_dim)?,
            });
        }

        let kv_total = cfg
            .n_layers
            .checked_mul(per_layer_kv)
            .ok_or_else(|| "KV layer total overflows".to_string())?;
        let scratch = 4 * dim + 2 * q_dim + 2 * kv_dim + 3 * cfg.hidden_dim + cfg.vocab_size + cfg.head_dim;
        let need = weight_bytes
            .checked_add(2 * kv_total * 4)
            .and_then(|v| v.checked_add(scratch * 4))
            .ok_or_else(|| "VRAM estimate overflows".to_string())?;
        let vram = info.total_memory as usize;
        if vram > 0 && need > vram / 100 * 92 {
            return Err(format!(
                "needs ~{:.2} GiB (weights {:.2} + KV {:.2} + scratch) but {} reports {:.2} GiB; \
                 lower SOFUU_CUDA_CTX or use a smaller quant",
                need as f64 / (1024.0 * 1024.0 * 1024.0),
                weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                (2 * kv_total * 4) as f64 / (1024.0 * 1024.0 * 1024.0),
                info.name,
                vram as f64 / (1024.0 * 1024.0 * 1024.0)
            ));
        }

        let k_cache = engine
            .alloc(kv_total * 4)
            .map_err(|e| format!("KV cache K ({kv_total} floats): {e}"))?;
        let v_cache = engine
            .alloc(kv_total * 4)
            .map_err(|e| format!("KV cache V ({kv_total} floats): {e}"))?;

        let x = alloc_f32!(dim, "x");
        let x_norm = alloc_f32!(dim, "x_norm");
        let q = alloc_f32!(q_dim, "q");
        let k = alloc_f32!(kv_dim, "k");
        let v = alloc_f32!(kv_dim, "v");
        let attn_out = alloc_f32!(q_dim, "attn_out");
        let proj = alloc_f32!(dim, "proj");
        let gate = alloc_f32!(cfg.hidden_dim, "gate");
        let up = alloc_f32!(cfg.hidden_dim, "up");
        let act = alloc_f32!(cfg.hidden_dim, "act");
        let down = alloc_f32!(dim, "down");
        let logits_dev = alloc_f32!(cfg.vocab_size, "logits");
        let rope_dev = alloc_f32!(cfg.head_dim, "rope_table");

        eprintln!(
            "[cuda] {} cc{}.{} · {:.2} GiB VRAM · weights {:.2} GiB · KV {} rows ({:.0} MiB x2)",
            info.name,
            info.cc_major,
            info.cc_minor,
            vram as f64 / (1024.0 * 1024.0 * 1024.0),
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            cap,
            (kv_total * 4) as f64 / (1024.0 * 1024.0)
        );

        Ok(Self {
            engine,
            owner: std::thread::current().id(),
            dim,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            hidden_dim: cfg.hidden_dim,
            vocab: cfg.vocab_size,
            q_dim,
            kv_dim,
            per_layer_kv,
            eps: cfg.norm_eps,
            rope_base: cfg.rope_freq_base,
            rope_n_rot: n_rot,
            rope_mscale: if cfg.rope_mscale == 0.0 { 1.0 } else { cfg.rope_mscale },
            rope_interleaved: cfg.rope_interleaved,
            rope_adjust: rope_adjust.to_vec(),
            layers: cuda_layers,
            output: output_buf,
            output_quant,
            output_norm: output_norm_buf,
            cap,
            pos: 0,
            k_cache,
            v_cache,
            x,
            x_norm,
            q,
            k,
            v,
            attn_out,
            proj,
            gate,
            up,
            act,
            down,
            logits_dev,
            rope_dev,
            logits_host: vec![0.0; cfg.vocab_size],
            logit_bytes: vec![0u8; cfg.vocab_size * 4],
            rope_host: vec![0.0; cfg.head_dim],
        })
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits_host
    }

    pub fn reset(&mut self) {
        // Every read is bounded by the next pos+1, so stale rows are invisible.
        self.pos = 0;
    }

    /// Run one token. `x_host` is the embedding row (dim f32); `pos` is the
    /// absolute position, which is also the KV row this token writes.
    pub fn forward(
        &mut self,
        x_host: &[f32],
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), String> {
        if std::thread::current().id() != self.owner {
            return Err(
                "CUDA context belongs to another thread (load and generate on the same thread)"
                    .into(),
            );
        }
        if x_host.len() < self.dim {
            return Err(format!("embedding row has {} values, need {}", x_host.len(), self.dim));
        }
        if pos >= self.cap {
            return Err(format!(
                "position {pos} is past the {}-row CUDA KV cache; raise SOFUU_CUDA_CTX",
                self.cap
            ));
        }
        if pos != self.pos {
            return Err(format!(
                "non-contiguous position {pos}: the device cache holds rows 0..{}; a new \
                 request must call reset_cache() first",
                self.pos
            ));
        }

        let e = &self.engine;
        e.upload(self.x, f32_bytes(&x_host[..self.dim]))
            .map_err(|err| format!("upload embedding: {err}"))?;

        // The host table reproduces quant::rope_style_ex exactly: cos|sin for
        // this position, with the model's n_rot/mscale/frequency factors.
        quant::rope_table_rows(
            &mut self.rope_host,
            self.head_dim,
            self.rope_n_rot,
            self.rope_mscale,
            self.rope_base,
            &self.rope_adjust,
            &[pos],
        );
        e.upload(self.rope_dev, f32_bytes(&self.rope_host))
            .map_err(|err| format!("upload rope table: {err}"))?;

        let half = self.head_dim / 2;
        let sin_dev = DeviceBuf {
            ptr: self.rope_dev.ptr + (half * 4) as u64,
            bytes: self.rope_dev.bytes - half * 4,
        };
        let seq_len = pos + 1;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

        for (li, l) in self.layers.iter().enumerate() {
            e.rmsnorm(self.x, l.attn_norm, self.x_norm, self.dim, self.eps)
                .map_err(|err| format!("layer {li} attn_norm: {err}"))?;
            e.gemv(l.attn_q_quant, l.attn_q, self.x_norm, self.q, self.q_dim, self.dim)
                .map_err(|err| format!("layer {li} attn_q: {err}"))?;
            e.gemv(l.attn_k_quant, l.attn_k, self.x_norm, self.k, self.kv_dim, self.dim)
                .map_err(|err| format!("layer {li} attn_k: {err}"))?;
            e.gemv(l.attn_v_quant, l.attn_v, self.x_norm, self.v, self.kv_dim, self.dim)
                .map_err(|err| format!("layer {li} attn_v: {err}"))?;

            e.rope(
                self.q,
                self.rope_dev,
                sin_dev,
                self.n_heads,
                self.head_dim,
                self.rope_n_rot,
                self.rope_interleaved,
            )
            .map_err(|err| format!("layer {li} q rope: {err}"))?;
            e.rope(
                self.k,
                self.rope_dev,
                sin_dev,
                self.n_kv_heads,
                self.head_dim,
                self.rope_n_rot,
                self.rope_interleaved,
            )
            .map_err(|err| format!("layer {li} k rope: {err}"))?;

            let k_slice = layer_slice(self.k_cache, li, self.per_layer_kv);
            let v_slice = layer_slice(self.v_cache, li, self.per_layer_kv);
            e.kv_write(
                k_slice,
                v_slice,
                self.k,
                self.v,
                pos,
                self.n_kv_heads,
                self.head_dim,
                self.cap,
            )
            .map_err(|err| format!("layer {li} kv_write: {err}"))?;
            e.attn_decode(
                self.q,
                k_slice,
                v_slice,
                self.attn_out,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                seq_len,
                self.cap,
                scale,
            )
            .map_err(|err| format!("layer {li} attention: {err}"))?;
            e.gemv(
                l.attn_output_quant,
                l.attn_output,
                self.attn_out,
                self.proj,
                self.dim,
                self.q_dim,
            )
            .map_err(|err| format!("layer {li} attn_output: {err}"))?;
            e.add(self.x, self.proj, self.x, self.dim)
                .map_err(|err| format!("layer {li} attn residual: {err}"))?;

            e.rmsnorm(self.x, l.ffn_norm, self.x_norm, self.dim, self.eps)
                .map_err(|err| format!("layer {li} ffn_norm: {err}"))?;
            e.gemv(
                l.ffn_gate_quant,
                l.ffn_gate,
                self.x_norm,
                self.gate,
                self.hidden_dim,
                self.dim,
            )
            .map_err(|err| format!("layer {li} ffn_gate: {err}"))?;
            e.gemv(
                l.ffn_up_quant,
                l.ffn_up,
                self.x_norm,
                self.up,
                self.hidden_dim,
                self.dim,
            )
            .map_err(|err| format!("layer {li} ffn_up: {err}"))?;
            e.swiglu(self.gate, self.up, self.act, self.hidden_dim)
                .map_err(|err| format!("layer {li} swiglu: {err}"))?;
            e.gemv(
                l.ffn_down_quant,
                l.ffn_down,
                self.act,
                self.down,
                self.dim,
                self.hidden_dim,
            )
            .map_err(|err| format!("layer {li} ffn_down: {err}"))?;
            e.add(self.x, self.down, self.x, self.dim)
                .map_err(|err| format!("layer {li} ffn residual: {err}"))?;
        }

        if compute_logits {
            e.rmsnorm(self.x, self.output_norm, self.x_norm, self.dim, self.eps)
                .map_err(|err| format!("output_norm: {err}"))?;
            e.gemv(
                self.output_quant,
                self.output,
                self.x_norm,
                self.logits_dev,
                self.vocab,
                self.dim,
            )
            .map_err(|err| format!("lm_head: {err}"))?;
            // The launches above are asynchronous. Synchronize once per token,
            // where the host actually consumes the result; this also surfaces
            // any deferred kernel fault with the position that caused it.
            e.sync()
                .map_err(|err| format!("sync before logits: {err}"))?;
            e.download(&mut self.logit_bytes, self.logits_dev)
                .map_err(|err| format!("download logits: {err}"))?;
            for (i, c) in self.logit_bytes.chunks_exact(4).enumerate() {
                self.logits_host[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }

        self.pos = pos + 1;
        Ok(())
    }
}

fn effective_n_rot(cfg: &ModelConfig) -> usize {
    if cfg.rope_n_rot == 0 || cfg.rope_n_rot > cfg.head_dim {
        cfg.head_dim
    } else {
        cfg.rope_n_rot
    }
}

fn context_cap(max_seq: usize) -> usize {
    let want = std::env::var("SOFUU_CUDA_CTX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 64)
        .unwrap_or(max_seq);
    want.min(max_seq.max(1))
}

fn layer_slice(base: DeviceBuf, layer: usize, per_layer: usize) -> DeviceBuf {
    let off = layer * per_layer * std::mem::size_of::<f32>();
    DeviceBuf {
        ptr: base.ptr + off as u64,
        bytes: per_layer * std::mem::size_of::<f32>(),
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // Safe: f32 has no padding and every bit pattern is a valid u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn tensor_quant(tv: &TensorView, rows: usize, cols: usize) -> Result<Quant, String> {
    let q = match tv.tensor_type {
        GGML_TYPE_F32 => Quant::F32,
        GGML_TYPE_Q8_0 => Quant::Q8_0,
        GGML_TYPE_Q4_K => Quant::Q4K,
        GGML_TYPE_Q6_K => Quant::Q6K,
        other => {
            return Err(format!(
                "tensor type {} is not implemented by the CUDA kernels (F32/Q8_0/Q4_K/Q6_K)",
                quant::ggml_type_name(other)
            ))
        }
    };
    if q.row_bytes(cols).is_none() {
        return Err(format!(
            "{cols} values per row is not a whole number of {} blocks",
            quant::ggml_type_name(tv.tensor_type)
        ));
    }
    if tv.shape.len() != 2 || tv.shape[0] != cols || tv.shape[1] != rows {
        return Err(format!(
            "shape {:?} does not describe [{cols}, {rows}]",
            tv.shape
        ));
    }
    Ok(q)
}

fn raw_tensor<'a>(
    gguf: &'a GgufFile,
    tv: &TensorView,
    what: &str,
) -> Result<&'a [u8], String> {
    let start = gguf
        .data_offset
        .checked_add(tv.offset)
        .ok_or_else(|| format!("{what}: tensor offset overflows"))?;
    let end = start
        .checked_add(tv.size_bytes)
        .ok_or_else(|| format!("{what}: tensor size overflows"))?;
    gguf.as_bytes()
        .get(start..end)
        .ok_or_else(|| format!("{what}: bytes {start}..{end} are outside the GGUF file"))
}

fn upload_weight(
    engine: &CudaEngine,
    gguf: &GgufFile,
    tv: &TensorView,
    what: &str,
    rows: usize,
    cols: usize,
) -> Result<DeviceBuf, String> {
    let q = tensor_quant(tv, rows, cols).map_err(|e| format!("{what}: {e}"))?;
    let row_bytes = q.row_bytes(cols).unwrap();
    let want = row_bytes
        .checked_mul(rows)
        .ok_or_else(|| format!("{what}: {rows} rows x {row_bytes} B overflows"))?;
    if tv.size_bytes != want {
        return Err(format!(
            "{what}: tensor is {} B, expected {rows} x {row_bytes} = {want} B",
            tv.size_bytes
        ));
    }
    let data = raw_tensor(gguf, tv, what)?;
    let buf = engine
        .alloc(want)
        .map_err(|e| format!("{what}: allocate {want} B: {e}"))?;
    engine
        .upload(buf, data)
        .map_err(|e| format!("{what}: upload {want} B: {e}"))?;
    Ok(buf)
}

fn upload_norm(
    engine: &CudaEngine,
    gguf: &GgufFile,
    tv: &TensorView,
    dim: usize,
    what: &str,
) -> Result<DeviceBuf, String> {
    if tv.tensor_type != GGML_TYPE_F32 {
        return Err(format!(
            "{what}: RMSNorm weight is {}, not F32",
            quant::ggml_type_name(tv.tensor_type)
        ));
    }
    let want = dim * 4;
    if tv.size_bytes != want {
        return Err(format!(
            "{what}: norm is {} B, expected {dim} f32 = {want} B",
            tv.size_bytes
        ));
    }
    let data = raw_tensor(gguf, tv, what)?;
    let buf = engine
        .alloc(want)
        .map_err(|e| format!("{what}: allocate {want} B: {e}"))?;
    engine
        .upload(buf, data)
        .map_err(|e| format!("{what}: upload: {e}"))?;
    Ok(buf)
}
