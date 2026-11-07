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

    // q8_1 activations for the dp4a GEMV path, sized for hidden_dim (the
    // widest activation vector any q4_K projection consumes).
    xq: DeviceBuf,
    xd: DeviceBuf,
    xsum: DeviceBuf,

    logits_host: Vec<f32>,
    logit_bytes: Vec<u8>,
    rope_host: Vec<f32>,

    // Batched-prefill scratch: eight token rows per buffer, token-major. The
    // kernels use one warp per token, so the cap of eight is a kernel contract,
    // not a tuning knob.
    xb: DeviceBuf,
    x_norm_b: DeviceBuf,
    q_b: DeviceBuf,
    k_b: DeviceBuf,
    v_b: DeviceBuf,
    attn_b: DeviceBuf,
    proj_b: DeviceBuf,
    gate_b: DeviceBuf,
    up_b: DeviceBuf,
    act_b: DeviceBuf,
    down_b: DeviceBuf,
    rope_b: DeviceBuf,
    rope_host_b: Vec<f32>,
}

/// Tokens per prefill chunk. Every b8 kernel has eight warps, one per token.
pub const PREFILL_BATCH: usize = 8;

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

        // Layout split for the arming receipt: with Q4_K_M checkpoints the
        // output/lm_head matrix is conventionally Q6_K, which is both larger
        // per value and slower per byte than the Q4_K block stack.
        let mut quant_bytes = [0usize; 4];
        add_quant_bytes(&mut quant_bytes, output);
        add_quant_bytes(&mut quant_bytes, output_norm);
        for l in layers.iter() {
            for tv in [
                &l.attn_norm,
                &l.attn_q,
                &l.attn_k,
                &l.attn_v,
                &l.attn_output,
                &l.ffn_norm,
                &l.ffn_gate,
                &l.ffn_up,
                &l.ffn_down,
            ] {
                add_quant_bytes(&mut quant_bytes, tv);
            }
        }

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
        // Scratch: the single-token row set plus the eight-row prefill set,
        // plus the q8_1 activation set (hidden int8 + two hidden/32 vectors).
        let single = 4 * dim + 2 * q_dim + 2 * kv_dim + 3 * cfg.hidden_dim;
        let q8_scratch =
            cfg.hidden_dim.max(dim).max(cfg.n_heads * cfg.head_dim) / 4
                + cfg.hidden_dim.max(dim).max(cfg.n_heads * cfg.head_dim) / 16;
        let scratch = single + cfg.vocab_size + cfg.head_dim + q8_scratch
            + PREFILL_BATCH * (single + cfg.head_dim);
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

        // q8_1 activation set: one int8 per value, one f32 scale and one i32
        // integer sum per 32 values. Sized for the widest vector that any
        // q4_K projection consumes.
        let q8_values = cfg.hidden_dim.max(dim).max(q_dim);
        let q8_blocks = q8_values / 32;
        let xq = engine
            .alloc(q8_values)
            .map_err(|e| format!("q8 activations: allocate {q8_values} B: {e}"))?;
        let xd = alloc_f32!(q8_blocks, "q8 scales");
        let xsum = alloc_f32!(q8_blocks, "q8 sums");

        let xb = alloc_f32!(PREFILL_BATCH * dim, "x_batch");
        let x_norm_b = alloc_f32!(PREFILL_BATCH * dim, "x_norm_batch");
        let q_b = alloc_f32!(PREFILL_BATCH * q_dim, "q_batch");
        let k_b = alloc_f32!(PREFILL_BATCH * kv_dim, "k_batch");
        let v_b = alloc_f32!(PREFILL_BATCH * kv_dim, "v_batch");
        let attn_b = alloc_f32!(PREFILL_BATCH * q_dim, "attn_batch");
        let proj_b = alloc_f32!(PREFILL_BATCH * dim, "proj_batch");
        let gate_b = alloc_f32!(PREFILL_BATCH * cfg.hidden_dim, "gate_batch");
        let up_b = alloc_f32!(PREFILL_BATCH * cfg.hidden_dim, "up_batch");
        let act_b = alloc_f32!(PREFILL_BATCH * cfg.hidden_dim, "act_batch");
        let down_b = alloc_f32!(PREFILL_BATCH * dim, "down_batch");
        let rope_b = alloc_f32!(PREFILL_BATCH * cfg.head_dim, "rope_batch");

        eprintln!(
            "[cuda] {} rev {} cc{}.{} · {:.2} GiB VRAM · weights {:.2} GiB \
             (f32 {:.1}, q8_0 {:.1}, q4_K {:.1}, q6_K {:.1} MiB) · KV {} rows ({:.0} MiB x2)",
            info.name,
            env!("SOFUU_BUILD"),
            info.cc_major,
            info.cc_minor,
            vram as f64 / (1024.0 * 1024.0 * 1024.0),
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            quant_bytes[0] as f64 / (1024.0 * 1024.0),
            quant_bytes[1] as f64 / (1024.0 * 1024.0),
            quant_bytes[2] as f64 / (1024.0 * 1024.0),
            quant_bytes[3] as f64 / (1024.0 * 1024.0),
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
            xq,
            xd,
            xsum,
            logits_host: vec![0.0; cfg.vocab_size],
            logit_bytes: vec![0u8; cfg.vocab_size * 4],
            rope_host: vec![0.0; cfg.head_dim],
            xb,
            x_norm_b,
            q_b,
            k_b,
            v_b,
            attn_b,
            proj_b,
            gate_b,
            up_b,
            act_b,
            down_b,
            rope_b,
            rope_host_b: vec![0.0; PREFILL_BATCH * cfg.head_dim],
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

    /// Quantize the activation set the dp4a GEMV path consumes. Lengths that
    /// are not a multiple of 32 belong to non-q4_K consumers, which read the
    /// f32 buffer instead, so they are a no-op here.
    fn quantize_act(&self, e: &CudaEngine, src: DeviceBuf, n: usize) -> Result<(), String> {
        if !n.is_multiple_of(32) {
            return Ok(());
        }
        e.quantize_q8_1(src, self.xq, self.xd, self.xsum, n)
            .map_err(|err| format!("quantize q8_1 ({n} values): {err}"))
    }

    /// GEMV that reads the q8_1 activations when the weight layout has a dp4a
    /// kernel, and the f32-activation kernel otherwise. `x` is ignored on the
    /// dp4a path, so the caller must quantize the matching buffer first.
    fn gemv_act(
        &self,
        e: &CudaEngine,
        quant: Quant,
        w: DeviceBuf,
        x: DeviceBuf,
        y: DeviceBuf,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        if quant == Quant::Q4K {
            e.gemv_q4_k_q8(w, self.xq, self.xd, self.xsum, y, rows, cols)
                .map_err(|err| format!("gemv q4_K x q8_1: {err}"))
        } else {
            e.gemv(quant, w, x, y, rows, cols)
                .map_err(|err| format!("gemv: {err}"))
        }
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
        // SOFUU_CUDA_PROFILE=1: one stream sync per phase so the host can time
        // attention, the output projection, the FFN and lm_head separately.
        // The syncs inflate each group by the driver round-trip, but at ~18 ms
        // per token the phase split is still the useful signal.
        let profile = std::env::var_os("SOFUU_CUDA_PROFILE").is_some();
        let mut prof_us = [0u128; 4];
        let mut t_group = std::time::Instant::now();

        for (li, l) in self.layers.iter().enumerate() {
            if profile {
                t_group = std::time::Instant::now();
            }
            e.rmsnorm(self.x, l.attn_norm, self.x_norm, self.dim, self.eps)
                .map_err(|err| format!("layer {li} attn_norm: {err}"))?;
            // q/k/v all read the same normed row, so quantize it once.
            self.quantize_act(e, self.x_norm, self.dim)
                .map_err(|err| format!("layer {li}: {err}"))?;
            self.gemv_act(
                e,
                l.attn_q_quant,
                l.attn_q,
                self.x_norm,
                self.q,
                self.q_dim,
                self.dim,
            )
            .map_err(|err| format!("layer {li} attn_q: {err}"))?;
            self.gemv_act(
                e,
                l.attn_k_quant,
                l.attn_k,
                self.x_norm,
                self.k,
                self.kv_dim,
                self.dim,
            )
            .map_err(|err| format!("layer {li} attn_k: {err}"))?;
            self.gemv_act(
                e,
                l.attn_v_quant,
                l.attn_v,
                self.x_norm,
                self.v,
                self.kv_dim,
                self.dim,
            )
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
            if profile {
                e.sync()
                    .map_err(|err| format!("layer {li} profile sync (attn): {err}"))?;
                prof_us[0] += t_group.elapsed().as_micros();
                t_group = std::time::Instant::now();
            }
            self.quantize_act(e, self.attn_out, self.q_dim)
                .map_err(|err| format!("layer {li}: {err}"))?;
            self.gemv_act(
                e,
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
            if profile {
                e.sync()
                    .map_err(|err| format!("layer {li} profile sync (o_proj): {err}"))?;
                prof_us[1] += t_group.elapsed().as_micros();
                t_group = std::time::Instant::now();
            }

            e.rmsnorm(self.x, l.ffn_norm, self.x_norm, self.dim, self.eps)
                .map_err(|err| format!("layer {li} ffn_norm: {err}"))?;
            self.quantize_act(e, self.x_norm, self.dim)
                .map_err(|err| format!("layer {li}: {err}"))?;
            self.gemv_act(
                e,
                l.ffn_gate_quant,
                l.ffn_gate,
                self.x_norm,
                self.gate,
                self.hidden_dim,
                self.dim,
            )
            .map_err(|err| format!("layer {li} ffn_gate: {err}"))?;
            self.gemv_act(
                e,
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
            self.quantize_act(e, self.act, self.hidden_dim)
                .map_err(|err| format!("layer {li}: {err}"))?;
            self.gemv_act(
                e,
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
            if profile {
                e.sync()
                    .map_err(|err| format!("layer {li} profile sync (ffn): {err}"))?;
                prof_us[2] += t_group.elapsed().as_micros();
                t_group = std::time::Instant::now();
            }
        }

        if compute_logits {
            e.rmsnorm(self.x, self.output_norm, self.x_norm, self.dim, self.eps)
                .map_err(|err| format!("output_norm: {err}"))?;
            if self.output_quant == Quant::Q4K {
                self.quantize_act(e, self.x_norm, self.dim)
                    .map_err(|err| format!("lm_head: {err}"))?;
            }
            self.gemv_act(
                e,
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
            if profile {
                prof_us[3] += t_group.elapsed().as_micros();
            }
        }

        if profile {
            eprintln!(
                "[cuda-profile] attn {:.2}ms  o_proj {:.2}ms  ffn {:.2}ms  lm_head {:.2}ms  total {:.2}ms",
                prof_us[0] as f64 / 1000.0,
                prof_us[1] as f64 / 1000.0,
                prof_us[2] as f64 / 1000.0,
                prof_us[3] as f64 / 1000.0,
                (prof_us[0] + prof_us[1] + prof_us[2] + prof_us[3]) as f64 / 1000.0
            );
        }

        self.pos = pos + 1;
        Ok(())
    }

    /// Prefill `n_tokens` (1..=8) embeddings starting at `pos` in one pass.
    /// `x_host` is [n_tokens, dim], token-major. The layer order and arithmetic
    /// match `forward`, but each weight tile is read once and shared by all
    /// eight token warps instead of being streamed once per token.
    pub fn forward_batch(
        &mut self,
        x_host: &[f32],
        pos: usize,
        n_tokens: usize,
        compute_logits: bool,
    ) -> Result<(), String> {
        if std::thread::current().id() != self.owner {
            return Err(
                "CUDA context belongs to another thread (load and generate on the same thread)"
                    .into(),
            );
        }
        if n_tokens == 0 || n_tokens > PREFILL_BATCH {
            return Err(format!(
                "prefill chunk has {n_tokens} tokens; the b8 kernels take 1..={PREFILL_BATCH}"
            ));
        }
        if x_host.len() < n_tokens * self.dim {
            return Err(format!(
                "prefill chunk has {} embedding values, need {}",
                x_host.len(),
                n_tokens * self.dim
            ));
        }
        if pos
            .checked_add(n_tokens)
            .is_none_or(|end| end > self.cap)
        {
            return Err(format!(
                "chunk {pos}..{} is past the {}-row CUDA KV cache; raise SOFUU_CUDA_CTX",
                pos + n_tokens,
                self.cap
            ));
        }
        if pos != self.pos {
            return Err(format!(
                "non-contiguous position {pos}: the device cache holds rows 0..{}; a new                  request must call reset_cache() first",
                self.pos
            ));
        }

        let e = &self.engine;
        e.upload(self.xb, f32_bytes(&x_host[..n_tokens * self.dim]))
            .map_err(|err| format!("upload prefill embeddings: {err}"))?;

        let mut positions = [0usize; PREFILL_BATCH];
        for (i, p) in positions.iter_mut().take(n_tokens).enumerate() {
            *p = pos + i;
        }
        quant::rope_table_rows(
            &mut self.rope_host_b,
            self.head_dim,
            self.rope_n_rot,
            self.rope_mscale,
            self.rope_base,
            &self.rope_adjust,
            &positions[..n_tokens],
        );
        e.upload(
            self.rope_b,
            f32_bytes(&self.rope_host_b[..n_tokens * self.head_dim]),
        )
        .map_err(|err| format!("upload rope batch table: {err}"))?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let n = n_tokens;

        for (li, l) in self.layers.iter().enumerate() {
            e.rmsnorm_batch(
                self.xb,
                l.attn_norm,
                self.x_norm_b,
                self.dim,
                self.eps,
                n,
            )
            .map_err(|err| format!("layer {li} attn_norm batch: {err}"))?;
            e.gemm_b8(
                l.attn_q_quant,
                l.attn_q,
                self.x_norm_b,
                self.q_b,
                self.q_dim,
                self.dim,
                n,
            )
            .map_err(|err| format!("layer {li} attn_q batch: {err}"))?;
            e.gemm_b8(
                l.attn_k_quant,
                l.attn_k,
                self.x_norm_b,
                self.k_b,
                self.kv_dim,
                self.dim,
                n,
            )
            .map_err(|err| format!("layer {li} attn_k batch: {err}"))?;
            e.gemm_b8(
                l.attn_v_quant,
                l.attn_v,
                self.x_norm_b,
                self.v_b,
                self.kv_dim,
                self.dim,
                n,
            )
            .map_err(|err| format!("layer {li} attn_v batch: {err}"))?;

            e.rope_batch(
                self.q_b,
                self.rope_b,
                self.n_heads,
                self.head_dim,
                self.rope_n_rot,
                self.rope_interleaved,
                n,
            )
            .map_err(|err| format!("layer {li} q rope batch: {err}"))?;
            e.rope_batch(
                self.k_b,
                self.rope_b,
                self.n_kv_heads,
                self.head_dim,
                self.rope_n_rot,
                self.rope_interleaved,
                n,
            )
            .map_err(|err| format!("layer {li} k rope batch: {err}"))?;

            let k_slice = layer_slice(self.k_cache, li, self.per_layer_kv);
            let v_slice = layer_slice(self.v_cache, li, self.per_layer_kv);
            e.kv_write_batch(
                k_slice,
                v_slice,
                self.k_b,
                self.v_b,
                pos,
                n,
                self.n_kv_heads,
                self.head_dim,
                self.cap,
            )
            .map_err(|err| format!("layer {li} kv_write batch: {err}"))?;
            e.attn_prefill(
                self.q_b,
                k_slice,
                v_slice,
                self.attn_b,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                pos,
                n,
                self.cap,
                scale,
            )
            .map_err(|err| format!("layer {li} prefill attention: {err}"))?;
            e.gemm_b8(
                l.attn_output_quant,
                l.attn_output,
                self.attn_b,
                self.proj_b,
                self.dim,
                self.q_dim,
                n,
            )
            .map_err(|err| format!("layer {li} attn_output batch: {err}"))?;
            e.add(self.xb, self.proj_b, self.xb, n * self.dim)
                .map_err(|err| format!("layer {li} attn residual batch: {err}"))?;

            e.rmsnorm_batch(
                self.xb,
                l.ffn_norm,
                self.x_norm_b,
                self.dim,
                self.eps,
                n,
            )
            .map_err(|err| format!("layer {li} ffn_norm batch: {err}"))?;
            e.gemm_b8(
                l.ffn_gate_quant,
                l.ffn_gate,
                self.x_norm_b,
                self.gate_b,
                self.hidden_dim,
                self.dim,
                n,
            )
            .map_err(|err| format!("layer {li} ffn_gate batch: {err}"))?;
            e.gemm_b8(
                l.ffn_up_quant,
                l.ffn_up,
                self.x_norm_b,
                self.up_b,
                self.hidden_dim,
                self.dim,
                n,
            )
            .map_err(|err| format!("layer {li} ffn_up batch: {err}"))?;
            e.swiglu(
                self.gate_b,
                self.up_b,
                self.act_b,
                n * self.hidden_dim,
            )
            .map_err(|err| format!("layer {li} swiglu batch: {err}"))?;
            e.gemm_b8(
                l.ffn_down_quant,
                l.ffn_down,
                self.act_b,
                self.down_b,
                self.dim,
                self.hidden_dim,
                n,
            )
            .map_err(|err| format!("layer {li} ffn_down batch: {err}"))?;
            e.add(self.xb, self.down_b, self.xb, n * self.dim)
                .map_err(|err| format!("layer {li} ffn residual batch: {err}"))?;
        }

        if compute_logits {
            // Only the last token of the chunk is needed: the caller samples
            // from it. The other rows stay in `xb` for the next decode token's
            // KV history.
            let off = (n - 1) * self.dim * std::mem::size_of::<f32>();
            let x_last = DeviceBuf {
                ptr: self.xb.ptr + off as u64,
                bytes: self.dim * std::mem::size_of::<f32>(),
            };
            let xn_last = DeviceBuf {
                ptr: self.x_norm_b.ptr + off as u64,
                bytes: self.dim * std::mem::size_of::<f32>(),
            };
            e.rmsnorm(x_last, self.output_norm, xn_last, self.dim, self.eps)
                .map_err(|err| format!("output_norm batch: {err}"))?;
            e.gemv(
                self.output_quant,
                self.output,
                xn_last,
                self.logits_dev,
                self.vocab,
                self.dim,
            )
            .map_err(|err| format!("lm_head batch: {err}"))?;
            e.sync()
                .map_err(|err| format!("sync before batch logits: {err}"))?;
            e.download(&mut self.logit_bytes, self.logits_dev)
                .map_err(|err| format!("download batch logits: {err}"))?;
            for (i, c) in self.logit_bytes.chunks_exact(4).enumerate() {
                self.logits_host[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }

        self.pos = pos + n;
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

/// Bytes uploaded per layout, for the load receipt. The decode cost of a
/// checkpoint is a function of which layouts its matrices use, so the line
/// that arms the decoder reports the split instead of one total.
fn add_quant_bytes(totals: &mut [usize; 4], tv: &TensorView) {
    let slot = match tv.tensor_type {
        GGML_TYPE_F32 => 0,
        GGML_TYPE_Q8_0 => 1,
        GGML_TYPE_Q4_K => 2,
        GGML_TYPE_Q6_K => 3,
        _ => return,
    };
    totals[slot] += tv.size_bytes;
}

fn tensor_quant(tv: &TensorView, rows: usize, cols: usize) -> Result<Quant, String> {    let q = match tv.tensor_type {
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
