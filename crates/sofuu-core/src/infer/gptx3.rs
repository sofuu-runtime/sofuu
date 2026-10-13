// crates/sofuu-core/src/infer/gptx3.rs — GPT-X3 hybrid GDN2/GQA backend.
//
// Implements the reference Hugging Face GPTX3ForCausalLM recurrence on CPU.
// The model alternates gated-delta recurrence layers with a small number of
// conventional GQA layers, so it needs its own state layout and cache.

use super::block_graph::WeightMatrix;
use super::model::ModelConfig;
use super::quant::{bf16_to_f32, f16_to_f32, swiglu_in_place};
use super::npy::TensorDType;
use super::tensor_table::UnifiedTensorTable;


#[derive(Clone)]
enum Gptx3Mixer {
    Gdn {
        q_proj: WeightMatrix,
        k_proj: WeightMatrix,
        v_proj: WeightMatrix,
        b_proj: WeightMatrix,
        w_proj: WeightMatrix,
        o_proj: WeightMatrix,
        f_down: WeightMatrix,
        f_up: WeightMatrix,
        g_down: WeightMatrix,
        g_up: WeightMatrix,
        q_conv: Vec<f32>,
        k_conv: Vec<f32>,
        v_conv: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        g_bias: Vec<f32>,
        o_norm: Vec<f32>,
    },
    Attention {
        q_gate: WeightMatrix,
        k_proj: WeightMatrix,
        v_proj: WeightMatrix,
        o_proj: WeightMatrix,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
}

struct Gptx3Layer {
    pre_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Gptx3Mixer,
    gate: WeightMatrix,
    up: WeightMatrix,
    down: WeightMatrix,
}

pub struct Gptx3Model {
    pub config: ModelConfig,
    embed: WeightMatrix,
    layers: Vec<Gptx3Layer>,
    final_norm: Vec<f32>,
    logits: Vec<f32>,
    x: Vec<f32>,
    x_norm: Vec<f32>,
    attn_out: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_out: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    low: Vec<f32>,
    b: Vec<f32>,
    w: Vec<f32>,
    head_out: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    recurrent: Vec<f32>,
    conv_history: Vec<f32>,
}

fn matrix(table: &UnifiedTensorTable, name: &str) -> Result<WeightMatrix, String> {
    let (meta, data) = table
        .get_tensor_data(name)
        .ok_or_else(|| format!("missing GPT-X3 tensor: {name}"))?;
    let rows = meta.shape[0];
    let cols = meta.shape.get(1).copied().unwrap_or(1);
    Ok(WeightMatrix::new_from_data(rows, cols, meta.dtype, data))
}

fn vector(table: &UnifiedTensorTable, name: &str) -> Result<Vec<f32>, String> {
    let (meta, slice) = table
        .get_tensor(name)
        .ok_or_else(|| format!("missing GPT-X3 tensor: {name}"))?;
    let len = meta.shape.iter().product::<usize>();
    match meta.dtype {
        TensorDType::F32 => Ok(unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, len) }.to_vec()),
        TensorDType::F16 => Ok(unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u16, len) }
            .iter()
            .map(|&v| f16_to_f32(v))
            .collect()),
        TensorDType::BF16 => Ok(unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u16, len) }
            .iter()
            .map(|&v| bf16_to_f32(v))
            .collect()),
        _ => Err(format!("unsupported GPT-X3 dtype for {name}: {:?}", meta.dtype)),
    }
}

fn conv_weight(table: &UnifiedTensorTable, name: &str) -> Result<Vec<f32>, String> {
    let (meta, slice) = table
        .get_tensor(name)
        .ok_or_else(|| format!("missing GPT-X3 tensor: {name}"))?;
    let len = meta.shape.iter().product::<usize>();
    if meta.dtype != TensorDType::F32 {
        return Err(format!("GPT-X3 conv expects F32, found {:?} for {name}", meta.dtype));
    }
    Ok(unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, len) }.to_vec())
}

#[inline]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn rmsnorm_local(x: &mut [f32], weight: &[f32], eps: f32) {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / ((sum_sq / x.len() as f32) + eps).sqrt();
    for (v, w) in x.iter_mut().zip(weight) {
        *v *= scale * w;
    }
}

fn rmsnorm_offset(x: &mut [f32], weight: &[f32], eps: f32) {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / ((sum_sq / x.len() as f32) + eps).sqrt();
    for (v, w) in x.iter_mut().zip(weight) {
        *v *= scale * (1.0 + w);
    }
}

fn l2_normalize(x: &mut [f32]) {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() + 1e-6;
    let scale = 1.0 / sum_sq.sqrt();
    for v in x {
        *v *= scale;
    }
}

fn rotary_partial(vec: &mut [f32], n_heads: usize, head_dim: usize, rotary_dim: usize, pos: usize, base: f32) {
    let pairs = rotary_dim / 2;
    for h in 0..n_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        for i in 0..pairs {
            let freq = 1.0 / base.powf((2 * i) as f32 / rotary_dim as f32);
            let theta = pos as f32 * freq;
            let (sin, cos) = theta.sin_cos();
            let even = 2 * i;
            let odd = even + 1;
            let a = head[even];
            let b = head[odd];
            head[even] = a * cos - b * sin;
            head[odd] = a * sin + b * cos;
        }
    }
}

impl Gptx3Model {
    pub fn from_tensor_table(table: &UnifiedTensorTable, max_seq_len: usize) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::json!({});
        let _ = cfg;
        let dim = 576;
        let n_layers = 24;
        let n_heads = 9;
        let n_kv_heads = 3;
        let head_dim = 64;
        let gdn_head_dim = 64;
        let hidden_dim = 1584;
        let vocab_size = 65536;
        let norm_eps = 1e-6;
        let gdn_norm_eps = 1e-5;
        let rope_base = 100000.0;
        let rotary_dim = 32;
        let conv_size = 4;
        let effective_seq = if max_seq_len > 0 { max_seq_len.min(8192) } else { 4096 };

        let embed = matrix(table, "transformer.wte.weight")?;
        let final_norm = vector(table, "transformer.ln_f.weight")?;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let base = format!("transformer.h.{i}");
            let is_attention = table.get_tensor_data(&format!("{base}.attn.q_gate_proj.weight")).is_some();
            let mixer = if is_attention {
                Gptx3Mixer::Attention {
                    q_gate: matrix(table, &format!("{base}.attn.q_gate_proj.weight"))?,
                    k_proj: matrix(table, &format!("{base}.attn.k_proj.weight"))?,
                    v_proj: matrix(table, &format!("{base}.attn.v_proj.weight"))?,
                    o_proj: matrix(table, &format!("{base}.attn.o_proj.weight"))?,
                    q_norm: vector(table, &format!("{base}.attn.q_norm.weight"))?,
                    k_norm: vector(table, &format!("{base}.attn.k_norm.weight"))?,
                }
            } else {
                Gptx3Mixer::Gdn {
                    q_proj: matrix(table, &format!("{base}.attn.kernel.q_proj.weight"))?,
                    k_proj: matrix(table, &format!("{base}.attn.kernel.k_proj.weight"))?,
                    v_proj: matrix(table, &format!("{base}.attn.kernel.v_proj.weight"))?,
                    b_proj: matrix(table, &format!("{base}.attn.kernel.b_proj.weight"))?,
                    w_proj: matrix(table, &format!("{base}.attn.kernel.w_proj.weight"))?,
                    o_proj: matrix(table, &format!("{base}.attn.kernel.o_proj.weight"))?,
                    f_down: matrix(table, &format!("{base}.attn.kernel.f_proj.0.weight"))?,
                    f_up: matrix(table, &format!("{base}.attn.kernel.f_proj.1.weight"))?,
                    g_down: matrix(table, &format!("{base}.attn.kernel.g_proj.0.weight"))?,
                    g_up: matrix(table, &format!("{base}.attn.kernel.g_proj.1.weight"))?,
                    q_conv: conv_weight(table, &format!("{base}.attn.kernel.q_conv1d.weight"))?,
                    k_conv: conv_weight(table, &format!("{base}.attn.kernel.k_conv1d.weight"))?,
                    v_conv: conv_weight(table, &format!("{base}.attn.kernel.v_conv1d.weight"))?,
                    a_log: vector(table, &format!("{base}.attn.kernel.A_log"))?,
                    dt_bias: vector(table, &format!("{base}.attn.kernel.dt_bias"))?,
                    g_bias: vector(table, &format!("{base}.attn.kernel.g_proj.1.bias"))?,
                    o_norm: vector(table, &format!("{base}.attn.kernel.o_norm.weight"))?,
                }
            };
            layers.push(Gptx3Layer {
                pre_norm: vector(table, &format!("{base}.ln_1.weight"))?,
                post_norm: vector(table, &format!("{base}.ln_2.weight"))?,
                mixer,
                gate: matrix(table, &format!("{base}.mlp.w_gate.weight"))?,
                up: matrix(table, &format!("{base}.mlp.w_up.weight"))?,
                down: matrix(table, &format!("{base}.mlp.w_down.weight"))?,
            });
        }

        let config = ModelConfig {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_dim,
            vocab_size,
            norm_eps,
            rope_freq_base: rope_base,
            max_seq_len: effective_seq,
            sliding_window: 0,
            rope_interleaved: true,
        };

        let kv_cache_len = n_layers * effective_seq * n_kv_heads * head_dim;
        let recurrent_len = n_layers * dim * gdn_head_dim * gdn_head_dim;
        let conv_len = n_layers * 3 * dim * (conv_size - 1);

        Ok(Self {
            config,
            embed,
            layers,
            final_norm,
            logits: vec![0.0; vocab_size],
            x: vec![0.0; dim],
            x_norm: vec![0.0; dim],
            attn_out: vec![0.0; dim],
            ffn_gate: vec![0.0; hidden_dim],
            ffn_up: vec![0.0; hidden_dim],
            ffn_out: vec![0.0; dim],
            q: vec![0.0; n_heads * head_dim],
            k: vec![0.0; n_kv_heads * head_dim],
            v: vec![0.0; n_kv_heads * head_dim],
            gate: vec![0.0; n_heads * head_dim],
            low: vec![0.0; gdn_head_dim],
            b: vec![0.0; dim],
            w: vec![0.0; dim],
            head_out: vec![0.0; n_heads * head_dim],
            k_cache: vec![0.0; kv_cache_len],
            v_cache: vec![0.0; kv_cache_len],
            recurrent: vec![0.0; recurrent_len],
            conv_history: vec![0.0; conv_len],
        })
    }

    pub fn reset_cache(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
        self.recurrent.fill(0.0);
        self.conv_history.fill(0.0);
    }

    pub fn compact_kv_cache(&mut self, _current_seq_len: usize, _sink: usize, _drop: usize) -> usize {
        // GPT-X3's recurrent layers do not support lossless RoPE-style cache
        // shifting. Long-context continuation is intentionally disallowed.
        self.reset_cache();
        0
    }

    fn embedding_row(&self, token: u32, dst: &mut [f32]) {
        let dim = self.config.dim;
        let row_size = match self.embed.dtype {
            TensorDType::F32 => dim * 4,
            TensorDType::F16 => dim * 2,
            TensorDType::BF16 => dim * 2,
            _ => dim * 4,
        };
        let start = token as usize * row_size;
        let bytes = self.embed.data.as_slice();
        if start + row_size > bytes.len() {
            dst.fill(0.0);
            return;
        }
        let row = &bytes[start..start + row_size];
        match self.embed.dtype {
            TensorDType::F32 => {
                let src = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const f32, dim) };
                dst.copy_from_slice(src);
            }
            TensorDType::F16 => {
                let src = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u16, dim) };
                for (d, &h) in dst.iter_mut().zip(src) {
                    *d = f16_to_f32(h);
                }
            }
            TensorDType::BF16 => {
                let src = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u16, dim) };
                for (d, &h) in dst.iter_mut().zip(src) {
                    *d = bf16_to_f32(h);
                }
            }
            _ => dst.fill(0.0),
        }
    }

    fn depthwise_conv(&mut self, raw: &[f32], weights: &[f32], layer: usize, stream: usize, out: &mut [f32]) {
        let dim = self.config.dim;
        let hist_len = 3;
        let hist_base = (layer * 3 + stream) * dim * hist_len;
        for c in 0..dim {
            let mut acc = 0.0f32;
            for t in 0..hist_len {
                acc += self.conv_history[hist_base + c * hist_len + t] * weights[c * 4 + t];
            }
            acc += raw[c] * weights[c * 4 + 3];
            out[c] = acc / (1.0 + (-acc).exp());
            for t in 0..hist_len - 1 {
                self.conv_history[hist_base + c * hist_len + t] =
                    self.conv_history[hist_base + c * hist_len + t + 1];
            }
            self.conv_history[hist_base + c * hist_len + hist_len - 1] = raw[c];
        }
    }

    fn gdn_forward(&mut self, layer_idx: usize, pos: usize) {
        let dim = self.config.dim;
        let gdn_dim = 64;
        let heads = dim / gdn_dim;
        let mixer = self.layers[layer_idx].mixer.clone();
        let (q_proj, k_proj, v_proj, b_proj, w_proj, o_proj, f_down, f_up, g_down, g_up, q_conv, k_conv, v_conv, a_log, dt_bias, g_bias, o_norm) =
            match mixer {
                Gptx3Mixer::Gdn { q_proj, k_proj, v_proj, b_proj, w_proj, o_proj, f_down, f_up, g_down, g_up, q_conv, k_conv, v_conv, a_log, dt_bias, g_bias, o_norm } =>
                    (q_proj, k_proj, v_proj, b_proj, w_proj, o_proj, f_down, f_up, g_down, g_up, q_conv, k_conv, v_conv, a_log, dt_bias, g_bias, o_norm),
                _ => unreachable!(),
            };

        let xn = self.x_norm.clone();
        let mut qr = vec![0.0f32; dim];
        let mut kr = vec![0.0f32; dim];
        let mut vr = vec![0.0f32; dim];
        q_proj.gemv(&xn, &mut qr);
        k_proj.gemv(&xn, &mut kr);
        v_proj.gemv(&xn, &mut vr);
        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() {
            eprintln!("[gdn proj] layer={} xn0..4={:?} q0..4={:?} k0..4={:?} v0..4={:?}", layer_idx, &xn[..4], &qr[..4], &kr[..4], &vr[..4]);
        }
        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() {
            let bad = |v: &[f32]| v.iter().any(|x| !x.is_finite());
            eprintln!(
                "[gdn in] layer={} xn={} qr={} kr={} vr={}",
                layer_idx, bad(&xn), bad(&qr), bad(&kr), bad(&vr)
            );
        }
        let qr_in = qr.clone(); self.depthwise_conv(&qr_in, &q_conv, layer_idx, 0, &mut qr);
        let kr_in = kr.clone(); self.depthwise_conv(&kr_in, &k_conv, layer_idx, 1, &mut kr);
        let vr_in = vr.clone(); self.depthwise_conv(&vr_in, &v_conv, layer_idx, 2, &mut vr);
        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() {
            eprintln!("[gdn conv] layer={} q0..4={:?} k0..4={:?} v0..4={:?}", layer_idx, &qr[..4], &kr[..4], &vr[..4]);
        }

        f_down.gemv(&xn, &mut self.low);
        f_up.gemv(&self.low, &mut self.x_norm);
        for (v, bias) in self.x_norm.iter_mut().zip(dt_bias) {
            let value = *v + bias;
            *v = value.max(0.0) + (1.0 + (-value.abs()).exp()).ln();
        }
        for h in 0..heads {
            let decay_factor = -(-a_log[h]).exp();
            for v in self.x_norm[h * gdn_dim..(h + 1) * gdn_dim].iter_mut() {
                *v *= decay_factor;
            }
        }
        b_proj.gemv(&xn, &mut self.b);
        w_proj.gemv(&xn, &mut self.w);
        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() {
            let bad = |v: &[f32]| v.iter().any(|x| !x.is_finite());
            eprintln!(
                "[gdn mid] layer={} low={} g={} b={} w={} qr={} kr={} vr={}",
                layer_idx, bad(&self.low), bad(&self.x_norm), bad(&self.b), bad(&self.w), bad(&qr), bad(&kr), bad(&vr)
            );
        }

        let state_base = layer_idx * dim * gdn_dim * gdn_dim;
        for h in 0..heads {
            let q_head = &mut qr[h * gdn_dim..(h + 1) * gdn_dim];
            let k_head = &mut kr[h * gdn_dim..(h + 1) * gdn_dim];
            l2_normalize(q_head);
            l2_normalize(k_head);
            let scale = (gdn_dim as f32).sqrt().recip();
            for v in q_head.iter_mut() {
                *v *= scale;
            }
            if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && layer_idx == 1 && h == 0 {
                eprintln!("[gdn qnorm] layer={} head={} q0..8={:?} norm={}", layer_idx, h, &q_head[..8], q_head.iter().map(|v| v*v).sum::<f32>());
            }
        }

        for h in 0..heads {
            let decay: Vec<f32> = self.x_norm[h * gdn_dim..(h + 1) * gdn_dim]
                .iter()
                .map(|&v| v.exp())
                .collect();
            if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() {
                let dmin = decay.iter().cloned().fold(f32::INFINITY, f32::min);
                let dmax = decay.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!("[gdn decay] layer={} head={} decay=({dmin},{dmax}) g=({},{})", layer_idx, h, self.x_norm[h * gdn_dim..(h + 1) * gdn_dim].iter().cloned().fold(f32::INFINITY, f32::min), self.x_norm[h * gdn_dim..(h + 1) * gdn_dim].iter().cloned().fold(f32::NEG_INFINITY, f32::max));
            }
            let state_start = state_base + h * gdn_dim * gdn_dim;
            let mut decayed = vec![0.0f32; gdn_dim * gdn_dim];
            for i in 0..gdn_dim {
                for j in 0..gdn_dim {
                    decayed[i * gdn_dim + j] = self.recurrent[state_start + i * gdn_dim + j] * decay[i];
                }
            }
            let mut erase = vec![0.0f32; gdn_dim];
            for i in 0..gdn_dim {
                let bk = sigmoid(self.b[h * gdn_dim + i]) * kr[h * gdn_dim + i];
                for j in 0..gdn_dim {
                    erase[j] += bk * decayed[i * gdn_dim + j];
                }
            }
            let mut update = vec![0.0f32; gdn_dim];
            for j in 0..gdn_dim {
                update[j] = sigmoid(self.w[h * gdn_dim + j]) * vr[h * gdn_dim + j] - erase[j];
            }
            if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && h == 0 {
                eprintln!("[gdn upd] layer={} k0..4={:?} v0..4={:?} b0..4={:?} w0..4={:?} erase0..4={:?} update0..4={:?}", layer_idx, &kr[..4], &vr[..4], &self.b[..4], &self.w[..4], &erase[..4], &update[..4]);
            }
            for i in 0..gdn_dim {
                for j in 0..gdn_dim {
                    self.recurrent[state_start + i * gdn_dim + j] =
                        decayed[i * gdn_dim + j] + kr[h * gdn_dim + i] * update[j];
                }
            }
            for d in 0..gdn_dim {
                let mut sum = 0.0f32;
                for j in 0..gdn_dim {
                    sum += qr[h * gdn_dim + j] * self.recurrent[state_start + j * gdn_dim + d];
                }
                self.head_out[h * gdn_dim + d] = sum;
            }
            if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && h == 0 {
                eprintln!("[gdn state] layer={} first={:?} out_first={:?}", layer_idx, &self.recurrent[state_start..state_start + 8], &self.head_out[..8]);
            }
        }

        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && layer_idx == 0 {
            eprintln!("[gdn raw] layer={} out={:?} onorm={:?} len={}", layer_idx, &self.head_out[..64], &o_norm[..8], o_norm.len());
        }
        for h in 0..heads {
            let out = &mut self.head_out[h * gdn_dim..(h + 1) * gdn_dim];
            let sum_sq: f32 = out.iter().map(|v| v * v).sum::<f32>();
            let scale = 1.0 / (sum_sq / gdn_dim as f32 + 1e-5).sqrt();
            for (v, w) in out.iter_mut().zip(o_norm.iter()) {
                *v *= scale * w;
            }
        }

        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && layer_idx == 0 {
            eprintln!("[gdn norm] layer={} out0..8={:?}", layer_idx, &self.head_out[..8]);
        }
        g_down.gemv(&xn, &mut self.low);
        g_up.gemv(&self.low, &mut self.x_norm);
        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && layer_idx == 0 {
            eprintln!("[gdn gate] layer={} gate0..8={:?} gated0..8={:?}", layer_idx, &self.x_norm[..8], {
                let mut z = self.head_out.clone();
                for h in 0..heads { for d in 0..gdn_dim { let idx=h*gdn_dim+d; z[idx] *= sigmoid(self.x_norm[idx]+g_bias[idx]); } }
                z[..8].to_vec()
            });
        }
        for h in 0..heads {
            for d in 0..gdn_dim {
                let idx = h * gdn_dim + d;
                let gate = sigmoid(self.x_norm[idx] + g_bias[idx]);
                self.head_out[idx] *= gate;
            }
        }
        o_proj.gemv(&self.head_out, &mut self.attn_out);
        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() {
            eprintln!("[gdn sample] layer={} {:?}", layer_idx, &self.attn_out[..8]);
            let bad = |v: &[f32]| v.iter().any(|x| !x.is_finite());
            let min = self.head_out.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = self.head_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let state = &self.recurrent[state_base..state_base + dim * 64 * 64];
            let smin = state.iter().cloned().fold(f32::INFINITY, f32::min);
            let smax = state.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[gdn out] layer={} out={} attn={} out_range=({min},{max}) state_bad={} state_range=({smin},{smax})", layer_idx, bad(&self.head_out), bad(&self.attn_out), bad(state));
        }
    }

    fn attention_forward(&mut self, layer_idx: usize, pos: usize) {
        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let rep = n_heads / n_kv;
        let mixer = self.layers[layer_idx].mixer.clone();
        let (q_gate, k_proj, v_proj, o_proj, q_norm, k_norm) = match mixer {
            Gptx3Mixer::Attention { q_gate, k_proj, v_proj, o_proj, q_norm, k_norm } =>
                (q_gate, k_proj, v_proj, o_proj, q_norm, k_norm),
            _ => unreachable!(),
        };

        let mut qg = vec![0.0f32; 2 * n_heads * head_dim];
        q_gate.gemv(&self.x_norm, &mut qg);
        self.q.copy_from_slice(&qg[..n_heads * head_dim]);
        self.gate.copy_from_slice(&qg[n_heads * head_dim..]);
        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && layer_idx == 3 {
            eprintln!("[gqa raw] layer={} q0..8={:?} qnorm0..8={:?}", layer_idx, &self.q[..8], &q_norm[..8]);
        }
        k_proj.gemv(&self.x_norm, &mut self.k);
        v_proj.gemv(&self.x_norm, &mut self.v);

        for h in 0..n_heads {
            let q_head = &mut self.q[h * head_dim..(h + 1) * head_dim];
            rmsnorm_offset(q_head, &q_norm, self.config.norm_eps);
        }
        for h in 0..n_kv {
            let k_head = &mut self.k[h * head_dim..(h + 1) * head_dim];
            rmsnorm_offset(k_head, &k_norm, self.config.norm_eps);
        }
        rotary_partial(&mut self.q, n_heads, head_dim, 32, pos, self.config.rope_freq_base);
        rotary_partial(&mut self.k, n_kv, head_dim, 32, pos, self.config.rope_freq_base);

        let seq_len = pos + 1;
        let kv_dim = n_kv * head_dim;
        let cache_base = layer_idx * self.config.max_seq_len * kv_dim;
        self.k_cache[cache_base + pos * kv_dim..cache_base + (pos + 1) * kv_dim]
            .copy_from_slice(&self.k);
        self.v_cache[cache_base + pos * kv_dim..cache_base + (pos + 1) * kv_dim]
            .copy_from_slice(&self.v);

        let mut scores = vec![0.0f32; seq_len];
        for h in 0..n_heads {
            let kv_h = h / rep;
            let q_head = &self.q[h * head_dim..(h + 1) * head_dim];
            let head_base = cache_base + kv_h * self.config.max_seq_len * head_dim;
            let mut max_score = f32::NEG_INFINITY;
            for p in 0..seq_len {
                let k_head = &self.k_cache[head_base + p * head_dim..head_base + (p + 1) * head_dim];
                let score = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum::<f32>() / (head_dim as f32).sqrt();
                scores[p] = score;
                max_score = max_score.max(score);
            }
            let mut sum = 0.0f32;
            for score in scores.iter_mut().take(seq_len) {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            let out_head = &mut self.head_out[h * head_dim..(h + 1) * head_dim];
            out_head.fill(0.0);
            for p in 0..seq_len {
                let weight = scores[p] / sum;
                let v_head = &self.v_cache[head_base + p * head_dim..head_base + (p + 1) * head_dim];
                for (o, v) in out_head.iter_mut().zip(v_head) {
                    *o += weight * v;
                }
            }
            for (o, g) in out_head.iter_mut().zip(&self.gate[h * head_dim..(h + 1) * head_dim]) {
                *o *= 1.0 / (1.0 + (-g).exp());
            }
        }
        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && layer_idx == 3 {
            eprintln!("[gqa] layer={} q0..8={:?} k0..8={:?} v0..8={:?} out0..8={:?}", layer_idx, &self.q[..8], &self.k[..8], &self.v[..8], &self.head_out[..8]);
        }
        o_proj.gemv(&self.head_out, &mut self.attn_out);
        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && layer_idx == 3 {
            eprintln!("[gqa attn] layer={} attn0..8={:?}", layer_idx, &self.attn_out[..8]);
        }
    }

    fn forward_impl(&mut self, token: u32, pos: usize, compute_logits: bool) {
        let mut x = vec![0.0f32; self.config.dim];
        self.embedding_row(token, &mut x);
        self.x.copy_from_slice(&x);
        if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() {
            eprintln!("[embed] token={} x0..4={:?} norm0..4={:?}", token, &self.x[..576], &self.layers[0].pre_norm[..4]);
        }
        for layer_idx in 0..self.config.n_layers {
            self.x_norm.copy_from_slice(&self.x);
            rmsnorm_local(&mut self.x_norm, &self.layers[layer_idx].pre_norm, self.config.norm_eps);
            if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && layer_idx < 2 {
                eprintln!("[after norm] layer={} residual0..8={:?} xnorm0..8={:?}", layer_idx, &self.x[..576], &self.x_norm[..8]);
            }
            match &self.layers[layer_idx].mixer {
                Gptx3Mixer::Gdn { .. } => self.gdn_forward(layer_idx, pos),
                Gptx3Mixer::Attention { .. } => self.attention_forward(layer_idx, pos),
            }
            if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && layer_idx == 0 {
                eprintln!("[gptx3 attnres] layer={} attn0..16={:?} residual0..16={:?}", layer_idx, &self.attn_out[..576], {
                    let mut z=self.x.clone(); for (x,a) in z.iter_mut().zip(&self.attn_out) {*x+=a;} z[..16].to_vec()
                });
            }
            for (x, a) in self.x.iter_mut().zip(&self.attn_out) {
                *x += a;
            }
            if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && self.x.iter().any(|v| !v.is_finite()) {
                eprintln!("[gptx3 nan] layer={} pos={}", layer_idx, pos);
            }
            self.x_norm.copy_from_slice(&self.x);
            rmsnorm_local(&mut self.x_norm, &self.layers[layer_idx].post_norm, self.config.norm_eps);
            self.layers[layer_idx].gate.gemv(&self.x_norm, &mut self.ffn_gate);
            self.layers[layer_idx].up.gemv(&self.x_norm, &mut self.ffn_up);
            swiglu_in_place(&mut self.ffn_gate, &self.ffn_up);
            self.layers[layer_idx].down.gemv(&self.ffn_gate, &mut self.ffn_out);
            if std::env::var("SOFUU_DEBUG_GPTX3").is_ok() && layer_idx == 0 {
                eprintln!("[gptx3 mlp] layer={} postnorm0..8={:?} gate0..8={:?} up0..8={:?} swiglu0..8={:?} out0..8={:?}", layer_idx, &self.x[..8], &self.ffn_gate[..8], &self.ffn_up[..8], &self.ffn_gate[..8], &self.ffn_out[..576]);
            }
            for (x, f) in self.x.iter_mut().zip(&self.ffn_out) {
                *x += f;
            }
        }
        if compute_logits {
            rmsnorm_local(&mut self.x, &self.final_norm, self.config.norm_eps);
            self.x_norm.copy_from_slice(&self.x);
            self.embed.gemv(&self.x, &mut self.logits);
        }
    }

    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        self.forward_impl(token, pos, true);
        if std::env::var("SOFUU_DEBUG_TOPK").is_ok() {
            let mut top: Vec<(u32, f32)> = self.logits.iter().enumerate().map(|(i, v)| (i as u32, *v)).collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top.truncate(5);
            eprintln!("[gptx3 top] {:?}", top);
        }
        &self.logits
    }

    pub fn forward_prefill(&mut self, token: u32, pos: usize) {
        self.forward_impl(token, pos, false);
    }

    pub fn forward_prefill_batch(&mut self, tokens: &[u32], start_pos: usize) {
        for (i, &token) in tokens.iter().enumerate() {
            self.forward_impl(token, start_pos + i, false);
        }
    }

    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], mut start_pos: usize) -> &[f32] {
        if tokens.is_empty() {
            return &self.logits;
        }
        for &token in &tokens[..tokens.len() - 1] {
            self.forward_impl(token, start_pos, false);
            start_pos += 1;
        }
        self.forward_impl(tokens[tokens.len() - 1], start_pos, true);
        &self.logits
    }
}
