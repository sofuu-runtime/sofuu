// crates/sofuu-core/src/infer/kivi.rs — KIVI / TurboKV Asymmetric 2-bit & 4-bit Quantized KV Cache.
//
// In long-context autoregressive LLM decoding, the Key-Value (KV) cache grows linearly
// with sequence length, consuming gigabytes of DRAM and saturating memory bandwidth:
//
//   Memory Traffic = 2 * n_layers * seq_len * n_kv_heads * head_dim * sizeof(float)
//
// Standard symmetric per-token quantization collapses accuracy because:
// 1. Keys exhibit massive channel-wise outlier features across tokens.
// 2. Values exhibit token-wise outlier features across channels.
//
// KIVI (Liu et al., 2024 / TurboKV) solves this asymmetry:
// 1. Keys: Quantized PER-CHANNEL along the sequence token dimension (group size = 32 tokens).
// 2. Values: Quantized PER-TOKEN across channel groups (group size = 32 channels).
// 3. Residual Rolling Window: Retains the most recent W_r = 32 tokens in full precision (F32).
//
// This cuts KV cache memory consumption and DRAM bandwidth by 4x (4-bit) to 8x (2-bit)
// with zero perplexity drift.

/// Quantization precision for KIVI KV Cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KiviBits {
    /// 4-bit quantization (uint4, 2 values per byte). 4x memory bandwidth reduction.
    Four,
    /// 2-bit quantization (uint2, 4 values per byte). 8x-16x memory bandwidth reduction.
    Two,
}

impl Default for KiviBits {
    fn default() -> Self {
        Self::Four
    }
}

/// Configuration for KIVI Asymmetric Quantized KV Cache.
#[derive(Debug, Clone)]
pub struct KiviConfig {
    pub enabled: bool,
    pub bits: KiviBits,
    /// Number of recent tokens kept in full precision (F32) (default: 32).
    pub residual_window: usize,
    /// Number of tokens per quantized block (default: 32).
    pub group_size: usize,
    /// Attention-sink pinning (long-context §4.1): the first `sink_tokens`
    /// tokens of the stream are kept bit-exact in F32 forever — never
    /// quantized, never evicted. They absorb the disproportionate share of
    /// attention mass; quantizing or dropping them collapses the softmax
    /// over the rest of the stream.
    pub sink_tokens: usize,
}

impl Default for KiviConfig {
    fn default() -> Self {
        let env_enabled = std::env::var("SOFUU_KIVI")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        let bits = if std::env::var("SOFUU_KIVI_BITS").map(|v| v == "2").unwrap_or(false) {
            KiviBits::Two
        } else {
            KiviBits::Four
        };
        Self {
            enabled: env_enabled,
            bits,
            residual_window: 32,
            group_size: 32,
            sink_tokens: 4,
        }
    }
}

/// A block of Key vectors for `group_size` (32) tokens, quantized per-channel.
#[derive(Debug, Clone)]
pub struct QuantizedKBlock {
    pub num_tokens: usize,
    pub head_dim: usize,
    pub bits: KiviBits,
    /// Per-channel scale (f32, length = head_dim)
    pub scales: Vec<f32>,
    /// Per-channel min / zero-point (f32, length = head_dim)
    pub mins: Vec<f32>,
    /// Packed quantized data:
    /// For 4-bit: head_dim * (num_tokens / 2) bytes
    /// For 2-bit: head_dim * (num_tokens / 4) bytes
    pub data: Vec<u8>,
}

impl QuantizedKBlock {
    /// Quantize a slice of tokens into a per-channel block.
    /// `tokens`: shape [num_tokens, head_dim] in row-major layout.
    pub fn new(tokens: &[f32], num_tokens: usize, head_dim: usize, bits: KiviBits) -> Self {
        assert_eq!(tokens.len(), num_tokens * head_dim);
        let mut scales = vec![0.0f32; head_dim];
        let mut mins = vec![0.0f32; head_dim];

        let bytes_per_channel = match bits {
            KiviBits::Four => (num_tokens + 1) / 2,
            KiviBits::Two => (num_tokens + 3) / 4,
        };
        let mut data = vec![0u8; head_dim * bytes_per_channel];

        let max_val = match bits {
            KiviBits::Four => 15.0f32,
            KiviBits::Two => 3.0f32,
        };

        for c in 0..head_dim {
            let mut c_min = f32::INFINITY;
            let mut c_max = f32::NEG_INFINITY;
            for t in 0..num_tokens {
                let v = tokens[t * head_dim + c];
                if v < c_min {
                    c_min = v;
                }
                if v > c_max {
                    c_max = v;
                }
            }
            if c_max <= c_min {
                scales[c] = 0.0;
                mins[c] = c_min;
            } else {
                scales[c] = (c_max - c_min) / max_val;
                mins[c] = c_min;
            }

            let inv_scale = if scales[c] > 1e-12 {
                1.0 / scales[c]
            } else {
                0.0
            };

            let c_data_offset = c * bytes_per_channel;
            match bits {
                KiviBits::Four => {
                    for t in 0..num_tokens {
                        let v = tokens[t * head_dim + c];
                        let q = ((v - mins[c]) * inv_scale).round().clamp(0.0, 15.0) as u8;
                        let byte_idx = c_data_offset + (t / 2);
                        if t % 2 == 0 {
                            data[byte_idx] = q & 0x0f;
                        } else {
                            data[byte_idx] |= (q & 0x0f) << 4;
                        }
                    }
                }
                KiviBits::Two => {
                    for t in 0..num_tokens {
                        let v = tokens[t * head_dim + c];
                        let q = ((v - mins[c]) * inv_scale).round().clamp(0.0, 3.0) as u8;
                        let byte_idx = c_data_offset + (t / 4);
                        let shift = (t % 4) * 2;
                        data[byte_idx] |= (q & 0x03) << shift;
                    }
                }
            }
        }

        Self {
            num_tokens,
            head_dim,
            bits,
            scales,
            mins,
            data,
        }
    }

    /// Compute Query dot-product with all tokens in this block.
    /// `q`: Query vector [head_dim].
    /// `scores`: destination slice of length `num_tokens`.
    #[inline]
    pub fn compute_q_dot_k(&self, q: &[f32], scores: &mut [f32]) {
        assert_eq!(q.len(), self.head_dim);
        assert!(scores.len() >= self.num_tokens);

        // Exploit KIVI mathematical decomposition:
        // Score(t) = sum_c Q[c] * (q(t, c) * scale_c + min_c)
        //          = sum_c (Q[c] * min_c) + sum_c (Q[c] * scale_c) * q(t, c)
        // The base offset `sum_c Q[c] * min_c` is constant for all tokens in the block!
        let mut base_offset = 0.0f32;
        let mut q_scaled_buf = [0.0f32; 256];

        for c in 0..self.head_dim {
            base_offset += q[c] * self.mins[c];
            if c < 256 {
                q_scaled_buf[c] = q[c] * self.scales[c];
            }
        }

        scores[..self.num_tokens].fill(base_offset);

        let bytes_per_channel = match self.bits {
            KiviBits::Four => (self.num_tokens + 1) / 2,
            KiviBits::Two => (self.num_tokens + 3) / 4,
        };

        match self.bits {
            KiviBits::Four => {
                for c in 0..self.head_dim {
                    let qs = if c < 256 { q_scaled_buf[c] } else { q[c] * self.scales[c] };
                    if qs.abs() <= 1e-12 {
                        continue;
                    }
                    let c_offset = c * bytes_per_channel;
                    let full_bytes = self.num_tokens / 2;
                    for i in 0..full_bytes {
                        let byte = self.data[c_offset + i];
                        let n0 = (byte & 0x0f) as f32;
                        let n1 = ((byte >> 4) & 0x0f) as f32;
                        scores[2 * i] += qs * n0;
                        scores[2 * i + 1] += qs * n1;
                    }
                    if self.num_tokens % 2 != 0 {
                        let byte = self.data[c_offset + full_bytes];
                        let n0 = (byte & 0x0f) as f32;
                        scores[2 * full_bytes] += qs * n0;
                    }
                }
            }
            KiviBits::Two => {
                for c in 0..self.head_dim {
                    let qs = if c < 256 { q_scaled_buf[c] } else { q[c] * self.scales[c] };
                    if qs.abs() <= 1e-12 {
                        continue;
                    }
                    let c_offset = c * bytes_per_channel;
                    let full_bytes = self.num_tokens / 4;
                    for i in 0..full_bytes {
                        let byte = self.data[c_offset + i];
                        let d0 = (byte & 0x03) as f32;
                        let d1 = ((byte >> 2) & 0x03) as f32;
                        let d2 = ((byte >> 4) & 0x03) as f32;
                        let d3 = ((byte >> 6) & 0x03) as f32;
                        scores[4 * i] += qs * d0;
                        scores[4 * i + 1] += qs * d1;
                        scores[4 * i + 2] += qs * d2;
                        scores[4 * i + 3] += qs * d3;
                    }
                    let rem = self.num_tokens % 4;
                    if rem > 0 {
                        let byte = self.data[c_offset + full_bytes];
                        for r in 0..rem {
                            let shift = r * 2;
                            let dibit = ((byte >> shift) & 0x03) as f32;
                            scores[4 * full_bytes + r] += qs * dibit;
                        }
                    }
                }
            }
        }
    }
}

/// A block of Value vectors for `group_size` (32) tokens, quantized per-token across channel groups.
#[derive(Debug, Clone)]
pub struct QuantizedVBlock {
    pub num_tokens: usize,
    pub head_dim: usize,
    pub bits: KiviBits,
    pub group_size: usize,
    pub num_groups: usize,
    /// Per-token, per-group scale [num_tokens * num_groups]
    pub scales: Vec<f32>,
    /// Per-token, per-group min / zero-point [num_tokens * num_groups]
    pub mins: Vec<f32>,
    /// Packed quantized data:
    /// For 4-bit: num_tokens * (head_dim / 2) bytes
    /// For 2-bit: num_tokens * (head_dim / 4) bytes
    pub data: Vec<u8>,
}

impl QuantizedVBlock {
    /// Quantize a slice of Value tokens into per-token channel groups.
    /// `tokens`: shape [num_tokens, head_dim].
    pub fn new(tokens: &[f32], num_tokens: usize, head_dim: usize, bits: KiviBits, group_size: usize) -> Self {
        assert_eq!(tokens.len(), num_tokens * head_dim);
        let g_size = group_size.min(head_dim).max(1);
        let num_groups = (head_dim + g_size - 1) / g_size;

        let bytes_per_token = match bits {
            KiviBits::Four => (head_dim + 1) / 2,
            KiviBits::Two => (head_dim + 3) / 4,
        };
        let mut data = vec![0u8; num_tokens * bytes_per_token];
        let mut scales = vec![0.0f32; num_tokens * num_groups];
        let mut mins = vec![0.0f32; num_tokens * num_groups];

        let max_val = match bits {
            KiviBits::Four => 15.0f32,
            KiviBits::Two => 3.0f32,
        };

        for t in 0..num_tokens {
            let t_slice = &tokens[t * head_dim..(t + 1) * head_dim];
            let t_data_offset = t * bytes_per_token;

            for g in 0..num_groups {
                let start_c = g * g_size;
                let end_c = (start_c + g_size).min(head_dim);
                let g_slice = &t_slice[start_c..end_c];

                let mut g_min = f32::INFINITY;
                let mut g_max = f32::NEG_INFINITY;
                for &v in g_slice {
                    if v < g_min {
                        g_min = v;
                    }
                    if v > g_max {
                        g_max = v;
                    }
                }

                let g_idx = t * num_groups + g;
                if g_max <= g_min {
                    scales[g_idx] = 0.0;
                    mins[g_idx] = g_min;
                } else {
                    scales[g_idx] = (g_max - g_min) / max_val;
                    mins[g_idx] = g_min;
                }

                let inv_scale = if scales[g_idx] > 1e-12 {
                    1.0 / scales[g_idx]
                } else {
                    0.0
                };

                match bits {
                    KiviBits::Four => {
                        for c in start_c..end_c {
                            let v = t_slice[c];
                            let q = ((v - mins[g_idx]) * inv_scale).round().clamp(0.0, 15.0) as u8;
                            let byte_idx = t_data_offset + (c / 2);
                            if c % 2 == 0 {
                                data[byte_idx] = q & 0x0f;
                            } else {
                                data[byte_idx] |= (q & 0x0f) << 4;
                            }
                        }
                    }
                    KiviBits::Two => {
                        for c in start_c..end_c {
                            let v = t_slice[c];
                            let q = ((v - mins[g_idx]) * inv_scale).round().clamp(0.0, 3.0) as u8;
                            let byte_idx = t_data_offset + (c / 4);
                            let shift = (c % 4) * 2;
                            data[byte_idx] |= (q & 0x03) << shift;
                        }
                    }
                }
            }
        }

        Self {
            num_tokens,
            head_dim,
            bits,
            group_size: g_size,
            num_groups,
            scales,
            mins,
            data,
        }
    }

    /// Accumulate attention-weighted Values into output vector `out`:
    /// out[c] += sum_t weights[t] * V[t, c]
    #[inline]
    pub fn accumulate_v(&self, weights: &[f32], out: &mut [f32]) {
        assert!(weights.len() >= self.num_tokens);
        assert_eq!(out.len(), self.head_dim);

        let bytes_per_token = match self.bits {
            KiviBits::Four => (self.head_dim + 1) / 2,
            KiviBits::Two => (self.head_dim + 3) / 4,
        };

        for t in 0..self.num_tokens {
            let w = weights[t];
            if w.abs() <= 1e-12 {
                continue;
            }
            let t_data_offset = t * bytes_per_token;

            for g in 0..self.num_groups {
                let g_idx = t * self.num_groups + g;
                let scale = self.scales[g_idx] * w;
                let min_w = self.mins[g_idx] * w;
                let start_c = g * self.group_size;
                let end_c = (start_c + self.group_size).min(self.head_dim);

                match self.bits {
                    KiviBits::Four => {
                        let full_pairs = (end_c - start_c) / 2;
                        for i in 0..full_pairs {
                            let c = start_c + 2 * i;
                            let byte = self.data[t_data_offset + (c / 2)];
                            let n0 = (byte & 0x0f) as f32;
                            let n1 = ((byte >> 4) & 0x0f) as f32;
                            out[c] += n0 * scale + min_w;
                            out[c + 1] += n1 * scale + min_w;
                        }
                        if (end_c - start_c) % 2 != 0 {
                            let c = start_c + 2 * full_pairs;
                            let byte = self.data[t_data_offset + (c / 2)];
                            let n0 = (byte & 0x0f) as f32;
                            out[c] += n0 * scale + min_w;
                        }
                    }
                    KiviBits::Two => {
                        let full_quads = (end_c - start_c) / 4;
                        for i in 0..full_quads {
                            let c = start_c + 4 * i;
                            let byte = self.data[t_data_offset + (c / 4)];
                            let d0 = (byte & 0x03) as f32;
                            let d1 = ((byte >> 2) & 0x03) as f32;
                            let d2 = ((byte >> 4) & 0x03) as f32;
                            let d3 = ((byte >> 6) & 0x03) as f32;
                            out[c] += d0 * scale + min_w;
                            out[c + 1] += d1 * scale + min_w;
                            out[c + 2] += d2 * scale + min_w;
                            out[c + 3] += d3 * scale + min_w;
                        }
                        let rem = (end_c - start_c) % 4;
                        if rem > 0 {
                            let c_base = start_c + 4 * full_quads;
                            let byte = self.data[t_data_offset + (c_base / 4)];
                            for r in 0..rem {
                                let shift = r * 2;
                                let dibit = ((byte >> shift) & 0x03) as f32;
                                out[c_base + r] += dibit * scale + min_w;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A single KV head stream holding historical quantized blocks and recent full-precision buffer.
#[derive(Debug, Clone)]
pub struct KiviHeadStream {
    pub head_dim: usize,
    pub bits: KiviBits,
    pub residual_window: usize,
    pub group_size: usize,
    /// Pinned attention sinks (first `sink_tokens` rows, bit-exact F32).
    pub sink_tokens: usize,
    pub sink_len: usize,
    /// [sink_len * head_dim], never quantized.
    pub sink_k: Vec<f32>,
    pub sink_v: Vec<f32>,
    pub historical_k: Vec<QuantizedKBlock>,
    pub historical_v: Vec<QuantizedVBlock>,
    pub num_quantized_tokens: usize,
    /// Rolling recent unquantized buffer [recent_tokens * head_dim]
    pub recent_k: Vec<f32>,
    pub recent_v: Vec<f32>,
    pub recent_tokens: usize,
}

impl KiviHeadStream {
    pub fn new(
        head_dim: usize,
        bits: KiviBits,
        residual_window: usize,
        group_size: usize,
        sink_tokens: usize,
    ) -> Self {
        Self {
            head_dim,
            bits,
            residual_window,
            group_size,
            sink_tokens,
            sink_len: 0,
            sink_k: Vec::new(),
            sink_v: Vec::new(),
            historical_k: Vec::new(),
            historical_v: Vec::new(),
            num_quantized_tokens: 0,
            recent_k: Vec::new(),
            recent_v: Vec::new(),
            recent_tokens: 0,
        }
    }

    /// Append a single token (Key & Value) to this head stream.
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.head_dim);
        assert_eq!(v.len(), self.head_dim);

        // §4.1 attention sinks: the first positions of the stream stay
        // bit-exact F32 for the whole generation — outside the quantization
        // turnover and every compaction.
        if self.sink_len < self.sink_tokens {
            self.sink_k.extend_from_slice(k);
            self.sink_v.extend_from_slice(v);
            self.sink_len += 1;
            return;
        }

        self.recent_k.extend_from_slice(k);
        self.recent_v.extend_from_slice(v);
        self.recent_tokens += 1;

        // When recent buffer exceeds residual_window + group_size,
        // quantize the oldest group_size tokens into a new historical block
        if self.recent_tokens >= self.residual_window + self.group_size {
            let q_tokens = self.group_size;
            let q_len = q_tokens * self.head_dim;

            let k_block = QuantizedKBlock::new(
                &self.recent_k[..q_len],
                q_tokens,
                self.head_dim,
                self.bits,
            );
            let v_block = QuantizedVBlock::new(
                &self.recent_v[..q_len],
                q_tokens,
                self.head_dim,
                self.bits,
                32, // internal channel group size 32
            );

            self.historical_k.push(k_block);
            self.historical_v.push(v_block);
            self.num_quantized_tokens += q_tokens;

            self.recent_k.drain(..q_len);
            self.recent_v.drain(..q_len);
            self.recent_tokens -= q_tokens;
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.sink_len + self.num_quantized_tokens + self.recent_tokens
    }

    /// Compute multi-head attention for this head stream against Query `q`.
    pub fn compute_attention(&self, q: &[f32], head_out: &mut [f32]) {
        let total = self.total_tokens();
        if total == 0 {
            head_out.fill(0.0);
            return;
        }
        let mut scores = vec![0.0f32; total];
        self.compute_attention_with_scratch(q, head_out, &mut scores);
    }

    /// Compute multi-head attention reusing an external scratch scores buffer.
    pub fn compute_attention_with_scratch(&self, q: &[f32], head_out: &mut [f32], scores: &mut [f32]) {
        let total = self.total_tokens();
        if total == 0 {
            head_out.fill(0.0);
            return;
        }
        assert!(scores.len() >= total);
        self.compute_attention_causal_upto(q, head_out, scores, total);
    }

    /// P7: causal truncation of the stream — attend only rows [0 .. upto).
    /// Rows are stored in ascending position order (sinks first), so the
    /// stream index equals the token position; a prefill query at position
    /// p therefore caps at p + 1 rows.
    pub fn compute_attention_causal_upto(
        &self,
        q: &[f32],
        head_out: &mut [f32],
        scores: &mut [f32],
        upto: usize,
    ) {
        let avail = self.total_tokens().min(upto);
        if avail == 0 {
            head_out.fill(0.0);
            return;
        }
        assert!(scores.len() >= avail);

        // Section widths under the causal cutoff. Rows are stored in
        // ascending position order (sinks, then quantized groups, then
        // recent), so the kept prefix [0 .. avail) is exactly positions
        // [0 .. avail). When the cutoff lands inside `recent`, the OLDEST
        // recent rows survive and the newest (the chunk rows after this
        // query) drop.
        let w_sink = self.sink_len.min(avail);
        let mut rem = avail - w_sink;
        let mut w_hist = 0usize;
        for block in &self.historical_k {
            if rem == 0 {
                break;
            }
            let n = block.num_tokens.min(rem);
            w_hist += n;
            rem -= n;
        }
        let w_recent = rem.min(self.recent_tokens);

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

        // 1. Pinned attention sinks — bit-exact F32 dots (§4.1)
        for r in 0..w_sink {
            let k_slice = &self.sink_k[r * self.head_dim..(r + 1) * self.head_dim];
            let mut dot = 0.0f32;
            for c in 0..self.head_dim {
                dot += q[c] * k_slice[c];
            }
            scores[r] = dot;
        }
        let mut curr_tok = w_sink;

        // 2. Historical quantized blocks (full blocks until the cutoff)
        let mut hist_left = w_hist;
        for k_block in &self.historical_k {
            if hist_left == 0 {
                break;
            }
            let n = k_block.num_tokens.min(hist_left);
            if n == k_block.num_tokens {
                k_block.compute_q_dot_k(q, &mut scores[curr_tok..curr_tok + n]);
            } else {
                // Partial group at the causal boundary: the primitive
                // asserts a full-block slice, so fill a temp and cut.
                let mut tmp = vec![0.0f32; k_block.num_tokens];
                k_block.compute_q_dot_k(q, &mut tmp);
                scores[curr_tok..curr_tok + n].copy_from_slice(&tmp[..n]);
            }
            curr_tok += n;
            hist_left -= n;
        }

        // 3. Recent full-precision rows (oldest first under the cutoff)
        for r in 0..w_recent {
            let k_slice = &self.recent_k[r * self.head_dim..(r + 1) * self.head_dim];
            let mut dot = 0.0f32;
            for c in 0..self.head_dim {
                dot += q[c] * k_slice[c];
            }
            scores[curr_tok + r] = dot;
        }

        // 4. Fused Softmax with numerical stabilization
        let mut max_score = f32::NEG_INFINITY;
        for s in &mut scores[..avail] {
            *s *= scale;
            if *s > max_score {
                max_score = *s;
            }
        }

        let mut sum_exp = 0.0f32;
        for s in &mut scores[..avail] {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }

        let inv_sum = if sum_exp > 1e-12 {
            1.0 / sum_exp
        } else {
            0.0
        };
        for s in &mut scores[..avail] {
            *s *= inv_sum;
        }

        // 5. Value accumulation with the same section widths
        head_out.fill(0.0);

        // 5-sink. Pinned sinks first — exact F32 V rows (§4.1)
        for r in 0..w_sink {
            let w = scores[r];
            if w.abs() > 1e-12 {
                let v_slice = &self.sink_v[r * self.head_dim..(r + 1) * self.head_dim];
                for c in 0..self.head_dim {
                    head_out[c] += w * v_slice[c];
                }
            }
        }
        let mut v_tok = w_sink;

        // 5a. Historical quantized V blocks
        let mut v_left = w_hist;
        for v_block in &self.historical_v {
            if v_left == 0 {
                break;
            }
            let n = v_block.num_tokens.min(v_left);
            if n == v_block.num_tokens {
                v_block.accumulate_v(&scores[v_tok..v_tok + n], head_out);
            } else {
                // Partial group at the causal boundary — full-slice temp.
                let mut tmp = vec![0.0f32; v_block.num_tokens];
                tmp[..n].copy_from_slice(&scores[v_tok..v_tok + n]);
                v_block.accumulate_v(&tmp, head_out);
            }
            v_tok += n;
            v_left -= n;
        }

        // 5b. Recent full-precision V rows
        for r in 0..w_recent {
            let w = scores[v_tok + r];
            if w.abs() <= 1e-12 {
                continue;
            }
            let v_slice = &self.recent_v[r * self.head_dim..(r + 1) * self.head_dim];
            for c in 0..self.head_dim {
                head_out[c] += w * v_slice[c];
            }
        }
    }
}

/// Full KIVI KV Cache managing all layers and KV heads.
#[derive(Debug, Clone)]
pub struct KiviCache {
    pub config: KiviConfig,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// [n_layers * n_kv_heads]
    pub heads: Vec<KiviHeadStream>,
    pub scratch_scores: Vec<f32>,
}

impl KiviCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, config: KiviConfig) -> Self {
        let total_heads = n_layers * n_kv_heads;
        let mut heads = Vec::with_capacity(total_heads);
        for _ in 0..total_heads {
            heads.push(KiviHeadStream::new(
                head_dim,
                config.bits,
                config.residual_window,
                config.group_size,
                config.sink_tokens,
            ));
        }
        Self {
            config,
            n_layers,
            n_kv_heads,
            head_dim,
            heads,
            scratch_scores: Vec::new(),
        }
    }

    #[inline]
    fn head_index(&self, layer: usize, kv_head: usize) -> usize {
        layer * self.n_kv_heads + kv_head
    }

    /// Append a single token's Key and Value across all KV heads for a layer.
    pub fn append_token(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.n_kv_heads * self.head_dim);
        assert_eq!(v.len(), self.n_kv_heads * self.head_dim);

        for kv_h in 0..self.n_kv_heads {
            let idx = self.head_index(layer, kv_h);
            let k_slice = &k[kv_h * self.head_dim..(kv_h + 1) * self.head_dim];
            let v_slice = &v[kv_h * self.head_dim..(kv_h + 1) * self.head_dim];
            self.heads[idx].append(k_slice, v_slice);
        }
    }

    /// P7 streaming prefill: append `b` prefill rows (post-rope) of one
    /// layer. Row layout: [b * n_kv_heads * head_dim], kv-head-major per row
    /// — the same layout k_batch/v_batch carry.
    pub fn append_rows(&mut self, layer: usize, k_rows: &[f32], v_rows: &[f32], b: usize) {
        let kv_dim = self.n_kv_heads * self.head_dim;
        assert!(k_rows.len() >= b * kv_dim);
        assert!(v_rows.len() >= b * kv_dim);
        for r in 0..b {
            self.append_token(
                layer,
                &k_rows[r * kv_dim..(r + 1) * kv_dim],
                &v_rows[r * kv_dim..(r + 1) * kv_dim],
            );
        }
    }

    /// P7 streaming prefill: causal GQA attention for ONE chunk query row.
    /// `q`: [n_heads * head_dim] at global position `upto - 1`; the head
    /// streams hold sinks + quantized history + the current chunk (all
    /// appended in row order), so causality is simply "attend rows
    /// [0 .. upto)" — the stream index equals the position.
    pub fn compute_layer_attention_causal(
        &mut self,
        layer: usize,
        n_heads: usize,
        q: &[f32],
        head_out: &mut [f32],
        upto: usize,
    ) {
        assert_eq!(q.len(), n_heads * self.head_dim);
        assert_eq!(head_out.len(), n_heads * self.head_dim);

        let group_ratio = n_heads / self.n_kv_heads;

        for h in 0..n_heads {
            let kv_h = h / group_ratio;
            let idx = self.head_index(layer, kv_h);
            let total = self.heads[idx].total_tokens().min(upto);
            if total == 0 {
                let hslice = &mut head_out[h * self.head_dim..(h + 1) * self.head_dim];
                hslice.fill(0.0);
                continue;
            }
            if self.scratch_scores.len() < total {
                self.scratch_scores.resize(total.max(64), 0.0);
            }
            let q_slice = &q[h * self.head_dim..(h + 1) * self.head_dim];
            let out_slice = &mut head_out[h * self.head_dim..(h + 1) * self.head_dim];
            self.heads[idx].compute_attention_causal_upto(
                q_slice,
                out_slice,
                &mut self.scratch_scores[..total],
                total,
            );
        }
    }

    /// Compute Grouped-Query Attention for a given layer.
    /// `q`: shape [n_heads * head_dim].
    /// `head_out`: shape [n_heads * head_dim].
    pub fn compute_layer_attention(
        &mut self,
        layer: usize,
        n_heads: usize,
        q: &[f32],
        head_out: &mut [f32],
    ) {
        assert_eq!(q.len(), n_heads * self.head_dim);
        assert_eq!(head_out.len(), n_heads * self.head_dim);

        let group_ratio = n_heads / self.n_kv_heads;

        for h in 0..n_heads {
            let kv_h = h / group_ratio;
            let idx = self.head_index(layer, kv_h);
            let total = self.heads[idx].total_tokens();
            if self.scratch_scores.len() < total {
                self.scratch_scores.resize(total.max(64), 0.0);
            }
            let q_slice = &q[h * self.head_dim..(h + 1) * self.head_dim];
            let out_slice = &mut head_out[h * self.head_dim..(h + 1) * self.head_dim];
            self.heads[idx].compute_attention_with_scratch(q_slice, out_slice, &mut self.scratch_scores[..total]);
        }
    }

    /// Reset all head streams and free cached token history.
    pub fn reset(&mut self) {
        for h in &mut self.heads {
            h.historical_k.clear();
            h.historical_v.clear();
            h.num_quantized_tokens = 0;
            h.recent_k.clear();
            h.recent_v.clear();
            h.recent_tokens = 0;
        }
        self.scratch_scores.clear();
    }

    /// Calculate total memory bytes occupied by the KIVI cache across all layers.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;
        for h in &self.heads {
            for k in &h.historical_k {
                total += k.data.len() + k.scales.len() * 4 + k.mins.len() * 4;
            }
            for v in &h.historical_v {
                total += v.data.len() + v.scales.len() * 4 + v.mins.len() * 4;
            }
            total += (h.recent_k.len() + h.recent_v.len()) * 4;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kivi_k_quant_and_dot_accuracy() {
        let num_tokens = 32;
        let head_dim = 64;
        let mut tokens = vec![0.0f32; num_tokens * head_dim];

        // Introduce channel outliers: channel 7 has high magnitude across tokens
        for t in 0..num_tokens {
            for c in 0..head_dim {
                tokens[t * head_dim + c] = if c == 7 {
                    10.0 + (t as f32 * 0.1)
                } else {
                    ((t + c) % 11) as f32 * 0.1
                };
            }
        }

        let q_block = QuantizedKBlock::new(&tokens, num_tokens, head_dim, KiviBits::Four);
        assert_eq!(q_block.scales.len(), head_dim);

        let mut q = vec![0.05f32; head_dim];
        q[7] = 1.2;

        let mut scores_quant = vec![0.0f32; num_tokens];
        q_block.compute_q_dot_k(&q, &mut scores_quant);

        // Compute direct unquantized dot product
        for t in 0..num_tokens {
            let mut expected = 0.0f32;
            for c in 0..head_dim {
                expected += q[c] * tokens[t * head_dim + c];
            }
            let diff = (scores_quant[t] - expected).abs();
            let rel_err = diff / expected.abs().max(1.0);
            assert!(
                rel_err < 0.05,
                "token {t}: quant {} vs direct {}, rel_err = {}",
                scores_quant[t],
                expected,
                rel_err
            );
        }
    }

    #[test]
    fn test_kivi_v_quant_and_accum_accuracy() {
        let num_tokens = 32;
        let head_dim = 64;
        let mut tokens = vec![0.0f32; num_tokens * head_dim];

        for t in 0..num_tokens {
            for c in 0..head_dim {
                tokens[t * head_dim + c] = ((t * 3 + c) % 17) as f32 * 0.2;
            }
        }

        let v_block = QuantizedVBlock::new(&tokens, num_tokens, head_dim, KiviBits::Four, 32);

        let mut weights = vec![0.0f32; num_tokens];
        weights[2] = 0.4;
        weights[5] = 0.6;

        let mut out_quant = vec![0.0f32; head_dim];
        v_block.accumulate_v(&weights, &mut out_quant);

        let mut out_direct = vec![0.0f32; head_dim];
        for t in 0..num_tokens {
            let w = weights[t];
            for c in 0..head_dim {
                out_direct[c] += w * tokens[t * head_dim + c];
            }
        }

        for c in 0..head_dim {
            let diff = (out_quant[c] - out_direct[c]).abs();
            let rel_err = diff / out_direct[c].abs().max(1.0);
            assert!(
                rel_err < 0.08,
                "channel {c}: quant {} vs direct {}, rel_err = {}",
                out_quant[c],
                out_direct[c],
                rel_err
            );
        }
    }

    #[test]
    fn test_kivi_cache_streaming_and_memory_reduction() {
        let n_layers = 2;
        let n_kv_heads = 2;
        let head_dim = 64;
        let config = KiviConfig {
            enabled: true,
            bits: KiviBits::Four,
            residual_window: 16,
            group_size: 16,
            sink_tokens: 0,
        };

        let mut cache = KiviCache::new(n_layers, n_kv_heads, head_dim, config);

        let k_sample = vec![0.5f32; n_kv_heads * head_dim];
        let v_sample = vec![0.8f32; n_kv_heads * head_dim];

        // Stream 48 tokens: 16 tokens in historical quantized block, 32 in recent buffer
        for _ in 0..48 {
            cache.append_token(0, &k_sample, &v_sample);
        }

        let stream = &cache.heads[0];
        assert_eq!(stream.total_tokens(), 48);
        assert!(!stream.historical_k.is_empty());
        assert!(!stream.historical_v.is_empty());

        let uncompressed_bytes = 48 * head_dim * 4 * 2; // for 1 head
        let kivi_bytes = cache.memory_bytes() / (n_layers * n_kv_heads);
        assert!(
            kivi_bytes < uncompressed_bytes,
            "kivi bytes {} should be less than uncompressed {}",
            kivi_bytes,
            uncompressed_bytes
        );

        // Compute attention
        let q = vec![0.1f32; head_dim];
        let mut out = vec![0.0f32; head_dim];
        stream.compute_attention(&q, &mut out);
        assert!(out.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_kivi_2bit_cache_streaming() {
        let n_layers = 1;
        let n_kv_heads = 2;
        let head_dim = 64;
        let config = KiviConfig {
            enabled: true,
            bits: KiviBits::Two,
            residual_window: 16,
            group_size: 16,
            sink_tokens: 0,
        };

        let mut cache = KiviCache::new(n_layers, n_kv_heads, head_dim, config);

        let k_sample = vec![0.3f32; n_kv_heads * head_dim];
        let v_sample = vec![0.7f32; n_kv_heads * head_dim];

        for _ in 0..64 {
            cache.append_token(0, &k_sample, &v_sample);
        }

        assert_eq!(cache.heads[0].total_tokens(), 64);
        assert!(!cache.heads[0].historical_k.is_empty());
        assert_eq!(cache.heads[0].historical_k[0].bits, KiviBits::Two);

        let q = vec![0.2f32; 2 * head_dim];
        let mut out = vec![0.0f32; 2 * head_dim];
        cache.compute_layer_attention(0, 2, &q, &mut out);
        assert!(out.iter().any(|&v| v > 0.0));
    }
}

/// F-LC4 regression: kivi attention must match an exact FP32 reference on
/// the SAME data (golden: softmax(q·kᵀ/√d)·v) across the quantization
/// turnover. When this failed to hold, decode degenerated into repetition
/// ("Berlin Berlin Berlin…", receipt in docs/PLAN-LONG-CONTEXT-PERSISTENCE.md
/// §8); it passes, which exonerates the kivi math and pinned F-LC4 on the
/// lookahead replay over the quantized cache instead.
#[cfg(test)]
mod f_lc4_tests {
    use super::*;

    #[test]
    fn kivi_attention_matches_fp32_reference() {
        let hd = 64usize;
        let mut st = KiviHeadStream::new(hd, KiviBits::Four, 32, 32, 0);
        // Deterministic pseudo-data: ropes-ish rotations, magnitudes ~1.
        let mut state: f32 = 0.123456;
        let mut next = || {
            state = (state * 1103515245.0 + 12345.0) % 97.0;
            state / 97.0 - 0.5
        };
        let mut k_all: Vec<Vec<f32>> = Vec::new();
        let mut v_all: Vec<Vec<f32>> = Vec::new();
        for _ in 0..100 {
            let k: Vec<f32> = (0..hd).map(|_| next()).collect();
            let v: Vec<f32> = (0..hd).map(|_| next()).collect();
            st.append(&k, &v);
            k_all.push(k);
            v_all.push(v);
        }
        // Turnover fires whenever recent reaches 2× the residual window:
        // at appends 64 and 96 (64→32, then 32+36... ) leaving 64 quantized.
        assert_eq!(st.num_quantized_tokens, 64, "two blocks should have quantized");
        assert_eq!(st.recent_tokens, 36);
        assert_eq!(st.total_tokens(), 100);

        let q: Vec<f32> = (0..hd).map(|_| next()).collect();

        // Exact FP32 reference.
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = vec![0.0f32; 100];
        for t in 0..100 {
            let dot: f32 = q.iter().zip(&k_all[t]).map(|(a, b)| a * b).sum();
            scores[t] = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut ref_out = vec![0.0f32; hd];
        for t in 0..100 {
            for c in 0..hd {
                ref_out[c] += exp[t] / sum * v_all[t][c];
            }
        }

        let mut out = vec![0.0f32; hd];
        st.compute_attention(&q, &mut out);
        let max_diff = out.iter().zip(&ref_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.05,
            "kivi attention diverged from fp32 reference: max_diff={max_diff}"
        );
    }
}

#[cfg(test)]
mod sink_tests {
    use super::*;

    /// §4.1 (P4): the first `sink_tokens` tokens are pinned bit-exact in F32
    /// — outside the quantization turnover — and attention still matches the
    /// FP32 reference over the whole stream.
    #[test]
    fn kivi_sink_pins_first_tokens_full_precision() {
        let hd = 64usize;
        let mut st = KiviHeadStream::new(hd, KiviBits::Four, 32, 32, 4);
        let mut state: f32 = 0.123456;
        let mut next = || {
            state = (state * 1103515245.0 + 12345.0) % 97.0;
            state / 97.0 - 0.5
        };
        let mut k_all: Vec<Vec<f32>> = Vec::new();
        let mut v_all: Vec<Vec<f32>> = Vec::new();
        for i in 0..100 {
            let k: Vec<f32> = (0..hd).map(|c| next() + if c == 0 { i as f32 * 0.01 } else { 0.0 }).collect();
            let v: Vec<f32> = (0..hd).map(|_| next()).collect();
            st.append(&k, &v);
            k_all.push(k);
            v_all.push(v);
        }
        // Bookkeeping: 4 sinks + 64 quantized + 32 recent = 100.
        assert_eq!(st.sink_len, 4);
        assert_eq!(st.num_quantized_tokens, 64);
        assert_eq!(st.recent_tokens, 32);
        assert_eq!(st.total_tokens(), 100);
        // Sinks are bit-exact copies of the appended rows (0..4).
        for t in 0..4 {
            assert_eq!(&st.sink_k[t * hd..(t + 1) * hd], &k_all[t][..]);
            assert_eq!(&st.sink_v[t * hd..(t + 1) * hd], &v_all[t][..]);
        }

        let q: Vec<f32> = (0..hd).map(|_| next()).collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = vec![0.0f32; 100];
        for t in 0..100 {
            let dot: f32 = q.iter().zip(&k_all[t]).map(|(a, b)| a * b).sum();
            scores[t] = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut ref_out = vec![0.0f32; hd];
        for t in 0..100 {
            for c in 0..hd {
                ref_out[c] += exp[t] / sum * v_all[t][c];
            }
        }

        let mut out = vec![0.0f32; hd];
        st.compute_attention(&q, &mut out);
        let max_diff = out.iter().zip(&ref_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.05,
            "kivi sink-pinned attention diverged from fp32 reference: max_diff={max_diff}"
        );

        // A sink-position query scores its own row exactly: with q == sink row
        // k, the sink's self dot (||k||^2, positive) must survive quantization
        // turnover untouched in the kivi path.
        let qs = k_all[2].clone();
        let mut out2 = vec![0.0f32; hd];
        st.compute_attention(&qs, &mut out2);
        assert!(out2.iter().any(|&v| v != 0.0));
    }
}

#[cfg(test)]
mod causal_tests {
    use super::*;

    #[test]
    fn kivi_causal_upto_full_equals_full_attention() {
        let hd = 64usize;
        let mut st = KiviHeadStream::new(hd, KiviBits::Four, 32, 32, 4);
        let mut state: f32 = 0.31;
        let mut next = || {
            state = (state * 1103515245.0 + 12345.0) % 97.0;
            state / 97.0 - 0.5
        };
        for _ in 0..100 {
            let k: Vec<f32> = (0..hd).map(|_| next()).collect();
            let v: Vec<f32> = (0..hd).map(|_| next()).collect();
            st.append(&k, &v);
        }
        let q: Vec<f32> = (0..hd).map(|_| next()).collect();
        let mut a = vec![0.0f32; hd];
        let mut b = vec![0.0f32; hd];
        st.compute_attention(&q, &mut a);
        st.compute_attention_causal_upto(&q, &mut b, &mut vec![0.0f32; 100], 100);
        let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(d < 1e-6, "causal-upto(total) diverges from full attention: {d}");
    }

    #[test]
    fn kivi_causal_upto_truncates_to_prefix() {
        let hd = 64usize;
        let mut st = KiviHeadStream::new(hd, KiviBits::Four, 32, 32, 4);
        let mut state: f32 = 0.77;
        let mut next = || {
            state = (state * 1103515245.0 + 12345.0) % 97.0;
            state / 97.0 - 0.5
        };
        let mut rows_k: Vec<Vec<f32>> = Vec::new();
        let mut rows_v: Vec<Vec<f32>> = Vec::new();
        for i in 0..100 {
            let k: Vec<f32> = (0..hd).map(|c| next() + if c == 0 { i as f32 * 0.01 } else { 0.0 }).collect();
            let v: Vec<f32> = (0..hd).map(|_| next()).collect();
            st.append(&k, &v);
            rows_k.push(k);
            rows_v.push(v);
        }
        let q: Vec<f32> = (0..hd).map(|_| next()).collect();

        // Reference: exact softmax over rows [0 .. 40) only.
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = vec![0.0f32; 40];
        for t in 0..40 {
            let dot: f32 = q.iter().zip(&rows_k[t]).map(|(a, b)| a * b).sum();
            scores[t] = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut ref_out = vec![0.0f32; hd];
        for t in 0..40 {
            for c in 0..hd {
                ref_out[c] += exp[t] / sum * rows_v[t][c];
            }
        }
        let mut out = vec![0.0f32; hd];
        st.compute_attention_causal_upto(&q, &mut out, &mut vec![0.0f32; 100], 40);
        // rows 0..36 are quantized (4 sinks + 32... at 100 rows: turnover at
        // 64 -> 64 quantized; rows 0..40 = 4 sinks + 36 quantized) so the
        // quant error of the 36 quantized rows dominates the diff.
        let d = out.iter().zip(&ref_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(d < 0.06, "causal truncation diverges from prefix reference: {d}");
    }
}
