// crates/sofuu-core/src/infer/sampler.rs — Fast logit sampling strategies.
//
// Supports greedy argmax, temperature scaling, top-p (nucleus), top-k, and repetition penalty.

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
        }
    }
}

pub struct Sampler {
    config: SamplerConfig,
    rng_state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        // Seed PRNG with current timestamp or fixed state
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x853c49e6748fea9b);
        Self {
            config,
            rng_state: seed ^ 0x9e3779b97f4a7c15,
        }
    }

    /// Fast xorshift64* pseudo-random number generator.
    fn next_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        let val = self.rng_state.wrapping_mul(0x2545F4914F6CDD1D);
        ((val >> 40) as f32) / 16777216.0
    }

    #[inline(always)]
    fn sift_down_min(heap: &mut [(u32, f32)], mut idx: usize) {
        let n = heap.len();
        while (2 * idx + 1) < n {
            let left = 2 * idx + 1;
            let right = left + 1;
            let mut smallest = idx;
            if heap[left].1 < heap[smallest].1 {
                smallest = left;
            }
            if right < n && heap[right].1 < heap[smallest].1 {
                smallest = right;
            }
            if smallest != idx {
                heap.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    fn heapify_min(heap: &mut [(u32, f32)]) {
        let n = heap.len();
        for i in (0..n / 2).rev() {
            Self::sift_down_min(heap, i);
        }
    }

    /// Sample the next token from logits given the history of previous tokens.
    /// Uses a bounded min-heap for Top-K extraction to eliminate full-vocab sorting
    /// and millions of unnecessary exp() evaluations.
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        // Greedy decoding if temperature is near zero
        if self.config.temperature <= 0.01 {
            return self.argmax(logits);
        }

        // 1. Repetition penalty (applied to recent unique tokens)
        if self.config.repetition_penalty > 1.0 && !history.is_empty() {
            let window_start = history.len().saturating_sub(64);
            for &prev_tok in &history[window_start..] {
                if let Some(l) = logits.get_mut(prev_tok as usize) {
                    if *l < 0.0 {
                        *l *= self.config.repetition_penalty;
                    } else {
                        *l /= self.config.repetition_penalty;
                    }
                }
            }
        }

        // 2. Extract Top-K candidates using a bounded min-heap of size K
        let k = if self.config.top_k > 0 { self.config.top_k } else { 40 }.min(logits.len());
        let mut top_candidates = Vec::with_capacity(k + 1);

        for (i, &l) in logits.iter().enumerate() {
            if top_candidates.len() < k {
                top_candidates.push((i as u32, l));
                if top_candidates.len() == k {
                    Self::heapify_min(&mut top_candidates);
                }
            } else if l > top_candidates[0].1 {
                top_candidates[0] = (i as u32, l);
                Self::sift_down_min(&mut top_candidates, 0);
            }
        }

        if top_candidates.is_empty() {
            return 0;
        }

        // 3. Sort ONLY the K candidates in descending order
        top_candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Temperature scaling + Stable Softmax on ONLY the K candidates
        let inv_temp = 1.0 / self.config.temperature;
        let max_l = top_candidates[0].1 * inv_temp;
        let mut sum_exp = 0.0f32;
        for c in &mut top_candidates {
            c.1 = ((c.1 * inv_temp) - max_l).exp();
            sum_exp += c.1;
        }
        let inv_sum = 1.0 / (sum_exp + 1e-10);
        for c in &mut top_candidates {
            c.1 *= inv_sum;
        }

        // 5. Nucleus (top-p) cutoff
        if self.config.top_p < 1.0 {
            let mut cum_p = 0.0f32;
            let mut cutoff_idx = top_candidates.len();
            for (idx, (_, p)) in top_candidates.iter().enumerate() {
                cum_p += p;
                if cum_p >= self.config.top_p && idx > 0 {
                    cutoff_idx = idx + 1;
                    break;
                }
            }
            top_candidates.truncate(cutoff_idx);
        }

        // 6. Renormalize and sample
        let total_p: f32 = top_candidates.iter().map(|(_, p)| p).sum();
        let rand_val = self.next_f32() * total_p;

        let mut acc = 0.0f32;
        for (token_id, p) in &top_candidates {
            acc += p;
            if rand_val <= acc {
                return *token_id;
            }
        }

        top_candidates.first().map(|(id, _)| *id).unwrap_or(0)
    }

    #[inline(always)]
    fn argmax(&self, logits: &[f32]) -> u32 {
        let mut max_idx = 0;
        let mut max_val = logits[0];
        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_argmax() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        let mut logits = vec![1.0, 5.0, 2.0, 4.0];
        assert_eq!(sampler.sample(&mut logits, &[]), 1);
    }

    #[test]
    fn test_sampler_top_k() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
        });
        let mut logits = vec![0.1, 10.0, 0.2, 9.0, 0.05];
        let chosen = sampler.sample(&mut logits, &[]);
        assert!(chosen == 1 || chosen == 3);
    }
}
