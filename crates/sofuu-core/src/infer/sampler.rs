// crates/sofuu-core/src/infer/sampler.rs — Fast logit sampling strategies.
//
// Supports greedy argmax, temperature scaling, top-p (nucleus), top-k, and
// repetition penalty, plus the entropy-adaptive presence penalty of
// PLAN-LONG-CONTEXT-PERSISTENCE §4.3: long contexts drift into repetition
// loops, and a static repetition_penalty over the last 64 tokens is too
// myopic to notice. The adaptive penalty watches the unigram entropy of a
// 1,024-token window; only when it collapses below the 2.4-bit threshold
// does it penalise every token already seen in the window (λ = 1.15,
// same sign convention as the repetition penalty). Healthy prose sits at
// 6-8 bits, so normal sampling is bit-for-bit unchanged — including greedy
// runs, where the penalty is applied BEFORE argmax so a locked loop can
// actually break. SOFUU_ENTROPY_PENALTY=0 disables it.

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

/// §4.3 constants: window, entropy floor (bits), presence-penalty λ.
pub const ENTROPY_WINDOW: usize = 1024;
pub const ENTROPY_THRESHOLD_BITS: f32 = 2.4;
pub const ENTROPY_PRESENCE_LAMBDA: f32 = 1.15;
/// The judge needs this many GENERATED tokens before it may penalise.
pub const MIN_GENERATED_BEFORE_PENALTY: usize = 64;

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
    /// History length at the FIRST sample() call = the prompt. The entropy
    /// window must watch only what the model itself generated: a code-heavy
    /// PROMPT tail has sub-2.4-bit unigram entropy by nature, and penalising
    /// over it flipped healthy greedy runs to immediate EOS (caught live by
    /// scripts/bench_niah.py — every 8k depth returned zero tokens).
    prompt_len: Option<usize>,
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
            prompt_len: None,
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

        // 0. Entropy-adaptive presence penalty (§4.3). Runs before the greedy
        // short-circuit so a temp-0 repetition loop can be broken; with a
        // healthy window entropy it is a no-op.
        let prompt_len = *self.prompt_len.get_or_insert(history.len());
        Self::apply_entropy_penalty(logits, history, prompt_len);

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

    /// Shannon entropy (bits) of the unigram distribution over the last
    /// `ENTROPY_WINDOW` GENERATED tokens (everything after `prompt_len`).
    /// Public so tests and harnesses can assert the exact quantity the gate
    /// uses.
    pub fn window_entropy(history: &[u32], prompt_len: usize) -> f32 {
        let gen = &history[prompt_len.min(history.len())..];
        let window = &gen[gen.len().saturating_sub(ENTROPY_WINDOW)..];
        if window.is_empty() {
            return f32::INFINITY;
        }
        let mut counts: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::with_capacity(256);
        for &t in window {
            *counts.entry(t).or_insert(0) += 1;
        }
        let n = window.len() as f32;
        let mut h = 0.0f32;
        for &c in counts.values() {
            let p = c as f32 / n;
            h -= p * p.log2();
        }
        h
    }

    /// §4.3: if the window entropy has collapsed below the threshold,
    /// divide/multiply the logits of every token seen in the window by
    /// λ = 1.15 (the presence-penalty sign convention). A no-op when the
    /// window is healthy or the kill switch is set.
    fn apply_entropy_penalty(logits: &mut [f32], history: &[u32], prompt_len: usize) {
        if history.is_empty() || logits.is_empty() {
            return;
        }
        if let Ok(v) = std::env::var("SOFUU_ENTROPY_PENALTY") {
            if v == "0" {
                return;
            }
        }
        let gen = &history[prompt_len.min(history.len())..];
        // Too little generated text to judge — never penalise off the prompt.
        if gen.len() < MIN_GENERATED_BEFORE_PENALTY {
            return;
        }
        if Self::window_entropy(history, prompt_len) >= ENTROPY_THRESHOLD_BITS {
            return;
        }
        let window = &gen[gen.len().saturating_sub(ENTROPY_WINDOW)..];
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::with_capacity(256);
        for &t in window {
            if t as usize >= logits.len() {
                continue;
            }
            if seen.insert(t) {
                let l = &mut logits[t as usize];
                if *l < 0.0 {
                    *l *= ENTROPY_PRESENCE_LAMBDA;
                } else {
                    *l /= ENTROPY_PRESENCE_LAMBDA;
                }
            }
        }
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

    /// §4.3: known entropies (prompt_len = 0 here).
    #[test]
    fn test_window_entropy_known_values() {
        let hist = vec![7u32; 512];
        assert!(Sampler::window_entropy(&hist, 0) <= 1e-4, "constant window → 0 bits");
        let mut two = vec![7u32; 512];
        two.extend(std::iter::repeat(9u32).take(512));
        let h = Sampler::window_entropy(&two, 0);
        assert!((h - 1.0).abs() < 1e-3, "50/50 window → 1 bit, got {h}");
        // Only the last ENTROPY_WINDOW generated tokens count.
        let mut padded = vec![1u32; ENTROPY_WINDOW];
        padded.extend(std::iter::repeat(7u32).take(64));
        assert!(Sampler::window_entropy(&padded, 0) < 0.35,
            "window is the tail, got {}", Sampler::window_entropy(&padded, 0));
        // The prompt is excluded: same tail, huge prompt prefix, same answer.
        let mut with_prompt = vec![3u32; 4096];
        with_prompt.extend_from_slice(&padded);
        assert!((Sampler::window_entropy(&with_prompt, 4096)
                 - Sampler::window_entropy(&padded, 0)).abs() < 1e-4);
        // Healthy varied text sits well above the 2.4-bit threshold.
        let varied: Vec<u32> = (0..600u32).collect();
        assert!(Sampler::window_entropy(&varied, 0) > 8.0);
    }

    /// The live failure that forced this design: a low-entropy (code-heavy)
    /// PROMPT tail must never trigger the penalty — only generated text is
    /// judged, and a fresh generation has nothing to judge yet.
    #[test]
    fn test_entropy_penalty_ignores_prompt_tail() {
        let code_tail: Vec<u32> = std::iter::repeat(5u32).take(1024).collect(); // H = 0 bits
        let mut sampler = Sampler::new(SamplerConfig { temperature: 0.0, ..Default::default() });
        let mut logits = vec![4.5f32, 5.0, 3.0];
        assert_eq!(sampler.sample(&mut logits, &code_tail), 1,
            "prompt-only history → argmax untouched");
        assert_eq!(logits, vec![4.5, 5.0, 3.0], "prompt-only history → logits untouched");
    }

    /// Under MIN_GENERATED_BEFORE_PENALTY generated tokens the judge stays
    /// silent even if those few tokens repeat.
    #[test]
    fn test_entropy_penalty_needs_minimum_generated() {
        let hist: Vec<u32> = std::iter::repeat(1u32).take(200).collect(); // 200-token prompt…
        let mut sampler = Sampler::new(SamplerConfig { temperature: 0.0, ..Default::default() });
        let mut logits = vec![4.5f32, 5.0, 3.0];
        // …first sample() pins prompt_len=200; a loop of <64 generated tokens
        // cannot trigger yet.
        let short: Vec<u32> = hist.iter().cloned().chain(std::iter::repeat(1u32).take(32)).collect();
        assert_eq!(sampler.sample(&mut logits, &short), 1);
        assert_eq!(logits, vec![4.5, 5.0, 3.0]);
    }

    /// §4.3: a collapsed window breaks even a GREEDY loop — the penalty runs
    /// before argmax, and the looped token loses to an equal-logit rival.
    /// The kill switch is exercised in the SAME test (env is process-global;
    /// two parallel #[test]s racing set_var would flake).
    #[test]
    fn test_entropy_penalty_breaks_greedy_loop_and_kill_switch() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        // The first call pins prompt_len = 1 (a one-token prompt); the model
        // has then generated a 511-token loop of token 1.
        let mut pin = vec![0.0f32, 0.0, 0.0];
        let _ = sampler.sample(&mut pin, &[9u32]);
        let mut looping: Vec<u32> = vec![9u32];
        looping.extend(std::iter::repeat(1u32).take(511));

        std::env::set_var("SOFUU_ENTROPY_PENALTY", "1");
        let mut logits = vec![4.5f32, 5.0, 3.0];
        let tok = sampler.sample(&mut logits, &looping);
        // Token 1's positive logit was divided by λ (5.0 → 4.35); the unseen
        // token 0 (4.5) stays put and wins.
        assert_eq!(tok, 0);
        assert!(logits[1] < logits[0]);

        // SOFUU_ENTROPY_PENALTY=0 restores pre-§4.3 behavior exactly: the
        // un-penalized argmax (token 1) wins again.
        std::env::set_var("SOFUU_ENTROPY_PENALTY", "0");
        let mut logits2 = vec![4.5f32, 5.0, 3.0];
        assert_eq!(sampler.sample(&mut logits2, &looping), 1, "kill switch → argmax unchanged");
        std::env::remove_var("SOFUU_ENTROPY_PENALTY");
    }

    /// A healthy window is untouched: same argmax AND same logit values.
    #[test]
    fn test_entropy_penalty_is_noop_on_healthy_windows() {
        let varied: Vec<u32> = (0..600u32).collect();
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        let mut logits = vec![1.0f32, 5.0, 2.0, 4.0];
        let before = logits.clone();
        let tok = sampler.sample(&mut logits, &varied);
        assert_eq!(tok, 1);
        assert_eq!(logits, before, "healthy entropy must not perturb logits");
    }

}
