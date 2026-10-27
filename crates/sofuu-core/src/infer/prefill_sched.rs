// crates/sofuu-core/src/infer/prefill_sched.rs — Chunked, interleavable prefill.
//
// WHY
// The continuous scheduler used to run a whole prompt through the model inside
// ONE iteration (continuous.rs, Prefilling arm ->
// forward_prefill_all_with_logits(prompt_tokens, 0)). Every other stream in the
// iteration — an actively decoding one, or a short request that arrived while
// the long prompt was still prefilling — waited for the entire prompt. A 3k
// token prompt is seconds of wall clock, so a one-line question behind it paid
// the full prefill.
//
// WHAT
// The Prefilling arm now consumes at most one SLICE of the prompt per scheduler
// iteration. A slice is a run of whole backend prefill groups (see
// group_rows), so the sequence of batched calls the model sees is exactly the
// sequence it would have seen in a single one-shot call — chunking moves the
// boundaries between scheduler iterations, never between rows of a batch.
// Slice size: SOFUU_PREFILL_LIMIT when set (the knob benchmarks already export;
// it also truncates the non-continuous stream() path), else DEFAULT_CHUNK.
// SOFUU_PREFILL_LIMIT=0 keeps the old whole-prompt behaviour for A/B runs.
// The value is rounded UP to a whole number of backend prefill groups.
//
// EXACTNESS ARGUMENT (why the logits cannot move)
// For a prompt of N tokens every backend's forward_prefill_all_with_logits
// already splits the work into groups: the leading groups of G rows with the
// last-token projection OFF, then the final (N mod G, or G) rows through the
// group that computes the projection. Those groups are a function of N and G
// alone — not of where the caller stopped. Slicing at multiples of G therefore
// issues the identical calls in the identical order: intermediate slices run
// through forward_prefill_batch (same group loop, projection off) and the last
// slice through forward_prefill_all_with_logits, which reproduces the tail
// exactly.

use std::sync::OnceLock;

use super::{InferModel, ModelBackend};

/// Rows per batched model call for the default slice when SOFUU_PREFILL_LIMIT
/// is unset. 256 = two 128-row groups on the llama-class backends.
pub const DEFAULT_CHUNK: usize = 256;

/// Rows per batched model call inside the backend's own prefill loop. A slice
/// boundary that is not a multiple of this re-groups rows, so chunk sizes are
/// always rounded up to a whole number of groups.
pub fn group_rows(model: &InferModel) -> usize {
    match &model.backend {
        // Both loop with max_prefill_batch() (SOFUU_PREFILL_BATCH, 4..=512).
        ModelBackend::Gguf(_) | ModelBackend::Spark(_) => super::model::max_prefill_batch().max(1),
        // forward_batch_impl() strides PRE_FILL_CHUNK (128) from the slice start.
        ModelBackend::Qwen35(_) => 128,
        // forward_prefill_batch()/all_with_logits() stride PRE_CHUNK (256).
        ModelBackend::BlockGraph(_) => 256,
        // gptx3 batches inside its own stride; treat that stride as the group.
        ModelBackend::Gptx3(_) => super::model::max_prefill_batch().max(1),
        // lfm2 batches any slice >= LFM2_BATCH_MIN (8) as one panel, so a slice
        // boundary IS a batch boundary; keep slices a multiple of 8.
        ModelBackend::Lfm2(_) => 8,
    }
}

/// Round a requested slice to whole backend groups. 0 stays 0 (one shot).
pub fn resolve_chunk(requested: usize, group: usize) -> usize {
    if requested == 0 {
        return 0;
    }
    let g = group.max(1);
    requested.div_ceil(g) * g
}

/// The slice size for this model: SOFUU_PREFILL_LIMIT (0 = whole prompt) else
/// DEFAULT_CHUNK, in both cases rounded up to whole backend groups.
pub fn chunk_tokens(model: &InferModel) -> usize {
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        let requested = std::env::var("SOFUU_PREFILL_LIMIT")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_CHUNK);
        resolve_chunk(requested, group_rows(model))
    })
}

/// Result of one prefill slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceStep {
    /// KV written for [start, start + n); the prompt has more tokens left.
    Continue,
    /// Prompt fully prefilled; logits_out holds the last token's full-vocab row.
    Complete,
}

/// Prefill at most chunk tokens of tokens starting at start, writing the last
/// token's logits into logits_out when the prompt completes.
///
/// start is the number of prompt tokens already in the KV cache (0 on the
/// first call). Chunking is exact only for a chunk that is a multiple of
/// group_rows(); chunk_tokens() guarantees that.
pub fn step(
    model: &mut InferModel,
    tokens: &[u32],
    start: usize,
    chunk: usize,
    logits_out: &mut [f32],
) -> SliceStep {
    let n = tokens.len();
    if n == 0 {
        return SliceStep::Complete;
    }
    if n == 1 {
        // The pre-existing single-token fast path, unchanged.
        let logits = model.forward(tokens[0], 0);
        let m = logits_out.len().min(logits.len());
        logits_out[..m].copy_from_slice(&logits[..m]);
        return SliceStep::Complete;
    }
    if start >= n {
        // Already prefilled (defensive: the caller sets cur_pos = n on Complete).
        return SliceStep::Complete;
    }
    let remaining = n - start;
    let take = if chunk == 0 { remaining } else { chunk.min(remaining) };
    let slice = &tokens[start..start + take];
    if take == remaining {
        let logits = model.forward_prefill_all_with_logits(slice, start);
        let m = logits_out.len().min(logits.len());
        logits_out[..m].copy_from_slice(&logits[..m]);
        SliceStep::Complete
    } else {
        model.forward_prefill_batch(slice, start);
        SliceStep::Continue
    }
}

/// Iteration ordering key for one scheduler pass: decode streams first (their
/// step is already short and latency-critical), then prefilling streams
/// shortest-remaining-first so a 20-token question is not queued behind the
/// 11th slice of a 3k-token prompt. Stable within each class, so FIFO is kept
/// among equals.
#[inline]
pub fn order_key(prefilling: bool, remaining_tokens: usize) -> (u8, usize) {
    if prefilling {
        (1, remaining_tokens)
    } else {
        (0, 0)
    }
}
