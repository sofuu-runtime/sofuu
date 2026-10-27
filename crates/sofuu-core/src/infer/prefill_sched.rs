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

/// -- STICKY PREFILL LEASE (chunked prefill across concurrent streams) --------
///
/// WHY THIS EXISTS
/// A model backend owns exactly ONE KV cache (k_cache / v_cache), indexed by
/// ABSOLUTE position: forward_prefill_batch(slice, start) writes rows
/// [start, start+len) and every row's attention reads positions [0, i). There
/// is no per-stream slot mapping in the serving path (the BlockPool /
/// SequenceBlockTable bookkeeping in continuous.rs is not consulted by any
/// forward), so two streams that take prefill slices in one scheduler pass
/// write the SAME positions and each then attends over the other's rows.
///
/// Concretely, with two prompts prefilling at once and the default 256-token
/// slice: pass 1 gives stream A [0,256) and then stream B [0,256) -- B's rows
/// land on top of A's -- and in pass 2 A resumes at cur_pos = 256 with its
/// first 256 positions holding B's tokens. A's logits then describe a document
/// made of half of each prompt. The same aliasing hits a decoding stream whose
/// positions fall inside another stream's prefill range.
///
/// WHAT THE LEASE IS
/// A stream that has already written KV for its prompt (cur_pos > 0 while
/// Prefilling) keeps the cache until its prompt is fully prefilled
/// (SliceStep::Complete). Exactly one prefilling stream takes a slice per
/// scheduler pass, so slices from different streams cannot interleave into one
/// cache. A stream that has not started yet (cur_pos == 0) waits, keeping its
/// status and queue position, until the holder completes.
///
/// The decision is a pure function of the pass's stream set -- no extra state,
/// no lock, nothing to leak on abort: the holder simply stops being a
/// candidate when it leaves the Prefilling arm. With ONE stream the owner is
/// that stream, so the pass is byte-for-byte the pre-lease pass.
///
/// SOFUU_PREFILL_LEASE=0 restores the old interleaving (the A/B arm for the
/// corruption receipt); the default is on.
pub fn lease_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SOFUU_PREFILL_LEASE")
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// The single stream allowed to take a prefill slice this pass, or None when no
/// stream is prefilling. streams is (stream id, is Prefilling, cur_pos, prompt
/// tokens still to prefill).
///
/// Order of precedence:
/// 1. a stream ALREADY mid-prefill (cur_pos > 0) -- it holds the cache, and the
///    largest cur_pos wins so the most-invested prompt finishes first;
/// 2. otherwise the shortest remaining prompt, matching the pass's
///    shortest-remaining-first admission order, so a one-line question is not
///    queued behind a 3k-token prefill.
/// Ties break on the lowest stream id (FIFO), so the choice is deterministic.
pub fn cache_owner<I>(streams: I) -> Option<u64>
where
    I: IntoIterator<Item = (u64, bool, usize, usize)>,
{
    let mut mid: Option<(usize, u64)> = None;
    let mut fresh: Option<(usize, u64)> = None;
    for (id, prefilling, cur_pos, remaining) in streams {
        if !prefilling {
            continue;
        }
        if cur_pos > 0 {
            let better = match mid {
                None => true,
                Some((p, i)) => cur_pos > p || (cur_pos == p && id < i),
            };
            if better {
                mid = Some((cur_pos, id));
            }
        } else {
            let better = match fresh {
                None => true,
                Some((r, i)) => remaining < r || (remaining == r && id < i),
            };
            if better {
                fresh = Some((remaining, id));
            }
        }
    }
    mid.or(fresh).map(|(_, id)| id)
}

#[cfg(test)]
mod lease_tests {
    use super::*;

    /// One test rather than four: the rule being pinned is that the prefill
    /// cache has exactly ONE owner, and the four clauses only make sense
    /// together (a rule that picks an owner for one stream but drops the
    /// mid-prefill stream would still pass any single assertion).
    #[test]
    fn the_prefill_cache_lease_has_one_owner_at_a_time() {
        // A stream that already holds KV for positions > 0 keeps the cache
        // (resuming it elsewhere would rewrite another stream's KV), even
        // against a fresh stream with less work left.
        assert_eq!(
            cache_owner(vec![(1u64, true, 0usize, 40usize), (2, true, 512, 700)]),
            Some(2)
        );
        // With nobody mid-prefill, the stream closest to done goes first.
        assert_eq!(
            cache_owner(vec![(1u64, true, 0usize, 3000usize), (2, true, 0, 20)]),
            Some(2)
        );
        // A decoding stream is not a prefill owner (and if every stream is
        // decoding, there is no lease at all: the prefill path is idle).
        assert_eq!(
            cache_owner(vec![(1u64, false, 4096usize, 0usize), (2, true, 0, 64)]),
            Some(2)
        );
        assert_eq!(cache_owner(vec![(1u64, false, 4096, 0)]), None);
        // Membership order must not matter: ties fall to the lowest id.
        assert_eq!(
            cache_owner(vec![(7u64, true, 256usize, 900usize), (3, true, 256, 500)]),
            Some(3)
        );
        assert_eq!(
            cache_owner(vec![(9u64, true, 0usize, 64usize), (4, true, 0, 64)]),
            Some(4)
        );
    }
}

