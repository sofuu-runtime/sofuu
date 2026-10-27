// crates/sofuu-core/src/infer/continuous.rs — Serverless Continuous Batching Engine.
//
// Replaces coarse global engine mutexes with an iteration-level continuous batching scheduler.
// Dynamically admits requests, interleaves token generation across active streams,
// manages paged KV block tables, and puts the inference worker to sleep when idle (0% CPU).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use super::paged_kv::{BlockPool, SequenceBlockTable};
use super::prefill_sched::{self, SliceStep};
use super::sampler::{Sampler, SamplerConfig};
use super::{GenerateOptions, InferEngine};

// ── Phase-aware kernel-pool sizing ───────────────────────────────
// Prefill and decode want different teams. Prefill is compute-bound
// (batched GEMM over the whole prompt) and speeds up nearly linearly
// with threads on the performance cluster; decode is bandwidth-bound
// (every weight byte crosses memory each token) and has a tuned
// count resembling 4-6. The pool is global, so each scheduling
// iteration snapshots the phase of work it is about to run and resizes
// at most when the value changed; the rebuild is sub-millisecond and
// happens twice per turn at most.
//
// Both are host-settable:
//   SOFUU_PREFILL_THREADS / SOFUU_DECODE_THREADS — explicit
//   SOFUU_NUM_THREADS — decode fallback (the app's CPU-threads setting)

/// Read a thread-count env var if present and sane.
fn env_threads(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|n| *n > 0 && *n <= 8)
}

#[derive(PartialEq, Clone, Copy)]
enum PoolPhase { Prefill, Decode }

fn pool_threads_for(phase: PoolPhase) -> usize {
    let online = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    match phase {
        PoolPhase::Prefill => env_threads("SOFUU_PREFILL_THREADS")
            // Default = the kernel pool's own mobile default (4 = the big
            // cluster). More threads only help with UNEVEN chunk splits —
            // parallel_for splits equally, so every little-core worker in
            // the barrier caps prefill at that core's pace. 8-16 threads
            // remain available via env for kernels that support them.
            .unwrap_or_else(|| 4.min(online)),
        PoolPhase::Decode => env_threads("SOFUU_DECODE_THREADS")
            .or_else(|| env_threads("SOFUU_NUM_THREADS"))
            .unwrap_or_else(|| 4.min(online)),
    }
}

/// Resize the pool for the phase if the current count differs. The pool
/// works on its own lock, disjoint from the scheduler/engine mutexes —
/// never taken while a forward is running, so no lock-order hazard.
fn size_pool_for_phase(phase: PoolPhase) {
    let want = pool_threads_for(phase);
    if crate::infer::quant::get_thread_count() != want {
        crate::infer::quant::set_thread_count(want);
    }
}

// ── Phase metrics (Parts 0/1 instrument) ────────────────────────
// On-device split of a decode token: forward (the weight sweep, inside
// the model), sampler, and the rest (state handling, channel sends,
// scheduler glue). Aggregate in atomics; a line is appended to
// $SOFUU_PHASE_METRICS every 256 tokens and at prefill boundaries.
// Zero-cost unless the env var is set.

static M_FWD_US: AtomicU64 = AtomicU64::new(0);
static M_SAMPLE_US: AtomicU64 = AtomicU64::new(0);
static M_ARM_US: AtomicU64 = AtomicU64::new(0);
static M_PREFILL_US: AtomicU64 = AtomicU64::new(0);
static M_PREFILLS: AtomicU64 = AtomicU64::new(0);
static M_TOKENS: AtomicU64 = AtomicU64::new(0);
static M_PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

fn metrics_file() -> &'static Option<String> {
    M_PATH.get_or_init(|| {
        std::env::var("SOFUU_PHASE_METRICS")
            .ok()
            .filter(|s| !s.trim().is_empty())
    })
}

fn metrics_add(fwd_us: u64, sample_us: u64, arm_us: u64, prefill_us: u64, prefills: u64, tokens: u64) {
    if metrics_file().is_none() {
        return;
    }
    M_FWD_US.fetch_add(fwd_us, Ordering::Relaxed);
    M_SAMPLE_US.fetch_add(sample_us, Ordering::Relaxed);
    M_ARM_US.fetch_add(arm_us, Ordering::Relaxed);
    M_PREFILL_US.fetch_add(prefill_us, Ordering::Relaxed);
    M_PREFILLS.fetch_add(prefills, Ordering::Relaxed);
    let total = M_TOKENS.fetch_add(tokens, Ordering::Relaxed) + tokens;
    // Snapshot on a 256-token cadence, plus every prefill boundary (its
    // per-call cost would otherwise wash out of a 256-token window).
    let crossed = (total - tokens) / 256 != total / 256;
    decode_metrics_emit(crossed || (prefill_us > 0 && prefills > 0));
}

/// Flush the aggregate line. Called internally on the 256-token cadence;
/// `force` is for prefill boundaries.
fn decode_metrics_emit(cond: bool) {
    if !cond || metrics_file().is_none() {
        return;
    }
    let fwd = M_FWD_US.swap(0, Ordering::Relaxed);
    let sample = M_SAMPLE_US.swap(0, Ordering::Relaxed);
    let arm = M_ARM_US.swap(0, Ordering::Relaxed);
    let pre = M_PREFILL_US.swap(0, Ordering::Relaxed);
    let prefills = M_PREFILLS.swap(0, Ordering::Relaxed);
    let tokens = M_TOKENS.load(Ordering::Relaxed);
    if tokens == 0 && fwd == 0 && pre == 0 {
        return; // nothing measured in this window
    }
    let per_tok = if tokens > 0 { arm / tokens } else { 0 };
    let line = format!(
        "window tokens={} fwd_tok_us={} sample_tok_us={} arm_tok_us={} glue_tok_us={} prefill_us={} prefills={}\n",
        tokens,
        if tokens > 0 { fwd / tokens } else { 0 },
        if tokens > 0 { sample / tokens } else { 0 },
        per_tok,
        per_tok.saturating_sub(fwd / tokens.max(1)).saturating_sub(sample / tokens.max(1)),
        pre,
        prefills,
    );
    if let Some(path) = metrics_file() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = write!(f, "{}", line);
        }
    }
}

/// Message emitted by an active generation stream to the caller/HTTP writer.
#[derive(Debug, Clone)]
pub enum StreamOutput {
    Token(String),
    Done(usize, usize), // (prompt_tokens, completion_tokens)
    Error(String),
}

/// Generation stream status.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamStatus {
    Queued,
    Prefilling,
    Decoding,
    Done,
    Aborted,
}

/// Request to submit a new generation stream.
pub struct StreamRequest {
    pub prompt: String,
    pub options: GenerateOptions,
    pub tx: std::sync::mpsc::Sender<StreamOutput>,
    pub aborted: Arc<AtomicBool>,
}

/// Per-sequence state inside the continuous batch scheduler.
pub struct ActiveStream {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    /// prompt + every sampled token, appended ONCE per token. The decode
    /// step used to rebuild this by cloning prompt_tokens and splicing
    /// generated_tokens on EVERY iteration — O(turn²) element copies by
    /// the end of a long reply — just to hand the sampler a history.
    pub full_history: Vec<u32>,
    pub cur_pos: usize,
    pub options: GenerateOptions,
    pub tx: std::sync::mpsc::Sender<StreamOutput>,
    pub aborted: Arc<AtomicBool>,
    pub sampler: Sampler,
    pub logits_buf: Vec<f32>,
    pub status: StreamStatus,
    pub block_tables: Vec<SequenceBlockTable>,
    pub full_generated: String,
    /// Wall time this stream has spent in the Prefilling arm, summed over its
    /// slices. One prompt can now span many scheduler iterations (chunked
    /// prefill), so the phase metric is accumulated here and reported once, at
    /// the slice that completes the prompt — not per slice.
    pub prefill_us: u64,
    /// Prompt tokens this stream got from the shared prefix cache on its first
    /// slice (0 = the whole prompt is being recomputed). Restored rows are real
    /// KV for [0, prefix_reuse), so the slice planner starts at them.
    pub prefix_reuse: usize,
}

/// Shared state of the continuous batch scheduler.
pub struct SchedulerState {
    pub pending: VecDeque<StreamRequest>,
    pub active: Vec<ActiveStream>,
    pub running: bool,
}

/// Continuous batch inference daemon and server.
pub struct ContinuousBatchServer {
    state: Arc<Mutex<SchedulerState>>,
    condvar: Arc<Condvar>,
    engine: Arc<Mutex<InferEngine>>,
    block_pools: Arc<Mutex<Vec<BlockPool>>>,
    next_id: AtomicU64,
    worker_handle: Option<std::thread::JoinHandle<()>>,
}

impl ContinuousBatchServer {
    /// Create and start a new continuous batching server for the given engine.
    pub fn new(engine: Arc<Mutex<InferEngine>>) -> Self {
        let (n_layers, n_kv_heads, head_dim) = {
            let eng = engine.lock().unwrap();
            (
                eng.model.config.n_layers,
                eng.model.config.n_kv_heads,
                eng.model.config.head_dim,
            )
        };

        // Initialize block pools per layer: 64 initial blocks (1024 tokens) up to 2048 blocks (32,768 tokens)
        let mut pools = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            pools.push(BlockPool::new(n_kv_heads, head_dim, 64, 2048));
        }

        let state = Arc::new(Mutex::new(SchedulerState {
            pending: VecDeque::new(),
            active: Vec::new(),
            running: true,
        }));
        let condvar = Arc::new(Condvar::new());
        let block_pools = Arc::new(Mutex::new(pools));

        let state_clone = state.clone();
        let condvar_clone = condvar.clone();
        let engine_clone = engine.clone();
        let pools_clone = block_pools.clone();

        let worker_handle = std::thread::spawn(move || {
            Self::worker_loop(state_clone, condvar_clone, engine_clone, pools_clone);
        });

        Self {
            state,
            condvar,
            engine,
            block_pools,
            next_id: AtomicU64::new(1),
            worker_handle: Some(worker_handle),
        }
    }

    /// Submit a new prompt for streaming generation. Non-blocking.
    pub fn submit(
        &self,
        prompt: &str,
        options: GenerateOptions,
        tx: std::sync::mpsc::Sender<StreamOutput>,
        aborted: Arc<AtomicBool>,
    ) {
        let req = StreamRequest {
            prompt: prompt.to_string(),
            options,
            tx,
            aborted,
        };
        {
            let mut state = self.state.lock().unwrap();
            state.pending.push_back(req);
        }
        self.condvar.notify_one();
    }

    /// Get current metrics for server diagnostics.
    pub fn metrics(&self) -> (usize, usize, usize) {
        let (pending, active) = {
            let state = self.state.lock().unwrap();
            (state.pending.len(), state.active.len())
        };
        let allocated_blocks = {
            let pools = self.block_pools.lock().unwrap();
            pools.iter().map(|p| p.allocated_blocks_count()).sum()
        };
        (pending, active, allocated_blocks)
    }

    fn worker_loop(
        state_arc: Arc<Mutex<SchedulerState>>,
        condvar: Arc<Condvar>,
        engine_arc: Arc<Mutex<InferEngine>>,
        pools_arc: Arc<Mutex<Vec<BlockPool>>>,
    ) {
        let mut next_seq_id = 1u64;

        loop {
            // 1. Wait when idle
            let (has_work, is_running) = {
                let mut state = state_arc.lock().unwrap();
                while state.running && state.pending.is_empty() && state.active.is_empty() {
                    state = condvar.wait(state).unwrap();
                }
                (!state.pending.is_empty() || !state.active.is_empty(), state.running)
            };

            if !is_running && !has_work {
                break;
            }

            // 2. Admit pending requests into active streams
            {
                let mut state = state_arc.lock().unwrap();
                // Phase snapshot BEFORE the work: admissions mean prefill;
                // any active Decoding stream means decode. Runs while the
                // scheduler state is held; the pool lock is disjoint.
                if state.active.iter().any(|st| st.status == StreamStatus::Decoding) {
                    size_pool_for_phase(PoolPhase::Decode);
                } else {
                    size_pool_for_phase(PoolPhase::Prefill);
                }
                while let Some(req) = state.pending.pop_front() {
                    if req.aborted.load(Ordering::Relaxed) {
                        let _ = req.tx.send(StreamOutput::Done(0, 0));
                        continue;
                    }

                    let (input_tokens, vocab_size, n_layers) = {
                        let eng = engine_arc.lock().unwrap();
                        let tokens = eng.tokenizer.encode(&req.prompt);
                        (tokens, eng.model.config.vocab_size, eng.model.config.n_layers)
                    };

                    if input_tokens.is_empty() {
                        let _ = req.tx.send(StreamOutput::Error("Prompt produced 0 tokens".to_string()));
                        continue;
                    }

                    let sampler = Sampler::new(SamplerConfig {
                        temperature: req.options.temperature,
                        top_p: req.options.top_p,
                        top_k: req.options.top_k,
                        repetition_penalty: req.options.repetition_penalty,
                    });

                    let mut block_tables = Vec::with_capacity(n_layers);
                    for _ in 0..n_layers {
                        block_tables.push(SequenceBlockTable::new());
                    }

                    let full_history = input_tokens.clone();
                    let active_stream = ActiveStream {
                        id: next_seq_id,
                        prompt_tokens: input_tokens,
                        generated_tokens: Vec::new(),
                        full_history,
                        cur_pos: 0,
                        options: req.options,
                        tx: req.tx,
                        aborted: req.aborted,
                        sampler,
                        logits_buf: vec![0.0f32; vocab_size],
                        status: StreamStatus::Prefilling,
                        block_tables,
                        full_generated: String::new(),
                        prefill_us: 0,
                        prefix_reuse: 0,
                    };
                    next_seq_id += 1;
                    state.active.push(active_stream);
                }
            }

            // 3. Execute iteration-level step across all active streams
            let mut progressed = false;
            let mut streams_to_retain = Vec::new();
            {
                let mut state = state_arc.lock().unwrap();
                let mut active_streams = std::mem::take(&mut state.active);

                // ADMISSION ORDER (chunked prefill): decoding streams keep the
                // front of the pass (their step is short and latency-critical),
                // prefilling streams follow shortest-remaining-first, so a short
                // request admitted behind a 3k-token prompt gets its slice — and
                // therefore its first token — in THIS iteration. Stable sort:
                // FIFO is preserved among equals, and no stream is dropped,
                // merged or skipped (pure reordering).
                active_streams.sort_by_key(|st| {
                    prefill_sched::order_key(
                        st.status == StreamStatus::Prefilling,
                        st.prompt_tokens.len().saturating_sub(st.cur_pos),
                    )
                });

                for mut stream in active_streams {
                    if stream.aborted.load(Ordering::Relaxed) {
                        let p_len = stream.prompt_tokens.len();
                        let g_len = stream.generated_tokens.len();
                        let _ = stream.tx.send(StreamOutput::Done(p_len, g_len));
                        // Free blocks back to pools
                        let mut pools = pools_arc.lock().unwrap();
                        for (tbl, pool) in stream.block_tables.iter_mut().zip(pools.iter_mut()) {
                            tbl.clear(pool);
                        }
                        continue;
                    }

                    match stream.status {
                        StreamStatus::Prefilling => {
                            // CHUNKED PREFILL: one slice per iteration, so a
                            // long prompt stops monopolising the scheduler.
                            // cur_pos doubles as "prompt tokens already in the
                            // KV cache" while Prefilling. Slice size comes from
                            // prefill_sched (SOFUU_PREFILL_LIMIT, 0 = one shot)
                            // and is a whole number of backend prefill groups,
                            // so the batched calls the model sees are the ones
                            // a one-shot prefill would have issued — bit-exact.
                            let pf_t0 = std::time::Instant::now();
                            let mut eng = engine_arc.lock().unwrap();
                            let p_len = stream.prompt_tokens.len();
                            let chunk = prefill_sched::chunk_tokens(&eng.model);

                            // SHARED-PREFIX KV REUSE, once, before the first
                            // slice. The restore leaves `reuse` prompt tokens
                            // really present in the KV cache, so cur_pos — which
                            // doubles as "prompt tokens already in KV" while
                            // Prefilling — starts there and the slice planner just
                            // sees a shorter remainder. Reuse is aligned to whole
                            // backend prefill groups (prefix_cache_align), so the
                            // group that computes the last-token projection,
                            // rows [p_len - p_len % G, p_len), is the SAME group a
                            // cold run would have used: identity survives slicing.
                            if stream.cur_pos == 0 && p_len > 1 {
                                let reuse = eng.prefix_cache_restore(&stream.prompt_tokens);
                                stream.prefix_reuse = reuse;
                                stream.cur_pos = reuse;
                            }

                            let start = stream.cur_pos;
                            match prefill_sched::step(
                                &mut eng.model,
                                &stream.prompt_tokens,
                                start,
                                chunk,
                                &mut stream.logits_buf,
                            ) {
                                SliceStep::Continue => {
                                    // Continue only when the slice fit whole.
                                    stream.cur_pos = start + chunk.min(p_len - start);
                                }
                                SliceStep::Complete => {
                                    stream.cur_pos = p_len;
                                    // SOFUU_DUMP_LOGITS receipt on the SERVING
                                    // path: the prefill projection of this
                                    // request, so a cold and a cache-warm run
                                    // can be compared bit for bit ({n} = stream
                                    // id). Emitted once, at the slice that
                                    // completes the prompt.
                                    crate::infer::dump_logits_receipt(
                                        "SOFUU_DUMP_LOGITS",
                                        "logits-dump",
                                        stream.id,
                                        &stream.logits_buf,
                                    );
                                    // Rows are final here and nothing has
                                    // compacted the cache, so this prompt's
                                    // prefix is safe to publish for the next
                                    // request. Counters describe the WHOLE
                                    // prompt: reuse is what the restore wrote
                                    // before the first slice, the rest is what
                                    // all slices computed.
                                    eng.prefix_cache_insert(&stream.prompt_tokens, stream.cur_pos);
                                    eng.last_prefill_tokens = p_len - stream.prefix_reuse;
                                    eng.last_prefix_reuse = stream.prefix_reuse;
                                    eng.last_prefill_duration =
                                        std::time::Duration::from_micros(
                                            stream.prefill_us
                                                + pf_t0.elapsed().as_micros() as u64,
                                        );
                                    stream.status = StreamStatus::Decoding;
                                }
                            }
                            drop(eng);
                            stream.prefill_us += pf_t0.elapsed().as_micros() as u64;
                            if stream.status == StreamStatus::Decoding {
                                // Prefill boundary: report the whole prompt's
                                // prefill time (all slices) once.
                                metrics_add(0, 0, 0, stream.prefill_us, 1, 0);
                                decode_metrics_emit(false);
                            }
                            progressed = true;
                            streams_to_retain.push(stream);
                        }
                        StreamStatus::Decoding => {
                            let arm_t0 = std::time::Instant::now();
                            let s_t0 = std::time::Instant::now();
                            let next_token =
                                stream.sampler.sample(&mut stream.logits_buf, &stream.full_history);
                            let sample_us = s_t0.elapsed().as_micros() as u64;

                            let (is_eos, raw_tok_str, piece) = {
                                let eng = engine_arc.lock().unwrap();
                                let eos = eng.tokenizer.eos_token_id == Some(next_token);
                                let raw = eng.tokenizer.decode_token(next_token).map(|s| s.to_string());
                                let piece = eng.tokenizer.decode_piece(next_token);
                                (eos, raw, piece)
                            };

                            let is_stop = is_eos
                                || raw_tok_str.as_deref() == Some("<|im_end|>")
                                || raw_tok_str.as_deref() == Some("<|endoftext|>")
                                || raw_tok_str.as_deref() == Some("<|eot_id|>")
                                || stream.options.stop_tokens.iter().any(|s| raw_tok_str.as_deref() == Some(s.as_str()))
                                || stream.options.stop_tokens.iter().any(|s| piece.contains(s.as_str()));

                            if is_stop || stream.generated_tokens.len() >= stream.options.max_tokens {
                                stream.status = StreamStatus::Done;
                                let p_len = stream.prompt_tokens.len();
                                let g_len = stream.generated_tokens.len();
                                let _ = stream.tx.send(StreamOutput::Done(p_len, g_len));
                                // Free blocks back to pools
                                let mut pools = pools_arc.lock().unwrap();
                                for (tbl, pool) in stream.block_tables.iter_mut().zip(pools.iter_mut()) {
                                    tbl.clear(pool);
                                }
                                progressed = true;
                                continue;
                            }

                            // Emit token chunk to HTTP listener
                            stream.full_generated.push_str(&piece);
                            stream.generated_tokens.push(next_token);
                            stream.full_history.push(next_token);
                            let _ = stream.tx.send(StreamOutput::Token(piece));
                            progressed = true;

                            // Advance decode forward
                            let f_t0 = std::time::Instant::now();
                            {
                                let mut eng = engine_arc.lock().unwrap();
                                let logits = eng.model.forward(next_token, stream.cur_pos);
                                stream.logits_buf.copy_from_slice(logits);
                                // SOFUU_DUMP_LOGITS_DEC receipt: the first decode
                                // step's full-vocabulary logits of this stream
                                // (the engine's decode receipt, same container).
                                if stream.generated_tokens.len() == 1 {
                                    crate::infer::dump_logits_receipt(
                                        "SOFUU_DUMP_LOGITS_DEC",
                                        "logits-dump-dec",
                                        stream.id,
                                        &stream.logits_buf,
                                    );
                                }
                            }
                            let fwd_us = f_t0.elapsed().as_micros() as u64;
                            stream.cur_pos += 1;
                            metrics_add(fwd_us, sample_us, arm_t0.elapsed().as_micros() as u64, 0, 0, 1);
                            decode_metrics_emit(false);
                            streams_to_retain.push(stream);
                        }
                        _ => {}
                    }
                }
                state.active = streams_to_retain;
            }

            // Yield only when the iteration did no work. While streams are
            // decoding, every pass consumes real compute (sample + forward),
            // so sleeping here would just add a 1 kHz wakeup on one core for
            // the entire generation — measurable heat on a phone for nothing.
            // An empty-to-idle transition parks on the condvar at the top.
            if !progressed {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

impl Drop for ContinuousBatchServer {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.running = false;
        }
        self.condvar.notify_all();
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_request_and_output() {
        let (tx, rx) = std::sync::mpsc::channel();
        let aborted = Arc::new(AtomicBool::new(false));
        let req = StreamRequest {
            prompt: "Hello world".to_string(),
            options: GenerateOptions::default(),
            tx,
            aborted,
        };
        assert_eq!(req.prompt, "Hello world");
        let _ = req.tx.send(StreamOutput::Token("Hi".to_string()));
        if let Ok(StreamOutput::Token(t)) = rx.try_recv() {
            assert_eq!(t, "Hi");
        } else {
            panic!("Expected Token");
        }
    }
}
