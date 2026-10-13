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
use super::sampler::{Sampler, SamplerConfig};
use super::{GenerateOptions, InferEngine};

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
    pub cur_pos: usize,
    pub options: GenerateOptions,
    pub tx: std::sync::mpsc::Sender<StreamOutput>,
    pub aborted: Arc<AtomicBool>,
    pub sampler: Sampler,
    pub logits_buf: Vec<f32>,
    pub status: StreamStatus,
    pub block_tables: Vec<SequenceBlockTable>,
    pub full_generated: String,
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

                    let active_stream = ActiveStream {
                        id: next_seq_id,
                        prompt_tokens: input_tokens,
                        generated_tokens: Vec::new(),
                        cur_pos: 0,
                        options: req.options,
                        tx: req.tx,
                        aborted: req.aborted,
                        sampler,
                        logits_buf: vec![0.0f32; vocab_size],
                        status: StreamStatus::Prefilling,
                        block_tables,
                        full_generated: String::new(),
                    };
                    next_seq_id += 1;
                    state.active.push(active_stream);
                }
            }

            // 3. Execute iteration-level step across all active streams
            let mut streams_to_retain = Vec::new();
            {
                let mut state = state_arc.lock().unwrap();
                let active_streams = std::mem::take(&mut state.active);

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
                            let mut eng = engine_arc.lock().unwrap();
                            let p_len = stream.prompt_tokens.len();

                            if p_len == 1 {
                                let tok = stream.prompt_tokens[0];
                                let logits = eng.model.forward(tok, 0);
                                stream.logits_buf.copy_from_slice(logits);
                                stream.cur_pos = 1;
                            } else {
                                // Unified fast prefill
                                let logits = eng.model.forward_prefill_all_with_logits(&stream.prompt_tokens, 0);
                                stream.logits_buf.copy_from_slice(logits);
                                stream.cur_pos = p_len;
                            }
                            stream.status = StreamStatus::Decoding;
                            streams_to_retain.push(stream);
                        }
                        StreamStatus::Decoding => {
                            let next_token = {
                                let mut history = stream.prompt_tokens.clone();
                                history.extend_from_slice(&stream.generated_tokens);
                                stream.sampler.sample(&mut stream.logits_buf, &history)
                            };

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
                                continue;
                            }

                            // Emit token chunk to HTTP listener
                            stream.full_generated.push_str(&piece);
                            stream.generated_tokens.push(next_token);
                            let _ = stream.tx.send(StreamOutput::Token(piece));

                            // Advance decode forward
                            {
                                let mut eng = engine_arc.lock().unwrap();
                                let logits = eng.model.forward(next_token, stream.cur_pos);
                                stream.logits_buf.copy_from_slice(logits);
                            }
                            stream.cur_pos += 1;
                            streams_to_retain.push(stream);
                        }
                        _ => {}
                    }
                }
                state.active = streams_to_retain;
            }

            // Yield briefly to prevent CPU starvation on single-threaded event loop
            std::thread::sleep(Duration::from_millis(1));
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
