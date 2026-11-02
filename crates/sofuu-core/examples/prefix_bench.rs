// crates/sofuu-core/examples/prefix_bench.rs — prefix-cache serving harness.
//
// Drives N requests that share one long system prompt and differ in the user
// turn (the agent/RAG shape), through either the continuous-batching server
// (--mode serve, the path `sofuu serve` / sofuu.ai.stream uses) or the engine
// (--mode engine, the path the CLI uses), and prints one JSON line per request
// with TTFT, the prefill it actually computed, and how much of the prompt the
// prefix cache let it skip.
//
// One process = ONE arm: --arm cold sets SOFUU_PREFIX_CACHE=0 before the model
// loads (the reproducible path), --arm warm leaves the cache on. The driver runs
// the arms interleaved (ABBA) so a machine that drifts cannot favour one.
//
//   cargo run --profile bench -p sofuu-core --example prefix_bench -- \
//     --model ~/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf --arm warm \
//     --mode serve --requests 8 --max-tokens 8 --prompt-file /tmp/gen_prompt.txt
//
// --dump-dir writes, per request, the KV rows of positions [0, prompt_tokens)
// (SFKV01) and the generated text. Cold and warm runs that agree byte for byte
// prove the shared KV is bit-identical, not merely close.

use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use sofuu_core::infer::continuous::{ContinuousBatchServer, StreamOutput};
use sofuu_core::infer::{GenerateOptions, InferEngine};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn load_avg() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Chat framing. Only the SHAPE matters for a prefix-reuse benchmark, but a
/// model that was trained with one template produces sane turns with it, so a
/// greedy decode actually stops instead of rambling.
fn render(template: &str, system: &str, user: &str) -> String {
    match template {
        "llama3" => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        _ => format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        ),
    }
}

const USER_TURNS: [&str; 8] = [
    "In one sentence: what is this project?",
    "List three runtime features, comma separated.",
    "What binary size does it claim?",
    "Name the inference formats it supports.",
    "Which platforms are listed?",
    "Summarise the licence in five words.",
    "What is the version string?",
    "Give one reason a team would pick it.",
];

fn opts(max_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..Default::default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = arg(&args, "--model").expect("--model <gguf>");
    let arm = arg(&args, "--arm").unwrap_or_else(|| "warm".to_string());
    let mode = arg(&args, "--mode").unwrap_or_else(|| "serve".to_string());
    let requests: usize = arg(&args, "--requests").and_then(|v| v.parse().ok()).unwrap_or(8);
    let max_tokens: usize = arg(&args, "--max-tokens").and_then(|v| v.parse().ok()).unwrap_or(8);
    let prompt_file = arg(&args, "--prompt-file").unwrap_or_else(|| "/tmp/gen_prompt.txt".to_string());
    let dump_dir = arg(&args, "--dump-dir");
    let template = arg(&args, "--template").unwrap_or_else(|| {
        if model_path.to_lowercase().contains("llama") {
            "llama3".to_string()
        } else {
            "chatml".to_string()
        }
    });

    // The arm is fixed BEFORE the model loads: the cache config is read once,
    // per engine. Case-insensitive so a driver that labels arms "Cold"/"Warm"
    // cannot silently run a cold request against a warm cache.
    let cold_arm = arm.to_lowercase().starts_with("cold");
    if cold_arm {
        std::env::set_var("SOFUU_PREFIX_CACHE", "0");
    } else {
        std::env::remove_var("SOFUU_PREFIX_CACHE");
    }
    if let Some(d) = &dump_dir {
        let _ = std::fs::create_dir_all(d);
    }

    let system = std::fs::read_to_string(&prompt_file)
        .unwrap_or_else(|e| panic!("read {prompt_file}: {e}"));
    // --distinct costs the cache's PREFIX in the system turn itself, so every
    // request shares nothing: that is the single-user-chat / no-reuse case, and
    // it measures what the cache costs when it can never pay (snapshot work on
    // the cold path, zero hits).
    let distinct = args.iter().any(|a| a == "--distinct");
    let prompts: Vec<String> = (0..requests)
        .map(|i| {
            let sys = if distinct {
                format!("[session {i}]\n{system}")
            } else {
                system.clone()
            };
            render(&template, &sys, USER_TURNS[i % USER_TURNS.len()])
        })
        .collect();

    println!(
        "{{\"kind\":\"start\",\"arm\":\"{}\",\"mode\":\"{}\",\"model\":\"{}\",\"load\":\"{}\",\"requests\":{},\"max_tokens\":{},\"template\":\"{}\"}}",
        arm, mode, json_escape(&model_path), json_escape(&load_avg()), requests, max_tokens, template
    );

    let engine = InferEngine::load(&model_path).expect("load model");
    let prompt_tokens = engine.tokenizer.encode(&prompts[0]).len();
    println!(
        "{{\"kind\":\"setup\",\"arm\":\"{}\",\"prompt_tokens\":{},\"system_tokens\":{},\"prefix_enabled\":{}}}",
        arm,
        prompt_tokens,
        engine.tokenizer.encode(&system).len(),
        if engine.prefix_cache.enabled() { 1 } else { 0 }
    );

    let shared = Arc::new(Mutex::new(engine));
    let server = if mode == "serve" {
        Some(ContinuousBatchServer::new(shared.clone()))
    } else {
        None
    };

    let mut texts = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let t0 = Instant::now();
        let mut text = String::new();
        let ttft_ms;
        let mut first_ms = f64::NAN;
        match &server {
            Some(srv) => {
                let (tx, rx) = std::sync::mpsc::channel();
                let aborted = Arc::new(AtomicBool::new(false));
                srv.submit(prompt, opts(max_tokens), tx, aborted);
                loop {
                    match rx.recv() {
                        Ok(StreamOutput::Token(t)) => {
                            if first_ms.is_nan() {
                                first_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            }
                            text.push_str(&t);
                        }
                        Ok(StreamOutput::Done(_, _)) => break,
                        Ok(StreamOutput::Error(e)) => panic!("serving error: {e}"),
                        Err(e) => panic!("stream channel closed: {e}"),
                    }
                }
            }
            None => {
                let mut eng = shared.lock().unwrap();
                let mut cb = |t: &str| {
                    if first_ms.is_nan() {
                        first_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    }
                    text.push_str(t);
                    true
                };
                let res = eng.stream(prompt, opts(max_tokens), &mut cb);
                if let Err(e) = res {
                    panic!("engine error: {e}");
                }
            }
        }
        ttft_ms = first_ms;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Engine-side facts for THIS request: what the prefill computed, how
        // long it took, how many prompt tokens the cache let it skip.
        let (prefill_tokens, prefill_ms, reuse_tokens, restore_ms, cache_blocks, cache_bytes, cache_hits, ptoks) = {
            let eng = shared.lock().unwrap();
            let prefill_ms = eng.last_prefill_duration.as_secs_f64() * 1000.0;
            let st = eng.prefix_cache_stats();
            (
                eng.last_prefill_tokens,
                prefill_ms,
                eng.last_prefix_reuse,
                eng.last_prefix_restore.as_secs_f64() * 1000.0,
                st.blocks,
                st.bytes,
                st.hits,
                eng.tokenizer.encode(prompt).len(),
            )
        };
        let eff_tps = if prefill_ms > 0.0 { ptoks as f64 / (prefill_ms / 1000.0) } else { 0.0 };
        let real_tps = if prefill_ms > 0.0 {
            prefill_tokens as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        };
        println!(
            "{{\"kind\":\"req\",\"arm\":\"{}\",\"mode\":\"{}\",\"req\":{},\"prompt_tokens\":{},\"reuse_tokens\":{},\"prefill_tokens\":{},\"restore_ms\":{:.3},\"prefill_ms\":{:.3},\"ttft_ms\":{:.3},\"total_ms\":{:.3},\"eff_prefill_tps\":{:.1},\"real_prefill_tps\":{:.1},\"gen_tokens\":{},\"cache_blocks\":{},\"cache_bytes\":{},\"cache_hits\":{},\"text_sha\":\"{}\"}}",
            arm,
            mode,
            i + 1,
            ptoks,
            reuse_tokens,
            prefill_tokens,
            restore_ms,
            prefill_ms,
            ttft_ms,
            total_ms,
            eff_tps,
            real_tps,
            text.split_whitespace().count(),
            cache_blocks,
            cache_bytes,
            cache_hits,
            fnv_hex(&text)
        );
        std::io::stdout().flush().ok();

        if let Some(dir) = &dump_dir {
            let eng = shared.lock().unwrap();
            let kv_path = format!("{dir}/{arm}-kv-req{:02}.bin", i + 1);
            match eng.dump_kv_prefix(ptoks, &kv_path) {
                Ok(()) => println!(
                    "{{\"kind\":\"kvdump\",\"arm\":\"{}\",\"req\":{},\"path\":\"{}\",\"tokens\":{}}}",
                    arm,
                    i + 1,
                    json_escape(&kv_path),
                    ptoks
                ),
                Err(e) => println!("{{\"kind\":\"kvdump-error\",\"err\":\"{}\"}}", json_escape(&e)),
            }
            let txt_path = format!("{dir}/{arm}-text-req{:02}.txt", i + 1);
            let _ = std::fs::write(&txt_path, &text);
        }
        texts.push(text);
    }

    {
        let eng = shared.lock().unwrap();
        let st = eng.prefix_cache_stats();
        println!(
            "{{\"kind\":\"summary\",\"arm\":\"{}\",\"mode\":\"{}\",\"cache_hits\":{},\"cache_misses\":{},\"hit_tokens\":{},\"reused_tokens\":{},\"blocks\":{},\"bytes\":{},\"blocks_created\":{},\"blocks_evicted\":{},\"load_end\":\"{}\"}}",
            arm,
            mode,
            st.hits,
            st.misses,
            st.hit_tokens,
            st.reused_tokens,
            st.blocks,
            st.bytes,
            st.blocks_created,
            st.blocks_evicted,
            json_escape(&load_avg())
        );
    }

    drop(server);
    let _ = texts;
}

/// Stable 64-bit FNV-1a of the generated text: the greedy-output receipt that
/// does not depend on how the harness prints it.
fn fnv_hex(s: &str) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

