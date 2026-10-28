//! kv_slope — the before/after instrument for the quantized-KV kernel work.
//!
//! It answers one question with numbers: **does the quantized KV cache pay for
//! itself at long context?** It measures decode cost as a function of context
//! length for several KV-cache arms on the same model, fits
//! `ms/token = a + b*ctx` per arm, and reports the slope ratio
//! `b_quantized / b_f32`. That ratio is the ship gate: the quantized cache
//! only "pays" once its marginal cost per context token is no greater than
//! f32's (`ratio <= 1.0`). Anything above 1.0 is a slope penalty that gets
//! *worse* with context — no context length turns it into a win.
//!
//! ## Usage
//!
//! ```text
//! # 1. build the engine binary (the thing users actually run)
//! cargo build --release --bin sofuu
//! # 2. build this instrument
//! cargo build --release --example kv_slope
//! # 3. run it (model path is the only required argument)
//! ./target/release/examples/kv_slope ~/Downloads/Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf
//! ```
//!
//! Flags (defaults reproduce the gold-standard matrix from
//! `~/kv-evidence/REPORT-kvlong.md`):
//!
//! ```text
//!   <model.gguf>                  model to measure (positional, required)
//!   --ctx-points 493,1497,2993,3994    prompt token counts to measure at
//!   --arms f32,8,4                KV-cache arms; "f32" is the reference and is
//!                                 required; 8 / 4 map to --kv-dtype 8 / 4
//!   --tokens N                    [32] decode tokens per run (N-1 trace rows;
//!                                 the cold first forward is excluded)
//!   --rounds R                    [3] ABBA rounds per point -> 2R runs per arm
//!   --prompt-file PATH            [/tmp/gen_prompt.txt] base prose to concatenate
//!   --bin PATH                    [./target/release/sofuu, ./target/debug/sofuu,
//!                                 then $SOFUU_BIN, then $PATH]
//!   --ctx N                       pass --ctx N through to the engine (default:
//!                                 engine default window, 4096 for these models)
//!   --warmup-skip N               [1] leading trace rows dropped per run
//!   --load-max F                  [6.0] load1 above this marks a point UNRELIABLE
//!   --spread-max F                [1.6] slowest/fastest run inside one arm at
//!                                 one point above this marks the point
//!                                 UNRELIABLE (catches GPU contention and swap,
//!                                 which loadavg does not see)
//!   --json PATH                   also write every run record as JSONL
//!   --keep-traces                 keep the per-run --trace files
//!   --run-timeout SECONDS         [900] hard kill per engine run
//!   --dry-run                     build the prompts and print their token counts
//!   --no-gpu-flags                do not set SOFUU_GPU_PREFILL/SOFUU_GEMM_MMA8
//! ```
//!
//! Exit codes: 0 = matrix complete and a verdict printed; 1 = usage/runtime
//! error; 2 = an arm was INERT (see below) so no table was printed; 3 = matrix
//! complete but at least one point was UNRELIABLE, so no overall verdict.
//!
//! ## How the measurement works (and why it is fair)
//!
//! * **Prompts are exactly N tokens.** The base text is concatenated as whole
//!   copies plus a word-boundary prefix until the model's own tokenizer says the
//!   prompt is N tokens. The engine's `── prefill: N tk` line is compared
//!   against that and any mismatch is printed.
//! * **Arms are interleaved ABBA.** Per round the order is `A B C C B A`, so
//!   slow thermal drift hits both ends of every arm equally. Per point that is
//!   `2R` runs per arm (6 at the default R=3, the same repetition count as
//!   REPORT-kvlong.md).
//! * **Decode = median of the per-token `forward_ns` from `--trace`**, with
//!   the cold first forward dropped. The CLI's own `decode: ... tk/s` line is
//!   NOT used: it includes per-token work outside the timed forward.
//!   `ms/token min` is the best single steady token across the arm's runs
//!   (best case); `med` is the typical case. Both are reported and they
//!   should agree.
//! * **Prefill** is the engine's own `── prefill: N tk in T` figure, median
//!   of the arm's runs.
//! * **Serial decode** (`--no-lookahead`) at temp 0, so all arms produce
//!   comparable token streams and speculation cannot add noise.
//! * **Load is quoted** (`sysctl -n vm.loadavg`) before and after every
//!   point, and around every run. A point whose peak load1 exceeds
//!   `--load-max` is marked UNRELIABLE, and then no overall verdict is
//!   printed.
//! * **Dispersion is quoted too.** loadavg cannot see a second GPU client or
//!   swap thrash: a machine can sit at load 2 while another process serialises
//!   on the GPU and a 9 ms forward becomes 190 ms. So the slowest/fastest run
//!   inside each arm is reported per point and a spread above `--spread-max`
//!   marks the point UNRELIABLE. This is how you tell "the arm is slow" from
//!   "the machine hiccuped".
//!
//! ## Engagement: an arm that reads identical to f32 is not a fast run
//!
//! kv4 is armed only when the context WINDOW exceeds the 192-row f32 residual
//! ring, the GPU decode path is enabled, the head geometry fits the 32-token
//! groups (`head_dim % 32 == 0`, `head_dim <= 128`), and the model is not
//! resident. If any of that is false the engine keeps the f32 cache, prints
//! `[kv4] ... does not apply here ...`, and the "8-bit" arm is silently the
//! f32 arm wearing a label: it will time exactly like f32 and look like a tie.
//! So every quantized arm, at every point, must show:
//!
//! 1. the binary's own `[kv4] quantized KV armed: ...` receipt on stderr, and
//! 2. a numeric difference from f32 — either `|dlogprob| >= 1e-6` on a step
//!    where both arms emitted the same token, or a diverging token.
//!
//! If an arm shows neither, this tool prints a FATAL naming the failed check and
//! exits 2 without printing a timing table, because that table would be f32
//! timings under a quantized label.
//!
//! ## Reading the table
//!
//! ```text
//! | prompt tok | arm | decode tk/s med | decode tk/s best | ms/token med | ms/token min | run spread | prefill tk/s | prefill s |
//! |        493 | f32 |           111.6 |            118.1 |         8.96 |         8.25 |      1.02x |         1616 |      0.30 |
//! ```
//!
//! * `prompt tok` is the engine-reported prompt length (includes BOS).
//! * `decode tk/s med` = 1000 / (median over the arm's runs of each run's
//!   median steady `forward_ns`). This is the column the ship decision uses.
//! * `decode tk/s best` = 1000 / (best single steady token across runs) — the
//!   column the gold-standard report calls `decode tk/s min`.
//! * `run spread` = slowest run / fastest run of that arm at that point.
//!   1.00x-1.3x is a settled machine; above `--spread-max` the point is
//!   UNRELIABLE no matter what loadavg said, so re-run that point alone
//!   (`--ctx-points 3994`) when the machine is idle.
//! * `prefill tk/s` is where a quantized cache loses most: the packed blocks
//!   must be written and then dequantized per attended token.
//!
//! Then:
//!
//! * **Deltas vs f32** — sign and size of each arm's loss at each point. The
//!   *sign* matters more than the magnitude: a deltas table that is negative
//!   everywhere means the arm never wins.
//! * **Slope fit** — `a` is the context-independent cost per token (weights,
//!   sampling); `b` is the attention cost per context token. Arms that differ
//!   only in attention should have equal `a` and different `b` — that is the
//!   check that the comparison is apples-to-apples.
//! * **Slope ratio** — the gate. `b_8bit / b_f32 = 1.82x` means each extra
//!   context token costs 82% more in the 8-bit arm, so as ctx grows the
//!   quantized arm tends to 82% slower per token and there is no crossover.
//!   `<= 1.0` means the kernel rewrite finally pays: the quantized arm's cost
//!   per context token is no worse than f32's.
//!
//! ## Known-good baseline (the reproduction target)
//!
//! Qwen3-0.6B Q4_K_M, default 4096 window, temp 0, 32 decode tokens, 3 ABBA
//! rounds, `SOFUU_GPU_PREFILL=1 SOFUU_GEMM_MMA8=1`:
//!
//! ```text
//! prompt tok   f32 tk/s med   8-bit tk/s med   4-bit tk/s med
//!        493          111.6           100.1            98.3
//!       1497           86.3            69.6            69.8
//!       2993           66.2            46.9            47.5
//!       3994           56.3            38.8            38.6
//! slope b (us per ctx token): f32 2.49, 8-bit 4.53 (1.82x), 4-bit 4.49 (1.80x)
//! ```
//!
//! Landing within ~10% means the instrument agrees with the gold-standard run;
//! the slope ratio and the sign of every delta are what must match.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;
use sofuu_core::infer::gguf::{GgufFile, GgufValue};
use sofuu_core::infer::tokenizer::BpeTokenizer;

const DEFAULT_PROMPT_FILE: &str = "/tmp/gen_prompt.txt";
const DEFAULT_LOAD_MAX: f64 = 6.0;
/// max/min of an arm's per-run ms/token inside one point; above this the point is
/// contended by something loadavg cannot see (another GPU client, swap thrash).
const DEFAULT_SPREAD_MAX: f64 = 1.6;
/// The trace prints logprobs to 6 decimal places, so this is the smallest
/// difference that can be observed at all.
const DLP_TOL: f64 = 1e-6;

// ------------------------------------------------------------------
// args
// ------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Arm {
    /// user-facing key: "f32", "8", "4"
    key: String,
    /// table label: "f32", "8-bit", "4-bit"
    label: String,
    /// value for --kv-dtype; None = engine default (f32)
    kv_arg: Option<String>,
}

struct Options {
    model: PathBuf,
    ctx_points: Vec<usize>,
    arms: Vec<Arm>,
    tokens: usize,
    rounds: usize,
    prompt_file: PathBuf,
    bin: Option<PathBuf>,
    ctx: Option<usize>,
    warmup_skip: usize,
    load_max: f64,
    spread_max: f64,
    json: Option<PathBuf>,
    gpu_flags: bool,
    keep_traces: bool,
    dry_run: bool,
    run_timeout_s: u64,
}

fn usage() -> String {
    let mut s = String::new();
    s.push_str("kv_slope <model.gguf> [--ctx-points 493,1497,2993,3994] [--arms f32,8,4]\n");
    s.push_str("         [--tokens 32] [--rounds 3] [--prompt-file /tmp/gen_prompt.txt]\n");
    s.push_str("         [--bin target/release/sofuu] [--ctx N] [--warmup-skip 1]\n");
    s.push_str("         [--load-max 6.0] [--spread-max 1.6] [--json out.jsonl]\n");
    s.push_str("         [--keep-traces] [--dry-run]\n");
    s.push_str("         [--run-timeout 900] [--no-gpu-flags]\n");
    s
}

fn arg_value(argv: &[String], i: usize, what: &str) -> Result<String, String> {
    argv.get(i + 1)
        .cloned()
        .ok_or_else(|| format!("{} needs a value", what))
}

fn parse_args(argv: &[String]) -> Result<Option<Options>, String> {
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        print!("{}", usage());
        return Ok(None);
    }
    let mut model: Option<PathBuf> = None;
    let mut opts = Options {
        model: PathBuf::new(),
        ctx_points: vec![493, 1497, 2993, 3994],
        arms: Vec::new(),
        tokens: 32,
        rounds: 3,
        prompt_file: PathBuf::from(DEFAULT_PROMPT_FILE),
        bin: None,
        ctx: None,
        warmup_skip: 1,
        load_max: DEFAULT_LOAD_MAX,
        spread_max: DEFAULT_SPREAD_MAX,
        json: None,
        gpu_flags: true,
        keep_traces: false,
        dry_run: false,
        run_timeout_s: 900,
    };
    let mut arm_keys = String::from("f32,8,4");
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        match a {
            "--ctx-points" => {
                let v = arg_value(argv, i, a)?;
                i += 2;
                let mut pts = Vec::new();
                for p in v.split(',') {
                    let n: usize = p
                        .trim()
                        .parse()
                        .map_err(|_| format!("--ctx-points: bad number '{}'", p))?;
                    if n < 8 {
                        return Err("--ctx-points: every point needs at least 8 tokens".into());
                    }
                    pts.push(n);
                }
                pts.sort_unstable();
                pts.dedup();
                opts.ctx_points = pts;
            }
            "--arms" => {
                arm_keys = arg_value(argv, i, a)?;
                i += 2;
            }
            "--tokens" => {
                opts.tokens = arg_value(argv, i, a)?
                    .parse()
                    .map_err(|_| "--tokens: expected a number".to_string())?;
                i += 2;
            }
            "--rounds" => {
                opts.rounds = arg_value(argv, i, a)?
                    .parse()
                    .map_err(|_| "--rounds: expected a number".to_string())?;
                i += 2;
            }
            "--prompt-file" => {
                opts.prompt_file = PathBuf::from(arg_value(argv, i, a)?);
                i += 2;
            }
            "--bin" => {
                opts.bin = Some(PathBuf::from(arg_value(argv, i, a)?));
                i += 2;
            }
            "--ctx" => {
                let n: usize = arg_value(argv, i, a)?
                    .parse()
                    .map_err(|_| "--ctx: expected a number".to_string())?;
                opts.ctx = Some(n);
                i += 2;
            }
            "--warmup-skip" => {
                opts.warmup_skip = arg_value(argv, i, a)?
                    .parse()
                    .map_err(|_| "--warmup-skip: expected a number".to_string())?;
                i += 2;
            }
            "--load-max" => {
                opts.load_max = arg_value(argv, i, a)?
                    .parse()
                    .map_err(|_| "--load-max: expected a number".to_string())?;
                i += 2;
            }
            "--spread-max" => {
                opts.spread_max = arg_value(argv, i, a)?
                    .parse()
                    .map_err(|_| "--spread-max: expected a number".to_string())?;
                i += 2;
            }
            "--json" => {
                opts.json = Some(PathBuf::from(arg_value(argv, i, a)?));
                i += 2;
            }
            "--run-timeout" => {
                opts.run_timeout_s = arg_value(argv, i, a)?
                    .parse()
                    .map_err(|_| "--run-timeout: expected seconds".to_string())?;
                i += 2;
            }
            "--keep-traces" => {
                opts.keep_traces = true;
                i += 1;
            }
            "--dry-run" => {
                opts.dry_run = true;
                i += 1;
            }
            "--no-gpu-flags" => {
                opts.gpu_flags = false;
                i += 1;
            }
            _ if a.starts_with('-') => {
                return Err(format!("unknown flag '{}' (try --help)", a));
            }
            _ => {
                if model.is_some() {
                    return Err(format!("unexpected argument '{}'", a));
                }
                model = Some(PathBuf::from(a));
                i += 1;
            }
        }
    }
    let mut head = String::from("a model path is required\n\n");
    head.push_str(&usage());
    opts.model = model.ok_or(head)?;
    if !opts.model.exists() {
        return Err(format!("model not found: {}", opts.model.display()));
    }
    if opts.tokens < 3 {
        return Err("--tokens must be >= 3 (need >= 2 steady steps)".into());
    }
    if opts.rounds == 0 {
        return Err("--rounds must be >= 1".into());
    }
    if opts.warmup_skip >= opts.tokens {
        return Err("--warmup-skip must be < --tokens".into());
    }

    let mut arms: Vec<Arm> = Vec::new();
    for raw in arm_keys.split(',') {
        let k = raw.trim().to_ascii_lowercase();
        let arm = match k.as_str() {
            "f32" | "fp32" => Arm {
                key: "f32".into(),
                label: "f32".into(),
                kv_arg: None,
            },
            "8" | "8bit" | "q8" => Arm {
                key: "8".into(),
                label: "8-bit".into(),
                kv_arg: Some("8".into()),
            },
            "4" | "4bit" | "q4" => Arm {
                key: "4".into(),
                label: "4-bit".into(),
                kv_arg: Some("4".into()),
            },
            "" => continue,
            other => return Err(format!("--arms: '{}' is not an arm (use f32, 8, 4)", other)),
        };
        if !arms.iter().any(|x| x.key == arm.key) {
            arms.push(arm);
        }
    }
    if !arms.iter().any(|a| a.key == "f32") {
        return Err("--arms must include f32: it is the reference arm every delta and the slope ratio are measured against".into());
    }
    if arms.len() < 2 {
        return Err("--arms needs at least one quantized arm besides f32".into());
    }
    opts.arms = arms;
    Ok(Some(opts))
}

// ------------------------------------------------------------------
// prompt construction
// ------------------------------------------------------------------

fn count_tokens(tok: &BpeTokenizer, text: &str) -> usize {
    tok.encode(text).len()
}

/// Build a prompt that is `target` tokens as the model's own tokenizer counts
/// them (BOS included), by concatenating whole copies of `base` and then a
/// word-boundary prefix of it.
fn build_prompt(base: &str, target: usize, tok: &BpeTokenizer) -> Result<(String, usize), String> {
    let base = base.trim_end();
    let t1 = count_tokens(tok, base);
    if t1 == 0 {
        return Err("prompt file is empty".into());
    }
    let words: Vec<&str> = base.split_whitespace().collect();
    if words.is_empty() {
        return Err("prompt file has no words".into());
    }
    // target is shorter than one whole copy: trim at a word boundary
    if target <= t1 {
        let head_only = |w: usize| -> String { words[..w].join(" ") };
        let (mut lo, mut hi) = (0usize, words.len());
        let mut best = 0usize;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            if count_tokens(tok, &head_only(mid)) <= target {
                best = mid;
                lo = mid + 1;
            } else if mid == 0 {
                break;
            } else {
                hi = mid - 1;
            }
        }
        let mut pick = (head_only(best), count_tokens(tok, &head_only(best)));
        for w in [best + 1, best.saturating_sub(1)] {
            if w <= words.len() {
                let c = count_tokens(tok, &head_only(w));
                if (c as i64 - target as i64).abs() < (pick.1 as i64 - target as i64).abs() {
                    pick = (head_only(w), c);
                }
            }
        }
        return Ok(pick);
    }
    let append = |head: &str, w: usize| -> String {
        if w == 0 {
            head.to_string()
        } else {
            format!("{}\n\n{}", head, words[..w].join(" "))
        }
    };
    // largest word count whose encoding stays <= target
    let best_prefix = |head: &str| -> usize {
        let (mut lo, mut hi) = (0usize, words.len());
        let mut best = 0usize;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let c = count_tokens(tok, &append(head, mid));
            if c <= target {
                best = mid;
                lo = mid + 1;
            } else if mid == 0 {
                break;
            } else {
                hi = mid - 1;
            }
        }
        best
    };

    // candidate heads: k whole copies bracketing the target
    let k0 = (target / t1).max(1);
    let mut candidates: Vec<String> = Vec::new();
    for k in [k0.saturating_sub(1), k0, k0 + 1] {
        if k == 0 {
            continue;
        }
        if k == 1 {
            candidates.push(base.to_string());
        } else {
            candidates.push(vec![base; k].join("\n\n"));
        }
    }

    let mut chosen: Option<(String, usize)> = None;
    for head in candidates {
        if count_tokens(tok, &head) > target {
            continue;
        }
        let widx = best_prefix(&head);
        for cand_w in [widx, (widx + 1).min(words.len())] {
            let text = append(&head, cand_w);
            let c = count_tokens(tok, &text);
            let dist = (c as i64 - target as i64).abs();
            let take = match &chosen {
                None => true,
                Some((_, cc)) => dist < (*cc as i64 - target as i64).abs(),
            };
            if take {
                chosen = Some((text, c));
            }
        }
    }
    let (text, c) =
        chosen.ok_or_else(|| "could not build a prompt of the requested length".to_string())?;
    Ok((text, c))
}

/// Deterministic stand-in for /tmp/gen_prompt.txt when it is missing: same shape
/// as the original (12 paragraphs of word-salad prose, ~5.9 KB, ~877 tokens), so
/// token density and therefore the calibration behave the same way.
fn synthetic_prompt() -> String {
    const WORDS: [&str; 40] = [
        "paragraph", "discusses", "weight", "ceiling", "sink", "order", "draft", "a", "quantized",
        "engine", "channel", "bias", "streams", "through", "acceptance", "rope", "dispatch",
        "logits", "attention", "table", "token", "bandwidth", "latency", "block", "broadcast",
        "scale", "decode", "group", "reduction", "outlier", "scheduler", "rolling", "window",
        "kernel", "residual", "verify", "stream", "cache", "prefill", "sequence",
    ];
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as usize) % WORDS.len()
    };
    let mut out = String::new();
    for p in 1..=12 {
        out.push_str(&format!("Paragraph {} ", p));
        let mut sentence = String::new();
        for w in 0..70 {
            let word = WORDS[next()];
            if w == 0 {
                sentence.push_str(word);
            } else {
                sentence.push(' ');
                sentence.push_str(word);
            }
            if (w + 1) % 15 == 0 {
                out.push_str(&sentence);
                out.push_str(". ");
                sentence.clear();
            }
        }
        if !sentence.is_empty() {
            out.push_str(&sentence);
            out.push_str(". ");
        }
        out.push('\n');
    }
    out
}

fn ensure_prompt_file(path: &Path) -> Result<(String, bool), String> {
    if let Ok(s) = fs::read_to_string(path) {
        if !s.trim().is_empty() {
            return Ok((s, false));
        }
    }
    let text = synthetic_prompt();
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            let _ = fs::create_dir_all(dir);
        }
    }
    fs::write(path, &text).map_err(|e| format!("cannot write {}: {}", path.display(), e))?;
    Ok((text, true))
}

// ------------------------------------------------------------------
// engine calls
// ------------------------------------------------------------------

fn loadavg() -> String {
    if let Ok(o) = Command::new("sysctl").arg("-n").arg("vm.loadavg").output() {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    if let Ok(s) = fs::read_to_string("/proc/loadavg") {
        return format!("{{ {} }}", s.trim());
    }
    "unavailable".to_string()
}

fn load1(s: &str) -> f64 {
    let t = s
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim();
    let v: Vec<&str> = t.split_whitespace().collect();
    if v.len() >= 3 {
        if let Ok(x) = v[0].parse::<f64>() {
            return x;
        }
    }
    f64::NAN
}

#[derive(Clone, Deserialize, Debug)]
struct Step {
    i: usize,
    pos: usize,
    tok: u32,
    forward_ns: u64,
    logprob: f64,
}

#[derive(Default, Clone)]
struct RunResult {
    arm: usize,
    round: usize,
    slot: usize,
    prompt_tok: usize,
    prefill_s: f64,
    prefill_tps: f64,
    cold_ms: f64,
    med_ms: f64,
    min_ms: f64,
    steps: Vec<Step>,
    kv4_armed: Option<String>,
    kv4_note: Option<String>,
    cpu_fallback: bool,
    prompt_too_long: Option<String>,
    fatal: Option<String>,
    error: Option<String>,
    load_before: String,
    load_after: String,
    wall_s: f64,
    rc: Option<i32>,
}

impl RunResult {
    fn ok(&self) -> bool {
        self.error.is_none()
            && self.fatal.is_none()
            && self.prompt_too_long.is_none()
            && !self.steps.is_empty()
    }
}

fn strip_ansi(s: &str) -> String {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == '\u{1b}' {
            i += 1;
            if i < b.len() && b[i] == '[' {
                i += 1;
                while i < b.len() && !b[i].is_ascii_alphabetic() {
                    i += 1;
                }
                i += 1;
                continue;
            }
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

fn parse_secs(tok: &str) -> Option<f64> {
    let t = tok.trim();
    if let Some(x) = t.strip_suffix("ms") {
        return x.parse::<f64>().ok().map(|v| v / 1e3);
    }
    if let Some(x) = t.strip_suffix("µs") {
        return x.parse::<f64>().ok().map(|v| v / 1e6);
    }
    if let Some(x) = t.strip_suffix("us") {
        return x.parse::<f64>().ok().map(|v| v / 1e6);
    }
    if let Some(x) = t.strip_suffix("ns") {
        return x.parse::<f64>().ok().map(|v| v / 1e9);
    }
    if let Some(x) = t.strip_suffix('s') {
        return x.parse::<f64>().ok();
    }
    None
}

/// Parse the engine's "prefill: 493 tk in 300.99ms (1637.9 tk/s)" line into
/// (tokens, seconds, tokens per second).
fn parse_prefill(line: &str) -> Option<(usize, f64, f64)> {
    let p = line.find("prefill: ")?;
    let rest = &line[p + "prefill: ".len()..];
    let mut it = rest.split_whitespace();
    let n: usize = it.next()?.parse().ok()?;
    if it.next()? != "tk" || it.next()? != "in" {
        return None;
    }
    let dur = parse_secs(it.next()?)?;
    let rate_tok = it.next()?;
    let rate = rate_tok
        .trim_start_matches('(')
        .parse::<f64>()
        .ok()
        .unwrap_or(if dur > 0.0 { n as f64 / dur } else { 0.0 });
    Some((n, dur, rate))
}

struct ExecOutcome {
    rc: Option<i32>,
    timed_out: bool,
    err_text: String,
    wall_s: f64,
}

fn exec_engine(
    bin: &Path,
    opts: &Options,
    arm: &Arm,
    prompt: &str,
    trace: &Path,
    out_path: &Path,
    err_path: &Path,
) -> Result<ExecOutcome, String> {
    let create = |p: &Path| -> Result<fs::File, String> {
        fs::File::create(p).map_err(|e| format!("cannot create {}: {}", p.display(), e))
    };
    let mut cmd = Command::new(bin);
    cmd.arg("infer")
        .arg(&opts.model)
        .arg(prompt)
        .arg("--max-tokens")
        .arg(opts.tokens.to_string())
        .arg("--temp")
        .arg("0")
        .arg("--no-lookahead")
        .arg("--trace")
        .arg(trace);
    if let Some(kv) = &arm.kv_arg {
        cmd.arg("--kv-dtype").arg(kv);
    }
    if let Some(ctx) = opts.ctx {
        cmd.arg("--ctx").arg(ctx.to_string());
    }
    // The arm is chosen explicitly; an inherited SOFUU_KV_DTYPE would override
    // the f32 reference arm without saying so.
    cmd.env_remove("SOFUU_KV_DTYPE");
    if opts.gpu_flags {
        cmd.env("SOFUU_GPU_PREFILL", "1");
        cmd.env("SOFUU_GEMM_MMA8", "1");
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::from(create(out_path)?))
        .stderr(Stdio::from(create(err_path)?));

    let t0 = Instant::now();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("cannot spawn {}: {}", bin.display(), e))?;
    let deadline = t0 + Duration::from_secs(opts.run_timeout_s);
    let mut timed_out = false;
    let rc = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st.code(),
            Ok(None) => {}
            Err(e) => return Err(format!("wait failed: {e}")),
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            timed_out = true;
            break None;
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    let wall_s = t0.elapsed().as_secs_f64();
    let err_text = fs::read_to_string(err_path).unwrap_or_default();
    Ok(ExecOutcome {
        rc,
        timed_out,
        err_text,
        wall_s,
    })
}

fn read_trace(path: &Path) -> Result<Vec<Step>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read trace: {e}"))?;
    let mut steps = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let s: Step = serde_json::from_str(line).map_err(|e| format!("bad trace row: {e}"))?;
        steps.push(s);
    }
    Ok(steps)
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

fn least_squares(xs: &[f64], ys: &[f64]) -> Option<(f64, f64, f64)> {
    if xs.len() < 2 || xs.len() != ys.len() {
        return None;
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        sxx += (x - mx) * (x - mx);
        sxy += (x - mx) * (y - my);
    }
    if sxx == 0.0 {
        return None;
    }
    let b = sxy / sxx;
    let a = my - b * mx;
    let mut ssres = 0.0;
    let mut sstot = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let yhat = a + b * x;
        ssres += (y - yhat) * (y - yhat);
        sstot += (y - my) * (y - my);
    }
    let r2 = if sstot == 0.0 { 1.0 } else { 1.0 - ssres / sstot };
    Some((a, b, r2))
}

// ------------------------------------------------------------------
// analysis
// ------------------------------------------------------------------

#[derive(Default, Clone)]
struct Engage {
    armed: usize,
    compared: usize,
    identical: usize,
    diverged: usize,
    max_dlp: f64,
    n_diff: usize,
    engaged: bool,
}

/// Returns (compared steps, identical stream?, max |dlp|, differing steps, diverged?).
fn compare_to_ref(run: &RunResult, reference: &RunResult) -> (usize, bool, f64, usize, bool) {
    let n = run.steps.len().min(reference.steps.len());
    let mut max_dlp: f64 = 0.0;
    let mut n_diff = 0usize;
    let mut diverged = false;
    let mut compared = 0usize;
    for k in 0..n {
        // only same-position, same-token steps are comparable
        if run.steps[k].pos != reference.steps[k].pos {
            continue;
        }
        if run.steps[k].tok != reference.steps[k].tok {
            diverged = true;
            continue;
        }
        compared += 1;
        let d = (run.steps[k].logprob - reference.steps[k].logprob).abs();
        if d > max_dlp {
            max_dlp = d;
        }
        if d >= DLP_TOL {
            n_diff += 1;
        }
    }
    let identical = !diverged && run.steps.len() == reference.steps.len();
    (compared, identical, max_dlp, n_diff, diverged)
}

#[derive(Default, Clone)]
struct ArmStat {
    arm: usize,
    n_ok: usize,
    prompt_tok: usize,
    med_ms: f64,
    min_ms: f64,
    cold_ms: f64,
    tps_med: f64,
    tps_best: f64,
    prefill_tps: f64,
    prefill_s: f64,
    /// max/min of this arm's per-run ms/token at this point
    spread: f64,
    kv4_count: usize,
    engage: Option<Engage>,
}

struct PointStat {
    target: usize,
    built_tokens: usize,
    prompt_tok: usize,
    load_before: String,
    load_after: String,
    max_load: f64,
    /// Some(arm label): this point's runs inside one arm were too dispersed
    spread_bad: Option<String>,
    reliable: bool,
    cpu_fallback: bool,
    skip: Option<String>,
    arms: Vec<ArmStat>,
}

fn arm_stats(opts: &Options, runs: &[RunResult], point_idx: usize, built_tokens: usize) -> PointStat {
    let mut stat = PointStat {
        target: opts.ctx_points[point_idx],
        built_tokens,
        prompt_tok: 0,
        load_before: String::new(),
        load_after: String::new(),
        max_load: 0.0,
        spread_bad: None,
        reliable: true,
        cpu_fallback: false,
        skip: None,
        arms: Vec::new(),
    };
    // the reference every quantized arm is judged against at this point
    let ref_run = runs
        .iter()
        .find(|r| opts.arms[r.arm].key == "f32" && r.ok());
    for (ai, arm) in opts.arms.iter().enumerate() {
        let my: Vec<&RunResult> = runs.iter().filter(|r| r.arm == ai).collect();
        let ok: Vec<&RunResult> = my.iter().copied().filter(|r| r.ok()).collect();
        let mut s = ArmStat {
            arm: ai,
            n_ok: ok.len(),
            ..Default::default()
        };
        if let Some(first) = ok.first() {
            s.prompt_tok = first.prompt_tok;
            if stat.prompt_tok == 0 {
                stat.prompt_tok = first.prompt_tok;
            }
        }
        if !ok.is_empty() {
            s.med_ms = median(ok.iter().map(|r| r.med_ms).collect()).unwrap_or(f64::NAN);
            s.min_ms = ok.iter().map(|r| r.min_ms).fold(f64::INFINITY, f64::min);
            s.cold_ms = median(ok.iter().map(|r| r.cold_ms).collect()).unwrap_or(f64::NAN);
            s.prefill_s = median(ok.iter().map(|r| r.prefill_s).collect()).unwrap_or(f64::NAN);
            s.prefill_tps = median(ok.iter().map(|r| r.prefill_tps).collect()).unwrap_or(f64::NAN);
            s.tps_med = if s.med_ms > 0.0 { 1000.0 / s.med_ms } else { f64::NAN };
            s.tps_best = if s.min_ms > 0.0 { 1000.0 / s.min_ms } else { f64::NAN };
            s.kv4_count = ok.iter().filter(|r| r.kv4_armed.is_some()).count();
            let lo = ok.iter().map(|r| r.med_ms).fold(f64::INFINITY, f64::min);
            let hi = ok.iter().map(|r| r.med_ms).fold(0.0f64, f64::max);
            s.spread = if lo > 0.0 { hi / lo } else { f64::NAN };
        }
        if arm.kv_arg.is_some() {
            if let Some(refr) = ref_run {
                let mut e = Engage::default();
                for r in ok.iter().copied() {
                    let (compared, identical, max_dlp, n_diff, diverged) = compare_to_ref(r, refr);
                    e.compared += compared;
                    e.identical += if identical { 1 } else { 0 };
                    e.diverged += if diverged { 1 } else { 0 };
                    e.n_diff += n_diff;
                    if max_dlp > e.max_dlp {
                        e.max_dlp = max_dlp;
                    }
                    if r.kv4_armed.is_some() {
                        e.armed += 1;
                    }
                }
                e.engaged = e.armed > 0 && (e.max_dlp >= DLP_TOL || e.diverged > 0);
                s.engage = Some(e);
            }
        }
        stat.arms.push(s);
    }
    let mut max_load: f64 = 0.0;
    for r in runs {
        for l in [&r.load_before, &r.load_after] {
            let v = load1(l);
            if v.is_finite() && v > max_load {
                max_load = v;
            }
        }
        if r.cpu_fallback {
            stat.cpu_fallback = true;
        }
    }
    stat.max_load = max_load;
    if let Some(r) = runs.iter().find(|r| r.prompt_too_long.is_some()) {
        stat.skip = r.prompt_too_long.clone();
    }
    let any_failed = runs.iter().any(|r| !r.ok() && r.prompt_too_long.is_none());
    for a in &stat.arms {
        if a.n_ok > 1 && a.spread.is_finite() && a.spread > opts.spread_max {
            stat.spread_bad = Some(format!("{} spread {:.2}x", opts.arms[a.arm].label, a.spread));
        }
    }
    stat.reliable = stat.skip.is_none()
        && !stat.cpu_fallback
        && !any_failed
        && max_load <= opts.load_max
        && stat.spread_bad.is_none()
        && stat.arms.iter().all(|a| a.n_ok > 0);
    stat
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------

fn find_bin(opts: &Options) -> Result<PathBuf, String> {
    if let Some(b) = &opts.bin {
        if b.exists() {
            return Ok(b.clone());
        }
        return Err(format!("--bin {} does not exist", b.display()));
    }
    if let Ok(b) = std::env::var("SOFUU_BIN") {
        let p = PathBuf::from(b);
        if p.exists() {
            return Ok(p);
        }
    }
    let mut cands: Vec<PathBuf> = vec![
        PathBuf::from("target/release/sofuu"),
        PathBuf::from("target/debug/sofuu"),
    ];
    if let Ok(cwd) = std::env::current_dir() {
        for up in ["", "..", "../..", "../../.."] {
            cands.push(cwd.join(up).join("target/release/sofuu"));
            cands.push(cwd.join(up).join("target/debug/sofuu"));
        }
    }
    for c in cands {
        if c.exists() {
            return Ok(c);
        }
    }
    if let Ok(o) = Command::new("which").arg("sofuu").output() {
        if o.status.success() {
            let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !p.is_empty() {
                return Ok(PathBuf::from(p));
            }
        }
    }
    Err("could not find the sofuu binary: run 'cargo build --release --bin sofuu' first, or pass --bin PATH".into())
}

fn main() {
    match real_main() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("kv_slope: error: {e}");
            std::process::exit(1);
        }
    }
}

fn real_main() -> Result<i32, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse_args(&argv)? {
        Some(o) => o,
        None => return Ok(0),
    };

    // -- prompt base + model tokenizer ------------------------------
    let (base, recreated) = ensure_prompt_file(&opts.prompt_file)?;
    if recreated {
        eprintln!(
            "kv_slope: warning: {} was missing; wrote a deterministic synthetic prompt ({} B). \
             Numbers stay internally valid but are not directly comparable with runs that used the original file.",
            opts.prompt_file.display(),
            base.len()
        );
    }
    let mut gguf = GgufFile::open(&opts.model)
        .map_err(|e| format!("failed to open model {}: {}", opts.model.display(), e))?;
    let mut tok = BpeTokenizer::from_gguf(&mut gguf)
        .ok_or_else(|| "failed to parse the GGUF tokenizer".to_string())?;
    if matches!(
        gguf.metadata.get("tokenizer.ggml.add_bos_token"),
        Some(GgufValue::Bool(false))
    ) {
        tok.bos_token_id = None;
    }
    let base_tokens = count_tokens(&tok, &base);
    let bin = find_bin(&opts)?;

    println!("# kv_slope - quantized-KV slope instrument");
    println!();
    println!("- model: {}", opts.model.display());
    println!("- engine binary: {}", bin.display());
    let arm_labels: Vec<String> = opts.arms.iter().map(|a| a.label.clone()).collect();
    println!("- arms: {} (f32 is the reference)", arm_labels.join(", "));
    println!(
        "- decode tokens/run: {} (trace rows {}, steady {} after dropping the cold first forward)",
        opts.tokens,
        opts.tokens - 1,
        opts.tokens.saturating_sub(1 + opts.warmup_skip)
    );
    println!(
        "- rounds: {} (ABBA A B C C B A per round -> {} runs per arm per point, {} runs per point)",
        opts.rounds,
        opts.rounds * 2,
        opts.rounds * 2 * opts.arms.len()
    );
    println!(
        "- ctx window: {}",
        match opts.ctx {
            Some(c) => format!("--ctx {}", c),
            None => "engine default".to_string(),
        }
    );
    println!(
        "- prompt base: {} ({} B, {} tokens){}",
        opts.prompt_file.display(),
        base.len(),
        base_tokens,
        if recreated { " [RECREATED]" } else { "" }
    );
    if opts.gpu_flags {
        println!("- child env: SOFUU_GPU_PREFILL=1 SOFUU_GEMM_MMA8=1 (GPU prefill/GEMM; without these you measure the CPU)");
    } else {
        println!("- child env: GPU flags NOT set (--no-gpu-flags) - these are not GPU numbers");
    }
    println!(
        "- load discipline: sysctl -n vm.loadavg before/after every point and run; a point peaking above {:.1} is UNRELIABLE",
        opts.load_max
    );
    println!();

    // -- build prompts (dry-run can stop here) ----------------------
    let mut prompts: Vec<(String, usize)> = Vec::new();
    for &target in &opts.ctx_points {
        let (text, c) = build_prompt(&base, target, &tok)?;
        if target != c {
            eprintln!(
                "kv_slope: note: point {} built to {} tokens (tokenizer granularity)",
                target, c
            );
        }
        prompts.push((text, c));
    }
    if opts.dry_run {
        println!("## Prompt calibration (dry run)");
        println!();
        println!("| requested tok | built tok | bytes |");
        println!("|---|---|---|");
        for (p, (text, c)) in opts.ctx_points.iter().zip(prompts.iter()) {
            println!("| {} | {} | {} |", p, c, text.len());
        }
        return Ok(0);
    }

    // -- run the matrix ---------------------------------------------
    let workdir = std::env::temp_dir().join(format!("kv_slope_{}", std::process::id()));
    fs::create_dir_all(&workdir).map_err(|e| format!("cannot create work dir: {e}"))?;
    let mut all_runs: Vec<RunResult> = Vec::new();
    let mut points: Vec<PointStat> = Vec::new();
    let runs_per_point = opts.rounds * 2 * opts.arms.len();
    let total_runs = runs_per_point * opts.ctx_points.len();
    let mut run_idx = 0usize;

    for (pi, &target) in opts.ctx_points.iter().enumerate() {
        let (prompt, built) = prompts[pi].clone();
        let load_before_point = loadavg();
        let mut point_runs: Vec<RunResult> = Vec::new();
        let mut slot_no = 0usize;
        eprintln!(
            "[kv_slope] point {} ({}/{}): load {}",
            target,
            pi + 1,
            opts.ctx_points.len(),
            load_before_point
        );
        for round in 0..opts.rounds {
            // ABBA: forward order then reversed
            let mut order: Vec<usize> = (0..opts.arms.len()).collect();
            let mut back = order.clone();
            back.reverse();
            order.extend(back);
            for &ai in order.iter() {
                slot_no += 1;
                run_idx += 1;
                let arm = opts.arms[ai].clone();
                let trace = workdir.join(format!(
                    "p{}_r{}_s{}_{}.jsonl",
                    target, round, slot_no, arm.key
                ));
                let out_path = trace.with_extension("out");
                let err_path = trace.with_extension("err");
                let lb = loadavg();
                let outcome = exec_engine(&bin, &opts, &arm, &prompt, &trace, &out_path, &err_path)?;
                let la = loadavg();

                let mut r = RunResult {
                    arm: ai,
                    round,
                    slot: slot_no,
                    load_before: lb,
                    load_after: la,
                    wall_s: outcome.wall_s,
                    rc: outcome.rc,
                    ..Default::default()
                };
                let plain = strip_ansi(&outcome.err_text);
                for line in plain.lines() {
                    if line.contains("[kv4] quantized KV armed") {
                        r.kv4_armed = Some(line.trim().to_string());
                    } else if line.contains("[kv4] kv") && line.contains("KV cache:") {
                        r.kv4_note = Some(line.trim().to_string());
                    } else if line.contains("prefill: ") && line.contains(" tk in ") {
                        if let Some((n, dur, rate)) = parse_prefill(line) {
                            r.prompt_tok = n;
                            r.prefill_s = dur;
                            r.prefill_tps = rate;
                        }
                    }
                    if line.contains("prompt is ") && line.contains("but the context window is only")
                    {
                        r.prompt_too_long = Some(line.trim().to_string());
                    }
                    if line.contains("[kv4]") && line.contains("does not apply here") {
                        r.kv4_note = Some(line.trim().to_string());
                    }
                    if line.contains("[kv4]") && line.contains("FATAL") {
                        r.fatal = Some(line.trim().to_string());
                    }
                    if line.contains("decode client NOT armed") || line.contains("decodes on the CPU")
                    {
                        r.cpu_fallback = true;
                    }
                }
                if outcome.timed_out {
                    r.error = Some(format!("timed out after {}s", opts.run_timeout_s));
                } else if r.prompt_too_long.is_some() {
                    // a skip reason, not a failure
                } else if outcome.rc != Some(0) {
                    let tail: Vec<&str> = plain.lines().rev().take(4).collect();
                    r.error = Some(format!(
                        "engine exited with {:?}: {}",
                        outcome.rc,
                        tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
                    ));
                } else {
                    match read_trace(&trace) {
                        Ok(steps) => {
                            if steps.len() <= opts.warmup_skip {
                                r.error = Some(format!(
                                    "trace has {} rows, warmup-skip is {}",
                                    steps.len(),
                                    opts.warmup_skip
                                ));
                            } else {
                                let cold = steps
                                    .first()
                                    .map(|s| s.forward_ns as f64 / 1e6)
                                    .unwrap_or(f64::NAN);
                                // trace steps are 1-based; drop the cold first forward(s)
                                let steady: Vec<f64> = steps
                                    .iter()
                                    .filter(|s| s.i > opts.warmup_skip)
                                    .map(|s| s.forward_ns as f64 / 1e6)
                                    .collect();
                                r.med_ms = median(steady.clone()).unwrap_or(f64::NAN);
                                r.min_ms = steady.iter().copied().fold(f64::INFINITY, f64::min);
                                r.cold_ms = cold;
                                r.steps = steps;
                            }
                        }
                        Err(e) => r.error = Some(e),
                    }
                }
                if r.prompt_tok == 0 && r.error.is_none() && r.prompt_too_long.is_none() {
                    r.error = Some("no prefill line in engine stderr".into());
                }
                if r.ok() {
                    eprintln!(
                        "[{}/{}] pt {:>4} arm {:<6} r{} slot {}: {:.3} ms/tok (best {:.3}) | prefill {:.1} tk/s | {} | load {} -> {}",
                        run_idx,
                        total_runs,
                        target,
                        arm.label,
                        round,
                        slot_no,
                        r.med_ms,
                        r.min_ms,
                        r.prefill_tps,
                        if r.kv4_armed.is_some() {
                            "kv4 armed"
                        } else {
                            "no kv4 receipt"
                        },
                        r.load_before.trim(),
                        r.load_after.trim()
                    );
                } else {
                    eprintln!(
                        "[{}/{}] pt {:>4} arm {:<6} r{} slot {}: FAILED: {}",
                        run_idx,
                        total_runs,
                        target,
                        arm.label,
                        round,
                        slot_no,
                        r.error.clone().unwrap_or_else(|| "skipped".into())
                    );
                }

                // Early engagement precheck: an inert arm is a broken
                // instrument, so stop before spending the rest of the matrix.
                if r.ok() && opts.arms[ai].kv_arg.is_some() {
                    if let Some(refr) = point_runs
                        .iter()
                        .find(|x| opts.arms[x.arm].key == "f32" && x.ok())
                    {
                        let (compared, identical, max_dlp, n_diff, diverged) =
                            compare_to_ref(&r, refr);
                        let armed = r.kv4_armed.is_some();
                        if !armed || (max_dlp < DLP_TOL && !diverged) {
                            eprintln!();
                            eprintln!(
                                "[kv_slope] FATAL: arm {} is INERT at {} prompt tokens.",
                                arm.label, r.prompt_tok
                            );
                            eprintln!(
                                "  kv4 receipt: {} | {} decode steps compared against f32 | token stream {} | max |dlogprob| = {:.6} over {} differing steps",
                                if armed { "present" } else { "ABSENT" },
                                compared,
                                if identical {
                                    "identical".to_string()
                                } else if diverged {
                                    "diverged".to_string()
                                } else {
                                    "shorter".to_string()
                                },
                                max_dlp,
                                n_diff
                            );
                            eprintln!(
                                "  A quantized arm whose logits match f32 exactly is not reading the packed blocks: the engine kept the f32 cache, so this run measures f32 speed under a quantized label."
                            );
                            eprintln!(
                                "  Timing it would publish f32 numbers as quantized numbers, so no table is printed."
                            );
                            if let Some(n) = &r.kv4_note {
                                eprintln!("  engine said: {}", n);
                            }
                            eprintln!(
                                "  Real gate: context window > 192 rows (the default 4096 passes), GPU decode on (SOFUU_GPU_DECODE unset or 1), head_dim % 32 == 0 and <= 128, model not resident, and the packed cache actually allocated."
                            );
                            if !opts.keep_traces {
                                let _ = fs::remove_dir_all(&workdir);
                            }
                            return Ok(2);
                        }
                    }
                }

                point_runs.push(r);
            }
            if point_runs.iter().any(|r| r.prompt_too_long.is_some()) {
                break;
            }
        }
        let load_after_point = loadavg();
        let mut st = arm_stats(&opts, &point_runs, pi, built);
        st.load_before = load_before_point;
        st.load_after = load_after_point;
        eprintln!(
            "[kv_slope] point {}: load before {} after {} (max {:.2}){}",
            target,
            st.load_before,
            st.load_after,
            st.max_load,
            if st.reliable { "" } else { "  <<< UNRELIABLE" }
        );
        all_runs.extend(point_runs);
        points.push(st);
        if !opts.keep_traces && pi + 1 == opts.ctx_points.len() {
            let _ = fs::remove_dir_all(&workdir);
        }
    }

    if let Some(path) = &opts.json {
        let mut f =
            fs::File::create(path).map_err(|e| format!("cannot write {}: {}", path.display(), e))?;
        for r in &all_runs {
            let rec = serde_json::json!({
                "arm": opts.arms[r.arm].label,
                "round": r.round,
                "slot": r.slot,
                "prompt_tok": r.prompt_tok,
                "prefill_s": r.prefill_s,
                "prefill_tps": r.prefill_tps,
                "decode_ms_med": r.med_ms,
                "decode_ms_min": r.min_ms,
                "cold_ms": r.cold_ms,
                "n_trace_rows": r.steps.len(),
                "kv4": r.kv4_armed,
                "cpu_fallback": r.cpu_fallback,
                "load_before": r.load_before,
                "load_after": r.load_after,
                "wall_s": r.wall_s,
                "rc": r.rc,
                "error": r.error,
            });
            writeln!(f, "{}", rec).map_err(|e| format!("write {}: {}", path.display(), e))?;
        }
        eprintln!("[kv_slope] run records -> {}", path.display());
    }

    // -- report -----------------------------------------------------
    let mut out = String::new();
    let mut w = |s: &str| out.push_str(s);

    w("\n## Measurements\n\n");
    w("| prompt tok | arm | decode tk/s med | decode tk/s best | ms/token med | ms/token min | run spread | prefill tk/s | prefill s | kv4 | load1 max |\n");
    w("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for p in &points {
        if p.skip.is_some() {
            w(&format!(
                "| {} | - | - | - | - | - | - | - | - | - |\n",
                p.target
            ));
            continue;
        }
        for a in &p.arms {
            let label = &opts.arms[a.arm].label;
            if a.n_ok == 0 {
                w(&format!(
                    "| {} | {} | FAILED | - | - | - | - | - | - | - | {:.2} |\n",
                    p.prompt_tok, label, p.max_load
                ));
                continue;
            }
            w(&format!(
                "| {} | {} | {:.1} | {:.1} | {:.2} | {:.2} | {:.2}x | {:.0} | {:.2} | {} | {:.2}{} |\n",
                p.prompt_tok,
                label,
                a.tps_med,
                a.tps_best,
                a.med_ms,
                a.min_ms,
                a.spread,
                a.prefill_tps,
                a.prefill_s,
                if a.kv4_count > 0 {
                    format!("armed {}/{}", a.kv4_count, a.n_ok)
                } else {
                    "-".to_string()
                },
                p.max_load,
                if p.reliable { "" } else { " UNRELIABLE" }
            ));
        }
    }
    w("\nrun spread = slowest run / fastest run of that arm at that point (1.00x is ideal; over 1.60x means something else was using the machine and the point is UNRELIABLE even when loadavg looks calm). med = median over the arm's runs of each run's median steady forward_ns; best = best single steady token (1000 / min ms), the column REPORT-kvlong.md labels decode tk/s min. The cold first forward is excluded from both.\n");

    w("\nPrompt construction (tokenizer of the model itself, engine-reported length):\n\n");
    w("| requested tok | built tok | engine-reported tok |\n|---|---|---|\n");
    for p in &points {
        w(&format!(
            "| {} | {} | {} |\n",
            p.target, p.built_tokens, p.prompt_tok
        ));
    }
    for p in points.iter().filter(|p| p.skip.is_some()) {
        w(&format!(
            "\n_skipped point {}: {}_\n",
            p.target,
            p.skip.clone().unwrap_or_default()
        ));
    }

    // deltas
    w("\n## Deltas vs f32\n\n");
    w("| prompt tok | arm | decode tk/s | ms/token | prefill tk/s |\n");
    w("|---|---|---|---|---|\n");
    for p in &points {
        if p.skip.is_some() {
            continue;
        }
        let base_arm = p.arms.iter().find(|a| opts.arms[a.arm].key == "f32");
        let Some(base_arm) = base_arm else { continue };
        if base_arm.n_ok == 0 {
            continue;
        }
        for a in p.arms.iter().filter(|a| opts.arms[a.arm].key != "f32") {
            if a.n_ok == 0 {
                continue;
            }
            let pct = |x: f64, b: f64| {
                if b > 0.0 {
                    (x / b - 1.0) * 100.0
                } else {
                    f64::NAN
                }
            };
            w(&format!(
                "| {} | {} | {:+.1}% | {:+.1}% | {:+.1}% |\n",
                p.prompt_tok,
                opts.arms[a.arm].label,
                pct(a.tps_med, base_arm.tps_med),
                pct(a.med_ms, base_arm.med_ms),
                pct(a.prefill_tps, base_arm.prefill_tps)
            ));
        }
    }

    // engagement
    w("\n## Engagement (proof the quantized cache actually ran)\n\n");
    w("| prompt tok | arm | kv4 receipt | same-token steps vs f32 | stream vs f32 | max abs dlogprob | steps that differ | verdict |\n");
    w("|---|---|---|---|---|---|---|---|\n");
    let mut any_inert = false;
    for p in &points {
        if p.skip.is_some() {
            continue;
        }
        for a in &p.arms {
            let arm = &opts.arms[a.arm];
            if arm.kv_arg.is_none() {
                continue;
            }
            match &a.engage {
                None => w(&format!(
                    "| {} | {} | no f32 reference | - | - | - | - | NOT PROVEN |\n",
                    p.prompt_tok, arm.label
                )),
                Some(e) => {
                    if !e.engaged {
                        any_inert = true;
                    }
                    w(&format!(
                        "| {} | {} | {}/{} | {} | {} | {:.6} | {} | {} |\n",
                        p.prompt_tok,
                        arm.label,
                        e.armed,
                        a.n_ok,
                        e.compared,
                        if e.identical == a.n_ok {
                            format!("identical {}/{}", e.identical, a.n_ok)
                        } else {
                            format!("diverged in {}/{}", e.diverged, a.n_ok)
                        },
                        e.max_dlp,
                        e.n_diff,
                        if e.engaged {
                            "ENGAGED"
                        } else {
                            "INERT - DO NOT TRUST"
                        }
                    ));
                }
            }
        }
    }
    w("\nmax abs dlogprob is the largest per-step |log P(token) - log P_f32(token)| on steps where both arms emitted the same token; the last two columns are totals across the arm's runs. 0.000000 together with an identical token stream means the quantized kernel never touched the attention read.\n");

    // fits
    w("\n## Slope fit: ms/token = a + b*ctx\n\n");
    w("| arm | a (ms/token) | b (us per ctx token) | r2 | points used |\n");
    w("|---|---|---|---|---|\n");
    let mut fits: Vec<(usize, f64, f64, f64)> = Vec::new();
    for (ai, arm) in opts.arms.iter().enumerate() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for p in &points {
            if !p.reliable || p.skip.is_some() {
                continue;
            }
            if let Some(a) = p.arms.iter().find(|a| a.arm == ai && a.n_ok > 0) {
                xs.push(p.prompt_tok as f64);
                ys.push(a.med_ms);
            }
        }
        match least_squares(&xs, &ys) {
            Some((a, b, r2)) => {
                fits.push((ai, a, b, r2));
                w(&format!(
                    "| {} | {:.3} | {:.3} | {:.5} | {} |\n",
                    arm.label,
                    a,
                    b * 1000.0,
                    r2,
                    xs.len()
                ));
            }
            None => w(&format!(
                "| {} | - | - | - | {} (need >= 2 reliable points) |\n",
                arm.label,
                xs.len()
            )),
        }
    }

    // gate
    w("\n## Slope ratio - the ship gate\n\n");
    let base_fit = fits
        .iter()
        .find(|(ai, _, _, _)| opts.arms[*ai].key == "f32")
        .copied();
    let mut gate_pass = true;
    let mut gate_lines: Vec<String> = Vec::new();
    match base_fit {
        None => w("f32 slope could not be fitted (need >= 2 reliable points), so no ratio.\n"),
        Some((_, a0, b0, _)) => {
            w("| arm | b (us/ctx token) | b / b_f32 | reading |\n");
            w("|---|---|---|---|\n");
            for fit in fits.iter().filter(|f| opts.arms[f.0].key != "f32") {
                let ai = fit.0;
                let a = fit.1;
                let b = fit.2;
                let ratio = b / b0;
                if ratio > 1.0 {
                    gate_pass = false;
                }
                let reading = if ratio <= 1.0 {
                    "PAYS: no worse than f32 per context token".to_string()
                } else {
                    let cross = if b < b0 && a > a0 {
                        Some((a - a0) / (b0 - b))
                    } else {
                        None
                    };
                    match cross {
                        Some(c) => format!("PAYS only beyond ~{:.0} prompt tokens", c),
                        None => format!(
                            "DOES NOT PAY: penalty reaches {:.2}x ms/token as ctx grows, no crossover",
                            ratio
                        ),
                    }
                };
                w(&format!(
                    "| {} | {:.3} | {:.2}x | {} |\n",
                    opts.arms[ai].label,
                    b * 1000.0,
                    ratio,
                    reading
                ));
                gate_lines.push(format!(
                    "- {}: b = {:.3} us/context token ({:.2}x f32)",
                    opts.arms[ai].label,
                    b * 1000.0,
                    ratio
                ));
            }
        }
    }
    if !gate_pass {
        w("\nSHIP GATE: **FAIL** - at least one quantized arm costs more per context token than f32, so promoting it makes the default slower as context grows.\n");
    } else if base_fit.is_some() {
        w("\nSHIP GATE: **PASS** - every quantized arm's slope is <= f32's, so the quantized cache finally pays at long context.\n");
    }

    // load
    w("\n## Load discipline (sysctl -n vm.loadavg)\n\n");
    w("| prompt tok | before point | after point | max load1 | spread guard | reliable |\n");
    w("|---|---|---|---|---|---|\n");
    let mut unreliable: Vec<usize> = Vec::new();
    for p in &points {
        if !p.reliable && p.skip.is_none() {
            unreliable.push(p.target);
        }
        w(&format!(
            "| {} | {} | {} | {:.2} | {} | {} |\n",
            p.target,
            p.load_before.trim(),
            p.load_after.trim(),
            p.max_load,
            match &p.spread_bad {
                Some(why) => format!("TRIPPED ({})", why),
                None => format!("ok (limit {:.2}x)", opts.spread_max),
            },
            if p.reliable { "yes" } else { "NO" }
        ));
    }
    if !unreliable.is_empty() {
        w(&format!(
            "\nUNRELIABLE points: {:?} - a failed run, a CPU fallback, load1 above {:.1}, or a run spread above {:.2}x during the point.\n",
            unreliable, opts.load_max, opts.spread_max
        ));
    }
    if any_inert {
        w("\n**INERT ARM DETECTED** - a row in the engagement table is not ENGAGED; the timing table must not be used.\n");
    }

    // verdict
    w("\n## Verdict\n\n");
    let blocked = !unreliable.is_empty() || any_inert || base_fit.is_none();
    if blocked {
        w("**NO OVERALL VERDICT.** This matrix cannot support one:");
        if !unreliable.is_empty() {
            w(&format!(
                " {} of {} points were unreliable ({:?}), so the fit and the slope ratio above are indicative only.",
                unreliable.len(),
                points.len(),
                unreliable
            ));
        }
        if any_inert {
            w(" An arm was inert, so its timings are f32 timings under a quantized label.");
        }
        if base_fit.is_none() {
            w(" The f32 slope could not be fitted.");
        }
        w("\n");
        for l in &gate_lines {
            w(&format!("{}\n", l));
        }
    } else {
        for l in &gate_lines {
            w(&format!("{}\n", l));
        }
        if gate_pass {
            w("\n- **PASS**: promote the quantized arm - its marginal cost per context token is not above f32's.\n");
        } else {
            w("\n- **FAIL**: keep f32 as the default. Every quantized arm pays a slope penalty that grows with context; the deltas above give its size at each point.\n");
        }
        let f32_cold = points.iter().find_map(|p| {
            p.arms
                .iter()
                .find(|a| opts.arms[a.arm].key == "f32" && a.n_ok > 0)
                .map(|a| (a.cold_ms, a.med_ms))
        });
        if let Some((cold, med)) = f32_cold {
            w(&format!(
                "\n- cold first forward (excluded) on f32 at the first point: {:.2} ms vs {:.2} ms steady - excluding it keeps med from being an arm-dependent warmup artefact.\n",
                cold, med
            ));
        }
    }

    print!("{}", out);
    let _ = std::io::stdout().flush();

    if any_inert {
        return Ok(2);
    }
    if !unreliable.is_empty() || base_fit.is_none() {
        return Ok(3);
    }
    Ok(0)
}
