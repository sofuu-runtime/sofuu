// rlm/real_model.rs — real-model coverage for the RLM episode loop.
//
// RULES.md §1.9: model inference must run the real implementation and report
// its actual result. The episode/sandbox machinery is covered by the PURE
// tests in episode.rs and js_api.rs (fixed replies in, deterministic state
// machine out). THIS module covers the part that genuinely needs a model:
// the episode loop driven by replies produced by the shipped inference engine
// (crate::infer::InferEngine — the same engine rt/ai.rs loads for
// `provider: "local"`) from weights that already exist on this machine.
// No reply is ever written by hand here.
//
// Model selection never downloads anything (AGENTS.md):
//   1. $SOFUU_RLM_TEST_MODEL — explicit operator selection.
//   2. $SOFUU_CONTRACT_MODEL — the repository's existing real-weight variable.
//   3. the AGENTS.md working set under ~/Downloads, in a documented order.
// Vocabulary-only (ggml-vocab-*.gguf) and synthetic (sofuu-fixture.gguf)
// files are rejected: they are parser fixtures, not models.
//
// Run:  cargo test --offline -p sofuu-core --lib -- rlm::real_model -- --ignored

use std::path::{Path, PathBuf};

use sofuu_ffi::qjs;

use super::episode::{Episode, EpisodeAction};
use super::{Message, RlmOpts, RlmRequest, RlmResult};

/// Real local models this machine may already hold (the AGENTS.md working
/// set), smallest/fastest first. `SOFUU_RLM_TEST_MODEL` overrides the list.
const CANDIDATE_MODELS: &[&str] = &[
    "SmolLM2-360M-Instruct-Q8_0.gguf",
    "Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf",
    "Llama-3.2-1B-Instruct-Q8_0.gguf",
    "laya_multilingual_q8_0.gguf",
    "Spark-X2.5-4B-Q4_K_M.gguf",
    "K2-Horizon-4B-Q8_0.gguf",
    "Qwen3-8B-Q4_K_M.gguf",
    "Phi-4-mini-reasoning-UD-Q6_K_XL.gguf",
];

/// How many independent end-to-end runs the real-model claim needs.
const REAL_RUNS: usize = 3;

/// Model replies longer than this are truncated by the episode anyway; the
/// cap only bounds the inference cost of one turn.
const MAX_TOKENS: usize = 320;

/// Upper bound on the episode steps this driver takes (the episode's own
/// round budget is the real limit; this only stops a driver that never
/// finishes from looping forever).
const MAX_STEPS: usize = 16;

/// The shipped inference engine keeps PROCESS-wide state — the compute
/// thread pool and the GPU decode client — so two concurrent decodes
/// corrupt each other (the shipped local provider serializes them the same
/// way in `rt/ai.rs`). The real-model runs are serialized here so the
/// coverage does not depend on the harness' thread count.
static INFERENCE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take the inference lock, recovering from a poisoned mutex (a previous
/// failure must not make every later run impossible).
fn lock_inference() -> std::sync::MutexGuard<'static, ()> {
    INFERENCE_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The chat template families the shipped engine can render. The family is
/// read from the GGUF's own `tokenizer.chat_template` — the same detection
/// `sofuu infer --chat` performs — so no template is guessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Template {
    ChatMl,
    Llama3,
}

/// A parser fixture, not a model: these may never stand in for inference.
fn is_parser_fixture(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    name == "sofuu-fixture.gguf" || name.starts_with("ggml-vocab-")
}

/// Resolve the real local model this run must use.
fn resolve_model_path() -> Result<PathBuf, String> {
    for var in ["SOFUU_RLM_TEST_MODEL", "SOFUU_CONTRACT_MODEL"] {
        let Some(value) = std::env::var_os(var) else { continue };
        let path = PathBuf::from(value);
        if !path.is_file() {
            return Err(format!(
                "{var} names {}, which is not a readable file",
                path.display()
            ));
        }
        if is_parser_fixture(&path) {
            return Err(format!(
                "{var} names {}, a parser fixture — it cannot stand in for a model",
                path.display()
            ));
        }
        return Ok(path);
    }
    let home = std::env::var_os("HOME").ok_or_else(|| {
        "HOME is not set; point SOFUU_RLM_TEST_MODEL at an existing local GGUF".to_string()
    })?;
    let dir = PathBuf::from(home).join("Downloads");
    for name in CANDIDATE_MODELS {
        let path = dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(format!(
        "BLOCKED rlm real-model: no real local GGUF on this machine. Set SOFUU_RLM_TEST_MODEL to an \
         existing local model (looked for {} under {}). Weights are never downloaded (AGENTS.md).",
        CANDIDATE_MODELS.join(", "),
        dir.display()
    ))
}

/// The model's own chat template family, from the GGUF metadata.
fn detect_template(path: &Path) -> Result<Template, String> {
    let gguf = crate::infer::gguf::GgufFile::open(path).map_err(|e| {
        format!("cannot read the GGUF header of {}: {e}", path.display())
    })?;
    let template = gguf.get_str("tokenizer.chat_template").unwrap_or_default();
    if template.contains("im_start") {
        Ok(Template::ChatMl)
    } else if template.contains("start_header_id") {
        Ok(Template::Llama3)
    } else {
        Err(format!(
            "{} carries no known chat template; the episode transcript cannot be rendered for it",
            path.display()
        ))
    }
}

/// Render the episode transcript in the model's own chat format.
fn render(messages: &[Message], template: Template) -> String {
    let mut out = String::new();
    match template {
        Template::ChatMl => {
            for m in messages {
                out.push_str("<|im_start|>");
                out.push_str(&m.role);
                out.push('\n');
                out.push_str(&m.content);
                out.push_str("<|im_end|>\n");
            }
            out.push_str("<|im_start|>assistant\n");
        }
        Template::Llama3 => {
            out.push_str("<|begin_of_text|>");
            for m in messages {
                out.push_str("<|start_header_id|>");
                out.push_str(&m.role);
                out.push_str("<|end_header_id|>\n\n");
                out.push_str(&m.content);
                out.push_str("<|eot_id|>");
            }
            out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
    }
    out
}

/// A real local model: shipped weights, shipped engine.
struct RealModel {
    engine: crate::infer::InferEngine,
    template: Template,
}

impl RealModel {
    fn load(path: &Path) -> Result<Self, String> {
        let template = detect_template(path)?;
        let engine = crate::infer::InferEngine::load(path).map_err(|e| {
            format!("cannot load the local model {}: {e}", path.display())
        })?;
        Ok(Self { engine, template })
    }

    /// One REAL completion. Greedy (temperature 0) so a run is reproducible.
    fn complete(&mut self, messages: &[Message]) -> Result<String, String> {
        let prompt = render(messages, self.template);
        self.engine.generate(
            &prompt,
            crate::infer::GenerateOptions {
                max_tokens: MAX_TOKENS,
                temperature: 0.0,
                threads: 0,
                ..Default::default()
            },
        )
    }
}

/// The corpus the model has to read. The marker exists ONLY here — it is
/// never in the prompt, so an answer carrying it can only come from the
/// sandbox.
fn corpus_with(marker: &str) -> String {
    format!(
        "Operations memo.\n\nAccess control section: the current access code is {marker}.\n\nEnd of memo.\n"
    )
}

/// The caller's question. It states the shape of the snippet the episode
/// needs — read the context, finish with final() — because the check here
/// is the RLM LOOP (a real model writes a snippet, the real sandbox runs it,
/// the answer is grounded in the real context), not whether a model can
/// discover the sandbox API on its own.
const QUESTION: &str = "What is the access code in the memo? Reply with one ```js snippet that \
                       is exactly: final(chunk(0))";

fn request_for(context: &str) -> RlmRequest {
    let mut opts = RlmOpts::default();
    // A real turn takes seconds, not milliseconds: give the episode a wall
    // budget that fits real inference, and cap the rounds so a stuck model
    // cannot spin.
    opts.max_wall_ms = 600_000;
    opts.max_rounds = 6;
    opts.trace = true;
    RlmRequest {
        context: context.to_string(),
        question: QUESTION.to_string(),
        opts,
    }
}

/// What one real run produced: the episode result plus every reply the
/// model gave (kept for the failure message — a failed run must show what
/// the model actually said).
struct RealRun {
    result: RlmResult,
    replies: Vec<String>,
}

/// Drive one episode with the real model. Tool suspensions are NOT fulfilled
/// with a placeholder: this driver has no tool executor, so a model that
/// asks for a tool fails the run instead of receiving a fabricated result.
fn drive(model: &mut RealModel, episode: &mut Episode) -> Result<RealRun, String> {
    let mut action = episode.start();
    let mut replies: Vec<String> = Vec::new();
    for _ in 0..MAX_STEPS {
        action = match action {
            EpisodeAction::SendMessages(messages) => {
                let reply = model.complete(&messages)?;
                replies.push(reply.clone());
                episode.step(&reply)
            }
            EpisodeAction::ResolveLlm { prompts, .. } => {
                let mut answers = Vec::with_capacity(prompts.len());
                for prompt in &prompts {
                    answers.push(model.complete(&[Message::user(prompt)])?);
                }
                episode.feed_llm_answers(answers)
            }
            EpisodeAction::ResolveTool { calls, .. } => {
                return Err(format!(
                    "the real-model driver has no tool executor; the model asked for {:?}",
                    calls
                ));
            }
            EpisodeAction::Done(result) => return Ok(RealRun { result, replies }),
        };
    }
    Err("the episode did not terminate".to_string())
}

/// One independently initiated end-to-end run: a fresh model load, a fresh
/// episode and a fresh marker.
fn real_run(path: &Path, marker: &str) -> Result<RealRun, String> {
    let mut model = RealModel::load(path)?;
    let mut episode = Episode::new(request_for(&corpus_with(marker)), 0)?;
    drive(&mut model, &mut episode)
}

/// The real-model claim: three independently initiated successful end-to-end
/// runs, each grounded in the sandbox context by a marker the prompt never
/// contained. A model that fails to follow the instruction fails the run.
#[test]
#[ignore = "real-model coverage: needs an existing local GGUF (SOFUU_RLM_TEST_MODEL); run with --ignored"]
fn real_model_episode_grounds_its_answer_in_the_sandbox_context() {
    let _guard = lock_inference();
    let path = resolve_model_path().expect("a real local model must be available for this run");
    for run in 0..REAL_RUNS {
        let marker = format!("VQ7-KITE-{}", 4821 + run * 137);
        let outcome = real_run(&path, &marker)
            .unwrap_or_else(|e| panic!("run {run}: the real episode failed: {e}"));
        let result = outcome.result;
        eprintln!(
            "run {run}: marker {marker} → model replied {:?} → answer {:?}",
            outcome.replies.first().map(|r| r.chars().take(160).collect::<String>()),
            result.answer.chars().take(160).collect::<String>()
        );
        assert!(
            result.answer.contains(&marker),
            "run {run}: the answer must carry the marker {marker} read from the sandbox; got: {}\n\
             the model replied: {:?}",
            result.answer,
            outcome.replies
        );
        assert_eq!(
            result.stopped, None,
            "run {run}: the episode must finish through final(), not through a budget stop"
        );
        assert!(
            result.rounds >= 1,
            "run {run}: the model must answer; rounds: {}",
            result.rounds
        );
        let kinds: Vec<&str> = result.trace.iter().map(|e| e.kind.as_str()).collect();
        assert!(
            kinds.contains(&"snippet"),
            "run {run}: the trace must carry the model's snippet: {kinds:?}"
        );
        assert!(
            kinds.contains(&"final"),
            "run {run}: the trace must carry the final answer: {kinds:?}"
        );
    }
}

/// The JS-facing seam with a REAL model reply: the action JSON the shipped
/// driver consumes must carry the state the episode reached from a real
/// model turn.
#[test]
#[ignore = "real-model coverage: needs an existing local GGUF (SOFUU_RLM_TEST_MODEL); run with --ignored"]
fn js_api_wire_carries_a_real_model_reply() {
    use sofuu_ffi::qjs;

    let _guard = lock_inference();
    let path = resolve_model_path().expect("a real local model must be available for this run");
    let mut model = RealModel::load(&path).expect("load the real local model");

    // SAFETY: single-threaded test; ctx/rt are torn down at the end.
    unsafe {
        let rt = qjs::JS_NewRuntime();
        assert!(!rt.is_null());
        let ctx = qjs::JS_NewContext(rt);
        assert!(!ctx.is_null());
        super::js_api::register_primitives(ctx);

        // Drive the wire the way the shipped JS driver does: each host
        // function returns a JSON string that JS parses.
        let request = serde_json::json!({
            "context": corpus_with("VQ7-KITE-9001"),
            "question": QUESTION,
            "opts": { "trace": true, "maxWallMs": 600_000, "maxRounds": 6 }
        })
        .to_string();
        let created = eval_json(ctx, &format!("__rlm_new({})", js_string(&request)));
        let created_json: serde_json::Value =
            serde_json::from_str(&created).expect("create action json");
        let id = created_json["id"]
            .as_str()
            .unwrap_or_else(|| panic!("no episode id: {created}"))
            .to_string();

        let started = eval_json(ctx, &format!("__rlm_start({})", js_string(&id)));
        let started_json: serde_json::Value =
            serde_json::from_str(&started).expect("start action json");
        assert_eq!(started_json["kind"], "messages");
        let messages: Vec<Message> = started_json["messages"]
            .as_array()
            .expect("transcript messages")
            .iter()
            .map(|m| Message {
                role: m["role"].as_str().unwrap_or_default().to_string(),
                content: m["content"].as_str().unwrap_or_default().to_string(),
            })
            .collect();

        let reply = model.complete(&messages).expect("real inference");
        assert!(
            !reply.trim().is_empty(),
            "the real model must return a reply for the episode"
        );

        let stepped = eval_json(
            ctx,
            &format!(
                "__rlm_step({}, {})",
                js_string(&id),
                js_string(&reply)
            ),
        );
        let stepped_json: serde_json::Value =
            serde_json::from_str(&stepped).expect("step action json");
        let kind = stepped_json["kind"].as_str().unwrap_or_default().to_string();
        assert!(
            kind == "messages" || kind == "resolve_llm" || kind == "done",
            "the real model reply must drive the episode to a known action; got: {stepped}"
        );

        // __rlm_free answers with a boolean, so it is stringified for the
        // harness (the other primitives answer with JSON strings).
        let freed = eval_json(ctx, &format!("String(__rlm_free({}))", js_string(&id)));
        assert_eq!(freed, "true", "the episode must be released");

        qjs::JS_FreeContext(ctx);
        qjs::JS_FreeRuntime(rt);
    }
}

/// A JS string literal for an arbitrary Rust string.
fn js_string(value: &str) -> String {
    serde_json::to_string(value).expect("json string literal")
}

/// Evaluate a script on a QuickJS context and return its value as a string.
unsafe fn eval_json(ctx: *mut qjs::JSContext, script: &str) -> String {
    use std::ffi::CStr;
    let c_script = std::ffi::CString::new(script).unwrap_or_default();
    let value = qjs::JS_Eval(
        ctx,
        c_script.as_ptr(),
        script.len(),
        c"<rlm-real-model-test>".as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL,
    );
    if qjs::is_exception(value) {
        let exception = qjs::JS_GetException(ctx);
        let ptr = qjs::sofuu_js_to_cstring(ctx, exception);
        let message = if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        if !ptr.is_null() {
            qjs::sofuu_js_free_cstring(ctx, ptr);
        }
        qjs::sofuu_js_free_value(ctx, exception);
        qjs::sofuu_js_free_value(ctx, value);
        panic!("the RLM wire script threw: {message}");
    }
    let ctp = qjs::CtxPtr::new(ctx);
    let out = qjs::value_to_string(ctp, value).unwrap_or_default();
    qjs::sofuu_js_free_value(ctx, value);
    out
}
