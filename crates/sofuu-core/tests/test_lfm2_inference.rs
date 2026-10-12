use std::path::Path;
use sofuu_core::infer::{GenerateOptions, InferEngine};

#[test]
fn test_lfm2_liquid_350_inference() {
    let dir = Path::new("/Users/priyanshuboruah/Downloads/liqiud-350");
    if !dir.exists() {
        eprintln!("Directory not found, skipping: {:?}", dir);
        return;
    }

    println!("Loading liqiud-350 via InferEngine...");
    let t0 = std::time::Instant::now();
    let mut engine = InferEngine::load(dir).expect("failed to load liqiud-350");
    let load_dur = t0.elapsed();
    println!("Loaded in {:.2?}: dim={}, layers={}, heads={}, vocab={}",
        load_dur,
        engine.model.config.dim,
        engine.model.config.n_layers,
        engine.model.config.n_heads,
        engine.model.config.vocab_size
    );

    assert_eq!(engine.model.config.dim, 1024);
    assert_eq!(engine.model.config.n_layers, 16);
    assert_eq!(engine.model.config.vocab_size, 65536);

    let prompt = "The capital of France is";
    println!("Prompt: {:?}", prompt);

    let mut generated_tokens = Vec::new();
    let res = engine.stream(
        prompt,
        GenerateOptions {
            max_tokens: 10,
            temperature: 0.0, // greedy argmax
            ..Default::default()
        },
        |token| {
            print!("{token}");
            generated_tokens.push(token.to_string());
            true
        },
    );
    println!();

    assert!(res.is_ok(), "stream generation failed: {:?}", res);
    println!("Generated {} tokens: {:?}", generated_tokens.len(), generated_tokens);
    assert!(!generated_tokens.is_empty(), "expected at least one generated token");
}

#[test]
fn test_lfm2_liquid_mlx_inference() {
    let dir = Path::new("/Users/priyanshuboruah/Downloads/liquid");
    if !dir.exists() {
        eprintln!("Directory not found, skipping: {:?}", dir);
        return;
    }

    println!("Loading liquid MLX via InferEngine...");
    let t0 = std::time::Instant::now();
    let mut engine = InferEngine::load(dir).expect("failed to load liquid MLX");
    let load_dur = t0.elapsed();
    println!("Loaded in {:.2?}: dim={}, layers={}, heads={}, vocab={}",
        load_dur,
        engine.model.config.dim,
        engine.model.config.n_layers,
        engine.model.config.n_heads,
        engine.model.config.vocab_size
    );

    assert_eq!(engine.model.config.dim, 2048);
    assert_eq!(engine.model.config.n_layers, 30);
    assert_eq!(engine.model.config.vocab_size, 65536);

    let prompt = "Hello";
    println!("Prompt: {:?}", prompt);
    let enc = engine.tokenizer.encode(prompt);
    println!("Encoded prompt: {:?}, eos_token_id={:?}", enc, engine.tokenizer.eos_token_id);
    let logits = engine.model.forward(enc[0], 0);
    let mut top_idx = 0;
    let mut top_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val > top_val {
            top_val = val;
            top_idx = idx;
        }
    }
    println!("Argmax token: id={}, val={}, str={:?}", top_idx, top_val, engine.tokenizer.decode(&[top_idx as u32]));

    let mut generated_tokens = Vec::new();
    let res = engine.stream(
        prompt,
        GenerateOptions {
            max_tokens: 5,
            temperature: 0.0,
            ..Default::default()
        },
        |token| {
            print!("{token}");
            generated_tokens.push(token.to_string());
            true
        },
    );
    println!();

    assert!(res.is_ok(), "stream generation failed: {:?}", res);
    println!("Generated {} tokens: {:?}", generated_tokens.len(), generated_tokens);
    assert!(!generated_tokens.is_empty(), "expected at least one generated token");
}
