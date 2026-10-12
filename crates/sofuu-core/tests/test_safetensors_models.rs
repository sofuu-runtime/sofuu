use sofuu_core::infer::safetensors::SafeTensorsFile;
use sofuu_core::infer::tokenizer::BpeTokenizer;
use std::path::Path;

#[test]
fn test_load_liquid_350_safetensors() {
    let dir = Path::new("/Users/priyanshuboruah/Downloads/liqiud-350");
    if !dir.exists() {
        eprintln!("Directory not found, skipping: {:?}", dir);
        return;
    }

    let model_path = dir.join("model.safetensors");
    let sf = SafeTensorsFile::open(&model_path).expect("failed to open liqiud-350 model.safetensors");
    println!("liqiud-350 total tensors: {}", sf.tensors.len());
    assert!(sf.tensors.len() >= 140, "expected at least 140 tensors");

    // Check embed_tokens
    let (tv, slice) = sf.get_tensor("model.embed_tokens.weight").expect("embed_tokens.weight missing");
    println!("embed_tokens.weight: dtype={:?}, shape={:?}, size_bytes={}", tv.dtype, tv.shape, slice.len());
    assert_eq!(tv.shape, vec![65536, 1024]);
    assert_eq!(slice.len(), 65536 * 1024 * 2); // BF16 = 2 bytes per item

    // Check tokenizer
    let tok_path = dir.join("tokenizer.json");
    let tok = BpeTokenizer::from_hf_file(&tok_path).expect("failed to load tokenizer.json");
    println!("Tokenizer vocab size: {}", tok.vocab.len());
    assert!(tok.vocab.len() >= 64000);

    let tokens = tok.encode("Hello world! How are you?");
    println!("Encoded tokens count: {:?}, tokens={:?}", tokens.len(), tokens);
    assert!(!tokens.is_empty());
}

#[test]
fn test_load_liquid_mlx_safetensors() {
    let dir = Path::new("/Users/priyanshuboruah/Downloads/liquid");
    if !dir.exists() {
        eprintln!("Directory not found, skipping: {:?}", dir);
        return;
    }

    let model_path = dir.join("model.safetensors");
    let sf = SafeTensorsFile::open(&model_path).expect("failed to open liquid model.safetensors");
    println!("liquid MLX total tensors: {}", sf.tensors.len());
    assert!(sf.tensors.len() >= 500, "expected at least 500 tensors");

    // Check embed_tokens
    let (tv, slice) = sf.get_tensor("model.embed_tokens.weight").expect("embed_tokens.weight missing");
    println!("embed_tokens.weight: dtype={:?}, shape={:?}, size_bytes={}", tv.dtype, tv.shape, slice.len());
    assert_eq!(tv.shape, vec![65536, 256]); // U32 packed 4-bit
}
