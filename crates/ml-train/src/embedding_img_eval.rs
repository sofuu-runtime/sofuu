//! M1: IMG1 acceptance eval — runs against the SHIPPED loader path
//! (`baked_model_img`, quantized int8), never the f32 trainer.
//!
//! Held-out = fresh seed + tags never generated under the training seed.
//! Metrics: text→image R@1/R@5 vs (A) raw-144d cosine, (B) lexical
//! caption∩tag overlap; plus a mixed-pool text non-regression check
//! (image vectors must not dethrone true text hits).
//!
//! Ship bars (recommended; owner decides): R@5 ≥ 0.80, beats both
//! baselines by ≥ 5pp, mixed-pool text R@10 within 2pp of text-only.
//! Exit 0 = PASS.

use std::collections::HashSet;

use sofuu_core::embedding::image;

use crate::data_image_gen;

const EVAL_SEED: u64 = 0xE7A1;
const EVAL_N: usize = 600;

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// A hit is ANY same-caption candidate in top-k (siblings are all correct
/// answers to their shared description — exact-draw matching punishes
/// larger candidate pools for having more right answers).
fn recall_at(queries: &[Vec<f32>], qcaps: &[String], cands: &[Vec<f32>], ccaps: &[String], k: usize) -> f32 {
    if queries.is_empty() || cands.is_empty() {
        return 0.0;
    }
    let mut hits = 0;
    for (qi, q) in queries.iter().enumerate() {
        let mut scored: Vec<(f32, usize)> =
            cands.iter().enumerate().map(|(ci, c)| (cos(q, c), ci)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if scored.iter().take(k).any(|&(_, ci)| ccaps[ci] == qcaps[qi]) {
            hits += 1;
        }
    }
    hits as f32 / queries.len() as f32
}

fn lex_score(caption: &str, tag: &str) -> f32 {
    let cw: HashSet<String> = caption.split_whitespace().map(|w| w.to_lowercase()).collect();
    let tw: HashSet<String> = tag.split(|c: char| !c.is_alphanumeric()).map(|w| w.to_lowercase()).collect();
    cw.intersection(&tw).count() as f32
}

pub fn run_img_eval() -> i32 {
    // Model must exist (trained artifact in-tree).
    let model = match image::baked_model_img() {
        Ok(m) => m,
        Err(e) => {
            println!("img-eval: no trained artifact ({e}) — train first (ml-train img-train)");
            return 2;
        }
    };
    println!("img-eval: artifact {:016x}", model.artifact);

    // Held-out scenes: the trainer's hold-tag set (same bar), fresh draws.
    let hold_path = std::env::var("SOFUU_IMG_HOLDTAGS")
        .unwrap_or_else(|_| "/tmp/sofuu_img_hold_tags.txt".to_string());
    let hold_tags: HashSet<String> = match std::fs::read_to_string(&hold_path) {
        Ok(body) => body.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect(),
        Err(_) => {
            println!("img-eval: hold-tag file missing ({hold_path}) — run img-train first");
            return 2;
        }
    };
    let scenes: Vec<_> = data_image_gen::generate(EVAL_SEED, EVAL_N)
        .into_iter()
        .filter(|s| hold_tags.contains(&s.tag))
        .collect();
    println!("img-eval: {} held-out scenes ({} unseen tags)", scenes.len(), hold_tags.len());
    if scenes.len() < 50 {
        println!("img-eval: too few held-out scenes — widen EVAL_N");
        return 2;
    }

    let mut queries: Vec<Vec<f32>> = Vec::new(); // caption SEM2 vecs
    let mut imgs: Vec<Vec<f32>> = Vec::new(); // student image vecs
    let mut raws: Vec<Vec<f32>> = Vec::new(); // raw 144-d (normalized)
    let mut tags: Vec<String> = Vec::new();
    let mut captions: Vec<String> = Vec::new();
    for s in &scenes {
        let t = match sofuu_core::embedding::semantic_v2::semantic_v2(&s.caption) {
            Some(v) => v,
            None => continue,
        };
        let f = image::features_from_rgb(&s.rgb, data_image_gen::SCENE_W, data_image_gen::SCENE_H);
        let v = match model.forward(&f) {
            Some(v) => v.to_vec(),
            None => continue,
        };
        let mut n = 0.0f32;
        for x in f.iter() {
            n += x * x;
        }
        n = n.sqrt().max(1e-12);
        raws.push(f.iter().map(|x| x / n).collect());
        queries.push(t);
        imgs.push(v);
        tags.push(s.tag.clone());
        captions.push(s.caption.clone());
    }
    println!("img-eval: {} usable pairs", queries.len());

    let r1 = recall_at(&queries, &captions, &imgs, &captions, 1);
    let r5 = recall_at(&queries, &captions, &imgs, &captions, 5);
    let raw_r5 = recall_at(&queries, &captions, &raws, &captions, 5);
    // Lexical baseline: rank by caption∩tag word overlap.
    let mut lex_hits = 0;
    for (qi, cap) in captions.iter().enumerate() {
        let mut scored: Vec<(f32, usize)> =
            tags.iter().enumerate().map(|(ci, t)| (lex_score(cap, t), ci)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if scored.iter().take(5).any(|&(_, ci)| ci == qi) {
            lex_hits += 1;
        }
    }
    let lex_r5 = lex_hits as f32 / captions.len().max(1) as f32;

    // Non-regression: the risk is TEXT retrieval degrading when images
    // share the pool — a text query must still find same-caption TEXTS in
    // top-10. (Self-index matching is meaningless under procedural
    // duplicates: identical vectors tie at 1.0 and stable-sort order, not
    // the model, decides ranks.) Compare mixed pool vs text-only pool.
    let text_r = recall_at(&queries, &captions, &queries, &captions, 10);
    let mut pool: Vec<Vec<f32>> = queries.clone();
    let mut pool_caps: Vec<String> = captions.clone();
    pool.extend(imgs.iter().cloned());
    pool_caps.extend(captions.iter().cloned());
    // Mark pool entries as text (true) or image (false) for hit filtering.
    let mut pool_is_text: Vec<bool> = vec![true; queries.len()];
    pool_is_text.extend(std::iter::repeat(false).take(imgs.len()));
    let mut reg_hits = 0;
    for (qi, q) in queries.iter().enumerate() {
        let mut scored: Vec<(f32, usize)> =
            pool.iter().enumerate().map(|(ci, c)| (cos(q, c), ci)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if scored.iter().take(10).any(|&(_, ci)| pool_is_text[ci] && pool_caps[ci] == captions[qi]) {
            reg_hits += 1;
        }
    }
    let reg_r = reg_hits as f32 / queries.len().max(1) as f32;
    println!("img-eval: text-only R@10 {:.3} vs mixed-pool {:.3}", text_r, reg_r);

    println!("img-eval: text→image R@1 {:.3}  R@5 {:.3}", r1, r5);
    println!("img-eval: raw-144d R@5 {:.3}  lexical R@5 {:.3}", raw_r5, lex_r5);

    let mut pass = true;
    let gate = |ok: bool, name: &str, detail: String| {
        println!("img-eval: [{}] {} — {}", if ok { "PASS" } else { "FAIL" }, name, detail);
        ok
    };
    pass &= gate(r5 >= 0.80, "R@5 floor", format!("{r5:.3} >= 0.80"));
    pass &= gate(r5 - raw_r5 >= 0.05, "beats raw features", format!("+{:.3}", r5 - raw_r5));
    pass &= gate(r5 - lex_r5 >= 0.05, "beats lexical", format!("+{:.3}", r5 - lex_r5));
    pass &= gate(reg_r >= text_r - 0.02, "no text degradation", format!("mixed {reg_r:.3} vs text-only {text_r:.3}"));
    if pass {
        println!("img-eval: SHIP");
        0
    } else {
        println!("img-eval: NO-SHIP (see FAIL lines)");
        1
    }
}
