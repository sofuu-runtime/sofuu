//! M1: IMG1 diagnostics — is the task well-posed?
//!
//! 1. Teacher separation: cosine stats between caption SEM2 vectors,
//!    same-family vs cross-family (near-1.0 everywhere = ill-posed).
//! 2. Student spread: std of pairwise cosine among held-out image vecs
//!    (≈0 = collapse to one direction).
//! 3. Raw-feature R@5 on held-out (what "no learning" scores).

use std::collections::HashSet;

use sofuu_core::embedding::image;

use crate::data_image_gen;

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn run_img_diag() {
    let scenes = data_image_gen::generate(11, 1200);
    // Distinct captions only (duplicates carry no signal here).
    let mut seen = HashSet::new();
    let mut caps: Vec<(String, u8)> = Vec::new();
    for s in &scenes {
        if seen.insert(s.caption.clone()) {
            caps.push((s.caption.clone(), s.family));
        }
    }
    let mut tvs: Vec<(Vec<f32>, u8)> = Vec::new();
    for (c, f) in &caps {
        if let Some(v) = sofuu_core::embedding::semantic_v2::semantic_v2(c) {
            tvs.push((v, *f));
        }
    }
    println!("img-diag: {} distinct captions embedded", tvs.len());
    let (mut same_sum, mut same_n) = (0.0f64, 0usize);
    let (mut cross_sum, mut cross_n) = (0.0f64, 0usize);
    let (mut same_min, mut cross_min) = (2.0f32, 2.0f32);
    for i in 0..tvs.len() {
        for j in (i + 1)..tvs.len() {
            let c = cos(&tvs[i].0, &tvs[j].0);
            if tvs[i].1 == tvs[j].1 {
                same_sum += c as f64;
                same_n += 1;
                same_min = same_min.min(c);
            } else {
                cross_sum += c as f64;
                cross_n += 1;
                cross_min = cross_min.min(c);
            }
        }
    }
    println!(
        "img-diag: caption cos — same-family mean {:.4} min {:.4} | cross-family mean {:.4} min {:.4}",
        same_sum / same_n.max(1) as f64,
        same_min,
        cross_sum / cross_n.max(1) as f64,
        cross_min
    );

    // Student spread on a few scenes (if a model exists).
    match image::baked_model_img() {
        Ok(m) => {
            let mut vecs: Vec<Vec<f32>> = Vec::new();
            for s in scenes.iter().take(200) {
                let f = image::features_from_rgb(&s.rgb, data_image_gen::SCENE_W, data_image_gen::SCENE_H);
                if let Some(v) = m.forward(&f) {
                    vecs.push(v.to_vec());
                }
            }
            let mut sum = 0.0f64;
            let mut sum2 = 0.0f64;
            let mut n = 0usize;
            for i in 0..vecs.len() {
                for j in (i + 1)..vecs.len() {
                    let c = cos(&vecs[i], &vecs[j]) as f64;
                    sum += c;
                    sum2 += c * c;
                    n += 1;
                }
            }
            let mean = sum / n.max(1) as f64;
            let std = ((sum2 / n.max(1) as f64) - mean * mean).max(0.0).sqrt();
            println!("img-diag: student pairwise cos mean {:.4} std {:.4} (std≈0 = collapse)", mean, std);
        }
        Err(e) => println!("img-diag: no baked model ({e})"),
    }
}
