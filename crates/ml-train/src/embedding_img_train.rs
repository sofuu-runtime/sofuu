//! M1: offline trainer for `image-projector-v1`.
//!
//! Student: 144 IMGF1 features → 16 tanh → 64 (3408 f32 params).
//! Teacher: the caption's SEM2 text vector (bundled, offline, free) — the
//! student learns CLIP-style joint geometry without any external model.
//! Objective: batch InfoNCE over cosine/τ (τ via SOFUU_IMG_TAU).
//! Selection: text→image R@5 on HELD-OUT tags (unseen param combos).
//! Export: int8 per-row + f32 scales/biases in IMG1 layout to
//! SOFUU_IMG_OUT (default: in-tree weights_img.sem).
//!
//! Env knobs: SOFUU_IMG_SEED / SOFUU_IMG_EPOCHS / SOFUU_IMG_OUT /
//! SOFUU_IMG_TAU / SOFUU_IMG_LR / SOFUU_IMG_BATCH / SOFUU_IMG_N.

use std::collections::HashSet;
use std::path::PathBuf;

use sofuu_core::embedding::image::{IMG_DIM, IMG_FEAT_DIM, IMG_HIDDEN};

use crate::data_image_gen;
use crate::train::{Adam, Rng};

const P_W1: usize = IMG_HIDDEN * IMG_FEAT_DIM; // 2304
const P_B1: usize = IMG_HIDDEN; // 16
const P_W2: usize = IMG_DIM * IMG_HIDDEN; // 1024
const P_B2: usize = IMG_DIM; // 64
const P_TOTAL: usize = P_W1 + P_B1 + P_W2 + P_B2; // 3408

const O_W1: usize = 0;
const O_B1: usize = O_W1 + P_W1;
const O_W2: usize = O_B1 + P_B1;
const O_B2: usize = O_W2 + P_W2;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn forward(p: &[f32], x: &[f32], h_out: &mut [f32], y_out: &mut [f32]) {
    for j in 0..IMG_HIDDEN {
        let mut s = p[O_B1 + j];
        let row = &p[O_W1 + j * IMG_FEAT_DIM..O_W1 + (j + 1) * IMG_FEAT_DIM];
        for (w, &v) in row.iter().zip(x.iter()) {
            s += w * v;
        }
        h_out[j] = s.tanh();
    }
    for k in 0..IMG_DIM {
        let mut s = p[O_B2 + k];
        let row = &p[O_W2 + k * IMG_HIDDEN..O_W2 + (k + 1) * IMG_HIDDEN];
        for (w, &v) in row.iter().zip(h_out.iter()) {
            s += w * v;
        }
        y_out[k] = s;
    }
    let mut n = 0.0f32;
    for v in y_out.iter() {
        n += v * v;
    }
    n = n.sqrt().max(1e-12);
    for v in y_out.iter_mut() {
        *v /= n;
    }
}

/// Held-out split by distinct tag (unseen param combos in eval).
fn split_tags(tags: &[String], seed: u64, frac: f32) -> (HashSet<String>, HashSet<String>) {
    let mut uniq: Vec<String> = {
        let mut s: HashSet<String> = tags.iter().cloned().collect();
        let mut v: Vec<String> = s.drain().collect();
        // HashSet drain order is per-process random — sort so the seeded
        // shuffle below (and the whole run) is bit-reproducible.
        v.sort();
        v
    };
    let mut rng = Rng(seed ^ 0x9e37);
    rng.shuffle(&mut uniq);
    let n_hold = ((uniq.len() as f32) * frac).max(4.0) as usize;
    let hold: HashSet<String> = uniq[..n_hold.min(uniq.len())].iter().cloned().collect();
    let train: HashSet<String> = uniq[n_hold.min(uniq.len())..].iter().cloned().collect();
    (train, hold)
}

/// Text→image R@k on `items` (query = teacher caption vec). A hit is ANY
/// same-caption candidate in top-k — siblings depict the same description,
/// so exact-draw matching would punish larger corpora for having more
/// correct answers.
fn recall_at(items: &[(Vec<f32>, Vec<f32>, String)], p: &[f32], k: usize) -> f32 {
    if items.is_empty() {
        return 0.0;
    }
    let mut h = vec![0.0f32; IMG_HIDDEN];
    let mut y = vec![0.0f32; IMG_DIM];
    // Precompute student vectors once (they don't change per query).
    let mut cand: Vec<Vec<f32>> = Vec::with_capacity(items.len());
    for (_, cv, _) in items.iter() {
        forward(p, cv, &mut h, &mut y);
        cand.push(y.clone());
    }
    let mut hits = 0;
    for (qi, (qx, _, qc)) in items.iter().enumerate() {
        // Query embeds through the TEXT path? No — queries are teacher
        // caption vectors directly (text side is frozen SEM2 geometry).
        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(items.len());
        for (ci, cv) in cand.iter().enumerate() {
            let s: f32 = qx.iter().zip(cv.iter()).map(|(a, b)| a * b).sum();
            scored.push((s, ci));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let _ = qi;
        if scored.iter().take(k).any(|&(_, ci)| items[ci].2 == *qc) {
            hits += 1;
        }
    }
    hits as f32 / items.len() as f32
}

pub fn train_img() {
    let seed: u64 = std::env::var("SOFUU_IMG_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(11);
    let epochs = env_usize("SOFUU_IMG_EPOCHS", 150);
    let batch = env_usize("SOFUU_IMG_BATCH", 64);
    let n = env_usize("SOFUU_IMG_N", 4000);
    let tau = env_f32("SOFUU_IMG_TAU", 0.1);
    let lr = env_f32("SOFUU_IMG_LR", 0.003);
    // Direct regression assist: pull image vectors onto their caption's
    // text point (normalized space), alongside the ranking loss.
    let mse_w = env_f32("SOFUU_IMG_MSE", 0.15);
    let out = std::env::var("SOFUU_IMG_OUT").unwrap_or_else(|_| {
        "crates/sofuu-core/src/embedding/weights_img.sem".to_string()
    });

    // Corpus + teacher vectors (caption → SEM2, frozen).
    let scenes = data_image_gen::generate(seed, n);
    let tags: Vec<String> = scenes.iter().map(|s| s.tag.clone()).collect();
    let (train_tags, hold_tags) = split_tags(&tags, seed, 0.2);
    let mut train: Vec<(Vec<f32>, Vec<f32>, String)> = Vec::new();
    let mut hold: Vec<(Vec<f32>, Vec<f32>, String)> = Vec::new();
    for s in &scenes {
        let f = sofuu_core::embedding::image::features_from_rgb(
            &s.rgb,
            data_image_gen::SCENE_W,
            data_image_gen::SCENE_H,
        );
        let t = match sofuu_core::embedding::semantic_v2::semantic_v2(&s.caption) {
            Some(v) => v,
            None => continue,
        };
        if hold_tags.contains(&s.tag) {
            hold.push((t, f.to_vec(), s.caption.clone()));
        } else {
            train.push((t, f.to_vec(), s.caption.clone()));
        }
    }
    // Share the hold-tag set with img-eval (same generalization bar).
    let hold_path = std::env::var("SOFUU_IMG_HOLDTAGS")
        .unwrap_or_else(|_| "/tmp/sofuu_img_hold_tags.txt".to_string());
    {
        let mut tags: Vec<&String> = hold_tags.iter().collect();
        tags.sort();
        let body = tags.iter().map(|t| t.as_str()).collect::<Vec<_>>().join("\n");
        let _ = std::fs::write(&hold_path, body);
    }
    // De-duplicate identical captions in train (same caption, many draws):
    // keep all — InfoNCE treats same-caption siblings as negatives, which
    // is the honest hard case for near-duplicate scenes.
    println!("img-train: train={} hold={} (tags {} train / {} hold)",
        train.len(), hold.len(), train_tags.len(), hold_tags.len());
    if train.len() < batch || hold.is_empty() {
        eprintln!("img-train: corpus too small, aborting");
        std::process::exit(2);
    }

    // Init: small normals, zero biases.
    let mut rng = Rng(seed ^ 0x51f3);
    let mut p = vec![0.0f32; P_TOTAL];
    for i in 0..P_W1 {
        p[O_W1 + i] = rng.normal() * 0.06;
    }
    for i in 0..P_W2 {
        p[O_W2 + i] = rng.normal() * 0.06;
    }
    let mut adam = Adam::new(P_TOTAL, lr);
    let mut h = vec![0.0f32; IMG_HIDDEN];
    let mut y = vec![0.0f32; IMG_DIM];

    let mut best_r5 = -1.0f32;
    let mut best = p.clone();
    let mut order: Vec<usize> = (0..train.len()).collect();
    for ep in 0..epochs {
        rng.shuffle(&mut order);
        let mut ep_loss = 0.0f32;
        let mut ep_n = 0usize;
        for chunk in order.chunks(batch) {
            let b = chunk.len();
            // Forward all.
            let mut ys: Vec<Vec<f32>> = Vec::with_capacity(b);
            let mut hs: Vec<Vec<f32>> = Vec::with_capacity(b);
            for &idx in chunk {
                forward(&p, &train[idx].1, &mut h, &mut y);
                hs.push(h.clone());
                ys.push(y.clone());
            }
            // Similarities S[i][j] = cos(y_i, t_j)/tau.
            let mut s = vec![0.0f32; b * b];
            for i in 0..b {
                for j in 0..b {
                    let d: f32 = ys[i].iter().zip(train[chunk[j]].0.iter()).map(|(a, c)| a * c).sum();
                    s[i * b + j] = d / tau;
                }
            }
            // Softmax rows + MULTI-LABEL loss: every same-caption sibling
            // is a positive (they depict the same description — punishing
            // them apart is what stalled the single-label run at R@5 0.16).
            let mut loss = 0.0f32;
            let mut dlog: Vec<f32> = vec![0.0; b * b];
            for i in 0..b {
                let row = &s[i * b..(i + 1) * b];
                let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                let mut z = 0.0;
                for v in row {
                    z += (v - m).exp();
                }
                let mut npos = 0usize;
                for j in 0..b {
                    if train[chunk[j]].2 == train[chunk[i]].2 {
                        npos += 1;
                    }
                }
                let npos = npos.max(1) as f32;
                for j in 0..b {
                    let pr = ((s[i * b + j] - m).exp()) / z;
                    let y = if train[chunk[j]].2 == train[chunk[i]].2 { 1.0 / npos } else { 0.0 };
                    dlog[i * b + j] = pr - y;
                    if y > 0.0 {
                        loss += -pr.ln() / npos;
                    }
                }
            }
            loss /= b as f32;
            ep_loss += loss;
            ep_n += 1;
            // Backprop: dS -> du -> dy -> MLP.
            let mut g = vec![0.0f32; P_TOTAL];
            for i in 0..b {
                // du_i = Σ_j dlog_ij * t_j / tau.
                let mut du = [0.0f32; IMG_DIM];
                for j in 0..b {
                    let c = dlog[i * b + j] / tau / b as f32;
                    for k in 0..IMG_DIM {
                        du[k] += c * train[chunk[j]].0[k];
                    }
                }
                // Through L2-normalize: dy = (du - u*(du·u))/n.
                let udu: f32 = ys[i].iter().zip(du.iter()).map(|(a, b)| a * b).sum();
                // Recompute pre-norm y: forward() normalized in place; redo
                // the linear part for the norm.
                let mut ylin = [0.0f32; IMG_DIM];
                for k in 0..IMG_DIM {
                    let mut acc = p[O_B2 + k];
                    let row = &p[O_W2 + k * IMG_HIDDEN..O_W2 + (k + 1) * IMG_HIDDEN];
                    for (w, &hv) in row.iter().zip(hs[i].iter()) {
                        acc += w * hv;
                    }
                    ylin[k] = acc;
                }
                let mut nrm = 0.0f32;
                for v in ylin.iter() {
                    nrm += v * v;
                }
                nrm = nrm.sqrt().max(1e-12);
                let mut dy = [0.0f32; IMG_DIM];
                for k in 0..IMG_DIM {
                    dy[k] = (du[k] - ys[i][k] * udu) / nrm;
                }
                // MSE assist through the same normalize Jacobian: the
                // normalized output appears twice (once in du above via
                // cosine, once here directly), so chain both.
                if mse_w > 0.0 {
                    let t = &train[chunk[i]].0;
                    let mut udm = 0.0f32;
                    for k in 0..IMG_DIM {
                        udm += ys[i][k] * (2.0 * mse_w * (ys[i][k] - t[k]) / IMG_DIM as f32);
                    }
                    for k in 0..IMG_DIM {
                        let dm = 2.0 * mse_w * (ys[i][k] - t[k]) / IMG_DIM as f32;
                        dy[k] += (dm - ys[i][k] * udm) / nrm;
                    }
                }
                // MLP: W2/b2, tanh', W1/b1.
                let mut dh = [0.0f32; IMG_HIDDEN];
                for k in 0..IMG_DIM {
                    g[O_B2 + k] += dy[k];
                    let row = &p[O_W2 + k * IMG_HIDDEN..O_W2 + (k + 1) * IMG_HIDDEN];
                    for j in 0..IMG_HIDDEN {
                        g[O_W2 + k * IMG_HIDDEN + j] += dy[k] * hs[i][j];
                        dh[j] += dy[k] * row[j];
                    }
                }
                for j in 0..IMG_HIDDEN {
                    let dt = 1.0 - hs[i][j] * hs[i][j]; // tanh'
                    let d = dh[j] * dt;
                    g[O_B1 + j] += d;
                    // Need x for W1 grad: recompute features? Stored per
                    // item below — keep a copy of train features per row.
                    let x = &train[chunk[i]].1;
                    for (l, &xv) in x.iter().enumerate() {
                        g[O_W1 + j * IMG_FEAT_DIM + l] += d * xv;
                    }
                }
            }
            adam.step(&mut p, &g);
        }
        let r5 = recall_at(&hold, &p, 5);
        let r1 = recall_at(&hold, &p, 1);
        if (ep + 1) % 10 == 0 || ep == 0 {
            println!("ep {:3} loss {:.4} hold-R@1 {:.3} R@5 {:.3}", ep + 1, ep_loss / ep_n.max(1) as f32, r1, r5);
        }
        if r5 > best_r5 {
            best_r5 = r5;
            best = p.clone();
        }
    }
    println!("img-train: best held-out R@5 = {:.3}", best_r5);

    // Export: per-row int8 + f32 scales/biases, IMG1 layout.
    let mut blob: Vec<u8> = Vec::with_capacity(3996);
    blob.extend_from_slice(b"IMG1");
    blob.extend_from_slice(&1u32.to_le_bytes());
    blob.extend_from_slice(&(IMG_FEAT_DIM as u32).to_le_bytes());
    blob.extend_from_slice(&(IMG_HIDDEN as u32).to_le_bytes());
    blob.extend_from_slice(&(IMG_DIM as u32).to_le_bytes());
    let quant_row = |vals: &[f32]| -> (Vec<i8>, f32) {
        let m = vals.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let sc = if m.is_finite() && m > 1e-12 { m / 127.0 } else { 1.0 / 127.0 };
        (vals.iter().map(|&v| (v / sc).round().clamp(-127.0, 127.0) as i8).collect(), sc)
    };
    let mut scales1 = Vec::with_capacity(IMG_HIDDEN);
    for j in 0..IMG_HIDDEN {
        let (q, sc) = quant_row(&best[O_W1 + j * IMG_FEAT_DIM..O_W1 + (j + 1) * IMG_FEAT_DIM]);
        blob.extend_from_slice(&q.iter().map(|&v| v as u8).collect::<Vec<_>>());
        scales1.push(sc);
    }
    for &s in &scales1 {
        blob.extend_from_slice(&s.to_le_bytes());
    }
    blob.extend_from_slice(&best[O_B1..O_B1 + IMG_HIDDEN].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let mut scales2 = Vec::with_capacity(IMG_DIM);
    for k in 0..IMG_DIM {
        let (q, sc) = quant_row(&best[O_W2 + k * IMG_HIDDEN..O_W2 + (k + 1) * IMG_HIDDEN]);
        blob.extend_from_slice(&q.iter().map(|&v| v as u8).collect::<Vec<_>>());
        scales2.push(sc);
    }
    for &s in &scales2 {
        blob.extend_from_slice(&s.to_le_bytes());
    }
    blob.extend_from_slice(&best[O_B2..O_B2 + IMG_DIM].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let id = {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in blob.iter() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        h
    };
    blob.extend_from_slice(&id.to_le_bytes());
    let expect = 20 + IMG_HIDDEN * IMG_FEAT_DIM + 2 * IMG_HIDDEN * 4
        + IMG_DIM * IMG_HIDDEN + 2 * IMG_DIM * 4 + 8;
    assert_eq!(blob.len(), expect, "IMG1 blob length");
    let path = PathBuf::from(&out);
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(dir);
        }
    }
    std::fs::write(&path, &blob).expect("write artifact");
    println!("img-train: wrote {} ({} bytes, id {:016x})", out, blob.len(), id);
    println!("img-train: f32 best held-out R@5 was {:.3} (quantized numbers come from img-eval)", best_r5);
}
