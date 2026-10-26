// crates/ml-train/src/trainer_v2.rs — Modular Training Engine for v2 Models
// (PLAN-ML-V2-UPGRADE §3/§7).
//
// Encapsulates all v2 training runs, asymmetric focal penalties, threshold tuning,
// and binary weight baking.

use std::path::PathBuf;

use sofuu_core::ml::net::{param_count, TinyMlp};
use crate::train::*;

pub const FRESHNESS_ARCH_V2: (u32, u32, u32) = (34, 128, 64); // 12,801 params
pub const COMPACTION_ARCH_V2: (u32, u32, u32) = (41, 128, 64); // 13,697 params
pub const RELEVANCE_ARCH_V2: (u32, u32, u32) = (45, 128, 64); // 14,209 params
pub const SUPERVISOR_ARCH_V2: (u32, u32, u32) = (42, 128, 64); // 13,825 params
pub const ALLOC_ARCH_V2: (u32, u32, u32) = (30, 128, 64); // 12,289 params

pub const SELECT_RECALL_F: f32 = 0.95;
pub const SELECT_PRECISION_C: f32 = 0.95;
pub const SELECT_RECALL_R: f32 = 0.90;
pub const SELECT_RECALL_S: f32 = 0.90;
pub const SELECT_RECALL_A: f32 = 0.92;

pub fn weights_path_v2(model: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(format!("../sofuu-core/src/ml/{model}/weights_v2.f32"));
    p
}

fn print_metrics(tag: &str, m: &Metrics) {
    println!(
        "  {tag:<24} acc={:.3}  precision={:.3}  recall={:.3}  f1={:.3}  (n={}, pos={})",
        m.accuracy, m.precision, m.recall, m.f1, m.n, m.positives
    );
}

pub fn train_freshness_v2() {
    println!("=== TRAINING FRESHNESS V2 MODEL ===");
    let processed_path = crate::preprocess::processed_data_dir().join("freshness_processed.jsonl");
    let data = if processed_path.exists() {
        println!("  Loading preprocessed dataset from {}", processed_path.display());
        crate::preprocess::load_processed_examples(&processed_path)
    } else {
        println!("  Building freshness v2 dataset (34 features)…");
        crate::data_freshness::build_v2()
    };
    let pos = data.iter().filter(|e| e.y >= 0.5).count();
    println!("  {} examples ({} positive / {} negative)", data.len(), pos, data.len() - pos);
    let (in_dim, h1, h2) = FRESHNESS_ARCH_V2;
    println!("arch: {in_dim} → {h1} → {h2} → 1 = {} params", param_count(in_dim, h1, h2));

    let split = split_by_groups(&data, 2, 2);
    let mut net = TinyMlp::new(in_dim, h1, h2, vec![0.0; param_count(in_dim, h1, h2) as usize]);
    train_mlp(
        &mut net,
        &data,
        &split,
        &TrainCfg { lr: 0.003, batch: 64, max_epochs: 400, patience: 60, min_epochs: 30, seed: 1, weight_decay: 1e-4 },
    );

    let vp: Vec<f32> = split.val.iter().map(|&i| predict(&net, &data[i].x)).collect();
    let vy: Vec<f32> = split.val.iter().map(|&i| data[i].y).collect();
    let threshold = threshold_for_recall(&vp, &vy, SELECT_RECALL_F);
    println!("  threshold (val, recall≥{SELECT_RECALL_F}): {threshold:.2}");

    let tp: Vec<f32> = split.test.iter().map(|&i| predict(&net, &data[i].x)).collect();
    let ty: Vec<f32> = split.test.iter().map(|&i| data[i].y).collect();
    let m_net = evaluate(&tp, &ty, threshold);
    print_metrics("tiny mlp v2 (freshness)", &m_net);

    let blob = net.to_blob();
    let path = weights_path_v2("freshness");
    std::fs::write(&path, &blob).expect("write weights");
    println!("SUCCESS: Wrote v2 freshness weights to {} ({} bytes)\n", path.display(), blob.len());
}

pub fn train_compaction_v2() {
    println!("=== TRAINING COMPACTION V2 MODEL ===");
    let processed_path = crate::preprocess::processed_data_dir().join("compaction_processed.jsonl");
    let data = if processed_path.exists() {
        println!("  Loading preprocessed dataset from {}", processed_path.display());
        crate::preprocess::load_processed_examples(&processed_path)
    } else {
        println!("  Building compaction v2 dataset (41 features)…");
        crate::data_compaction::build_v2()
    };
    let pos = data.iter().filter(|e| e.y >= 0.5).count();
    println!("  {} examples ({} disposable / {} keep)", data.len(), pos, data.len() - pos);
    let (in_dim, h1, h2) = COMPACTION_ARCH_V2;
    println!("arch: {in_dim} → {h1} → {h2} → 1 = {} params", param_count(in_dim, h1, h2));

    let split = split_by_groups(&data, 2, 2);
    let mut net = TinyMlp::new(in_dim, h1, h2, vec![0.0; param_count(in_dim, h1, h2) as usize]);
    train_mlp(
        &mut net,
        &data,
        &split,
        &TrainCfg { lr: 0.003, batch: 64, max_epochs: 400, patience: 60, min_epochs: 30, seed: 1, weight_decay: 1e-4 },
    );

    let vp: Vec<f32> = split.val.iter().map(|&i| predict(&net, &data[i].x)).collect();
    let vy: Vec<f32> = split.val.iter().map(|&i| data[i].y).collect();
    let threshold = threshold_for_precision(&vp, &vy, SELECT_PRECISION_C);
    println!("  threshold (val, prec≥{SELECT_PRECISION_C}): {threshold:.2}");

    let tp: Vec<f32> = split.test.iter().map(|&i| predict(&net, &data[i].x)).collect();
    let ty: Vec<f32> = split.test.iter().map(|&i| data[i].y).collect();
    let m_net = evaluate(&tp, &ty, threshold);
    print_metrics("tiny mlp v2 (compaction)", &m_net);

    let blob = net.to_blob();
    let path = weights_path_v2("compaction");
    std::fs::write(&path, &blob).expect("write weights");
    println!("SUCCESS: Wrote v2 compaction weights to {} ({} bytes)\n", path.display(), blob.len());
}

pub fn train_relevance_v2() {
    println!("=== TRAINING RELEVANCE V2 MODEL ===");
    let processed_path = crate::preprocess::processed_data_dir().join("relevance_processed.jsonl");
    let data = if processed_path.exists() {
        println!("  Loading preprocessed dataset from {}", processed_path.display());
        crate::preprocess::load_processed_examples(&processed_path)
    } else {
        println!("  Building relevance v2 dataset (45 features)…");
        crate::data_relevance::build_v2()
    };
    let pos = data.iter().filter(|e| e.y >= 0.5).count();
    println!("  {} examples ({} useful / {} noise)", data.len(), pos, data.len() - pos);
    let (in_dim, h1, h2) = RELEVANCE_ARCH_V2;
    println!("arch: {in_dim} → {h1} → {h2} → 1 = {} params", param_count(in_dim, h1, h2));

    let split = split_by_groups(&data, 3, 3);
    let mut net = TinyMlp::new(in_dim, h1, h2, vec![0.0; param_count(in_dim, h1, h2) as usize]);
    train_mlp(
        &mut net,
        &data,
        &split,
        &TrainCfg { lr: 0.003, batch: 64, max_epochs: 400, patience: 60, min_epochs: 30, seed: 1, weight_decay: 1e-4 },
    );

    let vp: Vec<f32> = split.val.iter().map(|&i| predict(&net, &data[i].x)).collect();
    let vy: Vec<f32> = split.val.iter().map(|&i| data[i].y).collect();
    let threshold = threshold_for_recall(&vp, &vy, SELECT_RECALL_R);
    println!("  threshold (val, recall≥{SELECT_RECALL_R}): {threshold:.2}");

    let tp: Vec<f32> = split.test.iter().map(|&i| predict(&net, &data[i].x)).collect();
    let ty: Vec<f32> = split.test.iter().map(|&i| data[i].y).collect();
    let m_net = evaluate(&tp, &ty, threshold);
    print_metrics("tiny mlp v2 (relevance)", &m_net);

    let blob = net.to_blob();
    let path = weights_path_v2("relevance");
    std::fs::write(&path, &blob).expect("write weights");
    println!("SUCCESS: Wrote v2 relevance weights to {} ({} bytes)\n", path.display(), blob.len());
}

pub fn train_supervisor_v2() {
    println!("=== TRAINING SUPERVISOR V2 MODEL ===");
    let processed_path = crate::preprocess::processed_data_dir().join("supervisor_processed.jsonl");
    let data = if processed_path.exists() {
        println!("  Loading preprocessed dataset from {}", processed_path.display());
        crate::preprocess::load_processed_examples(&processed_path)
    } else {
        println!("  Building supervisor v2 dataset (42 features)…");
        crate::data_supervisor::build_v2()
    };
    let pos = data.iter().filter(|e| e.y >= 0.5).count();
    println!("  {} examples ({} waste / {} clean)", data.len(), pos, data.len() - pos);
    let (in_dim, h1, h2) = SUPERVISOR_ARCH_V2;
    println!("arch: {in_dim} → {h1} → {h2} → 1 = {} params", param_count(in_dim, h1, h2));

    let split = split_by_groups(&data, 3, 3);
    let mut net = TinyMlp::new(in_dim, h1, h2, vec![0.0; param_count(in_dim, h1, h2) as usize]);
    train_mlp(
        &mut net,
        &data,
        &split,
        &TrainCfg { lr: 0.003, batch: 64, max_epochs: 400, patience: 60, min_epochs: 30, seed: 1, weight_decay: 1e-4 },
    );

    let vp: Vec<f32> = split.val.iter().map(|&i| predict(&net, &data[i].x)).collect();
    let vy: Vec<f32> = split.val.iter().map(|&i| data[i].y).collect();
    let threshold = threshold_for_recall(&vp, &vy, SELECT_RECALL_S);
    println!("  threshold (val, recall≥{SELECT_RECALL_S}): {threshold:.2}");

    let tp: Vec<f32> = split.test.iter().map(|&i| predict(&net, &data[i].x)).collect();
    let ty: Vec<f32> = split.test.iter().map(|&i| data[i].y).collect();
    let m_net = evaluate(&tp, &ty, threshold);
    print_metrics("tiny mlp v2 (supervisor)", &m_net);

    let blob = net.to_blob();
    let path = weights_path_v2("supervisor");
    std::fs::write(&path, &blob).expect("write weights");
    println!("SUCCESS: Wrote v2 supervisor weights to {} ({} bytes)\n", path.display(), blob.len());
}

pub fn train_alloc_v2() {
    println!("=== TRAINING ALLOC V2 MODEL ===");
    let processed_path = crate::preprocess::processed_data_dir().join("alloc_processed.jsonl");
    let data = if processed_path.exists() {
        println!("  Loading preprocessed dataset from {}", processed_path.display());
        crate::preprocess::load_processed_examples(&processed_path)
    } else {
        println!("  Building alloc v2 dataset (30 features)…");
        crate::data_alloc::build_v2()
    };
    let pos = data.iter().filter(|e| e.y >= 0.5).count();
    println!("  {} examples ({} pressure / {} safe)", data.len(), pos, data.len() - pos);
    let (in_dim, h1, h2) = ALLOC_ARCH_V2;
    println!("arch: {in_dim} → {h1} → {h2} → 1 = {} params", param_count(in_dim, h1, h2));

    let split = split_by_groups(&data, 2, 2);
    let mut net = TinyMlp::new(in_dim, h1, h2, vec![0.0; param_count(in_dim, h1, h2) as usize]);
    train_mlp(
        &mut net,
        &data,
        &split,
        &TrainCfg { lr: 0.003, batch: 64, max_epochs: 400, patience: 60, min_epochs: 30, seed: 1, weight_decay: 1e-4 },
    );

    let vp: Vec<f32> = split.val.iter().map(|&i| predict(&net, &data[i].x)).collect();
    let vy: Vec<f32> = split.val.iter().map(|&i| data[i].y).collect();
    let threshold = threshold_for_recall(&vp, &vy, SELECT_RECALL_A);
    println!("  threshold (val, recall≥{SELECT_RECALL_A}): {threshold:.2}");

    let tp: Vec<f32> = split.test.iter().map(|&i| predict(&net, &data[i].x)).collect();
    let ty: Vec<f32> = split.test.iter().map(|&i| data[i].y).collect();
    let m_net = evaluate(&tp, &ty, threshold);
    print_metrics("tiny mlp v2 (alloc)", &m_net);

    let blob = net.to_blob();
    let path = weights_path_v2("alloc");
    std::fs::write(&path, &blob).expect("write weights");
    println!("SUCCESS: Wrote v2 alloc weights to {} ({} bytes)\n", path.display(), blob.len());
}

pub fn train_all_v2() {
    println!("=======================================================");
    println!("  SOFUU ON-DEVICE DECISION ML SUITE (v2) TRAINING RUN  ");
    println!("=======================================================\n");
    train_freshness_v2();
    train_compaction_v2();
    train_relevance_v2();
    train_supervisor_v2();
    train_alloc_v2();
    println!("=======================================================");
    println!("  ALL 5 V2 ON-DEVICE MODELS SUCCESSFULLY TRAINED & BAKED");
    println!("=======================================================");
}
