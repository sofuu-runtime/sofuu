//! Where does the load path actually spend its time?
//!
//! Usage: seg_bench <brain.qtsq> [reps]
//!
//! Reads the manifest, then times, per segment, the two halves of
//! `load_segmented_opts`'s inner loop — the digest check and the decode — so
//! a load optimisation can be aimed instead of guessed at. Prints totals and
//! median/p90 per segment.

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: seg_bench <brain.qtsq> [reps]");
            std::process::exit(2);
        }
    };
    let reps: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(1);
    /* --threads N: decode the same files concurrently and CHECK the result
     * against the sequential pass. The load path can only be parallelised if
     * this proves the decoder is thread-safe. */
    let threads: usize = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    let dir = sofuu_core::rt::brain_store::segments_dir(&path);
    let files: Vec<String> = sofuu_core::rt::brain_store::list_segment_files(&dir)
        .into_iter()
        .map(|f| format!("{dir}/{f}"))
        .collect();
    if files.is_empty() {
        eprintln!("seg_bench: no segments under {dir}");
        std::process::exit(1);
    }
    let bytes: u64 = files
        .iter()
        .filter_map(|f| std::fs::metadata(f).ok())
        .map(|md| md.len())
        .sum();

    let mut digest_us: Vec<f64> = Vec::with_capacity(files.len() * reps);
    let mut read_us: Vec<f64> = Vec::with_capacity(files.len() * reps);
    let mut nrec = 0usize;

    for _ in 0..reps {
        for f in &files {
            let t = Instant::now();
            let _d = sofuu_core::rt::brain_store::file_digest(f);
            digest_us.push(t.elapsed().as_secs_f64() * 1e6);

            let t = Instant::now();
            // SAFETY: read-only decode of a file this process just stat'ed.
            let r = unsafe { sofuu_core::rt::brain_store::read_segment_ex(f) };
            read_us.push(t.elapsed().as_secs_f64() * 1e6);
            match r {
                Ok((recs, _legacy)) => nrec += recs.len(),
                Err(e) => eprintln!("seg_bench: {f}: decode error {e}"),
            }
        }
    }

    let total = |v: &Vec<f64>| v.iter().sum::<f64>() / 1000.0;
    let mut sorted = read_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "{} segments, {:.1} MB, {} records x {} rep(s)",
        files.len(),
        bytes as f64 / 1e6,
        nrec / reps,
        reps
    );
    println!(
        "digest  {:8.1} ms total  {:7.3} ms/segment",
        total(&digest_us),
        total(&digest_us) / digest_us.len() as f64
    );
    println!(
        "decode  {:8.1} ms total  {:7.3} ms/segment  (min {:.3} p50 {:.3} p90 {:.3})",
        total(&read_us),
        total(&read_us) / read_us.len() as f64,
        sorted[0],
        sorted[sorted.len() / 2] / 1000.0,
        sorted[sorted.len() * 9 / 10] / 1000.0
    );
    if threads > 1 {
        let t = Instant::now();
        let per = files.len().div_ceil(threads).max(1);
        let out = std::sync::Mutex::new(Vec::with_capacity(files.len()));
        std::thread::scope(|sc| {
            for part in files.chunks(per) {
                let out = &out;
                sc.spawn(move || {
                    let mut local = Vec::with_capacity(part.len());
                    for f in part {
                        let d = sofuu_core::rt::brain_store::file_digest(f);
                        let r = unsafe { sofuu_core::rt::brain_store::read_segment_ex(f) };
                        local.push((f.clone(), d, r.map(|(v, l)| (v.len(), l))));
                    }
                    out.lock().expect("out").extend(local);
                });
            }
        });
        let par_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut got = out.into_inner().expect("out");
        got.sort_by(|a, b| a.0.cmp(&b.0));
        let mut bad = 0usize;
        for (f, d, r) in &got {
            let seq_d = sofuu_core::rt::brain_store::file_digest(f);
            let seq_r = unsafe { sofuu_core::rt::brain_store::read_segment_ex(f) };
            let seq_r = seq_r.map(|(v, l)| (v.len(), l));
            if d != &seq_d || r != &seq_r {
                bad += 1;
                eprintln!("seg_bench: MISMATCH {f}: par={r:?} seq={seq_r:?}");
            }
        }
        println!(
            "parallel decode ({} threads): {:8.1} ms total, mismatches {}/{}",
            threads,
            par_ms,
            bad,
            files.len()
        );
    }

    println!(
        "decode throughput: {:.1} MB/s",
        (bytes as f64 * reps as f64) / 1e6 / (total(&read_us) / 1000.0)
    );
}
