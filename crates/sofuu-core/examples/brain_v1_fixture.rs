// brain_v1_fixture — build a **v1** brain (the flat single-file container the
// retired C shell wrote: a "memories" tensor + a "metadata" JSON stream) so
// `sofuu brain migrate <path>` can be exercised end to end.
//
// It calls exactly what the runtime's own test helper
// (`build_legacy_hash_brain`, crates/sofuu-core/src/rt/memory.rs) calls, only
// through public API: the same 768-dim hash embedder, the same CMA record
// table, and `rt::brain_store::container_write` for the two streams. Nothing
// here re-implements the codec or the brain format.
//
//   cargo run -p sofuu-core --example brain_v1_fixture -- /tmp/brain-v1.qtsq

#[cfg(has_qtsq)]
fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: brain_v1_fixture <path>");
        std::process::exit(2);
    };
    let texts = [
        "The retired C shell wrote one flat container per brain.",
        "Vector space hash-v1 is 768-dimensional and needs no model artifact.",
        "A v1 brain predates the segmented store: one file, no manifest.",
        "Migrating it must import every record and keep a pre-migration copy.",
    ];
    let dim = sofuu_core::embedding::HASH_DIM;
    let mut cma = sofuu_core::memory::cma::Cma::new(dim);
    let mut flat: Vec<f32> = Vec::with_capacity(texts.len() * dim);
    for (i, text) in texts.iter().enumerate() {
        let v = sofuu_core::embedding::hash_v1_features(text);
        flat.extend_from_slice(&v);
        cma.remember(&v, text, "memory", 7 + i as u32);
    }
    let meta = cma.records_json();
    let dims: [u32; 2] = [texts.len() as u32, dim as u32];
    let rc = unsafe {
        sofuu_core::rt::brain_store::container_write(
            &path,
            &[
                sofuu_core::rt::brain_store::Stream::Tensor {
                    name: "memories",
                    data: &flat,
                    dims: &dims,
                },
                sofuu_core::rt::brain_store::Stream::Json {
                    name: "metadata",
                    body: &meta,
                },
            ],
        )
    };
    match rc {
        Ok(()) => println!("wrote v1 brain: {path} ({} records, {dim}-dim)", texts.len()),
        Err(rc) => {
            eprintln!("container_write failed: rc {rc}");
            std::process::exit(1);
        }
    }
}

/* Codec-less builds have no rt::brain_store at all; say so instead of
 * failing to compile, so `cargo build --examples` stays green everywhere. */
#[cfg(not(has_qtsq))]
fn main() {
    eprintln!("this build has no QTSQ codec — it cannot write a v1 brain");
    std::process::exit(1);
}
