// rt/memory.rs — Memory subsystem (PLAN-RUST-MIGRATION M9).
//
// Port of the deleted `src/memory/{mod_memory.c, mod_kv.c, mod_agent.c,
// qtsq_adapter.c}` (506 + 585 + 159 + 292 lines), semantics verbatim:
//   sofuu.memory.open(path, vec_dim) → CMA (JS shell over the Rust Cma —
//   the same object ffi_exports.rs's sofuu_cma_* hand to C; here the shell
//   calls crate::memory::cma::Cma directly, no FFI round-trip)
//   cma.remember/rememberEntity/recall/kvHints/forget/flush/decayTick/
//   consolidate/count/markPositive
//   sofuu.kv.open(path, cfg) → KVStore (QTSQ-backed page store, 2-page RAM
//   LRU, streams kv_%08u_K / kv_%08u_V, index.json with %g-formatted 64-dim
//   k_summary signatures — cross-restart search ranks real pages)
//   sofuu.agent.create(name, cma, kv) → agent.prefetch(query, topK)
//
// C symbols replaced: `mod_memory_register`, `mod_kv_register`,
// `mod_agent_register` (engine.c calls them unchanged under its
// SOFUU_QTSQ_PRESENT guard). The exports exist UNCONDITIONALLY; only the
// QTSQ-backed bodies are `#[cfg(has_qtsq)]` (sofuu-core/build.rs mirrors
// the sofuu-ffi probe). Without QTSQ they are no-ops, matching the C build
// where engine.c simply never calls them — JS sees no sofuu.memory/kv/agent.
//
// The brain-file format (two QTSQ streams: "memories" f32 tensor + "metadata"
// JSON) and the KV page format (quantized K/V tensors, stream names
// kv_%08u_{K,V}) are unchanged — files written by the retired C adapter load
// as before, and vice versa. The deterministic local vault password
// ("sofuu-kv-local-v1", qtsq_adapter.c:23) avoids macOS Keychain prompts on
// every launch; encryption is an at-rest obfuscation layer.
//
// Safety notes (deviations from the C, all crash-avoidance only):
//   - C segfaulted (strncpy/memcpy from NULL) on non-string args; here they
//     fail gracefully (init errors / -1 results).
//   - C read vec_dim floats from caller buffers that could be shorter
//     (recall/kvHints/prefetch); here a short vector is a TypeError/empty
//     result instead of an out-of-bounds read.
//   - C's kv_page_save memcpy'd `count` floats from JS arrays without
//     checking their length; here a short K/V array refuses the save.

use sofuu_ffi::qjs::JSContext;

#[cfg(has_qtsq)]
mod impl_qtsq {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::ptr;
    use std::sync::atomic::{AtomicU32, Ordering};

    use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
    use sofuu_ffi::qtsq::{self, QtsqContext};

    // SIMD kernels stay C (src/simd/{neon,avx}.c) — called over FFI, same
    // as rt/ai.rs does.
    extern "C" {
        fn sofuu_cosine_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    }

    const KV_SUMMARY_DIM: usize = 64; // formats.h KV_SUMMARY_DIM
    const QTSQ_OK: c_int = qtsq::QTSQ_OK;

    /// Deterministic local vault password — deliberately no OS keystore
    /// (qtsq_adapter.c:23-41: an unsigned binary would trigger a macOS
    /// Keychain prompt on EVERY launch; brain files are machine-local
    /// at-rest obfuscation, so a fixed password avoids all prompts).
    const SOFUU_QTSQ_LOCAL_PASSWORD: &str = "sofuu-kv-local-v1";

    unsafe fn cstr_opt(p: *const c_char) -> Option<String> {
        if p.is_null() {
            None
        } else {
            std::ffi::CStr::from_ptr(p).to_str().ok().map(|s| s.to_string())
        }
    }

    /* ══════════════════════════════════════════════════════════════
     * QTSQ adapter (port of src/memory/qtsq_adapter.c)
     * ══════════════════════════════════════════════════════════════ */

    unsafe fn qtsq_adapter_encrypt(ctx: *mut QtsqContext) -> c_int {
        let pw = CString::new(SOFUU_QTSQ_LOCAL_PASSWORD).unwrap_or_default();
        qtsq::qtsq_vault_encrypt_password(ctx, pw.as_ptr())
    }

    /// Decrypt a context loaded with qtsq_read when its header says encrypted.
    unsafe fn qtsq_adapter_decrypt(ctx: *mut QtsqContext) -> c_int {
        if !(*ctx).is_encrypted() {
            return QTSQ_OK;
        }
        let pw = CString::new(SOFUU_QTSQ_LOCAL_PASSWORD).unwrap_or_default();
        qtsq::qtsq_vault_decrypt_password(ctx, pw.as_ptr())
    }

    /// cma_qtsq_save — write the "memories" f32 tensor + "metadata" JSON
    /// streams, then pack → encrypt → fail-closed write.
    unsafe fn cma_qtsq_save(
        path: &str,
        vecs: Option<&[f32]>,
        n_memories: usize,
        vec_dim: usize,
        metadata_json: &str,
    ) -> c_int {
        let ctmp = CString::new(format!("{path}.tmp")).unwrap_or_default();
        /* The brain lives at <project>/.sofuu/brain/brain.qtsq now — make
         * sure the folder exists before the codec (which does not mkdir)
         * writes the tmp file. */
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let container = qtsq::qtsq_ctx_alloc();
        if container.is_null() {
            return qtsq::QTSQ_ERR_ALLOC;
        }
        let mut r = qtsq::qtsq_container_create(container);
        if r == QTSQ_OK {
            /* Phase 7: every write context rides the qtc compressor — the
             * zlib write path in the current checkout produces containers
             * its own read path cannot parse back (2026-08-30). zlib stays
             * read-only for legacy files. */
            unsafe { (*container).set_codec(qtsq::QTSQ_CODEC_QTC) };
            /* Stream 1: memory vectors (lossless f32, flat table).
             * OMITTED when there is nothing to store: the QTSQ tensor
             * compressor rejects a zero-length payload, so an "empty"
             * stream cannot be written. An empty brain is therefore a
             * valid container with metadata only — the loader recognizes
             * that shape (see cma_open) instead of calling it corrupt. */
            if let Some(vecs) = vecs {
                if !vecs.is_empty() && vec_dim > 0 {
                    let sub = qtsq::qtsq_ctx_alloc();
                    if sub.is_null() {
                        qtsq::qtsq_free(container);
                        qtsq::qtsq_ctx_free(container);
                        return qtsq::QTSQ_ERR_ALLOC;
                    }
                    let init_rc = qtsq::qtsq_init(sub);
                    if init_rc != QTSQ_OK {
                        qtsq::qtsq_ctx_free(sub);
                        qtsq::qtsq_free(container);
                        qtsq::qtsq_ctx_free(container);
                        return init_rc;
                    }
                    unsafe { (*sub).set_codec(qtsq::QTSQ_CODEC_QTC) };
                    let dims: [u32; 2] = [n_memories as u32, vec_dim as u32];
                    r = qtsq::qtsq_compress_tensor(sub, vecs.as_ptr(), n_memories * vec_dim, dims.as_ptr(), 2);
                    if r == QTSQ_OK {
                        r = qtsq::qtsq_container_add_stream(container, sub, c"memories".as_ptr());
                    }
                    qtsq::qtsq_free(sub);
                    qtsq::qtsq_ctx_free(sub);
                    if r != QTSQ_OK {
                        qtsq::qtsq_free(container);
                        qtsq::qtsq_ctx_free(container);
                        return r;
                    }
                }
            }

            /* Stream 2: metadata JSON */
            if !metadata_json.is_empty() {
                let cmeta = CString::new(metadata_json).unwrap_or_default();
                let sub = qtsq::qtsq_ctx_alloc();
                if sub.is_null() {
                    qtsq::qtsq_free(container);
                    qtsq::qtsq_ctx_free(container);
                    return qtsq::QTSQ_ERR_ALLOC;
                }
                let init_rc = qtsq::qtsq_init(sub);
                if init_rc != QTSQ_OK {
                    qtsq::qtsq_ctx_free(sub);
                    qtsq::qtsq_free(container);
                    qtsq::qtsq_ctx_free(container);
                    return init_rc;
                }
                unsafe { (*sub).set_codec(qtsq::QTSQ_CODEC_QTC) };
                r = qtsq::qtsq_compress_json(sub, cmeta.as_ptr(), cmeta.as_bytes().len());
                if r == QTSQ_OK {
                    r = qtsq::qtsq_container_add_stream(container, sub, c"metadata".as_ptr());
                }
                qtsq::qtsq_free(sub);
                qtsq::qtsq_ctx_free(sub);
                if r != QTSQ_OK {
                    qtsq::qtsq_free(container);
                    qtsq::qtsq_ctx_free(container);
                    return r;
                }
            }

            /* Pack → encrypt → fail-closed write. Write to a .tmp sibling
             * and atomically rename over the real path so a crash mid-write
             * can never corrupt the existing brain file. */
            r = qtsq::qtsq_container_pack(container);
            if r == QTSQ_OK {
                r = qtsq_adapter_encrypt(container);
            }
            if r == QTSQ_OK {
                r = qtsq::qtsq_write(container, ctmp.as_ptr());
            }
        }
        qtsq::qtsq_free(container);
        qtsq::qtsq_ctx_free(container);
        if r == QTSQ_OK {
            /* Atomic publish: rename tmp → real. On failure the tmp file
             * remains (never wipe a good brain with a partial write). */
            if std::fs::rename(format!("{path}.tmp"), path).is_err() {
                r = qtsq::QTSQ_ERR_FORMAT;
            }
        } else {
            let _ = std::fs::remove_file(format!("{path}.tmp"));
        }
        r
    }

    /// Copy a brain we are about to replace to a backup path that is NEVER
    /// clobbered: `<path>.bak`, then `.bak.1`, `.bak.2`, … An earlier
    /// fixed-name `.bak` meant every reopen overwrote the only copy of a
    /// brain we could not read — the one moment the backup matters most.
    fn backup_brain(path: &str) -> String {
        let first = format!("{path}.bak");
        let mut target = first.clone();
        let mut n = 1u32;
        while std::path::Path::new(&target).exists() && n < 1000 {
            target = format!("{path}.bak.{n}");
            n += 1;
        }
        if std::fs::copy(path, &target).is_err() {
            return first;
        }
        target
    }

    struct BrainData {
        vecs: Vec<f32>,
        n: usize,
        dim: usize,
        meta: Option<String>,
        found_memories: bool, /* the "memories" stream was present */
    }

    /// cma_qtsq_load — read + decrypt a container, extract the "memories"
    /// tensor (n/dim from its schema) and the "metadata" JSON stream.
    unsafe fn cma_qtsq_load(path: &str) -> Result<BrainData, c_int> {
        // P3 (AUDIT-2026-09-07): every open is the one chance to clear the
        // `{brain}.tmp` a crashed writer left behind (age-guarded — see
        // sweep_stale_tmp_file).
        sweep_stale_tmp_file(&format!("{path}.tmp"));
        let cpath = CString::new(path).unwrap_or_default();
        let container = qtsq::qtsq_ctx_alloc();
        if container.is_null() {
            return Err(qtsq::QTSQ_ERR_ALLOC);
        }
        let mut r = qtsq::qtsq_init(container);
        if r == QTSQ_OK {
            r = qtsq::qtsq_read(container, cpath.as_ptr());
        }
        if r == QTSQ_OK {
            r = qtsq_adapter_decrypt(container);
        }
        if r == QTSQ_OK && (*container).data_type() != qtsq::QTSQ_TYPE_CONTAINER {
            r = qtsq::QTSQ_ERR_TYPE;
        }
        let mut num_streams: u32 = 0;
        if r == QTSQ_OK {
            r = qtsq::qtsq_container_get_count(container, &mut num_streams);
        }

        let mut vecs: Vec<f32> = Vec::new();
        let mut n = 0usize;
        let mut dim = 0usize;
        let mut meta: Option<String> = None;
        let mut found_memories = false;
        let mut i: u32 = 0;
        while i < num_streams && r == QTSQ_OK {
            let sub = qtsq::qtsq_ctx_alloc();
            if sub.is_null() {
                r = qtsq::QTSQ_ERR_ALLOC;
                break;
            }
            let mut name_buf = [0u8; 256];
            r = qtsq::qtsq_container_get_stream(
                container,
                i,
                sub,
                name_buf.as_mut_ptr() as *mut c_char,
                name_buf.len(),
            );
            if r != QTSQ_OK {
                /* The library only qtsq_init()s out_ctx on SUCCESS — freeing
                 * an uninitialized context here would free garbage pointers.
                 * Drop just the storage block (zeros — safe). */
                qtsq::qtsq_ctx_free(sub);
                break;
            }
            let name = CStr::from_ptr(name_buf.as_ptr() as *const c_char).to_bytes();
            if name == b"memories" && vecs.is_empty() {
                found_memories = true;
                let mut data: *mut f32 = ptr::null_mut();
                let mut count: usize = 0;
                if qtsq::qtsq_decompress_tensor(sub, &mut data, &mut count) == QTSQ_OK {
                    /* ml-2 (AUDIT-2026-09-07): the header dims are hostile-
                     * controlled file content — the C compressor validates
                     * neither side. Accept the tensor only when the declared
                     * [n, dim] shape multiplies out to the real float count;
                     * on mismatch skip the stream entirely (vecs stays empty
                     * + n/dim 0 → cma_open falls through to a fresh shell).
                     * The writer always emits [n, dim], so legit files pass. */
                    let sd = (*sub).schema_num_dims();
                    let shape_ok = if sd >= 2 {
                        ((*sub).schema_dim(0) as usize)
                            .saturating_mul((*sub).schema_dim(1) as usize)
                            == count
                    } else {
                        false
                    };
                    if shape_ok {
                        let slice = std::slice::from_raw_parts(data, count);
                        vecs = slice.to_vec();
                        n = (*sub).schema_dim(0) as usize;
                        dim = (*sub).schema_dim(1) as usize;
                    }
                    libc::free(data as *mut c_void);
                }
            } else if name == b"metadata" && meta.is_none() {
                let mut raw: *mut u8 = ptr::null_mut();
                let mut raw_size: usize = 0;
                if qtsq::qtsq_decompress_horizon(sub, &mut raw, &mut raw_size) == QTSQ_OK {
                    let bytes = std::slice::from_raw_parts(raw, raw_size);
                    meta = Some(String::from_utf8_lossy(bytes).into_owned());
                    libc::free(raw as *mut c_void);
                }
            }
            qtsq::qtsq_free(sub);
            qtsq::qtsq_ctx_free(sub);
            i += 1;
        }

        qtsq::qtsq_free(container);
        qtsq::qtsq_ctx_free(container);

        if r != QTSQ_OK {
            return Err(r);
        }
        Ok(BrainData { vecs, n, dim, meta, found_memories })
    }

    /// kv_qtsq_save_page — quantized K/V tensors as kv_%08u_{K,V} streams.
    /// Returns the first failing rc, or QTSQ_OK (=== 0) for count==0 — the
    /// C returned bare 0 for an empty page and the caller continued.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kv_qtsq_save_page(
        container: *mut QtsqContext,
        k: *const f32,
        v: *const f32,
        n_layers: usize,
        n_heads: usize,
        n_tokens: usize,
        head_dim: usize,
        page_id: u32,
        k_precision: qtsq::QtsqTensorPrecision,
        v_precision: qtsq::QtsqTensorPrecision,
    ) -> c_int {
        if container.is_null() || k.is_null() || v.is_null() {
            return -1;
        }
        let count = n_layers
            .wrapping_mul(n_heads)
            .wrapping_mul(n_tokens)
            .wrapping_mul(head_dim);
        if count == 0 {
            return 0;
        }

        let dims: [u32; 4] = [
            n_layers as u32,
            n_heads as u32,
            n_tokens as u32,
            head_dim as u32,
        ];

        /* 1. K tensor */
        let ctx_k = qtsq::qtsq_ctx_alloc();
        if ctx_k.is_null() {
            return qtsq::QTSQ_ERR_ALLOC;
        }
        qtsq::qtsq_init(ctx_k);
        unsafe { (*ctx_k).set_codec(qtsq::QTSQ_CODEC_QTC) };
        let name_k = format!("kv_{:08}_K", page_id);
        let cname_k = CString::new(name_k).unwrap_or_default();
        let mut r = qtsq::qtsq_compress_tensor_quantized(ctx_k, k, count, dims.as_ptr(), 4, k_precision);
        if r == QTSQ_OK {
            r = qtsq::qtsq_container_add_stream(container, ctx_k, cname_k.as_ptr());
        }
        qtsq::qtsq_free(ctx_k);
        qtsq::qtsq_ctx_free(ctx_k);
        if r != QTSQ_OK {
            return r;
        }

        /* 2. V tensor */
        let ctx_v = qtsq::qtsq_ctx_alloc();
        if ctx_v.is_null() {
            return qtsq::QTSQ_ERR_ALLOC;
        }
        qtsq::qtsq_init(ctx_v);
        unsafe { (*ctx_v).set_codec(qtsq::QTSQ_CODEC_QTC) };
        let name_v = format!("kv_{:08}_V", page_id);
        let cname_v = CString::new(name_v).unwrap_or_default();
        r = qtsq::qtsq_compress_tensor_quantized(ctx_v, v, count, dims.as_ptr(), 4, v_precision);
        if r == QTSQ_OK {
            r = qtsq::qtsq_container_add_stream(container, ctx_v, cname_v.as_ptr());
        }
        qtsq::qtsq_free(ctx_v);
        qtsq::qtsq_ctx_free(ctx_v);

        r
    }

    /// kv_qtsq_load_page — decompress one page's K/V tensors. The K schema
    /// is [layers, heads, tokens, dim] — layers/tokens are read from it.
    unsafe fn kv_qtsq_load_page(
        kv_file: &str,
        page_id: u32,
    ) -> Result<(Vec<f32>, Vec<f32>, usize, usize), c_int> {
        let cpath = CString::new(kv_file).unwrap_or_default();
        let container = qtsq::qtsq_ctx_alloc();
        if container.is_null() {
            return Err(qtsq::QTSQ_ERR_ALLOC);
        }
        let mut r = qtsq::qtsq_init(container);
        if r == QTSQ_OK {
            r = qtsq::qtsq_read(container, cpath.as_ptr());
        }
        if r == QTSQ_OK {
            /* V3: files are written encrypted (fail-closed) — decrypt in place. */
            r = qtsq_adapter_decrypt(container);
        }
        let mut num_streams: u32 = 0;
        if r == QTSQ_OK {
            r = qtsq::qtsq_container_get_count(container, &mut num_streams);
        }

        let name_k = format!("kv_{:08}_K", page_id);
        let name_v = format!("kv_{:08}_V", page_id);
        let cname_k = CString::new(name_k).unwrap_or_default();
        let cname_v = CString::new(name_v).unwrap_or_default();

        let mut k_data: Vec<f32> = Vec::new();
        let mut v_data: Vec<f32> = Vec::new();
        let mut n_layers: usize = 0;
        let mut n_tokens: usize = 0;

        if r == QTSQ_OK {
            for i in 0..num_streams {
                let sub = qtsq::qtsq_ctx_alloc();
                if sub.is_null() {
                    r = qtsq::QTSQ_ERR_ALLOC;
                    break;
                }
                let mut name_buf = [0u8; 256];
                if qtsq::qtsq_container_get_stream(
                    container,
                    i,
                    sub,
                    name_buf.as_mut_ptr() as *mut c_char,
                    name_buf.len(),
                ) == QTSQ_OK
                {
                    let name = CStr::from_ptr(name_buf.as_ptr() as *const c_char).to_bytes();
                    if name == cname_k.as_bytes() {
                        let mut data: *mut f32 = ptr::null_mut();
                        let mut count: usize = 0;
                        if qtsq::qtsq_decompress_tensor(sub, &mut data, &mut count) == QTSQ_OK {
                            /* ml-2 (AUDIT-2026-09-07): the K header is
                             * attacker-controlled in a crafted file — the old
                             * count-parity tail gate accepted hostile dims and
                             * handed them to ActivePage. Require the full
                             * [l, h, tok, d] schema whose saturating product
                             * equals the real float count (the writer always
                             * emits 4-D, so legit pages pass); on mismatch
                             * leave k_data empty — the tail gate then rejects
                             * the page with QTSQ_ERR_FORMAT. */
                            let sd = (*sub).schema_num_dims();
                            let schema_ok = if sd >= 4 {
                                let product = ((*sub).schema_dim(0) as usize)
                                    .saturating_mul((*sub).schema_dim(1) as usize)
                                    .saturating_mul((*sub).schema_dim(2) as usize)
                                    .saturating_mul((*sub).schema_dim(3) as usize);
                                product == count
                            } else {
                                false
                            };
                            if schema_ok {
                                k_data = std::slice::from_raw_parts(data, count).to_vec();
                                n_layers = (*sub).schema_dim(0) as usize;
                                n_tokens = (*sub).schema_dim(2) as usize; /* [l, h, tok, d] */
                            }
                            libc::free(data as *mut c_void);
                        }
                    } else if name == cname_v.as_bytes() {
                        let mut data: *mut f32 = ptr::null_mut();
                        let mut count: usize = 0;
                        if qtsq::qtsq_decompress_tensor(sub, &mut data, &mut count) == QTSQ_OK {
                            v_data = std::slice::from_raw_parts(data, count).to_vec();
                            libc::free(data as *mut c_void);
                        }
                    }
                    qtsq::qtsq_free(sub);
                }
                qtsq::qtsq_ctx_free(sub);
            }
        }

        qtsq::qtsq_free(container);
        qtsq::qtsq_ctx_free(container);

        if r != QTSQ_OK {
            return Err(r);
        }
        if !k_data.is_empty() && !v_data.is_empty() && k_data.len() == v_data.len() {
            return Ok((k_data, v_data, n_layers, n_tokens));
        }
        Err(qtsq::QTSQ_ERR_FORMAT) /* Page not fully found or corrupted */
    }

    /// kv_qtsq_flush — pack the container (streams → singularity), encrypt
    /// (fail-closed gate), then write atomically (tmp + rename).
    unsafe fn kv_qtsq_flush(container: *mut QtsqContext, path: &str) -> c_int {
        let ctmp = CString::new(format!("{path}.tmp")).unwrap_or_default();
        let mut r = qtsq::qtsq_container_pack(container);
        if r == QTSQ_OK {
            r = qtsq_adapter_encrypt(container);
        }
        if r == QTSQ_OK {
            r = qtsq::qtsq_write(container, ctmp.as_ptr());
        }
        if r == QTSQ_OK {
            if std::fs::rename(format!("{path}.tmp"), path).is_err() {
                // P3 (AUDIT-2026-09-07): the old code left the tmp behind on
                // a failed rename — the exact leak the atomic-write scheme
                // exists to avoid.
                let _ = std::fs::remove_file(format!("{path}.tmp"));
                r = qtsq::QTSQ_ERR_FORMAT;
            }
        } else {
            let _ = std::fs::remove_file(format!("{path}.tmp"));
        }
        r
    }

    /* ══════════════════════════════════════════════════════════════
     * CMA shell (port of src/memory/mod_memory.c)
     * ══════════════════════════════════════════════════════════════ */

    // All memory state lives in crate::memory::cma::Cma; this shell owns the
    // mind-file path and the vector dim (i.e. the C cma_t minus its vestigial
    // ctx/obj fields, which the retired C shell never read back).
    struct CmaShell {
        mind_path: String,
        vec_dim: usize,
        /// Identity of the vector space stored in this brain.  A dimension
        /// alone is not enough: two 64-dimensional models are not
        /// interchangeable, so the manifest is checked on every open.
        embedding_id: String,
        cma: crate::memory::cma::Cma,
    }

    static CMA_CLASS_ID: AtomicU32 = AtomicU32::new(0);

    unsafe extern "C" fn cma_finalizer(_rt: *mut qjs::JSRuntime, val: JSValue) {
        let cma = qjs::JS_GetOpaque(val, CMA_CLASS_ID.load(Ordering::Relaxed)) as *mut CmaShell;
        if !cma.is_null() {
            /* Do NOT call cma.flush here — the JS context may already be
             * freed. JS code must call cma.flush() explicitly. (C comment,
             * kept verbatim.) */
            drop(Box::from_raw(cma));
        }
    }

    const CMA_CLASS_DEF: qjs::JSClassDef = qjs::JSClassDef {
        class_name: c"CMA".as_ptr(),
        finalizer: Some(cma_finalizer),
        gc_mark: ptr::null_mut(),
        call: ptr::null_mut(),
        exotic: ptr::null_mut(),
    };

    fn inferred_legacy_embedding_id(dim: usize) -> Option<&'static str> {
        /* The retired local brain used the 768-dimensional hash embedder and
         * wrote no model manifest.  That one legacy space is safe to infer;
         * an unmanifested 64-dimensional file is deliberately ambiguous and
         * is refused rather than mixed with the learned model. */
        if dim == crate::embedding::HASH_DIM {
            Some(crate::embedding::INPUT_EMBEDDER_ID)
        } else {
            None
        }
    }

    #[derive(Clone, Debug)]
    struct EmbeddingManifest {
        id: String,
        input: String,
        dimension: usize,
        artifact: String,
    }

    fn embedding_manifest(meta: &str) -> Option<EmbeddingManifest> {
        let root: serde_json::Value = serde_json::from_str(meta).ok()?;
        let embedding = root.get("embedding")?;
        Some(EmbeddingManifest {
            id: embedding.get("id")?.as_str()?.to_string(),
            input: embedding
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            dimension: embedding.get("dimension")?.as_u64()? as usize,
            artifact: embedding
                .get("artifact")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        })
    }

    fn manifest_matches(meta: &str, expected_id: &str, expected_dim: usize) -> bool {        let Some(manifest) = embedding_manifest(meta) else {
            return false;
        };
        if manifest.id != expected_id || manifest.dimension != expected_dim {
            return false;
        }
        if expected_id == crate::embedding::MODEL_ID {
            manifest.input == crate::embedding::INPUT_EMBEDDER_ID
                && manifest.artifact == crate::embedding::model_artifact_id()
        } else if expected_id == crate::embedding::semantic_v2::MODEL_ID_V2 {
            // SEM2 is its own space: same input embedder (hash-v1 features),
            // different table + artifact — verified identically strict.
            manifest.input == crate::embedding::semantic_v2::INPUT_EMBEDDER_ID_V2
                && manifest.artifact == crate::embedding::semantic_v2::model_artifact_id_v2()
        } else {
            true
        }
    }

    fn metadata_with_embedding(shell: &CmaShell) -> String {
        // P3 (AUDIT-2026-09-07): take the Value directly — the old
        // records_json → from_str round trip serialized and re-parsed the
        // whole record table on every metadata write.
        let mut root: serde_json::Value = shell.cma.records_value();
        if let Some(object) = root.as_object_mut() {
            let (input, artifact) = if shell.embedding_id == crate::embedding::MODEL_ID {
                (
                    crate::embedding::INPUT_EMBEDDER_ID,
                    crate::embedding::model_artifact_id(),
                )
            } else if shell.embedding_id == crate::embedding::semantic_v2::MODEL_ID_V2 {
                (
                    crate::embedding::semantic_v2::INPUT_EMBEDDER_ID_V2,
                    crate::embedding::semantic_v2::model_artifact_id_v2(),
                )
            } else {
                ("", String::new())
            };
            object.insert("schema".into(), serde_json::json!("sofuu-cma@2"));
            object.insert(
                "embedding".into(),
                serde_json::json!({
                    "id": shell.embedding_id,
                    "input": input,
                    "dimension": shell.vec_dim,
                    "artifact": artifact,
                }),
            );
        }
        serde_json::to_string(&root).unwrap_or_else(|_| shell.cma.records_json())
    }

    fn unique_sidecar(path: &str, suffix: &str) -> String {
        let base = format!("{path}.{suffix}");
        // P3 (AUDIT-2026-09-07): pick-by-exists()-then-act was a TOCTOU race —
        // two processes (desktop + CLI on one project) could both claim the
        // same name and the second would clobber the first (a lost backup).
        // create_new claims the name atomically; callers overwrite their own
        // reservation immediately after, and a losing claimant just retries
        // with the next candidate.
        let mut candidate = base.clone();
        if std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
            .is_ok()
        {
            return candidate;
        }
        for i in 1..10_000u32 {
            candidate = format!("{base}.{i}");
            if std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
                .is_ok()
            {
                return candidate;
            }
        }
        // Last resort: the pid-scoped name can only collide with ourselves.
        candidate = format!("{base}.{}.overflow", std::process::id());
        let _ = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate);
        candidate
    }

    /// P3 (AUDIT-2026-09-07): atomic writes stage through `{file}.tmp`
    /// siblings, and a crash between write and rename left them behind
    /// forever. Remove one at open time — but only when clearly stale
    /// (10 minutes old): a concurrent desktop/CLI process on the same
    /// project legitimately holds a fresh tmp mid-write, and deleting that
    /// would turn its rename into a failed save. An unreadable mtime can
    /// never prove liveness, so such files are left alone.
    const STALE_TMP_SECS: u64 = 600;

    fn sweep_stale_tmp_file(file_tmp: &str) {
        let Ok(md) = std::fs::metadata(file_tmp) else {
            return;
        };
        if let Ok(modified) = md.modified() {
            if let Ok(age) = std::time::SystemTime::now().duration_since(modified) {
                if age.as_secs() >= STALE_TMP_SECS {
                    let _ = std::fs::remove_file(file_tmp);
                }
            }
        }
    }

    /// Same sweep for a directory of page containers (`*.qtsq.tmp` etc.).
    fn sweep_stale_tmp_dir(dir: &str) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e == "tmp") {
                sweep_stale_tmp_file(&p.to_string_lossy());
            }
        }
    }

    /// Re-embed a legacy 768-dimensional hash brain into the current
    /// semantic space.  The old file is copied to a unique backup first; the
    /// new QTSQ is staged, read back and manifest-verified before it replaces
    /// the canonical path. Ok((records, backup)) on success; Err(reason)
    /// leaves the original file untouched in every case.
    unsafe fn migrate_legacy_hash_brain(
        path: &str,
        brain: &BrainData,
        shell: &mut CmaShell,
    ) -> Result<(usize, String), String> {
        let refuse = |msg: &str| {
            eprintln!("[sofuu/cma] {msg}; migration refused");
            Err(msg.to_string())
        };
        let Some(meta) = brain.meta.as_deref() else {
            return refuse("legacy brain has no metadata");
        };
        let Ok(root) = serde_json::from_str::<serde_json::Value>(meta) else {
            return refuse("legacy brain metadata is not JSON");
        };
        let Some(records) = root.get("records").and_then(|v| v.as_array()) else {
            return refuse("legacy brain has no records array");
        };
        if records.len() != brain.n || brain.vecs.len() != brain.n * brain.dim {
            return refuse("legacy brain record/vector count mismatch");
        }

        let mut flat = Vec::with_capacity(brain.n * crate::embedding::SEMANTIC_DIM);
        for record in records {
            let Some(text) = record.get("text").and_then(|v| v.as_str()) else {
                return refuse("legacy record has no text");
            };
            let Some(vector) = crate::embedding::semantic_v1(text) else {
                return refuse("semantic model unavailable");
            };
            if vector.len() != crate::embedding::SEMANTIC_DIM
                || vector.iter().any(|v| !v.is_finite())
            {
                return refuse("semantic re-embedding returned invalid data");
            }
            flat.extend_from_slice(&vector);
        }
        if !shell.cma.hydrate(&flat, brain.n, crate::embedding::SEMANTIC_DIM, meta) {
            return refuse("re-embedded brain failed CMA validation");
        }

        let staging = unique_sidecar(path, "semantic-migration.tmp");
        if cma_flush_to(shell, &staging) != 0 {
            let _ = std::fs::remove_file(&staging);
            return refuse("staged semantic brain write failed; old brain kept");
        }
        let staged = match cma_qtsq_load(&staging) {
            Ok(value)
                if value.n == brain.n
                    && value.dim == crate::embedding::SEMANTIC_DIM
                    && value.vecs.len() == brain.n * crate::embedding::SEMANTIC_DIM
                    && value
                        .meta
                        .as_deref()
                        .map(|m| {
                            manifest_matches(
                                m,
                                crate::embedding::MODEL_ID,
                                crate::embedding::SEMANTIC_DIM,
                            )
                        })
                        .unwrap_or(false) => value,
            _ => {
                let _ = std::fs::remove_file(&staging);
                return refuse("staged semantic brain verification failed; old brain kept");
            }
        };
        let _ = staged;

        let backup = unique_sidecar(path, "hash-v1.bak");
        if std::fs::copy(path, &backup).is_err() {
            let _ = std::fs::remove_file(&staging);
            return refuse("could not preserve legacy brain backup");
        }
        if std::fs::rename(&staging, path).is_err() {
            let _ = std::fs::remove_file(&staging);
            return refuse("could not publish semantic brain; old brain kept");
        }
        eprintln!(
            "[sofuu/cma] migrated legacy hash brain to {} (backup: {})",
            crate::embedding::MODEL_ID,
            backup
        );
        Ok((brain.n, backup))
    }

    /// cma_open — validate dims, build the Rust CMA, hydrate from the QTSQ
    /// brain file (vectors + metadata JSON). None when the open fails.
    unsafe fn cma_open(path: &str, vec_dim: usize, requested_embedding_id: Option<&str>) -> Option<Box<CmaShell>> {
        if vec_dim == 0 || vec_dim > 8192 {
            eprintln!("[sofuu/cma] invalid vector dim {}", vec_dim);
            return None;
        }
        let embedding_id = requested_embedding_id
            .filter(|id| !id.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| {
                if vec_dim == crate::embedding::SEMANTIC_DIM {
                    Some(crate::embedding::MODEL_ID.to_string())
                } else if vec_dim == crate::embedding::HASH_DIM {
                    Some(crate::embedding::INPUT_EMBEDDER_ID.to_string())
                } else {
                    Some(format!("legacy-dim-{vec_dim}"))
                }
            })
            .unwrap();
        if embedding_id == crate::embedding::MODEL_ID && crate::embedding::baked_model().is_err() {
            eprintln!("[sofuu/cma] semantic embedding model is unavailable");
            return None;
        }
        if embedding_id == crate::embedding::semantic_v2::MODEL_ID_V2
            && crate::embedding::semantic_v2::baked_model_v2().is_err()
        {
            eprintln!("[sofuu/cma] SEM2 embedding model is unavailable");
            return None;
        }
        // M1: image space (img1-64, joint with sem2-64 text geometry).
        if embedding_id == crate::embedding::image::MODEL_ID_IMG
            && crate::embedding::image::baked_model_img().is_err()
        {
            eprintln!("[sofuu/cma] IMG1 embedding model is unavailable");
            return None;
        }

        let mut shell = Box::new(CmaShell {
            mind_path: path.to_string(),
            vec_dim,
            embedding_id: embedding_id.clone(),
            cma: crate::memory::cma::Cma::new(vec_dim),
        });

        /* Hydrate from the QTSQ brain file (vectors + metadata JSON). */
        match cma_qtsq_load(path) {
            Ok(brain) => {
                if !brain.vecs.is_empty() && brain.n > 0 {
                    let manifest = brain.meta.as_deref().and_then(embedding_manifest);
                    let source_id = manifest
                        .as_ref()
                        .map(|m| m.id.as_str())
                        .or_else(|| inferred_legacy_embedding_id(brain.dim));
                    let same_space = brain.dim == vec_dim
                        && source_id == Some(embedding_id.as_str())
                        && brain.meta.as_deref().map(|m| {
                            if manifest.is_some() {
                                manifest_matches(m, &embedding_id, vec_dim)
                            } else {
                                embedding_id == crate::embedding::INPUT_EMBEDDER_ID
                            }
                        }).unwrap_or(false);
                    if same_space {
                        if let Some(meta) = &brain.meta {
                            if !shell.cma.hydrate(&brain.vecs, brain.n, brain.dim, meta) {
                                eprintln!("[sofuu/cma] brain file metadata unreadable — open refused");
                                return None;
                            }
                        } else {
                            eprintln!("[sofuu/cma] brain has no metadata — open refused");
                            return None;
                        }
                    } else if embedding_id == crate::embedding::MODEL_ID
                        && brain.dim == crate::embedding::HASH_DIM
                        && source_id == Some(crate::embedding::INPUT_EMBEDDER_ID)
                    {
                        if migrate_legacy_hash_brain(path, &brain, &mut shell).is_err() {
                            return None;
                        }
                    } else {
                        eprintln!(
                            "[sofuu/cma] incompatible brain vector space (file: {} / {:?}, requested: {} / {}) — open refused",
                            brain.dim,
                            source_id,
                            vec_dim,
                            embedding_id
                        );
                        return None;
                    }
                } else if std::path::Path::new(path).exists() && !brain.found_memories {
                    /* No vectors AND no "memories" stream. That is EITHER a
                     * provably EMPTY brain (metadata says zero records) or
                     * a container we cannot read. A brain with no records
                     * has nothing to protect: the old code called it
                     * "empty/corrupt", copied it to a fixed .bak on EVERY
                     * open and printed a false alarm in every fresh
                     * project. Adopt it silently — a legacy manifest id is
                     * refreshed on the next write, and a file that cannot
                     * be parsed at all still takes the backup path. */
                    let provably_empty = brain
                        .meta
                        .as_deref()
                        .and_then(|meta| serde_json::from_str::<serde_json::Value>(meta).ok())
                        .and_then(|root| root.get("records").cloned())
                        .map(|records| records.as_array().map(|a| a.is_empty()).unwrap_or(false))
                        .unwrap_or(false);
                    if !provably_empty {
                        /* Unusable (no memories stream, and we cannot prove
                         * it is empty). Back it up BEFORE the next flush
                         * could silently overwrite it — never wipe a brain
                         * silently, and never clobber an existing .bak. */
                        let bak = backup_brain(path);
                        eprintln!(
                            "[sofuu/cma] brain file {} is unreadable (no memories stream) — \
                             backed up to {} and starting fresh",
                            path, bak
                        );
                    }
                }
            }
            Err(code) => {
                /* A load that FAILED outright (garbage container, wrong
                 * magic, structurally invalid) used to fall through to a
                 * silent fresh shell — the next flush would then overwrite
                 * the user's brain file with no trace. Same rule as the
                 * empty-container branch above: back up, warn, start
                 * fresh — never wipe silently, never break chat. */
                if std::path::Path::new(path).exists() {
                    let bak = backup_brain(path);
                    eprintln!(
                        "[sofuu/cma] brain file {} failed to load (qtsq error {}) — \
                         backed up to {} and starting fresh",
                        path, code, bak
                    );
                }
            }
        }
        Some(shell)
    }

    /// cma_flush — dump vectors + records JSON (with the embedding
    /// manifest, so the next open can verify the vector space), write via
    /// the QTSQ adapter (same stream layout as before — brain files stay
    /// compatible).
    unsafe fn cma_flush_to(shell: &mut CmaShell, path: &str) -> c_int {
        let n = shell.cma.index.len();
        let dim = shell.cma.vec_dim;
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            flat.extend_from_slice(shell.cma.index.vector(i as u32));
        }
        let vecs = if n == 0 { None } else { Some(flat.as_slice()) };
        let meta = metadata_with_embedding(shell);
        let r = cma_qtsq_save(path, vecs, n, dim, &meta);
        if r == QTSQ_OK {
            0
        } else {
            -1
        }
    }

    unsafe fn cma_flush(shell: &mut CmaShell) -> c_int {
        let path = shell.mind_path.clone();
        cma_flush_to(shell, &path)
    }

    unsafe fn cma_remember(
        shell: &mut CmaShell,
        vec: &[f32],
        text: Option<&str>,
        role: Option<&str>,
        kv_page_id: u32,
    ) -> c_int {
        let Some(text) = text else {
            return -1;
        };
        shell.cma.remember(vec, text, role.unwrap_or("unknown"), kv_page_id)
    }

    /// recall → JSON array string (same shape ffi_exports.rs documents:
    /// [{id, distance, score, role, text, tier, strength, entity}] —
    /// `entity` is the namespace for TIER_ENTITY records (agent memory
    /// scopes), null for ordinary memories).
    unsafe fn cma_recall_json(shell: &mut CmaShell, query: &[f32], top_k: usize) -> String {
        let hits = shell.cma.recall(query, top_k);
        let arr: Vec<serde_json::Value> = hits
            .iter()
            .map(|h| {
                serde_json::json!({
                    "id": h.id,
                    "distance": h.distance,
                    "score": h.score,
                    "role": h.role,
                    "text": h.text,
                    "tier": h.tier,
                    "strength": h.strength,
                    "entity": h.entity,
                })
            })
            .collect();
        serde_json::to_string(&arr).unwrap_or_else(|_| "[]".into())
    }

    unsafe fn cma_remember_entity(
        shell: &mut CmaShell,
        vec: &[f32],
        text: Option<&str>,
        entity_name: Option<&str>,
        entity_type: Option<&str>,
    ) -> c_int {
        let (Some(text), Some(name)) = (text, entity_name) else {
            return -1;
        };
        shell
            .cma
            .remember_entity(vec, text, name, entity_type.unwrap_or("unknown"))
    }

    /* Extract a Float32Array arg → borrowed float pointer + element count.
     * Null pointer on error (exception already thrown). */
    unsafe fn cma_vec_arg(ctx: *mut JSContext, v: JSValueConst) -> (*const f32, usize) {
        let mut byte_length: usize = 0;
        let mut byte_offset: usize = 0;
        let ab = qjs::JS_GetTypedArrayBuffer(ctx, v, &mut byte_offset, &mut byte_length, ptr::null_mut());
        if qjs::is_exception(ab) {
            return (ptr::null(), 0);
        }
        let mut ab_size: usize = 0;
        let buf = qjs::JS_GetArrayBuffer(ctx, &mut ab_size, ab);
        qjs::sofuu_js_free_value(ctx, ab);
        if buf.is_null() {
            return (ptr::null(), 0);
        }
        let p = (buf as *const u8).add(byte_offset) as *const f32;
        (p, byte_length / std::mem::size_of::<f32>())
    }

    unsafe extern "C" fn js_cma_open(
        ctx: *mut JSContext,
        _this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 2 {
            return qjs::JS_ThrowTypeError(ctx, c"Expected path and vector dimension".as_ptr());
        }
        let path = qjs::sofuu_js_to_cstring(ctx, *argv);
        let mut dim: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut dim, *argv.add(1));
        // Optional 3rd arg: embedding space id ("hash-v1" |
        // "semantic-projector-v1" | "provider:model@rev" for remote).
        // Absent → inferred from the dimension (legacy behavior).
        let embed_id: Option<String> = if argc >= 3 && qjs::sofuu_js_is_string(*argv.add(2)) != 0 {
            let eptr = qjs::sofuu_js_to_cstring(ctx, *argv.add(2));
            let id = cstr_opt(eptr);
            if !eptr.is_null() {
                qjs::sofuu_js_free_cstring(ctx, eptr);
            }
            id.filter(|s| !s.is_empty())
        } else {
            None
        };

        let shell = cstr_opt(path).and_then(|p| cma_open(&p, dim as usize, embed_id.as_deref()));
        if !path.is_null() {
            qjs::sofuu_js_free_cstring(ctx, path);
        }
        let Some(shell) = shell else {
            return qjs::JS_ThrowInternalError(ctx, c"Failed to initialize CMA".as_ptr());
        };
        let shell = Box::into_raw(shell);

        let obj = qjs::JS_NewObjectClass(ctx, CMA_CLASS_ID.load(Ordering::Relaxed) as c_int);
        qjs::JS_SetOpaque(obj, shell as *mut c_void);
        obj
    }

    /// Mirrors the C's JS_GetOpaque2 + null check → JS_EXCEPTION.
    unsafe fn shell_from_this<'a>(
        ctx: *mut JSContext,
        this_val: JSValueConst,
    ) -> Option<&'a mut CmaShell> {
        let p = qjs::JS_GetOpaque2(ctx, this_val, CMA_CLASS_ID.load(Ordering::Relaxed))
            as *mut CmaShell;
        if p.is_null() {
            None
        } else {
            Some(&mut *p)
        }
    }

    unsafe extern "C" fn js_cma_remember(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 4 {
            return qjs::JS_ThrowTypeError(
                ctx,
                c"Expected vec(Float32Array), text, role, kv_page_id".as_ptr(),
            );
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };

        let (vec, vlen) = cma_vec_arg(ctx, *argv);
        if vec.is_null() {
            return qjs::JS_ThrowTypeError(ctx, c"Argument 1 must be Float32Array".as_ptr());
        }
        if vlen != cma.vec_dim {
            return qjs::JS_ThrowTypeError(ctx, c"Vector dimension mismatch".as_ptr());
        }

        let text = qjs::sofuu_js_to_cstring(ctx, *argv.add(1));
        let role = qjs::sofuu_js_to_cstring(ctx, *argv.add(2));
        let mut page_id: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut page_id, *argv.add(3));

        let idx = cma_remember(
            cma,
            std::slice::from_raw_parts(vec, vlen),
            cstr_opt(text).as_deref(),
            cstr_opt(role).as_deref(),
            page_id,
        );

        if !text.is_null() {
            qjs::sofuu_js_free_cstring(ctx, text);
        }
        if !role.is_null() {
            qjs::sofuu_js_free_cstring(ctx, role);
        }
        qjs::sofuu_js_new_int32(ctx, idx)
    }

    unsafe extern "C" fn js_cma_remember_entity(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 4 {
            return qjs::JS_ThrowTypeError(
                ctx,
                c"Expected vec, text, entityName, entityType".as_ptr(),
            );
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };

        let (vec, vlen) = cma_vec_arg(ctx, *argv);
        if vec.is_null() {
            return qjs::JS_ThrowTypeError(ctx, c"Argument 1 must be Float32Array".as_ptr());
        }
        if vlen != cma.vec_dim {
            return qjs::JS_ThrowTypeError(ctx, c"Vector dimension mismatch".as_ptr());
        }

        let text = qjs::sofuu_js_to_cstring(ctx, *argv.add(1));
        let entity_name = qjs::sofuu_js_to_cstring(ctx, *argv.add(2));
        let entity_type = qjs::sofuu_js_to_cstring(ctx, *argv.add(3));

        let idx = cma_remember_entity(
            cma,
            std::slice::from_raw_parts(vec, vlen),
            cstr_opt(text).as_deref(),
            cstr_opt(entity_name).as_deref(),
            cstr_opt(entity_type).as_deref(),
        );

        if !text.is_null() {
            qjs::sofuu_js_free_cstring(ctx, text);
        }
        if !entity_name.is_null() {
            qjs::sofuu_js_free_cstring(ctx, entity_name);
        }
        if !entity_type.is_null() {
            qjs::sofuu_js_free_cstring(ctx, entity_type);
        }
        qjs::sofuu_js_new_int32(ctx, idx)
    }

    unsafe extern "C" fn js_cma_recall(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 2 {
            return qjs::JS_ThrowTypeError(
                ctx,
                c"Expected queryVec(Float32Array) and topK".as_ptr(),
            );
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };

        let (vec, vlen) = cma_vec_arg(ctx, *argv);
        if vec.is_null() {
            return qjs::JS_ThrowTypeError(ctx, c"Argument 1 must be Float32Array".as_ptr());
        }
        if vlen < cma.vec_dim {
            /* Short vectors would out-of-bounds read in the C — refuse. */
            return qjs::JS_ThrowTypeError(ctx, c"Vector dimension mismatch".as_ptr());
        }

        let mut top_k: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut top_k, *argv.add(1));
        /* Clamp: an unbounded top_k would size the HNSW candidate heap
         * (ef = top_k*3) and OOM on a hostile/accidental huge value. */
        if top_k > 1024 {
            top_k = 1024;
        }

        let json = cma_recall_json(cma, std::slice::from_raw_parts(vec, cma.vec_dim), top_k as usize);
        let cj = CString::new(json).unwrap_or_default();
        let mut arr = qjs::JS_ParseJSON(ctx, cj.as_ptr(), cj.as_bytes().len(), c"<cma-recall>".as_ptr());
        if qjs::is_exception(arr) {
            qjs::sofuu_js_get_exception(ctx);
            arr = qjs::JS_NewArray(ctx);
        }
        arr
    }

    unsafe extern "C" fn js_cma_kv_hints(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 2 {
            return qjs::JS_ThrowTypeError(
                ctx,
                c"Expected queryVec(Float32Array) and n_hints".as_ptr(),
            );
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };

        let (vec, vlen) = cma_vec_arg(ctx, *argv);
        if vec.is_null() {
            return qjs::JS_ThrowTypeError(ctx, c"Argument 1 must be Float32Array".as_ptr());
        }
        if vlen < cma.vec_dim {
            /* Short vectors would out-of-bounds read in the C — refuse. */
            return qjs::JS_ThrowTypeError(ctx, c"Vector dimension mismatch".as_ptr());
        }

        let mut n_hints: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut n_hints, *argv.add(1));
        if n_hints > 1024 {
            n_hints = 1024;
        }

        let ids = cma.cma.kv_hints(std::slice::from_raw_parts(vec, cma.vec_dim), n_hints as usize);
        let arr = qjs::JS_NewArray(ctx);
        for (i, id) in ids.iter().enumerate() {
            qjs::JS_SetPropertyUint32(ctx, arr, i as u32, qjs::sofuu_js_new_uint32(ctx, *id));
        }
        arr
    }

    unsafe extern "C" fn js_cma_mark_positive(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 1 {
            return qjs::JS_ThrowTypeError(ctx, c"Expected Array of ids".as_ptr());
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        if qjs::JS_IsArray(ctx, *argv) == 0 {
            return qjs::JS_ThrowTypeError(ctx, c"Argument must be Array".as_ptr());
        }
        let len_val = qjs::sofuu_js_get_property_str(ctx, *argv, c"length".as_ptr());
        let mut len: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut len, len_val);
        qjs::sofuu_js_free_value(ctx, len_val);

        let mut ids: Vec<u32> = Vec::with_capacity(len as usize);
        for i in 0..len {
            let id_val = qjs::JS_GetPropertyUint32(ctx, *argv, i);
            let mut id: u32 = 0;
            qjs::sofuu_js_to_uint32(ctx, &mut id, id_val);
            qjs::sofuu_js_free_value(ctx, id_val);
            ids.push(id);
        }
        let marked = cma.cma.mark_positive(&ids);
        qjs::sofuu_js_new_uint32(ctx, marked)
    }

    unsafe extern "C" fn js_cma_decay_tick(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 1 {
            return qjs::JS_ThrowTypeError(ctx, c"Expected dt_seconds".as_ptr());
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        // P3 (AUDIT-2026-09-07): parse via float64, not ToUint32 — ToUint32
        // wraps negatives (ToUint32(-1) = 4294967295 "seconds" ≈ 136 years),
        // turning a caller bug into an instant permanent erase of every weak
        // record. Negative / non-finite dt is a no-op; huge positive dt
        // clamps to u32::MAX (full decay) instead of wrapping.
        let mut dt_f: f64 = 0.0;
        // SAFETY: argv[0] is a valid JSValueConst; JS_ToFloat64 writes dt_f
        // on success (non-numeric input throws and leaves dt_f at 0).
        unsafe { qjs::JS_ToFloat64(ctx, &mut dt_f, *argv) };
        let dt = if dt_f.is_finite() && dt_f > 0.0 {
            dt_f.min(u32::MAX as f64) as u32
        } else {
            0
        };
        cma.cma.decay_tick(dt);
        qjs::sofuu_js_undefined()
    }

    /// M3 (PLAN-MEMORY-TOKENS): physical prune of dead records. Returns the
    /// number dropped. Turn-boundary contract documented on Cma::retain.
    unsafe extern "C" fn js_cma_retain(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let dropped = cma.cma.retain();
        qjs::sofuu_js_new_uint32(ctx, dropped as u32)
    }

    unsafe extern "C" fn js_cma_forget(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 1 {
            return qjs::JS_ThrowTypeError(ctx, c"Expected vector_index".as_ptr());
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let mut v_idx: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut v_idx, *argv);
        let ok = cma.cma.forget(v_idx);
        qjs::sofuu_js_new_bool(ctx, if ok { 1 } else { 0 })
    }

    unsafe extern "C" fn js_cma_consolidate(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let n = cma.cma.consolidate() as c_int;
        qjs::sofuu_js_new_int32(ctx, n)
    }

    unsafe extern "C" fn js_cma_count(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        qjs::sofuu_js_new_uint32(ctx, cma.cma.len() as u32)
    }

    unsafe extern "C" fn js_cma_flush(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let r = cma_flush(cma);
        qjs::sofuu_js_new_bool(ctx, if r == 0 { 1 } else { 0 })
    }

    /// sofuu.memory.migrate(path) → JSON string.
    /// Explicit hash→semantic migration (§9.3): reports source/target,
    /// record count, backup path and outcome. Never deletes or blanks the
    /// original on any failure path.
    unsafe extern "C" fn js_memory_migrate(
        ctx: *mut JSContext,
        _this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        fn report(ok: bool, migrated: bool, records: usize, backup: &str, message: &str) -> String {
            serde_json::json!({
                "ok": ok, "migrated": migrated, "records": records,
                "backup": backup, "message": message,
                "source": crate::embedding::INPUT_EMBEDDER_ID,
                "target": crate::embedding::MODEL_ID,
            })
            .to_string()
        }
        if argc < 1 || qjs::sofuu_js_is_string(*argv) == 0 {
            return qjs::JS_ThrowTypeError(ctx, c"memory.migrate(path) expected a string".as_ptr());
        }
        let path_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
        let Some(path) = cstr_opt(path_ptr) else {
            if !path_ptr.is_null() {
                qjs::sofuu_js_free_cstring(ctx, path_ptr);
            }
            return qjs::sofuu_js_exception();
        };
        if !path_ptr.is_null() {
            qjs::sofuu_js_free_cstring(ctx, path_ptr);
        }
        if crate::embedding::baked_model().is_err() {
            let s = report(false, false, 0, "", "semantic model unavailable");
            return qjs::sofuu_js_new_string(ctx, CString::new(s).unwrap_or_default().as_ptr());
        }
        let brain = match cma_qtsq_load(&path) {
            Ok(b) => b,
            Err(_) => {
                let s = report(false, false, 0, "", "brain file unreadable");
                return qjs::sofuu_js_new_string(ctx, CString::new(s).unwrap_or_default().as_ptr());
            }
        };
        if !brain.found_memories || brain.n == 0 {
            let s = report(false, false, 0, "", "brain has no memories to migrate");
            return qjs::sofuu_js_new_string(ctx, CString::new(s).unwrap_or_default().as_ptr());
        }
        let manifest = brain.meta.as_deref().and_then(embedding_manifest);
        let source_id = manifest
            .as_ref()
            .map(|m| m.id.as_str())
            .or_else(|| inferred_legacy_embedding_id(brain.dim));
        if source_id == Some(crate::embedding::MODEL_ID)
            && brain.dim == crate::embedding::SEMANTIC_DIM
            && brain.meta.as_deref().map(|m| manifest_matches(m, crate::embedding::MODEL_ID, crate::embedding::SEMANTIC_DIM)).unwrap_or(false)
        {
            let s = report(true, false, brain.n, "", "already semantic-projector-v1");
            return qjs::sofuu_js_new_string(ctx, CString::new(s).unwrap_or_default().as_ptr());
        }
        if !(brain.dim == crate::embedding::HASH_DIM
            && source_id == Some(crate::embedding::INPUT_EMBEDDER_ID))
        {
            let s = report(false, false, brain.n, "", "unknown or incompatible vector space; preserved untouched");
            return qjs::sofuu_js_new_string(ctx, CString::new(s).unwrap_or_default().as_ptr());
        }
        // Open a scratch shell (NOT the canonical path) so a failed
        // migration can never disturb the live brain handle.
        let mut shell = Box::new(CmaShell {
            mind_path: path.clone(),
            vec_dim: crate::embedding::SEMANTIC_DIM,
            embedding_id: crate::embedding::MODEL_ID.to_string(),
            cma: crate::memory::cma::Cma::new(crate::embedding::SEMANTIC_DIM),
        });
        match migrate_legacy_hash_brain(&path, &brain, &mut shell) {
            Ok((records, backup)) => {
                let s = report(true, true, records, &backup, "migrated; original preserved at backup path");
                qjs::sofuu_js_new_string(ctx, CString::new(s).unwrap_or_default().as_ptr())
            }
            Err(reason) => {
                let s = report(false, false, brain.n, "", &reason);
                qjs::sofuu_js_new_string(ctx, CString::new(s).unwrap_or_default().as_ptr())
            }
        }
    }

    /// sofuu.memory.embeddingInfo() → JSON string describing the compiled
    /// memory backends (ids, dims, artifact). Selection reads this, never
    /// literals.
    unsafe extern "C" fn js_memory_embedding_info(
        ctx: *mut JSContext,
        _this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let json = format!(
            "{{\"hash\":{{\"id\":\"{}\",\"dimension\":{}}},\"semantic\":{}}}",
            crate::embedding::INPUT_EMBEDDER_ID,
            crate::embedding::HASH_DIM,
            crate::embedding::model_info_json(),
        );
        qjs::sofuu_js_new_string(ctx, CString::new(json).unwrap_or_default().as_ptr())
    }

    thread_local! {
        static CMA_PROTO_FUNCS: [qjs::JSCFunctionListEntry; 11] = [
            cfunc_entry(c"remember", 4, js_cma_remember),
            cfunc_entry(c"rememberEntity", 4, js_cma_remember_entity),
            cfunc_entry(c"recall", 2, js_cma_recall),
            cfunc_entry(c"kvHints", 2, js_cma_kv_hints),
            cfunc_entry(c"forget", 1, js_cma_forget),
            cfunc_entry(c"flush", 0, js_cma_flush),
            cfunc_entry(c"decayTick", 1, js_cma_decay_tick),
            cfunc_entry(c"retain", 0, js_cma_retain),
            cfunc_entry(c"consolidate", 0, js_cma_consolidate),
            cfunc_entry(c"count", 0, js_cma_count),
            cfunc_entry(c"markPositive", 1, js_cma_mark_positive),
        ];
        static CMA_MODULE_FUNCS: [qjs::JSCFunctionListEntry; 3] = [
            cfunc_entry(c"open", 2, js_cma_open),
            cfunc_entry(c"migrate", 1, js_memory_migrate),
            cfunc_entry(c"embeddingInfo", 0, js_memory_embedding_info),
        ];
    }

    /* ══════════════════════════════════════════════════════════════
     * KV page store (port of src/memory/mod_kv.c)
     * ══════════════════════════════════════════════════════════════ */

    struct PageSummary {
        page_id: u32,
        k_summary: [f32; KV_SUMMARY_DIM],
        strength: f32,
        created_at: u32,
    }

    impl Default for PageSummary {
        fn default() -> Self {
            Self {
                page_id: 0,
                k_summary: [0.0; KV_SUMMARY_DIM],
                strength: 0.0,
                created_at: 0,
            }
        }
    }

    /// An active decompressed page held in RAM (qtsq_decompress_tensor
    /// output, kept whole). We hold at most `max_active` of these.
    /// (The tensor fields mirror the C's active_kv_page_t: the C only ever
    /// freed them — the resident data itself had no consumer yet either.
    /// #[allow(dead_code)] keeps the mirror faithful without lint noise.)
    #[allow(dead_code)]
    struct ActivePage {
        page_id: u32,
        k: Vec<f32>,
        v: Vec<f32>,
        n_layers: usize,
        n_tokens: usize,
        last_accessed: u32, /* for LRU eviction */
    }

    struct Kvs {
        file_path: String,
        n_layers: usize,
        n_heads: usize,
        head_dim: usize,
        /* Memory capacities / limits (max_size_gb mirrors the C field — the
         * C also stored it without ever reading it back) */
        #[allow(dead_code)]
        max_size_gb: usize,
        k_precision: qtsq::QtsqTensorPrecision,
        v_precision: qtsq::QtsqTensorPrecision,
        /* In-memory index of all pages (very fast to search) */
        pages: Vec<PageSummary>,
        /* In-memory LRU cache of decompressed tensors */
        active: Vec<ActivePage>,
        max_active: usize,
        access_counter: u32,
        ctx: *mut JSContext,
    }

    static KV_CLASS_ID: AtomicU32 = AtomicU32::new(0);

    unsafe extern "C" fn kv_finalizer(_rt: *mut qjs::JSRuntime, val: JSValue) {
        let kv = qjs::JS_GetOpaque(val, KV_CLASS_ID.load(Ordering::Relaxed)) as *mut Kvs;
        if kv.is_null() {
            return;
        }
        /* Do NOT call kv_store_close here: it runs kv_flush, which builds JS
         * objects via kv->ctx — ctx may already be freed when a GC finalizer
         * runs (the exact use-after-free cma_finalizer documents and avoids).
         * Free only Rust resources; JS code must flush() explicitly. */
        drop(Box::from_raw(kv));
    }

    const KV_CLASS_DEF: qjs::JSClassDef = qjs::JSClassDef {
        class_name: c"KVStore".as_ptr(),
        finalizer: Some(kv_finalizer),
        gc_mark: ptr::null_mut(),
        call: ptr::null_mut(),
        exotic: ptr::null_mut(),
    };

    fn ensure_dir(path: &str) {
        if std::fs::metadata(path).is_err() {
            let _ = std::fs::create_dir(path);
        }
    }

    fn now_u32() -> u32 {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        (secs & 0xFFFF_FFFF) as u32
    }

    /// evict_lru_page — drop the least-recently-accessed active page
    /// (excluding `exclude`), shifting the array down (C semantics).
    fn evict_lru_page(kv: &mut Kvs, exclude: Option<usize>) {
        if kv.active.is_empty() {
            return;
        }
        let mut oldest_acc: u32 = u32::MAX;
        let mut oldest_idx: Option<usize> = None;
        for (i, page) in kv.active.iter().enumerate() {
            if exclude == Some(i) {
                continue;
            }
            if page.last_accessed < oldest_acc {
                oldest_acc = page.last_accessed;
                oldest_idx = Some(i);
            }
        }
        if let Some(idx) = oldest_idx {
            kv.active.remove(idx);
        }
    }

    /// Mean-pool the page's K tensor across (layers, heads, tokens) into a
    /// fixed-dim summary vector (the index's cosine-search signature).
    /// The K layout is [layer][head][token][head_dim], and the pooled vector
    /// always ends up L2-normalized so kv_search's cosine is well-defined.
    fn compute_k_summary(
        p: &mut PageSummary,
        k: &[f32],
        n_layers: usize,
        n_heads: usize,
        n_tokens: usize,
        head_dim: usize,
    ) {
        p.k_summary.fill(0.0);
        if k.is_empty() {
            return;
        }

        if n_tokens == 0 {
            p.k_summary[0] = 1.0; /* unit vector on zero length (norm stays valid) */
            return;
        }

        /* Mean-pool across COMPONENTS: the K layout is
         * [layer][head][token][head_dim]; each page's summary is the
         * average of all its head_dim-wide slices, so the index signature
         * is a distributed vector (matches the C-era index.json files).
         * (The first port pooled into a single slot (t % 64) which made
         * every summary degenerate to [1,0,0,...] after normalization —
         * all pages tied at rank 1.) */
        let pool = (n_layers.wrapping_mul(n_heads).wrapping_mul(n_tokens)).max(1);
        let mut sums = [0.0f32; KV_SUMMARY_DIM];
        let stride = head_dim.max(1);
        for l in 0..n_layers {
            for h in 0..n_heads {
                for t in 0..n_tokens {
                    let base = ((l * n_heads + h) * n_tokens + t) * stride;
                    if base + head_dim > k.len() {
                        continue; /* defensive — C read OOB here */
                    }
                    for j in 0..head_dim {
                        let slot = j % KV_SUMMARY_DIM;
                        sums[slot] += k[base + j] / pool as f32;
                    }
                }
            }
        }
        for (dst, src) in p.k_summary.iter_mut().zip(sums.iter()) {
            *dst = *src;
        }

        /* Normalize the pooled vector (guards empty pages + zero-norm
         * degenerates) */
        let mut norm = 0.0f32;
        for i in 0..KV_SUMMARY_DIM {
            norm += p.k_summary[i] * p.k_summary[i];
        }
        if norm > 0.0 {
            let inv = 1.0f32 / norm.sqrt();
            for v in p.k_summary.iter_mut() {
                *v *= inv;
            }
        } else {
            p.k_summary[0] = 1.0;
        }
    }

    /// Serialize one page's k_summary as a JSON float array using C `%g`
    /// formatting (via snprintf) so index.json stays byte-identical to the
    /// files the retired C kv_flush wrote.
    fn json_floats(vals: &[f32]) -> Option<Vec<u8>> {
        let mut out: Vec<u8> = Vec::with_capacity(vals.len() * 16 + 8);
        out.push(b'[');
        for (i, v) in vals.iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            let mut buf = [0u8; 64];
            // SAFETY: snprintf writes at most 63 bytes + NUL into buf.
            let n = unsafe {
                libc::snprintf(
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len(),
                    c"%g".as_ptr(),
                    *v as f64,
                )
            };
            if n <= 0 {
                return None;
            }
            out.extend_from_slice(&buf[..n as usize]);
        }
        out.push(b']');
        Some(out)
    }

    /// Parse a JSON float array (from the persisted index) back into `out`.
    /// Returns 0 on success, -1 when missing/malformed (C parse_float_array).
    unsafe fn parse_float_array(
        ctx: *mut JSContext,
        item: JSValueConst,
        key: &CStr,
        out: &mut [f32],
    ) -> c_int {
        let arr = qjs::sofuu_js_get_property_str(ctx, item, key.as_ptr());
        if qjs::JS_IsArray(ctx, arr) == 0 {
            qjs::sofuu_js_free_value(ctx, arr);
            return -1;
        }
        let mut len: u32 = 0;
        let lv = qjs::sofuu_js_get_property_str(ctx, arr, c"length".as_ptr());
        qjs::sofuu_js_to_uint32(ctx, &mut len, lv);
        qjs::sofuu_js_free_value(ctx, lv);
        let take = (len as usize).min(out.len());
        for i in 0..take {
            let el = qjs::JS_GetPropertyUint32(ctx, arr, i as u32);
            let mut d: f64 = 0.0;
            qjs::JS_ToFloat64(ctx, &mut d, el);
            out[i] = d as f32;
            qjs::sofuu_js_free_value(ctx, el);
        }
        if take < out.len() {
            out[take..].fill(0.0);
        }
        qjs::sofuu_js_free_value(ctx, arr);
        0
    }

    /// kv_store_open — load the index if it exists (JS-parsed, exactly like
    /// the C), else start with an empty summary index.
    unsafe fn kv_store_open(
        ctx: *mut JSContext,
        path: &str,
        n_layers: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Option<Box<Kvs>> {
        ensure_dir(path);
        // P3 (AUDIT-2026-09-07): crashed kv writers left `page_NNNN.qtsq.tmp`
        // siblings behind forever — sweep them once per open (age-guarded).
        sweep_stale_tmp_dir(path);

        let mut kv = Box::new(Kvs {
            file_path: path.to_string(),
            n_layers,
            n_heads,
            head_dim,
            max_size_gb: 256,
            k_precision: qtsq::QTSQ_TENSOR_F8,
            v_precision: qtsq::QTSQ_TENSOR_F16,
            pages: Vec::new(),
            active: Vec::new(),
            max_active: 2, /* Extremely strict by default: hold at most 2 decompressed pages to protect RAM */
            access_counter: 1,
            ctx,
        });

        let idx_path = format!("{}/index.json", path);
        if let Ok(data) = std::fs::read(&idx_path) {
            if !data.is_empty() {
                let mut json_buf = data.to_vec();
                json_buf.push(0);
                let meta_obj = qjs::JS_ParseJSON(
                    ctx,
                    json_buf.as_ptr() as *const c_char,
                    data.len(),
                    c"index.json".as_ptr(),
                );
                if !qjs::is_exception(meta_obj) {
                    let pages_arr = qjs::sofuu_js_get_property_str(ctx, meta_obj, c"pages".as_ptr());
                    if qjs::JS_IsArray(ctx, pages_arr) != 0 {
                        let mut arr_len: u32 = 0;
                        let len_val = qjs::sofuu_js_get_property_str(ctx, pages_arr, c"length".as_ptr());
                        qjs::sofuu_js_to_uint32(ctx, &mut arr_len, len_val);
                        qjs::sofuu_js_free_value(ctx, len_val);

                        /* ml-1 (AUDIT-2026-09-07): arr_len is the parsed
                         * array's real length (JS_IsArray-gated), so the
                         * loop below is item-bounded — but clamp the upfront
                         * reservation so a giant-but-well-formed index (or a
                         * future parser change) cannot turn reserve() into
                         * the amplifier. Vec growth covers the true count. */
                        const KV_PAGES_RESERVE_CAP: u32 = 1 << 16;
                        kv.pages.reserve(arr_len.min(KV_PAGES_RESERVE_CAP) as usize + 256);

                        for i in 0..arr_len {
                            let item = qjs::JS_GetPropertyUint32(ctx, pages_arr, i);
                            let mut p = PageSummary::default();

                            let v_id = qjs::sofuu_js_get_property_str(ctx, item, c"page_id".as_ptr());
                            qjs::sofuu_js_to_uint32(ctx, &mut p.page_id, v_id);
                            qjs::sofuu_js_free_value(ctx, v_id);

                            let v_str = qjs::sofuu_js_get_property_str(ctx, item, c"strength".as_ptr());
                            let mut st: f64 = 1.0;
                            qjs::JS_ToFloat64(ctx, &mut st, v_str);
                            p.strength = st as f32;
                            qjs::sofuu_js_free_value(ctx, v_str);

                            let v_time = qjs::sofuu_js_get_property_str(ctx, item, c"created_at".as_ptr());
                            qjs::sofuu_js_to_uint32(ctx, &mut p.created_at, v_time);
                            qjs::sofuu_js_free_value(ctx, v_time);

                            /* Restore the persisted 64-dim K summary. Older
                             * index files (or a zeroed entry) leave it
                             * zeroed, exactly as before — such pages just
                             * rank as neutral. */
                            if parse_float_array(ctx, item, c"k_summary", &mut p.k_summary) != 0 {
                                p.k_summary.fill(0.0);
                            }

                            qjs::sofuu_js_free_value(ctx, item);
                            kv.pages.push(p);
                        }
                    }
                    qjs::sofuu_js_free_value(ctx, pages_arr);
                }
                qjs::sofuu_js_free_value(ctx, meta_obj);
            }
        }
        Some(kv)
    }

    /// kv_page_save — compress and append a page; updates index K-summary.
    /// Returns the new page id (0 on failure).
    unsafe fn kv_page_save(kv: &mut Kvs, k: &[f32], v: &[f32], n_tokens: usize, strength: f32) -> u32 {
        let count = kv
            .n_layers
            .wrapping_mul(kv.n_heads)
            .wrapping_mul(n_tokens)
            .wrapping_mul(kv.head_dim);
        /* The C memcpy'd `count` floats from the caller's buffers without
         * checking their length — refuse short arrays (OOB-read fix). */
        if k.len() < count || v.len() < count {
            return 0;
        }

        let new_id = kv.pages.len() as u32 + 1;
        let page_path = format!("{}/page_{:08}.qtsq", kv.file_path, new_id);

        /* Create a mini container just for this single KV page to keep file
         * sizes small and GC easy. C ignored qtsq_init/container_create rc —
         * a failure surfaces via the save step below, same net result. */
        let container = qtsq::qtsq_ctx_alloc();
        if container.is_null() {
            return 0;
        }
        qtsq::qtsq_init(container);
        let _ = qtsq::qtsq_container_create(container);
        /* Phase 7: qtc write codec — see cma_qtsq_save's comment. */
        unsafe { (*container).set_codec(qtsq::QTSQ_CODEC_QTC) };

        let mut r = kv_qtsq_save_page(
            container,
            k.as_ptr(),
            v.as_ptr(),
            kv.n_layers,
            kv.n_heads,
            n_tokens,
            kv.head_dim,
            new_id,
            kv.k_precision,
            kv.v_precision,
        );
        if r == QTSQ_OK {
            r = kv_qtsq_flush(container, &page_path);
        }
        qtsq::qtsq_free(container);
        qtsq::qtsq_ctx_free(container);

        if r != QTSQ_OK {
            // P3 (AUDIT-2026-09-07): a failed flush can leave a partial
            // page_{id}.qtsq behind. The next save reuses this id (pages.len()
            // never advanced) and would overwrite it, but until then the file
            // on disk is a corrupt page for anything that reads by path —
            // remove it so disk never disagrees with the in-memory index.
            let _ = std::fs::remove_file(&page_path);
            return 0;
        }

        let mut p = PageSummary {
            page_id: new_id,
            strength,
            created_at: now_u32(),
            k_summary: [0.0; KV_SUMMARY_DIM],
        };
        compute_k_summary(&mut p, k, kv.n_layers, kv.n_heads, n_tokens, kv.head_dim);
        kv.pages.push(p);

        /* Also load it into RAM cache immediately since it was just generated */
        if kv.active.len() >= kv.max_active {
            evict_lru_page(kv, None);
        }
        kv.active.push(ActivePage {
            page_id: new_id,
            k: k[..count].to_vec(),
            v: v[..count].to_vec(),
            n_layers: kv.n_layers,
            n_tokens,
            last_accessed: kv.access_counter,
        });
        kv.access_counter = kv.access_counter.wrapping_add(1);

        new_id
    }

    /// kv_get_page — find in the LRU cache, else load from SSD. Returns
    /// true when the page is (now) resident.
    unsafe fn kv_get_page(kv: &mut Kvs, page_id: u32) -> bool {
        if page_id == 0 {
            return false;
        }

        /* Check cache */
        for page in kv.active.iter_mut() {
            if page.page_id == page_id {
                page.last_accessed = kv.access_counter;
                kv.access_counter = kv.access_counter.wrapping_add(1);
                return true;
            }
        }

        /* Load from SSD */
        let page_path = format!("{}/page_{:08}.qtsq", kv.file_path, page_id);
        let Ok((k, v, l, tok)) = kv_qtsq_load_page(&page_path, page_id) else {
            return false; /* Error or page missing */
        };

        if kv.active.len() >= kv.max_active {
            evict_lru_page(kv, None);
        }
        kv.active.push(ActivePage {
            page_id,
            k,
            v,
            n_layers: l,
            n_tokens: tok,
            last_accessed: kv.access_counter,
        });
        kv.access_counter = kv.access_counter.wrapping_add(1);

        true
    }

    /// kv_search — naive linear scan over the small summary index (64-float
    /// signatures, thousands scanned in <1ms with the SIMD cosine kernel).
    fn kv_search(kv: &mut Kvs, query_summary_64: &[f32], n_hints: usize) -> Vec<u32> {
        if kv.pages.is_empty() || query_summary_64.len() < KV_SUMMARY_DIM || n_hints == 0 {
            return Vec::new();
        }

        let mut matches: Vec<(u32, f32)> = Vec::with_capacity(kv.pages.len());
        for p in &kv.pages {
            // SAFETY: the SIMD kernel reads exactly 64 floats from both sides.
            let mut sim = unsafe { sofuu_cosine_f32(query_summary_64.as_ptr(), p.k_summary.as_ptr(), KV_SUMMARY_DIM) };
            /* Apply decay/strength weighting */
            sim *= p.strength;
            matches.push((p.page_id, sim));
        }

        /* Sort matches, simplest way for small counts is insertion for top N */
        let mut top_ids: Vec<u32> = Vec::with_capacity(n_hints);
        for _k in 0..n_hints.min(kv.pages.len()) {
            let mut best_sim: f32 = -2.0;
            let mut best_idx: usize = 0;
            for (i, m) in matches.iter().enumerate() {
                if m.1 > best_sim {
                    best_sim = m.1;
                    best_idx = i;
                }
            }
            if best_sim == -2.0 {
                break;
            }
            top_ids.push(matches[best_idx].0);
            matches[best_idx].1 = -3.0; /* Exclude from next pass */
        }
        top_ids
    }

    /// kv_flush — write the summary index to index.json (JS-stringified,
    /// same object shape + %g numbers as the C).
    unsafe fn kv_flush(kv: &mut Kvs) -> c_int {
        let idx_path = format!("{}/index.json", kv.file_path);

        let root = qjs::sofuu_js_new_object(kv.ctx);
        let arr = qjs::JS_NewArray(kv.ctx);

        for (i, p) in kv.pages.iter().enumerate() {
            let item = qjs::sofuu_js_new_object(kv.ctx);
            qjs::sofuu_js_set_property_str(
                kv.ctx,
                item,
                c"page_id".as_ptr(),
                qjs::sofuu_js_new_uint32(kv.ctx, p.page_id),
            );
            qjs::sofuu_js_set_property_str(
                kv.ctx,
                item,
                c"strength".as_ptr(),
                qjs::sofuu_js_new_float64(kv.ctx, p.strength as f64),
            );
            qjs::sofuu_js_set_property_str(
                kv.ctx,
                item,
                c"created_at".as_ptr(),
                qjs::sofuu_js_new_uint32(kv.ctx, p.created_at),
            );
            if let Some(j) = json_floats(&p.k_summary) {
                let cj = CString::new(j).unwrap_or_default();
                let jv = qjs::JS_ParseJSON(kv.ctx, cj.as_ptr(), cj.as_bytes().len(), c"<k_summary>".as_ptr());
                if !qjs::is_exception(jv) {
                    qjs::sofuu_js_set_property_str(kv.ctx, item, c"k_summary".as_ptr(), jv);
                } else {
                    qjs::sofuu_js_get_exception(kv.ctx); /* clear; persist page without summary */
                    qjs::sofuu_js_free_value(kv.ctx, jv);
                }
            }
            qjs::JS_SetPropertyUint32(kv.ctx, arr, i as u32, item);
        }
        qjs::sofuu_js_set_property_str(kv.ctx, root, c"pages".as_ptr(), arr);

        let str_val = qjs::JS_JSONStringify(kv.ctx, root, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined());
        let json_c = qjs::sofuu_js_to_cstring(kv.ctx, str_val);
        if json_c.is_null() {
            /* Stringify threw — free what we hold and report failure instead
             * of writing nothing (mirrors the C null-check). */
            qjs::sofuu_js_free_value(kv.ctx, str_val);
            qjs::sofuu_js_free_value(kv.ctx, root);
            return -1;
        }

        let bytes = CStr::from_ptr(json_c).to_bytes();
        let _ = std::fs::write(&idx_path, bytes);

        qjs::sofuu_js_free_cstring(kv.ctx, json_c);
        qjs::sofuu_js_free_value(kv.ctx, str_val);
        qjs::sofuu_js_free_value(kv.ctx, root);

        0
    }

    unsafe fn kv_from_this<'a>(ctx: *mut JSContext, this_val: JSValueConst) -> Option<&'a mut Kvs> {
        let p = qjs::JS_GetOpaque2(ctx, this_val, KV_CLASS_ID.load(Ordering::Relaxed)) as *mut Kvs;
        if p.is_null() {
            None
        } else {
            Some(&mut *p)
        }
    }

    unsafe extern "C" fn js_kv_open(
        ctx: *mut JSContext,
        _this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 2 {
            return qjs::JS_ThrowTypeError(ctx, c"Expected path and config object".as_ptr());
        }
        let path = qjs::sofuu_js_to_cstring(ctx, *argv);
        let cfg = *argv.add(1);

        let mut n_l: u32 = 0;
        let mut n_h: u32 = 0;
        let mut h_d: u32 = 0;
        let v_nl = qjs::sofuu_js_get_property_str(ctx, cfg, c"nLayers".as_ptr());
        qjs::sofuu_js_to_uint32(ctx, &mut n_l, v_nl);
        qjs::sofuu_js_free_value(ctx, v_nl);
        let v_nh = qjs::sofuu_js_get_property_str(ctx, cfg, c"nHeads".as_ptr());
        qjs::sofuu_js_to_uint32(ctx, &mut n_h, v_nh);
        qjs::sofuu_js_free_value(ctx, v_nh);
        let v_hd = qjs::sofuu_js_get_property_str(ctx, cfg, c"headDim".as_ptr());
        qjs::sofuu_js_to_uint32(ctx, &mut h_d, v_hd);
        qjs::sofuu_js_free_value(ctx, v_hd);

        let kv = cstr_opt(path).and_then(|p| {
            kv_store_open(ctx, &p, n_l as usize, n_h as usize, h_d as usize)
        });

        if !path.is_null() {
            qjs::sofuu_js_free_cstring(ctx, path);
        }

        let Some(kv) = kv else {
            return qjs::JS_ThrowInternalError(ctx, c"Failed to initialize KV".as_ptr());
        };
        let kv = Box::into_raw(kv);

        let obj = qjs::JS_NewObjectClass(ctx, KV_CLASS_ID.load(Ordering::Relaxed) as c_int);
        qjs::JS_SetOpaque(obj, kv as *mut c_void);
        obj
    }

    unsafe extern "C" fn js_kv_save(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 3 {
            return qjs::JS_ThrowTypeError(
                ctx,
                c"Expected K(Float32Array), V(Float32Array), n_tokens".as_ptr(),
            );
        }
        let Some(kv) = kv_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };

        let (k_ptr, k_len) = cma_vec_arg(ctx, *argv);
        if k_ptr.is_null() {
            return qjs::JS_ThrowTypeError(ctx, c"K must be Float32Array".as_ptr());
        }
        let (v_ptr, v_len) = cma_vec_arg(ctx, *argv.add(1));
        if v_ptr.is_null() {
            return qjs::JS_ThrowTypeError(ctx, c"V must be Float32Array".as_ptr());
        }

        let mut n_tok: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut n_tok, *argv.add(2));

        let k = std::slice::from_raw_parts(k_ptr, k_len);
        let v = std::slice::from_raw_parts(v_ptr, v_len);
        let pid = kv_page_save(kv, k, v, n_tok as usize, 1.0);
        qjs::sofuu_js_new_uint32(ctx, pid)
    }

    unsafe extern "C" fn js_kv_search(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 2 {
            return qjs::JS_ThrowTypeError(
                ctx,
                c"Expected query(Float32Array[64]) and n_hints".as_ptr(),
            );
        }
        let Some(kv) = kv_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };

        let (q_ptr, q_len) = cma_vec_arg(ctx, *argv);
        if q_ptr.is_null() {
            return qjs::JS_ThrowTypeError(ctx, c"Query must be Float32Array".as_ptr());
        }

        let mut n_hints: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut n_hints, *argv.add(1));
        if n_hints > 1024 {
            /* P1-6: same clamp as the CMA hints path — kv_search reserves
             * with_capacity(n_hints), so an unclamped count is a 17GB OOM. */
            n_hints = 1024;
        }

        let query = std::slice::from_raw_parts(q_ptr, q_len);
        let hints = kv_search(kv, query, n_hints as usize);

        let arr = qjs::JS_NewArray(ctx);
        for (i, id) in hints.iter().enumerate() {
            qjs::JS_SetPropertyUint32(ctx, arr, i as u32, qjs::sofuu_js_new_uint32(ctx, *id));
        }
        arr
    }

    unsafe extern "C" fn js_kv_flush(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(kv) = kv_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let r = kv_flush(kv);
        qjs::sofuu_js_new_int32(ctx, r)
    }

    thread_local! {
        static KV_PROTO_FUNCS: [qjs::JSCFunctionListEntry; 3] = [
            cfunc_entry(c"save", 3, js_kv_save),
            cfunc_entry(c"search", 2, js_kv_search),
            cfunc_entry(c"flush", 0, js_kv_flush),
        ];
        static KV_MODULE_FUNCS: [qjs::JSCFunctionListEntry; 1] =
            [cfunc_entry(c"open", 2, js_kv_open)];
    }

    /* ══════════════════════════════════════════════════════════════
     * Agent façade (port of src/memory/mod_agent.c)
     * ══════════════════════════════════════════════════════════════ */

    struct Agent {
        cma: *mut CmaShell, /* Cognitive Memory Architecture reference */
        kv: *mut Kvs,       /* SSD KV cache reference */
    }

    static AGENT_CLASS_ID: AtomicU32 = AtomicU32::new(0);

    unsafe extern "C" fn agent_finalizer(_rt: *mut qjs::JSRuntime, val: JSValue) {
        let agent = qjs::JS_GetOpaque(val, AGENT_CLASS_ID.load(Ordering::Relaxed)) as *mut Agent;
        if !agent.is_null() {
            /* agent_destroy — only the Agent itself; the underlying CMA/KV
             * stay owned by their own JS objects. */
            drop(Box::from_raw(agent));
        }
    }

    const AGENT_CLASS_DEF: qjs::JSClassDef = qjs::JSClassDef {
        class_name: c"Agent".as_ptr(),
        finalizer: Some(agent_finalizer),
        gc_mark: ptr::null_mut(),
        call: ptr::null_mut(),
        exotic: ptr::null_mut(),
    };

    /// agent_prefetch_context — the CMA finds the top-n nearest memories,
    /// returns their distinct kv_page_ids; each page is pulled into the KV
    /// RAM LRU cache. Returns the number of pages prefetched.
    unsafe fn agent_prefetch_context(agent: *mut Agent, query: &[f32], n_memories: usize) -> c_int {
        if agent.is_null() || (*agent).cma.is_null() || (*agent).kv.is_null() {
            return -1;
        }
        let cma = &mut *(*agent).cma;
        if query.len() < cma.vec_dim {
            return 0; /* short vectors would out-of-bounds read in the C */
        }
        let hints = cma.cma.kv_hints(&query[..cma.vec_dim], n_memories);
        let mut prefetched: u32 = 0;
        for &id in &hints {
            let kv = &mut *(*agent).kv;
            if kv_get_page(kv, id) {
                prefetched += 1;
            }
        }
        prefetched as c_int
    }

    unsafe extern "C" fn js_agent_create(
        ctx: *mut JSContext,
        _this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 3 {
            return qjs::JS_ThrowTypeError(ctx, c"Expected name, cma_obj, kv_obj".as_ptr());
        }
        let name = qjs::sofuu_js_to_cstring(ctx, *argv);

        /* The C grabbed the opaques blindly (class ids are module globals —
         * js_agent_create trusts the JS layer to pass CMA/KV objects). */
        let cma = qjs::JS_GetOpaque(*argv.add(1), CMA_CLASS_ID.load(Ordering::Relaxed)) as *mut CmaShell;
        let kv = qjs::JS_GetOpaque(*argv.add(2), KV_CLASS_ID.load(Ordering::Relaxed)) as *mut Kvs;

        let agent = Box::new(Agent { cma, kv });
        if !name.is_null() {
            qjs::sofuu_js_free_cstring(ctx, name);
        }

        let agent = Box::into_raw(agent);
        let obj = qjs::JS_NewObjectClass(ctx, AGENT_CLASS_ID.load(Ordering::Relaxed) as c_int);
        qjs::JS_SetOpaque(obj, agent as *mut c_void);

        /* Attach Cma/KV objects so they don't get GC'd */
        let prop_cma = qjs::sofuu_js_dup_value(ctx, *argv.add(1));
        let prop_kv = qjs::sofuu_js_dup_value(ctx, *argv.add(2));
        qjs::sofuu_js_set_property_str(ctx, obj, c"memory".as_ptr(), prop_cma);
        qjs::sofuu_js_set_property_str(ctx, obj, c"kv".as_ptr(), prop_kv);

        obj
    }

    unsafe extern "C" fn js_agent_prefetch(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 2 {
            return qjs::JS_ThrowTypeError(ctx, c"Expected query(Float32Array), topK".as_ptr());
        }
        let agent = qjs::JS_GetOpaque2(ctx, this_val, AGENT_CLASS_ID.load(Ordering::Relaxed))
            as *mut Agent;
        if agent.is_null() {
            return qjs::sofuu_js_exception();
        }

        let (q_ptr, q_len) = cma_vec_arg(ctx, *argv);
        if q_ptr.is_null() {
            return qjs::JS_ThrowTypeError(ctx, c"Query must be Float32Array".as_ptr());
        }

        let mut top_k: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut top_k, *argv.add(1));
        if top_k > 1024 {
            top_k = 1024;
        }

        let query = std::slice::from_raw_parts(q_ptr, q_len);
        let n = agent_prefetch_context(agent, query, top_k as usize);
        qjs::sofuu_js_new_int32(ctx, n)
    }

    thread_local! {
        static AGENT_PROTO_FUNCS: [qjs::JSCFunctionListEntry; 1] =
            [cfunc_entry(c"prefetch", 2, js_agent_prefetch)];
        static AGENT_MODULE_FUNCS: [qjs::JSCFunctionListEntry; 1] =
            [cfunc_entry(c"create", 3, js_agent_create)];
    }

    /* ══════════════════════════════════════════════════════════════
     * Registration (port of the three mod_*_register functions)
     * ══════════════════════════════════════════════════════════════ */

    /// JS_CFUNC_DEF(name, length, func) entry — magic=0, cproto=0
    /// (JSCFunctionListEntry is 32 bytes: name, prop_flags, def_type,
    /// magic, u:{length, cproto, _pad[6], cfunc}).
    const fn cfunc_entry(
        name: &'static std::ffi::CStr,
        length: u8,
        cfunc: qjs::JSCFunction,
    ) -> qjs::JSCFunctionListEntry {
        qjs::JSCFunctionListEntry {
            name: name.as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc {
                length,
                cproto: 0,
                _pad: [0; 6],
                cfunc,
            },
        }
    }

    /// Get (or create) the global `sofuu` namespace object (the C register
    /// functions each repeat this dance inline).
    unsafe fn sofuu_namespace(ctx: *mut JSContext) -> JSValue {
        let global_obj = qjs::sofuu_js_get_global_object(ctx);
        let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global_obj, c"sofuu".as_ptr());
        if qjs::is_undefined(sofuu_obj) {
            qjs::sofuu_js_free_value(ctx, sofuu_obj); /* undefined — no-op (C leaked it) */
            let fresh = qjs::sofuu_js_new_object(ctx);
            qjs::sofuu_js_set_property_str(ctx, global_obj, c"sofuu".as_ptr(), fresh);
            sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global_obj, c"sofuu".as_ptr());
        }
        qjs::sofuu_js_free_value(ctx, global_obj);
        sofuu_obj
    }

    unsafe fn register_class(
        ctx: *mut JSContext,
        class_def: &'static qjs::JSClassDef,
        class_id: &AtomicU32,
        proto_funcs: *const qjs::JSCFunctionListEntry,
        proto_len: c_int,
    ) {
        /* JS_NewClassID allocates only when *pclass_id == 0 — pass a fresh
         * zero local and store the assigned id (C kept a zeroed global). */
        let mut id: u32 = 0;
        qjs::JS_NewClassID(&mut id);
        class_id.store(id, Ordering::Relaxed);
        let rt = qjs::JS_GetRuntime(ctx);
        qjs::JS_NewClass(rt, id, class_def);
        let proto = qjs::sofuu_js_new_object(ctx);
        qjs::JS_SetPropertyFunctionList(ctx, proto, proto_funcs, proto_len);
        qjs::JS_SetClassProto(ctx, id, proto);
    }

    /// mod_memory_register — sofuu.memory.open + the CMA class.
    pub(crate) unsafe fn memory_register(ctx: *mut JSContext) {
        register_class(
            ctx,
            &CMA_CLASS_DEF,
            &CMA_CLASS_ID,
            CMA_PROTO_FUNCS.with(|t| t.as_ptr()),
            CMA_PROTO_FUNCS.with(|t| t.len() as c_int),
        );
        let sofuu_obj = sofuu_namespace(ctx);
        let mem_obj = qjs::sofuu_js_new_object(ctx);
        qjs::JS_SetPropertyFunctionList(
            ctx,
            mem_obj,
            CMA_MODULE_FUNCS.with(|t| t.as_ptr()),
            CMA_MODULE_FUNCS.with(|t| t.len() as c_int),
        );
        qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"memory".as_ptr(), mem_obj);
        qjs::sofuu_js_free_value(ctx, sofuu_obj);
    }

    /// mod_kv_register — sofuu.kv.open + the KVStore class.
    pub(crate) unsafe fn kv_register(ctx: *mut JSContext) {
        register_class(
            ctx,
            &KV_CLASS_DEF,
            &KV_CLASS_ID,
            KV_PROTO_FUNCS.with(|t| t.as_ptr()),
            KV_PROTO_FUNCS.with(|t| t.len() as c_int),
        );
        let sofuu_obj = sofuu_namespace(ctx);
        let kv_mod = qjs::sofuu_js_new_object(ctx);
        qjs::JS_SetPropertyFunctionList(
            ctx,
            kv_mod,
            KV_MODULE_FUNCS.with(|t| t.as_ptr()),
            KV_MODULE_FUNCS.with(|t| t.len() as c_int),
        );
        qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"kv".as_ptr(), kv_mod);
        qjs::sofuu_js_free_value(ctx, sofuu_obj);
    }

    /// mod_agent_register — sofuu.agent.create + the Agent class.
    pub(crate) unsafe fn agent_register(ctx: *mut JSContext) {
        register_class(
            ctx,
            &AGENT_CLASS_DEF,
            &AGENT_CLASS_ID,
            AGENT_PROTO_FUNCS.with(|t| t.as_ptr()),
            AGENT_PROTO_FUNCS.with(|t| t.len() as c_int),
        );
        let sofuu_obj = sofuu_namespace(ctx);
        let agent_mod = qjs::sofuu_js_new_object(ctx);
        qjs::JS_SetPropertyFunctionList(
            ctx,
            agent_mod,
            AGENT_MODULE_FUNCS.with(|t| t.as_ptr()),
            AGENT_MODULE_FUNCS.with(|t| t.len() as c_int),
        );
        qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"agent".as_ptr(), agent_mod);
        qjs::sofuu_js_free_value(ctx, sofuu_obj);
    }

    /* ── pure-Rust unit tests (no JS context needed) ──────────────── */

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn json_floats_matches_c_g() {
            // Spot-check C "%g" output (6 significant digits, trimmed):
            // 0.5 -> "0.5", 1.0 -> "1", 0.1f32 -> "0.1", 0.30000001 -> "0.3",
            // 123456.0 -> "123456", 1e7 -> "1e+07", 1e-5 -> "1e-05".
            let cases: [(&[f32], &str); 6] = [
                (&[0.5], "[0.5]"),
                (&[1.0, 2.0], "[1,2]"),
                (&[0.1], "[0.1]"),
                (&[0.30000001], "[0.3]"),
                (&[123456.0], "[123456]"),
                (&[1e7, 1e-5], "[1e+07,1e-05]"),
            ];
            for (vals, expect) in cases {
                let out = json_floats(vals).unwrap();
                assert_eq!(String::from_utf8_lossy(&out), expect);
            }
        }

        #[test]
        fn compute_k_summary_normalizes() {
            // 1 layer, 1 head, 1 token, dim 4 → the row is mean-pooled
            // ACROSS the 4 head_dim slots (the M9 fix — the first port
            // pooled every component into slot 0, degenerating every page
            // to [1,0,0,…]), then L2-normalized.
            let k = [1.0f32, 1.0, 1.0, 3.0]; // sums [1,1,1,3], norm √12
            let mut p = PageSummary::default();
            compute_k_summary(&mut p, &k, 1, 1, 1, 4);
            let inv = 1.0f32 / 12.0f32.sqrt();
            assert!((p.k_summary[0] - inv).abs() < 1e-6);
            assert!((p.k_summary[1] - inv).abs() < 1e-6);
            assert!((p.k_summary[2] - inv).abs() < 1e-6);
            assert!((p.k_summary[3] - 3.0 * inv).abs() < 1e-6);
            assert!(p.k_summary.iter().skip(4).all(|v| *v == 0.0));
            // zero-norm degenerate → unit on first slot
            let mut p2 = PageSummary::default();
            compute_k_summary(&mut p2, &[0.0f32; 4], 1, 1, 1, 4);
            assert_eq!(p2.k_summary[0], 1.0);
        }

        #[test]
        fn evict_lru_drops_oldest() {
            let mut kv = Kvs {
                file_path: String::new(),
                n_layers: 1,
                n_heads: 1,
                head_dim: 1,
                max_size_gb: 256,
                k_precision: qtsq::QTSQ_TENSOR_F8,
                v_precision: qtsq::QTSQ_TENSOR_F16,
                pages: Vec::new(),
                active: vec![
                    ActivePage { page_id: 1, k: vec![], v: vec![], n_layers: 1, n_tokens: 1, last_accessed: 5 },
                    ActivePage { page_id: 2, k: vec![], v: vec![], n_layers: 1, n_tokens: 1, last_accessed: 2 },
                    ActivePage { page_id: 3, k: vec![], v: vec![], n_layers: 1, n_tokens: 1, last_accessed: 9 },
                ],
                max_active: 2,
                access_counter: 10,
                ctx: ptr::null_mut(),
            };
            evict_lru_page(&mut kv, None);
            assert_eq!(kv.active.len(), 2);
            assert!(kv.active.iter().all(|p| p.page_id != 2));
            // exclude keeps the excluded slot; the oldest of the rest goes
            evict_lru_page(&mut kv, Some(0));
            assert_eq!(kv.active.len(), 1);
            assert_eq!(kv.active[0].page_id, 1);
        }
    }

    /* ── brain migration + vector-space fault tests ───────────────── */

    #[cfg(test)]
    fn fresh_test_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sofuu-mig-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(test)]
    fn dir_entries(dir: &std::path::Path) -> std::collections::BTreeSet<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    }

    /// A legacy brain exactly as the retired C shell wrote it: 768-dim
    /// hash vectors + plain records JSON, no embedding manifest.
    #[cfg(test)]
    fn build_legacy_hash_brain(path: &std::path::Path, texts: &[&str]) {
        let mut cma = crate::memory::cma::Cma::new(crate::embedding::HASH_DIM);
        let mut flat: Vec<f32> = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let v = crate::embedding::hash_v1_features(text);
            flat.extend_from_slice(&v);
            cma.remember(&v, text, "memory", (7 + i) as u32);
        }
        let r = unsafe {
            cma_qtsq_save(
                path.to_str().unwrap(),
                Some(&flat),
                texts.len(),
                crate::embedding::HASH_DIM,
                &cma.records_json(),
            )
        };
        assert_eq!(r, QTSQ_OK);
    }

    /// A semantic brain through the real open→remember→flush path, so the
    /// manifest is the one the runtime itself writes.
    #[cfg(test)]
    fn build_semantic_brain(path: &std::path::Path, texts: &[&str]) {
        let p = path.to_str().unwrap();
        let mut shell = unsafe {
            cma_open(p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
        }
        .expect("fresh semantic shell on a nonexistent file");
        for text in texts {
            let v = crate::embedding::semantic_v1(text).expect("baked model");
            let r = unsafe { cma_remember(&mut shell, &v, Some(text), Some("memory"), 7) };
            assert!(r >= 0);
        }
        assert_eq!(unsafe { cma_flush_to(&mut shell, p) }, 0);
    }

    #[test]
    fn migration_success_preserves_records_and_backup() {
        let dir = fresh_test_dir("mig-ok");
        let path = dir.join("brain.qtsq");
        let texts = [
            "decided to use qtc compression for the session store",
            "the payment webhook retries three times before dropping",
            "deploy checklist: bump version, tag, build, upload",
            "login timeout bug fixed by raising the pool limit",
            "grep the migration tests under crates/ml-train",
        ];
        build_legacy_hash_brain(&path, &texts);
        let before = std::fs::read(&path).unwrap();
        let p = path.to_str().unwrap().to_string();

        let mut shell = unsafe {
            cma_open(&p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
        }
        .expect("migration must succeed for a healthy legacy brain");
        assert_eq!(shell.embedding_id, crate::embedding::MODEL_ID);
        assert_eq!(shell.cma.len(), texts.len());
        let js = shell.cma.records_json();
        for text in &texts {
            assert!(js.contains(text), "record lost in migration: {text}");
        }

        /* Canonical file is now semantic with a matching manifest. */
        let brain = unsafe { cma_qtsq_load(&p) }.expect("reload canonical file");
        assert_eq!(brain.n, texts.len());
        assert_eq!(brain.dim, crate::embedding::SEMANTIC_DIM);
        assert!(manifest_matches(
            brain.meta.as_deref().unwrap(),
            crate::embedding::MODEL_ID,
            crate::embedding::SEMANTIC_DIM
        ));

        /* Backup is byte-identical to the pre-migration file and still a
         * loadable hash-v1 brain. */
        let backup = dir.join("brain.qtsq.hash-v1.bak");
        assert_eq!(std::fs::read(&backup).unwrap(), before);
        let old = unsafe { cma_qtsq_load(backup.to_str().unwrap()) }.expect("backup loads");
        assert_eq!(old.dim, crate::embedding::HASH_DIM);
        assert_eq!(old.n, texts.len());

        /* The migrated index actually retrieves. */
        let q = crate::embedding::semantic_v1("payment webhook retries").unwrap();
        assert!(!shell.cma.recall(&q, 5).is_empty());

        /* Restart: the migrated file reopens in the same space — no second
         * migration, no leftover sidecars. */
        let again = unsafe {
            cma_open(&p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
        }
        .expect("migrated brain reopens");
        assert_eq!(again.cma.len(), texts.len());
        let entries = dir_entries(&dir);
        assert!(entries.iter().any(|e| e.ends_with("hash-v1.bak")));
        assert!(
            !entries
                .iter()
                .any(|e| e.ends_with("hash-v1.bak.1") || e.contains("semantic-migration")),
            "leftover sidecars after restart: {entries:?}"
        );
    }

    #[test]
    fn v2_semantic_space_round_trips_and_is_isolated() {
        use crate::embedding::semantic_v2 as v2;

        let dir = fresh_test_dir("v2-isolation");
        let path = dir.join("brain-v2.qtsq");
        let p = path.to_str().unwrap().to_string();
        let texts = [
            "decided to use qtc compression for the session store",
            "the payment webhook retries three times before dropping",
            "deploy checklist: bump version, tag, build, upload",
        ];

        /* Fresh v2 open through the real remember→flush path. */
        let mut shell = unsafe {
            cma_open(&p, crate::embedding::SEMANTIC_DIM, Some(v2::MODEL_ID_V2))
        }
        .expect("fresh SEM2 shell on a nonexistent file");
        assert_eq!(shell.embedding_id, v2::MODEL_ID_V2);
        for text in &texts {
            let v = v2::semantic_v2(text).expect("baked SEM2 model");
            let r = unsafe { cma_remember(&mut shell, &v, Some(text), Some("memory"), 7) };
            assert!(r >= 0);
        }
        assert_eq!(unsafe { cma_flush_to(&mut shell, &p) }, 0);

        /* The manifest the runtime wrote identifies the v2 space exactly. */
        let brain = unsafe { cma_qtsq_load(&p) }.expect("reload v2 file");
        assert_eq!(brain.n, texts.len());
        assert_eq!(brain.dim, crate::embedding::SEMANTIC_DIM);
        let meta = brain.meta.as_deref().unwrap();
        assert!(manifest_matches(
            meta,
            v2::MODEL_ID_V2,
            crate::embedding::SEMANTIC_DIM
        ));
        let manifest = embedding_manifest(meta).unwrap();
        assert_eq!(manifest.id, v2::MODEL_ID_V2);
        assert_eq!(manifest.input, v2::INPUT_EMBEDDER_ID_V2);
        assert_eq!(manifest.artifact, v2::model_artifact_id_v2());

        /* Same space reopens; the index retrieves. */
        let mut again = unsafe {
            cma_open(&p, crate::embedding::SEMANTIC_DIM, Some(v2::MODEL_ID_V2))
        }
        .expect("v2 brain reopens");
        assert_eq!(again.cma.len(), texts.len());
        let q = v2::semantic_v2("payment webhook retries").unwrap();
        assert!(!again.cma.recall(&q, 5).is_empty());

        /* §12: a v1 semantic open of the SAME file must be refused —
         * both are 64-dim, so only the manifest can catch this. */
        assert!(
            unsafe {
                cma_open(&p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
            }
            .is_none(),
            "v1 open of a v2 brain must refuse (dimension collision)"
        );

        /* §12: the reverse — a v2 open of a v1 brain is refused too. */
        let v1_path = dir.join("brain.qtsq");
        build_semantic_brain(&v1_path, &texts);
        assert!(
            unsafe {
                cma_open(
                    v1_path.to_str().unwrap(),
                    crate::embedding::SEMANTIC_DIM,
                    Some(v2::MODEL_ID_V2),
                )
            }
            .is_none(),
            "v2 open of a v1 brain must refuse"
        );

        /* §12: a v2 open of a legacy hash brain must NOT fire the v1
         * migration — the file stays untouched, no backup sidecars. */
        let legacy = dir.join("legacy.qtsq");
        build_legacy_hash_brain(&legacy, &texts);
        let before = std::fs::read(&legacy).unwrap();
        assert!(
            unsafe {
                cma_open(
                    legacy.to_str().unwrap(),
                    crate::embedding::SEMANTIC_DIM,
                    Some(v2::MODEL_ID_V2),
                )
            }
            .is_none(),
            "v2 open of a hash-v1 brain must refuse"
        );
        assert_eq!(std::fs::read(&legacy).unwrap(), before, "hash brain mutated");
        let entries = dir_entries(&dir);
        assert!(
            entries.iter().all(|e| !e.contains("hash-v1.bak") && !e.contains("migration")),
            "v2 open must not migrate or back up: {entries:?}"
        );
    }

    #[test]
    fn migration_refused_when_a_record_has_no_text() {
        let dir = fresh_test_dir("mig-notext");
        let path = dir.join("brain.qtsq");
        build_legacy_hash_brain(
            &path,
            &["first memory", "second memory", "third memory"],
        );
        /* Corrupt exactly like a damaged legacy file: record 1 loses text. */
        let brain = unsafe { cma_qtsq_load(path.to_str().unwrap()) }.unwrap();
        let mut root: serde_json::Value =
            serde_json::from_str(brain.meta.as_deref().unwrap()).unwrap();
        root["records"][1].as_object_mut().unwrap().remove("text");
        assert_eq!(
            unsafe {
                cma_qtsq_save(
                    path.to_str().unwrap(),
                    Some(&brain.vecs),
                    brain.n,
                    brain.dim,
                    &root.to_string(),
                )
            },
            QTSQ_OK
        );
        let before = std::fs::read(&path).unwrap();
        let entries_before = dir_entries(&dir);

        let opened = unsafe {
            cma_open(
                path.to_str().unwrap(),
                crate::embedding::SEMANTIC_DIM,
                Some(crate::embedding::MODEL_ID),
            )
        };
        assert!(
            opened.is_none(),
            "migration must refuse a brain with a textless record"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "original file modified on refusal"
        );
        assert_eq!(
            dir_entries(&dir),
            entries_before,
            "sidecar left behind on refusal"
        );
    }

    #[test]
    fn semantic_brain_refuses_hash_open() {
        let dir = fresh_test_dir("sem-refuse-hash");
        let path = dir.join("brain.qtsq");
        build_semantic_brain(&path, &["semantic memory one", "semantic memory two"]);
        let p = path.to_str().unwrap();
        let before = std::fs::read(p).unwrap();

        /* No explicit id → a 768 request infers hash-v1 and must be
         * refused: the file's manifest names a different space. */
        assert!(unsafe { cma_open(p, crate::embedding::HASH_DIM, None) }.is_none());
        assert!(
            unsafe { cma_open(p, crate::embedding::HASH_DIM, Some(crate::embedding::INPUT_EMBEDDER_ID)) }
                .is_none()
        );
        assert_eq!(std::fs::read(p).unwrap(), before);

        /* The matching request still opens (nothing was damaged). */
        let mut shell = unsafe {
            cma_open(p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
        }
        .expect("matching space still opens after refusals");
        assert_eq!(shell.cma.len(), 2);
    }

    #[test]
    fn semantic_brain_refuses_stale_artifact_and_remote_vectors() {
        let dir = fresh_test_dir("sem-stale");
        let path = dir.join("brain.qtsq");
        build_semantic_brain(&path, &["stale artifact probe"]);
        let p = path.to_str().unwrap();

        /* Tamper: replace the artifact id with a stale one. */
        let brain = unsafe { cma_qtsq_load(p) }.unwrap();
        let mut root: serde_json::Value =
            serde_json::from_str(brain.meta.as_deref().unwrap()).unwrap();
        root["embedding"]["artifact"] = serde_json::json!("0000000000000000");
        assert_eq!(
            unsafe {
                cma_qtsq_save(p, Some(&brain.vecs), brain.n, brain.dim, &root.to_string())
            },
            QTSQ_OK
        );
        let before = std::fs::read(p).unwrap();
        assert!(
            unsafe { cma_open(p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID)) }
                .is_none(),
            "a stale artifact must not be mixed into the live semantic space"
        );
        assert_eq!(std::fs::read(p).unwrap(), before);

        /* A remote-provider brain (different embedding id) is likewise
         * refused from the local semantic space. */
        root["embedding"]["id"] = serde_json::json!("provider:test-model@rev1");
        root["embedding"]["artifact"] = serde_json::json!("");
        assert_eq!(
            unsafe {
                cma_qtsq_save(p, Some(&brain.vecs), brain.n, brain.dim, &root.to_string())
            },
            QTSQ_OK
        );
        let before = std::fs::read(p).unwrap();
        assert!(
            unsafe { cma_open(p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID)) }
                .is_none(),
            "remote vectors must not enter the local semantic space"
        );
        assert_eq!(std::fs::read(p).unwrap(), before);
    }

    /// A PROVABLY EMPTY brain (metadata says zero records) is a healthy
    /// fresh store, not damage: adopting it silently is what stops every
    /// chat start in a new project from printing "empty/corrupt" and
    /// churning the file through a .bak. The fresh shell keeps working.
    #[test]
    fn empty_brain_is_adopted_without_backup_noise() {
        let dir = fresh_test_dir("brain-empty");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap();
        /* Metadata-only container: no "memories" stream (the QTSQ tensor
         * codec cannot store a zero-length tensor, so an empty brain is
         * always written this way). */
        assert_eq!(
            unsafe { cma_qtsq_save(p, None, 0, 64, "{\"records\":[]}") },
            QTSQ_OK
        );
        let before = std::fs::read(&path).unwrap();

        let mut shell = unsafe { cma_open(p, crate::embedding::HASH_DIM, None) }
            .expect("an empty brain opens as a fresh store");
        assert!(shell.cma.is_empty());
        assert!(
            !dir.join("brain.qtsq.bak").exists(),
            "an empty brain has nothing to protect — no backup, no warning"
        );
        assert_eq!(std::fs::read(&path).unwrap(), before, "the file is left untouched");

        /* And it round-trips like any other store. */
        let v = crate::embedding::hash_v1_features("first real memory");
        assert!(unsafe {
            cma_remember(&mut shell, &v, Some("first real memory"), Some("memory"), 1)
        } >= 0);
        assert_eq!(unsafe { cma_flush_to(&mut shell, p) }, 0);
        let reloaded = unsafe { cma_qtsq_load(p) }.unwrap();
        assert_eq!(reloaded.n, 1);
        assert!(reloaded.found_memories);
    }

    /// The dangerous shape: metadata claims records but the vectors are
    /// GONE (no "memories" stream). That is real data loss the user has
    /// not seen yet — back it up before anything can overwrite it.
    #[test]
    fn brain_claiming_records_without_vectors_is_backed_up() {
        let dir = fresh_test_dir("brain-claims-records");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap();
        assert_eq!(
            unsafe { cma_qtsq_save(p, None, 0, 64, "{\"records\":[{\"text\":\"lost\"}]}") },
            QTSQ_OK
        );
        let before = std::fs::read(&path).unwrap();

        let mut shell = unsafe { cma_open(p, crate::embedding::HASH_DIM, None) }
            .expect("unreadable brain still starts fresh instead of failing the open");
        assert!(shell.cma.is_empty());
        assert_eq!(
            std::fs::read(dir.join("brain.qtsq.bak")).unwrap(),
            before,
            "a brain whose vectors are missing must be backed up before any flush"
        );
    }

    /// Backups must never clobber an earlier one: the moment a brain is
    /// unreadable is exactly when the previous copy matters most.
    #[test]
    fn repeated_backups_never_clobber() {
        let dir = fresh_test_dir("brain-backup-chain");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap();
        assert_eq!(
            unsafe { cma_qtsq_save(p, None, 0, 64, "{\"records\":[{\"text\":\"lost\"}]}") },
            QTSQ_OK
        );
        let first = unsafe { cma_open(p, crate::embedding::HASH_DIM, None) };
        assert!(first.is_some());
        let original = std::fs::read(dir.join("brain.qtsq.bak")).unwrap();
        /* Second failure on the same path must land beside the first. */
        let second = unsafe { backup_brain(p) };
        assert_ne!(second, format!("{p}.bak"));
        assert!(std::path::Path::new(&second).exists());
        assert_eq!(std::fs::read(dir.join("brain.qtsq.bak")).unwrap(), original);
    }

    #[test]
    fn corrupt_container_is_backed_up_then_fresh_start() {
        let dir = fresh_test_dir("brain-garbage");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap();
        std::fs::write(&path, b"not a qtsq container at all").unwrap();
        let before = std::fs::read(&path).unwrap();
        let mut shell = unsafe { cma_open(p, crate::embedding::HASH_DIM, None) }
            .expect("garbage must not fail the open — chat keeps working");
        assert!(shell.cma.is_empty());
        assert_eq!(std::fs::read(dir.join("brain.qtsq.bak")).unwrap(), before);
    }

    #[test]
    fn legacy_hash_brain_still_opens_as_hash() {
        let dir = fresh_test_dir("legacy-hash");
        let path = dir.join("brain.qtsq");
        build_legacy_hash_brain(&path, &["legacy memory one", "legacy memory two"]);
        let p = path.to_str().unwrap();
        let before = std::fs::read(&path).unwrap();
        let entries_before = dir_entries(&dir);

        let shell = unsafe { cma_open(p, crate::embedding::HASH_DIM, None) }
            .expect("legacy brain opens as hash-v1 without migrating");
        assert_eq!(shell.embedding_id, crate::embedding::INPUT_EMBEDDER_ID);
        assert_eq!(shell.cma.len(), 2);
        let js = shell.cma.records_json();
        assert!(js.contains("legacy memory one") && js.contains("legacy memory two"));
        /* A plain hash open must not migrate or touch anything. */
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(dir_entries(&dir), entries_before);
    }

    #[cfg(unix)]
    #[test]
    fn migration_refused_when_staging_write_fails() {
        use std::os::unix::fs::PermissionsExt;
        let dir = fresh_test_dir("mig-readonly");
        let path = dir.join("brain.qtsq");
        build_legacy_hash_brain(&path, &["read only probe one", "read only probe two"]);
        let p = path.to_str().unwrap().to_string();
        let before = std::fs::read(&path).unwrap();
        let entries_before = dir_entries(&dir);

        let mut perms = std::fs::metadata(&dir).unwrap().permissions();
        perms.set_mode(0o555);
        std::fs::set_permissions(&dir, perms).unwrap();

        let opened = unsafe {
            cma_open(&p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
        };

        let mut restore = std::fs::metadata(&dir).unwrap().permissions();
        restore.set_mode(0o755);
        std::fs::set_permissions(&dir, restore).unwrap();

        assert!(
            opened.is_none(),
            "migration must refuse when it cannot stage the new brain"
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(dir_entries(&dir), entries_before, "no sidecars on refusal");
    }

    /* ── P1-6: a hostile n_hints must never reach with_capacity raw ── */

    /// 1025 seeded pages make the clamp observable in the RESULT itself:
    /// kv.search(q, 4294967295) must come back with exactly 1024 hints —
    /// unclamped, js_kv_search hands u32::MAX to kv_search's
    /// Vec::with_capacity, a ~16GB reservation (allocator abort where the
    /// platform refuses it; a silent virtual reservation under macOS
    /// overcommit). Both signatures fail this test: process death, or
    /// resLen 1025 ≠ 1024.
    #[test]
    fn kv_search_hostile_n_hints_clamped() {
        use sofuu_ffi::qjs::CtxPtr;

        let dir = fresh_test_dir("kv-hints");
        let root = dir.join("kv");
        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);
            kv_register(ctx);

            /* count = 1 layer × 1 head × 1 token × headDim 8 = 8 floats/page. */
            let script = format!(
                "var kv = sofuu.kv.open('{}', {{nLayers:1, nHeads:1, headDim:8}});\n{}",
                root.to_str().unwrap(),
                r#"var k = new Float32Array(8);
var v = new Float32Array(8);
var saved = 0;
for (var i = 0; i < 1025; i++) {
  k[0] = (i % 7) + 0.25;
  if (kv.save(k, v, 1) > 0) saved++;
}
var q = new Float32Array(64);
q[0] = 1.0;
var hostile = kv.search(q, 4294967295);
var normal = kv.search(q, 3);
var seen = {};
var distinct = 0;
for (var i = 0; i < hostile.length; i++) {
  if (!seen[hostile[i]]) { seen[hostile[i]] = 1; distinct++; }
}
globalThis.out = JSON.stringify({saved: saved, resLen: hostile.length,
  first: hostile.length ? hostile[0] : -1, distinct: distinct,
  normalLen: normal.length});"#
            );

            let script_c = CString::new(script).unwrap();
            let r = qjs::JS_Eval(
                ctx,
                script_c.as_ptr(),
                script_c.as_bytes().len(),
                c"<p1-6>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            assert!(!qjs::is_exception(r), "kv search script threw");

            let global = qjs::sofuu_js_get_global_object(ctx);
            let v = qjs::sofuu_js_get_property_str(ctx, global, c"out".as_ptr());
            qjs::sofuu_js_free_value(ctx, global);
            let p = qjs::sofuu_js_to_cstring(ctx, v);
            qjs::sofuu_js_free_value(ctx, v);
            assert!(!p.is_null(), "script did not set globalThis.out");
            let out = CStr::from_ptr(p).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, p);

            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);

            let j: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(j["saved"], 1025, "every page save must succeed");
            assert_eq!(j["resLen"], 1024, "hostile n_hints must be clamped to 1024");
            let first = j["first"].as_u64().unwrap();
            assert!((1..=1025).contains(&first), "hint must be a real page id, got {first}");
            assert_eq!(j["distinct"], 1024, "hints must be distinct page ids");
            assert_eq!(j["normalLen"], 3, "a small n_hints stays exact");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ml-2 (AUDIT-2026-09-07): a hostile brain file whose "memories"
    /// tensor header claims dims [1000,1000] (product 1_000_000) but
    /// carries 6 floats. The C compressor validates neither side, so only
    /// the Rust loader can refuse. Pre-fix this loaded vecs.len()=6 with
    /// n=1000/dim=1000; the loader must skip the stream itself (vecs
    /// empty → cma_open falls through to a fresh shell, fail-safe).
    #[test]
    fn brain_load_rejects_inconsistent_memories_schema() {
        let dir = fresh_test_dir("ml2-brain");
        let path = dir.join("brain.qtsq");
        let path_s = path.to_str().unwrap().to_string();
        unsafe {
            let container = qtsq::qtsq_ctx_alloc();
            assert!(!container.is_null());
            assert_eq!(qtsq::qtsq_container_create(container), QTSQ_OK);
            (*container).set_codec(qtsq::QTSQ_CODEC_QTC);

            let sub = qtsq::qtsq_ctx_alloc();
            assert!(!sub.is_null());
            assert_eq!(qtsq::qtsq_init(sub), QTSQ_OK);
            (*sub).set_codec(qtsq::QTSQ_CODEC_QTC);
            let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let dims: [u32; 2] = [1000, 1000];
            assert_eq!(
                qtsq::qtsq_compress_tensor(sub, data.as_ptr(), 6, dims.as_ptr(), 2),
                QTSQ_OK,
                "the C compressor accepts the hostile header (no product check)"
            );
            assert_eq!(
                qtsq::qtsq_container_add_stream(container, sub, c"memories".as_ptr()),
                QTSQ_OK
            );
            qtsq::qtsq_free(sub);
            qtsq::qtsq_ctx_free(sub);

            assert_eq!(qtsq::qtsq_container_pack(container), QTSQ_OK);
            assert_eq!(qtsq_adapter_encrypt(container), QTSQ_OK);
            let ctmp = CString::new(format!("{path_s}.tmp")).unwrap();
            assert_eq!(qtsq::qtsq_write(container, ctmp.as_ptr()), QTSQ_OK);
            qtsq::qtsq_free(container);
            qtsq::qtsq_ctx_free(container);
            std::fs::rename(format!("{path_s}.tmp"), &path_s).unwrap();

            let brain = cma_qtsq_load(&path_s).expect("the container itself must still load");
            assert!(brain.found_memories, "the memories stream was present");
            assert!(
                brain.vecs.is_empty() && brain.n == 0 && brain.dim == 0,
                "inconsistent schema must be skipped (vecs={}, n={}, dim={})",
                brain.vecs.len(),
                brain.n,
                brain.dim
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ml-2 site B: a page file whose K stream claims dims
    /// [999999,1,999999,1] (product ≈1e12) but carries 8 floats, next to
    /// an honest V stream. The tail k/v length-parity gate alone accepted
    /// this (8 == 8) and handed n_layers=999999/n_tokens=999999 — read
    /// from the hostile header — to ActivePage.
    #[test]
    fn kv_page_load_rejects_inconsistent_k_schema() {
        let dir = fresh_test_dir("ml2-page");
        let path = dir.join("pages.qtsq");
        let path_s = path.to_str().unwrap().to_string();
        unsafe {
            let container = qtsq::qtsq_ctx_alloc();
            assert!(!container.is_null());
            assert_eq!(qtsq::qtsq_container_create(container), QTSQ_OK);
            (*container).set_codec(qtsq::QTSQ_CODEC_QTC);

            let ksub = qtsq::qtsq_ctx_alloc();
            assert!(!ksub.is_null());
            assert_eq!(qtsq::qtsq_init(ksub), QTSQ_OK);
            (*ksub).set_codec(qtsq::QTSQ_CODEC_QTC);
            let k: [f32; 8] = [0.5; 8];
            let kdims: [u32; 4] = [999_999, 1, 999_999, 1];
            assert_eq!(
                qtsq::qtsq_compress_tensor(ksub, k.as_ptr(), 8, kdims.as_ptr(), 4),
                QTSQ_OK
            );
            assert_eq!(
                qtsq::qtsq_container_add_stream(container, ksub, c"kv_00000007_K".as_ptr()),
                QTSQ_OK
            );
            qtsq::qtsq_free(ksub);
            qtsq::qtsq_ctx_free(ksub);

            let vsub = qtsq::qtsq_ctx_alloc();
            assert!(!vsub.is_null());
            assert_eq!(qtsq::qtsq_init(vsub), QTSQ_OK);
            (*vsub).set_codec(qtsq::QTSQ_CODEC_QTC);
            let v: [f32; 8] = [1.0; 8];
            let vdims: [u32; 4] = [1, 1, 1, 8];
            assert_eq!(
                qtsq::qtsq_compress_tensor(vsub, v.as_ptr(), 8, vdims.as_ptr(), 4),
                QTSQ_OK
            );
            assert_eq!(
                qtsq::qtsq_container_add_stream(container, vsub, c"kv_00000007_V".as_ptr()),
                QTSQ_OK
            );
            qtsq::qtsq_free(vsub);
            qtsq::qtsq_ctx_free(vsub);

            assert_eq!(qtsq::qtsq_container_pack(container), QTSQ_OK);
            assert_eq!(qtsq_adapter_encrypt(container), QTSQ_OK);
            let ctmp = CString::new(format!("{path_s}.tmp")).unwrap();
            assert_eq!(qtsq::qtsq_write(container, ctmp.as_ptr()), QTSQ_OK);
            qtsq::qtsq_free(container);
            qtsq::qtsq_ctx_free(container);
            std::fs::rename(format!("{path_s}.tmp"), &path_s).unwrap();

            let r = kv_qtsq_load_page(&path_s, 7);
            assert!(
                r.is_err(),
                "inconsistent K dims must reject the page, got {:?}",
                r.as_ref().map(|(_, _, l, t)| (*l, *t))
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ml-2 positive control: a well-formed page (K dims [1,1,2,4],
    /// count 8) must keep loading and report its schema dims.
    #[test]
    fn kv_page_load_accepts_consistent_k_schema() {
        let dir = fresh_test_dir("ml2-page-ok");
        let path = dir.join("pages.qtsq");
        let path_s = path.to_str().unwrap().to_string();
        unsafe {
            let container = qtsq::qtsq_ctx_alloc();
            assert!(!container.is_null());
            assert_eq!(qtsq::qtsq_container_create(container), QTSQ_OK);
            (*container).set_codec(qtsq::QTSQ_CODEC_QTC);
            for (suffix, vals) in [("_K", [0.5f32; 8]), ("_V", [1.0f32; 8])] {
                let sub = qtsq::qtsq_ctx_alloc();
                assert!(!sub.is_null());
                assert_eq!(qtsq::qtsq_init(sub), QTSQ_OK);
                (*sub).set_codec(qtsq::QTSQ_CODEC_QTC);
                let dims: [u32; 4] = [1, 1, 2, 4];
                assert_eq!(
                    qtsq::qtsq_compress_tensor(sub, vals.as_ptr(), 8, dims.as_ptr(), 4),
                    QTSQ_OK
                );
                let name = CString::new(format!("kv_00000003{suffix}")).unwrap();
                assert_eq!(qtsq::qtsq_container_add_stream(container, sub, name.as_ptr()), QTSQ_OK);
                qtsq::qtsq_free(sub);
                qtsq::qtsq_ctx_free(sub);
            }
            assert_eq!(qtsq::qtsq_container_pack(container), QTSQ_OK);
            assert_eq!(qtsq_adapter_encrypt(container), QTSQ_OK);
            let ctmp = CString::new(format!("{path_s}.tmp")).unwrap();
            assert_eq!(qtsq::qtsq_write(container, ctmp.as_ptr()), QTSQ_OK);
            qtsq::qtsq_free(container);
            qtsq::qtsq_ctx_free(container);
            std::fs::rename(format!("{path_s}.tmp"), &path_s).unwrap();

            let (k, v, n_layers, n_tokens) =
                kv_qtsq_load_page(&path_s, 3).expect("well-formed page must load");
            assert_eq!(k.len(), 8);
            assert_eq!(v.len(), 8);
            assert_eq!(n_layers, 1);
            assert_eq!(n_tokens, 2);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ml-1 (AUDIT-2026-09-07) no-regression guard: a 3000-page
    /// index.json still hydrates every page after the reserve clamp (the
    /// clamp bounds only the initial reservation; Vec growth covers the
    /// real item count, which JS_ParseJSON makes authoritative).
    #[test]
    fn kv_index_with_many_pages_loads_fully() {
        use sofuu_ffi::qjs::CtxPtr;

        let dir = fresh_test_dir("kv-ml1");
        let root = dir.join("kv");
        std::fs::create_dir_all(&root).unwrap();
        let _loop_guard = crate::rt::TEST_LOOP_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);

            let mut entries = String::from("{\"pages\":[");
            for i in 0..3000u32 {
                if i > 0 {
                    entries.push(',');
                }
                entries.push_str(&format!(
                    "{{\"page_id\":{},\"strength\":1.0,\"created_at\":{}}}",
                    i + 1,
                    1000 + i
                ));
            }
            entries.push_str("]}");
            std::fs::write(root.join("index.json"), entries).unwrap();

            let kv = kv_store_open(ctx, root.to_str().unwrap(), 1, 1, 8)
                .expect("kv_store_open must succeed");
            assert_eq!(kv.pages.len(), 3000, "every page entry must load");
            assert_eq!(kv.pages[2999].page_id, 3000);
            assert_eq!(kv.pages[0].strength, 1.0);
            // Omitted k_summary zero-fills (older-index path), not garbage.
            assert!(kv.pages[42].k_summary.iter().all(|v| *v == 0.0));

            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ── unconditional exports (linker contract with engine.c) ──────────
// engine.c declares these externs and calls them under
// `#if SOFUU_MEMORY && defined(SOFUU_QTSQ_PRESENT)` — the C compile sets
// that define exactly when sofuu-ffi's build.rs finds libqtsq.a, which is
// the same condition has_qtsq mirrors for this crate.

/// Register the sofuu.memory.* JS surface (C symbol mod_memory_register).
///
/// # Safety
/// `ctx` must be a live QuickJS context (engine.c passes its engine ctx).
#[no_mangle]
pub unsafe extern "C" fn mod_memory_register(ctx: *mut JSContext) {
    #[cfg(has_qtsq)]
    unsafe {
        impl_qtsq::memory_register(ctx)
    }
    #[cfg(not(has_qtsq))]
    let _ = ctx; /* QTSQ absent: engine.c's guard never calls us — JS sees
                  * no sofuu.memory, the same observable absence as the C */
}

/// Register the sofuu.kv.* JS surface (C symbol mod_kv_register).
///
/// # Safety
/// `ctx` must be a live QuickJS context.
#[no_mangle]
pub unsafe extern "C" fn mod_kv_register(ctx: *mut JSContext) {
    #[cfg(has_qtsq)]
    unsafe {
        impl_qtsq::kv_register(ctx)
    }
    #[cfg(not(has_qtsq))]
    let _ = ctx;
}

/// Register the sofuu.agent.* JS surface (C symbol mod_agent_register).
///
/// # Safety
/// `ctx` must be a live QuickJS context.
#[no_mangle]
pub unsafe extern "C" fn mod_agent_register(ctx: *mut JSContext) {
    #[cfg(has_qtsq)]
    unsafe {
        impl_qtsq::agent_register(ctx)
    }
    #[cfg(not(has_qtsq))]
    let _ = ctx;
}
