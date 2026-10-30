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

/// W2.1/W2.2: cross-process arbitration on a brain — the `flock` a writable
/// open takes on `<brain>.lock` (exclusive) and a readonly open takes shared
/// (on the lock file, or on the store directory when there is none). The
/// declaration lives HERE rather than in rt/mod.rs so the module ships with
/// the memory surface it serves; `#[path]` resolves beside this file.
#[path = "brain_lock.rs"]
pub mod brain_lock;

#[cfg(has_qtsq)]
mod impl_qtsq {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::ptr;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::brain_lock;
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
    /// B4: the vault password is the machine-local key (rt/brain_key.rs).
    /// The legacy fixed password is read-only, for migrating pre-B4 files.
    fn sofuu_qtsq_password() -> String {
        crate::rt::brain_store::vault_password()
    }
    const SOFUU_QTSQ_LEGACY_PASSWORD: &str = crate::rt::brain_store::LEGACY_PASSWORD;

    /// B0 (AUDIT-2026-10-13): the codec's tensor-compression level.
    /// 0 = full contest — the best ratio and the behaviour every brain/KV
    /// file was written with before the parameter existed (the codec's
    /// 2.1.0 API added it; the FFI used to omit it, leaving the callee to
    /// read an undefined register). Levels 1 (fast) and 2 (store) are
    /// available for future tuning; nothing uses them yet.
    const QTSQ_LEVEL_FULL: std::os::raw::c_int = crate::rt::brain_store::QTSQ_LEVEL_FULL;

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
        let pw = CString::new(sofuu_qtsq_password()).unwrap_or_default();
        qtsq::qtsq_vault_encrypt_password(ctx, pw.as_ptr())
    }

    /// Decrypt a context loaded with qtsq_read when its header says encrypted.
    /// B4: the machine key first, then the legacy fixed password — returning
    /// whether the LEGACY one opened it, so the caller re-encrypts on the
    /// next write. Retrying on the same context is safe: the codec verifies
    /// the HMAC before touching the ciphertext (qtsq_vault.c).
    unsafe fn qtsq_adapter_decrypt_ex(ctx: *mut QtsqContext) -> Result<bool, c_int> {
        if !(*ctx).is_encrypted() {
            return Ok(false);
        }
        let pw = CString::new(sofuu_qtsq_password()).unwrap_or_default();
        if qtsq::qtsq_vault_decrypt_password(ctx, pw.as_ptr()) == QTSQ_OK {
            return Ok(false);
        }
        let legacy = CString::new(SOFUU_QTSQ_LEGACY_PASSWORD).unwrap_or_default();
        if qtsq::qtsq_vault_decrypt_password(ctx, legacy.as_ptr()) == QTSQ_OK {
            return Ok(true);
        }
        Err(qtsq::QTSQ_ERR_FORMAT)
    }

    /// The pre-B4 call shape: decrypt, discarding the legacy flag.
    unsafe fn qtsq_adapter_decrypt(ctx: *mut QtsqContext) -> c_int {
        match qtsq_adapter_decrypt_ex(ctx) {
            Ok(_) => QTSQ_OK,
            Err(e) => e,
        }
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
                    r = qtsq::qtsq_compress_tensor(sub, vecs.as_ptr(), n_memories * vec_dim, dims.as_ptr(), 2, QTSQ_LEVEL_FULL);
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
            /* C2: durable atomic publish behind a write-ahead journal.
             * fsync the staged file, record the intent (+digest) and fsync
             * the record, rename, fsync the directory, then clear the
             * journal. A crash at any point is recovered on the next open:
             * roll forward when the staged bytes are intact and match,
             * roll back when they are not — the old brain is never left
             * half-replaced. A journal we cannot claim (a live writer on
             * the same project) degrades to the plain durable publish. */
            let tmp = format!("{path}.tmp");
            let dir = std::path::Path::new(path)
                .parent()
                .map(|d| d.to_string_lossy().to_string())
                .unwrap_or_else(|| ".".to_string());
            let mut journal = crate::rt::wal::Journal::begin(&dir);
            let recorded = journal
                .as_mut()
                .map(|j| j.record(&tmp, path))
                .unwrap_or(false);
            if crate::rt::wal::publish(&tmp, path) {
                if recorded {
                    if let Some(j) = journal {
                        j.commit();
                    }
                }
            } else {
                let _ = std::fs::remove_file(&tmp);
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
        /// Records already parsed, for the segmented loader: handing these to
        /// hydrate skips the write-to-text/parse-back round trip (470 ms +
        /// 135 ms of a 100k open). `None` means "parse `meta` instead".
        record_values: Option<Vec<serde_json::Value>>,
        found_memories: bool, /* the "memories" stream was present */
        /// B4: the LEGACY fixed password opened this file — the next write
        /// re-encrypts it under the machine key (backup kept first).
        legacy_encryption: bool,
    }

    /// cma_qtsq_load — read + decrypt a container, extract the "memories"
    /// tensor (n/dim from its schema) and the "metadata" JSON stream.
    /// The pre-W2 reader: sweeps a crashed writer's staging file, then loads.
    unsafe fn cma_qtsq_load(path: &str) -> Result<BrainData, c_int> {
        cma_qtsq_load_opts(path, true)
    }

    /// `cma_qtsq_load`, without the write when `sweep_tmp` is false — a
    /// readonly open (W2.2) may not even delete a stale `{brain}.tmp`.
    unsafe fn cma_qtsq_load_opts(path: &str, sweep_tmp: bool) -> Result<BrainData, c_int> {
        // P3 (AUDIT-2026-09-07): every open is the one chance to clear the
        // `{brain}.tmp` a crashed writer left behind (age-guarded — see
        // sweep_stale_tmp_file).
        if sweep_tmp {
            sweep_stale_tmp_file(&format!("{path}.tmp"));
        }
        let cpath = CString::new(path).unwrap_or_default();
        let container = qtsq::qtsq_ctx_alloc();
        if container.is_null() {
            return Err(qtsq::QTSQ_ERR_ALLOC);
        }
        let mut r = qtsq::qtsq_init(container);
        if r == QTSQ_OK {
            r = qtsq::qtsq_read(container, cpath.as_ptr());
        }
        let mut legacy_encryption = false;
        if r == QTSQ_OK {
            match qtsq_adapter_decrypt_ex(container) {
                Ok(l) => legacy_encryption = l,
                Err(e) => r = e,
            }
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
        Ok(BrainData {
            vecs,
            n,
            dim,
            meta,
            record_values: None,
            found_memories,
            legacy_encryption,
        })
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
        let mut r = qtsq::qtsq_compress_tensor_quantized(ctx_k, k, count, dims.as_ptr(), 4, k_precision, QTSQ_LEVEL_FULL);
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
        r = qtsq::qtsq_compress_tensor_quantized(ctx_v, v, count, dims.as_ptr(), 4, v_precision, QTSQ_LEVEL_FULL);
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
            /* C2: same durable publish as the brain — a page that survives
             * a power loss is either the complete new page or the old one,
             * never a torn file. */
            let tmp = format!("{path}.tmp");
            let dir = std::path::Path::new(path)
                .parent()
                .map(|d| d.to_string_lossy().to_string())
                .unwrap_or_else(|| ".".to_string());
            let mut journal = crate::rt::wal::Journal::begin(&dir);
            let recorded = journal
                .as_mut()
                .map(|j| j.record(&tmp, path))
                .unwrap_or(false);
            if crate::rt::wal::publish(&tmp, path) {
                if recorded {
                    if let Some(j) = journal {
                        j.commit();
                    }
                }
            } else {
                // P3 (AUDIT-2026-09-07): the old code left the tmp behind on
                // a failed rename — the exact leak the atomic-write scheme
                // exists to avoid.
                let _ = std::fs::remove_file(&tmp);
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
        /// B3: segmented-store state. `None` until the first segmented
        /// load/flush — a fresh brain, or a legacy single-file brain that
        /// has not been migrated yet.
        layout: Option<BrainLayout>,
        /// B4: this brain was opened with the LEGACY fixed password — the
        /// next write backs it up and re-encrypts under the machine key.
        legacy_encryption: bool,
        /// W6.1 (AUDIT-2026-10-29): what the LAST flush actually wrote.
        /// `flush()` used to answer a bare bool, so a host could not tell an
        /// empty flush from a full one, and had no record count at all.
        last_flush: Option<FlushStats>,
        /// W6.1/W1.7: why the last flush failed (code + message), when it did.
        flush_error: Option<(&'static str, String)>,
        /// Lifetime failed flushes on this handle — reported in stats().recovery.
        flush_errors: usize,
        /// W5.2: the verdict of the most recent `remember` — `remember()`
        /// keeps returning the numeric id (hosts do `forget(remember(..))`),
        /// and the `{id, merged, into}` record is read back through
        /// `lastRememberOutcome()`. Set on every remember, insert or merge.
        last_outcome: Option<crate::memory::cma::RememberOutcome>,
        /// W2.1/W2.2: the `flock` this handle holds for its whole lifetime
        /// (exclusive for a writable open, shared for a readonly one). Held
        /// here so Drop releases it — and so `close()` can release it early.
        lock: Option<brain_lock::BrainLock>,
        /// W2.2: this handle may NOT write anything. Every repair/heal/
        /// backup/manifest-publish path checks it, and `flush` refuses.
        readonly: bool,
        /// W3.3: whether the payloads have been read yet. A normal open
        /// reads the MANIFEST only and leaves `pending` set; the first row
        /// access (see `materialize`) replays the exact loader an eager
        /// open runs.
        loaded: bool,
        /// W3.3: the manifest a lazy open deferred (`None` for a fresh or
        /// legacy single-file brain, which have no segments to read).
        pending: Option<crate::rt::brain_store::Manifest>,
        /// W3.1: which generation the on-disk graph snapshot describes, and
        /// how many records it held, so a flush can tell whether rewriting it
        /// is worth ~30 ms of I/O.
        graph_snapshot_epoch: u64,
        graph_snapshot_records: Option<usize>,
        /// W3.4 perf: the last epoch whose superseded generations were swept,
        /// and how many flushes since the last sweep. A flush that only
        /// appends a segment cannot create an orphan or a superseded
        /// generation, so sweeping every flush was two full directory scans
        /// (plus a filename parse per segment) for nothing.
        last_sweep_epoch: Option<u64>,
        flushes_since_sweep: u32,
        /// W2.1 perf: (len, mtime) of the manifest this handle last
        /// published. The anti-concurrent-writer check used to decrypt and
        /// parse the whole manifest on EVERY flush; while this handle holds
        /// LOCK_EX and the bytes on disk are still the ones it wrote, that
        /// read cannot find a new writer, so a stat decides it instead.
        published_manifest_stamp: Option<(u64, Option<std::time::SystemTime>)>,
        /// W3.3: segment payloads this handle has actually read — 0 for a
        /// normal open until the first row access, N for `verify_full`.
        /// Reported as `stats().segments_read`.
        segments_read: usize,
        /// W3.3/W1.8: the recovery a lazy load discovered, delivered as
        /// events on the access that triggered it (the open had no ctx to
        /// emit them from).
        recovery_pending: bool,
        /// W2.1: `close()` released the lock — a further flush from this
        /// handle must not resurrect a store it no longer owns.
        closed: bool,
    }

    /// W6.1: the real numbers a flush produced.
    #[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
    struct FlushStats {
        /// Record VERSIONS appended/rewritten by this flush.
        records_written: usize,
        /// NEW segment files sealed (published) by this flush.
        segments_sealed: usize,
    }

    impl CmaShell {
        /// Record a failed flush: the code a host branches on, the message it
        /// reads, and the counter stats() reports. Called at every failure
        /// return of brain_flush_segmented, so no failure path can answer
        /// "ok" or an uncoded error.
        fn note_flush_failure(&mut self, code: &'static str, msg: impl Into<String>) {
            self.last_flush = None;
            self.flush_error = Some((code, msg.into()));
            self.flush_errors += 1;
        }

        fn note_flush_success(&mut self, stats: FlushStats) {
            self.last_flush = Some(stats);
            self.flush_error = None;
        }
    }

    /// W1.8 (AUDIT-2026-10-29): what this store's load had to GIVE UP.
    ///
    /// The damage was only ever visible as stderr ("[sofuu/cma] SEGMENT
    /// DAMAGE: …"). Every field here is a number the loader really computes
    /// in `SegScan` — nothing is invented; `index_rebuilt` counts opens that
    /// had to rebuild the vector index from SURVIVORS of a damaged store
    /// (a pristine open always re-derives the HNSW from the segment log, so
    /// counting that would make every healthy open look like a recovery).
    #[derive(Clone, Default, Debug)]
    struct Recovery {
        /// Record versions lost with missing/unreadable segments (an upper
        /// bound on lost memories: a later version of the same id may live).
        records_dropped: usize,
        /// Segment files moved aside as `<file>.torn`.
        quarantined: Vec<String>,
        /// Segment files the manifest lists that are not on disk.
        missing: Vec<String>,
        /// Segments whose digest did not match but whose bytes read cleanly
        /// (crash window between segment and manifest publish) — adopted.
        adopted: Vec<String>,
        /// W2.3: damaged files a READONLY load would not touch — it took
        /// neither their records nor any repair action, so no `.torn` copy
        /// and no rewritten manifest exist because of this handle.
        left_in_place: Vec<String>,
        /// Survivor records renumbered to keep ids dense (0 = no gap).
        ids_renumbered: usize,
        /// Opens that rebuilt the index from the survivors of a damaged store.
        index_rebuilt: usize,
        /// Loads that also HEALED the store on disk (manifest/generation).
        repaired: usize,
        /// W3.3: segment payloads this load actually read.
        segments_read: usize,
    }

    impl Recovery {
        fn damaged(&self) -> bool {
            self.records_dropped > 0
                || !self.quarantined.is_empty()
                || !self.missing.is_empty()
                || !self.adopted.is_empty()
                || !self.left_in_place.is_empty()
                || self.ids_renumbered > 0
        }

        fn to_json(&self) -> serde_json::Value {
            serde_json::json!({
                "records_dropped": self.records_dropped,
                "segments_quarantined": self.quarantined.len(),
                "segments_missing": self.missing.len(),
                "segments_adopted": self.adopted.len(),
                "segments_left_in_place": self.left_in_place.len(),
                "ids_renumbered": self.ids_renumbered,
                "index_rebuilt": self.index_rebuilt,
                "repairs": self.repaired,
                "segments_read": self.segments_read,
                "quarantined_files": self.quarantined,
                "left_in_place_files": self.left_in_place,
                "damaged": self.damaged(),
            })
        }
    }

    /// B3.1: the on-disk layout of a segmented brain.
    struct BrainLayout {
        /// `<path>.d` — where the segments live.
        dir: String,
        segments: Vec<crate::rt::brain_store::SegmentMeta>,
        /// Generation counter; segment names carry it so a compaction never
        /// collides with the files it replaces.
        epoch: u64,
        /// Records the store has already persisted (ids below this are on
        /// disk; ids at or above it are new).
        persisted: usize,
        /// Unix seconds the persisted strengths were current as of — a load
        /// replays the decay that happened while the process was down.
        last_decay_at: u64,
        /// W5.3: the decay policy the manifest carried, if any. Kept on the
        /// layout so a lazy open can inherit it at materialise time (the
        /// handle is not loaded when the manifest is first read).
        decay_tag: Option<u8>,
        decay_seconds: Option<f64>,
        /// Lifetime capacity evictions (reported, never silent).
        evicted_total: u64,
        /// The incremental path cannot express the current state (a legacy
        /// file that has not been migrated, or retain() renumbered ids):
        /// the next flush rewrites every live record into fresh segments.
        needs_compaction: bool,
        /// W1.8: what THIS store's load recovered/dropped. Carried on the
        /// layout because the loader is the only place that knows.
        recovery: Recovery,
    }

    impl BrainLayout {
        /// Placeholder used while the real layout is carried alongside the
        /// loaded data (the loader returns them together).
        fn empty() -> Self {
            Self {
                dir: String::new(),
                segments: Vec::new(),
                epoch: 1,
                persisted: 0,
                last_decay_at: 0,
                decay_tag: None,
                decay_seconds: None,
                evicted_total: 0,
                needs_compaction: false,
                recovery: Recovery::default(),
            }
        }
    }

    /// Default record cap for a segmented brain. Overridable with
    /// SOFUU_BRAIN_MAX_RECORDS; 0 disables the cap. 100k × 768 dims is
    /// already ~300 MB of resident vectors, so this is a safety net, not a
    /// working limit.
    const BRAIN_MAX_RECORDS_DEFAULT: usize = 100_000;

    fn brain_max_records() -> usize {
        /* Tests override per-thread: the env is process-global and cargo
         * runs tests in parallel, so an env-based override would leak into
         * every other brain test in flight. */
        #[cfg(test)]
        if let Some(v) = BRAIN_CAP_OVERRIDE.with(|c| c.get()) {
            return v;
        }
        match std::env::var("SOFUU_BRAIN_MAX_RECORDS") {
            Ok(v) => v.trim().parse::<usize>().unwrap_or(BRAIN_MAX_RECORDS_DEFAULT),
            Err(_) => BRAIN_MAX_RECORDS_DEFAULT,
        }
    }

    #[cfg(test)]
    thread_local! {
        static BRAIN_CAP_OVERRIDE: std::cell::Cell<Option<usize>> =
            const { std::cell::Cell::new(None) };
    }

    #[cfg(test)]
    fn set_brain_cap_override(v: Option<usize>) {
        BRAIN_CAP_OVERRIDE.with(|c| c.set(v));
    }

    /// The embedding block the manifest carries (same shape as a legacy
    /// brain's metadata `embedding` object).
    fn embedding_block(shell: &CmaShell) -> serde_json::Value {
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
        serde_json::json!({
            "id": shell.embedding_id,
            "input": input,
            "dimension": shell.vec_dim,
            "artifact": artifact,
        })
    }

    /// One record's metadata as the store writes it (the RecordJson shape,
    /// minus vector_index — a segment carries the id explicitly).
    fn record_meta(shell: &CmaShell, id: u32) -> serde_json::Value {
        let r = &shell.cma.records[id as usize];
        serde_json::json!({
            "kv_page_id": r.kv_page_id,
            "strength": r.strength,
            "age_seconds": r.age_seconds,
            "half_life": r.half_life,
            "tier": r.tier,
            "text": r.text,
            "role": r.role,
            "entity_name": r.entity_name,
            "entity_type": r.entity_type,
            "corecall_ids": r.corecall_ids,
            "total_recalls": r.total_recalls,
            "positive_recalls": r.positive_recalls,
            "created_at": r.created_at,
            "last_recalled_at": r.last_recalled_at,
            "source": r.source,
        })
    }

    unsafe fn seg_record(shell: &CmaShell, id: u32) -> crate::rt::brain_store::SegRecord {
        crate::rt::brain_store::SegRecord {
            id,
            vec: shell.cma.index.vector(id).to_vec(),
            meta: record_meta(shell, id),
        }
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
        /* C2: the migration's final rename is the commit — make it durable
         * (fsync staged file, rename, fsync directory). */
        if !crate::rt::wal::publish(&staging, path) {
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

    /// W2.2/W3.3: how an open was asked to behave.
    #[derive(Clone, Copy, Debug, Default)]
    struct OpenOpts {
        /// W2.2: take a SHARED lock and write NOTHING — no journal replay,
        /// no backup, no quarantine, no heal, no manifest publish, and no
        /// lock file created when there is none.
        readonly: bool,
        /// W3.3: read + digest every segment payload at open (the pre-W3.3
        /// paranoid behaviour) instead of on first row access.
        verify_full: bool,
    }

    /// Why an open was refused. `Corrupt` is exactly what the old
    /// `Option<Box<CmaShell>>` `None` meant; `Locked` is W2.1's writer
    /// conflict and maps to the crate's E_LOCKED.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum OpenError {
        Corrupt,
        Locked,
    }

    /// cma_open — the pre-W2 shape every in-tree caller (and the C ABI)
    /// uses: a writable open with the lazy loader and `None` on failure.
    unsafe fn cma_open(
        path: &str,
        vec_dim: usize,
        requested_embedding_id: Option<&str>,
    ) -> Option<Box<CmaShell>> {
        cma_open_ex(path, vec_dim, requested_embedding_id, OpenOpts::default()).ok()
    }

    /// cma_open_ex — validate dims, take the W2.1 lock, build the Rust CMA,
    /// and hand back a shell that has either loaded the brain or (W3.3) is
    /// ready to load its payloads on first access.
    unsafe fn cma_open_ex(
        path: &str,
        vec_dim: usize,
        requested_embedding_id: Option<&str>,
        opts: OpenOpts,
    ) -> Result<Box<CmaShell>, OpenError> {
        if vec_dim == 0 || vec_dim > 8192 {
            eprintln!("[sofuu/cma] invalid vector dim {}", vec_dim);
            return Err(OpenError::Corrupt);
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
            return Err(OpenError::Corrupt);
        }
        if embedding_id == crate::embedding::semantic_v2::MODEL_ID_V2
            && crate::embedding::semantic_v2::baked_model_v2().is_err()
        {
            eprintln!("[sofuu/cma] SEM2 embedding model is unavailable");
            return Err(OpenError::Corrupt);
        }
        // M1: image space (img1-64, joint with sem2-64 text geometry).
        if embedding_id == crate::embedding::image::MODEL_ID_IMG
            && crate::embedding::image::baked_model_img().is_err()
        {
            eprintln!("[sofuu/cma] IMG1 embedding model is unavailable");
            return Err(OpenError::Corrupt);
        }

        /* W2.1/W2.2: arbitration starts BEFORE a single byte is read, so two
         * writers can never both open the same store — the loser gets no
         * handle at all, not a handle whose flush later fails. A readonly
         * open takes the SHARED lock instead (and never creates the file). */
        let lock_mode = if opts.readonly {
            brain_lock::Mode::Shared
        } else {
            brain_lock::Mode::Exclusive
        };
        let lock = match brain_lock::acquire(path, lock_mode) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[sofuu/cma] {e}");
                return Err(match e {
                    brain_lock::LockError::Busy { .. } => OpenError::Locked,
                });
            }
        };

        let mut shell = Box::new(CmaShell {
            mind_path: path.to_string(),
            vec_dim,
            embedding_id: embedding_id.clone(),
            cma: crate::memory::cma::Cma::new(vec_dim),
            layout: None,
            legacy_encryption: false,
            last_flush: None,
            flush_error: None,
            flush_errors: 0,
            last_outcome: None,
            lock: Some(lock),
            readonly: opts.readonly,
            loaded: false,
            pending: None,
            graph_snapshot_epoch: 0,
            graph_snapshot_records: None,
            last_sweep_epoch: None,
            flushes_since_sweep: 0,
            published_manifest_stamp: None,
            segments_read: 0,
            recovery_pending: false,
            closed: false,
        });
        /* B5.3: SOFUU_BRAIN_EF_SEARCH widens the recall search; unset keeps
         * the built-in default. */
        shell.cma.set_ef_search(crate::memory::cma::ef_search_from_env());

        /* C2: replay any write-ahead journal a crashed commit left behind
         * BEFORE the load — a staged brain that was fully written but never
         * renamed is rolled forward here, so the open sees the newest
         * committed state instead of the stale one. W2.2: a readonly open
         * must not write, so it neither replays nor clears the journal; it
         * reads the store exactly as it stands on disk. */
        if !opts.readonly {
            if let Some(dir) = std::path::Path::new(path).parent() {
                let report = crate::rt::wal::recover(&dir.to_string_lossy());
                if report.journal_found {
                    eprintln!(
                        "[sofuu/cma] recovered interrupted brain commit ({} rolled forward, {} rolled back)",
                        report.rolled_forward, report.rolled_back
                    );
                }
            }
        }

        /* B3: a segmented brain is a MANIFEST at <path> plus segments in
         * <path>.d — load it through the segment log (later entries win).
         * Anything else (or no file yet) falls through to the legacy
         * single-file reader below. */
        /* B4: whether the LEGACY password opened whatever we load. */
        let mut legacy_encryption = false;
        /* Three outcomes, and the difference matters: a manifest we can
         * read (segmented), a file that is NOT a manifest (legacy
         * single-file brain), and a container we cannot open (wrong key,
         * torn file). Only the middle one may fall through — treating the
         * third as absent would open a wrong-key brain as EMPTY. */
        let manifest = if !std::path::Path::new(path).exists() {
            None /* no file yet → a fresh brain */
        } else {
            match crate::rt::brain_store::read_manifest_ex(path) {
                Ok(m) => m,
                Err(rc) => {
                    eprintln!(
                        "[sofuu/cma] {path} is a container this key cannot open (rc {rc}) — refusing to open"
                    );
                    return Err(OpenError::Corrupt);
                }
            }
        };
        /* W3.3 (BRAIN-LAZY-OPEN): a segmented store's MANIFEST is the whole
         * verdict. It carries the vector space (`embedding`), the segment
         * list and the stamped live row count, so every decision an open
         * must make is made without touching a payload. The payloads are
         * read by `materialize` — now for `verify_full`, otherwise on the
         * first row access — through the SAME loader the eager open used, so
         * a lazy store and a verify_full store can never disagree. */
        if let Some((m, manifest_legacy)) = manifest {
            let dim = m
                .embedding
                .get("dimension")
                .and_then(|d| d.as_u64())
                .unwrap_or(vec_dim as u64) as usize;
            /* The manifest's own embedding block, in the exact shape the
             * legacy loader hands the vector-space check, so a segmented
             * store is judged by the same rule a single-file brain is. */
            let space_meta = serde_json::json!({
                "schema": "sofuu-cma@2",
                "embedding": m.embedding,
                "records": [],
            })
            .to_string();
            let source_manifest = embedding_manifest(&space_meta);
            let source_id = source_manifest
                .as_ref()
                .map(|mm| mm.id.as_str())
                .or_else(|| inferred_legacy_embedding_id(dim));
            let same_space = dim == vec_dim
                && source_id == Some(embedding_id.as_str())
                && match source_manifest.as_ref() {
                    Some(_) => manifest_matches(&space_meta, &embedding_id, vec_dim),
                    None => embedding_id == crate::embedding::INPUT_EMBEDDER_ID,
                };
            /* An EMPTY store has no record whose space could be violated —
             * it is adopted, exactly as an empty single-file brain is. */
            if !m.segments.is_empty() && !same_space {
                eprintln!(
                    "[sofuu/cma] incompatible brain vector space (file: {} / {:?}, requested: {} / {}) — open refused",
                    dim, source_id, vec_dim, embedding_id
                );
                return Err(OpenError::Corrupt);
            }
            shell.legacy_encryption = manifest_legacy;
            shell.layout = Some(BrainLayout {
                dir: crate::rt::brain_store::segments_dir(path),
                segments: m.segments.clone(),
                epoch: m.epoch.max(1),
                /* The stamped live row count IS what is persisted (ids are
                 * dense), and it is the same number a full scan produces.
                 * Manifests predating the field carry 0 → versions. */
                persisted: if m.records > 0 {
                    m.records as usize
                } else {
                    m.total_versions()
                },
                last_decay_at: m.last_decay_at,
                decay_tag: m.decay_tag,
                decay_seconds: m.decay_seconds,
                evicted_total: m.evicted_total,
                needs_compaction: manifest_legacy,
                recovery: Recovery::default(),
            });
            shell.loaded = false;
            shell.pending = Some(m);
            if opts.verify_full {
                materialize(&mut shell).map_err(|rc| {
                    eprintln!(
                        "[sofuu/cma] segmented brain at {path} failed to load (rc {rc}) — refusing to open"
                    );
                    OpenError::Corrupt
                })?;
            }
            return Ok(shell);
        }

        /* Hydrate from the QTSQ brain file (vectors + metadata JSON).
         * W2.2: a readonly open does not even sweep the stale `{path}.tmp`
         * a crashed writer left behind (`cma_qtsq_load_opts`). */
        let loaded = cma_qtsq_load_opts(path, !opts.readonly);
        if let Ok(b) = &loaded {
            legacy_encryption |= b.legacy_encryption;
        }
        match loaded {
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
                                return Err(OpenError::Corrupt);
                            }
                        } else {
                            eprintln!("[sofuu/cma] brain has no metadata — open refused");
                            return Err(OpenError::Corrupt);
                        }
                        /* B3: a segmented brain's decay replay and layout are
                         * handled by `materialize` (W3.3) — a segmented store
                         * never reaches this legacy branch. */
                    } else if embedding_id == crate::embedding::MODEL_ID
                        && brain.dim == crate::embedding::HASH_DIM
                        && source_id == Some(crate::embedding::INPUT_EMBEDDER_ID)
                    {
                        /* W2.2: migration is a full rewrite + a backup. A
                         * readonly handle may not do either, so it refuses
                         * instead of half-doing it. */
                        if opts.readonly {
                            eprintln!(
                                "[sofuu/cma] {path} is a legacy hash brain and a readonly open cannot migrate it — open refused (reopen writable once to migrate)"
                            );
                            return Err(OpenError::Corrupt);
                        }
                        if migrate_legacy_hash_brain(path, &brain, &mut shell).is_err() {
                            return Err(OpenError::Corrupt);
                        }
                    } else {
                        eprintln!(
                            "[sofuu/cma] incompatible brain vector space (file: {} / {:?}, requested: {} / {}) — open refused",
                            brain.dim,
                            source_id,
                            vec_dim,
                            embedding_id
                        );
                        return Err(OpenError::Corrupt);
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
                         * silently, and never clobber an existing .bak.
                         * W2.2: a readonly open writes NOTHING, so it warns
                         * and leaves the file exactly where it is. */
                        if opts.readonly {
                            eprintln!(
                                "[sofuu/cma] brain file {} is unreadable (no memories stream) — readonly open: left untouched, no backup taken",
                                path
                            );
                        } else {
                            let bak = backup_brain(path);
                            eprintln!(
                                "[sofuu/cma] brain file {} is unreadable (no memories stream) — \
                                 backed up to {} and starting fresh",
                                path, bak
                            );
                        }
                    }
                }
            }
            Err(code) => {
                /* A load that FAILED outright (garbage container, wrong
                 * magic, structurally invalid) used to fall through to a
                 * silent fresh shell — the next flush would then overwrite
                 * the user's brain file with no trace. Same rule as the
                 * empty-container branch above: back up, warn, start
                 * fresh — never wipe silently, never break chat. W2.2: a
                 * readonly open takes no backup and leaves the bytes alone. */
                if std::path::Path::new(path).exists() {
                    if opts.readonly {
                        eprintln!(
                            "[sofuu/cma] brain file {} failed to load (qtsq error {}) — readonly open: left untouched, no backup taken",
                            path, code
                        );
                    } else {
                        let bak = backup_brain(path);
                        eprintln!(
                            "[sofuu/cma] brain file {} failed to load (qtsq error {}) — \
                             backed up to {} and starting fresh",
                            path, code, bak
                        );
                    }
                }
            }
        }
        /* B4: a brain the LEGACY password opened is re-encrypted on its
         * next write (backup first) — see brain_flush_segmented. */
        shell.legacy_encryption = legacy_encryption;
        /* Nothing to defer: a fresh or legacy single-file brain has no
         * segment payloads to read later (W3.3). */
        shell.loaded = true;
        Ok(shell)
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

    /* ══════════════════════════════════════════════════════════════
     * B3 (ROADMAP-BRAIN-CMA): segmented brain store
     * ══════════════════════════════════════════════════════════════ */

    /// R-25 (AUDIT-2026-10-29): what one segment store's load had to give
    /// up. A torn segment used to abort the WHOLE brain — `segment …
    /// failed its digest check — refusing to open` → `Failed to initialize
    /// CMA` — losing every previously flushed record along with the one
    /// file in flight. The scan recovers the maximum consistent set
    /// instead, and reports exactly what it dropped.
    #[derive(Default)]
    struct SegScan {
        /// Surviving records, in dense id order (ids remapped on a gap).
        recs: Vec<crate::rt::brain_store::SegRecord>,
        /// Manifest entries for the files that survived, carrying the
        /// digest of the bytes that were actually read.
        segments: Vec<crate::rt::brain_store::SegmentMeta>,
        /// Files moved aside as `<file>.torn` (bytes preserved for salvage).
        quarantined: Vec<String>,
        /// Segments whose published digest did not match but whose bytes
        /// read cleanly — the crash window between a segment publish and the
        /// manifest publish. Adopted, not dropped.
        adopted: Vec<String>,
        /// Files the manifest lists that are not on disk at all.
        missing: Vec<String>,
        /// W2.2/W2.3: files this scan found UNUSABLE but could not QUARANTINE
        /// because the handle is readonly (renaming them would be a write).
        /// The records are still dropped for this view; the bytes stay put so
        /// a writable open (or `repair()`) can heal them properly.
        left_in_place: Vec<String>,
        /// Record VERSIONS lost with those files. An upper bound on lost
        /// memories: a later version of the same id may have survived.
        dropped_versions: usize,
        any_legacy: bool,
        /// A gap in the survivor ids forced a renumbering.
        renumbered: bool,
        /// W3.3: segment PAYLOADS this scan actually read (the number a lazy
        /// open defers and `verify_full` pays up front).
        segments_read: usize,
    }

    impl SegScan {
        fn damaged(&self) -> bool {
            !self.quarantined.is_empty()
                || !self.adopted.is_empty()
                || !self.missing.is_empty()
                || !self.left_in_place.is_empty()
                || self.renumbered
        }
    }

    /// R-25: verify and read every segment the manifest lists, quarantining
    /// what cannot be trusted. This never fails on a per-segment problem —
    /// that is the whole point: the caller opens whatever survived.
    unsafe fn scan_segments(path: &str, m: &crate::rt::brain_store::Manifest) -> SegScan {
        scan_segments_opts(path, m, false)
    }

    /// `scan_segments`, told whether this handle may write. W2.3: a readonly
    /// scan REPORTS damage but never quarantines (no rename, no fsync) — the
    /// difference between "the reader lost these rows" and "the store was
    /// changed by a reader".
    unsafe fn scan_segments_opts(
        path: &str,
        m: &crate::rt::brain_store::Manifest,
        readonly: bool,
    ) -> SegScan {
        let dir = crate::rt::brain_store::segments_dir(path);
        let mut scan = SegScan::default();
        let mut by_id: std::collections::BTreeMap<u32, crate::rt::brain_store::SegRecord> =
            std::collections::BTreeMap::new();
        let prof = std::env::var_os("SOFUU_PROFILE").is_some();
        let mut t_digest = 0f64;
        let mut t_decode = 0f64;
        let mut t_nonfinite = 0f64;
        let mut t_byid = 0f64;

        /* Parallel digest+decode pass (W3.3 perf).
         *
         * Segments are independent immutable files and the C container
         * decoder is thread-safe (verified: 8 threads over 89 real segments
         * return byte-identical results, 9.2x faster), so the sequential read
         * was leaving every core but one idle — measured at 205 of 233 ms of
         * scan time at 20k records, i.e. ~1.0 s of a 100k open.
         *
         * SAFETY VALVE: parallelism must never change a verdict. If any
         * segment in the pass reported an error, or needed the legacy
         * password, the pass is DISCARDED and the loop below reads
         * sequentially exactly as it always did. A damaged or legacy store
         * therefore takes the old path, byte for byte; only the plain
         * all-machine-key store gets the parallel one.
         */
        let mut preloaded: std::collections::HashMap<
            String,
            (Option<String>, Result<(Vec<crate::rt::brain_store::SegRecord>, bool), c_int>),
        > = {
            let files: Vec<(String, String)> = m
                .segments
                .iter()
                .map(|sg| (sg.file.clone(), format!("{dir}/{}", sg.file)))
                .filter(|(_, p)| std::path::Path::new(p).is_file())
                .collect();
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .clamp(1, 8);
            if files.len() < 8 || threads < 2 {
                std::collections::HashMap::new()
            } else {
                let chunk = files.len().div_ceil(threads).max(1);
                let collected = std::sync::Mutex::new(Vec::with_capacity(files.len()));
                std::thread::scope(|sc| {
                    for part in files.chunks(chunk) {
                        let collected = &collected;
                        sc.spawn(move || {
                            let mut local = Vec::with_capacity(part.len());
                            for (name, path) in part {
                                let digest = crate::rt::brain_store::file_digest(path);
                                let read = crate::rt::brain_store::read_segment_ex(path);
                                local.push((name.clone(), (digest, read)));
                            }
                            collected.lock().expect("segment preload").extend(local);
                        });
                    }
                });
                let map: std::collections::HashMap<_, _> =
                    collected.into_inner().expect("segment preload").into_iter().collect();
                let suspicious = map.values().any(|(_, r)| match r {
                    Err(_) => true,
                    Ok((_, legacy)) => *legacy,
                });
                if suspicious {
                    std::collections::HashMap::new()
                } else {
                    map
                }
            }
        };

        for seg in &m.segments {
            let file = format!("{dir}/{}", seg.file);
            if !std::path::Path::new(&file).exists() {
                eprintln!(
                    "[sofuu/cma] SEGMENT DAMAGE: {} is MISSING from {dir} — dropping the {} record(s) it carried, opening the rest",
                    seg.file, seg.records
                );
                scan.missing.push(seg.file.clone());
                scan.dropped_versions += seg.records;
                continue;
            }
            /* Verify the published digest BEFORE reading: the codec accepts
             * a truncated container as a valid, silently-shortened one, so
             * without this a torn segment would read as a brain that
             * quietly lost records. */
            let pre = preloaded.remove(&seg.file);
            let t0 = std::time::Instant::now();
            let digest = match &pre {
                Some((d, _)) => d.clone(),
                None => crate::rt::brain_store::file_digest(&file),
            };
            t_digest += t0.elapsed().as_secs_f64() * 1000.0;
            let verified =
                seg.sha256.is_empty() || digest.as_deref() == Some(seg.sha256.as_str());
            /* W3.3: the counter a lazy open proves itself with. Counted
             * BEFORE the read so a segment that fails still counts (it was
             * touched). */
            scan.segments_read += 1;
            let t0 = std::time::Instant::now();
            let decode = match pre {
                Some((_, r)) => r,
                None => crate::rt::brain_store::read_segment_ex(&file),
            };
            t_decode += t0.elapsed().as_secs_f64() * 1000.0;
            let (recs, legacy) = match decode {
                Ok(v) => v,
                Err(_) => {
                    if readonly {
                        /* W2.3: the reader reports the loss but leaves the
                         * bytes untouched — no rename, no fsync, nothing a
                         * writable open (or repair) would have created. */
                        eprintln!(
                            "[sofuu/cma] SEGMENT DAMAGE: {} {} — a readonly open keeps it in place ({} record(s) unavailable to this reader)",
                            seg.file,
                            if seg.sha256.is_empty() || verified {
                                "cannot be read"
                            } else {
                                "failed its digest check and cannot be read"
                            },
                            seg.records
                        );
                        scan.left_in_place.push(seg.file.clone());
                        scan.dropped_versions += seg.records;
                        continue;
                    }
                    eprintln!(
                        "[sofuu/cma] SEGMENT DAMAGE: {} {} — quarantining it ({} record(s) dropped) and opening the rest",
                        seg.file,
                        if seg.sha256.is_empty() || verified {
                            "cannot be read"
                        } else {
                            "failed its digest check and cannot be read"
                        },
                        seg.records
                    );
                    match crate::rt::wal::quarantine(&file) {
                        Some(q) => {
                            eprintln!("[sofuu/cma]   torn segment preserved as {q}");
                            scan.quarantined.push(seg.file.clone());
                        }
                        None => {
                            eprintln!(
                                "[sofuu/cma]   could not quarantine {file} — leaving it in place"
                            );
                            scan.missing.push(seg.file.clone());
                        }
                    }
                    scan.dropped_versions += seg.records;
                    continue;
                }
            };
            /* W1.1 (+W1.8): a segment is a TRUST boundary as much as the JS
             * argument is. A NaN/Inf component — written by an older build, a
             * hostile file, or a codec bug — would poison every distance in
             * the graph (cosine of NaN is NaN, and the ordering comparisons
             * all fall back to Equal), so recall would silently misbehave
             * forever. Treat such a segment exactly like an unreadable one:
             * quarantine it, account for its records, and open the rest. */
            let t0 = std::time::Instant::now();
            let nonfinite = recs.iter().position(|r| r.vec.iter().any(|x| !x.is_finite()));
            t_nonfinite += t0.elapsed().as_secs_f64() * 1000.0;
            if let Some(at) = nonfinite {
                if readonly {
                    eprintln!(
                        "[sofuu/cma] SEGMENT DAMAGE: {} carries a non-finite vector (record {}/{}) — a readonly open keeps it in place ({} record(s) unavailable to this reader)",
                        seg.file,
                        at,
                        recs.len(),
                        seg.records
                    );
                    scan.left_in_place.push(seg.file.clone());
                    scan.dropped_versions += seg.records;
                    continue;
                }
                eprintln!(
                    "[sofuu/cma] SEGMENT DAMAGE: {} carries a non-finite vector (record {}/{}) — quarantining it ({} record(s) dropped) and opening the rest",
                    seg.file,
                    at,
                    recs.len(),
                    seg.records
                );
                match crate::rt::wal::quarantine(&file) {
                    Some(q) => {
                        eprintln!("[sofuu/cma]   poisoned segment preserved as {q}");
                        scan.quarantined.push(seg.file.clone());
                    }
                    None => {
                        eprintln!("[sofuu/cma]   could not quarantine {file} — leaving it in place");
                        scan.missing.push(seg.file.clone());
                    }
                }
                scan.dropped_versions += seg.records;
                continue;
            }
            if !verified {
                /* The bytes are a VALID segment, just not the ones the
                 * manifest describes: the writer crashed between publishing
                 * the segment and publishing the manifest. Adopt the newer
                 * bytes — dropping them would throw away the very flush that
                 * was in flight. */
                eprintln!(
                    "[sofuu/cma] SEGMENT DAMAGE: {} does not match its manifest digest but reads cleanly — the manifest was STALE (crash between segment and manifest publish); adopting the {} record(s) it carries",
                    seg.file,
                    recs.len()
                );
                scan.adopted.push(seg.file.clone());
            }
            scan.any_legacy |= legacy;
            scan.segments.push(crate::rt::brain_store::SegmentMeta {
                file: seg.file.clone(),
                records: recs.len(),
                /* A legacy (digest-less) entry stays shape-verified; an
                 * adopted one is re-pinned to the bytes we just read. */
                sha256: if seg.sha256.is_empty() {
                    String::new()
                } else {
                    digest.clone().unwrap_or_else(|| seg.sha256.clone())
                },
            });
            let t0 = std::time::Instant::now();
            for r in recs {
                by_id.insert(r.id, r);
            }
            t_byid += t0.elapsed().as_secs_f64() * 1000.0;
        }
        if prof {
            eprintln!(
                "[prof] scan: digest {t_digest:.1} decode {t_decode:.1} nonfinite {t_nonfinite:.1} by_id {t_byid:.1} ms"
            );
        }

        /* Ids must be dense: the HNSW store indexes vectors by position and
         * hydrate assigns ids by position. A dropped segment can leave a
         * gap, so survivors are renumbered (co-recall partners remapped with
         * them) instead of refusing the brain. */
        let renumbered = by_id
            .keys()
            .enumerate()
            .any(|(i, id)| *id as usize != i);
        scan.renumbered = renumbered;
        let mut remap: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::with_capacity(by_id.len());
        for (i, (id, mut r)) in by_id.into_iter().enumerate() {
            if renumbered {
                remap.insert(id, i as u32);
                r.id = i as u32;
            }
            scan.recs.push(r);
        }
        if renumbered {
            eprintln!(
                "[sofuu/cma] {} record id(s) were renumbered to keep the store dense after the damage",
                scan.recs.len()
            );
            for (i, r) in scan.recs.iter_mut().enumerate() {
                if let Some(o) = r.meta.as_object_mut() {
                    if let Some(arr) = o.get("corecall_ids").and_then(|v| v.as_array()) {
                        let mapped: Vec<serde_json::Value> = arr
                            .iter()
                            .filter_map(|v| v.as_u64())
                            .filter_map(|old| remap.get(&(old as u32)).copied())
                            .map(|new| serde_json::json!(new))
                            .collect();
                        o.insert("corecall_ids".into(), serde_json::Value::Array(mapped));
                    }
                    o.insert("vector_index".into(), serde_json::json!(i));
                }
            }
        } else {
            /* Pristine path: identical to the pre-R-25 loader. */
            for (i, r) in scan.recs.iter_mut().enumerate() {
                if let Some(o) = r.meta.as_object_mut() {
                    o.insert("vector_index".into(), serde_json::json!(i));
                }
            }
        }
        scan
    }

    /// R-25: make the on-disk store describe the bytes that survived.
    ///
    /// A renumbered store is rewritten into a FRESH generation (ids must be
    /// dense, which no append can express); otherwise the surviving segment
    /// files stay and only the manifest is re-published, with digests taken
    /// from the files themselves. Both writes use the same
    /// stage → fsync → rename → fsync-dir sequence (and the intent journal)
    /// the flush path uses: a crash here leaves either the old manifest or
    /// the new one, never a half-written pair.
    ///
    /// Returns `(segments, epoch, healed)`.
    unsafe fn heal_store(
        path: &str,
        m: &crate::rt::brain_store::Manifest,
        scan: &SegScan,
    ) -> (Vec<crate::rt::brain_store::SegmentMeta>, u64, bool) {
        let dir = crate::rt::brain_store::segments_dir(path);
        let mut segments = scan.segments.clone();
        let mut epoch = m.epoch.max(1);
        if scan.renumbered && !scan.recs.is_empty() {
            let dim = scan.recs[0].vec.len();
            epoch += 1;
            match write_segment_chunks(&dir, epoch, 1, dim, &scan.recs) {
                Ok(segs) => segments = segs,
                Err(_) => {
                    eprintln!(
                        "[sofuu/cma] could not write the repaired generation for {path} — the next flush will retry"
                    );
                    return (m.segments.clone(), m.epoch.max(1), false);
                }
            }
        }
        let manifest = crate::rt::brain_store::Manifest {
            schema: crate::rt::brain_store::SEGMENT_SCHEMA.to_string(),
            embedding: m.embedding.clone(),
            segments: segments.clone(),
            epoch,
            last_decay_at: m.last_decay_at,
            evicted_total: m.evicted_total,
            records: scan.recs.len() as u64,
            decay_tag: m.decay_tag,
            decay_seconds: m.decay_seconds,
        };
        let tmp = format!("{path}.tmp");
        if crate::rt::brain_store::write_manifest(&tmp, &manifest).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return (m.segments.clone(), m.epoch.max(1), false);
        }
        let parent = std::path::Path::new(path)
            .parent()
            .map(|d| d.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        let mut journal = crate::rt::wal::Journal::begin(&parent);
        let recorded = journal
            .as_mut()
            .map(|j| j.record(&tmp, path))
            .unwrap_or(false);
        if !crate::rt::wal::publish(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return (m.segments.clone(), m.epoch.max(1), false);
        }
        if recorded {
            if let Some(j) = journal {
                j.commit();
            }
        }
        /* Superseded generations are swept only AFTER the new manifest is
         * durable (age-guarded, exactly like the flush path). */
        let keep: Vec<String> = segments.iter().map(|s| s.file.clone()).collect();
        crate::rt::brain_store::remove_orphan_segments(&dir, &keep, 600);
        eprintln!(
            "[sofuu/cma] repaired the store at {path}: {} segment(s), {} record(s) — dropped {} record version(s)",
            segments.len(),
            scan.recs.len(),
            scan.dropped_versions
        );
        (segments, epoch, true)
    }

    /// Load a segmented brain: every segment in manifest order, later
    /// entries overriding earlier ones by id (the append-only update log).
    ///
    /// R-25 (AUDIT-2026-10-29): a segment that fails verification no longer
    /// takes the brain with it. The maximum consistent set is loaded, the
    /// bad file is quarantined as `<file>.torn`, and the store is repaired on
    /// disk before it is handed back — so `Err` now means only "not a brain
    /// this build can read at all" (no manifest / wrong key).
    unsafe fn load_segmented(
        path: &str,
        m: &crate::rt::brain_store::Manifest,
    ) -> Result<(BrainData, BrainLayout), c_int> {
        load_segmented_opts(path, m, false)
    }

    /// `load_segmented`, told whether this handle may write. W2.3: in
    /// readonly mode NOTHING is repaired — no quarantine, no healed
    /// generation, no manifest re-publish — the handle opens with exactly
    /// what the bytes give it and `healed` stays false.
    unsafe fn load_segmented_opts(
        path: &str,
        m: &crate::rt::brain_store::Manifest,
        readonly: bool,
    ) -> Result<(BrainData, BrainLayout), c_int> {
        let dir = crate::rt::brain_store::segments_dir(path);
        let mut scan = scan_segments_opts(path, m, readonly);
        let (segments, epoch, healed) = if scan.damaged() {
            if readonly {
                /* Survivors only: the manifest still names the unreadable
                 * file, and this handle must not pretend it owns the store. */
                (scan.segments.clone(), m.epoch.max(1), false)
            } else {
                heal_store(path, m, &scan)
            }
        } else {
            (m.segments.clone(), m.epoch.max(1), true)
        };
        if !healed {
            eprintln!(
                "[sofuu/cma] opening {path} from a DAMAGED store in memory — {}",
                if readonly {
                    "a readonly handle repairs NOTHING (reopen writable to heal it)"
                } else {
                    "the next flush rewrites it"
                }
            );
        }
        let n = scan.recs.len();
        let renumbered_n = if scan.renumbered { n } else { 0 };
        /* MOVE the records into the flat vector + metadata arrays instead of
         * borrowing them and deep-cloning each `meta` Value. That clone copied
         * 100k JSON objects — every key and every text — and was the largest
         * single cost of a 100k open (~1.0 s of a 1.37 s load, against 191 ms
         * for every other phase put together). `flat` is also pre-sized: it
         * used to grow to 25 MB by repeated reallocation. Nothing here needs
         * `scan.recs` afterwards; only its length and the renumber flag, both
         * captured above. */
        let dim_hint = scan.recs.first().map(|r| r.vec.len()).unwrap_or(0);
        let mut flat: Vec<f32> = Vec::with_capacity(n.saturating_mul(dim_hint));
        let mut metas: Vec<serde_json::Value> = Vec::with_capacity(n);
        for r in std::mem::take(&mut scan.recs) {
            flat.extend_from_slice(&r.vec);
            metas.push(r.meta);
        }
        if !scan.quarantined.is_empty() || !scan.missing.is_empty() || !scan.left_in_place.is_empty()
        {
            eprintln!(
                "[sofuu/cma] RECOVERY: {path} opens with {n} record(s); {} torn segment(s) quarantined, {} missing, {} left in place (readonly), {} record version(s) dropped (see the .torn files beside the store)",
                scan.quarantined.len(),
                scan.missing.len(),
                scan.left_in_place.len(),
                scan.dropped_versions
            );
        }
        /* The records are NOT serialised into this document: they are already
         * parsed values and hydrate takes them directly. Only the two fields
         * that live beside them stay. */
        let meta_json = serde_json::json!({
            "schema": "sofuu-cma@2",
            "embedding": m.embedding,
        })
        .to_string();
        let layout = BrainLayout {
            dir,
            segments,
            epoch,
            persisted: n,
            last_decay_at: m.last_decay_at,
            decay_tag: m.decay_tag,
            decay_seconds: m.decay_seconds,
            evicted_total: m.evicted_total,
            /* B4: a legacy-encrypted store rewrites under the machine key on
             * its next flush — the same full-rewrite path a compaction uses,
             * so the re-encryption needs no separate machinery. A store that
             * could not be healed on disk also rides that rewrite. */
            needs_compaction: scan.any_legacy || !healed,
            /* W1.8: the damage this load found (and, when healed, repaired).
             * Counters come straight from SegScan — see Recovery's docs. */
            recovery: {
                let mut rec = Recovery::default();
                rec.quarantined = scan.quarantined.clone();
                rec.missing = scan.missing.clone();
                rec.adopted = scan.adopted.clone();
                rec.left_in_place = scan.left_in_place.clone();
                rec.records_dropped = scan.dropped_versions;
                rec.ids_renumbered = renumbered_n;
                rec.index_rebuilt = usize::from(scan.damaged());
                rec.repaired = usize::from(scan.damaged() && healed);
                rec.segments_read = scan.segments_read;
                rec
            },
        };
        Ok((
            BrainData {
                vecs: flat,
                n,
                dim: 0, /* filled by the caller from the manifest */
                meta: Some(meta_json),
                record_values: Some(metas),
                found_memories: n > 0,
                legacy_encryption: scan.any_legacy,
            },
            layout,
        ))
    }

    /// W3.3 (BRAIN-LAZY-OPEN): read the payloads a lazy open deferred.
    ///
    /// A normal `open()` reads the manifest only (`segments_read == 0`); the
    /// first ROW access — recall, remember, list, count, stats, decay, flush,
    /// compact, export … every one of them goes through `shell_from_this` or
    /// `brain_flush_segmented`, both of which call this — replays the exact
    /// loader an eager open used, so a lazy handle and a `verify_full` handle
    /// can never disagree about what the bytes say.
    ///
    /// Idempotent: the second call is a no-op. On failure `pending` is left
    /// in place so the next access retries (and reports) instead of silently
    /// degrading to an empty brain.
    unsafe fn materialize(shell: &mut CmaShell) -> Result<(), c_int> {
        if shell.loaded {
            return Ok(());
        }
        let Some(m) = shell.pending.clone() else {
            shell.loaded = true;
            return Ok(());
        };
        let path = shell.mind_path.clone();
        let readonly = shell.readonly;
        let (mut brain, layout) = load_segmented_opts(&path, &m, readonly)?;
        /* The manifest's dimension is the one the space check already
         * agreed with; `hydrate` needs it and the segment reader leaves
         * it at 0 for the caller to fill. */
        brain.dim = shell.vec_dim;
        if !brain.vecs.is_empty() && brain.n > 0 {
            let Some(meta) = brain.meta.as_deref() else {
                eprintln!("[sofuu/cma] brain file metadata unreadable — refusing the load");
                return Err(qtsq::QTSQ_ERR_FORMAT);
            };
            /* W3.1: adopt this generation's graph snapshot when there is one.
             * The topology is the expensive part of a load (one HNSW search
             * per record to rebuild), so reusing it is what makes a large
             * brain's first access cheap instead of minutes. A snapshot that
             * does not prove it matches (count/dim are checked inside) is
             * ignored and the index is rebuilt exactly as before. */
            let snapshot = match crate::rt::brain_store::graph_path(&path, layout.epoch) {
                gp => {
                    let per_record: Vec<Vec<f32>> = (0..brain.n)
                        .map(|i| brain.vecs[i * brain.dim..(i + 1) * brain.dim].to_vec())
                        .collect();
                    crate::memory::hnsw::Hnsw::load_with_vectors(&gp, per_record).ok()
                }
            };
            let hydrated = match brain.record_values {
                Some(values) => shell
                    .cma
                    .hydrate_from_values(&brain.vecs, brain.n, brain.dim, values, snapshot),
                None => shell
                    .cma
                    .hydrate_with_snapshot(&brain.vecs, brain.n, brain.dim, meta, snapshot),
            };
            if !hydrated {
                eprintln!("[sofuu/cma] brain file metadata unreadable — refusing the load");
                return Err(qtsq::QTSQ_ERR_FORMAT);
            }
        }
        /* W5.3: inherit the decay POLICY the brain was last flushed under.
         * Without this a brain configured for exponential half-life (or a
         * TTL) reopened on the default curve and silently stopped forgetting
         * on schedule — the exact failure W5.3 names. An explicit config
         * from the caller still wins: this only fills in what the open did
         * not say, which is why it runs before the replay below (so the
         * replay uses the owner's curve, not the default one). */
        if let Some(tag) = layout.decay_tag {
            let seconds = layout.decay_seconds.unwrap_or(0.0);
            shell.cma.set_decay_params(tag, seconds);
        }
        /* B3: a segmented brain persists strengths as of the last flush and
         * replays the decay that happened while the process was down —
         * capped at 7 days, the same bound the turn-boundary tick uses.
         * In-RAM only, so a readonly handle replays it identically. */
        if layout.last_decay_at > 0 {
            let now = now_u32() as u64;
            let dt = now.saturating_sub(layout.last_decay_at);
            if dt > 0 {
                shell.cma.decay_tick(dt.min(7 * 86400) as u32);
            }
        }
        shell.segments_read = layout.recovery.segments_read;
        shell.recovery_pending = true;
        shell.pending = None;
        shell.loaded = true;
        shell.layout = Some(layout);
        Ok(())
    }

    /// W1.8 + W3.3: deliver the events a load produced. A lazy open has no
    /// ctx to report from at open time, so the access that triggered the
    /// load flushes them through here.
    unsafe fn flush_pending_recovery(ctx: *mut JSContext, shell: &mut CmaShell) {
        if !shell.recovery_pending {
            return;
        }
        shell.recovery_pending = false;
        let rec = shell
            .layout
            .as_ref()
            .map(|l| l.recovery.clone())
            .unwrap_or_default();
        let path = shell.mind_path.clone();
        let records = shell.cma.len();
        emit_recovery_events(ctx, &rec, &path, records);
    }

    /// R-25 (AUDIT-2026-10-29): the repair pass behind
    /// `rt::memory::repair(path)`. It runs the SAME scan + heal the open
    /// path runs — one implementation, so a CLI repair and a resilient open
    /// can never disagree about what "recoverable" means — and reports what
    /// it found instead of only logging it.
    pub fn repair_brain(path: &str) -> super::RepairReport {
        let mut report = super::RepairReport::default();
        unsafe {
            let manifest = match crate::rt::brain_store::read_manifest_ex(path) {
                Ok(Some((m, _))) => m,
                Ok(None) => {
                    report.message = format!(
                        "{path}: not a segmented brain (absent, legacy single-file, or a foreign file) — nothing to repair"
                    );
                    return report;
                }
                Err(rc) => {
                    report.message = format!(
                        "{path}: a container this key cannot open (rc {rc}) — the bytes are unreadable, so there is nothing to repair and the file is left untouched"
                    );
                    return report;
                }
            };
            report.found = true;
            let scan = scan_segments(path, &manifest);
            report.records = scan.recs.len();
            report.quarantined = scan.quarantined.clone();
            report.adopted = scan.adopted.clone();
            report.missing = scan.missing.clone();
            report.records_dropped = scan.dropped_versions;
            let (segments, _epoch, healed) = if scan.damaged() {
                heal_store(path, &manifest, &scan)
            } else {
                (manifest.segments.clone(), manifest.epoch.max(1), true)
            };
            report.segments = segments.len();
            report.rewritten = scan.damaged() && healed;
            report.ok = healed;
            report.message = if !scan.damaged() {
                format!(
                    "{path}: healthy — {} segment(s), {} record(s)",
                    segments.len(),
                    scan.recs.len()
                )
            } else if healed {
                format!(
                    "{path}: repaired — {} segment(s), {} record(s) kept, {} quarantined, {} record version(s) dropped",
                    segments.len(),
                    scan.recs.len(),
                    scan.quarantined.len(),
                    scan.dropped_versions
                )
            } else {
                format!(
                    "{path}: damaged and NOT repaired on disk (see the warnings above) — opening still recovers {} record(s) in memory",
                    scan.recs.len()
                )
            };
        }
        report
    }

    /// Write `recs` as one or more bounded segments, each published
    /// atomically (tmp → fsync → rename → fsync dir).
    unsafe fn write_segment_chunks(
        dir: &str,
        epoch: u64,
        start_seq: u64,
        dim: usize,
        recs: &[crate::rt::brain_store::SegRecord],
    ) -> Result<Vec<crate::rt::brain_store::SegmentMeta>, c_int> {
        let mut out = Vec::new();
        for (i, chunk) in recs.chunks(crate::rt::brain_store::SEGMENT_CAP).enumerate() {
            let name = crate::rt::brain_store::segment_file(epoch, start_seq + i as u64);
            let final_path = format!("{dir}/{name}");
            let tmp = format!("{final_path}.tmp");
            crate::rt::brain_store::write_segment(&tmp, dim, chunk)?;
            /* Digest the staged file (the bytes that will be renamed in). */
            let sha256 = crate::rt::brain_store::file_digest(&tmp).unwrap_or_default();
            if !crate::rt::wal::publish(&tmp, &final_path) {
                let _ = std::fs::remove_file(&tmp);
                return Err(qtsq::QTSQ_ERR_FORMAT);
            }
            out.push(crate::rt::brain_store::SegmentMeta {
                file: name,
                records: chunk.len(),
                sha256,
            });
        }
        Ok(out)
    }

    /// B3.3: capacity policy. Over the cap, the WEAKEST live records are
    /// tombstoned (never pins, never entities) — oldest first among equals.
    /// Tombstoning rather than removing keeps ids stable, so a caller
    /// holding recall ids across the flush cannot be handed a wrong record;
    /// retain() physically drops them at the next turn boundary.
    unsafe fn enforce_brain_capacity(shell: &mut CmaShell, layout: &mut BrainLayout) -> usize {
        let cap = brain_max_records();
        if cap == 0 {
            return 0;
        }
        let live = shell
            .cma
            .records
            .iter()
            .filter(|r| r.tier != crate::memory::cma::TIER_FORGOTTEN)
            .count();
        if live <= cap {
            return 0;
        }
        let over = live - cap;
        let mut candidates: Vec<(u32, f32, u32)> = shell
            .cma
            .records
            .iter()
            .filter(|r| {
                r.tier != crate::memory::cma::TIER_FORGOTTEN
                    && r.role != crate::memory::cma::PIN_ROLE
                    && r.tier != crate::memory::cma::TIER_ENTITY
            })
            .map(|r| (r.id, r.strength, r.created_at))
            .collect();
        /* Weakest first; among equals the oldest (smallest created_at, with
         * 0 = unknown sorting first). */
        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.2.cmp(&b.2))
        });
        let mut evicted = 0usize;
        for (id, _, _) in candidates.into_iter().take(over) {
            if shell.cma.forget(id) {
                evicted += 1;
            }
        }
        if evicted > 0 {
            layout.evicted_total += evicted as u64;
            eprintln!(
                "[sofuu/cma] capacity: tombstoned {evicted} weak record(s) over the {cap}-record cap"
            );
        }
        evicted
    }

    /// Rewrite every live record into fresh segments (B3.4). Ids are
    /// preserved (records are written in id order, one version each), so
    /// recall and any outstanding id references stay valid.
    unsafe fn compact_segments(
        shell: &mut CmaShell,
        layout: &mut BrainLayout,
    ) -> (c_int, FlushStats) {
        let dim = shell.vec_dim;
        let recs: Vec<crate::rt::brain_store::SegRecord> = (0..shell.cma.len() as u32)
            .map(|id| seg_record(shell, id))
            .collect();
        layout.epoch += 1;
        match write_segment_chunks(&layout.dir, layout.epoch, 1, dim, &recs) {
            Ok(segs) => {
                let stats = FlushStats {
                    records_written: recs.len(),
                    segments_sealed: segs.len(),
                };
                layout.segments = segs;
                layout.persisted = recs.len();
                layout.needs_compaction = false;
                (0, stats)
            }
            Err(_) => (-1, FlushStats::default()),
        }
    }

    /// R-25: the next free segment sequence for the current generation —
    /// one past the highest the manifest lists AND one past anything the
    /// directory already holds, so a crashed writer's orphan segment is
    /// swept later instead of being silently overwritten.
    fn next_segment_seq(
        dir: &str,
        segments: &[crate::rt::brain_store::SegmentMeta],
    ) -> u64 {
        fn seq_of(name: &str) -> Option<u64> {
            let body = name.strip_prefix("seg-")?.strip_suffix(".qtsq")?;
            let (_epoch, seq) = body.split_once('-')?;
            seq.parse::<u64>().ok()
        }
        let mut max = segments
            .iter()
            .filter_map(|s| seq_of(&s.file))
            .max()
            .unwrap_or(0);
        for name in crate::rt::brain_store::list_segment_files(dir) {
            if let Some(n) = seq_of(&name) {
                max = max.max(n);
            }
        }
        max + 1
    }

    /// B3.2: append only what changed — new records plus the dirty set — to
    /// a NEW segment. Everything else on disk is untouched.
    ///
    /// R-25 (AUDIT-2026-10-29): a published segment file is IMMUTABLE. The
    /// old path rewrote the active segment in place, which left the manifest
    /// carrying a stale digest for the whole window between the segment
    /// rename and the manifest rename — a crash there produced a manifest
    /// that could never verify its own segment, and (before this audit) a
    /// brain that refused to open AT ALL: every previously flushed record
    /// gone with the one file in flight (5 of 21 SIGKILL reps on the Pi).
    /// Appending a new file instead means a crash mid-flush leaves at worst
    /// an unreferenced orphan segment for the age-guarded sweep, and the
    /// committed manifest keeps describing bytes that still exist — so a
    /// committed manifest can never reference a torn segment.
    unsafe fn append_segments(
        shell: &mut CmaShell,
        layout: &mut BrainLayout,
    ) -> (c_int, FlushStats) {
        let dim = shell.vec_dim;
        let mut ids: Vec<u32> = (layout.persisted..shell.cma.len())
            .map(|i| i as u32)
            .collect();
        for id in shell.cma.take_dirty() {
            if (id as usize) >= layout.persisted {
                continue; /* already covered as new */
            }
            ids.push(id);
        }
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            /* Nothing changed: a successful flush that wrote nothing. */
            return (0, FlushStats::default());
        }
        let fresh: Vec<crate::rt::brain_store::SegRecord> =
            ids.iter().map(|id| seg_record(shell, *id)).collect();
        let seq = next_segment_seq(&layout.dir, &layout.segments);
        match write_segment_chunks(&layout.dir, layout.epoch, seq, dim, &fresh) {
            Ok(segs) => {
                let stats = FlushStats {
                    records_written: fresh.len(),
                    segments_sealed: segs.len(),
                };
                layout.segments.extend(segs);
                (0, stats)
            }
            Err(_) => (-1, FlushStats::default()),
        }
    }

    /// B3: the canonical flush — segmented store at `<path>`.
    unsafe fn brain_flush_segmented(shell: &mut CmaShell) -> c_int {
        /* W2.2: a readonly handle holds a SHARED lock and writes nothing. */
        if shell.readonly {
            let path = shell.mind_path.clone();
            shell.note_flush_failure(
                E_LOCKED,
                format!(
                    "the brain at {path} is open readonly (shared lock) — flush refused, nothing was written"
                ),
            );
            return -1;
        }
        if shell.closed {
            let path = shell.mind_path.clone();
            shell.note_flush_failure(
                E_LOCKED,
                format!("the brain at {path} was closed — its lock is released, so this handle cannot write it"),
            );
            return -1;
        }
        /* W3.3: a LAZY handle must materialise BEFORE it rewrites anything —
         * flushing a not-yet-loaded shell would publish an empty store over
         * the real one. */
        if let Err(rc) = materialize(shell) {
            let path = shell.mind_path.clone();
            shell.note_flush_failure(
                E_CORRUPT,
                format!("the brain store at {path} could not be read (qtsq error {rc}) — flush refused"),
            );
            return -1;
        }
        let path = shell.mind_path.clone();
        let dir = crate::rt::brain_store::segments_dir(&path);
        if std::fs::create_dir_all(&dir).is_err() {
            shell.note_flush_failure(
                E_ENOSPC,
                format!("could not create the segment directory {dir} (disk full or unwritable)"),
            );
            return -1;
        }
        let mut layout = match shell.layout.take() {
            Some(l) => l,
            None => {
                /* First segmented flush. A legacy single-file brain is
                 * preserved as a backup before the store moves on. */
                if std::path::Path::new(&path).exists() {
                    let bak = unique_sidecar(&path, "segmented.bak");
                    if std::fs::copy(&path, &bak).is_ok() {
                        eprintln!("[sofuu/cma] migrating to segmented storage (backup: {bak})");
                    } else {
                        eprintln!("[sofuu/cma] refusing to migrate: could not back up {path}");
                        shell.note_flush_failure(
                            E_ENOSPC,
                            format!("could not back {path} up before the segmented migration (disk full or unwritable)"),
                        );
                        return -1;
                    }
                }
                BrainLayout {
                    dir: dir.clone(),
                    segments: Vec::new(),
                    epoch: 1,
                    persisted: 0,
                    last_decay_at: 0,
                    decay_tag: None,
                    decay_seconds: None,
                    evicted_total: 0,
                    needs_compaction: true,
                    recovery: Recovery::default(),
                }
            }
        };
        layout.dir = dir.clone();

        /* P1 (audit 2026-10-14): writer arbitration. The manifest is the
         * only serialization point between processes. Two writers, each
         * appending their own segment and publishing their own manifest,
         * would strand the loser's records as unreferenced files — which
         * the age-guarded orphan sweep then DELETES (silent record loss).
         * The store is single-writer by contract; a manifest that moved
         * under this shell (any epoch bump, or a segment it has never
         * seen) means another live writer: refuse, keep this shell's
         * state in memory, and let the operator reopen to adopt. */
        let stamp = std::fs::metadata(&path)
            .ok()
            .map(|md| (md.len(), md.modified().ok()));
        let unchanged = stamp.is_some() && stamp == shell.published_manifest_stamp;
        if let (false, Ok(Some(disk))) = (unchanged, crate::rt::brain_store::read_manifest_ex(&path)) {
            let moved = disk.0.epoch != layout.epoch
                || disk
                    .0
                    .segments
                    .iter()
                    .any(|s| !layout.segments.iter().any(|mine| mine.file == s.file));
            shell.published_manifest_stamp = stamp;
            if moved {
                eprintln!(
                    "[sofuu/cma] brain store at {path} moved under this process (another writer?) — \
                     flush refused to avoid dropping its records; reopen to adopt them"
                );
                shell.note_flush_failure(
                    E_LOCKED,
                    format!(
                        "the brain store at {path} moved under this process (another writer?) —                          flush refused to avoid dropping its records; reopen to adopt them"
                    ),
                );
                shell.layout = Some(layout);
                return -1;
            }
        }

        /* B4: a brain the LEGACY password opened is re-encrypted under the
         * machine key. That is a full rewrite (every segment), so it rides
         * the compaction path; the old generation stays on disk as the
         * rollback copy until the age-guarded orphan sweep, and the
         * manifest itself is preserved beside it. */
        if shell.legacy_encryption {
            let bak = unique_sidecar(&path, "legacy-key.bak");
            if std::fs::copy(&path, &bak).is_ok() {
                eprintln!(
                    "[sofuu/cma] re-encrypting a legacy-key brain under the machine key (backup: {bak})"
                );
            } else {
                eprintln!("[sofuu/cma] refusing to re-encrypt: could not back up {path}");
                shell.note_flush_failure(
                    E_ENOSPC,
                    format!("could not back {path} up before re-encrypting it (disk full or unwritable)"),
                );
                shell.layout = Some(layout);
                return -1;
            }
            layout.needs_compaction = true;
        }

        /* A structural change the incremental path cannot express (retain()
         * renumbered ids, an explicit cma.compact(), an eviction) forces a
         * full rewrite — the Cma flag is the source of truth, the layout
         * mirrors it. */
        if shell.cma.layout_dirty {
            layout.needs_compaction = true;
        }
        if enforce_brain_capacity(shell, &mut layout) > 0 {
            shell.cma.layout_dirty = true;
            layout.needs_compaction = true;
        }

        let live = shell.cma.len();
        let versions: usize = layout.segments.iter().map(|s| s.records).sum();
        let dead_ratio = versions > live.saturating_mul(2) + 64;
        /* R-25: append-only publishing trades segment count for crash
         * safety (a published file is never rewritten), so bound what an
         * open has to read: compact once the store carries more than twice
         * the segments a dense layout needs, plus slack for a small brain
         * that flushes every turn. */
        let minimal = live / crate::rt::brain_store::SEGMENT_CAP + 1;
        let too_many_segments = layout.segments.len() > minimal * 2 + 8;
        let p_on = std::env::var_os("SOFUU_PROFILE").is_some();
        let p_t = std::time::Instant::now();
        let (rc, stats) = if layout.needs_compaction || dead_ratio || too_many_segments {
            compact_segments(shell, &mut layout)
        } else {
            append_segments(shell, &mut layout)
        };
        if rc != 0 {
            shell.note_flush_failure(
                E_ENOSPC,
                format!(
                    "could not write the segments for {path} (codec error {rc}: disk full or an                      unwritable segment directory)"
                ),
            );
            shell.layout = Some(layout);
            return rc;
        }

        let p_write = p_t.elapsed().as_secs_f64() * 1000.0;
        let p_t = std::time::Instant::now();
        layout.last_decay_at = now_u32() as u64;
        let manifest = crate::rt::brain_store::Manifest {
            schema: crate::rt::brain_store::SEGMENT_SCHEMA.to_string(),
            embedding: embedding_block(shell),
            segments: layout.segments.clone(),
            epoch: layout.epoch,
            last_decay_at: layout.last_decay_at,
            /* W5.3: the policy this flush is being written under, so a
             * reopen with a silent config inherits it instead of reverting
             * to the default curve. */
            decay_tag: Some(shell.cma.decay_params().0),
            decay_seconds: Some(shell.cma.decay_params().1),
            evicted_total: layout.evicted_total,
            /* The live row count (tombstones included) — the number a
             * fresh open's count() reports, not total_versions(). */
            records: live as u64,
        };
        let tmp = format!("{path}.tmp");
        if crate::rt::brain_store::write_manifest(&tmp, &manifest).is_err() {
            let _ = std::fs::remove_file(&tmp);
            shell.note_flush_failure(
                E_ENOSPC,
                format!("could not stage the manifest for {path} (disk full or unwritable)"),
            );
            shell.layout = Some(layout);
            return -1;
        }
        let parent = std::path::Path::new(&path)
            .parent()
            .map(|d| d.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        let mut journal = crate::rt::wal::Journal::begin(&parent);
        let recorded = journal
            .as_mut()
            .map(|j| j.record(&tmp, &path))
            .unwrap_or(false);
        if crate::rt::wal::publish(&tmp, &path) {
            if recorded {
                if let Some(j) = journal {
                    j.commit();
                }
            }
            shell.published_manifest_stamp = std::fs::metadata(&path)
                .ok()
                .map(|md| (md.len(), md.modified().ok()));
        } else {
            let _ = std::fs::remove_file(&tmp);
            shell.note_flush_failure(
                E_ENOSPC,
                format!("could not publish the manifest for {path} (the rename failed)"),
            );
            shell.layout = Some(layout);
            return -1;
        }

        /* W3.1: persist the graph for THIS generation, beside the segments.
         * Written after the manifest is durable, so a snapshot can never
         * describe a generation the manifest does not reference; a failure
         * here is not fatal (the load simply rebuilds).
         *
         * NOT on every flush: the snapshot is the whole graph (28.8 MB at
         * 100k records), so rewriting it for each 256-record append turns an
         * O(append) flush into O(store). It is written when the generation
         * changed (a compaction invalidates the topology anyway) or when the
         * graph has grown enough that the stale snapshot stopped being a good
         * approximation — which keeps the write count logarithmic in the
         * store's growth instead of linear in the flush count. */
        let records_now = shell.cma.len();
        let stale = match shell.graph_snapshot_records {
            None => true,
            Some(n) => layout.epoch != shell.graph_snapshot_epoch
                || n == 0
                || records_now >= n.saturating_mul(4),
        };
        let p_sweep0 = p_t.elapsed().as_secs_f64() * 1000.0;
        let p_t = std::time::Instant::now();
        if stale {
            let graph = crate::rt::brain_store::graph_path(&path, layout.epoch);
            match shell.cma.index.save_to(&graph) {
                Ok(()) => {
                    shell.graph_snapshot_epoch = layout.epoch;
                    shell.graph_snapshot_records = Some(records_now);
                }
                Err(e) => eprintln!(
                    "[sofuu/cma] graph snapshot not written ({e:?}) — the next open rebuilds it"
                ),
            }
        }

        let p_stage = p_t.elapsed().as_secs_f64() * 1000.0;
        let p_pub = p_t.elapsed().as_secs_f64() * 1000.0;
        let p_t = std::time::Instant::now();
        /* Sweep segments a crashed compaction left behind (age-guarded), and
         * reclaim superseded generations (W3.4), but only when the store could
         * actually hold garbage: a new epoch (compaction or re-seal) is the
         * only thing that supersedes a file, and a crash leftover is caught by
         * the periodic sweep below. Doing both sweeps on every append-only
         * flush cost two directory scans and a parse of every segment name.
         */
        shell.flushes_since_sweep = shell.flushes_since_sweep.saturating_add(1);
        let sweep = shell.last_sweep_epoch != Some(layout.epoch) || shell.flushes_since_sweep >= 64;
        let mut keep: Vec<String> = Vec::new();
        let mut reclaimed = 0usize;
        if sweep {
            shell.last_sweep_epoch = Some(layout.epoch);
            shell.flushes_since_sweep = 0;
            keep = layout.segments.iter().map(|s| s.file.clone()).collect();
            crate::rt::brain_store::remove_orphan_segments(&dir, &keep, 600);
        }

        /* W3.4: reclaim SUPERSEDED generations, so the store does not grow
         * with every flush. Without this, compaction and re-sealing leave
         * each old generation on disk forever (the 600 s orphan cliff the
         * plan measured): amplification climbs without bound until someone
         * runs a manual repair.
         *
         * `keep` is every generation still reachable — the manifest's own
         * epoch plus the generation of each listed segment.
         *
         * This reclaims IMMEDIATELY (min_age 0) rather than behind the 600 s
         * orphan guard, and that is safe for a precise reason: we get here
         * holding LOCK_EX on the brain. flock is exclusive against shared
         * readers too, so no other process can be reading an old generation
         * or publishing a new one — the guard exists for the crashed-writer
         * case, which the manifest we just published has already resolved. */
        if sweep {
            let mut keep_gens: Vec<u64> = Vec::with_capacity(layout.segments.len() + 1);
            keep_gens.push(layout.epoch);
            for st in &layout.segments {
                if let Some(g) = crate::rt::brain_store::generation_of(&st.file) {
                    if !keep_gens.contains(&g) {
                        keep_gens.push(g);
                    }
                }
            }
            reclaimed = crate::rt::brain_store::reclaim_unreferenced_now(&dir, &keep_gens, 0);
        }
        if reclaimed > 0 {
            eprintln!(
                "[sofuu/cma] reclaimed {reclaimed} superseded store file(s) (W3.4)"
            );
        }

        if p_on {
            eprintln!(
                "[prof] flush: store={p_write:.1} manifest_build={p_stage:.1} publish={p_pub:.1} snapshot={p_sweep0:.1} tail={:.1}",
                p_t.elapsed().as_secs_f64() * 1000.0
            );
        }
        shell.cma.mark_persisted();
        layout.persisted = live;
        shell.layout = Some(layout);
        /* The rewrite above used the machine key — the flag's job is done. */
        shell.legacy_encryption = false;
        /* W6.1: the numbers flush() reports. */
        shell.note_flush_success(stats);
        0
    }

    unsafe fn cma_flush(shell: &mut CmaShell) -> c_int {
        brain_flush_segmented(shell)
    }

    unsafe fn cma_remember(
        shell: &mut CmaShell,
        vec: &[f32],
        text: Option<&str>,
        role: Option<&str>,
        kv_page_id: u32,
        source: Option<&str>,
    ) -> c_int {
        let Some(text) = text else {
            return -1;
        };
        shell
            .cma
            .remember_with_source(vec, text, role.unwrap_or("unknown"), kv_page_id, source.unwrap_or(""))
    }


    /// W5.2: the same ingest path as `cma_remember`, but returning the
    /// merge/insert verdict the plan asks callers to be told about —
    /// `{ id, merged, into }`. Nothing about the write changes.
    unsafe fn cma_remember_outcome(
        shell: &mut CmaShell,
        vec: &[f32],
        text: Option<&str>,
        role: Option<&str>,
        kv_page_id: u32,
        source: Option<&str>,
    ) -> Option<crate::memory::cma::RememberOutcome> {
        let text = text?;
        Some(shell.cma.remember_with_source_outcome(
            vec,
            text,
            role.unwrap_or("unknown"),
            kv_page_id,
            source.unwrap_or(""),
        ))
    }

    /// Turn a `RememberOutcome` into the JS object W5.2 specifies. `into` is
    /// absent on an insert (not `null`-valued-but-present) so `"into" in r`
    /// is the test a caller wants.
    unsafe fn remember_outcome_value(
        ctx: *mut JSContext,
        outcome: Option<crate::memory::cma::RememberOutcome>,
    ) -> qjs::JSValue {
        let obj = qjs::sofuu_js_new_object(ctx);
        let (id, merged, into) = match outcome {
            Some(o) => (o.id, o.merged, o.into),
            None => (u32::MAX, false, None),
        };
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"id".as_ptr(),
            qjs::sofuu_js_new_int32(ctx, id as i32),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"merged".as_ptr(),
            qjs::sofuu_js_new_bool(ctx, merged as i32),
        );
        if let Some(into) = into {
            qjs::sofuu_js_set_property_str(
                ctx,
                obj,
                c"into".as_ptr(),
                qjs::sofuu_js_new_int32(ctx, into as i32),
            );
        }
        obj
    }

    /// recall → JSON array string (same shape ffi_exports.rs documents:
    /// [{id, distance, score, role, text, tier, strength, entity}] —
    /// `entity` is the namespace for TIER_ENTITY records (agent memory
    /// scopes), null for ordinary memories).
    unsafe fn cma_recall_json(shell: &mut CmaShell, query: &[f32], top_k: usize) -> String {
        let hits = shell.cma.recall(query, top_k);
        cma_hits_json(&hits)
    }

    /// M1.3: recall with the query-token re-rank — same JSON shape, the
    /// order (and scores) carry the rarity boost.
    unsafe fn cma_recall_text_json(
        shell: &mut CmaShell,
        query: &[f32],
        top_k: usize,
        text: &str,
    ) -> String {
        let hits = shell.cma.recall_tokens(query, top_k, text);
        cma_hits_json(&hits)
    }

    unsafe fn cma_hits_json(hits: &[crate::memory::cma::RecallHit]) -> String {
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

    enum F32View {
        Borrowed(*const f32, usize),
        Owned(Vec<f32>),
    }

    impl F32View {
        fn as_slice(&self) -> &[f32] {
            match self {
                F32View::Borrowed(p, n) => {
                    if p.is_null() || *n == 0 {
                        &[]
                    } else {
                        unsafe { std::slice::from_raw_parts(*p, *n) }
                    }
                }
                F32View::Owned(v) => v.as_slice(),
            }
        }
    }

    /* ══════════════════════════════════════════════════════════════
     * W1.7 (AUDIT-2026-10-29): coded errors on the JS memory surface.
     *
     * Every refusal this surface can produce carries `.code` from one fixed
     * set, so a host can branch on the reason instead of parsing English out
     * of a message (the C-ABI used to flatten them all to `js_exception`).
     * The thrown objects are REAL JS error instances (TypeError / RangeError
     * / Error) with `.code` added, so `instanceof` keeps working; memory_api.js
     * uses the same {ctor, code} pairing in its own `refuse()` helper.
     * ══════════════════════════════════════════════════════════════ */
    pub(crate) const E_TYPE: &str = "E_TYPE";
    pub(crate) const E_NONFINITE: &str = "E_NONFINITE";
    pub(crate) const E_DIM: &str = "E_DIM";
    pub(crate) const E_LOCKED: &str = "E_LOCKED";
    pub(crate) const E_CORRUPT: &str = "E_CORRUPT";
    pub(crate) const E_ENOSPC: &str = "E_ENOSPC";

    /// A CString that never silently empties: an interior NUL is escaped
    /// (the C API cannot carry one, and an empty message would be a lie).
    fn cstr_lossy(s: &str) -> CString {
        CString::new(s.replace('\0', "\\0")).unwrap_or_default()
    }

    /// The JS error class a code maps to.
    fn error_ctor_for_code(code: &str) -> &'static std::ffi::CStr {
        match code {
            E_TYPE | E_DIM => c"TypeError",
            E_NONFINITE => c"RangeError",
            _ => c"Error",
        }
    }

    /// Throw a real JS error of the mapped class carrying `.code` (and an
    /// explicit `.message`, which the plain-Error fallback would lack).
    unsafe fn throw_coded(ctx: *mut JSContext, code: &str, msg: &str) -> JSValue {
        let ctor_name = error_ctor_for_code(code);
        let global = qjs::sofuu_js_get_global_object(ctx);
        let ctor = if qjs::is_exception(global) {
            qjs::sofuu_js_exception()
        } else {
            qjs::sofuu_js_get_property_str(ctx, global, ctor_name.as_ptr())
        };
        if !qjs::is_exception(global) {
            qjs::sofuu_js_free_value(ctx, global);
        }
        let msg_c = cstr_lossy(msg);
        let msg_v = qjs::sofuu_js_new_string(ctx, msg_c.as_ptr());
        let err = if !qjs::is_exception(ctor) && qjs::JS_IsFunction(ctx, ctor) != 0 {
            let e = qjs::JS_CallConstructor(ctx, ctor, 1, &msg_v);
            if qjs::is_exception(e) {
                /* Constructing the error threw (hostile global): drop that
                 * exception and fall back — the .code still lands. */
                let exc = qjs::sofuu_js_get_exception(ctx);
                qjs::sofuu_js_free_value(ctx, exc);
                qjs::JS_NewError(ctx)
            } else {
                e
            }
        } else {
            qjs::JS_NewError(ctx)
        };
        if !qjs::is_exception(ctor) {
            qjs::sofuu_js_free_value(ctx, ctor);
        }
        if qjs::is_exception(err) {
            qjs::sofuu_js_free_value(ctx, msg_v);
            return err;
        }
        let code_v = qjs::sofuu_js_new_string(ctx, cstr_lossy(code).as_ptr());
        qjs::sofuu_js_set_property_str(ctx, err, c"code".as_ptr(), code_v);
        qjs::sofuu_js_set_property_str(
            ctx,
            err,
            c"message".as_ptr(),
            qjs::sofuu_js_new_string(ctx, msg_c.as_ptr()),
        );
        qjs::sofuu_js_free_value(ctx, msg_v);
        qjs::JS_Throw(ctx, err)
    }

    fn dim_mismatch_message(op: &str, got: usize, want: usize) -> String {
        format!(
            "memory.{op}: vector has {got} dimensions but this brain stores {want} — the \
             dimension is exact (a truncated or padded vector would silently score garbage)"
        )
    }

    /// W1.1: first non-finite element (NaN / ±Infinity), if any — ONE linear
    /// pass per vector at the boundary every ingest and query crosses.
    fn first_non_finite(v: &[f32]) -> Option<(usize, f32)> {
        v.iter()
            .enumerate()
            .find(|(_, x)| !x.is_finite())
            .map(|(i, x)| (i, *x))
    }

    fn non_finite_message(at: usize, val: f32) -> String {
        format!(
            "memory: vector element {at} is not finite ({val}) — NaN/Infinity must never reach \
             the graph or the persisted store; replace or drop the element"
        )
    }

    /// W1.8 (AUDIT-2026-10-29): deliver a damage event to
    /// `sofuu.memory.onEvent(cb)` handlers.
    ///
    /// The registry and the dispatch loop live in JS (src/js/memory_api.js
    /// installs `globalThis.__sofuu_memory_event`), so callbacks are ordinary
    /// JS values owned by the GC — the Rust side never holds a JSValue across
    /// a context teardown. A build/host without the flat API simply gets no
    /// event (the hook is absent), and a handler that throws cannot break the
    /// store: the exception is swallowed here.
    unsafe fn emit_memory_event(ctx: *mut JSContext, event: serde_json::Value) {
        let payload = cstr_lossy(&event.to_string());
        let global = qjs::sofuu_js_get_global_object(ctx);
        if qjs::is_exception(global) {
            let exc = qjs::sofuu_js_get_exception(ctx);
            qjs::sofuu_js_free_value(ctx, exc);
            return;
        }
        let hook = qjs::sofuu_js_get_property_str(ctx, global, c"__sofuu_memory_event".as_ptr());
        qjs::sofuu_js_free_value(ctx, global);
        if qjs::is_exception(hook) {
            let exc = qjs::sofuu_js_get_exception(ctx);
            qjs::sofuu_js_free_value(ctx, exc);
            return;
        }
        if qjs::JS_IsFunction(ctx, hook) != 0 {
            let arg = qjs::sofuu_js_new_string(ctx, payload.as_ptr());
            let ret = qjs::JS_Call(ctx, hook, qjs::sofuu_js_undefined(), 1, &arg);
            if qjs::is_exception(ret) {
                /* A handler throwing must never break the store. */
                let exc = qjs::sofuu_js_get_exception(ctx);
                qjs::sofuu_js_free_value(ctx, exc);
            } else {
                qjs::sofuu_js_free_value(ctx, ret);
            }
            qjs::sofuu_js_free_value(ctx, arg);
        }
        qjs::sofuu_js_free_value(ctx, hook);
    }

    /// W1.8: one event per REAL damage fact a load found (nothing invented —
    /// every field comes from the loader's own SegScan), then the rebuild.
    unsafe fn emit_recovery_events(
        ctx: *mut JSContext,
        rec: &Recovery,
        path: &str,
        records: usize,
    ) {
        for file in &rec.quarantined {
            emit_memory_event(
                ctx,
                serde_json::json!({"type": "segment_quarantined", "file": file, "path": path}),
            );
        }
        for file in &rec.missing {
            emit_memory_event(
                ctx,
                serde_json::json!({"type": "segment_missing", "file": file, "path": path}),
            );
        }
        for file in &rec.adopted {
            emit_memory_event(
                ctx,
                serde_json::json!({"type": "segment_adopted", "file": file, "path": path}),
            );
        }
        if rec.records_dropped > 0 {
            emit_memory_event(
                ctx,
                serde_json::json!({"type": "record_dropped", "count": rec.records_dropped, "path": path}),
            );
        }
        if rec.ids_renumbered > 0 {
            emit_memory_event(
                ctx,
                serde_json::json!({"type": "ids_renumbered", "count": rec.ids_renumbered, "path": path}),
            );
        }
        if rec.index_rebuilt > 0 {
            emit_memory_event(
                ctx,
                serde_json::json!({"type": "index_rebuilt", "records": records, "path": path}),
            );
        }
        if rec.repaired > 0 {
            emit_memory_event(
                ctx,
                serde_json::json!({"type": "store_repaired", "records": records, "path": path}),
            );
        }
    }

    /// W1.3 (AUDIT-2026-10-29): is this value a real Float32Array?
    ///
    /// `JS_GetTypedArrayBuffer` succeeds for EVERY ArrayBuffer view, so the
    /// old bridge re-interpreted a Float64Array's bytes as f32 (a 768-element
    /// Float64Array read as 1536 garbage floats, or a misleading "dimension
    /// mismatch"). `JS_GetTypedArrayType` is not exported by this QuickJS, so
    /// `JS_IsInstanceOf` against the realm's Float32Array is the check.
    unsafe fn is_float32_array(ctx: *mut JSContext, v: JSValueConst) -> bool {
        let global = qjs::sofuu_js_get_global_object(ctx);
        if qjs::is_exception(global) {
            let exc = qjs::sofuu_js_get_exception(ctx);
            qjs::sofuu_js_free_value(ctx, exc);
            return false;
        }
        let ctor = qjs::sofuu_js_get_property_str(ctx, global, c"Float32Array".as_ptr());
        qjs::sofuu_js_free_value(ctx, global);
        if qjs::is_exception(ctor) {
            let exc = qjs::sofuu_js_get_exception(ctx);
            qjs::sofuu_js_free_value(ctx, exc);
            return false;
        }
        let ok = if qjs::JS_IsFunction(ctx, ctor) != 0 {
            let r = qjs::JS_IsInstanceOf(ctx, v, ctor);
            if r < 0 {
                let exc = qjs::sofuu_js_get_exception(ctx);
                qjs::sofuu_js_free_value(ctx, exc);
                false
            } else {
                r == 1
            }
        } else {
            false
        };
        qjs::sofuu_js_free_value(ctx, ctor);
        ok
    }

    /// The offending value's constructor name, for an E_TYPE message that
    /// names what the caller actually passed ("Float64Array", "Int32Array", …).
    unsafe fn view_ctor_name(ctx: *mut JSContext, v: JSValueConst) -> String {
        const FALLBACK: &str = "non-Float32Array value";
        let ctor = qjs::sofuu_js_get_property_str(ctx, v, c"constructor".as_ptr());
        if qjs::is_exception(ctor) {
            let exc = qjs::sofuu_js_get_exception(ctx);
            qjs::sofuu_js_free_value(ctx, exc);
            return FALLBACK.to_string();
        }
        let name = qjs::sofuu_js_get_property_str(ctx, ctor, c"name".as_ptr());
        qjs::sofuu_js_free_value(ctx, ctor);
        if qjs::is_exception(name) {
            let exc = qjs::sofuu_js_get_exception(ctx);
            qjs::sofuu_js_free_value(ctx, exc);
            return FALLBACK.to_string();
        }
        let c = qjs::sofuu_js_to_cstring(ctx, name);
        qjs::sofuu_js_free_value(ctx, name);
        if c.is_null() {
            return FALLBACK.to_string();
        }
        let s = CStr::from_ptr(c).to_string_lossy().into_owned();
        qjs::sofuu_js_free_cstring(ctx, c);
        if s.is_empty() {
            FALLBACK.to_string()
        } else {
            s
        }
    }

    /// W1.3: the bridge is the actual boundary — Float32Array only. The plain
    /// Array path stays (it is documented ergonomics: `new Float32Array(final)`
    /// callers still work), but a WRONG TypedArray is never converted.
    ///
    /// Throws (with a code) and returns None on any refusal, so every caller
    /// just returns `sofuu_js_exception()`.
    unsafe fn cma_vec_view(ctx: *mut JSContext, v: JSValueConst) -> Option<F32View> {
        let mut byte_length: usize = 0;
        let mut byte_offset: usize = 0;
        let ab = qjs::JS_GetTypedArrayBuffer(ctx, v, &mut byte_offset, &mut byte_length, ptr::null_mut());
        if !qjs::is_exception(ab) {
            /* It IS an ArrayBuffer view — that is NOT enough (W1.3). */
            if !is_float32_array(ctx, v) {
                qjs::sofuu_js_free_value(ctx, ab);
                let name = view_ctor_name(ctx, v);
                throw_coded(
                    ctx,
                    E_TYPE,
                    &format!(
                        "memory: expected a Float32Array but received a {name} — a {name}'s \
                         bytes cannot be re-interpreted as float32; pass a Float32Array (or a \
                         plain Array of numbers)"
                    ),
                );
                return None;
            }
            let mut ab_size: usize = 0;
            let buf = qjs::JS_GetArrayBuffer(ctx, &mut ab_size, ab);
            qjs::sofuu_js_free_value(ctx, ab);
            if !buf.is_null() {
                /* R7.P2: the window is re-derived against the buffer before
                 * the pointer escapes (bounds, lane alignment, overflow). */
                let window = match crate::rt::view_guard::check_f32_view(
                    ab_size,
                    byte_offset,
                    byte_length,
                    None,
                ) {
                    Ok(w) => w,
                    Err(e) => {
                        throw_view_refusal(ctx, e);
                        return None;
                    }
                };
                let p = (buf as *const u8).add(window.offset) as *const f32;
                let view = F32View::Borrowed(p, window.count);
                /* W1.1: the one scan — ingest AND query arrive here. */
                if let Some((at, val)) = first_non_finite(view.as_slice()) {
                    throw_coded(ctx, E_NONFINITE, &non_finite_message(at, val));
                    return None;
                }
                return Some(view);
            }
            throw_coded(
                ctx,
                E_TYPE,
                "memory: the Float32Array's buffer is detached — pass a live Float32Array",
            );
            return None;
        }
        let exc = qjs::JS_GetException(ctx);
        qjs::sofuu_js_free_value(ctx, exc);

        // Plain JavaScript Array fallback: [0.1, 0.2, ...]
        if qjs::JS_IsArray(ctx, v) != 0 {
            let len_v = qjs::sofuu_js_get_property_str(ctx, v, c"length".as_ptr());
            let mut len32: u32 = 0;
            qjs::sofuu_js_to_uint32(ctx, &mut len32, len_v);
            qjs::sofuu_js_free_value(ctx, len_v);
            let mut out = vec![0.0f32; len32 as usize];
            for i in 0..len32 {
                let el = qjs::JS_GetPropertyUint32(ctx, v, i);
                let mut d: f64 = 0.0;
                qjs::JS_ToFloat64(ctx, &mut d, el);
                qjs::sofuu_js_free_value(ctx, el);
                out[i as usize] = d as f32;
            }
            /* W1.1, folded into the conversion pass: `d as f32` overflows a
             * finite f64 (1e300) to +Inf, so the CHECK is on the f32. */
            if let Some((at, val)) = first_non_finite(&out) {
                throw_coded(ctx, E_NONFINITE, &non_finite_message(at, val));
                return None;
            }
            return Some(F32View::Owned(out));
        }

        /* A Float32Array whose buffer could not be fetched (detached). */
        if is_float32_array(ctx, v) {
            throw_coded(
                ctx,
                E_TYPE,
                "memory: the Float32Array's buffer is detached — pass a live Float32Array",
            );
            return None;
        }
        let got = view_ctor_name(ctx, v);
        throw_coded(
            ctx,
            E_TYPE,
            &format!(
                "memory: expected a Float32Array or a plain Array of numbers, got a {got}"
            ),
        );
        None
    }

    /* W1.3: Float32Array arg → borrowed float pointer + element count. The
     * same type + finiteness gate as cma_vec_view; this raw-pointer shape is
     * used by the kv save/search and agent prefetch paths.
     * Null pointer on error (a CODED exception is already thrown). */
    unsafe fn cma_vec_arg(ctx: *mut JSContext, v: JSValueConst) -> (*const f32, usize) {
        let mut byte_length: usize = 0;
        let mut byte_offset: usize = 0;
        let ab = qjs::JS_GetTypedArrayBuffer(ctx, v, &mut byte_offset, &mut byte_length, ptr::null_mut());
        if qjs::is_exception(ab) {
            let exc = qjs::JS_GetException(ctx);
            qjs::sofuu_js_free_value(ctx, exc);
            let got = view_ctor_name(ctx, v);
            throw_coded(
                ctx,
                E_TYPE,
                &format!("memory: expected a Float32Array, got a {got}"),
            );
            return (ptr::null(), 0);
        }
        if !is_float32_array(ctx, v) {
            qjs::sofuu_js_free_value(ctx, ab);
            let name = view_ctor_name(ctx, v);
            throw_coded(
                ctx,
                E_TYPE,
                &format!(
                    "memory: expected a Float32Array but received a {name} — a {name}'s bytes \
                     cannot be re-interpreted as float32"
                ),
            );
            return (ptr::null(), 0);
        }
        let mut ab_size: usize = 0;
        let buf = qjs::JS_GetArrayBuffer(ctx, &mut ab_size, ab);
        qjs::sofuu_js_free_value(ctx, ab);
        if buf.is_null() {
            throw_coded(
                ctx,
                E_TYPE,
                "memory: the Float32Array's buffer is detached — pass a live Float32Array",
            );
            return (ptr::null(), 0);
        }
        /* R7.P2: same validated window as cma_vec_view. A zero-length vector
         * is refused HERE because this shape returns a bare pointer and its
         * callers read null as "already threw"; there is no empty kv key or
         * query to represent, so the honest answer is a coded error. */
        let window = match crate::rt::view_guard::check_f32_view(ab_size, byte_offset, byte_length, None) {
            Ok(w) => w,
            Err(e) => {
                throw_view_refusal(ctx, e);
                return (ptr::null(), 0);
            }
        };
        if window.count == 0 {
            throw_coded(ctx, E_TYPE, "memory: expected a non-empty Float32Array");
            return (ptr::null(), 0);
        }
        let p = (buf as *const u8).add(window.offset) as *const f32;
        let n = window.count;
        /* W1.1: the same single scan, before the pointer escapes. */
        if let Some((at, val)) = first_non_finite(std::slice::from_raw_parts(p, n)) {
            throw_coded(ctx, E_NONFINITE, &non_finite_message(at, val));
            return (ptr::null(), 0);
        }
        (p, n)
    }

    /// R7.P2: a view_guard refusal as a coded memory error (`E_TYPE`).
    /// Fail-closed: the caller returns the pending exception.
    unsafe fn throw_view_refusal(ctx: *mut JSContext, e: crate::rt::view_guard::ViewError) {
        throw_coded(ctx, E_TYPE, &format!("memory: {}", e.describe()));
    }

    /// W2.2/W3.3: `!!opts[name]` for an options object. Missing, null and
    /// undefined are all "false"; a hostile getter that throws is swallowed
    /// (the flag defaults to off, which is the safe direction for BOTH flags:
    /// no readonly lock downgrade, no eager full read).
    unsafe fn js_object_flag(
        ctx: *mut JSContext,
        obj: JSValueConst,
        name: &std::ffi::CStr,
    ) -> bool {
        let v = qjs::sofuu_js_get_property_str(ctx, obj, name.as_ptr());
        if qjs::is_exception(v) {
            let exc = qjs::sofuu_js_get_exception(ctx);
            qjs::sofuu_js_free_value(ctx, exc);
            return false;
        }
        if qjs::is_undefined(v) || qjs::is_null(v) {
            qjs::sofuu_js_free_value(ctx, v);
            return false;
        }
        let rc = qjs::JS_ToBool(ctx, v);
        if rc < 0 {
            let exc = qjs::sofuu_js_get_exception(ctx);
            qjs::sofuu_js_free_value(ctx, exc);
        }
        qjs::sofuu_js_free_value(ctx, v);
        rc > 0
    }

    unsafe extern "C" fn js_cma_open(
        ctx: *mut JSContext,
        _this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 2 {
            return throw_coded(ctx, E_TYPE, "Expected path and vector dimension");
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
        /* W2.2/W3.3: optional 4th arg — an options object
         * `{readonly, verify_full|verifyFull}` (memory_api.js forwards the
         * object form's flags here). Anything else, or nothing, keeps the
         * pre-W2/W3 behaviour exactly. */
        let mut open_opts = OpenOpts::default();
        if argc >= 4 && qjs::is_object(*argv.add(3)) {
            let obj = *argv.add(3);
            open_opts.readonly = js_object_flag(ctx, obj, c"readonly");
            open_opts.verify_full = js_object_flag(ctx, obj, c"verify_full")
                || js_object_flag(ctx, obj, c"verifyFull");
        }

        let path_owned = cstr_opt(path);
        let shell = match path_owned.as_deref() {
            Some(p) => cma_open_ex(p, dim as usize, embed_id.as_deref(), open_opts),
            None => Err(OpenError::Corrupt),
        };
        if !path.is_null() {
            qjs::sofuu_js_free_cstring(ctx, path);
        }
        let shell = match shell {
            Ok(s) => s,
            /* W2.1: a second writer (or a reader arriving while a writer
             * holds the brain) is a LOCKED refusal with the crate's own
             * code — not the generic corrupt-open message, which would send
             * the host looking for a damaged file that does not exist. */
            Err(OpenError::Locked) => {
                return throw_coded(
                    ctx,
                    E_LOCKED,
                    &format!(
                        "memory.open: the brain could not be locked — another handle or process holds a conflicting lock on {} (see the sofuu/cma warning for the holder's pid); nothing was opened",
                        path_owned.unwrap_or_default()
                    ),
                );
            }
            /* W1.7: an unopenable brain is E_CORRUPT (a bare InternalError
             * without a code made every failure indistinguishable). The
             * concrete reason is on stderr, where it always was. */
            Err(OpenError::Corrupt) => {
                return throw_coded(
                    ctx,
                    E_CORRUPT,
                    "memory.open: the brain could not be opened (unreadable container, wrong key, or an                  incompatible vector space — see the sofuu/cma warnings for the reason)",
                );
            }
        };
        /* W1.8 (+W3.3): report the damage this load found BEFORE the handle
         * is handed back — an EAGER (verify_full or legacy) open only. A
         * lazy open has read no payload yet, so it reports its recovery on
         * the access that triggers the load (`shell_from_this`). */
        let mut shell = shell;
        flush_pending_recovery(ctx, &mut shell);
        let shell = Box::into_raw(shell);

        let obj = qjs::JS_NewObjectClass(ctx, CMA_CLASS_ID.load(Ordering::Relaxed) as c_int);
        qjs::JS_SetOpaque(obj, shell as *mut c_void);
        obj
    }

    /// B1.5: read a brain file's manifest — `{ok, id, input, dimension,
    /// artifact, records}` — WITHOUT opening it. A host picks its embedder
    /// from this before `open`; the brain server used to hardcode dim 768
    /// and could therefore only ever serve hash brains.
    unsafe extern "C" fn js_memory_inspect(
        ctx: *mut JSContext,
        _this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 1 {
            return throw_coded(ctx, E_TYPE, "Expected path");
        }
        let path = qjs::sofuu_js_to_cstring(ctx, *argv);
        let path_s = cstr_opt(path);
        if !path.is_null() {
            qjs::sofuu_js_free_cstring(ctx, path);
        }
        let Some(path_s) = path_s else {
            return throw_coded(ctx, E_TYPE, "Expected path string");
        };
        /* B3 (audit 2026-10-14): a segmented brain's MANIFEST is the true,
         * cheap answer. The legacy single-file reader below sees only a
         * metadata-only container there, and reported {dimension:0, id:""}
         * for every post-B3 brain — and the brain server picks its
         * embedder from this, so a 64-dim semantic brain got opened as
         * hash-768, was refused, and every memory route answered 503.
         * `Ok(None)` = not segmented (missing file, foreign bytes, legacy
         * single-file brain) — fall through to the legacy reader. `Err` =
         * a container we cannot open (wrong key, torn file): fail loudly,
         * never as ok:true. */
        match crate::rt::brain_store::read_manifest_ex(&path_s) {
            Err(rc) => {
                let c = CString::new(
                    serde_json::json!({ "ok": false, "error": rc }).to_string(),
                )
                .unwrap_or_default();
                return qjs::sofuu_js_new_string(ctx, c.as_ptr());
            }
            Ok(Some((m, _legacy))) => {
                let emb = &m.embedding;
                let s = |k: &str| {
                    emb.get(k)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                };
                let records = if m.records > 0 {
                    m.records
                } else {
                    m.total_versions() as u64
                };
                let out = serde_json::json!({
                    "ok": true,
                    "id": s("id"),
                    "input": s("input"),
                    "dimension": emb.get("dimension").and_then(|v| v.as_u64()).unwrap_or(0),
                    "artifact": s("artifact"),
                    /* Append segments carry superseded VERSIONS of one
                     * record; prefer the live count the flush stamped. */
                    "records": records,
                });
                let c = CString::new(out.to_string()).unwrap_or_default();
                return qjs::sofuu_js_new_string(ctx, c.as_ptr());
            }
            Ok(None) => { /* legacy or absent — the reader below owns it */ }
        }
        let out = match cma_qtsq_load(&path_s) {
            Ok(brain) => {
                let manifest = brain.meta.as_deref().and_then(embedding_manifest);
                let (id, input, dimension, artifact) = match manifest {
                    Some(m) => (m.id, m.input, m.dimension, m.artifact),
                    None => (
                        inferred_legacy_embedding_id(brain.dim)
                            .unwrap_or("")
                            .to_string(),
                        String::new(),
                        brain.dim,
                        String::new(),
                    ),
                };
                serde_json::json!({
                    "ok": true,
                    "id": id,
                    "input": input,
                    "dimension": dimension,
                    "artifact": artifact,
                    "records": brain.n,
                })
            }
            Err(rc) => serde_json::json!({"ok": false, "error": rc}),
        };
        let c = CString::new(out.to_string()).unwrap_or_default();
        qjs::sofuu_js_new_string(ctx, c.as_ptr())
    }

    /// Mirrors the C's JS_GetOpaque2 + null check → JS_EXCEPTION.
    ///
    /// W3.3: this is also where a LAZY open's payloads are read. Every CMA
    /// instance method (recall, remember, list, count, stats, decayTick,
    /// flush, compact, export, …) resolves its shell through here, so the
    /// first row access is exactly the moment the deferred load runs — once
    /// per handle, never per query.
    unsafe fn shell_from_this<'a>(
        ctx: *mut JSContext,
        this_val: JSValueConst,
    ) -> Option<&'a mut CmaShell> {
        let p = qjs::JS_GetOpaque2(ctx, this_val, CMA_CLASS_ID.load(Ordering::Relaxed))
            as *mut CmaShell;
        if p.is_null() {
            None
        } else {
            let shell = &mut *p;
            if let Err(rc) = materialize(shell) {
                throw_coded(
                    ctx,
                    E_CORRUPT,
                    &format!(
                        "memory: the brain store at {} could not be read (qtsq error {rc}) — see the sofuu/cma warnings for the reason",
                        shell.mind_path
                    ),
                );
                return None;
            }
            flush_pending_recovery(ctx, shell);
            Some(shell)
        }
    }

    unsafe extern "C" fn js_cma_remember(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 1 {
            return throw_coded(ctx, E_TYPE, "Expected text or (vec, text[, role, kv_page_id, source])");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        /* W2.2: a readonly handle holds a SHARED lock. Buffering a write it
         * can never flush would be silent data loss, so refuse at the entry
         * point instead of at flush time. */
        if cma.readonly {
            return throw_coded(ctx, E_LOCKED, "this brain is open readonly (shared lock) — write refused");
        }


        // B8.1: If first argument is a string, auto-embed via local TF-IDF
        if qjs::sofuu_js_is_string(*argv) != 0 {
            let text = qjs::sofuu_js_to_cstring(ctx, *argv);
            let role = if argc >= 2 && qjs::sofuu_js_is_string(*argv.add(1)) != 0 {
                qjs::sofuu_js_to_cstring(ctx, *argv.add(1))
            } else {
                std::ptr::null_mut()
            };
            let mut page_id: u32 = 0;
            let mut source_idx: usize = 2;
            if argc >= 3 && qjs::sofuu_js_is_number(*argv.add(2)) != 0 {
                qjs::sofuu_js_to_uint32(ctx, &mut page_id, *argv.add(2));
                source_idx = 3;
            }
            let source = if (argc as usize) > source_idx && qjs::sofuu_js_is_string(*argv.add(source_idx)) != 0 {
                qjs::sofuu_js_to_cstring(ctx, *argv.add(source_idx))
            } else {
                std::ptr::null_mut()
            };

            let text_opt = cstr_opt(text);
            let role_opt = cstr_opt(role);
            let source_opt = cstr_opt(source);

            let mut emb = vec![0.0f32; cma.vec_dim];
            if let Some(ref t) = text_opt {
                crate::rt::ai::sofuu_tfidf_embed(t.as_bytes(), &mut emb, cma.vec_dim);
            }

            let outcome = cma_remember_outcome(
                cma,
                &emb,
                text_opt.as_deref(),
                role_opt.as_deref(),
                page_id,
                source_opt.as_deref(),
            );

            if !text.is_null() {
                qjs::sofuu_js_free_cstring(ctx, text);
            }
            if !role.is_null() {
                qjs::sofuu_js_free_cstring(ctx, role);
            }
            if !source.is_null() {
                qjs::sofuu_js_free_cstring(ctx, source);
            }
            cma.last_outcome = outcome;
            return qjs::sofuu_js_new_int32(ctx, outcome.map(|o| o.id as i32).unwrap_or(-1));
        }

        if argc < 2 {
            return throw_coded(ctx, E_TYPE, "Expected vec and text[, role, kv_page_id, source]");
        }

        let Some(vec_view) = cma_vec_view(ctx, *argv) else {
            /* W1.3/W1.1/W1.7: cma_vec_view already threw a CODED refusal. */
            return qjs::sofuu_js_exception();
        };
        let vec_slice = vec_view.as_slice();
        if vec_slice.len() != cma.vec_dim {
            return throw_coded(
                ctx,
                E_DIM,
                &dim_mismatch_message("remember", vec_slice.len(), cma.vec_dim),
            );
        }

        let text = qjs::sofuu_js_to_cstring(ctx, *argv.add(1));
        let role = if argc >= 3 && qjs::sofuu_js_is_string(*argv.add(2)) != 0 {
            qjs::sofuu_js_to_cstring(ctx, *argv.add(2))
        } else {
            std::ptr::null_mut()
        };
        let mut page_id: u32 = 0;
        if argc >= 4 && qjs::sofuu_js_is_number(*argv.add(3)) != 0 {
            qjs::sofuu_js_to_uint32(ctx, &mut page_id, *argv.add(3));
        }
        /* B1.2 provenance: optional 5th arg = where this memory came from
         * ("chat", "agent:researcher", "pin", …). Absent → unknown. */
        let source = if argc >= 5 && qjs::sofuu_js_is_string(*argv.add(4)) != 0 {
            qjs::sofuu_js_to_cstring(ctx, *argv.add(4))
        } else {
            std::ptr::null_mut()
        };

        let outcome = cma_remember_outcome(
            cma,
            vec_slice,
            cstr_opt(text).as_deref(),
            cstr_opt(role).as_deref(),
            page_id,
            cstr_opt(source).as_deref(),
        );

        if !text.is_null() {
            qjs::sofuu_js_free_cstring(ctx, text);
        }
        if !role.is_null() {
            qjs::sofuu_js_free_cstring(ctx, role);
        }
        if !source.is_null() {
            qjs::sofuu_js_free_cstring(ctx, source);
        }
        cma.last_outcome = outcome;
        qjs::sofuu_js_new_int32(ctx, outcome.map(|o| o.id as i32).unwrap_or(-1))
    }

    /// W5.2: `cma.lastRememberOutcome()` → `{id, merged, into?}` for the
    /// most recent `remember`/`learn`/`rememberEntity`-style write on this
    /// handle. A separate method keeps `remember()`'s numeric id intact.
    unsafe extern "C" fn js_cma_last_remember_outcome(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        remember_outcome_value(ctx, cma.last_outcome)
    }

    /// B8.1: `cma.learn(text[, role, source])` or `cma.learn(text, { role, source, pageId })`
    /// Dedicated 1-line high-level memory ingestion with automatic subword TF-IDF embedding.
    unsafe extern "C" fn js_cma_learn(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 1 || qjs::sofuu_js_is_string(*argv) == 0 {
            return throw_coded(ctx, E_TYPE, "Expected text (string)");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };

        let text_c = qjs::sofuu_js_to_cstring(ctx, *argv);
        let text_opt = cstr_opt(text_c);

        let mut role_str = String::from("user");
        let mut source_str = String::new();
        let mut page_id: u32 = 0;

        if argc >= 2 {
            let arg1 = *argv.add(1);
            if qjs::sofuu_js_is_string(arg1) != 0 {
                let r = qjs::sofuu_js_to_cstring(ctx, arg1);
                if let Some(rs) = cstr_opt(r) {
                    role_str = rs;
                }
                if !r.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, r);
                }
                if argc >= 3 && qjs::sofuu_js_is_string(*argv.add(2)) != 0 {
                    let s = qjs::sofuu_js_to_cstring(ctx, *argv.add(2));
                    if let Some(ss) = cstr_opt(s) {
                        source_str = ss;
                    }
                    if !s.is_null() {
                        qjs::sofuu_js_free_cstring(ctx, s);
                    }
                }
            } else if qjs::sofuu_js_is_object(arg1) != 0 {
                let r_val = qjs::sofuu_js_get_property_str(ctx, arg1, c"role".as_ptr());
                if qjs::sofuu_js_is_string(r_val) != 0 {
                    let r = qjs::sofuu_js_to_cstring(ctx, r_val);
                    if let Some(rs) = cstr_opt(r) {
                        role_str = rs;
                    }
                    if !r.is_null() {
                        qjs::sofuu_js_free_cstring(ctx, r);
                    }
                }
                qjs::sofuu_js_free_value(ctx, r_val);

                let s_val = qjs::sofuu_js_get_property_str(ctx, arg1, c"source".as_ptr());
                if qjs::sofuu_js_is_string(s_val) != 0 {
                    let s = qjs::sofuu_js_to_cstring(ctx, s_val);
                    if let Some(ss) = cstr_opt(s) {
                        source_str = ss;
                    }
                    if !s.is_null() {
                        qjs::sofuu_js_free_cstring(ctx, s);
                    }
                }
                qjs::sofuu_js_free_value(ctx, s_val);

                let pid_val = qjs::sofuu_js_get_property_str(ctx, arg1, c"pageId".as_ptr());
                if qjs::sofuu_js_is_number(pid_val) != 0 {
                    qjs::sofuu_js_to_uint32(ctx, &mut page_id, pid_val);
                } else {
                    let kpid_val = qjs::sofuu_js_get_property_str(ctx, arg1, c"kv_page_id".as_ptr());
                    if qjs::sofuu_js_is_number(kpid_val) != 0 {
                        qjs::sofuu_js_to_uint32(ctx, &mut page_id, kpid_val);
                    }
                    qjs::sofuu_js_free_value(ctx, kpid_val);
                }
                qjs::sofuu_js_free_value(ctx, pid_val);
            }
        }

        let mut emb = vec![0.0f32; cma.vec_dim];
        if let Some(ref t) = text_opt {
            crate::rt::ai::sofuu_tfidf_embed(t.as_bytes(), &mut emb, cma.vec_dim);
        }

        let idx = cma_remember(
            cma,
            &emb,
            text_opt.as_deref(),
            Some(&role_str),
            page_id,
            Some(&source_str),
        );

        if !text_c.is_null() {
            qjs::sofuu_js_free_cstring(ctx, text_c);
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
            return throw_coded(ctx, E_TYPE, "Expected vec, text, entityName, entityType");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        /* W2.2: a readonly handle holds a SHARED lock. Buffering a write it
         * can never flush would be silent data loss, so refuse at the entry
         * point instead of at flush time. */
        if cma.readonly {
            return throw_coded(ctx, E_LOCKED, "this brain is open readonly (shared lock) — write refused");
        }


        let Some(vec_view) = cma_vec_view(ctx, *argv) else {
            return qjs::sofuu_js_exception();
        };
        let vec_slice = vec_view.as_slice();
        if vec_slice.len() != cma.vec_dim {
            return throw_coded(
                ctx,
                E_DIM,
                &dim_mismatch_message("rememberEntity", vec_slice.len(), cma.vec_dim),
            );
        }

        let text = qjs::sofuu_js_to_cstring(ctx, *argv.add(1));
        let entity_name = qjs::sofuu_js_to_cstring(ctx, *argv.add(2));
        let entity_type = qjs::sofuu_js_to_cstring(ctx, *argv.add(3));

        let idx = cma_remember_entity(
            cma,
            vec_slice,
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
        if argc < 1 {
            return throw_coded(ctx, E_TYPE, "Expected queryText or (queryVec, topK[, text])");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };

        // B8.2: If first argument is a string, auto-embed query text + token re-rank
        if qjs::sofuu_js_is_string(*argv) != 0 {
            let text_c = qjs::sofuu_js_to_cstring(ctx, *argv);
            let text = if !text_c.is_null() {
                let s = std::ffi::CStr::from_ptr(text_c).to_string_lossy().into_owned();
                qjs::sofuu_js_free_cstring(ctx, text_c);
                s
            } else {
                String::new()
            };
            let mut top_k: u32 = 5;
            if argc >= 2 && qjs::sofuu_js_is_number(*argv.add(1)) != 0 {
                qjs::sofuu_js_to_uint32(ctx, &mut top_k, *argv.add(1));
            }
            if top_k > 1024 {
                top_k = 1024;
            }
            let mut query_vec = vec![0.0f32; cma.vec_dim];
            crate::rt::ai::sofuu_tfidf_embed(text.as_bytes(), &mut query_vec, cma.vec_dim);

            let json = cma_recall_text_json(cma, &query_vec, top_k as usize, &text);
            let cj = CString::new(json).unwrap_or_default();
            let mut arr = qjs::JS_ParseJSON(ctx, cj.as_ptr(), cj.as_bytes().len(), c"<cma-recall>".as_ptr());
            if qjs::is_exception(arr) {
                qjs::sofuu_js_get_exception(ctx);
                arr = qjs::JS_NewArray(ctx);
            }
            return arr;
        }

        if argc < 2 {
            return throw_coded(ctx, E_TYPE, "Expected queryVec and topK");
        }

        let Some(vec_view) = cma_vec_view(ctx, *argv) else {
            return qjs::sofuu_js_exception();
        };
        let vec_slice = vec_view.as_slice();
        /* W6.5 (AUDIT-2026-10-29): EXACT dimension. `<` silently truncated a
         * longer query (`&vec_slice[..cma.vec_dim]` below) — a 1024d query on
         * a 768d brain scored 768 of its 1024 dimensions and returned a hit
         * with no error. A short query was a codeless TypeError. Both are
         * E_DIM now, and neither can reach the scorer. */
        if vec_slice.len() != cma.vec_dim {
            return throw_coded(
                ctx,
                E_DIM,
                &dim_mismatch_message("recall", vec_slice.len(), cma.vec_dim),
            );
        }

        let mut top_k: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut top_k, *argv.add(1));
        /* Clamp: an unbounded top_k would size the HNSW candidate heap
         * (ef = top_k*3) and OOM on a hostile/accidental huge value. */
        if top_k > 1024 {
            top_k = 1024;
        }

        let json = if argc >= 3 && qjs::sofuu_js_is_string(*argv.add(2)) != 0 {
            /* M1.3: optional query text — recall WITH the token re-rank. */
            let text = qjs::sofuu_js_to_cstring(ctx, *argv.add(2));
            let text = std::ffi::CStr::from_ptr(text)
                .to_string_lossy()
                .into_owned();
            cma_recall_text_json(cma, &vec_slice[..cma.vec_dim], top_k as usize, &text)
        } else {
            cma_recall_json(cma, &vec_slice[..cma.vec_dim], top_k as usize)
        };
        let cj = CString::new(json).unwrap_or_default();
        let mut arr = qjs::JS_ParseJSON(ctx, cj.as_ptr(), cj.as_bytes().len(), c"<cma-recall>".as_ptr());
        if qjs::is_exception(arr) {
            qjs::sofuu_js_get_exception(ctx);
            arr = qjs::JS_NewArray(ctx);
        }
        arr
    }

    /// B8.2: `cma.search(queryText[, topK])`
    /// Dedicated 1-line high-level semantic recall with automatic subword TF-IDF embedding + BM25 re-ranking.
    unsafe extern "C" fn js_cma_search(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 1 || qjs::sofuu_js_is_string(*argv) == 0 {
            return throw_coded(ctx, E_TYPE, "Expected queryText (string)");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };

        let text_c = qjs::sofuu_js_to_cstring(ctx, *argv);
        let text = if !text_c.is_null() {
            let s = std::ffi::CStr::from_ptr(text_c).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, text_c);
            s
        } else {
            String::new()
        };

        let mut top_k: u32 = 5;
        if argc >= 2 && qjs::sofuu_js_is_number(*argv.add(1)) != 0 {
            qjs::sofuu_js_to_uint32(ctx, &mut top_k, *argv.add(1));
        }
        if top_k > 1024 {
            top_k = 1024;
        }

        let mut query_vec = vec![0.0f32; cma.vec_dim];
        crate::rt::ai::sofuu_tfidf_embed(text.as_bytes(), &mut query_vec, cma.vec_dim);

        let json = cma_recall_text_json(cma, &query_vec, top_k as usize, &text);
        let cj = CString::new(json).unwrap_or_default();
        let mut arr = qjs::JS_ParseJSON(ctx, cj.as_ptr(), cj.as_bytes().len(), c"<cma-search>".as_ptr());
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
            return throw_coded(ctx, E_TYPE, "Expected queryVec and n_hints");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };

        let Some(vec_view) = cma_vec_view(ctx, *argv) else {
            return qjs::sofuu_js_exception();
        };
        let vec_slice = vec_view.as_slice();
        /* Exact dimension, same rule as recall (a long vector used to be
         * silently truncated, a short one was a codeless TypeError). */
        if vec_slice.len() != cma.vec_dim {
            return throw_coded(
                ctx,
                E_DIM,
                &dim_mismatch_message("kvHints", vec_slice.len(), cma.vec_dim),
            );
        }

        let mut n_hints: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut n_hints, *argv.add(1));
        if n_hints > 1024 {
            n_hints = 1024;
        }

        let ids = cma.cma.kv_hints(&vec_slice[..cma.vec_dim], n_hints as usize);
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
            return throw_coded(ctx, E_TYPE, "Expected id or Array of ids");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let mut ids: Vec<u32> = Vec::new();
        if qjs::JS_IsArray(ctx, *argv) != 0 {
            let len_val = qjs::sofuu_js_get_property_str(ctx, *argv, c"length".as_ptr());
            let mut len: u32 = 0;
            qjs::sofuu_js_to_uint32(ctx, &mut len, len_val);
            qjs::sofuu_js_free_value(ctx, len_val);

            ids.reserve(len as usize);
            for i in 0..len {
                let id_val = qjs::JS_GetPropertyUint32(ctx, *argv, i);
                let mut id: u32 = 0;
                qjs::sofuu_js_to_uint32(ctx, &mut id, id_val);
                qjs::sofuu_js_free_value(ctx, id_val);
                ids.push(id);
            }
        } else if qjs::sofuu_js_is_number(*argv) != 0 {
            let mut id: u32 = 0;
            qjs::sofuu_js_to_uint32(ctx, &mut id, *argv);
            ids.push(id);
        } else {
            return throw_coded(ctx, E_TYPE, "Argument must be an id (number) or Array of ids");
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
            return throw_coded(ctx, E_TYPE, "Expected dt_seconds");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        /* W2.2: a readonly handle holds a SHARED lock. Buffering a write it
         * can never flush would be silent data loss, so refuse at the entry
         * point instead of at flush time. */
        if cma.readonly {
            return throw_coded(ctx, E_LOCKED, "this brain is open readonly (shared lock) — write refused");
        }

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
        /* W2.2: a readonly handle holds a SHARED lock. Buffering a write it
         * can never flush would be silent data loss, so refuse at the entry
         * point instead of at flush time. */
        if cma.readonly {
            return throw_coded(ctx, E_LOCKED, "this brain is open readonly (shared lock) — write refused");
        }

        let dropped = cma.cma.retain();
        qjs::sofuu_js_new_uint32(ctx, dropped as u32)
    }

    /// B1.1 read-out: `cma.list({tier?, role?, entity?, minStrength?,
    /// offset?, limit?})` → JSON `{total, offset, limit, records:[…]}`.
    /// The same shape every surface (chat browser, brain server, C ABI)
    /// renders — one contract, four callers.
    unsafe extern "C" fn js_cma_list(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let mut filter = crate::memory::cma::ListFilter::default();
        if argc >= 1 && qjs::is_object(*argv) {
            let opts = *argv;

            let v = qjs::sofuu_js_get_property_str(ctx, opts, c"tier".as_ptr());
            if qjs::sofuu_js_is_number(v) != 0 {
                let mut t: u32 = 0;
                qjs::sofuu_js_to_uint32(ctx, &mut t, v);
                filter.tier = Some(t.min(u8::MAX as u32) as u8);
            }
            qjs::sofuu_js_free_value(ctx, v);

            let v = qjs::sofuu_js_get_property_str(ctx, opts, c"role".as_ptr());
            if qjs::sofuu_js_is_string(v) != 0 {
                let s = qjs::sofuu_js_to_cstring(ctx, v);
                filter.role = cstr_opt(s);
                if !s.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, s);
                }
            }
            qjs::sofuu_js_free_value(ctx, v);

            let v = qjs::sofuu_js_get_property_str(ctx, opts, c"entity".as_ptr());
            if qjs::sofuu_js_is_string(v) != 0 {
                let s = qjs::sofuu_js_to_cstring(ctx, v);
                filter.entity = cstr_opt(s);
                if !s.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, s);
                }
            }
            qjs::sofuu_js_free_value(ctx, v);

            let v = qjs::sofuu_js_get_property_str(ctx, opts, c"text".as_ptr());
            if qjs::sofuu_js_is_string(v) != 0 {
                let s = qjs::sofuu_js_to_cstring(ctx, v);
                filter.text = cstr_opt(s);
                if !s.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, s);
                }
            }
            qjs::sofuu_js_free_value(ctx, v);

            let v = qjs::sofuu_js_get_property_str(ctx, opts, c"minStrength".as_ptr());
            if qjs::sofuu_js_is_number(v) != 0 {
                let mut m: f64 = 0.0;
                qjs::JS_ToFloat64(ctx, &mut m, v);
                filter.min_strength = Some(m as f32);
            }
            qjs::sofuu_js_free_value(ctx, v);

            let v = qjs::sofuu_js_get_property_str(ctx, opts, c"offset".as_ptr());
            if qjs::sofuu_js_is_number(v) != 0 {
                let mut o: u32 = 0;
                qjs::sofuu_js_to_uint32(ctx, &mut o, v);
                filter.offset = o as usize;
            }
            qjs::sofuu_js_free_value(ctx, v);

            let v = qjs::sofuu_js_get_property_str(ctx, opts, c"limit".as_ptr());
            if qjs::sofuu_js_is_number(v) != 0 {
                let mut l: u32 = 0;
                qjs::sofuu_js_to_uint32(ctx, &mut l, v);
                /* Clamp: an unbounded page would serialize the whole brain
                 * into one JSON string (an accidental 100 MB reply). */
                filter.limit = (l as usize).min(10_000);
            }
            qjs::sofuu_js_free_value(ctx, v);
        }
        let json = cma.cma.list_json(&filter);
        let c = CString::new(json).unwrap_or_default();
        qjs::sofuu_js_new_string(ctx, c.as_ptr())
    }

    /// B2.2: `cma.importRecords(vecsFloat32, recordsJson)` → JSON
    /// `{added, merged, skipped}`. `vecs` must already be in THIS store's
    /// space (the caller re-embeds when a card came from another space).
    ///
    /// W6.3 — the serialization contract, stated once: `recordsJson` is a JSON
    /// OBJECT of the form `{"records":[...]}`, one record object per vector,
    /// which is what `Cma::import_records` reads (`root["records"]`). A bare
    /// JSON array matches nothing there and used to answer
    /// `{added:0,merged:0,skipped:0}` — a silent no-op the flat JS bridge hit
    /// on every call. The shape is now asserted before the import, so a
    /// mismatch is a TypeError instead of an empty success.
    unsafe extern "C" fn js_cma_import(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 2 {
            return throw_coded(ctx, E_TYPE, "Expected vecs(Float32Array) and recordsJson");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        /* W2.2: a readonly handle holds a SHARED lock. Buffering a write it
         * can never flush would be silent data loss, so refuse at the entry
         * point instead of at flush time. */
        if cma.readonly {
            return throw_coded(ctx, E_LOCKED, "this brain is open readonly (shared lock) — write refused");
        }

        let Some(vec_view) = cma_vec_view(ctx, *argv) else {
            return qjs::sofuu_js_exception();
        };
        let vec_slice = vec_view.as_slice();
        let json = qjs::sofuu_js_to_cstring(ctx, *argv.add(1));
        let json_s = cstr_opt(json);
        if !json.is_null() {
            qjs::sofuu_js_free_cstring(ctx, json);
        }
        let Some(json_s) = json_s else {
            return throw_coded(ctx, E_TYPE, "Argument 2 must be a JSON string");
        };
        /* W6.3: refuse the wrong envelope loudly (see the contract above). */
        if !serde_json::from_str::<serde_json::Value>(&json_s)
            .ok()
            .and_then(|v| v.get("records").map(|r| r.is_array()))
            .unwrap_or(false)
        {
            return throw_coded(ctx, E_TYPE, "Argument 2 must be JSON of the form {\"records\":[...]}");
        }
        let dim = cma.vec_dim;
        /* The vecs buffer is records*stride — a partial trailing vector used
         * to be dropped by integer division. Refuse it instead. */
        if dim == 0 || vec_slice.len() % dim != 0 {
            return throw_coded(
                ctx,
                E_DIM,
                &format!(
                    "memory.importRecords: vecs has {} float(s), which is not a whole number of                      {dim}-dimensional records",
                    vec_slice.len()
                ),
            );
        }
        let n = vec_slice.len() / dim;
        let stats = cma.cma.import_records(
            vec_slice,
            n,
            dim,
            &json_s,
        );
        let out = serde_json::json!({
            "added": stats.added,
            "merged": stats.merged,
            "skipped": stats.skipped,
        });
        let c = CString::new(out.to_string()).unwrap_or_default();
        qjs::sofuu_js_new_string(ctx, c.as_ptr())
    }

    /// B3.5 (ROADMAP-BRAIN-CMA): `cma.retainAgainst(canonical)` — keep only
    /// records whose text appears in `canonical`'s live set, renumbering
    /// survivors exactly like `retain()`. The SEM2 sibling channel is a
    /// DERIVED index over the canonical store, so its lifecycle follows the
    /// canonical one instead of decaying on its own: this ends the double
    /// decay and the drift where one channel kept a memory the other had
    /// pruned. Returns the number of records removed.
    unsafe extern "C" fn js_cma_retain_against(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 1 {
            return throw_coded(ctx, E_TYPE, "Expected the canonical sofuu.memory handle");
        }
        let Some(this_shell) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let other = qjs::JS_GetOpaque2(ctx, *argv, CMA_CLASS_ID.load(Ordering::Relaxed))
            as *mut CmaShell;
        if other.is_null() {
            return throw_coded(ctx, E_TYPE, "Argument 1 must be a sofuu.memory handle");
        }
        /* Aliasing guard: the same handle would make `keep` borrow the very
         * records we drain — refuse instead of reading freed memory. */
        if std::ptr::eq(other, this_shell as *mut CmaShell) {
            return qjs::sofuu_js_new_int32(ctx, 0);
        }
        let other_shell = &*other;
        let keep: std::collections::HashSet<&str> = other_shell
            .cma
            .records
            .iter()
            .filter(|r| r.tier != crate::memory::cma::TIER_FORGOTTEN)
            .map(|r| r.text.as_str())
            .collect();
        let dropped = this_shell.cma.retain_matching(&keep);
        qjs::sofuu_js_new_int32(ctx, dropped as i32)
    }

    /// B3: `cma.stats()` → JSON diagnostics for the store — records,
    /// segments, versions on disk, bytes, evictions, decay stamp. The
    /// capacity policy reports here instead of silently shrinking a brain.
    /// B6.1: `cma.rememberWorking(vec, text, role, kvPageId, source)` →
    /// working id (high bit set). Session-scoped: recallable immediately,
    /// never serialized, gone on reload unless promoted.
    unsafe extern "C" fn js_cma_remember_working(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 2 {
            return throw_coded(ctx, E_TYPE, "Expected vec and text[, role, kv_page_id, source]");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        /* W2.2: a readonly handle holds a SHARED lock. Buffering a write it
         * can never flush would be silent data loss, so refuse at the entry
         * point instead of at flush time. */
        if cma.readonly {
            return throw_coded(ctx, E_LOCKED, "this brain is open readonly (shared lock) — write refused");
        }

        let Some(vec_view) = cma_vec_view(ctx, *argv) else {
            return qjs::sofuu_js_exception();
        };
        let vec_slice = vec_view.as_slice();
        if vec_slice.len() != cma.vec_dim {
            return throw_coded(
                ctx,
                E_DIM,
                &dim_mismatch_message("rememberWorking", vec_slice.len(), cma.vec_dim),
            );
        }
        let text = qjs::sofuu_js_to_cstring(ctx, *argv.add(1));
        let role = if argc >= 3 && qjs::sofuu_js_is_string(*argv.add(2)) != 0 {
            qjs::sofuu_js_to_cstring(ctx, *argv.add(2))
        } else {
            std::ptr::null_mut()
        };
        let mut page_id: u32 = 0;
        if argc >= 4 && qjs::sofuu_js_is_number(*argv.add(3)) != 0 {
            qjs::sofuu_js_to_uint32(ctx, &mut page_id, *argv.add(3));
        }
        let source = if argc >= 5 && qjs::sofuu_js_is_string(*argv.add(4)) != 0 {
            qjs::sofuu_js_to_cstring(ctx, *argv.add(4))
        } else {
            std::ptr::null_mut()
        };
        let id = cma.cma.remember_working(
            vec_slice,
            cstr_opt(text).as_deref().unwrap_or(""),
            cstr_opt(role).as_deref().unwrap_or("user"),
            page_id,
            cstr_opt(source).as_deref().unwrap_or(""),
        );
        if !text.is_null() {
            qjs::sofuu_js_free_cstring(ctx, text);
        }
        if !role.is_null() {
            qjs::sofuu_js_free_cstring(ctx, role);
        }
        if !source.is_null() {
            qjs::sofuu_js_free_cstring(ctx, source);
        }
        qjs::sofuu_js_new_int32(ctx, id as i32)
    }

    /// B6.2: `cma.promoteWorking(idsJson)` — promote the session working
    /// records with these ids (a JSON array of recall-hit ids, a JS array,
    /// or a single id) into the persistent store as episodic. Persistent
    /// ids are ignored. Returns `{promoted, merged}`.
    unsafe extern "C" fn js_cma_promote_working(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 1 {
            return throw_coded(ctx, E_TYPE, "Expected ids (Array of numbers, number, or JSON string)");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let mut ids: Vec<u32> = Vec::new();
        if qjs::JS_IsArray(ctx, *argv) != 0 {
            let len_val = qjs::sofuu_js_get_property_str(ctx, *argv, c"length".as_ptr());
            let mut len: u32 = 0;
            qjs::sofuu_js_to_uint32(ctx, &mut len, len_val);
            qjs::sofuu_js_free_value(ctx, len_val);

            ids.reserve(len as usize);
            for i in 0..len {
                let id_val = qjs::JS_GetPropertyUint32(ctx, *argv, i);
                let mut id: u32 = 0;
                qjs::sofuu_js_to_uint32(ctx, &mut id, id_val);
                qjs::sofuu_js_free_value(ctx, id_val);
                ids.push(id);
            }
        } else if qjs::sofuu_js_is_number(*argv) != 0 {
            let mut id: u32 = 0;
            qjs::sofuu_js_to_uint32(ctx, &mut id, *argv);
            ids.push(id);
        } else if qjs::sofuu_js_is_string(*argv) != 0 {
            let json_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
            if let Some(json) = cstr_opt(json_ptr) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(arr) = v.as_array() {
                        for item in arr {
                            if let Some(n) = item.as_u64() {
                                ids.push(n as u32);
                            }
                        }
                    } else if let Some(n) = v.as_u64() {
                        ids.push(n as u32);
                    }
                }
            }
            if !json_ptr.is_null() {
                qjs::sofuu_js_free_cstring(ctx, json_ptr);
            }
        } else {
            return throw_coded(ctx, E_TYPE, "Argument 1 must be an Array of ids, number, or JSON string");
        }
        let (promoted, merged) = cma.cma.promote_working_by_ids(&ids);
        let out = serde_json::json!({ "promoted": promoted, "merged": merged });
        let c = CString::new(out.to_string()).unwrap_or_default();
        qjs::sofuu_js_new_string(ctx, c.as_ptr())
    }

    /// B6: `cma.clearWorking()` — drop the session working set (session
    /// end). Returns how many records were dropped.
    unsafe extern "C" fn js_cma_clear_working(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        qjs::sofuu_js_new_int32(ctx, cma.cma.clear_working() as i32)
    }

    unsafe extern "C" fn js_cma_stats(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let layout = cma.layout.as_ref();
        let segments = layout.map(|l| l.segments.len()).unwrap_or(0);
        let versions: usize = layout
            .map(|l| l.segments.iter().map(|s| s.records).sum())
            .unwrap_or(0);
        /* Referenced segments only: a compaction leaves the previous
         * generation on disk until the age-guarded orphan sweep can prove
         * no concurrent reader holds it, and counting that would report a
         * store twice its real size. */
        let bytes: u64 = layout
            .map(|l| {
                l.segments
                    .iter()
                    .map(|s| {
                        std::fs::metadata(format!("{}/{}", l.dir, s.file))
                            .map(|md| md.len())
                            .unwrap_or(0)
                    })
                    .sum()
            })
            .unwrap_or(0);
        /* W1.8: what this store's LOAD had to give up — the damage that used
         * to be visible only as stderr. Every counter is measured by the
         * loader (SegScan); `flush_errors` is this handle's own tally. */
        let recovery = {
            let mut v = layout
                .map(|l| l.recovery.to_json())
                .unwrap_or_else(|| Recovery::default().to_json());
            if let Some(o) = v.as_object_mut() {
                o.insert("flush_errors".to_string(), serde_json::json!(cma.flush_errors));
                o.insert(
                    "last_flush_error".to_string(),
                    match &cma.flush_error {
                        Some((code, msg)) => serde_json::json!({"code": code, "message": msg}),
                        None => serde_json::Value::Null,
                    },
                );
            }
            v
        };
        let out = serde_json::json!({
            "records": cma.cma.len(),
            "segmented": layout.is_some(),
            "segments": segments,
            "versions": versions,
            "bytes": bytes,
            /* W3.3: how many segment PAYLOADS this handle has read. 0 right
             * after a normal open (and until the first row access) — the
             * cheap, checkable proof that open is O(manifest) now. NOTE:
             * stats() is itself a row access, so by the time this object is
             * returned the load has run; `loadState()` reads the same
             * counter WITHOUT triggering it. */
            "segments_read": cma.segments_read,
            "loaded": cma.loaded,
            /* W2.2: this handle holds a SHARED lock and may not write. */
            "readonly": cma.readonly,
            /* W2.1: who holds the brain. `mode` is "exclusive" for a
             * writable handle, "shared" for a readonly one. */
            "lock": match &cma.lock {
                Some(l) => serde_json::json!({
                    "mode": match l.mode() {
                        brain_lock::Mode::Exclusive => "exclusive",
                        brain_lock::Mode::Shared => "shared",
                    },
                    "path": l.path(),
                    /* false = no lock could be taken on this path (a broken
                     * or read-only location); the handle says so instead of
                     * pretending it arbitrates. */
                    "arbitrated": l.degraded().is_none(),
                }),
                None => serde_json::Value::Null,
            },
            "evicted_total": layout.map(|l| l.evicted_total).unwrap_or(0),
            "last_decay_at": layout.map(|l| l.last_decay_at).unwrap_or(0),
            "needs_compaction": cma.cma.layout_dirty,
            "dirty": cma.cma.dirty_count(),
            /* B5: ranking knobs, so a brain can report how it ranks. */
            "ef_search": cma.cma.ef_search.unwrap_or(0),
            "diversity_floor": cma.cma.diversity_floor,
            /* B6: session working set (never serialized). */
            "working": cma.cma.working_len(),
            /* B7.2: compaction generation (bumped on every rewrite). */
            "epoch": layout.map(|l| l.epoch).unwrap_or(0),
            /* M2 (Brain Core II): the DeathLedger — the most recent
             * physical deaths in this store (text clipped by the CMA's
             * own cap; at = unix seconds). */
            "recentDeaths": cma.cma.deaths.iter().rev().take(16).collect::<Vec<_>>(),
            /* W1.8: the damage-visibility block (present on EVERY stats()). */
            "recovery": recovery,
        });
        let c = CString::new(out.to_string()).unwrap_or_default();
        qjs::sofuu_js_new_string(ctx, c.as_ptr())
    }

    /// B3.4: `cma.compact()` — force a rewrite of the live records into
    /// fresh segments (drops superseded versions and tombstones the store
    /// no longer needs). Returns 0 on success.
    unsafe extern "C" fn js_cma_compact(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        /* W2.2: a readonly handle holds a SHARED lock. Buffering a write it
         * can never flush would be silent data loss, so refuse at the entry
         * point instead of at flush time. */
        if cma.readonly {
            return throw_coded(ctx, E_LOCKED, "this brain is open readonly (shared lock) — write refused");
        }

        cma.cma.layout_dirty = true;
        qjs::sofuu_js_new_int32(ctx, cma_flush(cma))
    }

    /// B1.1 export: the full record view plus the flat vector table and the
    /// embedding manifest — the portable payload B2's brain cards build on.
    /// `{schema, exported_at, embedding:{id,dimension,artifact}, total,
    /// records:[…], vectors:[…]}`.
    unsafe extern "C" fn js_cma_export(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        let filter = crate::memory::cma::ListFilter::default();
        let records: serde_json::Value =
            serde_json::from_str(&cma.cma.list_json(&filter))
                .unwrap_or_else(|_| serde_json::json!({"records":[]}));
        /* The SAME view's vectors — records[i] pairs with vectors[i*dim..]
         * (the unfiltered table would misalign any store with tombstones). */
        let vectors = cma.cma.flat_vectors_for(&filter);
        let (input, artifact) = if cma.embedding_id == crate::embedding::MODEL_ID {
            (
                crate::embedding::INPUT_EMBEDDER_ID,
                crate::embedding::model_artifact_id(),
            )
        } else if cma.embedding_id == crate::embedding::semantic_v2::MODEL_ID_V2 {
            (
                crate::embedding::semantic_v2::INPUT_EMBEDDER_ID_V2,
                crate::embedding::semantic_v2::model_artifact_id_v2(),
            )
        } else {
            ("", String::new())
        };
        let out = serde_json::json!({
            "schema": "sofuu-brain-card@2",
            "exported_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            "embedding": {
                "id": cma.embedding_id,
                "input": input,
                "dimension": cma.vec_dim,
                "artifact": artifact,
            },
            /* The view's count — the card must describe what it carries,
             * not the store's raw record count (tombstones excluded). */
            "total": vectors.len() / cma.vec_dim.max(1),
            "records": records.get("records").cloned().unwrap_or_else(|| serde_json::json!([])),
            "vectors": vectors,
        });
        let c = CString::new(out.to_string()).unwrap_or_default();
        qjs::sofuu_js_new_string(ctx, c.as_ptr())
    }

    unsafe extern "C" fn js_cma_forget(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 1 {
            return throw_coded(ctx, E_TYPE, "Expected vector_index");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        /* W2.2: a readonly handle holds a SHARED lock. Buffering a write it
         * can never flush would be silent data loss, so refuse at the entry
         * point instead of at flush time. */
        if cma.readonly {
            return throw_coded(ctx, E_LOCKED, "this brain is open readonly (shared lock) — write refused");
        }

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
        /* W2.2: a readonly handle holds a SHARED lock. Buffering a write it
         * can never flush would be silent data loss, so refuse at the entry
         * point instead of at flush time. */
        if cma.readonly {
            return throw_coded(ctx, E_LOCKED, "this brain is open readonly (shared lock) — write refused");
        }

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

    /// W6.1 (AUDIT-2026-10-29): `cma.flush()` → `{ok:true, recordsWritten:N,
    /// segmentsSealed:N}` and a CODED throw when the flush fails.
    ///
    /// Before this, flush answered a bare bool: a failed flush never threw
    /// (the caller had to notice `false`), and nothing reported how many
    /// records or segments the write actually produced. The object is still
    /// truthy, so `if (b.flush())` / `local = b.flush()` callers (e.g.
    /// plugins.js's storage facade) keep behaving exactly as before.
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
        if r != 0 {
            /* The failure path already recorded WHY (code + message). */
            let (code, msg) = cma
                .flush_error
                .clone()
                .unwrap_or((E_ENOSPC, "flush failed".to_string()));
            let path = cma.mind_path.clone();
            emit_memory_event(
                ctx,
                serde_json::json!({
                    "type": "flush_error",
                    "code": code,
                    "message": msg,
                    "path": path,
                }),
            );
            return throw_coded(ctx, code, &format!("memory.flush: {msg}"));
        }
        let stats = cma.last_flush.unwrap_or_default();
        let obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"ok".as_ptr(),
            qjs::sofuu_js_new_bool(ctx, 1),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"recordsWritten".as_ptr(),
            qjs::sofuu_js_new_uint32(ctx, stats.records_written as u32),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"segmentsSealed".as_ptr(),
            qjs::sofuu_js_new_uint32(ctx, stats.segments_sealed as u32),
        );
        obj
    }

    /// W3.3: the lazy-open counter, read WITHOUT triggering the load — the
    /// accessor that lets a test (or `sofuu doctor`) prove a normal open read
    /// 0 segment payloads and a `verify_full` open read them all.
    unsafe extern "C" fn js_cma_load_state(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        /* Deliberately NOT shell_from_this: that materialises. */
        let p = qjs::JS_GetOpaque2(ctx, this_val, CMA_CLASS_ID.load(Ordering::Relaxed))
            as *mut CmaShell;
        if p.is_null() {
            return qjs::sofuu_js_exception();
        }
        let cma = &*p;
        let obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"loaded".as_ptr(),
            qjs::sofuu_js_new_bool(ctx, cma.loaded as i32),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"segments_read".as_ptr(),
            qjs::sofuu_js_new_uint32(ctx, cma.segments_read as u32),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"pending".as_ptr(),
            qjs::sofuu_js_new_bool(ctx, cma.pending.is_some() as i32),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"readonly".as_ptr(),
            qjs::sofuu_js_new_bool(ctx, cma.readonly as i32),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"closed".as_ptr(),
            qjs::sofuu_js_new_bool(ctx, cma.closed as i32),
        );
        obj
    }

    /// W2.1/W2.2: `cma.close()` — flush a writable handle, then RELEASE the
    /// lock so the next process (or the next handle here) can open the brain
    /// without waiting for the GC to collect this object.
    ///
    /// A readonly handle never flushes: closing it is nothing but releasing
    /// the shared lock, so an open→read→close cycle leaves the store
    /// byte-identical. The flush failure of a writable handle is thrown as
    /// the same CODED error `flush()` throws, after the lock is released.
    unsafe extern "C" fn js_cma_close(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        if cma.closed {
            return qjs::sofuu_js_new_bool(ctx, 1); /* idempotent */
        }
        let mut failure: Option<(&'static str, String)> = None;
        if !cma.readonly && cma_flush(cma) != 0 {
            failure = cma
                .flush_error
                .clone()
                .or(Some((E_ENOSPC, "flush failed".to_string())));
        }
        cma.closed = true;
        if let Some(l) = cma.lock.as_mut() {
            /* Released BEFORE the error is thrown: a failed flush must not
             * also wedge the brain for the next opener. */
            l.release();
        }
        if let Some((code, msg)) = failure {
            let path = cma.mind_path.clone();
            emit_memory_event(
                ctx,
                serde_json::json!({
                    "type": "flush_error",
                    "code": code,
                    "message": msg,
                    "path": path,
                }),
            );
            return throw_coded(ctx, code, &format!("memory.close: {msg}"));
        }
        qjs::sofuu_js_new_bool(ctx, 1)
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
            return throw_coded(ctx, E_TYPE, "memory.migrate(path) expected a string");
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
            layout: None,
            legacy_encryption: false,
            last_flush: None,
            flush_error: None,
            flush_errors: 0,
            last_outcome: None,
            /* A scratch shell owns no store yet, so it takes no lock and may
             * write (it is the migration writer); nothing reads lazily. */
            lock: None,
            readonly: false,
            loaded: true,
            pending: None,
            graph_snapshot_epoch: 0,
            graph_snapshot_records: None,
            last_sweep_epoch: None,
            flushes_since_sweep: 0,
            published_manifest_stamp: None,
            segments_read: 0,
            recovery_pending: false,
            closed: false,
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

    /// M0 (Brain Core II): entity-namespaced TURN record — the agent
    /// scope's private memory accumulates: same namespace join
    /// (entity_name) as rememberEntity, but episodic decay/capacity
    /// treatment (the bare-name rememberEntity upsert collapsed every
    /// turn into one overwritten row).
    unsafe extern "C" fn js_cma_remember_entity_turn(
        ctx: *mut JSContext,
        this_val: JSValueConst,
        argc: c_int,
        argv: *const JSValueConst,
    ) -> JSValue {
        if argc < 4 {
            return throw_coded(ctx, E_TYPE, "Expected vec, text, entityName, entityType");
        }
        let Some(cma) = shell_from_this(ctx, this_val) else {
            return qjs::sofuu_js_exception();
        };
        /* W2.2: a readonly handle holds a SHARED lock. Buffering a write it
         * can never flush would be silent data loss, so refuse at the entry
         * point instead of at flush time. */
        if cma.readonly {
            return throw_coded(ctx, E_LOCKED, "this brain is open readonly (shared lock) — write refused");
        }

        let Some(vec_view) = cma_vec_view(ctx, *argv) else {
            return qjs::sofuu_js_exception();
        };
        let vec_slice = vec_view.as_slice();
        if vec_slice.len() != cma.vec_dim {
            return throw_coded(
                ctx,
                E_DIM,
                &dim_mismatch_message("rememberEntityTurn", vec_slice.len(), cma.vec_dim),
            );
        }
        let text = qjs::sofuu_js_to_cstring(ctx, *argv.add(1));
        let entity_name = qjs::sofuu_js_to_cstring(ctx, *argv.add(2));
        let entity_type = qjs::sofuu_js_to_cstring(ctx, *argv.add(3));
        let idx = cma.cma.remember_entity_scoped(
            vec_slice,
            cstr_opt(text).as_deref().unwrap_or(""),
            cstr_opt(entity_name).as_deref().unwrap_or(""),
            cstr_opt(entity_type).as_deref().unwrap_or("unknown"),
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

    thread_local! {
        static CMA_PROTO_FUNCS: [qjs::JSCFunctionListEntry; 26] = [
            cfunc_entry(c"remember", 4, js_cma_remember),
                cfunc_entry(c"lastRememberOutcome", 0, js_cma_last_remember_outcome),
            cfunc_entry(c"learn", 3, js_cma_learn),
            cfunc_entry(c"list", 1, js_cma_list),
            cfunc_entry(c"export", 0, js_cma_export),
            cfunc_entry(c"importRecords", 2, js_cma_import),
            cfunc_entry(c"stats", 0, js_cma_stats),
            /* W2.1/W2.2: flush + release the lock (the JS close() path). */
            cfunc_entry(c"close", 0, js_cma_close),
            /* W3.3: the lazy-open counter, without triggering the load. */
            cfunc_entry(c"loadState", 0, js_cma_load_state),
            cfunc_entry(c"compact", 0, js_cma_compact),
            cfunc_entry(c"retainAgainst", 1, js_cma_retain_against),
            cfunc_entry(c"rememberEntity", 4, js_cma_remember_entity),
            cfunc_entry(c"recall", 2, js_cma_recall),
            cfunc_entry(c"search", 2, js_cma_search),
            cfunc_entry(c"kvHints", 2, js_cma_kv_hints),
            cfunc_entry(c"forget", 1, js_cma_forget),
            cfunc_entry(c"flush", 0, js_cma_flush),
            cfunc_entry(c"decayTick", 1, js_cma_decay_tick),
            cfunc_entry(c"retain", 0, js_cma_retain),
            cfunc_entry(c"consolidate", 0, js_cma_consolidate),
            cfunc_entry(c"count", 0, js_cma_count),
            cfunc_entry(c"markPositive", 1, js_cma_mark_positive),
            cfunc_entry(c"rememberWorking", 5, js_cma_remember_working),
            cfunc_entry(c"rememberEntityTurn", 4, js_cma_remember_entity_turn),
            cfunc_entry(c"promoteWorking", 1, js_cma_promote_working),
            cfunc_entry(c"clearWorking", 0, js_cma_clear_working),
        ];
        static CMA_MODULE_FUNCS: [qjs::JSCFunctionListEntry; 4] = [
            cfunc_entry(c"open", 2, js_cma_open),
            cfunc_entry(c"migrate", 1, js_memory_migrate),
            cfunc_entry(c"embeddingInfo", 0, js_memory_embedding_info),
            cfunc_entry(c"inspect", 1, js_memory_inspect),
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
            /* C2: create_dir (single level) silently failed for a nested
             * store path whose parent did not exist yet — every page save
             * then failed with no directory to write into, and the old
             * kv_flush reported success anyway. create_dir_all matches what
             * the brain path already does. */
            let _ = std::fs::create_dir_all(path);
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
        /* C2: replay a crashed commit (page or index) before reading the
         * index, so the store opens on the newest committed state. */
        let recovery = crate::rt::wal::recover(path);
        if recovery.journal_found {
            eprintln!(
                "[sofuu/kv] recovered interrupted commit ({} rolled forward, {} rolled back)",
                recovery.rolled_forward, recovery.rolled_back
            );
        }

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
        let mut index_parsed = false;
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
                    index_parsed = true;
                } else {
                    /* C2: a corrupt index used to read as "no pages" — the
                     * whole store silently disappeared. Say so; the orphan
                     * adoption below then re-links the pages that are still
                     * on disk. */
                    eprintln!(
                        "[sofuu/kv] index.json is unreadable ({} bytes) — adopting pages from disk",
                        data.len()
                    );
                }
                qjs::sofuu_js_free_value(ctx, meta_obj);
            }
        }
        /* C2: a crash between a page write and the index flush leaves the
         * page on disk but absent from the index — invisible forever. Adopt
         * every page file the index does not know about; they rank neutral
         * (zeroed k_summary) until the next flush records them. */
        let mut adopted = 0usize;
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let Some(name) = name.to_str() else { continue };
                let Some(rest) = name.strip_prefix("page_") else { continue };
                let Some(id_str) = rest.strip_suffix(".qtsq") else { continue };
                let Ok(page_id) = id_str.parse::<u32>() else { continue };
                if page_id == 0 || kv.pages.iter().any(|p| p.page_id == page_id) {
                    continue;
                }
                kv.pages.push(PageSummary {
                    page_id,
                    strength: 0.0,
                    created_at: 0,
                    k_summary: [0.0; KV_SUMMARY_DIM],
                });
                adopted += 1;
            }
        }
        if adopted > 0 {
            kv.pages.sort_by_key(|p| p.page_id);
            eprintln!(
                "[sofuu/kv] adopted {adopted} page(s) missing from the index{}",
                if index_parsed { "" } else { " (index was unreadable)" }
            );
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
        /* C2: the index is the commit record for every page on disk — a bare
         * fs::write here meant a crash mid-write left a truncated index.json
         * that kv_store_open silently read as "no pages", orphaning the whole
         * store. Stage + journal + durable publish, exactly like a page. */
        let tmp_path = format!("{idx_path}.tmp");
        let wrote = std::fs::write(&tmp_path, bytes).is_ok();
        let mut journal = crate::rt::wal::Journal::begin(&kv.file_path);
        let recorded = journal
            .as_mut()
            .map(|j| j.record(&tmp_path, &idx_path))
            .unwrap_or(false);
        let published = wrote && crate::rt::wal::publish(&tmp_path, &idx_path);
        if published {
            if recorded {
                if let Some(j) = journal {
                    j.commit();
                }
            }
        } else {
            let _ = std::fs::remove_file(&tmp_path);
        }

        qjs::sofuu_js_free_cstring(kv.ctx, json_c);
        qjs::sofuu_js_free_value(kv.ctx, str_val);
        qjs::sofuu_js_free_value(kv.ctx, root);

        if published { 0 } else { -1 }
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

        /* cma_vec_arg throws the CODED refusal (E_TYPE / E_NONFINITE). */
        let (k_ptr, k_len) = cma_vec_arg(ctx, *argv);
        if k_ptr.is_null() {
            return qjs::sofuu_js_exception();
        }
        let (v_ptr, v_len) = cma_vec_arg(ctx, *argv.add(1));
        if v_ptr.is_null() {
            return qjs::sofuu_js_exception();
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
            return qjs::sofuu_js_exception();
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
            return qjs::sofuu_js_exception();
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
        qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"memory".as_ptr(), qjs::sofuu_js_dup_value(ctx, mem_obj));
        qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"brain".as_ptr(), mem_obj);
        qjs::sofuu_js_free_value(ctx, sofuu_obj);

        /* R1/T1.3 flat surface (src/js/memory_api.js): the contract documents
         * `sofuu.memory.remember({...})`, and a QTSQ-free build already
         * presents exactly that name set (typed-refusal stubs). Install the
         * same flat methods on the module object here — `sofuu.brain` is an
         * alias of the same object — delegating to the last-opened brain or
         * `{handle}`. Compiled out on capability-gated builds by this
         * function's own cfg, so their stubs stay the surface. */
        const MEMORY_FLAT_API_JS: &str = include_str!("../../../../src/js/memory_api.js");
        crate::shipped::eval_shipped(
            ctx as *mut std::ffi::c_void,
            MEMORY_FLAT_API_JS,
            "<memory-api>",
        );
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

    /// W3.3 CONTRACT CHANGE: a normal `cma_open` now reads the MANIFEST only
    /// and defers every segment payload to the first row access. Most tests
    /// here inspect the store's in-memory rows IMMEDIATELY after opening
    /// (`shell.cma.len()`, `shell.cma.records`, `layout.recovery`, …), which
    /// is exactly what the pre-W3.3 open had already done — so they are
    /// written against this helper, which performs that first access. It is
    /// the OLD open contract, kept for tests that are not about laziness;
    /// the tests that PROVE the laziness (`w33_*`, `w22_*`, `w21_*`) call
    /// `cma_open` directly.
    #[cfg(test)]
    unsafe fn cma_open_loaded(
        path: &str,
        vec_dim: usize,
        requested_embedding_id: Option<&str>,
    ) -> Option<Box<CmaShell>> {
        let mut shell = cma_open(path, vec_dim, requested_embedding_id)?;
        if materialize(&mut shell).is_err() {
            return None;
        }
        Some(shell)
    }

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

    /// W5.3 (Rust surface): a decay policy configured by the owner must
    /// survive the flush, so a reopen that says nothing about decay inherits
    /// it instead of silently reverting to the default curve. This is the
    /// exact failure the item names — a brain that stopped forgetting on
    /// schedule after a restart — and it is invisible unless a reopen is
    /// compared with the config that wrote the store.
    #[test]
    fn w5_3_a_flushed_decay_policy_is_inherited_by_a_silent_reopen() {
        const DIM: usize = 64;
        let dir = fresh_test_dir("w53-policy");
        let p = dir.join("brain.qtsq").to_string_lossy().to_string();
        unsafe {
            let mut sh = cma_open(&p, DIM, None).expect("a fresh brain opens");
            assert!(
                sh.cma
                    .set_decay_params(crate::memory::cma::DECAY_TAG_TTL, 50.0),
                "the fixture must be able to set a TTL policy"
            );
            for i in 0..3usize {
                let v = vec![i as f32; DIM];
                sh.cma.remember(&v, "ttl row", "user", 0);
            }
            assert_eq!(brain_flush_segmented(&mut sh), 0, "flush must succeed");
            drop(sh);

            /* A silent reopen: nothing here mentions decay. */
            let mut sh2 = cma_open(&p, DIM, None).expect("the brain reopens");
            materialize(&mut sh2).expect("materialise the lazy open");
            assert_eq!(
                sh2.cma.decay_params(),
                (crate::memory::cma::DECAY_TAG_TTL, 50.0),
                "W5.3: a reopen with no decay config must inherit the flushed policy"
            );
        }
    }

    /// W3.1 (Rust surface, the WIRING not the primitive): a flush must leave
    /// a graph snapshot beside the segments for the generation it published,
    /// a reopen must adopt it, and a corrupt snapshot must degrade to a
    /// correct rebuild rather than to a wrong answer. The primitive is tested
    /// in hnsw.rs; this pins the plumbing that was missing.
    #[test]
    fn w3_1_a_flush_persists_the_graph_and_a_corrupt_one_falls_back_to_a_rebuild() {
        const DIM: usize = 32;
        let dir = fresh_test_dir("w31-wiring");
        let p = dir.join("brain.qtsq").to_string_lossy().to_string();
        unsafe {
            let mut sh = cma_open(&p, DIM, None).expect("a fresh brain opens");
            /* Distinct per record: a cyclic fixture collapses under the
             * dedup gate (that is the gate working, not a fixture) and the
             * count below would be measuring the merge, not the snapshot. */
            for i in 0..40usize {
                let v: Vec<f32> = (0..DIM).map(|d| i as f32 + d as f32 * 0.01).collect();
                sh.cma.remember(&v, &format!("snapshot row {i}"), "user", 0);
            }
            assert_eq!(sh.cma.len(), 40, "40 distinct rows must survive the ingest");
            assert_eq!(brain_flush_segmented(&mut sh), 0, "flush must succeed");
            let epoch = sh.layout.as_ref().expect("layout").epoch;
            let graph = crate::rt::brain_store::graph_path(&p, epoch);
            assert!(
                std::path::Path::new(&graph).is_file(),
                "W3.1: a flush must persist the graph snapshot at {graph}"
            );
            drop(sh);

            /* A good snapshot: the reopen adopts it and answers the same. */
            let mut good = cma_open(&p, DIM, None).expect("reopen");
            materialize(&mut good).expect("materialise");
            assert_eq!(good.cma.len(), 40, "the snapshot-backed open keeps every record");

            /* A corrupt snapshot must not be trusted: the rebuild path has to
             * produce the same brain, never a broken one. */
            std::fs::write(&graph, b"this is not a snapshot").expect("corrupt the snapshot");
            let mut rebuilt = cma_open(&p, DIM, None).expect("reopen with a corrupt snapshot");
            materialize(&mut rebuilt).expect("a corrupt snapshot must fall back, not fail");
            assert_eq!(
                rebuilt.cma.len(),
                40,
                "W3.1: a corrupt snapshot must degrade to a rebuild, not lose records"
            );
        }
    }

    /// W3.4 (Rust surface): flushing must not accumulate superseded
    /// generations. The primitive is tested in brain_store; this pins the
    /// TRIGGER — a store that flushes repeatedly must not grow a file per
    /// flush, which is what the plan's 600 s orphan cliff looked like in
    /// practice.
    #[test]
    fn w3_4_repeated_flushes_do_not_accumulate_store_generations() {
        const DIM: usize = 64;
        let dir = fresh_test_dir("w34-trigger");
        let p = dir.join("brain.qtsq").to_string_lossy().to_string();
        unsafe {
            let mut sh = cma_open(&p, DIM, None).expect("a fresh brain opens");
            for round in 0..4usize {
                for i in 0..40usize {
                    let v = vec![(round * 40 + i) as f32; DIM];
                    sh.cma.remember(&v, "round row", "user", 0);
                }
                assert_eq!(brain_flush_segmented(&mut sh), 0, "flush {round} must succeed");
                let seg_dir = crate::rt::brain_store::segments_dir(&p);
                let files = crate::rt::brain_store::list_segment_files(&seg_dir).len();
                let referenced = sh.layout.as_ref().map(|l| l.segments.len()).unwrap_or(0);
                assert_eq!(
                    files, referenced,
                    "after flush {round}: {files} segment file(s) on disk but the manifest                      references {referenced} — superseded generations must be reclaimed"
                );
            }
        }
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
            cma_open_loaded(p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
        }
        .expect("fresh semantic shell on a nonexistent file");
        for text in texts {
            let v = crate::embedding::semantic_v1(text).expect("baked model");
            let r = unsafe { cma_remember(&mut shell, &v, Some(text), Some("memory"), 7, None) };
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
            cma_open_loaded(&p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
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

        /* W2.1: the migrating handle still holds the brain's lock, so it is
         * closed before the restart. */
        drop(shell);

        /* Restart: the migrated file reopens in the same space — no second
         * migration, no leftover sidecars. */
        let again = unsafe {
            cma_open_loaded(&p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
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
            cma_open_loaded(&p, crate::embedding::SEMANTIC_DIM, Some(v2::MODEL_ID_V2))
        }
        .expect("fresh SEM2 shell on a nonexistent file");
        assert_eq!(shell.embedding_id, v2::MODEL_ID_V2);
        for text in &texts {
            let v = v2::semantic_v2(text).expect("baked SEM2 model");
            let r = unsafe { cma_remember(&mut shell, &v, Some(text), Some("memory"), 7, None) };
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

        /* W2.1: close the writer before reopening the same brain. */
        drop(shell);

        /* Same space reopens; the index retrieves. */
        let mut again = unsafe {
            cma_open_loaded(&p, crate::embedding::SEMANTIC_DIM, Some(v2::MODEL_ID_V2))
        }
        .expect("v2 brain reopens");
        assert_eq!(again.cma.len(), texts.len());
        let q = v2::semantic_v2("payment webhook retries").unwrap();
        assert!(!again.cma.recall(&q, 5).is_empty());

        /* §12: a v1 semantic open of the SAME file must be refused —
         * both are 64-dim, so only the manifest can catch this. */
        assert!(
            unsafe {
                cma_open_loaded(&p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
            }
            .is_none(),
            "v1 open of a v2 brain must refuse (dimension collision)"
        );

        /* §12: the reverse — a v2 open of a v1 brain is refused too. */
        let v1_path = dir.join("brain.qtsq");
        build_semantic_brain(&v1_path, &texts);
        assert!(
            unsafe {
                cma_open_loaded(
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
                cma_open_loaded(
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
            cma_open_loaded(
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
        /* W2.1: the refused open still took (and released) the `.lock`, so
         * "no sidecar" is an assertion about STORE files. */
        let store_entries: std::collections::BTreeSet<String> = dir_entries(&dir)
            .into_iter()
            .filter(|e| !e.ends_with(".lock"))
            .collect();
        assert_eq!(
            store_entries, entries_before,
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
        assert!(unsafe { cma_open_loaded(p, crate::embedding::HASH_DIM, None) }.is_none());
        assert!(
            unsafe { cma_open_loaded(p, crate::embedding::HASH_DIM, Some(crate::embedding::INPUT_EMBEDDER_ID)) }
                .is_none()
        );
        assert_eq!(std::fs::read(p).unwrap(), before);

        /* The matching request still opens (nothing was damaged). */
        let mut shell = unsafe {
            cma_open_loaded(p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
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
            unsafe { cma_open_loaded(p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID)) }
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
            unsafe { cma_open_loaded(p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID)) }
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

        let mut shell = unsafe { cma_open_loaded(p, crate::embedding::HASH_DIM, None) }
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
            cma_remember(&mut shell, &v, Some("first real memory"), Some("memory"), 1, None)
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

        let mut shell = unsafe { cma_open_loaded(p, crate::embedding::HASH_DIM, None) }
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
        let first = unsafe { cma_open_loaded(p, crate::embedding::HASH_DIM, None) };
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
        let mut shell = unsafe { cma_open_loaded(p, crate::embedding::HASH_DIM, None) }
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

        let shell = unsafe { cma_open_loaded(p, crate::embedding::HASH_DIM, None) }
            .expect("legacy brain opens as hash-v1 without migrating");
        assert_eq!(shell.embedding_id, crate::embedding::INPUT_EMBEDDER_ID);
        assert_eq!(shell.cma.len(), 2);
        let js = shell.cma.records_json();
        assert!(js.contains("legacy memory one") && js.contains("legacy memory two"));
        /* A plain hash open must not migrate or touch anything. W2.1: the
         * writable open now creates `<brain>.lock` beside the brain, so the
         * claim is about STORE files — the lock is the arbitration the open
         * took, not a change to the brain. */
        assert_eq!(std::fs::read(&path).unwrap(), before);
        let store_entries: std::collections::BTreeSet<String> = dir_entries(&dir)
            .into_iter()
            .filter(|e| !e.ends_with(".lock"))
            .collect();
        assert_eq!(store_entries, entries_before);
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
            cma_open_loaded(&p, crate::embedding::SEMANTIC_DIM, Some(crate::embedding::MODEL_ID))
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

    /* ── C2: write-ahead journal + durable publish ─────────────────── */

    /// A completed brain save must leave no journal and no staging file.
    #[test]
    fn brain_commit_leaves_no_journal_or_tmp() {
        let dir = fresh_test_dir("c2-clean");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let mut shell = unsafe {
            cma_open_loaded(&p, crate::embedding::HASH_DIM, None)
        }
        .expect("fresh brain");
        let v = crate::embedding::hash_v1_features("the deploy checklist lives here");
        assert!(unsafe { cma_remember(&mut shell, &v, Some("deploy"), Some("memory"), 5, None) } >= 0);
        assert_eq!(unsafe { cma_flush_to(&mut shell, &p) }, 0);

        let entries = dir_entries(&dir);
        assert!(
            !entries.iter().any(|e| e.contains(".sofuu-wal")),
            "journal left behind after a clean commit: {entries:?}"
        );
        assert!(
            !entries.iter().any(|e| e.ends_with(".tmp")),
            "staging file left behind after a clean commit: {entries:?}"
        );
        assert!(entries.contains("brain.qtsq"), "{entries:?}");
    }

    /// A crash between "staged file written" and "rename" must be rolled
    /// FORWARD on the next open: the newer brain wins, the journal clears.
    #[test]
    fn crashed_brain_commit_rolls_forward_on_open() {
        let dir = fresh_test_dir("c2-forward");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();

        /* Old committed state: one record. */
        let mut old = unsafe { cma_open_loaded(&p, crate::embedding::HASH_DIM, None) }.expect("old brain");
        let v1 = crate::embedding::hash_v1_features("first record");
        assert!(unsafe { cma_remember(&mut old, &v1, Some("one"), Some("memory"), 5, None) } >= 0);
        assert_eq!(unsafe { cma_flush_to(&mut old, &p) }, 0);
        /* W2.1: `p` is locked by `old` for as long as that handle lives, so
         * it is closed before the recovery open (a second writable open
         * would be refused with E_LOCKED, not silently accepted). */
        drop(old);

        /* Newer state staged to a sibling and never renamed (the crash). */
        let staged_path = dir.join("brain.qtsq.staged");
        let staged = staged_path.to_str().unwrap().to_string();
        let mut newer = unsafe { cma_open_loaded(&staged, crate::embedding::HASH_DIM, None) }.expect("staged brain");
        for text in ["first record", "second record"] {
            let v = crate::embedding::hash_v1_features(text);
            assert!(unsafe { cma_remember(&mut newer, &v, Some(text), Some("memory"), 5, None) } >= 0);
        }
        assert_eq!(unsafe { cma_flush_to(&mut newer, &staged) }, 0);

        let d = dir.to_string_lossy().to_string();
        {
            let mut j = crate::rt::wal::Journal::begin(&d).expect("journal");
            assert!(j.record(&staged, &p));
            /* Drop = crash: the rename never happens. */
        }

        /* The open must replay the journal before reading. */
        let reopened = unsafe { cma_open_loaded(&p, crate::embedding::HASH_DIM, None) }.expect("recovered brain");
        assert_eq!(reopened.cma.len(), 2, "roll-forward must expose the staged state");
        let entries = dir_entries(&dir);
        assert!(
            !entries.iter().any(|e| e.contains(".sofuu-wal")),
            "journal must be cleared after recovery: {entries:?}"
        );
        assert!(!staged_path.exists(), "staging file must be consumed");
    }

    /// A page file on disk that the index does not know about (a crash
    /// between the page write and the index flush) must be adopted — and
    /// crucially, the next save must NOT reuse its id and clobber it.
    #[test]
    fn kv_open_adopts_orphan_pages() {
        use sofuu_ffi::qjs::CtxPtr;

        let dir = fresh_test_dir("c2-orphan");
        let root = dir.join("kv");
        std::fs::create_dir_all(&root).unwrap();
        /* An orphan page: written, never indexed. Content is irrelevant —
         * adoption keys off the filename. */
        std::fs::write(root.join("page_00000001.qtsq"), b"orphan").unwrap();

        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);
            kv_register(ctx);

            let script = format!(
                "var kv = sofuu.kv.open('{}', {{nLayers:1, nHeads:1, headDim:8}});\n\
                 var k = new Float32Array(8); var v = new Float32Array(8);\n\
                 k[0] = 1.0;\n\
                 globalThis.out = kv.save(k, v, 1);",
                root.to_str().unwrap()
            );
            let script_c = CString::new(script).unwrap();
            let r = qjs::JS_Eval(
                ctx,
                script_c.as_ptr(),
                script_c.as_bytes().len(),
                c"<c2-orphan>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            assert!(!qjs::is_exception(r), "script threw");
            let global = qjs::sofuu_js_get_global_object(ctx);
            let out = qjs::sofuu_js_get_property_str(ctx, global, c"out".as_ptr());
            qjs::sofuu_js_free_value(ctx, global);
            let mut id: i32 = 0;
            qjs::JS_ToInt32(ctx, &mut id, out);
            qjs::sofuu_js_free_value(ctx, out);
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
            /* Adopted page owns id 1, so the new save must be id 2. Without
             * adoption this returns 1 and overwrites the orphan. */
            assert_eq!(id, 2, "orphan page was not adopted (id reused)");
            assert_eq!(
                std::fs::read(root.join("page_00000001.qtsq")).unwrap(),
                b"orphan",
                "adoption must not rewrite the orphan file"
            );
        }
    }

    /* ── B4 (ROADMAP-BRAIN-CMA): machine-local key ─────────────────── */

    /// Write a single-file brain the way the PRE-B4 code did: encrypted with
    /// the legacy fixed password. Used to prove the migration path.
    unsafe fn write_legacy_encrypted_brain(path: &str, dim: usize, texts: &[&str]) {
        let n = texts.len();
        let mut flat: Vec<f32> = Vec::new();
        for (i, t) in texts.iter().enumerate() {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            flat.extend_from_slice(&v);
            let _ = t;
        }
        let dims: [u32; 2] = [n as u32, dim as u32];
        let records: Vec<serde_json::Value> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| {
                serde_json::json!({
                    "vector_index": i, "kv_page_id": 0, "strength": 1.0,
                    "age_seconds": 0, "half_life": 86400, "tier": 1,
                    "text": t, "role": "user", "total_recalls": 0, "positive_recalls": 0,
                    "created_at": 0, "last_recalled_at": 0, "source": "",
                })
            })
            .collect();
        let meta = serde_json::json!({
            "schema": "sofuu-cma@2",
            "embedding": {
                "id": crate::embedding::INPUT_EMBEDDER_ID,
                "input": crate::embedding::INPUT_EMBEDDER_ID,
                "dimension": dim,
                "artifact": "",
            },
            "records": records,
        })
        .to_string();

        let container = qtsq::qtsq_ctx_alloc();
        assert!(!container.is_null());
        assert_eq!(qtsq::qtsq_container_create(container), QTSQ_OK);
        (*container).set_codec(qtsq::QTSQ_CODEC_QTC);

        let sub = qtsq::qtsq_ctx_alloc();
        assert_eq!(qtsq::qtsq_init(sub), QTSQ_OK);
        (*sub).set_codec(qtsq::QTSQ_CODEC_QTC);
        assert_eq!(
            qtsq::qtsq_compress_tensor(sub, flat.as_ptr(), flat.len(), dims.as_ptr(), 2, QTSQ_LEVEL_FULL),
            QTSQ_OK
        );
        assert_eq!(
            qtsq::qtsq_container_add_stream(container, sub, c"memories".as_ptr()),
            QTSQ_OK
        );
        qtsq::qtsq_free(sub);
        qtsq::qtsq_ctx_free(sub);

        let jsub = qtsq::qtsq_ctx_alloc();
        assert_eq!(qtsq::qtsq_init(jsub), QTSQ_OK);
        (*jsub).set_codec(qtsq::QTSQ_CODEC_QTC);
        let cmeta = CString::new(meta).unwrap();
        assert_eq!(
            qtsq::qtsq_compress_json(jsub, cmeta.as_ptr(), cmeta.as_bytes().len()),
            QTSQ_OK
        );
        assert_eq!(
            qtsq::qtsq_container_add_stream(container, jsub, c"metadata".as_ptr()),
            QTSQ_OK
        );
        qtsq::qtsq_free(jsub);
        qtsq::qtsq_ctx_free(jsub);

        assert_eq!(qtsq::qtsq_container_pack(container), QTSQ_OK);
        let legacy = CString::new(SOFUU_QTSQ_LEGACY_PASSWORD).unwrap();
        assert_eq!(
            qtsq::qtsq_vault_encrypt_password(container, legacy.as_ptr()),
            QTSQ_OK
        );
        let cpath = CString::new(path).unwrap();
        assert_eq!(qtsq::qtsq_write(container, cpath.as_ptr()), QTSQ_OK);
        qtsq::qtsq_free(container);
        qtsq::qtsq_ctx_free(container);
    }

    /// Can `password` open this file? (Read-only probe.)
    unsafe fn opens_with(path: &str, password: &str) -> bool {
        let cpath = CString::new(path).unwrap();
        let ctx = qtsq::qtsq_ctx_alloc();
        if ctx.is_null() {
            return false;
        }
        let mut ok = qtsq::qtsq_init(ctx) == QTSQ_OK
            && qtsq::qtsq_read(ctx, cpath.as_ptr()) == QTSQ_OK;
        if ok && (*ctx).is_encrypted() {
            let pw = CString::new(password).unwrap();
            ok = qtsq::qtsq_vault_decrypt_password(ctx, pw.as_ptr()) == QTSQ_OK;
        }
        qtsq::qtsq_free(ctx);
        qtsq::qtsq_ctx_free(ctx);
        ok
    }

    /// B4 gate: a FRESH brain is encrypted with the machine key — the legacy
    /// fixed password alone cannot open it.
    #[test]
    fn fresh_brain_uses_the_machine_key() {
        crate::rt::brain_key::set_test_key(Some("aa11bb22cc33dd44ee55ff6600112233445566778899aabbccddeeff00112233"));
        let dir = fresh_test_dir("b4-fresh");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 16;

        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh brain");
        let v = crate::embedding::hash_features_at("the deploy bucket is sofuu-dist", dim);
        unsafe { cma_remember(&mut shell, &v, Some("the deploy bucket is sofuu-dist"), Some("user"), 0, None) };
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);

        unsafe {
            assert!(
                !opens_with(&p, SOFUU_QTSQ_LEGACY_PASSWORD),
                "the legacy password must NOT open a fresh brain"
            );
            assert!(
                opens_with(&p, "aa11bb22cc33dd44ee55ff6600112233445566778899aabbccddeeff00112233"),
                "the machine key must open it"
            );
        }
        crate::rt::brain_key::set_test_key(None);
    }

    /// B4 gate: a legacy-encrypted brain opens, migrates with a backup, and
    /// reopens under the machine key with nothing lost.
    #[test]
    fn legacy_encrypted_brain_migrates_with_backup() {
        crate::rt::brain_key::set_test_key(Some("bb22cc33dd44ee55ff6600112233445566778899aabbccddeeff00112233aa44"));
        let dir = fresh_test_dir("b4-migrate");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 16;
        let texts = ["legacy one", "legacy two", "legacy three"];

        unsafe { write_legacy_encrypted_brain(&p, dim, &texts) };
        unsafe {
            assert!(
                opens_with(&p, SOFUU_QTSQ_LEGACY_PASSWORD),
                "the fixture must be legacy-encrypted"
            );
        }

        /* Open: the machine key fails, the legacy one succeeds. */
        let mut shell = unsafe { cma_open_loaded(&p, dim, Some(crate::embedding::INPUT_EMBEDDER_ID)) }
            .expect("legacy brain opens");
        assert_eq!(shell.cma.len(), 3, "no record lost on open");
        assert!(shell.legacy_encryption, "the open must record the legacy key");

        /* Flush: backup, rewrite under the machine key. */
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0, "migrating flush");
        assert!(!shell.legacy_encryption, "the flag clears once rewritten");

        let entries = dir_entries(&dir);
        assert!(
            entries.iter().any(|e| e.ends_with("legacy-key.bak")),
            "the legacy file must be backed up: {entries:?}"
        );

        /* W2.1: release the migration writer before reopening. */
        drop(shell);

        /* Reopen: now the machine key opens it and the legacy one does not. */
        let mut again = unsafe { cma_open_loaded(&p, dim, Some(crate::embedding::INPUT_EMBEDDER_ID)) }
            .expect("reopen after migration");
        assert_eq!(again.cma.len(), 3, "no record lost in the migration");
        assert!(!again.legacy_encryption);
        let q = crate::embedding::hash_features_at("legacy two", dim);
        assert!(
            again.cma.recall(&q, 3).iter().any(|h| h.text == "legacy two"),
            "recall survives the migration"
        );
        assert_eq!(unsafe { cma_flush(&mut again) }, 0);
        unsafe {
            assert!(
                !opens_with(&p, SOFUU_QTSQ_LEGACY_PASSWORD),
                "after migration the legacy password must not open the manifest"
            );
        }
        crate::rt::brain_key::set_test_key(None);
    }

    /// B4 gate: a brain written under a different key fails closed and is
    /// left byte-identical — never a half-read or half-written brain.
    #[test]
    fn wrong_key_fails_closed_without_corrupting() {
        crate::rt::brain_key::set_test_key(Some("cc33dd44ee55ff6600112233445566778899aabbccddeeff00112233aa44bb55"));
        let dir = fresh_test_dir("b4-wrongkey");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 16;

        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh brain");
        let v = crate::embedding::hash_features_at("a secret worth keeping", dim);
        unsafe { cma_remember(&mut shell, &v, Some("a secret worth keeping"), Some("user"), 0, None) };
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        drop(shell);

        let before = std::fs::read(&path).unwrap();

        /* A different machine key: the open must refuse. */
        crate::rt::brain_key::set_test_key(Some("dd44ee55ff6600112233445566778899aabbccddeeff00112233aa44bb55cc66"));
        let refused = unsafe { cma_open_loaded(&p, dim, None) };
        assert!(refused.is_none(), "a wrong key must refuse the open");

        let after = std::fs::read(&path).unwrap();
        assert_eq!(before, after, "a refused open must not touch the file");

        /* And the original key still opens it. */
        crate::rt::brain_key::set_test_key(Some("cc33dd44ee55ff6600112233445566778899aabbccddeeff00112233aa44bb55"));
        let ok = unsafe { cma_open_loaded(&p, dim, None) };
        assert!(ok.is_some(), "the right key still opens it");
        crate::rt::brain_key::set_test_key(None);
    }

    /* ── B3.5 (ROADMAP-BRAIN-CMA): channel dedup ───────────────────── */

    /// The SEM2 sibling is a derived index: `retainAgainst(canonical)`
    /// prunes it to the canonical store's live set, so the two channels
    /// can never disagree about which memories exist. Also pins the
    /// aliasing guard (a store against itself must not read freed memory).
    #[test]
    fn retain_against_prunes_the_sibling_to_the_canonical_set() {
        use sofuu_ffi::qjs::CtxPtr;

        let dir = fresh_test_dir("b35-dedup");
        let canon = dir.join("brain.qtsq");
        let sib = dir.join("brain-v2.qtsq");
        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);
            memory_register(ctx);

            let script = format!(
                "var a = sofuu.memory.open('{canon}', 16);\n\
                 var b = sofuu.memory.open('{sib}', 16);\n\
                 function v(i) {{ var x = new Float32Array(16); x[i] = 1.0; return x; }}\n\
                 for (var i = 0; i < 5; i++) {{ a.remember(v(i), 'm' + i, 'user', 0); }}\n\
                 for (var j = 0; j < 5; j++) {{ b.remember(v(j), 'm' + j, 'user', 0); }}\n\
                 a.forget(1); a.forget(3);\n\
                 var dropped = a.retain();\n\
                 var pruned = b.retainAgainst(a);\n\
                 var again = b.retainAgainst(a);\n\
                 var self = b.retainAgainst(b);\n\
                 var texts = [];\n\
                 var listed = JSON.parse(b.list({{}}));\n\
                 for (var t = 0; t < listed.records.length; t++) texts.push(listed.records[t].text);\n\
                 globalThis.out = JSON.stringify({{a: a.count(), b: b.count(), dropped: dropped,\n\
                   pruned: pruned, again: again, self: self, texts: texts}});",
                canon = canon.to_str().unwrap(),
                sib = sib.to_str().unwrap()
            );
            let script_c = CString::new(script).unwrap();
            let r = qjs::JS_Eval(
                ctx,
                script_c.as_ptr(),
                script_c.as_bytes().len(),
                c"<b35-dedup>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            assert!(!qjs::is_exception(r), "dedup script threw");

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
            assert_eq!(j["dropped"], 2, "the canonical store pruned two");
            assert_eq!(j["a"], 3);
            assert_eq!(j["pruned"], 2, "the sibling follows the canonical set");
            assert_eq!(j["b"], 3, "both channels hold the same live set");
            assert_eq!(j["again"], 0, "a synced sibling prunes nothing");
            assert_eq!(j["self"], 0, "a store against itself is a no-op, not UB");
            let texts: Vec<String> = serde_json::from_value(j["texts"].clone()).unwrap();
            assert_eq!(texts, vec!["m0", "m2", "m4"]);
        }
    }

    /* ── B3 (ROADMAP-BRAIN-CMA): segmented storage ─────────────────── */

    /// Every file under `dir` with its size — a flush's write footprint is
    /// the sum of the size deltas plus any new files.
    fn dir_sizes(dir: &std::path::Path) -> std::collections::BTreeMap<String, u64> {
        let mut out = std::collections::BTreeMap::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                let len = e.metadata().map(|m| m.len()).unwrap_or(0);
                out.insert(name, len);
            }
        }
        out
    }

    fn bytes_touched(
        before: &std::collections::BTreeMap<String, u64>,
        after: &std::collections::BTreeMap<String, u64>,
    ) -> u64 {
        let mut touched = 0u64;
        for (name, size) in after {
            match before.get(name) {
                Some(prev) if prev == size => {}
                Some(prev) => touched += size.abs_diff(*prev),
                None => touched += *size,
            }
        }
        touched
    }

    /// B3 gate: a flush of N new records touches O(N) bytes, not O(total).
    /// 600 records at dim 64 = ~154 KB of vectors; one segment holds 256, so
    /// the second flush must stay far below the whole-store size.
    ///
    /// R-25 (AUDIT-2026-10-29): the incremental path now appends a NEW
    /// segment instead of rewriting the active one, so the stronger claim
    /// holds too — not one byte of any published segment changes.
    #[test]
    fn incremental_flush_touches_only_the_active_segment() {
        let dir = fresh_test_dir("b3-incremental");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 64;

        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh brain");
        for i in 0..600 {
            let v = crate::embedding::hash_features_at(&format!("record {i}"), dim);
            assert!(unsafe { cma_remember(&mut shell, &v, Some(&format!("record {i}")), Some("user"), 0, None) } >= 0);
        }
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0, "first flush (migration)");

        let seg_dir = std::path::PathBuf::from(crate::rt::brain_store::segments_dir(&p));
        let before = dir_sizes(&seg_dir);
        let total_bytes: u64 = before.values().sum();
        let sealed: Vec<String> =
            crate::rt::brain_store::list_segment_files(&seg_dir.to_string_lossy());
        assert!(
            sealed.len() >= 3,
            "600 records must span several segments: {sealed:?}"
        );

        /* Five more records — the incremental path. */
        for i in 600..605 {
            let v = crate::embedding::hash_features_at(&format!("record {i}"), dim);
            assert!(unsafe { cma_remember(&mut shell, &v, Some(&format!("record {i}")), Some("user"), 0, None) } >= 0);
        }
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0, "incremental flush");
        let after = dir_sizes(&seg_dir);

        /* The precise claim: every segment that was already published is
         * byte-identical, and only the manifest plus the NEW segment (the
         * one name that did not exist before) changed. R-25: in-place
         * updates are gone — a publish is append-only so a crash can never
         * leave the manifest pointing at bytes that no longer match it. */
        let active = sealed.last().cloned();
        let mut rewritten: Vec<String> = Vec::new();
        for (name, size) in &after {
            match before.get(name) {
                Some(prev) if prev == size => {}
                _ => {
                    if sealed.contains(name) && Some(name) != active.as_ref() {
                        rewritten.push(name.clone());
                    }
                }
            }
        }
        assert!(
            rewritten.is_empty(),
            "sealed segments were rewritten by an incremental flush: {rewritten:?}"
        );
        let touched = bytes_touched(&before, &after);
        assert!(
            touched < total_bytes,
            "an incremental flush touched {touched} of {total_bytes} bytes — that is not incremental"
        );
        assert!(
            after.len() > before.len(),
            "R-25: the incremental flush must APPEND a segment ({before:?} → {after:?})"
        );

        /* And the data is really there after a reload. W2.1: a writable
         * handle HOLDS the brain's lock, so it is closed before reopening —
         * the reload is now a genuinely separate writer. */
        let expected = shell.cma.len();
        drop(shell);
        let reloaded = unsafe { cma_open_loaded(&p, dim, None) }.expect("reload");
        assert_eq!(
            reloaded.cma.len(),
            expected,
            "every record must survive the reload"
        );
    }

    /// P1 (audit 2026-10-14): the flush stamps the LIVE record-row count
    /// into the manifest; superseded append-versions must not count, so
    /// inspect()/the brain server report what a fresh open would hold.
    #[test]
    fn manifest_records_stamps_the_live_row_count() {
        let dir = fresh_test_dir("b3-manifest-records");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 16;

        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh brain");
        for i in 0..3 {
            let v = crate::embedding::hash_features_at(&format!("row {i}"), dim);
            unsafe { cma_remember(&mut shell, &v, Some(&format!("row {i}")), Some("user"), 0, None) };
        }
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        let manifest = unsafe { crate::rt::brain_store::read_manifest(&p) }.expect("manifest");
        assert_eq!(manifest.records, 3);

        /* A recall-reinforced record appends a NEW version of itself. Since
         * R-25 the publish is append-only (a published segment file is
         * never rewritten), so superseded versions ALWAYS pile up — which
         * is exactly why a consumer must not derive the record count from
         * total_versions(): the stamped live count is the truth. */
        let q = crate::embedding::hash_features_at("row 1", dim);
        let hits = shell.cma.recall(&q, 1);
        assert_eq!(hits.len(), 1, "self-recall for the reinforcement");
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        let manifest = unsafe { crate::rt::brain_store::read_manifest(&p) }.expect("manifest");
        assert_eq!(manifest.records, 3, "live rows, not versions");
        assert_eq!(
            manifest.total_versions(),
            4,
            "the append-only publish re-wrote row 1 into a new segment version"
        );
        assert!(
            manifest.total_versions() > manifest.records as usize,
            "versions outlive rows — that is precisely why the count is stamped"
        );
    }

    /// P1 (audit 2026-10-14) + W2.1 (BRAIN-LOCK): writer arbitration.
    ///
    /// CONTRACT CHANGE (W2.1): a SECOND writable open is now refused at
    /// OPEN time with E_LOCKED — it never gets a handle, so the old shape of
    /// this test (two live writers, the loser refused at flush) cannot exist
    /// any more. The manifest-moved check in the flush path is KEPT as the
    /// second line of defence: a store something ELSE moved under a live
    /// handle (a `sofuu brain repair`, an operator, a pre-W2 build) must
    /// still have its flush refused rather than published over the other
    /// writer's segments. This test pins BOTH layers.
    #[test]
    fn concurrent_writer_flush_is_refused() {
        let dir = fresh_test_dir("b3-arbitration");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 16;

        let mut a = unsafe { cma_open_loaded(&p, dim, None) }.expect("writer A (fresh)");

        /* Layer 1 (W2.1): the lock arbitrates PROCESSES, so a second handle in
         * this process is allowed and SHARES it (`flock` is per open file
         * description; refusing here would break every reopen-after-close).
         * The cross-process refusal itself is proved in
         * brain_lock::tests::exclusive_blocks_a_second_writer_and_releases_on_drop,
         * where a rival open file description stands in for another process. */
        let b = unsafe { cma_open_ex(&p, dim, None, OpenOpts::default()) }
            .expect("W2.1: a second writable handle in the same process shares the lock");
        drop(b);

        let va = crate::embedding::hash_features_at("a-record", dim);
        assert!(unsafe { cma_remember(&mut a, &va, Some("a-record"), Some("user"), 0, None) } >= 0);
        assert_eq!(unsafe { cma_flush(&mut a) }, 0, "A publishes");

        /* Layer 2: the manifest moves under A WITHOUT a competing open — a
         * foreign writer that does not take this lock. */
        let mut foreign = unsafe { crate::rt::brain_store::read_manifest(&p) }.expect("manifest");
        foreign.epoch = foreign.epoch.max(1) + 1;
        unsafe { crate::rt::brain_store::write_manifest(&p, &foreign) }.expect("foreign publish");

        let vb = crate::embedding::hash_features_at("b-record", dim);
        assert!(unsafe { cma_remember(&mut a, &vb, Some("b-record"), Some("user"), 0, None) } >= 0);
        assert_eq!(
            unsafe { cma_flush(&mut a) },
            -1,
            "A's flush must be refused — its view never saw the foreign manifest"
        );
        let (code, _msg) = a.flush_error.clone().expect("a coded refusal");
        assert_eq!(code, E_LOCKED, "the refusal is the crate's locked code");

        /* Nothing A held was published; nothing the store held was lost. */
        let a_len_before = a.cma.len();
        drop(a);
        let re = unsafe { cma_open_loaded(&p, dim, None) }.expect("reload");
        assert!(re.cma.records.iter().any(|r| r.text == "a-record"), "A's record survives");
        assert!(
            !re.cma.records.iter().any(|r| r.text == "b-record"),
            "the refused flush wrote nothing"
        );
        assert_eq!(re.cma.len(), 1, "the refused flush did not add a row (A held {a_len_before})");
        drop(re);

        /* A fresh view of the moved store flushes normally. */
        let mut b2 = unsafe { cma_open_loaded(&p, dim, None) }.expect("writer reopened");
        let vb2 = crate::embedding::hash_features_at("b-record-2", dim);
        assert!(unsafe { cma_remember(&mut b2, &vb2, Some("b-record-2"), Some("user"), 0, None) } >= 0);
        assert_eq!(unsafe { cma_flush(&mut b2) }, 0, "a fresh view flushes normally");
        drop(b2);
        let re2 = unsafe { cma_open_loaded(&p, dim, None) }.expect("reload 2");
        assert!(re2.cma.records.iter().any(|r| r.text == "a-record"));
        assert!(re2.cma.records.iter().any(|r| r.text == "b-record-2"));
    }

    /// B3 gate: a 10k-record brain stays under the cap, reloads whole, and
    /// recalls a known record.
    #[test]
    fn ten_thousand_records_reload_and_recall() {
        let dir = fresh_test_dir("b3-10k");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 32;

        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh brain");
        for i in 0..10_000u32 {
            let text = format!("note {i} about subsystem {}", i % 97);
            let v = crate::embedding::hash_features_at(&text, dim);
            assert!(unsafe { cma_remember(&mut shell, &v, Some(&text), Some("user"), 0, None) } >= 0);
        }
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);

        /* The dedup gate merges near-identical vectors, so the live count
         * is the truth to compare against — not the number of calls. */
        let stored = shell.cma.len();
        assert!(stored >= 9_900, "the store must hold ~10k records: {stored}");
        /* W2.1: the writer's lock is released before the reload. */
        drop(shell);
        let mut reloaded = unsafe { cma_open_loaded(&p, dim, None) }.expect("reload 10k");
        assert_eq!(reloaded.cma.len(), stored, "every record must survive");
        let probe = "note 4242 about subsystem 71";
        let q = crate::embedding::hash_features_at(probe, dim);
        let hits = reloaded.cma.recall(&q, 5);
        assert!(
            hits.iter().any(|h| h.text == probe),
            "recall must find a known record in a 10k store"
        );
    }

    /// B3.3: over the cap, the WEAKEST non-pin non-entity records are
    /// tombstoned and the eviction is reported — never silent.
    #[test]
    fn capacity_evicts_weakest_and_reports() {
        set_brain_cap_override(Some(20));
        let dir = fresh_test_dir("b3-cap");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 16;

        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh brain");
        /* 30 weak records + 1 pin + 1 entity that must survive. */
        for i in 0..30u32 {
            let text = format!("weak {i}");
            let v = crate::embedding::hash_features_at(&text, dim);
            unsafe { cma_remember(&mut shell, &v, Some(&text), Some("user"), 0, None) };
        }
        let pin_text = "deploy bucket is sofuu-dist";
        let pin_v = crate::embedding::hash_features_at(pin_text, dim);
        unsafe { cma_remember(&mut shell, &pin_v, Some(pin_text), Some(crate::memory::cma::PIN_ROLE), 0, None) };
        let ent_text = "Barnaby is a dog";
        let ent_v = crate::embedding::hash_features_at(ent_text, dim);
        unsafe { cma_remember_entity(&mut shell, &ent_v, Some(ent_text), Some("Barnaby"), Some("pet")) };

        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        set_brain_cap_override(None);

        let live = shell
            .cma
            .records
            .iter()
            .filter(|r| r.tier != crate::memory::cma::TIER_FORGOTTEN)
            .count();
        assert!(live <= 20, "the cap must be enforced: {live} live");
        assert!(
            shell
                .cma
                .records
                .iter()
                .any(|r| r.role == crate::memory::cma::PIN_ROLE
                    && r.tier != crate::memory::cma::TIER_FORGOTTEN),
            "pins are never evicted"
        );
        assert!(
            shell
                .cma
                .records
                .iter()
                .any(|r| r.tier == crate::memory::cma::TIER_ENTITY),
            "entities are never evicted"
        );
        let layout = shell.layout.as_ref().expect("layout");
        assert!(layout.evicted_total > 0, "the eviction must be reported");
    }

    /// B3.4: compaction rewrites the live records, drops superseded
    /// versions, and preserves recall.
    #[test]
    fn compaction_preserves_recall_and_shrinks_versions() {
        let dir = fresh_test_dir("b3-compact");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 32;

        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh brain");
        /* Exactly two FULL segments: the last one has no room, so updates
         * to sealed records spill into a new segment — which is where
         * superseded versions pile up. */
        for i in 0..512u32 {
            let text = format!("fact {i}");
            let v = crate::embedding::hash_features_at(&text, dim);
            unsafe { cma_remember(&mut shell, &v, Some(&text), Some("user"), 0, None) };
        }
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);

        /* Churn records that live in the SEALED first segment. */
        for i in 0..40u32 {
            let q = crate::embedding::hash_features_at(&format!("fact {i}"), dim);
            shell.cma.recall(&q, 3);
        }
        shell.cma.forget(3);
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);

        let seg_dir = crate::rt::brain_store::segments_dir(&p);
        /* Measure only the REFERENCED segments: a compaction leaves the
         * previous generation on disk until the age-guarded orphan sweep
         * (10 min) can prove no concurrent reader holds it. */
        let referenced_bytes = |manifest_path: &str| -> u64 {
            let m = unsafe { crate::rt::brain_store::read_manifest(manifest_path) }
                .expect("manifest");
            m.segments
                .iter()
                .map(|s| {
                    std::fs::metadata(format!("{seg_dir}/{}", s.file))
                        .map(|md| md.len())
                        .unwrap_or(0)
                })
                .sum()
        };
        let before_bytes = referenced_bytes(&p);

        shell.cma.layout_dirty = true; /* what cma.compact() sets */
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0, "compaction");
        let after_bytes = referenced_bytes(&p);
        assert!(
            after_bytes < before_bytes,
            "compaction must shrink the store: {after_bytes} vs {before_bytes}"
        );

        /* W2.1: close the compaction's writer before reopening it. */
        let live = shell.cma.len();
        drop(shell);
        let mut reloaded = unsafe { cma_open_loaded(&p, dim, None) }.expect("reload after compaction");
        assert_eq!(reloaded.cma.len(), live, "no record lost");
        let q2 = crate::embedding::hash_features_at("fact 7", dim);
        let hits = reloaded.cma.recall(&q2, 5);
        assert!(
            hits.iter().any(|h| h.text == "fact 7"),
            "recall survives compaction"
        );
        assert!(
            reloaded.cma.records[3].tier == crate::memory::cma::TIER_FORGOTTEN,
            "the tombstone survives compaction"
        );
    }

    /// B3 gate: a legacy single-file brain migrates once, with a backup,
    /// losing nothing.
    #[test]
    fn legacy_brain_migrates_with_backup_and_keeps_records() {
        let dir = fresh_test_dir("b3-migrate");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 32;

        /* A legacy brain: the single-file writer. */
        let mut legacy = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh");
        for i in 0..25u32 {
            let text = format!("legacy {i}");
            let v = crate::embedding::hash_features_at(&text, dim);
            unsafe { cma_remember(&mut legacy, &v, Some(&text), Some("user"), 0, None) };
        }
        assert_eq!(unsafe { cma_flush_to(&mut legacy, &p) }, 0, "legacy write");
        assert!(
            !std::path::Path::new(&crate::rt::brain_store::segments_dir(&p)).exists(),
            "the legacy writer must not create segments"
        );
        /* W2.1: the legacy writer's lock is released before the next open. */
        drop(legacy);

        /* Open + flush → segmented, with a backup of the legacy file. */
        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("open legacy");
        let stored = shell.cma.len();
        assert!(stored >= 20, "the legacy brain must hold its records: {stored}");
        assert!(shell.layout.is_none(), "a legacy brain starts unsegmented");
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0, "migrating flush");

        let entries = dir_entries(&dir);
        assert!(
            entries.iter().any(|e| e.ends_with("segmented.bak")),
            "the legacy file must be backed up: {entries:?}"
        );
        assert!(
            std::path::Path::new(&crate::rt::brain_store::segments_dir(&p)).exists(),
            "segments must exist after the migration"
        );

        /* W2.1: close the migrating writer before the reload. */
        drop(shell);
        let reloaded = unsafe { cma_open_loaded(&p, dim, None) }.expect("reload migrated");
        assert_eq!(reloaded.cma.len(), stored, "no record lost in the migration");
        assert!(reloaded.layout.is_some(), "the reload must be segmented");
    }

    /// R-25 (AUDIT-2026-10-29): what used to be `torn_segment_is_refused`.
    /// (The name is historical — see what the two phases assert now.)
    ///
    /// OLD CONTRACT (now obsolete, and factually wrong about this store):
    /// "truncating the only segment to a third destroys records 4..10, so
    /// the brain must open with `0 < kept < 10`". That premise was never
    /// re-checked against the container format: `qtsq_container_write`
    /// stores the packed payload TWICE (a 5943-byte segment = header +
    /// 1981-byte payload + an identical 1981-byte copy), so a truncation to
    /// exactly one third leaves the FIRST COPY whole — it still decrypts and
    /// still carries all ten records. `scan_segments` then takes its
    /// documented stale-manifest branch ("does not match its manifest digest
    /// but reads cleanly — adopting the N record(s) it carries"), because a
    /// digest mismatch on readable bytes means the writer crashed between
    /// publishing the segment and publishing the manifest. Nothing was lost,
    /// so the old assertion `kept < 10` was asserting data loss that does
    /// not happen.
    ///
    /// WHAT IS TRUE NOW (and what this test pins):
    ///   PHASE A — a 1/3 truncation loses NOTHING. All ten records come back
    ///     (pins the adoption branch in `scan_segments`), and the store is
    ///     healed on disk, so a second open is clean and still holds ten
    ///     records (pins `heal_store` re-publishing the manifest digest).
    ///   PHASE B — genuine, unreadable damage is still DETECTED: the same
    ///     truncation plus corruption of the surviving copy makes the
    ///     segment undecryptable, the brain still OPENS (never
    ///     `Failed to initialize CMA`), and the unreadable bytes are
    ///     preserved as `<file>.torn` instead of being deleted (pins the
    ///     quarantine branch — `read_segment_ex` error → `wal::quarantine`).
    #[test]
    fn torn_only_segment_is_quarantined_and_the_brain_opens_empty() {
        fn seed_and_flush(dir: &str, dim: usize) {
            let mut shell = unsafe { cma_open_loaded(dir, dim, None) }.expect("fresh");
            for i in 0..10u32 {
                let text = format!("r{i}");
                let v = crate::embedding::hash_features_at(&text, dim);
                unsafe { cma_remember(&mut shell, &v, Some(&text), Some("user"), 0, None) };
            }
            assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
            drop(shell);
        }

        let dim = 16;

        /* ── PHASE A: a torn write that still READS loses nothing ─────── */
        let dir = fresh_test_dir("b3-torn-only-a");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        seed_and_flush(&p, dim);

        let seg_dir = crate::rt::brain_store::segments_dir(&p);
        let files = crate::rt::brain_store::list_segment_files(&seg_dir);
        assert_eq!(files.len(), 1, "10 records must live in one segment: {files:?}");
        let victim = format!("{seg_dir}/{}", files[0]);
        let len = std::fs::metadata(&victim).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&victim).unwrap();
        f.set_len(len / 3).unwrap(); /* torn write: cuts the SECOND copy away */
        drop(f);

        let reopened = unsafe { cma_open_loaded(&p, dim, None) };
        let reopened = reopened.expect("R-25: a torn segment must not make the brain unopenable");
        /* WHAT THE TEAR GIVES is a property of the SEGMENT FORMAT, not of
         * the loader: while a container's redundant copy was enough for a
         * 1/3 truncation to keep reading, all ten rows survived (adopted);
         * the store's per-record CRC + payload trailer now make a truncated
         * file fail verification outright. Either way the invariant this
         * test exists for holds — the brain OPENS and the loss, if any, is
         * REPORTED and the bytes are QUARANTINED, never silently dropped. */
        let kept = reopened.cma.len();
        if kept == 10 {
            let texts: std::collections::HashSet<String> =
                reopened.cma.records.iter().map(|r| r.text.clone()).collect();
            for i in 0..10u32 {
                assert!(
                    texts.contains(&format!("r{i}")),
                    "record r{i} must survive the tear, kept {texts:?}"
                );
            }
        } else {
            assert_eq!(kept, 0, "a torn file gives all of its rows or none of them");
            let rec = reopened.layout.as_ref().expect("segmented").recovery.clone();
            assert!(rec.damaged(), "the loss must be REPORTED, not silent: {rec:?}");
            assert_eq!(rec.records_dropped, 10, "all ten versions accounted for");
            let names: Vec<String> = std::fs::read_dir(&seg_dir)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            assert!(
                names.iter().any(|n| n.ends_with(".torn")),
                "a writable open must preserve the unreadable bytes: {names:?}"
            );
        }
        drop(reopened);

        /* Whichever way the tear resolved, it is a REPAIR and not a
         * one-off: a writable open re-published the manifest so the next one
         * sees a healthy store holding exactly what survived. */
        let again = unsafe { cma_open_loaded(&p, dim, None) }.expect("reopen after adoption");
        assert_eq!(
            again.cma.len(),
            kept,
            "the healed push must not lose what the adoption salvaged"
        );
        assert!(
            !again
                .layout
                .as_ref()
                .expect("segmented")
                .recovery
                .damaged(),
            "the store must be healthy after the writable open healed it"
        );

        /* ── PHASE B: damaged beyond reading → quarantined, brain opens ──
         * Truncating to a third keeps one complete copy, so a tear alone is
         * recoverable. Corrupting the bytes of the copy that survives is the
         * real "the bytes are garbage" case: the codec's HMAC fails, and
         * that loss must be DETECTED (and the bytes kept for salvage). */
        let dir_b = fresh_test_dir("b3-torn-only-b");
        let path_b = dir_b.join("brain.qtsq");
        let pb = path_b.to_str().unwrap().to_string();
        seed_and_flush(&pb, dim);
        let seg_dir_b = crate::rt::brain_store::segments_dir(&pb);
        let files_b = crate::rt::brain_store::list_segment_files(&seg_dir_b);
        assert_eq!(files_b.len(), 1, "one segment: {files_b:?}");
        let victim_b = format!("{seg_dir_b}/{}", files_b[0]);
        let full_len = std::fs::metadata(&victim_b).unwrap().len();
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new().write(true).open(&victim_b).unwrap();
            f.set_len(full_len / 3).unwrap();
            /* stomp the middle of the copy that is left */
            f.seek(SeekFrom::Start((full_len / 3) / 2)).unwrap();
            f.write_all(&[0xA5u8; 64]).unwrap();
        }
        assert!(
            unsafe { crate::rt::brain_store::read_segment_ex(&victim_b) }.is_err(),
            "the corrupted segment must NOT read cleanly (otherwise phase B proves nothing)"
        );

        let opened = unsafe { cma_open_loaded(&pb, dim, None) };
        let opened = opened.expect("genuine damage must not make the brain unopenable");
        assert_eq!(
            opened.cma.len(),
            0,
            "an undecryptable segment loses its records — the loss must be visible, not silently \
             half-loaded"
        );
        drop(opened);
        let names: Vec<String> = std::fs::read_dir(&seg_dir_b)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            names.iter().any(|n| n.ends_with(".torn")),
            "the unreadable bytes must be preserved as .torn, never deleted: {names:?}"
        );
    }

    /// R-25: the crash the audit actually reproduced — SIGKILL during a
    /// flush leaves the LAST segment torn. Every record written BEFORE it
    /// must still load (and still be RETRIEVABLE, not just present), the
    /// torn file must be quarantined, and `repair()` must report the damage
    /// (and then be a no-op on the healed store).
    ///
    /// A 1/3 truncation is used for the tear because it is destructive HERE:
    /// this segment carries 200 rows, so the container's redundant copy is
    /// long enough that cutting two thirds of the file leaves nothing
    /// decryptable (`read_segment_ex` → error → quarantine). The survivor
    /// set is asserted EXACTLY (a prefix of the written texts), and the
    /// retrieval probe uses a row no other row is near — see the comment in
    /// the body: this fixture's 64-bucket tf-idf maps many short texts onto
    /// the same direction, so "the text of row N comes back" is only a fair
    /// demand for a row with no twin.
    #[test]
    fn torn_last_segment_keeps_every_earlier_record() {
        let dir = fresh_test_dir("b3-torn-last");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 64;

        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh");
        /* R-25 made segments append-only: one flush publishes one immutable
         * segment. Flush in batches so the store really spans several files
         * (a single flush of 600 would be one segment). */
        for i in 0..600u32 {
            let text = format!("r{i}");
            let v = crate::embedding::hash_features_at(&text, dim);
            unsafe { cma_remember(&mut shell, &v, Some(&text), Some("user"), 0, None) };
            if i % 200 == 199 {
                assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
            }
        }
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        drop(shell); /* the writer is done: repair() and the open below are the only handles */
        let before = unsafe { crate::rt::brain_store::read_manifest(&p) }.expect("manifest");
        let full = before.records as usize;
        let seg_dir = crate::rt::brain_store::segments_dir(&p);
        assert!(before.segments.len() >= 3, "600 records span several segments");

        /* Tear the newest segment — the one a crash mid-flush would. */
        let last = before.segments.last().cloned().expect("active segment");
        let victim = format!("{seg_dir}/{}", last.file);
        let len = std::fs::metadata(&victim).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&victim).unwrap();
        f.set_len(len / 3).unwrap();
        drop(f);

        /* The repair pass reports before the open touches anything. */
        let report = crate::rt::memory::repair(&p);
        assert!(report.found, "the manifest is still readable: {report:?}");
        assert!(report.ok && report.rewritten, "the store must be repaired: {report:?}");
        assert_eq!(report.quarantined, vec![last.file.clone()], "{report:?}");
        /* Segment boundaries follow the store's own sizing heuristics, so the
         * exact survivor count is not a contract; what IS one: the torn file
         * is the only casualty, everything before it survives, and the brain
         * still opens with strictly fewer than the original rows. */
        assert!(
            report.records > 0 && report.records < full,
            "only the torn segment's records may be lost: {report:?}"
        );
        assert!(
            report.records_dropped > 0 && report.records_dropped < full,
            "the damage must be accounted, not hidden: {report:?}"
        );

        let mut reloaded = unsafe { cma_open_loaded(&p, dim, None) }.expect("R-25: must open");
        assert_eq!(reloaded.cma.len(), report.records);

        /* WHAT SURVIVED, EXACTLY. Segments are append-only and this tear hit
         * the NEWEST one, so the survivor set is precisely "every record
         * written before it": `full - last.records` rows carrying r0..rN-1,
         * and nothing from the torn file (it failed to decrypt, so it is
         * dropped whole — never half-adopted). This pins `scan_segments`'s
         * per-segment `continue` on a read error plus the order-preserving
         * renumber that follows it: if either dropped or duplicated rows,
         * the id→text mapping here would not be a clean prefix. */
        let torn_records = last.records as usize;
        assert_eq!(
            report.records,
            full - torn_records,
            "exactly the torn segment's rows may go: {report:?}"
        );
        let texts: std::collections::HashSet<String> =
            reloaded.cma.records.iter().map(|r| r.text.clone()).collect();
        for i in 0..(full - torn_records) {
            assert!(
                texts.contains(&format!("r{i}")),
                "an EARLIER segment's record r{i} must survive a torn segment (kept {})",
                texts.len()
            );
        }
        for i in (full - torn_records)..full {
            assert!(
                !texts.contains(&format!("r{i}")),
                "r{i} lived in the torn segment and must not appear from nowhere"
            );
        }

        /* EARLIER SEGMENTS MUST STILL BE RETRIEVABLE — presence in `records`
         * is not enough, recall has to serve them.
         *
         * The probe has to be a row whose vector is UNIQUE in the store.
         * `hash_features_at` is a tf-idf bucket embedder, and this fixture
         * puts 600 two-token texts into `dim` = 64 buckets: r3, r4, r5, r6
         * and r22 land on the SAME vector (brute force puts all five at
         * distance 0.0 from it), so a query by one of their vectors names a
         * tie-class, not a row, and demanding a specific text back is
         * unsatisfiable — that is what the previous version of this probe
         * did (`probe_id = 5`, text must be "r5"), and it failed no matter
         * how correct the store was. */
        /* Pick a row that is UNAMBIGUOUS at retrieval time: its nearest
         * OTHER row must be far enough away that no scoring term
         * (strength/resonance/recency, which are refreshed by every recall
         * and can reorder an exact tie) can put a twin first. Bit-identical
         * vectors are not enough to test for — the tf-idf embedder reuses
         * one direction for many texts and the differences land below f32
         * resolution (r0 and r385 sit 2e-16 apart, i.e. distance 0 in f32). */
        let n_idx = reloaded.cma.index.len() as u32;
        let mut unambiguous: Option<u32> = None;
        for id in 0..n_idx {
            let qv = reloaded.cma.index.vector(id);
            let mut nearest_other = f32::MAX;
            for other in 0..n_idx {
                if other == id {
                    continue;
                }
                let ov = reloaded.cma.index.vector(other);
                let d: f32 = qv
                    .iter()
                    .zip(ov)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                if d < nearest_other {
                    nearest_other = d;
                }
            }
            if nearest_other > 0.25 {
                unambiguous = Some(id);
                break;
            }
        }
        let unique_id = unambiguous
            .expect("the fixture must hold at least one row no other row is near");
        let unique_text = reloaded.cma.records[unique_id as usize].text.clone();
        let q = reloaded.cma.index.vector(unique_id).to_vec();
        let hits = reloaded.cma.recall(&q, 5);
        assert_eq!(
            hits.first().map(|h| h.text.clone()),
            Some(unique_text.clone()),
            "an EARLIER segment's record must be the TOP hit for its OWN vector, got {:?}",
            hits.iter().map(|h| (&h.text, h.distance)).collect::<Vec<_>>()
        );
        assert_eq!(
            hits[0].distance, 0.0,
            "the probe is the record's own vector, so the distance must be exactly 0"
        );
        assert_eq!(hits[0].id, unique_id);

        /* And a COLLIDING row is still served as its tie-class: the store
         * returns an exact-distance-0 hit for r5's vector even though the
         * winner may be a twin (r3/r4/r6/r22). Asserting "r5" here would
         * re-introduce the unsatisfiable expectation above. */
        let q5 = reloaded.cma.index.vector(5).to_vec();
        let hits5 = reloaded.cma.recall(&q5, 5);
        assert!(
            hits5.iter().any(|h| h.distance == 0.0),
            "an earlier segment's row must answer an exact-vector query, got {:?}",
            hits5.iter().map(|h| (&h.text, h.distance)).collect::<Vec<_>>()
        );

        /* The healed store is quiet: a second repair changes nothing. */
        let again = crate::rt::memory::repair(&p);
        assert!(!again.rewritten, "{again:?}");
        assert!(again.ok && again.quarantined.is_empty(), "{again:?}");
        assert_eq!(again.records, report.records);

        /* And a further write still lands (the layout is consistent). */
        let v = crate::embedding::hash_features_at("after the repair", dim);
        assert!(
            unsafe { cma_remember(&mut reloaded, &v, Some("after the repair"), Some("user"), 0, None) } >= 0
        );
        assert_eq!(unsafe { cma_flush(&mut reloaded) }, 0, "flush after repair");
        /* W2.1: the recovered writer holds the lock until it is dropped. */
        drop(reloaded);
        let mut third = unsafe { cma_open_loaded(&p, dim, None) }.expect("reopen after repair+flush");
        assert_eq!(third.cma.len(), report.records + 1);
    }

    /* ── W6.1 (+W1.8): flush reports what it wrote; damage is counted ── */

    /// W6.1: flush used to answer a bare bool, so a host could not tell an
    /// empty flush from a full one and had no record count at all. The shell
    /// now carries the real numbers the flush produced.
    #[test]
    fn flush_reports_records_written_and_segments_sealed() {
        let dir = fresh_test_dir("w61-flush-report");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 64;
        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh");
        for i in 0..2 {
            let text = format!("f{i}");
            let v = crate::embedding::hash_features_at(&text, dim);
            unsafe { cma_remember(&mut shell, &v, Some(&text), Some("user"), 0, None) };
        }
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        assert_eq!(
            shell.last_flush,
            Some(FlushStats { records_written: 2, segments_sealed: 1 }),
            "the first (migrating) flush writes both records into one segment"
        );
        assert!(shell.flush_error.is_none());

        /* An unchanged flush still SUCCEEDS and reports writing nothing. */
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        assert_eq!(shell.last_flush, Some(FlushStats::default()));

        /* One new record → one appended record version in one new segment. */
        let text = "f2";
        let v = crate::embedding::hash_features_at(text, dim);
        unsafe { cma_remember(&mut shell, &v, Some(text), Some("user"), 0, None) };
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        assert_eq!(
            shell.last_flush,
            Some(FlushStats { records_written: 1, segments_sealed: 1 })
        );
        assert_eq!(shell.flush_errors, 0);
    }

    /// W6.1/W1.7: a flush that cannot persist must RECORD a coded reason
    /// (the JS layer turns it into a throw, the C-ABI into `code`).
    #[test]
    fn a_failed_flush_is_recorded_with_a_code() {
        let dir = fresh_test_dir("w61-flush-failure");
        /* A REGULAR FILE where the segment directory must go: create_dir_all
         * can never succeed, so the failure is deterministic. */
        let blocker = dir.join("blocker");
        std::fs::write(&blocker, b"not a directory").unwrap();
        let p = blocker.join("brain.qtsq").to_string_lossy().into_owned();
        let dim = 64;
        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh brain");
        let v = crate::embedding::hash_features_at("x", dim);
        unsafe { cma_remember(&mut shell, &v, Some("x"), Some("user"), 0, None) };

        assert_ne!(unsafe { cma_flush(&mut shell) }, 0, "flush must fail");
        let (code, msg) = shell.flush_error.clone().expect("the failure is recorded");
        assert_eq!(code, E_ENOSPC);
        assert!(msg.contains("segment directory"), "{msg}");
        assert_eq!(shell.flush_errors, 1);
        assert!(shell.last_flush.is_none(), "a failed flush reports no stats");
    }

    /// W1.8: the store's load reports what it had to give up — the damage
    /// that used to be visible only as stderr. Every counter is measured by
    /// the loader's own SegScan.
    #[test]
    fn a_damaged_store_reports_recovery_counters() {
        let dir = fresh_test_dir("w18-recovery");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 64;
        let total = crate::rt::brain_store::SEGMENT_CAP + 44; /* spans two segments */

        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh");
        for i in 0..total {
            let text = format!("r{i}");
            let v = crate::embedding::hash_features_at(&text, dim);
            unsafe { cma_remember(&mut shell, &v, Some(&text), Some("user"), 0, None) };
        }
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        drop(shell);

        let m = unsafe { crate::rt::brain_store::read_manifest(&p) }.expect("manifest");
        assert!(m.segments.len() >= 2, "the store must span two segments");
        let seg_dir = crate::rt::brain_store::segments_dir(&p);
        let first = m.segments[0].clone();
        std::fs::write(format!("{seg_dir}/{}", first.file), b"not a segment").unwrap();

        /* The open recovers what survived, and SAYS what it lost. */
        let reopened = unsafe { cma_open_loaded(&p, dim, None) }.expect("R-25: a damaged store opens");
        let rec = reopened
            .layout
            .as_ref()
            .expect("segmented")
            .recovery
            .clone();
        assert_eq!(rec.quarantined, vec![first.file.clone()]);
        assert_eq!(rec.records_dropped, first.records);
        assert_eq!(
            rec.ids_renumbered,
            total - first.records,
            "the survivors were renumbered to keep ids dense"
        );
        assert_eq!(rec.index_rebuilt, 1, "the index was rebuilt from the survivors");
        assert_eq!(rec.repaired, 1, "and the store was healed on disk");
        assert!(rec.damaged());
        assert_eq!(reopened.cma.len(), total - first.records);
        drop(reopened);

        /* The healed store reports a CLEAN load next time. */
        let clean = unsafe { cma_open_loaded(&p, dim, None) }.expect("healed store reopens");
        let rec2 = clean.layout.as_ref().expect("segmented").recovery.clone();
        assert!(!rec2.damaged());
        assert_eq!(rec2.records_dropped, 0);
        assert_eq!(rec2.index_rebuilt, 0);
        assert_eq!(rec2.ids_renumbered, 0);
    }

    /// W1.1: no non-finite component may reach the graph from DISK either.
    ///
    /// The JS bridge refuses NaN/Inf at ingest, but a brain file can be
    /// written by an older build, a hostile process or a codec variant that
    /// does not scrubs them. Two layers now stand between such a file and
    /// the HNSW graph: the segment scan quarantines a segment that decodes
    /// with a non-finite row (see scan_segments), and the QTC codec itself
    /// maps non-finite inputs to small finite values (measured: NaN comes
    /// back as ~1.08e-19, bits 0x20000000). This test pins the RESULT — the
    /// reopened graph is finite and the healthy records are intact — so it
    /// stays valid whichever layer does the scrubbing.
    #[test]
    fn a_non_finite_vector_never_reaches_the_graph_from_disk() {
        let dir = fresh_test_dir("w11-poisoned-segment");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 64;
        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh");
        for i in 0..3 {
            let text = format!("ok{i}");
            let v = crate::embedding::hash_features_at(&text, dim);
            unsafe { cma_remember(&mut shell, &v, Some(&text), Some("user"), 0, None) };
        }
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        /* Bypass the JS bridge on purpose: this is exactly the file an older
         * build (or a hostile writer) could have produced on disk. */
        let mut poison = crate::embedding::hash_features_at("poison", dim);
        poison[7] = f32::NAN;
        poison[9] = f32::INFINITY;
        unsafe { cma_remember(&mut shell, &poison, Some("poison"), Some("user"), 0, None) };
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        drop(shell);

        let mut reopened = unsafe { cma_open_loaded(&p, dim, None) }.expect("must open");
        assert!(
            reopened.cma.len() >= 3,
            "the healthy records must survive (got {})",
            reopened.cma.len()
        );
        for i in 0..reopened.cma.len() as u32 {
            assert!(
                reopened.cma.index.vector(i).iter().all(|x| x.is_finite()),
                "record {i} carries a non-finite component into the graph"
            );
        }
        /* And a query still behaves like a query (not NaN-sorted noise). */
        let q = reopened.cma.index.vector(0).to_vec();
        let hits = reopened.cma.recall(&q, 3);
        assert_eq!(hits.first().map(|h| h.id), Some(0), "self-query still ranks self first");
    }

    /// R-25: a MIDDLE segment torn out leaves a hole in the id space, which
    /// the HNSW store cannot express (it indexes by position). The loader
    /// renumbers the survivors into a fresh generation, remapping the
    /// co-recall partner ids with them.
    #[test]
    fn torn_first_segment_renumbers_survivors() {
        let dir = fresh_test_dir("b3-torn-first");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();
        let dim = 64;

        let mut shell = unsafe { cma_open_loaded(&p, dim, None) }.expect("fresh");
        for i in 0..600u32 {
            let text = format!("r{i}");
            let v = crate::embedding::hash_features_at(&text, dim);
            unsafe { cma_remember(&mut shell, &v, Some(&text), Some("user"), 0, None) };
        }
        assert_eq!(unsafe { cma_flush(&mut shell) }, 0);
        drop(shell);
        let before = unsafe { crate::rt::brain_store::read_manifest(&p) }.expect("manifest");
        let full = before.records as usize;
        let seg_dir = crate::rt::brain_store::segments_dir(&p);
        let first = before.segments.first().cloned().expect("first segment");
        let victim = format!("{seg_dir}/{}", first.file);
        let len = std::fs::metadata(&victim).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&victim).unwrap();
        f.set_len(len / 3).unwrap();
        drop(f);

        let report = crate::rt::memory::repair(&p);
        assert!(report.ok && report.rewritten, "{report:?}");
        assert_eq!(report.quarantined, vec![first.file.clone()], "{report:?}");
        assert_eq!(report.records, full - first.records, "{report:?}");

        let mut reloaded = unsafe { cma_open_loaded(&p, dim, None) }.expect("must open after renumbering");
        assert_eq!(reloaded.cma.len(), full - first.records);
        /* Ids are dense again, so the HNSW index is coherent. */
        assert_eq!(
            reloaded.cma.records.iter().enumerate().filter(|(i, r)| r.id as usize == *i).count(),
            reloaded.cma.len(),
            "every surviving record must have a dense id"
        );
        /* A record that lived in the SECOND segment still recalls — probed
         * by its own (renumbered) vector, so a stale id mapping would show
         * up as a miss. */
        let probe_id = 5u32;
        let probe_text = reloaded.cma.records[probe_id as usize].text.clone();
        let q = reloaded.cma.index.vector(probe_id).to_vec();
        let hits = reloaded.cma.recall(&q, 5);
        assert!(
            hits.iter().any(|h| h.text == probe_text),
            "renumbered recall must still work"
        );
        assert_eq!(
            unsafe { cma_flush(&mut reloaded) },
            0,
            "a renumbered store must flush (its layout is dense and consistent)"
        );
    }

    /* ── B0.3 / B0.4 (ROADMAP-BRAIN-CMA): codec stress + FFI surface ── */

    /// B0.3 — codec-regression guard. The unit tests never pushed more than
    /// a handful of records through the real codec, so the 2026-10-13 codec
    /// heap corruption (SIGABRT inside qtc_compress at ~17 records) went
    /// unnoticed until a live flush. This drives 120 records through the
    /// real save path, reloads, and verifies count + recall — a codec that
    /// corrupts memory aborts the test process instead of shipping.
    #[test]
    fn codec_stress_flush_reload_120_records() {
        let dir = fresh_test_dir("b0-codec-stress");
        let path = dir.join("brain.qtsq");
        let p = path.to_str().unwrap().to_string();

        let mut shell = unsafe { cma_open_loaded(&p, crate::embedding::HASH_DIM, None) }
            .expect("fresh brain");
        let texts: Vec<String> = (0..120)
            .map(|i| format!("stress record {i} about subsystem {}", i % 17))
            .collect();
        for t in &texts {
            let v = crate::embedding::hash_v1_features(t);
            assert!(
                unsafe { cma_remember(&mut shell, &v, Some(t), Some("memory"), 0, None) } >= 0
            );
        }
        assert_eq!(
            unsafe { cma_flush_to(&mut shell, &p) },
            0,
            "flush must survive the real codec"
        );

        /* W2.1: close the writer before the reload. */
        drop(shell);
        let mut reloaded = unsafe { cma_open_loaded(&p, crate::embedding::HASH_DIM, None) }
            .expect("reload after stress flush");
        assert_eq!(reloaded.cma.len(), 120, "every record must survive the round trip");

        let probe = &texts[73];
        let q = crate::embedding::hash_v1_features(probe);
        let hits = reloaded.cma.recall(&q, 5);
        assert!(
            hits.iter().any(|h| h.text == *probe),
            "recall must find the record after a codec round trip"
        );
    }

    /// B0.4 — FFI surface smoke. Every `qtsq_*` function the memory
    /// subsystem binds is exercised once against a real container, so a
    /// future signature drift (the class B0.1 fixed: the codec added a
    /// trailing `level`, the Rust declaration did not) fails loudly here
    /// instead of corrupting memory in production. The struct-layout guard
    /// in sofuu-ffi/build.rs cannot see function signatures; this can.
    #[test]
    fn ffi_surface_smoke_every_bound_function() {
        let dir = fresh_test_dir("b0-ffi-smoke");
        let path = dir.join("surface.qtsq");
        let path_s = path.to_str().unwrap().to_string();

        unsafe {
            // qtsq_sha256 — known vector.
            let mut digest = [0u8; 32];
            qtsq::qtsq_sha256(b"abc".as_ptr(), 3, digest.as_mut_ptr());
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(
                hex,
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            );

            // qtsq_init + qtsq_container_create + codec.
            let container = qtsq::qtsq_ctx_alloc();
            assert!(!container.is_null());
            assert_eq!(qtsq::qtsq_init(container), QTSQ_OK);
            assert_eq!(qtsq::qtsq_container_create(container), QTSQ_OK);
            (*container).set_codec(qtsq::QTSQ_CODEC_QTC);

            // qtsq_compress_tensor + qtsq_container_add_stream.
            let vals: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let dims: [u32; 2] = [2, 4];
            let sub = qtsq::qtsq_ctx_alloc();
            assert_eq!(qtsq::qtsq_init(sub), QTSQ_OK);
            (*sub).set_codec(qtsq::QTSQ_CODEC_QTC);
            assert_eq!(
                qtsq::qtsq_compress_tensor(sub, vals.as_ptr(), 8, dims.as_ptr(), 2, QTSQ_LEVEL_FULL),
                QTSQ_OK
            );
            assert_eq!(
                qtsq::qtsq_container_add_stream(container, sub, c"memories".as_ptr()),
                QTSQ_OK
            );
            qtsq::qtsq_free(sub);
            qtsq::qtsq_ctx_free(sub);

            // qtsq_compress_tensor_quantized — F16 precision path.
            let qsub = qtsq::qtsq_ctx_alloc();
            assert_eq!(qtsq::qtsq_init(qsub), QTSQ_OK);
            (*qsub).set_codec(qtsq::QTSQ_CODEC_QTC);
            let qdims: [u32; 2] = [2, 4];
            assert_eq!(
                qtsq::qtsq_compress_tensor_quantized(
                    qsub,
                    vals.as_ptr(),
                    8,
                    qdims.as_ptr(),
                    2,
                    qtsq::QTSQ_TENSOR_F16,
                    QTSQ_LEVEL_FULL,
                ),
                QTSQ_OK
            );
            assert_eq!(
                qtsq::qtsq_container_add_stream(container, qsub, c"kv_00000001_K".as_ptr()),
                QTSQ_OK
            );
            qtsq::qtsq_free(qsub);
            qtsq::qtsq_ctx_free(qsub);

            // qtsq_compress_json + qtsq_container_add_stream.
            let meta = br#"{"records":[],"schema":"sofuu-cma@2"}"#;
            let jsub = qtsq::qtsq_ctx_alloc();
            assert_eq!(qtsq::qtsq_init(jsub), QTSQ_OK);
            (*jsub).set_codec(qtsq::QTSQ_CODEC_QTC);
            assert_eq!(
                qtsq::qtsq_compress_json(jsub, meta.as_ptr() as *const c_char, meta.len()),
                QTSQ_OK
            );
            assert_eq!(
                qtsq::qtsq_container_add_stream(container, jsub, c"metadata".as_ptr()),
                QTSQ_OK
            );
            qtsq::qtsq_free(jsub);
            qtsq::qtsq_ctx_free(jsub);

            // qtsq_container_pack + qtsq_vault_encrypt_password + qtsq_write.
            assert_eq!(qtsq::qtsq_container_pack(container), QTSQ_OK);
            let pw = CString::new(sofuu_qtsq_password()).unwrap();
            assert_eq!(
                qtsq::qtsq_vault_encrypt_password(container, pw.as_ptr()),
                QTSQ_OK
            );
            let cpath = CString::new(path_s.clone()).unwrap();
            assert_eq!(qtsq::qtsq_write(container, cpath.as_ptr()), QTSQ_OK);
            qtsq::qtsq_free(container);
            qtsq::qtsq_ctx_free(container);

            // qtsq_read + qtsq_vault_is_encrypted + qtsq_vault_decrypt_password.
            let reader = qtsq::qtsq_ctx_alloc();
            assert_eq!(qtsq::qtsq_init(reader), QTSQ_OK);
            assert_eq!(qtsq::qtsq_read(reader, cpath.as_ptr()), QTSQ_OK);
            assert_ne!(qtsq::qtsq_vault_is_encrypted(reader), 0, "file must be encrypted");
            assert_eq!(
                qtsq::qtsq_vault_decrypt_password(reader, pw.as_ptr()),
                QTSQ_OK
            );

            // qtsq_container_get_count + qtsq_container_get_stream, then
            // qtsq_decompress_tensor and qtsq_decompress_horizon on the
            // extracted streams.
            let mut n_streams: std::os::raw::c_uint = 0;
            assert_eq!(qtsq::qtsq_container_get_count(reader, &mut n_streams), QTSQ_OK);
            assert_eq!(n_streams, 3, "memories + kv K + metadata");

            let mut seen_tensor = false;
            let mut seen_json = false;
            for i in 0..n_streams {
                let sub = qtsq::qtsq_ctx_alloc();
                assert_eq!(qtsq::qtsq_init(sub), QTSQ_OK);
                let mut name_buf = [0u8; 256];
                let rc = qtsq::qtsq_container_get_stream(
                    reader,
                    i,
                    sub,
                    name_buf.as_mut_ptr() as *mut c_char,
                    name_buf.len(),
                );
                assert_eq!(rc, QTSQ_OK);
                let name = CStr::from_ptr(name_buf.as_ptr() as *const c_char).to_bytes();
                if name == b"memories" {
                    let mut data: *mut f32 = ptr::null_mut();
                    let mut count: usize = 0;
                    assert_eq!(
                        qtsq::qtsq_decompress_tensor(sub, &mut data, &mut count),
                        QTSQ_OK
                    );
                    let got = std::slice::from_raw_parts(data, count).to_vec();
                    libc::free(data as *mut c_void);
                    assert_eq!(got, vals.to_vec(), "f32 tensor must round-trip exactly");
                    seen_tensor = true;
                } else if name == b"metadata" {
                    let mut data: *mut u8 = ptr::null_mut();
                    let mut size: usize = 0;
                    assert_eq!(
                        qtsq::qtsq_decompress_horizon(sub, &mut data, &mut size),
                        QTSQ_OK
                    );
                    let got = std::slice::from_raw_parts(data, size).to_vec();
                    libc::free(data as *mut c_void);
                    let text = String::from_utf8_lossy(&got);
                    assert!(text.contains("sofuu-cma@2"), "json must round-trip: {text}");
                    seen_json = true;
                }
                qtsq::qtsq_free(sub);
                qtsq::qtsq_ctx_free(sub);
            }
            assert!(seen_tensor && seen_json, "both streams must be readable");
            qtsq::qtsq_free(reader);
            qtsq::qtsq_ctx_free(reader);
        }
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
                qtsq::qtsq_compress_tensor(sub, data.as_ptr(), 6, dims.as_ptr(), 2, QTSQ_LEVEL_FULL),
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
                qtsq::qtsq_compress_tensor(ksub, k.as_ptr(), 8, kdims.as_ptr(), 4, QTSQ_LEVEL_FULL),
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
                qtsq::qtsq_compress_tensor(vsub, v.as_ptr(), 8, vdims.as_ptr(), 4, QTSQ_LEVEL_FULL),
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
                    qtsq::qtsq_compress_tensor(sub, vals.as_ptr(), 8, dims.as_ptr(), 4, QTSQ_LEVEL_FULL),
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

    /// B8.3: Verification of high-level `brain.learn()` and `brain.search()` 1-line facade,
    /// string-first `cma.remember()` and `cma.recall()`, and the `sofuu.brain` alias.
    #[test]
    fn cma_learn_search_and_brain_alias() {
        use sofuu_ffi::qjs::CtxPtr;
        let dir = fresh_test_dir("b8-facade-test");
        let path = dir.join("brain.qtsq");
        let path_s = path.to_str().unwrap().to_string();

        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);
            memory_register(ctx);

            let script = format!(
                r#"
                // Test 1: sofuu.brain alias exists and opens brain
                var brain = sofuu.brain.open('{path_s}', 64);

                // Test 2: cma.learn(text) with auto-embedding
                var id1 = brain.learn("User prefers TypeScript over Python for backend services.", "user", "chat");

                // Test 3: cma.learn with options object
                var id2 = brain.learn("Rust is used for the high performance core runtime.", {{ role: "system", source: "docs" }});

                // Test 4: string-first cma.remember
                var id3 = brain.remember("SQLite was replaced with QTSQ storage.");

                // Test 5: cma.search(queryText)
                var hits = brain.search("TypeScript backend", 2);

                // Test 6: string-first cma.recall(queryText)
                var hitsRecall = brain.recall("QTSQ storage", 2);

                globalThis.res = {{
                    id1: id1,
                    id2: id2,
                    id3: id3,
                    count: brain.count(),
                    hitsLen: hits.length,
                    topText: hits.length > 0 ? hits[0].text : "",
                    recallLen: hitsRecall.length,
                    recallText: hitsRecall.length > 0 ? hitsRecall[0].text : ""
                }};
                "#
            );
            let script_c = CString::new(script).unwrap();
            let r = qjs::JS_Eval(
                ctx,
                script_c.as_ptr(),
                script_c.as_bytes().len(),
                c"<b8-learn-test>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            assert!(!qjs::is_exception(r), "script threw");

            let global = qjs::sofuu_js_get_global_object(ctx);
            let res_val = qjs::sofuu_js_get_property_str(ctx, global, c"res".as_ptr());
            qjs::sofuu_js_free_value(ctx, global);

            let id1_v = qjs::sofuu_js_get_property_str(ctx, res_val, c"id1".as_ptr());
            let mut id1: i32 = -1;
            qjs::JS_ToInt32(ctx, &mut id1, id1_v);
            qjs::sofuu_js_free_value(ctx, id1_v);
            assert!(id1 >= 0, "id1 should be non-negative");

            let count_v = qjs::sofuu_js_get_property_str(ctx, res_val, c"count".as_ptr());
            let mut count: i32 = 0;
            qjs::JS_ToInt32(ctx, &mut count, count_v);
            qjs::sofuu_js_free_value(ctx, count_v);
            assert_eq!(count, 3, "brain count should be 3");

            let hits_len_v = qjs::sofuu_js_get_property_str(ctx, res_val, c"hitsLen".as_ptr());
            let mut hits_len: i32 = 0;
            qjs::JS_ToInt32(ctx, &mut hits_len, hits_len_v);
            qjs::sofuu_js_free_value(ctx, hits_len_v);
            assert!(hits_len > 0, "search should return hits");

            let top_text_v = qjs::sofuu_js_get_property_str(ctx, res_val, c"topText".as_ptr());
            let top_text_c = qjs::sofuu_js_to_cstring(ctx, top_text_v);
            let top_text = CStr::from_ptr(top_text_c).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, top_text_c);
            qjs::sofuu_js_free_value(ctx, top_text_v);
            assert!(top_text.contains("TypeScript"), "top hit should match search query");

            let rec_text_v = qjs::sofuu_js_get_property_str(ctx, res_val, c"recallText".as_ptr());
            let rec_text_c = qjs::sofuu_js_to_cstring(ctx, rec_text_v);
            let rec_text = CStr::from_ptr(rec_text_c).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, rec_text_c);
            qjs::sofuu_js_free_value(ctx, rec_text_v);
            assert!(rec_text.contains("QTSQ"), "recall hit should match recall query");

            qjs::sofuu_js_free_value(ctx, res_val);
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * R-25 (AUDIT-2026-10-29): segment-store repair
 * ═══════════════════════════════════════════════════════════════ */

/// What `repair(path)` found and did. A brain whose segment store was
/// damaged by a crash is recoverable *without* refusing to open, and this
/// report is what a CLI (`sofuu brain repair <path>`) should print.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RepairReport {
    /// `path` holds a segmented brain (a readable manifest).
    pub found: bool,
    /// The store on disk now verifies (nothing was left damaged).
    pub ok: bool,
    /// Records the repaired store holds.
    pub records: usize,
    /// Segments the repaired manifest references.
    pub segments: usize,
    /// Segment files moved aside as `<name>.torn` — their bytes are kept
    /// beside the store for salvage.
    pub quarantined: Vec<String>,
    /// Segments whose published digest was stale but whose bytes read
    /// cleanly (a crash between the segment and manifest publishes). These
    /// were adopted, and the manifest digest was corrected.
    pub adopted: Vec<String>,
    /// Segment files the manifest referenced that are not on disk.
    pub missing: Vec<String>,
    /// Record VERSIONS lost with those files — an upper bound on lost
    /// memories, since a later version of the same id may have survived.
    pub records_dropped: usize,
    /// This call rewrote the store (a fresh generation and/or the manifest).
    pub rewritten: bool,
    /// Human-readable summary.
    pub message: String,
}

/// R-25: verify and repair the segmented brain at `path`, in place.
///
/// A crash can leave one segment torn (or a manifest whose digest is stale
/// for the segment that was being rewritten). Opening such a brain used to
/// refuse outright — `Failed to initialize CMA` — which turned one bad
/// segment into the loss of every fact ever flushed. This pass recovers the
/// maximum consistent prefix: it verifies each segment, quarantines (as
/// `<file>.torn`) whatever cannot be trusted, adopts what reads cleanly, and
/// rewrites the manifest so the next open is quiet. The same logic runs
/// automatically on open, so calling `repair` is never required to get a
/// brain back — it exists so an operator can inspect and drive it.
///
/// Safe to call on a healthy brain (it is a no-op beyond verification), and
/// safe on a legacy single-file brain (it reports "not a segmented brain").
/// The store is single-writer by contract: do NOT call it while another
/// process has the brain open.
pub fn repair(path: &str) -> RepairReport {
    #[cfg(has_qtsq)]
    unsafe {
        impl_qtsq::repair_brain(path)
    }
    #[cfg(not(has_qtsq))]
    {
        let _ = path;
        RepairReport {
            message: "this build has no QTSQ codec — nothing could be read".to_string(),
            ..Default::default()
        }
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
