// rt/npm.rs — npm resolution + package installer (PLAN-RUST-MIGRATION M7).
//
// Port of the deleted `src/npm/resolver.c` (708 lines), semantics verbatim:
//   npm_resolve  — Node-style node_modules walk-up (the Rust twin in
//                  sofuu_core::npm already served the loader since D3; the
//                  C symbol is now provided here for the remaining callers).
//   npm_install / npm_install_local_package_json — registry fetch via
//                  synchronous curl_easy_perform, mkstemp'd tgz, SHA-1
//                  verification, the Rust safe tarball extractor, and
//                  transitive dependency recursion. The C code read
//                  package.json dependency lists with throwaway QuickJS
//                  contexts; the port uses serde_json (identical behavior).
//
// C symbols replaced: `npm_resolve`, `npm_install`,
// `npm_install_local_package_json` (sofuu-ffi's SofuuRuntime + main.rs
// call them unchanged).

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;
extern "C" {
    // Rust twins (ffi_exports.rs) the installer delegates to, exactly like
    // the C code did under SOFUU_RUST_CORE.
    fn sofuu_npm_sha1_file_rs(path: *const c_char, out_hex: *mut c_char) -> c_int;
    fn sofuu_npm_extract_safe_rs(
        tgz_path: *const c_char,
        dest_dir: *const c_char,
        strip_components: usize,
        err: *mut c_char,
        err_cap: usize,
    ) -> c_int;
}

#[inline]
unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        None
    } else {
        CStr::from_ptr(p).to_str().ok()
    }
}

/// CURLOPT_WRITEFUNCTION for the installer: writes the received chunk into
/// the std::fs::File supplied as CURLOPT_WRITEDATA (a short count aborts
/// the transfer). Replaces the FILE*-based default callback — mkstemp/
/// fdopen have no portable std replacement.
unsafe extern "C" fn tgz_write_cb(
    ptr: *mut c_char,
    size: usize,
    nmemb: usize,
    userdata: *mut c_void,
) -> usize {
    let file = &mut *(userdata as *mut std::fs::File);
    let len = size.saturating_mul(nmemb);
    let chunk = std::slice::from_raw_parts(ptr as *const u8, len);
    match std::io::Write::write_all(file, chunk) {
        Ok(()) => len,
        Err(_) => 0,
    }
}

/// The resolver C symbol (declared in the retired resolver.h) — Rust twin
/// of the C walk-up. malloc'd absolute path or NULL.
///
/// # Safety
/// C-string contract: start_dir/module_name are NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn npm_resolve(start_dir: *const c_char, module_name: *const c_char) -> *mut c_char {
    let (Some(dir), Some(name)) = (cstr(start_dir), cstr(module_name)) else {
        return ptr::null_mut();
    };
    match crate::npm::npm_resolve(Path::new(dir), name) {
        Some(p) => {
            let s = p.to_string_lossy();
            let c = CString::new(s.as_bytes()).unwrap_or_default();
            let out = libc::malloc(c.as_bytes().len() + 1) as *mut c_char;
            if out.is_null() {
                return ptr::null_mut();
            }
            std::ptr::copy_nonoverlapping(c.as_ptr(), out, c.as_bytes().len() + 1);
            out
        }
        None => ptr::null_mut(),
    }
}

// ── Package install ──────────────────────────────────────────────────

/// Reject names/versions that could escape the install dir (mirrors the C
/// npm_spec_is_safe; the Rust twin in crate::npm is the D3 authority).
fn npm_spec_is_safe(s: &str) -> bool {
    crate::npm::spec_is_safe(s)
}

/// Cap on the buffered registry metadata: a misbehaving registry must not
/// grow memory without bound (the tarball download is file-backed).
const NPM_META_CAP: usize = 64 * 1024 * 1024;

struct CurlBuf {
    buf: Vec<u8>,
}

unsafe extern "C" fn curl_write_cb(
    data: *mut c_void,
    size: usize,
    nmemb: usize,
    userp: *mut c_void,
) -> usize {
    let b = userp as *mut CurlBuf;
    let total = size * nmemb;
    if (*b).buf.len() + total > NPM_META_CAP {
        return 0; /* hard cap; aborts the transfer */
    }
    let slice = std::slice::from_raw_parts(data as *const u8, total);
    (*b).buf.extend_from_slice(slice);
    total
}

struct InstallCtx {
    /* proc-6: name → resolved version. Dedup must compare versions, not
     * just names — treating `pkg@1.2.0` and `pkg@2.0.0` in one session as
     * "already installed" silently returned the wrong major. */
    installed: Vec<(String, String)>,
}

fn is_installed<'a>(ctx: &'a InstallCtx, name: &str) -> Option<&'a str> {
    ctx.installed.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
}

fn mark_installed(ctx: &mut InstallCtx, name: &str, version: &str) {
    ctx.installed.push((name.to_string(), version.to_string()));
}

/// proc-7: the registry endpoint is pinned to https; the tarball URL comes
/// from registry JSON, so re-verify it here too (defence in depth against
/// both a builder mistake and the option-level lock).
fn url_is_https(url: &str) -> bool {
    url.len() >= 8 && url[..8].eq_ignore_ascii_case("https://")
}

/// Minimal JSON string extraction equivalent (the C scanned substrings; the
/// registry endpoints are well-formed JSON, so serde is exact here).
fn json_get_str(json: &str, key: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    match key {
        "tarball" => v.get("dist")?.get("tarball")?.as_str().map(|s| s.to_string()),
        "version" => v.get("version").and_then(|x| x.as_str()).map(|s| s.to_string()),
        "shasum" => v.get("dist")?.get("shasum")?.as_str().map(|s| s.to_string()),
        "integrity" => v.get("dist")?.get("integrity")?.as_str().map(|s| s.to_string()),
        "main" => v.get("main").and_then(|x| x.as_str()).map(|s| s.to_string()),
        _ => None,
    }
}

unsafe fn npm_install_recursive(
    pkg_spec: &str,
    dest_dir: &str,
    inst_ctx: &mut InstallCtx,
) -> c_int {
    /* Split name@version (scoped: @scope/name@version) */
    let (name, version) = split_spec(pkg_spec);

    /* Strip semver constraint symbols so the registry URL works */
    let v_trimmed = version.trim_start_matches(|c| "^~=<>v ".contains(c));
    let clean_version = if v_trimmed.is_empty() || v_trimmed == "*" {
        "latest".to_string()
    } else {
        v_trimmed.to_string()
    };

    /* Reject path-traversal in name/version before they reach URLs/paths */
    if !npm_spec_is_safe(&name)
        || (clean_version != "latest" && !npm_spec_is_safe(&clean_version))
    {
        eprintln!("  \x1b[31m✗\x1b[0m Rejected unsafe package spec '{}'", pkg_spec);
        return 1;
    }

    /* proc-6: dedup on resolved versions, not names. Cheap pre-filter:
     * an exact-pin spec equal to a previously RESOLVED version means the
     * same package (no fetch needed). Everything else — including a
     * different-version request that used to be a silent no-op — falls
     * through to the metadata fetch, which resolves the real version and
     * either skips (same) or fails loudly (conflict). */
    if is_installed(inst_ctx, &name) == Some(clean_version.as_str()) {
        return 0; /* already installed or being installed in this session */
    }

    println!("  \x1b[36m→\x1b[0m Resolving {}@{} from registry.npmjs.org...", name, clean_version);

    /* ── Step 1: fetch package metadata ── */
    let meta_url = format!("https://registry.npmjs.org/{}/{}", name, clean_version);
    let meta_c = CString::new(meta_url).unwrap_or_default();

    let curl = sofuu_ffi::curl::curl_easy_init();
    if curl.is_null() {
        eprintln!("curl init failed");
        return 1;
    }
    let mut meta_buf = CurlBuf { buf: Vec::with_capacity(4096) };
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_URL, meta_c.as_ptr());
    /* proc-7: registry endpoint is https-only — lock BOTH the initial
     * protocol and redirects (REDIR alone never constrains the first hop). */
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_PROTOCOLS as c_int, sofuu_ffi::curl::CURLPROTO_HTTPS);
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_REDIR_PROTOCOLS as c_int, sofuu_ffi::curl::CURLPROTO_HTTPS);
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_WRITEFUNCTION, curl_write_cb as *const c_void);
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_WRITEDATA, &mut meta_buf as *mut CurlBuf as *mut c_void);
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_FOLLOWLOCATION as c_int, 1i64);
    let ua = CString::new(format!("sofuu/{}", env!("CARGO_PKG_VERSION"))).unwrap_or_default();
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_USERAGENT, ua.as_ptr());
    let rc = sofuu_ffi::curl::curl_easy_perform(curl);
    sofuu_ffi::curl::curl_easy_cleanup(curl);

    if rc != sofuu_ffi::curl::CURLE_OK || meta_buf.buf.is_empty() {
        let msg = CStr::from_ptr(sofuu_ffi::curl::curl_easy_strerror(rc)).to_string_lossy();
        eprintln!("  \x1b[31m✗\x1b[0m Failed to fetch metadata: {}", msg);
        return 1;
    }

    /* ── Step 2: extract tarball URL from metadata ── */
    let meta_text = String::from_utf8_lossy(&meta_buf.buf).into_owned();
    let tarball_url = json_get_str(&meta_text, "tarball");
    let resolved_version = json_get_str(&meta_text, "version");
    let want_shasum = json_get_str(&meta_text, "shasum");
    let want_integrity = json_get_str(&meta_text, "integrity");

    let Some(tarball_url) = tarball_url else {
        eprintln!("  \x1b[31m✗\x1b[0m Could not parse registry response");
        return 1;
    };

    /* proc-7: the tarball URL is registry-supplied — refuse non-https
     * before it can reach curl (belt to the CURLOPT_PROTOCOLS braces). */
    if !url_is_https(&tarball_url) {
        eprintln!("  \x1b[31m✗\x1b[0m Refusing non-https tarball URL for {}", name);
        return 1;
    }

    /* proc-6 (second half): the registry resolved the version. A same-name
     * install at a DIFFERENT resolved version is a real conflict — fail
     * loudly instead of the old silent same-name no-op; an exact match is
     * a dedup hit. */
    if let Some(prev) = is_installed(inst_ctx, &name) {
        match resolved_version.as_deref() {
            Some(rv) if rv == prev => return 0, /* identical resolved version */
            Some(rv) => {
                eprintln!(
                    "  \x1b[31m✗\x1b[0m Version conflict: {} already installed at {} this session; refusing to also install {}",
                    name, prev, rv
                );
                return 1;
            }
            None => {} /* registry gave no version — fall through to the old behavior */
        }
    }

    if want_shasum.as_deref().map(|s| s.is_empty()).unwrap_or(true) {
        /* Fail closed: modern registries ship dist.integrity (sha512) and
         * may omit the legacy shasum. We don't verify sha512 yet, so refuse
         * to install unverifiable code instead of silently skipping the
         * integrity check. */
        eprintln!(
            "  \x1b[31m✗\x1b[0m No verifiable shasum for {} (dist.integrity={} — sha512 verification not supported yet)",
            name,
            want_integrity.as_deref().unwrap_or("(none)")
        );
        return 1;
    }

    if let Some(rv) = &resolved_version {
        /* proc-6: mark with the RESOLVED version — the dep-recursion cycle
         * guard and the dedup key. (Old code marked name-only before the
         * fetch; a failed install then poisoned the name for the session.) */
        mark_installed(inst_ctx, &name, rv);
        println!("  \x1b[36m→\x1b[0m Installing {}@{}", name, rv);
    }

    /* ── Step 3: download .tgz to a securely-created temp file ──
     * std create_new = O_EXCL (mkstemp's guarantee, portable); 0600 on
     * unix. TMPDIR/TEMP-aware instead of the old hardcoded /tmp. */
    let mut tgz_opts = std::fs::OpenOptions::new();
    tgz_opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        tgz_opts.mode(0o600);
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut opened: Option<(std::fs::File, String)> = None;
    for attempt in 0..8 {
        let cand = std::env::temp_dir()
            .join(format!("sofuu_pkg_{}-{nanos:x}-{attempt}", std::process::id()))
            .to_string_lossy()
            .into_owned();
        match tgz_opts.open(Path::new(&cand)) {
            Ok(f) => {
                opened = Some((f, cand));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => break,
        }
    }
    let Some((mut tgz_file, tmp_tgz)) = opened else {
        eprintln!("  \x1b[31m✗\x1b[0m Cannot create temp file");
        return 1;
    };

    let curl = sofuu_ffi::curl::curl_easy_init();
    if curl.is_null() {
        drop(tgz_file);
        let _ = std::fs::remove_file(&tmp_tgz);
        return 1;
    }
    let tgz_c = CString::new(tarball_url).unwrap_or_default();
    /* Write through the std::fs::File — the default curl callback wants a
     * FILE*, and there is no portable fdopen for a std File handle. */
    let cb: unsafe extern "C" fn(*mut c_char, usize, usize, *mut c_void) -> usize = tgz_write_cb;
    let tgh = &mut tgz_file as *mut std::fs::File as *mut c_void;
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_URL, tgz_c.as_ptr());
    /* proc-7: tarball URL is registry-supplied (pre-checked https) — lock
     * the protocol set for the transfer and its redirects. */
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_PROTOCOLS as c_int, sofuu_ffi::curl::CURLPROTO_HTTPS);
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_REDIR_PROTOCOLS as c_int, sofuu_ffi::curl::CURLPROTO_HTTPS);
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_WRITEFUNCTION, cb as usize as *const c_void);
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_WRITEDATA, tgh);
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_FOLLOWLOCATION as c_int, 1i64);
    let ua = CString::new(format!("sofuu/{}", env!("CARGO_PKG_VERSION"))).unwrap_or_default();
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_USERAGENT, ua.as_ptr());
    /* A 404/5xx registry response must fail the download. */
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_FAILONERROR as c_int, 1i64);
    let rc = sofuu_ffi::curl::curl_easy_perform(curl);
    drop(tgz_file); /* flush + close (was fclose) */
    sofuu_ffi::curl::curl_easy_cleanup(curl);

    if rc != sofuu_ffi::curl::CURLE_OK {
        let msg = CStr::from_ptr(sofuu_ffi::curl::curl_easy_strerror(rc)).to_string_lossy();
        eprintln!("  \x1b[31m✗\x1b[0m Download failed: {}", msg);
        let _ = std::fs::remove_file(&tmp_tgz);
        return 1;
    }

    /* ── Step 3b: verify integrity against the registry shasum (SHA-1) ── */
    let tmp_c2 = CString::new(tmp_tgz.clone()).unwrap_or_default();
    if let Some(want) = &want_shasum {
        /* c_char is i8 on darwin/x86_64-linux but u8 on aarch64-linux —
           never hardcode i8. */
        let mut got = [0 as c_char; 41];
        if sofuu_npm_sha1_file_rs(tmp_c2.as_ptr(), got.as_mut_ptr()) != 0 {
            eprintln!("  \x1b[31m✗\x1b[0m Integrity check failed for {} (expected {})", name, want);
            let _ = std::fs::remove_file(&tmp_tgz);
            return 1;
        }
        let got_s = CStr::from_ptr(got.as_ptr()).to_string_lossy();
        if !got_s.eq_ignore_ascii_case(want) {
            eprintln!("  \x1b[31m✗\x1b[0m Integrity check failed for {} (expected {})", name, want);
            let _ = std::fs::remove_file(&tmp_tgz);
            return 1;
        }
    }

    /* ── Step 4: extract into node_modules/<name> ── */
    let nm_dir = Path::new(dest_dir).join("node_modules");
    let _ = std::fs::create_dir_all(&nm_dir);
    let pkg_dir = nm_dir.join(&name);

    /* Create parent directories for scoped packages */
    if name.starts_with('@') {
        if let Some(slash) = name[1..].find('/') {
            let scope_dir = nm_dir.join(&name[..1 + slash]);
            let _ = std::fs::create_dir_all(&scope_dir);
        }
    }
    /* Reject a pre-existing symlink at the extraction target. */
    if let Ok(meta) = std::fs::symlink_metadata(&pkg_dir) {
        if meta.file_type().is_symlink() {
            eprintln!("  \x1b[31m✗\x1b[0m Refusing to extract over symlink {}", pkg_dir.display());
            let _ = std::fs::remove_file(&tmp_tgz);
            return 1;
        }
    }
    let _ = std::fs::create_dir_all(&pkg_dir);

    /* npm tarballs have an extra "package/" prefix inside the .tgz. */
    let pkg_c = CString::new(pkg_dir.to_string_lossy().as_bytes()).unwrap_or_default();
    let mut errbuf = [0 as c_char; 256];
    let xrc = sofuu_npm_extract_safe_rs(tmp_c2.as_ptr(), pkg_c.as_ptr(), 1, errbuf.as_mut_ptr(), errbuf.len());
    let _ = std::fs::remove_file(&tmp_tgz);
    if xrc != 0 {
        let msg = CStr::from_ptr(errbuf.as_ptr()).to_string_lossy();
        eprintln!("  \x1b[31m✗\x1b[0m Extraction failed: {}", if msg.is_empty() { "unknown" } else { &msg });
        return 1;
    }

    println!(
        "  \x1b[32m✓\x1b[0m Installed {}@{} → node_modules/{}",
        name,
        resolved_version.as_deref().unwrap_or("?"),
        name
    );

    /* ── Step 5: package.json → transitive dependencies ── */
    let pjson_path = pkg_dir.join("package.json");
    if let Ok(text) = std::fs::read_to_string(&pjson_path) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(deps) = parsed.get("dependencies").and_then(|d| d.as_object()) {
                for (dep_name, ver) in deps {
                    if let Some(dep_ver) = ver.as_str() {
                        let child_spec = format!("{}@{}", dep_name, dep_ver);
                        let c = CString::new(child_spec).unwrap_or_default();
                        let d = CString::new(dest_dir).unwrap_or_default();
                        npm_install_recursive(c.to_str().unwrap_or(""), d.to_str().unwrap_or(""), inst_ctx);
                    }
                }
            }
        }
    }

    0
}

fn split_spec(pkg_spec: &str) -> (String, String) {
    let at = if pkg_spec.starts_with('@') {
        /* scoped: find @ after the / */
        match pkg_spec[1..].find('/') {
            Some(slash) => pkg_spec[1 + slash..].find('@').map(|i| 1 + slash + i),
            None => None,
        }
    } else {
        pkg_spec.find('@')
    };
    match at {
        Some(i) if i > 0 => (pkg_spec[..i].to_string(), pkg_spec[i + 1..].to_string()),
        _ => (pkg_spec.to_string(), "latest".to_string()),
    }
}

/// C symbol: install one package (the retired resolver.h contract).
///
/// # Safety
/// C-string contract (main.rs passes NUL-terminated strings).
#[no_mangle]
pub unsafe extern "C" fn npm_install(pkg_spec: *const c_char, dest_dir: *const c_char) -> c_int {
    let (Some(spec), Some(dest)) = (cstr(pkg_spec), cstr(dest_dir)) else {
        return 1;
    };
    let mut ctx = InstallCtx { installed: Vec::new() };
    let rc = npm_install_recursive(spec, dest, &mut ctx);
    rc
}

/// C symbol: install all dependencies + devDependencies from package.json.
///
/// # Safety
/// C-string contract (main.rs passes NUL-terminated strings).
#[no_mangle]
pub unsafe extern "C" fn npm_install_local_package_json(dest_dir: *const c_char) -> c_int {
    let Some(dest) = cstr(dest_dir) else {
        return 1;
    };
    let pjson = Path::new(dest).join("package.json");
    let Ok(pkg_json) = std::fs::read_to_string(&pjson) else {
        eprintln!("\x1b[31mError:\x1b[0m No package.json found in {}", dest);
        return 1;
    };

    let mut rc = 0;
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&pkg_json) else {
        eprintln!("\x1b[31mError:\x1b[0m Invalid package.json");
        return 1;
    };

    /* Install dependencies */
    if let Some(deps) = parsed.get("dependencies").and_then(|d| d.as_object()) {
        println!("\n\x1b[1mInstalling dependencies...\x1b[0m");
        for (dep_name, ver) in deps {
            if let Some(dep_ver) = ver.as_str() {
                let child_spec = format!("{}@{}", dep_name, dep_ver);
                let c = CString::new(child_spec).unwrap_or_default();
                let d = CString::new(dest).unwrap_or_default();
                if npm_install(c.as_ptr(), d.as_ptr()) != 0 {
                    rc = 1;
                }
            }
        }
    }

    /* Install devDependencies */
    if let Some(dev_deps) = parsed.get("devDependencies").and_then(|d| d.as_object()) {
        println!("\n\x1b[1mInstalling devDependencies...\x1b[0m");
        for (dep_name, ver) in dev_deps {
            if let Some(dep_ver) = ver.as_str() {
                let child_spec = format!("{}@{}", dep_name, dep_ver);
                let c = CString::new(child_spec).unwrap_or_default();
                let d = CString::new(dest).unwrap_or_default();
                if npm_install(c.as_ptr(), d.as_ptr()) != 0 {
                    rc = 1;
                }
            }
        }
    }

    rc
}

#[cfg(test)]
mod tests {
    use super::*;

    /* proc-7: tarball URLs come from registry JSON — the guard must accept
     * https (any case) and reject http, protocol-relative, and garbage. */
    #[test]
    fn url_is_https_accepts_only_https() {
        assert!(url_is_https("https://registry.npmjs.org/x.tgz"));
        assert!(url_is_https("HTTPS://example.com/pkg.tgz"));
        assert!(!url_is_https("http://registry.npmjs.org/x.tgz"));
        assert!(!url_is_https("//registry.npmjs.org/x.tgz"));
        assert!(!url_is_https("ftp://x/y.tgz"));
        assert!(!url_is_https("https:/missing-slash.tgz"));
        assert!(!url_is_https(""));
    }

    /* proc-6: the session dedup tracks RESOLVED versions — exact match is
     * a hit, a different version is NOT (it must reach the conflict
     * check after the metadata fetch). */
    #[test]
    fn install_dedup_matches_resolved_versions_only() {
        let mut ctx = InstallCtx { installed: Vec::new() };
        assert_eq!(is_installed(&ctx, "left-pad"), None);
        mark_installed(&mut ctx, "left-pad", "1.3.0");
        assert_eq!(is_installed(&ctx, "left-pad"), Some("1.3.0"));
        /* Same name, different resolved version → NOT a dedup hit. */
        assert_eq!(is_installed(&ctx, "left-pad") == Some("2.0.0"), false);
        /* Name lookups are exact. */
        assert_eq!(is_installed(&ctx, "left"), None);
    }
}
