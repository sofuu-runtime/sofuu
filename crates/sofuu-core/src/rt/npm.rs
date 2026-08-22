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
    installed: Vec<String>,
}

fn is_installed(ctx: &InstallCtx, name: &str) -> bool {
    ctx.installed.iter().any(|i| i == name)
}

fn mark_installed(ctx: &mut InstallCtx, name: &str) {
    ctx.installed.push(name.to_string());
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

    if is_installed(inst_ctx, &name) {
        return 0; /* already installed or being installed in this session */
    }
    mark_installed(inst_ctx, &name);

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
        println!("  \x1b[36m→\x1b[0m Installing {}@{}", name, rv);
    }

    /* ── Step 3: download .tgz to a securely-created temp file ── */
    let mut tmp_c = [0i8; 32];
    let tmp_tpl = c"/tmp/sofuu_pkg_XXXXXX";
    std::ptr::copy_nonoverlapping(tmp_tpl.as_ptr(), tmp_c.as_mut_ptr(), 21);
    let tfd = libc::mkstemp(tmp_c.as_mut_ptr()); /* O_EXCL + 0600 */
    if tfd < 0 {
        eprintln!("  \x1b[31m✗\x1b[0m Cannot create temp file");
        return 1;
    }
    let tmp_tgz = CStr::from_ptr(tmp_c.as_ptr()).to_string_lossy().into_owned();

    let curl = sofuu_ffi::curl::curl_easy_init();
    if curl.is_null() {
        libc::close(tfd);
        let _ = std::fs::remove_file(&tmp_tgz);
        return 1;
    }
    let tgz_f = libc::fdopen(tfd, c"wb".as_ptr());
    let tgz_c = CString::new(tarball_url).unwrap_or_default();
    if tgz_f.is_null() {
        libc::close(tfd);
        let _ = std::fs::remove_file(&tmp_tgz);
        sofuu_ffi::curl::curl_easy_cleanup(curl);
        return 1;
    }
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_URL, tgz_c.as_ptr());
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_WRITEFUNCTION, std::ptr::null::<c_void>() as *const c_void);
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_WRITEDATA, tgz_f);
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_FOLLOWLOCATION as c_int, 1i64);
    let ua = CString::new(format!("sofuu/{}", env!("CARGO_PKG_VERSION"))).unwrap_or_default();
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_USERAGENT, ua.as_ptr());
    /* A 404/5xx registry response must fail the download. */
    sofuu_ffi::curl::curl_easy_setopt(curl, sofuu_ffi::curl::CURLOPT_FAILONERROR as c_int, 1i64);
    let rc = sofuu_ffi::curl::curl_easy_perform(curl);
    libc::fclose(tgz_f);
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
        let mut got = [0i8; 41];
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
    let mut errbuf = [0i8; 256];
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
