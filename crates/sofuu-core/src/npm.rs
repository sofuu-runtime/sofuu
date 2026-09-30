// sofuu-core — npm module resolution + package install (safe core).
//
// Rust port of the security-sensitive pure logic in src/npm/resolver.c:
//   - safe package-spec validation (path traversal rejection)
//   - name@version parsing (scoped + semver-stripped)
//   - npm resolution algorithm (walk up node_modules, package.json "main")
//   - SHA-1 (for tarball integrity)
//   - safe JSON field extraction
//
// The network fetch (curl) and tarball extraction (tar) stay in C via FFI
// (extreme low-level I/O) — this module owns the parsing + validation.

use std::path::{Path, PathBuf};

// ── §1 Spec safety ──────────────────────────────────────────────

/// Reject names/versions that could escape the install dir (path traversal).
/// Scoped names legitimately contain one '/'; `..`, leading '/', backslashes,
/// control chars, and URL metacharacters (`?`/`#` — a registry or resolver
/// would treat everything after them as query/fragment, so what we install
/// would no longer match the name we validated) are rejected.
pub fn spec_is_safe(s: &str) -> bool {
    if s.is_empty() || s.starts_with('/') || s.starts_with('.') {
        return false;
    }
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        let c = bytes[i];
        if c == b'\\' || c < 0x20 || c == b'?' || c == b'#' {
            return false;
        }
        if c == b'.' && i + 1 < bytes.len() && bytes[i + 1] == b'.' {
            return false;
        }
    }
    true
}

/// Split `name@version` handling scoped packages (`@scope/name@version`).
/// Returns (name, version_or_latest).
pub fn split_spec(spec: &str) -> (String, String) {
    if spec.starts_with('@') {
        // scoped: the version '@' comes after the '/'
        if spec[1..].contains('/') {
            let after_scope = &spec[1..]; // skip leading @
            let slash_idx = after_scope.find('/').unwrap();
            if let Some(at) = spec[slash_idx + 1..].find('@') {
                let at = slash_idx + 1 + at;
                return (spec[..at].to_string(), spec[at + 1..].to_string());
            }
        }
        (spec.to_string(), "latest".to_string())
    } else if let Some(at) = spec.find('@') {
        if at == 0 {
            (spec.to_string(), "latest".to_string())
        } else {
            (spec[..at].to_string(), spec[at + 1..].to_string())
        }
    } else {
        (spec.to_string(), "latest".to_string())
    }
}

/// Strip semver constraint symbols (`^~=<>v `) and `*` → "latest".
pub fn clean_version(v: &str) -> String {
    let v = v.trim_start_matches(['^', '~', '=', '<', '>', 'v', ' ']);
    if v.is_empty() || v == "*" || v.starts_with('*') {
        "latest".to_string()
    } else {
        v.to_string()
    }
}

// ── §2 Resolution algorithm ─────────────────────────────────────

/// Read a JSON string field safely (first occurrence of "key":"value").
pub fn json_get_str(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let p = json.find(&pattern)?;
    let rest = &json[p + pattern.len()..];
    // Skip whitespace + colon.
    let rest = rest.trim_start_matches([' ', '\t', '\n', '\r']);
    let rest = rest.strip_prefix(':')?;
    let rest = rest.trim_start_matches([' ', '\t', '\n', '\r']);
    let rest = rest.strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            break;
        }
        if c == '\\' {
            if let Some(e) = chars.next() {
                out.push(match e {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    other => other,
                });
            }
            continue;
        }
        out.push(c);
    }
    Some(out)
}

fn read_file_str(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// Try to resolve a module inside a specific `node_modules/<name>` dir.
pub fn resolve_in_pkg_dir(pkg_dir: &Path) -> Option<PathBuf> {
    // 1. package.json "main" field.
    let pkg_json_path = pkg_dir.join("package.json");
    if let Some(json) = read_file_str(&pkg_json_path) {
        if let Some(main_val) = json_get_str(&json, "main") {
            let mrel = main_val.strip_prefix("./").unwrap_or(&main_val);
            let main_path = pkg_dir.join(mrel);
            if main_path.is_file() {
                return main_path.canonicalize().ok();
            }
            let with_js = PathBuf::from(format!("{}.js", main_path.to_string_lossy()));
            if with_js.is_file() {
                return with_js.canonicalize().ok();
            }
        }
    }
    // 2. index.js
    let idx = pkg_dir.join("index.js");
    if idx.is_file() {
        return idx.canonicalize().ok();
    }
    // 3. index.mjs
    let idxm = pkg_dir.join("index.mjs");
    if idxm.is_file() {
        return idxm.canonicalize().ok();
    }
    None
}

/// Node.js-style module resolution: walk up from `start_dir` looking for
/// `node_modules/<name>`.
pub fn npm_resolve(start_dir: &Path, module_name: &str) -> Option<PathBuf> {
    let mut dir = start_dir.to_path_buf();
    loop {
        let nm = dir.join("node_modules");
        if nm.is_dir() {
            let pkg_dir = nm.join(module_name);
            if pkg_dir.is_dir() {
                if let Some(r) = resolve_in_pkg_dir(&pkg_dir) {
                    return Some(r);
                }
            }
            // Direct file: node_modules/<name>.js
            let direct = nm.join(format!("{module_name}.js"));
            if direct.is_file() {
                return direct.canonicalize().ok();
            }
        }
        // Go up one directory.
        let parent = dir.parent()?;
        if parent == dir {
            break;
        }
        dir = parent.to_path_buf();
    }
    None
}

// ── §3 SHA-1 (tarball integrity) ────────────────────────────────

/// Minimal SHA-1 — used to verify npm tarball shasum (same algorithm as C).
pub struct Sha1 {
    h: [u32; 5],
    len: u64,
    buf: [u8; 64],
    n: usize,
}

impl Sha1 {
    pub fn new() -> Self {
        Self {
            h: [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0],
            len: 0,
            buf: [0u8; 64],
            n: 0,
        }
    }

    fn block(&mut self, p: &[u8]) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([p[i * 4], p[i * 4 + 1], p[i * 4 + 2], p[i * 4 + 3]]);
        }
        for i in 16..80 {
            let x = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16];
            w[i] = x.rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (self.h[0], self.h[1], self.h[2], self.h[3], self.h[4]);
        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        while !data.is_empty() {
            let k = (64 - self.n).min(data.len());
            self.buf[self.n..self.n + k].copy_from_slice(&data[..k]);
            self.n += k;
            data = &data[k..];
            if self.n == 64 {
                let block = self.buf;
                self.block(&block);
                self.n = 0;
            }
        }
    }

    /// Finalize and return lowercase hex digest.
    pub fn final_hex(mut self) -> String {
        let bits = self.len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.n != 56 {
            self.update(&[0]);
        }
        let mut lb = [0u8; 8];
        for i in 0..8 {
            lb[i] = (bits >> (56 - i * 8)) as u8;
        }
        self.update(&lb);
        self.h.iter().map(|h| format!("{h:08x}")).collect()
    }

    /// SHA-1 of a byte slice as lowercase hex.
    pub fn hex(data: &[u8]) -> String {
        let mut s = Self::new();
        s.update(data);
        s.final_hex()
    }

    /// SHA-1 of a file as lowercase hex.
    pub fn file_hex(path: &Path) -> Option<String> {
        let data = std::fs::read(path).ok()?;
        Some(Self::hex(&data))
    }
}

// ── §4 Safe tarball extraction (pure, no `tar` shell-out) ───────

/// A safe, minimal gzip-tar extractor: rejects path traversal, absolute
/// paths, and symlinks. This is the security-critical replacement for
/// shelling out to `tar` (which the C version does via fork+exec).
/// `strip_components` = 1 removes the leading "package/" prefix.
pub fn extract_tarball_safe(tgz: &[u8], dest: &Path, strip_components: usize) -> Result<usize, String> {
    // Decompress gzip (zlib via flate2 is a dep; keep it minimal here).
    // NOTE: flate2 is added in Cargo.toml. If unavailable, this falls back.
    let raw = inflate_gzip(tgz)?;
    extract_tar(&raw, dest, strip_components)
}

fn inflate_gzip(data: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Read;
    // Bomb guard: cap inflated size (gz bomb would OOM via read_to_end).
    const MAX_INFLATED: u64 = 64 * 1024 * 1024; // 64MB
    let mut out = Vec::new();
    let mut decoder = flate2::read::GzDecoder::new(data);
    decoder
        .take(MAX_INFLATED + 1)
        .read_to_end(&mut out)
        .map_err(|e| format!("gzip: {e}"))?;
    if out.len() as u64 > MAX_INFLATED {
        return Err("gzip bomb: inflated size exceeds 64MB cap".into());
    }
    Ok(out)
}

fn extract_tar(data: &[u8], dest: &Path, strip: usize) -> Result<usize, String> {
    // Tar bomb guards: file count, total bytes, per-file size.
    const MAX_TAR_FILES: usize = 10_000;
    const MAX_TAR_TOTAL: usize = 256 * 1024 * 1024; // 256MB
    const MAX_TAR_FILE: usize = 64 * 1024 * 1024; // 64MB single file
    let mut count = 0usize;
    let mut total: usize = 0;
    let mut off = 0usize;
    let block = 512usize;
    while off + block <= data.len() {
        let header = &data[off..off + block];
        // Empty block (all zeros) → end of archive (two zero blocks).
        if header.iter().all(|&b| b == 0) {
            break;
        }
        let name_bytes = &header[0..100];
        let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(100);
        let name = String::from_utf8_lossy(&name_bytes[..name_end]).into_owned();
        let size_bytes = &header[124..136];
        let size_str = String::from_utf8_lossy(size_bytes);
        let size = u64::from_str_radix(size_str.trim_end_matches(['\0', ' ']), 8)
            .map_err(|_| format!("bad tar size for {name}"))? as usize;
        let typeflag = header[156];
        let data_start = off + block;
        let data_end = data_start + size;

        // Skip the `package/` prefix.
        let rel = if strip > 0 {
            let parts: Vec<&str> = name.splitn(strip + 1, '/').collect();
            parts.get(strip).copied().unwrap_or("")
        } else {
            &name
        };
        if rel.is_empty() {
            // It's the stripped prefix dir itself — skip.
            off = data_end + (512 - data_end % 512) % 512;
            continue;
        }

        // Security: reject traversal, absolute paths, and symlinks.
        let rel_path = Path::new(rel);
        if rel_path.is_absolute()
            || rel_path.components().any(|c| matches!(c, std::path::Component::ParentDir))
            || typeflag == b'2' // symlink
        {
            return Err(format!("unsafe tar entry: {name}"));
        }
        if size > MAX_TAR_FILE {
            return Err(format!("tar entry too large ({size} bytes): {name}"));
        }
        if count >= MAX_TAR_FILES {
            return Err("tar bomb: too many files (>10000)".into());
        }
        if total.saturating_add(size) > MAX_TAR_TOTAL {
            return Err("tar bomb: total size exceeds 256MB cap".into());
        }

        let target = dest.join(rel_path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
        }
        match typeflag {
            b'5' => {
                // directory
                std::fs::create_dir_all(&target).map_err(|e| format!("mkdir {rel}: {e}"))?;
            }
            b'0' | b'\0' | b' ' => {
                // regular file
                std::fs::write(&target, &data[data_start..data_end.min(data.len())])
                    .map_err(|e| format!("write {rel}: {e}"))?;
                count += 1;
                total = total.saturating_add(size);
            }
            _ => { /* ignore other types (hardlinks, devices) */ }
        }
        off = data_end + (512 - data_end % 512) % 512;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_safety() {
        assert!(spec_is_safe("lodash"));
        assert!(spec_is_safe("@scope/pkg"));
        assert!(!spec_is_safe("../evil"));
        assert!(!spec_is_safe("/abs"));
        assert!(!spec_is_safe("a\\b"));
        assert!(!spec_is_safe(".."));
        // P3 (AUDIT-2026-09-07): URL metacharacters — everything after ? or
        // # would be treated as query/fragment by a registry/resolver, so
        // the installed package would no longer match the validated name.
        assert!(!spec_is_safe("lodash?version=1"));
        assert!(!spec_is_safe("lodash#frag"));
        assert!(!spec_is_safe("@scope/pkg#main"));
    }

    #[test]
    fn split_scoped_and_versioned() {
        assert_eq!(split_spec("lodash"), ("lodash".into(), "latest".into()));
        assert_eq!(split_spec("lodash@4.17.21"), ("lodash".into(), "4.17.21".into()));
        assert_eq!(split_spec("@scope/pkg@1.2.3"), ("@scope/pkg".into(), "1.2.3".into()));
        assert_eq!(split_spec("@scope/pkg"), ("@scope/pkg".into(), "latest".into()));
    }

    #[test]
    fn clean_versions() {
        assert_eq!(clean_version("^1.2.3"), "1.2.3");
        assert_eq!(clean_version("~0.4"), "0.4");
        assert_eq!(clean_version(">=2.0.0"), "2.0.0");
        assert_eq!(clean_version("v1.0"), "1.0");
        assert_eq!(clean_version("*"), "latest");
        assert_eq!(clean_version(""), "latest");
    }

    #[test]
    fn json_extraction() {
        let j = r#"{"name":"lodash","version":"4.17.21","dist":{"tarball":"https://x/y.tgz","shasum":"abc123"}}"#;
        assert_eq!(json_get_str(j, "name"), Some("lodash".into()));
        assert_eq!(json_get_str(j, "version"), Some("4.17.21".into()));
        assert_eq!(json_get_str(j, "tarball"), Some("https://x/y.tgz".into()));
        assert_eq!(json_get_str(j, "missing"), None);
    }

    #[test]
    fn json_escapes() {
        let j = r#"{"msg":"a \"quoted\"\nline"}"#;
        assert_eq!(json_get_str(j, "msg"), Some("a \"quoted\"\nline".into()));
    }

    #[test]
    fn sha1_known_vector() {
        // "abc" → a9993e364706816aba3e25717850c26c9cd0d89d
        assert_eq!(Sha1::hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(Sha1::hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }

    #[test]
    fn sha1_multiblock() {
        // 1000 'a' chars — known SHA-1 (multi-block, exercises padding).
        let s = "a".repeat(1000);
        assert_eq!(Sha1::hex(s.as_bytes()), "291e9a6c66994949b57ba5e650361e98fc36b1ba");
    }

    #[test]
    fn resolve_walks_up() {
        // Build a temp tree: /tmp/x/node_modules/foo/index.js
        let tmp = std::env::temp_dir().join(format!("sofuu-npm-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let pkg = tmp.join("node_modules/foo");
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::write(pkg.join("index.js"), "module.exports=1;").unwrap();
        let start = tmp.join("a/b/c");
        std::fs::create_dir_all(&start).unwrap();

        let r = npm_resolve(&start, "foo").expect("resolve");
        assert!(r.ends_with("index.js"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn resolve_respects_package_main() {
        let tmp = std::env::temp_dir().join(format!("sofuu-npm-test2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let pkg = tmp.join("node_modules/bar");
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::write(pkg.join("package.json"), r#"{"main":"lib/entry.js"}"#).unwrap();
        std::fs::create_dir_all(pkg.join("lib")).unwrap();
        std::fs::write(pkg.join("lib/entry.js"), "module.exports=2;").unwrap();

        let r = npm_resolve(&tmp, "bar").expect("resolve");
        assert!(r.ends_with("lib/entry.js"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn tar_extraction_rejects_traversal() {
        // Craft a tar with a `../evil` entry.
        let mut tar = Vec::new();
        let mut header = [0u8; 512];
        let name = b"../evil";
        header[..name.len()].copy_from_slice(name);
        // size = 4 octal
        let size = format!("{:07o}", 4);
        header[124..131].copy_from_slice(size.as_bytes());
        header[156] = b'0';
        tar.extend_from_slice(&header);
        tar.extend_from_slice(b"pwn!");
        tar.extend_from_slice(&[0u8; 508]); // pad to 512
        tar.extend_from_slice(&[0u8; 1024]); // end blocks

        let dest = std::env::temp_dir().join(format!("sofuu-tar-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dest);
        std::fs::create_dir_all(&dest).unwrap();
        let r = extract_tar(&tar, &dest, 0);
        assert!(r.is_err(), "traversal must be rejected");
        let _ = std::fs::remove_dir_all(&dest);
    }
}
