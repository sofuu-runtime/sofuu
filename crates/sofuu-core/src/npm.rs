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

/// Parse `package.json`. `None` when the file is missing or not an object —
/// the caller then falls back to the legacy fields, which is exactly what a
/// malformed manifest deserves.
fn read_package_json(pkg_dir: &Path) -> Option<serde_json::Value> {
    let text = read_file_str(&pkg_dir.join("package.json"))?;
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(v) if v.is_object() => Some(v),
        _ => None,
    }
}

// ── §2a Conditional "exports" (PLAN-MODERN-NPM, pillar A) ───────

/// The condition keys an ES module import evaluates, in precedence order.
///
/// `import` first because an ESM import is the primary path; `module` is the
/// bundler convention a large share of modern packages publish; then
/// `default`, which is Node's own fallback. `node` is last on purpose: it
/// selects Node-specific builds that usually reach for native APIs we do not
/// have, so it is a fallback rather than a primary match.
const EXPORT_CONDITIONS_ESM: [&str; 4] = ["import", "module", "default", "node"];

/// The same list for `require()`. The only difference is the leading
/// `require` condition: a package that publishes separate ESM and CJS builds
/// must give `require()` the CJS one, or a `.mjs` file is loaded as
/// CommonJS (which is exactly the "unsupported keyword: export" failure this
/// list exists to prevent).
const EXPORT_CONDITIONS_CJS: [&str; 5] = ["require", "module", "default", "node", "import"];

/// Split a bare specifier into `(package name, exports subpath)`.
///
/// `"pkg"` → `("pkg", ".")`, `"pkg/sub/x.js"` → `("pkg", "./sub/x.js")`,
/// `"@scope/pkg/sub"` → `("@scope/pkg", "./sub")`.
///
/// `None` for relative/absolute/builtin-style specifiers: those are not
/// package specifiers and the legacy walk below keeps handling them exactly
/// as before.
pub fn split_package_spec(spec: &str) -> Option<(String, String)> {
    if spec.is_empty() || spec.starts_with('.') || spec.starts_with('/') || spec.starts_with('#') {
        return None;
    }
    if spec.starts_with('@') {
        let mut parts = spec.splitn(3, '/');
        let scope = parts.next()?;
        let name = parts.next()?;
        if scope.len() < 2 || name.is_empty() {
            return None;
        }
        let pkg = format!("{scope}/{name}");
        return Some(match parts.next() {
            Some(tail) if !tail.is_empty() => (pkg, format!("./{tail}")),
            _ => (pkg, ".".to_string()),
        });
    }
    match spec.split_once('/') {
        Some((name, tail)) if !name.is_empty() && !tail.is_empty() => {
            Some((name.to_string(), format!("./{tail}")))
        }
        _ => Some((spec.to_string(), ".".to_string())),
    }
}

/// Resolve `subpath` ("." or "./x") inside a package directory.
///
/// Order: `exports` first (the modern contract), then for "." the legacy
/// `module` → `main` → `index.js`/`index.mjs` chain, and for a subpath a
/// literal deep import. Being permissive on the deep-import fallback is
/// deliberate: Node refuses it when `exports` exists, but failing an import
/// is a worse outcome for an agent runtime than reading a file the package
/// already ships. Traversal is still refused.
pub fn resolve_in_pkg(pkg_dir: &Path, subpath: &str) -> Option<PathBuf> {
    resolve_in_pkg_with(pkg_dir, subpath, &EXPORT_CONDITIONS_ESM)
}

/// `resolve_in_pkg` with an explicit condition order (ESM vs CJS).
pub fn resolve_in_pkg_with(
    pkg_dir: &Path,
    subpath: &str,
    conditions: &[&str],
) -> Option<PathBuf> {
    let manifest = read_package_json(pkg_dir);
    if let Some(v) = &manifest {
        if let Some(exports) = v.get("exports") {
            if let Some(p) = resolve_exports_value(pkg_dir, exports, subpath, conditions, 0) {
                return Some(p);
            }
        }
        if subpath == "." {
            for key in ["module", "main"] {
                if let Some(rel) = v.get(key).and_then(|x| x.as_str()) {
                    if let Some(p) = package_relative_path(pkg_dir, rel) {
                        return Some(p);
                    }
                }
            }
        }
    }
    if subpath == "." {
        let idx = pkg_dir.join("index.js");
        if idx.is_file() {
            return idx.canonicalize().ok();
        }
        let idxm = pkg_dir.join("index.mjs");
        if idxm.is_file() {
            return idxm.canonicalize().ok();
        }
        None
    } else {
        package_relative_path(pkg_dir, subpath)
    }
}

/// Try to resolve a module inside a specific `node_modules/<name>` dir.
pub fn resolve_in_pkg_dir(pkg_dir: &Path) -> Option<PathBuf> {
    resolve_in_pkg(pkg_dir, ".")
}

/// One exports target, resolved to a real file.
///
/// Handles every shape the specification allows: a string, an array of
/// alternatives, a conditions object (evaluated in `EXPORT_CONDITIONS`
/// order), and a subpath map with exact or single-`*` wildcard keys.
///
/// Existence is checked at every string leaf, not once at the end: an array
/// or condition chain must be able to skip a target that points at a file
/// the package did not ship, exactly like Node does. `depth` bounds the
/// recursion so a hostile manifest cannot spin.
fn resolve_exports_value(
    pkg_dir: &Path,
    v: &serde_json::Value,
    subpath: &str,
    conditions: &[&str],
    depth: u8,
) -> Option<PathBuf> {
    if depth > 8 {
        return None;
    }
    match v {
        serde_json::Value::String(s) => {
            if subpath == "." {
                package_relative_path(pkg_dir, s)
            } else {
                None
            }
        }
        serde_json::Value::Array(items) => items
            .iter()
            .find_map(|x| resolve_exports_value(pkg_dir, x, subpath, conditions, depth + 1)),
        serde_json::Value::Object(map) => {
            let is_subpath_map = !map.is_empty() && map.keys().all(|k| k.starts_with('.'));
            if !is_subpath_map {
                /* A conditions object is only meaningful for the root. */
                return if subpath == "." {
                    resolve_condition_object(pkg_dir, map, None, conditions, depth + 1)
                } else {
                    None
                };
            }
            if let Some(target) = map.get(subpath) {
                return resolve_export_target(pkg_dir, target, None, conditions, depth + 1);
            }
            /* Wildcard: the LONGEST matching prefix wins (spec). */
            let mut best: Option<(usize, &serde_json::Value, String)> = None;
            for (key, target) in map {
                let Some(star) = key.find('*') else { continue };
                let prefix = &key[..star];
                let suffix = &key[star + 1..];
                if !subpath.starts_with(prefix) || !subpath.ends_with(suffix) {
                    continue;
                }
                if subpath.len() < prefix.len() + suffix.len() {
                    continue;
                }
                let matched = &subpath[prefix.len()..subpath.len() - suffix.len()];
                if best.as_ref().map_or(true, |b| prefix.len() > b.0) {
                    best = Some((prefix.len(), target, matched.to_string()));
                }
            }
            best.and_then(|(_, target, matched)| {
                resolve_export_target(pkg_dir, target, Some(&matched), conditions, depth + 1)
            })
        }
        _ => None,
    }
}

/// A target value inside `exports`: string (with `*` substitution), array of
/// alternatives, or a nested conditions object.
fn resolve_export_target(
    pkg_dir: &Path,
    v: &serde_json::Value,
    star: Option<&str>,
    conditions: &[&str],
    depth: u8,
) -> Option<PathBuf> {
    if depth > 8 {
        return None;
    }
    match v {
        serde_json::Value::String(s) => {
            let target = match star {
                Some(m) => s.replace('*', m),
                None => s.clone(),
            };
            package_relative_path(pkg_dir, &target)
        }
        serde_json::Value::Array(items) => items
            .iter()
            .find_map(|x| resolve_export_target(pkg_dir, x, star, conditions, depth + 1)),
        serde_json::Value::Object(map) => {
            resolve_condition_object(pkg_dir, map, star, conditions, depth + 1)
        }
        _ => None,
    }
}

fn resolve_condition_object(
    pkg_dir: &Path,
    map: &serde_json::Map<String, serde_json::Value>,
    star: Option<&str>,
    conditions: &[&str],
    depth: u8,
) -> Option<PathBuf> {
    for cond in conditions {
        if let Some(target) = map.get(*cond) {
            if let Some(r) = resolve_export_target(pkg_dir, target, star, conditions, depth) {
                return Some(r);
            }
        }
    }
    None
}

/// Turn a package-relative target into an existing file.
///
/// Rejects absolute paths and any `..` component (a package must not point a
/// resolution outside itself), then accepts the literal path or one of the
/// usual script extensions. `exports` targets are exact by specification, but
/// extensionless targets are common enough in the wild that refusing them
/// would cost real packages; the file must still exist either way.
fn package_relative_path(pkg_dir: &Path, rel: &str) -> Option<PathBuf> {
    let rel = rel.strip_prefix("./").unwrap_or(rel);
    if rel.is_empty() {
        return None;
    }
    let rel_path = Path::new(rel);
    if rel_path.is_absolute()
        || rel_path.components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return None;
    }
    let base = pkg_dir.join(rel_path);
    let mut candidates = vec![base.clone()];
    for ext in ["js", "mjs", "cjs", "ts"] {
        candidates.push(PathBuf::from(format!("{}.{ext}", base.to_string_lossy())));
    }
    candidates.push(base.join("index.js"));
    candidates.push(base.join("index.mjs"));
    for c in candidates {
        if c.is_file() {
            /* canonicalize for the loader (and stack traces); confinement is
             * lexical, so a pnpm/link farm whose real path lives outside the
             * package dir keeps working. */
            return Some(c.canonicalize().unwrap_or(c));
        }
    }
    None
}

/// Node.js-style module resolution: walk up from `start_dir` looking for
/// `node_modules/<name>`, honouring `exports` / `module` / `main`.
pub fn npm_resolve(start_dir: &Path, module_name: &str) -> Option<PathBuf> {
    let spec = split_package_spec(module_name);
    let mut dir = start_dir.to_path_buf();
    loop {
        let nm = dir.join("node_modules");
        if nm.is_dir() {
            if let Some((pkg, subpath)) = &spec {
                let pkg_dir = nm.join(pkg);
                if pkg_dir.is_dir() {
                    if let Some(r) = resolve_in_pkg(&pkg_dir, subpath) {
                        return Some(r);
                    }
                }
            }
            /* Direct file: node_modules/<name>.js (a full specifier, e.g.
             * "foo/lib/x.js", or a plain "foo.js" package). */
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

/// `npm_resolve` with CommonJS conditions (`require` first).
pub fn npm_resolve_cjs(start_dir: &Path, module_name: &str) -> Option<PathBuf> {
    let spec = split_package_spec(module_name);
    let mut dir = start_dir.to_path_buf();
    loop {
        let nm = dir.join("node_modules");
        if nm.is_dir() {
            if let Some((pkg, subpath)) = &spec {
                let pkg_dir = nm.join(pkg);
                if pkg_dir.is_dir() {
                    if let Some(r) = resolve_in_pkg_with(&pkg_dir, subpath, &EXPORT_CONDITIONS_CJS)
                    {
                        return Some(r);
                    }
                }
            }
            let direct = nm.join(format!("{module_name}.js"));
            if direct.is_file() {
                return direct.canonicalize().ok();
            }
        }
        let parent = dir.parent()?;
        if parent == dir {
            break;
        }
        dir = parent.to_path_buf();
    }
    None
}

/// Why `module_name` did not resolve, in one sentence a developer can act on
/// (PLAN-MODERN-NPM pillar C, third row). `None` for non-package specifiers.
///
/// Only called on the failure path, so the extra walk costs nothing in the
/// common case.
pub fn explain_module_not_found(start_dir: &Path, module_name: &str) -> Option<String> {
    let (pkg, subpath) = split_package_spec(module_name)?;
    let mut dir = start_dir.to_path_buf();
    loop {
        let nm = dir.join("node_modules");
        if nm.is_dir() {
            let pkg_dir = nm.join(&pkg);
            if pkg_dir.is_dir() {
                let manifest = read_package_json(&pkg_dir);
                let name = manifest
                    .as_ref()
                    .and_then(|v| v.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or(&pkg);
                let label = match manifest
                    .as_ref()
                    .and_then(|v| v.get("version"))
                    .and_then(|v| v.as_str())
                {
                    Some(ver) => format!("{name}@{ver}"),
                    None => name.to_string(),
                };
                if subpath != "." {
                    let exported: Vec<&str> = manifest
                        .as_ref()
                        .and_then(|v| v.get("exports"))
                        .and_then(|v| v.as_object())
                        .map(|m| m.keys().map(|k| k.as_str()).take(8).collect())
                        .unwrap_or_default();
                    let hint = if exported.is_empty() {
                        " (the package publishes no 'exports' map)".to_string()
                    } else {
                        format!(" — published subpaths: {}", exported.join(", "))
                    };
                    return Some(format!("package '{label}' does not provide '{subpath}'{hint}"));
                }
                return Some(format!(
                    "package '{label}' has no resolvable entry point (no matching 'exports', \
                     'module', 'main' or index.js)"
                ));
            }
        }
        let Some(parent) = dir.parent() else { break };
        if parent == dir {
            break;
        }
        dir = parent.to_path_buf();
    }
    Some(format!("package '{pkg}' is not installed (run `sofuu add {pkg}`)"))
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
    let decoder = flate2::read::GzDecoder::new(data);
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

    // ── PLAN-MODERN-NPM pillar A: conditional exports ──────────────

    /// Build a temp project: `pkg_json` at node_modules/<pkg>/package.json,
    /// plus each (rel, body) file under the same package.
    fn exports_fixture(tag: &str, pkg: &str, pkg_json: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("sofuu-npm-exp-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("node_modules").join(pkg);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("package.json"), pkg_json).unwrap();
        for (rel, body) in files {
            let p = dir.join(rel);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(p, body).unwrap();
        }
        root
    }

    fn resolved(root: &Path, spec: &str) -> Option<String> {
        npm_resolve(root, spec).map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
    }

    #[test]
    fn exports_exact_subpath_wins_over_main() {
        let root = exports_fixture(
            "exact",
            "pkg",
            r#"{"main":"legacy.js","exports":{".":"./dist/index.js","./feature":"./dist/feature.mjs"}}"#,
            &[("legacy.js", "1"), ("dist/index.js", "2"), ("dist/feature.mjs", "3")],
        );
        assert_eq!(resolved(&root, "pkg").as_deref(), Some("index.js"));
        assert_eq!(resolved(&root, "pkg/feature").as_deref(), Some("feature.mjs"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn exports_wildcard_substitutes_the_matched_segment() {
        let root = exports_fixture(
            "wild",
            "pkg",
            r#"{"exports":{"./locales/*":"./dist/locales/*.js"}}"#,
            &[("dist/locales/fr.js", "1"), ("dist/locales/de/extra.js", "2")],
        );
        assert_eq!(resolved(&root, "pkg/locales/fr").as_deref(), Some("fr.js"));
        /* The `*` may span slashes, per spec. */
        assert_eq!(resolved(&root, "pkg/locales/de/extra").as_deref(), Some("extra.js"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn exports_conditions_prefer_import_then_module_then_default_then_node() {
        let root = exports_fixture(
            "cond",
            "pkg",
            r#"{"exports":{".":{"node":"./n.js","default":"./d.js","module":"./m.js","import":"./i.js"}}}"#,
            &[("n.js", "1"), ("d.js", "2"), ("m.js", "3"), ("i.js", "4")],
        );
        assert_eq!(resolved(&root, "pkg").as_deref(), Some("i.js"));
        let _ = std::fs::remove_dir_all(&root);

        /* Without "import" the bundler key wins next. */
        let root = exports_fixture(
            "cond2",
            "pkg",
            r#"{"exports":{".":{"node":"./n.js","default":"./d.js","module":"./m.js"}}}"#,
            &[("n.js", "1"), ("d.js", "2"), ("m.js", "3")],
        );
        assert_eq!(resolved(&root, "pkg").as_deref(), Some("m.js"));
        let _ = std::fs::remove_dir_all(&root);

        /* ...then "default", and "node" only when nothing else matches. */
        let root = exports_fixture(
            "cond3",
            "pkg",
            r#"{"exports":{".":{"node":"./n.js","default":"./d.js"}}}"#,
            &[("n.js", "1"), ("d.js", "2")],
        );
        assert_eq!(resolved(&root, "pkg").as_deref(), Some("d.js"));
        let _ = std::fs::remove_dir_all(&root);

        let root = exports_fixture(
            "cond4",
            "pkg",
            r#"{"exports":{".":{"node":"./n.js"}}}"#,
            &[("n.js", "1")],
        );
        assert_eq!(resolved(&root, "pkg").as_deref(), Some("n.js"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn exports_condition_value_can_be_an_array_or_nested() {
        /* The first existing alternative wins. */
        let root = exports_fixture(
            "arr",
            "pkg",
            r#"{"exports":{".":{"import":["./missing.js","./real.js"],"default":"./d.js"}}}"#,
            &[("real.js", "1"), ("d.js", "2")],
        );
        assert_eq!(resolved(&root, "pkg").as_deref(), Some("real.js"));
        let _ = std::fs::remove_dir_all(&root);

        /* Nested conditions: import → (module → default). */
        let root = exports_fixture(
            "nested",
            "pkg",
            r#"{"exports":{".":{"import":{"module":"./m.js","default":"./i.js"},"default":"./d.js"}}}"#,
            &[("m.js", "1"), ("i.js", "2"), ("d.js", "3")],
        );
        assert_eq!(resolved(&root, "pkg").as_deref(), Some("m.js"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn exports_string_form_maps_only_the_root() {
        let root = exports_fixture(
            "str",
            "pkg",
            r#"{"exports":"./entry.mjs"}"#,
            &[("entry.mjs", "1"), ("extra.js", "2")],
        );
        assert_eq!(resolved(&root, "pkg").as_deref(), Some("entry.mjs"));
        /* "./extra" is not exported; the deep-import fallback still finds
         * the file the package ships. */
        assert_eq!(resolved(&root, "pkg/extra").as_deref(), Some("extra.js"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn exports_target_cannot_escape_the_package() {
        let root = exports_fixture(
            "escape",
            "pkg",
            r#"{"exports":{"./evil":"../outside.js"}}"#,
            &[],
        );
        /* Create the file the target would reach, so only the guard can
         * stop it. */
        std::fs::write(root.join("node_modules/outside.js"), "x").unwrap();
        assert_eq!(npm_resolve(&root, "pkg/evil"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn legacy_module_beats_main_and_index_remains_the_floor() {
        let root = exports_fixture(
            "legacy",
            "pkg",
            r#"{"main":"cjs.js","module":"esm.mjs"}"#,
            &[("cjs.js", "1"), ("esm.mjs", "2"), ("index.js", "3")],
        );
        assert_eq!(resolved(&root, "pkg").as_deref(), Some("esm.mjs"));
        let _ = std::fs::remove_dir_all(&root);

        let root = exports_fixture("floor", "pkg", r#"{"name":"pkg"}"#, &[("index.js", "1")]);
        assert_eq!(resolved(&root, "pkg").as_deref(), Some("index.js"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn require_conditions_pick_the_cjs_build() {
        let root = exports_fixture(
            "cjscond",
            "dual",
            r#"{"exports":{".":{"require":"./dist/index.cjs","import":"./dist/index.mjs"}}}"#,
            &[("dist/index.cjs", "1"), ("dist/index.mjs", "2")],
        );
        /* ESM import takes the .mjs build... */
        assert_eq!(resolved(&root, "dual").as_deref(), Some("index.mjs"));
        /* ...and require() takes the .cjs one. */
        let cjs = npm_resolve_cjs(&root, "dual").map(|p| p.file_name().unwrap().to_string_lossy().into_owned());
        assert_eq!(cjs.as_deref(), Some("index.cjs"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn scoped_package_subpath_resolves() {
        let root = exports_fixture(
            "scoped",
            "@scope/pkg",
            r#"{"exports":{".":{"import":"./index.mjs"},"./util":"./src/util.js"}}"#,
            &[("index.mjs", "1"), ("src/util.js", "2")],
        );
        assert_eq!(resolved(&root, "@scope/pkg").as_deref(), Some("index.mjs"));
        assert_eq!(resolved(&root, "@scope/pkg/util").as_deref(), Some("util.js"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn specifier_split_matches_the_spec() {
        assert_eq!(
            split_package_spec("pkg"),
            Some(("pkg".to_string(), ".".to_string()))
        );
        assert_eq!(
            split_package_spec("pkg/a/b"),
            Some(("pkg".to_string(), "./a/b".to_string()))
        );
        assert_eq!(
            split_package_spec("@s/p"),
            Some(("@s/p".to_string(), ".".to_string()))
        );
        assert_eq!(
            split_package_spec("@s/p/sub"),
            Some(("@s/p".to_string(), "./sub".to_string()))
        );
        assert_eq!(split_package_spec("./rel.js"), None);
        assert_eq!(split_package_spec("/abs.js"), None);
        assert_eq!(split_package_spec("#internal"), None);
    }

    #[test]
    fn module_not_found_explanations_name_the_real_reason() {
        let root = exports_fixture(
            "explain",
            "pkg",
            r#"{"name":"pkg","version":"2.1.0","exports":{".":"./index.js","./feature":"./f.js"}}"#,
            &[("index.js", "1"), ("f.js", "2")],
        );
        let missing_sub = explain_module_not_found(&root, "pkg/nope").unwrap();
        assert!(missing_sub.contains("pkg@2.1.0"), "{missing_sub}");
        assert!(missing_sub.contains("./nope"), "{missing_sub}");
        assert!(missing_sub.contains("./feature"), "{missing_sub}");

        let not_installed = explain_module_not_found(&root, "ghost").unwrap();
        assert!(not_installed.contains("not installed"), "{not_installed}");

        /* A package dir with no entry point at all. */
        let bare = exports_fixture("bare", "pkg", r#"{"name":"bare-pkg"}"#, &[]);
        let no_entry = explain_module_not_found(&bare, "pkg").unwrap();
        assert!(no_entry.contains("no resolvable entry point"), "{no_entry}");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&bare);
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
