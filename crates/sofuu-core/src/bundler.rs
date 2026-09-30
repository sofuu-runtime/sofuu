// sofuu-core — ESM bundler.
//
// Rust port of src/bundler/bundler.c — resolves an ESM import graph from an
// entry file and emits a single self-contained JS bundle using a
// `__d(id, factory)` / `__r(id)` registry. Handles:
//   - import default/named/namespace/side-effect/dynamic
//   - export default / named / star / star-as-namespace / re-export
//     (`export * [as ns] from '…'` and `export { … } from '…'` — the
//     re-export target also joins the import graph)
//   - export const/let/var/function/class
//   - TypeScript source (via crate::ts::strip)
//
// Safe Rust: String-based, no manual buffers, no strtok.

use std::path::{Path, PathBuf};

use crate::ts;

// ── §1 Import specifier extraction ──────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    pub spec: String,
    pub is_dynamic: bool,
}

fn is_id_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || c == b'$'
}
fn is_id_cont(c: u8) -> bool {
    is_id_start(c) || c.is_ascii_digit()
}

fn skip_string(src: &[u8], mut pos: usize, len: usize) -> usize {
    let q = src[pos];
    pos += 1;
    while pos < len && src[pos] != q {
        if src[pos] == b'\\' {
            pos += 1;
        }
        pos += 1;
    }
    if pos < len {
        pos += 1;
    }
    pos
}

fn skip_template(src: &[u8], mut pos: usize, len: usize) -> usize {
    pos += 1;
    while pos < len && src[pos] != b'`' {
        if src[pos] == b'\\' {
            pos += 2;
            continue;
        }
        pos += 1;
    }
    if pos < len {
        pos += 1;
    }
    pos
}

fn skip_line_comment(src: &[u8], mut pos: usize, len: usize) -> usize {
    while pos < len && src[pos] != b'\n' {
        pos += 1;
    }
    pos
}

fn skip_block_comment(src: &[u8], mut pos: usize, len: usize) -> usize {
    pos += 2;
    while pos + 1 < len && !(src[pos] == b'*' && src[pos + 1] == b'/') {
        pos += 1;
    }
    if pos + 1 < len {
        pos += 2;
    }
    pos
}

fn skip_ws(src: &[u8], mut pos: usize, len: usize) -> usize {
    while pos < len && (src[pos] == b' ' || src[pos] == b'\t') {
        pos += 1;
    }
    pos
}

fn skip_wsn(src: &[u8], mut pos: usize, len: usize) -> usize {
    while pos < len && (src[pos] as char).is_whitespace() {
        pos += 1;
    }
    pos
}

/* proc-12: raw byte scans for `;`/newline walked straight through string
 * literals, so `export const s = "a;b"` truncated the module at the `;`
 * inside the string (dropping every later export). stmt_end finds the
 * first statement terminator OUTSIDE any literal (skip_string/skip_template
 * above) or line comment. */
fn stmt_end(src: &[u8], mut pos: usize, len: usize) -> usize {
    while pos < len {
        let c = src[pos];
        if c == b'"' || c == b'\'' {
            pos = skip_string(src, pos, len);
            continue;
        }
        if c == b'`' {
            pos = skip_template(src, pos, len);
            continue;
        }
        if c == b'/' && pos + 1 < len && src[pos + 1] == b'/' {
            while pos < len && src[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        if c == b';' || c == b'\n' {
            return pos;
        }
        pos += 1;
    }
    len
}

/// Read a quoted string at `pos` (on the quote). Returns the inner string.
fn read_quoted(src: &[u8], pos: &mut usize, len: usize) -> Option<String> {
    *pos = skip_wsn(src, *pos, len);
    if *pos >= len {
        return None;
    }
    let q = src[*pos];
    if q != b'"' && q != b'\'' {
        return None;
    }
    *pos += 1;
    let start = *pos;
    while *pos < len && src[*pos] != q {
        if src[*pos] == b'\\' {
            *pos += 1;
        }
        *pos += 1;
    }
    let s = String::from_utf8_lossy(&src[start..*pos]).into_owned();
    if *pos < len {
        *pos += 1;
    }
    Some(s)
}

/// After `import`, scan for the specifier; `end_pos` = end of statement.
fn parse_import_specifier(
    src: &[u8],
    mut pos: usize,
    len: usize,
    end_pos: &mut usize,
    is_dynamic: &mut bool,
) -> Option<String> {
    *is_dynamic = false;
    pos = skip_ws(src, pos, len);

    // Dynamic import: import('...')
    if pos < len && src[pos] == b'(' {
        *is_dynamic = true;
        pos += 1;
        pos = skip_wsn(src, pos, len);
        let spec = read_quoted(src, &mut pos, len);
        pos = skip_wsn(src, pos, len);
        if pos < len && src[pos] == b')' {
            pos += 1;
        }
        /* proc-12: string-aware statement end (dynamic import specifier
         * could contain a quoted `;`). */
        pos = stmt_end(src, pos, len);
        if pos < len && src[pos] == b';' {
            pos += 1;
        }
        *end_pos = pos;
        return spec;
    }

    // Bare import: import 'specifier'
    if pos < len && (src[pos] == b'"' || src[pos] == b'\'') {
        let spec = read_quoted(src, &mut pos, len);
        /* proc-12: string-aware statement end. */
        pos = stmt_end(src, pos, len);
        if pos < len && src[pos] == b';' {
            pos += 1;
        }
        *end_pos = pos;
        return spec;
    }

    // Bindings — scan for `from` at depth 0.
    let mut depth = 0i32;
    while pos < len {
        let c = src[pos];
        if c == b'"' || c == b'\'' {
            pos = skip_string(src, pos, len);
            continue;
        }
        if c == b'`' {
            pos = skip_template(src, pos, len);
            continue;
        }
        if c == b'/' && pos + 1 < len && src[pos + 1] == b'/' {
            pos = skip_line_comment(src, pos, len);
            continue;
        }
        if c == b'/' && pos + 1 < len && src[pos + 1] == b'*' {
            pos = skip_block_comment(src, pos, len);
            continue;
        }
        if c == b'{' || c == b'(' {
            depth += 1;
            pos += 1;
            continue;
        }
        if c == b'}' || c == b')' {
            depth -= 1;
            pos += 1;
            continue;
        }
        if depth == 0
            && c == b'f'
            && pos + 4 <= len
            && &src[pos..pos + 4] == b"from"
            && (pos + 4 >= len || !is_id_cont(src[pos + 4]))
        {
            pos += 4;
            pos = skip_wsn(src, pos, len);
            let spec = read_quoted(src, &mut pos, len);
            pos = skip_ws(src, pos, len);
            /* proc-12: the specifier itself was read; a trailing `;` here
             * is directly adjacent — safe either way, kept explicit. */
            if pos < len && src[pos] == b';' {
                pos += 1;
            }
            *end_pos = pos;
            return spec;
        }
        pos += 1;
    }
    *end_pos = pos;
    None
}

/// After `export`, extract a re-export specifier if the statement is one of
/// `export * from`, `export * as ns from`, or `export { … } from 'spec'`.
/// Returns None for plain exports (they can't pull in a new module).
fn parse_reexport_specifier(src: &[u8], pos: usize, len: usize) -> Option<String> {
    let mut p = skip_wsn(src, pos, len);

    // `export * [as ns] from 'spec'`
    if p < len && src[p] == b'*' {
        p = skip_wsn(src, p + 1, len);
        // Optional `as ns`.
        if p + 2 < len && &src[p..p + 2] == b"as" && !is_id_cont(src[p + 2]) {
            p = skip_wsn(src, p + 2, len);
            while p < len && is_id_cont(src[p]) {
                p += 1;
            }
            p = skip_wsn(src, p, len);
        }
        if p + 4 <= len && &src[p..p + 4] == b"from" && (p + 4 >= len || !is_id_cont(src[p + 4])) {
            p = skip_wsn(src, p + 4, len);
            return read_quoted(src, &mut p, len);
        }
        return None;
    }

    // `export { a, b as c } from 'spec'` — find the closing brace, then `from`.
    if p < len && src[p] == b'{' {
        let mut depth = 0i32;
        while p < len {
            let c = src[p];
            if c == b'"' || c == b'\'' {
                p = skip_string(src, p, len);
                continue;
            }
            if c == b'`' {
                p = skip_template(src, p, len);
                continue;
            }
            if c == b'{' {
                depth += 1;
            } else if c == b'}' {
                depth -= 1;
                if depth == 0 {
                    p += 1;
                    break;
                }
            }
            p += 1;
        }
        p = skip_wsn(src, p, len);
        if p + 4 <= len && &src[p..p + 4] == b"from" && (p + 4 >= len || !is_id_cont(src[p + 4])) {
            p = skip_wsn(src, p + 4, len);
            return read_quoted(src, &mut p, len);
        }
    }
    None
}

/// Collect all import specifiers from a source file.
pub fn collect_specifiers(src: &str) -> Vec<Spec> {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut out = Vec::new();
    let mut p = 0usize;
    while p < len {
        let c = bytes[p];
        if c == b'"' || c == b'\'' {
            p = skip_string(bytes, p, len);
            continue;
        }
        if c == b'`' {
            p = skip_template(bytes, p, len);
            continue;
        }
        if c == b'/' && p + 1 < len && bytes[p + 1] == b'/' {
            p = skip_line_comment(bytes, p, len);
            continue;
        }
        if c == b'/' && p + 1 < len && bytes[p + 1] == b'*' {
            p = skip_block_comment(bytes, p, len);
            continue;
        }
        if c == b'i'
            && p + 6 <= len
            && &bytes[p..p + 6] == b"import"
            && (p == 0 || !is_id_cont(bytes[p - 1]))
            && (p + 6 >= len || !is_id_cont(bytes[p + 6]))
        {
            let mut end = p;
            let mut is_dyn = false;
            if let Some(spec) = parse_import_specifier(bytes, p + 6, len, &mut end, &mut is_dyn) {
                out.push(Spec {
                    spec,
                    is_dynamic: is_dyn,
                });
                p = end;
                continue;
            }
        }
        /* P1-13 (AUDIT-2026-09-07): re-export targets must join the graph —
         * `export * as ns from './dep'` (and the plain-star / brace forms)
         * transform into a require() of the target, so a module reachable
         * ONLY through a re-export used to miss from the bundle ("missing
         * module" stub at runtime). Collected as a static (non-dynamic) spec. */
        if c == b'e'
            && p + 6 <= len
            && &bytes[p..p + 6] == b"export"
            && (p == 0 || !is_id_cont(bytes[p - 1]))
            && (p + 6 >= len || !is_id_cont(bytes[p + 6]))
        {
            if let Some(spec) = parse_reexport_specifier(bytes, p + 6, len) {
                out.push(Spec {
                    spec,
                    is_dynamic: false,
                });
                // Advance past the statement so the `from` clause's module-id
                // text can't be rescanned (it holds no further statements).
                /* proc-12: string-aware — `export { x } from './a;b'` must
                 * not have its scan stopped inside the specifier. */
                p = stmt_end(bytes, p, len);
                if p < len && bytes[p] == b';' {
                    p += 1;
                }
                continue;
            }
        }
        p += 1;
    }
    out
}

// ── §2 Path utilities ───────────────────────────────────────────

fn path_is_relative(spec: &str) -> bool {
    let b = spec.as_bytes();
    b.first() == Some(&b'.') && (b.get(1) == Some(&b'/') || (b.get(1) == Some(&b'.') && b.get(2) == Some(&b'/')))
}

fn is_reg_file(p: &Path) -> bool {
    std::fs::metadata(p).map(|m| m.is_file()).unwrap_or(false)
}

/// Try common extensions on a path without one.
fn try_extensions(abs: &Path) -> Option<PathBuf> {
    if is_reg_file(abs) {
        return Some(abs.to_path_buf());
    }
    let base = abs.to_string_lossy().into_owned();
    for ext in [".js", ".mjs", ".ts", "/index.js", "/index.ts"] {
        let cand = PathBuf::from(format!("{base}{ext}"));
        if is_reg_file(&cand) {
            return Some(cand);
        }
    }
    None
}

/// Resolve a specifier to an absolute path + bundle id.
fn resolve_specifier(
    spec: &str,
    mod_dir: &Path,
    root_dir: &Path,
    npm_resolve: &dyn Fn(&Path, &str) -> Option<PathBuf>,
) -> Option<(PathBuf, String)> {
    if path_is_relative(spec) {
        let joined = mod_dir.join(spec);
        let abs = try_extensions(&joined)?;
        let id = if let Ok(rel) = abs.strip_prefix(root_dir) {
            format!("./{}", rel.to_string_lossy().replace('\\', "/"))
        } else {
            abs.to_string_lossy().into_owned()
        };
        Some((abs, id))
    } else {
        // npm module — resolved via the resolver callback.
        let resolved = npm_resolve(mod_dir, spec)?;
        /* P2-13 (AUDIT-2026-09-01): the id used to be the RAW specifier —
         * `pkg` vs `pkg/index.js` (and any subpath forms that resolve to
         * the same file) minted TWO nodes → duplicate module instances and
         * broken `instanceof`. Key the id on the RESOLVED path so every
         * specifier that resolves here shares one module. */
        let base = resolved
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| spec.to_string());
        let parent = resolved
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let id = format!("node_modules/{}/{}", parent.replace('\\', "/"), base);
        Some((resolved, id))
    }
}

// ── §3 Module graph ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BundleMod {
    pub abs_path: PathBuf,
    pub id: String,
    pub source: String,
    pub transformed: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub mods: Vec<BundleMod>,
    pub root_dir: PathBuf,
    pub entry_id: String,
}

/// Build the import graph from an entry file, BFS.
pub fn build_graph(
    entry_abs: &Path,
    npm_resolve: &dyn Fn(&Path, &str) -> Option<PathBuf>,
) -> Result<Graph, String> {
    let entry_abs = entry_abs.canonicalize().map_err(|e| format!("cannot resolve entry: {e}"))?;
    let root_dir = entry_abs.parent().unwrap_or(Path::new(".")).to_path_buf();
    let entry_id = format!("./{}", entry_abs.file_name().unwrap_or_default().to_string_lossy());

    let mut graph = Graph {
        mods: Vec::new(),
        root_dir: root_dir.clone(),
        entry_id: entry_id.clone(),
    };
    graph.mods.push(BundleMod {
        abs_path: entry_abs,
        id: entry_id,
        source: String::new(),
        transformed: None,
    });

    let mut head = 0usize;
    while head < graph.mods.len() {
        let m = graph.mods[head].clone();
        head += 1;

        let source = std::fs::read_to_string(&m.abs_path)
            .map_err(|e| format!("cannot read {}: {e}", m.abs_path.display()))?;
        let idx = graph
            .mods
            .iter()
            .position(|x| x.abs_path == m.abs_path)
            .ok_or("graph index lost")?;
        graph.mods[idx].source = source.clone();

        // Strip TS if needed.
        let src = if m.abs_path.extension().map(|e| e == "ts").unwrap_or(false) {
            ts::strip(&source)
        } else {
            source.clone()
        };

        let specs = collect_specifiers(&src);
        let mod_dir = m.abs_path.parent().unwrap_or(Path::new(".")).to_path_buf();
        for spec in &specs {
            if let Some((dep_abs, dep_id)) =
                resolve_specifier(&spec.spec, &mod_dir, &root_dir, npm_resolve)
            {
                if !graph.mods.iter().any(|x| x.abs_path == dep_abs) {
                    graph.mods.push(BundleMod {
                        abs_path: dep_abs,
                        id: dep_id,
                        source: String::new(),
                        transformed: None,
                    });
                }
            }
        }
    }
    Ok(graph)
}

// ── §4 Transformer ──────────────────────────────────────────────

fn resolve_id_in_graph(
    graph: &Graph,
    mod_abs: &Path,
    spec: &str,
    npm_resolve: &dyn Fn(&Path, &str) -> Option<PathBuf>,
) -> String {
    let mod_dir = mod_abs.parent().unwrap_or(Path::new("."));
    if let Some((dep_abs, dep_id)) = resolve_specifier(spec, mod_dir, &graph.root_dir, npm_resolve) {
        if graph.mods.iter().any(|x| x.abs_path == dep_abs) {
            return dep_id;
        }
    }
    spec.to_string()
}

fn trim(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace())
}

/// Extract the binding text between `import` and `from`.
fn extract_binding(src: &[u8], start: usize, end: usize) -> String {
    let mut out = String::new();
    let mut p = start;
    let mut depth = 0i32;
    while p < end {
        let c = src[p];
        if c == b'"' || c == b'\'' {
            let se = skip_string(src, p, end);
            out.push_str(&String::from_utf8_lossy(&src[p..se]));
            p = se;
            continue;
        }
        if c == b'{' {
            depth += 1;
            out.push(c as char);
            p += 1;
            continue;
        }
        if c == b'}' {
            depth -= 1;
            out.push(c as char);
            p += 1;
            continue;
        }
        if depth == 0
            && c == b'f'
            && p + 4 <= end
            && &src[p..p + 4] == b"from"
            && (p + 4 >= end || !is_id_cont(src[p + 4]))
        {
            break;
        }
        out.push(c as char);
        p += 1;
    }
    trim(&out).to_string()
}

/// Transform one module's source into bundle-registry form.
pub fn transform_module(
    graph: &Graph,
    m: &BundleMod,
    src: &str,
    npm_resolve: &dyn Fn(&Path, &str) -> Option<PathBuf>,
) -> String {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut out = String::new();
    let mut p = 0usize;

    while p < len {
        let c = bytes[p];

        // Pass through strings/templates/comments verbatim.
        if c == b'"' || c == b'\'' {
            let end = skip_string(bytes, p, len);
            out.push_str(&src[p..end]);
            p = end;
            continue;
        }
        if c == b'`' {
            let end = skip_template(bytes, p, len);
            out.push_str(&src[p..end]);
            p = end;
            continue;
        }
        if c == b'/' && p + 1 < len && bytes[p + 1] == b'/' {
            let end = skip_line_comment(bytes, p, len);
            out.push_str(&src[p..end]);
            p = end;
            continue;
        }
        if c == b'/' && p + 1 < len && bytes[p + 1] == b'*' {
            let end = skip_block_comment(bytes, p, len);
            out.push_str(&src[p..end]);
            p = end;
            continue;
        }

        // ── import ──
        if c == b'i'
            && p + 6 <= len
            && &bytes[p..p + 6] == b"import"
            && (p == 0 || !is_id_cont(bytes[p - 1]))
            && (p + 6 >= len || !is_id_cont(bytes[p + 6]))
        {
            // `import type ...` → erase whole statement.
            let tp = skip_ws(bytes, p + 6, len);
            if tp + 4 <= len && &bytes[tp..tp + 4] == b"type" && !is_id_cont(bytes[tp + 4]) {
                let peek = skip_ws(bytes, tp + 4, len);
                if peek < len && bytes[peek] != b'(' {
                    /* proc-12: string-aware — `import type {A} from './a;b'`
                     * must erase the WHOLE statement, not stop at the `;`
                     * inside the specifier (which leaked `b';` into output). */
                    let mut end = stmt_end(bytes, peek, len);
                    if end < len && bytes[end] == b';' {
                        end += 1;
                    }
                    p = end;
                    continue;
                }
            }

            let mut end = p;
            let mut is_dyn = false;
            if let Some(spec) = parse_import_specifier(bytes, p + 6, len, &mut end, &mut is_dyn) {
                let id = resolve_id_in_graph(graph, &m.abs_path, &spec, npm_resolve);
                if is_dyn {
                    out.push_str(&format!("Promise.resolve(require('{id}'))"));
                } else {
                    let bp = skip_ws(bytes, p + 6, len);
                    if bp < len && (bytes[bp] == b'(' || bytes[bp] == b'"' || bytes[bp] == b'\'') {
                        out.push_str(&format!("require('{id}')"));
                    } else {
                        let bind = extract_binding(bytes, bp, end);
                        out.push_str(&emit_import(&bind, &id, p));
                    }
                }
                out.push(';');
                p = end;
                if p < len && bytes[p] == b'\n' {
                    out.push('\n');
                    p += 1;
                }
                continue;
            }
        }

        // ── export ──
        if c == b'e'
            && p + 6 <= len
            && &bytes[p..p + 6] == b"export"
            && (p == 0 || !is_id_cont(bytes[p - 1]))
            && (p + 6 < len && !is_id_cont(bytes[p + 6]))
        {
            let ep0 = skip_wsn(bytes, p + 6, len);

            // export default ...
            if ep0 + 7 <= len && &bytes[ep0..ep0 + 7] == b"default" && !is_id_cont(bytes[ep0 + 7]) {
                let mut ep = skip_ws(bytes, ep0 + 7, len);
                let mut expr = String::new();
                if ep + 8 <= len && &bytes[ep..ep + 8] == b"function"
                    || ep + 5 <= len && &bytes[ep..ep + 5] == b"class"
                {
                    let mut d = 0i32;
                    while ep < len {
                        if bytes[ep] == b'{' {
                            d += 1;
                            expr.push('{');
                            ep += 1;
                            continue;
                        }
                        if bytes[ep] == b'}' {
                            d -= 1;
                            expr.push('}');
                            ep += 1;
                            if d == 0 {
                                break;
                            }
                            continue;
                        }
                        expr.push(bytes[ep] as char);
                        ep += 1;
                    }
                } else if ep < len && bytes[ep] == b'{' {
                    /* P2-11 (AUDIT-2026-09-01): `export default {\n a: 1\n};`
                     * used to stop at the FIRST newline — emitting
                     * `exports.default=…={;` (a syntax error). A brace
                     * expression counts braces so multi-line object
                     * literals survive; strings/chars containing braces
                     * are NOT parsed here (matching the fn/class path's
                     * documented limitation). */
                    let mut d = 0i32;
                    while ep < len {
                        let b = bytes[ep];
                        if b == b'{' {
                            d += 1;
                            expr.push('{');
                            ep += 1;
                            continue;
                        }
                        if b == b'}' {
                            d -= 1;
                            expr.push('}');
                            ep += 1;
                            if d == 0 {
                                break;
                            }
                            continue;
                        }
                        expr.push(b as char);
                        ep += 1;
                    }
                    /* consume an optional trailing ';' */
                    if ep < len && bytes[ep] == b';' {
                        ep += 1;
                    }
                } else {
                    /* proc-12: string-aware — `export default "a;b"` must
                     * capture the WHOLE expression, not stop at the `;`
                     * inside the literal. */
                    let e = stmt_end(bytes, ep, len);
                    expr.push_str(&String::from_utf8_lossy(&bytes[ep..e]));
                    ep = e;
                    if ep < len && bytes[ep] == b';' {
                        ep += 1;
                    }
                }
                out.push_str(&format!("exports.default=exports.__default={expr};"));
                p = ep;
                if p < len && bytes[p] == b'\n' {
                    out.push('\n');
                    p += 1;
                }
                continue;
            }

            // export * from './x'  |  export * as ns from './x'
            if ep0 < len && bytes[ep0] == b'*' {
                let mut ep = skip_wsn(bytes, ep0 + 1, len);
                /* P1-13 (AUDIT-2026-09-07): the optional `as ns` clause used
                 * to be left in the output verbatim — `export` is illegal in
                 * the __d factory, so any dependency using star-as-namespace
                 * failed to load with a SyntaxError. Parse it and emit the
                 * namespace object onto exports. */
                let mut ns: Option<String> = None;
                if ep + 2 < len && &bytes[ep..ep + 2] == b"as" && !is_id_cont(bytes[ep + 2]) {
                    ep = skip_wsn(bytes, ep + 2, len);
                    let name_start = ep;
                    while ep < len && is_id_cont(bytes[ep]) {
                        ep += 1;
                    }
                    ns = Some(src[name_start..ep].to_string());
                    ep = skip_wsn(bytes, ep, len);
                }
                if ep + 4 <= len && &bytes[ep..ep + 4] == b"from" {
                    ep = skip_wsn(bytes, ep + 4, len);
                    if let Some(spec) = read_quoted(bytes, &mut ep, len) {
                        let id = resolve_id_in_graph(graph, &m.abs_path, &spec, npm_resolve);
                        match ns {
                            Some(name) => out.push_str(&format!(
                                "var __sx=require('{id}');exports.{name}=__sx;"
                            )),
                            None => out.push_str(&format!("Object.assign(exports,require('{id}'));")),
                        }
                        while ep < len && bytes[ep] != b'\n' {
                            ep += 1;
                        }
                        p = ep;
                        if p < len && bytes[p] == b'\n' {
                            out.push('\n');
                            p += 1;
                        }
                        continue;
                    }
                }
            }

            // export { a, b } [from './x']
            if ep0 < len && bytes[ep0] == b'{' {
                let mut ep = ep0 + 1;
                let mut names = String::new();
                while ep < len && bytes[ep] != b'}' {
                    names.push(bytes[ep] as char);
                    ep += 1;
                }
                if ep < len {
                    ep += 1;
                }
                ep = skip_ws(bytes, ep, len);
                let mut from_spec: Option<String> = None;
                if ep + 4 < len && &bytes[ep..ep + 4] == b"from" && !is_id_cont(bytes[ep + 4]) {
                    ep = skip_wsn(bytes, ep + 4, len);
                    from_spec = read_quoted(bytes, &mut ep, len);
                }
                /* proc-12: string-aware — `export { a } from './a;b'` must
                 * consume the whole statement. */
                let e = stmt_end(bytes, ep, len);
                ep = e;
                if ep < len && bytes[ep] == b';' {
                    ep += 1;
                }
                let id = match &from_spec {
                    Some(s) => resolve_id_in_graph(graph, &m.abs_path, s, npm_resolve),
                    None => String::new(),
                };
                out.push_str(&emit_named_exports(&names, &id));
                p = ep;
                if p < len && bytes[p] == b'\n' {
                    out.push('\n');
                    p += 1;
                }
                continue;
            }

            // export const/let/var X = ...
            if ep0 + 5 <= len
                && (&bytes[ep0..ep0 + 5] == b"const"
                    || &bytes[ep0..ep0 + 3] == b"let"
                    || &bytes[ep0..ep0 + 3] == b"var")
            {
                let kw_start = ep0;
                let mut ep = ep0;
                while ep < len && !bytes[ep].is_ascii_whitespace() {
                    ep += 1;
                }
                let kw = &src[kw_start..ep];
                ep = skip_ws(bytes, ep, len);
                let name_start = ep;
                while ep < len && is_id_cont(bytes[ep]) {
                    ep += 1;
                }
                let name = &src[name_start..ep];
                let mut rest = String::new();
                let mut d = 0i32;
                while ep < len {
                    /* proc-12: skip string literals FIRST — a `;` inside
                     * `"a;b"` (or a `{`/`}` in a string) must not touch the
                     * depth counter or end the initializer. */
                    let b0 = bytes[ep];
                    if b0 == b'"' || b0 == b'\'' {
                        let e = skip_string(bytes, ep, len);
                        rest.push_str(&String::from_utf8_lossy(&bytes[ep..e]));
                        ep = e;
                        continue;
                    }
                    if b0 == b'`' {
                        let e = skip_template(bytes, ep, len);
                        rest.push_str(&String::from_utf8_lossy(&bytes[ep..e]));
                        ep = e;
                        continue;
                    }
                    if bytes[ep] == b'{' || bytes[ep] == b'(' || bytes[ep] == b'[' {
                        d += 1;
                    }
                    if bytes[ep] == b'}' || bytes[ep] == b')' || bytes[ep] == b']' {
                        d -= 1;
                    }
                    if d == 0 && bytes[ep] == b';' {
                        rest.push(';');
                        ep += 1;
                        break;
                    }
                    if d < 0 {
                        break;
                    }
                    rest.push(bytes[ep] as char);
                    ep += 1;
                }
                out.push_str(kw);
                out.push(' ');
                out.push_str(name);
                out.push_str(if rest.is_empty() { ";" } else { &rest });
                out.push_str(&format!("\nexports.{name}={name};"));
                p = ep;
                if p < len && bytes[p] == b'\n' {
                    out.push('\n');
                    p += 1;
                }
                continue;
            }

            // export function foo() {} / export class Foo {}
            if ep0 + 8 <= len
                && (&bytes[ep0..ep0 + 8] == b"function" || &bytes[ep0..ep0 + 5] == b"class")
            {
                let kw_start = ep0;
                let mut ep = ep0;
                while ep < len && !bytes[ep].is_ascii_whitespace() && bytes[ep] != b'(' {
                    ep += 1;
                }
                ep = skip_ws(bytes, ep, len);
                let name_start = ep;
                while ep < len && is_id_cont(bytes[ep]) {
                    ep += 1;
                }
                let name = &src[name_start..ep];
                let mut d = 0i32;
                let mut in_block = false;
                while ep < len {
                    if bytes[ep] == b'{' {
                        d += 1;
                        in_block = true;
                    }
                    if bytes[ep] == b'}' {
                        d -= 1;
                        ep += 1;
                        if in_block && d == 0 {
                            break;
                        }
                        continue;
                    }
                    ep += 1;
                }
                out.push_str(&src[kw_start..ep]);
                out.push_str(&format!("\nexports.{name}={name};"));
                p = ep;
                if p < len && bytes[p] == b'\n' {
                    out.push('\n');
                    p += 1;
                }
                continue;
            }
        }

        // Default: copy verbatim.
        out.push(c as char);
        p += 1;
    }
    out
}

fn emit_import(bind: &str, id: &str, pos: usize) -> String {
    let mut out = String::new();
    if bind.is_empty() {
        return format!("require('{id}')");
    }
    let b = bind.as_bytes();
    if b[0] == b'*' {
        // import * as ns from '...'
        let ns = bind.split("as").nth(1).map(|s| trim(s).to_string()).unwrap_or_default();
        let name: String = ns.chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$').collect();
        return format!("var {name}=require('{id}')");
    }
    if b[0] == b'{' {
        // import { a, b as c } from '...'
        let inner = bind
            .trim_start_matches('{')
            .trim_end_matches('}')
            .split(',')
            .map(trim)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(", ");
        if !inner.contains(" as ") {
            return format!("var {{{inner}}}=require('{id}')");
        }
        let tmp = format!("__ri_{:04x}", pos & 0xffff);
        out.push_str(&format!("var {tmp}=require('{id}');"));
        for tok in inner.split(',') {
            let tok = trim(tok);
            if tok.is_empty() {
                continue;
            }
            if let Some((orig, alias)) = tok.split_once(" as ") {
                let orig = trim(orig);
                let alias = trim(alias);
                out.push_str(&format!("var {alias}={tmp}.{orig};"));
            } else {
                out.push_str(&format!("var {tok}={tmp}.{tok};"));
            }
        }
        return out;
    }
    // import Default from / import Default, { named } from
    if let Some((def_name, named)) = bind.split_once(',') {
        let def_name = trim(def_name);
        out.push_str(&format!(
            "var __r_{def_name}=require('{id}');var {def_name}=__r_{def_name}.default||__r_{def_name};"
        ));
        let named = trim(named);
        if let Some(inner) = named.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
            for tok in inner.split(',') {
                let tok = trim(tok);
                if tok.is_empty() {
                    continue;
                }
                if let Some((orig, alias)) = tok.split_once(" as ") {
                    let orig = trim(orig);
                    let alias = trim(alias);
                    out.push_str(&format!("var {alias}=__r_{def_name}.{orig};"));
                } else {
                    out.push_str(&format!("var {tok}=__r_{def_name}.{tok};"));
                }
            }
        } else if named.starts_with('*') {
            /* P2-12 (AUDIT-2026-09-01): `import Default, * as ns from '…'`
             * used to emit `var * as ns=…` — invalid JS. The `*` check was
             * only applied to b[0], so the mixed form fell through here. */
            let nsa = named.split("as").nth(1).map(|s| trim(s).to_string()).unwrap_or_default();
            let ns_name: String = nsa
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
                .collect();
            out.push_str(&format!("var {ns_name}=require('{id}');"));
        } else {
            out.push_str(&format!("var {named}=__r_{def_name}"));
        }
        return out;
    }
    format!("var __rd_=require('{id}');var {bind}=__rd_.default||__rd_")
}

fn emit_named_exports(names: &str, id: &str) -> String {
    let mut out = String::new();
    let has_from = !id.is_empty();
    if has_from {
        out.push_str(&format!("{{var __rx=require('{id}');"));
    }
    for tok in names.split(',') {
        let tok = trim(tok);
        if tok.is_empty() {
            continue;
        }
        if let Some((orig, alias)) = tok.split_once(" as ") {
            let orig = trim(orig);
            let alias = trim(alias);
            if alias == "default" {
                // P3: the old `if has_from { "" } else { "" }` was dead AND
                // masked a latent bug — without a from-clause there is no
                // `__rx`, so re-export the local name directly.
                if has_from {
                    out.push_str(&format!("exports.default=exports.__default=__rx.{orig};"));
                } else {
                    out.push_str(&format!("exports.default=exports.__default={orig};"));
                }
            } else if has_from {
                out.push_str(&format!("exports.{alias}=__rx.{orig};"));
            } else {
                out.push_str(&format!("exports.{alias}={orig};"));
            }
        } else if has_from {
            out.push_str(&format!("exports.{tok}=__rx.{tok};"));
        } else {
            out.push_str(&format!("exports.{tok}={tok};"));
        }
    }
    if has_from {
        out.push('}');
    }
    out
}

// ── §5 Emit ─────────────────────────────────────────────────────

pub const PREAMBLE: &str = "// ⚡ Generated by sofuu bundle — https://sofuu.dev\n\
// Run with: sofuu run <this_file>\n\
(function(){\n\
'use strict';\n\
var __m={};\n\
var __f={};\n\
function __d(id,factory){__f[id]=factory;}\n\
function __r(id){\n\
  if(Object.prototype.hasOwnProperty.call(__m,id))return __m[id];\n\
  var exp={};\n\
  __m[id]=exp;\n\
  if(!__f[id]){console.error('[sofuu bundle] missing module:',id);return exp;}\n\
  __f[id](exp,__r);\n\
  return __m[id];\n\
}\n";

/// Emit the full bundle string for a graph.
pub fn emit_bundle(graph: &Graph) -> String {
    let mut out = String::from(PREAMBLE);
    // Modules in reverse BFS order (deps before dependents).
    for m in graph.mods.iter().rev() {
        if let Some(t) = &m.transformed {
            out.push_str(&format!("\n/* === module: {} === */\n", m.id));
            out.push_str(&format!("__d('{}',function(exports,require){{\n", m.id));
            out.push_str(t);
            out.push_str("\n});\n");
        }
    }
    out.push_str(&format!("\n/* === entry === */\nvar __entry=__r('{}');\n", graph.entry_id));
    out.push_str("if(typeof __entry==='function')__entry();\n})();\n");
    out
}

/// Full bundle pipeline: build graph, transform all, emit.
pub fn bundle(
    entry: &Path,
    npm_resolve: &dyn Fn(&Path, &str) -> Option<PathBuf>,
) -> Result<String, String> {
    let mut graph = build_graph(entry, npm_resolve)?;
    for i in 0..graph.mods.len() {
        let src = if graph.mods[i].abs_path.extension().map(|e| e == "ts").unwrap_or(false) {
            ts::strip(&graph.mods[i].source)
        } else {
            graph.mods[i].source.clone()
        };
        let m = graph.mods[i].clone();
        let transformed = transform_module(&graph, &m, &src, npm_resolve);
        graph.mods[i].transformed = Some(transformed);
    }
    Ok(emit_bundle(&graph))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolver(_d: &Path, _s: &str) -> Option<PathBuf> {
        None // tests use relative imports only
    }

    #[test]
    fn collects_imports() {
        let src = "import a from './a';\nimport { b, c as d } from './b';\nimport * as ns from './ns';\nimport './side';\nconst x = import('./dyn');\n";
        let specs = collect_specifiers(src);
        assert_eq!(specs.len(), 5);
        assert_eq!(specs[0].spec, "./a");
        assert_eq!(specs[1].spec, "./b");
        assert_eq!(specs[3].spec, "./side");
        assert!(specs[4].is_dynamic);
    }

    #[test]
    fn ignores_imports_in_strings_and_comments() {
        let src = "const s = 'import x from \"./fake\"';\n// import y from './fake2'\n/* import z from './fake3' */\nconst t = 1;";
        assert!(collect_specifiers(src).is_empty());
    }

    #[test]
    fn transforms_default_import() {
        let g = Graph {
            mods: vec![],
            root_dir: PathBuf::from("."),
            entry_id: "./e.js".into(),
        };
        let m = BundleMod {
            abs_path: PathBuf::from("/e.js"),
            id: "./e.js".into(),
            source: String::new(),
            transformed: None,
        };
        let out = transform_module(&g, &m, "import Foo from './foo';\n", &resolver);
        assert!(out.contains("var __rd_=require('./foo');var Foo=__rd_.default||__rd_"));
    }

    #[test]
    fn transforms_named_import_no_rename() {
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let out = transform_module(&g, &m, "import { a, b } from './m';\n", &resolver);
        assert!(out.contains("var {a, b}=require('./m')"));
    }

    #[test]
    fn transforms_named_import_with_rename() {
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let out = transform_module(&g, &m, "import { a as x } from './m';\n", &resolver);
        assert!(out.contains("var x="));
        assert!(out.contains(".a;"));
    }

    #[test]
    fn transforms_export_default() {
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let out = transform_module(&g, &m, "export default 42;\n", &resolver);
        assert!(out.contains("exports.default=exports.__default=42"));
    }

    #[test]
    fn transforms_export_named_default_no_from() {
        // P3 (AUDIT-2026-09-07): `export { x as default }` with NO
        // from-clause used to emit `__rx.x` — a variable the bundle never
        // defines — because the `if has_from { "" } else { "" }` conditional
        // was dead on both arms.
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let out = transform_module(&g, &m, "function x(){return 1}\nexport { x as default };\n", &resolver);
        assert!(out.contains("exports.default=exports.__default=x;"), "no-from default re-exports the local, got: {out}");
        assert!(!out.contains("__rx"), "no __rx may be referenced without a from-clause, got: {out}");
    }

    #[test]
    fn transforms_export_named() {
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let out = transform_module(&g, &m, "const a = 1;\nexport { a };\n", &resolver);
        assert!(out.contains("exports.a=a;"));
    }

    #[test]
    fn transforms_export_const() {
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let out = transform_module(&g, &m, "export const x = 5;\n", &resolver);
        assert!(out.contains("const x = 5;"));
        assert!(out.contains("exports.x=x;"));
    }

    #[test]
    fn transforms_export_star() {
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let out = transform_module(&g, &m, "export * from './lib';\n", &resolver);
        assert!(out.contains("Object.assign(exports,require('./lib'))"));
    }

    /* P1-13 (AUDIT-2026-09-07): star-as-namespace used to be copied verbatim
     * into the __d factory — `export` is illegal in a function body, so any
     * dependency using it failed to load with a SyntaxError. */
    #[test]
    fn transforms_export_star_as_namespace() {
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let out = transform_module(&g, &m, "export * as ns from './dep';\n", &resolver);
        assert!(out.contains("var __sx=require('./dep');exports.ns=__sx;"), "got: {out}");
        assert!(!out.contains("export *"), "verbatim export leaked: {out}");
    }

    /* proc-12 (AUDIT-2026-09-07): the statement scanners stopped at the
     * FIRST `;` or newline even when it sat inside a string literal —
     * `export const s = "a;b";` emitted `const s = "a;` and the rest of
     * the module after it was parsed from inside an unterminated string. */
    #[test]
    fn semicolon_inside_string_does_not_truncate_export() {
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let src = "export const s = \"a;b\";\nexport function f() { return 2; }\n";
        let out = transform_module(&g, &m, src, &resolver);
        assert!(out.contains("\"a;b\""), "string literal must survive intact: {out}");
        assert!(out.contains("exports.s=s;"), "export const must still emit its export: {out}");
        assert!(
            out.contains("function f"),
            "the statement after the string-bearing one must survive: {out}"
        );
    }

    /* proc-12: same family — `import type ... from './a;b'` must erase
     * through the REAL terminator; the raw scan stopped inside the quoted
     * specifier and swallowed the next statement into a string. */
    #[test]
    fn semicolon_in_import_type_specifier_does_not_leak() {
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let src = "import type {A} from './a;b';\nexport default 1;\n";
        let out = transform_module(&g, &m, src, &resolver);
        assert!(
            out.contains("exports.default=exports.__default=1"),
            "the statement after the type import must transform: {out}"
        );
    }

    /* proc-12: stmt_end must skip literals AND line comments when hunting
     * for the terminator. */
    #[test]
    fn stmt_end_skips_strings_and_comments() {
        let src = b"let x = \"a;b\" // no; end\nlet y = 2;";
        let e = stmt_end(src, 4, src.len());
        assert_eq!(&src[4..e], b"x = \"a;b\" // no; end");
    }

    /* P1-13: re-export targets must join the module graph — a module reachable
     * only through `export * [as ns] from` used to miss from the bundle and hit
     * the "missing module" stub at runtime. Plain exports must NOT be collected. */
    #[test]
    fn collects_reexport_specifiers() {
        let src = "export * as ns from './a';\nexport * from './b';\nexport { c } from './c';\nexport { local };\nexport const q = 1;\nexport default 7;\n";
        let specs = collect_specifiers(src);
        let got: Vec<&str> = specs.iter().map(|s| s.spec.as_str()).collect();
        assert_eq!(got, vec!["./a", "./b", "./c"], "got: {got:?}");
        assert!(specs.iter().all(|s| !s.is_dynamic));
    }

    #[test]
    fn bundles_star_as_ns_reexport_chain() {
        let tmp = std::env::temp_dir().join(format!("sofuu-bundle-ns-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("dep.js"), "export const alpha = 1;\nexport const beta = 2;\nexport default 42;\n").unwrap();
        std::fs::write(tmp.join("extra.js"), "export const gamma = 3;\n").unwrap();
        std::fs::write(tmp.join("mid.js"), "export * as ns from './dep.js';\nexport * from './extra.js';\n").unwrap();
        std::fs::write(tmp.join("entry.js"), "import { ns } from './mid.js';\nconst t = ns;\n").unwrap();
        let out = bundle(&tmp.join("entry.js"), &resolver).unwrap();
        // Every module in the chain made it into the graph…
        assert!(out.contains("__d('./dep.js'"), "dep missing: {out}");
        assert!(out.contains("__d('./extra.js'"), "extra missing: {out}");
        assert!(out.contains("__d('./mid.js'"), "mid missing: {out}");
        // …the star-as transform emitted the namespace binding…
        assert!(out.contains("var __sx=require('./dep.js');exports.ns=__sx;"), "got: {out}");
        assert!(out.contains("Object.assign(exports,require('./extra.js'))"));
        // …and no illegal verbatim `export` statement survived.
        assert!(!out.contains("export *"), "verbatim export leaked: {out}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn transform_type_import_erased() {
        let g = Graph { mods: vec![], root_dir: PathBuf::from("."), entry_id: "./e.js".into() };
        let m = BundleMod { abs_path: PathBuf::from("/e.js"), id: "./e.js".into(), source: String::new(), transformed: None };
        let out = transform_module(&g, &m, "import type { T } from './types';\nconst x = 1;\n", &resolver);
        assert!(!out.contains("require('./types')"));
        assert!(out.contains("const x = 1;"));
    }

    #[test]
    fn emit_has_preamble_and_entry() {
        let g = Graph {
            mods: vec![BundleMod {
                abs_path: PathBuf::from("/e.js"),
                id: "./e.js".into(),
                source: "export default 1;".into(),
                transformed: Some("exports.default=exports.__default=1;".into()),
            }],
            root_dir: PathBuf::from("/"),
            entry_id: "./e.js".into(),
        };
        let out = emit_bundle(&g);
        assert!(out.contains("var __m={};"));
        assert!(out.contains("function __d(id,factory)"));
        assert!(out.contains("__d('./e.js',function(exports,require)"));
        assert!(out.contains("var __entry=__r('./e.js');"));
    }
}
