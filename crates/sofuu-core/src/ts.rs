// sofuu-core — TypeScript type stripper.
//
// Rust port of src/ts/stripper.c — replaces type-only syntax with spaces
// while preserving newlines (so line numbers stay accurate for error
// messages). String/comment/template aware, handles interfaces, type
// aliases, type imports/exports, declare, access modifiers, `as`/`satisfies`
// casts, type annotations, generics, non-null assertions, optional params.
//
// Safe Rust: operates on Vec<u8>/String with explicit indices; no raw
// pointers, no manual realloc.

/// Strip TypeScript type syntax from `src`, returning JS source of the
/// same length (types replaced by spaces, newlines preserved).
pub fn strip(src: &str) -> String {
    let bytes: Vec<u8> = src.as_bytes().to_vec();
    let len = bytes.len();
    let mut out = bytes.clone();
    let mut pos = 0usize;
    let mut last_sig = false;

    let is_id_start = |c: u8| c.is_ascii_alphabetic() || c == b'_' || c == b'$';
    let is_id_cont = |c: u8| is_id_start(c) || c.is_ascii_digit();

    let mods = ["public", "private", "protected", "readonly", "abstract", "override"];

    while pos < len {
        let c = bytes[pos];

        // ── Comments ──
        if c == b'/' && pos + 1 < len && bytes[pos + 1] == b'/' {
            pos = skip_line_comment(&bytes, pos, len);
            continue;
        }
        if c == b'/' && pos + 1 < len && bytes[pos + 1] == b'*' {
            pos = skip_block_comment(&bytes, pos, len);
            continue;
        }
        // ── String / template literals ──
        if c == b'\'' || c == b'"' {
            pos = skip_string(&bytes, pos, len);
            last_sig = true;
            continue;
        }
        if c == b'`' {
            pos = skip_template(&bytes, pos, len);
            last_sig = true;
            continue;
        }
        // ── Whitespace ──
        if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
            pos += 1;
            continue;
        }

        // ── interface Foo<T> extends Bar { ... } ──
        if kw(&bytes, pos, len, b"interface") {
            let start = pos;
            pos += 9;
            pos = skip_ws(&bytes, pos, len);
            skip_ident(&bytes, &mut pos, len);
            pos = skip_ws(&bytes, pos, len);
            if pos < len && bytes[pos] == b'<' {
                pos = skip_balanced(&bytes, pos, len, b'<', b'>');
            }
            pos = skip_ws(&bytes, pos, len);
            if kw(&bytes, pos, len, b"extends") {
                pos += 7;
                while pos < len && bytes[pos] != b'{' {
                    if bytes[pos] == b'<' {
                        pos = skip_balanced(&bytes, pos, len, b'<', b'>');
                    } else {
                        pos += 1;
                    }
                }
            }
            if pos < len && bytes[pos] == b'{' {
                pos = skip_balanced(&bytes, pos, len, b'{', b'}');
            }
            blank(&mut out, start, pos);
            last_sig = false;
            continue;
        }

        // ── type Foo<T> = ... ; ──
        if kw(&bytes, pos, len, b"type") {
            let mut q = pos + 4;
            while q < len && (bytes[q] == b' ' || bytes[q] == b'\t') {
                q += 1;
            }
            if q < len && is_id_start(bytes[q]) {
                let start = pos;
                pos = q;
                skip_ident(&bytes, &mut pos, len);
                pos = skip_ws(&bytes, pos, len);
                if pos < len && bytes[pos] == b'<' {
                    pos = skip_balanced(&bytes, pos, len, b'<', b'>');
                }
                pos = skip_ws(&bytes, pos, len);
                if pos < len && bytes[pos] == b'=' {
                    pos += 1;
                    let mut d = 0i32;
                    while pos < len {
                        let cc = bytes[pos];
                        if cc == b'{' || cc == b'(' || cc == b'[' {
                            d += 1;
                            pos += 1;
                        } else if cc == b'}' || cc == b')' || cc == b']' {
                            if d == 0 {
                                break;
                            }
                            d -= 1;
                            pos += 1;
                        } else if d == 0 && cc == b';' {
                            pos += 1;
                            break;
                        } else if d == 0 && cc == b'\n' {
                            break;
                        } else {
                            pos += 1;
                        }
                    }
                }
                blank(&mut out, start, pos);
                last_sig = false;
                continue;
            }
        }

        // ── import type { ... } from '...' ──
        if kw(&bytes, pos, len, b"import") {
            let mut q = pos + 6;
            while q < len && (bytes[q] == b' ' || bytes[q] == b'\t') {
                q += 1;
            }
            if q + 4 <= len && &bytes[q..q + 4] == b"type" && !is_id_cont(bytes[q + 4]) {
                let mut peek = q + 4;
                while peek < len && (bytes[peek] == b' ' || bytes[peek] == b'\t') {
                    peek += 1;
                }
                if peek >= len || bytes[peek] != b'(' {
                    // not import(type) — erase whole statement
                    let start = pos;
                    pos = q + 4;
                    while pos < len && bytes[pos] != b';' && bytes[pos] != b'\n' {
                        if bytes[pos] == b'\'' || bytes[pos] == b'"' {
                            pos = skip_string(&bytes, pos, len);
                        } else if bytes[pos] == b'{' {
                            pos = skip_balanced(&bytes, pos, len, b'{', b'}');
                        } else {
                            pos += 1;
                        }
                    }
                    if pos < len && bytes[pos] == b';' {
                        pos += 1;
                    }
                    blank(&mut out, start, pos);
                    last_sig = false;
                    continue;
                }
            }
        }

        // ── export type { ... } → blank just "type" ──
        if kw(&bytes, pos, len, b"export") {
            let mut q = pos + 6;
            while q < len && (bytes[q] == b' ' || bytes[q] == b'\t') {
                q += 1;
            }
            if q + 4 <= len && &bytes[q..q + 4] == b"type" && !is_id_cont(bytes[q + 4]) {
                blank(&mut out, q, q + 4);
                pos += 6;
                last_sig = false;
                continue;
            }
        }

        // ── declare ... ──
        if kw(&bytes, pos, len, b"declare") {
            let start = pos;
            pos += 7;
            pos = skip_ws(&bytes, pos, len);
            let mut d = 0i32;
            while pos < len {
                let cc = bytes[pos];
                if cc == b'{' || cc == b'(' || cc == b'[' {
                    d += 1;
                    pos += 1;
                } else if cc == b'}' || cc == b')' || cc == b']' {
                    if d == 0 {
                        break;
                    }
                    d -= 1;
                    pos += 1;
                } else if d == 0 && cc == b';' {
                    pos += 1;
                    break;
                } else {
                    pos += 1;
                }
            }
            blank(&mut out, start, pos);
            last_sig = false;
            continue;
        }

        // ── access modifiers ──
        {
            let mut hit = false;
            for m in &mods {
                if kw(&bytes, pos, len, m.as_bytes()) {
                    let ml = m.len();
                    let mut q = pos + ml;
                    while q < len && (bytes[q] == b' ' || bytes[q] == b'\t') {
                        q += 1;
                    }
                    if q < len
                        && (is_id_start(bytes[q])
                            || bytes[q] == b'#'
                            || bytes[q] == b'['
                            || bytes[q] == b'*'
                            || bytes[q] == b'(')
                    {
                        blank(&mut out, pos, pos + ml);
                        pos += ml;
                        hit = true;
                        break;
                    }
                }
            }
            if hit {
                continue;
            }
        }

        // ── as Type / satisfies Type ──
        if kw(&bytes, pos, len, b"as") && last_sig {
            let start = pos;
            pos += 2;
            pos = skip_ws(&bytes, pos, len);
            skip_type(&bytes, &mut pos, len);
            blank(&mut out, start, pos);
            continue;
        }
        if kw(&bytes, pos, len, b"satisfies") && last_sig {
            let start = pos;
            pos += 9;
            pos = skip_ws(&bytes, pos, len);
            skip_type(&bytes, &mut pos, len);
            blank(&mut out, start, pos);
            continue;
        }

        // ── : TypeAnnotation ──
        if c == b':' && last_sig {
            let mut q = pos + 1;
            while q < len && (bytes[q] == b' ' || bytes[q] == b'\t') {
                q += 1;
            }
            let nc = if q < len { bytes[q] } else { 0 };
            if is_id_start(nc)
                || nc == b'('
                || nc == b'{'
                || nc == b'['
                || nc == b'|'
                || nc == b'&'
            {
                let start = pos;
                pos = q;
                skip_type(&bytes, &mut pos, len);
                blank(&mut out, start, pos);
                last_sig = false;
                continue;
            }
        }

        // ── <TypeParams> on functions/classes ──
        if c == b'<' && last_sig {
            let start = pos;
            let saved = pos;
            pos = skip_balanced(&bytes, pos, len, b'<', b'>');
            let mut q = pos;
            while q < len && (bytes[q] == b' ' || bytes[q] == b'\t') {
                q += 1;
            }
            if q < len && (bytes[q] == b'(' || bytes[q] == b'{' || bytes[q] == b',') {
                blank(&mut out, start, pos);
                last_sig = false;
                continue;
            }
            pos = saved; // not a generic — fall through
        }

        // ── ! non-null assertion ──
        if c == b'!' && last_sig && pos + 1 < len && bytes[pos + 1] != b'=' {
            blank(&mut out, pos, pos + 1);
            pos += 1;
            continue;
        }

        // ── ?: optional param ──
        if c == b'?' && last_sig && pos + 1 < len && bytes[pos + 1] == b':' {
            blank(&mut out, pos, pos + 1);
            pos += 1;
            continue;
        }

        // ── Default: advance, update last_sig ──
        if is_id_start(c) {
            skip_ident(&bytes, &mut pos, len);
            last_sig = true;
        } else if c.is_ascii_digit() {
            while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'.') {
                pos += 1;
            }
            last_sig = true;
        } else if c == b')' || c == b']' || c == b'}' {
            last_sig = true;
            pos += 1;
        } else if c == b'.' && pos + 2 < len && bytes[pos + 1] == b'.' && bytes[pos + 2] == b'.' {
            last_sig = false;
            pos += 3;
        } else {
            last_sig = false;
            pos += 1;
        }
    }

    // Convert back to String (all bytes valid — we only blanked with spaces).
    String::from_utf8(out).unwrap_or_else(|_| src.to_string())
}

// ── Helpers ─────────────────────────────────────────────────────

fn blank(out: &mut [u8], from: usize, to: usize) {
    let end = to.min(out.len());
    for b in &mut out[from..end] {
        if *b != b'\n' && *b != b'\r' {
            *b = b' ';
        }
    }
}

fn kw(bytes: &[u8], pos: usize, len: usize, w: &[u8]) -> bool {
    if pos + w.len() > len {
        return false;
    }
    if &bytes[pos..pos + w.len()] != w {
        return false;
    }
    let id_cont = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'$';
    if pos > 0 && id_cont(bytes[pos - 1]) {
        return false;
    }
    if pos + w.len() < len && id_cont(bytes[pos + w.len()]) {
        return false;
    }
    true
}

fn skip_ws(bytes: &[u8], pos: usize, len: usize) -> usize {
    let mut p = pos;
    while p < len && (bytes[p] == b' ' || bytes[p] == b'\t') {
        p += 1;
    }
    p
}

fn skip_ident(bytes: &[u8], pos: &mut usize, len: usize) {
    let id_cont = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'$';
    while *pos < len && id_cont(bytes[*pos]) {
        *pos += 1;
    }
}

fn skip_string(bytes: &[u8], pos: usize, len: usize) -> usize {
    let q = bytes[pos];
    let mut p = pos + 1;
    while p < len && bytes[p] != q {
        if bytes[p] == b'\\' {
            p += 1;
        }
        p += 1;
    }
    if p < len {
        p += 1;
    }
    p
}

fn skip_template(bytes: &[u8], pos: usize, len: usize) -> usize {
    let mut p = pos + 1;
    while p < len && bytes[p] != b'`' {
        if bytes[p] == b'\\' {
            p += 2;
            continue;
        }
        if bytes[p] == b'$' && p + 1 < len && bytes[p + 1] == b'{' {
            p += 2;
            let mut d = 1i32;
            while p < len && d > 0 {
                if bytes[p] == b'{' {
                    d += 1;
                } else if bytes[p] == b'}' {
                    d -= 1;
                }
                p += 1;
            }
            continue;
        }
        p += 1;
    }
    if p < len {
        p += 1;
    }
    p
}

fn skip_line_comment(bytes: &[u8], pos: usize, len: usize) -> usize {
    let mut p = pos;
    while p < len && bytes[p] != b'\n' {
        p += 1;
    }
    p
}

fn skip_block_comment(bytes: &[u8], pos: usize, len: usize) -> usize {
    let mut p = pos + 2;
    while p + 1 < len && !(bytes[p] == b'*' && bytes[p + 1] == b'/') {
        p += 1;
    }
    if p + 1 < len {
        p += 2;
    }
    p
}

/// Skip balanced brackets, handling nested strings/comments.
fn skip_balanced(bytes: &[u8], pos: usize, len: usize, open: u8, close: u8) -> usize {
    if pos >= len || bytes[pos] != open {
        return pos;
    }
    let mut p = pos + 1;
    let mut d = 1i32;
    while p < len && d > 0 {
        let c = bytes[p];
        if c == b'\'' || c == b'"' {
            p = skip_string(bytes, p, len);
            continue;
        }
        if c == b'`' {
            p = skip_template(bytes, p, len);
            continue;
        }
        if c == b'/' && p + 1 < len {
            if bytes[p + 1] == b'/' {
                p = skip_line_comment(bytes, p, len);
                continue;
            }
            if bytes[p + 1] == b'*' {
                p = skip_block_comment(bytes, p, len);
                continue;
            }
        }
        if c == open {
            d += 1;
        } else if c == close {
            d -= 1;
        }
        p += 1;
    }
    p
}

/// Advance past a TS type expression. Stops at depth-0 `, ; ) ] } =`.
fn skip_type(bytes: &[u8], pos: &mut usize, len: usize) {
    let mut d = 0i32;
    while *pos < len {
        let c = bytes[*pos];
        if c == b'(' || c == b'[' || c == b'<' {
            d += 1;
            *pos += 1;
            continue;
        }
        if c == b'{' {
            if d == 0 {
                break;
            }
            d += 1;
            *pos += 1;
            continue;
        }
        if c == b')' || c == b']' || c == b'}' || c == b'>' {
            if d == 0 {
                break;
            }
            d -= 1;
            *pos += 1;
            continue;
        }
        if d == 0 && (c == b',' || c == b';' || c == b'=') {
            break;
        }
        if d == 0 && c == b'\n' {
            let mut q = *pos + 1;
            while q < len && (bytes[q] == b' ' || bytes[q] == b'\t') {
                q += 1;
            }
            if q < len && (bytes[q] == b'|' || bytes[q] == b'&') {
                *pos = q;
                continue;
            }
            break;
        }
        if c == b'\'' || c == b'"' {
            *pos = skip_string(bytes, *pos, len);
            continue;
        }
        if c == b'`' {
            *pos = skip_template(bytes, *pos, len);
            continue;
        }
        *pos += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_type_annotation() {
        // `: number ` (including the trailing space before `=`) is blanked.
        let js = strip("const x: number = 5;");
        assert_eq!(js, "const x         = 5;");
    }

    #[test]
    fn preserves_newlines_and_length() {
        let src = "const a: number = 1;\nconst b: string = \"hi\";\n";
        let js = strip(src);
        assert_eq!(js.len(), src.len());
        assert_eq!(js.matches('\n').count(), 2);
    }

    #[test]
    fn strips_interface() {
        let src = "interface Foo { x: number; }\nconst f = 1;";
        let js = strip(src);
        assert!(!js.contains("interface"));
        assert!(js.contains("const f = 1;"));
    }

    #[test]
    fn strips_type_alias() {
        let js = strip("type ID = string | number;\nlet id: ID = 1;");
        assert!(!js.contains("type ID"));
        assert!(js.contains("let id"));
    }

    #[test]
    fn strips_type_import() {
        let js = strip("import type { Foo } from './foo';\nimport { Bar } from './bar';");
        assert!(!js.contains("import type"));
        assert!(js.contains("import { Bar } from './bar';"));
    }

    #[test]
    fn keeps_value_import() {
        let js = strip("import { x } from './m';\n");
        assert!(js.contains("import { x } from './m';"));
    }

    #[test]
    fn strips_as_cast() {
        let js = strip("const x = y as number;");
        assert_eq!(js, "const x = y          ;");
    }

    #[test]
    fn strips_access_modifiers() {
        let js = strip("class A { public x = 1; private y = 2; }");
        assert!(!js.contains("public"));
        assert!(!js.contains("private"));
        assert!(js.contains("class A {"));
    }

    #[test]
    fn keeps_strings_and_comments() {
        let src = "const s = \"a: b\"; // type: not real\nconst t = 1;";
        let js = strip(src);
        assert!(js.contains("\"a: b\""));
        assert!(js.contains("// type: not real"));
        assert!(js.contains("const t = 1;"));
    }

    #[test]
    fn strips_generics_on_function() {
        // <T> blanked, and `: T` (incl. space) blanked after the params.
        let js = strip("function id<T>(x: T): T { return x; }");
        assert_eq!(js, "function id   (x   )    { return x; }");
    }

    #[test]
    fn strips_optional_and_nonnull() {
        let js = strip("function f(x?: number) { return x!; }");
        assert!(js.contains("x?") == false); // ? erased
        assert!(!js.contains("x!"));
    }

    #[test]
    fn strips_declare() {
        let js = strip("declare const g: number;");
        assert!(!js.contains("declare"));
    }

    #[test]
    fn handles_template_literals() {
        let js = strip("const t = `x: ${y}`; const z: number = 1;");
        assert!(js.contains("`x: ${y}`"));
        assert!(js.contains("const z"));
    }
}
