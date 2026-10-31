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
    /* Enums carry runtime values, so they cannot be blanked like a type.
     * Rewrite them to frozen object literals first (the same transform the
     * plugin driver's `_stripTs` performs), then run the length-preserving
     * scanner over the result. A source with no enum is returned verbatim. */
    /* Enums and constructor parameter properties both carry runtime
     * semantics a type-stripper would otherwise drop, so they are rewritten
     * into plain JS first; the length-preserving scan below then only has to
     * blank real type syntax. */
    let rewritten = rewrite_ctor_param_props(&rewrite_enums(src));
    let bytes: Vec<u8> = rewritten.as_bytes().to_vec();
    let len = bytes.len();
    let mut out = bytes.clone();
    let mut pos = 0usize;
    let mut last_sig = false;
    // Brace/paren context for the `:` rule (see in_object_key_position).
    let mut paren_depth: usize = 0;
    // (is_object_literal, paren_depth_at_open, preceding significant byte)
    let mut brace_ctx: Vec<(bool, usize, u8)> = Vec::new();
    // The last closed brace, so `({ a }: T)` can still be annotated.
    let mut last_closed: Option<(bool, usize, u8)> = None;
    /* What the previous significant token was: only an identifier (`x: T`) or
     * a closing paren (`(): T`) can be followed by a type annotation. Without
     * this, the colon of a ternary whose branches are object literals
     * (`c ? {a: 1} : {b: 2}`) was read as an annotation and deleted.
     * AUDIT-2026-10-31. */
    const KIND_OTHER: u8 = 0;
    const KIND_IDENT: u8 = 1;
    const KIND_RPAREN: u8 = 2;
    const KIND_RBRACKET: u8 = 3;
    const KIND_RBRACE: u8 = 4;
    let mut last_kind: u8 = KIND_OTHER;

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
                /* AUDIT-2026-10-31 P1-1: `export type Foo = ...` and
                 * `export interface I { ... }` had their declaration blanked
                 * and left a bare `export` behind ("invalid export syntax").
                 * A type-only RE-export (`export type { X }`) still needs its
                 * `export`, so only the declaration form drops the keyword. */
                let mut r = q + 4;
                while r < len && (bytes[r] == b' ' || bytes[r] == b'\t') {
                    r += 1;
                }
                if r < len && bytes[r] != b'{' {
                    blank(&mut out, pos, r);
                } else {
                    blank(&mut out, q, q + 4);
                }
                pos = q + 4;
                last_sig = false;
                continue;
            }
            // `export interface` / `export declare`: the declaration goes away,
            // so the keyword must go with it.
            let mut dropped = false;
            for kwlen in [9usize, 7usize] {
                if q + kwlen <= len
                    && (&bytes[q..q + kwlen] == b"interface" || &bytes[q..q + kwlen] == b"declare")
                    && !is_id_cont(bytes[q + kwlen])
                {
                    blank(&mut out, pos, q);
                    pos = q;
                    last_sig = false;
                    dropped = true;
                    break;
                }
            }
            if dropped {
                continue;
            }
            // Anything else (`export const`, `export function`, `export default`)
            // is real JavaScript: leave it alone.
        }

        // ── class C implements A, B { ... } ──
        /* AUDIT-2026-10-31 P1-1: `implements` clauses were rejected outright.
         * They are erased TypeScript — nothing at runtime needs them. */
        if kw(&bytes, pos, len, b"implements") {
            let start = pos;
            pos += 10;
            while pos < len && bytes[pos] != b'{' {
                if bytes[pos] == b'<' {
                    pos = skip_balanced(&bytes, pos, len, b'<', b'>');
                } else {
                    pos += 1;
                }
            }
            blank(&mut out, start, pos);
            last_sig = false;
            continue;
        }

        // ── abstract class ──
        if kw(&bytes, pos, len, b"abstract") {
            let mut q = pos + 8;
            while q < len && (bytes[q] == b' ' || bytes[q] == b'\t') {
                q += 1;
            }
            if kw(&bytes, q, len, b"class") {
                blank(&mut out, pos, q);
                pos = q;
                last_sig = false;
                continue;
            }
            /* An abstract MEMBER (`abstract m(): void;`) has no body, so
             * blanking just the keyword would leave `m();` — not a valid class
             * member. The whole declaration is erased by design. */
            let mut r = pos;
            while r < len && bytes[r] != b';' && bytes[r] != b'}' {
                if bytes[r] == b'(' || bytes[r] == b'{' {
                    r = skip_balanced(&bytes, r, len, bytes[r], if bytes[r] == b'(' { b')' } else { b'}' });
                } else {
                    r += 1;
                }
            }
            if r < len && bytes[r] == b';' {
                blank(&mut out, pos, r + 1);
                pos = r + 1;
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

        // ── Brace / paren context (the `:` rule depends on it) ──
        /* AUDIT-2026-10-31 P0: `{ k: x }` used to be read as "key: <type>" and
         * the value was blanked, so every object literal whose value is a bare
         * identifier compiled to `{ k }` — a ReferenceError at runtime, in the
         * one construct JS code uses most. Record which braces open an object
         * literal (or a destructuring pattern), because that is the only place
         * a `key: value` pair can appear. */
        if c == b'(' {
            paren_depth += 1;
            last_sig = false;
            last_kind = KIND_OTHER;
            pos += 1;
            continue;
        }
        if c == b')' {
            paren_depth = paren_depth.saturating_sub(1);
            last_sig = true;
            last_kind = KIND_RPAREN;
            pos += 1;
            continue;
        }
        if c == b'{' {
            brace_ctx.push((
                opens_object_literal(&bytes, pos),
                paren_depth,
                prev_sig_byte(&bytes, pos),
            ));
            last_sig = false;
            last_kind = KIND_OTHER;
            pos += 1;
            continue;
        }
        if c == b'}' {
            last_closed = brace_ctx.pop();
            last_sig = true;
            last_kind = KIND_RBRACE;
            pos += 1;
            continue;
        }

        // ── : TypeAnnotation ──
        /* A `:` may follow a destructuring pattern in a parameter list —
         * `function C({ a, b }: Props)` — which ends in `}`, but a ternary
         * branch also ends in `}`. They are told apart by what opened the
         * brace: `(` or `,` means a pattern, `?` or `:` means a ternary. */
        let after_pattern = match last_closed {
            Some((true, depth, p)) => depth == paren_depth && (p == b'(' || p == b','),
            _ => false,
        };
        if c == b':'
            && last_sig
            && (last_kind == KIND_IDENT || last_kind == KIND_RPAREN || after_pattern)
            && !in_object_key_position(&brace_ctx, paren_depth)
        {
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

        // ── <TypeParams> on functions/classes/arrow functions ──
        /* AUDIT-2026-10-31 P1-1: `const f = <T>(v: T) => v` was left as-is and
         * failed with a syntax error, because the rule required a preceding
         * value. After `=`/`(`/`,`/`return` a `<` can only start type
         * parameters — `a < b > (c)` never has one of those immediately before
         * the `<`. */
        if c == b'<' && (last_sig || prev_is_expression_start(&bytes, pos)) {
            let start = pos;
            let saved = pos;
            /* AUDIT-2026-10-31: a comparison such as `a < b > (c)` used to be
             * blanked as if it were type parameters, deleting real code. A
             * generic application is written tight (`f<T>(x)`); a comparison is
             * not. When the preceding value is an identifier, require `<` to
             * touch it AND the closing `>` to touch the `(`. */
            let tight_before = pos > 0 && !matches!(bytes[pos - 1], b' ' | b'\t' | b'\r' | b'\n');
            pos = skip_balanced(&bytes, pos, len, b'<', b'>');
            let tight_after = pos < len && (bytes[pos] == b'(' || bytes[pos] == b'{');
            let mut q = pos;
            while q < len && (bytes[q] == b' ' || bytes[q] == b'\t') {
                q += 1;
            }
            let followed_by_call = q < len && (bytes[q] == b'(' || bytes[q] == b'{' || bytes[q] == b',');
            if followed_by_call && tight_after && (tight_before || !last_sig) {
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
            last_kind = KIND_IDENT;
        } else if c.is_ascii_digit() {
            while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'.') {
                pos += 1;
            }
            last_sig = true;
            last_kind = KIND_IDENT;
        } else if c == b']' {
            last_sig = true;
            last_kind = KIND_RBRACKET;
            pos += 1;
        } else if c == b'.' && pos + 2 < len && bytes[pos + 1] == b'.' && bytes[pos + 2] == b'.' {
            last_sig = false;
            last_kind = KIND_OTHER;
            pos += 3;
        } else {
            last_sig = false;
            last_kind = KIND_OTHER;
            pos += 1;
        }
    }

    // Convert back to String (all bytes valid — we only blanked with spaces).
    String::from_utf8(out).unwrap_or_else(|_| src.to_string())
}


/// True when the token immediately before `pos` can only begin an expression
/// (`=`, `(`, `[`, `,`, `:`, `?`, `return` …) — the positions where `<T>`
/// starts type parameters rather than a comparison.
fn prev_is_expression_start(b: &[u8], pos: usize) -> bool {
    let mut p = pos;
    loop {
        if p == 0 {
            return true;
        }
        p -= 1;
        match b[p] {
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            _ => break,
        }
    }
    match b[p] {
        b'=' | b'(' | b'[' | b',' | b':' | b'?' | b'{' | b';' | b'+' | b'-' | b'*' | b'/'
        | b'%' | b'!' | b'&' | b'|' | b'^' | b'~' => true,
        c if c.is_ascii_alphanumeric() || c == b'_' || c == b'$' => {
            let mut q = p + 1;
            while q > 0 && (b[q - 1].is_ascii_alphanumeric() || b[q - 1] == b'_' || b[q - 1] == b'$') {
                q -= 1;
            }
            matches!(&b[q..=p], b"return" | b"typeof" | b"await" | b"yield" | b"new" | b"in" | b"of")
        }
        _ => true,
    }
}

/// True when a `:` here separates an object-literal (or destructuring-pattern)
/// key from its value, where stripping would delete the value. A `:` inside a
/// nested paren list is a real annotation — a method's or arrow's parameters —
/// so the paren depth at which the brace was opened has to match.
fn in_object_key_position(brace_ctx: &[(bool, usize, u8)], paren_depth: usize) -> bool {
    match brace_ctx.last() {
        Some(&(is_object, depth, _)) => is_object && depth == paren_depth,
        None => false,
    }
}

/// The previous significant byte, skipping whitespace and block comments.
fn prev_sig_byte(b: &[u8], pos: usize) -> u8 {
    let mut p = pos;
    loop {
        if p == 0 {
            return 0;
        }
        p -= 1;
        match b[p] {
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            b'/' if p > 0 && b[p - 1] == b'*' => {
                let mut q = p - 1;
                while q > 1 && !(b[q - 1] == b'/' && b[q] == b'*') {
                    q -= 1;
                }
                p = q.saturating_sub(1);
                continue;
            }
            c => return c,
        }
    }
}

/// True when the `{` at `pos` opens an object literal or a destructuring
/// pattern rather than a block or a class/interface/type body.
///
/// The two are syntactically identical, so the preceding significant token
/// decides: after an operator, `=`, `(`, `[`, `,`, `:`, `?` or a keyword that
/// begins an expression, a brace can only be an object. When the context cannot
/// be established the answer is `true` — treating it as an object suppresses
/// stripping, so the worst case is a loud SyntaxError rather than a silently
/// deleted value.
fn opens_object_literal(b: &[u8], pos: usize) -> bool {
    let mut p = pos;
    loop {
        if p == 0 {
            // `{` at the very start of a file is a block.
            return false;
        }
        p -= 1;
        match b[p] {
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            b'/' if p > 0 && b[p - 1] == b'*' => {
                // End of a block comment: rewind past its opening `/*`.
                let mut q = p - 1;
                while q > 1 && !(b[q - 1] == b'/' && b[q] == b'*') {
                    q -= 1;
                }
                p = q.saturating_sub(1);
                continue;
            }
            _ => break,
        }
    }
    match b[p] {
        b'=' | b'(' | b'[' | b',' | b':' | b'?' | b'+' | b'-' | b'*' | b'/' | b'%' | b'!'
        | b'&' | b'|' | b'^' | b'~' | b'<' => true,
        b')' | b']' | b'}' | b';' | b'{' | b'>' => false,
        c if c.is_ascii_alphanumeric() || c == b'_' || c == b'$' => {
            let mut q = p + 1;
            while q > 0 && (b[q - 1].is_ascii_alphanumeric() || b[q - 1] == b'_' || b[q - 1] == b'$') {
                q -= 1;
            }
            matches!(
                &b[q..=p],
                b"return"
                    | b"const"
                    | b"let"
                    | b"var"
                    | b"typeof"
                    | b"new"
                    | b"delete"
                    | b"void"
                    | b"throw"
                    | b"yield"
                    | b"await"
                    | b"in"
                    | b"of"
                    | b"case"
                    | b"default"
            )
        }
        _ => true,
    }
}

// ── Constructor parameter properties ───────────────────────────

const PARAM_PROP_MODS: [&[u8]; 5] = [b"public", b"private", b"protected", b"readonly", b"override"];

/// Rewrite TypeScript *parameter properties* into explicit assignments.
///
/// `constructor(public a: number, private readonly b = 2) { … }` is sugar: the
/// modifier both declares a field and assigns it. A stripper that only blanks
/// the modifier leaves `constructor(a, b) { … }` with no `this.a = a`, so every
/// read of `this.a` is wrong (undefined, or NaN for a numeric field) while the
/// program still runs — the one TS construct that miscompiled silently instead
/// of raising. Rewriting the source before the length-preserving scan restores
/// the semantics TS users expect. (AUDIT-2026-10-31 P0.)
fn rewrite_ctor_param_props(src: &str) -> String {
    let b = src.as_bytes();
    let len = b.len();
    let mut out = String::with_capacity(len + 32);
    let mut last = 0usize;
    let mut i = 0usize;

    while i < len {
        // Never rewrite inside a string, template or comment.
        if b[i] == b'/' && i + 1 < len && b[i + 1] == b'/' {
            i = skip_line_comment(b, i, len);
            continue;
        }
        if b[i] == b'/' && i + 1 < len && b[i + 1] == b'*' {
            i = skip_block_comment(b, i, len);
            continue;
        }
        if b[i] == b'\'' || b[i] == b'"' {
            i = skip_string(b, i, len);
            continue;
        }
        if b[i] == b'`' {
            i = skip_template(b, i, len);
            continue;
        }
        if !kw(b, i, len, b"constructor") {
            i += 1;
            continue;
        }
        let mut p = skip_ws(b, i + 11, len);
        if p >= len || b[p] != b'(' {
            i += 1;
            continue;
        }
        let open = p;
        let close = skip_balanced(b, open, len, b'(', b')');
        if close <= open + 1 || close > len {
            i += 1;
            continue;
        }
        let Some((deletes, assigns)) = scan_ctor_params(src, open + 1, close - 1) else {
            i += 1;
            continue;
        };
        if deletes.is_empty() {
            i += 1;
            continue;
        }
        /* A constructor cannot carry a return type, so the body opens right
         * after the parameter list. An overload signature has no body: leave
         * it untouched and let the implementation below be rewritten. */
        let body = skip_ws(b, close, len);
        if body >= len || b[body] != b'{' {
            i += 1;
            continue;
        }
        let mut q = last;
        for (s, e) in &deletes {
            if *s >= q && *e <= body + 1 {
                out.push_str(&src[q..*s]);
                q = *e;
            }
        }
        out.push_str(&src[q..body + 1]);
        out.push_str(&assigns);
        last = body + 1;
        i = body + 1;
    }

    out.push_str(&src[last..]);
    out
}

/// Collect the modifier byte-ranges of a constructor parameter list and build
/// the `this.x = x;` statements to inject. `None` means "shape not understood"
/// (a decorator or an unmodified destructuring parameter), in which case the
/// caller leaves the source exactly as it found it.
fn scan_ctor_params(src: &str, start: usize, end: usize) -> Option<(Vec<(usize, usize)>, String)> {
    let b = src.as_bytes();
    let len = b.len();
    let mut deletes: Vec<(usize, usize)> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut i = start;

    while i < end {
        while i < end && matches!(b[i], b' ' | b'\t' | b'\r' | b'\n' | b',') {
            i += 1;
        }
        if i >= end {
            break;
        }
        if b[i] == b'@' {
            return None;
        }
        let mut saw_mod = false;
        'mods: loop {
            for m in PARAM_PROP_MODS {
                let l = m.len();
                if i + l <= end && &b[i..i + l] == m {
                    let after = i + l;
                    let boundary = after >= len
                        || !(b[after].is_ascii_alphanumeric() || b[after] == b'_' || b[after] == b'$');
                    if boundary {
                        deletes.push((i, after));
                        i = after;
                        saw_mod = true;
                        while i < end && matches!(b[i], b' ' | b'\t' | b'\r' | b'\n') {
                            i += 1;
                        }
                        continue 'mods;
                    }
                }
            }
            break;
        }
        if saw_mod {
            let name_start = i;
            let mut j = i;
            while j < end && (b[j].is_ascii_alphanumeric() || b[j] == b'_' || b[j] == b'$') {
                j += 1;
            }
            // A modifier with no plain identifier after it (destructuring,
            // a computed name) is not something we can desugar safely.
            if j == name_start {
                return None;
            }
            names.push(src[name_start..j].to_string());
            i = j;
        }
        // Skip the rest of this parameter, honouring nesting.
        let mut depth = 0i32;
        while i < end {
            match b[i] {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                b',' if depth == 0 => break,
                b'\'' | b'"' => {
                    i = skip_string(b, i, len);
                    continue;
                }
                b'`' => {
                    i = skip_template(b, i, len);
                    continue;
                }
                _ => {}
            }
            i += 1;
        }
    }

    let mut assigns = String::new();
    for n in &names {
        assigns.push_str("this.");
        assigns.push_str(n);
        assigns.push_str(" = ");
        assigns.push_str(n);
        assigns.push_str("; ");
    }
    Some((deletes, assigns))
}

// ── Enum rewriting ──────────────────────────────────────────────

/// `enum Name { A, B = 2, C }` → `var Name = Object.freeze({"A":0,"B":2,"C":3});`
/// plus one newline per removed line. Returns the source unchanged when a
/// member is not a plain identifier with a numeric or string literal value —
/// QuickJS then reports a syntax error rather than the runtime silently
/// binding the wrong value. `declare enum` (type-only) is erased.
fn rewrite_enums(src: &str) -> String {
    let b = src.as_bytes();
    let len = b.len();
    let mut out = String::with_capacity(len);
    let mut last = 0usize;
    let mut i = 0usize;

    while i < len {
        if !kw(b, i, len, b"enum") {
            i += 1;
            continue;
        }
        let Some(decl_start) = stmt_prefix_start(b, i) else {
            i += 1;
            continue;
        };

        let mut q = skip_ws(b, i + 4, len);
        let name_start = q;
        skip_ident(b, &mut q, len);
        if q == name_start {
            i += 1;
            continue;
        }
        let name = &src[name_start..q];
        q = skip_ws(b, q, len);
        if q >= len || b[q] != b'{' {
            i += 1;
            continue;
        }
        let close = skip_balanced(b, q, len, b'{', b'}');
        if close <= q + 1 || close > len {
            i += 1;
            continue;
        }
        let body = &src[q + 1..close - 1];

        let Some(replacement) = enum_to_object(name, body) else {
            i += 1;
            continue;
        };

        let is_declare = {
            let mut p = decl_start;
            let mut declare = false;
            while p < i {
                if kw(b, p, len, b"declare") {
                    declare = true;
                }
                p += 1;
            }
            declare
        };

        out.push_str(&src[last..decl_start]);
        if !is_declare {
            out.push_str(&replacement);
        }
        for _ in src[decl_start..close].bytes().filter(|c| *c == b'\n') {
            out.push('\n');
        }
        last = close;
        i = close;
    }

    out.push_str(&src[last..]);
    out
}

/// The start of the statement that declares an `enum` at `enum_pos`, allowing
/// `export` / `const` / `declare` prefixes. `None` when the keyword is not at
/// a statement boundary (inside a string, a comment, or a member access).
fn stmt_prefix_start(b: &[u8], enum_pos: usize) -> Option<usize> {
    let mut p = enum_pos;
    loop {
        while p > 0 && matches!(b[p - 1], b' ' | b'\t' | b'\n' | b'\r') {
            p -= 1;
        }
        let mut matched = false;
        for kwd in [b"export".as_slice(), b"const".as_slice(), b"declare".as_slice()] {
            let l = kwd.len();
            if p >= l
                && &b[p - l..p] == kwd
                && (p == l || !(b[p - l - 1].is_ascii_alphanumeric() || b[p - l - 1] == b'_' || b[p - l - 1] == b'$'))
            {
                p -= l;
                matched = true;
                break;
            }
        }
        if !matched {
            break;
        }
    }
    if p == 0 {
        return Some(0);
    }
    match b[p - 1] {
        b';' | b'{' | b'}' | b')' => Some(p),
        _ => None,
    }
}

fn is_plain_ident(s: &str) -> bool {
    let mut it = s.bytes();
    match it.next() {
        Some(c) if c.is_ascii_alphabetic() || c == b'_' || c == b'$' => {}
        _ => return false,
    }
    it.all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'$')
}

/// Split an enum body on top-level commas (quotes/brackets aware).
fn split_members(body: &str) -> Vec<&str> {
    let b = body.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut depth = 0i32;
    while i < b.len() {
        match b[i] {
            b'"' | b'\'' => {
                let q = b[i];
                i += 1;
                while i < b.len() && b[i] != q {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'{' | b'(' | b'[' => depth += 1,
            b'}' | b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                out.push(&body[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    out.push(&body[start..]);
    out
}

/// One literal member value as JSON text (number or string).
fn member_literal(raw: &str) -> Option<String> {
    let v = raw.trim();
    if v.is_empty() {
        return None;
    }
    let first = v.as_bytes()[0];
    if first == b'"' || first == b'\'' {
        if v.len() < 2 || v.as_bytes()[v.len() - 1] != first {
            return None;
        }
        let inner = &v[1..v.len() - 1];
        let mut escaped = String::with_capacity(inner.len() + 2);
        for ch in inner.chars() {
            match ch {
                '"' => escaped.push_str("\\\""),
                '\\' => escaped.push_str("\\\\"),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                _ => escaped.push(ch),
            }
        }
        return Some(format!("\"{escaped}\""));
    }
    if v.bytes()
        .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'-' | b'+' | b'e' | b'E' | b'x' | b'X' | b'a'..=b'f' | b'A'..=b'F'))
        && v.bytes().any(|c| c.is_ascii_digit())
    {
        return Some(v.to_string());
    }
    None
}

fn json_number(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 9.0e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

fn enum_to_object(name: &str, body: &str) -> Option<String> {
    if !is_plain_ident(name) {
        return None;
    }
    let mut members: Vec<(String, String)> = Vec::new();
    let mut auto: f64 = 0.0;
    for raw in split_members(body) {
        let mem = raw.trim();
        if mem.is_empty() {
            continue;
        }
        let (mname, value) = match mem.find('=') {
            Some(eq) => (mem[..eq].trim(), Some(mem[eq + 1..].trim())),
            None => (mem, None),
        };
        if !is_plain_ident(mname) {
            return None;
        }
        let json_val = match value {
            None => {
                let v = json_number(auto);
                auto += 1.0;
                v
            }
            Some(v) => {
                let lit = member_literal(v)?;
                if !lit.starts_with('"') {
                    if let Ok(n) = lit.parse::<f64>() {
                        auto = n + 1.0;
                    }
                }
                lit
            }
        };
        members.push((mname.to_string(), json_val));
    }

    let mut obj = String::from("{");
    for (i, (k, v)) in members.iter().enumerate() {
        if i > 0 {
            obj.push(',');
        }
        obj.push('"');
        obj.push_str(k);
        obj.push_str("\":");
        obj.push_str(v);
    }
    obj.push('}');
    Some(format!("var {name} = Object.freeze({obj});"))
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
    fn keeps_object_literal_values() {
        // AUDIT-2026-10-31 P0: `{ k: x }` used to strip to `{ k }`.
        let js = strip("const x = 1; const o = { k: x };");
        assert!(js.contains("k: x"), "object value dropped: {js}");
        let js2 = strip("const o = { a: x, b: 2, c: { d: e } };");
        assert!(js2.contains("a: x"), "object value dropped: {js2}");
        assert!(js2.contains("d: e"), "nested object value dropped: {js2}");
        let js3 = strip("function f() { return { inner: x }; }");
        assert!(js3.contains("inner: x"), "returned object value dropped: {js3}");
    }

    #[test]
    fn keeps_plain_exports_and_drops_type_only_declarations() {
        let js = strip("export const x = 1;");
        assert!(js.contains("export const x = 1;"), "plain export damaged: {js}");
        let iface = strip("export interface I { a: number }\nconst y = 1;");
        assert!(!iface.contains("export"), "dangling export after interface: {iface}");
        assert!(iface.contains("const y = 1;"), "following code eaten: {iface}");
        let alias = strip("export type F = number;\nconst z = 2;");
        assert!(!alias.contains("export type"), "dangling export type: {alias}");
        // A type-only re-export still needs its `export`.
        let re = strip("export type { X } from './x.ts';");
        assert!(re.contains("export"), "re-export lost its export: {re}");
    }

    #[test]
    fn strips_implements_and_abstract() {
        let js = strip("class A implements B, C { x = 1; }");
        assert!(!js.contains("implements"), "implements survived: {js}");
        assert!(js.contains("class A"), "class name lost: {js}");
        let ab = strip("abstract class A { abstract m(): void; n = 1; }");
        assert!(!ab.contains("abstract"), "abstract survived: {ab}");
    }

    #[test]
    fn strips_generic_arrow_functions() {
        let js = strip("const f = <T>(v: T): T => v;");
        assert!(!js.contains("<T>"), "type parameters survived: {js}");
        assert!(!js.contains(": T"), "parameter/return annotation survived: {js}");
        assert!(js.contains("=> v"), "arrow body lost: {js}");
        // A real comparison must be untouched.
        let cmp = strip("const r = a < b > (c);");
        assert!(cmp.contains("a < b > (c)"), "comparison mangled: {cmp}");
    }

    #[test]
    fn keeps_ternary_colons_after_object_literals() {
        // The colon of `c ? {a: 1} : {b: 2}` used to be blanked as a type.
        let js = strip("const t = c ? { a: 1 } : { b: 2 };");
        assert!(js.contains("} : {"), "ternary colon eaten: {js}");
        assert!(js.contains("a: 1"), "branch value dropped: {js}");
    }

    #[test]
    fn keeps_destructuring_renames() {
        let js = strip("const { a: b } = obj;");
        assert!(js.contains("a: b"), "rename dropped: {js}");
        let js2 = strip("function f({ a: b }) { return b; }");
        assert!(js2.contains("a: b"), "param rename dropped: {js2}");
    }

    #[test]
    fn still_strips_real_annotations() {
        assert!(!strip("const x: number = 1;").contains("number"));
        assert!(!strip("function f(a: number) { return a; }").contains("number"));
        assert!(!strip("class A { n: number = 0; }").contains("number"));
        assert!(!strip("class A { m(a: string) { return a; } }").contains("string"));
    }

    #[test]
    fn rewrites_constructor_parameter_properties() {
        // AUDIT-2026-10-31 P0: these used to be blanked without an assignment,
        // so \`this.a\` was undefined at runtime while the program still ran.
        let js = strip("class A { constructor(public a: number, private readonly b = 2) { } }");
        assert!(!js.contains("public"), "modifier survived: {js}");
        assert!(!js.contains("readonly"), "modifier survived: {js}");
        assert!(js.contains("this.a = a;"), "missing field assignment: {js}");
        assert!(js.contains("this.b = b;"), "missing field assignment: {js}");
    }

    #[test]
    fn leaves_a_plain_constructor_alone() {
        let src = "class A { constructor(a: number) { this.a = a; } }";
        let js = strip(src);
        assert!(js.contains("constructor("), "signature mangled: {js}");
        assert_eq!(js.matches("this.a = a;").count(), 1, "duplicated assignment: {js}");
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

    #[test]
    fn rewrites_numeric_enum() {
        let js = strip("enum Color { Red, Green, Blue = 5, Next }");
        assert_eq!(
            js.trim(),
            r#"var Color = Object.freeze({"Red":0,"Green":1,"Blue":5,"Next":6});"#
        );
    }

    #[test]
    fn rewrites_string_enum() {
        let js = strip("export enum Dir { Up = 'up', Down = \"down\" } plain;");
        assert!(
            js.contains(r#"var Dir = Object.freeze({"Up":"up","Down":"down"});"#),
            "{js}"
        );
        assert!(!js.contains("export enum"), "{js}");
    }

    #[test]
    fn leaves_plain_object_literals_alone() {
        let js = strip("const o = { a: 1 };");
        assert!(js.contains("{ a: 1 }"), "{js}");
    }

    #[test]
    fn enum_rewrite_preserves_line_count() {
        let src = "enum E {\n  A,\n  B\n}\nlet x = 1;";
        let js = strip(src);
        assert_eq!(src.matches('\n').count(), js.matches('\n').count(), "{js}");
        assert!(js.contains("let x = 1;"));
    }

    #[test]
    fn declare_enum_is_erased() {
        let js = strip("declare enum E { A }\nlet y = 2;");
        assert!(!js.contains("enum"), "{js}");
        assert!(js.contains("let y = 2;"));
    }

    #[test]
    fn unsupported_enum_member_is_left_intact() {
        // A computed member is not a literal: rewriting it would bind a wrong
        // value, so the stripper refuses and QuickJS reports a syntax error.
        let src = "enum E { A = foo() }";
        let js = strip(src);
        assert!(js.contains("enum E"), "{js}");
    }
}
