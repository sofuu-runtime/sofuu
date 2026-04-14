/*
 * src/ts/stripper.c — Sofuu TypeScript type stripper
 *
 * Replaces type-only syntax with spaces, preserving newlines so
 * line numbers remain accurate for JS error messages.
 */
#include "stripper.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ── Internal state ─────────────────────────────────────────── */
typedef struct {
    const char *src;
    char       *out;
    size_t      len;
    size_t      pos;
} TS;

/* Replace range [from,to) with spaces, keep newlines */
static void blank(TS *t, size_t from, size_t to) {
    for (size_t i = from; i < to; i++)
        if (t->out[i] != '\n' && t->out[i] != '\r') t->out[i] = ' ';
}

static int ids(char c)  { return isalpha((unsigned char)c) || c == '_' || c == '$'; }
static int idc(char c)  { return isalnum((unsigned char)c) || c == '_' || c == '$'; }

/* Keyword match at t->pos with word boundaries */
static int kw(TS *t, const char *w) {
    size_t wl = strlen(w);
    if (t->pos + wl > t->len) return 0;
    if (memcmp(t->src + t->pos, w, wl)) return 0;
    if (t->pos > 0 && idc(t->src[t->pos - 1])) return 0;
    if (t->pos + wl < t->len && idc(t->src[t->pos + wl])) return 0;
    return 1;
}

static void skip_ws(TS *t) {
    while (t->pos < t->len &&
           (t->src[t->pos] == ' ' || t->src[t->pos] == '\t')) t->pos++;
}

static void skip_ident(TS *t) {
    while (t->pos < t->len && idc(t->src[t->pos])) t->pos++;
}

static void skip_string(TS *t, char q) {
    t->pos++;
    while (t->pos < t->len && t->src[t->pos] != q) {
        if (t->src[t->pos] == '\\') t->pos++;
        t->pos++;
    }
    if (t->pos < t->len) t->pos++;
}

static void skip_template(TS *t) {
    t->pos++;
    while (t->pos < t->len && t->src[t->pos] != '`') {
        if (t->src[t->pos] == '\\') { t->pos += 2; continue; }
        if (t->src[t->pos] == '$' && t->pos + 1 < t->len
                && t->src[t->pos + 1] == '{') {
            t->pos += 2;
            int d = 1;
            while (t->pos < t->len && d > 0) {
                if      (t->src[t->pos] == '{') d++;
                else if (t->src[t->pos] == '}') d--;
                t->pos++;
            }
            continue;
        }
        t->pos++;
    }
    if (t->pos < t->len) t->pos++;
}

static void skip_line_comment(TS *t) {
    while (t->pos < t->len && t->src[t->pos] != '\n') t->pos++;
}

static void skip_block_comment(TS *t) {
    t->pos += 2;
    while (t->pos + 1 < t->len &&
           !(t->src[t->pos] == '*' && t->src[t->pos + 1] == '/')) t->pos++;
    if (t->pos + 1 < t->len) t->pos += 2;
}

/* Skip balanced brackets, handling nested strings/comments */
static void skip_balanced(TS *t, char open, char close) {
    if (t->pos >= t->len || t->src[t->pos] != open) return;
    int d = 1; t->pos++;
    while (t->pos < t->len && d > 0) {
        char c = t->src[t->pos];
        if (c == '\'' || c == '"') { skip_string(t, c); continue; }
        if (c == '`')              { skip_template(t);  continue; }
        if (c == '/' && t->pos + 1 < t->len) {
            if (t->src[t->pos + 1] == '/') { skip_line_comment(t);  continue; }
            if (t->src[t->pos + 1] == '*') { skip_block_comment(t); continue; }
        }
        if (c == open)  d++;
        else if (c == close) d--;
        t->pos++;
    }
}

/*
 * skip_type — advance past a TypeScript type expression.
 * Stops at depth-0: , ; ) ] } =
 * Handles unions/intersections (| &), nested <> [] () {}
 */
static void skip_type(TS *t) {
    int d = 0;
    while (t->pos < t->len) {
        char c = t->src[t->pos];
        /* Track depth — but { opens a new block; stop if at d==0 */
        if (c == '(') { d++; t->pos++; continue; }
        if (c == '[') { d++; t->pos++; continue; }
        if (c == '{') {
            if (d == 0) break;  /* never consume a function/object body */
            d++; t->pos++; continue;
        }
        if (c == ')' || c == ']' || c == '}') {
            if (d == 0) break;
            d--; t->pos++; continue;
        }
        /* angle brackets for generics */
        if (c == '<') { d++; t->pos++; continue; }
        if (c == '>') {
            if (d == 0) break;
            d--; t->pos++; continue;
        }
        /* stop at top-level delimiters */
        if (d == 0 && (c == ',' || c == ';' || c == '=')) break;
        /* newline may end type (check if next line continues with | &) */
        if (d == 0 && c == '\n') {
            size_t q = t->pos + 1;
            while (q < t->len && (t->src[q] == ' ' || t->src[q] == '\t')) q++;
            if (q < t->len && (t->src[q] == '|' || t->src[q] == '&')) {
                t->pos = q; continue;
            }
            break;
        }
        /* string types (template literal types etc.) */
        if (c == '\'' || c == '"') { skip_string(t, c); continue; }
        if (c == '`')              { skip_template(t);  continue; }
        t->pos++;
    }
}

/* ── Main stripping logic ────────────────────────────────────── */

char *ts_strip(const char *src, size_t len, size_t *out_len) {
    char *buf = (char *)malloc(len + 1);
    if (!buf) return NULL;
    memcpy(buf, src, len);
    buf[len] = '\0';

    TS t = { src, buf, len, 0 };

    /*
     * last_sig: 1 if last significant token was an identifier,
     * literal, ) or ] — used to disambiguate ':' and 'as' etc.
     */
    int last_sig = 0;

    static const char * const mods[] = {
        "public", "private", "protected", "readonly",
        "abstract", "override", NULL
    };

    while (t.pos < len) {
        char c = src[t.pos];

        /* ── Comments ── */
        if (c == '/' && t.pos + 1 < len && src[t.pos + 1] == '/') {
            skip_line_comment(&t); continue;
        }
        if (c == '/' && t.pos + 1 < len && src[t.pos + 1] == '*') {
            skip_block_comment(&t); continue;
        }

        /* ── String / template literals ── */
        if (c == '\'' || c == '"') { skip_string(&t, c);  last_sig = 1; continue; }
        if (c == '`')              { skip_template(&t);   last_sig = 1; continue; }

        /* ── Whitespace ── */
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { t.pos++; continue; }

        /* ════════════════════════════════════════════════════════
         *  TypeScript erasable syntax — strip to spaces
         * ════════════════════════════════════════════════════════ */

        /* interface Foo<T> extends Bar { ... } */
        if (kw(&t, "interface")) {
            size_t start = t.pos;
            t.pos += 9; skip_ws(&t); skip_ident(&t); skip_ws(&t);
            if (t.pos < len && src[t.pos] == '<') skip_balanced(&t, '<', '>');
            skip_ws(&t);
            if (kw(&t, "extends")) {
                t.pos += 7;
                while (t.pos < len && src[t.pos] != '{') {
                    if (src[t.pos] == '<') skip_balanced(&t, '<', '>');
                    else t.pos++;
                }
            }
            if (t.pos < len && src[t.pos] == '{') skip_balanced(&t, '{', '}');
            blank(&t, start, t.pos);
            last_sig = 0; continue;
        }

        /* type Foo<T> = ... ; */
        if (kw(&t, "type")) {
            size_t q = t.pos + 4;
            while (q < len && (src[q] == ' ' || src[q] == '\t')) q++;
            /* must be followed by an identifier (type alias, not typeof) */
            if (q < len && ids(src[q])) {
                size_t start = t.pos;
                t.pos = q; skip_ident(&t); skip_ws(&t);
                if (t.pos < len && src[t.pos] == '<') skip_balanced(&t, '<', '>');
                skip_ws(&t);
                if (t.pos < len && src[t.pos] == '=') {
                    t.pos++;
                    int d = 0;
                    while (t.pos < len) {
                        char cc = src[t.pos];
                        if (cc == '{' || cc == '(' || cc == '[') { d++; t.pos++; }
                        else if (cc == '}' || cc == ')' || cc == ']') {
                            if (!d) break; d--; t.pos++;
                        }
                        else if (!d && cc == ';') { t.pos++; break; }
                        else if (!d && cc == '\n') break;
                        else t.pos++;
                    }
                }
                blank(&t, start, t.pos);
                last_sig = 0; continue;
            }
        }

        /* import type { ... } from '...' → blank entire statement */
        if (kw(&t, "import")) {
            size_t q = t.pos + 6;
            while (q < len && (src[q] == ' ' || src[q] == '\t')) q++;
            if (q + 4 <= len && memcmp(src + q, "type", 4) == 0 && !idc(src[q + 4])) {
                /* don't strip `import type(...)` — that's import() */
                size_t peek = q + 4;
                while (peek < len && (src[peek] == ' ' || src[peek] == '\t')) peek++;
                if (peek >= len || src[peek] != '(') {
                    size_t start = t.pos;
                    t.pos = q + 4;
                    while (t.pos < len && src[t.pos] != ';' && src[t.pos] != '\n') {
                        if (src[t.pos] == '\'' || src[t.pos] == '"')
                            skip_string(&t, src[t.pos]);
                        else if (src[t.pos] == '{') skip_balanced(&t, '{', '}');
                        else t.pos++;
                    }
                    if (t.pos < len && src[t.pos] == ';') t.pos++;
                    blank(&t, start, t.pos);
                    last_sig = 0; continue;
                }
            }
        }

        /* export type { ... } → blank just "type" */
        if (kw(&t, "export")) {
            size_t q = t.pos + 6;
            while (q < len && (src[q] == ' ' || src[q] == '\t')) q++;
            if (q + 4 <= len && memcmp(src + q, "type", 4) == 0 && !idc(src[q + 4])) {
                blank(&t, q, q + 4);
                t.pos += 6; last_sig = 0; continue;
            }
        }

        /* declare ... */
        if (kw(&t, "declare")) {
            size_t start = t.pos;
            t.pos += 7; skip_ws(&t);
            int d = 0;
            while (t.pos < len) {
                char cc = src[t.pos];
                if (cc == '{' || cc == '(' || cc == '[') { d++; t.pos++; }
                else if (cc == '}' || cc == ')' || cc == ']') {
                    if (!d) break; d--; t.pos++;
                }
                else if (!d && cc == ';') { t.pos++; break; }
                else t.pos++;
            }
            blank(&t, start, t.pos);
            last_sig = 0; continue;
        }

        /* public / private / protected / readonly / abstract / override */
        {
            int hit = 0;
            for (int i = 0; mods[i]; i++) {
                if (kw(&t, mods[i])) {
                    size_t ml = strlen(mods[i]);
                    size_t q  = t.pos + ml;
                    while (q < len && (src[q] == ' '|| src[q] == '\t')) q++;
                    if (q < len && (ids(src[q]) || src[q] == '#'
                                    || src[q] == '[' || src[q] == '*'
                                    || src[q] == '(')) {
                        blank(&t, t.pos, t.pos + ml);
                        t.pos += ml; hit = 1; break;
                    }
                }
            }
            if (hit) continue;
        }

        /* as Type */
        if (kw(&t, "as") && last_sig) {
            size_t start = t.pos;
            t.pos += 2; skip_ws(&t); skip_type(&t);
            blank(&t, start, t.pos);
            /* last_sig stays 1 — the value expression before `as` remains */
            continue;
        }

        /* satisfies Type */
        if (kw(&t, "satisfies") && last_sig) {
            size_t start = t.pos;
            t.pos += 9; skip_ws(&t); skip_type(&t);
            blank(&t, start, t.pos);
            continue;
        }

        /* : TypeAnnotation */
        if (c == ':' && last_sig) {
            size_t q = t.pos + 1;
            while (q < len && (src[q] == ' ' || src[q] == '\t')) q++;
            char nc = (q < len) ? src[q] : 0;
            /* Only strip if next token looks like a type */
            if (ids(nc) || nc == '(' || nc == '{' || nc == '[' || nc == '|' || nc == '&') {
                size_t start = t.pos;
                t.pos = q;
                skip_type(&t);
                blank(&t, start, t.pos);
                last_sig = 0; continue;
            }
        }

        /* <TypeParams> on functions / classes: fn<T>() or new Foo<T>() */
        if (c == '<' && last_sig) {
            size_t start = t.pos;
            size_t saved = t.pos;
            skip_balanced(&t, '<', '>');
            size_t q = t.pos;
            while (q < len && (src[q] == ' ' || src[q] == '\t')) q++;
            if (q < len && (src[q] == '(' || src[q] == '{' || src[q] == ',')) {
                blank(&t, start, t.pos);
                last_sig = 0; continue;
            }
            t.pos = saved; /* not a generic — fall through */
        }

        /* ! non-null assertion: x! */
        if (c == '!' && last_sig
                && t.pos + 1 < len && src[t.pos + 1] != '=') {
            blank(&t, t.pos, t.pos + 1);
            t.pos++; continue;
        }

        /* ?: optional parameter: (x?: Type) — strip the ? */
        if (c == '?' && last_sig
                && t.pos + 1 < len && src[t.pos + 1] == ':') {
            blank(&t, t.pos, t.pos + 1);
            t.pos++; continue;
        }

        /* ── Default: advance, update last_sig ── */
        if (ids(c)) {
            skip_ident(&t); last_sig = 1;
        } else if (isdigit((unsigned char)c)) {
            while (t.pos < len && (isalnum((unsigned char)src[t.pos])
                                    || src[t.pos] == '.')) t.pos++;
            last_sig = 1;
        } else if (c == ')' || c == ']' || c == '}') {
            last_sig = 1; t.pos++;
        } else if (c == '.' && t.pos + 2 < len
                   && src[t.pos + 1] == '.' && src[t.pos + 2] == '.') {
            /* spread / rest — not a type position after this */
            last_sig = 0; t.pos += 3;
        } else {
            last_sig = 0; t.pos++;
        }
    }

    if (out_len) *out_len = len;
    return buf;
}
