/*
 * src/bundler/bundler.c — Sofuu single-file JS bundler
 *
 * Produces a self-contained bundle from an ESM entry point:
 *
 *   sofuu bundle src/main.js -o dist/app.bundle.js
 *
 * Algorithm:
 *   1. Parse entry file for import specifiers
 *   2. BFS walk the import graph, resolving every specifier
 *   3. Strip TypeScript types (if .ts)
 *   4. Transform each module: ESM imports/exports → bundle registry calls
 *   5. Emit: preamble + __d(id, factory) per module + __r(entry) call
 *
 * Bundle registry (emitted as preamble):
 *
 *   var __m = {};          // module export cache
 *   var __f = {};          // module factory functions
 *   function __d(id, fn) { __f[id] = fn; }
 *   function __r(id) {
 *     if (__m[id]) return __m[id];
 *     var exp = {}; __m[id] = exp;
 *     __f[id](exp, __r);
 *     return __m[id];
 *   }
 *
 * Each module is wrapped as:
 *   __d("./src/foo.js", function(exports, require) {
 *     // ... transformed source ...
 *   });
 *
 * Import transforms:
 *   import Foo from './foo'          → var Foo = require('./foo').default
 *   import { a, b } from './bar'     → var {a,b} = require('./bar')
 *   import * as ns from './baz'      → var ns = require('./baz')
 *   import './side-effects'          → require('./side-effects')
 *   import('./dyn')                  → Promise.resolve(require('./dyn'))
 *
 * Export transforms:
 *   export default X                 → exports.default = exports.__default = X
 *   export { a, b }                  → exports.a = a; exports.b = b
 *   export const/let/var X = Y      → const/let/var X = exports.X = Y
 *   export function foo() {}        → function foo() {} exports.foo = foo
 *   export class Foo {}             → class Foo {} exports.Foo = Foo
 *   export * from './x'             → Object.assign(exports, require('./x'))
 *   export { a } from './x'         → var __rx=require('./x'); exports.a=__rx.a
 */
#include "bundler.h"
#include "../ts/stripper.h"
#include "../npm/resolver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <libgen.h>  /* dirname, basename */

/* ═══════════════════════════════════════════════════════════════
   §1  Growable string buffer
   ═══════════════════════════════════════════════════════════════ */

typedef struct { char *buf; size_t len; size_t cap; } SBuf;

static void sb_init(SBuf *s) { s->buf = NULL; s->len = s->cap = 0; }

static int sb_grow(SBuf *s, size_t extra) {
    if (s->len + extra + 1 <= s->cap) return 1;
    size_t nc = (s->cap == 0 ? 8192 : s->cap * 2) + extra;
    char *nb = (char *)realloc(s->buf, nc);
    if (!nb) return 0;
    s->buf = nb; s->cap = nc;
    return 1;
}

static void sb_app(SBuf *s, const char *d, size_t n) {
    if (!d || !sb_grow(s, n)) return;
    memcpy(s->buf + s->len, d, n);
    s->len += n;
    s->buf[s->len] = '\0';
}
static void sb_str(SBuf *s, const char *str) { if (str) sb_app(s, str, strlen(str)); }
static void sb_ch(SBuf *s, char c)           { sb_app(s, &c, 1); }
static void sb_fmt(SBuf *s, const char *fmt, ...) {
    char tmp[1024];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    sb_app(s, tmp, (size_t)(n < 0 ? 0 : n));
}
static void sb_free(SBuf *s) { free(s->buf); s->buf = NULL; s->len = s->cap = 0; }

/* ═══════════════════════════════════════════════════════════════
   §2  Module graph
   ═══════════════════════════════════════════════════════════════ */

#define MAX_MODS 4096

typedef struct {
    char   abs_path[PATH_MAX]; /* absolute resolved path on disk */
    char   id[PATH_MAX];       /* relative module id (from bundle root) */
    char  *source;             /* raw source (malloc'd) */
    char  *transformed;        /* after transform (malloc'd, may be same as source) */
} BundleMod;

typedef struct {
    BundleMod *mods[MAX_MODS];
    int        count;
    char       root_dir[PATH_MAX]; /* directory of the entry file */
    char       entry_id[PATH_MAX]; /* id of the entry module */
} Graph;

static void graph_free(Graph *g) {
    for (int i = 0; i < g->count; i++) {
        free(g->mods[i]->source);
        free(g->mods[i]->transformed);
        free(g->mods[i]);
    }
    g->count = 0;
}

/* Find module by abs_path (dedup check) */
static BundleMod *graph_find(Graph *g, const char *abs_path) {
    for (int i = 0; i < g->count; i++)
        if (strcmp(g->mods[i]->abs_path, abs_path) == 0) return g->mods[i];
    return NULL;
}

/* Register a new module slot (doesn't set source/transformed yet) */
static BundleMod *graph_add(Graph *g, const char *abs_path, const char *id) {
    if (g->count >= MAX_MODS) return NULL;
    BundleMod *m = (BundleMod *)calloc(1, sizeof(BundleMod));
    if (!m) return NULL;
    strncpy(m->abs_path, abs_path, PATH_MAX - 1);
    strncpy(m->id, id, PATH_MAX - 1);
    g->mods[g->count++] = m;
    return m;
}

/* ═══════════════════════════════════════════════════════════════
   §3  Source scanner — string/comment aware
   ═══════════════════════════════════════════════════════════════ */

/* Skip a quoted string starting at src[pos] (pos is on the opening quote).
   Returns position AFTER the closing quote. */
static size_t skip_string(const char *src, size_t pos, size_t len) {
    char q = src[pos++];
    while (pos < len) {
        if (src[pos] == '\\') { pos += 2; continue; }
        if (src[pos] == q) { pos++; break; }
        pos++;
    }
    return pos;
}

/* Skip a template literal starting at src[pos] (pos is on the backtick).
   Simplified: doesn't handle nested ${...} with quotes inside.
   Returns position AFTER the closing backtick. */
static size_t skip_template(const char *src, size_t pos, size_t len) {
    pos++; /* skip opening ` */
    while (pos < len) {
        if (src[pos] == '\\') { pos += 2; continue; }
        if (src[pos] == '`') { pos++; break; }
        pos++;
    }
    return pos;
}

/* Skip from current pos (on '/') past // comment to end-of-line.
   Returns position after '\n'. */
static size_t skip_line_comment(const char *src, size_t pos, size_t len) {
    while (pos < len && src[pos] != '\n') pos++;
    return pos;
}

/* Skip from current pos past a block comment ( slash-star ... star-slash ).
   Returns position after the closing delimiter. */
static size_t skip_block_comment(const char *src, size_t pos, size_t len) {
    pos += 2; /* skip opening delimiter */
    while (pos + 1 < len) {
        if (src[pos] == '*' && src[pos+1] == '/') { pos += 2; break; }
        pos++;
    }
    return pos;
}

/* Is the character at pos the start of an identifier character? */
static int is_id_start(char c) { return isalpha((unsigned char)c) || c == '_' || c == '$'; }
static int is_id_cont(char c)  { return is_id_start(c) || isdigit((unsigned char)c); }

/* Skip whitespace (not newlines). */
static size_t skip_ws(const char *src, size_t pos, size_t len) {
    while (pos < len && (src[pos] == ' ' || src[pos] == '\t')) pos++;
    return pos;
}

/* Skip any whitespace including newlines. */
static size_t skip_wsn(const char *src, size_t pos, size_t len) {
    while (pos < len && isspace((unsigned char)src[pos])) pos++;
    return pos;
}

/* ═══════════════════════════════════════════════════════════════
   §4  Import specifier extraction
   ═══════════════════════════════════════════════════════════════ */

/*
 * Scan src from pos, expecting we just consumed "import".
 * Collects the quoted specifier (the 'from "..."' part).
 * Updates *end_pos to just after the statement.
 * Returns malloc'd specifier string, or NULL if not an import-from or plain import.
 *
 * Also handles:  import('dyn')  by looking for ( then quoted string then )
 * In that case sets *is_dynamic = 1.
 */
static char *read_quoted(const char *src, size_t *pos, size_t len) {
    *pos = skip_wsn(src, *pos, len);
    if (*pos >= len) return NULL;
    char q = src[*pos];
    if (q != '"' && q != '\'') return NULL;
    (*pos)++; /* opening quote */
    size_t start = *pos;
    while (*pos < len && src[*pos] != q) {
        if (src[*pos] == '\\') (*pos)++;
        (*pos)++;
    }
    size_t slen = *pos - start;
    if (*pos < len) (*pos)++; /* closing quote */
    char *out = (char *)malloc(slen + 1);
    if (!out) return NULL;
    memcpy(out, src + start, slen);
    out[slen] = '\0';
    return out;
}

/* After reading "import", scan forward to find the quoted specifier.
 * Skips over binding forms like {a,b}, * as ns, defaultExport.
 * Returns specifier or NULL. end_pos set to after semicolon/newline. */
static char *parse_import_specifier(const char *src, size_t pos, size_t len,
                                    size_t *end_pos, int *is_dynamic) {
    *is_dynamic = 0;
    size_t p = skip_ws(src, pos, len);

    /* Dynamic import: import('...') */
    if (p < len && src[p] == '(') {
        *is_dynamic = 1;
        p++; /* skip '(' */
        p = skip_wsn(src, p, len);
        char *spec = read_quoted(src, &p, len);
        p = skip_wsn(src, p, len);
        if (p < len && src[p] == ')') p++;
        /* skip to end of statement */
        while (p < len && src[p] != ';' && src[p] != '\n') p++;
        if (p < len && src[p] == ';') p++;
        *end_pos = p;
        return spec;
    }

    /* Bare import (no bindings): import 'specifier' */
    if (p < len && (src[p] == '"' || src[p] == '\'')) {
        char *spec = read_quoted(src, &p, len);
        while (p < len && src[p] != ';' && src[p] != '\n') p++;
        if (p < len && src[p] == ';') p++;
        *end_pos = p;
        return spec;
    }

    /* Has bindings — scan forward looking for 'from' keyword at top brace depth */
    int depth = 0;
    while (p < len) {
        char c = src[p];

        if (c == '"' || c == '\'') { p = skip_string(src, p, len); continue; }
        if (c == '`')              { p = skip_template(src, p, len); continue; }
        if (c == '/' && p+1 < len && src[p+1] == '/') { p = skip_line_comment(src, p, len); continue; }
        if (c == '/' && p+1 < len && src[p+1] == '*') { p = skip_block_comment(src, p, len); continue; }

        if (c == '{' || c == '(') { depth++; p++; continue; }
        if (c == '}' || c == ')') { depth--; p++; continue; }

        /* At depth 0, check for 'from' keyword */
        if (depth == 0 && c == 'f' && p+4 <= len &&
            strncmp(src + p, "from", 4) == 0 &&
            (p + 4 >= len || !is_id_cont(src[p+4]))) {
            p += 4;
            p = skip_wsn(src, p, len);
            char *spec = read_quoted(src, &p, len);
            /* skip optional semicolon */
            p = skip_ws(src, p, len);
            if (p < len && src[p] == ';') p++;
            *end_pos = p;
            return spec;
        }
        p++;
    }
    *end_pos = p;
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════
   §5  Collect all import specifiers from a source file
   ═══════════════════════════════════════════════════════════════ */

typedef struct { char *spec; int is_dynamic; } SpecItem;

/* Collects all import specifiers from src.
   Returns malloc'd array of SpecItems, count in *out_count.
   Caller must free each spec and the array. */
static SpecItem *collect_specifiers(const char *src, size_t len, int *out_count) {
    SpecItem *items = NULL;
    int cap = 0, count = 0;

    size_t p = 0;
    while (p < len) {
        char c = src[p];

        /* Skip strings and comments */
        if (c == '"' || c == '\'') { p = skip_string(src, p, len); continue; }
        if (c == '`')              { p = skip_template(src, p, len); continue; }
        if (c == '/' && p+1 < len && src[p+1] == '/') { p = skip_line_comment(src, p, len); continue; }
        if (c == '/' && p+1 < len && src[p+1] == '*') { p = skip_block_comment(src, p, len); continue; }

        /* Detect 'import' keyword */
        if (c == 'i' && p + 6 <= len && strncmp(src + p, "import", 6) == 0) {
            /* make sure it's not part of a longer identifier */
            int preceded_by_id = (p > 0 && is_id_cont(src[p-1]));
            int followed_by_id = (p + 6 < len && is_id_cont(src[p+6]));
            if (!preceded_by_id && !followed_by_id) {
                size_t end = p;
                int is_dyn = 0;
                char *spec = parse_import_specifier(src, p + 6, len, &end, &is_dyn);
                if (spec) {
                    if (count >= cap) {
                        cap = cap == 0 ? 16 : cap * 2;
                        items = (SpecItem *)realloc(items, cap * sizeof(SpecItem));
                    }
                    items[count].spec = spec;
                    items[count].is_dynamic = is_dyn;
                    count++;
                    p = end;
                    continue;
                }
            }
        }
        p++;
    }
    *out_count = count;
    return items;
}

/* ═══════════════════════════════════════════════════════════════
   §6  Path utilities
   ═══════════════════════════════════════════════════════════════ */

static int path_is_relative(const char *spec) {
    return spec[0] == '.' && (spec[1] == '/' || (spec[1] == '.' && spec[2] == '/'));
}

/* Join dir + "/" + rel and normalize. Result into out (PATH_MAX). */
static void path_join(char *out, const char *dir, const char *rel) {
    char tmp[PATH_MAX];
    snprintf(tmp, PATH_MAX, "%s/%s", dir, rel);
    /* realpath resolves ..'s etc, but file may not exist yet — use it best-effort */
    if (!realpath(tmp, out)) strncpy(out, tmp, PATH_MAX - 1);
}

/* Try common extensions on a path that doesn't have one */
static int try_extensions(char *abs, size_t cap) {
    struct stat st;
    static const char *exts[] = {".js", ".mjs", ".ts", "/index.js", "/index.ts", NULL};
    if (stat(abs, &st) == 0 && S_ISREG(st.st_mode)) return 1;
    size_t base_len = strlen(abs);
    for (int i = 0; exts[i]; i++) {
        if (base_len + strlen(exts[i]) + 1 > cap) continue;
        strncpy(abs + base_len, exts[i], cap - base_len - 1);
        if (stat(abs, &st) == 0 && S_ISREG(st.st_mode)) return 1;
    }
    abs[base_len] = '\0';
    return 0;
}

/* Resolve a specifier to abs path. mod_dir = directory of the importing file. */
static int resolve_specifier(const char *spec, const char *mod_dir, const char *root_dir,
                              char *out_abs, char *out_id) {
    if (path_is_relative(spec)) {
        char abs[PATH_MAX];
        path_join(abs, mod_dir, spec);
        if (!try_extensions(abs, PATH_MAX)) return 0;
        strncpy(out_abs, abs, PATH_MAX - 1);
        /* Make id relative from root_dir */
        const char *rel = strstr(abs, root_dir);
        if (rel && rel == abs) {
            rel += strlen(root_dir);
            if (*rel == '/') rel++;
            snprintf(out_id, PATH_MAX, "./%s", rel);
        } else {
            strncpy(out_id, abs, PATH_MAX - 1);
        }
        return 1;
    } else {
        /* npm module — use existing npm_resolve */
        char *resolved = npm_resolve(mod_dir, spec);
        if (!resolved) return 0;
        strncpy(out_abs, resolved, PATH_MAX - 1);
        snprintf(out_id, PATH_MAX, "node_modules/%s", spec);
        free(resolved);
        return 1;
    }
}

/* ═══════════════════════════════════════════════════════════════
   §7  BFS graph walker
   ═══════════════════════════════════════════════════════════════ */

static char *read_file_alloc(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    if (sz <= 0) { fclose(f); return calloc(1, 1); }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

static int build_graph(Graph *g, const char *entry_abs) {
    /* BFS queue: we reuse g->mods as the queue by tracking a head index */
    int head = 0;

    char entry_dir[PATH_MAX];
    strncpy(entry_dir, entry_abs, PATH_MAX - 1);
    /* dirname modifies its argument — use a copy */
    char tmp_entry[PATH_MAX];
    strncpy(tmp_entry, entry_abs, PATH_MAX - 1);
    strncpy(entry_dir, dirname(tmp_entry), PATH_MAX - 1);
    strncpy(g->root_dir, entry_dir, PATH_MAX - 1);

    /* Add entry module */
    char entry_id[PATH_MAX];
    snprintf(entry_id, PATH_MAX, "./%s",
             /* use basename of the entry as the id */
             entry_abs + strlen(entry_dir) + (entry_dir[strlen(entry_dir)-1] == '/' ? 0 : 1));
    strncpy(g->entry_id, entry_id, PATH_MAX - 1);

    BundleMod *em = graph_add(g, entry_abs, entry_id);
    if (!em) return 0;

    while (head < g->count) {
        BundleMod *m = g->mods[head++];

        /* Read source */
        m->source = read_file_alloc(m->abs_path);
        if (!m->source) {
            fprintf(stderr, "  \033[31m✗\033[0m Cannot read: %s\n", m->abs_path);
            continue;
        }

        /* Strip TypeScript if needed */
        const char *src = m->source;
        char *stripped = NULL;
        size_t src_len = strlen(src);
        size_t slen = strlen(m->abs_path);
        if (slen >= 3 && strcmp(m->abs_path + slen - 3, ".ts") == 0) {
            size_t out_len = 0;
            stripped = ts_strip(src, src_len, &out_len);
            if (stripped) src = stripped;
        }

        /* Collect imports */
        int spec_count = 0;
        SpecItem *specs = collect_specifiers(src, strlen(src), &spec_count);
        free(stripped);

        /* Get this module's directory */
        char mod_dir[PATH_MAX];
        strncpy(tmp_entry, m->abs_path, PATH_MAX - 1);
        strncpy(mod_dir, dirname(tmp_entry), PATH_MAX - 1);

        /* Resolve each specifier and enqueue if not yet seen */
        for (int i = 0; i < spec_count; i++) {
            char dep_abs[PATH_MAX], dep_id[PATH_MAX];
            if (resolve_specifier(specs[i].spec, mod_dir, g->root_dir, dep_abs, dep_id)) {
                if (!graph_find(g, dep_abs)) {
                    BundleMod *dep = graph_add(g, dep_abs, dep_id);
                    if (dep) {
                        /* progress indicator */
                        printf("  \033[90m+\033[0m %s\n", dep_id);
                    }
                }
            }
            free(specs[i].spec);
        }
        free(specs);
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════
   §8  Source transformer: ESM → bundle registry format
   ═══════════════════════════════════════════════════════════════ */

/* Resolve a specifier relative to mod_id within the graph (returns the target's id, or
   falls back to a node_modules id). */
static const char *resolve_id_in_graph(Graph *g, const char *mod_abs,
                                       const char *spec, char *buf, size_t cap) {
    char mod_dir[PATH_MAX], tmp[PATH_MAX];
    strncpy(tmp, mod_abs, PATH_MAX - 1);
    strncpy(mod_dir, dirname(tmp), PATH_MAX - 1);

    char dep_abs[PATH_MAX], dep_id[PATH_MAX];
    if (resolve_specifier(spec, mod_dir, g->root_dir, dep_abs, dep_id)) {
        BundleMod *dep = graph_find(g, dep_abs);
        if (dep) { strncpy(buf, dep->id, cap - 1); return buf; }
    }
    strncpy(buf, spec, cap - 1);
    return buf;
}

/* Transform a module's source for bundle use.
 *
 * Replaces:
 *   import X, { a, b } from './foo'    → var X=require('./foo').default;var{a,b}=require('./foo')
 *   import * as ns from './bar'        → var ns=require('./bar')
 *   import './side'                    → require('./side')
 *   import('./dyn')                    → Promise.resolve(require('./dyn'))
 *   export default X                   → exports.default=exports.__default=X
 *   export { a, b }                    → exports.a=a;exports.b=b
 *   export { a as default }            → exports.default=exports.__default=a
 *   export const/let/var X = ...       → const/let/var X=...;\nexports.X=X
 *   export function foo() { ... }      → function foo() { ... }\nexports.foo=foo
 *   export class Foo { ... }           → class Foo { ... }\nexports.Foo=Foo
 *   export * from './x'                → Object.assign(exports,require('./x'))
 *   export { a, b } from './x'         → {var __rx=require('./x');exports.a=__rx.a;exports.b=__rx.b}
 */
static char *transform_module(Graph *g, BundleMod *m, const char *src) {
    size_t len = strlen(src);
    SBuf out;
    sb_init(&out);

    size_t p = 0;
    while (p < len) {
        char c = src[p];

        /* Pass through strings, template literals, comments verbatim */
        if (c == '"' || c == '\'') {
            size_t end = skip_string(src, p, len);
            sb_app(&out, src + p, end - p);
            p = end; continue;
        }
        if (c == '`') {
            size_t end = skip_template(src, p, len);
            sb_app(&out, src + p, end - p);
            p = end; continue;
        }
        if (c == '/' && p+1 < len && src[p+1] == '/') {
            size_t end = skip_line_comment(src, p, len);
            sb_app(&out, src + p, end - p);
            p = end; continue;
        }
        if (c == '/' && p+1 < len && src[p+1] == '*') {
            size_t end = skip_block_comment(src, p, len);
            sb_app(&out, src + p, end - p);
            p = end; continue;
        }

        /* ── import ──────────────────────────────────────────── */
        if (c == 'i' && p+6 <= len && strncmp(src+p, "import", 6) == 0 &&
            (p == 0 || !is_id_cont(src[p-1])) &&
            (p+6 >= len || !is_id_cont(src[p+6]))) {

            size_t end = p;
            int is_dyn = 0;
            char *spec = parse_import_specifier(src, p+6, len, &end, &is_dyn);

            if (spec) {
                char id_buf[PATH_MAX];
                resolve_id_in_graph(g, m->abs_path, spec, id_buf, PATH_MAX);
                free(spec);

                if (is_dyn) {
                    sb_fmt(&out, "Promise.resolve(require('%s'))", id_buf);
                } else {
                    /* Extract the binding portion between 'import' and 'from' or the specifier */
                    /* Re-parse to get the binding text */
                    size_t bp = p + 6;
                    bp = skip_ws(src, bp, len);

                    if (src[bp] == '(' || src[bp] == '"' || src[bp] == '\'') {
                        /* import 'side-effect' or import('dyn') — already handled */
                        sb_fmt(&out, "require('%s')", id_buf);
                    } else {
                        /* Collect binding text up to 'from' */
                        SBuf binding; sb_init(&binding);
                        int depth = 0;
                        while (bp < end) {
                            char bc = src[bp];
                            if (bc == '"' || bc == '\'') {
                                size_t se = skip_string(src, bp, end);
                                sb_app(&binding, src+bp, se-bp); bp = se; continue;
                            }
                            if (bc == '{') { depth++; sb_ch(&binding, bc); bp++; continue; }
                            if (bc == '}') { depth--; sb_ch(&binding, bc); bp++; continue; }
                            /* Stop at 'from' at depth 0 */
                            if (depth == 0 && bc == 'f' && bp+4 <= end &&
                                strncmp(src+bp, "from", 4) == 0 &&
                                (bp+4 >= end || !is_id_cont(src[bp+4]))) break;
                            sb_ch(&binding, bc); bp++;
                        }
                        /* Trim whitespace from binding */
                        char *bind = binding.buf;
                        if (bind) {
                            while (*bind && isspace((unsigned char)*bind)) bind++;
                            size_t bl = strlen(bind);
                            while (bl > 0 && isspace((unsigned char)bind[bl-1])) bl--;
                            bind[bl] = '\0';
                        } else { bind = (char *)""; }

                        /* Now emit the appropriate require form */
                        if (!*bind) {
                            sb_fmt(&out, "require('%s')", id_buf);
                        } else if (bind[0] == '*') {
                            /* import * as ns from '...' */
                            const char *ns = strstr(bind, "as");
                            if (ns) {
                                ns += 2;
                                while (*ns && isspace((unsigned char)*ns)) ns++;
                                char ns_name[256] = {0};
                                size_t ni = 0;
                                while (ns[ni] && is_id_cont(ns[ni])) { ns_name[ni] = ns[ni]; ni++; }
                                sb_fmt(&out, "var %s=require('%s')", ns_name, id_buf);
                            } else {
                                sb_fmt(&out, "require('%s')", id_buf);
                            }
                        } else if (bind[0] == '{') {
                            /* import { a, b as c } from '...'
                             * CJS destructuring doesn't support 'as', so expand each name. */
                            /* Check if there are any 'as' renames */
                            int has_as = (strstr(bind, " as ") != NULL);
                            if (!has_as) {
                                /* Simple: var {a,b}=require('...') */
                                sb_fmt(&out, "var %s=require('%s')", bind, id_buf);
                            } else {
                                /* Complex: expand each name independently */
                                char tmp_req[64];
                                snprintf(tmp_req, sizeof(tmp_req), "__ri_%04x", (unsigned)(p & 0xffff));
                                sb_fmt(&out, "var %s=require('%s');", tmp_req, id_buf);
                                /* tokenize the {a, b as c} list */
                                char bind_cp[4096]; strncpy(bind_cp, bind + 1, sizeof(bind_cp)-1);
                                /* strip trailing } */
                                char *close = strrchr(bind_cp, '}');
                                if (close) *close = '\0';
                                char *btok = strtok(bind_cp, ",");
                                while (btok) {
                                    while (*btok && isspace((unsigned char)*btok)) btok++;
                                    char *as_p = strstr(btok, " as ");
                                    if (as_p) {
                                        char orig[128]={0}, alias[128]={0};
                                        size_t ol = (size_t)(as_p - btok);
                                        while (ol>0 && isspace((unsigned char)btok[ol-1])) ol--;
                                        memcpy(orig, btok, ol<127?ol:127);
                                        const char *al = as_p + 4;
                                        while (*al && isspace((unsigned char)*al)) al++;
                                        size_t all = strlen(al);
                                        while (all>0 && isspace((unsigned char)al[all-1])) all--;
                                        memcpy(alias, al, all<127?all:127);
                                        sb_fmt(&out, "var %s=%s.%s;", alias, tmp_req, orig);
                                    } else {
                                        size_t tl = strlen(btok);
                                        while (tl>0 && isspace((unsigned char)btok[tl-1])) tl--;
                                        btok[tl] = '\0';
                                        if (*btok) sb_fmt(&out, "var %s=%s.%s;", btok, tmp_req, btok);
                                    }
                                    btok = strtok(NULL, ",");
                                }
                            }
                        } else {
                            /* import Default from '...' or import Default, { a } from '...' */
                            const char *comma = strchr(bind, ',');
                            if (comma) {
                                char def_name[256] = {0};
                                size_t def_len = (size_t)(comma - bind);
                                while (def_len > 0 && isspace((unsigned char)bind[def_len-1])) def_len--;
                                memcpy(def_name, bind, def_len < 255 ? def_len : 255);
                                /* Emit default */
                                sb_fmt(&out,
                                    "var __r_%s=require('%s');var %s=__r_%s.default||__r_%s;",
                                    def_name, id_buf, def_name, def_name, def_name);
                                /* Find the {named} part and expand each alias */
                                const char *named = comma + 1;
                                while (*named && isspace((unsigned char)*named)) named++;
                                /* named is like "{ greet as sayHi }" — strip braces */
                                if (*named == '{') {
                                    named++;
                                    char ncp[2048]; strncpy(ncp, named, sizeof(ncp)-1);
                                    char *ncl = strrchr(ncp, '}');
                                    if (ncl) *ncl = '\0';
                                    char *ntok = strtok(ncp, ",");
                                    while (ntok) {
                                        while (*ntok && isspace((unsigned char)*ntok)) ntok++;
                                        char *asp = strstr(ntok, " as ");
                                        if (asp) {
                                            char orig[128]={0}, alias[128]={0};
                                            size_t ol=(size_t)(asp-ntok);
                                            while(ol>0&&isspace((unsigned char)ntok[ol-1]))ol--;
                                            memcpy(orig,ntok,ol<127?ol:127);
                                            const char *ap=asp+4;
                                            while(*ap&&isspace((unsigned char)*ap))ap++;
                                            size_t al=strlen(ap);
                                            while(al>0&&isspace((unsigned char)ap[al-1]))al--;
                                            memcpy(alias,ap,al<127?al:127);
                                            sb_fmt(&out,"var %s=__r_%s.%s;",alias,def_name,orig);
                                        } else {
                                            size_t tl=strlen(ntok);
                                            while(tl>0&&isspace((unsigned char)ntok[tl-1]))tl--;
                                            ntok[tl]='\0';
                                            if(*ntok) sb_fmt(&out,"var %s=__r_%s.%s;",ntok,def_name,ntok);
                                        }
                                        ntok = strtok(NULL, ",");
                                    }
                                } else {
                                    /* Fallback: just emit as-is */
                                    sb_fmt(&out, "var %s=__r_%s", named, def_name);
                                }
                            } else {
                                sb_fmt(&out,
                                    "var __rd_=require('%s');var %s=__rd_.default||__rd_",
                                    id_buf, bind);
                            }
                        }
                        sb_free(&binding);
                    }
                }
                sb_ch(&out, ';');
                p = end;
                /* Add newline if original had one */
                if (p < len && src[p] == '\n') { sb_ch(&out, '\n'); p++; }
                continue;
            }
        }

        /* ── export ──────────────────────────────────────────── */
        if (c == 'e' && p+6 <= len && strncmp(src+p, "export", 6) == 0 &&
            (p == 0 || !is_id_cont(src[p-1])) &&
            (p+6 < len && !is_id_cont(src[p+6]))) {

            size_t ep = p + 6;
            ep = skip_wsn(src, ep, len);

            /* export default */
            if (strncmp(src+ep, "default", 7) == 0 && !is_id_cont(src[ep+7])) {
                ep += 7;
                /* Copy the expression after 'default' all the way to semicolon/EOF,
                   respecting depth */
                ep = skip_ws(src, ep, len);
                SBuf expr; sb_init(&expr);
                /* If it starts with function/class, grab to end of block */
                if (strncmp(src+ep, "function", 8) == 0 || strncmp(src+ep, "class", 5) == 0) {
                    /* find the matching closing brace */
                    int d = 0; size_t start = ep; int found = 0;
                    while (ep < len) {
                        if (src[ep] == '{') { d++; sb_ch(&expr, src[ep]); ep++; continue; }
                        if (src[ep] == '}') {
                            d--;
                            sb_ch(&expr, src[ep]); ep++;
                            if (d == 0) { found = 1; break; }
                            continue;
                        }
                        sb_ch(&expr, src[ep]); ep++;
                    }
                    (void)start; (void)found;
                } else {
                    while (ep < len && src[ep] != ';' && src[ep] != '\n') {
                        sb_ch(&expr, src[ep]); ep++;
                    }
                    if (ep < len && src[ep] == ';') ep++;
                }
                sb_fmt(&out, "exports.default=exports.__default=%s;", expr.buf ? expr.buf : "");
                sb_free(&expr);
                p = ep;
                if (p < len && src[p] == '\n') { sb_ch(&out, '\n'); p++; }
                continue;
            }

            /* export * from './x' */
            if (src[ep] == '*') {
                ep++;
                ep = skip_wsn(src, ep, len);
                if (strncmp(src+ep, "from", 4) == 0) {
                    ep += 4;
                    ep = skip_wsn(src, ep, len);
                    char *spec = read_quoted(src, &ep, len);
                    if (spec) {
                        char id_buf[PATH_MAX];
                        resolve_id_in_graph(g, m->abs_path, spec, id_buf, PATH_MAX);
                        free(spec);
                        sb_fmt(&out, "Object.assign(exports,require('%s'));", id_buf);
                        while (ep < len && src[ep] != '\n') ep++;
                        p = ep;
                        if (p < len && src[p] == '\n') { sb_ch(&out, '\n'); p++; }
                        continue;
                    }
                }
            }

            /* export { a, b } or export { a } from './x' */
            if (src[ep] == '{') {
                /* collect names */
                ep++;
                SBuf names; sb_init(&names);
                while (ep < len && src[ep] != '}') {
                    sb_ch(&names, src[ep]); ep++;
                }
                if (ep < len) ep++; /* skip } */
                ep = skip_ws(src, ep, len);

                char *from_spec = NULL;
                if (strncmp(src+ep, "from", 4) == 0 && !is_id_cont(src[ep+4])) {
                    ep += 4;
                    ep = skip_wsn(src, ep, len);
                    from_spec = read_quoted(src, &ep, len);
                }
                while (ep < len && src[ep] != ';' && src[ep] != '\n') ep++;
                if (ep < len && src[ep] == ';') ep++;

                char id_buf[PATH_MAX] = {0};
                if (from_spec) {
                    resolve_id_in_graph(g, m->abs_path, from_spec, id_buf, PATH_MAX);
                    free(from_spec);
                }

                /* Emit exports for each name: a as b → exports.b = __rx.a */
                char *ns = names.buf ? names.buf : (char*)"";
                if (from_spec || id_buf[0]) {
                    sb_fmt(&out, "{var __rx=require('%s');", id_buf);
                }
                /* tokenize the names list */
                char *nc = strdup(ns);
                char *tok = strtok(nc, ",");
                while (tok) {
                    while (*tok && isspace((unsigned char)*tok)) tok++;
                    /* Check for 'X as Y' */
                    char *as_ptr = strstr(tok, " as ");
                    if (!as_ptr) as_ptr = strstr(tok, "\tas");
                    if (as_ptr) {
                        char orig[128] = {0}, alias[128] = {0};
                        size_t ol = (size_t)(as_ptr - tok);
                        while (ol > 0 && isspace((unsigned char)tok[ol-1])) ol--;
                        memcpy(orig, tok, ol < 127 ? ol : 127);
                        const char *a = as_ptr + 4;
                        while (*a && isspace((unsigned char)*a)) a++;
                        size_t al = strlen(a);
                        while (al > 0 && isspace((unsigned char)a[al-1])) al--;
                        memcpy(alias, a, al < 127 ? al : 127);
                        if (strcmp(alias, "default") == 0)
                            sb_fmt(&out, "exports.default=exports.__default=%s%s;",
                                   id_buf[0] ? "__rx." : "", orig);
                        else if (id_buf[0])
                            sb_fmt(&out, "exports.%s=__rx.%s;", alias, orig);
                        else
                            sb_fmt(&out, "exports.%s=%s;", alias, orig);
                    } else {
                        /* trim trailing whitespace */
                        size_t tl = strlen(tok);
                        while (tl > 0 && isspace((unsigned char)tok[tl-1])) tl--;
                        tok[tl] = '\0';
                        if (*tok) {
                            if (id_buf[0])
                                sb_fmt(&out, "exports.%s=__rx.%s;", tok, tok);
                            else
                                sb_fmt(&out, "exports.%s=%s;", tok, tok);
                        }
                    }
                    tok = strtok(NULL, ",");
                }
                free(nc);
                sb_free(&names);
                if (id_buf[0]) sb_str(&out, "}");
                p = ep;
                if (p < len && src[p] == '\n') { sb_ch(&out, '\n'); p++; }
                continue;
            }

            /* export const/let/var X = ... */
            if (strncmp(src+ep, "const", 5)==0 || strncmp(src+ep, "let", 3)==0 ||
                strncmp(src+ep, "var", 3)==0) {
                size_t kw_start = ep;
                while (ep < len && !isspace((unsigned char)src[ep])) ep++;
                const char *kw_end = src + ep;
                /* Extract variable name */
                ep = skip_ws(src, ep, len);
                size_t name_start = ep;
                while (ep < len && is_id_cont(src[ep])) ep++;
                char name[256] = {0};
                size_t nl = ep - name_start;
                if (nl >= 256) nl = 255;
                memcpy(name, src+name_start, nl);

                /* Copy the rest up to semicolon or end-of-block */
                SBuf rest; sb_init(&rest);
                int depth = 0;
                while (ep < len) {
                    if (src[ep] == '{' || src[ep] == '(' || src[ep] == '[') depth++;
                    if (src[ep] == '}' || src[ep] == ')' || src[ep] == ']') depth--;
                    if (depth == 0 && src[ep] == ';') { sb_ch(&rest, src[ep]); ep++; break; }
                    if (depth < 0) break; /* shouldn't happen */
                    sb_ch(&rest, src[ep]); ep++;
                }
                /* Emit: const/let/var X = ...; exports.X = X */
                sb_app(&out, src+kw_start, (size_t)(kw_end-src-kw_start));
                sb_ch(&out, ' ');
                sb_str(&out, name);
                sb_str(&out, rest.buf ? rest.buf : ";");
                sb_fmt(&out, "\nexports.%s=%s;", name, name);
                sb_free(&rest);
                p = ep;
                if (p < len && src[p] == '\n') { sb_ch(&out, '\n'); p++; }
                continue;
            }

            /* export function foo() {} or export class Foo {} */
            if (strncmp(src+ep, "function", 8)==0 || strncmp(src+ep, "class", 5)==0) {
                int is_fn = (src[ep] == 'f');
                size_t kw_start = ep;
                /* Skip keyword */
                while (ep < len && !isspace((unsigned char)src[ep]) && src[ep]!='(') ep++;
                ep = skip_ws(src, ep, len);
                /* Extract name */
                size_t name_start = ep;
                while (ep < len && is_id_cont(src[ep])) ep++;
                char name[256] = {0};
                size_t nl = ep - name_start;
                if (nl >= 256) nl = 255;
                memcpy(name, src+name_start, nl);
                /* Copy to end of block */
                int depth = 0;
                int in_block = 0;
                while (ep < len) {
                    if (src[ep] == '{') { depth++; in_block = 1; }
                    if (src[ep] == '}') { depth--; ep++; if (in_block && depth == 0) break; continue; }
                    else ep++;
                }
                /* Emit: function foo() { ... }\nexports.foo = foo */
                sb_app(&out, src+kw_start, ep - kw_start);
                sb_fmt(&out, "\nexports.%s=%s;", name, name);
                (void)is_fn;
                p = ep;
                if (p < len && src[p] == '\n') { sb_ch(&out, '\n'); p++; }
                continue;
            }
        }

        /* Default: copy character verbatim */
        sb_ch(&out, c);
        p++;
    }

    return out.buf; /* caller owns */
}

/* ═══════════════════════════════════════════════════════════════
   §9  Bundle output
   ═══════════════════════════════════════════════════════════════ */

static const char PREAMBLE[] =
"// ⚡ Generated by sofuu bundle — https://sofuu.dev\n"
"// Run with: sofuu run <this_file>\n"
"(function(){\n"
"'use strict';\n"
"var __m={};\n"
"var __f={};\n"
"function __d(id,factory){__f[id]=factory;}\n"
"function __r(id){\n"
"  if(Object.prototype.hasOwnProperty.call(__m,id))return __m[id];\n"
"  var exp={};\n"
"  __m[id]=exp;\n"
"  if(!__f[id]){console.error('[sofuu bundle] missing module:',id);return exp;}\n"
"  __f[id](exp,__r);\n"
"  return __m[id];\n"
"}\n";

static int write_bundle(Graph *g, FILE *out_f) {
    fputs(PREAMBLE, out_f);

    /* Modules in reverse BFS order (dependencies before dependents) */
    for (int i = g->count - 1; i >= 0; i--) {
        BundleMod *m = g->mods[i];
        if (!m->transformed) continue;
        fprintf(out_f, "\n/* === module: %s === */\n", m->id);
        fprintf(out_f, "__d('%s',function(exports,require){\n", m->id);
        fputs(m->transformed, out_f);
        fputs("\n});\n", out_f);
    }

    /* Kick off the entry module */
    fprintf(out_f, "\n/* === entry === */\n");
    fprintf(out_f, "var __entry=__r('%s');\n", g->entry_id);
    fprintf(out_f, "if(typeof __entry==='function')__entry();\n");
    fputs("})();\n", out_f);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
   §10  Public API
   ═══════════════════════════════════════════════════════════════ */

int sofuu_bundle(const char *entry_path, const char *output_path, int minify) {
    (void)minify; /* future: whitespace collapsing */

    /* Resolve entry to absolute path */
    char entry_abs[PATH_MAX];
    if (!realpath(entry_path, entry_abs)) {
        fprintf(stderr, "\033[31m✗\033[0m Cannot find entry: %s\n", entry_path);
        return 1;
    }

    printf("\n\033[1m⚡ sofuu bundle\033[0m\n");
    printf("  Entry: %s\n\n", entry_abs);

    /* Build module graph */
    Graph g;
    memset(&g, 0, sizeof(g));

    printf("  \033[90m+\033[0m %s\n", entry_abs);
    if (!build_graph(&g, entry_abs)) {
        fprintf(stderr, "\033[31m✗\033[0m Graph build failed\n");
        graph_free(&g);
        return 1;
    }

    printf("\n  Collected \033[36m%d\033[0m module(s). Transforming...\n", g.count);

    /* Transform each module */
    for (int i = 0; i < g.count; i++) {
        BundleMod *m = g.mods[i];
        if (!m->source) continue;

        /* Strip TypeScript if needed */
        const char *src = m->source;
        char *stripped = NULL;
        size_t slen = strlen(m->abs_path);
        if (slen >= 3 && strcmp(m->abs_path + slen - 3, ".ts") == 0) {
            size_t out_len = 0;
            stripped = ts_strip(src, strlen(src), &out_len);
            if (stripped) src = stripped;
        }

        m->transformed = transform_module(&g, m, src);
        free(stripped);
    }

    /* Write output */
    FILE *out_f = fopen(output_path, "w");
    if (!out_f) {
        fprintf(stderr, "\033[31m✗\033[0m Cannot write: %s\n", output_path);
        graph_free(&g);
        return 1;
    }

    write_bundle(&g, out_f);
    fclose(out_f);

    /* Report */
    struct stat st;
    stat(output_path, &st);
    long sz = (long)st.st_size;
    printf("\n  \033[32m✓\033[0m Bundle written: \033[1m%s\033[0m", output_path);
    if (sz < 1024)        printf(" (%ld B)\n", sz);
    else if (sz < 1<<20)  printf(" (%.1f KB)\n", sz / 1024.0);
    else                  printf(" (%.1f MB)\n", sz / 1048576.0);
    printf("  \033[90mRun with: sofuu run %s\033[0m\n\n", output_path);

    graph_free(&g);
    return 0;
}
