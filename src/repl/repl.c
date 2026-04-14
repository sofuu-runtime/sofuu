/*
 * src/repl/repl.c — Sofuu Interactive REPL
 *
 * Features:
 *   - readline integration (history, arrows, Ctrl-C, Ctrl-D)
 *   - Multi-line detection (unbalanced braces/parens/brackets/templates)
 *   - Coloured output: numbers=yellow, strings=green, errors=red
 *   - Persistent history in ~/.sofuu_history
 *   - .help  .clear  .exit  .version  dot-commands
 *   - Persistent JS state across lines (let/const/var all survive)
 */
#include "repl.h"
#include "engine.h"
#include "sofuu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ── Readline (optional) ─────────────────────────────────── */
#if defined(SOFUU_HAVE_READLINE)
#  include <readline/readline.h>
#  include <readline/history.h>
#  define HAVE_RL 1
#else
#  define HAVE_RL 0
#endif

/* ── ANSI colours ────────────────────────────────────────── */
#define RST  "\033[0m"
#define CYN  "\033[36m"
#define BLD  "\033[1m"
#define DIM  "\033[2m"
#define GRN  "\033[32m"
#define RED  "\033[31m"
#define YLW  "\033[33m"

#define SOFUU_VERSION "0.1.0-alpha"
#define MAX_ACCUM     65536

/* ── Banner ──────────────────────────────────────────────── */
static void print_repl_banner(void) {
    printf("\n");
    printf(CYN BLD "  ⚡ Sofuu (素風) REPL" RST "  " DIM "v%s" RST "\n", SOFUU_VERSION);
    printf(DIM   "  QuickJS · libuv · AI native · NEON/AVX2 SIMD\n" RST);
    printf(DIM   "  Type " RST BLD ".help" RST DIM " for commands, " RST BLD "Ctrl-D" RST DIM " to exit\n" RST);
    printf("\n");
}

/* ── Multi-line balance detection ────────────────────────── */
static int needs_more(const char *line) {
    int br = 0, par = 0, sqb = 0;
    int sq = 0, dq = 0, tmpl = 0;
    size_t len = strlen(line);
    for (size_t i = 0; i < len; i++) {
        char c = line[i];
        if ((sq || dq) && c == '\\') { i++; continue; }
        if (c == '\'' && !dq && !tmpl) { sq   = !sq;   continue; }
        if (c == '"'  && !sq && !tmpl) { dq   = !dq;   continue; }
        if (c == '`'  && !sq && !dq)   { tmpl = !tmpl; continue; }
        if (sq || dq || tmpl) continue;
        if (c == '/' && i+1 < len && line[i+1] == '/') break;
        if (c == '{') br++;  else if (c == '}') br--;
        if (c == '(') par++; else if (c == ')') par--;
        if (c == '[') sqb++; else if (c == ']') sqb--;
    }
    return br > 0 || par > 0 || sqb > 0 || tmpl;
}

/* ── Read one line ───────────────────────────────────────── */
static char *read_line(const char *prompt) {
#if HAVE_RL
    char *line = readline(prompt);
    if (!line) return NULL;
    if (*line) add_history(line);
    return line;
#else
    printf("%s", prompt);
    fflush(stdout);
    char *buf = NULL;
    size_t cap = 0;
    ssize_t n = getline(&buf, &cap, stdin);
    if (n < 0) { free(buf); return NULL; }
    if (n > 0 && buf[n-1] == '\n') buf[n-1] = '\0';
    return buf;
#endif
}

/* ── Dot-commands ────────────────────────────────────────── */
static int handle_dot(const char *cmd) {
    if (!strcmp(cmd, ".exit") || !strcmp(cmd, ".quit")) {
        printf("\n  Bye! 👋\n\n");
        exit(0);
    }
    if (!strcmp(cmd, ".help")) {
        printf(
            "\n" BLD "  Commands:\n" RST
            "    " CYN ".help" RST "     — this help\n"
            "    " CYN ".clear" RST "    — clear screen\n"
            "    " CYN ".exit" RST "     — quit REPL\n"
            "    " CYN ".version" RST "  — Sofuu version\n"
            "\n" BLD "  Shortcuts:\n" RST
            "    " CYN "Ctrl-D" RST "    — exit\n"
            "    " CYN "Ctrl-C" RST "    — cancel input\n"
            "    " CYN "↑ / ↓" RST "    — history\n"
            "\n" BLD "  Example:\n" RST
            "    > const x = [1,2,3]\n"
            "    > x.map(n => n * 2)\n"
            "    ← [2,4,6]\n\n"
        );
        return 1;
    }
    if (!strcmp(cmd, ".clear")) {
        printf("\033[2J\033[H");
        print_repl_banner();
        return 1;
    }
    if (!strcmp(cmd, ".version")) {
        printf("  sofuu " YLW SOFUU_VERSION RST "\n\n");
        return 1;
    }
    return 0;
}

/* ── REPL entry point ────────────────────────────────────── */
int sofuu_repl(void) {
    print_repl_banner();

    SofuuRuntime *rt = sofuu_init();
    if (!rt) {
        fprintf(stderr, RED "Fatal:" RST " runtime init failed\n");
        return 1;
    }
    SofuuEngine *eng = (SofuuEngine *)sofuu_get_engine(rt);

#if HAVE_RL
    char hist[512];
    const char *home = getenv("HOME");
    snprintf(hist, sizeof(hist), "%s/.sofuu_history", home ? home : ".");
    read_history(hist);
    rl_bind_key('\t', rl_insert);
#endif

    char accum[MAX_ACCUM];
    accum[0] = '\0';

    for (;;) {
        /* Dynamic prompt */
        const char *prompt = accum[0] ? DIM "… " RST : CYN "> " RST;
        char *line = read_line(prompt);

        if (!line) { printf("\n"); break; }          /* Ctrl-D */

        /* Trim trailing whitespace */
        size_t l = strlen(line);
        while (l > 0 && (line[l-1] == ' ' || line[l-1] == '\t')) line[--l] = '\0';

        /* Dot-commands (only valid at the start of fresh input) */
        if (line[0] == '.' && accum[0] == '\0') {
            if (handle_dot(line)) { free(line); continue; }
            printf(RED "  ? Unknown: %s" RST "\n\n", line);
            free(line); continue;
        }

        /* Accumulate */
        size_t alen = strlen(accum);
        if (alen > 0) {
            accum[alen] = '\n'; accum[alen+1] = '\0';
        }
        strncat(accum, line, MAX_ACCUM - strlen(accum) - 1);
        free(line);

        /* Wait for more on unbalanced input */
        if (needs_more(accum)) continue;

        /* Eval */
        int is_err = 0;
        char *result = engine_eval_repl(eng, accum, &is_err);
        accum[0] = '\0';

        if (result) {
            /* Only show arrow for non-undefined results */
            int is_undef = !is_err && strstr(result, "undefined") != NULL
                           && strlen(result) < 30;
            if (!is_undef) {
                printf("%s%s%s\n",
                       is_err ? RED "← " RST : DIM "← " RST,
                       result,
                       RST);
            }
            free(result);
        }
        printf("\n");
    }

#if HAVE_RL
    write_history(hist);
#endif

    sofuu_destroy(rt);
    return 0;
}
