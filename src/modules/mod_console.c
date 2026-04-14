/*
 * mod_console.c — console.log, console.error, console.warn, console.info,
 *                 console.time, console.timeEnd, console.table, console.assert
 */
#include "mod_console.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static const char *js_val_to_str(JSContext *ctx, JSValue val, int *needs_free) {
    *needs_free = 1;
    if (JS_IsUndefined(val)) { *needs_free = 0; return "undefined"; }
    if (JS_IsNull(val))      { *needs_free = 0; return "null"; }
    if (JS_IsBool(val))      { *needs_free = 0; return JS_ToBool(ctx, val) ? "true" : "false"; }
    if (JS_IsObject(val) || JS_IsArray(ctx, val)) {
        JSValue global    = JS_GetGlobalObject(ctx);
        JSValue json      = JS_GetPropertyStr(ctx, global, "JSON");
        JSValue stringify = JS_GetPropertyStr(ctx, json, "stringify");
        JSValue args[3];
        args[0] = val;
        args[1] = JS_UNDEFINED;
        args[2] = JS_NewInt32(ctx, 2);
        JSValue result = JS_Call(ctx, stringify, json, 3, args);
        JS_FreeValue(ctx, args[2]);
        JS_FreeValue(ctx, stringify);
        JS_FreeValue(ctx, json);
        JS_FreeValue(ctx, global);
        if (JS_IsException(result)) {
            JS_FreeValue(ctx, result);
            *needs_free = 0;
            return "[object Object]";
        }
        const char *s = JS_ToCString(ctx, result);
        JS_FreeValue(ctx, result);
        return s;
    }
    return JS_ToCString(ctx, val);
}

static JSValue console_print(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv,
                              FILE *out, const char *ansi_prefix,
                              const char *ansi_reset) {
    (void)this_val;
    if (ansi_prefix && *ansi_prefix) fputs(ansi_prefix, out);
    for (int i = 0; i < argc; i++) {
        if (i > 0) fputc(' ', out);
        int needs_free;
        const char *s = js_val_to_str(ctx, argv[i], &needs_free);
        fputs(s, out);
        if (needs_free) JS_FreeCString(ctx, s);
    }
    if (ansi_reset && *ansi_reset) fputs(ansi_reset, out);
    fputc('\n', out);
    fflush(out);
    return JS_UNDEFINED;
}

/* ------------------------------------------------------------------ */
/* console.log / info / warn / error / assert                           */
/* ------------------------------------------------------------------ */

static JSValue js_console_log(JSContext *ctx, JSValueConst this_val,
                               int argc, JSValueConst *argv) {
    return console_print(ctx, this_val, argc, argv, stdout, "", "");
}
static JSValue js_console_info(JSContext *ctx, JSValueConst this_val,
                                int argc, JSValueConst *argv) {
    return console_print(ctx, this_val, argc, argv, stdout, "\033[36m", "\033[0m");
}
static JSValue js_console_warn(JSContext *ctx, JSValueConst this_val,
                                int argc, JSValueConst *argv) {
    return console_print(ctx, this_val, argc, argv, stderr, "\033[33m", "\033[0m");
}
static JSValue js_console_error(JSContext *ctx, JSValueConst this_val,
                                 int argc, JSValueConst *argv) {
    return console_print(ctx, this_val, argc, argv, stderr, "\033[31m", "\033[0m");
}
static JSValue js_console_assert(JSContext *ctx, JSValueConst this_val,
                                  int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1 || !JS_ToBool(ctx, argv[0])) {
        const char *msg = (argc >= 2) ? JS_ToCString(ctx, argv[1]) : NULL;
        fprintf(stderr, "\033[31mAssertion failed:\033[0m %s\n",
                msg ? msg : "console.assert");
        if (msg) JS_FreeCString(ctx, msg);
    }
    return JS_UNDEFINED;
}

/* ------------------------------------------------------------------ */
/* console.time / console.timeEnd — real wall-clock timers              */
/* ------------------------------------------------------------------ */

#define TIMER_MAX 64
typedef struct { char label[128]; struct timespec ts; int active; } TimerEntry;
static TimerEntry s_timers[TIMER_MAX];

static JSValue js_console_time(JSContext *ctx, JSValueConst this_val,
                                int argc, JSValueConst *argv) {
    (void)this_val;
    const char *label = (argc > 0) ? JS_ToCString(ctx, argv[0]) : NULL;
    const char *key   = label ? label : "default";
    for (int i = 0; i < TIMER_MAX; i++) {
        if (!s_timers[i].active) {
            s_timers[i].active = 1;
            strncpy(s_timers[i].label, key, 127);
            s_timers[i].label[127] = '\0';
            clock_gettime(CLOCK_MONOTONIC, &s_timers[i].ts);
            break;
        }
    }
    if (label) JS_FreeCString(ctx, label);
    return JS_UNDEFINED;
}

static JSValue js_console_timeEnd(JSContext *ctx, JSValueConst this_val,
                                   int argc, JSValueConst *argv) {
    (void)this_val;
    const char *label = (argc > 0) ? JS_ToCString(ctx, argv[0]) : NULL;
    const char *key   = label ? label : "default";
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    for (int i = 0; i < TIMER_MAX; i++) {
        if (s_timers[i].active && strcmp(s_timers[i].label, key) == 0) {
            long long us =
                (long long)(now.tv_sec  - s_timers[i].ts.tv_sec)  * 1000000LL +
                (long long)(now.tv_nsec - s_timers[i].ts.tv_nsec) / 1000LL;
            fprintf(stdout, "%s: %.3fms\n", key, us / 1000.0);
            fflush(stdout);
            s_timers[i].active = 0;
            break;
        }
    }
    if (label) JS_FreeCString(ctx, label);
    return JS_UNDEFINED;
}

/* ------------------------------------------------------------------ */
/* console.table — renders an array of objects as an ASCII table        */
/* ------------------------------------------------------------------ */

#define TBL_MAX_COLS  16
#define TBL_MAX_ROWS 256

static void tbl_sep(int *w, int nc) {
    fputc('+', stdout);
    for (int c = 0; c < nc; c++) {
        for (int i = 0; i < w[c] + 2; i++) fputc('-', stdout);
        fputc('+', stdout);
    }
    fputc('\n', stdout);
}

static JSValue js_console_table(JSContext *ctx, JSValueConst this_val,
                                 int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1 || !JS_IsArray(ctx, argv[0]))
        return console_print(ctx, this_val, argc, argv, stdout, "", "");

    JSValue arr = argv[0];
    char col_keys[TBL_MAX_COLS][128];
    int  ncols = 0;

    /* Collect column names from first element */
    JSValue row0 = JS_GetPropertyUint32(ctx, arr, 0);
    if (JS_IsObject(row0)) {
        JSPropertyEnum *props; uint32_t nprops;
        if (JS_GetOwnPropertyNames(ctx, &props, &nprops, row0,
                                   JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY) == 0) {
            for (uint32_t p = 0; p < nprops && ncols < TBL_MAX_COLS; p++) {
                const char *k = JS_AtomToCString(ctx, props[p].atom);
                if (k) { strncpy(col_keys[ncols], k, 127); col_keys[ncols][127]='\0'; ncols++; JS_FreeCString(ctx, k); }
                JS_FreeAtom(ctx, props[p].atom);
            }
            js_free(ctx, props);
        }
    }
    JS_FreeValue(ctx, row0);
    if (ncols == 0)
        return console_print(ctx, this_val, argc, argv, stdout, "", "");

    /* Collect row count */
    JSValue lv = JS_GetPropertyStr(ctx, arr, "length");
    int32_t nrows = 0;
    JS_ToInt32(ctx, &nrows, lv);
    JS_FreeValue(ctx, lv);
    if (nrows > TBL_MAX_ROWS) nrows = TBL_MAX_ROWS;

    /* Column widths (at least as wide as the header) */
    int widths[TBL_MAX_COLS];
    for (int c = 0; c < ncols; c++) widths[c] = (int)strlen(col_keys[c]);

    /* Collect all cells */
    char *cells[TBL_MAX_ROWS][TBL_MAX_COLS];
    for (int r = 0; r < nrows; r++) {
        JSValue row = JS_GetPropertyUint32(ctx, arr, (uint32_t)r);
        for (int c = 0; c < ncols; c++) {
            JSValue cell = JS_GetPropertyStr(ctx, row, col_keys[c]);
            int nf; const char *s = js_val_to_str(ctx, cell, &nf);
            cells[r][c] = strdup(s ? s : "");
            if (nf && s) JS_FreeCString(ctx, s);
            JS_FreeValue(ctx, cell);
            int l = (int)strlen(cells[r][c]);
            if (l > widths[c]) widths[c] = l;
        }
        JS_FreeValue(ctx, row);
    }

    /* Render */
    tbl_sep(widths, ncols);
    fputc('|', stdout);
    for (int c = 0; c < ncols; c++)
        fprintf(stdout, " \033[1m%-*s\033[0m |", widths[c], col_keys[c]);
    fputc('\n', stdout);
    tbl_sep(widths, ncols);
    for (int r = 0; r < nrows; r++) {
        fputc('|', stdout);
        for (int c = 0; c < ncols; c++) {
            fprintf(stdout, " %-*s |", widths[c], cells[r][c]);
            free(cells[r][c]);
        }
        fputc('\n', stdout);
    }
    tbl_sep(widths, ncols);
    fflush(stdout);
    return JS_UNDEFINED;
}

/* ------------------------------------------------------------------ */
/* Registration                                                         */
/* ------------------------------------------------------------------ */

void mod_console_register(JSContext *ctx) {
    JSValue global  = JS_GetGlobalObject(ctx);
    JSValue console = JS_NewObject(ctx);

    JS_SetPropertyStr(ctx, console, "log",     JS_NewCFunction(ctx, js_console_log,     "log",     0));
    JS_SetPropertyStr(ctx, console, "info",    JS_NewCFunction(ctx, js_console_info,    "info",    0));
    JS_SetPropertyStr(ctx, console, "warn",    JS_NewCFunction(ctx, js_console_warn,    "warn",    0));
    JS_SetPropertyStr(ctx, console, "error",   JS_NewCFunction(ctx, js_console_error,   "error",   0));
    JS_SetPropertyStr(ctx, console, "assert",  JS_NewCFunction(ctx, js_console_assert,  "assert",  1));
    JS_SetPropertyStr(ctx, console, "time",    JS_NewCFunction(ctx, js_console_time,    "time",    1));
    JS_SetPropertyStr(ctx, console, "timeEnd", JS_NewCFunction(ctx, js_console_timeEnd, "timeEnd", 1));
    JS_SetPropertyStr(ctx, console, "table",   JS_NewCFunction(ctx, js_console_table,   "table",   1));

    JS_SetPropertyStr(ctx, global, "console", console);
    JS_FreeValue(ctx, global);
}
