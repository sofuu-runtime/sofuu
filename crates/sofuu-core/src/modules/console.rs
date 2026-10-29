// modules/console.rs — console.log/info/warn/error/assert/time/timeEnd/
// table (PLAN-RUST-MIGRATION M2).
//
// Port of the deleted `src/modules/mod_console.c`, semantics verbatim —
// including the exact ANSI prefixes, JSON.stringify-with-indent-2 rendering
// for objects, and the ASCII table format for console.table.
//
// C symbol replaced: `mod_console_register` (engine.c calls it unchanged).

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_int, c_void};
use std::io::Write;
use std::ptr;
use std::time::Instant;

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

use crate::embed_config;

/// Render a JS value the way the C port did:
///   undefined/null/bool → literal text; object/array → JSON.stringify with
///   indent 2 (falling back to "[object Object]" on exception); everything
///   else → JS_ToCString (NULL → "<unprintable>" handled by the caller).
///
/// # Safety
/// `ctx` valid; `val` a live value of `ctx`.
unsafe fn js_val_to_str(ctx: *mut JSContext, val: JSValueConst) -> Option<String> {
    // SAFETY: tag checks via shims.
    if qjs::is_undefined(val) {
        return Some("undefined".to_string());
    }
    if qjs::is_null(val) {
        return Some("null".to_string());
    }
    // Booleans and numbers fall through to JS_ToCString below ("true" /
    // "false" / decimal), exactly like the C port's JS_ToCString tail.
    if qjs::is_object(val) {
        /* C: walk global.JSON.stringify(val, undefined, 2) */
        let global = qjs::sofuu_js_get_global_object(ctx);
        let json = qjs::sofuu_js_get_property_str(ctx, global, c"JSON".as_ptr());
        let stringify = qjs::sofuu_js_get_property_str(ctx, json, c"stringify".as_ptr());
        let mut args: [JSValueConst; 3] = [
            val,
            qjs::sofuu_js_undefined(),
            qjs::sofuu_js_new_int32(ctx, 2),
        ];
        let result = qjs::JS_Call(ctx, stringify, json, 3, args.as_mut_ptr());
        qjs::sofuu_js_free_value(ctx, args[2]);
        qjs::sofuu_js_free_value(ctx, stringify);
        qjs::sofuu_js_free_value(ctx, json);
        qjs::sofuu_js_free_value(ctx, global);
        if qjs::is_exception(result) {
            qjs::sofuu_js_free_value(ctx, result);
            return Some("[object Object]".to_string());
        }
        let s = qjs::sofuu_js_to_cstring(ctx, result);
        let out = if s.is_null() {
            None
        } else {
            Some(CStr::from_ptr(s).to_string_lossy().into_owned())
        };
        qjs::sofuu_js_free_value(ctx, result);
        if !s.is_null() {
            qjs::sofuu_js_free_cstring(ctx, s);
        }
        return out;
    }
    let s = qjs::sofuu_js_to_cstring(ctx, val);
    if s.is_null() {
        return None; /* e.g. Symbol — caller substitutes "<unprintable>" */
    }
    let out = Some(CStr::from_ptr(s).to_string_lossy().into_owned());
    qjs::sofuu_js_free_cstring(ctx, s);
    out
}

/// Render one argument with the console's existing rules (JSON for objects,
/// String() for primitives, "<unprintable>" for Symbols).
///
/// # Safety
/// `ctx` valid; `val` a live value of `ctx`.
unsafe fn arg_render(ctx: *mut JSContext, val: JSValueConst) -> String {
    // SAFETY: val is a live value.
    match unsafe { js_val_to_str(ctx, val) } {
        Some(s) => s,
        None => "<unprintable>".to_string(),
    }
}

/// Numeric conversion for `%d`/`%i`/`%f` that never leaves a pending
/// exception behind (a Symbol argument throws in JS_ToFloat64; console.log
/// must swallow that, not poison the context).
///
/// # Safety
/// `ctx` valid; `val` a live value of `ctx`.
unsafe fn arg_number(ctx: *mut JSContext, val: JSValueConst) -> f64 {
    let mut d = 0f64;
    // SAFETY: val is a live value; JS_ToFloat64 writes d on success.
    if unsafe { qjs::JS_ToFloat64(ctx, &mut d, val) } != 0 {
        // SAFETY: clears the conversion exception the failed call threw.
        let exc = unsafe { qjs::sofuu_js_get_exception(ctx) };
        unsafe { qjs::sofuu_js_free_value(ctx, exc) };
        return f64::NAN;
    }
    d
}

/// `%j` — compact JSON, like Node's util.format. `undefined` results (a
/// function, a symbol) print as `undefined`; a value JSON cannot serialise
/// (a cycle, a BigInt) prints as `[Circular]`, as Node does.
///
/// # Safety
/// `ctx` valid; `val` a live value of `ctx`.
unsafe fn arg_json(ctx: *mut JSContext, val: JSValueConst) -> String {
    // SAFETY: every value is freed exactly once below.
    let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
    let json = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"JSON".as_ptr()) };
    let stringify = unsafe { qjs::sofuu_js_get_property_str(ctx, json, c"stringify".as_ptr()) };
    let result = unsafe {
        qjs::JS_Call(ctx, stringify, json, 1, &val as *const JSValueConst)
    };
    unsafe {
        qjs::sofuu_js_free_value(ctx, stringify);
        qjs::sofuu_js_free_value(ctx, json);
        qjs::sofuu_js_free_value(ctx, global);
    }
    if unsafe { qjs::is_exception(result) } {
        // Clear the JSON.stringify throw (cycle / BigInt) — console.log must
        // never leave a pending exception behind.
        let pending = unsafe { qjs::sofuu_js_get_exception(ctx) };
        unsafe { qjs::sofuu_js_free_value(ctx, pending) };
        unsafe { qjs::sofuu_js_free_value(ctx, result) };
        return "[Circular]".to_string();
    }
    if unsafe { qjs::is_undefined(result) } {
        unsafe { qjs::sofuu_js_free_value(ctx, result) };
        return "undefined".to_string();
    }
    let s = unsafe { qjs::sofuu_js_to_cstring(ctx, result) };
    let out = if s.is_null() {
        "[Circular]".to_string()
    } else {
        // SAFETY: s is NUL-terminated and freed below.
        let text = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
        unsafe { qjs::sofuu_js_free_cstring(ctx, s) };
        text
    };
    unsafe { qjs::sofuu_js_free_value(ctx, result) };
    out
}

/// Node-style printf substitution (R-15):
/// `console.log('%s=%d', 'fmt', 7)` printed `%s=%d fmt 7`; it now prints
/// `fmt=7`.
///
/// Supported directives: `%s` (String), `%d`/`%i` (integer), `%f` (float),
/// `%j` (JSON), `%o`/`%O` (compact JSON), `%c` (CSS — consumes an argument and
/// prints nothing, as in Node) and `%%` (literal `%`). A directive with no
/// argument left is printed verbatim; an unknown directive is left as-is.
/// Arguments consumed here are advanced in `next`; the caller appends the
/// rest space-separated, which is what makes Node's `%s %s 'a','b','c'`
/// print `a b c`.
///
/// # Safety
/// `ctx` valid; `argv` valid for `argc` entries.
unsafe fn format_substitute(
    ctx: *mut JSContext,
    fmt: &str,
    argc: c_int,
    argv: *const JSValueConst,
    next: &mut c_int,
) -> String {
    let mut out = String::with_capacity(fmt.len());
    let mut chars = fmt.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }
        let Some(&spec) = chars.peek() else {
            out.push('%');
            break;
        };
        let dir = spec.to_ascii_lowercase();
        if dir == '%' {
            chars.next();
            out.push('%');
            continue;
        }
        // An upper-case O is a distinct directive; it maps to the same
        // object rendering as %o here.
        let recognised = matches!(dir, 's' | 'd' | 'i' | 'f' | 'j' | 'o' | 'c');
        if !recognised {
            out.push('%');
            continue;
        }
        if *next >= argc {
            // No argument left: Node leaves the directive untouched.
            out.push('%');
            out.push(spec);
            chars.next();
            continue;
        }
        chars.next();
        let arg = unsafe { *argv.add(*next as usize) };
        *next += 1;
        let rendered = match dir {
            's' => unsafe { arg_render(ctx, arg) },
            'd' | 'i' => {
                let n = unsafe { arg_number(ctx, arg) };
                if n.is_nan() {
                    "NaN".to_string()
                } else if n.is_infinite() {
                    if n > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
                } else if n.abs() < 9.0e18 {
                    format!("{}", n.trunc() as i64)
                } else {
                    format!("{}", n.trunc())
                }
            }
            'f' => {
                let n = unsafe { arg_number(ctx, arg) };
                if n.is_nan() {
                    "NaN".to_string()
                } else if n.is_infinite() {
                    if n > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
                } else {
                    format!("{}", n)
                }
            }
            'j' => unsafe { arg_json(ctx, arg) },
            /* Node's %o/%O use util.inspect; this runtime has no inspect.
             * Compact single-line JSON is the closest deterministic answer:
             * it never truncates, never wraps, and round-trips. Documented
             * deviation (R-15) — `%s` keeps the human-readable path. */
            'o' => unsafe { arg_json(ctx, arg) },
            /* %O is accepted via `dir` (lower-cased) too. */
            'c' => String::new(),
            _ => unreachable!(),
        };
        out.push_str(&rendered);
    }
    out
}

/// Print like console_print() in C: [prefix] arg arg arg [reset]\n to the
/// given stream. None values become "<unprintable>". When a host log
/// callback is installed (embedded mode), the plain line is routed there
/// instead — no ANSI codes, no trailing newline (PLAN-HEADLESS H2.3).
///
/// # Safety
/// `ctx` valid; `argv` valid for `argc` entries.
unsafe fn console_print(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
    out: &mut dyn Write,
    level: &str,
    ansi_prefix: &str,
    ansi_reset: &str,
) {
    let mut body = String::new();
    // R-15: the first argument is a format string only when it IS a string
    // and there is another argument to substitute (Node's util.format rule).
    if argc > 1 && unsafe { qjs::sofuu_js_is_string(*argv) } != 0 {
        // SAFETY: argv holds argc values; every read is index-checked.
        let fmt_s = unsafe { arg_render(ctx, *argv) };
        let mut next: c_int = 1;
        body.push_str(&unsafe { format_substitute(ctx, &fmt_s, argc, argv, &mut next) });
        while next < argc {
            body.push(' ');
            body.push_str(&unsafe { arg_render(ctx, *argv.add(next as usize)) });
            next += 1;
        }
    } else {
        for i in 0..argc {
            if i > 0 {
                body.push(' ');
            }
            // SAFETY: argv valid for argc.
            match unsafe { js_val_to_str(ctx, *argv.add(i as usize)) } {
                Some(s) => body.push_str(&s),
                None => body.push_str("<unprintable>"),
            }
        }
    }
    if embed_config::log(level, &body) {
        return; /* host callback consumed it */
    }
    let wrote = write!(out, "{}{}{}\n", ansi_prefix, body, ansi_reset)
        .and_then(|()| out.flush());
    if let Err(e) = wrote {
        /* A closed output pipe (`| head`, `| grep -q`) ends the process, as it
         * always has — SIGPIPE itself is ignored process-wide so a socket peer
         * cannot kill a server (rt::broken_pipe_exit), which leaves our own
         * stream to say when the reader is gone. Swallowing this error would
         * turn the old quiet exit into an endless write into a dead pipe. */
        if e.kind() == std::io::ErrorKind::BrokenPipe {
            crate::rt::broken_pipe_exit();
        }
    }
}

unsafe extern "C" fn js_console_log(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut out = std::io::stdout().lock();
    console_print(ctx, argc, argv, &mut out, "log", "", "");
    drop(out);
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_console_info(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut out = std::io::stdout().lock();
    console_print(ctx, argc, argv, &mut out, "info", "\x1b[36m", "\x1b[0m");
    drop(out);
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_console_warn(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut out = std::io::stderr().lock();
    console_print(ctx, argc, argv, &mut out, "warn", "\x1b[33m", "\x1b[0m");
    drop(out);
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_console_error(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut out = std::io::stderr().lock();
    console_print(ctx, argc, argv, &mut out, "error", "\x1b[31m", "\x1b[0m");
    drop(out);
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_console_assert(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let cond = if argc >= 1 {
        qjs::JS_ToBool(ctx, *argv)
    } else {
        0
    };
    if cond == 0 {
        let msg = if argc >= 2 {
            let p = qjs::sofuu_js_to_cstring(ctx, *argv.add(1));
            if p.is_null() {
                None
            } else {
                let s = Some(CStr::from_ptr(p).to_string_lossy().into_owned());
                qjs::sofuu_js_free_cstring(ctx, p);
                s
            }
        } else {
            None
        };
        let body = format!(
            "Assertion failed: {}",
            msg.unwrap_or_else(|| "console.assert".to_string())
        );
        if !embed_config::log("error", &body) {
            eprintln!("\x1b[31m{}\x1b[0m", body);
        }
    }
    qjs::sofuu_js_undefined()
}

// ── console.time / timeEnd — real wall-clock timers ──────────────────

const TIMER_MAX: usize = 64;
struct TimerEntry {
    label: String,
    ts: Instant,
    active: bool,
}

thread_local! {
    static CONSOLE_TIMERS: RefCell<Vec<TimerEntry>> = const { RefCell::new(Vec::new()) };
}

unsafe extern "C" fn js_console_time(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let label = if argc > 0 {
        let p = qjs::sofuu_js_to_cstring(ctx, *argv);
        if p.is_null() {
            None
        } else {
            let s = Some(CStr::from_ptr(p).to_string_lossy().into_owned());
            qjs::sofuu_js_free_cstring(ctx, p);
            s
        }
    } else {
        None
    };
    let key = label.as_deref().unwrap_or("default");
    CONSOLE_TIMERS.with(|ts| {
        let mut ts = ts.borrow_mut();
        // C scans a 64-slot fixed array for the first free slot.
        if ts.len() < TIMER_MAX {
            if let Some(slot) = ts.iter().position(|e| !e.active) {
                ts[slot] = TimerEntry { label: key.to_string(), ts: Instant::now(), active: true };
            } else {
                ts.push(TimerEntry { label: key.to_string(), ts: Instant::now(), active: true });
            }
        }
    });
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_console_time_end(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let label = if argc > 0 {
        let p = qjs::sofuu_js_to_cstring(ctx, *argv);
        if p.is_null() {
            None
        } else {
            let s = Some(CStr::from_ptr(p).to_string_lossy().into_owned());
            qjs::sofuu_js_free_cstring(ctx, p);
            s
        }
    } else {
        None
    };
    let key = label.as_deref().unwrap_or("default");
    let now = Instant::now();
    CONSOLE_TIMERS.with(|ts| {
        let mut ts = ts.borrow_mut();
        if let Some(slot) = ts.iter().position(|e| e.active && e.label == key) {
            let entry = &ts[slot];
            let us = now.duration_since(entry.ts).as_micros() as f64;
            let body = format!("{}: {:.3}ms", key, us / 1000.0);
            if !embed_config::log("log", &body) {
                print!("{}\n", body);
                let _ = std::io::stdout().flush();
            }
            ts[slot].active = false;
        }
    });
    qjs::sofuu_js_undefined()
}

// ── console.table — renders an array of objects as an ASCII table ────

const TBL_MAX_COLS: usize = 16;
const TBL_MAX_ROWS: usize = 256;

fn tbl_sep(widths: &[usize], out: &mut String) {
    let mut line = String::from("+");
    for w in widths {
        line.push_str(&"-".repeat(w + 2));
        line.push('+');
    }
    line.push('\n');
    out.push_str(&line);
}

unsafe extern "C" fn js_console_table(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut out = std::io::stdout().lock();
    if argc < 1 || qjs::JS_IsArray(ctx, *argv) == 0 {
        console_print(ctx, argc, argv, &mut out, "log", "", "");
        drop(out);
        return qjs::sofuu_js_undefined();
    }

    let arr = *argv;
    let mut col_keys: Vec<String> = Vec::new();

    /* Collect column names from the first element */
    let row0 = qjs::JS_GetPropertyUint32(ctx, arr, 0);
    if qjs::is_object(row0) {
        let mut props: *mut qjs::JSPropertyEnum = ptr::null_mut();
        let mut nprops: u32 = 0;
        if qjs::JS_GetOwnPropertyNames(
            ctx,
            &mut props,
            &mut nprops,
            row0,
            qjs::JS_GPN_STRING_MASK | qjs::JS_GPN_ENUM_ONLY,
        ) == 0
        {
            for p in 0..(nprops as usize).min(TBL_MAX_COLS) {
                // SAFETY: props array of nprops entries, malloc'd by quickjs.
                let atom = (*props.add(p)).atom;
                let k = qjs::JS_AtomToCString(ctx, atom);
                if !k.is_null() {
                    col_keys.push(CStr::from_ptr(k).to_string_lossy().into_owned());
                    qjs::sofuu_js_free_cstring(ctx, k);
                }
                qjs::JS_FreeAtom(ctx, atom);
            }
            qjs::js_free(ctx, props as *mut c_void);
        }
    }
    qjs::sofuu_js_free_value(ctx, row0);
    if col_keys.is_empty() {
        console_print(ctx, argc, argv, &mut out, "log", "", "");
        drop(out);
        return qjs::sofuu_js_undefined();
    }

    /* Collect row count */
    let lv = qjs::sofuu_js_get_property_str(ctx, arr, c"length".as_ptr());
    let mut nrows: i32 = 0;
    qjs::JS_ToInt32(ctx, &mut nrows, lv);
    qjs::sofuu_js_free_value(ctx, lv);
    if nrows > TBL_MAX_ROWS as i32 {
        nrows = TBL_MAX_ROWS as i32;
    }

    /* Column widths (at least as wide as the header) */
    let mut widths: Vec<usize> = col_keys.iter().map(|k| k.len()).collect();

    /* Collect all cells */
    let mut cells: Vec<Vec<String>> = Vec::new();
    for r in 0..nrows {
        let row = qjs::JS_GetPropertyUint32(ctx, arr, r as u32);
        let mut row_cells: Vec<String> = Vec::new();
        for c in 0..col_keys.len() {
            let key = CString::new(col_keys[c].as_str()).unwrap_or_default();
            let cell = qjs::sofuu_js_get_property_str(ctx, row, key.as_ptr());
            let s = js_val_to_str(ctx, cell).unwrap_or_default();
            row_cells.push(s.clone());
            qjs::sofuu_js_free_value(ctx, cell);
            if s.len() > widths[c] {
                widths[c] = s.len();
            }
        }
        qjs::sofuu_js_free_value(ctx, row);
        cells.push(row_cells);
    }

    /* Render into a buffer so a host log callback can take the whole
     * table as one message (H2.3). */
    let mut table = String::new();
    tbl_sep(&widths, &mut table);
    let mut header = String::from("|");
    for (c, k) in col_keys.iter().enumerate() {
        header.push_str(&format!(" \x1b[1m{:width$}\x1b[0m |", k, width = widths[c]));
    }
    header.push('\n');
    table.push_str(&header);
    tbl_sep(&widths, &mut table);
    for row_cells in &cells {
        let mut line = String::from("|");
        for (c, cell) in row_cells.iter().enumerate() {
            line.push_str(&format!(" {:width$} |", cell, width = widths[c]));
        }
        line.push('\n');
        table.push_str(&line);
    }
    tbl_sep(&widths, &mut table);
    if !embed_config::log("log", &table) {
        let _ = write!(out, "{}", table);
        let _ = out.flush();
    }
    drop(out);
    qjs::sofuu_js_undefined()
}

// ── Registration (C symbol replacement for mod_console_register) ─────

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_console_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let console = qjs::sofuu_js_new_object(ctx);

    qjs::sofuu_js_set_property_str(
        ctx,
        console,
        c"log".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_console_log, c"log".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        console,
        c"info".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_console_info, c"info".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        console,
        c"warn".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_console_warn, c"warn".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        console,
        c"error".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_console_error, c"error".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        console,
        c"assert".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_console_assert, c"assert".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        console,
        c"time".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_console_time, c"time".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        console,
        c"timeEnd".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_console_time_end, c"timeEnd".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        console,
        c"table".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_console_table, c"table".as_ptr(), 1),
    );

    qjs::sofuu_js_set_property_str(ctx, global, c"console".as_ptr(), console);
    qjs::sofuu_js_free_value(ctx, global);
}
