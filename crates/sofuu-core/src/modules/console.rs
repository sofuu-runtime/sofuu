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
    for i in 0..argc {
        if i > 0 {
            body.push(' ');
        }
        // SAFETY: argv valid for argc.
        match js_val_to_str(ctx, *argv.add(i as usize)) {
            Some(s) => body.push_str(&s),
            None => body.push_str("<unprintable>"),
        }
    }
    if embed_config::log(level, &body) {
        return; /* host callback consumed it */
    }
    let _ = write!(out, "{}{}{}\n", ansi_prefix, body, ansi_reset);
    let _ = out.flush();
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
