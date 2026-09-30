// rt/tui.rs — minimal alt-screen TUI for the Sofuu chat (M10).
//
// Port of the retired `src/io/tui.c` with the same rendering model: the
// terminal is fully owned while active; the conversation area is a buffer
// of display lines where every entry is EXACTLY ONE screen row (incoming
// text is split on '\n' at ingestion), so renders derive screen rows from
// (count, scroll, viewport). Redraws use absolute row positioning
// (ESC[row;1H + erase-line).
//
// The readline's input box + suggestions + footer are drawn by
// crates/sofuu-core/src/modules/process.rs via these row helpers. Resize
// handling lives in process.rs too (uv_signal_t SIGWINCH watcher → calls
// tui_relayout() + re-renders the input chrome). Nothing renders from a
// raw POSIX signal handler here.
//
// All `tui_*` symbols are exported with their C names — process.rs and
// sofuu-ffi (sofuu_tui_active/log/width aliases) link against them exactly
// as they linked against tui.c.

use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::io::Write;
use std::os::raw::{c_char, c_int};

pub const GUTTER: usize = 2;
const MAX_LINES: usize = 2048;
const LINE_CAP: usize = 1024;

thread_local! {
    static G_LINES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static G_SCROLL: Cell<i32> = const { Cell::new(0) };
    static G_LAST_LINES: Cell<i32> = const { Cell::new(0) };
    static G_HEADER: RefCell<String> = const { RefCell::new(String::new()) };
    static G_ACTIVE: Cell<i32> = const { Cell::new(0) };
    static G_INIT_DONE: Cell<i32> = const { Cell::new(0) };
}

/// Write a raw byte string to stdout and flush (C: fputs + fflush).
fn write_out(s: &str) {
    let mut out = std::io::stdout();
    let _ = out.write_all(s.as_bytes());
    let _ = out.flush();
}

/// Overwrite one absolute screen row in place (used by the animated logo).
/// Does NOT touch the conversation buffer — purely a screen overlay. Does
/// NOT move the cursor: the readline renders the cursor position on every
/// keystroke, so an animation tick must never fight it.
#[no_mangle]
pub unsafe extern "C" fn tui_overlay_row(row: c_int, content: *const c_char) {
    let bytes = c_str(content);
    write_out(&format!("\x1b[{};1H\x1b[K", row));
    let mut out = std::io::stdout();
    let _ = out.write_all(bytes);
    let _ = out.flush();
}

fn c_str<'a>(s: *const c_char) -> &'a [u8] {
    if s.is_null() {
        return b"";
    }
    // SAFETY: s must be a NUL-terminated C string (C contract). Callers must
    // never pass a dangling pointer from an empty Rust String — see the
    // empty-row guard in render_rows.
    unsafe { std::ffi::CStr::from_ptr(s) }.to_bytes()
}

/// Display cells for one character: East-Asian wide ranges occupy 2 cells
/// on every real terminal; everything else is 1. Kept in sync with
/// chat.rs's `cell_w` so panel padding and viewport math agree.
fn char_cells(c: char) -> usize {
    let o = c as u32;
    if (0x1100..=0x115F).contains(&o)
        || (0x2E80..=0xA4CF).contains(&o)
        || (0xAC00..=0xD7A3).contains(&o)
        || (0xF900..=0xFAFF).contains(&o)
        || (0xFE30..=0xFE6F).contains(&o)
        || (0xFF00..=0xFF60).contains(&o)
        || (0xFFE0..=0xFFE6).contains(&o)
        || (0x1F300..=0x1F64F).contains(&o)
        || (0x1F900..=0x1F9FF).contains(&o)
        || (0x20000..=0x3FFFD).contains(&o)
    {
        2
    } else {
        1
    }
}

/// tui_disp_width — UTF-8 display cells, ANSI escape sequences skipped,
/// wide (CJK/emoji) characters counted as 2 cells like a real terminal.
/// Uses the caller-provided `len` (NOT strlen) so a dangling pointer from
/// an empty Rust String (as_ptr() of a 0-length buffer) is safe: len 0
/// returns 0 immediately without dereferencing the pointer.
#[no_mangle]
pub unsafe extern "C" fn tui_disp_width(s: *const c_char, len: usize) -> c_int {
    if s.is_null() || len == 0 {
        return 0;
    }
    // SAFETY: s must be readable for len bytes (C contract). We never walk
    // past len, so empty/dangling pointers are safe (len == 0 → early out).
    let bytes = unsafe { std::slice::from_raw_parts(s as *const u8, len) };
    let n = bytes.len();
    let mut w: usize = 0;
    let mut i = 0;
    while i < n {
        let c = bytes[i];
        if c == 0x1b {
            i += 1;
            if i < n && bytes[i] == b'[' {
                while i + 1 < n {
                    i += 1;
                    let x = bytes[i];
                    if (0x40..=0x7E).contains(&x) {
                        break;
                    }
                }
            }
            i += 1;
            continue;
        }
        // Decode one UTF-8 char and count its true cell width.
        let ln = if c >= 0xF0 {
            4
        } else if c >= 0xE0 {
            3
        } else if c >= 0xC0 {
            2
        } else {
            1
        };
        if let Ok(chunk) = std::str::from_utf8(&bytes[i..(i + ln).min(n)]) {
            if let Some(ch) = chunk.chars().next() {
                w += char_cells(ch);
            }
        }
        i += ln;
    }
    w as c_int
}

fn tui_size(w: &mut c_int, h: &mut c_int) {
    #[cfg(not(target_os = "windows"))]
    {
        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
        let r = unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) };
        if r == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
            *w = ws.ws_col as c_int;
            *h = ws.ws_row as c_int;
        } else {
            *w = 80;
            *h = 24;
        }
    }
    #[cfg(target_os = "windows")]
    {
        // TIOCGWINSZ has no libc binding on Windows; the TUI runs at the
        // classic default there.
        *w = 80;
        *h = 24;
    }
    if *w < 40 {
        *w = 40;
    }
    if *h < 12 {
        *h = 12;
    }
}

#[no_mangle]
pub unsafe extern "C" fn tui_width() -> c_int {
    let (mut w, mut h) = (0, 0);
    tui_size(&mut w, &mut h);
    w
}

#[no_mangle]
pub unsafe extern "C" fn tui_height() -> c_int {
    let (mut w, mut h) = (0, 0);
    tui_size(&mut w, &mut h);
    h
}

// ── Lifecycle ─────────────────────────────────────────────────────

/// Windows: classic conhost (cmd.exe) does not interpret ANSI escapes
/// until the process opts in with ENABLE_VIRTUAL_TERMINAL_PROCESSING —
/// Windows Terminal defaults it on, cmd.exe does not, so every
/// alt-screen/color sequence the TUI writes prints as literal
/// `[2J[?25l` soup. Enable it once on stdout + stderr before the first
/// escape leaves the process. GetConsoleMode fails on redirected handles
/// (file/pipe), which is the signal to skip. Left on for the life of the
/// process — cmd builtins are unaffected by the flag.
#[cfg(target_os = "windows")]
pub fn windows_enable_vt() {
    use std::os::raw::c_void;

    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetStdHandle(nStdHandle: u32) -> *mut c_void;
        fn GetConsoleMode(hConsoleHandle: *mut c_void, lpMode: *mut u32) -> i32;
        fn SetConsoleMode(hConsoleHandle: *mut c_void, dwMode: u32) -> i32;
    }

    unsafe fn enable(which: u32) {
        let h = GetStdHandle(which);
        // NULL (no handle) or INVALID_HANDLE_VALUE (-1): nothing to set.
        if h.is_null() || h as isize == -1 {
            return;
        }
        let mut mode: u32 = 0;
        // 0 return = not a console (redirected to file/pipe) — skip.
        if GetConsoleMode(h, &mut mode) == 0 {
            return;
        }
        SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }

    unsafe {
        enable(STD_OUTPUT_HANDLE);
        enable(STD_ERROR_HANDLE);
    }
}

/// Non-Windows terminals interpret ANSI natively — nothing to opt into.
#[cfg(not(target_os = "windows"))]
pub fn windows_enable_vt() {}

#[no_mangle]
pub unsafe extern "C" fn tui_init() {
    let mut done = 0;
    G_INIT_DONE.with(|d| done = d.get());
    if done != 0 {
        return;
    }
    // Idempotent belt-and-suspenders: main() already calls this on
    // Windows before any output; keep it here so every tui_entry path
    // is covered even if init ordering changes.
    windows_enable_vt();
    G_INIT_DONE.with(|d| d.set(1));
}

#[no_mangle]
pub unsafe extern "C" fn tui_enter() {
    tui_init();
    let mut active = 0;
    G_ACTIVE.with(|a| active = a.get());
    if active != 0 {
        return;
    }
    G_ACTIVE.with(|a| a.set(1));
    G_SCROLL.with(|s| s.set(0));
    // alt screen on + cursor hidden (the armed readline shows it) + clear.
    write_out(concat!("\x1b[?1049h", "\x1b[?25l", "\x1b[2J\x1b[H"));
    tui_render_conversation();
}

#[no_mangle]
pub unsafe extern "C" fn tui_exit() {
    let mut active = 0;
    G_ACTIVE.with(|a| active = a.get());
    if active == 0 {
        return;
    }
    G_ACTIVE.with(|a| a.set(0));
    write_out("\x1b[?25h\x1b[?1049l"); /* cursor back, alt screen off */
}

#[no_mangle]
pub unsafe extern "C" fn tui_active() -> c_int {
    G_ACTIVE.with(|a| a.get())
}

unsafe fn tui_clear_rows(a: c_int, b: c_int) {
    if a > b {
        return;
    }
    let mut out = String::with_capacity(((b - a + 1) as usize) * 16);
    for r in a..=b {
        out.push_str(&format!("\x1b[{};1H\x1b[K", r));
    }
    write_out(&out);
}

/// Clear the 2-row gap between conversation and input box (h-7, h-6).
/// Callers that owned a larger overlay (picker) should also clear its tail.
#[no_mangle]
pub unsafe extern "C" fn tui_clear_gap() {
    let mut active = 0;
    G_ACTIVE.with(|a| active = a.get());
    if active == 0 {
        return;
    }
    let (mut _w, mut h) = (0, 0);
    tui_size(&mut _w, &mut h);
    tui_clear_rows(h - 7, h - 6);
}

#[no_mangle]
pub unsafe extern "C" fn tui_discard_last() {
    G_LAST_LINES.with(|g| g.set(0));
}

/// Re-paint everything at the current terminal size. Called from the
/// uv SIGWINCH watcher (process.rs), which then re-renders the input
/// chrome (box/suggestions/footer) itself.
#[no_mangle]
pub unsafe extern "C" fn tui_relayout() {
    let mut active = 0;
    G_ACTIVE.with(|a| active = a.get());
    if active == 0 {
        return;
    }
    write_out("\x1b[2J\x1b[H");
    tui_render_conversation();
    // Gap rows are outside conversation (h-7,h-6) — ensure they are blank
    // after a full-screen clear, so a prior picker/info row cannot survive.
    tui_clear_gap();
}

// ── Stored-only header (no persistent bar — kept for compat) ──────

#[no_mangle]
pub unsafe extern "C" fn tui_set_header(line: *const c_char) {
    if line.is_null() {
        return;
    }
    let s = c_str(line);
    G_HEADER.with(|g| {
        let s = String::from_utf8_lossy(&s[..s.len().min(159)]);
        *g.borrow_mut() = s.into_owned();
    });
}

// ── Layout constants (see src/io/tui.h for the diagram) ───────────
// With terminal height h:
//   conversation : rows 1 .. h-8
//   gap          : h-7, h-6            (2 rows)
//   input box    : h-5 (top), h-4 (input), h-3 (bottom)
//   footer       : h-2  ← sits DIRECTLY below the box (status + hints)
//   metric       : h-1

#[no_mangle]
pub unsafe extern "C" fn tui_conv_top() -> c_int {
    1
}

#[no_mangle]
pub unsafe extern "C" fn tui_conv_bottom() -> c_int {
    let (mut w, mut h) = (0, 0);
    tui_size(&mut w, &mut h);
    h - 8
}

#[no_mangle]
pub unsafe extern "C" fn tui_box_top() -> c_int {
    let (mut w, mut h) = (0, 0);
    tui_size(&mut w, &mut h);
    h - 5
}

/// The gap row directly above the input box top border (h-6) — home of the
/// animated agent-phase indicator ("Thinking"/"Coding"/…). Same row
/// `tui_clear_gap()` blanks, so a resize or turn-end that clears the gap
/// also erases a stale phase line with no extra bookkeeping.
#[no_mangle]
pub unsafe extern "C" fn tui_phase_row() -> c_int {
    let (mut w, mut h) = (0, 0);
    tui_size(&mut w, &mut h);
    h - 6
}

#[no_mangle]
pub unsafe extern "C" fn tui_input_row() -> c_int {
    let (mut w, mut h) = (0, 0);
    tui_size(&mut w, &mut h);
    h - 4
}

#[no_mangle]
pub unsafe extern "C" fn tui_footer_row() -> c_int {
    let (mut w, mut h) = (0, 0);
    tui_size(&mut w, &mut h);
    h - 2
}

#[no_mangle]
pub unsafe extern "C" fn tui_footer2_row() -> c_int {
    let (mut w, mut h) = (0, 0);
    tui_size(&mut w, &mut h);
    h - 1
}

/// Bounded truncation over a Rust byte slice — the strlen-free twin of
/// `tui_truncate_cells`, same semantics: whole CSI escapes consumed,
/// multibyte chars never split, trailing zero-cell escapes absorbed,
/// degenerate input returns the full length. The input length comes from
/// the slice, never from a NUL scan, so callers holding non-NUL-terminated
/// buffers (the footer statics are `Vec<u8>` from `CStr::to_bytes()`) can
/// never have the walk run past the allocation. Returns a byte index
/// `<= bytes.len()`, usable directly as a slice bound.
pub fn truncate_cells(bytes: &[u8], max_cells: usize) -> usize {
    let n = bytes.len();
    if max_cells == 0 {
        return 0;
    }
    let mut cells = 0;
    let mut i = 0;
    while i < n {
        let c = bytes[i];
        if c == 0x1b {
            // keep the whole escape sequence
            let start = i;
            i += 1;
            if i < n && bytes[i] == b'[' {
                while i + 1 < n {
                    i += 1;
                    if (0x40..=0x7E).contains(&bytes[i]) {
                        break;
                    }
                }
            }
            i += 1;
            if i - start >= max_cells {
                return n; /* degenerate */
            }
            continue;
        }
        if (c & 0xC0) == 0x80 {
            i += 1;
            continue; /* continuation byte */
        }
        // One UTF-8 lead byte: count the char's true cell width.
        let ln = if c >= 0xF0 {
            4
        } else if c >= 0xE0 {
            3
        } else if c >= 0xC0 {
            2
        } else {
            1
        };
        let cw = std::str::from_utf8(&bytes[i..(i + ln).min(n)])
            .ok()
            .and_then(|ch| ch.chars().next())
            .map(char_cells)
            .unwrap_or(1);
        cells += cw;
        i += ln;
        if cells >= max_cells {
            /* never split a multibyte char: its full bytes are consumed
             * above, so the slice boundary is valid. */
            /* TRAILING zero-cell escapes after the cut must survive: a row
             * whose visible width is exactly the budget and ends with a
             * color reset used to lose its last glyph to a spurious "…"
             * (the welcome panel's right borders). Absorb any escape
             * sequences that follow — they cost no cells. */
            while i < n && bytes[i] == 0x1b {
                i += 1;
                if i < n && bytes[i] == b'[' {
                    while i + 1 < n {
                        i += 1;
                        if (0x40..=0x7E).contains(&bytes[i]) {
                            break;
                        }
                    }
                }
                i += 1;
            }
            break;
        }
    }
    /* A trailing lone ESC (absorption: `i += 1` twice from n-1) or a lead
     * byte near the end claiming more bytes than remain can overshoot n;
     * the returned index must stay a valid slice bound. */
    i.min(n)
}

/// C-ABI entry for genuinely NUL-terminated callers (CString-backed pickers,
/// C tests). Rust callers holding `Vec<u8>`/`String` slices must use the
/// bounded `truncate_cells` above — this walks the pointer with strlen and
/// reads past a non-terminated allocation.
#[no_mangle]
pub unsafe extern "C" fn tui_truncate_cells(s: *const c_char, max_cells: c_int) -> usize {
    if max_cells <= 0 {
        return 0;
    }
    truncate_cells(c_str(s), max_cells as usize)
}

// ── Core renderer ─────────────────────────────────────────────────
// The one mapping from buffer → screen lives here. Content is top-
// anchored while it fits the viewport, bottom-anchored (auto-scroll)
// once it overflows; g_scroll shifts the window up into history.

unsafe fn render_rows(top_row: c_int, bottom_row: c_int) {
    let mut active = 0;
    G_ACTIVE.with(|a| active = a.get());
    if active == 0 {
        return;
    }
    let (mut w, mut h) = (0, 0);
    tui_size(&mut w, &mut h);
    let top = unsafe { tui_conv_top() };
    let vh = unsafe { tui_conv_bottom() } - top + 1;

    let count = G_LINES.with(|l| l.borrow().len()) as i32;
    let over = count - vh;
    let scroll = G_SCROLL.with(|s| s.get());
    let scroll = if scroll > over { over } else { scroll };
    let scroll = if scroll < 0 { 0 } else { scroll };
    G_SCROLL.with(|s| s.set(scroll));

    let end = count - scroll; /* exclusive */
    let mut start = end - vh;
    if start < 0 {
        start = 0;
    }

    let mut t = top_row;
    if t < top {
        t = top;
    }
    let mut b = bottom_row;
    let cb = unsafe { tui_conv_bottom() };
    if b > cb {
        b = cb;
    }

    let lines = G_LINES.with(|l| l.borrow().clone());
    let mut out = String::with_capacity(((b - t + 1) as usize) * 80);
    /* Every conversation row is indented by GUTTER cells so the text shares
     * the input box's left margin ("│ ··text") instead of touching the
     * screen edge; the truncation budget shrinks to match. */
    let budget = (w as usize).saturating_sub(GUTTER);
    /* App-managed selection (mouse drag + Ctrl-K copy): selected rows
     * paint inverse-video. The range is held by the readline (process.rs)
     * in SCREEN rows — the same coordinate this loop paints. */
    let (sel_a, sel_b) = crate::modules::process::sel_range();
    let sel_lo = if sel_a >= 0 && sel_b >= 0 { sel_a.min(sel_b) } else { -1 };
    let sel_hi = if sel_a >= 0 && sel_b >= 0 { sel_a.max(sel_b) } else { -1 };
    for row in t..=b {
        out.push_str(&format!("\x1b[{};1H\x1b[K", row));
        let idx = start + (row - top);
        if idx >= 0 && (idx as usize) < lines.len() {
            let l = &lines[idx as usize];
            let full = l.len();
            /* Empty rows are common (blank separator lines in the panel).
             * An empty Rust String's as_ptr() is a dangling non-null
             * pointer — passing it to CStr::from_ptr (strlen) reads past
             * the 0-length allocation and SIGSEGVs. Skip them entirely. */
            if full == 0 {
                continue;
            }
            out.push_str(&" ".repeat(GUTTER));
            let mut bl = truncate_cells(l.as_bytes(), budget);
            /* the full row is usable — deferred wrap is absorbed by the
             * next row's ESC[…H */
            if bl < full && unsafe { tui_disp_width(l.as_ptr() as *const c_char, bl) } > budget as c_int - 1 {
                bl = truncate_cells(l.as_bytes(), budget - 1); /* room for "…" */
            }
            /* truncate_cells is bounded: it returns at most l.len(), so the
             * slice below is always valid (no strlen involved). */
            if bl > full {
                bl = full;
            }
            let selected = row >= sel_lo && row <= sel_hi;
            if selected {
                /* Inverse video across the whole visible row (text +
                 * trailing pad) so the selection reads as a solid block
                 * like a native terminal selection. */
                out.push_str("\x1b[7m");
                out.push_str(&l[..bl]);
                if bl < full {
                    out.push('…');
                }
                out.push_str("\x1b[0m");
            } else {
                out.push_str(&l[..bl]);
                if bl < full {
                    out.push('…'); /* only when truly truncated */
                }
            }
        }
    }
    write_out(&out);
}

/// Screen row where buffer index `idx` would paint (current mapping).
unsafe fn screen_row_of(idx: i32) -> c_int {
    let top = unsafe { tui_conv_top() };
    let vh = unsafe { tui_conv_bottom() } - top + 1;
    let count = G_LINES.with(|l| l.borrow().len()) as i32;
    let scroll = G_SCROLL.with(|s| s.get());
    let end = count - scroll;
    let mut start = end - vh;
    if start < 0 {
        start = 0;
    }
    if idx < start {
        return top; /* above the viewport → repaint all */
    }
    top + (idx - start)
}

#[no_mangle]
pub unsafe extern "C" fn tui_render_conversation_range(bottom: c_int) {
    unsafe { render_rows(tui_conv_top(), bottom) };
}

#[no_mangle]
pub unsafe extern "C" fn tui_render_conversation() {
    unsafe { render_rows(tui_conv_top(), tui_conv_bottom()) };
}

// ── Buffer mutation ───────────────────────────────────────────────

fn tui_push(line: &str) {
    G_LINES.with(|g| {
        let mut lines = g.borrow_mut();
        if lines.len() < MAX_LINES {
            lines.push(line.to_string());
        } else {
            // drop the oldest, shift everything down (ring semantics)
            lines.remove(0);
            lines.push(line.to_string());
        }
    });
    // Keep a scrolled view pinned to the same content while rows are
    // appended below it (covers both the growing and the ring-shift case).
    G_SCROLL.with(|s| {
        if s.get() > 0 {
            s.set(s.get() + 1);
        }
    });
}

/// Push text that may contain '\n' as multiple one-row entries. A single
/// trailing empty segment ("…\n") is dropped (println semantics); interior
/// and leading empties are kept. Returns the number of rows added.
///
/// Long segments are SOFT-WRAPPED to the current terminal width at word
/// boundaries (hard-broken when a single token overflows) so a streamed
/// answer is fully readable instead of clipped with "…". ANSI escapes are
/// zero-width and never split; wide chars count 2 cells. Rows pushed while
/// the terminal was wider keep their stored shape (a resize re-wraps only
/// future output) — an acceptable trade for a chat scrollback.
fn tui_push_split(text: &[u8]) -> c_int {
    // Terminal-output sanitizer: model/host text may carry hostile escapes.
    // Strip OSC (ESC ] … BEL/ESC\), DCS/SOS/PM/APC (ESC P/X/^/_ … ST) and
    // non-SGR CSI, but KEEP SGR colors (ESC [ … m) so output stays pretty.
    // Without this a model answer containing ESC]52;… writes the user's
    // clipboard (paste-hijack) via the terminal's own OSC52 handling.
    let text: Vec<u8> = {
        let mut out: Vec<u8> = Vec::with_capacity(text.len());
        let mut i = 0;
        while i < text.len() {
            if text[i] == 0x1b && i + 1 < text.len() {
                let c1 = text[i + 1];
                if c1 == b']' {
                    // OSC: consume until BEL or ESC\.
                    i += 2;
                    while i < text.len() {
                        if text[i] == 0x07 {
                            i += 1;
                            break;
                        }
                        if text[i] == 0x1b && i + 1 < text.len() && text[i + 1] == b'\\' {
                            i += 2;
                            break;
                        }
                        i += 1;
                    }
                    continue;
                } else if c1 == b'P' || c1 == b'X' || c1 == b'^' || c1 == b'_' {
                    // DCS/SOS/PM/APC: consume until ST (ESC\).
                    i += 2;
                    while i + 1 < text.len() {
                        if text[i] == 0x1b && text[i + 1] == b'\\' {
                            i += 2;
                            break;
                        }
                        i += 1;
                    }
                    continue;
                } else if c1 == b'[' {
                    // CSI: keep SGR (…m), drop everything else.
                    let mut j = i + 2;
                    while j < text.len() && !(0x40..=0x7e).contains(&text[j]) {
                        j += 1;
                    }
                    if j < text.len() {
                        let fin = text[j];
                        if fin == b'm' {
                            out.extend_from_slice(&text[i..=j]);
                        }
                        // else: drop hostile CSI (cursor, scroll, etc.)
                        i = j + 1;
                        continue;
                    } else {
                        break;
                    }
                }
            }
            out.push(text[i]);
            i += 1;
        }
        out
    };
    let text = &text[..];
    let (mut w, mut _h) = (0, 0);
    tui_size(&mut w, &mut _h);
    // Rows render GUTTER cells indented (see render_rows), so wrap to the
    // remaining width to avoid a second "…" truncation pass at paint time.
    let w = (w as usize).saturating_sub(GUTTER);
    let mut added: usize = 0;
    let mut saw_nl = false;
    let mut p: usize = 0;
    loop {
        let nl = text[p..].iter().position(|&b| b == b'\n');
        let mut len = match nl {
            Some(off) => off,
            None => text.len() - p,
        };
        if nl.is_none() && saw_nl && len == 0 {
            break; /* drop one trailing empty */
        }
        if len >= LINE_CAP {
            len = LINE_CAP - 1;
            // don't end the entry mid-glyph: back off a split UTF-8 char
            while len > 0 && (text[p + len - 1] & 0xC0) == 0x80 {
                len -= 1;
            }
            if len > 0 && text[p + len - 1] >= 0xC0 {
                len -= 1;
            }
        }
        let seg = String::from_utf8_lossy(&text[p..p + len]).into_owned();
        for row in wrap_row(&seg, w) {
            tui_push(&row);
            added += 1;
        }
        match nl {
            None => break,
            Some(off) => {
                saw_nl = true;
                p += off + 1;
            }
        }
    }
    added as c_int
}

/// One wrapping token: an ANSI escape (0 cells), a space, or one visible
/// char with its true cell width.
struct WrapTok<'a> {
    s: &'a str,
    cells: usize,
    space: bool,
}

fn tokenize<'a>(s: &'a str) -> Vec<WrapTok<'a>> {
    let b = s.as_bytes();
    let n = b.len();
    let mut toks = Vec::with_capacity(n / 2 + 2);
    let mut i = 0;
    while i < n {
        if b[i] == 0x1b {
            let start = i;
            i += 1;
            if i < n && b[i] == b'[' {
                while i + 1 < n {
                    i += 1;
                    if (0x40..=0x7E).contains(&b[i]) {
                        break;
                    }
                }
            }
            i += 1;
            toks.push(WrapTok { s: &s[start..i.min(n)], cells: 0, space: false });
            continue;
        }
        let c = b[i];
        let ln = if c >= 0xF0 {
            4
        } else if c >= 0xE0 {
            3
        } else if c >= 0xC0 {
            2
        } else {
            1
        };
        let end = (i + ln).min(n);
        let chunk = &s[i..end];
        let space = chunk == " ";
        let cells = if space {
            1
        } else {
            chunk.chars().next().map(char_cells).unwrap_or(1)
        };
        toks.push(WrapTok { s: chunk, cells, space });
        i = end;
    }
    toks
}

/// Wrap one already-newline-free row to `w` cells. Breaks at the last
/// space that fits; hard-breaks tokens longer than a full line. Trailing
/// plain spaces on the wrapped line are dropped (the break space is
/// consumed); zero-cell escapes pass through untouched.
/// `pub` (not pub(crate)): the chat.rs panel tests live in the bin crate
/// and assert rows survive this wrap byte-identical.
pub fn wrap_row(s: &str, w: usize) -> Vec<String> {
    if w < 8 || s.is_empty() {
        return vec![s.to_string()];
    }
    let toks = tokenize(s);
    let total: usize = toks.iter().map(|t| t.cells).sum();
    if total <= w {
        return vec![s.to_string()];
    }
    let mut lines: Vec<String> = Vec::new();
    let mut cur: Vec<&WrapTok> = Vec::new();
    let mut cells = 0usize;
    let mut last_space: isize = -1;
    for t in toks.iter() {
        if t.space {
            last_space = cur.len() as isize;
        }
        if cells + t.cells > w && !cur.is_empty() {
            let (line_toks, mut rest): (Vec<&WrapTok>, Vec<&WrapTok>) = if last_space > 0 {
                let cut = last_space as usize;
                // the space itself is consumed by the break; when it is the
                // LAST token of the line the tail is simply empty (cut+1 ==
                // len is a valid open-ended bound, but be explicit so the
                // intent survives refactors)
                (
                    cur[..cut].to_vec(),
                    if cut + 1 >= cur.len() {
                        Vec::new()
                    } else {
                        cur[cut + 1..].to_vec()
                    },
                )
            } else {
                (cur.clone(), Vec::new())
            };
            rest.push(t);
            let mut line: String = line_toks.iter().map(|x| x.s).collect();
            // drop trailing plain spaces (keep escapes — they may carry style)
            while line.ends_with(' ') {
                line.pop();
            }
            lines.push(line);
            cur = rest;
            cells = cur.iter().map(|x| x.cells).sum();
            last_space = -1;
        } else {
            cur.push(t);
            cells += t.cells;
        }
    }
    if !cur.is_empty() {
        let mut line: String = cur.iter().map(|x| x.s).collect();
        while line.ends_with(' ') {
            line.pop();
        }
        lines.push(line);
    }
    lines
}

#[no_mangle]
pub unsafe extern "C" fn tui_log(line: *const c_char) {
    /* NULL or empty (a dangling pointer from an empty Rust String) must not
     * reach CStr::from_ptr/strlen — the welcome panel logs several empty
     * rows and this was a real SIGSEGV at chat startup. */
    if line.is_null() {
        return;
    }
    // SAFETY: line is a live NUL-terminated C string (C contract).
    let bytes = unsafe { CStr::from_ptr(line) }.to_bytes();
    if bytes.is_empty() {
        return;
    }
    let bytes = bytes.to_vec();
    let first_new = G_LINES.with(|l| l.borrow().len()) as c_int;
    let added = tui_push_split(&bytes);
    G_LAST_LINES.with(|g| g.set(0)); /* a fresh log seals the previous replaceable unit */
    let mut active = 0;
    G_ACTIVE.with(|a| active = a.get());
    if active == 0 {
        return;
    }
    let mut scroll = 0;
    G_SCROLL.with(|s| scroll = s.get());
    if scroll != 0 {
        return; /* scrolled: the view holds its content */
    }
    if added <= 0 {
        return;
    }
    unsafe { render_rows(screen_row_of(first_new), tui_conv_bottom()) };
}

#[no_mangle]
pub unsafe extern "C" fn tui_log_last(line: *const c_char) {
    if line.is_null() {
        return;
    }
    // SAFETY: line is a live NUL-terminated C string (C contract).
    let bytes = unsafe { CStr::from_ptr(line) }.to_bytes();
    if bytes.is_empty() {
        return;
    }
    let bytes = bytes.to_vec();
    let count = G_LINES.with(|l| l.borrow().len()) as i32;
    let mut active = 0;
    G_ACTIVE.with(|a| active = a.get());
    let mut scroll = 0;
    G_SCROLL.with(|s| scroll = s.get());

    if count == 0 {
        // nothing to replace — start a replaceable unit
        let added = tui_push_split(&bytes);
        G_LAST_LINES.with(|g| g.set(added));
        if active != 0 && scroll == 0 {
            unsafe { render_rows(screen_row_of(0), tui_conv_bottom()) };
        }
        return;
    }

    let old_rows = {
        let g = G_LAST_LINES.with(|x| x.get());
        if g > count {
            count
        } else {
            g
        }
    };
    let top_old = if active != 0 && scroll == 0 {
        unsafe { screen_row_of(count - old_rows) }
    } else {
        0
    };

    // Removing tail rows below a scrolled view pulls the bottom up with
    // them — compensate so the visible window keeps its content.
    if scroll > 0 {
        let ns = scroll - old_rows;
        G_SCROLL.with(|s| s.set(if ns < 0 { 0 } else { ns }));
    }
    G_LINES.with(|l| {
        let mut lines = l.borrow_mut();
        for _ in 0..old_rows {
            lines.pop();
        }
    });
    let added = tui_push_split(&bytes);
    G_LAST_LINES.with(|g| g.set(added));

    G_ACTIVE.with(|a| active = a.get());
    G_SCROLL.with(|s| scroll = s.get());
    if active == 0 || scroll != 0 {
        return; /* buffer updated; view frozen */
    }
    let new_count = G_LINES.with(|l| l.borrow().len()) as i32;
    let top_new = unsafe { screen_row_of(new_count - added) };
    let top = if top_old < top_new { top_old } else { top_new };
    unsafe { render_rows(top, tui_conv_bottom()) };
}

/// Clear the conversation buffer + scroll. Used when config changes so the
/// welcome panel can be re-logged fresh (the model/provider line updates).
#[no_mangle]
pub unsafe extern "C" fn tui_reset() {
    G_LINES.with(|l| l.borrow_mut().clear());
    G_SCROLL.with(|s| s.set(0));
    G_LAST_LINES.with(|g| g.set(0));
}

#[no_mangle]
pub unsafe extern "C" fn tui_scroll_pos() -> c_int {
    G_SCROLL.with(|s| s.get())
}

/// Text of conversation screen rows r0..=r1 (inclusive), newline-joined.
/// Mirrors the buffer→screen mapping in render_rows exactly (viewport
/// height, G_SCROLL clamp, top anchor) so a highlighted row copies the
/// very text the user sees on it. ANSI styling is stripped by the caller.
#[no_mangle]
pub unsafe extern "C" fn tui_selected_rows(r0: c_int, r1: c_int) -> *mut c_char {
    let top = unsafe { tui_conv_top() };
    let vh = unsafe { tui_conv_bottom() } - top + 1;
    let count = G_LINES.with(|l| l.borrow().len()) as i32;
    let over = count - vh;
    let scroll = G_SCROLL.with(|s| s.get());
    let scroll = scroll.min(over).max(0);
    let end = count - scroll; /* exclusive */
    let start = (end - vh).max(0);
    let mut parts: Vec<String> = Vec::new();
    for row in r0..=r1 {
        let idx = start + (row - top);
        if idx >= 0 && (idx as usize) < G_LINES.with(|l| l.borrow().len()) {
            parts.push(G_LINES.with(|l| l.borrow()[idx as usize].clone()));
        }
    }
    let joined = parts.join("\n");
    let c = std::ffi::CString::new(joined).unwrap_or_default();
    let p = libc::malloc(c.as_bytes_with_nul().len()) as *mut c_char;
    if !p.is_null() {
        std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.as_bytes_with_nul().len());
    }
    p
}

#[no_mangle]
pub unsafe extern "C" fn tui_scroll(delta_rows: c_int) {
    let mut active = 0;
    G_ACTIVE.with(|a| active = a.get());
    if active == 0 {
        return;
    }
    // If we're not yet in alt-screen (e.g. early input), enter so rows 1..h-8 exist
    if G_LINES.with(|l| l.borrow().is_empty()) {
        // still allow scroll clamp math to run — over will be negative
    }
    let top = unsafe { tui_conv_top() };
    let vh = unsafe { tui_conv_bottom() } - top + 1;
    let count = G_LINES.with(|l| l.borrow().len()) as i32;
    let over = count - vh;
    G_SCROLL.with(|s| {
        let mut v = s.get() + delta_rows;
        if v > over {
            v = over;
        }
        if v < 0 {
            v = 0;
        }
        s.set(v);
    });
    tui_render_conversation();
}

#[no_mangle]
pub unsafe extern "C" fn tui_place_cursor(col: usize) {
    let mut active = 0;
    G_ACTIVE.with(|a| active = a.get());
    if active == 0 {
        return;
    }
    let (mut w, mut h) = (0, 0);
    tui_size(&mut w, &mut h);
    /* The cursor belongs INSIDE the input box, on the input row. */
    write_out(&format!("\x1b[{};{}H", tui_input_row(), col + 1));
}

// ── sofuu-ffi bridges (session.rs prints → conversation area) ─────
// These three kept their original sofuu_tui_* names so sofuu-ffi's
// externs link unchanged.

#[no_mangle]
pub unsafe extern "C" fn sofuu_tui_active() -> c_int {
    tui_active()
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_tui_log(line: *const c_char) {
    if line.is_null() {
        return;
    }
    tui_log(line);
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_tui_width() -> c_int {
    tui_width()
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_tui_discard_last() {
    tui_discard_last();
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_tui_clear_gap() {
    tui_clear_gap();
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_tui_reset() {
    tui_reset();
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_tui_overlay_row(row: c_int, content: *const c_char) {
    tui_overlay_row(row, content);
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_tui_phase_row() -> c_int {
    tui_phase_row()
}

#[cfg(test)]
mod tests {
    use super::{tui_truncate_cells, truncate_cells, wrap_row};
    use std::ffi::CString;

    /// For every cell budget, the returned byte length must land on a UTF-8
    /// char boundary (callers slice `&s[..n]` — a mid-char offset panics).
    /// Regression: box-drawing frames broke TTY chat with
    /// "byte index N is not a char boundary … inside '─'".
    #[test]
    fn truncate_never_splits_multibyte_chars() {
        let samples = [
            "╭──────────╮ welcome",   // 3-byte box chars (the panic case)
            "ascii only string",
            "emoji 🎉🎉🎉 tail",
            "CJK 素風素風 mixed",
            "\x1b[1m╭──╮\x1b[0m bold frame", // styling + multibyte
        ];
        for text in samples {
            let c = CString::new(text).unwrap();
            for cells in 0..=(text.chars().count() as i32 + 4) {
                let n = unsafe { tui_truncate_cells(c.as_ptr(), cells) };
                assert!(n <= text.len(), "len overrun: {text:?} cells={cells} n={n}");
                assert!(
                    text.is_char_boundary(n),
                    "mid-char cut: {text:?} cells={cells} n={n} (byte {:#x} is {:?})",
                    text.as_bytes().get(n).copied().unwrap_or(0),
                    text.as_bytes().get(n).copied().map(|b| b as char),
                );
            }
        }
        // Untruncated passthrough: generous budget returns full length.
        let text = "╭──╮ done";
        let c = CString::new(text).unwrap();
        let n = unsafe { tui_truncate_cells(c.as_ptr(), 999) };
        assert_eq!(n, text.len());
    }

    /// The welcome-border regression: a row whose visible width is EXACTLY
    /// the budget and ends with an ANSI reset must return the FULL byte
    /// length — trailing zero-cell escapes are absorbed, so render_rows
    /// never replaces the closing ╮/│/╯ with "…".
    #[test]
    fn truncate_absorbs_trailing_escapes() {
        // ╭ + 8×─ + ╮ = 10 cells, wrapped in dim-magenta escapes.
        let text = format!("\x1b[2;35m╭{}╮\x1b[0m", "─".repeat(8));
        let c = CString::new(text.as_str()).unwrap();
        let n = unsafe { tui_truncate_cells(c.as_ptr(), 10) };
        assert_eq!(n, text.len(), "exactly-budget row must not clip its border");
        // One cell less genuinely truncates (… appended by the renderer).
        let n9 = unsafe { tui_truncate_cells(c.as_ptr(), 9) };
        assert!(n9 < text.len());
    }

    /// Wide (CJK/emoji) chars occupy 2 cells: budgets must count them
    /// correctly or rows overflow real terminals.
    #[test]
    fn truncate_counts_wide_chars_as_two() {
        let text = "素風素風"; // 4 chars × 2 cells = 8 cells
        let c = CString::new(text).unwrap();
        // 4 cells = 2 chars → byte offset 6.
        assert_eq!(unsafe { tui_truncate_cells(c.as_ptr(), 4) }, "素風".len());
        // 8 cells = all.
        assert_eq!(unsafe { tui_truncate_cells(c.as_ptr(), 8) }, text.len());
    }

    /// The footer status/metric statics (process.rs) are plain `Vec<u8>`
    /// WITHOUT a NUL terminator; the strlen-walking C-ABI entry used to read
    /// past the allocation and return an out-of-range index — the
    /// 2026-09-09 abort ("range end index 141 out of range for slice of
    /// length 80" at process.rs:1366) killed the CLI mid-session. The
    /// bounded entry must return an in-range bound for every budget and
    /// agree with the C-ABI entry on genuinely terminated input.
    #[test]
    fn truncate_cells_bounded_on_unterminated_buffers() {
        // The crash shape: an 80-byte metric line, budget far past it
        // (wide terminal), no NUL anywhere.
        let buf: Vec<u8> = b"ctx 12.3k/32.7k (38%) in 5.1k out 890 tk"
            .iter()
            .copied()
            .cycle()
            .take(80)
            .collect();
        for cells in 0..=200usize {
            let n = truncate_cells(&buf, cells);
            assert!(n <= buf.len(), "overrun: cells={cells} n={n}");
        }
        // Huge budget returns the full length.
        assert_eq!(truncate_cells(&buf, 9999), buf.len());
        // Empty slice.
        assert_eq!(truncate_cells(b"", 10), 0);
        // Overshoot regressions — the old body returned past n for these:
        // a trailing lone ESC ran the absorption loop to n+1 …
        assert_eq!(truncate_cells(b"ab\x1b", 10), 3);
        // … and a 4-byte lead byte claiming 4 when only 2 remain.
        assert_eq!(truncate_cells(b"a\xF0\x9F", 10), 3);
        // Equivalence with the C-ABI entry on NUL-terminated input: the
        // wrapper is a thin delegate, semantics must be identical.
        let s = "╭──╮ \x1b[2mdim\x1b[0m 素風";
        let c = CString::new(s).unwrap();
        for cells in 0..=(s.len() + 4) as usize {
            assert_eq!(
                unsafe { tui_truncate_cells(c.as_ptr(), cells as i32) },
                truncate_cells(s.as_bytes(), cells),
                "wrapper mismatch at cells={cells}"
            );
        }
    }

    /// Soft wrap: long prose breaks at spaces, never mid-word, every line
    /// within the budget (wide chars counted).
    #[test]
    fn wrap_breaks_at_word_boundaries() {
        let words: Vec<&str> = vec!["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"];
        let text = words.join(" "); // 44 chars
        let lines = wrap_row(&text, 15);
        assert!(lines.len() >= 3, "want wrapping, got {lines:?}");
        for l in &lines {
            assert!(l.chars().count() <= 15, "line over budget: {l:?}");
        }
        // No mid-word cuts: every line is a sequence of whole words.
        for l in &lines {
            for w in l.split(' ') {
                assert!(words.contains(&w), "mid-word cut produced {w:?} in {l:?}");
            }
        }
        // Content preserved (modulo the spaces consumed by breaks).
        let joined: String = lines.join(" ");
        assert_eq!(joined.replace(' ', ""), text.replace(' ', ""));
    }

    /// Unbreakable tokens (URLs, hashes) hard-break instead of overflowing.
    #[test]
    fn wrap_hard_breaks_long_tokens() {
        let text = "x ".to_string() + &"a".repeat(60);
        let lines = wrap_row(&text, 20);
        assert!(lines.len() >= 3, "{lines:?}");
        for l in &lines {
            assert!(l.chars().count() <= 20, "{l:?}");
        }
    }

    /// Regression (found by the 62-col pty capture): when a line-filling
    /// token arrives right AFTER a trailing space, that space is the last
    /// token of the current line and the break cut hit `cur[cut+1..]` with
    /// cut+1 == len — a panic. The tail must simply be empty.
    #[test]
    fn wrap_break_after_trailing_space() {
        // "aaaa … " fills past w exactly at a trailing space boundary.
        let text = format!("{} {}", "a".repeat(18), "b".repeat(18));
        let lines = wrap_row(&text, 20);
        assert!(!lines.is_empty());
        assert_eq!(lines[0], "a".repeat(18));
        assert_eq!(lines[1], "b".repeat(18));
        // Space exactly at the budget boundary with more text behind it.
        let text2 = format!("{} b c", "a".repeat(19));
        let lines2 = wrap_row(&text2, 20);
        for l in &lines2 {
            assert!(l.chars().count() <= 20, "{l:?}");
        }
    }

    /// ANSI escapes are zero-width: wrapped halves keep their styling and
    /// never split an escape across lines.
    #[test]
    fn wrap_preserves_escapes() {
        let text = format!("\x1b[1m{}\x1b[0m", "bold words ".repeat(8));
        let lines = wrap_row(&text, 20);
        assert!(lines.len() > 1);
        for l in &lines {
            assert!(!l.contains("\x1b[1\x1b"), "split escape: {l:?}");
            // every line must be valid stand-alone ANSI (starts styled or plain)
            assert!(!l.ends_with("\x1b[1m"), "dangling SGR start: {l:?}");
        }
    }

    /// Wide-char wrapping: CJK text must respect its 2-cell width.
    #[test]
    fn wrap_counts_wide_cells() {
        let text = "素風 ".repeat(12); // 12 × (2+1+1... "素風 " = 5 cells) 
        let lines = wrap_row(&text.trim(), 10);
        for l in &lines {
            let cells: usize = l.chars().map(super::char_cells).sum();
            assert!(cells <= 10, "over budget ({cells}): {l:?}");
        }
    }

    /// Short lines pass through untouched (single row, no reflow churn).
    #[test]
    fn wrap_passthrough_short_lines() {
        let text = "\x1b[2;35m│  short row\x1b[0m";
        assert_eq!(wrap_row(text, 100), vec![text.to_string()]);
        assert_eq!(wrap_row("", 100), vec![String::new()]);
    }
}
