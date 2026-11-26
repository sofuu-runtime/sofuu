// kaze.rs — the thin Rust door to `libsofuu_kaze.a`.
//
// Plan 14: the compiler and the bytecode VM are pure Kaze → C99. This module
// only marshals bytes and function pointers; it contains no lexer, no parser,
// no checker and no interpreter. `Compiler` owns one `kz_compiler_t`; `Vm`
// owns one `kz_vm_t`. Both are single-threaded because the engine uses a
// per-call arena swap, which is how it stays reentrant without global locks.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;
use std::sync::Mutex;

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedResult {
    ok: bool,
    text: *const c_char,
    len: usize,
    err_code: *const c_char,
    err_line: c_int,
}

#[repr(C)]
struct CompilerRaw {
    _private: [u8; 0],
}

#[repr(C)]
struct VmRaw {
    _private: [u8; 0],
}

extern "C" {
    fn kz_embed_compiler_new(arena_bytes: usize) -> *mut CompilerRaw;
    fn kz_embed_compiler_free(c: *mut CompilerRaw);
    fn kz_embed_check(c: *mut CompilerRaw, src: *const c_char, len: usize) -> EmbedResult;
    fn kz_embed_compile(c: *mut CompilerRaw, src: *const c_char, len: usize) -> EmbedResult;
    fn kz_embed_emit_c(
        c: *mut CompilerRaw,
        src: *const c_char,
        len: usize,
        standalone: bool,
    ) -> EmbedResult;
    fn kz_embed_to_machine(c: *mut CompilerRaw, src: *const c_char, len: usize) -> EmbedResult;
    fn kz_embed_to_human(c: *mut CompilerRaw, src: *const c_char, len: usize) -> EmbedResult;
    fn kz_embed_version(c: *mut CompilerRaw) -> EmbedResult;

    fn kz_vm_create(
        bc: *const u8,
        bc_len: usize,
        arena_bytes: usize,
        fuel_limit: u64,
    ) -> *mut VmRaw;
    fn kz_vm_destroy(vm: *mut VmRaw);
    fn kz_vm_set_host(vm: *mut VmRaw, cbs: *const HostCbs) -> c_int;
    fn kz_vm_set_heap_limit(vm: *mut VmRaw, heap_bytes: usize) -> c_int;

    fn kz_vm_item_count(vm: *const VmRaw) -> usize;
    fn kz_vm_item_name(vm: *const VmRaw, index: usize) -> *const c_char;
    fn kz_vm_item_tag(vm: *const VmRaw, index: usize) -> *const c_char;
    fn kz_vm_item_param_count(vm: *const VmRaw, index: usize) -> usize;
    fn kz_vm_item_param_name(vm: *const VmRaw, index: usize, param: usize) -> *const c_char;
    fn kz_vm_item_period_ms(vm: *const VmRaw, index: usize) -> f64;
    fn kz_vm_item_line(vm: *const VmRaw, index: usize) -> u32;
    fn kz_vm_needs_count(vm: *const VmRaw) -> usize;
    fn kz_vm_need(vm: *const VmRaw, index: usize) -> *const c_char;
    fn kz_vm_const_count(vm: *const VmRaw) -> usize;
    fn kz_vm_cap_count(vm: *const VmRaw) -> usize;
    fn kz_vm_code_bytes(vm: *const VmRaw) -> usize;

    fn kz_vm_start(vm: *mut VmRaw, item_index: usize, args_json: *const c_char) -> c_int;
    fn kz_vm_step_slice(
        vm: *mut VmRaw,
        max_steps: usize,
        req_family: *mut *const c_char,
        req_verb: *mut *const c_char,
        req_args_json: *mut *const c_char,
    ) -> c_int;
    fn kz_vm_supply_cap_result(
        vm: *mut VmRaw,
        result_json: *const c_char,
        err_msg: *const c_char,
    ) -> bool;
    fn kz_vm_finish(vm: *const VmRaw) -> EmbedResult;
    fn kz_vm_value_json(vm: *const VmRaw) -> *const c_char;
    fn kz_vm_error_code(vm: *const VmRaw) -> *const c_char;
    fn kz_vm_error_message(vm: *const VmRaw) -> *const c_char;
    fn kz_vm_error_line(vm: *const VmRaw) -> c_int;
    fn kz_vm_instructions(vm: *const VmRaw) -> u64;
    fn kz_vm_fuel_used(vm: *const VmRaw) -> u64;
    fn kz_vm_verdict_kind(vm: *const VmRaw) -> *const c_char;
    fn kz_vm_verdict_text(vm: *const VmRaw) -> *const c_char;
    fn kz_vm_value_text(vm: *const VmRaw) -> *const c_char;
    fn kz_vm_pending_line(vm: *const VmRaw) -> u32;
    fn kz_vm_pending_family(vm: *const VmRaw) -> *const c_char;
    fn kz_vm_pending_verb(vm: *const VmRaw) -> *const c_char;
    fn kz_vm_pending_args_json(vm: *const VmRaw) -> *const c_char;
    fn kz_vm_last_error() -> *const c_char;
}

pub const SLICE_DONE: c_int = 0;
pub const SLICE_YIELD: c_int = 1;
pub const SLICE_CAP_REQ: c_int = 2;
pub const SLICE_ERR: c_int = 3;

pub const CAP_OK: c_int = 0;
pub const CAP_ASYNC: c_int = 1;
pub const CAP_FAILED: c_int = 2;

/// The result shape shared by the compiler functions and `finish`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbedOutcome {
    pub ok: bool,
    pub text: String,
    pub err_code: Option<String>,
    pub err_line: i32,
}

fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

impl From<EmbedResult> for EmbedOutcome {
    fn from(r: EmbedResult) -> Self {
        let text = if r.text.is_null() {
            String::new()
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(r.text as *const u8, r.len) };
            String::from_utf8_lossy(bytes).into_owned()
        };
        let err_code = if r.err_code.is_null() {
            None
        } else {
            Some(cstr(r.err_code))
        };
        EmbedOutcome {
            ok: r.ok,
            text,
            err_code,
            err_line: r.err_line as i32,
        }
    }
}

/// A private compiler handle. Dropping it releases the handle's arena.
pub struct Compiler {
    raw: *mut CompilerRaw,
}

/// The generated compiler swaps file-scope scratch state around each call.
/// Handles are independent, but the swap itself is process-global, so calls
/// serialise here. The VM below has no such global and runs concurrently.
static COMPILER_LOCK: Mutex<()> = Mutex::new(());

impl Compiler {
    /// `arena_bytes == 0` selects the engine's 256 MiB default.
    pub fn new(arena_bytes: usize) -> Option<Self> {
        let raw = unsafe { kz_embed_compiler_new(arena_bytes) };
        if raw.is_null() {
            None
        } else {
            Some(Compiler { raw })
        }
    }

    fn call(&mut self, f: impl FnOnce(*mut CompilerRaw) -> EmbedResult) -> EmbedOutcome {
        let _guard = COMPILER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        f(self.raw).into()
    }

    pub fn check(&mut self, src: &str) -> EmbedOutcome {
        self.call(|c| unsafe { kz_embed_check(c, src.as_ptr() as *const c_char, src.len()) })
    }

    pub fn compile(&mut self, src: &str) -> EmbedOutcome {
        self.call(|c| unsafe { kz_embed_compile(c, src.as_ptr() as *const c_char, src.len()) })
    }

    pub fn emit_c(&mut self, src: &str, standalone: bool) -> EmbedOutcome {
        self.call(|c| unsafe {
            kz_embed_emit_c(c, src.as_ptr() as *const c_char, src.len(), standalone)
        })
    }

    pub fn to_machine(&mut self, src: &str) -> EmbedOutcome {
        self.call(|c| unsafe { kz_embed_to_machine(c, src.as_ptr() as *const c_char, src.len()) })
    }

    pub fn to_human(&mut self, src: &str) -> EmbedOutcome {
        self.call(|c| unsafe { kz_embed_to_human(c, src.as_ptr() as *const c_char, src.len()) })
    }

    pub fn version(&mut self) -> EmbedOutcome {
        self.call(|c| unsafe { kz_embed_version(c) })
    }
}

impl Drop for Compiler {
    fn drop(&mut self) {
        unsafe { kz_embed_compiler_free(self.raw) }
    }
}

/// One host capability call. The C VM copies the struct, so the boxed
/// callbacks in [`Vm`] must outlive the VM.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HostCbs {
    pub ctx: *mut c_void,
    pub cap_dispatch: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            family: *const c_char,
            verb: *const c_char,
            args_json: *const c_char,
            out_json: *mut c_char,
            out_cap: usize,
            out_len: *mut usize,
            out_err: *mut c_char,
            err_cap: usize,
        ) -> c_int,
    >,
    pub authorise: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            family: *const c_char,
            verb: *const c_char,
            args_json: *const c_char,
            out_err: *mut c_char,
            err_cap: usize,
        ) -> c_int,
    >,
    pub poll_deadline: Option<unsafe extern "C" fn(ctx: *mut c_void) -> c_int>,
    pub freshness_check: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            family: *const c_char,
            field: *const c_char,
        ) -> c_int,
    >,
    pub link_silent_ms: Option<unsafe extern "C" fn(ctx: *mut c_void, out_ms: *mut u64) -> c_int>,
}

impl Default for HostCbs {
    fn default() -> Self {
        HostCbs {
            ctx: ptr::null_mut(),
            cap_dispatch: None,
            authorise: None,
            poll_deadline: None,
            freshness_check: None,
            link_silent_ms: None,
        }
    }
}

/// Metadata for one bytecode item, read without running it.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemInfo {
    pub name: String,
    pub tag: String,
    pub params: Vec<String>,
    pub period_ms: f64,
    pub line: u32,
}

/// A resumable VM instance. `step` never blocks and never exits the process.
pub struct Vm {
    raw: *mut VmRaw,
}

impl Vm {
    pub fn new(bc: &[u8], arena_bytes: usize, fuel_limit: u64) -> Option<Self> {
        let raw = unsafe { kz_vm_create(bc.as_ptr(), bc.len(), arena_bytes, fuel_limit) };
        if raw.is_null() {
            None
        } else {
            Some(Vm { raw })
        }
    }

    pub fn last_error() -> String {
        cstr(unsafe { kz_vm_last_error() })
    }

    pub fn set_host(&mut self, cbs: &HostCbs) -> bool {
        unsafe { kz_vm_set_host(self.raw, cbs) != 0 }
    }

    pub fn set_heap_limit(&mut self, bytes: usize) -> bool {
        unsafe { kz_vm_set_heap_limit(self.raw, bytes) != 0 }
    }

    pub fn item_count(&self) -> usize {
        unsafe { kz_vm_item_count(self.raw) }
    }

    pub fn item_name(&self, index: usize) -> String {
        cstr(unsafe { kz_vm_item_name(self.raw, index) })
    }

    pub fn item_tag(&self, index: usize) -> String {
        cstr(unsafe { kz_vm_item_tag(self.raw, index) })
    }

    pub fn item_param_count(&self, index: usize) -> usize {
        unsafe { kz_vm_item_param_count(self.raw, index) }
    }

    pub fn item_param_name(&self, index: usize, param: usize) -> String {
        cstr(unsafe { kz_vm_item_param_name(self.raw, index, param) })
    }

    pub fn item_period_ms(&self, index: usize) -> f64 {
        unsafe { kz_vm_item_period_ms(self.raw, index) }
    }

    pub fn item_line(&self, index: usize) -> u32 {
        unsafe { kz_vm_item_line(self.raw, index) }
    }

    pub fn item(&self, index: usize) -> ItemInfo {
        ItemInfo {
            name: self.item_name(index),
            tag: self.item_tag(index),
            params: (0..self.item_param_count(index))
                .map(|i| self.item_param_name(index, i))
                .collect(),
            period_ms: self.item_period_ms(index),
            line: self.item_line(index),
        }
    }

    pub fn item_index(&self, name: &str) -> Option<usize> {
        (0..self.item_count()).find(|&i| self.item_name(i) == name)
    }

    pub fn needs(&self) -> Vec<String> {
        (0..unsafe { kz_vm_needs_count(self.raw) })
            .map(|i| cstr(unsafe { kz_vm_need(self.raw, i) }))
            .collect()
    }

    pub fn const_count(&self) -> usize {
        unsafe { kz_vm_const_count(self.raw) }
    }

    pub fn cap_count(&self) -> usize {
        unsafe { kz_vm_cap_count(self.raw) }
    }

    pub fn code_bytes(&self) -> usize {
        unsafe { kz_vm_code_bytes(self.raw) }
    }

    pub fn start(&mut self, item_index: usize, args_json: Option<&str>) -> bool {
        // The C ABI takes a NUL-terminated string; a Rust `&str` is not one.
        let args = match args_json.map(CString::new) {
            Some(Ok(text)) => text,
            Some(Err(_)) => return false,
            None => CString::default(),
        };
        let raw = if args_json.is_some() {
            args.as_ptr()
        } else {
            ptr::null()
        };
        unsafe { kz_vm_start(self.raw, item_index, raw) != 0 }
    }

    /// Step at most `max_steps` instructions. On `SLICE_CAP_REQ`, the three
    /// strings describe the pending call; use [`Self::supply`] to answer it.
    pub fn step(&mut self, max_steps: usize) -> (c_int, Option<(String, String, String)>) {
        let mut family: *const c_char = ptr::null();
        let mut verb: *const c_char = ptr::null();
        let mut args: *const c_char = ptr::null();
        let status =
            unsafe { kz_vm_step_slice(self.raw, max_steps, &mut family, &mut verb, &mut args) };
        if status == SLICE_CAP_REQ {
            (status, Some((cstr(family), cstr(verb), cstr(args))))
        } else {
            (status, None)
        }
    }

    pub fn pending(&self) -> (u32, String, String, String) {
        unsafe {
            (
                kz_vm_pending_line(self.raw),
                cstr(kz_vm_pending_family(self.raw)),
                cstr(kz_vm_pending_verb(self.raw)),
                cstr(kz_vm_pending_args_json(self.raw)),
            )
        }
    }

    pub fn supply(&mut self, result_json: Option<&str>, err_msg: Option<&str>) -> bool {
        // Both arguments cross a `const char *` ABI, so they need a NUL.
        let result = match result_json.map(CString::new) {
            Some(Ok(text)) => text,
            Some(Err(_)) => return false,
            None => CString::default(),
        };
        let err = match err_msg.map(CString::new) {
            Some(Ok(text)) => text,
            Some(Err(_)) => return false,
            None => CString::default(),
        };
        let result_ptr = if result_json.is_some() {
            result.as_ptr()
        } else {
            ptr::null()
        };
        let err_ptr = if err_msg.is_some() {
            err.as_ptr()
        } else {
            ptr::null()
        };
        unsafe { kz_vm_supply_cap_result(self.raw, result_ptr, err_ptr) }
    }

    pub fn finish(&self) -> EmbedOutcome {
        unsafe { kz_vm_finish(self.raw) }.into()
    }

    pub fn value_json(&self) -> String {
        cstr(unsafe { kz_vm_value_json(self.raw) })
    }

    pub fn value_text(&self) -> String {
        cstr(unsafe { kz_vm_value_text(self.raw) })
    }

    pub fn error_code(&self) -> String {
        cstr(unsafe { kz_vm_error_code(self.raw) })
    }

    pub fn error_message(&self) -> String {
        cstr(unsafe { kz_vm_error_message(self.raw) })
    }

    pub fn error_line(&self) -> u32 {
        unsafe { kz_vm_error_line(self.raw) as u32 }
    }

    pub fn instructions(&self) -> u64 {
        unsafe { kz_vm_instructions(self.raw) }
    }

    pub fn fuel_used(&self) -> u64 {
        unsafe { kz_vm_fuel_used(self.raw) }
    }

    pub fn verdict_kind(&self) -> Option<String> {
        let p = unsafe { kz_vm_verdict_kind(self.raw) };
        if p.is_null() {
            None
        } else {
            Some(cstr(p))
        }
    }

    pub fn verdict_text(&self) -> String {
        cstr(unsafe { kz_vm_verdict_text(self.raw) })
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        unsafe { kz_vm_destroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_reports_version_and_round_trips_a_program() {
        let mut c = Compiler::new(8 * 1024 * 1024).expect("compiler handle");
        assert_eq!(c.version().text, "1.0");
        let src = "action main():\n    return 1\n";
        let checked = c.check(src);
        assert!(checked.ok, "{}", checked.text);
        let compiled = c.compile(src);
        assert!(compiled.ok, "{}", compiled.text);
        assert!(
            compiled.text.starts_with("4b415a45424331"),
            "{}",
            compiled.text
        );
        let emitted = c.emit_c(src, true);
        assert!(emitted.ok, "{}", emitted.text);
        assert!(emitted.text.contains("int main("));
    }

    unsafe extern "C" fn pause_once(
        ctx: *mut c_void,
        family: *const c_char,
        verb: *const c_char,
        _args_json: *const c_char,
        out_json: *mut c_char,
        out_cap: usize,
        out_len: *mut usize,
        _out_err: *mut c_char,
        _err_cap: usize,
    ) -> c_int {
        assert_eq!(cstr(family), "memory");
        assert_eq!(cstr(verb), "save");
        let calls = &mut *(ctx as *mut u32);
        *calls += 1;
        if *calls == 1 {
            return CAP_ASYNC;
        }
        let ok = b"null\0";
        assert!(out_cap >= ok.len());
        ptr::copy_nonoverlapping(ok.as_ptr() as *const c_char, out_json, ok.len());
        if !out_len.is_null() {
            *out_len = ok.len() - 1;
        }
        CAP_OK
    }

    #[test]
    fn vm_pauses_for_async_capability_and_resumes_once() {
        let mut c = Compiler::new(8 * 1024 * 1024).expect("compiler handle");
        let src = "needs: memory\naction main():\n    memory.save(\"k\", \"v\")\n    return 1\n";
        let hex = c.compile(src);
        assert!(hex.ok, "{}", hex.text);
        let bytes: Vec<u8> = (0..hex.text.len() / 2)
            .map(|i| u8::from_str_radix(&hex.text[i * 2..i * 2 + 2], 16).unwrap())
            .collect();
        let mut vm = Vm::new(&bytes, 4 * 1024 * 1024, 200_000).expect("vm");
        let mut calls = 0u32;
        let cbs = HostCbs {
            ctx: &mut calls as *mut u32 as *mut c_void,
            cap_dispatch: Some(pause_once),
            ..HostCbs::default()
        };
        assert!(vm.set_host(&cbs));
        let main = vm.item_index("main").expect("main");
        assert!(vm.start(main, None));
        let mut paused = false;
        let mut status = SLICE_YIELD;
        for _ in 0..1000 {
            let (s, req) = vm.step(256);
            status = s;
            if s == SLICE_CAP_REQ {
                let (family, verb, _) = req.expect("pending capability");
                assert_eq!((family.as_str(), verb.as_str()), ("memory", "save"));
                assert!(
                    vm.supply(Some("null"), None),
                    "supply failed: {:?}",
                    vm.pending()
                );
                paused = true;
                continue;
            }
            if s != SLICE_YIELD {
                break;
            }
        }
        assert!(paused, "the host never saw a pause");
        assert_eq!(status, SLICE_DONE, "{}", vm.error_message());
        assert_eq!(vm.value_text(), "1");
    }

    #[test]
    fn vm_start_accepts_arguments_from_a_rust_string() {
        let mut c = Compiler::new(8 * 1024 * 1024).expect("compiler handle");
        let src = "action main(n):\n    return n + 1\n";
        let hex = c.compile(src);
        assert!(hex.ok, "{}", hex.text);
        let bytes: Vec<u8> = (0..hex.text.len() / 2)
            .map(|i| u8::from_str_radix(&hex.text[i * 2..i * 2 + 2], 16).unwrap())
            .collect();
        let mut vm = Vm::new(&bytes, 4 * 1024 * 1024, 200_000).expect("vm");
        let main = vm.item_index("main").expect("main");
        // `&str` is not NUL-terminated; this is the C-ABI boundary case.
        assert!(vm.start(main, Some("[41]")));
        let mut status = SLICE_YIELD;
        for _ in 0..1000 {
            let (s, _) = vm.step(256);
            status = s;
            if s != SLICE_YIELD {
                break;
            }
        }
        assert_eq!(status, SLICE_DONE, "{}", vm.error_message());
        assert_eq!(vm.value_text(), "42");
    }

    #[test]
    fn vm_runs_a_pure_module_without_a_host() {
        let mut c = Compiler::new(8 * 1024 * 1024).expect("compiler handle");
        let src = "action main():\n    return 7\n";
        let hex = c.compile(src);
        assert!(hex.ok, "{}", hex.text);
        let bytes: Vec<u8> = (0..hex.text.len() / 2)
            .map(|i| u8::from_str_radix(&hex.text[i * 2..i * 2 + 2], 16).unwrap())
            .collect();
        let mut vm = Vm::new(&bytes, 4 * 1024 * 1024, 200_000).expect("vm");
        let main = vm.item_index("main").expect("main");
        assert!(vm.start(main, None));
        let mut status = SLICE_YIELD;
        for _ in 0..1000 {
            let (s, _) = vm.step(256);
            status = s;
            if s != SLICE_YIELD {
                break;
            }
        }
        assert_eq!(status, SLICE_DONE, "{}", vm.error_message());
        assert_eq!(vm.value_text(), "7");
        assert_eq!(vm.finish().text, "7");
    }
}
